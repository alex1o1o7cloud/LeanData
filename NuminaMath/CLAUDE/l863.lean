import Mathlib

namespace tangent_line_x_ln_x_at_1_l863_86300

/-- The equation of the tangent line to y = x ln x at x = 1 is x - y - 1 = 0 -/
theorem tangent_line_x_ln_x_at_1 : 
  let f : ℝ → ℝ := λ x => x * Real.log x
  let tangent_line : ℝ → ℝ := λ x => x - 1
  (∀ x, x > 0 → HasDerivAt f (Real.log x + 1) x) ∧ 
  HasDerivAt f 1 1 ∧
  f 1 = 0 →
  ∀ x y, y = tangent_line x ↔ x - y - 1 = 0 :=
by sorry

end tangent_line_x_ln_x_at_1_l863_86300


namespace monotonic_f_implies_a_range_l863_86354

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- Define the property of being monotonic in an interval
def isMonotonicIn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

-- Theorem statement
theorem monotonic_f_implies_a_range (a : ℝ) :
  isMonotonicIn (f a) 1 2 → a ≤ -1 ∨ a ≥ 0 := by
  sorry

end monotonic_f_implies_a_range_l863_86354


namespace burn_time_3x5_grid_l863_86382

/-- Represents a grid of toothpicks -/
structure ToothpickGrid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the time taken for a toothpick grid to burn completely -/
def burnTime (grid : ToothpickGrid) (burnTimePerToothpick : ℕ) : ℕ :=
  sorry

/-- Theorem stating that a 3x5 grid burns in 65 seconds -/
theorem burn_time_3x5_grid :
  let grid := ToothpickGrid.mk 3 5
  let burnTimePerToothpick := 10
  burnTime grid burnTimePerToothpick = 65 :=
sorry

end burn_time_3x5_grid_l863_86382


namespace greatest_integer_less_than_negative_twenty_five_sixths_l863_86352

theorem greatest_integer_less_than_negative_twenty_five_sixths :
  Int.floor (-25 / 6 : ℚ) = -5 := by sorry

end greatest_integer_less_than_negative_twenty_five_sixths_l863_86352


namespace division_remainder_l863_86366

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 686) (h2 : divisor = 36) (h3 : quotient = 19) :
  dividend % divisor = 2 := by
sorry

end division_remainder_l863_86366


namespace lending_interest_rate_l863_86364

/-- The interest rate at which a person lends money, given specific borrowing and lending conditions -/
theorem lending_interest_rate (borrowed_amount : ℝ) (borrowing_rate : ℝ) (lending_years : ℝ) (yearly_gain : ℝ) : 
  borrowed_amount = 5000 →
  borrowing_rate = 4 →
  lending_years = 2 →
  yearly_gain = 200 →
  (borrowed_amount * borrowing_rate * lending_years / 100 + 2 * yearly_gain) / (borrowed_amount * lending_years / 100) = 8 := by
  sorry

end lending_interest_rate_l863_86364


namespace transformation_solvable_l863_86372

/-- A transformation that replaces two numbers with their product -/
def transformation (numbers : List ℝ) (i j : Nat) : List ℝ :=
  if i < numbers.length ∧ j < numbers.length ∧ i ≠ j then
    let product := numbers[i]! * numbers[j]!
    numbers.set i product |>.set j product
  else
    numbers

/-- Predicate to check if all numbers in the list are the same -/
def allSame (numbers : List ℝ) : Prop :=
  ∀ i j, i < numbers.length → j < numbers.length → numbers[i]! = numbers[j]!

/-- The main theorem stating when the problem is solvable -/
theorem transformation_solvable (n : ℕ) :
  (∃ (numbers : List ℝ) (k : ℕ), numbers.length = n ∧ 
   ∃ (transformations : List (ℕ × ℕ)), 
     allSame (transformations.foldl (λ acc (i, j) => transformation acc i j) numbers)) ↔ 
  (n % 2 = 0 ∨ n = 1) :=
sorry

end transformation_solvable_l863_86372


namespace event_ticket_revenue_l863_86334

theorem event_ticket_revenue :
  ∀ (full_price : ℚ) (full_count half_count : ℕ),
    full_count + half_count = 180 →
    full_price * full_count + (full_price / 2) * half_count = 2652 →
    full_price * full_count = 984 :=
by
  sorry

end event_ticket_revenue_l863_86334


namespace line_points_k_value_l863_86307

/-- 
Given two points (m, n) and (m + 2, n + k) on the line x = 2y + 5,
prove that k = 0.
-/
theorem line_points_k_value (m n k : ℝ) : 
  (m = 2*n + 5) → 
  (m + 2 = 2*(n + k) + 5) → 
  k = 0 := by
sorry

end line_points_k_value_l863_86307


namespace card_number_sum_l863_86313

theorem card_number_sum (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
  sorry

end card_number_sum_l863_86313


namespace adults_who_ate_proof_l863_86332

/-- Represents the number of adults who had their meal -/
def adults_who_ate : ℕ := sorry

/-- The total number of adults in the group -/
def total_adults : ℕ := 55

/-- The total number of children in the group -/
def total_children : ℕ := 70

/-- The meal capacity for adults -/
def meal_capacity_adults : ℕ := 70

/-- The meal capacity for children -/
def meal_capacity_children : ℕ := 90

/-- The number of children that can be fed with remaining food after some adults eat -/
def remaining_children_fed : ℕ := 72

theorem adults_who_ate_proof :
  adults_who_ate = 14 ∧
  adults_who_ate ≤ total_adults ∧
  (meal_capacity_adults - adults_who_ate) * meal_capacity_children / meal_capacity_adults = remaining_children_fed :=
sorry

end adults_who_ate_proof_l863_86332


namespace seven_number_sequence_average_l863_86306

theorem seven_number_sequence_average (a b c d e f g : ℝ) :
  (a + b + c + d) / 4 = 4 →
  (d + e + f + g) / 4 = 4 →
  d = 11 →
  (a + b + c + d + e + f + g) / 7 = 3 := by
sorry

end seven_number_sequence_average_l863_86306


namespace white_balls_count_l863_86303

theorem white_balls_count (yellow_balls : ℕ) (yellow_prob : ℚ) : 
  yellow_balls = 15 → yellow_prob = 3/4 → 
  ∃ (white_balls : ℕ), (yellow_balls : ℚ) / ((white_balls : ℚ) + yellow_balls) = yellow_prob ∧ white_balls = 5 := by
  sorry

end white_balls_count_l863_86303


namespace wifes_raise_is_760_l863_86336

/-- Calculates the raise amount for Don's wife given the conditions of the problem -/
def wifes_raise (dons_raise : ℚ) (income_difference : ℚ) (raise_percentage : ℚ) : ℚ :=
  let dons_income := dons_raise / raise_percentage
  let wifes_income := dons_income - (income_difference / (1 + raise_percentage))
  wifes_income * raise_percentage

/-- Proves that Don's wife's raise is 760 given the problem conditions -/
theorem wifes_raise_is_760 : 
  wifes_raise 800 540 (8/100) = 760 := by
  sorry

#eval wifes_raise 800 540 (8/100)

end wifes_raise_is_760_l863_86336


namespace unique_quadratic_trinomial_l863_86329

theorem unique_quadratic_trinomial :
  ∃! (a b c : ℝ), 
    (∀ x : ℝ, (a + 1) * x^2 + b * x + c = 0 → (∃! y : ℝ, y = x)) ∧
    (∀ x : ℝ, a * x^2 + (b + 1) * x + c = 0 → (∃! y : ℝ, y = x)) ∧
    (∀ x : ℝ, a * x^2 + b * x + (c + 1) = 0 → (∃! y : ℝ, y = x)) ∧
    a = 1/8 ∧ b = -3/4 ∧ c = 1/8 :=
by sorry

end unique_quadratic_trinomial_l863_86329


namespace toothfairy_money_is_11_90_l863_86381

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The number of each type of coin Joan received -/
def coin_count : ℕ := 14

/-- The total value of coins Joan received from the toothfairy -/
def toothfairy_money : ℚ := coin_count * quarter_value + coin_count * half_dollar_value + coin_count * dime_value

theorem toothfairy_money_is_11_90 : toothfairy_money = 11.90 := by
  sorry

end toothfairy_money_is_11_90_l863_86381


namespace smallest_tripling_period_l863_86351

-- Define the annual interest rate
def r : ℝ := 0.3334

-- Define the function that calculates the investment value after n years
def investment_value (n : ℕ) : ℝ := (1 + r) ^ n

-- Theorem statement
theorem smallest_tripling_period :
  ∀ n : ℕ, (investment_value n > 3 ∧ ∀ m : ℕ, m < n → investment_value m ≤ 3) → n = 4 :=
sorry

end smallest_tripling_period_l863_86351


namespace no_base_with_final_digit_one_l863_86386

theorem no_base_with_final_digit_one : 
  ∀ b : ℕ, 2 ≤ b ∧ b ≤ 9 → ¬(∃ k : ℕ, 360 = k * b + 1) := by
  sorry

end no_base_with_final_digit_one_l863_86386


namespace g_of_two_equals_fourteen_l863_86319

-- Define g as a function from ℝ to ℝ
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_of_two_equals_fourteen :
  (∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) →
  g 2 = 14 := by
  sorry

end g_of_two_equals_fourteen_l863_86319


namespace train_speed_l863_86312

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) :
  train_length = 900 →
  crossing_time = 53.99568034557235 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63.0036) < 0.0001 := by
  sorry

#check train_speed

end train_speed_l863_86312


namespace doughnuts_left_l863_86392

theorem doughnuts_left (total_doughnuts : ℕ) (staff_count : ℕ) (doughnuts_per_staff : ℕ) :
  total_doughnuts = 50 →
  staff_count = 19 →
  doughnuts_per_staff = 2 →
  total_doughnuts - (staff_count * doughnuts_per_staff) = 12 :=
by sorry

end doughnuts_left_l863_86392


namespace max_side_length_triangle_l863_86348

/-- A triangle with integer side lengths and perimeter 24 has maximum side length 11 -/
theorem max_side_length_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧ -- Three different side lengths
  a + b + c = 24 ∧ -- Perimeter is 24
  a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
  c ≤ 11 := by
sorry

end max_side_length_triangle_l863_86348


namespace garden_ants_approximation_l863_86326

/-- The number of ants in a rectangular garden --/
def number_of_ants (width_feet : ℝ) (length_feet : ℝ) (ants_per_sq_inch : ℝ) : ℝ :=
  width_feet * length_feet * (12 * 12) * ants_per_sq_inch

/-- Theorem stating that the number of ants in the garden is approximately 72 million --/
theorem garden_ants_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1000000 ∧ 
  abs (number_of_ants 200 500 5 - 72000000) < ε :=
sorry

end garden_ants_approximation_l863_86326


namespace square_area_with_four_circles_l863_86357

/-- The area of a square containing four touching circles -/
theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by
  sorry

end square_area_with_four_circles_l863_86357


namespace modified_fibonacci_series_sum_l863_86398

/-- Modified Fibonacci sequence -/
def G : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n + 2) => G (n + 1) + G n

/-- The sum of the series G_n / 5^n from n = 0 to infinity -/
noncomputable def series_sum : ℚ := ∑' n, G n / (5 : ℚ) ^ n

/-- Theorem stating that the sum of the series G_n / 5^n from n = 0 to infinity equals 50/19 -/
theorem modified_fibonacci_series_sum : series_sum = 50 / 19 := by sorry

end modified_fibonacci_series_sum_l863_86398


namespace jane_finishing_days_l863_86310

/-- The number of days Jane needs to finish arranging the remaining vases -/
def days_needed (jane_rate mark_rate mark_days total_vases : ℕ) : ℕ :=
  let combined_rate := jane_rate + mark_rate
  let vases_arranged := combined_rate * mark_days
  let remaining_vases := total_vases - vases_arranged
  (remaining_vases + jane_rate - 1) / jane_rate

theorem jane_finishing_days :
  days_needed 16 20 3 248 = 9 := by
  sorry

end jane_finishing_days_l863_86310


namespace projection_theorem_l863_86321

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the projection theorem
theorem projection_theorem (t : Triangle) : 
  t.a = t.b * Real.cos t.C + t.c * Real.cos t.B ∧
  t.b = t.c * Real.cos t.A + t.a * Real.cos t.C ∧
  t.c = t.a * Real.cos t.B + t.b * Real.cos t.A :=
by sorry

end projection_theorem_l863_86321


namespace proposition_1_proposition_3_l863_86344

-- Define the set S
def S (m l : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ l}

-- Define the property that for all x in S, x² is also in S
def square_closed (m l : ℝ) : Prop :=
  ∀ x, x ∈ S m l → x^2 ∈ S m l

-- Theorem 1: If m = 1, then S = {1}
theorem proposition_1 (l : ℝ) (h : square_closed 1 l) :
  S 1 l = {1} :=
sorry

-- Theorem 2: If m = -1/3, then l ∈ [1/9, 1]
theorem proposition_3 (l : ℝ) (h : square_closed (-1/3) l) :
  1/9 ≤ l ∧ l ≤ 1 :=
sorry

end proposition_1_proposition_3_l863_86344


namespace min_complex_value_is_zero_l863_86388

theorem min_complex_value_is_zero 
  (a b c : ℤ) 
  (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_omega_power : ω^4 = 1) 
  (h_omega_not_one : ω ≠ 1) :
  ∃ (a' b' c' : ℤ) (h_distinct' : a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c'),
    Complex.abs (a' + b' * ω + c' * ω^3) = 0 ∧
    ∀ (x y z : ℤ) (h_xyz_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z),
      Complex.abs (x + y * ω + z * ω^3) ≥ 0 :=
by sorry

end min_complex_value_is_zero_l863_86388


namespace range_of_a_l863_86346

theorem range_of_a (x a : ℝ) : 
  (∀ x, (a ≤ x ∧ x < a + 2) → (|x| ≠ 1)) ∧ 
  (∃ x, |x| ≠ 1 ∧ ¬(a ≤ x ∧ x < a + 2)) →
  a ∈ Set.Iic (-3) ∪ Set.Ioi 1 :=
sorry

end range_of_a_l863_86346


namespace max_consoles_assembled_l863_86327

def chipA : ℕ := 467
def chipB : ℕ := 413
def chipC : ℕ := 532
def chipD : ℕ := 356
def chipE : ℕ := 494

def dailyProduction : List ℕ := [chipA, chipB, chipC, chipD, chipE]

theorem max_consoles_assembled (consoles : ℕ) :
  consoles = List.minimum dailyProduction → consoles = 356 := by
  sorry

end max_consoles_assembled_l863_86327


namespace last_digit_of_difference_l863_86390

/-- A function that returns the last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if a number is a power of 10 -/
def isPowerOfTen (n : ℕ) : Prop := ∃ k : ℕ, n = 10^k

theorem last_digit_of_difference (p q : ℕ) 
  (hp : p > 0) (hq : q > 0) 
  (hpq : p > q)
  (hpLast : lastDigit p ≠ 0) 
  (hqLast : lastDigit q ≠ 0)
  (hProduct : isPowerOfTen (p * q)) : 
  lastDigit (p - q) ≠ 5 := by
sorry

end last_digit_of_difference_l863_86390


namespace fraction_equality_implies_sum_l863_86362

theorem fraction_equality_implies_sum (C D : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
    (D * x - 17) / (x^2 - 8*x + 15) = C / (x - 3) + 2 / (x - 5)) →
  C + D = 32/5 := by
sorry

end fraction_equality_implies_sum_l863_86362


namespace average_price_per_book_l863_86360

theorem average_price_per_book (books_shop1 : ℕ) (cost_shop1 : ℕ) (books_shop2 : ℕ) (cost_shop2 : ℕ) 
  (h1 : books_shop1 = 42)
  (h2 : cost_shop1 = 520)
  (h3 : books_shop2 = 22)
  (h4 : cost_shop2 = 248) :
  (cost_shop1 + cost_shop2) / (books_shop1 + books_shop2) = 12 := by
  sorry

end average_price_per_book_l863_86360


namespace smallest_value_l863_86385

theorem smallest_value (A B C D : ℝ) : 
  A = Real.sin (50 * π / 180) * Real.cos (39 * π / 180) - Real.sin (40 * π / 180) * Real.cos (51 * π / 180) →
  B = -2 * Real.sin (40 * π / 180)^2 + 1 →
  C = 2 * Real.sin (6 * π / 180) * Real.cos (6 * π / 180) →
  D = Real.sqrt 3 / 2 * Real.sin (43 * π / 180) - 1 / 2 * Real.cos (43 * π / 180) →
  B < A ∧ B < C ∧ B < D :=
by sorry


end smallest_value_l863_86385


namespace lindas_savings_l863_86394

theorem lindas_savings (savings : ℝ) : 
  (3 / 4 : ℝ) * savings + 210 = savings → savings = 840 :=
by
  sorry

end lindas_savings_l863_86394


namespace matthews_water_glass_size_l863_86378

/-- Given Matthew's water drinking habits, prove the number of ounces in each glass. -/
theorem matthews_water_glass_size 
  (glasses_per_day : ℕ) 
  (bottle_size : ℕ) 
  (fills_per_week : ℕ) 
  (h1 : glasses_per_day = 4)
  (h2 : bottle_size = 35)
  (h3 : fills_per_week = 4) :
  (bottle_size * fills_per_week) / (glasses_per_day * 7) = 5 := by
  sorry

#check matthews_water_glass_size

end matthews_water_glass_size_l863_86378


namespace parabola_point_relationship_l863_86396

/-- Parabola function -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Theorem stating the relationship between y-coordinates of three points on the parabola -/
theorem parabola_point_relationship : 
  let y₁ := f (-5)
  let y₂ := f 1
  let y₃ := f 12
  y₂ < y₁ ∧ y₁ < y₃ := by sorry

end parabola_point_relationship_l863_86396


namespace nested_fraction_equality_l863_86338

theorem nested_fraction_equality : 
  (1 / (1 + 1 / (2 + 1 / 5))) = 11 / 16 := by
  sorry

end nested_fraction_equality_l863_86338


namespace fourth_root_over_seventh_root_of_seven_l863_86337

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (hx : x = 7) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
sorry

end fourth_root_over_seventh_root_of_seven_l863_86337


namespace difference_of_squares_l863_86317

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 72) (h2 : a - b = 16) : a^2 - b^2 = 1152 := by
  sorry

end difference_of_squares_l863_86317


namespace janet_key_search_time_l863_86323

/-- The number of minutes Janet spends looking for her keys every day -/
def key_search_time : ℝ := 8

/-- The number of minutes Janet spends complaining after finding her keys -/
def complain_time : ℝ := 3

/-- The number of days in a week -/
def days_per_week : ℝ := 7

/-- The number of minutes Janet would save per week if she stops losing her keys -/
def time_saved_per_week : ℝ := 77

theorem janet_key_search_time :
  key_search_time = (time_saved_per_week - days_per_week * complain_time) / days_per_week :=
by sorry

end janet_key_search_time_l863_86323


namespace geometric_arithmetic_sequence_l863_86322

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (4 * a 1 - a 3 = a 3 - 2 * a 2) →  -- arithmetic sequence condition
  q = -1 ∨ q = 2 := by
sorry

end geometric_arithmetic_sequence_l863_86322


namespace squirrel_calories_proof_l863_86380

/-- The number of squirrels Brandon can catch in 1 hour -/
def squirrels_per_hour : ℕ := 6

/-- The number of rabbits Brandon can catch in 1 hour -/
def rabbits_per_hour : ℕ := 2

/-- The number of calories in each rabbit -/
def calories_per_rabbit : ℕ := 800

/-- The additional calories Brandon gets from catching squirrels instead of rabbits in 1 hour -/
def additional_calories : ℕ := 200

/-- The number of calories in each squirrel -/
def calories_per_squirrel : ℕ := 300

theorem squirrel_calories_proof :
  squirrels_per_hour * calories_per_squirrel = 
  rabbits_per_hour * calories_per_rabbit + additional_calories :=
by sorry

end squirrel_calories_proof_l863_86380


namespace tourist_attraction_temperature_difference_l863_86347

/-- The temperature difference between the highest and lowest temperatures -/
def temperature_difference (highest lowest : ℝ) : ℝ :=
  highest - lowest

/-- Proof that the temperature difference is 10°C given the highest and lowest temperatures -/
theorem tourist_attraction_temperature_difference :
  temperature_difference 8 (-2) = 10 := by
  sorry

end tourist_attraction_temperature_difference_l863_86347


namespace tangent_line_equation_l863_86367

/-- The equation of the line passing through the points of tangency of the tangents drawn from a point to a circle. -/
theorem tangent_line_equation (P : ℝ × ℝ) (c : ℝ × ℝ → Prop) :
  P = (2, 1) →
  (∀ x y, c (x, y) ↔ x^2 + y^2 = 4) →
  ∃ A B : ℝ × ℝ,
    (c A ∧ c B) ∧
    (∀ t : ℝ, c ((1 - t) * A.1 + t * B.1, (1 - t) * A.2 + t * B.2) → t = 0 ∨ t = 1) ∧
    (∀ x y, 2 * x + y - 4 = 0 ↔ ∃ t : ℝ, x = (1 - t) * A.1 + t * B.1 ∧ y = (1 - t) * A.2 + t * B.2) :=
by sorry

end tangent_line_equation_l863_86367


namespace yellow_parrots_count_l863_86339

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) (h1 : total = 108) (h2 : red_fraction = 5/6) :
  total * (1 - red_fraction) = 18 := by
  sorry

end yellow_parrots_count_l863_86339


namespace pumpkin_spiderweb_ratio_l863_86301

/-- Represents the Halloween decorations problem --/
def halloween_decorations (total : ℕ) (skulls : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) 
  (cauldron : ℕ) (budget : ℕ) (left_to_put : ℕ) : Prop :=
  ∃ (pumpkins : ℕ),
    total = skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget + left_to_put ∧
    pumpkins = 2 * spiderwebs

/-- The ratio of pumpkins to spiderwebs is 2:1 given the specified conditions --/
theorem pumpkin_spiderweb_ratio :
  halloween_decorations 83 12 4 12 1 20 10 := by
  sorry

end pumpkin_spiderweb_ratio_l863_86301


namespace squared_roots_equation_l863_86335

/-- Given a quadratic equation x^2 + px + q = 0, this theorem proves that
    the equation x^2 - (p^2 - 2q)x + q^2 = 0 has roots that are the squares
    of the roots of the original equation. -/
theorem squared_roots_equation (p q : ℝ) :
  let original_eq (x : ℝ) := x^2 + p*x + q
  let new_eq (x : ℝ) := x^2 - (p^2 - 2*q)*x + q^2
  ∀ (r : ℝ), original_eq r = 0 → new_eq (r^2) = 0 :=
by sorry

end squared_roots_equation_l863_86335


namespace dataset_manipulation_result_l863_86302

def calculate_final_dataset_size (initial_size : ℕ) : ℕ :=
  let size_after_increase := initial_size + (initial_size * 15 / 100)
  let size_after_addition := size_after_increase + 40
  let size_after_removal := size_after_addition - (size_after_addition / 6)
  let final_size := size_after_removal - (size_after_removal * 10 / 100)
  final_size

theorem dataset_manipulation_result :
  calculate_final_dataset_size 300 = 289 := by
  sorry

end dataset_manipulation_result_l863_86302


namespace anthony_pencils_l863_86369

theorem anthony_pencils (x : ℕ) : x + 56 = 65 → x = 9 := by
  sorry

end anthony_pencils_l863_86369


namespace fourth_sample_seat_number_l863_86376

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_sample : ℕ

/-- Calculates the nth element in a systematic sample -/
def nth_sample (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_sample + (n - 1) * (s.population_size / s.sample_size)

theorem fourth_sample_seat_number 
  (sample : SystematicSample)
  (h1 : sample.population_size = 56)
  (h2 : sample.sample_size = 4)
  (h3 : sample.first_sample = 4)
  (h4 : nth_sample sample 2 = 18)
  (h5 : nth_sample sample 3 = 46) :
  nth_sample sample 4 = 32 := by
  sorry

end fourth_sample_seat_number_l863_86376


namespace vector_perpendicular_condition_l863_86399

/-- Given two vectors a and b in R², where a = (1, m) and b = (3, -2),
    if a + b is perpendicular to b, then m = 8. -/
theorem vector_perpendicular_condition (m : ℝ) :
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (3, -2)
  (a.1 + b.1, a.2 + b.2) • b = 0 → m = 8 := by
  sorry

end vector_perpendicular_condition_l863_86399


namespace polygon_sides_count_l863_86324

/-- Given a polygon where the sum of its interior angles is 180° less than three times
    the sum of its exterior angles, prove that it has 5 sides. -/
theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 - 180 →
  n = 5 := by
  sorry

end polygon_sides_count_l863_86324


namespace rationalize_denominator_l863_86368

theorem rationalize_denominator : 
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 81 (1/3)) = (1/4 : ℝ) := by sorry

end rationalize_denominator_l863_86368


namespace hyperbrick_probability_l863_86365

open Set
open Real
open Finset

-- Define the set of numbers
def S : Finset ℕ := Finset.range 500

-- Define the type for our 9 randomly selected numbers
structure NineNumbers :=
  (numbers : Finset ℕ)
  (size_eq : numbers.card = 9)
  (subset_S : numbers ⊆ S)

-- Define the probability function
def probability (n : NineNumbers) : ℚ :=
  -- Implementation details omitted
  sorry

-- The main theorem
theorem hyperbrick_probability :
  ∀ n : NineNumbers, probability n = 16 / 63 :=
sorry

end hyperbrick_probability_l863_86365


namespace base_conversion_sum_equality_l863_86314

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

-- Define the fractions in their respective bases
def fraction1_numerator : List Nat := [4, 6, 2]
def fraction1_denominator : List Nat := [2, 1]
def fraction2_numerator : List Nat := [4, 4, 1]
def fraction2_denominator : List Nat := [3, 3]

-- Define the bases
def base1 : Nat := 8
def base2 : Nat := 4
def base3 : Nat := 5

-- State the theorem
theorem base_conversion_sum_equality :
  (baseToDecimal fraction1_numerator base1 / baseToDecimal fraction1_denominator base2 : ℚ) +
  (baseToDecimal fraction2_numerator base3 / baseToDecimal fraction2_denominator base2 : ℚ) =
  499 / 15 := by sorry

end base_conversion_sum_equality_l863_86314


namespace proposition_truth_l863_86363

theorem proposition_truth : (∀ x ∈ Set.Ioo 0 (Real.pi / 2), Real.sin x - x < 0) ∧
  ¬(∃ x₀ ∈ Set.Ioi 0, (2 : ℝ) ^ x₀ = 1 / 2) := by
  sorry

end proposition_truth_l863_86363


namespace complement_intersection_theorem_l863_86389

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {0, 1}

-- Define set N
def N : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {2, 3} := by sorry

end complement_intersection_theorem_l863_86389


namespace not_perfect_squares_l863_86349

theorem not_perfect_squares : 
  (∃ x : ℝ, (6 : ℝ)^3032 = x^2) ∧ 
  (∀ x : ℝ, (7 : ℝ)^3033 ≠ x^2) ∧ 
  (∃ x : ℝ, (8 : ℝ)^3034 = x^2) ∧ 
  (∀ x : ℝ, (9 : ℝ)^3035 ≠ x^2) ∧ 
  (∃ x : ℝ, (10 : ℝ)^3036 = x^2) := by
  sorry

end not_perfect_squares_l863_86349


namespace max_value_constraint_l863_86395

theorem max_value_constraint (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) :
  a*b + b*c + c*d + d*a + a*c + 4*b*d ≤ 5/2 := by
sorry

end max_value_constraint_l863_86395


namespace absolute_value_equality_l863_86355

theorem absolute_value_equality (a b : ℝ) : |a| = |b| → a = b ∨ a = -b := by
  sorry

end absolute_value_equality_l863_86355


namespace instrument_players_l863_86304

theorem instrument_players (total_people : ℕ) 
  (fraction_at_least_one : ℚ) 
  (prob_exactly_one : ℚ) 
  (h1 : total_people = 800) 
  (h2 : fraction_at_least_one = 2/5) 
  (h3 : prob_exactly_one = 28/100) : 
  ℕ := by
  sorry

#check instrument_players

end instrument_players_l863_86304


namespace focus_of_hyperbola_l863_86343

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 / 3 - x^2 / 6 = 1

/-- Definition of a focus for this hyperbola -/
def is_focus (x y : ℝ) : Prop :=
  x^2 + y^2 = (3 : ℝ)^2 ∧ hyperbola x y

/-- Theorem: (0, 3) is a focus of the given hyperbola -/
theorem focus_of_hyperbola : is_focus 0 3 := by sorry

end focus_of_hyperbola_l863_86343


namespace one_not_in_set_l863_86356

theorem one_not_in_set : 1 ∉ {x : ℝ | ∃ a : ℕ+, x = -a^2 + 1} := by sorry

end one_not_in_set_l863_86356


namespace simplify_expression_1_simplify_expression_2_l863_86331

variable (a b : ℝ)
variable (x y : ℝ)

theorem simplify_expression_1 : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 := by
  sorry

theorem simplify_expression_2 : 3 * (x^2 * y - 1/2 * x * y^2) - 1/2 * (4 * x^2 * y - 3 * x * y^2) = x^2 * y := by
  sorry

end simplify_expression_1_simplify_expression_2_l863_86331


namespace sum_fourth_fifth_sixth_l863_86345

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_second_third : a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem sum_fourth_fifth_sixth (seq : ArithmeticSequence) :
  seq.a 4 + seq.a 5 + seq.a 6 = 42 :=
sorry

end sum_fourth_fifth_sixth_l863_86345


namespace winter_temperature_uses_negative_numbers_specific_winter_day_uses_negative_numbers_l863_86371

-- Define a temperature range
structure TemperatureRange where
  min : ℤ
  max : ℤ
  h : min ≤ max

-- Define a predicate for a scenario using negative numbers
def usesNegativeNumbers (range : TemperatureRange) : Prop :=
  range.min < 0

-- Theorem: The given temperature range uses negative numbers
theorem winter_temperature_uses_negative_numbers :
  ∃ (range : TemperatureRange), usesNegativeNumbers range :=
by
  -- The proof would go here
  sorry

-- Example of the temperature range mentioned in the solution
def winter_day_range : TemperatureRange :=
  { min := -2
  , max := 5
  , h := by norm_num }

-- Theorem: The specific winter day range uses negative numbers
theorem specific_winter_day_uses_negative_numbers :
  usesNegativeNumbers winter_day_range :=
by
  -- The proof would go here
  sorry

end winter_temperature_uses_negative_numbers_specific_winter_day_uses_negative_numbers_l863_86371


namespace bryan_bookshelves_l863_86330

/-- The number of books on each bookshelf -/
def books_per_shelf : ℕ := 23

/-- The number of magazines on each bookshelf -/
def magazines_per_shelf : ℕ := 61

/-- The total number of books and magazines -/
def total_items : ℕ := 2436

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 29

theorem bryan_bookshelves :
  (books_per_shelf + magazines_per_shelf) * num_bookshelves = total_items :=
sorry

end bryan_bookshelves_l863_86330


namespace system_solution_l863_86308

theorem system_solution (x y : ℝ) : 
  x^2 = 4*y^2 + 19 ∧ x*y + 2*y^2 = 18 → 
  (x = 55 / Real.sqrt 91 ∨ x = -55 / Real.sqrt 91) ∧
  (y = 18 / Real.sqrt 91 ∨ y = -18 / Real.sqrt 91) :=
by sorry

end system_solution_l863_86308


namespace photo_lineup_arrangements_l863_86361

def students : ℕ := 4
def teachers : ℕ := 3

def arrangements_teachers_together : ℕ := 720
def arrangements_teachers_together_students_split : ℕ := 144
def arrangements_teachers_apart : ℕ := 1440

theorem photo_lineup_arrangements :
  (students = 4 ∧ teachers = 3) →
  (arrangements_teachers_together = 720 ∧
   arrangements_teachers_together_students_split = 144 ∧
   arrangements_teachers_apart = 1440) := by
  sorry

end photo_lineup_arrangements_l863_86361


namespace survey_respondents_l863_86377

/-- Represents the number of people preferring each brand in a survey. -/
structure BrandPreference where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of respondents in a survey. -/
def totalRespondents (bp : BrandPreference) : ℕ :=
  bp.x + bp.y + bp.z

/-- Theorem stating that given the conditions of the survey, 
    the total number of respondents is 350. -/
theorem survey_respondents : 
  ∀ (bp : BrandPreference), 
    bp.x = 200 → 
    4 * bp.z = bp.x → 
    2 * bp.z = bp.y → 
    totalRespondents bp = 350 := by
  sorry

end survey_respondents_l863_86377


namespace ants_in_field_approx_50_million_l863_86333

/-- Represents the dimensions of a rectangular field in feet -/
structure FieldDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the area of a field in square inches -/
def fieldAreaInSquareInches (d : FieldDimensions) : ℝ :=
  d.width * d.length * 144  -- 144 = 12^2, converting square feet to square inches

/-- Calculates the total number of ants in a field -/
def totalAnts (d : FieldDimensions) (antsPerSquareInch : ℝ) : ℝ :=
  fieldAreaInSquareInches d * antsPerSquareInch

/-- Theorem stating that the number of ants in the given field is approximately 50 million -/
theorem ants_in_field_approx_50_million :
  let d : FieldDimensions := { width := 300, length := 400 }
  let antsPerSquareInch : ℝ := 3
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1000000 ∧ 
    abs (totalAnts d antsPerSquareInch - 50000000) < ε := by
  sorry

end ants_in_field_approx_50_million_l863_86333


namespace bug_probability_l863_86341

/-- Probability of the bug being at vertex A after n meters -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/3 * (1 - P n)

/-- The probability of the bug being at vertex A after 8 meters is 1823/6561 -/
theorem bug_probability : P 8 = 1823 / 6561 := by
  sorry

end bug_probability_l863_86341


namespace ian_painted_faces_l863_86397

/-- The number of faces of a cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids Ian painted -/
def number_of_cuboids : ℕ := 8

/-- The total number of faces painted by Ian -/
def total_faces_painted : ℕ := faces_per_cuboid * number_of_cuboids

theorem ian_painted_faces :
  total_faces_painted = 48 :=
by sorry

end ian_painted_faces_l863_86397


namespace p_plus_q_equals_30_l863_86370

theorem p_plus_q_equals_30 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 20 * x + 35) / (x - 3)) →
  P + Q = 30 := by
sorry

end p_plus_q_equals_30_l863_86370


namespace three_power_plus_five_power_plus_fourteen_equals_factorial_l863_86311

theorem three_power_plus_five_power_plus_fourteen_equals_factorial :
  ∀ x y z : ℕ, 3^x + 5^y + 14 = z! ↔ (x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6) := by
  sorry

end three_power_plus_five_power_plus_fourteen_equals_factorial_l863_86311


namespace pretzel_ratio_l863_86359

/-- The number of pretzels bought by Angie -/
def angie_pretzels : ℕ := 18

/-- The number of pretzels bought by Barry -/
def barry_pretzels : ℕ := 12

/-- The number of pretzels bought by Shelly -/
def shelly_pretzels : ℕ := angie_pretzels / 3

/-- Theorem stating the ratio of pretzels Shelly bought to pretzels Barry bought -/
theorem pretzel_ratio : 
  (shelly_pretzels : ℚ) / barry_pretzels = 1 / 2 := by
  sorry

end pretzel_ratio_l863_86359


namespace op_twice_equals_identity_l863_86305

-- Define the operation ⊕
def op (x y : ℝ) : ℝ := x^3 - y

-- Statement to prove
theorem op_twice_equals_identity (h : ℝ) : op h (op h h) = h := by
  sorry

end op_twice_equals_identity_l863_86305


namespace dice_roll_sum_l863_86358

theorem dice_roll_sum (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 216 →
  a + b + c + d ≠ 18 := by
sorry

end dice_roll_sum_l863_86358


namespace sum_of_digits_of_square_not_five_l863_86387

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of a perfect square cannot be 5 -/
theorem sum_of_digits_of_square_not_five (n : ℕ) : 
  ∃ m : ℕ, n = m^2 → sumOfDigits n ≠ 5 := by sorry

end sum_of_digits_of_square_not_five_l863_86387


namespace corrected_mean_problem_l863_86316

/-- Calculates the corrected mean of a set of observations after fixing an error -/
def corrected_mean (n : ℕ) (initial_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * initial_mean + correct_value - wrong_value) / n

/-- Theorem stating that the corrected mean is 36.14 given the problem conditions -/
theorem corrected_mean_problem :
  let n : ℕ := 50
  let initial_mean : ℚ := 36
  let wrong_value : ℚ := 23
  let correct_value : ℚ := 30
  corrected_mean n initial_mean wrong_value correct_value = 36.14 := by
sorry

#eval corrected_mean 50 36 23 30

end corrected_mean_problem_l863_86316


namespace triangle_increase_l863_86342

theorem triangle_increase (AB BC : ℝ) (h1 : AB = 24) (h2 : BC = 10) :
  let AC := Real.sqrt (AB^2 + BC^2)
  let AB' := AB + 6
  let BC' := BC + 6
  let AC' := Real.sqrt (AB'^2 + BC'^2)
  AC' - AC = 8 := by sorry

end triangle_increase_l863_86342


namespace candy_distribution_l863_86315

theorem candy_distribution (total_candy : ℕ) (candy_per_friend : ℕ) (h1 : total_candy = 45) (h2 : candy_per_friend = 5) :
  total_candy / candy_per_friend = 9 :=
by sorry

end candy_distribution_l863_86315


namespace pencil_average_cost_l863_86340

/-- The average cost per pencil, including shipping -/
def average_cost (num_pencils : ℕ) (pencil_cost shipping_cost : ℚ) : ℚ :=
  (pencil_cost + shipping_cost) / num_pencils

/-- Theorem stating the average cost per pencil for the given problem -/
theorem pencil_average_cost :
  average_cost 150 (24.75 : ℚ) (8.50 : ℚ) = (33.25 : ℚ) / 150 :=
by
  sorry

end pencil_average_cost_l863_86340


namespace no_upper_bound_for_y_l863_86375

-- Define the equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - (y - 3)^2 / 9 = 1

-- Theorem stating that there is no upper bound for y
theorem no_upper_bound_for_y :
  ∀ M : ℝ, ∃ x y : ℝ, hyperbola_equation x y ∧ y > M :=
by sorry

end no_upper_bound_for_y_l863_86375


namespace marks_lost_per_wrong_answer_l863_86391

/-- Prove that the number of marks lost for each wrong answer is 1 -/
theorem marks_lost_per_wrong_answer
  (total_questions : ℕ)
  (marks_per_correct : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (h1 : total_questions = 60)
  (h2 : marks_per_correct = 4)
  (h3 : total_marks = 130)
  (h4 : correct_answers = 38) :
  ∃ (marks_lost : ℕ), 
    marks_lost = 1 ∧ 
    total_marks = correct_answers * marks_per_correct - (total_questions - correct_answers) * marks_lost :=
by
  sorry

end marks_lost_per_wrong_answer_l863_86391


namespace hyperbola_k_range_l863_86384

/-- A hyperbola with equation x^2 + (k-1)y^2 = k+1 and foci on the x-axis -/
structure Hyperbola (k : ℝ) where
  eq : ∀ (x y : ℝ), x^2 + (k-1)*y^2 = k+1
  foci_on_x : True  -- This is a placeholder for the foci condition

/-- The range of k values for which the hyperbola is well-defined -/
def valid_k_range : Set ℝ := {k | ∃ h : Hyperbola k, True}

theorem hyperbola_k_range :
  valid_k_range = Set.Ioo (-1 : ℝ) 1 := by sorry

end hyperbola_k_range_l863_86384


namespace second_number_calculation_l863_86373

theorem second_number_calculation (A B : ℝ) (h1 : A = 456) (h2 : 0.5 * A = 0.4 * B + 180) : B = 120 := by
  sorry

end second_number_calculation_l863_86373


namespace cost_price_determination_l863_86309

theorem cost_price_determination (selling_price_profit selling_price_loss : ℝ) 
  (h : selling_price_profit = 54 ∧ selling_price_loss = 40) :
  ∃ cost_price : ℝ, 
    selling_price_profit - cost_price = cost_price - selling_price_loss ∧ 
    cost_price = 47 := by
  sorry

end cost_price_determination_l863_86309


namespace reflected_arcs_area_l863_86328

/-- The area of the region bounded by 8 reflected arcs in a regular octagon inscribed in a circle -/
theorem reflected_arcs_area (s : ℝ) (h : s = 2) : 
  let r := Real.sqrt (2 * Real.sqrt 2)
  let sector_area := π * r^2 / 8
  let triangle_area := s^2 / 4
  let reflected_arc_area := sector_area - triangle_area
  8 * reflected_arc_area = 2 * π * Real.sqrt 2 - 8 := by
  sorry

end reflected_arcs_area_l863_86328


namespace quadratic_root_implies_m_l863_86350

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 6 = 0 ∧ x = 1) → m = 5 := by
sorry

end quadratic_root_implies_m_l863_86350


namespace product_equality_l863_86353

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Proves that if a * (reversed b) = 143, then a * b = 143 -/
theorem product_equality (a b : ℕ) 
  (ha : 100 ≤ a ∧ a < 1000) 
  (hb : 10 ≤ b ∧ b < 100) 
  (h : a * (reverse_digits b) = 143) : 
  a * b = 143 := by
  sorry

end product_equality_l863_86353


namespace simplify_expression_l863_86393

theorem simplify_expression :
  1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1)) =
  ((Real.sqrt 3 - 2 * Real.sqrt 5 - 1) * (-16 - 2 * Real.sqrt 3)) / 244 := by
  sorry

end simplify_expression_l863_86393


namespace line_plane_perpendicularity_l863_86379

-- Define the types for lines and planes
def Line : Type := ℝ × ℝ × ℝ → Prop
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 : Plane) (p2 : Plane) : Prop := sorry

-- State the theorem
theorem line_plane_perpendicularity (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → perpendicular_planes α β := by sorry

end line_plane_perpendicularity_l863_86379


namespace irrational_pair_sum_six_l863_86374

theorem irrational_pair_sum_six : ∃ (x y : ℝ), Irrational x ∧ Irrational y ∧ x + y = 6 := by
  sorry

end irrational_pair_sum_six_l863_86374


namespace equilateral_triangle_perimeter_area_ratio_l863_86318

/-- The ratio of the perimeter to the area of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_perimeter_area_ratio :
  let side_length : ℝ := 6
  let perimeter : ℝ := 3 * side_length
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  perimeter / area = 2 * Real.sqrt 3 / 3 := by
sorry

end equilateral_triangle_perimeter_area_ratio_l863_86318


namespace cubic_root_sum_product_l863_86325

theorem cubic_root_sum_product (p q r : ℝ) : 
  (6 * p^3 - 4 * p^2 + 7 * p - 3 = 0) ∧ 
  (6 * q^3 - 4 * q^2 + 7 * q - 3 = 0) ∧ 
  (6 * r^3 - 4 * r^2 + 7 * r - 3 = 0) → 
  p * q + q * r + r * p = 7/6 := by
sorry

end cubic_root_sum_product_l863_86325


namespace base_spheres_in_triangular_pyramid_l863_86383

/-- The number of spheres in the nth layer of a regular triangular pyramid -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of spheres in a regular triangular pyramid with n layers -/
def total_spheres (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- 
Given a regular triangular pyramid of tightly packed identical spheres,
if the total number of spheres is 120, then the number of spheres in the base is 36.
-/
theorem base_spheres_in_triangular_pyramid :
  ∃ n : ℕ, total_spheres n = 120 ∧ triangular_number n = 36 :=
by sorry

end base_spheres_in_triangular_pyramid_l863_86383


namespace wendy_time_l863_86320

-- Define the race participants
structure Racer where
  name : String
  time : Real

-- Define the race
def waterslideRace (bonnie wendy : Racer) : Prop :=
  wendy.time + 0.25 = bonnie.time ∧ bonnie.time = 7.80

-- Theorem to prove
theorem wendy_time (bonnie wendy : Racer) :
  waterslideRace bonnie wendy → wendy.time = 7.55 := by
  sorry

end wendy_time_l863_86320
