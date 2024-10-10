import Mathlib

namespace power_function_solution_l1282_128238

-- Define a power function type
def PowerFunction := ℝ → ℝ

-- Define the properties of our specific power function
def isPowerFunctionThroughPoint (f : PowerFunction) : Prop :=
  ∃ α : ℝ, (∀ x : ℝ, f x = x ^ α) ∧ f (-2) = -1/8

-- State the theorem
theorem power_function_solution 
  (f : PowerFunction) 
  (h : isPowerFunctionThroughPoint f) : 
  ∃ x : ℝ, f x = 27 ∧ x = 1/3 := by
sorry

end power_function_solution_l1282_128238


namespace property_sale_gain_l1282_128212

/-- Represents the sale of two properties with given selling prices and percentage changes --/
def PropertySale (house_price store_price : ℝ) (house_loss store_gain : ℝ) : Prop :=
  ∃ (house_cost store_cost : ℝ),
    house_price = house_cost * (1 - house_loss) ∧
    store_price = store_cost * (1 + store_gain) ∧
    house_price + store_price - (house_cost + store_cost) = 1000

/-- Theorem stating that the given property sale results in a $1000 gain --/
theorem property_sale_gain :
  PropertySale 15000 18000 0.25 0.50 := by
  sorry

#check property_sale_gain

end property_sale_gain_l1282_128212


namespace pizza_cost_equality_l1282_128295

theorem pizza_cost_equality (total_cost : ℚ) (num_slices : ℕ) 
  (h1 : total_cost = 13)
  (h2 : num_slices = 10) :
  let cost_per_slice := total_cost / num_slices
  5 * cost_per_slice = 5 * cost_per_slice := by
sorry

end pizza_cost_equality_l1282_128295


namespace problem_solution_l1282_128261

theorem problem_solution (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 2 = d + Real.sqrt (a + b + c - d + 1) → d = 9/4 := by
  sorry

end problem_solution_l1282_128261


namespace ratio_evaluation_l1282_128264

theorem ratio_evaluation : (2^3002 * 3^3005) / 6^3003 = 9/2 := by
  sorry

end ratio_evaluation_l1282_128264


namespace bubble_theorem_l1282_128218

/-- Given a hemisphere with radius 4∛2 cm and volume double that of an initial spherical bubble,
    prove the radius of the original bubble and the volume of a new sphere with doubled radius. -/
theorem bubble_theorem (r : ℝ) (h1 : r = 4 * Real.rpow 2 (1/3)) :
  let R := Real.rpow 4 (1/3)
  let V_new := (64/3) * Real.pi * Real.rpow 4 (1/3)
  (2/3) * Real.pi * r^3 = 2 * ((4/3) * Real.pi * R^3) ∧ 
  (4/3) * Real.pi * (2*R)^3 = V_new := by
  sorry

end bubble_theorem_l1282_128218


namespace cd_cost_l1282_128239

theorem cd_cost (num_films : ℕ) (num_books : ℕ) (num_cds : ℕ) 
  (film_cost : ℕ) (book_cost : ℕ) (total_spent : ℕ) :
  num_films = 9 →
  num_books = 4 →
  num_cds = 6 →
  film_cost = 5 →
  book_cost = 4 →
  total_spent = 79 →
  (total_spent - (num_films * film_cost + num_books * book_cost)) / num_cds = 3 := by
  sorry

#eval (79 - (9 * 5 + 4 * 4)) / 6

end cd_cost_l1282_128239


namespace right_triangle_inscribed_circle_angles_l1282_128207

theorem right_triangle_inscribed_circle_angles (k : ℝ) (k_pos : k > 0) :
  ∃ (α β : ℝ),
    α + β = π / 2 ∧
    (α = π / 4 - Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1))) ∨
     α = π / 4 + Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1)))) ∧
    (β = π / 4 - Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1))) ∨
     β = π / 4 + Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1)))) :=
by sorry


end right_triangle_inscribed_circle_angles_l1282_128207


namespace angle_measure_in_special_pentagon_l1282_128257

/-- Given a pentagon PQRST where ∠P ≅ ∠R ≅ ∠T and ∠Q is supplementary to ∠S,
    the measure of ∠T is 120°. -/
theorem angle_measure_in_special_pentagon (P Q R S T : ℝ) : 
  P + Q + R + S + T = 540 →  -- Sum of angles in a pentagon
  Q + S = 180 →              -- ∠Q and ∠S are supplementary
  P = T ∧ R = T →            -- ∠P ≅ ∠R ≅ ∠T
  T = 120 := by
sorry

end angle_measure_in_special_pentagon_l1282_128257


namespace fifteen_percent_of_x_is_ninety_l1282_128263

theorem fifteen_percent_of_x_is_ninety (x : ℝ) : (15 / 100) * x = 90 → x = 600 := by
  sorry

end fifteen_percent_of_x_is_ninety_l1282_128263


namespace cow_ratio_theorem_l1282_128236

theorem cow_ratio_theorem (big_cows small_cows : ℕ) 
  (h : big_cows * 7 = small_cows * 6) : 
  (small_cows - big_cows : ℚ) / small_cows = 1 / 7 := by
  sorry

end cow_ratio_theorem_l1282_128236


namespace F_4_f_5_equals_69_l1282_128294

-- Define the functions f and F
def f (a : ℝ) : ℝ := 2 * a - 2

def F (a b : ℝ) : ℝ := b^2 + a + 1

-- State the theorem
theorem F_4_f_5_equals_69 : F 4 (f 5) = 69 := by
  sorry

end F_4_f_5_equals_69_l1282_128294


namespace intersection_implies_sum_l1282_128293

def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 3 = 0}
def B (p q : ℝ) : Set ℝ := {x | x^2 - q*x - p = 0}

theorem intersection_implies_sum (p q : ℝ) : A p ∩ B p q = {-1} → 2*p + q = -7 := by
  sorry

end intersection_implies_sum_l1282_128293


namespace zoes_flower_purchase_l1282_128232

theorem zoes_flower_purchase (flower_price : ℕ) (roses_bought : ℕ) (total_spent : ℕ) : 
  flower_price = 3 →
  roses_bought = 8 →
  total_spent = 30 →
  (total_spent - roses_bought * flower_price) / flower_price = 2 := by
sorry

end zoes_flower_purchase_l1282_128232


namespace arithmetic_sequence_problem_l1282_128209

/-- An arithmetic sequence {a_n} where a_1 = 1/3, a_2 + a_5 = 4, and a_n = 33 has n = 50 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) (n : ℕ) :
  (∀ k : ℕ, a (k + 1) - a k = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 1 / 3 →
  a 2 + a 5 = 4 →
  a n = 33 →
  n = 50 := by
sorry

end arithmetic_sequence_problem_l1282_128209


namespace limit_of_function_at_one_l1282_128226

theorem limit_of_function_at_one :
  let f : ℝ → ℝ := λ x ↦ 2 * x - 3 - 1 / x
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |f x - (-2)| < ε :=
by sorry

end limit_of_function_at_one_l1282_128226


namespace projected_attendance_increase_l1282_128217

theorem projected_attendance_increase (A : ℝ) (h1 : A > 0) : 
  let actual_attendance := 0.8 * A
  let projected_attendance := (1 + P / 100) * A
  0.8 * A = 0.64 * ((1 + P / 100) * A) →
  P = 25 :=
by
  sorry

end projected_attendance_increase_l1282_128217


namespace jamie_dives_for_pearls_l1282_128205

/-- Given that 25% of oysters have pearls, Jamie can collect 16 oysters per dive,
    and Jamie needs to collect 56 pearls, prove that Jamie needs to make 14 dives. -/
theorem jamie_dives_for_pearls (pearl_probability : ℚ) (oysters_per_dive : ℕ) (total_pearls : ℕ) :
  pearl_probability = 1/4 →
  oysters_per_dive = 16 →
  total_pearls = 56 →
  (total_pearls : ℚ) / (pearl_probability * oysters_per_dive) = 14 := by
  sorry

end jamie_dives_for_pearls_l1282_128205


namespace function_properties_l1282_128235

/-- A function f(x) = x^2 + bx + c where b and c are real numbers -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The theorem statement -/
theorem function_properties (b c : ℝ) 
  (h : ∀ x : ℝ, 2*x + b ≤ f b c x) : 
  (∀ x : ℝ, x ≥ 0 → f b c x ≤ (x + c)^2) ∧ 
  (∃ m : ℝ, m = 3/2 ∧ ∀ b' c' : ℝ, (∀ x : ℝ, 2*x + b' ≤ f b' c' x) → 
    f b' c' c' - f b' c' b' ≤ m*(c'^2 - b'^2) ∧
    ∀ m' : ℝ, (∀ b' c' : ℝ, (∀ x : ℝ, 2*x + b' ≤ f b' c' x) → 
      f b' c' c' - f b' c' b' ≤ m'*(c'^2 - b'^2)) → m ≤ m') := by
  sorry

end function_properties_l1282_128235


namespace base_conversion_subtraction_l1282_128287

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base_7_num := [3, 0, 1, 2, 5]  -- 52103 in base 7 (least significant digit first)
  let base_5_num := [0, 2, 1, 3, 4]  -- 43120 in base 5 (least significant digit first)
  to_base_10 base_7_num 7 - to_base_10 base_5_num 5 = 9833 := by
  sorry

end base_conversion_subtraction_l1282_128287


namespace expression_simplification_l1282_128282

theorem expression_simplification (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end expression_simplification_l1282_128282


namespace investment_growth_l1282_128234

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.08

/-- The time period in years -/
def time_period : ℕ := 28

/-- The initial investment amount in dollars -/
def initial_investment : ℝ := 3500

/-- The final value after the investment period in dollars -/
def final_value : ℝ := 31500

/-- Compound interest formula: A = P(1 + r)^t 
    Where A is the final amount, P is the principal (initial investment),
    r is the annual interest rate, and t is the time in years -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_growth :
  compound_interest initial_investment interest_rate time_period = final_value := by
  sorry

end investment_growth_l1282_128234


namespace solve_equation_l1282_128274

-- Define the new operation
def star_op (a b : ℝ) : ℝ := 3 * a - 2 * b^2

-- Theorem statement
theorem solve_equation (a : ℝ) (h : star_op a 4 = 10) : a = 14 := by
  sorry

end solve_equation_l1282_128274


namespace rectangular_prism_width_l1282_128216

theorem rectangular_prism_width (l h d w : ℝ) : 
  l = 5 → h = 7 → d = 15 → d^2 = l^2 + w^2 + h^2 → w^2 = 151 := by
  sorry

end rectangular_prism_width_l1282_128216


namespace palindrome_product_sum_l1282_128253

/-- A number is a three-digit palindrome if it's between 100 and 999 (inclusive) and reads the same forwards and backwards. -/
def IsThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ ((n / 10) % 10 = (n % 100) / 10)

theorem palindrome_product_sum (a b : ℕ) : 
  IsThreeDigitPalindrome a → IsThreeDigitPalindrome b → a * b = 334491 → a + b = 1324 := by
  sorry

end palindrome_product_sum_l1282_128253


namespace ball_count_l1282_128292

theorem ball_count (num_red : ℕ) (prob_red : ℚ) (total : ℕ) : 
  num_red = 4 → prob_red = 1/3 → total = num_red / prob_red → total = 12 := by
  sorry

end ball_count_l1282_128292


namespace no_integer_solutions_l1282_128242

theorem no_integer_solutions : ¬ ∃ (a b c : ℤ), a^2 + b^2 = 8*c + 6 := by
  sorry

end no_integer_solutions_l1282_128242


namespace subtraction_problem_l1282_128255

theorem subtraction_problem : 
  2000000000000 - 1111111111111 - 222222222222 = 666666666667 := by
  sorry

end subtraction_problem_l1282_128255


namespace rhombus_longer_diagonal_l1282_128230

/-- A rhombus with given properties -/
structure Rhombus where
  /-- Length of the shorter diagonal -/
  shorter_diagonal : ℝ
  /-- Length of the longer diagonal -/
  longer_diagonal : ℝ
  /-- Perimeter of the rhombus -/
  perimeter : ℝ
  /-- The shorter diagonal is 30 cm -/
  shorter_diagonal_length : shorter_diagonal = 30
  /-- The perimeter is 156 cm -/
  perimeter_length : perimeter = 156
  /-- The longer diagonal is longer than the shorter diagonal -/
  diagonal_order : longer_diagonal ≥ shorter_diagonal

/-- Theorem: In a rhombus with one diagonal of 30 cm and a perimeter of 156 cm, 
    the length of the other diagonal is 72 cm -/
theorem rhombus_longer_diagonal (r : Rhombus) : r.longer_diagonal = 72 := by
  sorry

#check rhombus_longer_diagonal

end rhombus_longer_diagonal_l1282_128230


namespace max_ash_win_probability_l1282_128211

/-- Represents the types of monsters -/
inductive MonsterType
  | Fire
  | Grass
  | Water

/-- A lineup of monsters -/
def Lineup := List MonsterType

/-- The number of monsters in each lineup -/
def lineupSize : Nat := 15

/-- Calculates the probability of Ash winning given his lineup strategy -/
noncomputable def ashWinProbability (ashStrategy : Lineup) : ℝ :=
  sorry

/-- Theorem stating the maximum probability of Ash winning -/
theorem max_ash_win_probability :
  ∃ (optimalStrategy : Lineup),
    ashWinProbability optimalStrategy = 1 - (2/3)^lineupSize ∧
    ∀ (strategy : Lineup),
      ashWinProbability strategy ≤ ashWinProbability optimalStrategy :=
  sorry

end max_ash_win_probability_l1282_128211


namespace merchant_transaction_loss_l1282_128203

theorem merchant_transaction_loss : 
  ∀ (cost_profit cost_loss : ℝ),
  cost_profit * 1.15 = 1955 →
  cost_loss * 0.85 = 1955 →
  (1955 + 1955) - (cost_profit + cost_loss) = -90 :=
by
  sorry

end merchant_transaction_loss_l1282_128203


namespace unique_solution_for_equation_l1282_128260

theorem unique_solution_for_equation (N : ℕ) (a b c : ℕ+) :
  N > 3 →
  Odd N →
  a ^ N = b ^ N + 2 ^ N + a * b * c →
  c ≤ 5 * 2 ^ (N - 1) →
  N = 5 ∧ a = 3 ∧ b = 1 ∧ c = 70 :=
by sorry

end unique_solution_for_equation_l1282_128260


namespace sarahs_bowling_score_l1282_128265

/-- Sarah's bowling score problem -/
theorem sarahs_bowling_score :
  ∀ (sarah_score greg_score : ℕ),
  sarah_score = greg_score + 50 →
  (sarah_score + greg_score) / 2 = 110 →
  sarah_score = 135 :=
by
  sorry

end sarahs_bowling_score_l1282_128265


namespace product_less_than_2400_l1282_128228

theorem product_less_than_2400 : 817 * 3 < 2400 := by
  sorry

end product_less_than_2400_l1282_128228


namespace seating_arrangements_l1282_128284

/-- The number of seating arrangements for four students and two teachers under different conditions. -/
theorem seating_arrangements (n_students : Nat) (n_teachers : Nat) : n_students = 4 ∧ n_teachers = 2 →
  (∃ (arrangements_middle : Nat), arrangements_middle = 48) ∧
  (∃ (arrangements_together : Nat), arrangements_together = 144) ∧
  (∃ (arrangements_separate : Nat), arrangements_separate = 144) := by
  sorry

#check seating_arrangements

end seating_arrangements_l1282_128284


namespace absolute_sum_vs_square_sum_l1282_128220

theorem absolute_sum_vs_square_sum :
  (∀ x y : ℝ, (abs x + abs y ≤ 1) → (x^2 + y^2 ≤ 1)) ∧
  (∃ x y : ℝ, (x^2 + y^2 ≤ 1) ∧ (abs x + abs y > 1)) := by
  sorry

end absolute_sum_vs_square_sum_l1282_128220


namespace dividend_calculation_l1282_128259

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 9) :
  divisor * quotient + remainder = 162 := by
  sorry

end dividend_calculation_l1282_128259


namespace multiplication_mistake_l1282_128250

theorem multiplication_mistake (x : ℚ) : (43 * x - 34 * x = 1251) → x = 139 := by
  sorry

end multiplication_mistake_l1282_128250


namespace smallest_n_square_and_cube_l1282_128225

/-- 
Given a positive integer n, we define two properties:
1. 5n is a perfect square
2. 7n is a perfect cube

This theorem states that 1225 is the smallest positive integer satisfying both properties.
-/
theorem smallest_n_square_and_cube : ∀ n : ℕ+, 
  (∃ k : ℕ+, 5 * n = k^2) ∧ 
  (∃ m : ℕ+, 7 * n = m^3) → 
  n ≥ 1225 :=
sorry

end smallest_n_square_and_cube_l1282_128225


namespace function_minimum_condition_l1282_128251

def f (x a : ℝ) := x^2 - 2*a*x + a

theorem function_minimum_condition (a : ℝ) :
  (∃ x, x < 1 ∧ ∀ y < 1, f y a ≥ f x a) → a < 1 := by
  sorry

end function_minimum_condition_l1282_128251


namespace total_eggs_supplied_weekly_l1282_128299

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens supplied to Store A daily -/
def dozens_to_store_A : ℕ := 5

/-- The number of eggs supplied to Store B daily -/
def eggs_to_store_B : ℕ := 30

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total number of eggs supplied to both stores in a week is 630 -/
theorem total_eggs_supplied_weekly : 
  (dozens_to_store_A * eggs_per_dozen + eggs_to_store_B) * days_in_week = 630 := by
sorry

end total_eggs_supplied_weekly_l1282_128299


namespace reflect_A_across_x_axis_l1282_128246

def reflect_point_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflect_A_across_x_axis :
  let A : ℝ × ℝ := (-5, -2)
  reflect_point_x_axis A = (-5, 2) := by
  sorry

end reflect_A_across_x_axis_l1282_128246


namespace meat_for_forty_burgers_l1282_128285

/-- The number of pounds of meat needed to make a given number of hamburgers -/
def meatNeeded (initialPounds : ℚ) (initialBurgers : ℕ) (targetBurgers : ℕ) : ℚ :=
  (initialPounds / initialBurgers) * targetBurgers

/-- Theorem stating that 20 pounds of meat are needed for 40 hamburgers
    given that 5 pounds of meat make 10 hamburgers -/
theorem meat_for_forty_burgers :
  meatNeeded 5 10 40 = 20 := by
  sorry

#eval meatNeeded 5 10 40

end meat_for_forty_burgers_l1282_128285


namespace new_car_distance_l1282_128267

theorem new_car_distance (old_car_speed : ℝ) (old_car_distance : ℝ) (new_car_speed : ℝ) : 
  old_car_distance = 180 →
  new_car_speed = old_car_speed * 1.15 →
  new_car_speed * (old_car_distance / old_car_speed) = 207 :=
by sorry

end new_car_distance_l1282_128267


namespace seventh_observation_l1282_128221

theorem seventh_observation (n : ℕ) (initial_avg : ℚ) (new_avg : ℚ) : 
  n = 6 → 
  initial_avg = 12 → 
  new_avg = 11 → 
  (n * initial_avg + (n + 1) * new_avg - n * initial_avg) / (n + 1) = 5 := by
  sorry

end seventh_observation_l1282_128221


namespace molly_gift_cost_per_package_l1282_128272

/-- The cost per package for Molly's Christmas gifts --/
def cost_per_package (total_relatives : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / total_relatives

/-- Theorem: The cost per package for Molly's Christmas gifts is $5 --/
theorem molly_gift_cost_per_package :
  let total_relatives : ℕ := 14
  let total_cost : ℚ := 70
  cost_per_package total_relatives total_cost = 5 := by
  sorry


end molly_gift_cost_per_package_l1282_128272


namespace determinant_max_value_l1282_128243

open Real

theorem determinant_max_value : 
  let det := fun θ : ℝ => 
    Matrix.det !![1, 1, 1; 1, 1 + cos θ, 1; 1 + sin θ, 1, 1]
  ∃ (max_val : ℝ), max_val = 1/2 ∧ ∀ θ, det θ ≤ max_val :=
by sorry

end determinant_max_value_l1282_128243


namespace red_star_wins_l1282_128245

theorem red_star_wins (total_matches : ℕ) (total_points : ℕ) 
  (h1 : total_matches = 9)
  (h2 : total_points = 23)
  (h3 : ∀ (wins draws : ℕ), wins + draws = total_matches → 3 * wins + draws = total_points) :
  ∃ (wins draws : ℕ), wins = 7 ∧ draws = 2 := by
  sorry

end red_star_wins_l1282_128245


namespace machine_completion_time_l1282_128201

/-- Given two machines where one takes T hours and the other takes 8 hours to complete an order,
    if they complete the order together in 4.235294117647059 hours, then T = 9. -/
theorem machine_completion_time (T : ℝ) : 
  (1 / T + 1 / 8 = 1 / 4.235294117647059) → T = 9 := by
sorry

end machine_completion_time_l1282_128201


namespace round_recurring_decimal_to_thousandth_l1282_128233

/-- The repeating decimal 36.3636... -/
def recurring_decimal : ℚ := 36 + 36 / 99

/-- Rounding a number to the nearest thousandth -/
def round_to_thousandth (x : ℚ) : ℚ := 
  (⌊x * 1000 + 0.5⌋) / 1000

/-- Proof that rounding 36.3636... to the nearest thousandth equals 36.363 -/
theorem round_recurring_decimal_to_thousandth : 
  round_to_thousandth recurring_decimal = 36363 / 1000 := by
  sorry

end round_recurring_decimal_to_thousandth_l1282_128233


namespace juan_speed_l1282_128278

/-- Given a distance of 80 miles and a time of 8 hours, prove that the speed is 10 miles per hour. -/
theorem juan_speed (distance : ℝ) (time : ℝ) (h1 : distance = 80) (h2 : time = 8) :
  distance / time = 10 := by
  sorry

end juan_speed_l1282_128278


namespace pyramid_lateral_angle_l1282_128240

/-- Given a pyramid with an isosceles triangular base of area S, angle α between the equal sides,
    and volume V, the angle θ between the lateral edges and the base plane is:
    θ = arctan((3V * cos(α/2) / S) * sqrt(2 * sin(α) / S)) -/
theorem pyramid_lateral_angle (S V : ℝ) (α : ℝ) (hS : S > 0) (hV : V > 0) (hα : 0 < α ∧ α < π) :
  ∃ θ : ℝ, θ = Real.arctan ((3 * V * Real.cos (α / 2) / S) * Real.sqrt (2 * Real.sin α / S)) :=
sorry

end pyramid_lateral_angle_l1282_128240


namespace monotonic_increasing_condition_l1282_128271

/-- Given a function f(x) = ax - a/x - 2ln(x) where a ≥ 0, if f(x) is monotonically increasing
    on its domain (0, +∞), then a > 1. -/
theorem monotonic_increasing_condition (a : ℝ) (h_a : a ≥ 0) :
  (∀ x : ℝ, x > 0 → Monotone (fun x => a * x - a / x - 2 * Real.log x)) →
  a > 1 := by
  sorry

end monotonic_increasing_condition_l1282_128271


namespace divisors_of_m_squared_count_specific_divisors_l1282_128268

def m : ℕ := 2^40 * 5^24

theorem divisors_of_m_squared (d : ℕ) : 
  (d ∣ m^2) ∧ (d < m) ∧ ¬(d ∣ m) ↔ d ∈ Finset.filter (λ x => (x ∣ m^2) ∧ (x < m) ∧ ¬(x ∣ m)) (Finset.range (m + 1)) :=
sorry

theorem count_specific_divisors : 
  Finset.card (Finset.filter (λ x => (x ∣ m^2) ∧ (x < m) ∧ ¬(x ∣ m)) (Finset.range (m + 1))) = 959 :=
sorry

end divisors_of_m_squared_count_specific_divisors_l1282_128268


namespace income_ratio_proof_l1282_128270

/-- Proves that the ratio of A's monthly income to B's monthly income is 2.5:1 -/
theorem income_ratio_proof (c_monthly_income b_monthly_income a_annual_income : ℝ) 
  (h1 : c_monthly_income = 14000)
  (h2 : b_monthly_income = c_monthly_income * 1.12)
  (h3 : a_annual_income = 470400) : 
  (a_annual_income / 12) / b_monthly_income = 2.5 := by
  sorry

#check income_ratio_proof

end income_ratio_proof_l1282_128270


namespace smallest_n_value_l1282_128275

theorem smallest_n_value (r g b : ℕ+) (h : 10 * r = 18 * g ∧ 18 * g = 20 * b) :
  ∃ (n : ℕ+), 30 * n = 10 * r ∧ ∀ (m : ℕ+), 30 * m = 10 * r → n ≤ m :=
by sorry

end smallest_n_value_l1282_128275


namespace golden_retriever_weight_at_8_years_l1282_128289

/-- Calculates the weight of a golden retriever given its age and initial conditions -/
def goldenRetrieverWeight (initialWeight : ℕ) (firstYearGain : ℕ) (yearlyGain : ℕ) (yearlyLoss : ℕ) (age : ℕ) : ℕ :=
  let weightAfterFirstYear := initialWeight + firstYearGain
  let netYearlyGain := yearlyGain - yearlyLoss
  weightAfterFirstYear + (age - 1) * netYearlyGain

/-- Theorem stating the weight of a specific golden retriever at 8 years old -/
theorem golden_retriever_weight_at_8_years :
  goldenRetrieverWeight 3 15 11 3 8 = 74 := by
  sorry

end golden_retriever_weight_at_8_years_l1282_128289


namespace lunch_percentage_theorem_l1282_128248

theorem lunch_percentage_theorem (total : ℕ) (boy_ratio girl_ratio : ℕ) 
  (boy_lunch_percent girl_lunch_percent : ℚ) :
  boy_ratio + girl_ratio > 0 →
  boy_lunch_percent ≥ 0 →
  boy_lunch_percent ≤ 1 →
  girl_lunch_percent ≥ 0 →
  girl_lunch_percent ≤ 1 →
  boy_ratio = 6 →
  girl_ratio = 4 →
  boy_lunch_percent = 6/10 →
  girl_lunch_percent = 4/10 →
  (((boy_ratio * boy_lunch_percent + girl_ratio * girl_lunch_percent) / 
    (boy_ratio + girl_ratio)) : ℚ) = 52/100 := by
  sorry

end lunch_percentage_theorem_l1282_128248


namespace contradiction_assumption_l1282_128210

theorem contradiction_assumption (x y z : ℝ) :
  (¬ (x > 0 ∨ y > 0 ∨ z > 0)) ↔ (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) := by
  sorry

end contradiction_assumption_l1282_128210


namespace xiao_tian_hat_l1282_128290

-- Define the type for hat numbers
inductive HatNumber
  | one
  | two
  | three
  | four
  | five

-- Define the type for people
inductive Person
  | xiaoWang
  | xiaoKong
  | xiaoTian
  | xiaoYan
  | xiaoWei

-- Define the function that assigns hat numbers to people
def hatAssignment : Person → HatNumber := sorry

-- Define the function that determines if one person can see another's hat
def canSee : Person → Person → Bool := sorry

-- State the theorem
theorem xiao_tian_hat :
  (∀ p, ¬canSee Person.xiaoWang p) →
  (∃! p, canSee Person.xiaoKong p ∧ hatAssignment p = HatNumber.four) →
  (∃ p, canSee Person.xiaoTian p ∧ hatAssignment p = HatNumber.one) →
  (¬∃ p, canSee Person.xiaoTian p ∧ hatAssignment p = HatNumber.three) →
  (∃ p₁ p₂ p₃, canSee Person.xiaoYan p₁ ∧ canSee Person.xiaoYan p₂ ∧ canSee Person.xiaoYan p₃ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃) →
  (¬∃ p, canSee Person.xiaoYan p ∧ hatAssignment p = HatNumber.three) →
  (∃ p₁ p₂, canSee Person.xiaoWei p₁ ∧ canSee Person.xiaoWei p₂ ∧
    hatAssignment p₁ = HatNumber.three ∧ hatAssignment p₂ = HatNumber.two) →
  (∀ p₁ p₂, p₁ ≠ p₂ → hatAssignment p₁ ≠ hatAssignment p₂) →
  hatAssignment Person.xiaoTian = HatNumber.two :=
sorry

end xiao_tian_hat_l1282_128290


namespace equality_condition_l1282_128214

theorem equality_condition (x : ℝ) (hx : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) = 15 ↔ x = 3 ∨ x = 1 := by
  sorry

end equality_condition_l1282_128214


namespace min_cards_to_draw_l1282_128269

/- Define the number of suits in a deck -/
def num_suits : ℕ := 4

/- Define the number of cards in each suit -/
def cards_per_suit : ℕ := 13

/- Define the number of cards needed in the same suit -/
def cards_needed_same_suit : ℕ := 4

/- Define the number of jokers in the deck -/
def num_jokers : ℕ := 2

/- Theorem: The minimum number of cards to draw to ensure 4 of the same suit is 15 -/
theorem min_cards_to_draw : 
  (num_suits - 1) * (cards_needed_same_suit - 1) + cards_needed_same_suit + num_jokers = 15 := by
  sorry

end min_cards_to_draw_l1282_128269


namespace concert_stay_probability_l1282_128266

/-- The probability that at least 4 people stay for an entire concert, given the conditions. -/
theorem concert_stay_probability (total : ℕ) (certain : ℕ) (uncertain : ℕ) (p : ℚ) : 
  total = 8 →
  certain = 5 →
  uncertain = 3 →
  p = 1/3 →
  ∃ (prob : ℚ), prob = 19/27 ∧ 
    prob = (uncertain.choose 1 * p * (1-p)^2 + 
            uncertain.choose 2 * p^2 * (1-p) + 
            uncertain.choose 3 * p^3) := by
  sorry


end concert_stay_probability_l1282_128266


namespace leja_theorem_l1282_128237

/-- A set of points in a plane where any three points lie on a circle of radius r -/
def SpecialPointSet (P : Set (ℝ × ℝ)) (r : ℝ) : Prop :=
  ∀ p q s : ℝ × ℝ, p ∈ P → q ∈ P → s ∈ P → p ≠ q → q ≠ s → p ≠ s →
    ∃ c : ℝ × ℝ, dist c p = r ∧ dist c q = r ∧ dist c s = r

/-- Leja's theorem -/
theorem leja_theorem (P : Set (ℝ × ℝ)) (r : ℝ) (h : SpecialPointSet P r) :
  ∃ A : ℝ × ℝ, ∀ p ∈ P, dist A p ≤ r := by
  sorry

end leja_theorem_l1282_128237


namespace mary_regular_hours_l1282_128224

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  regularHours : ℕ
  overtimeHours : ℕ
  regularRate : ℕ
  overtimeRate : ℕ
  totalEarnings : ℕ

/-- Calculates the total earnings based on the work schedule --/
def calculateEarnings (schedule : WorkSchedule) : ℕ :=
  schedule.regularHours * schedule.regularRate + schedule.overtimeHours * schedule.overtimeRate

/-- The main theorem stating Mary's work hours at regular rate --/
theorem mary_regular_hours :
  ∃ (schedule : WorkSchedule),
    schedule.regularHours = 40 ∧
    schedule.regularRate = 8 ∧
    schedule.overtimeRate = 10 ∧
    schedule.regularHours + schedule.overtimeHours ≤ 40 ∧
    calculateEarnings schedule = 360 :=
by
  sorry

#check mary_regular_hours

end mary_regular_hours_l1282_128224


namespace smallest_next_divisor_after_221_l1282_128200

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : m ≥ 1000 ∧ m ≤ 9999) 
  (h2 : Even m) (h3 : m % 221 = 0) :
  ∃ (d : ℕ), d > 221 ∧ m % d = 0 ∧ d ≥ 238 ∧ 
  ∀ (d' : ℕ), d' > 221 ∧ m % d' = 0 → d' ≥ 238 :=
sorry

end smallest_next_divisor_after_221_l1282_128200


namespace tan_fifteen_thirty_product_l1282_128252

theorem tan_fifteen_thirty_product : (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end tan_fifteen_thirty_product_l1282_128252


namespace lava_lamp_probability_l1282_128231

/-- The probability of a specific arrangement of lava lamps -/
theorem lava_lamp_probability :
  let total_lamps : ℕ := 6
  let red_lamps : ℕ := 3
  let blue_lamps : ℕ := 3
  let lamps_on : ℕ := 3
  let color_arrangements := Nat.choose total_lamps red_lamps
  let on_arrangements := Nat.choose total_lamps lamps_on
  let remaining_lamps : ℕ := 4
  let remaining_red : ℕ := 2
  let remaining_color_arrangements := Nat.choose remaining_lamps remaining_red
  let remaining_on_arrangements := Nat.choose remaining_lamps remaining_red
  (remaining_color_arrangements * remaining_on_arrangements : ℚ) / (color_arrangements * on_arrangements) = 9 / 100 :=
by sorry

end lava_lamp_probability_l1282_128231


namespace pierre_cake_consumption_l1282_128249

theorem pierre_cake_consumption (cake_weight : ℝ) (num_parts : ℕ) 
  (h1 : cake_weight = 400)
  (h2 : num_parts = 8)
  (h3 : num_parts > 0) :
  let part_weight := cake_weight / num_parts
  let nathalie_ate := part_weight
  let pierre_ate := 2 * nathalie_ate
  pierre_ate = 100 := by
  sorry

end pierre_cake_consumption_l1282_128249


namespace integral_x_cos_x_over_sin_cubed_x_l1282_128256

open Real

theorem integral_x_cos_x_over_sin_cubed_x (x : ℝ) :
  deriv (fun x => - (x + cos x * sin x) / (2 * sin x ^ 2)) x = 
    x * cos x / sin x ^ 3 := by
  sorry

end integral_x_cos_x_over_sin_cubed_x_l1282_128256


namespace fruit_basket_count_l1282_128280

/-- The number of fruit baskets -/
def num_baskets : ℕ := 4

/-- The number of apples in each of the first three baskets -/
def apples_per_basket : ℕ := 9

/-- The number of oranges in each of the first three baskets -/
def oranges_per_basket : ℕ := 15

/-- The number of bananas in each of the first three baskets -/
def bananas_per_basket : ℕ := 14

/-- The number of fruits that are reduced in the fourth basket -/
def reduction : ℕ := 2

/-- The total number of fruits in all baskets -/
def total_fruits : ℕ := 146

theorem fruit_basket_count :
  (3 * (apples_per_basket + oranges_per_basket + bananas_per_basket)) +
  ((apples_per_basket - reduction) + (oranges_per_basket - reduction) + (bananas_per_basket - reduction)) =
  total_fruits := by sorry

end fruit_basket_count_l1282_128280


namespace no_extra_savings_when_combined_l1282_128215

def book_price : ℕ := 120
def alice_books : ℕ := 10
def bob_books : ℕ := 15

def calculate_cost (num_books : ℕ) : ℕ :=
  let free_books := (num_books / 5) * 2
  let paid_books := num_books - free_books
  paid_books * book_price

def calculate_savings (num_books : ℕ) : ℕ :=
  num_books * book_price - calculate_cost num_books

theorem no_extra_savings_when_combined :
  calculate_savings alice_books + calculate_savings bob_books =
  calculate_savings (alice_books + bob_books) :=
by sorry

end no_extra_savings_when_combined_l1282_128215


namespace multiplication_puzzle_l1282_128223

theorem multiplication_puzzle :
  ∀ (A B E F : ℕ),
    A < 10 → B < 10 → E < 10 → F < 10 →
    A ≠ B → A ≠ E → A ≠ F → B ≠ E → B ≠ F → E ≠ F →
    (100 * A + 10 * B + E) * F = 1000 * E + 100 * A + 10 * E + A →
    A + B = 5 := by
sorry

end multiplication_puzzle_l1282_128223


namespace systematic_sampling_most_suitable_for_C_l1282_128222

/-- Characteristics of systematic sampling -/
structure SystematicSampling where
  large_population : Bool
  regular_interval : Bool
  balanced_group : Bool

/-- Sampling scenario -/
structure SamplingScenario where
  population_size : Nat
  sample_size : Nat
  is_homogeneous : Bool

/-- Check if a scenario is suitable for systematic sampling -/
def is_suitable_for_systematic_sampling (scenario : SamplingScenario) : Bool :=
  scenario.population_size > scenario.sample_size ∧ 
  scenario.population_size ≥ 1000 ∧ 
  scenario.sample_size ≥ 100 ∧
  scenario.is_homogeneous

/-- The four sampling scenarios -/
def scenario_A : SamplingScenario := ⟨2000, 200, false⟩
def scenario_B : SamplingScenario := ⟨2000, 5, true⟩
def scenario_C : SamplingScenario := ⟨2000, 200, true⟩
def scenario_D : SamplingScenario := ⟨20, 5, true⟩

theorem systematic_sampling_most_suitable_for_C :
  is_suitable_for_systematic_sampling scenario_C ∧
  ¬is_suitable_for_systematic_sampling scenario_A ∧
  ¬is_suitable_for_systematic_sampling scenario_B ∧
  ¬is_suitable_for_systematic_sampling scenario_D :=
sorry

end systematic_sampling_most_suitable_for_C_l1282_128222


namespace expression_value_at_five_l1282_128283

theorem expression_value_at_five : 
  let x : ℝ := 5
  (x^3 - 4*x^2 + 3*x) / (x - 3) = 20 := by sorry

end expression_value_at_five_l1282_128283


namespace problem_1_problem_2_l1282_128208

-- Problem 1
theorem problem_1 : Real.sqrt 9 + |3 - Real.pi| - Real.sqrt ((-3)^2) = Real.pi - 3 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x : ℝ, 3 * (x - 1)^3 = 81 ∧ x = 4 := by
  sorry

end problem_1_problem_2_l1282_128208


namespace difference_of_sums_l1282_128204

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def smallest_two_digit_multiple_of_three : ℕ := 12

def largest_two_digit_multiple_of_three : ℕ := 99

def smallest_two_digit_non_multiple_of_three : ℕ := 10

def largest_two_digit_non_multiple_of_three : ℕ := 98

theorem difference_of_sums : 
  (largest_two_digit_multiple_of_three + smallest_two_digit_multiple_of_three) -
  (largest_two_digit_non_multiple_of_three + smallest_two_digit_non_multiple_of_three) = 3 := by
  sorry

end difference_of_sums_l1282_128204


namespace quadratic_equation_roots_l1282_128291

theorem quadratic_equation_roots (x : ℝ) :
  x^2 - 4*x - 2 = 0 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 :=
sorry

end quadratic_equation_roots_l1282_128291


namespace rationalize_denominator_cube_roots_l1282_128258

theorem rationalize_denominator_cube_roots :
  let x := (3 : ℝ)^(1/3)
  let y := (2 : ℝ)^(1/3)
  1 / (x - y) = x^2 + x*y + y^2 :=
by sorry

end rationalize_denominator_cube_roots_l1282_128258


namespace circumscribed_trapezoid_leg_length_l1282_128227

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The acute angle at the base of the trapezoid -/
  base_angle : ℝ
  /-- The length of the trapezoid's leg -/
  leg_length : ℝ

/-- Theorem stating the relationship between the area, base angle, and leg length of a circumscribed trapezoid -/
theorem circumscribed_trapezoid_leg_length 
  (t : CircumscribedTrapezoid) 
  (h1 : t.area = 32 * Real.sqrt 3)
  (h2 : t.base_angle = π / 3) :
  t.leg_length = 8 := by
  sorry

#check circumscribed_trapezoid_leg_length

end circumscribed_trapezoid_leg_length_l1282_128227


namespace average_marks_combined_classes_l1282_128247

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 30 →
  avg2 = 60 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = ((n1 + n2) : ℚ) * (48.75 : ℚ) := by
  sorry

end average_marks_combined_classes_l1282_128247


namespace dandelion_puffs_distribution_l1282_128254

theorem dandelion_puffs_distribution (total : ℕ) (given_away : ℕ) (friends : ℕ) 
  (h_total : total = 40)
  (h_given_away : given_away = 3 + 3 + 5 + 2)
  (h_friends : friends = 3)
  (h_positive : friends > 0) :
  (total - given_away) / friends = 9 :=
sorry

end dandelion_puffs_distribution_l1282_128254


namespace quadratic_root_range_l1282_128298

theorem quadratic_root_range (a : ℝ) :
  (∃ x y : ℝ, x^2 + (a^2 - 1)*x + a - 2 = 0 ∧ x > 1 ∧ y < 1 ∧ y^2 + (a^2 - 1)*y + a - 2 = 0) →
  a ∈ Set.Ioo (-2 : ℝ) 1 :=
by sorry

end quadratic_root_range_l1282_128298


namespace tv_price_with_tax_l1282_128277

/-- Calculates the final price of a TV including value-added tax -/
theorem tv_price_with_tax (original_price : ℝ) (tax_rate : ℝ) (final_price : ℝ) :
  original_price = 1700 →
  tax_rate = 0.15 →
  final_price = original_price * (1 + tax_rate) →
  final_price = 1955 := by
  sorry

end tv_price_with_tax_l1282_128277


namespace perpendicular_line_plane_iff_perpendicular_two_lines_l1282_128219

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Predicate for a line being perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate for a line being perpendicular to another line -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines are distinct -/
def distinct_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- The main theorem to be proven false -/
theorem perpendicular_line_plane_iff_perpendicular_two_lines (l : Line3D) (p : Plane3D) :
  perpendicular_line_plane l p ↔ 
  ∃ (l1 l2 : Line3D), line_in_plane l1 p ∧ line_in_plane l2 p ∧ 
                      distinct_lines l1 l2 ∧ 
                      perpendicular_lines l l1 ∧ perpendicular_lines l l2 :=
  sorry

end perpendicular_line_plane_iff_perpendicular_two_lines_l1282_128219


namespace third_candidate_votes_l1282_128288

theorem third_candidate_votes : 
  ∀ (total_votes : ℕ) (invalid_percentage : ℚ) (first_candidate_percentage : ℚ) (second_candidate_percentage : ℚ),
  total_votes = 10000 →
  invalid_percentage = 1/4 →
  first_candidate_percentage = 1/2 →
  second_candidate_percentage = 3/10 →
  ∃ (third_candidate_votes : ℕ),
    third_candidate_votes = total_votes * (1 - invalid_percentage) - 
      (total_votes * (1 - invalid_percentage) * first_candidate_percentage + 
       total_votes * (1 - invalid_percentage) * second_candidate_percentage) ∧
    third_candidate_votes = 1500 :=
by sorry

end third_candidate_votes_l1282_128288


namespace triangle_inequalities_l1282_128276

/-- Triangle inequalities -/
theorem triangle_inequalities (a b c P S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : P = a + b + c)
  (h_area : S = Real.sqrt ((P/2) * ((P/2) - a) * ((P/2) - b) * ((P/2) - c))) :
  (1/a + 1/b + 1/c ≥ 9/P) ∧
  (a^2 + b^2 + c^2 ≥ P^2/3) ∧
  (P^2 ≥ 12 * Real.sqrt 3 * S) ∧
  (a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S) ∧
  (a^3 + b^3 + c^3 ≥ P^3/9) ∧
  (a^3 + b^3 + c^3 ≥ (4 * Real.sqrt 3 / 3) * S * P) ∧
  (a^4 + b^4 + c^4 ≥ 16 * S^2) := by
  sorry

end triangle_inequalities_l1282_128276


namespace budget_allocation_l1282_128273

theorem budget_allocation (home_electronics food_additives gm_microorganisms industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ) :
  home_electronics = 24 →
  food_additives = 15 →
  gm_microorganisms = 19 →
  industrial_lubricants = 8 →
  basic_astrophysics_degrees = 72 →
  let basic_astrophysics := (basic_astrophysics_degrees / 360) * 100
  let total_known := home_electronics + food_additives + gm_microorganisms + industrial_lubricants + basic_astrophysics
  let microphotonics := 100 - total_known
  microphotonics = 14 := by
  sorry

end budget_allocation_l1282_128273


namespace car_trade_profit_l1282_128262

theorem car_trade_profit (original_price : ℝ) (h : original_price > 0) :
  let buying_price := 0.9 * original_price
  let selling_price := buying_price * 1.8
  let profit := selling_price - original_price
  profit / original_price = 0.62 := by
sorry

end car_trade_profit_l1282_128262


namespace product_digit_sum_l1282_128296

/-- Converts a base 7 number to base 10 -/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base 7 -/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The product of 35₇ and 12₇ in base 7 -/
def product : ℕ := toBase7 (toBase10 35 * toBase10 12)

theorem product_digit_sum :
  sumOfDigitsBase7 product = 12 := by sorry

end product_digit_sum_l1282_128296


namespace mode_of_student_ages_l1282_128229

def student_ages : List ℕ := [13, 14, 15, 14, 14, 15]

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_student_ages :
  mode student_ages = 14 := by sorry

end mode_of_student_ages_l1282_128229


namespace least_number_of_cubes_l1282_128297

/-- Represents the dimensions of a cuboidal block in centimeters -/
structure CuboidalBlock where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the side length of a cube in centimeters -/
def CubeSideLength : ℕ := 3

/-- The given cuboidal block -/
def given_block : CuboidalBlock := ⟨18, 27, 36⟩

/-- The volume of a cuboidal block -/
def volume_cuboid (b : CuboidalBlock) : ℕ := b.length * b.width * b.height

/-- The volume of a cube -/
def volume_cube (side : ℕ) : ℕ := side * side * side

/-- The number of cubes that can be cut from a cuboidal block -/
def number_of_cubes (b : CuboidalBlock) (side : ℕ) : ℕ :=
  volume_cuboid b / volume_cube side

/-- Theorem: The least possible number of equal cubes with side lengths in a fixed ratio of 1:2:3
    that can be cut from the given cuboidal block is 648 -/
theorem least_number_of_cubes :
  number_of_cubes given_block CubeSideLength = 648 := by sorry

end least_number_of_cubes_l1282_128297


namespace unique_perfect_square_polynomial_l1282_128202

theorem unique_perfect_square_polynomial : 
  ∃! x : ℤ, ∃ y : ℤ, x^4 + 8*x^3 + 18*x^2 + 8*x + 36 = y^2 :=
by sorry

end unique_perfect_square_polynomial_l1282_128202


namespace lcm_problem_l1282_128244

theorem lcm_problem (a b : ℕ+) (h_product : a * b = 18750) (h_hcf : Nat.gcd a b = 25) :
  Nat.lcm a b = 750 := by
  sorry

end lcm_problem_l1282_128244


namespace f_and_g_properties_l1282_128279

/-- Given a function f and constants a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 + b * x

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * x + b

/-- Function g defined as the sum of f and its derivative -/
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x + f' a b x

/-- g is an odd function -/
axiom g_odd (a b : ℝ) : ∀ x, g a b (-x) = -(g a b x)

theorem f_and_g_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x = -1/3 * x^3 + x^2) ∧
    (∀ x ∈ Set.Icc 1 2, g a b x ≤ 4 * Real.sqrt 2 / 3) ∧
    (∀ x ∈ Set.Icc 1 2, g a b x ≥ 4 / 3) ∧
    (g a b (Real.sqrt 2) = 4 * Real.sqrt 2 / 3) ∧
    (g a b 2 = 4 / 3) := by sorry

end f_and_g_properties_l1282_128279


namespace tennis_players_count_l1282_128281

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  badminton_players : ℕ
  neither_players : ℕ
  both_players : ℕ

/-- Calculate the number of tennis players in the sports club -/
def tennis_players (club : SportsClub) : ℕ :=
  club.total_members - club.neither_players - (club.badminton_players - club.both_players)

/-- Theorem stating the number of tennis players in the given club configuration -/
theorem tennis_players_count (club : SportsClub) 
  (h1 : club.total_members = 30)
  (h2 : club.badminton_players = 17)
  (h3 : club.neither_players = 2)
  (h4 : club.both_players = 8) :
  tennis_players club = 19 := by
  sorry

#eval tennis_players ⟨30, 17, 2, 8⟩

end tennis_players_count_l1282_128281


namespace inequality_proof_l1282_128213

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) > Real.sqrt 3 * a := by
  sorry

end inequality_proof_l1282_128213


namespace green_ball_probability_l1282_128286

-- Define the containers and their contents
def containerA : ℕ × ℕ := (5, 7)  -- (red, green)
def containerB : ℕ × ℕ := (8, 6)
def containerC : ℕ × ℕ := (3, 9)

-- Define the probability of selecting each container
def containerProb : ℚ := 1 / 3

-- Define the probability of selecting a green ball from each container
def greenProbA : ℚ := containerA.2 / (containerA.1 + containerA.2)
def greenProbB : ℚ := containerB.2 / (containerB.1 + containerB.2)
def greenProbC : ℚ := containerC.2 / (containerC.1 + containerC.2)

-- Theorem: The probability of selecting a green ball is 127/252
theorem green_ball_probability : 
  containerProb * greenProbA + containerProb * greenProbB + containerProb * greenProbC = 127 / 252 := by
  sorry


end green_ball_probability_l1282_128286


namespace inequality_theorem_l1282_128206

/-- A function f: ℝ → ℝ satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, Differentiable ℝ f ∧ (x - 1) * (deriv (deriv f) x) < 0

/-- Theorem stating the inequality for functions satisfying the condition -/
theorem inequality_theorem (f : ℝ → ℝ) (h : SatisfiesCondition f) :
  f 0 + f 2 < 2 * f 1 := by
  sorry

end inequality_theorem_l1282_128206


namespace rectangle_area_relation_l1282_128241

/-- Given a rectangle with area 10 and adjacent sides x and y, 
    prove that the relationship between x and y is y = 10/x -/
theorem rectangle_area_relation (x y : ℝ) (h : x * y = 10) : y = 10 / x := by
  sorry

end rectangle_area_relation_l1282_128241
