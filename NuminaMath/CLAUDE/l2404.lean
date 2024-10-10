import Mathlib

namespace percent_equation_solution_l2404_240490

theorem percent_equation_solution :
  ∃ x : ℝ, (0.75 / 100) * x = 0.06 ∧ x = 8 := by sorry

end percent_equation_solution_l2404_240490


namespace ship_cargo_theorem_l2404_240470

def initial_cargo : ℕ := 5973
def loaded_cargo : ℕ := 8723

theorem ship_cargo_theorem : 
  initial_cargo + loaded_cargo = 14696 := by sorry

end ship_cargo_theorem_l2404_240470


namespace second_price_increase_l2404_240430

/-- Given an initial price increase of 20% followed by a second price increase,
    if the total price increase is 38%, then the second price increase is 15%. -/
theorem second_price_increase (P : ℝ) (x : ℝ) 
  (h1 : P > 0)
  (h2 : 1.20 * P * (1 + x / 100) = 1.38 * P) : 
  x = 15 := by
  sorry

end second_price_increase_l2404_240430


namespace ralphs_cards_l2404_240426

/-- 
Given that Ralph initially collected some cards and his father gave him additional cards,
this theorem proves the total number of cards Ralph has.
-/
theorem ralphs_cards (initial_cards additional_cards : ℕ) 
  (h1 : initial_cards = 4)
  (h2 : additional_cards = 8) : 
  initial_cards + additional_cards = 12 := by
  sorry

end ralphs_cards_l2404_240426


namespace odd_squares_sum_representation_l2404_240475

theorem odd_squares_sum_representation (k n : ℕ) (h : k ≠ n) :
  ((2 * k + 1)^2 + (2 * n + 1)^2) / 2 = (k + n + 1)^2 + (k - n)^2 := by
  sorry

end odd_squares_sum_representation_l2404_240475


namespace max_sum_of_product_48_l2404_240427

theorem max_sum_of_product_48 :
  (∃ (x y : ℕ+), x * y = 48 ∧ x + y = 49) ∧
  (∀ (a b : ℕ+), a * b = 48 → a + b ≤ 49) := by
  sorry

end max_sum_of_product_48_l2404_240427


namespace average_change_after_removal_l2404_240482

def average_after_removal (n : ℕ) (initial_avg : ℚ) (removed1 removed2 : ℚ) : ℚ :=
  ((n : ℚ) * initial_avg - removed1 - removed2) / ((n - 2) : ℚ)

theorem average_change_after_removal :
  average_after_removal 50 38 45 55 = 37.5 := by
  sorry

end average_change_after_removal_l2404_240482


namespace expected_successes_bernoulli_l2404_240480

/-- The expected number of successes in 2N Bernoulli trials with p = 0.5 is N -/
theorem expected_successes_bernoulli (N : ℕ) : 
  let n := 2 * N
  let p := (1 : ℝ) / 2
  n * p = N := by sorry

end expected_successes_bernoulli_l2404_240480


namespace angle_sum_theorem_l2404_240408

theorem angle_sum_theorem (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  (1 + Real.tan α) * (1 + Real.tan β) = 2 →
  α + β = π/4 := by
sorry

end angle_sum_theorem_l2404_240408


namespace three_lines_common_points_l2404_240402

/-- A line in 3D space --/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne : point1 ≠ point2

/-- The number of common points of three lines in 3D space --/
def commonPointCount (l1 l2 l3 : Line3D) : Nat :=
  sorry

/-- Three lines determine three planes --/
def determineThreePlanes (l1 l2 l3 : Line3D) : Prop :=
  sorry

theorem three_lines_common_points 
  (l1 l2 l3 : Line3D) 
  (h : determineThreePlanes l1 l2 l3) : 
  commonPointCount l1 l2 l3 = 0 ∨ commonPointCount l1 l2 l3 = 1 :=
sorry

end three_lines_common_points_l2404_240402


namespace fifth_score_calculation_l2404_240401

theorem fifth_score_calculation (s1 s2 s3 s4 : ℕ) (avg : ℚ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 82) (h5 : avg = 75) :
  ∃ (s5 : ℕ), (s1 + s2 + s3 + s4 + s5) / 5 = avg ∧ s5 = 85 := by
  sorry

end fifth_score_calculation_l2404_240401


namespace quadratic_inequality_properties_l2404_240483

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

-- Define the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = Set.Ioo (-1) 2) :
  a < 0 ∧
  a + b + c > 0 ∧
  solution_set b c (3*a) = Set.Iic (-3) ∪ Set.Ioi 1 :=
sorry

end quadratic_inequality_properties_l2404_240483


namespace zero_not_in_range_of_g_l2404_240403

-- Define the function g
noncomputable def g : ℝ → ℤ
| x => if x > -3 then Int.ceil (1 / (x + 3))
       else if x < -3 then Int.floor (1 / (x + 3))
       else 0  -- This value doesn't matter as g is undefined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by
  sorry

end zero_not_in_range_of_g_l2404_240403


namespace second_party_amount_2000_4_16_l2404_240449

/-- Calculates the amount received by the second party in a two-party division given a total amount and a ratio --/
def calculate_second_party_amount (total : ℕ) (ratio1 : ℕ) (ratio2 : ℕ) : ℕ :=
  let total_parts := ratio1 + ratio2
  let part_value := total / total_parts
  ratio2 * part_value

theorem second_party_amount_2000_4_16 :
  calculate_second_party_amount 2000 4 16 = 1600 := by
  sorry

end second_party_amount_2000_4_16_l2404_240449


namespace smallest_angle_in_special_right_triangle_l2404_240439

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The statement of the problem -/
theorem smallest_angle_in_special_right_triangle :
  ∀ a b : ℕ,
    a + b = 90 →
    a > b →
    isPrime a →
    isPrime b →
    isPrime (a - b) →
    b ≥ 17 :=
by sorry

end smallest_angle_in_special_right_triangle_l2404_240439


namespace shelves_used_l2404_240441

theorem shelves_used (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 130 → books_sold = 47 → books_per_shelf = 15 →
  (initial_stock - books_sold + books_per_shelf - 1) / books_per_shelf = 6 := by
  sorry

#eval (130 - 47 + 15 - 1) / 15

end shelves_used_l2404_240441


namespace cos_beta_equals_cos_alpha_l2404_240460

-- Define the angles α and β
variable (α β : Real)

-- Define the conditions
axiom vertices_at_origin : True  -- This condition is implicit in the angle definitions
axiom initial_sides_on_x_axis : True  -- This condition is implicit in the angle definitions
axiom terminal_sides_symmetric : β = 2 * Real.pi - α
axiom cos_alpha : Real.cos α = 2/3

-- Theorem to prove
theorem cos_beta_equals_cos_alpha : Real.cos β = 2/3 := by
  sorry

end cos_beta_equals_cos_alpha_l2404_240460


namespace counterexample_condition_counterexample_existence_l2404_240479

def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def IsPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k ∧ n ≠ 0

theorem counterexample_condition (n : ℕ) : Prop :=
  n > 5 ∧
  ¬(n % 3 = 0) ∧
  ¬(∃ (p q : ℕ), IsPrime p ∧ IsPowerOfTwo q ∧ n = p + q)

theorem counterexample_existence : 
  (∃ n : ℕ, counterexample_condition n) →
  ¬(∀ n : ℕ, n > 5 → ¬(n % 3 = 0) → 
    ∃ (p q : ℕ), IsPrime p ∧ IsPowerOfTwo q ∧ n = p + q) :=
by
  sorry

end counterexample_condition_counterexample_existence_l2404_240479


namespace ratio_equals_three_tenths_l2404_240448

-- Define the system of equations
def system (k x y z w : ℝ) : Prop :=
  x + 2*k*y + 4*z - w = 0 ∧
  4*x + k*y + 2*z + w = 0 ∧
  3*x + 5*y - 3*z + 2*w = 0 ∧
  2*x + 3*y + z - 4*w = 0

-- Theorem statement
theorem ratio_equals_three_tenths :
  ∃ (k x y z w : ℝ), 
    system k x y z w ∧ 
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 ∧
    x * y / (z * w) = 3 / 10 := by
  sorry

end ratio_equals_three_tenths_l2404_240448


namespace congruence_solution_l2404_240429

theorem congruence_solution (n : ℤ) : 
  (15 * n) % 47 = 9 ↔ n % 47 = 18 := by sorry

end congruence_solution_l2404_240429


namespace savings_calculation_l2404_240484

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem: Given an income of 18000 and an income-to-expenditure ratio of 5:4, the savings are 3600 --/
theorem savings_calculation :
  calculate_savings 18000 5 4 = 3600 := by
  sorry

end savings_calculation_l2404_240484


namespace range_of_f_l2404_240443

def f (x : ℤ) : ℤ := x^2 + 2*x

def domain : Set ℤ := {x | -2 ≤ x ∧ x ≤ 1}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} :=
sorry

end range_of_f_l2404_240443


namespace min_framing_for_enlarged_picture_l2404_240468

/-- Calculates the minimum number of linear feet of framing needed for an enlarged picture with a border. -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (final_width + final_height)
  let perimeter_feet := (perimeter_inches + 11) / 12  -- Round up to nearest foot
  perimeter_feet

/-- The theorem states that for a 4-inch by 6-inch picture enlarged by quadrupling its dimensions
    and adding a 3-inch border on each side, the minimum number of linear feet of framing needed is 9. -/
theorem min_framing_for_enlarged_picture :
  min_framing_feet 4 6 4 3 = 9 := by
  sorry

end min_framing_for_enlarged_picture_l2404_240468


namespace james_run_duration_l2404_240456

/-- Calculates the duration of James' run in minutes -/
def run_duration (bags : ℕ) (oz_per_bag : ℕ) (cal_per_oz : ℕ) (cal_per_min : ℕ) (excess_cal : ℕ) : ℕ :=
  let total_oz := bags * oz_per_bag
  let total_cal := total_oz * cal_per_oz
  let cal_to_burn := total_cal - excess_cal
  cal_to_burn / cal_per_min

/-- Proves that James' run duration is 40 minutes given the problem conditions -/
theorem james_run_duration :
  run_duration 3 2 150 12 420 = 40 := by
  sorry

end james_run_duration_l2404_240456


namespace max_triangle_side_length_l2404_240476

theorem max_triangle_side_length (a b c : ℕ) : 
  a < b ∧ b < c ∧                -- Three different side lengths
  a + b + c = 24 ∧               -- Perimeter is 24
  a + b > c ∧ a + c > b ∧ b + c > a →  -- Triangle inequality
  c ≤ 11 :=
by sorry

end max_triangle_side_length_l2404_240476


namespace reflection_composition_maps_points_l2404_240451

-- Define points in 2D space
def Point := ℝ × ℝ

-- Define reflection operations
def reflectY (p : Point) : Point :=
  (-p.1, p.2)

def reflectX (p : Point) : Point :=
  (p.1, -p.2)

-- Define the composition of reflections
def reflectYX (p : Point) : Point :=
  reflectX (reflectY p)

-- Theorem statement
theorem reflection_composition_maps_points :
  let C : Point := (3, -2)
  let D : Point := (4, -5)
  let C' : Point := (-3, 2)
  let D' : Point := (-4, 5)
  reflectYX C = C' ∧ reflectYX D = D' := by sorry

end reflection_composition_maps_points_l2404_240451


namespace algebraic_expression_equality_l2404_240424

theorem algebraic_expression_equality (x : ℝ) (h : x * (x + 2) = 2023) :
  2 * (x + 3) * (x - 1) - 2018 = 2022 := by
  sorry

end algebraic_expression_equality_l2404_240424


namespace correct_percentage_calculation_l2404_240459

theorem correct_percentage_calculation (x : ℝ) (h : x > 0) :
  let total_problems := 7 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  (correct_problems / total_problems) * 100 = (5 / 7) * 100 := by
sorry

end correct_percentage_calculation_l2404_240459


namespace at_least_one_female_selection_l2404_240419

-- Define the total number of athletes
def total_athletes : ℕ := 10

-- Define the number of male athletes
def male_athletes : ℕ := 6

-- Define the number of female athletes
def female_athletes : ℕ := 4

-- Define the number of athletes to be selected
def selected_athletes : ℕ := 5

-- Theorem statement
theorem at_least_one_female_selection :
  (Nat.choose total_athletes selected_athletes) - (Nat.choose male_athletes selected_athletes) = 246 :=
sorry

end at_least_one_female_selection_l2404_240419


namespace yellow_tint_percentage_l2404_240450

/-- Calculates the percentage of yellow tint in a new mixture after adding more yellow tint -/
theorem yellow_tint_percentage 
  (initial_volume : ℝ)
  (initial_yellow_percentage : ℝ)
  (added_yellow : ℝ) :
  initial_volume = 50 →
  initial_yellow_percentage = 25 →
  added_yellow = 10 →
  let initial_yellow := initial_volume * (initial_yellow_percentage / 100)
  let new_yellow := initial_yellow + added_yellow
  let new_volume := initial_volume + added_yellow
  (new_yellow / new_volume) * 100 = 37.5 := by
sorry


end yellow_tint_percentage_l2404_240450


namespace freshman_percentage_l2404_240433

-- Define the total number of students
variable (T : ℝ)
-- Define the fraction of freshmen (to be proven)
variable (F : ℝ)

-- Conditions from the problem
axiom liberal_arts : F * T * 0.5 = T * 0.1 / 0.5

-- Theorem to prove
theorem freshman_percentage : F = 0.4 := by
  sorry

end freshman_percentage_l2404_240433


namespace min_sum_squares_l2404_240416

theorem min_sum_squares (x y z : ℝ) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 8 → 
  ∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → 
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧ 
  ∃ p q r : ℝ, p^3 + q^3 + r^3 - 3*p*q*r = 8 ∧ p^2 + q^2 + r^2 = 4 :=
by sorry

end min_sum_squares_l2404_240416


namespace cows_eating_grass_l2404_240491

-- Define the amount of hectares a cow eats per week
def cow_eat_rate : ℚ := 1/2

-- Define the amount of hectares of grass that grows per week
def grass_growth_rate : ℚ := 1/2

-- Define the function that calculates the amount of grass eaten
def grass_eaten (cows : ℕ) (weeks : ℕ) : ℚ :=
  (cows : ℚ) * cow_eat_rate * (weeks : ℚ)

-- Define the function that calculates the amount of grass regrown
def grass_regrown (hectares : ℕ) (weeks : ℕ) : ℚ :=
  (hectares : ℚ) * grass_growth_rate * (weeks : ℚ)

-- Theorem statement
theorem cows_eating_grass (cows : ℕ) : 
  (grass_eaten 3 2 - grass_regrown 2 2 = 2) →
  (grass_eaten 2 4 - grass_regrown 2 4 = 2) →
  (grass_eaten cows 6 - grass_regrown 6 6 = 6) →
  cows = 3 := by
  sorry

end cows_eating_grass_l2404_240491


namespace product_equality_l2404_240445

theorem product_equality (h : 213 * 16 = 3408) : 16 * 21.3 = 340.8 := by
  sorry

end product_equality_l2404_240445


namespace smallest_base_for_27_l2404_240418

theorem smallest_base_for_27 : 
  ∃ (b : ℕ), b = 5 ∧ 
  (∀ (x : ℕ), x < b → ¬(x^2 ≤ 27 ∧ 27 < x^3)) ∧
  (b^2 ≤ 27 ∧ 27 < b^3) := by
sorry

end smallest_base_for_27_l2404_240418


namespace inverse_of_A_cubed_l2404_240407

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = ![![1, 4], ![-2, -7]] →
  (A^3)⁻¹ = ![![41, 140], ![-90, -335]] := by sorry

end inverse_of_A_cubed_l2404_240407


namespace julie_leftover_money_l2404_240474

def bike_cost : ℕ := 2345
def initial_savings : ℕ := 1500
def lawns_to_mow : ℕ := 20
def lawn_pay : ℕ := 20
def newspapers_to_deliver : ℕ := 600
def newspaper_pay : ℚ := 40/100
def dogs_to_walk : ℕ := 24
def dog_walk_pay : ℕ := 15

theorem julie_leftover_money :
  let total_earnings := lawns_to_mow * lawn_pay + 
                        (newspapers_to_deliver : ℚ) * newspaper_pay + 
                        dogs_to_walk * dog_walk_pay
  let total_money := (initial_savings : ℚ) + total_earnings
  let leftover := total_money - bike_cost
  leftover = 155 := by sorry

end julie_leftover_money_l2404_240474


namespace ellipse_equation_l2404_240452

/-- An ellipse with center at the origin, foci on the x-axis, and point P(2, √3) on the ellipse. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a > b
  h4 : a^2 = b^2 + c^2
  h5 : (4 : ℝ) / a^2 + 3 / b^2 = 1

/-- The distances |PF₁|, |F₁F₂|, and |PF₂| form an arithmetic progression. -/
def is_arithmetic_progression (e : Ellipse) : Prop :=
  ∃ (d : ℝ), 2 * e.a = 4 * e.c ∧ e.c > 0

theorem ellipse_equation (e : Ellipse) (h : is_arithmetic_progression e) :
  e.a = 2 * Real.sqrt 2 ∧ e.b = Real.sqrt 6 :=
sorry

end ellipse_equation_l2404_240452


namespace tire_swap_optimal_l2404_240469

/-- Represents the wear rate of a tire in km^(-1) -/
def WearRate := ℝ

/-- Calculates the remaining life of a tire after driving a certain distance -/
def remaining_life (total_life : ℝ) (distance_driven : ℝ) : ℝ :=
  total_life - distance_driven

/-- Theorem: Swapping tires at 9375 km results in simultaneous wear-out -/
theorem tire_swap_optimal (front_life rear_life swap_distance : ℝ)
  (h_front : front_life = 25000)
  (h_rear : rear_life = 15000)
  (h_swap : swap_distance = 9375) :
  remaining_life front_life swap_distance / rear_life =
  remaining_life rear_life swap_distance / front_life := by
  sorry

#check tire_swap_optimal

end tire_swap_optimal_l2404_240469


namespace bill_score_l2404_240489

theorem bill_score (john sue ella bill : ℕ) 
  (h1 : bill = john + 20)
  (h2 : bill * 2 = sue)
  (h3 : ella = bill + john - 10)
  (h4 : bill + john + sue + ella = 250) : 
  bill = 50 := by
sorry

end bill_score_l2404_240489


namespace digit_swap_l2404_240414

theorem digit_swap (x : ℕ) (h : 9 < x ∧ x < 100) : 
  10 * (x % 10) + (x / 10) = 10 * (x % 10) + (x / 10) :=
by
  sorry

#check digit_swap

end digit_swap_l2404_240414


namespace inequality_proof_l2404_240457

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 + 1 / b^2 ≥ 8) ∧ (1 / a + 1 / b + 1 / (a * b) ≥ 8) := by
  sorry

end inequality_proof_l2404_240457


namespace log_expressibility_l2404_240494

-- Define the given logarithm values
noncomputable def log10_5 : ℝ := 0.6990
noncomputable def log10_7 : ℝ := 0.8451

-- Define a function to represent expressibility using given logarithms
def expressible (x : ℝ) : Prop :=
  ∃ (a b c : ℚ), x = a * log10_5 + b * log10_7 + c

-- Theorem statement
theorem log_expressibility :
  (¬ expressible (Real.log 27 / Real.log 10)) ∧
  (¬ expressible (Real.log 21 / Real.log 10)) ∧
  (expressible (Real.log (Real.sqrt 35) / Real.log 10)) ∧
  (expressible (Real.log 1000 / Real.log 10)) ∧
  (expressible (Real.log 0.2 / Real.log 10)) :=
sorry

end log_expressibility_l2404_240494


namespace simplify_expression_l2404_240431

theorem simplify_expression : (5^8 + 3^7)*(0^5 - (-1)^5)^10 = 392812 := by
  sorry

end simplify_expression_l2404_240431


namespace keith_grew_six_turnips_l2404_240406

/-- The number of turnips Alyssa grew -/
def alyssas_turnips : ℕ := 9

/-- The total number of turnips Keith and Alyssa grew together -/
def total_turnips : ℕ := 15

/-- The number of turnips Keith grew -/
def keiths_turnips : ℕ := total_turnips - alyssas_turnips

theorem keith_grew_six_turnips : keiths_turnips = 6 := by
  sorry

end keith_grew_six_turnips_l2404_240406


namespace ternary_10201_equals_100_l2404_240415

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem ternary_10201_equals_100 :
  ternary_to_decimal [1, 0, 2, 0, 1] = 100 := by
  sorry

end ternary_10201_equals_100_l2404_240415


namespace initial_average_price_l2404_240444

/-- The price of an apple in cents -/
def apple_price : ℕ := 40

/-- The price of an orange in cents -/
def orange_price : ℕ := 60

/-- The total number of fruits Mary initially selects -/
def total_fruits : ℕ := 10

/-- The number of oranges Mary puts back -/
def oranges_removed : ℕ := 5

/-- The average price of remaining fruits after removing oranges, in cents -/
def remaining_avg_price : ℕ := 48

theorem initial_average_price (a o : ℕ) 
  (h1 : a + o = total_fruits)
  (h2 : (apple_price * a + orange_price * o) / total_fruits = 54)
  (h3 : (apple_price * a + orange_price * (o - oranges_removed)) / (total_fruits - oranges_removed) = remaining_avg_price) :
  (apple_price * a + orange_price * o) / total_fruits = 54 :=
sorry

end initial_average_price_l2404_240444


namespace remainder_of_sum_squares_plus_20_l2404_240498

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem remainder_of_sum_squares_plus_20 : 
  (sum_of_squares 15 + 20) % 13 = 1 := by
  sorry

end remainder_of_sum_squares_plus_20_l2404_240498


namespace apple_problem_l2404_240438

theorem apple_problem (x : ℚ) : 
  (((x / 2 + 10) * 2 / 3 + 2) / 2 + 1 = 12) → x = 40 := by
  sorry

end apple_problem_l2404_240438


namespace reciprocal_of_negative_two_l2404_240412

theorem reciprocal_of_negative_two :
  ∀ x : ℚ, x * (-2) = 1 → x = -1/2 := by
  sorry

end reciprocal_of_negative_two_l2404_240412


namespace comics_bought_l2404_240461

theorem comics_bought (initial_amount remaining_amount cost_per_comic : ℕ) 
  (h1 : initial_amount = 87)
  (h2 : remaining_amount = 55)
  (h3 : cost_per_comic = 4) :
  (initial_amount - remaining_amount) / cost_per_comic = 8 := by
  sorry

end comics_bought_l2404_240461


namespace no_solutions_for_equation_l2404_240499

theorem no_solutions_for_equation : 
  ¬ ∃ (n : ℕ+), (n + 900) / 60 = ⌊Real.sqrt n⌋ := by
sorry

end no_solutions_for_equation_l2404_240499


namespace min_value_and_inequality_l2404_240422

-- Define the function f
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

-- State the theorem
theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 1) 
  (hmin_exists : ∃ x, f a b x = 1) : 
  (2*a + b = 2) ∧ 
  (∀ t : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b ≥ t*a*b) → t ≤ 9/2) :=
sorry

end min_value_and_inequality_l2404_240422


namespace cupcake_business_net_profit_l2404_240462

/-- Calculates the net profit from a cupcake business given the following conditions:
  * Each cupcake costs $0.75 to make
  * First 2 dozen cupcakes burnt and were thrown out
  * Next 2 dozen came out perfectly
  * 5 cupcakes were eaten right away
  * Later made 2 more dozen cupcakes
  * 4 more cupcakes were eaten
  * Remaining cupcakes are sold at $2.00 each
-/
theorem cupcake_business_net_profit :
  let cost_per_cupcake : ℚ := 75 / 100
  let sell_price : ℚ := 2
  let dozen : ℕ := 12
  let burnt_cupcakes : ℕ := 2 * dozen
  let eaten_cupcakes : ℕ := 5 + 4
  let total_cupcakes : ℕ := 6 * dozen
  let remaining_cupcakes : ℕ := total_cupcakes - burnt_cupcakes - eaten_cupcakes
  let revenue : ℚ := remaining_cupcakes * sell_price
  let total_cost : ℚ := total_cupcakes * cost_per_cupcake
  let net_profit : ℚ := revenue - total_cost
  net_profit = 24 := by sorry

end cupcake_business_net_profit_l2404_240462


namespace unique_increasing_function_theorem_l2404_240442

def IncreasingFunction (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, x ≤ y → f x ≤ f y

theorem unique_increasing_function_theorem (f : ℕ+ → ℕ+) 
  (h_increasing : IncreasingFunction f)
  (h_inequality : ∀ x : ℕ+, (f x) * (f (f x)) ≤ x^2) :
  ∀ x : ℕ+, f x = x :=
sorry

end unique_increasing_function_theorem_l2404_240442


namespace magnitude_of_z_l2404_240473

/-- The magnitude of the complex number z = 1 / (2 + i) is equal to √3 / 3 -/
theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := 1 / (2 + i)
  Complex.abs z = Real.sqrt 3 / 3 := by
sorry

end magnitude_of_z_l2404_240473


namespace total_marbles_is_193_l2404_240485

/-- The number of marbles in the jar when Ben, Leo, and Tim combine their marbles. -/
def totalMarbles : ℕ :=
  let benMarbles : ℕ := 56
  let leoMarbles : ℕ := benMarbles + 20
  let timMarbles : ℕ := leoMarbles - 15
  benMarbles + leoMarbles + timMarbles

/-- Theorem stating that the total number of marbles in the jar is 193. -/
theorem total_marbles_is_193 : totalMarbles = 193 := by
  sorry

end total_marbles_is_193_l2404_240485


namespace terminal_side_in_second_quadrant_l2404_240465

theorem terminal_side_in_second_quadrant (α : Real) 
  (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  ∃ x y : Real, x < 0 ∧ y > 0 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) ∧ 
  Real.sin α = y / Real.sqrt (x^2 + y^2) :=
sorry

end terminal_side_in_second_quadrant_l2404_240465


namespace apple_distribution_l2404_240486

/-- The number of apples to be distributed -/
def total_apples : ℕ := 30

/-- The number of people receiving apples -/
def num_people : ℕ := 3

/-- The minimum number of apples each person must receive -/
def min_apples : ℕ := 3

/-- The number of ways to distribute the apples -/
def distribution_ways : ℕ := (total_apples - num_people * min_apples + num_people - 1).choose (num_people - 1)

theorem apple_distribution :
  distribution_ways = 253 := by
  sorry

end apple_distribution_l2404_240486


namespace sqrt_four_squared_five_cubed_divided_by_five_l2404_240467

theorem sqrt_four_squared_five_cubed_divided_by_five (x : ℝ) :
  x = (Real.sqrt (4^2 * 5^3)) / 5 → x = 4 * Real.sqrt 5 := by
  sorry

end sqrt_four_squared_five_cubed_divided_by_five_l2404_240467


namespace dice_product_probability_dice_product_probability_proof_l2404_240481

/-- The probability of obtaining a product of 2 when tossing four standard dice -/
theorem dice_product_probability : ℝ :=
  let n_dice : ℕ := 4
  let dice_sides : ℕ := 6
  let target_product : ℕ := 2
  1 / 324

/-- Proof of the dice product probability theorem -/
theorem dice_product_probability_proof :
  dice_product_probability = 1 / 324 := by
  sorry

end dice_product_probability_dice_product_probability_proof_l2404_240481


namespace infinitely_many_heinersch_triples_l2404_240472

/-- A positive integer is heinersch if it can be written as the sum of a positive square and positive cube. -/
def IsHeinersch (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^3 ∧ a > 0 ∧ b > 0

/-- The main theorem stating the existence of infinitely many heinersch numbers h such that h-1 and h+1 are also heinersch. -/
theorem infinitely_many_heinersch_triples :
  ∀ N : ℕ, ∃ t : ℕ, t > N ∧
    let h := ((9*t^4)^3 - 1) / 2
    IsHeinersch h ∧
    IsHeinersch (h-1) ∧
    IsHeinersch (h+1) := by
  sorry

/-- Helper lemma for the identity used in the proof -/
lemma cube_identity (t : ℕ) :
  (9*t^3 - 1)^3 + (9*t^4 - 3*t)^3 = (9*t^4)^3 - 1 := by
  sorry

end infinitely_many_heinersch_triples_l2404_240472


namespace remainder_x13_plus_1_div_x_minus_1_l2404_240436

theorem remainder_x13_plus_1_div_x_minus_1 (x : ℝ) : (x^13 + 1) % (x - 1) = 2 := by
  sorry

end remainder_x13_plus_1_div_x_minus_1_l2404_240436


namespace min_value_sum_reciprocals_l2404_240437

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_abc : a + b + c = 3) : 
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by
  sorry

end min_value_sum_reciprocals_l2404_240437


namespace mean_of_solutions_l2404_240463

-- Define the polynomial
def f (x : ℝ) := x^3 + 5*x^2 - 14*x

-- Define the set of solutions
def solutions := {x : ℝ | f x = 0}

-- State the theorem
theorem mean_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solutions ∧ s.card = 3 ∧ (s.sum id) / s.card = -5/3 := by
  sorry

end mean_of_solutions_l2404_240463


namespace value_of_A_l2404_240458

/-- Given the values of words and letters, prove the value of A -/
theorem value_of_A (L LEAD DEAL DELL : ℤ) (h1 : L = 15) (h2 : LEAD = 50) (h3 : DEAL = 55) (h4 : DELL = 60) : ∃ A : ℤ, A = 25 := by
  sorry

end value_of_A_l2404_240458


namespace cube_equation_solution_l2404_240435

theorem cube_equation_solution :
  ∃! x : ℝ, (x - 5)^3 = (1/27)⁻¹ :=
by
  -- The unique solution is x = 8
  use 8
  sorry

end cube_equation_solution_l2404_240435


namespace jimmy_change_l2404_240428

def pen_cost : ℕ := 1
def notebook_cost : ℕ := 3
def folder_cost : ℕ := 5
def num_pens : ℕ := 3
def num_notebooks : ℕ := 4
def num_folders : ℕ := 2
def paid_amount : ℕ := 50

def total_cost : ℕ := 
  num_pens * pen_cost + num_notebooks * notebook_cost + num_folders * folder_cost

theorem jimmy_change : paid_amount - total_cost = 25 := by
  sorry

end jimmy_change_l2404_240428


namespace floor_distinctness_iff_range_l2404_240455

variable (N M : ℕ)
variable (a : ℝ)

-- Define the property that floor values of ka are distinct
def distinctFloorMultiples : Prop :=
  ∀ k l, k ≠ l → k ≤ N → l ≤ N → ⌊k * a⌋ ≠ ⌊l * a⌋

-- Define the property that floor values of k/a are distinct
def distinctFloorDivisions : Prop :=
  ∀ k l, k ≠ l → k ≤ M → l ≤ M → ⌊k / a⌋ ≠ ⌊l / a⌋

theorem floor_distinctness_iff_range (hN : N > 1) (hM : M > 1) :
  (distinctFloorMultiples N a ∧ distinctFloorDivisions M a) ↔
  ((N - 1 : ℝ) / N ≤ a ∧ a ≤ M / (M - 1 : ℝ)) :=
sorry

end floor_distinctness_iff_range_l2404_240455


namespace school_sampling_l2404_240464

theorem school_sampling (total_students sample_size : ℕ) 
  (h_total : total_students = 1200)
  (h_sample : sample_size = 200)
  (h_stratified : ∃ (boys girls : ℕ), 
    boys + girls = sample_size ∧ 
    boys = girls + 10 ∧ 
    (boys : ℚ) / total_students = (boys : ℚ) / sample_size) :
  ∃ (school_boys : ℕ), school_boys = 630 ∧ 
    (school_boys : ℚ) / total_students = 
    ((sample_size / 2 + 5) : ℚ) / sample_size :=
by sorry

end school_sampling_l2404_240464


namespace b_worked_nine_days_l2404_240466

/-- The number of days worked by person a -/
def days_a : ℕ := 6

/-- The number of days worked by person c -/
def days_c : ℕ := 4

/-- The daily wage of person c in dollars -/
def wage_c : ℕ := 100

/-- The total earnings of all three persons in dollars -/
def total_earnings : ℕ := 1480

/-- The ratio of daily wages for a, b, and c respectively -/
def wage_ratio : Fin 3 → ℕ
| 0 => 3
| 1 => 4
| 2 => 5

/-- The number of days worked by person b -/
def days_b : ℕ := 9

theorem b_worked_nine_days :
  ∃ (wage_a wage_b : ℕ),
    wage_a = wage_c * wage_ratio 0 / wage_ratio 2 ∧
    wage_b = wage_c * wage_ratio 1 / wage_ratio 2 ∧
    days_a * wage_a + days_b * wage_b + days_c * wage_c = total_earnings :=
by sorry

end b_worked_nine_days_l2404_240466


namespace union_of_A_and_B_l2404_240496

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-1, 3}

theorem union_of_A_and_B : A ∪ B = {-1, 1, 2, 3} := by
  sorry

end union_of_A_and_B_l2404_240496


namespace polynomial_coefficient_b_l2404_240477

-- Define the polynomial Q(x)
def Q (x d b e : ℝ) : ℝ := x^3 + d*x^2 + b*x + e

-- State the theorem
theorem polynomial_coefficient_b (d b e : ℝ) :
  -- Conditions
  (∀ x, Q x d b e = 0 → -d/3 = x) ∧  -- Mean of zeros
  (∀ x y z, Q x d b e = 0 ∧ Q y d b e = 0 ∧ Q z d b e = 0 → x*y*z = -e) ∧  -- Product of zeros
  (-d/3 = 1 + d + b + e) ∧  -- Sum of coefficients equals mean of zeros
  (e = 6)  -- y-intercept is 6
  →
  b = -31 :=
by sorry

end polynomial_coefficient_b_l2404_240477


namespace crystal_beads_cost_l2404_240413

/-- The cost of one set of crystal beads -/
def crystal_cost : ℝ := sorry

/-- The cost of one set of metal beads -/
def metal_cost : ℝ := 10

/-- The number of crystal bead sets Nancy buys -/
def crystal_sets : ℕ := 1

/-- The number of metal bead sets Nancy buys -/
def metal_sets : ℕ := 2

/-- The total amount Nancy spends -/
def total_spent : ℝ := 29

theorem crystal_beads_cost :
  crystal_cost = 9 :=
by
  have h1 : crystal_cost + metal_cost * metal_sets = total_spent := sorry
  sorry

end crystal_beads_cost_l2404_240413


namespace total_weight_moved_proof_l2404_240495

/-- Calculates the total weight moved during three sets of back squat, front squat, and deadlift exercises --/
def total_weight_moved (initial_back_squat : ℝ) (back_squat_increase : ℝ) : ℝ :=
  let updated_back_squat := initial_back_squat + back_squat_increase
  let front_squat_ratio := 0.8
  let deadlift_ratio := 1.2
  let back_squat_increase_ratio := 1.05
  let front_squat_increase_ratio := 1.04
  let deadlift_increase_ratio := 1.03
  let back_squat_performance_ratio := 1.0
  let front_squat_performance_ratio := 0.9
  let deadlift_performance_ratio := 0.85
  let back_squat_reps := 3
  let front_squat_reps := 3
  let deadlift_reps := 2

  let back_squat_set1 := updated_back_squat * back_squat_performance_ratio * back_squat_reps
  let back_squat_set2 := updated_back_squat * back_squat_increase_ratio * back_squat_performance_ratio * back_squat_reps
  let back_squat_set3 := updated_back_squat * back_squat_increase_ratio * back_squat_increase_ratio * back_squat_performance_ratio * back_squat_reps

  let front_squat_base := updated_back_squat * front_squat_ratio
  let front_squat_set1 := front_squat_base * front_squat_performance_ratio * front_squat_reps
  let front_squat_set2 := front_squat_base * front_squat_increase_ratio * front_squat_performance_ratio * front_squat_reps
  let front_squat_set3 := front_squat_base * front_squat_increase_ratio * front_squat_increase_ratio * front_squat_performance_ratio * front_squat_reps

  let deadlift_base := updated_back_squat * deadlift_ratio
  let deadlift_set1 := deadlift_base * deadlift_performance_ratio * deadlift_reps
  let deadlift_set2 := deadlift_base * deadlift_increase_ratio * deadlift_performance_ratio * deadlift_reps
  let deadlift_set3 := deadlift_base * deadlift_increase_ratio * deadlift_increase_ratio * deadlift_performance_ratio * deadlift_reps

  back_squat_set1 + back_squat_set2 + back_squat_set3 +
  front_squat_set1 + front_squat_set2 + front_squat_set3 +
  deadlift_set1 + deadlift_set2 + deadlift_set3

theorem total_weight_moved_proof (initial_back_squat : ℝ) (back_squat_increase : ℝ) :
  initial_back_squat = 200 → back_squat_increase = 50 →
  total_weight_moved initial_back_squat back_squat_increase = 5626.398 := by
  sorry

end total_weight_moved_proof_l2404_240495


namespace flagpole_height_l2404_240411

/-- Represents the height and shadow length of an object -/
structure Object where
  height : ℝ
  shadowLength : ℝ

/-- Given two objects under similar conditions, their height-to-shadow ratios are equal -/
def similarConditions (obj1 obj2 : Object) : Prop :=
  obj1.height / obj1.shadowLength = obj2.height / obj2.shadowLength

theorem flagpole_height
  (flagpole : Object)
  (building : Object)
  (h_flagpole_shadow : flagpole.shadowLength = 45)
  (h_building_height : building.height = 24)
  (h_building_shadow : building.shadowLength = 60)
  (h_similar : similarConditions flagpole building) :
  flagpole.height = 18 := by
  sorry


end flagpole_height_l2404_240411


namespace completing_square_equivalence_l2404_240417

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 :=
by sorry

end completing_square_equivalence_l2404_240417


namespace trajectory_and_tangent_lines_l2404_240440

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the projection line
def projection_line (x : ℝ) : Prop := x = 3

-- Define the point P
def point_P (M N : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  P.1 = M.1 + N.1 - 0 ∧ P.2 = M.2 + N.2 - 0

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define point A
def point_A : ℝ × ℝ := (1, 4)

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 1
def tangent_line_2 (x y : ℝ) : Prop := 3*x + 4*y - 19 = 0

theorem trajectory_and_tangent_lines :
  ∀ (M N P : ℝ × ℝ),
    ellipse M.1 M.2 →
    projection_line N.1 →
    point_P M N P →
    (∀ (x y : ℝ), P = (x, y) → trajectory_E x y) ∧
    (∃ (x y : ℝ), (x, y) = point_A ∧ 
      (tangent_line_1 x ∨ tangent_line_2 x y) ∧
      (∀ (t : ℝ), trajectory_E (x + t) (y + t) → t = 0)) :=
by sorry

end trajectory_and_tangent_lines_l2404_240440


namespace work_completion_time_l2404_240478

theorem work_completion_time (b_time : ℕ) (b_worked : ℕ) (a_remaining : ℕ) : 
  b_time = 15 → b_worked = 10 → a_remaining = 3 → 
  ∃ (a_time : ℕ), a_time = 9 ∧ 
    (b_worked : ℚ) / b_time + a_remaining / a_time = 1 := by
  sorry

end work_completion_time_l2404_240478


namespace randy_spends_two_dollars_per_trip_l2404_240446

/-- Calculates the amount spent per store trip -/
def amount_per_trip (initial_amount final_amount trips_per_month months : ℕ) : ℚ :=
  (initial_amount - final_amount : ℚ) / (trips_per_month * months : ℚ)

/-- Theorem: Randy spends $2 per store trip -/
theorem randy_spends_two_dollars_per_trip :
  amount_per_trip 200 104 4 12 = 2 := by
  sorry

end randy_spends_two_dollars_per_trip_l2404_240446


namespace unique_solution_quadratic_l2404_240492

theorem unique_solution_quadratic (a c : ℝ) : 
  (∃! x, a * x^2 + 30 * x + c = 0) →  -- exactly one solution
  (a + c = 35) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) := by
sorry

end unique_solution_quadratic_l2404_240492


namespace modem_download_time_l2404_240425

theorem modem_download_time (time_a : ℝ) (speed_ratio : ℝ) (time_b : ℝ) : 
  time_a = 25.5 →
  speed_ratio = 0.17 →
  time_b = time_a / speed_ratio →
  time_b = 150 := by
sorry

end modem_download_time_l2404_240425


namespace exists_polyhedron_with_hidden_vertices_l2404_240471

/-- A polyhedron in 3D space -/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  is_valid : True  -- Additional conditions for a valid polyhedron

/-- Checks if a point is outside a polyhedron -/
def is_outside (P : Polyhedron) (Q : Fin 3 → ℝ) : Prop :=
  Q ∉ P.vertices ∧ ∀ f ∈ P.faces, Q ∉ f

/-- Checks if a line segment intersects the interior of a polyhedron -/
def intersects_interior (P : Polyhedron) (A B : Fin 3 → ℝ) : Prop :=
  ∃ C : Fin 3 → ℝ, C ≠ A ∧ C ≠ B ∧ C ∈ P.vertices ∧ 
    ∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = λ i => (1 - t) * A i + t * B i

/-- The main theorem -/
theorem exists_polyhedron_with_hidden_vertices : 
  ∃ (P : Polyhedron) (Q : Fin 3 → ℝ), 
    is_outside P Q ∧ 
    ∀ V ∈ P.vertices, intersects_interior P Q V :=
  sorry

end exists_polyhedron_with_hidden_vertices_l2404_240471


namespace rectangle_area_sum_l2404_240423

theorem rectangle_area_sum (a b : ℤ) (h1 : a > b) (h2 : b > 1) : 
  (2 * (a - b).natAbs * (a + b).natAbs = 50) → a + b = 25 := by
  sorry

end rectangle_area_sum_l2404_240423


namespace final_water_level_approx_34cm_l2404_240405

/-- Represents the properties of a liquid in a cylindrical vessel -/
structure Liquid where
  density : ℝ
  initial_height : ℝ

/-- Represents a system of two connected cylindrical vessels with different liquids -/
structure ConnectedVessels where
  water : Liquid
  oil : Liquid

/-- Calculates the final water level in the first vessel after opening the valve -/
def final_water_level (vessels : ConnectedVessels) : ℝ :=
  sorry

/-- The theorem states that given the initial conditions, the final water level
    will be approximately 34 cm -/
theorem final_water_level_approx_34cm (vessels : ConnectedVessels)
  (h_water_density : vessels.water.density = 1000)
  (h_oil_density : vessels.oil.density = 700)
  (h_initial_height : vessels.water.initial_height = 40 ∧ vessels.oil.initial_height = 40) :
  ∃ ε > 0, |final_water_level vessels - 34| < ε :=
sorry

end final_water_level_approx_34cm_l2404_240405


namespace blocks_in_prism_l2404_240453

/-- The number of unit blocks needed to fill a rectangular prism -/
def num_blocks (length width height : ℕ) : ℕ := length * width * height

/-- The dimensions of the rectangular prism -/
def prism_length : ℕ := 4
def prism_width : ℕ := 3
def prism_height : ℕ := 3

/-- Theorem: The number of 1 cm³ blocks needed to fill the given rectangular prism is 36 -/
theorem blocks_in_prism : 
  num_blocks prism_length prism_width prism_height = 36 := by
  sorry

end blocks_in_prism_l2404_240453


namespace find_divisor_l2404_240487

theorem find_divisor (x d : ℚ) : 
  x = 55 → 
  x / d + 10 = 21 → 
  d = 5 := by sorry

end find_divisor_l2404_240487


namespace larger_number_proof_l2404_240493

/-- Given two positive integers with specific HCF and LCM, prove the larger one is 391 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf_cond : Nat.gcd a b = 23)
  (lcm_cond : Nat.lcm a b = 23 * 13 * 17) :
  max a b = 391 := by
  sorry

end larger_number_proof_l2404_240493


namespace largest_non_representable_integer_l2404_240497

theorem largest_non_representable_integer
  (a b c : ℕ+) 
  (coprime_ab : Nat.Coprime a b)
  (coprime_bc : Nat.Coprime b c)
  (coprime_ca : Nat.Coprime c a) :
  ¬ ∃ (x y z : ℕ), 2 * a * b * c - a * b - b * c - c * a = x * b * c + y * c * a + z * a * b :=
sorry

end largest_non_representable_integer_l2404_240497


namespace marks_lost_per_incorrect_sum_l2404_240400

/-- Given Sandy's quiz results, prove the number of marks lost per incorrect sum --/
theorem marks_lost_per_incorrect_sum :
  ∀ (marks_per_correct : ℕ) 
    (total_attempts : ℕ) 
    (total_marks : ℕ) 
    (correct_sums : ℕ) 
    (marks_lost_per_incorrect : ℕ),
  marks_per_correct = 3 →
  total_attempts = 30 →
  total_marks = 60 →
  correct_sums = 24 →
  marks_lost_per_incorrect * (total_attempts - correct_sums) = 
    marks_per_correct * correct_sums - total_marks →
  marks_lost_per_incorrect = 2 :=
by sorry

end marks_lost_per_incorrect_sum_l2404_240400


namespace binomial_12_11_squared_l2404_240488

theorem binomial_12_11_squared : (Nat.choose 12 11)^2 = 144 := by sorry

end binomial_12_11_squared_l2404_240488


namespace men_per_table_l2404_240421

theorem men_per_table 
  (num_tables : ℕ) 
  (women_per_table : ℕ) 
  (total_customers : ℕ) 
  (h1 : num_tables = 9) 
  (h2 : women_per_table = 7) 
  (h3 : total_customers = 90) : 
  (total_customers - num_tables * women_per_table) / num_tables = 3 := by
sorry

end men_per_table_l2404_240421


namespace max_value_tan_cos_l2404_240409

open Real

theorem max_value_tan_cos (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  ∃ (max : Real), max = 2 * (Real.sqrt ((-9 + Real.sqrt 117) / 2))^3 / 
    Real.sqrt (1 - (Real.sqrt ((-9 + Real.sqrt 117) / 2))^2) ∧
  ∀ (x : Real), 0 < x ∧ x < π/2 → 
    tan (x/2) * (1 - cos x) ≤ max := by sorry

end max_value_tan_cos_l2404_240409


namespace product_bound_l2404_240404

theorem product_bound (m : ℕ) (a : ℕ → ℕ) (h1 : ∀ i, i ∈ Finset.range m → a i > 0)
  (h2 : ∀ i, i ∈ Finset.range m → a i ≠ 10)
  (h3 : (Finset.range m).sum a = 10 * m) :
  ((Finset.range m).prod a) ^ (1 / m : ℝ) ≤ 3 * Real.sqrt 11 := by
  sorry

end product_bound_l2404_240404


namespace optimal_start_time_maximizes_minimum_attention_l2404_240432

/-- Represents the attention index of students during a class -/
noncomputable def attentionIndex (x : ℝ) : ℝ :=
  if x ≤ 8 then 2 * x + 68
  else -1/8 * x^2 + 4 * x + 60

/-- The duration of the class in minutes -/
def classDuration : ℝ := 45

/-- The duration of the key explanation in minutes -/
def keyExplanationDuration : ℝ := 24

/-- The optimal start time for the key explanation -/
def optimalStartTime : ℝ := 4

theorem optimal_start_time_maximizes_minimum_attention :
  ∀ t : ℝ, 0 ≤ t ∧ t + keyExplanationDuration ≤ classDuration →
    (∀ x : ℝ, t ≤ x ∧ x ≤ t + keyExplanationDuration →
      attentionIndex x ≥ min (attentionIndex t) (attentionIndex (t + keyExplanationDuration))) →
    t = optimalStartTime := by sorry


end optimal_start_time_maximizes_minimum_attention_l2404_240432


namespace line_symmetry_l2404_240410

/-- Given two lines in the form y = mx + b, this function checks if they are symmetrical about the x-axis -/
def symmetrical_about_x_axis (m1 b1 m2 b2 : ℝ) : Prop :=
  m1 = -m2 ∧ b1 = -b2

/-- The original line y = 3x - 4 -/
def original_line (x : ℝ) : ℝ := 3 * x - 4

/-- The proposed symmetrical line y = -3x + 4 -/
def symmetrical_line (x : ℝ) : ℝ := -3 * x + 4

theorem line_symmetry :
  symmetrical_about_x_axis 3 (-4) (-3) 4 :=
sorry

end line_symmetry_l2404_240410


namespace leah_coin_value_l2404_240447

/-- Represents the types of coins Leah has --/
inductive Coin
| Penny
| Nickel
| Dime

/-- The value of a coin in cents --/
def coinValue : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10

/-- Leah's coin collection --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  total_coins : pennies + nickels + dimes = 15
  dime_nickel_relation : dimes - 1 = nickels

theorem leah_coin_value (c : CoinCollection) : 
  c.pennies * coinValue Coin.Penny + 
  c.nickels * coinValue Coin.Nickel + 
  c.dimes * coinValue Coin.Dime = 89 := by
  sorry

#check leah_coin_value

end leah_coin_value_l2404_240447


namespace intersection_equals_nonnegative_reals_l2404_240434

-- Define set A
def A : Set ℝ := {x : ℝ | |x| = x}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 + x ≥ 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_nonnegative_reals :
  A_intersect_B = {x : ℝ | x ≥ 0} := by sorry

end intersection_equals_nonnegative_reals_l2404_240434


namespace two_pump_fill_time_l2404_240420

/-- The time taken for two pumps to fill a tank together -/
theorem two_pump_fill_time (small_pump_rate large_pump_rate : ℝ) 
  (h1 : small_pump_rate = 1 / 2)
  (h2 : large_pump_rate = 3)
  (h3 : small_pump_rate > 0)
  (h4 : large_pump_rate > 0) :
  1 / (small_pump_rate + large_pump_rate) = 1 / 3.5 :=
by sorry

end two_pump_fill_time_l2404_240420


namespace fraction_simplification_l2404_240454

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end fraction_simplification_l2404_240454
