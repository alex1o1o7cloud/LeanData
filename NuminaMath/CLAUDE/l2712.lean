import Mathlib

namespace domain_of_g_l2712_271259

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-2) 4

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (-x)

-- Define the domain of g
def domain_g : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem domain_of_g :
  ∀ x, x ∈ domain_g ↔ (x ∈ domain_f ∧ (-x) ∈ domain_f) :=
sorry

end domain_of_g_l2712_271259


namespace sqrt_meaningful_range_l2712_271217

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_range_l2712_271217


namespace sufficient_not_necessary_condition_l2712_271202

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 0 → a^2 + a ≥ 0) ∧ ¬(a^2 + a ≥ 0 → a > 0) := by
  sorry

end sufficient_not_necessary_condition_l2712_271202


namespace other_x_intercept_l2712_271201

/-- A quadratic function with vertex (5, -3) and one x-intercept at (0, 0) -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem other_x_intercept (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 5)^2 - 3) →  -- vertex form
  QuadraticFunction a b c 0 = 0 →                        -- (0, 0) is an x-intercept
  ∃ x, x ≠ 0 ∧ QuadraticFunction a b c x = 0 ∧ x = 10 :=
by sorry

end other_x_intercept_l2712_271201


namespace tangent_line_circle_min_value_l2712_271255

theorem tangent_line_circle_min_value (a b : ℝ) :
  a > 0 →
  b > 0 →
  a^2 + 4*b^2 = 2 →
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a*x + 2*b*y + 2 = 0 ∧ x^2 + y^2 = 2) →
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a'^2 + 4*b'^2 = 2 →
    (∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ a'*x' + 2*b'*y' + 2 = 0 ∧ x'^2 + y'^2 = 2) →
    1/a^2 + 1/b^2 ≤ 1/a'^2 + 1/b'^2) →
  1/a^2 + 1/b^2 = 9/2 :=
by sorry

end tangent_line_circle_min_value_l2712_271255


namespace system_solution_l2712_271207

def solution_set : Set (ℝ × ℝ) :=
  {(-2/Real.sqrt 5, 1/Real.sqrt 5), (-2/Real.sqrt 5, -1/Real.sqrt 5),
   (2/Real.sqrt 5, -1/Real.sqrt 5), (2/Real.sqrt 5, 1/Real.sqrt 5)}

def satisfies_system (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 1 ∧
  16*x^4 - 8*x^2*y^2 + y^4 - 40*x^2 - 10*y^2 + 25 = 0

theorem system_solution :
  ∀ x y : ℝ, satisfies_system x y ↔ (x, y) ∈ solution_set :=
by sorry

end system_solution_l2712_271207


namespace f_continuous_l2712_271247

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + 3*x + 5

-- State the theorem
theorem f_continuous : Continuous f := by sorry

end f_continuous_l2712_271247


namespace salary_increase_l2712_271213

theorem salary_increase (S : ℝ) (h1 : S > 0) : 
  0.08 * (S + S * (10 / 100)) = 1.4667 * (0.06 * S) := by
  sorry

#check salary_increase

end salary_increase_l2712_271213


namespace magnitude_of_z_l2712_271282

/-- The magnitude of the complex number z = (1+i)/i is equal to √2 -/
theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.abs ((1 + i) / i) = Real.sqrt 2 := by
  sorry

end magnitude_of_z_l2712_271282


namespace apple_selling_price_l2712_271278

/-- The selling price of an apple, given its cost price and loss ratio. -/
def selling_price (cost_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  cost_price * (1 - loss_ratio)

/-- Theorem stating that the selling price of an apple is 15,
    given a cost price of 18 and a loss ratio of 1/6. -/
theorem apple_selling_price :
  selling_price 18 (1/6) = 15 := by
  sorry

end apple_selling_price_l2712_271278


namespace power_of_power_equals_base_l2712_271295

theorem power_of_power_equals_base (x : ℝ) (h : x > 0) : (x^(4/5))^(5/4) = x := by
  sorry

end power_of_power_equals_base_l2712_271295


namespace james_carrot_sticks_l2712_271276

/-- The number of carrot sticks James ate before dinner -/
def before_dinner : ℕ := 22

/-- The number of carrot sticks James ate after dinner -/
def after_dinner : ℕ := 15

/-- The total number of carrot sticks James ate -/
def total_carrot_sticks : ℕ := before_dinner + after_dinner

theorem james_carrot_sticks : total_carrot_sticks = 37 := by
  sorry

end james_carrot_sticks_l2712_271276


namespace parabola_properties_l2712_271231

/-- Parabola properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_c_neg : c < 0
  h_n_ge_3 : ∃ (n : ℝ), n ≥ 3 ∧ a * n^2 + b * n + c = 0
  h_passes_1_1 : a + b + c = 1
  h_passes_m_0 : ∃ (m : ℝ), a * m^2 + b * m + c = 0

/-- Main theorem -/
theorem parabola_properties (p : Parabola) :
  (p.b > 0) ∧
  (4 * p.a * p.c - p.b^2 < 4 * p.a) ∧
  (∀ (t : ℝ), p.a * 2^2 + p.b * 2 + p.c = t → t > 1) ∧
  (∃ (x : ℝ), p.a * x^2 + p.b * x + p.c = x ∧
    (∃ (m : ℝ), p.a * m^2 + p.b * m + p.c = 0 ∧ 0 < m ∧ m ≤ 1/3)) :=
by sorry

end parabola_properties_l2712_271231


namespace scout_profit_is_250_l2712_271211

/-- Calculates the profit for a scout troop selling candy bars -/
def scout_profit (num_bars : ℕ) (buy_rate : ℚ) (sell_rate : ℚ) : ℚ :=
  let cost_per_bar := 3 / (6 : ℚ)
  let sell_per_bar := 2 / (3 : ℚ)
  let total_cost := (num_bars : ℚ) * cost_per_bar
  let total_revenue := (num_bars : ℚ) * sell_per_bar
  total_revenue - total_cost

/-- The profit for a scout troop selling 1500 candy bars is $250 -/
theorem scout_profit_is_250 :
  scout_profit 1500 (3/6) (2/3) = 250 := by
  sorry

end scout_profit_is_250_l2712_271211


namespace trapezoid_longer_base_l2712_271212

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  midline : ℝ
  midline_difference : ℝ
  longer_base : ℝ
  shorter_base : ℝ

/-- The theorem stating the properties of the specific trapezoid -/
theorem trapezoid_longer_base 
  (t : Trapezoid) 
  (h1 : t.midline = 10)
  (h2 : t.midline_difference = 3)
  (h3 : t.midline = (t.longer_base + t.shorter_base) / 2)
  (h4 : t.midline_difference = (t.longer_base - t.shorter_base) / 2) :
  t.longer_base = 13 := by
    sorry


end trapezoid_longer_base_l2712_271212


namespace license_plate_palindrome_probability_l2712_271289

/-- Probability of a palindrome in a four-letter sequence -/
def prob_letter_palindrome : ℚ := 1 / 676

/-- Probability of a palindrome in a four-digit sequence -/
def prob_digit_palindrome : ℚ := 1 / 100

/-- Total number of possible license plate arrangements -/
def total_arrangements : ℕ := 26^4 * 10^4

theorem license_plate_palindrome_probability :
  let prob_at_least_one_palindrome := prob_letter_palindrome + prob_digit_palindrome - 
                                      (prob_letter_palindrome * prob_digit_palindrome)
  prob_at_least_one_palindrome = 775 / 67600 := by
  sorry

end license_plate_palindrome_probability_l2712_271289


namespace solve_linear_equation_l2712_271223

theorem solve_linear_equation (x : ℝ) (h : 3*x - 5*x + 7*x = 150) : x = 30 := by
  sorry

end solve_linear_equation_l2712_271223


namespace oranges_remaining_l2712_271233

/-- The number of oranges Michaela needs to get full -/
def michaela_oranges : ℕ := 20

/-- The number of oranges Cassandra needs to get full -/
def cassandra_oranges : ℕ := 2 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 90

/-- The number of oranges remaining after Michaela and Cassandra have eaten until they are full -/
theorem oranges_remaining : total_oranges - (michaela_oranges + cassandra_oranges) = 30 := by
  sorry

end oranges_remaining_l2712_271233


namespace larger_number_problem_l2712_271210

theorem larger_number_problem (x y : ℝ) : 
  y = 2 * x - 3 → x + y = 51 → max x y = 33 := by
  sorry

end larger_number_problem_l2712_271210


namespace game_time_calculation_l2712_271268

/-- Calculates the total time before playing a game given download, installation, and tutorial times. -/
def totalGameTime (downloadTime : ℕ) : ℕ :=
  let installTime := downloadTime / 2
  let combinedTime := downloadTime + installTime
  let tutorialTime := 3 * combinedTime
  combinedTime + tutorialTime

/-- Theorem stating that for a download time of 10 minutes, the total time before playing is 60 minutes. -/
theorem game_time_calculation :
  totalGameTime 10 = 60 := by
  sorry

end game_time_calculation_l2712_271268


namespace sqrt_difference_equals_2sqrt3_l2712_271237

theorem sqrt_difference_equals_2sqrt3 : 
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end sqrt_difference_equals_2sqrt3_l2712_271237


namespace expression_evaluation_l2712_271250

theorem expression_evaluation : 2^3 + 4 * 5 - Real.sqrt 9 + (3^2 * 2) / 3 = 31 := by
  sorry

end expression_evaluation_l2712_271250


namespace expression_evaluation_l2712_271248

theorem expression_evaluation (m : ℝ) (h : m = 2) : 
  (m^2 - 9) / (m^2 - 6*m + 9) / (1 - 2/(m-3)) = -5/3 := by sorry

end expression_evaluation_l2712_271248


namespace smallest_cube_divisor_l2712_271293

theorem smallest_cube_divisor (p q r s : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → Nat.Prime s →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  (∀ m : ℕ, m > 0 → m^3 % (p * q^2 * r^4 * s^3) = 0 → m ≥ p * q * r^2 * s) ∧
  (p * q * r^2 * s)^3 % (p * q^2 * r^4 * s^3) = 0 :=
by sorry

end smallest_cube_divisor_l2712_271293


namespace compare_exponential_and_quadratic_l2712_271216

theorem compare_exponential_and_quadratic (n : ℕ) :
  (n ≥ 3 → 2^(2*n) > (2*n + 1)^2) ∧
  ((n = 1 ∨ n = 2) → 2^(2*n) < (2*n + 1)^2) := by
  sorry

end compare_exponential_and_quadratic_l2712_271216


namespace quadratic_equation_solution_l2712_271292

theorem quadratic_equation_solution :
  let x₁ : ℝ := (3 + Real.sqrt 3) / 2
  let x₂ : ℝ := (3 - Real.sqrt 3) / 2
  2 * x₁^2 - 6 * x₁ + 3 = 0 ∧ 2 * x₂^2 - 6 * x₂ + 3 = 0 :=
by sorry

end quadratic_equation_solution_l2712_271292


namespace range_of_a_l2712_271269

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, StrictMono (fun x => (3 - 2*a)^x)
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 < x^2 + 2*a*x + 4

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  a ≤ -2 ∨ (1 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l2712_271269


namespace three_prime_divisors_l2712_271299

theorem three_prime_divisors (p : Nat) (h_prime : Prime p) 
  (h_cong : (2^(p-1)) % (p^2) = 1) (n : Nat) : 
  ∃ (q₁ q₂ q₃ : Nat), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ 
  q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧
  (q₁ ∣ ((p-1) * (Nat.factorial p + 2^n))) ∧
  (q₂ ∣ ((p-1) * (Nat.factorial p + 2^n))) ∧
  (q₃ ∣ ((p-1) * (Nat.factorial p + 2^n))) := by
sorry

end three_prime_divisors_l2712_271299


namespace sum_of_fractions_theorem_l2712_271256

variable (a b c P Q : ℝ)

theorem sum_of_fractions_theorem (h1 : a + b + c = 0) 
  (h2 : a^2 / (2*a^2 + b*c) + b^2 / (2*b^2 + a*c) + c^2 / (2*c^2 + a*b) = P - 3*Q) : 
  Q = 8 := by sorry

end sum_of_fractions_theorem_l2712_271256


namespace total_goats_l2712_271252

def washington_herd : ℕ := 5000
def paddington_difference : ℕ := 220

theorem total_goats : washington_herd + (washington_herd + paddington_difference) = 10220 := by
  sorry

end total_goats_l2712_271252


namespace committee_with_female_count_l2712_271215

def total_members : ℕ := 30
def female_members : ℕ := 12
def male_members : ℕ := 18
def committee_size : ℕ := 5

theorem committee_with_female_count :
  (Nat.choose total_members committee_size) - (Nat.choose male_members committee_size) = 133938 :=
by sorry

end committee_with_female_count_l2712_271215


namespace marblesPerJar_eq_five_l2712_271230

/-- The number of marbles in each jar, given the conditions of the problem -/
def marblesPerJar : ℕ :=
  let numJars : ℕ := 16
  let numPots : ℕ := numJars / 2
  let totalMarbles : ℕ := 200
  let marblesPerPot : ℕ → ℕ := fun x ↦ 3 * x
  (totalMarbles / (numJars + numPots * 3))

theorem marblesPerJar_eq_five : marblesPerJar = 5 := by
  sorry

end marblesPerJar_eq_five_l2712_271230


namespace square_of_101_l2712_271275

theorem square_of_101 : 101 * 101 = 10201 := by
  sorry

end square_of_101_l2712_271275


namespace midpoint_coordinate_product_l2712_271235

theorem midpoint_coordinate_product (p1 p2 : ℝ × ℝ) :
  let m := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  p1 = (10, -3) → p2 = (-4, 9) → m.1 * m.2 = 9 := by
sorry

end midpoint_coordinate_product_l2712_271235


namespace correct_num_arrangements_l2712_271219

/-- The number of different arrangements for 7 students in a row,
    where one student must stand in the center and two other students must stand together. -/
def num_arrangements : ℕ := 192

/-- The number of students -/
def total_students : ℕ := 7

/-- The number of students that must stand together (excluding the center student) -/
def students_together : ℕ := 2

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_num_arrangements :
  num_arrangements = 
    2 * (Nat.factorial students_together) * 
    (Nat.choose (total_students - 3) 1) * 
    (Nat.factorial 2) * 
    (Nat.factorial 3) :=
by
  sorry


end correct_num_arrangements_l2712_271219


namespace three_cell_corners_count_l2712_271222

theorem three_cell_corners_count (total_cells : ℕ) (x y : ℕ) : 
  total_cells = 22 → 
  3 * x + 4 * y = total_cells → 
  (x = 2 ∨ x = 6) ∧ (y = 4 ∨ y = 1) :=
sorry

end three_cell_corners_count_l2712_271222


namespace semi_annual_compound_interest_rate_l2712_271298

/-- Proves that the annual interest rate of a semi-annually compounded account is approximately 7.96%
    given specific conditions on the initial investment and interest earned. -/
theorem semi_annual_compound_interest_rate (principal : ℝ) (simple_rate : ℝ) (diff : ℝ) :
  principal = 5000 →
  simple_rate = 0.08 →
  diff = 6 →
  ∃ (compound_rate : ℝ),
    (principal * (1 + compound_rate / 2)^2 - principal) = 
    (principal * simple_rate + diff) ∧
    abs (compound_rate - 0.0796) < 0.0001 := by
  sorry

end semi_annual_compound_interest_rate_l2712_271298


namespace village_panic_percentage_l2712_271243

theorem village_panic_percentage (original_population : ℕ) 
  (initial_disappearance_rate : ℚ) (final_population : ℕ) :
  original_population = 7200 →
  initial_disappearance_rate = 1/10 →
  final_population = 4860 →
  (1 - (final_population : ℚ) / ((1 - initial_disappearance_rate) * original_population)) * 100 = 25 := by
  sorry

end village_panic_percentage_l2712_271243


namespace salt_solution_concentration_l2712_271266

/-- Given a mixture of pure water and salt solution, prove the original salt solution concentration -/
theorem salt_solution_concentration 
  (pure_water_volume : ℝ) 
  (salt_solution_volume : ℝ) 
  (final_mixture_concentration : ℝ) 
  (h1 : pure_water_volume = 1)
  (h2 : salt_solution_volume = 0.5)
  (h3 : final_mixture_concentration = 15) :
  let total_volume := pure_water_volume + salt_solution_volume
  let salt_amount := final_mixture_concentration / 100 * total_volume
  salt_amount / salt_solution_volume * 100 = 45 := by
  sorry

end salt_solution_concentration_l2712_271266


namespace essay_competition_probability_l2712_271262

theorem essay_competition_probability (n : ℕ) (h : n = 6) :
  let p := (n - 1) / n
  p = 5 / 6 := by
  sorry

end essay_competition_probability_l2712_271262


namespace product_97_squared_l2712_271232

theorem product_97_squared : 97 * 97 = 9409 := by
  sorry

end product_97_squared_l2712_271232


namespace bea_lemonade_sales_l2712_271257

theorem bea_lemonade_sales (bea_price dawn_price : ℚ) (dawn_sales : ℕ) (extra_earnings : ℚ) :
  bea_price = 25/100 →
  dawn_price = 28/100 →
  dawn_sales = 8 →
  extra_earnings = 26/100 →
  ∃ bea_sales : ℕ, bea_sales * bea_price = dawn_sales * dawn_price + extra_earnings ∧ bea_sales = 10 :=
by
  sorry

end bea_lemonade_sales_l2712_271257


namespace f_greater_than_one_f_monotonicity_f_non_negative_iff_l2712_271254

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (Real.exp x) / (x^2) - k * x + 2 * k * Real.log x

-- State the theorems to be proved
theorem f_greater_than_one (x : ℝ) (hx : x > 0) : f 0 x > 1 := by sorry

theorem f_monotonicity (x : ℝ) (hx : x > 0) :
  (x > 2 → (∀ y > x, f 1 y > f 1 x)) ∧
  (x < 2 → (∀ y ∈ Set.Ioo 0 x, f 1 y > f 1 x)) := by sorry

theorem f_non_negative_iff (k : ℝ) :
  (∀ x > 0, f k x ≥ 0) ↔ k ≤ Real.exp 1 := by sorry

end

end f_greater_than_one_f_monotonicity_f_non_negative_iff_l2712_271254


namespace combined_weight_calculation_l2712_271244

/-- The combined weight of Leo and Kendra -/
def combinedWeight (leoWeight kenWeight : ℝ) : ℝ := leoWeight + kenWeight

/-- Leo's weight after gaining 10 pounds -/
def leoWeightGained (leoWeight : ℝ) : ℝ := leoWeight + 10

/-- Condition that Leo's weight after gaining 10 pounds is 50% more than Kendra's weight -/
def weightCondition (leoWeight kenWeight : ℝ) : Prop :=
  leoWeightGained leoWeight = kenWeight * 1.5

theorem combined_weight_calculation (leoWeight kenWeight : ℝ) 
  (h1 : leoWeight = 98) 
  (h2 : weightCondition leoWeight kenWeight) : 
  combinedWeight leoWeight kenWeight = 170 := by
  sorry

end combined_weight_calculation_l2712_271244


namespace system_solution_l2712_271273

theorem system_solution : ∃ (x y z : ℝ), 
  (x^2 + y - 2*z = -3) ∧ 
  (3*x + y + z^2 = 14) ∧ 
  (7*x - y^2 + 4*z = 25) ∧
  (x = 2 ∧ y = -1 ∧ z = 3) := by
  sorry

end system_solution_l2712_271273


namespace roller_skate_attendance_l2712_271283

/-- The number of wheels on the floor when all people skated -/
def total_wheels : ℕ := 320

/-- The number of roller skates per person -/
def skates_per_person : ℕ := 2

/-- The number of wheels per roller skate -/
def wheels_per_skate : ℕ := 2

/-- The number of people who showed up to roller skate -/
def num_people : ℕ := total_wheels / (skates_per_person * wheels_per_skate)

theorem roller_skate_attendance : num_people = 80 := by
  sorry

end roller_skate_attendance_l2712_271283


namespace three_tangent_lines_l2712_271234

/-- A line that passes through the point (0, 2) and has only one common point with the parabola y^2 = 8x -/
structure TangentLine where
  -- The slope of the line (None if vertical)
  slope : Option ℝ
  -- Condition that the line passes through (0, 2)
  passes_through_point : True
  -- Condition that the line has only one common point with y^2 = 8x
  single_intersection : True

/-- The number of lines passing through (0, 2) with only one common point with y^2 = 8x -/
def count_tangent_lines : ℕ := sorry

/-- Theorem stating that there are exactly 3 such lines -/
theorem three_tangent_lines : count_tangent_lines = 3 := by sorry

end three_tangent_lines_l2712_271234


namespace simplify_and_evaluate_l2712_271281

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 - 4) :
  (4 - x) / (x - 2) / (x + 2 - 12 / (x - 2)) = -Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l2712_271281


namespace new_student_weight_new_student_weight_is_46_l2712_271221

/-- The weight of a new student who replaces an 86 kg student in a group of 8,
    resulting in an average weight decrease of 5 kg. -/
theorem new_student_weight : ℝ :=
  let n : ℕ := 8 -- number of students
  let avg_decrease : ℝ := 5 -- average weight decrease in kg
  let replaced_weight : ℝ := 86 -- weight of the replaced student in kg
  replaced_weight - n * avg_decrease

/-- Proof that the new student's weight is 46 kg -/
theorem new_student_weight_is_46 : new_student_weight = 46 := by
  sorry

end new_student_weight_new_student_weight_is_46_l2712_271221


namespace hyperbola_eccentricity_range_l2712_271274

/-- The eccentricity of a hyperbola given its equation and a point in the "up" region -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_up : b / a < 2) : ∃ e : ℝ, 1 < e ∧ e < Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_range_l2712_271274


namespace range_of_f_l2712_271294

noncomputable def f (x : ℝ) : ℝ := 2 * (x + 7) * (x - 5) / (x + 7)

theorem range_of_f :
  Set.range f = {y | y < -24 ∨ y > -24} := by sorry

end range_of_f_l2712_271294


namespace bird_watching_percentage_difference_l2712_271277

def gabrielle_robins : ℕ := 5
def gabrielle_cardinals : ℕ := 4
def gabrielle_blue_jays : ℕ := 3

def chase_robins : ℕ := 2
def chase_blue_jays : ℕ := 3
def chase_cardinals : ℕ := 5

def gabrielle_total : ℕ := gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
def chase_total : ℕ := chase_robins + chase_blue_jays + chase_cardinals

theorem bird_watching_percentage_difference :
  (gabrielle_total - chase_total : ℚ) / chase_total * 100 = 20 := by
  sorry

end bird_watching_percentage_difference_l2712_271277


namespace intersection_of_logarithmic_graphs_l2712_271286

theorem intersection_of_logarithmic_graphs :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * x) := by sorry

end intersection_of_logarithmic_graphs_l2712_271286


namespace hyperbola_sum_l2712_271206

-- Define the hyperbola parameters
def center : ℝ × ℝ := (3, -2)
def focus : ℝ × ℝ := (3, 5)
def vertex : ℝ × ℝ := (3, 0)

-- Define h and k from the center
def h : ℝ := center.1
def k : ℝ := center.2

-- Define a as the distance from center to vertex
def a : ℝ := |center.2 - vertex.2|

-- Define c as the distance from center to focus
def c : ℝ := |center.2 - focus.2|

-- Define b using the relationship c^2 = a^2 + b^2
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

-- Theorem statement
theorem hyperbola_sum : h + k + a + b = 3 + 3 * Real.sqrt 5 := by sorry

end hyperbola_sum_l2712_271206


namespace geometric_sum_ratio_l2712_271296

/-- Given a geometric sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- The ratio of S_5 to S_10 is 1/3 -/
axiom ratio_condition : S 5 / S 10 = 1 / 3

/-- Theorem: If S_5 / S_10 = 1/3, then S_5 / (S_20 + S_10) = 1/18 -/
theorem geometric_sum_ratio : S 5 / (S 20 + S 10) = 1 / 18 := by sorry

end geometric_sum_ratio_l2712_271296


namespace find_boys_in_first_group_l2712_271242

/-- Represents the daily work done by a single person -/
structure WorkRate :=
  (amount : ℝ)

/-- Represents a group of workers -/
structure WorkGroup :=
  (men : ℕ)
  (boys : ℕ)

/-- Represents the time taken to complete a job -/
def completeJob (g : WorkGroup) (d : ℕ) (m : WorkRate) (b : WorkRate) : ℝ :=
  d * (g.men * m.amount + g.boys * b.amount)

theorem find_boys_in_first_group :
  ∀ (m b : WorkRate) (x : ℕ),
    m.amount = 2 * b.amount →
    completeJob ⟨12, x⟩ 5 m b = completeJob ⟨13, 24⟩ 4 m b →
    x = 16 := by
  sorry

end find_boys_in_first_group_l2712_271242


namespace books_read_l2712_271291

theorem books_read (total : ℕ) (unread : ℕ) (h1 : total = 21) (h2 : unread = 8) :
  total - unread = 13 := by
  sorry

end books_read_l2712_271291


namespace smallest_n_for_exact_tax_l2712_271224

theorem smallest_n_for_exact_tax : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬∃ (x : ℕ), x > 0 ∧ 107 * x = 100 * m) ∧
  (∃ (x : ℕ), x > 0 ∧ 107 * x = 100 * n) ∧
  n = 107 := by
  sorry

end smallest_n_for_exact_tax_l2712_271224


namespace oscars_bus_ride_l2712_271258

/-- Oscar's bus ride to school problem -/
theorem oscars_bus_ride (charlie_ride : ℝ) (oscar_difference : ℝ) :
  charlie_ride = 0.25 →
  oscar_difference = 0.5 →
  charlie_ride + oscar_difference = 0.75 := by
  sorry

end oscars_bus_ride_l2712_271258


namespace ilwoong_drive_files_l2712_271220

theorem ilwoong_drive_files (num_folders : ℕ) (subfolders_per_folder : ℕ) (files_per_subfolder : ℕ) :
  num_folders = 25 →
  subfolders_per_folder = 10 →
  files_per_subfolder = 8 →
  num_folders * subfolders_per_folder * files_per_subfolder = 2000 := by
  sorry

end ilwoong_drive_files_l2712_271220


namespace parabola_focus_vertex_ratio_l2712_271239

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- The locus of midpoints of line segments AB on a parabola P where ∠AV₁B = 90° -/
def midpoint_locus (p : Parabola) : Parabola := sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_focus_vertex_ratio :
  let p := Parabola.mk 4 0 0
  let q := midpoint_locus p
  let v1 := vertex p
  let v2 := vertex q
  let f1 := focus p
  let f2 := focus q
  distance f1 f2 / distance v1 v2 = 1/16 := by
  sorry

end parabola_focus_vertex_ratio_l2712_271239


namespace sqrt_floor_impossibility_l2712_271200

theorem sqrt_floor_impossibility (x : ℝ) (h1 : 100 ≤ x ∧ x ≤ 200) (h2 : ⌊Real.sqrt x⌋ = 14) : 
  ⌊Real.sqrt (50 * x)⌋ ≠ 140 := by
sorry

end sqrt_floor_impossibility_l2712_271200


namespace cubic_root_inequality_l2712_271225

/-- Given a cubic polynomial with real coefficients and three real roots,
    prove the inequality involving the difference between the largest and smallest roots. -/
theorem cubic_root_inequality (a b c : ℝ) (α β γ : ℝ) : 
  let p : ℝ → ℝ := λ x => x^3 + a*x^2 + b*x + c
  (∀ x, p x = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  α < β →
  β < γ →
  Real.sqrt (a^2 - 3*b) < γ - α ∧ γ - α ≤ 2 * Real.sqrt ((a^2 / 3) - b) := by
sorry

end cubic_root_inequality_l2712_271225


namespace total_money_l2712_271229

theorem total_money (mark : ℚ) (carolyn : ℚ) (david : ℚ)
  (h1 : mark = 5 / 6)
  (h2 : carolyn = 4 / 9)
  (h3 : david = 7 / 12) :
  mark + carolyn + david = 67 / 36 := by
  sorry

end total_money_l2712_271229


namespace scientific_notation_of_149000000_l2712_271209

theorem scientific_notation_of_149000000 :
  149000000 = 1.49 * (10 : ℝ)^8 :=
by sorry

end scientific_notation_of_149000000_l2712_271209


namespace system_solution_ratio_l2712_271290

theorem system_solution_ratio (x y a b : ℝ) (h1 : 4*x - 3*y = a) (h2 : 6*y - 8*x = b) (h3 : b ≠ 0) :
  a / b = -1 / 2 := by
sorry

end system_solution_ratio_l2712_271290


namespace children_going_to_zoo_l2712_271287

/-- The number of children per seat in the bus -/
def children_per_seat : ℕ := 2

/-- The total number of seats needed in the bus -/
def total_seats : ℕ := 29

/-- The total number of children taking the bus to the zoo -/
def total_children : ℕ := children_per_seat * total_seats

theorem children_going_to_zoo : total_children = 58 := by
  sorry

end children_going_to_zoo_l2712_271287


namespace y_in_terms_of_z_l2712_271288

theorem y_in_terms_of_z (x y z : ℝ) : 
  x = 90 * (1 + 0.11) →
  y = x * (1 - 0.27) →
  z = y/2 + 3 →
  y = 2*z - 6 := by
  sorry

end y_in_terms_of_z_l2712_271288


namespace product_equals_243_l2712_271249

theorem product_equals_243 : 
  (1 / 3 : ℚ) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end product_equals_243_l2712_271249


namespace guard_skipped_circles_l2712_271253

def warehouse_length : ℕ := 600
def warehouse_width : ℕ := 400
def intended_circles : ℕ := 10
def actual_distance : ℕ := 16000

def perimeter : ℕ := 2 * (warehouse_length + warehouse_width)
def intended_distance : ℕ := intended_circles * perimeter
def skipped_distance : ℕ := intended_distance - actual_distance
def times_skipped : ℕ := skipped_distance / perimeter

theorem guard_skipped_circles :
  times_skipped = 2 := by sorry

end guard_skipped_circles_l2712_271253


namespace profit_share_ratio_l2712_271261

/-- Given two investors P and Q with their respective investments, 
    calculate the ratio of their profit shares. -/
theorem profit_share_ratio 
  (p_investment q_investment : ℕ) 
  (h_p : p_investment = 40000) 
  (h_q : q_investment = 60000) : 
  (p_investment : ℚ) / q_investment = 2 / 3 := by
  sorry

end profit_share_ratio_l2712_271261


namespace evaluate_expression_l2712_271203

theorem evaluate_expression (x : ℝ) (h : x = 6) : 
  (x^9 - 24*x^6 + 144*x^3 - 512) / (x^3 - 8) = 43264 := by
  sorry

end evaluate_expression_l2712_271203


namespace sunlight_rice_yield_correlation_l2712_271204

-- Define the concept of a relationship between two variables
def Relationship (X Y : Type) := X → Y → Prop

-- Define functional relationship
def FunctionalRelationship (X Y : Type) (r : Relationship X Y) : Prop :=
  ∀ (x : X), ∃! (y : Y), r x y

-- Define correlation relationship
def CorrelationRelationship (X Y : Type) (r : Relationship X Y) : Prop :=
  ¬FunctionalRelationship X Y r ∧ ∃ (x₁ x₂ : X) (y₁ y₂ : Y), r x₁ y₁ ∧ r x₂ y₂

-- Define the relationships for each option
def CubeVolumeEdgeLength : Relationship ℝ ℝ := sorry
def AngleSine : Relationship ℝ ℝ := sorry
def SunlightRiceYield : Relationship ℝ ℝ := sorry
def HeightVision : Relationship ℝ ℝ := sorry

-- State the theorem
theorem sunlight_rice_yield_correlation :
  CorrelationRelationship ℝ ℝ SunlightRiceYield ∧
  ¬CorrelationRelationship ℝ ℝ CubeVolumeEdgeLength ∧
  ¬CorrelationRelationship ℝ ℝ AngleSine ∧
  ¬CorrelationRelationship ℝ ℝ HeightVision :=
sorry

end sunlight_rice_yield_correlation_l2712_271204


namespace quadratic_inequality_range_l2712_271241

theorem quadratic_inequality_range (α : Real) (h : 0 ≤ α ∧ α ≤ π) :
  (∀ x : Real, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) ↔
  (0 ≤ α ∧ α ≤ π / 6) ∨ (5 * π / 6 ≤ α ∧ α ≤ π) := by
  sorry

end quadratic_inequality_range_l2712_271241


namespace state_fair_revenue_l2712_271227

/-- Represents the revenue calculation for a state fair -/
theorem state_fair_revenue
  (ticket_price : ℝ)
  (total_ticket_revenue : ℝ)
  (food_price : ℝ)
  (ride_price : ℝ)
  (souvenir_price : ℝ)
  (game_price : ℝ)
  (h1 : ticket_price = 8)
  (h2 : total_ticket_revenue = 8000)
  (h3 : food_price = 10)
  (h4 : ride_price = 6)
  (h5 : souvenir_price = 18)
  (h6 : game_price = 5) :
  ∃ (total_revenue : ℝ),
    total_revenue = total_ticket_revenue +
      (3/5 * (total_ticket_revenue / ticket_price) * food_price) +
      (1/3 * (total_ticket_revenue / ticket_price) * ride_price) +
      (1/6 * (total_ticket_revenue / ticket_price) * souvenir_price) +
      (1/10 * (total_ticket_revenue / ticket_price) * game_price) ∧
    total_revenue = 19486 := by
  sorry


end state_fair_revenue_l2712_271227


namespace least_number_with_special_division_property_l2712_271284

theorem least_number_with_special_division_property : ∃ k : ℕ, 
  k > 0 ∧ 
  k / 5 = k % 34 + 8 ∧ 
  (∀ m : ℕ, m > 0 → m / 5 = m % 34 + 8 → k ≤ m) ∧
  k = 68 := by
sorry

end least_number_with_special_division_property_l2712_271284


namespace correct_cube_root_l2712_271267

-- Define the cube root function for real numbers
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Theorem statement
theorem correct_cube_root : cubeRoot (-125) = -5 := by
  sorry

end correct_cube_root_l2712_271267


namespace apple_distribution_l2712_271238

theorem apple_distribution (total_apples : ℕ) (num_friends : ℕ) (apples_per_friend : ℕ) : 
  total_apples = 9 → num_friends = 3 → total_apples / num_friends = apples_per_friend → apples_per_friend = 3 := by
  sorry

end apple_distribution_l2712_271238


namespace certain_number_problem_l2712_271265

theorem certain_number_problem (x : ℝ) : 
  (15 - 2 + x) / 2 * 8 = 77 → x = 6.25 := by
  sorry

end certain_number_problem_l2712_271265


namespace original_people_in_room_l2712_271264

theorem original_people_in_room (x : ℝ) : 
  (x / 2 = 15) → 
  (x / 3 + x / 4 * (2 / 3) + 15 = x) → 
  x = 30 := by
sorry

end original_people_in_room_l2712_271264


namespace geometric_sequence_sum_l2712_271240

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l2712_271240


namespace equilateral_triangle_semi_regular_hexagon_l2712_271272

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a hexagon -/
structure Hexagon :=
  (vertices : Fin 6 → Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Divides each side of a triangle into three equal parts -/
def divideSides (t : Triangle) : Fin 6 → Point := sorry

/-- Forms a hexagon from the division points and opposite vertices -/
def formHexagon (t : Triangle) (divisionPoints : Fin 6 → Point) : Hexagon := sorry

/-- Checks if a hexagon is semi-regular -/
def isSemiRegular (h : Hexagon) : Prop := sorry

/-- Main theorem: The hexagon formed by dividing the sides of an equilateral triangle
    and connecting division points to opposite vertices is semi-regular -/
theorem equilateral_triangle_semi_regular_hexagon 
  (t : Triangle) (h : isEquilateral t) : 
  isSemiRegular (formHexagon t (divideSides t)) := by
  sorry

end equilateral_triangle_semi_regular_hexagon_l2712_271272


namespace smallest_prime_factors_difference_l2712_271270

def number : Nat := 96043

theorem smallest_prime_factors_difference (p q : Nat) : 
  Prime p ∧ Prime q ∧ p ∣ number ∧ q ∣ number ∧
  (∀ r, Prime r → r ∣ number → r ≥ p) ∧
  (∀ r, Prime r → r ∣ number → r = p ∨ r ≥ q) →
  q - p = 4 := by sorry

end smallest_prime_factors_difference_l2712_271270


namespace rent_expenditure_l2712_271245

def monthly_salary : ℕ := 18000
def savings_percentage : ℚ := 1/10
def savings : ℕ := 1800
def milk_expense : ℕ := 1500
def groceries_expense : ℕ := 4500
def education_expense : ℕ := 2500
def petrol_expense : ℕ := 2000
def misc_expense : ℕ := 700

theorem rent_expenditure :
  let total_expenses := milk_expense + groceries_expense + education_expense + petrol_expense + misc_expense
  let rent := monthly_salary - (total_expenses + savings)
  (savings = (savings_percentage * monthly_salary).num) →
  rent = 6000 := by sorry

end rent_expenditure_l2712_271245


namespace smallest_b_for_composite_l2712_271260

theorem smallest_b_for_composite (b : ℕ+) (h : b = 9) :
  (∀ x : ℤ, ∃ a c : ℤ, a > 1 ∧ c > 1 ∧ x^4 + b^2 = a * c) ∧
  (∀ b' : ℕ+, b' < b → ∃ x : ℤ, ∀ a c : ℤ, (a > 1 ∧ c > 1 → x^4 + b'^2 ≠ a * c)) :=
sorry

end smallest_b_for_composite_l2712_271260


namespace quadratic_greater_than_linear_l2712_271280

theorem quadratic_greater_than_linear (x : ℝ) :
  let y₁ : ℝ → ℝ := λ x => x + 1
  let y₂ : ℝ → ℝ := λ x => (1/2) * x^2 - (1/2) * x - 1
  (y₂ x > y₁ x) ↔ (x < -1 ∨ x > 4) := by
  sorry

end quadratic_greater_than_linear_l2712_271280


namespace pencils_with_eraser_count_l2712_271208

/-- The number of pencils with an eraser sold in a stationery store -/
def pencils_with_eraser : ℕ := sorry

/-- The price of a pencil with an eraser -/
def price_eraser : ℚ := 8/10

/-- The price of a regular pencil -/
def price_regular : ℚ := 1/2

/-- The price of a short pencil -/
def price_short : ℚ := 4/10

/-- The number of regular pencils sold -/
def regular_sold : ℕ := 40

/-- The number of short pencils sold -/
def short_sold : ℕ := 35

/-- The total revenue from all pencil sales -/
def total_revenue : ℚ := 194

/-- Theorem stating that the number of pencils with an eraser sold is 200 -/
theorem pencils_with_eraser_count : pencils_with_eraser = 200 :=
  by sorry

end pencils_with_eraser_count_l2712_271208


namespace product_of_roots_l2712_271214

theorem product_of_roots (a b c d : ℂ) : 
  (3 * a^4 - 8 * a^3 + a^2 + 4 * a - 10 = 0) ∧ 
  (3 * b^4 - 8 * b^3 + b^2 + 4 * b - 10 = 0) ∧ 
  (3 * c^4 - 8 * c^3 + c^2 + 4 * c - 10 = 0) ∧ 
  (3 * d^4 - 8 * d^3 + d^2 + 4 * d - 10 = 0) →
  a * b * c * d = -10/3 := by
sorry

end product_of_roots_l2712_271214


namespace smallest_change_l2712_271228

def original : ℚ := 0.123456

def change_digit (n : ℕ) (d : ℕ) : ℚ :=
  if n = 1 then 0.823456
  else if n = 2 then 0.183456
  else if n = 3 then 0.128456
  else if n = 4 then 0.123856
  else if n = 6 then 0.123458
  else original

theorem smallest_change :
  ∀ n : ℕ, n ≠ 6 → change_digit 6 8 < change_digit n 8 :=
by sorry

end smallest_change_l2712_271228


namespace cubic_sum_of_roots_l2712_271285

theorem cubic_sum_of_roots (p q r s : ℝ) : 
  (r^2 - p*r - q = 0) → (s^2 - p*s - q = 0) → (r^3 + s^3 = p^3 + 3*p*q) :=
by sorry

end cubic_sum_of_roots_l2712_271285


namespace xy_value_l2712_271251

theorem xy_value (x y : ℝ) (h_distinct : x ≠ y) 
  (h_eq : x^2 + 2/x^2 = y^2 + 2/y^2) : 
  x * y = Real.sqrt 2 ∨ x * y = -Real.sqrt 2 := by
  sorry

end xy_value_l2712_271251


namespace rationalize_denominator_sqrt343_l2712_271226

theorem rationalize_denominator_sqrt343 : 
  7 / Real.sqrt 343 = Real.sqrt 7 / 7 := by sorry

end rationalize_denominator_sqrt343_l2712_271226


namespace quadratic_polynomial_satisfies_conditions_l2712_271218

theorem quadratic_polynomial_satisfies_conditions :
  ∃ p : ℝ → ℝ,
    (∀ x, p x = x^2 + 1) ∧
    p (-3) = 10 ∧
    p 0 = 1 ∧
    p 2 = 5 := by
  sorry

end quadratic_polynomial_satisfies_conditions_l2712_271218


namespace range_of_a_l2712_271263

/-- Given a function f(x) = x^2 + 2(a-1)x + 2 that is monotonically decreasing on (-∞, 4],
    prove that the range of a is (-∞, -3]. -/
theorem range_of_a (a : ℝ) : 
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → (x^2 + 2*(a-1)*x + 2) > (y^2 + 2*(a-1)*y + 2)) →
  a ∈ Set.Iic (-3 : ℝ) :=
sorry

end range_of_a_l2712_271263


namespace apples_per_pie_l2712_271297

theorem apples_per_pie (initial_apples : Nat) (handed_out : Nat) (num_pies : Nat)
  (h1 : initial_apples = 96)
  (h2 : handed_out = 42)
  (h3 : num_pies = 9) :
  (initial_apples - handed_out) / num_pies = 6 := by
  sorry

end apples_per_pie_l2712_271297


namespace line_y_axis_intersection_l2712_271205

/-- The line equation is 5y + 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- Theorem: The point (0, 3) is the intersection of the line 5y + 3x = 15 with the y-axis -/
theorem line_y_axis_intersection :
  line_equation 0 3 ∧ on_y_axis 0 3 ∧
  ∀ x y : ℝ, line_equation x y ∧ on_y_axis x y → x = 0 ∧ y = 3 := by
  sorry

end line_y_axis_intersection_l2712_271205


namespace equation_solution_l2712_271246

theorem equation_solution : ∃! x : ℝ, (x + 1 ≠ 0 ∧ 2*x - 1 ≠ 0) ∧ (2 / (x + 1) = 3 / (2*x - 1)) := by
  sorry

end equation_solution_l2712_271246


namespace cd_combined_length_l2712_271279

/-- The combined length of 3 CDs is 6 hours, given that two CDs are 1.5 hours each and the third CD is twice as long as the shorter ones. -/
theorem cd_combined_length : 
  let short_cd_length : ℝ := 1.5
  let long_cd_length : ℝ := 2 * short_cd_length
  let total_length : ℝ := 2 * short_cd_length + long_cd_length
  total_length = 6 := by sorry

end cd_combined_length_l2712_271279


namespace range_of_a_for_sqrt_function_l2712_271271

theorem range_of_a_for_sqrt_function (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (2^x - a)) → a ≤ 0 :=
by sorry

end range_of_a_for_sqrt_function_l2712_271271


namespace divisor_problem_l2712_271236

theorem divisor_problem :
  ∃! d : ℕ+, d > 5 ∧
  (∃ x q : ℤ, x = q * d.val + 5) ∧
  (∃ x p : ℤ, 4 * x = p * d.val + 6) :=
sorry

end divisor_problem_l2712_271236
