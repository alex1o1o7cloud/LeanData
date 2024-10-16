import Mathlib

namespace NUMINAMATH_CALUDE_unique_satisfying_function_l2843_284357

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + f (x + y) - f x * f y = 0

/-- The theorem stating that there is exactly one function satisfying the equation -/
theorem unique_satisfying_function : ∃! f : ℝ → ℝ, SatisfyingFunction f := by sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l2843_284357


namespace NUMINAMATH_CALUDE_rectangle_fold_area_l2843_284308

theorem rectangle_fold_area (a b : ℝ) (h1 : a = 4) (h2 : b = 8) : 
  let diagonal := Real.sqrt (a^2 + b^2)
  let height := diagonal / 2
  (1/2) * diagonal * height = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_fold_area_l2843_284308


namespace NUMINAMATH_CALUDE_x_cube_minus_3x_eq_6_l2843_284318

theorem x_cube_minus_3x_eq_6 (x : ℝ) (h : x^3 - 3*x = 6) :
  x^6 + 27*x^2 = 36*x^2 + 36*x + 36 := by
  sorry

end NUMINAMATH_CALUDE_x_cube_minus_3x_eq_6_l2843_284318


namespace NUMINAMATH_CALUDE_field_ratio_l2843_284305

theorem field_ratio (l w : ℝ) (h1 : ∃ k : ℕ, l = k * w) 
  (h2 : l = 36) (h3 : 81 = (1/8) * (l * w)) : l / w = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_l2843_284305


namespace NUMINAMATH_CALUDE_smallest_y_value_l2843_284313

theorem smallest_y_value (x y z : ℝ) : 
  (4 < x ∧ x < z ∧ z < y ∧ y < 10) →
  (∀ a b : ℝ, (4 < a ∧ a < z ∧ z < b ∧ b < 10) → (⌊b⌋ - ⌊a⌋ : ℤ) ≤ 5) →
  (∃ a b : ℝ, (4 < a ∧ a < z ∧ z < b ∧ b < 10) ∧ (⌊b⌋ - ⌊a⌋ : ℤ) = 5) →
  9 ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_value_l2843_284313


namespace NUMINAMATH_CALUDE_point_outside_circle_implies_a_range_l2843_284381

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 2 = 0

-- Define the condition for a point to be outside the circle
def point_outside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 2 > 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -2 ∨ (2 < a ∧ a < 6)

-- Theorem statement
theorem point_outside_circle_implies_a_range :
  ∀ a : ℝ, point_outside_circle 1 1 a → a_range a :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_implies_a_range_l2843_284381


namespace NUMINAMATH_CALUDE_bucky_fish_count_l2843_284319

/-- The number of fish Bucky caught on Sunday -/
def F : ℕ := 5

/-- The price of the video game -/
def game_price : ℕ := 60

/-- The amount Bucky earned last weekend -/
def last_weekend_earnings : ℕ := 35

/-- The price of a trout -/
def trout_price : ℕ := 5

/-- The price of a blue-gill -/
def blue_gill_price : ℕ := 4

/-- The percentage of trout caught -/
def trout_percentage : ℚ := 3/5

/-- The percentage of blue-gill caught -/
def blue_gill_percentage : ℚ := 2/5

/-- The additional amount Bucky needs to save -/
def additional_savings : ℕ := 2

theorem bucky_fish_count :
  F * (trout_percentage * trout_price + blue_gill_percentage * blue_gill_price) =
  game_price - last_weekend_earnings - additional_savings :=
sorry

end NUMINAMATH_CALUDE_bucky_fish_count_l2843_284319


namespace NUMINAMATH_CALUDE_kaleb_final_score_l2843_284365

/-- Calculates Kaleb's final adjusted score in a trivia game -/
theorem kaleb_final_score : 
  let first_half_score : ℝ := 43
  let first_half_bonus1 : ℝ := 0.20
  let first_half_bonus2 : ℝ := 0.05
  let second_half_score : ℝ := 23
  let second_half_penalty1 : ℝ := 0.10
  let second_half_penalty2 : ℝ := 0.08
  
  let first_half_adjusted := first_half_score * (1 + first_half_bonus1 + first_half_bonus2)
  let second_half_adjusted := second_half_score * (1 - second_half_penalty1 - second_half_penalty2)
  
  first_half_adjusted + second_half_adjusted = 72.61
  := by sorry

end NUMINAMATH_CALUDE_kaleb_final_score_l2843_284365


namespace NUMINAMATH_CALUDE_A_neither_sufficient_nor_necessary_l2843_284354

-- Define propositions A and B
def proposition_A (a b : ℝ) : Prop := a + b ≠ 4
def proposition_B (a b : ℝ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem stating that A is neither sufficient nor necessary for B
theorem A_neither_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, proposition_A a b → proposition_B a b) ∧
  ¬(∀ a b : ℝ, proposition_B a b → proposition_A a b) :=
sorry

end NUMINAMATH_CALUDE_A_neither_sufficient_nor_necessary_l2843_284354


namespace NUMINAMATH_CALUDE_composite_function_solution_l2843_284346

theorem composite_function_solution (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x / 3 + 2)
  (hg : ∀ x, g x = 5 - 2 * x)
  (h : f (g a) = 4) :
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_composite_function_solution_l2843_284346


namespace NUMINAMATH_CALUDE_questionnaire_C_count_l2843_284302

/-- Represents the system sampling method described in the problem -/
def SystemSampling (totalPopulation sampleSize firstDrawn : ℕ) : 
  List ℕ := sorry

/-- Counts the number of elements in a list that fall within a given range -/
def CountInRange (list : List ℕ) (lower upper : ℕ) : ℕ := sorry

theorem questionnaire_C_count :
  let totalPopulation : ℕ := 960
  let sampleSize : ℕ := 32
  let firstDrawn : ℕ := 5
  let sample := SystemSampling totalPopulation sampleSize firstDrawn
  CountInRange sample 751 960 = 7 := by sorry

end NUMINAMATH_CALUDE_questionnaire_C_count_l2843_284302


namespace NUMINAMATH_CALUDE_vince_earnings_per_head_l2843_284337

/-- Represents Vince's hair salon business model -/
structure HairSalon where
  earningsPerHead : ℝ
  customersPerMonth : ℕ
  monthlyRentAndElectricity : ℝ
  recreationPercentage : ℝ
  monthlySavings : ℝ

/-- Theorem stating that Vince's earnings per head is $72 -/
theorem vince_earnings_per_head (salon : HairSalon)
    (h1 : salon.customersPerMonth = 80)
    (h2 : salon.monthlyRentAndElectricity = 280)
    (h3 : salon.recreationPercentage = 0.2)
    (h4 : salon.monthlySavings = 872)
    (h5 : salon.earningsPerHead * ↑salon.customersPerMonth * (1 - salon.recreationPercentage) =
          salon.earningsPerHead * ↑salon.customersPerMonth - salon.monthlyRentAndElectricity - salon.monthlySavings) :
    salon.earningsPerHead = 72 := by
  sorry

#check vince_earnings_per_head

end NUMINAMATH_CALUDE_vince_earnings_per_head_l2843_284337


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_property_l2843_284316

/-- Given a geometric sequence {aₙ} satisfying certain conditions, 
    prove that tan(a₁a₁₃) = √3 -/
theorem geometric_sequence_tan_property 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_condition : a 3 * a 11 + 2 * (a 7)^2 = 4 * Real.pi) : 
  Real.tan (a 1 * a 13) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_property_l2843_284316


namespace NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l2843_284317

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  /-- The base angle of the isosceles triangle -/
  base_angle : ℝ
  /-- The altitude of the isosceles triangle -/
  altitude : ℝ
  /-- The side length of the isosceles triangle -/
  side_length : ℝ
  /-- Constraint that the base angle is between 0 and π/2 -/
  angle_constraint : 0 < base_angle ∧ base_angle < π / 2
  /-- Constraint that the altitude is positive -/
  altitude_positive : altitude > 0
  /-- Constraint that the side length is positive -/
  side_positive : side_length > 0

/-- Theorem stating that there exist multiple non-congruent isosceles triangles
    with the same base angle and altitude -/
theorem isosceles_triangle_not_unique :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1.base_angle = t2.base_angle ∧
    t1.altitude = t2.altitude ∧
    t1.side_length ≠ t2.side_length :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l2843_284317


namespace NUMINAMATH_CALUDE_angle_A_value_max_value_angle_B_at_max_l2843_284399

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Side lengths
variable (S : ℝ) -- Area of the triangle

-- Define the conditions
axiom triangle_condition : a^2 = b^2 + c^2 + Real.sqrt 3 * a * b
axiom side_a_value : a = Real.sqrt 3

-- Define the theorems to be proved
theorem angle_A_value : A = 5 * Real.pi / 6 :=
sorry

theorem max_value : 
  ∃ (max : ℝ), ∀ (B C : ℝ), S + 3 * Real.cos B * Real.cos C ≤ max ∧ 
  ∃ (B₀ C₀ : ℝ), S + 3 * Real.cos B₀ * Real.cos C₀ = max ∧ max = 3 :=
sorry

theorem angle_B_at_max : 
  ∃ (B₀ C₀ : ℝ), S + 3 * Real.cos B₀ * Real.cos C₀ = 3 ∧ B₀ = Real.pi / 12 :=
sorry

end

end NUMINAMATH_CALUDE_angle_A_value_max_value_angle_B_at_max_l2843_284399


namespace NUMINAMATH_CALUDE_ratio_problem_l2843_284352

/-- Given three positive real numbers A, B, and C with specified ratios,
    prove the fraction of C to A and the ratio of A to C. -/
theorem ratio_problem (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hAB : A / B = 7 / 3) (hBC : B / C = 6 / 5) :
  C / A = 5 / 14 ∧ A / C = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2843_284352


namespace NUMINAMATH_CALUDE_store_coloring_books_l2843_284390

theorem store_coloring_books 
  (sold : ℕ) 
  (shelves : ℕ) 
  (books_per_shelf : ℕ) 
  (h1 : sold = 33) 
  (h2 : shelves = 9) 
  (h3 : books_per_shelf = 6) : 
  sold + shelves * books_per_shelf = 87 := by
  sorry

end NUMINAMATH_CALUDE_store_coloring_books_l2843_284390


namespace NUMINAMATH_CALUDE_min_value_of_s_min_value_of_s_achieved_l2843_284321

theorem min_value_of_s (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (y*z/x + z*x/y + x*y/z) ≥ Real.sqrt 3 := by
  sorry

theorem min_value_of_s_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 1 ∧
    b*c/a + c*a/b + a*b/c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_s_min_value_of_s_achieved_l2843_284321


namespace NUMINAMATH_CALUDE_james_teaching_years_l2843_284364

theorem james_teaching_years (james partner : ℕ) 
  (h1 : james = partner + 10)
  (h2 : james + partner = 70) : 
  james = 40 := by
sorry

end NUMINAMATH_CALUDE_james_teaching_years_l2843_284364


namespace NUMINAMATH_CALUDE_root_implies_k_value_l2843_284383

theorem root_implies_k_value (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 9 * x + 8 = 0 ∧ x = 1) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l2843_284383


namespace NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l2843_284379

theorem sufficient_condition_for_f_less_than_one 
  (a : ℝ) (h_a : a > 1) :
  ∀ x : ℝ, -1 < x ∧ x < 0 → (a * x + 2 * x) < 1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l2843_284379


namespace NUMINAMATH_CALUDE_mechanic_worked_six_hours_l2843_284322

def mechanic_hours (total_cost parts_cost labor_rate : ℚ) : ℚ :=
  let parts_total := 2 * parts_cost
  let labor_cost := total_cost - parts_total
  let minutes_worked := labor_cost / labor_rate
  minutes_worked / 60

theorem mechanic_worked_six_hours :
  mechanic_hours 220 20 0.5 = 6 := by sorry

end NUMINAMATH_CALUDE_mechanic_worked_six_hours_l2843_284322


namespace NUMINAMATH_CALUDE_symbol_values_l2843_284329

theorem symbol_values (triangle star : ℤ) 
  (eq1 : 3 * triangle + 2 * star = 14)
  (eq2 : 2 * star + 5 * triangle = 18) : 
  triangle = 2 ∧ star = 4 := by
sorry

end NUMINAMATH_CALUDE_symbol_values_l2843_284329


namespace NUMINAMATH_CALUDE_min_value_of_squares_l2843_284393

theorem min_value_of_squares (a b c d : ℝ) (h1 : a * b = 3) (h2 : c + 3 * d = 0) :
  (a - c)^2 + (b - d)^2 ≥ 18/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l2843_284393


namespace NUMINAMATH_CALUDE_relay_team_permutations_l2843_284309

theorem relay_team_permutations (n : ℕ) (h : n = 3) : Nat.factorial n = 6 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l2843_284309


namespace NUMINAMATH_CALUDE_tangent_addition_subtraction_l2843_284335

theorem tangent_addition_subtraction (γ β : Real) (h1 : Real.tan γ = 5) (h2 : Real.tan β = 3) :
  Real.tan (γ + β) = -4/7 ∧ Real.tan (γ - β) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_tangent_addition_subtraction_l2843_284335


namespace NUMINAMATH_CALUDE_min_cost_for_range_l2843_284362

/-- The cost of a "yes" answer in rubles -/
def yes_cost : ℕ := 2

/-- The cost of a "no" answer in rubles -/
def no_cost : ℕ := 1

/-- The range of possible hidden numbers -/
def number_range : Set ℕ := Finset.range 144

/-- The minimum cost function for guessing a number in a given set -/
noncomputable def min_cost (S : Set ℕ) : ℕ :=
  sorry

/-- The theorem stating that the minimum cost to guess any number in [1, 144] is 11 rubles -/
theorem min_cost_for_range : min_cost number_range = 11 :=
  sorry

end NUMINAMATH_CALUDE_min_cost_for_range_l2843_284362


namespace NUMINAMATH_CALUDE_college_students_count_l2843_284306

theorem college_students_count (num_girls : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) : 
  num_girls = 120 → ratio_boys = 8 → ratio_girls = 5 →
  (num_girls + (num_girls * ratio_boys) / ratio_girls : ℕ) = 312 := by
sorry

end NUMINAMATH_CALUDE_college_students_count_l2843_284306


namespace NUMINAMATH_CALUDE_ten_mile_taxi_cost_l2843_284327

/-- Calculates the cost of a taxi ride -/
def taxiCost (baseFare mileCharge flatCharge thresholdMiles miles : ℚ) : ℚ :=
  baseFare + mileCharge * miles + if miles > thresholdMiles then flatCharge else 0

/-- Theorem: The cost of a 10-mile taxi ride is $5.50 -/
theorem ten_mile_taxi_cost :
  taxiCost 2 0.3 0.5 8 10 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_ten_mile_taxi_cost_l2843_284327


namespace NUMINAMATH_CALUDE_probability_of_two_white_balls_l2843_284333

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def black_balls : ℕ := 2

def probability_two_white : ℚ := 3 / 10

theorem probability_of_two_white_balls :
  (Nat.choose white_balls 2) / (Nat.choose total_balls 2) = probability_two_white :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_white_balls_l2843_284333


namespace NUMINAMATH_CALUDE_bucket_problem_l2843_284385

/-- Represents the capacity of a bucket --/
structure Bucket where
  capacity : ℝ
  sand : ℝ

/-- Proves that given the conditions of the bucket problem, 
    the initial fraction of Bucket B's capacity filled with sand is 3/8 --/
theorem bucket_problem (bucketA bucketB : Bucket) : 
  bucketA.sand = (1/4) * bucketA.capacity →
  bucketB.capacity = (1/2) * bucketA.capacity →
  bucketA.sand + bucketB.sand = (0.4375) * bucketA.capacity →
  bucketB.sand / bucketB.capacity = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_bucket_problem_l2843_284385


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2843_284320

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 - 2*(m-2)*x₁ + m^2 = 0 ∧ 
   x₂^2 - 2*(m-2)*x₂ + m^2 = 0) → 
  m < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2843_284320


namespace NUMINAMATH_CALUDE_craftsman_earnings_solution_l2843_284395

def craftsman_earnings (hours_worked : ℕ) (wage_A wage_B : ℚ) : Prop :=
  let earnings_A := hours_worked * wage_A
  let earnings_B := hours_worked * wage_B
  wage_A ≠ wage_B ∧
  (hours_worked - 1) * wage_A = 720 ∧
  (hours_worked - 5) * wage_B = 800 ∧
  (hours_worked - 1) * wage_B - (hours_worked - 5) * wage_A = 360 ∧
  earnings_A = 750 ∧
  earnings_B = 1000

theorem craftsman_earnings_solution :
  ∃ (hours_worked : ℕ) (wage_A wage_B : ℚ),
    craftsman_earnings hours_worked wage_A wage_B :=
by
  sorry

end NUMINAMATH_CALUDE_craftsman_earnings_solution_l2843_284395


namespace NUMINAMATH_CALUDE_max_value_of_k_l2843_284377

theorem max_value_of_k : ∃ (k : ℝ), k = Real.sqrt 10 ∧ 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 7 → Real.sqrt (x - 2) + Real.sqrt (7 - x) ≤ k) ∧
  (∀ ε > 0, ∃ x : ℝ, 2 ≤ x ∧ x ≤ 7 ∧ Real.sqrt (x - 2) + Real.sqrt (7 - x) > k - ε) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_k_l2843_284377


namespace NUMINAMATH_CALUDE_repair_charge_is_30_l2843_284340

/-- Represents the services and pricing at Cid's mechanic shop --/
structure MechanicShop where
  oil_change_price : ℕ
  car_wash_price : ℕ
  oil_changes : ℕ
  repairs : ℕ
  car_washes : ℕ
  total_earnings : ℕ

/-- Theorem stating that the repair charge is $30 --/
theorem repair_charge_is_30 (shop : MechanicShop) 
  (h1 : shop.oil_change_price = 20)
  (h2 : shop.car_wash_price = 5)
  (h3 : shop.oil_changes = 5)
  (h4 : shop.repairs = 10)
  (h5 : shop.car_washes = 15)
  (h6 : shop.total_earnings = 475) :
  (shop.total_earnings - (shop.oil_changes * shop.oil_change_price + shop.car_washes * shop.car_wash_price)) / shop.repairs = 30 :=
by
  sorry

#check repair_charge_is_30

end NUMINAMATH_CALUDE_repair_charge_is_30_l2843_284340


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l2843_284388

-- Define the initial amount
def initial_amount : ℚ := 6160

-- Define the interest rates
def interest_rate_year1 : ℚ := 10 / 100
def interest_rate_year2 : ℚ := 12 / 100

-- Define the function to calculate the amount after one year
def amount_after_one_year (principal : ℚ) (rate : ℚ) : ℚ :=
  principal * (1 + rate)

-- Define the function to calculate the final amount after two years
def final_amount : ℚ :=
  amount_after_one_year (amount_after_one_year initial_amount interest_rate_year1) interest_rate_year2

-- State the theorem
theorem compound_interest_calculation :
  final_amount = 7589.12 := by sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l2843_284388


namespace NUMINAMATH_CALUDE_residue_11_2048_mod_17_l2843_284323

theorem residue_11_2048_mod_17 : 11^2048 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_residue_11_2048_mod_17_l2843_284323


namespace NUMINAMATH_CALUDE_ab_nonzero_sufficient_not_necessary_for_a_nonzero_l2843_284371

theorem ab_nonzero_sufficient_not_necessary_for_a_nonzero :
  (∀ a b : ℝ, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ a * b = 0) :=
by sorry

end NUMINAMATH_CALUDE_ab_nonzero_sufficient_not_necessary_for_a_nonzero_l2843_284371


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2843_284338

theorem max_value_sqrt_sum (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) ≤ 12 ∧
  ∃ y, -36 ≤ y ∧ y ≤ 36 ∧ Real.sqrt (36 + y) + Real.sqrt (36 - y) = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2843_284338


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2843_284380

theorem polynomial_coefficient_sum (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (1 + m * x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = -63) →
  (m = 3 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2843_284380


namespace NUMINAMATH_CALUDE_factory_uses_systematic_sampling_l2843_284368

/-- Represents a sampling method used in quality control --/
inductive SamplingMethod
| Systematic
| Random
| Stratified
| Cluster

/-- Represents a factory with a conveyor belt and quality inspection process --/
structure Factory where
  /-- The interval between product inspections in minutes --/
  inspection_interval : ℕ
  /-- Whether the inspection position on the conveyor belt is fixed --/
  fixed_position : Bool

/-- Determines if a given factory uses systematic sampling --/
def uses_systematic_sampling (f : Factory) : Prop :=
  f.inspection_interval > 0 ∧ f.fixed_position

/-- The factory described in the problem --/
def problem_factory : Factory :=
  { inspection_interval := 10
  , fixed_position := true }

/-- Theorem stating that the factory in the problem uses systematic sampling --/
theorem factory_uses_systematic_sampling :
  uses_systematic_sampling problem_factory :=
sorry

end NUMINAMATH_CALUDE_factory_uses_systematic_sampling_l2843_284368


namespace NUMINAMATH_CALUDE_square_grid_divisible_four_parts_l2843_284372

/-- A square grid of cells that can be divided into four equal parts -/
structure SquareGrid where
  n : ℕ
  n_even : Even n
  n_ge_2 : n ≥ 2

/-- The number of cells in each part when the grid is divided into four equal parts -/
def cells_per_part (grid : SquareGrid) : ℕ := grid.n * grid.n / 4

/-- Theorem stating that a square grid can be divided into four equal parts -/
theorem square_grid_divisible_four_parts (grid : SquareGrid) :
  ∃ (part_size : ℕ), part_size = cells_per_part grid ∧ 
  grid.n * grid.n = 4 * part_size :=
sorry

end NUMINAMATH_CALUDE_square_grid_divisible_four_parts_l2843_284372


namespace NUMINAMATH_CALUDE_half_fourth_of_twelve_y_plus_three_l2843_284330

theorem half_fourth_of_twelve_y_plus_three (y : ℝ) :
  (1/2) * (1/4) * (12 * y + 3) = (3 * y) / 2 + 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_half_fourth_of_twelve_y_plus_three_l2843_284330


namespace NUMINAMATH_CALUDE_xiaohua_school_time_l2843_284344

-- Define a custom type for time
structure SchoolTime where
  hours : ℕ
  minutes : ℕ
  is_pm : Bool

-- Define a function to calculate the time difference in minutes
def time_diff (t1 t2 : SchoolTime) : ℕ :=
  let total_minutes1 := t1.hours * 60 + t1.minutes + (if t1.is_pm then 12 * 60 else 0)
  let total_minutes2 := t2.hours * 60 + t2.minutes + (if t2.is_pm then 12 * 60 else 0)
  total_minutes2 - total_minutes1

-- Define Xiaohua's schedule
def morning_arrival : SchoolTime := ⟨7, 20, false⟩
def morning_departure : SchoolTime := ⟨11, 45, false⟩
def afternoon_arrival : SchoolTime := ⟨1, 50, true⟩
def afternoon_departure : SchoolTime := ⟨5, 15, true⟩

-- Theorem statement
theorem xiaohua_school_time :
  time_diff morning_arrival morning_departure +
  time_diff afternoon_arrival afternoon_departure = 7 * 60 + 50 := by
  sorry

end NUMINAMATH_CALUDE_xiaohua_school_time_l2843_284344


namespace NUMINAMATH_CALUDE_teal_color_survey_l2843_284398

theorem teal_color_survey (total : ℕ) (more_blue : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 150)
  (h2 : more_blue = 85)
  (h3 : both = 47)
  (h4 : neither = 22) :
  total - (more_blue - both + both + neither) = 90 := by
  sorry

end NUMINAMATH_CALUDE_teal_color_survey_l2843_284398


namespace NUMINAMATH_CALUDE_smallest_positive_integer_2010m_44000n_l2843_284325

theorem smallest_positive_integer_2010m_44000n : 
  (∃ (k : ℕ), k > 0 ∧ ∃ (m n : ℤ), k = 2010 * m + 44000 * n) ∧ 
  (∀ (k : ℕ), k > 0 → (∃ (m n : ℤ), k = 2010 * m + 44000 * n) → k ≥ 10) ∧
  (∃ (m n : ℤ), 10 = 2010 * m + 44000 * n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_2010m_44000n_l2843_284325


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2843_284332

theorem division_remainder_problem (j : ℕ+) (h : ∃ b : ℕ, 120 = b * j^2 + 12) :
  ∃ k : ℕ, 180 = k * j + 0 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2843_284332


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l2843_284311

/-- 
Given two people traveling in opposite directions for 1.5 hours, 
with one person traveling at 5 miles per hour and ending up 19.5 miles apart, 
prove that the other person's speed must be 8 miles per hour.
-/
theorem opposite_direction_speed 
  (time : ℝ) 
  (distance : ℝ) 
  (speed_peter : ℝ) 
  (speed_juan : ℝ) : 
  time = 1.5 → 
  distance = 19.5 → 
  speed_peter = 5 → 
  distance = (speed_juan + speed_peter) * time → 
  speed_juan = 8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l2843_284311


namespace NUMINAMATH_CALUDE_square_difference_sqrt5_sqrt2_l2843_284386

theorem square_difference_sqrt5_sqrt2 :
  let x : ℝ := Real.sqrt 5
  let y : ℝ := Real.sqrt 2
  (x - y)^2 = 7 - 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_square_difference_sqrt5_sqrt2_l2843_284386


namespace NUMINAMATH_CALUDE_maria_towels_l2843_284345

theorem maria_towels (green_towels white_towels given_to_mother : ℕ) 
  (h1 : green_towels = 58)
  (h2 : white_towels = 43)
  (h3 : given_to_mother = 87) :
  green_towels + white_towels - given_to_mother = 14 :=
by sorry

end NUMINAMATH_CALUDE_maria_towels_l2843_284345


namespace NUMINAMATH_CALUDE_button_probability_l2843_284367

/-- Given a jar with red and blue buttons, prove the probability of selecting two red buttons after a specific removal process. -/
theorem button_probability (initial_red initial_blue : ℕ) 
  (h1 : initial_red = 6)
  (h2 : initial_blue = 10)
  (h3 : ∃ (removed : ℕ), 
    removed ≤ initial_red ∧ 
    removed ≤ initial_blue ∧ 
    initial_red + initial_blue - 2 * removed = (3 / 4) * (initial_red + initial_blue)) :
  let total_initial := initial_red + initial_blue
  let removed := (total_initial - (3 / 4) * total_initial) / 2
  let red_a := initial_red - removed
  let total_a := (3 / 4) * total_initial
  let prob_red_a := red_a / total_a
  let prob_red_b := removed / (2 * removed)
  prob_red_a * prob_red_b = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_button_probability_l2843_284367


namespace NUMINAMATH_CALUDE_solution_set_implies_k_value_l2843_284363

theorem solution_set_implies_k_value (k : ℝ) : 
  (∀ x : ℝ, |k * x - 4| ≤ 2 ↔ 1 ≤ x ∧ x ≤ 3) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_k_value_l2843_284363


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l2843_284347

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (bp : BatsmanPerformance) (runsInLastInning : ℕ) : ℚ :=
  (bp.totalRuns + runsInLastInning) / (bp.innings + 1)

/-- Theorem: If a batsman's average increases by 2 after scoring 50 in the 17th inning, 
    then the new average is 18 -/
theorem batsman_average_after_17th_inning 
  (bp : BatsmanPerformance)
  (h1 : bp.innings = 16)
  (h2 : newAverage bp 50 = bp.average + 2)
  : newAverage bp 50 = 18 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l2843_284347


namespace NUMINAMATH_CALUDE_intersection_M_N_l2843_284303

-- Define set M
def M : Set ℝ := {0, 1, 2}

-- Define set N
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2843_284303


namespace NUMINAMATH_CALUDE_prime_sum_product_l2843_284373

/-- Given prime numbers a, b, and c satisfying abc + a + b + c = 99,
    prove that two of the numbers are 2 and the other is 19 -/
theorem prime_sum_product (a b c : ℕ) : 
  Prime a → Prime b → Prime c → a * b * c + a + b + c = 99 →
  ((a = 2 ∧ b = 2 ∧ c = 19) ∨ (a = 2 ∧ b = 19 ∧ c = 2) ∨ (a = 19 ∧ b = 2 ∧ c = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_product_l2843_284373


namespace NUMINAMATH_CALUDE_basic_algorithm_statements_correct_l2843_284341

/-- Represents the types of algorithm statements -/
inductive AlgorithmStatement
  | INPUT
  | PRINT
  | IF_THEN
  | DO
  | END
  | WHILE
  | END_IF

/-- Defines the set of basic algorithm statements -/
def BasicAlgorithmStatements : Set AlgorithmStatement :=
  {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT, AlgorithmStatement.IF_THEN, 
   AlgorithmStatement.DO, AlgorithmStatement.WHILE}

/-- Theorem: The set of basic algorithm statements is exactly 
    {INPUT, PRINT, IF-THEN, DO, WHILE} -/
theorem basic_algorithm_statements_correct :
  BasicAlgorithmStatements = 
    {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT, AlgorithmStatement.IF_THEN, 
     AlgorithmStatement.DO, AlgorithmStatement.WHILE} := by
  sorry

end NUMINAMATH_CALUDE_basic_algorithm_statements_correct_l2843_284341


namespace NUMINAMATH_CALUDE_best_standing_for_consistent_93rd_l2843_284369

/-- Represents a cycling competition -/
structure CyclingCompetition where
  stages : ℕ
  participants : ℕ
  daily_position : ℕ

/-- The best possible overall standing for a competitor -/
def best_possible_standing (comp : CyclingCompetition) : ℕ :=
  comp.participants - min (comp.stages * (comp.participants - comp.daily_position)) (comp.participants - 1)

/-- Theorem: In a 14-stage competition with 100 participants, 
    a competitor finishing 93rd each day can achieve 2nd place at best -/
theorem best_standing_for_consistent_93rd :
  let comp : CyclingCompetition := ⟨14, 100, 93⟩
  best_possible_standing comp = 2 := by
  sorry

#eval best_possible_standing ⟨14, 100, 93⟩

end NUMINAMATH_CALUDE_best_standing_for_consistent_93rd_l2843_284369


namespace NUMINAMATH_CALUDE_min_value_f_when_a_is_1_range_of_a_when_solution_exists_l2843_284342

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

-- Theorem 1: Minimum value of f when a = 1
theorem min_value_f_when_a_is_1 :
  ∀ x : ℝ, f 1 x ≥ 2 :=
sorry

-- Theorem 2: Range of a when solution set of f(x) ≤ 3 is non-empty
theorem range_of_a_when_solution_exists :
  (∃ x : ℝ, f a x ≤ 3) → |3 - a| ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_when_a_is_1_range_of_a_when_solution_exists_l2843_284342


namespace NUMINAMATH_CALUDE_garden_center_discount_l2843_284350

/-- Represents the purchase and payment details at a garden center --/
structure GardenPurchase where
  pansy_count : ℕ
  pansy_price : ℚ
  hydrangea_count : ℕ
  hydrangea_price : ℚ
  petunia_count : ℕ
  petunia_price : ℚ
  paid_amount : ℚ
  change_received : ℚ

/-- Calculates the discount offered by the garden center --/
def calculate_discount (purchase : GardenPurchase) : ℚ :=
  let total_cost := purchase.pansy_count * purchase.pansy_price +
                    purchase.hydrangea_count * purchase.hydrangea_price +
                    purchase.petunia_count * purchase.petunia_price
  let amount_paid := purchase.paid_amount - purchase.change_received
  total_cost - amount_paid

/-- Theorem stating that the discount for the given purchase is $3.00 --/
theorem garden_center_discount :
  let purchase := GardenPurchase.mk 5 2.5 1 12.5 5 1 50 23
  calculate_discount purchase = 3 := by sorry

end NUMINAMATH_CALUDE_garden_center_discount_l2843_284350


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2843_284301

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a > b → a - b > -2) ∧ ¬(a - b > -2 → a > b) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2843_284301


namespace NUMINAMATH_CALUDE_certain_number_problem_l2843_284331

theorem certain_number_problem (x y : ℝ) : 
  0.12 / x * y = 12 ∧ x = 0.1 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2843_284331


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2843_284304

/-- Three points in 3D space are collinear if the determinant of the matrix formed by subtracting
    the coordinates of the first point from the other two is zero. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  let (x3, y3, z3) := p3
  (x2 - x1) * ((y3 - y1) * (z2 - z1) - (z3 - z1) * (y2 - y1)) = 0

/-- If the points (2,a,b), (a,3,b), and (a,b,4) are collinear, then a + b = -2. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = -2 :=
by sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2843_284304


namespace NUMINAMATH_CALUDE_f_nested_application_l2843_284326

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 1

theorem f_nested_application : f (f (f (f (f 1)))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_nested_application_l2843_284326


namespace NUMINAMATH_CALUDE_earnings_difference_l2843_284361

def bert_phones : ℕ := 8
def bert_price : ℕ := 18
def tory_guns : ℕ := 7
def tory_price : ℕ := 20

theorem earnings_difference : bert_phones * bert_price - tory_guns * tory_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_l2843_284361


namespace NUMINAMATH_CALUDE_at_least_one_blue_multiple_of_three_l2843_284339

/-- Represents a marked point on the circle --/
structure MarkedPoint where
  value : Int

/-- Represents a chord on the circle --/
structure Chord where
  points : List MarkedPoint

/-- The configuration of chords and points on the circle --/
structure CircleConfiguration where
  chords : List Chord
  endpointZeros : Nat
  endpointOnes : Nat

/-- Calculates yellow numbers (sum of endpoint values) for a chord --/
def yellowNumbers (chord : Chord) : List Int :=
  sorry

/-- Calculates blue numbers (absolute difference of endpoint values) for a chord --/
def blueNumbers (chord : Chord) : List Int :=
  sorry

/-- Checks if the yellow numbers are consecutive from 0 to N --/
def isConsecutiveYellow (yellowNums : List Int) : Bool :=
  sorry

theorem at_least_one_blue_multiple_of_three 
  (config : CircleConfiguration) 
  (h1 : config.chords.length = 2019)
  (h2 : config.endpointZeros = 2019)
  (h3 : config.endpointOnes = 2019)
  (h4 : ∀ c ∈ config.chords, c.points.length ≥ 2)
  (h5 : let allYellow := config.chords.map yellowNumbers |>.join
        isConsecutiveYellow allYellow) :
  ∃ (b : Int), b ∈ (config.chords.map blueNumbers |>.join) ∧ b % 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_at_least_one_blue_multiple_of_three_l2843_284339


namespace NUMINAMATH_CALUDE_f_inequality_l2843_284310

noncomputable def f (x : ℝ) := x^2 - Real.cos x

theorem f_inequality : f 0 < f (-0.5) ∧ f (-0.5) < f 0.6 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2843_284310


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l2843_284370

theorem angle_measure_in_triangle (P Q R : ℝ) (h : P + Q = 60) : P + Q + R = 180 → R = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l2843_284370


namespace NUMINAMATH_CALUDE_place_mat_length_l2843_284355

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) : 
  r = 4 →
  n = 6 →
  w = 1 →
  (x + 2 * Real.sqrt 3 - 1/2)^2 = 63/4 →
  x = (3 * Real.sqrt 7 - Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_place_mat_length_l2843_284355


namespace NUMINAMATH_CALUDE_tom_gems_calculation_l2843_284378

/-- The number of gems Tom receives for each dollar spent -/
def gems_per_dollar : ℕ := 100

/-- The amount Tom spends in dollars -/
def amount_spent : ℕ := 250

/-- The bonus percentage Tom receives -/
def bonus_percentage : ℚ := 20 / 100

/-- The total number of gems Tom ends up with -/
def total_gems : ℕ := 30000

theorem tom_gems_calculation :
  (gems_per_dollar * amount_spent) * (1 + bonus_percentage) = total_gems := by
  sorry

end NUMINAMATH_CALUDE_tom_gems_calculation_l2843_284378


namespace NUMINAMATH_CALUDE_existence_of_nondivisible_power_l2843_284324

theorem existence_of_nondivisible_power (a b c : ℕ+) (h : Nat.gcd a b.val = 1 ∧ Nat.gcd (Nat.gcd a b.val) c.val = 1) :
  ∃ n : ℕ+, ∀ k : ℕ+, ¬(2^n.val ∣ a^k.val + b^k.val + c^k.val) :=
sorry

end NUMINAMATH_CALUDE_existence_of_nondivisible_power_l2843_284324


namespace NUMINAMATH_CALUDE_pacos_countertop_marble_weight_l2843_284343

theorem pacos_countertop_marble_weight : 
  let weights : List ℝ := [0.33, 0.33, 0.08, 0.25, 0.02, 0.12, 0.15]
  weights.sum = 1.28 := by
  sorry

end NUMINAMATH_CALUDE_pacos_countertop_marble_weight_l2843_284343


namespace NUMINAMATH_CALUDE_fold_sum_l2843_284374

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a fold on a piece of graph paper -/
structure Fold where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Theorem: If a piece of graph paper is folded such that (0,3) matches with (5,0) 
    and (8,4) matches with (p,q), then p + q = 10 -/
theorem fold_sum (f : Fold) (h1 : f.p1 = ⟨0, 3⟩) (h2 : f.p2 = ⟨5, 0⟩) 
    (h3 : f.p3 = ⟨8, 4⟩) (h4 : f.p4 = ⟨f.p4.x, f.p4.y⟩) : 
    f.p4.x + f.p4.y = 10 := by
  sorry

end NUMINAMATH_CALUDE_fold_sum_l2843_284374


namespace NUMINAMATH_CALUDE_garden_multiplier_l2843_284336

theorem garden_multiplier (width length perimeter : ℝ) 
  (h1 : perimeter = 2 * length + 2 * width)
  (h2 : perimeter = 100)
  (h3 : length = 38)
  (h4 : ∃ m : ℝ, length = m * width + 2) :
  ∃ m : ℝ, length = m * width + 2 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_garden_multiplier_l2843_284336


namespace NUMINAMATH_CALUDE_hayley_sticker_distribution_l2843_284353

theorem hayley_sticker_distribution (total_stickers : ℕ) (num_friends : ℕ) 
  (h1 : total_stickers = 72) 
  (h2 : num_friends = 9) 
  (h3 : total_stickers % num_friends = 0) : 
  total_stickers / num_friends = 8 := by
sorry

end NUMINAMATH_CALUDE_hayley_sticker_distribution_l2843_284353


namespace NUMINAMATH_CALUDE_ellipse_tangent_properties_l2843_284328

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the companion circle E
def companion_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define a point on the companion circle
def point_on_circle (P : ℝ × ℝ) : Prop := companion_circle P.1 P.2

-- Define a tangent line to the ellipse
def is_tangent (P A : ℝ × ℝ) : Prop :=
  point_on_circle P ∧ ellipse A.1 A.2 ∧
  ∀ t : ℝ, t ≠ 0 → ¬(ellipse (A.1 + t * (P.1 - A.1)) (A.2 + t * (P.2 - A.2)))

-- Main theorem
theorem ellipse_tangent_properties :
  ∀ P A B Q : ℝ × ℝ,
  point_on_circle P →
  is_tangent P A →
  is_tangent P B →
  companion_circle Q.1 Q.2 →
  (∃ t : ℝ, Q.1 = A.1 + t * (P.1 - A.1) ∧ Q.2 = A.2 + t * (P.2 - A.2)) →
  (A ≠ B) →
  (∀ k₁ k₂ : ℝ,
    (P.1 ≠ 0 ∨ P.2 ≠ 0) →
    (Q.1 ≠ 0 ∨ Q.2 ≠ 0) →
    k₁ = P.2 / P.1 →
    k₂ = Q.2 / Q.1 →
    (((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0) ∧
     (k₁ * k₂ = -1/3))) :=
sorry

end NUMINAMATH_CALUDE_ellipse_tangent_properties_l2843_284328


namespace NUMINAMATH_CALUDE_cookie_problem_l2843_284360

theorem cookie_problem (C : ℕ) : C ≥ 187 ∧ (3 : ℚ) / 70 * C = 8 → C = 187 :=
by
  sorry

#check cookie_problem

end NUMINAMATH_CALUDE_cookie_problem_l2843_284360


namespace NUMINAMATH_CALUDE_ones_divisible_by_27_l2843_284334

def ones_number (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem ones_divisible_by_27 :
  ∃ k : ℕ, ones_number 27 = 27 * k :=
sorry

end NUMINAMATH_CALUDE_ones_divisible_by_27_l2843_284334


namespace NUMINAMATH_CALUDE_green_block_weight_l2843_284389

/-- The weight of the yellow block in pounds -/
def yellow_weight : ℝ := 0.6

/-- The difference in weight between the yellow and green blocks in pounds -/
def weight_difference : ℝ := 0.2

/-- The weight of the green block in pounds -/
def green_weight : ℝ := yellow_weight - weight_difference

theorem green_block_weight : green_weight = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_green_block_weight_l2843_284389


namespace NUMINAMATH_CALUDE_total_amount_formula_total_amount_after_five_months_l2843_284396

/-- Savings account with monthly interest -/
structure SavingsAccount where
  initialDeposit : ℝ
  monthlyInterestRate : ℝ

/-- Calculate total amount after x months -/
def totalAmount (account : SavingsAccount) (months : ℝ) : ℝ :=
  account.initialDeposit + account.initialDeposit * account.monthlyInterestRate * months

/-- Theorem: Total amount after x months is 100 + 0.36x -/
theorem total_amount_formula (account : SavingsAccount) (months : ℝ) 
    (h1 : account.initialDeposit = 100)
    (h2 : account.monthlyInterestRate = 0.0036) : 
    totalAmount account months = 100 + 0.36 * months := by
  sorry

/-- Theorem: Total amount after 5 months is 101.8 -/
theorem total_amount_after_five_months (account : SavingsAccount) 
    (h1 : account.initialDeposit = 100)
    (h2 : account.monthlyInterestRate = 0.0036) : 
    totalAmount account 5 = 101.8 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_formula_total_amount_after_five_months_l2843_284396


namespace NUMINAMATH_CALUDE_bob_orders_12_muffins_l2843_284376

/-- The number of muffins Bob orders per day -/
def muffins_per_day : ℕ := sorry

/-- The cost price of each muffin in cents -/
def cost_price : ℕ := 75

/-- The selling price of each muffin in cents -/
def selling_price : ℕ := 150

/-- The profit Bob makes per week in cents -/
def weekly_profit : ℕ := 6300

/-- Theorem stating that Bob orders 12 muffins per day -/
theorem bob_orders_12_muffins : muffins_per_day = 12 := by
  sorry

end NUMINAMATH_CALUDE_bob_orders_12_muffins_l2843_284376


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2843_284312

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (3 + i) / (i^2)
  Complex.abs z = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2843_284312


namespace NUMINAMATH_CALUDE_congruent_triangles_equal_perimeters_l2843_284315

/-- Two triangles are congruent if they have the same shape and size -/
def CongruentTriangles (T1 T2 : Set (ℝ × ℝ)) : Prop := sorry

/-- The perimeter of a triangle is the sum of the lengths of its sides -/
def Perimeter (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- If two triangles are congruent, then their perimeters are equal -/
theorem congruent_triangles_equal_perimeters (T1 T2 : Set (ℝ × ℝ)) :
  CongruentTriangles T1 T2 → Perimeter T1 = Perimeter T2 := by sorry

end NUMINAMATH_CALUDE_congruent_triangles_equal_perimeters_l2843_284315


namespace NUMINAMATH_CALUDE_triangle_abc_isosceles_l2843_284366

/-- Given a triangle ABC where 2sin(A) * cos(B) = sin(C), prove that the triangle is isosceles -/
theorem triangle_abc_isosceles (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_condition : 2 * Real.sin A * Real.cos B = Real.sin C) :
  A = B ∨ B = C ∨ A = C :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_isosceles_l2843_284366


namespace NUMINAMATH_CALUDE_current_books_count_l2843_284359

/-- The number of books in a library over time -/
def library_books (initial_old_books : ℕ) (bought_two_years_ago : ℕ) (bought_last_year : ℕ) (donated_this_year : ℕ) : ℕ :=
  initial_old_books + bought_two_years_ago + bought_last_year - donated_this_year

/-- Theorem: The current number of books in the library is 1000 -/
theorem current_books_count :
  let initial_old_books : ℕ := 500
  let bought_two_years_ago : ℕ := 300
  let bought_last_year : ℕ := bought_two_years_ago + 100
  let donated_this_year : ℕ := 200
  library_books initial_old_books bought_two_years_ago bought_last_year donated_this_year = 1000 := by
  sorry

end NUMINAMATH_CALUDE_current_books_count_l2843_284359


namespace NUMINAMATH_CALUDE_beaker_problem_solution_l2843_284375

/-- Represents the capacity of a beaker -/
structure Beaker where
  capacity : ℚ
  filled : ℚ
  h_filled_nonneg : 0 ≤ filled
  h_filled_le_capacity : filled ≤ capacity

/-- The fraction of a beaker that is filled -/
def fraction_filled (b : Beaker) : ℚ :=
  b.filled / b.capacity

/-- Represents the problem setup with two beakers -/
structure BeakerProblem where
  small : Beaker
  large : Beaker
  h_small_half_filled : fraction_filled small = 1/2
  h_large_capacity : large.capacity = 5 * small.capacity
  h_large_fifth_filled : fraction_filled large = 1/5

/-- The main theorem to prove -/
theorem beaker_problem_solution (problem : BeakerProblem) :
  let final_large := Beaker.mk
    problem.large.capacity
    (problem.large.filled + problem.small.filled)
    (by sorry) -- Proof that the new filled amount is non-negative
    (by sorry) -- Proof that the new filled amount is ≤ capacity
  fraction_filled final_large = 3/10 := by sorry


end NUMINAMATH_CALUDE_beaker_problem_solution_l2843_284375


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2843_284394

-- Define set A
def A : Set ℝ := {x | 0 < 3 - x ∧ 3 - x ≤ 2}

-- Define set B
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2843_284394


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2843_284397

theorem tangent_line_intersection (x₀ : ℝ) (m : ℝ) : 
  (0 < m) → (m < 1) →
  (2 * x₀ = 1 / m) →
  (x₀^2 - Real.log (2 * x₀) - 1 = 0) →
  (Real.sqrt 2 < x₀) ∧ (x₀ < Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2843_284397


namespace NUMINAMATH_CALUDE_range_of_a_when_p_or_q_false_l2843_284358

def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0

def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a_when_p_or_q_false :
  {a : ℝ | ¬(p a ∨ q a)} = Set.Ioo (-1) 0 ∪ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_p_or_q_false_l2843_284358


namespace NUMINAMATH_CALUDE_expression_equality_l2843_284392

/-- Given two real numbers a and b, prove that the expression
    "the difference between three times the number for A and the number for B
    divided by the sum of the number for A and twice the number for B"
    is equal to (3a - b) / (a + 2b) -/
theorem expression_equality (a b : ℝ) : 
  (3 * a - b) / (a + 2 * b) = 
  (3 * a - b) / (a + 2 * b) := by sorry

end NUMINAMATH_CALUDE_expression_equality_l2843_284392


namespace NUMINAMATH_CALUDE_alpha_values_l2843_284349

theorem alpha_values (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 2 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^4 - 1) = 4 * Complex.abs (α - 1)) :
  α = Complex.I * Real.sqrt 3 ∨ α = -Complex.I * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_alpha_values_l2843_284349


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2843_284314

/-- Given two parallel vectors a and b in R³, where a = (2, -1, 2) and b = (-4, 2, x),
    prove that x = -4. -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ × ℝ) (x : ℝ) :
  a = (2, -1, 2) →
  b = (-4, 2, x) →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  x = -4 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2843_284314


namespace NUMINAMATH_CALUDE_probability_non_adjacent_rational_terms_l2843_284384

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def non_adjacent_arrangements (irrational_terms rational_terms : ℕ) : ℕ :=
  Nat.factorial irrational_terms * Nat.choose (irrational_terms + 1) rational_terms

theorem probability_non_adjacent_rational_terms 
  (total_terms : ℕ) 
  (irrational_terms : ℕ) 
  (rational_terms : ℕ) 
  (h1 : total_terms = irrational_terms + rational_terms) 
  (h2 : irrational_terms = 6) 
  (h3 : rational_terms = 3) :
  (non_adjacent_arrangements irrational_terms rational_terms : ℚ) / 
  (total_arrangements total_terms : ℚ) = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_adjacent_rational_terms_l2843_284384


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2843_284348

/-- The asymptotes of the hyperbola (y²/16) - (x²/9) = 1 are y = ±(4/3)x -/
theorem hyperbola_asymptotes :
  let hyperbola := (fun (x y : ℝ) => (y^2 / 16) - (x^2 / 9) = 1)
  ∃ (m : ℝ), m > 0 ∧
    (∀ (x y : ℝ), hyperbola x y → (y = m*x ∨ y = -m*x)) ∧
    m = 4/3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2843_284348


namespace NUMINAMATH_CALUDE_tax_reduction_percentage_l2843_284387

/-- Proves that if a tax rate is reduced by X%, consumption increases by 15%,
    and revenue decreases by 3.4%, then X = 16%. -/
theorem tax_reduction_percentage
  (T : ℝ)  -- Original tax rate (in percentage)
  (X : ℝ)  -- Percentage by which tax is reduced
  (h1 : T > 0)  -- Assumption that original tax rate is positive
  (h2 : X > 0)  -- Assumption that tax reduction is positive
  (h3 : X < T)  -- Assumption that tax reduction is less than original tax
  : ((T - X) / 100 * 115 = T / 100 * 96.6) → X = 16 :=
by sorry

end NUMINAMATH_CALUDE_tax_reduction_percentage_l2843_284387


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2843_284351

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2843_284351


namespace NUMINAMATH_CALUDE_fixed_point_and_parabola_l2843_284391

/-- The fixed point P that the line passes through for all values of a -/
def P : ℝ × ℝ := (2, -8)

/-- The line equation for any real number a -/
def line_equation (a x y : ℝ) : Prop :=
  (2*a + 3)*x + y - 4*a + 2 = 0

/-- The parabola equation with y-axis as the axis of symmetry -/
def parabola_equation_y (x y : ℝ) : Prop :=
  y^2 = 32*x

/-- The parabola equation with x-axis as the axis of symmetry -/
def parabola_equation_x (x y : ℝ) : Prop :=
  x^2 = -1/2*y

theorem fixed_point_and_parabola :
  (∀ a : ℝ, line_equation a P.1 P.2) ∧
  (parabola_equation_y P.1 P.2 ∨ parabola_equation_x P.1 P.2) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_and_parabola_l2843_284391


namespace NUMINAMATH_CALUDE_gasoline_canister_detonation_probability_l2843_284356

/-- The probability of detonating a gasoline canister -/
theorem gasoline_canister_detonation_probability :
  let n : ℕ := 5  -- number of available shots
  let p : ℚ := 2/3  -- probability of hitting the target
  let q : ℚ := 1 - p  -- probability of missing the target
  -- Assumption: shots are independent (implied by using binomial probability)
  -- Assumption: first successful hit causes a leak, second causes detonation (implied by the problem setup)
  232/243 = 1 - (q^n + n * q^(n-1) * p) :=
by sorry

end NUMINAMATH_CALUDE_gasoline_canister_detonation_probability_l2843_284356


namespace NUMINAMATH_CALUDE_fraction_addition_l2843_284307

theorem fraction_addition : (4 / 7 : ℚ) / 5 + 1 / 3 = 47 / 105 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l2843_284307


namespace NUMINAMATH_CALUDE_star_sum_24_five_pointed_star_24_l2843_284300

/-- Represents the vertices of a five-pointed star -/
inductive StarVertex
| A | B | C | D | E | F | G | H | J | K

/-- Assignment of numbers to the vertices of the star -/
def star_assignment : StarVertex → ℤ
| StarVertex.A => 1
| StarVertex.B => 2
| StarVertex.C => 3
| StarVertex.D => 4
| StarVertex.E => 5
| StarVertex.F => 10
| StarVertex.G => 12
| StarVertex.H => 9
| StarVertex.J => 6
| StarVertex.K => 8

/-- The set of all straight lines in the star -/
def star_lines : List (List StarVertex) := [
  [StarVertex.E, StarVertex.F, StarVertex.H, StarVertex.J],
  [StarVertex.F, StarVertex.G, StarVertex.K, StarVertex.J],
  [StarVertex.H, StarVertex.J, StarVertex.K, StarVertex.B],
  [StarVertex.J, StarVertex.E, StarVertex.K, StarVertex.C],
  [StarVertex.A, StarVertex.J, StarVertex.G, StarVertex.B]
]

/-- Theorem stating that the sum of numbers on each straight line equals 24 -/
theorem star_sum_24 : ∀ line ∈ star_lines, 
  (line.map star_assignment).sum = 24 := by sorry

/-- Main theorem proving the existence of a valid assignment -/
theorem five_pointed_star_24 : 
  ∃ (f : StarVertex → ℤ), ∀ line ∈ star_lines, (line.map f).sum = 24 := by
  use star_assignment
  exact star_sum_24

end NUMINAMATH_CALUDE_star_sum_24_five_pointed_star_24_l2843_284300


namespace NUMINAMATH_CALUDE_a_n_properties_l2843_284382

def a_n (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2^n + 1 else 2^n - 1

theorem a_n_properties : ∀ n : ℕ,
  (∃ m : ℕ, if n % 2 = 0 then a_n n = 5 * m^2 else a_n n = m^2) :=
by sorry

end NUMINAMATH_CALUDE_a_n_properties_l2843_284382
