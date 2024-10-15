import Mathlib

namespace NUMINAMATH_CALUDE_x_squared_equals_one_l3223_322330

theorem x_squared_equals_one (x : ℝ) (h1 : x > 0) (h2 : Real.sin (Real.arctan x) = 1 / x) : x^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_equals_one_l3223_322330


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l3223_322389

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^4 + (x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = -8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l3223_322389


namespace NUMINAMATH_CALUDE_f_two_values_l3223_322355

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f x - f y| = |x - y|

/-- Theorem stating the possible values of f(2) given the conditions -/
theorem f_two_values (f : ℝ → ℝ) (h : special_function f) (h1 : f 1 = 3) :
  f 2 = 2 ∨ f 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_two_values_l3223_322355


namespace NUMINAMATH_CALUDE_factor_calculation_l3223_322346

theorem factor_calculation : ∃ f : ℝ, (2 * 9 + 6) * f = 72 ∧ f = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l3223_322346


namespace NUMINAMATH_CALUDE_johns_age_l3223_322308

theorem johns_age : ∃ (j : ℝ), j = 22.5 ∧ (j - 10 = (1 / 3) * (j + 15)) := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l3223_322308


namespace NUMINAMATH_CALUDE_x_gt_neg_two_necessary_not_sufficient_l3223_322373

theorem x_gt_neg_two_necessary_not_sufficient :
  (∃ x : ℝ, x > -2 ∧ (x + 2) * (x - 3) ≥ 0) ∧
  (∀ x : ℝ, (x + 2) * (x - 3) < 0 → x > -2) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_neg_two_necessary_not_sufficient_l3223_322373


namespace NUMINAMATH_CALUDE_tangent_line_and_max_value_l3223_322375

open Real

noncomputable def f (x : ℝ) := -log x + (1/2) * x^2

theorem tangent_line_and_max_value :
  (∀ x, x ∈ Set.Icc (1/Real.exp 1) (Real.sqrt (Real.exp 1)) →
    f x ≤ 1 + 1 / (2 * (Real.exp 1)^2)) ∧
  (∃ x, x ∈ Set.Icc (1/Real.exp 1) (Real.sqrt (Real.exp 1)) ∧
    f x = 1 + 1 / (2 * (Real.exp 1)^2)) ∧
  (3 * 2 - 2 * f 2 - 2 - 2 * log 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_max_value_l3223_322375


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l3223_322394

/-- A quadratic polynomial -/
def QuadraticPolynomial (R : Type*) [Field R] := R → R

/-- Property that p(n) = 1/n^2 for n = 1, 2, 3 -/
def SatisfiesCondition (p : QuadraticPolynomial ℝ) : Prop :=
  p 1 = 1 ∧ p 2 = 1/4 ∧ p 3 = 1/9

theorem quadratic_polynomial_property (p : QuadraticPolynomial ℝ) 
  (h : SatisfiesCondition p) : p 4 = -9/16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l3223_322394


namespace NUMINAMATH_CALUDE_simplify_expression_l3223_322351

theorem simplify_expression : 3 * (((1 + 2 + 3 + 4) * 3) + ((1 * 4 + 16) / 4)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3223_322351


namespace NUMINAMATH_CALUDE_race_time_comparison_l3223_322392

theorem race_time_comparison (a V : ℝ) (h_a : a > 0) (h_V : V > 0) :
  let planned_time := a / V
  let first_half_time := a / (2 * 1.25 * V)
  let second_half_time := a / (2 * 0.8 * V)
  let actual_time := first_half_time + second_half_time
  actual_time > planned_time := by sorry

end NUMINAMATH_CALUDE_race_time_comparison_l3223_322392


namespace NUMINAMATH_CALUDE_intercept_length_min_distance_l3223_322350

-- Define the family of curves C
def C (m : ℝ) (x y : ℝ) : Prop :=
  4 * x^2 + 5 * y^2 - 8 * m * x - 20 * m * y + 24 * m^2 - 20 = 0

-- Define the circle
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * x + 4 * Real.sqrt 6 * y + 30 = 0

-- Define the lines
def Line1 (x y : ℝ) : Prop := y = 2 * x + 2
def Line2 (x y : ℝ) : Prop := y = 2 * x - 2

-- Theorem for part 1
theorem intercept_length (m : ℝ) :
  ∀ x y, C m x y → (Line1 x y ∨ Line2 x y) →
  ∃ x1 y1 x2 y2, C m x1 y1 ∧ C m x2 y2 ∧
  ((Line1 x1 y1 ∧ Line1 x2 y2) ∨ (Line2 x1 y1 ∧ Line2 x2 y2)) ∧
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 5 * Real.sqrt 5 / 3 :=
sorry

-- Theorem for part 2
theorem min_distance :
  ∀ m x1 y1 x2 y2, C m x1 y1 → Circle x2 y2 →
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) ≥ 2 * Real.sqrt 5 - 1 :=
sorry

end NUMINAMATH_CALUDE_intercept_length_min_distance_l3223_322350


namespace NUMINAMATH_CALUDE_parabola_b_value_l3223_322324

/-- A parabola with equation y = x^2 + bx + 3 passing through the points (1, 5), (3, 5), and (0, 3) has b = 1 -/
theorem parabola_b_value : ∃ b : ℝ,
  (∀ x y : ℝ, y = x^2 + b*x + 3 →
    ((x = 1 ∧ y = 5) ∨ (x = 3 ∧ y = 5) ∨ (x = 0 ∧ y = 3))) →
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3223_322324


namespace NUMINAMATH_CALUDE_remaining_food_feeds_children_l3223_322335

/-- Represents the amount of food required for one adult. -/
def adult_meal : ℚ := 1

/-- Represents the amount of food required for one child. -/
def child_meal : ℚ := 7/9

/-- Represents the total amount of food available. -/
def total_food : ℚ := 70 * adult_meal

/-- Theorem stating that if 35 adults have their meal, the remaining food can feed 45 children. -/
theorem remaining_food_feeds_children : 
  total_food - 35 * adult_meal = 45 * child_meal := by
  sorry

#check remaining_food_feeds_children

end NUMINAMATH_CALUDE_remaining_food_feeds_children_l3223_322335


namespace NUMINAMATH_CALUDE_complement_intersection_eq_set_l3223_322307

def U : Finset ℕ := {1,2,3,4,5}
def A : Finset ℕ := {1,2,3}
def B : Finset ℕ := {3,4,5}

theorem complement_intersection_eq_set : 
  (U \ (A ∩ B)) = {1,2,4,5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_eq_set_l3223_322307


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3223_322358

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3223_322358


namespace NUMINAMATH_CALUDE_similar_right_triangles_leg_length_l3223_322300

/-- Given two similar right triangles, where one triangle has legs of length 12 and 9,
    and the other triangle has one leg of length 6, prove that the length of the other
    leg in the second triangle is 4.5. -/
theorem similar_right_triangles_leg_length
  (a b c d : ℝ)
  (h1 : a = 12)
  (h2 : b = 9)
  (h3 : c = 6)
  (h4 : a / b = c / d)
  : d = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_similar_right_triangles_leg_length_l3223_322300


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3223_322332

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = 2 * x + 2 * y + 8) :
  ∀ x : ℝ, g x = -2 * x - 7 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3223_322332


namespace NUMINAMATH_CALUDE_intersection_empty_union_equals_B_l3223_322362

-- Define set A
def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Theorem for the first part
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

-- Theorem for the second part
theorem union_equals_B (a : ℝ) : A a ∪ B = B ↔ a > 5 ∨ a < -4 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_union_equals_B_l3223_322362


namespace NUMINAMATH_CALUDE_intersection_A_B_l3223_322323

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {x : ℝ | x * (x - 1) ≤ 0}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3223_322323


namespace NUMINAMATH_CALUDE_decreasing_interval_of_quadratic_l3223_322369

def f (x : ℝ) := x^2 - 2*x - 3

theorem decreasing_interval_of_quadratic :
  ∀ x : ℝ, (∀ y : ℝ, y < x → f y < f x) ↔ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_quadratic_l3223_322369


namespace NUMINAMATH_CALUDE_calculation_proof_l3223_322327

theorem calculation_proof :
  ((-36) * (1/3 - 1/2) + 16 / (-2)^3 = 4) ∧
  ((-5 + 2) * (1/3) + 5^2 / (-5) = -6) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3223_322327


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3223_322328

-- Define a normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P {α : Type} (event : Set α) : ℝ := sorry

-- Define the random variable ξ
def ξ : normal_distribution 0 σ := sorry

-- State the theorem
theorem normal_distribution_probability 
  (h : P {x | -2 ≤ x ∧ x ≤ 0} = 0.4) : 
  P {x | x > 2} = 0.1 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3223_322328


namespace NUMINAMATH_CALUDE_staff_discount_percentage_l3223_322302

theorem staff_discount_percentage (d : ℝ) : 
  d > 0 →  -- Assuming the original price is positive
  let discounted_price := 0.85 * d  -- Price after 15% discount
  let final_price := 0.765 * d      -- Price staff member pays
  let staff_discount_percent := (discounted_price - final_price) / discounted_price * 100
  staff_discount_percent = 10 := by
sorry

end NUMINAMATH_CALUDE_staff_discount_percentage_l3223_322302


namespace NUMINAMATH_CALUDE_min_operations_to_250_l3223_322347

/-- Represents the possible operations: adding 1 or multiplying by 2 -/
inductive Operation
  | addOne
  | multiplyTwo

/-- Applies an operation to a natural number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.addOne => n + 1
  | Operation.multiplyTwo => n * 2

/-- Checks if a sequence of operations transforms 1 into the target -/
def isValidSequence (target : ℕ) (ops : List Operation) : Prop :=
  ops.foldl applyOperation 1 = target

/-- The minimum number of operations needed to transform 1 into 250 -/
def minOperations : ℕ := 12

/-- Theorem stating that the minimum number of operations to reach 250 from 1 is 12 -/
theorem min_operations_to_250 :
  (∃ (ops : List Operation), isValidSequence 250 ops ∧ ops.length = minOperations) ∧
  (∀ (ops : List Operation), isValidSequence 250 ops → ops.length ≥ minOperations) :=
sorry

end NUMINAMATH_CALUDE_min_operations_to_250_l3223_322347


namespace NUMINAMATH_CALUDE_cafeteria_combos_l3223_322305

/-- Represents the number of options for each part of the lunch combo -/
structure LunchOptions where
  mainDishes : Nat
  sides : Nat
  drinks : Nat
  desserts : Nat

/-- Calculates the total number of distinct lunch combos -/
def totalCombos (options : LunchOptions) : Nat :=
  options.mainDishes * options.sides * options.drinks * options.desserts

/-- The specific lunch options available in the cafeteria -/
def cafeteriaOptions : LunchOptions :=
  { mainDishes := 3
  , sides := 2
  , drinks := 2
  , desserts := 2 }

theorem cafeteria_combos :
  totalCombos cafeteriaOptions = 24 := by
  sorry

#eval totalCombos cafeteriaOptions

end NUMINAMATH_CALUDE_cafeteria_combos_l3223_322305


namespace NUMINAMATH_CALUDE_tan_identities_l3223_322378

theorem tan_identities (α : Real) (h : Real.tan (π / 4 + α) = 3) :
  (Real.tan α = 1 / 2) ∧
  (Real.tan (2 * α) = 4 / 3) ∧
  ((2 * Real.sin α * Real.cos α + 3 * Real.cos (2 * α)) / 
   (5 * Real.cos (2 * α) - 3 * Real.sin (2 * α)) = 13 / 3) := by
  sorry

end NUMINAMATH_CALUDE_tan_identities_l3223_322378


namespace NUMINAMATH_CALUDE_standard_deviation_of_data_set_l3223_322398

def data_set : List ℝ := [10, 5, 4, 2, 2, 1]

theorem standard_deviation_of_data_set :
  let x := data_set[2]
  ∀ (mode median : ℝ),
    x ≠ 5 →
    mode = 2 →
    median = (x + 2) / 2 →
    mode = 2/3 * median →
    let mean := (data_set.sum) / (data_set.length : ℝ)
    let variance := (data_set.map (λ y => (y - mean)^2)).sum / (data_set.length : ℝ)
    Real.sqrt variance = 3 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_of_data_set_l3223_322398


namespace NUMINAMATH_CALUDE_second_level_treasures_is_two_l3223_322337

/-- Represents the number of points scored per treasure -/
def points_per_treasure : ℕ := 4

/-- Represents the number of treasures found on the first level -/
def first_level_treasures : ℕ := 6

/-- Represents the total score -/
def total_score : ℕ := 32

/-- Calculates the number of treasures found on the second level -/
def second_level_treasures : ℕ :=
  (total_score - (first_level_treasures * points_per_treasure)) / points_per_treasure

/-- Theorem stating that the number of treasures found on the second level is 2 -/
theorem second_level_treasures_is_two : second_level_treasures = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_level_treasures_is_two_l3223_322337


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3223_322396

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3223_322396


namespace NUMINAMATH_CALUDE_solution_range_l3223_322320

def P (a : ℝ) : Set ℝ := {x : ℝ | (x + 1) / (x + a) < 2}

theorem solution_range (a : ℝ) : (1 ∉ P a) ↔ a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l3223_322320


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_26_l3223_322395

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay_rate (hours : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (ac_efficiency_decrease : ℝ) (pay_per_mile : ℝ) (gas_price : ℝ) 
  (gas_price_increase : ℝ) : ℝ :=
  let distance := hours * speed
  let adjusted_fuel_efficiency := fuel_efficiency * (1 - ac_efficiency_decrease)
  let gas_used := distance / adjusted_fuel_efficiency
  let earnings := pay_per_mile * distance
  let new_gas_price := gas_price * (1 + gas_price_increase)
  let gas_cost := new_gas_price * gas_used
  let net_earnings := earnings - gas_cost
  let net_rate := net_earnings / hours
  net_rate

/-- Proves that the driver's net rate of pay is $26 per hour under given conditions --/
theorem driver_net_pay_is_26 :
  driver_net_pay_rate 3 50 30 0.1 0.6 2 0.2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_26_l3223_322395


namespace NUMINAMATH_CALUDE_segment_movement_area_reduction_l3223_322391

theorem segment_movement_area_reduction (AB d : ℝ) (hAB : AB > 0) (hd : d > 0) :
  ∃ (swept_area : ℝ), swept_area < (AB * d) / 10000 ∧ swept_area ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_segment_movement_area_reduction_l3223_322391


namespace NUMINAMATH_CALUDE_price_markup_markdown_l3223_322387

theorem price_markup_markdown (x : ℝ) (h : x > 0) : x * (1 + 0.1) * (1 - 0.1) < x := by
  sorry

end NUMINAMATH_CALUDE_price_markup_markdown_l3223_322387


namespace NUMINAMATH_CALUDE_second_group_size_l3223_322312

/-- The number of men in the first group -/
def men_group1 : ℕ := 4

/-- The number of hours worked per day by the first group -/
def hours_per_day_group1 : ℕ := 10

/-- The earnings per week of the first group in rupees -/
def earnings_group1 : ℕ := 1000

/-- The number of hours worked per day by the second group -/
def hours_per_day_group2 : ℕ := 6

/-- The earnings per week of the second group in rupees -/
def earnings_group2 : ℕ := 1350

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of men in the second group -/
def men_group2 : ℕ := 9

theorem second_group_size :
  men_group2 * hours_per_day_group2 * days_per_week * earnings_group1 =
  men_group1 * hours_per_day_group1 * days_per_week * earnings_group2 :=
by sorry

end NUMINAMATH_CALUDE_second_group_size_l3223_322312


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3223_322338

theorem complex_number_quadrant : ∃ (z : ℂ), 
  (z + Complex.I) * (1 - 2 * Complex.I) = 2 ∧ 
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3223_322338


namespace NUMINAMATH_CALUDE_f_equality_min_t_value_range_N_minus_n_l3223_322360

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - x - 3

-- Define the function g
def g (a x : ℝ) : ℝ := f x + 2 * x^3 - (a + 2) * x^2 + x + 5

-- Theorem 1: Prove that f(2x-1) = 8x^2 - 10x
theorem f_equality (x : ℝ) : f (2 * x - 1) = 8 * x^2 - 10 * x := by sorry

-- Theorem 2: Prove the minimum value of t
theorem min_t_value : 
  ∃ t : ℝ, t = 2 * Real.exp 2 - 2 ∧ 
  ∀ x ∈ Set.Icc (-2) 2, f (Real.exp x) ≤ t * Real.exp x - 3 + Real.exp 2 ∧
  ∀ s : ℝ, (∀ x ∈ Set.Icc (-2) 2, f (Real.exp x) ≤ s * Real.exp x - 3 + Real.exp 2) → s ≥ t := by sorry

-- Theorem 3: Prove the range of N-n
theorem range_N_minus_n (a : ℝ) (h : 0 < a ∧ a < 3) :
  let N := max (g a 0) (g a 1)
  let n := min (g a 0) (g a 1)
  ∃ d : ℝ, d = N - n ∧ 8/27 ≤ d ∧ d < 2 := by sorry

end NUMINAMATH_CALUDE_f_equality_min_t_value_range_N_minus_n_l3223_322360


namespace NUMINAMATH_CALUDE_odd_number_set_characterization_l3223_322363

def OddNumberSet : Set ℤ :=
  {x | -8 < x ∧ x < 20 ∧ ∃ k : ℤ, x = 2 * k + 1}

theorem odd_number_set_characterization :
  OddNumberSet = {x : ℤ | -8 < x ∧ x < 20 ∧ ∃ k : ℤ, x = 2 * k + 1} := by
  sorry

end NUMINAMATH_CALUDE_odd_number_set_characterization_l3223_322363


namespace NUMINAMATH_CALUDE_managers_salary_l3223_322344

/-- Proves that the manager's salary is 3400 given the conditions of the problem -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (salary_increase : ℚ) : 
  num_employees = 20 →
  avg_salary = 1300 →
  salary_increase = 100 →
  (num_employees * avg_salary + 3400) / (num_employees + 1) = avg_salary + salary_increase :=
by
  sorry

#check managers_salary

end NUMINAMATH_CALUDE_managers_salary_l3223_322344


namespace NUMINAMATH_CALUDE_sakshi_investment_dividend_l3223_322354

/-- Calculate the total dividend per annum for Sakshi's investment --/
theorem sakshi_investment_dividend
  (total_investment : ℝ)
  (investment_12_percent : ℝ)
  (price_12_percent : ℝ)
  (price_15_percent : ℝ)
  (dividend_rate_12_percent : ℝ)
  (dividend_rate_15_percent : ℝ)
  (h1 : total_investment = 12000)
  (h2 : investment_12_percent = 4000.000000000002)
  (h3 : price_12_percent = 120)
  (h4 : price_15_percent = 125)
  (h5 : dividend_rate_12_percent = 0.12)
  (h6 : dividend_rate_15_percent = 0.15) :
  ∃ (total_dividend : ℝ), abs (total_dividend - 1680) < 1 :=
sorry

end NUMINAMATH_CALUDE_sakshi_investment_dividend_l3223_322354


namespace NUMINAMATH_CALUDE_equal_solution_is_two_l3223_322304

/-- Given a system of equations for nonnegative real numbers, prove that the only solution where all numbers are equal is 2. -/
theorem equal_solution_is_two (n : ℕ) (x : ℕ → ℝ) : 
  n > 2 →
  (∀ k, k ∈ Finset.range n → x k ≥ 0) →
  (∀ k, k ∈ Finset.range n → x k + x ((k + 1) % n) = (x ((k + 2) % n))^2) →
  (∀ i j, i ∈ Finset.range n → j ∈ Finset.range n → x i = x j) →
  (∀ k, k ∈ Finset.range n → x k = 2) := by
sorry

end NUMINAMATH_CALUDE_equal_solution_is_two_l3223_322304


namespace NUMINAMATH_CALUDE_trajectory_and_range_l3223_322315

-- Define the circle D
def circle_D (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 32

-- Define point P
def P : ℝ × ℝ := (-6, 3)

-- Define the trajectory of M
def trajectory_M (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 8

-- Define the range of t
def t_range (t : ℝ) : Prop :=
  t ∈ Set.Icc (-Real.sqrt 5 - 1) (-Real.sqrt 5 + 1) ∪
      Set.Icc (Real.sqrt 5 - 1) (Real.sqrt 5 + 1)

theorem trajectory_and_range :
  (∀ x y : ℝ, ∃ x_H y_H : ℝ,
    circle_D x_H y_H ∧
    x = (x_H + P.1) / 2 ∧
    y = (y_H + P.2) / 2 →
    trajectory_M x y) ∧
  (∀ k t : ℝ,
    (∃ x_B y_B x_C y_C : ℝ,
      trajectory_M x_B y_B ∧
      trajectory_M x_C y_C ∧
      y_B = k * x_B ∧
      y_C = k * x_C ∧
      (x_B - 0) * (x_C - 0) + (y_B - t) * (y_C - t) = 0) →
    t_range t) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_range_l3223_322315


namespace NUMINAMATH_CALUDE_fraction_condition_necessary_not_sufficient_l3223_322325

theorem fraction_condition_necessary_not_sufficient :
  ∀ x : ℝ, (|x - 1| < 1 → (x + 3) / (x - 2) < 0) ∧
  ¬(∀ x : ℝ, (x + 3) / (x - 2) < 0 → |x - 1| < 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_condition_necessary_not_sufficient_l3223_322325


namespace NUMINAMATH_CALUDE_stratified_sample_female_count_l3223_322385

/-- Calculates the number of female athletes in a stratified sample -/
theorem stratified_sample_female_count 
  (total_athletes : ℕ) 
  (female_athletes : ℕ) 
  (male_athletes : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = female_athletes + male_athletes)
  (h2 : total_athletes = 98)
  (h3 : female_athletes = 42)
  (h4 : male_athletes = 56)
  (h5 : sample_size = 28) :
  (female_athletes : ℚ) * (sample_size : ℚ) / (total_athletes : ℚ) = 12 := by
  sorry

#check stratified_sample_female_count

end NUMINAMATH_CALUDE_stratified_sample_female_count_l3223_322385


namespace NUMINAMATH_CALUDE_binomial_12_9_l3223_322319

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l3223_322319


namespace NUMINAMATH_CALUDE_sum_of_first_five_terms_l3223_322365

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem sum_of_first_five_terms
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 1 = 1) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 31 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_five_terms_l3223_322365


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l3223_322345

-- Define the total number of students
def total_students : ℕ := 800

-- Define the number of students preferring spaghetti
def spaghetti_preference : ℕ := 320

-- Define the number of students preferring fettuccine
def fettuccine_preference : ℕ := 160

-- Theorem to prove the ratio
theorem pasta_preference_ratio : 
  (spaghetti_preference : ℚ) / (fettuccine_preference : ℚ) = 2 := by
  sorry


end NUMINAMATH_CALUDE_pasta_preference_ratio_l3223_322345


namespace NUMINAMATH_CALUDE_triangle_side_length_squared_l3223_322306

theorem triangle_side_length_squared (A B C : ℝ × ℝ) :
  let area := 10
  let tan_ABC := 5
  area = (1/2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1) →
  tan_ABC = (B.2 - A.2) / (B.1 - A.1) →
  ∃ (AC_squared : ℝ), AC_squared = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧
    AC_squared ≥ -8 + 8 * Real.sqrt 26 :=
by sorry

#check triangle_side_length_squared

end NUMINAMATH_CALUDE_triangle_side_length_squared_l3223_322306


namespace NUMINAMATH_CALUDE_problems_left_to_solve_l3223_322333

def math_test (total_problems : ℕ) (first_20min : ℕ) (second_20min : ℕ) : Prop :=
  total_problems = 75 ∧
  first_20min = 10 ∧
  second_20min = 2 * first_20min ∧
  total_problems - (first_20min + second_20min) = 45

theorem problems_left_to_solve :
  ∀ (total_problems first_20min second_20min : ℕ),
    math_test total_problems first_20min second_20min →
    total_problems - (first_20min + second_20min) = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_problems_left_to_solve_l3223_322333


namespace NUMINAMATH_CALUDE_runner_ends_at_start_l3223_322381

/-- A runner on a circular track --/
structure Runner where
  start : ℝ  -- Starting position on the track (in feet)
  distance : ℝ  -- Total distance run (in feet)

/-- The circular track --/
def track_circumference : ℝ := 60

/-- Theorem: A runner who starts at any point and runs exactly 5400 feet will end at the same point --/
theorem runner_ends_at_start (runner : Runner) (h : runner.distance = 5400) :
  runner.start = (runner.start + runner.distance) % track_circumference := by
  sorry

end NUMINAMATH_CALUDE_runner_ends_at_start_l3223_322381


namespace NUMINAMATH_CALUDE_savings_problem_l3223_322329

theorem savings_problem (S : ℝ) : 
  (S * 1.1 * (2 / 10) = 44) → S = 200 := by sorry

end NUMINAMATH_CALUDE_savings_problem_l3223_322329


namespace NUMINAMATH_CALUDE_phone_bill_ratio_l3223_322386

theorem phone_bill_ratio (jan_total feb_total internet_charge : ℚ)
  (h1 : jan_total = 46)
  (h2 : feb_total = 76)
  (h3 : internet_charge = 16) :
  (feb_total - internet_charge) / (jan_total - internet_charge) = 2 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_ratio_l3223_322386


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3223_322384

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Statement for the first expression
theorem simplify_expression_1 : 
  2 * Real.sqrt 3 * (1.5 : ℝ) ^ (1/3) * 12 ^ (1/6) = 6 := by sorry

-- Statement for the second expression
theorem simplify_expression_2 : 
  log10 25 + (2/3) * log10 8 + log10 5 * log10 20 + (log10 2)^2 = 3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3223_322384


namespace NUMINAMATH_CALUDE_sequence_with_nondivisible_sums_l3223_322341

theorem sequence_with_nondivisible_sums (k : ℕ) (h : Even k) (h' : k > 0) :
  ∃ π : Fin (k - 1) → Fin (k - 1), Function.Bijective π ∧
    ∀ (i j : Fin (k - 1)), i ≤ j →
      ¬(k ∣ (Finset.sum (Finset.Icc i j) (fun n => (π n).val + 1))) :=
sorry

end NUMINAMATH_CALUDE_sequence_with_nondivisible_sums_l3223_322341


namespace NUMINAMATH_CALUDE_pool_water_volume_l3223_322397

/-- The volume of water in a cylindrical pool with a cylindrical column inside -/
theorem pool_water_volume 
  (pool_diameter : ℝ) 
  (pool_depth : ℝ) 
  (column_diameter : ℝ) 
  (column_depth : ℝ) 
  (h_pool_diameter : pool_diameter = 20)
  (h_pool_depth : pool_depth = 6)
  (h_column_diameter : column_diameter = 4)
  (h_column_depth : column_depth = pool_depth) :
  let pool_radius : ℝ := pool_diameter / 2
  let column_radius : ℝ := column_diameter / 2
  let pool_volume : ℝ := π * pool_radius^2 * pool_depth
  let column_volume : ℝ := π * column_radius^2 * column_depth
  pool_volume - column_volume = 576 * π := by
sorry


end NUMINAMATH_CALUDE_pool_water_volume_l3223_322397


namespace NUMINAMATH_CALUDE_original_cat_count_l3223_322317

theorem original_cat_count (original_dogs original_cats current_dogs current_cats : ℕ) :
  original_dogs = original_cats / 2 →
  current_dogs = original_dogs + 20 →
  current_dogs = 2 * current_cats →
  current_cats = 20 →
  original_cats = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_original_cat_count_l3223_322317


namespace NUMINAMATH_CALUDE_inequalities_in_quadrants_I_and_II_exists_points_in_both_quadrants_l3223_322374

/-- Represents a point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defines the region satisfying the given inequalities -/
def SatisfiesInequalities (p : Point) : Prop :=
  p.y > -2 * p.x + 3 ∧ p.y > 1/2 * p.x + 1

/-- Checks if a point is in Quadrant I -/
def InQuadrantI (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a point is in Quadrant II -/
def InQuadrantII (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating that all points satisfying the inequalities are in Quadrants I and II -/
theorem inequalities_in_quadrants_I_and_II :
  ∀ p : Point, SatisfiesInequalities p → (InQuadrantI p ∨ InQuadrantII p) :=
by
  sorry

/-- Theorem stating that there exist points in both Quadrants I and II that satisfy the inequalities -/
theorem exists_points_in_both_quadrants :
  (∃ p : Point, SatisfiesInequalities p ∧ InQuadrantI p) ∧
  (∃ p : Point, SatisfiesInequalities p ∧ InQuadrantII p) :=
by
  sorry

end NUMINAMATH_CALUDE_inequalities_in_quadrants_I_and_II_exists_points_in_both_quadrants_l3223_322374


namespace NUMINAMATH_CALUDE_symmetry_implies_exponential_l3223_322336

-- Define the logarithm function base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the symmetry condition
def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_exponential (f : ℝ → ℝ) :
  (∀ x > 0, f (log3 x) = x) →
  symmetric_wrt_y_eq_x f log3 →
  ∀ x, f x = 3^x :=
sorry

end NUMINAMATH_CALUDE_symmetry_implies_exponential_l3223_322336


namespace NUMINAMATH_CALUDE_stool_height_is_85_alice_can_reach_light_bulb_l3223_322361

/-- The minimum height of the stool Alice needs to reach the light bulb -/
def stool_height : ℝ :=
  let ceiling_height : ℝ := 280  -- in cm
  let light_bulb_height : ℝ := ceiling_height - 15
  let alice_height : ℝ := 150  -- in cm
  let alice_reach : ℝ := alice_height + 30
  light_bulb_height - alice_reach

theorem stool_height_is_85 :
  stool_height = 85 := by sorry

/-- Alice can reach the light bulb with the calculated stool height -/
theorem alice_can_reach_light_bulb :
  let ceiling_height : ℝ := 280  -- in cm
  let light_bulb_height : ℝ := ceiling_height - 15
  let alice_height : ℝ := 150  -- in cm
  let alice_reach : ℝ := alice_height + 30
  alice_reach + stool_height = light_bulb_height := by sorry

end NUMINAMATH_CALUDE_stool_height_is_85_alice_can_reach_light_bulb_l3223_322361


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l3223_322348

/-- The probability of drawing a red ball from a bag with red and black balls -/
theorem probability_of_red_ball (red_balls black_balls : ℕ) : 
  red_balls = 3 → black_balls = 9 → 
  (red_balls : ℚ) / (red_balls + black_balls) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l3223_322348


namespace NUMINAMATH_CALUDE_cube_surface_area_doubles_l3223_322364

/-- Theorem: Doubling the edge length of a cube increases its surface area by a factor of 4 -/
theorem cube_surface_area_doubles (a : ℝ) (h : a > 0) :
  (6 * (2 * a)^2) / (6 * a^2) = 4 := by
  sorry

#check cube_surface_area_doubles

end NUMINAMATH_CALUDE_cube_surface_area_doubles_l3223_322364


namespace NUMINAMATH_CALUDE_rahul_savings_fraction_l3223_322382

def total_savings : ℕ := 180000
def ppf_savings : ℕ := 72000
def nsc_savings : ℕ := total_savings - ppf_savings

def fraction_equality (x : ℚ) : Prop :=
  x * nsc_savings = (1/2 : ℚ) * ppf_savings

theorem rahul_savings_fraction : 
  ∃ (x : ℚ), fraction_equality x ∧ x = (1/3 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_rahul_savings_fraction_l3223_322382


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3223_322339

-- Define the quadratic inequality
def quadratic_inequality (a x : ℝ) : Prop := a * x^2 - 2 * x + a ≤ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | quadratic_inequality a x}

theorem quadratic_inequality_solution :
  (∃! x, x ∈ solution_set 1) ∧
  (0 ∈ solution_set a ∧ -1 ∉ solution_set a → a ∈ Set.Ioc (-1) 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3223_322339


namespace NUMINAMATH_CALUDE_total_amount_is_952_20_l3223_322372

/-- Calculate the total amount paid for three items with given original prices, discounts, and sales taxes. -/
def total_amount_paid (vase_price teacups_price plate_price : ℝ)
                      (vase_discount teacups_discount : ℝ)
                      (vase_tax teacups_tax plate_tax : ℝ) : ℝ :=
  let vase_sale_price := vase_price * (1 - vase_discount)
  let teacups_sale_price := teacups_price * (1 - teacups_discount)
  let vase_total := vase_sale_price * (1 + vase_tax)
  let teacups_total := teacups_sale_price * (1 + teacups_tax)
  let plate_total := plate_price * (1 + plate_tax)
  vase_total + teacups_total + plate_total

/-- The total amount paid for the three porcelain items is $952.20. -/
theorem total_amount_is_952_20 :
  total_amount_paid 200 300 500 0.35 0.20 0.10 0.08 0.10 = 952.20 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_952_20_l3223_322372


namespace NUMINAMATH_CALUDE_walmart_gift_card_value_l3223_322366

/-- Given information about gift cards and their usage, determine the value of each Walmart gift card -/
theorem walmart_gift_card_value 
  (best_buy_count : ℕ) 
  (best_buy_value : ℕ) 
  (walmart_count : ℕ) 
  (used_best_buy : ℕ) 
  (used_walmart : ℕ) 
  (total_remaining_value : ℕ) :
  best_buy_count = 6 →
  best_buy_value = 500 →
  walmart_count = 9 →
  used_best_buy = 1 →
  used_walmart = 2 →
  total_remaining_value = 3900 →
  (walmart_count - used_walmart) * 
    ((total_remaining_value - (best_buy_count - used_best_buy) * best_buy_value) / 
     (walmart_count - used_walmart)) = 
  (walmart_count - used_walmart) * 200 :=
by sorry

end NUMINAMATH_CALUDE_walmart_gift_card_value_l3223_322366


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3223_322322

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}

theorem complement_of_A_in_U :
  U \ A = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3223_322322


namespace NUMINAMATH_CALUDE_poster_ratio_l3223_322399

theorem poster_ratio (total : ℕ) (small_fraction : ℚ) (large : ℕ) : 
  total = 50 → 
  small_fraction = 2 / 5 → 
  large = 5 → 
  (total - (small_fraction * total).num - large) / total = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_poster_ratio_l3223_322399


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3223_322353

def f (x : ℝ) : ℝ := -x^2 + x + 6

theorem quadratic_function_properties :
  (f (-3) = -6 ∧ f 0 = 6 ∧ f 2 = 4) →
  (∀ x : ℝ, f x = -x^2 + x + 6) ∧
  (∀ x : ℝ, f x ≤ 25/4) ∧
  (f (1/2) = 25/4) ∧
  (∀ x : ℝ, f (x + 1/2) = f (1/2 - x)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3223_322353


namespace NUMINAMATH_CALUDE_will_remaining_candy_l3223_322359

/-- Represents the number of pieces in each type of candy box -/
structure CandyBox where
  chocolate : Nat
  mint : Nat
  caramel : Nat

/-- Represents the number of boxes of each type of candy -/
structure CandyInventory where
  chocolate : Nat
  mint : Nat
  caramel : Nat

def initial_inventory : CandyInventory := 
  { chocolate := 7, mint := 5, caramel := 4 }

def pieces_per_box : CandyBox := 
  { chocolate := 12, mint := 15, caramel := 10 }

def boxes_given_away : CandyInventory := 
  { chocolate := 3, mint := 2, caramel := 1 }

/-- Calculates the total number of candy pieces for a given inventory -/
def total_pieces (inventory : CandyInventory) (box : CandyBox) : Nat :=
  inventory.chocolate * box.chocolate + 
  inventory.mint * box.mint + 
  inventory.caramel * box.caramel

/-- Calculates the remaining inventory after giving away boxes -/
def remaining_inventory (initial : CandyInventory) (given_away : CandyInventory) : CandyInventory :=
  { chocolate := initial.chocolate - given_away.chocolate,
    mint := initial.mint - given_away.mint,
    caramel := initial.caramel - given_away.caramel }

theorem will_remaining_candy : 
  total_pieces (remaining_inventory initial_inventory boxes_given_away) pieces_per_box = 123 := by
  sorry

end NUMINAMATH_CALUDE_will_remaining_candy_l3223_322359


namespace NUMINAMATH_CALUDE_htf_sequence_probability_l3223_322331

/-- A fair coin has equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of a specific sequence of three independent coin flips -/
def prob_sequence (p : ℝ) : ℝ := p * p * p

theorem htf_sequence_probability :
  ∀ p : ℝ, fair_coin p → prob_sequence p = 1/8 := by sorry

end NUMINAMATH_CALUDE_htf_sequence_probability_l3223_322331


namespace NUMINAMATH_CALUDE_equation_equivalence_l3223_322368

theorem equation_equivalence (x y : ℕ) : 
  (∃ (a b c d : ℕ), x = a + 2*b + 3*c + 7*d ∧ y = b + 2*c + 5*d) ↔ 
  5*x ≥ 7*y := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3223_322368


namespace NUMINAMATH_CALUDE_no_four_digit_n_over_5_and_5n_l3223_322380

theorem no_four_digit_n_over_5_and_5n : 
  ¬ ∃ (n : ℕ), n > 0 ∧ 
    (1000 ≤ n / 5 ∧ n / 5 ≤ 9999) ∧ 
    (1000 ≤ 5 * n ∧ 5 * n ≤ 9999) :=
by sorry

end NUMINAMATH_CALUDE_no_four_digit_n_over_5_and_5n_l3223_322380


namespace NUMINAMATH_CALUDE_fifth_term_is_negative_9216_l3223_322343

def alternating_sequence (n : ℕ) : ℤ := 
  if n % 2 = 0 then (101 - n)^2 else -((101 - n)^2)

def sequence_sum (n : ℕ) : ℤ := 
  (List.range n).map alternating_sequence |>.sum

theorem fifth_term_is_negative_9216 (h : sequence_sum 100 = 5050) : 
  alternating_sequence 5 = -9216 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_negative_9216_l3223_322343


namespace NUMINAMATH_CALUDE_simplify_expression_l3223_322342

theorem simplify_expression : 5 * (18 / (-9)) * (24 / 36) = -20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3223_322342


namespace NUMINAMATH_CALUDE_system_solution_l3223_322383

theorem system_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2 * x - Real.sqrt (x * y) - 4 * Real.sqrt (x / y) + 2 = 0 ∧
   2 * x^2 + x^2 * y^4 = 18 * y^2) ↔
  ((x = 2 ∧ y = 2) ∨ (x = Real.rpow 286 (1/4) / 4 ∧ y = Real.rpow 286 (1/4))) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l3223_322383


namespace NUMINAMATH_CALUDE_fraction_problem_l3223_322371

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x = 145) (h2 : x - f * x = 58) : f = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3223_322371


namespace NUMINAMATH_CALUDE_polyhedron_volume_l3223_322388

theorem polyhedron_volume (prism_volume : ℝ) (pyramid_base_side : ℝ) (pyramid_height : ℝ) :
  prism_volume = Real.sqrt 2 - 1 →
  pyramid_base_side = 1 →
  pyramid_height = 1 / 2 →
  prism_volume + 2 * (1 / 3 * pyramid_base_side^2 * pyramid_height) = Real.sqrt 2 - 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l3223_322388


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l3223_322356

theorem quadratic_equation_solution_difference : 
  ∀ x₁ x₂ : ℝ, 
  (x₁^2 - 5*x₁ + 11 = x₁ + 27) → 
  (x₂^2 - 5*x₂ + 11 = x₂ + 27) → 
  x₁ ≠ x₂ →
  |x₁ - x₂| = 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l3223_322356


namespace NUMINAMATH_CALUDE_central_cell_only_solution_l3223_322377

/-- Represents a 5x5 grid with boolean values (true for "+", false for "-") -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Represents a subgrid position and size -/
structure Subgrid where
  row : Fin 5
  col : Fin 5
  size : Nat
  size_valid : 2 ≤ size ∧ size ≤ 5

/-- Flips the signs in a subgrid -/
def flip_subgrid (g : Grid) (sg : Subgrid) : Grid :=
  λ i j => if i < sg.row + sg.size ∧ j < sg.col + sg.size
           then !g i j
           else g i j

/-- Checks if all cells in the grid are positive -/
def all_positive (g : Grid) : Prop :=
  ∀ i j, g i j = true

/-- Initial grid with only the specified cell negative -/
def initial_grid (row col : Fin 5) : Grid :=
  λ i j => ¬(i = row ∧ j = col)

/-- Theorem stating that only the central cell as initial negative allows for all positive end state -/
theorem central_cell_only_solution :
  ∀ (row col : Fin 5),
    (∃ (moves : List Subgrid), all_positive (moves.foldl flip_subgrid (initial_grid row col))) ↔
    (row = 2 ∧ col = 2) :=
  sorry

end NUMINAMATH_CALUDE_central_cell_only_solution_l3223_322377


namespace NUMINAMATH_CALUDE_problem_solution_l3223_322309

theorem problem_solution : ∃ x : ℝ, (0.25 * x = 0.15 * 1500 - 15) ∧ (x = 840) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3223_322309


namespace NUMINAMATH_CALUDE_distinct_sums_count_l3223_322314

/-- The set of ball numbers -/
def BallNumbers : Finset ℕ := {1, 2, 3, 4, 5}

/-- The sum of two numbers drawn from BallNumbers with replacement -/
def SumOfDraws : Finset ℕ := Finset.image (λ (x : ℕ × ℕ) => x.1 + x.2) (BallNumbers.product BallNumbers)

/-- The number of distinct possible sums -/
theorem distinct_sums_count : Finset.card SumOfDraws = 9 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_count_l3223_322314


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l3223_322334

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: When a = 2, prove the solution set of f(x) ≥ 4
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Prove the range of a for which f(x) ≥ 4
theorem range_of_a :
  {a : ℝ | ∃ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l3223_322334


namespace NUMINAMATH_CALUDE_prob_rain_at_least_one_day_l3223_322393

def prob_rain_friday : ℝ := 0.6
def prob_rain_saturday : ℝ := 0.7
def prob_rain_sunday : ℝ := 0.4

theorem prob_rain_at_least_one_day :
  let prob_no_rain_friday := 1 - prob_rain_friday
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let prob_no_rain_all_days := prob_no_rain_friday * prob_no_rain_saturday * prob_no_rain_sunday
  let prob_rain_at_least_one_day := 1 - prob_no_rain_all_days
  prob_rain_at_least_one_day = 0.928 := by
sorry

end NUMINAMATH_CALUDE_prob_rain_at_least_one_day_l3223_322393


namespace NUMINAMATH_CALUDE_third_derivative_y_l3223_322370

noncomputable def y (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_y (x : ℝ) :
  HasDerivAt (fun x => (deriv (deriv y)) x) (4 / (1 + x^2)^2) x :=
sorry

end NUMINAMATH_CALUDE_third_derivative_y_l3223_322370


namespace NUMINAMATH_CALUDE_meaningful_expression_l3223_322326

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 5)) ↔ x > 5 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3223_322326


namespace NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l3223_322376

theorem maximize_x_cubed_y_fourth (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 60) :
  x^3 * y^4 ≤ (3/7 * 60)^3 * (4/7 * 60)^4 ∧
  x^3 * y^4 = (3/7 * 60)^3 * (4/7 * 60)^4 ↔ x = 3/7 * 60 ∧ y = 4/7 * 60 :=
by sorry

end NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l3223_322376


namespace NUMINAMATH_CALUDE_initial_average_production_is_50_l3223_322311

/-- Calculates the initial average daily production given the number of past days,
    today's production, and the new average including today. -/
def initialAverageProduction (n : ℕ) (todayProduction : ℕ) (newAverage : ℚ) : ℚ :=
  (newAverage * (n + 1) - todayProduction) / n

theorem initial_average_production_is_50 :
  initialAverageProduction 10 105 55 = 50 := by sorry

end NUMINAMATH_CALUDE_initial_average_production_is_50_l3223_322311


namespace NUMINAMATH_CALUDE_count_good_pairs_l3223_322303

def is_good_pair (a p : ℕ) : Prop :=
  a > p ∧ (a^3 + p^3) % (a^2 - p^2) = 0

def is_prime_less_than_20 (p : ℕ) : Prop :=
  Nat.Prime p ∧ p < 20

theorem count_good_pairs :
  ∃ (S : Finset (ℕ × ℕ)), 
    S.card = 24 ∧
    (∀ (a p : ℕ), (a, p) ∈ S ↔ is_good_pair a p ∧ is_prime_less_than_20 p) :=
sorry

end NUMINAMATH_CALUDE_count_good_pairs_l3223_322303


namespace NUMINAMATH_CALUDE_central_diamond_area_l3223_322310

/-- The area of the central diamond-shaped region in a 10x10 square --/
theorem central_diamond_area (square_side : ℝ) (h : square_side = 10) : 
  let diagonal_length : ℝ := square_side * Real.sqrt 2
  let midpoint_distance : ℝ := square_side / 2
  let diamond_area : ℝ := diagonal_length * midpoint_distance / 2
  diamond_area = 50 := by sorry

end NUMINAMATH_CALUDE_central_diamond_area_l3223_322310


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3223_322379

theorem trigonometric_identity : 
  (Real.sin (65 * π / 180) + Real.sin (15 * π / 180) * Real.sin (10 * π / 180)) / 
  (Real.sin (25 * π / 180) - Real.cos (15 * π / 180) * Real.cos (80 * π / 180)) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3223_322379


namespace NUMINAMATH_CALUDE_cost_price_correct_l3223_322321

/-- The cost price of a piece of clothing -/
def cost_price : ℝ := 108

/-- The marked price of the clothing -/
def marked_price : ℝ := 132

/-- The discount rate applied to the clothing -/
def discount_rate : ℝ := 0.1

/-- The profit rate after applying the discount -/
def profit_rate : ℝ := 0.1

/-- Theorem stating that the cost price is correct given the conditions -/
theorem cost_price_correct :
  marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_correct_l3223_322321


namespace NUMINAMATH_CALUDE_arccos_one_half_l3223_322316

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l3223_322316


namespace NUMINAMATH_CALUDE_planes_parallel_l3223_322301

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the geometric relations
variable (belongs_to : Point → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (noncoplanar : Line → Line → Prop)

-- State the theorem
theorem planes_parallel 
  (α β : Plane) (a b : Line) :
  noncoplanar a b →
  subset a α →
  subset b β →
  parallel_line_plane a β →
  parallel_line_plane b α →
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_l3223_322301


namespace NUMINAMATH_CALUDE_shoes_outside_library_l3223_322318

/-- The total number of shoes outside the library -/
def total_shoes (regular_shoes sandals slippers : ℕ) : ℕ :=
  2 * regular_shoes + 2 * sandals + 2 * slippers

/-- Proof that the total number of shoes is 20 -/
theorem shoes_outside_library :
  let total_people : ℕ := 10
  let regular_shoe_wearers : ℕ := 4
  let sandal_wearers : ℕ := 3
  let slipper_wearers : ℕ := 3
  total_people = regular_shoe_wearers + sandal_wearers + slipper_wearers →
  total_shoes regular_shoe_wearers sandal_wearers slipper_wearers = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_shoes_outside_library_l3223_322318


namespace NUMINAMATH_CALUDE_tangent_line_circle_a_value_l3223_322390

/-- A line is tangent to a circle if and only if the distance from the center of the circle to the line equals the radius of the circle. -/
axiom line_tangent_to_circle_iff_distance_eq_radius {a b c d e f : ℝ} :
  (∀ x y, a*x + b*y + c = 0 → (x - d)^2 + (y - e)^2 = f^2) ↔
  |a*d + b*e + c| / Real.sqrt (a^2 + b^2) = f

/-- Given that the line 5x + 12y + a = 0 is tangent to the circle x^2 - 2x + y^2 = 0,
    prove that a = 8 or a = -18 -/
theorem tangent_line_circle_a_value :
  (∀ x y, 5*x + 12*y + a = 0 → x^2 - 2*x + y^2 = 0) →
  a = 8 ∨ a = -18 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_a_value_l3223_322390


namespace NUMINAMATH_CALUDE_expression_simplification_l3223_322352

theorem expression_simplification (x : ℝ) (h : x ≠ 0) :
  (x * (3 - 4 * x) + 2 * x^2 * (x - 1)) / (-2 * x) = -x^2 + 3 * x - 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3223_322352


namespace NUMINAMATH_CALUDE_marbles_shared_proof_l3223_322349

/-- The number of marbles Carolyn started with -/
def initial_marbles : ℕ := 47

/-- The number of marbles Carolyn ended up with after sharing -/
def final_marbles : ℕ := 5

/-- The number of marbles Carolyn shared -/
def shared_marbles : ℕ := initial_marbles - final_marbles

theorem marbles_shared_proof : shared_marbles = 42 := by
  sorry

end NUMINAMATH_CALUDE_marbles_shared_proof_l3223_322349


namespace NUMINAMATH_CALUDE_ellipse_vertex_distance_l3223_322313

/-- The distance between the vertices of the ellipse x^2/49 + y^2/64 = 1 is 16 -/
theorem ellipse_vertex_distance :
  let a := Real.sqrt (max 49 64)
  let ellipse := {(x, y) : ℝ × ℝ | x^2/49 + y^2/64 = 1}
  2 * a = 16 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_vertex_distance_l3223_322313


namespace NUMINAMATH_CALUDE_robins_hair_length_l3223_322340

/-- Robin's hair length problem -/
theorem robins_hair_length (initial_length cut_length : ℕ) (h1 : initial_length = 14) (h2 : cut_length = 13) :
  initial_length - cut_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_length_l3223_322340


namespace NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l3223_322357

theorem inverse_proportion_percentage_change 
  (x y x' y' k q : ℝ) 
  (h_positive : x > 0 ∧ y > 0)
  (h_inverse : x * y = k)
  (h_y_decrease : y' = y * (1 - q / 100))
  (h_constant : x' * y' = k) :
  (x' - x) / x * 100 = 100 * q / (100 - q) := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l3223_322357


namespace NUMINAMATH_CALUDE_village_language_probability_l3223_322367

/-- Given a village with the following properties:
  - Total population is 1500
  - 800 people speak Tamil
  - 650 people speak English
  - 250 people speak both Tamil and English
  Prove that the probability of a randomly chosen person speaking neither English nor Tamil is 1/5 -/
theorem village_language_probability (total : ℕ) (tamil : ℕ) (english : ℕ) (both : ℕ)
  (h_total : total = 1500)
  (h_tamil : tamil = 800)
  (h_english : english = 650)
  (h_both : both = 250) :
  (total - (tamil + english - both)) / total = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_village_language_probability_l3223_322367
