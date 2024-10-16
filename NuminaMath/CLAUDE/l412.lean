import Mathlib

namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l412_41220

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 7 / 13 → Nat.gcd a b = 15 → Nat.lcm a b = 91 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l412_41220


namespace NUMINAMATH_CALUDE_trig_equation_solution_l412_41274

theorem trig_equation_solution (z : ℝ) :
  5 * (Real.sin (2 * z))^4 - 4 * (Real.sin (2 * z))^2 * (Real.cos (2 * z))^2 - (Real.cos (2 * z))^4 + 4 * Real.cos (4 * z) = 0 →
  (∃ k : ℤ, z = π / 8 * (2 * ↑k + 1)) ∨ (∃ n : ℤ, z = π / 6 * (3 * ↑n + 1) ∨ z = π / 6 * (3 * ↑n - 1)) := by
sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l412_41274


namespace NUMINAMATH_CALUDE_expression_evaluation_l412_41270

theorem expression_evaluation (b : ℝ) :
  let x : ℝ := b + 9
  2 * x - b + 5 = b + 23 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l412_41270


namespace NUMINAMATH_CALUDE_trig_values_for_special_angle_l412_41248

/-- The intersection point of two lines -/
def intersection_point (l₁ l₂ : ℝ × ℝ → Prop) : ℝ × ℝ :=
  sorry

/-- The angle whose terminal side passes through a given point -/
def angle_from_point (p : ℝ × ℝ) : ℝ :=
  sorry

/-- The sine of an angle -/
def sine (α : ℝ) : ℝ :=
  sorry

/-- The cosine of an angle -/
def cosine (α : ℝ) : ℝ :=
  sorry

/-- The tangent of an angle -/
def tangent (α : ℝ) : ℝ :=
  sorry

theorem trig_values_for_special_angle :
  let l₁ : ℝ × ℝ → Prop := λ (x, y) ↦ x - y = 0
  let l₂ : ℝ × ℝ → Prop := λ (x, y) ↦ 2*x + y - 3 = 0
  let p := intersection_point l₁ l₂
  let α := angle_from_point p
  sine α = Real.sqrt 2 / 2 ∧ cosine α = Real.sqrt 2 / 2 ∧ tangent α = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_values_for_special_angle_l412_41248


namespace NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l412_41263

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_factorials : 
  units_digit (3 * (factorial 1 + factorial 2 + factorial 3 + factorial 4)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l412_41263


namespace NUMINAMATH_CALUDE_defective_items_probability_l412_41240

theorem defective_items_probability 
  (p_zero : ℝ) 
  (p_one : ℝ) 
  (p_two : ℝ) 
  (p_three : ℝ) 
  (h1 : p_zero = 0.18) 
  (h2 : p_one = 0.53) 
  (h3 : p_two = 0.27) 
  (h4 : p_three = 0.02) : 
  (p_two + p_three = 0.29) ∧ (p_zero + p_one = 0.71) := by
  sorry

end NUMINAMATH_CALUDE_defective_items_probability_l412_41240


namespace NUMINAMATH_CALUDE_conjugate_sum_product_l412_41297

theorem conjugate_sum_product (c d : ℝ) : 
  ((c + Real.sqrt d) + (c - Real.sqrt d) = -6) →
  ((c + Real.sqrt d) * (c - Real.sqrt d) = 4) →
  c + d = 2 := by
  sorry

end NUMINAMATH_CALUDE_conjugate_sum_product_l412_41297


namespace NUMINAMATH_CALUDE_punch_bowl_theorem_l412_41213

/-- The capacity of the punch bowl in gallons -/
def bowl_capacity : ℝ := 16

/-- The amount of punch Mark adds in the second refill -/
def second_refill : ℝ := 4

/-- The amount of punch Sally drinks -/
def sally_drinks : ℝ := 2

/-- The amount of punch Mark adds to completely fill the bowl at the end -/
def final_addition : ℝ := 12

/-- The initial amount of punch Mark added to the bowl -/
def initial_amount : ℝ := 4

theorem punch_bowl_theorem :
  let after_cousin := initial_amount / 2
  let after_second_refill := after_cousin + second_refill
  let after_sally := after_second_refill - sally_drinks
  after_sally + final_addition = bowl_capacity :=
by sorry

end NUMINAMATH_CALUDE_punch_bowl_theorem_l412_41213


namespace NUMINAMATH_CALUDE_eight_squares_sharing_two_vertices_l412_41278

/-- A square in a 2D plane -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ
  is_square : IsSquare vertices

/-- Two squares share two vertices -/
def SharesTwoVertices (s1 s2 : Square) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ s1.vertices i = s2.vertices i ∧ s1.vertices j = s2.vertices j

/-- The main theorem -/
theorem eight_squares_sharing_two_vertices (s : Square) :
  ∃ (squares : Finset Square), squares.card = 8 ∧
    ∀ s' ∈ squares, SharesTwoVertices s s' ∧
    ∀ s', SharesTwoVertices s s' → s' ∈ squares :=
  sorry

end NUMINAMATH_CALUDE_eight_squares_sharing_two_vertices_l412_41278


namespace NUMINAMATH_CALUDE_sam_distance_theorem_l412_41246

/-- Calculates the distance traveled given an average speed and time -/
def distanceTraveled (avgSpeed : ℝ) (time : ℝ) : ℝ := avgSpeed * time

theorem sam_distance_theorem (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
    (h1 : marguerite_distance = 100)
    (h2 : marguerite_time = 2.4)
    (h3 : sam_time = 3) :
  distanceTraveled (marguerite_distance / marguerite_time) sam_time = 125 := by
  sorry

#check sam_distance_theorem

end NUMINAMATH_CALUDE_sam_distance_theorem_l412_41246


namespace NUMINAMATH_CALUDE_wash_time_proof_l412_41276

/-- The number of weeks between each wash -/
def wash_interval : ℕ := 4

/-- The time in minutes it takes to wash the pillowcases -/
def wash_time : ℕ := 30

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Calculates the total time spent washing pillowcases in a year -/
def total_wash_time_per_year : ℕ :=
  (weeks_per_year / wash_interval) * wash_time

theorem wash_time_proof :
  total_wash_time_per_year = 390 :=
by sorry

end NUMINAMATH_CALUDE_wash_time_proof_l412_41276


namespace NUMINAMATH_CALUDE_one_more_book_than_movie_l412_41236

/-- The number of different movies in the 'crazy silly school' series -/
def num_movies : ℕ := 14

/-- The number of different books in the 'crazy silly school' series -/
def num_books : ℕ := 15

/-- Theorem stating that there is one more book than movie in the series -/
theorem one_more_book_than_movie : num_books - num_movies = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_more_book_than_movie_l412_41236


namespace NUMINAMATH_CALUDE_tan_triple_angle_l412_41255

theorem tan_triple_angle (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_triple_angle_l412_41255


namespace NUMINAMATH_CALUDE_problem_statement_l412_41280

theorem problem_statement (r p q : ℝ) 
  (hr : r > 0) 
  (hpq : p * q ≠ 0) 
  (hineq : p^2 * r > q^2 * r) : 
  p^2 > q^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l412_41280


namespace NUMINAMATH_CALUDE_banana_cost_l412_41209

/-- Given that 4 bananas cost $20, prove that one banana costs $5. -/
theorem banana_cost : 
  ∀ (cost : ℝ), (4 * cost = 20) → (cost = 5) := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_l412_41209


namespace NUMINAMATH_CALUDE_y_properties_l412_41214

/-- A function y(x) composed of two directly proportional components -/
def y (x : ℝ) (k₁ k₂ : ℝ) : ℝ := k₁ * (x - 3) + k₂ * (x^2 + 1)

/-- Theorem stating the properties of the function y(x) -/
theorem y_properties :
  ∃ (k₁ k₂ : ℝ),
    (y 0 k₁ k₂ = -2) ∧
    (y 1 k₁ k₂ = 4) ∧
    (∀ x, y x k₁ k₂ = 4*x^2 + 2*x - 2) ∧
    (y (-1) k₁ k₂ = 0) ∧
    (y (1/2) k₁ k₂ = 0) := by
  sorry

#check y_properties

end NUMINAMATH_CALUDE_y_properties_l412_41214


namespace NUMINAMATH_CALUDE_solution_set_correct_l412_41253

/-- The solution set of the inequality a*x^2 - (a+2)*x + 2 < 0 for x, where a ∈ ℝ -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x > 1 }
  else if 0 < a ∧ a < 2 then { x | 1 < x ∧ x < 2/a }
  else if a = 2 then ∅
  else if a > 2 then { x | 2/a < x ∧ x < 1 }
  else { x | x < 2/a ∨ x > 1 }

/-- Theorem stating that the solution_set function correctly describes the solutions of the inequality -/
theorem solution_set_correct (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a*x^2 - (a+2)*x + 2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l412_41253


namespace NUMINAMATH_CALUDE_irrational_density_l412_41224

theorem irrational_density (α : ℝ) (h_irrational : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℕ), a < m * α - n ∧ m * α - n < b :=
sorry

end NUMINAMATH_CALUDE_irrational_density_l412_41224


namespace NUMINAMATH_CALUDE_dividing_chord_length_l412_41272

/-- An octagon inscribed in a circle -/
structure InscribedOctagon :=
  (side_length_1 : ℝ)
  (side_length_2 : ℝ)
  (h1 : side_length_1 > 0)
  (h2 : side_length_2 > 0)

/-- The chord dividing the octagon into two quadrilaterals -/
def dividing_chord (o : InscribedOctagon) : ℝ := sorry

/-- Theorem stating the length of the dividing chord -/
theorem dividing_chord_length (o : InscribedOctagon) 
  (h3 : o.side_length_1 = 4)
  (h4 : o.side_length_2 = 6) : 
  dividing_chord o = 4 := by sorry

end NUMINAMATH_CALUDE_dividing_chord_length_l412_41272


namespace NUMINAMATH_CALUDE_stream_speed_l412_41275

theorem stream_speed (swim_speed : ℝ) (upstream_time downstream_time : ℝ) :
  swim_speed = 12 ∧ 
  upstream_time = 2 * downstream_time ∧ 
  upstream_time > 0 ∧ 
  downstream_time > 0 →
  ∃ stream_speed : ℝ,
    stream_speed = 4 ∧
    (swim_speed - stream_speed) * upstream_time = (swim_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l412_41275


namespace NUMINAMATH_CALUDE_peanut_mixture_equation_l412_41273

/-- Represents the cost per pound of each type of peanut --/
structure PeanutCosts where
  virginia : ℝ
  spanish : ℝ
  texan : ℝ

/-- Represents the amount of each type of peanut in pounds --/
structure PeanutAmounts where
  virginia : ℝ
  spanish : ℝ
  texan : ℝ

/-- Calculates the total cost of a peanut mixture --/
def totalCost (costs : PeanutCosts) (amounts : PeanutAmounts) : ℝ :=
  costs.virginia * amounts.virginia +
  costs.spanish * amounts.spanish +
  costs.texan * amounts.texan

/-- Calculates the total weight of a peanut mixture --/
def totalWeight (amounts : PeanutAmounts) : ℝ :=
  amounts.virginia + amounts.spanish + amounts.texan

/-- Theorem: For any mixture of 10 pounds of Virginia peanuts, S pounds of Spanish peanuts,
    and T pounds of Texan peanuts that costs $3.60/pound, the equation 0.40T - 0.60S = 1 holds --/
theorem peanut_mixture_equation (costs : PeanutCosts) (amounts : PeanutAmounts)
    (h1 : costs.virginia = 3.5)
    (h2 : costs.spanish = 3)
    (h3 : costs.texan = 4)
    (h4 : amounts.virginia = 10)
    (h5 : totalCost costs amounts / totalWeight amounts = 3.6) :
    0.4 * amounts.texan - 0.6 * amounts.spanish = 1 := by
  sorry


end NUMINAMATH_CALUDE_peanut_mixture_equation_l412_41273


namespace NUMINAMATH_CALUDE_new_vessel_capacity_l412_41261

/-- Given two vessels with different alcohol concentrations, prove the capacity of a new vessel that contains their combined contents plus water to achieve a specific concentration. -/
theorem new_vessel_capacity
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percent : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percent : ℝ)
  (total_liquid : ℝ)
  (new_concentration : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percent = 0.25)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percent = 0.40)
  (h5 : total_liquid = 8)
  (h6 : new_concentration = 0.29000000000000004) :
  (vessel1_capacity * vessel1_alcohol_percent + vessel2_capacity * vessel2_alcohol_percent) / new_concentration = 10 := by
  sorry

#eval (2 * 0.25 + 6 * 0.40) / 0.29000000000000004

end NUMINAMATH_CALUDE_new_vessel_capacity_l412_41261


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l412_41292

theorem necessary_not_sufficient_condition (x y : ℝ) (hx : x > 0) :
  (∀ y, x > |y| → x > y) ∧ (∃ y, x > y ∧ ¬(x > |y|)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l412_41292


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l412_41233

theorem average_of_six_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (subset : ℕ)
  (avg_subset : ℚ)
  (h_total : total = 10)
  (h_avg_all : avg_all = 80)
  (h_subset : subset = 4)
  (h_avg_subset : avg_subset = 113) :
  let remaining := total - subset
  let sum_all := total * avg_all
  let sum_subset := subset * avg_subset
  let sum_remaining := sum_all - sum_subset
  (sum_remaining : ℚ) / remaining = 58 := by sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l412_41233


namespace NUMINAMATH_CALUDE_martha_reading_challenge_l412_41260

def pages_read : List Nat := [12, 18, 14, 20, 11, 13, 19, 15, 17]
def total_days : Nat := 10
def target_average : Nat := 15

theorem martha_reading_challenge :
  ∃ (x : Nat), 
    (List.sum pages_read + x) / total_days = target_average ∧
    x = 11 := by
  sorry

end NUMINAMATH_CALUDE_martha_reading_challenge_l412_41260


namespace NUMINAMATH_CALUDE_guitar_sales_l412_41225

theorem guitar_sales (total_revenue : ℕ) (electric_price acoustic_price : ℕ) (electric_sold : ℕ) : 
  total_revenue = 3611 →
  electric_price = 479 →
  acoustic_price = 339 →
  electric_sold = 4 →
  ∃ (acoustic_sold : ℕ), electric_sold + acoustic_sold = 9 ∧ 
    electric_sold * electric_price + acoustic_sold * acoustic_price = total_revenue := by
  sorry

end NUMINAMATH_CALUDE_guitar_sales_l412_41225


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_greater_than_neg_three_fourths_l412_41288

theorem inequality_holds_iff_m_greater_than_neg_three_fourths (m : ℝ) :
  (∀ x : ℝ, m^2 * x^2 - 2*m*x > -x^2 - x - 1) ↔ m > -3/4 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_greater_than_neg_three_fourths_l412_41288


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l412_41267

theorem sum_of_roots_quadratic (x : ℝ) : x^2 = 16*x - 5 → ∃ y : ℝ, x^2 = 16*x - 5 ∧ x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l412_41267


namespace NUMINAMATH_CALUDE_translation_result_l412_41298

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, 5)
def B : ℝ × ℝ := (-3, 0)
def C : ℝ × ℝ := (3, 8)

-- Define the translation vector
def translation_vector : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define point D as the result of translating B
def D : ℝ × ℝ := (B.1 + translation_vector.1, B.2 + translation_vector.2)

-- Theorem statement
theorem translation_result : D = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l412_41298


namespace NUMINAMATH_CALUDE_translation_problem_l412_41265

/-- A translation in the complex plane -/
def ComplexTranslation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

theorem translation_problem (T : ℂ → ℂ) (h : T = ComplexTranslation (3 + 5*I)) :
  T (3 - I) = 6 + 4*I := by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l412_41265


namespace NUMINAMATH_CALUDE_digit_interchange_theorem_l412_41205

theorem digit_interchange_theorem (a b m : ℕ) (h1 : a > 0 ∧ a < 10) (h2 : b < 10) 
  (h3 : 10 * a + b = m * (a * b)) :
  10 * b + a = (11 - m) * (a * b) :=
sorry

end NUMINAMATH_CALUDE_digit_interchange_theorem_l412_41205


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l412_41291

theorem arithmetic_sequence_count (a₁ aₙ d : ℕ) (h1 : a₁ = 6) (h2 : aₙ = 91) (h3 : d = 5) :
  ∃ n : ℕ, n = 18 ∧ aₙ = a₁ + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l412_41291


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l412_41258

-- Problem 1
theorem problem_1 : -3^2 + (-1/2)^2 + (2023 - Real.pi)^0 - |-2| = -47/4 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (-2*a^2)^3 * a^2 + a^8 = -7*a^8 := by sorry

-- Problem 3
theorem problem_3 : 2023^2 - 2024 * 2022 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l412_41258


namespace NUMINAMATH_CALUDE_division_sum_theorem_l412_41215

theorem division_sum_theorem (dividend : ℕ) (divisor : ℕ) (h1 : dividend = 54) (h2 : divisor = 9) :
  dividend / divisor + dividend + divisor = 69 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l412_41215


namespace NUMINAMATH_CALUDE_first_two_nonzero_digits_of_one_over_137_l412_41290

theorem first_two_nonzero_digits_of_one_over_137 :
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ (1 : ℚ) / 137 = (a * 10 + b : ℕ) / 1000 + r ∧ 0 ≤ r ∧ r < 1 / 100 ∧ a = 7 ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_first_two_nonzero_digits_of_one_over_137_l412_41290


namespace NUMINAMATH_CALUDE_jerrys_age_l412_41266

/-- Given that Mickey's age is 20 years old and 10 years more than 200% of Jerry's age, prove that Jerry is 5 years old. -/
theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 20 → 
  mickey_age = 2 * jerry_age + 10 → 
  jerry_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_age_l412_41266


namespace NUMINAMATH_CALUDE_stephanies_remaining_payment_l412_41254

/-- The total amount Stephanie still needs to pay to finish her bills -/
def remaining_payment (electricity gas water internet : ℝ) 
                      (gas_paid_fraction : ℝ) 
                      (gas_additional_payment : ℝ) 
                      (water_paid_fraction : ℝ) 
                      (internet_payments : ℕ) 
                      (internet_payment_amount : ℝ) : ℝ :=
  (gas - gas_paid_fraction * gas - gas_additional_payment) +
  (water - water_paid_fraction * water) +
  (internet - internet_payments * internet_payment_amount)

/-- Stephanie's remaining bill payment theorem -/
theorem stephanies_remaining_payment :
  remaining_payment 60 40 40 25 (3/4) 5 (1/2) 4 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_stephanies_remaining_payment_l412_41254


namespace NUMINAMATH_CALUDE_subcommittee_count_l412_41257

theorem subcommittee_count (n m k : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 5) :
  (Nat.choose n k) - (Nat.choose (n - m) k) = 771 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l412_41257


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_is_negative_one_l412_41286

theorem reciprocal_equals_self_is_negative_one (x : ℝ) :
  x < 0 ∧ 1 / x = x → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_is_negative_one_l412_41286


namespace NUMINAMATH_CALUDE_total_pages_purchased_l412_41277

def total_budget : ℚ := 10
def cost_per_notepad : ℚ := 5/4
def pages_per_notepad : ℕ := 60

theorem total_pages_purchased :
  (total_budget / cost_per_notepad).floor * pages_per_notepad = 480 :=
by sorry

end NUMINAMATH_CALUDE_total_pages_purchased_l412_41277


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l412_41210

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1 / x₀ + 4 / y₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l412_41210


namespace NUMINAMATH_CALUDE_total_snowfall_calculation_l412_41294

theorem total_snowfall_calculation (monday tuesday wednesday : Real) 
  (h1 : monday = 0.327)
  (h2 : tuesday = 0.216)
  (h3 : wednesday = 0.184) :
  monday + tuesday + wednesday = 0.727 := by
  sorry

end NUMINAMATH_CALUDE_total_snowfall_calculation_l412_41294


namespace NUMINAMATH_CALUDE_translation_motions_l412_41216

/-- Represents a type of motion. -/
inductive Motion
  | Swing
  | VerticalElevator
  | PlanetMovement
  | ConveyorBelt

/-- Determines if a given motion is a translation. -/
def isTranslation (m : Motion) : Prop :=
  match m with
  | Motion.VerticalElevator => True
  | Motion.ConveyorBelt => True
  | _ => False

/-- The theorem stating which motions are translations. -/
theorem translation_motions :
  (∀ m : Motion, isTranslation m ↔ (m = Motion.VerticalElevator ∨ m = Motion.ConveyorBelt)) :=
by sorry

end NUMINAMATH_CALUDE_translation_motions_l412_41216


namespace NUMINAMATH_CALUDE_senate_subcommittee_count_l412_41250

theorem senate_subcommittee_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 8
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (Nat.choose total_republicans subcommittee_republicans) *
  (Nat.choose total_democrats subcommittee_democrats) = 11760 := by
  sorry

end NUMINAMATH_CALUDE_senate_subcommittee_count_l412_41250


namespace NUMINAMATH_CALUDE_intersection_point_first_quadrant_l412_41238

-- Define the quadratic and linear functions
def f (x : ℝ) : ℝ := x^2 - x - 5
def g (x : ℝ) : ℝ := 2*x - 1

-- Define the first quadrant
def first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

-- Theorem statement
theorem intersection_point_first_quadrant :
  ∃! p : ℝ × ℝ, first_quadrant p ∧ f p.1 = g p.1 ∧ f p.1 = p.2 ∧ p = (4, 7) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_first_quadrant_l412_41238


namespace NUMINAMATH_CALUDE_fib_70_mod_10_l412_41201

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Fibonacci sequence modulo 10 -/
def fibMod10 (n : ℕ) : ℕ := fib n % 10

/-- Period of Fibonacci sequence modulo 10 -/
def fibMod10Period : ℕ := 60

theorem fib_70_mod_10 :
  fibMod10 70 = 5 := by sorry

end NUMINAMATH_CALUDE_fib_70_mod_10_l412_41201


namespace NUMINAMATH_CALUDE_calculate_expression_l412_41218

theorem calculate_expression : (121^2 - 110^2 + 11) / 10 = 255.2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l412_41218


namespace NUMINAMATH_CALUDE_initial_piggy_bank_amount_l412_41227

-- Define the variables
def weekly_allowance : ℕ := 10
def weeks : ℕ := 8
def final_amount : ℕ := 83

-- Define the function to calculate the amount added to the piggy bank
def amount_added (w : ℕ) : ℕ := w * (weekly_allowance / 2)

-- Theorem statement
theorem initial_piggy_bank_amount :
  ∃ (initial : ℕ), initial + amount_added weeks = final_amount :=
sorry

end NUMINAMATH_CALUDE_initial_piggy_bank_amount_l412_41227


namespace NUMINAMATH_CALUDE_max_apple_recipients_l412_41293

theorem max_apple_recipients : ∃ n : ℕ, n = 13 ∧ 
  (∀ k : ℕ, k > n → k * (k + 1) > 200) ∧
  (n * (n + 1) ≤ 200) := by
  sorry

end NUMINAMATH_CALUDE_max_apple_recipients_l412_41293


namespace NUMINAMATH_CALUDE_floor_sqrt_10_l412_41289

theorem floor_sqrt_10 : ⌊Real.sqrt 10⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_10_l412_41289


namespace NUMINAMATH_CALUDE_probability_A_equals_B_l412_41206

open Set
open MeasureTheory
open Real

-- Define the set of valid pairs (a, b)
def ValidPairs : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (a, b) := p
               cos (cos a) = cos (cos b) ∧
               -5*π/2 ≤ a ∧ a ≤ 5*π/2 ∧
               -5*π/2 ≤ b ∧ b ≤ 5*π/2}

-- Define the set of pairs where A = B
def EqualPairs : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (a, b) := p; a = b}

-- Define the probability measure on ValidPairs
noncomputable def ProbMeasure : Measure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_A_equals_B :
  ProbMeasure (ValidPairs ∩ EqualPairs) / ProbMeasure ValidPairs = 1/5 :=
sorry

end NUMINAMATH_CALUDE_probability_A_equals_B_l412_41206


namespace NUMINAMATH_CALUDE_bread_slice_cost_l412_41208

-- Define the problem parameters
def num_loaves : ℕ := 3
def slices_per_loaf : ℕ := 20
def payment_amount : ℕ := 40  -- in dollars
def change_received : ℕ := 16  -- in dollars

-- Define the theorem
theorem bread_slice_cost :
  let total_cost : ℕ := payment_amount - change_received
  let total_slices : ℕ := num_loaves * slices_per_loaf
  let cost_per_slice_cents : ℕ := (total_cost * 100) / total_slices
  cost_per_slice_cents = 40 := by
  sorry

end NUMINAMATH_CALUDE_bread_slice_cost_l412_41208


namespace NUMINAMATH_CALUDE_cartesian_plane_problem_l412_41251

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, 0)

-- Define vectors
def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B

-- Define the length of OC
def OC_length : ℝ := 1

-- Theorem statement
theorem cartesian_plane_problem :
  -- Part 1: Angle between OA and OB is 45°
  let angle := Real.arccos ((OA.1 * OB.1 + OA.2 * OB.2) / (Real.sqrt (OA.1^2 + OA.2^2) * Real.sqrt (OB.1^2 + OB.2^2)))
  angle = Real.pi / 4 ∧
  -- Part 2: If OC ⊥ OA, then C has coordinates (±√2/2, ±√2/2)
  (∀ C : ℝ × ℝ, (C.1 * OA.1 + C.2 * OA.2 = 0 ∧ C.1^2 + C.2^2 = OC_length^2) →
    (C = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ∨ C = (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2))) ∧
  -- Part 3: Range of |OA + OB + OC|
  (∀ C : ℝ × ℝ, C.1^2 + C.2^2 = OC_length^2 →
    Real.sqrt 10 - 1 ≤ Real.sqrt ((OA.1 + OB.1 + C.1)^2 + (OA.2 + OB.2 + C.2)^2) ∧
    Real.sqrt ((OA.1 + OB.1 + C.1)^2 + (OA.2 + OB.2 + C.2)^2) ≤ Real.sqrt 10 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_cartesian_plane_problem_l412_41251


namespace NUMINAMATH_CALUDE_average_value_of_z_squared_l412_41211

theorem average_value_of_z_squared (z : ℝ) : 
  (z^2 + 3*z^2 + 6*z^2 + 12*z^2 + 24*z^2) / 5 = (46 * z^2) / 5 := by
  sorry

end NUMINAMATH_CALUDE_average_value_of_z_squared_l412_41211


namespace NUMINAMATH_CALUDE_amy_work_hours_school_year_l412_41221

/-- Calculates the number of hours per week Amy must work during the school year
    to meet her financial goal, given her summer work schedule and earnings,
    and her school year work duration and earnings goal. -/
theorem amy_work_hours_school_year 
  (summer_weeks : ℕ) 
  (summer_hours_per_week : ℕ) 
  (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) 
  (school_year_earnings_goal : ℕ) 
  (h1 : summer_weeks = 8)
  (h2 : summer_hours_per_week = 40)
  (h3 : summer_earnings = 3200)
  (h4 : school_year_weeks = 32)
  (h5 : school_year_earnings_goal = 4800) :
  (school_year_earnings_goal * summer_weeks * summer_hours_per_week) / 
  (summer_earnings * school_year_weeks) = 15 :=
by
  sorry

#check amy_work_hours_school_year

end NUMINAMATH_CALUDE_amy_work_hours_school_year_l412_41221


namespace NUMINAMATH_CALUDE_edith_books_count_edith_books_count_proof_l412_41232

theorem edith_books_count : ℕ → Prop :=
  fun total : ℕ =>
    ∃ (x y : ℕ),
      x = (120 * 56) / 100 ∧  -- 20% more than 56
      y = (x + 56) / 2 ∧      -- half of total novels
      total = x + 56 + y ∧    -- total books
      total = 185             -- correct answer

-- The proof goes here
theorem edith_books_count_proof : edith_books_count 185 := by
  sorry

end NUMINAMATH_CALUDE_edith_books_count_edith_books_count_proof_l412_41232


namespace NUMINAMATH_CALUDE_usual_price_equals_sale_price_l412_41268

/-- Represents the laundry detergent scenario -/
structure DetergentScenario where
  loads_per_bottle : ℕ
  sale_price_per_bottle : ℚ
  cost_per_load : ℚ

/-- The usual price of a bottle of detergent is equal to the sale price -/
theorem usual_price_equals_sale_price (scenario : DetergentScenario)
  (h1 : scenario.loads_per_bottle = 80)
  (h2 : scenario.sale_price_per_bottle = 20)
  (h3 : scenario.cost_per_load = 1/4) :
  scenario.sale_price_per_bottle = scenario.loads_per_bottle * scenario.cost_per_load := by
  sorry

#check usual_price_equals_sale_price

end NUMINAMATH_CALUDE_usual_price_equals_sale_price_l412_41268


namespace NUMINAMATH_CALUDE_perimeter_difference_is_one_l412_41234

-- Define the figures
def figure1_width : ℕ := 4
def figure1_height : ℕ := 2
def figure1_extra_square : ℕ := 1

def figure2_width : ℕ := 6
def figure2_height : ℕ := 2

-- Define the perimeter calculation functions
def perimeter_figure1 (w h e : ℕ) : ℕ :=
  2 * (w + h) + 3 * e

def perimeter_figure2 (w h : ℕ) : ℕ :=
  2 * (w + h)

-- Theorem statement
theorem perimeter_difference_is_one :
  Int.natAbs (perimeter_figure1 figure1_width figure1_height figure1_extra_square -
              perimeter_figure2 figure2_width figure2_height) = 1 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_is_one_l412_41234


namespace NUMINAMATH_CALUDE_new_boarders_count_l412_41282

theorem new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders : ℕ) (initial_ratio_day : ℕ) (final_ratio_boarders : ℕ) (final_ratio_day : ℕ) :
  initial_boarders = 220 →
  initial_ratio_boarders = 5 →
  initial_ratio_day = 12 →
  final_ratio_boarders = 1 →
  final_ratio_day = 2 →
  ∃ (new_boarders : ℕ),
    new_boarders = 44 ∧
    (initial_boarders + new_boarders) * final_ratio_day = initial_boarders * initial_ratio_day * final_ratio_boarders :=
by sorry


end NUMINAMATH_CALUDE_new_boarders_count_l412_41282


namespace NUMINAMATH_CALUDE_nth_equation_holds_l412_41256

theorem nth_equation_holds (n : ℕ) (hn : n > 0) : 
  (4 * n^2 : ℚ) / (2 * n - 1) - (2 * n + 1) = 1 - ((2 * n - 2) : ℚ) / (2 * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_holds_l412_41256


namespace NUMINAMATH_CALUDE_sign_white_area_l412_41281

/-- Represents the dimensions and areas of the letters in the sign --/
structure LetterAreas where
  m_area : ℝ
  a_area : ℝ
  t_area : ℝ
  h_area : ℝ

/-- Calculates the white area of the sign after drawing the letters "MATH" --/
def white_area (sign_width sign_height : ℝ) (letters : LetterAreas) : ℝ :=
  sign_width * sign_height - (letters.m_area + letters.a_area + letters.t_area + letters.h_area)

/-- Theorem stating that the white area of the sign is 42.5 square units --/
theorem sign_white_area :
  let sign_width := 20
  let sign_height := 4
  let letters := LetterAreas.mk 12 7.5 7 11
  white_area sign_width sign_height letters = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_sign_white_area_l412_41281


namespace NUMINAMATH_CALUDE_max_sum_at_9_l412_41264

/-- An arithmetic sequence with first term 1 and common difference d -/
def arithmetic_sequence (d : ℚ) : ℕ → ℚ := λ n => 1 + (n - 1 : ℚ) * d

/-- The sum of the first n terms of the arithmetic sequence -/
def S (d : ℚ) (n : ℕ) : ℚ := (n : ℚ) * (2 + (n - 1 : ℚ) * d) / 2

/-- The theorem stating that Sn reaches its maximum when n = 9 -/
theorem max_sum_at_9 (d : ℚ) (h : -2/17 < d ∧ d < -1/9) :
  ∀ k : ℕ, S d 9 ≥ S d k :=
sorry

end NUMINAMATH_CALUDE_max_sum_at_9_l412_41264


namespace NUMINAMATH_CALUDE_men_working_with_boys_l412_41285

-- Define the work done by one man per day
def work_man : ℚ := 1 / 48

-- Define the work done by one boy per day
def work_boy : ℚ := 5 / 96

-- Define the total work to be done
def total_work : ℚ := 1

theorem men_working_with_boys : ℕ :=
  let men_count : ℕ := 1
  have h1 : 2 * work_man + 4 * work_boy = total_work / 4 := by sorry
  have h2 : men_count * work_man + 6 * work_boy = total_work / 3 := by sorry
  have h3 : 2 * work_boy = 5 * work_man := by sorry
  men_count

end NUMINAMATH_CALUDE_men_working_with_boys_l412_41285


namespace NUMINAMATH_CALUDE_xyz_value_l412_41296

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 6 * y = -24)
  (eq2 : y * z + 6 * z = -24)
  (eq3 : z * x + 6 * x = -24) :
  x * y * z = 192 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l412_41296


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l412_41207

theorem negation_of_existence_inequality (p : Prop) :
  (¬ p ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0) ↔
  (p ↔ ∃ x₀ : ℝ, x₀^2 - x₀ + 1/4 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l412_41207


namespace NUMINAMATH_CALUDE_certain_number_proof_l412_41223

theorem certain_number_proof (x q : ℝ) 
  (h1 : 3 / x = 8)
  (h2 : 3 / q = 18)
  (h3 : x - q = 0.20833333333333334) :
  x = 0.375 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l412_41223


namespace NUMINAMATH_CALUDE_zacks_marbles_l412_41226

theorem zacks_marbles (M : ℕ) : 
  (∃ k : ℕ, M = 3 * k + 5) → 
  (M - 60 = 5) → 
  M = 65 := by
sorry

end NUMINAMATH_CALUDE_zacks_marbles_l412_41226


namespace NUMINAMATH_CALUDE_bond_face_value_l412_41249

/-- The face value of a bond -/
def face_value : ℝ := 5000

/-- The interest rate as a percentage of face value -/
def interest_rate : ℝ := 0.05

/-- The selling price of the bond -/
def selling_price : ℝ := 3846.153846153846

/-- The interest amount as a percentage of selling price -/
def interest_percentage : ℝ := 0.065

theorem bond_face_value :
  face_value = selling_price * interest_percentage / interest_rate :=
by sorry

end NUMINAMATH_CALUDE_bond_face_value_l412_41249


namespace NUMINAMATH_CALUDE_sphere_volume_and_radius_ratio_l412_41271

theorem sphere_volume_and_radius_ratio (V_large V_small : ℝ) (h1 : V_large = 432 * Real.pi) (h2 : V_small = 0.15 * V_large) : 
  ∃ (r_large r_small : ℝ), 
    (4 / 3 * Real.pi * r_large ^ 3 = V_large) ∧ 
    (4 / 3 * Real.pi * r_small ^ 3 = V_small) ∧
    (r_small / r_large = Real.rpow 1.8 (1/3) / 2) ∧
    (V_large + V_small = 496.8 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_and_radius_ratio_l412_41271


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l412_41287

/-- Represents a box of stationery -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents a person's stationery usage -/
structure Usage where
  sheetsPerLetter : ℕ
  unusedSheets : ℕ
  unusedEnvelopes : ℕ

theorem stationery_box_sheets (box : StationeryBox) (john mary : Usage) : box.sheets = 240 :=
  by
  have h1 : john.sheetsPerLetter = 2 := by sorry
  have h2 : mary.sheetsPerLetter = 4 := by sorry
  have h3 : john.unusedSheets = 40 := by sorry
  have h4 : mary.unusedEnvelopes = 40 := by sorry
  have h5 : box.sheets = john.sheetsPerLetter * box.envelopes + john.unusedSheets := by sorry
  have h6 : box.sheets = mary.sheetsPerLetter * (box.envelopes - mary.unusedEnvelopes) := by sorry
  sorry

end NUMINAMATH_CALUDE_stationery_box_sheets_l412_41287


namespace NUMINAMATH_CALUDE_smaller_pyramid_volume_l412_41202

/-- The volume of a smaller pyramid cut from a right rectangular pyramid -/
theorem smaller_pyramid_volume
  (base_length : ℝ) (base_width : ℝ) (slant_edge : ℝ) (cut_height : ℝ)
  (h_base_length : base_length = 10 * Real.sqrt 2)
  (h_base_width : base_width = 6 * Real.sqrt 2)
  (h_slant_edge : slant_edge = 12)
  (h_cut_height : cut_height = 4) :
  ∃ (volume : ℝ),
    volume = 20 * ((2 * Real.sqrt 19 - 4) / (2 * Real.sqrt 19))^3 * (2 * Real.sqrt 19 - 4) :=
by sorry

end NUMINAMATH_CALUDE_smaller_pyramid_volume_l412_41202


namespace NUMINAMATH_CALUDE_quadratic_a_value_main_quadratic_theorem_l412_41212

/-- A quadratic function with vertex form y = a(x - h)^2 + k, where (h, k) is the vertex -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The theorem stating the value of 'a' for a quadratic function with given properties -/
theorem quadratic_a_value (f : QuadraticFunction) 
  (vertex_condition : f.h = -3 ∧ f.k = 0)
  (point_condition : f.a * (2 - f.h)^2 + f.k = -36) :
  f.a = -36/25 := by
  sorry

/-- The main theorem proving the value of 'a' for the given quadratic function -/
theorem main_quadratic_theorem :
  ∃ f : QuadraticFunction, 
    f.h = -3 ∧ 
    f.k = 0 ∧ 
    f.a * (2 - f.h)^2 + f.k = -36 ∧
    f.a = -36/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_a_value_main_quadratic_theorem_l412_41212


namespace NUMINAMATH_CALUDE_detective_problem_l412_41228

theorem detective_problem (n : ℕ) (h : n = 80) :
  ∃ (S : Fin 12 → Set (Fin n)), ∀ (w c : Fin n), w ≠ c →
    ∃ (i : Fin 12), w ∈ S i ∧ c ∉ S i :=
sorry

end NUMINAMATH_CALUDE_detective_problem_l412_41228


namespace NUMINAMATH_CALUDE_samara_tire_expense_l412_41245

/-- Calculates Samara's spending on tires given the other expenses -/
def samaras_tire_spending (alberto_total : ℕ) (samara_oil : ℕ) (samara_detailing : ℕ) (difference : ℕ) : ℕ :=
  alberto_total - (samara_oil + samara_detailing + difference)

theorem samara_tire_expense :
  samaras_tire_spending 2457 25 79 1886 = 467 := by
  sorry

end NUMINAMATH_CALUDE_samara_tire_expense_l412_41245


namespace NUMINAMATH_CALUDE_calculate_b_investment_l412_41295

/-- Calculates B's investment in a partnership given the investments of A and C, 
    the total profit, and A's share of the profit. -/
theorem calculate_b_investment (a_investment c_investment total_profit a_profit : ℕ) : 
  a_investment = 6300 →
  c_investment = 10500 →
  total_profit = 14200 →
  a_profit = 4260 →
  ∃ b_investment : ℕ, 
    b_investment = 4220 ∧ 
    (a_investment : ℚ) / (a_investment + b_investment + c_investment) = 
    (a_profit : ℚ) / total_profit :=
by sorry

end NUMINAMATH_CALUDE_calculate_b_investment_l412_41295


namespace NUMINAMATH_CALUDE_ticket_price_uniqueness_l412_41230

theorem ticket_price_uniqueness : ∃! x : ℕ+, 
  (x : ℕ) ∣ 72 ∧ 
  (x : ℕ) ∣ 90 ∧ 
  1 ≤ 72 / (x : ℕ) ∧ 72 / (x : ℕ) ≤ 10 ∧
  1 ≤ 90 / (x : ℕ) ∧ 90 / (x : ℕ) ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_ticket_price_uniqueness_l412_41230


namespace NUMINAMATH_CALUDE_collinear_points_sum_l412_41235

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop := sorry

/-- The main theorem: If the given points are collinear, then a + b = 4. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (1, b, a) (b, 2, a) (b, a, 3) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l412_41235


namespace NUMINAMATH_CALUDE_days_to_pay_for_cash_register_l412_41299

/-- Represents the daily sales and costs for Marie's bakery --/
structure BakeryFinances where
  breadPrice : ℝ
  breadQuantity : ℝ
  bagelPrice : ℝ
  bagelQuantity : ℝ
  cakePrice : ℝ
  cakeQuantity : ℝ
  muffinPrice : ℝ
  muffinQuantity : ℝ
  rent : ℝ
  electricity : ℝ
  wages : ℝ
  ingredientCosts : ℝ
  salesTax : ℝ

/-- Calculates the number of days needed to pay for the cash register --/
def daysToPayForCashRegister (finances : BakeryFinances) (cashRegisterCost : ℝ) : ℕ :=
  sorry

/-- Theorem stating that it takes 17 days to pay for the cash register --/
theorem days_to_pay_for_cash_register :
  ∃ (finances : BakeryFinances),
    finances.breadPrice = 2 ∧
    finances.breadQuantity = 40 ∧
    finances.bagelPrice = 1.5 ∧
    finances.bagelQuantity = 20 ∧
    finances.cakePrice = 12 ∧
    finances.cakeQuantity = 6 ∧
    finances.muffinPrice = 3 ∧
    finances.muffinQuantity = 10 ∧
    finances.rent = 20 ∧
    finances.electricity = 2 ∧
    finances.wages = 80 ∧
    finances.ingredientCosts = 30 ∧
    finances.salesTax = 0.08 ∧
    daysToPayForCashRegister finances 1040 = 17 :=
  sorry

end NUMINAMATH_CALUDE_days_to_pay_for_cash_register_l412_41299


namespace NUMINAMATH_CALUDE_power_of_power_three_l412_41241

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l412_41241


namespace NUMINAMATH_CALUDE_min_area_line_correct_l412_41222

/-- A line passing through a point (2, 1) and intersecting the positive x and y axes -/
structure MinAreaLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (2, 1) -/
  passes_through : m * 2 + b = 1
  /-- The line intersects the positive x-axis -/
  x_intercept_positive : -b / m > 0
  /-- The line intersects the positive y-axis -/
  y_intercept_positive : b > 0

/-- The equation of the line that minimizes the area of the triangle formed with the axes -/
def min_area_line_equation (l : MinAreaLine) : Prop :=
  l.m = -1/2 ∧ l.b = 2

theorem min_area_line_correct (l : MinAreaLine) :
  min_area_line_equation l ↔ l.m * 1 + l.b * 2 = 4 :=
sorry

end NUMINAMATH_CALUDE_min_area_line_correct_l412_41222


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l412_41283

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 267)
  (sum_of_products : a*b + b*c + c*a = 131) :
  a + b + c = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l412_41283


namespace NUMINAMATH_CALUDE_correct_loan_amounts_l412_41237

/-- Represents the loan amounts and interest rates for a company's two types of loans. -/
structure LoanInfo where
  typeA : ℝ  -- Amount of Type A loan in yuan
  typeB : ℝ  -- Amount of Type B loan in yuan
  rateA : ℝ  -- Annual interest rate for Type A loan
  rateB : ℝ  -- Annual interest rate for Type B loan

/-- Theorem stating the correct loan amounts given the problem conditions. -/
theorem correct_loan_amounts (loan : LoanInfo) : 
  loan.typeA = 200000 ∧ loan.typeB = 300000 ↔ 
  loan.typeA + loan.typeB = 500000 ∧ 
  loan.rateA * loan.typeA + loan.rateB * loan.typeB = 44000 ∧
  loan.rateA = 0.1 ∧ 
  loan.rateB = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_correct_loan_amounts_l412_41237


namespace NUMINAMATH_CALUDE_max_donuts_is_17_seventeen_donuts_possible_l412_41259

-- Define the prices and budget
def single_price : ℕ := 1
def pack4_price : ℕ := 3
def pack8_price : ℕ := 5
def budget : ℕ := 11

-- Define a function to calculate the number of donuts for a given combination
def donut_count (singles pack4 pack8 : ℕ) : ℕ :=
  singles + 4 * pack4 + 8 * pack8

-- Define a function to calculate the total cost for a given combination
def total_cost (singles pack4 pack8 : ℕ) : ℕ :=
  singles * single_price + pack4 * pack4_price + pack8 * pack8_price

-- Theorem stating that 17 is the maximum number of donuts that can be purchased
theorem max_donuts_is_17 :
  ∀ (singles pack4 pack8 : ℕ),
    total_cost singles pack4 pack8 ≤ budget →
    donut_count singles pack4 pack8 ≤ 17 :=
by
  sorry

-- Theorem stating that 17 donuts can actually be purchased
theorem seventeen_donuts_possible :
  ∃ (singles pack4 pack8 : ℕ),
    total_cost singles pack4 pack8 ≤ budget ∧
    donut_count singles pack4 pack8 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_max_donuts_is_17_seventeen_donuts_possible_l412_41259


namespace NUMINAMATH_CALUDE_water_bottle_count_l412_41203

theorem water_bottle_count (initial bottles_drunk bottles_bought : ℕ) 
  (h1 : initial = 42)
  (h2 : bottles_drunk = 25)
  (h3 : bottles_bought = 30) : 
  initial - bottles_drunk + bottles_bought = 47 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_count_l412_41203


namespace NUMINAMATH_CALUDE_cylinder_height_from_balls_l412_41279

/-- The height of a cylinder formed by melting steel balls -/
theorem cylinder_height_from_balls (num_balls : ℕ) (ball_radius cylinder_radius : ℝ) :
  num_balls = 12 →
  ball_radius = 2 →
  cylinder_radius = 3 →
  (4 / 3 * π * num_balls * ball_radius ^ 3) / (π * cylinder_radius ^ 2) = 128 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_from_balls_l412_41279


namespace NUMINAMATH_CALUDE_largest_unachievable_sum_l412_41242

theorem largest_unachievable_sum (a : ℕ) (ha : Odd a) (ha_pos : 0 < a) :
  let n := (a^2 + 5*a + 4) / 2
  (∀ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z → a*x + (a+1)*y + (a+2)*z ≠ n) ∧
  (∀ m : ℕ, n < m → ∃ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ a*x + (a+1)*y + (a+2)*z = m) :=
by sorry

end NUMINAMATH_CALUDE_largest_unachievable_sum_l412_41242


namespace NUMINAMATH_CALUDE_total_quantities_l412_41262

theorem total_quantities (total_avg : ℝ) (subset1_count : ℕ) (subset1_avg : ℝ) (subset2_count : ℕ) (subset2_avg : ℝ) :
  total_avg = 6 →
  subset1_count = 3 →
  subset1_avg = 4 →
  subset2_count = 2 →
  subset2_avg = 33 →
  ∃ (n : ℕ), n = subset1_count + subset2_count ∧ n = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_total_quantities_l412_41262


namespace NUMINAMATH_CALUDE_ice_cream_jog_speed_l412_41204

/-- Calculates the required speed in miles per hour to cover a given distance within a time limit -/
def required_speed (time_limit : ℚ) (distance_blocks : ℕ) (block_length : ℚ) : ℚ :=
  (distance_blocks : ℚ) * block_length * (60 / time_limit)

theorem ice_cream_jog_speed :
  let time_limit : ℚ := 10  -- Time limit in minutes
  let distance_blocks : ℕ := 16  -- Distance in blocks
  let block_length : ℚ := 1/8  -- Length of each block in miles
  required_speed time_limit distance_blocks block_length = 12 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_jog_speed_l412_41204


namespace NUMINAMATH_CALUDE_image_of_A_under_f_l412_41247

def A : Set Int := {-1, 3, 5}

def f (x : Int) : Int := 2 * x - 1

theorem image_of_A_under_f :
  (Set.image f A) = {-3, 5, 9} := by
  sorry

end NUMINAMATH_CALUDE_image_of_A_under_f_l412_41247


namespace NUMINAMATH_CALUDE_no_infinite_line_family_l412_41269

theorem no_infinite_line_family :
  ¬ ∃ (k : ℕ → ℝ), 
    (∀ n, k n ≠ 0) ∧ 
    (∀ n, k (n + 1) = (1 - 1 / k n) - (1 - k n)) ∧
    (∀ n, k n * k (n + 1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_line_family_l412_41269


namespace NUMINAMATH_CALUDE_product_of_roots_l412_41239

theorem product_of_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 12*x^2 + 48*x + 28 = (x - r₁) * (x - r₂) * (x - r₃)) →
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 12*x^2 + 48*x + 28 = (x - r₁) * (x - r₂) * (x - r₃) ∧ r₁ * r₂ * r₃ = -28) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l412_41239


namespace NUMINAMATH_CALUDE_snack_machine_purchase_l412_41219

/-- The number of pieces of chocolate bought -/
def chocolate_pieces : ℕ := 2

/-- The cost of a candy bar in cents -/
def candy_bar_cost : ℕ := 25

/-- The cost of a piece of chocolate in cents -/
def chocolate_cost : ℕ := 75

/-- The cost of a pack of juice in cents -/
def juice_cost : ℕ := 50

/-- The total number of quarters used -/
def total_quarters : ℕ := 11

theorem snack_machine_purchase :
  chocolate_pieces * chocolate_cost + 3 * candy_bar_cost + juice_cost = total_quarters * 25 :=
by sorry

end NUMINAMATH_CALUDE_snack_machine_purchase_l412_41219


namespace NUMINAMATH_CALUDE_system_of_inequalities_l412_41284

theorem system_of_inequalities (x : ℝ) :
  (2 * x + 1 < 3) ∧ (x / 2 + (1 - 3 * x) / 4 ≤ 1) → -3 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l412_41284


namespace NUMINAMATH_CALUDE_bus_ride_is_75_minutes_l412_41200

/-- Calculates the bus ride duration given the total trip time, train ride duration, and walking time. -/
def bus_ride_duration (total_trip_time : ℕ) (train_ride_duration : ℕ) (walking_time : ℕ) : ℕ :=
  let total_minutes := total_trip_time * 60
  let train_minutes := train_ride_duration * 60
  let waiting_time := walking_time * 2
  total_minutes - train_minutes - waiting_time - walking_time

/-- Proves that given the specified conditions, the bus ride duration is 75 minutes. -/
theorem bus_ride_is_75_minutes :
  bus_ride_duration 8 6 15 = 75 := by
  sorry

#eval bus_ride_duration 8 6 15

end NUMINAMATH_CALUDE_bus_ride_is_75_minutes_l412_41200


namespace NUMINAMATH_CALUDE_first_four_digits_1973_l412_41252

theorem first_four_digits_1973 (n : ℕ) (h : ∀ k : ℕ, n ≠ 10^k) :
  ∃ j k : ℕ, j > 0 ∧ k > 0 ∧ 1973 ≤ (n^j : ℝ) / (10^k : ℝ) ∧ (n^j : ℝ) / (10^k : ℝ) < 1974 :=
sorry

end NUMINAMATH_CALUDE_first_four_digits_1973_l412_41252


namespace NUMINAMATH_CALUDE_total_pens_after_changes_l412_41229

-- Define the initial number of pens
def initial_red : ℕ := 65
def initial_blue : ℕ := 45
def initial_black : ℕ := 58
def initial_green : ℕ := 36
def initial_purple : ℕ := 27

-- Define the changes in pen quantities
def red_decrease : ℕ := 15
def blue_decrease : ℕ := 20
def black_increase : ℕ := 12
def green_decrease : ℕ := 10
def purple_increase : ℕ := 5

-- Define the theorem
theorem total_pens_after_changes : 
  (initial_red - red_decrease) + 
  (initial_blue - blue_decrease) + 
  (initial_black + black_increase) + 
  (initial_green - green_decrease) + 
  (initial_purple + purple_increase) = 203 := by
  sorry

end NUMINAMATH_CALUDE_total_pens_after_changes_l412_41229


namespace NUMINAMATH_CALUDE_maci_pen_cost_l412_41217

/-- The total cost of pens for Maci --/
def total_cost (blue_pens red_pens : ℕ) (blue_cost : ℚ) : ℚ :=
  (blue_pens : ℚ) * blue_cost + (red_pens : ℚ) * (2 * blue_cost)

/-- Theorem stating that Maci's total cost for pens is $4.00 --/
theorem maci_pen_cost :
  total_cost 10 15 (1/10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_maci_pen_cost_l412_41217


namespace NUMINAMATH_CALUDE_bd_length_is_15_l412_41243

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a kite
def is_kite (q : Quadrilateral) : Prop :=
  let AB := dist q.A q.B
  let BC := dist q.B q.C
  let CD := dist q.C q.D
  let DA := dist q.D q.A
  AB = CD ∧ BC = DA

-- Define the specific quadrilateral from the problem
def problem_quadrilateral : Quadrilateral :=
  { A := (0, 0),  -- Arbitrary placement
    B := (7, 0),  -- AB = 7
    C := (7, 19), -- BC = 19
    D := (0, 11)  -- DA = 11
  }

-- Theorem statement
theorem bd_length_is_15 (q : Quadrilateral) :
  is_kite q →
  dist q.A q.B = 7 →
  dist q.B q.C = 19 →
  dist q.C q.D = 7 →
  dist q.D q.A = 11 →
  dist q.B q.D = 15 :=
by sorry

#check bd_length_is_15

end NUMINAMATH_CALUDE_bd_length_is_15_l412_41243


namespace NUMINAMATH_CALUDE_a2_value_l412_41244

theorem a2_value (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₂ = 42 := by
sorry

end NUMINAMATH_CALUDE_a2_value_l412_41244


namespace NUMINAMATH_CALUDE_completing_square_l412_41231

theorem completing_square (x : ℝ) : x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l412_41231
