import Mathlib

namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1021_102123

theorem complex_fraction_simplification :
  1 + 3 / (2 + 5/6) = 35/17 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1021_102123


namespace NUMINAMATH_CALUDE_product_digit_count_l1021_102149

theorem product_digit_count (k n : ℕ) (a b : ℕ) :
  (10^(k-1) ≤ a ∧ a < 10^k) →
  (10^(n-1) ≤ b ∧ b < 10^n) →
  (10^(k+n-1) ≤ a * b ∧ a * b < 10^(k+n+1)) :=
sorry

end NUMINAMATH_CALUDE_product_digit_count_l1021_102149


namespace NUMINAMATH_CALUDE_set_inclusion_condition_l1021_102102

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2 * x + 3}
def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = x ^ 2}

-- State the theorem
theorem set_inclusion_condition (a : ℝ) :
  C a ⊆ B a ↔ (1/2 ≤ a ∧ a ≤ 2) ∨ (a ≥ 3) ∨ (a < -2) :=
sorry

end NUMINAMATH_CALUDE_set_inclusion_condition_l1021_102102


namespace NUMINAMATH_CALUDE_problem_solution_l1021_102191

theorem problem_solution (x y : ℝ) (hx : x = 1 - Real.sqrt 2) (hy : y = 1 + Real.sqrt 2) : 
  x^2 + 3*x*y + y^2 = 3 ∧ y/x - x/y = -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1021_102191


namespace NUMINAMATH_CALUDE_school_height_ratio_l1021_102155

theorem school_height_ratio (total_avg : ℝ) (female_avg : ℝ) (male_avg : ℝ)
  (h_total : total_avg = 180)
  (h_female : female_avg = 170)
  (h_male : male_avg = 182) :
  ∃ (m w : ℝ), m > 0 ∧ w > 0 ∧ m / w = 5 ∧
    male_avg * m + female_avg * w = total_avg * (m + w) :=
by
  sorry

end NUMINAMATH_CALUDE_school_height_ratio_l1021_102155


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1021_102105

theorem necessary_not_sufficient_condition :
  (∀ a b c d : ℝ, a + b < c + d → (a < c ∨ b < d)) ∧
  (∃ a b c d : ℝ, (a < c ∨ b < d) ∧ ¬(a + b < c + d)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1021_102105


namespace NUMINAMATH_CALUDE_sara_quarters_proof_l1021_102126

/-- The number of quarters Sara's dad gave her -/
def quarters_from_dad (initial_quarters final_quarters : ℕ) : ℕ :=
  final_quarters - initial_quarters

/-- Proof that Sara's dad gave her 49 quarters -/
theorem sara_quarters_proof (initial_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 21)
  (h2 : final_quarters = 70) :
  quarters_from_dad initial_quarters final_quarters = 49 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_proof_l1021_102126


namespace NUMINAMATH_CALUDE_juanita_sunscreen_usage_l1021_102116

/-- Proves that Juanita uses 1 bottle of sunscreen per month -/
theorem juanita_sunscreen_usage
  (months_per_year : ℕ)
  (discount_rate : ℚ)
  (bottle_cost : ℚ)
  (total_discounted_cost : ℚ)
  (h1 : months_per_year = 12)
  (h2 : discount_rate = 30 / 100)
  (h3 : bottle_cost = 30)
  (h4 : total_discounted_cost = 252) :
  (total_discounted_cost / ((1 - discount_rate) * bottle_cost)) / months_per_year = 1 :=
sorry

end NUMINAMATH_CALUDE_juanita_sunscreen_usage_l1021_102116


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l1021_102121

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (vote_difference : ℕ) 
  (candidate_percentage : ℚ) :
  total_votes = 7900 →
  vote_difference = 2370 →
  candidate_percentage = total_votes.cast / 100 * 
    ((total_votes.cast - vote_difference.cast) / (2 * total_votes.cast)) →
  candidate_percentage = 35 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l1021_102121


namespace NUMINAMATH_CALUDE_expression_simplification_l1021_102187

theorem expression_simplification (y : ℝ) :
  2 * y * (4 * y^2 - 3 * y + 1) - 6 * (y^2 - 3 * y + 4) =
  8 * y^3 - 12 * y^2 + 20 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1021_102187


namespace NUMINAMATH_CALUDE_football_field_area_is_9600_l1021_102181

/-- The total area of a football field in square yards -/
def football_field_area : ℝ := 9600

/-- The total amount of fertilizer used on the entire field in pounds -/
def total_fertilizer : ℝ := 1200

/-- The amount of fertilizer used on a part of the field in pounds -/
def partial_fertilizer : ℝ := 700

/-- The area covered by the partial fertilizer in square yards -/
def partial_area : ℝ := 5600

/-- Theorem stating that the football field area is 9600 square yards -/
theorem football_field_area_is_9600 : 
  football_field_area = (total_fertilizer * partial_area) / partial_fertilizer := by
  sorry

end NUMINAMATH_CALUDE_football_field_area_is_9600_l1021_102181


namespace NUMINAMATH_CALUDE_sum_is_linear_l1021_102171

/-- The original parabola function -/
def original_parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

/-- The function f(x) derived from the original parabola -/
def f (a h k x : ℝ) : ℝ := -a * (x - h - 3)^2 - k

/-- The function g(x) derived from the original parabola -/
def g (a h k x : ℝ) : ℝ := a * (x - h + 7)^2 + k

/-- The sum of f(x) and g(x) -/
def f_plus_g (a h k x : ℝ) : ℝ := f a h k x + g a h k x

theorem sum_is_linear (a h k : ℝ) (ha : a ≠ 0) :
  ∃ m b : ℝ, (∀ x : ℝ, f_plus_g a h k x = m * x + b) ∧ m ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_sum_is_linear_l1021_102171


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l1021_102166

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l1021_102166


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1021_102189

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width height : ℝ) : ℝ :=
  length * width + 2 * length * height + 2 * width * height

/-- Theorem: The wet surface area of a cistern with given dimensions is 49 square meters -/
theorem cistern_wet_surface_area :
  wetSurfaceArea 6 4 1.25 = 49 := by sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1021_102189


namespace NUMINAMATH_CALUDE_range_of_a_l1021_102109

open Set

theorem range_of_a (p : ∀ x ∈ Icc 1 2, x^2 - a ≥ 0) 
                   (q : ∃ x₀ : ℝ, x₀ + 2*a*x₀ + 2 - a = 0) : 
  a ≤ -2 ∨ a = 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1021_102109


namespace NUMINAMATH_CALUDE_equation_solution_l1021_102167

theorem equation_solution : 
  {x : ℝ | x^2 + (x-1)*(x+3) = 3*x + 5} = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1021_102167


namespace NUMINAMATH_CALUDE_basketball_team_allocation_schemes_l1021_102122

theorem basketball_team_allocation_schemes (n : ℕ) (k : ℕ) (m : ℕ) 
  (h1 : n = 8)  -- number of classes
  (h2 : k = 10) -- total number of players
  (h3 : m = k - n) -- remaining spots after each class contributes one player
  : (n.choose 2) + (n.choose 1) = 36 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_allocation_schemes_l1021_102122


namespace NUMINAMATH_CALUDE_subtract_negative_l1021_102141

theorem subtract_negative : -2 - (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l1021_102141


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l1021_102195

theorem sum_of_three_consecutive_integers : ∃ (n : ℤ),
  (n - 1) + n + (n + 1) = 21 ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 17) ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 11) ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 25) ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l1021_102195


namespace NUMINAMATH_CALUDE_group_size_proof_l1021_102118

theorem group_size_proof (total_spent : ℕ) (mango_price : ℕ) (pineapple_price : ℕ) (pineapple_spent : ℕ) :
  total_spent = 94 →
  mango_price = 5 →
  pineapple_price = 6 →
  pineapple_spent = 54 →
  ∃ (mango_count pineapple_count : ℕ),
    mango_count * mango_price + pineapple_count * pineapple_price = total_spent ∧
    pineapple_count * pineapple_price = pineapple_spent ∧
    mango_count + pineapple_count = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l1021_102118


namespace NUMINAMATH_CALUDE_basil_planter_problem_l1021_102108

theorem basil_planter_problem (total_seeds : ℕ) (num_large_planters : ℕ) (large_planter_capacity : ℕ) (small_planter_capacity : ℕ) 
  (h1 : total_seeds = 200)
  (h2 : num_large_planters = 4)
  (h3 : large_planter_capacity = 20)
  (h4 : small_planter_capacity = 4) :
  (total_seeds - num_large_planters * large_planter_capacity) / small_planter_capacity = 30 := by
  sorry

end NUMINAMATH_CALUDE_basil_planter_problem_l1021_102108


namespace NUMINAMATH_CALUDE_weight_difference_is_19_l1021_102136

/-- The combined weight difference between the lightest and heaviest individual -/
def weightDifference (john roy derek samantha : ℕ) : ℕ :=
  max john (max roy (max derek samantha)) - min john (min roy (min derek samantha))

/-- Theorem: The combined weight difference between the lightest and heaviest individual is 19 pounds -/
theorem weight_difference_is_19 :
  weightDifference 81 79 91 72 = 19 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_is_19_l1021_102136


namespace NUMINAMATH_CALUDE_expression_simplification_l1021_102182

theorem expression_simplification (x y : ℚ) (hx : x = 3) (hy : y = -1/3) :
  3 * x * y^2 - (x * y - 2 * (2 * x * y - 3/2 * x^2 * y) + 3 * x * y^2) + 3 * x^2 * y = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1021_102182


namespace NUMINAMATH_CALUDE_andrew_total_work_hours_l1021_102129

/-- The total hours Andrew worked on his Science report over three days -/
def total_hours (day1 day2 day3 : Real) : Real :=
  day1 + day2 + day3

/-- Theorem stating that Andrew worked 9.25 hours in total -/
theorem andrew_total_work_hours :
  let day1 : Real := 2.5
  let day2 : Real := day1 + 0.5
  let day3 : Real := 3.75
  total_hours day1 day2 day3 = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_andrew_total_work_hours_l1021_102129


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1021_102100

theorem sum_of_roots_quadratic (m n : ℝ) : 
  (m ^ 2 - 3 * m - 1 = 0) → (n ^ 2 - 3 * n - 1 = 0) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1021_102100


namespace NUMINAMATH_CALUDE_gain_amount_proof_l1021_102131

/-- Given an article sold at $180 with a 20% gain, prove that the gain amount is $30. -/
theorem gain_amount_proof (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 180)
  (h2 : gain_percentage = 0.20) : 
  let cost_price := selling_price / (1 + gain_percentage)
  selling_price - cost_price = 30 := by
sorry


end NUMINAMATH_CALUDE_gain_amount_proof_l1021_102131


namespace NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l1021_102111

theorem matrix_inverse_scalar_multiple (d k : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 4; 6, d]
  (A⁻¹ = k • A) → d = -1 ∧ k = (1 : ℝ) / 25 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l1021_102111


namespace NUMINAMATH_CALUDE_mark_deck_project_cost_l1021_102104

/-- Calculates the total cost of a multi-layered deck project --/
def deck_project_cost (length width : ℝ) 
                      (material_a_cost material_b_cost material_c_cost : ℝ) 
                      (beam_cost sealant_cost : ℝ) 
                      (railing_cost_30 railing_cost_40 : ℝ) 
                      (tax_rate : ℝ) : ℝ :=
  let area := length * width
  let material_cost := area * (material_a_cost + material_b_cost + material_c_cost)
  let beam_cost_total := area * beam_cost * 2
  let sealant_cost_total := area * sealant_cost
  let railing_cost_total := 2 * (railing_cost_30 + railing_cost_40)
  let subtotal := material_cost + beam_cost_total + sealant_cost_total + railing_cost_total
  let tax := subtotal * tax_rate
  subtotal + tax

/-- The total cost of Mark's deck project is $25423.20 --/
theorem mark_deck_project_cost : 
  deck_project_cost 30 40 3 5 8 2 1 120 160 0.07 = 25423.20 := by
  sorry

end NUMINAMATH_CALUDE_mark_deck_project_cost_l1021_102104


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1021_102115

theorem unique_triple_solution (a b c : ℝ) : 
  a > 5 → b > 5 → c > 5 →
  (a + 3)^2 / (b + c - 3) + (b + 6)^2 / (c + a - 6) + (c + 9)^2 / (a + b - 9) = 81 →
  a = 15 ∧ b = 12 ∧ c = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1021_102115


namespace NUMINAMATH_CALUDE_product_equality_l1021_102199

theorem product_equality (square : ℕ) : 
  10 * 20 * 30 * 40 * 50 = 100 * 2 * 300 * 4 * square → square = 50 := by
sorry

end NUMINAMATH_CALUDE_product_equality_l1021_102199


namespace NUMINAMATH_CALUDE_hcf_problem_l1021_102160

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2460) (h2 : Nat.lcm a b = 205) :
  Nat.gcd a b = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l1021_102160


namespace NUMINAMATH_CALUDE_certain_number_proof_l1021_102165

theorem certain_number_proof : 
  ∃ x : ℝ, (15 / 100) * x = (2.5 / 100) * 450 ∧ x = 75 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1021_102165


namespace NUMINAMATH_CALUDE_magnified_diameter_is_0_3_l1021_102158

/-- The magnification factor of the electron microscope -/
def magnification : ℝ := 1000

/-- The actual diameter of the tissue in centimeters -/
def actual_diameter : ℝ := 0.0003

/-- The diameter of the magnified image in centimeters -/
def magnified_diameter : ℝ := actual_diameter * magnification

/-- Theorem stating that the magnified diameter is 0.3 centimeters -/
theorem magnified_diameter_is_0_3 : magnified_diameter = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_magnified_diameter_is_0_3_l1021_102158


namespace NUMINAMATH_CALUDE_total_wheels_count_l1021_102176

def bicycle_count : Nat := 3
def tricycle_count : Nat := 4
def unicycle_count : Nat := 7

def bicycle_wheels : Nat := 2
def tricycle_wheels : Nat := 3
def unicycle_wheels : Nat := 1

theorem total_wheels_count : 
  bicycle_count * bicycle_wheels + 
  tricycle_count * tricycle_wheels + 
  unicycle_count * unicycle_wheels = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_count_l1021_102176


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l1021_102185

def binary_to_nat (b : List Bool) : Nat :=
  b.foldr (λ x acc => 2 * acc + if x then 1 else 0) 0

theorem binary_addition_subtraction :
  let a := [true, true, false, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, false, false, true] -- 1001₂
  let d := [true, false, true, false] -- 1010₂
  let result := [true, false, true, true, true] -- 10111₂
  binary_to_nat a + binary_to_nat b - binary_to_nat c + binary_to_nat d = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l1021_102185


namespace NUMINAMATH_CALUDE_point_inside_circle_range_l1021_102186

/-- Given that the point (1, 1) is inside the circle (x-a)^2+(y+a)^2=4, 
    prove that the range of a is -1 < a < 1 -/
theorem point_inside_circle_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_range_l1021_102186


namespace NUMINAMATH_CALUDE_base_10_to_base_6_l1021_102106

theorem base_10_to_base_6 : 
  (1 * 6^4 + 3 * 6^3 + 0 * 6^2 + 5 * 6^1 + 4 * 6^0 : ℕ) = 1978 := by
  sorry

#eval 1 * 6^4 + 3 * 6^3 + 0 * 6^2 + 5 * 6^1 + 4 * 6^0

end NUMINAMATH_CALUDE_base_10_to_base_6_l1021_102106


namespace NUMINAMATH_CALUDE_ending_number_is_300_l1021_102156

theorem ending_number_is_300 (ending_number : ℕ) : 
  (∃ (multiples : List ℕ), 
    multiples.length = 67 ∧ 
    (∀ n ∈ multiples, n % 3 = 0) ∧
    (∀ n ∈ multiples, 100 ≤ n ∧ n ≤ ending_number) ∧
    (∀ n, 100 ≤ n ∧ n ≤ ending_number ∧ n % 3 = 0 → n ∈ multiples)) →
  ending_number = 300 := by
sorry

end NUMINAMATH_CALUDE_ending_number_is_300_l1021_102156


namespace NUMINAMATH_CALUDE_angle_difference_l1021_102140

theorem angle_difference (a β : ℝ) 
  (h1 : 3 * Real.sin a - Real.cos a = 0)
  (h2 : 7 * Real.sin β + Real.cos β = 0)
  (h3 : 0 < a) (h4 : a < Real.pi / 2)
  (h5 : Real.pi / 2 < β) (h6 : β < Real.pi) :
  2 * a - β = - 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_difference_l1021_102140


namespace NUMINAMATH_CALUDE_revenue_change_l1021_102161

/-- Given a projected revenue increase and the ratio of actual to projected revenue,
    calculate the actual percent change in revenue. -/
theorem revenue_change
  (projected_increase : ℝ)
  (actual_to_projected_ratio : ℝ)
  (h1 : projected_increase = 0.20)
  (h2 : actual_to_projected_ratio = 0.75) :
  (1 + projected_increase) * actual_to_projected_ratio - 1 = -0.10 := by
  sorry

#check revenue_change

end NUMINAMATH_CALUDE_revenue_change_l1021_102161


namespace NUMINAMATH_CALUDE_proposition_A_sufficient_not_necessary_l1021_102127

/-- Defines a geometric sequence of three real numbers -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b ^ 2 = a * c) ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

theorem proposition_A_sufficient_not_necessary :
  (∀ a b c : ℝ, b ^ 2 ≠ a * c → ¬ is_geometric_sequence a b c) ∧
  (∃ a b c : ℝ, ¬ is_geometric_sequence a b c ∧ b ^ 2 = a * c) :=
by sorry

end NUMINAMATH_CALUDE_proposition_A_sufficient_not_necessary_l1021_102127


namespace NUMINAMATH_CALUDE_product_equality_l1021_102197

theorem product_equality : ∃ x : ℝ, 469138 * x = 4690910862 ∧ x = 10000.1 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1021_102197


namespace NUMINAMATH_CALUDE_equation_solution_difference_l1021_102178

theorem equation_solution_difference : ∃ x₁ x₂ : ℝ,
  (x₁ + 3)^2 / (3 * x₁ + 29) = 2 ∧
  (x₂ + 3)^2 / (3 * x₂ + 29) = 2 ∧
  x₁ ≠ x₂ ∧
  x₂ - x₁ = 14 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l1021_102178


namespace NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l1021_102179

-- Part 1: Equation solution
theorem equation_solution :
  ∀ x : ℝ, x * (x - 4) = x - 6 ↔ x = 2 ∨ x = 3 := by sorry

-- Part 2: System of inequalities solution
theorem inequalities_solution :
  ∀ x : ℝ, (4 * x - 2 ≥ 3 * (x - 1) ∧ (x - 5) / 2 + 1 > x - 3) ↔ -1 ≤ x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l1021_102179


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1021_102135

theorem exponent_multiplication (a : ℝ) : a^6 * a^2 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1021_102135


namespace NUMINAMATH_CALUDE_expression_evaluation_l1021_102117

theorem expression_evaluation :
  100 + (120 / 15) + (18 * 20) - 250 - (360 / 12) = 188 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1021_102117


namespace NUMINAMATH_CALUDE_inequality_implies_bound_l1021_102152

theorem inequality_implies_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 3, x^2 - a*x + 4 ≥ 0) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_bound_l1021_102152


namespace NUMINAMATH_CALUDE_large_font_pages_l1021_102110

/-- Represents the number of words per page for large font -/
def large_font_words_per_page : ℕ := 1800

/-- Represents the number of words per page for small font -/
def small_font_words_per_page : ℕ := 2400

/-- Represents the total number of pages allowed -/
def total_pages : ℕ := 21

/-- Represents the ratio of large font pages to small font pages -/
def font_ratio : Rat := 2 / 3

theorem large_font_pages : ℕ :=
  let large_pages : ℕ := 8
  let small_pages : ℕ := total_pages - large_pages
  have h1 : large_pages + small_pages = total_pages := by sorry
  have h2 : (large_pages : Rat) / (small_pages : Rat) = font_ratio := by sorry
  have h3 : large_pages * large_font_words_per_page + small_pages * small_font_words_per_page ≤ 48000 := by sorry
  large_pages

end NUMINAMATH_CALUDE_large_font_pages_l1021_102110


namespace NUMINAMATH_CALUDE_incorrect_statement_about_converses_l1021_102168

/-- A proposition in mathematics -/
structure Proposition where
  statement : Prop

/-- A theorem in mathematics -/
structure Theorem where
  statement : Prop
  proof : statement

/-- The converse of a proposition -/
def converse (p : Proposition) : Proposition :=
  ⟨¬p.statement⟩

theorem incorrect_statement_about_converses :
  ¬(∀ (p : Proposition), ∃ (c : Proposition), c = converse p ∧
     ∃ (t : Theorem), ¬∃ (c : Proposition), c = converse ⟨t.statement⟩) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_about_converses_l1021_102168


namespace NUMINAMATH_CALUDE_female_democrats_count_l1021_102142

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) 
  (h1 : female + male = total)
  (h2 : total = 720)
  (h3 : female / 2 + male / 4 = total / 3) :
  female / 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_female_democrats_count_l1021_102142


namespace NUMINAMATH_CALUDE_perimeter_difference_is_zero_l1021_102130

/-- A figure composed of unit squares -/
structure UnitSquareFigure where
  squares : ℕ
  perimeter : ℕ

/-- T-shaped figure with 5 unit squares -/
def t_shape : UnitSquareFigure :=
  { squares := 5,
    perimeter := 8 }

/-- Cross-shaped figure with 5 unit squares -/
def cross_shape : UnitSquareFigure :=
  { squares := 5,
    perimeter := 8 }

/-- The positive difference between the perimeters of the T-shape and cross-shape is 0 -/
theorem perimeter_difference_is_zero :
  (t_shape.perimeter : ℤ) - (cross_shape.perimeter : ℤ) = 0 := by
  sorry

#check perimeter_difference_is_zero

end NUMINAMATH_CALUDE_perimeter_difference_is_zero_l1021_102130


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1021_102164

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 9 * x₁ + 7 = 0) → 
  (2 * x₂^2 - 9 * x₂ + 7 = 0) → 
  x₁^2 + x₂^2 = 53/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1021_102164


namespace NUMINAMATH_CALUDE_union_of_sets_l1021_102139

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 5}
  A ∪ B = {1, 2, 3, 5} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1021_102139


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1021_102196

-- Define the complex number z
def z : ℂ := (1 - Complex.I) * (2 + Complex.I)

-- Theorem statement
theorem z_in_fourth_quadrant :
  Complex.re z > 0 ∧ Complex.im z < 0 :=
by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1021_102196


namespace NUMINAMATH_CALUDE_sylvester_theorem_l1021_102194

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define when a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define when points are collinear
def collinear (p1 p2 p3 : Point) : Prop :=
  ∃ l : Line, pointOnLine p1 l ∧ pointOnLine p2 l ∧ pointOnLine p3 l

-- Define when a set of points is not all collinear
def notAllCollinear (E : Set Point) : Prop :=
  ∃ p1 p2 p3 : Point, p1 ∈ E ∧ p2 ∈ E ∧ p3 ∈ E ∧ ¬collinear p1 p2 p3

-- Sylvester's theorem statement
theorem sylvester_theorem (E : Set Point) (h1 : E.Finite) (h2 : notAllCollinear E) :
  ∃ l : Line, ∃ p1 p2 : Point, p1 ∈ E ∧ p2 ∈ E ∧ p1 ≠ p2 ∧
    pointOnLine p1 l ∧ pointOnLine p2 l ∧
    ∀ p3 : Point, p3 ∈ E → pointOnLine p3 l → (p3 = p1 ∨ p3 = p2) :=
  sorry

end NUMINAMATH_CALUDE_sylvester_theorem_l1021_102194


namespace NUMINAMATH_CALUDE_expression_value_l1021_102112

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 + z^2 + 2*x*y*z = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1021_102112


namespace NUMINAMATH_CALUDE_segment_length_parallel_to_x_axis_l1021_102184

/-- Given two points M and N, where M's coordinates depend on parameter a,
    and MN is parallel to the x-axis, prove that the length of MN is 6. -/
theorem segment_length_parallel_to_x_axis 
  (a : ℝ) 
  (M : ℝ × ℝ := (a + 3, a - 4))
  (N : ℝ × ℝ := (-1, -2))
  (h_parallel : M.2 = N.2) : 
  abs (M.1 - N.1) = 6 := by
sorry

end NUMINAMATH_CALUDE_segment_length_parallel_to_x_axis_l1021_102184


namespace NUMINAMATH_CALUDE_wall_height_calculation_l1021_102120

/-- Calculates the height of a wall given its dimensions and the number and size of bricks used --/
theorem wall_height_calculation (brick_length brick_width brick_height : ℝ)
  (wall_length wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 20 →
  brick_width = 10 →
  brick_height = 7.5 →
  wall_length = 27 →
  wall_width = 2 →
  num_bricks = 27000 →
  ∃ (wall_height : ℝ), wall_height = 0.75 ∧
    wall_length * wall_width * wall_height = (brick_length * brick_width * brick_height * num_bricks) / 1000000 := by
  sorry

#check wall_height_calculation

end NUMINAMATH_CALUDE_wall_height_calculation_l1021_102120


namespace NUMINAMATH_CALUDE_problem_solution_l1021_102138

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 6 * x^3 + 18 * x^2 * y * z = 3 * x^4 + 6 * x^3 * y * z) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1021_102138


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1021_102177

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 2 + Real.log 0.04 / Real.log 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1021_102177


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1021_102101

/-- A line that does not pass through the origin -/
structure Line where
  slope : ℝ
  intercept : ℝ
  not_through_origin : intercept ≠ 0

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = x^2 -/
def parabola (p : Point) : Prop :=
  p.y = p.x^2

/-- The line intersects the parabola at two points -/
def intersects_parabola (l : Line) (A B : Point) : Prop :=
  parabola A ∧ parabola B ∧
  A.y = l.slope * A.x + l.intercept ∧
  B.y = l.slope * B.x + l.intercept ∧
  A ≠ B

/-- The circle with diameter AB passes through the origin -/
def circle_through_origin (A B : Point) : Prop :=
  A.x * B.x + A.y * B.y = 0

/-- The main theorem -/
theorem line_passes_through_fixed_point (l : Line) (A B : Point)
  (h_intersects : intersects_parabola l A B)
  (h_circle : circle_through_origin A B) :
  ∃ (P : Point), P.x = 0 ∧ P.y = 1 ∧ P.y = l.slope * P.x + l.intercept :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1021_102101


namespace NUMINAMATH_CALUDE_vector_independence_l1021_102153

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b (m : ℝ) : Fin 2 → ℝ := ![m - 1, m + 3]

theorem vector_independence (m : ℝ) :
  LinearIndependent ℝ ![vector_a, vector_b m] ↔ m ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_independence_l1021_102153


namespace NUMINAMATH_CALUDE_all_rationals_same_color_l1021_102154

-- Define a color type
def Color := Nat

-- Define a coloring function
def coloring : ℚ → Color := sorry

-- Define the main theorem
theorem all_rationals_same_color (n : Nat) 
  (h : ∀ a b : ℚ, coloring a ≠ coloring b → 
       coloring ((a + b) / 2) ≠ coloring a ∧ 
       coloring ((a + b) / 2) ≠ coloring b) : 
  ∀ x y : ℚ, coloring x = coloring y := by sorry

end NUMINAMATH_CALUDE_all_rationals_same_color_l1021_102154


namespace NUMINAMATH_CALUDE_max_value_on_circle_l1021_102132

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 10 →
  ∃ (max : ℝ), max = 5 * Real.sqrt 10 ∧ ∀ (a b : ℝ), a^2 + b^2 = 10 → 3*a + 4*b ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l1021_102132


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1021_102173

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point (A : Point) (l1 l2 : Line) :
  A.x = 2 ∧ A.y = -3 ∧
  l1.a = 1 ∧ l1.b = -2 ∧ l1.c = -3 ∧
  l2.a = 2 ∧ l2.b = 1 ∧ l2.c = -1 →
  A.liesOn l2 ∧ l1.perpendicular l2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1021_102173


namespace NUMINAMATH_CALUDE_largest_intersection_point_l1021_102124

/-- The polynomial function -/
def polynomial (a : ℝ) (x : ℝ) : ℝ := x^6 - 8*x^5 + 22*x^4 + 6*x^3 + a*x^2

/-- The line function -/
def line (c : ℝ) (x : ℝ) : ℝ := 2*x + c

/-- The intersection function -/
def intersection (a c : ℝ) (x : ℝ) : ℝ := polynomial a x - line c x

theorem largest_intersection_point (a c : ℝ) :
  (∃ p q : ℝ, p ≠ q ∧ 
    (∀ x : ℝ, intersection a c x = 0 ↔ (x = p ∨ x = q)) ∧
    (∀ x : ℝ, (x - p)^3 * (x - q) = intersection a c x)) →
  (∀ x : ℝ, intersection a c x = 0 → x ≤ 7) ∧
  (∃ x : ℝ, intersection a c x = 0 ∧ x = 7) :=
by sorry

end NUMINAMATH_CALUDE_largest_intersection_point_l1021_102124


namespace NUMINAMATH_CALUDE_peaches_picked_l1021_102180

def initial_peaches : ℝ := 34.0
def total_peaches : ℕ := 120

theorem peaches_picked (picked : ℕ) : 
  picked = total_peaches - Int.floor initial_peaches := by sorry

end NUMINAMATH_CALUDE_peaches_picked_l1021_102180


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l1021_102151

/-- A geometric sequence with specified terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  geometric_sequence a → a 2 = -2 → a 6 = -32 → a 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l1021_102151


namespace NUMINAMATH_CALUDE_sleep_hours_calculation_l1021_102175

def hours_in_day : ℕ := 24
def work_hours : ℕ := 6
def chore_hours : ℕ := 5

theorem sleep_hours_calculation :
  hours_in_day - (work_hours + chore_hours) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sleep_hours_calculation_l1021_102175


namespace NUMINAMATH_CALUDE_triangle_theorem_l1021_102148

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.c * Real.sin t.A = t.a * Real.cos t.C)
  (h2 : 0 < t.A ∧ t.A < Real.pi)
  (h3 : 0 < t.B ∧ t.B < Real.pi)
  (h4 : 0 < t.C ∧ t.C < Real.pi)
  (h5 : t.A + t.B + t.C = Real.pi) : 
  (t.C = Real.pi / 4) ∧ 
  (∃ (max : Real), ∀ (A B : Real), 
    (0 < A ∧ A < 3 * Real.pi / 4) → 
    (B = 3 * Real.pi / 4 - A) → 
    (Real.sqrt 3 * Real.sin A - Real.cos (B + Real.pi / 4) ≤ max) ∧
    (max = 2)) ∧
  (Real.sqrt 3 * Real.sin (Real.pi / 3) - Real.cos (5 * Real.pi / 12 + Real.pi / 4) = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1021_102148


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1021_102157

theorem fraction_equation_solution (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1021_102157


namespace NUMINAMATH_CALUDE_chef_apples_used_l1021_102150

/-- The number of apples the chef used to make pies -/
def applesUsed (initialApples remainingApples : ℕ) : ℕ :=
  initialApples - remainingApples

theorem chef_apples_used :
  let initialApples : ℕ := 43
  let remainingApples : ℕ := 2
  applesUsed initialApples remainingApples = 41 := by
  sorry

end NUMINAMATH_CALUDE_chef_apples_used_l1021_102150


namespace NUMINAMATH_CALUDE_butterfat_mixture_l1021_102107

/-- Proves that adding 16 gallons of 10% butterfat milk to 8 gallons of 40% butterfat milk 
    results in a mixture with 20% butterfat. -/
theorem butterfat_mixture : 
  let initial_volume : ℝ := 8
  let initial_butterfat_percent : ℝ := 40
  let added_volume : ℝ := 16
  let added_butterfat_percent : ℝ := 10
  let final_butterfat_percent : ℝ := 20
  let total_volume := initial_volume + added_volume
  let total_butterfat := (initial_volume * initial_butterfat_percent / 100) + 
                         (added_volume * added_butterfat_percent / 100)
  (total_butterfat / total_volume) * 100 = final_butterfat_percent :=
by
  sorry

#check butterfat_mixture

end NUMINAMATH_CALUDE_butterfat_mixture_l1021_102107


namespace NUMINAMATH_CALUDE_reciprocal_roots_equation_l1021_102143

theorem reciprocal_roots_equation (m n : ℝ) (hn : n ≠ 0) :
  let original_eq := fun x => x^2 + m*x + n
  let reciprocal_eq := fun x => n*x^2 + m*x + 1
  ∀ x, original_eq x = 0 → reciprocal_eq (1/x) = 0 :=
sorry


end NUMINAMATH_CALUDE_reciprocal_roots_equation_l1021_102143


namespace NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l1021_102183

/-- The ratio of side lengths of an equilateral triangle and a regular pentagon with equal perimeters -/
theorem triangle_pentagon_side_ratio :
  let triangle_perimeter : ℝ := 60
  let pentagon_perimeter : ℝ := 60
  let triangle_side : ℝ := triangle_perimeter / 3
  let pentagon_side : ℝ := pentagon_perimeter / 5
  triangle_side / pentagon_side = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l1021_102183


namespace NUMINAMATH_CALUDE_special_numbers_l1021_102134

def last_digit (n : ℕ) : ℕ := n % 10

theorem special_numbers : 
  {n : ℕ | (last_digit n) * 2016 = n} = {4032, 8064, 12096, 16128} :=
by sorry

end NUMINAMATH_CALUDE_special_numbers_l1021_102134


namespace NUMINAMATH_CALUDE_properties_of_f_l1021_102128

noncomputable def f (x : ℝ) : ℝ := (3/2) ^ x

theorem properties_of_f (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (f (x₁ + x₂) = f x₁ * f x₂) ∧
  ((f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
  (1 < x₁ → x₁ < x₂ → f x₁ / (x₁ - 1) > f x₂ / (x₂ - 1)) :=
by sorry

end NUMINAMATH_CALUDE_properties_of_f_l1021_102128


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l1021_102103

theorem intersection_point_x_coordinate 
  (a b : ℝ) 
  (h1 : a ≠ b) 
  (h2 : ∃! x y : ℝ, x^2 + 2*a*x + 6*b = x^2 + 2*b*x + 6*a) : 
  ∃ x y : ℝ, x^2 + 2*a*x + 6*b = x^2 + 2*b*x + 6*a ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l1021_102103


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l1021_102190

theorem quadratic_root_m_value :
  ∀ m : ℝ, ((-1 : ℝ)^2 + m * (-1) + 1 = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l1021_102190


namespace NUMINAMATH_CALUDE_solution_sets_equal_implies_alpha_value_l1021_102144

/-- The solution set of the inequality |2x-3| < 2 -/
def solution_set_1 : Set ℝ := {x : ℝ | |2*x - 3| < 2}

/-- The solution set of the inequality x^2 + αx + b < 0 -/
def solution_set_2 (α b : ℝ) : Set ℝ := {x : ℝ | x^2 + α*x + b < 0}

/-- Theorem stating that if the solution sets are equal, then α = -3 -/
theorem solution_sets_equal_implies_alpha_value (b : ℝ) :
  (∃ α, solution_set_1 = solution_set_2 α b) → 
  (∃ α, solution_set_1 = solution_set_2 α b ∧ α = -3) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_equal_implies_alpha_value_l1021_102144


namespace NUMINAMATH_CALUDE_august_math_problems_l1021_102133

theorem august_math_problems (first_answer second_answer third_answer : ℕ) : 
  first_answer = 600 →
  second_answer = 2 * first_answer →
  third_answer = first_answer + second_answer - 400 →
  first_answer + second_answer + third_answer = 3200 := by
sorry

end NUMINAMATH_CALUDE_august_math_problems_l1021_102133


namespace NUMINAMATH_CALUDE_mcdonald_accounting_error_l1021_102137

theorem mcdonald_accounting_error (x : ℝ) : x = 3.57 ↔ 9 * x = 32.13 := by sorry

end NUMINAMATH_CALUDE_mcdonald_accounting_error_l1021_102137


namespace NUMINAMATH_CALUDE_average_age_increase_l1021_102162

theorem average_age_increase (initial_men : ℕ) (replaced_men_ages : List ℕ) (women_avg_age : ℚ) : 
  initial_men = 8 →
  replaced_men_ages = [20, 10] →
  women_avg_age = 23 →
  (((initial_men : ℚ) * women_avg_age + (women_avg_age * 2 - replaced_men_ages.sum)) / initial_men) - women_avg_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l1021_102162


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l1021_102146

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l1021_102146


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l1021_102172

theorem least_sum_of_bases : ∃ (c d : ℕ+), 
  (∀ (c' d' : ℕ+), (2 * c' + 9 = 9 * d' + 2) → (c'.val + d'.val ≥ c.val + d.val)) ∧ 
  (2 * c + 9 = 9 * d + 2) ∧
  (c.val + d.val = 13) := by
  sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l1021_102172


namespace NUMINAMATH_CALUDE_product_of_distinct_solutions_l1021_102174

theorem product_of_distinct_solutions (x y : ℝ) : 
  x ≠ 0 → y ≠ 0 → x ≠ y → (x + 3 / x = y + 3 / y) → x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_solutions_l1021_102174


namespace NUMINAMATH_CALUDE_min_packs_for_event_l1021_102147

/-- Represents a pack of utensils -/
structure UtensilPack where
  total : Nat
  knife : Nat
  fork : Nat
  spoon : Nat
  equal_distribution : knife = fork ∧ fork = spoon
  pack_size : total = knife + fork + spoon

/-- Represents the required ratio of utensils -/
structure UtensilRatio where
  knife : Nat
  fork : Nat
  spoon : Nat

def min_packs_needed (pack : UtensilPack) (ratio : UtensilRatio) (min_spoons : Nat) : Nat :=
  sorry

theorem min_packs_for_event (pack : UtensilPack) (ratio : UtensilRatio) (min_spoons : Nat) :
  pack.total = 30 ∧
  ratio.knife = 2 ∧ ratio.fork = 3 ∧ ratio.spoon = 5 ∧
  min_spoons = 50 →
  min_packs_needed pack ratio min_spoons = 5 :=
sorry

end NUMINAMATH_CALUDE_min_packs_for_event_l1021_102147


namespace NUMINAMATH_CALUDE_equation_solution_l1021_102192

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 →
  (-15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) + 1) ↔ (x = 5/4 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1021_102192


namespace NUMINAMATH_CALUDE_circle_point_distance_range_l1021_102159

/-- Given a circle C with equation (x-a)^2 + (y-a+2)^2 = 1 and a point A(0,2),
    if there exists a point M on C such that MA^2 + MO^2 = 10,
    then 0 ≤ a ≤ 3. -/
theorem circle_point_distance_range (a : ℝ) :
  (∃ x y : ℝ, (x - a)^2 + (y - a + 2)^2 = 1 ∧ x^2 + y^2 + x^2 + (y - 2)^2 = 10) →
  0 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_point_distance_range_l1021_102159


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l1021_102145

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) 
  (h : ∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) :
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2) ∧
  (a₁ + a₃ + a₅ + a₇ = -1094) ∧
  (a₀ + a₂ + a₄ + a₆ = 1093) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l1021_102145


namespace NUMINAMATH_CALUDE_tangent_line_of_quartic_curve_l1021_102119

/-- The curve y = x^4 has a tangent line parallel to x + 2y - 8 = 0, 
    and this tangent line has the equation 8x + 16y + 3 = 0 -/
theorem tangent_line_of_quartic_curve (x y : ℝ) : 
  y = x^4 → 
  ∃ (x₀ y₀ : ℝ), y₀ = x₀^4 ∧ 
    (∀ (x' y' : ℝ), y' - y₀ = 4 * x₀^3 * (x' - x₀) → 
      ∃ (k : ℝ), y' - y₀ = k * (x' - x₀) ∧ k = -1/2) →
    8 * x₀ + 16 * y₀ + 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_of_quartic_curve_l1021_102119


namespace NUMINAMATH_CALUDE_equation_C_is_linear_l1021_102114

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x + 3 = 7 is linear -/
theorem equation_C_is_linear : is_linear_equation (λ x => 2 * x + 3) :=
by
  sorry

#check equation_C_is_linear

end NUMINAMATH_CALUDE_equation_C_is_linear_l1021_102114


namespace NUMINAMATH_CALUDE_max_projection_area_unit_cube_max_projection_area_unit_cube_proof_l1021_102163

/-- The maximum area of the orthogonal projection of a unit cube onto any plane -/
theorem max_projection_area_unit_cube : ℝ :=
  2 * Real.sqrt 3

/-- Theorem: The maximum area of the orthogonal projection of a unit cube onto any plane is 2√3 -/
theorem max_projection_area_unit_cube_proof :
  max_projection_area_unit_cube = 2 * Real.sqrt 3 := by
  sorry

#check max_projection_area_unit_cube_proof

end NUMINAMATH_CALUDE_max_projection_area_unit_cube_max_projection_area_unit_cube_proof_l1021_102163


namespace NUMINAMATH_CALUDE_inscribed_sphere_sum_l1021_102113

/-- A sphere inscribed in a right cone with base radius 15 cm and height 30 cm -/
structure InscribedSphere :=
  (b : ℝ)
  (d : ℝ)
  (radius : ℝ)
  (cone_base_radius : ℝ)
  (cone_height : ℝ)
  (radius_eq : radius = b * (Real.sqrt d - 1))
  (cone_base_radius_eq : cone_base_radius = 15)
  (cone_height_eq : cone_height = 30)

/-- Theorem stating that b + d = 12.5 for the inscribed sphere -/
theorem inscribed_sphere_sum (s : InscribedSphere) : s.b + s.d = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_sum_l1021_102113


namespace NUMINAMATH_CALUDE_rearrangement_impossibility_l1021_102188

theorem rearrangement_impossibility : ¬ ∃ (arrangement : Fin 3972 → ℕ),
  (∀ i : Fin 1986, ∃ (m n : Fin 3972), m < n ∧ 
    arrangement m = i.val + 1 ∧ 
    arrangement n = i.val + 1 ∧ 
    n.val - m.val - 1 = i.val) ∧
  (∀ k : Fin 3972, ∃ i : Fin 1986, arrangement k = i.val + 1) :=
sorry

end NUMINAMATH_CALUDE_rearrangement_impossibility_l1021_102188


namespace NUMINAMATH_CALUDE_geometry_propositions_l1021_102125

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) :
  {p₁ ∧ p₄, ¬p₂ ∨ p₃, ¬p₃ ∨ ¬p₄} = 
  {q : Prop | q = (p₁ ∧ p₄) ∨ q = (¬p₂ ∨ p₃) ∨ q = (¬p₃ ∨ ¬p₄)} ∩ 
  {q : Prop | q} :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1021_102125


namespace NUMINAMATH_CALUDE_find_b_l1021_102170

/-- Given the conditions, prove that b = -2 --/
theorem find_b (a : ℕ) (b : ℝ) : 
  (2 * (a.choose 2) - (a.choose 1 - 1) * 6 = 0) →  -- Condition 1
  (b ≠ 0) →                                        -- Condition 3
  (a.choose 1 * b = -12) →                         -- Condition 2 (simplified)
  b = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l1021_102170


namespace NUMINAMATH_CALUDE_fourth_side_length_l1021_102193

/-- A quadrilateral inscribed in a circle with radius 200√2, where three sides have length 200 -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of three sides of the quadrilateral -/
  side_length : ℝ
  /-- The fourth side of the quadrilateral -/
  fourth_side : ℝ
  /-- Assertion that the radius is 200√2 -/
  radius_eq : radius = 200 * Real.sqrt 2
  /-- Assertion that three sides have length 200 -/
  three_sides_eq : side_length = 200

/-- Theorem stating that the fourth side of the quadrilateral has length 500 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.fourth_side = 500 := by
  sorry

#check fourth_side_length

end NUMINAMATH_CALUDE_fourth_side_length_l1021_102193


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_35_l1021_102169

/-- The complement of an angle in degrees -/
def complement (α : ℝ) : ℝ := 90 - α

/-- The supplement of an angle in degrees -/
def supplement (α : ℝ) : ℝ := 180 - α

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125 degrees -/
theorem supplement_of_complement_of_35 :
  supplement (complement 35) = 125 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_35_l1021_102169


namespace NUMINAMATH_CALUDE_equation_solution_l1021_102198

theorem equation_solution :
  ∃! (x : ℝ), x ≠ 0 ∧ (7*x)^5 = (14*x)^4 ∧ x = 16/7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1021_102198
