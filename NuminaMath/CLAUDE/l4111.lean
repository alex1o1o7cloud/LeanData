import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4111_411183

theorem complex_modulus_problem (z : ℂ) (h : (1 + 2*I)*z = 1 - I) : Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4111_411183


namespace NUMINAMATH_CALUDE_fraction_problem_l4111_411127

theorem fraction_problem (N : ℝ) (F : ℝ) : 
  F * (1/4 * N) = 15 ∧ (3/10) * N = 54 → F = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l4111_411127


namespace NUMINAMATH_CALUDE_peanut_cost_per_pound_l4111_411139

/-- The cost per pound of peanuts at Peanut Emporium -/
def cost_per_pound : ℝ := 3

/-- The minimum purchase amount in pounds -/
def minimum_purchase : ℝ := 15

/-- The amount purchased over the minimum in pounds -/
def over_minimum : ℝ := 20

/-- The total cost of the purchase in dollars -/
def total_cost : ℝ := 105

/-- Proof that the cost per pound of peanuts is $3 -/
theorem peanut_cost_per_pound :
  cost_per_pound = total_cost / (minimum_purchase + over_minimum) := by
  sorry

end NUMINAMATH_CALUDE_peanut_cost_per_pound_l4111_411139


namespace NUMINAMATH_CALUDE_lathe_processing_time_l4111_411175

/-- Given that 3 lathes can process 180 parts in 4 hours,
    prove that 5 lathes will process 600 parts in 8 hours. -/
theorem lathe_processing_time
  (initial_lathes : ℕ)
  (initial_parts : ℕ)
  (initial_hours : ℕ)
  (target_lathes : ℕ)
  (target_parts : ℕ)
  (h1 : initial_lathes = 3)
  (h2 : initial_parts = 180)
  (h3 : initial_hours = 4)
  (h4 : target_lathes = 5)
  (h5 : target_parts = 600)
  : (target_parts : ℚ) / (target_lathes : ℚ) * (initial_lathes : ℚ) / (initial_parts : ℚ) * (initial_hours : ℚ) = 8 := by
  sorry


end NUMINAMATH_CALUDE_lathe_processing_time_l4111_411175


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_five_min_value_attained_l4111_411165

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3*b = 5*a*b → 3*x + 4*y ≤ 3*a + 4*b :=
by sorry

theorem min_value_is_five (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 3*y = 5*x*y ∧ 3*x + 4*y < 5 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_five_min_value_attained_l4111_411165


namespace NUMINAMATH_CALUDE_fraction_difference_simplification_l4111_411141

theorem fraction_difference_simplification :
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_simplification_l4111_411141


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l4111_411186

theorem smallest_n_congruence (n : ℕ) : 
  (0 ≤ n ∧ n < 53 ∧ 50 * n % 53 = 47 % 53) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l4111_411186


namespace NUMINAMATH_CALUDE_book_division_proof_l4111_411187

def number_of_divisions (total : ℕ) (target : ℕ) : ℕ :=
  if total ≤ target then 0
  else 1 + number_of_divisions (total / 2) target

theorem book_division_proof :
  number_of_divisions 400 25 = 4 :=
by sorry

end NUMINAMATH_CALUDE_book_division_proof_l4111_411187


namespace NUMINAMATH_CALUDE_point_coordinates_proof_l4111_411110

/-- Given two points M and N, and a point P such that MP = 1/2 * MN, 
    prove that P has specific coordinates. -/
theorem point_coordinates_proof (M N P : ℝ × ℝ) : 
  M = (3, 2) → 
  N = (-5, -5) → 
  P - M = (1 / 2 : ℝ) • (N - M) → 
  P = (-1, -3/2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_proof_l4111_411110


namespace NUMINAMATH_CALUDE_container_dimensions_l4111_411125

theorem container_dimensions (a b c : ℝ) :
  a * b * 16 = 2400 →
  a * c * 10 = 2400 →
  b * c * 9.6 = 2400 →
  a = 12 ∧ b = 12.5 ∧ c = 20 := by
sorry

end NUMINAMATH_CALUDE_container_dimensions_l4111_411125


namespace NUMINAMATH_CALUDE_function_identity_l4111_411115

open Real

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, f (1 - cos x) = sin x ^ 2) :
  ∀ x, f x = 2 * x - x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l4111_411115


namespace NUMINAMATH_CALUDE_base8_to_base5_conversion_l4111_411140

-- Define a function to convert from base 8 to base 10
def base8_to_base10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from base 10 to base 5
def base10_to_base5 (n : Nat) : Nat :=
  let thousands := n / 625
  let hundreds := (n % 625) / 125
  let tens := ((n % 625) % 125) / 25
  let ones := (((n % 625) % 125) % 25) / 5
  thousands * 1000 + hundreds * 100 + tens * 10 + ones

theorem base8_to_base5_conversion :
  base10_to_base5 (base8_to_base10 653) = 3202 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base5_conversion_l4111_411140


namespace NUMINAMATH_CALUDE_triangle_otimes_calculation_l4111_411100

def triangle (a b : ℝ) : ℝ := a + b + a * b - 1

def otimes (a b : ℝ) : ℝ := a^2 - a * b + b^2

theorem triangle_otimes_calculation : triangle 3 (otimes 2 4) = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_otimes_calculation_l4111_411100


namespace NUMINAMATH_CALUDE_quadratic_sum_l4111_411192

/-- Given a quadratic expression x^2 - 20x + 49, prove that when written in the form (x+b)^2 + c,
    the sum of b and c is equal to -61. -/
theorem quadratic_sum (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 49 = (x + b)^2 + c) → b + c = -61 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l4111_411192


namespace NUMINAMATH_CALUDE_simplified_expression_and_evaluation_l4111_411135

theorem simplified_expression_and_evaluation (x : ℝ) 
  (h1 : x ≠ -3) (h2 : x ≠ 3) :
  (3 / (x - 3) - 3 * x / (x^2 - 9)) / ((3 * x - 9) / (x^2 - 6 * x + 9)) = 3 / (x + 3) ∧
  (3 / (1 + 3) = 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_and_evaluation_l4111_411135


namespace NUMINAMATH_CALUDE_conditional_probability_good_air_quality_l4111_411119

-- Define the probability of good air quality on any given day
def p_good_day : ℝ := 0.75

-- Define the probability of good air quality for two consecutive days
def p_two_good_days : ℝ := 0.6

-- State the theorem
theorem conditional_probability_good_air_quality :
  (p_two_good_days / p_good_day : ℝ) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_good_air_quality_l4111_411119


namespace NUMINAMATH_CALUDE_vegetarians_count_l4111_411138

/-- Represents the eating habits of a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_non_veg : ℕ
  both : ℕ

/-- Calculates the total number of people who eat vegetarian food -/
def total_vegetarians (fd : FamilyDiet) : ℕ :=
  fd.only_veg + fd.both

/-- Theorem stating that the number of vegetarians in the given family is 20 -/
theorem vegetarians_count (fd : FamilyDiet) 
  (h1 : fd.only_veg = 11)
  (h2 : fd.only_non_veg = 6)
  (h3 : fd.both = 9) :
  total_vegetarians fd = 20 := by
  sorry

#eval total_vegetarians ⟨11, 6, 9⟩

end NUMINAMATH_CALUDE_vegetarians_count_l4111_411138


namespace NUMINAMATH_CALUDE_matrix_product_l4111_411195

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 4]

theorem matrix_product :
  A * B = !![17, -5; 16, -20] := by sorry

end NUMINAMATH_CALUDE_matrix_product_l4111_411195


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l4111_411178

theorem no_solution_fractional_equation :
  ¬∃ (x : ℝ), (3 / x) + (6 / (x - 1)) - ((x + 5) / (x^2 - x)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l4111_411178


namespace NUMINAMATH_CALUDE_common_difference_not_three_l4111_411158

def is_valid_sequence (d : ℕ+) : Prop :=
  ∃ (n : ℕ+), 1 + (n - 1) * d = 81

theorem common_difference_not_three :
  ¬(is_valid_sequence 3) := by
  sorry

end NUMINAMATH_CALUDE_common_difference_not_three_l4111_411158


namespace NUMINAMATH_CALUDE_fuel_purchase_calculation_l4111_411143

/-- Given the cost of fuel per gallon, the fuel consumption rate per hour,
    and the total time to consume all fuel, calculate the number of gallons purchased. -/
theorem fuel_purchase_calculation 
  (cost_per_gallon : ℝ) 
  (consumption_rate_per_hour : ℝ) 
  (total_hours : ℝ) 
  (h1 : cost_per_gallon = 0.70)
  (h2 : consumption_rate_per_hour = 0.40)
  (h3 : total_hours = 175) :
  (consumption_rate_per_hour * total_hours) / cost_per_gallon = 100 := by
  sorry

end NUMINAMATH_CALUDE_fuel_purchase_calculation_l4111_411143


namespace NUMINAMATH_CALUDE_storks_and_birds_l4111_411112

theorem storks_and_birds (initial_birds initial_storks joining_storks : ℕ) :
  initial_birds = 4 →
  initial_storks = 3 →
  joining_storks = 6 →
  (initial_storks + joining_storks) - initial_birds = 5 := by
  sorry

end NUMINAMATH_CALUDE_storks_and_birds_l4111_411112


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l4111_411145

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p ^ 2 + 9 * p - 21 = 0) → 
  (3 * q ^ 2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = 122 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l4111_411145


namespace NUMINAMATH_CALUDE_root_product_theorem_l4111_411189

theorem root_product_theorem (x₁ x₂ x₃ : ℝ) : 
  x₃ < x₂ ∧ x₂ < x₁ →
  (Real.sqrt 120 * x₁^3 - 480 * x₁^2 + 8 * x₁ + 1 = 0) →
  (Real.sqrt 120 * x₂^3 - 480 * x₂^2 + 8 * x₂ + 1 = 0) →
  (Real.sqrt 120 * x₃^3 - 480 * x₃^2 + 8 * x₃ + 1 = 0) →
  x₂ * (x₁ + x₃) = -1/120 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l4111_411189


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l4111_411149

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- A line that intersects the hyperbola -/
def intersecting_line (m : ℝ) (x y : ℝ) : Prop :=
  x - y + m = 0

/-- Two points are perpendicular from the origin -/
def perpendicular_from_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_intersection_theorem (h : Hyperbola)
  (h_eccentricity : h.a / Real.sqrt (h.a^2 + h.b^2) = Real.sqrt 3 / 3)
  (h_imaginary_axis : h.b = Real.sqrt 2)
  (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h_on_hyperbola₁ : hyperbola_equation h x₁ y₁)
  (h_on_hyperbola₂ : hyperbola_equation h x₂ y₂)
  (h_on_line₁ : intersecting_line m x₁ y₁)
  (h_on_line₂ : intersecting_line m x₂ y₂)
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂))
  (h_perpendicular : perpendicular_from_origin x₁ y₁ x₂ y₂) :
  m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l4111_411149


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l4111_411168

theorem no_real_roots_condition (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 ≠ 0) ↔ k < -1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l4111_411168


namespace NUMINAMATH_CALUDE_complex_expression_equals_two_l4111_411162

theorem complex_expression_equals_two :
  (2023 - Real.pi) ^ 0 + (1/2)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (π/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_two_l4111_411162


namespace NUMINAMATH_CALUDE_quadratic_equations_roots_l4111_411193

theorem quadratic_equations_roots :
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = -5 ∧ x₁^2 + 6*x₁ + 5 = 0 ∧ x₂^2 + 6*x₂ + 5 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 2 + Real.sqrt 5 ∧ y₂ = 2 - Real.sqrt 5 ∧ y₁^2 - 4*y₁ - 1 = 0 ∧ y₂^2 - 4*y₂ - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_roots_l4111_411193


namespace NUMINAMATH_CALUDE_doctors_distribution_l4111_411164

def distribute_doctors (n : ℕ) (k : ℕ) : Prop :=
  ∃ (ways : ℕ),
    n = 7 ∧
    k = 3 ∧
    ways = (Nat.choose 2 1) * (Nat.choose 5 2) * (Nat.choose 3 1) +
           (Nat.choose 5 3) * (Nat.choose 2 1) ∧
    ways = 80

theorem doctors_distribution :
  ∀ (n k : ℕ), distribute_doctors n k :=
sorry

end NUMINAMATH_CALUDE_doctors_distribution_l4111_411164


namespace NUMINAMATH_CALUDE_triangle_inequalities_l4111_411126

/-- Triangle inequality theorems -/
theorem triangle_inequalities (a b c : ℝ) (S : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0) : 
  (a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S) ∧ 
  (a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 ≥ 4 * Real.sqrt 3 * S) ∧
  ((a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S ∨ 
    a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 = 4 * Real.sqrt 3 * S) ↔ 
   a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l4111_411126


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4111_411161

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 4) :
  (((2 * x + 2) / (x^2 - 1) + 1) / ((x + 1) / (x^2 - 2*x + 1))) = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4111_411161


namespace NUMINAMATH_CALUDE_river_flow_speed_l4111_411147

/-- Proves that the speed of river flow is 2 km/hr given the conditions of the boat journey --/
theorem river_flow_speed 
  (distance : ℝ) 
  (boat_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : distance = 32) 
  (h2 : boat_speed = 6) 
  (h3 : total_time = 12) : 
  ∃ (v : ℝ), v = 2 ∧ 
    (distance / (boat_speed - v) + distance / (boat_speed + v) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_river_flow_speed_l4111_411147


namespace NUMINAMATH_CALUDE_alexas_weight_l4111_411116

/-- Given the combined weight of two people and the weight of one person, 
    calculate the weight of the other person. -/
theorem alexas_weight (total_weight katerina_weight : ℕ) 
  (h1 : total_weight = 95)
  (h2 : katerina_weight = 49) :
  total_weight - katerina_weight = 46 := by
  sorry

end NUMINAMATH_CALUDE_alexas_weight_l4111_411116


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_equals_three_l4111_411191

theorem sqrt_equality_implies_t_equals_three (t : ℝ) : 
  Real.sqrt (2 * Real.sqrt (t - 2)) = (7 - t) ^ (1/4) → t = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_equals_three_l4111_411191


namespace NUMINAMATH_CALUDE_not_necessarily_p_or_q_l4111_411130

theorem not_necessarily_p_or_q (h1 : ¬p) (h2 : ¬(p ∧ q)) : 
  ¬∀ (p q : Prop), p ∨ q := by
sorry

end NUMINAMATH_CALUDE_not_necessarily_p_or_q_l4111_411130


namespace NUMINAMATH_CALUDE_trumpets_fraction_in_band_l4111_411196

-- Define the total number of each instrument
def total_flutes : ℕ := 20
def total_clarinets : ℕ := 30
def total_trumpets : ℕ := 60
def total_pianists : ℕ := 20

-- Define the fraction of each instrument that got in
def flutes_fraction : ℚ := 4/5  -- 80%
def clarinets_fraction : ℚ := 1/2
def pianists_fraction : ℚ := 1/10

-- Define the total number of people in the band
def total_in_band : ℕ := 53

-- Theorem to prove
theorem trumpets_fraction_in_band : 
  (total_in_band - 
   (flutes_fraction * total_flutes + 
    clarinets_fraction * total_clarinets + 
    pianists_fraction * total_pianists)) / total_trumpets = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_trumpets_fraction_in_band_l4111_411196


namespace NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l4111_411137

def calculate_koolaid_percentage (initial_powder : ℚ) (initial_water : ℚ) (evaporated_water : ℚ) (water_multiplier : ℚ) : ℚ :=
  let remaining_water := initial_water - evaporated_water
  let final_water := remaining_water * water_multiplier
  let total_liquid := initial_powder + final_water
  (initial_powder / total_liquid) * 100

theorem koolaid_percentage_is_four_percent :
  calculate_koolaid_percentage 2 16 4 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l4111_411137


namespace NUMINAMATH_CALUDE_cereal_eating_time_l4111_411154

def mr_fat_rate : ℚ := 1 / 20
def mr_thin_rate : ℚ := 1 / 25
def total_cereal : ℚ := 4

def combined_rate : ℚ := mr_fat_rate + mr_thin_rate

theorem cereal_eating_time :
  (total_cereal / combined_rate) = 400 / 9 := by sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l4111_411154


namespace NUMINAMATH_CALUDE_paper_stack_height_l4111_411182

/-- Given a ream of paper with known thickness and sheet count, 
    calculate the number of sheets in a stack of a different height -/
theorem paper_stack_height (ream_sheets : ℕ) (ream_thickness : ℝ) (stack_height : ℝ) :
  ream_sheets > 0 →
  ream_thickness > 0 →
  stack_height > 0 →
  ream_sheets * (stack_height / ream_thickness) = 900 :=
by
  -- Assuming ream_sheets = 400, ream_thickness = 4, and stack_height = 9
  sorry

#check paper_stack_height 400 4 9

end NUMINAMATH_CALUDE_paper_stack_height_l4111_411182


namespace NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_intersection_l4111_411136

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem union_necessary_not_sufficient_for_intersection :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧
  (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) :=
sorry

end NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_intersection_l4111_411136


namespace NUMINAMATH_CALUDE_donation_amount_l4111_411167

def barbara_stuffed_animals : ℕ := 9
def trish_stuffed_animals : ℕ := 2 * barbara_stuffed_animals
def sam_stuffed_animals : ℕ := barbara_stuffed_animals + 5
def linda_stuffed_animals : ℕ := sam_stuffed_animals - 7

def barbara_price : ℚ := 2
def trish_price : ℚ := (3:ℚ)/2
def sam_price : ℚ := (5:ℚ)/2
def linda_price : ℚ := 3

def discount_rate : ℚ := (1:ℚ)/10

theorem donation_amount (barbara_stuffed_animals : ℕ) (trish_stuffed_animals : ℕ) 
  (sam_stuffed_animals : ℕ) (linda_stuffed_animals : ℕ) (barbara_price : ℚ) 
  (trish_price : ℚ) (sam_price : ℚ) (linda_price : ℚ) (discount_rate : ℚ) :
  trish_stuffed_animals = 2 * barbara_stuffed_animals →
  sam_stuffed_animals = barbara_stuffed_animals + 5 →
  linda_stuffed_animals = sam_stuffed_animals - 7 →
  barbara_price = 2 →
  trish_price = (3:ℚ)/2 →
  sam_price = (5:ℚ)/2 →
  linda_price = 3 →
  discount_rate = (1:ℚ)/10 →
  (1 - discount_rate) * (barbara_stuffed_animals * barbara_price + 
    trish_stuffed_animals * trish_price + sam_stuffed_animals * sam_price + 
    linda_stuffed_animals * linda_price) = (909:ℚ)/10 := by
  sorry

end NUMINAMATH_CALUDE_donation_amount_l4111_411167


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l4111_411113

theorem fourth_root_equation_solution :
  let f (x : ℝ) := (Real.rpow (61 - 3*x) (1/4) + Real.rpow (17 + 3*x) (1/4))
  ∀ x : ℝ, f x = 6 ↔ x = 7 ∨ x = -23 := by
sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l4111_411113


namespace NUMINAMATH_CALUDE_bread_products_wasted_l4111_411174

/-- Calculates the pounds of bread products wasted in a food fight scenario -/
theorem bread_products_wasted (minimum_wage hours_worked meat_pounds meat_price 
  fruit_veg_pounds fruit_veg_price bread_price janitor_hours janitor_normal_pay : ℝ) 
  (h1 : minimum_wage = 8)
  (h2 : hours_worked = 50)
  (h3 : meat_pounds = 20)
  (h4 : meat_price = 5)
  (h5 : fruit_veg_pounds = 15)
  (h6 : fruit_veg_price = 4)
  (h7 : bread_price = 1.5)
  (h8 : janitor_hours = 10)
  (h9 : janitor_normal_pay = 10) : 
  (minimum_wage * hours_worked - 
   (meat_pounds * meat_price + 
    fruit_veg_pounds * fruit_veg_price + 
    janitor_hours * janitor_normal_pay * 1.5)) / bread_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_bread_products_wasted_l4111_411174


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_min_value_sqrt_sum_attained_l4111_411169

theorem min_value_sqrt_sum (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((1 - x)^2 + (-1 + x)^2) ≥ Real.sqrt 10 :=
sorry

theorem min_value_sqrt_sum_attained : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((1 - x)^2 + (-1 + x)^2) = Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_min_value_sqrt_sum_attained_l4111_411169


namespace NUMINAMATH_CALUDE_brown_mice_count_l4111_411176

theorem brown_mice_count (total white brown : ℕ) : 
  total = white + brown →
  (2 : ℚ) / 3 * total = white →
  white = 14 →
  brown = 7 := by
sorry

end NUMINAMATH_CALUDE_brown_mice_count_l4111_411176


namespace NUMINAMATH_CALUDE_max_sleep_duration_l4111_411172

/-- A time represented by hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Checks if a given time is a happy moment -/
def is_happy_moment (t : Time) : Prop :=
  (t.hours = 4 * t.minutes) ∨ (t.minutes = 4 * t.hours)

/-- List of all happy moments in a day -/
def happy_moments : List Time :=
  sorry

/-- Calculates the time difference between two times in minutes -/
def time_difference (t1 t2 : Time) : ℕ :=
  sorry

/-- Theorem stating the maximum sleep duration -/
theorem max_sleep_duration :
  ∃ (t1 t2 : Time),
    t1 ∈ happy_moments ∧
    t2 ∈ happy_moments ∧
    time_difference t1 t2 = 239 ∧
    ∀ (t3 t4 : Time),
      t3 ∈ happy_moments →
      t4 ∈ happy_moments →
      time_difference t3 t4 ≤ 239 :=
  sorry

end NUMINAMATH_CALUDE_max_sleep_duration_l4111_411172


namespace NUMINAMATH_CALUDE_distance_between_homes_l4111_411163

def uphill_speed : ℝ := 3
def downhill_speed : ℝ := 6
def time_vasya_to_petya : ℝ := 2.5
def time_petya_to_vasya : ℝ := 3.5

theorem distance_between_homes : ℝ := by
  -- Define the distance between homes
  let distance : ℝ := 12

  -- Prove that the distance satisfies the given conditions
  have h1 : distance / uphill_speed + distance / downhill_speed = time_vasya_to_petya := by sorry
  have h2 : distance / downhill_speed + distance / uphill_speed = time_petya_to_vasya := by sorry

  -- Conclude that the distance is 12 km
  exact distance

end NUMINAMATH_CALUDE_distance_between_homes_l4111_411163


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l4111_411123

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l4111_411123


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l4111_411156

theorem complex_fraction_equality (z : ℂ) (h : z + Complex.I = 4 - Complex.I) :
  z / (4 + 2 * Complex.I) = (3 - 4 * Complex.I) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l4111_411156


namespace NUMINAMATH_CALUDE_approximate_probability_of_high_quality_l4111_411103

def sample_sizes : List ℕ := [20, 50, 100, 200, 500, 1000, 1500, 2000]

def high_quality_counts : List ℕ := [19, 47, 91, 184, 462, 921, 1379, 1846]

def frequencies : List ℚ := [
  950/1000, 940/1000, 910/1000, 920/1000, 924/1000, 921/1000, 919/1000, 923/1000
]

theorem approximate_probability_of_high_quality (ε : ℚ) (hε : ε = 1/100) :
  ∃ (p : ℚ), abs (p - (List.sum frequencies / frequencies.length)) ≤ ε ∧ p = 92/100 := by
  sorry

end NUMINAMATH_CALUDE_approximate_probability_of_high_quality_l4111_411103


namespace NUMINAMATH_CALUDE_exist_good_numbers_not_preserving_sum_of_digits_l4111_411128

/-- A natural number is "good" if its decimal representation contains only zeros and ones. -/
def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The sum of digits of a natural number in base 10. -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem stating that there exist good numbers whose product is good,
    but the sum of digits property doesn't hold. -/
theorem exist_good_numbers_not_preserving_sum_of_digits :
  ∃ (A B : ℕ), is_good A ∧ is_good B ∧ is_good (A * B) ∧
    sum_of_digits (A * B) ≠ sum_of_digits A * sum_of_digits B := by
  sorry

end NUMINAMATH_CALUDE_exist_good_numbers_not_preserving_sum_of_digits_l4111_411128


namespace NUMINAMATH_CALUDE_original_index_is_12_l4111_411118

/-- Given an original sequence and a new sequence formed by inserting 3 numbers
    between every two adjacent terms of the original sequence, 
    this function returns the index in the original sequence that corresponds
    to the 49th term in the new sequence. -/
def original_index_of_49th_new_term : ℕ :=
  let x := (49 - 1) / 4
  x + 1

theorem original_index_is_12 : original_index_of_49th_new_term = 12 := by
  sorry

end NUMINAMATH_CALUDE_original_index_is_12_l4111_411118


namespace NUMINAMATH_CALUDE_triangle_side_length_l4111_411198

theorem triangle_side_length (a b c : ℝ) (C : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : C = π/3) :
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) → c = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4111_411198


namespace NUMINAMATH_CALUDE_complex_subtraction_l4111_411184

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 4 + 3*I) :
  a - 3*b = -7 - 12*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l4111_411184


namespace NUMINAMATH_CALUDE_expression_simplification_l4111_411188

theorem expression_simplification (a : ℝ) (h : a = 1 - Real.sqrt 3) :
  (1 - (2 * a - 1) / (a ^ 2)) / ((a - 1) / (a ^ 2)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4111_411188


namespace NUMINAMATH_CALUDE_max_container_volume_height_for_max_volume_l4111_411105

/-- Represents the volume of a rectangular container as a function of one side length --/
def containerVolume (x : ℝ) : ℝ := x * (x + 0.5) * (3.45 - x)

/-- The total length of the steel strip used for the container frame --/
def totalLength : ℝ := 14.8

/-- Theorem stating the maximum volume of the container --/
theorem max_container_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 3.45 ∧
  containerVolume x = 3.675 ∧
  ∀ (y : ℝ), y > 0 → y < 3.45 → containerVolume y ≤ containerVolume x :=
sorry

/-- Theorem stating the height that achieves the maximum volume --/
theorem height_for_max_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 3.45 ∧
  containerVolume x = 3.675 ∧
  (3.45 - x) = 2.45 :=
sorry

end NUMINAMATH_CALUDE_max_container_volume_height_for_max_volume_l4111_411105


namespace NUMINAMATH_CALUDE_triangle_circle_area_ratio_l4111_411107

theorem triangle_circle_area_ratio : 
  let a : ℝ := 8
  let b : ℝ := 15
  let c : ℝ := 17
  let s : ℝ := (a + b + c) / 2
  let triangle_area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let circle_radius : ℝ := c / 2
  let circle_area : ℝ := π * circle_radius^2
  let semicircle_area : ℝ := circle_area / 2
  let outside_triangle_area : ℝ := semicircle_area - triangle_area
  abs ((outside_triangle_area / semicircle_area) - 0.471) < 0.001 := by
sorry

end NUMINAMATH_CALUDE_triangle_circle_area_ratio_l4111_411107


namespace NUMINAMATH_CALUDE_apple_bags_theorem_l4111_411181

theorem apple_bags_theorem (A B C : ℕ) 
  (h1 : A + B = 11) 
  (h2 : B + C = 18) 
  (h3 : A + C = 19) : 
  A + B + C = 24 := by
sorry

end NUMINAMATH_CALUDE_apple_bags_theorem_l4111_411181


namespace NUMINAMATH_CALUDE_apartment_households_l4111_411121

/-- Represents the position and structure of an apartment building --/
structure ApartmentBuilding where
  houses_per_row : ℕ
  floors : ℕ
  households_per_house : ℕ

/-- Represents the position of Mijoo's house in the apartment building --/
structure MijooHousePosition where
  from_left : ℕ
  from_right : ℕ
  from_top : ℕ
  from_bottom : ℕ

/-- Calculates the total number of households in the apartment building --/
def total_households (building : ApartmentBuilding) : ℕ :=
  building.houses_per_row * building.floors * building.households_per_house

/-- Theorem stating the total number of households in the apartment building --/
theorem apartment_households 
  (building : ApartmentBuilding)
  (mijoo_position : MijooHousePosition)
  (h1 : mijoo_position.from_left = 1)
  (h2 : mijoo_position.from_right = 7)
  (h3 : mijoo_position.from_top = 2)
  (h4 : mijoo_position.from_bottom = 4)
  (h5 : building.houses_per_row = mijoo_position.from_left + mijoo_position.from_right - 1)
  (h6 : building.floors = mijoo_position.from_top + mijoo_position.from_bottom - 1)
  (h7 : building.households_per_house = 3) :
  total_households building = 105 := by
  sorry

#eval total_households { houses_per_row := 7, floors := 5, households_per_house := 3 }

end NUMINAMATH_CALUDE_apartment_households_l4111_411121


namespace NUMINAMATH_CALUDE_yellow_balls_count_l4111_411109

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  white = 22 ∧
  green = 18 ∧
  red = 3 ∧
  purple = 1 ∧
  prob = 95 / 100 ∧
  (white + green + (total - white - green - red - purple)) / total = prob →
  total - white - green - red - purple = 17 := by
    sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l4111_411109


namespace NUMINAMATH_CALUDE_fraction_simplification_l4111_411159

theorem fraction_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 - 1 / (x + 2)) / ((x^2 - 1) / (x + 2)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4111_411159


namespace NUMINAMATH_CALUDE_second_number_proof_l4111_411190

theorem second_number_proof (x : ℤ) (h1 : x + (x + 4) = 56) : x + 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l4111_411190


namespace NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l4111_411144

/-- A fair coin is tossed eight times. -/
def coin_tosses : ℕ := 8

/-- The coin is fair, meaning the probability of heads is 1/2. -/
def fair_coin_prob : ℚ := 1/2

/-- The number of heads we're looking for. -/
def target_heads : ℕ := 3

/-- The probability of getting exactly three heads in eight tosses of a fair coin. -/
theorem three_heads_in_eight_tosses : 
  (Nat.choose coin_tosses target_heads : ℚ) * fair_coin_prob^target_heads * (1 - fair_coin_prob)^(coin_tosses - target_heads) = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l4111_411144


namespace NUMINAMATH_CALUDE_green_ball_theorem_l4111_411166

/-- Represents the price and quantity information for green balls --/
structure GreenBallInfo where
  saltyCost : ℚ
  saltyQuantity : ℕ
  duckCost : ℚ
  duckQuantity : ℕ

/-- Represents a purchase plan --/
structure PurchasePlan where
  saltyQuantity : ℕ
  duckQuantity : ℕ

/-- Represents an exchange method --/
structure ExchangeMethod where
  coupons : ℕ
  saltyCoupons : ℕ
  duckCoupons : ℕ

/-- Main theorem about green ball prices, purchase plans, and exchange methods --/
theorem green_ball_theorem (info : GreenBallInfo) 
  (h1 : info.duckCost = 2 * info.saltyCost)
  (h2 : info.duckCost * info.duckQuantity = 40)
  (h3 : info.saltyCost * info.saltyQuantity = 30)
  (h4 : info.saltyQuantity = info.duckQuantity + 4)
  (h5 : ∀ plan : PurchasePlan, 
    plan.saltyQuantity ≥ 20 ∧ 
    plan.duckQuantity ≥ 20 ∧ 
    plan.saltyQuantity % 10 = 0 ∧
    info.saltyCost * plan.saltyQuantity + info.duckCost * plan.duckQuantity = 200)
  (h6 : ∀ method : ExchangeMethod,
    1 < method.coupons ∧ 
    method.coupons < 10 ∧
    method.saltyCoupons + method.duckCoupons = method.coupons) :
  (info.saltyCost = 5/2 ∧ info.duckCost = 5) ∧
  (∃ (plans : List PurchasePlan), plans = 
    [(PurchasePlan.mk 20 30), (PurchasePlan.mk 30 25), (PurchasePlan.mk 40 20)]) ∧
  (∃ (methods : List ExchangeMethod), methods = 
    [(ExchangeMethod.mk 5 5 0), (ExchangeMethod.mk 5 0 5), 
     (ExchangeMethod.mk 8 6 2), (ExchangeMethod.mk 8 1 7)]) :=
by sorry

end NUMINAMATH_CALUDE_green_ball_theorem_l4111_411166


namespace NUMINAMATH_CALUDE_triangle_inequality_sign_l4111_411170

/-- Given a triangle ABC with sides a, b, c (a ≤ b ≤ c), circumradius R, and inradius r,
    the sign of a + b - 2R - 2r depends on angle C as follows:
    1. If π/3 ≤ C < π/2, then a + b - 2R - 2r > 0
    2. If C = π/2, then a + b - 2R - 2r = 0
    3. If π/2 < C < π, then a + b - 2R - 2r < 0 -/
theorem triangle_inequality_sign (a b c R r : ℝ) (C : ℝ) :
  a ≤ b ∧ b ≤ c ∧ 0 < a ∧ 0 < R ∧ 0 < r ∧ 0 < C ∧ C < π →
  (π/3 ≤ C ∧ C < π/2 → a + b - 2*R - 2*r > 0) ∧
  (C = π/2 → a + b - 2*R - 2*r = 0) ∧
  (π/2 < C ∧ C < π → a + b - 2*R - 2*r < 0) := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequality_sign_l4111_411170


namespace NUMINAMATH_CALUDE_weight_replacement_l4111_411173

theorem weight_replacement (initial_count : ℕ) (weight_increase : ℝ) (new_person_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 6 →
  new_person_weight = 88 →
  ∃ (replaced_weight : ℝ),
    replaced_weight = 40 ∧
    (initial_count : ℝ) * weight_increase = new_person_weight - replaced_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l4111_411173


namespace NUMINAMATH_CALUDE_max_ratio_of_two_digit_integers_with_mean_45_l4111_411124

theorem max_ratio_of_two_digit_integers_with_mean_45 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y : ℚ) / 2 = 45 →
  ∀ z : ℚ,
  (z : ℚ) = x / y →
  z ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_of_two_digit_integers_with_mean_45_l4111_411124


namespace NUMINAMATH_CALUDE_max_bc_value_l4111_411120

theorem max_bc_value (a b c : ℂ) 
  (h : ∀ z : ℂ, Complex.abs z ≤ 1 → Complex.abs (a * z^2 + b * z + c) ≤ 1) : 
  Complex.abs (b * c) ≤ (3 * Real.sqrt 3) / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_bc_value_l4111_411120


namespace NUMINAMATH_CALUDE_gcf_of_50_and_75_l4111_411157

theorem gcf_of_50_and_75 : Nat.gcd 50 75 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_50_and_75_l4111_411157


namespace NUMINAMATH_CALUDE_power_three_times_three_l4111_411114

theorem power_three_times_three (x : ℝ) : x^3 * x^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_three_times_three_l4111_411114


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l4111_411102

/-- The probability of a randomly selected point from a square with side length 6
    being inside or on a circle with radius 2 centered at the center of the square -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 6 →
  circle_radius = 2 →
  (circle_radius^2 * Real.pi) / square_side^2 = Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l4111_411102


namespace NUMINAMATH_CALUDE_fraction_simplification_l4111_411180

theorem fraction_simplification (x y z : ℝ) :
  (16 * x^4 * z^4 - x^4 * y^16 - 64 * x^4 * y^2 * z^4 + 4 * x^4 * y^18 + 32 * x^2 * y * z^4 - 2 * x^2 * y^17 + 16 * y^2 * z^4 - y^18) /
  ((2 * x^2 * y - x^2 - y) * (8 * z^3 + 2 * y^8 * z + 4 * y^4 * z^2 + y^12) * (2 * z - y^4)) =
  -(2 * x^2 * y + x^2 + y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4111_411180


namespace NUMINAMATH_CALUDE_bat_survey_result_l4111_411101

theorem bat_survey_result :
  ∀ (total : ℕ) 
    (blind_believers : ℕ) 
    (ebola_believers : ℕ),
  (blind_believers : ℚ) = 0.750 * total →
  (ebola_believers : ℚ) = 0.523 * blind_believers →
  ebola_believers = 49 →
  total = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_bat_survey_result_l4111_411101


namespace NUMINAMATH_CALUDE_fraction_addition_l4111_411150

theorem fraction_addition : (18 : ℚ) / 42 + 2 / 9 = 41 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l4111_411150


namespace NUMINAMATH_CALUDE_fraction_simplification_l4111_411133

theorem fraction_simplification :
  (1 / 20 : ℚ) - (1 / 21 : ℚ) + (1 / (20 * 21) : ℚ) = (1 / 210 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4111_411133


namespace NUMINAMATH_CALUDE_norbs_age_l4111_411153

def guesses : List Nat := [25, 29, 33, 35, 37, 39, 42, 45, 48, 50]

def is_prime (n : Nat) : Prop := Nat.Prime n

def half_guesses_too_low (age : Nat) : Prop :=
  (guesses.filter (· < age)).length = guesses.length / 2

def two_guesses_off_by_one (age : Nat) : Prop :=
  (guesses.filter (λ g => g = age - 1 ∨ g = age + 1)).length = 2

theorem norbs_age : 
  ∃ (age : Nat), age = 47 ∧ 
    is_prime age ∧ 
    half_guesses_too_low age ∧ 
    two_guesses_off_by_one age ∧
    ∀ (n : Nat), n ≠ 47 → 
      ¬(is_prime n ∧ half_guesses_too_low n ∧ two_guesses_off_by_one n) :=
by sorry

end NUMINAMATH_CALUDE_norbs_age_l4111_411153


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_even_sides_l4111_411142

/-- A triangle with sides 2a, 2a+2, and 2a+4 is a right triangle if and only if a = 3 -/
theorem right_triangle_consecutive_even_sides (a : ℕ) : 
  (2*a)^2 + (2*a+2)^2 = (2*a+4)^2 ↔ a = 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_even_sides_l4111_411142


namespace NUMINAMATH_CALUDE_black_balls_count_l4111_411131

theorem black_balls_count (red white : ℕ) (p : ℚ) (black : ℕ) : 
  red = 3 → 
  white = 5 → 
  p = 1/4 → 
  (white : ℚ) / ((red : ℚ) + (white : ℚ) + (black : ℚ)) = p → 
  black = 12 := by
sorry

end NUMINAMATH_CALUDE_black_balls_count_l4111_411131


namespace NUMINAMATH_CALUDE_circle_area_ratio_l4111_411132

/-- Given three circles S, R, and T, where R's diameter is 20% of S's diameter,
    and T's diameter is 40% of R's diameter, prove that the combined area of
    R and T is 4.64% of the area of S. -/
theorem circle_area_ratio (S R T : ℝ) (hR : R = 0.2 * S) (hT : T = 0.4 * R) :
  (π * (R / 2)^2 + π * (T / 2)^2) / (π * (S / 2)^2) = 0.0464 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l4111_411132


namespace NUMINAMATH_CALUDE_eight_sided_dice_divisible_by_four_probability_l4111_411104

theorem eight_sided_dice_divisible_by_four_probability : 
  let dice_outcomes : Finset ℕ := Finset.range 8
  let divisible_by_four : Finset ℕ := {4, 8}
  let total_outcomes : ℕ := dice_outcomes.card * dice_outcomes.card
  let favorable_outcomes : ℕ := divisible_by_four.card * divisible_by_four.card
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_eight_sided_dice_divisible_by_four_probability_l4111_411104


namespace NUMINAMATH_CALUDE_sum_of_m_and_n_is_zero_l4111_411106

theorem sum_of_m_and_n_is_zero 
  (h1 : ∃ p : ℝ, m * n + p^2 + 4 = 0) 
  (h2 : m - n = 4) : 
  m + n = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_m_and_n_is_zero_l4111_411106


namespace NUMINAMATH_CALUDE_fraction_to_fourth_power_l4111_411185

theorem fraction_to_fourth_power (a b : ℝ) (hb : b ≠ 0) :
  (2 * a / b) ^ 4 = 16 * a ^ 4 / b ^ 4 := by sorry

end NUMINAMATH_CALUDE_fraction_to_fourth_power_l4111_411185


namespace NUMINAMATH_CALUDE_school_attendance_l4111_411134

/-- Represents the attendance schedule for a group of students -/
inductive AttendanceSchedule
  | A -- Attends Mondays and Wednesdays
  | B -- Attends Tuesdays and Thursdays
  | C -- Attends Fridays

/-- Represents a day of the week -/
inductive WeekDay
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- The school attendance problem -/
theorem school_attendance
  (total_students : Nat)
  (home_learning_percentage : Rat)
  (group_schedules : List AttendanceSchedule)
  (h1 : total_students = 1000)
  (h2 : home_learning_percentage = 60 / 100)
  (h3 : group_schedules = [AttendanceSchedule.A, AttendanceSchedule.B, AttendanceSchedule.C]) :
  ∃ (attendance : WeekDay → Nat),
    attendance WeekDay.Monday = 133 ∧
    attendance WeekDay.Tuesday = 133 ∧
    attendance WeekDay.Wednesday = 133 ∧
    attendance WeekDay.Thursday = 133 ∧
    attendance WeekDay.Friday = 134 := by
  sorry

end NUMINAMATH_CALUDE_school_attendance_l4111_411134


namespace NUMINAMATH_CALUDE_ellipse_max_y_coordinate_l4111_411122

theorem ellipse_max_y_coordinate :
  let ellipse := {(x, y) : ℝ × ℝ | (x^2 / 49) + ((y - 3)^2 / 25) = 1}
  ∃ (y_max : ℝ), y_max = 8 ∧ ∀ (x y : ℝ), (x, y) ∈ ellipse → y ≤ y_max :=
by sorry

end NUMINAMATH_CALUDE_ellipse_max_y_coordinate_l4111_411122


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_l4111_411108

theorem polynomial_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 3001*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 118 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_l4111_411108


namespace NUMINAMATH_CALUDE_right_triangle_properties_l4111_411194

/-- Properties of a specific right triangle -/
theorem right_triangle_properties :
  ∀ (a b c : ℝ),
  a = 24 →
  b = 2 * a + 10 →
  c^2 = a^2 + b^2 →
  (1/2 * a * b = 696) ∧ (c = Real.sqrt 3940) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l4111_411194


namespace NUMINAMATH_CALUDE_exists_n_sum_of_digits_square_eq_2002_l4111_411148

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a positive integer n such that the sum of the digits of n^2 is 2002 -/
theorem exists_n_sum_of_digits_square_eq_2002 : ∃ n : ℕ+, sumOfDigits (n^2) = 2002 := by sorry

end NUMINAMATH_CALUDE_exists_n_sum_of_digits_square_eq_2002_l4111_411148


namespace NUMINAMATH_CALUDE_squirrel_is_red_l4111_411171

-- Define the color of the squirrel
inductive SquirrelColor
  | Red
  | Gray

-- Define the state of a hollow
inductive HollowState
  | Empty
  | HasNuts

-- Define the structure for the two hollows
structure Hollows :=
  (first : HollowState)
  (second : HollowState)

-- Define the statements made by the squirrel
def statement1 (h : Hollows) : Prop :=
  h.first = HollowState.Empty

def statement2 (h : Hollows) : Prop :=
  h.first = HollowState.HasNuts ∨ h.second = HollowState.HasNuts

-- Define the truthfulness of the squirrel based on its color
def isTruthful (c : SquirrelColor) : Prop :=
  match c with
  | SquirrelColor.Red => True
  | SquirrelColor.Gray => False

-- Theorem: The squirrel must be red
theorem squirrel_is_red (h : Hollows) :
  (isTruthful SquirrelColor.Red → statement1 h ∧ statement2 h) ∧
  (isTruthful SquirrelColor.Gray → ¬(statement1 h) ∧ ¬(statement2 h)) →
  ∃ (h : Hollows), statement1 h ∧ statement2 h →
  SquirrelColor.Red = SquirrelColor.Red :=
by sorry

end NUMINAMATH_CALUDE_squirrel_is_red_l4111_411171


namespace NUMINAMATH_CALUDE_product_inequality_l4111_411111

theorem product_inequality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) (h₅ : a₅ > 1) : 
  (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) ≤ 16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l4111_411111


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l4111_411151

theorem arithmetic_mean_problem :
  let numbers : Finset ℚ := {7/8, 9/10, 4/5, 17/20}
  17/20 ∈ numbers ∧ 
  9/10 ∈ numbers ∧ 
  4/5 ∈ numbers ∧
  (9/10 + 4/5) / 2 = 17/20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l4111_411151


namespace NUMINAMATH_CALUDE_max_ratio_concentric_circles_polyline_l4111_411197

/-- The maximum common ratio for concentric circles allowing a closed polyline -/
theorem max_ratio_concentric_circles_polyline :
  ∃ (q : ℝ), q = (Real.sqrt 5 + 1) / 2 ∧
  ∀ (r : ℝ) (A : Fin 5 → ℝ × ℝ),
  (∀ i : Fin 5, ‖A i‖ = r * q ^ i.val) →
  (∀ i : Fin 5, ‖A i - A (i + 1)‖ = ‖A 0 - A 1‖) →
  (A 0 = A 4) →
  ∀ q' > q, ¬∃ (A' : Fin 5 → ℝ × ℝ),
    (∀ i : Fin 5, ‖A' i‖ = r * q' ^ i.val) ∧
    (∀ i : Fin 5, ‖A' i - A' (i + 1)‖ = ‖A' 0 - A' 1‖) ∧
    (A' 0 = A' 4) :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_concentric_circles_polyline_l4111_411197


namespace NUMINAMATH_CALUDE_ceiling_minus_value_l4111_411179

theorem ceiling_minus_value (x : ℝ) (h : ⌈(2 * x)⌉ - ⌊(2 * x)⌋ = 0) : 
  ⌈(2 * x)⌉ - (2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_value_l4111_411179


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l4111_411155

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ + 3 = 0 → x₂^2 - 4*x₂ + 3 = 0 → x₁ + x₂ - 2*x₁*x₂ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l4111_411155


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l4111_411199

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (x₁ y₁ x₂ y₂ : ℝ), y₁ = m₁ * x₁ ∧ y₂ = m₂ * x₂ ∧ (y₂ - y₁) * (x₂ - x₁) = 0)

/-- The slope of a line ax + y + c = 0 is -a -/
axiom line_slope (a c : ℝ) : ∃ (m : ℝ), m = -a ∧ ∀ (x y : ℝ), a * x + y + c = 0 → y = m * x - c

theorem perpendicular_lines_b_value :
  ∀ (b : ℝ), 
  (∀ (x y : ℝ), 3 * x + y - 5 = 0 → bx + y + 2 = 0 → 
    ∃ (m₁ m₂ : ℝ), (m₁ * m₂ = -1 ∧ 
      (∀ (x₁ y₁ : ℝ), 3 * x₁ + y₁ - 5 = 0 → y₁ = m₁ * x₁ + 5) ∧
      (∀ (x₂ y₂ : ℝ), b * x₂ + y₂ + 2 = 0 → y₂ = m₂ * x₂ - 2))) →
  b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l4111_411199


namespace NUMINAMATH_CALUDE_least_possible_z_l4111_411160

theorem least_possible_z (x y z : ℤ) : 
  (∃ k : ℤ, x = 2 * k) →  -- x is even
  (∃ m n : ℤ, y = 2 * m + 1 ∧ z = 2 * n + 1) →  -- y and z are odd
  y - x > 5 →
  (∀ w : ℤ, w - x ≥ 9 → z ≤ w) →  -- least possible value of z - x is 9
  z ≥ 11 ∧ (∀ v : ℤ, v ≥ 11 → z ≤ v) :=  -- z is at least 11 and is the least such value
by sorry

end NUMINAMATH_CALUDE_least_possible_z_l4111_411160


namespace NUMINAMATH_CALUDE_range_of_a_l4111_411146

def p (x : ℝ) : Prop := |4 - x| ≤ 6

def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≥ 0

def sufficient_not_necessary (P Q : ℝ → Prop) : Prop :=
  (∀ x, ¬(P x) → Q x) ∧ ∃ x, Q x ∧ P x

theorem range_of_a :
  ∀ a : ℝ, 
    (a > 0 ∧ 
     sufficient_not_necessary p (q · a)) →
    (0 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4111_411146


namespace NUMINAMATH_CALUDE_evaluate_expression_l4111_411129

theorem evaluate_expression : (24^18) / (72^9) = 8^9 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4111_411129


namespace NUMINAMATH_CALUDE_weekend_to_weekday_practice_ratio_l4111_411152

/-- Given Daniel's basketball practice schedule, prove the ratio of weekend to weekday practice time -/
theorem weekend_to_weekday_practice_ratio :
  let weekday_daily_practice : ℕ := 15
  let weekday_count : ℕ := 5
  let total_weekly_practice : ℕ := 135
  let weekday_practice := weekday_daily_practice * weekday_count
  let weekend_practice := total_weekly_practice - weekday_practice
  (weekend_practice : ℚ) / weekday_practice = 4 / 5 := by
sorry


end NUMINAMATH_CALUDE_weekend_to_weekday_practice_ratio_l4111_411152


namespace NUMINAMATH_CALUDE_trees_died_l4111_411117

/-- Proof that 15 trees died in the park --/
theorem trees_died (initial : ℕ) (cut : ℕ) (remaining : ℕ) (died : ℕ) : 
  initial = 86 → cut = 23 → remaining = 48 → died = initial - cut - remaining → died = 15 := by
  sorry

end NUMINAMATH_CALUDE_trees_died_l4111_411117


namespace NUMINAMATH_CALUDE_third_square_perimeter_l4111_411177

/-- Given two squares with perimeters 60 cm and 48 cm, prove that a third square
    whose area is equal to the difference of the areas of the first two squares
    has a perimeter of 36 cm. -/
theorem third_square_perimeter (square1 square2 square3 : ℝ → ℝ) :
  (∀ s, square1 s = s^2) →
  (∀ s, square2 s = s^2) →
  (∀ s, square3 s = s^2) →
  (4 * Real.sqrt (square1 (60 / 4))) = 60 →
  (4 * Real.sqrt (square2 (48 / 4))) = 48 →
  square3 (Real.sqrt (square1 (60 / 4) - square2 (48 / 4))) =
    square1 (60 / 4) - square2 (48 / 4) →
  (4 * Real.sqrt (square3 (Real.sqrt (square1 (60 / 4) - square2 (48 / 4))))) = 36 :=
by sorry

end NUMINAMATH_CALUDE_third_square_perimeter_l4111_411177
