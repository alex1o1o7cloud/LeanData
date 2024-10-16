import Mathlib

namespace NUMINAMATH_CALUDE_greatest_t_value_l551_55121

theorem greatest_t_value : ∃ (t : ℝ), 
  (∀ (s : ℝ), (s^2 - s - 40) / (s - 8) = 5 / (s + 5) → s ≤ t) ∧
  (t^2 - t - 40) / (t - 8) = 5 / (t + 5) ∧
  t = -2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_t_value_l551_55121


namespace NUMINAMATH_CALUDE_inequality_proof_l551_55133

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a - b + c) * (1/a - 1/b + 1/c) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l551_55133


namespace NUMINAMATH_CALUDE_eliminate_denominators_l551_55168

theorem eliminate_denominators (x : ℚ) : 
  (2*x - 1) / 2 = 1 - (3 - x) / 3 ↔ 3*(2*x - 1) = 6 - 2*(3 - x) := by
sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l551_55168


namespace NUMINAMATH_CALUDE_factorial_division_l551_55194

theorem factorial_division : Nat.factorial 5 / Nat.factorial (5 - 3) = 60 := by sorry

end NUMINAMATH_CALUDE_factorial_division_l551_55194


namespace NUMINAMATH_CALUDE_batsman_average_increase_17_innings_l551_55147

def batsman_average_increase (total_innings : ℕ) (last_innings_score : ℕ) (final_average : ℚ) : ℚ :=
  let previous_total := (total_innings - 1) * (total_innings * final_average - last_innings_score) / total_innings
  let previous_average := previous_total / (total_innings - 1)
  final_average - previous_average

theorem batsman_average_increase_17_innings 
  (h1 : total_innings = 17)
  (h2 : last_innings_score = 85)
  (h3 : final_average = 37) :
  batsman_average_increase total_innings last_innings_score final_average = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_17_innings_l551_55147


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l551_55175

theorem triangle_sine_inequality (A B C : Real) (h_triangle : A + B + C = π) :
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l551_55175


namespace NUMINAMATH_CALUDE_part_one_part_two_l551_55125

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

-- Part 1
theorem part_one :
  let a := 2
  (∀ x ∈ Set.Icc (-1) 3, f a x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-1) 3, f a x = -2) ∧
  (∀ x ∈ Set.Icc (-1) 3, f a x ≤ 14) ∧
  (∃ x ∈ Set.Icc (-1) 3, f a x = 14) :=
sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, f a x ≥ 3) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 3) ↔
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l551_55125


namespace NUMINAMATH_CALUDE_optimal_station_is_75km_l551_55197

/-- Represents a petrol station with its distance from a given point --/
structure PetrolStation :=
  (distance : ℝ)

/-- Represents a car with its fuel consumption rate --/
structure Car :=
  (consumption : ℝ)  -- litres per km

/-- Represents a journey with various parameters --/
structure Journey :=
  (totalDistance : ℝ)
  (initialFuel : ℝ)
  (initialDriven : ℝ)
  (stations : List PetrolStation)
  (tankCapacity : ℝ)

def Journey.optimalStation (j : Journey) (c : Car) : Option PetrolStation :=
  sorry

theorem optimal_station_is_75km 
  (j : Journey)
  (c : Car)
  (h1 : j.totalDistance = 520)
  (h2 : j.initialFuel = 14)
  (h3 : c.consumption = 0.1)
  (h4 : j.initialDriven = 55)
  (h5 : j.stations = [
    { distance := 35 },
    { distance := 45 },
    { distance := 55 },
    { distance := 75 },
    { distance := 95 }
  ])
  (h6 : j.tankCapacity = 40) :
  (Journey.optimalStation j c).map PetrolStation.distance = some 75 := by
  sorry

end NUMINAMATH_CALUDE_optimal_station_is_75km_l551_55197


namespace NUMINAMATH_CALUDE_complex_power_equality_l551_55161

theorem complex_power_equality : (((1 - Complex.I) / Real.sqrt 2) ^ 44 : ℂ) = -1 := by sorry

end NUMINAMATH_CALUDE_complex_power_equality_l551_55161


namespace NUMINAMATH_CALUDE_min_value_expression_l551_55117

theorem min_value_expression (x : ℝ) : 
  (∃ (m : ℝ), ∀ (y : ℝ), (15 - y) * (13 - y) * (15 + y) * (13 + y) + 200 * y^2 ≥ m) ∧ 
  (∃ (z : ℝ), (15 - z) * (13 - z) * (15 + z) * (13 + z) + 200 * z^2 = 33) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l551_55117


namespace NUMINAMATH_CALUDE_application_methods_eq_sixteen_l551_55190

/-- The number of universities -/
def total_universities : ℕ := 6

/-- The number of universities to be chosen -/
def universities_to_choose : ℕ := 3

/-- The number of universities with overlapping schedules -/
def overlapping_universities : ℕ := 2

/-- The function to calculate the number of different application methods -/
def application_methods : ℕ := sorry

/-- Theorem stating that the number of different application methods is 16 -/
theorem application_methods_eq_sixteen :
  application_methods = 16 := by sorry

end NUMINAMATH_CALUDE_application_methods_eq_sixteen_l551_55190


namespace NUMINAMATH_CALUDE_probability_four_white_balls_l551_55155

/-- The number of white balls in the box -/
def white_balls : ℕ := 7

/-- The number of black balls in the box -/
def black_balls : ℕ := 8

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 5

/-- The number of white balls we want to draw -/
def target_white : ℕ := 4

/-- The number of black balls we want to draw -/
def target_black : ℕ := drawn_balls - target_white

theorem probability_four_white_balls : 
  (Nat.choose white_balls target_white * Nat.choose black_balls target_black : ℚ) / 
  Nat.choose total_balls drawn_balls = 280 / 3003 := by
sorry

end NUMINAMATH_CALUDE_probability_four_white_balls_l551_55155


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l551_55184

theorem quadratic_equation_problem (k : ℝ) : 
  (∀ x, 4 * x^2 - 6 * x * Real.sqrt 3 + k = 0 → 
    (6 * Real.sqrt 3)^2 - 4 * 4 * k = 18) → 
  k = 45/8 ∧ ∃ x y, x ≠ y ∧ 4 * x^2 - 6 * x * Real.sqrt 3 + k = 0 ∧ 
                           4 * y^2 - 6 * y * Real.sqrt 3 + k = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l551_55184


namespace NUMINAMATH_CALUDE_hotel_stay_duration_l551_55139

theorem hotel_stay_duration (cost_per_night_per_person : ℕ) (num_people : ℕ) (total_cost : ℕ) : 
  cost_per_night_per_person = 40 →
  num_people = 3 →
  total_cost = 360 →
  total_cost = cost_per_night_per_person * num_people * 3 :=
by
  sorry

end NUMINAMATH_CALUDE_hotel_stay_duration_l551_55139


namespace NUMINAMATH_CALUDE_equation_solutions_l551_55119

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 4 ↔ x = 4 ∨ x = 0) ∧
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 ↔ x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l551_55119


namespace NUMINAMATH_CALUDE_first_product_of_98_l551_55103

/-- The first product of the digits of a two-digit number -/
def first_digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- Theorem: The first product of the digits of 98 is 72 -/
theorem first_product_of_98 : first_digit_product 98 = 72 := by
  sorry

end NUMINAMATH_CALUDE_first_product_of_98_l551_55103


namespace NUMINAMATH_CALUDE_equation_solution_l551_55130

theorem equation_solution : 
  ∃! y : ℝ, 7 * (2 * y - 3) + 5 = -3 * (4 - 5 * y) ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l551_55130


namespace NUMINAMATH_CALUDE_product_of_roots_l551_55127

theorem product_of_roots (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ - 1 = 0) → (x₂^2 + x₂ - 1 = 0) → x₁ * x₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l551_55127


namespace NUMINAMATH_CALUDE_square_area_increase_l551_55149

/-- Given a square with initial side length 4, if the side length increases by x
    and the area increases by y, then y = x^2 + 8x -/
theorem square_area_increase (x y : ℝ) : 
  (4 + x)^2 - 4^2 = y → y = x^2 + 8*x := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l551_55149


namespace NUMINAMATH_CALUDE_stevens_falls_l551_55143

theorem stevens_falls (steven_falls : ℕ) (stephanie_falls : ℕ) (sonya_falls : ℕ) 
  (h1 : stephanie_falls = steven_falls + 13)
  (h2 : sonya_falls = 6)
  (h3 : sonya_falls = stephanie_falls / 2 - 2) : 
  steven_falls = 3 := by
  sorry

end NUMINAMATH_CALUDE_stevens_falls_l551_55143


namespace NUMINAMATH_CALUDE_midpoint_y_coordinate_l551_55172

theorem midpoint_y_coordinate (a : Real) (h1 : 0 < a) (h2 : a < π / 2) :
  let f := λ x : Real => Real.sin x
  let g := λ x : Real => Real.cos x
  let M := (a, f a)
  let N := (a, g a)
  abs (f a - g a) = 1/5 →
  (f a + g a) / 2 = 7/10 := by
sorry

end NUMINAMATH_CALUDE_midpoint_y_coordinate_l551_55172


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l551_55122

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 20) :
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 20 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l551_55122


namespace NUMINAMATH_CALUDE_vector_norm_condition_l551_55120

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given non-zero vectors a and b, a = -2b is a sufficient but not necessary condition
    for |a| - |b| = |a + b| --/
theorem vector_norm_condition (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a = -2 • b → ‖a‖ - ‖b‖ = ‖a + b‖) ∧
  ¬(‖a‖ - ‖b‖ = ‖a + b‖ → a = -2 • b) :=
sorry

end NUMINAMATH_CALUDE_vector_norm_condition_l551_55120


namespace NUMINAMATH_CALUDE_translation_theorem_l551_55196

def original_function (x : ℝ) : ℝ := (x - 2)^2 + 1

def translated_function (x : ℝ) : ℝ := original_function (x + 2) - 2

theorem translation_theorem :
  ∀ x : ℝ, translated_function x = x^2 - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l551_55196


namespace NUMINAMATH_CALUDE_fraction_simplification_l551_55113

theorem fraction_simplification (a b : ℚ) (ha : a = 5) (hb : b = 4) :
  (1 / b) / (1 / a) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l551_55113


namespace NUMINAMATH_CALUDE_polynomial_factorization_l551_55188

theorem polynomial_factorization (x y : ℝ) : 
  x^2 - y^2 - 2*x - 4*y - 3 = (x+y+1)*(x-y-3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l551_55188


namespace NUMINAMATH_CALUDE_pears_cost_l551_55170

theorem pears_cost (initial_amount : ℕ) (banana_cost : ℕ) (banana_packs : ℕ) (asparagus_cost : ℕ) (chicken_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 55 →
  banana_cost = 4 →
  banana_packs = 2 →
  asparagus_cost = 6 →
  chicken_cost = 11 →
  remaining_amount = 28 →
  initial_amount - (banana_cost * banana_packs + asparagus_cost + chicken_cost + remaining_amount) = 2 :=
by
  sorry

#check pears_cost

end NUMINAMATH_CALUDE_pears_cost_l551_55170


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l551_55187

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_prod : a 7 * a 9 = 4) 
  (h_a4 : a 4 = 1) : 
  a 12 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l551_55187


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l551_55152

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hsum : x + y = 8) (hprod : x * y = 12) : 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 8 → a * b = 12 → 1/x + 1/y ≤ 1/a + 1/b) ∧ 
  1/x + 1/y = 2/3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l551_55152


namespace NUMINAMATH_CALUDE_count_valid_a_theorem_l551_55131

/-- Count of valid 'a' values satisfying the conditions --/
def count_valid_a : ℕ := 548

/-- Theorem stating the count of valid 'a' values --/
theorem count_valid_a_theorem :
  (∃ (S : Finset ℕ), 
    S.card = count_valid_a ∧
    (∀ a ∈ S, ∃ b c d : ℕ, 
      a > b ∧ b > c ∧ c > d ∧
      a + b + c + d = 2200 ∧
      a^2 - b^2 + c^2 - d^2 = 2200) ∧
    (∀ a : ℕ, 
      (∃ b c d : ℕ, 
        a > b ∧ b > c ∧ c > d ∧
        a + b + c + d = 2200 ∧
        a^2 - b^2 + c^2 - d^2 = 2200) →
      a ∈ S)) :=
by sorry

end NUMINAMATH_CALUDE_count_valid_a_theorem_l551_55131


namespace NUMINAMATH_CALUDE_optimal_soap_cost_l551_55102

/-- Represents the discount percentage based on the number of bars purchased -/
def discount (bars : ℕ) : ℚ :=
  if bars ≥ 8 then 15/100
  else if bars ≥ 6 then 10/100
  else if bars ≥ 4 then 5/100
  else 0

/-- Calculates the cost of soap for a year -/
def soap_cost (price_per_bar : ℚ) (months_per_bar : ℕ) (months_in_year : ℕ) : ℚ :=
  let bars_needed := months_in_year / months_per_bar
  let total_cost := price_per_bar * bars_needed
  total_cost * (1 - discount bars_needed)

theorem optimal_soap_cost :
  soap_cost 8 2 12 = 432/10 :=
sorry

end NUMINAMATH_CALUDE_optimal_soap_cost_l551_55102


namespace NUMINAMATH_CALUDE_variation_problem_l551_55136

theorem variation_problem (R S T : ℚ) (c : ℚ) : 
  (∀ R S T, R = c * S / T) →  -- R varies directly as S and inversely as T
  (2 = c * 4 / (1/2)) →       -- When R = 2, T = 1/2, S = 4
  (8 = c * S / (1/3)) →       -- When R = 8 and T = 1/3
  S = 32/3 := by
sorry

end NUMINAMATH_CALUDE_variation_problem_l551_55136


namespace NUMINAMATH_CALUDE_cylinder_height_ratio_l551_55115

/-- 
Given two right circular cylinders V and B, where:
- The radius of V is twice the radius of B
- It costs $4 to fill half of B
- It costs $16 to fill V completely
This theorem proves that the ratio of the height of V to the height of B is 1:2
-/
theorem cylinder_height_ratio 
  (r_B : ℝ) -- radius of cylinder B
  (h_B : ℝ) -- height of cylinder B
  (h_V : ℝ) -- height of cylinder V
  (cost_half_B : ℝ) -- cost to fill half of cylinder B
  (cost_full_V : ℝ) -- cost to fill cylinder V completely
  (h_radius : r_B > 0) -- radius of B is positive
  (h_cost_half_B : cost_half_B = 4) -- cost to fill half of B is $4
  (h_cost_full_V : cost_full_V = 16) -- cost to fill V completely is $16
  : h_V / h_B = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_ratio_l551_55115


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l551_55129

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 + 10*x = 56 ↔ x = 4 ∨ x = -14) ∧
  (∀ x : ℝ, 4*x^2 + 48 = 32*x ↔ x = 6 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 + 20 = 12*x ↔ x = 10 ∨ x = 2) ∧
  (∀ x : ℝ, 3*x^2 - 36 = 32*x - x^2 ↔ x = 9 ∨ x = -1) ∧
  (∀ x : ℝ, x^2 + 8*x = 20) ∧
  (∀ x : ℝ, 3*x^2 = 12*x + 63) ∧
  (∀ x : ℝ, x^2 + 16 = 8*x) ∧
  (∀ x : ℝ, 6*x^2 + 12*x = 90) ∧
  (∀ x : ℝ, (1/2)*x^2 + x = 7.5) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l551_55129


namespace NUMINAMATH_CALUDE_least_n_with_1987_zeros_l551_55126

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The least natural number n such that n! ends in exactly 1987 zeros -/
theorem least_n_with_1987_zeros : ∃ (n : ℕ), trailingZeros n = 1987 ∧ ∀ m < n, trailingZeros m < 1987 :=
  sorry

end NUMINAMATH_CALUDE_least_n_with_1987_zeros_l551_55126


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l551_55138

/-- The number formed by concatenating digits a, 7, 1, 9 in that order -/
def number (a : ℕ) : ℕ := a * 1000 + 719

/-- The alternating sum of digits used in the divisibility rule for 11 -/
def alternating_sum (a : ℕ) : ℤ := a - 7 + 1 - 9

theorem divisible_by_eleven (a : ℕ) : 
  (0 ≤ a ∧ a ≤ 9) → (number a % 11 = 0 ↔ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l551_55138


namespace NUMINAMATH_CALUDE_frog_jump_difference_l551_55163

/-- The frog's jump distance in inches -/
def frog_jump : ℕ := 39

/-- The grasshopper's jump distance in inches -/
def grasshopper_jump : ℕ := 17

/-- Theorem: The frog jumped 22 inches farther than the grasshopper -/
theorem frog_jump_difference : frog_jump - grasshopper_jump = 22 := by
  sorry

end NUMINAMATH_CALUDE_frog_jump_difference_l551_55163


namespace NUMINAMATH_CALUDE_shirt_shoe_cost_multiple_l551_55105

/-- The multiple of the cost of the shirt that represents the cost of the shoes -/
def multiple_of_shirt_cost (total_cost shirt_cost shoe_cost : ℚ) : ℚ :=
  (shoe_cost - 9) / shirt_cost

theorem shirt_shoe_cost_multiple :
  let total_cost : ℚ := 300
  let shirt_cost : ℚ := 97
  let shoe_cost : ℚ := total_cost - shirt_cost
  shoe_cost = multiple_of_shirt_cost total_cost shirt_cost shoe_cost * shirt_cost + 9 →
  multiple_of_shirt_cost total_cost shirt_cost shoe_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_shirt_shoe_cost_multiple_l551_55105


namespace NUMINAMATH_CALUDE_polyhedron_20_faces_l551_55164

/-- A polyhedron with triangular faces -/
structure Polyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ
  triangular_faces : faces * 3 = edges * 2
  euler_formula : vertices - edges + faces = 2

/-- Theorem: A polyhedron with 20 triangular faces has 12 vertices and 30 edges -/
theorem polyhedron_20_faces (P : Polyhedron) (h : P.faces = 20) : 
  P.vertices = 12 ∧ P.edges = 30 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_20_faces_l551_55164


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l551_55191

/-- A geometric sequence with a_1 * a_3 = a_4 = 4 has a_6 = 8 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_condition : a 1 * a 3 = a 4 ∧ a 4 = 4) : a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l551_55191


namespace NUMINAMATH_CALUDE_range_of_m_l551_55169

def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def is_increasing_on_reals (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (m^2 - m + 1)^x < (m^2 - m + 1)^y

def p (m : ℝ) : Prop := has_two_distinct_real_roots m

def q (m : ℝ) : Prop := is_increasing_on_reals m

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m) →
  ((-2 ≤ m ∧ m < 0) ∨ (1 < m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l551_55169


namespace NUMINAMATH_CALUDE_smallest_x_is_correct_smallest_x_works_l551_55106

/-- The smallest positive integer x such that 1800x is a perfect cube -/
def smallest_x : ℕ := 15

theorem smallest_x_is_correct :
  ∀ y : ℕ, y > 0 → (∃ m : ℕ, 1800 * y = m^3) → y ≥ smallest_x :=
by sorry

theorem smallest_x_works :
  ∃ m : ℕ, 1800 * smallest_x = m^3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_is_correct_smallest_x_works_l551_55106


namespace NUMINAMATH_CALUDE_fraction_equality_l551_55189

theorem fraction_equality (a b c : ℝ) (hb : b ≠ 0) (hc : c^2 + 1 ≠ 0) :
  (a * (c^2 + 1)) / (b * (c^2 + 1)) = a / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l551_55189


namespace NUMINAMATH_CALUDE_variation_relationship_l551_55181

theorem variation_relationship (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_variation_relationship_l551_55181


namespace NUMINAMATH_CALUDE_y_value_l551_55182

theorem y_value (x y : ℝ) (h1 : x = 4) (h2 : y = 3 * x) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l551_55182


namespace NUMINAMATH_CALUDE_prob_different_suits_l551_55123

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in the combined deck -/
def CombinedDeck : ℕ := 2 * StandardDeck

/-- Represents the number of cards of the same suit in the combined deck -/
def SameSuitCards : ℕ := 26

/-- The probability of drawing two cards of different suits from a pile of two shuffled standard 52-card decks -/
theorem prob_different_suits : 
  (CombinedDeck - 1 - SameSuitCards) / (CombinedDeck - 1 : ℚ) = 78 / 103 := by
sorry

end NUMINAMATH_CALUDE_prob_different_suits_l551_55123


namespace NUMINAMATH_CALUDE_cosine_amplitude_l551_55158

theorem cosine_amplitude (c d : ℝ) (hc : c < 0) (hd : d > 0) 
  (hmax : ∀ x, c * Real.cos (d * x) ≤ 3) 
  (hmin : ∀ x, -3 ≤ c * Real.cos (d * x)) 
  (hmax_achieved : ∃ x, c * Real.cos (d * x) = 3) 
  (hmin_achieved : ∃ x, c * Real.cos (d * x) = -3) : 
  c = -3 := by
sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l551_55158


namespace NUMINAMATH_CALUDE_green_fish_count_l551_55156

theorem green_fish_count (T : ℕ) : ℕ := by
  -- Define the number of blue fish
  let blue : ℕ := T / 2

  -- Define the number of orange fish
  let orange : ℕ := blue - 15

  -- Define the number of green fish
  let green : ℕ := T - blue - orange

  -- Prove that green = 15
  sorry

end NUMINAMATH_CALUDE_green_fish_count_l551_55156


namespace NUMINAMATH_CALUDE_purple_balls_count_l551_55171

/-- Represents the number of green balls in the bin -/
def green_balls : ℕ := 5

/-- Represents the win amount for drawing a green ball -/
def green_win : ℚ := 2

/-- Represents the loss amount for drawing a purple ball -/
def purple_loss : ℚ := 2

/-- Represents the expected winnings -/
def expected_win : ℚ := (1 : ℚ) / 2

/-- 
Given a bin with 5 green balls and k purple balls, where k is a positive integer,
and a game where drawing a green ball wins 2 dollars and drawing a purple ball loses 2 dollars,
if the expected amount won is 50 cents, then k must equal 3.
-/
theorem purple_balls_count (k : ℕ+) : 
  (green_balls : ℚ) / (green_balls + k) * green_win + 
  (k : ℚ) / (green_balls + k) * (-purple_loss) = expected_win → 
  k = 3 := by
  sorry


end NUMINAMATH_CALUDE_purple_balls_count_l551_55171


namespace NUMINAMATH_CALUDE_hamburger_cost_is_four_l551_55166

/-- The cost of Morgan's lunch items and transaction details -/
structure LunchOrder where
  hamburger_cost : ℝ
  onion_rings_cost : ℝ
  smoothie_cost : ℝ
  total_paid : ℝ
  change_received : ℝ

/-- Theorem stating the cost of the hamburger in Morgan's lunch order -/
theorem hamburger_cost_is_four (order : LunchOrder)
  (h1 : order.onion_rings_cost = 2)
  (h2 : order.smoothie_cost = 3)
  (h3 : order.total_paid = 20)
  (h4 : order.change_received = 11) :
  order.hamburger_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_cost_is_four_l551_55166


namespace NUMINAMATH_CALUDE_f_composition_of_three_l551_55101

def f (x : ℝ) : ℝ := -3 * x + 5

theorem f_composition_of_three : f (f (f 3)) = -46 := by sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l551_55101


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_six_l551_55185

/-- A function with the property f(1-x) = f(3+x) for all x -/
def symmetric_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 - x) = f (3 + x)

/-- The set of zeros of a function -/
def zeros (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = 0}

/-- Theorem: If f is a symmetric function with exactly three distinct zeros,
    then the sum of these zeros is 6 -/
theorem sum_of_zeros_is_six (f : ℝ → ℝ) 
    (h_sym : symmetric_function f) 
    (h_zeros : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ zeros f = {a, b, c}) :
  ∃ a b c : ℝ, zeros f = {a, b, c} ∧ a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_six_l551_55185


namespace NUMINAMATH_CALUDE_degree_of_specific_monomial_l551_55145

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (m : Polynomial ℚ) : ℕ :=
  sorry

/-- The monomial 2/3 * a^3 * b -/
def monomial : Polynomial ℚ :=
  sorry

theorem degree_of_specific_monomial :
  degree_of_monomial monomial = 4 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_specific_monomial_l551_55145


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l551_55146

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_power : 
  tens_digit ((3 + 4)^25) + ones_digit ((3 + 4)^25) = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l551_55146


namespace NUMINAMATH_CALUDE_hamburgers_served_equals_three_l551_55180

/-- The number of hamburgers made by the restaurant -/
def total_hamburgers : ℕ := 9

/-- The number of hamburgers left over -/
def leftover_hamburgers : ℕ := 6

/-- The number of hamburgers served -/
def served_hamburgers : ℕ := total_hamburgers - leftover_hamburgers

theorem hamburgers_served_equals_three : served_hamburgers = 3 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_served_equals_three_l551_55180


namespace NUMINAMATH_CALUDE_line_equation_l551_55112

/-- A line passing through a point with given intercepts -/
structure Line where
  point : ℝ × ℝ
  x_intercept : ℝ
  y_intercept : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfies_equation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Check if an equation represents the given line -/
def is_equation_of_line (l : Line) (eq : LineEquation) : Prop :=
  satisfies_equation l.point eq ∧
  (eq.a ≠ 0 → satisfies_equation (l.x_intercept, 0) eq) ∧
  (eq.b ≠ 0 → satisfies_equation (0, l.y_intercept) eq)

/-- The main theorem -/
theorem line_equation (l : Line) 
    (h1 : l.point = (1, 2))
    (h2 : l.x_intercept = 2 * l.y_intercept) :
  (is_equation_of_line l ⟨2, -1, 0⟩) ∨ 
  (is_equation_of_line l ⟨1, 2, -5⟩) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l551_55112


namespace NUMINAMATH_CALUDE_rope_for_third_post_l551_55114

theorem rope_for_third_post 
  (total_rope : ℕ) 
  (first_post second_post fourth_post : ℕ) 
  (h1 : total_rope = 70)
  (h2 : first_post = 24)
  (h3 : second_post = 20)
  (h4 : fourth_post = 12) :
  total_rope - (first_post + second_post + fourth_post) = 14 := by
  sorry

end NUMINAMATH_CALUDE_rope_for_third_post_l551_55114


namespace NUMINAMATH_CALUDE_cubic_diff_linear_diff_mod_six_l551_55195

theorem cubic_diff_linear_diff_mod_six (x y : ℤ) : 
  (x^3 - y^3) % 6 = (x - y) % 6 := by sorry

end NUMINAMATH_CALUDE_cubic_diff_linear_diff_mod_six_l551_55195


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l551_55198

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l551_55198


namespace NUMINAMATH_CALUDE_g_sum_property_l551_55140

-- Define the function g
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + 5

-- State the theorem
theorem g_sum_property (a b c : ℝ) : g a b c 10 = 3 → g a b c 10 + g a b c (-10) = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_property_l551_55140


namespace NUMINAMATH_CALUDE_inequality_equivalence_l551_55151

theorem inequality_equivalence (x : ℝ) : -1/2 * x - 1 < 0 ↔ x > -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l551_55151


namespace NUMINAMATH_CALUDE_c_symmetric_l551_55116

def c : ℕ → ℕ → ℤ
  | m, 0 => 1
  | 0, n => 1
  | m+1, n+1 => c m (n+1) - (n+1) * c m n

theorem c_symmetric (m n : ℕ) (hm : m > 0) (hn : n > 0) : c m n = c n m := by
  sorry

end NUMINAMATH_CALUDE_c_symmetric_l551_55116


namespace NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l551_55173

def alice_number : ℕ := 60

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n → p ∣ m)

theorem smallest_number_with_same_prime_factors :
  ∃ (bob_number : ℕ), 
    has_all_prime_factors alice_number bob_number ∧
    ∀ (m : ℕ), has_all_prime_factors alice_number m → bob_number ≤ m ∧
    bob_number = 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l551_55173


namespace NUMINAMATH_CALUDE_min_vertices_blue_triangle_or_red_K4_l551_55111

/-- A type representing a 2-coloring of edges in a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Bool

/-- Predicate for the existence of a blue triangle in a 2-coloring -/
def has_blue_triangle (n : ℕ) (c : TwoColoring n) : Prop :=
  ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    c i j = false ∧ c j k = false ∧ c i k = false

/-- Predicate for the existence of a red K4 in a 2-coloring -/
def has_red_K4 (n : ℕ) (c : TwoColoring n) : Prop :=
  ∃ i j k l : Fin n, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    c i j = true ∧ c i k = true ∧ c i l = true ∧
    c j k = true ∧ c j l = true ∧ c k l = true

/-- The main theorem -/
theorem min_vertices_blue_triangle_or_red_K4 :
  (∀ n < 9, ∃ c : TwoColoring n, ¬has_blue_triangle n c ∧ ¬has_red_K4 n c) ∧
  (∀ c : TwoColoring 9, has_blue_triangle 9 c ∨ has_red_K4 9 c) :=
sorry

end NUMINAMATH_CALUDE_min_vertices_blue_triangle_or_red_K4_l551_55111


namespace NUMINAMATH_CALUDE_hundred_thirteen_in_sequence_l551_55124

/-- Ewan's sequence starting at 3 and increasing by 11 each time -/
def ewans_sequence (n : ℕ) : ℤ := 11 * n - 8

/-- Theorem stating that 113 is in Ewan's sequence -/
theorem hundred_thirteen_in_sequence : ∃ n : ℕ, ewans_sequence n = 113 := by
  sorry

end NUMINAMATH_CALUDE_hundred_thirteen_in_sequence_l551_55124


namespace NUMINAMATH_CALUDE_pen_cost_l551_55128

/-- Given the cost of pens and pencils in two different combinations, 
    prove that the cost of a single pen is 39 cents. -/
theorem pen_cost (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 183) 
  (eq2 : 5 * x + 4 * y = 327) : 
  x = 39 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l551_55128


namespace NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_120_l551_55107

theorem greatest_common_multiple_10_15_under_120 : 
  ∃ (n : ℕ), n = Nat.lcm 10 15 ∧ n < 120 ∧ ∀ m : ℕ, (m = Nat.lcm 10 15 ∧ m < 120) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_120_l551_55107


namespace NUMINAMATH_CALUDE_max_popsicles_with_10_dollars_l551_55193

/-- Represents the available popsicle purchase options -/
structure PopsicleOption where
  quantity : ℕ
  price : ℕ

/-- Finds the maximum number of popsicles that can be purchased with a given budget -/
def maxPopsicles (options : List PopsicleOption) (budget : ℕ) : ℕ :=
  sorry

/-- The main theorem proving that 23 is the maximum number of popsicles that can be purchased -/
theorem max_popsicles_with_10_dollars :
  let options : List PopsicleOption := [
    ⟨1, 1⟩,  -- Single popsicle
    ⟨3, 2⟩,  -- 3-popsicle box
    ⟨5, 3⟩,  -- 5-popsicle box
    ⟨10, 4⟩  -- 10-popsicle box
  ]
  let budget := 10
  maxPopsicles options budget = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_popsicles_with_10_dollars_l551_55193


namespace NUMINAMATH_CALUDE_income_comparison_l551_55118

/-- Given that Mary's income is 60% more than Tim's income, and Tim's income is 20% less than Juan's income, 
    prove that Mary's income is 128% of Juan's income. -/
theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.8)
  (h2 : mary = tim * 1.6) : 
  mary = juan * 1.28 := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l551_55118


namespace NUMINAMATH_CALUDE_series_sum_equals_20_over_3_l551_55104

/-- The sum of the series (7n+2)/k^n from n=1 to infinity -/
noncomputable def series_sum (k : ℝ) : ℝ := ∑' n, (7 * n + 2) / k^n

/-- Theorem stating that if k > 1 and the series sum equals 20/3, then k = 2.9 -/
theorem series_sum_equals_20_over_3 (k : ℝ) (h1 : k > 1) (h2 : series_sum k = 20/3) : k = 2.9 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_20_over_3_l551_55104


namespace NUMINAMATH_CALUDE_james_sales_problem_l551_55176

theorem james_sales_problem :
  let houses_day1 : ℕ := 20
  let houses_day2 : ℕ := 40
  let sale_rate_day2 : ℚ := 4/5
  let total_items : ℕ := 104
  let items_per_house : ℕ := 2
  
  (houses_day1 * items_per_house + 
   (houses_day2 : ℚ) * sale_rate_day2 * (items_per_house : ℚ) = (total_items : ℚ)) ∧
  (houses_day2 = 2 * houses_day1) :=
by
  sorry

end NUMINAMATH_CALUDE_james_sales_problem_l551_55176


namespace NUMINAMATH_CALUDE_relationship_between_b_and_c_l551_55160

theorem relationship_between_b_and_c 
  (a b c d : ℝ) 
  (h1 : a * b * c * d < 0) 
  (h2 : a > 0) 
  (h3 : b > c) 
  (h4 : d < 0) : 
  (0 < c ∧ c < b) ∨ (c < b ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_b_and_c_l551_55160


namespace NUMINAMATH_CALUDE_circle_trajectory_is_ellipse_l551_55159

/-- Circle type representing a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of an ellipse -/
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ 2 ∧ x ≠ -2

/-- External tangency between two circles -/
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Internal tangency between two circles -/
def internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

theorem circle_trajectory_is_ellipse (M N P : Circle) :
  M.center = (-1, 0) ∧ M.radius = 1 ∧
  N.center = (1, 0) ∧ N.radius = 3 ∧
  externally_tangent P M ∧
  internally_tangent P N →
  is_ellipse P.center.1 P.center.2 :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_is_ellipse_l551_55159


namespace NUMINAMATH_CALUDE_james_total_distance_l551_55110

/-- Calculates the total distance driven given initial speed, initial time, and multipliers for the second part of the journey. -/
def totalDistance (initialSpeed : ℝ) (initialTime : ℝ) (timeMult : ℝ) (speedMult : ℝ) : ℝ :=
  initialSpeed * initialTime + (speedMult * initialSpeed) * (timeMult * initialTime)

/-- Proves that James drove 75 miles in total -/
theorem james_total_distance :
  totalDistance 30 0.5 2 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_james_total_distance_l551_55110


namespace NUMINAMATH_CALUDE_vector_magnitude_sum_l551_55179

/-- Given two vectors a and b in ℝ², prove that if |a| = 3, |b| = 4, 
    and a - b = (√2, √7), then |a + b| = √41 -/
theorem vector_magnitude_sum (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 3)
  (h2 : ‖b‖ = 4)
  (h3 : a - b = (Real.sqrt 2, Real.sqrt 7)) :
  ‖a + b‖ = Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_vector_magnitude_sum_l551_55179


namespace NUMINAMATH_CALUDE_opposite_reciprocal_sum_l551_55157

theorem opposite_reciprocal_sum (m n c d : ℝ) : 
  m = -n → c * d = 1 → m + n + 3 * c * d - 10 = -7 := by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_sum_l551_55157


namespace NUMINAMATH_CALUDE_other_x_intercept_l551_55199

/-- A quadratic function with vertex (5, -3) and one x-intercept at (0, 0) -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem other_x_intercept (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 5)^2 - 3) →  -- vertex form
  QuadraticFunction a b c 0 = 0 →                        -- (0, 0) is an x-intercept
  ∃ x, x ≠ 0 ∧ QuadraticFunction a b c x = 0 ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_other_x_intercept_l551_55199


namespace NUMINAMATH_CALUDE_sum_smallest_largest_primes_1_to_50_l551_55132

theorem sum_smallest_largest_primes_1_to_50 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧
    1 < p ∧ p ≤ 50 ∧
    1 < q ∧ q ≤ 50 ∧
    (∀ r : ℕ, r.Prime → 1 < r → r ≤ 50 → p ≤ r ∧ r ≤ q) ∧
    p + q = 49 :=
  sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_primes_1_to_50_l551_55132


namespace NUMINAMATH_CALUDE_line_tangent_to_fixed_circle_l551_55153

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a line -/
structure Line :=
  (p1 : Point)
  (p2 : Point)

/-- Function to check if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Function to get the circumcircle of a triangle -/
def circumcircle (t : Triangle) : Circle := sorry

/-- Function to check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Function to check if a point is in a half-plane relative to a line -/
def isInHalfPlane (p : Point) (l : Line) : Prop := sorry

/-- Function to get the perpendicular bisector of a line segment -/
def perpendicularBisector (p1 : Point) (p2 : Point) : Line := sorry

/-- Function to get the intersection of two lines -/
def lineIntersection (l1 : Line) (l2 : Line) : Point := sorry

/-- Function to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Main theorem -/
theorem line_tangent_to_fixed_circle 
  (A B C : Point) 
  (h1 : isAcuteAngled (Triangle.mk A B C))
  (h2 : ∀ C', isOnCircle C' (circumcircle (Triangle.mk A B C)) → 
              isInHalfPlane C' (Line.mk A B) → 
              ∃ M N : Point,
                M = lineIntersection (perpendicularBisector B C') (Line.mk A C') ∧
                N = lineIntersection (perpendicularBisector A C') (Line.mk B C') ∧
                ∃ fixedCircle : Circle, isTangent (Line.mk M N) fixedCircle) :
  ∃ fixedCircle : Circle, ∀ C' M N : Point,
    isOnCircle C' (circumcircle (Triangle.mk A B C)) →
    isInHalfPlane C' (Line.mk A B) →
    M = lineIntersection (perpendicularBisector B C') (Line.mk A C') →
    N = lineIntersection (perpendicularBisector A C') (Line.mk B C') →
    isTangent (Line.mk M N) fixedCircle :=
by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_fixed_circle_l551_55153


namespace NUMINAMATH_CALUDE_peters_class_size_l551_55186

theorem peters_class_size :
  ∀ (hands_without_peter : ℕ) (hands_per_student : ℕ),
    hands_without_peter = 20 →
    hands_per_student = 2 →
    hands_without_peter / hands_per_student + 1 = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_peters_class_size_l551_55186


namespace NUMINAMATH_CALUDE_f_4_equals_24_l551_55134

-- Define the function f recursively
def f : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * f n

-- State the theorem
theorem f_4_equals_24 : f 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_f_4_equals_24_l551_55134


namespace NUMINAMATH_CALUDE_intersection_complement_equals_seven_l551_55109

def U : Finset Nat := {4,5,6,7,8}
def M : Finset Nat := {5,8}
def N : Finset Nat := {1,3,5,7,9}

theorem intersection_complement_equals_seven :
  (N ∩ (U \ M)) = {7} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_seven_l551_55109


namespace NUMINAMATH_CALUDE_train_meeting_distance_l551_55135

/-- Represents the distance traveled by a train given its speed and time -/
def distanceTraveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the total initial distance between the trains -/
def totalDistance : ℝ := 350

/-- Represents the speed of Train A in miles per hour -/
def speedA : ℝ := 40

/-- Represents the speed of Train B in miles per hour -/
def speedB : ℝ := 30

/-- Theorem stating that Train A will have traveled 200 miles when the trains meet -/
theorem train_meeting_distance :
  ∃ (t : ℝ), t > 0 ∧ 
  distanceTraveled speedA t + distanceTraveled speedB t = totalDistance ∧
  distanceTraveled speedA t = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_distance_l551_55135


namespace NUMINAMATH_CALUDE_classroom_students_l551_55177

theorem classroom_students (n : ℕ) : 
  n < 50 ∧ n % 6 = 4 ∧ n % 4 = 2 ↔ n ∈ ({10, 22, 34} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_classroom_students_l551_55177


namespace NUMINAMATH_CALUDE_lowest_possible_score_l551_55174

def exam_count : ℕ := 4
def max_score : ℕ := 100
def first_exam_score : ℕ := 84
def second_exam_score : ℕ := 67
def target_average : ℕ := 75

theorem lowest_possible_score :
  ∃ (third_exam_score fourth_exam_score : ℕ),
    third_exam_score ≤ max_score ∧
    fourth_exam_score ≤ max_score ∧
    (first_exam_score + second_exam_score + third_exam_score + fourth_exam_score) / exam_count ≥ target_average ∧
    (third_exam_score = 49 ∨ fourth_exam_score = 49) ∧
    ∀ (x y : ℕ),
      x ≤ max_score →
      y ≤ max_score →
      (first_exam_score + second_exam_score + x + y) / exam_count ≥ target_average →
      x ≥ 49 ∧ y ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l551_55174


namespace NUMINAMATH_CALUDE_find_A_l551_55183

theorem find_A (A : ℕ) (B : ℕ) (h1 : 0 ≤ B ∧ B ≤ 999) 
  (h2 : 1000 * A + B = A * (A + 1) / 2) : A = 1999 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l551_55183


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l551_55148

/-- Quadrilateral EFGH with given properties -/
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)
  (EF_perp_FG : (F.1 - E.1) * (G.2 - F.2) + (F.2 - E.2) * (G.1 - F.1) = 0)
  (HG_perp_FG : (G.1 - H.1) * (G.2 - F.2) + (G.2 - H.2) * (G.1 - F.1) = 0)
  (EF_length : Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) = 12)
  (HG_length : Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) = 3)
  (FG_length : Real.sqrt ((G.1 - F.1)^2 + (G.2 - F.2)^2) = 16)

/-- The perimeter of quadrilateral EFGH is 31 + √337 -/
theorem quadrilateral_perimeter (q : Quadrilateral) :
  Real.sqrt ((q.E.1 - q.F.1)^2 + (q.E.2 - q.F.2)^2) +
  Real.sqrt ((q.F.1 - q.G.1)^2 + (q.F.2 - q.G.2)^2) +
  Real.sqrt ((q.G.1 - q.H.1)^2 + (q.G.2 - q.H.2)^2) +
  Real.sqrt ((q.H.1 - q.E.1)^2 + (q.H.2 - q.E.2)^2) =
  31 + Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l551_55148


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l551_55142

theorem no_solution_for_equation :
  ¬∃ (x : ℝ), x ≠ 1 ∧ x ≠ -1 ∧ (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l551_55142


namespace NUMINAMATH_CALUDE_library_visitors_on_sunday_l551_55192

/-- Proves that the average number of visitors on Sundays is 140 given the specified conditions --/
theorem library_visitors_on_sunday (
  total_days : Nat) 
  (sunday_count : Nat)
  (avg_visitors_per_day : ℝ)
  (avg_visitors_non_sunday : ℝ)
  (h1 : total_days = 30)
  (h2 : sunday_count = 5)
  (h3 : avg_visitors_per_day = 90)
  (h4 : avg_visitors_non_sunday = 80)
  : ℝ :=
by
  -- Proof goes here
  sorry

#check library_visitors_on_sunday

end NUMINAMATH_CALUDE_library_visitors_on_sunday_l551_55192


namespace NUMINAMATH_CALUDE_area_between_line_and_curve_l551_55150

/-- The area enclosed between the line y = 2x and the curve y = x^2 from x = 0 to x = 2 is 4/3 -/
theorem area_between_line_and_curve : 
  ∫ x in (0 : ℝ)..2, (2 * x - x^2) = 4/3 := by sorry

end NUMINAMATH_CALUDE_area_between_line_and_curve_l551_55150


namespace NUMINAMATH_CALUDE_parallel_vectors_expression_l551_55154

noncomputable def θ : ℝ := Real.arctan (3 : ℝ)

theorem parallel_vectors_expression (a b : ℝ × ℝ) :
  a = (3, 1) →
  b = (Real.sin θ, Real.cos θ) →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_expression_l551_55154


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sarahs_scores_l551_55165

def sarahs_scores : List ℝ := [87, 90, 86, 93, 89, 92]

theorem arithmetic_mean_of_sarahs_scores :
  (sarahs_scores.sum / sarahs_scores.length : ℝ) = 89.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sarahs_scores_l551_55165


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_remainder_l551_55137

theorem polynomial_division_quotient_remainder (z : ℚ) :
  3 * z^4 - 4 * z^3 + 5 * z^2 - 11 * z + 2 =
  (2 + 3 * z) * (z^3 - 2 * z^2 + 3 * z - 17/3) + 40/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_remainder_l551_55137


namespace NUMINAMATH_CALUDE_sourball_candies_distribution_l551_55178

def nellie_limit : ℕ := 12
def jacob_limit : ℕ := nellie_limit / 2
def lana_limit : ℕ := jacob_limit - 3
def total_candies : ℕ := 30
def num_people : ℕ := 3

theorem sourball_candies_distribution :
  (total_candies - (nellie_limit + jacob_limit + lana_limit)) / num_people = 3 := by
  sorry

end NUMINAMATH_CALUDE_sourball_candies_distribution_l551_55178


namespace NUMINAMATH_CALUDE_solution_subset_nonpositive_l551_55108

/-- The solution set of |x| > ax + 1 is a subset of {x | x ≤ 0} if and only if a ≥ 1 -/
theorem solution_subset_nonpositive (a : ℝ) :
  (∀ x : ℝ, |x| > a * x + 1 → x ≤ 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_subset_nonpositive_l551_55108


namespace NUMINAMATH_CALUDE_trigonometric_expressions_l551_55100

theorem trigonometric_expressions :
  (2 * Real.sin (30 * π / 180) + 3 * Real.cos (60 * π / 180) - 4 * Real.tan (45 * π / 180) = -3/2) ∧
  (Real.tan (60 * π / 180) - (4 - π)^0 + 2 * Real.cos (30 * π / 180) + (1/4)⁻¹ = 2 * Real.sqrt 3 + 3) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_l551_55100


namespace NUMINAMATH_CALUDE_initial_money_calculation_l551_55167

/-- Calculates the initial amount of money given spending habits and remaining balance --/
theorem initial_money_calculation 
  (spend_per_trip : ℕ)
  (trips_per_month : ℕ)
  (months : ℕ)
  (money_left : ℕ)
  (h1 : spend_per_trip = 2)
  (h2 : trips_per_month = 4)
  (h3 : months = 12)
  (h4 : money_left = 104) :
  spend_per_trip * trips_per_month * months + money_left = 200 := by
  sorry

#check initial_money_calculation

end NUMINAMATH_CALUDE_initial_money_calculation_l551_55167


namespace NUMINAMATH_CALUDE_new_person_weight_l551_55144

/-- The weight of the new person given the conditions of the problem -/
def weight_new_person (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + n * avg_increase

/-- Theorem stating that the weight of the new person is 87.5 kg -/
theorem new_person_weight :
  weight_new_person 9 2.5 65 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l551_55144


namespace NUMINAMATH_CALUDE_oscars_bus_ride_l551_55141

/-- Oscar's bus ride to school problem -/
theorem oscars_bus_ride (charlie_ride : ℝ) (oscar_difference : ℝ) :
  charlie_ride = 0.25 →
  oscar_difference = 0.5 →
  charlie_ride + oscar_difference = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_oscars_bus_ride_l551_55141


namespace NUMINAMATH_CALUDE_current_speed_l551_55162

/-- Proves that given a man's speed with and against the current, the speed of the current can be determined. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 21)
  (h2 : speed_against_current = 12.4) :
  ∃ (current_speed : ℝ), current_speed = 4.3 := by
  sorry


end NUMINAMATH_CALUDE_current_speed_l551_55162
