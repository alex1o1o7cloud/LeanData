import Mathlib

namespace NUMINAMATH_CALUDE_spinner_ice_cream_prices_l1173_117338

-- Define the price of a spinner and an ice cream
variable (s m : ℝ)

-- Define Petya's and Vasya's claims
def petya_claim := 2 * s > 5 * m
def vasya_claim := 3 * s > 8 * m

-- Theorem statement
theorem spinner_ice_cream_prices 
  (h1 : (petya_claim s m ∧ ¬vasya_claim s m) ∨ (¬petya_claim s m ∧ vasya_claim s m))
  (h2 : vasya_claim s m) :
  7 * s ≤ 19 * m := by
  sorry

end NUMINAMATH_CALUDE_spinner_ice_cream_prices_l1173_117338


namespace NUMINAMATH_CALUDE_value_of_N_l1173_117339

theorem value_of_N : ∃ N : ℝ, (0.25 * N = 0.55 * 3010) ∧ (N = 6622) := by
  sorry

end NUMINAMATH_CALUDE_value_of_N_l1173_117339


namespace NUMINAMATH_CALUDE_division_remainder_l1173_117378

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 222 → divisor = 13 → quotient = 17 → 
  dividend = divisor * quotient + remainder → remainder = 1 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l1173_117378


namespace NUMINAMATH_CALUDE_vertex_locus_is_hyperbola_l1173_117326

/-- The locus of the vertex of a parabola is a hyperbola -/
theorem vertex_locus_is_hyperbola 
  (a b : ℝ) 
  (h : 8 * a^2 + 4 * a * b = b^3) : 
  ∃ (x y : ℝ), x * y = 1 ∧ 
  x = -b / (2 * a) ∧ 
  y = (4 * a - b^2) / (4 * a) := by
  sorry

end NUMINAMATH_CALUDE_vertex_locus_is_hyperbola_l1173_117326


namespace NUMINAMATH_CALUDE_tony_midpoint_age_l1173_117342

/-- Represents Tony's age and earnings over a 60-day period --/
structure TonyEarnings where
  daysWorked : Nat
  hoursPerDay : Nat
  hourlyRateMultiplier : Rat
  startAge : Nat
  midAge : Nat
  endAge : Nat
  totalEarnings : Rat

/-- Calculates Tony's earnings based on his age and work details --/
def calculateEarnings (t : TonyEarnings) : Rat :=
  let firstHalfDays := t.daysWorked / 2
  let secondHalfDays := t.daysWorked - firstHalfDays
  (t.hoursPerDay * t.hourlyRateMultiplier * t.startAge * firstHalfDays : Rat) +
  (t.hoursPerDay * t.hourlyRateMultiplier * t.endAge * secondHalfDays : Rat)

/-- Theorem stating that Tony's age at the midpoint was 11 --/
theorem tony_midpoint_age (t : TonyEarnings) 
  (h1 : t.daysWorked = 60)
  (h2 : t.hoursPerDay = 3)
  (h3 : t.hourlyRateMultiplier = 3/4)
  (h4 : t.startAge = 10)
  (h5 : t.endAge = 12)
  (h6 : t.totalEarnings = 1125)
  (h7 : calculateEarnings t = t.totalEarnings) :
  t.midAge = 11 := by
  sorry

end NUMINAMATH_CALUDE_tony_midpoint_age_l1173_117342


namespace NUMINAMATH_CALUDE_point_position_on_line_l1173_117304

/-- Given points on a line, prove the position of a point P satisfying a ratio condition -/
theorem point_position_on_line 
  (O A B C D P : ℝ) 
  (h_order : O ≤ A ∧ A ≤ B ∧ B ≤ C ∧ C ≤ D)
  (h_dist_OA : A - O = a)
  (h_dist_OB : B - O = b)
  (h_dist_OC : C - O = c)
  (h_dist_OD : D - O = d)
  (h_P_between : B ≤ P ∧ P ≤ C)
  (h_ratio : (P - A) / (D - P) = 2 * ((P - B) / (C - P))) :
  P - O = b + c - a :=
sorry

end NUMINAMATH_CALUDE_point_position_on_line_l1173_117304


namespace NUMINAMATH_CALUDE_sector_area_l1173_117366

/-- The area of a circular sector with central angle 240° and radius 6 is 24π -/
theorem sector_area (θ : Real) (r : Real) : 
  θ = 240 * π / 180 → r = 6 → (1/2) * r^2 * θ = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1173_117366


namespace NUMINAMATH_CALUDE_original_blueberry_count_l1173_117314

/-- Represents the number of blueberry jelly beans Camilla originally had -/
def blueberry : ℕ := sorry

/-- Represents the number of cherry jelly beans Camilla originally had -/
def cherry : ℕ := sorry

/-- Theorem stating the original number of blueberry jelly beans -/
theorem original_blueberry_count : blueberry = 30 := by
  have h1 : blueberry = 3 * cherry := sorry
  have h2 : blueberry - 20 = 2 * (cherry - 5) := sorry
  sorry


end NUMINAMATH_CALUDE_original_blueberry_count_l1173_117314


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l1173_117376

/-- Calculates the principal amount given the simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem: If a loan at 12% annual simple interest generates Rs. 1500 interest in 10 years, 
    then the principal amount was Rs. 1250 -/
theorem loan_principal_calculation :
  let interest : ℚ := 1500
  let rate : ℚ := 12
  let time : ℚ := 10
  calculate_principal interest rate time = 1250 := by
sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l1173_117376


namespace NUMINAMATH_CALUDE_nonCoplanarChoices_eq_141_l1173_117323

/-- The number of ways to choose 4 non-coplanar points from the vertices and midpoints of a tetrahedron -/
def nonCoplanarChoices : ℕ :=
  Nat.choose 10 4 - (4 * Nat.choose 6 4 + 6 + 3)

/-- Theorem stating that the number of ways to choose 4 non-coplanar points
    from the vertices and midpoints of a tetrahedron is 141 -/
theorem nonCoplanarChoices_eq_141 : nonCoplanarChoices = 141 := by
  sorry

end NUMINAMATH_CALUDE_nonCoplanarChoices_eq_141_l1173_117323


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1173_117386

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x / (x - 1) < 1 - a ∧ x ≠ 1}
  (a > 0 → S = Set.Ioo ((a - 1) / a) 1) ∧
  (a = 0 → S = Set.Iio 1) ∧
  (a < 0 → S = Set.Iio 1 ∪ Set.Ioi ((a - 1) / a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1173_117386


namespace NUMINAMATH_CALUDE_det_specific_matrix_l1173_117327

theorem det_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 7; 2, 3]
  Matrix.det A = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l1173_117327


namespace NUMINAMATH_CALUDE_sally_fries_proof_l1173_117390

/-- Calculates the final number of fries Sally has after receiving one-third of Mark's fries -/
def sallys_final_fries (sally_initial : ℕ) (mark_initial : ℕ) : ℕ :=
  sally_initial + mark_initial / 3

/-- Proves that Sally's final fry count is 26 given the initial conditions -/
theorem sally_fries_proof :
  sallys_final_fries 14 36 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sally_fries_proof_l1173_117390


namespace NUMINAMATH_CALUDE_triangle_inequality_l1173_117350

theorem triangle_inequality (a b c : ℝ) (α : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hα : 0 < α ∧ α < π) (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) 
  (h_cosine : a^2 = b^2 + c^2 - 2*b*c*(Real.cos α)) :
  (2*b*c*(Real.cos α))/(b + c) < b + c - a ∧ b + c - a < (2*b*c)/a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1173_117350


namespace NUMINAMATH_CALUDE_power_equality_l1173_117351

theorem power_equality (m : ℕ) : 16^6 = 4^m → m = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1173_117351


namespace NUMINAMATH_CALUDE_model_height_is_correct_l1173_117313

/-- The height of the actual observatory tower in meters -/
def actual_height : ℝ := 60

/-- The volume of water the actual observatory tower can hold in liters -/
def actual_volume : ℝ := 200000

/-- The volume of water Carson's miniature model can hold in liters -/
def model_volume : ℝ := 0.2

/-- The height of Carson's miniature tower in meters -/
def model_height : ℝ := 0.6

/-- Theorem stating that the calculated model height is correct -/
theorem model_height_is_correct :
  model_height = actual_height * (model_volume / actual_volume)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_model_height_is_correct_l1173_117313


namespace NUMINAMATH_CALUDE_equal_distribution_problem_l1173_117346

theorem equal_distribution_problem (earnings : Fin 5 → ℕ) 
  (h1 : earnings 0 = 18)
  (h2 : earnings 1 = 27)
  (h3 : earnings 2 = 30)
  (h4 : earnings 3 = 35)
  (h5 : earnings 4 = 50) :
  50 - (earnings 0 + earnings 1 + earnings 2 + earnings 3 + earnings 4) / 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_problem_l1173_117346


namespace NUMINAMATH_CALUDE_chocolate_chip_cookies_l1173_117330

theorem chocolate_chip_cookies (cookies_per_bag : ℕ) (baggies : ℕ) (oatmeal_cookies : ℕ) :
  cookies_per_bag = 5 →
  baggies = 7 →
  oatmeal_cookies = 2 →
  cookies_per_bag * baggies - oatmeal_cookies = 33 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookies_l1173_117330


namespace NUMINAMATH_CALUDE_prove_research_paper_requirement_l1173_117344

def research_paper_requirement (yvonne_words janna_extra_words removed_words added_multiplier additional_words : ℕ) : Prop :=
  let janna_words := yvonne_words + janna_extra_words
  let initial_total := yvonne_words + janna_words
  let after_removal := initial_total - removed_words
  let added_words := removed_words * added_multiplier
  let after_addition := after_removal + added_words
  let final_requirement := after_addition + additional_words
  final_requirement = 1000

theorem prove_research_paper_requirement :
  research_paper_requirement 400 150 20 2 30 := by
  sorry

end NUMINAMATH_CALUDE_prove_research_paper_requirement_l1173_117344


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1173_117369

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) :
  (a 3) ^ 2 - 6 * (a 3) + 8 = 0 ∧
  (a 15) ^ 2 - 6 * (a 15) + 8 = 0 →
  (a 1 * a 17) / a 9 = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1173_117369


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l1173_117359

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l1173_117359


namespace NUMINAMATH_CALUDE_maplewood_population_estimate_l1173_117361

theorem maplewood_population_estimate :
  ∀ (avg_population : ℝ),
  (25 : ℝ) > 0 →
  6200 ≤ avg_population →
  avg_population ≤ 6800 →
  ∃ (total_population : ℝ),
  total_population = 25 * avg_population ∧
  total_population = 162500 :=
by sorry

end NUMINAMATH_CALUDE_maplewood_population_estimate_l1173_117361


namespace NUMINAMATH_CALUDE_tara_spent_more_on_ice_cream_l1173_117306

/-- The amount Tara spent more on ice cream than on yogurt -/
def ice_cream_yogurt_difference : ℕ :=
  let ice_cream_cartons : ℕ := 19
  let yogurt_cartons : ℕ := 4
  let ice_cream_price : ℕ := 7
  let yogurt_price : ℕ := 1
  (ice_cream_cartons * ice_cream_price) - (yogurt_cartons * yogurt_price)

/-- Theorem stating that Tara spent $129 more on ice cream than on yogurt -/
theorem tara_spent_more_on_ice_cream : ice_cream_yogurt_difference = 129 := by
  sorry

end NUMINAMATH_CALUDE_tara_spent_more_on_ice_cream_l1173_117306


namespace NUMINAMATH_CALUDE_max_squares_covered_two_inch_card_l1173_117347

/-- Represents a square card with a given side length -/
structure SquareCard where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a checkerboard with squares of a given side length -/
structure Checkerboard where
  square_side_length : ℝ
  square_side_length_pos : square_side_length > 0

/-- The maximum number of squares a card can cover on a checkerboard -/
def max_squares_covered (card : SquareCard) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating that a 2-inch square card can cover at most 9 one-inch squares on a checkerboard -/
theorem max_squares_covered_two_inch_card :
  ∀ (card : SquareCard) (board : Checkerboard),
    card.side_length = 2 →
    board.square_side_length = 1 →
    max_squares_covered card board = 9 :=
  sorry

end NUMINAMATH_CALUDE_max_squares_covered_two_inch_card_l1173_117347


namespace NUMINAMATH_CALUDE_partition_equality_l1173_117341

/-- The number of partitions of n into non-negative powers of 2 -/
def b (n : ℕ) : ℕ := sorry

/-- The number of partitions of n which include at least one of every power of 2 
    from 1 up to the highest power of 2 in the partition -/
def c (n : ℕ) : ℕ := sorry

/-- For any non-negative integer n, b(n+1) = 2c(n) -/
theorem partition_equality (n : ℕ) : b (n + 1) = 2 * c n := by sorry

end NUMINAMATH_CALUDE_partition_equality_l1173_117341


namespace NUMINAMATH_CALUDE_total_colors_needed_l1173_117367

/-- Represents the number of moons for each planet in the solar system -/
def moons : Fin 8 → ℕ
  | 0 => 0  -- Mercury
  | 1 => 0  -- Venus
  | 2 => 1  -- Earth
  | 3 => 2  -- Mars
  | 4 => 79 -- Jupiter
  | 5 => 82 -- Saturn
  | 6 => 27 -- Uranus
  | 7 => 14 -- Neptune

/-- The number of planets in the solar system -/
def num_planets : ℕ := 8

/-- The number of people coloring -/
def num_people : ℕ := 3

/-- The total number of celestial bodies (planets and moons) -/
def total_bodies : ℕ := num_planets + (Finset.sum Finset.univ moons)

/-- Theorem stating the total number of colors needed -/
theorem total_colors_needed : num_people * total_bodies = 639 := by
  sorry


end NUMINAMATH_CALUDE_total_colors_needed_l1173_117367


namespace NUMINAMATH_CALUDE_triangle_properties_l1173_117354

/-- Given a triangle ABC with A = 2B, b = 2, and c = 4, prove the following properties -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  (h_A_2B : A = 2 * B)
  (h_b : b = 2)
  (h_c : c = 4) :
  a = 2 * b * Real.cos B ∧ B = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1173_117354


namespace NUMINAMATH_CALUDE_two_numbers_product_cube_sum_l1173_117362

theorem two_numbers_product_cube_sum : ∃ (a b : ℚ), 
  (∃ (x : ℚ), a + (a * b) = x^3) ∧ 
  (∃ (y : ℚ), b + (a * b) = y^3) ∧ 
  a = 112/13 ∧ 
  b = 27/169 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_product_cube_sum_l1173_117362


namespace NUMINAMATH_CALUDE_largest_root_is_four_l1173_117363

/-- The polynomial function representing the difference between the curve and the line -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^6 - 10*x^5 + 29*x^4 - 4*x^3 + a*x^2 - b*x - c

/-- The statement that the polynomial has exactly three distinct roots, each with multiplicity 2 -/
def has_three_double_roots (a b c : ℝ) : Prop :=
  ∃ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    ∀ x, f a b c x = 0 ↔ (x = p ∨ x = q ∨ x = r)

/-- The theorem stating that under the given conditions, 4 is the largest root -/
theorem largest_root_is_four (a b c : ℝ) (h : has_three_double_roots a b c) :
  ∃ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (∀ x, f a b c x = 0 ↔ (x = p ∨ x = q ∨ x = r)) ∧
    4 = max p (max q r) :=
  sorry

end NUMINAMATH_CALUDE_largest_root_is_four_l1173_117363


namespace NUMINAMATH_CALUDE_triangle_property_l1173_117356

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle angle condition
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a / Real.sin A = b / Real.sin B →  -- Law of sines (partial)
  a / Real.sin A = c / Real.sin C →  -- Law of sines (partial)
  (2 * c + b) * Real.cos A + a * Real.cos B = 0 →  -- Given equation
  a = Real.sqrt 3 →  -- Given side length
  A = 2 * π / 3 ∧ Real.sqrt 3 < 2 * b + c ∧ 2 * b + c < 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_property_l1173_117356


namespace NUMINAMATH_CALUDE_sum_modulo_thirteen_l1173_117317

theorem sum_modulo_thirteen : (9375 + 9376 + 9377 + 9378 + 9379) % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_modulo_thirteen_l1173_117317


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1173_117335

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 48 18 = 159 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1173_117335


namespace NUMINAMATH_CALUDE_pair_in_six_cascades_valid_coloring_exists_l1173_117308

-- Define a cascade
def cascade (r : ℕ) : Set ℕ := {n : ℕ | ∃ k : ℕ, k ≤ 12 ∧ n = k * r}

-- Part a: Existence of a pair in six cascades
theorem pair_in_six_cascades : ∃ a b : ℕ, ∃ r₁ r₂ r₃ r₄ r₅ r₆ : ℕ,
  r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ r₁ ≠ r₆ ∧
  r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧ r₂ ≠ r₆ ∧
  r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧ r₃ ≠ r₆ ∧
  r₄ ≠ r₅ ∧ r₄ ≠ r₆ ∧
  r₅ ≠ r₆ ∧
  a ∈ cascade r₁ ∧ b ∈ cascade r₁ ∧
  a ∈ cascade r₂ ∧ b ∈ cascade r₂ ∧
  a ∈ cascade r₃ ∧ b ∈ cascade r₃ ∧
  a ∈ cascade r₄ ∧ b ∈ cascade r₄ ∧
  a ∈ cascade r₅ ∧ b ∈ cascade r₅ ∧
  a ∈ cascade r₆ ∧ b ∈ cascade r₆ := by
  sorry

-- Part b: Existence of a valid coloring function
theorem valid_coloring_exists : ∃ f : ℕ → Fin 12, ∀ r : ℕ, ∀ k₁ k₂ : ℕ,
  k₁ ≤ 12 → k₂ ≤ 12 → k₁ ≠ k₂ → f (k₁ * r) ≠ f (k₂ * r) := by
  sorry

end NUMINAMATH_CALUDE_pair_in_six_cascades_valid_coloring_exists_l1173_117308


namespace NUMINAMATH_CALUDE_sum_is_non_horizontal_line_l1173_117370

-- Define the original parabola
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define f(x) as the reflection and left translation of the original parabola
def f (a b c : ℝ) (x : ℝ) : ℝ := original_parabola a b c (x + 5)

-- Define g(x) as the reflection and right translation of the original parabola
def g (a b c : ℝ) (x : ℝ) : ℝ := -original_parabola a b c (x - 5)

-- Define the sum function (f + g)(x)
def f_plus_g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

-- Theorem: The sum function (f + g)(x) is a non-horizontal line
theorem sum_is_non_horizontal_line (a b c : ℝ) :
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x : ℝ, f_plus_g a b c x = m * x + k :=
sorry

end NUMINAMATH_CALUDE_sum_is_non_horizontal_line_l1173_117370


namespace NUMINAMATH_CALUDE_cost_difference_formula_option_A_cheaper_at_50_l1173_117334

/-- The number of teachers -/
def num_teachers : ℕ := 5

/-- The full ticket price -/
def full_price : ℕ := 40

/-- Cost calculation for Option A -/
def cost_A (x : ℕ) : ℕ := 20 * x + 200

/-- Cost calculation for Option B -/
def cost_B (x : ℕ) : ℕ := 24 * x + 120

/-- The cost difference between Option B and Option A -/
def cost_difference (x : ℕ) : ℤ := (cost_B x : ℤ) - (cost_A x : ℤ)

theorem cost_difference_formula (x : ℕ) : 
  cost_difference x = 4 * x - 80 :=
sorry

theorem option_A_cheaper_at_50 : 
  cost_A 50 < cost_B 50 :=
sorry

end NUMINAMATH_CALUDE_cost_difference_formula_option_A_cheaper_at_50_l1173_117334


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1173_117340

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Theorem statement
theorem hyperbola_asymptote :
  ∀ (x y : ℝ), (hyperbola x y) → (asymptote x y) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1173_117340


namespace NUMINAMATH_CALUDE_derivative_of_odd_is_even_l1173_117315

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The derivative of an odd function is an even function -/
theorem derivative_of_odd_is_even (f : ℝ → ℝ) (hf : IsOdd f) (hf' : Differentiable ℝ f) :
  IsEven (deriv f) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_odd_is_even_l1173_117315


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_seven_l1173_117328

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 1) * x + 2

-- State the theorem
theorem monotone_decreasing_implies_a_leq_neg_seven (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a y < f a x) →
  a ≤ -7 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_seven_l1173_117328


namespace NUMINAMATH_CALUDE_volume_theorem_l1173_117353

noncomputable def volume_of_body : ℝ :=
  let surface1 (x y z : ℝ) := 2 * z = x^2 + y^2
  let surface2 (z : ℝ) := z = 2
  let surface3 (x : ℝ) := x = 0
  let surface4 (x y : ℝ) := y = 2 * x
  let arctan2 := Real.arctan 2
  2 * arctan2

theorem volume_theorem :
  volume_of_body = 1.704 * Real.pi := by sorry

end NUMINAMATH_CALUDE_volume_theorem_l1173_117353


namespace NUMINAMATH_CALUDE_inverse_variation_cube_l1173_117349

/-- Given two positive real numbers x and y that vary inversely with respect to x^3,
    if y = 8 when x = 2, then x = 1 when y = 64. -/
theorem inverse_variation_cube (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ (k : ℝ), ∀ x y, x^3 * y = k)
  (h_initial : 2^3 * 8 = (x^3 * y)) :
  y = 64 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_l1173_117349


namespace NUMINAMATH_CALUDE_square_starts_with_123456789_l1173_117312

theorem square_starts_with_123456789 : ∃ (n : ℕ) (k : ℕ), 
  (123456789 : ℕ) * 10^k ≤ n^2 ∧ n^2 < (123456790 : ℕ) * 10^k :=
sorry

end NUMINAMATH_CALUDE_square_starts_with_123456789_l1173_117312


namespace NUMINAMATH_CALUDE_function_range_l1173_117355

theorem function_range (x : ℝ) : 
  (∀ a : ℝ, a ∈ Set.Icc (-1 : ℝ) 1 → 
    (a * x^2 - (2*a + 1) * x + a + 1 < 0)) → 
  (1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1173_117355


namespace NUMINAMATH_CALUDE_weight_of_doubled_cube_l1173_117318

/-- Given a cube of metal weighing 6 pounds, prove that another cube of the same metal
    with sides twice as long will weigh 48 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (weight : ℝ) (h1 : weight = 6) :
  let new_weight := weight * (2^3)
  new_weight = 48 := by sorry

end NUMINAMATH_CALUDE_weight_of_doubled_cube_l1173_117318


namespace NUMINAMATH_CALUDE_bus_driver_average_hours_l1173_117321

/-- The average number of hours the bus driver drives each day -/
def average_hours : ℝ := 2

/-- The average speed from Monday to Wednesday in km/h -/
def speed_mon_wed : ℝ := 12

/-- The average speed from Thursday to Friday in km/h -/
def speed_thu_fri : ℝ := 9

/-- The total distance traveled in 5 days in km -/
def total_distance : ℝ := 108

/-- The number of days driven from Monday to Wednesday -/
def days_mon_wed : ℝ := 3

/-- The number of days driven from Thursday to Friday -/
def days_thu_fri : ℝ := 2

theorem bus_driver_average_hours :
  average_hours * speed_mon_wed * days_mon_wed +
  average_hours * speed_thu_fri * days_thu_fri = total_distance :=
sorry

end NUMINAMATH_CALUDE_bus_driver_average_hours_l1173_117321


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_l1173_117371

/-- Calculates the total cost of fruits with a discount -/
def totalCostWithDiscount (cherryPrice olivePrice : ℚ) (bagCount : ℕ) (discountPercentage : ℚ) : ℚ :=
  let discountFactor : ℚ := 1 - discountPercentage / 100
  let discountedCherryPrice : ℚ := cherryPrice * discountFactor
  let discountedOlivePrice : ℚ := olivePrice * discountFactor
  (discountedCherryPrice + discountedOlivePrice) * bagCount

/-- Proves that the total cost for 50 bags each of cherries and olives with a 10% discount is $540 -/
theorem fruit_purchase_cost : 
  totalCostWithDiscount 5 7 50 10 = 540 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_cost_l1173_117371


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_5_minus_5_4_l1173_117375

theorem least_prime_factor_of_5_5_minus_5_4 :
  Nat.minFac (5^5 - 5^4) = 2 := by
sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_5_minus_5_4_l1173_117375


namespace NUMINAMATH_CALUDE_diagonal_length_isosceles_trapezoid_l1173_117385

-- Define the isosceles trapezoid
structure IsoscelesTrapezoid :=
  (AB : ℝ) -- longer base
  (CD : ℝ) -- shorter base
  (AD : ℝ) -- leg
  (BC : ℝ) -- leg
  (isIsosceles : AD = BC)
  (isPositive : AB > 0 ∧ CD > 0 ∧ AD > 0)
  (baseOrder : AB > CD)

-- Theorem statement
theorem diagonal_length_isosceles_trapezoid (T : IsoscelesTrapezoid) 
  (h1 : T.AB = 25) 
  (h2 : T.CD = 13) 
  (h3 : T.AD = 12) :
  Real.sqrt ((25 - 13) ^ 2 / 4 + (Real.sqrt (12 ^ 2 - ((25 - 13) / 2) ^ 2)) ^ 2) = 12 :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_isosceles_trapezoid_l1173_117385


namespace NUMINAMATH_CALUDE_salary_relation_l1173_117365

theorem salary_relation (A B C : ℝ) :
  A + B + C = 10000 ∧
  A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧
  0.1 * A + 0.15 * B = 0.2 * C →
  A = 20000 / 3 - 7 * B / 6 :=
by sorry

end NUMINAMATH_CALUDE_salary_relation_l1173_117365


namespace NUMINAMATH_CALUDE_coefficient_of_expansion_l1173_117372

theorem coefficient_of_expansion (x : ℝ) : 
  ∃ a b c d e : ℝ, (2*x + 1)^5 = a + b*(x+1) + c*(x+1)^2 + d*(x+1)^3 + e*(x+1)^4 + (-5)*(x+1)^5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_expansion_l1173_117372


namespace NUMINAMATH_CALUDE_sequence_formulas_l1173_117394

/-- Given an arithmetic sequence a_n with first term 19 and common difference -2,
    and a geometric sequence b_n - a_n with first term 1 and common ratio 3,
    prove the formulas for a_n, S_n, b_n, and T_n. -/
theorem sequence_formulas (n : ℕ) :
  let a : ℕ → ℝ := λ k => 19 - 2 * (k - 1)
  let S : ℕ → ℝ := λ k => (k * (a 1 + a k)) / 2
  let b : ℕ → ℝ := λ k => a k + 3^(k - 1)
  let T : ℕ → ℝ := λ k => S k + (3^k - 1) / 2
  (a n = 21 - 2 * n) ∧
  (S n = 20 * n - n^2) ∧
  (b n = 21 - 2 * n + 3^(n - 1)) ∧
  (T n = 20 * n - n^2 + (3^n - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formulas_l1173_117394


namespace NUMINAMATH_CALUDE_fraction_sum_l1173_117332

theorem fraction_sum : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1173_117332


namespace NUMINAMATH_CALUDE_underpaid_amount_is_correct_l1173_117373

/-- Represents the time it takes for the minute hand to coincide with the hour hand once on an accurate clock (in minutes) -/
def accurate_clock_time : ℚ := 60 + 60 / 11

/-- Represents the time it takes for the minute hand to coincide with the hour hand once on the inaccurate clock (in minutes) -/
def inaccurate_clock_time : ℚ := 69

/-- Represents the hourly wage rate (in yuan) -/
def hourly_wage : ℚ := 6

/-- Represents the nominal workday length (in hours) -/
def nominal_workday : ℚ := 8

/-- Calculates the actual working time in a day (in hours) -/
def actual_working_time : ℚ :=
  nominal_workday * (inaccurate_clock_time / accurate_clock_time)

/-- Calculates the excess time worked (in hours) -/
def excess_time : ℚ := actual_working_time - nominal_workday

/-- Calculates the amount underpaid to each worker per day (in yuan) -/
def underpaid_amount : ℚ := hourly_wage * excess_time

theorem underpaid_amount_is_correct :
  underpaid_amount = 26 / 10 := by sorry

end NUMINAMATH_CALUDE_underpaid_amount_is_correct_l1173_117373


namespace NUMINAMATH_CALUDE_green_sweets_count_l1173_117395

theorem green_sweets_count (total : ℕ) (red : ℕ) (neither : ℕ) (h1 : total = 285) (h2 : red = 49) (h3 : neither = 177) :
  total - red - neither = 59 := by
  sorry

end NUMINAMATH_CALUDE_green_sweets_count_l1173_117395


namespace NUMINAMATH_CALUDE_subtraction_problem_l1173_117392

theorem subtraction_problem (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l1173_117392


namespace NUMINAMATH_CALUDE_x_squared_plus_x_is_quadratic_binomial_l1173_117329

/-- A quadratic binomial is a polynomial of degree 2 with two terms. -/
def is_quadratic_binomial (p : Polynomial ℝ) : Prop :=
  p.degree = 2 ∧ p.support.card = 2

/-- x^2 + x is a quadratic binomial -/
theorem x_squared_plus_x_is_quadratic_binomial :
  is_quadratic_binomial (X^2 + X : Polynomial ℝ) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_x_is_quadratic_binomial_l1173_117329


namespace NUMINAMATH_CALUDE_parallelogram_height_calculation_l1173_117381

/-- Given a parallelogram-shaped field with specified dimensions and costs, 
    calculate the perpendicular distance from the other side. -/
theorem parallelogram_height_calculation 
  (base : ℝ)
  (cost_per_10sqm : ℝ)
  (total_cost : ℝ)
  (h : base = 54)
  (i : cost_per_10sqm = 50)
  (j : total_cost = 6480) :
  (total_cost / cost_per_10sqm * 10) / base = 24 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_height_calculation_l1173_117381


namespace NUMINAMATH_CALUDE_largest_angle_of_triangle_l1173_117393

/-- Given a triangle PQR with side lengths p, q, and r satisfying certain conditions,
    prove that its largest angle is 120 degrees. -/
theorem largest_angle_of_triangle (p q r : ℝ) (h1 : p + 3*q + 3*r = p^2) (h2 : p + 3*q - 3*r = -1) :
  ∃ (P Q R : ℝ), 
    P + Q + R = 180 ∧ 
    0 < P ∧ 0 < Q ∧ 0 < R ∧
    P ≤ 120 ∧ Q ≤ 120 ∧ R = 120 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_of_triangle_l1173_117393


namespace NUMINAMATH_CALUDE_three_mn_odd_l1173_117357

theorem three_mn_odd (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  (3 * m * n) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_mn_odd_l1173_117357


namespace NUMINAMATH_CALUDE_calculation_proof_l1173_117364

theorem calculation_proof :
  (1) * (Real.pi - 3.14) ^ 0 - |2 - Real.sqrt 3| + (-1/2)^2 = Real.sqrt 3 - 3/4 ∧
  Real.sqrt (1/3) + Real.sqrt 6 * (1/Real.sqrt 2 + Real.sqrt 8) = 16 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1173_117364


namespace NUMINAMATH_CALUDE_vector_collinearity_implies_x_value_l1173_117383

theorem vector_collinearity_implies_x_value (x : ℝ) 
  (hx : x > 0) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (8, x/2))
  (hb : b = (x, 1))
  (hcollinear : ∃ (k : ℝ), k ≠ 0 ∧ (a - 2 • b) = k • (2 • a + b)) :
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_implies_x_value_l1173_117383


namespace NUMINAMATH_CALUDE_milburg_children_count_l1173_117319

/-- The number of children in Milburg -/
def children_count (total_population grown_ups : ℕ) : ℕ :=
  total_population - grown_ups

/-- Theorem stating the number of children in Milburg -/
theorem milburg_children_count :
  children_count 8243 5256 = 2987 := by
  sorry

end NUMINAMATH_CALUDE_milburg_children_count_l1173_117319


namespace NUMINAMATH_CALUDE_correct_non_attacking_placements_non_attacking_placements_positive_l1173_117398

/-- Represents a chess piece type -/
inductive ChessPiece
  | Rook
  | King
  | Bishop
  | Knight
  | Queen

/-- Represents the dimensions of a chessboard -/
def BoardSize : Nat := 8

/-- Calculates the number of ways to place two pieces of the same type on a chessboard without attacking each other -/
def nonAttackingPlacements (piece : ChessPiece) : Nat :=
  match piece with
  | ChessPiece.Rook => 1568
  | ChessPiece.King => 1806
  | ChessPiece.Bishop => 1736
  | ChessPiece.Knight => 1848
  | ChessPiece.Queen => 1288

/-- Theorem stating the correct number of non-attacking placements for each piece type -/
theorem correct_non_attacking_placements :
  (nonAttackingPlacements ChessPiece.Rook = 1568) ∧
  (nonAttackingPlacements ChessPiece.King = 1806) ∧
  (nonAttackingPlacements ChessPiece.Bishop = 1736) ∧
  (nonAttackingPlacements ChessPiece.Knight = 1848) ∧
  (nonAttackingPlacements ChessPiece.Queen = 1288) := by
  sorry

/-- Theorem stating that the number of non-attacking placements is always positive -/
theorem non_attacking_placements_positive (piece : ChessPiece) :
  nonAttackingPlacements piece > 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_non_attacking_placements_non_attacking_placements_positive_l1173_117398


namespace NUMINAMATH_CALUDE_relationship_correctness_l1173_117380

theorem relationship_correctness :
  (∃ a b c : ℝ, (a > b ↔ a * c^2 > b * c^2) → False) ∧
  (∃ a b : ℝ, (a > b → 1/a < 1/b) → False) ∧
  (∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c > d → a/d > b/c) ∧
  (∃ a b c : ℝ, (a > b ∧ b > 0 → a^c < b^c) → False) :=
by sorry

end NUMINAMATH_CALUDE_relationship_correctness_l1173_117380


namespace NUMINAMATH_CALUDE_equation_solutions_l1173_117333

theorem equation_solutions (x : ℝ) : 
  x ≠ -2 → 
  ((16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 48) ↔ 
  (x = 1.2 ∨ x = -81.2) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1173_117333


namespace NUMINAMATH_CALUDE_negation_equivalence_l1173_117368

theorem negation_equivalence : 
  (¬(∀ x : ℝ, |x| < 2 → x < 2)) ↔ (∀ x : ℝ, |x| ≥ 2 → x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1173_117368


namespace NUMINAMATH_CALUDE_pencil_count_l1173_117324

theorem pencil_count :
  ∀ (pens pencils : ℕ),
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 4 →
  pencils = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1173_117324


namespace NUMINAMATH_CALUDE_ceiling_floor_expression_l1173_117358

theorem ceiling_floor_expression : 
  ⌈(7:ℝ)/3⌉ + ⌊-(7:ℝ)/3⌋ - ⌈(2:ℝ)/3⌉ = -1 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_expression_l1173_117358


namespace NUMINAMATH_CALUDE_one_absent_one_present_probability_l1173_117397

/-- The probability of a student being absent on any given day -/
def absent_prob : ℚ := 1 / 30

/-- The probability of a student being present on any given day -/
def present_prob : ℚ := 1 - absent_prob

/-- The probability that out of two randomly chosen students, exactly one is absent while the other is present -/
def one_absent_one_present_prob : ℚ := 2 * (absent_prob * present_prob)

theorem one_absent_one_present_probability :
  one_absent_one_present_prob = 58 / 900 := by sorry

end NUMINAMATH_CALUDE_one_absent_one_present_probability_l1173_117397


namespace NUMINAMATH_CALUDE_one_fourth_of_six_point_eight_l1173_117343

theorem one_fourth_of_six_point_eight : (1 / 4 : ℚ) * (68 / 10 : ℚ) = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_six_point_eight_l1173_117343


namespace NUMINAMATH_CALUDE_custom_op_three_six_l1173_117382

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a.val ^ 2 * b.val) / (a.val + b.val)

/-- Theorem stating that 3 @ 6 = 6 -/
theorem custom_op_three_six :
  custom_op 3 6 = 6 := by sorry

end NUMINAMATH_CALUDE_custom_op_three_six_l1173_117382


namespace NUMINAMATH_CALUDE_vegetarian_gluten_free_fraction_is_one_twentyfifth_l1173_117352

/-- Represents the menu of a restaurant --/
structure Menu where
  total_dishes : ℕ
  vegetarian_dishes : ℕ
  gluten_free_vegetarian_dishes : ℕ

/-- The fraction of dishes that are both vegetarian and gluten-free --/
def vegetarian_gluten_free_fraction (menu : Menu) : ℚ :=
  menu.gluten_free_vegetarian_dishes / menu.total_dishes

/-- Theorem stating the fraction of vegetarian and gluten-free dishes --/
theorem vegetarian_gluten_free_fraction_is_one_twentyfifth 
  (menu : Menu) 
  (h1 : menu.vegetarian_dishes = 5)
  (h2 : menu.vegetarian_dishes = menu.total_dishes / 5)
  (h3 : menu.gluten_free_vegetarian_dishes = menu.vegetarian_dishes - 4) :
  vegetarian_gluten_free_fraction menu = 1 / 25 := by
  sorry

#check vegetarian_gluten_free_fraction_is_one_twentyfifth

end NUMINAMATH_CALUDE_vegetarian_gluten_free_fraction_is_one_twentyfifth_l1173_117352


namespace NUMINAMATH_CALUDE_systematic_sample_selection_l1173_117307

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Checks if a number is selected in a systematic sample -/
def is_selected (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.total

theorem systematic_sample_selection 
  (s : SystematicSample)
  (h_total : s.total = 900)
  (h_size : s.sample_size = 150)
  (h_start : s.start = 15)
  (h_interval : s.interval = s.total / s.sample_size)
  (h_15_selected : is_selected s 15)
  : is_selected s 81 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_selection_l1173_117307


namespace NUMINAMATH_CALUDE_other_number_proof_l1173_117377

/-- Given two positive integers with specified HCF and LCM, prove that if one number is 36, the other is 176. -/
theorem other_number_proof (A B : ℕ+) : 
  Nat.gcd A B = 16 →
  Nat.lcm A B = 396 →
  A = 36 →
  B = 176 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l1173_117377


namespace NUMINAMATH_CALUDE_jessie_friends_l1173_117360

/-- The number of friends Jessie invited -/
def num_friends (total_muffins : ℕ) (muffins_per_person : ℕ) : ℕ :=
  total_muffins / muffins_per_person - 1

/-- Theorem stating that Jessie invited 4 friends -/
theorem jessie_friends : num_friends 20 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jessie_friends_l1173_117360


namespace NUMINAMATH_CALUDE_artist_painting_rate_l1173_117389

/-- Proves that given the specified conditions, the artist can paint 1.5 square meters per hour -/
theorem artist_painting_rate 
  (mural_length : ℝ) 
  (mural_width : ℝ) 
  (paint_cost_per_sqm : ℝ) 
  (artist_hourly_rate : ℝ) 
  (total_mural_cost : ℝ) 
  (h1 : mural_length = 6) 
  (h2 : mural_width = 3) 
  (h3 : paint_cost_per_sqm = 4) 
  (h4 : artist_hourly_rate = 10) 
  (h5 : total_mural_cost = 192) : 
  (mural_length * mural_width) / ((total_mural_cost - (paint_cost_per_sqm * mural_length * mural_width)) / artist_hourly_rate) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_artist_painting_rate_l1173_117389


namespace NUMINAMATH_CALUDE_linear_arrangement_paths_count_l1173_117316

/-- Represents a linear arrangement of nodes -/
structure LinearArrangement (n : ℕ) where
  nodes : Fin n → ℕ

/-- Counts the number of paths of a given length in a linear arrangement -/
def countPaths (arr : LinearArrangement 10) (length : ℕ) : ℕ :=
  sorry

/-- Theorem: The number of paths of length 4 in a linear arrangement of 10 nodes is 2304 -/
theorem linear_arrangement_paths_count :
  ∀ (arr : LinearArrangement 10), countPaths arr 4 = 2304 := by
  sorry

end NUMINAMATH_CALUDE_linear_arrangement_paths_count_l1173_117316


namespace NUMINAMATH_CALUDE_point_q_coordinates_l1173_117379

/-- A point on the unit circle --/
structure PointOnUnitCircle where
  x : ℝ
  y : ℝ
  on_circle : x^2 + y^2 = 1

/-- The arc length between two points on the unit circle --/
def arcLength (p q : PointOnUnitCircle) : ℝ := sorry

theorem point_q_coordinates :
  ∀ (p q : PointOnUnitCircle),
  p.x = 1 ∧ p.y = 0 →
  arcLength p q = π / 3 →
  q.x = 1 / 2 ∧ q.y = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_point_q_coordinates_l1173_117379


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l1173_117309

/-- Represents the modified cube structure --/
structure ModifiedCube where
  initialSize : Nat
  smallCubeSize : Nat
  removedCubes : Nat
  cornerSize : Nat

/-- Calculates the surface area of the modified cube structure --/
def surfaceArea (c : ModifiedCube) : Nat :=
  let remainingCubes := 27 - c.removedCubes
  let visibleCornersPerCube := 4
  let surfaceUnitsPerCorner := 3
  remainingCubes * visibleCornersPerCube * surfaceUnitsPerCorner

/-- The theorem to be proved --/
theorem modified_cube_surface_area :
  ∀ (c : ModifiedCube),
  c.initialSize = 6 ∧
  c.smallCubeSize = 2 ∧
  c.removedCubes = 7 ∧
  c.cornerSize = 1 →
  surfaceArea c = 240 := by
  sorry


end NUMINAMATH_CALUDE_modified_cube_surface_area_l1173_117309


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l1173_117310

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_perfect_square_factors (a b c d : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1) * (d + 1)

theorem perfect_square_factors_count :
  count_perfect_square_factors 6 7 8 4 = 2520 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l1173_117310


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1173_117311

theorem smaller_number_proof (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : 
  min a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1173_117311


namespace NUMINAMATH_CALUDE_ball_distribution_ratio_l1173_117337

/-- The number of ways to distribute n identical objects into k distinct bins --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n identical objects into k distinct bins,
    where each bin receives a specified number of objects --/
def distributeSpecific (n k : ℕ) (bins : Fin k → ℕ) : ℕ := sorry

theorem ball_distribution_ratio : 
  let total_balls : ℕ := 15
  let num_bins : ℕ := 4
  let pattern1 : Fin num_bins → ℕ := ![3, 6, 3, 3]
  let pattern2 : Fin num_bins → ℕ := ![3, 2, 3, 7]
  
  (distributeSpecific total_balls num_bins pattern1) / 
  (distributeSpecific total_balls num_bins pattern2) = 560 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_ratio_l1173_117337


namespace NUMINAMATH_CALUDE_q_must_be_true_l1173_117396

theorem q_must_be_true (h1 : ¬p) (h2 : p ∨ q) : q :=
sorry

end NUMINAMATH_CALUDE_q_must_be_true_l1173_117396


namespace NUMINAMATH_CALUDE_correct_guess_probability_l1173_117302

/-- The number of possible digits in a phone number -/
def num_digits : ℕ := 10

/-- The length of a phone number -/
def phone_number_length : ℕ := 7

/-- The probability of correctly guessing a single unknown digit in a phone number -/
def probability_correct_guess : ℚ := 1 / num_digits

theorem correct_guess_probability : 
  probability_correct_guess = 1 / num_digits :=
by sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l1173_117302


namespace NUMINAMATH_CALUDE_evaluate_expression_l1173_117336

theorem evaluate_expression : 5 - 7 * (8 - 12 / (3^2)) * 6 = -275 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1173_117336


namespace NUMINAMATH_CALUDE_ab_equality_l1173_117300

theorem ab_equality (a b : ℝ) : 2 * a * b + 3 * b * a = 5 * a * b := by
  sorry

end NUMINAMATH_CALUDE_ab_equality_l1173_117300


namespace NUMINAMATH_CALUDE_betty_sugar_purchase_l1173_117303

def min_sugar_purchase (s : ℕ) : Prop :=
  ∃ (f : ℝ),
    f ≥ 4 + s / 3 ∧
    f ≤ 3 * s ∧
    2 * s + 3 * f ≤ 36 ∧
    ∀ (s' : ℕ), s' < s → ¬∃ (f' : ℝ),
      f' ≥ 4 + s' / 3 ∧
      f' ≤ 3 * s' ∧
      2 * s' + 3 * f' ≤ 36

theorem betty_sugar_purchase : min_sugar_purchase 4 :=
sorry

end NUMINAMATH_CALUDE_betty_sugar_purchase_l1173_117303


namespace NUMINAMATH_CALUDE_polynomial_characterization_l1173_117325

/-- A homogeneous polynomial of degree n in two variables -/
noncomputable def HomogeneousPolynomial (n : ℕ) := (ℝ → ℝ → ℝ)

/-- The property of being homogeneous of degree n -/
def IsHomogeneous (P : HomogeneousPolynomial n) : Prop :=
  ∀ (t x y : ℝ), P (t * x) (t * y) = t^n * P x y

/-- The second condition from the problem -/
def SatisfiesCondition2 (P : HomogeneousPolynomial n) : Prop :=
  ∀ (a b c : ℝ), P (a + b) c + P (b + c) a + P (c + a) b = 0

/-- The third condition from the problem -/
def SatisfiesCondition3 (P : HomogeneousPolynomial n) : Prop :=
  P 1 0 = 1

/-- The theorem statement -/
theorem polynomial_characterization (n : ℕ) (P : HomogeneousPolynomial n)
  (h1 : IsHomogeneous P)
  (h2 : SatisfiesCondition2 P)
  (h3 : SatisfiesCondition3 P) :
  ∀ (x y : ℝ), P x y = (x + y)^(n - 1) * (x - 2*y) :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l1173_117325


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l1173_117374

theorem intersection_implies_m_equals_one :
  ∀ m : ℝ, ({3, 4, m^2 - 3*m - 1} ∩ {2*m, -3} : Set ℝ) = {-3} → m = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l1173_117374


namespace NUMINAMATH_CALUDE_min_M_and_F_M_l1173_117322

def is_k_multiple (n : ℕ) (k : ℤ) : Prop :=
  let thousands := n / 1000
  let hundreds := (n / 100) % 10
  let tens := (n / 10) % 10
  let units := n % 10
  (thousands + hundreds : ℤ) = k * (tens - units)

def swap_hundreds_tens (n : ℕ) : ℕ :=
  let thousands := n / 1000
  let hundreds := (n / 100) % 10
  let tens := (n / 10) % 10
  let units := n % 10
  thousands * 1000 + tens * 100 + hundreds * 10 + units

def F (m : ℕ) : ℚ :=
  let a : ℕ := (m + 1) / 2
  let b : ℕ := (m - 1) / 2
  (a : ℚ) / b

theorem min_M_and_F_M :
  ∃ (M : ℕ),
    M ≥ 1000 ∧ M < 10000 ∧
    is_k_multiple M 4 ∧
    is_k_multiple (M - 4) (-3) ∧
    is_k_multiple (swap_hundreds_tens M) 4 ∧
    (∀ (N : ℕ), N ≥ 1000 ∧ N < 10000 ∧
      is_k_multiple N 4 ∧
      is_k_multiple (N - 4) (-3) ∧
      is_k_multiple (swap_hundreds_tens N) 4 →
      M ≤ N) ∧
    M = 6663 ∧
    F M = 3332 / 3331 := by sorry

end NUMINAMATH_CALUDE_min_M_and_F_M_l1173_117322


namespace NUMINAMATH_CALUDE_edward_money_problem_l1173_117345

/-- Proves that if a person spends $17, then receives $10, and ends up with $7, they must have started with $14. -/
theorem edward_money_problem (initial_amount spent received final_amount : ℤ) :
  spent = 17 →
  received = 10 →
  final_amount = 7 →
  initial_amount - spent + received = final_amount →
  initial_amount = 14 :=
by sorry

end NUMINAMATH_CALUDE_edward_money_problem_l1173_117345


namespace NUMINAMATH_CALUDE_zero_point_implies_a_range_l1173_117331

theorem zero_point_implies_a_range (a : ℝ) : 
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, a * x + 1 - 2 * a = 0) → 
  a ∈ Set.Ioo (1/3 : ℝ) 1 := by
sorry

end NUMINAMATH_CALUDE_zero_point_implies_a_range_l1173_117331


namespace NUMINAMATH_CALUDE_log_base_value_log_inequality_greater_than_one_log_inequality_less_than_one_l1173_117399

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem 1
theorem log_base_value (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  log a 8 = 3 → a = 2 := by sorry

-- Theorem 2
theorem log_inequality_greater_than_one (a : ℝ) (ha : a > 1) (x : ℝ) :
  log a x ≤ log a (2 - 3*x) ↔ (0 < x ∧ x ≤ 1/2) := by sorry

-- Theorem 3
theorem log_inequality_less_than_one (a : ℝ) (ha : 0 < a ∧ a < 1) (x : ℝ) :
  log a x ≤ log a (2 - 3*x) ↔ (1/2 ≤ x ∧ x < 2/3) := by sorry

end NUMINAMATH_CALUDE_log_base_value_log_inequality_greater_than_one_log_inequality_less_than_one_l1173_117399


namespace NUMINAMATH_CALUDE_park_fencing_cost_l1173_117320

/-- The cost of fencing one side of the square park -/
def cost_per_side : ℕ := 56

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- The total cost of fencing the square park -/
def total_cost : ℕ := cost_per_side * num_sides

theorem park_fencing_cost : total_cost = 224 := by
  sorry

end NUMINAMATH_CALUDE_park_fencing_cost_l1173_117320


namespace NUMINAMATH_CALUDE_popcorn_shrimp_orders_l1173_117388

/-- Proves that the number of popcorn shrimp orders is 9 given the conditions -/
theorem popcorn_shrimp_orders 
  (catfish_cost : ℝ) 
  (shrimp_cost : ℝ) 
  (total_orders : ℕ) 
  (total_amount : ℝ) 
  (h1 : catfish_cost = 6)
  (h2 : shrimp_cost = 3.5)
  (h3 : total_orders = 26)
  (h4 : total_amount = 133.5) :
  ∃ (catfish_orders shrimp_orders : ℕ), 
    catfish_orders + shrimp_orders = total_orders ∧ 
    catfish_cost * (catfish_orders : ℝ) + shrimp_cost * (shrimp_orders : ℝ) = total_amount ∧
    shrimp_orders = 9 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_shrimp_orders_l1173_117388


namespace NUMINAMATH_CALUDE_gcd_153_119_l1173_117391

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_119_l1173_117391


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1173_117387

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_implies_a_value :
  ∃ (a : ℝ), A a ∩ B a = {9} → a = -3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1173_117387


namespace NUMINAMATH_CALUDE_complex_power_eight_l1173_117384

theorem complex_power_eight :
  (3 * (Complex.cos (π / 6) + Complex.I * Complex.sin (π / 6))) ^ 8 =
  Complex.mk (-3280.5) (-3280.5 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eight_l1173_117384


namespace NUMINAMATH_CALUDE_special_function_value_l1173_117348

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 2

/-- The main theorem -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) (h₀ : f 1 = 0) : 
  f 2010 = 4018 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l1173_117348


namespace NUMINAMATH_CALUDE_average_of_multiples_of_four_is_even_l1173_117305

theorem average_of_multiples_of_four_is_even (m n : ℤ) : 
  ∃ k : ℤ, (4*m + 4*n) / 2 = 2*k := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_four_is_even_l1173_117305


namespace NUMINAMATH_CALUDE_kims_morning_routine_l1173_117301

/-- Kim's morning routine calculation -/
theorem kims_morning_routine (coffee_time : ℕ) (status_update_time : ℕ) (payroll_update_time : ℕ) (num_employees : ℕ) :
  coffee_time = 5 →
  status_update_time = 2 →
  payroll_update_time = 3 →
  num_employees = 9 →
  coffee_time + num_employees * (status_update_time + payroll_update_time) = 50 := by
  sorry

#check kims_morning_routine

end NUMINAMATH_CALUDE_kims_morning_routine_l1173_117301
