import Mathlib

namespace NUMINAMATH_CALUDE_painted_cube_problem_l3305_330511

theorem painted_cube_problem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 4 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l3305_330511


namespace NUMINAMATH_CALUDE_expand_expression_l3305_330506

theorem expand_expression (x : ℝ) : (1 + x^2) * (1 - x^4) = 1 + x^2 - x^4 - x^6 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3305_330506


namespace NUMINAMATH_CALUDE_robins_hair_growth_l3305_330547

/-- Calculates the hair growth given initial length, final length, and cut length -/
def hair_growth (initial_length final_length cut_length : ℕ) : ℕ :=
  cut_length + final_length - initial_length

/-- Theorem: Given Robin's hair scenario, the hair growth is 8 inches -/
theorem robins_hair_growth :
  hair_growth 14 2 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_growth_l3305_330547


namespace NUMINAMATH_CALUDE_book_cost_calculation_l3305_330596

def total_cost : ℝ := 6
def num_books : ℕ := 2

theorem book_cost_calculation :
  (total_cost / num_books : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l3305_330596


namespace NUMINAMATH_CALUDE_prob_three_unused_correct_expected_hits_correct_l3305_330589

-- Define the probability of hitting a target with a single shot
variable (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1)

-- Define the number of rockets and targets
def num_rockets : ℕ := 10
def num_targets_a : ℕ := 5
def num_targets_b : ℕ := 9

-- Part (a): Probability of exactly three unused rockets
def prob_three_unused : ℝ := 10 * p^3 * (1-p)^2

-- Part (b): Expected number of targets hit
def expected_hits : ℝ := 10*p - p^10

-- Theorem statements
theorem prob_three_unused_correct :
  prob_three_unused p = 10 * p^3 * (1-p)^2 :=
sorry

theorem expected_hits_correct :
  expected_hits p = 10*p - p^10 :=
sorry

end NUMINAMATH_CALUDE_prob_three_unused_correct_expected_hits_correct_l3305_330589


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3305_330505

/-- Represents the number of exam papers checked in a school -/
structure SchoolSample where
  total : ℕ
  sampled : ℕ

/-- Calculates the total number of exam papers checked across all schools -/
def totalSampled (schools : List SchoolSample) : ℕ :=
  schools.map (fun s => s.sampled) |>.sum

theorem stratified_sampling_theorem (schoolA schoolB schoolC : SchoolSample) :
  schoolA.total = 1260 →
  schoolB.total = 720 →
  schoolC.total = 900 →
  schoolC.sampled = 45 →
  schoolA.sampled = schoolA.total / (schoolC.total / schoolC.sampled) →
  schoolB.sampled = schoolB.total / (schoolC.total / schoolC.sampled) →
  totalSampled [schoolA, schoolB, schoolC] = 144 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3305_330505


namespace NUMINAMATH_CALUDE_function_equation_solution_l3305_330500

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * y^2) →
  (∀ x : ℝ, f x = x^2 ∨ f x = -x^2) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3305_330500


namespace NUMINAMATH_CALUDE_combination_problem_l3305_330512

theorem combination_problem (n : ℕ) (h : Nat.choose n 13 = Nat.choose n 7) :
  Nat.choose n 2 = 190 := by
  sorry

end NUMINAMATH_CALUDE_combination_problem_l3305_330512


namespace NUMINAMATH_CALUDE_min_value_problem_l3305_330551

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  (4*x/(x-1)) + (9*y/(y-1)) ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), (4*x₀/(x₀-1)) + (9*y₀/(y₀-1)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3305_330551


namespace NUMINAMATH_CALUDE_stating_six_suitcases_attempts_stating_ten_suitcases_attempts_l3305_330549

/-- 
Given n suitcases and n keys, where it is unknown which key opens which suitcase,
this function calculates the minimum number of attempts needed to ensure all suitcases are opened.
-/
def minAttempts (n : ℕ) : ℕ := (n - 1) * n / 2

/-- 
Theorem stating that for 6 suitcases and 6 keys, the minimum number of attempts is 15.
-/
theorem six_suitcases_attempts : minAttempts 6 = 15 := by sorry

/-- 
Theorem stating that for 10 suitcases and 10 keys, the minimum number of attempts is 45.
-/
theorem ten_suitcases_attempts : minAttempts 10 = 45 := by sorry

end NUMINAMATH_CALUDE_stating_six_suitcases_attempts_stating_ten_suitcases_attempts_l3305_330549


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3305_330532

/-- Given a principal amount where the compound interest for 2 years at 5% per annum is 51.25,
    prove that the simple interest for the same period and rate is 250. -/
theorem simple_interest_calculation (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 51.25 → P * 0.05 * 2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3305_330532


namespace NUMINAMATH_CALUDE_absolute_value_plus_exponent_l3305_330568

theorem absolute_value_plus_exponent : |-4| + (3 - Real.pi)^0 = 5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_plus_exponent_l3305_330568


namespace NUMINAMATH_CALUDE_solve_equation_l3305_330591

theorem solve_equation :
  ∃ y : ℚ, 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3305_330591


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3305_330561

/-- The trajectory of the midpoint between a moving point on the unit circle and the fixed point (3, 0) -/
theorem midpoint_trajectory :
  ∀ (a b x y : ℝ),
  a^2 + b^2 = 1 →  -- point (a, b) is on the unit circle
  x = (a + 3) / 2 →  -- x-coordinate of midpoint
  y = b / 2 →  -- y-coordinate of midpoint
  x^2 + y^2 - 3*x + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3305_330561


namespace NUMINAMATH_CALUDE_uneaten_fish_l3305_330557

def fish_cells : List Nat := [3, 4, 16, 12, 20, 6]

def cat_eating_rate : Nat := 3

theorem uneaten_fish (eaten_count : Nat) (total_time : Nat) :
  eaten_count = 5 →
  total_time * cat_eating_rate = (fish_cells.take eaten_count).sum →
  total_time > 0 →
  (fish_cells.take eaten_count).sum % cat_eating_rate = 1 →
  fish_cells[eaten_count]! = 6 := by
  sorry

end NUMINAMATH_CALUDE_uneaten_fish_l3305_330557


namespace NUMINAMATH_CALUDE_sum_21_implies_n_6_l3305_330562

/-- Represents a sequence where a₁ = 1 and aₙ₊₁ = aₙ + 1 -/
def ArithmeticSequence (n : ℕ) : ℕ :=
  n

/-- Sum of the first n terms of the arithmetic sequence -/
def Sn (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Theorem: If Sn = 21, then n = 6 -/
theorem sum_21_implies_n_6 : Sn 6 = 21 :=
  by sorry

end NUMINAMATH_CALUDE_sum_21_implies_n_6_l3305_330562


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_1789_units_digit_of_1777_pow_1777_pow_1777_l3305_330519

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem for the first part
theorem units_digit_of_3_pow_1789 :
  unitsDigit (3^1789) = 3 := by sorry

-- Theorem for the second part
theorem units_digit_of_1777_pow_1777_pow_1777 :
  unitsDigit (1777^(1777^1777)) = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_1789_units_digit_of_1777_pow_1777_pow_1777_l3305_330519


namespace NUMINAMATH_CALUDE_digit_cube_equals_square_l3305_330526

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem digit_cube_equals_square (n : Nat) : 
  n ∈ Finset.range 1000 → (n^2 = (sum_of_digits n)^3 ↔ n = 1 ∨ n = 27) := by
  sorry

end NUMINAMATH_CALUDE_digit_cube_equals_square_l3305_330526


namespace NUMINAMATH_CALUDE_hawks_percentage_l3305_330529

/-- Represents the percentages of different bird types in a nature reserve -/
structure BirdReserve where
  hawks : ℝ
  paddyfieldWarblers : ℝ
  kingfishers : ℝ
  others : ℝ

/-- The conditions of the bird reserve problem -/
def validBirdReserve (b : BirdReserve) : Prop :=
  b.paddyfieldWarblers = 0.4 * (100 - b.hawks) ∧
  b.kingfishers = 0.25 * b.paddyfieldWarblers ∧
  b.others = 35 ∧
  b.hawks + b.paddyfieldWarblers + b.kingfishers + b.others = 100

/-- The theorem stating that hawks make up 30% of the birds in a valid bird reserve -/
theorem hawks_percentage (b : BirdReserve) (h : validBirdReserve b) : b.hawks = 30 := by
  sorry

end NUMINAMATH_CALUDE_hawks_percentage_l3305_330529


namespace NUMINAMATH_CALUDE_function_inequality_l3305_330533

theorem function_inequality (f : ℤ → ℤ) 
  (h1 : ∀ k : ℤ, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (h2 : f 4 = 25) : 
  ∀ k : ℤ, k ≥ 4 → f k ≥ k^2 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l3305_330533


namespace NUMINAMATH_CALUDE_solve_inequality_find_a_range_l3305_330566

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
def g (a : ℝ) : ℝ := a^2 - a - 2

-- Theorem for the first part of the problem
theorem solve_inequality (x : ℝ) :
  f x 3 > g 3 + 2 ↔ x < -4 ∨ x > 2 := by sorry

-- Theorem for the second part of the problem
theorem find_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-a) 1, f x a ≤ g a) → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_find_a_range_l3305_330566


namespace NUMINAMATH_CALUDE_minimum_gloves_needed_l3305_330553

theorem minimum_gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) : 
  participants = 43 → gloves_per_participant = 2 → participants * gloves_per_participant = 86 := by
  sorry

end NUMINAMATH_CALUDE_minimum_gloves_needed_l3305_330553


namespace NUMINAMATH_CALUDE_pioneer_camp_group_l3305_330573

theorem pioneer_camp_group (x y z w : ℕ) : 
  x + y + z + w = 23 →
  10 * x + 11 * y + 12 * z + 13 * w = 253 →
  z = (3 : ℕ) / 2 * w →
  z = 6 := by
  sorry

end NUMINAMATH_CALUDE_pioneer_camp_group_l3305_330573


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l3305_330585

/-- Two vectors are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- The problem statement -/
theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (2, 4) (x, 6) → x = 3 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l3305_330585


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3305_330517

-- Problem 1
theorem problem_1 (x : ℝ) : x * (x + 6) + (x - 3)^2 = 2 * x^2 + 9 := by
  sorry

-- Problem 2
theorem problem_2 (m n : ℝ) (hm : m ≠ 0) (hmn : 3 * m ≠ n) :
  (3 + n / m) / ((9 * m^2 - n^2) / m) = 1 / (3 * m - n) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3305_330517


namespace NUMINAMATH_CALUDE_system_solution_unique_l3305_330548

theorem system_solution_unique : 
  ∃! (x y : ℝ), (x - 2*y = 0) ∧ (3*x + 2*y = 8) ∧ (x = 2) ∧ (y = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3305_330548


namespace NUMINAMATH_CALUDE_min_value_expression_l3305_330560

theorem min_value_expression (x y z : ℝ) 
  (hx : -1/2 ≤ x ∧ x ≤ 1/2) 
  (hy : -1/2 ≤ y ∧ y ≤ 1/2) 
  (hz : -1/2 ≤ z ∧ z ≤ 1/2) : 
  (1/((1 - x^2)*(1 - y^2)*(1 - z^2))) + (1/((1 + x^2)*(1 + y^2)*(1 + z^2))) ≥ 2 ∧
  (1/((1 - 0^2)*(1 - 0^2)*(1 - 0^2))) + (1/((1 + 0^2)*(1 + 0^2)*(1 + 0^2))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3305_330560


namespace NUMINAMATH_CALUDE_triangle_properties_l3305_330507

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  2 * t.a * Real.cos t.A = t.b * Real.cos t.C + t.c * Real.cos t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : given_condition t) : 
  t.A = π / 3 ∧ 
  ∀ x, x ∈ Set.Icc (-1 : ℝ) (-1/2) ↔ 
    ∃ (B C : ℝ), t.B = B ∧ t.C = C ∧ x = Real.cos B - Real.sqrt 3 * Real.sin C :=
sorry


end NUMINAMATH_CALUDE_triangle_properties_l3305_330507


namespace NUMINAMATH_CALUDE_sqrt_inequality_at_least_one_positive_l3305_330558

-- Problem 1
theorem sqrt_inequality (a : ℝ) (h : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

-- Problem 2
theorem at_least_one_positive (x y z : ℝ) :
  let a := x^2 - 2*y + π/2
  let b := y^2 - 2*z + π/3
  let c := z^2 - 2*x + π/6
  max a (max b c) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_at_least_one_positive_l3305_330558


namespace NUMINAMATH_CALUDE_solve_equation_l3305_330542

theorem solve_equation (y : ℚ) (h : (2 / 7) * (1 / 5) * y = 4) : y = 70 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3305_330542


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3305_330530

noncomputable def f (x : ℝ) := Real.sin x - Real.cos x + x + 1

theorem f_monotonicity_and_extrema :
  ∀ x : ℝ, 0 < x → x < 2 * Real.pi →
  (∀ y : ℝ, 0 < y → y < Real.pi → HasDerivAt f (Real.cos y + Real.sin y + 1) y) ∧
  (∀ y : ℝ, Real.pi < y → y < 3 * Real.pi / 2 → HasDerivAt f (Real.cos y + Real.sin y + 1) y) ∧
  (∀ y : ℝ, 3 * Real.pi / 2 < y → y < 2 * Real.pi → HasDerivAt f (Real.cos y + Real.sin y + 1) y) ∧
  (f (3 * Real.pi / 2) = 3 * Real.pi / 2) ∧
  (f Real.pi = Real.pi + 2) ∧
  (∀ y : ℝ, 0 < y → y < 2 * Real.pi → f y ≥ 3 * Real.pi / 2) ∧
  (∀ y : ℝ, 0 < y → y < 2 * Real.pi → f y ≤ Real.pi + 2) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3305_330530


namespace NUMINAMATH_CALUDE_shaded_squares_percentage_l3305_330571

/-- Given a 5x5 grid with 9 shaded squares, the percentage of shaded squares is 36%. -/
theorem shaded_squares_percentage :
  ∀ (total_squares shaded_squares : ℕ),
    total_squares = 5 * 5 →
    shaded_squares = 9 →
    (shaded_squares : ℚ) / total_squares * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_shaded_squares_percentage_l3305_330571


namespace NUMINAMATH_CALUDE_choose_starters_with_twins_l3305_330580

def total_players : ℕ := 12
def twin_players : ℕ := 2
def starters : ℕ := 5

theorem choose_starters_with_twins :
  (total_players.choose starters) = (total_players - twin_players).choose (starters - twin_players) :=
sorry

end NUMINAMATH_CALUDE_choose_starters_with_twins_l3305_330580


namespace NUMINAMATH_CALUDE_pirate_game_solution_l3305_330567

def pirate_game (initial_coins : ℕ) : (ℕ × ℕ) :=
  let after_first_transfer := (initial_coins / 2, initial_coins + initial_coins / 2)
  let after_second_transfer := (after_first_transfer.1 + after_first_transfer.2 / 2, after_first_transfer.2 / 2)
  (after_second_transfer.1 / 2, after_second_transfer.2 + after_second_transfer.1 / 2)

theorem pirate_game_solution :
  ∃ (x : ℕ), pirate_game x = (15, 33) ∧ x = 24 :=
sorry

end NUMINAMATH_CALUDE_pirate_game_solution_l3305_330567


namespace NUMINAMATH_CALUDE_triangular_sum_perfect_squares_l3305_330575

def triangular_sum (K : ℕ) : ℕ := K * (K + 1) / 2

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem triangular_sum_perfect_squares :
  {K : ℕ | K > 0 ∧ K < 50 ∧ is_perfect_square (triangular_sum K)} = {1, 8, 49} := by
sorry

end NUMINAMATH_CALUDE_triangular_sum_perfect_squares_l3305_330575


namespace NUMINAMATH_CALUDE_harmonious_number_properties_l3305_330540

/-- A harmonious number is a three-digit number where the tens digit 
    is equal to the sum of its units digit and hundreds digit. -/
def is_harmonious (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ 
  (n / 10 % 10 = n % 10 + n / 100)

/-- The smallest harmonious number -/
def smallest_harmonious : ℕ := 110

/-- The largest harmonious number -/
def largest_harmonious : ℕ := 990

/-- Algebraic expression for a harmonious number -/
def harmonious_expression (a b : ℕ) : ℕ := 110 * b - 99 * a

theorem harmonious_number_properties :
  (∀ n : ℕ, is_harmonious n → smallest_harmonious ≤ n ∧ n ≤ largest_harmonious) ∧
  (∀ n : ℕ, is_harmonious n → 
    ∃ a b : ℕ, a ≥ 0 ∧ b ≥ 1 ∧ b > a ∧ 
    n = harmonious_expression a b) :=
sorry

end NUMINAMATH_CALUDE_harmonious_number_properties_l3305_330540


namespace NUMINAMATH_CALUDE_mass_percentage_h_in_water_l3305_330522

/-- The mass percentage of hydrogen in water, considering isotopic composition --/
theorem mass_percentage_h_in_water (h1_abundance : Real) (h2_abundance : Real)
  (h1_mass : Real) (h2_mass : Real) (o_mass : Real)
  (h1_abundance_val : h1_abundance = 0.9998)
  (h2_abundance_val : h2_abundance = 0.0002)
  (h1_mass_val : h1_mass = 1)
  (h2_mass_val : h2_mass = 2)
  (o_mass_val : o_mass = 16) :
  let avg_h_mass := h1_abundance * h1_mass + h2_abundance * h2_mass
  let water_mass := 2 * avg_h_mass + o_mass
  let mass_percentage := (2 * avg_h_mass) / water_mass * 100
  ∃ ε > 0, |mass_percentage - 11.113| < ε :=
sorry

end NUMINAMATH_CALUDE_mass_percentage_h_in_water_l3305_330522


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_l3305_330588

theorem sum_of_squares_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_l3305_330588


namespace NUMINAMATH_CALUDE_valid_sequences_count_l3305_330583

/-- A transformation on a regular hexagon -/
inductive HexagonTransform
| T1  -- 60° clockwise rotation
| T2  -- 60° counterclockwise rotation
| T3  -- reflection across x-axis
| T4  -- reflection across y-axis

/-- A sequence of transformations -/
def TransformSequence := List HexagonTransform

/-- The identity transformation -/
def identity : TransformSequence := []

/-- Applies a single transformation to a sequence -/
def applyTransform (t : HexagonTransform) (s : TransformSequence) : TransformSequence :=
  t :: s

/-- Checks if a sequence of transformations results in the identity transformation -/
def isIdentity (s : TransformSequence) : Bool :=
  sorry

/-- Counts the number of valid 18-transformation sequences -/
def countValidSequences : Nat :=
  sorry

/-- Main theorem: There are 286 valid sequences of 18 transformations -/
theorem valid_sequences_count : countValidSequences = 286 := by
  sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l3305_330583


namespace NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l3305_330523

theorem angle_with_supplement_four_times_complement : ∃ (x : ℝ), 
  x = 60 ∧ 
  (180 - x) = 4 * (90 - x) := by
  sorry

end NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l3305_330523


namespace NUMINAMATH_CALUDE_bella_steps_to_meet_l3305_330541

/-- The number of steps Bella takes when meeting Ella -/
def steps_to_meet (distance : ℕ) (speed_ratio : ℕ) (step_length : ℕ) : ℕ :=
  (distance * 2) / ((speed_ratio + 1) * step_length)

/-- Theorem stating that Bella takes 1056 steps to meet Ella under given conditions -/
theorem bella_steps_to_meet :
  steps_to_meet 15840 4 3 = 1056 :=
by sorry

end NUMINAMATH_CALUDE_bella_steps_to_meet_l3305_330541


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3305_330508

/-- Given a complex number z and a real number a, if |z| = 2 and (z - a)² = a, then a = 2 -/
theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.abs z = 2) 
  (h2 : (z - a)^2 = a) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3305_330508


namespace NUMINAMATH_CALUDE_complex_expression_equals_81_l3305_330502

theorem complex_expression_equals_81 :
  3 * ((-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_81_l3305_330502


namespace NUMINAMATH_CALUDE_train_crossing_time_l3305_330503

theorem train_crossing_time (train_length : ℝ) (platform1_length : ℝ) (platform2_length : ℝ) (time2 : ℝ) :
  train_length = 230 →
  platform1_length = 130 →
  platform2_length = 250 →
  time2 = 20 →
  let speed := (train_length + platform2_length) / time2
  let time1 := (train_length + platform1_length) / speed
  time1 = 15 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3305_330503


namespace NUMINAMATH_CALUDE_sign_sum_zero_l3305_330520

theorem sign_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_sum : a + b + c = 0) :
  a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_sum_zero_l3305_330520


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l3305_330525

theorem seventh_root_unity_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 10 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l3305_330525


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_multiple_l3305_330579

theorem smallest_number_divisible_by_multiple (x : ℕ) : x = 34 ↔ 
  (∀ y : ℕ, y < x → ¬(∃ k : ℕ, y - 10 = 2 * k ∧ y - 10 = 6 * k ∧ y - 10 = 12 * k ∧ y - 10 = 24 * k)) ∧
  (∃ k : ℕ, x - 10 = 2 * k ∧ x - 10 = 6 * k ∧ x - 10 = 12 * k ∧ x - 10 = 24 * k) :=
by sorry

#check smallest_number_divisible_by_multiple

end NUMINAMATH_CALUDE_smallest_number_divisible_by_multiple_l3305_330579


namespace NUMINAMATH_CALUDE_initial_amount_is_750_l3305_330581

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- Final amount calculation using simple interest -/
def final_amount (principal rate time : ℝ) : ℝ := principal + simple_interest principal rate time

/-- Theorem stating that given the conditions, the initial amount must be 750 -/
theorem initial_amount_is_750 :
  ∀ (P : ℝ),
  final_amount P 0.06 5 = 975 →
  P = 750 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_is_750_l3305_330581


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3305_330582

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (45 * q) * Real.sqrt (15 * q) * Real.sqrt (10 * q) = 30 * q * Real.sqrt (15 * q) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3305_330582


namespace NUMINAMATH_CALUDE_abc_right_triangle_l3305_330531

/-- Parabola defined by y^2 = 4x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

/-- Point A -/
def A : ℝ × ℝ := (1, 2)

/-- Point P -/
def P : ℝ × ℝ := (5, -2)

/-- B and C are on the parabola -/
def on_parabola (B C : ℝ × ℝ) : Prop := parabola B ∧ parabola C

/-- Line BC passes through P -/
def line_through_P (B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, B.1 + t * (C.1 - B.1) = P.1 ∧ B.2 + t * (C.2 - B.2) = P.2

/-- Triangle ABC is right-angled -/
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

theorem abc_right_triangle (B C : ℝ × ℝ) :
  on_parabola B C → line_through_P B C → is_right_triangle A B C := by sorry

end NUMINAMATH_CALUDE_abc_right_triangle_l3305_330531


namespace NUMINAMATH_CALUDE_hansel_salary_l3305_330578

theorem hansel_salary (hansel_initial : ℝ) (gretel_initial : ℝ) :
  hansel_initial = gretel_initial →
  hansel_initial * 1.10 + 1500 = gretel_initial * 1.15 →
  hansel_initial = 30000 := by
  sorry

end NUMINAMATH_CALUDE_hansel_salary_l3305_330578


namespace NUMINAMATH_CALUDE_teds_age_l3305_330574

theorem teds_age (s : ℝ) (t : ℝ) (a : ℝ) 
  (h1 : t = 2 * s + 17)
  (h2 : a = s / 2)
  (h3 : t + s + a = 72) : 
  ⌊t⌋ = 48 := by
sorry

end NUMINAMATH_CALUDE_teds_age_l3305_330574


namespace NUMINAMATH_CALUDE_jonny_sarah_marble_difference_l3305_330595

/-- The number of marbles Jonny has -/
def jonny_marbles : ℕ := 18

/-- The number of bags Sarah initially has -/
def sarah_bags : ℕ := 4

/-- The number of marbles in each of Sarah's bags -/
def sarah_marbles_per_bag : ℕ := 6

/-- The total number of marbles Sarah initially has -/
def sarah_total_marbles : ℕ := sarah_bags * sarah_marbles_per_bag

/-- The number of marbles Sarah has after giving half to Jared -/
def sarah_remaining_marbles : ℕ := sarah_total_marbles / 2

theorem jonny_sarah_marble_difference :
  jonny_marbles - sarah_remaining_marbles = 6 := by
  sorry

end NUMINAMATH_CALUDE_jonny_sarah_marble_difference_l3305_330595


namespace NUMINAMATH_CALUDE_circle_properties_l3305_330518

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 6 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 6 = 0

-- Define the common chord equation
def common_chord (x y : ℝ) : Prop := 3*x - 2*y = 0

-- Theorem stating the properties of the circles
theorem circle_properties :
  -- The circles intersect
  (∃ x y : ℝ, C₁ x y ∧ C₂ x y) ∧
  -- The common chord equation is correct
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord x y) ∧
  -- The length of the common chord is (2√1182) / 13
  (let chord_length := (2 * Real.sqrt 1182) / 13
   ∃ x₁ y₁ x₂ y₂ : ℝ,
     C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧
     common_chord x₁ y₁ ∧ common_chord x₂ y₂ ∧
     Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length) :=
by sorry


end NUMINAMATH_CALUDE_circle_properties_l3305_330518


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3305_330536

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (1 / (x^2 + y*z)) + (1 / (y^2 + z*x)) + (1 / (z^2 + x*y)) ≤ 
  (1 / 2) * ((1 / (x*y)) + (1 / (y*z)) + (1 / (z*x))) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3305_330536


namespace NUMINAMATH_CALUDE_geometric_sequence_bound_l3305_330534

/-- Given two geometric sequences with specified properties, prove that the first term of the first sequence must be less than 4/3 -/
theorem geometric_sequence_bound (a b : ℝ) (r_a r_b : ℝ) : 
  (∑' i, a * r_a ^ i = 1) →
  (∑' i, b * r_b ^ i = 1) →
  (∑' i, (a * r_a ^ i) ^ 2) * (∑' i, (b * r_b ^ i) ^ 2) = ∑' i, (a * r_a ^ i) * (b * r_b ^ i) →
  a < 4/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_bound_l3305_330534


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_difference_l3305_330552

theorem sin_cos_fourth_power_difference (α : ℝ) :
  Real.sin (π / 2 - 2 * α) = 3 / 5 →
  Real.sin α ^ 4 - Real.cos α ^ 4 = -(3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_difference_l3305_330552


namespace NUMINAMATH_CALUDE_fence_posts_for_grazing_area_l3305_330598

/-- The minimum number of fence posts required to enclose a rectangular area -/
def min_fence_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * length + width
  let num_intervals := perimeter / post_spacing
  num_intervals + 1

theorem fence_posts_for_grazing_area :
  min_fence_posts 60 36 12 = 12 :=
sorry

end NUMINAMATH_CALUDE_fence_posts_for_grazing_area_l3305_330598


namespace NUMINAMATH_CALUDE_fraction_positivity_implies_x_range_l3305_330592

theorem fraction_positivity_implies_x_range (x : ℝ) : (-6 : ℝ) / (7 - x) > 0 → x > 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_positivity_implies_x_range_l3305_330592


namespace NUMINAMATH_CALUDE_distance_between_foci_l3305_330515

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25

/-- The first focus of the ellipse -/
def F1 : ℝ × ℝ := (4, 3)

/-- The second focus of the ellipse -/
def F2 : ℝ × ℝ := (-6, 9)

/-- The theorem stating the distance between the foci -/
theorem distance_between_foci :
  Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l3305_330515


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l3305_330501

theorem charity_ticket_revenue :
  ∀ (full_price : ℕ) (full_count half_count : ℕ),
  full_count + half_count = 200 →
  full_count = 3 * half_count →
  full_count * full_price + half_count * (full_price / 2) = 3501 →
  full_count * full_price = 3000 :=
by
  sorry

end NUMINAMATH_CALUDE_charity_ticket_revenue_l3305_330501


namespace NUMINAMATH_CALUDE_problem_statement_l3305_330563

theorem problem_statement (a b : ℝ) (ha : a = 3) (hb : b = 2) :
  2 * (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3305_330563


namespace NUMINAMATH_CALUDE_sin_1_lt_log_3_sqrt_7_l3305_330514

theorem sin_1_lt_log_3_sqrt_7 :
  ∀ (sin : ℝ → ℝ) (log : ℝ → ℝ → ℝ),
  (0 < 1 ∧ 1 < π/3 ∧ π/3 < π/2) →
  sin (π/3) = Real.sqrt 3 / 2 →
  3^7 < 7^4 →
  sin 1 < log 3 (Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_sin_1_lt_log_3_sqrt_7_l3305_330514


namespace NUMINAMATH_CALUDE_regression_line_equation_l3305_330599

/-- Given a regression line with slope 1.23 passing through (4,5), prove its equation is y = 1.23x + 0.08 -/
theorem regression_line_equation (slope : ℝ) (center_x center_y : ℝ) :
  slope = 1.23 →
  center_x = 4 →
  center_y = 5 →
  ∃ (b : ℝ), b = 0.08 ∧ ∀ (x y : ℝ), y = slope * x + b ↔ y - center_y = slope * (x - center_x) :=
sorry

end NUMINAMATH_CALUDE_regression_line_equation_l3305_330599


namespace NUMINAMATH_CALUDE_inverse_of_M_l3305_330524

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 2, 3]
def M : Matrix (Fin 2) (Fin 2) ℝ := B * A

theorem inverse_of_M : 
  M⁻¹ = !![3/10, -1/10; 1/5, -2/5] :=
sorry

end NUMINAMATH_CALUDE_inverse_of_M_l3305_330524


namespace NUMINAMATH_CALUDE_x_values_l3305_330504

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^2 + 1/y = 13) (h2 : y^2 + 1/x = 8) :
  x = Real.sqrt 13 ∨ x = -Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_x_values_l3305_330504


namespace NUMINAMATH_CALUDE_sin_300_degrees_l3305_330569

theorem sin_300_degrees : 
  Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l3305_330569


namespace NUMINAMATH_CALUDE_collage_glue_drops_l3305_330576

/-- Calculates the total number of glue drops needed for a collage -/
def total_glue_drops (num_friends : ℕ) (clippings_per_friend : ℕ) (glue_drops_per_clipping : ℕ) : ℕ :=
  num_friends * clippings_per_friend * glue_drops_per_clipping

/-- Proves that for 7 friends, 3 clippings per friend, and 6 drops of glue per clipping, 
    the total number of glue drops needed is 126 -/
theorem collage_glue_drops : 
  total_glue_drops 7 3 6 = 126 := by
  sorry

end NUMINAMATH_CALUDE_collage_glue_drops_l3305_330576


namespace NUMINAMATH_CALUDE_ninety_eight_squared_l3305_330586

theorem ninety_eight_squared : 98 * 98 = 9604 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ninety_eight_squared_l3305_330586


namespace NUMINAMATH_CALUDE_six_hundred_million_scientific_notation_l3305_330593

-- Define 600 million
def six_hundred_million : ℝ := 600000000

-- Theorem statement
theorem six_hundred_million_scientific_notation :
  six_hundred_million = 6 * 10^8 := by
  sorry

end NUMINAMATH_CALUDE_six_hundred_million_scientific_notation_l3305_330593


namespace NUMINAMATH_CALUDE_decreasing_interval_of_quadratic_b_range_for_decreasing_l3305_330528

/-- A quadratic function f(x) = ax^2 + bx -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 2 * a * x + b

theorem decreasing_interval_of_quadratic (a b : ℝ) :
  (f_derivative a b 3 = 24) →  -- Tangent at x=3 is parallel to 24x-y+1=0
  (f_derivative a b 1 = 0) →   -- Extreme value at x=1
  ∀ x > 1, f_derivative a b x < 0 := by sorry

theorem b_range_for_decreasing (b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f_derivative 1 b x ≤ 0) →
  b ≤ -2 := by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_quadratic_b_range_for_decreasing_l3305_330528


namespace NUMINAMATH_CALUDE_find_divisor_l3305_330527

theorem find_divisor (N : ℝ) (D : ℝ) (h1 : N = 95) (h2 : N / D + 23 = 42) : D = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3305_330527


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l3305_330559

theorem chess_tournament_participants : ∃ n : ℕ, 
  n > 0 ∧ 
  (n * (n - 1)) / 2 = 190 ∧ 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l3305_330559


namespace NUMINAMATH_CALUDE_ln_range_l3305_330577

open Real

theorem ln_range (f : ℝ → ℝ) (x : ℝ) :
  (∀ y, f y = log y) →
  f (x - 1) < 1 →
  1 < x ∧ x < exp 1 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ln_range_l3305_330577


namespace NUMINAMATH_CALUDE_min_root_product_sum_l3305_330555

def f (x : ℝ) : ℝ := x^4 + 14*x^3 + 52*x^2 + 56*x + 16

theorem min_root_product_sum (z₁ z₂ z₃ z₄ : ℝ) 
  (hroots : (∀ x, f x = 0 ↔ x = z₁ ∨ x = z₂ ∨ x = z₃ ∨ x = z₄)) :
  (∀ (σ : Equiv.Perm (Fin 4)), 
    |z₁ * z₂ + z₃ * z₄| ≥ 8 ∧
    |z₁ * z₃ + z₂ * z₄| ≥ 8 ∧
    |z₁ * z₄ + z₂ * z₃| ≥ 8) ∧
  (∃ (σ : Equiv.Perm (Fin 4)), 
    |z₁ * z₂ + z₃ * z₄| = 8 ∨
    |z₁ * z₃ + z₂ * z₄| = 8 ∨
    |z₁ * z₄ + z₂ * z₃| = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_root_product_sum_l3305_330555


namespace NUMINAMATH_CALUDE_L_intersects_C_twice_L_min_chord_correct_l3305_330543

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L
def L (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Statement 1: L always intersects C at two points for any real m
theorem L_intersects_C_twice : ∀ m : ℝ, ∃! (p q : ℝ × ℝ), 
  p ≠ q ∧ C p.1 p.2 ∧ C q.1 q.2 ∧ L m p.1 p.2 ∧ L m q.1 q.2 :=
sorry

-- Statement 2: Equation of L with minimum chord length
def L_min_chord (x y : ℝ) : Prop := 2*x - y - 5 = 0

theorem L_min_chord_correct : 
  (∀ m : ℝ, ∃ (p q : ℝ × ℝ), p ≠ q ∧ C p.1 p.2 ∧ C q.1 q.2 ∧ L m p.1 p.2 ∧ L m q.1 q.2 ∧ 
    ∀ (r s : ℝ × ℝ), r ≠ s ∧ C r.1 r.2 ∧ C s.1 s.2 ∧ L_min_chord r.1 r.2 ∧ L_min_chord s.1 s.2 →
      (p.1 - q.1)^2 + (p.2 - q.2)^2 ≥ (r.1 - s.1)^2 + (r.2 - s.2)^2) ∧
  (∃ (p q : ℝ × ℝ), p ≠ q ∧ C p.1 p.2 ∧ C q.1 q.2 ∧ L_min_chord p.1 p.2 ∧ L_min_chord q.1 q.2) :=
sorry

end NUMINAMATH_CALUDE_L_intersects_C_twice_L_min_chord_correct_l3305_330543


namespace NUMINAMATH_CALUDE_circular_fountain_area_l3305_330590

theorem circular_fountain_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : 
  let AD := AB / 2
  let R := Real.sqrt (AD ^ 2 + DC ^ 2)
  π * R ^ 2 = 244 * π := by sorry

end NUMINAMATH_CALUDE_circular_fountain_area_l3305_330590


namespace NUMINAMATH_CALUDE_fourth_number_unit_digit_l3305_330584

def unit_digit (n : ℕ) : ℕ := n % 10

def product_unit_digit (a b c d : ℕ) : ℕ :=
  unit_digit (unit_digit a * unit_digit b * unit_digit c * unit_digit d)

theorem fourth_number_unit_digit :
  ∃ (x : ℕ), product_unit_digit 624 708 463 x = 8 ∧ unit_digit x = 3 :=
by sorry

end NUMINAMATH_CALUDE_fourth_number_unit_digit_l3305_330584


namespace NUMINAMATH_CALUDE_tv_price_difference_l3305_330594

theorem tv_price_difference (budget : ℝ) (initial_discount : ℝ) (percentage_discount : ℝ) : 
  budget = 1000 →
  initial_discount = 100 →
  percentage_discount = 0.2 →
  budget - (budget - initial_discount) * (1 - percentage_discount) = 280 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_difference_l3305_330594


namespace NUMINAMATH_CALUDE_initial_pencils_count_l3305_330565

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := sorry

/-- The number of pencils Tim added to the drawer -/
def pencils_added : ℕ := 3

/-- The total number of pencils after Tim's addition -/
def total_pencils : ℕ := 5

theorem initial_pencils_count : initial_pencils = 2 :=
  by sorry

end NUMINAMATH_CALUDE_initial_pencils_count_l3305_330565


namespace NUMINAMATH_CALUDE_factorization_equality_l3305_330538

theorem factorization_equality (x y : ℝ) : (x + 2) * (x - 2) - 4 * y * (x - y) = (x - 2*y + 2) * (x - 2*y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3305_330538


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l3305_330546

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 25) : 
  r - p = 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l3305_330546


namespace NUMINAMATH_CALUDE_maria_coin_stacks_maria_coin_stacks_proof_l3305_330521

/-- Given that Maria has a total of 15 coins and each stack contains 3 coins,
    prove that the number of stacks she has is 5. -/
theorem maria_coin_stacks : ℕ → ℕ → ℕ → Prop :=
  fun (total_coins : ℕ) (coins_per_stack : ℕ) (num_stacks : ℕ) =>
    total_coins = 15 ∧ coins_per_stack = 3 →
    num_stacks * coins_per_stack = total_coins →
    num_stacks = 5

#check maria_coin_stacks

/-- Proof of the theorem -/
theorem maria_coin_stacks_proof : maria_coin_stacks 15 3 5 := by
  sorry

end NUMINAMATH_CALUDE_maria_coin_stacks_maria_coin_stacks_proof_l3305_330521


namespace NUMINAMATH_CALUDE_secret_reaches_1093_on_sunday_l3305_330544

def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

theorem secret_reaches_1093_on_sunday : 
  ∃ n : ℕ, secret_spread n = 1093 ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_secret_reaches_1093_on_sunday_l3305_330544


namespace NUMINAMATH_CALUDE_intersection_parallel_line_l3305_330539

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line -/
theorem intersection_parallel_line (x y : ℝ) : 
  (2 * x - 3 * y + 2 = 0) →  -- l₁
  (3 * x - 4 * y - 2 = 0) →  -- l₂
  ∃ (k : ℝ), (4 * x - 2 * y + k = 0) ∧  -- parallel line
  (2 * x - y - 18 = 0) :=  -- result
by sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_l3305_330539


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l3305_330554

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total_people = men + women →
  men = 10 →
  women = 5 →
  selected = 4 →
  (1 - (Nat.choose men selected : ℚ) / (Nat.choose total_people selected : ℚ)) = 77 / 91 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l3305_330554


namespace NUMINAMATH_CALUDE_angle_greater_iff_sin_greater_l3305_330597

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- State the theorem
theorem angle_greater_iff_sin_greater (t : Triangle) : t.A > t.B ↔ Real.sin t.A > Real.sin t.B := by
  sorry

end NUMINAMATH_CALUDE_angle_greater_iff_sin_greater_l3305_330597


namespace NUMINAMATH_CALUDE_equation_solutions_l3305_330537

def solution_set : Set (ℤ × ℤ) :=
  {(2, 1), (1, 0), (2, 2), (0, 0), (1, 2), (0, 1)}

theorem equation_solutions :
  ∀ (x y : ℤ), (x + y = x^2 - x*y + y^2) ↔ (x, y) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3305_330537


namespace NUMINAMATH_CALUDE_aiguo_seashells_l3305_330564

/-- The number of seashells collected by Aiguo, Vail, and Stefan satisfies the given conditions -/
def seashell_collection (aiguo vail stefan : ℕ) : Prop :=
  stefan = vail + 16 ∧ 
  vail + 5 = aiguo ∧ 
  aiguo + vail + stefan = 66

/-- Aiguo had 20 seashells -/
theorem aiguo_seashells :
  ∃ (vail stefan : ℕ), seashell_collection 20 vail stefan := by
  sorry

end NUMINAMATH_CALUDE_aiguo_seashells_l3305_330564


namespace NUMINAMATH_CALUDE_all_propositions_false_l3305_330545

-- Define the correlation coefficient
def correlation_coefficient : ℝ → ℝ := sorry

-- Define the degree of linear correlation
def linear_correlation_degree : ℝ → ℝ := sorry

-- Define the cubic function
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- Define what it means for f to have an extreme value at x = -1
def has_extreme_value_at_neg_one (a b : ℝ) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x + 1| ∧ |x + 1| < ε →
    (f a b (-1) - f a b x) * (f a b (-1) - f a b (-1 - (x + 1))) > 0

theorem all_propositions_false :
  -- Proposition 1
  (∀ r₁ r₂ : ℝ, |r₁| < |r₂| → linear_correlation_degree r₁ < linear_correlation_degree r₂) ∧
  -- Proposition 2
  (¬(∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 > 0) ∧
  -- Proposition 3
  (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧
  -- Proposition 4
  (∀ a b : ℝ, has_extreme_value_at_neg_one a b → (a = 1 ∧ b = 9))
  → False := by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l3305_330545


namespace NUMINAMATH_CALUDE_total_muffins_after_baking_l3305_330510

def initial_muffins : ℕ := 35
def additional_muffins : ℕ := 48

theorem total_muffins_after_baking :
  initial_muffins + additional_muffins = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_muffins_after_baking_l3305_330510


namespace NUMINAMATH_CALUDE_sum_modulo_thirteen_l3305_330572

theorem sum_modulo_thirteen : (9245 + 9246 + 9247 + 9248 + 9249 + 9250) % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_modulo_thirteen_l3305_330572


namespace NUMINAMATH_CALUDE_intersection_point_is_correct_l3305_330513

-- Define the slope of the first line
def m₁ : ℚ := 2

-- Define the first line: y = 2x + 3
def line₁ (x y : ℚ) : Prop := y = m₁ * x + 3

-- Define the slope of the perpendicular line
def m₂ : ℚ := -1 / m₁

-- Define the point that the perpendicular line passes through
def point : ℚ × ℚ := (3, 8)

-- Define the perpendicular line passing through (3, 8)
def line₂ (x y : ℚ) : Prop :=
  y - point.2 = m₂ * (x - point.1)

-- Define the intersection point
def intersection_point : ℚ × ℚ := (13/5, 41/5)

-- Theorem statement
theorem intersection_point_is_correct :
  line₁ intersection_point.1 intersection_point.2 ∧
  line₂ intersection_point.1 intersection_point.2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_correct_l3305_330513


namespace NUMINAMATH_CALUDE_cube_root_of_negative_64_l3305_330570

theorem cube_root_of_negative_64 : ∃ x : ℝ, x^3 = -64 ∧ x = -4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_64_l3305_330570


namespace NUMINAMATH_CALUDE_flour_needed_proof_l3305_330535

/-- The amount of flour Katie needs in pounds -/
def katie_flour : ℝ := 3

/-- The additional amount of flour Sheila needs compared to Katie in pounds -/
def sheila_additional : ℝ := 2

/-- The multiplier for John's flour needs compared to Sheila's -/
def john_multiplier : ℝ := 1.5

/-- The amount of flour Sheila needs in pounds -/
def sheila_flour : ℝ := katie_flour + sheila_additional

/-- The amount of flour John needs in pounds -/
def john_flour : ℝ := john_multiplier * sheila_flour

/-- The total amount of flour needed by Katie, Sheila, and John -/
def total_flour : ℝ := katie_flour + sheila_flour + john_flour

theorem flour_needed_proof : total_flour = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_proof_l3305_330535


namespace NUMINAMATH_CALUDE_pasture_problem_l3305_330509

/-- The number of horses c put in the pasture -/
def c_horses : ℕ := 18

/-- The total cost of the pasture in Rs -/
def total_cost : ℕ := 870

/-- b's payment for the pasture in Rs -/
def b_payment : ℕ := 360

/-- a's horses -/
def a_horses : ℕ := 12

/-- b's horses -/
def b_horses : ℕ := 16

/-- a's months -/
def a_months : ℕ := 8

/-- b's months -/
def b_months : ℕ := 9

/-- c's months -/
def c_months : ℕ := 6

theorem pasture_problem :
  c_horses * c_months * total_cost = 
    b_payment * (a_horses * a_months + b_horses * b_months + c_horses * c_months) - 
    b_horses * b_months * total_cost := by
  sorry

end NUMINAMATH_CALUDE_pasture_problem_l3305_330509


namespace NUMINAMATH_CALUDE_derivative_cos_ln_l3305_330556

open Real

theorem derivative_cos_ln (x : ℝ) (h : x > 0) :
  deriv (λ x => cos (log x)) x = -1/x * sin (log x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_cos_ln_l3305_330556


namespace NUMINAMATH_CALUDE_collinear_vectors_problem_l3305_330550

/-- Given vectors a, b, and c in ℝ², prove that if a + b is collinear with c, then the y-coordinate of c is 1. -/
theorem collinear_vectors_problem (a b c : ℝ × ℝ) 
    (ha : a = (1, 2))
    (hb : b = (1, -3))
    (hc : c.1 = -2) 
    (h_collinear : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 + b.1, a.2 + b.2) = (k * c.1, k * c.2)) :
  c.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_problem_l3305_330550


namespace NUMINAMATH_CALUDE_base_three_20121_equals_178_l3305_330516

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_three_20121_equals_178 :
  base_three_to_decimal [2, 0, 1, 2, 1] = 178 := by
  sorry

end NUMINAMATH_CALUDE_base_three_20121_equals_178_l3305_330516


namespace NUMINAMATH_CALUDE_initial_white_cookies_l3305_330587

def cookie_problem (w : ℕ) : Prop :=
  let b := w + 50
  let remaining_black := b / 2
  let remaining_white := w / 4
  remaining_black + remaining_white = 85

theorem initial_white_cookies : ∃ w : ℕ, cookie_problem w ∧ w = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_white_cookies_l3305_330587
