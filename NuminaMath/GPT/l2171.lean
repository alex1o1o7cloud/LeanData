import Mathlib

namespace NUMINAMATH_GPT_correct_number_of_six_letter_words_l2171_217165

def number_of_six_letter_words (alphabet_size : ℕ) : ℕ :=
  alphabet_size ^ 4

theorem correct_number_of_six_letter_words :
  number_of_six_letter_words 26 = 456976 :=
by
  -- We write 'sorry' to omit the detailed proof.
  sorry

end NUMINAMATH_GPT_correct_number_of_six_letter_words_l2171_217165


namespace NUMINAMATH_GPT_gcd_323_391_l2171_217164

theorem gcd_323_391 : Nat.gcd 323 391 = 17 := 
by sorry

end NUMINAMATH_GPT_gcd_323_391_l2171_217164


namespace NUMINAMATH_GPT_positive_difference_of_complementary_ratio_5_1_l2171_217104

-- Define angles satisfying the ratio condition and being complementary
def angle_pair (a b : ℝ) : Prop := (a + b = 90) ∧ (a = 5 * b ∨ b = 5 * a)

theorem positive_difference_of_complementary_ratio_5_1 :
  ∃ a b : ℝ, angle_pair a b ∧ abs (a - b) = 60 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_complementary_ratio_5_1_l2171_217104


namespace NUMINAMATH_GPT_negation_of_universal_statement_l2171_217159

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, x^2 ≤ 1) ↔ ∃ x : ℝ, x^2 > 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l2171_217159


namespace NUMINAMATH_GPT_evaluate_expression_l2171_217196

theorem evaluate_expression (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2171_217196


namespace NUMINAMATH_GPT_marcella_shoes_l2171_217143

theorem marcella_shoes :
  ∀ (original_pairs lost_shoes : ℕ), original_pairs = 27 → lost_shoes = 9 → 
  ∃ (remaining_pairs : ℕ), remaining_pairs = 18 ∧ remaining_pairs ≤ original_pairs - lost_shoes / 2 :=
by
  intros original_pairs lost_shoes h1 h2
  use 18
  constructor
  . exact rfl
  . sorry

end NUMINAMATH_GPT_marcella_shoes_l2171_217143


namespace NUMINAMATH_GPT_solve_for_x_l2171_217106

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 * x + 7 * x = 140) : x = 28 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2171_217106


namespace NUMINAMATH_GPT_arccos_equivalence_l2171_217108

open Real

theorem arccos_equivalence (α : ℝ) (h₀ : α ∈ Set.Icc 0 (2 * π)) (h₁ : cos α = 1 / 3) :
  α = arccos (1 / 3) ∨ α = 2 * π - arccos (1 / 3) := 
by 
  sorry

end NUMINAMATH_GPT_arccos_equivalence_l2171_217108


namespace NUMINAMATH_GPT_maximum_value_P_l2171_217110

open Classical

noncomputable def P (a b c d : ℝ) : ℝ := a * b + b * c + c * d + d * a

theorem maximum_value_P : ∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 40 → P a b c d ≤ 800 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_P_l2171_217110


namespace NUMINAMATH_GPT_percentage_support_of_surveyed_population_l2171_217120

-- Definitions based on the conditions
def men_percentage_support : ℝ := 0.70
def women_percentage_support : ℝ := 0.75
def men_surveyed : ℕ := 200
def women_surveyed : ℕ := 800

-- Proof statement
theorem percentage_support_of_surveyed_population : 
  ((men_percentage_support * men_surveyed + women_percentage_support * women_surveyed) / 
   (men_surveyed + women_surveyed) * 100) = 74 := 
by
  sorry

end NUMINAMATH_GPT_percentage_support_of_surveyed_population_l2171_217120


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l2171_217103

theorem sqrt_meaningful_range (x : ℝ) (h : x - 2 ≥ 0) : x ≥ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_sqrt_meaningful_range_l2171_217103


namespace NUMINAMATH_GPT_regular_polygon_sides_l2171_217113

theorem regular_polygon_sides (P s : ℕ) (hP : P = 150) (hs : s = 15) :
  P / s = 10 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2171_217113


namespace NUMINAMATH_GPT_solve_problem_l2171_217156

-- Define the constants c and d
variables (c d : ℝ)

-- Define the conditions of the problem
def condition1 : Prop := 
  (∀ x : ℝ, (x + c) * (x + d) * (x + 15) = 0 ↔ x = -c ∨ x = -d ∨ x = -15) ∧
  -4 ≠ -c ∧ -4 ≠ -d ∧ -4 ≠ -15

def condition2 : Prop := 
  (∀ x : ℝ, (x + 3 * c) * (x + 4) * (x + 9) = 0 ↔ x = -4) ∧
  d ≠ -4 ∧ d ≠ -15

-- We need to prove this final result under the given conditions
theorem solve_problem (h1 : condition1 c d) (h2 : condition2 c d) : 100 * c + d = -291 := 
  sorry

end NUMINAMATH_GPT_solve_problem_l2171_217156


namespace NUMINAMATH_GPT_multiple_of_large_block_length_l2171_217142

-- Define the dimensions and volumes
variables (w d l : ℝ) -- Normal block dimensions
variables (V_normal V_large : ℝ) -- Volumes
variables (m : ℝ) -- Multiple for the length of the large block

-- Volume conditions for normal and large blocks
def normal_volume_condition (w d l : ℝ) (V_normal : ℝ) : Prop :=
  V_normal = w * d * l

def large_volume_condition (w d l m V_large : ℝ) : Prop :=
  V_large = (2 * w) * (2 * d) * (m * l)

-- Given problem conditions
axiom V_normal_eq_3 : normal_volume_condition w d l 3
axiom V_large_eq_36 : large_volume_condition w d l m 36

-- Statement we want to prove
theorem multiple_of_large_block_length : m = 3 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_multiple_of_large_block_length_l2171_217142


namespace NUMINAMATH_GPT_combined_distance_is_12_l2171_217161

-- Define the distances the two ladies walked
def distance_second_lady : ℝ := 4
def distance_first_lady := 2 * distance_second_lady

-- Define the combined total distance
def combined_distance := distance_first_lady + distance_second_lady

-- Statement of the problem as a proof goal in Lean
theorem combined_distance_is_12 : combined_distance = 12 :=
by
  -- Definitions required for the proof
  let second := distance_second_lady
  let first := distance_first_lady
  let total := combined_distance
  
  -- Insert the necessary calculations and proof steps here
  -- Conclude with the desired result
  sorry

end NUMINAMATH_GPT_combined_distance_is_12_l2171_217161


namespace NUMINAMATH_GPT_smallest_period_find_a_l2171_217193

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + a

theorem smallest_period (a : ℝ) : 
  ∃ T > 0, ∀ x, f x a = f (x + T) a ∧ (∀ T' > 0, (∀ x, f x a = f (x + T') a) → T ≤ T') :=
by
  sorry

theorem find_a :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) ∧ (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = 4) ∧ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_period_find_a_l2171_217193


namespace NUMINAMATH_GPT_carol_remaining_distance_l2171_217155

def fuel_efficiency : ℕ := 25 -- miles per gallon
def gas_tank_capacity : ℕ := 18 -- gallons
def distance_to_home : ℕ := 350 -- miles

def total_distance_on_full_tank : ℕ := fuel_efficiency * gas_tank_capacity
def distance_after_home : ℕ := total_distance_on_full_tank - distance_to_home

theorem carol_remaining_distance :
  distance_after_home = 100 :=
sorry

end NUMINAMATH_GPT_carol_remaining_distance_l2171_217155


namespace NUMINAMATH_GPT_max_value_of_x_plus_y_plus_z_l2171_217119

theorem max_value_of_x_plus_y_plus_z : ∀ (x y z : ℤ), (∃ k : ℤ, x = 5 * k ∧ 6 = y * k ∧ z = 2 * k) → x + y + z ≤ 43 :=
by
  intros x y z h
  rcases h with ⟨k, hx, hy, hz⟩
  sorry

end NUMINAMATH_GPT_max_value_of_x_plus_y_plus_z_l2171_217119


namespace NUMINAMATH_GPT_expected_score_is_6_l2171_217150

-- Define the probabilities of making a shot
def p : ℝ := 0.5

-- Define the scores for each scenario
def score_first_shot : ℝ := 8
def score_second_shot : ℝ := 6
def score_third_shot : ℝ := 4
def score_no_shot : ℝ := 0

-- Compute the expected value
def expected_score : ℝ :=
  p * score_first_shot +
  (1 - p) * p * score_second_shot +
  (1 - p) * (1 - p) * p * score_third_shot +
  (1 - p) * (1 - p) * (1 - p) * score_no_shot

theorem expected_score_is_6 : expected_score = 6 := by
  sorry

end NUMINAMATH_GPT_expected_score_is_6_l2171_217150


namespace NUMINAMATH_GPT_problem_solution_l2171_217146

-- Define the sets and the conditions given in the problem
def setA : Set ℝ := 
  {y | ∃ (x : ℝ), (x ∈ Set.Icc (3 / 4) 2) ∧ (y = x^2 - (3 / 2) * x + 1)}

def setB (m : ℝ) : Set ℝ := 
  {x | x + m^2 ≥ 1}

-- The proof statement contains two parts
theorem problem_solution (m : ℝ) :
  -- Part (I) - Prove the set A
  setA = Set.Icc (7 / 16) 2
  ∧
  -- Part (II) - Prove the range for m
  (∀ x, x ∈ setA → x ∈ setB m) → (m ≥ 3 / 4 ∨ m ≤ -3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2171_217146


namespace NUMINAMATH_GPT_total_votes_l2171_217187

theorem total_votes (V : ℕ) 
  (h1 : V * 45 / 100 + V * 25 / 100 + V * 15 / 100 + 180 + 50 = V) : 
  V = 1533 := 
by
  sorry

end NUMINAMATH_GPT_total_votes_l2171_217187


namespace NUMINAMATH_GPT_initial_percentage_increase_l2171_217117

variable (S : ℝ) (P : ℝ)

theorem initial_percentage_increase :
  (S + (P / 100) * S) - 0.10 * (S + (P / 100) * S) = S + 0.15 * S →
  P = 16.67 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_increase_l2171_217117


namespace NUMINAMATH_GPT_average_number_of_stickers_per_album_is_correct_l2171_217166

def average_stickers_per_album (albums : List ℕ) (n : ℕ) : ℚ := (albums.sum : ℚ) / n

theorem average_number_of_stickers_per_album_is_correct :
  average_stickers_per_album [5, 7, 9, 14, 19, 12, 26, 18, 11, 15] 10 = 13.6 := 
by
  sorry

end NUMINAMATH_GPT_average_number_of_stickers_per_album_is_correct_l2171_217166


namespace NUMINAMATH_GPT_total_hours_watched_l2171_217184

theorem total_hours_watched (Monday Tuesday Wednesday Thursday Friday : ℕ) (hMonday : Monday = 12) (hTuesday : Tuesday = 4) (hWednesday : Wednesday = 6) (hThursday : Thursday = (Monday + Tuesday + Wednesday) / 2) (hFriday : Friday = 19) :
  Monday + Tuesday + Wednesday + Thursday + Friday = 52 := by
  sorry

end NUMINAMATH_GPT_total_hours_watched_l2171_217184


namespace NUMINAMATH_GPT_vacation_days_l2171_217129

theorem vacation_days (total_miles miles_per_day : ℕ) 
  (h1 : total_miles = 1250) (h2 : miles_per_day = 250) :
  total_miles / miles_per_day = 5 := by
  sorry

end NUMINAMATH_GPT_vacation_days_l2171_217129


namespace NUMINAMATH_GPT_solve_rational_equation_l2171_217192

theorem solve_rational_equation : 
  ∀ x : ℝ, x ≠ 1 -> (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) → 
  (x = 6 ∨ x = -2) :=
by
  intro x
  intro h
  intro h_eq
  sorry

end NUMINAMATH_GPT_solve_rational_equation_l2171_217192


namespace NUMINAMATH_GPT_find_k_l2171_217177

theorem find_k : 
  ∃ x y k : ℝ, y = 7 * x - 2 ∧ y = -3 * x + 14 ∧ y = 4 * x + k ∧ k = 2.8 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2171_217177


namespace NUMINAMATH_GPT_chef_potatoes_l2171_217126

theorem chef_potatoes (total_potatoes cooked_potatoes time_per_potato rest_time: ℕ)
  (h1 : total_potatoes = 15)
  (h2 : time_per_potato = 9)
  (h3 : rest_time = 63)
  (h4 : time_per_potato * (total_potatoes - cooked_potatoes) = rest_time) :
  cooked_potatoes = 8 :=
by sorry

end NUMINAMATH_GPT_chef_potatoes_l2171_217126


namespace NUMINAMATH_GPT_min_value_expression_l2171_217132

theorem min_value_expression : ∀ (x y : ℝ), ∃ z : ℝ, z ≥ 3*x^2 + 2*x*y + 3*y^2 + 5 ∧ z = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l2171_217132


namespace NUMINAMATH_GPT_maximize_a_n_l2171_217124

-- Given sequence definition
noncomputable def a_n (n : ℕ) := (n + 2) * (7 / 8) ^ n

-- Prove that n = 5 or n = 6 maximizes the sequence
theorem maximize_a_n : ∃ n, (n = 5 ∨ n = 6) ∧ (∀ k, a_n k ≤ a_n n) :=
by
  sorry

end NUMINAMATH_GPT_maximize_a_n_l2171_217124


namespace NUMINAMATH_GPT_ellipse_foci_y_axis_range_l2171_217147

theorem ellipse_foci_y_axis_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1) ↔ (m < -1 ∨ (1 < m ∧ m < 3 / 2)) :=
sorry

end NUMINAMATH_GPT_ellipse_foci_y_axis_range_l2171_217147


namespace NUMINAMATH_GPT_pages_copyable_l2171_217136

-- Define the conditions
def cents_per_dollar : ℕ := 100
def dollars_available : ℕ := 25
def cost_per_page : ℕ := 3

-- Define the total cents available
def total_cents : ℕ := dollars_available * cents_per_dollar

-- Define the expected number of full pages
def expected_pages : ℕ := 833

theorem pages_copyable :
  (total_cents : ℕ) / cost_per_page = expected_pages := sorry

end NUMINAMATH_GPT_pages_copyable_l2171_217136


namespace NUMINAMATH_GPT_quadratic_positivity_range_l2171_217109

variable (a : ℝ)

def quadratic_function (x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 3

theorem quadratic_positivity_range :
  (∀ x, 0 < x ∧ x < 3 → quadratic_function a x > 0)
  ↔ (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3) := sorry

end NUMINAMATH_GPT_quadratic_positivity_range_l2171_217109


namespace NUMINAMATH_GPT_angles_sum_n_l2171_217131

/-- Given that the sum of the measures in degrees of angles A, B, C, D, E, and F is 90 * n,
    we need to prove that n = 4. -/
theorem angles_sum_n (A B C D E F : ℝ) (n : ℕ) 
  (h : A + B + C + D + E + F = 90 * n) :
  n = 4 :=
sorry

end NUMINAMATH_GPT_angles_sum_n_l2171_217131


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l2171_217169

open Real

noncomputable def f (x : ℝ) := (sqrt 3) * sin x * cos x - (1 / 2) * cos (2 * x)

theorem problem_part1 : 
  (∀ x : ℝ, -1 ≤ f x) ∧ 
  (∃ T : ℝ, (T > 0) ∧ ∀ x : ℝ, f (x + T) = f x ∧ T = π) := 
sorry

theorem problem_part2 (C A B c : ℝ) :
  (f C = 1) → 
  (B = π / 6) → 
  (c = 2 * sqrt 3) → 
  ∃ b : ℝ, ∃ area : ℝ, b = 2 ∧ area = (1 / 2) * b * c * sin A ∧ area = 2 * sqrt 3 := 
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l2171_217169


namespace NUMINAMATH_GPT_total_expenditure_eq_fourteen_l2171_217153

variable (cost_barrette cost_comb : ℕ)
variable (kristine_barrettes kristine_combs crystal_barrettes crystal_combs : ℕ)

theorem total_expenditure_eq_fourteen 
  (h_cost_barrette : cost_barrette = 3)
  (h_cost_comb : cost_comb = 1)
  (h_kristine_barrettes : kristine_barrettes = 1)
  (h_kristine_combs : kristine_combs = 1)
  (h_crystal_barrettes : crystal_barrettes = 3)
  (h_crystal_combs : crystal_combs = 1) :
  (kristine_barrettes * cost_barrette + kristine_combs * cost_comb) +
  (crystal_barrettes * cost_barrette + crystal_combs * cost_comb) = 14 := 
by 
  sorry

end NUMINAMATH_GPT_total_expenditure_eq_fourteen_l2171_217153


namespace NUMINAMATH_GPT_local_minimum_of_function_l2171_217138

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

theorem local_minimum_of_function : 
  (∃ a, a = 1 ∧ ∀ ε > 0, f a ≤ f (a + ε) ∧ f a ≤ f (a - ε)) := sorry

end NUMINAMATH_GPT_local_minimum_of_function_l2171_217138


namespace NUMINAMATH_GPT_g_diff_l2171_217114

def g (n : ℤ) : ℤ := (1 / 4 : ℤ) * n * (n + 1) * (n + 2) * (n + 3)

theorem g_diff (r : ℤ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
  sorry

end NUMINAMATH_GPT_g_diff_l2171_217114


namespace NUMINAMATH_GPT_ned_washed_shirts_l2171_217188

theorem ned_washed_shirts (short_sleeve long_sleeve not_washed: ℕ) (h1: short_sleeve = 9) (h2: long_sleeve = 21) (h3: not_washed = 1) : 
    (short_sleeve + long_sleeve - not_washed = 29) :=
by
  sorry

end NUMINAMATH_GPT_ned_washed_shirts_l2171_217188


namespace NUMINAMATH_GPT_range_of_t_sum_of_squares_l2171_217152

-- Define the conditions and the problem statement in Lean

variables (a b c t x : ℝ)
variables (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variables (ineq1 : |x + 1| - |x - 2| ≥ |t - 1| + t)
variables (sum_pos : 2 * a + b + c = 2)

theorem range_of_t :
  (∃ x, |x + 1| - |x - 2| ≥ |t - 1| + t) → t ≤ 2 :=
sorry

theorem sum_of_squares :
  2 * a + b + c = 2 → 0 < a → 0 < b → 0 < c → a^2 + b^2 + c^2 ≥ 2 / 3 :=
sorry

end NUMINAMATH_GPT_range_of_t_sum_of_squares_l2171_217152


namespace NUMINAMATH_GPT_max_sum_of_squares_l2171_217163

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17) 
  (h2 : ab + c + d = 85) 
  (h3 : ad + bc = 196) 
  (h4 : cd = 120) : 
  ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 918 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_sum_of_squares_l2171_217163


namespace NUMINAMATH_GPT_candy_problem_l2171_217168

theorem candy_problem
  (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999)
  (h3 : n + 7 ≡ 0 [MOD 9])
  (h4 : n - 9 ≡ 0 [MOD 6]) :
  n = 101 :=
sorry

end NUMINAMATH_GPT_candy_problem_l2171_217168


namespace NUMINAMATH_GPT_negation_of_proposition_l2171_217158

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, 0 ≤ x ∧ (x^2 - 2*x - 3 = 0)) ↔ (∀ x : ℝ, 0 ≤ x → (x^2 - 2*x - 3 ≠ 0)) := 
by 
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l2171_217158


namespace NUMINAMATH_GPT_probability_non_adjacent_sum_l2171_217178

-- Definitions and conditions from the problem
def total_trees := 13
def maple_trees := 4
def oak_trees := 3
def birch_trees := 6

-- Total possible arrangements of 13 trees
def total_arrangements := Nat.choose total_trees birch_trees

-- Number of ways to arrange birch trees with no two adjacent
def favorable_arrangements := Nat.choose (maple_trees + oak_trees + 1) birch_trees

-- Probability calculation
def probability_non_adjacent := (favorable_arrangements : ℚ) / (total_arrangements : ℚ)

-- This value should be simplified to form m/n in lowest terms
def fraction_part_m := 7
def fraction_part_n := 429

-- Verify m + n
def sum_m_n := fraction_part_m + fraction_part_n

-- Check that sum_m_n is equal to 436
theorem probability_non_adjacent_sum :
  sum_m_n = 436 := by {
    -- Placeholder proof
    sorry
}

end NUMINAMATH_GPT_probability_non_adjacent_sum_l2171_217178


namespace NUMINAMATH_GPT_E_eq_F_l2171_217199

noncomputable def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }

noncomputable def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem E_eq_F : E = F := 
sorry

end NUMINAMATH_GPT_E_eq_F_l2171_217199


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2171_217130

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 2 * x > 0 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2171_217130


namespace NUMINAMATH_GPT_midpoint_of_polar_line_segment_l2171_217139

theorem midpoint_of_polar_line_segment
  (r θ : ℝ)
  (hr : r > 0)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (hA : ∃ A, A = (8, 5 * Real.pi / 12))
  (hB : ∃ B, B = (8, -3 * Real.pi / 12)) :
  (r, θ) = (4, Real.pi / 12) := 
sorry

end NUMINAMATH_GPT_midpoint_of_polar_line_segment_l2171_217139


namespace NUMINAMATH_GPT_max_volume_at_6_l2171_217151

noncomputable def volume (x : ℝ) : ℝ :=
  x * (36 - 2 * x)^2

theorem max_volume_at_6 :
  ∃ x : ℝ, (0 < x) ∧ (x < 18) ∧ 
  (∀ y : ℝ, (0 < y) ∧ (y < 18) → volume y ≤ volume 6) :=
by
  sorry

end NUMINAMATH_GPT_max_volume_at_6_l2171_217151


namespace NUMINAMATH_GPT_negation_of_p_l2171_217118

variable {x : ℝ}

def proposition_p : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

theorem negation_of_p :
  ¬ (∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_p_l2171_217118


namespace NUMINAMATH_GPT_cost_price_USD_l2171_217191

-- Assume the conditions in Lean as given:
variable {C_USD : ℝ}

def condition1 (C_USD : ℝ) : Prop := 0.9 * C_USD + 200 = 1.04 * C_USD

theorem cost_price_USD (h : condition1 C_USD) : C_USD = 200 / 0.14 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_USD_l2171_217191


namespace NUMINAMATH_GPT_smallest_integer_satisfying_mod_conditions_l2171_217148

theorem smallest_integer_satisfying_mod_conditions :
  ∃ n : ℕ, n > 0 ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 4) ∧ 
  (n % 7 = 6) ∧ 
  (n % 11 = 10) ∧ 
  n = 1154 := 
sorry

end NUMINAMATH_GPT_smallest_integer_satisfying_mod_conditions_l2171_217148


namespace NUMINAMATH_GPT_derivative_at_two_l2171_217141

noncomputable def f (a : ℝ) (g : ℝ) (x : ℝ) : ℝ := a * x^3 + g * x^2 + 3

theorem derivative_at_two (a f_prime_2 : ℝ) (h_deriv_at_1 : deriv (f a f_prime_2) 1 = -5) :
  deriv (f a f_prime_2) 2 = -5 := by
  sorry

end NUMINAMATH_GPT_derivative_at_two_l2171_217141


namespace NUMINAMATH_GPT_monotonic_increasing_k_l2171_217140

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (3 * k - 2) * x - 5

theorem monotonic_increasing_k (k : ℝ) : (∀ x y : ℝ, 1 ≤ x → x ≤ y → f k x ≤ f k y) ↔ k ∈ Set.Ici (2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_k_l2171_217140


namespace NUMINAMATH_GPT_toys_per_box_l2171_217176

theorem toys_per_box (number_of_boxes total_toys : ℕ) (h₁ : number_of_boxes = 4) (h₂ : total_toys = 32) :
  total_toys / number_of_boxes = 8 :=
by
  sorry

end NUMINAMATH_GPT_toys_per_box_l2171_217176


namespace NUMINAMATH_GPT_profit_amount_calc_l2171_217179

-- Define the conditions as hypotheses
variables (SP : ℝ) (profit_percent : ℝ) (cost_price profit_amount : ℝ)

-- Given conditions
axiom selling_price : SP = 900
axiom profit_percentage : profit_percent = 50
axiom profit_formula : profit_amount = 0.5 * cost_price
axiom selling_price_formula : SP = cost_price + profit_amount

-- The theorem to be proven
theorem profit_amount_calc : profit_amount = 300 :=
by
  sorry

end NUMINAMATH_GPT_profit_amount_calc_l2171_217179


namespace NUMINAMATH_GPT_sqrt_450_simplified_l2171_217170

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end NUMINAMATH_GPT_sqrt_450_simplified_l2171_217170


namespace NUMINAMATH_GPT_cards_per_page_l2171_217183

noncomputable def total_cards (new_cards old_cards : ℕ) : ℕ := new_cards + old_cards

theorem cards_per_page
  (new_cards old_cards : ℕ)
  (total_pages : ℕ)
  (h_new_cards : new_cards = 3)
  (h_old_cards : old_cards = 13)
  (h_total_pages : total_pages = 2) :
  total_cards new_cards old_cards / total_pages = 8 :=
by
  rw [h_new_cards, h_old_cards, h_total_pages]
  rfl

end NUMINAMATH_GPT_cards_per_page_l2171_217183


namespace NUMINAMATH_GPT_lowest_possible_price_l2171_217115

theorem lowest_possible_price
  (regular_discount_rate : ℚ)
  (sale_discount_rate : ℚ)
  (manufacturer_price : ℚ)
  (H1 : regular_discount_rate = 0.30)
  (H2 : sale_discount_rate = 0.20)
  (H3 : manufacturer_price = 35) :
  (manufacturer_price * (1 - regular_discount_rate) * (1 - sale_discount_rate)) = 19.60 := by
  sorry

end NUMINAMATH_GPT_lowest_possible_price_l2171_217115


namespace NUMINAMATH_GPT_match_sequences_count_l2171_217105

-- Definitions based on the given conditions
def team_size : ℕ := 7
def total_matches : ℕ := 2 * team_size - 1

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement: number of possible match sequences
theorem match_sequences_count : 
  2 * binomial_coefficient total_matches team_size = 3432 :=
by
  sorry

end NUMINAMATH_GPT_match_sequences_count_l2171_217105


namespace NUMINAMATH_GPT_ratio_of_work_done_by_women_to_men_l2171_217157

theorem ratio_of_work_done_by_women_to_men 
  (total_work_men : ℕ := 15 * 21 * 8)
  (total_work_women : ℕ := 21 * 36 * 5) :
  (total_work_women : ℚ) / (total_work_men : ℚ) = 2 / 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_work_done_by_women_to_men_l2171_217157


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2171_217186

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 6 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2171_217186


namespace NUMINAMATH_GPT_multiple_of_3_b_multiple_of_3_a_minus_b_multiple_of_3_a_minus_c_multiple_of_3_c_minus_b_l2171_217102

variable (a b c : ℕ)

-- Define the conditions as hypotheses
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k, n = 3 * k
def is_multiple_of_12 (n : ℕ) : Prop := ∃ k, n = 12 * k
def is_multiple_of_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

-- Hypotheses
axiom ha : is_multiple_of_3 a
axiom hb : is_multiple_of_12 b
axiom hc : is_multiple_of_9 c

-- Statements to be proved
theorem multiple_of_3_b : is_multiple_of_3 b := sorry
theorem multiple_of_3_a_minus_b : is_multiple_of_3 (a - b) := sorry
theorem multiple_of_3_a_minus_c : is_multiple_of_3 (a - c) := sorry
theorem multiple_of_3_c_minus_b : is_multiple_of_3 (c - b) := sorry

end NUMINAMATH_GPT_multiple_of_3_b_multiple_of_3_a_minus_b_multiple_of_3_a_minus_c_multiple_of_3_c_minus_b_l2171_217102


namespace NUMINAMATH_GPT_min_abc_value_l2171_217127

noncomputable def minValue (a b c : ℝ) : ℝ := (a + b) / (a * b * c)

theorem min_abc_value (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
  (minValue a b c) ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_min_abc_value_l2171_217127


namespace NUMINAMATH_GPT_k_less_than_zero_l2171_217194

variable (k : ℝ)

def function_decreases (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂

theorem k_less_than_zero (h : function_decreases (λ x => k * x - 5)) : k < 0 :=
sorry

end NUMINAMATH_GPT_k_less_than_zero_l2171_217194


namespace NUMINAMATH_GPT_sector_area_l2171_217101

def central_angle := 120 -- in degrees
def radius := 3 -- in units

theorem sector_area (n : ℕ) (R : ℕ) (h₁ : n = central_angle) (h₂ : R = radius) :
  (n * R^2 * Real.pi / 360) = 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sector_area_l2171_217101


namespace NUMINAMATH_GPT_unique_solution_nat_numbers_l2171_217133

theorem unique_solution_nat_numbers (a b c : ℕ) (h : 2^a + 9^b = 2 * 5^c + 5) : 
  (a, b, c) = (1, 0, 0) :=
sorry

end NUMINAMATH_GPT_unique_solution_nat_numbers_l2171_217133


namespace NUMINAMATH_GPT_jordan_length_eq_six_l2171_217198

def carol_length := 12
def carol_width := 15
def jordan_width := 30

theorem jordan_length_eq_six
  (h1 : carol_length * carol_width = jordan_width * jordan_length) : 
  jordan_length = 6 := by
  sorry

end NUMINAMATH_GPT_jordan_length_eq_six_l2171_217198


namespace NUMINAMATH_GPT_number_of_BMWs_sold_l2171_217154

theorem number_of_BMWs_sold (total_cars : ℕ) (Audi_percent Toyota_percent Acura_percent Ford_percent : ℝ)
  (h_total_cars : total_cars = 250) 
  (h_percentages : Audi_percent = 0.10 ∧ Toyota_percent = 0.20 ∧ Acura_percent = 0.15 ∧ Ford_percent = 0.25) :
  ∃ (BMWs_sold : ℕ), BMWs_sold = 75 := 
by
  sorry

end NUMINAMATH_GPT_number_of_BMWs_sold_l2171_217154


namespace NUMINAMATH_GPT_integer_solutions_to_equation_l2171_217171

theorem integer_solutions_to_equation :
  { p : ℤ × ℤ | (p.1 ^ 2 * p.2 + 1 = p.1 ^ 2 + 2 * p.1 * p.2 + 2 * p.1 + p.2) } =
  { (-1, -1), (0, 1), (1, -1), (2, -7), (3, 7) } :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_to_equation_l2171_217171


namespace NUMINAMATH_GPT_exponents_of_equation_l2171_217175

theorem exponents_of_equation :
  ∃ (x y : ℕ), 2 * (3 ^ 8) ^ 2 * (2 ^ 3) ^ 2 * 3 = 2 ^ x * 3 ^ y ∧ x = 7 ∧ y = 17 :=
by
  use 7
  use 17
  sorry

end NUMINAMATH_GPT_exponents_of_equation_l2171_217175


namespace NUMINAMATH_GPT_triangle_properties_l2171_217172

open Real

noncomputable def is_isosceles_triangle (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧ (b = c)

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def area (a b c : ℝ) (A : ℝ) : ℝ :=
  1/2 * b * c * sin A

theorem triangle_properties 
  (A B C a b c : ℝ) 
  (h1 : sin B * sin C = 1/4) 
  (h2 : tan B * tan C = 1/3) 
  (h3 : a = 4 * sqrt 3) 
  (h4 : A + B + C = π) 
  (isosceles : is_isosceles_triangle A B C a b c) :
  is_isosceles_triangle A B C a b c ∧ 
  perimeter a b c = 8 + 4 * sqrt 3 ∧ 
  area a b c A = 4 * sqrt 3 :=
sorry

end NUMINAMATH_GPT_triangle_properties_l2171_217172


namespace NUMINAMATH_GPT_equation_of_chord_l2171_217185

-- Define the ellipse equation and point P
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144
def P : ℝ × ℝ := (3, 2)
def is_midpoint (A B P : ℝ × ℝ) : Prop := A.1 + B.1 = 2 * P.1 ∧ A.2 + B.2 = 2 * P.2
def on_chord (A B : ℝ × ℝ) (x y : ℝ) : Prop := (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1)

-- Lean Statement
theorem equation_of_chord :
  ∀ A B : ℝ × ℝ,
    ellipse_eq A.1 A.2 →
    ellipse_eq B.1 B.2 →
    is_midpoint A B P →
    ∀ x y : ℝ,
      on_chord A B x y →
      2 * x + 3 * y = 12 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_chord_l2171_217185


namespace NUMINAMATH_GPT_blackRhinoCount_correct_l2171_217190

noncomputable def numberOfBlackRhinos : ℕ :=
  let whiteRhinoCount := 7
  let whiteRhinoWeight := 5100
  let blackRhinoWeightInTons := 1
  let totalWeight := 51700
  let oneTonInPounds := 2000
  let totalWhiteRhinoWeight := whiteRhinoCount * whiteRhinoWeight
  let totalBlackRhinoWeight := totalWeight - totalWhiteRhinoWeight
  totalBlackRhinoWeight / (blackRhinoWeightInTons * oneTonInPounds)

theorem blackRhinoCount_correct : numberOfBlackRhinos = 8 := by
  sorry

end NUMINAMATH_GPT_blackRhinoCount_correct_l2171_217190


namespace NUMINAMATH_GPT_store_hours_open_per_day_l2171_217134

theorem store_hours_open_per_day
  (rent_per_week : ℝ)
  (utility_percentage : ℝ)
  (employees_per_shift : ℕ)
  (hourly_wage : ℝ)
  (days_per_week_open : ℕ)
  (weekly_expenses : ℝ)
  (H_rent : rent_per_week = 1200)
  (H_utility_percentage : utility_percentage = 0.20)
  (H_employees_per_shift : employees_per_shift = 2)
  (H_hourly_wage : hourly_wage = 12.50)
  (H_days_open : days_per_week_open = 5)
  (H_weekly_expenses : weekly_expenses = 3440) :
  (16 : ℝ) = weekly_expenses / ((rent_per_week * (1 + utility_percentage)) + (employees_per_shift * hourly_wage * days_per_week_open)) :=
by
  sorry

end NUMINAMATH_GPT_store_hours_open_per_day_l2171_217134


namespace NUMINAMATH_GPT_non_defective_probability_l2171_217162

theorem non_defective_probability :
  let p_B := 0.03
  let p_C := 0.01
  let p_def := p_B + p_C
  let p_non_def := 1 - p_def
  p_non_def = 0.96 :=
by
  let p_B := 0.03
  let p_C := 0.01
  let p_def := p_B + p_C
  let p_non_def := 1 - p_def
  sorry

end NUMINAMATH_GPT_non_defective_probability_l2171_217162


namespace NUMINAMATH_GPT_blender_sales_inversely_proportional_l2171_217121

theorem blender_sales_inversely_proportional (k : ℝ) (p : ℝ) (c : ℝ) 
  (h1 : p * c = k) (h2 : 10 * 300 = k) : (p * 600 = k) → p = 5 := 
by
  intros
  sorry

end NUMINAMATH_GPT_blender_sales_inversely_proportional_l2171_217121


namespace NUMINAMATH_GPT_houses_with_both_l2171_217197

theorem houses_with_both (G P N Total B : ℕ) 
  (hG : G = 50) 
  (hP : P = 40) 
  (hN : N = 10) 
  (hTotal : Total = 65)
  (hEquation : G + P - B = Total - N) 
  : B = 35 := 
by 
  sorry

end NUMINAMATH_GPT_houses_with_both_l2171_217197


namespace NUMINAMATH_GPT_fraction_squares_sum_l2171_217116

theorem fraction_squares_sum (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 3)
  (h2 : a / x + b / y + c / z = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := 
sorry

end NUMINAMATH_GPT_fraction_squares_sum_l2171_217116


namespace NUMINAMATH_GPT_largest_prime_factor_of_set_l2171_217137

def largest_prime_factor (n : ℕ) : ℕ :=
  -- pseudo-code for determining the largest prime factor of n
  sorry

lemma largest_prime_factor_45 : largest_prime_factor 45 = 5 := sorry
lemma largest_prime_factor_65 : largest_prime_factor 65 = 13 := sorry
lemma largest_prime_factor_85 : largest_prime_factor 85 = 17 := sorry
lemma largest_prime_factor_119 : largest_prime_factor 119 = 17 := sorry
lemma largest_prime_factor_143 : largest_prime_factor 143 = 13 := sorry

theorem largest_prime_factor_of_set :
  max (largest_prime_factor 45)
    (max (largest_prime_factor 65)
      (max (largest_prime_factor 85)
        (max (largest_prime_factor 119)
          (largest_prime_factor 143)))) = 17 :=
by
  rw [largest_prime_factor_45,
      largest_prime_factor_65,
      largest_prime_factor_85,
      largest_prime_factor_119,
      largest_prime_factor_143]
  sorry

end NUMINAMATH_GPT_largest_prime_factor_of_set_l2171_217137


namespace NUMINAMATH_GPT_minimize_sum_of_squares_of_perpendiculars_l2171_217128

open Real

variable {α β c : ℝ} -- angles and side length

theorem minimize_sum_of_squares_of_perpendiculars
    (habc : α + β = π)
    (P : ℝ)
    (AP BP : ℝ)
    (x : AP + BP = c)
    (u : ℝ)
    (v : ℝ)
    (hAP : AP = P)
    (hBP : BP = c - P)
    (hu : u = P * sin α)
    (hv : v = (c - P) * sin β)
    (f : ℝ)
    (hf : f = (P * sin α)^2 + ((c - P) * sin β)^2):
  (AP / BP = (sin β)^2 / (sin α)^2) := sorry

end NUMINAMATH_GPT_minimize_sum_of_squares_of_perpendiculars_l2171_217128


namespace NUMINAMATH_GPT_total_pies_eq_l2171_217125

-- Definitions for the number of pies made by each person
def pinky_pies : ℕ := 147
def helen_pies : ℕ := 56
def emily_pies : ℕ := 89
def jake_pies : ℕ := 122

-- The theorem stating the total number of pies
theorem total_pies_eq : pinky_pies + helen_pies + emily_pies + jake_pies = 414 :=
by sorry

end NUMINAMATH_GPT_total_pies_eq_l2171_217125


namespace NUMINAMATH_GPT_time_to_cut_mans_hair_l2171_217100

theorem time_to_cut_mans_hair :
  ∃ (x : ℕ),
    (3 * 50) + (2 * x) + (3 * 25) = 255 ∧ x = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_time_to_cut_mans_hair_l2171_217100


namespace NUMINAMATH_GPT_find_k_collinear_l2171_217181

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, -1)
def c : ℝ × ℝ := (1, 2)

theorem find_k_collinear : ∃ k : ℝ, (1 - 2 * k, 3 - k) = (-k, k) * c ∧ k = -1/3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_collinear_l2171_217181


namespace NUMINAMATH_GPT_find_m_eccentricity_l2171_217160

theorem find_m_eccentricity :
  (∃ m : ℝ, (m > 0) ∧ (∃ c : ℝ, (c = 4 - m ∧ c = (1 / 2) * 2) ∨ (c = m - 4 ∧ c = (1 / 2) * 2)) ∧
  (m = 3 ∨ m = 16 / 3)) :=
sorry

end NUMINAMATH_GPT_find_m_eccentricity_l2171_217160


namespace NUMINAMATH_GPT_blue_marbles_initial_count_l2171_217135

variables (x y : ℕ)

theorem blue_marbles_initial_count (h1 : 5 * x = 8 * y) (h2 : 3 * (x - 12) = y + 21) : x = 24 :=
sorry

end NUMINAMATH_GPT_blue_marbles_initial_count_l2171_217135


namespace NUMINAMATH_GPT_solve_first_sales_amount_l2171_217107

noncomputable def first_sales_amount
  (S : ℝ) (R : ℝ) (next_sales_royalties : ℝ) (next_sales_amount : ℝ) : Prop :=
  (3 = R * S) ∧ (next_sales_royalties = 0.85 * R * next_sales_amount)

theorem solve_first_sales_amount (S R : ℝ) :
  first_sales_amount S R 9 108 → S = 30.6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_first_sales_amount_l2171_217107


namespace NUMINAMATH_GPT_product_of_possible_values_of_x_l2171_217173

noncomputable def product_of_roots (a b c : ℤ) : ℤ :=
  c / a

theorem product_of_possible_values_of_x :
  ∃ x : ℝ, (x + 3) * (x - 4) = 18 ∧ product_of_roots 1 (-1) (-30) = -30 := 
by
  sorry

end NUMINAMATH_GPT_product_of_possible_values_of_x_l2171_217173


namespace NUMINAMATH_GPT_problem_l2171_217189

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem problem (h₁ : f a b c 0 = f a b c 4) (h₂ : f a b c 4 > f a b c 1) : a > 0 ∧ 4 * a + b = 0 :=
by 
  sorry

end NUMINAMATH_GPT_problem_l2171_217189


namespace NUMINAMATH_GPT_total_students_at_competition_l2171_217123

variable (K H N : ℕ)

theorem total_students_at_competition
  (H_eq : H = (3/5) * K)
  (N_eq : N = 2 * (K + H))
  (total_students : K + H + N = 240) :
  K + H + N = 240 :=
by
  sorry

end NUMINAMATH_GPT_total_students_at_competition_l2171_217123


namespace NUMINAMATH_GPT_least_number_divisible_increased_by_seven_l2171_217149

theorem least_number_divisible_increased_by_seven : 
  ∃ n : ℕ, (∀ k ∈ [24, 32, 36, 54], (n + 7) % k = 0) ∧ n = 857 := 
by
  sorry

end NUMINAMATH_GPT_least_number_divisible_increased_by_seven_l2171_217149


namespace NUMINAMATH_GPT_largest_possible_m_value_l2171_217112

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem largest_possible_m_value :
  ∃ (m x y : ℕ), is_three_digit m ∧ is_prime x ∧ is_prime y ∧ x ≠ y ∧
  x < 10 ∧ y < 10 ∧ is_prime (10 * x - y) ∧ m = x * y * (10 * x - y) ∧ m = 705 := sorry

end NUMINAMATH_GPT_largest_possible_m_value_l2171_217112


namespace NUMINAMATH_GPT_perimeter_is_32_l2171_217144

-- Define the side lengths of the triangle
def a : ℕ := 13
def b : ℕ := 9
def c : ℕ := 10

-- Definition of the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Theorem stating the perimeter is 32
theorem perimeter_is_32 : perimeter a b c = 32 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_is_32_l2171_217144


namespace NUMINAMATH_GPT_f_neg1_gt_f_1_l2171_217180

-- Definition of the function f and its properties.
variable {f : ℝ → ℝ}
variable (df : Differentiable ℝ f)
variable (eq_f : ∀ x : ℝ, f x = x^2 + 2 * x * f' 2)

-- The problem statement to prove f(-1) > f(1).
theorem f_neg1_gt_f_1 (h_deriv : ∀ x : ℝ, deriv f x = 2 * x - 8):
  f (-1) > f 1 :=
by
  sorry

end NUMINAMATH_GPT_f_neg1_gt_f_1_l2171_217180


namespace NUMINAMATH_GPT_gcd_of_powers_l2171_217145

theorem gcd_of_powers (a b : ℕ) (h1 : a = 2^300 - 1) (h2 : b = 2^315 - 1) :
  gcd a b = 32767 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_powers_l2171_217145


namespace NUMINAMATH_GPT_eldest_child_age_l2171_217195

variables (y m e : ℕ)

theorem eldest_child_age (h1 : m = y + 3)
                        (h2 : e = 3 * y)
                        (h3 : e = y + m + 2) : e = 15 :=
by
  sorry

end NUMINAMATH_GPT_eldest_child_age_l2171_217195


namespace NUMINAMATH_GPT_pencils_in_total_l2171_217167

theorem pencils_in_total
  (rows : ℕ) (pencils_per_row : ℕ) (total_pencils : ℕ)
  (h1 : rows = 14)
  (h2 : pencils_per_row = 11)
  (h3 : total_pencils = rows * pencils_per_row) :
  total_pencils = 154 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_pencils_in_total_l2171_217167


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l2171_217174

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + x + y = 75) : x^2 + y^2 = 3205 / 121 :=
by
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l2171_217174


namespace NUMINAMATH_GPT_positive_number_equals_seven_l2171_217182

theorem positive_number_equals_seven (x : ℝ) (h_pos : x > 0) (h_eq : x - 4 = 21 / x) : x = 7 :=
sorry

end NUMINAMATH_GPT_positive_number_equals_seven_l2171_217182


namespace NUMINAMATH_GPT_distance_center_of_ball_travels_l2171_217122

noncomputable def radius_of_ball : ℝ := 2
noncomputable def R1 : ℝ := 100
noncomputable def R2 : ℝ := 60
noncomputable def R3 : ℝ := 80

noncomputable def adjusted_R1 : ℝ := R1 - radius_of_ball
noncomputable def adjusted_R2 : ℝ := R2 + radius_of_ball
noncomputable def adjusted_R3 : ℝ := R3 - radius_of_ball

noncomputable def distance_travelled : ℝ :=
  (Real.pi * adjusted_R1) +
  (Real.pi * adjusted_R2) +
  (Real.pi * adjusted_R3)

theorem distance_center_of_ball_travels : distance_travelled = 238 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_distance_center_of_ball_travels_l2171_217122


namespace NUMINAMATH_GPT_total_students_l2171_217111

theorem total_students (boys girls : ℕ) (h_boys : boys = 127) (h_girls : girls = boys + 212) : boys + girls = 466 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l2171_217111
