import Mathlib

namespace NUMINAMATH_GPT_square_side_length_l749_74942

theorem square_side_length (A : ℝ) (h : A = 169) : ∃ s : ℝ, s^2 = A ∧ s = 13 := by
  sorry

end NUMINAMATH_GPT_square_side_length_l749_74942


namespace NUMINAMATH_GPT_marcus_leah_together_l749_74946

def num_games_with_combination (n k : ℕ) : ℕ :=
  Nat.choose n k

def num_games_together (total_players players_per_game : ℕ) (games_with_each_combination: ℕ) : ℕ :=
  total_players / players_per_game * games_with_each_combination

/-- Prove that Marcus and Leah play 210 games together. -/
theorem marcus_leah_together :
  let total_players := 12
  let players_per_game := 6
  let total_games := num_games_with_combination total_players players_per_game
  let marc_per_game := total_games / 2
  let together_pcnt := 5 / 11
  together_pcnt * marc_per_game = 210 :=
by
  sorry

end NUMINAMATH_GPT_marcus_leah_together_l749_74946


namespace NUMINAMATH_GPT_shelby_gold_stars_today_l749_74969

-- Define the number of gold stars Shelby earned yesterday
def gold_stars_yesterday := 4

-- Define the total number of gold stars Shelby earned
def total_gold_stars := 7

-- Define the number of gold stars Shelby earned today
def gold_stars_today := total_gold_stars - gold_stars_yesterday

-- The theorem to prove
theorem shelby_gold_stars_today : gold_stars_today = 3 :=
by 
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_shelby_gold_stars_today_l749_74969


namespace NUMINAMATH_GPT_number_of_b_objects_l749_74997

theorem number_of_b_objects
  (total_objects : ℕ) 
  (a_objects : ℕ) 
  (b_objects : ℕ) 
  (h1 : total_objects = 35) 
  (h2 : a_objects = 17) 
  (h3 : total_objects = a_objects + b_objects) :
  b_objects = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_b_objects_l749_74997


namespace NUMINAMATH_GPT_find_coefficients_l749_74943

theorem find_coefficients (A B : ℚ) :
  (∀ x : ℚ, 2 * x + 7 = A * (x + 7) + B * (x - 9)) →
  A = 25 / 16 ∧ B = 7 / 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_coefficients_l749_74943


namespace NUMINAMATH_GPT_number_of_subsets_including_1_and_10_l749_74998

def A : Set ℕ := {a : ℕ | ∃ x y z : ℕ, a = 2^x * 3^y * 5^z}
def B : Set ℕ := {b : ℕ | b ∈ A ∧ 1 ≤ b ∧ b ≤ 10}

theorem number_of_subsets_including_1_and_10 :
  ∃ (s : Finset (Finset ℕ)), (∀ x ∈ s, 1 ∈ x ∧ 10 ∈ x) ∧ s.card = 128 := by
  sorry

end NUMINAMATH_GPT_number_of_subsets_including_1_and_10_l749_74998


namespace NUMINAMATH_GPT_sum_of_real_values_l749_74958

theorem sum_of_real_values (x : ℝ) (h : |3 * x - 15| + |x - 5| = 92) : (x = 28 ∨ x = -18) → x + 10 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_real_values_l749_74958


namespace NUMINAMATH_GPT_age_is_nine_l749_74933

-- Define the conditions
def current_age (X : ℕ) :=
  X = 3 * (X - 6)

-- The theorem: Prove that the age X is equal to 9 under the conditions given
theorem age_is_nine (X : ℕ) (h : current_age X) : X = 9 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_age_is_nine_l749_74933


namespace NUMINAMATH_GPT_minimum_value_fraction_l749_74949

theorem minimum_value_fraction (a : ℝ) (h : a > 1) : (a^2 - a + 1) / (a - 1) ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_fraction_l749_74949


namespace NUMINAMATH_GPT_sunflower_seeds_more_than_half_on_day_three_l749_74985

-- Define the initial state and parameters
def initial_sunflower_seeds : ℚ := 0.4
def initial_other_seeds : ℚ := 0.6
def daily_added_sunflower_seeds : ℚ := 0.2
def daily_added_other_seeds : ℚ := 0.3
def daily_sunflower_eaten_factor : ℚ := 0.7
def daily_other_eaten_factor : ℚ := 0.4

-- Define the recurrence relations for sunflower seeds and total seeds
def sunflower_seeds (n : ℕ) : ℚ :=
  match n with
  | 0     => initial_sunflower_seeds
  | (n+1) => daily_sunflower_eaten_factor * sunflower_seeds n + daily_added_sunflower_seeds

def total_seeds (n : ℕ) : ℚ := 1 + (n : ℚ) * 0.5

-- Define the main theorem stating that on Tuesday (Day 3), sunflower seeds are more than half
theorem sunflower_seeds_more_than_half_on_day_three : sunflower_seeds 2 / total_seeds 2 > 0.5 :=
by
  -- Formal proof will go here
  sorry

end NUMINAMATH_GPT_sunflower_seeds_more_than_half_on_day_three_l749_74985


namespace NUMINAMATH_GPT_systematic_sampling_interval_l749_74903

-- Definitions based on the conditions in part a)
def total_students : ℕ := 1500
def sample_size : ℕ := 30

-- The goal is to prove that the interval k in systematic sampling equals 50
theorem systematic_sampling_interval :
  (total_students / sample_size = 50) :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_interval_l749_74903


namespace NUMINAMATH_GPT_domain_of_v_l749_74977

noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (Real.cos x)

theorem domain_of_v :
  (∀ x : ℝ, (∃ n : ℤ, 2 * n * Real.pi - Real.pi / 2 < x ∧ x < 2 * n * Real.pi + Real.pi / 2) ↔ 
    ∀ x : ℝ, ∀ x_in_domain : ℝ, (0 < Real.cos x ∧ 1 / Real.sqrt (Real.cos x) = x_in_domain)) :=
sorry

end NUMINAMATH_GPT_domain_of_v_l749_74977


namespace NUMINAMATH_GPT_inequality_transformation_l749_74915

theorem inequality_transformation (m n : ℝ) (h : -m / 2 < -n / 6) : 3 * m > n := by
  sorry

end NUMINAMATH_GPT_inequality_transformation_l749_74915


namespace NUMINAMATH_GPT_sin_45_eq_sqrt2_div_2_l749_74999

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end NUMINAMATH_GPT_sin_45_eq_sqrt2_div_2_l749_74999


namespace NUMINAMATH_GPT_sin_P_equals_one_l749_74908

theorem sin_P_equals_one
  (x y : ℝ) (h1 : (1 / 2) * x * y * Real.sin 1 = 50) (h2 : x * y = 100) :
  Real.sin 1 = 1 :=
by sorry

end NUMINAMATH_GPT_sin_P_equals_one_l749_74908


namespace NUMINAMATH_GPT_melanie_total_amount_l749_74955

theorem melanie_total_amount :
  let g1 := 12
  let g2 := 15
  let g3 := 8
  let g4 := 10
  let g5 := 20
  g1 + g2 + g3 + g4 + g5 = 65 :=
by
  sorry

end NUMINAMATH_GPT_melanie_total_amount_l749_74955


namespace NUMINAMATH_GPT_initial_distance_l749_74963

-- Definitions based on conditions
def speed_thief : ℝ := 8 -- in km/hr
def speed_policeman : ℝ := 10 -- in km/hr
def distance_thief_runs : ℝ := 0.7 -- in km

-- Theorem statement
theorem initial_distance
  (relative_speed := speed_policeman - speed_thief) -- Relative speed (in km/hr)
  (time_to_overtake := distance_thief_runs / relative_speed) -- Time for the policeman to overtake the thief (in hours)
  (initial_distance := speed_policeman * time_to_overtake) -- Initial distance (in km)
  : initial_distance = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_initial_distance_l749_74963


namespace NUMINAMATH_GPT_number_of_steaks_needed_l749_74995

-- Definitions based on the conditions
def family_members : ℕ := 5
def pounds_per_member : ℕ := 1
def ounces_per_pound : ℕ := 16
def ounces_per_steak : ℕ := 20

-- Prove the number of steaks needed equals 4
theorem number_of_steaks_needed : (family_members * pounds_per_member * ounces_per_pound) / ounces_per_steak = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_steaks_needed_l749_74995


namespace NUMINAMATH_GPT_remainder_div_x_plus_1_l749_74907

noncomputable def polynomial1 : Polynomial ℝ := Polynomial.X ^ 11 - 1

theorem remainder_div_x_plus_1 :
  Polynomial.eval (-1) polynomial1 = -2 := by
  sorry

end NUMINAMATH_GPT_remainder_div_x_plus_1_l749_74907


namespace NUMINAMATH_GPT_ellipse_condition_l749_74986

theorem ellipse_condition (m n : ℝ) :
  (mn > 0) → (¬ (∃ x y : ℝ, (m = 1) ∧ (n = 1) ∧ (x^2)/m + (y^2)/n = 1 ∧ (x, y) ≠ (0,0))) :=
sorry

end NUMINAMATH_GPT_ellipse_condition_l749_74986


namespace NUMINAMATH_GPT_range_of_expression_l749_74924

theorem range_of_expression (x : ℝ) (h1 : 1 - 3 * x ≥ 0) (h2 : 2 * x ≠ 0) : x ≤ 1 / 3 ∧ x ≠ 0 := by
  sorry

end NUMINAMATH_GPT_range_of_expression_l749_74924


namespace NUMINAMATH_GPT_geometric_series_sum_test_l749_74947

-- Let's define all necessary variables
variable (a : ℤ) (r : ℤ) (n : ℕ)

-- Define the geometric series sum formula
noncomputable def geometric_series_sum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r ^ n - 1) / (r - 1))

-- Define the specific test case as per our conditions
theorem geometric_series_sum_test :
  geometric_series_sum (-2) 3 7 = -2186 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_test_l749_74947


namespace NUMINAMATH_GPT_cost_of_baking_soda_l749_74994

-- Definitions of the condition
def students : ℕ := 23
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def total_cost_of_supplies : ℕ := 184

-- Main statement to prove
theorem cost_of_baking_soda : 
  (∀ (students : ℕ) (cost_of_bow : ℕ) (cost_of_vinegar : ℕ) (total_cost_of_supplies : ℕ),
    total_cost_of_supplies = students * (cost_of_bow + cost_of_vinegar) + students) → 
  total_cost_of_supplies = 23 * (5 + 2) + 23 → 
  184 = 23 * (5 + 2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_cost_of_baking_soda_l749_74994


namespace NUMINAMATH_GPT_carol_initial_cupcakes_l749_74920

/--
For the school bake sale, Carol made some cupcakes. She sold 9 of them and then made 28 more.
Carol had 49 cupcakes. We need to show that Carol made 30 cupcakes initially.
-/
theorem carol_initial_cupcakes (x : ℕ) 
  (h1 : x - 9 + 28 = 49) : 
  x = 30 :=
by 
  -- The proof is not required as per instruction.
  sorry

end NUMINAMATH_GPT_carol_initial_cupcakes_l749_74920


namespace NUMINAMATH_GPT_rectangle_area_increase_l749_74966

theorem rectangle_area_increase (x y : ℕ) 
  (hxy : x * y = 180) 
  (hperimeter : 2 * x + 2 * y = 54) : 
  (x + 6) * (y + 6) = 378 :=
by sorry

end NUMINAMATH_GPT_rectangle_area_increase_l749_74966


namespace NUMINAMATH_GPT_continuity_at_x_0_l749_74951

def f (x : ℝ) := -2 * x^2 + 9
def x_0 : ℝ := 4

theorem continuity_at_x_0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x_0| < δ → |f x - f x_0| < ε :=
by
  sorry

end NUMINAMATH_GPT_continuity_at_x_0_l749_74951


namespace NUMINAMATH_GPT_cannot_determine_right_triangle_l749_74910

/-- Proof that the condition \(a^2 = 5\), \(b^2 = 12\), \(c^2 = 13\) cannot determine that \(\triangle ABC\) is a right triangle. -/
theorem cannot_determine_right_triangle (a b c : ℝ) (ha : a^2 = 5) (hb : b^2 = 12) (hc : c^2 = 13) : 
  ¬(a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := 
by
  sorry

end NUMINAMATH_GPT_cannot_determine_right_triangle_l749_74910


namespace NUMINAMATH_GPT_undefined_expression_value_l749_74917

theorem undefined_expression_value {a : ℝ} : (a^3 - 8 = 0) ↔ (a = 2) :=
by sorry

end NUMINAMATH_GPT_undefined_expression_value_l749_74917


namespace NUMINAMATH_GPT_smallest_positive_multiple_l749_74960

theorem smallest_positive_multiple (a : ℕ) (h₁ : a % 6 = 0) (h₂ : a % 15 = 0) : a = 30 :=
sorry

end NUMINAMATH_GPT_smallest_positive_multiple_l749_74960


namespace NUMINAMATH_GPT_union_cardinality_inequality_l749_74940

open Set

/-- Given three finite sets A, B, and C such that A ∩ B ∩ C = ∅,
prove that |A ∪ B ∪ C| ≥ 1/2 (|A| + |B| + |C|) -/
theorem union_cardinality_inequality (A B C : Finset ℕ)
  (h : (A ∩ B ∩ C) = ∅) : (A ∪ B ∪ C).card ≥ (A.card + B.card + C.card) / 2 := sorry

end NUMINAMATH_GPT_union_cardinality_inequality_l749_74940


namespace NUMINAMATH_GPT_max_hawthorns_satisfying_conditions_l749_74913

theorem max_hawthorns_satisfying_conditions :
  ∃ x : ℕ, 
    x > 100 ∧ 
    x % 3 = 1 ∧ 
    x % 4 = 2 ∧ 
    x % 5 = 3 ∧ 
    x % 6 = 4 ∧ 
    (∀ y : ℕ, 
      y > 100 ∧ 
      y % 3 = 1 ∧ 
      y % 4 = 2 ∧ 
      y % 5 = 3 ∧ 
      y % 6 = 4 → y ≤ 178) :=
sorry

end NUMINAMATH_GPT_max_hawthorns_satisfying_conditions_l749_74913


namespace NUMINAMATH_GPT_linear_regression_equation_l749_74975

-- Given conditions
variables (x y : ℝ)
variable (corr_pos : x ≠ 0 → y / x > 0)
noncomputable def x_mean : ℝ := 2.4
noncomputable def y_mean : ℝ := 3.2

-- Regression line equation
theorem linear_regression_equation :
  (y = 0.5 * x + 2) ∧ (∀ x' y', (x' = x_mean ∧ y' = y_mean) → (y' = 0.5 * x' + 2)) :=
by
  sorry

end NUMINAMATH_GPT_linear_regression_equation_l749_74975


namespace NUMINAMATH_GPT_find_alpha_plus_beta_l749_74901

theorem find_alpha_plus_beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos α = (Real.sqrt 5) / 5) (h4 : Real.sin β = (3 * Real.sqrt 10) / 10) : 
  α + β = 3 * π / 4 :=
sorry

end NUMINAMATH_GPT_find_alpha_plus_beta_l749_74901


namespace NUMINAMATH_GPT_infinite_series_sum_eq_l749_74927

noncomputable def infinite_series_sum : Rat :=
  ∑' n : ℕ, (2 * n + 1) * (2000⁻¹) ^ n

theorem infinite_series_sum_eq : infinite_series_sum = (2003000 / 3996001) := by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_eq_l749_74927


namespace NUMINAMATH_GPT_find_missing_number_l749_74957

theorem find_missing_number
  (x y : ℕ)
  (h1 : 30 = 6 * 5)
  (h2 : 600 = 30 * x)
  (h3 : x = 5 * y) :
  y = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l749_74957


namespace NUMINAMATH_GPT_find_a_9_l749_74952

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (d : ℤ)

-- Assumptions and definitions from the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def condition_one (a : ℕ → ℤ) : Prop := (a 1) + (a 2)^2 = -3
def condition_two (S : ℕ → ℤ) : Prop := S 5 = 10

-- Main theorem statement
theorem find_a_9 (h_arithmetic : arithmetic_sequence a d)
                 (h_sum : sum_of_arithmetic_sequence S a)
                 (h_cond1 : condition_one a)
                 (h_cond2 : condition_two S) : a 9 = 20 := 
sorry

end NUMINAMATH_GPT_find_a_9_l749_74952


namespace NUMINAMATH_GPT_solution_set_inequality_x0_1_solution_set_inequality_x0_half_l749_74906

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem solution_set_inequality_x0_1 : 
  ∀ (c : ℝ), (∀ x, 0 < x → f x - f 1 ≥ c * (x - 1)) ↔ c ∈ Set.Icc (-1) 1 := 
by
  sorry

theorem solution_set_inequality_x0_half : 
  ∀ (c : ℝ), (∀ x, 0 < x → f x - f (1 / 2) ≥ c * (x - 1 / 2)) ↔ c = -2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_x0_1_solution_set_inequality_x0_half_l749_74906


namespace NUMINAMATH_GPT_percentage_died_by_bombardment_l749_74950

noncomputable def initial_population : ℕ := 8515
noncomputable def final_population : ℕ := 6514

theorem percentage_died_by_bombardment :
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ 100) ∧
  8515 - ((x / 100) * 8515) - (15 / 100) * (8515 - ((x / 100) * 8515)) = 6514 ∧
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_died_by_bombardment_l749_74950


namespace NUMINAMATH_GPT_find_x_l749_74979

theorem find_x (x : ℕ) : (x > 20) ∧ (x < 120) ∧ (∃ y : ℕ, x = y^2) ∧ (x % 3 = 0) ↔ (x = 36) ∨ (x = 81) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l749_74979


namespace NUMINAMATH_GPT_cos_sin_sum_l749_74911

open Real

theorem cos_sin_sum (α : ℝ) (h : (cos (2 * α)) / (sin (α - π / 4)) = -sqrt 2 / 2) : cos α + sin α = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_sin_sum_l749_74911


namespace NUMINAMATH_GPT_mixed_number_calculation_l749_74974

theorem mixed_number_calculation :
  47 * (2 + 2/3 - (3 + 1/4)) / (3 + 1/2 + (2 + 1/5)) = -4 - 25/38 :=
by
  sorry

end NUMINAMATH_GPT_mixed_number_calculation_l749_74974


namespace NUMINAMATH_GPT_solve_boys_left_l749_74937

--given conditions
variable (boys_initial girls_initial boys_left girls_entered children_end: ℕ)
variable (h_boys_initial : boys_initial = 5)
variable (h_girls_initial : girls_initial = 4)
variable (h_girls_entered : girls_entered = 2)
variable (h_children_end : children_end = 8)

-- Problem definition
def boys_left_proof : Prop :=
  ∃ (B : ℕ), boys_left = B ∧ boys_initial - B + girls_initial + girls_entered = children_end ∧ B = 3

-- The statement to be proven
theorem solve_boys_left : boys_left_proof boys_initial girls_initial boys_left girls_entered children_end := by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_solve_boys_left_l749_74937


namespace NUMINAMATH_GPT_nonnegative_fraction_iff_interval_l749_74983

theorem nonnegative_fraction_iff_interval (x : ℝ) : 
  0 ≤ x ∧ x < 3 ↔ 0 ≤ (x^2 - 12 * x^3 + 36 * x^4) / (9 - x^3) := by
  sorry

end NUMINAMATH_GPT_nonnegative_fraction_iff_interval_l749_74983


namespace NUMINAMATH_GPT_first_group_checked_correctly_l749_74923

-- Define the given conditions
def total_factories : ℕ := 169
def checked_by_second_group : ℕ := 52
def remaining_unchecked : ℕ := 48

-- Define the number of factories checked by the first group
def checked_by_first_group : ℕ := total_factories - checked_by_second_group - remaining_unchecked

-- State the theorem to be proved
theorem first_group_checked_correctly : checked_by_first_group = 69 :=
by
  -- The proof is not provided, use sorry to skip the proof steps
  sorry

end NUMINAMATH_GPT_first_group_checked_correctly_l749_74923


namespace NUMINAMATH_GPT_value_of_a_l749_74929

theorem value_of_a (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : (a + b + c) / 3 = 4 * b) (h4 : c / b = 11) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l749_74929


namespace NUMINAMATH_GPT_find_x_for_parallel_vectors_l749_74964

noncomputable def vector_m : (ℝ × ℝ) := (1, 2)
noncomputable def vector_n (x : ℝ) : (ℝ × ℝ) := (x, 2 - 2 * x)

theorem find_x_for_parallel_vectors :
  ∀ x : ℝ, (1, 2).fst * (2 - 2 * x) - (1, 2).snd * x = 0 → x = 1 / 2 :=
by
  intros
  exact sorry

end NUMINAMATH_GPT_find_x_for_parallel_vectors_l749_74964


namespace NUMINAMATH_GPT_jamie_dimes_l749_74925

theorem jamie_dimes (p n d : ℕ) (h1 : p + n + d = 50) (h2 : p + 5 * n + 10 * d = 240) : d = 10 :=
sorry

end NUMINAMATH_GPT_jamie_dimes_l749_74925


namespace NUMINAMATH_GPT_canoes_rented_more_than_kayaks_l749_74941

-- Defining the constants
def canoe_cost : ℕ := 11
def kayak_cost : ℕ := 16
def total_revenue : ℕ := 460
def canoe_ratio : ℕ := 4
def kayak_ratio : ℕ := 3

-- Main statement to prove
theorem canoes_rented_more_than_kayaks :
  ∃ (C K : ℕ), canoe_cost * C + kayak_cost * K = total_revenue ∧ (canoe_ratio * K = kayak_ratio * C) ∧ (C - K = 5) :=
by
  have h1 : canoe_cost = 11 := rfl
  have h2 : kayak_cost = 16 := rfl
  have h3 : total_revenue = 460 := rfl
  have h4 : canoe_ratio = 4 := rfl
  have h5 : kayak_ratio = 3 := rfl
  sorry

end NUMINAMATH_GPT_canoes_rented_more_than_kayaks_l749_74941


namespace NUMINAMATH_GPT_average_price_of_5_baskets_l749_74962

/-- Saleem bought 4 baskets with an average cost of $4 each. --/
def average_cost_first_4_baskets : ℝ := 4

/-- Saleem buys the fifth basket with the price of $8. --/
def price_fifth_basket : ℝ := 8

/-- Prove that the average price of the 5 baskets is $4.80. --/
theorem average_price_of_5_baskets :
  (4 * average_cost_first_4_baskets + price_fifth_basket) / 5 = 4.80 := 
by
  sorry

end NUMINAMATH_GPT_average_price_of_5_baskets_l749_74962


namespace NUMINAMATH_GPT_no_such_function_exists_l749_74944

theorem no_such_function_exists :
  ¬(∃ (f : ℝ → ℝ), ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2) :=
sorry

end NUMINAMATH_GPT_no_such_function_exists_l749_74944


namespace NUMINAMATH_GPT_sum_x_y_eq_two_l749_74936

theorem sum_x_y_eq_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 28) : x + y = 2 :=
sorry

end NUMINAMATH_GPT_sum_x_y_eq_two_l749_74936


namespace NUMINAMATH_GPT_cricket_count_l749_74965

theorem cricket_count (x : ℕ) (h : x + 11 = 18) : x = 7 :=
by sorry

end NUMINAMATH_GPT_cricket_count_l749_74965


namespace NUMINAMATH_GPT_number_of_selection_plans_l749_74948

-- Definitions based on conditions
def male_students : Nat := 5
def female_students : Nat := 4
def total_volunteers : Nat := 3

def choose (n k : Nat) : Nat :=
  Nat.choose n k

def arrangement_count : Nat :=
  Nat.factorial total_volunteers

-- Theorem that states the total number of selection plans
theorem number_of_selection_plans :
  (choose male_students 2 * choose female_students 1 + choose male_students 1 * choose female_students 2) * arrangement_count = 420 :=
by
  sorry

end NUMINAMATH_GPT_number_of_selection_plans_l749_74948


namespace NUMINAMATH_GPT_sum_simplest_form_probability_eq_7068_l749_74987

/-- A jar has 15 red candies and 20 blue candies. Terry picks three candies at random,
    then Mary picks three of the remaining candies at random.
    Given that the probability that they get the same color combination (all reds or all blues, irrespective of order),
    find this probability in the simplest form. The sum of the numerator and denominator in simplest form is: 7068. -/
noncomputable def problem_statement : Nat :=
  let total_candies := 15 + 20;
  let terry_red_prob := (15 * 14 * 13) / (total_candies * (total_candies - 1) * (total_candies - 2));
  let mary_red_prob := (12 * 11 * 10) / ((total_candies - 3) * (total_candies - 4) * (total_candies - 5));
  let both_red := terry_red_prob * mary_red_prob;

  let terry_blue_prob := (20 * 19 * 18) / (total_candies * (total_candies - 1) * (total_candies - 2));
  let mary_blue_prob := (17 * 16 * 15) / ((total_candies - 3) * (total_candies - 4) * (total_candies - 5));
  let both_blue := terry_blue_prob * mary_blue_prob;

  let total_probability := both_red + both_blue;
  let simplest := 243 / 6825; -- This should be simplified form
  243 + 6825 -- Sum of numerator and denominator

theorem sum_simplest_form_probability_eq_7068 : problem_statement = 7068 :=
by sorry

end NUMINAMATH_GPT_sum_simplest_form_probability_eq_7068_l749_74987


namespace NUMINAMATH_GPT_final_price_lower_than_budget_l749_74993

theorem final_price_lower_than_budget :
  let budget := 1500
  let T := 750 -- budget equally split for TV
  let S := 750 -- budget equally split for Sound System
  let TV_price_with_discount := (T - 150) * 0.80
  let SoundSystem_price_with_discount := S * 0.85
  let combined_price_before_tax := TV_price_with_discount + SoundSystem_price_with_discount
  let final_price_with_tax := combined_price_before_tax * 1.08
  budget - final_price_with_tax = 293.10 :=
by
  sorry

end NUMINAMATH_GPT_final_price_lower_than_budget_l749_74993


namespace NUMINAMATH_GPT_youseff_distance_l749_74900

theorem youseff_distance (x : ℕ) 
  (walk_time_per_block : ℕ := 1)
  (bike_time_per_block_secs : ℕ := 20)
  (time_difference : ℕ := 12) :
  (x : ℕ) = 18 :=
by
  -- walking time
  let walk_time := x * walk_time_per_block
  
  -- convert bike time per block to minutes
  let bike_time_per_block := (bike_time_per_block_secs : ℚ) / 60

  -- biking time
  let bike_time := x * bike_time_per_block

  -- set up the equation for time difference
  have time_eq := walk_time - bike_time = time_difference
  
  -- from here, the actual proof steps would follow, 
  -- but we include "sorry" as a placeholder since the focus is on the statement.
  sorry

end NUMINAMATH_GPT_youseff_distance_l749_74900


namespace NUMINAMATH_GPT_necessarily_positive_l749_74991

-- Conditions
variables {x y z : ℝ}

-- Statement to prove
theorem necessarily_positive (h1 : 0 < x) (h2 : x < 1) (h3 : -2 < y) (h4 : y < 0) (h5 : 2 < z) (h6 : z < 3) :
  0 < y + 2 * z :=
sorry

end NUMINAMATH_GPT_necessarily_positive_l749_74991


namespace NUMINAMATH_GPT_total_weight_full_bucket_l749_74992

theorem total_weight_full_bucket (x y c d : ℝ) 
(h1 : x + 3/4 * y = c) 
(h2 : x + 1/3 * y = d) :
x + y = (8 * c - 3 * d) / 5 :=
sorry

end NUMINAMATH_GPT_total_weight_full_bucket_l749_74992


namespace NUMINAMATH_GPT_minimum_value_768_l749_74912

noncomputable def min_value_expression (a b c : ℝ) := a^2 + 8 * a * b + 16 * b^2 + 2 * c^5

theorem minimum_value_768 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_condition : a * b^2 * c^3 = 256) : 
  min_value_expression a b c = 768 :=
sorry

end NUMINAMATH_GPT_minimum_value_768_l749_74912


namespace NUMINAMATH_GPT_percent_enclosed_by_hexagons_l749_74926

variable (b : ℝ) -- side length of smaller squares

def area_of_small_square : ℝ := b^2
def area_of_large_square : ℝ := 16 * area_of_small_square b
def area_of_hexagon : ℝ := 3 * area_of_small_square b
def total_area_of_hexagons : ℝ := 2 * area_of_hexagon b

theorem percent_enclosed_by_hexagons :
  (total_area_of_hexagons b / area_of_large_square b) * 100 = 37.5 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_percent_enclosed_by_hexagons_l749_74926


namespace NUMINAMATH_GPT_xiaochun_age_l749_74981

theorem xiaochun_age
  (x y : ℕ)
  (h1 : x = y - 18)
  (h2 : 2 * (x + 3) = y + 3) :
  x = 15 :=
sorry

end NUMINAMATH_GPT_xiaochun_age_l749_74981


namespace NUMINAMATH_GPT_operation_correct_l749_74938

def operation (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem operation_correct :
  operation 4 2 = 18 :=
by
  show 2 * 4 + 5 * 2 = 18
  sorry

end NUMINAMATH_GPT_operation_correct_l749_74938


namespace NUMINAMATH_GPT_place_value_ratio_l749_74921

theorem place_value_ratio :
  let d8_place := 0.1
  let d7_place := 10
  d8_place / d7_place = 0.01 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_place_value_ratio_l749_74921


namespace NUMINAMATH_GPT_find_e_l749_74990

theorem find_e (d e f : ℝ) (h1 : f = 5)
  (h2 : -d / 3 = -f)
  (h3 : -f = 1 + d + e + f) :
  e = -26 := 
by
  sorry

end NUMINAMATH_GPT_find_e_l749_74990


namespace NUMINAMATH_GPT_flowchart_correct_option_l749_74988

-- Definitions based on conditions
def typical_flowchart (start_points end_points : ℕ) : Prop :=
  start_points = 1 ∧ end_points ≥ 1

-- Theorem to prove
theorem flowchart_correct_option :
  ∃ (start_points end_points : ℕ), typical_flowchart start_points end_points ∧ "Option C" = "Option C" :=
by {
  sorry -- This part skips the proof itself,
}

end NUMINAMATH_GPT_flowchart_correct_option_l749_74988


namespace NUMINAMATH_GPT_evaluate_expression_is_41_l749_74996

noncomputable def evaluate_expression : ℚ :=
  (121 * (1 / 13 - 1 / 17) + 169 * (1 / 17 - 1 / 11) + 289 * (1 / 11 - 1 / 13)) /
  (11 * (1 / 13 - 1 / 17) + 13 * (1 / 17 - 1 / 11) + 17 * (1 / 11 - 1 / 13))

theorem evaluate_expression_is_41 : evaluate_expression = 41 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_is_41_l749_74996


namespace NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l749_74945

noncomputable def a3 := (2 : ℚ) / 11
noncomputable def a11 := (5 : ℚ) / 6

noncomputable def a7 := (a3 + a11) / 2

theorem arithmetic_sequence_seventh_term :
  a7 = 67 / 132 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l749_74945


namespace NUMINAMATH_GPT_value_of_x_l749_74930

theorem value_of_x (x : ℝ) (h : (x / 5 / 3) = (5 / (x / 3))) : x = 15 ∨ x = -15 := 
by sorry

end NUMINAMATH_GPT_value_of_x_l749_74930


namespace NUMINAMATH_GPT_sequence_formula_l749_74972

noncomputable def a : ℕ → ℕ
| 0       => 2
| (n + 1) => a n ^ 2 - n * a n + 1

theorem sequence_formula (n : ℕ) : a n = n + 2 :=
by
  induction n with
  | zero => sorry
  | succ n ih => sorry

end NUMINAMATH_GPT_sequence_formula_l749_74972


namespace NUMINAMATH_GPT_functional_equation_solution_l749_74967

theorem functional_equation_solution {
  f : ℝ → ℝ
} (h : ∀ x y : ℝ, f ((x - y)^2) = x^2 - 2 * y * f x + (f y)^2) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = x + 1) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l749_74967


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l749_74934

theorem boat_speed_in_still_water (b : ℝ) (h : (36 / (b - 2)) - (36 / (b + 2)) = 1.5) : b = 10 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l749_74934


namespace NUMINAMATH_GPT_gcd_45736_123456_l749_74932

theorem gcd_45736_123456 : Nat.gcd 45736 123456 = 352 :=
by sorry

end NUMINAMATH_GPT_gcd_45736_123456_l749_74932


namespace NUMINAMATH_GPT_minimum_bounces_to_reach_height_l749_74914

noncomputable def height_after_bounces (initial_height : ℝ) (bounce_factor : ℝ) (k : ℕ) : ℝ :=
  initial_height * (bounce_factor ^ k)

theorem minimum_bounces_to_reach_height
  (initial_height : ℝ) (bounce_factor : ℝ) (min_height : ℝ) :
  initial_height = 800 → bounce_factor = 0.5 → min_height = 2 →
  (∀ k : ℕ, height_after_bounces initial_height bounce_factor k < min_height ↔ k ≥ 9) := 
by
  intros h₀ b₀ m₀
  rw [h₀, b₀, m₀]
  sorry

end NUMINAMATH_GPT_minimum_bounces_to_reach_height_l749_74914


namespace NUMINAMATH_GPT_rectangular_plot_area_l749_74973

theorem rectangular_plot_area (P : ℝ) (L W : ℝ) (h1 : P = 24) (h2 : L = 2 * W) :
    A = 32 := by
  sorry

end NUMINAMATH_GPT_rectangular_plot_area_l749_74973


namespace NUMINAMATH_GPT_three_pow_2040_mod_5_l749_74956

theorem three_pow_2040_mod_5 : (3^2040) % 5 = 1 := by
  sorry

end NUMINAMATH_GPT_three_pow_2040_mod_5_l749_74956


namespace NUMINAMATH_GPT_total_pens_bought_l749_74922

-- Define the problem conditions
def pens_given_to_friends : ℕ := 22
def pens_kept_for_herself : ℕ := 34

-- Theorem statement
theorem total_pens_bought : pens_given_to_friends + pens_kept_for_herself = 56 := by
  sorry

end NUMINAMATH_GPT_total_pens_bought_l749_74922


namespace NUMINAMATH_GPT_wall_with_5_peaks_has_14_cubes_wall_with_2014_peaks_has_6041_cubes_painted_area_wall_with_2014_peaks_l749_74909

noncomputable def number_of_cubes (n : ℕ) : ℕ :=
  n + (n - 1) + n

noncomputable def painted_area (n : ℕ) : ℕ :=
  (5 * n) + (3 * (n + 1)) + (2 * (n - 2))

theorem wall_with_5_peaks_has_14_cubes : number_of_cubes 5 = 14 :=
  by sorry

theorem wall_with_2014_peaks_has_6041_cubes : number_of_cubes 2014 = 6041 :=
  by sorry

theorem painted_area_wall_with_2014_peaks : painted_area 2014 = 20139 :=
  by sorry

end NUMINAMATH_GPT_wall_with_5_peaks_has_14_cubes_wall_with_2014_peaks_has_6041_cubes_painted_area_wall_with_2014_peaks_l749_74909


namespace NUMINAMATH_GPT_determine_m_l749_74931

-- Definition of complex numbers z1 and z2
def z1 (m : ℝ) : ℂ := m + 2 * Complex.I
def z2 : ℂ := 2 + Complex.I

-- Condition that the product z1 * z2 is a pure imaginary number
def pure_imaginary (c : ℂ) : Prop := c.re = 0 

-- The proof statement
theorem determine_m (m : ℝ) : pure_imaginary (z1 m * z2) → m = 1 := 
sorry

end NUMINAMATH_GPT_determine_m_l749_74931


namespace NUMINAMATH_GPT_distance_between_trees_l749_74968

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (num_spaces : ℕ) (distance : ℕ)
  (h1 : yard_length = 180)
  (h2 : num_trees = 11)
  (h3 : num_spaces = num_trees - 1)
  (h4 : distance = yard_length / num_spaces) :
  distance = 18 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l749_74968


namespace NUMINAMATH_GPT_simplify_fraction_l749_74954

variable (x y : ℝ)
variable (h1 : x ≠ 0)
variable (h2 : y ≠ 0)
variable (h3 : x - y^2 ≠ 0)

theorem simplify_fraction :
  (y^2 - 1/x) / (x - y^2) = (x * y^2 - 1) / (x^2 - x * y^2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l749_74954


namespace NUMINAMATH_GPT_biker_distance_and_speed_l749_74916

variable (D V : ℝ)

theorem biker_distance_and_speed (h1 : D / 2 = V * 2.5)
                                  (h2 : D / 2 = (V + 2) * (7 / 3)) :
  D = 140 ∧ V = 28 :=
by
  sorry

end NUMINAMATH_GPT_biker_distance_and_speed_l749_74916


namespace NUMINAMATH_GPT_min_value_a_2b_l749_74976

theorem min_value_a_2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = a * b) :
  a + 2 * b ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_a_2b_l749_74976


namespace NUMINAMATH_GPT_astronaut_days_on_orbius_l749_74982

noncomputable def days_in_year : ℕ := 250
noncomputable def seasons_in_year : ℕ := 5
noncomputable def seasons_stayed : ℕ := 3

theorem astronaut_days_on_orbius :
  (days_in_year / seasons_in_year) * seasons_stayed = 150 := by
  sorry

end NUMINAMATH_GPT_astronaut_days_on_orbius_l749_74982


namespace NUMINAMATH_GPT_quoted_value_of_stock_l749_74980

theorem quoted_value_of_stock (F P : ℝ) (h1 : F > 0) (h2 : P = 1.25 * F) : 
  (0.10 * F) / P = 0.08 := 
sorry

end NUMINAMATH_GPT_quoted_value_of_stock_l749_74980


namespace NUMINAMATH_GPT_parabola_focus_coords_l749_74918

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus coordinates
def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

-- The math proof problem statement
theorem parabola_focus_coords :
  ∀ x y, parabola x y → focus x y :=
by
  intros x y hp
  sorry

end NUMINAMATH_GPT_parabola_focus_coords_l749_74918


namespace NUMINAMATH_GPT_ratio_night_to_day_l749_74953

-- Definitions based on conditions
def birds_day : ℕ := 8
def birds_total : ℕ := 24
def birds_night : ℕ := birds_total - birds_day

-- Theorem statement
theorem ratio_night_to_day : birds_night / birds_day = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_night_to_day_l749_74953


namespace NUMINAMATH_GPT_son_completion_time_l749_74928

theorem son_completion_time (M S F : ℝ) 
  (h1 : M = 1 / 10) 
  (h2 : M + S = 1 / 5) 
  (h3 : S + F = 1 / 4) : 
  1 / S = 10 := 
  sorry

end NUMINAMATH_GPT_son_completion_time_l749_74928


namespace NUMINAMATH_GPT_distinct_integers_are_squares_l749_74984

theorem distinct_integers_are_squares
  (n : ℕ) 
  (h_n : n = 2000) 
  (x : Fin n → ℕ) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → x i ≠ x j)
  (h_product_square : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ∃ (m : ℕ), x i * x j * x k = m^2) :
  ∀ i : Fin n, ∃ (m : ℕ), x i = m^2 := 
sorry

end NUMINAMATH_GPT_distinct_integers_are_squares_l749_74984


namespace NUMINAMATH_GPT_profit_percentage_is_10_percent_l749_74959

theorem profit_percentage_is_10_percent
  (market_price_per_pen : ℕ)
  (retailer_buys_40_pens_for_36_price : 40 * market_price_per_pen = 36 * market_price_per_pen)
  (discount_percentage : ℕ)
  (selling_price_with_discount : ℕ) :
  discount_percentage = 1 →
  selling_price_with_discount = market_price_per_pen - (market_price_per_pen / 100) →
  (selling_price_with_discount * 40 - 36 * market_price_per_pen) / (36 * market_price_per_pen) * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_10_percent_l749_74959


namespace NUMINAMATH_GPT_area_of_triangle_example_l749_74904

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_example : 
  area_of_triangle (3, 3) (3, 10) (12, 19) = 31.5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_example_l749_74904


namespace NUMINAMATH_GPT_area_of_circumcircle_l749_74970

-- Define the problem:
theorem area_of_circumcircle 
  (a b c : ℝ) 
  (A B C : Real) 
  (h_cosC : Real.cos C = (2 * Real.sqrt 2) / 3) 
  (h_bcosA_acoB : b * Real.cos A + a * Real.cos B = 2)
  (h_sides : c = 2):
  let sinC := Real.sqrt (1 - (2 * Real.sqrt 2 / 3)^2)
  let R := c / (2 * sinC)
  let area := Real.pi * R^2
  area = 9 * Real.pi / 5 :=
by 
  sorry

end NUMINAMATH_GPT_area_of_circumcircle_l749_74970


namespace NUMINAMATH_GPT_Lily_points_l749_74971

variable (x y z : ℕ) -- points for inner ring (x), middle ring (y), and outer ring (z)

-- Tom's score
axiom Tom_score : 3 * x + y + 2 * z = 46

-- John's score
axiom John_score : x + 3 * y + 2 * z = 34

-- Lily's score
def Lily_score : ℕ := 40

theorem Lily_points : ∀ (x y z : ℕ), 3 * x + y + 2 * z = 46 → x + 3 * y + 2 * z = 34 → Lily_score = 40 := by
  intros x y z Tom_score John_score
  sorry

end NUMINAMATH_GPT_Lily_points_l749_74971


namespace NUMINAMATH_GPT_rodney_lift_l749_74939

theorem rodney_lift :
  ∃ (Ry : ℕ), 
  (∃ (Re R Ro : ℕ), 
  Re + Ry + R + Ro = 450 ∧
  Ry = 2 * R ∧
  R = Ro + 5 ∧
  Re = 3 * Ro - 20 ∧
  20 ≤ Ry ∧ Ry ≤ 200 ∧
  20 ≤ R ∧ R ≤ 200 ∧
  20 ≤ Ro ∧ Ro ≤ 200 ∧
  20 ≤ Re ∧ Re ≤ 200) ∧
  Ry = 140 :=
by
  sorry

end NUMINAMATH_GPT_rodney_lift_l749_74939


namespace NUMINAMATH_GPT_reciprocal_lcm_of_24_and_208_l749_74935

theorem reciprocal_lcm_of_24_and_208 :
  (1 / (Nat.lcm 24 208)) = (1 / 312) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_lcm_of_24_and_208_l749_74935


namespace NUMINAMATH_GPT_remainder_of_exp_l749_74902

theorem remainder_of_exp (x : ℝ) :
  (x + 1) ^ 2100 % (x^4 - x^2 + 1) = x^2 := 
sorry

end NUMINAMATH_GPT_remainder_of_exp_l749_74902


namespace NUMINAMATH_GPT_balance_of_diamondsuits_and_bullets_l749_74919

variable (a b c : ℕ)

theorem balance_of_diamondsuits_and_bullets 
  (h1 : 4 * a + 2 * b = 12 * c)
  (h2 : a = b + 3 * c) :
  3 * b = 6 * c := 
sorry

end NUMINAMATH_GPT_balance_of_diamondsuits_and_bullets_l749_74919


namespace NUMINAMATH_GPT_base9_39457_to_base10_is_26620_l749_74905

-- Define the components of the base 9 number 39457_9
def base9_39457 : ℕ := 39457
def base9_digits : List ℕ := [3, 9, 4, 5, 7]

-- Define the base
def base : ℕ := 9

-- Convert each position to its base 10 equivalent
def base9_to_base10 : ℕ :=
  3 * base ^ 4 + 9 * base ^ 3 + 4 * base ^ 2 + 5 * base ^ 1 + 7 * base ^ 0

-- State the theorem
theorem base9_39457_to_base10_is_26620 : base9_to_base10 = 26620 := by
  sorry

end NUMINAMATH_GPT_base9_39457_to_base10_is_26620_l749_74905


namespace NUMINAMATH_GPT_anne_carries_total_weight_l749_74989

-- Definitions for the conditions
def weight_female_cat : ℕ := 2
def weight_male_cat : ℕ := 2 * weight_female_cat

-- Problem statement
theorem anne_carries_total_weight : weight_female_cat + weight_male_cat = 6 :=
by
  sorry

end NUMINAMATH_GPT_anne_carries_total_weight_l749_74989


namespace NUMINAMATH_GPT_average_age_after_person_leaves_l749_74978

theorem average_age_after_person_leaves 
  (initial_people : ℕ) 
  (initial_average_age : ℕ) 
  (person_leaving_age : ℕ) 
  (remaining_people : ℕ) 
  (new_average_age : ℝ)
  (h1 : initial_people = 7) 
  (h2 : initial_average_age = 32) 
  (h3 : person_leaving_age = 22) 
  (h4 : remaining_people = 6) :
  new_average_age = 34 := 
by 
  sorry

end NUMINAMATH_GPT_average_age_after_person_leaves_l749_74978


namespace NUMINAMATH_GPT_least_common_multiple_of_wang_numbers_l749_74961

noncomputable def wang_numbers (n : ℕ) : List ℕ :=
  -- A function that returns the wang numbers in the set from 1 to n
  sorry

noncomputable def LCM (list : List ℕ) : ℕ :=
  -- A function that computes the least common multiple of a list of natural numbers
  sorry

theorem least_common_multiple_of_wang_numbers :
  LCM (wang_numbers 100) = 10080 :=
sorry

end NUMINAMATH_GPT_least_common_multiple_of_wang_numbers_l749_74961
