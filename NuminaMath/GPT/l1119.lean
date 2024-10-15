import Mathlib

namespace NUMINAMATH_GPT_price_per_cup_l1119_111998

theorem price_per_cup
  (num_trees : ℕ)
  (oranges_per_tree_g : ℕ)
  (oranges_per_tree_a : ℕ)
  (oranges_per_tree_m : ℕ)
  (oranges_per_cup : ℕ)
  (total_income : ℕ)
  (h_g : num_trees = 110)
  (h_a : oranges_per_tree_g = 600)
  (h_al : oranges_per_tree_a = 400)
  (h_m : oranges_per_tree_m = 500)
  (h_o : oranges_per_cup = 3)
  (h_income : total_income = 220000) :
  total_income / (((num_trees * oranges_per_tree_g) + (num_trees * oranges_per_tree_a) + (num_trees * oranges_per_tree_m)) / oranges_per_cup) = 4 :=
by
  repeat {sorry}

end NUMINAMATH_GPT_price_per_cup_l1119_111998


namespace NUMINAMATH_GPT_namjoon_used_pencils_l1119_111968

variable (taehyungUsed : ℕ) (namjoonUsed : ℕ)

/-- 
Statement:
Taehyung and Namjoon each initially have 10 pencils.
Taehyung gives 3 of his remaining pencils to Namjoon.
After this, Taehyung ends up with 6 pencils and Namjoon ends up with 6 pencils.
We need to prove that Namjoon used 7 pencils.
-/
theorem namjoon_used_pencils (H1 : 10 - taehyungUsed = 9 - 3)
  (H2 : 13 - namjoonUsed = 6) : namjoonUsed = 7 :=
sorry

end NUMINAMATH_GPT_namjoon_used_pencils_l1119_111968


namespace NUMINAMATH_GPT_complement_union_example_l1119_111921

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3, 4}

-- Define set B
def B : Set ℕ := {2, 4}

-- State the theorem we want to prove
theorem complement_union_example : (U \ A) ∪ B = {2, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_example_l1119_111921


namespace NUMINAMATH_GPT_person_A_arrives_before_B_l1119_111982

variable {a b S : ℝ}

theorem person_A_arrives_before_B (h : a ≠ b) (a_pos : 0 < a) (b_pos : 0 < b) (S_pos : 0 < S) :
  (2 * S / (a + b)) < ((a + b) * S / (2 * a * b)) :=
by
  sorry

end NUMINAMATH_GPT_person_A_arrives_before_B_l1119_111982


namespace NUMINAMATH_GPT_hexagon_side_squares_sum_l1119_111995

variables {P Q R P' Q' R' A B C D E F : Type}
variables (a1 a2 a3 b1 b2 b3 : ℝ)
variables (h_eq_triangles : congruent (triangle P Q R) (triangle P' Q' R'))
variables (h_sides : 
  AB = a1 ∧ BC = b1 ∧ CD = a2 ∧ 
  DE = b2 ∧ EF = a3 ∧ FA = b3)
  
theorem hexagon_side_squares_sum :
  a1^2 + a2^2 + a3^2 = b1^2 + b2^2 + b3^2 :=
sorry

end NUMINAMATH_GPT_hexagon_side_squares_sum_l1119_111995


namespace NUMINAMATH_GPT_ratio_of_areas_of_triangles_l1119_111930

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_triangles_l1119_111930


namespace NUMINAMATH_GPT_sum_of_first_15_terms_is_largest_l1119_111992

theorem sum_of_first_15_terms_is_largest
  (a : ℕ → ℝ)
  (s : ℕ → ℝ)
  (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : ∀ n, s n = n * a 1 + (n * (n - 1) * d) / 2)
  (h1: 13 * a 6 = 19 * (a 6 + 3 * d))
  (h2: a 1 > 0) : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≠ 15 → s 15 > s n :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_15_terms_is_largest_l1119_111992


namespace NUMINAMATH_GPT_supplement_complement_diff_l1119_111977

theorem supplement_complement_diff (α : ℝ) : (180 - α) - (90 - α) = 90 := 
by
  sorry

end NUMINAMATH_GPT_supplement_complement_diff_l1119_111977


namespace NUMINAMATH_GPT_twice_total_credits_l1119_111923

-- Define the variables and conditions
variables (Aria Emily Spencer Hannah : ℕ)
variables (h1 : Aria = 2 * Emily) 
variables (h2 : Emily = 2 * Spencer)
variables (h3 : Emily = 20)
variables (h4 : Hannah = 3 * Spencer)

-- Proof statement
theorem twice_total_credits : 2 * (Aria + Emily + Spencer + Hannah) = 200 :=
by 
  -- Proof steps are omitted with sorry
  sorry

end NUMINAMATH_GPT_twice_total_credits_l1119_111923


namespace NUMINAMATH_GPT_number_of_people_in_group_l1119_111933

theorem number_of_people_in_group (n : ℕ) (h1 : 110 - 60 = 5 * n) : n = 10 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_people_in_group_l1119_111933


namespace NUMINAMATH_GPT_determine_coefficients_l1119_111974

theorem determine_coefficients (p q : ℝ) :
  (∃ x : ℝ, x^2 + p * x + q = 0 ∧ x = p) ∧ (∃ y : ℝ, y^2 + p * y + q = 0 ∧ y = q)
  ↔ (p = 0 ∧ q = 0) ∨ (p = 1 ∧ q = -2) := by
sorry

end NUMINAMATH_GPT_determine_coefficients_l1119_111974


namespace NUMINAMATH_GPT_beckett_younger_than_olaf_l1119_111960

-- Define variables for ages
variables (O B S J : ℕ) (x : ℕ)

-- Express conditions as Lean hypotheses
def conditions :=
  B = O - x ∧  -- Beckett's age
  B = 12 ∧    -- Beckett is 12 years old
  S = O - 2 ∧ -- Shannen's age
  J = 2 * S + 5 ∧ -- Jack's age
  O + B + S + J = 71 -- Sum of ages
  
-- The theorem stating that Beckett is 8 years younger than Olaf
theorem beckett_younger_than_olaf (h : conditions O B S J x) : x = 8 :=
by
  -- The proof is omitted (using sorry)
  sorry

end NUMINAMATH_GPT_beckett_younger_than_olaf_l1119_111960


namespace NUMINAMATH_GPT_min_value_of_m_l1119_111935

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x + 5

theorem min_value_of_m {m : ℝ} (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ m) : m = 5 := 
sorry

end NUMINAMATH_GPT_min_value_of_m_l1119_111935


namespace NUMINAMATH_GPT_borrowing_period_l1119_111981

theorem borrowing_period 
  (principal : ℕ) (rate_1 : ℕ) (rate_2 : ℕ) (gain : ℕ)
  (h1 : principal = 5000)
  (h2 : rate_1 = 4)
  (h3 : rate_2 = 8)
  (h4 : gain = 200)
  : ∃ n : ℕ, n = 1 :=
by
  sorry

end NUMINAMATH_GPT_borrowing_period_l1119_111981


namespace NUMINAMATH_GPT_boarders_initial_count_l1119_111909

noncomputable def initial_boarders (x : ℕ) : ℕ := 7 * x

theorem boarders_initial_count (x : ℕ) (h1 : 80 + initial_boarders x = (2 : ℝ) * 16) :
  initial_boarders x = 560 :=
by
  sorry

end NUMINAMATH_GPT_boarders_initial_count_l1119_111909


namespace NUMINAMATH_GPT_problems_completed_l1119_111971

theorem problems_completed (p t : ℕ) (hp : p > 10) (eqn : p * t = (2 * p - 2) * (t - 1)) :
  p * t = 48 := 
sorry

end NUMINAMATH_GPT_problems_completed_l1119_111971


namespace NUMINAMATH_GPT_solve_siblings_age_problem_l1119_111938

def siblings_age_problem (x : ℕ) : Prop :=
  let age_eldest := 20
  let age_middle := 15
  let age_youngest := 10
  (age_eldest + x) + (age_middle + x) + (age_youngest + x) = 75 → x = 10

theorem solve_siblings_age_problem : siblings_age_problem 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_siblings_age_problem_l1119_111938


namespace NUMINAMATH_GPT_vectors_parallel_y_eq_minus_one_l1119_111973

theorem vectors_parallel_y_eq_minus_one (y : ℝ) :
  let a := (1, 2)
  let b := (1, -2 * y)
  b.1 * a.2 - a.1 * b.2 = 0 → y = -1 :=
by
  intros a b h
  simp at h
  sorry

end NUMINAMATH_GPT_vectors_parallel_y_eq_minus_one_l1119_111973


namespace NUMINAMATH_GPT_students_in_diligence_before_transfer_l1119_111987

theorem students_in_diligence_before_transfer (D I : ℕ) 
  (h1 : D + 2 = I - 2) 
  (h2 : D + I = 50) : 
  D = 23 := 
by
  sorry

end NUMINAMATH_GPT_students_in_diligence_before_transfer_l1119_111987


namespace NUMINAMATH_GPT_estevan_initial_blankets_l1119_111966

theorem estevan_initial_blankets (B : ℕ) 
  (polka_dot_initial : ℕ) 
  (polka_dot_total : ℕ) 
  (h1 : (1 / 3 : ℚ) * B = polka_dot_initial) 
  (h2 : polka_dot_initial + 2 = polka_dot_total) 
  (h3 : polka_dot_total = 10) : 
  B = 24 := 
by 
  sorry

end NUMINAMATH_GPT_estevan_initial_blankets_l1119_111966


namespace NUMINAMATH_GPT_calculate_exponent_product_l1119_111962

theorem calculate_exponent_product : (2^2021) * (-1/2)^2022 = (1/2) :=
by
  sorry

end NUMINAMATH_GPT_calculate_exponent_product_l1119_111962


namespace NUMINAMATH_GPT_sam_initial_watermelons_l1119_111970

theorem sam_initial_watermelons (x : ℕ) (h : x + 3 = 7) : x = 4 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_sam_initial_watermelons_l1119_111970


namespace NUMINAMATH_GPT_total_soccer_balls_donated_l1119_111919

def num_elementary_classes_per_school := 4
def num_middle_classes_per_school := 5
def num_schools := 2
def soccer_balls_per_class := 5

theorem total_soccer_balls_donated : 
  (num_elementary_classes_per_school + num_middle_classes_per_school) * num_schools * soccer_balls_per_class = 90 :=
by
  sorry

end NUMINAMATH_GPT_total_soccer_balls_donated_l1119_111919


namespace NUMINAMATH_GPT_ab_sum_l1119_111929

theorem ab_sum (a b : ℕ) (h1: (a + b) % 9 = 8) (h2: (a - b) % 11 = 7) : a + b = 8 :=
sorry

end NUMINAMATH_GPT_ab_sum_l1119_111929


namespace NUMINAMATH_GPT_cost_of_one_lesson_l1119_111924

-- Define the conditions
def total_cost_for_lessons : ℝ := 360
def total_hours_of_lessons : ℝ := 18
def duration_of_one_lesson : ℝ := 1.5

-- Define the theorem statement
theorem cost_of_one_lesson :
  (total_cost_for_lessons / total_hours_of_lessons) * duration_of_one_lesson = 30 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cost_of_one_lesson_l1119_111924


namespace NUMINAMATH_GPT_train_cross_time_proof_l1119_111964

noncomputable def train_cross_time_opposite (L : ℝ) (v1 v2 : ℝ) (t_same : ℝ) : ℝ :=
  let speed_same := (v1 - v2) * (5/18)
  let dist_same := speed_same * t_same
  let speed_opposite := (v1 + v2) * (5/18)
  dist_same / speed_opposite

theorem train_cross_time_proof : 
  train_cross_time_opposite 69.444 50 40 50 = 5.56 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_time_proof_l1119_111964


namespace NUMINAMATH_GPT_decreasing_interval_b_l1119_111965

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := - (1 / 2) * x ^ 2 + b * Real.log x

theorem decreasing_interval_b (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici (Real.sqrt 2) → ∀ x1 x2 : ℝ, x1 ∈ Set.Ici (Real.sqrt 2) → x2 ∈ Set.Ici (Real.sqrt 2) → 
   x1 ≤ x2 → f x1 b ≥ f x2 b) ↔ b ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_b_l1119_111965


namespace NUMINAMATH_GPT_prob_shooting_A_first_l1119_111996

-- Define the probabilities
def prob_A_hits : ℝ := 0.4
def prob_A_misses : ℝ := 0.6
def prob_B_hits : ℝ := 0.6
def prob_B_misses : ℝ := 0.4

-- Define the overall problem
theorem prob_shooting_A_first (k : ℕ) (ξ : ℕ) (hξ : ξ = k) :
  ((prob_A_misses * prob_B_misses)^(k-1)) * (1 - (prob_A_misses * prob_B_misses)) = 0.24^(k-1) * 0.76 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_prob_shooting_A_first_l1119_111996


namespace NUMINAMATH_GPT_max_angle_AFB_l1119_111927

noncomputable def focus_of_parabola := (2, 0)
def parabola (x y : ℝ) := y^2 = 8 * x
def on_parabola (A B : ℝ × ℝ) := parabola A.1 A.2 ∧ parabola B.1 B.2
def condition (x1 x2 : ℝ) (AB : ℝ) := x1 + x2 + 4 = (2 * Real.sqrt 3 / 3) * AB

theorem max_angle_AFB (A B : ℝ × ℝ) (x1 x2 : ℝ) (AB : ℝ)
  (h1 : on_parabola A B)
  (h2 : condition x1 x2 AB)
  (hA : A.1 = x1)
  (hB : B.1 = x2) :
  ∃ θ, θ ≤ Real.pi * 2 / 3 := 
  sorry

end NUMINAMATH_GPT_max_angle_AFB_l1119_111927


namespace NUMINAMATH_GPT_area_of_circle_with_radius_2_is_4pi_l1119_111975

theorem area_of_circle_with_radius_2_is_4pi :
  ∀ (π : ℝ), ∀ (r : ℝ), r = 2 → π > 0 → π * r^2 = 4 * π := 
by
  intros π r hr hπ
  sorry

end NUMINAMATH_GPT_area_of_circle_with_radius_2_is_4pi_l1119_111975


namespace NUMINAMATH_GPT_binomial_coeff_arithmetic_seq_l1119_111993

theorem binomial_coeff_arithmetic_seq (n : ℕ) (x : ℝ) (h : ∀ (a b c : ℝ), a = 1 ∧ b = n/2 ∧ c = n*(n-1)/8 → (b - a) = (c - b)) : n = 8 :=
sorry

end NUMINAMATH_GPT_binomial_coeff_arithmetic_seq_l1119_111993


namespace NUMINAMATH_GPT_LeahsCoinsValueIs68_l1119_111999

def LeahsCoinsWorthInCents (p n d : Nat) : Nat :=
  p * 1 + n * 5 + d * 10

theorem LeahsCoinsValueIs68 {p n d : Nat} (h1 : p + n + d = 17) (h2 : n + 2 = p) :
  LeahsCoinsWorthInCents p n d = 68 := by
  sorry

end NUMINAMATH_GPT_LeahsCoinsValueIs68_l1119_111999


namespace NUMINAMATH_GPT_find_exponent_M_l1119_111980

theorem find_exponent_M (M : ℕ) : (32^4) * (4^6) = 2^M → M = 32 := by
  sorry

end NUMINAMATH_GPT_find_exponent_M_l1119_111980


namespace NUMINAMATH_GPT_jack_christina_speed_l1119_111925

noncomputable def speed_of_jack_christina (d_jack_christina : ℝ) (v_lindy : ℝ) (d_lindy : ℝ) (relative_speed_factor : ℝ := 2) : ℝ :=
d_lindy * relative_speed_factor / d_jack_christina

theorem jack_christina_speed :
  speed_of_jack_christina 240 10 400 = 3 := by
  sorry

end NUMINAMATH_GPT_jack_christina_speed_l1119_111925


namespace NUMINAMATH_GPT_julia_download_songs_l1119_111948

-- Basic definitions based on conditions
def internet_speed_MBps : ℕ := 20
def song_size_MB : ℕ := 5
def half_hour_seconds : ℕ := 30 * 60

-- Statement of the proof problem
theorem julia_download_songs : 
  (internet_speed_MBps * half_hour_seconds) / song_size_MB = 7200 :=
by
  sorry

end NUMINAMATH_GPT_julia_download_songs_l1119_111948


namespace NUMINAMATH_GPT_steve_nickels_dimes_l1119_111949

theorem steve_nickels_dimes (n d : ℕ) (h1 : d = n + 4) (h2 : 5 * n + 10 * d = 70) : n = 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_steve_nickels_dimes_l1119_111949


namespace NUMINAMATH_GPT_inequality_proof_l1119_111912

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x + y + z ≥ 1/x + 1/y + 1/z) : 
  x/y + y/z + z/x ≥ 1/(x * y) + 1/(y * z) + 1/(z * x) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1119_111912


namespace NUMINAMATH_GPT_count_multiples_of_12_between_25_and_200_l1119_111945

theorem count_multiples_of_12_between_25_and_200 :
  ∃ n, (∀ i, 25 < i ∧ i < 200 → (∃ k, i = 12 * k)) ↔ n = 14 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_of_12_between_25_and_200_l1119_111945


namespace NUMINAMATH_GPT_fraction_of_earth_surface_habitable_for_humans_l1119_111956

theorem fraction_of_earth_surface_habitable_for_humans
  (total_land_fraction : ℚ) (habitable_land_fraction : ℚ)
  (h1 : total_land_fraction = 1/3)
  (h2 : habitable_land_fraction = 3/4) :
  (total_land_fraction * habitable_land_fraction) = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_earth_surface_habitable_for_humans_l1119_111956


namespace NUMINAMATH_GPT_value_of_expression_l1119_111953

-- Conditions
def isOdd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def isIncreasingOn (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def hasMaxOn (f : ℝ → ℝ) (a b : ℝ) (M : ℝ) := ∃ x, a ≤ x ∧ x ≤ b ∧ f x = M
def hasMinOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) := ∃ x, a ≤ x ∧ x ≤ b ∧ f x = m

-- Proof statement
theorem value_of_expression (f : ℝ → ℝ) 
  (hf1 : isOdd f)
  (hf2 : isIncreasingOn f 3 7)
  (hf3 : hasMaxOn f 3 6 8)
  (hf4 : hasMinOn f 3 6 (-1)) :
  2 * f (-6) + f (-3) = -15 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1119_111953


namespace NUMINAMATH_GPT_min_value_of_a_l1119_111908

theorem min_value_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → x^2 + 2 * a * x + 1 ≥ 0) → a ≥ -5 / 4 := 
sorry

end NUMINAMATH_GPT_min_value_of_a_l1119_111908


namespace NUMINAMATH_GPT_percent_of_value_and_divide_l1119_111991

theorem percent_of_value_and_divide (x : ℝ) (y : ℝ) (z : ℝ) (h : x = 1/300 * 180) (h1 : y = x / 6) : 
  y = 0.1 := 
by
  sorry

end NUMINAMATH_GPT_percent_of_value_and_divide_l1119_111991


namespace NUMINAMATH_GPT_finiteness_of_triples_l1119_111922

theorem finiteness_of_triples (x : ℚ) : ∃! (a b c : ℤ), a < 0 ∧ b^2 - 4*a*c = 5 ∧ (a*x^2 + b*x + c > 0) := sorry

end NUMINAMATH_GPT_finiteness_of_triples_l1119_111922


namespace NUMINAMATH_GPT_sequence_bound_l1119_111915

variable {a : ℕ+ → ℝ}

theorem sequence_bound (h : ∀ k m : ℕ+, |a (k + m) - a k - a m| ≤ 1) :
    ∀ (p q : ℕ+), |a p / p - a q / q| < 1 / p + 1 / q :=
by
  sorry

end NUMINAMATH_GPT_sequence_bound_l1119_111915


namespace NUMINAMATH_GPT_mikes_lower_rate_l1119_111955

theorem mikes_lower_rate (x : ℕ) (high_rate : ℕ) (total_paid : ℕ) (lower_payments : ℕ) (higher_payments : ℕ)
  (h1 : high_rate = 310)
  (h2 : total_paid = 3615)
  (h3 : lower_payments = 5)
  (h4 : higher_payments = 7)
  (h5 : lower_payments * x + higher_payments * high_rate = total_paid) :
  x = 289 :=
sorry

end NUMINAMATH_GPT_mikes_lower_rate_l1119_111955


namespace NUMINAMATH_GPT_tan_of_angle_123_l1119_111920

variable (a : ℝ)
variable (h : Real.sin 123 = a)

theorem tan_of_angle_123 : Real.tan 123 = a / Real.cos 123 :=
by
  sorry

end NUMINAMATH_GPT_tan_of_angle_123_l1119_111920


namespace NUMINAMATH_GPT_minimum_value_l1119_111913

open Real

variables {A B C M : Type}
variables (AB AC : ℝ) 
variables (S_MBC x y : ℝ)

-- Assume the given conditions
axiom dot_product_AB_AC : AB * AC = 2 * sqrt 3
axiom angle_BAC_30 : (30 : Real) = π / 6
axiom area_MBC : S_MBC = 1/2
axiom area_sum : x + y = 1/2

-- Define the minimum value problem
theorem minimum_value : 
  ∃ m, m = 18 ∧ (∀ x y, (1/x + 4/y) ≥ m) :=
sorry

end NUMINAMATH_GPT_minimum_value_l1119_111913


namespace NUMINAMATH_GPT_mr_blue_carrots_l1119_111916

theorem mr_blue_carrots :
  let steps_length := 3 -- length of each step in feet
  let garden_length_steps := 25 -- length of garden in steps
  let garden_width_steps := 35 -- width of garden in steps
  let length_feet := garden_length_steps * steps_length -- length of garden in feet
  let width_feet := garden_width_steps * steps_length -- width of garden in feet
  let area_feet2 := length_feet * width_feet -- area of garden in square feet
  let yield_rate := 3 / 4 -- yield rate of carrots in pounds per square foot
  let expected_yield := area_feet2 * yield_rate -- expected yield in pounds
  expected_yield = 5906.25
:= by
  sorry

end NUMINAMATH_GPT_mr_blue_carrots_l1119_111916


namespace NUMINAMATH_GPT_annual_depletion_rate_l1119_111937

theorem annual_depletion_rate
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (time : ℝ) 
  (depletion_rate : ℝ)
  (h_initial_value : initial_value = 40000)
  (h_final_value : final_value = 36100)
  (h_time : time = 2)
  (decay_eq : final_value = initial_value * (1 - depletion_rate)^time) :
  depletion_rate = 0.05 :=
by 
  sorry

end NUMINAMATH_GPT_annual_depletion_rate_l1119_111937


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1119_111984

theorem solution_set_of_inequality :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1119_111984


namespace NUMINAMATH_GPT_intersection_range_l1119_111900

-- Define the line equation
def line (k x : ℝ) : ℝ := k * x - k + 1

-- Define the curve equation
def curve (x y m : ℝ) : Prop := x^2 + 2 * y^2 = m

-- State the problem: Given the line and the curve have a common point, prove the range of m is m >= 3
theorem intersection_range (k m : ℝ) (h : ∃ x y, line k x = y ∧ curve x y m) : m ≥ 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_range_l1119_111900


namespace NUMINAMATH_GPT_white_marbles_count_l1119_111939

theorem white_marbles_count (total_marbles blue_marbles red_marbles : ℕ) (probability_red_or_white : ℚ)
    (h_total : total_marbles = 60)
    (h_blue : blue_marbles = 5)
    (h_red : red_marbles = 9)
    (h_probability : probability_red_or_white = 0.9166666666666666) :
    ∃ W : ℕ, W = total_marbles - blue_marbles - red_marbles ∧ probability_red_or_white = (red_marbles + W)/(total_marbles) ∧ W = 46 :=
by
  sorry

end NUMINAMATH_GPT_white_marbles_count_l1119_111939


namespace NUMINAMATH_GPT_find_angles_and_area_l1119_111963

noncomputable def angles_in_arithmetic_progression (A B C : ℝ) : Prop :=
  A + C = 2 * B ∧ A + B + C = 180

noncomputable def side_ratios (a b : ℝ) : Prop :=
  a / b = Real.sqrt 2 / Real.sqrt 3

noncomputable def triangle_area (a b c A B C : ℝ) : ℝ :=
  (1/2) * a * c * Real.sin B

theorem find_angles_and_area :
  ∃ (A B C a b c : ℝ), 
    angles_in_arithmetic_progression A B C ∧ 
    side_ratios a b ∧ 
    c = 2 ∧ 
    A = 45 ∧ 
    B = 60 ∧ 
    C = 75 ∧ 
    triangle_area a b c A B C = 3 - Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_angles_and_area_l1119_111963


namespace NUMINAMATH_GPT_number_of_cows_l1119_111926

/-- 
The number of cows Mr. Reyansh has on his dairy farm 
given the conditions of water consumption and total water used in a week. 
-/
theorem number_of_cows (C : ℕ) 
  (h1 : ∀ (c : ℕ), (c = 80 * 7))
  (h2 : ∀ (s : ℕ), (s = 10 * C))
  (h3 : ∀ (d : ℕ), (d = 20 * 7))
  (h4 : 1960 * C = 78400) : 
  C = 40 :=
sorry

end NUMINAMATH_GPT_number_of_cows_l1119_111926


namespace NUMINAMATH_GPT_percentage_gain_on_powerlifting_total_l1119_111957

def initialTotal : ℝ := 2200
def initialWeight : ℝ := 245
def weightIncrease : ℝ := 8
def finalWeight : ℝ := initialWeight + weightIncrease
def liftingRatio : ℝ := 10
def finalTotal : ℝ := finalWeight * liftingRatio

theorem percentage_gain_on_powerlifting_total :
  ∃ (P : ℝ), initialTotal * (1 + P / 100) = finalTotal :=
by
  sorry

end NUMINAMATH_GPT_percentage_gain_on_powerlifting_total_l1119_111957


namespace NUMINAMATH_GPT_number_of_handshakes_l1119_111946

-- Define the context of the problem
def total_women := 8
def teams (n : Nat) := 4

-- Define the number of people each woman will shake hands with (excluding her partner)
def handshakes_per_woman := total_women - 2

-- Define the total number of handshakes
def total_handshakes := (total_women * handshakes_per_woman) / 2

-- The theorem that we're to prove
theorem number_of_handshakes : total_handshakes = 24 :=
by
  sorry

end NUMINAMATH_GPT_number_of_handshakes_l1119_111946


namespace NUMINAMATH_GPT_find_x_l1119_111942

theorem find_x (x : ℝ) (h : 5020 - (x / 100.4) = 5015) : x = 502 :=
sorry

end NUMINAMATH_GPT_find_x_l1119_111942


namespace NUMINAMATH_GPT_transactions_Mabel_l1119_111951

variable {M A C J : ℝ}

theorem transactions_Mabel (h1 : A = 1.10 * M)
                          (h2 : C = 2 / 3 * A)
                          (h3 : J = C + 18)
                          (h4 : J = 84) :
  M = 90 :=
by
  sorry

end NUMINAMATH_GPT_transactions_Mabel_l1119_111951


namespace NUMINAMATH_GPT_parallel_tangent_line_l1119_111906

theorem parallel_tangent_line (b : ℝ) :
  (∃ b : ℝ, (∀ x y : ℝ, x + 2 * y + b = 0 → (x^2 + y^2 = 5))) →
  (b = 5 ∨ b = -5) :=
by
  sorry

end NUMINAMATH_GPT_parallel_tangent_line_l1119_111906


namespace NUMINAMATH_GPT_alice_number_l1119_111988

theorem alice_number (n : ℕ) 
  (h1 : 180 ∣ n) 
  (h2 : 75 ∣ n) 
  (h3 : 900 ≤ n) 
  (h4 : n ≤ 3000) : 
  n = 900 ∨ n = 1800 ∨ n = 2700 := 
by
  sorry

end NUMINAMATH_GPT_alice_number_l1119_111988


namespace NUMINAMATH_GPT_find_m_l1119_111901

-- Definition of the constraints and the values of x and y that satisfy them
def constraint1 (x y : ℝ) : Prop := x + y - 2 ≥ 0
def constraint2 (x y : ℝ) : Prop := x - y + 1 ≥ 0
def constraint3 (x : ℝ) : Prop := x ≤ 3

-- Given conditions
def satisfies_constraints (x y : ℝ) : Prop := 
  constraint1 x y ∧ constraint2 x y ∧ constraint3 x

-- The objective to prove
theorem find_m (x y m : ℝ) (h : satisfies_constraints x y) : 
  (∀ x y, satisfies_constraints x y → (- 3 = m * x + y)) → m = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1119_111901


namespace NUMINAMATH_GPT_product_of_solutions_eq_zero_l1119_111928

theorem product_of_solutions_eq_zero :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) → (x = 0 ∨ x = -4 / 7)) → (0 = 0) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_product_of_solutions_eq_zero_l1119_111928


namespace NUMINAMATH_GPT_basement_pump_time_l1119_111959

/-- A basement has a 30-foot by 36-foot rectangular floor, flooded to a depth of 24 inches.
Using three pumps, each pumping 10 gallons per minute, and knowing that a cubic foot of water
contains 7.5 gallons, this theorem asserts it will take 540 minutes to pump out all the water. -/
theorem basement_pump_time :
  let length := 30 -- in feet
  let width := 36 -- in feet
  let depth_inch := 24 -- in inches
  let depth := depth_inch / 12 -- converting depth to feet
  let volume_ft3 := length * width * depth -- volume in cubic feet
  let gallons_per_ft3 := 7.5 -- gallons per cubic foot
  let total_gallons := volume_ft3 * gallons_per_ft3 -- total volume in gallons
  let pump_capacity_gpm := 10 -- gallons per minute per pump
  let total_pumps := 3 -- number of pumps
  let total_pump_gpm := pump_capacity_gpm * total_pumps -- total gallons per minute for all pumps
  let pump_time := total_gallons / total_pump_gpm -- time in minutes to pump all the water
  pump_time = 540 := sorry

end NUMINAMATH_GPT_basement_pump_time_l1119_111959


namespace NUMINAMATH_GPT_difference_brothers_l1119_111990

def aaron_brothers : ℕ := 4
def bennett_brothers : ℕ := 6

theorem difference_brothers : 2 * aaron_brothers - bennett_brothers = 2 := by
  sorry

end NUMINAMATH_GPT_difference_brothers_l1119_111990


namespace NUMINAMATH_GPT_unique_solution_pair_l1119_111989

theorem unique_solution_pair (x y : ℝ) :
  (4 * x ^ 2 + 6 * x + 4) * (4 * y ^ 2 - 12 * y + 25) = 28 →
  (x, y) = (-3 / 4, 3 / 2) := by
  intro h
  sorry

end NUMINAMATH_GPT_unique_solution_pair_l1119_111989


namespace NUMINAMATH_GPT_largest_divisor_composite_difference_l1119_111934

-- Define when an integer is composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem largest_divisor_composite_difference (n : ℕ) (h : is_composite n) : 6 ∣ (n^4 - n) :=
by sorry

end NUMINAMATH_GPT_largest_divisor_composite_difference_l1119_111934


namespace NUMINAMATH_GPT_find_x_l1119_111985

theorem find_x (x : ℝ) (h₁ : x > 0) (h₂ : x^4 = 390625) : x = 25 := 
by sorry

end NUMINAMATH_GPT_find_x_l1119_111985


namespace NUMINAMATH_GPT_ab_difference_l1119_111902

theorem ab_difference (a b : ℝ) 
  (h1 : 10 = a * 3 + b)
  (h2 : 22 = a * 7 + b) : 
  a - b = 2 := 
  sorry

end NUMINAMATH_GPT_ab_difference_l1119_111902


namespace NUMINAMATH_GPT_problem_I_problem_II_problem_III_problem_IV_l1119_111958

/-- Problem I: Given: (2x - y)^2 = 1, Prove: y = 2x - 1 ∨ y = 2x + 1 --/
theorem problem_I (x y : ℝ) : (2 * x - y) ^ 2 = 1 → (y = 2 * x - 1) ∨ (y = 2 * x + 1) := 
sorry

/-- Problem II: Given: 16x^4 - 8x^2y^2 + y^4 - 8x^2 - 2y^2 + 1 = 0, Prove: y = 2x - 1 ∨ y = -2x - 1 ∨ y = 2x + 1 ∨ y = -2x + 1 --/
theorem problem_II (x y : ℝ) : 16 * x^4 - 8 * x^2 * y^2 + y^4 - 8 * x^2 - 2 * y^2 + 1 = 0 ↔ 
    (y = 2 * x - 1) ∨ (y = -2 * x - 1) ∨ (y = 2 * x + 1) ∨ (y = -2 * x + 1) := 
sorry

/-- Problem III: Given: x^2 * (1 - |y| / y) + y^2 + y * |y| = 8, Prove: (y = 2 ∧ y > 0) ∨ ((x = 2 ∨ x = -2) ∧ y < 0) --/
theorem problem_III (x y : ℝ) (hy : y ≠ 0) : x^2 * (1 - abs y / y) + y^2 + y * abs y = 8 →
    (y = 2 ∧ y > 0) ∨ ((x = 2 ∨ x = -2) ∧ y < 0) := 
sorry

/-- Problem IV: Given: x^2 + x * |x| + y^2 + (|x| * y^2 / x) = 8, Prove: x^2 + y^2 = 4 ∧ x > 0 --/
theorem problem_IV (x y : ℝ) (hx : x ≠ 0) : x^2 + x * abs x + y^2 + (abs x * y^2 / x) = 8 →
    (x^2 + y^2 = 4 ∧ x > 0) := 
sorry

end NUMINAMATH_GPT_problem_I_problem_II_problem_III_problem_IV_l1119_111958


namespace NUMINAMATH_GPT_sufficient_prime_logarithms_l1119_111918

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

-- Statement of the properties of logarithms
axiom log_mul (b x y : ℝ) : log_b b (x * y) = log_b b x + log_b b y
axiom log_div (b x y : ℝ) : log_b b (x / y) = log_b b x - log_b b y
axiom log_pow (b x : ℝ) (n : ℝ) : log_b b (x ^ n) = n * log_b b x

-- Main theorem
theorem sufficient_prime_logarithms (b : ℝ) (hb : 1 < b) :
  (∀ p : ℕ, is_prime p → ∃ Lp : ℝ, log_b b p = Lp) →
  ∀ n : ℕ, n > 0 → ∃ Ln : ℝ, log_b b n = Ln :=
by
  sorry

end NUMINAMATH_GPT_sufficient_prime_logarithms_l1119_111918


namespace NUMINAMATH_GPT_total_hours_until_joy_sees_grandma_l1119_111914

theorem total_hours_until_joy_sees_grandma
  (days_until_grandma: ℕ)
  (hours_in_a_day: ℕ)
  (timezone_difference: ℕ)
  (H_days : days_until_grandma = 2)
  (H_hours : hours_in_a_day = 24)
  (H_timezone : timezone_difference = 3) :
  (days_until_grandma * hours_in_a_day = 48) :=
by
  sorry

end NUMINAMATH_GPT_total_hours_until_joy_sees_grandma_l1119_111914


namespace NUMINAMATH_GPT_shirt_cost_is_43_l1119_111997

def pantsCost : ℕ := 140
def tieCost : ℕ := 15
def totalPaid : ℕ := 200
def changeReceived : ℕ := 2

def totalCostWithoutShirt := totalPaid - changeReceived
def totalCostWithPantsAndTie := pantsCost + tieCost
def shirtCost := totalCostWithoutShirt - totalCostWithPantsAndTie

theorem shirt_cost_is_43 : shirtCost = 43 := by
  have h1 : totalCostWithoutShirt = 198 := by rfl
  have h2 : totalCostWithPantsAndTie = 155 := by rfl
  have h3 : shirtCost = totalCostWithoutShirt - totalCostWithPantsAndTie := by rfl
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_shirt_cost_is_43_l1119_111997


namespace NUMINAMATH_GPT_evaluate_m_l1119_111944

theorem evaluate_m :
  ∀ m : ℝ, (243:ℝ)^(1/5) = 3^m → m = 1 :=
by
  intro m
  sorry

end NUMINAMATH_GPT_evaluate_m_l1119_111944


namespace NUMINAMATH_GPT_sin_75_deg_l1119_111961

theorem sin_75_deg : Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := 
by sorry

end NUMINAMATH_GPT_sin_75_deg_l1119_111961


namespace NUMINAMATH_GPT_bill_score_l1119_111947

theorem bill_score (B J S E : ℕ)
                   (h1 : B = J + 20)
                   (h2 : B = S / 2)
                   (h3 : E = B + J - 10)
                   (h4 : B + J + S + E = 250) :
                   B = 50 := 
by sorry

end NUMINAMATH_GPT_bill_score_l1119_111947


namespace NUMINAMATH_GPT_james_distance_l1119_111911

-- Definitions and conditions
def speed : ℝ := 80.0
def time : ℝ := 16.0

-- Proof problem statement
theorem james_distance : speed * time = 1280.0 := by
  sorry

end NUMINAMATH_GPT_james_distance_l1119_111911


namespace NUMINAMATH_GPT_swimmers_speed_in_still_water_l1119_111903

theorem swimmers_speed_in_still_water
  (v : ℝ) -- swimmer's speed in still water
  (current_speed : ℝ) -- speed of the water current
  (time : ℝ) -- time taken to swim against the current
  (distance : ℝ) -- distance swum against the current
  (h_current_speed : current_speed = 2)
  (h_time : time = 3.5)
  (h_distance : distance = 7)
  (h_eqn : time = distance / (v - current_speed)) :
  v = 4 :=
by
  sorry

end NUMINAMATH_GPT_swimmers_speed_in_still_water_l1119_111903


namespace NUMINAMATH_GPT_sum_prime_factors_1170_l1119_111983

theorem sum_prime_factors_1170 : 
  let smallest_prime_factor := 2
  let largest_prime_factor := 13
  (smallest_prime_factor + largest_prime_factor) = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_prime_factors_1170_l1119_111983


namespace NUMINAMATH_GPT_sum_series_eq_four_l1119_111967

noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, if n = 0 then 0 else (3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3))

theorem sum_series_eq_four :
  series_sum = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_eq_four_l1119_111967


namespace NUMINAMATH_GPT_find_sum_of_p_q_r_s_l1119_111986

theorem find_sum_of_p_q_r_s 
    (p q r s : ℝ)
    (h1 : r + s = 12 * p)
    (h2 : r * s = -13 * q)
    (h3 : p + q = 12 * r)
    (h4 : p * q = -13 * s)
    (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
    p + q + r + s = 2028 := 
sorry

end NUMINAMATH_GPT_find_sum_of_p_q_r_s_l1119_111986


namespace NUMINAMATH_GPT_sqrt_equality_l1119_111910

theorem sqrt_equality (m : ℝ) (n : ℝ) (h1 : 0 < m) (h2 : -3 * m ≤ n) (h3 : n ≤ 3 * m) :
    (Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2))
     - Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2))
    = 2 * Real.sqrt (3 * m - n)) :=
sorry

end NUMINAMATH_GPT_sqrt_equality_l1119_111910


namespace NUMINAMATH_GPT_initial_birds_in_tree_l1119_111904

theorem initial_birds_in_tree (x : ℕ) (h : x + 81 = 312) : x = 231 := 
by
  sorry

end NUMINAMATH_GPT_initial_birds_in_tree_l1119_111904


namespace NUMINAMATH_GPT_green_turtles_1066_l1119_111940

def number_of_turtles (G H : ℕ) : Prop :=
  H = 2 * G ∧ G + H = 3200

theorem green_turtles_1066 : ∃ G : ℕ, number_of_turtles G (2 * G) ∧ G = 1066 :=
by
  sorry

end NUMINAMATH_GPT_green_turtles_1066_l1119_111940


namespace NUMINAMATH_GPT_initial_oranges_is_sum_l1119_111976

-- Define the number of oranges taken by Jonathan
def oranges_taken : ℕ := 45

-- Define the number of oranges left in the box
def oranges_left : ℕ := 51

-- The theorem states that the initial number of oranges is the sum of the oranges taken and those left
theorem initial_oranges_is_sum : oranges_taken + oranges_left = 96 := 
by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_initial_oranges_is_sum_l1119_111976


namespace NUMINAMATH_GPT_dice_probability_l1119_111952

theorem dice_probability :
  let prob_one_digit := (9:ℚ) / 20
  let prob_two_digit := (11:ℚ) / 20
  let prob := 10 * (prob_two_digit^2) * (prob_one_digit^3)
  prob = 1062889 / 128000000 := 
by 
  sorry

end NUMINAMATH_GPT_dice_probability_l1119_111952


namespace NUMINAMATH_GPT_klinker_twice_as_old_l1119_111950

theorem klinker_twice_as_old :
  ∃ x : ℕ, (∀ (m k d : ℕ), m = 35 → d = 10 → m + x = 2 * (d + x)) → x = 15 :=
by
  sorry

end NUMINAMATH_GPT_klinker_twice_as_old_l1119_111950


namespace NUMINAMATH_GPT_students_called_back_l1119_111979

theorem students_called_back (girls boys not_called_back called_back : ℕ) 
  (h1 : girls = 17)
  (h2 : boys = 32)
  (h3 : not_called_back = 39)
  (h4 : called_back = (girls + boys) - not_called_back):
  called_back = 10 := by
  sorry

end NUMINAMATH_GPT_students_called_back_l1119_111979


namespace NUMINAMATH_GPT_envelopes_left_l1119_111954

theorem envelopes_left (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (number_of_friends : ℕ) (remaining_envelopes : ℕ) :
  initial_envelopes = 37 → envelopes_per_friend = 3 → number_of_friends = 5 → remaining_envelopes = initial_envelopes - (envelopes_per_friend * number_of_friends) → remaining_envelopes = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end NUMINAMATH_GPT_envelopes_left_l1119_111954


namespace NUMINAMATH_GPT_price_reduction_to_achieve_profit_l1119_111978

/-- 
A certain store sells clothing that cost $45$ yuan each to purchase for $65$ yuan each.
On average, they can sell $30$ pieces per day. For each $1$ yuan price reduction, 
an additional $5$ pieces can be sold per day. Given these conditions, 
prove that to achieve a daily profit of $800$ yuan, 
the price must be reduced by $10$ yuan per piece.
-/
theorem price_reduction_to_achieve_profit :
  ∃ x : ℝ, x = 10 ∧
    let original_cost := 45
    let original_price := 65
    let original_pieces_sold := 30
    let additional_pieces_per_yuan := 5
    let target_profit := 800
    let new_profit_per_piece := (original_price - original_cost) - x
    let new_pieces_sold := original_pieces_sold + additional_pieces_per_yuan * x
    new_profit_per_piece * new_pieces_sold = target_profit :=
by {
  sorry
}

end NUMINAMATH_GPT_price_reduction_to_achieve_profit_l1119_111978


namespace NUMINAMATH_GPT_find_m_l1119_111969

-- Define the vectors a and b and the condition for parallelicity
def a : ℝ × ℝ := (2, 1)
def b (m : ℝ) : ℝ × ℝ := (m, 2)
def parallel (u v : ℝ × ℝ) := u.1 * v.2 = u.2 * v.1

-- State the theorem with the given conditions and required proof goal
theorem find_m (m : ℝ) (h : parallel a (b m)) : m = 4 :=
by sorry  -- skipping proof

end NUMINAMATH_GPT_find_m_l1119_111969


namespace NUMINAMATH_GPT_amanda_needs_how_many_bags_of_grass_seeds_l1119_111941

theorem amanda_needs_how_many_bags_of_grass_seeds
    (lot_length : ℕ := 120)
    (lot_width : ℕ := 60)
    (concrete_length : ℕ := 40)
    (concrete_width : ℕ := 40)
    (bag_coverage : ℕ := 56) :
    (lot_length * lot_width - concrete_length * concrete_width) / bag_coverage = 100 := by
  sorry

end NUMINAMATH_GPT_amanda_needs_how_many_bags_of_grass_seeds_l1119_111941


namespace NUMINAMATH_GPT_find_second_liquid_parts_l1119_111931

-- Define the given constants
def first_liquid_kerosene_percentage : ℝ := 0.25
def second_liquid_kerosene_percentage : ℝ := 0.30
def first_liquid_parts : ℝ := 6
def mixture_kerosene_percentage : ℝ := 0.27

-- Define the amount of kerosene from each liquid
def kerosene_from_first_liquid := first_liquid_kerosene_percentage * first_liquid_parts
def kerosene_from_second_liquid (x : ℝ) := second_liquid_kerosene_percentage * x

-- Define the total parts of mixture
def total_mixture_parts (x : ℝ) := first_liquid_parts + x

-- Define the total kerosene in the mixture
def total_kerosene_in_mixture (x : ℝ) := mixture_kerosene_percentage * total_mixture_parts x

-- State the theorem
theorem find_second_liquid_parts (x : ℝ) :
  kerosene_from_first_liquid + kerosene_from_second_liquid x = total_kerosene_in_mixture x → 
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_second_liquid_parts_l1119_111931


namespace NUMINAMATH_GPT_at_least_one_not_beyond_20m_l1119_111907

variables (p q : Prop)

theorem at_least_one_not_beyond_20m : (¬ p ∨ ¬ q) ↔ ¬ (p ∧ q) :=
by sorry

end NUMINAMATH_GPT_at_least_one_not_beyond_20m_l1119_111907


namespace NUMINAMATH_GPT_length_of_pipe_is_correct_l1119_111917

-- Definitions of the conditions
def step_length : ℝ := 0.8
def steps_same_direction : ℤ := 210
def steps_opposite_direction : ℤ := 100

-- The distance moved by the tractor in one step
noncomputable def tractor_step_distance : ℝ := (steps_same_direction * step_length - steps_opposite_direction * step_length) / (steps_opposite_direction + steps_same_direction : ℝ)

-- The length of the pipe
noncomputable def length_of_pipe (steps_same_direction steps_opposite_direction : ℤ) (step_length : ℝ) : ℝ :=
 steps_same_direction * (step_length - tractor_step_distance)

-- Proof statement
theorem length_of_pipe_is_correct :
  length_of_pipe steps_same_direction steps_opposite_direction step_length = 108 :=
sorry

end NUMINAMATH_GPT_length_of_pipe_is_correct_l1119_111917


namespace NUMINAMATH_GPT_part_time_employees_l1119_111943

theorem part_time_employees (total_employees : ℕ) (full_time_employees : ℕ) (h1 : total_employees = 65134) (h2 : full_time_employees = 63093) :
  total_employees - full_time_employees = 2041 :=
by
  -- Suppose that total_employees - full_time_employees = 2041
  sorry

end NUMINAMATH_GPT_part_time_employees_l1119_111943


namespace NUMINAMATH_GPT_p_iff_q_l1119_111932

variables {a b c : ℝ}
def p (a b c : ℝ) : Prop := ∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0
def q (a b c : ℝ) : Prop := a + b + c = 0

theorem p_iff_q (h : a ≠ 0) : p a b c ↔ q a b c :=
sorry

end NUMINAMATH_GPT_p_iff_q_l1119_111932


namespace NUMINAMATH_GPT_parallelogram_area_l1119_111994

variable (d : ℕ) (h : ℕ)

theorem parallelogram_area (h_d : d = 30) (h_h : h = 20) : 
  ∃ a : ℕ, a = 600 := 
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1119_111994


namespace NUMINAMATH_GPT_shortest_distance_to_left_focus_l1119_111972

def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

def left_focus : ℝ × ℝ := (-5, 0)

theorem shortest_distance_to_left_focus : 
  ∃ P : ℝ × ℝ, 
  hyperbola P.1 P.2 ∧ 
  (∀ Q : ℝ × ℝ, hyperbola Q.1 Q.2 → dist Q left_focus ≥ dist P left_focus) ∧ 
  dist P left_focus = 2 :=
sorry

end NUMINAMATH_GPT_shortest_distance_to_left_focus_l1119_111972


namespace NUMINAMATH_GPT_number_of_children_bikes_l1119_111905

theorem number_of_children_bikes (c : ℕ) 
  (regular_bikes : ℕ) (wheels_per_regular_bike : ℕ) 
  (wheels_per_children_bike : ℕ) (total_wheels : ℕ)
  (h1 : regular_bikes = 7) 
  (h2 : wheels_per_regular_bike = 2) 
  (h3 : wheels_per_children_bike = 4) 
  (h4 : total_wheels = 58) 
  (h5 : total_wheels = (regular_bikes * wheels_per_regular_bike) + (c * wheels_per_children_bike)) 
  : c = 11 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_bikes_l1119_111905


namespace NUMINAMATH_GPT_probability_of_three_black_balls_l1119_111936

def total_ball_count : ℕ := 4 + 8

def white_ball_count : ℕ := 4

def black_ball_count : ℕ := 8

def total_combinations : ℕ := Nat.choose total_ball_count 3

def black_combinations : ℕ := Nat.choose black_ball_count 3

def probability_three_black : ℚ := black_combinations / total_combinations

theorem probability_of_three_black_balls : 
  probability_three_black = 14 / 55 := 
sorry

end NUMINAMATH_GPT_probability_of_three_black_balls_l1119_111936
