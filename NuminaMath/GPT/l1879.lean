import Mathlib

namespace NUMINAMATH_GPT_no_triples_exist_l1879_187954

theorem no_triples_exist (m p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hm : m > 0) :
  2^m * p^2 + 1 ≠ q^7 :=
sorry

end NUMINAMATH_GPT_no_triples_exist_l1879_187954


namespace NUMINAMATH_GPT_general_formulas_max_b_seq_l1879_187996

noncomputable def a_seq (n : ℕ) : ℕ := 4 * n - 2
noncomputable def b_seq (n : ℕ) : ℕ := 4 * n - 2 - 2^(n - 1)

-- The general formulas to be proved
theorem general_formulas :
  (∀ n : ℕ, a_seq n = 4 * n - 2) ∧ 
  (∀ n : ℕ, b_seq n = 4 * n - 2 - 2^(n - 1)) :=
by
  sorry

-- The maximum value condition to be proved
theorem max_b_seq :
  ((∀ n : ℕ, b_seq n ≤ b_seq 3) ∨ (∀ n : ℕ, b_seq n ≤ b_seq 4)) :=
by
  sorry

end NUMINAMATH_GPT_general_formulas_max_b_seq_l1879_187996


namespace NUMINAMATH_GPT_boat_stream_ratio_l1879_187923

theorem boat_stream_ratio (B S : ℝ) (h : 2 * (B - S) = B + S) : B / S = 3 :=
by
  sorry

end NUMINAMATH_GPT_boat_stream_ratio_l1879_187923


namespace NUMINAMATH_GPT_ratio_of_sides_l1879_187950

theorem ratio_of_sides (a b c d : ℝ) 
  (h1 : a / c = 4 / 5) 
  (h2 : b / d = 4 / 5) : b / d = 4 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_of_sides_l1879_187950


namespace NUMINAMATH_GPT_minimum_xy_l1879_187911

theorem minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : 
  x * y ≥ 18 :=
sorry

end NUMINAMATH_GPT_minimum_xy_l1879_187911


namespace NUMINAMATH_GPT_pascal_triangle_41_l1879_187946

theorem pascal_triangle_41:
  ∃ (n : Nat), ∀ (k : Nat), n = 41 ∧ (Nat.choose n k = 41) :=
sorry

end NUMINAMATH_GPT_pascal_triangle_41_l1879_187946


namespace NUMINAMATH_GPT_count_mod_6_mod_11_lt_1000_l1879_187955

theorem count_mod_6_mod_11_lt_1000 : ∃ n : ℕ, (∀ x : ℕ, (x < n + 1) ∧ ((6 + 11 * x) < 1000) ∧ (6 + 11 * x) % 11 = 6) ∧ (n + 1 = 91) :=
by
  sorry

end NUMINAMATH_GPT_count_mod_6_mod_11_lt_1000_l1879_187955


namespace NUMINAMATH_GPT_domain_of_f_monotonicity_of_f_l1879_187959

noncomputable def f (a x : ℝ) := Real.log (a ^ x - 1) / Real.log a

theorem domain_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (a > 1 → ∀ x : ℝ, f a x ∈ Set.Ioi 0) ∧ (0 < a ∧ a < 1 → ∀ x : ℝ, f a x ∈ Set.Iio 0) :=
sorry

theorem monotonicity_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (a > 1 → StrictMono (f a)) ∧ (0 < a ∧ a < 1 → StrictMono (f a)) :=
sorry

end NUMINAMATH_GPT_domain_of_f_monotonicity_of_f_l1879_187959


namespace NUMINAMATH_GPT_axis_of_symmetry_exists_l1879_187906

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem axis_of_symmetry_exists :
  ∃ k : ℤ, ∃ x : ℝ, (x = -5 * Real.pi / 12 ∧ f x = Real.sin (Real.pi / 2 + k * Real.pi))
  ∨ (x = Real.pi / 12 + k * Real.pi / 2 ∧ f x = Real.sin (Real.pi / 2 + k * Real.pi)) :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_exists_l1879_187906


namespace NUMINAMATH_GPT_min_total_number_of_stamps_l1879_187928

theorem min_total_number_of_stamps
  (r s t : ℕ)
  (h1 : 1 ≤ r)
  (h2 : 1 ≤ s)
  (h3 : 85 * r + 66 * s = 100 * t) :
  r + s = 7 := 
sorry

end NUMINAMATH_GPT_min_total_number_of_stamps_l1879_187928


namespace NUMINAMATH_GPT_black_grid_after_rotation_l1879_187958
open ProbabilityTheory

noncomputable def probability_black_grid_after_rotation : ℚ := 6561 / 65536

theorem black_grid_after_rotation (p : ℚ) (h : p = 1 / 2) :
  probability_black_grid_after_rotation = (3 / 4) ^ 8 := 
sorry

end NUMINAMATH_GPT_black_grid_after_rotation_l1879_187958


namespace NUMINAMATH_GPT_solve_keychain_problem_l1879_187962

def keychain_problem : Prop :=
  let f_class := 6
  let f_club := f_class / 2
  let thread_total := 108
  let total_friends := f_class + f_club
  let threads_per_keychain := thread_total / total_friends
  threads_per_keychain = 12

theorem solve_keychain_problem : keychain_problem :=
  by sorry

end NUMINAMATH_GPT_solve_keychain_problem_l1879_187962


namespace NUMINAMATH_GPT_mechanism_parts_l1879_187998

-- Definitions
def total_parts (S L : Nat) : Prop := S + L = 25
def condition1 (S L : Nat) : Prop := ∀ (A : Finset (Fin 25)), (A.card = 12) → ∃ i, i ∈ A ∧ i < S
def condition2 (S L : Nat) : Prop := ∀ (B : Finset (Fin 25)), (B.card = 15) → ∃ i, i ∈ B ∧ i >= S

-- Main statement
theorem mechanism_parts :
  ∃ (S L : Nat), 
  total_parts S L ∧ 
  condition1 S L ∧ 
  condition2 S L ∧ 
  S = 14 ∧ 
  L = 11 :=
sorry

end NUMINAMATH_GPT_mechanism_parts_l1879_187998


namespace NUMINAMATH_GPT_lock_combination_correct_l1879_187989

noncomputable def lock_combination : ℤ := 812

theorem lock_combination_correct :
  ∀ (S T A R : ℕ), S ≠ T → S ≠ A → S ≠ R → T ≠ A → T ≠ R → A ≠ R →
  ((S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + S) + 
   (T * 9^4 + A * 9^3 + R * 9^2 + T * 9 + S) + 
   (S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + T)) % 9^5 = 
  S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + S →
  (S * 9^2 + T * 9^1 + A) = lock_combination := 
by
  intros S T A R hST hSA hSR hTA hTR hAR h_eq
  sorry

end NUMINAMATH_GPT_lock_combination_correct_l1879_187989


namespace NUMINAMATH_GPT_ten_faucets_fill_50_gallon_in_60_seconds_l1879_187930

-- Define the conditions
def five_faucets_fill_tub (faucet_rate : ℝ) : Prop :=
  5 * faucet_rate * 8 = 200

def all_faucets_same_rate (tub_capacity time : ℝ) (num_faucets : ℕ) (faucet_rate : ℝ) : Prop :=
  num_faucets * faucet_rate * time = tub_capacity

-- Define the main theorem to be proven
theorem ten_faucets_fill_50_gallon_in_60_seconds (faucet_rate : ℝ) :
  (∃ faucet_rate, five_faucets_fill_tub faucet_rate) →
  all_faucets_same_rate 50 1 10 faucet_rate →
  10 * faucet_rate * (1 / 60) = 50 :=
by
  sorry

end NUMINAMATH_GPT_ten_faucets_fill_50_gallon_in_60_seconds_l1879_187930


namespace NUMINAMATH_GPT_complementary_angle_difference_l1879_187961

def is_complementary (a b : ℝ) : Prop := a + b = 90

def in_ratio (a b : ℝ) (m n : ℝ) : Prop := a / b = m / n

theorem complementary_angle_difference (a b : ℝ) (h1 : is_complementary a b) (h2 : in_ratio a b 5 1) : abs (a - b) = 60 := 
by
  sorry

end NUMINAMATH_GPT_complementary_angle_difference_l1879_187961


namespace NUMINAMATH_GPT_subset_M_N_l1879_187991

-- Definitions of M and N as per the problem statement
def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | 1 / x < 2}

-- Lean statement for the proof problem: M ⊆ N
theorem subset_M_N : M ⊆ N := by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_subset_M_N_l1879_187991


namespace NUMINAMATH_GPT_number_of_boxes_l1879_187973

-- Define the conditions
def apples_per_crate : ℕ := 180
def number_of_crates : ℕ := 12
def rotten_apples : ℕ := 160
def apples_per_box : ℕ := 20

-- Define the statement to prove
theorem number_of_boxes : (apples_per_crate * number_of_crates - rotten_apples) / apples_per_box = 100 := 
by 
  sorry -- Proof skipped

end NUMINAMATH_GPT_number_of_boxes_l1879_187973


namespace NUMINAMATH_GPT_distance_between_A_and_B_l1879_187944

theorem distance_between_A_and_B 
  (v_pas0 v_freight0 : ℝ) -- original speeds of passenger and freight train
  (t_freight : ℝ) -- time taken by freight train
  (d : ℝ) -- distance sought
  (h1 : t_freight = d / v_freight0) 
  (h2 : d + 288 = v_pas0 * t_freight) 
  (h3 : (d / (v_freight0 + 10)) + 2.4 = d / (v_pas0 + 10))
  : d = 360 := 
sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l1879_187944


namespace NUMINAMATH_GPT_roots_not_integers_l1879_187915

theorem roots_not_integers (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
    ¬ ∃ x₁ x₂ : ℤ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_roots_not_integers_l1879_187915


namespace NUMINAMATH_GPT_negation_exists_geq_l1879_187975

theorem negation_exists_geq :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_geq_l1879_187975


namespace NUMINAMATH_GPT_inclination_angle_of_vertical_line_l1879_187963

theorem inclination_angle_of_vertical_line :
  ∀ x : ℝ, x = Real.tan (60 * Real.pi / 180) → ∃ θ : ℝ, θ = 90 := by
  sorry

end NUMINAMATH_GPT_inclination_angle_of_vertical_line_l1879_187963


namespace NUMINAMATH_GPT_movie_time_difference_l1879_187951

theorem movie_time_difference
  (Nikki_movie : ℝ)
  (Michael_movie : ℝ)
  (Ryn_movie : ℝ)
  (Joyce_movie : ℝ)
  (total_hours : ℝ)
  (h1 : Nikki_movie = 30)
  (h2 : Michael_movie = Nikki_movie / 3)
  (h3 : Ryn_movie = (4 / 5) * Nikki_movie)
  (h4 : total_hours = 76)
  (h5 : total_hours = Michael_movie + Nikki_movie + Ryn_movie + Joyce_movie) :
  Joyce_movie - Michael_movie = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_movie_time_difference_l1879_187951


namespace NUMINAMATH_GPT_balboa_earnings_correct_l1879_187922

def students_from_allen_days : Nat := 7 * 3
def students_from_balboa_days : Nat := 4 * 5
def students_from_carver_days : Nat := 5 * 9
def total_student_days : Nat := students_from_allen_days + students_from_balboa_days + students_from_carver_days
def total_payment : Nat := 744
def daily_wage : Nat := total_payment / total_student_days
def balboa_earnings : Nat := daily_wage * students_from_balboa_days

theorem balboa_earnings_correct : balboa_earnings = 180 := by
  sorry

end NUMINAMATH_GPT_balboa_earnings_correct_l1879_187922


namespace NUMINAMATH_GPT_value_2_std_devs_less_than_mean_l1879_187978

-- Define the arithmetic mean
def mean : ℝ := 15.5

-- Define the standard deviation
def standard_deviation : ℝ := 1.5

-- Define the value that is 2 standard deviations less than the mean
def value_2_std_less_than_mean : ℝ := mean - 2 * standard_deviation

-- The theorem we want to prove
theorem value_2_std_devs_less_than_mean : value_2_std_less_than_mean = 12.5 := by
  sorry

end NUMINAMATH_GPT_value_2_std_devs_less_than_mean_l1879_187978


namespace NUMINAMATH_GPT_union_sets_example_l1879_187982

theorem union_sets_example : ({0, 1} ∪ {2} : Set ℕ) = {0, 1, 2} := by 
  sorry

end NUMINAMATH_GPT_union_sets_example_l1879_187982


namespace NUMINAMATH_GPT_train_cross_signal_pole_time_l1879_187957

theorem train_cross_signal_pole_time :
  ∀ (train_length platform_length platform_cross_time signal_cross_time : ℝ),
  train_length = 300 →
  platform_length = 300 →
  platform_cross_time = 36 →
  signal_cross_time = train_length / ((train_length + platform_length) / platform_cross_time) →
  signal_cross_time = 18 :=
by
  intros train_length platform_length platform_cross_time signal_cross_time h_train_length h_platform_length h_platform_cross_time h_signal_cross_time
  rw [h_train_length, h_platform_length, h_platform_cross_time] at h_signal_cross_time
  sorry

end NUMINAMATH_GPT_train_cross_signal_pole_time_l1879_187957


namespace NUMINAMATH_GPT_distance_between_parallel_sides_l1879_187960

theorem distance_between_parallel_sides (a b : ℝ) (h : ℝ) (A : ℝ) :
  a = 20 → b = 10 → A = 150 → (A = 1 / 2 * (a + b) * h) → h = 10 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_distance_between_parallel_sides_l1879_187960


namespace NUMINAMATH_GPT_pond_diameter_l1879_187984

theorem pond_diameter 
  (h k r : ℝ)
  (H1 : (4 - h) ^ 2 + (11 - k) ^ 2 = r ^ 2)
  (H2 : (12 - h) ^ 2 + (9 - k) ^ 2 = r ^ 2)
  (H3 : (2 - h) ^ 2 + (7 - k) ^ 2 = (r - 1) ^ 2) :
  2 * r = 9.2 :=
sorry

end NUMINAMATH_GPT_pond_diameter_l1879_187984


namespace NUMINAMATH_GPT_triangle_prime_sides_l1879_187948

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem triangle_prime_sides :
  ∃ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ is_prime a ∧ is_prime b ∧ is_prime c ∧ 
  a + b + c = 25 ∧
  (a = b ∨ b = c ∨ a = c) ∧
  (∀ (x y z : ℕ), x ≤ y ∧ y ≤ z ∧ is_prime x ∧ is_prime y ∧ is_prime z ∧ x + y + z = 25 → (x, y, z) = (3, 11, 11) ∨ (x, y, z) = (7, 7, 11)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_prime_sides_l1879_187948


namespace NUMINAMATH_GPT_problem_statement_l1879_187905

theorem problem_statement (x : ℕ) (h : 4 * (3^x) = 2187) : (x + 2) * (x - 2) = 21 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1879_187905


namespace NUMINAMATH_GPT_part_a_part_b_l1879_187939

def bright (n : ℕ) := ∃ a b : ℕ, n = a^2 + b^3

theorem part_a (r s : ℕ) (h₀ : r > 0) (h₁ : s > 0) : 
  ∃ᶠ n in at_top, bright (r + n) ∧ bright (s + n) := 
by sorry

theorem part_b (r s : ℕ) (h₀ : r > 0) (h₁ : s > 0) : 
  ∃ᶠ m in at_top, bright (r * m) ∧ bright (s * m) := 
by sorry

end NUMINAMATH_GPT_part_a_part_b_l1879_187939


namespace NUMINAMATH_GPT_cutoff_score_admission_l1879_187985

theorem cutoff_score_admission (x : ℝ) 
  (h1 : (2 / 5) * (x + 15) + (3 / 5) * (x - 20) = 90) : x = 96 :=
sorry

end NUMINAMATH_GPT_cutoff_score_admission_l1879_187985


namespace NUMINAMATH_GPT_total_volume_of_all_cubes_l1879_187936

/-- Carl has 4 cubes each with a side length of 3 -/
def carl_cubes_side_length := 3
def carl_cubes_count := 4

/-- Kate has 6 cubes each with a side length of 4 -/
def kate_cubes_side_length := 4
def kate_cubes_count := 6

/-- Total volume of 10 cubes with given conditions -/
theorem total_volume_of_all_cubes : 
  carl_cubes_count * (carl_cubes_side_length ^ 3) + 
  kate_cubes_count * (kate_cubes_side_length ^ 3) = 492 := by
  sorry

end NUMINAMATH_GPT_total_volume_of_all_cubes_l1879_187936


namespace NUMINAMATH_GPT_angle_between_vectors_is_45_degrees_l1879_187942

-- Define the vectors
def u : ℝ × ℝ := (4, -1)
def v : ℝ × ℝ := (5, 3)

-- Define the theorem to prove the angle between these vectors is 45 degrees
theorem angle_between_vectors_is_45_degrees : 
  let dot_product := (4 * 5) + (-1 * 3)
  let norm_u := Real.sqrt ((4^2) + (-1)^2)
  let norm_v := Real.sqrt ((5^2) + (3^2))
  let cos_theta := dot_product / (norm_u * norm_v)
  let theta := Real.arccos cos_theta
  45 = (theta * 180 / Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_angle_between_vectors_is_45_degrees_l1879_187942


namespace NUMINAMATH_GPT_find_nth_number_in_s_l1879_187900

def s (k : ℕ) : ℕ := 8 * k + 5

theorem find_nth_number_in_s (n : ℕ) (number_in_s : ℕ) (h : number_in_s = 573) :
  ∃ k : ℕ, s k = number_in_s ∧ n = k + 1 := 
sorry

end NUMINAMATH_GPT_find_nth_number_in_s_l1879_187900


namespace NUMINAMATH_GPT_algebraic_expression_value_l1879_187926

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1879_187926


namespace NUMINAMATH_GPT_AM_GM_contradiction_l1879_187901

open Real

theorem AM_GM_contradiction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
      ¬ (6 < a + 4 / b ∧ 6 < b + 9 / c ∧ 6 < c + 16 / a) := by
  sorry

end NUMINAMATH_GPT_AM_GM_contradiction_l1879_187901


namespace NUMINAMATH_GPT_min_value_of_t_l1879_187933

theorem min_value_of_t (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : 
  ∃ t : ℝ, t = 3 + 2 * Real.sqrt 2 ∧ t = 1 / a + 1 / b :=
sorry

end NUMINAMATH_GPT_min_value_of_t_l1879_187933


namespace NUMINAMATH_GPT_difference_between_numbers_l1879_187972

theorem difference_between_numbers (x y : ℕ) 
  (h1 : x + y = 20000) 
  (h2 : y = 7 * x) : y - x = 15000 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l1879_187972


namespace NUMINAMATH_GPT_complete_square_transform_l1879_187979

theorem complete_square_transform (x : ℝ) (h : x^2 + 8*x + 7 = 0) : (x + 4)^2 = 9 :=
by sorry

end NUMINAMATH_GPT_complete_square_transform_l1879_187979


namespace NUMINAMATH_GPT_retailer_overhead_expenses_l1879_187990

theorem retailer_overhead_expenses (purchase_price selling_price profit_percent : ℝ) (overhead_expenses : ℝ) 
  (h1 : purchase_price = 225) 
  (h2 : selling_price = 300) 
  (h3 : profit_percent = 25) 
  (h4 : selling_price = (purchase_price + overhead_expenses) * (1 + profit_percent / 100)) : 
  overhead_expenses = 15 := 
by
  sorry

end NUMINAMATH_GPT_retailer_overhead_expenses_l1879_187990


namespace NUMINAMATH_GPT_sin_double_angle_15_eq_half_l1879_187966

theorem sin_double_angle_15_eq_half : 2 * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_15_eq_half_l1879_187966


namespace NUMINAMATH_GPT_value_of_expression_l1879_187995

theorem value_of_expression (a : ℝ) (h : 10 * a^2 + 3 * a + 2 = 5) : 
  3 * a + 2 = (31 + 3 * Real.sqrt 129) / 20 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l1879_187995


namespace NUMINAMATH_GPT_problem1_problem2_l1879_187940

-- Proof problem for the first condition
theorem problem1 {p : ℕ} (hp : Nat.Prime p) 
  (h : ∃ n : ℕ, (7^(p-1) - 1) = p * n^2) : p = 3 :=
sorry

-- Proof problem for the second condition
theorem problem2 {p : ℕ} (hp : Nat.Prime p)
  (h : ∃ n : ℕ, (11^(p-1) - 1) = p * n^2) : false :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1879_187940


namespace NUMINAMATH_GPT_possible_age_of_youngest_child_l1879_187943

noncomputable def valid_youngest_age (father_fee : ℝ) (child_fee_per_year : ℝ) (total_bill : ℝ) (triplet_age : ℝ) : ℝ :=
  total_bill - father_fee -  (3 * triplet_age * child_fee_per_year)

theorem possible_age_of_youngest_child (father_fee : ℝ) (child_fee_per_year : ℝ) (total_bill : ℝ) (t y : ℝ)
  (h1 : father_fee = 16)
  (h2 : child_fee_per_year = 0.8)
  (h3 : total_bill = 43.2)
  (age_condition : y = (total_bill - father_fee) / child_fee_per_year - 3 * t) :
  y = 1 ∨ y = 4 :=
by
  sorry

end NUMINAMATH_GPT_possible_age_of_youngest_child_l1879_187943


namespace NUMINAMATH_GPT_num_decompositions_144_l1879_187983

theorem num_decompositions_144 : ∃ D, D = 45 ∧ 
  (∀ (factors : List ℕ), 
    (∀ x, x ∈ factors → x > 1) ∧ factors.prod = 144 → 
    factors.permutations.length = D) :=
sorry

end NUMINAMATH_GPT_num_decompositions_144_l1879_187983


namespace NUMINAMATH_GPT_emily_total_beads_l1879_187953

theorem emily_total_beads (necklaces : ℕ) (beads_per_necklace : ℕ) (total_beads : ℕ) : 
  necklaces = 11 → 
  beads_per_necklace = 28 → 
  total_beads = necklaces * beads_per_necklace → 
  total_beads = 308 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end NUMINAMATH_GPT_emily_total_beads_l1879_187953


namespace NUMINAMATH_GPT_second_consecutive_odd_integer_l1879_187924

theorem second_consecutive_odd_integer (n : ℤ) : 
  (n - 2) + (n + 2) = 152 → n = 76 := 
by 
  sorry

end NUMINAMATH_GPT_second_consecutive_odd_integer_l1879_187924


namespace NUMINAMATH_GPT_field_size_l1879_187999

theorem field_size
  (cost_per_foot : ℝ)
  (total_money : ℝ)
  (cannot_fence : ℝ)
  (cost_per_foot_eq : cost_per_foot = 30)
  (total_money_eq : total_money = 120000)
  (cannot_fence_eq : cannot_fence > 1000) :
  ∃ (side_length : ℝ), side_length * side_length = 1000000 := 
by
  sorry

end NUMINAMATH_GPT_field_size_l1879_187999


namespace NUMINAMATH_GPT_jackson_pbj_sandwiches_l1879_187945

-- The number of Wednesdays and Fridays in the 36-week school year
def total_weeks : ℕ := 36
def total_wednesdays : ℕ := total_weeks
def total_fridays : ℕ := total_weeks

-- Public holidays on Wednesdays and Fridays
def holidays_wednesdays : ℕ := 2
def holidays_fridays : ℕ := 3

-- Days Jackson missed
def missed_wednesdays : ℕ := 1
def missed_fridays : ℕ := 2

-- Number of times Jackson asks for a ham and cheese sandwich every 4 weeks
def weeks_for_ham_and_cheese : ℕ := total_weeks / 4

-- Number of ham and cheese sandwich days
def ham_and_cheese_wednesdays : ℕ := weeks_for_ham_and_cheese
def ham_and_cheese_fridays : ℕ := weeks_for_ham_and_cheese * 2

-- Remaining days for peanut butter and jelly sandwiches
def remaining_wednesdays : ℕ := total_wednesdays - holidays_wednesdays - missed_wednesdays
def remaining_fridays : ℕ := total_fridays - holidays_fridays - missed_fridays

def pbj_wednesdays : ℕ := remaining_wednesdays - ham_and_cheese_wednesdays
def pbj_fridays : ℕ := remaining_fridays - ham_and_cheese_fridays

-- Total peanut butter and jelly sandwiches
def total_pbj : ℕ := pbj_wednesdays + pbj_fridays

theorem jackson_pbj_sandwiches : total_pbj = 37 := by
  -- We don't require the proof steps, just the statement
  sorry

end NUMINAMATH_GPT_jackson_pbj_sandwiches_l1879_187945


namespace NUMINAMATH_GPT_problem_solution_l1879_187981

theorem problem_solution (x : ℕ) (h : x = 3) : x + x * x^(x^2) = 59052 :=
by
  rw [h]
  -- The condition is now x = 3
  let t := 3 + 3 * 3^(3^2)
  have : t = 59052 := sorry
  exact this

end NUMINAMATH_GPT_problem_solution_l1879_187981


namespace NUMINAMATH_GPT_m_squared_plus_reciprocal_squared_l1879_187932

theorem m_squared_plus_reciprocal_squared (m : ℝ) (h : m^2 - 2 * m - 1 = 0) : m^2 + 1 / m^2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_m_squared_plus_reciprocal_squared_l1879_187932


namespace NUMINAMATH_GPT_raisin_fraction_of_mixture_l1879_187904

noncomputable def raisin_nut_cost_fraction (R : ℝ) : ℝ :=
  let raisin_cost := 3 * R
  let nut_cost := 4 * (4 * R)
  let total_cost := raisin_cost + nut_cost
  raisin_cost / total_cost

theorem raisin_fraction_of_mixture (R : ℝ) : raisin_nut_cost_fraction R = 3 / 19 :=
by
  sorry

end NUMINAMATH_GPT_raisin_fraction_of_mixture_l1879_187904


namespace NUMINAMATH_GPT_boxes_calculation_l1879_187956

theorem boxes_calculation (total_bottles : ℕ) (bottles_per_bag : ℕ) (bags_per_box : ℕ) (boxes : ℕ) :
  total_bottles = 8640 → bottles_per_bag = 12 → bags_per_box = 6 → boxes = total_bottles / (bottles_per_bag * bags_per_box) → boxes = 120 :=
by
  intros h_total h_bottles_per_bag h_bags_per_box h_boxes
  rw [h_total, h_bottles_per_bag, h_bags_per_box] at h_boxes
  norm_num at h_boxes
  exact h_boxes

end NUMINAMATH_GPT_boxes_calculation_l1879_187956


namespace NUMINAMATH_GPT_mark_garden_total_flowers_l1879_187914

theorem mark_garden_total_flowers :
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  total_yellow_purple + green = 35 :=
by
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  simp [yellow, purple, total_yellow_purple, green]
  sorry

end NUMINAMATH_GPT_mark_garden_total_flowers_l1879_187914


namespace NUMINAMATH_GPT_shaded_area_is_correct_l1879_187971

noncomputable def octagon_side_length := 3
noncomputable def octagon_area := 2 * (1 + Real.sqrt 2) * octagon_side_length^2
noncomputable def semicircle_radius := octagon_side_length / 2
noncomputable def semicircle_area := (1 / 2) * Real.pi * semicircle_radius^2
noncomputable def total_semicircle_area := 8 * semicircle_area
noncomputable def shaded_region_area := octagon_area - total_semicircle_area

theorem shaded_area_is_correct : shaded_region_area = 54 + 36 * Real.sqrt 2 - 9 * Real.pi :=
by
  -- Proof goes here, but we're inserting sorry to skip it
  sorry

end NUMINAMATH_GPT_shaded_area_is_correct_l1879_187971


namespace NUMINAMATH_GPT_apples_in_second_group_l1879_187913

theorem apples_in_second_group : 
  ∀ (A O : ℝ) (x : ℕ), 
  6 * A + 3 * O = 1.77 ∧ x * A + 5 * O = 1.27 ∧ A = 0.21 → 
  x = 2 :=
by
  intros A O x h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_apples_in_second_group_l1879_187913


namespace NUMINAMATH_GPT_initial_candies_count_l1879_187916

-- Definitions based on conditions
def NelliesCandies : Nat := 12
def JacobsCandies : Nat := NelliesCandies / 2
def LanasCandies : Nat := JacobsCandies - 3
def TotalCandiesEaten : Nat := NelliesCandies + JacobsCandies + LanasCandies
def RemainingCandies : Nat := 3 * 3
def InitialCandies := TotalCandiesEaten + RemainingCandies

-- Theorem stating the initial candies count
theorem initial_candies_count : InitialCandies = 30 := by 
  sorry

end NUMINAMATH_GPT_initial_candies_count_l1879_187916


namespace NUMINAMATH_GPT_P_inter_M_l1879_187968

def set_P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def set_M : Set ℝ := {x | x^2 ≤ 9}

theorem P_inter_M :
  set_P ∩ set_M = {x | 0 ≤ x ∧ x < 3} := sorry

end NUMINAMATH_GPT_P_inter_M_l1879_187968


namespace NUMINAMATH_GPT_wholesale_cost_calc_l1879_187912

theorem wholesale_cost_calc (wholesale_cost : ℝ) 
  (h_profit : 0.15 * wholesale_cost = 28 - wholesale_cost) : 
  wholesale_cost = 28 / 1.15 :=
by
  sorry

end NUMINAMATH_GPT_wholesale_cost_calc_l1879_187912


namespace NUMINAMATH_GPT_option_c_opp_numbers_l1879_187997

theorem option_c_opp_numbers : (- (2 ^ 2)) = - ((-2) ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_option_c_opp_numbers_l1879_187997


namespace NUMINAMATH_GPT_highest_price_without_lowering_revenue_minimum_annual_sales_volume_and_price_l1879_187994

noncomputable def original_price : ℝ := 25
noncomputable def original_sales_volume : ℝ := 80000
noncomputable def sales_volume_decrease_per_yuan_increase : ℝ := 2000

-- Question 1
theorem highest_price_without_lowering_revenue :
  ∀ (x : ℝ), 
  25 ≤ x ∧ (8 - (x - original_price) * 0.2) * x ≥ 25 * 8 → 
  x ≤ 40 :=
sorry

-- Question 2
noncomputable def tech_reform_fee (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600)
noncomputable def fixed_promotion_fee : ℝ := 50
noncomputable def variable_promotion_fee (x : ℝ) : ℝ := (1 / 5) * x

theorem minimum_annual_sales_volume_and_price (x : ℝ) (a : ℝ) :
  x > 25 →
  (a * x ≥ 25 * 8 + fixed_promotion_fee + tech_reform_fee x + variable_promotion_fee x) →
  (a ≥ 10.2 ∧ x = 30) :=
sorry

end NUMINAMATH_GPT_highest_price_without_lowering_revenue_minimum_annual_sales_volume_and_price_l1879_187994


namespace NUMINAMATH_GPT_Hannah_cut_strands_l1879_187938

variable (H : ℕ)

theorem Hannah_cut_strands (h : 2 * (H + 3) = 22) : H = 8 :=
by
  sorry

end NUMINAMATH_GPT_Hannah_cut_strands_l1879_187938


namespace NUMINAMATH_GPT_minimum_z_value_l1879_187908

theorem minimum_z_value (x y : ℝ) (h : (x - 2)^2 + (y - 3)^2 = 1) : x^2 + y^2 ≥ 14 - 2 * Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_minimum_z_value_l1879_187908


namespace NUMINAMATH_GPT_part1_part2_l1879_187977

section
  variable {x a : ℝ}

  def f (x a : ℝ) := |x - a| + 3 * x

  theorem part1 (h : a = 1) : 
    (∀ x, f x a ≥ 3 * x + 2 ↔ (x ≥ 3 ∨ x ≤ -1)) :=
    sorry

  theorem part2 : 
    (∀ x, (f x a) ≤ 0 ↔ (x ≤ -1)) → a = 2 :=
    sorry
end

end NUMINAMATH_GPT_part1_part2_l1879_187977


namespace NUMINAMATH_GPT_trigonometric_identity_l1879_187910

noncomputable def sin110cos40_minus_cos70sin40 : ℝ := 
  Real.sin (110 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) - 
  Real.cos (70 * Real.pi / 180) * Real.sin (40 * Real.pi / 180)

theorem trigonometric_identity : 
  sin110cos40_minus_cos70sin40 = 1 / 2 := 
by sorry

end NUMINAMATH_GPT_trigonometric_identity_l1879_187910


namespace NUMINAMATH_GPT_probability_sum_3_correct_l1879_187976

noncomputable def probability_of_sum_3 : ℚ := 2 / 36

theorem probability_sum_3_correct :
  probability_of_sum_3 = 1 / 18 :=
by
  sorry

end NUMINAMATH_GPT_probability_sum_3_correct_l1879_187976


namespace NUMINAMATH_GPT_starting_number_is_10_l1879_187987

axiom between_nums_divisible_by_10 (n : ℕ) : 
  (∃ start : ℕ, start ≤ n ∧ n ≤ 76 ∧ 
  ∀ m, start ≤ m ∧ m ≤ n → m % 10 = 0 ∧ 
  (¬ (76 % 10 = 0) → start = 10) ∧ 
  ((76 - (76 % 10)) / 10 = 6) )

theorem starting_number_is_10 
  (start : ℕ) 
  (h1 : ∃ n, (start ≤ n ∧ n ≤ 76 ∧ 
             ∀ m, start ≤ m ∧ m ≤ n → m % 10 = 0 ∧ 
             (n - start) / 10 = 6)):
  start = 10 :=
sorry

end NUMINAMATH_GPT_starting_number_is_10_l1879_187987


namespace NUMINAMATH_GPT_lizzie_scored_six_l1879_187934

-- Definitions based on the problem conditions
def lizzie_score : Nat := sorry
def nathalie_score := lizzie_score + 3
def aimee_score := 2 * (lizzie_score + nathalie_score)

-- Total score condition
def total_score := 50
def teammates_score := 17
def combined_score := total_score - teammates_score

-- Proven statement
theorem lizzie_scored_six:
  (lizzie_score + nathalie_score + aimee_score = combined_score) → lizzie_score = 6 :=
by sorry

end NUMINAMATH_GPT_lizzie_scored_six_l1879_187934


namespace NUMINAMATH_GPT_min_a_for_inequality_l1879_187947

theorem min_a_for_inequality :
  (∀ (x : ℝ), |x + a| - |x + 1| ≤ 2 * a) → a ≥ 1/3 :=
sorry

end NUMINAMATH_GPT_min_a_for_inequality_l1879_187947


namespace NUMINAMATH_GPT_student_A_more_stable_than_B_l1879_187967

theorem student_A_more_stable_than_B 
    (avg_A : ℝ := 98) (avg_B : ℝ := 98) 
    (var_A : ℝ := 0.2) (var_B : ℝ := 0.8) : 
    var_A < var_B :=
by sorry

end NUMINAMATH_GPT_student_A_more_stable_than_B_l1879_187967


namespace NUMINAMATH_GPT_problem1_problem2_l1879_187917

def f (x y : ℝ) : ℝ := x^2 * y

def P0 : ℝ × ℝ := (5, 4)

def Δx : ℝ := 0.1
def Δy : ℝ := -0.2

def Δf (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (Δx Δy : ℝ) : ℝ :=
  f (P.1 + Δx) (P.2 + Δy) - f P.1 P.2

def df (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (Δx Δy : ℝ) : ℝ :=
  (2 * P.1 * P.2) * Δx + (P.1^2) * Δy

theorem problem1 : Δf f P0 Δx Δy = -1.162 := 
  sorry

theorem problem2 : df f P0 Δx Δy = -1 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1879_187917


namespace NUMINAMATH_GPT_mary_days_eq_11_l1879_187907

variable (x : ℝ) -- Number of days Mary takes to complete the work
variable (m_eff : ℝ) -- Efficiency of Mary (work per day)
variable (r_eff : ℝ) -- Efficiency of Rosy (work per day)

-- Given conditions
axiom rosy_efficiency : r_eff = 1.1 * m_eff
axiom rosy_days : r_eff * 10 = 1

-- Define the efficiency of Mary in terms of days
axiom mary_efficiency : m_eff = 1 / x

-- The theorem to prove
theorem mary_days_eq_11 : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_mary_days_eq_11_l1879_187907


namespace NUMINAMATH_GPT_min_cookies_divisible_by_13_l1879_187992

theorem min_cookies_divisible_by_13 (a b : ℕ) : ∃ n : ℕ, n > 0 ∧ n % 13 = 0 ∧ (∃ a b : ℕ, n = 10 * a + 21 * b) ∧ n = 52 :=
by
  sorry

end NUMINAMATH_GPT_min_cookies_divisible_by_13_l1879_187992


namespace NUMINAMATH_GPT_ram_weight_increase_percentage_l1879_187902

theorem ram_weight_increase_percentage :
  ∃ r s r_new: ℝ,
  r / s = 4 / 5 ∧ 
  r + s = 72 ∧ 
  s * 1.19 = 47.6 ∧
  r_new = 82.8 - 47.6 ∧ 
  (r_new - r) / r * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_ram_weight_increase_percentage_l1879_187902


namespace NUMINAMATH_GPT_sin_identity_proof_l1879_187980

theorem sin_identity_proof (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) :
  Real.sin (5 * π / 6 - x) + Real.sin (π / 3 - x) ^ 2 = 19 / 16 :=
by
  sorry

end NUMINAMATH_GPT_sin_identity_proof_l1879_187980


namespace NUMINAMATH_GPT_repaved_before_today_correct_l1879_187931

variable (total_repaved_so_far repaved_today repaved_before_today : ℕ)

axiom given_conditions : total_repaved_so_far = 4938 ∧ repaved_today = 805 

theorem repaved_before_today_correct :
  total_repaved_so_far = 4938 →
  repaved_today = 805 →
  repaved_before_today = total_repaved_so_far - repaved_today →
  repaved_before_today = 4133 :=
by
  intros
  sorry

end NUMINAMATH_GPT_repaved_before_today_correct_l1879_187931


namespace NUMINAMATH_GPT_smaller_number_l1879_187941

theorem smaller_number (x y : ℤ) (h1 : x + y = 22) (h2 : x - y = 16) : y = 3 :=
by
  sorry

end NUMINAMATH_GPT_smaller_number_l1879_187941


namespace NUMINAMATH_GPT_team_total_mistakes_l1879_187937

theorem team_total_mistakes (total_questions : ℕ) (riley_mistakes : ℕ) (ofelia_correction: (ℕ → ℕ) ) : total_questions = 35 → riley_mistakes = 3 → (∀ riley_correct_answers, riley_correct_answers = total_questions - riley_mistakes → ofelia_correction riley_correct_answers = (riley_correct_answers / 2) + 5) → (riley_mistakes + (total_questions - (ofelia_correction (total_questions - riley_mistakes)))) = 17 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_team_total_mistakes_l1879_187937


namespace NUMINAMATH_GPT_graph_does_not_pass_through_second_quadrant_l1879_187969

theorem graph_does_not_pass_through_second_quadrant :
  ¬ ∃ x : ℝ, x < 0 ∧ 2 * x - 3 > 0 :=
by
  -- Include the necessary steps to complete the proof, but for now we provide a placeholder:
  sorry

end NUMINAMATH_GPT_graph_does_not_pass_through_second_quadrant_l1879_187969


namespace NUMINAMATH_GPT_chef_meals_prepared_l1879_187974

theorem chef_meals_prepared (S D_added D_total L R : ℕ)
  (hS : S = 12)
  (hD_added : D_added = 5)
  (hD_total : D_total = 10)
  (hR : R + D_added = D_total)
  (hL : L = S + R) : L = 17 :=
by
  sorry

end NUMINAMATH_GPT_chef_meals_prepared_l1879_187974


namespace NUMINAMATH_GPT_fraction_value_l1879_187929

variable {x y : ℝ}

theorem fraction_value (hx : x ≠ 0) (hy : y ≠ 0) (h : (2 * x - 3 * y) / (x + 2 * y) = 3) :
  (x - 2 * y) / (2 * x + 3 * y) = 11 / 15 :=
  sorry

end NUMINAMATH_GPT_fraction_value_l1879_187929


namespace NUMINAMATH_GPT_jared_march_texts_l1879_187918

def T (n : ℕ) : ℕ := ((n ^ 2) + 1) * (n.factorial)

theorem jared_march_texts : T 5 = 3120 := by
  -- The details of the proof would go here, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_jared_march_texts_l1879_187918


namespace NUMINAMATH_GPT_sum_divisible_by_7_l1879_187935

theorem sum_divisible_by_7 (n : ℕ) : (8^n + 6) % 7 = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_divisible_by_7_l1879_187935


namespace NUMINAMATH_GPT_symmetric_complex_division_l1879_187919

theorem symmetric_complex_division :
  (∀ (z1 z2 : ℂ), z1 = 3 - (1 : ℂ) * Complex.I ∧ z2 = -(Complex.re z1) + (Complex.im z1) * Complex.I 
   → (z1 / z2) = -4/5 + (3/5) * Complex.I) := sorry

end NUMINAMATH_GPT_symmetric_complex_division_l1879_187919


namespace NUMINAMATH_GPT_sum_of_coordinates_eq_69_l1879_187903

theorem sum_of_coordinates_eq_69 {f k : ℝ → ℝ} (h₁ : f 4 = 8) (h₂ : ∀ x, k x = (f x)^2 + 1) : 4 + k 4 = 69 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_eq_69_l1879_187903


namespace NUMINAMATH_GPT_equilateral_triangle_coloring_l1879_187964

theorem equilateral_triangle_coloring (color : Fin 3 → Prop) :
  (∀ i, color i = true ∨ color i = false) →
  ∃ i j : Fin 3, i ≠ j ∧ color i = color j :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_coloring_l1879_187964


namespace NUMINAMATH_GPT_find_k_l1879_187921

theorem find_k (k : ℝ) : 
  (∀ x : ℝ, -4 < x ∧ x < 3 → k * (x^2 + 6 * x - k) * (x^2 + x - 12) > 0) ↔ (k ≤ -9) :=
by sorry

end NUMINAMATH_GPT_find_k_l1879_187921


namespace NUMINAMATH_GPT_parallelogram_area_l1879_187970

theorem parallelogram_area (base height : ℝ) (h_base : base = 10) (h_height : height = 20) :
  base * height = 200 := 
by 
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1879_187970


namespace NUMINAMATH_GPT_parallelogram_sides_eq_l1879_187993

theorem parallelogram_sides_eq (x y : ℚ) :
  (5 * x - 2 = 10 * x - 4) → 
  (3 * y + 7 = 6 * y + 13) → 
  x + y = -1.6 := by
  sorry

end NUMINAMATH_GPT_parallelogram_sides_eq_l1879_187993


namespace NUMINAMATH_GPT_kenneth_money_left_l1879_187952

theorem kenneth_money_left (I : ℕ) (C_b : ℕ) (N_b : ℕ) (C_w : ℕ) (N_w : ℕ) (L : ℕ) :
  I = 50 → C_b = 2 → N_b = 2 → C_w = 1 → N_w = 2 → L = I - (N_b * C_b + N_w * C_w) → L = 44 :=
by
  intros h₀ h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_kenneth_money_left_l1879_187952


namespace NUMINAMATH_GPT_limit_to_infinity_zero_l1879_187965

variable (f : ℝ → ℝ)

theorem limit_to_infinity_zero (h_continuous : Continuous f)
  (h_alpha : ∀ (α : ℝ), α > 0 → Filter.Tendsto (fun n : ℕ => f (n * α)) Filter.atTop (nhds 0)) :
  Filter.Tendsto f Filter.atTop (nhds 0) :=
sorry

end NUMINAMATH_GPT_limit_to_infinity_zero_l1879_187965


namespace NUMINAMATH_GPT_families_with_neither_l1879_187920

theorem families_with_neither (total_families : ℕ) (families_with_cats : ℕ) (families_with_dogs : ℕ) (families_with_both : ℕ) :
  total_families = 40 → families_with_cats = 18 → families_with_dogs = 24 → families_with_both = 10 → 
  total_families - (families_with_cats + families_with_dogs - families_with_both) = 8 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_families_with_neither_l1879_187920


namespace NUMINAMATH_GPT_impossible_to_obtain_one_l1879_187925

theorem impossible_to_obtain_one (N : ℕ) (h : N % 3 = 0) : ¬(∃ k : ℕ, (∀ m : ℕ, (∃ q : ℕ, (N + 3 * m = 5 * q) ∧ (q = 1 → m + 1 ≤ k)))) :=
sorry

end NUMINAMATH_GPT_impossible_to_obtain_one_l1879_187925


namespace NUMINAMATH_GPT_num_integer_solutions_abs_eq_3_l1879_187927

theorem num_integer_solutions_abs_eq_3 :
  (∀ (x y : ℤ), (|x| + |y| = 3) → 
  ∃ (s : Finset (ℤ × ℤ)), s.card = 12 ∧ (∀ (a b : ℤ), (a, b) ∈ s ↔ (|a| + |b| = 3))) :=
by
  sorry

end NUMINAMATH_GPT_num_integer_solutions_abs_eq_3_l1879_187927


namespace NUMINAMATH_GPT_triangle_min_perimeter_l1879_187986

-- Definitions of points A, B, and C and the conditions specified in the problem.
def pointA : ℝ × ℝ := (3, 2)
def pointB (t : ℝ) : ℝ × ℝ := (t, t)
def pointC (c : ℝ) : ℝ × ℝ := (c, 0)

-- Main theorem which states that the minimum perimeter of triangle ABC is sqrt(26).
theorem triangle_min_perimeter : 
  ∃ (B C : ℝ × ℝ), B = pointB (B.1) ∧ C = pointC (C.1) ∧ 
  ∀ (B' C' : ℝ × ℝ), B' = pointB (B'.1) ∧ C' = pointC (C'.1) →
  (dist pointA B + dist B C + dist C pointA ≥ dist (2, 3) (3, -2)) :=
by 
  sorry

end NUMINAMATH_GPT_triangle_min_perimeter_l1879_187986


namespace NUMINAMATH_GPT_total_pitches_missed_l1879_187909

theorem total_pitches_missed (tokens_to_pitches : ℕ → ℕ) 
  (macy_used : ℕ) (piper_used : ℕ) 
  (macy_hits : ℕ) (piper_hits : ℕ) 
  (h1 : tokens_to_pitches 1 = 15) 
  (h_macy_used : macy_used = 11) 
  (h_piper_used : piper_used = 17) 
  (h_macy_hits : macy_hits = 50) 
  (h_piper_hits : piper_hits = 55) :
  let total_pitches := tokens_to_pitches macy_used + tokens_to_pitches piper_used
  let total_hits := macy_hits + piper_hits
  total_pitches - total_hits = 315 :=
by
  sorry

end NUMINAMATH_GPT_total_pitches_missed_l1879_187909


namespace NUMINAMATH_GPT_seth_pounds_lost_l1879_187988

-- Definitions
def pounds_lost_by_Seth (S : ℝ) : Prop := 
  let total_loss := S + 3 * S + (S + 1.5)
  total_loss = 89

theorem seth_pounds_lost (S : ℝ) : pounds_lost_by_Seth S → S = 17.5 := by
  sorry

end NUMINAMATH_GPT_seth_pounds_lost_l1879_187988


namespace NUMINAMATH_GPT_part_a_l1879_187949

theorem part_a (x y : ℝ) : (x + y) * (x^2 - x * y + y^2) = x^3 + y^3 := sorry

end NUMINAMATH_GPT_part_a_l1879_187949
