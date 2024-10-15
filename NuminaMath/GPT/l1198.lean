import Mathlib

namespace NUMINAMATH_GPT_min_value_frac_sum_l1198_119832

open Real

theorem min_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) :
    ∃ (z : ℝ), z = 1 + (sqrt 3) / 2 ∧ 
    (∀ t, (t > 0 → ∃ (u : ℝ), u > 0 ∧ t + u = 4 → ∀ t' (h : t' = (1 / t) + (3 / u)), t' ≥ z)) :=
by sorry

end NUMINAMATH_GPT_min_value_frac_sum_l1198_119832


namespace NUMINAMATH_GPT_smallest_base_for_100_l1198_119845

theorem smallest_base_for_100 :
  ∃ (b : ℕ), (b^2 ≤ 100) ∧ (100 < b^3) ∧ ∀ (b' : ℕ), (b'^2 ≤ 100) ∧ (100 < b'^3) → b ≤ b' :=
by
  sorry

end NUMINAMATH_GPT_smallest_base_for_100_l1198_119845


namespace NUMINAMATH_GPT_cos_triple_angle_l1198_119858

theorem cos_triple_angle
  (θ : ℝ)
  (h : Real.cos θ = 1/3) :
  Real.cos (3 * θ) = -23 / 27 :=
by
  sorry

end NUMINAMATH_GPT_cos_triple_angle_l1198_119858


namespace NUMINAMATH_GPT_find_angle_C_60_find_min_value_of_c_l1198_119812

theorem find_angle_C_60 (a b c : ℝ) (A B C : ℝ)
  (h_cos_eq : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) : 
  C = 60 := 
sorry

theorem find_min_value_of_c (a b c : ℝ) (A B C : ℝ)
  (h_cos_eq : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h_area : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) :
  c ≥ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_find_angle_C_60_find_min_value_of_c_l1198_119812


namespace NUMINAMATH_GPT_min_value_of_expression_l1198_119878

theorem min_value_of_expression (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + y = 1) : 
  ∃ min_value, min_value = 9 / 2 ∧ ∀ z, z = (1 / (x + 1) + 4 / y) → z ≥ min_value :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1198_119878


namespace NUMINAMATH_GPT_fathers_age_after_further_8_years_l1198_119880

variable (R F : ℕ)

def age_relation_1 : Prop := F = 4 * R
def age_relation_2 : Prop := F + 8 = (5 * (R + 8)) / 2

theorem fathers_age_after_further_8_years (h1 : age_relation_1 R F) (h2 : age_relation_2 R F) : (F + 16) = 2 * (R + 16) :=
by 
  sorry

end NUMINAMATH_GPT_fathers_age_after_further_8_years_l1198_119880


namespace NUMINAMATH_GPT_jack_bill_age_difference_l1198_119839

def jack_bill_ages_and_difference (a b : ℕ) :=
  let jack_age := 10 * a + b
  let bill_age := 10 * b + a
  (a + b = 2) ∧ (7 * a - 29 * b = 14) → jack_age - bill_age = 18

theorem jack_bill_age_difference (a b : ℕ) (h₀ : a + b = 2) (h₁ : 7 * a - 29 * b = 14) : 
  let jack_age := 10 * a + b
  let bill_age := 10 * b + a
  jack_age - bill_age = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_jack_bill_age_difference_l1198_119839


namespace NUMINAMATH_GPT_two_pow_n_plus_one_square_or_cube_l1198_119817

theorem two_pow_n_plus_one_square_or_cube (n : ℕ) :
  (∃ a : ℕ, 2^n + 1 = a^2) ∨ (∃ a : ℕ, 2^n + 1 = a^3) → n = 3 :=
by
  sorry

end NUMINAMATH_GPT_two_pow_n_plus_one_square_or_cube_l1198_119817


namespace NUMINAMATH_GPT_min_a2_b2_l1198_119857

theorem min_a2_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + 2 * x^2 + b * x + 1 = 0) : a^2 + b^2 ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_a2_b2_l1198_119857


namespace NUMINAMATH_GPT_sum_of_adjacents_to_15_l1198_119813

-- Definitions of the conditions
def divisorsOf225 : Set ℕ := {3, 5, 9, 15, 25, 45, 75, 225}

-- Definition of the adjacency relationship
def isAdjacent (x y : ℕ) (s : Set ℕ) : Prop :=
  x ∈ s ∧ y ∈ s ∧ Nat.gcd x y > 1

-- Problem statement in Lean 4
theorem sum_of_adjacents_to_15 :
  ∃ x y : ℕ, isAdjacent 15 x divisorsOf225 ∧ isAdjacent 15 y divisorsOf225 ∧ x + y = 120 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_adjacents_to_15_l1198_119813


namespace NUMINAMATH_GPT_relationship_among_variables_l1198_119822

theorem relationship_among_variables (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (h1 : a^2 = 2) (h2 : b^3 = 3) (h3 : c^4 = 4) (h4 : d^5 = 5) : a = c ∧ a < d ∧ d < b := 
by
  sorry

end NUMINAMATH_GPT_relationship_among_variables_l1198_119822


namespace NUMINAMATH_GPT_total_colored_hangers_l1198_119876

theorem total_colored_hangers (pink_hangers green_hangers : ℕ) (h1 : pink_hangers = 7) (h2 : green_hangers = 4)
  (blue_hangers yellow_hangers : ℕ) (h3 : blue_hangers = green_hangers - 1) (h4 : yellow_hangers = blue_hangers - 1) :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_colored_hangers_l1198_119876


namespace NUMINAMATH_GPT_boat_distance_travelled_upstream_l1198_119884

theorem boat_distance_travelled_upstream (v : ℝ) (d : ℝ) :
  ∀ (boat_speed_in_still_water upstream_time downstream_time : ℝ),
  boat_speed_in_still_water = 25 →
  upstream_time = 1 →
  downstream_time = 0.25 →
  d = (boat_speed_in_still_water - v) * upstream_time →
  d = (boat_speed_in_still_water + v) * downstream_time →
  d = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boat_distance_travelled_upstream_l1198_119884


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1198_119853

theorem geometric_sequence_sum (a_n : ℕ → ℝ) (q : ℝ) (n : ℕ) 
    (S_n : ℝ) (S_3n : ℝ) (S_4n : ℝ)
    (h1 : S_n = 2) 
    (h2 : S_3n = 14) 
    (h3 : ∀ m : ℕ, S_m = a_n 1 * (1 - q^m) / (1 - q)) :
    S_4n = 30 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1198_119853


namespace NUMINAMATH_GPT_sequence_formula_sequence_inequality_l1198_119867

open Nat

-- Definition of the sequence based on the given conditions
noncomputable def a : ℕ → ℚ
| 0     => 1                -- 0-indexed for Lean handling convenience, a_1 = 1 is a(0) in Lean
| (n+1) => 2 - 1 / (a n)    -- recurrence relation

-- Proof for part (I) that a_n = (n + 1) / n
theorem sequence_formula (n : ℕ) : a (n + 1) = (n + 2) / (n + 1) := sorry

-- Proof for part (II)
theorem sequence_inequality (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  (1 + a (n + 1)) / a (k + 1) < 2 ∨ (1 + a (k + 1)) / a (n + 1) < 2 := sorry

end NUMINAMATH_GPT_sequence_formula_sequence_inequality_l1198_119867


namespace NUMINAMATH_GPT_work_rate_l1198_119855

theorem work_rate (x : ℝ) (h : (1 / x + 1 / 15 = 1 / 6)) : x = 10 :=
sorry

end NUMINAMATH_GPT_work_rate_l1198_119855


namespace NUMINAMATH_GPT_geometric_sequence_eighth_term_l1198_119849

theorem geometric_sequence_eighth_term 
  (a : ℕ) (r : ℕ) (h1 : a = 4) (h2 : r = 16 / 4) :
  a * r^(7) = 65536 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_eighth_term_l1198_119849


namespace NUMINAMATH_GPT_smallest_integer_rel_prime_to_1020_l1198_119807

theorem smallest_integer_rel_prime_to_1020 : ∃ n : ℕ, n > 1 ∧ n = 7 ∧ gcd n 1020 = 1 := by
  -- Here we state the theorem
  sorry

end NUMINAMATH_GPT_smallest_integer_rel_prime_to_1020_l1198_119807


namespace NUMINAMATH_GPT_board_game_cost_l1198_119866

theorem board_game_cost
  (v h : ℝ)
  (h1 : 3 * v = h + 490)
  (h2 : 5 * v = 2 * h + 540) :
  h = 830 := by
  sorry

end NUMINAMATH_GPT_board_game_cost_l1198_119866


namespace NUMINAMATH_GPT_chloe_total_books_l1198_119834

noncomputable def total_books (average_books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (science_fiction_shelves : ℕ) (history_shelves : ℕ) : ℕ :=
  (mystery_shelves + picture_shelves + science_fiction_shelves + history_shelves) * average_books_per_shelf

theorem chloe_total_books : 
  total_books 85 7 5 3 2 = 14500 / 100 :=
  by
  sorry

end NUMINAMATH_GPT_chloe_total_books_l1198_119834


namespace NUMINAMATH_GPT_find_correction_time_l1198_119899

-- Define the conditions
def loses_minutes_per_day : ℚ := 2 + 1/2
def initial_time_set : ℚ := 1 * 60 -- 1 PM in minutes
def time_on_march_21 : ℚ := 9 * 60 -- 9 AM in minutes on March 21
def total_minutes_per_day : ℚ := 24 * 60
def days_between : ℚ := 6 - 4/24 -- 6 days minus 4 hours

-- Calculate effective functioning minutes per day
def effective_minutes_per_day : ℚ := total_minutes_per_day - loses_minutes_per_day

-- Calculate the ratio of actual time to the watch's time
def time_ratio : ℚ := total_minutes_per_day / effective_minutes_per_day

-- Calculate the total actual time in minutes between initial set time and the given time showing on the watch
def total_actual_time : ℚ := days_between * total_minutes_per_day + initial_time_set

-- Calculate the actual time according to the ratio
def actual_time_according_to_ratio : ℚ := total_actual_time * time_ratio

-- Calculate the correction required 'n'
def required_minutes_correction : ℚ := actual_time_according_to_ratio - total_actual_time

-- The theorem stating that the required correction is as calculated
theorem find_correction_time : required_minutes_correction = (14 + 14/23) := by
  sorry

end NUMINAMATH_GPT_find_correction_time_l1198_119899


namespace NUMINAMATH_GPT_Sandy_original_number_l1198_119891

theorem Sandy_original_number (x : ℝ) (h : (3 * x + 20)^2 = 2500) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_Sandy_original_number_l1198_119891


namespace NUMINAMATH_GPT_average_percent_score_l1198_119864

def num_students : ℕ := 180

def score_distrib : List (ℕ × ℕ) :=
[(95, 12), (85, 30), (75, 50), (65, 45), (55, 30), (45, 13)]

noncomputable def total_score : ℕ :=
(95 * 12) + (85 * 30) + (75 * 50) + (65 * 45) + (55 * 30) + (45 * 13)

noncomputable def average_score : ℕ :=
total_score / num_students

theorem average_percent_score : average_score = 70 :=
by 
  -- Here you would provide the proof, but for now we will leave it as:
  sorry

end NUMINAMATH_GPT_average_percent_score_l1198_119864


namespace NUMINAMATH_GPT_length_of_other_train_is_correct_l1198_119860

noncomputable def length_of_other_train
  (l1 : ℝ) -- length of the first train in meters
  (s1 : ℝ) -- speed of the first train in km/hr
  (s2 : ℝ) -- speed of the second train in km/hr
  (t : ℝ)  -- time in seconds
  (h1 : l1 = 500)
  (h2 : s1 = 240)
  (h3 : s2 = 180)
  (h4 : t = 12) :
  ℝ :=
  let s1_m_s := s1 * 1000 / 3600
  let s2_m_s := s2 * 1000 / 3600
  let relative_speed := s1_m_s + s2_m_s
  let total_distance := relative_speed * t
  total_distance - l1

theorem length_of_other_train_is_correct :
  length_of_other_train 500 240 180 12 rfl rfl rfl rfl = 900 := sorry

end NUMINAMATH_GPT_length_of_other_train_is_correct_l1198_119860


namespace NUMINAMATH_GPT_find_width_of_metallic_sheet_l1198_119862

noncomputable def width_of_metallic_sheet (w : ℝ) : Prop :=
  let length := 48
  let square_side := 8
  let new_length := length - 2 * square_side
  let new_width := w - 2 * square_side
  let height := square_side
  let volume := new_length * new_width * height
  volume = 5120

theorem find_width_of_metallic_sheet (w : ℝ) :
  width_of_metallic_sheet w -> w = 36 := 
sorry

end NUMINAMATH_GPT_find_width_of_metallic_sheet_l1198_119862


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1198_119843

-- Definitions for the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4}

-- Proof statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1198_119843


namespace NUMINAMATH_GPT_smallest_other_divisor_of_40_l1198_119833

theorem smallest_other_divisor_of_40 (n : ℕ) (h₁ : n > 1) (h₂ : 40 % n = 0) (h₃ : n ≠ 8) :
  (∀ m : ℕ, m > 1 → 40 % m = 0 → m ≠ 8 → n ≤ m) → n = 5 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_other_divisor_of_40_l1198_119833


namespace NUMINAMATH_GPT_complement_of_A_l1198_119820

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set A
def A : Set ℕ := {2, 4, 5}

-- Define the complement of A with respect to U
def CU : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- State the theorem that the complement of A with respect to U is {1, 3, 6, 7}
theorem complement_of_A : CU = {1, 3, 6, 7} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_l1198_119820


namespace NUMINAMATH_GPT_term_free_of_x_l1198_119893

namespace PolynomialExpansion

theorem term_free_of_x (m n k : ℕ) (h : (x : ℝ)^(m * k - (m + n) * r) = 1) :
  (m * k) % (m + n) = 0 :=
by
  sorry

end PolynomialExpansion

end NUMINAMATH_GPT_term_free_of_x_l1198_119893


namespace NUMINAMATH_GPT_divisible_by_7_iff_l1198_119825

variable {x y : ℤ}

theorem divisible_by_7_iff :
  7 ∣ (2 * x + 3 * y) ↔ 7 ∣ (5 * x + 4 * y) :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_7_iff_l1198_119825


namespace NUMINAMATH_GPT_fraction_scaling_l1198_119806

theorem fraction_scaling (x y : ℝ) :
  ((5 * x - 5 * 5 * y) / ((5 * x) ^ 2 + (5 * y) ^ 2)) = (1 / 5) * ((x - 5 * y) / (x ^ 2 + y ^ 2)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_scaling_l1198_119806


namespace NUMINAMATH_GPT_total_pieces_eq_21_l1198_119881

-- Definitions based on conditions
def red_pieces : Nat := 5
def yellow_pieces : Nat := 7
def green_pieces : Nat := 11

-- Derived definitions from conditions
def red_cuts : Nat := red_pieces - 1
def yellow_cuts : Nat := yellow_pieces - 1
def green_cuts : Nat := green_pieces - 1

-- Total cuts and the resulting total pieces
def total_cuts : Nat := red_cuts + yellow_cuts + green_cuts
def total_pieces : Nat := total_cuts + 1

-- Prove the total number of pieces is 21
theorem total_pieces_eq_21 : total_pieces = 21 := by
  sorry

end NUMINAMATH_GPT_total_pieces_eq_21_l1198_119881


namespace NUMINAMATH_GPT_exists_integers_abcd_l1198_119856

theorem exists_integers_abcd (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
sorry

end NUMINAMATH_GPT_exists_integers_abcd_l1198_119856


namespace NUMINAMATH_GPT_smallest_t_for_circle_covered_l1198_119805

theorem smallest_t_for_circle_covered:
  ∃ t, (∀ θ, 0 ≤ θ → θ ≤ t → (∃ r, r = Real.sin θ)) ∧
         (∀ t', (∀ θ, 0 ≤ θ → θ ≤ t' → (∃ r, r = Real.sin θ)) → t' ≥ t) :=
sorry

end NUMINAMATH_GPT_smallest_t_for_circle_covered_l1198_119805


namespace NUMINAMATH_GPT_probability_is_correct_l1198_119897

noncomputable def probability_cashier_opens_early : ℝ :=
  let x1 : ℝ := sorry
  let x2 : ℝ := sorry
  let x3 : ℝ := sorry
  let x4 : ℝ := sorry
  let x5 : ℝ := sorry
  let x6 : ℝ := sorry
  if (0 <= x1) ∧ (x1 <= 15) ∧
     (0 <= x2) ∧ (x2 <= 15) ∧
     (0 <= x3) ∧ (x3 <= 15) ∧
     (0 <= x4) ∧ (x4 <= 15) ∧
     (0 <= x5) ∧ (x5 <= 15) ∧
     (0 <= x6) ∧ (x6 <= 15) ∧
     (x1 < x6) ∧ (x2 < x6) ∧ (x3 < x6) ∧ (x4 < x6) ∧ (x5 < x6) then 
    let p_not_A : ℝ := (12 / 15) ^ 6
    1 - p_not_A
  else
    0

theorem probability_is_correct : probability_cashier_opens_early = 0.738 :=
by sorry

end NUMINAMATH_GPT_probability_is_correct_l1198_119897


namespace NUMINAMATH_GPT_MattSkipsRopesTimesPerSecond_l1198_119886

theorem MattSkipsRopesTimesPerSecond:
  ∀ (minutes_jumped : ℕ) (total_skips : ℕ), 
  minutes_jumped = 10 → 
  total_skips = 1800 → 
  (total_skips / (minutes_jumped * 60)) = 3 :=
by
  intros minutes_jumped total_skips h_jumped h_skips
  sorry

end NUMINAMATH_GPT_MattSkipsRopesTimesPerSecond_l1198_119886


namespace NUMINAMATH_GPT_product_of_roots_of_polynomial_l1198_119815

theorem product_of_roots_of_polynomial : 
  ∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x^2 - x - 34 = 0) ∧ (a * b = -34) :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_of_polynomial_l1198_119815


namespace NUMINAMATH_GPT_part_a_part_b_l1198_119803

-- Part (a)
theorem part_a (f : ℚ → ℝ) (h_add : ∀ x y : ℚ, f (x + y) = f x + f y) (h_mul : ∀ x y : ℚ, f (x * y) = f x * f y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = 0) :=
sorry

-- Part (b)
theorem part_b (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) (h_mul : ∀ x y : ℝ, f (x * y) = f x * f y) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 0) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1198_119803


namespace NUMINAMATH_GPT_f_sum_neg_l1198_119823

def f : ℝ → ℝ := sorry

theorem f_sum_neg (x₁ x₂ : ℝ)
  (h1 : ∀ x, f (4 - x) = - f x)
  (h2 : ∀ x, x < 2 → ∀ y, y < x → f y < f x)
  (h3 : x₁ + x₂ > 4)
  (h4 : (x₁ - 2) * (x₂ - 2) < 0)
  : f x₁ + f x₂ < 0 := 
sorry

end NUMINAMATH_GPT_f_sum_neg_l1198_119823


namespace NUMINAMATH_GPT_andy_last_problem_l1198_119898

theorem andy_last_problem (start_num : ℕ) (num_solved : ℕ) (result : ℕ) : 
  start_num = 78 → 
  num_solved = 48 → 
  result = start_num + num_solved - 1 → 
  result = 125 :=
by
  sorry

end NUMINAMATH_GPT_andy_last_problem_l1198_119898


namespace NUMINAMATH_GPT_M_eq_N_l1198_119824

noncomputable def M (a : ℝ) : ℝ :=
  a^2 + (a + 3)^2 + (a + 5)^2 + (a + 6)^2

noncomputable def N (a : ℝ) : ℝ :=
  (a + 1)^2 + (a + 2)^2 + (a + 4)^2 + (a + 7)^2

theorem M_eq_N (a : ℝ) : M a = N a :=
by
  sorry

end NUMINAMATH_GPT_M_eq_N_l1198_119824


namespace NUMINAMATH_GPT_pipe_fill_without_hole_l1198_119863

theorem pipe_fill_without_hole :
  ∀ (T : ℝ), 
  (1 / T - 1 / 60 = 1 / 20) → 
  T = 15 := 
by
  intros T h
  sorry

end NUMINAMATH_GPT_pipe_fill_without_hole_l1198_119863


namespace NUMINAMATH_GPT_find_coefficients_sum_l1198_119896

theorem find_coefficients_sum :
  let f := (2 * x - 1) ^ 5 + (x + 2) ^ 4
  let a_0 := 15
  let a_1 := 42
  let a_2 := -16
  let a_5 := 32
  (|a_0| + |a_1| + |a_2| + |a_5| = 105) := 
by {
  sorry
}

end NUMINAMATH_GPT_find_coefficients_sum_l1198_119896


namespace NUMINAMATH_GPT_speed_of_stream_l1198_119819

-- Define the problem conditions
variables (b s : ℝ)
axiom cond1 : 21 = b + s
axiom cond2 : 15 = b - s

-- State the theorem
theorem speed_of_stream : s = 3 :=
sorry

end NUMINAMATH_GPT_speed_of_stream_l1198_119819


namespace NUMINAMATH_GPT_problem_statement_l1198_119850

theorem problem_statement (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1198_119850


namespace NUMINAMATH_GPT_find_a5_l1198_119842

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a 1 + (n - 1) * d

theorem find_a5 (a : ℕ → ℤ) (d : ℤ)
  (h_seq : arithmetic_sequence a d)
  (h1 : a 1 + a 5 = 8)
  (h4 : a 4 = 7) : 
  a 5 = 10 := sorry

end NUMINAMATH_GPT_find_a5_l1198_119842


namespace NUMINAMATH_GPT_poly_constant_or_sum_constant_l1198_119873

-- definitions of the polynomials as real-coefficient polynomials
variables (P Q R : Polynomial ℝ)

-- conditions
#check ∀ x, P.eval (Q.eval x) + P.eval (R.eval x) = (1 : ℝ) -- Considering 'constant' as 1 for simplicity

-- target
theorem poly_constant_or_sum_constant 
  (h : ∀ x, P.eval (Q.eval x) + P.eval (R.eval x) = (1 : ℝ)) :
  (∃ c : ℝ, ∀ x, P.eval x = c) ∨ (∃ c : ℝ, ∀ x, Q.eval x + R.eval x = c) :=
sorry

end NUMINAMATH_GPT_poly_constant_or_sum_constant_l1198_119873


namespace NUMINAMATH_GPT_ones_digit_34_pow_34_pow_17_pow_17_l1198_119810

-- Definitions from the conditions
def ones_digit (n : ℕ) : ℕ := n % 10

-- Translation of the original problem statement
theorem ones_digit_34_pow_34_pow_17_pow_17 :
  ones_digit (34 ^ (34 * 17 ^ 17)) = 4 :=
sorry

end NUMINAMATH_GPT_ones_digit_34_pow_34_pow_17_pow_17_l1198_119810


namespace NUMINAMATH_GPT_country_of_second_se_asian_fields_medal_recipient_l1198_119874

-- Given conditions as definitions
def is_highest_recognition (award : String) : Prop :=
  award = "Fields Medal"

def fields_medal_freq (years : Nat) : Prop :=
  years = 4 -- Fields Medal is awarded every four years

def second_se_asian_recipient (name : String) : Prop :=
  name = "Ngo Bao Chau"

-- The main theorem to prove
theorem country_of_second_se_asian_fields_medal_recipient :
  ∀ (award : String) (years : Nat) (name : String),
    is_highest_recognition award ∧ fields_medal_freq years ∧ second_se_asian_recipient name →
    (name = "Ngo Bao Chau" → ∃ (country : String), country = "Vietnam") :=
by
  intros award years name h
  sorry

end NUMINAMATH_GPT_country_of_second_se_asian_fields_medal_recipient_l1198_119874


namespace NUMINAMATH_GPT_brass_weight_l1198_119829

theorem brass_weight (copper zinc brass : ℝ) (h_ratio : copper / zinc = 3 / 7) (h_zinc : zinc = 70) : brass = 100 :=
by
  sorry

end NUMINAMATH_GPT_brass_weight_l1198_119829


namespace NUMINAMATH_GPT_tan_value_of_point_on_exp_graph_l1198_119800

theorem tan_value_of_point_on_exp_graph (a : ℝ) (h1 : (a, 9) ∈ {p : ℝ × ℝ | ∃ x, p = (x, 3^x)}) : 
  Real.tan (a * Real.pi / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_value_of_point_on_exp_graph_l1198_119800


namespace NUMINAMATH_GPT_matching_shoes_probability_is_one_ninth_l1198_119836

def total_shoes : ℕ := 10
def pairs_of_shoes : ℕ := 5
def total_combinations : ℕ := (total_shoes * (total_shoes - 1)) / 2
def matching_combinations : ℕ := pairs_of_shoes

def matching_shoes_probability : ℚ := matching_combinations / total_combinations

theorem matching_shoes_probability_is_one_ninth :
  matching_shoes_probability = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_matching_shoes_probability_is_one_ninth_l1198_119836


namespace NUMINAMATH_GPT_units_digit_of_expression_l1198_119872

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_expression : units_digit (7 * 18 * 1978 - 7^4) = 7 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_expression_l1198_119872


namespace NUMINAMATH_GPT_max_value_of_f_l1198_119814

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^3 + Real.cos (2 * x) - (Real.cos x)^2 - Real.sin x

theorem max_value_of_f :
  ∃ x : ℝ, f x = 5 / 27 ∧ ∀ y : ℝ, f y ≤ 5 / 27 :=
sorry

end NUMINAMATH_GPT_max_value_of_f_l1198_119814


namespace NUMINAMATH_GPT_function_relationship_l1198_119846

variable {A B : Type} [Nonempty A] [Nonempty B]
variable (f : A → B) 

def domain (f : A → B) : Set A := {a | ∃ b, f a = b}
def range (f : A → B) : Set B := {b | ∃ a, f a = b}

theorem function_relationship (M : Set A) (N : Set B) (hM : M = Set.univ)
                              (hN : N = range f) : M = Set.univ ∧ N ⊆ Set.univ :=
  sorry

end NUMINAMATH_GPT_function_relationship_l1198_119846


namespace NUMINAMATH_GPT_aleksey_divisible_l1198_119821

theorem aleksey_divisible
  (x y a b S : ℤ)
  (h1 : x + y = S)
  (h2 : S ∣ (a * x + b * y)) :
  S ∣ (b * x + a * y) := 
sorry

end NUMINAMATH_GPT_aleksey_divisible_l1198_119821


namespace NUMINAMATH_GPT_least_number_subtracted_l1198_119854

/-- The least number that must be subtracted from 50248 so that the 
remaining number is divisible by both 20 and 37 is 668. -/
theorem least_number_subtracted (n : ℕ) (x : ℕ ) (y : ℕ ) (a : ℕ) (b : ℕ) :
  n = 50248 → x = 20 → y = 37 → (a = 20 * 37) →
  (50248 - b) % a = 0 → 50248 - b < a → b = 668 :=
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l1198_119854


namespace NUMINAMATH_GPT_locus_of_center_of_circle_l1198_119852

theorem locus_of_center_of_circle (x y a : ℝ)
  (hC : x^2 + y^2 - (2 * a^2 - 4) * x - 4 * a^2 * y + 5 * a^4 - 4 = 0) :
  2 * x - y + 4 = 0 ∧ -2 ≤ x ∧ x < 0 :=
sorry

end NUMINAMATH_GPT_locus_of_center_of_circle_l1198_119852


namespace NUMINAMATH_GPT_final_position_correct_total_distance_correct_l1198_119835

def movements : List Int := [15, -25, 20, -35]

-- Final Position: 
def final_position (moves : List Int) : Int := moves.sum

-- Total Distance Traveled calculated by taking the absolutes and summing:
def total_distance (moves : List Int) : Nat :=
  moves.map (λ x => Int.natAbs x) |>.sum

theorem final_position_correct : final_position movements = -25 :=
by
  sorry

theorem total_distance_correct : total_distance movements = 95 :=
by
  sorry

end NUMINAMATH_GPT_final_position_correct_total_distance_correct_l1198_119835


namespace NUMINAMATH_GPT_lcm_of_three_numbers_l1198_119811

theorem lcm_of_three_numbers :
  ∀ (a b c : ℕ) (hcf : ℕ), hcf = Nat.gcd (Nat.gcd a b) c → a = 136 → b = 144 → c = 168 → hcf = 8 →
  Nat.lcm (Nat.lcm a b) c = 411264 :=
by
  intros a b c hcf h1 h2 h3 h4
  rw [h2, h3, h4]
  sorry

end NUMINAMATH_GPT_lcm_of_three_numbers_l1198_119811


namespace NUMINAMATH_GPT_rectangle_ratio_l1198_119802

noncomputable def ratio_of_length_to_width (w : ℝ) : ℝ :=
  40 / w

theorem rectangle_ratio (w : ℝ) 
  (hw1 : 35 * (w + 5) = 40 * w + 75) : 
  ratio_of_length_to_width w = 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l1198_119802


namespace NUMINAMATH_GPT_sandwiches_per_day_l1198_119801

theorem sandwiches_per_day (S : ℕ) 
  (h1 : ∀ n, n = 4 * S)
  (h2 : 7 * 4 * S = 280) : S = 10 := 
by
  sorry

end NUMINAMATH_GPT_sandwiches_per_day_l1198_119801


namespace NUMINAMATH_GPT_exists_two_digit_pair_product_l1198_119840

theorem exists_two_digit_pair_product (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (hprod : a * b = 8670) : a * b = 8670 :=
by
  exact hprod

end NUMINAMATH_GPT_exists_two_digit_pair_product_l1198_119840


namespace NUMINAMATH_GPT_cost_of_cheaper_feed_l1198_119804

theorem cost_of_cheaper_feed (C : ℝ)
  (total_weight : ℝ) (weight_cheaper : ℝ) (price_expensive : ℝ) (total_value : ℝ) : 
  total_weight = 35 → 
  total_value = 0.36 * total_weight → 
  weight_cheaper = 17 → 
  price_expensive = 0.53 →
  (total_value = weight_cheaper * C + (total_weight - weight_cheaper) * price_expensive) →
  C = 0.18 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_cheaper_feed_l1198_119804


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1198_119895

variable (x : ℝ)

theorem sufficient_not_necessary (h : x^2 - 3 * x + 2 > 0) : x > 2 → (∀ x : ℝ, x^2 - 3 * x + 2 > 0 ↔ x > 2 ∨ x < -1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1198_119895


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1198_119883

-- Problem 1
theorem problem1 (x : ℝ) (h : x = 10) : (2 / 5) * x = 4 :=
by sorry

-- Problem 2
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : (2 / 5) * m = (- (1 / 5) * m^2) + 2 * m) : m = 8 :=
by sorry

-- Problem 3
theorem problem3 (t : ℝ) (h1 : ∃ t, (2 / 5) * (32 - t) + (- (1 / 5) * t^2) + 2 * t = 16) : true :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1198_119883


namespace NUMINAMATH_GPT_task_completion_time_l1198_119838

variable (x : Real) (y : Real)

theorem task_completion_time :
  (1 / 16) * y + (1 / 12) * x = 1 ∧ y + 5 = 8 → x = 3 ∧ y = 3 :=
  by {
    sorry 
  }

end NUMINAMATH_GPT_task_completion_time_l1198_119838


namespace NUMINAMATH_GPT_possible_values_of_expr_l1198_119877

-- Define conditions
variables (x y : ℝ)
axiom h1 : x + y = 2
axiom h2 : y > 0
axiom h3 : x ≠ 0

-- Define the expression we're investigating
noncomputable def expr : ℝ := (1 / (abs x)) + (abs x / (y + 2))

-- The statement of the problem
theorem possible_values_of_expr :
  expr x y = 3 / 4 ∨ expr x y = 5 / 4 :=
sorry

end NUMINAMATH_GPT_possible_values_of_expr_l1198_119877


namespace NUMINAMATH_GPT_prime_expression_integer_value_l1198_119828

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_expression_integer_value (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  ∃ n, (p * q + p^p + q^q) % (p + q) = 0 → n = 3 :=
by
  sorry

end NUMINAMATH_GPT_prime_expression_integer_value_l1198_119828


namespace NUMINAMATH_GPT_beads_taken_out_l1198_119875

/--
There is 1 green bead, 2 brown beads, and 3 red beads in a container.
Tom took some beads out of the container and left 4 in.
Prove that Tom took out 2 beads.
-/
theorem beads_taken_out : 
  let green_beads := 1
  let brown_beads := 2
  let red_beads := 3
  let initial_beads := green_beads + brown_beads + red_beads
  let beads_left := 4
  initial_beads - beads_left = 2 :=
by
  let green_beads := 1
  let brown_beads := 2
  let red_beads := 3
  let initial_beads := green_beads + brown_beads + red_beads
  let beads_left := 4
  show initial_beads - beads_left = 2
  sorry

end NUMINAMATH_GPT_beads_taken_out_l1198_119875


namespace NUMINAMATH_GPT_compare_abc_l1198_119887

open Real

theorem compare_abc
  (a b c : ℝ)
  (ha : 0 < a ∧ a < π / 2)
  (hb : 0 < b ∧ b < π / 2)
  (hc : 0 < c ∧ c < π / 2)
  (h1 : cos a = a)
  (h2 : sin (cos b) = b)
  (h3 : cos (sin c) = c) :
  c > a ∧ a > b :=
sorry

end NUMINAMATH_GPT_compare_abc_l1198_119887


namespace NUMINAMATH_GPT_distance_between_centers_of_circles_l1198_119890

theorem distance_between_centers_of_circles :
  ∀ (rect_width rect_height circle_radius distance_between_centers : ℝ),
  rect_width = 11 
  ∧ rect_height = 7 
  ∧ circle_radius = rect_height / 2 
  ∧ distance_between_centers = rect_width - 2 * circle_radius 
  → distance_between_centers = 4 := by
  intros rect_width rect_height circle_radius distance_between_centers
  sorry

end NUMINAMATH_GPT_distance_between_centers_of_circles_l1198_119890


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l1198_119809

def f (x : ℝ) : ℝ := x^5 - 6 * x^4 + 11 * x^3 + 21 * x^2 - 17 * x + 10

theorem remainder_when_divided_by_x_minus_2 : (f 2) = 84 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l1198_119809


namespace NUMINAMATH_GPT_total_amount_is_175_l1198_119885

noncomputable def calc_total_amount (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
x + y + z

theorem total_amount_is_175 (x y z : ℝ) 
  (h1 : y = 0.45 * x)
  (h2 : z = 0.30 * x)
  (h3 : y = 45) :
  calc_total_amount x y z = 175 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_total_amount_is_175_l1198_119885


namespace NUMINAMATH_GPT_shadow_projection_height_l1198_119870

theorem shadow_projection_height :
  ∃ (x : ℝ), (∃ (shadow_area : ℝ), shadow_area = 192) ∧ 1000 * x = 25780 :=
by
  sorry

end NUMINAMATH_GPT_shadow_projection_height_l1198_119870


namespace NUMINAMATH_GPT_cubic_polynomial_has_three_real_roots_l1198_119869

open Polynomial

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry
noncomputable def R : Polynomial ℝ := sorry

axiom P_degree : degree P = 2
axiom Q_degree : degree Q = 3
axiom R_degree : degree R = 3
axiom PQR_relationship : ∀ x : ℝ, P.eval x ^ 2 + Q.eval x ^ 2 = R.eval x ^ 2

theorem cubic_polynomial_has_three_real_roots : 
  (∃ x : ℝ, Q.eval x = 0 ∧ ∃ y : ℝ, Q.eval y = 0 ∧ ∃ z : ℝ, Q.eval z = 0) ∨
  (∃ x : ℝ, R.eval x = 0 ∧ ∃ y : ℝ, R.eval y = 0 ∧ ∃ z : ℝ, R.eval z = 0) :=
sorry

end NUMINAMATH_GPT_cubic_polynomial_has_three_real_roots_l1198_119869


namespace NUMINAMATH_GPT_race_distance_l1198_119871

theorem race_distance (D : ℝ)
  (A_time : D / 36 * 45 = D + 20) : 
  D = 80 :=
by
  sorry

end NUMINAMATH_GPT_race_distance_l1198_119871


namespace NUMINAMATH_GPT_tangent_properties_l1198_119831

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function f

-- Given conditions
axiom differentiable_f : Differentiable ℝ f
axiom func_eq : ∀ x, f (x - 2) = f (-x)
axiom tangent_eq_at_1 : ∀ x, (x = 1 → f x = 2 * x + 1)

-- Prove the required results
theorem tangent_properties :
  (deriv f 1 = 2) ∧ (∃ B C, (∀ x, (x = -3) → f x = B -2 * (x + 3)) ∧ (B = 3) ∧ (C = -3)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_properties_l1198_119831


namespace NUMINAMATH_GPT_pow_congruence_modulus_p_squared_l1198_119889

theorem pow_congruence_modulus_p_squared (p : ℕ) (a b : ℤ) (hp : Nat.Prime p) (h : a ≡ b [ZMOD p]) : a^p ≡ b^p [ZMOD p^2] :=
sorry

end NUMINAMATH_GPT_pow_congruence_modulus_p_squared_l1198_119889


namespace NUMINAMATH_GPT_four_students_three_classes_l1198_119818

-- Define the function that calculates the number of valid assignments
def valid_assignments (students : ℕ) (classes : ℕ) : ℕ :=
  if students = 4 ∧ classes = 3 then 36 else 0  -- Using given conditions to return 36 when appropriate

-- Define the theorem to prove that there are 36 valid ways
theorem four_students_three_classes : valid_assignments 4 3 = 36 :=
  by
  -- The proof is not required, so we use sorry to skip it
  sorry

end NUMINAMATH_GPT_four_students_three_classes_l1198_119818


namespace NUMINAMATH_GPT_average_pushups_is_correct_l1198_119847

theorem average_pushups_is_correct :
  ∀ (David Zachary Emily : ℕ),
    David = 510 →
    Zachary = David - 210 →
    Emily = David - 132 →
    (David + Zachary + Emily) / 3 = 396 :=
by
  intro David Zachary Emily hDavid hZachary hEmily
  -- All calculations and proofs will go here, but we'll leave them as sorry for now.
  sorry

end NUMINAMATH_GPT_average_pushups_is_correct_l1198_119847


namespace NUMINAMATH_GPT_parts_in_batch_l1198_119861

theorem parts_in_batch (a : ℕ) (h₁ : 20 * (a / 20) + 13 = a) (h₂ : 27 * (a / 27) + 20 = a) 
  (h₃ : 500 ≤ a) (h₄ : a ≤ 600) : a = 533 :=
by sorry

end NUMINAMATH_GPT_parts_in_batch_l1198_119861


namespace NUMINAMATH_GPT_necessary_condition_l1198_119827

theorem necessary_condition (m : ℝ) : 
  (∀ x > 0, (x / 2) + (1 / (2 * x)) - (3 / 2) > m) → (m ≤ -1 / 2) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_necessary_condition_l1198_119827


namespace NUMINAMATH_GPT_madeline_needs_work_hours_l1198_119892

def rent : ℝ := 1200
def groceries : ℝ := 400
def medical_expenses : ℝ := 200
def utilities : ℝ := 60
def emergency_savings : ℝ := 200
def hourly_wage : ℝ := 15

def total_expenses : ℝ := rent + groceries + medical_expenses + utilities + emergency_savings

noncomputable def total_hours_needed : ℝ := total_expenses / hourly_wage

theorem madeline_needs_work_hours :
  ⌈total_hours_needed⌉ = 138 := by
  sorry

end NUMINAMATH_GPT_madeline_needs_work_hours_l1198_119892


namespace NUMINAMATH_GPT_max_possible_player_salary_l1198_119837

theorem max_possible_player_salary (n : ℕ) (min_salary total_salary : ℕ) (num_players : ℕ) 
  (h1 : num_players = 24) 
  (h2 : min_salary = 20000) 
  (h3 : total_salary = 960000)
  (h4 : n = 23 * min_salary + 500000) 
  (h5 : 23 * min_salary + 500000 ≤ total_salary) 
  : n = total_salary :=
by {
  -- The proof will replace this sorry.
  sorry
}

end NUMINAMATH_GPT_max_possible_player_salary_l1198_119837


namespace NUMINAMATH_GPT_min_value_of_a_plus_b_l1198_119894

theorem min_value_of_a_plus_b (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc : 1 = 1) 
    (h1 : b^2 > 4 * a) (h2 : b < 2 * a) (h3 : b < a + 1) : a + b = 10 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_plus_b_l1198_119894


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1198_119826

theorem sum_of_reciprocals (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 11) :
  (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 31 / 21) :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1198_119826


namespace NUMINAMATH_GPT_triangle_with_angle_ratios_l1198_119865

theorem triangle_with_angle_ratios {α β γ : ℝ} (h : α + β + γ = 180 ∧ (α / 2 = β / 3) ∧ (α / 2 = γ / 5)) : (α = 90 ∨ β = 90 ∨ γ = 90) :=
by
  sorry

end NUMINAMATH_GPT_triangle_with_angle_ratios_l1198_119865


namespace NUMINAMATH_GPT_rationalize_denominator_l1198_119868

noncomputable def sqrt_12 := Real.sqrt 12
noncomputable def sqrt_5 := Real.sqrt 5
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_15 := Real.sqrt 15

theorem rationalize_denominator :
  (sqrt_12 + sqrt_5) / (sqrt_3 + sqrt_5) = (-1 / 2) + (sqrt_15 / 2) :=
sorry

end NUMINAMATH_GPT_rationalize_denominator_l1198_119868


namespace NUMINAMATH_GPT_value_of_n_l1198_119844

theorem value_of_n (n : ℝ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : n = 3.5 :=
by sorry

end NUMINAMATH_GPT_value_of_n_l1198_119844


namespace NUMINAMATH_GPT_cubic_root_sum_cubed_l1198_119808

theorem cubic_root_sum_cubed
  (p q r : ℂ)
  (h1 : 3 * p^3 - 9 * p^2 + 27 * p - 6 = 0)
  (h2 : 3 * q^3 - 9 * q^2 + 27 * q - 6 = 0)
  (h3 : 3 * r^3 - 9 * r^2 + 27 * r - 6 = 0)
  (hpq : p ≠ q)
  (hqr : q ≠ r)
  (hrp : r ≠ p) :
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 585 := 
  sorry

end NUMINAMATH_GPT_cubic_root_sum_cubed_l1198_119808


namespace NUMINAMATH_GPT_sum_of_areas_of_tangent_circles_l1198_119841

theorem sum_of_areas_of_tangent_circles :
  ∀ (a b c : ℝ), 
    a + b = 5 →
    a + c = 12 →
    b + c = 13 →
    π * (a^2 + b^2 + c^2) = 113 * π :=
by
  intros a b c h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_tangent_circles_l1198_119841


namespace NUMINAMATH_GPT_four_digit_number_l1198_119882

-- Defining the cards and their holders
def cards : List ℕ := [2, 0, 1, 5]
def A : ℕ := 5
def B : ℕ := 1
def C : ℕ := 2
def D : ℕ := 0

-- Conditions based on statements
def A_statement (a b c d : ℕ) : Prop := 
  ¬ ((b = a + 1) ∨ (b = a - 1) ∨ (c = a + 1) ∨ (c = a - 1) ∨ (d = a + 1) ∨ (d = a - 1))

def B_statement (a b c d : ℕ) : Prop := 
  (b = a + 1) ∨ (b = a - 1) ∨ (c = a + 1) ∨ (c = a - 1) ∨ (d = a + 1) ∨ (d = a - 1)

def C_statement (c : ℕ) : Prop := ¬ (c = 1 ∨ c = 2 ∨ c = 5)
def D_statement (d : ℕ) : Prop := d ≠ 0

-- Truth conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

def tells_truth (n : ℕ) : Prop := is_odd n
def lies (n : ℕ) : Prop := is_even n

-- Proof statement
theorem four_digit_number (a b c d : ℕ) 
  (ha : a ∈ cards) (hb : b ∈ cards) (hc : c ∈ cards) (hd : d ∈ cards) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (truth_A : tells_truth a → A_statement a b c d)
  (lie_A : lies a → ¬ A_statement a b c d)
  (truth_B : tells_truth b → B_statement a b c d)
  (lie_B : lies b → ¬ B_statement a b c d)
  (truth_C : tells_truth c → C_statement c)
  (lie_C : lies c → ¬ C_statement c)
  (truth_D : tells_truth d → D_statement d)
  (lie_D : lies d → ¬ D_statement d) :
  a * 1000 + b * 100 + c * 10 + d = 5120 := 
  by
    sorry

end NUMINAMATH_GPT_four_digit_number_l1198_119882


namespace NUMINAMATH_GPT_cost_of_first_ring_is_10000_l1198_119859

theorem cost_of_first_ring_is_10000 (x : ℝ) (h₁ : x + 2*x - x/2 = 25000) : x = 10000 :=
sorry

end NUMINAMATH_GPT_cost_of_first_ring_is_10000_l1198_119859


namespace NUMINAMATH_GPT_arithmetic_mean_q_r_l1198_119879

theorem arithmetic_mean_q_r (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 22) 
  (h3 : r - p = 24) : 
  (q + r) / 2 = 22 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_q_r_l1198_119879


namespace NUMINAMATH_GPT_slope_angle_l1198_119888

theorem slope_angle (A B : ℝ × ℝ) (θ : ℝ) (hA : A = (-1, 3)) (hB : B = (1, 1)) (hθ : θ ∈ Set.Ico 0 Real.pi)
  (hslope : Real.tan θ = (B.2 - A.2) / (B.1 - A.1)) :
  θ = (3 / 4) * Real.pi :=
by
  cases hA
  cases hB
  simp at hslope
  sorry

end NUMINAMATH_GPT_slope_angle_l1198_119888


namespace NUMINAMATH_GPT_middle_admitted_is_correct_l1198_119848

-- Define the total number of admitted people.
def total_admitted := 100

-- Define the proportions of South, North, and Middle volumes.
def south_ratio := 11
def north_ratio := 7
def middle_ratio := 2

-- Calculating the total ratio.
def total_ratio := south_ratio + north_ratio + middle_ratio

-- Hypothesis that we are dealing with the correct ratio and total.
def middle_admitted (total_admitted : ℕ) (total_ratio : ℕ) (middle_ratio : ℕ) : ℕ :=
  total_admitted * middle_ratio / total_ratio

-- Proof statement
theorem middle_admitted_is_correct :
  middle_admitted total_admitted total_ratio middle_ratio = 10 :=
by
  -- This line would usually contain the detailed proof steps, which are omitted here.
  sorry

end NUMINAMATH_GPT_middle_admitted_is_correct_l1198_119848


namespace NUMINAMATH_GPT_sum_of_x_y_l1198_119851

theorem sum_of_x_y (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 48) : x + y = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_x_y_l1198_119851


namespace NUMINAMATH_GPT_num_solutions_l1198_119816

theorem num_solutions :
  ∃ n, (∀ a b c : ℤ, (|a + b| + c = 21 ∧ a * b + |c| = 85) ↔ n = 12) :=
sorry

end NUMINAMATH_GPT_num_solutions_l1198_119816


namespace NUMINAMATH_GPT_sqrt_49_times_sqrt_25_eq_7sqrt5_l1198_119830

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end NUMINAMATH_GPT_sqrt_49_times_sqrt_25_eq_7sqrt5_l1198_119830
