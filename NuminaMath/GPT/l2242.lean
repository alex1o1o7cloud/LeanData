import Mathlib

namespace NUMINAMATH_GPT_range_of_f_is_real_l2242_224291

noncomputable def f (x : ℝ) (m : ℝ) := Real.log (5^x + 4 / 5^x + m)

theorem range_of_f_is_real (m : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f x m = y) ↔ m ≤ -4 :=
sorry

end NUMINAMATH_GPT_range_of_f_is_real_l2242_224291


namespace NUMINAMATH_GPT_inlet_pipe_rate_l2242_224206

theorem inlet_pipe_rate (capacity : ℕ) (t_empty : ℕ) (t_with_inlet : ℕ) (R_out : ℕ) :
  capacity = 6400 →
  t_empty = 10 →
  t_with_inlet = 16 →
  R_out = capacity / t_empty →
  (R_out - (capacity / t_with_inlet)) / 60 = 4 :=
by
  intros h1 h2 h3 h4 
  sorry

end NUMINAMATH_GPT_inlet_pipe_rate_l2242_224206


namespace NUMINAMATH_GPT_number_of_sides_on_die_l2242_224229

theorem number_of_sides_on_die (n : ℕ) 
  (h1 : n ≥ 6) 
  (h2 : (∃ k : ℕ, k = 5) → (5 : ℚ) / (n ^ 2 : ℚ) = (5 : ℚ) / (36 : ℚ)) 
  : n = 6 :=
sorry

end NUMINAMATH_GPT_number_of_sides_on_die_l2242_224229


namespace NUMINAMATH_GPT_B_days_to_complete_work_l2242_224242

theorem B_days_to_complete_work (B : ℕ) (hB : B ≠ 0)
  (A_work_days : ℕ := 9) (combined_days : ℕ := 6)
  (work_rate_A : ℚ := 1 / A_work_days) (work_rate_combined : ℚ := 1 / combined_days):
  (1 / B : ℚ) = work_rate_combined - work_rate_A → B = 18 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_B_days_to_complete_work_l2242_224242


namespace NUMINAMATH_GPT_ellen_painting_time_l2242_224296

def time_to_paint_lilies := 5
def time_to_paint_roses := 7
def time_to_paint_orchids := 3
def time_to_paint_vines := 2

def number_of_lilies := 17
def number_of_roses := 10
def number_of_orchids := 6
def number_of_vines := 20

def total_time := 213

theorem ellen_painting_time:
  time_to_paint_lilies * number_of_lilies +
  time_to_paint_roses * number_of_roses +
  time_to_paint_orchids * number_of_orchids +
  time_to_paint_vines * number_of_vines = total_time := by
  sorry

end NUMINAMATH_GPT_ellen_painting_time_l2242_224296


namespace NUMINAMATH_GPT_find_xy_l2242_224277

theorem find_xy (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) : 
  2^x - 5 = 11^y ↔ (x = 4 ∧ y = 1) :=
by sorry

end NUMINAMATH_GPT_find_xy_l2242_224277


namespace NUMINAMATH_GPT_num_emails_received_after_second_deletion_l2242_224294

-- Define the initial conditions and final question
variable (initialEmails : ℕ)    -- Initial number of emails
variable (deletedEmails1 : ℕ)   -- First batch of deleted emails
variable (receivedEmails1 : ℕ)  -- First batch of received emails
variable (deletedEmails2 : ℕ)   -- Second batch of deleted emails
variable (receivedEmails2 : ℕ)  -- Second batch of received emails
variable (receivedEmails3 : ℕ)  -- Third batch of received emails
variable (finalEmails : ℕ)      -- Final number of emails in the inbox

-- Conditions based on the problem description
axiom initialEmails_def : initialEmails = 0
axiom deletedEmails1_def : deletedEmails1 = 50
axiom receivedEmails1_def : receivedEmails1 = 15
axiom deletedEmails2_def : deletedEmails2 = 20
axiom receivedEmails3_def : receivedEmails3 = 10
axiom finalEmails_def : finalEmails = 30

-- Question: Prove that the number of emails received after the second deletion is 5
theorem num_emails_received_after_second_deletion : receivedEmails2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_num_emails_received_after_second_deletion_l2242_224294


namespace NUMINAMATH_GPT_magic_square_y_l2242_224221

theorem magic_square_y (a b c d e y : ℚ) (h1 : y - 61 = a) (h2 : 2 * y - 125 = b) 
    (h3 : y + 25 + 64 = 3 + (y - 61) + (2 * y - 125)) : y = 272 / 3 :=
by
  sorry

end NUMINAMATH_GPT_magic_square_y_l2242_224221


namespace NUMINAMATH_GPT_div_by_7_or_11_l2242_224249

theorem div_by_7_or_11 (z x y : ℕ) (hx : x < 1000) (hz : z = 1000 * y + x) (hdiv7 : (x - y) % 7 = 0 ∨ (x - y) % 11 = 0) :
  z % 7 = 0 ∨ z % 11 = 0 :=
by
  sorry

end NUMINAMATH_GPT_div_by_7_or_11_l2242_224249


namespace NUMINAMATH_GPT_candy_count_l2242_224266

variables (S M L : ℕ)

theorem candy_count :
  S + M + L = 110 ∧ S + L = 100 ∧ L = S + 20 → S = 40 ∧ M = 10 ∧ L = 60 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_candy_count_l2242_224266


namespace NUMINAMATH_GPT_ratio_of_quadratic_roots_l2242_224262

theorem ratio_of_quadratic_roots (a b c : ℝ) (h : 2 * b^2 = 9 * a * c) : 
  ∃ (x₁ x₂ : ℝ), (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ (x₁ / x₂ = 2) :=
sorry

end NUMINAMATH_GPT_ratio_of_quadratic_roots_l2242_224262


namespace NUMINAMATH_GPT_B_completion_time_l2242_224239

-- Definitions based on the conditions
def A_work : ℚ := 1 / 24
def B_work : ℚ := 1 / 16
def C_work : ℚ := 1 / 32  -- Since C takes twice the time as B, C_work = B_work / 2

-- Combined work rates based on the conditions
def combined_ABC_work := A_work + B_work + C_work
def combined_AB_work := A_work + B_work

-- Question: How long does B take to complete the job alone?
-- Answer: 16 days

theorem B_completion_time : 
  (combined_ABC_work = 1 / 8) ∧ 
  (combined_AB_work = 1 / 12) ∧ 
  (A_work = 1 / 24) ∧ 
  (C_work = B_work / 2) → 
  (1 / B_work = 16) := 
by 
  sorry

end NUMINAMATH_GPT_B_completion_time_l2242_224239


namespace NUMINAMATH_GPT_find_x_plus_y_l2242_224265

theorem find_x_plus_y (x y : ℝ) 
  (h1 : x + Real.cos y = 1005) 
  (h2 : x + 1005 * Real.sin y = 1003) 
  (h3 : π ≤ y ∧ y ≤ 3 * π / 2) : 
  x + y = 1005 + 3 * π / 2 :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l2242_224265


namespace NUMINAMATH_GPT_average_interest_rate_correct_l2242_224250

-- Constants representing the conditions
def totalInvestment : ℝ := 5000
def rateA : ℝ := 0.035
def rateB : ℝ := 0.07

-- The condition that return from investment at 7% is twice that at 3.5%
def return_condition (x : ℝ) : Prop := 0.07 * x = 2 * 0.035 * (5000 - x)

-- The average rate of interest formula
noncomputable def average_rate_of_interest (x : ℝ) : ℝ := 
  (0.07 * x + 0.035 * (5000 - x)) / 5000

-- The theorem to prove the average rate is 5.25%
theorem average_interest_rate_correct : ∃ (x : ℝ), return_condition x ∧ average_rate_of_interest x = 0.0525 := 
by
  sorry

end NUMINAMATH_GPT_average_interest_rate_correct_l2242_224250


namespace NUMINAMATH_GPT_total_face_value_of_notes_l2242_224223

theorem total_face_value_of_notes :
  let face_value := 5
  let number_of_notes := 440 * 10^6
  face_value * number_of_notes = 2200000000 := 
by
  sorry

end NUMINAMATH_GPT_total_face_value_of_notes_l2242_224223


namespace NUMINAMATH_GPT_tetrahedron_sphere_surface_area_l2242_224230

-- Define the conditions
variables (a : ℝ) (mid_AB_C : ℝ → Prop) (S : ℝ)
variables (h1 : a > 0)
variables (h2 : mid_AB_C a)
variables (h3 : S = 3 * Real.sqrt 2)

-- Theorem statement
theorem tetrahedron_sphere_surface_area (h1 : a = 2 * Real.sqrt 3) : 
  4 * Real.pi * ( (Real.sqrt 6 / 4) * a )^2 = 18 * Real.pi := by
  sorry

end NUMINAMATH_GPT_tetrahedron_sphere_surface_area_l2242_224230


namespace NUMINAMATH_GPT_crayons_left_l2242_224276

-- Define initial number of crayons and the number taken by Mary
def initial_crayons : ℝ := 7.5
def taken_crayons : ℝ := 2.25

-- Calculate remaining crayons
def remaining_crayons := initial_crayons - taken_crayons

-- Prove that the remaining crayons are 5.25
theorem crayons_left : remaining_crayons = 5.25 := by
  sorry

end NUMINAMATH_GPT_crayons_left_l2242_224276


namespace NUMINAMATH_GPT_union_complement_eq_univ_l2242_224283

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 7}

-- Define set M
def M : Set ℕ := {1, 3, 5, 7}

-- Define set N
def N : Set ℕ := {3, 5}

-- Define the complement of N with respect to U
def complement_U_N : Set ℕ := {1, 2, 4, 7}

-- Prove that U = M ∪ complement_U_N
theorem union_complement_eq_univ : U = M ∪ complement_U_N := 
sorry

end NUMINAMATH_GPT_union_complement_eq_univ_l2242_224283


namespace NUMINAMATH_GPT_probability_A_not_winning_l2242_224219

theorem probability_A_not_winning 
  (prob_draw : ℚ := 1/2)
  (prob_B_wins : ℚ := 1/3) : 
  (prob_draw + prob_B_wins) = 5 / 6 := 
by
  sorry

end NUMINAMATH_GPT_probability_A_not_winning_l2242_224219


namespace NUMINAMATH_GPT_updated_mean_166_l2242_224218

/-- The mean of 50 observations is 200. Later, it was found that there is a decrement of 34 
from each observation. Prove that the updated mean of the observations is 166. -/
theorem updated_mean_166
  (mean : ℝ) (n : ℕ) (decrement : ℝ) (updated_mean : ℝ)
  (h1 : mean = 200) (h2 : n = 50) (h3 : decrement = 34) (h4 : updated_mean = 166) :
  mean - (decrement * n) / n = updated_mean :=
by
  sorry

end NUMINAMATH_GPT_updated_mean_166_l2242_224218


namespace NUMINAMATH_GPT_sum_of_edges_l2242_224258

-- Define the number of edges for a triangle and a rectangle
def edges_triangle : Nat := 3
def edges_rectangle : Nat := 4

-- The theorem states that the sum of the edges of a triangle and a rectangle is 7
theorem sum_of_edges : edges_triangle + edges_rectangle = 7 := 
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_sum_of_edges_l2242_224258


namespace NUMINAMATH_GPT_domain_of_sqrt_tan_x_minus_sqrt_3_l2242_224252

noncomputable def domain_of_function : Set Real :=
  {x | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2}

theorem domain_of_sqrt_tan_x_minus_sqrt_3 :
  { x : Real | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2 } = domain_of_function :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_tan_x_minus_sqrt_3_l2242_224252


namespace NUMINAMATH_GPT_time_addition_and_sum_l2242_224217

noncomputable def time_after_addition (hours_1 minutes_1 seconds_1 hours_2 minutes_2 seconds_2 : ℕ) : (ℕ × ℕ × ℕ) :=
  let total_seconds := seconds_1 + seconds_2
  let extra_minutes := total_seconds / 60
  let result_seconds := total_seconds % 60
  let total_minutes := minutes_1 + minutes_2 + extra_minutes
  let extra_hours := total_minutes / 60
  let result_minutes := total_minutes % 60
  let total_hours := hours_1 + hours_2 + extra_hours
  let result_hours := total_hours % 12
  (result_hours, result_minutes, result_seconds)

theorem time_addition_and_sum :
  let current_hours := 3
  let current_minutes := 0
  let current_seconds := 0
  let add_hours := 300
  let add_minutes := 55
  let add_seconds := 30
  let (final_hours, final_minutes, final_seconds) := time_after_addition current_hours current_minutes current_seconds add_hours add_minutes add_seconds
  final_hours + final_minutes + final_seconds = 88 :=
by
  sorry

end NUMINAMATH_GPT_time_addition_and_sum_l2242_224217


namespace NUMINAMATH_GPT_sum_even_integers_l2242_224234

theorem sum_even_integers (sum_first_50_even : Nat) (sum_from_100_to_200 : Nat) : 
  sum_first_50_even = 2550 → sum_from_100_to_200 = 7550 :=
by
  sorry

end NUMINAMATH_GPT_sum_even_integers_l2242_224234


namespace NUMINAMATH_GPT_cube_root_of_neg_27_l2242_224231

theorem cube_root_of_neg_27 : ∃ y : ℝ, y^3 = -27 ∧ y = -3 := by
  sorry

end NUMINAMATH_GPT_cube_root_of_neg_27_l2242_224231


namespace NUMINAMATH_GPT_find_a5_l2242_224284

variable {a : ℕ → ℝ}  -- Define the sequence a(n)

-- Define the conditions of the problem
variable (a1_positive : ∀ n, a n > 0)
variable (geo_seq : ∀ n, a (n + 1) = a n * 2)
variable (condition : (a 3) * (a 11) = 16)

theorem find_a5 (a1_positive : ∀ n, a n > 0) (geo_seq : ∀ n, a (n + 1) = a n * 2)
(condition : (a 3) * (a 11) = 16) : a 5 = 1 := by
  sorry

end NUMINAMATH_GPT_find_a5_l2242_224284


namespace NUMINAMATH_GPT_pyramid_volume_l2242_224259

theorem pyramid_volume 
(EF FG QE : ℝ) 
(base_area : ℝ) 
(volume : ℝ)
(h1 : EF = 10)
(h2 : FG = 5)
(h3 : base_area = EF * FG)
(h4 : QE = 9)
(h5 : volume = (1 / 3) * base_area * QE) : 
volume = 150 :=
by
  simp [h1, h2, h3, h4, h5]
  sorry

end NUMINAMATH_GPT_pyramid_volume_l2242_224259


namespace NUMINAMATH_GPT_board_divisible_into_hexominos_l2242_224207

theorem board_divisible_into_hexominos {m n : ℕ} (h_m_gt_5 : m > 5) (h_n_gt_5 : n > 5) 
  (h_m_div_by_3 : m % 3 = 0) (h_n_div_by_4 : n % 4 = 0) : 
  (m * n) % 6 = 0 :=
by
  sorry

end NUMINAMATH_GPT_board_divisible_into_hexominos_l2242_224207


namespace NUMINAMATH_GPT_area_square_hypotenuse_l2242_224205

theorem area_square_hypotenuse 
(a : ℝ) 
(h1 : ∀ a: ℝ,  ∃ YZ: ℝ, YZ = a + 3) 
(h2: ∀ XY: ℝ, ∃ total_area: ℝ, XY^2 + XY * (XY + 3) + (2 * XY^2 + 6 * XY + 9) = 450) :
  ∃ XZ: ℝ, (2 * a^2 + 6 * a + 9 = XZ) → XZ = 201 := by
  sorry

end NUMINAMATH_GPT_area_square_hypotenuse_l2242_224205


namespace NUMINAMATH_GPT_perfect_square_expression_l2242_224200
open Real

theorem perfect_square_expression (x : ℝ) :
  (12.86 * 12.86 + 12.86 * x + 0.14 * 0.14 = (12.86 + 0.14)^2) → x = 0.28 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_expression_l2242_224200


namespace NUMINAMATH_GPT_exists_unique_i_l2242_224247

-- Let p be an odd prime number.
variable {p : ℕ} [Fact (Nat.Prime p)] (odd_prime : p % 2 = 1)

-- Let a be an integer in the sequence {2, 3, 4, ..., p-3, p-2}
variable (a : ℕ) (a_range : 2 ≤ a ∧ a ≤ p - 2)

-- Prove that there exists a unique i such that i * a ≡ 1 (mod p) and i ≠ a
theorem exists_unique_i (h1 : ∀ k, 1 ≤ k ∧ k ≤ p - 1 → Nat.gcd k p = 1) :
  ∃! (i : ℕ), 1 ≤ i ∧ i ≤ p - 1 ∧ i * a % p = 1 ∧ i ≠ a :=
by 
  sorry

end NUMINAMATH_GPT_exists_unique_i_l2242_224247


namespace NUMINAMATH_GPT_ellipse_foci_on_y_axis_l2242_224209

theorem ellipse_foci_on_y_axis (k : ℝ) (h1 : 5 + k > 3 - k) (h2 : 3 - k > 0) (h3 : 5 + k > 0) : -1 < k ∧ k < 3 :=
by 
  sorry

end NUMINAMATH_GPT_ellipse_foci_on_y_axis_l2242_224209


namespace NUMINAMATH_GPT_lowest_possible_number_of_students_l2242_224288

theorem lowest_possible_number_of_students :
  Nat.lcm 18 24 = 72 :=
by
  sorry

end NUMINAMATH_GPT_lowest_possible_number_of_students_l2242_224288


namespace NUMINAMATH_GPT_unique_triple_primes_l2242_224211

theorem unique_triple_primes (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) (h3 : (p^3 + q^3 + r^3) / (p + q + r) = 249) : r = 19 :=
sorry

end NUMINAMATH_GPT_unique_triple_primes_l2242_224211


namespace NUMINAMATH_GPT_tom_watching_days_l2242_224228

def show_a_season_1_time : Nat := 20 * 22
def show_a_season_2_time : Nat := 18 * 24
def show_a_season_3_time : Nat := 22 * 26
def show_a_season_4_time : Nat := 15 * 30

def show_b_season_1_time : Nat := 24 * 42
def show_b_season_2_time : Nat := 16 * 48
def show_b_season_3_time : Nat := 12 * 55

def show_c_season_1_time : Nat := 10 * 60
def show_c_season_2_time : Nat := 13 * 58
def show_c_season_3_time : Nat := 15 * 50
def show_c_season_4_time : Nat := 11 * 52
def show_c_season_5_time : Nat := 9 * 65

def show_a_total_time : Nat :=
  show_a_season_1_time + show_a_season_2_time +
  show_a_season_3_time + show_a_season_4_time

def show_b_total_time : Nat :=
  show_b_season_1_time + show_b_season_2_time + show_b_season_3_time

def show_c_total_time : Nat :=
  show_c_season_1_time + show_c_season_2_time +
  show_c_season_3_time + show_c_season_4_time +
  show_c_season_5_time

def total_time : Nat := show_a_total_time + show_b_total_time + show_c_total_time

def daily_watch_time : Nat := 120

theorem tom_watching_days : (total_time + daily_watch_time - 1) / daily_watch_time = 64 := sorry

end NUMINAMATH_GPT_tom_watching_days_l2242_224228


namespace NUMINAMATH_GPT_solve_for_nonzero_x_l2242_224235

open Real

theorem solve_for_nonzero_x (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 → x = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_nonzero_x_l2242_224235


namespace NUMINAMATH_GPT_melanie_cats_l2242_224251

theorem melanie_cats (jacob_cats : ℕ) (annie_cats : ℕ) (melanie_cats : ℕ) 
  (h_jacob : jacob_cats = 90)
  (h_annie : annie_cats = jacob_cats / 3)
  (h_melanie : melanie_cats = annie_cats * 2) :
  melanie_cats = 60 := by
  sorry

end NUMINAMATH_GPT_melanie_cats_l2242_224251


namespace NUMINAMATH_GPT_john_daily_reading_hours_l2242_224287

-- Definitions from the conditions
def reading_rate := 50  -- pages per hour
def total_pages := 2800  -- pages
def weeks := 4
def days_per_week := 7

-- Hypotheses derived from the conditions
def total_hours := total_pages / reading_rate  -- 2800 / 50 = 56 hours
def total_days := weeks * days_per_week  -- 4 * 7 = 28 days

-- Theorem to prove 
theorem john_daily_reading_hours : (total_hours / total_days) = 2 := by
  sorry

end NUMINAMATH_GPT_john_daily_reading_hours_l2242_224287


namespace NUMINAMATH_GPT_isosceles_triangle_no_obtuse_l2242_224268

theorem isosceles_triangle_no_obtuse (A B C : ℝ) 
  (h1 : A = 70) 
  (h2 : B = 70) 
  (h3 : A + B + C = 180) 
  (h_iso : A = B) 
  : (A ≤ 90) ∧ (B ≤ 90) ∧ (C ≤ 90) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_no_obtuse_l2242_224268


namespace NUMINAMATH_GPT_arthur_walks_total_distance_l2242_224255

theorem arthur_walks_total_distance :
  let east_blocks := 8
  let north_blocks := 10
  let west_blocks := 3
  let block_distance := 1 / 3
  let total_blocks := east_blocks + north_blocks + west_blocks
  let total_miles := total_blocks * block_distance
  total_miles = 7 :=
by
  sorry

end NUMINAMATH_GPT_arthur_walks_total_distance_l2242_224255


namespace NUMINAMATH_GPT_pizza_slices_l2242_224261

-- Definitions of conditions
def slices (H C : ℝ) : Prop :=
  (H / 2 - 3 + 2 * C / 3 = 11) ∧ (H = C)

-- Stating the theorem to prove
theorem pizza_slices (H C : ℝ) (h : slices H C) : H = 12 :=
sorry

end NUMINAMATH_GPT_pizza_slices_l2242_224261


namespace NUMINAMATH_GPT_younger_age_is_12_l2242_224208

theorem younger_age_is_12 
  (y elder : ℕ)
  (h_diff : elder = y + 20)
  (h_past : elder - 7 = 5 * (y - 7)) :
  y = 12 :=
by
  sorry

end NUMINAMATH_GPT_younger_age_is_12_l2242_224208


namespace NUMINAMATH_GPT_min_value_z_l2242_224244

variable {x y : ℝ}

def constraint1 (x y : ℝ) : Prop := x + y ≤ 3
def constraint2 (x y : ℝ) : Prop := x - y ≥ -1
def constraint3 (y : ℝ) : Prop := y ≥ 1

theorem min_value_z (x y : ℝ) 
  (h1 : constraint1 x y) 
  (h2 : constraint2 x y) 
  (h3 : constraint3 y) 
  (hx_pos : x > 0) 
  (hy_pos : y > 0) : 
  ∃ x y, x > 0 ∧ y ≥ 1 ∧ x + y ≤ 3 ∧ x - y ≥ -1 ∧ (∀ x' y', x' > 0 ∧ y' ≥ 1 ∧ x' + y' ≤ 3 ∧ x' - y' ≥ -1 → (y' / x' ≥ y / x)) ∧ y / x = 1 / 2 := 
sorry

end NUMINAMATH_GPT_min_value_z_l2242_224244


namespace NUMINAMATH_GPT_student_a_score_l2242_224298

def total_questions : ℕ := 100
def correct_responses : ℕ := 87
def incorrect_responses : ℕ := total_questions - correct_responses
def score : ℕ := correct_responses - 2 * incorrect_responses

theorem student_a_score : score = 61 := by
  unfold score
  unfold correct_responses
  unfold incorrect_responses
  norm_num
  -- At this point, the theorem is stated, but we insert sorry to satisfy the requirement of not providing the proof.
  sorry

end NUMINAMATH_GPT_student_a_score_l2242_224298


namespace NUMINAMATH_GPT_garden_dimensions_l2242_224245

variable {w l x : ℝ}

-- Definition of the problem conditions
def garden_length_eq_three_times_width (w l : ℝ) : Prop := l = 3 * w
def combined_area_eq (w x : ℝ) : Prop := (w + 2 * x) * (3 * w + 2 * x) = 432
def walkway_area_eq (w x : ℝ) : Prop := 8 * w * x + 4 * x^2 = 108

-- The main theorem statement
theorem garden_dimensions (w l x : ℝ)
  (h1 : garden_length_eq_three_times_width w l)
  (h2 : combined_area_eq w x)
  (h3 : walkway_area_eq w x) :
  w = 6 * Real.sqrt 3 ∧ l = 18 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_garden_dimensions_l2242_224245


namespace NUMINAMATH_GPT_sum_f_to_2017_l2242_224285

noncomputable def f (x : ℕ) : ℝ := Real.cos (x * Real.pi / 3)

theorem sum_f_to_2017 : (Finset.range 2017).sum f = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_f_to_2017_l2242_224285


namespace NUMINAMATH_GPT_divisibility_expression_l2242_224237

variable {R : Type*} [CommRing R] (x a b : R)

theorem divisibility_expression :
  ∃ k : R, (x + a + b) ^ 3 - x ^ 3 - a ^ 3 - b ^ 3 = (x + a) * (x + b) * k :=
sorry

end NUMINAMATH_GPT_divisibility_expression_l2242_224237


namespace NUMINAMATH_GPT_retailer_received_extra_boxes_l2242_224215
-- Necessary import for mathematical proofs

-- Define the conditions
def dozen_boxes := 12
def dozens_ordered := 3
def discount_percent := 25

-- Calculate the total boxes ordered and the discount factor
def total_boxes := dozen_boxes * dozens_ordered
def discount_factor := (100 - discount_percent) / 100

-- Define the number of boxes paid for and the extra boxes received
def paid_boxes := total_boxes * discount_factor
def extra_boxes := total_boxes - paid_boxes

-- Statement of the proof problem
theorem retailer_received_extra_boxes : extra_boxes = 9 :=
by
    -- This is the place where the proof would be written
    sorry

end NUMINAMATH_GPT_retailer_received_extra_boxes_l2242_224215


namespace NUMINAMATH_GPT_smallest_circle_equation_l2242_224293

theorem smallest_circle_equation :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ (x - 1)^2 + y^2 = 1 ∧ ((x - 1)^2 + y^2 = 1) = (x^2 + y^2 = 1) := 
sorry

end NUMINAMATH_GPT_smallest_circle_equation_l2242_224293


namespace NUMINAMATH_GPT_candy_bar_cost_correct_l2242_224278

noncomputable def candy_bar_cost : ℕ := 25 -- Correct answer from the solution

theorem candy_bar_cost_correct (C : ℤ) (H1 : 3 * C + 150 + 50 = 11 * 25)
  (H2 : ∃ C, C ≥ 0) : C = candy_bar_cost :=
by
  sorry

end NUMINAMATH_GPT_candy_bar_cost_correct_l2242_224278


namespace NUMINAMATH_GPT_fraction_division_l2242_224216

theorem fraction_division : (3 / 4) / (2 / 5) = 15 / 8 := 
by
  -- We need to convert this division into multiplication by the reciprocal
  -- (3 / 4) / (2 / 5) = (3 / 4) * (5 / 2)
  -- Now perform the multiplication of the numerators and denominators
  -- (3 * 5) / (4 * 2) = 15 / 8
  sorry

end NUMINAMATH_GPT_fraction_division_l2242_224216


namespace NUMINAMATH_GPT_range_of_m_l2242_224297

def isDistinctRealRootsInInterval (a b x : ℝ) : Prop :=
  a * x^2 + b * x + 4 = 0 ∧ 0 < x ∧ x ≤ 3

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ isDistinctRealRootsInInterval 1 (- (m + 1)) x ∧ isDistinctRealRootsInInterval 1 (- (m + 1)) y) ↔
  (3 < m ∧ m ≤ 10 / 3) :=
sorry

end NUMINAMATH_GPT_range_of_m_l2242_224297


namespace NUMINAMATH_GPT_statement_C_l2242_224270

theorem statement_C (x : ℝ) (h : x^2 < 4) : x < 2 := 
sorry

end NUMINAMATH_GPT_statement_C_l2242_224270


namespace NUMINAMATH_GPT_markov_coprime_squares_l2242_224233

def is_coprime (x y : ℕ) : Prop :=
Nat.gcd x y = 1

theorem markov_coprime_squares (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  x^2 + y^2 + z^2 = 3 * x * y * z →
  ∃ a b c: ℕ, (a, b, c) = (2, 1, 1) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (1, 1, 2) ∧ 
  (a ≠ 1 → ∃ p q : ℕ, is_coprime p q ∧ a = p^2 + q^2) :=
sorry

end NUMINAMATH_GPT_markov_coprime_squares_l2242_224233


namespace NUMINAMATH_GPT_solve_division_problem_l2242_224263

-- Problem Conditions
def division_problem : ℚ := 0.25 / 0.005

-- Proof Problem Statement
theorem solve_division_problem : division_problem = 50 := by
  sorry

end NUMINAMATH_GPT_solve_division_problem_l2242_224263


namespace NUMINAMATH_GPT_determine_top_5_median_required_l2242_224241

theorem determine_top_5_median_required (scores : Fin 9 → ℝ) (unique_scores : ∀ (i j : Fin 9), i ≠ j → scores i ≠ scores j) :
  ∃ median,
  (∀ (student_score : ℝ), 
    (student_score > median ↔ ∃ (idx_top : Fin 5), student_score = scores ⟨idx_top.1, sorry⟩)) :=
sorry

end NUMINAMATH_GPT_determine_top_5_median_required_l2242_224241


namespace NUMINAMATH_GPT_count_true_statements_l2242_224286

theorem count_true_statements (x : ℝ) (h : x > -3) :
  (if (x > -3 → x > -6) then 1 else 0) +
  (if (¬ (x > -3 → x > -6)) then 1 else 0) +
  (if (x > -6 → x > -3) then 1 else 0) +
  (if (¬ (x > -6 → x > -3)) then 1 else 0) = 2 :=
sorry

end NUMINAMATH_GPT_count_true_statements_l2242_224286


namespace NUMINAMATH_GPT_increasing_interval_l2242_224236

-- Given function definition
def quad_func (x : ℝ) : ℝ := -x^2 + 1

-- Property to be proven: The function is increasing on the interval (-∞, 0]
theorem increasing_interval : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → quad_func x < quad_func y := by
  sorry

end NUMINAMATH_GPT_increasing_interval_l2242_224236


namespace NUMINAMATH_GPT_xy_value_l2242_224274

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := 
by sorry

end NUMINAMATH_GPT_xy_value_l2242_224274


namespace NUMINAMATH_GPT_merchant_loss_is_15_yuan_l2242_224240

noncomputable def profit_cost_price : ℝ := (180 : ℝ) / 1.2
noncomputable def loss_cost_price : ℝ := (180 : ℝ) / 0.8

theorem merchant_loss_is_15_yuan :
  (180 + 180) - (profit_cost_price + loss_cost_price) = -15 := by
  sorry

end NUMINAMATH_GPT_merchant_loss_is_15_yuan_l2242_224240


namespace NUMINAMATH_GPT_min_straight_line_cuts_l2242_224257

theorem min_straight_line_cuts (can_overlap : Prop) : 
  ∃ (cuts : ℕ), cuts = 4 ∧ 
  (∀ (square : ℕ), square = 3 →
   ∀ (unit : ℕ), unit = 1 → 
   ∀ (divided : Prop), divided = True → 
   (unit * unit) * 9 = (square * square)) :=
by
  sorry

end NUMINAMATH_GPT_min_straight_line_cuts_l2242_224257


namespace NUMINAMATH_GPT_totalAttendees_l2242_224260

def numberOfBuses : ℕ := 8
def studentsPerBus : ℕ := 45
def chaperonesList : List ℕ := [2, 3, 4, 5, 3, 4, 2, 6]

theorem totalAttendees : 
    numberOfBuses * studentsPerBus + chaperonesList.sum = 389 := 
by
  sorry

end NUMINAMATH_GPT_totalAttendees_l2242_224260


namespace NUMINAMATH_GPT_reflection_identity_l2242_224204

-- Define the reflection function
def reflect (O P : ℝ × ℝ) : ℝ × ℝ := (2 * O.1 - P.1, 2 * O.2 - P.2)

-- Given three points and a point P
variables (O1 O2 O3 P : ℝ × ℝ)

-- Define the sequence of reflections
def sequence_reflection (P : ℝ × ℝ) : ℝ × ℝ :=
  reflect O3 (reflect O2 (reflect O1 P))

-- Lean 4 statement to prove the mathematical theorem
theorem reflection_identity :
  sequence_reflection O1 O2 O3 (sequence_reflection O1 O2 O3 P) = P :=
by sorry

end NUMINAMATH_GPT_reflection_identity_l2242_224204


namespace NUMINAMATH_GPT_isosceles_triangle_inequality_l2242_224273

theorem isosceles_triangle_inequality
  (a b : ℝ)
  (hb : b > 0)
  (h₁₂ : 12 * (π / 180) = π / 15) 
  (h_sin6 : Real.sin (6 * (π / 180)) > 1 / 10)
  (h_eq : a = 2 * b * Real.sin (6 * (π / 180))) : 
  b < 5 * a := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_inequality_l2242_224273


namespace NUMINAMATH_GPT_evaluate_expression_at_two_l2242_224220

theorem evaluate_expression_at_two: 
  (3 * 2^2 - 4 * 2 + 2) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_two_l2242_224220


namespace NUMINAMATH_GPT_pupils_like_only_maths_l2242_224267

noncomputable def number_pupils_like_only_maths (total: ℕ) (maths_lovers: ℕ) (english_lovers: ℕ) 
(neither_lovers: ℕ) (both_lovers: ℕ) : ℕ :=
maths_lovers - both_lovers

theorem pupils_like_only_maths : 
∀ (total: ℕ) (maths_lovers: ℕ) (english_lovers: ℕ) (neither_lovers: ℕ) (both_lovers: ℕ),
total = 30 →
maths_lovers = 20 →
english_lovers = 18 →
both_lovers = 2 * neither_lovers →
neither_lovers + maths_lovers + english_lovers - both_lovers - both_lovers = total →
number_pupils_like_only_maths total maths_lovers english_lovers neither_lovers both_lovers = 4 :=
by
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_pupils_like_only_maths_l2242_224267


namespace NUMINAMATH_GPT_total_children_on_playground_l2242_224282

theorem total_children_on_playground (girls boys : ℕ) (h_girls : girls = 28) (h_boys : boys = 35) : girls + boys = 63 := 
by 
  sorry

end NUMINAMATH_GPT_total_children_on_playground_l2242_224282


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2242_224275

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : a ≠ b) (h4 : a + b > b) (h5 : a + b > a) 
: ∃ p : ℝ, p = 10 :=
by
  -- Using the given conditions to determine the perimeter
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2242_224275


namespace NUMINAMATH_GPT_jill_travel_time_to_school_is_20_minutes_l2242_224295

variables (dave_rate : ℕ) (dave_step : ℕ) (dave_time : ℕ)
variables (jill_rate : ℕ) (jill_step : ℕ)

def dave_distance : ℕ := dave_rate * dave_step * dave_time
def jill_time_to_school : ℕ := dave_distance dave_rate dave_step dave_time / (jill_rate * jill_step)

theorem jill_travel_time_to_school_is_20_minutes : 
  dave_rate = 85 → dave_step = 80 → dave_time = 18 → 
  jill_rate = 120 → jill_step = 50 → jill_time_to_school 85 80 18 120 50 = 20 :=
by
  intros
  unfold jill_time_to_school
  unfold dave_distance
  sorry

end NUMINAMATH_GPT_jill_travel_time_to_school_is_20_minutes_l2242_224295


namespace NUMINAMATH_GPT_power_half_mod_prime_l2242_224248

-- Definitions of odd prime and coprime condition
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1
def coprime (a p : ℕ) : Prop := Nat.gcd a p = 1

-- Main statement
theorem power_half_mod_prime (p a : ℕ) (hp : is_odd_prime p) (ha : coprime a p) :
  a ^ ((p - 1) / 2) % p = 1 ∨ a ^ ((p - 1) / 2) % p = p - 1 := 
  sorry

end NUMINAMATH_GPT_power_half_mod_prime_l2242_224248


namespace NUMINAMATH_GPT_sum_of_1_to_17_is_odd_l2242_224238

-- Define the set of natural numbers from 1 to 17
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

-- Proof that the sum of these numbers is odd
theorem sum_of_1_to_17_is_odd : (List.sum nums) % 2 = 1 := 
by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_sum_of_1_to_17_is_odd_l2242_224238


namespace NUMINAMATH_GPT_average_weight_increase_l2242_224222

theorem average_weight_increase (W_new : ℝ) (W_old : ℝ) (num_persons : ℝ): 
  W_new = 94 ∧ W_old = 70 ∧ num_persons = 8 → 
  (W_new - W_old) / num_persons = 3 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_average_weight_increase_l2242_224222


namespace NUMINAMATH_GPT_max_sum_composite_shape_l2242_224264

theorem max_sum_composite_shape :
  let faces_hex_prism := 8
  let edges_hex_prism := 18
  let vertices_hex_prism := 12

  let faces_hex_with_pyramid := 8 - 1 + 6
  let edges_hex_with_pyramid := 18 + 6
  let vertices_hex_with_pyramid := 12 + 1
  let sum_hex_with_pyramid := faces_hex_with_pyramid + edges_hex_with_pyramid + vertices_hex_with_pyramid

  let faces_rec_with_pyramid := 8 - 1 + 5
  let edges_rec_with_pyramid := 18 + 4
  let vertices_rec_with_pyramid := 12 + 1
  let sum_rec_with_pyramid := faces_rec_with_pyramid + edges_rec_with_pyramid + vertices_rec_with_pyramid

  sum_hex_with_pyramid = 50 ∧ sum_rec_with_pyramid = 46 ∧ sum_hex_with_pyramid ≥ sum_rec_with_pyramid := 
by
  have faces_hex_prism := 8
  have edges_hex_prism := 18
  have vertices_hex_prism := 12

  have faces_hex_with_pyramid := 8 - 1 + 6
  have edges_hex_with_pyramid := 18 + 6
  have vertices_hex_with_pyramid := 12 + 1
  have sum_hex_with_pyramid := faces_hex_with_pyramid + edges_hex_with_pyramid + vertices_hex_with_pyramid

  have faces_rec_with_pyramid := 8 - 1 + 5
  have edges_rec_with_pyramid := 18 + 4
  have vertices_rec_with_pyramid := 12 + 1
  have sum_rec_with_pyramid := faces_rec_with_pyramid + edges_rec_with_pyramid + vertices_rec_with_pyramid

  sorry -- proof omitted

end NUMINAMATH_GPT_max_sum_composite_shape_l2242_224264


namespace NUMINAMATH_GPT_exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum_l2242_224289

theorem exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum :
  ∃ (a b c : ℤ), 2 * (a * b + b * c + c * a) = 4 * (a + b + c) :=
by
  -- Here we prove the existence of such integers a, b, c, which is stated in the theorem
  sorry

end NUMINAMATH_GPT_exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum_l2242_224289


namespace NUMINAMATH_GPT_root_inverse_cubes_l2242_224226

theorem root_inverse_cubes (a b c r s : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) :
  (1 / r^3) + (1 / s^3) = (-b^3 + 3 * a * b * c) / c^3 :=
by
  sorry

end NUMINAMATH_GPT_root_inverse_cubes_l2242_224226


namespace NUMINAMATH_GPT_gain_percent_is_40_l2242_224272

-- Define the conditions
def purchase_price : ℕ := 800
def repair_costs : ℕ := 200
def selling_price : ℕ := 1400

-- Define the total cost
def total_cost : ℕ := purchase_price + repair_costs

-- Define the gain
def gain : ℕ := selling_price - total_cost

-- Define the gain percent
def gain_percent : ℕ := (gain * 100) / total_cost

theorem gain_percent_is_40 : gain_percent = 40 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_gain_percent_is_40_l2242_224272


namespace NUMINAMATH_GPT_no_perfect_square_in_range_l2242_224290

def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem no_perfect_square_in_range :
  ∀ (n : ℕ), 4 ≤ n ∧ n ≤ 12 → ¬ isPerfectSquare (2*n*n + 3*n + 2) :=
by
  intro n
  intro h
  sorry

end NUMINAMATH_GPT_no_perfect_square_in_range_l2242_224290


namespace NUMINAMATH_GPT_bottle_caps_cost_l2242_224253

-- Conditions
def cost_per_bottle_cap : ℕ := 2
def number_of_bottle_caps : ℕ := 6

-- Statement of the problem
theorem bottle_caps_cost : (cost_per_bottle_cap * number_of_bottle_caps) = 12 :=
by
  sorry

end NUMINAMATH_GPT_bottle_caps_cost_l2242_224253


namespace NUMINAMATH_GPT_find_k_l2242_224203

theorem find_k (x y z k : ℝ) 
  (h1 : 9 / (x + y) = k / (x + 2 * z)) 
  (h2 : 9 / (x + y) = 14 / (z - y)) 
  (h3 : y = 2 * x) 
  (h4 : x + z = 10) :
  k = 46 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2242_224203


namespace NUMINAMATH_GPT_stock_worth_l2242_224227

theorem stock_worth (W : Real) 
  (profit_part : Real := 0.25 * W * 0.20)
  (loss_part1 : Real := 0.35 * W * 0.10)
  (loss_part2 : Real := 0.40 * W * 0.15)
  (overall_loss_eq : loss_part1 + loss_part2 - profit_part = 1200) : 
  W = 26666.67 :=
by
  sorry

end NUMINAMATH_GPT_stock_worth_l2242_224227


namespace NUMINAMATH_GPT_johns_running_hours_l2242_224243

-- Define the conditions
variable (x : ℕ) -- let x represent the number of hours at 8 mph and 6 mph
variable (total_hours : ℕ) (total_distance : ℕ)
variable (speed_8 : ℕ) (speed_6 : ℕ) (speed_5 : ℕ)
variable (distance_8 : ℕ := speed_8 * x)
variable (distance_6 : ℕ := speed_6 * x)
variable (distance_5 : ℕ := speed_5 * (total_hours - 2 * x))

-- Total hours John completes the marathon
axiom h1: total_hours = 15

-- Total distance John completes in miles
axiom h2: total_distance = 95

-- Speed factors
axiom h3: speed_8 = 8
axiom h4: speed_6 = 6
axiom h5: speed_5 = 5

-- Distance equation
axiom h6: distance_8 + distance_6 + distance_5 = total_distance

-- Prove the number of hours John ran at each speed
theorem johns_running_hours : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_johns_running_hours_l2242_224243


namespace NUMINAMATH_GPT_second_group_students_l2242_224224

-- Define the number of groups and their respective sizes
def num_groups : ℕ := 4
def first_group_students : ℕ := 5
def third_group_students : ℕ := 7
def fourth_group_students : ℕ := 4
def total_students : ℕ := 24

-- Define the main theorem to prove
theorem second_group_students :
  (∃ second_group_students : ℕ,
    total_students = first_group_students + second_group_students + third_group_students + fourth_group_students ∧
    second_group_students = 8) :=
sorry

end NUMINAMATH_GPT_second_group_students_l2242_224224


namespace NUMINAMATH_GPT_quadratic_equation_factored_form_correct_l2242_224214

theorem quadratic_equation_factored_form_correct :
  ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_quadratic_equation_factored_form_correct_l2242_224214


namespace NUMINAMATH_GPT_delta_y_over_delta_x_l2242_224269

variable (Δx : ℝ)

def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem delta_y_over_delta_x : (f (1 + Δx) - f 1) / Δx = 4 + 2 * Δx :=
by
  sorry

end NUMINAMATH_GPT_delta_y_over_delta_x_l2242_224269


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2242_224213

-- Let's define the conditions and the theorem to be proved in Lean 4
theorem sufficient_but_not_necessary : ∀ x : ℝ, (x > 1 → x > 0) ∧ ¬(∀ x : ℝ, x > 0 → x > 1) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2242_224213


namespace NUMINAMATH_GPT_gcd_180_270_450_l2242_224281

theorem gcd_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 := by 
  sorry

end NUMINAMATH_GPT_gcd_180_270_450_l2242_224281


namespace NUMINAMATH_GPT_quadratic_equation_roots_l2242_224254

theorem quadratic_equation_roots (a b k k1 k2 : ℚ)
  (h_roots : ∀ x : ℚ, k * (x^2 - x) + x + 2 = 0)
  (h_ab_condition : (a / b) + (b / a) = 3 / 7)
  (h_k_values : ∀ x : ℚ, 7 * x^2 - 20 * x - 21 = 0)
  (h_k1k2 : k1 + k2 = 20 / 7)
  (h_k1k2_prod : k1 * k2 = -21 / 7) :
  (k1 / k2) + (k2 / k1) = -104 / 21 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_roots_l2242_224254


namespace NUMINAMATH_GPT_unique_sum_of_cubes_lt_1000_l2242_224212

theorem unique_sum_of_cubes_lt_1000 : 
  let max_cube := 11 
  let max_val := 1000 
  ∃ n : ℕ, n = 35 ∧ ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ max_cube → 1 ≤ b ∧ b ≤ max_cube → a^3 + b^3 < max_val :=
sorry

end NUMINAMATH_GPT_unique_sum_of_cubes_lt_1000_l2242_224212


namespace NUMINAMATH_GPT_plane_equation_parallel_to_Oz_l2242_224279

theorem plane_equation_parallel_to_Oz (A B D : ℝ)
  (h1 : A * 1 + B * 0 + D = 0)
  (h2 : A * (-2) + B * 1 + D = 0)
  (h3 : ∀ z : ℝ, exists c : ℝ, A * z + B * c + D = 0):
  A = 1 ∧ B = 3 ∧ D = -1 :=
  by
  sorry

end NUMINAMATH_GPT_plane_equation_parallel_to_Oz_l2242_224279


namespace NUMINAMATH_GPT_simplified_expression_eq_l2242_224246

noncomputable def simplify_expression : ℚ :=
  1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)

-- We need to prove that the simplifed expression is equal to 1 / 39
theorem simplified_expression_eq : simplify_expression = 1 / 39 :=
by sorry

end NUMINAMATH_GPT_simplified_expression_eq_l2242_224246


namespace NUMINAMATH_GPT_goods_train_speed_l2242_224256

theorem goods_train_speed
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_taken : ℝ)
  (speed_kmph : ℝ)
  (h1 : length_train = 240.0416)
  (h2 : length_platform = 280)
  (h3 : time_taken = 26)
  (h4 : speed_kmph = 72.00576) :
  speed_kmph = ((length_train + length_platform) / time_taken) * 3.6 := sorry

end NUMINAMATH_GPT_goods_train_speed_l2242_224256


namespace NUMINAMATH_GPT_hexagonal_prism_sum_maximum_l2242_224210

noncomputable def hexagonal_prism_max_sum (h_u h_v h_w h_x h_y h_z : ℕ) (u v w x y z : ℝ) : ℝ :=
  u + v + w + x + y + z

def max_sum_possible (h_u h_v h_w h_x h_y h_z : ℕ) : ℝ :=
  if h_u = 4 ∧ h_v = 7 ∧ h_w = 10 ∨
     h_u = 4 ∧ h_x = 7 ∧ h_y = 10 ∨
     h_u = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_v = 4 ∧ h_x = 7 ∧ h_w = 10 ∨
     h_v = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_w = 4 ∧ h_x = 7 ∧ h_z = 10
  then 78
  else 0

theorem hexagonal_prism_sum_maximum (h_u h_v h_w h_x h_y h_z : ℕ) :
  max_sum_possible h_u h_v h_w h_x h_y h_z = 78 → ∃ (u v w x y z : ℝ), hexagonal_prism_max_sum h_u h_v h_w h_x h_y h_z u v w x y z = 78 := 
by 
  sorry

end NUMINAMATH_GPT_hexagonal_prism_sum_maximum_l2242_224210


namespace NUMINAMATH_GPT_ferns_have_1260_leaves_l2242_224280

def num_ferns : ℕ := 6
def fronds_per_fern : ℕ := 7
def leaves_per_frond : ℕ := 30
def total_leaves : ℕ := num_ferns * fronds_per_fern * leaves_per_frond

theorem ferns_have_1260_leaves : total_leaves = 1260 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ferns_have_1260_leaves_l2242_224280


namespace NUMINAMATH_GPT_difference_rabbits_antelopes_l2242_224201

variable (A R H W L : ℕ)
variable (x : ℕ)

def antelopes := 80
def rabbits := antelopes + x
def hyenas := (antelopes + rabbits) - 42
def wild_dogs := hyenas + 50
def leopards := rabbits / 2
def total_animals := 605

theorem difference_rabbits_antelopes
  (h1 : antelopes = 80)
  (h2 : rabbits = antelopes + x)
  (h3 : hyenas = (antelopes + rabbits) - 42)
  (h4 : wild_dogs = hyenas + 50)
  (h5 : leopards = rabbits / 2)
  (h6 : antelopes + rabbits + hyenas + wild_dogs + leopards = total_animals) : rabbits - antelopes = 70 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_difference_rabbits_antelopes_l2242_224201


namespace NUMINAMATH_GPT_Shiela_drawings_l2242_224271

theorem Shiela_drawings (n_neighbors : ℕ) (drawings_per_neighbor : ℕ) (total_drawings : ℕ) 
    (h1 : n_neighbors = 6) (h2 : drawings_per_neighbor = 9) : total_drawings = 54 :=
by 
  sorry

end NUMINAMATH_GPT_Shiela_drawings_l2242_224271


namespace NUMINAMATH_GPT_a_minus_b_value_l2242_224299

theorem a_minus_b_value (a b : ℤ) :
  (∀ x : ℝ, 9 * x^3 + y^2 + a * x - b * x^3 + x + 5 = y^2 + 5) → a - b = -10 :=
by
  sorry

end NUMINAMATH_GPT_a_minus_b_value_l2242_224299


namespace NUMINAMATH_GPT_main_theorem_l2242_224225

-- Define the distribution
def P0 : ℝ := 0.4
def P2 : ℝ := 0.4
def P1 (p : ℝ) : ℝ := p

-- Define a hypothesis that the sum of probabilities is 1
def prob_sum_eq_one (p : ℝ) : Prop := P0 + P1 p + P2 = 1

-- Define the expected value of X
def E_X (p : ℝ) : ℝ := 0 * P0 + 1 * P1 p + 2 * P2

-- Define variance computation
def variance (p : ℝ) : ℝ := P0 * (0 - E_X p) ^ 2 + P1 p * (1 - E_X p) ^ 2 + P2 * (2 - E_X p) ^ 2

-- State the main theorem
theorem main_theorem : (∃ p : ℝ, prob_sum_eq_one p) ∧ variance 0.2 = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_main_theorem_l2242_224225


namespace NUMINAMATH_GPT_haley_stickers_l2242_224292

theorem haley_stickers (friends : ℕ) (stickers_per_friend : ℕ) (total_stickers : ℕ) :
  friends = 9 → stickers_per_friend = 8 → total_stickers = friends * stickers_per_friend → total_stickers = 72 :=
by
  intros h_friends h_stickers_per_friend h_total_stickers
  rw [h_friends, h_stickers_per_friend] at h_total_stickers
  exact h_total_stickers

end NUMINAMATH_GPT_haley_stickers_l2242_224292


namespace NUMINAMATH_GPT_math_problem_l2242_224232

theorem math_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9 * x + y - x * y = 0) : 
  ((9 * x + y) * (9 / y + 1 / x) = x * y) ∧ ¬ ((x / 9) + y = 10) ∧ 
  ((x + y = 16) ↔ (x = 4 ∧ y = 12)) ∧ 
  ((x * y = 36) ↔ (x = 2 ∧ y = 18)) :=
by {
  sorry
}

end NUMINAMATH_GPT_math_problem_l2242_224232


namespace NUMINAMATH_GPT_divisible_by_11_of_sum_divisible_l2242_224202

open Int

theorem divisible_by_11_of_sum_divisible (a b : ℤ) (h : 11 ∣ (a^2 + b^2)) : 11 ∣ a ∧ 11 ∣ b :=
sorry

end NUMINAMATH_GPT_divisible_by_11_of_sum_divisible_l2242_224202
