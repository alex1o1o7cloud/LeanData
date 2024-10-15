import Mathlib

namespace NUMINAMATH_GPT_determinant_value_l2273_227381

theorem determinant_value (t₁ t₂ : ℤ)
    (h₁ : t₁ = 2 * 3 + 3 * 5)
    (h₂ : t₂ = 5) :
    Matrix.det ![
      ![1, -1, t₁],
      ![0, 1, -1],
      ![-1, t₂, -6]
    ] = 14 := by
  rw [h₁, h₂]
  -- Actual proof would go here
  sorry

end NUMINAMATH_GPT_determinant_value_l2273_227381


namespace NUMINAMATH_GPT_parallel_segments_l2273_227324

structure Point2D where
  x : Int
  y : Int

def vector (P Q : Point2D) : Point2D :=
  { x := Q.x - P.x, y := Q.y - P.y }

def is_parallel (v1 v2 : Point2D) : Prop :=
  ∃ k : Int, v2.x = k * v1.x ∧ v2.y = k * v1.y 

theorem parallel_segments :
  let A := { x := 1, y := 3 }
  let B := { x := 2, y := -1 }
  let C := { x := 0, y := 4 }
  let D := { x := 2, y := -4 }
  is_parallel (vector A B) (vector C D) := 
  sorry

end NUMINAMATH_GPT_parallel_segments_l2273_227324


namespace NUMINAMATH_GPT_train_speeds_l2273_227383

theorem train_speeds (v t : ℕ) (h1 : t = 1)
  (h2 : v + v * t = 90)
  (h3 : 90 * t = 90) :
  v = 45 := by
  sorry

end NUMINAMATH_GPT_train_speeds_l2273_227383


namespace NUMINAMATH_GPT_find_n_plus_c_l2273_227315

variables (n c : ℝ)

-- Conditions from the problem
def line1 := ∀ (x y : ℝ), (x = 4) → (y = 11) → (y = n * x + 3)
def line2 := ∀ (x y : ℝ), (x = 4) → (y = 11) → (y = 5 * x + c)

theorem find_n_plus_c (h1 : line1 n)
                      (h2 : line2 c) :
  n + c = -7 := by
  sorry

end NUMINAMATH_GPT_find_n_plus_c_l2273_227315


namespace NUMINAMATH_GPT_cos_equivalent_l2273_227316

open Real

theorem cos_equivalent (alpha : ℝ) (h : sin (π / 3 + alpha) = 1 / 3) : 
  cos (5 * π / 6 + alpha) = -1 / 3 :=
sorry

end NUMINAMATH_GPT_cos_equivalent_l2273_227316


namespace NUMINAMATH_GPT_find_pairs_l2273_227394

theorem find_pairs (p n : ℕ) (hp : Nat.Prime p) (h1 : n ≤ 2 * p) (h2 : n^(p-1) ∣ (p-1)^n + 1) : 
    (p = 2 ∧ n = 2) ∨ (p = 3 ∧ n = 3) ∨ (n = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l2273_227394


namespace NUMINAMATH_GPT_range_of_a_l2273_227370

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ Real.pi / 2 →
    (x + 3 + 2 * Real.sin θ * Real.cos θ) ^ 2 +
    (x + a * Real.sin θ + a * Real.cos θ) ^ 2 ≥ 1 / 8) ↔
  (a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2273_227370


namespace NUMINAMATH_GPT_sequence_term_and_k_value_l2273_227373

/-- Given a sequence {a_n} whose sum of the first n terms is S_n = n^2 - 9n,
    prove the sequence term a_n = 2n - 10, and if 5 < a_k < 8, then k = 8. -/
theorem sequence_term_and_k_value (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 9 * n) :
  (∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) →
  (∀ n, a n = 2 * n - 10) ∧ (∀ k, 5 < a k ∧ a k < 8 → k = 8) :=
by {
  -- Given S_n = n^2 - 9n, we need to show a_n = 2n - 10 and verify when 5 < a_k < 8, then k = 8
  sorry
}

end NUMINAMATH_GPT_sequence_term_and_k_value_l2273_227373


namespace NUMINAMATH_GPT_three_digit_diff_l2273_227379

theorem three_digit_diff (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) :
  ∃ d : ℕ, d = a - b ∧ (d < 10 ∨ (10 ≤ d ∧ d < 100) ∨ (100 ≤ d ∧ d < 1000)) :=
sorry

end NUMINAMATH_GPT_three_digit_diff_l2273_227379


namespace NUMINAMATH_GPT_problem1_simplified_problem2_simplified_l2273_227390

-- Definition and statement for the first problem
def problem1_expression (x y : ℝ) : ℝ := 
  -3 * x * y - 3 * x^2 + 4 * x * y + 2 * x^2

theorem problem1_simplified (x y : ℝ) : 
  problem1_expression x y = x * y - x^2 := 
by
  sorry

-- Definition and statement for the second problem
def problem2_expression (a b : ℝ) : ℝ := 
  3 * (a^2 - 2 * a * b) - 5 * (a^2 + 4 * a * b)

theorem problem2_simplified (a b : ℝ) : 
  problem2_expression a b = -2 * a^2 - 26 * a * b :=
by
  sorry

end NUMINAMATH_GPT_problem1_simplified_problem2_simplified_l2273_227390


namespace NUMINAMATH_GPT_difference_between_numbers_l2273_227341

theorem difference_between_numbers (x y : ℕ) (h : x - y = 9) :
  (10 * x + y) - (10 * y + x) = 81 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l2273_227341


namespace NUMINAMATH_GPT_probability_one_painted_face_and_none_painted_l2273_227368

-- Define the total number of smaller unit cubes
def total_cubes : ℕ := 125

-- Define the number of cubes with exactly one painted face
def one_painted_face : ℕ := 25

-- Define the number of cubes with no painted faces
def no_painted_faces : ℕ := 125 - 25 - 12

-- Define the total number of ways to select two cubes uniformly at random
def total_pairs : ℕ := (total_cubes * (total_cubes - 1)) / 2

-- Define the number of successful outcomes
def successful_outcomes : ℕ := one_painted_face * no_painted_faces

-- Define the sought probability
def desired_probability : ℚ := (successful_outcomes : ℚ) / (total_pairs : ℚ)

-- Lean statement to prove the probability
theorem probability_one_painted_face_and_none_painted :
  desired_probability = 44 / 155 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_painted_face_and_none_painted_l2273_227368


namespace NUMINAMATH_GPT_joshua_finishes_after_malcolm_l2273_227378

def time_difference_between_runners
  (race_length : ℕ)
  (malcolm_speed : ℕ)
  (joshua_speed : ℕ)
  (malcolm_finish_time : ℕ := malcolm_speed * race_length)
  (joshua_finish_time : ℕ := joshua_speed * race_length) : ℕ :=
joshua_finish_time - malcolm_finish_time

theorem joshua_finishes_after_malcolm
  (race_length : ℕ)
  (malcolm_speed : ℕ)
  (joshua_speed : ℕ)
  (h_race_length : race_length = 12)
  (h_malcolm_speed : malcolm_speed = 7)
  (h_joshua_speed : joshua_speed = 9) : time_difference_between_runners race_length malcolm_speed joshua_speed = 24 :=
by 
  subst h_race_length
  subst h_malcolm_speed
  subst h_joshua_speed
  rfl

#print joshua_finishes_after_malcolm

end NUMINAMATH_GPT_joshua_finishes_after_malcolm_l2273_227378


namespace NUMINAMATH_GPT_slope_of_asymptotes_l2273_227311

theorem slope_of_asymptotes (a b : ℝ) (h : a^2 = 144) (k : b^2 = 81) : (b / a = 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_slope_of_asymptotes_l2273_227311


namespace NUMINAMATH_GPT_area_of_rectangular_field_l2273_227332

-- Definitions from conditions
def L : ℕ := 20
def total_fencing : ℕ := 32

-- Additional variables inferred from the conditions
def W : ℕ := (total_fencing - L) / 2

-- The theorem statement
theorem area_of_rectangular_field : L * W = 120 :=
by
  -- Definitions and substitutions are included in the theorem proof
  sorry

end NUMINAMATH_GPT_area_of_rectangular_field_l2273_227332


namespace NUMINAMATH_GPT_room_length_l2273_227346

-- Defining conditions
def room_height : ℝ := 5
def room_width : ℝ := 7
def door_height : ℝ := 3
def door_width : ℝ := 1
def num_doors : ℝ := 2
def window1_height : ℝ := 1.5
def window1_width : ℝ := 2
def window2_height : ℝ := 1.5
def window2_width : ℝ := 1
def num_window2 : ℝ := 2
def paint_cost_per_sq_m : ℝ := 3
def total_paint_cost : ℝ := 474

-- Defining the problem as a statement to prove x (room length) is 10 meters
theorem room_length {x : ℝ} 
  (H1 : total_paint_cost = paint_cost_per_sq_m * ((2 * (x * room_height) + 2 * (room_width * room_height)) - (num_doors * (door_height * door_width) + (window1_height * window1_width) + num_window2 * (window2_height * window2_width)))) 
  : x = 10 :=
by 
  sorry

end NUMINAMATH_GPT_room_length_l2273_227346


namespace NUMINAMATH_GPT_simplify_expression_l2273_227313

theorem simplify_expression : 
    1 - 1 / (1 + Real.sqrt (2 + Real.sqrt 3)) + 1 / (1 - Real.sqrt (2 - Real.sqrt 3)) 
    = 1 + (Real.sqrt (2 - Real.sqrt 3) + Real.sqrt (2 + Real.sqrt 3)) / (-1 - Real.sqrt 3) := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2273_227313


namespace NUMINAMATH_GPT_minimum_value_x_squared_plus_12x_plus_5_l2273_227306

theorem minimum_value_x_squared_plus_12x_plus_5 : ∃ x : ℝ, x^2 + 12 * x + 5 = -31 :=
by sorry

end NUMINAMATH_GPT_minimum_value_x_squared_plus_12x_plus_5_l2273_227306


namespace NUMINAMATH_GPT_only_pair_2_2_satisfies_l2273_227369

theorem only_pair_2_2_satisfies :
  ∀ a b : ℕ, (∀ n : ℕ, ∃ c : ℕ, a ^ n + b ^ n = c ^ (n + 1)) → (a = 2 ∧ b = 2) :=
by sorry

end NUMINAMATH_GPT_only_pair_2_2_satisfies_l2273_227369


namespace NUMINAMATH_GPT_quadratic_root_m_l2273_227337

theorem quadratic_root_m (m : ℝ) : (∃ x : ℝ, x^2 + x + m^2 - 1 = 0 ∧ x = 0) → (m = 1 ∨ m = -1) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_root_m_l2273_227337


namespace NUMINAMATH_GPT_sequence_conjecture_l2273_227356

theorem sequence_conjecture (a : ℕ → ℝ) (h₁ : a 1 = 7)
  (h₂ : ∀ n, a (n + 1) = 7 * a n / (a n + 7)) :
  ∀ n, a n = 7 / n :=
by
  sorry

end NUMINAMATH_GPT_sequence_conjecture_l2273_227356


namespace NUMINAMATH_GPT_eugene_initial_pencils_l2273_227353

theorem eugene_initial_pencils (e given left : ℕ) (h1 : given = 6) (h2 : left = 45) (h3 : e = given + left) : e = 51 := by
  sorry

end NUMINAMATH_GPT_eugene_initial_pencils_l2273_227353


namespace NUMINAMATH_GPT_probability_f_leq_zero_l2273_227300

noncomputable def f (k x : ℝ) : ℝ := k * x - 1

theorem probability_f_leq_zero : 
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) →
  (∀ k ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f k x ≤ 0) →
  (∃ k ∈ Set.Icc (-2 : ℝ) (1 : ℝ), f k x ≤ 0) →
  ((1 - (-2)) / (2 - (-2)) = 3 / 4) :=
by sorry

end NUMINAMATH_GPT_probability_f_leq_zero_l2273_227300


namespace NUMINAMATH_GPT_cistern_problem_l2273_227331

theorem cistern_problem (fill_rate empty_rate net_rate : ℝ) (T : ℝ) : 
  fill_rate = 1 / 3 →
  net_rate = 7 / 30 →
  empty_rate = 1 / T →
  net_rate = fill_rate - empty_rate →
  T = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cistern_problem_l2273_227331


namespace NUMINAMATH_GPT_max_value_of_quadratic_l2273_227386

def quadratic_func (x : ℝ) : ℝ := -3 * (x - 2) ^ 2 - 3

theorem max_value_of_quadratic : 
  ∃ x : ℝ, quadratic_func x = -3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l2273_227386


namespace NUMINAMATH_GPT_painters_work_days_l2273_227384

theorem painters_work_days 
  (six_painters_days : ℝ) (number_six_painters : ℝ) (total_work_units : ℝ)
  (number_four_painters : ℝ) 
  (h1 : number_six_painters = 6)
  (h2 : six_painters_days = 1.4)
  (h3 : total_work_units = number_six_painters * six_painters_days) 
  (h4 : number_four_painters = 4) :
  2 + 1 / 10 = total_work_units / number_four_painters :=
by
  rw [h3, h1, h2, h4]
  sorry

end NUMINAMATH_GPT_painters_work_days_l2273_227384


namespace NUMINAMATH_GPT_other_candidate_votes_l2273_227305

theorem other_candidate_votes (h1 : one_candidate_votes / valid_votes = 0.6)
    (h2 : 0.3 * total_votes = invalid_votes)
    (h3 : total_votes = 9000)
    (h4 : valid_votes + invalid_votes = total_votes) :
    valid_votes - one_candidate_votes = 2520 :=
by
  sorry

end NUMINAMATH_GPT_other_candidate_votes_l2273_227305


namespace NUMINAMATH_GPT_select_10_teams_l2273_227325

def football_problem (teams : Finset ℕ) (played_on_day1 : Finset (ℕ × ℕ)) (played_on_day2 : Finset (ℕ × ℕ)) : Prop :=
  ∀ (v : ℕ), v ∈ teams → (∃ u w : ℕ, (u, v) ∈ played_on_day1 ∧ (v, w) ∈ played_on_day2)

theorem select_10_teams {teams : Finset ℕ}
  (h : teams.card = 20)
  {played_on_day1 played_on_day2 : Finset (ℕ × ℕ)}
  (h1 : ∀ ⦃u v : ℕ⦄, (u, v) ∈ played_on_day1 → u ∈ teams ∧ v ∈ teams)
  (h2 : ∀ ⦃u v : ℕ⦄, (u, v) ∈ played_on_day2 → u ∈ teams ∧ v ∈ teams)
  (h3 : ∀ x ∈ teams, ∃ u w, (u, x) ∈ played_on_day1 ∧ (x, w) ∈ played_on_day2) :
  ∃ S : Finset ℕ, S.card = 10 ∧ (∀ ⦃x y⦄, x ∈ S → y ∈ S → x ≠ y → (¬((x, y) ∈ played_on_day1) ∧ ¬((x, y) ∈ played_on_day2))) :=
by
  sorry

end NUMINAMATH_GPT_select_10_teams_l2273_227325


namespace NUMINAMATH_GPT_complete_work_together_in_days_l2273_227371

/-
p is 60% more efficient than q.
p can complete the work in 26 days.
Prove that p and q together will complete the work in approximately 18.57 days.
-/

noncomputable def work_together_days (p_efficiency q_efficiency : ℝ) (p_days : ℝ) : ℝ :=
  let p_work_rate := 1 / p_days
  let q_work_rate := q_efficiency / p_efficiency * p_work_rate
  let combined_work_rate := p_work_rate + q_work_rate
  1 / combined_work_rate

theorem complete_work_together_in_days :
  ∀ (p_efficiency q_efficiency p_days : ℝ),
  p_efficiency = 1 ∧ q_efficiency = 0.4 ∧ p_days = 26 →
  abs (work_together_days p_efficiency q_efficiency p_days - 18.57) < 0.01 := by
  intros p_efficiency q_efficiency p_days
  rintro ⟨heff_p, heff_q, hdays_p⟩
  simp [heff_p, heff_q, hdays_p, work_together_days]
  sorry

end NUMINAMATH_GPT_complete_work_together_in_days_l2273_227371


namespace NUMINAMATH_GPT_construct_origin_from_A_and_B_l2273_227333

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨3, 1⟩
def isAboveAndToLeft (p₁ p₂ : Point) : Prop := p₁.x < p₂.x ∧ p₁.y > p₂.y
def isOriginConstructed (A B : Point) : Prop := ∃ O : Point, O = ⟨0, 0⟩

theorem construct_origin_from_A_and_B : 
  isAboveAndToLeft A B → isOriginConstructed A B :=
by
  sorry

end NUMINAMATH_GPT_construct_origin_from_A_and_B_l2273_227333


namespace NUMINAMATH_GPT_calculate_p_p1_neg1_p_neg5_neg2_l2273_227399

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then
    x + y
  else if x < 0 ∧ y < 0 then
    x - 2 * y
  else
    3 * x + y

theorem calculate_p_p1_neg1_p_neg5_neg2 :
  p (p 1 (-1)) (p (-5) (-2)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_calculate_p_p1_neg1_p_neg5_neg2_l2273_227399


namespace NUMINAMATH_GPT_total_players_l2273_227308

-- Definitions for conditions
def K : Nat := 10
def KK : Nat := 30
def B : Nat := 5

-- Statement of the proof problem
theorem total_players : K + KK - B = 35 :=
by
  -- Proof not required, just providing the statement
  sorry

end NUMINAMATH_GPT_total_players_l2273_227308


namespace NUMINAMATH_GPT_arithmetic_sequence_9th_term_l2273_227330

theorem arithmetic_sequence_9th_term (S : ℕ → ℕ) (d : ℕ) (Sn : ℕ) (a9 : ℕ) :
  (∀ n, S n = (n * (2 * S 1 + (n - 1) * d)) / 2) →
  d = 2 →
  Sn = 81 →
  S 9 = Sn →
  a9 = S 1 + 8 * d →
  a9 = 17 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_9th_term_l2273_227330


namespace NUMINAMATH_GPT_range_of_f_l2273_227336

noncomputable def f (x y z : ℝ) := ((x * y + y * z + z * x) * (x + y + z)) / ((x + y) * (y + z) * (z + x))

theorem range_of_f :
  ∃ x y z : ℝ, (0 < x ∧ 0 < y ∧ 0 < z) ∧ f x y z = r ↔ 1 ≤ r ∧ r ≤ 9 / 8 :=
sorry

end NUMINAMATH_GPT_range_of_f_l2273_227336


namespace NUMINAMATH_GPT_intersection_M_N_l2273_227314

def M : Set ℝ := { x | x^2 - x - 6 ≤ 0 }
def N : Set ℝ := { x | -2 < x ∧ x ≤ 4 }

theorem intersection_M_N : (M ∩ N) = { x | -2 < x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2273_227314


namespace NUMINAMATH_GPT_number_of_n_with_odd_tens_digit_in_square_l2273_227304

def ends_in_3_or_7 (n : ℕ) : Prop :=
  n % 10 = 3 ∨ n % 10 = 7

def tens_digit_odd (n : ℕ) : Prop :=
  ((n * n / 10) % 10) % 2 = 1

theorem number_of_n_with_odd_tens_digit_in_square :
  ∀ n ∈ {n : ℕ | n ≤ 50 ∧ ends_in_3_or_7 n}, ¬tens_digit_odd n :=
by 
  sorry

end NUMINAMATH_GPT_number_of_n_with_odd_tens_digit_in_square_l2273_227304


namespace NUMINAMATH_GPT_total_carrots_computation_l2273_227359

-- Definitions
def initial_carrots : ℕ := 19
def thrown_out_carrots : ℕ := 4
def next_day_carrots : ℕ := 46

def total_carrots (c1 c2 t : ℕ) : ℕ := (c1 - t) + c2

-- The statement to prove
theorem total_carrots_computation :
  total_carrots initial_carrots next_day_carrots thrown_out_carrots = 61 :=
by sorry

end NUMINAMATH_GPT_total_carrots_computation_l2273_227359


namespace NUMINAMATH_GPT_print_time_325_pages_l2273_227392

theorem print_time_325_pages (pages : ℕ) (rate : ℕ) (delay_pages : ℕ) (delay_time : ℕ)
  (h_pages : pages = 325) (h_rate : rate = 25) (h_delay_pages : delay_pages = 100) (h_delay_time : delay_time = 1) :
  let print_time := pages / rate
  let delays := pages / delay_pages
  let total_time := print_time + delays * delay_time
  total_time = 16 :=
by
  sorry

end NUMINAMATH_GPT_print_time_325_pages_l2273_227392


namespace NUMINAMATH_GPT_surface_area_increase_l2273_227361

theorem surface_area_increase :
  let l := 4
  let w := 3
  let h := 2
  let side_cube := 1
  let original_surface := 2 * (l * w + l * h + w * h)
  let additional_surface := 6 * side_cube * side_cube
  let new_surface := original_surface + additional_surface
  new_surface = original_surface + 6 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_increase_l2273_227361


namespace NUMINAMATH_GPT_determine_a_l2273_227307

theorem determine_a (a b c : ℤ) (h_eq : ∀ x : ℤ, (x - a) * (x - 15) + 4 = (x + b) * (x + c)) :
  a = 16 ∨ a = 21 :=
  sorry

end NUMINAMATH_GPT_determine_a_l2273_227307


namespace NUMINAMATH_GPT_x_y_sum_l2273_227382

theorem x_y_sum (x y : ℝ) (h1 : |x| - 2 * x + y = 1) (h2 : x - |y| + y = 8) :
  x + y = 17 ∨ x + y = 1 :=
by
  sorry

end NUMINAMATH_GPT_x_y_sum_l2273_227382


namespace NUMINAMATH_GPT_cylinder_ellipse_major_axis_l2273_227317

-- Given a right circular cylinder of radius 2
-- and a plane intersecting it forming an ellipse
-- with the major axis being 50% longer than the minor axis,
-- prove that the length of the major axis is 6.

theorem cylinder_ellipse_major_axis :
  ∀ (r : ℝ) (major minor : ℝ),
    r = 2 → major = 1.5 * minor → minor = 2 * r → major = 6 :=
by
  -- Proof step to be filled by the prover.
  sorry

end NUMINAMATH_GPT_cylinder_ellipse_major_axis_l2273_227317


namespace NUMINAMATH_GPT_parallel_planes_of_perpendicular_lines_l2273_227396

-- Definitions of planes and lines
variable (Plane Line : Type)
variable (α β γ : Plane)
variable (m n : Line)

-- Relations between planes and lines
variable (perpendicular : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Conditions for the proof
variable (m_perp_α : perpendicular α m)
variable (n_perp_β : perpendicular β n)
variable (m_par_n : line_parallel m n)

-- Statement of the theorem
theorem parallel_planes_of_perpendicular_lines :
  parallel α β :=
sorry

end NUMINAMATH_GPT_parallel_planes_of_perpendicular_lines_l2273_227396


namespace NUMINAMATH_GPT_max_product_xyz_l2273_227389

theorem max_product_xyz : ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 12 ∧ z ≤ 3 * x ∧ ∀ (a b c : ℕ), a + b + c = 12 → c ≤ 3 * a → 0 < a ∧ 0 < b ∧ 0 < c → a * b * c ≤ 48 :=
by
  sorry

end NUMINAMATH_GPT_max_product_xyz_l2273_227389


namespace NUMINAMATH_GPT_arithmetic_seq_common_diff_l2273_227348

theorem arithmetic_seq_common_diff (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 0 + a 2 = 10) 
  (h2 : a 3 + a 5 = 4)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d) :
  d = -1 := 
  sorry

end NUMINAMATH_GPT_arithmetic_seq_common_diff_l2273_227348


namespace NUMINAMATH_GPT_circle_diameter_of_circumscribed_square_l2273_227351

theorem circle_diameter_of_circumscribed_square (r : ℝ) (s : ℝ) (h1 : s = 2 * r) (h2 : 4 * s = π * r^2) : 2 * r = 16 / π := by
  sorry

end NUMINAMATH_GPT_circle_diameter_of_circumscribed_square_l2273_227351


namespace NUMINAMATH_GPT_points_per_member_l2273_227352

theorem points_per_member
  (total_members : ℕ)
  (members_didnt_show : ℕ)
  (total_points : ℕ)
  (H1 : total_members = 14)
  (H2 : members_didnt_show = 7)
  (H3 : total_points = 35) :
  total_points / (total_members - members_didnt_show) = 5 :=
by
  sorry

end NUMINAMATH_GPT_points_per_member_l2273_227352


namespace NUMINAMATH_GPT_intersection_eq_l2273_227374

def A : Set ℝ := { x | abs x ≤ 2 }
def B : Set ℝ := { x | 3 * x - 2 ≥ 1 }

theorem intersection_eq :
  A ∩ B = { x | 1 ≤ x ∧ x ≤ 2 } :=
sorry

end NUMINAMATH_GPT_intersection_eq_l2273_227374


namespace NUMINAMATH_GPT_geometric_series_sum_l2273_227364

theorem geometric_series_sum :
  let a := 1 / 4
  let r := - (1 / 4)
  ∃ S : ℚ, S = (a * (1 - r^6)) / (1 - r) ∧ S = 4095 / 81920 :=
by
  let a := 1 / 4
  let r := - (1 / 4)
  exists (a * (1 - r^6)) / (1 - r)
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l2273_227364


namespace NUMINAMATH_GPT_arithmetic_geometric_l2273_227301

theorem arithmetic_geometric (a_n : ℕ → ℤ) (h1 : ∀ n, a_n n = a_n 0 + n * 2)
  (h2 : ∃ a, a = a_n 0 ∧ (a_n 0 + 4)^2 = a_n 0 * (a_n 0 + 6)) : a_n 0 = -8 := by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_l2273_227301


namespace NUMINAMATH_GPT_sequence_sum_l2273_227363

variable (P Q R S T U V : ℤ)
variable (hR : R = 7)
variable (h1 : P + Q + R = 36)
variable (h2 : Q + R + S = 36)
variable (h3 : R + S + T = 36)
variable (h4 : S + T + U = 36)
variable (h5 : T + U + V = 36)

theorem sequence_sum (P Q R S T U V : ℤ)
  (hR : R = 7)
  (h1 : P + Q + R = 36)
  (h2 : Q + R + S = 36)
  (h3 : R + S + T = 36)
  (h4 : S + T + U = 36)
  (h5 : T + U + V = 36) :
  P + V = 29 := 
sorry

end NUMINAMATH_GPT_sequence_sum_l2273_227363


namespace NUMINAMATH_GPT_min_total_weight_l2273_227319

theorem min_total_weight (crates: Nat) (weight_per_crate: Nat) (h1: crates = 6) (h2: weight_per_crate ≥ 120): 
  crates * weight_per_crate ≥ 720 :=
by
  sorry

end NUMINAMATH_GPT_min_total_weight_l2273_227319


namespace NUMINAMATH_GPT_maurice_needs_7_letters_l2273_227343
noncomputable def prob_no_job (n : ℕ) : ℝ := (4 / 5) ^ n

theorem maurice_needs_7_letters :
  ∃ n : ℕ, (prob_no_job n) ≤ 1 / 4 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_maurice_needs_7_letters_l2273_227343


namespace NUMINAMATH_GPT_at_least_two_squares_same_size_l2273_227335

theorem at_least_two_squares_same_size (S : ℝ) : 
  ∃ a b : ℝ, a = b ∧ 
  (∀ i : ℕ, i < 10 → 
   ∀ j : ℕ, j < 10 → 
   (∃ k : ℕ, k < 9 ∧ 
    ((∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ x ≠ y → 
          (i = x ∧ j = y)) → 
        ((S / 10) = (a * k)) ∨ ((S / 10) = (b * k))))) := sorry

end NUMINAMATH_GPT_at_least_two_squares_same_size_l2273_227335


namespace NUMINAMATH_GPT_sunglasses_cap_probability_l2273_227380

theorem sunglasses_cap_probability
  (sunglasses_count : ℕ) (caps_count : ℕ)
  (P_cap_and_sunglasses_given_cap : ℚ)
  (H1 : sunglasses_count = 60)
  (H2 : caps_count = 40)
  (H3 : P_cap_and_sunglasses_given_cap = 2/5) :
  (∃ (x : ℚ), x = (16 : ℚ) / 60 ∧ x = 4 / 15) := sorry

end NUMINAMATH_GPT_sunglasses_cap_probability_l2273_227380


namespace NUMINAMATH_GPT_find_m_l2273_227377

theorem find_m (x m : ℝ) :
  (2 * x + m) * (x - 3) = 2 * x^2 - 3 * m ∧ 
  (∀ c : ℝ, c * x = 0 → c = 0) → 
  m = 6 :=
by sorry

end NUMINAMATH_GPT_find_m_l2273_227377


namespace NUMINAMATH_GPT_eval_expr_l2273_227328

theorem eval_expr : (1 / (5^2)^4 * 5^11 * 2) = 250 := by
  sorry

end NUMINAMATH_GPT_eval_expr_l2273_227328


namespace NUMINAMATH_GPT_divisibility_equiv_l2273_227355

theorem divisibility_equiv (n : ℕ) : (7 ∣ 3^n + n^3) ↔ (7 ∣ 3^n * n^3 + 1) :=
by sorry

end NUMINAMATH_GPT_divisibility_equiv_l2273_227355


namespace NUMINAMATH_GPT_tank_cost_minimization_l2273_227387

def volume := 4800
def depth := 3
def cost_per_sqm_bottom := 150
def cost_per_sqm_walls := 120

theorem tank_cost_minimization (x : ℝ) 
  (S1 : ℝ := volume / depth)
  (S2 : ℝ := 6 * (x + (S1 / x)))
  (cost := cost_per_sqm_bottom * S1 + cost_per_sqm_walls * S2) :
  (x = 40) → cost = 297600 :=
sorry

end NUMINAMATH_GPT_tank_cost_minimization_l2273_227387


namespace NUMINAMATH_GPT_start_A_to_B_l2273_227320

theorem start_A_to_B (x : ℝ)
  (A_to_C : x = 1000 * (1000 / 571.43) - 1000)
  (h1 : 1000 / (1000 - 600) = 1000 / (1000 - 428.57))
  (h2 : x = 1750 - 1000) :
  x = 750 :=
by
  rw [h2]
  sorry   -- Proof to be filled in.

end NUMINAMATH_GPT_start_A_to_B_l2273_227320


namespace NUMINAMATH_GPT_ratio_to_percentage_l2273_227391

theorem ratio_to_percentage (x y : ℚ) (h : (2/3 * x) / (4/5 * y) = 5 / 6) : (5 / 6 : ℚ) * 100 = 83.33 :=
by
  sorry

end NUMINAMATH_GPT_ratio_to_percentage_l2273_227391


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2273_227344

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 3

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → f a x - f a 1 ≥ 0) ↔ (a ≤ -2) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2273_227344


namespace NUMINAMATH_GPT_percentage_increase_weekends_l2273_227376

def weekday_price : ℝ := 18
def weekend_price : ℝ := 27

theorem percentage_increase_weekends : 
  (weekend_price - weekday_price) / weekday_price * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_weekends_l2273_227376


namespace NUMINAMATH_GPT_find_divisor_l2273_227388

theorem find_divisor (d : ℕ) : ((23 = (d * 7) + 2) → d = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l2273_227388


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2273_227309

def A : Set ℤ := {-2, -1}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B :
  A ∩ B = {-1} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2273_227309


namespace NUMINAMATH_GPT_false_proposition_p_and_q_l2273_227322

open Classical

-- Define the propositions
def p (a b c : ℝ) : Prop := b * b = a * c
def q (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- We provide the conditions specified in the problem
variable (a b c : ℝ)
variable (f : ℝ → ℝ)
axiom hq : ∀ x, f x = f (-x)
axiom hp : ¬ (∀ a b c, p a b c ↔ (b ≠ 0 ∧ c ≠ 0 ∧ b * b = a * c))

-- The false proposition among the given options is "p and q"
theorem false_proposition_p_and_q : ¬ (∀ a b c (f : ℝ → ℝ), p a b c ∧ q f) :=
by
  -- This is where the proof would go, but is marked as a placeholder
  sorry

end NUMINAMATH_GPT_false_proposition_p_and_q_l2273_227322


namespace NUMINAMATH_GPT_max_lateral_surface_area_l2273_227340

theorem max_lateral_surface_area : ∀ (x y : ℝ), 6 * x + 3 * y = 12 → (3 * x * y) ≤ 6 :=
by
  intros x y h
  have xy_le_2 : x * y ≤ 2 :=
    by
      sorry
  have max_area_6 : 3 * x * y ≤ 6 :=
    by
      sorry
  exact max_area_6

end NUMINAMATH_GPT_max_lateral_surface_area_l2273_227340


namespace NUMINAMATH_GPT_total_integers_at_least_eleven_l2273_227310

theorem total_integers_at_least_eleven (n neg_count : ℕ) 
  (h1 : neg_count % 2 = 1)
  (h2 : neg_count ≤ 11) :
  n ≥ 11 := 
sorry

end NUMINAMATH_GPT_total_integers_at_least_eleven_l2273_227310


namespace NUMINAMATH_GPT_samantha_spends_on_dog_toys_l2273_227302

theorem samantha_spends_on_dog_toys:
  let toy_price := 12.00
  let discount := 0.5
  let num_toys := 4
  let tax_rate := 0.08
  let full_price_toys := num_toys / 2
  let half_price_toys := num_toys / 2
  let total_cost_before_tax := full_price_toys * toy_price + half_price_toys * (toy_price * discount)
  let sales_tax := total_cost_before_tax * tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  total_cost_after_tax = 38.88 :=
by {
  sorry
}

end NUMINAMATH_GPT_samantha_spends_on_dog_toys_l2273_227302


namespace NUMINAMATH_GPT_jason_pokemon_cards_l2273_227318

theorem jason_pokemon_cards :
  ∀ (initial_cards trade_benny_lost trade_benny_gain trade_sean_lost trade_sean_gain give_to_brother : ℕ),
  initial_cards = 5 →
  trade_benny_lost = 2 →
  trade_benny_gain = 3 →
  trade_sean_lost = 3 →
  trade_sean_gain = 4 →
  give_to_brother = 2 →
  initial_cards - trade_benny_lost + trade_benny_gain - trade_sean_lost + trade_sean_gain - give_to_brother = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jason_pokemon_cards_l2273_227318


namespace NUMINAMATH_GPT_pentagon_area_l2273_227357

theorem pentagon_area {a b c d e : ℕ} (split: ℕ) (non_parallel1 non_parallel2 parallel1 parallel2 : ℕ)
  (h1 : a = 16) (h2 : b = 25) (h3 : c = 30) (h4 : d = 26) (h5 : e = 25)
  (split_condition : a + b + c + d + e = 5 * split)
  (np_condition1: non_parallel1 = c) (np_condition2: non_parallel2 = a)
  (p_condition1: parallel1 = d) (p_condition2: parallel2 = e)
  (area_triangle: 1 / 2 * b * a = 200)
  (area_trapezoid: 1 / 2 * (parallel1 + parallel2) * non_parallel1 = 765) :
  a + b + c + d + e = 965 := by
  sorry

end NUMINAMATH_GPT_pentagon_area_l2273_227357


namespace NUMINAMATH_GPT_final_price_correct_l2273_227338

def cost_price : ℝ := 20
def profit_percentage : ℝ := 0.30
def sale_discount_percentage : ℝ := 0.50
def local_tax_percentage : ℝ := 0.10
def packaging_fee : ℝ := 2

def selling_price_before_discount : ℝ := cost_price * (1 + profit_percentage)
def sale_discount : ℝ := sale_discount_percentage * selling_price_before_discount
def price_after_discount : ℝ := selling_price_before_discount - sale_discount
def tax : ℝ := local_tax_percentage * price_after_discount
def price_with_tax : ℝ := price_after_discount + tax
def final_price : ℝ := price_with_tax + packaging_fee

theorem final_price_correct : final_price = 16.30 :=
by
  sorry

end NUMINAMATH_GPT_final_price_correct_l2273_227338


namespace NUMINAMATH_GPT_mean_problem_l2273_227303

theorem mean_problem (x : ℝ) (h : (12 + x + 42 + 78 + 104) / 5 = 62) :
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 :=
by
  sorry

end NUMINAMATH_GPT_mean_problem_l2273_227303


namespace NUMINAMATH_GPT_jericho_money_left_l2273_227358

/--
Given:
1. Twice the money Jericho has is 60.
2. Jericho owes Annika $14.
3. Jericho owes Manny half as much as he owes Annika.

Prove:
Jericho will be left with $9 after paying off all his debts.
-/
theorem jericho_money_left (j_money : ℕ) (annika_owes : ℕ) (manny_multiplier : ℕ) (debt : ℕ) (remaining_money : ℕ) :
  2 * j_money = 60 →
  annika_owes = 14 →
  manny_multiplier = 1 / 2 →
  debt = annika_owes + manny_multiplier * annika_owes →
  remaining_money = j_money - debt →
  remaining_money = 9 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_jericho_money_left_l2273_227358


namespace NUMINAMATH_GPT_Tim_sleep_hours_l2273_227372

theorem Tim_sleep_hours (x : ℕ) : 
  (x + x + 10 + 10 = 32) → x = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Tim_sleep_hours_l2273_227372


namespace NUMINAMATH_GPT_items_per_charge_is_five_l2273_227393

-- Define the number of dog treats, chew toys, rawhide bones, and credit cards as constants.
def num_dog_treats := 8
def num_chew_toys := 2
def num_rawhide_bones := 10
def num_credit_cards := 4

-- Define the total number of items.
def total_items := num_dog_treats + num_chew_toys + num_rawhide_bones

-- Prove that the number of items per credit card charge is 5.
theorem items_per_charge_is_five :
  (total_items / num_credit_cards) = 5 :=
by
  -- Proof goes here (we use sorry to skip the actual proof)
  sorry

end NUMINAMATH_GPT_items_per_charge_is_five_l2273_227393


namespace NUMINAMATH_GPT_evaluate_expression_l2273_227366

theorem evaluate_expression :
  (4^1001 * 9^1002) / (6^1002 * 4^1000) = (3^1002) / (2^1000) :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l2273_227366


namespace NUMINAMATH_GPT_find_m_value_l2273_227312

noncomputable def is_direct_proportion_function (m : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 1) * x ^ (2 - m^2) = k * x

theorem find_m_value (m : ℝ) (hk : ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 1) * x ^ (2 - m^2) = k * x) : m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l2273_227312


namespace NUMINAMATH_GPT_max_area_of_triangle_ABC_l2273_227360

noncomputable def max_triangle_area (a b c : ℝ) (A B C : ℝ) := 
  1 / 2 * b * c * Real.sin A

theorem max_area_of_triangle_ABC :
  ∀ (a b c A B C : ℝ)
  (ha : a = 2)
  (hTrig : a = Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A))
  (hCondition: 3 * b * Real.sin C - 5 * c * Real.sin B * Real.cos A = 0),
  max_triangle_area a b c A B C ≤ 2 := 
by
  intros a b c A B C ha hTrig hCondition
  sorry

end NUMINAMATH_GPT_max_area_of_triangle_ABC_l2273_227360


namespace NUMINAMATH_GPT_solution_to_system_l2273_227375

theorem solution_to_system (x y a b : ℝ) 
  (h1 : x = 1) (h2 : y = 2) 
  (h3 : a * x + b * y = 4) 
  (h4 : b * x - a * y = 7) : 
  a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_system_l2273_227375


namespace NUMINAMATH_GPT_students_total_l2273_227397

theorem students_total (scavenger_hunt_students : ℕ) (ski_trip_students : ℕ) 
  (h1 : ski_trip_students = 2 * scavenger_hunt_students) 
  (h2 : scavenger_hunt_students = 4000) : 
  scavenger_hunt_students + ski_trip_students = 12000 := 
by
  sorry

end NUMINAMATH_GPT_students_total_l2273_227397


namespace NUMINAMATH_GPT_proof_l2273_227342

variable (p : ℕ) (ε : ℤ)
variable (RR NN NR RN : ℕ)

-- Conditions
axiom h1 : ∀ n ≤ p - 2, 
  (n % 2 = 0 ∧ (n + 1) % 2 = 0) ∨ 
  (n % 2 ≠ 0 ∧ (n + 1) % 2 ≠ 0) ∨ 
  (n % 2 ≠ 0 ∧ (n + 1) % 2 = 0 ) ∨ 
  (n % 2 = 0 ∧ (n + 1) % 2 ≠ 0) 

axiom h2 :  RR + NN - RN - NR = 1

axiom h3 : ε = (-1) ^ ((p - 1) / 2)

axiom h4 : RR + RN = (p - 2 - ε) / 2

axiom h5 : RR + NR = (p - 1) / 2 - 1

axiom h6 : NR + NN = (p - 2 + ε) / 2

axiom h7 : RN + NN = (p - 1) / 2  

-- To prove
theorem proof : 
  RR = (p / 4) - (ε + 4) / 4 ∧ 
  RN = (p / 4) - (ε) / 4 ∧ 
  NN = (p / 4) + (ε - 2) / 4 ∧ 
  NR = (p / 4) + (ε - 2) / 4 := 
sorry

end NUMINAMATH_GPT_proof_l2273_227342


namespace NUMINAMATH_GPT_square_area_720_l2273_227395

noncomputable def length_squared {α : Type*} [EuclideanDomain α] (a b : α) := a * a + b * b

theorem square_area_720
  (side x : ℝ)
  (h1 : BE = 20) (h2 : EF = 20) (h3 : FD = 20)
  (h4 : AE = 2 * ED) (h5 : BF = 2 * FC)
  : x * x = 720 :=
by
  let AE := 2/3 * side
  let ED := 1/3 * side
  let BF := 2/3 * side
  let FC := 1/3 * side
  have h6 : length_squared BF EF = BE * BE := sorry
  have h7 : x * x = 720 := sorry
  exact h7

end NUMINAMATH_GPT_square_area_720_l2273_227395


namespace NUMINAMATH_GPT_max_value_quadratic_function_l2273_227349

open Real

theorem max_value_quadratic_function (r : ℝ) (x₀ y₀ : ℝ) (P_tangent : (2 / x₀) * x - y₀ = 0) 
  (circle_tangent : (x₀ - 3) * (x - 3) + y₀ * y = r^2) :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 1 / 2 * x * (3 - x)) ∧ 
  (∀ (x : ℝ), f x ≤ 9 / 8) :=
by
  sorry

end NUMINAMATH_GPT_max_value_quadratic_function_l2273_227349


namespace NUMINAMATH_GPT_value_of_expression_l2273_227362

open Real

theorem value_of_expression (α : ℝ) (h : 3 * sin α + cos α = 0) :
  1 / (cos α ^ 2 + sin (2 * α)) = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2273_227362


namespace NUMINAMATH_GPT_candies_remaining_after_yellow_eaten_l2273_227334

theorem candies_remaining_after_yellow_eaten :
  let red_candies := 40
  let yellow_candies := 3 * red_candies - 20
  let blue_candies := yellow_candies / 2
  red_candies + blue_candies = 90 :=
by
  sorry

end NUMINAMATH_GPT_candies_remaining_after_yellow_eaten_l2273_227334


namespace NUMINAMATH_GPT_circle_tangent_l2273_227350

theorem circle_tangent {m : ℝ} (h : ∃ (x y : ℝ), (x - 3)^2 + (y - 4)^2 = 25 - m ∧ x^2 + y^2 = 1) :
  m = 9 :=
sorry

end NUMINAMATH_GPT_circle_tangent_l2273_227350


namespace NUMINAMATH_GPT_rectangle_perimeter_l2273_227398

theorem rectangle_perimeter :
  ∃ (a b : ℤ), a ≠ b ∧ a * b = 2 * (2 * a + 2 * b) ∧ 2 * (a + b) = 36 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l2273_227398


namespace NUMINAMATH_GPT_algebraic_expression_correct_l2273_227345

-- Definition of the problem
def algebraic_expression (x : ℝ) : ℝ :=
  2 * x + 3

-- Theorem statement
theorem algebraic_expression_correct (x : ℝ) :
  algebraic_expression x = 2 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_correct_l2273_227345


namespace NUMINAMATH_GPT_Jean_calls_thursday_l2273_227339

theorem Jean_calls_thursday :
  ∃ (thursday_calls : ℕ), thursday_calls = 61 ∧ 
  (∃ (mon tue wed fri : ℕ),
    mon = 35 ∧ 
    tue = 46 ∧ 
    wed = 27 ∧ 
    fri = 31 ∧ 
    (mon + tue + wed + thursday_calls + fri = 40 * 5)) :=
sorry

end NUMINAMATH_GPT_Jean_calls_thursday_l2273_227339


namespace NUMINAMATH_GPT_rationalize_denominator_simplify_l2273_227323

theorem rationalize_denominator_simplify :
  let a : ℝ := 3
  let b : ℝ := 2
  let c : ℝ := 1
  let d : ℝ := 2
  ∀ (x y z : ℝ), 
  (x = 3 * Real.sqrt 2) → 
  (y = 3) → 
  (z = Real.sqrt 3) → 
  (x / (y - z) = (3 * Real.sqrt 2 + Real.sqrt 6) / 2) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_simplify_l2273_227323


namespace NUMINAMATH_GPT_value_of_expression_l2273_227354

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2273_227354


namespace NUMINAMATH_GPT_number_of_milkshakes_l2273_227385

-- Define the amounts and costs
def initial_money : ℕ := 132
def remaining_money : ℕ := 70
def hamburger_cost : ℕ := 4
def milkshake_cost : ℕ := 5
def hamburgers_bought : ℕ := 8

-- Defining the money spent calculations
def hamburgers_spent : ℕ := hamburgers_bought * hamburger_cost
def total_spent : ℕ := initial_money - remaining_money
def milkshake_spent : ℕ := total_spent - hamburgers_spent

-- The final theorem to prove
theorem number_of_milkshakes : (milkshake_spent / milkshake_cost) = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_milkshakes_l2273_227385


namespace NUMINAMATH_GPT_rectangle_area_change_l2273_227367

theorem rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A := L * B
  let L' := 1.15 * L
  let B' := 0.80 * B
  let A' := L' * B'
  A' = 0.92 * A :=
by
  let A := L * B
  let L' := 1.15 * L
  let B' := 0.80 * B
  let A' := L' * B'
  show A' = 0.92 * A
  sorry

end NUMINAMATH_GPT_rectangle_area_change_l2273_227367


namespace NUMINAMATH_GPT_race_participants_minimum_l2273_227321

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end NUMINAMATH_GPT_race_participants_minimum_l2273_227321


namespace NUMINAMATH_GPT_rabbit_speed_final_result_l2273_227347

def rabbit_speed : ℕ := 45

def double_speed (speed : ℕ) : ℕ := speed * 2

def add_four (n : ℕ) : ℕ := n + 4

def final_operation : ℕ := double_speed (add_four (double_speed rabbit_speed))

theorem rabbit_speed_final_result : final_operation = 188 := 
by
  sorry

end NUMINAMATH_GPT_rabbit_speed_final_result_l2273_227347


namespace NUMINAMATH_GPT_geom_seq_thm_l2273_227329

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
a 1 ≠ 0 ∧ ∀ n, a (n + 1) = (a n ^ 2) / (a (n - 1))

theorem geom_seq_thm (a : ℕ → ℝ) (h : geom_seq a) (h_neg : ∀ n, a n < 0) 
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) : a 3 + a 5 = -6 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_thm_l2273_227329


namespace NUMINAMATH_GPT_sum_of_roots_abs_gt_six_l2273_227365

theorem sum_of_roots_abs_gt_six {p r1 r2 : ℝ} (h1 : r1 + r2 = -p) (h2 : r1 * r2 = 9) (h3 : r1 ≠ r2) (h4 : p^2 > 36) : |r1 + r2| > 6 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_abs_gt_six_l2273_227365


namespace NUMINAMATH_GPT_last_five_digits_of_sequence_l2273_227327

theorem last_five_digits_of_sequence (seq : Fin 36 → Fin 2) 
  (h0 : seq 0 = 0) (h1 : seq 1 = 0) (h2 : seq 2 = 0) (h3 : seq 3 = 0) (h4 : seq 4 = 0)
  (unique_combos : ∀ (combo: Fin 32 → Fin 2), 
    ∃ (start_index : Fin 32), ∀ (i : Fin 5),
      combo i = seq ((start_index + i) % 36)) :
  seq 31 = 1 ∧ seq 32 = 1 ∧ seq 33 = 1 ∧ seq 34 = 0 ∧ seq 35 = 1 :=
by
  sorry

end NUMINAMATH_GPT_last_five_digits_of_sequence_l2273_227327


namespace NUMINAMATH_GPT_total_pay_l2273_227326

-- Definitions based on the conditions
def y_pay : ℕ := 290
def x_pay : ℕ := (120 * y_pay) / 100

-- The statement to prove that the total pay is Rs. 638
theorem total_pay : x_pay + y_pay = 638 := 
by
  -- skipping the proof for now
  sorry

end NUMINAMATH_GPT_total_pay_l2273_227326
