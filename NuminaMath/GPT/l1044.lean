import Mathlib

namespace NUMINAMATH_GPT_find_middle_number_l1044_104436

theorem find_middle_number (a b c d e : ℝ) (h1 : (a + b + c + d + e) / 5 = 12.5)
  (h2 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h3 : (a + b + c) / 3 = 11.6)
  (h4 : (c + d + e) / 3 = 13.5) : c = 12.8 :=
sorry

end NUMINAMATH_GPT_find_middle_number_l1044_104436


namespace NUMINAMATH_GPT_find_part_of_number_l1044_104442

theorem find_part_of_number (x y : ℕ) (h₁ : x = 1925) (h₂ : x / 7 = y + 100) : y = 175 :=
sorry

end NUMINAMATH_GPT_find_part_of_number_l1044_104442


namespace NUMINAMATH_GPT_infinite_sum_equals_one_fourth_l1044_104411

theorem infinite_sum_equals_one_fourth :
  ∑' n : ℕ, (3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
sorry

end NUMINAMATH_GPT_infinite_sum_equals_one_fourth_l1044_104411


namespace NUMINAMATH_GPT_construct_rectangle_l1044_104471

-- Define the essential properties of the rectangles
structure Rectangle where
  length : ℕ
  width : ℕ 

-- Define the given rectangles
def r1 : Rectangle := ⟨7, 1⟩
def r2 : Rectangle := ⟨6, 1⟩
def r3 : Rectangle := ⟨5, 1⟩
def r4 : Rectangle := ⟨4, 1⟩
def r5 : Rectangle := ⟨3, 1⟩
def r6 : Rectangle := ⟨2, 1⟩
def s  : Rectangle := ⟨1, 1⟩

-- Hypothesis for condition that length of each side of resulting rectangle should be > 1
def validSide (rect : Rectangle) : Prop :=
  rect.length > 1 ∧ rect.width > 1

-- The proof statement
theorem construct_rectangle : 
  (∃ rect1 rect2 rect3 rect4 : Rectangle, 
      rect1 = ⟨7, 1⟩ ∧ rect2 = ⟨6, 1⟩ ∧ rect3 = ⟨5, 1⟩ ∧ rect4 = ⟨4, 1⟩) →
  (∃ rect5 rect6 : Rectangle, 
      rect5 = ⟨3, 1⟩ ∧ rect6 = ⟨2, 1⟩) →
  (∃ square : Rectangle, 
      square = ⟨1, 1⟩) →
  (∃ compositeRect : Rectangle, 
      compositeRect.length = 7 ∧ 
      compositeRect.width = 4 ∧ 
      validSide compositeRect) :=
sorry

end NUMINAMATH_GPT_construct_rectangle_l1044_104471


namespace NUMINAMATH_GPT_min_distance_parabola_midpoint_l1044_104439

theorem min_distance_parabola_midpoint 
  (a : ℝ) (m : ℝ) (h_pos_a : a > 0) :
  (m ≥ 1 / a → ∃ M_y : ℝ, M_y = (2 * m * a - 1) / (4 * a)) ∧ 
  (m < 1 / a → ∃ M_y : ℝ, M_y = a * m^2 / 4) := 
by 
  sorry

end NUMINAMATH_GPT_min_distance_parabola_midpoint_l1044_104439


namespace NUMINAMATH_GPT_find_a_circle_line_intersection_l1044_104423

theorem find_a_circle_line_intersection
  (h1 : ∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0)
  (h2 : ∀ x y : ℝ, x + 2 * y + 1 = 0) :
  a = 3 := 
sorry

end NUMINAMATH_GPT_find_a_circle_line_intersection_l1044_104423


namespace NUMINAMATH_GPT_problem_1_minimum_value_problem_2_range_of_a_l1044_104486

noncomputable def e : ℝ := Real.exp 1  -- Definition of e as exp(1)

-- Question I:
-- Prove that the minimum value of the function f(x) = e^x - e*x - e is -e.
theorem problem_1_minimum_value :
  ∃ x : ℝ, (∀ y : ℝ, (Real.exp x - e * x - e) ≤ (Real.exp y - e * y - e))
  ∧ (Real.exp x - e * x - e) = -e := 
sorry

-- Question II:
-- Prove that the range of values for a such that f(x) = e^x - a*x - a >= 0 for all x is [0, 1].
theorem problem_2_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, (Real.exp x - a * x - a) ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_problem_1_minimum_value_problem_2_range_of_a_l1044_104486


namespace NUMINAMATH_GPT_max_set_size_divisible_diff_l1044_104437

theorem max_set_size_divisible_diff (S : Finset ℕ) (h1 : ∀ x ∈ S, ∀ y ∈ S, x ≠ y → (5 ∣ (x - y) ∨ 25 ∣ (x - y))) : S.card ≤ 25 :=
sorry

end NUMINAMATH_GPT_max_set_size_divisible_diff_l1044_104437


namespace NUMINAMATH_GPT_find_Z_l1044_104478

open Complex

-- Definitions
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_Z (Z : ℂ) (h1 : abs Z = 3) (h2 : is_pure_imaginary (Z + (3 * Complex.I))) : Z = 3 * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_find_Z_l1044_104478


namespace NUMINAMATH_GPT_union_A_B_inter_complement_A_B_range_a_l1044_104468

-- Define the sets A, B, and C
def A : Set ℝ := { x | 2 < x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | 5 - a < x ∧ x < a }

-- Part (I)
theorem union_A_B : A ∪ B = { x | 2 < x ∧ x < 10 } := sorry

theorem inter_complement_A_B :
  (Set.univ \ A) ∩ B = { x | 7 ≤ x ∧ x < 10 } := sorry

-- Part (II)
theorem range_a (a : ℝ) (h : C a ⊆ B) : a ≤ 3 := sorry

end NUMINAMATH_GPT_union_A_B_inter_complement_A_B_range_a_l1044_104468


namespace NUMINAMATH_GPT_sample_size_is_13_l1044_104493

noncomputable def stratified_sample_size : ℕ :=
  let A := 120
  let B := 80
  let C := 60
  let total_units := A + B + C
  let sampled_C_units := 3
  let sampling_fraction := sampled_C_units / C
  let n := sampling_fraction * total_units
  n

theorem sample_size_is_13 :
  stratified_sample_size = 13 := by
  sorry

end NUMINAMATH_GPT_sample_size_is_13_l1044_104493


namespace NUMINAMATH_GPT_find_a_range_l1044_104426

noncomputable
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 0 then x * Real.exp x else a * x ^ 2 - 2 * x

theorem find_a_range (a : ℝ) : (∀ x : ℝ, -1 / Real.exp 1 ≤ f x a) → a ∈ Set.Ici (Real.exp 1) :=
  sorry

end NUMINAMATH_GPT_find_a_range_l1044_104426


namespace NUMINAMATH_GPT_pablo_puzzle_pieces_per_hour_l1044_104487

theorem pablo_puzzle_pieces_per_hour
  (num_300_puzzles : ℕ)
  (num_500_puzzles : ℕ)
  (pieces_per_300_puzzle : ℕ)
  (pieces_per_500_puzzle : ℕ)
  (max_hours_per_day : ℕ)
  (total_days : ℕ)
  (total_pieces_completed : ℕ)
  (total_hours_spent : ℕ)
  (P : ℕ)
  (h1 : num_300_puzzles = 8)
  (h2 : num_500_puzzles = 5)
  (h3 : pieces_per_300_puzzle = 300)
  (h4 : pieces_per_500_puzzle = 500)
  (h5 : max_hours_per_day = 7)
  (h6 : total_days = 7)
  (h7 : total_pieces_completed = (num_300_puzzles * pieces_per_300_puzzle + num_500_puzzles * pieces_per_500_puzzle))
  (h8 : total_hours_spent = max_hours_per_day * total_days)
  (h9 : P = total_pieces_completed / total_hours_spent) :
  P = 100 :=
sorry

end NUMINAMATH_GPT_pablo_puzzle_pieces_per_hour_l1044_104487


namespace NUMINAMATH_GPT_zach_needs_more_tickets_l1044_104490

theorem zach_needs_more_tickets {ferris_wheel_tickets roller_coaster_tickets log_ride_tickets zach_tickets : ℕ} :
  ferris_wheel_tickets = 2 ∧
  roller_coaster_tickets = 7 ∧
  log_ride_tickets = 1 ∧
  zach_tickets = 1 →
  (ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets - zach_tickets = 9) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_zach_needs_more_tickets_l1044_104490


namespace NUMINAMATH_GPT_vasya_lowest_position_l1044_104462

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end NUMINAMATH_GPT_vasya_lowest_position_l1044_104462


namespace NUMINAMATH_GPT_diana_shopping_for_newborns_l1044_104416

-- Define the conditions
def num_toddlers : ℕ := 6
def num_teenagers : ℕ := 5 * num_toddlers
def total_children : ℕ := 40

-- Define the problem statement
theorem diana_shopping_for_newborns : (total_children - (num_toddlers + num_teenagers)) = 4 := by
  sorry

end NUMINAMATH_GPT_diana_shopping_for_newborns_l1044_104416


namespace NUMINAMATH_GPT_calculate_result_l1044_104449

theorem calculate_result :
  (-24) * ((5 / 6 : ℚ) - (4 / 3) + (5 / 8)) = -3 := 
by
  sorry

end NUMINAMATH_GPT_calculate_result_l1044_104449


namespace NUMINAMATH_GPT_quadratic_specific_a_l1044_104405

noncomputable def quadratic_root_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, (a + 2) * x^2 + 2 * a * x + 1 = 0

theorem quadratic_specific_a (a : ℝ) (h : quadratic_root_condition a) :
  a = 2 ∨ a = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_specific_a_l1044_104405


namespace NUMINAMATH_GPT_ladder_cost_l1044_104402

theorem ladder_cost (ladders1 ladders2 rung_count1 rung_count2 cost_per_rung : ℕ)
  (h1 : ladders1 = 10) (h2 : ladders2 = 20) (h3 : rung_count1 = 50) (h4 : rung_count2 = 60) (h5 : cost_per_rung = 2) :
  (ladders1 * rung_count1 + ladders2 * rung_count2) * cost_per_rung = 3400 :=
by 
  sorry

end NUMINAMATH_GPT_ladder_cost_l1044_104402


namespace NUMINAMATH_GPT_problem_1956_Tokyo_Tech_l1044_104445

theorem problem_1956_Tokyo_Tech (a b c : ℝ) (ha : 0 < a) (ha_lt_one : a < 1) (hb : 0 < b) 
(hb_lt_one : b < 1) (hc : 0 < c) (hc_lt_one : c < 1) : a + b + c - a * b * c < 2 := 
sorry

end NUMINAMATH_GPT_problem_1956_Tokyo_Tech_l1044_104445


namespace NUMINAMATH_GPT_solve_xyz_l1044_104469

theorem solve_xyz (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) :
  (x / 21) * (y / 189) + z = 1 ↔ x = 21 ∧ y = 567 ∧ z = 0 :=
sorry

end NUMINAMATH_GPT_solve_xyz_l1044_104469


namespace NUMINAMATH_GPT_players_per_group_l1044_104444

-- Definitions for given conditions
def num_new_players : Nat := 48
def num_returning_players : Nat := 6
def num_groups : Nat := 9

-- Proof that the number of players in each group is 6
theorem players_per_group :
  let total_players := num_new_players + num_returning_players
  total_players / num_groups = 6 := by
  sorry

end NUMINAMATH_GPT_players_per_group_l1044_104444


namespace NUMINAMATH_GPT_probability_AB_together_l1044_104407

theorem probability_AB_together : 
  let total_events := 6
  let ab_together_events := 4
  let probability := ab_together_events / total_events
  probability = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_AB_together_l1044_104407


namespace NUMINAMATH_GPT_pound_of_rice_cost_l1044_104421

theorem pound_of_rice_cost 
(E R K : ℕ) (h1: E = R) (h2: K = 4 * (E / 12)) (h3: K = 11) : R = 33 := by
  sorry

end NUMINAMATH_GPT_pound_of_rice_cost_l1044_104421


namespace NUMINAMATH_GPT_range_of_x_squared_f_x_lt_x_squared_minus_f_1_l1044_104460

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def satisfies_inequality (f f' : ℝ → ℝ) : Prop :=
∀ x : ℝ, 2 * f x + x * f' x < 2

theorem range_of_x_squared_f_x_lt_x_squared_minus_f_1 (f f' : ℝ → ℝ)
  (h_even : even_function f)
  (h_ineq : satisfies_inequality f f')
  : {x : ℝ | x^2 * f x - f 1 < x^2 - 1} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
sorry

end NUMINAMATH_GPT_range_of_x_squared_f_x_lt_x_squared_minus_f_1_l1044_104460


namespace NUMINAMATH_GPT_trig_system_solution_l1044_104414

theorem trig_system_solution (x y : ℝ) (hx : 0 ≤ x ∧ x < 2 * Real.pi) (hy : 0 ≤ y ∧ y < 2 * Real.pi)
  (h1 : Real.sin x + Real.cos y = 0) (h2 : Real.cos x * Real.sin y = -1/2) :
    (x = Real.pi / 4 ∧ y = 5 * Real.pi / 4) ∨
    (x = 3 * Real.pi / 4 ∧ y = 3 * Real.pi / 4) ∨
    (x = 5 * Real.pi / 4 ∧ y = Real.pi / 4) ∨
    (x = 7 * Real.pi / 4 ∧ y = 7 * Real.pi / 4) := by
  sorry

end NUMINAMATH_GPT_trig_system_solution_l1044_104414


namespace NUMINAMATH_GPT_problem_solution_l1044_104425

variables {R : Type} [LinearOrder R]

def M (x y : R) : R := max x y
def m (x y : R) : R := min x y

theorem problem_solution (p q r s t : R) (h : p < q) (h1 : q < r) (h2 : r < s) (h3 : s < t) :
  M (M p (m q r)) (m s (M p t)) = q :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1044_104425


namespace NUMINAMATH_GPT_diagonal_perimeter_ratio_l1044_104476

theorem diagonal_perimeter_ratio
    (b : ℝ)
    (h : b ≠ 0) -- To ensure the garden has non-zero side lengths
    (a : ℝ) (h1: a = 3 * b) 
    (d : ℝ) (h2: d = (Real.sqrt (b^2 + a^2)))
    (P : ℝ) (h3: P = 2 * a + 2 * b)
    (h4 : d = b * (Real.sqrt 10)) :
  d / P = (Real.sqrt 10) / 8 := by
    sorry

end NUMINAMATH_GPT_diagonal_perimeter_ratio_l1044_104476


namespace NUMINAMATH_GPT_critical_force_rod_truncated_cone_l1044_104466

-- Define the given conditions
variable (r0 : ℝ) (q : ℝ) (E : ℝ) (l : ℝ) (π : ℝ)

-- Assumptions
axiom q_positive : q > 0

-- Definition for the new radius based on q
def r1 : ℝ := r0 * (1 + q)

-- Proof problem statement
theorem critical_force_rod_truncated_cone (h : q > 0) : 
  ∃ Pkp : ℝ, Pkp = (E * π * r0^4 * 4.743 / l^2) * (1 + 2 * q) :=
sorry

end NUMINAMATH_GPT_critical_force_rod_truncated_cone_l1044_104466


namespace NUMINAMATH_GPT_diameter_of_circle_A_l1044_104473

theorem diameter_of_circle_A (r_B r_C : ℝ) (h1 : r_B = 12) (h2 : r_C = 3)
  (area_relation : ∀ (r_A : ℝ), π * (r_B^2 - r_A^2) = 4 * (π * r_C^2)) :
  ∃ r_A : ℝ, 2 * r_A = 12 * Real.sqrt 3 := by
  -- We will club the given conditions and logical sequence here
  sorry

end NUMINAMATH_GPT_diameter_of_circle_A_l1044_104473


namespace NUMINAMATH_GPT_rectangular_field_diagonal_length_l1044_104435

noncomputable def diagonal_length_of_rectangular_field (a : ℝ) (A : ℝ) : ℝ :=
  let b := A / a
  let d := Real.sqrt (a^2 + b^2)
  d

theorem rectangular_field_diagonal_length :
  let a : ℝ := 14
  let A : ℝ := 135.01111065390137
  abs (diagonal_length_of_rectangular_field a A - 17.002) < 0.001 := by
    sorry

end NUMINAMATH_GPT_rectangular_field_diagonal_length_l1044_104435


namespace NUMINAMATH_GPT_find_point_P_l1044_104475

noncomputable def tangent_at (f : ℝ → ℝ) (x : ℝ) : ℝ := (deriv f) x

theorem find_point_P :
  ∃ (x₀ y₀ : ℝ), (y₀ = (1 / x₀)) 
  ∧ (0 < x₀)
  ∧ (tangent_at (fun x => x^2) 2 = 4)
  ∧ (tangent_at (fun x => (1 / x)) x₀ = -1 / 4) 
  ∧ (x₀ = 2)
  ∧ (y₀ = 1 / 2) :=
sorry

end NUMINAMATH_GPT_find_point_P_l1044_104475


namespace NUMINAMATH_GPT_g_is_even_and_symmetric_l1044_104467

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin (2 * x) - Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (4 * x)

theorem g_is_even_and_symmetric :
  (∀ x : ℝ, g x = g (-x)) ∧ (∀ k : ℤ, g ((2 * k - 1) * π / 8) = 0) :=
by
  sorry

end NUMINAMATH_GPT_g_is_even_and_symmetric_l1044_104467


namespace NUMINAMATH_GPT_proof_correct_props_l1044_104479

variable (p1 : Prop) (p2 : Prop) (p3 : Prop) (p4 : Prop)

def prop1 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ (1 / 2) * x₀ < (1 / 3) * x₀
def prop2 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ x₀ < 1 ∧ Real.log x₀ / Real.log (1 / 2) > Real.log x₀ / Real.log (1 / 3)
def prop3 : Prop := ∀ (x : ℝ), 0 < x ∧ (1 / 2) ^ x > Real.log x / Real.log (1 / 2)
def prop4 : Prop := ∀ (x : ℝ), 0 < x ∧ x < 1 / 3 ∧ (1 / 2) ^ x < Real.log x / Real.log (1 / 3)

theorem proof_correct_props : prop2 ∧ prop4 :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_proof_correct_props_l1044_104479


namespace NUMINAMATH_GPT_notebooks_last_days_l1044_104418

theorem notebooks_last_days (n p u : Nat) (total_pages days : Nat) 
  (h1 : n = 5)
  (h2 : p = 40)
  (h3 : u = 4)
  (h_total : total_pages = n * p)
  (h_days  : days = total_pages / u) :
  days = 50 := 
by
  sorry

end NUMINAMATH_GPT_notebooks_last_days_l1044_104418


namespace NUMINAMATH_GPT_range_of_c_l1044_104432

theorem range_of_c :
  (∃ (c : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = 4) → ((12 * x - 5 * y + c) / 13 = 1))
  → (c > -13 ∧ c < 13) := 
sorry

end NUMINAMATH_GPT_range_of_c_l1044_104432


namespace NUMINAMATH_GPT_find_marks_in_english_l1044_104408

theorem find_marks_in_english 
    (avg : ℕ) (math_marks : ℕ) (physics_marks : ℕ) (chemistry_marks : ℕ) (biology_marks : ℕ) (total_subjects : ℕ)
    (avg_eq : avg = 78) 
    (math_eq : math_marks = 65) 
    (physics_eq : physics_marks = 82) 
    (chemistry_eq : chemistry_marks = 67) 
    (biology_eq : biology_marks = 85) 
    (subjects_eq : total_subjects = 5) : 
    math_marks + physics_marks + chemistry_marks + biology_marks + E = 78 * 5 → 
    E = 91 :=
by sorry

end NUMINAMATH_GPT_find_marks_in_english_l1044_104408


namespace NUMINAMATH_GPT_mark_sideline_time_l1044_104483

def total_game_time : ℕ := 90
def initial_play : ℕ := 20
def second_play : ℕ := 35
def total_play_time : ℕ := initial_play + second_play
def sideline_time : ℕ := total_game_time - total_play_time

theorem mark_sideline_time : sideline_time = 35 := by
  sorry

end NUMINAMATH_GPT_mark_sideline_time_l1044_104483


namespace NUMINAMATH_GPT_all_tell_truth_at_same_time_l1044_104446

-- Define the probabilities of each person telling the truth.
def prob_Alice := 0.7
def prob_Bob := 0.6
def prob_Carol := 0.8
def prob_David := 0.5

-- Prove that the probability that all four tell the truth at the same time is 0.168.
theorem all_tell_truth_at_same_time :
  prob_Alice * prob_Bob * prob_Carol * prob_David = 0.168 :=
by
  sorry

end NUMINAMATH_GPT_all_tell_truth_at_same_time_l1044_104446


namespace NUMINAMATH_GPT_cubic_eq_roots_l1044_104406

theorem cubic_eq_roots (x1 x2 x3 : ℕ) (P : ℕ) 
  (h1 : x1 + x2 + x3 = 10) 
  (h2 : x1 * x2 * x3 = 30) 
  (h3 : x1 * x2 + x2 * x3 + x3 * x1 = P) : 
  P = 31 := by
  sorry

end NUMINAMATH_GPT_cubic_eq_roots_l1044_104406


namespace NUMINAMATH_GPT_marbles_remaining_l1044_104430

def original_marbles : Nat := 64
def given_marbles : Nat := 14
def remaining_marbles : Nat := original_marbles - given_marbles

theorem marbles_remaining : remaining_marbles = 50 :=
  by
    sorry

end NUMINAMATH_GPT_marbles_remaining_l1044_104430


namespace NUMINAMATH_GPT_length_proof_l1044_104427

noncomputable def length_of_plot 
  (b : ℝ) -- breadth in meters
  (fence_cost_flat : ℝ) -- cost of fencing per meter on flat ground
  (height_rise : ℝ) -- total height rise in meters
  (total_cost: ℝ) -- total cost of fencing
  (length_increase : ℝ) -- length increase in meters more than breadth
  (cost_increase_rate : ℝ) -- percentage increase in cost per meter rise in height
  (breadth_cost_increase_factor : ℝ) -- scaling factor for cost increase on breadth
  (increased_breadth_cost_rate : ℝ) -- actual increased cost rate per meter for breadth
: ℝ :=
2 * (b + length_increase) * fence_cost_flat + 
2 * b * (fence_cost_flat + fence_cost_flat * (height_rise * cost_increase_rate))

theorem length_proof
  (b : ℝ) -- breadth in meters
  (fence_cost_flat : ℝ := 26.50) -- cost of fencing per meter on flat ground
  (height_rise : ℝ := 5) -- total height rise in meters
  (total_cost: ℝ := 5300) -- total cost of fencing
  (length_increase : ℝ := 20) -- length increase in meters more than breadth
  (cost_increase_rate : ℝ := 0.10) -- percentage increase in cost per meter rise in height
  (breadth_cost_increase_factor : ℝ := fence_cost_flat * 0.5) -- increased cost factor
  (increased_breadth_cost_rate : ℝ := 39.75) -- recalculated cost rate per meter for breadth
  (length: ℝ := b + length_increase)
  (proof_step : total_cost = length_of_plot b fence_cost_flat height_rise total_cost length_increase cost_increase_rate breadth_cost_increase_factor increased_breadth_cost_rate)
: length = 52 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_length_proof_l1044_104427


namespace NUMINAMATH_GPT_episode_length_l1044_104474

/-- Subject to the conditions provided, we prove the length of each episode watched by Maddie. -/
theorem episode_length
  (total_episodes : ℕ)
  (monday_minutes : ℕ)
  (thursday_minutes : ℕ)
  (weekend_minutes : ℕ)
  (episodes_length : ℕ)
  (monday_watch : monday_minutes = 138)
  (thursday_watch : thursday_minutes = 21)
  (weekend_watch : weekend_minutes = 105)
  (total_episodes_watch : total_episodes = 8)
  (total_minutes : monday_minutes + thursday_minutes + weekend_minutes = total_episodes * episodes_length) :
  episodes_length = 33 := 
by 
  sorry

end NUMINAMATH_GPT_episode_length_l1044_104474


namespace NUMINAMATH_GPT_train_length_is_correct_l1044_104488

noncomputable def speed_of_train_kmph : ℝ := 77.993280537557

noncomputable def speed_of_man_kmph : ℝ := 6

noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * conversion_factor

noncomputable def speed_of_man_mps : ℝ := speed_of_man_kmph * conversion_factor

noncomputable def relative_speed : ℝ := speed_of_train_mps + speed_of_man_mps

noncomputable def time_to_pass_man : ℝ := 6

noncomputable def length_of_train : ℝ := relative_speed * time_to_pass_man

theorem train_length_is_correct : length_of_train = 139.99 := by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l1044_104488


namespace NUMINAMATH_GPT_find_number_l1044_104465

theorem find_number (x : ℝ) (h : 0.85 * x = (4 / 5) * 25 + 14) : x = 40 :=
sorry

end NUMINAMATH_GPT_find_number_l1044_104465


namespace NUMINAMATH_GPT_interior_sum_nine_l1044_104424

-- Defining the function for the sum of the interior numbers in the nth row of Pascal's Triangle
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

-- Given conditions
axiom interior_sum_4 : interior_sum 4 = 6
axiom interior_sum_5 : interior_sum 5 = 14

-- Goal to prove
theorem interior_sum_nine : interior_sum 9 = 254 := by
  sorry

end NUMINAMATH_GPT_interior_sum_nine_l1044_104424


namespace NUMINAMATH_GPT_sum_of_perpendiculars_eq_altitude_l1044_104419

variables {A B C P A' B' C' : Type*}
variables (AB AC BC PA' PB' PC' h : ℝ)

-- Conditions
def is_isosceles_triangle (AB AC BC : ℝ) : Prop :=
  AB = AC

def point_inside_triangle (P A B C : Type*) : Prop :=
  true -- Assume point P is inside the triangle

def is_perpendiculars_dropped (PA' PB' PC' : ℝ) : Prop :=
  true -- Assume PA', PB', PC' are the lengths of the perpendiculars from P to the sides BC, CA, AB

def base_of_triangle (BC : ℝ) : Prop :=
  true -- Assume BC is the base of triangle

-- Theorem statement
theorem sum_of_perpendiculars_eq_altitude
  (h : ℝ) (AB AC BC PA' PB' PC' : ℝ)
  (isosceles : is_isosceles_triangle AB AC BC)
  (point_inside_triangle' : point_inside_triangle P A B C)
  (perpendiculars_dropped : is_perpendiculars_dropped PA' PB' PC')
  (base_of_triangle' : base_of_triangle BC) : 
  PA' + PB' + PC' = h := 
sorry

end NUMINAMATH_GPT_sum_of_perpendiculars_eq_altitude_l1044_104419


namespace NUMINAMATH_GPT_cell_count_at_end_of_days_l1044_104472

-- Defining the conditions
def initial_cells : ℕ := 2
def split_ratio : ℕ := 3
def days : ℕ := 9
def cycle_days : ℕ := 3

-- The main statement to be proved
theorem cell_count_at_end_of_days :
  (initial_cells * split_ratio^((days / cycle_days) - 1)) = 18 :=
by
  sorry

end NUMINAMATH_GPT_cell_count_at_end_of_days_l1044_104472


namespace NUMINAMATH_GPT_find_BC_line_eq_l1044_104434

def line1_altitude : Prop := ∃ x y : ℝ, 2*x - 3*y + 1 = 0
def line2_altitude : Prop := ∃ x y : ℝ, x + y = 0
def vertex_A : Prop := ∃ a1 a2 : ℝ, a1 = 1 ∧ a2 = 2
def side_BC_equation : Prop := ∃ b c d : ℝ, b = 2 ∧ c = 3 ∧ d = 7

theorem find_BC_line_eq (H1 : line1_altitude) (H2 : line2_altitude) (H3 : vertex_A) : side_BC_equation :=
sorry

end NUMINAMATH_GPT_find_BC_line_eq_l1044_104434


namespace NUMINAMATH_GPT_average_lifespan_is_28_l1044_104484

-- Define the given data
def batteryLifespans : List ℕ := [30, 35, 25, 25, 30, 34, 26, 25, 29, 21]

-- Define a function to calculate the average of a list of natural numbers
def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

-- State the theorem to be proved
theorem average_lifespan_is_28 :
  average batteryLifespans = 28 := by
  sorry

end NUMINAMATH_GPT_average_lifespan_is_28_l1044_104484


namespace NUMINAMATH_GPT_lcm_even_numbers_between_14_and_21_l1044_104410

-- Define the even numbers between 14 and 21
def evenNumbers := [14, 16, 18, 20]

-- Define a function to compute the LCM of a list of integers
def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Theorem statement: the LCM of the even numbers between 14 and 21 equals 5040
theorem lcm_even_numbers_between_14_and_21 :
  lcm_list evenNumbers = 5040 :=
by
  sorry

end NUMINAMATH_GPT_lcm_even_numbers_between_14_and_21_l1044_104410


namespace NUMINAMATH_GPT_sam_collected_42_cans_l1044_104489

noncomputable def total_cans_collected (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ) : ℕ :=
  bags_saturday + bags_sunday * cans_per_bag

theorem sam_collected_42_cans :
  total_cans_collected 4 3 6 = 42 :=
by
  sorry

end NUMINAMATH_GPT_sam_collected_42_cans_l1044_104489


namespace NUMINAMATH_GPT_odd_function_expression_l1044_104409

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 3 * x - 4 else - (x^2 - 3 * x - 4)

theorem odd_function_expression (x : ℝ) (h : x < 0) : 
  f x = -x^2 + 3 * x + 4 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_expression_l1044_104409


namespace NUMINAMATH_GPT_sally_eats_sandwiches_l1044_104420

theorem sally_eats_sandwiches
  (saturday_sandwiches : ℕ)
  (bread_per_sandwich : ℕ)
  (total_bread : ℕ)
  (one_sandwich_on_sunday : ℕ)
  (saturday_bread : saturday_sandwiches * bread_per_sandwich = 4)
  (total_bread_consumed : total_bread = 6)
  (bread_on_sundy : bread_per_sandwich = 2) :
  (total_bread - saturday_sandwiches * bread_per_sandwich) / bread_per_sandwich = one_sandwich_on_sunday :=
sorry

end NUMINAMATH_GPT_sally_eats_sandwiches_l1044_104420


namespace NUMINAMATH_GPT_min_value_l1044_104401

theorem min_value (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/29 :=
sorry

end NUMINAMATH_GPT_min_value_l1044_104401


namespace NUMINAMATH_GPT_common_point_of_function_and_inverse_l1044_104470

-- Define the points P, Q, M, and N
def P : ℝ × ℝ := (1, 1)
def Q : ℝ × ℝ := (1, 2)
def M : ℝ × ℝ := (2, 3)
def N : ℝ × ℝ := (0.5, 0.25)

-- Define a predicate to check if a point lies on the line y = x
def lies_on_y_eq_x (point : ℝ × ℝ) : Prop := point.1 = point.2

-- The main theorem statement
theorem common_point_of_function_and_inverse (a : ℝ) : 
  lies_on_y_eq_x P ∧ ¬ lies_on_y_eq_x Q ∧ ¬ lies_on_y_eq_x M ∧ ¬ lies_on_y_eq_x N :=
by
  -- We write 'sorry' here to skip the proof
  sorry

end NUMINAMATH_GPT_common_point_of_function_and_inverse_l1044_104470


namespace NUMINAMATH_GPT_purchasing_plans_and_optimal_plan_l1044_104457

def company_time := 10
def model_A_cost := 60000
def model_B_cost := 40000
def model_A_production := 15
def model_B_production := 10
def budget := 440000
def production_capacity := 102

theorem purchasing_plans_and_optimal_plan (x y : ℕ) (h1 : x + y = company_time) (h2 : model_A_cost * x + model_B_cost * y ≤ budget) :
  (x = 0 ∧ y = 10) ∨ (x = 1 ∧ y = 9) ∨ (x = 2 ∧ y = 8) ∧ (x = 1 ∧ y = 9) :=
by 
  sorry

end NUMINAMATH_GPT_purchasing_plans_and_optimal_plan_l1044_104457


namespace NUMINAMATH_GPT_mod_2_200_sub_3_l1044_104477

theorem mod_2_200_sub_3 (h1 : 2^1 % 7 = 2) (h2 : 2^2 % 7 = 4) (h3 : 2^3 % 7 = 1) : (2^200 - 3) % 7 = 1 := 
by
  sorry

end NUMINAMATH_GPT_mod_2_200_sub_3_l1044_104477


namespace NUMINAMATH_GPT_equation_of_line_l1044_104496

theorem equation_of_line (θ : ℝ) (b : ℝ) (k : ℝ) (y x : ℝ) :
  θ = Real.pi / 4 ∧ b = 2 ∧ k = Real.tan θ ∧ k = 1 ∧ y = k * x + b ↔ y = x + 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_equation_of_line_l1044_104496


namespace NUMINAMATH_GPT_math_proof_problem_l1044_104485

noncomputable def a : ℝ := Real.sqrt 18
noncomputable def b : ℝ := (-1 / 3) ^ (-2 : ℤ)
noncomputable def c : ℝ := abs (-3 * Real.sqrt 2)
noncomputable def d : ℝ := (1 - Real.sqrt 2) ^ 0

theorem math_proof_problem : a - b - c - d = -10 := by
  -- Sorry is used to skip the proof, as the proof steps are not required for this problem.
  sorry

end NUMINAMATH_GPT_math_proof_problem_l1044_104485


namespace NUMINAMATH_GPT_area_ratio_of_squares_l1044_104443

-- Definition of squares, and their perimeters' relationship
def perimeter (side_length : ℝ) := 4 * side_length

theorem area_ratio_of_squares (a b : ℝ) (h : perimeter a = 4 * perimeter b) : (a * a) = 16 * (b * b) :=
by
  -- We assume the given condition
  have ha : a = 4 * b := sorry
  -- We then prove the area ratio
  sorry

end NUMINAMATH_GPT_area_ratio_of_squares_l1044_104443


namespace NUMINAMATH_GPT_tiling_scheme_3_3_3_3_6_l1044_104499

-- Definitions based on the conditions.
def angle_equilateral_triangle := 60
def angle_regular_hexagon := 120

-- The theorem states that using four equilateral triangles and one hexagon around a point forms a valid tiling.
theorem tiling_scheme_3_3_3_3_6 : 
  4 * angle_equilateral_triangle + angle_regular_hexagon = 360 := 
by
  -- Skip the proof with sorry
  sorry

end NUMINAMATH_GPT_tiling_scheme_3_3_3_3_6_l1044_104499


namespace NUMINAMATH_GPT_digit_D_is_five_l1044_104440

variable (A B C D : Nat)
variable (h1 : (B * A) % 10 = A % 10)
variable (h2 : ∀ (C : Nat), B - A = B % 10 ∧ C ≤ A)

theorem digit_D_is_five : D = 5 :=
by
  sorry

end NUMINAMATH_GPT_digit_D_is_five_l1044_104440


namespace NUMINAMATH_GPT_division_remainder_l1044_104492

theorem division_remainder (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (hrem : x % y = 3) (hdiv : (x : ℚ) / y = 96.15) : y = 20 :=
sorry

end NUMINAMATH_GPT_division_remainder_l1044_104492


namespace NUMINAMATH_GPT_documentaries_count_l1044_104456

def number_of_documents
  (novels comics albums crates capacity : ℕ)
  (total_items := crates * capacity)
  (known_items := novels + comics + albums)
  (documentaries := total_items - known_items) : ℕ :=
  documentaries

theorem documentaries_count
  : number_of_documents 145 271 209 116 9 = 419 :=
by
  sorry

end NUMINAMATH_GPT_documentaries_count_l1044_104456


namespace NUMINAMATH_GPT_building_shadow_length_l1044_104431

theorem building_shadow_length
  (flagpole_height : ℝ) (flagpole_shadow : ℝ) (building_height : ℝ)
  (h_flagpole : flagpole_height = 18) (s_flagpole : flagpole_shadow = 45) 
  (h_building : building_height = 26) :
  ∃ (building_shadow : ℝ), (building_height / building_shadow = flagpole_height / flagpole_shadow) ∧ building_shadow = 65 :=
by
  use 65
  sorry

end NUMINAMATH_GPT_building_shadow_length_l1044_104431


namespace NUMINAMATH_GPT_line_passing_through_M_l1044_104459

-- Define the point M
def M : ℝ × ℝ := (-3, 4)

-- Define the predicate for a line equation having equal intercepts and passing through point M
def line_eq (x y : ℝ) (a b : ℝ) : Prop :=
  ∃ c : ℝ, ((a = 0 ∧ b = 0 ∧ 4 * x + 3 * y = 0) ∨ (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ x + y = 1)) 

theorem line_passing_through_M (x y : ℝ) (a b : ℝ) (h₀ : (-3, 4) = M) (h₁ : ∃ c : ℝ, (a = 0 ∧ b = 0 ∧ 4 * x + 3 * y = 0) ∨ (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ x + y = 1)) :
  (4 * x + 3 * y = 0) ∨ (x + y = 1) :=
by
  -- We add 'sorry' to skip the proof
  sorry

end NUMINAMATH_GPT_line_passing_through_M_l1044_104459


namespace NUMINAMATH_GPT_weight_of_B_l1044_104480

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

theorem weight_of_B :
  (A + B + C) / 3 = 45 → 
  (A + B) / 2 = 40 → 
  (B + C) / 2 = 43 → 
  B = 31 :=
by
  intros h1 h2 h3
  -- detailed proof steps omitted
  sorry

end NUMINAMATH_GPT_weight_of_B_l1044_104480


namespace NUMINAMATH_GPT_minimum_days_l1044_104461

theorem minimum_days (n : ℕ) (rain_afternoon : ℕ) (sunny_afternoon : ℕ) (sunny_morning : ℕ) :
  rain_afternoon + sunny_afternoon = 7 ∧
  sunny_afternoon <= 5 ∧
  sunny_morning <= 6 ∧
  sunny_morning + rain_afternoon = 7 ∧
  n = 11 :=
by
  sorry

end NUMINAMATH_GPT_minimum_days_l1044_104461


namespace NUMINAMATH_GPT_basketball_team_first_competition_games_l1044_104481

-- Definitions given the conditions
def first_competition_games (x : ℕ) := x
def second_competition_games (x : ℕ) := (5 * x) / 8
def third_competition_games (x : ℕ) := x + (5 * x) / 8
def total_games (x : ℕ) := x + (5 * x) / 8 + (x + (5 * x) / 8)

-- Lean 4 statement to prove the correct answer
theorem basketball_team_first_competition_games : 
  ∃ x : ℕ, total_games x = 130 ∧ first_competition_games x = 40 :=
by
  sorry

end NUMINAMATH_GPT_basketball_team_first_competition_games_l1044_104481


namespace NUMINAMATH_GPT_librarian_took_books_l1044_104498

-- Define variables and conditions
def total_books : ℕ := 46
def books_per_shelf : ℕ := 4
def shelves_needed : ℕ := 9

-- Define the number of books Oliver has left to put away
def books_left : ℕ := shelves_needed * books_per_shelf

-- Define the number of books the librarian took
def books_taken : ℕ := total_books - books_left

-- State the theorem
theorem librarian_took_books : books_taken = 10 := by
  sorry

end NUMINAMATH_GPT_librarian_took_books_l1044_104498


namespace NUMINAMATH_GPT_part1_part2_l1044_104415

-- Given conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Proof statements to be demonstrated
theorem part1 (a : ℝ) : a = 1 := sorry

theorem part2 (f_inv : ℝ → ℝ) : 
  (∀ x : ℝ, x > -1 ∧ x < 1 → f (f_inv x) = x ∧ f_inv (f x) = x) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1044_104415


namespace NUMINAMATH_GPT_prime_divisor_property_l1044_104417

open Classical

theorem prime_divisor_property (p n q : ℕ) (hp : Nat.Prime p) (hn : 0 < n) (hq : q ∣ (n + 1)^p - n^p) : p ∣ q - 1 :=
by
  sorry

end NUMINAMATH_GPT_prime_divisor_property_l1044_104417


namespace NUMINAMATH_GPT_quadratic_root_condition_l1044_104403

theorem quadratic_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Ici 10 ∪ Set.Iic (-10) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_root_condition_l1044_104403


namespace NUMINAMATH_GPT_sum_in_range_l1044_104458

def a : ℚ := 4 + 1/4
def b : ℚ := 2 + 3/4
def c : ℚ := 7 + 1/8

theorem sum_in_range : 14 < a + b + c ∧ a + b + c < 15 := by
  sorry

end NUMINAMATH_GPT_sum_in_range_l1044_104458


namespace NUMINAMATH_GPT_sum_consecutive_evens_l1044_104495

theorem sum_consecutive_evens (n k : ℕ) (hn : 2 < n) (hk : 2 < k) : 
  ∃ (m : ℕ), n * (n - 1)^(k - 1) = n * (2 * m + (n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_sum_consecutive_evens_l1044_104495


namespace NUMINAMATH_GPT_quadratic_even_coeff_l1044_104413

theorem quadratic_even_coeff (a b c : ℤ) (h₁ : a ≠ 0) (h₂ : ∃ r s : ℚ, r * s + b * r + c = 0) : (a % 2 = 0) ∨ (b % 2 = 0) ∨ (c % 2 = 0) := by
  sorry

end NUMINAMATH_GPT_quadratic_even_coeff_l1044_104413


namespace NUMINAMATH_GPT_no_primes_sum_to_53_l1044_104453

open Nat

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem no_primes_sum_to_53 :
  ¬ ∃ (p q : Nat), p + q = 53 ∧ isPrime p ∧ isPrime q ∧ (p < 30 ∨ q < 30) :=
by
  sorry

end NUMINAMATH_GPT_no_primes_sum_to_53_l1044_104453


namespace NUMINAMATH_GPT_gcd_fact8_fact7_l1044_104438

noncomputable def fact8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
noncomputable def fact7 : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

theorem gcd_fact8_fact7 : Nat.gcd fact8 fact7 = fact7 := by
  unfold fact8 fact7
  exact sorry

end NUMINAMATH_GPT_gcd_fact8_fact7_l1044_104438


namespace NUMINAMATH_GPT_viable_combinations_l1044_104451

-- Given conditions
def totalHerbs : Nat := 4
def totalCrystals : Nat := 6
def incompatibleComb1 : Nat := 2
def incompatibleComb2 : Nat := 1

-- Theorem statement proving the number of viable combinations
theorem viable_combinations : totalHerbs * totalCrystals - (incompatibleComb1 + incompatibleComb2) = 21 := by
  sorry

end NUMINAMATH_GPT_viable_combinations_l1044_104451


namespace NUMINAMATH_GPT_initial_walnut_trees_l1044_104452

/-- 
  Given there are 29 walnut trees in the park after cutting down 13 walnut trees, 
  prove that initially there were 42 walnut trees in the park.
-/
theorem initial_walnut_trees (cut_walnut_trees remaining_walnut_trees initial_walnut_trees : ℕ) 
  (h₁ : cut_walnut_trees = 13)
  (h₂ : remaining_walnut_trees = 29)
  (h₃ : initial_walnut_trees = cut_walnut_trees + remaining_walnut_trees) :
  initial_walnut_trees = 42 := 
sorry

end NUMINAMATH_GPT_initial_walnut_trees_l1044_104452


namespace NUMINAMATH_GPT_clothing_loss_l1044_104433

theorem clothing_loss
  (a : ℝ)
  (h1 : ∃ x y : ℝ, x * 1.25 = a ∧ y * 0.75 = a ∧ x + y - 2 * a = -8) :
  a = 60 :=
sorry

end NUMINAMATH_GPT_clothing_loss_l1044_104433


namespace NUMINAMATH_GPT_yard_area_l1044_104491

theorem yard_area (posts : Nat) (spacing : Real) (longer_factor : Nat) (shorter_side_posts longer_side_posts : Nat)
  (h1 : posts = 24)
  (h2 : spacing = 3)
  (h3 : longer_factor = 3)
  (h4 : 2 * (shorter_side_posts + longer_side_posts) = posts - 4)
  (h5 : longer_side_posts = 3 * shorter_side_posts + 2) :
  (spacing * (shorter_side_posts - 1)) * (spacing * (longer_side_posts - 1)) = 144 :=
by
  sorry

end NUMINAMATH_GPT_yard_area_l1044_104491


namespace NUMINAMATH_GPT_candy_store_food_colouring_amount_l1044_104464

theorem candy_store_food_colouring_amount :
  let lollipop_colour := 5 -- each lollipop uses 5ml of food colouring
  let hard_candy_colour := 20 -- each hard candy uses 20ml of food colouring
  let num_lollipops := 100 -- the candy store makes 100 lollipops in one day
  let num_hard_candies := 5 -- the candy store makes 5 hard candies in one day
  (num_lollipops * lollipop_colour) + (num_hard_candies * hard_candy_colour) = 600 :=
by
  let lollipop_colour := 5
  let hard_candy_colour := 20
  let num_lollipops := 100
  let num_hard_candies := 5
  show (num_lollipops * lollipop_colour) + (num_hard_candies * hard_candy_colour) = 600
  sorry

end NUMINAMATH_GPT_candy_store_food_colouring_amount_l1044_104464


namespace NUMINAMATH_GPT_sheets_in_height_l1044_104429

theorem sheets_in_height (sheets_per_ream : ℕ) (thickness_per_ream : ℝ) (target_thickness : ℝ) 
  (h₀ : sheets_per_ream = 500) (h₁ : thickness_per_ream = 5.0) (h₂ : target_thickness = 7.5) :
  target_thickness / (thickness_per_ream / sheets_per_ream) = 750 :=
by sorry

end NUMINAMATH_GPT_sheets_in_height_l1044_104429


namespace NUMINAMATH_GPT_sufficient_condition_l1044_104497

theorem sufficient_condition (p q r : Prop) (hpq : p → q) (hqr : q → r) : p → r :=
by
  intro hp
  apply hqr
  apply hpq
  exact hp

end NUMINAMATH_GPT_sufficient_condition_l1044_104497


namespace NUMINAMATH_GPT_average_wx_l1044_104441

theorem average_wx (w x a b : ℝ) (i : ℂ) (h_i : i * i = -1)
  (h1 : 6 / w + 6 / x = 6 / (a + b * i))
  (h2 : w * x = a + b * i) :
  (w + x) / 2 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_average_wx_l1044_104441


namespace NUMINAMATH_GPT_initial_birds_on_fence_l1044_104454

theorem initial_birds_on_fence (B S : ℕ) (S_val : S = 2) (total : B + 5 + S = 10) : B = 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_birds_on_fence_l1044_104454


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l1044_104412

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l1044_104412


namespace NUMINAMATH_GPT_unique_root_value_l1044_104494

theorem unique_root_value {x n : ℝ} (h : (15 - n) = 15 - (35 / 4)) :
  (x + 5) * (x + 3) = n + 3 * x → n = 35 / 4 :=
sorry

end NUMINAMATH_GPT_unique_root_value_l1044_104494


namespace NUMINAMATH_GPT_number_of_tricycles_l1044_104463

def num_bicycles : Nat := 24
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3
def total_wheels : Nat := 90

theorem number_of_tricycles : ∃ T : Nat, (wheels_per_bicycle * num_bicycles) + (wheels_per_tricycle * T) = total_wheels ∧ T = 14 := by
  sorry

end NUMINAMATH_GPT_number_of_tricycles_l1044_104463


namespace NUMINAMATH_GPT_functions_are_equal_l1044_104450

-- Define the functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := (x^4)^(1/4)

-- Statement to be proven
theorem functions_are_equal : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_GPT_functions_are_equal_l1044_104450


namespace NUMINAMATH_GPT_luncheon_cost_l1044_104447

section LuncheonCosts

variables (s c p : ℝ)

/- Conditions -/
def eq1 : Prop := 2 * s + 5 * c + 2 * p = 6.25
def eq2 : Prop := 5 * s + 8 * c + 3 * p = 12.10

/- Goal -/
theorem luncheon_cost : eq1 s c p → eq2 s c p → s + c + p = 1.55 :=
by
  intro h1 h2
  sorry

end LuncheonCosts

end NUMINAMATH_GPT_luncheon_cost_l1044_104447


namespace NUMINAMATH_GPT_sum_of_cubes_pattern_l1044_104448

theorem sum_of_cubes_pattern :
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_pattern_l1044_104448


namespace NUMINAMATH_GPT_problem_statement_l1044_104400

noncomputable def f (x : ℝ) := 3 * x ^ 5 + 4 * x ^ 4 - 5 * x ^ 3 + 2 * x ^ 2 + x + 6
noncomputable def d (x : ℝ) := x ^ 3 + 2 * x ^ 2 - x - 3
noncomputable def q (x : ℝ) := 3 * x ^ 2 - 2 * x + 1
noncomputable def r (x : ℝ) := 19 * x ^ 2 - 11 * x - 57

theorem problem_statement : (f 1 = q 1 * d 1 + r 1) ∧ q 1 + r 1 = -47 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1044_104400


namespace NUMINAMATH_GPT_hall_width_l1044_104482

theorem hall_width 
  (L H cost total_expenditure : ℕ)
  (W : ℕ)
  (h1 : L = 20)
  (h2 : H = 5)
  (h3 : cost = 20)
  (h4 : total_expenditure = 19000)
  (h5 : total_expenditure = (L * W + 2 * (H * L) + 2 * (H * W)) * cost) :
  W = 25 := 
sorry

end NUMINAMATH_GPT_hall_width_l1044_104482


namespace NUMINAMATH_GPT_pirate_coins_total_l1044_104455

theorem pirate_coins_total (x : ℕ) (hx : x ≠ 0) (h_paul : ∃ k : ℕ, k = x / 2) (h_pete : ∃ m : ℕ, m = 5 * (x / 2)) 
  (h_ratio : (m : ℝ) = (k : ℝ) * 5) : (x = 4) → 
  ∃ total : ℕ, total = k + m ∧ total = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_pirate_coins_total_l1044_104455


namespace NUMINAMATH_GPT_tina_final_balance_l1044_104422

noncomputable def monthlyIncome : ℝ := 1000
noncomputable def juneBonusRate : ℝ := 0.1
noncomputable def investmentReturnRate : ℝ := 0.05
noncomputable def taxRate : ℝ := 0.1

-- Savings rates
noncomputable def juneSavingsRate : ℝ := 0.25
noncomputable def julySavingsRate : ℝ := 0.20
noncomputable def augustSavingsRate : ℝ := 0.30

-- Expenses
noncomputable def juneRent : ℝ := 200
noncomputable def juneGroceries : ℝ := 100
noncomputable def juneBookRate : ℝ := 0.05

noncomputable def julyRent : ℝ := 250
noncomputable def julyGroceries : ℝ := 150
noncomputable def julyShoesRate : ℝ := 0.15

noncomputable def augustRent : ℝ := 300
noncomputable def augustGroceries : ℝ := 175
noncomputable def augustMiscellaneousRate : ℝ := 0.1

theorem tina_final_balance :
  let juneIncome := monthlyIncome * (1 + juneBonusRate)
  let juneSavings := juneIncome * juneSavingsRate
  let juneExpenses := juneRent + juneGroceries + juneIncome * juneBookRate
  let juneRemaining := juneIncome - juneSavings - juneExpenses

  let julyIncome := monthlyIncome
  let julyInvestmentReturn := juneSavings * investmentReturnRate
  let julyTotalIncome := julyIncome + julyInvestmentReturn
  let julySavings := julyTotalIncome * julySavingsRate
  let julyExpenses := julyRent + julyGroceries + julyIncome * julyShoesRate
  let julyRemaining := julyTotalIncome - julySavings - julyExpenses

  let augustIncome := monthlyIncome
  let augustInvestmentReturn := julySavings * investmentReturnRate
  let augustTotalIncome := augustIncome + augustInvestmentReturn
  let augustSavings := augustTotalIncome * augustSavingsRate
  let augustExpenses := augustRent + augustGroceries + augustIncome * augustMiscellaneousRate
  let augustRemaining := augustTotalIncome - augustSavings - augustExpenses

  let totalInvestmentReturn := julyInvestmentReturn + augustInvestmentReturn
  let totalTaxOnInvestment := totalInvestmentReturn * taxRate

  let finalBalance := juneRemaining + julyRemaining + augustRemaining - totalTaxOnInvestment

  finalBalance = 860.7075 := by
  sorry

end NUMINAMATH_GPT_tina_final_balance_l1044_104422


namespace NUMINAMATH_GPT_find_angle_A_find_perimeter_l1044_104428

noncomputable def cos_rule (b c a : ℝ) (h : b^2 + c^2 - a^2 = b * c) : ℝ :=
(b^2 + c^2 - a^2) / (2 * b * c)

theorem find_angle_A (A B C : ℝ) (a b c : ℝ)
  (h1 : b^2 + c^2 - a^2 = b * c) (hA : cos_rule b c a h1 = 1 / 2) :
  A = Real.arccos (1 / 2) :=
by sorry

theorem find_perimeter (a b c : ℝ)
  (h_a : a = Real.sqrt 2) (hA : Real.sin (Real.arccos (1 / 2))^2 = (Real.sqrt 3 / 2)^2)
  (hBC : Real.sin (Real.arccos (1 / 2))^2 = Real.sin (Real.arccos (1 / 2)) * Real.sin (Real.arccos (1 / 2)))
  (h_bc : b * c = 2)
  (h_bc_eq : b^2 + c^2 - a^2 = b * c) :
  a + b + c = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_find_angle_A_find_perimeter_l1044_104428


namespace NUMINAMATH_GPT_value_depletion_rate_l1044_104404

theorem value_depletion_rate (P F : ℝ) (t : ℝ) (r : ℝ) (h₁ : P = 1100) (h₂ : F = 891) (h₃ : t = 2) (decay_formula : F = P * (1 - r) ^ t) : r = 0.1 :=
by 
  sorry

end NUMINAMATH_GPT_value_depletion_rate_l1044_104404
