import Mathlib

namespace NUMINAMATH_GPT_value_of_f1_plus_g3_l1023_102334

def f (x : ℝ) := 3 * x - 4
def g (x : ℝ) := x + 2

theorem value_of_f1_plus_g3 : f (1 + g 3) = 14 := by
  sorry

end NUMINAMATH_GPT_value_of_f1_plus_g3_l1023_102334


namespace NUMINAMATH_GPT_friends_meeting_time_l1023_102341

noncomputable def speed_B (t : ℕ) : ℝ := 4 + 0.75 * (t - 1)

noncomputable def distance_B (t : ℕ) : ℝ :=
  t * 4 + (0.375 * t * (t - 1))

noncomputable def distance_A (t : ℕ) : ℝ := 5 * t

theorem friends_meeting_time :
  ∃ t : ℝ, 5 * t + (t / 2) * (7.25 + 0.75 * t) = 120 ∧ t = 8 :=
by
  sorry

end NUMINAMATH_GPT_friends_meeting_time_l1023_102341


namespace NUMINAMATH_GPT_free_endpoints_eq_1001_l1023_102349

theorem free_endpoints_eq_1001 : 
  ∃ k : ℕ, 1 + 4 * k = 1001 :=
by {
  sorry
}

end NUMINAMATH_GPT_free_endpoints_eq_1001_l1023_102349


namespace NUMINAMATH_GPT_arc_length_of_circle_l1023_102397

theorem arc_length_of_circle (r θ : ℝ) (h1 : r = 2) (h2 : θ = 5 * Real.pi / 3) : (θ * r) = 10 * Real.pi / 3 :=
by
  rw [h1, h2]
  -- subsequent steps would go here 
  sorry

end NUMINAMATH_GPT_arc_length_of_circle_l1023_102397


namespace NUMINAMATH_GPT_june_ride_time_l1023_102343

theorem june_ride_time (d1 d2 : ℝ) (t1 : ℝ) (rate : ℝ) (t2 : ℝ) :
  d1 = 2 ∧ t1 = 6 ∧ rate = (d1 / t1) ∧ d2 = 5 ∧ t2 = d2 / rate → t2 = 15 := by
  intros h
  sorry

end NUMINAMATH_GPT_june_ride_time_l1023_102343


namespace NUMINAMATH_GPT_Davey_Barbeck_ratio_is_1_l1023_102355

-- Assume the following given conditions as definitions in Lean
variables (guitars Davey Barbeck : ℕ)

-- Condition 1: Davey has 18 guitars
def Davey_has_18 : Prop := Davey = 18

-- Condition 2: Barbeck has the same number of guitars as Davey
def Davey_eq_Barbeck : Prop := Davey = Barbeck

-- The problem statement: Prove the ratio of the number of guitars Davey has to the number of guitars Barbeck has is 1:1
theorem Davey_Barbeck_ratio_is_1 (h1 : Davey_has_18 Davey) (h2 : Davey_eq_Barbeck Davey Barbeck) :
  Davey / Barbeck = 1 :=
by
  sorry

end NUMINAMATH_GPT_Davey_Barbeck_ratio_is_1_l1023_102355


namespace NUMINAMATH_GPT_BoatWorks_total_canoes_by_April_l1023_102378

def BoatWorksCanoes : ℕ → ℕ
| 0 => 5
| (n+1) => 2 * BoatWorksCanoes n

theorem BoatWorks_total_canoes_by_April : (BoatWorksCanoes 0) + (BoatWorksCanoes 1) + (BoatWorksCanoes 2) + (BoatWorksCanoes 3) = 75 :=
by
  sorry

end NUMINAMATH_GPT_BoatWorks_total_canoes_by_April_l1023_102378


namespace NUMINAMATH_GPT_inequality_always_true_l1023_102340

theorem inequality_always_true (a : ℝ) : (∀ x : ℝ, |x - 1| - |x + 2| ≤ a) ↔ 3 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_inequality_always_true_l1023_102340


namespace NUMINAMATH_GPT_scientific_notation_example_l1023_102328

theorem scientific_notation_example :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 218000000 = a * 10 ^ n ∧ a = 2.18 ∧ n = 8 :=
by {
  -- statement of the problem conditions
  sorry
}

end NUMINAMATH_GPT_scientific_notation_example_l1023_102328


namespace NUMINAMATH_GPT_cuboid_can_form_square_projection_l1023_102388

-- Definitions and conditions based directly on the problem
def length1 := 3
def length2 := 4
def length3 := 6

-- Statement to prove
theorem cuboid_can_form_square_projection (x y : ℝ) :
  (4 * x * x + y * y = 36) ∧ (x + y = 4) → True :=
by sorry

end NUMINAMATH_GPT_cuboid_can_form_square_projection_l1023_102388


namespace NUMINAMATH_GPT_number_of_cows_l1023_102336

variable (x y z : ℕ)

theorem number_of_cows (h1 : 4 * x + 2 * y + 2 * z = 24 + 2 * (x + y + z)) (h2 : z = y / 2) : x = 12 := 
sorry

end NUMINAMATH_GPT_number_of_cows_l1023_102336


namespace NUMINAMATH_GPT_fraction_zero_value_x_l1023_102348

theorem fraction_zero_value_x (x : ℝ) (h1 : (x - 2) / (1 - x) = 0) (h2 : 1 - x ≠ 0) : x = 2 := 
sorry

end NUMINAMATH_GPT_fraction_zero_value_x_l1023_102348


namespace NUMINAMATH_GPT_Johnson_family_seating_l1023_102351

theorem Johnson_family_seating : 
  ∃ n : ℕ, number_of_ways_to_seat_Johnson_family = n ∧ n = 288 :=
sorry

end NUMINAMATH_GPT_Johnson_family_seating_l1023_102351


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1023_102342

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l1023_102342


namespace NUMINAMATH_GPT_find_a4_l1023_102314

variables {a : ℕ → ℝ} (q : ℝ) (h_positive : ∀ n, 0 < a n)
variables (h_seq : ∀ n, a (n+1) = q * a n)
variables (h1 : a 1 + (2/3) * a 2 = 3)
variables (h2 : (a 4)^2 = (1/9) * a 3 * a 7)

-- Proof problem statement
theorem find_a4 : a 4 = 27 :=
sorry

end NUMINAMATH_GPT_find_a4_l1023_102314


namespace NUMINAMATH_GPT_largest_divisor_of_10000_not_dividing_9999_l1023_102327

theorem largest_divisor_of_10000_not_dividing_9999 : ∃ d, d ∣ 10000 ∧ ¬ (d ∣ 9999) ∧ ∀ y, (y ∣ 10000 ∧ ¬ (y ∣ 9999)) → y ≤ d := 
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_10000_not_dividing_9999_l1023_102327


namespace NUMINAMATH_GPT_total_envelopes_l1023_102331

def total_stamps : ℕ := 52
def lighter_envelopes : ℕ := 6
def stamps_per_lighter_envelope : ℕ := 2
def stamps_per_heavier_envelope : ℕ := 5

theorem total_envelopes (total_stamps lighter_envelopes stamps_per_lighter_envelope stamps_per_heavier_envelope : ℕ) 
  (h : total_stamps = 52 ∧ lighter_envelopes = 6 ∧ stamps_per_lighter_envelope = 2 ∧ stamps_per_heavier_envelope = 5) : 
  lighter_envelopes + (total_stamps - (stamps_per_lighter_envelope * lighter_envelopes)) / stamps_per_heavier_envelope = 14 :=
by
  sorry

end NUMINAMATH_GPT_total_envelopes_l1023_102331


namespace NUMINAMATH_GPT_sin_alpha_eq_sqrt5_over_3_l1023_102330

theorem sin_alpha_eq_sqrt5_over_3 {α : ℝ} (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end NUMINAMATH_GPT_sin_alpha_eq_sqrt5_over_3_l1023_102330


namespace NUMINAMATH_GPT_length_on_ninth_day_l1023_102307

-- Define relevant variables and conditions.
variables (a1 d : ℕ)

-- Define conditions as hypotheses.
def problem_conditions : Prop :=
  (7 * a1 + 21 * d = 28) ∧ 
  (a1 + d + a1 + 4 * d + a1 + 7 * d = 15)

theorem length_on_ninth_day (h : problem_conditions a1 d) : (a1 + 8 * d = 9) :=
  sorry

end NUMINAMATH_GPT_length_on_ninth_day_l1023_102307


namespace NUMINAMATH_GPT_contrapositive_equiv_l1023_102337

variable (x : Type)

theorem contrapositive_equiv (Q R : x → Prop) :
  (∀ x, Q x → R x) ↔ (∀ x, ¬ (R x) → ¬ (Q x)) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_equiv_l1023_102337


namespace NUMINAMATH_GPT_arithmetic_fraction_subtraction_l1023_102309

theorem arithmetic_fraction_subtraction :
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_fraction_subtraction_l1023_102309


namespace NUMINAMATH_GPT_line_circle_intersect_l1023_102370

theorem line_circle_intersect (m : ℤ) :
  (∃ x y : ℝ, 4 * x + 3 * y + 2 * m = 0 ∧ (x + 3)^2 + (y - 1)^2 = 1) ↔ 2 < m ∧ m < 7 :=
by
  sorry

end NUMINAMATH_GPT_line_circle_intersect_l1023_102370


namespace NUMINAMATH_GPT_new_volume_proof_l1023_102326

variable (r h : ℝ)
variable (π : ℝ := Real.pi) -- Lean's notation for π
variable (original_volume : ℝ := 15) -- given original volume

-- Define original volume of the cylinder
def V := π * r^2 * h

-- Define new volume of the cylinder using new dimensions
def new_V := π * (3 * r)^2 * (2 * h)

-- Prove that new_V is 270 when V = 15
theorem new_volume_proof (hV : V = 15) : new_V = 270 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_new_volume_proof_l1023_102326


namespace NUMINAMATH_GPT_brainiacs_like_neither_l1023_102381

variables 
  (total : ℕ) -- Total number of brainiacs.
  (R : ℕ) -- Number of brainiacs who like rebus teasers.
  (M : ℕ) -- Number of brainiacs who like math teasers.
  (both : ℕ) -- Number of brainiacs who like both rebus and math teasers.
  (math_only : ℕ) -- Number of brainiacs who like only math teasers.

-- Given conditions in the problem
def twice_as_many_rebus : Prop := R = 2 * M
def both_teasers : Prop := both = 18
def math_teasers_not_rebus : Prop := math_only = 20
def total_brainiacs : Prop := total = 100

noncomputable def exclusion_inclusion : ℕ := R + M - both

-- Proof statement: The number of brainiacs who like neither rebus nor math teasers totals to 4
theorem brainiacs_like_neither
  (h_total : total_brainiacs total)
  (h_twice : twice_as_many_rebus R M)
  (h_both : both_teasers both)
  (h_math_only : math_teasers_not_rebus math_only)
  (h_M : M = both + math_only) :
  total - exclusion_inclusion R M both = 4 :=
sorry

end NUMINAMATH_GPT_brainiacs_like_neither_l1023_102381


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1023_102394

def A := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def B := { x : ℝ | -1 < x ∧ x < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1023_102394


namespace NUMINAMATH_GPT_number_of_common_tangents_l1023_102379

/-- Define the circle C1 with center (2, -1) and radius 2. -/
def C1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 4}

/-- Define the symmetry line x + y - 3 = 0. -/
def symmetry_line := {p : ℝ × ℝ | p.1 + p.2 = 3}

/-- Circle C2 is symmetric to C1 about the line x + y = 3. -/
def C2 := {p : ℝ × ℝ | (p.1 - 4)^2 + (p.2 - 1)^2 = 4}

/-- Circle C3 with the given condition MA^2 + MO^2 = 10 for any point M on the circle. 
    A(0, 2) and O is the origin. -/
def C3 := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 4}

/-- The number of common tangents between circle C2 and circle C3 is 3. -/
theorem number_of_common_tangents
  (C1_sym_C2 : ∀ p : ℝ × ℝ, p ∈ C1 ↔ p ∈ C2)
  (M_on_C3 : ∀ M : ℝ × ℝ, M ∈ C3 → ((M.1)^2 + (M.2 - 2)^2) + ((M.1)^2 + (M.2)^2) = 10) :
  ∃ tangents : ℕ, tangents = 3 :=
sorry

end NUMINAMATH_GPT_number_of_common_tangents_l1023_102379


namespace NUMINAMATH_GPT_total_red_beads_l1023_102338

theorem total_red_beads (total_beads : ℕ) (pattern_length : ℕ) (green_beads : ℕ) (red_beads : ℕ) (yellow_beads : ℕ) 
                         (h_total: total_beads = 85) 
                         (h_pattern: pattern_length = green_beads + red_beads + yellow_beads) 
                         (h_cycle: green_beads = 3 ∧ red_beads = 4 ∧ yellow_beads = 1) : 
                         (red_beads * (total_beads / pattern_length)) + (min red_beads (total_beads % pattern_length)) = 42 :=
by
  sorry

end NUMINAMATH_GPT_total_red_beads_l1023_102338


namespace NUMINAMATH_GPT_chuck_team_leads_by_2_l1023_102360

open Nat

noncomputable def chuck_team_score_first_quarter := 9 * 2 + 5 * 1
noncomputable def yellow_team_score_first_quarter := 7 * 2 + 4 * 3

noncomputable def chuck_team_score_second_quarter := 6 * 2 + 3 * 3
noncomputable def yellow_team_score_second_quarter := 5 * 2 + 2 * 3 + 3 * 1

noncomputable def chuck_team_score_third_quarter := 4 * 2 + 2 * 3 + 6 * 1
noncomputable def yellow_team_score_third_quarter := 6 * 2 + 2 * 3

noncomputable def chuck_team_score_fourth_quarter := 8 * 2 + 1 * 3
noncomputable def yellow_team_score_fourth_quarter := 4 * 2 + 3 * 3 + 2 * 1

noncomputable def chuck_team_technical_fouls := 3
noncomputable def yellow_team_technical_fouls := 2

noncomputable def total_chuck_team_score :=
  chuck_team_score_first_quarter + chuck_team_score_second_quarter + 
  chuck_team_score_third_quarter + chuck_team_score_fourth_quarter + 
  chuck_team_technical_fouls

noncomputable def total_yellow_team_score :=
  yellow_team_score_first_quarter + yellow_team_score_second_quarter + 
  yellow_team_score_third_quarter + yellow_team_score_fourth_quarter + 
  yellow_team_technical_fouls

noncomputable def chuck_team_lead :=
  total_chuck_team_score - total_yellow_team_score

theorem chuck_team_leads_by_2 :
  chuck_team_lead = 2 :=
by
  sorry

end NUMINAMATH_GPT_chuck_team_leads_by_2_l1023_102360


namespace NUMINAMATH_GPT_distance_travelled_downstream_in_12_minutes_l1023_102384

noncomputable def speed_boat_still : ℝ := 15 -- in km/hr
noncomputable def rate_current : ℝ := 3 -- in km/hr
noncomputable def time_downstream : ℝ := 12 / 60 -- in hr (since 12 minutes is 12/60 hours)
noncomputable def effective_speed_downstream : ℝ := speed_boat_still + rate_current -- in km/hr
noncomputable def distance_downstream := effective_speed_downstream * time_downstream -- in km

theorem distance_travelled_downstream_in_12_minutes :
  distance_downstream = 3.6 := 
by
  sorry

end NUMINAMATH_GPT_distance_travelled_downstream_in_12_minutes_l1023_102384


namespace NUMINAMATH_GPT_percent_flamingos_among_non_parrots_l1023_102395

theorem percent_flamingos_among_non_parrots
  (total_birds : ℝ) (flamingos : ℝ) (parrots : ℝ) (eagles : ℝ) (owls : ℝ)
  (h_total : total_birds = 100)
  (h_flamingos : flamingos = 40)
  (h_parrots : parrots = 20)
  (h_eagles : eagles = 15)
  (h_owls : owls = 25) :
  ((flamingos / (total_birds - parrots)) * 100 = 50) :=
by sorry

end NUMINAMATH_GPT_percent_flamingos_among_non_parrots_l1023_102395


namespace NUMINAMATH_GPT_change_in_responses_max_min_diff_l1023_102346

open Classical

theorem change_in_responses_max_min_diff :
  let initial_yes := 40
  let initial_no := 40
  let initial_undecided := 20
  let end_yes := 60
  let end_no := 30
  let end_undecided := 10
  let min_change := 20
  let max_change := 80
  max_change - min_change = 60 := by
  intros; sorry

end NUMINAMATH_GPT_change_in_responses_max_min_diff_l1023_102346


namespace NUMINAMATH_GPT_probability_of_red_ball_l1023_102347

theorem probability_of_red_ball (total_balls red_balls black_balls white_balls : ℕ)
  (h1 : total_balls = 7)
  (h2 : red_balls = 2)
  (h3 : black_balls = 4)
  (h4 : white_balls = 1) :
  (red_balls / total_balls : ℚ) = 2 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_of_red_ball_l1023_102347


namespace NUMINAMATH_GPT_fill_time_l1023_102321

def inflow_rate : ℕ := 24 -- gallons per second
def outflow_rate : ℕ := 4 -- gallons per second
def basin_volume : ℕ := 260 -- gallons

theorem fill_time (inflow_rate outflow_rate basin_volume : ℕ) (h₁ : inflow_rate = 24) (h₂ : outflow_rate = 4) 
  (h₃ : basin_volume = 260) : basin_volume / (inflow_rate - outflow_rate) = 13 :=
by
  sorry

end NUMINAMATH_GPT_fill_time_l1023_102321


namespace NUMINAMATH_GPT_solve_for_question_mark_l1023_102329

theorem solve_for_question_mark :
  let question_mark := 4135 / 45
  (45 * question_mark) + (625 / 25) - (300 * 4) = 2950 + (1500 / (75 * 2)) :=
by
  let question_mark := 4135 / 45
  sorry

end NUMINAMATH_GPT_solve_for_question_mark_l1023_102329


namespace NUMINAMATH_GPT_cos_2theta_l1023_102358

theorem cos_2theta (θ : ℝ) (h : Real.tan θ = Real.sqrt 5) : Real.cos (2 * θ) = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_2theta_l1023_102358


namespace NUMINAMATH_GPT_AMHSE_1988_l1023_102339

theorem AMHSE_1988 (x y : ℝ) (h1 : |x| + x + y = 10) (h2 : x + |y| - y = 12) : x + y = 18 / 5 :=
sorry

end NUMINAMATH_GPT_AMHSE_1988_l1023_102339


namespace NUMINAMATH_GPT_birthday_check_value_l1023_102383

theorem birthday_check_value : 
  ∃ C : ℝ, (150 + C) / 4 = C ↔ C = 50 :=
by
  sorry

end NUMINAMATH_GPT_birthday_check_value_l1023_102383


namespace NUMINAMATH_GPT_high_school_sampling_problem_l1023_102356

theorem high_school_sampling_problem :
  let first_year_classes := 20
  let first_year_students_per_class := 50
  let first_year_total_students := first_year_classes * first_year_students_per_class
  let second_year_classes := 24
  let second_year_students_per_class := 45
  let second_year_total_students := second_year_classes * second_year_students_per_class
  let total_students := first_year_total_students + second_year_total_students
  let survey_students := 208
  let first_year_sample := (first_year_total_students * survey_students) / total_students
  let second_year_sample := (second_year_total_students * survey_students) / total_students
  let A_selected_probability := first_year_sample / first_year_total_students
  let B_selected_probability := second_year_sample / second_year_total_students
  (survey_students = 208) →
  (first_year_sample = 100) →
  (second_year_sample = 108) →
  (A_selected_probability = 1 / 10) →
  (B_selected_probability = 1 / 10) →
  (A_selected_probability = B_selected_probability) →
  (student_A_in_first_year : true) →
  (student_B_in_second_year : true) →
  true :=
  by sorry

end NUMINAMATH_GPT_high_school_sampling_problem_l1023_102356


namespace NUMINAMATH_GPT_EmilySixthQuizScore_l1023_102323

theorem EmilySixthQuizScore (x : ℕ) : 
  let scores := [85, 92, 88, 90, 93]
  let total_scores_with_x := scores.sum + x
  let desired_average := 91
  total_scores_with_x = 6 * desired_average → x = 98 := by
  sorry

end NUMINAMATH_GPT_EmilySixthQuizScore_l1023_102323


namespace NUMINAMATH_GPT_determinant_A_l1023_102316

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![  2,  4, -2],
    ![  3, -1,  5],
    ![-1,  3,  2]
  ]

theorem determinant_A : det A = -94 := by
  sorry

end NUMINAMATH_GPT_determinant_A_l1023_102316


namespace NUMINAMATH_GPT_frog_jump_distance_l1023_102353

variable (grasshopper_jump frog_jump mouse_jump : ℕ)
variable (H1 : grasshopper_jump = 19)
variable (H2 : grasshopper_jump = frog_jump + 4)
variable (H3 : mouse_jump = frog_jump - 44)

theorem frog_jump_distance : frog_jump = 15 := by
  sorry

end NUMINAMATH_GPT_frog_jump_distance_l1023_102353


namespace NUMINAMATH_GPT_angle_sum_around_point_l1023_102333

theorem angle_sum_around_point (x : ℝ) (h : 2 * x + 140 = 360) : x = 110 := 
  sorry

end NUMINAMATH_GPT_angle_sum_around_point_l1023_102333


namespace NUMINAMATH_GPT_truncated_trigonal_pyramid_circumscribed_sphere_l1023_102302

theorem truncated_trigonal_pyramid_circumscribed_sphere
  (h R_1 R_2 : ℝ)
  (O_1 T_1 O_2 T_2 : ℝ)
  (circumscribed : ∃ r : ℝ, h = 2 * r)
  (sphere_touches_lower_base : ∀ P, dist P T_1 = r)
  (sphere_touches_upper_base : ∀ Q, dist Q T_2 = r)
  (dist_O1_T1 : ℝ)
  (dist_O2_T2 : ℝ) :
  R_1 * R_2 * h^2 = (R_1^2 - dist_O1_T1^2) * (R_2^2 - dist_O2_T2^2) :=
sorry

end NUMINAMATH_GPT_truncated_trigonal_pyramid_circumscribed_sphere_l1023_102302


namespace NUMINAMATH_GPT_largest_integer_lt_100_with_rem_4_div_7_l1023_102393

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end NUMINAMATH_GPT_largest_integer_lt_100_with_rem_4_div_7_l1023_102393


namespace NUMINAMATH_GPT_boris_possible_amount_l1023_102354

theorem boris_possible_amount (k : ℕ) : ∃ k : ℕ, 1 + 74 * k = 823 :=
by
  use 11
  sorry

end NUMINAMATH_GPT_boris_possible_amount_l1023_102354


namespace NUMINAMATH_GPT_trains_clear_time_l1023_102376

noncomputable def length_train1 : ℝ := 150
noncomputable def length_train2 : ℝ := 165
noncomputable def speed_train1_kmh : ℝ := 80
noncomputable def speed_train2_kmh : ℝ := 65
noncomputable def kmh_to_mps (v : ℝ) : ℝ := v * (5/18)
noncomputable def speed_train1 : ℝ := kmh_to_mps speed_train1_kmh
noncomputable def speed_train2 : ℝ := kmh_to_mps speed_train2_kmh
noncomputable def total_distance : ℝ := length_train1 + length_train2
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

theorem trains_clear_time : time_to_clear = 7.82 := 
sorry

end NUMINAMATH_GPT_trains_clear_time_l1023_102376


namespace NUMINAMATH_GPT_irrational_roots_of_odd_quadratic_l1023_102398

theorem irrational_roots_of_odd_quadratic (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ p q : ℤ, q ≠ 0 ∧ gcd p q = 1 ∧ p * p = a * (p / q) * (p / q) + b * (p / q) + c := sorry

end NUMINAMATH_GPT_irrational_roots_of_odd_quadratic_l1023_102398


namespace NUMINAMATH_GPT_painting_together_time_l1023_102311

theorem painting_together_time (jamshid_time taimour_time time_together : ℝ) 
  (h1 : jamshid_time = taimour_time / 2)
  (h2 : taimour_time = 21)
  (h3 : time_together = 7) :
  (1 / taimour_time + 1 / jamshid_time) * time_together = 1 := 
sorry

end NUMINAMATH_GPT_painting_together_time_l1023_102311


namespace NUMINAMATH_GPT_find_divisor_exists_four_numbers_in_range_l1023_102396

theorem find_divisor_exists_four_numbers_in_range :
  ∃ n : ℕ, (n > 1) ∧ (∀ k : ℕ, 39 ≤ k ∧ k ≤ 79 → ∃ a : ℕ, k = n * a) ∧ (∃! (k₁ k₂ k₃ k₄ : ℕ), 39 ≤ k₁ ∧ k₁ ≤ 79 ∧ 39 ≤ k₂ ∧ k₂ ≤ 79 ∧ 39 ≤ k₃ ∧ k₃ ≤ 79 ∧ 39 ≤ k₄ ∧ k₄ ≤ 79 ∧ k₁ ≠ k₂ ∧ k₁ ≠ k₃ ∧ k₁ ≠ k₄ ∧ k₂ ≠ k₃ ∧ k₂ ≠ k₄ ∧ k₃ ≠ k₄ ∧ k₁ % n = 0 ∧ k₂ % n = 0 ∧ k₃ % n = 0 ∧ k₄ % n = 0) → n = 19 :=
by sorry

end NUMINAMATH_GPT_find_divisor_exists_four_numbers_in_range_l1023_102396


namespace NUMINAMATH_GPT_gcd_f100_f101_l1023_102362

def f (x : ℤ) : ℤ := x^2 - 3 * x + 2023

theorem gcd_f100_f101 : Int.gcd (f 100) (f 101) = 2 :=
by
  sorry

end NUMINAMATH_GPT_gcd_f100_f101_l1023_102362


namespace NUMINAMATH_GPT_largest_k_consecutive_sum_l1023_102319

theorem largest_k_consecutive_sum (k : ℕ) (h1 : (∃ n : ℕ, 3^12 = k * n + (k*(k-1))/2)) : k ≤ 729 :=
by
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_largest_k_consecutive_sum_l1023_102319


namespace NUMINAMATH_GPT_customer_bought_two_pens_l1023_102368

noncomputable def combination (n k : ℕ) : ℝ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem customer_bought_two_pens :
  ∃ n : ℕ, combination 5 n / combination 8 n = 0.3571428571428571 ↔ n = 2 := by
  sorry

end NUMINAMATH_GPT_customer_bought_two_pens_l1023_102368


namespace NUMINAMATH_GPT_disproving_equation_l1023_102374

theorem disproving_equation 
  (a b c d : ℚ)
  (h : a / b = c / d)
  (ha : a ≠ 0)
  (hc : c ≠ 0) : 
  a + d ≠ (a / b) * (b + c) := 
by 
  sorry

end NUMINAMATH_GPT_disproving_equation_l1023_102374


namespace NUMINAMATH_GPT_no_solutions_for_specific_a_l1023_102366

theorem no_solutions_for_specific_a (a : ℝ) :
  (a < -9) ∨ (a > 0) →
  ¬ ∃ x : ℝ, 5 * |x - 4 * a| + |x - a^2| + 4 * x - 3 * a = 0 :=
by sorry

end NUMINAMATH_GPT_no_solutions_for_specific_a_l1023_102366


namespace NUMINAMATH_GPT_find_primes_pqr_eq_5_sum_l1023_102322

theorem find_primes_pqr_eq_5_sum (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) :
  p * q * r = 5 * (p + q + r) → (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 7 ∧ r = 5) ∨
                                         (p = 5 ∧ q = 2 ∧ r = 7) ∨ (p = 5 ∧ q = 7 ∧ r = 2) ∨
                                         (p = 7 ∧ q = 2 ∧ r = 5) ∨ (p = 7 ∧ q = 5 ∧ r = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_primes_pqr_eq_5_sum_l1023_102322


namespace NUMINAMATH_GPT_largest_allowed_set_size_correct_l1023_102312

noncomputable def largest_allowed_set_size (N : ℕ) : ℕ :=
  N - Nat.floor (N / 4)

def is_allowed (S : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (a ∣ b → b ∣ c → False)

theorem largest_allowed_set_size_correct (N : ℕ) (hN : 0 < N) : 
  ∃ S : Finset ℕ, is_allowed S ∧ S.card = largest_allowed_set_size N := sorry

end NUMINAMATH_GPT_largest_allowed_set_size_correct_l1023_102312


namespace NUMINAMATH_GPT_john_steps_l1023_102310

/-- John climbs up 9 flights of stairs. Each flight is 10 feet. -/
def flights := 9
def flight_height_feet := 10

/-- Conversion factor between feet and inches. -/
def feet_to_inches := 12

/-- Each step is 18 inches. -/
def step_height_inches := 18

/-- The total number of steps John climbs. -/
theorem john_steps :
  (flights * flight_height_feet * feet_to_inches) / step_height_inches = 60 :=
by
  sorry

end NUMINAMATH_GPT_john_steps_l1023_102310


namespace NUMINAMATH_GPT_first_grade_sample_count_l1023_102301

-- Defining the total number of students and their ratio in grades 1, 2, and 3.
def total_students : ℕ := 2400
def ratio_grade1 : ℕ := 5
def ratio_grade2 : ℕ := 4
def ratio_grade3 : ℕ := 3
def total_ratio := ratio_grade1 + ratio_grade2 + ratio_grade3

-- Defining the sample size
def sample_size : ℕ := 120

-- Proving that the number of first-grade students sampled should be 50.
theorem first_grade_sample_count : 
  (sample_size * ratio_grade1) / total_ratio = 50 :=
by
  -- sorry is added here to skip the proof
  sorry

end NUMINAMATH_GPT_first_grade_sample_count_l1023_102301


namespace NUMINAMATH_GPT_quadratic_eq_with_given_roots_l1023_102371

theorem quadratic_eq_with_given_roots (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 12) :
    (a + b = 16) ∧ (a * b = 144) ∧ (∀ (x : ℝ), x^2 - (a + b) * x + (a * b) = 0 ↔ x^2 - 16 * x + 144 = 0) := by
  sorry

end NUMINAMATH_GPT_quadratic_eq_with_given_roots_l1023_102371


namespace NUMINAMATH_GPT_sum_of_three_different_squares_l1023_102382

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def existing_list (ns : List Nat) : Prop :=
  ∀ n ∈ ns, is_perfect_square n

theorem sum_of_three_different_squares (a b c : Nat) :
  existing_list [a, b, c] →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 128 →
  false :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_three_different_squares_l1023_102382


namespace NUMINAMATH_GPT_circus_tent_capacity_l1023_102391

theorem circus_tent_capacity (num_sections : ℕ) (people_per_section : ℕ) 
  (h1 : num_sections = 4) (h2 : people_per_section = 246) :
  num_sections * people_per_section = 984 :=
by
  sorry

end NUMINAMATH_GPT_circus_tent_capacity_l1023_102391


namespace NUMINAMATH_GPT_min_value_f_min_achieved_l1023_102361

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 3)) + x

theorem min_value_f : ∀ x : ℝ, x > 3 → f x ≥ 5 :=
by
  intro x hx
  sorry

theorem min_achieved : f 4 = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_min_achieved_l1023_102361


namespace NUMINAMATH_GPT_find_p_l1023_102350

theorem find_p (a : ℕ) (ha : a = 2030) : 
  let p := 2 * a + 1;
  let q := a * (a + 1);
  p = 4061 ∧ Nat.gcd p q = 1 := by
  sorry

end NUMINAMATH_GPT_find_p_l1023_102350


namespace NUMINAMATH_GPT_rachel_budget_proof_l1023_102399

-- Define the prices Sara paid for shoes and the dress
def shoes_price : ℕ := 50
def dress_price : ℕ := 200

-- Total amount Sara spent
def sara_total : ℕ := shoes_price + dress_price

-- Rachel's budget should be double of Sara's total spending
def rachels_budget : ℕ := 2 * sara_total

-- The theorem statement
theorem rachel_budget_proof : rachels_budget = 500 := by
  unfold rachels_budget sara_total shoes_price dress_price
  rfl

end NUMINAMATH_GPT_rachel_budget_proof_l1023_102399


namespace NUMINAMATH_GPT_five_person_lineup_l1023_102369

theorem five_person_lineup : 
  let total_ways := Nat.factorial 5
  let invalid_first := Nat.factorial 4
  let invalid_last := Nat.factorial 4
  let valid_ways := total_ways - (invalid_first + invalid_last)
  valid_ways = 72 :=
by
  sorry

end NUMINAMATH_GPT_five_person_lineup_l1023_102369


namespace NUMINAMATH_GPT_find_m_l1023_102375

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 + 3 * x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + (m + 1) * x + m = 0}

theorem find_m (m : ℝ) : B m ⊆ A → (m = 1 ∨ m = 2) :=
sorry

end NUMINAMATH_GPT_find_m_l1023_102375


namespace NUMINAMATH_GPT_fraction_sum_l1023_102367

variable (a b : ℝ)

theorem fraction_sum
  (hb : b + 1 ≠ 0) :
  (a / (b + 1)) + (2 * a / (b + 1)) - (3 * a / (b + 1)) = 0 :=
by sorry

end NUMINAMATH_GPT_fraction_sum_l1023_102367


namespace NUMINAMATH_GPT_percentage_reduction_is_10_percent_l1023_102386

-- Definitions based on the given conditions
def rooms_rented_for_40 : ℕ := sorry
def rooms_rented_for_60 : ℕ := sorry
def total_rent : ℕ := 2000
def rent_per_room_40 : ℕ := 40
def rent_per_room_60 : ℕ := 60
def rooms_switch_count : ℕ := 10

-- Define the hypothetical new total if the rooms were rented at different rates
def new_total_rent : ℕ := (rent_per_room_40 * (rooms_rented_for_40 + rooms_switch_count)) + (rent_per_room_60 * (rooms_rented_for_60 - rooms_switch_count))

-- Calculate the percentage reduction
noncomputable def percentage_reduction : ℝ := (((total_rent: ℝ) - (new_total_rent: ℝ)) / (total_rent: ℝ)) * 100

-- Statement to prove
theorem percentage_reduction_is_10_percent : percentage_reduction = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_reduction_is_10_percent_l1023_102386


namespace NUMINAMATH_GPT_sqrt_factorial_product_l1023_102313

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end NUMINAMATH_GPT_sqrt_factorial_product_l1023_102313


namespace NUMINAMATH_GPT_sum_of_factors_eq_12_l1023_102305

-- Define the polynomial for n = 1
def poly (x : ℤ) : ℤ := x^5 + x + 1

-- Define the two factors when x = 2
def factor1 (x : ℤ) : ℤ := x^3 - x^2 + 1
def factor2 (x : ℤ) : ℤ := x^2 + x + 1

-- State the sum of the two factors at x = 2 equals 12
theorem sum_of_factors_eq_12 (x : ℤ) (h : x = 2) : factor1 x + factor2 x = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_factors_eq_12_l1023_102305


namespace NUMINAMATH_GPT_jindra_initial_dice_count_l1023_102324

-- Given conditions about the dice stacking
def number_of_dice_per_layer : ℕ := 36
def layers_stacked_completely : ℕ := 6
def dice_received : ℕ := 18

-- We need to prove that the initial number of dice Jindra had is 234
theorem jindra_initial_dice_count : 
    (layers_stacked_completely * number_of_dice_per_layer + dice_received) = 234 :=
    by 
        sorry

end NUMINAMATH_GPT_jindra_initial_dice_count_l1023_102324


namespace NUMINAMATH_GPT_simplify_expression_l1023_102373

theorem simplify_expression :
  (512 : ℝ)^(1/4) * (343 : ℝ)^(1/2) = 28 * (14 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1023_102373


namespace NUMINAMATH_GPT_digit_divisibility_l1023_102364

theorem digit_divisibility : 
  (∃ (A : ℕ), A < 10 ∧ 
   (4573198080 + A) % 2 = 0 ∧ 
   (4573198080 + A) % 5 = 0 ∧ 
   (4573198080 + A) % 8 = 0 ∧ 
   (4573198080 + A) % 10 = 0 ∧ 
   (4573198080 + A) % 16 = 0 ∧ A = 0) := 
by { use 0; sorry }

end NUMINAMATH_GPT_digit_divisibility_l1023_102364


namespace NUMINAMATH_GPT_range_of_x_range_of_a_l1023_102380

variable (a x : ℝ)

-- Define proposition p: x^2 - 3ax + 2a^2 < 0
def p (a x : ℝ) : Prop := x^2 - 3 * a * x + 2 * a^2 < 0

-- Define proposition q: x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- First theorem: Prove the range of x when a = 2 and p ∨ q is true
theorem range_of_x (h : p 2 x ∨ q x) : 2 < x ∧ x < 4 := 
by sorry

-- Second theorem: Prove the range of a when ¬p is necessary but not sufficient for ¬q
theorem range_of_a (h : ∀ x, q x → p a x) : 3/2 ≤ a ∧ a ≤ 2 := 
by sorry

end NUMINAMATH_GPT_range_of_x_range_of_a_l1023_102380


namespace NUMINAMATH_GPT_remainder_when_divided_by_product_l1023_102308

noncomputable def Q : Polynomial ℝ := sorry

theorem remainder_when_divided_by_product (Q : Polynomial ℝ)
    (h1 : Q.eval 20 = 100)
    (h2 : Q.eval 100 = 20) :
    ∃ R : Polynomial ℝ, ∃ a b : ℝ, Q = (Polynomial.X - 20) * (Polynomial.X - 100) * R + Polynomial.C a * Polynomial.X + Polynomial.C b ∧
    a = -1 ∧ b = 120 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_product_l1023_102308


namespace NUMINAMATH_GPT_eval_f_at_3_l1023_102332

-- Define the polynomial function
def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

-- State the theorem to prove f(3) = 41
theorem eval_f_at_3 : f 3 = 41 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_eval_f_at_3_l1023_102332


namespace NUMINAMATH_GPT_find_a5_l1023_102357

-- Define the sequence and its properties
def geom_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a (n + m) = (2^m) * a n

-- Define the problem statement
def sum_of_first_five_terms_is_31 (a : ℕ → ℕ) : Prop :=
a 1 + a 2 + a 3 + a 4 + a 5 = 31

-- State the theorem to prove
theorem find_a5 (a : ℕ → ℕ) (h_geom : geom_sequence a) (h_sum : sum_of_first_five_terms_is_31 a) : a 5 = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_a5_l1023_102357


namespace NUMINAMATH_GPT_total_surface_area_of_cubes_aligned_side_by_side_is_900_l1023_102390

theorem total_surface_area_of_cubes_aligned_side_by_side_is_900 :
  let volumes := [27, 64, 125, 216, 512]
  let side_lengths := volumes.map (fun v => v^(1/3))
  let surface_areas := side_lengths.map (fun s => 6 * s^2)
  (surface_areas.sum = 900) :=
by
  sorry

end NUMINAMATH_GPT_total_surface_area_of_cubes_aligned_side_by_side_is_900_l1023_102390


namespace NUMINAMATH_GPT_common_difference_is_half_l1023_102317

variable (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ) (q p : ℕ)

-- Conditions
def condition1 : Prop := a p = 4
def condition2 : Prop := a q = 2
def condition3 : Prop := p = 4 + q
def arithmetic_sequence : Prop := ∀ n : ℕ, a n = a₁ + (n - 1) * d

-- Proof statement
theorem common_difference_is_half 
  (h1 : condition1 a p)
  (h2 : condition2 a q)
  (h3 : condition3 p q)
  (as : arithmetic_sequence a a₁ d)
  : d = 1 / 2 := 
sorry

end NUMINAMATH_GPT_common_difference_is_half_l1023_102317


namespace NUMINAMATH_GPT_abs_eq_two_iff_l1023_102389

theorem abs_eq_two_iff (a : ℝ) : |a| = 2 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_two_iff_l1023_102389


namespace NUMINAMATH_GPT_problem_solution_l1023_102306

theorem problem_solution (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (h : x - y = x / y) : 
  (1 / x - 1 / y = -1 / y^2) := 
by sorry

end NUMINAMATH_GPT_problem_solution_l1023_102306


namespace NUMINAMATH_GPT_no_positive_numbers_satisfy_conditions_l1023_102320

theorem no_positive_numbers_satisfy_conditions :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c = ab + ac + bc) ∧ (ab + ac + bc = abc) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_numbers_satisfy_conditions_l1023_102320


namespace NUMINAMATH_GPT_sum_first_10_terms_l1023_102335

-- Define the conditions for the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def arithmetic_sequence (b c d : ℝ) : Prop :=
  2 * c = b + d

def conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 1 ∧
  geometric_sequence a q ∧
  arithmetic_sequence (4 * a 1) (2 * a 2) (a 3)

-- Define the sum of the first n terms of a geometric sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Prove the final result
theorem sum_first_10_terms (a : ℕ → ℝ) (q : ℝ) (h : conditions a q) :
  sum_first_n_terms a 10 = 1023 :=
sorry

end NUMINAMATH_GPT_sum_first_10_terms_l1023_102335


namespace NUMINAMATH_GPT_calories_per_orange_is_correct_l1023_102325

noncomputable def calories_per_orange
  (oranges pieces_per_orange num_people calories_per_person : ℕ)
  (h_oranges : oranges = 5)
  (h_pieces_per_orange : pieces_per_orange = 8)
  (h_num_people : num_people = 4)
  (h_calories_per_person : calories_per_person = 100) : ℕ :=
by
  -- Definitions derived from conditions
  let total_pieces := oranges * pieces_per_orange
  let pieces_per_person := total_pieces / num_people
  let total_calories := calories_per_person
  have calories_per_piece := total_calories / pieces_per_person

  -- Conclusion
  have calories_per_orange := pieces_per_orange * calories_per_piece
  exact calories_per_orange

theorem calories_per_orange_is_correct
  (oranges pieces_per_orange num_people calories_per_person : ℕ)
  (h_oranges : oranges = 5)
  (h_pieces_per_orange : pieces_per_orange = 8)
  (h_num_people : num_people = 4)
  (h_calories_per_person : calories_per_person = 100) :
  calories_per_orange oranges pieces_per_orange num_people calories_per_person
    h_oranges h_pieces_per_orange h_num_people h_calories_per_person = 100 :=
by
  simp [calories_per_orange]
  sorry  -- Proof omitted

end NUMINAMATH_GPT_calories_per_orange_is_correct_l1023_102325


namespace NUMINAMATH_GPT_hapok_max_coins_l1023_102300

/-- The maximum number of coins Hapok can guarantee himself regardless of Glazok's actions is 46 coins. -/
theorem hapok_max_coins (total_coins : ℕ) (max_handfuls : ℕ) (coins_per_handful : ℕ) :
  total_coins = 100 ∧ max_handfuls = 9 ∧ (∀ h : ℕ, h ≤ max_handfuls) ∧ coins_per_handful ≤ total_coins →
  ∃ k : ℕ, k ≤ total_coins ∧ k = 46 :=
by {
  sorry
}

end NUMINAMATH_GPT_hapok_max_coins_l1023_102300


namespace NUMINAMATH_GPT_initial_chips_in_bag_l1023_102359

-- Definitions based on conditions
def chips_given_to_brother : ℕ := 7
def chips_given_to_sister : ℕ := 5
def chips_kept_by_nancy : ℕ := 10

-- Theorem statement
theorem initial_chips_in_bag (total_chips := chips_given_to_brother + chips_given_to_sister + chips_kept_by_nancy) : total_chips = 22 := 
by 
  -- we state the assertion
  sorry

end NUMINAMATH_GPT_initial_chips_in_bag_l1023_102359


namespace NUMINAMATH_GPT_slices_leftover_l1023_102318

def total_slices (small_pizzas large_pizzas : ℕ) : ℕ :=
  (3 * 4) + (2 * 8)

def slices_eaten_by_people (george bob susie bill fred mark : ℕ) : ℕ :=
  george + bob + susie + bill + fred + mark

theorem slices_leftover :
  total_slices 3 2 - slices_eaten_by_people 3 4 2 3 3 3 = 10 :=
by sorry

end NUMINAMATH_GPT_slices_leftover_l1023_102318


namespace NUMINAMATH_GPT_decreased_price_correct_l1023_102363

def actual_cost : ℝ := 250
def percentage_decrease : ℝ := 0.2

theorem decreased_price_correct : actual_cost - (percentage_decrease * actual_cost) = 200 :=
by
  sorry

end NUMINAMATH_GPT_decreased_price_correct_l1023_102363


namespace NUMINAMATH_GPT_problem1_problem2_1_problem2_2_l1023_102377

-- Define the quadratic function and conditions
def quadratic (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c

-- Problem 1: Expression of the quadratic function given vertex
theorem problem1 (b c : ℝ) : (quadratic 2 b c = 0) ∧ (∀ x : ℝ, quadratic x b c = (x - 2)^2) ↔ (b = -4) ∧ (c = 4) := sorry

-- Problem 2.1: Given n < -5 and y1 = y2, range of b + c
theorem problem2_1 (n y1 y2 b c : ℝ) (h1 : n < -5) (h2 : quadratic (3*n - 4) b c = y1)
  (h3 : quadratic (5*n + 6) b c = y2) (h4 : y1 = y2) : b + c < -38 := sorry

-- Problem 2.2: Given n < -5 and c > 0, compare values of y1 and y2
theorem problem2_2 (n y1 y2 b c : ℝ) (h1 : n < -5) (h2 : c > 0) 
  (h3 : quadratic (3*n - 4) b c = y1) (h4 : quadratic (5*n + 6) b c = y2) : y1 < y2 := sorry

end NUMINAMATH_GPT_problem1_problem2_1_problem2_2_l1023_102377


namespace NUMINAMATH_GPT_benny_eggs_l1023_102385

def dozen := 12

def total_eggs (n: Nat) := n * dozen

theorem benny_eggs:
  total_eggs 7 = 84 := 
by 
  sorry

end NUMINAMATH_GPT_benny_eggs_l1023_102385


namespace NUMINAMATH_GPT_train_length_l1023_102344

theorem train_length
  (t1 : ℕ) (t2 : ℕ)
  (d_platform : ℕ)
  (h1 : t1 = 8)
  (h2 : t2 = 20)
  (h3 : d_platform = 279)
  : ∃ (L : ℕ), (L : ℕ) = 186 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1023_102344


namespace NUMINAMATH_GPT_profit_percentage_l1023_102365

theorem profit_percentage (cost_price selling_price profit_percentage : ℚ) 
  (h_cost_price : cost_price = 240) 
  (h_selling_price : selling_price = 288) 
  (h_profit_percentage : profit_percentage = 20) : 
  profit_percentage = ((selling_price - cost_price) / cost_price) * 100 := 
by 
  sorry

end NUMINAMATH_GPT_profit_percentage_l1023_102365


namespace NUMINAMATH_GPT_xyz_value_l1023_102352

theorem xyz_value (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 21) :
  x * y * z = 28 / 3 :=
by
  sorry

end NUMINAMATH_GPT_xyz_value_l1023_102352


namespace NUMINAMATH_GPT_journey_ratio_proof_l1023_102345

def journey_ratio (x y : ℝ) : Prop :=
  (x + y = 448) ∧ (x / 21 + y / 24 = 20) → (x / y = 1)

theorem journey_ratio_proof : ∃ x y : ℝ, journey_ratio x y :=
by
  sorry

end NUMINAMATH_GPT_journey_ratio_proof_l1023_102345


namespace NUMINAMATH_GPT_total_metal_wasted_l1023_102387

noncomputable def wasted_metal (a b : ℝ) (h : b ≤ 2 * a) : ℝ := 
  2 * a * b - (b ^ 2 / 2)

theorem total_metal_wasted (a b : ℝ) (h : b ≤ 2 * a) : 
  wasted_metal a b h = 2 * a * b - b ^ 2 / 2 :=
sorry

end NUMINAMATH_GPT_total_metal_wasted_l1023_102387


namespace NUMINAMATH_GPT_car_speed_l1023_102392

theorem car_speed
  (v : ℝ)       -- the unknown speed of the car in km/hr
  (time_80 : ℝ := 45)  -- the time in seconds to travel 1 km at 80 km/hr
  (time_plus_10 : ℝ := 55)  -- the time in seconds to travel 1 km at speed v

  (h1 : time_80 = 3600 / 80)
  (h2 : time_plus_10 = time_80 + 10) :
  v = 3600 / (55 / 3600) := sorry

end NUMINAMATH_GPT_car_speed_l1023_102392


namespace NUMINAMATH_GPT_range_of_k_l1023_102303

-- Definitions for the conditions of p and q
def is_ellipse (k : ℝ) : Prop := (0 < k) ∧ (k < 4)
def is_hyperbola (k : ℝ) : Prop := 1 < k ∧ k < 3

-- The main proposition
theorem range_of_k (k : ℝ) : (is_ellipse k ∨ is_hyperbola k) → (1 < k ∧ k < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1023_102303


namespace NUMINAMATH_GPT_degrees_to_radians_216_l1023_102304

theorem degrees_to_radians_216 : (216 / 180 : ℝ) * Real.pi = (6 / 5 : ℝ) * Real.pi := by
  sorry

end NUMINAMATH_GPT_degrees_to_radians_216_l1023_102304


namespace NUMINAMATH_GPT_average_price_correct_l1023_102315

-- Define the conditions
def books_shop1 : ℕ := 65
def price_shop1 : ℕ := 1480
def books_shop2 : ℕ := 55
def price_shop2 : ℕ := 920

-- Define the total books and total price based on conditions
def total_books : ℕ := books_shop1 + books_shop2
def total_price : ℕ := price_shop1 + price_shop2

-- Define the average price based on total books and total price
def average_price : ℕ := total_price / total_books

-- Theorem stating the average price per book Sandy paid
theorem average_price_correct : average_price = 20 :=
  by
  sorry

end NUMINAMATH_GPT_average_price_correct_l1023_102315


namespace NUMINAMATH_GPT_fifth_graders_more_than_eighth_graders_l1023_102372

theorem fifth_graders_more_than_eighth_graders 
  (cost : ℕ) 
  (h_cost : cost > 0) 
  (h_div_234 : 234 % cost = 0) 
  (h_div_312 : 312 % cost = 0) 
  (h_40_fifth_graders : 40 > 0) : 
  (312 / cost) - (234 / cost) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_fifth_graders_more_than_eighth_graders_l1023_102372
