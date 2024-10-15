import Mathlib

namespace NUMINAMATH_GPT_megan_popsicles_l1257_125792

variable (t_rate : ℕ) (t_hours : ℕ)

def popsicles_eaten (rate: ℕ) (hours: ℕ) : ℕ :=
  60 * hours / rate

theorem megan_popsicles : popsicles_eaten 20 5 = 15 := by
  sorry

end NUMINAMATH_GPT_megan_popsicles_l1257_125792


namespace NUMINAMATH_GPT_hyperbola_asymptote_l1257_125718

theorem hyperbola_asymptote (y x : ℝ) :
  (y^2 / 9 - x^2 / 16 = 1) → (y = x * 3 / 4 ∨ y = -x * 3 / 4) :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l1257_125718


namespace NUMINAMATH_GPT_trig_identity_l1257_125781

open Real

theorem trig_identity :
  (1 - 1 / cos (23 * π / 180)) *
  (1 + 1 / sin (67 * π / 180)) *
  (1 - 1 / sin (23 * π / 180)) * 
  (1 + 1 / cos (67 * π / 180)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1257_125781


namespace NUMINAMATH_GPT_can_form_all_numbers_l1257_125776

noncomputable def domino_tiles : List (ℕ × ℕ) := [(1, 3), (6, 6), (6, 2), (3, 2)]

def form_any_number (n : ℕ) : Prop :=
  ∃ (comb : List (ℕ × ℕ)), comb ⊆ domino_tiles ∧ (comb.bind (λ p => [p.1, p.2])).sum = n

theorem can_form_all_numbers : ∀ n, 1 ≤ n → n ≤ 23 → form_any_number n :=
by sorry

end NUMINAMATH_GPT_can_form_all_numbers_l1257_125776


namespace NUMINAMATH_GPT_stewart_farm_horse_food_l1257_125784

def sheep_to_horse_ratio := 3 / 7
def horses_needed (sheep : ℕ) := (sheep * 7) / 3 
def daily_food_per_horse := 230
def sheep_count := 24
def total_horses := horses_needed sheep_count
def total_daily_horse_food := total_horses * daily_food_per_horse

theorem stewart_farm_horse_food : total_daily_horse_food = 12880 := by
  have num_horses : horses_needed 24 = 56 := by
    unfold horses_needed
    sorry -- Omitted for brevity, this would be solved

  have food_needed : 56 * 230 = 12880 := by
    sorry -- Omitted for brevity, this would be solved

  exact food_needed

end NUMINAMATH_GPT_stewart_farm_horse_food_l1257_125784


namespace NUMINAMATH_GPT_original_time_to_cover_distance_l1257_125707

theorem original_time_to_cover_distance (S : ℝ) (T : ℝ) (D : ℝ) :
  (0.8 * S) * (T + 10 / 60) = S * T → T = 2 / 3 :=
  by sorry

end NUMINAMATH_GPT_original_time_to_cover_distance_l1257_125707


namespace NUMINAMATH_GPT_unique_solution_of_system_l1257_125761

theorem unique_solution_of_system (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : x * (x + y + z) = 26) (h2 : y * (x + y + z) = 27) (h3 : z * (x + y + z) = 28) :
  x = 26 / 9 ∧ y = 3 ∧ z = 28 / 9 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_of_system_l1257_125761


namespace NUMINAMATH_GPT_each_child_play_time_l1257_125722

-- Define the conditions
def number_of_children : ℕ := 6
def pair_play_time : ℕ := 120
def pairs_playing_at_a_time : ℕ := 2

-- Define main theorem
theorem each_child_play_time : 
  (pairs_playing_at_a_time * pair_play_time) / number_of_children = 40 :=
sorry

end NUMINAMATH_GPT_each_child_play_time_l1257_125722


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1257_125720

/--
  Given a geometric sequence with the first three terms:
  a₁ = 27,
  a₂ = 54,
  a₃ = 108,
  prove that the common ratio is r = 2.
-/
theorem geometric_sequence_common_ratio :
  let a₁ := 27
  let a₂ := 54
  let a₃ := 108
  ∃ r : ℕ, (a₂ = r * a₁) ∧ (a₃ = r * a₂) ∧ r = 2 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1257_125720


namespace NUMINAMATH_GPT_total_length_segments_in_figure2_l1257_125719

-- Define the original dimensions of the figure
def vertical_side : ℕ := 10
def bottom_horizontal_side : ℕ := 3
def middle_horizontal_side : ℕ := 4
def topmost_horizontal_side : ℕ := 2

-- Define the lengths that are removed to form Figure 2
def removed_sides_length : ℕ :=
  bottom_horizontal_side + topmost_horizontal_side + vertical_side

-- Define the remaining lengths in Figure 2
def remaining_vertical_side : ℕ := vertical_side
def remaining_horizontal_side : ℕ := middle_horizontal_side

-- Total length of segments in Figure 2
def total_length_figure2 : ℕ :=
  remaining_vertical_side + remaining_horizontal_side

-- Conjecture that this total length is 14 units
theorem total_length_segments_in_figure2 : total_length_figure2 = 14 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_length_segments_in_figure2_l1257_125719


namespace NUMINAMATH_GPT_cube_surface_area_l1257_125768

-- Define the volume condition
def volume (s : ℕ) : ℕ := s^3

-- Define the surface area function
def surface_area (s : ℕ) : ℕ := 6 * s^2

-- State the theorem to be proven
theorem cube_surface_area (s : ℕ) (h : volume s = 729) : surface_area s = 486 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l1257_125768


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1257_125782

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, |2*x - 1| ≤ x → x^2 + x - 2 ≤ 0) ∧ 
  ¬(∀ x : ℝ, x^2 + x - 2 ≤ 0 → |2 * x - 1| ≤ x) := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1257_125782


namespace NUMINAMATH_GPT_fractional_eq_has_root_l1257_125798

theorem fractional_eq_has_root (x : ℝ) (m : ℝ) (h : x ≠ 4) :
    (3 / (x - 4) + (x + m) / (4 - x) = 1) → m = -1 :=
by
    intros h_eq
    sorry

end NUMINAMATH_GPT_fractional_eq_has_root_l1257_125798


namespace NUMINAMATH_GPT_opposite_of_negative_five_l1257_125794

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_negative_five_l1257_125794


namespace NUMINAMATH_GPT_point_N_in_second_quadrant_l1257_125743

theorem point_N_in_second_quadrant (a b : ℝ) (h1 : 1 + a < 0) (h2 : 2 * b - 1 < 0) :
    (a - 1 < 0) ∧ (1 - 2 * b > 0) :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_point_N_in_second_quadrant_l1257_125743


namespace NUMINAMATH_GPT_addition_subtraction_result_l1257_125721

theorem addition_subtraction_result :
  27474 + 3699 + 1985 - 2047 = 31111 :=
by {
  sorry
}

end NUMINAMATH_GPT_addition_subtraction_result_l1257_125721


namespace NUMINAMATH_GPT_distance_focus_directrix_parabola_l1257_125797

theorem distance_focus_directrix_parabola (p : ℝ) (h : y^2 = 20 * x) : 
  2 * p = 10 :=
by
  -- h represents the given condition y^2 = 20x.
  sorry

end NUMINAMATH_GPT_distance_focus_directrix_parabola_l1257_125797


namespace NUMINAMATH_GPT_pure_water_to_achieve_desired_concentration_l1257_125773

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end NUMINAMATH_GPT_pure_water_to_achieve_desired_concentration_l1257_125773


namespace NUMINAMATH_GPT_f_minimum_at_l1257_125763

noncomputable def f (x : ℝ) : ℝ := x * 2^x

theorem f_minimum_at : ∀ x : ℝ, x = -Real.log 2 → (∀ y : ℝ, f y ≥ f x) :=
by
  sorry

end NUMINAMATH_GPT_f_minimum_at_l1257_125763


namespace NUMINAMATH_GPT_distance_from_center_to_line_l1257_125765

-- Define the circle and its center
def is_circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def center : (ℝ × ℝ) := (1, 0)

-- Define the line equation y = tan(30°) * x
def is_line (x y : ℝ) : Prop := y = (1 / Real.sqrt 3) * x

-- Function to compute the distance from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  (abs (A * p.1 + B * p.2 + C)) / Real.sqrt (A^2 + B^2)

-- The main theorem to be proven:
theorem distance_from_center_to_line : 
  distance_point_to_line center (1 / Real.sqrt 3) (-1) 0 = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_distance_from_center_to_line_l1257_125765


namespace NUMINAMATH_GPT_skillful_hands_award_prob_cannot_enter_finals_after_training_l1257_125771

noncomputable def combinatorial_probability : ℚ :=
  let P1 := (4 * 3) / (10 * 10)    -- P1: 1 specified, 2 creative
  let P2 := (6 * 3) / (10 * 10)    -- P2: 2 specified, 1 creative
  let P3 := (6 * 3) / (10 * 10)    -- P3: 2 specified, 2 creative
  P1 + P2 + P3

theorem skillful_hands_award_prob : combinatorial_probability = 33 / 50 := 
  sorry

def after_training_probability := 3 / 4
theorem cannot_enter_finals_after_training : after_training_probability * 5 < 4 := 
  sorry

end NUMINAMATH_GPT_skillful_hands_award_prob_cannot_enter_finals_after_training_l1257_125771


namespace NUMINAMATH_GPT_cannot_inscribe_good_tetrahedron_in_good_parallelepiped_l1257_125775

-- Definitions related to the problem statements
def good_tetrahedron (V S : ℝ) := V = S

def good_parallelepiped (V' S1 S2 S3 : ℝ) := V' = 2 * (S1 + S2 + S3)

-- Theorem statement
theorem cannot_inscribe_good_tetrahedron_in_good_parallelepiped
  (V V' S : ℝ) (S1 S2 S3 : ℝ) (h1 h2 h3 : ℝ)
  (HT : good_tetrahedron V S)
  (HP : good_parallelepiped V' S1 S2 S3)
  (Hheights : S1 ≥ S2 ∧ S2 ≥ S3) :
  ¬ (V = S ∧ V' = 2 * (S1 + S2 + S3) ∧ h1 > 6 * S1 ∧ h2 > 6 * S2 ∧ h3 > 6 * S3) := 
sorry

end NUMINAMATH_GPT_cannot_inscribe_good_tetrahedron_in_good_parallelepiped_l1257_125775


namespace NUMINAMATH_GPT_find_g_one_l1257_125724

variable {α : Type} [AddGroup α]

def is_odd (f : α → α) : Prop :=
∀ x, f (-x) = - f x

def is_even (g : α → α) : Prop :=
∀ x, g (-x) = g x

theorem find_g_one
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 := by
  sorry

end NUMINAMATH_GPT_find_g_one_l1257_125724


namespace NUMINAMATH_GPT_linear_inequality_m_eq_one_l1257_125729

theorem linear_inequality_m_eq_one
  (m : ℤ)
  (h1 : |m| = 1)
  (h2 : m + 1 ≠ 0) :
  m = 1 :=
sorry

end NUMINAMATH_GPT_linear_inequality_m_eq_one_l1257_125729


namespace NUMINAMATH_GPT_percentage_difference_l1257_125733

theorem percentage_difference :
  let a := 0.80 * 40
  let b := (4 / 5) * 15
  a - b = 20 := by
sorry

end NUMINAMATH_GPT_percentage_difference_l1257_125733


namespace NUMINAMATH_GPT_coffee_per_cup_for_weak_l1257_125728

-- Defining the conditions
def weak_coffee_cups : ℕ := 12
def strong_coffee_cups : ℕ := 12
def total_coffee_tbsp : ℕ := 36
def weak_increase_factor : ℕ := 1
def strong_increase_factor : ℕ := 2

-- The theorem stating the problem
theorem coffee_per_cup_for_weak :
  ∃ W : ℝ, (weak_coffee_cups * W + strong_coffee_cups * (strong_increase_factor * W) = total_coffee_tbsp) ∧ (W = 1) :=
  sorry

end NUMINAMATH_GPT_coffee_per_cup_for_weak_l1257_125728


namespace NUMINAMATH_GPT_find_p_q_coprime_sum_l1257_125774

theorem find_p_q_coprime_sum (x y n m: ℕ) (h_sum: x + y = 30)
  (h_prob: ((n/x) * (n-1)/(x-1) * (n-2)/(x-2)) * ((m/y) * (m-1)/(y-1) * (m-2)/(y-2)) = 18/25)
  : ∃ p q : ℕ, p.gcd q = 1 ∧ p + q = 1006 :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_coprime_sum_l1257_125774


namespace NUMINAMATH_GPT_determine_F_value_l1257_125736

theorem determine_F_value (D E F : ℕ) (h1 : (9 + 6 + D + 1 + E + 8 + 2) % 3 = 0) (h2 : (5 + 4 + E + D + 2 + 1 + F) % 3 = 0) : 
  F = 2 := 
by
  sorry

end NUMINAMATH_GPT_determine_F_value_l1257_125736


namespace NUMINAMATH_GPT_p_sq_plus_q_sq_l1257_125708

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_p_sq_plus_q_sq_l1257_125708


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1257_125787

variable (x y : ℝ)
variable (hx_pos : x > 0) (hy_pos : y > 0) (hxy_ne : x ≠ y)

noncomputable def a := (x + y) / 2
noncomputable def b := Real.sqrt (x * y)
noncomputable def c := 2 / ((1 / x) + (1 / y))

theorem relationship_among_a_b_c :
    a > b ∧ b > c := by
    sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1257_125787


namespace NUMINAMATH_GPT_nonagon_side_length_l1257_125786

theorem nonagon_side_length (perimeter : ℝ) (n : ℕ) (h_reg_nonagon : n = 9) (h_perimeter : perimeter = 171) :
  perimeter / n = 19 := by
  sorry

end NUMINAMATH_GPT_nonagon_side_length_l1257_125786


namespace NUMINAMATH_GPT_range_of_a_l1257_125723

variable (a : ℝ)

def proposition_p : Prop :=
  ∃ x₀ : ℝ, x₀^2 - a * x₀ + a = 0

def proposition_q : Prop :=
  ∀ x : ℝ, 1 < x → x + 1 / (x - 1) ≥ a

theorem range_of_a (h : ¬proposition_p a ∧ proposition_q a) : 0 < a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1257_125723


namespace NUMINAMATH_GPT_solve_congruence_l1257_125711

theorem solve_congruence : ∃ n : ℕ, 0 ≤ n ∧ n < 43 ∧ 11 * n % 43 = 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_congruence_l1257_125711


namespace NUMINAMATH_GPT_total_cost_eq_1400_l1257_125734

theorem total_cost_eq_1400 (stove_cost : ℝ) (wall_repair_fraction : ℝ) (wall_repair_cost : ℝ) (total_cost : ℝ)
  (h₁ : stove_cost = 1200)
  (h₂ : wall_repair_fraction = 1/6)
  (h₃ : wall_repair_cost = stove_cost * wall_repair_fraction)
  (h₄ : total_cost = stove_cost + wall_repair_cost) :
  total_cost = 1400 :=
sorry

end NUMINAMATH_GPT_total_cost_eq_1400_l1257_125734


namespace NUMINAMATH_GPT_average_headcount_11600_l1257_125770

theorem average_headcount_11600 : 
  let h02_03 := 11700
  let h03_04 := 11500
  let h04_05 := 11600
  (h02_03 + h03_04 + h04_05) / 3 = 11600 := 
by
  sorry

end NUMINAMATH_GPT_average_headcount_11600_l1257_125770


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1257_125777

/-- 
Given the expression (1 + 1 / (x - 2)) ÷ ((x ^ 2 - 2 * x + 1) / (x - 2)), 
prove that it evaluates to -1 when x = 0.
-/
theorem simplify_and_evaluate (x : ℝ) (h : x = 0) :
  (1 + 1 / (x - 2)) / ((x^2 - 2 * x + 1) / (x - 2)) = -1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1257_125777


namespace NUMINAMATH_GPT_walking_representation_l1257_125756

-- Definitions based on conditions
def represents_walking_eastward (m : ℤ) : Prop := m > 0

-- The theorem to prove based on the problem statement
theorem walking_representation :
  represents_walking_eastward 5 →
  ¬ represents_walking_eastward (-10) ∧ abs (-10) = 10 :=
by
  sorry

end NUMINAMATH_GPT_walking_representation_l1257_125756


namespace NUMINAMATH_GPT_factor_expression_l1257_125778

theorem factor_expression (z : ℂ) : 
  75 * z^12 + 162 * z^24 + 27 = 3 * (9 + z^12 * (25 + 54 * z^12)) :=
sorry

end NUMINAMATH_GPT_factor_expression_l1257_125778


namespace NUMINAMATH_GPT_xiao_ming_returns_and_distance_is_correct_l1257_125725

theorem xiao_ming_returns_and_distance_is_correct :
  ∀ (walk_distance : ℝ) (turn_angle : ℝ), 
  walk_distance = 5 ∧ turn_angle = 20 → 
  (∃ n : ℕ, (360 % turn_angle = 0) ∧ n = 360 / turn_angle ∧ walk_distance * n = 90) :=
by
  sorry

end NUMINAMATH_GPT_xiao_ming_returns_and_distance_is_correct_l1257_125725


namespace NUMINAMATH_GPT_triangle_equilateral_l1257_125779

variables {A B C : ℝ} -- angles of the triangle
variables {a b c : ℝ} -- sides opposite to the angles

-- Given conditions
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C = c * Real.cos A ∧ (b * b = a * c)

-- The proof goal
theorem triangle_equilateral (A B C : ℝ) (a b c : ℝ) :
  triangle A B C a b c → a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_triangle_equilateral_l1257_125779


namespace NUMINAMATH_GPT_basic_computer_price_l1257_125785

variables (C P : ℕ)

theorem basic_computer_price (h1 : C + P = 2500)
                            (h2 : C + 500 + P = 6 * P) : C = 2000 :=
by
  sorry

end NUMINAMATH_GPT_basic_computer_price_l1257_125785


namespace NUMINAMATH_GPT_min_value_of_expr_l1257_125766

noncomputable def min_expr (a b c : ℝ) := (2 * a / b) + (3 * b / c) + (4 * c / a)

theorem min_value_of_expr (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) 
    (habc : a * b * c = 1) : 
  min_expr a b c ≥ 9 := 
sorry

end NUMINAMATH_GPT_min_value_of_expr_l1257_125766


namespace NUMINAMATH_GPT_prob_odd_sum_l1257_125713

-- Given conditions on the spinners
def spinner_P := [1, 2, 3]
def spinner_Q := [2, 4, 6]
def spinner_R := [1, 3, 5]

-- Probability of spinner P landing on an even number is 1/3
def prob_even_P : ℚ := 1 / 3

-- Probability of odd sum from spinners P, Q, and R
theorem prob_odd_sum : 
  (prob_even_P = 1 / 3) → 
  ∃ p : ℚ, p = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_prob_odd_sum_l1257_125713


namespace NUMINAMATH_GPT_average_minutes_proof_l1257_125715

noncomputable def average_minutes_heard (total_minutes : ℕ) (total_attendees : ℕ) (full_listened_fraction : ℚ) (none_listened_fraction : ℚ) (half_remainder_fraction : ℚ) : ℚ := 
  let full_listeners := full_listened_fraction * total_attendees
  let none_listeners := none_listened_fraction * total_attendees
  let remaining_listeners := total_attendees - full_listeners - none_listeners
  let half_listeners := half_remainder_fraction * remaining_listeners
  let quarter_listeners := remaining_listeners - half_listeners
  let total_heard := (full_listeners * total_minutes) + (none_listeners * 0) + (half_listeners * (total_minutes / 2)) + (quarter_listeners * (total_minutes / 4))
  total_heard / total_attendees

theorem average_minutes_proof : 
  average_minutes_heard 120 100 (30/100) (15/100) (40/100) = 59.1 := 
by
  sorry

end NUMINAMATH_GPT_average_minutes_proof_l1257_125715


namespace NUMINAMATH_GPT_parking_lot_motorcycles_l1257_125703

theorem parking_lot_motorcycles
  (x y : ℕ)
  (h1 : x + y = 24)
  (h2 : 3 * x + 4 * y = 86) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_parking_lot_motorcycles_l1257_125703


namespace NUMINAMATH_GPT_mean_equality_l1257_125748

theorem mean_equality (y : ℝ) (h : (6 + 9 + 18) / 3 = (12 + y) / 2) : y = 10 :=
by sorry

end NUMINAMATH_GPT_mean_equality_l1257_125748


namespace NUMINAMATH_GPT_fold_minus2_2_3_coincides_neg3_fold_minus1_3_7_coincides_neg5_fold_distanceA_to_B_coincide_l1257_125731

section FoldingNumberLine

-- Part (1)
def coincides_point_3_if_minus2_2_fold (x : ℝ) : Prop :=
  x = -3

theorem fold_minus2_2_3_coincides_neg3 :
  coincides_point_3_if_minus2_2_fold 3 :=
by
  sorry

-- Part (2) ①
def coincides_point_7_if_minus1_3_fold (x : ℝ) : Prop :=
  x = -5

theorem fold_minus1_3_7_coincides_neg5 :
  coincides_point_7_if_minus1_3_fold 7 :=
by
  sorry

-- Part (2) ②
def B_position_after_folding (m : ℝ) (h : m > 0) (A B : ℝ) : Prop :=
  B = 1 + m / 2

theorem fold_distanceA_to_B_coincide (m : ℝ) (h : m > 0) (A B : ℝ) :
  B_position_after_folding m h A B :=
by
  sorry

end FoldingNumberLine

end NUMINAMATH_GPT_fold_minus2_2_3_coincides_neg3_fold_minus1_3_7_coincides_neg5_fold_distanceA_to_B_coincide_l1257_125731


namespace NUMINAMATH_GPT_probability_of_opposite_middle_vertex_l1257_125752

noncomputable def ant_moves_to_opposite_middle_vertex_prob : ℚ := 1 / 2

-- Specification of the problem conditions
structure Octahedron :=
  (middle_vertices : Finset ℕ) -- Assume some identification of middle vertices
  (adjacent_vertices : ℕ → Finset ℕ) -- Function mapping a vertex to its adjacent vertices
  (is_middle_vertex : ℕ → Prop) -- Predicate to check if a vertex is a middle vertex
  (is_top_or_bottom_vertex : ℕ → Prop) -- Predicate to check if a vertex is a top or bottom vertex
  (start_vertex : ℕ)

variables (O : Octahedron)

-- Main theorem statement
theorem probability_of_opposite_middle_vertex :
  ∃ A B : ℕ, A ∈ O.adjacent_vertices O.start_vertex ∧ B ∈ O.adjacent_vertices A ∧ B ≠ O.start_vertex ∧ (∃ x ∈ O.middle_vertices, x = B) →
  (∀ (A B : ℕ), (A ∈ O.adjacent_vertices O.start_vertex ∧ B ∈ O.adjacent_vertices A ∧ B ≠ O.start_vertex ∧ (∃ x ∈ O.middle_vertices, x = B)) →
    ant_moves_to_opposite_middle_vertex_prob = 1 / 2) := sorry

end NUMINAMATH_GPT_probability_of_opposite_middle_vertex_l1257_125752


namespace NUMINAMATH_GPT_min_value_4x_3y_l1257_125790

theorem min_value_4x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y = 5 * x * y) : 
  4 * x + 3 * y ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_4x_3y_l1257_125790


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1257_125717

-- We state the main problem in Lean as a theorem.
theorem geometric_sequence_sum (S : ℕ → ℕ) (S_4_eq : S 4 = 8) (S_8_eq : S 8 = 24) : S 12 = 88 :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1257_125717


namespace NUMINAMATH_GPT_park_area_l1257_125704

theorem park_area (w : ℝ) (h1 : 2 * (w + 3 * w) = 72) : w * (3 * w) = 243 :=
by
  sorry

end NUMINAMATH_GPT_park_area_l1257_125704


namespace NUMINAMATH_GPT_unique_10_digit_number_property_l1257_125783

def ten_digit_number (N : ℕ) : Prop :=
  10^9 ≤ N ∧ N < 10^10

def first_digits_coincide (N : ℕ) : Prop :=
  ∀ M : ℕ, N^2 < 10^M → N^2 / 10^(M - 10) = N

theorem unique_10_digit_number_property :
  ∀ (N : ℕ), ten_digit_number N ∧ first_digits_coincide N → N = 1000000000 := 
by
  intros N hN
  sorry

end NUMINAMATH_GPT_unique_10_digit_number_property_l1257_125783


namespace NUMINAMATH_GPT_work_completion_days_l1257_125751

theorem work_completion_days (D : ℕ) (W : ℕ) :
  (D : ℕ) = 6 :=
by 
  -- define constants and given conditions
  let original_men := 10
  let additional_men := 10
  let early_days := 3

  -- define the premise
  -- work done with original men in original days
  have work_done_original : W = (original_men * D) := sorry
  -- work done with additional men in reduced days
  have work_done_with_additional : W = ((original_men + additional_men) * (D - early_days)) := sorry

  -- prove the equality from the condition
  have eq : original_men * D = (original_men + additional_men) * (D - early_days) := sorry

  -- simplify to solve for D
  have solution : D = 6 := sorry

  exact solution

end NUMINAMATH_GPT_work_completion_days_l1257_125751


namespace NUMINAMATH_GPT_find_integer_l1257_125741

theorem find_integer (n : ℤ) 
  (h1 : 50 ≤ n ∧ n ≤ 150)
  (h2 : n % 7 = 0)
  (h3 : n % 9 = 3)
  (h4 : n % 6 = 3) : 
  n = 63 := by 
  sorry

end NUMINAMATH_GPT_find_integer_l1257_125741


namespace NUMINAMATH_GPT_sequence_is_arithmetic_max_value_a_n_b_n_l1257_125769

open Real

theorem sequence_is_arithmetic (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (Sn : ℕ → ℝ) 
  (h_Sn : ∀ n, Sn n = (a n ^ 2 + a n) / 2) :
    ∀ n, a n = n := sorry 

theorem max_value_a_n_b_n (a b : ℕ → ℝ)
  (h_b : ∀ n, b n = - n + 5)
  (h_a : ∀ n, a n = n) :
    ∀ n, n ≥ 2 → n ≤ 3 → 
    ∃ k, a k * b k = 25 / 4 := by 
      sorry

end NUMINAMATH_GPT_sequence_is_arithmetic_max_value_a_n_b_n_l1257_125769


namespace NUMINAMATH_GPT_six_digit_numbers_with_zero_count_l1257_125746

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end NUMINAMATH_GPT_six_digit_numbers_with_zero_count_l1257_125746


namespace NUMINAMATH_GPT_seeds_per_can_l1257_125799

theorem seeds_per_can (total_seeds : ℝ) (number_of_cans : ℝ) (h1 : total_seeds = 54.0) (h2 : number_of_cans = 9.0) : (total_seeds / number_of_cans = 6.0) :=
by
  rw [h1, h2]
  norm_num
  -- sorry

end NUMINAMATH_GPT_seeds_per_can_l1257_125799


namespace NUMINAMATH_GPT_intersection_point_a_l1257_125701

theorem intersection_point_a : ∃ (x y : ℝ), y = 4 * x - 32 ∧ y = -6 * x + 8 ∧ x = 4 ∧ y = -16 :=
sorry

end NUMINAMATH_GPT_intersection_point_a_l1257_125701


namespace NUMINAMATH_GPT_max_distance_unit_circle_l1257_125795

open Complex

theorem max_distance_unit_circle : 
  ∀ (z : ℂ), abs z = 1 → ∃ M : ℝ, M = abs (z - (1 : ℂ) - I) ∧ ∀ w : ℂ, abs w = 1 → abs (w - 1 - I) ≤ M :=
by
  sorry

end NUMINAMATH_GPT_max_distance_unit_circle_l1257_125795


namespace NUMINAMATH_GPT_binary_representation_253_l1257_125755

-- Define the decimal number
def decimal := 253

-- Define the number of zeros (x) and ones (y) in the binary representation of 253
def num_zeros := 1
def num_ones := 7

-- Prove that 2y - x = 13 given these conditions
theorem binary_representation_253 : (2 * num_ones - num_zeros) = 13 :=
by
  sorry

end NUMINAMATH_GPT_binary_representation_253_l1257_125755


namespace NUMINAMATH_GPT_leak_emptying_time_l1257_125744

theorem leak_emptying_time (A_rate L_rate : ℚ) 
  (hA : A_rate = 1 / 4)
  (hCombined : A_rate - L_rate = 1 / 8) :
  1 / L_rate = 8 := 
by
  sorry

end NUMINAMATH_GPT_leak_emptying_time_l1257_125744


namespace NUMINAMATH_GPT_range_of_m_l1257_125726

def M := {y : ℝ | ∃ (x : ℝ), y = (1/2)^x}
def N (m : ℝ) := {y : ℝ | ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ y = ((1/(m-1) + 1) * (x - 1) + (|m| - 1) * (x - 2))}

theorem range_of_m (m : ℝ) : (∀ y ∈ N m, y ∈ M) ↔ -1 < m ∧ m < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1257_125726


namespace NUMINAMATH_GPT_tape_length_division_l1257_125760

theorem tape_length_division (n_pieces : ℕ) (length_piece overlap : ℝ) (n_parts : ℕ) 
  (h_pieces : n_pieces = 5) (h_length : length_piece = 2.7) (h_overlap : overlap = 0.3) 
  (h_parts : n_parts = 6) : 
  ((n_pieces * length_piece) - ((n_pieces - 1) * overlap)) / n_parts = 2.05 :=
  by
    sorry

end NUMINAMATH_GPT_tape_length_division_l1257_125760


namespace NUMINAMATH_GPT_P_roots_implies_Q_square_roots_l1257_125789

noncomputable def P (x : ℝ) : ℝ := x^3 - 2 * x + 1

noncomputable def Q (x : ℝ) : ℝ := x^3 - 4 * x^2 + 4 * x - 1

theorem P_roots_implies_Q_square_roots (r : ℝ) (h : P r = 0) : Q (r^2) = 0 := sorry

end NUMINAMATH_GPT_P_roots_implies_Q_square_roots_l1257_125789


namespace NUMINAMATH_GPT_product_of_radii_l1257_125709

theorem product_of_radii (x y r₁ r₂ : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hr₁ : (x - r₁)^2 + (y - r₁)^2 = r₁^2)
  (hr₂ : (x - r₂)^2 + (y - r₂)^2 = r₂^2)
  (hroots : r₁ + r₂ = 2 * (x + y)) : r₁ * r₂ = x^2 + y^2 := by
  sorry

end NUMINAMATH_GPT_product_of_radii_l1257_125709


namespace NUMINAMATH_GPT_real_solutions_count_l1257_125727

theorem real_solutions_count :
  (∃ x : ℝ, |x - 2| - 4 = 1 / |x - 3|) ∧
  (∃ y : ℝ, |y - 2| - 4 = 1 / |y - 3| ∧ x ≠ y) :=
sorry

end NUMINAMATH_GPT_real_solutions_count_l1257_125727


namespace NUMINAMATH_GPT_express_in_scientific_notation_l1257_125780

-- Definition for expressing number in scientific notation
def scientific_notation (n : ℝ) (a : ℝ) (b : ℕ) : Prop :=
  n = a * 10 ^ b

-- Condition of the problem
def condition : ℝ := 1300000

-- Stating the theorem to be proved
theorem express_in_scientific_notation : scientific_notation condition 1.3 6 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l1257_125780


namespace NUMINAMATH_GPT_abs_val_of_minus_two_and_half_l1257_125754

-- Definition of the absolute value function for real numbers
def abs_val (x : ℚ) : ℚ := if x < 0 then -x else x

-- Prove that the absolute value of -2.5 (which is -5/2) is equal to 2.5 (which is 5/2)
theorem abs_val_of_minus_two_and_half : abs_val (-5/2) = 5/2 := by
  sorry

end NUMINAMATH_GPT_abs_val_of_minus_two_and_half_l1257_125754


namespace NUMINAMATH_GPT_cards_drawn_to_product_even_l1257_125710

theorem cards_drawn_to_product_even :
  ∃ n, (∀ (cards_drawn : Finset ℕ), 
    (cards_drawn ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}) ∧
    (cards_drawn.card = n) → 
    ¬ (∀ c ∈ cards_drawn, c % 2 = 1)
  ) ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_cards_drawn_to_product_even_l1257_125710


namespace NUMINAMATH_GPT_tamia_total_slices_and_pieces_l1257_125737

-- Define the conditions
def num_bell_peppers : ℕ := 5
def slices_per_pepper : ℕ := 20
def num_large_slices : ℕ := num_bell_peppers * slices_per_pepper
def num_half_slices : ℕ := num_large_slices / 2
def small_pieces_per_slice : ℕ := 3
def num_small_pieces : ℕ := num_half_slices * small_pieces_per_slice
def num_uncut_slices : ℕ := num_half_slices

-- Define the total number of pieces and slices
def total_pieces_and_slices : ℕ := num_uncut_slices + num_small_pieces

-- State the theorem and provide a placeholder for the proof
theorem tamia_total_slices_and_pieces : total_pieces_and_slices = 200 :=
by
  sorry

end NUMINAMATH_GPT_tamia_total_slices_and_pieces_l1257_125737


namespace NUMINAMATH_GPT_my_inequality_l1257_125714

open Real

variable {a b c : ℝ}

theorem my_inequality 
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a * b + b * c + c * a = 1) :
  sqrt (a ^ 3 + a) + sqrt (b ^ 3 + b) + sqrt (c ^ 3 + c) ≥ 2 * sqrt (a + b + c) := 
  sorry

end NUMINAMATH_GPT_my_inequality_l1257_125714


namespace NUMINAMATH_GPT_expression_equivalence_l1257_125740

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 :=
by
  sorry

end NUMINAMATH_GPT_expression_equivalence_l1257_125740


namespace NUMINAMATH_GPT_simplify_expression_eq_sqrt3_l1257_125757

theorem simplify_expression_eq_sqrt3
  (a : ℝ)
  (h : a = Real.sqrt 3 + 1) :
  ( (a + 1) / a / (a - (1 + 2 * a^2) / (3 * a)) ) = Real.sqrt 3 := sorry

end NUMINAMATH_GPT_simplify_expression_eq_sqrt3_l1257_125757


namespace NUMINAMATH_GPT_minimum_purchase_price_mod6_l1257_125793

theorem minimum_purchase_price_mod6 
  (coin_values : List ℕ)
  (h1 : (1 : ℕ) ∈ coin_values)
  (h15 : (15 : ℕ) ∈ coin_values)
  (h50 : (50 : ℕ) ∈ coin_values)
  (A C : ℕ)
  (k : ℕ)
  (hA : A ≡ k [MOD 7])
  (hC : C ≡ k + 1 [MOD 7])
  (hP : ∃ P, P = A - C) : 
  ∃ P, P ≡ 6 [MOD 7] ∧ P > 0 :=
by
  sorry

end NUMINAMATH_GPT_minimum_purchase_price_mod6_l1257_125793


namespace NUMINAMATH_GPT_find_y_given_conditions_l1257_125762

theorem find_y_given_conditions (k : ℝ) (h1 : ∀ (x y : ℝ), xy = k) (h2 : ∀ (x y : ℝ), x + y = 30) (h3 : ∀ (x y : ℝ), x - y = 10) :
    ∀ x y, x = 8 → y = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_y_given_conditions_l1257_125762


namespace NUMINAMATH_GPT_new_ratio_alcohol_water_l1257_125796

theorem new_ratio_alcohol_water (alcohol water: ℕ) (initial_ratio: alcohol * 3 = water * 4) 
  (extra_water: ℕ) (extra_water_added: extra_water = 4) (alcohol_given: alcohol = 20):
  20 * 19 = alcohol * (water + extra_water) :=
by
  sorry

end NUMINAMATH_GPT_new_ratio_alcohol_water_l1257_125796


namespace NUMINAMATH_GPT_age_ratio_l1257_125747

theorem age_ratio (R D : ℕ) (h1 : R + 2 = 26) (h2 : D = 18) : R / D = 4 / 3 :=
sorry

end NUMINAMATH_GPT_age_ratio_l1257_125747


namespace NUMINAMATH_GPT_factor_expression_l1257_125745

theorem factor_expression (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2) ^ 2 :=
by sorry

end NUMINAMATH_GPT_factor_expression_l1257_125745


namespace NUMINAMATH_GPT_find_h_l1257_125788

noncomputable def h (x : ℝ) : ℝ := -x^4 - 2 * x^3 + 4 * x^2 + 9 * x - 5

def f (x : ℝ) : ℝ := x^4 + 2 * x^3 - x^2 - 4 * x + 1

def p (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 4

theorem find_h (x : ℝ) : (f x) + (h x) = p x :=
by sorry

end NUMINAMATH_GPT_find_h_l1257_125788


namespace NUMINAMATH_GPT_sqrt_of_sum_of_powers_l1257_125764

theorem sqrt_of_sum_of_powers : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end NUMINAMATH_GPT_sqrt_of_sum_of_powers_l1257_125764


namespace NUMINAMATH_GPT_inequality_solution_l1257_125705

theorem inequality_solution (x : ℝ) :
  ( (x^2 + 3*x + 3) > 0 ) → ( ((x^2 + 3*x + 3)^(5*x^3 - 3*x^2)) ≤ ((x^2 + 3*x + 3)^(3*x^3 + 5*x)) )
  ↔ ( x ∈ (Set.Iic (-2) ∪ ({-1} : Set ℝ) ∪ Set.Icc 0 (5/2)) ) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1257_125705


namespace NUMINAMATH_GPT_tennis_player_games_l1257_125749

theorem tennis_player_games (b : ℕ → ℕ) (h1 : ∀ k, b k ≥ k) (h2 : ∀ k, b k ≤ 12 * (k / 7)) :
  ∃ i j : ℕ, i < j ∧ b j - b i = 20 :=
by
  sorry

end NUMINAMATH_GPT_tennis_player_games_l1257_125749


namespace NUMINAMATH_GPT_largest_number_eq_l1257_125758

theorem largest_number_eq (x y z : ℚ) (h1 : x + y + z = 82) (h2 : z - y = 10) (h3 : y - x = 4) :
  z = 106 / 3 :=
sorry

end NUMINAMATH_GPT_largest_number_eq_l1257_125758


namespace NUMINAMATH_GPT_total_buttons_needed_l1257_125712

def shirts_sewn_on_monday := 4
def shirts_sewn_on_tuesday := 3
def shirts_sewn_on_wednesday := 2
def buttons_per_shirt := 5

theorem total_buttons_needed : 
  (shirts_sewn_on_monday + shirts_sewn_on_tuesday + shirts_sewn_on_wednesday) * buttons_per_shirt = 45 :=
by 
  sorry

end NUMINAMATH_GPT_total_buttons_needed_l1257_125712


namespace NUMINAMATH_GPT_smallest_w_l1257_125738

theorem smallest_w (w : ℕ) (h1 : 1916 = 2^2 * 479) (h2 : w > 0) : w = 74145392000 ↔ 
  (∀ p e, (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11) → (∃ k, (1916 * w = p^e * k ∧ e ≥ if p = 2 then 6 else 3))) :=
sorry

end NUMINAMATH_GPT_smallest_w_l1257_125738


namespace NUMINAMATH_GPT_Q_equals_10_04_l1257_125750
-- Import Mathlib for mathematical operations and equivalence checking

-- Define the given conditions
def a := 6
def b := 3
def c := 2

-- Define the expression to be evaluated
def Q : ℚ := (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2)

-- Prove that the expression equals 10.04
theorem Q_equals_10_04 : Q = 10.04 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Q_equals_10_04_l1257_125750


namespace NUMINAMATH_GPT_product_is_in_A_l1257_125767

def is_sum_of_squares (z : Int) : Prop :=
  ∃ t s : Int, z = t^2 + s^2

variable {x y : Int}

theorem product_is_in_A (hx : is_sum_of_squares x) (hy : is_sum_of_squares y) :
  is_sum_of_squares (x * y) :=
sorry

end NUMINAMATH_GPT_product_is_in_A_l1257_125767


namespace NUMINAMATH_GPT_inequality_geq_27_l1257_125702

theorem inequality_geq_27 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_eq : a + b + c + 2 = a * b * c) : (a + 1) * (b + 1) * (c + 1) ≥ 27 := 
    sorry

end NUMINAMATH_GPT_inequality_geq_27_l1257_125702


namespace NUMINAMATH_GPT_initial_investment_l1257_125739

theorem initial_investment (P : ℝ) 
  (h1: ∀ (r : ℝ) (n : ℕ), r = 0.20 ∧ n = 3 → P * (1 + r)^n = P * 1.728)
  (h2: ∀ (A : ℝ), A = P * 1.728 → 3 * A = 5.184 * P)
  (h3: ∀ (P_new : ℝ) (r_new : ℝ), P_new = 5.184 * P ∧ r_new = 0.15 → P_new * (1 + r_new) = 5.9616 * P)
  (h4: 5.9616 * P = 59616)
  : P = 10000 :=
sorry

end NUMINAMATH_GPT_initial_investment_l1257_125739


namespace NUMINAMATH_GPT_prism_cubes_paint_condition_l1257_125759

theorem prism_cubes_paint_condition
  (m n r : ℕ)
  (h1 : m ≤ n)
  (h2 : n ≤ r)
  (h3 : (m - 2) * (n - 2) * (r - 2)
        - 2 * ((m - 2) * (n - 2) + (m - 2) * (r - 2) + (n - 2) * (r - 2)) 
        + 4 * (m - 2 + n - 2 + r - 2)
        = 1985) :
  (m = 5 ∧ n = 7 ∧ r = 663) ∨
  (m = 5 ∧ n = 5 ∧ r = 1981) ∨
  (m = 3 ∧ n = 3 ∧ r = 1981) ∨
  (m = 1 ∧ n = 7 ∧ r = 399) ∨
  (m = 1 ∧ n = 3 ∧ r = 1987) := 
sorry

end NUMINAMATH_GPT_prism_cubes_paint_condition_l1257_125759


namespace NUMINAMATH_GPT_max_value_char_l1257_125753

theorem max_value_char (m x a b : ℕ) (h_sum : 28 * m + x + a + 2 * b = 368)
  (h1 : x ≤ 23) (h2 : x > a) (h3 : a > b) (h4 : b ≥ 0) :
  m + x ≤ 35 := 
sorry

end NUMINAMATH_GPT_max_value_char_l1257_125753


namespace NUMINAMATH_GPT_home_run_difference_l1257_125772

def hank_aaron_home_runs : ℕ := 755
def dave_winfield_home_runs : ℕ := 465

theorem home_run_difference :
  2 * dave_winfield_home_runs - hank_aaron_home_runs = 175 := by
  sorry

end NUMINAMATH_GPT_home_run_difference_l1257_125772


namespace NUMINAMATH_GPT_sum_of_numbers_l1257_125700

-- Definitions for the numbers involved
def n1 : Nat := 1235
def n2 : Nat := 2351
def n3 : Nat := 3512
def n4 : Nat := 5123

-- Proof statement
theorem sum_of_numbers :
  n1 + n2 + n3 + n4 = 12221 := by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1257_125700


namespace NUMINAMATH_GPT_min_value_of_ratio_l1257_125716

theorem min_value_of_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  (4 / x + 1 / y) ≥ 6 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_ratio_l1257_125716


namespace NUMINAMATH_GPT_tom_seashells_l1257_125791

theorem tom_seashells (days : ℕ) (seashells_per_day : ℕ) (h1 : days = 5) (h2 : seashells_per_day = 7) : 
  seashells_per_day * days = 35 := 
by
  sorry

end NUMINAMATH_GPT_tom_seashells_l1257_125791


namespace NUMINAMATH_GPT_average_marks_l1257_125730

/-- Shekar scored 76, 65, 82, 67, and 85 marks in Mathematics, Science, Social Studies, English, and Biology respectively.
    We aim to prove that his average marks are 75. -/

def marks : List ℕ := [76, 65, 82, 67, 85]

theorem average_marks : (marks.sum / marks.length) = 75 := by
  sorry

end NUMINAMATH_GPT_average_marks_l1257_125730


namespace NUMINAMATH_GPT_value_of_b_minus_a_l1257_125742

open Real

def condition (a b : ℝ) : Prop := 
  abs a = 3 ∧ abs b = 2 ∧ a + b > 0

theorem value_of_b_minus_a (a b : ℝ) (h : condition a b) :
  b - a = -1 ∨ b - a = -5 :=
  sorry

end NUMINAMATH_GPT_value_of_b_minus_a_l1257_125742


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l1257_125735

noncomputable def radius_of_circumcircle (a b c : ℚ) (h_a : a = 15/2) (h_b : b = 10) (h_c : c = 25/2) : ℚ :=
if h_triangle : a^2 + b^2 = c^2 then (c / 2) else 0

theorem circumscribed_circle_radius :
  radius_of_circumcircle (15/2 : ℚ) 10 (25/2 : ℚ) (by norm_num) (by norm_num) (by norm_num) = 25 / 4 := 
by
  sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l1257_125735


namespace NUMINAMATH_GPT_y_intercept_of_line_l1257_125706

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  -- The proof steps will go here.
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1257_125706


namespace NUMINAMATH_GPT_problem_a_problem_b_l1257_125732

noncomputable def gini_coefficient_separate_operations : ℝ := 
  let population_north := 24
  let population_south := population_north / 4
  let income_per_north_inhabitant := (6000 * 18) / population_north
  let income_per_south_inhabitant := (6000 * 12) / population_south
  let total_population := population_north + population_south
  let total_income := 6000 * (18 + 12)
  let share_pop_north := population_north / total_population
  let share_income_north := (income_per_north_inhabitant * population_north) / total_income
  share_pop_north - share_income_north

theorem problem_a : gini_coefficient_separate_operations = 0.2 := 
  by sorry

noncomputable def change_in_gini_coefficient_after_collaboration : ℝ :=
  let previous_income_north := 6000 * 18
  let compensation := previous_income_north + 1983
  let total_combined_income := 6000 * 30.5
  let remaining_income_south := total_combined_income - compensation
  let population := 24 + 6
  let income_per_capita_north := compensation / 24
  let income_per_capita_south := remaining_income_south / 6
  let new_gini_coefficient := 
    let share_pop_north := 24 / population
    let share_income_north := compensation / total_combined_income
    share_pop_north - share_income_north
  (0.2 - new_gini_coefficient)

theorem problem_b : change_in_gini_coefficient_after_collaboration = 0.001 := 
  by sorry

end NUMINAMATH_GPT_problem_a_problem_b_l1257_125732
