import Mathlib

namespace NUMINAMATH_GPT_fourth_root_squared_cubed_l2221_222170

theorem fourth_root_squared_cubed (x : ℝ) (h : (x^(1/4))^2^3 = 1296) : x = 256 :=
sorry

end NUMINAMATH_GPT_fourth_root_squared_cubed_l2221_222170


namespace NUMINAMATH_GPT_melissa_remaining_bananas_l2221_222157

theorem melissa_remaining_bananas :
  let initial_bananas := 88
  let shared_bananas := 4
  initial_bananas - shared_bananas = 84 :=
by
  sorry

end NUMINAMATH_GPT_melissa_remaining_bananas_l2221_222157


namespace NUMINAMATH_GPT_fraction_comparison_and_differences_l2221_222150

theorem fraction_comparison_and_differences :
  (1/3 < 0.5) ∧ (0.5 < 3/5) ∧ 
  (0.5 - 1/3 = 1/6) ∧ 
  (3/5 - 0.5 = 1/10) :=
by
  sorry

end NUMINAMATH_GPT_fraction_comparison_and_differences_l2221_222150


namespace NUMINAMATH_GPT_time_to_traverse_nth_mile_l2221_222117

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 3) : ∃ t : ℕ, t = (n - 2)^2 :=
by
  -- Given:
  -- Speed varies inversely as the square of the number of miles already traveled.
  -- Speed is constant for each mile.
  -- The third mile is traversed in 4 hours.
  -- Show that:
  -- The time to traverse the nth mile is (n - 2)^2 hours.
  sorry

end NUMINAMATH_GPT_time_to_traverse_nth_mile_l2221_222117


namespace NUMINAMATH_GPT_find_speed_way_home_l2221_222160

theorem find_speed_way_home
  (speed_to_mother : ℝ)
  (average_speed : ℝ)
  (speed_to_mother_val : speed_to_mother = 130)
  (average_speed_val : average_speed = 109) :
  ∃ v : ℝ, v = 109 * 130 / 151 := by
  sorry

end NUMINAMATH_GPT_find_speed_way_home_l2221_222160


namespace NUMINAMATH_GPT_intersection_of_M_N_l2221_222107

-- Definitions of the sets M and N
def M : Set ℝ := { x | (x + 2) * (x - 1) < 0 }
def N : Set ℝ := { x | x + 1 < 0 }

-- Proposition stating that the intersection of M and N is { x | -2 < x < -1 }
theorem intersection_of_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < -1 } :=
  by
    sorry

end NUMINAMATH_GPT_intersection_of_M_N_l2221_222107


namespace NUMINAMATH_GPT_two_digit_numbers_condition_l2221_222192

theorem two_digit_numbers_condition : ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
    10 * a + b ≥ 10 ∧ 10 * a + b ≤ 99 ∧
    (10 * a + b) / (a + b) = (a + b) / 3 ∧ 
    (10 * a + b = 27 ∨ 10 * a + b = 48) := 
by
    sorry

end NUMINAMATH_GPT_two_digit_numbers_condition_l2221_222192


namespace NUMINAMATH_GPT_problem_ineq_l2221_222143

theorem problem_ineq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_neq : a ≠ b) :
  (a^2 * b + a + b^2) * (a * b^2 + a^2 + b) > 9 * a^2 * b^2 := 
by 
  sorry

end NUMINAMATH_GPT_problem_ineq_l2221_222143


namespace NUMINAMATH_GPT_parking_space_unpainted_side_l2221_222175

theorem parking_space_unpainted_side 
  (L W : ℝ) 
  (h1 : 2 * W + L = 37) 
  (h2 : L * W = 125) : 
  L = 8.90 := 
by 
  sorry

end NUMINAMATH_GPT_parking_space_unpainted_side_l2221_222175


namespace NUMINAMATH_GPT_solve_for_five_minus_a_l2221_222138

theorem solve_for_five_minus_a (a b : ℤ) 
  (h1 : 5 + a = 6 - b)
  (h2 : 6 + b = 9 + a) : 
  5 - a = 6 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_five_minus_a_l2221_222138


namespace NUMINAMATH_GPT_average_waiting_time_l2221_222131

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_average_waiting_time_l2221_222131


namespace NUMINAMATH_GPT_find_x2_plus_y2_l2221_222121

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
  sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l2221_222121


namespace NUMINAMATH_GPT_tan_double_angle_l2221_222158

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (2 * α) = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l2221_222158


namespace NUMINAMATH_GPT_octagon_perimeter_correct_l2221_222186

def octagon_perimeter (n : ℕ) (side_length : ℝ) : ℝ :=
  n * side_length

theorem octagon_perimeter_correct :
  octagon_perimeter 8 3 = 24 :=
by
  sorry

end NUMINAMATH_GPT_octagon_perimeter_correct_l2221_222186


namespace NUMINAMATH_GPT_two_point_distribution_p_value_l2221_222173

noncomputable def X : Type := ℕ -- discrete random variable (two-point)
def p (E_X2 : ℝ): ℝ := E_X2 -- p == E(X)

theorem two_point_distribution_p_value (var_X : ℝ) (E_X : ℝ) (E_X2 : ℝ) 
    (h1 : var_X = 2 / 9) 
    (h2 : E_X = p E_X2) 
    (h3 : E_X2 = E_X): 
    E_X = 1 / 3 ∨ E_X = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_two_point_distribution_p_value_l2221_222173


namespace NUMINAMATH_GPT_well_depth_l2221_222141

def daily_climb_up : ℕ := 4
def daily_slip_down : ℕ := 3
def total_days : ℕ := 27

theorem well_depth : (daily_climb_up * (total_days - 1) - daily_slip_down * (total_days - 1)) + daily_climb_up = 30 := by
  -- conditions
  let net_daily_progress := daily_climb_up - daily_slip_down
  let net_26_days_progress := net_daily_progress * (total_days - 1)

  -- proof to be completed
  sorry

end NUMINAMATH_GPT_well_depth_l2221_222141


namespace NUMINAMATH_GPT_combination_identity_l2221_222103

theorem combination_identity (C : ℕ → ℕ → ℕ)
  (comb_formula : ∀ n r, C r n = Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r)))
  (identity_1 : ∀ n r, C r n = C (n-r) n)
  (identity_2 : ∀ n r, C r (n+1) = C r n + C (r-1) n) :
  C 2 100 + C 97 100 = C 3 101 :=
by sorry

end NUMINAMATH_GPT_combination_identity_l2221_222103


namespace NUMINAMATH_GPT_length_of_first_train_is_270_l2221_222191

/-- 
Given:
1. Speed of the first train = 120 kmph
2. Speed of the second train = 80 kmph
3. Time to cross each other = 9 seconds
4. Length of the second train = 230.04 meters
  
Prove that the length of the first train is 270 meters.
-/
theorem length_of_first_train_is_270
  (speed_first_train : ℝ := 120)
  (speed_second_train : ℝ := 80)
  (time_to_cross : ℝ := 9)
  (length_second_train : ℝ := 230.04)
  (conversion_factor : ℝ := 1000/3600) :
  (length_second_train + (speed_first_train + speed_second_train) * conversion_factor * time_to_cross - length_second_train) = 270 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_train_is_270_l2221_222191


namespace NUMINAMATH_GPT_percentage_difference_l2221_222196

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.70 * x) (h2 : z = 1.50 * y) :
   x / z = 39.22 / 100 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l2221_222196


namespace NUMINAMATH_GPT_brass_total_l2221_222101

theorem brass_total (p_cu : ℕ) (p_zn : ℕ) (m_zn : ℕ) (B : ℕ) 
  (h_ratio : p_cu = 13) 
  (h_zn_ratio : p_zn = 7) 
  (h_zn_mass : m_zn = 35) : 
  (h_brass_total :  p_zn / (p_cu + p_zn) * B = m_zn) → B = 100 :=
sorry

end NUMINAMATH_GPT_brass_total_l2221_222101


namespace NUMINAMATH_GPT_descent_time_on_moving_escalator_standing_l2221_222161

theorem descent_time_on_moving_escalator_standing (l v_mont v_ek t : ℝ)
  (H1 : l / v_mont = 42)
  (H2 : l / (v_mont + v_ek) = 24)
  : t = 56 := by
  sorry

end NUMINAMATH_GPT_descent_time_on_moving_escalator_standing_l2221_222161


namespace NUMINAMATH_GPT_painted_cube_probability_l2221_222178

-- Define the conditions
def cube_size : Nat := 5
def total_unit_cubes : Nat := cube_size ^ 3
def corner_cubes_with_three_faces : Nat := 1
def edges_with_two_faces : Nat := 3 * (cube_size - 2) -- 3 edges, each (5 - 2) = 3
def faces_with_one_face : Nat := 2 * (cube_size * cube_size - corner_cubes_with_three_faces - edges_with_two_faces)
def no_painted_faces_cubes : Nat := total_unit_cubes - corner_cubes_with_three_faces - faces_with_one_face

-- Compute the probability
def probability := (corner_cubes_with_three_faces * no_painted_faces_cubes) / (total_unit_cubes * (total_unit_cubes - 1) / 2)

-- The theorem statement
theorem painted_cube_probability :
  probability = (2 : ℚ) / 155 := 
by {
  sorry
}

end NUMINAMATH_GPT_painted_cube_probability_l2221_222178


namespace NUMINAMATH_GPT_range_of_x_l2221_222197

open Real

theorem range_of_x (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 / a + 1 / b = 1) :
  a + b ≥ 3 + 2 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_x_l2221_222197


namespace NUMINAMATH_GPT_total_games_in_season_l2221_222194

-- Definitions based on the conditions
def num_teams := 16
def teams_per_division := 8
def num_divisions := num_teams / teams_per_division

-- Each team plays every other team in its division twice
def games_within_division_per_team := (teams_per_division - 1) * 2

-- Each team plays every team in the other division once
def games_across_divisions_per_team := teams_per_division

-- Total games per team
def games_per_team := games_within_division_per_team + games_across_divisions_per_team

-- Total preliminary games for all teams (each game is counted twice)
def preliminary_total_games := games_per_team * num_teams

-- Since each game is counted twice, the final number of games
def total_games := preliminary_total_games / 2

theorem total_games_in_season : total_games = 176 :=
by
  -- Sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_total_games_in_season_l2221_222194


namespace NUMINAMATH_GPT_surface_area_of_cone_l2221_222149

-- Definitions based solely on conditions
def central_angle (θ : ℝ) := θ = (2 * Real.pi) / 3
def slant_height (l : ℝ) := l = 2
def radius_cone (r : ℝ) := ∃ (θ l : ℝ), central_angle θ ∧ slant_height l ∧ θ * l = 2 * Real.pi * r
def lateral_surface_area (A₁ : ℝ) (r l : ℝ) := A₁ = Real.pi * r * l
def base_area (A₂ : ℝ) (r : ℝ) := A₂ = Real.pi * r^2
def total_surface_area (A A₁ A₂ : ℝ) := A = A₁ + A₂

-- The theorem proving the total surface area is as specified
theorem surface_area_of_cone :
  ∃ (r l A₁ A₂ A : ℝ), central_angle ((2 * Real.pi) / 3) ∧ slant_height 2 ∧ radius_cone r ∧
  lateral_surface_area A₁ r 2 ∧ base_area A₂ r ∧ total_surface_area A A₁ A₂ ∧ A = (16 * Real.pi) / 9 := sorry

end NUMINAMATH_GPT_surface_area_of_cone_l2221_222149


namespace NUMINAMATH_GPT_fraction_eq_zero_l2221_222108

theorem fraction_eq_zero {x : ℝ} (h : (6 * x) ≠ 0) : (x - 5) / (6 * x) = 0 ↔ x = 5 := 
by
  sorry

end NUMINAMATH_GPT_fraction_eq_zero_l2221_222108


namespace NUMINAMATH_GPT_horner_evaluation_at_two_l2221_222118

/-- Define the polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 * x^6 + 3 * x^5 + 5 * x^3 + 6 * x^2 + 7 * x + 8

/-- States that the value of f(2) using Horner's Rule equals 14. -/
theorem horner_evaluation_at_two : f 2 = 14 :=
sorry

end NUMINAMATH_GPT_horner_evaluation_at_two_l2221_222118


namespace NUMINAMATH_GPT_min_value_expression_l2221_222195

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) (hxy : x + y = 6) : 
  ( (x - 1)^2 / (y - 2) + ( (y - 1)^2 / (x - 2) ) ) >= 8 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_expression_l2221_222195


namespace NUMINAMATH_GPT_fill_time_calculation_l2221_222159

-- Definitions based on conditions
def pool_volume : ℝ := 24000
def number_of_hoses : ℕ := 6
def water_per_hose_per_minute : ℝ := 3
def minutes_per_hour : ℝ := 60

-- Theorem statement translating the mathematically equivalent proof problem
theorem fill_time_calculation :
  pool_volume / (number_of_hoses * water_per_hose_per_minute * minutes_per_hour) = 22 :=
by
  sorry

end NUMINAMATH_GPT_fill_time_calculation_l2221_222159


namespace NUMINAMATH_GPT_sum_first_100_sum_51_to_100_l2221_222169

noncomputable def sum_natural_numbers (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem sum_first_100 : sum_natural_numbers 100 = 5050 :=
  sorry

theorem sum_51_to_100 : sum_natural_numbers 100 - sum_natural_numbers 50 = 3775 :=
  sorry

end NUMINAMATH_GPT_sum_first_100_sum_51_to_100_l2221_222169


namespace NUMINAMATH_GPT_prob_selecting_green_ball_l2221_222154

-- Definition of the number of red and green balls in each container
def containerI_red := 10
def containerI_green := 5
def containerII_red := 3
def containerII_green := 5
def containerIII_red := 2
def containerIII_green := 6
def containerIV_red := 4
def containerIV_green := 4

-- Total number of balls in each container
def total_balls_I := containerI_red + containerI_green
def total_balls_II := containerII_red + containerII_green
def total_balls_III := containerIII_red + containerIII_green
def total_balls_IV := containerIV_red + containerIV_green

-- Probability of selecting a green ball from each container
def prob_green_I := containerI_green / total_balls_I
def prob_green_II := containerII_green / total_balls_II
def prob_green_III := containerIII_green / total_balls_III
def prob_green_IV := containerIV_green / total_balls_IV

-- Probability of selecting any one container
def prob_select_container := (1:ℚ) / 4

-- Combined probability for a green ball from each container
def combined_prob_I := prob_select_container * prob_green_I 
def combined_prob_II := prob_select_container * prob_green_II 
def combined_prob_III := prob_select_container * prob_green_III 
def combined_prob_IV := prob_select_container * prob_green_IV 

-- Total probability of selecting a green ball
def total_prob_green := combined_prob_I + combined_prob_II + combined_prob_III + combined_prob_IV 

-- Theorem to prove
theorem prob_selecting_green_ball : total_prob_green = 53 / 96 :=
by sorry

end NUMINAMATH_GPT_prob_selecting_green_ball_l2221_222154


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l2221_222112

noncomputable def hypotenuse_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem right_triangle_hypotenuse :
  ∀ (a b : ℝ),
  (1/3) * Real.pi * b^2 * a = 675 * Real.pi →
  (1/3) * Real.pi * a^2 * b = 1215 * Real.pi →
  hypotenuse_length a b = 3 * Real.sqrt 106 :=
  by
  intros a b h1 h2
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l2221_222112


namespace NUMINAMATH_GPT_nonneg_real_inequality_l2221_222167

theorem nonneg_real_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := 
by
  sorry

end NUMINAMATH_GPT_nonneg_real_inequality_l2221_222167


namespace NUMINAMATH_GPT_smallest_possible_value_of_AP_plus_BP_l2221_222185

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem smallest_possible_value_of_AP_plus_BP :
  let A := (1, 0)
  let B := (-3, 4)
  ∃ P : ℝ × ℝ, (P.2 ^ 2 = 4 * P.1) ∧
  (distance A P + distance B P = 12) :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_AP_plus_BP_l2221_222185


namespace NUMINAMATH_GPT_value_of_s_l2221_222176

-- Conditions: (m - 8) is a factor of m^2 - sm - 24

theorem value_of_s (s : ℤ) (m : ℤ) (h : (m - 8) ∣ (m^2 - s*m - 24)) : s = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_s_l2221_222176


namespace NUMINAMATH_GPT_sum_items_l2221_222198

theorem sum_items (A B : ℕ) (h1 : A = 585) (h2 : A = B + 249) : A + B = 921 :=
by
  -- Proof step skipped
  sorry

end NUMINAMATH_GPT_sum_items_l2221_222198


namespace NUMINAMATH_GPT_nat_pairs_solution_l2221_222144

theorem nat_pairs_solution (a b : ℕ) :
  a * (a + 5) = b * (b + 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) :=
by
  sorry

end NUMINAMATH_GPT_nat_pairs_solution_l2221_222144


namespace NUMINAMATH_GPT_largest_integer_solution_l2221_222147

theorem largest_integer_solution (x : ℤ) : 
  x < (92 / 21 : ℝ) → ∀ y : ℤ, y < (92 / 21 : ℝ) → y ≤ x :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_solution_l2221_222147


namespace NUMINAMATH_GPT_range_of_a_l2221_222135

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then -x^2 + 4 * x else Real.logb 2 x - a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → 1 < a :=
sorry

end NUMINAMATH_GPT_range_of_a_l2221_222135


namespace NUMINAMATH_GPT_project_completion_time_l2221_222164

theorem project_completion_time :
  let A_work_rate := (1 / 30) * (2 / 3)
  let B_work_rate := (1 / 60) * (3 / 4)
  let C_work_rate := (1 / 40) * (5 / 6)
  let combined_work_rate_per_12_days := 12 * (A_work_rate + B_work_rate + C_work_rate)
  let remaining_work_after_12_days := 1 - (2 / 3)
  let additional_work_rates_over_5_days := 
        5 * A_work_rate + 
        5 * B_work_rate + 
        5 * C_work_rate
  let remaining_work_after_5_days := remaining_work_after_12_days - additional_work_rates_over_5_days
  let B_additional_time := remaining_work_after_5_days / B_work_rate
  12 + 5 + B_additional_time = 17.5 :=
sorry

end NUMINAMATH_GPT_project_completion_time_l2221_222164


namespace NUMINAMATH_GPT_sign_of_a_l2221_222120

theorem sign_of_a (a b c d : ℝ) (h : b * (3 * d + 2) ≠ 0) (ineq : a / b < -c / (3 * d + 2)) : 
  (a = 0 ∨ a > 0 ∨ a < 0) :=
sorry

end NUMINAMATH_GPT_sign_of_a_l2221_222120


namespace NUMINAMATH_GPT_incorrect_proposition_C_l2221_222129

theorem incorrect_proposition_C (p q : Prop) :
  (¬(p ∨ q) → ¬p ∧ ¬q) ↔ False :=
by sorry

end NUMINAMATH_GPT_incorrect_proposition_C_l2221_222129


namespace NUMINAMATH_GPT_icosahedron_colorings_l2221_222137

theorem icosahedron_colorings :
  let n := 10
  let f := 9
  n! / 5 = 72576 :=
by
  sorry

end NUMINAMATH_GPT_icosahedron_colorings_l2221_222137


namespace NUMINAMATH_GPT_y_increases_as_x_increases_l2221_222152

-- Define the linear function y = (m^2 + 2)x
def linear_function (m x : ℝ) : ℝ := (m^2 + 2) * x

-- Prove that y increases as x increases
theorem y_increases_as_x_increases (m x1 x2 : ℝ) (h : x1 < x2) : linear_function m x1 < linear_function m x2 :=
by
  -- because m^2 + 2 is always positive, the function is strictly increasing
  have hm : 0 < m^2 + 2 := by linarith [pow_two_nonneg m]
  have hx : (m^2 + 2) * x1 < (m^2 + 2) * x2 := by exact (mul_lt_mul_left hm).mpr h
  exact hx

end NUMINAMATH_GPT_y_increases_as_x_increases_l2221_222152


namespace NUMINAMATH_GPT_high_school_students_total_l2221_222180

theorem high_school_students_total
    (students_taking_music : ℕ)
    (students_taking_art : ℕ)
    (students_taking_both_music_and_art : ℕ)
    (students_taking_neither : ℕ)
    (h1 : students_taking_music = 50)
    (h2 : students_taking_art = 20)
    (h3 : students_taking_both_music_and_art = 10)
    (h4 : students_taking_neither = 440) :
    students_taking_music - students_taking_both_music_and_art + students_taking_art - students_taking_both_music_and_art + students_taking_both_music_and_art + students_taking_neither = 500 :=
by
  sorry

end NUMINAMATH_GPT_high_school_students_total_l2221_222180


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l2221_222134

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l2221_222134


namespace NUMINAMATH_GPT_max_possible_value_of_a_l2221_222190

theorem max_possible_value_of_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 150) : 
  a ≤ 8924 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_possible_value_of_a_l2221_222190


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2221_222109

theorem quadratic_inequality_solution (a b: ℝ) (h1: ∀ x: ℝ, 1 < x ∧ x < 2 → ax^2 + bx - 4 > 0) (h2: ∀ x: ℝ, x ≤ 1 ∨ x ≥ 2 → ax^2 + bx - 4 ≤ 0) : a + b = 4 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2221_222109


namespace NUMINAMATH_GPT_sin_double_angle_l2221_222130

theorem sin_double_angle (x : ℝ) (h : Real.sin (Real.pi / 4 - x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 := by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l2221_222130


namespace NUMINAMATH_GPT_probability_of_x_in_interval_l2221_222106

noncomputable def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def probability_in_interval : ℝ :=
  let length_total := interval_length (-2) 1
  let length_sub := interval_length 0 1
  length_sub / length_total

theorem probability_of_x_in_interval :
  probability_in_interval = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_x_in_interval_l2221_222106


namespace NUMINAMATH_GPT_handshake_problem_l2221_222179

theorem handshake_problem (x : ℕ) (hx : (x * (x - 1)) / 2 = 55) : x = 11 := 
sorry

end NUMINAMATH_GPT_handshake_problem_l2221_222179


namespace NUMINAMATH_GPT_serves_probability_l2221_222153

variable (p : ℝ) (hpos : 0 < p) (hneq0 : p ≠ 0)

def ExpectedServes (p : ℝ) : ℝ :=
  p + 2 * p * (1 - p) + 3 * (1 - p) ^ 2

theorem serves_probability (h : ExpectedServes p > 1.75) : 0 < p ∧ p < 1 / 2 :=
  sorry

end NUMINAMATH_GPT_serves_probability_l2221_222153


namespace NUMINAMATH_GPT_number_of_solutions_pi_equation_l2221_222100

theorem number_of_solutions_pi_equation : 
  ∃ (x0 x1 : ℝ), (x0 = 0 ∧ x1 = 1) ∧ ∀ x : ℝ, (π^(x-1) * x^2 + π^(x^2) * x - π^(x^2) = x^2 + x - 1 ↔ x = x0 ∨ x = x1)
:=
by sorry

end NUMINAMATH_GPT_number_of_solutions_pi_equation_l2221_222100


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l2221_222132

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := abs (a - x)

def setA (a : ℝ) : Set ℝ := {x | f a (2 * x - 3 / 2) > 2 * f a (x + 2) + 2}

theorem problem_part1 {a : ℝ} (h : a = 3 / 2) : setA a = {x | x < 0} := by
  sorry

theorem problem_part2 {a : ℝ} (h : a = 3 / 2) (x0 : ℝ) (hx0 : x0 ∈ setA a) (x : ℝ) : 
    f a (x0 * x) ≥ x0 * f a x + f a (a * x0) := by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l2221_222132


namespace NUMINAMATH_GPT_proof_of_neg_p_or_neg_q_l2221_222126

variables (p q : Prop)

theorem proof_of_neg_p_or_neg_q (h₁ : ¬ (p ∧ q)) (h₂ : p ∨ q) : ¬ p ∨ ¬ q :=
  sorry

end NUMINAMATH_GPT_proof_of_neg_p_or_neg_q_l2221_222126


namespace NUMINAMATH_GPT_three_digit_number_digits_difference_l2221_222133

theorem three_digit_number_digits_difference (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : a < b) (h4 : b < c) :
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  reversed_number - original_number = 198 := by
  sorry

end NUMINAMATH_GPT_three_digit_number_digits_difference_l2221_222133


namespace NUMINAMATH_GPT_quadratic_real_roots_leq_l2221_222142

theorem quadratic_real_roots_leq (m : ℝ) :
  ∃ x : ℝ, x^2 - 3 * x + 2 * m = 0 → m ≤ 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_leq_l2221_222142


namespace NUMINAMATH_GPT_part_I_equality_condition_part_II_l2221_222155

-- Lean statement for Part (I)
theorem part_I (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) : 2 * Real.sqrt x + Real.sqrt (5 - x) ≤ 5 :=
sorry

theorem equality_condition (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) :
  (2 * Real.sqrt x + Real.sqrt (5 - x) = 5) ↔ (x = 4) :=
sorry

-- Lean statement for Part (II)
theorem part_II (m : ℝ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 5) → 2 * Real.sqrt x + Real.sqrt (5 - x) ≤ |m - 2|) →
  (m ≥ 7 ∨ m ≤ -3) :=
sorry

end NUMINAMATH_GPT_part_I_equality_condition_part_II_l2221_222155


namespace NUMINAMATH_GPT_arrangements_21_leaders_l2221_222110

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the permutations A_n^k
def permutations (n k : ℕ) : ℕ :=
  if k ≤ n then factorial n / factorial (n - k) else 0

theorem arrangements_21_leaders : permutations 2 2 * permutations 18 18 = factorial 18 ^ 2 :=
by 
  sorry

end NUMINAMATH_GPT_arrangements_21_leaders_l2221_222110


namespace NUMINAMATH_GPT_num_books_second_shop_l2221_222145

-- Define the conditions
def num_books_first_shop : ℕ := 32
def cost_first_shop : ℕ := 1500
def cost_second_shop : ℕ := 340
def avg_price_per_book : ℕ := 20

-- Define the proof statement
theorem num_books_second_shop : 
  (num_books_first_shop + (cost_second_shop + cost_first_shop) / avg_price_per_book) - num_books_first_shop = 60 := by
  sorry

end NUMINAMATH_GPT_num_books_second_shop_l2221_222145


namespace NUMINAMATH_GPT_calculate_geometric_sequence_sum_l2221_222140

def geometric_sequence (a₁ r : ℤ) (n : ℕ) : ℤ :=
  a₁ * r^n

theorem calculate_geometric_sequence_sum :
  let a₁ := 1
  let r := -2
  let a₂ := geometric_sequence a₁ r 1
  let a₃ := geometric_sequence a₁ r 2
  let a₄ := geometric_sequence a₁ r 3
  a₁ + |a₂| + a₃ + |a₄| = 15 :=
by
  sorry

end NUMINAMATH_GPT_calculate_geometric_sequence_sum_l2221_222140


namespace NUMINAMATH_GPT_largest_perfect_square_factor_1760_l2221_222181

theorem largest_perfect_square_factor_1760 :
  ∃ n, (∃ k, n = k^2) ∧ n ∣ 1760 ∧ ∀ m, (∃ j, m = j^2) ∧ m ∣ 1760 → m ≤ n := by
  sorry

end NUMINAMATH_GPT_largest_perfect_square_factor_1760_l2221_222181


namespace NUMINAMATH_GPT_problem1_problem2_l2221_222187

theorem problem1 (a b c : ℝ) (h1 : a = 5.42) (h2 : b = 3.75) (h3 : c = 0.58) :
  a - (b - c) = 2.25 :=
by sorry

theorem problem2 (d e f g h : ℝ) (h4 : d = 4 / 5) (h5 : e = 7.7) (h6 : f = 0.8) (h7 : g = 3.3) (h8 : h = 1) :
  d * e + f * g - d = 8 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2221_222187


namespace NUMINAMATH_GPT_smallest_a_l2221_222182

theorem smallest_a (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) : a = 17 :=
sorry

end NUMINAMATH_GPT_smallest_a_l2221_222182


namespace NUMINAMATH_GPT_rent_percentage_increase_l2221_222114

theorem rent_percentage_increase 
  (E : ℝ) 
  (h1 : ∀ (E : ℝ), rent_last_year = 0.25 * E)
  (h2 : ∀ (E : ℝ), earnings_this_year = 1.45 * E)
  (h3 : ∀ (E : ℝ), rent_this_year = 0.35 * earnings_this_year) :
  (rent_this_year / rent_last_year) * 100 = 203 := 
by 
  sorry

end NUMINAMATH_GPT_rent_percentage_increase_l2221_222114


namespace NUMINAMATH_GPT_avg_marks_chem_math_l2221_222139

variable (P C M : ℝ)

theorem avg_marks_chem_math (h : P + C + M = P + 140) : (C + M) / 2 = 70 :=
by
  -- skip the proof, just provide the statement
  sorry

end NUMINAMATH_GPT_avg_marks_chem_math_l2221_222139


namespace NUMINAMATH_GPT_sum_of_sides_of_similar_triangle_l2221_222127

theorem sum_of_sides_of_similar_triangle (a b c : ℕ) (scale_factor : ℕ) (longest_side_sim : ℕ) (sum_of_other_sides_sim : ℕ) : 
  a * scale_factor = 21 → c = 7 → b = 5 → a = 3 → 
  sum_of_other_sides = a * scale_factor + b * scale_factor → 
sum_of_other_sides = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_sides_of_similar_triangle_l2221_222127


namespace NUMINAMATH_GPT_SamaraSpentOnDetailing_l2221_222166

def costSamara (D : ℝ) : ℝ := 25 + 467 + D
def costAlberto : ℝ := 2457
def difference : ℝ := 1886

theorem SamaraSpentOnDetailing : 
  ∃ (D : ℝ), costAlberto = costSamara D + difference ∧ D = 79 := 
sorry

end NUMINAMATH_GPT_SamaraSpentOnDetailing_l2221_222166


namespace NUMINAMATH_GPT_greatest_possible_fourth_term_l2221_222122

theorem greatest_possible_fourth_term {a d : ℕ} (h : 5 * a + 10 * d = 60) : a + 3 * (12 - a) ≤ 34 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_possible_fourth_term_l2221_222122


namespace NUMINAMATH_GPT_lives_per_each_player_l2221_222105

def num_initial_players := 8
def num_quit_players := 3
def total_remaining_lives := 15
def num_remaining_players := num_initial_players - num_quit_players
def lives_per_remaining_player := total_remaining_lives / num_remaining_players

theorem lives_per_each_player :
  lives_per_remaining_player = 3 := by
  sorry

end NUMINAMATH_GPT_lives_per_each_player_l2221_222105


namespace NUMINAMATH_GPT_blue_ball_weight_l2221_222111

variable (b t x : ℝ)
variable (c1 : b = 3.12)
variable (c2 : t = 9.12)
variable (c3 : t = b + x)

theorem blue_ball_weight : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_blue_ball_weight_l2221_222111


namespace NUMINAMATH_GPT_find_a8_l2221_222136

variable {α : Type} [LinearOrderedField α]

/-- Given conditions of an arithmetic sequence -/
def arithmetic_sequence (a_n : ℕ → α) : Prop :=
  ∃ (a1 d : α), ∀ n : ℕ, a_n n = a1 + n * d

theorem find_a8 (a_n : ℕ → ℝ)
  (h_arith : arithmetic_sequence a_n)
  (h3 : a_n 3 = 5)
  (h5 : a_n 5 = 3) :
  a_n 8 = 0 :=
sorry

end NUMINAMATH_GPT_find_a8_l2221_222136


namespace NUMINAMATH_GPT_find_m_for_parallel_vectors_l2221_222115

theorem find_m_for_parallel_vectors (m : ℝ) :
  let a := (1, m)
  let b := (2, -1)
  (2 * a.1 + b.1, 2 * a.2 + b.2) = (k * (a.1 - 2 * b.1), k * (a.2 - 2 * b.2)) → m = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_for_parallel_vectors_l2221_222115


namespace NUMINAMATH_GPT_three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four_l2221_222183

theorem three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four :
  (3.242 * 12) / 100 = 0.38904 :=
by 
  sorry

end NUMINAMATH_GPT_three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four_l2221_222183


namespace NUMINAMATH_GPT_min_value_f_l2221_222189

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1)^2 + (Real.exp (-x) - 1)^2

theorem min_value_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -2 :=
sorry

end NUMINAMATH_GPT_min_value_f_l2221_222189


namespace NUMINAMATH_GPT_bouquet_carnations_l2221_222162

def proportion_carnations (P : ℚ) (R : ℚ) (PC : ℚ) (RC : ℚ) : ℚ := PC + RC

theorem bouquet_carnations :
  let P := (7 / 10 : ℚ)
  let R := (3 / 10 : ℚ)
  let PC := (1 / 2) * P
  let RC := (2 / 3) * R
  let C := proportion_carnations P R PC RC
  (C * 100) = 55 :=
by
  sorry

end NUMINAMATH_GPT_bouquet_carnations_l2221_222162


namespace NUMINAMATH_GPT_handshakes_minimum_l2221_222128

/-- Given 30 people and each person shakes hands with exactly three others,
    the minimum possible number of handshakes is 45. -/
theorem handshakes_minimum (n k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  (n * k) / 2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_handshakes_minimum_l2221_222128


namespace NUMINAMATH_GPT_time_without_moving_walkway_l2221_222113

/--
Assume a person walks from one end to the other of a 90-meter long moving walkway at a constant rate in 30 seconds, assisted by the walkway. When this person reaches the end, they reverse direction and continue walking with the same speed, but this time it takes 120 seconds because the person is traveling against the direction of the moving walkway.

Prove that if the walkway were to stop moving, it would take this person 48 seconds to walk from one end of the walkway to the other.
-/
theorem time_without_moving_walkway : 
  ∀ (v_p v_w : ℝ),
  (v_p + v_w) * 30 = 90 →
  (v_p - v_w) * 120 = 90 →
  90 / v_p = 48 :=
by
  intros v_p v_w h1 h2
  have hpw := eq_of_sub_eq_zero (sub_eq_zero.mpr h1)
  have hmw := eq_of_sub_eq_zero (sub_eq_zero.mpr h2)
  sorry

end NUMINAMATH_GPT_time_without_moving_walkway_l2221_222113


namespace NUMINAMATH_GPT_cone_cylinder_volume_ratio_l2221_222171

theorem cone_cylinder_volume_ratio (h r : ℝ) (hc_pos : h > 0) (r_pos : r > 0) :
  let V_cylinder := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * (3 / 4 * h)
  (V_cone / V_cylinder) = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_cone_cylinder_volume_ratio_l2221_222171


namespace NUMINAMATH_GPT_find_capacity_of_second_vessel_l2221_222148

noncomputable def capacity_of_second_vessel (x : ℝ) : Prop :=
  let alcohol_from_first_vessel := 0.25 * 2
  let alcohol_from_second_vessel := 0.40 * x
  let total_liquid := 2 + x
  let total_alcohol := alcohol_from_first_vessel + alcohol_from_second_vessel
  let new_concentration := (total_alcohol / 10) * 100
  2 + x = 8 ∧ new_concentration = 29

open scoped Real

theorem find_capacity_of_second_vessel : ∃ x : ℝ, capacity_of_second_vessel x ∧ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_capacity_of_second_vessel_l2221_222148


namespace NUMINAMATH_GPT_pencils_calculation_l2221_222184

variable (C B D : ℕ)

theorem pencils_calculation : 
  (C = B + 5) ∧
  (B = 2 * D - 3) ∧
  (C = 20) →
  D = 9 :=
by sorry

end NUMINAMATH_GPT_pencils_calculation_l2221_222184


namespace NUMINAMATH_GPT_polynomial_factors_l2221_222172

theorem polynomial_factors (t q : ℤ) (h1 : 81 - 3 * t + q = 0) (h2 : -3 + t + q = 0) : |3 * t - 2 * q| = 99 :=
sorry

end NUMINAMATH_GPT_polynomial_factors_l2221_222172


namespace NUMINAMATH_GPT_inscribed_cube_volume_l2221_222193

noncomputable def side_length_of_inscribed_cube (d : ℝ) : ℝ :=
d / Real.sqrt 3

noncomputable def volume_of_inscribed_cube (s : ℝ) : ℝ :=
s^3

theorem inscribed_cube_volume :
  (volume_of_inscribed_cube (side_length_of_inscribed_cube 12)) = 192 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_cube_volume_l2221_222193


namespace NUMINAMATH_GPT_inserting_eights_is_composite_l2221_222177

theorem inserting_eights_is_composite (n : ℕ) : ¬ Nat.Prime (2000 * 10^n + 8 * ((10^n - 1) / 9) + 21) := 
by sorry

end NUMINAMATH_GPT_inserting_eights_is_composite_l2221_222177


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l2221_222151

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 4) : y = 2 * x - 4 :=
by
  sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l2221_222151


namespace NUMINAMATH_GPT_cistern_leak_time_l2221_222163

theorem cistern_leak_time (R : ℝ) (L : ℝ) (eff_R : ℝ) : 
  (R = 1/5) → 
  (eff_R = 1/6) → 
  (eff_R = R - L) → 
  (1 / L = 30) :=
by
  intros hR heffR heffRate
  sorry

end NUMINAMATH_GPT_cistern_leak_time_l2221_222163


namespace NUMINAMATH_GPT_sum_inequality_l2221_222123

theorem sum_inequality (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (1 / (b * (a + b))) + (1 / (c * (b + c))) + (1 / (a * (c + a))) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_sum_inequality_l2221_222123


namespace NUMINAMATH_GPT_minimum_product_xyz_l2221_222119

noncomputable def minimalProduct (x y z : ℝ) : ℝ :=
  3 * x^2 * (1 - 4 * x)

theorem minimum_product_xyz :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  z = 3 * x →
  x ≤ y ∧ y ≤ z →
  minimalProduct x y z = (9 / 343) :=
by
  intros x y z x_pos y_pos z_pos sum_eq1 z_eq3x inequalities
  sorry

end NUMINAMATH_GPT_minimum_product_xyz_l2221_222119


namespace NUMINAMATH_GPT_positive_difference_even_odd_sum_l2221_222199

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end NUMINAMATH_GPT_positive_difference_even_odd_sum_l2221_222199


namespace NUMINAMATH_GPT_percent_increase_stock_l2221_222124

theorem percent_increase_stock (P_open P_close: ℝ) (h1: P_open = 30) (h2: P_close = 45):
  (P_close - P_open) / P_open * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_stock_l2221_222124


namespace NUMINAMATH_GPT_expression_equality_l2221_222168

theorem expression_equality (a b c : ℝ) : a * (a + b - c) = a^2 + a * b - a * c :=
by
  sorry

end NUMINAMATH_GPT_expression_equality_l2221_222168


namespace NUMINAMATH_GPT_initial_leaves_l2221_222165

theorem initial_leaves (l_0 : ℕ) (blown_away : ℕ) (leaves_left : ℕ) (h1 : blown_away = 244) (h2 : leaves_left = 112) (h3 : l_0 - blown_away = leaves_left) : l_0 = 356 :=
by
  sorry

end NUMINAMATH_GPT_initial_leaves_l2221_222165


namespace NUMINAMATH_GPT_Petya_workout_duration_l2221_222188

theorem Petya_workout_duration :
  ∃ x : ℕ, (x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 135) ∧
            (x + 7 > x) ∧
            (x + 14 > x + 7) ∧
            (x + 21 > x + 14) ∧
            (x + 28 > x + 21) ∧
            x = 13 :=
by sorry

end NUMINAMATH_GPT_Petya_workout_duration_l2221_222188


namespace NUMINAMATH_GPT_rainfall_difference_l2221_222156

-- Define the conditions
def day1_rainfall := 26
def day2_rainfall := 34
def average_rainfall := 140
def less_rainfall := 58

-- Calculate the total rainfall this year in the first three days
def total_rainfall_this_year := average_rainfall - less_rainfall

-- Calculate the total rainfall in the first two days
def total_first_two_days := day1_rainfall + day2_rainfall

-- Calculate the rainfall on the third day
def day3_rainfall := total_rainfall_this_year - total_first_two_days

-- The proof problem
theorem rainfall_difference : day2_rainfall - day3_rainfall = 12 := 
by
  sorry

end NUMINAMATH_GPT_rainfall_difference_l2221_222156


namespace NUMINAMATH_GPT_value_of_a_plus_b_l2221_222125

theorem value_of_a_plus_b (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 3 ↔ ax^2 + bx + 3 < 0) :
  a + b = -3 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l2221_222125


namespace NUMINAMATH_GPT_interval_monotonicity_minimum_value_range_of_a_l2221_222116

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a / x

theorem interval_monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x, 0 < x ∧ x < a → f x a > 0) ∧ (∀ x, x > a → f x a < 0) :=
sorry

theorem minimum_value (a : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ Real.exp 1 → f x a ≥ 1) ∧ (∃ x, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f x a = 1) → a = 1 :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x, x > 1 → f x a < 1 / 2 * x) → a < 1 / 2 :=
sorry

end NUMINAMATH_GPT_interval_monotonicity_minimum_value_range_of_a_l2221_222116


namespace NUMINAMATH_GPT_apples_in_each_box_l2221_222146

variable (A : ℕ)
variable (ApplesSaturday : ℕ := 50 * A)
variable (ApplesSunday : ℕ := 25 * A)
variable (ApplesLeft : ℕ := 3 * A)
variable (ApplesSold : ℕ := 720)

theorem apples_in_each_box :
  (ApplesSaturday + ApplesSunday - ApplesSold = ApplesLeft) → A = 10 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_each_box_l2221_222146


namespace NUMINAMATH_GPT_max_probability_of_winning_is_correct_l2221_222102

noncomputable def max_probability_of_winning : ℚ :=
  sorry

theorem max_probability_of_winning_is_correct :
  max_probability_of_winning = 17 / 32 :=
sorry

end NUMINAMATH_GPT_max_probability_of_winning_is_correct_l2221_222102


namespace NUMINAMATH_GPT_minimum_norm_of_v_l2221_222104

open Real 

-- Define the vector v and condition
noncomputable def v : ℝ × ℝ := sorry

-- Define the condition
axiom v_condition : ‖(v.1 + 4, v.2 + 2)‖ = 10

-- The statement that we need to prove
theorem minimum_norm_of_v : ‖v‖ = 10 - 2 * sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_norm_of_v_l2221_222104


namespace NUMINAMATH_GPT_sum_of_fractions_eq_one_l2221_222174

variable {a b c d : ℝ} (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)
          (h_equiv : (a * d + b * c) / (b * d) = (a * c) / (b * d))

theorem sum_of_fractions_eq_one : b / a + d / c = 1 :=
by sorry

end NUMINAMATH_GPT_sum_of_fractions_eq_one_l2221_222174
