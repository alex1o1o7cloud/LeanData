import Mathlib

namespace remainder_of_expression_l147_147313

theorem remainder_of_expression (a b c d : ℕ) (h1 : a = 8) (h2 : b = 20) (h3 : c = 34) (h4 : d = 3) :
  (a * b ^ c + d ^ c) % 7 = 5 := 
by 
  rw [h1, h2, h3, h4]
  sorry

end remainder_of_expression_l147_147313


namespace remaining_amount_eq_40_l147_147073

-- Definitions and conditions
def initial_amount : ℕ := 100
def food_spending : ℕ := 20
def rides_spending : ℕ := 2 * food_spending
def total_spending : ℕ := food_spending + rides_spending

-- The proposition to be proved
theorem remaining_amount_eq_40 :
  initial_amount - total_spending = 40 :=
by
  sorry

end remaining_amount_eq_40_l147_147073


namespace sufficient_condition_l147_147062

theorem sufficient_condition (a b : ℝ) (h : |a + b| > 1) : |a| + |b| > 1 := 
by sorry

end sufficient_condition_l147_147062


namespace no_x_satisfies_inequality_l147_147599

def f (x : ℝ) : ℝ := x^2 + x

theorem no_x_satisfies_inequality : ¬ ∃ x : ℝ, f (x - 2) + f x < 0 :=
by 
  unfold f 
  sorry

end no_x_satisfies_inequality_l147_147599


namespace distinct_arrangements_balloon_l147_147564

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l147_147564


namespace find_d_l147_147481

theorem find_d (c : ℕ) (d : ℕ) : 
  (∀ n : ℕ, c = 3 ∧ ∀ k : ℕ, k ≠ 30 → ((1 : ℚ) * (29 / 30) * (28 / 30) = 203 / 225) → d = 203) := 
by
  intros
  sorry

end find_d_l147_147481


namespace lunch_cost_before_tip_l147_147769

theorem lunch_cost_before_tip (C : ℝ) (h : C + 0.20 * C = 60.24) : C = 50.20 :=
sorry

end lunch_cost_before_tip_l147_147769


namespace solve_floor_trig_eq_l147_147902

-- Define the floor function
def floor (x : ℝ) : ℤ := by 
  sorry

-- Define the condition and theorem
theorem solve_floor_trig_eq (x : ℝ) (n : ℤ) : 
  floor (Real.sin x + Real.cos x) = 1 ↔ (∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ (2 * Real.pi * n + Real.pi / 2)) := 
  by 
  sorry

end solve_floor_trig_eq_l147_147902


namespace container_volume_ratio_l147_147425

theorem container_volume_ratio
  (A B C : ℝ)
  (h1 : (3 / 4) * A - (5 / 8) * B = (7 / 8) * C - (1 / 2) * C)
  (h2 : B =  (5 / 8) * B)
  (h3 : (5 / 8) * B =  (3 / 8) * C)
  (h4 : A =  (24 / 40) * C) : 
  A / C = 4 / 5 := sorry

end container_volume_ratio_l147_147425


namespace iterative_average_difference_l147_147630

theorem iterative_average_difference :
  let numbers : List ℕ := [2, 4, 6, 8, 10] 
  let avg2 (a b : ℝ) := (a + b) / 2
  let avg (init : ℝ) (lst : List ℕ) := lst.foldl (λ acc x => avg2 acc x) init
  let max_avg := avg 2 [4, 6, 8, 10]
  let min_avg := avg 10 [8, 6, 4, 2] 
  max_avg - min_avg = 4.25 := 
by
  sorry

end iterative_average_difference_l147_147630


namespace pizzas_returned_l147_147811

theorem pizzas_returned (total_pizzas served_pizzas : ℕ) (h_total : total_pizzas = 9) (h_served : served_pizzas = 3) : (total_pizzas - served_pizzas) = 6 :=
by
  sorry

end pizzas_returned_l147_147811


namespace quadratic_inequality_solution_l147_147162

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (x^2 - 4*x + 3) < 0 ↔ 1 < x ∧ x < 3 :=
by
  sorry

end quadratic_inequality_solution_l147_147162


namespace which_is_lying_l147_147530

-- Ben's statement
def ben_says (dan_truth cam_truth : Bool) : Bool :=
  (dan_truth ∧ ¬ cam_truth) ∨ (¬ dan_truth ∧ cam_truth)

-- Dan's statement
def dan_says (ben_truth cam_truth : Bool) : Bool :=
  (ben_truth ∧ ¬ cam_truth) ∨ (¬ ben_truth ∧ cam_truth)

-- Cam's statement
def cam_says (ben_truth dan_truth : Bool) : Bool :=
  ¬ ben_truth ∧ ¬ dan_truth

-- Lean statement to be proven
theorem which_is_lying :
  (∃ (ben_truth dan_truth cam_truth : Bool), 
    ben_says dan_truth cam_truth ∧ 
    dan_says ben_truth cam_truth ∧ 
    cam_says ben_truth dan_truth ∧
    ¬ ben_truth ∧ ¬ dan_truth ∧ cam_truth) ↔ (¬ ben_truth ∧ ¬ dan_truth ∧ cam_truth) :=
sorry

end which_is_lying_l147_147530


namespace roots_of_polynomial_l147_147653

noncomputable def poly (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem roots_of_polynomial :
  ∀ x : ℝ, poly x = 0 ↔ (x = -1 ∨ x = 1 ∨ x = 2) :=
by
  sorry

end roots_of_polynomial_l147_147653


namespace complex_num_sum_l147_147250

def is_complex_num (a b : ℝ) (z : ℂ) : Prop :=
  z = a + b * Complex.I

theorem complex_num_sum (a b : ℝ) (z : ℂ) (h : is_complex_num a b z) :
  z = (1 - Complex.I) ^ 2 / (1 + Complex.I) → a + b = -2 :=
by
  sorry

end complex_num_sum_l147_147250


namespace eggs_left_after_taking_l147_147874

def eggs_in_box_initial : Nat := 47
def eggs_taken_by_Harry : Nat := 5
theorem eggs_left_after_taking : eggs_in_box_initial - eggs_taken_by_Harry = 42 := 
by
  -- Proof placeholder
  sorry

end eggs_left_after_taking_l147_147874


namespace largest_y_l147_147427

theorem largest_y : ∃ (y : ℤ), (y ≤ 3) ∧ (∀ (z : ℤ), (z > y) → ¬ (z / 4 + 6 / 7 < 7 / 4)) :=
by
  -- There exists an integer y such that y <= 3 and for all integers z greater than y, the inequality does not hold
  sorry

end largest_y_l147_147427


namespace total_number_of_marbles_is_1050_l147_147166

def total_marbles : Nat :=
  let marbles_in_second_bowl := 600
  let marbles_in_first_bowl := (3 * marbles_in_second_bowl) / 4
  marbles_in_first_bowl + marbles_in_second_bowl

theorem total_number_of_marbles_is_1050 : total_marbles = 1050 := by
  sorry

end total_number_of_marbles_is_1050_l147_147166


namespace original_amount_of_milk_is_720_l147_147258

variable (M : ℝ) -- The original amount of milk in milliliters

theorem original_amount_of_milk_is_720 :
  ((5 / 6) * M) - ((2 / 5) * ((5 / 6) * M)) - ((2 / 3) * (((5 / 6) * M) - ((2 / 5) * ((5 / 6) * M)))) = 120 → 
  M = 720 := by
  sorry

end original_amount_of_milk_is_720_l147_147258


namespace midpoint_of_intersection_l147_147240

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 2 * t)

noncomputable def polar_curve (θ : ℝ) : ℝ :=
  2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2)

theorem midpoint_of_intersection :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, 
      A = parametric_line t₁ ∧ 
      B = parametric_line t₂ ∧ 
      (A.1 ^ 2 / 4 + A.2 ^ 2 = 1) ∧ 
      (B.1 ^ 2 / 4 + B.2 ^ 2 = 1)) ∧
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (4 / 5, -1 / 5) :=
sorry

end midpoint_of_intersection_l147_147240


namespace power_difference_l147_147844

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end power_difference_l147_147844


namespace domain_of_fx_l147_147849

theorem domain_of_fx {x : ℝ} : (2 * x) / (x - 1) = (2 * x) / (x - 1) ↔ x ∈ {y : ℝ | y ≠ 1} :=
by
  sorry

end domain_of_fx_l147_147849


namespace inequality_of_ab_bc_ca_l147_147640

theorem inequality_of_ab_bc_ca (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : a^4 + b^4 + c^4 = 3) : 
  (1 / (4 - a * b)) + (1 / (4 - b * c)) + (1 / (4 - c * a)) ≤ 1 :=
by
  sorry

end inequality_of_ab_bc_ca_l147_147640


namespace jame_weeks_tearing_cards_l147_147873

def cards_tears_per_time : ℕ := 30
def cards_per_deck : ℕ := 55
def tears_per_week : ℕ := 3
def decks_bought : ℕ := 18

theorem jame_weeks_tearing_cards :
  (cards_tears_per_time * tears_per_week * decks_bought * cards_per_deck) / (cards_tears_per_time * tears_per_week) = 11 := by
  sorry

end jame_weeks_tearing_cards_l147_147873


namespace binom_coeff_mult_l147_147780

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l147_147780


namespace minimum_value_of_sum_l147_147670

noncomputable def left_focus (a b c : ℝ) : ℝ := -c 

noncomputable def right_focus (a b c : ℝ) : ℝ := c

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
magnitude (q.1 - p.1, q.2 - p.2)

def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = 1

def P_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola_eq P.1 P.2

theorem minimum_value_of_sum (P : ℝ × ℝ) (A : ℝ × ℝ) (F F' : ℝ × ℝ) (a b c : ℝ)
  (h1 : F = (-c, 0)) (h2 : F' = (c, 0)) (h3 : A = (1, 4)) (h4 : 2 * a = 4)
  (h5 : c^2 = a^2 + b^2) (h6 : P_on_hyperbola P) :
  (|distance P F| + |distance P A|) ≥ 9 :=
sorry

end minimum_value_of_sum_l147_147670


namespace smallest_positive_integer_mod_l147_147520

theorem smallest_positive_integer_mod (a : ℕ) (h1 : a ≡ 4 [MOD 5]) (h2 : a ≡ 6 [MOD 7]) : a = 34 :=
by
  sorry

end smallest_positive_integer_mod_l147_147520


namespace angle_between_lines_is_arctan_one_third_l147_147993

theorem angle_between_lines_is_arctan_one_third
  (l1 : ∀ x y : ℝ, 2 * x - y + 1 = 0)
  (l2 : ∀ x y : ℝ, x - y - 2 = 0)
  : ∃ θ : ℝ, θ = Real.arctan (1 / 3) := 
sorry

end angle_between_lines_is_arctan_one_third_l147_147993


namespace range_of_values_l147_147290

theorem range_of_values (x : ℝ) (h1 : x - 1 ≥ 0) (h2 : x ≠ 0) : x ≥ 1 := 
sorry

end range_of_values_l147_147290


namespace simplify_expression_l147_147464

theorem simplify_expression (x : ℝ) : 2 * x * (x - 4) - (2 * x - 3) * (x + 2) = -9 * x + 6 :=
by
  sorry

end simplify_expression_l147_147464


namespace line_through_point_parallel_l147_147719

theorem line_through_point_parallel 
    (x y : ℝ)
    (h0 : (x = -1) ∧ (y = 3))
    (h1 : ∃ c : ℝ, (∀ x y : ℝ, x - 2 * y + c = 0 ↔ x - 2 * y + 3 = 0)) :
     ∃ c : ℝ, ∀ x y : ℝ, (x = -1) ∧ (y = 3) → (∃ (a b : ℝ), a - 2 * b + c = 0) :=
by
  sorry

end line_through_point_parallel_l147_147719


namespace exist_students_with_comparable_scores_l147_147177

theorem exist_students_with_comparable_scores :
  ∃ (A B : ℕ) (a1 a2 a3 b1 b2 b3 : ℕ), 
    A ≠ B ∧ A < 49 ∧ B < 49 ∧
    (0 ≤ a1 ∧ a1 ≤ 7) ∧ (0 ≤ a2 ∧ a2 ≤ 7) ∧ (0 ≤ a3 ∧ a3 ≤ 7) ∧ 
    (0 ≤ b1 ∧ b1 ≤ 7) ∧ (0 ≤ b2 ∧ b2 ≤ 7) ∧ (0 ≤ b3 ∧ b3 ≤ 7) ∧ 
    (a1 ≥ b1) ∧ (a2 ≥ b2) ∧ (a3 ≥ b3) := 
sorry

end exist_students_with_comparable_scores_l147_147177


namespace bus_stops_for_4_minutes_per_hour_l147_147584

theorem bus_stops_for_4_minutes_per_hour
  (V_excluding_stoppages V_including_stoppages : ℝ)
  (h1 : V_excluding_stoppages = 90)
  (h2 : V_including_stoppages = 84) :
  (60 * (V_excluding_stoppages - V_including_stoppages)) / V_excluding_stoppages = 4 :=
by
  sorry

end bus_stops_for_4_minutes_per_hour_l147_147584


namespace a_number_M_middle_digit_zero_l147_147077

theorem a_number_M_middle_digit_zero (d e f M : ℕ) (h1 : M = 36 * d + 6 * e + f)
  (h2 : M = 64 * f + 8 * e + d) (hd : d < 6) (he : e < 6) (hf : f < 6) : e = 0 :=
by sorry

end a_number_M_middle_digit_zero_l147_147077


namespace total_meters_examined_l147_147856

-- Define the conditions
def proportion_defective : ℝ := 0.1
def defective_meters : ℕ := 10

-- The statement to prove
theorem total_meters_examined (T : ℝ) (h : proportion_defective * T = defective_meters) : T = 100 :=
by
  sorry

end total_meters_examined_l147_147856


namespace points_per_touchdown_l147_147108

theorem points_per_touchdown (number_of_touchdowns : ℕ) (total_points : ℕ) (h1 : number_of_touchdowns = 3) (h2 : total_points = 21) : (total_points / number_of_touchdowns) = 7 :=
by
  sorry

end points_per_touchdown_l147_147108


namespace max_xy_l147_147575

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 7 * x + 8 * y = 112) : xy ≤ 56 :=
sorry

end max_xy_l147_147575


namespace science_and_technology_group_total_count_l147_147246

theorem science_and_technology_group_total_count 
  (number_of_girls : ℕ)
  (number_of_boys : ℕ)
  (h1 : number_of_girls = 18)
  (h2 : number_of_girls = 2 * number_of_boys - 2)
  : number_of_girls + number_of_boys = 28 := 
by
  sorry

end science_and_technology_group_total_count_l147_147246


namespace option_A_is_linear_equation_l147_147527

-- Definitions for considering an equation being linear in two variables
def is_linear_equation (e : Prop) : Prop :=
  ∃ (a b c : ℝ), e = (a = b + c) ∧ a ≠ 0 ∧ b ≠ 0

-- The given equation in option A
def Eq_A : Prop := ∀ (x y : ℝ), (2 * y - 1) / 5 = 2 - (3 * x - 2) / 4

-- Proof problem statement
theorem option_A_is_linear_equation : is_linear_equation Eq_A :=
sorry

end option_A_is_linear_equation_l147_147527


namespace shaded_area_of_hexagon_with_semicircles_l147_147500

theorem shaded_area_of_hexagon_with_semicircles :
  let s := 3
  let r := 3 / 2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let semicircle_area := 3 * (1/2 * Real.pi * r^2)
  let shaded_area := hexagon_area - semicircle_area
  shaded_area = 13.5 * Real.sqrt 3 - 27 * Real.pi / 8 :=
by
  sorry

end shaded_area_of_hexagon_with_semicircles_l147_147500


namespace charlie_max_success_ratio_l147_147194

-- Given:
-- Alpha scored 180 points out of 360 attempted on day one.
-- Alpha scored 120 points out of 240 attempted on day two.
-- Charlie did not attempt 360 points on the first day.
-- Charlie's success ratio on each day was less than Alpha’s.
-- Total points attempted by Charlie on both days are 600.
-- Alpha's two-day success ratio is 300/600 = 1/2.
-- Find the largest possible two-day success ratio that Charlie could have achieved.

theorem charlie_max_success_ratio:
  ∀ (x y z w : ℕ),
  0 < x ∧ 0 < z ∧ 0 < y ∧ 0 < w ∧
  y + w = 600 ∧
  (2 * x < y) ∧ (2 * z < w) ∧
  (x + z < 300) -> (299 / 600 = 299 / 600) :=
by
  sorry

end charlie_max_success_ratio_l147_147194


namespace speed_of_stream_l147_147068

theorem speed_of_stream (x : ℝ) (boat_speed : ℝ) (distance_one_way : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 16) 
  (h2 : distance_one_way = 7560) 
  (h3 : total_time = 960) 
  (h4 : (distance_one_way / (boat_speed + x)) + (distance_one_way / (boat_speed - x)) = total_time) 
  : x = 2 := 
  sorry

end speed_of_stream_l147_147068


namespace thirteen_pow_seven_mod_nine_l147_147343

theorem thirteen_pow_seven_mod_nine : (13^7 % 9 = 4) :=
by {
  sorry
}

end thirteen_pow_seven_mod_nine_l147_147343


namespace value_of_c_infinite_solutions_l147_147387

theorem value_of_c_infinite_solutions (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 18 * y + 15) ↔ (c = 3) :=
by
  sorry

end value_of_c_infinite_solutions_l147_147387


namespace carries_average_speed_is_approx_34_29_l147_147888

noncomputable def CarriesActualAverageSpeed : ℝ :=
  let jerry_speed := 40 -- in mph
  let jerry_time := 1/2 -- in hours, 30 minutes = 0.5 hours
  let jerry_distance := jerry_speed * jerry_time

  let beth_distance := jerry_distance + 5
  let beth_time := jerry_time + (20 / 60) -- converting 20 minutes to hours

  let carrie_distance := 2 * jerry_distance
  let carrie_time := 1 + (10 / 60) -- converting 10 minutes to hours

  carrie_distance / carrie_time

theorem carries_average_speed_is_approx_34_29 : 
  |CarriesActualAverageSpeed - 34.29| < 0.01 :=
sorry

end carries_average_speed_is_approx_34_29_l147_147888


namespace inequality_solution_set_l147_147925

noncomputable def solution_set := { x : ℝ | (x < -1 ∨ 1 < x) ∧ x ≠ 4 }

theorem inequality_solution_set : 
  { x : ℝ | (x^2 - 1) / (4 - x)^2 ≥ 0 } = solution_set :=
  by 
    sorry

end inequality_solution_set_l147_147925


namespace student_average_always_greater_l147_147110

theorem student_average_always_greater (x y z : ℝ) (h1 : x < z) (h2 : z < y) :
  (B = (x + z + 2 * y) / 4) > (A = (x + y + z) / 3) := by
  sorry

end student_average_always_greater_l147_147110


namespace smallest_b_l147_147836

theorem smallest_b {a b c d : ℕ} (r : ℕ) 
  (h1 : a = b - r) (h2 : c = b + r) (h3 : d = b + 2 * r) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h5 : a * b * c * d = 256) : b = 4 :=
by
  sorry

end smallest_b_l147_147836


namespace sequence_term_expression_l147_147112

theorem sequence_term_expression (a : ℕ → ℝ) (S : ℕ → ℝ) (C : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, S n + n * a n = C)
  (h3 : ∀ n ≥ 2, (n + 1) * a n = (n - 1) * a (n - 1)) :
  ∀ n, a n = 2 / (n * (n + 1)) :=
by
  sorry

end sequence_term_expression_l147_147112


namespace original_number_of_men_l147_147450

theorem original_number_of_men 
  (x : ℕ) 
  (H1 : x * 15 = (x - 8) * 18) : 
  x = 48 := 
sorry

end original_number_of_men_l147_147450


namespace geometric_series_sum_condition_l147_147190

def geometric_series_sum (a q n : ℕ) : ℕ := a * (1 - q^n) / (1 - q)

theorem geometric_series_sum_condition (S : ℕ → ℕ) (a : ℕ) (q : ℕ) (h1 : a = 1) 
  (h2 : ∀ n, S n = geometric_series_sum a q n)
  (h3 : S 7 - 4 * S 6 + 3 * S 5 = 0) : 
  S 4 = 40 := 
by 
  sorry

end geometric_series_sum_condition_l147_147190


namespace pool_balls_pyramid_arrangement_l147_147124

/-- In how many distinguishable ways can 10 distinct pool balls be arranged in a pyramid
    (6 on the bottom, 3 in the middle, 1 on the top), assuming that all rotations of the pyramid are indistinguishable? -/
def pyramid_pool_balls_distinguishable_arrangements : Nat :=
  let total_arrangements := Nat.factorial 10
  let indistinguishable_rotations := 9
  total_arrangements / indistinguishable_rotations

theorem pool_balls_pyramid_arrangement :
  pyramid_pool_balls_distinguishable_arrangements = 403200 :=
by
  -- Proof will be added here
  sorry

end pool_balls_pyramid_arrangement_l147_147124


namespace sourball_candies_division_l147_147140

theorem sourball_candies_division (N J L : ℕ) (total_candies : ℕ) (remaining_candies : ℕ) :
  N = 12 →
  J = N / 2 →
  L = J - 3 →
  total_candies = 30 →
  remaining_candies = total_candies - (N + J + L) →
  (remaining_candies / 3) = 3 :=
by 
  sorry

end sourball_candies_division_l147_147140


namespace basketball_cards_price_l147_147061

theorem basketball_cards_price :
  let toys_cost := 3 * 10
  let shirts_cost := 5 * 6
  let total_cost := 70
  let basketball_cards_cost := total_cost - (toys_cost + shirts_cost)
  let packs_of_cards := 2
  (basketball_cards_cost / packs_of_cards) = 5 :=
by
  sorry

end basketball_cards_price_l147_147061


namespace least_k_divisible_by_240_l147_147725

theorem least_k_divisible_by_240 : ∃ (k : ℕ), k^2 % 240 = 0 ∧ k = 60 :=
by
  sorry

end least_k_divisible_by_240_l147_147725


namespace beach_trip_time_l147_147929

noncomputable def totalTripTime (driveTime eachWay : ℝ) (beachTimeFactor : ℝ) : ℝ :=
  let totalDriveTime := eachWay * 2
  totalDriveTime + (totalDriveTime * beachTimeFactor)

theorem beach_trip_time :
  totalTripTime 2 2 2.5 = 14 := 
by
  sorry

end beach_trip_time_l147_147929


namespace advanced_purchase_tickets_sold_l147_147458

theorem advanced_purchase_tickets_sold (A D : ℕ) 
  (h1 : A + D = 140)
  (h2 : 8 * A + 14 * D = 1720) : 
  A = 40 :=
by
  sorry

end advanced_purchase_tickets_sold_l147_147458


namespace exists_distinct_numbers_satisfy_conditions_l147_147024

theorem exists_distinct_numbers_satisfy_conditions :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a + b + c = 6) ∧
  (2 * b = a + c) ∧
  ((b^2 = a * c) ∨ (a^2 = b * c) ∨ (c^2 = a * b)) :=
by
  sorry

end exists_distinct_numbers_satisfy_conditions_l147_147024


namespace sally_more_cards_than_dan_l147_147416

theorem sally_more_cards_than_dan :
  let sally_initial := 27
  let sally_bought := 20
  let dan_cards := 41
  sally_initial + sally_bought - dan_cards = 6 :=
by
  sorry

end sally_more_cards_than_dan_l147_147416


namespace find_a_l147_147697

variable (m : ℝ)

def root1 := 2 * m - 1
def root2 := m + 4

theorem find_a (h : root1 ^ 2 = root2 ^ 2) : ∃ a : ℝ, a = 9 :=
by
  sorry

end find_a_l147_147697


namespace one_positive_real_solution_l147_147878

noncomputable def f (x : ℝ) : ℝ := x^4 + 5 * x^3 + 10 * x^2 + 2023 * x - 2021

theorem one_positive_real_solution : 
  ∃! x : ℝ, 0 < x ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end one_positive_real_solution_l147_147878


namespace simplify_expression_l147_147256

open Real

theorem simplify_expression (a b : ℝ) 
  (h1 : b ≠ 0) (h2 : b ≠ -3 * a) (h3 : b ≠ a) (h4 : b ≠ -a) : 
  ((2 * b + a - (4 * a ^ 2 - b ^ 2) / a) / (b ^ 3 + 2 * a * b ^ 2 - 3 * a ^ 2 * b)) *
  ((a ^ 3 * b - 2 * a ^ 2 * b ^ 2 + a * b ^ 3) / (a ^ 2 - b ^ 2)) = 
  (a - b) / (a + b) :=
by
  sorry

end simplify_expression_l147_147256


namespace total_food_pounds_l147_147057

theorem total_food_pounds (chicken hamburger hot_dogs sides : ℕ) 
  (h1 : chicken = 16) 
  (h2 : hamburger = chicken / 2) 
  (h3 : hot_dogs = hamburger + 2) 
  (h4 : sides = hot_dogs / 2) : 
  chicken + hamburger + hot_dogs + sides = 39 := 
  by 
    sorry

end total_food_pounds_l147_147057


namespace smallest_product_of_two_distinct_primes_greater_than_50_l147_147412

theorem smallest_product_of_two_distinct_primes_greater_than_50 : 
  ∃ (p q : ℕ), p > 50 ∧ q > 50 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 3127 :=
by 
  sorry

end smallest_product_of_two_distinct_primes_greater_than_50_l147_147412


namespace keaton_annual_profit_l147_147744

theorem keaton_annual_profit :
  let orange_harvests_per_year := 12 / 2
  let apple_harvests_per_year := 12 / 3
  let peach_harvests_per_year := 12 / 4
  let blackberry_harvests_per_year := 12 / 6

  let orange_profit_per_harvest := 50 - 20
  let apple_profit_per_harvest := 30 - 15
  let peach_profit_per_harvest := 45 - 25
  let blackberry_profit_per_harvest := 70 - 30

  let total_orange_profit := orange_harvests_per_year * orange_profit_per_harvest
  let total_apple_profit := apple_harvests_per_year * apple_profit_per_harvest
  let total_peach_profit := peach_harvests_per_year * peach_profit_per_harvest
  let total_blackberry_profit := blackberry_harvests_per_year * blackberry_profit_per_harvest

  let total_annual_profit := total_orange_profit + total_apple_profit + total_peach_profit + total_blackberry_profit

  total_annual_profit = 380
:= by
  sorry

end keaton_annual_profit_l147_147744


namespace last_ball_probability_l147_147778

theorem last_ball_probability (w b : ℕ) (H : w > 0 ∨ b > 0) :
  (w % 2 = 1 → ∃ p : ℝ, p = 1 ∧ (∃ n, (∀ (k : ℕ), k < n → (sorry))) ) ∧ 
  (w % 2 = 0 → ∃ p : ℝ, p = 0 ∧ (∃ n, (∀ (k : ℕ), k < n → (sorry))) ) :=
by sorry

end last_ball_probability_l147_147778


namespace tan_alpha_value_l147_147688

theorem tan_alpha_value
  (α : ℝ)
  (h_cos : Real.cos α = -4/5)
  (h_range : (Real.pi / 2) < α ∧ α < Real.pi) :
  Real.tan α = -3/4 := by
  sorry

end tan_alpha_value_l147_147688


namespace avg_income_pr_l147_147433

theorem avg_income_pr (P Q R : ℝ) 
  (h_avgPQ : (P + Q) / 2 = 5050) 
  (h_avgQR : (Q + R) / 2 = 6250)
  (h_P : P = 4000) 
  : (P + R) / 2 = 5200 := 
by 
  sorry

end avg_income_pr_l147_147433


namespace part_I_part_II_part_III_l147_147515

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- Part (I)
theorem part_I (a : ℝ) (h_a : a = 1) : 
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 0 ∧ f x a * g x = 1 := sorry

-- Part (II)
theorem part_II (a : ℝ) (h_a : a = -1) (k : ℝ) :
  (∃ x : ℝ, f x a = k * g x ∧ ∀ y : ℝ, y ≠ x → f y a ≠ k * g y) ↔ 
  (k > 3 * Real.exp (-2) ∨ (0 < k ∧ k < 1 * Real.exp (-1))) := sorry

-- Part (III)
theorem part_III (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), (x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ x₁ ≠ x₂) →
  abs (f x₁ a - f x₂ a) < abs (g x₁ - g x₂)) ↔
  (-1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2) := sorry

end part_I_part_II_part_III_l147_147515


namespace animal_group_divisor_l147_147840

theorem animal_group_divisor (cows sheep goats total groups : ℕ)
    (hc : cows = 24) 
    (hs : sheep = 7) 
    (hg : goats = 113) 
    (ht : total = cows + sheep + goats) 
    (htotal : total = 144) 
    (hdiv : groups ∣ total) 
    (hexclude1 : groups ≠ 1) 
    (hexclude144 : groups ≠ 144) : 
    ∃ g, g = groups ∧ g ∈ [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72] :=
  by 
  sorry

end animal_group_divisor_l147_147840


namespace possible_values_of_a_l147_147299

def A (a : ℤ) : Set ℤ := {2, 4, a^3 - 2 * a^2 - a + 7}
def B (a : ℤ) : Set ℤ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem possible_values_of_a (a : ℤ) :
  A a ∩ B a = {2, 5} ↔ a = -1 ∨ a = 2 :=
by
  sorry

end possible_values_of_a_l147_147299


namespace mike_total_spending_l147_147837

def mike_spent_on_speakers : ℝ := 235.87
def mike_spent_on_tires : ℝ := 281.45
def mike_spent_on_steering_wheel_cover : ℝ := 179.99
def mike_spent_on_seat_covers : ℝ := 122.31
def mike_spent_on_headlights : ℝ := 98.63

theorem mike_total_spending :
  mike_spent_on_speakers + mike_spent_on_tires + mike_spent_on_steering_wheel_cover + mike_spent_on_seat_covers + mike_spent_on_headlights = 918.25 :=
  sorry

end mike_total_spending_l147_147837


namespace find_a3_l147_147641

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_a3
  (a : ℕ → α) (q : α)
  (h_geom : geometric_sequence a q)
  (h_a1 : a 1 = 2)
  (h_cond : a 3 * a 5 = 4 * (a 6) ^ 2) :
  a 3 = 1 :=
by
  sorry

end find_a3_l147_147641


namespace total_time_taken_l147_147826

theorem total_time_taken
  (speed_boat : ℝ)
  (speed_stream : ℝ)
  (distance : ℝ)
  (h_boat : speed_boat = 12)
  (h_stream : speed_stream = 5)
  (h_distance : distance = 325) :
  (distance / (speed_boat - speed_stream) + distance / (speed_boat + speed_stream)) = 65.55 :=
by
  sorry

end total_time_taken_l147_147826


namespace geometric_seq_20th_term_l147_147455

theorem geometric_seq_20th_term (a r : ℕ)
  (h1 : a * r ^ 4 = 5)
  (h2 : a * r ^ 11 = 1280) :
  a * r ^ 19 = 2621440 :=
sorry

end geometric_seq_20th_term_l147_147455


namespace calculate_parallel_segment_length_l147_147668

theorem calculate_parallel_segment_length :
  ∀ (d : ℝ), 
    ∃ (X Y Z P : Type) 
    (XY YZ XZ : ℝ), 
    XY = 490 ∧ 
    YZ = 520 ∧ 
    XZ = 560 ∧ 
    ∃ (D D' E E' F F' : Type),
      (D ≠ E ∧ E ≠ F ∧ F ≠ D') ∧  
      (XZ - (d * (520/490) + d * (520/560))) = d → d = 268.148148 :=
by
  sorry

end calculate_parallel_segment_length_l147_147668


namespace quadratic_real_roots_l147_147519

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l147_147519


namespace find_land_area_l147_147367

variable (L : ℝ) -- cost of land per square meter
variable (B : ℝ) -- cost of bricks per 1000 bricks
variable (R : ℝ) -- cost of roof tiles per tile
variable (numBricks : ℝ) -- number of bricks needed
variable (numTiles : ℝ) -- number of roof tiles needed
variable (totalCost : ℝ) -- total construction cost

theorem find_land_area (h1 : L = 50) 
                       (h2 : B = 100)
                       (h3 : R = 10) 
                       (h4 : numBricks = 10000) 
                       (h5 : numTiles = 500) 
                       (h6 : totalCost = 106000) : 
                       ∃ x : ℝ, 50 * x + (numBricks / 1000) * B + numTiles * R = totalCost ∧ x = 2000 := 
by 
  use 2000
  simp [h1, h2, h3, h4, h5, h6]
  norm_num
  done

end find_land_area_l147_147367


namespace tip_percentage_calculation_l147_147755

theorem tip_percentage_calculation :
  let a := 8
  let r := 20
  let w := 3
  let n_w := 2
  let d := 6
  let t := 38
  let discount := 0.5
  let full_cost_without_tip := a + r + (w * n_w) + d
  let discounted_meal_cost := a + (r - (r * discount)) + (w * n_w) + d
  let tip_amount := t - discounted_meal_cost
  let tip_percentage := (tip_amount / full_cost_without_tip) * 100
  tip_percentage = 20 :=
by
  sorry

end tip_percentage_calculation_l147_147755


namespace jerry_age_l147_147133

variable (M J : ℕ) -- Declare Mickey's and Jerry's ages as natural numbers

-- Define the conditions as hypotheses
def condition1 := M = 2 * J - 6
def condition2 := M = 18

-- Theorem statement where we need to prove J = 12 given the conditions
theorem jerry_age
  (h1 : condition1 M J)
  (h2 : condition2 M) :
  J = 12 :=
sorry

end jerry_age_l147_147133


namespace range_of_a_l147_147278

noncomputable def quadratic_inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  x^2 - 2 * (a - 2) * x + a > 0

theorem range_of_a :
  (∀ x : ℝ, (x < 1 ∨ x > 5) → quadratic_inequality_condition a x) ↔ (1 < a ∧ a ≤ 5) :=
by
  sorry

end range_of_a_l147_147278


namespace max_value_of_f_on_S_l147_147330

noncomputable def S : Set ℝ := { x | x^4 - 13 * x^2 + 36 ≤ 0 }
noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_value_of_f_on_S : ∃ x ∈ S, ∀ y ∈ S, f y ≤ f x ∧ f x = 18 :=
by
  sorry

end max_value_of_f_on_S_l147_147330


namespace part_a_l147_147199

theorem part_a (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x + y = 2) : 
  (1 / x + 1 / y) ≤ (1 / x^2 + 1 / y^2) := 
sorry

end part_a_l147_147199


namespace last_digit_of_4_over_3_power_5_l147_147028

noncomputable def last_digit_of_fraction (n d : ℕ) : ℕ :=
  (n * 10^5 / d) % 10

def four : ℕ := 4
def three_power_five : ℕ := 3^5

theorem last_digit_of_4_over_3_power_5 :
  last_digit_of_fraction four three_power_five = 7 :=
by
  sorry

end last_digit_of_4_over_3_power_5_l147_147028


namespace bailey_discount_l147_147615

noncomputable def discount_percentage (total_cost_without_discount amount_spent : ℝ) : ℝ :=
  ((total_cost_without_discount - amount_spent) / total_cost_without_discount) * 100

theorem bailey_discount :
  let guest_sets := 2
  let master_sets := 4
  let price_guest := 40
  let price_master := 50
  let amount_spent := 224
  let total_cost_without_discount := (guest_sets * price_guest) + (master_sets * price_master)
  discount_percentage total_cost_without_discount amount_spent = 20 := 
by
  sorry

end bailey_discount_l147_147615


namespace area_of_feasible_region_l147_147017

theorem area_of_feasible_region :
  (∃ k m : ℝ, (∀ x y : ℝ,
    (kx - y + 1 ≥ 0 ∧ kx - my ≤ 0 ∧ y ≥ 0) ↔
    (x - y + 1 ≥ 0 ∧ x + y ≤ 0 ∧ y ≥ 0)) ∧
    k = 1 ∧ m = -1) →
  ∃ a : ℝ, a = 1 / 4 :=
by sorry

end area_of_feasible_region_l147_147017


namespace balls_remaining_l147_147533

-- Define the initial number of balls in the box
def initial_balls := 10

-- Define the number of balls taken by Yoongi
def balls_taken := 3

-- Define the number of balls left after Yoongi took some balls
def balls_left := initial_balls - balls_taken

-- The theorem statement to be proven
theorem balls_remaining : balls_left = 7 :=
by
    -- Skipping the proof
    sorry

end balls_remaining_l147_147533


namespace find_a_of_pure_imaginary_z_l147_147995

-- Definition of a pure imaginary number
def pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Main theorem statement
theorem find_a_of_pure_imaginary_z (a : ℝ) (z : ℂ) (hz : pure_imaginary z) (h : (2 - I) * z = 4 + 2 * a * I) : a = 4 :=
by
  sorry

end find_a_of_pure_imaginary_z_l147_147995


namespace f_2021_l147_147223

noncomputable def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom period_f : ∀ x : ℝ, f (x) = f (2 - x)
axiom f_neg1 : f (-1) = 1

theorem f_2021 : f (2021) = -1 :=
by
  sorry

end f_2021_l147_147223


namespace smallest_of_consecutive_even_numbers_l147_147165

theorem smallest_of_consecutive_even_numbers (n : ℤ) (h : ∃ a b c : ℤ, a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ b = a + 2 ∧ c = a + 4 ∧ c = 2 * n + 1) :
  ∃ a b c : ℤ, a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ b = a + 2 ∧ c = a + 4 ∧ a = 2 * n - 3 :=
by
  sorry

end smallest_of_consecutive_even_numbers_l147_147165


namespace odometer_problem_l147_147069

theorem odometer_problem
  (a b c : ℕ) -- a, b, c are natural numbers
  (h1 : 1 ≤ a) -- condition (a ≥ 1)
  (h2 : a + b + c ≤ 7) -- condition (a + b + c ≤ 7)
  (h3 : 99 * (c - a) % 55 = 0) -- 99(c - a) must be divisible by 55
  (h4 : 100 * a + 10 * b + c < 1000) -- ensuring a, b, c keeps numbers within 3-digits
  (h5 : 100 * c + 10 * b + a < 1000) -- ensuring a, b, c keeps numbers within 3-digits
  : a^2 + b^2 + c^2 = 37 := sorry

end odometer_problem_l147_147069


namespace lines_intersect_l147_147152

theorem lines_intersect :
  ∃ x y : ℚ, 
  8 * x - 5 * y = 40 ∧ 
  6 * x - y = -5 ∧ 
  x = 15 / 38 ∧ 
  y = 140 / 19 :=
by { sorry }

end lines_intersect_l147_147152


namespace Mayor_decision_to_adopt_model_A_l147_147377

-- Define the conditions
def num_people := 17

def radicals_support_model_A := (0 : ℕ)

def socialists_support_model_B (y : ℕ) := y

def republicans_support_model_B (x y : ℕ) := x - y

def independents_support_model_B (x y : ℕ) := (y + (x - y)) / 2

-- The number of individuals supporting model A and model B
def support_model_B (x y : ℕ) := radicals_support_model_A + socialists_support_model_B y + republicans_support_model_B x y + independents_support_model_B x y

def support_model_A (x : ℕ) := 4 * x - support_model_B x x / 2

-- Statement to prove
theorem Mayor_decision_to_adopt_model_A (x : ℕ) (h : x = num_people) : 
  support_model_A x > support_model_B x x := 
by {
  -- Proof goes here
  sorry
}

end Mayor_decision_to_adopt_model_A_l147_147377


namespace temperature_difference_l147_147209

theorem temperature_difference 
  (lowest: ℤ) (highest: ℤ) 
  (h_lowest : lowest = -4)
  (h_highest : highest = 5) :
  highest - lowest = 9 := 
by
  --relies on the correctness of problem and given simplyifying
  sorry

end temperature_difference_l147_147209


namespace evaluate_expression_l147_147084

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 7)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 6)

theorem evaluate_expression : g_inv (g_inv 6 + g_inv 7) = 4 := by
  sorry

end evaluate_expression_l147_147084


namespace equation_solution_l147_147126

theorem equation_solution (x : ℤ) (h : x + 1 = 2) : x = 1 :=
sorry

end equation_solution_l147_147126


namespace bottles_more_than_apples_l147_147678

def bottles_regular : ℕ := 72
def bottles_diet : ℕ := 32
def apples : ℕ := 78

def total_bottles : ℕ := bottles_regular + bottles_diet

theorem bottles_more_than_apples : (total_bottles - apples) = 26 := by
  sorry

end bottles_more_than_apples_l147_147678


namespace cube_sum_identity_l147_147472

theorem cube_sum_identity (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end cube_sum_identity_l147_147472


namespace complex_division_evaluation_l147_147882

open Complex

theorem complex_division_evaluation :
  (2 : ℂ) / (I * (3 - I)) = (1 / 5 : ℂ) - (3 / 5) * I :=
by
  sorry

end complex_division_evaluation_l147_147882


namespace range_of_a_l147_147163

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f (2^x) = x^2 - 2 * a * x + a^2 - 1) →
  (∀ x, 2^(a-1) ≤ x ∧ x ≤ 2^(a^2 - 2*a + 2) → -1 ≤ f x ∧ f x ≤ 0) →
  ((3 - Real.sqrt 5) / 2 ≤ a ∧ a ≤ 1) ∨ (2 ≤ a ∧ a ≤ (3 + Real.sqrt 5) / 2) :=
by
  sorry

end range_of_a_l147_147163


namespace inequality_nonnegative_reals_l147_147210

theorem inequality_nonnegative_reals (a b c : ℝ) (h_a : 0 ≤ a) (h_b : 0 ≤ b) (h_c : 0 ≤ c) :
  |(c * a - a * b)| + |(a * b - b * c)| + |(b * c - c * a)| ≤ |(b^2 - c^2)| + |(c^2 - a^2)| + |(a^2 - b^2)| :=
by
  sorry

end inequality_nonnegative_reals_l147_147210


namespace math_problem_l147_147094

theorem math_problem
  (a b c d : ℚ)
  (h₁ : a = 1 / 3)
  (h₂ : b = 1 / 6)
  (h₃ : c = 1 / 9)
  (h₄ : d = 1 / 18) :
  9 * (a + b + c + d)⁻¹ = 27 / 2 := 
sorry

end math_problem_l147_147094


namespace problem_statement_l147_147351

def P (m n : ℕ) : ℕ :=
  let coeff_x := Nat.choose 4 m
  let coeff_y := Nat.choose 6 n
  coeff_x * coeff_y

theorem problem_statement : P 2 1 + P 1 2 = 96 :=
by
  sorry

end problem_statement_l147_147351


namespace math_problem_l147_147972

theorem math_problem (x : ℂ) (hx : x + 1/x = 3) : x^6 + 1/x^6 = 322 := 
by 
  sorry

end math_problem_l147_147972


namespace number_of_marbles_pat_keeps_l147_147407

theorem number_of_marbles_pat_keeps 
  (x : ℕ) 
  (h1 : x / 6 = 9) 
  : x / 3 = 18 :=
by
  sorry

end number_of_marbles_pat_keeps_l147_147407


namespace data_a_value_l147_147030

theorem data_a_value (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a + b + c = 96) : a = 12 :=
by
  sorry

end data_a_value_l147_147030


namespace triangle_side_square_sum_eq_three_times_centroid_dist_square_sum_l147_147336

theorem triangle_side_square_sum_eq_three_times_centroid_dist_square_sum
  {A B C O : EuclideanSpace ℝ (Fin 2)}
  (h_centroid : O = (1/3 : ℝ) • (A + B + C)) :
  (dist A B)^2 + (dist B C)^2 + (dist C A)^2 =
  3 * ((dist O A)^2 + (dist O B)^2 + (dist O C)^2) :=
sorry

end triangle_side_square_sum_eq_three_times_centroid_dist_square_sum_l147_147336


namespace find_number_l147_147097

theorem find_number (x q : ℕ) (h1 : x = 7 * q) (h2 : q + x + 7 = 175) : x = 147 := 
by
  sorry

end find_number_l147_147097


namespace sum_of_angles_in_triangle_sum_of_angles_in_polygon_exponential_equation_logarithmic_equation_l147_147337

-- 1. Sum of the interior angles in a triangle is 180 degrees.
theorem sum_of_angles_in_triangle : ∀ a : ℕ, (∀ x y z : ℕ, x + y + z = 180) → a = 180 := by
  intros a h
  have : a = 180 := sorry
  exact this

-- 2. Sum of interior angles of a regular b-sided polygon is 1080 degrees.
theorem sum_of_angles_in_polygon : ∀ b : ℕ, ((b - 2) * 180 = 1080) → b = 8 := by
  intros b h
  have : b = 8 := sorry
  exact this

-- 3. Exponential equation involving b.
theorem exponential_equation : ∀ p b : ℕ, (8 ^ b = p ^ 21) ∧ (b = 8) → p = 2 := by
  intros p b h
  have : p = 2 := sorry
  exact this

-- 4. Logarithmic equation involving p.
theorem logarithmic_equation : ∀ q p : ℕ, (p = Real.log 81 / Real.log q) ∧ (p = 2) → q = 9 := by
  intros q p h
  have : q = 9 := sorry
  exact this

end sum_of_angles_in_triangle_sum_of_angles_in_polygon_exponential_equation_logarithmic_equation_l147_147337


namespace grassy_width_excluding_path_l147_147631

theorem grassy_width_excluding_path
  (l : ℝ) (w : ℝ) (p : ℝ)
  (h1: l = 110) (h2: w = 65) (h3: p = 2.5) :
  w - 2 * p = 60 :=
by
  sorry

end grassy_width_excluding_path_l147_147631


namespace volume_of_new_pyramid_l147_147022

theorem volume_of_new_pyramid (l w h : ℝ) (h_vol : (1 / 3) * l * w * h = 80) :
  (1 / 3) * (3 * l) * w * (1.8 * h) = 432 :=
by
  sorry

end volume_of_new_pyramid_l147_147022


namespace pine_saplings_in_sample_l147_147528

-- Definitions based on conditions
def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

-- Main theorem to prove
theorem pine_saplings_in_sample : (pine_saplings * sample_size) / total_saplings = 20 :=
by sorry

end pine_saplings_in_sample_l147_147528


namespace evaluate_expression_l147_147764

theorem evaluate_expression : 6 / (-1 / 2 + 1 / 3) = -36 := 
by
  sorry

end evaluate_expression_l147_147764


namespace three_integers_product_sum_l147_147486

theorem three_integers_product_sum (a b c : ℤ) (h : a * b * c = -5) :
    a + b + c = 5 ∨ a + b + c = -3 ∨ a + b + c = -7 :=
sorry

end three_integers_product_sum_l147_147486


namespace total_insects_l147_147967

theorem total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (ants_per_leaf : ℕ) (caterpillars_every_third_leaf : ℕ) :
  leaves = 84 →
  ladybugs_per_leaf = 139 →
  ants_per_leaf = 97 →
  caterpillars_every_third_leaf = 53 →
  (84 * 139) + (84 * 97) + (53 * (84 / 3)) = 21308 := 
by
  sorry

end total_insects_l147_147967


namespace johnson_family_seating_l147_147352

-- Defining the total number of children:
def total_children := 8

-- Defining the number of sons and daughters:
def sons := 5
def daughters := 3

-- Factoring in the total number of unrestricted seating arrangements:
def total_seating_arrangements : ℕ := Nat.factorial total_children

-- Factoring in the number of non-adjacent seating arrangements for sons:
def non_adjacent_arrangements : ℕ :=
  (Nat.factorial daughters) * (Nat.factorial sons)

-- The lean proof statement to prove:
theorem johnson_family_seating :
  total_seating_arrangements - non_adjacent_arrangements = 39600 :=
by
  sorry

end johnson_family_seating_l147_147352


namespace bags_total_on_next_day_l147_147452

def bags_on_monday : ℕ := 7
def additional_bags : ℕ := 5
def bags_on_next_day : ℕ := bags_on_monday + additional_bags

theorem bags_total_on_next_day : bags_on_next_day = 12 := by
  unfold bags_on_next_day
  unfold bags_on_monday
  unfold additional_bags
  sorry

end bags_total_on_next_day_l147_147452


namespace problem_statement_l147_147893

open Classical

variable (p q : Prop)

theorem problem_statement (h1 : p ∨ q) (h2 : ¬(p ∧ q)) (h3 : ¬ p) : (p = (5 + 2 = 6) ∧ q = (6 > 2)) :=
by
  have hp : p = False := by sorry
  have hq : q = True := by sorry
  exact ⟨by sorry, by sorry⟩

end problem_statement_l147_147893


namespace triangle_right_hypotenuse_l147_147989

theorem triangle_right_hypotenuse (c : ℝ) (a : ℝ) (h₀ : c = 4) (h₁ : 0 < a) (h₂ : a^2 + b^2 = c^2) :
  a ≤ 2 * Real.sqrt 2 :=
sorry

end triangle_right_hypotenuse_l147_147989


namespace gcd_456_357_l147_147923

theorem gcd_456_357 : Nat.gcd 456 357 = 3 := by
  sorry

end gcd_456_357_l147_147923


namespace multiply_and_divide_equiv_l147_147835

/-- Defines the operation of first multiplying by 4/5 and then dividing by 4/7 -/
def multiply_and_divide (x : ℚ) : ℚ :=
  (x * (4 / 5)) / (4 / 7)

/-- Statement to prove the operation is equivalent to multiplying by 7/5 -/
theorem multiply_and_divide_equiv (x : ℚ) : 
  multiply_and_divide x = x * (7 / 5) :=
by 
  -- This requires a proof, which we can assume here
  sorry

end multiply_and_divide_equiv_l147_147835


namespace charity_event_fund_raising_l147_147280

theorem charity_event_fund_raising :
  let n := 9
  let I := 2000
  let p := 0.10
  let increased_total := I * (1 + p)
  let amount_per_person := increased_total / n
  amount_per_person = 244.44 := by
  sorry

end charity_event_fund_raising_l147_147280


namespace abs_eq_neg_iff_nonpositive_l147_147235

theorem abs_eq_neg_iff_nonpositive (x : ℝ) : |x| = -x ↔ x ≤ 0 := by
  sorry

end abs_eq_neg_iff_nonpositive_l147_147235


namespace sector_area_l147_147585

-- Define the properties and conditions
def perimeter_of_sector (r l : ℝ) : Prop :=
  l + 2 * r = 8

def central_angle_arc_length (r : ℝ) : ℝ :=
  2 * r

-- Theorem to prove the area of the sector
theorem sector_area (r : ℝ) (l : ℝ) 
  (h_perimeter : perimeter_of_sector r l) 
  (h_arc_length : l = central_angle_arc_length r) : 
  1 / 2 * l * r = 4 := 
by
  -- This is the place where the proof would go; we use sorry to indicate it's incomplete
  sorry

end sector_area_l147_147585


namespace probability_at_least_one_two_l147_147018

def num_dice := 2
def sides_dice := 8
def total_outcomes := sides_dice ^ num_dice
def num_non_favorable_outcomes := (sides_dice - 1) ^ num_dice
def num_favorable_outcomes := total_outcomes - num_non_favorable_outcomes
def probability_favorable_outcomes := (15 : ℚ) / (64 : ℚ)

theorem probability_at_least_one_two :
  probability_favorable_outcomes = 15 / 64 :=
sorry

end probability_at_least_one_two_l147_147018


namespace total_yield_l147_147013

theorem total_yield (x y z : ℝ)
  (h1 : 0.4 * z + 0.2 * x = 1)
  (h2 : 0.1 * y - 0.1 * z = -0.5)
  (h3 : 0.1 * x + 0.2 * y = 4) :
  x + y + z = 15 :=
sorry

end total_yield_l147_147013


namespace jimmy_sells_less_l147_147834

-- Definitions based on conditions
def num_figures : ℕ := 5
def value_figure_1_to_4 : ℕ := 15
def value_figure_5 : ℕ := 20
def total_earned : ℕ := 55

-- Formulation of the problem statement in Lean
theorem jimmy_sells_less (total_value : ℕ := (4 * value_figure_1_to_4) + value_figure_5) (difference : ℕ := total_value - total_earned) (amount_less_per_figure : ℕ := difference / num_figures) : amount_less_per_figure = 5 := by
  sorry

end jimmy_sells_less_l147_147834


namespace initial_trees_l147_147436

theorem initial_trees (DeadTrees CutTrees LeftTrees : ℕ) (h1 : DeadTrees = 15) (h2 : CutTrees = 23) (h3 : LeftTrees = 48) :
  DeadTrees + CutTrees + LeftTrees = 86 :=
by
  sorry

end initial_trees_l147_147436


namespace students_more_than_Yoongi_l147_147936

theorem students_more_than_Yoongi (total_players : ℕ) (less_than_Yoongi : ℕ) (total_players_eq : total_players = 21) (less_than_eq : less_than_Yoongi = 11) : 
  ∃ more_than_Yoongi : ℕ, more_than_Yoongi = (total_players - 1 - less_than_Yoongi) ∧ more_than_Yoongi = 8 :=
by
  sorry

end students_more_than_Yoongi_l147_147936


namespace avg_of_first_21_multiples_l147_147611

theorem avg_of_first_21_multiples (n : ℕ) (h : (21 * 11 * n / 21) = 88) : n = 8 :=
by
  sorry

end avg_of_first_21_multiples_l147_147611


namespace karting_routes_10_min_l147_147742

-- Define the recursive function for M_{n, A}
def num_routes : ℕ → ℕ
| 0 => 1   -- Starting point at A for 0 minutes (0 routes)
| 1 => 0   -- Impossible to end at A in just 1 move
| 2 => 1   -- Only one way to go A -> B -> A in 2 minutes
| n + 1 =>
  if n = 1 then 0 -- Additional base case for n=2 as defined
  else if n = 2 then 1
  else num_routes (n - 1) + num_routes (n - 2)

theorem karting_routes_10_min : num_routes 10 = 34 := by
  -- Proof steps go here
  sorry

end karting_routes_10_min_l147_147742


namespace sum_two_and_four_l147_147934

theorem sum_two_and_four : 2 + 4 = 6 := by
  sorry

end sum_two_and_four_l147_147934


namespace total_balloons_l147_147571

-- Define the number of yellow balloons each person has
def tom_balloons : Nat := 18
def sara_balloons : Nat := 12
def alex_balloons : Nat := 7

-- Prove that the total number of balloons is 37
theorem total_balloons : tom_balloons + sara_balloons + alex_balloons = 37 := 
by 
  sorry

end total_balloons_l147_147571


namespace evaluate_expression_l147_147807

theorem evaluate_expression (a : ℕ) (h : a = 3) : a^2 * a^5 = 2187 :=
by sorry

end evaluate_expression_l147_147807


namespace train_lengths_l147_147909

theorem train_lengths (L_A L_P L_B : ℕ) (speed_A_km_hr speed_B_km_hr : ℕ) (time_A_seconds : ℕ)
                      (h1 : L_P = L_A)
                      (h2 : speed_A_km_hr = 72)
                      (h3 : speed_B_km_hr = 80)
                      (h4 : time_A_seconds = 60)
                      (h5 : L_B = L_P / 2)
                      (h6 : L_A + L_P = (speed_A_km_hr * 1000 / 3600) * time_A_seconds) :
  L_A = 600 ∧ L_B = 300 :=
by
  sorry

end train_lengths_l147_147909


namespace var_of_or_l147_147055

theorem var_of_or (p q : Prop) (h : ¬ (p ∧ q)) : (p ∨ q = true) ∨ (p ∨ q = false) :=
by
  sorry

end var_of_or_l147_147055


namespace parameter_a_range_l147_147392

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * a * x + 2 * a + 1

theorem parameter_a_range :
  (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → quadratic_function a x ≥ 1) ↔ (0 ≤ a) :=
by
  sorry

end parameter_a_range_l147_147392


namespace fraction_students_received_Bs_l147_147717

theorem fraction_students_received_Bs (fraction_As : ℝ) (fraction_As_or_Bs : ℝ) (h1 : fraction_As = 0.7) (h2 : fraction_As_or_Bs = 0.9) :
  fraction_As_or_Bs - fraction_As = 0.2 :=
by
  sorry

end fraction_students_received_Bs_l147_147717


namespace weekly_allowance_l147_147839

theorem weekly_allowance
  (video_game_cost : ℝ)
  (sales_tax_percentage : ℝ)
  (weeks_to_save : ℕ)
  (total_with_tax : ℝ)
  (total_savings : ℝ) :
  video_game_cost = 50 →
  sales_tax_percentage = 0.10 →
  weeks_to_save = 11 →
  total_with_tax = video_game_cost * (1 + sales_tax_percentage) →
  total_savings = weeks_to_save * (0.5 * total_savings) →
  total_savings = total_with_tax →
  total_savings = 55 :=
by
  intros
  sorry

end weekly_allowance_l147_147839


namespace max_ratio_two_digit_mean_50_l147_147542

theorem max_ratio_two_digit_mean_50 : 
  ∀ (x y : ℕ), (10 ≤ x ∧ x ≤ 99) ∧ (10 ≤ y ∧ y ≤ 99) ∧ (x + y = 100) → ( x / y ) ≤ 99 := 
by
  intros x y h
  obtain ⟨hx, hy, hsum⟩ := h
  sorry

end max_ratio_two_digit_mean_50_l147_147542


namespace altered_solution_ratio_l147_147315

variable (b d w : ℕ)
variable (b' d' w' : ℕ)
variable (ratio_orig_bd_ratio_orig_dw_ratio_orig_bw : Rat)
variable (ratio_new_bd_ratio_new_dw_ratio_new_bw : Rat)

noncomputable def orig_ratios (ratio_orig_bd ratio_orig_bw : Rat) (d w : ℕ) : Prop := 
    ratio_orig_bd = 2 / 40 ∧ ratio_orig_bw = 40 / 100

noncomputable def new_ratios (ratio_new_bd : Rat) (d' : ℕ) : Prop :=
    ratio_new_bd = 6 / 40 ∧ d' = 60

noncomputable def new_solution (w' : ℕ) : Prop :=
    w' = 300

theorem altered_solution_ratio : 
    ∀ (orig_ratios: Prop) (new_ratios: Prop) (new_solution: Prop),
    orig_ratios ∧ new_ratios ∧ new_solution →
    (d' / w = 2 / 5) :=
by
    sorry

end altered_solution_ratio_l147_147315


namespace probability_satisfies_inequality_l147_147512

/-- Define the conditions for the points (x, y) -/
def within_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5

def satisfies_inequality (x y : ℝ) : Prop :=
  x + 2 * y ≤ 6

/-- Compute the probability that a randomly selected point within the rectangle
also satisfies the inequality -/
theorem probability_satisfies_inequality : (∃ p : ℚ, p = 3 / 10) :=
sorry

end probability_satisfies_inequality_l147_147512


namespace original_price_l147_147465

theorem original_price (P : ℝ) (h : 0.684 * P = 6800) : P = 10000 :=
by
  sorry

end original_price_l147_147465


namespace robot_path_length_l147_147632

/--
A robot moves in the plane in a straight line, but every one meter it turns 90° to the right or to the left. At some point it reaches its starting point without having visited any other point more than once, and stops immediately. Prove that the possible path lengths of the robot are 4k for some integer k with k >= 3.
-/
theorem robot_path_length (n : ℕ) (h : n > 0) (Movement : n % 4 = 0) :
  ∃ k : ℕ, n = 4 * k ∧ k ≥ 3 :=
sorry

end robot_path_length_l147_147632


namespace acute_angle_slope_neg_product_l147_147131

   theorem acute_angle_slope_neg_product (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) (acute_inclination : ∃ (k : ℝ), k > 0 ∧ y = -a/b): (a * b < 0) :=
   by
     sorry
   
end acute_angle_slope_neg_product_l147_147131


namespace fill_in_the_blank_correct_option_l147_147961

-- Assume each option is defined
def options := ["the other", "some", "another", "other"]

-- Define a helper function to validate the correct option
def is_correct_option (opt: String) : Prop :=
  opt = "another"

-- The main problem statement
theorem fill_in_the_blank_correct_option :
  (∀ opt, opt ∈ options → is_correct_option opt → opt = "another") :=
by
  intro opt h_option h_correct
  simp [is_correct_option] at h_correct
  exact h_correct

-- Test case to check the correct option
example : is_correct_option "another" :=
by
  simp [is_correct_option]

end fill_in_the_blank_correct_option_l147_147961


namespace find_x_for_mean_l147_147494

theorem find_x_for_mean 
(x : ℝ) 
(h_mean : (3 + 11 + 7 + 9 + 15 + 13 + 8 + 19 + 17 + 21 + 14 + x) / 12 = 12) : 
x = 7 :=
sorry

end find_x_for_mean_l147_147494


namespace value_of_expression_l147_147091

theorem value_of_expression (x : ℤ) (h : x ^ 2 = 2209) : (x + 2) * (x - 2) = 2205 := 
by
  -- the proof goes here
  sorry

end value_of_expression_l147_147091


namespace eq_of_frac_eq_and_neq_neg_one_l147_147511

theorem eq_of_frac_eq_and_neq_neg_one
  (a b c d : ℝ)
  (h : (a + b) / (c + d) = (b + c) / (a + d))
  (h_neq : (a + b) / (c + d) ≠ -1) :
  a = c :=
sorry

end eq_of_frac_eq_and_neq_neg_one_l147_147511


namespace off_road_vehicle_cost_l147_147521

theorem off_road_vehicle_cost
  (dirt_bike_count : ℕ) (dirt_bike_cost : ℕ)
  (off_road_vehicle_count : ℕ) (register_cost : ℕ)
  (total_cost : ℕ) (off_road_vehicle_cost : ℕ) :
  dirt_bike_count = 3 → dirt_bike_cost = 150 →
  off_road_vehicle_count = 4 → register_cost = 25 →
  total_cost = 1825 →
  3 * dirt_bike_cost + 4 * off_road_vehicle_cost + 7 * register_cost = total_cost →
  off_road_vehicle_cost = 300 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end off_road_vehicle_cost_l147_147521


namespace missing_fraction_is_73_div_60_l147_147435

-- Definition of the given fractions
def fraction1 : ℚ := 1/3
def fraction2 : ℚ := 1/2
def fraction3 : ℚ := -5/6
def fraction4 : ℚ := 1/5
def fraction5 : ℚ := 1/4
def fraction6 : ℚ := -5/6

-- Total sum provided in the problem
def total_sum : ℚ := 50/60  -- 0.8333333333333334 in decimal form

-- The summation of given fractions
def sum_of_fractions : ℚ := fraction1 + fraction2 + fraction3 + fraction4 + fraction5 + fraction6

-- The statement to prove that the missing fraction is 73/60
theorem missing_fraction_is_73_div_60 : (total_sum - sum_of_fractions) = 73/60 := by
  sorry

end missing_fraction_is_73_div_60_l147_147435


namespace arithmetic_sequence_common_difference_l147_147651

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) 
    (h1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 81)
    (h2 : a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 171) : 
    ∃ d, d = 10 := 
by 
  sorry

end arithmetic_sequence_common_difference_l147_147651


namespace focus_of_parabola_l147_147041

theorem focus_of_parabola (m : ℝ) (m_nonzero : m ≠ 0) :
    ∃ (focus_x focus_y : ℝ), (focus_x, focus_y) = (m, 0) ∧
        ∀ (y : ℝ), (x = 1/(4*m) * y^2) := 
sorry

end focus_of_parabola_l147_147041


namespace highlights_part_to_whole_relation_l147_147462

/-- A predicate representing different types of statistical graphs. -/
inductive StatGraphType where
  | BarGraph : StatGraphType
  | PieChart : StatGraphType
  | LineGraph : StatGraphType
  | FrequencyDistributionHistogram : StatGraphType

/-- A lemma specifying that the PieChart is the graph type that highlights the relationship between a part and the whole. -/
theorem highlights_part_to_whole_relation (t : StatGraphType) : t = StatGraphType.PieChart :=
  sorry

end highlights_part_to_whole_relation_l147_147462


namespace sharks_at_newport_l147_147132

theorem sharks_at_newport :
  ∃ (x : ℕ), (∃ (y : ℕ), y = 4 * x ∧ x + y = 110) ∧ x = 22 :=
by {
  sorry
}

end sharks_at_newport_l147_147132


namespace certain_amount_l147_147341

theorem certain_amount (x : ℝ) (h1 : 2 * x = 86 - 54) (h2 : 8 + 3 * 8 = 24) (h3 : 86 - 54 + 32 = 86) : x = 43 := 
by {
  sorry
}

end certain_amount_l147_147341


namespace cricket_team_members_count_l147_147300

theorem cricket_team_members_count 
(captain_age : ℕ) (wk_keeper_age : ℕ) (whole_team_avg_age : ℕ)
(remaining_players_avg_age : ℕ) (n : ℕ) 
(h1 : captain_age = 28)
(h2 : wk_keeper_age = captain_age + 3)
(h3 : whole_team_avg_age = 25)
(h4 : remaining_players_avg_age = 24)
(h5 : (n * whole_team_avg_age - (captain_age + wk_keeper_age)) / (n - 2) = remaining_players_avg_age) :
n = 11 := 
sorry

end cricket_team_members_count_l147_147300


namespace hcf_462_5_1_l147_147987

theorem hcf_462_5_1 (a b c : ℕ) (h₁ : a = 462) (h₂ : b = 5) (h₃ : c = 2310) (h₄ : Nat.lcm a b = c) : Nat.gcd a b = 1 := by
  sorry

end hcf_462_5_1_l147_147987


namespace number_of_students_run_red_light_l147_147145

theorem number_of_students_run_red_light :
  let total_students := 300
  let yes_responses := 90
  let odd_id_students := 75
  let coin_probability := 1/2
  -- Calculate using the conditions:
  total_students / 2 - odd_id_students / 2 * coin_probability + total_students / 2 * coin_probability = 30 :=
by
  sorry

end number_of_students_run_red_light_l147_147145


namespace speed_of_stream_l147_147439

variable (b s : ℝ)

theorem speed_of_stream (h1 : 110 = (b + s + 3) * 5)
                        (h2 : 85 = (b - s + 2) * 6) : s = 3.4 :=
by
  sorry

end speed_of_stream_l147_147439


namespace missing_digit_divisible_by_9_l147_147297

theorem missing_digit_divisible_by_9 (x : ℕ) (h : 0 ≤ x ∧ x < 10) : (3 + 5 + 1 + 9 + 2 + x) % 9 = 0 ↔ x = 7 :=
by
  sorry

end missing_digit_divisible_by_9_l147_147297


namespace equation_verification_l147_147758

theorem equation_verification :
  (96 / 12 = 8) ∧ (45 - 37 = 8) := 
by
  -- We can add the necessary proofs later
  sorry

end equation_verification_l147_147758


namespace sum_of_p_and_q_l147_147524

-- Definitions for points and collinearity condition
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := {x := 1, y := 3, z := -2}
def B : Point3D := {x := 2, y := 5, z := 1}
def C (p q : ℝ) : Point3D := {x := p, y := 7, z := q - 2}

def collinear (A B C : Point3D) : Prop :=
  ∃ (k : ℝ), B.x - A.x = k * (C.x - A.x) ∧ B.y - A.y = k * (C.y - A.y) ∧ B.z - A.z = k * (C.z - A.z)

theorem sum_of_p_and_q (p q : ℝ) (h : collinear A B (C p q)) : p + q = 9 := by
  sorry

end sum_of_p_and_q_l147_147524


namespace independent_variable_range_l147_147786

/-- In the function y = 1 / (x - 2), the range of the independent variable x is all real numbers except 2. -/
theorem independent_variable_range (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end independent_variable_range_l147_147786


namespace negative_integers_abs_le_4_l147_147095

theorem negative_integers_abs_le_4 :
  ∀ x : ℤ, x < 0 ∧ |x| ≤ 4 ↔ (x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4) :=
by
  sorry

end negative_integers_abs_le_4_l147_147095


namespace side_length_S2_l147_147036

theorem side_length_S2 (r s : ℕ) (h1 : 2 * r + s = 2260) (h2 : 2 * r + 3 * s = 3782) : s = 761 :=
by
  -- proof omitted
  sorry

end side_length_S2_l147_147036


namespace triangle_inequality_half_perimeter_l147_147001

theorem triangle_inequality_half_perimeter 
  (a b c : ℝ)
  (h_a : a < b + c)
  (h_b : b < a + c)
  (h_c : c < a + b) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := 
sorry

end triangle_inequality_half_perimeter_l147_147001


namespace numbers_must_be_equal_l147_147237

theorem numbers_must_be_equal
  (n : ℕ) (nums : Fin n → ℕ)
  (hn_pos : n = 99)
  (hbound : ∀ i, nums i < 100)
  (hdiv : ∀ (s : Finset (Fin n)) (hs : 2 ≤ s.card), ¬ 100 ∣ s.sum nums) :
  ∀ i j, nums i = nums j := 
sorry

end numbers_must_be_equal_l147_147237


namespace total_payment_correct_l147_147771

theorem total_payment_correct 
  (bob_bill : ℝ) 
  (kate_bill : ℝ) 
  (bob_discount_rate : ℝ) 
  (kate_discount_rate : ℝ) 
  (bob_discount : ℝ := bob_bill * bob_discount_rate / 100) 
  (kate_discount : ℝ := kate_bill * kate_discount_rate / 100) 
  (bob_final_payment : ℝ := bob_bill - bob_discount) 
  (kate_final_payment : ℝ := kate_bill - kate_discount) : 
  (bob_bill = 30) → 
  (kate_bill = 25) → 
  (bob_discount_rate = 5) → 
  (kate_discount_rate = 2) → 
  (bob_final_payment + kate_final_payment = 53) :=
by
  intros
  sorry

end total_payment_correct_l147_147771


namespace number_of_paths_A_to_D_l147_147420

-- Definition of conditions
def ways_A_to_B : Nat := 2
def ways_B_to_C : Nat := 2
def ways_C_to_D : Nat := 2
def direct_A_to_D : Nat := 1

-- Theorem statement for the total number of paths from A to D
theorem number_of_paths_A_to_D : ways_A_to_B * ways_B_to_C * ways_C_to_D + direct_A_to_D = 9 := by
  sorry

end number_of_paths_A_to_D_l147_147420


namespace Elza_winning_strategy_l147_147963

-- Define a hypothetical graph structure
noncomputable def cities := {i : ℕ // 1 ≤ i ∧ i ≤ 2013}
def connected (c1 c2 : cities) : Prop := sorry

theorem Elza_winning_strategy 
  (N : ℕ) 
  (roads : (cities × cities) → Prop) 
  (h1 : ∀ c1 c2, roads (c1, c2) → connected c1 c2)
  (h2 : N = 1006): 
  ∃ (strategy : cities → Prop), 
  (∃ c1 c2 : cities, (strategy c1 ∧ strategy c2)) ∧ connected c1 c2 :=
by 
  sorry

end Elza_winning_strategy_l147_147963


namespace expansion_gameplay_hours_l147_147005

theorem expansion_gameplay_hours :
  let total_gameplay := 100
  let boring_percentage := 80 / 100
  let enjoyable_percentage := 1 - boring_percentage
  let enjoyable_gameplay_original := enjoyable_percentage * total_gameplay
  let enjoyable_gameplay_total := 50
  let expansion_hours := enjoyable_gameplay_total - enjoyable_gameplay_original
  expansion_hours = 30 :=
by
  let total_gameplay := 100
  let boring_percentage := 80 / 100
  let enjoyable_percentage := 1 - boring_percentage
  let enjoyable_gameplay_original := enjoyable_percentage * total_gameplay
  let enjoyable_gameplay_total := 50
  let expansion_hours := enjoyable_gameplay_total - enjoyable_gameplay_original
  show expansion_hours = 30
  sorry

end expansion_gameplay_hours_l147_147005


namespace problem_sin_cos_k_l147_147537

open Real

theorem problem_sin_cos_k {k : ℝ} :
  (∃ x : ℝ, sin x ^ 2 + cos x + k = 0) ↔ -2 ≤ k ∧ k ≤ 0 := by
  sorry

end problem_sin_cos_k_l147_147537


namespace roster_representation_of_M_l147_147647

def M : Set ℚ := {x | ∃ m n : ℤ, x = m / n ∧ |m| < 2 ∧ 1 ≤ n ∧ n ≤ 3}

theorem roster_representation_of_M :
  M = {-1, -1/2, -1/3, 0, 1/2, 1/3} :=
by sorry

end roster_representation_of_M_l147_147647


namespace southton_capsule_depth_l147_147782

theorem southton_capsule_depth :
  ∃ S : ℕ, 4 * S + 12 = 48 ∧ S = 9 :=
by
  sorry

end southton_capsule_depth_l147_147782


namespace area_ratio_of_square_side_multiplied_by_10_l147_147417

theorem area_ratio_of_square_side_multiplied_by_10 (s : ℝ) (A_original A_resultant : ℝ) 
  (h1 : A_original = s^2)
  (h2 : A_resultant = (10 * s)^2) :
  (A_original / A_resultant) = (1 / 100) :=
by
  sorry

end area_ratio_of_square_side_multiplied_by_10_l147_147417


namespace find_four_numbers_l147_147014

theorem find_four_numbers (a b c d : ℕ) 
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7)
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) :
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := 
  by
    sorry

end find_four_numbers_l147_147014


namespace triangle_QR_length_l147_147607

noncomputable def length_PM : ℝ := 6 -- PM = 6 cm
noncomputable def length_MA : ℝ := 12 -- MA = 12 cm
noncomputable def length_NB : ℝ := 9 -- NB = 9 cm
def MN_parallel_PQ : Prop := true -- MN ∥ PQ

theorem triangle_QR_length 
  (h1 : MN_parallel_PQ)
  (h2 : length_PM = 6)
  (h3 : length_MA = 12)
  (h4 : length_NB = 9) : 
  length_QR = 27 :=
sorry

end triangle_QR_length_l147_147607


namespace prob1_part1_prob1_part2_find_integers_x_prob3_min_value_prob4_l147_147287

-- Problem 1
theorem prob1_part1 : |-4 + 6| = 2 := sorry
theorem prob1_part2 : |-2 - 4| = 6 := sorry

-- Problem 2
theorem find_integers_x :
  {x : ℤ | |x + 2| + |x - 1| = 3} = {-2, -1, 0, 1} :=
sorry

-- Problem 3
theorem prob3 (a : ℤ) (h : -4 ≤ a ∧ a ≤ 6) : |a + 4| + |a - 6| = 10 :=
sorry

-- Problem 4
theorem min_value_prob4 :
  ∃ (a : ℤ), |a - 1| + |a + 5| + |a - 4| = 9 ∧ ∀ (b : ℤ), |b - 1| + |b + 5| + |b - 4| ≥ 9 :=
sorry

end prob1_part1_prob1_part2_find_integers_x_prob3_min_value_prob4_l147_147287


namespace n_squared_divides_2n_plus_1_l147_147327

theorem n_squared_divides_2n_plus_1 (n : ℕ) (hn : n > 0) :
  (n ^ 2) ∣ (2 ^ n + 1) ↔ (n = 1 ∨ n = 3) :=
by sorry

end n_squared_divides_2n_plus_1_l147_147327


namespace smallest_possible_gcd_l147_147136

noncomputable def smallestGCD (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 9) : ℕ :=
  Nat.gcd (12 * a) (18 * b)

theorem smallest_possible_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 9) : 
  smallestGCD a b h1 h2 h3 = 54 :=
sorry

end smallest_possible_gcd_l147_147136


namespace base7_to_base5_l147_147860

theorem base7_to_base5 (n : ℕ) (h : n = 305) : 
    3 * 7 ^ 2 + 0 * 7 ^ 1 + 5 = 152 → 152 = 1 * 5 ^ 3 + 1 * 5 ^ 2 + 0 * 5 ^ 1 + 2 * 5 ^ 0 → 305 = 1102 :=
by
  intros h1 h2
  sorry

end base7_to_base5_l147_147860


namespace bowling_ball_weight_l147_147415

theorem bowling_ball_weight :
  (∃ (b c : ℝ), 8 * b = 4 * c ∧ 2 * c = 64) → ∃ b : ℝ, b = 16 :=
by
  sorry

end bowling_ball_weight_l147_147415


namespace janet_extra_cost_l147_147257

theorem janet_extra_cost :
  let clarinet_hourly_rate := 40
  let clarinet_hours_per_week := 3
  let clarinet_weeks_per_year := 50
  let clarinet_yearly_cost := clarinet_hourly_rate * clarinet_hours_per_week * clarinet_weeks_per_year

  let piano_hourly_rate := 28
  let piano_hours_per_week := 5
  let piano_weeks_per_year := 50
  let piano_yearly_cost := piano_hourly_rate * piano_hours_per_week * piano_weeks_per_year
  let piano_discount_rate := 0.10
  let piano_discounted_yearly_cost := piano_yearly_cost * (1 - piano_discount_rate)

  let violin_hourly_rate := 35
  let violin_hours_per_week := 2
  let violin_weeks_per_year := 50
  let violin_yearly_cost := violin_hourly_rate * violin_hours_per_week * violin_weeks_per_year
  let violin_discount_rate := 0.15
  let violin_discounted_yearly_cost := violin_yearly_cost * (1 - violin_discount_rate)

  let singing_hourly_rate := 45
  let singing_hours_per_week := 1
  let singing_weeks_per_year := 50
  let singing_yearly_cost := singing_hourly_rate * singing_hours_per_week * singing_weeks_per_year

  let combined_cost := piano_discounted_yearly_cost + violin_discounted_yearly_cost + singing_yearly_cost
  combined_cost - clarinet_yearly_cost = 5525 := 
  sorry

end janet_extra_cost_l147_147257


namespace find_angle_B_l147_147310

variables {A B C a b c : ℝ} (h1 : 3 * a * Real.cos C = 2 * c * Real.cos A) (h2 : Real.tan A = 1 / 3)

theorem find_angle_B (h1 : 3 * a * Real.cos C = 2 * c * Real.cos A) (h2 : Real.tan A = 1 / 3) : B = 3 * Real.pi / 4 :=
by
  sorry

end find_angle_B_l147_147310


namespace initial_friends_l147_147842

theorem initial_friends (n : ℕ) (h1 : 120 / (n - 4) = 120 / n + 8) : n = 10 := 
by
  sorry

end initial_friends_l147_147842


namespace percentage_of_female_students_25_or_older_l147_147025

theorem percentage_of_female_students_25_or_older
  (T : ℝ) (M F : ℝ) (P : ℝ)
  (h1 : M = 0.40 * T)
  (h2 : F = 0.60 * T)
  (h3 : 0.56 = (0.20 * T) + (0.60 * (1 - P) * T)) :
  P = 0.40 :=
by
  sorry

end percentage_of_female_students_25_or_older_l147_147025


namespace column_sum_correct_l147_147296

theorem column_sum_correct : 
  -- Define x to be the sum of the first column (which is also the minuend of the second column)
  ∃ x : ℕ, 
  -- x should match the expected valid sum provided:
  (x = 1001) := 
sorry

end column_sum_correct_l147_147296


namespace question1_question2_l147_147501

def A (x : ℝ) : Prop := x^2 - 2*x - 3 ≤ 0
def B (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + m^2 - 4 ≤ 0

-- Question 1: If A ∩ B = [1, 3], then m = 3
theorem question1 (m : ℝ) : (∀ x, A x ∧ B m x ↔ (1 ≤ x ∧ x ≤ 3)) → m = 3 :=
sorry

-- Question 2: If A is a subset of the complement of B in ℝ, then m > 5 or m < -3
theorem question2 (m : ℝ) : (∀ x, A x → ¬ B m x) → (m > 5 ∨ m < -3) :=
sorry

end question1_question2_l147_147501


namespace condition_needs_l147_147832

theorem condition_needs (a b c d : ℝ) :
  a + c > b + d → (¬ (a > b ∧ c > d) ∧ (a > b ∧ c > d)) :=
by
  sorry

end condition_needs_l147_147832


namespace cheapest_shipping_option_l147_147589

/-- Defines the cost options for shipping, given a weight of 5 pounds. -/
def cost_A (weight : ℕ) : ℝ := 5.00 + 0.80 * weight
def cost_B (weight : ℕ) : ℝ := 4.50 + 0.85 * weight
def cost_C (weight : ℕ) : ℝ := 3.00 + 0.95 * weight

/-- Proves that for a package weighing 5 pounds, the cheapest shipping option is Option C costing $7.75. -/
theorem cheapest_shipping_option : cost_C 5 < cost_A 5 ∧ cost_C 5 < cost_B 5 ∧ cost_C 5 = 7.75 :=
by
  -- Calculation is omitted
  sorry

end cheapest_shipping_option_l147_147589


namespace trig_order_l147_147797

theorem trig_order (θ : ℝ) (h1 : -Real.pi / 8 < θ) (h2 : θ < 0) : Real.tan θ < Real.sin θ ∧ Real.sin θ < Real.cos θ := 
sorry

end trig_order_l147_147797


namespace wooden_easel_cost_l147_147872

noncomputable def cost_paintbrush : ℝ := 1.5
noncomputable def cost_set_of_paints : ℝ := 4.35
noncomputable def amount_already_have : ℝ := 6.5
noncomputable def additional_amount_needed : ℝ := 12
noncomputable def total_cost_items : ℝ := cost_paintbrush + cost_set_of_paints
noncomputable def total_amount_needed : ℝ := amount_already_have + additional_amount_needed

theorem wooden_easel_cost :
  total_amount_needed - total_cost_items = 12.65 :=
by
  sorry

end wooden_easel_cost_l147_147872


namespace surface_area_comparison_l147_147710

theorem surface_area_comparison (a R : ℝ) (h_eq_volumes : (4 / 3) * Real.pi * R^3 = a^3) :
  6 * a^2 > 4 * Real.pi * R^2 :=
by
  sorry

end surface_area_comparison_l147_147710


namespace simplify_expression_l147_147334

variable (w : ℝ)

theorem simplify_expression : 3 * w + 5 - 6 * w^2 + 4 * w - 7 + 9 * w^2 = 3 * w^2 + 7 * w - 2 := by
  sorry

end simplify_expression_l147_147334


namespace work_completion_time_for_A_l147_147787

-- Define the conditions
def B_completion_time : ℕ := 30
def joint_work_days : ℕ := 4
def work_left_fraction : ℚ := 2 / 3

-- Define the required proof statement
theorem work_completion_time_for_A (x : ℚ) : 
  (4 * (1 / x + 1 / B_completion_time) = 1 / 3) → x = 20 := 
by
  sorry

end work_completion_time_for_A_l147_147787


namespace value_of_k_l147_147369

theorem value_of_k (k : ℝ) :
  (5 + ∑' n : ℕ, (5 + k * (2^n / 4^n))) / 4^n = 10 → k = 15 :=
by
  sorry

end value_of_k_l147_147369


namespace sarah_interviewed_students_l147_147168

theorem sarah_interviewed_students :
  let oranges := 70
  let pears := 120
  let apples := 147
  let strawberries := 113
  oranges + pears + apples + strawberries = 450 := by
sorry

end sarah_interviewed_students_l147_147168


namespace cerulean_survey_l147_147138

theorem cerulean_survey :
  let total_people := 120
  let kind_of_blue := 80
  let kind_and_green := 35
  let neither := 20
  total_people = kind_of_blue + (total_people - kind_of_blue - neither)
  → (kind_and_green + (total_people - kind_of_blue - kind_and_green - neither) + neither) = total_people
  → 55 = (kind_and_green + (total_people - kind_of_blue - kind_and_green - neither)) :=
by
  sorry

end cerulean_survey_l147_147138


namespace problem_1_problem_2_l147_147676

-- First problem: Find the solution set for the inequality |x - 1| + |x + 2| ≥ 5
theorem problem_1 (x : ℝ) : (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) :=
sorry

-- Second problem: Find the range of real number a such that |x - a| + |x + 2| ≤ |x + 4| for all x in [0, 1]
theorem problem_2 (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x - a| + |x + 2| ≤ |x + 4|) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end problem_1_problem_2_l147_147676


namespace gcd_459_357_eq_51_l147_147828

theorem gcd_459_357_eq_51 :
  gcd 459 357 = 51 := 
by
  sorry

end gcd_459_357_eq_51_l147_147828


namespace at_least_one_negative_root_l147_147800

theorem at_least_one_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (x^2 - 6*a*x - 2 + 2*a + 9*a^2 = 0)) ↔ a < (-1 + Real.sqrt 19) / 9 := by
  sorry

end at_least_one_negative_root_l147_147800


namespace sum_of_four_consecutive_integers_is_even_l147_147205

theorem sum_of_four_consecutive_integers_is_even (n : ℤ) : 2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by sorry

end sum_of_four_consecutive_integers_is_even_l147_147205


namespace b_remainder_l147_147093

theorem b_remainder (n : ℕ) (hn : n > 0) : ∃ b : ℕ, b % 11 = 5 :=
by
  sorry

end b_remainder_l147_147093


namespace neither_sufficient_nor_necessary_condition_l147_147323

noncomputable def p (x : ℝ) : Prop := (x - 2) * (x - 1) > 0

noncomputable def q (x : ℝ) : Prop := x - 2 > 0 ∨ x - 1 > 0

theorem neither_sufficient_nor_necessary_condition (x : ℝ) : ¬(p x → q x) ∧ ¬(q x → p x) :=
by
  sorry

end neither_sufficient_nor_necessary_condition_l147_147323


namespace add_to_frac_eq_l147_147127

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l147_147127


namespace find_cos_alpha_l147_147346

theorem find_cos_alpha 
  (α : ℝ) 
  (h₁ : Real.tan (π - α) = 3/4) 
  (h₂ : α ∈ Set.Ioo (π/2) π) 
: Real.cos α = -4/5 :=
sorry

end find_cos_alpha_l147_147346


namespace factorize_polynomial_l147_147224

theorem factorize_polynomial (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3) ^ 2 :=
by sorry

end factorize_polynomial_l147_147224


namespace claudia_total_earnings_l147_147770

def cost_per_beginner_class : Int := 15
def cost_per_advanced_class : Int := 20
def num_beginner_kids_saturday : Int := 20
def num_advanced_kids_saturday : Int := 10
def num_sibling_pairs : Int := 5
def sibling_discount : Int := 3

theorem claudia_total_earnings : 
  let beginner_earnings_saturday := num_beginner_kids_saturday * cost_per_beginner_class
  let advanced_earnings_saturday := num_advanced_kids_saturday * cost_per_advanced_class
  let total_earnings_saturday := beginner_earnings_saturday + advanced_earnings_saturday
  
  let num_beginner_kids_sunday := num_beginner_kids_saturday / 2
  let num_advanced_kids_sunday := num_advanced_kids_saturday / 2
  let beginner_earnings_sunday := num_beginner_kids_sunday * cost_per_beginner_class
  let advanced_earnings_sunday := num_advanced_kids_sunday * cost_per_advanced_class
  let total_earnings_sunday := beginner_earnings_sunday + advanced_earnings_sunday

  let total_earnings_no_discount := total_earnings_saturday + total_earnings_sunday

  let total_sibling_discount := num_sibling_pairs * 2 * sibling_discount
  
  let total_earnings := total_earnings_no_discount - total_sibling_discount
  total_earnings = 720 := 
by
  sorry

end claudia_total_earnings_l147_147770


namespace lisa_balls_count_l147_147912

def stepNumber := 1729

def base7DigitsSum(x : Nat) : Nat :=
  x / 7 ^ 3 + (x % 343) / 7 ^ 2 + (x % 49) / 7 + x % 7

theorem lisa_balls_count (h1 : stepNumber = 1729) : base7DigitsSum stepNumber = 11 := by
  sorry

end lisa_balls_count_l147_147912


namespace problem_l147_147151

theorem problem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 + a*b*c = 4) : 
  a + b + c ≤ 3 := 
sorry

end problem_l147_147151


namespace balance_three_diamonds_l147_147922

-- Define the problem conditions
variables (a b c : ℕ)

-- Four Δ's and two ♦'s will balance twelve ●'s
def condition1 : Prop :=
  4 * a + 2 * b = 12 * c

-- One Δ will balance a ♦ and two ●'s
def condition2 : Prop :=
  a = b + 2 * c

-- Theorem to prove how many ●'s will balance three ♦'s
theorem balance_three_diamonds (h1 : condition1 a b c) (h2 : condition2 a b c) : 3 * b = 2 * c :=
by sorry

end balance_three_diamonds_l147_147922


namespace sum_of_two_consecutive_negative_integers_l147_147440

theorem sum_of_two_consecutive_negative_integers (n : ℤ) (h : n * (n + 1) = 2210) (hn : n < 0) : n + (n + 1) = -95 := 
sorry

end sum_of_two_consecutive_negative_integers_l147_147440


namespace average_speed_second_half_l147_147841

theorem average_speed_second_half
  (d : ℕ) (s1 : ℕ) (t : ℕ)
  (h1 : d = 3600)
  (h2 : s1 = 90)
  (h3 : t = 30) :
  (d / 2) / (t - (d / 2 / s1)) = 180 := by
  sorry

end average_speed_second_half_l147_147841


namespace part1_l147_147248

theorem part1 (a n : ℕ) (hne : a % 2 = 1) : (4 ∣ a^n - 1) → (n % 2 = 0) :=
by
  sorry

end part1_l147_147248


namespace average_blinks_in_normal_conditions_l147_147301

theorem average_blinks_in_normal_conditions (blink_gaming : ℕ) (k : ℚ) (blink_normal : ℚ) 
  (h_blink_gaming : blink_gaming = 10)
  (h_k : k = (3 / 5))
  (h_condition : blink_gaming = blink_normal - k * blink_normal) : 
  blink_normal = 25 := 
by 
  sorry

end average_blinks_in_normal_conditions_l147_147301


namespace max_value_expression_l147_147753

theorem max_value_expression (a b c d : ℝ) 
  (h1 : -11.5 ≤ a ∧ a ≤ 11.5)
  (h2 : -11.5 ≤ b ∧ b ≤ 11.5)
  (h3 : -11.5 ≤ c ∧ c ≤ 11.5)
  (h4 : -11.5 ≤ d ∧ d ≤ 11.5):
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 552 :=
by
  sorry

end max_value_expression_l147_147753


namespace determine_n_l147_147147

theorem determine_n (k : ℕ) (n : ℕ) (h1 : 21^k ∣ n) (h2 : 7^k - k^7 = 1) : n = 1 :=
sorry

end determine_n_l147_147147


namespace maxwell_walking_speed_l147_147898

theorem maxwell_walking_speed :
  ∃ v : ℝ, (8 * v + 6 * 7 = 74) ∧ v = 4 :=
by
  exists 4
  constructor
  { norm_num }
  rfl

end maxwell_walking_speed_l147_147898


namespace arithmetic_progression_sum_squares_l147_147484

theorem arithmetic_progression_sum_squares (a1 a2 a3 : ℚ)
  (h1 : a2 = (a1 + a3) / 2)
  (h2 : a1 + a2 + a3 = 2)
  (h3 : a1^2 + a2^2 + a3^2 = 14/9) :
  (a1 = 1/3 ∧ a2 = 2/3 ∧ a3 = 1) ∨ (a1 = 1 ∧ a2 = 2/3 ∧ a3 = 1/3) :=
sorry

end arithmetic_progression_sum_squares_l147_147484


namespace sets_of_headphones_l147_147808

-- Definitions of the conditions
variable (M H : ℕ)

-- Theorem statement for proving the question given the conditions
theorem sets_of_headphones (h1 : 5 * M + 30 * H = 840) (h2 : 3 * M + 120 = 480) : H = 8 := by
  sorry

end sets_of_headphones_l147_147808


namespace find_x1_value_l147_147255

theorem find_x1_value (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3) (h2 : x3 ≤ x2) (h3 : x2 ≤ x1) (h4 : x1 ≤ 1) 
  (h_eq : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 3) : 
  x1 = 2 / 3 := 
sorry

end find_x1_value_l147_147255


namespace number_of_trees_is_eleven_l147_147499

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l147_147499


namespace B_share_correct_l147_147208

noncomputable def total_share : ℕ := 120
noncomputable def B_share : ℕ := 20
noncomputable def A_share (x : ℕ) : ℕ := x + 20
noncomputable def C_share (x : ℕ) : ℕ := x + 40

theorem B_share_correct : ∃ x : ℕ, total_share = (A_share x) + x + (C_share x) ∧ x = B_share := by
  sorry

end B_share_correct_l147_147208


namespace botanical_garden_correct_path_length_l147_147806

noncomputable def correct_path_length_on_ground
  (inch_length_on_map : ℝ)
  (inch_per_error_segment : ℝ)
  (conversion_rate : ℝ) : ℝ :=
  (inch_length_on_map * conversion_rate) - (inch_per_error_segment * conversion_rate)

theorem botanical_garden_correct_path_length :
  correct_path_length_on_ground 6.5 0.75 1200 = 6900 := 
by
  sorry

end botanical_garden_correct_path_length_l147_147806


namespace constant_sum_of_distances_l147_147421

open Real

theorem constant_sum_of_distances (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
    (ellipse_condition : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → ∀ A B : ℝ × ℝ, A.2 > 0 ∧ B.2 > 0)
    (foci : (ℝ × ℝ) × (ℝ × ℝ) := ((-c, 0), (c, 0)))
    (points_AB : ∃ (A B : ℝ × ℝ), A.2 > 0 ∧ B.2 > 0 ∧ (A.1 - c)^2 / a^2 + A.2^2 / b^2 = 1 ∧ (B.1 - -c)^2 / a^2 + B.2^2 / b^2 = 1)
    (AF1_parallel_BF2 : ∀ (A B : ℝ × ℝ), (A.1 - -c) * (B.2 - 0) - (A.2 - 0) * (B.1 - c) = 0)
    (intersection_P: ∀ (A B : ℝ × ℝ), ∃ P : ℝ × ℝ, ((A.1 - c) * (B.2 - 0) = (A.2 - 0) * (P.1 - c)) ∧ ((B.1 - -c) * (A.2 - 0) = (B.2 - 0) * (P.1 - -c))) :
    ∃ k : ℝ, ∀ (P : ℝ × ℝ), dist P (foci.fst) + dist P (foci.snd) = k := 
sorry

end constant_sum_of_distances_l147_147421


namespace pow_mod_26_l147_147411

theorem pow_mod_26 (a b n : ℕ) (hn : n = 2023) (h₁ : a = 17) (h₂ : b = 26) :
  a ^ n % b = 7 := by
  sorry

end pow_mod_26_l147_147411


namespace population_ratio_l147_147106

variables (Px Py Pz : ℕ)

theorem population_ratio (h1 : Py = 2 * Pz) (h2 : Px = 8 * Py) : Px / Pz = 16 :=
by
  sorry

end population_ratio_l147_147106


namespace weight_of_second_piece_of_wood_l147_147789

/--
Given: 
1) The density and thickness of the wood are uniform.
2) The first piece of wood is a square with a side length of 3 inches and a weight of 15 ounces.
3) The second piece of wood is a square with a side length of 6 inches.
Theorem: 
The weight of the second piece of wood is 60 ounces.
-/
theorem weight_of_second_piece_of_wood (s1 s2 w1 w2 : ℕ) (h1 : s1 = 3) (h2 : w1 = 15) (h3 : s2 = 6) :
  w2 = 60 :=
sorry

end weight_of_second_piece_of_wood_l147_147789


namespace original_number_is_144_l147_147089

theorem original_number_is_144 :
  ∃ (A B C : ℕ), A ≠ 0 ∧
  (100 * A + 11 * B = 144) ∧
  (A * B^2 = 10 * A + C) ∧
  (A * C = C) ∧
  A = 1 ∧ B = 4 ∧ C = 6 :=
by
  sorry

end original_number_is_144_l147_147089


namespace simplify_expr1_simplify_expr2_l147_147948

-- First expression
theorem simplify_expr1 (a b : ℝ) : a * (a - b) - (a + b) * (a - 2 * b) = 2 * b ^ 2 :=
by
  sorry

-- Second expression
theorem simplify_expr2 (x : ℝ) : 
  ( ( (4 * x - 9) / (3 - x) - x + 3 ) / ( (x ^ 2 - 4) / (x - 3) ) ) = - (x / (x + 2)) :=
by
  sorry

end simplify_expr1_simplify_expr2_l147_147948


namespace sally_garden_area_l147_147656

theorem sally_garden_area :
  ∃ (a b : ℕ), 2 * (a + b) = 24 ∧ b + 1 = 3 * (a + 1) ∧ 
     (3 * (a - 1) * 3 * (b - 1) = 297) :=
by {
  sorry
}

end sally_garden_area_l147_147656


namespace harmonic_mean_pairs_l147_147508

theorem harmonic_mean_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) 
    (hmean : (2 * x * y) / (x + y) = 2^30) :
    (∃! n, n = 29) :=
by
  sorry

end harmonic_mean_pairs_l147_147508


namespace volume_of_given_solid_l147_147143

noncomputable def volume_of_solid (s : ℝ) (h : ℝ) : ℝ :=
  (h / 3) * (s^2 + (s * (3 / 2))^2 + (s * (3 / 2)) * s)

theorem volume_of_given_solid : volume_of_solid 8 10 = 3040 / 3 :=
by
  sorry

end volume_of_given_solid_l147_147143


namespace find_x_satisfies_equation_l147_147976

theorem find_x_satisfies_equation :
  let x : ℤ := -14
  ∃ x : ℤ, (36 - x) - (14 - x) = 2 * ((36 - x) - (18 - x)) :=
by
  let x := -14
  use x
  sorry

end find_x_satisfies_equation_l147_147976


namespace RachelStillToColor_l147_147884

def RachelColoringBooks : Prop :=
  let initial_books := 23 + 32
  let colored := 44
  initial_books - colored = 11

theorem RachelStillToColor : RachelColoringBooks := 
  by
    let initial_books := 23 + 32
    let colored := 44
    show initial_books - colored = 11
    sorry

end RachelStillToColor_l147_147884


namespace max_earnings_mary_l147_147675

def wage_rate : ℝ := 8
def first_hours : ℕ := 20
def max_hours : ℕ := 80
def regular_tip_rate : ℝ := 2
def overtime_rate_increase : ℝ := 1.25
def overtime_tip_rate : ℝ := 3
def overtime_bonus_threshold : ℕ := 5
def overtime_bonus_amount : ℝ := 20

noncomputable def total_earnings (hours : ℕ) : ℝ :=
  let regular_hours := min hours first_hours
  let overtime_hours := if hours > first_hours then hours - first_hours else 0
  let overtime_blocks := overtime_hours / overtime_bonus_threshold
  let regular_earnings := regular_hours * (wage_rate + regular_tip_rate)
  let overtime_earnings := overtime_hours * (wage_rate * overtime_rate_increase + overtime_tip_rate)
  let bonuses := (overtime_blocks) * overtime_bonus_amount
  regular_earnings + overtime_earnings + bonuses

theorem max_earnings_mary : total_earnings max_hours = 1220 := by
  sorry

end max_earnings_mary_l147_147675


namespace smallest_perfect_square_divisible_by_2_3_5_l147_147822

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l147_147822


namespace probability_of_selecting_red_books_is_3_div_14_l147_147466

-- Define the conditions
def total_books : ℕ := 8
def red_books : ℕ := 4
def blue_books : ℕ := 4
def books_selected : ℕ := 2

-- Define the calculation of the probability
def probability_red_books_selected : ℚ :=
  let total_outcomes := Nat.choose total_books books_selected
  let favorable_outcomes := Nat.choose red_books books_selected
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- State the theorem
theorem probability_of_selecting_red_books_is_3_div_14 :
  probability_red_books_selected = 3 / 14 :=
by
  sorry

end probability_of_selecting_red_books_is_3_div_14_l147_147466


namespace example_problem_l147_147721

def Z (x y : ℝ) : ℝ := x^2 - 3 * x * y + y^2

theorem example_problem :
  Z 4 3 = -11 := 
by
  -- proof goes here
  sorry

end example_problem_l147_147721


namespace minimum_waste_l147_147709

/-- Zenobia's cookout problem setup -/
def LCM_hot_dogs_buns : Nat := Nat.lcm 10 12

def hot_dog_packages : Nat := LCM_hot_dogs_buns / 10
def bun_packages : Nat := LCM_hot_dogs_buns / 12

def waste_hot_dog_packages : ℝ := hot_dog_packages * 0.4
def waste_bun_packages : ℝ := bun_packages * 0.3
def total_waste : ℝ := waste_hot_dog_packages + waste_bun_packages

theorem minimum_waste :
  hot_dog_packages = 6 ∧ bun_packages = 5 ∧ total_waste = 3.9 :=
by
  sorry

end minimum_waste_l147_147709


namespace proof_inequality_l147_147408

noncomputable def problem (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1 → a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d

theorem proof_inequality (a b c d : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_inequality_l147_147408


namespace solve_system_of_equations_l147_147682

theorem solve_system_of_equations (x y z : ℝ) :
  (x * y = z) ∧ (x * z = y) ∧ (y * z = x) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = 1 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = -1) ∨
  (x = -1 ∧ y = -1 ∧ z = 1) ∨
  (x = 0 ∧ y = 0 ∧ z = 0) := by
  sorry

end solve_system_of_equations_l147_147682


namespace quadratic_distinct_real_roots_l147_147260

theorem quadratic_distinct_real_roots (k : ℝ) :
  ((k - 1) * x^2 + 6 * x + 3 = 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k - 1) * x1^2 + 6 * x1 + 3 = 0) ∧ ((k - 1) * x2^2 + 6 * x2 + 3 = 0)) ↔ (k < 4 ∧ k ≠ 1) :=
by {
  sorry
}

end quadratic_distinct_real_roots_l147_147260


namespace amount_diana_owes_l147_147597

-- Problem definitions
def principal : ℝ := 75
def rate : ℝ := 0.07
def time : ℝ := 1
def interest := principal * rate * time
def total_owed := principal + interest

-- Theorem to prove that the total amount owed is $80.25
theorem amount_diana_owes : total_owed = 80.25 := by
  sorry

end amount_diana_owes_l147_147597


namespace equation_of_line_l147_147945

theorem equation_of_line (l : ℝ → ℝ) :
  (l 1 = 2 ∧ (∃ a : ℝ, l 0 = 2 * a ∧ a ≠ 0 ∧ ∀ x : ℝ, l x = (2 * l a / a) * x))
  ∨ (l 1 = 2 ∧ (∃ a : ℝ, l 0 = 2 * a ∧ a ≠ 0 ∧ ∀ x y : ℝ, 2 * x + y - 4 = 0)) := sorry

end equation_of_line_l147_147945


namespace rational_expression_l147_147617

theorem rational_expression {x : ℚ} : (∃ a : ℚ, x / (x^2 + x + 1) = a) → (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) := by
  sorry

end rational_expression_l147_147617


namespace derivative_at_one_l147_147295

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * Real.log x

theorem derivative_at_one : (deriv f 1) = 5 := 
by 
  sorry

end derivative_at_one_l147_147295


namespace min_value_of_A_sq_sub_B_sq_l147_147286

noncomputable def A (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 13)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (2 * x + 2) + Real.sqrt (2 * y + 2) + Real.sqrt (2 * z + 2)

theorem min_value_of_A_sq_sub_B_sq (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  A x y z ^ 2 - B x y z ^ 2 ≥ 36 :=
  sorry

end min_value_of_A_sq_sub_B_sq_l147_147286


namespace michael_birth_year_l147_147042

theorem michael_birth_year (first_AMC8_year : ℕ) (tenth_AMC8_year : ℕ) (age_during_tenth_AMC8 : ℕ) 
  (h1 : first_AMC8_year = 1985) (h2 : tenth_AMC8_year = (first_AMC8_year + 9)) (h3 : age_during_tenth_AMC8 = 15) :
  (tenth_AMC8_year - age_during_tenth_AMC8) = 1979 :=
by
  sorry

end michael_birth_year_l147_147042


namespace alpha_more_economical_l147_147102

theorem alpha_more_economical (n : ℕ) : n ≥ 12 → 80 + 12 * n < 10 + 18 * n := 
by
  sorry

end alpha_more_economical_l147_147102


namespace double_average_l147_147019

theorem double_average (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (h1 : n = 25) (h2 : initial_avg = 70) (h3 : new_avg * n = 2 * (initial_avg * n)) : new_avg = 140 :=
sorry

end double_average_l147_147019


namespace B_visible_from_A_l147_147347

noncomputable def visibility_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 → 4 * x - 2 > 2 * x^2

theorem B_visible_from_A (a : ℝ) : visibility_condition a ↔ a < 10 :=
by
  -- sorry statement is used to skip the proof part.
  sorry

end B_visible_from_A_l147_147347


namespace simplify_expr_l147_147543

variable (x : ℝ)

theorem simplify_expr : (2 * x^2 + 5 * x - 7) - (x^2 + 9 * x - 3) = x^2 - 4 * x - 4 :=
by
  sorry

end simplify_expr_l147_147543


namespace Vince_ride_longer_l147_147706

def Vince_ride_length : ℝ := 0.625
def Zachary_ride_length : ℝ := 0.5

theorem Vince_ride_longer : Vince_ride_length - Zachary_ride_length = 0.125 := by
  sorry

end Vince_ride_longer_l147_147706


namespace find_parallelogram_base_length_l147_147216

variable (A h b : ℕ)
variable (parallelogram_area : A = 240)
variable (parallelogram_height : h = 10)
variable (area_formula : A = b * h)

theorem find_parallelogram_base_length : b = 24 :=
by
  have h₁ : A = 240 := parallelogram_area
  have h₂ : h = 10 := parallelogram_height
  have h₃ : A = b * h := area_formula
  sorry

end find_parallelogram_base_length_l147_147216


namespace gcd_930_868_l147_147181

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end gcd_930_868_l147_147181


namespace tory_sold_to_neighbor_l147_147729

def total_cookies : ℕ := 50
def sold_to_grandmother : ℕ := 12
def sold_to_uncle : ℕ := 7
def to_be_sold : ℕ := 26

def sold_to_neighbor : ℕ :=
  total_cookies - to_be_sold - (sold_to_grandmother + sold_to_uncle)

theorem tory_sold_to_neighbor :
  sold_to_neighbor = 5 :=
by
  intros
  sorry

end tory_sold_to_neighbor_l147_147729


namespace cherries_per_quart_of_syrup_l147_147064

-- Definitions based on conditions
def time_to_pick_cherries : ℚ := 2
def cherries_picked_in_time : ℚ := 300
def time_to_make_syrup : ℚ := 3
def total_time_for_all_syrup : ℚ := 33
def total_quarts : ℚ := 9

-- Derivation of how many cherries are needed per quart
theorem cherries_per_quart_of_syrup : 
  (cherries_picked_in_time / time_to_pick_cherries) * (total_time_for_all_syrup - total_quarts * time_to_make_syrup) / total_quarts = 100 :=
by
  repeat { sorry }

end cherries_per_quart_of_syrup_l147_147064


namespace prize_difference_l147_147593

def mateo_hourly_rate : ℕ := 20
def sydney_daily_rate : ℕ := 400
def hours_in_a_week : ℕ := 24 * 7
def days_in_a_week : ℕ := 7

def mateo_total : ℕ := mateo_hourly_rate * hours_in_a_week
def sydney_total : ℕ := sydney_daily_rate * days_in_a_week

def difference_amount : ℕ := 560

theorem prize_difference : mateo_total - sydney_total = difference_amount := sorry

end prize_difference_l147_147593


namespace correct_subsidy_equation_l147_147933

-- Define the necessary variables and conditions
def sales_price (x : ℝ) := x  -- sales price of the mobile phone in yuan
def subsidy_rate : ℝ := 0.13  -- 13% subsidy rate
def number_of_phones : ℝ := 20  -- 20 units sold
def total_subsidy : ℝ := 2340  -- total subsidy provided

-- Lean theorem statement to prove the correct equation
theorem correct_subsidy_equation (x : ℝ) :
  number_of_phones * x * subsidy_rate = total_subsidy :=
by
  sorry -- proof to be completed

end correct_subsidy_equation_l147_147933


namespace root_of_polynomial_l147_147701

theorem root_of_polynomial :
  ∀ x : ℝ, (x^2 - 3 * x + 2) * x * (x - 4) = 0 ↔ (x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 4) :=
by 
  sorry

end root_of_polynomial_l147_147701


namespace least_possible_value_of_z_minus_x_l147_147894

theorem least_possible_value_of_z_minus_x 
  (x y z : ℤ) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : y - x > 5) 
  (h4 : ∃ n : ℤ, x = 2 * n)
  (h5 : ∃ m : ℤ, y = 2 * m + 1) 
  (h6 : ∃ k : ℤ, z = 2 * k + 1) : 
  z - x = 9 := 
sorry

end least_possible_value_of_z_minus_x_l147_147894


namespace steel_bar_lengths_l147_147419

theorem steel_bar_lengths
  (x y z : ℝ)
  (h1 : 2 * x + y + 3 * z = 23)
  (h2 : x + 4 * y + 5 * z = 36) :
  x + 2 * y + 3 * z = 22 := 
sorry

end steel_bar_lengths_l147_147419


namespace spherical_distance_between_points_l147_147314

noncomputable def spherical_distance (R : ℝ) (α : ℝ) : ℝ :=
  α * R

theorem spherical_distance_between_points 
  (R : ℝ) 
  (α : ℝ) 
  (hR : R > 0) 
  (hα : α = π / 6) : 
  spherical_distance R α = (π / 6) * R :=
by
  rw [hα]
  unfold spherical_distance
  ring

end spherical_distance_between_points_l147_147314


namespace total_clothes_l147_147459

-- Defining the conditions
def shirts := 12
def pants := 5 * shirts
def shorts := (1 / 4) * pants

-- Theorem to prove the total number of pieces of clothes
theorem total_clothes : shirts + pants + shorts = 87 := by
  -- using sorry to skip the proof
  sorry

end total_clothes_l147_147459


namespace part_i_l147_147442

theorem part_i (n : ℕ) (a : Fin (n+1) → ℤ) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (a i - a j) % n = 0 := by
  sorry

end part_i_l147_147442


namespace least_faces_triangular_pyramid_l147_147788

def triangular_prism_faces : ℕ := 5
def quadrangular_prism_faces : ℕ := 6
def triangular_pyramid_faces : ℕ := 4
def quadrangular_pyramid_faces : ℕ := 5
def truncated_quadrangular_pyramid_faces : ℕ := 5 -- assuming the minimum possible value

theorem least_faces_triangular_pyramid :
  triangular_pyramid_faces < triangular_prism_faces ∧
  triangular_pyramid_faces < quadrangular_prism_faces ∧
  triangular_pyramid_faces < quadrangular_pyramid_faces ∧
  triangular_pyramid_faces ≤ truncated_quadrangular_pyramid_faces :=
by
  sorry

end least_faces_triangular_pyramid_l147_147788


namespace ten_years_less_than_average_age_l147_147938

theorem ten_years_less_than_average_age (L : ℕ) :
  (2 * L - 14) = 
    (2 * L - 4) - 10 :=
by {
  sorry
}

end ten_years_less_than_average_age_l147_147938


namespace fencing_problem_l147_147576

theorem fencing_problem (W L : ℝ) (hW : W = 40) (hArea : W * L = 320) : 
  2 * L + W = 56 :=
by
  sorry

end fencing_problem_l147_147576


namespace multiple_time_second_artifact_is_three_l147_147657

-- Define the conditions as Lean definitions
def months_in_year : ℕ := 12
def total_time_both_artifacts_years : ℕ := 10
def total_time_first_artifact_months : ℕ := 6 + 24

-- Convert total time of both artifacts from years to months
def total_time_both_artifacts_months : ℕ := total_time_both_artifacts_years * months_in_year

-- Define the time for the second artifact
def time_second_artifact_months : ℕ :=
  total_time_both_artifacts_months - total_time_first_artifact_months

-- Define the sought multiple
def multiple_second_first : ℕ :=
  time_second_artifact_months / total_time_first_artifact_months

-- The theorem stating the required proof
theorem multiple_time_second_artifact_is_three :
  multiple_second_first = 3 :=
by
  sorry

end multiple_time_second_artifact_is_three_l147_147657


namespace boat_speed_in_still_water_l147_147357

open Real

theorem boat_speed_in_still_water (V_s d t : ℝ) (h1 : V_s = 6) (h2 : d = 72) (h3 : t = 3.6) :
  ∃ (V_b : ℝ), V_b = 14 := by
  have V_d := d / t
  have V_b := V_d - V_s
  use V_b
  sorry

end boat_speed_in_still_water_l147_147357


namespace swimming_pool_width_l147_147409

theorem swimming_pool_width 
  (V_G : ℝ) (G_CF : ℝ) (height_inch : ℝ) (L : ℝ) (V_CF : ℝ) (height_ft : ℝ) (A : ℝ) (W : ℝ) :
  V_G = 3750 → G_CF = 7.48052 → height_inch = 6 → L = 40 →
  V_CF = V_G / G_CF → height_ft = height_inch / 12 →
  A = L * W → V_CF = A * height_ft →
  W = 25.067 :=
by
  intros hV hG hH hL hVC hHF hA hVF
  sorry

end swimming_pool_width_l147_147409


namespace sandy_spent_on_shirt_l147_147184

-- Define the conditions
def cost_of_shorts : ℝ := 13.99
def cost_of_jacket : ℝ := 7.43
def total_spent_on_clothes : ℝ := 33.56

-- Define the amount spent on the shirt
noncomputable def cost_of_shirt : ℝ :=
  total_spent_on_clothes - (cost_of_shorts + cost_of_jacket)

-- Prove that Sandy spent $12.14 on the shirt
theorem sandy_spent_on_shirt : cost_of_shirt = 12.14 :=
by
  sorry

end sandy_spent_on_shirt_l147_147184


namespace chuck_play_area_l147_147391

-- Define the conditions for the problem in Lean
def shed_length1 : ℝ := 3
def shed_length2 : ℝ := 4
def leash_length : ℝ := 4

-- State the theorem we want to prove
theorem chuck_play_area :
  let sector_area1 := (3 / 4) * Real.pi * (leash_length ^ 2)
  let sector_area2 := (1 / 4) * Real.pi * (1 ^ 2)
  sector_area1 + sector_area2 = (49 / 4) * Real.pi := 
by
  -- The proof is omitted for brevity
  sorry

end chuck_play_area_l147_147391


namespace minimum_inverse_sum_l147_147636

theorem minimum_inverse_sum (a b : ℝ) (h1 : (a > 0) ∧ (b > 0)) 
  (h2 : 3 * a + 4 * b = 55) : 
  (1 / a) + (1 / b) ≥ (7 + 4 * Real.sqrt 3) / 55 :=
sorry

end minimum_inverse_sum_l147_147636


namespace combined_age_in_years_l147_147231

theorem combined_age_in_years (years : ℕ) (adam_age : ℕ) (tom_age : ℕ) (target_age : ℕ) :
  adam_age = 8 → tom_age = 12 → target_age = 44 → (adam_age + tom_age) + 2 * years = target_age → years = 12 :=
by
  intros h_adam h_tom h_target h_combined
  rw [h_adam, h_tom, h_target] at h_combined
  linarith

end combined_age_in_years_l147_147231


namespace greatest_perimeter_of_triangle_l147_147741

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l147_147741


namespace storks_more_than_birds_l147_147206

theorem storks_more_than_birds :
  let initial_birds := 3
  let additional_birds := 2
  let storks := 6
  storks - (initial_birds + additional_birds) = 1 :=
by
  sorry

end storks_more_than_birds_l147_147206


namespace min_value_f_when_a_eq_1_no_extrema_implies_a_ge_four_thirds_l147_147381

section
variables {a x : ℝ}

/-- Define the function f(x) = ax^3 - 2x^2 + x + c where c = 1 -/
def f (a x : ℝ) : ℝ := a * x^3 - 2 * x^2 + x + 1

/-- Proposition 1: Minimum value of f when a = 1 and f passes through (0,1) is 1 -/
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f 1 x ≥ 1) := 
by {
  -- Sorry for the full proof
  sorry
}

/-- Proposition 2: If f has no extremum points, then a ≥ 4/3 -/
theorem no_extrema_implies_a_ge_four_thirds (h : ∀ x : ℝ, 3 * a * x^2 - 4 * x + 1 ≠ 0) : 
  a ≥ (4 / 3) :=
by {
  -- Sorry for the full proof
  sorry
}

end

end min_value_f_when_a_eq_1_no_extrema_implies_a_ge_four_thirds_l147_147381


namespace initial_eggs_proof_l147_147802

noncomputable def initial_eggs (total_cost : ℝ) (price_per_egg : ℝ) (leftover_eggs : ℝ) : ℝ :=
  let eggs_sold := total_cost / price_per_egg
  eggs_sold + leftover_eggs

theorem initial_eggs_proof : initial_eggs 5 0.20 5 = 30 := by
  sorry

end initial_eggs_proof_l147_147802


namespace number_of_laborers_in_crew_l147_147504

theorem number_of_laborers_in_crew (present : ℕ) (percentage : ℝ) (total : ℕ) 
    (h1 : present = 70) (h2 : percentage = 44.9 / 100) (h3 : present = percentage * total) : 
    total = 156 := 
sorry

end number_of_laborers_in_crew_l147_147504


namespace internal_angle_sine_l147_147943

theorem internal_angle_sine (α : ℝ) (h1 : α > 0 ∧ α < 180) (h2 : Real.sin (α * (Real.pi / 180)) = 1 / 2) : α = 30 ∨ α = 150 :=
sorry

end internal_angle_sine_l147_147943


namespace largest_number_less_than_2_l147_147973

theorem largest_number_less_than_2 (a b c : ℝ) (h_a : a = 0.8) (h_b : b = 1/2) (h_c : c = 0.5) : 
  a < 2 ∧ b < 2 ∧ c < 2 ∧ (∀ x, (x = a ∨ x = b ∨ x = c) → x < 2) → 
  a = 0.8 ∧ 
  (a > b ∧ a > c) ∧ 
  (a < 2) :=
by sorry

end largest_number_less_than_2_l147_147973


namespace gcd_ab_l147_147907

def a : ℕ := 130^2 + 215^2 + 310^2
def b : ℕ := 131^2 + 216^2 + 309^2

theorem gcd_ab : Nat.gcd a b = 1 := by
  sorry

end gcd_ab_l147_147907


namespace thomas_friends_fraction_l147_147633

noncomputable def fraction_of_bars_taken (x : ℝ) (initial_bars : ℝ) (returned_bars : ℝ) 
  (piper_bars : ℝ) (remaining_bars : ℝ) : ℝ :=
  x / initial_bars

theorem thomas_friends_fraction 
  (initial_bars : ℝ)
  (total_taken_by_all : ℝ)
  (returned_bars : ℝ)
  (piper_bars : ℝ)
  (remaining_bars : ℝ)
  (h_initial : initial_bars = 200)
  (h_remaining : remaining_bars = 110)
  (h_taken : 200 - 110 = 90)
  (h_total_taken_by_all : total_taken_by_all = 90)
  (h_returned : returned_bars = 5)
  (h_x_calculation : 2 * (total_taken_by_all + returned_bars - initial_bars) + initial_bars = total_taken_by_all + returned_bars)
  : fraction_of_bars_taken ((total_taken_by_all + returned_bars - initial_bars) + 2 * initial_bars) initial_bars returned_bars piper_bars remaining_bars = 21 / 80 :=
  sorry

end thomas_friends_fraction_l147_147633


namespace initial_blocks_l147_147004

theorem initial_blocks (used_blocks remaining_blocks : ℕ) (h1 : used_blocks = 25) (h2 : remaining_blocks = 72) : 
  used_blocks + remaining_blocks = 97 := by
  sorry

end initial_blocks_l147_147004


namespace calculate_number_of_sides_l147_147793

theorem calculate_number_of_sides (n : ℕ) (h : n ≥ 6) :
  ((6 : ℚ) / n^2) * ((6 : ℚ) / n^2) = 0.027777777777777776 →
  n = 6 :=
by
  sorry

end calculate_number_of_sides_l147_147793


namespace cross_fills_space_without_gaps_l147_147197

structure Cube :=
(x : ℤ)
(y : ℤ)
(z : ℤ)

structure Cross :=
(center : Cube)
(adjacent : List Cube)

def is_adjacent (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ abs (c1.z - c2.z) = 1) ∨
  (c1.x = c2.x ∧ abs (c1.y - c2.y) = 1 ∧ c1.z = c2.z) ∨
  (abs (c1.x - c2.x) = 1 ∧ c1.y = c2.y ∧ c1.z = c2.z)

def valid_cross (c : Cross) : Prop :=
  ∀ (adj : Cube), adj ∈ c.adjacent → is_adjacent c.center adj

def fills_space (crosses : List Cross) : Prop :=
  ∀ (pos : Cube), ∃ (c : Cross), c ∈ crosses ∧ 
    (pos = c.center ∨ pos ∈ c.adjacent)

theorem cross_fills_space_without_gaps 
  (crosses : List Cross) 
  (Hcross : ∀ c ∈ crosses, valid_cross c) : 
  fills_space crosses :=
sorry

end cross_fills_space_without_gaps_l147_147197


namespace break_25_ruble_bill_l147_147026

theorem break_25_ruble_bill (x y z : ℕ) :
  (x + y + z = 11 ∧ 1 * x + 3 * y + 5 * z = 25) ↔ 
    (x = 4 ∧ y = 7 ∧ z = 0) ∨ 
    (x = 5 ∧ y = 5 ∧ z = 1) ∨ 
    (x = 6 ∧ y = 3 ∧ z = 2) ∨ 
    (x = 7 ∧ y = 1 ∧ z = 3) :=
sorry

end break_25_ruble_bill_l147_147026


namespace number_of_n_divisible_by_prime_lt_20_l147_147870

theorem number_of_n_divisible_by_prime_lt_20 (N : ℕ) : 
  (N = 69) :=
by
  sorry

end number_of_n_divisible_by_prime_lt_20_l147_147870


namespace sum_ratio_arithmetic_sequence_l147_147690

theorem sum_ratio_arithmetic_sequence (a₁ d : ℚ) (h : d ≠ 0) 
  (S : ℕ → ℚ)
  (h_sum : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2)
  (h_ratio : S 3 / S 6 = 1 / 3) :
  S 6 / S 12 = 3 / 10 :=
by
  sorry

end sum_ratio_arithmetic_sequence_l147_147690


namespace remainder_when_divided_by_5_l147_147142

-- Definitions of the conditions
def condition1 (N : ℤ) : Prop := ∃ R1 : ℤ, N = 5 * 2 + R1
def condition2 (N : ℤ) : Prop := ∃ Q2 : ℤ, N = 4 * Q2 + 2

-- Statement to prove
theorem remainder_when_divided_by_5 (N : ℤ) (R1 : ℤ) (Q2 : ℤ) :
  (N = 5 * 2 + R1) ∧ (N = 4 * Q2 + 2) → (R1 = 4) :=
by
  sorry

end remainder_when_divided_by_5_l147_147142


namespace non_degenerate_triangles_l147_147549

theorem non_degenerate_triangles :
  let total_points := 16
  let collinear_points := 5
  let total_triangles := Nat.choose total_points 3
  let degenerate_triangles := 2 * Nat.choose collinear_points 3
  let nondegenerate_triangles := total_triangles - degenerate_triangles
  nondegenerate_triangles = 540 := 
by
  sorry

end non_degenerate_triangles_l147_147549


namespace find_b_l147_147858

-- Definitions based on the conditions in the problem
def eq1 (a : ℝ) := 3 * a + 3 = 0
def eq2 (a b : ℝ) := 2 * b - a = 4

-- Statement of the proof problem
theorem find_b (a b : ℝ) (h1 : eq1 a) (h2 : eq2 a b) : b = 3 / 2 :=
by
  sorry

end find_b_l147_147858


namespace balloons_remaining_l147_147128
-- Importing the necessary libraries

-- Defining the conditions
def originalBalloons : Nat := 709
def givenBalloons : Nat := 221

-- Stating the theorem
theorem balloons_remaining : originalBalloons - givenBalloons = 488 := by
  sorry

end balloons_remaining_l147_147128


namespace longer_bus_ride_l147_147975

theorem longer_bus_ride :
  let oscar := 0.75
  let charlie := 0.25
  oscar - charlie = 0.50 :=
by
  sorry

end longer_bus_ride_l147_147975


namespace youngest_child_age_l147_147726

variable (Y : ℕ) (O : ℕ) -- Y: the youngest child's present age
variable (P₀ P₁ P₂ P₃ : ℕ) -- P₀, P₁, P₂, P₃: the present ages of the 4 original family members

-- Conditions translated to Lean
variable (h₁ : ((P₀ - 10) + (P₁ - 10) + (P₂ - 10) + (P₃ - 10)) / 4 = 24)
variable (h₂ : O = Y + 2)
variable (h₃ : ((P₀ + P₁ + P₂ + P₃) + Y + O) / 6 = 24)

theorem youngest_child_age (h₁ : ((P₀ - 10) + (P₁ - 10) + (P₂ - 10) + (P₃ - 10)) / 4 = 24)
                       (h₂ : O = Y + 2)
                       (h₃ : ((P₀ + P₁ + P₂ + P₃) + Y + O) / 6 = 24) :
  Y = 3 := by 
  sorry

end youngest_child_age_l147_147726


namespace cost_of_fencing_irregular_pentagon_l147_147868

noncomputable def total_cost_fencing (AB BC CD DE AE : ℝ) (cost_per_meter : ℝ) : ℝ := 
  (AB + BC + CD + DE + AE) * cost_per_meter

theorem cost_of_fencing_irregular_pentagon :
  total_cost_fencing 20 25 30 35 40 2 = 300 := 
by
  sorry

end cost_of_fencing_irregular_pentagon_l147_147868


namespace fabric_needed_for_coats_l147_147817

variable (m d : ℝ)

def condition1 := 4 * m + 2 * d = 16
def condition2 := 2 * m + 6 * d = 18

theorem fabric_needed_for_coats (h1 : condition1 m d) (h2 : condition2 m d) :
  m = 3 ∧ d = 2 :=
by
  sorry

end fabric_needed_for_coats_l147_147817


namespace suraj_innings_count_l147_147236

theorem suraj_innings_count
  (A : ℕ := 24)  -- average before the last innings
  (new_average : ℕ := 28)  -- Suraj’s average after the last innings
  (last_score : ℕ := 92)  -- Suraj’s score in the last innings
  (avg_increase : ℕ := 4)  -- the increase in average after the last innings
  (n : ℕ)  -- number of innings before the last one
  (h_avg : A + avg_increase = new_average)  -- A + 4 = 28
  (h_eqn : n * A + last_score = (n + 1) * new_average) :  -- n * 24 + 92 = (n + 1) * 28
  n = 16 :=
by {
  sorry
}

end suraj_innings_count_l147_147236


namespace minimum_positive_Sn_l147_147643

theorem minimum_positive_Sn (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ n, a (n+1) = a n + d) →
  a 11 / a 10 < -1 →
  (∃ N, ∀ n > N, S n < S (n + 1) ∧ S 1 ≤ S n ∧ ∀ n > N, S n < 0) →
  S 19 > 0 ∧ ∀ k < 19, S k > S 19 → S 19 < 0 →
  n = 19 :=
by
  sorry

end minimum_positive_Sn_l147_147643


namespace num_integers_satisfying_inequality_l147_147229

theorem num_integers_satisfying_inequality :
  ∃ (x : ℕ), ∀ (y: ℤ), (-3 ≤ 3 * y + 2 → 3 * y + 2 ≤ 8) ↔ 4 = x :=
by
  sorry

end num_integers_satisfying_inequality_l147_147229


namespace a_is_zero_l147_147845

theorem a_is_zero (a b : ℤ)
  (h : ∀ n : ℕ, ∃ x : ℤ, a * 2013^n + b = x^2) : a = 0 :=
by
  sorry

end a_is_zero_l147_147845


namespace find_parallel_lines_a_l147_147105

/--
Given two lines \(l_1\): \(x + 2y - 3 = 0\) and \(l_2\): \(2x - ay + 3 = 0\),
prove that if the lines are parallel, then \(a = -4\).
-/
theorem find_parallel_lines_a (a : ℝ) :
  (∀ (x y : ℝ), x + 2*y - 3 = 0) 
  → (∀ (x y : ℝ), 2*x - a*y + 3 = 0)
  → (-1 / 2 = 2 / -a) 
  → a = -4 :=
by
  intros
  sorry

end find_parallel_lines_a_l147_147105


namespace Lorelai_jellybeans_correct_l147_147071

def Gigi_jellybeans : ℕ := 15
def Rory_jellybeans : ℕ := Gigi_jellybeans + 30
def Total_jellybeans : ℕ := Rory_jellybeans + Gigi_jellybeans
def Lorelai_jellybeans : ℕ := 3 * Total_jellybeans

theorem Lorelai_jellybeans_correct : Lorelai_jellybeans = 180 := by
  sorry

end Lorelai_jellybeans_correct_l147_147071


namespace equivalent_equation_l147_147054

theorem equivalent_equation (x y : ℝ) 
  (x_ne_0 : x ≠ 0) (x_ne_3 : x ≠ 3) 
  (y_ne_0 : y ≠ 0) (y_ne_5 : y ≠ 5)
  (main_equation : (3 / x) + (4 / y) = 1 / 3) : 
  x = 9 * y / (y - 12) :=
sorry

end equivalent_equation_l147_147054


namespace worth_of_used_car_l147_147720

theorem worth_of_used_car (earnings remaining : ℝ) (earnings_eq : earnings = 5000) (remaining_eq : remaining = 1000) : 
  ∃ worth : ℝ, worth = earnings - remaining ∧ worth = 4000 :=
by
  sorry

end worth_of_used_car_l147_147720


namespace nine_y_squared_eq_x_squared_z_squared_l147_147333

theorem nine_y_squared_eq_x_squared_z_squared (x y z : ℝ) (h : x / y = 3 / z) : 9 * y ^ 2 = x ^ 2 * z ^ 2 :=
by
  sorry

end nine_y_squared_eq_x_squared_z_squared_l147_147333


namespace area_of_given_triangle_l147_147586

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

theorem area_of_given_triangle :
  area_of_triangle (0, 0) (4, 0) (4, 6) = 12.0 :=
by 
  sorry

end area_of_given_triangle_l147_147586


namespace g_60_l147_147382

noncomputable def g : ℝ → ℝ :=
sorry

axiom g_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y

axiom g_45 : g 45 = 15

theorem g_60 : g 60 = 11.25 :=
by
  sorry

end g_60_l147_147382


namespace medical_team_formation_l147_147259

theorem medical_team_formation (m f : ℕ) (h_m : m = 5) (h_f : f = 4) :
  (m + f).choose 3 - m.choose 3 - f.choose 3 = 70 :=
by
  sorry

end medical_team_formation_l147_147259


namespace parabola_equation_l147_147708

theorem parabola_equation (p : ℝ) (h_pos : p > 0) (M : ℝ) (h_Mx : M = 3) (h_MF : abs (M + p/2) = 2 * p) :
  (forall x y, y^2 = 2 * p * x) -> (forall x y, y^2 = 4 * x) :=
by
  sorry

end parabola_equation_l147_147708


namespace arithmetic_sequence_a3_is_8_l147_147523

-- Define the arithmetic sequence
def arithmetic_sequence (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Theorem to prove a3 = 8 given a1 = 4 and d = 2
theorem arithmetic_sequence_a3_is_8 (a1 d : ℕ) (h1 : a1 = 4) (h2 : d = 2) : arithmetic_sequence a1 d 3 = 8 :=
by
  sorry -- Proof not required as per instruction

end arithmetic_sequence_a3_is_8_l147_147523


namespace no_solution_fractions_eq_l147_147418

open Real

theorem no_solution_fractions_eq (x : ℝ) :
  (x-2)/(2*x-1) + 1 = 3/(2-4*x) → False :=
by
  intro h
  have h1 : ¬ (2*x - 1 = 0) := by
    -- 2*x - 1 ≠ 0
    sorry
  have h2 : ¬ (2 - 4*x = 0) := by
    -- 2 - 4*x ≠ 0
    sorry
  -- Solve the equation and show no solutions exist without contradicting the conditions
  sorry

end no_solution_fractions_eq_l147_147418


namespace number_of_tens_in_sum_l147_147035

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l147_147035


namespace power_function_value_l147_147997

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_value (α : ℝ) (h : 2 ^ α = (Real.sqrt 2) / 2) : f 4 α = 1 / 2 := 
by 
  sorry

end power_function_value_l147_147997


namespace max_geometric_sequence_sum_l147_147238

theorem max_geometric_sequence_sum (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a * b * c = 216) (h4 : ∃ r : ℕ, b = a * r ∧ c = b * r) : 
  a + b + c ≤ 43 :=
sorry

end max_geometric_sequence_sum_l147_147238


namespace total_prizes_l147_147818

-- Definitions of the conditions
def stuffedAnimals : ℕ := 14
def frisbees : ℕ := 18
def yoYos : ℕ := 18

-- The statement to be proved
theorem total_prizes : stuffedAnimals + frisbees + yoYos = 50 := by
  sorry

end total_prizes_l147_147818


namespace cuboidal_box_area_l147_147546

/-- Given conditions about a cuboidal box:
    - The area of one face is 72 cm²
    - The area of an adjacent face is 60 cm²
    - The volume of the cuboidal box is 720 cm³,
    Prove that the area of the third adjacent face is 120 cm². -/
theorem cuboidal_box_area (l w h : ℝ) (h1 : l * w = 72) (h2 : w * h = 60) (h3 : l * w * h = 720) :
  l * h = 120 :=
sorry

end cuboidal_box_area_l147_147546


namespace maximum_value_l147_147016

theorem maximum_value (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
    (h_eq : a^2 * (b + c - a) = b^2 * (a + c - b) ∧ b^2 * (a + c - b) = c^2 * (b + a - c)) :
    (2 * b + 3 * c) / a = 5 := 
sorry

end maximum_value_l147_147016


namespace find_g_inv_l147_147953

noncomputable def g (x : ℝ) : ℝ :=
  (x^7 - 1) / 4

noncomputable def g_inv_value : ℝ :=
  (51 / 32)^(1/7)

theorem find_g_inv (h : g (g_inv_value) = 19 / 128) : g_inv_value = (51 / 32)^(1/7) :=
by
  sorry

end find_g_inv_l147_147953


namespace georgie_ghost_enter_exit_diff_window_l147_147983

theorem georgie_ghost_enter_exit_diff_window (n : ℕ) (h : n = 8) :
    (∃ enter exit, enter ≠ exit ∧ 1 ≤ enter ∧ enter ≤ n ∧ 1 ≤ exit ∧ exit ≤ n) ∧
    (∃ W : ℕ, W = (n * (n - 1))) :=
sorry

end georgie_ghost_enter_exit_diff_window_l147_147983


namespace M_eq_N_l147_147824

def M : Set ℤ := { u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l }
def N : Set ℤ := { u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r }

theorem M_eq_N : M = N := by
  sorry

end M_eq_N_l147_147824


namespace minimum_w_coincide_after_translation_l147_147410

noncomputable def period_of_cosine (w : ℝ) : ℝ := (2 * Real.pi) / w

theorem minimum_w_coincide_after_translation
  (w : ℝ) (h_w_pos : 0 < w) :
  period_of_cosine w = (4 * Real.pi) / 3 → w = 3 / 2 :=
by
  sorry

end minimum_w_coincide_after_translation_l147_147410


namespace average_speed_distance_div_time_l147_147561

theorem average_speed_distance_div_time (distance : ℕ) (time_minutes : ℕ) (average_speed : ℕ) : 
  distance = 8640 → time_minutes = 36 → average_speed = distance / (time_minutes * 60) → average_speed = 4 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  assumption

end average_speed_distance_div_time_l147_147561


namespace total_packs_of_groceries_is_14_l147_147038

-- Define the number of packs of cookies
def packs_of_cookies : Nat := 2

-- Define the number of packs of cakes
def packs_of_cakes : Nat := 12

-- Define the total packs of groceries as the sum of packs of cookies and cakes
def total_packs_of_groceries : Nat := packs_of_cookies + packs_of_cakes

-- The theorem which states that the total packs of groceries is 14
theorem total_packs_of_groceries_is_14 : total_packs_of_groceries = 14 := by
  -- this is where the proof would go
  sorry

end total_packs_of_groceries_is_14_l147_147038


namespace problem_rect_ratio_l147_147937

theorem problem_rect_ratio (W X Y Z U V R S : ℝ × ℝ) 
  (hYZ : Y = (0, 0))
  (hW : W = (0, 6))
  (hZ : Z = (7, 6))
  (hX : X = (7, 4))
  (hU : U = (5, 0))
  (hV : V = (4, 4))
  (hR : R = (5 / 3, 4))
  (hS : S = (0, 4))
  : (dist R S) / (dist X V) = 5 / 9 := 
sorry

end problem_rect_ratio_l147_147937


namespace all_three_white_probability_l147_147414

noncomputable def box_probability : ℚ :=
  let total_white := 4
  let total_black := 7
  let total_balls := total_white + total_black
  let draw_count := 3
  let total_combinations := (total_balls.choose draw_count : ℕ)
  let favorable_combinations := (total_white.choose draw_count : ℕ)
  (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem all_three_white_probability :
  box_probability = 4 / 165 :=
by
  sorry

end all_three_white_probability_l147_147414


namespace five_natural_numbers_increase_15_times_l147_147980

noncomputable def prod_of_decreased_factors_is_15_times_original (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  (a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * (a1 * a2 * a3 * a4 * a5)

theorem five_natural_numbers_increase_15_times {a1 a2 a3 a4 a5 : ℕ} :
  a1 * a2 * a3 * a4 * a5 = 48 → prod_of_decreased_factors_is_15_times_original a1 a2 a3 a4 a5 :=
by
  sorry

end five_natural_numbers_increase_15_times_l147_147980


namespace log_inequality_l147_147931

open Real

theorem log_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  log (1 + sqrt (a * b)) ≤ (1 / 2) * (log (1 + a) + log (1 + b)) :=
sorry

end log_inequality_l147_147931


namespace curve_is_line_l147_147513

theorem curve_is_line : ∀ (r θ : ℝ), r = 2 / (2 * Real.sin θ - Real.cos θ) → ∃ m b, ∀ (x y : ℝ), x = r * Real.cos θ → y = r * Real.sin θ → y = m * x + b :=
by
  intros r θ h
  sorry

end curve_is_line_l147_147513


namespace extreme_value_when_a_is_neg_one_range_of_a_for_f_non_positive_l147_147373

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x * exp x - (x + 1) ^ 2

-- Question 1: Extreme value when a = -1
theorem extreme_value_when_a_is_neg_one : 
  f (-1) (-1) = 1 / exp 1 := sorry

-- Question 2: Range of a such that ∀ x ∈ [-1, 1], f(x) ≤ 0
theorem range_of_a_for_f_non_positive :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 4 / exp 1 := sorry

end extreme_value_when_a_is_neg_one_range_of_a_for_f_non_positive_l147_147373


namespace line_form_x_eq_ky_add_b_perpendicular_y_line_form_x_eq_ky_add_b_perpendicular_x_l147_147877

theorem line_form_x_eq_ky_add_b_perpendicular_y {k b : ℝ} : 
  ¬ ∃ c : ℝ, x = c ∧ ∀ y : ℝ, x = k*y + b :=
sorry

theorem line_form_x_eq_ky_add_b_perpendicular_x {b : ℝ} : 
  ∃ k : ℝ, k = 0 ∧ ∀ y : ℝ, x = k*y + b :=
sorry

end line_form_x_eq_ky_add_b_perpendicular_y_line_form_x_eq_ky_add_b_perpendicular_x_l147_147877


namespace find_n_l147_147496

theorem find_n (n : ℤ) (hn : -180 ≤ n ∧ n ≤ 180) (hsin : Real.sin (n * Real.pi / 180) = Real.sin (750 * Real.pi / 180)) :
  n = 30 ∨ n = 150 ∨ n = -30 ∨ n = -150 :=
by
  sorry

end find_n_l147_147496


namespace similar_triangle_side_length_l147_147249

theorem similar_triangle_side_length
  (A_1 A_2 : ℕ)
  (area_diff : A_1 - A_2 = 32)
  (area_ratio : A_1 = 9 * A_2)
  (side_small_triangle : ℕ)
  (side_small_triangle_eq : side_small_triangle = 5)
  (side_ratio : ∃ r : ℕ, r = 3) :
  ∃ side_large_triangle : ℕ, side_large_triangle = side_small_triangle * 3 := by
sorry

end similar_triangle_side_length_l147_147249


namespace min_cost_theater_tickets_l147_147541

open Real

variable (x y : ℝ)

theorem min_cost_theater_tickets :
  (x + y = 140) →
  (y ≥ 2 * x) →
  ∀ x y, 60 * x + 100 * y ≥ 12160 :=
by
  sorry

end min_cost_theater_tickets_l147_147541


namespace total_white_roses_l147_147974

-- Define the constants
def n_b : ℕ := 5
def n_t : ℕ := 7
def r_b : ℕ := 5
def r_t : ℕ := 12

-- State the theorem
theorem total_white_roses :
  n_t * r_t + n_b * r_b = 109 :=
by
  -- Automatic proof can be here; using sorry as placeholder
  sorry

end total_white_roses_l147_147974


namespace operation_on_b_l147_147268

variables (t b b' : ℝ)
variable (C : ℝ := t * b ^ 4)
variable (e : ℝ := 16 * C)

theorem operation_on_b :
  tb'^4 = 16 * tb^4 → b' = 2 * b := by
  sorry

end operation_on_b_l147_147268


namespace subletter_payment_correct_l147_147040

noncomputable def johns_monthly_rent : ℕ := 900
noncomputable def johns_yearly_rent : ℕ := johns_monthly_rent * 12
noncomputable def johns_profit_per_year : ℕ := 3600
noncomputable def total_rent_collected : ℕ := johns_yearly_rent + johns_profit_per_year
noncomputable def number_of_subletters : ℕ := 3
noncomputable def subletter_annual_payment : ℕ := total_rent_collected / number_of_subletters
noncomputable def subletter_monthly_payment : ℕ := subletter_annual_payment / 12

theorem subletter_payment_correct :
  subletter_monthly_payment = 400 :=
by
  sorry

end subletter_payment_correct_l147_147040


namespace evaluate_expression_l147_147582

theorem evaluate_expression : 
  |-2| + (1 / 4) - 1 - 4 * Real.cos (Real.pi / 4) + Real.sqrt 8 = 5 / 4 :=
by
  sorry

end evaluate_expression_l147_147582


namespace line_through_A1_slope_neg4_over_3_line_through_A2_l147_147482

-- (1) The line passing through point (1, 3) with a slope -4/3
theorem line_through_A1_slope_neg4_over_3 : 
    ∃ (a b c : ℝ), a * 1 + b * 3 + c = 0 ∧ ∃ m : ℝ, m = -4 / 3 ∧ a * m + b = 0 ∧ b ≠ 0 ∧ c = -13 := by
sorry

-- (2) The line passing through point (-5, 2) with x-intercept twice the y-intercept
theorem line_through_A2 : 
    ∃ (a b c : ℝ), (a * -5 + b * 2 + c = 0) ∧ ((∃ m : ℝ, m = 2 ∧ a * m + b = 0 ∧ b = -a) ∨ ((b = -2 / 5 * a) ∧ (a * 2 + b = 0))) := by
sorry

end line_through_A1_slope_neg4_over_3_line_through_A2_l147_147482


namespace Michelangelo_ceiling_painting_l147_147514

theorem Michelangelo_ceiling_painting (C : ℕ) : 
  ∃ C, (C + (1/4) * C = 15) ∧ (28 - (C + (1/4) * C) = 13) :=
sorry

end Michelangelo_ceiling_painting_l147_147514


namespace student_range_exact_student_count_l147_147058

-- Definitions for the conditions
def retail_price (x : ℕ) : ℕ := 240
def wholesale_price (x : ℕ) : ℕ := 260 / (x + 60)

def student_conditions (x : ℕ) : Prop := (x < 250) ∧ (x + 60 ≥ 250)
def wholesale_retail_equation (a : ℕ) : Prop := (240^2 / a) * 240 = (260 / (a+60)) * 288

-- Proofs of the required statements
theorem student_range (x : ℕ) (hc : student_conditions x) : 190 ≤ x ∧ x < 250 :=
by {
  sorry
}

theorem exact_student_count (a : ℕ) (heq : wholesale_retail_equation a) : a = 200 :=
by {
  sorry
}

end student_range_exact_student_count_l147_147058


namespace min_abs_A_l147_147956

def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

def A (a d : ℚ) (n : ℕ) : ℚ :=
  (arithmetic_sequence a d n) + (arithmetic_sequence a d (n + 1)) + 
  (arithmetic_sequence a d (n + 2)) + (arithmetic_sequence a d (n + 3)) + 
  (arithmetic_sequence a d (n + 4)) + (arithmetic_sequence a d (n + 5)) + 
  (arithmetic_sequence a d (n + 6))

theorem min_abs_A : (arithmetic_sequence 19 (-4/5) 26 = -1) ∧ 
                    (∀ n, 1 ≤ n) →
                    ∃ n : ℕ, |A 19 (-4/5) n| = 7/5 :=
by
  sorry

end min_abs_A_l147_147956


namespace radius_correct_l147_147060

noncomputable def radius_of_circle (chord_length tang_secant_segment : ℝ) : ℝ :=
  let r := 6.25
  r

theorem radius_correct
  (chord_length : ℝ)
  (tangent_secant_segment : ℝ)
  (parallel_secant_internal_segment : ℝ)
  : chord_length = 10 ∧ parallel_secant_internal_segment = 12 → radius_of_circle chord_length parallel_secant_internal_segment = 6.25 :=
by
  intros h
  sorry

end radius_correct_l147_147060


namespace interest_rate_increase_l147_147383

-- Define the conditions
def principal (P : ℕ) := P = 1000
def time (t : ℕ) := t = 5
def original_amount (A : ℕ) := A = 1500
def new_amount (A' : ℕ) := A' = 1750

-- Prove that the interest rate increase is 50%
theorem interest_rate_increase
  (P : ℕ) (t : ℕ) (A A' : ℕ)
  (hP : principal P)
  (ht : time t)
  (hA : original_amount A)
  (hA' : new_amount A') :
  (((((A' - P) / (P * t)) - ((A - P) / (P * t))) / ((A - P) / (P * t))) * 100) = 50 := by
  sorry

end interest_rate_increase_l147_147383


namespace find_amount_l147_147045

-- Definitions based on the conditions provided
def gain : ℝ := 0.70
def gain_percent : ℝ := 1.0

-- The theorem statement
theorem find_amount (h : gain_percent = 1) : ∀ (amount : ℝ), amount = gain / (gain_percent / 100) → amount = 70 :=
by
  intros amount h_calc
  sorry

end find_amount_l147_147045


namespace operation_result_l147_147853

-- Define the new operation x # y
def op (x y : ℕ) : ℤ := 2 * x * y - 3 * x + y

-- Prove that (6 # 4) - (4 # 6) = -8
theorem operation_result : op 6 4 - op 4 6 = -8 :=
by
  sorry

end operation_result_l147_147853


namespace a1_a2_a3_sum_l147_147329

-- Given conditions and hypothesis
variables (a0 a1 a2 a3 : ℝ)
axiom H : ∀ x : ℝ, 1 + x + x^2 + x^3 = a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3

-- Goal statement to be proven
theorem a1_a2_a3_sum : a1 + a2 + a3 = -3 :=
sorry

end a1_a2_a3_sum_l147_147329


namespace desktops_to_sell_l147_147573

theorem desktops_to_sell (laptops desktops : ℕ) (ratio_laptops desktops_sold laptops_expected : ℕ) :
  ratio_laptops = 5 → desktops_sold = 3 → laptops_expected = 40 → 
  desktops = (desktops_sold * laptops_expected) / ratio_laptops :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry -- This is where the proof would go, but it's not needed for this task

end desktops_to_sell_l147_147573


namespace solve_for_y_l147_147796

theorem solve_for_y (y : ℝ) (h : 7 - y = 12) : y = -5 := sorry

end solve_for_y_l147_147796


namespace go_games_l147_147999

theorem go_games (total_go_balls : ℕ) (go_balls_per_game : ℕ) (h_total : total_go_balls = 901) (h_game : go_balls_per_game = 53) : (total_go_balls / go_balls_per_game) = 17 := by
  sorry

end go_games_l147_147999


namespace intersection_one_point_l147_147751

open Set

def A (x y : ℝ) : Prop := x^2 - 3*x*y + 4*y^2 = 7 / 2
def B (k x y : ℝ) : Prop := k > 0 ∧ k*x + y = 2

theorem intersection_one_point (k : ℝ) (h : k > 0) :
  (∃ x y : ℝ, A x y ∧ B k x y) → (∀ x₁ y₁ x₂ y₂ : ℝ, (A x₁ y₁ ∧ B k x₁ y₁) ∧ (A x₂ y₂ ∧ B k x₂ y₂) → x₁ = x₂ ∧ y₁ = y₂) ↔ k = 1 / 4 :=
sorry

end intersection_one_point_l147_147751


namespace smallest_number_l147_147356

theorem smallest_number
  (A : ℕ := 2^3 + 2^2 + 2^1 + 2^0)
  (B : ℕ := 2 * 6^2 + 1 * 6)
  (C : ℕ := 1 * 4^3)
  (D : ℕ := 8 + 1) :
  A < B ∧ A < C ∧ A < D :=
by {
  sorry
}

end smallest_number_l147_147356


namespace twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number_l147_147747

theorem twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number :
  ∃ n : ℝ, (80 - 0.25 * 80) = (5 / 4) * n ∧ n = 48 := 
by
  sorry

end twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number_l147_147747


namespace tina_more_than_katya_l147_147648

-- Define the number of glasses sold by Katya, Ricky, and the condition for Tina's sales
def katya_sales : ℕ := 8
def ricky_sales : ℕ := 9

def combined_sales : ℕ := katya_sales + ricky_sales
def tina_sales : ℕ := 2 * combined_sales

-- Define the theorem to prove that Tina sold 26 more glasses than Katya
theorem tina_more_than_katya : tina_sales = katya_sales + 26 := by
  sorry

end tina_more_than_katya_l147_147648


namespace train_speed_kmph_l147_147601

theorem train_speed_kmph (len_train : ℝ) (len_platform : ℝ) (time_cross : ℝ) (total_distance : ℝ) (speed_mps : ℝ) (speed_kmph : ℝ) 
  (h1 : len_train = 250) 
  (h2 : len_platform = 150.03) 
  (h3 : time_cross = 20) 
  (h4 : total_distance = len_train + len_platform) 
  (h5 : speed_mps = total_distance / time_cross) 
  (h6 : speed_kmph = speed_mps * 3.6) : 
  speed_kmph = 72.0054 := 
by 
  -- This is where the proof would go
  sorry

end train_speed_kmph_l147_147601


namespace extreme_value_a_one_range_of_a_l147_147724

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + 3

theorem extreme_value_a_one :
  ∀ x > 0, f x 1 ≤ f 1 1 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≤ 0) → a ≥ Real.exp 2 :=
sorry

end extreme_value_a_one_range_of_a_l147_147724


namespace tan_sin_equality_l147_147595

theorem tan_sin_equality :
  (Real.tan (30 * Real.pi / 180))^2 + (Real.sin (45 * Real.pi / 180))^2 = 5 / 6 :=
by sorry

end tan_sin_equality_l147_147595


namespace milburg_children_count_l147_147213

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 → grown_ups = 5256 → 
  (total_population - grown_ups) = 2987 :=
by
  intros total_population grown_ups h1 h2
  sorry

end milburg_children_count_l147_147213


namespace geometric_sequence_a12_l147_147731

noncomputable def a_n (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r ^ (n - 1)

theorem geometric_sequence_a12 (a1 r : ℝ) 
  (h1 : a_n a1 r 7 * a_n a1 r 9 = 4)
  (h2 : a_n a1 r 4 = 1) :
  a_n a1 r 12 = 16 := sorry

end geometric_sequence_a12_l147_147731


namespace solve_linear_system_l147_147911

theorem solve_linear_system (m x y : ℝ) 
  (h1 : x + y = 3 * m) 
  (h2 : x - y = 5 * m)
  (h3 : 2 * x + 3 * y = 10) : 
  m = 2 := 
by 
  sorry

end solve_linear_system_l147_147911


namespace find_m_l147_147887

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}

def isArithmeticSeq (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a_n (n + 1) = a_n n + d

def sumSeq (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
∀ n, S n = (n * (a_n 1 + a_n n)) / 2

theorem find_m
  (d : ℤ)
  (a_1 : ℤ)
  (a_n : ∀ n, ℤ)
  (S : ℕ → ℤ)
  (h_arith : isArithmeticSeq a_n d)
  (h_sum : sumSeq S a_n)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end find_m_l147_147887


namespace fraction_to_terminating_decimal_l147_147368

theorem fraction_to_terminating_decimal : (21 : ℚ) / 40 = 0.525 := 
by
  sorry

end fraction_to_terminating_decimal_l147_147368


namespace slope_of_arithmetic_sequence_l147_147245

variable {α : Type*} [LinearOrderedField α]

noncomputable def S (a_1 d n : α) : α := n * a_1 + n * (n-1) / 2 * d

theorem slope_of_arithmetic_sequence (a_1 d n : α) 
  (hS2 : S a_1 d 2 = 10)
  (hS5 : S a_1 d 5 = 55)
  : (a_1 + 2 * d - a_1) / 2 = 4 :=
by
  sorry

end slope_of_arithmetic_sequence_l147_147245


namespace non_zero_real_solution_l147_147851

theorem non_zero_real_solution (x : ℝ) (hx : x ≠ 0) (h : (3 * x)^5 = (9 * x)^4) : x = 27 :=
sorry

end non_zero_real_solution_l147_147851


namespace circumference_of_cone_l147_147692

theorem circumference_of_cone (V : ℝ) (h : ℝ) (C : ℝ) 
  (hV : V = 36 * Real.pi) (hh : h = 3) : 
  C = 12 * Real.pi :=
sorry

end circumference_of_cone_l147_147692


namespace intersection_of_A_and_complement_B_l147_147792

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x | x < 3}
def complement_B : Set ℝ := {x | x ≥ 3}

theorem intersection_of_A_and_complement_B : A ∩ complement_B = {3, 4, 5} :=
by
  sorry

end intersection_of_A_and_complement_B_l147_147792


namespace total_spent_l147_147180

def price_almond_croissant : ℝ := 4.50
def price_salami_cheese_croissant : ℝ := 4.50
def price_plain_croissant : ℝ := 3.00
def price_focaccia : ℝ := 4.00
def price_latte : ℝ := 2.50
def num_lattes : ℕ := 2

theorem total_spent :
  price_almond_croissant + price_salami_cheese_croissant + price_plain_croissant +
  price_focaccia + (num_lattes * price_latte) = 21.00 := by
  sorry

end total_spent_l147_147180


namespace misha_students_count_l147_147649

theorem misha_students_count :
  (∀ n : ℕ, n = 60 → (exists better worse : ℕ, better = n - 1 ∧  worse = n - 1)) →
  (∀ n : ℕ, n = 60 → (better + worse + 1 = 119)) :=
by
  sorry

end misha_students_count_l147_147649


namespace curve_intersections_l147_147219

theorem curve_intersections (m : ℝ) :
  (∃ x y : ℝ, ((x-1)^2 + y^2 = 1) ∧ (y = mx + m) ∧ (y ≠ 0) ∧ (y^2 = 0)) =
  ((m > -Real.sqrt 3 / 3) ∧ (m < 0)) ∨ ((m > 0) ∧ (m < Real.sqrt 3 / 3)) := 
sorry

end curve_intersections_l147_147219


namespace pigeons_among_non_sparrows_l147_147628

theorem pigeons_among_non_sparrows (P_total P_parrots P_peacocks P_sparrows : ℝ)
    (h1 : P_total = 20)
    (h2 : P_parrots = 30)
    (h3 : P_peacocks = 15)
    (h4 : P_sparrows = 35) :
    (P_total / (100 - P_sparrows)) * 100 = 30.77 :=
by
  -- Proof will be provided here
  sorry

end pigeons_among_non_sparrows_l147_147628


namespace find_denominator_l147_147098

noncomputable def original_denominator (d : ℝ) : Prop :=
  (7 / (d + 3)) = 2 / 3

theorem find_denominator : ∃ d : ℝ, original_denominator d ∧ d = 7.5 :=
by
  use 7.5
  unfold original_denominator
  sorry

end find_denominator_l147_147098


namespace largest_four_digit_divisible_by_14_l147_147149

theorem largest_four_digit_divisible_by_14 :
  ∃ (A : ℕ), A = 9898 ∧ 
  (∃ a b : ℕ, A = 1010 * a + 101 * b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) ∧
  (A % 14 = 0) ∧
  (A = (d1 * 100 + d2 * 10 + d1) * 101)
  :=
sorry

end largest_four_digit_divisible_by_14_l147_147149


namespace inequality_proof_l147_147926

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2 / 3 :=
sorry

end inequality_proof_l147_147926


namespace eval_expr_correct_l147_147664

noncomputable def eval_expr : ℝ :=
  let a := (12:ℝ)^5 * (6:ℝ)^4
  let b := (3:ℝ)^2 * (36:ℝ)^2
  let c := Real.sqrt 9 * Real.log (27:ℝ)
  (a / b) + c

theorem eval_expr_correct : eval_expr = 27657.887510597983 := by
  sorry

end eval_expr_correct_l147_147664


namespace candidate_lost_by_2340_votes_l147_147635

theorem candidate_lost_by_2340_votes
  (total_votes : ℝ)
  (candidate_percentage : ℝ)
  (rival_percentage : ℝ)
  (candidate_votes : ℝ)
  (rival_votes : ℝ)
  (votes_difference : ℝ)
  (h1 : total_votes = 7800)
  (h2 : candidate_percentage = 0.35)
  (h3 : rival_percentage = 0.65)
  (h4 : candidate_votes = candidate_percentage * total_votes)
  (h5 : rival_votes = rival_percentage * total_votes)
  (h6 : votes_difference = rival_votes - candidate_votes) :
  votes_difference = 2340 :=
by
  sorry

end candidate_lost_by_2340_votes_l147_147635


namespace fill_tank_time_is_18_l147_147469

def rate1 := 1 / 20
def rate2 := 1 / 30
def combined_rate := rate1 + rate2
def effective_rate := (2 / 3) * combined_rate
def T := 1 / effective_rate

theorem fill_tank_time_is_18 : T = 18 := by
  sorry

end fill_tank_time_is_18_l147_147469


namespace total_pens_bought_l147_147350

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l147_147350


namespace complement_of_P_l147_147773

def U : Set ℤ := {-1, 0, 1, 2}
def P : Set ℤ := {x | x^2 < 2}

theorem complement_of_P :
  (U \ P) = {2} :=
by
  sorry

end complement_of_P_l147_147773


namespace correct_option_is_B_l147_147949

variable (f : ℝ → ℝ)
variable (h0 : f 0 = 2)
variable (h1 : ∀ x : ℝ, deriv f x > f x + 1)

theorem correct_option_is_B : 3 * Real.exp (1 : ℝ) < f 2 + 1 := sorry

end correct_option_is_B_l147_147949


namespace bhanu_house_rent_expenditure_l147_147233

variable (Income house_rent_expenditure petrol_expenditure remaining_income : ℝ)
variable (h1 : petrol_expenditure = (30 / 100) * Income)
variable (h2 : remaining_income = Income - petrol_expenditure)
variable (h3 : house_rent_expenditure = (20 / 100) * remaining_income)
variable (h4 : petrol_expenditure = 300)

theorem bhanu_house_rent_expenditure :
  house_rent_expenditure = 140 :=
by sorry

end bhanu_house_rent_expenditure_l147_147233


namespace machines_needed_l147_147548

variables (R x m N : ℕ) (h1 : 4 * R * 6 = x)
           (h2 : N * R * 6 = m * x)

theorem machines_needed : N = m * 4 :=
by sorry

end machines_needed_l147_147548


namespace sector_area_sexagesimal_l147_147463

theorem sector_area_sexagesimal (r : ℝ) (n : ℝ) (α_sex : ℝ) (π : ℝ) (two_pi : ℝ):
  r = 4 →
  n = 6000 →
  α_sex = 625 →
  two_pi = 2 * π →
  (1/2 * (α_sex / n * two_pi) * r^2) = (5 * π) / 3 :=
by
  intros
  sorry

end sector_area_sexagesimal_l147_147463


namespace checkered_rectangles_containing_one_gray_cell_l147_147083

def total_number_of_rectangles_with_one_gray_cell :=
  let gray_cells := 40
  let blue_cells := 36
  let red_cells := 4
  
  let blue_rectangles_each := 4
  let red_rectangles_each := 8
  
  (blue_cells * blue_rectangles_each) + (red_cells * red_rectangles_each)

theorem checkered_rectangles_containing_one_gray_cell : total_number_of_rectangles_with_one_gray_cell = 176 :=
by 
  sorry

end checkered_rectangles_containing_one_gray_cell_l147_147083


namespace find_values_of_a_l147_147441

noncomputable def has_one_real_solution (a : ℝ) : Prop :=
  ∃ x: ℝ, (x^3 - a*x^2 - 3*a*x + a^2 - 1 = 0) ∧ (∀ y: ℝ, (y^3 - a*y^2 - 3*a*y + a^2 - 1 = 0) → y = x)

theorem find_values_of_a : ∀ a: ℝ, has_one_real_solution a ↔ a < -(5 / 4) :=
by
  sorry

end find_values_of_a_l147_147441


namespace determine_k_l147_147498

theorem determine_k 
  (k : ℝ) 
  (r s : ℝ) 
  (h1 : r + s = -k) 
  (h2 : r * s = 6) 
  (h3 : (r + 5) + (s + 5) = k) : 
  k = 5 := 
by 
  sorry

end determine_k_l147_147498


namespace tickets_used_63_l147_147990

def rides_ferris_wheel : ℕ := 5
def rides_bumper_cars : ℕ := 4
def cost_per_ride : ℕ := 7
def total_rides : ℕ := rides_ferris_wheel + rides_bumper_cars
def total_tickets_used : ℕ := total_rides * cost_per_ride

theorem tickets_used_63 : total_tickets_used = 63 := by
  unfold total_tickets_used
  unfold total_rides
  unfold rides_ferris_wheel
  unfold rides_bumper_cars
  unfold cost_per_ride
  -- proof goes here
  sorry

end tickets_used_63_l147_147990


namespace percentage_passed_l147_147838

def swim_club_members := 100
def not_passed_course_taken := 40
def not_passed_course_not_taken := 30
def not_passed := not_passed_course_taken + not_passed_course_not_taken

theorem percentage_passed :
  ((swim_club_members - not_passed).toFloat / swim_club_members.toFloat * 100) = 30 := by
  sorry

end percentage_passed_l147_147838


namespace Andrey_Gleb_distance_l147_147404

theorem Andrey_Gleb_distance (AB VG : ℕ) (AG : ℕ) (BV : ℕ) (cond1 : AB = 600) (cond2 : VG = 600) (cond3 : AG = 3 * BV) :
  AG = 900 ∨ AG = 1800 := 
sorry

end Andrey_Gleb_distance_l147_147404


namespace find_f_sqrt_5753_l147_147067

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_sqrt_5753 (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) = x * f y + y * f x)
  (h2 : ∀ x y : ℝ, f (x + y) = f (x * 1993) + f (y * 1993)) :
  f (Real.sqrt 5753) = 0 :=
sorry

end find_f_sqrt_5753_l147_147067


namespace farmer_rewards_l147_147320

theorem farmer_rewards (x y : ℕ) (h1 : x + y = 60) (h2 : 1000 * x + 3000 * y = 100000) : x = 40 ∧ y = 20 :=
by {
  sorry
}

end farmer_rewards_l147_147320


namespace max_value_proof_l147_147590

noncomputable def max_value_b_minus_a (a b : ℝ) : ℝ :=
  b - a

theorem max_value_proof (a b : ℝ) (h1 : a < 0) (h2 : ∀ x, (x^2 + 2017 * a) * (x + 2016 * b) ≥ 0) : max_value_b_minus_a a b ≤ 2017 :=
sorry

end max_value_proof_l147_147590


namespace remaining_wire_in_cm_l147_147622

theorem remaining_wire_in_cm (total_mm : ℝ) (per_mobile_mm : ℝ) (conversion_factor : ℝ) :
  total_mm = 117.6 →
  per_mobile_mm = 4 →
  conversion_factor = 10 →
  ((total_mm % per_mobile_mm) / conversion_factor) = 0.16 :=
by
  intros htotal hmobile hconv
  sorry

end remaining_wire_in_cm_l147_147622


namespace ratio_of_a_to_b_l147_147932

variables (a b x m : ℝ)
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variables (h_x : x = a + 0.25 * a)
variables (h_m : m = b - 0.80 * b)
variables (h_ratio : m / x = 0.2)

theorem ratio_of_a_to_b (h_pos_a : 0 < a) (h_pos_b : 0 < b)
                        (h_x : x = a + 0.25 * a)
                        (h_m : m = b - 0.80 * b)
                        (h_ratio : m / x = 0.2) :
  a / b = 5 / 4 := by
  sorry

end ratio_of_a_to_b_l147_147932


namespace gcd_of_256_180_600_l147_147349

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l147_147349


namespace salt_solution_concentration_l147_147065

theorem salt_solution_concentration (m x : ℝ) (h1 : m > 30) (h2 : (m * m / 100) = ((m - 20) / 100) * (m + 2 * x)) :
  x = 10 * m / (m + 20) :=
sorry

end salt_solution_concentration_l147_147065


namespace amount_subtracted_correct_l147_147156

noncomputable def find_subtracted_amount (N : ℝ) (A : ℝ) : Prop :=
  0.40 * N - A = 23

theorem amount_subtracted_correct :
  find_subtracted_amount 85 11 :=
by
  sorry

end amount_subtracted_correct_l147_147156


namespace cos_of_sin_given_l147_147239

theorem cos_of_sin_given (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by
  sorry

end cos_of_sin_given_l147_147239


namespace sum_of_b_for_one_solution_l147_147316

theorem sum_of_b_for_one_solution (b : ℝ) (has_single_solution : ∃ x, 3 * x^2 + (b + 12) * x + 11 = 0) :
  ∃ b₁ b₂ : ℝ, (3 * x^2 + (b + 12) * x + 11) = 0 ∧ b₁ + b₂ = -24 := by
  sorry

end sum_of_b_for_one_solution_l147_147316


namespace largest_value_l147_147604

theorem largest_value (A B C D E : ℕ)
  (hA : A = (3 + 5 + 2 + 8))
  (hB : B = (3 * 5 + 2 + 8))
  (hC : C = (3 + 5 * 2 + 8))
  (hD : D = (3 + 5 + 2 * 8))
  (hE : E = (3 * 5 * 2 * 8)) :
  max (max (max (max A B) C) D) E = E := 
sorry

end largest_value_l147_147604


namespace lily_patch_cover_entire_lake_l147_147227

noncomputable def days_to_cover_half (initial_days : ℕ) := 33

theorem lily_patch_cover_entire_lake (initial_days : ℕ) (h : days_to_cover_half initial_days = 33) :
  initial_days + 1 = 34 :=
by
  sorry

end lily_patch_cover_entire_lake_l147_147227


namespace C_increases_as_n_increases_l147_147322

theorem C_increases_as_n_increases (e n R r : ℝ) (he : 0 < e) (hn : 0 < n) (hR : 0 < R) (hr : 0 < r) :
  0 < (2 * e * n * R + e * n^2 * r) / (R + n * r)^2 :=
by
  sorry

end C_increases_as_n_increases_l147_147322


namespace time_no_traffic_is_4_hours_l147_147966

-- Definitions and conditions
def distance : ℕ := 200
def time_traffic : ℕ := 5

axiom traffic_speed_relation : ∃ (speed_traffic : ℕ), distance = speed_traffic * time_traffic
axiom speed_difference : ∀ (speed_traffic speed_no_traffic : ℕ), speed_no_traffic = speed_traffic + 10

-- Prove that the time when there's no traffic is 4 hours
theorem time_no_traffic_is_4_hours : ∀ (speed_traffic speed_no_traffic : ℕ), 
  distance = speed_no_traffic * (distance / speed_no_traffic) -> (distance / speed_no_traffic) = 4 :=
by
  intros speed_traffic speed_no_traffic h
  sorry

end time_no_traffic_is_4_hours_l147_147966


namespace problem_statement_l147_147700

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

variable (a x y t : ℝ) 

theorem problem_statement : 
  (log_base a x + 3 * log_base x a - log_base x y = 3) ∧ (a > 1) ∧ (x = a ^ t) ∧ (0 < t ∧ t ≤ 2) ∧ (y = 8) 
  → (a = 16) ∧ (x = 64) := 
by 
  sorry

end problem_statement_l147_147700


namespace geometric_progression_sum_ratio_l147_147988

theorem geometric_progression_sum_ratio (a : ℝ) (r n : ℕ) (hn : r = 3)
  (h : (a * (1 - r^n) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 28) : n = 6 :=
by
  -- Place the steps of the proof here, which are not required as per instructions.
  sorry

end geometric_progression_sum_ratio_l147_147988


namespace sequence_is_arithmetic_l147_147096

-- Define a_n as a sequence in terms of n, where the formula is given.
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Theorem stating that the sequence is arithmetic with a common difference of 2.
theorem sequence_is_arithmetic : ∀ (n : ℕ), n > 0 → (a_n n) - (a_n (n - 1)) = 2 :=
by
  sorry

end sequence_is_arithmetic_l147_147096


namespace eggs_per_basket_l147_147551

theorem eggs_per_basket
  (kids : ℕ)
  (friends : ℕ)
  (adults : ℕ)
  (baskets : ℕ)
  (eggs_per_person : ℕ)
  (htotal : kids + friends + adults + 1 = 20)
  (eggs_total : (kids + friends + adults + 1) * eggs_per_person = 180)
  (baskets_count : baskets = 15)
  : (180 / 15) = 12 :=
by
  sorry

end eggs_per_basket_l147_147551


namespace evaluate_imaginary_expression_l147_147981

theorem evaluate_imaginary_expression (i : ℂ) (h_i2 : i^2 = -1) (h_i4 : i^4 = 1) :
  i^14 + i^19 + i^24 + i^29 + 3 * i^34 + 2 * i^39 = -3 - 2 * i :=
by sorry

end evaluate_imaginary_expression_l147_147981


namespace sum_of_squares_is_289_l147_147050

theorem sum_of_squares_is_289 (x y : ℤ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
by
  sorry

end sum_of_squares_is_289_l147_147050


namespace factorize_expr_l147_147552

theorem factorize_expr (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by
  sorry

end factorize_expr_l147_147552


namespace midpoint_fraction_l147_147401

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (a + b) / 2 = 19/24 := by
  sorry

end midpoint_fraction_l147_147401


namespace arithmetic_sequence_problem_l147_147714

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (d : ℝ)
  (a1 : ℝ)
  (h_arithmetic : ∀ n, a n = a1 + (n - 1) * d)
  (h_a4 : a 4 = 5) :
  2 * a 1 - a 5 + a 11 = 10 := 
by
  sorry

end arithmetic_sequence_problem_l147_147714


namespace Jason_spent_on_music_store_l147_147379

theorem Jason_spent_on_music_store:
  let flute := 142.46
  let music_stand := 8.89
  let song_book := 7.00
  flute + music_stand + song_book = 158.35 := sorry

end Jason_spent_on_music_store_l147_147379


namespace stuffed_animal_cost_is_6_l147_147186

-- Definitions for the costs of items
def sticker_cost (s : ℕ) := s
def magnet_cost (m : ℕ) := m
def stuffed_animal_cost (a : ℕ) := a

-- Conditions given in the problem
def conditions (m s a : ℕ) :=
  (m = 3) ∧
  (m = 3 * s) ∧
  (m = (2 * a) / 4)

-- The theorem stating the cost of a single stuffed animal
theorem stuffed_animal_cost_is_6 (s m a : ℕ) (h : conditions m s a) : a = 6 :=
by
  sorry

end stuffed_animal_cost_is_6_l147_147186


namespace number_of_turns_l147_147302

/-
  Given the cyclist's speed v = 5 m/s, time duration t = 5 s,
  and the circumference of the wheel c = 1.25 m, 
  prove that the number of complete turns n the wheel makes is equal to 20.
-/
theorem number_of_turns (v t c : ℝ) (h_v : v = 5) (h_t : t = 5) (h_c : c = 1.25) : 
  (v * t) / c = 20 :=
by
  sorry

end number_of_turns_l147_147302


namespace population_of_town_l147_147879

theorem population_of_town (F : ℝ) (males : ℕ) (female_glasses : ℝ) (percentage_glasses : ℝ) (total_population : ℝ) 
  (h1 : males = 2000) 
  (h2 : percentage_glasses = 0.30) 
  (h3 : female_glasses = 900) 
  (h4 : percentage_glasses * F = female_glasses) 
  (h5 : total_population = males + F) :
  total_population = 5000 :=
sorry

end population_of_town_l147_147879


namespace value_of_m_l147_147279

theorem value_of_m (a b m : ℝ)
    (h1: 2 ^ a = m)
    (h2: 5 ^ b = m)
    (h3: 1 / a + 1 / b = 1 / 2) :
    m = 100 :=
sorry

end value_of_m_l147_147279


namespace alcohol_to_water_ratio_l147_147642

variable {V p q : ℚ}

def alcohol_volume_jar1 (V p : ℚ) : ℚ := (2 * p) / (2 * p + 3) * V
def water_volume_jar1 (V p : ℚ) : ℚ := 3 / (2 * p + 3) * V
def alcohol_volume_jar2 (V q : ℚ) : ℚ := q / (q + 2) * 2 * V
def water_volume_jar2 (V q : ℚ) : ℚ := 2 / (q + 2) * 2 * V

def total_alcohol_volume (V p q : ℚ) : ℚ :=
  alcohol_volume_jar1 V p + alcohol_volume_jar2 V q

def total_water_volume (V p q : ℚ) : ℚ :=
  water_volume_jar1 V p + water_volume_jar2 V q

theorem alcohol_to_water_ratio (V p q : ℚ) :
  (total_alcohol_volume V p q) / (total_water_volume V p q) = (2 * p + 2 * q) / (3 * p + q + 10) :=
by
  sorry

end alcohol_to_water_ratio_l147_147642


namespace change_in_nickels_l147_147801

theorem change_in_nickels (cost_bread cost_cheese given_amount : ℝ) (quarters dimes : ℕ) (nickel_value : ℝ) 
  (h1 : cost_bread = 4.2) (h2 : cost_cheese = 2.05) (h3 : given_amount = 7.0)
  (h4 : quarters = 1) (h5 : dimes = 1) (hnickel_value : nickel_value = 0.05) : 
  ∃ n : ℕ, n = 8 :=
by
  sorry

end change_in_nickels_l147_147801


namespace minimum_value_l147_147009

noncomputable def minValue (x y : ℝ) : ℝ := (2 / x) + (3 / y)

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 20) : minValue x y = 1 :=
sorry

end minimum_value_l147_147009


namespace garden_perimeter_is_64_l147_147821

theorem garden_perimeter_is_64 :
    ∀ (width_garden length_garden width_playground length_playground : ℕ),
    width_garden = 24 →
    width_playground = 12 →
    length_playground = 16 →
    width_playground * length_playground = width_garden * length_garden →
    2 * length_garden + 2 * width_garden = 64 :=
by
  intros width_garden length_garden width_playground length_playground
  intro h1
  intro h2
  intro h3
  intro h4
  sorry

end garden_perimeter_is_64_l147_147821


namespace find_sum_of_digits_l147_147699

theorem find_sum_of_digits (a c : ℕ) (h1 : 200 + 10 * a + 3 + 427 = 600 + 10 * c + 9) (h2 : (600 + 10 * c + 9) % 3 = 0) : a + c = 4 :=
sorry

end find_sum_of_digits_l147_147699


namespace sides_of_triangle_inequality_l147_147185

theorem sides_of_triangle_inequality {a b c : ℝ} (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2*a^2*b^2 - 2*b^2*c^2 - 2*c^2*a^2 < 0 :=
by
  sorry

end sides_of_triangle_inequality_l147_147185


namespace a_2023_le_1_l147_147959

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_ineq : ∀ n, (a (n+1))^2 + a n * a (n+2) ≤ a n + a (n+2))

theorem a_2023_le_1 : a 2023 ≤ 1 := by
  sorry

end a_2023_le_1_l147_147959


namespace general_equation_M_range_distance_D_to_l_l147_147994

noncomputable def parametric_to_general (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := 2 * Real.sin θ
  x^2 + y^2 / 4 = 1

noncomputable def distance_range (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := 2 * Real.sin θ
  let l := x + y - 4
  let d := |x + 2 * y - 4| / Real.sqrt 2
  let min_dist := (4 * Real.sqrt 2 - Real.sqrt 10) / 2
  let max_dist := (4 * Real.sqrt 2 + Real.sqrt 10) / 2
  min_dist ≤ d ∧ d ≤ max_dist

theorem general_equation_M (θ : ℝ) : parametric_to_general θ := sorry

theorem range_distance_D_to_l (θ : ℝ) : distance_range θ := sorry

end general_equation_M_range_distance_D_to_l_l147_147994


namespace number_of_pieces_of_paper_l147_147081

def three_digit_number_with_unique_digits (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n / 100 ≠ (n / 10) % 10 ∧ n / 100 ≠ n % 10 ∧ (n / 10) % 10 ≠ n % 10

theorem number_of_pieces_of_paper (n : ℕ) (k : ℕ) (h1 : three_digit_number_with_unique_digits n) (h2 : 2331 = k * n) : k = 9 :=
by
  sorry

end number_of_pieces_of_paper_l147_147081


namespace largest_positive_integer_not_sum_of_multiple_30_and_composite_l147_147495

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

noncomputable def largest_not_sum_of_multiple_30_and_composite : ℕ :=
  211

theorem largest_positive_integer_not_sum_of_multiple_30_and_composite {m : ℕ} :
  m = largest_not_sum_of_multiple_30_and_composite ↔ 
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ k : ℕ, (b = k * 30) ∨ is_composite b) → m ≠ 30 * a + b) :=
sorry

end largest_positive_integer_not_sum_of_multiple_30_and_composite_l147_147495


namespace range_of_set_of_three_numbers_l147_147958

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l147_147958


namespace rowing_distance_l147_147711

theorem rowing_distance (v_b : ℝ) (v_s : ℝ) (t_total : ℝ) (D : ℝ) :
  v_b = 9 → v_s = 1.5 → t_total = 48 → D / (v_b + v_s) + D / (v_b - v_s) = t_total → D = 210 :=
by
  intros
  sorry

end rowing_distance_l147_147711


namespace polynomial_value_l147_147253

theorem polynomial_value (x y : ℝ) (h : x - 2 * y + 3 = 8) : x - 2 * y = 5 :=
by
  sorry

end polynomial_value_l147_147253


namespace largest_multiple_of_11_less_than_100_l147_147915

theorem largest_multiple_of_11_less_than_100 : 
  ∀ n, n < 100 → (∃ k, n = k * 11) → n ≤ 99 :=
by
  intro n hn hmul
  sorry

end largest_multiple_of_11_less_than_100_l147_147915


namespace steers_cows_unique_solution_l147_147454

-- Definition of the problem
def steers_and_cows_problem (s c : ℕ) : Prop :=
  25 * s + 26 * c = 1000 ∧ s > 0 ∧ c > 0

-- The theorem statement to be proved
theorem steers_cows_unique_solution :
  ∃! (s c : ℕ), steers_and_cows_problem s c ∧ c > s :=
sorry

end steers_cows_unique_solution_l147_147454


namespace probability_all_six_draws_white_l147_147049

theorem probability_all_six_draws_white :
  let total_balls := 14
  let white_balls := 7
  let single_draw_white_probability := (white_balls : ℚ) / total_balls
  (single_draw_white_probability ^ 6 = (1 : ℚ) / 64) :=
by
  sorry

end probability_all_six_draws_white_l147_147049


namespace parabola_equation_l147_147326

theorem parabola_equation 
  (vertex_x vertex_y : ℝ)
  (a b c : ℝ)
  (h_vertex : vertex_x = 3 ∧ vertex_y = 5)
  (h_point : ∃ x y: ℝ, x = 2 ∧ y = 2 ∧ y = a * (x - vertex_x)^2 + vertex_y)
  (h_vertical_axis : ∃ a b c, a = -3 ∧ b = 18 ∧ c = -22):
  ∀ x: ℝ, x ≠ vertex_x → b^2 - 4 * a * c > 0 := 
    sorry

end parabola_equation_l147_147326


namespace smallest_integer_n_l147_147265

theorem smallest_integer_n (m n : ℕ) (r : ℝ) :
  (m = (n + r)^3) ∧ (0 < r) ∧ (r < 1 / 2000) ∧ (m = n^3 + 3 * n^2 * r + 3 * n * r^2 + r^3) →
  n = 26 :=
by 
  sorry

end smallest_integer_n_l147_147265


namespace equivalent_problem_l147_147388

theorem equivalent_problem : 2 ^ (1 + 2 + 3) - (2 ^ 1 + 2 ^ 2 + 2 ^ 3) = 50 := by
  sorry

end equivalent_problem_l147_147388


namespace cost_of_500_pencils_in_dollars_l147_147679

def cost_of_pencil := 3 -- cost of 1 pencil in cents
def pencils_quantity := 500 -- number of pencils
def cents_in_dollar := 100 -- number of cents in 1 dollar

theorem cost_of_500_pencils_in_dollars :
  (pencils_quantity * cost_of_pencil) / cents_in_dollar = 15 := by
    sorry

end cost_of_500_pencils_in_dollars_l147_147679


namespace max_value_E_zero_l147_147365

noncomputable def E (a b c : ℝ) : ℝ :=
  a * b * c * (a - b * c^2) * (b - c * a^2) * (c - a * b^2)

theorem max_value_E_zero (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≥ b * c^2) (h2 : b ≥ c * a^2) (h3 : c ≥ a * b^2) :
  E a b c ≤ 0 :=
by
  sorry

end max_value_E_zero_l147_147365


namespace stickers_given_l147_147687

def total_stickers : ℕ := 100
def andrew_ratio : ℚ := 1 / 5
def bill_ratio : ℚ := 3 / 10

theorem stickers_given (zander_collection : ℕ)
                       (andrew_received : ℚ)
                       (bill_received : ℚ)
                       (total_given : ℚ):
  zander_collection = total_stickers →
  andrew_received = andrew_ratio →
  bill_received = bill_ratio →
  total_given = (andrew_received * zander_collection) + (bill_received * (zander_collection - (andrew_received * zander_collection))) →
  total_given = 44 :=
by
  intros hz har hbr htg
  sorry

end stickers_given_l147_147687


namespace quadratic_solution_1_quadratic_solution_2_l147_147623

theorem quadratic_solution_1 (x : ℝ) : x^2 - 8 * x + 12 = 0 ↔ x = 2 ∨ x = 6 := 
by
  sorry

theorem quadratic_solution_2 (x : ℝ) : (x - 3)^2 = 2 * x * (x - 3) ↔ x = 3 ∨ x = -3 := 
by
  sorry

end quadratic_solution_1_quadratic_solution_2_l147_147623


namespace complex_roots_equilateral_l147_147244

noncomputable def omega : ℂ := -1/2 + Complex.I * Real.sqrt 3 / 2

theorem complex_roots_equilateral (z1 z2 p q : ℂ) (h₁ : z2 = omega * z1) (h₂ : -p = (1 + omega) * z1) (h₃ : q = omega * z1 ^ 2) :
  p^2 / q = 1 + Complex.I * Real.sqrt 3 :=
by sorry

end complex_roots_equilateral_l147_147244


namespace green_duck_percentage_l147_147203

theorem green_duck_percentage (G_small G_large : ℝ) (D_small D_large : ℕ)
    (H1 : G_small = 0.20) (H2 : D_small = 20)
    (H3 : G_large = 0.15) (H4 : D_large = 80) : 
    ((G_small * D_small + G_large * D_large) / (D_small + D_large)) * 100 = 16 := 
by
  sorry

end green_duck_percentage_l147_147203


namespace circumscribed_circle_radius_l147_147598

noncomputable def circumradius_of_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℚ :=
  c / 2

theorem circumscribed_circle_radius :
  circumradius_of_right_triangle 30 40 50 (by norm_num : 30^2 + 40^2 = 50^2) = 25 := by
norm_num /- correct answer confirmed -/
sorry

end circumscribed_circle_radius_l147_147598


namespace victory_points_value_l147_147090

theorem victory_points_value (V : ℕ) (H : ∀ (v d t : ℕ), 
    v + d + t = 20 ∧ v * V + d ≥ 40 ∧ v ≥ 6 ∧ (t = 20 - 5)) : 
    V = 3 := 
sorry

end victory_points_value_l147_147090


namespace regular_price_coffee_l147_147285

theorem regular_price_coffee (y : ℝ) (h1 : 0.4 * y / 4 = 4) : y = 40 :=
by
  sorry

end regular_price_coffee_l147_147285


namespace VishalInvestedMoreThanTrishulBy10Percent_l147_147977

variables (R T V : ℝ)

-- Given conditions
def RaghuInvests (R : ℝ) : Prop := R = 2500
def TrishulInvests (R T : ℝ) : Prop := T = 0.9 * R
def TotalInvestment (R T V : ℝ) : Prop := V + T + R = 7225
def PercentageInvestedMore (T V : ℝ) (P : ℝ) : Prop := P * T = V - T

-- Main theorem to prove
theorem VishalInvestedMoreThanTrishulBy10Percent (R T V : ℝ) (P : ℝ) :
  RaghuInvests R ∧ TrishulInvests R T ∧ TotalInvestment R T V → PercentageInvestedMore T V P → P = 0.1 :=
by
  intros
  sorry

end VishalInvestedMoreThanTrishulBy10Percent_l147_147977


namespace num_elements_intersection_l147_147794

def A : Finset ℕ := {1, 2, 3, 4}
def B : Finset ℕ := {2, 4, 6, 8}

theorem num_elements_intersection : (A ∩ B).card = 2 := by
  sorry

end num_elements_intersection_l147_147794


namespace good_pair_exists_l147_147451

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ (∃ k1 k2 : ℕ, m * n = k1 * k1 ∧ (m + 1) * (n + 1) = k2 * k2) :=
by
  sorry

end good_pair_exists_l147_147451


namespace sequence_a_n_eq_5050_l147_147396

theorem sequence_a_n_eq_5050 (a : ℕ → ℕ) (h1 : ∀ n > 1, (n - 1) * a n = (n + 1) * a (n - 1)) (h2 : a 1 = 1) : 
  a 100 = 5050 := 
by
  sorry

end sequence_a_n_eq_5050_l147_147396


namespace parallel_lines_l147_147532

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y + 1 = 0) ∧ (∀ x y : ℝ, (a-1) * x - y + a = 0) →
  (a = 2 ↔ (∀ x1 y1 x2 y2 : ℝ, 2 * x1 - a * y1 + 1 = 0 ∧ (a-1) * x2 - y2 + a = 0 →
  (2 * x1 = (a * y1 - 1) ∧ (a-1) * x2 = y2 - a))) :=
sorry

end parallel_lines_l147_147532


namespace range_of_m_l147_147311

theorem range_of_m : 3 < 3 * Real.sqrt 2 - 1 ∧ 3 * Real.sqrt 2 - 1 < 4 :=
by
  have h1 : 3 * Real.sqrt 2 < 5 := sorry
  have h2 : 4 < 3 * Real.sqrt 2 := sorry
  exact ⟨by linarith, by linarith⟩

end range_of_m_l147_147311


namespace sums_ratio_l147_147448

theorem sums_ratio (total_sums : ℕ) (sums_right : ℕ) (sums_wrong: ℕ) (h1 : total_sums = 24) (h2 : sums_right = 8) (h3 : sums_wrong = total_sums - sums_right) :
  sums_wrong / Nat.gcd sums_wrong sums_right = 2 ∧ sums_right / Nat.gcd sums_wrong sums_right = 1 := by
  sorry

end sums_ratio_l147_147448


namespace square_area_of_triangle_on_hyperbola_l147_147823

noncomputable def centroid_is_vertex (triangle : Set (ℝ × ℝ)) : Prop :=
  ∃ v : ℝ × ℝ, v ∈ triangle ∧ v.1 * v.2 = 4

noncomputable def triangle_properties (triangle : Set (ℝ × ℝ)) : Prop :=
  centroid_is_vertex triangle ∧
  (∃ centroid : ℝ × ℝ, 
    centroid_is_vertex triangle ∧ 
    (∀ p ∈ triangle, centroid ∈ triangle))

theorem square_area_of_triangle_on_hyperbola :
  ∃ triangle : Set (ℝ × ℝ), triangle_properties triangle ∧ (∃ area_sq : ℝ, area_sq = 1728) :=
by
  sorry

end square_area_of_triangle_on_hyperbola_l147_147823


namespace least_possible_integral_BC_l147_147646

theorem least_possible_integral_BC :
  ∃ (BC : ℕ), (BC > 0) ∧ (BC ≥ 15) ∧ 
    (7 + BC > 15) ∧ (25 + 10 > BC) ∧ 
    (7 + 15 > BC) ∧ (25 + BC > 10) := by
    sorry

end least_possible_integral_BC_l147_147646


namespace smallest_possible_degree_p_l147_147892

theorem smallest_possible_degree_p (p : Polynomial ℝ) :
  (∀ x, 0 < |x| → ∃ C, |((3 * x^7 + 2 * x^6 - 4 * x^3 + x - 5) / (p.eval x)) - C| < ε)
  → (Polynomial.degree p) ≥ 7 := by
  sorry

end smallest_possible_degree_p_l147_147892


namespace opposite_of_seven_l147_147790

theorem opposite_of_seven : ∃ x : ℤ, 7 + x = 0 ∧ x = -7 :=
by
  sorry

end opposite_of_seven_l147_147790


namespace day_of_week_after_n_days_l147_147304

theorem day_of_week_after_n_days (birthday : ℕ) (n : ℕ) (day_of_week : ℕ) :
  birthday = 4 → (n % 7) = 2 → day_of_week = 6 :=
by sorry

end day_of_week_after_n_days_l147_147304


namespace factorization1_factorization2_factorization3_l147_147134

-- Problem 1
theorem factorization1 (a x : ℝ) : a * x^2 - 4 * a = a * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorization2 (m x y : ℝ) : m * x^2 + 2 * m * x * y + m * y^2 = m * (x + y)^2 :=
sorry

-- Problem 3
theorem factorization3 (a b : ℝ) : (1 / 2) * a^2 - a * b + (1 / 2) * b^2 = (1 / 2) * (a - b)^2 :=
sorry

end factorization1_factorization2_factorization3_l147_147134


namespace probability_of_exactly_9_correct_matches_is_zero_l147_147406

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∃ (P : ℕ → ℕ → ℕ), 
    (∀ (total correct : ℕ), 
      total = 10 → 
      correct = 9 → 
      P total correct = 0) := 
by {
  sorry
}

end probability_of_exactly_9_correct_matches_is_zero_l147_147406


namespace g_at_neg_1001_l147_147243

-- Defining the function g and the conditions
def g (x : ℝ) : ℝ := 2.5 * x - 0.5

-- Defining the main theorem to be proved
theorem g_at_neg_1001 : g (-1001) = -2503 := by
  sorry

end g_at_neg_1001_l147_147243


namespace num_assignments_l147_147535

/-- 
Mr. Wang originally planned to grade at a rate of 6 assignments per hour.
After grading for 2 hours, he increased his rate to 8 assignments per hour,
finishing 3 hours earlier than initially planned. 
Prove that the total number of assignments is 84. 
-/
theorem num_assignments (x : ℕ) (h : ℕ) (H1 : 6 * h = x) (H2 : 8 * (h - 5) = x - 12) : x = 84 :=
by
  sorry

end num_assignments_l147_147535


namespace soccer_uniform_probability_l147_147505

-- Definitions for the conditions of the problem
def colorsSocks : List String := ["red", "blue"]
def colorsShirts : List String := ["red", "blue", "green"]

noncomputable def differentColorConfigurations : Nat :=
  let validConfigs := [("red", "blue"), ("red", "green"), ("blue", "red"), ("blue", "green")]
  validConfigs.length

noncomputable def totalConfigurations : Nat :=
  colorsSocks.length * colorsShirts.length

noncomputable def probabilityDifferentColors : ℚ :=
  (differentColorConfigurations : ℚ) / (totalConfigurations : ℚ)

-- The theorem to prove
theorem soccer_uniform_probability :
  probabilityDifferentColors = 2 / 3 :=
by
  sorry

end soccer_uniform_probability_l147_147505


namespace total_time_spent_l147_147663

-- Define the conditions
def t1 : ℝ := 2.5
def t2 : ℝ := 3 * t1

-- Define the theorem to prove
theorem total_time_spent : t1 + t2 = 10 := by
  sorry

end total_time_spent_l147_147663


namespace primes_digit_sum_difference_l147_147269

def is_prime (a : ℕ) : Prop := Nat.Prime a

def sum_digits (n : ℕ) : ℕ := 
  Nat.digits 10 n |>.sum

theorem primes_digit_sum_difference (p q r : ℕ) (n : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (hneq : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (hpqr : p * q * r = 1899 * 10^n + 962) :
  (sum_digits p + sum_digits q + sum_digits r - sum_digits (p * q * r) = 8) := 
sorry

end primes_digit_sum_difference_l147_147269


namespace find_diameters_l147_147087

theorem find_diameters (x y z : ℕ) (hx : x ≠ y) (hy : y ≠ z) (hz : x ≠ z) :
  x + y + z = 26 ∧ x^2 + y^2 + z^2 = 338 :=
  sorry

end find_diameters_l147_147087


namespace probability_all_three_dice_twenty_l147_147645

theorem probability_all_three_dice_twenty (d1 d2 d3 d4 d5 : ℕ)
  (h1 : 1 ≤ d1 ∧ d1 ≤ 20) (h2 : 1 ≤ d2 ∧ d2 ≤ 20) (h3 : 1 ≤ d3 ∧ d3 ≤ 20)
  (h4 : 1 ≤ d4 ∧ d4 ≤ 20) (h5 : 1 ≤ d5 ∧ d5 ≤ 20)
  (h6 : d1 = 20) (h7 : d2 = 19)
  (h8 : (if d1 = 20 then 1 else 0) + (if d2 = 20 then 1 else 0) +
        (if d3 = 20 then 1 else 0) + (if d4 = 20 then 1 else 0) +
        (if d5 = 20 then 1 else 0) ≥ 3) :
  (1 / 58 : ℚ) = (if d3 = 20 ∧ d4 = 20 ∧ d5 = 20 then 1 else 0) /
                 ((if d3 = 20 ∧ d4 = 20 then 19 else 0) +
                  (if d3 = 20 ∧ d5 = 20 then 19 else 0) +
                  (if d4 = 20 ∧ d5 = 20 then 19 else 0) + 
                  (if d3 = 20 ∧ d4 = 20 ∧ d5 = 20 then 1 else 0) : ℚ) :=
sorry

end probability_all_three_dice_twenty_l147_147645


namespace difference_is_minus_four_l147_147964

def percentage_scoring_60 : ℝ := 0.15
def percentage_scoring_75 : ℝ := 0.25
def percentage_scoring_85 : ℝ := 0.40
def percentage_scoring_95 : ℝ := 1 - (percentage_scoring_60 + percentage_scoring_75 + percentage_scoring_85)

def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_85 : ℝ := 85
def score_95 : ℝ := 95

def mean_score : ℝ :=
  (percentage_scoring_60 * score_60) +
  (percentage_scoring_75 * score_75) +
  (percentage_scoring_85 * score_85) +
  (percentage_scoring_95 * score_95)

def median_score : ℝ := score_85

def difference_mean_median : ℝ := mean_score - median_score

theorem difference_is_minus_four : difference_mean_median = -4 :=
by
  sorry

end difference_is_minus_four_l147_147964


namespace abs_neg_one_eq_one_l147_147752

theorem abs_neg_one_eq_one : abs (-1 : ℚ) = 1 := 
by
  sorry

end abs_neg_one_eq_one_l147_147752


namespace distinct_sums_count_l147_147810

theorem distinct_sums_count (n : ℕ) (a : Fin n.succ → ℕ) (h_distinct : Function.Injective a) :
  ∃ (S : Finset ℕ), S.card ≥ n * (n + 1) / 2 := sorry

end distinct_sums_count_l147_147810


namespace correct_assignment_statement_l147_147282

theorem correct_assignment_statement (a b : ℕ) : 
  (2 = a → False) ∧ 
  (a = a + 1 → True) ∧ 
  (a * b = 2 → False) ∧ 
  (a + 1 = a → False) :=
by {
  sorry
}

end correct_assignment_statement_l147_147282


namespace manuscript_pages_l147_147371

theorem manuscript_pages (P : ℕ) (rate_first : ℕ) (rate_revision : ℕ) 
  (revised_once_pages : ℕ) (revised_twice_pages : ℕ) (total_cost : ℕ) :
  rate_first = 6 →
  rate_revision = 4 →
  revised_once_pages = 35 →
  revised_twice_pages = 15 →
  total_cost = 860 →
  6 * (P - 35 - 15) + 10 * 35 + 14 * 15 = total_cost →
  P = 100 :=
by
  intros h_first h_revision h_once h_twice h_cost h_eq
  sorry

end manuscript_pages_l147_147371


namespace two_digit_sum_divisible_by_17_l147_147348

theorem two_digit_sum_divisible_by_17 :
  ∃ A : ℕ, A ≥ 10 ∧ A < 100 ∧ ∃ B : ℕ, B = (A % 10) * 10 + (A / 10) ∧ (A + B) % 17 = 0 ↔ A = 89 ∨ A = 98 := 
sorry

end two_digit_sum_divisible_by_17_l147_147348


namespace geometric_mean_of_1_and_4_l147_147570

theorem geometric_mean_of_1_and_4 :
  ∃ a : ℝ, a^2 = 4 ∧ (a = 2 ∨ a = -2) :=
by
  sorry

end geometric_mean_of_1_and_4_l147_147570


namespace correct_answer_l147_147924

-- Definitions of the groups
def group_1_well_defined : Prop := false -- Smaller numbers
def group_2_well_defined : Prop := true  -- Non-negative even numbers not greater than 10
def group_3_well_defined : Prop := true  -- All triangles
def group_4_well_defined : Prop := false -- Tall male students

-- Propositions representing the options
def option_A : Prop := group_1_well_defined ∧ group_4_well_defined
def option_B : Prop := group_2_well_defined ∧ group_3_well_defined
def option_C : Prop := group_2_well_defined
def option_D : Prop := group_3_well_defined

-- Theorem stating Option B is the correct answer
theorem correct_answer : option_B ∧ ¬option_A ∧ ¬option_C ∧ ¬option_D := by
  sorry

end correct_answer_l147_147924


namespace abs_x_minus_one_sufficient_not_necessary_l147_147477

variable (x : ℝ) -- x is a real number

theorem abs_x_minus_one_sufficient_not_necessary (h : |x - 1| > 2) :
  (x^2 > 1) ∧ (∃ (y : ℝ), x^2 > 1 ∧ |y - 1| ≤ 2) := by
  sorry

end abs_x_minus_one_sufficient_not_necessary_l147_147477


namespace ratio_of_B_to_C_l147_147674

theorem ratio_of_B_to_C
  (A B C : ℕ) 
  (h1 : A = B + 2) 
  (h2 : A + B + C = 47) 
  (h3 : B = 18) : B / C = 2 := 
by 
  sorry

end ratio_of_B_to_C_l147_147674


namespace tablets_taken_l147_147179

theorem tablets_taken (total_time interval_time : ℕ) (h1 : total_time = 60) (h2 : interval_time = 15) : total_time / interval_time = 4 :=
by
  sorry

end tablets_taken_l147_147179


namespace number_of_six_digit_palindromes_l147_147353

def is_six_digit_palindrome (n : ℕ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ n = a * 100001 + b * 10010 + c * 1100

theorem number_of_six_digit_palindromes : ∃ p, p = 900 ∧ (∀ n, is_six_digit_palindrome n → n = p) :=
by
  sorry

end number_of_six_digit_palindromes_l147_147353


namespace arithmetic_sequence_a7_l147_147362

theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (h1 : a 1 = 3) (h3 : a 3 = 5) (h_arith : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)) : a 7 = 9 :=
by
  sorry

end arithmetic_sequence_a7_l147_147362


namespace inequality_solution_set_l147_147438

theorem inequality_solution_set :
  {x : ℝ | (x - 5) * (x + 1) > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 5} :=
by
  sorry

end inequality_solution_set_l147_147438


namespace complement_union_eq_l147_147292

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_union_eq :
  U = {1, 2, 3, 4, 5, 6, 7, 8} →
  A = {1, 3, 5, 7} →
  B = {2, 4, 5} →
  U \ (A ∪ B) = {6, 8} :=
by
  intros hU hA hB
  -- Proof goes here
  sorry

end complement_union_eq_l147_147292


namespace sequence_1005th_term_l147_147554

-- Definitions based on conditions
def first_term : ℚ := sorry
def second_term : ℚ := 10
def third_term : ℚ := 4 * first_term - (1:ℚ)
def fourth_term : ℚ := 4 * first_term + (1:ℚ)

-- Common difference
def common_difference : ℚ := (fourth_term - third_term)

-- Arithmetic sequence term calculation
def nth_term (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n-1) * d

-- Theorem statement
theorem sequence_1005th_term : nth_term first_term common_difference 1005 = 5480 := sorry

end sequence_1005th_term_l147_147554


namespace calc_expr_value_l147_147545

theorem calc_expr_value : (0.5 ^ 4) / (0.05 ^ 2.5) = 559.06 := 
by 
  sorry

end calc_expr_value_l147_147545


namespace investment_calculation_l147_147039

theorem investment_calculation :
  ∃ (x : ℝ), x * (1.04 ^ 14) = 1000 := by
  use 571.75
  sorry

end investment_calculation_l147_147039


namespace solve_equation_1_solve_equation_2_l147_147869

theorem solve_equation_1 (x : ℝ) : x^2 - 3 * x = 4 ↔ x = 4 ∨ x = -1 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by
  sorry

end solve_equation_1_solve_equation_2_l147_147869


namespace ze_age_conditions_l147_147076

theorem ze_age_conditions 
  (z g t : ℕ)
  (h1 : z = 2 * g + 3 * t)
  (h2 : 2 * (z + 15) = 2 * (g + 15) + 3 * (t + 15))
  (h3 : 2 * (g + 15) = 3 * (t + 15)) :
  z = 45 ∧ t = 5 :=
by
  sorry

end ze_age_conditions_l147_147076


namespace fraction_of_emilys_coins_l147_147594

theorem fraction_of_emilys_coins {total_states : ℕ} (h1 : total_states = 30)
    {states_from_1790_to_1799 : ℕ} (h2 : states_from_1790_to_1799 = 9) :
    (states_from_1790_to_1799 / total_states : ℚ) = 3 / 10 := by
  sorry

end fraction_of_emilys_coins_l147_147594


namespace boxes_to_eliminate_l147_147460

noncomputable def total_boxes : ℕ := 26
noncomputable def high_value_boxes : ℕ := 6
noncomputable def threshold_probability : ℚ := 1 / 2

-- Define the condition for having the minimum number of boxes
def min_boxes_needed_for_probability (total high_value : ℕ) (prob : ℚ) : ℕ :=
  total - high_value - ((total - high_value) / 2)

theorem boxes_to_eliminate :
  min_boxes_needed_for_probability total_boxes high_value_boxes threshold_probability = 15 :=
by
  sorry

end boxes_to_eliminate_l147_147460


namespace problem_f_symmetric_l147_147951

theorem problem_f_symmetric (f : ℝ → ℝ) (k : ℝ) (h : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + k * f b) (h_not_zero : ∃ x : ℝ, f x ≠ 0) :
  ∀ x : ℝ, f (-x) = f x :=
sorry

end problem_f_symmetric_l147_147951


namespace profit_sharing_l147_147338

theorem profit_sharing 
  (total_profit : ℝ) 
  (managing_share_percentage : ℝ) 
  (capital_a : ℝ) 
  (capital_b : ℝ) 
  (managing_partner_share : ℝ)
  (total_capital : ℝ) 
  (remaining_profit : ℝ) 
  (proportion_a : ℝ)
  (share_a_remaining : ℝ)
  (total_share_a : ℝ) : 
  total_profit = 8800 → 
  managing_share_percentage = 0.125 → 
  capital_a = 50000 → 
  capital_b = 60000 → 
  managing_partner_share = managing_share_percentage * total_profit → 
  total_capital = capital_a + capital_b → 
  remaining_profit = total_profit - managing_partner_share → 
  proportion_a = capital_a / total_capital → 
  share_a_remaining = proportion_a * remaining_profit → 
  total_share_a = managing_partner_share + share_a_remaining → 
  total_share_a = 4600 :=
by sorry

end profit_sharing_l147_147338


namespace jessica_repay_l147_147730

theorem jessica_repay (P : ℝ) (r : ℝ) (n : ℝ) (x : ℕ)
  (hx : P = 20)
  (hr : r = 0.12)
  (hn : n = 3 * P) :
  x = 17 :=
sorry

end jessica_repay_l147_147730


namespace subtraction_example_l147_147218

theorem subtraction_example : -1 - 3 = -4 := 
  sorry

end subtraction_example_l147_147218


namespace inscribed_circle_radius_l147_147578

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem inscribed_circle_radius :
  radius_of_inscribed_circle 8 8 5 = 38 / 21 :=
by
  sorry

end inscribed_circle_radius_l147_147578


namespace original_surface_area_l147_147366

theorem original_surface_area (R : ℝ) (h : 2 * π * R^2 = 4 * π) : 4 * π * R^2 = 8 * π :=
by
  sorry

end original_surface_area_l147_147366


namespace chord_of_ellipse_bisected_by_point_l147_147011

theorem chord_of_ellipse_bisected_by_point :
  ∀ (x y : ℝ),
  (∃ (x₁ x₂ y₁ y₂ : ℝ), 
    ( (x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 2) ∧ 
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧ 
    (x₂^2 / 36 + y₂^2 / 9 = 1)) →
  (x + 2 * y = 8) :=
by
  sorry

end chord_of_ellipse_bisected_by_point_l147_147011


namespace particular_number_l147_147703

theorem particular_number {x : ℕ} (h : x - 29 + 64 = 76) : x = 41 := by
  sorry

end particular_number_l147_147703


namespace number_of_books_is_10_l147_147901

def costPerBookBeforeDiscount : ℝ := 5
def discountPerBook : ℝ := 0.5
def totalPayment : ℝ := 45

theorem number_of_books_is_10 (n : ℕ) (h : (costPerBookBeforeDiscount - discountPerBook) * n = totalPayment) : n = 10 := by
  sorry

end number_of_books_is_10_l147_147901


namespace solution_set_f_ge_1_l147_147335

noncomputable def f (x : ℝ) (a : ℝ) :=
  if x >= 0 then |x - 2| + a else -(|-x - 2| + a)

theorem solution_set_f_ge_1 {a : ℝ} (ha : a = -2) :
  {x : ℝ | f x a ≥ 1} = {x : ℝ | x ≤ -1 ∨ x ≥ 5} :=
by sorry

end solution_set_f_ge_1_l147_147335


namespace min_value_f_l147_147791

-- Define the function f(x)
def f (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x) + 200 * x^2

-- State the theorem to be proved
theorem min_value_f : ∃ (x : ℝ), (∀ y : ℝ, f y ≥ 33) ∧ f x = 33 := by
  sorry

end min_value_f_l147_147791


namespace find_x_l147_147746

theorem find_x (x : ℕ) (h : 1 + 2 + 3 + 4 + 5 + x = 21 + 22 + 23 + 24 + 25) : x = 100 :=
by {
  sorry
}

end find_x_l147_147746


namespace mass_percentage_C_in_CaCO3_is_correct_l147_147175

structure Element where
  name : String
  molar_mass : ℚ

def Ca : Element := ⟨"Ca", 40.08⟩
def C : Element := ⟨"C", 12.01⟩
def O : Element := ⟨"O", 16.00⟩

def molar_mass_CaCO3 : ℚ :=
  Ca.molar_mass + C.molar_mass + 3 * O.molar_mass

def mass_percentage_C_in_CaCO3 : ℚ :=
  (C.molar_mass / molar_mass_CaCO3) * 100

theorem mass_percentage_C_in_CaCO3_is_correct :
  mass_percentage_C_in_CaCO3 = 12.01 :=
by
  sorry

end mass_percentage_C_in_CaCO3_is_correct_l147_147175


namespace total_distance_traveled_l147_147816

theorem total_distance_traveled (d d1 d2 d3 d4 d5 : ℕ) 
  (h1 : d1 = d)
  (h2 : d2 = 2 * d)
  (h3 : d3 = 40)
  (h4 : d = 2 * d3)
  (h5 : d4 = 2 * (d1 + d2 + d3))
  (h6 : d5 = 3 * d4 / 2) 
  : d1 + d2 + d3 + d4 + d5 = 1680 :=
by
  have hd : d = 80 := sorry
  have hd1 : d1 = 80 := sorry
  have hd2 : d2 = 160 := sorry
  have hd4 : d4 = 560 := sorry
  have hd5 : d5 = 840 := sorry
  sorry

end total_distance_traveled_l147_147816


namespace eccentricity_range_l147_147051

-- We start with the given problem and conditions
variables {a c b : ℝ}
def C1 := ∀ x y, x^2 + 2 * c * x + y^2 = 0
def C2 := ∀ x y, x^2 - 2 * c * x + y^2 = 0
def ellipse := ∀ x y, x^2 / a^2 + y^2 / b^2 = 1

-- Ellipse semi-latus rectum condition and circles inside the ellipse
axiom h1 : c = b^2 / a
axiom h2 : a > 2 * c

-- Proving the range of the eccentricity
theorem eccentricity_range : 0 < c / a ∧ c / a < 1 / 2 :=
by
  sorry

end eccentricity_range_l147_147051


namespace correct_statements_l147_147644

theorem correct_statements : 
    let statement1 := "The regression effect is characterized by the relevant exponent R^{2}. The larger the R^{2}, the better the fitting effect."
    let statement2 := "The properties of a sphere are inferred from the properties of a circle by analogy."
    let statement3 := "Any two complex numbers cannot be compared in size."
    let statement4 := "Flowcharts are often used to represent some dynamic processes, usually with a 'starting point' and an 'ending point'."
    true -> (statement1 = "correct" ∧ statement2 = "correct" ∧ statement3 = "incorrect" ∧ statement4 = "incorrect") :=
by
  -- proof
  sorry

end correct_statements_l147_147644


namespace complement_N_star_in_N_l147_147562

-- The set of natural numbers
def N : Set ℕ := { n | true }

-- The set of positive integers
def N_star : Set ℕ := { n | n > 0 }

-- The complement of N_star in N is the set {0}
theorem complement_N_star_in_N : { n | n ∈ N ∧ n ∉ N_star } = {0} := by
  sorry

end complement_N_star_in_N_l147_147562


namespace transportation_degrees_correct_l147_147904

-- Define the percentages for the different categories.
def salaries_percent := 0.60
def research_development_percent := 0.09
def utilities_percent := 0.05
def equipment_percent := 0.04
def supplies_percent := 0.02

-- Define the total percentage of non-transportation categories.
def non_transportation_percent := 
  salaries_percent + research_development_percent + utilities_percent + equipment_percent + supplies_percent

-- Define the full circle in degrees.
def full_circle_degrees := 360.0

-- Total percentage which must sum to 1 (i.e., 100%).
def total_budget_percent := 1.0

-- Calculate the percentage for transportation.
def transportation_percent := total_budget_percent - non_transportation_percent

-- Define the result for degrees allocated to transportation.
def transportation_degrees := transportation_percent * full_circle_degrees

-- Prove that the transportation degrees are 72.
theorem transportation_degrees_correct : transportation_degrees = 72.0 :=
by
  unfold transportation_degrees transportation_percent non_transportation_percent
  sorry

end transportation_degrees_correct_l147_147904


namespace apple_counting_l147_147855

theorem apple_counting
  (n m : ℕ)
  (vasya_trees_a_b petya_trees_a_b vasya_trees_b_c petya_trees_b_c vasya_trees_c_d petya_trees_c_d vasya_apples_a_b petya_apples_a_b vasya_apples_c_d petya_apples_c_d : ℕ)
  (h1 : petya_trees_a_b = 2 * vasya_trees_a_b)
  (h2 : petya_apples_a_b = 7 * vasya_apples_a_b)
  (h3 : petya_trees_b_c = 2 * vasya_trees_b_c)
  (h4 : petya_trees_c_d = 2 * vasya_trees_c_d)
  (h5 : n = vasya_trees_a_b + petya_trees_a_b)
  (h6 : m = vasya_apples_a_b + petya_apples_a_b)
  (h7 : vasya_trees_c_d = n / 3)
  (h8 : petya_trees_c_d = 2 * (n / 3))
  (h9 : vasya_apples_c_d = 3 * petya_apples_c_d)
  : vasya_apples_c_d = 3 * petya_apples_c_d :=
by 
  sorry

end apple_counting_l147_147855


namespace fifth_inequality_proof_l147_147376

theorem fifth_inequality_proof :
  (1 + 1 / (2^2 : ℝ) + 1 / (3^2 : ℝ) + 1 / (4^2 : ℝ) + 1 / (5^2 : ℝ) + 1 / (6^2 : ℝ) < 11 / 6) 
  := 
sorry

end fifth_inequality_proof_l147_147376


namespace volume_inequality_holds_l147_147733

def volume (x : ℕ) : ℤ :=
  (x^2 - 16) * (x^3 + 25)

theorem volume_inequality_holds :
  ∃ (n : ℕ), n = 1 ∧ ∃ x : ℕ, volume x < 1000 ∧ (x - 4) > 0 :=
by
  sorry

end volume_inequality_holds_l147_147733


namespace common_difference_of_arithmetic_sequence_l147_147935

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
noncomputable def S_n (n : ℕ) : ℝ := -n^2 + 4*n

theorem common_difference_of_arithmetic_sequence :
  (∀ n : ℕ, S n = S_n n) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d ∧ d = -2 :=
by
  intro h
  use -2
  sorry

end common_difference_of_arithmetic_sequence_l147_147935


namespace new_girl_weight_l147_147146

theorem new_girl_weight (W : ℝ) (h : (W + 24) / 8 = W / 8 + 3) :
  (W + 24) - (W - 70) = 94 :=
by
  sorry

end new_girl_weight_l147_147146


namespace cos_sum_condition_l147_147384

theorem cos_sum_condition {x y z : ℝ} (h1 : Real.cos x + Real.cos y + Real.cos z = 1) (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 1 := 
by 
  sorry

end cos_sum_condition_l147_147384


namespace avg_of_five_consecutive_from_b_l147_147684

-- Conditions
def avg_of_five_even_consecutive (a : ℕ) : ℕ := (2 * a + (2 * a + 2) + (2 * a + 4) + (2 * a + 6) + (2 * a + 8)) / 5

-- The main theorem
theorem avg_of_five_consecutive_from_b (a : ℕ) : 
  avg_of_five_even_consecutive a = 2 * a + 4 → 
  ((2 * a + 4 + (2 * a + 4 + 1) + (2 * a + 4 + 2) + (2 * a + 4 + 3) + (2 * a + 4 + 4)) / 5) = 2 * a + 6 :=
by
  sorry

end avg_of_five_consecutive_from_b_l147_147684


namespace proof_problem_l147_147712

noncomputable def find_values (a b c x y z : ℝ) := 
  14 * x + b * y + c * z = 0 ∧ 
  a * x + 24 * y + c * z = 0 ∧ 
  a * x + b * y + 43 * z = 0 ∧ 
  a ≠ 14 ∧ b ≠ 24 ∧ c ≠ 43 ∧ x ≠ 0

theorem proof_problem (a b c x y z : ℝ) 
  (h : find_values a b c x y z):
  (a / (a - 14)) + (b / (b - 24)) + (c / (c - 43)) = 1 :=
by
  sorry

end proof_problem_l147_147712


namespace tiling_scenarios_unique_l147_147681

theorem tiling_scenarios_unique (m n : ℕ) 
  (h1 : 60 * m + 150 * n = 360) : m = 1 ∧ n = 2 :=
by {
  -- The proof will be provided here
  sorry
}

end tiling_scenarios_unique_l147_147681


namespace waiting_period_l147_147137

-- Variable declarations
variables (P : ℕ) (H : ℕ) (W : ℕ) (A : ℕ) (T : ℕ)
-- Condition declarations
variables (hp : P = 3) (hh : H = 5 * P) (ha : A = 3 * 7) (ht : T = 39)
-- Total time equation
variables (h_total : P + H + W + A = T)

-- Statement to prove
theorem waiting_period (hp : P = 3) (hh : H = 5 * P) (ha : A = 3 * 7) (ht : T = 39) (h_total : P + H + W + A = T) : 
  W = 3 :=
sorry

end waiting_period_l147_147137


namespace arithmetic_sum_l147_147344

variables {a d : ℝ}

theorem arithmetic_sum (h : 15 * a + 105 * d = 90) : 2 * a + 14 * d = 12 :=
sorry

end arithmetic_sum_l147_147344


namespace max_ab_value_l147_147913

noncomputable def max_ab (a b : ℝ) : ℝ := a * b

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 8) : max_ab a b ≤ 4 :=
by
  sorry

end max_ab_value_l147_147913


namespace value_of_pq_s_l147_147044

-- Definitions for the problem
def polynomial_divisible (p q s : ℚ) : Prop :=
  ∀ x : ℚ, (x^3 + 4 * x^2 + 16 * x + 8) ∣ (x^4 + 6 * x^3 + 8 * p * x^2 + 6 * q * x + s)

-- The main theorem statement to prove
theorem value_of_pq_s (p q s : ℚ) (h : polynomial_divisible p q s) : (p + q) * s = 332 / 3 :=
sorry -- Proof omitted

end value_of_pq_s_l147_147044


namespace stamps_initial_count_l147_147992

theorem stamps_initial_count (total_stamps stamps_received initial_stamps : ℕ) 
  (h1 : total_stamps = 61)
  (h2 : stamps_received = 27)
  (h3 : initial_stamps = total_stamps - stamps_received) :
  initial_stamps = 34 :=
sorry

end stamps_initial_count_l147_147992


namespace part_a_part_b_l147_147516

-- Define sum conditions for consecutive odd integers
def consecutive_odd_sum (N : ℕ) : Prop :=
  ∃ (n k : ℕ), n ≥ 2 ∧ N = n * (2 * k + n)

-- Part (a): Prove 2005 can be written as sum of consecutive odd positive integers
theorem part_a : consecutive_odd_sum 2005 :=
by
  sorry

-- Part (b): Prove 2006 cannot be written as sum of consecutive odd positive integers
theorem part_b : ¬consecutive_odd_sum 2006 :=
by
  sorry

end part_a_part_b_l147_147516


namespace smallest_n_for_fraction_with_digits_439_l147_147358

theorem smallest_n_for_fraction_with_digits_439 (m n : ℕ) (hmn : Nat.gcd m n = 1) (hmn_pos : 0 < m ∧ m < n) (digits_439 : ∃ X : ℕ, (m : ℚ) / n = (439 + 1000 * X) / 1000) : n = 223 :=
by
  sorry

end smallest_n_for_fraction_with_digits_439_l147_147358


namespace petya_run_12_seconds_l147_147669

-- Define the conditions
variable (petya_speed classmates_speed : ℕ → ℕ) -- speeds of Petya and his classmates
variable (total_distance : ℕ := 100) -- each participant needs to run 100 meters
variable (initial_total_distance_run : ℕ := 288) -- total distance run by all in the first 12 seconds
variable (remaining_distance_when_petya_finished : ℕ := 40) -- remaining distance for others when Petya finished
variable (time_to_first_finish : ℕ) -- the time Petya takes to finish the race

-- Assume constant speeds for all participants
axiom constant_speed_petya (t : ℕ) : petya_speed t = petya_speed 0
axiom constant_speed_classmates (t : ℕ) : classmates_speed t = classmates_speed 0

-- Summarized total distances run by participants
axiom total_distance_run_all (t : ℕ) :
  petya_speed t * t + classmates_speed t * t = initial_total_distance_run + remaining_distance_when_petya_finished + (total_distance - remaining_distance_when_petya_finished) * 3

-- Given conditions converted to Lean
axiom initial_distance_run (t : ℕ) :
  t = 12 → petya_speed t * t + classmates_speed t * t = initial_total_distance_run

axiom petya_completion (t : ℕ) :
  t = time_to_first_finish → petya_speed t * t = total_distance

axiom remaining_distance_classmates (t : ℕ) :
  t = time_to_first_finish → classmates_speed t * (t - time_to_first_finish) = remaining_distance_when_petya_finished
  
-- Define the proof goal using the conditions
theorem petya_run_12_seconds (d : ℕ) :
  (∃ t, t = 12 ∧ d = petya_speed t * t) → d = 80 :=
by
  sorry

end petya_run_12_seconds_l147_147669


namespace train_length_l147_147610

theorem train_length (t_post t_platform l_platform : ℕ) (L : ℚ) : 
  t_post = 15 → t_platform = 25 → l_platform = 100 →
  (L / t_post) = (L + l_platform) / t_platform → 
  L = 150 :=
by 
  intros h1 h2 h3 h4
  -- Proof steps would go here
  sorry

end train_length_l147_147610


namespace clea_total_time_l147_147859

-- Definitions based on conditions given
def walking_time_on_stationary (x y : ℝ) (h1 : 80 * x = y) : ℝ :=
  80

def walking_time_on_moving (x y : ℝ) (k : ℝ) (h2 : 32 * (x + k) = y) : ℝ :=
  32

def escalator_speed (x k : ℝ) (h3 : k = 1.5 * x) : ℝ :=
  1.5 * x

-- The actual theorem based on the question
theorem clea_total_time 
  (x y k : ℝ)
  (h1 : 80 * x = y)
  (h2 : 32 * (x + k) = y)
  (h3 : k = 1.5 * x) :
  let t1 := y / (2 * x)
  let t2 := y / (3 * x)
  t1 + t2 = 200 / 3 :=
by
  sorry

end clea_total_time_l147_147859


namespace toy_ratio_l147_147652

variable (Jaxon : ℕ) (Gabriel : ℕ) (Jerry : ℕ)

theorem toy_ratio (h1 : Jerry = Gabriel + 8) 
                  (h2 : Jaxon = 15)
                  (h3 : Gabriel + Jerry + Jaxon = 83) :
                  Gabriel / Jaxon = 2 := 
by
  sorry

end toy_ratio_l147_147652


namespace minimum_value_l147_147602

open Real

-- Given the conditions
variables (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k)

-- The theorem
theorem minimum_value (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < k) : 
  ∃ x, x = (3 : ℝ) / k ∧ ∀ y, y = (a / (k * b) + b / (k * c) + c / (k * a)) → y ≥ x :=
sorry

end minimum_value_l147_147602


namespace present_age_of_son_l147_147489

variable (S M : ℕ)

-- Conditions
def condition1 := M = S + 28
def condition2 := M + 2 = 2 * (S + 2)

-- Theorem to be proven
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 26 := by
  sorry

end present_age_of_son_l147_147489


namespace product_of_digits_l147_147082

-- Define the conditions and state the theorem
theorem product_of_digits (A B : ℕ) (h1 : (10 * A + B) % 12 = 0) (h2 : A + B = 12) : A * B = 32 :=
  sorry

end product_of_digits_l147_147082


namespace base6_sum_correct_l147_147910

theorem base6_sum_correct {S H E : ℕ} (hS : S < 6) (hH : H < 6) (hE : E < 6) 
  (dist : S ≠ H ∧ H ≠ E ∧ S ≠ E) 
  (rightmost : (E + E) % 6 = S) 
  (second_rightmost : (H + H + if E + E < 6 then 0 else 1) % 6 = E) :
  S + H + E = 11 := 
by sorry

end base6_sum_correct_l147_147910


namespace opposite_of_neg_eight_l147_147007

theorem opposite_of_neg_eight (y : ℤ) (h : y + (-8) = 0) : y = 8 :=
by {
  -- proof goes here
  sorry
}

end opposite_of_neg_eight_l147_147007


namespace difference_cubed_divisible_by_27_l147_147046

theorem difference_cubed_divisible_by_27 (a b : ℤ) :
    ((3 * a + 2) ^ 3 - (3 * b + 2) ^ 3) % 27 = 0 := 
by
  sorry

end difference_cubed_divisible_by_27_l147_147046


namespace calculate_expression_l147_147000

theorem calculate_expression : 
  ((13^13 / 13^12)^3 * 3^3) / 3^6 = 27 :=
by
  sorry

end calculate_expression_l147_147000


namespace days_to_fulfill_order_l147_147200

theorem days_to_fulfill_order (bags_per_batch : ℕ) (total_order : ℕ) (initial_bags : ℕ) (required_days : ℕ) :
  bags_per_batch = 10 →
  total_order = 60 →
  initial_bags = 20 →
  required_days = (total_order - initial_bags) / bags_per_batch →
  required_days = 4 :=
by
  intros
  sorry

end days_to_fulfill_order_l147_147200


namespace g_g_g_of_3_eq_neg_6561_l147_147429

def g (x : ℤ) : ℤ := -x^2

theorem g_g_g_of_3_eq_neg_6561 : g (g (g 3)) = -6561 := by
  sorry

end g_g_g_of_3_eq_neg_6561_l147_147429


namespace probability_of_qualified_product_l147_147572

theorem probability_of_qualified_product :
  let p1 := 0.30   -- Proportion of the first batch
  let d1 := 0.05   -- Defect rate of the first batch
  let p2 := 0.70   -- Proportion of the second batch
  let d2 := 0.04   -- Defect rate of the second batch
  -- Probability of selecting a qualified product
  p1 * (1 - d1) + p2 * (1 - d2) = 0.957 :=
by
  sorry

end probability_of_qualified_product_l147_147572


namespace domain_sqrt_product_domain_log_fraction_l147_147509

theorem domain_sqrt_product (x : ℝ) (h1 : x - 2 ≥ 0) (h2 : x + 2 ≥ 0) : 
  2 ≤ x :=
by sorry

theorem domain_log_fraction (x : ℝ) (h1 : x + 1 > 0) (h2 : -x^2 - 3 * x + 4 > 0) : 
  -1 < x ∧ x < 1 :=
by sorry

end domain_sqrt_product_domain_log_fraction_l147_147509


namespace collinear_points_sum_l147_147885

theorem collinear_points_sum (p q : ℝ) 
  (h1 : p = 2) (h2 : q = 4) 
  (collinear : ∃ (s : ℝ), 
     (2, p, q) = (2, s*p, s*q) ∧ 
     (p, 3, q) = (s*p, 3, s*q) ∧ 
     (p, q, 4) = (s*p, s*q, 4)): 
  p + q = 6 := by
  sorry

end collinear_points_sum_l147_147885


namespace one_clerk_forms_per_hour_l147_147864

theorem one_clerk_forms_per_hour
  (total_forms : ℕ)
  (total_hours : ℕ)
  (total_clerks : ℕ) 
  (h1 : total_forms = 2400)
  (h2 : total_hours = 8)
  (h3 : total_clerks = 12) :
  (total_forms / total_hours) / total_clerks = 25 :=
by
  have forms_per_hour := total_forms / total_hours
  have forms_per_clerk_per_hour := forms_per_hour / total_clerks
  sorry

end one_clerk_forms_per_hour_l147_147864


namespace preservation_time_at_33_degrees_l147_147553

noncomputable def preservation_time (x : ℝ) (k : ℝ) (b : ℝ) : ℝ :=
  Real.exp (k * x + b)

theorem preservation_time_at_33_degrees (k b : ℝ) 
  (h1 : Real.exp b = 192)
  (h2 : Real.exp (22 * k + b) = 48) :
  preservation_time 33 k b = 24 := by
  sorry

end preservation_time_at_33_degrees_l147_147553


namespace businessman_earnings_l147_147620

theorem businessman_earnings : 
  let P : ℝ := 1000
  let day1_stock := 1000 / P
  let day2_stock := 1000 / (P * 1.1)
  let day3_stock := 1000 / (P * 1.1^2)
  let value_on_day4 stock := stock * (P * 1.1^3)
  let total_earnings := value_on_day4 day1_stock + value_on_day4 day2_stock + value_on_day4 day3_stock
  total_earnings = 3641 := sorry

end businessman_earnings_l147_147620


namespace sum_first_4_terms_of_arithmetic_sequence_eq_8_l147_147588

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 1 - a 0)

def S4 (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_first_4_terms_of_arithmetic_sequence_eq_8
  (a : ℕ → ℤ) 
  (h_seq : arithmetic_seq a) 
  (h_a2 : a 1 = 1) 
  (h_a3 : a 2 = 3) :
  S4 a = 8 :=
by
  sorry

end sum_first_4_terms_of_arithmetic_sequence_eq_8_l147_147588


namespace alex_age_thrice_ben_in_n_years_l147_147781

-- Definitions based on the problem's conditions
def Ben_current_age := 4
def Alex_current_age := Ben_current_age + 30

-- The main problem defined as a theorem to be proven
theorem alex_age_thrice_ben_in_n_years :
  ∃ n : ℕ, Alex_current_age + n = 3 * (Ben_current_age + n) ∧ n = 11 :=
by
  sorry

end alex_age_thrice_ben_in_n_years_l147_147781


namespace minimal_circle_intersect_l147_147914

noncomputable def circle_eq := 
  ∀ (x y : ℝ), 
    (x^2 + y^2 + 4 * x + y + 1 = 0) ∧
    (x^2 + y^2 + 2 * x + 2 * y + 1 = 0) → 
    (x^2 + y^2 + (6/5) * x + (3/5) * y + 1 = 0)

theorem minimal_circle_intersect :
  circle_eq :=
by
  sorry

end minimal_circle_intersect_l147_147914


namespace problems_per_worksheet_l147_147534

theorem problems_per_worksheet (total_worksheets : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ)
    (h1 : total_worksheets = 16) (h2 : graded_worksheets = 8) (h3 : remaining_problems = 32) :
    remaining_problems / (total_worksheets - graded_worksheets) = 4 :=
by
  sorry

end problems_per_worksheet_l147_147534


namespace largest_uncovered_squares_l147_147478

theorem largest_uncovered_squares (board_size : ℕ) (total_squares : ℕ) (domino_size : ℕ) 
  (odd_property : ∀ (n : ℕ), n % 2 = 1 → (n - domino_size) % 2 = 1)
  (can_place_more : ∀ (placed_squares odd_squares : ℕ), placed_squares + domino_size ≤ total_squares → odd_squares - domino_size % 2 = 1 → odd_squares ≥ 0)
  : ∃ max_uncovered : ℕ, max_uncovered = 7 := by
  sorry

end largest_uncovered_squares_l147_147478


namespace problem1_l147_147568

noncomputable def log6_7 : ℝ := Real.logb 6 7
noncomputable def log7_6 : ℝ := Real.logb 7 6

theorem problem1 : log6_7 > log7_6 := 
by
  sorry

end problem1_l147_147568


namespace amount_c_is_1600_l147_147271

-- Given conditions
def total_money : ℕ := 2000
def ratio_b_c : (ℕ × ℕ) := (4, 16)

-- Define the total_parts based on the ratio
def total_parts := ratio_b_c.fst + ratio_b_c.snd

-- Define the value of each part
def value_per_part := total_money / total_parts

-- Calculate the amount for c
def amount_c_gets := ratio_b_c.snd * value_per_part

-- Main theorem stating the problem
theorem amount_c_is_1600 : amount_c_gets = 1600 := by
  -- Proof would go here
  sorry

end amount_c_is_1600_l147_147271


namespace team_X_played_24_games_l147_147654

def games_played_X (x : ℕ) : ℕ := x
def games_played_Y (x : ℕ) : ℕ := x + 9
def games_won_X (x : ℕ) : ℚ := 3 / 4 * x
def games_won_Y (x : ℕ) : ℚ := 2 / 3 * (x + 9)

theorem team_X_played_24_games (x : ℕ) 
  (h1 : games_won_Y x = games_won_X x + 4) : games_played_X x = 24 :=
by
  sorry

end team_X_played_24_games_l147_147654


namespace quadrilateral_parallelogram_iff_l147_147234

variable (a b c d e f MN : ℝ)

-- Define a quadrilateral as a structure with sides and diagonals 
structure Quadrilateral :=
  (a b c d e f : ℝ)

-- Define the condition: sum of squares of diagonals equals sum of squares of sides
def sum_of_squares_condition (q : Quadrilateral) : Prop :=
  q.e ^ 2 + q.f ^ 2 = q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2

-- Define what it means for a quadrilateral to be a parallelogram:
-- Midpoints of the diagonals coincide (MN = 0)
def is_parallelogram (q : Quadrilateral) (MN : ℝ) : Prop :=
  MN = 0

-- Main theorem to prove
theorem quadrilateral_parallelogram_iff (q : Quadrilateral) (MN : ℝ) :
  is_parallelogram q MN ↔ sum_of_squares_condition q :=
sorry

end quadrilateral_parallelogram_iff_l147_147234


namespace rectangle_area_l147_147616

theorem rectangle_area (length width : ℝ) 
  (h1 : width = 0.9 * length) 
  (h2 : length = 15) : 
  length * width = 202.5 := 
by
  sorry

end rectangle_area_l147_147616


namespace bob_wins_l147_147833

-- Define the notion of nim-sum used in nim-games
def nim_sum (a b : ℕ) : ℕ := Nat.xor a b

-- Define nim-values for given walls based on size
def nim_value : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| 3 => 3
| 4 => 1
| 5 => 4
| 6 => 3
| 7 => 2
| _ => 0

-- Calculate the nim-value of a given configuration
def nim_config (c : List ℕ) : ℕ :=
c.foldl (λ acc n => nim_sum acc (nim_value n)) 0

-- Prove that the configuration (7, 3, 1) gives a nim-value of 0
theorem bob_wins : nim_config [7, 3, 1] = 0 := by
  sorry

end bob_wins_l147_147833


namespace time_comparison_l147_147581

-- Definitions from the conditions
def speed_first_trip (v : ℝ) : ℝ := v
def distance_first_trip : ℝ := 80
def distance_second_trip : ℝ := 240
def speed_second_trip (v : ℝ) : ℝ := 4 * v

-- Theorem to prove
theorem time_comparison (v : ℝ) (hv : v > 0) :
  (distance_second_trip / speed_second_trip v) = (3 / 4) * (distance_first_trip / speed_first_trip v) :=
by
  -- Outline of the proof, we skip the actual steps
  sorry

end time_comparison_l147_147581


namespace angle_A_is_60_degrees_triangle_area_l147_147430

-- Define the basic setup for the triangle and its angles
variables (a b c : ℝ) -- internal angles of the triangle ABC
variables (B C : ℝ) -- sides opposite to angles b and c respectively

-- Given conditions
axiom equation_1 : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a
axiom perimeter_condition : a + b + c = 8
axiom circumradius_condition : ∃ R : ℝ, R = Real.sqrt 3

-- Question 1: Prove the measure of angle A is 60 degrees
theorem angle_A_is_60_degrees (h : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a) : 
  a = 60 :=
sorry

-- Question 2: Prove the area of triangle ABC
theorem triangle_area (h : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a)
(h_perimeter : a + b + c = 8) (h_circumradius : ∃ R : ℝ, R = Real.sqrt 3) :
  ∃ S : ℝ, S = 4 * Real.sqrt 3 / 3 :=
sorry

end angle_A_is_60_degrees_triangle_area_l147_147430


namespace cookies_per_bag_l147_147113

-- Definitions based on given conditions
def total_cookies : ℕ := 75
def number_of_bags : ℕ := 25

-- The statement of the problem
theorem cookies_per_bag : total_cookies / number_of_bags = 3 := by
  sorry

end cookies_per_bag_l147_147113


namespace price_sugar_salt_l147_147954

/-- The price of two kilograms of sugar and five kilograms of salt is $5.50. If a kilogram of sugar 
    costs $1.50, then how much is the price of three kilograms of sugar and some kilograms of salt, 
    if the total price is $5? -/
theorem price_sugar_salt 
  (price_sugar_per_kg : ℝ)
  (price_total_2kg_sugar_5kg_salt : ℝ)
  (total_price : ℝ) :
  price_sugar_per_kg = 1.50 →
  price_total_2kg_sugar_5kg_salt = 5.50 →
  total_price = 5 →
  2 * price_sugar_per_kg + 5 * (price_total_2kg_sugar_5kg_salt - 2 * price_sugar_per_kg) / 5 = 5.50 →
  3 * price_sugar_per_kg + (total_price - 3 * price_sugar_per_kg) / ((price_total_2kg_sugar_5kg_salt - 2 * price_sugar_per_kg) / 5) = 1 →
  true :=
by
  sorry

end price_sugar_salt_l147_147954


namespace binary_addition_to_hex_l147_147034

theorem binary_addition_to_hex :
  let n₁ := (0b11111111111 : ℕ)
  let n₂ := (0b11111111 : ℕ)
  n₁ + n₂ = 0x8FE :=
by {
  sorry
}

end binary_addition_to_hex_l147_147034


namespace mysterious_neighbor_is_13_l147_147328

variable (x : ℕ) (h1 : x < 15) (h2 : 2 * x * 30 = 780)

theorem mysterious_neighbor_is_13 : x = 13 :=
by {
    sorry 
}

end mysterious_neighbor_is_13_l147_147328


namespace simplify_expression_l147_147962

-- Define the question and conditions
theorem simplify_expression (x y : ℝ) (h : |x + 1| + (2 * y - 4)^2 = 0) :
  (2*x^2*y - 3*x*y) - 2*(x^2*y - x*y + 1/2*x*y^2) + x*y = 4 :=
by
  -- proof steps if needed, but currently replaced with 'sorry' to indicate proof needed
  sorry

end simplify_expression_l147_147962


namespace find_a7_l147_147172

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = -4/3 ∧ (∀ n, a (n + 2) = 1 / (a n + 1))

theorem find_a7 (a : ℕ → ℚ) (h : seq a) : a 7 = 2 :=
by
  sorry

end find_a7_l147_147172


namespace equation_solution_l147_147707

theorem equation_solution (x : ℝ) (h : 8^(Real.log 5 / Real.log 8) = 10 * x + 3) : x = 1 / 5 :=
sorry

end equation_solution_l147_147707


namespace area_ratio_of_quadrilateral_ADGJ_to_decagon_l147_147750

noncomputable def ratio_of_areas (k : ℝ) : ℝ :=
  (2 * k^2 * Real.sin (72 * Real.pi / 180)) / (5 * Real.sqrt (5 + 2 * Real.sqrt 5))

theorem area_ratio_of_quadrilateral_ADGJ_to_decagon
  (k : ℝ) :
  ∃ (n m : ℝ), m / n = ratio_of_areas k :=
  sorry

end area_ratio_of_quadrilateral_ADGJ_to_decagon_l147_147750


namespace initial_apples_l147_147490

-- Define the initial conditions
def r : Nat := 14
def s : Nat := 2 * r
def remaining : Nat := 32
def total_removed : Nat := r + s

-- The proof problem: Prove that the initial number of apples is 74
theorem initial_apples : (total_removed + remaining = 74) :=
by
  sorry

end initial_apples_l147_147490


namespace min_value_fraction_sum_l147_147120

theorem min_value_fraction_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (∃ x : ℝ, x = (1 / a + 4 / b) ∧ x = 9 / 4) :=
by
  sorry

end min_value_fraction_sum_l147_147120


namespace parabola_vertex_l147_147497

theorem parabola_vertex:
  ∀ x: ℝ, ∀ y: ℝ, (y = (1 / 2) * x ^ 2 - 4 * x + 3) → (x = 4 ∧ y = -5) :=
sorry

end parabola_vertex_l147_147497


namespace value_of_work_clothes_l147_147121

theorem value_of_work_clothes (x y : ℝ) (h1 : x + 70 = 30 * y) (h2 : x + 20 = 20 * y) : x = 80 :=
by
  sorry

end value_of_work_clothes_l147_147121


namespace maximum_area_of_enclosed_poly_l147_147650

theorem maximum_area_of_enclosed_poly (k : ℕ) : 
  ∃ (A : ℕ), (A = 4 * k + 1) :=
sorry

end maximum_area_of_enclosed_poly_l147_147650


namespace points_satisfying_inequality_l147_147777

theorem points_satisfying_inequality (x y : ℝ) :
  ( ( (x * y + 1) / (x + y) )^2 < 1) ↔ 
  ( (-1 < x ∧ x < 1) ∧ (y < -1 ∨ y > 1) ) ∨ 
  ( (x < -1 ∨ x > 1) ∧ (-1 < y ∧ y < 1) ) := 
sorry

end points_satisfying_inequality_l147_147777


namespace tetrahedron_volume_l147_147871

theorem tetrahedron_volume (a b c : ℝ)
  (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) :
  ∃ V : ℝ, 
    V = (1 / (6 * Real.sqrt 2)) * 
        Real.sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2) * (b^2 + c^2 - a^2)) :=
sorry

end tetrahedron_volume_l147_147871


namespace sum_two_numbers_l147_147074

theorem sum_two_numbers (x y : ℝ) (h₁ : x * y = 16) (h₂ : 1 / x = 3 * (1 / y)) : x + y = 16 * Real.sqrt 3 / 3 :=
by
  -- Proof follows the steps outlined in the solution, but this is where the proof ends for now.
  sorry

end sum_two_numbers_l147_147074


namespace digit_makes_57A2_divisible_by_9_l147_147228

theorem digit_makes_57A2_divisible_by_9 (A : ℕ) (h : 0 ≤ A ∧ A ≤ 9) : 
  (5 + 7 + A + 2) % 9 = 0 ↔ A = 4 :=
by
  sorry

end digit_makes_57A2_divisible_by_9_l147_147228


namespace sum_four_least_tau_equals_eight_l147_147056

def tau (n : ℕ) : ℕ := n.divisors.card

theorem sum_four_least_tau_equals_eight :
  ∃ n1 n2 n3 n4 : ℕ, 
    tau n1 + tau (n1 + 1) = 8 ∧ 
    tau n2 + tau (n2 + 1) = 8 ∧
    tau n3 + tau (n3 + 1) = 8 ∧
    tau n4 + tau (n4 + 1) = 8 ∧
    n1 + n2 + n3 + n4 = 80 := 
sorry

end sum_four_least_tau_equals_eight_l147_147056


namespace rational_cubes_rational_values_l147_147319

theorem rational_cubes_rational_values {a b : ℝ} (ha : 0 < a) (hb : 0 < b) 
  (hab : a + b = 1) (ha3 : ∃ r : ℚ, a^3 = r) (hb3 : ∃ s : ℚ, b^3 = s) : 
  ∃ r s : ℚ, a = r ∧ b = s :=
sorry

end rational_cubes_rational_values_l147_147319


namespace daily_rate_first_week_l147_147298

-- Definitions from given conditions
variable (x : ℝ) (h1 : ∀ y : ℝ, 0 ≤ y)
def cost_first_week := 7 * x
def additional_days_cost := 16 * 14
def total_cost := cost_first_week + additional_days_cost

-- Theorem to solve the problem
theorem daily_rate_first_week (h : total_cost = 350) : x = 18 :=
sorry

end daily_rate_first_week_l147_147298


namespace lees_friend_initial_money_l147_147583

theorem lees_friend_initial_money (lee_initial_money friend_initial_money total_cost change : ℕ) 
  (h1 : lee_initial_money = 10) 
  (h2 : total_cost = 15) 
  (h3 : change = 3) 
  (h4 : (lee_initial_money + friend_initial_money) - total_cost = change) : 
  friend_initial_money = 8 := by
  sorry

end lees_friend_initial_money_l147_147583


namespace equation_descr_circle_l147_147078

theorem equation_descr_circle : ∀ (x y : ℝ), (x - 0) ^ 2 + (y - 0) ^ 2 = 25 → ∃ (c : ℝ × ℝ) (r : ℝ), c = (0, 0) ∧ r = 5 ∧ ∀ (p : ℝ × ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
by
  sorry

end equation_descr_circle_l147_147078


namespace volume_of_cube_with_surface_area_l147_147070

theorem volume_of_cube_with_surface_area (S : ℝ) (hS : S = 294) : 
  ∃ V : ℝ, V = 343 :=
by
  let s := (S / 6).sqrt
  have hs : s = 7 := by sorry
  use s ^ 3
  simp [hs]
  exact sorry

end volume_of_cube_with_surface_area_l147_147070


namespace tiles_on_square_area_l147_147812

theorem tiles_on_square_area (n : ℕ) (h1 : 2 * n - 1 = 25) : n ^ 2 = 169 :=
by
  sorry

end tiles_on_square_area_l147_147812


namespace base_four_odd_last_digit_l147_147867

theorem base_four_odd_last_digit :
  ∃ b : ℕ, b = 4 ∧ (b^4 ≤ 625 ∧ 625 < b^5) ∧ (625 % b % 2 = 1) :=
by
  sorry

end base_four_odd_last_digit_l147_147867


namespace Tina_profit_correct_l147_147125

theorem Tina_profit_correct :
  ∀ (price_per_book cost_per_book books_per_customer total_customers : ℕ),
  price_per_book = 20 →
  cost_per_book = 5 →
  books_per_customer = 2 →
  total_customers = 4 →
  (price_per_book * (books_per_customer * total_customers) - 
   cost_per_book * (books_per_customer * total_customers) = 120) :=
by
  intros price_per_book cost_per_book books_per_customer total_customers
  sorry

end Tina_profit_correct_l147_147125


namespace area_triangle_ABC_l147_147468

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_triangle_ABC :
  area_of_triangle (2, 4) (-1, 1) (1, -1) = 6 :=
by
  sorry

end area_triangle_ABC_l147_147468


namespace fraction_meaningful_l147_147743

theorem fraction_meaningful (a : ℝ) : (a + 3 ≠ 0) ↔ (a ≠ -3) :=
by
  sorry

end fraction_meaningful_l147_147743


namespace area_of_rectangle_l147_147457

-- Define the conditions
def width : ℕ := 6
def perimeter : ℕ := 28

-- Define the theorem statement
theorem area_of_rectangle (w : ℕ) (p : ℕ) (h_width : w = width) (h_perimeter : p = perimeter) :
  ∃ l : ℕ, (2 * (l + w) = p) → (l * w = 48) :=
by
  use 8
  intro h
  simp only [h_width, h_perimeter] at h
  sorry

end area_of_rectangle_l147_147457


namespace rectangle_sides_l147_147485

def side_length_square : ℝ := 18
def num_rectangles : ℕ := 5

variable (a b : ℝ)
variables (h1 : 2 * (a + b) = side_length_square) (h2 : 3 * a = side_length_square)

theorem rectangle_sides : a = 6 ∧ b = 3 :=
by {
  sorry
}

end rectangle_sides_l147_147485


namespace vessel_capacity_proof_l147_147423

variable (V1_capacity : ℕ) (V2_capacity : ℕ) (total_mixture : ℕ) (final_vessel_capacity : ℕ)
variable (A1_percentage : ℕ) (A2_percentage : ℕ)

theorem vessel_capacity_proof
  (h1 : V1_capacity = 2)
  (h2 : A1_percentage = 35)
  (h3 : V2_capacity = 6)
  (h4 : A2_percentage = 50)
  (h5 : total_mixture = 8)
  (h6 : final_vessel_capacity = 10)
  : final_vessel_capacity = 10 := 
by
  sorry

end vessel_capacity_proof_l147_147423


namespace range_of_n_l147_147480

noncomputable def parabola (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b

variable {a b n y1 y2 : ℝ}

theorem range_of_n (h_a : a > 0) 
  (hA : parabola a b (2*n + 3) = y1) 
  (hB : parabola a b (n - 1) = y2)
  (h_sym : y1 < y2) 
  (h_opposite_sides : (2*n + 3 - 1) * (n - 1 - 1) < 0) :
  -1 < n ∧ n < 0 :=
sorry

end range_of_n_l147_147480


namespace points_per_game_l147_147161

theorem points_per_game (total_points : ℝ) (num_games : ℝ) (h1 : total_points = 120.0) (h2 : num_games = 10.0) : (total_points / num_games) = 12.0 :=
by 
  rw [h1, h2]
  norm_num
  -- sorry


end points_per_game_l147_147161


namespace goal_l147_147774

-- Definitions based on the conditions
def condition1 (x : ℕ) : Prop := 10 < x ∧ x < 17
def condition2 (x : ℕ) : Prop := 11 < x ∧ x < 18
def condition3 (x : ℕ) : Prop := x % 2 = 1

-- Definition for checking if exactly two conditions hold
def exactly_two_holds (h1 h2 h3 : Prop) : Prop :=
  (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)

-- Main goal: find x where exactly two out of three conditions hold
theorem goal (x : ℕ) : exactly_two_holds (condition1 x) (condition2 x) (condition3 x) ↔ 
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end goal_l147_147774


namespace longest_train_length_l147_147672

theorem longest_train_length :
  ∀ (speedA : ℝ) (timeA : ℝ) (speedB : ℝ) (timeB : ℝ) (speedC : ℝ) (timeC : ℝ),
  speedA = 60 * (5 / 18) → timeA = 5 →
  speedB = 80 * (5 / 18) → timeB = 7 →
  speedC = 50 * (5 / 18) → timeC = 9 →
  speedB * timeB > speedA * timeA ∧ speedB * timeB > speedC * timeC ∧ speedB * timeB = 155.54 := by
  sorry

end longest_train_length_l147_147672


namespace mean_score_l147_147955

variable (mean stddev : ℝ)

-- Conditions
axiom condition1 : 42 = mean - 5 * stddev
axiom condition2 : 67 = mean + 2.5 * stddev

theorem mean_score : mean = 58.67 := 
by 
  -- You would need to provide proof here
  sorry

end mean_score_l147_147955


namespace prime_solution_unique_l147_147813

theorem prime_solution_unique {x y : ℕ} 
  (hx : Nat.Prime x)
  (hy : Nat.Prime y)
  (h : x ^ y - y ^ x = x * y ^ 2 - 19) :
  (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) :=
sorry

end prime_solution_unique_l147_147813


namespace find_point_P_l147_147431

structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def vector (P Q : Point) : Point :=
⟨Q.x - P.x, Q.y - P.y⟩

def magnitude_ratio (P A B : Point) (r : ℝ) : Prop :=
  let AP := vector A P
  let PB := vector P B
  (AP.x, AP.y) = (r * PB.x, r * PB.y)

theorem find_point_P (P : Point) : 
  magnitude_ratio P A B (4/3) → (P.x = 10 ∧ P.y = -21) :=
sorry

end find_point_P_l147_147431


namespace evaluate_expression_l147_147191

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) :
  (a^4 + b^4) / (a^2 - a * b + b^2) = 97 / 7 := by
  sorry

example : (3^4 + 2^4) / (3^2 - 3 * 2 + 2^2) = 97 / 7 := evaluate_expression 3 2 rfl rfl

end evaluate_expression_l147_147191


namespace number_of_rabbits_l147_147385

-- Defining the problem conditions
variables (x y : ℕ)
axiom heads_condition : x + y = 40
axiom legs_condition : 4 * x = 10 * 2 * y - 8

--  Prove the number of rabbits is 33
theorem number_of_rabbits : x = 33 :=
by
  sorry

end number_of_rabbits_l147_147385


namespace crow_eating_time_l147_147119

/-- 
We are given that a crow eats a fifth of the total number of nuts in 6 hours.
We are to prove that it will take the crow 7.5 hours to finish a quarter of the nuts.
-/
theorem crow_eating_time (h : (1/5:ℚ) * t = 6) : (1/4) * t = 7.5 := 
by 
  -- Skipping the proof
  sorry

end crow_eating_time_l147_147119


namespace remainder_sum_modulo_l147_147492

theorem remainder_sum_modulo :
  (9156 + 9157 + 9158 + 9159 + 9160) % 9 = 7 :=
by
sorry

end remainder_sum_modulo_l147_147492


namespace tiling_vertex_squares_octagons_l147_147002

theorem tiling_vertex_squares_octagons (m n : ℕ) 
  (h1 : 135 * n + 90 * m = 360) : 
  m = 1 ∧ n = 2 :=
by
  sorry

end tiling_vertex_squares_octagons_l147_147002


namespace problem_1_problem_2_l147_147621

def op (x y : ℝ) : ℝ := 3 * x - y

theorem problem_1 (x : ℝ) : op x (op 2 3) = 1 ↔ x = 4 / 3 := by
  -- definitions from conditions
  let def_op_2_3 := op 2 3
  let eq1 := op x def_op_2_3
  -- problem in lean representation
  sorry

theorem problem_2 (x : ℝ) : op (x ^ 2) 2 = 10 ↔ x = 2 ∨ x = -2 := by
  -- problem in lean representation
  sorry

end problem_1_problem_2_l147_147621


namespace increasing_sequence_a_range_l147_147896

theorem increasing_sequence_a_range (a : ℝ) (a_seq : ℕ → ℝ) (h_def : ∀ n, a_seq n = 
  if n ≤ 2 then a * n^2 - ((7 / 8) * a + 17 / 4) * n + 17 / 2
  else a ^ n) : 
  (∀ n, a_seq n < a_seq (n + 1)) → a > 2 :=
by
  sorry

end increasing_sequence_a_range_l147_147896


namespace angle_measure_l147_147242

theorem angle_measure (x : ℝ) (h1 : 180 - x = 6 * (90 - x)) : x = 72 := by
  sorry

end angle_measure_l147_147242


namespace cube_surface_area_l147_147380

noncomputable def volume_of_cube (s : ℝ) := s ^ 3
noncomputable def surface_area_of_cube (s : ℝ) := 6 * (s ^ 2)

theorem cube_surface_area (s : ℝ) (h : volume_of_cube s = 1728) : surface_area_of_cube s = 864 :=
  sorry

end cube_surface_area_l147_147380


namespace number_of_unique_products_l147_147272

-- Define the sets a and b
def setA : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}
def setB : Set ℕ := {2, 4, 6, 19, 21, 24, 27, 31, 35}

-- Define the number of unique products
def numUniqueProducts : ℕ := 405

-- Statement that needs to be proved
theorem number_of_unique_products :
  (∀ A1 ∈ setA, ∀ B ∈ setB, ∀ A2 ∈ setA, ∃ p, p = A1 * B * A2) ∧ 
  (∃ count, count = 45 * 9) ∧ 
  (∃ result, result = numUniqueProducts) :=
  by {
    sorry
  }

end number_of_unique_products_l147_147272


namespace math_problem_l147_147178

variable (a b c : ℝ)

theorem math_problem (h1 : -10 ≤ a ∧ a < 0) (h2 : 0 < a ∧ a < b ∧ b < c) : 
  (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) :=
by
  sorry

end math_problem_l147_147178


namespace calc_expression_l147_147100

theorem calc_expression : 3 * 3^4 - 9^32 / 9^30 = 162 := by
  -- We would provide the proof here, but skipping with sorry
  sorry

end calc_expression_l147_147100


namespace total_arrangements_l147_147312

-- Defining the selection and arrangement problem conditions
def select_and_arrange (n m : ℕ) : ℕ :=
  Nat.choose n m * Nat.factorial m

-- Specifying the specific problem's constraints and results
theorem total_arrangements : select_and_arrange 8 2 * select_and_arrange 6 2 = 60 := by
  -- Proof omitted
  sorry

end total_arrangements_l147_147312


namespace find_same_color_integers_l147_147745

variable (Color : Type) (red blue green yellow : Color)

theorem find_same_color_integers
  (color : ℤ → Color)
  (m n : ℤ)
  (hm : Odd m)
  (hn : Odd n)
  (h_not_zero : m + n ≠ 0) :
  ∃ a b : ℤ, color a = color b ∧ (a - b = m ∨ a - b = n ∨ a - b = m + n ∨ a - b = m - n) :=
sorry

end find_same_color_integers_l147_147745


namespace sandy_shopping_l147_147247

theorem sandy_shopping (T : ℝ) (h : 0.70 * T = 217) : T = 310 := sorry

end sandy_shopping_l147_147247


namespace sequence_result_l147_147550

theorem sequence_result (initial_value : ℕ) (total_steps : ℕ) 
    (net_effect_one_cycle : ℕ) (steps_per_cycle : ℕ) : 
    initial_value = 100 ∧ total_steps = 26 ∧ 
    net_effect_one_cycle = (15 - 12 + 3) ∧ steps_per_cycle = 3 
    → 
    ∀ (resulting_value : ℕ), resulting_value = 151 :=
by
  sorry

end sequence_result_l147_147550


namespace lucy_snowballs_l147_147702

theorem lucy_snowballs : ∀ (c l : ℕ), c = l + 31 → c = 50 → l = 19 :=
by
  intros c l h1 h2
  sorry

end lucy_snowballs_l147_147702


namespace cube_faces_opposite_10_is_8_l147_147629

theorem cube_faces_opposite_10_is_8 (nums : Finset ℕ) (h_nums : nums = {6, 7, 8, 9, 10, 11})
  (sum_lateral_first : ℕ) (h_sum_lateral_first : sum_lateral_first = 36)
  (sum_lateral_second : ℕ) (h_sum_lateral_second : sum_lateral_second = 33)
  (faces_opposite_10 : ℕ) (h_faces_opposite_10 : faces_opposite_10 ∈ nums) :
  faces_opposite_10 = 8 :=
by
  sorry

end cube_faces_opposite_10_is_8_l147_147629


namespace race_problem_l147_147359

theorem race_problem 
  (A B C : ℝ) 
  (h1 : A = 100) 
  (h2 : B = 100 - x) 
  (h3 : C = 72) 
  (h4 : B = C + 4)
  : x = 24 := 
by 
  sorry

end race_problem_l147_147359


namespace radius_of_semi_circle_l147_147493

-- Given definitions and conditions
def perimeter : ℝ := 33.934511513692634
def pi_approx : ℝ := 3.141592653589793

-- The formula for the perimeter of a semi-circle
def semi_circle_perimeter (r : ℝ) : ℝ := pi_approx * r + 2 * r

-- The theorem we want to prove
theorem radius_of_semi_circle (r : ℝ) (h: semi_circle_perimeter r = perimeter) : r = 6.6 :=
sorry

end radius_of_semi_circle_l147_147493


namespace negation_of_real_root_proposition_l147_147079

theorem negation_of_real_root_proposition :
  (¬ ∃ m : ℝ, ∃ (x : ℝ), x^2 + m * x + 1 = 0) ↔ (∀ m : ℝ, ∀ (x : ℝ), x^2 + m * x + 1 ≠ 0) :=
by
  sorry

end negation_of_real_root_proposition_l147_147079


namespace uniform_prob_correct_l147_147148

noncomputable def uniform_prob_within_interval 
  (α β γ δ : ℝ) 
  (h₁ : α ≤ β) 
  (h₂ : α ≤ γ) 
  (h₃ : γ < δ) 
  (h₄ : δ ≤ β) : ℝ :=
  (δ - γ) / (β - α)

theorem uniform_prob_correct 
  (α β γ δ : ℝ) 
  (hαβ : α ≤ β) 
  (hαγ : α ≤ γ) 
  (hγδ : γ < δ) 
  (hδβ : δ ≤ β) :
  uniform_prob_within_interval α β γ δ hαβ hαγ hγδ hδβ = (δ - γ) / (β - α) := sorry

end uniform_prob_correct_l147_147148


namespace abc_sub_c_minus_2023_eq_2023_l147_147638

theorem abc_sub_c_minus_2023_eq_2023 (a b c : ℝ) (h : a * b = 1) : 
  a * b * c - (c - 2023) = 2023 := 
by sorry

end abc_sub_c_minus_2023_eq_2023_l147_147638


namespace students_exceed_pets_by_70_l147_147118

theorem students_exceed_pets_by_70 :
  let n_classrooms := 5
  let students_per_classroom := 22
  let rabbits_per_classroom := 3
  let hamsters_per_classroom := 5
  let total_students := students_per_classroom * n_classrooms
  let total_rabbits := rabbits_per_classroom * n_classrooms
  let total_hamsters := hamsters_per_classroom * n_classrooms
  let total_pets := total_rabbits + total_hamsters
  total_students - total_pets = 70 :=
  by
    sorry

end students_exceed_pets_by_70_l147_147118


namespace number_of_piles_l147_147157

-- Defining the number of walnuts in total
def total_walnuts : Nat := 55

-- Defining the number of walnuts in the first pile
def first_pile_walnuts : Nat := 7

-- Defining the number of walnuts in each of the rest of the piles
def other_pile_walnuts : Nat := 12

-- The proposition we want to prove
theorem number_of_piles (n : Nat) :
  (n > 1) →
  (other_pile_walnuts * (n - 1) + first_pile_walnuts = total_walnuts) → n = 5 :=
sorry

end number_of_piles_l147_147157


namespace find_n_l147_147130

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 8): n = 24 := by
  sorry

end find_n_l147_147130


namespace JackEmails_l147_147372

theorem JackEmails (E : ℕ) (h1 : 10 = E + 7) : E = 3 :=
by
  sorry

end JackEmails_l147_147372


namespace tower_height_l147_147809

theorem tower_height (h d : ℝ) 
  (tan_30_eq : Real.tan (Real.pi / 6) = h / d)
  (tan_45_eq : Real.tan (Real.pi / 4) = h / (d - 20)) :
  h = 20 * Real.sqrt 3 :=
by
  sorry

end tower_height_l147_147809


namespace solve_diamond_l147_147577

theorem solve_diamond : 
  (∃ (Diamond : ℤ), Diamond * 5 + 3 = Diamond * 6 + 2) →
  (∃ (Diamond : ℤ), Diamond = 1) :=
by
  sorry

end solve_diamond_l147_147577


namespace complement_intersection_eq_l147_147660

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_eq :
  U \ (A ∩ B) = {1, 4, 5} := by
  sorry

end complement_intersection_eq_l147_147660


namespace letter_lock_rings_l147_147776

theorem letter_lock_rings (n : ℕ) (h : n^3 - 1 ≤ 215) : n = 6 :=
by { sorry }

end letter_lock_rings_l147_147776


namespace ratio_water_to_orange_juice_l147_147207

variable (O W : ℝ)

-- Conditions:
-- 1. Amount of orange juice is O for both days.
-- 2. Amount of water is W on the first day and 2W on the second day.
-- 3. Price per glass is $0.60 on the first day and $0.40 on the second day.

theorem ratio_water_to_orange_juice 
  (h : (O + W) * 0.60 = (O + 2 * W) * 0.40) : 
  W / O = 1 := 
by 
  -- The proof is skipped
  sorry

end ratio_water_to_orange_juice_l147_147207


namespace cylinder_volume_ratio_l147_147160

theorem cylinder_volume_ratio (h_C r_D : ℝ) (V_C V_D : ℝ) :
  h_C = 3 * r_D →
  r_D = h_C →
  V_C = 3 * V_D →
  V_C = (1 / 9) * π * h_C^3 :=
by
  sorry

end cylinder_volume_ratio_l147_147160


namespace percent_of_x_is_y_l147_147426

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y / x = 0.25 :=
by
  -- proof omitted
  sorry

end percent_of_x_is_y_l147_147426


namespace john_buys_1000_balloons_l147_147092

-- Define conditions
def balloon_volume : ℕ := 10
def tank_volume : ℕ := 500
def num_tanks : ℕ := 20

-- Define the total volume of gas
def total_gas_volume : ℕ := num_tanks * tank_volume

-- Define the number of balloons
def num_balloons : ℕ := total_gas_volume / balloon_volume

-- Prove that the number of balloons is 1,000
theorem john_buys_1000_balloons : num_balloons = 1000 := by
  sorry

end john_buys_1000_balloons_l147_147092


namespace quotient_of_N_div_3_l147_147940

-- Define the number N
def N : ℕ := 7 * 12 + 4

-- Statement we need to prove
theorem quotient_of_N_div_3 : N / 3 = 29 :=
by
  sorry

end quotient_of_N_div_3_l147_147940


namespace determine_m_l147_147139

def f (x : ℝ) := 5 * x^2 + 3 * x + 7
def g (x : ℝ) (m : ℝ) := 2 * x^2 - m * x + 1

theorem determine_m (m : ℝ) : f 5 - g 5 m = 55 → m = -7 :=
by
  unfold f
  unfold g
  sorry

end determine_m_l147_147139


namespace not_subset_T_to_S_l147_147968

def is_odd (x : ℤ) : Prop := ∃ n : ℤ, x = 2 * n + 1
def is_of_form_4k_plus_1 (y : ℤ) : Prop := ∃ k : ℤ, y = 4 * k + 1

theorem not_subset_T_to_S :
  ¬ (∀ y, is_of_form_4k_plus_1 y → is_odd y) :=
sorry

end not_subset_T_to_S_l147_147968


namespace determine_n_l147_147614

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem determine_n (n : ℕ) (h1 : binom n 2 + binom n 1 = 6) : n = 3 := 
by
  sorry

end determine_n_l147_147614


namespace length_of_wall_correct_l147_147950

noncomputable def length_of_wall (s : ℝ) (w : ℝ) : ℝ :=
  let area_mirror := s * s
  let area_wall := 2 * area_mirror
  area_wall / w

theorem length_of_wall_correct : length_of_wall 18 32 = 20.25 :=
by
  -- This is the place for proof which is omitted deliberately
  sorry

end length_of_wall_correct_l147_147950


namespace garden_borders_length_l147_147015

theorem garden_borders_length 
  (a b c d e : ℕ)
  (h1 : 6 * 7 = a^2 + b^2 + c^2 + d^2 + e^2)
  (h2 : a * a + b * b + c * c + d * d + e * e = 42) -- This is analogous to the condition
    
: 15 = (4*a + 4*b + 4*c + 4*d + 4*e - 2*(6 + 7)) / 2 :=
by sorry

end garden_borders_length_l147_147015


namespace gcd_90_405_l147_147402

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l147_147402


namespace number_of_perfect_square_factors_l147_147903

theorem number_of_perfect_square_factors :
  let n := (2^14) * (3^9) * (5^20)
  ∃ (count : ℕ), 
  (∀ (a : ℕ) (h : a ∣ n), (∃ k, a = k^2) → true) →
  count = 440 :=
by
  sorry

end number_of_perfect_square_factors_l147_147903


namespace new_daily_average_wage_l147_147321

theorem new_daily_average_wage (x : ℝ) : 
  (∀ y : ℝ, 25 - x = y) → 
  (∀ z : ℝ, 20 * (25 - x) = 30 * (10)) → 
  x = 10 :=
by
  intro h1 h2
  sorry

end new_daily_average_wage_l147_147321


namespace compare_abc_l147_147814

noncomputable def a : ℝ := 1 / Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.exp 0.5
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > c ∧ c > a := by
  sorry

end compare_abc_l147_147814


namespace max_difference_two_digit_numbers_l147_147155

theorem max_difference_two_digit_numbers (A B : ℤ) (hA : 10 ≤ A ∧ A ≤ 99) (hB : 10 ≤ B ∧ B ≤ 99) (h : 2 * A * 3 = 2 * B * 7) : 
  56 ≤ A - B :=
sorry

end max_difference_two_digit_numbers_l147_147155


namespace who_stole_the_pan_l147_147761

def Frog_statement := "Lackey-Lech stole the pan"
def LackeyLech_statement := "I did not steal any pan"
def KnaveOfHearts_statement := "I stole the pan"

axiom no_more_than_one_liar : ∀ (frog_is_lying : Prop) (lackey_lech_is_lying : Prop) (knave_of_hearts_is_lying : Prop), (frog_is_lying → ¬ lackey_lech_is_lying) ∧ (frog_is_lying → ¬ knave_of_hearts_is_lying) ∧ (lackey_lech_is_lying → ¬ knave_of_hearts_is_lying)

theorem who_stole_the_pan : KnaveOfHearts_statement = "I stole the pan" :=
sorry

end who_stole_the_pan_l147_147761


namespace am_gm_inequality_l147_147226

theorem am_gm_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (x + y + z)^2 / 3 ≥ x * Real.sqrt (y * z) + y * Real.sqrt (z * x) + z * Real.sqrt (x * y) := 
by sorry

end am_gm_inequality_l147_147226


namespace fractional_sides_l147_147491

variable {F : ℕ} -- Number of fractional sides
variable {D : ℕ} -- Number of diagonals

theorem fractional_sides (h1 : D = 2 * F) (h2 : D = F * (F - 3) / 2) : F = 7 :=
by
  sorry

end fractional_sides_l147_147491


namespace car_travel_distance_l147_147398

-- Define the conditions
def speed : ℝ := 23
def time : ℝ := 3

-- Define the formula for distance
def distance_traveled (s : ℝ) (t : ℝ) : ℝ := s * t

-- State the theorem to prove the distance the car traveled
theorem car_travel_distance : distance_traveled speed time = 69 :=
by
  -- The proof would normally go here, but we're skipping it as per the instructions
  sorry

end car_travel_distance_l147_147398


namespace avg_daily_distance_third_dog_summer_l147_147891

theorem avg_daily_distance_third_dog_summer :
  ∀ (total_days weekends miles_walked_weekday : ℕ), 
    total_days = 30 → weekends = 8 → miles_walked_weekday = 3 →
    (66 / 30 : ℝ) = 2.2 :=
by
  intros total_days weekends miles_walked_weekday h_total h_weekends h_walked
  -- proof goes here
  sorry

end avg_daily_distance_third_dog_summer_l147_147891


namespace order_of_abc_l147_147222

noncomputable def a : ℝ := (1 / 3) * Real.logb 2 (1 / 4)
noncomputable def b : ℝ := 1 - Real.logb 2 3
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 6)

theorem order_of_abc : c < a ∧ a < b := by
  sorry

end order_of_abc_l147_147222


namespace least_number_subtracted_divisible_17_l147_147655

theorem least_number_subtracted_divisible_17 :
  ∃ n : ℕ, 165826 - n % 17 = 0 ∧ n = 12 :=
by
  use 12
  sorry  -- Proof will go here.

end least_number_subtracted_divisible_17_l147_147655


namespace nonoverlapping_area_difference_l147_147471

theorem nonoverlapping_area_difference :
  let radius := 3
  let side := 2
  let circle_area := Real.pi * radius^2
  let square_area := side^2
  ∃ (x : ℝ), (circle_area - x) - (square_area - x) = 9 * Real.pi - 4 :=
by
  sorry

end nonoverlapping_area_difference_l147_147471


namespace polygon_with_45_deg_exterior_angle_is_eight_gon_l147_147536

theorem polygon_with_45_deg_exterior_angle_is_eight_gon
  (each_exterior_angle : ℝ) (h1 : each_exterior_angle = 45) 
  (sum_exterior_angles : ℝ) (h2 : sum_exterior_angles = 360) :
  ∃ (n : ℕ), n = 8 :=
by
  sorry

end polygon_with_45_deg_exterior_angle_is_eight_gon_l147_147536


namespace odd_function_f_x_pos_l147_147875

variable (f : ℝ → ℝ)

theorem odd_function_f_x_pos {x : ℝ} (h1 : ∀ x < 0, f x = x^2 + x)
  (h2 : ∀ x, f x = -f (-x)) (hx : 0 < x) :
  f x = -x^2 + x := by
  sorry

end odd_function_f_x_pos_l147_147875


namespace range_of_slope_ellipse_chord_l147_147689

theorem range_of_slope_ellipse_chord :
  ∀ (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ),
    (x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2) →
    (x₁^2 + y₁^2 / 4 = 1 ∧ x₂^2 + y₂^2 / 4 = 1) →
    ((1 / 2) ≤ y₀ ∧ y₀ ≤ 1) →
    (-4 ≤ -2 / y₀ ∧ -2 / y₀ ≤ -2) :=
by
  sorry

end range_of_slope_ellipse_chord_l147_147689


namespace cylinder_properties_l147_147799

theorem cylinder_properties (h r : ℝ) (h_eq : h = 15) (r_eq : r = 5) :
  let total_surface_area := 2 * Real.pi * r^2 + 2 * Real.pi * r * h
  let volume := Real.pi * r^2 * h
  total_surface_area = 200 * Real.pi ∧ volume = 375 * Real.pi :=
by
  sorry

end cylinder_properties_l147_147799


namespace shaded_quilt_fraction_l147_147625

-- Define the basic structure of the problem using conditions from step a

def is_unit_square (s : ℕ) : Prop := s = 1

def grid_size : ℕ := 4
def total_squares : ℕ := grid_size * grid_size

def shaded_squares : ℕ := 2
def half_shaded_squares : ℕ := 4

def fraction_shaded (shaded: ℕ) (total: ℕ) : ℚ := shaded / total

theorem shaded_quilt_fraction :
  fraction_shaded (shaded_squares + half_shaded_squares / 2) total_squares = 1 / 4 :=
by
  sorry

end shaded_quilt_fraction_l147_147625


namespace problem_statement_l147_147123

def horse_lap_times : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Least common multiple of a set of numbers
noncomputable def LCM_set (s : List ℕ) : ℕ :=
s.foldl LCM 1

-- Calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem problem_statement :
  let T := LCM_set [2, 3, 5, 7, 11, 13]
  sum_of_digits T = 6 := by
  sorry

end problem_statement_l147_147123


namespace greatest_consecutive_integers_sum_120_l147_147374

def sum_of_consecutive_integers (n : ℤ) (a : ℤ) : ℤ :=
  n * (2 * a + n - 1) / 2

theorem greatest_consecutive_integers_sum_120 (N : ℤ) (a : ℤ) (h1 : sum_of_consecutive_integers N a = 120) : N ≤ 240 :=
by {
  -- Here we would provide the proof, but it's omitted with 'sorry'.
  sorry
}

end greatest_consecutive_integers_sum_120_l147_147374


namespace total_ninja_stars_l147_147705

variable (e c j : ℕ)
variable (H1 : e = 4) -- Eric has 4 ninja throwing stars
variable (H2 : c = 2 * e) -- Chad has twice as many ninja throwing stars as Eric
variable (H3 : j = c - 2) -- Chad sells 2 ninja stars to Jeff
variable (H4 : j = 6) -- Jeff now has 6 ninja throwing stars

theorem total_ninja_stars :
  e + (c - 2) + 6 = 16 :=
by
  sorry

end total_ninja_stars_l147_147705


namespace height_difference_l147_147567

-- Define the heights of Eiffel Tower and Burj Khalifa as constants
def eiffelTowerHeight : ℕ := 324
def burjKhalifaHeight : ℕ := 830

-- Define the statement that needs to be proven
theorem height_difference : burjKhalifaHeight - eiffelTowerHeight = 506 := by
  sorry

end height_difference_l147_147567


namespace min_tablets_to_get_two_each_l147_147704

def least_tablets_to_ensure_two_each (A B : ℕ) (A_eq : A = 10) (B_eq : B = 10) : ℕ :=
  if A ≥ 2 ∧ B ≥ 2 then 4 else 12

theorem min_tablets_to_get_two_each :
  least_tablets_to_ensure_two_each 10 10 rfl rfl = 12 :=
by
  sorry

end min_tablets_to_get_two_each_l147_147704


namespace min_value_of_f_inequality_for_a_b_l147_147012

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 2)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 3 := by
  intro x
  sorry

theorem inequality_for_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : 1/a + 1/b = Real.sqrt 3) : 
  1/a^2 + 2/b^2 ≥ 2 := by
  sorry

end min_value_of_f_inequality_for_a_b_l147_147012


namespace sphere_radius_vol_eq_area_l147_147985

noncomputable def volume (r : ℝ) : ℝ := (4/3) * Real.pi * r ^ 3
noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r ^ 2

theorem sphere_radius_vol_eq_area (r : ℝ) :
  volume r = surface_area r → r = 3 :=
by
  sorry

end sphere_radius_vol_eq_area_l147_147985


namespace number_of_alligators_l147_147613

theorem number_of_alligators (A : ℕ) 
  (num_snakes : ℕ := 18) 
  (total_eyes : ℕ := 56) 
  (eyes_per_snake : ℕ := 2) 
  (eyes_per_alligator : ℕ := 2) 
  (snakes_eyes : ℕ := num_snakes * eyes_per_snake) 
  (alligators_eyes : ℕ := A * eyes_per_alligator) 
  (total_animals_eyes : ℕ := snakes_eyes + alligators_eyes) 
  (total_eyes_eq : total_animals_eyes = total_eyes) 
: A = 10 :=
by 
  sorry

end number_of_alligators_l147_147613


namespace midpoint_chord_hyperbola_l147_147144

-- Definitions to use in our statement
variables (a b x y : ℝ)
def ellipse : Prop := (x^2)/(a^2) + (y^2)/(b^2) = 1
def line_ellipse : Prop := x / (a^2) + y / (b^2) = 0
def hyperbola : Prop := (x^2)/(a^2) - (y^2)/(b^2) = 1
def line_hyperbola : Prop := x / (a^2) - y / (b^2) = 0

-- The theorem to prove
theorem midpoint_chord_hyperbola (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (x y : ℝ) 
    (h_ellipse : ellipse a b x y)
    (h_line_ellipse : line_ellipse a b x y)
    (h_hyperbola : hyperbola a b x y) :
    line_hyperbola a b x y :=
sorry

end midpoint_chord_hyperbola_l147_147144


namespace toms_remaining_speed_l147_147241

-- Defining the constants and conditions
def total_distance : ℝ := 100
def first_leg_distance : ℝ := 50
def first_leg_speed : ℝ := 20
def avg_speed : ℝ := 28.571428571428573

-- Proving Tom's speed during the remaining part of the trip
theorem toms_remaining_speed :
  ∃ (remaining_leg_speed : ℝ),
    (remaining_leg_speed = 50) ∧
    (total_distance = first_leg_distance + 50) ∧
    ((first_leg_distance / first_leg_speed + 50 / remaining_leg_speed) = total_distance / avg_speed) :=
by
  sorry

end toms_remaining_speed_l147_147241


namespace find_y_when_x_is_1_l147_147591

theorem find_y_when_x_is_1 
  (k : ℝ) 
  (h1 : ∀ y, x = k / y^2) 
  (h2 : x = 1) 
  (h3 : x = 0.1111111111111111) 
  (y : ℝ) 
  (hy : y = 6) 
  (hx_k : k = 0.1111111111111111 * 36) :
  y = 2 := sorry

end find_y_when_x_is_1_l147_147591


namespace two_sin_cos_75_eq_half_l147_147066

noncomputable def two_sin_cos_of_75_deg : ℝ :=
  2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180)

theorem two_sin_cos_75_eq_half : two_sin_cos_of_75_deg = 1 / 2 :=
by
  -- The steps to prove this theorem are omitted deliberately
  sorry

end two_sin_cos_75_eq_half_l147_147066


namespace polygon_number_of_sides_l147_147736

theorem polygon_number_of_sides (h : ∀ (n : ℕ), (360 : ℝ) / (n : ℝ) = 1) : 
  360 = (1:ℝ) :=
  sorry

end polygon_number_of_sides_l147_147736


namespace interest_rate_second_part_l147_147827

theorem interest_rate_second_part 
    (total_investment : ℝ) 
    (annual_interest : ℝ) 
    (P1 : ℝ) 
    (rate1 : ℝ) 
    (P2 : ℝ)
    (rate2 : ℝ) : 
    total_investment = 3600 → 
    annual_interest = 144 → 
    P1 = 1800 → 
    rate1 = 3 → 
    P2 = total_investment - P1 → 
    (annual_interest - (P1 * rate1 / 100)) = (P2 * rate2 / 100) →
    rate2 = 5 :=
by 
  intros total_investment_eq annual_interest_eq P1_eq rate1_eq P2_eq interest_eq
  sorry

end interest_rate_second_part_l147_147827


namespace train_length_l147_147317

theorem train_length (L V : ℝ) (h1 : V = L / 15) (h2 : V = (L + 100) / 40) : L = 60 := by
  sorry

end train_length_l147_147317


namespace r_has_money_l147_147400

-- Define the variables and the conditions in Lean
variable (p q r : ℝ)
variable (h1 : p + q + r = 4000)
variable (h2 : r = (2/3) * (p + q))

-- Define the proof statement
theorem r_has_money : r = 1600 := 
  by
    sorry

end r_has_money_l147_147400


namespace CatCafePawRatio_l147_147825

-- Define the context
def CatCafeMeow (P : ℕ) := 3 * P
def CatCafePaw (P : ℕ) := P
def CatCafeCool := 5
def TotalCats (P : ℕ) := CatCafeMeow P + CatCafePaw P

-- State the theorem
theorem CatCafePawRatio (P : ℕ) (n : ℕ) : 
  CatCafeCool = 5 →
  CatCafeMeow P = 3 * CatCafePaw P →
  TotalCats P = 40 →
  P = 10 →
  n * CatCafeCool = P →
  n = 2 :=
by
  intros
  sorry

end CatCafePawRatio_l147_147825


namespace add_expression_l147_147395

theorem add_expression {k : ℕ} :
  (2 * k + 2) + (2 * k + 3) = (2 * k + 2) + (2 * k + 3) := sorry

end add_expression_l147_147395


namespace ratio_B_to_A_l147_147129

theorem ratio_B_to_A (A B C : ℝ) 
  (hA : A = 1 / 21) 
  (hC : C = 2 * B) 
  (h_sum : A + B + C = 1 / 3) : 
  B / A = 2 := 
by 
  /- Proof goes here, but it's omitted as per instructions -/
  sorry

end ratio_B_to_A_l147_147129


namespace prime_product_sum_91_l147_147905

theorem prime_product_sum_91 (p1 p2 : ℕ) (h1 : Nat.Prime p1) (h2 : Nat.Prime p2) (h3 : p1 + p2 = 91) : p1 * p2 = 178 :=
sorry

end prime_product_sum_91_l147_147905


namespace calculate_expression_l147_147798

theorem calculate_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end calculate_expression_l147_147798


namespace lines_perpendicular_l147_147487

/-- Given two lines l1: 3x + 4y + 1 = 0 and l2: 4x - 3y + 2 = 0, 
    prove that the lines are perpendicular. -/
theorem lines_perpendicular :
  ∀ (x y : ℝ), (3 * x + 4 * y + 1 = 0) → (4 * x - 3 * y + 2 = 0) → (- (3 / 4) * (4 / 3) = -1) :=
by
  intro x y h₁ h₂
  sorry

end lines_perpendicular_l147_147487


namespace no_real_solutions_for_equation_l147_147170

theorem no_real_solutions_for_equation (x : ℝ) : ¬(∃ x : ℝ, (8 * x^2 + 150 * x - 5) / (3 * x + 50) = 4 * x + 7) :=
sorry

end no_real_solutions_for_equation_l147_147170


namespace collinear_condition_l147_147252

variable {R : Type*} [LinearOrderedField R]
variable {x1 y1 x2 y2 x3 y3 : R}

theorem collinear_condition : 
  x1 * y2 + x2 * y3 + x3 * y1 = y1 * x2 + y2 * x3 + y3 * x1 →
  ∃ k l m : R, k * (x2 - x1) = l * (y2 - y1) ∧ k * (x3 - x1) = m * (y3 - y1) :=
by
  sorry

end collinear_condition_l147_147252


namespace roger_bike_rides_total_l147_147053

theorem roger_bike_rides_total 
  (r1 : ℕ) (h1 : r1 = 2) 
  (r2 : ℕ) (h2 : r2 = 5 * r1) 
  (r : ℕ) (h : r = r1 + r2) : 
  r = 12 := 
by
  sorry

end roger_bike_rides_total_l147_147053


namespace ellipse_condition_l147_147986

theorem ellipse_condition (x y m : ℝ) :
  (1 < m ∧ m < 3) → (∀ x y, (∃ k1 k2: ℝ, k1 > 0 ∧ k2 > 0 ∧ k1 ≠ k2 ∧ (x^2 / k1 + y^2 / k2 = 1 ↔ (1 < m ∧ m < 3 ∧ m ≠ 2)))) :=
by 
  sorry

end ellipse_condition_l147_147986


namespace largest_whole_number_lt_150_l147_147880

theorem largest_whole_number_lt_150 : 
  ∃ x : ℕ, (9 * x < 150) ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) :=
  sorry

end largest_whole_number_lt_150_l147_147880


namespace length_of_second_train_l147_147861

theorem length_of_second_train 
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (time_to_cross : ℝ)
  (relative_speed : ℝ)
  (total_distance_covered : ℝ)
  (L : ℝ)
  (h1 : length_first_train = 210)
  (h2 : speed_first_train = 120 * 1000 / 3600)
  (h3 : speed_second_train = 80 * 1000 / 3600)
  (h4 : time_to_cross = 9)
  (h5 : relative_speed = (120 * 1000 / 3600) + (80 * 1000 / 3600))
  (h6 : total_distance_covered = relative_speed * time_to_cross)
  (h7 : total_distance_covered = length_first_train + L) : 
  L = 289.95 :=
by {
  sorry
}

end length_of_second_train_l147_147861


namespace find_a_find_min_difference_l147_147023

noncomputable def f (a x : ℝ) : ℝ := x + a * Real.log x
noncomputable def g (a b x : ℝ) : ℝ := f a x + (1 / 2) * x ^ 2 - b * x

theorem find_a (a : ℝ) (h_perpendicular : (1 : ℝ) + a = 2) : a = 1 := 
sorry

theorem find_min_difference (a b x1 x2 : ℝ) (h_b : b ≥ (7 / 2)) 
    (hx1_lt_hx2 : x1 < x2) (hx_sum : x1 + x2 = b - 1)
    (hx_prod : x1 * x2 = 1) :
    g a b x1 - g a b x2 = (15 / 8) - 2 * Real.log 2 :=
sorry

end find_a_find_min_difference_l147_147023


namespace minimum_sum_l147_147606

open Matrix

noncomputable def a := 54
noncomputable def b := 40
noncomputable def c := 5
noncomputable def d := 4

theorem minimum_sum 
  (a b c d : ℕ) 
  (ha : 4 * a = 24 * a - 27 * b) 
  (hb : 4 * b = 15 * a - 17 * b) 
  (hc : 3 * c = 24 * c - 27 * d) 
  (hd : 3 * d = 15 * c - 17 * d) 
  (Hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : 
  a + b + c + d = 103 :=
by
  sorry

end minimum_sum_l147_147606


namespace smallest_a_value_l147_147917

theorem smallest_a_value :
  ∃ (a : ℝ), (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
    2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) ∧
    ∀ a' : ℝ, (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
      2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a' + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) →
      a ≤ a'
  := sorry

end smallest_a_value_l147_147917


namespace simplified_expression_term_count_l147_147762

def even_exponents_terms_count : ℕ :=
  let n := 2008
  let k := 1004
  Nat.choose (k + 2) 2

theorem simplified_expression_term_count :
  even_exponents_terms_count = 505815 :=
sorry

end simplified_expression_term_count_l147_147762


namespace triangle_inequality_l147_147174

theorem triangle_inequality (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_triangle : (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2)) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) :=
sorry

end triangle_inequality_l147_147174


namespace highway_extension_l147_147318

def initial_length : ℕ := 200
def final_length : ℕ := 650
def first_day_construction : ℕ := 50
def second_day_construction : ℕ := 3 * first_day_construction
def total_construction : ℕ := first_day_construction + second_day_construction
def total_extension_needed : ℕ := final_length - initial_length
def miles_still_needed : ℕ := total_extension_needed - total_construction

theorem highway_extension : miles_still_needed = 250 := by
  sorry

end highway_extension_l147_147318


namespace triangle_area_change_l147_147605

theorem triangle_area_change (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let A_original := (B * H) / 2
  let H_new := H * 0.60
  let B_new := B * 1.40
  let A_new := (B_new * H_new) / 2
  (A_new = A_original * 0.84) :=
by
  sorry

end triangle_area_change_l147_147605


namespace value_of_fg3_l147_147262

namespace ProofProblem

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2

theorem value_of_fg3 : f (g 3) = 83 := 
by 
  sorry -- Proof not needed

end ProofProblem

end value_of_fg3_l147_147262


namespace president_vice_president_ways_l147_147461

theorem president_vice_president_ways :
  let boys := 14
  let girls := 10
  let total_boys_ways := boys * (boys - 1)
  let total_girls_ways := girls * (girls - 1)
  total_boys_ways + total_girls_ways = 272 := 
by
  sorry

end president_vice_president_ways_l147_147461


namespace average_weight_of_class_is_61_67_l147_147225

noncomputable def totalWeightA (avgWeightA : ℝ) (numStudentsA : ℕ) : ℝ := avgWeightA * numStudentsA
noncomputable def totalWeightB (avgWeightB : ℝ) (numStudentsB : ℕ) : ℝ := avgWeightB * numStudentsB
noncomputable def totalWeightClass (totalWeightA : ℝ) (totalWeightB : ℝ) : ℝ := totalWeightA + totalWeightB
noncomputable def totalStudentsClass (numStudentsA : ℕ) (numStudentsB : ℕ) : ℕ := numStudentsA + numStudentsB
noncomputable def averageWeightClass (totalWeightClass : ℝ) (totalStudentsClass : ℕ) : ℝ := totalWeightClass / totalStudentsClass

theorem average_weight_of_class_is_61_67 :
  averageWeightClass (totalWeightClass (totalWeightA 50 50) (totalWeightB 70 70))
    (totalStudentsClass 50 70) = 61.67 := by
  sorry

end average_weight_of_class_is_61_67_l147_147225


namespace simplify_and_evaluate_expression_l147_147167

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 + 3) :
  ( (x^2 - 1) / (x^2 - 6 * x + 9) * (1 - x / (x - 1)) / ((x + 1) / (x - 3)) ) = - (Real.sqrt 2 / 2) :=
  sorry

end simplify_and_evaluate_expression_l147_147167


namespace more_than_half_millet_on_day_5_l147_147214

noncomputable def millet_amount (n : ℕ) : ℚ :=
  1 - (3 / 4)^n

theorem more_than_half_millet_on_day_5 : millet_amount 5 > 1 / 2 :=
by
  sorry

end more_than_half_millet_on_day_5_l147_147214


namespace negation_of_squared_inequality_l147_147819

theorem negation_of_squared_inequality (p : ∀ n : ℕ, n^2 ≤ 2*n + 5) : 
  ∃ n : ℕ, n^2 > 2*n + 5 :=
sorry

end negation_of_squared_inequality_l147_147819


namespace cost_effective_for_3000_cost_equal_at_2500_l147_147284

def cost_company_A (x : Nat) : Nat :=
  2 * x / 10 + 500

def cost_company_B (x : Nat) : Nat :=
  4 * x / 10

theorem cost_effective_for_3000 : cost_company_A 3000 < cost_company_B 3000 := 
by {
  sorry
}

theorem cost_equal_at_2500 : cost_company_A 2500 = cost_company_B 2500 := 
by {
  sorry
}

end cost_effective_for_3000_cost_equal_at_2500_l147_147284


namespace exists_sum_of_digits_div_11_l147_147556

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_sum_of_digits_div_11 (H : Finset ℕ) (h₁ : H.card = 39) :
  ∃ (a : ℕ) (h : a ∈ H), sum_of_digits a % 11 = 0 :=
by
  sorry

end exists_sum_of_digits_div_11_l147_147556


namespace complement_intersection_complement_in_U_l147_147332

universe u
open Set

variable (U : Set ℕ) (A B : Set ℕ)

-- Definitions based on the conditions
def universal_set : Set ℕ := { x ∈ (Set.univ : Set ℕ) | x ≤ 4 }
def set_A : Set ℕ := {1, 4}
def set_B : Set ℕ := {2, 4}

-- Problem to be proven
theorem complement_intersection_complement_in_U :
  (U = universal_set) → (A = set_A) → (B = set_B) →
  compl (A ∩ B) ∩ U = {1, 2, 3} :=
by
  intro hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_intersection_complement_in_U_l147_147332


namespace positive_integer_solution_l147_147763

theorem positive_integer_solution (x : Int) (h_pos : x > 0) (h_cond : x + 1000 > 1000 * x) : x = 2 :=
sorry

end positive_integer_solution_l147_147763


namespace part1_part2_l147_147525

open Real

variables (x a : ℝ)

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2 * x - 8 > 0

theorem part1 (h : a = 1) (h_pq : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
by sorry

theorem part2 (hpq : ∀ (a x : ℝ), ¬ p x a → ¬ q x) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end part1_part2_l147_147525


namespace fraction_of_historical_fiction_new_releases_l147_147737

theorem fraction_of_historical_fiction_new_releases
  (total_books : ℕ)
  (historical_fiction_percentage : ℝ := 0.4)
  (historical_fiction_new_releases_percentage : ℝ := 0.4)
  (other_genres_new_releases_percentage : ℝ := 0.7)
  (total_historical_fiction_books := total_books * historical_fiction_percentage)
  (total_other_books := total_books * (1 - historical_fiction_percentage))
  (historical_fiction_new_releases := total_historical_fiction_books * historical_fiction_new_releases_percentage)
  (other_genres_new_releases := total_other_books * other_genres_new_releases_percentage)
  (total_new_releases := historical_fiction_new_releases + other_genres_new_releases) :
  historical_fiction_new_releases / total_new_releases = 8 / 29 := 
by 
  sorry

end fraction_of_historical_fiction_new_releases_l147_147737


namespace slope_of_intersection_points_l147_147942

theorem slope_of_intersection_points {s x y : ℝ} 
  (h1 : 2 * x - 3 * y = 6 * s - 5) 
  (h2 : 3 * x + y = 9 * s + 4) : 
  ∃ m : ℝ, m = 3 ∧ (∀ s : ℝ, (∃ x y : ℝ, 2 * x - 3 * y = 6 * s - 5 ∧ 3 * x + y = 9 * s + 4) → y = m * x + (23/11)) := 
by
  sorry

end slope_of_intersection_points_l147_147942


namespace aardvark_total_distance_l147_147201

noncomputable def total_distance (r_small r_large : ℝ) : ℝ :=
  let small_circumference := 2 * Real.pi * r_small
  let large_circumference := 2 * Real.pi * r_large
  let half_small_circumference := small_circumference / 2
  let half_large_circumference := large_circumference / 2
  let radial_distance := r_large - r_small
  let total_radial_distance := radial_distance + r_large
  half_small_circumference + radial_distance + half_large_circumference + total_radial_distance

theorem aardvark_total_distance :
  total_distance 15 30 = 45 * Real.pi + 45 :=
by
  sorry

end aardvark_total_distance_l147_147201


namespace find_b_l147_147088

noncomputable def geom_seq_term (a b c : ℝ) : Prop :=
∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r

theorem find_b (b : ℝ) (h_geom : geom_seq_term 160 b (108 / 64)) (h_pos : b > 0) :
  b = 15 * Real.sqrt 6 :=
by
  sorry

end find_b_l147_147088


namespace fg_square_diff_l147_147339

open Real

noncomputable def f (x: ℝ) : ℝ := sorry
noncomputable def g (x: ℝ) : ℝ := sorry

axiom h1 (x: ℝ) (hx : -π / 2 < x ∧ x < π / 2) : f x + g x = sqrt ((1 + cos (2 * x)) / (1 - sin x))
axiom h2 : ∀ x, f (-x) = -f x
axiom h3 : ∀ x, g (-x) = g x

theorem fg_square_diff (x : ℝ) (hx : -π / 2 < x ∧ x < π / 2) : (f x)^2 - (g x)^2 = -2 * cos x := 
sorry

end fg_square_diff_l147_147339


namespace largest_square_tile_for_board_l147_147531

theorem largest_square_tile_for_board (length width gcd_val : ℕ) (h1 : length = 16) (h2 : width = 24) 
  (h3 : gcd_val = Int.gcd length width) : gcd_val = 8 := by
  sorry

end largest_square_tile_for_board_l147_147531


namespace problem_statement_l147_147309

noncomputable def a := Real.log 2 / Real.log 14
noncomputable def b := Real.log 2 / Real.log 7
noncomputable def c := Real.log 2 / Real.log 4

theorem problem_statement : (1 / a - 1 / b + 1 / c) = 3 := by
  sorry

end problem_statement_l147_147309


namespace total_amount_spent_l147_147154

def cost_of_tshirt : ℕ := 100
def cost_of_pants : ℕ := 250
def num_of_tshirts : ℕ := 5
def num_of_pants : ℕ := 4

theorem total_amount_spent : (num_of_tshirts * cost_of_tshirt) + (num_of_pants * cost_of_pants) = 1500 := by
  sorry

end total_amount_spent_l147_147154


namespace quadratic_value_at_3_l147_147930

theorem quadratic_value_at_3 (a b c : ℝ) :
  (a * (-2)^2 + b * (-2) + c = -13 / 2) →
  (a * (-1)^2 + b * (-1) + c = -4) →
  (a * 0^2 + b * 0 + c = -2.5) →
  (a * 1^2 + b * 1 + c = -2) →
  (a * 2^2 + b * 2 + c = -2.5) →
  (a * 3^2 + b * 3 + c = -4) :=
by
  sorry

end quadratic_value_at_3_l147_147930


namespace geometric_series_sum_l147_147033

theorem geometric_series_sum :
  let a := 1
  let r := (1 / 4 : ℚ)
  (a / (1 - r)) = 4 / 3 :=
by
  sorry

end geometric_series_sum_l147_147033


namespace parking_methods_count_l147_147768

theorem parking_methods_count : 
  ∃ (n : ℕ), n = 72 ∧ (∃ (spaces cars slots remainingSlots : ℕ), 
  spaces = 7 ∧ cars = 3 ∧ slots = 1 ∧ remainingSlots = 4 ∧
  ∃ (perm_ways slot_ways : ℕ), perm_ways = 6 ∧ slot_ways = 12 ∧ n = perm_ways * slot_ways) :=
  by
    sorry

end parking_methods_count_l147_147768


namespace sum_M_N_K_l147_147846

theorem sum_M_N_K (d K M N : ℤ) 
(h : ∀ x : ℤ, (x^2 + 3*x + 1) ∣ (x^4 - d*x^3 + M*x^2 + N*x + K)) :
  M + N + K = 5*K - 4*d - 11 := 
sorry

end sum_M_N_K_l147_147846


namespace can_weight_is_two_l147_147403

theorem can_weight_is_two (c : ℕ) (h1 : 100 = 20 * c + 6 * ((100 - 20 * c) / 6)) (h2 : 160 = 10 * ((100 - 20 * c) / 6) + 3 * 20) : c = 2 :=
by
  sorry

end can_weight_is_two_l147_147403


namespace min_value_of_inverse_sum_l147_147637

noncomputable def minimumValue (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) : ℝ :=
  9 + 6 * Real.sqrt 2

theorem min_value_of_inverse_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 / 3 ∧ (1/x + 1/y) = 9 + 6 * Real.sqrt 2 := by
  sorry

end min_value_of_inverse_sum_l147_147637


namespace bus_capacity_total_kids_l147_147779

-- Definitions based on conditions
def total_rows : ℕ := 25
def lower_deck_rows : ℕ := 15
def upper_deck_rows : ℕ := 10
def lower_deck_capacity_per_row : ℕ := 5
def upper_deck_capacity_per_row : ℕ := 3
def staff_members : ℕ := 4

-- Theorem statement
theorem bus_capacity_total_kids : 
  (lower_deck_rows * lower_deck_capacity_per_row) + 
  (upper_deck_rows * upper_deck_capacity_per_row) - staff_members = 101 := 
by
  sorry

end bus_capacity_total_kids_l147_147779


namespace flutes_tried_out_l147_147405

theorem flutes_tried_out (flutes clarinets trumpets pianists : ℕ) 
  (percent_flutes_in : ℕ → ℕ) (percent_clarinets_in : ℕ → ℕ) 
  (percent_trumpets_in : ℕ → ℕ) (percent_pianists_in : ℕ → ℕ) 
  (total_in_band : ℕ) :
  percent_flutes_in flutes = 80 / 100 * flutes ∧
  percent_clarinets_in clarinets = 30 / 2 ∧
  percent_trumpets_in trumpets = 60 / 3 ∧
  percent_pianists_in pianists = 20 / 10 ∧
  total_in_band = 53 →
  flutes = 20 :=
by
  sorry

end flutes_tried_out_l147_147405


namespace gcd_eq_gcd_of_division_l147_147221

theorem gcd_eq_gcd_of_division (a b q r : ℕ) (h1 : a = b * q + r) (h2 : 0 < r) (h3 : r < b) (h4 : a > b) : 
  Nat.gcd a b = Nat.gcd b r :=
by
  sorry

end gcd_eq_gcd_of_division_l147_147221


namespace largest_sum_of_ABCD_l147_147449

theorem largest_sum_of_ABCD :
  ∃ (A B C D : ℕ), 10 ≤ A ∧ A < 100 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100 ∧ 10 ≤ D ∧ D < 100 ∧
  B = 3 * C ∧ D = 2 * B - C ∧ A = B + D ∧ A + B + C + D = 204 :=
by
  sorry

end largest_sum_of_ABCD_l147_147449


namespace fg_at_3_l147_147399

-- Define the functions f and g according to the conditions
def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x + 2)^2

theorem fg_at_3 : f (g 3) = 103 :=
by
  sorry

end fg_at_3_l147_147399


namespace max_value_x_plus_y_l147_147634

theorem max_value_x_plus_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 48) (hx_mult_4 : x % 4 = 0) : x + y ≤ 49 :=
sorry

end max_value_x_plus_y_l147_147634


namespace cost_of_one_box_of_paper_clips_l147_147895

theorem cost_of_one_box_of_paper_clips (p i : ℝ) 
  (h1 : 15 * p + 7 * i = 55.40) 
  (h2 : 12 * p + 10 * i = 61.70) : 
  p = 1.835 := 
by 
  sorry

end cost_of_one_box_of_paper_clips_l147_147895


namespace journey_speed_first_half_l147_147117

theorem journey_speed_first_half (total_distance : ℕ) (total_time : ℕ) (second_half_distance : ℕ) (second_half_speed : ℕ)
  (distance_first_half_eq_half_total : second_half_distance = total_distance / 2)
  (time_for_journey_eq : total_time = 20)
  (journey_distance_eq : total_distance = 240)
  (second_half_speed_eq : second_half_speed = 15) :
  let v := second_half_distance / (total_time - (second_half_distance / second_half_speed))
  v = 10 := 
by
  sorry

end journey_speed_first_half_l147_147117


namespace kids_played_on_tuesday_l147_147772

-- Define the total number of kids Julia played with
def total_kids : ℕ := 18

-- Define the number of kids Julia played with on Monday
def monday_kids : ℕ := 4

-- Define the number of kids Julia played with on Tuesday
def tuesday_kids : ℕ := total_kids - monday_kids

-- The proof goal:
theorem kids_played_on_tuesday : tuesday_kids = 14 :=
by sorry

end kids_played_on_tuesday_l147_147772


namespace points_product_l147_147908

def f (n : ℕ) : ℕ :=
  if n % 6 == 0 then 6
  else if n % 2 == 0 then 2
  else 0

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

def allie_rolls := [5, 4, 1, 2]
def betty_rolls := [6, 3, 3, 2]

def allie_points := total_points allie_rolls
def betty_points := total_points betty_rolls

theorem points_product : allie_points * betty_points = 32 := by
  sorry

end points_product_l147_147908


namespace figure_50_squares_l147_147215

-- Define the quadratic function with the given number of squares for figures 0, 1, 2, and 3.
def g (n : ℕ) : ℕ := 2 * n ^ 2 + 4 * n + 2

-- Prove that the number of nonoverlapping unit squares in figure 50 is 5202.
theorem figure_50_squares : g 50 = 5202 := 
by 
  sorry

end figure_50_squares_l147_147215


namespace initial_oranges_in_bowl_l147_147863

theorem initial_oranges_in_bowl (A O : ℕ) (R : ℚ) (h1 : A = 14) (h2 : R = 0.7) 
    (h3 : R * (A + O - 15) = A) : O = 21 := 
by 
  sorry

end initial_oranges_in_bowl_l147_147863


namespace sin_alpha_plus_7pi_over_12_l147_147748

theorem sin_alpha_plus_7pi_over_12 (α : Real) 
  (h1 : Real.cos (α + π / 12) = 1 / 5) : 
  Real.sin (α + 7 * π / 12) = 1 / 5 :=
by
  sorry

end sin_alpha_plus_7pi_over_12_l147_147748


namespace isosceles_triangle_base_angle_l147_147196

theorem isosceles_triangle_base_angle
    (X : ℝ)
    (h1 : 0 < X)
    (h2 : 2 * X + X + X = 180)
    (h3 : X + X + 2 * X = 180) :
    X = 45 ∨ X = 72 :=
by sorry

end isosceles_triangle_base_angle_l147_147196


namespace ceil_square_of_neg_five_thirds_l147_147187

theorem ceil_square_of_neg_five_thirds : Int.ceil ((-5 / 3:ℚ)^2) = 3 := by
  sorry

end ceil_square_of_neg_five_thirds_l147_147187


namespace lance_read_yesterday_l147_147270

-- Definitions based on conditions
def total_pages : ℕ := 100
def pages_tomorrow : ℕ := 35
def pages_yesterday (Y : ℕ) : ℕ := Y
def pages_today (Y : ℕ) : ℕ := Y - 5

-- The statement that we need to prove
theorem lance_read_yesterday (Y : ℕ) (h : pages_yesterday Y + pages_today Y + pages_tomorrow = total_pages) : Y = 35 :=
by sorry

end lance_read_yesterday_l147_147270


namespace max_value_of_expression_l147_147273

-- Define the real numbers p, q, r and the conditions
variables {p q r : ℝ}

-- Define the main goal
theorem max_value_of_expression 
(h : 9 * p^2 + 4 * q^2 + 25 * r^2 = 4) : 
  (5 * p + 3 * q + 10 * r) ≤ (10 * Real.sqrt 13 / 3) :=
sorry

end max_value_of_expression_l147_147273


namespace alicia_candies_problem_l147_147220

theorem alicia_candies_problem :
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ (n % 9 = 7) ∧ (n % 7 = 5) ∧ n = 124 :=
by
  sorry

end alicia_candies_problem_l147_147220


namespace Susan_ate_six_candies_l147_147626

def candy_consumption_weekly : Prop :=
  ∀ (candies_bought_Tue candies_bought_Wed candies_bought_Thu candies_bought_Fri : ℕ)
    (candies_left : ℕ) (total_spending : ℕ),
    candies_bought_Tue = 3 →
    candies_bought_Wed = 0 →
    candies_bought_Thu = 5 →
    candies_bought_Fri = 2 →
    candies_left = 4 →
    total_spending = 9 →
    candies_bought_Tue + candies_bought_Wed + candies_bought_Thu + candies_bought_Fri - candies_left = 6

theorem Susan_ate_six_candies : candy_consumption_weekly :=
by {
  -- The proof will be filled in later
  sorry
}

end Susan_ate_six_candies_l147_147626


namespace find_solutions_l147_147566

theorem find_solutions (x y : ℝ) :
    (x * y^2 = 15 * x^2 + 17 * x * y + 15 * y^2 ∧ x^2 * y = 20 * x^2 + 3 * y^2) ↔ 
    (x = 0 ∧ y = 0) ∨ (x = -19 ∧ y = -2) :=
by sorry

end find_solutions_l147_147566


namespace smallest_positive_period_and_symmetry_l147_147713

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + (7 * Real.pi / 4)) + 
  Real.cos (x - (3 * Real.pi / 4))

theorem smallest_positive_period_and_symmetry :
  (∃ T > 0, T = 2 * Real.pi ∧ ∀ x, f (x + T) = f x) ∧ 
  (∃ a, a = - (Real.pi / 4) ∧ ∀ x, f (2 * a - x) = f x) :=
by
  sorry

end smallest_positive_period_and_symmetry_l147_147713


namespace no_real_solution_ratio_l147_147080

theorem no_real_solution_ratio (x : ℝ) : (x + 3) / (2 * x + 5) = (5 * x + 4) / (8 * x + 5) → false :=
by {
  sorry
}

end no_real_solution_ratio_l147_147080


namespace total_triangles_in_figure_l147_147666

theorem total_triangles_in_figure :
  let row1 := 3
  let row2 := 2
  let row3 := 1
  let small_triangles := row1 + row2 + row3
  let two_small_comb := 3
  let three_small_comb := 1
  let all_small_comb := 1
  small_triangles + two_small_comb + three_small_comb + all_small_comb = 11 :=
by
  let row1 := 3
  let row2 := 2
  let row3 := 1
  let small_triangles := row1 + row2 + row3
  let two_small_comb := 3
  let three_small_comb := 1
  let all_small_comb := 1
  show small_triangles + two_small_comb + three_small_comb + all_small_comb = 11
  sorry

end total_triangles_in_figure_l147_147666


namespace jeremy_school_distance_l147_147360

theorem jeremy_school_distance :
  ∃ d : ℝ, d = 9.375 ∧
  (∃ v : ℝ, (d = v * (15 / 60)) ∧ (d = (v + 25) * (9 / 60))) := by
  sorry

end jeremy_school_distance_l147_147360


namespace accounting_vs_calling_clients_l147_147182

/--
Given:
1. Total time Maryann worked today is 560 minutes.
2. Maryann spent 70 minutes calling clients.

Prove:
Maryann spends 7 times longer doing accounting than calling clients.
-/
theorem accounting_vs_calling_clients 
  (total_time : ℕ) 
  (calling_time : ℕ) 
  (h_total : total_time = 560) 
  (h_calling : calling_time = 70) : 
  (total_time - calling_time) / calling_time = 7 :=
  sorry

end accounting_vs_calling_clients_l147_147182


namespace monotonic_increasing_iff_l147_147031

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + 1 / x

theorem monotonic_increasing_iff (a : ℝ) :
  (∀ x : ℝ, 1 < x → f a x ≥ f a 1) ↔ a ≥ 1 :=
by
  sorry

end monotonic_increasing_iff_l147_147031


namespace luke_total_score_l147_147921

theorem luke_total_score (points_per_round : ℕ) (number_of_rounds : ℕ) (total_score : ℕ) : 
  points_per_round = 146 ∧ number_of_rounds = 157 ∧ total_score = points_per_round * number_of_rounds → 
  total_score = 22822 := by 
  sorry

end luke_total_score_l147_147921


namespace natural_solutions_3x_4y_eq_12_l147_147111

theorem natural_solutions_3x_4y_eq_12 :
  ∃ x y : ℕ, (3 * x + 4 * y = 12) ∧ ((x = 4 ∧ y = 0) ∨ (x = 0 ∧ y = 3)) := 
sorry

end natural_solutions_3x_4y_eq_12_l147_147111


namespace third_median_length_l147_147696

theorem third_median_length (m1 m2 area : ℝ) (h1 : m1 = 5) (h2 : m2 = 10) (h3 : area = 10 * Real.sqrt 10) : 
  ∃ m3 : ℝ, m3 = 3 * Real.sqrt 10 :=
by
  sorry

end third_median_length_l147_147696


namespace number_of_four_digit_numbers_l147_147325

theorem number_of_four_digit_numbers (digits: Finset ℕ) (h: digits = {1, 1, 2, 0}) :
  ∃ count : ℕ, (count = 9) ∧ 
  (∀ n ∈ digits, n ≠ 0 → n * 1000 + n ≠ 0) := 
sorry

end number_of_four_digit_numbers_l147_147325


namespace balance_balls_l147_147563

variable (R O B P : ℝ)

-- Conditions based on the problem statement
axiom h1 : 4 * R = 8 * B
axiom h2 : 3 * O = 7.5 * B
axiom h3 : 8 * B = 6 * P

-- The theorem we need to prove
theorem balance_balls : 5 * R + 3 * O + 3 * P = 21.5 * B :=
by 
  sorry

end balance_balls_l147_147563


namespace area_larger_sphere_l147_147029

variables {r1 r2 r : ℝ}
variables {A1 A2 : ℝ}

-- Declare constants for the problem
def radius_smaller_sphere : ℝ := 4 -- r1
def radius_larger_sphere : ℝ := 6  -- r2
def radius_ball : ℝ := 1           -- r
def area_smaller_sphere : ℝ := 27  -- A1

-- Given conditions
axiom radius_smaller_sphere_condition : r1 = radius_smaller_sphere
axiom radius_larger_sphere_condition : r2 = radius_larger_sphere
axiom radius_ball_condition : r = radius_ball
axiom area_smaller_sphere_condition : A1 = area_smaller_sphere

-- Statement to be proved
theorem area_larger_sphere :
  r1 = radius_smaller_sphere → r2 = radius_larger_sphere → r = radius_ball → A1 = area_smaller_sphere → A2 = 60.75 :=
by
  intros
  sorry

end area_larger_sphere_l147_147029


namespace roots_polynomial_identity_l147_147192

theorem roots_polynomial_identity (a b x₁ x₂ : ℝ) 
  (h₁ : x₁^2 + b*x₁ + b^2 + a = 0) 
  (h₂ : x₂^2 + b*x₂ + b^2 + a = 0) : x₁^2 + x₁*x₂ + x₂^2 + a = 0 :=
by 
  sorry

end roots_polynomial_identity_l147_147192


namespace plan_b_more_cost_effective_l147_147446

theorem plan_b_more_cost_effective (x : ℕ) : 
  (12 * x : ℤ) > (3000 + 8 * x : ℤ) → x ≥ 751 :=
sorry

end plan_b_more_cost_effective_l147_147446


namespace simplify_expression_l147_147830

theorem simplify_expression (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : a + c > b) :
  |a + b - c| - |b - a - c| = 2 * b - 2 * c :=
by
  sorry

end simplify_expression_l147_147830


namespace oj_fraction_is_11_over_30_l147_147659

-- Define the capacity of each pitcher
def pitcher_capacity : ℕ := 600

-- Define the fraction of orange juice in each pitcher
def fraction_oj_pitcher1 : ℚ := 1 / 3
def fraction_oj_pitcher2 : ℚ := 2 / 5

-- Define the amount of orange juice in each pitcher
def oj_amount_pitcher1 := pitcher_capacity * fraction_oj_pitcher1
def oj_amount_pitcher2 := pitcher_capacity * fraction_oj_pitcher2

-- Define the total amount of orange juice after both pitchers are poured into the large container
def total_oj_amount := oj_amount_pitcher1 + oj_amount_pitcher2

-- Define the total volume of the mixture in the large container
def total_mixture_volume := 2 * pitcher_capacity

-- Define the fraction of the mixture that is orange juice
def oj_fraction_in_mixture := total_oj_amount / total_mixture_volume

-- Prove that the fraction of the mixture that is orange juice is 11/30
theorem oj_fraction_is_11_over_30 : oj_fraction_in_mixture = 11 / 30 := by
  sorry

end oj_fraction_is_11_over_30_l147_147659


namespace total_questions_attempted_l147_147560

theorem total_questions_attempted (C W T : ℕ) (hC : C = 42) (h_score : 4 * C - W = 150) : T = C + W → T = 60 :=
by
  sorry

end total_questions_attempted_l147_147560


namespace inches_repaired_before_today_l147_147393

-- Definitions and assumptions based on the conditions.
def total_inches_repaired : ℕ := 4938
def inches_repaired_today : ℕ := 805

-- Target statement that needs to be proven.
theorem inches_repaired_before_today : total_inches_repaired - inches_repaired_today = 4133 :=
by
  sorry

end inches_repaired_before_today_l147_147393


namespace piggy_bank_exceed_five_dollars_l147_147658

noncomputable def sequence_sum (n : ℕ) : ℕ := 2^n - 1

theorem piggy_bank_exceed_five_dollars (n : ℕ) (start_day : Nat) (day_of_week : Fin 7) :
  ∃ (n : ℕ), sequence_sum n > 500 ∧ n = 9 ∧ (start_day + n) % 7 = 2 := 
sorry

end piggy_bank_exceed_five_dollars_l147_147658


namespace chord_length_l147_147037

theorem chord_length (r d : ℝ) (h_r : r = 5) (h_d : d = 4) : 
  ∃ EF : ℝ, EF = 6 :=
by
  sorry

end chord_length_l147_147037


namespace opposite_blue_face_is_white_l147_147667

-- Define colors
inductive Color
| Red
| Blue
| Orange
| Purple
| Green
| Yellow
| White

-- Define the positions of colors on the cube
structure CubeConfig :=
(top : Color)
(front : Color)
(bottom : Color)
(back : Color)
(left : Color)
(right : Color)

-- The given conditions
def cube_conditions (c : CubeConfig) : Prop :=
  c.top = Color.Purple ∧
  c.front = Color.Green ∧
  c.bottom = Color.Yellow ∧
  c.back = Color.Orange ∧
  c.left = Color.Blue ∧
  c.right = Color.White

-- The statement we need to prove
theorem opposite_blue_face_is_white (c : CubeConfig) (h : cube_conditions c) :
  c.right = Color.White :=
by
  -- Proof placeholder
  sorry

end opposite_blue_face_is_white_l147_147667


namespace div_by_17_l147_147059

theorem div_by_17 (n : ℕ) (h : ¬ 17 ∣ n) : 17 ∣ (n^8 + 1) ∨ 17 ∣ (n^8 - 1) := 
by sorry

end div_by_17_l147_147059


namespace find_a_add_b_l147_147171

theorem find_a_add_b (a b : ℝ) 
  (h1 : ∀ (x : ℝ), y = a + b / (x^2 + 1))
  (h2 : (y = 3) → (x = 1)) 
  (h3 : (y = 2) → (x = 0)) : a + b = 2 :=
by
  sorry

end find_a_add_b_l147_147171


namespace perimeter_triangle_ABC_eq_18_l147_147694

theorem perimeter_triangle_ABC_eq_18 (h1 : ∀ (Δ : ℕ), Δ = 9) 
(h2 : ∀ (p : ℕ), p = 6) : 
∀ (perimeter_ABC : ℕ), perimeter_ABC = 18 := by
sorry

end perimeter_triangle_ABC_eq_18_l147_147694


namespace shrimp_per_pound_l147_147960

theorem shrimp_per_pound (shrimp_per_guest guests : ℕ) (cost_per_pound : ℝ) (total_spent : ℝ)
  (hshrimp_per_guest : shrimp_per_guest = 5) (hguests : guests = 40) (hcost_per_pound : cost_per_pound = 17.0) (htotal_spent : total_spent = 170.0) :
  let total_shrimp := shrimp_per_guest * guests
  let total_pounds := total_spent / cost_per_pound
  total_shrimp / total_pounds = 20 :=
by
  sorry

end shrimp_per_pound_l147_147960


namespace cost_of_paintbrush_l147_147547

noncomputable def cost_of_paints : ℝ := 4.35
noncomputable def cost_of_easel : ℝ := 12.65
noncomputable def amount_already_has : ℝ := 6.50
noncomputable def additional_amount_needed : ℝ := 12.00

-- Let's define the total cost needed and the total costs of items
noncomputable def total_cost_of_paints_and_easel : ℝ := cost_of_paints + cost_of_easel
noncomputable def total_amount_needed : ℝ := amount_already_has + additional_amount_needed

-- And now we can state our theorem that needs to be proved.
theorem cost_of_paintbrush : total_amount_needed - total_cost_of_paints_and_easel = 1.50 :=
by
  sorry

end cost_of_paintbrush_l147_147547


namespace rectangle_area_l147_147173

theorem rectangle_area (AB AD AE : ℝ) (S_trapezoid S_triangle : ℝ) (perim_triangle perim_trapezoid : ℝ)
  (h1 : AD - AB = 9)
  (h2 : S_trapezoid = 5 * S_triangle)
  (h3 : perim_triangle + 68 = perim_trapezoid)
  (h4 : S_trapezoid + S_triangle = S_triangle * 6)
  (h5 : perim_triangle = AB + AE + (AE - AB))
  (h6 : perim_trapezoid = AB + AD + AE + (2 * (AD - AE))) :
  AD * AB = 3060 := by
  sorry

end rectangle_area_l147_147173


namespace average_reading_days_l147_147052

theorem average_reading_days :
  let days_participated := [2, 3, 4, 5, 6]
  let students := [5, 4, 7, 3, 6]
  let total_days := List.zipWith (· * ·) days_participated students |>.sum
  let total_students := students.sum
  let average := total_days / total_students
  average = 4.04 := sorry

end average_reading_days_l147_147052


namespace wizard_elixir_combinations_l147_147447

theorem wizard_elixir_combinations :
  let herbs := 4
  let crystals := 6
  let invalid_combinations := 3
  herbs * crystals - invalid_combinations = 21 := 
by
  sorry

end wizard_elixir_combinations_l147_147447


namespace sector_area_max_radius_l147_147847

noncomputable def arc_length (R : ℝ) : ℝ := 20 - 2 * R

noncomputable def sector_area (R : ℝ) : ℝ :=
  let l := arc_length R
  0.5 * l * R

theorem sector_area_max_radius :
  ∃ (R : ℝ), sector_area R = -R^2 + 10 * R ∧
             R = 5 :=
sorry

end sector_area_max_radius_l147_147847


namespace mail_distribution_l147_147727

def total_mail : ℕ := 2758
def mail_for_first_block : ℕ := 365
def mail_for_second_block : ℕ := 421
def remaining_mail : ℕ := total_mail - (mail_for_first_block + mail_for_second_block)
def remaining_blocks : ℕ := 3
def mail_per_remaining_block : ℕ := remaining_mail / remaining_blocks

theorem mail_distribution :
  mail_per_remaining_block = 657 := by
  sorry

end mail_distribution_l147_147727


namespace Zack_kept_5_marbles_l147_147267

-- Define the initial number of marbles Zack had
def Zack_initial_marbles : ℕ := 65

-- Define the number of marbles each friend receives
def marbles_per_friend : ℕ := 20

-- Define the total number of friends
def friends : ℕ := 3

noncomputable def marbles_given_away : ℕ := friends * marbles_per_friend

-- Define the amount of marbles kept by Zack
noncomputable def marbles_kept_by_Zack : ℕ := Zack_initial_marbles - marbles_given_away

-- The theorem to prove
theorem Zack_kept_5_marbles : marbles_kept_by_Zack = 5 := by
  -- Proof skipped with sorry
  sorry

end Zack_kept_5_marbles_l147_147267


namespace distinct_flags_count_l147_147456

theorem distinct_flags_count : 
  ∃ n, n = 36 ∧ (∀ c1 c2 c3 : Fin 4, c1 ≠ c2 ∧ c2 ≠ c3 → n = 4 * 3 * 3) := 
sorry

end distinct_flags_count_l147_147456


namespace arithmetic_geometric_sequence_l147_147342

theorem arithmetic_geometric_sequence : 
  ∀ (a : ℤ), (∀ n : ℤ, a_n = a + (n-1) * 2) → 
  (a + 4)^2 = a * (a + 6) → 
  (a + 10 = 2) :=
by
  sorry

end arithmetic_geometric_sequence_l147_147342


namespace solve_for_y_l147_147766

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + 2 * y^(1/3)) : y = 1000 := 
by
  sorry

end solve_for_y_l147_147766


namespace square_garden_dimensions_and_area_increase_l147_147854

def original_length : ℝ := 60
def original_width : ℝ := 20

def original_area : ℝ := original_length * original_width
def original_perimeter : ℝ := 2 * (original_length + original_width)

theorem square_garden_dimensions_and_area_increase
    (L : ℝ := 60) (W : ℝ := 20)
    (orig_area : ℝ := L * W)
    (orig_perimeter : ℝ := 2 * (L + W))
    (square_side_length : ℝ := orig_perimeter / 4)
    (new_area : ℝ := square_side_length * square_side_length)
    (area_increase : ℝ := new_area - orig_area) :
    square_side_length = 40 ∧ area_increase = 400 :=
by {sorry}

end square_garden_dimensions_and_area_increase_l147_147854


namespace sequence_arithmetic_l147_147978

-- Define the sequence and sum conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ) (p : ℝ)

-- We are given that the sum of the first n terms is Sn = n * p * a_n
axiom sum_condition (n : ℕ) (hpos : n > 0) : S n = n * p * a n

-- Also, given that a_1 ≠ a_2
axiom a1_ne_a2 : a 1 ≠ a 2

-- Define what we need to prove
theorem sequence_arithmetic (n : ℕ) (hn : n ≥ 2) :
  ∃ (a2 : ℝ), p = 1/2 ∧ a n = (n-1) * a2 :=
by
  sorry

end sequence_arithmetic_l147_147978


namespace part1_l147_147759

variable (A B C : ℝ)
variable (a b c S : ℝ)
variable (h1 : a * (1 + Real.cos C) + c * (1 + Real.cos A) = (5 / 2) * b)
variable (h2 : a * Real.cos C + c * Real.cos A = b)

theorem part1 : 2 * (a + c) = 3 * b := 
sorry

end part1_l147_147759


namespace abs_val_inequality_solution_l147_147116

theorem abs_val_inequality_solution (x : ℝ) : |x - 2| + |x + 3| ≥ 4 ↔ x ≤ - (5 / 2) :=
by
  sorry

end abs_val_inequality_solution_l147_147116


namespace find_number_of_eggs_l147_147159

namespace HalloweenCleanup

def eggs (E : ℕ) (seconds_per_egg : ℕ) (minutes_per_roll : ℕ) (total_time : ℕ) (num_rolls : ℕ) : Prop :=
  seconds_per_egg = 15 ∧
  minutes_per_roll = 30 ∧
  total_time = 225 ∧
  num_rolls = 7 ∧
  E * (seconds_per_egg / 60) + num_rolls * minutes_per_roll = total_time

theorem find_number_of_eggs : ∃ E : ℕ, eggs E 15 30 225 7 :=
  by
    use 60
    unfold eggs
    simp
    exact sorry

end HalloweenCleanup

end find_number_of_eggs_l147_147159


namespace cylindrical_container_volume_increase_l147_147164

theorem cylindrical_container_volume_increase (R H : ℝ)
  (initial_volume : ℝ)
  (x : ℝ) : 
  R = 10 ∧ H = 5 ∧ initial_volume = π * R^2 * H →
  π * (R + 2 * x)^2 * H = π * R^2 * (H + 3 * x) →
  x = 5 :=
by
  -- Given conditions
  intro conditions volume_equation
  obtain ⟨hR, hH, hV⟩ := conditions
  -- Simplifying and solving the resulting equation
  sorry

end cylindrical_container_volume_increase_l147_147164


namespace probability_at_most_one_A_B_selected_l147_147866

def total_employees : ℕ := 36
def ratio_3_2_1 : (ℕ × ℕ × ℕ) := (3, 2, 1)
def sample_size : ℕ := 12
def youth_group_size : ℕ := 6
def total_combinations_youth : ℕ := Nat.choose 6 2
def event_complementary : ℕ := Nat.choose 2 2

theorem probability_at_most_one_A_B_selected :
  let prob := 1 - event_complementary / total_combinations_youth
  prob = (14 : ℚ) / 15 := sorry

end probability_at_most_one_A_B_selected_l147_147866


namespace no_square_divisible_by_six_in_range_l147_147470

theorem no_square_divisible_by_six_in_range : ¬ ∃ y : ℕ, (∃ k : ℕ, y = k * k) ∧ (6 ∣ y) ∧ (50 ≤ y ∧ y ≤ 120) :=
by
  sorry

end no_square_divisible_by_six_in_range_l147_147470


namespace integer_solutions_inequality_system_l147_147565

noncomputable def check_inequality_system (x : ℤ) : Prop :=
  (3 * x + 1 < x - 3) ∧ ((1 + x) / 2 ≤ (1 + 2 * x) / 3 + 1)

theorem integer_solutions_inequality_system :
  {x : ℤ | check_inequality_system x} = {-5, -4, -3} :=
by
  sorry

end integer_solutions_inequality_system_l147_147565


namespace product_lcm_gcd_eq_product_original_numbers_l147_147538

theorem product_lcm_gcd_eq_product_original_numbers :
  let a := 12
  let b := 18
  (Int.gcd a b) * (Int.lcm a b) = a * b :=
by
  sorry

end product_lcm_gcd_eq_product_original_numbers_l147_147538


namespace seven_power_units_digit_l147_147916

def units_digit (n : ℕ) : ℕ := (7^n) % 10

theorem seven_power_units_digit : units_digit 2023 = 3 := by
  -- Proof omitted for simplification
  sorry

end seven_power_units_digit_l147_147916


namespace shoveling_driveways_l147_147275

-- Definitions of the conditions
def cost_of_candy_bars := 2 * 0.75
def cost_of_lollipops := 4 * 0.25
def total_cost := cost_of_candy_bars + cost_of_lollipops
def portion_of_earnings := total_cost * 6
def charge_per_driveway := 1.50
def number_of_driveways := portion_of_earnings / charge_per_driveway

-- The theorem to prove Jimmy shoveled 10 driveways
theorem shoveling_driveways :
  number_of_driveways = 10 := 
by
  sorry

end shoveling_driveways_l147_147275


namespace simplest_form_expression_l147_147008

variable {b : ℝ}

theorem simplest_form_expression (h : b ≠ 1) :
  1 - (1 / (2 + (b / (1 - b)))) = 1 / (2 - b) :=
by
  sorry

end simplest_form_expression_l147_147008


namespace intersection_A_B_l147_147047

def A : Set ℝ := { x | 1 < x - 1 ∧ x - 1 ≤ 3 }
def B : Set ℝ := { 2, 3, 4 }

theorem intersection_A_B : A ∩ B = {3, 4} := 
by 
  sorry

end intersection_A_B_l147_147047


namespace product_of_possible_values_l147_147728

theorem product_of_possible_values (x : ℚ) (h : abs ((18 : ℚ) / (2 * x) - 4) = 3) : (x = 9 ∨ x = 9/7) → (9 * (9/7) = 81/7) :=
by
  intros
  sorry

end product_of_possible_values_l147_147728


namespace expression_equals_12_l147_147276

-- Define the values of a, b, c, and k
def a : ℤ := 10
def b : ℤ := 15
def c : ℤ := 3
def k : ℤ := 2

-- Define the expression to be evaluated
def expr : ℤ := (a - (b - k * c)) - ((a - b) - k * c)

-- Prove that the expression equals 12
theorem expression_equals_12 : expr = 12 :=
by
  -- The proof will go here, leaving a placeholder for now
  sorry

end expression_equals_12_l147_147276


namespace largest_common_value_less_than_1000_l147_147952

def arithmetic_sequence_1 (n : ℕ) : ℕ := 2 + 3 * n
def arithmetic_sequence_2 (m : ℕ) : ℕ := 4 + 8 * m

theorem largest_common_value_less_than_1000 :
  ∃ a n m : ℕ, a = arithmetic_sequence_1 n ∧ a = arithmetic_sequence_2 m ∧ a < 1000 ∧ a = 980 :=
by { sorry }

end largest_common_value_less_than_1000_l147_147952


namespace find_parabola_equation_l147_147517

-- Define the problem conditions
def parabola_vertex_at_origin (f : ℝ → ℝ) : Prop :=
  f 0 = 0

def axis_of_symmetry_x_or_y (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = 0) ∨ (∀ y, f 0 = y)

def passes_through_point (f : ℝ → ℝ) (pt : ℝ × ℝ) : Prop :=
  f pt.1 = pt.2

-- Define the specific forms we expect the equations of the parabola to take
def equation1 (x y : ℝ) : Prop :=
  y^2 = - (9 / 2) * x

def equation2 (x y : ℝ) : Prop :=
  x^2 = (4 / 3) * y

-- state the main theorem
theorem find_parabola_equation :
  ∃ f : ℝ → ℝ, parabola_vertex_at_origin f ∧ axis_of_symmetry_x_or_y f ∧ passes_through_point f (-2, 3) ∧
  (equation1 (-2) (f (-2)) ∨ equation2 (-2) (f (-2))) :=
sorry

end find_parabola_equation_l147_147517


namespace solve_toenail_problem_l147_147740

def toenail_problem (b_toenails r_toenails_already r_toenails_more : ℕ) : Prop :=
  (b_toenails = 20) ∧
  (r_toenails_already = 40) ∧
  (r_toenails_more = 20) →
  (r_toenails_already + r_toenails_more = 60)

theorem solve_toenail_problem : toenail_problem 20 40 20 :=
by {
  sorry
}

end solve_toenail_problem_l147_147740


namespace quadratic_roots_l147_147506

theorem quadratic_roots (x : ℝ) : (x ^ 2 - 3 = 0) → (x = Real.sqrt 3 ∨ x = -Real.sqrt 3) :=
by
  intro h
  sorry

end quadratic_roots_l147_147506


namespace sum_of_decimals_as_fraction_l147_147281

axiom decimal_to_fraction :
  0.2 = 2 / 10 ∧
  0.04 = 4 / 100 ∧
  0.006 = 6 / 1000 ∧
  0.0008 = 8 / 10000 ∧
  0.00010 = 10 / 100000 ∧
  0.000012 = 12 / 1000000

theorem sum_of_decimals_as_fraction:
  0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 + 0.000012 = (3858:ℚ) / 15625 :=
by
  have h := decimal_to_fraction
  sorry

end sum_of_decimals_as_fraction_l147_147281


namespace six_digit_mod_27_l147_147202

theorem six_digit_mod_27 (X : ℕ) (hX : 100000 ≤ X ∧ X < 1000000) (Y : ℕ) (hY : ∃ a b : ℕ, 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 ∧ X = 1000 * a + b ∧ Y = 1000 * b + a) :
  X % 27 = Y % 27 := 
by
  sorry

end six_digit_mod_27_l147_147202


namespace sum_first_11_even_numbers_is_132_l147_147941

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem sum_first_11_even_numbers_is_132 : sum_first_n_even_numbers 11 = 132 := 
  by
    sorry

end sum_first_11_even_numbers_is_132_l147_147941


namespace david_cups_consumed_l147_147574

noncomputable def cups_of_water (time_in_minutes : ℕ) : ℝ :=
  time_in_minutes / 20

theorem david_cups_consumed : cups_of_water 225 = 11.25 := by
  sorry

end david_cups_consumed_l147_147574


namespace percent_c_of_b_l147_147618

variable (a b c : ℝ)

theorem percent_c_of_b (h1 : c = 0.20 * a) (h2 : b = 2 * a) : 
  ∃ x : ℝ, c = (x / 100) * b ∧ x = 10 :=
by
  sorry

end percent_c_of_b_l147_147618


namespace retrievers_count_l147_147176

-- Definitions of given conditions
def huskies := 5
def pitbulls := 2
def retrievers := Nat
def husky_pups := 3
def pitbull_pups := 3
def retriever_extra_pups := 2
def total_pups_excess := 30

-- Equation derived from the problem conditions
def total_pups (G : Nat) := huskies * husky_pups + pitbulls * pitbull_pups + G * (husky_pups + retriever_extra_pups)
def total_adults (G : Nat) := huskies + pitbulls + G

theorem retrievers_count : ∃ G : Nat, G = 4 ∧ total_pups G = total_adults G + total_pups_excess :=
by
  sorry

end retrievers_count_l147_147176


namespace minimum_doors_to_safety_l147_147251

-- Definitions in Lean 4 based on the conditions provided
def spaceship (corridors : ℕ) : Prop := corridors = 23

def command_closes (N : ℕ) (corridors : ℕ) : Prop := N ≤ corridors

-- Theorem based on the question and conditions
theorem minimum_doors_to_safety (N : ℕ) (corridors : ℕ)
  (h_corridors : spaceship corridors)
  (h_command : command_closes N corridors) :
  N = 22 :=
sorry

end minimum_doors_to_safety_l147_147251


namespace person6_number_l147_147507

theorem person6_number (a : ℕ → ℕ) (x : ℕ → ℕ) 
  (mod12 : ∀ i, a (i % 12) = a i)
  (h5 : x 5 = 5)
  (h6 : x 6 = 8)
  (h7 : x 7 = 11) 
  (h_avg : ∀ i, x i = (a (i-1) + a (i+1)) / 2) : 
  a 6 = 6 := sorry

end person6_number_l147_147507


namespace find_first_number_l147_147559

theorem find_first_number (a b : ℕ) (k : ℕ) (h1 : a = 3 * k) (h2 : b = 4 * k) (h3 : Nat.lcm a b = 84) : a = 21 := 
sorry

end find_first_number_l147_147559


namespace wendy_chocolates_l147_147230

theorem wendy_chocolates (h : ℕ) : 
  let chocolates_per_4_hours := 1152
  let chocolates_per_hour := chocolates_per_4_hours / 4
  (chocolates_per_hour * h) = 288 * h :=
by
  sorry

end wendy_chocolates_l147_147230


namespace joe_bath_shop_bottles_l147_147389

theorem joe_bath_shop_bottles (b : ℕ) (n : ℕ) (m : ℕ) 
    (h1 : 5 * n = b * m)
    (h2 : 5 * n = 95)
    (h3 : b * m = 95)
    (h4 : b ≠ 1)
    (h5 : b ≠ 95): 
    b = 19 := 
by 
    sorry

end joe_bath_shop_bottles_l147_147389


namespace largest_class_students_l147_147749

theorem largest_class_students (n1 n2 n3 n4 n5 : ℕ) (h1 : n1 = x) (h2 : n2 = x - 2) (h3 : n3 = x - 4) (h4 : n4 = x - 6) (h5 : n5 = x - 8) (h_sum : n1 + n2 + n3 + n4 + n5 = 140) : x = 32 :=
by {
  sorry
}

end largest_class_students_l147_147749


namespace johns_climb_height_correct_l147_147101

noncomputable def johns_total_height : ℝ :=
  let stair1_height := 4 * 15
  let stair2_height := 5 * 12.5
  let total_stair_height := stair1_height + stair2_height
  let rope1_height := (2 / 3) * stair1_height
  let rope2_height := (3 / 5) * stair2_height
  let total_rope_height := rope1_height + rope2_height
  let rope1_height_m := rope1_height / 3.281
  let rope2_height_m := rope2_height / 3.281
  let total_rope_height_m := rope1_height_m + rope2_height_m
  let ladder_height := 1.5 * total_rope_height_m * 3.281
  let rock_wall_height := (2 / 3) * ladder_height
  let total_pre_tree := total_stair_height + total_rope_height + ladder_height + rock_wall_height
  let tree_height := (3 / 4) * total_pre_tree - 10
  total_stair_height + total_rope_height + ladder_height + rock_wall_height + tree_height

theorem johns_climb_height_correct : johns_total_height = 679.115 := by
  sorry

end johns_climb_height_correct_l147_147101


namespace inscribed_square_ab_l147_147261

theorem inscribed_square_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^2 + b^2 = 32) : 2 * a * b = -7 :=
by
  sorry

end inscribed_square_ab_l147_147261


namespace zachary_more_pushups_l147_147677

def zachary_pushups : ℕ := 51
def david_pushups : ℕ := 44

theorem zachary_more_pushups : zachary_pushups - david_pushups = 7 := by
  sorry

end zachary_more_pushups_l147_147677


namespace gcd_polynomial_multiple_l147_147109

theorem gcd_polynomial_multiple (b : ℕ) (hb : 620 ∣ b) : gcd (4 * b^3 + 2 * b^2 + 5 * b + 93) b = 93 := by
  sorry

end gcd_polynomial_multiple_l147_147109


namespace degrees_to_radians_90_l147_147639

theorem degrees_to_radians_90 : (90 : ℝ) * (Real.pi / 180) = (Real.pi / 2) :=
by
  sorry

end degrees_to_radians_90_l147_147639


namespace tom_age_l147_147775

theorem tom_age (S T : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 := by
  sorry

end tom_age_l147_147775


namespace figure_count_mistake_l147_147843

theorem figure_count_mistake
    (b g : ℕ)
    (total_figures : ℕ)
    (boy_circles boy_squares girl_circles girl_squares : ℕ)
    (total_figures_counted : ℕ) :
  boy_circles = 3 → boy_squares = 8 → girl_circles = 9 → girl_squares = 2 →
  total_figures_counted = 4046 →
  (∃ (b g : ℕ), 11 * b + 11 * g ≠ 4046) :=
by
  intros
  sorry

end figure_count_mistake_l147_147843


namespace employee_discount_percentage_l147_147020

theorem employee_discount_percentage:
  let purchase_price := 500
  let markup_percentage := 0.15
  let savings := 57.5
  let retail_price := purchase_price * (1 + markup_percentage)
  let discount_percentage := (savings / retail_price) * 100
  discount_percentage = 10 :=
by
  sorry

end employee_discount_percentage_l147_147020


namespace alex_buys_17p3_pounds_of_corn_l147_147685

noncomputable def pounds_of_corn (c b : ℝ) : Prop :=
    c + b = 30 ∧ 1.05 * c + 0.39 * b = 23.10

theorem alex_buys_17p3_pounds_of_corn :
    ∃ c b, pounds_of_corn c b ∧ c = 17.3 :=
by
    sorry

end alex_buys_17p3_pounds_of_corn_l147_147685


namespace complete_square_transformation_l147_147881

theorem complete_square_transformation (x : ℝ) : 
  2 * x^2 - 4 * x - 3 = 0 ↔ (x - 1)^2 - (5 / 2) = 0 :=
sorry

end complete_square_transformation_l147_147881


namespace number_of_pecan_pies_is_4_l147_147831

theorem number_of_pecan_pies_is_4 (apple_pies pumpkin_pies total_pies pecan_pies : ℕ) 
  (h1 : apple_pies = 2) 
  (h2 : pumpkin_pies = 7) 
  (h3 : total_pies = 13) 
  (h4 : pecan_pies = total_pies - (apple_pies + pumpkin_pies)) 
  : pecan_pies = 4 := 
by 
  sorry

end number_of_pecan_pies_is_4_l147_147831


namespace paving_cost_l147_147767

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 300
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost : cost = 6187.50 := by
  -- length = 5.5
  -- width = 3.75
  -- rate = 300
  -- area = length * width = 20.625
  -- cost = area * rate = 6187.50
  sorry

end paving_cost_l147_147767


namespace relatively_prime_ratios_l147_147996

theorem relatively_prime_ratios (r s : ℕ) (h_coprime: Nat.gcd r s = 1) 
  (h_cond: (r : ℝ) / s = 2 * (Real.sqrt 2 + Real.sqrt 10) / (5 * Real.sqrt (3 + Real.sqrt 5))) :
  r = 4 ∧ s = 5 :=
by
  sorry

end relatively_prime_ratios_l147_147996


namespace minimal_blue_chips_value_l147_147303

noncomputable def minimal_blue_chips (r g b : ℕ) : Prop :=
b ≥ r / 3 ∧
b ≤ g / 4 ∧
r + g ≥ 75

theorem minimal_blue_chips_value : ∃ (b : ℕ), minimal_blue_chips 33 44 b ∧ b = 11 :=
by
  have b := 11
  use b
  sorry

end minimal_blue_chips_value_l147_147303


namespace total_polled_votes_correct_l147_147723

variable (V : ℕ) -- Valid votes

-- Condition: One candidate got 30% of the valid votes
variable (C1_votes : ℕ) (C2_votes : ℕ)
variable (H1 : C1_votes = (3 * V) / 10)

-- Condition: The other candidate won by 5000 votes
variable (H2 : C2_votes = C1_votes + 5000)

-- Condition: One candidate got 70% of the valid votes
variable (H3 : C2_votes = (7 * V) / 10)

-- Condition: 100 votes were invalid
variable (invalid_votes : ℕ := 100)

-- Total polled votes (valid + invalid)
def total_polled_votes := V + invalid_votes

theorem total_polled_votes_correct 
  (V : ℕ) 
  (H1 : C1_votes = (3 * V) / 10) 
  (H2 : C2_votes = C1_votes + 5000) 
  (H3 : C2_votes = (7 * V) / 10) 
  (invalid_votes : ℕ := 100) : 
  total_polled_votes V = 12600 :=
by
  -- The steps of the proof are omitted
  sorry

end total_polled_votes_correct_l147_147723


namespace discount_percentage_is_ten_l147_147795

-- Definitions based on given conditions
def cost_price : ℝ := 42
def markup (S : ℝ) : ℝ := 0.30 * S
def selling_price (S : ℝ) : Prop := S = cost_price + markup S
def profit : ℝ := 6

-- To prove the discount percentage
theorem discount_percentage_is_ten (S SP : ℝ) 
  (h_sell_price : selling_price S) 
  (h_SP : SP = S - profit) : 
  ((S - SP) / S) * 100 = 10 := 
by
  sorry

end discount_percentage_is_ten_l147_147795


namespace symmetric_points_parabola_l147_147734

theorem symmetric_points_parabola (x1 x2 y1 y2 m : ℝ) (h1 : y1 = 2 * x1^2) (h2 : y2 = 2 * x2^2)
    (h3 : x1 * x2 = -3 / 4) (h_sym: (y2 - y1) / (x2 - x1) = -1)
    (h_mid: (y2 + y1) / 2 = (x2 + x1) / 2 + m) :
    m = 2 := sorry

end symmetric_points_parabola_l147_147734


namespace total_people_l147_147592

theorem total_people (N B : ℕ) (h1 : N = 4 * B + 10) (h2 : N = 5 * B + 1) : N = 46 := by
  -- The proof will follow from the conditions, but it is not required in this script.
  sorry

end total_people_l147_147592


namespace alex_score_l147_147820

theorem alex_score (initial_students : ℕ) (initial_average : ℕ) (total_students : ℕ) (new_average : ℕ) (initial_total : ℕ) (new_total : ℕ) :
  initial_students = 19 →
  initial_average = 76 →
  total_students = 20 →
  new_average = 78 →
  initial_total = initial_students * initial_average →
  new_total = total_students * new_average →
  new_total - initial_total = 116 :=
by
  sorry

end alex_score_l147_147820


namespace roots_rational_l147_147293

/-- Prove that the roots of the equation x^2 + px + q = 0 are always rational,
given the rational numbers p and q, and a rational n where p = n + q / n. -/
theorem roots_rational
  (n p q : ℚ)
  (hp : p = n + q / n)
  : ∃ x y : ℚ, x^2 + p * x + q = 0 ∧ y^2 + p * y + q = 0 ∧ x ≠ y :=
sorry

end roots_rational_l147_147293


namespace crayons_left_l147_147294

-- Define the initial number of crayons and the number taken
def initial_crayons : ℕ := 7
def crayons_taken : ℕ := 3

-- Prove the number of crayons left in the drawer
theorem crayons_left : initial_crayons - crayons_taken = 4 :=
by
  sorry

end crayons_left_l147_147294


namespace slant_height_of_cone_l147_147324

theorem slant_height_of_cone (r : ℝ) (h : ℝ) (s : ℝ) (unfolds_to_semicircle : s = π) (base_radius : r = 1) : s = 2 :=
by
  sorry

end slant_height_of_cone_l147_147324


namespace polynomial_transformation_exists_l147_147939

theorem polynomial_transformation_exists (P : ℝ → ℝ → ℝ) (hP : ∀ x y, P (x - 1) (y - 2 * x + 1) = P x y) :
  ∃ Φ : ℝ → ℝ, ∀ x y, P x y = Φ (y - x^2) := by
  sorry

end polynomial_transformation_exists_l147_147939


namespace melanie_total_dimes_l147_147662

theorem melanie_total_dimes (d_1 d_2 d_3 : ℕ) (h₁ : d_1 = 19) (h₂ : d_2 = 39) (h₃ : d_3 = 25) : d_1 + d_2 + d_3 = 83 := by
  sorry

end melanie_total_dimes_l147_147662


namespace parallelogram_coordinates_l147_147453

/-- Given points A, B, and C, prove the coordinates of point D for the parallelogram -/
theorem parallelogram_coordinates (A B C: (ℝ × ℝ)) 
  (hA : A = (3, 7)) 
  (hB : B = (4, 6))
  (hC : C = (1, -2)) :
  D = (0, -1) ∨ D = (2, -3) ∨ D = (6, 15) :=
sorry

end parallelogram_coordinates_l147_147453


namespace cars_to_hours_l147_147683

def car_interval := 20 -- minutes
def num_cars := 30
def minutes_per_hour := 60

theorem cars_to_hours :
  (car_interval * num_cars) / minutes_per_hour = 10 := by
  sorry

end cars_to_hours_l147_147683


namespace determine_a_l147_147169

theorem determine_a (a : ℝ) : (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) := 
sorry

end determine_a_l147_147169


namespace machine_P_additional_hours_unknown_l147_147086

noncomputable def machine_A_rate : ℝ := 1.0000000000000013

noncomputable def machine_Q_rate : ℝ := machine_A_rate + 0.10 * machine_A_rate

noncomputable def total_sprockets : ℝ := 110

noncomputable def machine_Q_hours : ℝ := total_sprockets / machine_Q_rate

variable (x : ℝ) -- additional hours taken by Machine P

theorem machine_P_additional_hours_unknown :
  ∃ x, total_sprockets / machine_Q_rate + x = total_sprockets / ((total_sprockets + total_sprockets / machine_Q_rate * x) / total_sprockets) :=
sorry

end machine_P_additional_hours_unknown_l147_147086


namespace range_of_m_l147_147444

-- Definitions based on the conditions
def p (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 > 0

theorem range_of_m (m : ℝ) (hm : m > 0) :
  (∀ x : ℝ, p x → q x m) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m) ↔ 0 < m ∧ m ≤ 2 := sorry

end range_of_m_l147_147444


namespace proof_second_number_is_30_l147_147443

noncomputable def second_number_is_30 : Prop :=
  ∃ (a b c : ℕ), 
    a + b + c = 98 ∧ 
    (a / (gcd a b) = 2) ∧ (b / (gcd a b) = 3) ∧
    (b / (gcd b c) = 5) ∧ (c / (gcd b c) = 8) ∧
    b = 30

theorem proof_second_number_is_30 : second_number_is_30 :=
  sorry

end proof_second_number_is_30_l147_147443


namespace wrapping_paper_l147_147691

theorem wrapping_paper (total_used : ℚ) (decoration_used : ℚ) (presents : ℕ) (other_presents : ℕ) (individual_used : ℚ) 
  (h1 : total_used = 5 / 8) 
  (h2 : decoration_used = 1 / 24) 
  (h3 : presents = 4) 
  (h4 : other_presents = 3) 
  (h5 : individual_used = (5 / 8 - 1 / 24) / 3) : 
  individual_used = 7 / 36 := 
by
  -- The theorem will be proven here.
  sorry

end wrapping_paper_l147_147691


namespace total_customers_l147_147757

-- Define the initial number of customers
def initial_customers : ℕ := 14

-- Define the number of customers that left
def customers_left : ℕ := 3

-- Define the number of new customers gained
def new_customers : ℕ := 39

-- Prove that the total number of customers is 50
theorem total_customers : initial_customers - customers_left + new_customers = 50 := 
by
  sorry

end total_customers_l147_147757


namespace equivalence_of_statements_l147_147982

theorem equivalence_of_statements 
  (Q P : Prop) :
  (Q → ¬ P) ↔ (P → ¬ Q) := sorry

end equivalence_of_statements_l147_147982


namespace incorrect_option_D_l147_147555

-- definition of geometric objects and their properties
def octahedron_faces : Nat := 8
def tetrahedron_can_be_cut_into_4_pyramids : Prop := True
def frustum_extension_lines_intersect_at_a_point : Prop := True
def rectangle_rotated_around_side_forms_cylinder : Prop := True

-- incorrect identification of incorrect statement
theorem incorrect_option_D : 
  (∃ statement : String, statement = "D" ∧ ¬rectangle_rotated_around_side_forms_cylinder)  → False :=
by
  -- Proof of incorrect identification is not required per problem instructions
  sorry

end incorrect_option_D_l147_147555


namespace range_of_a_l147_147264

theorem range_of_a (a : ℝ) (h_a : a > 0) :
  (∃ t : ℝ, (5 * t + 1)^2 + (12 * t - 1)^2 = 2 * a * (5 * t + 1)) ↔ (0 < a ∧ a ≤ 17 / 25) := 
sorry

end range_of_a_l147_147264


namespace x_intercept_perpendicular_l147_147970

theorem x_intercept_perpendicular (k m x y : ℝ) (h1 : 4 * x - 3 * y = 12) (h2 : y = -3/4 * x + 3) :
  x = 4 :=
by
  sorry

end x_intercept_perpendicular_l147_147970


namespace amusement_park_ticket_length_l147_147394

theorem amusement_park_ticket_length (Area Width Length : ℝ) (h₀ : Area = 1.77) (h₁ : Width = 3) (h₂ : Area = Width * Length) : Length = 0.59 :=
by
  -- Proof will go here
  sorry

end amusement_park_ticket_length_l147_147394


namespace triangle_area_l147_147021

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- Statement of the theorem
theorem triangle_area : (1 / 2) * |(a.1 * b.2 - a.2 * b.1)| = 4.5 := by
  sorry

end triangle_area_l147_147021


namespace complex_division_l147_147716

open Complex

theorem complex_division :
  (1 + 2 * I) / (3 - 4 * I) = -1 / 5 + 2 / 5 * I :=
by
  sorry

end complex_division_l147_147716


namespace inequality_l147_147693

theorem inequality (A B : ℝ) (n : ℕ) (hA : 0 ≤ A) (hB : 0 ≤ B) (hn : 1 ≤ n) : (A + B)^n ≤ 2^(n - 1) * (A^n + B^n) := 
  sorry

end inequality_l147_147693


namespace fraction_is_one_third_l147_147722

noncomputable def fraction_studying_japanese (J S : ℕ) (h1 : S = 2 * J) (h2 : 3 / 8 * S + 1 / 4 * J = J) : ℚ :=
  J / (J + S)

theorem fraction_is_one_third (J S : ℕ) (h1 : S = 2 * J) (h2 : 3 / 8 * S + 1 / 4 * J = J) : 
  fraction_studying_japanese J S h1 h2 = 1 / 3 :=
  sorry

end fraction_is_one_third_l147_147722


namespace curve_crosses_itself_l147_147518

theorem curve_crosses_itself :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (t1^2 - 3 = t2^2 - 3) ∧ (t1^3 - 6*t1 + 2 = t2^3 - 6*t2 + 2) ∧
  ((t1^2 - 3 = 3) ∧ (t1^3 - 6*t1 + 2 = 2)) :=
by
  sorry

end curve_crosses_itself_l147_147518


namespace fraction_calculation_l147_147965

theorem fraction_calculation : 
  (1/2 - 1/3) / (3/7 * 2/8) = 14/9 :=
by
  sorry

end fraction_calculation_l147_147965


namespace number_of_nickels_l147_147673

def dimes : ℕ := 10
def pennies_per_dime : ℕ := 10
def pennies_per_nickel : ℕ := 5
def total_pennies : ℕ := 150

theorem number_of_nickels (total_value_dimes : ℕ := dimes * pennies_per_dime)
  (pennies_needed_from_nickels : ℕ := total_pennies - total_value_dimes)
  (n : ℕ) : n = pennies_needed_from_nickels / pennies_per_nickel → n = 10 := by
  sorry

end number_of_nickels_l147_147673


namespace uncovered_side_length_l147_147115

theorem uncovered_side_length
  (A : ℝ) (F : ℝ)
  (h1 : A = 600)
  (h2 : F = 130) :
  ∃ L : ℝ, L = 120 :=
by {
  sorry
}

end uncovered_side_length_l147_147115


namespace arithmetic_sequence_a2015_l147_147890

theorem arithmetic_sequence_a2015 :
  ∀ {a : ℕ → ℤ}, (a 1 = 2 ∧ a 5 = 6 ∧ (∀ n, a (n + 1) = a n + a 2 - a 1)) → a 2015 = 2016 :=
by
  sorry

end arithmetic_sequence_a2015_l147_147890


namespace smallest_x_for_cubic_l147_147754

theorem smallest_x_for_cubic (x N : ℕ) (h1 : 1260 * x = N^3) : x = 7350 :=
sorry

end smallest_x_for_cubic_l147_147754


namespace determine_better_robber_l147_147784

def sum_of_odd_series (k : ℕ) : ℕ := k * k
def sum_of_even_series (k : ℕ) : ℕ := k * (k + 1)

def first_robber_coins (n k : ℕ) (r : ℕ) : ℕ := 
  if r < 2 * k - 1 then (k - 1) * (k - 1) + r else k * k

def second_robber_coins (n k : ℕ) (r : ℕ) : ℕ := 
  if r < 2 * k - 1 then k * (k + 1) else k * k - k + r

theorem determine_better_robber (n k r : ℕ) :
  if 2 * k * k - 2 * k < n ∧ n < 2 * k * k then
    first_robber_coins n k r > second_robber_coins n k r
  else if 2 * k * k < n ∧ n < 2 * k * k + 2 * k then
    second_robber_coins n k r > first_robber_coins n k r
  else 
    false :=
sorry

end determine_better_robber_l147_147784


namespace rectangle_perimeter_l147_147756

theorem rectangle_perimeter (long_side short_side : ℝ) 
  (h_long : long_side = 1) 
  (h_short : short_side = long_side - 2/8) : 
  2 * long_side + 2 * short_side = 3.5 := 
by 
  sorry

end rectangle_perimeter_l147_147756


namespace triangle_sides_square_perfect_l147_147899

theorem triangle_sides_square_perfect (x y z : ℕ) (h : ∃ h_x h_y h_z, 
  h_x = h_y + h_z ∧ 
  2 * h_x * x = 2 * h_y * y ∧ 
  2 * h_x * x = 2 * h_z * z ) :
  ∃ k : ℕ, x^2 + y^2 + z^2 = k^2 :=
by
  sorry

end triangle_sides_square_perfect_l147_147899


namespace basic_astrophysics_degrees_l147_147695

theorem basic_astrophysics_degrees :
  let microphotonics_pct := 12
  let home_electronics_pct := 24
  let food_additives_pct := 15
  let gmo_pct := 29
  let industrial_lubricants_pct := 8
  let total_budget_percentage := 100
  let full_circle_degrees := 360
  let given_pct_sum := microphotonics_pct + home_electronics_pct + food_additives_pct + gmo_pct + industrial_lubricants_pct
  let astrophysics_pct := total_budget_percentage - given_pct_sum
  let astrophysics_degrees := (astrophysics_pct * full_circle_degrees) / total_budget_percentage
  astrophysics_degrees = 43.2 := by
  sorry

end basic_astrophysics_degrees_l147_147695


namespace price_of_chips_l147_147587

theorem price_of_chips (P : ℝ) (h1 : 1.5 = 1.5) (h2 : 45 = 45) (h3 : 15 = 15) (h4 : 10 = 10) :
  15 * P + 10 * 1.5 = 45 → P = 2 :=
by
  sorry

end price_of_chips_l147_147587


namespace HCF_of_two_numbers_l147_147308

theorem HCF_of_two_numbers (a b : ℕ) (h1 : a * b = 2562) (h2 : Nat.lcm a b = 183) : Nat.gcd a b = 14 := 
by
  sorry

end HCF_of_two_numbers_l147_147308


namespace xy_sum_square_l147_147624

theorem xy_sum_square (x y : ℕ) (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by
  sorry

end xy_sum_square_l147_147624


namespace min_A_div_B_l147_147522

theorem min_A_div_B (x A B : ℝ) (hx_pos : 0 < x) (hA_pos : 0 < A) (hB_pos : 0 < B) 
  (h1 : x^2 + 1 / x^2 = A) (h2 : x - 1 / x = B + 3) : 
  (A / B) = 6 + 2 * Real.sqrt 11 :=
sorry

end min_A_div_B_l147_147522


namespace construction_company_total_weight_l147_147603

noncomputable def total_weight_of_materials_in_pounds : ℝ :=
  let weight_of_concrete := 12568.3
  let weight_of_bricks := 2108 * 2.20462
  let weight_of_stone := 7099.5
  let weight_of_wood := 3778 * 2.20462
  let weight_of_steel := 5879 * (1 / 16)
  let weight_of_glass := 12.5 * 2000
  let weight_of_sand := 2114.8
  weight_of_concrete + weight_of_bricks + weight_of_stone + weight_of_wood + weight_of_steel + weight_of_glass + weight_of_sand

theorem construction_company_total_weight : total_weight_of_materials_in_pounds = 60129.72 :=
by
  sorry

end construction_company_total_weight_l147_147603


namespace find_weight_of_a_l147_147355

variables (a b c d e : ℕ)

-- Conditions
def cond1 : Prop := a + b + c = 252
def cond2 : Prop := a + b + c + d = 320
def cond3 : Prop := e = d + 7
def cond4 : Prop := b + c + d + e = 316

theorem find_weight_of_a (h1 : cond1 a b c) (h2 : cond2 a b c d) (h3 : cond3 d e) (h4 : cond4 b c d e) :
  a = 79 :=
by sorry

end find_weight_of_a_l147_147355


namespace sum_of_other_endpoint_coordinates_l147_147848

theorem sum_of_other_endpoint_coordinates (x y : ℤ)
  (h1 : (6 + x) / 2 = 3)
  (h2 : (-1 + y) / 2 = 6) :
  x + y = 13 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l147_147848


namespace find_surcharge_l147_147539

-- The property tax in 1996 is increased by 6% over the 1995 tax.
def increased_tax (T_1995 : ℝ) : ℝ := T_1995 * 1.06

-- Petersons' property tax for the year 1995 is $1800.
def T_1995 : ℝ := 1800

-- The Petersons' 1996 tax totals $2108.
def T_1996 : ℝ := 2108

-- Additional surcharge for a special project.
def surcharge (T_1996 : ℝ) (increased_tax : ℝ) : ℝ := T_1996 - increased_tax

theorem find_surcharge : surcharge T_1996 (increased_tax T_1995) = 200 := by
  sorry

end find_surcharge_l147_147539


namespace proof_of_A_inter_complement_B_l147_147927

variable (U : Set Nat) 
variable (A B : Set Nat)

theorem proof_of_A_inter_complement_B :
    (U = {1, 2, 3, 4}) →
    (B = {1, 2}) →
    (compl (A ∪ B) = {4}) →
    (A ∩ compl B = {3}) :=
by
  intros hU hB hCompl
  sorry

end proof_of_A_inter_complement_B_l147_147927


namespace problem_statement_l147_147390

theorem problem_statement : 6 * (3/2 + 2/3) = 13 :=
by
  sorry

end problem_statement_l147_147390


namespace total_journey_distance_l147_147600

/-- 
A woman completes a journey in 5 hours. She travels the first half of the journey 
at 21 km/hr and the second half at 24 km/hr. Find the total journey in km.
-/
theorem total_journey_distance :
  ∃ D : ℝ, (D / 2) / 21 + (D / 2) / 24 = 5 ∧ D = 112 :=
by
  use 112
  -- Please prove the following statements
  sorry

end total_journey_distance_l147_147600


namespace max_a_for_integer_roots_l147_147428

theorem max_a_for_integer_roots (a : ℕ) :
  (∀ x : ℤ, x^2 - 2 * (a : ℤ) * x + 64 = 0 → (∃ y : ℤ, x = y)) →
  (∀ x1 x2 : ℤ, x1 * x2 = 64 ∧ x1 + x2 = 2 * (a : ℤ)) →
  a ≤ 17 := 
sorry

end max_a_for_integer_roots_l147_147428


namespace hyperbola_condition_l147_147364

theorem hyperbola_condition (m : ℝ) :
  (∃ x y : ℝ, m * x^2 + (2 - m) * y^2 = 1) → m < 0 ∨ m > 2 :=
sorry

end hyperbola_condition_l147_147364


namespace find_some_number_l147_147345

-- The conditions of the problem
variables (x y : ℝ)
axiom cond1 : 2 * x + y = 7
axiom cond2 : x + 2 * y = 5

-- The "some number" we want to prove exists
def some_number := 3

-- Statement of the problem: the value of 2xy / some_number should equal 2
theorem find_some_number (x y : ℝ) (cond1 : 2 * x + y = 7) (cond2 : x + 2 * y = 5) :
  2 * x * y / some_number = 2 :=
sorry

end find_some_number_l147_147345


namespace proof_by_contradiction_x_gt_y_implies_x3_gt_y3_l147_147889

theorem proof_by_contradiction_x_gt_y_implies_x3_gt_y3
  (x y: ℝ) (h: x > y) : ¬ (x^3 ≤ y^3) :=
by
  -- We need to show that assuming x^3 <= y^3 leads to a contradiction
  sorry

end proof_by_contradiction_x_gt_y_implies_x3_gt_y3_l147_147889


namespace fractions_proper_or_improper_l147_147998

theorem fractions_proper_or_improper : 
  ∀ (a b : ℚ), (∃ p q : ℚ, a = p / q ∧ p < q) ∨ (∃ r s : ℚ, a = r / s ∧ r ≥ s) :=
by 
  sorry

end fractions_proper_or_improper_l147_147998


namespace largest_value_satisfies_abs_equation_l147_147183

theorem largest_value_satisfies_abs_equation (x : ℝ) : |5 - x| = 15 + x → x = -5 := by
  intros h
  sorry

end largest_value_satisfies_abs_equation_l147_147183


namespace inverse_proportion_decreasing_l147_147445

theorem inverse_proportion_decreasing (k : ℝ) (x : ℝ) (hx : x > 0) :
  (y = (k - 1) / x) → (k > 1) :=
by
  sorry

end inverse_proportion_decreasing_l147_147445


namespace a_4_is_11_l147_147010

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_4_is_11 : a 4 = 11 := by
  sorry

end a_4_is_11_l147_147010


namespace temperature_conversion_l147_147979

noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ :=
  (c * (9 / 5)) + 32

theorem temperature_conversion (c : ℝ) (hf : c = 60) :
  celsius_to_fahrenheit c = 140 :=
by {
  rw [hf, celsius_to_fahrenheit];
  norm_num
}

end temperature_conversion_l147_147979


namespace pair_B_equal_l147_147085

theorem pair_B_equal : (∀ x : ℝ, 4 * x^4 = |x|) :=
by sorry

end pair_B_equal_l147_147085


namespace solve_quadratic_eq_l147_147474

theorem solve_quadratic_eq (x : ℝ) : x^2 + 8 * x = 9 ↔ x = -9 ∨ x = 1 :=
by
  sorry

end solve_quadratic_eq_l147_147474


namespace jesse_bananas_total_l147_147738

theorem jesse_bananas_total (friends : ℝ) (bananas_per_friend : ℝ) (friends_eq : friends = 3) (bananas_per_friend_eq : bananas_per_friend = 21) : 
  friends * bananas_per_friend = 63 := by
  rw [friends_eq, bananas_per_friend_eq]
  norm_num

end jesse_bananas_total_l147_147738


namespace dollar_symmetric_l147_147397

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_symmetric {x y : ℝ} : dollar (x + y) (y + x) = 0 :=
by
  sorry

end dollar_symmetric_l147_147397


namespace tan_alpha_plus_pi_over_4_l147_147619

noncomputable def tan_sum_formula (α : ℝ) : ℝ :=
  (Real.tan α + Real.tan (Real.pi / 4)) / (1 - Real.tan α * Real.tan (Real.pi / 4))

theorem tan_alpha_plus_pi_over_4 
  (α : ℝ) 
  (h1 : Real.cos (2 * α) + Real.sin α * (2 * Real.sin α - 1) = 2 / 5) 
  (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : 
  tan_sum_formula α = 1 / 7 := 
sorry

end tan_alpha_plus_pi_over_4_l147_147619


namespace city_population_l147_147107

theorem city_population (P: ℝ) (h: 0.85 * P = 85000) : P = 100000 := 
by
  sorry

end city_population_l147_147107


namespace cookies_indeterminate_l147_147765

theorem cookies_indeterminate (bananas : ℕ) (boxes : ℕ) (bananas_per_box : ℕ) (cookies : ℕ)
  (h1 : bananas = 40)
  (h2 : boxes = 8)
  (h3 : bananas_per_box = 5)
  : ∃ c : ℕ, c = cookies :=
by sorry

end cookies_indeterminate_l147_147765


namespace hyperbola_eccentricity_l147_147671

/-- Given a hyperbola with the equation x^2/a^2 - y^2/b^2 = 1, point B(0, b),
the line F1B intersects with the two asymptotes at points P and Q. 
We are given that vector QP = 4 * vector PF1. Prove that the eccentricity 
of the hyperbola is 3/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (F1 : ℝ × ℝ) (B : ℝ × ℝ) (P Q : ℝ × ℝ) 
  (h_F1 : F1 = (-c, 0)) (h_B : B = (0, b)) 
  (h_int_P : P = (-a * c / (c + a), b * c / (c + a)))
  (h_int_Q : Q = (a * c / (c - a), b * c / (c - a)))
  (h_vec : (Q.1 - P.1, Q.2 - P.2) = (4 * (P.1 - F1.1), 4 * (P.2 - F1.2))) :
  (eccentricity : ℝ) = 3 / 2 :=
sorry

end hyperbola_eccentricity_l147_147671


namespace claire_flour_cost_l147_147195

def num_cakes : ℕ := 2
def flour_per_cake : ℕ := 2
def cost_per_flour : ℕ := 3
def total_cost (num_cakes flour_per_cake cost_per_flour : ℕ) : ℕ := 
  num_cakes * flour_per_cake * cost_per_flour

theorem claire_flour_cost : total_cost num_cakes flour_per_cake cost_per_flour = 12 := by
  sorry

end claire_flour_cost_l147_147195


namespace route_a_faster_by_8_minutes_l147_147104

theorem route_a_faster_by_8_minutes :
  let route_a_distance := 8 -- miles
  let route_a_speed := 40 -- miles per hour
  let route_b_distance := 9 -- miles
  let route_b_speed := 45 -- miles per hour
  let route_b_stop := 8 -- minutes
  let time_route_a := route_a_distance / route_a_speed * 60 -- time in minutes
  let time_route_b := (route_b_distance / route_b_speed) * 60 + route_b_stop -- time in minutes
  time_route_b - time_route_a = 8 :=
by
  sorry

end route_a_faster_by_8_minutes_l147_147104


namespace evaporation_fraction_l147_147544

theorem evaporation_fraction (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1)
  (h : (1 - x) * (3 / 4) = 1 / 6) : x = 7 / 9 :=
by
  sorry

end evaporation_fraction_l147_147544


namespace approx_val_l147_147189

variable (x : ℝ) (y : ℝ)

-- Definitions based on rounding condition
def approx_0_000315 : ℝ := 0.0003
def approx_7928564 : ℝ := 8000000

-- Main theorem statement
theorem approx_val (h1: x = approx_0_000315) (h2: y = approx_7928564) :
  x * y = 2400 := by
  sorry

end approx_val_l147_147189


namespace inequality_holds_for_any_xyz_l147_147627

theorem inequality_holds_for_any_xyz (x y z : ℝ) : 
  x^4 + y^4 + z^2 + 1 ≥ 2 * x * (x * y^2 - x + z + 1) := 
by 
  sorry

end inequality_holds_for_any_xyz_l147_147627


namespace initial_orange_balloons_l147_147437

-- Definitions
variable (x : ℕ)
variable (h1 : x - 2 = 7)

-- Theorem to prove
theorem initial_orange_balloons (h1 : x - 2 = 7) : x = 9 :=
sorry

end initial_orange_balloons_l147_147437


namespace base_number_is_two_l147_147386

theorem base_number_is_two (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^18) (h2 : n = 17) : x = 2 :=
by sorry

end base_number_is_two_l147_147386


namespace complex_number_solution_l147_147063

theorem complex_number_solution (z : ℂ) (h : z / Complex.I = 3 - Complex.I) : z = 1 + 3 * Complex.I :=
sorry

end complex_number_solution_l147_147063


namespace find_d_plus_f_l147_147043

noncomputable def a : ℂ := sorry
noncomputable def c : ℂ := sorry
noncomputable def e : ℂ := -2 * a - c
noncomputable def d : ℝ := sorry
noncomputable def f : ℝ := sorry

theorem find_d_plus_f (a c e : ℂ) (d f : ℝ) (h₁ : e = -2 * a - c) (h₂ : a.im + d + f = 4) (h₃ : a.re + c.re + e.re = 0) (h₄ : 2 + d + f = 4) : d + f = 2 :=
by
  -- proof goes here
  sorry

end find_d_plus_f_l147_147043


namespace chord_length_perpendicular_l147_147857

theorem chord_length_perpendicular 
  (R a b : ℝ)  
  (h1 : a + b = R)
  (h2 : (1 / 2) * Real.pi * R^2 - (1 / 2) * Real.pi * (a^2 + b^2) = 10 * Real.pi) :
  2 * Real.sqrt 10 = 6.32 :=
by 
  sorry

end chord_length_perpendicular_l147_147857


namespace Q_share_of_profit_l147_147211

theorem Q_share_of_profit (P Q T : ℕ) (hP : P = 54000) (hQ : Q = 36000) (hT : T = 18000) : Q's_share = 7200 :=
by
  -- Definitions and conditions
  let P := 54000
  let Q := 36000
  let T := 18000
  have P_ratio := 3
  have Q_ratio := 2
  have ratio_sum := P_ratio + Q_ratio
  have Q's_share := (T * Q_ratio) / ratio_sum
  
  -- Q's share of the profit
  sorry

end Q_share_of_profit_l147_147211


namespace candy_problem_minimum_candies_l147_147732

theorem candy_problem_minimum_candies : ∃ (N : ℕ), N > 1 ∧ N % 2 = 1 ∧ N % 3 = 1 ∧ N % 5 = 1 ∧ N = 31 :=
by
  sorry

end candy_problem_minimum_candies_l147_147732


namespace infinite_geometric_series_sum_l147_147529

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 3
  let r := -(3 : ℚ) / 4
  |r| < 1 →
  (∀ S, S = a / (1 - r) → S = 20 / 21) :=
by
  intros a r h_abs_r S h_S
  sorry

end infinite_geometric_series_sum_l147_147529


namespace simplify_expression_l147_147340

theorem simplify_expression : (2^8 + 4^5) * ((1^3 - (-1)^3)^8) = 327680 := by
  sorry

end simplify_expression_l147_147340


namespace books_bought_l147_147331

def cost_price_of_books (n : ℕ) (C : ℝ) (S : ℝ) : Prop :=
  n * C = 16 * S

def gain_or_loss_percentage (gain_loss_percent : ℝ) : Prop :=
  gain_loss_percent = 0.5

def loss_selling_price (C : ℝ) (S : ℝ) (gain_loss_percent : ℝ) : Prop :=
  S = (1 - gain_loss_percent) * C
  
theorem books_bought (n : ℕ) (C : ℝ) (S : ℝ) (gain_loss_percent : ℝ) 
  (h1 : cost_price_of_books n C S) 
  (h2 : gain_or_loss_percentage gain_loss_percent) 
  (h3 : loss_selling_price C S gain_loss_percent) : 
  n = 8 := 
sorry 

end books_bought_l147_147331


namespace TileD_in_AreaZ_l147_147612

namespace Tiles

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def TileA : Tile := {top := 5, right := 3, bottom := 2, left := 4}
def TileB : Tile := {top := 2, right := 4, bottom := 5, left := 3}
def TileC : Tile := {top := 3, right := 6, bottom := 1, left := 5}
def TileD : Tile := {top := 5, right := 2, bottom := 3, left := 6}

variables (X Y Z W : Tile)
variable (tiles : List Tile := [TileA, TileB, TileC, TileD])

noncomputable def areaZContains : Tile := sorry

theorem TileD_in_AreaZ  : areaZContains = TileD := sorry

end Tiles

end TileD_in_AreaZ_l147_147612


namespace overlapping_area_zero_l147_147865

-- Definition of the points and triangles
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def point0 : Point := { x := 0, y := 0 }
def point1 : Point := { x := 2, y := 2 }
def point2 : Point := { x := 2, y := 0 }
def point3 : Point := { x := 0, y := 2 }
def point4 : Point := { x := 1, y := 1 }

def triangle1 : Triangle := { p1 := point0, p2 := point1, p3 := point2 }
def triangle2 : Triangle := { p1 := point3, p2 := point1, p3 := point0 }

-- Function to calculate the area of a triangle
def area (t : Triangle) : ℝ :=
  0.5 * abs (t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y))

-- Using collinear points theorem to prove that the area of the overlapping region is zero
theorem overlapping_area_zero : area { p1 := point0, p2 := point1, p3 := point4 } = 0 := 
by 
  -- This follows directly from the fact that the points (0,0), (2,2), and (1,1) are collinear
  -- skipping the actual geometric proof for conciseness
  sorry

end overlapping_area_zero_l147_147865


namespace television_price_reduction_l147_147735

theorem television_price_reduction (P : ℝ) (h₁ : 0 ≤ P):
  ((P - (P * 0.7 * 0.8)) / P) * 100 = 44 :=
by
  sorry

end television_price_reduction_l147_147735


namespace songs_before_camp_l147_147217

theorem songs_before_camp (total_songs : ℕ) (learned_at_camp : ℕ) (songs_before_camp : ℕ) (h1 : total_songs = 74) (h2 : learned_at_camp = 18) : songs_before_camp = 56 :=
by
  sorry

end songs_before_camp_l147_147217


namespace workers_planted_33_walnut_trees_l147_147503

def initial_walnut_trees : ℕ := 22
def total_walnut_trees_after_planting : ℕ := 55
def walnut_trees_planted (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem workers_planted_33_walnut_trees :
  walnut_trees_planted initial_walnut_trees total_walnut_trees_after_planting = 33 :=
by
  unfold walnut_trees_planted
  rfl

end workers_planted_33_walnut_trees_l147_147503


namespace correct_inequality_l147_147254

-- Define the conditions
variables (a b : ℝ)
variable (h : a > 1 ∧ 1 > b ∧ b > 0)

-- State the theorem to prove
theorem correct_inequality (h : a > 1 ∧ 1 > b ∧ b > 0) : 
  (1 / Real.log a) > (1 / Real.log b) :=
sorry

end correct_inequality_l147_147254


namespace most_likely_outcome_is_draw_l147_147850

variable (P_A_wins : ℝ) (P_A_not_loses : ℝ)

def P_draw (P_A_wins P_A_not_loses : ℝ) : ℝ := 
  P_A_not_loses - P_A_wins

def P_B_wins (P_A_not_loses P_A_wins : ℝ) : ℝ :=
  1 - P_A_not_loses

theorem most_likely_outcome_is_draw 
  (h₁: P_A_wins = 0.3) 
  (h₂: P_A_not_loses = 0.7)
  (h₃: 0 ≤ P_A_wins) 
  (h₄: P_A_wins ≤ 1) 
  (h₅: 0 ≤ P_A_not_loses) 
  (h₆: P_A_not_loses ≤ 1) : 
  max (P_A_wins) (max (P_B_wins P_A_not_loses P_A_wins) (P_draw P_A_wins P_A_not_loses)) = P_draw P_A_wins P_A_not_loses :=
by
  sorry

end most_likely_outcome_is_draw_l147_147850


namespace smallest_value_of_N_l147_147370

theorem smallest_value_of_N :
  ∃ N : ℕ, ∀ (P1 P2 P3 P4 P5 : ℕ) (x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 : ℕ),
    (P1 = 1 ∧ P2 = 2 ∧ P3 = 3 ∧ P4 = 4 ∧ P5 = 5) →
    (x1 = a_1 ∧ x2 = N + a_2 ∧ x3 = 2 * N + a_3 ∧ x4 = 3 * N + a_4 ∧ x5 = 4 * N + a_5) →
    (y1 = 5 * (a_1 - 1) + 1 ∧ y2 = 5 * (a_2 - 1) + 2 ∧ y3 = 5 * (a_3 - 1) + 3 ∧ y4 = 5 * (a_4 - 1) + 4 ∧ y5 = 5 * (a_5 - 1) + 5) →
    (x1 = y2 ∧ x2 = y1 ∧ x3 = y4 ∧ x4 = y5 ∧ x5 = y3) →
    N = 149 :=
sorry

end smallest_value_of_N_l147_147370


namespace statement_1_incorrect_statement_3_incorrect_statement_4_incorrect_l147_147557

-- Define the notion of line and plane
def Line := Type
def Plane := Type

-- Define the relations: parallel, contained-in, and intersection
def parallel (a b : Line) : Prop := sorry
def contained_in (a : Line) (α : Plane) : Prop := sorry
def intersects_at (a : Line) (α : Plane) (P : Type) : Prop := sorry

-- Conditions translated into Lean
def cond1 (a : Line) (α : Plane) (b : Line) : Prop := parallel a α ∧ contained_in b α → parallel a b
def cond2 (a : Line) (α : Plane) (b : Line) {P : Type} : Prop := intersects_at a α P ∧ contained_in b α → ¬ parallel a b
def cond3 (a : Line) (α : Plane) : Prop := ¬ contained_in a α → parallel a α
def cond4 (a : Line) (α : Plane) (b : Line) : Prop := parallel a α ∧ parallel b α → parallel a b

-- The statements that need to be proved incorrect
theorem statement_1_incorrect (a : Line) (α : Plane) (b : Line) : ¬ (cond1 a α b) := sorry
theorem statement_3_incorrect (a : Line) (α : Plane) : ¬ (cond3 a α) := sorry
theorem statement_4_incorrect (a : Line) (α : Plane) (b : Line) : ¬ (cond4 a α b) := sorry

end statement_1_incorrect_statement_3_incorrect_statement_4_incorrect_l147_147557


namespace flowers_bouquets_l147_147900

theorem flowers_bouquets (tulips: ℕ) (roses: ℕ) (extra: ℕ) (total: ℕ) (used_for_bouquets: ℕ) 
(h1: tulips = 36) 
(h2: roses = 37) 
(h3: extra = 3) 
(h4: total = tulips + roses)
(h5: used_for_bouquets = total - extra) :
used_for_bouquets = 70 := by
  sorry

end flowers_bouquets_l147_147900


namespace savings_calculation_l147_147375

-- Definitions of the given conditions
def window_price : ℕ := 100
def free_window_offer (purchased : ℕ) : ℕ := purchased / 4

-- Number of windows needed
def dave_needs : ℕ := 7
def doug_needs : ℕ := 8

-- Calculations based on the conditions
def individual_costs : ℕ :=
  (dave_needs - free_window_offer dave_needs) * window_price +
  (doug_needs - free_window_offer doug_needs) * window_price

def together_costs : ℕ :=
  let total_needs := dave_needs + doug_needs
  (total_needs - free_window_offer total_needs) * window_price

def savings : ℕ := individual_costs - together_costs

-- Proof statement
theorem savings_calculation : savings = 100 := by
  sorry

end savings_calculation_l147_147375


namespace value_of_fraction_l147_147852

theorem value_of_fraction (x y : ℤ) (h : x / y = 7 / 2) : (x - 2 * y) / y = 3 / 2 := by
  sorry

end value_of_fraction_l147_147852


namespace min_value_of_X_l147_147263

theorem min_value_of_X (n : ℕ) (h : n ≥ 2) 
  (X : Finset ℕ) 
  (B : Fin n → Finset ℕ) 
  (hB : ∀ i, (B i).card = 2) :
  ∃ (Y : Finset ℕ), Y.card = n ∧ ∀ i, (Y ∩ (B i)).card ≤ 1 →
  X.card = 2 * n - 1 :=
sorry

end min_value_of_X_l147_147263


namespace sum_of_integers_l147_147266

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 103) 
                        (h2 : Nat.gcd a b = 1) 
                        (h3 : a < 20) 
                        (h4 : b < 20) : 
                        a + b = 19 :=
  by sorry

end sum_of_integers_l147_147266


namespace find_y_l147_147580

variable (x y z : ℕ)

-- Conditions
def condition1 : Prop := 100 + 200 + 300 + x = 1000
def condition2 : Prop := 300 + z + 100 + x + y = 1000

-- Theorem to be proven
theorem find_y (h1 : condition1 x) (h2 : condition2 x y z) : z + y = 200 :=
sorry

end find_y_l147_147580


namespace rectangle_area_l147_147075

structure Rectangle where
  length : ℕ    -- Length of the rectangle in cm
  width : ℕ     -- Width of the rectangle in cm
  perimeter : ℕ -- Perimeter of the rectangle in cm
  h : length = width + 4 -- Distance condition from the diagonal intersection

theorem rectangle_area (r : Rectangle) (h_perim : r.perimeter = 56) : r.length * r.width = 192 := by
  sorry

end rectangle_area_l147_147075


namespace machines_work_together_l147_147718

theorem machines_work_together (x : ℝ) (h₁ : 1/(x+4) + 1/(x+2) + 1/(x+3) = 1/x) : x = 1 :=
sorry

end machines_work_together_l147_147718


namespace weight_of_new_person_l147_147783

-- Define the problem conditions
variables (W : ℝ) -- Weight of the new person
variable (initial_weight : ℝ := 65) -- Weight of the person being replaced
variable (increase_in_avg : ℝ := 4) -- Increase in average weight
variable (num_persons : ℕ := 8) -- Number of persons

-- Define the total increase in weight due to the new person
def total_increase : ℝ := num_persons * increase_in_avg

-- The Lean statement to prove
theorem weight_of_new_person (W : ℝ) (h : total_increase = W - initial_weight) : W = 97 := sorry

end weight_of_new_person_l147_147783


namespace A_superset_C_l147_147114

-- Definitions of the sets as given in the problem statement
def U : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {-1, 3}
def C : Set ℝ := {x | -1 < x ∧ x < 3}

-- Statement to be proved: A ⊇ C
theorem A_superset_C : A ⊇ C :=
by sorry

end A_superset_C_l147_147114


namespace geometric_sequence_sum_t_value_l147_147003

theorem geometric_sequence_sum_t_value 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (t : ℝ)
  (h1 : ∀ n : ℕ, S_n n = 3^((n:ℝ)-1) + t)
  (h2 : a_n 1 = 3^0 + t)
  (geometric : ∀ n : ℕ, n ≥ 2 → a_n n = 2 * 3^(n-2)) :
  t = -1/3 :=
by
  sorry

end geometric_sequence_sum_t_value_l147_147003


namespace isosceles_triangle_perimeter_1_isosceles_triangle_perimeter_2_l147_147422

-- Problem 1
theorem isosceles_triangle_perimeter_1 (a b : ℕ) (h1: a = 4 ∨ a = 6) (h2: b = 4 ∨ b = 6) (h3: a ≠ b): 
  (a + b + b = 14 ∨ a + b + b = 16) :=
sorry

-- Problem 2
theorem isosceles_triangle_perimeter_2 (a b : ℕ) (h1: a = 2 ∨ a = 6) (h2: b = 2 ∨ b = 6) (h3: a ≠ b ∨ (a = 2 ∧ 2 + 2 ≥ 6 ∧ 6 = b)):
  (a + b + b = 14) :=
sorry

end isosceles_triangle_perimeter_1_isosceles_triangle_perimeter_2_l147_147422


namespace negation_proposition_l147_147198

theorem negation_proposition :
  (¬ ∃ x : ℝ, (x > -1 ∧ x < 3) ∧ (x^2 - 1 ≤ 2 * x)) ↔ 
  (∀ x : ℝ, (x > -1 ∧ x < 3) → (x^2 - 1 > 2 * x)) :=
by {
  sorry
}

end negation_proposition_l147_147198


namespace time_after_midnight_1453_minutes_l147_147805

def minutes_to_time (minutes : Nat) : Nat × Nat :=
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes)

def time_of_day (hours : Nat) : Nat × Nat :=
  let days := hours / 24
  let remaining_hours := hours % 24
  (days, remaining_hours)

theorem time_after_midnight_1453_minutes : 
  let midnight := (0, 0) -- Midnight as a tuple of hours and minutes
  let total_minutes := 1453
  let (total_hours, minutes) := minutes_to_time total_minutes
  let (days, hours) := time_of_day total_hours
  days = 1 ∧ hours = 0 ∧ minutes = 13
  := by
    let midnight := (0, 0)
    let total_minutes := 1453
    let (total_hours, minutes) := minutes_to_time total_minutes
    let (days, hours) := time_of_day total_hours
    sorry

end time_after_midnight_1453_minutes_l147_147805


namespace correct_equation_l147_147289

theorem correct_equation : -(-5) = |-5| :=
by
  -- sorry is used here to skip the actual proof steps which are not required.
  sorry

end correct_equation_l147_147289


namespace range_of_independent_variable_x_l147_147971

noncomputable def range_of_x (x : ℝ) : Prop :=
  x > -2

theorem range_of_independent_variable_x (x : ℝ) :
  ∀ x, (x + 2 > 0) → range_of_x x :=
by
  intro x h
  unfold range_of_x
  linarith

end range_of_independent_variable_x_l147_147971


namespace age_problem_l147_147558

theorem age_problem (A N : ℕ) (h₁: A = 18) (h₂: N * (A + 3) - N * (A - 3) = A) : N = 3 := by
  sorry

end age_problem_l147_147558


namespace product_a_b_l147_147361

variable (a b c : ℝ)
variable (h_pos_a : a > 0)
variable (h_pos_b : b > 0)
variable (h_pos_c : c > 0)
variable (h_c : c = 3)
variable (h_a : a = b^2)
variable (h_bc : b + c = b * c)

theorem product_a_b : a * b = 27 / 8 :=
by
  -- We need to prove that given the above conditions, a * b = 27 / 8
  sorry

end product_a_b_l147_147361


namespace polynomial_at_x_neg_four_l147_147883

noncomputable def f (x : ℝ) : ℝ :=
  12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

theorem polynomial_at_x_neg_four : 
  f (-4) = 220 := by
  sorry

end polynomial_at_x_neg_four_l147_147883


namespace minimum_value_of_reciprocals_l147_147283

theorem minimum_value_of_reciprocals {m n : ℝ} 
  (hmn : m > 0 ∧ n > 0 ∧ (m * n > 0)) 
  (hline : 2 * m + 2 * n = 1) : 
  (1 / m + 1 / n) = 8 :=
sorry

end minimum_value_of_reciprocals_l147_147283


namespace inequality_proof_l147_147919

open Real

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) : 
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2 :=
sorry

end inequality_proof_l147_147919


namespace find_guest_sets_l147_147475

-- Definitions based on conditions
def cost_per_guest_set : ℝ := 32.0
def cost_per_master_set : ℝ := 40.0
def num_master_sets : ℕ := 4
def total_cost : ℝ := 224.0

-- The mathematical problem
theorem find_guest_sets (G : ℕ) (total_cost_eq : total_cost = cost_per_guest_set * G + cost_per_master_set * num_master_sets) : G = 2 :=
by
  sorry

end find_guest_sets_l147_147475


namespace min_bottles_l147_147698

theorem min_bottles (a b : ℕ) (h1 : a > b) (h2 : b > 1) : 
  ∃ x : ℕ, x = Nat.ceil (a - a / b) := sorry

end min_bottles_l147_147698


namespace sum_of_7_more_likely_than_sum_of_8_l147_147862

noncomputable def probability_sum_equals_seven : ℚ := 6 / 36
noncomputable def probability_sum_equals_eight : ℚ := 5 / 36

theorem sum_of_7_more_likely_than_sum_of_8 :
  probability_sum_equals_seven > probability_sum_equals_eight :=
by 
  sorry

end sum_of_7_more_likely_than_sum_of_8_l147_147862


namespace cafeteria_green_apples_l147_147153

def number_of_green_apples (G : ℕ) : Prop :=
  42 + G - 9 = 40 → G = 7

theorem cafeteria_green_apples
  (red_apples : ℕ)
  (students_wanting_fruit : ℕ)
  (extra_fruit : ℕ)
  (G : ℕ)
  (h1 : red_apples = 42)
  (h2 : students_wanting_fruit = 9)
  (h3 : extra_fruit = 40)
  : number_of_green_apples G :=
by
  -- Place for proof omitted intentionally
  sorry

end cafeteria_green_apples_l147_147153


namespace compound_statement_false_l147_147510

theorem compound_statement_false (p q : Prop) (h : ¬ (p ∧ q)) : ¬ p ∨ ¬ q :=
sorry

end compound_statement_false_l147_147510


namespace box_contents_l147_147502

-- Definitions for the boxes and balls
inductive Ball
| Black | White | Green

-- Define the labels on each box
def label_box1 := "white"
def label_box2 := "black"
def label_box3 := "white or green"

-- Conditions based on the problem
def box1_label := label_box1
def box2_label := label_box2
def box3_label := label_box3

-- Statement of the problem
theorem box_contents (b1 b2 b3 : Ball) 
  (h1 : b1 ≠ Ball.White) 
  (h2 : b2 ≠ Ball.Black) 
  (h3 : b3 = Ball.Black) 
  (h4 : ∀ (x y z : Ball), x ≠ y ∧ y ≠ z ∧ z ≠ x → 
        (x = b1 ∨ y = b1 ∨ z = b1) ∧
        (x = b2 ∨ y = b2 ∨ z = b2) ∧
        (x = b3 ∨ y = b3 ∨ z = b3)) : 
  b1 = Ball.Green ∧ b2 = Ball.White ∧ b3 = Ball.Black :=
sorry

end box_contents_l147_147502


namespace sum_of_three_consecutive_integers_product_504_l147_147413

theorem sum_of_three_consecutive_integers_product_504 : 
  ∃ n : ℤ, n * (n + 1) * (n + 2) = 504 ∧ n + (n + 1) + (n + 2) = 24 := 
by
  sorry

end sum_of_three_consecutive_integers_product_504_l147_147413


namespace books_on_shelves_l147_147150

-- Definitions based on the problem conditions.
def bookshelves : ℕ := 1250
def books_per_shelf : ℕ := 45
def total_books : ℕ := 56250

-- Theorem statement
theorem books_on_shelves : bookshelves * books_per_shelf = total_books := 
by
  sorry

end books_on_shelves_l147_147150


namespace parallel_lines_coefficient_l147_147526

theorem parallel_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, (a * x + 2 * y + 2 = 0) → (3 * x - y - 2 = 0)) → a = -6 :=
  by
    sorry

end parallel_lines_coefficient_l147_147526


namespace johns_age_l147_147829

theorem johns_age (J : ℕ) (h : J + 9 = 3 * (J - 11)) : J = 21 :=
sorry

end johns_age_l147_147829


namespace equation_of_line_l_l147_147715

noncomputable def line_eq (a b c : ℚ) : ℚ → ℚ → Prop := λ x y => a * x + b * y + c = 0

theorem equation_of_line_l : 
  ∃ m : ℚ, 
  (∀ x y : ℚ, 
    (2 * x - 3 * y - 3 = 0 ∧ x + y + 2 = 0 → line_eq 3 1 m x y) ∧ 
    (3 * x + y - 1 = 0 → line_eq 3 1 0 x y)
  ) →
  line_eq 15 5 16 (-3/5) (-7/5) :=
by 
  sorry

end equation_of_line_l_l147_147715


namespace total_fireworks_l147_147354

-- Definitions of the given conditions
def koby_boxes : Nat := 2
def koby_box_sparklers : Nat := 3
def koby_box_whistlers : Nat := 5
def cherie_boxes : Nat := 1
def cherie_box_sparklers : Nat := 8
def cherie_box_whistlers : Nat := 9

-- Statement to prove the total number of fireworks
theorem total_fireworks : 
  let koby_fireworks := koby_boxes * (koby_box_sparklers + koby_box_whistlers)
  let cherie_fireworks := cherie_boxes * (cherie_box_sparklers + cherie_box_whistlers)
  koby_fireworks + cherie_fireworks = 33 := by
  sorry

end total_fireworks_l147_147354


namespace fraction_to_decimal_equiv_l147_147306

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l147_147306


namespace range_of_a_l147_147928

theorem range_of_a (a : ℝ) (h_pos : a > 0)
  (p : ∀ x : ℝ, x^2 - 4 * a * x + 3 * a^2 ≤ 0)
  (q : ∀ x : ℝ, (x^2 - x - 6 < 0) ∧ (x^2 + 2 * x - 8 > 0)) :
  (a ∈ ((Set.Ioo 0 (2 / 3)) ∪ (Set.Ici 3))) :=
by
  sorry

end range_of_a_l147_147928


namespace combined_work_days_l147_147027

theorem combined_work_days (W D : ℕ) (h1: ∀ a b : ℕ, a + b = 4) (h2: (1/6:ℝ) = (1/6:ℝ)) :
  D = 4 :=
by
  sorry

end combined_work_days_l147_147027


namespace garden_width_l147_147946

theorem garden_width (w : ℕ) (h1 : ∀ l : ℕ, l = w + 12 → l * w ≥ 120) : w = 6 := 
by
  sorry

end garden_width_l147_147946


namespace probability_both_cards_are_diamonds_l147_147424

-- Conditions definitions
def total_cards : ℕ := 52
def diamonds_in_deck : ℕ := 13
def two_draws : ℕ := 2

-- Calculation definitions
def total_possible_outcomes : ℕ := (total_cards * (total_cards - 1)) / two_draws
def favorable_outcomes : ℕ := (diamonds_in_deck * (diamonds_in_deck - 1)) / two_draws

-- Definition of the probability asked in the question
def probability_both_diamonds : ℚ := favorable_outcomes / total_possible_outcomes

theorem probability_both_cards_are_diamonds :
  probability_both_diamonds = 1 / 17 := 
sorry

end probability_both_cards_are_diamonds_l147_147424


namespace smallest_rectangles_required_l147_147876

theorem smallest_rectangles_required :
  ∀ (r h : ℕ) (area_square length_square : ℕ),
  r = 3 → h = 4 →
  (∀ k, (k: ℕ) ∣ (r * h) → (k: ℕ) = r * h) →
  length_square = 12 →
  area_square = length_square * length_square →
  (area_square / (r * h) = 12) :=
by
  intros
  /- The mathematical proof steps will be filled here -/
  sorry

end smallest_rectangles_required_l147_147876


namespace all_positive_integers_are_nice_l147_147920

def isNice (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin k → ℕ), (∀ i, ∃ m : ℕ, a i = 2 ^ m) ∧ n = (Finset.univ.sum a) / k

theorem all_positive_integers_are_nice : ∀ n : ℕ, 0 < n → isNice n := sorry

end all_positive_integers_are_nice_l147_147920


namespace part1_part2_l147_147991

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 9 → x ∈ Set.Icc (-2 : ℝ) 4 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = -x^2 + a) →
  (a ∈ Set.Icc (19 / 4) (7 : ℝ)) :=
sorry

end part1_part2_l147_147991


namespace expected_score_shooting_competition_l147_147135

theorem expected_score_shooting_competition (hit_rate : ℝ)
  (miss_both_score : ℝ) (hit_one_score : ℝ) (hit_both_score : ℝ)
  (prob_0 : ℝ) (prob_10 : ℝ) (prob_15 : ℝ) :
  hit_rate = 4 / 5 →
  miss_both_score = 0 →
  hit_one_score = 10 →
  hit_both_score = 15 →
  prob_0 = (1 - 4 / 5) * (1 - 4 / 5) →
  prob_10 = 2 * (4 / 5) * (1 - 4 / 5) →
  prob_15 = (4 / 5) * (4 / 5) →
  (0 * prob_0 + 10 * prob_10 + 15 * prob_15) = 12.8 :=
by
  intros h_hit_rate h_miss_both_score h_hit_one_score h_hit_both_score
         h_prob_0 h_prob_10 h_prob_15
  sorry

end expected_score_shooting_competition_l147_147135


namespace no_solutions_for_a_ne_4_solutions_for_a_eq_4_infinite_l147_147432

-- Part (i)
theorem no_solutions_for_a_ne_4 (a : ℕ) (h : a ≠ 4) :
  ¬∃ (u v : ℕ), (u > 0 ∧ v > 0 ∧ u^2 + v^2 - a * u * v + 2 = 0) :=
by sorry

-- Part (ii)
theorem solutions_for_a_eq_4_infinite :
  ∃ (a_seq : ℕ → ℕ),
    (a_seq 0 = 1 ∧ a_seq 1 = 3 ∧
     ∀ n, a_seq (n + 2) = 4 * a_seq (n + 1) - a_seq n ∧
    ∀ n, (a_seq n) > 0 ∧ (a_seq (n + 1)) > 0 ∧ (a_seq n)^2 + (a_seq (n + 1))^2 - 4 * (a_seq n) * (a_seq (n + 1)) + 2 = 0) :=
by sorry

end no_solutions_for_a_ne_4_solutions_for_a_eq_4_infinite_l147_147432


namespace sixDigitIntegersCount_l147_147122

-- Define the digits to use.
def digits : List ℕ := [1, 2, 2, 5, 9, 9]

-- Define the factorial function as it might not be pre-defined in Mathlib.
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Calculate the number of unique permutations accounting for repeated digits.
def numberOfUniquePermutations : ℕ :=
  factorial 6 / (factorial 2 * factorial 2)

-- State the theorem proving that we can form exactly 180 unique six-digit integers.
theorem sixDigitIntegersCount : numberOfUniquePermutations = 180 :=
  sorry

end sixDigitIntegersCount_l147_147122


namespace inequality_and_equality_conditions_l147_147886

theorem inequality_and_equality_conditions
    {a b c d : ℝ}
    (ha : 0 < a)
    (hb : 0 < b)
    (hc : 0 < c)
    (hd : 0 < d) :
  (a ^ (1/3) * b ^ (1/3) + c ^ (1/3) * d ^ (1/3) ≤ (a + b + c) ^ (1/3) * (a + c + d) ^ (1/3)) ↔ 
  (b = (a / c) * (a + c) ∧ d = (c / a) * (a + c)) :=
  sorry

end inequality_and_equality_conditions_l147_147886


namespace total_students_mrs_mcgillicuddy_l147_147969

-- Define the conditions as variables
def students_registered_morning : ℕ := 25
def students_absent_morning : ℕ := 3
def students_registered_afternoon : ℕ := 24
def students_absent_afternoon : ℕ := 4

-- Prove the total number of students present over the two sessions
theorem total_students_mrs_mcgillicuddy : 
  students_registered_morning - students_absent_morning + students_registered_afternoon - students_absent_afternoon = 42 :=
by
  sorry

end total_students_mrs_mcgillicuddy_l147_147969


namespace number_division_l147_147957

theorem number_division (n : ℕ) (h1 : 555 + 445 = 1000) (h2 : 555 - 445 = 110) 
  (h3 : n % 1000 = 80) (h4 : n / 1000 = 220) : n = 220080 :=
by {
  -- proof steps would go here
  sorry
}

end number_division_l147_147957


namespace crayons_left_l147_147686

-- Define the initial number of crayons
def initial_crayons : ℕ := 440

-- Define the crayons given away
def crayons_given : ℕ := 111

-- Define the crayons lost
def crayons_lost : ℕ := 106

-- Prove the final number of crayons left
theorem crayons_left : (initial_crayons - crayons_given - crayons_lost) = 223 :=
by
  sorry

end crayons_left_l147_147686


namespace remainder_3_pow_19_mod_10_l147_147803

theorem remainder_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 :=
by
  sorry

end remainder_3_pow_19_mod_10_l147_147803


namespace geometric_sequence_logarithm_identity_l147_147072

variable {a : ℕ+ → ℝ}

-- Assumptions
def common_ratio (a : ℕ+ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_logarithm_identity
  (r : ℝ)
  (hr : r = -Real.sqrt 2)
  (h : common_ratio a r) :
  Real.log (a 2017)^2 - Real.log (a 2016)^2 = Real.log 2 :=
by
  sorry

end geometric_sequence_logarithm_identity_l147_147072


namespace necessary_but_not_sufficient_not_sufficient_condition_l147_147569

theorem necessary_but_not_sufficient (a b : ℝ) : (a > 2 ∧ b > 2) → (a + b > 4) :=
sorry

theorem not_sufficient_condition (a b : ℝ) : (a + b > 4) → ¬(a > 2 ∧ b > 2) :=
sorry

end necessary_but_not_sufficient_not_sufficient_condition_l147_147569


namespace gcd_256_162_450_l147_147739

theorem gcd_256_162_450 : Nat.gcd (Nat.gcd 256 162) 450 = 2 := sorry

end gcd_256_162_450_l147_147739


namespace abs_diff_eq_two_l147_147680

def equation (x y : ℝ) : Prop := y^2 + x^4 = 2 * x^2 * y + 1

theorem abs_diff_eq_two (a b e : ℝ) (ha : equation e a) (hb : equation e b) (hab : a ≠ b) :
  |a - b| = 2 :=
sorry

end abs_diff_eq_two_l147_147680


namespace find_n_from_binomial_term_l147_147609

noncomputable def binomial_coefficient (n r : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem find_n_from_binomial_term :
  (∃ n : ℕ, 3^2 * binomial_coefficient n 2 = 54) ↔ n = 4 :=
by
  sorry

end find_n_from_binomial_term_l147_147609


namespace statement_A_l147_147479

theorem statement_A (x : ℝ) (h : x > 1) : x^2 > x := 
by
  sorry

end statement_A_l147_147479


namespace weight_of_each_bag_is_correct_l147_147363

noncomputable def weightOfEachBag
    (days1 : ℕ := 60)
    (consumption1 : ℕ := 2)
    (days2 : ℕ := 305)
    (consumption2 : ℕ := 4)
    (ouncesPerPound : ℕ := 16)
    (numberOfBags : ℕ := 17) : ℝ :=
        let totalOunces := (days1 * consumption1) + (days2 * consumption2)
        let totalPounds := totalOunces / ouncesPerPound
        totalPounds / numberOfBags

theorem weight_of_each_bag_is_correct :
  weightOfEachBag = 4.93 :=
by
  sorry

end weight_of_each_bag_is_correct_l147_147363


namespace olivia_time_spent_l147_147473

theorem olivia_time_spent :
  ∀ (x : ℕ), 7 * x + 3 = 31 → x = 4 :=
by
  intro x
  intro h
  sorry

end olivia_time_spent_l147_147473


namespace fruit_seller_sp_l147_147307

theorem fruit_seller_sp (CP SP : ℝ)
    (h1 : SP = 0.75 * CP)
    (h2 : 19.93 = 1.15 * CP) :
    SP = 13.00 :=
by
  sorry

end fruit_seller_sp_l147_147307


namespace incorrect_statements_l147_147488

-- Defining the first condition
def condition1 : Prop :=
  let a_sq := 169
  let b_sq := 144
  let c_sq := a_sq - b_sq
  let c_ := Real.sqrt c_sq
  let focal_points := [(0, c_), (0, -c_)]
  ¬((c_, 0) ∈ focal_points) ∧ ¬((-c_, 0) ∈ focal_points)

-- Defining the second condition
def condition2 : Prop :=
  let m := 1  -- Example choice since m is unspecified
  let a_sq := m^2 + 1
  let b_sq := m^2
  let c_sq := a_sq - b_sq
  let c_ := Real.sqrt c_sq
  let focal_points := [(0, c_), (0, -c_)]
  (0, 1) ∈ focal_points ∧ (0, -1) ∈ focal_points

-- Defining the third condition
def condition3 : Prop :=
  let a1_sq := 16
  let b1_sq := 7
  let c1_sq := a1_sq - b1_sq
  let c1_ := Real.sqrt c1_sq
  let focal_points1 := [(c1_, 0), (-c1_, 0)]
  
  let m := 10  -- Example choice since m > 0 is unspecified
  let a2_sq := m - 5
  let b2_sq := m + 4
  let c2_sq := a2_sq - b2_sq
  let focal_points2 := [(0, Real.sqrt c2_sq), (0, -Real.sqrt c2_sq)]
  
  ¬ (focal_points1 = focal_points2)

-- Defining the fourth condition
def condition4 : Prop :=
  let B := (-3, 0)
  let C := (3, 0)
  let BC := (C.1 - B.1, C.2 - B.2)
  let BC_dist := Real.sqrt (BC.1^2 + BC.2^2)
  let A_locus_eq := ∀ (x y : ℝ), x^2 / 36 + y^2 / 27 = 1
  2 * BC_dist = 12

-- Proof verification
theorem incorrect_statements : Prop :=
  condition1 ∧ condition3

end incorrect_statements_l147_147488


namespace find_d_l147_147204

namespace NineDigitNumber

variables {A B C D E F G : ℕ}

theorem find_d 
  (h1 : 6 + A + B = 13) 
  (h2 : A + B + C = 13)
  (h3 : B + C + D = 13)
  (h4 : C + D + E = 13)
  (h5 : D + E + F = 13)
  (h6 : E + F + G = 13)
  (h7 : F + G + 3 = 13) :
  D = 4 :=
sorry

end NineDigitNumber

end find_d_l147_147204


namespace slope_of_line_of_intersections_l147_147897

theorem slope_of_line_of_intersections : 
  ∀ s : ℝ, let x := (41 * s + 13) / 11
           let y := -((2 * s + 6) / 11)
           ∃ m : ℝ, m = -22 / 451 :=
sorry

end slope_of_line_of_intersections_l147_147897


namespace odd_expressions_l147_147288

theorem odd_expressions (m n p : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) (hp : p % 2 = 0) : 
  ((2 * m * n + 5) ^ 2 % 2 = 1) ∧ (5 * m * n + p % 2 = 1) := 
by
  sorry

end odd_expressions_l147_147288


namespace carina_total_coffee_l147_147947

def number_of_ten_ounce_packages : ℕ := 4
def number_of_five_ounce_packages : ℕ := number_of_ten_ounce_packages + 2
def ounces_in_each_ten_ounce_package : ℕ := 10
def ounces_in_each_five_ounce_package : ℕ := 5

def total_coffee_ounces : ℕ := 
  (number_of_ten_ounce_packages * ounces_in_each_ten_ounce_package) +
  (number_of_five_ounce_packages * ounces_in_each_five_ounce_package)

theorem carina_total_coffee : total_coffee_ounces = 70 := by
  -- proof to be provided
  sorry

end carina_total_coffee_l147_147947


namespace coords_of_a_in_m_n_l147_147785

variable {R : Type} [Field R]

def coords_in_basis (a : R × R) (p q : R × R) (c1 c2 : R) : Prop :=
  a = c1 • p + c2 • q

theorem coords_of_a_in_m_n
  (a p q m n : R × R)
  (hp : p = (1, -1)) (hq : q = (2, 1)) (hm : m = (-1, 1)) (hn : n = (1, 2))
  (coords_pq : coords_in_basis a p q (-2) 2) :
  coords_in_basis a m n 0 2 :=
by
  sorry

end coords_of_a_in_m_n_l147_147785


namespace cube_problem_l147_147291

-- Define the conditions
def cube_volume (s : ℝ) : ℝ := s^3
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

theorem cube_problem (x : ℝ) (s : ℝ) :
  cube_volume s = 8 * x ∧ cube_surface_area s = 4 * x → x = 216 :=
by
  intro h
  sorry

end cube_problem_l147_147291


namespace jasper_sold_31_drinks_l147_147804

def chips := 27
def hot_dogs := chips - 8
def drinks := hot_dogs + 12

theorem jasper_sold_31_drinks : drinks = 31 := by
  sorry

end jasper_sold_31_drinks_l147_147804


namespace relationship_between_a_b_c_l147_147476

noncomputable def a : ℝ := 2^(4/3)
noncomputable def b : ℝ := 4^(2/5)
noncomputable def c : ℝ := 25^(1/3)

theorem relationship_between_a_b_c : c > a ∧ a > b := 
by
  have ha : a = 2^(4/3) := rfl
  have hb : b = 4^(2/5) := rfl
  have hc : c = 25^(1/3) := rfl

  sorry

end relationship_between_a_b_c_l147_147476


namespace john_cakes_bought_l147_147579

-- Conditions
def cake_price : ℕ := 12
def john_paid : ℕ := 18

-- Definition of the total cost
def total_cost : ℕ := 2 * john_paid

-- Calculate number of cakes
def num_cakes (total_cost cake_price : ℕ) : ℕ := total_cost / cake_price

-- Theorem to prove that the number of cakes John Smith bought is 3
theorem john_cakes_bought : num_cakes total_cost cake_price = 3 := by
  sorry

end john_cakes_bought_l147_147579


namespace upper_limit_of_x_l147_147232

theorem upper_limit_of_x 
  {x : ℤ} 
  (h1 : 0 < x) 
  (h2 : x < 15) 
  (h3 : -1 < x) 
  (h4 : x < 5) 
  (h5 : 0 < x) 
  (h6 : x < 3) 
  (h7 : x + 2 < 4) 
  (h8 : x = 1) : 
  0 < x ∧ x < 2 := 
by 
  sorry

end upper_limit_of_x_l147_147232


namespace sports_minutes_in_newscast_l147_147665

-- Definitions based on the conditions
def total_newscast_minutes : ℕ := 30
def national_news_minutes : ℕ := 12
def international_news_minutes : ℕ := 5
def weather_forecasts_minutes : ℕ := 2
def advertising_minutes : ℕ := 6

-- The problem statement
theorem sports_minutes_in_newscast (t : ℕ) (n : ℕ) (i : ℕ) (w : ℕ) (a : ℕ) :
  t = 30 → n = 12 → i = 5 → w = 2 → a = 6 → t - n - i - w - a = 5 := 
by sorry

end sports_minutes_in_newscast_l147_147665


namespace charity_distribution_l147_147141

theorem charity_distribution
    (amount_raised : ℝ)
    (donation_percentage : ℝ)
    (num_organizations : ℕ)
    (h_amount_raised : amount_raised = 2500)
    (h_donation_percentage : donation_percentage = 0.80)
    (h_num_organizations : num_organizations = 8) :
    (amount_raised * donation_percentage) / num_organizations = 250 := by
  sorry

end charity_distribution_l147_147141


namespace function_relationship_value_of_x_l147_147193

variable {x y : ℝ}

-- Given conditions:
-- Condition 1: y is inversely proportional to x
def inversely_proportional (p : ℝ) (q : ℝ) (k : ℝ) : Prop := p = k / q

-- Condition 2: y(2) = -3
def specific_value (x_val y_val : ℝ) : Prop := y_val = -3 ∧ x_val = 2

-- Questions rephrased as Lean theorems:

-- The function relationship between y and x is y = -6 / x
theorem function_relationship (k : ℝ) (hx : x ≠ 0) 
  (h_inv_prop: inversely_proportional y x k) (h_spec : specific_value 2 (-3)) : k = -6 :=
by
  sorry

-- When y = 2, x = -3
theorem value_of_x (hx : x ≠ 0) (hy : y = 2)
  (h_inv_prop : inversely_proportional y x (-6)) : x = -3 :=
by
  sorry

end function_relationship_value_of_x_l147_147193


namespace quadratic_inequality_solution_l147_147467

theorem quadratic_inequality_solution
  (a b c : ℝ)
  (h1: ∀ x : ℝ, (-1/3 < x ∧ x < 2) → (ax^2 + bx + c) > 0)
  (h2: a < 0):
  ∀ x : ℝ, ((-3 < x ∧ x < 1/2) ↔ (cx^2 + bx + a) < 0) :=
by
  sorry

end quadratic_inequality_solution_l147_147467


namespace audio_per_cd_l147_147760

theorem audio_per_cd (total_audio : ℕ) (max_per_cd : ℕ) (num_cds : ℕ) 
  (h1 : total_audio = 360) 
  (h2 : max_per_cd = 60) 
  (h3 : num_cds = total_audio / max_per_cd): 
  (total_audio / num_cds = max_per_cd) :=
by
  sorry

end audio_per_cd_l147_147760


namespace four_digit_number_2010_l147_147212

theorem four_digit_number_2010 (a b c d : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 1000 ≤ 1000 * a + 100 * b + 10 * c + d ∧
        1000 * a + 100 * b + 10 * c + d < 10000)
  (h_eq : a * (a + b + c + d) * (a^2 + b^2 + c^2 + d^2) * (a^6 + 2 * b^6 + 3 * c^6 + 4 * d^6)
          = 1000 * a + 100 * b + 10 * c + d)
  : 1000 * a + 100 * b + 10 * c + d = 2010 :=
sorry

end four_digit_number_2010_l147_147212


namespace canary_possible_distances_l147_147434

noncomputable def distance_from_bus_stop (bus_stop swallow sparrow canary : ℝ) : Prop :=
  swallow = 380 ∧
  sparrow = 450 ∧
  (sparrow - swallow) = (canary - sparrow) ∨
  (swallow - sparrow) = (sparrow - canary)

theorem canary_possible_distances (swallow sparrow canary : ℝ) :
  distance_from_bus_stop 0 swallow sparrow canary →
  canary = 520 ∨ canary = 1280 :=
by
  sorry

end canary_possible_distances_l147_147434


namespace range_of_x_l147_147596

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / x) + 2 * Real.sin x

theorem range_of_x (x : ℝ) (h₀ : x > 0) (h₁ : f (1 - x) > f x) : x < (1 / 2) :=
by
  sorry

end range_of_x_l147_147596


namespace parabola_directrix_l147_147099

theorem parabola_directrix (a : ℝ) (h1 : ∀ x : ℝ, - (1 / (4 * a)) = 2):
  a = -(1 / 8) :=
sorry

end parabola_directrix_l147_147099


namespace expression_meaningful_l147_147188

theorem expression_meaningful (x : ℝ) : 
  (x - 1 ≠ 0 ∧ true) ↔ x ≠ 1 := 
sorry

end expression_meaningful_l147_147188


namespace solution_set_of_inequality_l147_147032

theorem solution_set_of_inequality :
  ∀ x : ℝ, 3 * x^2 - 2 * x + 1 > 7 ↔ (x < -2/3 ∨ x > 3) :=
by
  sorry

end solution_set_of_inequality_l147_147032


namespace Wilson_sledding_l147_147048

variable (T S : ℕ)

theorem Wilson_sledding (h1 : S = T / 2) (h2 : (2 * T) + (3 * S) = 14) : T = 4 := by
  sorry

end Wilson_sledding_l147_147048


namespace train_length_is_400_l147_147906

-- Define the conditions
def time := 40 -- seconds
def speed_kmh := 36 -- km/h

-- Conversion factor from km/h to m/s
def kmh_to_ms (v : ℕ) := (v * 5) / 18

def speed_ms := kmh_to_ms speed_kmh -- convert speed to m/s

-- Definition of length of the train using the given conditions
def train_length := speed_ms * time

-- Theorem to prove the length of the train is 400 meters
theorem train_length_is_400 : train_length = 400 := by
  sorry

end train_length_is_400_l147_147906


namespace problem_proof_l147_147540

-- Define the given conditions and the target statement
theorem problem_proof (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 10.5) : a^2 + b^2 = 25 := 
by sorry

end problem_proof_l147_147540


namespace intersection_of_sets_l147_147984

def setA (x : ℝ) : Prop := x^2 - 4 * x - 5 > 0

def setB (x : ℝ) : Prop := 4 - x^2 > 0

theorem intersection_of_sets :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -2 < x ∧ x < -1} :=
by
  sorry

end intersection_of_sets_l147_147984


namespace p_or_q_iff_not_p_and_not_q_false_l147_147158

variables (p q : Prop)

theorem p_or_q_iff_not_p_and_not_q_false : (p ∨ q) ↔ ¬(¬p ∧ ¬q) :=
by sorry

end p_or_q_iff_not_p_and_not_q_false_l147_147158


namespace simplify_fraction_subtraction_l147_147103

theorem simplify_fraction_subtraction : (1 / 210) - (17 / 35) = -101 / 210 := by
  sorry

end simplify_fraction_subtraction_l147_147103


namespace sufficient_condition_l147_147944

theorem sufficient_condition (a b : ℝ) : ab ≠ 0 → a ≠ 0 :=
sorry

end sufficient_condition_l147_147944


namespace pedestrian_travel_time_l147_147378

noncomputable def travel_time (d : ℝ) (x y : ℝ) : ℝ :=
  d / x

theorem pedestrian_travel_time
  (d : ℝ)
  (x y : ℝ)
  (h1 : d = 1)
  (h2 : 3 * x = 1 - x - y)
  (h3 : (1 / 2) * (x + y) = 1 - x - y)
  : travel_time d x y = 9 := 
sorry

end pedestrian_travel_time_l147_147378


namespace minimum_path_proof_l147_147608

noncomputable def minimum_path (r : ℝ) (h : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let R := Real.sqrt (r^2 + h^2)
  let theta := 2 * Real.pi * (R / (2 * Real.pi * r))
  let A := (d1, 0)
  let B := (-d2 * Real.cos (theta / 2), -d2 * Real.sin (theta / 2))
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem minimum_path_proof :
  minimum_path 800 (300 * Real.sqrt 3) 150 (450 * Real.sqrt 2) = 562.158 := 
by 
  sorry

end minimum_path_proof_l147_147608


namespace Tyler_age_l147_147274

variable (T B S : ℕ)

theorem Tyler_age :
  (T = B - 3) ∧
  (S = B + 2) ∧
  (S = 2 * T) ∧
  (T + B + S = 30) →
  T = 5 := by
  sorry

end Tyler_age_l147_147274


namespace f_inequality_l147_147815

def f (x : ℝ) : ℝ := sorry

axiom f_defined : ∀ x : ℝ, 0 < x → ∃ y : ℝ, f x = y

axiom f_additive : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y

axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

axiom f_two : f 2 = 1

theorem f_inequality (x : ℝ) : 3 < x → x ≤ 4 → f x + f (x - 3) ≤ 2 :=
sorry

end f_inequality_l147_147815


namespace find_angle_C_find_a_and_b_l147_147661

-- Conditions from the problem
variables {A B C : ℝ} {a b c : ℝ}
variables {m n : ℝ × ℝ}
variables (h1 : m = (Real.sin A, Real.sin B - Real.sin C))
variables (h2 : n = (a - Real.sqrt 3 * b, b + c))
variables (h3 : m.1 * n.1 + m.2 * n.2 = 0)
variables (h4 : ∀ θ ∈ Set.Ioo 0 Real.pi, θ ≠ C → Real.cos θ = (a^2 + b^2 - c^2) / (2 * a * b))

-- Hypotheses for part (2)
variables (circumradius : ℝ) (area : ℝ)
variables (h5 : circumradius = 2)
variables (h6 : area = Real.sqrt 3)
variables (h7 : a > b)

-- Theorem statement for part (1)
theorem find_angle_C (h1 : m = (Real.sin A, Real.sin B - Real.sin C))
  (h2 : n = (a - Real.sqrt 3 * b, b + c))
  (h3 : m.1 * n.1 + m.2 * n.2 = 0)
  (h4 : ∀ C ∈ Set.Ioo 0 Real.pi, Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) : 
  C = Real.pi / 6 := sorry

-- Theorem statement for part (2)
theorem find_a_and_b (circumradius : ℝ) (area : ℝ) (a b : ℝ)
  (h5 : circumradius = 2) (h6 : area = Real.sqrt 3) (h7 : a > b)
  (h8 : ∀ C ∈ Set.Ioo 0 Real.pi, Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b))
  (h9 : Real.sin C ≠ 0): 
  a = 2 * Real.sqrt 3 ∧ b = 2 := sorry

end find_angle_C_find_a_and_b_l147_147661


namespace minimum_people_correct_answer_l147_147277

theorem minimum_people_correct_answer (people questions : ℕ) (common_correct : ℕ) (h_people : people = 21) (h_questions : questions = 15) (h_common_correct : ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ people → 1 ≤ b ∧ b ≤ people → a ≠ b → common_correct ≥ 1) :
  ∃ (min_correct : ℕ), min_correct = 7 := 
sorry

end minimum_people_correct_answer_l147_147277


namespace solution_set_a_range_m_l147_147918

theorem solution_set_a (a : ℝ) :
  (∀ x : ℝ, |x - a| ≤ 3 ↔ -6 ≤ x ∧ x ≤ 0) ↔ a = -3 :=
by
  sorry

theorem range_m (m : ℝ) :
  (∀ x : ℝ, |x + 3| + |x + 8| ≥ 2 * m) ↔ m ≤ 5 / 2 :=
by
  sorry

end solution_set_a_range_m_l147_147918


namespace odd_function_strictly_decreasing_inequality_solutions_l147_147305

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom positive_for_neg_x (x : ℝ) : x < 0 → f x > 0

theorem odd_function : ∀ (x : ℝ), f (-x) = -f x := sorry

theorem strictly_decreasing : ∀ (x₁ x₂ : ℝ), x₁ > x₂ → f x₁ < f x₂ := sorry

theorem inequality_solutions (a x : ℝ) :
  (a = 0 ∧ false) ∨ 
  (a > 3 ∧ 3 < x ∧ x < a) ∨ 
  (a < 3 ∧ a < x ∧ x < 3) := sorry

end odd_function_strictly_decreasing_inequality_solutions_l147_147305


namespace hospital_bed_occupancy_l147_147006

theorem hospital_bed_occupancy 
  (x : ℕ)
  (beds_A := x)
  (beds_B := 2 * x)
  (beds_C := 3 * x)
  (occupied_A := (1 / 3) * x)
  (occupied_B := (1 / 2) * (2 * x))
  (occupied_C := (1 / 4) * (3 * x))
  (max_capacity_B := (3 / 4) * (2 * x))
  (max_capacity_C := (5 / 6) * (3 * x)) :
  (4 / 3 * x) / (2 * x) = 2 / 3 ∧ (3 / 4 * x) / (3 * x) = 1 / 4 := 
  sorry

end hospital_bed_occupancy_l147_147006


namespace min_value_one_over_a_plus_two_over_b_l147_147483

theorem min_value_one_over_a_plus_two_over_b :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 2) →
  ∃ (min_val : ℝ), min_val = (1 / a + 2 / b) ∧ min_val = 9 / 2 :=
by
  sorry

end min_value_one_over_a_plus_two_over_b_l147_147483
