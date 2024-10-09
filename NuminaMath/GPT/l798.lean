import Mathlib

namespace time_to_fill_remaining_l798_79891

-- Define the rates at which pipes P and Q fill the cistern
def rate_P := 1 / 12
def rate_Q := 1 / 15

-- Define the time both pipes are open together
def time_both_open := 4

-- Calculate the combined rate when both pipes are open
def combined_rate := rate_P + rate_Q

-- Calculate the amount of the cistern filled in the time both pipes are open
def filled_amount_both_open := time_both_open * combined_rate

-- Calculate the remaining amount to fill after Pipe P is turned off
def remaining_amount := 1 - filled_amount_both_open

-- Calculate the time it will take for Pipe Q alone to fill the remaining amount
def time_Q_to_fill_remaining := remaining_amount / rate_Q

-- The final theorem
theorem time_to_fill_remaining : time_Q_to_fill_remaining = 6 := by
  sorry

end time_to_fill_remaining_l798_79891


namespace no_two_obtuse_angles_in_triangle_l798_79816

theorem no_two_obtuse_angles_in_triangle (A B C : ℝ) 
  (h1 : 0 < A) (h2 : A < 180) 
  (h3 : 0 < B) (h4 : B < 180) 
  (h5 : 0 < C) (h6 : C < 180)
  (h7 : A + B + C = 180) 
  (h8 : A > 90) (h9 : B > 90) : false :=
by
  sorry

end no_two_obtuse_angles_in_triangle_l798_79816


namespace race_min_distance_l798_79814

noncomputable def min_distance : ℝ :=
  let A : ℝ × ℝ := (0, 300)
  let B : ℝ × ℝ := (1200, 500)
  let wall_length : ℝ := 1200
  let B' : ℝ × ℝ := (1200, -500)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A B'

theorem race_min_distance :
  min_distance = 1442 := sorry

end race_min_distance_l798_79814


namespace max_b_no_lattice_points_line_l798_79867

theorem max_b_no_lattice_points_line (b : ℝ) (h : ∀ (m : ℝ), 0 < m ∧ m < b → ∀ (x : ℤ), 0 < (x : ℝ) ∧ (x : ℝ) ≤ 150 → ¬∃ (y : ℤ), y = m * x + 5) :
  b ≤ 1 / 151 :=
by sorry

end max_b_no_lattice_points_line_l798_79867


namespace measure_angle_YPZ_is_142_l798_79893

variables (X Y Z : Type) [Inhabited X] [Inhabited Y] [Inhabited Z]
variables (XM YN ZO : Type) [Inhabited XM] [Inhabited YN] [Inhabited ZO]

noncomputable def angle_XYZ : ℝ := 65
noncomputable def angle_XZY : ℝ := 38
noncomputable def angle_YXZ : ℝ := 180 - angle_XYZ - angle_XZY
noncomputable def angle_YNZ : ℝ := 90 - angle_YXZ
noncomputable def angle_ZMY : ℝ := 90 - angle_XYZ
noncomputable def angle_YPZ : ℝ := 180 - angle_YNZ - angle_ZMY

theorem measure_angle_YPZ_is_142 :
  angle_YPZ = 142 := sorry

end measure_angle_YPZ_is_142_l798_79893


namespace common_difference_of_common_terms_l798_79819

def sequence_a (n : ℕ) : ℕ := 4 * n - 3
def sequence_b (k : ℕ) : ℕ := 3 * k - 1

theorem common_difference_of_common_terms :
  ∃ (d : ℕ), (∀ (m : ℕ), 12 * m + 5 ∈ { x | ∃ (n k : ℕ), sequence_a n = x ∧ sequence_b k = x }) ∧ d = 12 := 
sorry

end common_difference_of_common_terms_l798_79819


namespace competition_score_difference_l798_79869

theorem competition_score_difference :
  let perc_60 := 0.20
  let perc_75 := 0.25
  let perc_85 := 0.15
  let perc_90 := 0.30
  let perc_95 := 0.10
  let mean := (perc_60 * 60) + (perc_75 * 75) + (perc_85 * 85) + (perc_90 * 90) + (perc_95 * 95)
  let median := 85
  (median - mean = 5) := by
sorry

end competition_score_difference_l798_79869


namespace tom_seashells_l798_79896

theorem tom_seashells (fred_seashells : ℕ) (total_seashells : ℕ) (tom_seashells : ℕ)
  (h1 : fred_seashells = 43)
  (h2 : total_seashells = 58)
  (h3 : total_seashells = fred_seashells + tom_seashells) : tom_seashells = 15 :=
by
  sorry

end tom_seashells_l798_79896


namespace triangle_expression_value_l798_79838

theorem triangle_expression_value :
  ∀ (A B C : ℝ) (a b c : ℝ),
  A = 60 ∧ b = 1 ∧ (1 / 2) * b * c * (Real.sin A) = Real.sqrt 3 →
  (a + 2 * b - 3 * c) / (Real.sin A + 2 * Real.sin B - 3 * Real.sin C) = 2 * (Real.sqrt 39) / 3 :=
by
  intro A B C a b c
  rintro ⟨hA, hb, h_area⟩
  sorry

end triangle_expression_value_l798_79838


namespace roots_equation_l798_79866

theorem roots_equation (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : a * 4^3 + b * 4^2 + c * 4 + d = 0) (h₃ : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -13 :=
by
  sorry

end roots_equation_l798_79866


namespace area_of_isosceles_triangle_PQR_l798_79841

noncomputable def area_of_triangle (P Q R : ℝ) (PQ PR QR PS QS SR : ℝ) : Prop :=
PQ = 17 ∧ PR = 17 ∧ QR = 16 ∧ PS = 15 ∧ QS = 8 ∧ SR = 8 →
(1 / 2) * QR * PS = 120

theorem area_of_isosceles_triangle_PQR :
  ∀ (P Q R : ℝ), 
  ∀ (PQ PR QR PS QS SR : ℝ), 
  PQ = 17 → PR = 17 → QR = 16 → PS = 15 → QS = 8 → SR = 8 →
  area_of_triangle P Q R PQ PR QR PS QS SR := 
by
  intros P Q R PQ PR QR PS QS SR hPQ hPR hQR hPS hQS hSR
  unfold area_of_triangle
  simp [hPQ, hPR, hQR, hPS, hQS, hSR]
  sorry

end area_of_isosceles_triangle_PQR_l798_79841


namespace find_z_l798_79826

def M (z : ℂ) : Set ℂ := {1, 2, z * Complex.I}
def N : Set ℂ := {3, 4}

theorem find_z (z : ℂ) (h : M z ∩ N = {4}) : z = -4 * Complex.I := by
  sorry

end find_z_l798_79826


namespace interval_comparison_l798_79881

theorem interval_comparison (x : ℝ) :
  ((x - 1) * (x + 3) < 0) → ¬((x + 1) * (x - 3) < 0) ∧ ¬((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0) :=
by
  sorry

end interval_comparison_l798_79881


namespace train_speed_l798_79807

theorem train_speed (time_seconds : ℕ) (length_meters : ℕ) (speed_kmph : ℕ)
  (h1 : time_seconds = 9) (h2 : length_meters = 135) : speed_kmph = 54 :=
sorry

end train_speed_l798_79807


namespace pancake_cut_l798_79820

theorem pancake_cut (n : ℕ) (h : 3 ≤ n) :
  ∃ (cut_piece : ℝ), cut_piece > 0 :=
sorry

end pancake_cut_l798_79820


namespace geometric_seq_arith_mean_l798_79879

theorem geometric_seq_arith_mean 
  (b : ℕ → ℝ) 
  (r : ℝ) 
  (b_geom : ∀ n, b (n + 1) = r * b n)
  (h_arith_mean : b 9 = (3 + 5) / 2) :
  b 1 * b 17 = 16 :=
by
  sorry

end geometric_seq_arith_mean_l798_79879


namespace part1_part2_l798_79834

-- Definitions based on the conditions
def a_i (i : ℕ) : ℕ := sorry -- Define ai's values based on the given conditions
def f (n : ℕ) : ℕ := sorry  -- Define f(n) as the number of n-digit wave numbers satisfying the given conditions

-- Prove the first part: f(10) = 3704
theorem part1 : f 10 = 3704 := sorry

-- Prove the second part: f(2008) % 13 = 10
theorem part2 : (f 2008) % 13 = 10 := sorry

end part1_part2_l798_79834


namespace find_multiple_of_ron_l798_79884

variable (R_d R_g R_n m : ℕ)

def rodney_can_lift_146 : Prop := R_d = 146
def combined_weight_239 : Prop := R_d + R_g + R_n = 239
def rodney_twice_as_roger : Prop := R_d = 2 * R_g
def roger_seven_less_than_multiple_of_ron : Prop := R_g = m * R_n - 7

theorem find_multiple_of_ron (h1 : rodney_can_lift_146 R_d) 
                             (h2 : combined_weight_239 R_d R_g R_n) 
                             (h3 : rodney_twice_as_roger R_d R_g) 
                             (h4 : roger_seven_less_than_multiple_of_ron R_g R_n m) 
                             : m = 4 :=
by 
    sorry

end find_multiple_of_ron_l798_79884


namespace fraction_numerator_exceeds_denominator_l798_79818

theorem fraction_numerator_exceeds_denominator (x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 3) :
  4 * x + 5 > 10 - 3 * x ↔ (5 / 7) < x ∧ x ≤ 3 :=
by 
  sorry

end fraction_numerator_exceeds_denominator_l798_79818


namespace sum_of_real_solutions_l798_79858

theorem sum_of_real_solutions (x : ℝ) (h : (x^2 + 2*x + 3)^( (x^2 + 2*x + 3)^( (x^2 + 2*x + 3) )) = 2012) : 
  ∃ (x1 x2 : ℝ), (x1 + x2 = -2) ∧ (x1^2 + 2*x1 + 3 = x2^2 + 2*x2 + 3 ∧ x2^2 + 2*x2 + 3 = x^2 + 2*x + 3) := 
by
  sorry

end sum_of_real_solutions_l798_79858


namespace find_P_plus_Q_l798_79831

theorem find_P_plus_Q (P Q : ℝ) (h : ∃ b c : ℝ, (x^2 + 3 * x + 4) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) : 
P + Q = 15 :=
by
  sorry

end find_P_plus_Q_l798_79831


namespace chinese_team_wins_gold_l798_79813

noncomputable def prob_player_a_wins : ℚ := 3 / 7
noncomputable def prob_player_b_wins : ℚ := 1 / 4

theorem chinese_team_wins_gold : prob_player_a_wins + prob_player_b_wins = 19 / 28 := by
  sorry

end chinese_team_wins_gold_l798_79813


namespace lcm_of_two_numbers_l798_79886

variable (a b hcf lcm : ℕ)

theorem lcm_of_two_numbers (ha : a = 330) (hb : b = 210) (hhcf : Nat.gcd a b = 30) :
  Nat.lcm a b = 2310 := by
  sorry

end lcm_of_two_numbers_l798_79886


namespace paul_prays_more_than_bruce_l798_79846

-- Conditions as definitions in Lean 4
def prayers_per_day_paul := 20
def prayers_per_sunday_paul := 2 * prayers_per_day_paul
def prayers_per_day_bruce := prayers_per_day_paul / 2
def prayers_per_sunday_bruce := 2 * prayers_per_sunday_paul

def weekly_prayers_paul := 6 * prayers_per_day_paul + prayers_per_sunday_paul
def weekly_prayers_bruce := 6 * prayers_per_day_bruce + prayers_per_sunday_bruce

-- Statement of the proof problem
theorem paul_prays_more_than_bruce :
  (weekly_prayers_paul - weekly_prayers_bruce) = 20 := by
  sorry

end paul_prays_more_than_bruce_l798_79846


namespace axis_of_symmetry_l798_79856

-- Define the condition for the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  x = -4 * y^2

-- Define the statement that needs to be proven
theorem axis_of_symmetry (x : ℝ) (y : ℝ) (h : parabola_equation x y) : x = 1 / 16 :=
  sorry

end axis_of_symmetry_l798_79856


namespace sequence_an_l798_79850

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom S_formula (n : ℕ) (h₁ : n > 0) : S n = 2 * a n - 2

-- Proof goal
theorem sequence_an (n : ℕ) (h₁ : n > 0) : a n = 2 ^ n := by
  sorry

end sequence_an_l798_79850


namespace tan_sin_div_l798_79872

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_div_l798_79872


namespace volume_of_triangular_pyramid_l798_79871

variable (a b : ℝ)

noncomputable def volume_of_pyramid (a b : ℝ) : ℝ :=
  (a * b / 12) * Real.sqrt (3 * b ^ 2 - a ^ 2)

theorem volume_of_triangular_pyramid (a b : ℝ) :
  volume_of_pyramid a b = (a * b / 12) * Real.sqrt (3 * b ^ 2 - a ^ 2) :=
by
  sorry

end volume_of_triangular_pyramid_l798_79871


namespace arithmetic_expression_equality_l798_79802

theorem arithmetic_expression_equality :
  ( ( (4 + 6 + 5) * 2 ) / 4 - ( (3 * 2) / 4 ) ) = 6 :=
by sorry

end arithmetic_expression_equality_l798_79802


namespace find_t_squared_l798_79840
noncomputable section

-- Definitions of the given conditions
def hyperbola_opens_vertically (x y : ℝ) : Prop :=
  (y^2 / 4 - 5 * x^2 / 16 = 1)

-- Statement of the problem
theorem find_t_squared (t : ℝ) 
  (h1 : hyperbola_opens_vertically 4 (-3))
  (h2 : hyperbola_opens_vertically 0 (-2))
  (h3 : hyperbola_opens_vertically 2 t) : 
  t^2 = 8 := 
sorry -- Proof is omitted, it's just the statement

end find_t_squared_l798_79840


namespace buying_pets_l798_79845

theorem buying_pets {puppies kittens hamsters birds : ℕ} :
(∃ pets : ℕ, pets = 12 * 8 * 10 * 5 * 4 * 3 * 2) ∧ 
puppies = 12 ∧ kittens = 8 ∧ hamsters = 10 ∧ birds = 5 → 
12 * 8 * 10 * 5 * 4 * 3 * 2 = 115200 :=
by
  intros h
  sorry

end buying_pets_l798_79845


namespace no_solution_exists_l798_79888

theorem no_solution_exists :
  ¬ ∃ (x1 x2 x3 x4 : ℝ), 
    (x1 + x2 = 1) ∧
    (x2 + x3 - x4 = 1) ∧
    (0 ≤ x1) ∧
    (0 ≤ x2) ∧
    (0 ≤ x3) ∧
    (0 ≤ x4) ∧
    ∀ (F : ℝ), F = x1 - x2 + 2 * x3 - x4 → 
    ∀ (b : ℝ), F ≤ b :=
by sorry

end no_solution_exists_l798_79888


namespace largest_int_less_100_remainder_5_l798_79810

theorem largest_int_less_100_remainder_5 (a : ℕ) (h1 : a < 100) (h2 : a % 9 = 5) :
  a = 95 :=
sorry

end largest_int_less_100_remainder_5_l798_79810


namespace krikor_speed_increase_l798_79806

/--
Krikor traveled to work on two consecutive days, Monday and Tuesday, at different speeds.
Both days, he covered the same distance. On Monday, he traveled for 0.5 hours, and on
Tuesday, he traveled for \( \frac{5}{12} \) hours. Prove that the percentage increase in his speed 
from Monday to Tuesday is 20%.
-/
theorem krikor_speed_increase :
  ∀ (v1 v2 : ℝ), (0.5 * v1 = (5 / 12) * v2) → (v2 = (6 / 5) * v1) → 
  ((v2 - v1) / v1 * 100 = 20) :=
by
  -- Proof goes here
  sorry

end krikor_speed_increase_l798_79806


namespace extreme_values_l798_79868

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4 * x + 6

theorem extreme_values :
  (∃ x : ℝ, f x = 34/3 ∧ (x = -2 ∨ x = 4)) ∧
  (∃ x : ℝ, f x = 2/3 ∧ x = 2) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 4, f x ≤ 34/3 ∧ 2/3 ≤ f x) :=
by
  sorry

end extreme_values_l798_79868


namespace fg_at_2_l798_79844

def f (x : ℝ) : ℝ := x^3
def g (x : ℝ) : ℝ := 2*x + 5

theorem fg_at_2 : f (g 2) = 729 := by
  sorry

end fg_at_2_l798_79844


namespace two_times_sum_of_fourth_power_is_perfect_square_l798_79803

theorem two_times_sum_of_fourth_power_is_perfect_square (a b c : ℤ) 
  (h : a + b + c = 0) : 2 * (a^4 + b^4 + c^4) = (a^2 + b^2 + c^2)^2 := 
by sorry

end two_times_sum_of_fourth_power_is_perfect_square_l798_79803


namespace games_bought_at_garage_sale_l798_79878

-- Definitions based on conditions
def games_from_friend : ℕ := 2
def defective_games : ℕ := 2
def good_games : ℕ := 2

-- Prove the number of games bought at the garage sale equals 2
theorem games_bought_at_garage_sale (G : ℕ) 
  (h : games_from_friend + G - defective_games = good_games) : G = 2 :=
by 
  -- use the given information and work out the proof here
  sorry

end games_bought_at_garage_sale_l798_79878


namespace school_spent_on_grass_seeds_bottle_capacity_insufficient_l798_79887

-- Problem 1: Cost Calculation
theorem school_spent_on_grass_seeds (kg_seeds : ℝ) (cost_per_kg : ℝ) (total_cost : ℝ) 
  (h1 : kg_seeds = 3.3) (h2 : cost_per_kg = 9.48) :
  total_cost = 31.284 :=
  by
    sorry

-- Problem 2: Bottle Capacity
theorem bottle_capacity_insufficient (total_seeds : ℝ) (max_capacity_per_bottle : ℝ) (num_bottles : ℕ)
  (h1 : total_seeds = 3.3) (h2 : max_capacity_per_bottle = 0.35) (h3 : num_bottles = 9) :
  3.3 > 0.35 * 9 :=
  by
    sorry

end school_spent_on_grass_seeds_bottle_capacity_insufficient_l798_79887


namespace percentage_paid_to_X_l798_79843

theorem percentage_paid_to_X (X Y : ℝ) (h1 : X + Y = 880) (h2 : Y = 400) : 
  ((X / Y) * 100) = 120 :=
by
  sorry

end percentage_paid_to_X_l798_79843


namespace finitely_many_negative_terms_l798_79811

theorem finitely_many_negative_terms (A : ℝ) :
  (∀ (x : ℕ → ℝ), (∀ n, x n ≠ 0) ∧ (∀ n, x (n+1) = A - 1 / x n) →
  (∃ N, ∀ n ≥ N, x n ≥ 0)) ↔ A ≥ 2 :=
sorry

end finitely_many_negative_terms_l798_79811


namespace smaller_variance_stability_l798_79824

variable {α : Type*}
variable [Nonempty α]

def same_average (X Y : α → ℝ) (avg : ℝ) : Prop := 
  (∀ x, X x = avg) ∧ (∀ y, Y y = avg)

def smaller_variance_is_stable (X Y : α → ℝ) : Prop := 
  (X = Y)

theorem smaller_variance_stability {X Y : α → ℝ} (avg : ℝ) :
  same_average X Y avg → smaller_variance_is_stable X Y :=
by sorry

end smaller_variance_stability_l798_79824


namespace alloy_problem_l798_79823

theorem alloy_problem (x y : ℝ) 
  (h1 : x + y = 1000) 
  (h2 : 0.25 * x + 0.50 * y = 450) 
  (hx : 0 ≤ x) 
  (hy : 0 ≤ y) :
  x = 200 ∧ y = 800 := 
sorry

end alloy_problem_l798_79823


namespace sum_of_turning_angles_l798_79829

variable (radius distance : ℝ) (C : ℝ)

theorem sum_of_turning_angles (H1 : radius = 10) (H2 : distance = 30000) (H3 : C = 2 * radius * Real.pi) :
  (distance / C) * 2 * Real.pi ≥ 2998 :=
by
  sorry

end sum_of_turning_angles_l798_79829


namespace trajectory_equation_l798_79808

theorem trajectory_equation 
  (P : ℝ × ℝ)
  (h : (P.2 / (P.1 + 4)) * (P.2 / (P.1 - 4)) = -4 / 9) :
  P.1 ≠ 4 ∧ P.1 ≠ -4 → P.1^2 / 64 + P.2^2 / (64 / 9) = 1 :=
by
  sorry

end trajectory_equation_l798_79808


namespace problem_inequality_l798_79864

theorem problem_inequality (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) :
  (x^2 - 2*x + 2) * (y^2 - 2*y + 2) * (z^2 - 2*z + 2) ≤ (x*y*z)^2 - 2*(x*y*z) + 2 := sorry

end problem_inequality_l798_79864


namespace measure_of_one_interior_angle_of_regular_octagon_l798_79848

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l798_79848


namespace shot_put_surface_area_l798_79827

noncomputable def radius (d : ℝ) : ℝ := d / 2

noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem shot_put_surface_area :
  surface_area (radius 5) = 25 * Real.pi :=
by
  sorry

end shot_put_surface_area_l798_79827


namespace quadratic_equal_real_roots_l798_79889

theorem quadratic_equal_real_roots :
  ∃ k : ℝ, (∀ x : ℝ, x^2 - 4 * x + k = 0) ∧ k = 4 := by
  sorry

end quadratic_equal_real_roots_l798_79889


namespace fish_filets_total_l798_79854

def fish_caught_by_ben : ℕ := 4
def fish_caught_by_judy : ℕ := 1
def fish_caught_by_billy : ℕ := 3
def fish_caught_by_jim : ℕ := 2
def fish_caught_by_susie : ℕ := 5
def fish_thrown_back : ℕ := 3
def filets_per_fish : ℕ := 2

theorem fish_filets_total : 
  (fish_caught_by_ben + fish_caught_by_judy + fish_caught_by_billy + fish_caught_by_jim + fish_caught_by_susie - fish_thrown_back) * filets_per_fish = 24 := 
by
  sorry

end fish_filets_total_l798_79854


namespace trajectory_midpoint_l798_79835

theorem trajectory_midpoint {x y : ℝ} (hx : 2 * y + 1 = 2 * (2 * x)^2 + 1) :
  y = 4 * x^2 := 
by sorry

end trajectory_midpoint_l798_79835


namespace cars_per_day_l798_79800

noncomputable def paul_rate : ℝ := 2
noncomputable def jack_rate : ℝ := 3
noncomputable def paul_jack_rate : ℝ := paul_rate + jack_rate
noncomputable def hours_per_day : ℝ := 8
noncomputable def total_cars : ℝ := paul_jack_rate * hours_per_day

theorem cars_per_day : total_cars = 40 := by
  sorry

end cars_per_day_l798_79800


namespace remainder_when_dividing_sum_l798_79855

theorem remainder_when_dividing_sum (k m : ℤ) (c d : ℤ) (h1 : c = 60 * k + 47) (h2 : d = 42 * m + 17) :
  (c + d) % 21 = 1 :=
by
  sorry

end remainder_when_dividing_sum_l798_79855


namespace distance_to_x_axis_l798_79842

def point_P : ℝ × ℝ := (2, -3)

theorem distance_to_x_axis : abs (point_P.snd) = 3 := by
  sorry

end distance_to_x_axis_l798_79842


namespace prism_volume_l798_79894

theorem prism_volume 
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 :=
sorry

end prism_volume_l798_79894


namespace ivan_travel_time_l798_79870

theorem ivan_travel_time (d V_I V_P : ℕ) (h1 : d = 3 * V_I * 40)
  (h2 : ∀ t, t = d / V_P + 10) : 
  (d / V_I = 75) :=
by
  sorry

end ivan_travel_time_l798_79870


namespace max_value_point_l798_79865

noncomputable def f (x : ℝ) : ℝ := x + Real.cos (2 * x)

theorem max_value_point : ∃ x ∈ Set.Ioo 0 Real.pi, (∀ y ∈ Set.Ioo 0 Real.pi, f x ≥ f y) ∧ x = Real.pi / 12 :=
by sorry

end max_value_point_l798_79865


namespace find_k_range_l798_79847

open Nat

def a_n (n : ℕ) : ℕ := 2^ (5 - n)

def b_n (n : ℕ) (k : ℤ) : ℤ := n + k

def c_n (n : ℕ) (k : ℤ) : ℤ :=
if (a_n n : ℤ) ≤ (b_n n k) then b_n n k else a_n n

theorem find_k_range : 
  (∀ n ∈ { m : ℕ | m > 0 }, c_n 5 = a_n 5 ∧ c_n 5 ≤ c_n n) → 
  (∃ k : ℤ, -5 ≤ k ∧ k ≤ -3) :=
by
  sorry

end find_k_range_l798_79847


namespace smallest_n_for_g4_l798_79857

def g (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc a => acc + (List.range (n + 1)).countP (λ b => a * a + b * b = n)) 0

theorem smallest_n_for_g4 : ∃ n : ℕ, g n = 4 ∧ 
  (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 65
  -- Proof would go here
  sorry

end smallest_n_for_g4_l798_79857


namespace parabola_p_q_r_sum_l798_79873

noncomputable def parabola_vertex (p q r : ℝ) (x_vertex y_vertex : ℝ) :=
  ∀ (x : ℝ), p * (x - x_vertex) ^ 2 + y_vertex = p * x ^ 2 + q * x + r

theorem parabola_p_q_r_sum
  (p q r : ℝ)
  (vertex_x vertex_y : ℝ)
  (hx_vertex : vertex_x = 3)
  (hy_vertex : vertex_y = 10)
  (h_vertex : parabola_vertex p q r vertex_x vertex_y)
  (h_contains : p * (0 - 3) ^ 2 + 10 = 7) :
  p + q + r = 23 / 3 :=
sorry

end parabola_p_q_r_sum_l798_79873


namespace order_of_abc_l798_79861

noncomputable def a : ℝ := Real.log 3 / Real.log 4
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log (4/3) / Real.log (3/4)

theorem order_of_abc : b > a ∧ a > c := by
  sorry

end order_of_abc_l798_79861


namespace cary_ivy_removal_days_correct_l798_79853

noncomputable def cary_ivy_removal_days (initial_ivy : ℕ) (ivy_removed_per_day : ℕ) (ivy_growth_per_night : ℕ) : ℕ :=
  initial_ivy / (ivy_removed_per_day - ivy_growth_per_night)

theorem cary_ivy_removal_days_correct :
  cary_ivy_removal_days 40 6 2 = 10 :=
by
  -- The body of the proof is omitted; it will be filled with the actual proof.
  sorry

end cary_ivy_removal_days_correct_l798_79853


namespace pizza_cost_difference_l798_79837

theorem pizza_cost_difference :
  let p := 12 -- Cost of plain pizza
  let m := 3 -- Cost of mushrooms
  let o := 4 -- Cost of olives
  let s := 12 -- Total number of slices
  (m + o + p) / s * 10 - (m + o + p) / s * 2 = 12.67 :=
by
  sorry

end pizza_cost_difference_l798_79837


namespace overall_profit_or_loss_l798_79830

def price_USD_to_INR(price_usd : ℝ) : ℝ := price_usd * 75
def price_EUR_to_INR(price_eur : ℝ) : ℝ := price_eur * 80
def price_GBP_to_INR(price_gbp : ℝ) : ℝ := price_gbp * 100
def price_JPY_to_INR(price_jpy : ℝ) : ℝ := price_jpy * 0.7

def CP_grinder : ℝ := price_USD_to_INR (150 + 0.1 * 150)
def SP_grinder : ℝ := price_USD_to_INR (165 - 0.04 * 165)

def CP_mobile_phone : ℝ := price_EUR_to_INR ((100 - 0.05 * 100) + 0.15 * (100 - 0.05 * 100))
def SP_mobile_phone : ℝ := price_EUR_to_INR ((109.25 : ℝ) + 0.1 * 109.25)

def CP_laptop : ℝ := price_GBP_to_INR (200 + 0.08 * 200)
def SP_laptop : ℝ := price_GBP_to_INR (216 - 0.08 * 216)

def CP_camera : ℝ := price_JPY_to_INR ((12000 - 0.12 * 12000) + 0.05 * (12000 - 0.12 * 12000))
def SP_camera : ℝ := price_JPY_to_INR (11088 + 0.15 * 11088)

def total_CP : ℝ := CP_grinder + CP_mobile_phone + CP_laptop + CP_camera
def total_SP : ℝ := SP_grinder + SP_mobile_phone + SP_laptop + SP_camera

theorem overall_profit_or_loss :
  (total_SP - total_CP) = -184.76 := 
sorry

end overall_profit_or_loss_l798_79830


namespace function_is_monotonically_increasing_l798_79875

theorem function_is_monotonically_increasing (a : ℝ) :
  (∀ x : ℝ, (x^2 + 2*x + a) ≥ 0) ↔ (1 ≤ a) := 
sorry

end function_is_monotonically_increasing_l798_79875


namespace calculate_sum_l798_79885

theorem calculate_sum : 5 * 12 + 7 * 15 + 13 * 4 + 6 * 9 = 271 :=
by
  sorry

end calculate_sum_l798_79885


namespace math_problem_l798_79817

noncomputable def problem : Real :=
  (2 * Real.sqrt 2 - 1) ^ 2 + (1 + Real.sqrt 5) * (1 - Real.sqrt 5)

theorem math_problem :
  problem = 5 - 4 * Real.sqrt 2 :=
by
  sorry

end math_problem_l798_79817


namespace final_student_count_l798_79860

def initial_students := 150
def students_joined := 30
def students_left := 15

theorem final_student_count : initial_students + students_joined - students_left = 165 := by
  sorry

end final_student_count_l798_79860


namespace daryl_max_crate_weight_l798_79849

variable (crates : ℕ) (weight_nails : ℕ) (bags_nails : ℕ)
variable (weight_hammers : ℕ) (bags_hammers : ℕ) (weight_planks : ℕ)
variable (bags_planks : ℕ) (weight_left_out : ℕ)

def max_weight_per_crate (total_weight: ℕ) (total_crates: ℕ) : ℕ :=
  total_weight / total_crates

-- State the problem in Lean
theorem daryl_max_crate_weight
  (h1 : crates = 15) 
  (h2 : bags_nails = 4) 
  (h3 : weight_nails = 5)
  (h4 : bags_hammers = 12) 
  (h5 : weight_hammers = 5) 
  (h6 : bags_planks = 10) 
  (h7 : weight_planks = 30) 
  (h8 : weight_left_out = 80):
  max_weight_per_crate ((bags_nails * weight_nails + bags_hammers * weight_hammers + bags_planks * weight_planks) - weight_left_out) crates = 20 :=
  by sorry

end daryl_max_crate_weight_l798_79849


namespace train_length_l798_79804

theorem train_length
  (train_speed_kmph : ℝ)
  (person_speed_kmph : ℝ)
  (time_seconds : ℝ)
  (h_train_speed : train_speed_kmph = 80)
  (h_person_speed : person_speed_kmph = 16)
  (h_time : time_seconds = 15)
  : (train_speed_kmph - person_speed_kmph) * (5/18) * time_seconds = 266.67 := 
by
  rw [h_train_speed, h_person_speed, h_time]
  norm_num
  sorry

end train_length_l798_79804


namespace sara_wrapping_paper_l798_79852

theorem sara_wrapping_paper (s : ℚ) (l : ℚ) (total : ℚ) : 
  total = 3 / 8 → 
  l = 2 * s →
  4 * s + 2 * l = total → 
  s = 3 / 64 :=
by
  intros h1 h2 h3
  sorry

end sara_wrapping_paper_l798_79852


namespace blocks_for_sculpture_l798_79859

noncomputable def volume_block := 8 * 3 * 1
noncomputable def radius_cylinder := 3
noncomputable def height_cylinder := 8
noncomputable def volume_cylinder := Real.pi * radius_cylinder^2 * height_cylinder
noncomputable def blocks_needed := Nat.ceil (volume_cylinder / volume_block)

theorem blocks_for_sculpture : blocks_needed = 10 := by
  sorry

end blocks_for_sculpture_l798_79859


namespace problem_condition_neither_sufficient_nor_necessary_l798_79863

theorem problem_condition_neither_sufficient_nor_necessary 
  (m n : ℕ) (hm : m > 0) (hn : n > 0) (a b : ℝ) :
  (a > b → a^(m + n) + b^(m + n) > a^n * b^m + a^m * b^n) ∧
  (a^(m + n) + b^(m + n) > a^n * b^m + a^m * b^n → a > b) = false :=
by sorry

end problem_condition_neither_sufficient_nor_necessary_l798_79863


namespace larger_number_of_two_l798_79825

theorem larger_number_of_two (x y : ℝ) (h1 : x * y = 30) (h2 : x + y = 13) : max x y = 10 :=
sorry

end larger_number_of_two_l798_79825


namespace candle_blow_out_l798_79880

-- Definitions related to the problem.
def funnel := true -- Simplified representation of the funnel
def candle_lit := true -- Simplified representation of the lit candle
def airflow_concentration (align: Bool) : Prop :=
if align then true -- Airflow intersects the flame correctly
else false -- Airflow does not intersect the flame correctly

theorem candle_blow_out (align : Bool) : funnel ∧ candle_lit ∧ airflow_concentration align → align := sorry

end candle_blow_out_l798_79880


namespace santiago_more_roses_l798_79828

def red_roses_santiago := 58
def red_roses_garrett := 24
def red_roses_difference := red_roses_santiago - red_roses_garrett

theorem santiago_more_roses : red_roses_difference = 34 := by
  sorry

end santiago_more_roses_l798_79828


namespace find_n_l798_79801

theorem find_n (n : ℕ) (d : ℕ) (h_pos : n > 0) (h_digit : d < 10) (h_equiv : n * 999 = 810 * (100 * d + 25)) : n = 750 :=
  sorry

end find_n_l798_79801


namespace no_integer_solutions_l798_79897

theorem no_integer_solutions (x y : ℤ) (hx : x ≠ 1) : (x^7 - 1) / (x - 1) ≠ y^5 - 1 :=
by
  sorry

end no_integer_solutions_l798_79897


namespace maximize_perimeter_l798_79809

theorem maximize_perimeter 
  (l : ℝ) (c_f : ℝ) (C : ℝ) (b : ℝ)
  (hl: l = 400) (hcf: c_f = 5) (hC: C = 1500) :
  ∃ (y : ℝ), y = 180 :=
by
  sorry

end maximize_perimeter_l798_79809


namespace smallest_possible_value_l798_79839

theorem smallest_possible_value 
  (a : ℂ)
  (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ z : ℂ, z = 3 * a + 1 ∧ z.re = -1 / 8 :=
by
  sorry

end smallest_possible_value_l798_79839


namespace roots_of_equation_l798_79851

theorem roots_of_equation (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∀ x : ℝ, a^2 * (x - b) / (a - b) * (x - c) / (a - c) + b^2 * (x - a) / (b - a) * (x - c) / (b - c) + c^2 * (x - a) / (c - a) * (x - b) / (c - b) = x^2 :=
by
  intros
  sorry

end roots_of_equation_l798_79851


namespace group_total_payment_l798_79898

-- Declare the costs of the tickets as constants
def cost_adult : ℝ := 9.50
def cost_child : ℝ := 6.50

-- Conditions for the group
def total_moviegoers : ℕ := 7
def number_adults : ℕ := 3

-- Calculate the number of children
def number_children : ℕ := total_moviegoers - number_adults

-- Define the total cost paid by the group
def total_cost_paid : ℝ :=
  (number_adults * cost_adult) + (number_children * cost_child)

-- The proof problem: Prove that the total amount paid by the group is $54.50
theorem group_total_payment : total_cost_paid = 54.50 := by
  sorry

end group_total_payment_l798_79898


namespace street_length_l798_79892

theorem street_length
  (time_minutes : ℕ)
  (speed_kmph : ℕ)
  (length_meters : ℕ)
  (h1 : time_minutes = 12)
  (h2 : speed_kmph = 9)
  (h3 : length_meters = 1800) :
  length_meters = (speed_kmph * 1000 / 60) * time_minutes :=
by sorry

end street_length_l798_79892


namespace percentage_of_birth_in_june_l798_79874

theorem percentage_of_birth_in_june (total_scientists: ℕ) (born_in_june: ℕ) (h_total: total_scientists = 150) (h_june: born_in_june = 15) : (born_in_june * 100 / total_scientists) = 10 := 
by 
  sorry

end percentage_of_birth_in_june_l798_79874


namespace arithmetic_sequence_seventh_term_l798_79815

theorem arithmetic_sequence_seventh_term (a d : ℝ) 
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 14) 
  (h2 : a + 4 * d = 9) : 
  a + 6 * d = 13.4 := 
sorry

end arithmetic_sequence_seventh_term_l798_79815


namespace num_Q_polynomials_l798_79882

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 5)

#check Exists

theorem num_Q_polynomials :
  ∃ (Q : Polynomial ℝ), 
  (∃ (R : Polynomial ℝ), R.degree = 3 ∧ P (Q.eval x) = P x * R.eval x) ∧
  Q.degree = 2 ∧ (Q.coeff 1 = 6) ∧ (∃ (n : ℕ), n = 22) :=
sorry

end num_Q_polynomials_l798_79882


namespace complex_ratio_of_cubes_l798_79821

theorem complex_ratio_of_cubes (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 10) (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 8 :=
by
  sorry

end complex_ratio_of_cubes_l798_79821


namespace simplify_and_evaluate_expression_l798_79895

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 5 + 1) : 
  ( ( (x^2 - 1) / x ) / (1 + 1 / x) ) = Real.sqrt 5 :=
by 
  sorry

end simplify_and_evaluate_expression_l798_79895


namespace minute_hand_only_rotates_l798_79812

-- Define what constitutes translation and rotation
def is_translation (motion : ℝ → ℝ → Prop) : Prop :=
  ∀ (p1 p2 : ℝ), motion p1 p2 → (∃ d : ℝ, ∀ t : ℝ, motion (p1 + t) (p2 + t) ∧ |p1 - p2| = d)

def is_rotation (motion : ℝ → ℝ → Prop) : Prop :=
  ∀ (p : ℝ), ∃ c : ℝ, ∃ r : ℝ, (∀ (t : ℝ), |p - c| = r)

-- Define the condition that the minute hand of a clock undergoes a specific motion
def minute_hand_motion (p : ℝ) (t : ℝ) : Prop :=
  -- The exact definition here would involve trigonometric representation
  sorry

-- The main proof statement
theorem minute_hand_only_rotates :
  is_rotation minute_hand_motion ∧ ¬ is_translation minute_hand_motion :=
sorry

end minute_hand_only_rotates_l798_79812


namespace total_cost_of_shirts_is_24_l798_79883

-- Definitions based on conditions
def cost_first_shirt : ℕ := 15
def cost_difference : ℕ := 6
def cost_second_shirt : ℕ := cost_first_shirt - cost_difference

-- The proof problem statement: Calculate total cost given the conditions
theorem total_cost_of_shirts_is_24 : cost_first_shirt + cost_second_shshirt = 24 :=
by
  sorry

end total_cost_of_shirts_is_24_l798_79883


namespace percentage_female_guests_from_jay_family_l798_79805

def total_guests : ℕ := 240
def female_guests_percentage : ℕ := 60
def female_guests_from_jay_family : ℕ := 72

theorem percentage_female_guests_from_jay_family :
  (female_guests_from_jay_family : ℚ) / (total_guests * (female_guests_percentage / 100) : ℚ) * 100 = 50 := by
  sorry

end percentage_female_guests_from_jay_family_l798_79805


namespace smallest_prime_after_five_consecutive_nonprimes_l798_79832

theorem smallest_prime_after_five_consecutive_nonprimes :
  ∃ p : ℕ, Nat.Prime p ∧ 
          (∀ n : ℕ, n < p → ¬ (n ≥ 24 ∧ n < 29 ∧ ¬ Nat.Prime n)) ∧
          p = 29 :=
by
  sorry

end smallest_prime_after_five_consecutive_nonprimes_l798_79832


namespace find_smallest_subtract_l798_79862

-- Definitions for multiples
def is_mul_2 (n : ℕ) : Prop := 2 ∣ n
def is_mul_3 (n : ℕ) : Prop := 3 ∣ n
def is_mul_5 (n : ℕ) : Prop := 5 ∣ n

-- Statement of the problem
theorem find_smallest_subtract (x : ℕ) :
  (is_mul_2 (134 - x)) ∧ (is_mul_3 (134 - x)) ∧ (is_mul_5 (134 - x)) → x = 14 :=
by
  sorry

end find_smallest_subtract_l798_79862


namespace quadratic_function_characterization_l798_79836

variable (f : ℝ → ℝ)

def quadratic_function_satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 = 2) ∧ (∀ x, f (x + 1) - f x = 2 * x - 1)

theorem quadratic_function_characterization
  (hf : quadratic_function_satisfies_conditions f) : 
  (∀ x, f x = x^2 - 2 * x + 2) ∧ 
  (f (-1) = 5) ∧ 
  (f 1 = 1) ∧ 
  (f 2 = 2) := by
sorry

end quadratic_function_characterization_l798_79836


namespace correct_operation_l798_79822

theorem correct_operation (a : ℝ) : 2 * a^3 / a^2 = 2 * a := 
sorry

end correct_operation_l798_79822


namespace max_marks_are_700_l798_79877

/-- 
A student has to obtain 33% of the total marks to pass.
The student got 175 marks and failed by 56 marks.
Prove that the maximum marks are 700.
-/
theorem max_marks_are_700 (M : ℝ) (h1 : 0.33 * M = 175 + 56) : M = 700 :=
sorry

end max_marks_are_700_l798_79877


namespace contradiction_in_triangle_l798_79890

theorem contradiction_in_triangle :
  (∀ (A B C : ℝ), A + B + C = 180 ∧ A < 60 ∧ B < 60 ∧ C < 60 → false) :=
by
  sorry

end contradiction_in_triangle_l798_79890


namespace initial_percentage_of_grape_juice_l798_79899

theorem initial_percentage_of_grape_juice
  (P : ℝ)    -- P is the initial percentage in decimal
  (h₁ : 0 ≤ P ∧ P ≤ 1)    -- P is a valid probability
  (h₂ : 40 * P + 10 = 0.36 * 50):    -- Given condition from the problem
  P = 0.2 := 
sorry

end initial_percentage_of_grape_juice_l798_79899


namespace classrooms_students_guinea_pigs_difference_l798_79876

theorem classrooms_students_guinea_pigs_difference :
  let students_per_classroom := 22
  let guinea_pigs_per_classroom := 3
  let number_of_classrooms := 5
  let total_students := students_per_classroom * number_of_classrooms
  let total_guinea_pigs := guinea_pigs_per_classroom * number_of_classrooms
  total_students - total_guinea_pigs = 95 :=
  by
    sorry

end classrooms_students_guinea_pigs_difference_l798_79876


namespace supplement_of_complement_is_125_l798_79833

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l798_79833
