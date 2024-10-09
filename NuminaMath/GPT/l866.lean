import Mathlib

namespace triangle_is_isosceles_l866_86617

theorem triangle_is_isosceles 
  (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) 
  (h_sin_identity : Real.sin A = 2 * Real.sin C * Real.cos B) : 
  (B = C) :=
sorry

end triangle_is_isosceles_l866_86617


namespace part1_part2_l866_86622

section
variable (x y : ℝ)

def A : ℝ := 3 * x^2 + 2 * y^2 - 2 * x * y
def B : ℝ := y^2 - x * y + 2 * x^2

-- Part (1): Prove that 2A - 3B = y^2 - xy
theorem part1 : 2 * A x y - 3 * B x y = y^2 - x * y := 
sorry

-- Part (2): Given |2x - 3| + (y + 2)^2 = 0, prove that 2A - 3B = 7
theorem part2 (h : |2 * x - 3| + (y + 2)^2 = 0) : 2 * A x y - 3 * B x y = 7 :=
sorry

end

end part1_part2_l866_86622


namespace rectangular_sheet_integer_side_l866_86684

theorem rectangular_sheet_integer_side
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_cut_a : ∀ x, x ≤ a → ∃ n : ℕ, x = n ∨ x = n + 1)
  (h_cut_b : ∀ y, y ≤ b → ∃ n : ℕ, y = n ∨ y = n + 1) :
  ∃ n m : ℕ, a = n ∨ b = m := 
sorry

end rectangular_sheet_integer_side_l866_86684


namespace pastries_made_initially_l866_86637

theorem pastries_made_initially 
  (sold : ℕ) (remaining : ℕ) (initial : ℕ) 
  (h1 : sold = 103) (h2 : remaining = 45) : 
  initial = 148 :=
by
  have h := h1
  have r := h2
  sorry

end pastries_made_initially_l866_86637


namespace price_first_variety_is_126_l866_86643

variable (x : ℝ) -- price of the first variety per kg (unknown we need to solve for)
variable (p2 : ℝ := 135) -- price of the second variety per kg
variable (p3 : ℝ := 175.5) -- price of the third variety per kg
variable (mix_ratio : ℝ := 4) -- total weight ratio of the mixture
variable (mix_price : ℝ := 153) -- price of the mixture per kg
variable (w1 w2 w3 : ℝ := 1) -- weights of the first two varieties
variable (w4 : ℝ := 2) -- weight of the third variety

theorem price_first_variety_is_126:
  (w1 * x + w2 * p2 + w4 * p3) / mix_ratio = mix_price → x = 126 := by
  sorry

end price_first_variety_is_126_l866_86643


namespace option_D_correct_l866_86670

-- Definitions representing conditions
variables (a b : Line) (α : Plane)

-- Conditions
def line_parallel_plane (a : Line) (α : Plane) : Prop := sorry
def line_parallel_line (a b : Line) : Prop := sorry
def line_in_plane (b : Line) (α : Plane) : Prop := sorry

-- Theorem stating the correctness of option D
theorem option_D_correct (h1 : line_parallel_plane a α)
                         (h2 : line_parallel_line a b) :
                         (line_in_plane b α) ∨ (line_parallel_plane b α) :=
by
  sorry

end option_D_correct_l866_86670


namespace problem1_problem2_l866_86606

-- Problem 1
theorem problem1 (a b : ℤ) (h1 : a = 4) (h2 : b = 5) : a - b = -1 := 
by {
  sorry
}

-- Problem 2
theorem problem2 (a b m n s : ℤ) (h1 : a + b = 0) (h2 : m * n = 1) (h3 : |s| = 3) :
  a + b + m * n + s = 4 ∨ a + b + m * n + s = -2 := 
by {
  sorry
}

end problem1_problem2_l866_86606


namespace common_difference_range_l866_86695

noncomputable def arithmetic_sequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
  a₁ + (n - 1) * d

theorem common_difference_range :
  let a1 := -24
  let a9 := arithmetic_sequence 9 a1 d
  let a10 := arithmetic_sequence 10 a1 d
  (a10 > 0) ∧ (a9 <= 0) → 8 / 3 < d ∧ d <= 3 :=
by
  let a1 := -24
  let a9 := arithmetic_sequence 9 a1 d
  let a10 := arithmetic_sequence 10 a1 d
  intro h
  sorry

end common_difference_range_l866_86695


namespace find_divisor_l866_86687

-- Define the conditions as hypotheses and the main problem as a theorem
theorem find_divisor (x y : ℕ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 6) / y = 6) : y = 8 := sorry

end find_divisor_l866_86687


namespace correct_meteor_passing_time_l866_86644

theorem correct_meteor_passing_time :
  let T1 := 7
  let T2 := 13
  let harmonic_mean := (2 * T1 * T2) / (T1 + T2)
  harmonic_mean = 9.1 := 
by
  sorry

end correct_meteor_passing_time_l866_86644


namespace integral_root_of_equation_l866_86681

theorem integral_root_of_equation : 
  ∀ x : ℤ, (x - 8 / (x - 4)) = 2 - 8 / (x - 4) ↔ x = 2 := 
sorry

end integral_root_of_equation_l866_86681


namespace smallest_number_is_32_l866_86646

theorem smallest_number_is_32 (a b c : ℕ) (h1 : a + b + c = 90) (h2 : b = 25) (h3 : c = 25 + 8) : a = 32 :=
by {
  sorry
}

end smallest_number_is_32_l866_86646


namespace quadratic_does_not_pass_third_quadrant_l866_86699

-- Definitions of the functions
def linear_function (a b x : ℝ) : ℝ := -a * x + b
def quadratic_function (a b x : ℝ) : ℝ := -a * x^2 + b * x

-- Conditions
variables (a b : ℝ)
axiom a_nonzero : a ≠ 0
axiom passes_first_third_fourth : ∀ x, (linear_function a b x > 0 ∧ x > 0) ∨ (linear_function a b x < 0 ∧ x < 0) ∨ (linear_function a b x < 0 ∧ x > 0)

-- Theorem stating the problem
theorem quadratic_does_not_pass_third_quadrant :
  ¬ (∃ x, quadratic_function a b x < 0 ∧ x < 0) := 
sorry

end quadratic_does_not_pass_third_quadrant_l866_86699


namespace go_stones_perimeter_count_l866_86608

def stones_per_side : ℕ := 6
def sides_of_square : ℕ := 4
def corner_stones : ℕ := 4

theorem go_stones_perimeter_count :
  (stones_per_side * sides_of_square) - corner_stones = 20 := 
by
  sorry

end go_stones_perimeter_count_l866_86608


namespace length_CK_angle_BCA_l866_86693

variables {A B C O O₁ O₂ K K₁ K₂ K₃ : Point}
variables {r R : ℝ}
variables {AC CK AK₁ AK₂ : ℝ}

-- Definitions and conditions
def triangle_ABC (A B C : Point) : Prop := True
def incenter (A B C O : Point) : Prop := True
def in_radius_is_equal (O₁ O₂ : Point) (r : ℝ) : Prop := True
def circle_touches_side (circle_center : Point) (side_point : Point) (distance : ℝ) : Prop := True
def circumcenter (A C B O₁ : Point) : Prop := True
def angle (A B C : Point) (θ : ℝ) : Prop := True

-- Conditions from the problem
axiom cond1 : triangle_ABC A B C
axiom cond2 : in_radius_is_equal O₁ O₂ r
axiom cond3 : incenter A B C O
axiom cond4 : circle_touches_side O₁ K₁ 6
axiom cond5 : circle_touches_side O₂ K₂ 8
axiom cond6 : AC = 21
axiom cond7 : circle_touches_side O K 9
axiom cond8 : circumcenter O K₁ K₃ O₁

-- Statements to prove
theorem length_CK : CK = 9 := by
  sorry

theorem angle_BCA : angle B C A 60 := by
  sorry

end length_CK_angle_BCA_l866_86693


namespace q_can_complete_work_in_30_days_l866_86697

theorem q_can_complete_work_in_30_days (W_p W_q W_r : ℝ)
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = 1/10)
  (h3 : W_r = 1/30) :
  1 / W_q = 30 :=
by
  -- Note: You can add proof here, but it's not required in the task.
  sorry

end q_can_complete_work_in_30_days_l866_86697


namespace exists_numbering_for_nonagon_no_numbering_for_decagon_l866_86602

-- Definitions for the problem setup
variable (n : ℕ) 
variable (A : Fin n → Point)
variable (O : Point)

-- Definition for the numbering function
variable (f : Fin (2 * n) → ℕ)

-- First statement for n = 9
theorem exists_numbering_for_nonagon :
  ∃ (f : Fin 18 → ℕ), (∀ i : Fin 9, f (i : Fin 9) + f (i + 9) + f ((i + 1) % 9) = 15) :=
sorry

-- Second statement for n = 10
theorem no_numbering_for_decagon :
  ¬ ∃ (f : Fin 20 → ℕ), (∀ i : Fin 10, f (i : Fin 10) + f (i + 10) + f ((i + 1) % 10) = 16) :=
sorry

end exists_numbering_for_nonagon_no_numbering_for_decagon_l866_86602


namespace yz_sub_zx_sub_xy_l866_86612

theorem yz_sub_zx_sub_xy (x y z : ℝ) (h1 : x - y - z = 19) (h2 : x^2 + y^2 + z^2 ≠ 19) :
  yz - zx - xy = 171 := by
  sorry

end yz_sub_zx_sub_xy_l866_86612


namespace smallest_class_size_l866_86650

theorem smallest_class_size :
  ∀ (x : ℕ), 4 * x + 3 > 50 → 4 * x + 3 = 51 :=
by
  sorry

end smallest_class_size_l866_86650


namespace chatterboxes_total_jokes_l866_86654

theorem chatterboxes_total_jokes :
  let num_chatterboxes := 10
  let jokes_increasing := (100 * (100 + 1)) / 2
  let jokes_decreasing := (99 * (99 + 1)) / 2
  (jokes_increasing + jokes_decreasing) / num_chatterboxes = 1000 :=
by
  sorry

end chatterboxes_total_jokes_l866_86654


namespace pure_ghee_percentage_l866_86665

theorem pure_ghee_percentage (Q : ℝ) (P : ℝ) (H1 : Q = 10) (H2 : (P / 100) * Q + 10 = 0.80 * (Q + 10)) :
  P = 60 :=
sorry

end pure_ghee_percentage_l866_86665


namespace triangles_in_extended_figure_l866_86604

theorem triangles_in_extended_figure : 
  ∀ (row1_tri : ℕ) (row2_tri : ℕ) (row3_tri : ℕ) (row4_tri : ℕ) 
  (row1_2_med_tri : ℕ) (row2_3_med_tri : ℕ) (row3_4_med_tri : ℕ) 
  (large_tri : ℕ), 
  row1_tri = 6 →
  row2_tri = 5 →
  row3_tri = 4 →
  row4_tri = 3 →
  row1_2_med_tri = 5 →
  row2_3_med_tri = 2 →
  row3_4_med_tri = 1 →
  large_tri = 1 →
  row1_tri + row2_tri + row3_tri + row4_tri
  + row1_2_med_tri + row2_3_med_tri + row3_4_med_tri
  + large_tri = 27 :=
by
  intro row1_tri row2_tri row3_tri row4_tri
  intro row1_2_med_tri row2_3_med_tri row3_4_med_tri
  intro large_tri
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end triangles_in_extended_figure_l866_86604


namespace more_pencils_than_pens_l866_86632

theorem more_pencils_than_pens : 
  ∀ (P L : ℕ), L = 30 → (P / L: ℚ) = 5 / 6 → ((L - P) = 5) := by
  intros P L hL hRatio
  sorry

end more_pencils_than_pens_l866_86632


namespace find_y_given_conditions_l866_86631

def is_value_y (x y : ℕ) : Prop :=
  (100 + 200 + 300 + x) / 4 = 250 ∧ (300 + 150 + 100 + x + y) / 5 = 200

theorem find_y_given_conditions : ∃ y : ℕ, ∀ x : ℕ, (100 + 200 + 300 + x) / 4 = 250 ∧ (300 + 150 + 100 + x + y) / 5 = 200 → y = 50 :=
by
  sorry

end find_y_given_conditions_l866_86631


namespace cos_sq_minus_exp_equals_neg_one_fourth_l866_86689

theorem cos_sq_minus_exp_equals_neg_one_fourth :
  (Real.cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1 / 4 := by
sorry

end cos_sq_minus_exp_equals_neg_one_fourth_l866_86689


namespace geometric_series_sum_eq_l866_86621

theorem geometric_series_sum_eq (a r : ℝ) 
  (h_sum : (∑' n:ℕ, a * r^n) = 20) 
  (h_odd_sum : (∑' n:ℕ, a * r^(2 * n + 1)) = 8) : 
  r = 2 / 3 := 
sorry

end geometric_series_sum_eq_l866_86621


namespace functions_not_exist_l866_86678

theorem functions_not_exist :
  ¬ (∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), x ≠ y → |f x - f y| + |g x - g y| > 1) :=
by
  sorry

end functions_not_exist_l866_86678


namespace solution_set_f_lt_zero_a_two_solution_set_f_gt_zero_l866_86668

-- Given function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a - 1) * x - a

-- Problem 1: for a = 2, solution to f(x) < 0
theorem solution_set_f_lt_zero_a_two :
  { x : ℝ | f x 2 < 0 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

-- Problem 2: for any a in ℝ, solution to f(x) > 0
theorem solution_set_f_gt_zero (a : ℝ) :
  { x : ℝ | f x a > 0 } =
  if a > -1 then
    {x : ℝ | x < -1} ∪ {x : ℝ | x > a}
  else if a = -1 then
    {x : ℝ | x ≠ -1}
  else
    {x : ℝ | x < a} ∪ {x : ℝ | x > -1} :=
sorry

end solution_set_f_lt_zero_a_two_solution_set_f_gt_zero_l866_86668


namespace symmetric_y_axis_function_l866_86615

theorem symmetric_y_axis_function (f g : ℝ → ℝ) (h : ∀ (x : ℝ), g x = 3^x + 1) :
  (∀ x, f x = f (-x)) → (∀ x, f x = g (-x)) → (∀ x, f x = 3^(-x) + 1) :=
by
  intros h1 h2
  sorry

end symmetric_y_axis_function_l866_86615


namespace hashtag_3_8_l866_86688

-- Define the hashtag operation
def hashtag (a b : ℤ) : ℤ := a * b - b + b ^ 2

-- Prove that 3 # 8 equals 80
theorem hashtag_3_8 : hashtag 3 8 = 80 := by
  sorry

end hashtag_3_8_l866_86688


namespace angle_of_skew_lines_in_range_l866_86690

noncomputable def angle_between_skew_lines (θ : ℝ) (θ_range : 0 < θ ∧ θ ≤ 90) : Prop :=
  θ ∈ (Set.Ioc 0 90)

-- We assume the existence of such an angle θ formed by two skew lines
theorem angle_of_skew_lines_in_range (θ : ℝ) (h_skew : true) : angle_between_skew_lines θ (⟨sorry, sorry⟩) :=
  sorry

end angle_of_skew_lines_in_range_l866_86690


namespace loss_of_30_yuan_is_minus_30_yuan_l866_86660

def profit (p : ℤ) : Prop := p = 20
def loss (l : ℤ) : Prop := l = -30

theorem loss_of_30_yuan_is_minus_30_yuan (p : ℤ) (l : ℤ) (h : profit p) : loss l :=
by
  sorry

end loss_of_30_yuan_is_minus_30_yuan_l866_86660


namespace train_crosses_bridge_in_30_seconds_l866_86640

noncomputable def train_length : ℝ := 100
noncomputable def bridge_length : ℝ := 200
noncomputable def train_speed_kmph : ℝ := 36

noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

noncomputable def total_distance : ℝ := train_length + bridge_length

noncomputable def crossing_time : ℝ := total_distance / train_speed_mps

theorem train_crosses_bridge_in_30_seconds :
  crossing_time = 30 := 
by
  sorry

end train_crosses_bridge_in_30_seconds_l866_86640


namespace gcd_48_72_120_l866_86655

theorem gcd_48_72_120 : Nat.gcd (Nat.gcd 48 72) 120 = 24 :=
by
  sorry

end gcd_48_72_120_l866_86655


namespace fraction_identity_l866_86652

theorem fraction_identity
  (m : ℝ)
  (h : (m - 1) / m = 3) : (m^2 + 1) / m^2 = 5 :=
by
  sorry

end fraction_identity_l866_86652


namespace polynomial_sum_l866_86671

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l866_86671


namespace base7_addition_l866_86682

theorem base7_addition (X Y : ℕ) (h1 : X + 5 = 9) (h2 : Y + 2 = 4) : X + Y = 6 :=
by
  sorry

end base7_addition_l866_86682


namespace selection_and_arrangement_l866_86698

-- Defining the problem conditions
def volunteers : Nat := 5
def roles : Nat := 4
def A_excluded_role : String := "music_composer"
def total_methods : Nat := 96

theorem selection_and_arrangement (h1 : volunteers = 5) (h2 : roles = 4) (h3 : A_excluded_role = "music_composer") :
  total_methods = 96 :=
by
  sorry

end selection_and_arrangement_l866_86698


namespace hyperbola_eccentricity_l866_86675

theorem hyperbola_eccentricity (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∀ x : ℝ, y = (3 / 4) * x → y = (b / a) * x) : 
  (b = (3 / 4) * a) → (e = 5 / 4) := 
by
  sorry

end hyperbola_eccentricity_l866_86675


namespace total_cost_is_21_l866_86664

-- Definitions of the costs
def cost_almond_croissant : Float := 4.50
def cost_salami_and_cheese_croissant : Float := 4.50
def cost_plain_croissant : Float := 3.00
def cost_focaccia : Float := 4.00
def cost_latte : Float := 2.50

-- Theorem stating the total cost
theorem total_cost_is_21 :
  (cost_almond_croissant + cost_salami_and_cheese_croissant) + (2 * cost_latte) + cost_plain_croissant + cost_focaccia = 21.00 :=
by
  sorry

end total_cost_is_21_l866_86664


namespace no_nontrivial_integer_solutions_l866_86639

theorem no_nontrivial_integer_solutions (a b c d : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * d^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
by
  intro h
  sorry

end no_nontrivial_integer_solutions_l866_86639


namespace avg_score_is_94_l866_86692

-- Define the math scores of the four children
def june_score : ℕ := 97
def patty_score : ℕ := 85
def josh_score : ℕ := 100
def henry_score : ℕ := 94

-- Define the total number of children
def num_children : ℕ := 4

-- Define the total score
def total_score : ℕ := june_score + patty_score + josh_score + henry_score

-- Define the average score
def avg_score : ℕ := total_score / num_children

-- The theorem we want to prove
theorem avg_score_is_94 : avg_score = 94 := by
  -- skipping the proof
  sorry

end avg_score_is_94_l866_86692


namespace max_workers_l866_86657

-- Each worker produces 10 bricks a day and steals as many bricks per day as there are workers at the factory.
def worker_bricks_produced_per_day : ℕ := 10
def worker_bricks_stolen_per_day (n : ℕ) : ℕ := n

-- The factory must have at least 13 more bricks at the end of the day.
def factory_brick_surplus_requirement : ℕ := 13

-- Prove the maximum number of workers that can be hired so that the factory has at least 13 more bricks than at the beginning:
theorem max_workers
  (n : ℕ) -- Let \( n \) be the number of workers at the brick factory.
  (h : worker_bricks_produced_per_day * n - worker_bricks_stolen_per_day n + 13 ≥ factory_brick_surplus_requirement): 
  n = 8 := 
sorry

end max_workers_l866_86657


namespace calculate_difference_square_l866_86616

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end calculate_difference_square_l866_86616


namespace counterexample_exists_l866_86659

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- State the theorem equivalently in Lean
theorem counterexample_exists : (sum_of_digits 33 % 6 = 0) ∧ (33 % 6 ≠ 0) := by
  sorry

end counterexample_exists_l866_86659


namespace zoe_total_cost_l866_86651

theorem zoe_total_cost 
  (app_cost : ℕ)
  (monthly_cost : ℕ)
  (item_cost : ℕ)
  (feature_cost : ℕ)
  (months_played : ℕ)
  (h1 : app_cost = 5)
  (h2 : monthly_cost = 8)
  (h3 : item_cost = 10)
  (h4 : feature_cost = 12)
  (h5 : months_played = 2) :
  app_cost + (months_played * monthly_cost) + item_cost + feature_cost = 43 := 
by 
  sorry

end zoe_total_cost_l866_86651


namespace coordinates_with_respect_to_origin_l866_86645

theorem coordinates_with_respect_to_origin (P : ℝ × ℝ) (h : P = (2, -3)) : P = (2, -3) :=
by
  sorry

end coordinates_with_respect_to_origin_l866_86645


namespace number_of_circles_l866_86620

theorem number_of_circles (side : ℝ) (enclosed_area : ℝ) (num_circles : ℕ) (radius : ℝ) :
  side = 14 ∧ enclosed_area = 42.06195997410015 ∧ 2 * radius = side ∧ π * radius^2 = 49 * π → num_circles = 4 :=
by
  intros
  sorry

end number_of_circles_l866_86620


namespace passenger_speed_relative_forward_correct_l866_86696

-- Define the conditions
def train_speed : ℝ := 60     -- Train's speed in km/h
def passenger_speed_inside_train : ℝ := 3  -- Passenger's speed inside the train in km/h

-- Define the effective speed of the passenger relative to the railway track when moving forward
def passenger_speed_relative_forward (train_speed passenger_speed_inside_train : ℝ) : ℝ :=
  train_speed + passenger_speed_inside_train

-- Prove that the passenger's speed relative to the railway track is 63 km/h when moving forward
theorem passenger_speed_relative_forward_correct :
  passenger_speed_relative_forward train_speed passenger_speed_inside_train = 63 := by
  sorry

end passenger_speed_relative_forward_correct_l866_86696


namespace unique_three_digit_multiple_of_66_ending_in_4_l866_86662

theorem unique_three_digit_multiple_of_66_ending_in_4 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 66 = 0 ∧ n % 10 = 4 := sorry

end unique_three_digit_multiple_of_66_ending_in_4_l866_86662


namespace solve_for_x_l866_86614

theorem solve_for_x (x : ℝ) (hp : 0 < x) (h : 4 * x^2 = 1024) : x = 16 :=
sorry

end solve_for_x_l866_86614


namespace find_m_l866_86663

noncomputable def given_hyperbola (x y : ℝ) (m : ℝ) : Prop :=
    x^2 / m - y^2 / 3 = 1

noncomputable def hyperbola_eccentricity (m : ℝ) (e : ℝ) : Prop :=
    e = Real.sqrt (1 + 3 / m)

theorem find_m (m : ℝ) (h1 : given_hyperbola 1 1 m) (h2 : hyperbola_eccentricity m 2) : m = 1 :=
by
  sorry

end find_m_l866_86663


namespace shaded_area_fraction_l866_86634

theorem shaded_area_fraction (total_grid_squares : ℕ) (number_1_squares : ℕ) (number_9_squares : ℕ) (number_8_squares : ℕ) (partial_squares_1 : ℕ) (partial_squares_2 : ℕ) (partial_squares_3 : ℕ) :
  total_grid_squares = 18 * 8 →
  number_1_squares = 8 →
  number_9_squares = 15 →
  number_8_squares = 16 →
  partial_squares_1 = 6 →
  partial_squares_2 = 6 →
  partial_squares_3 = 8 →
  (2 * (number_1_squares + number_9_squares + number_9_squares + number_8_squares) + (partial_squares_1 + partial_squares_2 + partial_squares_3)) = 2 * (74 : ℕ) →
  (74 / 144 : ℚ) = 37 / 72 :=
by
  intros _ _ _ _ _ _ _ _
  sorry

end shaded_area_fraction_l866_86634


namespace angle_sum_x_y_l866_86673

theorem angle_sum_x_y 
  (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) (x : ℝ) (y : ℝ) 
  (hA : angle_A = 34) (hB : angle_B = 80) (hC : angle_C = 30) 
  (hexagon_property : ∀ A B x y : ℝ, A + B + 360 - x + 90 + 120 - y = 720) :
  x + y = 36 :=
by
  sorry

end angle_sum_x_y_l866_86673


namespace part_1_part_2_l866_86680

theorem part_1 (a b A B : ℝ)
  (h : b * (Real.sin A)^2 = Real.sqrt 3 * a * Real.cos A * Real.sin B) 
  (h_sine_law : b / Real.sin B = a / Real.sin A)
  (A_in_range: A ∈ Set.Ioo 0 Real.pi):
  A = Real.pi / 3 := 
sorry

theorem part_2 (x : ℝ)
  (A : ℝ := Real.pi / 3)
  (h_sin_cos : ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
                f x = (Real.sin A * (Real.cos x)^2) - (Real.sin (A / 2))^2 * (Real.sin (2 * x))) :
  Set.image f (Set.Icc 0 (Real.pi / 2)) = Set.Icc ((Real.sqrt 3 - 2)/4) (Real.sqrt 3 / 2) :=
sorry

end part_1_part_2_l866_86680


namespace find_Allyson_age_l866_86677

variable (Hiram Allyson : ℕ)

theorem find_Allyson_age (h : Hiram = 40)
  (condition : Hiram + 12 = 2 * Allyson - 4) : Allyson = 28 := by
  sorry

end find_Allyson_age_l866_86677


namespace total_marbles_l866_86610

-- Definitions to state the problem
variables {r b g : ℕ}
axiom ratio_condition : r / b = 2 / 4 ∧ r / g = 2 / 6
axiom blue_marbles : b = 30

-- Theorem statement
theorem total_marbles : r + b + g = 90 :=
by sorry

end total_marbles_l866_86610


namespace sum_of_cubes_eq_91_l866_86636

theorem sum_of_cubes_eq_91 (a b : ℤ) (h₁ : a^3 + b^3 = 91) (h₂ : a * b = 12) : a^3 + b^3 = 91 :=
by
  exact h₁

end sum_of_cubes_eq_91_l866_86636


namespace find_n_divides_2n_plus_2_l866_86674

theorem find_n_divides_2n_plus_2 :
  ∃ n : ℕ, (100 ≤ n ∧ n ≤ 1997 ∧ n ∣ (2 * n + 2)) ∧ n = 946 :=
by {
  sorry
}

end find_n_divides_2n_plus_2_l866_86674


namespace inequality_solution_l866_86607

theorem inequality_solution (x : ℝ) (h : 4 ≤ |x + 2| ∧ |x + 2| ≤ 8) :
  (-10 : ℝ) ≤ x ∧ x ≤ -6 ∨ (2 : ℝ) ≤ x ∧ x ≤ 6 :=
sorry

end inequality_solution_l866_86607


namespace flat_fee_shipping_l866_86676

theorem flat_fee_shipping (w : ℝ) (c : ℝ) (C : ℝ) (F : ℝ) 
  (h_w : w = 5) 
  (h_c : c = 0.80) 
  (h_C : C = 9)
  (h_shipping : C = F + (c * w)) :
  F = 5 :=
by
  -- proof skipped
  sorry

end flat_fee_shipping_l866_86676


namespace K_time_correct_l866_86647

open Real

noncomputable def K_speed : ℝ := sorry
noncomputable def M_speed : ℝ := K_speed - 1 / 2
noncomputable def K_time : ℝ := 45 / K_speed
noncomputable def M_time : ℝ := 45 / M_speed

theorem K_time_correct (K_speed_correct : 45 / K_speed - 45 / M_speed = 1 / 2) : K_time = 45 / K_speed :=
by
  sorry

end K_time_correct_l866_86647


namespace wire_problem_l866_86686

theorem wire_problem (a b : ℝ) (h_perimeter : a = b) : a / b = 1 := by
  sorry

end wire_problem_l866_86686


namespace two_colonies_reach_limit_in_same_time_l866_86648

theorem two_colonies_reach_limit_in_same_time (d : ℕ) (h : 16 = d): 
  d = 16 :=
by
  /- Asserting that if one colony takes 16 days, two starting together will also take 16 days -/
  sorry

end two_colonies_reach_limit_in_same_time_l866_86648


namespace point_in_second_quadrant_l866_86625

theorem point_in_second_quadrant {x : ℝ} (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
sorry

end point_in_second_quadrant_l866_86625


namespace cartesian_coordinates_problem_l866_86623

theorem cartesian_coordinates_problem
  (x1 y1 x2 y2 : ℕ)
  (h1 : x1 < y1)
  (h2 : x2 > y2)
  (h3 : x2 * y2 = x1 * y1 + 67)
  (h4 : 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2)
  : Nat.digits 10 (x1 * 1000 + y1 * 100 + x2 * 10 + y2) = [1, 9, 8, 5] :=
by
  sorry

end cartesian_coordinates_problem_l866_86623


namespace solution_set_of_inequality_l866_86642

theorem solution_set_of_inequality (x : ℝ) : 
  |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1 :=
sorry

end solution_set_of_inequality_l866_86642


namespace number_of_pupils_l866_86603

theorem number_of_pupils (n : ℕ) 
  (h1 : 83 - 63 = 20) 
  (h2 : (20 : ℝ) / n = 1 / 2) : 
  n = 40 := 
sorry

end number_of_pupils_l866_86603


namespace max_abs_sum_l866_86630

-- Define the condition for the ellipse equation
def ellipse_condition (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Prove that the largest possible value of |x| + |y| given the condition is 2√3
theorem max_abs_sum (x y : ℝ) (h : ellipse_condition x y) : |x| + |y| ≤ 2 * Real.sqrt 3 :=
sorry

end max_abs_sum_l866_86630


namespace problem_solution_l866_86683

-- Define the variables and the conditions
variable (a b c : ℝ)
axiom h1 : a^2 + 2 * b = 7
axiom h2 : b^2 - 2 * c = -1
axiom h3 : c^2 - 6 * a = -17

-- State the theorem to be proven
theorem problem_solution : a + b + c = 3 := 
by sorry

end problem_solution_l866_86683


namespace sam_investment_l866_86624

noncomputable def compound_interest (P: ℝ) (r: ℝ) (n: ℕ) (t: ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sam_investment :
  compound_interest 3000 0.10 4 1 = 3311.44 :=
by
  sorry

end sam_investment_l866_86624


namespace total_people_on_hike_l866_86666

-- Definitions of the conditions
def n_cars : ℕ := 3
def n_people_per_car : ℕ := 4
def n_taxis : ℕ := 6
def n_people_per_taxi : ℕ := 6
def n_vans : ℕ := 2
def n_people_per_van : ℕ := 5

-- Statement of the problem
theorem total_people_on_hike : 
  n_cars * n_people_per_car + n_taxis * n_people_per_taxi + n_vans * n_people_per_van = 58 :=
by sorry

end total_people_on_hike_l866_86666


namespace intersection_with_y_axis_l866_86628

theorem intersection_with_y_axis (y : ℝ) : 
  (∃ y, (0, y) ∈ {(x, 2 * x + 4) | x : ℝ}) ↔ y = 4 :=
by 
  sorry

end intersection_with_y_axis_l866_86628


namespace value_of_expression_l866_86626

theorem value_of_expression (a b : ℝ) (h1 : a^2 + 2012 * a + 1 = 0) (h2 : b^2 + 2012 * b + 1 = 0) :
  (2 + 2013 * a + a^2) * (2 + 2013 * b + b^2) = -2010 := 
  sorry

end value_of_expression_l866_86626


namespace min_distance_between_parallel_lines_distance_when_line_parallel_to_x_axis_l866_86613

noncomputable def line_equation (A B C x y : ℝ) : Prop := A * x + B * y + C = 0

noncomputable def point_on_line (x y A B C : ℝ) : Prop := line_equation A B C x y

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  (|C2 - C1|) / (Real.sqrt (A^2 + B^2))

theorem min_distance_between_parallel_lines :
  ∀ (A B C1 C2 x y : ℝ),
  point_on_line x y A B C1 ∧ point_on_line x y A B C2 →
  distance_between_parallel_lines A B C1 C2 = 3 :=
by
  intros A B C1 C2 x y h
  sorry

theorem distance_when_line_parallel_to_x_axis :
  ∀ (x1 x2 y k A B C1 C2 : ℝ),
  k = 3 →
  point_on_line x1 k A B C1 →
  point_on_line x2 k A B C2 →
  |x2 - x1| = 5 :=
by
  intros x1 x2 y k A B C1 C2 hk h1 h2
  sorry

end min_distance_between_parallel_lines_distance_when_line_parallel_to_x_axis_l866_86613


namespace gcd_lcm_888_1147_l866_86649

theorem gcd_lcm_888_1147 :
  Nat.gcd 888 1147 = 37 ∧ Nat.lcm 888 1147 = 27528 := by
  sorry

end gcd_lcm_888_1147_l866_86649


namespace max_product_sum_1988_l866_86679

theorem max_product_sum_1988 :
  ∃ (n : ℕ) (a : ℕ), n + a = 1988 ∧ a = 1 ∧ n = 662 ∧ (3^n * 2^a) = 2 * 3^662 :=
by
  sorry

end max_product_sum_1988_l866_86679


namespace range_of_m_l866_86658

noncomputable def y (m x : ℝ) := m * (1/4)^x - (1/2)^x + 1

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, y m x = 0) → (m ≤ 0 ∨ m = 1 / 4) := sorry

end range_of_m_l866_86658


namespace minimum_n_for_3_zeros_l866_86685

theorem minimum_n_for_3_zeros :
  ∃ n : ℕ, (∀ m : ℕ, (m < n → ∀ k < 10, m + k ≠ 5 * m ∧ m + k ≠ 5 * m + 25)) ∧
  (∀ k < 10, n + k = 16 ∨ n + k = 16 + 9) ∧
  n = 16 :=
sorry

end minimum_n_for_3_zeros_l866_86685


namespace min_value_x2_2xy_y2_l866_86619

theorem min_value_x2_2xy_y2 (x y : ℝ) : ∃ (a b : ℝ), (x = a ∧ y = b) → x^2 + 2*x*y + y^2 = 0 :=
by {
  sorry
}

end min_value_x2_2xy_y2_l866_86619


namespace sum_areas_frequency_distribution_histogram_l866_86618

theorem sum_areas_frequency_distribution_histogram :
  ∀ (rectangles : List ℝ), (∀ r ∈ rectangles, 0 ≤ r ∧ r ≤ 1) → rectangles.sum = 1 := 
  by
    intro rectangles h
    sorry

end sum_areas_frequency_distribution_histogram_l866_86618


namespace percentage_subtraction_l866_86656

variable (a b x m : ℝ) (p : ℝ)

-- Conditions extracted from the problem.
def ratio_a_to_b : Prop := a / b = 4 / 5
def definition_of_x : Prop := x = 1.75 * a
def definition_of_m : Prop := m = b * (1 - p / 100)
def value_m_div_x : Prop := m / x = 0.14285714285714285

-- The proof problem in the form of a Lean statement.
theorem percentage_subtraction 
  (h1 : ratio_a_to_b a b)
  (h2 : definition_of_x a x)
  (h3 : definition_of_m b m p)
  (h4 : value_m_div_x x m) : p = 80 := 
sorry

end percentage_subtraction_l866_86656


namespace find_x_l866_86611

open Real

noncomputable def log_base (b x : ℝ) : ℝ := log x / log b

theorem find_x :
  ∃ x : ℝ, 0 < x ∧
  log_base 5 (x - 1) + log_base (sqrt 5) (x^2 - 1) + log_base (1/5) (x - 1) = 3 ∧
  x = sqrt (5 * sqrt 5 + 1) :=
by
  sorry

end find_x_l866_86611


namespace problem1_problem2_l866_86691

-- Problem 1
theorem problem1 (x : ℝ) : 
  (x + 2) * (x - 2) - 2 * (x - 3) = 3 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (x + 3)^2 = (1 - 2 * x)^2 ↔ x = 4 ∨ x = -2 / 3 := 
sorry

end problem1_problem2_l866_86691


namespace quadratic_one_real_root_l866_86635

theorem quadratic_one_real_root (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - 2 * x + 1 = 0) ↔ ((a = 0) ∨ (a = 1))) :=
sorry

end quadratic_one_real_root_l866_86635


namespace statement_A_statement_B_statement_C_statement_D_statement_E_l866_86600

def diamond (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem statement_A : ∀ (x y : ℝ), diamond x y = diamond y x := sorry

theorem statement_B : ∀ (x y : ℝ), 2 * (diamond x y) ≠ diamond (2 * x) (2 * y) := sorry

theorem statement_C : ∀ (x : ℝ), diamond x 0 = x^2 := sorry

theorem statement_D : ∀ (x : ℝ), diamond x x = 0 := sorry

theorem statement_E : ∀ (x y : ℝ), x = y → diamond x y = 0 := sorry

end statement_A_statement_B_statement_C_statement_D_statement_E_l866_86600


namespace pipe_A_fill_time_l866_86601

theorem pipe_A_fill_time :
  (∃ x : ℕ, (1 / (x : ℝ) + 1 / 60 - 1 / 72 = 1 / 40) ∧ x = 45) :=
sorry

end pipe_A_fill_time_l866_86601


namespace smallest_base10_integer_l866_86641

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end smallest_base10_integer_l866_86641


namespace box_2008_count_l866_86629

noncomputable def box_count (a : ℕ → ℕ) : Prop :=
  a 1 = 7 ∧ a 4 = 8 ∧ ∀ n : ℕ, 1 ≤ n ∧ n + 3 ≤ 2008 → a n + a (n + 1) + a (n + 2) + a (n + 3) = 30

theorem box_2008_count (a : ℕ → ℕ) (h : box_count a) : a 2008 = 8 :=
by
  sorry

end box_2008_count_l866_86629


namespace trapezium_distance_l866_86672

theorem trapezium_distance (h : ℝ) (a b A : ℝ) 
  (h_area : A = 95) (h_a : a = 20) (h_b : b = 18) :
  A = (1/2 * (a + b) * h) → h = 5 :=
by
  sorry

end trapezium_distance_l866_86672


namespace number_of_partitions_indistinguishable_balls_into_boxes_l866_86694

/-- The number of distinct ways to partition 6 indistinguishable balls into 3 indistinguishable boxes is 7. -/
theorem number_of_partitions_indistinguishable_balls_into_boxes :
  ∃ n : ℕ, n = 7 := sorry

end number_of_partitions_indistinguishable_balls_into_boxes_l866_86694


namespace cube_faces_sum_l866_86669

open Nat

theorem cube_faces_sum (a b c d e f : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) 
    (h7 : (a + d) * (b + e) * (c + f) = 1386) : 
    a + b + c + d + e + f = 38 := 
sorry

end cube_faces_sum_l866_86669


namespace parallel_segment_length_l866_86667

/-- In \( \triangle ABC \), given side lengths AB = 500, BC = 550, and AC = 650,
there exists an interior point P such that each segment drawn parallel to the
sides of the triangle and passing through P splits the sides into segments proportional
to the overall sides of the triangle. Prove that the length \( d \) of each segment
parallel to the sides is 28.25 -/
theorem parallel_segment_length
  (A B C P : Type)
  (d AB BC AC : ℝ)
  (ha : AB = 500)
  (hb : BC = 550)
  (hc : AC = 650)
  (hp : AB * BC = AC * 550) -- This condition ensures proportionality of segments
  : d = 28.25 :=
sorry

end parallel_segment_length_l866_86667


namespace alyssa_games_this_year_l866_86653

theorem alyssa_games_this_year : 
    ∀ (X: ℕ), 
    (13 + X + 15 = 39) → 
    X = 11 := 
by
  intros X h
  have h₁ : 13 + 15 = 28 := by norm_num
  have h₂ : X + 28 = 39 := by linarith
  have h₃ : X = 11 := by linarith
  exact h₃

end alyssa_games_this_year_l866_86653


namespace number_of_convex_quadrilaterals_l866_86633

-- Each definition used in Lean 4 statement should directly appear in the conditions problem.

variable {n : ℕ} -- Definition of n in Lean

-- Conditions
def distinct_points_on_circle (n : ℕ) : Prop := n = 10

-- Question and correct answer
theorem number_of_convex_quadrilaterals (h : distinct_points_on_circle n) : 
    (n.choose 4) = 210 := by
  sorry

end number_of_convex_quadrilaterals_l866_86633


namespace price_of_large_pizza_l866_86609

variable {price_small_pizza : ℕ}
variable {total_revenue : ℕ}
variable {small_pizzas_sold : ℕ}
variable {large_pizzas_sold : ℕ}
variable {price_large_pizza : ℕ}

theorem price_of_large_pizza
  (h1 : price_small_pizza = 2)
  (h2 : total_revenue = 40)
  (h3 : small_pizzas_sold = 8)
  (h4 : large_pizzas_sold = 3) :
  price_large_pizza = 8 :=
by
  sorry

end price_of_large_pizza_l866_86609


namespace how_many_toys_l866_86627

theorem how_many_toys (initial_savings : ℕ) (allowance : ℕ) (toy_cost : ℕ)
  (h1 : initial_savings = 21)
  (h2 : allowance = 15)
  (h3 : toy_cost = 6) :
  (initial_savings + allowance) / toy_cost = 6 :=
by
  sorry

end how_many_toys_l866_86627


namespace number_of_herds_l866_86661

-- Definitions from the conditions
def total_sheep : ℕ := 60
def sheep_per_herd : ℕ := 20

-- The statement to prove
theorem number_of_herds : total_sheep / sheep_per_herd = 3 := by
  sorry

end number_of_herds_l866_86661


namespace perimeter_of_cube_face_is_28_l866_86638

-- Define the volume of the cube
def volume_of_cube : ℝ := 343

-- Define the side length of the cube based on the volume
def side_length_of_cube : ℝ := volume_of_cube^(1/3)

-- Define the perimeter of one face of the cube
def perimeter_of_one_face (side_length : ℝ) : ℝ := 4 * side_length

-- Theorem: Prove the perimeter of one face of the cube is 28 cm given the volume is 343 cm³
theorem perimeter_of_cube_face_is_28 : 
  perimeter_of_one_face side_length_of_cube = 28 := 
by
  sorry

end perimeter_of_cube_face_is_28_l866_86638


namespace joan_remaining_oranges_l866_86605

def total_oranges_joan_picked : ℕ := 37
def oranges_sara_sold : ℕ := 10

theorem joan_remaining_oranges : total_oranges_joan_picked - oranges_sara_sold = 27 := by
  sorry

end joan_remaining_oranges_l866_86605
