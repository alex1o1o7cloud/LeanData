import Mathlib

namespace Tim_sweets_are_multiple_of_4_l225_22567

-- Define the conditions
def sweets_are_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Given definitions
def Peter_sweets : ℕ := 44
def largest_possible_number_per_tray : ℕ := 4

-- Define the proposition to be proven
theorem Tim_sweets_are_multiple_of_4 (O : ℕ) (h1 : sweets_are_divisible_by_4 Peter_sweets) (h2 : sweets_are_divisible_by_4 largest_possible_number_per_tray) :
  sweets_are_divisible_by_4 O :=
sorry

end Tim_sweets_are_multiple_of_4_l225_22567


namespace express_train_speed_ratio_l225_22589

noncomputable def speed_ratio (c h : ℝ) (x : ℝ) : Prop :=
  let t1 := h / ((1 + x) * c)
  let t2 := h / ((x - 1) * c)
  x = t2 / t1

theorem express_train_speed_ratio 
  (c h : ℝ) (x : ℝ) 
  (hc : c > 0) (hh : h > 0) (hx : x > 1) : 
  speed_ratio c h (1 + Real.sqrt 2) := 
by
  sorry

end express_train_speed_ratio_l225_22589


namespace add_base_12_l225_22581

theorem add_base_12 :
  let a := 5*12^2 + 1*12^1 + 8*12^0
  let b := 2*12^2 + 7*12^1 + 6*12^0
  let result := 7*12^2 + 9*12^1 + 2*12^0
  a + b = result :=
by
  -- Placeholder for the actual proof
  sorry

end add_base_12_l225_22581


namespace volume_s_l225_22514

def condition1 (x y : ℝ) : Prop := |9 - x| + y ≤ 12
def condition2 (x y : ℝ) : Prop := 3 * y - x ≥ 18
def S (x y : ℝ) : Prop := condition1 x y ∧ condition2 x y

def is_volume_correct (m n : ℕ) (p : ℕ) :=
  (m + n + p = 153) ∧ (m = 135) ∧ (n = 8) ∧ (p = 10)

theorem volume_s (m n p : ℕ) :
  (∀ x y : ℝ, S x y) → is_volume_correct m n p :=
by 
  sorry

end volume_s_l225_22514


namespace max_min_x2_sub_xy_add_y2_l225_22527

/-- Given a point \((x, y)\) on the curve defined by \( |5x + y| + |5x - y| = 20 \), prove that the maximum value of \(x^2 - xy + y^2\) is 124 and the minimum value is 3. -/
theorem max_min_x2_sub_xy_add_y2 (x y : ℝ) (h : abs (5 * x + y) + abs (5 * x - y) = 20) :
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end max_min_x2_sub_xy_add_y2_l225_22527


namespace total_worksheets_l225_22540

theorem total_worksheets (x : ℕ) (h1 : 7 * (x - 8) = 63) : x = 17 := 
by {
  sorry
}

end total_worksheets_l225_22540


namespace tan_neg_two_sin_cos_sum_l225_22523

theorem tan_neg_two_sin_cos_sum (θ : ℝ) (h : Real.tan θ = -2) : 
  Real.sin (2 * θ) + Real.cos (2 * θ) = -7 / 5 :=
by
  sorry

end tan_neg_two_sin_cos_sum_l225_22523


namespace average_annual_growth_rate_eq_l225_22565

-- Definition of variables based on given conditions
def sales_2021 := 298 -- in 10,000 units
def sales_2023 := 850 -- in 10,000 units
def years := 2

-- Problem statement in Lean 4
theorem average_annual_growth_rate_eq :
  sales_2021 * (1 + x) ^ years = sales_2023 :=
sorry

end average_annual_growth_rate_eq_l225_22565


namespace moles_ethane_and_hexachloroethane_l225_22594

-- Define the conditions
def balanced_eq (a b c d : ℕ) : Prop :=
  a * 6 = b ∧ d * 6 = c

-- The main theorem statement
theorem moles_ethane_and_hexachloroethane (moles_Cl2 : ℕ) :
  moles_Cl2 = 18 → balanced_eq 1 1 18 3 :=
by
  sorry

end moles_ethane_and_hexachloroethane_l225_22594


namespace last_digit_base4_of_389_l225_22578

theorem last_digit_base4_of_389 : (389 % 4 = 1) :=
by sorry

end last_digit_base4_of_389_l225_22578


namespace total_weight_of_remaining_eggs_is_correct_l225_22595

-- Define the initial conditions and the question as Lean definitions
def total_eggs : Nat := 12
def weight_per_egg : Nat := 10
def num_boxes : Nat := 4
def melted_boxes : Nat := 1

-- Calculate the total weight of the eggs
def total_weight : Nat := total_eggs * weight_per_egg

-- Calculate the number of eggs per box
def eggs_per_box : Nat := total_eggs / num_boxes

-- Calculate the weight per box
def weight_per_box : Nat := eggs_per_box * weight_per_egg

-- Calculate the number of remaining boxes after one is tossed out
def remaining_boxes : Nat := num_boxes - melted_boxes

-- Calculate the total weight of the remaining chocolate eggs
def remaining_weight : Nat := remaining_boxes * weight_per_box

-- The proof task
theorem total_weight_of_remaining_eggs_is_correct : remaining_weight = 90 := by
  sorry

end total_weight_of_remaining_eggs_is_correct_l225_22595


namespace cos_identity_of_angle_l225_22538

open Real

theorem cos_identity_of_angle (α : ℝ) :
  sin (π / 6 + α) = sqrt 3 / 3 → cos (π / 3 - α) = sqrt 3 / 3 :=
by
  intro h
  sorry

end cos_identity_of_angle_l225_22538


namespace base_b_digits_l225_22516

theorem base_b_digits (b : ℕ) : b^4 ≤ 500 ∧ 500 < b^5 → b = 4 := by
  intro h
  sorry

end base_b_digits_l225_22516


namespace bottle_ratio_l225_22566

theorem bottle_ratio (C1 C2 : ℝ)  
  (h1 : (C1 / 2) + (C2 / 4) = (C1 + C2) / 3) :
  C2 = 2 * C1 :=
sorry

end bottle_ratio_l225_22566


namespace solve_inequality_1_solve_inequality_2_l225_22585

-- Definitions based on given conditions
noncomputable def f (x : ℝ) : ℝ := abs (x + 1)

-- Lean statement for the first proof problem
theorem solve_inequality_1 :
  ∀ x : ℝ, f x ≤ 5 - f (x - 3) ↔ -2 ≤ x ∧ x ≤ 3 :=
by
  sorry

-- Lean statement for the second proof problem
theorem solve_inequality_2 (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ 2 * f x + abs (x + a) ≤ x + 4) ↔ -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end solve_inequality_1_solve_inequality_2_l225_22585


namespace infinite_grid_coloring_l225_22571

theorem infinite_grid_coloring (color : ℕ × ℕ → Fin 4)
  (h_coloring_condition : ∀ (i j : ℕ), color (i, j) ≠ color (i + 1, j) ∧
                                      color (i, j) ≠ color (i, j + 1) ∧
                                      color (i, j) ≠ color (i + 1, j + 1) ∧
                                      color (i + 1, j) ≠ color (i, j + 1)) :
  ∃ m : ℕ, ∃ a b : Fin 4, ∀ n : ℕ, color (m, n) = a ∨ color (m, n) = b :=
sorry

end infinite_grid_coloring_l225_22571


namespace part_a_part_b_l225_22509

variable {α β γ δ AB CD : ℝ}
variable {A B C D : Point}
variable {A_obtuse B_obtuse : Prop}
variable {α_gt_δ β_gt_γ : Prop}

-- Definition of a convex quadrilateral
def convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Conditions for part (a)
axiom angle_A_obtuse : A_obtuse
axiom angle_B_obtuse : B_obtuse

-- Conditions for part (b)
axiom angle_α_gt_δ : α_gt_δ
axiom angle_β_gt_γ : β_gt_γ

-- Part (a) statement: Given angles A and B are obtuse, AB ≤ CD
theorem part_a {A B C D : Point} (h_convex : convex_quadrilateral A B C D) 
    (h_A_obtuse : A_obtuse) (h_B_obtuse : B_obtuse) : AB ≤ CD :=
sorry

-- Part (b) statement: Given angle A > angle D and angle B > angle C, AB < CD
theorem part_b {A B C D : Point} (h_convex : convex_quadrilateral A B C D) 
    (h_angle_α_gt_δ : α_gt_δ) (h_angle_β_gt_γ : β_gt_γ) : AB < CD :=
sorry

end part_a_part_b_l225_22509


namespace total_blossoms_l225_22560

theorem total_blossoms (first second third : ℕ) (h1 : first = 2) (h2 : second = 2 * first) (h3 : third = 4 * second) : first + second + third = 22 :=
by
  sorry

end total_blossoms_l225_22560


namespace range_of_a2_div_a1_l225_22577

theorem range_of_a2_div_a1 (a_1 a_2 d : ℤ) : 
  1 ≤ a_1 ∧ a_1 ≤ 3 ∧ 
  a_2 = a_1 + d ∧ 
  6 ≤ 3 * a_1 + 2 * d ∧ 
  3 * a_1 + 2 * d ≤ 15 
  → (2 / 3 : ℚ) ≤ (a_2 : ℚ) / a_1 ∧ (a_2 : ℚ) / a_1 ≤ 5 :=
sorry

end range_of_a2_div_a1_l225_22577


namespace luna_budget_l225_22576

variable {H F P : ℝ}

theorem luna_budget (h1: F = 0.60 * H) (h2: P = 0.10 * F) (h3: H + F + P = 249) :
  H + F = 240 :=
by
  -- The proof will be filled in here. For now, we use sorry.
  sorry

end luna_budget_l225_22576


namespace total_amount_shared_l225_22573

theorem total_amount_shared (x y z : ℝ) (h1 : x = 1.25 * y) (h2 : y = 1.2 * z) (h3 : z = 400) :
  x + y + z = 1480 :=
by
  sorry

end total_amount_shared_l225_22573


namespace determine_var_phi_l225_22522

open Real

theorem determine_var_phi (φ : ℝ) (h₀ : 0 ≤ φ ∧ φ ≤ 2 * π) :
  (∀ x, sin (x + φ) = sin (x - π / 6)) → φ = 11 * π / 6 :=
by
  sorry

end determine_var_phi_l225_22522


namespace even_function_zero_coefficient_l225_22588

theorem even_function_zero_coefficient: ∀ a : ℝ, (∀ x : ℝ, (x^2 + a * x + 1) = ((-x)^2 + a * (-x) + 1)) → a = 0 :=
by
  intros a h
  sorry

end even_function_zero_coefficient_l225_22588


namespace total_bill_is_correct_l225_22503

-- Given conditions
def hourly_rate := 45
def parts_cost := 225
def hours_worked := 5

-- Total bill calculation
def labor_cost := hourly_rate * hours_worked
def total_bill := labor_cost + parts_cost

-- Prove that the total bill is equal to 450 dollars
theorem total_bill_is_correct : total_bill = 450 := by
  sorry

end total_bill_is_correct_l225_22503


namespace probability_square_product_l225_22521

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_favorable_outcomes : ℕ :=
  List.length [(1, 1), (1, 4), (2, 2), (4, 1), (3, 3), (4, 4), (2, 8), (8, 2), (5, 5), (4, 9), (6, 6), (7, 7), (8, 8), (9, 9)]

def total_outcomes : ℕ := 12 * 8

theorem probability_square_product :
  (count_favorable_outcomes : ℚ) / (total_outcomes : ℚ) = (7 : ℚ) / (48 : ℚ) := 
by 
  sorry

end probability_square_product_l225_22521


namespace yoongi_stacked_higher_by_one_cm_l225_22525

def height_box_A : ℝ := 3
def height_box_B : ℝ := 3.5
def boxes_stacked_by_Taehyung : ℕ := 16
def boxes_stacked_by_Yoongi : ℕ := 14
def height_Taehyung_stack : ℝ := height_box_A * boxes_stacked_by_Taehyung
def height_Yoongi_stack : ℝ := height_box_B * boxes_stacked_by_Yoongi

theorem yoongi_stacked_higher_by_one_cm :
  height_Yoongi_stack = height_Taehyung_stack + 1 :=
by
  sorry

end yoongi_stacked_higher_by_one_cm_l225_22525


namespace triangle_condition_isosceles_or_right_l225_22575

theorem triangle_condition_isosceles_or_right {A B C : ℝ} {a b c : ℝ} 
  (h_triangle : A + B + C = π) (h_cos_eq : a * Real.cos A = b * Real.cos B) : 
  (A = B) ∨ (A + B = π / 2) :=
sorry

end triangle_condition_isosceles_or_right_l225_22575


namespace expression_incorrect_l225_22526

theorem expression_incorrect (x : ℝ) : 5 * (x + 7) ≠ 5 * x + 7 := 
by 
  sorry

end expression_incorrect_l225_22526


namespace part1_part2_l225_22544

-- Definitions for the sets A and B
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Proof statement for the first part
theorem part1 (m : ℝ) (h : m = 4) : A ∪ B m = { x | -2 ≤ x ∧ x ≤ 7 } :=
sorry

-- Proof statement for the second part
theorem part2 (h : ∀ {m : ℝ}, B m ⊆ A) : ∀ m : ℝ, m ∈ Set.Iic 3 :=
sorry

end part1_part2_l225_22544


namespace algebraic_expression_value_l225_22593

-- Define the conditions 
variables (x y : ℝ)
def condition1 : Prop := x + y = 2
def condition2 : Prop := x - y = 4

-- State the main theorem
theorem algebraic_expression_value (h1 : condition1 x y) (h2 : condition2 x y) :
  1 + x^2 - y^2 = 9 :=
sorry

end algebraic_expression_value_l225_22593


namespace find_x_in_list_l225_22598

theorem find_x_in_list :
  ∃ x : ℕ, x > 0 ∧ x ≤ 120 ∧ (45 + 76 + 110 + x + x) / 5 = 2 * x ∧ x = 29 :=
by
  sorry

end find_x_in_list_l225_22598


namespace total_cost_of_tennis_balls_l225_22548

theorem total_cost_of_tennis_balls
  (packs : ℕ) (balls_per_pack : ℕ) (cost_per_ball : ℕ)
  (h1 : packs = 4) (h2 : balls_per_pack = 3) (h3 : cost_per_ball = 2) : 
  packs * balls_per_pack * cost_per_ball = 24 := by
  sorry

end total_cost_of_tennis_balls_l225_22548


namespace work_done_in_11_days_l225_22511

-- Given conditions as definitions
def a_days := 24
def b_days := 30
def c_days := 40
def combined_work_rate := (1 / a_days) + (1 / b_days) + (1 / c_days)
def days_c_leaves_before_completion := 4

-- Statement of the problem to be proved
theorem work_done_in_11_days :
  ∃ (D : ℕ), D = 11 ∧ ((D - days_c_leaves_before_completion) * combined_work_rate) + 
  (days_c_leaves_before_completion * ((1 / a_days) + (1 / b_days))) = 1 :=
sorry

end work_done_in_11_days_l225_22511


namespace old_manufacturing_cost_l225_22552

theorem old_manufacturing_cost (P : ℝ) (h1 : 50 = 0.50 * P) : 0.60 * P = 60 :=
by
  sorry

end old_manufacturing_cost_l225_22552


namespace antonio_correct_answers_l225_22510

theorem antonio_correct_answers :
  ∃ c w : ℕ, c + w = 15 ∧ 6 * c - 3 * w = 36 ∧ c = 9 :=
by
  sorry

end antonio_correct_answers_l225_22510


namespace import_tax_amount_in_excess_l225_22532

theorem import_tax_amount_in_excess (X : ℝ) 
  (h1 : 0.07 * (2590 - X) = 111.30) : 
  X = 1000 :=
by
  sorry

end import_tax_amount_in_excess_l225_22532


namespace george_score_l225_22519

theorem george_score (avg_without_george avg_with_george : ℕ) (num_students : ℕ) 
(h1 : avg_without_george = 75) (h2 : avg_with_george = 76) (h3 : num_students = 20) :
  (num_students * avg_with_george) - ((num_students - 1) * avg_without_george) = 95 :=
by 
  sorry

end george_score_l225_22519


namespace tangerines_count_l225_22501

theorem tangerines_count (apples pears tangerines : ℕ)
  (h1 : apples = 45)
  (h2 : pears = apples - 21)
  (h3 : tangerines = pears + 18) :
  tangerines = 42 :=
by
  sorry

end tangerines_count_l225_22501


namespace A_alone_time_l225_22583

theorem A_alone_time (x : ℕ) (h1 : 3 * x / 4  = 12) : x / 3 = 16 := by
  sorry

end A_alone_time_l225_22583


namespace smallest_positive_integer_linear_combination_l225_22524

theorem smallest_positive_integer_linear_combination : ∃ m n : ℤ, 3003 * m + 55555 * n = 1 :=
by
  sorry

end smallest_positive_integer_linear_combination_l225_22524


namespace find_time_when_velocity_is_one_l225_22564

-- Define the equation of motion
def equation_of_motion (t : ℝ) : ℝ := 7 * t^2 + 8

-- Define the velocity function as the derivative of the equation of motion
def velocity (t : ℝ) : ℝ := by
  let s := equation_of_motion t
  exact 14 * t  -- Since we calculated the derivative above

-- Statement of the problem to be proved
theorem find_time_when_velocity_is_one : 
  (velocity (1 / 14)) = 1 :=
by
  -- Placeholder for the proof
  sorry

end find_time_when_velocity_is_one_l225_22564


namespace solve_equation_l225_22597

theorem solve_equation (x : ℝ) : (⌊Real.sin x⌋:ℝ)^2 = Real.cos x ^ 2 - 1 ↔ ∃ n : ℤ, x = n * Real.pi := by
  sorry

end solve_equation_l225_22597


namespace ratio_evaluation_l225_22513

theorem ratio_evaluation :
  (10 ^ 2003 + 10 ^ 2001) / (2 * 10 ^ 2002) = 101 / 20 := 
by sorry

end ratio_evaluation_l225_22513


namespace lines_intersect_lines_parallel_lines_coincident_l225_22517

-- Define line equations
def l1 (m x y : ℝ) := (m + 2) * x + (m + 3) * y - 5 = 0
def l2 (m x y : ℝ) := 6 * x + (2 * m - 1) * y - 5 = 0

-- Prove conditions for intersection
theorem lines_intersect (m : ℝ) : ¬(m = -5 / 2 ∨ m = 4) ↔
  ∃ x y : ℝ, l1 m x y ∧ l2 m x y := sorry

-- Prove conditions for parallel lines
theorem lines_parallel (m : ℝ) : m = -5 / 2 ↔
  ∀ x y : ℝ, l1 m x y ∧ l2 m x y → l1 m x y → l2 m x y := sorry

-- Prove conditions for coincident lines
theorem lines_coincident (m : ℝ) : m = 4 ↔
  ∀ x y : ℝ, l1 m x y ↔ l2 m x y := sorry

end lines_intersect_lines_parallel_lines_coincident_l225_22517


namespace football_team_total_players_l225_22551

/-- Let's denote the total number of players on the football team as P.
    We know that there are 31 throwers, and all of them are right-handed.
    The rest of the team is divided so one third are left-handed and the rest are right-handed.
    There are a total of 57 right-handed players on the team.
    Prove that the total number of players on the football team is 70. -/
theorem football_team_total_players 
  (P : ℕ) -- total number of players
  (T : ℕ := 31) -- number of throwers
  (L : ℕ) -- number of left-handed players
  (R : ℕ := 57) -- total number of right-handed players
  (H_all_throwers_rhs: ∀ x : ℕ, (x < P) → (x < T) → (x = T → x < R)) -- all throwers are right-handed
  (H_rest_division: ∀ x : ℕ, (x < P - T) → (x = L) → (x = 2 * L))
  : P = 70 :=
  sorry

end football_team_total_players_l225_22551


namespace find_theta_l225_22570

theorem find_theta (Theta : ℕ) (h1 : 1 ≤ Theta ∧ Theta ≤ 9)
  (h2 : 294 / Theta = (30 + Theta) + 3 * Theta) : Theta = 6 :=
by sorry

end find_theta_l225_22570


namespace greg_distance_work_to_market_l225_22506

-- Given conditions translated into definitions
def total_distance : ℝ := 40
def time_from_market_to_home : ℝ := 0.5  -- in hours
def speed_from_market_to_home : ℝ := 20  -- in miles per hour

-- Distance calculation from farmer's market to home
def distance_from_market_to_home := speed_from_market_to_home * time_from_market_to_home

-- Definition for the distance from workplace to the farmer's market
def distance_from_work_to_market := total_distance - distance_from_market_to_home

-- The theorem to be proved
theorem greg_distance_work_to_market : distance_from_work_to_market = 30 := by
  -- Skipping the detailed proof
  sorry

end greg_distance_work_to_market_l225_22506


namespace initial_bales_l225_22502

theorem initial_bales (B : ℕ) (cond1 : B + 35 = 82) : B = 47 :=
by
  sorry

end initial_bales_l225_22502


namespace forty_percent_of_number_l225_22557

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 17) : 0.4 * N = 204 :=
sorry

end forty_percent_of_number_l225_22557


namespace smallest_n_is_1770_l225_22591

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

def is_smallest_n (n : ℕ) : Prop :=
  n = sum_of_digits n + 1755 ∧ (∀ m : ℕ, (m < n → m ≠ sum_of_digits m + 1755))

theorem smallest_n_is_1770 : is_smallest_n 1770 :=
sorry

end smallest_n_is_1770_l225_22591


namespace pants_cost_l225_22535

def total_cost (P : ℕ) : ℕ := 4 * 8 + 2 * 60 + 2 * P

theorem pants_cost :
  (∃ P : ℕ, total_cost P = 188) →
  ∃ P : ℕ, P = 18 :=
by
  intro h
  sorry

end pants_cost_l225_22535


namespace effective_annual_interest_rate_is_correct_l225_22534

noncomputable def quarterly_interest_rate : ℝ := 0.02

noncomputable def annual_interest_rate (quarterly_rate : ℝ) : ℝ :=
  ((1 + quarterly_rate) ^ 4 - 1) * 100

theorem effective_annual_interest_rate_is_correct :
  annual_interest_rate quarterly_interest_rate = 8.24 :=
by
  sorry

end effective_annual_interest_rate_is_correct_l225_22534


namespace range_of_a_l225_22563

-- Define the set A
def A (a x : ℝ) := 6 * x + a > 0

-- Theorem stating the range of a given the conditions
theorem range_of_a (a : ℝ) (h : ¬ A a 1) : a ≤ -6 :=
by
  -- Here we would provide the proof
  sorry

end range_of_a_l225_22563


namespace necessary_condition_for_abs_ab_l225_22512

theorem necessary_condition_for_abs_ab {a b : ℝ} (h : |a - b| = |a| - |b|) : ab ≥ 0 :=
sorry

end necessary_condition_for_abs_ab_l225_22512


namespace asymptotes_of_hyperbola_l225_22507

-- Definitions
variables (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)

-- Theorem: Equation of the asymptotes of the given hyperbola
theorem asymptotes_of_hyperbola (h_equiv : b = 2 * a) :
  ∀ x y : ℝ, 
    (x ≠ 0 ∧ y ≠ 0 ∧ (y = (2 : ℝ) * x ∨ y = - (2 : ℝ) * x)) ↔ (x, y) ∈ {p : ℝ × ℝ | (x^2 / a^2) - (y^2 / b^2) = 1} := 
sorry

end asymptotes_of_hyperbola_l225_22507


namespace expand_expression_l225_22568

theorem expand_expression (x : ℝ) : 
  (x - 3) * (x + 3) * (x^2 + 5) = x^4 - 4 * x^2 - 45 := 
by
  sorry

end expand_expression_l225_22568


namespace algebraic_expression_domain_l225_22547

theorem algebraic_expression_domain (x : ℝ) : (∃ y : ℝ, y = 1 / (x + 2)) ↔ (x ≠ -2) := 
sorry

end algebraic_expression_domain_l225_22547


namespace inequality_proof_l225_22556

open Real

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h1 : a ^ x = b * c) 
  (h2 : b ^ y = c * a) 
  (h3 : c ^ z = a * b) :
  (1 / (2 + x) + 1 / (2 + y) + 1 / (2 + z)) ≤ 3 / 4 := 
sorry

end inequality_proof_l225_22556


namespace find_sum_of_a_and_b_l225_22518

variable (a b w y z S : ℕ)

-- Conditions based on problem statement
axiom condition1 : 19 + w + 23 = S
axiom condition2 : 22 + y + a = S
axiom condition3 : b + 18 + z = S
axiom condition4 : 19 + 22 + b = S
axiom condition5 : w + y + 18 = S
axiom condition6 : 23 + a + z = S
axiom condition7 : 19 + y + z = S
axiom condition8 : 23 + y + b = S

theorem find_sum_of_a_and_b : a + b = 23 :=
by
  sorry  -- To be provided with the actual proof later

end find_sum_of_a_and_b_l225_22518


namespace exists_numbers_with_prime_sum_and_product_l225_22579

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem exists_numbers_with_prime_sum_and_product :
  ∃ a b c : ℕ, is_prime (a + b + c) ∧ is_prime (a * b * c) :=
  by
    -- First import the prime definitions and variables.
    let a := 1
    let b := 1
    let c := 3
    have h1 : is_prime (a + b + c) := by sorry
    have h2 : is_prime (a * b * c) := by sorry
    exact ⟨a, b, c, h1, h2⟩

end exists_numbers_with_prime_sum_and_product_l225_22579


namespace fixed_point_of_exponential_function_l225_22559

theorem fixed_point_of_exponential_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : 
  ∃ p : ℝ × ℝ, p = (-2, 1) ∧ ∀ x : ℝ, (x, a^(x + 2)) = p → x = -2 ∧ a^(x + 2) = 1 :=
by
  sorry

end fixed_point_of_exponential_function_l225_22559


namespace original_cost_l225_22586

theorem original_cost (original_cost : ℝ) (h : 0.30 * original_cost = 588) : original_cost = 1960 :=
sorry

end original_cost_l225_22586


namespace min_b_geometric_sequence_l225_22562

theorem min_b_geometric_sequence (a b c : ℝ) (h_geom : b^2 = a * c) (h_1_4 : (a = 1 ∨ b = 1 ∨ c = 1) ∧ (a = 4 ∨ b = 4 ∨ c = 4)) :
  b ≥ -2 ∧ (∃ b', b' < b → b' ≥ -2) :=
by {
  sorry -- Proof required
}

end min_b_geometric_sequence_l225_22562


namespace find_a4_plus_a6_l225_22554

variable {a : ℕ → ℝ}

-- Geometric sequence definition
def is_geometric_seq (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Conditions for the problem
axiom seq_geometric : is_geometric_seq a
axiom seq_positive : ∀ n : ℕ, n > 0 → a n > 0
axiom given_equation : a 3 * a 5 + 2 * a 4 * a 6 + a 5 * a 7 = 81

-- The problem to prove
theorem find_a4_plus_a6 : a 4 + a 6 = 9 :=
sorry

end find_a4_plus_a6_l225_22554


namespace expression_value_at_neg3_l225_22542

theorem expression_value_at_neg3 (p q : ℤ) (h : 27 * p + 3 * q = 14) :
  (p * (-3)^3 + q * (-3) - 1) = -15 :=
sorry

end expression_value_at_neg3_l225_22542


namespace months_rent_in_advance_required_l225_22528

def janet_savings : ℕ := 2225
def rent_per_month : ℕ := 1250
def deposit : ℕ := 500
def additional_needed : ℕ := 775

theorem months_rent_in_advance_required : 
  (janet_savings + additional_needed - deposit) / rent_per_month = 2 :=
by
  sorry

end months_rent_in_advance_required_l225_22528


namespace fractions_equal_l225_22569

theorem fractions_equal (x y z : ℝ) (hx1 : x ≠ 1) (hy1 : y ≠ 1) (hxy : x ≠ y)
  (h : (yz - x^2) / (1 - x) = (xz - y^2) / (1 - y)) : (yz - x^2) / (1 - x) = x + y + z ∧ (xz - y^2) / (1 - y) = x + y + z :=
sorry

end fractions_equal_l225_22569


namespace election_result_l225_22500

def votes_A : ℕ := 12
def votes_B : ℕ := 3
def votes_C : ℕ := 15

def is_class_president (candidate_votes : ℕ) : Prop :=
  candidate_votes = max (max votes_A votes_B) votes_C

theorem election_result : is_class_president votes_C :=
by
  unfold is_class_president
  rw [votes_A, votes_B, votes_C]
  sorry

end election_result_l225_22500


namespace total_value_of_pile_l225_22550

def value_of_pile (total_coins dimes : ℕ) (value_dime value_nickel : ℝ) : ℝ :=
  let nickels := total_coins - dimes
  let value_dimes := dimes * value_dime
  let value_nickels := nickels * value_nickel
  value_dimes + value_nickels

theorem total_value_of_pile :
  value_of_pile 50 14 0.10 0.05 = 3.20 := by
  sorry

end total_value_of_pile_l225_22550


namespace buckets_required_l225_22580

theorem buckets_required (C : ℕ) (h : C > 0) : 
  let original_buckets := 25
  let reduced_capacity := 2 / 5
  let total_capacity := original_buckets * C
  let new_buckets := total_capacity / ((2 / 5) * C)
  new_buckets = 63 := 
by
  sorry

end buckets_required_l225_22580


namespace three_star_five_l225_22574

-- Definitions based on conditions
def star (a b : ℕ) : ℕ := 2 * a^2 + 3 * a * b + 2 * b^2

-- Theorem statement to be proved
theorem three_star_five : star 3 5 = 113 := by
  sorry

end three_star_five_l225_22574


namespace sqrt_72_eq_6_sqrt_2_l225_22536

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := 
by
  sorry

end sqrt_72_eq_6_sqrt_2_l225_22536


namespace find_may_monday_l225_22549

noncomputable def weekday (day_of_month : ℕ) (first_day_weekday : ℕ) : ℕ :=
(day_of_month + first_day_weekday - 1) % 7

theorem find_may_monday (r n : ℕ) (condition1 : weekday r 5 = 5) (condition2 : weekday n 5 = 1) (condition3 : 15 < n ∧ n < 25) : 
  n = 20 :=
by
  -- Proof omitted.
  sorry

end find_may_monday_l225_22549


namespace tim_movie_marathon_duration_l225_22587

-- Define the durations of each movie
def first_movie_duration : ℕ := 2

def second_movie_duration : ℕ := 
  first_movie_duration + (first_movie_duration / 2)

def combined_first_two_movies_duration : ℕ :=
  first_movie_duration + second_movie_duration

def last_movie_duration : ℕ := 
  combined_first_two_movies_duration - 1

-- Define the total movie marathon duration
def total_movie_marathon_duration : ℕ := 
  first_movie_duration + second_movie_duration + last_movie_duration

-- Problem statement to be proved
theorem tim_movie_marathon_duration : total_movie_marathon_duration = 9 := by
  sorry

end tim_movie_marathon_duration_l225_22587


namespace combined_time_is_45_l225_22599

-- Definitions based on conditions
def Pulsar_time : ℕ := 10
def Polly_time : ℕ := 3 * Pulsar_time
def Petra_time : ℕ := (1 / 6 ) * Polly_time

-- Total combined time
def total_time : ℕ := Pulsar_time + Polly_time + Petra_time

-- Theorem to prove
theorem combined_time_is_45 : total_time = 45 := by
  sorry

end combined_time_is_45_l225_22599


namespace possible_values_for_a_l225_22508

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) - x - 1

theorem possible_values_for_a (a : ℝ) (h: a ≠ 0) : 
  (∀ x : ℝ, f a x ≥ 0) ↔ a = 1 :=
by
  sorry

end possible_values_for_a_l225_22508


namespace serum_prevents_colds_l225_22543

noncomputable def hypothesis_preventive_effect (H : Prop) : Prop :=
  let K2 := 3.918
  let critical_value := 3.841
  let P_threshold := 0.05
  K2 >= critical_value ∧ P_threshold = 0.05 → H

theorem serum_prevents_colds (H : Prop) : hypothesis_preventive_effect H → H :=
by
  -- Proof will be added here
  sorry

end serum_prevents_colds_l225_22543


namespace height_of_carton_is_70_l225_22545

def carton_dimensions : ℕ × ℕ := (25, 42)
def soap_box_dimensions : ℕ × ℕ × ℕ := (7, 6, 5)
def max_soap_boxes : ℕ := 300

theorem height_of_carton_is_70 :
  let (carton_length, carton_width) := carton_dimensions
  let (soap_box_length, soap_box_width, soap_box_height) := soap_box_dimensions
  let boxes_per_layer := (carton_length / soap_box_length) * (carton_width / soap_box_width)
  let num_layers := max_soap_boxes / boxes_per_layer
  (num_layers * soap_box_height) = 70 :=
by
  have carton_length := 25
  have carton_width := 42
  have soap_box_length := 7
  have soap_box_width := 6
  have soap_box_height := 5
  have max_soap_boxes := 300
  have boxes_per_layer := (25 / 7) * (42 / 6)
  have num_layers := max_soap_boxes / boxes_per_layer
  sorry

end height_of_carton_is_70_l225_22545


namespace algae_cells_count_10_days_l225_22505

-- Define the initial condition where the pond starts with one algae cell.
def initial_algae_cells : ℕ := 1

-- Define the daily splitting of each cell into 3 new cells.
def daily_split (cells : ℕ) : ℕ := cells * 3

-- Define the function to compute the number of algae cells after n days.
def algae_cells_after_days (n : ℕ) : ℕ :=
  initial_algae_cells * (3 ^ n)

-- State the theorem to be proved.
theorem algae_cells_count_10_days : algae_cells_after_days 10 = 59049 :=
by {
  sorry
}

end algae_cells_count_10_days_l225_22505


namespace ratio_karen_beatrice_l225_22537

noncomputable def karen_crayons : ℕ := 128
noncomputable def judah_crayons : ℕ := 8
noncomputable def gilbert_crayons : ℕ := 4 * judah_crayons
noncomputable def beatrice_crayons : ℕ := 2 * gilbert_crayons

theorem ratio_karen_beatrice :
  karen_crayons / beatrice_crayons = 2 := by
sorry

end ratio_karen_beatrice_l225_22537


namespace pq_square_sum_l225_22504

theorem pq_square_sum (p q : ℝ) (h1 : p * q = 9) (h2 : p + q = 6) : p^2 + q^2 = 18 := 
by
  sorry

end pq_square_sum_l225_22504


namespace squirrel_walnuts_l225_22592

theorem squirrel_walnuts :
  let boy_gathered := 6
  let boy_dropped := 1
  let initial_in_burrow := 12
  let girl_brought := 5
  let girl_ate := 2
  initial_in_burrow + (boy_gathered - boy_dropped) + girl_brought - girl_ate = 20 :=
by
  sorry

end squirrel_walnuts_l225_22592


namespace total_weight_is_40_l225_22529

def marco_strawberries_weight : ℕ := 8
def dad_strawberries_weight : ℕ := 32
def total_strawberries_weight := marco_strawberries_weight + dad_strawberries_weight

theorem total_weight_is_40 : total_strawberries_weight = 40 := by
  sorry

end total_weight_is_40_l225_22529


namespace element_in_set_l225_22558

variable (A : Set ℕ) (a b : ℕ)
def condition : Prop := A = {a, b, 1}

theorem element_in_set (h : condition A a b) : 1 ∈ A :=
by sorry

end element_in_set_l225_22558


namespace third_student_gold_stickers_l225_22520

theorem third_student_gold_stickers:
  ∃ (n : ℕ), n = 41 ∧ 
  (∃ (a1 a2 a4 a5 a6 : ℕ), 
    a1 = 29 ∧ 
    a2 = 35 ∧ 
    a4 = 47 ∧ 
    a5 = 53 ∧ 
    a6 = 59 ∧ 
    a2 - a1 = 6 ∧ 
    a5 - a4 = 6 ∧ 
    ∀ k, k = 3 → n = a2 + 6) := 
sorry

end third_student_gold_stickers_l225_22520


namespace bill_due_in_months_l225_22561

noncomputable def true_discount_time (TD A R : ℝ) : ℝ :=
  let P := A - TD
  let T := TD / (P * R / 100)
  12 * T

theorem bill_due_in_months :
  ∀ (TD A R : ℝ), TD = 189 → A = 1764 → R = 16 →
  abs (true_discount_time TD A R - 10.224) < 1 :=
by
  intros TD A R hTD hA hR
  sorry

end bill_due_in_months_l225_22561


namespace min_value_square_distance_l225_22590

theorem min_value_square_distance (x y : ℝ) (h : x^2 + y^2 - 4*x + 2 = 0) : 
  ∃ c, (∀ x y : ℝ, x^2 + y^2 - 4*x + 2 = 0 → x^2 + (y - 2)^2 ≥ c) ∧ c = 2 :=
sorry

end min_value_square_distance_l225_22590


namespace find_ABC_l225_22584

theorem find_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
  (hA : A < 5) (hB : B < 5) (hC : C < 5) (h_nonzeroA : A ≠ 0) (h_nonzeroB : B ≠ 0) (h_nonzeroC : C ≠ 0)
  (h4 : B + C = 5) (h5 : A + 1 = C) (h6 : A + B = C) : A = 3 ∧ B = 1 ∧ C = 4 := 
by
  sorry

end find_ABC_l225_22584


namespace factor_expression_l225_22539

theorem factor_expression (
  x y z : ℝ
) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3) = 
  (x + y) * (y + z) * (z + x) := 
sorry

end factor_expression_l225_22539


namespace tara_marbles_modulo_l225_22572

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem tara_marbles_modulo : 
  let B := 6  -- number of blue marbles
  let Y := 12  -- additional yellow marbles for balance
  let total_marbles := B + Y  -- total marbles
  let arrangements := binom (total_marbles + B) B
  let N := arrangements
  N % 1000 = 564 :=
by
  let B := 6  -- number of blue marbles
  let Y := 12  -- additional yellow marbles for balance
  let total_marbles := B + Y  -- total marbles
  let arrangements := binom (total_marbles + B) B
  let N := arrangements
  have : N % 1000 = 564 := sorry
  exact this

end tara_marbles_modulo_l225_22572


namespace find_value_of_p_l225_22530

-- Definition of the parabola and ellipse
def parabola (p : ℝ) : Set (ℝ × ℝ) := {xy | xy.1 ^ 2 = 2 * p * xy.2}
def ellipse : Set (ℝ × ℝ) := {xy | xy.1 ^ 2 / 6 + xy.2 ^ 2 / 4 = 1}

-- Hypotheses
variables (p : ℝ) (h_pos : p > 0)

-- Latus rectum tangent to the ellipse
theorem find_value_of_p (h_tangent : ∃ (x y : ℝ),
  (parabola p (x, y) ∧ ellipse (x, y) ∧ y = -p / 2)) : p = 4 := sorry

end find_value_of_p_l225_22530


namespace quadratic_linear_common_solution_l225_22555

theorem quadratic_linear_common_solution
  (a x1 x2 d e : ℝ)
  (ha : a ≠ 0) (hx1x2 : x1 ≠ x2) (hd : d ≠ 0)
  (h_quad : ∀ x, a * (x - x1) * (x - x2) = 0 → x = x1 ∨ x = x2)
  (h_linear : d * x1 + e = 0)
  (h_combined : ∀ x, a * (x - x1) * (x - x2) + d * x + e = 0 → x = x1) :
  d = a * (x2 - x1) :=
by sorry

end quadratic_linear_common_solution_l225_22555


namespace winnieKeepsBalloons_l225_22546

-- Given conditions
def redBalloons : Nat := 24
def whiteBalloons : Nat := 39
def greenBalloons : Nat := 72
def chartreuseBalloons : Nat := 91
def totalFriends : Nat := 11

-- Total balloons
def totalBalloons : Nat := redBalloons + whiteBalloons + greenBalloons + chartreuseBalloons

-- Theorem: Prove the number of balloons Winnie keeps for herself
theorem winnieKeepsBalloons :
  totalBalloons % totalFriends = 6 :=
by
  -- Placeholder for the proof
  sorry

end winnieKeepsBalloons_l225_22546


namespace units_digit_6_pow_4_l225_22541

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Define the main theorem to prove
theorem units_digit_6_pow_4 : units_digit (6 ^ 4) = 6 := 
by
  sorry

end units_digit_6_pow_4_l225_22541


namespace sum_of_series_l225_22553

theorem sum_of_series : 
  (3 + 13 + 23 + 33 + 43) + (11 + 21 + 31 + 41 + 51) = 270 := by
  sorry

end sum_of_series_l225_22553


namespace bryden_amount_correct_l225_22533

-- Each state quarter has a face value of $0.25.
def face_value (q : ℕ) : ℝ := 0.25 * q

-- The collector offers to buy the state quarters for 1500% of their face value.
def collector_multiplier : ℝ := 15

-- Bryden has 10 state quarters.
def bryden_quarters : ℕ := 10

-- Calculate the amount Bryden will get for his 10 state quarters.
def amount_received : ℝ := collector_multiplier * face_value bryden_quarters

-- Prove that the amount received by Bryden equals $37.5.
theorem bryden_amount_correct : amount_received = 37.5 :=
by
  sorry

end bryden_amount_correct_l225_22533


namespace inequality_solution_set_l225_22531

theorem inequality_solution_set (a : ℤ) : 
  (∀ x : ℤ, (1 + a) * x > 1 + a → x < 1) → a < -1 :=
sorry

end inequality_solution_set_l225_22531


namespace billy_win_probability_l225_22596

-- Definitions of states and transition probabilities
def alice_step_prob_pos : ℚ := 1 / 2
def alice_step_prob_neg : ℚ := 1 / 2
def billy_step_prob_pos : ℚ := 2 / 3
def billy_step_prob_neg : ℚ := 1 / 3

-- Definitions of states in the Markov chain
inductive State
| S0 | S1 | Sm1 | S2 | Sm2 -- Alice's states
| T0 | T1 | Tm1 | T2 | Tm2 -- Billy's states

open State

-- The theorem statement: the probability that Billy wins the game
theorem billy_win_probability : 
  ∃ (P : State → ℚ), 
  P S0 = 11 / 19 ∧ P T0 = 14 / 19 ∧ 
  P S1 = 1 / 2 * P T0 ∧
  P Sm1 = 1 / 2 * P S0 + 1 / 2 ∧
  P T0 = 2 / 3 * P T1 + 1 / 3 * P Tm1 ∧
  P T1 = 2 / 3 + 1 / 3 * P S0 ∧
  P Tm1 = 2 / 3 * P T0 ∧
  P S2 = 0 ∧ P Sm2 = 1 ∧ P T2 = 1 ∧ P Tm2 = 0 := 
by 
  sorry

end billy_win_probability_l225_22596


namespace no_three_distinct_rational_roots_l225_22582

theorem no_three_distinct_rational_roots (a b : ℝ) : 
  ¬ ∃ (u v w : ℚ), 
    u + v + w = -(2 * a + 1) ∧ 
    u * v + v * w + w * u = (2 * a^2 + 2 * a - 3) ∧ 
    u * v * w = b := sorry

end no_three_distinct_rational_roots_l225_22582


namespace infinite_natural_numbers_with_factored_polynomial_l225_22515

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end infinite_natural_numbers_with_factored_polynomial_l225_22515
