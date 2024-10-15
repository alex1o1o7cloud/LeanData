import Mathlib

namespace NUMINAMATH_GPT_negation_proposition_l2115_211543

theorem negation_proposition :
  (¬ (∀ x : ℝ, ∃ n : ℕ, n > 0 ∧ n ≥ x)) ↔ (∃ x : ℝ, ∀ n : ℕ, n > 0 → n < x^2) := 
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l2115_211543


namespace NUMINAMATH_GPT_problem_l2115_211532

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem problem :
  let A := 3.14159265
  let B := Real.sqrt 36
  let C := Real.sqrt 7
  let D := 4.1
  is_irrational C := by
  sorry

end NUMINAMATH_GPT_problem_l2115_211532


namespace NUMINAMATH_GPT_range_of_b_over_a_l2115_211514

noncomputable def f (a b x : ℝ) : ℝ := (x - a)^3 * (x - b)
noncomputable def g_k (a b k x : ℝ) : ℝ := (f a b x - f a b k) / (x - k)

theorem range_of_b_over_a (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < 1)
    (hk_inc : ∀ k : ℤ, ∀ x : ℝ, k < x → g_k a b k x ≥ g_k a b k (k + 1)) :
  1 < b / a ∧ b / a ≤ 3 :=
by
  sorry


end NUMINAMATH_GPT_range_of_b_over_a_l2115_211514


namespace NUMINAMATH_GPT_absolute_value_equation_sum_l2115_211571

theorem absolute_value_equation_sum (x1 x2 : ℝ) (h1 : 3 * x1 - 12 = 6) (h2 : 3 * x2 - 12 = -6) : x1 + x2 = 8 := 
sorry

end NUMINAMATH_GPT_absolute_value_equation_sum_l2115_211571


namespace NUMINAMATH_GPT_min_value_reciprocal_l2115_211511

theorem min_value_reciprocal (m n : ℝ) (hmn_gt : 0 < m * n) (hmn_add : m + n = 2) :
  (∃ x : ℝ, x = (1/m + 1/n) ∧ x = 2) :=
by sorry

end NUMINAMATH_GPT_min_value_reciprocal_l2115_211511


namespace NUMINAMATH_GPT_show_linear_l2115_211538

-- Define the conditions as given in the problem
variables (a b : ℤ)

-- The hypothesis that the equation is linear
def linear_equation_hypothesis : Prop :=
  (a + b = 1) ∧ (3 * a + 2 * b - 4 = 1)

-- Define the theorem we need to prove
theorem show_linear (h : linear_equation_hypothesis a b) : a + b = 1 := 
by
  sorry

end NUMINAMATH_GPT_show_linear_l2115_211538


namespace NUMINAMATH_GPT_op_identity_l2115_211530

-- Define the operation ⊕ as given by the table
def op (x y : ℕ) : ℕ :=
  match (x, y) with
  | (1, 1) => 4
  | (1, 2) => 1
  | (1, 3) => 2
  | (1, 4) => 3
  | (2, 1) => 1
  | (2, 2) => 3
  | (2, 3) => 4
  | (2, 4) => 2
  | (3, 1) => 2
  | (3, 2) => 4
  | (3, 3) => 1
  | (3, 4) => 3
  | (4, 1) => 3
  | (4, 2) => 2
  | (4, 3) => 3
  | (4, 4) => 4
  | _ => 0  -- default case for completeness

-- State the theorem
theorem op_identity : op (op 4 1) (op 2 3) = 3 := by
  sorry

end NUMINAMATH_GPT_op_identity_l2115_211530


namespace NUMINAMATH_GPT_prob_sum_24_four_dice_l2115_211546

-- The probability of each die landing on six
def prob_die_six : ℚ := 1 / 6

-- The probability of all four dice showing six
theorem prob_sum_24_four_dice : 
  prob_die_six ^ 4 = 1 / 1296 :=
by
  -- Equivalent Lean statement asserting the probability problem
  sorry

end NUMINAMATH_GPT_prob_sum_24_four_dice_l2115_211546


namespace NUMINAMATH_GPT_filled_sandbag_weight_is_correct_l2115_211561

-- Define the conditions
def sandbag_weight : ℝ := 250
def fill_percent : ℝ := 0.80
def heavier_factor : ℝ := 1.40

-- Define the intermediate weights
def sand_weight : ℝ := sandbag_weight * fill_percent
def extra_weight : ℝ := sand_weight * (heavier_factor - 1)
def filled_material_weight : ℝ := sand_weight + extra_weight

-- Define the total weight including the empty sandbag
def total_weight : ℝ := sandbag_weight + filled_material_weight

-- Prove the total weight is correct
theorem filled_sandbag_weight_is_correct : total_weight = 530 := 
by sorry

end NUMINAMATH_GPT_filled_sandbag_weight_is_correct_l2115_211561


namespace NUMINAMATH_GPT_cricket_runs_product_l2115_211575

theorem cricket_runs_product :
  let runs_first_10 := [11, 6, 7, 5, 12, 8, 3, 10, 9, 4]
  let total_runs_first_10 := runs_first_10.sum
  let total_runs := total_runs_first_10 + 2 + 7
  2 < 15 ∧ 7 < 15 ∧ (total_runs_first_10 + 2) % 11 = 0 ∧ (total_runs_first_10 + 2 + 7) % 12 = 0 →
  (2 * 7) = 14 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_cricket_runs_product_l2115_211575


namespace NUMINAMATH_GPT_verify_ages_l2115_211551

noncomputable def correct_ages (S M D W : ℝ) : Prop :=
  (M = S + 29) ∧
  (M + 2 = 2 * (S + 2)) ∧
  (D = S - 3.5) ∧
  (W = 1.5 * D) ∧
  (S = 27) ∧
  (M = 56) ∧
  (D = 23.5) ∧
  (W = 35.25)

theorem verify_ages : ∃ (S M D W : ℝ), correct_ages S M D W :=
by
  sorry

end NUMINAMATH_GPT_verify_ages_l2115_211551


namespace NUMINAMATH_GPT_probability_of_color_change_is_1_over_6_l2115_211556

noncomputable def watchColorChangeProbability : ℚ :=
  let cycleDuration := 45 + 5 + 40
  let favorableDuration := 5 + 5 + 5
  favorableDuration / cycleDuration

theorem probability_of_color_change_is_1_over_6 :
  watchColorChangeProbability = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_color_change_is_1_over_6_l2115_211556


namespace NUMINAMATH_GPT_distinct_solutions_diff_l2115_211566

theorem distinct_solutions_diff (r s : ℝ) (hr : r > s) 
  (h : ∀ x, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
sorry

end NUMINAMATH_GPT_distinct_solutions_diff_l2115_211566


namespace NUMINAMATH_GPT_equalize_expenses_l2115_211567

def total_expenses := 130 + 160 + 150 + 180
def per_person_share := total_expenses / 4
def tom_owes := per_person_share - 130
def dorothy_owes := per_person_share - 160
def sammy_owes := per_person_share - 150
def alice_owes := per_person_share - 180
def t := tom_owes
def d := dorothy_owes

theorem equalize_expenses : t - dorothy_owes = 30 := by
  sorry

end NUMINAMATH_GPT_equalize_expenses_l2115_211567


namespace NUMINAMATH_GPT_area_of_gray_region_l2115_211549

theorem area_of_gray_region (r : ℝ) (h1 : r * 3 - r = 3) : 
  π * (3 * r) ^ 2 - π * r ^ 2 = 18 * π :=
by
  sorry

end NUMINAMATH_GPT_area_of_gray_region_l2115_211549


namespace NUMINAMATH_GPT_find_a_l2115_211533

theorem find_a (P : ℝ) (hP : P ≠ 0) (S : ℕ → ℝ) (a_n : ℕ → ℝ)
  (hSn : ∀ n, S n = 3^n + a)
  (ha_n : ∀ n, a_n (n + 1) = P * a_n n)
  (hS1 : S 1 = a_n 1)
  (hS2 : S 2 = S 1 + a_n 2 - a_n 1)
  (hS3 : S 3 = S 2 + a_n 3 - a_n 2) :
  a = -1 := sorry

end NUMINAMATH_GPT_find_a_l2115_211533


namespace NUMINAMATH_GPT_period_2_students_l2115_211591

theorem period_2_students (x : ℕ) (h1 : 2 * x - 5 = 11) : x = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_period_2_students_l2115_211591


namespace NUMINAMATH_GPT_min_balls_for_color_15_l2115_211547

theorem min_balls_for_color_15
  (red green yellow blue white black : ℕ)
  (h_red : red = 28)
  (h_green : green = 20)
  (h_yellow : yellow = 19)
  (h_blue : blue = 13)
  (h_white : white = 11)
  (h_black : black = 9) :
  ∃ n, n = 76 ∧ ∀ balls_drawn, balls_drawn = n →
  ∃ color, 
    (color = "red" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= red) ∨
    (color = "green" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= green) ∨
    (color = "yellow" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= yellow) ∨
    (color = "blue" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= blue) ∨
    (color = "white" ∧ balls_drawn >= 15 ∧ balls_drawn <= white) ∨
    (color = "black" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= black) := 
sorry

end NUMINAMATH_GPT_min_balls_for_color_15_l2115_211547


namespace NUMINAMATH_GPT_tangent_line_intersection_l2115_211525

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end NUMINAMATH_GPT_tangent_line_intersection_l2115_211525


namespace NUMINAMATH_GPT_max_and_min_sum_of_factors_of_2000_l2115_211558

theorem max_and_min_sum_of_factors_of_2000 :
  ∃ (a b c d e : ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧ 1 < d ∧ 1 < e ∧ a * b * c * d * e = 2000
  ∧ (a + b + c + d + e = 133 ∨ a + b + c + d + e = 23) :=
by
  sorry

end NUMINAMATH_GPT_max_and_min_sum_of_factors_of_2000_l2115_211558


namespace NUMINAMATH_GPT_grandma_olga_daughters_l2115_211518

theorem grandma_olga_daughters :
  ∃ (D : ℕ), ∃ (S : ℕ),
  S = 3 ∧
  (∃ (total_grandchildren : ℕ), total_grandchildren = 33) ∧
  (∀ D', 6 * D' + 5 * S = 33 → D = D')
:=
sorry

end NUMINAMATH_GPT_grandma_olga_daughters_l2115_211518


namespace NUMINAMATH_GPT_least_possible_value_of_y_l2115_211595

theorem least_possible_value_of_y
  (x y z : ℤ)
  (hx : Even x)
  (hy : Odd y)
  (hz : Odd z)
  (h1 : y - x > 5)
  (h2 : ∀ z', z' - x ≥ 9 → z' ≥ 9) :
  y ≥ 7 :=
by
  -- Proof is not required here
  sorry

end NUMINAMATH_GPT_least_possible_value_of_y_l2115_211595


namespace NUMINAMATH_GPT_probability_sum_greater_than_9_l2115_211553

def num_faces := 6
def total_outcomes := num_faces * num_faces
def favorable_outcomes := 6
def probability := favorable_outcomes / total_outcomes

theorem probability_sum_greater_than_9 (h : total_outcomes = 36) :
  probability = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_sum_greater_than_9_l2115_211553


namespace NUMINAMATH_GPT_tip_percentage_is_30_l2115_211593

theorem tip_percentage_is_30
  (appetizer_cost : ℝ)
  (entree_cost : ℝ)
  (num_entrees : ℕ)
  (dessert_cost : ℝ)
  (total_price_including_tip : ℝ)
  (h_appetizer : appetizer_cost = 9.0)
  (h_entree : entree_cost = 20.0)
  (h_num_entrees : num_entrees = 2)
  (h_dessert : dessert_cost = 11.0)
  (h_total : total_price_including_tip = 78.0) :
  let total_before_tip := appetizer_cost + num_entrees * entree_cost + dessert_cost
  let tip_amount := total_price_including_tip - total_before_tip
  let tip_percentage := (tip_amount / total_before_tip) * 100
  tip_percentage = 30 :=
by
  sorry

end NUMINAMATH_GPT_tip_percentage_is_30_l2115_211593


namespace NUMINAMATH_GPT_sphere_shot_radius_l2115_211570

theorem sphere_shot_radius (R : ℝ) (N : ℕ) (π : ℝ) (r : ℝ) 
  (h₀ : R = 4) (h₁ : N = 64) 
  (h₂ : (4 / 3) * π * (R ^ 3) / ((4 / 3) * π * (r ^ 3)) = N) : 
  r = 1 := 
by
  sorry

end NUMINAMATH_GPT_sphere_shot_radius_l2115_211570


namespace NUMINAMATH_GPT_point_on_curve_l2115_211510

-- Define the parametric curve equations
def onCurve (θ : ℝ) (x y : ℝ) : Prop :=
  x = Real.sin (2 * θ) ∧ y = Real.cos θ + Real.sin θ

-- Define the general form of the curve
def curveEquation (x y : ℝ) : Prop :=
  y^2 = 1 + x

-- The proof statement
theorem point_on_curve : 
  curveEquation (-3/4) (1/2) ∧ ∃ θ : ℝ, onCurve θ (-3/4) (1/2) :=
by
  sorry

end NUMINAMATH_GPT_point_on_curve_l2115_211510


namespace NUMINAMATH_GPT_max_shortest_part_duration_l2115_211521

theorem max_shortest_part_duration (film_duration : ℕ) (part1 part2 part3 part4 : ℕ)
  (h_total : part1 + part2 + part3 + part4 = 192)
  (h_diff1 : part2 ≥ part1 + 6)
  (h_diff2 : part3 ≥ part2 + 6)
  (h_diff3 : part4 ≥ part3 + 6) :
  part1 ≤ 39 := 
sorry

end NUMINAMATH_GPT_max_shortest_part_duration_l2115_211521


namespace NUMINAMATH_GPT_pentagon_zero_impossible_l2115_211574

theorem pentagon_zero_impossible
  (x : Fin 5 → ℝ)
  (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 = 0)
  (operation : ∀ i : Fin 5, ∀ y : Fin 5 → ℝ,
    y i = (x i + x ((i + 1) % 5)) / 2 ∧ y ((i + 1) % 5) = (x i + x ((i + 1) % 5)) / 2) :
  ¬ ∃ (y : ℕ → (Fin 5 → ℝ)), ∃ N : ℕ, y N = 0 := 
sorry

end NUMINAMATH_GPT_pentagon_zero_impossible_l2115_211574


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_octagon_l2115_211583

theorem sum_of_interior_angles_of_octagon (n : ℕ) (h : n = 8) : (n - 2) * 180 = 1080 := by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_octagon_l2115_211583


namespace NUMINAMATH_GPT_sum_of_distances_to_focus_is_ten_l2115_211501

theorem sum_of_distances_to_focus_is_ten (P : ℝ × ℝ) (A B F : ℝ × ℝ)
  (hP : P = (2, 1))
  (hA : A.1^2 = 12 * A.2)
  (hB : B.1^2 = 12 * B.2)
  (hMidpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hFocus : F = (3, 0)) :
  |A.1 - F.1| + |B.1 - F.1| = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distances_to_focus_is_ten_l2115_211501


namespace NUMINAMATH_GPT_sum_due_is_363_l2115_211569

/-
Conditions:
1. BD = 78
2. TD = 66
3. The formula: BD = TD + (TD^2 / PV)
This should imply that PV = 363 given the conditions.
-/

theorem sum_due_is_363 (BD TD PV : ℝ) (h1 : BD = 78) (h2 : TD = 66) (h3 : BD = TD + (TD^2 / PV)) : PV = 363 :=
by
  sorry

end NUMINAMATH_GPT_sum_due_is_363_l2115_211569


namespace NUMINAMATH_GPT_clothing_store_profit_l2115_211550

theorem clothing_store_profit 
  (cost_price selling_price : ℕ)
  (initial_items_per_day items_increment items_reduction : ℕ)
  (initial_profit_per_day : ℕ) :
  -- Conditions
  cost_price = 50 ∧
  selling_price = 90 ∧
  initial_items_per_day = 20 ∧
  items_increment = 2 ∧
  items_reduction = 1 ∧
  initial_profit_per_day = 1200 →
  -- Question
  exists x, 
  (selling_price - x - cost_price) * (initial_items_per_day + items_increment * x) = initial_profit_per_day ∧
  x = 20 := 
sorry

end NUMINAMATH_GPT_clothing_store_profit_l2115_211550


namespace NUMINAMATH_GPT_Alden_nephews_10_years_ago_l2115_211508

noncomputable def nephews_Alden_now : ℕ := sorry
noncomputable def nephews_Alden_10_years_ago (N : ℕ) : ℕ := N / 2
noncomputable def nephews_Vihaan_now (N : ℕ) : ℕ := N + 60
noncomputable def total_nephews (N : ℕ) : ℕ := N + (nephews_Vihaan_now N)

theorem Alden_nephews_10_years_ago (N : ℕ) (h1 : total_nephews N = 260) : 
  nephews_Alden_10_years_ago N = 50 :=
by
  sorry

end NUMINAMATH_GPT_Alden_nephews_10_years_ago_l2115_211508


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l2115_211526

theorem perpendicular_line_through_point (m t : ℝ) (h : 2 * m^2 + m + t = 0) :
  m = 1 → t = -3 → (∀ x y : ℝ, m^2 * x + m * y + t = 0 ↔ x + y - 3 = 0) :=
by
  intros hm ht
  subst hm
  subst ht
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l2115_211526


namespace NUMINAMATH_GPT_value_of_a_minus_b_l2115_211586

theorem value_of_a_minus_b (a b c : ℝ) 
    (h1 : 2011 * a + 2015 * b + c = 2021)
    (h2 : 2013 * a + 2017 * b + c = 2023)
    (h3 : 2012 * a + 2016 * b + 2 * c = 2026) : 
    a - b = -2 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l2115_211586


namespace NUMINAMATH_GPT_fred_red_marbles_l2115_211579

variable (R G B : ℕ)
variable (total : ℕ := 63)
variable (B_val : ℕ := 6)
variable (G_def : G = (1 / 2) * R)
variable (eq1 : R + G + B = total)
variable (eq2 : B = B_val)

theorem fred_red_marbles : R = 38 := 
by
  sorry

end NUMINAMATH_GPT_fred_red_marbles_l2115_211579


namespace NUMINAMATH_GPT_clear_queue_with_three_windows_l2115_211516

def time_to_clear_queue_one_window (a x y : ℕ) : Prop := a / (x - y) = 40

def time_to_clear_queue_two_windows (a x y : ℕ) : Prop := a / (2 * x - y) = 16

theorem clear_queue_with_three_windows (a x y : ℕ) 
  (h1 : time_to_clear_queue_one_window a x y) 
  (h2 : time_to_clear_queue_two_windows a x y) : 
  a / (3 * x - y) = 10 :=
by
  sorry

end NUMINAMATH_GPT_clear_queue_with_three_windows_l2115_211516


namespace NUMINAMATH_GPT_problem1_problem2_l2115_211565

theorem problem1 : (Real.sqrt 18 - Real.sqrt 8 - Real.sqrt 2) = 0 := 
by sorry

theorem problem2 : (6 * Real.sqrt 2 * Real.sqrt 3 + 3 * Real.sqrt 30 / Real.sqrt 5) = 9 * Real.sqrt 6 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2115_211565


namespace NUMINAMATH_GPT_original_decimal_number_l2115_211537

theorem original_decimal_number (I : ℤ) (d : ℝ) (h1 : 0 ≤ d) (h2 : d < 1) (h3 : I + 4 * (I + d) = 21.2) : I + d = 4.3 :=
by
  sorry

end NUMINAMATH_GPT_original_decimal_number_l2115_211537


namespace NUMINAMATH_GPT_evaluate_x3_minus_y3_l2115_211509

theorem evaluate_x3_minus_y3 (x y : ℤ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x^3 - y^3 = -448 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_x3_minus_y3_l2115_211509


namespace NUMINAMATH_GPT_doodads_for_thingamabobs_l2115_211505

-- Definitions for the conditions
def doodads_per_widgets : ℕ := 18
def widgets_per_thingamabobs : ℕ := 11
def widgets_count : ℕ := 5
def thingamabobs_count : ℕ := 4
def target_thingamabobs : ℕ := 80

-- Definition for the final proof statement
theorem doodads_for_thingamabobs : 
    doodads_per_widgets * (target_thingamabobs * widgets_per_thingamabobs / thingamabobs_count / widgets_count) = 792 := 
by
  sorry

end NUMINAMATH_GPT_doodads_for_thingamabobs_l2115_211505


namespace NUMINAMATH_GPT_castle_lego_ratio_l2115_211581

def total_legos : ℕ := 500
def legos_put_back : ℕ := 245
def legos_missing : ℕ := 5
def legos_used : ℕ := total_legos - legos_put_back - legos_missing
def ratio (a b : ℕ) : ℚ := a / b

theorem castle_lego_ratio : ratio legos_used total_legos = 1 / 2 :=
by
  unfold ratio legos_used total_legos legos_put_back legos_missing
  norm_num

end NUMINAMATH_GPT_castle_lego_ratio_l2115_211581


namespace NUMINAMATH_GPT_xy_system_sol_l2115_211555

theorem xy_system_sol (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^3 + y^3 = 416000 / 729 :=
by
  sorry

end NUMINAMATH_GPT_xy_system_sol_l2115_211555


namespace NUMINAMATH_GPT_cylinder_lateral_surface_area_l2115_211587

theorem cylinder_lateral_surface_area
    (r h : ℝ) (hr : r = 3) (hh : h = 10) :
    2 * Real.pi * r * h = 60 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cylinder_lateral_surface_area_l2115_211587


namespace NUMINAMATH_GPT_evaluate_expression_l2115_211576

theorem evaluate_expression : 
  (20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12 - 13 + 14 - 15 + 16 - 17 + 18 - 19 + 20)
  = 10 / 11 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2115_211576


namespace NUMINAMATH_GPT_final_weight_is_correct_l2115_211504

-- Define the various weights after each week
def initial_weight : ℝ := 180
def first_week_removed : ℝ := 0.28 * initial_weight
def first_week_remaining : ℝ := initial_weight - first_week_removed
def second_week_removed : ℝ := 0.18 * first_week_remaining
def second_week_remaining : ℝ := first_week_remaining - second_week_removed
def third_week_removed : ℝ := 0.20 * second_week_remaining
def final_weight : ℝ := second_week_remaining - third_week_removed

-- State the theorem to prove the final weight equals 85.0176 kg
theorem final_weight_is_correct : final_weight = 85.0176 := 
by 
  sorry

end NUMINAMATH_GPT_final_weight_is_correct_l2115_211504


namespace NUMINAMATH_GPT_find_integer_n_l2115_211529

theorem find_integer_n (n : ℕ) (hn1 : 0 ≤ n) (hn2 : n < 102) (hmod : 99 * n % 102 = 73) : n = 97 :=
  sorry

end NUMINAMATH_GPT_find_integer_n_l2115_211529


namespace NUMINAMATH_GPT_mixtape_first_side_songs_l2115_211597

theorem mixtape_first_side_songs (total_length : ℕ) (second_side_songs : ℕ) (song_length : ℕ) :
  total_length = 40 → second_side_songs = 4 → song_length = 4 → (total_length - second_side_songs * song_length) / song_length = 6 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_mixtape_first_side_songs_l2115_211597


namespace NUMINAMATH_GPT_function_two_common_points_with_xaxis_l2115_211592

theorem function_two_common_points_with_xaxis (c : ℝ) :
  (∀ x : ℝ, x^3 - 3 * x + c = 0 → x = -1 ∨ x = 1) → (c = -2 ∨ c = 2) :=
by
  sorry

end NUMINAMATH_GPT_function_two_common_points_with_xaxis_l2115_211592


namespace NUMINAMATH_GPT_volume_ratio_l2115_211598

variable (A B : ℝ)

theorem volume_ratio (h1 : (3 / 4) * A = (5 / 8) * B) :
  A / B = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_l2115_211598


namespace NUMINAMATH_GPT_part_time_job_pay_per_month_l2115_211552

def tuition_fee : ℝ := 90
def scholarship_percent : ℝ := 0.30
def scholarship_amount := scholarship_percent * tuition_fee
def amount_after_scholarship := tuition_fee - scholarship_amount
def remaining_amount : ℝ := 18
def months_to_pay : ℝ := 3
def amount_paid_so_far := amount_after_scholarship - remaining_amount

theorem part_time_job_pay_per_month : amount_paid_so_far / months_to_pay = 15 := by
  sorry

end NUMINAMATH_GPT_part_time_job_pay_per_month_l2115_211552


namespace NUMINAMATH_GPT_ivan_max_13_bars_a_ivan_max_13_bars_b_l2115_211512

variable (n : ℕ) (ivan_max_bags : ℕ)

-- Condition 1: initial count of bars in the chest
def initial_bars := 13

-- Condition 2: function to check if transfers are possible
def can_transfer (bars_in_chest : ℕ) (bars_in_bag : ℕ) (last_transfer : ℕ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ t₁ ≠ last_transfer ∧ t₂ ≠ last_transfer ∧
           t₁ + bars_in_bag ≤ initial_bars ∧ bars_in_chest - t₁ + t₂ = bars_in_chest

-- Proof Problem (a): Given initially 13 bars, prove Ivan can secure 13 bars
theorem ivan_max_13_bars_a 
  (initial_bars : ℕ := 13) 
  (target_bars : ℕ := 13)
  (can_transfer : ∀ (bars_in_chest bars_in_bag last_transfer : ℕ), can_transfer bars_in_chest bars_in_bag last_transfer) 
  (h_initial_bars : initial_bars = 13) :
  ivan_max_bags = target_bars :=
by
  sorry

-- Proof Problem (b): Given initially 14 bars, prove Ivan can secure 13 bars
theorem ivan_max_13_bars_b 
  (initial_bars : ℕ := 14)
  (target_bars : ℕ := 13)
  (can_transfer : ∀ (bars_in_chest bars_in_bag last_transfer : ℕ), can_transfer bars_in_chest bars_in_bag last_transfer) 
  (h_initial_bars : initial_bars = 14) :
  ivan_max_bags = target_bars :=
by
  sorry

end NUMINAMATH_GPT_ivan_max_13_bars_a_ivan_max_13_bars_b_l2115_211512


namespace NUMINAMATH_GPT_determine_condition_l2115_211589

theorem determine_condition (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) 
    (h_eq : (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) : 
    b + c = 12 :=
by
  sorry

end NUMINAMATH_GPT_determine_condition_l2115_211589


namespace NUMINAMATH_GPT_second_divisor_27_l2115_211599

theorem second_divisor_27 (N : ℤ) (D : ℤ) (k : ℤ) (q : ℤ) (h1 : N = 242 * k + 100) (h2 : N = D * q + 19) : D = 27 := by
  sorry

end NUMINAMATH_GPT_second_divisor_27_l2115_211599


namespace NUMINAMATH_GPT_cubes_with_all_three_faces_l2115_211527

theorem cubes_with_all_three_faces (total_cubes red_cubes blue_cubes green_cubes: ℕ) 
  (h_total: total_cubes = 100)
  (h_red: red_cubes = 80)
  (h_blue: blue_cubes = 85)
  (h_green: green_cubes = 75) :
  40 ≤ total_cubes - ((total_cubes - red_cubes) + (total_cubes - blue_cubes) + (total_cubes - green_cubes)) ∧ (total_cubes - ((total_cubes - red_cubes) + (total_cubes - blue_cubes) + (total_cubes - green_cubes))) ≤ 75 :=
by {
  sorry
}

end NUMINAMATH_GPT_cubes_with_all_three_faces_l2115_211527


namespace NUMINAMATH_GPT_x4_plus_inverse_x4_l2115_211507

theorem x4_plus_inverse_x4 (x : ℝ) (hx : x ^ 2 + 1 / x ^ 2 = 2) : x ^ 4 + 1 / x ^ 4 = 2 := 
sorry

end NUMINAMATH_GPT_x4_plus_inverse_x4_l2115_211507


namespace NUMINAMATH_GPT_sand_weight_proof_l2115_211562

-- Definitions for the given conditions
def side_length : ℕ := 40
def bag_weight : ℕ := 30
def area_per_bag : ℕ := 80

-- Total area of the sandbox
def total_area := side_length * side_length

-- Number of bags needed
def number_of_bags := total_area / area_per_bag

-- Total weight of sand needed
def total_weight := number_of_bags * bag_weight

-- The proof statement
theorem sand_weight_proof :
  total_weight = 600 :=
by
  sorry

end NUMINAMATH_GPT_sand_weight_proof_l2115_211562


namespace NUMINAMATH_GPT_min_value_of_sum_inverse_l2115_211560

theorem min_value_of_sum_inverse (m n : ℝ) 
  (H1 : ∃ (x y : ℝ), (x + y - 1 = 0 ∧ 3 * x - y - 7 = 0) ∧ (mx + y + n = 0))
  (H2 : mn > 0) : 
  ∃ k : ℝ, k = 8 ∧ ∀ (m n : ℝ), mn > 0 → (2 * m + n = 1) → 1 / m + 2 / n ≥ k :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_sum_inverse_l2115_211560


namespace NUMINAMATH_GPT_area_of_inscribed_rectangle_l2115_211534

theorem area_of_inscribed_rectangle (r l w : ℝ) (h1 : r = 8) (h2 : l / w = 3) (h3 : w = 2 * r) : l * w = 768 :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_rectangle_l2115_211534


namespace NUMINAMATH_GPT_f_diff_l2115_211541

def f (n : ℕ) : ℚ :=
  (Finset.range (3 * n)).sum (λ k => (1 : ℚ) / (k + 1))

theorem f_diff (n : ℕ) : f (n + 1) - f n = (1 / (3 * n) + 1 / (3 * n + 1) + 1 / (3 * n + 2)) :=
by
  sorry

end NUMINAMATH_GPT_f_diff_l2115_211541


namespace NUMINAMATH_GPT_evaporation_period_length_l2115_211578

theorem evaporation_period_length
  (initial_water : ℕ) (daily_evaporation : ℝ) (evaporated_percentage : ℝ) : 
  evaporated_percentage * (initial_water : ℝ) / 100 / daily_evaporation = 22 :=
by
  -- Conditions of the problem
  let initial_water := 12
  let daily_evaporation := 0.03
  let evaporated_percentage := 5.5
  -- Sorry proof placeholder
  sorry

end NUMINAMATH_GPT_evaporation_period_length_l2115_211578


namespace NUMINAMATH_GPT_unique_solution_is_2_or_minus_2_l2115_211520

theorem unique_solution_is_2_or_minus_2 (a : ℝ) :
  (∃ x : ℝ, ∀ y : ℝ, (y^2 + a * y + 1 = 0 ↔ y = x)) → (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_GPT_unique_solution_is_2_or_minus_2_l2115_211520


namespace NUMINAMATH_GPT_Youseff_time_difference_l2115_211585

theorem Youseff_time_difference 
  (blocks : ℕ)
  (walk_time_per_block : ℕ) 
  (bike_time_per_block_sec : ℕ) 
  (sec_per_min : ℕ)
  (h_blocks : blocks = 12) 
  (h_walk_time_per_block : walk_time_per_block = 1) 
  (h_bike_time_per_block_sec : bike_time_per_block_sec = 20) 
  (h_sec_per_min : sec_per_min = 60) : 
  (blocks * walk_time_per_block) - ((blocks * bike_time_per_block_sec) / sec_per_min) = 8 :=
by 
  sorry

end NUMINAMATH_GPT_Youseff_time_difference_l2115_211585


namespace NUMINAMATH_GPT_number_of_adults_l2115_211545

-- Given constants
def children : ℕ := 200
def price_child (price_adult : ℕ) : ℕ := price_adult / 2
def total_amount : ℕ := 16000

-- Based on the problem conditions
def price_adult := 32

-- The generated proof problem
theorem number_of_adults 
    (price_adult_gt_0 : price_adult > 0)
    (h_price_adult : price_adult = 32)
    (h_total_amount : total_amount = 16000) 
    (h_price_relation : ∀ price_adult, price_adult / 2 * 2 = price_adult) :
  ∃ A : ℕ, 32 * A + 16 * 200 = 16000 ∧ price_child price_adult = 16 := by
  sorry

end NUMINAMATH_GPT_number_of_adults_l2115_211545


namespace NUMINAMATH_GPT_sufficient_condition_of_implications_l2115_211580

variables (P1 P2 θ : Prop)

theorem sufficient_condition_of_implications
  (h1 : P1 → θ)
  (h2 : P2 → P1) :
  P2 → θ :=
by sorry

end NUMINAMATH_GPT_sufficient_condition_of_implications_l2115_211580


namespace NUMINAMATH_GPT_denomination_is_20_l2115_211563

noncomputable def denomination_of_250_coins (x : ℕ) : Prop :=
  250 * x + 84 * 25 = 7100

theorem denomination_is_20 (x : ℕ) (h : denomination_of_250_coins x) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_denomination_is_20_l2115_211563


namespace NUMINAMATH_GPT_min_xy_min_x_add_y_l2115_211588

open Real

theorem min_xy (x y : ℝ) (h1 : log x + log y = log (x + y + 3)) (hx : x > 0) (hy : y > 0) : xy ≥ 9 := sorry

theorem min_x_add_y (x y : ℝ) (h1 : log x + log y = log (x + y + 3)) (hx : x > 0) (hy : y > 0) : x + y ≥ 6 := sorry

end NUMINAMATH_GPT_min_xy_min_x_add_y_l2115_211588


namespace NUMINAMATH_GPT_find_rectangle_dimensions_area_30_no_rectangle_dimensions_area_32_l2115_211596

noncomputable def length_width_rectangle_area_30 : Prop :=
∃ (x y : ℝ), x * y = 30 ∧ 2 * (x + y) = 22 ∧ x = 6 ∧ y = 5

noncomputable def impossible_rectangle_area_32 : Prop :=
¬(∃ (x y : ℝ), x * y = 32 ∧ 2 * (x + y) = 22)

-- Proof statements (without proofs)
theorem find_rectangle_dimensions_area_30 : length_width_rectangle_area_30 :=
sorry

theorem no_rectangle_dimensions_area_32 : impossible_rectangle_area_32 :=
sorry

end NUMINAMATH_GPT_find_rectangle_dimensions_area_30_no_rectangle_dimensions_area_32_l2115_211596


namespace NUMINAMATH_GPT_blue_cards_in_box_l2115_211522

theorem blue_cards_in_box (x : ℕ) (h : 0.6 = (x : ℝ) / (x + 8)) : x = 12 :=
sorry

end NUMINAMATH_GPT_blue_cards_in_box_l2115_211522


namespace NUMINAMATH_GPT_decimal_representation_of_fraction_l2115_211568

theorem decimal_representation_of_fraction :
  (3 / 40 : ℝ) = 0.075 :=
sorry

end NUMINAMATH_GPT_decimal_representation_of_fraction_l2115_211568


namespace NUMINAMATH_GPT_prod_72516_9999_l2115_211582

theorem prod_72516_9999 : 72516 * 9999 = 724987484 :=
by
  sorry

end NUMINAMATH_GPT_prod_72516_9999_l2115_211582


namespace NUMINAMATH_GPT_runners_adjacent_vertices_after_2013_l2115_211528

def hexagon_run_probability (t : ℕ) : ℚ :=
  (2 / 3) + (1 / 3) * ((1 / 4) ^ t)

theorem runners_adjacent_vertices_after_2013 :
  hexagon_run_probability 2013 = (2 / 3) + (1 / 3) * ((1 / 4) ^ 2013) := 
by 
  sorry

end NUMINAMATH_GPT_runners_adjacent_vertices_after_2013_l2115_211528


namespace NUMINAMATH_GPT_sequence_sum_l2115_211559

theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (H_n_def : H_n = (a 1 + (2:ℕ) * a 2 + (2:ℕ) ^ (n - 1) * a n) / n)
  (H_n_val : H_n = 2^n) :
  S n = n * (n + 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l2115_211559


namespace NUMINAMATH_GPT_min_distance_l2115_211540

noncomputable def curve (x : ℝ) : ℝ := (1/4) * x^2 - (1/2) * Real.log x
noncomputable def line (x : ℝ) : ℝ := (3/4) * x - 1

theorem min_distance :
  ∀ P Q : ℝ × ℝ, 
  P.2 = curve P.1 → 
  Q.2 = line Q.1 → 
  ∃ min_dist : ℝ, 
  min_dist = (2 - 2 * Real.log 2) / 5 := 
sorry

end NUMINAMATH_GPT_min_distance_l2115_211540


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l2115_211513

theorem asymptotes_of_hyperbola (a : ℝ) :
  (∃ a : ℝ, 9 + a = 13) →
  (∀ x y : ℝ, (x^2 / 9 - y^2 / a = 1) → (a = 4)) →
  (forall (x y : ℝ), (x^2 / 9 - y^2 / 4 = 0) → 
    (y = (2/3) * x) ∨ (y = -(2/3) * x)) :=
by
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l2115_211513


namespace NUMINAMATH_GPT_average_students_is_12_l2115_211535

-- Definitions based on the problem's conditions
variables (a b c : Nat)

-- Given conditions
axiom condition1 : a + b + c = 30
axiom condition2 : a + c = 19
axiom condition3 : b + c = 9

-- Prove that the number of average students (c) is 12
theorem average_students_is_12 : c = 12 := by 
  sorry

end NUMINAMATH_GPT_average_students_is_12_l2115_211535


namespace NUMINAMATH_GPT_carmen_rope_gcd_l2115_211577

/-- Carmen has three ropes with lengths 48, 64, and 80 inches respectively.
    She needs to cut these ropes into pieces of equal length for a craft project,
    ensuring no rope is left unused.
    Prove that the greatest length in inches that each piece can have is 16. -/
theorem carmen_rope_gcd :
  Nat.gcd (Nat.gcd 48 64) 80 = 16 := by
  sorry

end NUMINAMATH_GPT_carmen_rope_gcd_l2115_211577


namespace NUMINAMATH_GPT_smallest_number_divisible_conditions_l2115_211557

theorem smallest_number_divisible_conditions :
  ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m % 8 = 6 ∧ m % 7 = 5 → n ≤ m →
  n % 9 = 0 := by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_conditions_l2115_211557


namespace NUMINAMATH_GPT_find_vertex_C_l2115_211594

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)
def euler_line (x y : ℝ) : Prop := x - y + 2 = 0

theorem find_vertex_C 
  (C : ℝ × ℝ)
  (h_centroid : (2 + C.1) / 3 = (4 + C.2) / 3)
  (h_euler_line : euler_line ((2 + C.1) / 3) ((4 + C.2) / 3))
  (h_circumcenter : (C.1 + 1)^2 + (C.2 - 1)^2 = 10) :
  C = (-4, 0) :=
sorry

end NUMINAMATH_GPT_find_vertex_C_l2115_211594


namespace NUMINAMATH_GPT_peggy_dolls_after_all_events_l2115_211572

def initial_dolls : Nat := 6
def grandmother_gift : Nat := 28
def birthday_gift : Nat := grandmother_gift / 2
def lost_dolls (total : Nat) : Nat := (10 * total + 9) / 100  -- using integer division for rounding 10% up
def easter_gift : Nat := (birthday_gift + 2) / 3  -- using integer division for rounding one-third up
def friend_exchange_gain : Int := -1  -- gaining 1 doll but losing 2
def christmas_gift (easter_dolls : Nat) : Nat := (20 * easter_dolls) / 100 + easter_dolls  -- 20% more dolls
def ruined_dolls : Nat := 3

theorem peggy_dolls_after_all_events : initial_dolls + grandmother_gift + birthday_gift - lost_dolls (initial_dolls + grandmother_gift + birthday_gift) + easter_gift + friend_exchange_gain.toNat + christmas_gift easter_gift - ruined_dolls = 50 :=
by
  sorry

end NUMINAMATH_GPT_peggy_dolls_after_all_events_l2115_211572


namespace NUMINAMATH_GPT_remaining_area_is_correct_l2115_211542

-- Define the large rectangle's side lengths
def large_rectangle_length1 (x : ℝ) := x + 7
def large_rectangle_length2 (x : ℝ) := x + 5

-- Define the hole's side lengths
def hole_length1 (x : ℝ) := x + 1
def hole_length2 (x : ℝ) := x + 4

-- Calculate the areas
def large_rectangle_area (x : ℝ) := large_rectangle_length1 x * large_rectangle_length2 x
def hole_area (x : ℝ) := hole_length1 x * hole_length2 x

-- Define the remaining area after subtracting the hole area from the large rectangle area
def remaining_area (x : ℝ) := large_rectangle_area x - hole_area x

-- Problem statement: prove that the remaining area is 7x + 31
theorem remaining_area_is_correct (x : ℝ) : remaining_area x = 7 * x + 31 :=
by 
  -- The proof should be provided here, but for now we use 'sorry' to omit it
  sorry

end NUMINAMATH_GPT_remaining_area_is_correct_l2115_211542


namespace NUMINAMATH_GPT_block_fraction_visible_above_water_l2115_211502

-- Defining constants
def weight_of_block : ℝ := 30 -- N
def buoyant_force_submerged : ℝ := 50 -- N

-- Defining the proof problem
theorem block_fraction_visible_above_water (W Fb : ℝ) (hW : W = weight_of_block) (hFb : Fb = buoyant_force_submerged) :
  (1 - W / Fb) = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_block_fraction_visible_above_water_l2115_211502


namespace NUMINAMATH_GPT_task1_task2_l2115_211536

/-- Given conditions -/
def cost_A : Nat := 30
def cost_B : Nat := 40
def sell_A : Nat := 35
def sell_B : Nat := 50
def max_cost : Nat := 1550
def min_profit : Nat := 365
def total_cars : Nat := 40

/-- Task 1: Prove maximum B-type cars produced if 10 A-type cars are produced -/
theorem task1 (A: Nat) (B: Nat) (hA: A = 10) (hC: cost_A * A + cost_B * B ≤ max_cost) : B ≤ 31 :=
by sorry

/-- Task 2: Prove the possible production plans producing 40 cars meeting profit and cost constraints -/
theorem task2 (A: Nat) (B: Nat) (hTotal: A + B = total_cars)
(hCost: cost_A * A + cost_B * B ≤ max_cost) 
(hProfit: (sell_A - cost_A) * A + (sell_B - cost_B) * B ≥ min_profit) : 
  (A = 5 ∧ B = 35) ∨ (A = 6 ∧ B = 34) ∨ (A = 7 ∧ B = 33) 
∧ (375 ≤ (sell_A - cost_A) * 5 + (sell_B - cost_B) * 35 ∧ 375 ≥ (sell_A - cost_A) * 5 + (sell_B - cost_B) * 35) :=
by sorry

end NUMINAMATH_GPT_task1_task2_l2115_211536


namespace NUMINAMATH_GPT_intersections_count_l2115_211564

theorem intersections_count
  (c : ℕ)  -- crosswalks per intersection
  (l : ℕ)  -- lines per crosswalk
  (t : ℕ)  -- total lines
  (h_c : c = 4)
  (h_l : l = 20)
  (h_t : t = 400) :
  t / (c * l) = 5 :=
  by
    sorry

end NUMINAMATH_GPT_intersections_count_l2115_211564


namespace NUMINAMATH_GPT_concave_side_probability_l2115_211515

theorem concave_side_probability (tosses : ℕ) (frequency_convex : ℝ) (htosses : tosses = 1000) (hfrequency : frequency_convex = 0.44) :
  ∀ probability_concave : ℝ, probability_concave = 1 - frequency_convex → probability_concave = 0.56 :=
by
  intros probability_concave h
  rw [hfrequency] at h
  rw [h]
  norm_num
  done

end NUMINAMATH_GPT_concave_side_probability_l2115_211515


namespace NUMINAMATH_GPT_initial_apples_l2115_211539

theorem initial_apples (Initially_Apples : ℕ) (Added_Apples : ℕ) (Total_Apples : ℕ)
  (h1 : Added_Apples = 8) (h2 : Total_Apples = 17) : Initially_Apples = 9 :=
by
  have h3 : Added_Apples + Initially_Apples = Total_Apples := by
    sorry
  linarith

end NUMINAMATH_GPT_initial_apples_l2115_211539


namespace NUMINAMATH_GPT_no_consecutive_squares_of_arithmetic_progression_l2115_211554

theorem no_consecutive_squares_of_arithmetic_progression (d : ℕ):
  (d % 10000 = 2019) →
  (∀ a b c : ℕ, a < b ∧ b < c → b^2 - a^2 = d ∧ c^2 - b^2 = d →
  false) :=
sorry

end NUMINAMATH_GPT_no_consecutive_squares_of_arithmetic_progression_l2115_211554


namespace NUMINAMATH_GPT_alcohol_added_l2115_211517

-- Definitions from conditions
def initial_volume : ℝ := 40
def initial_alcohol_concentration : ℝ := 0.05
def initial_alcohol_amount : ℝ := initial_volume * initial_alcohol_concentration
def added_water_volume : ℝ := 3.5
def final_alcohol_concentration : ℝ := 0.17

-- The problem to be proven
theorem alcohol_added :
  ∃ x : ℝ,
    x = (final_alcohol_concentration * (initial_volume + x + added_water_volume) - initial_alcohol_amount) :=
by
  sorry

end NUMINAMATH_GPT_alcohol_added_l2115_211517


namespace NUMINAMATH_GPT_price_per_piece_l2115_211519

variable (y : ℝ)

theorem price_per_piece (h : (20 + y - 12) * (240 - 40 * y) = 1980) :
  20 + y = 21 ∨ 20 + y = 23 :=
sorry

end NUMINAMATH_GPT_price_per_piece_l2115_211519


namespace NUMINAMATH_GPT_each_person_bids_five_times_l2115_211548

noncomputable def auction_bidding : Prop :=
  let initial_price := 15
  let final_price := 65
  let price_increase_per_bid := 5
  let number_of_bidders := 2
  let total_increase := final_price - initial_price
  let total_bids := total_increase / price_increase_per_bid
  total_bids / number_of_bidders = 5

theorem each_person_bids_five_times : auction_bidding :=
by
  -- The proof will be filled in here.
  sorry

end NUMINAMATH_GPT_each_person_bids_five_times_l2115_211548


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l2115_211500

theorem common_ratio_geometric_sequence
  (a_1 : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (geom_sum : ∀ n q, q ≠ 1 → S n = a_1 * (1 - q^n) / (1 - q))
  (h_arithmetic : 2 * S 4 = S 5 + S 6)
  : (∃ q : ℝ, ∀ n : ℕ, q ≠ 1 → S n = a_1 * (1 - q^n) / (1 - q)) → q = -2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l2115_211500


namespace NUMINAMATH_GPT_intersection_point_l2115_211503

noncomputable def line1 (x : ℝ) : ℝ := 3 * x + 10

noncomputable def slope_perp : ℝ := -1/3

noncomputable def line_perp (x : ℝ) : ℝ := slope_perp * x + (2 - slope_perp * 3)

theorem intersection_point : 
  ∃ (x y : ℝ), y = line1 x ∧ y = line_perp x ∧ x = -21 / 10 ∧ y = 37 / 10 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l2115_211503


namespace NUMINAMATH_GPT_probability_of_one_machine_maintenance_l2115_211584

theorem probability_of_one_machine_maintenance :
  let pA := 0.1
  let pB := 0.2
  let pC := 0.4
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC) = 0.444 :=
by {
  let pA := 0.1
  let pB := 0.2
  let pC := 0.4
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  show (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC) = 0.444
  sorry
}

end NUMINAMATH_GPT_probability_of_one_machine_maintenance_l2115_211584


namespace NUMINAMATH_GPT_sin_sum_cos_product_tan_sum_tan_product_l2115_211573

theorem sin_sum_cos_product
  (A B C : ℝ)
  (h : A + B + C = π) : 
  (Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) :=
sorry

theorem tan_sum_tan_product
  (A B C : ℝ)
  (h : A + B + C = π) :
  (Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) := 
sorry

end NUMINAMATH_GPT_sin_sum_cos_product_tan_sum_tan_product_l2115_211573


namespace NUMINAMATH_GPT_largest_unorderable_dumplings_l2115_211531

theorem largest_unorderable_dumplings : 
  ∀ (a b c : ℕ), 43 ≠ 6 * a + 9 * b + 20 * c :=
by sorry

end NUMINAMATH_GPT_largest_unorderable_dumplings_l2115_211531


namespace NUMINAMATH_GPT_min_AP_l2115_211506

noncomputable def A : ℝ × ℝ := (2, 0)
noncomputable def B' : ℝ × ℝ := (8, 6)
def parabola (P' : ℝ × ℝ) : Prop := P'.2^2 = 8 * P'.1

theorem min_AP'_plus_BP' : 
  ∃ P' : ℝ × ℝ, parabola P' ∧ (dist A P' + dist B' P' = 12) := 
sorry

end NUMINAMATH_GPT_min_AP_l2115_211506


namespace NUMINAMATH_GPT_maritza_study_hours_l2115_211590

noncomputable def time_to_study_for_citizenship_test (num_mc_questions num_fitb_questions time_mc time_fitb : ℕ) : ℕ :=
  (num_mc_questions * time_mc + num_fitb_questions * time_fitb) / 60

theorem maritza_study_hours :
  time_to_study_for_citizenship_test 30 30 15 25 = 20 :=
by
  sorry

end NUMINAMATH_GPT_maritza_study_hours_l2115_211590


namespace NUMINAMATH_GPT_max_gcd_is_one_l2115_211523

-- Defining the sequence a_n
def a_n (n : ℕ) : ℕ := 101 + n^3

-- Defining the gcd function for a_n and a_(n+1)
def d_n (n : ℕ) : ℕ := Nat.gcd (a_n n) (a_n (n + 1))

-- The theorem stating the maximum value of d_n is 1
theorem max_gcd_is_one : ∀ n : ℕ, d_n n = 1 := by
  -- Proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_max_gcd_is_one_l2115_211523


namespace NUMINAMATH_GPT_area_of_triangle_l2115_211544

theorem area_of_triangle (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) : 
  1/2 * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l2115_211544


namespace NUMINAMATH_GPT_apples_given_by_anita_l2115_211524

variable (initial_apples current_apples needed_apples : ℕ)

theorem apples_given_by_anita (h1 : initial_apples = 4) 
                               (h2 : needed_apples = 10)
                               (h3 : needed_apples - current_apples = 1) : 
  current_apples - initial_apples = 5 := 
by
  sorry

end NUMINAMATH_GPT_apples_given_by_anita_l2115_211524
