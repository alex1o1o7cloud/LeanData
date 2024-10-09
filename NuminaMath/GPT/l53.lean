import Mathlib

namespace rectangle_diagonal_length_l53_5379

theorem rectangle_diagonal_length (L W : ℝ) (h1 : L * W = 20) (h2 : L + W = 9) :
  (L^2 + W^2) = 41 :=
by
  sorry

end rectangle_diagonal_length_l53_5379


namespace triangle_shape_l53_5357

theorem triangle_shape (a b c : ℝ) (h : a^4 - b^4 + (b^2 * c^2 - a^2 * c^2) = 0) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end triangle_shape_l53_5357


namespace miss_davis_items_left_l53_5328

theorem miss_davis_items_left 
  (popsicle_sticks_per_group : ℕ := 15) 
  (straws_per_group : ℕ := 20) 
  (num_groups : ℕ := 10) 
  (total_items_initial : ℕ := 500) : 
  total_items_initial - (num_groups * (popsicle_sticks_per_group + straws_per_group)) = 150 :=
by 
  sorry

end miss_davis_items_left_l53_5328


namespace eval_imaginary_expression_l53_5348

theorem eval_imaginary_expression :
  ∀ (i : ℂ), i^2 = -1 → i^2022 + i^2023 + i^2024 + i^2025 = 0 :=
by
  sorry

end eval_imaginary_expression_l53_5348


namespace product_of_integers_l53_5395

theorem product_of_integers
  (A B C D : ℕ)
  (hA : A > 0)
  (hB : B > 0)
  (hC : C > 0)
  (hD : D > 0)
  (h_sum : A + B + C + D = 72)
  (h_eq : A + 3 = B - 3 ∧ B - 3 = C * 3 ∧ C * 3 = D / 2) :
  A * B * C * D = 68040 := 
by
  sorry

end product_of_integers_l53_5395


namespace ellipse_ratio_squared_l53_5374

theorem ellipse_ratio_squared (a b c : ℝ) 
    (h1 : b / a = a / c) 
    (h2 : c^2 = a^2 - b^2) : (b / a)^2 = 1 / 2 :=
by
  sorry

end ellipse_ratio_squared_l53_5374


namespace sequence_formula_l53_5382

-- Definitions of the sequence and conditions
def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) a

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S a n + a n = 2 * n + 1

-- Proposition to prove
theorem sequence_formula (a : ℕ → ℝ) (h : satisfies_condition a) : 
  ∀ n : ℕ, a n = 2 - 1 / 2^n := sorry

end sequence_formula_l53_5382


namespace exists_a_l53_5327

noncomputable def a : ℕ → ℕ := sorry

theorem exists_a : a (a (a (a 1))) = 458329 :=
by
  -- proof skipped
  sorry

end exists_a_l53_5327


namespace soccer_substitutions_mod_2000_l53_5331

theorem soccer_substitutions_mod_2000 :
  let a_0 := 1
  let a_1 := 11 * 11
  let a_2 := 11 * 10 * a_1
  let a_3 := 11 * 9 * a_2
  let a_4 := 11 * 8 * a_3
  let n := a_0 + a_1 + a_2 + a_3 + a_4
  n % 2000 = 942 :=
by
  sorry

end soccer_substitutions_mod_2000_l53_5331


namespace linear_dependency_k_val_l53_5360

theorem linear_dependency_k_val (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 2 * c1 + 4 * c2 = 0 ∧ 3 * c1 + k * c2 = 0) ↔ k = 6 :=
by sorry

end linear_dependency_k_val_l53_5360


namespace evaluate_g_at_3_l53_5338

def g (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g 3 = 200 := by
  sorry

end evaluate_g_at_3_l53_5338


namespace variance_of_scores_l53_5341

-- Define the list of scores
def scores : List ℕ := [110, 114, 121, 119, 126]

-- Define the formula for variance calculation
def variance (l : List ℕ) : ℚ :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  (l.map (λ x => ((x : ℚ) - mean) ^ 2)).sum / n

-- The main theorem to be proved
theorem variance_of_scores :
  variance scores = 30.8 := 
  by
    sorry

end variance_of_scores_l53_5341


namespace simplify_expression_l53_5346

theorem simplify_expression (x y : ℝ) :
  (2 * x^3 * y^2 - 3 * x^2 * y^3) / (1 / 2 * x * y)^2 = 8 * x - 12 * y := by
  sorry

end simplify_expression_l53_5346


namespace sum_of_max_marks_l53_5377

theorem sum_of_max_marks :
  ∀ (M S E : ℝ),
  (30 / 100 * M = 180) ∧
  (50 / 100 * S = 200) ∧
  (40 / 100 * E = 120) →
  M + S + E = 1300 :=
by
  intros M S E h
  sorry

end sum_of_max_marks_l53_5377


namespace sum_of_digits_is_11_l53_5305

def digits_satisfy_conditions (A B C : ℕ) : Prop :=
  (C = 0 ∨ C = 5) ∧
  (A = 2 * B) ∧
  (A * B * C = 40)

theorem sum_of_digits_is_11 (A B C : ℕ) (h : digits_satisfy_conditions A B C) : A + B + C = 11 :=
by
  sorry

end sum_of_digits_is_11_l53_5305


namespace riverton_soccer_physics_l53_5333

theorem riverton_soccer_physics : 
  let total_players := 15
  let math_players := 9
  let both_subjects := 3
  let only_physics := total_players - math_players
  let physics_players := only_physics + both_subjects
  physics_players = 9 :=
by
  sorry

end riverton_soccer_physics_l53_5333


namespace expression_nonnegative_l53_5312

theorem expression_nonnegative (x : ℝ) :
  0 <= x ∧ x < 3 → (2*x - 6*x^2 + 9*x^3) / (9 - x^3) ≥ 0 := 
by
  sorry

end expression_nonnegative_l53_5312


namespace oranges_in_bin_after_changes_l53_5358

-- Define the initial number of oranges
def initial_oranges : ℕ := 34

-- Define the number of oranges thrown away
def oranges_thrown_away : ℕ := 20

-- Define the number of new oranges added
def new_oranges_added : ℕ := 13

-- Theorem statement to prove the final number of oranges in the bin
theorem oranges_in_bin_after_changes :
  initial_oranges - oranges_thrown_away + new_oranges_added = 27 := by
  sorry

end oranges_in_bin_after_changes_l53_5358


namespace find_dolls_l53_5380

namespace DollsProblem

variables (S D : ℕ) -- Define S and D as natural numbers

-- Conditions as per the problem
def cond1 : Prop := 4 * S + 3 = D
def cond2 : Prop := 5 * S = D + 6

-- Theorem stating the problem
theorem find_dolls (h1 : cond1 S D) (h2 : cond2 S D) : D = 39 :=
by
  sorry

end DollsProblem

end find_dolls_l53_5380


namespace compute_pqr_l53_5371

theorem compute_pqr (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h_sum : p + q + r = 30) 
  (h_equation : 1 / p + 1 / q + 1 / r + 420 / (p * q * r) = 1) : 
  p * q * r = 576 :=
sorry

end compute_pqr_l53_5371


namespace open_box_volume_l53_5342

-- Define the initial conditions
def length_of_sheet := 100
def width_of_sheet := 50
def height_of_parallelogram := 10
def base_of_parallelogram := 10

-- Define the expected dimensions of the box after cutting
def length_of_box := length_of_sheet - 2 * base_of_parallelogram
def width_of_box := width_of_sheet - 2 * base_of_parallelogram
def height_of_box := height_of_parallelogram

-- Define the expected volume of the box
def volume_of_box := length_of_box * width_of_box * height_of_box

-- Theorem to prove the correct volume of the box based on the given dimensions
theorem open_box_volume : volume_of_box = 24000 := by
  -- The proof will be included here
  sorry

end open_box_volume_l53_5342


namespace solve_quadratic_l53_5367

-- Problem Definition
def quadratic_equation (x : ℝ) : Prop :=
  2 * x^2 - 6 * x + 3 = 0

-- Solution Definition
def solution1 (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2

-- Lean Theorem Statement
theorem solve_quadratic : ∀ x : ℝ, quadratic_equation x ↔ solution1 x :=
sorry

end solve_quadratic_l53_5367


namespace find_a_l53_5355

theorem find_a (a : ℝ) (x : ℝ) (h : ∀ (x : ℝ), 2 * x - a ≤ -1 ↔ x ≤ 1) : a = 3 :=
sorry

end find_a_l53_5355


namespace negation_of_divisible_by_2_even_l53_5363

theorem negation_of_divisible_by_2_even :
  (¬ ∀ n : ℤ, (∃ k, n = 2 * k) → (∃ k, n = 2 * k ∧ n % 2 = 0)) ↔
  ∃ n : ℤ, (∃ k, n = 2 * k) ∧ ¬ (n % 2 = 0) :=
by
  sorry

end negation_of_divisible_by_2_even_l53_5363


namespace total_swimming_hours_over_4_weeks_l53_5332

def weekday_swimming_per_day : ℕ := 2  -- Percy swims 2 hours per weekday
def weekday_days_per_week : ℕ := 5     -- Percy swims for 5 days a week
def weekend_swimming_per_week : ℕ := 3 -- Percy swims 3 hours on the weekend
def weeks : ℕ := 4                     -- The number of weeks is 4

-- Define the total swimming hours over 4 weeks
theorem total_swimming_hours_over_4_weeks :
  weekday_swimming_per_day * weekday_days_per_week * weeks + weekend_swimming_per_week * weeks = 52 :=
by
  sorry

end total_swimming_hours_over_4_weeks_l53_5332


namespace opposite_of_2_is_minus_2_l53_5384

-- Define the opposite function
def opposite (x : ℤ) : ℤ := -x

-- Assert the theorem to prove that the opposite of 2 is -2
theorem opposite_of_2_is_minus_2 : opposite 2 = -2 := by
  sorry -- Placeholder for the proof

end opposite_of_2_is_minus_2_l53_5384


namespace prize_winner_is_B_l53_5349

-- Define the possible entries winning the prize
inductive Prize
| A
| B
| C
| D

open Prize

-- Define each student's predictions
def A_pred (prize : Prize) : Prop := prize = C ∨ prize = D
def B_pred (prize : Prize) : Prop := prize = B
def C_pred (prize : Prize) : Prop := prize ≠ A ∧ prize ≠ D
def D_pred (prize : Prize) : Prop := prize = C

-- Define the main theorem to prove
theorem prize_winner_is_B (prize : Prize) :
  (A_pred prize ∧ B_pred prize ∧ ¬C_pred prize ∧ ¬D_pred prize) ∨
  (A_pred prize ∧ ¬B_pred prize ∧ C_pred prize ∧ ¬D_pred prize) ∨
  (¬A_pred prize ∧ B_pred prize ∧ C_pred prize ∧ ¬D_pred prize) ∨
  (¬A_pred prize ∧ ¬B_pred prize ∧ C_pred prize ∧ D_pred prize) →
  prize = B :=
sorry

end prize_winner_is_B_l53_5349


namespace topics_assignment_l53_5366

theorem topics_assignment (students groups arrangements : ℕ) (h1 : students = 6) (h2 : groups = 3) (h3 : arrangements = 90) :
  let T := arrangements / (students * (students - 1) / 2 * (4 * 3 / 2 * 1))
  T = 1 :=
by
  sorry

end topics_assignment_l53_5366


namespace polynomial_remainder_l53_5359

theorem polynomial_remainder (P : Polynomial ℝ) (H1 : P.eval 1 = 2) (H2 : P.eval 2 = 1) :
  ∃ Q : Polynomial ℝ, P = Q * (Polynomial.X - 1) * (Polynomial.X - 2) + (3 - Polynomial.X) :=
by
  sorry

end polynomial_remainder_l53_5359


namespace ratio_value_l53_5340

theorem ratio_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
(h1 : (y + 1) / (x - z + 1) = (x + y + 2) / (z + 2)) 
(h2 : (x + y + 2) / (z + 2) = (x + 1) / (y + 1)) :
  (x + 1) / (y + 1) = 2 :=
by
  sorry

end ratio_value_l53_5340


namespace second_cyclist_speed_l53_5391

-- Definitions of the given conditions
def total_course_length : ℝ := 45
def first_cyclist_speed : ℝ := 14
def meeting_time : ℝ := 1.5

-- Lean 4 statement for the proof problem
theorem second_cyclist_speed : 
  ∃ v : ℝ, first_cyclist_speed * meeting_time + v * meeting_time = total_course_length → v = 16 := 
by 
  sorry

end second_cyclist_speed_l53_5391


namespace abs_inequality_solution_l53_5336

theorem abs_inequality_solution :
  {x : ℝ | |x - 2| + |x + 3| < 7} = {x : ℝ | -4 < x ∧ x < 3} :=
sorry

end abs_inequality_solution_l53_5336


namespace longest_side_of_triangle_l53_5307

theorem longest_side_of_triangle :
  ∃ y : ℚ, 6 + (y + 3) + (3 * y - 2) = 40 ∧ max (6 : ℚ) (max (y + 3) (3 * y - 2)) = 91 / 4 :=
by
  sorry

end longest_side_of_triangle_l53_5307


namespace infinite_3_stratum_numbers_l53_5329

-- Condition for 3-stratum number
def is_3_stratum_number (n : ℕ) : Prop :=
  ∃ (A B C : Finset ℕ), A ∪ B ∪ C = (Finset.range (n + 1)).filter (λ x => n % x = 0) ∧
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id

-- Part (a): Find a 3-stratum number
example : is_3_stratum_number 120 := sorry

-- Part (b): Prove there are infinitely many 3-stratum numbers
theorem infinite_3_stratum_numbers : ∃ (f : ℕ → ℕ), ∀ n, is_3_stratum_number (f n) := sorry

end infinite_3_stratum_numbers_l53_5329


namespace blue_sequins_per_row_l53_5337

theorem blue_sequins_per_row : 
  ∀ (B : ℕ),
  (6 * B) + (5 * 12) + (9 * 6) = 162 → B = 8 :=
by
  intro B
  sorry

end blue_sequins_per_row_l53_5337


namespace single_cone_scoops_l53_5398

theorem single_cone_scoops (banana_split_scoops : ℕ) (waffle_bowl_scoops : ℕ) (single_cone_scoops : ℕ) (double_cone_scoops : ℕ)
  (h1 : banana_split_scoops = 3 * single_cone_scoops)
  (h2 : waffle_bowl_scoops = banana_split_scoops + 1)
  (h3 : double_cone_scoops = 2 * single_cone_scoops)
  (h4 : single_cone_scoops + banana_split_scoops + waffle_bowl_scoops + double_cone_scoops = 10) :
  single_cone_scoops = 1 :=
by
  sorry

end single_cone_scoops_l53_5398


namespace ratio_of_w_y_l53_5300

variable (w x y z : ℚ)

theorem ratio_of_w_y (h1 : w / x = 4 / 3)
                     (h2 : y / z = 3 / 2)
                     (h3 : z / x = 1 / 3) :
                     w / y = 8 / 3 := by
  sorry

end ratio_of_w_y_l53_5300


namespace solve_equation_one_solve_equation_two_l53_5394

theorem solve_equation_one (x : ℝ) : (x - 3) ^ 2 - 4 = 0 ↔ x = 5 ∨ x = 1 := sorry

theorem solve_equation_two (x : ℝ) : (x + 2) ^ 2 - 2 * (x + 2) = 3 ↔ x = 1 ∨ x = -1 := sorry

end solve_equation_one_solve_equation_two_l53_5394


namespace cp_of_apple_l53_5339

theorem cp_of_apple (SP : ℝ) (hSP : SP = 17) (loss_fraction : ℝ) (h_loss_fraction : loss_fraction = 1 / 6) : 
  ∃ CP : ℝ, CP = 20.4 ∧ SP = CP - loss_fraction * CP :=
by
  -- Placeholder for proof
  sorry

end cp_of_apple_l53_5339


namespace point_value_of_other_questions_l53_5352

theorem point_value_of_other_questions (x y p : ℕ) 
  (h1 : x = 10) 
  (h2 : x + y = 40) 
  (h3 : 40 + 30 * p = 100) : 
  p = 2 := 
  sorry

end point_value_of_other_questions_l53_5352


namespace river_current_speed_l53_5335

/-- A man rows 18 miles upstream in three hours more time than it takes him to row 
the same distance downstream. If he halves his usual rowing rate, the time upstream 
becomes only two hours more than the time downstream. Prove that the speed of 
the river's current is 2 miles per hour. -/
theorem river_current_speed (r w : ℝ) 
    (h1 : 18 / (r - w) - 18 / (r + w) = 3)
    (h2 : 18 / (r / 2 - w) - 18 / (r / 2 + w) = 2) : 
    w = 2 := 
sorry

end river_current_speed_l53_5335


namespace James_total_area_l53_5321

theorem James_total_area :
  let initial_length := 13
  let initial_width := 18
  let increased_length := initial_length + 2
  let increased_width := initial_width + 2
  let single_room_area := increased_length * increased_width
  let four_rooms_area := 4 * single_room_area
  let larger_room_area := 2 * single_room_area
  let total_area := four_rooms_area + larger_room_area
  total_area = 1800 :=
by
  let initial_length := 13
  let initial_width := 18
  let increased_length := initial_length + 2
  let increased_width := initial_width + 2
  let single_room_area := increased_length * increased_width
  let four_rooms_area := 4 * single_room_area
  let larger_room_area := 2 * single_room_area
  let total_area := four_rooms_area + larger_room_area
  have h : total_area = 1800 := by sorry
  exact h

end James_total_area_l53_5321


namespace number_of_solutions_l53_5350

theorem number_of_solutions :
  ∃ sols: Finset (ℕ × ℕ), (∀ (x y : ℕ), (x, y) ∈ sols ↔ x^2 + y^2 + 2*x*y - 1988*x - 1988*y = 1989 ∧ x > 0 ∧ y > 0)
  ∧ sols.card = 1988 :=
by
  sorry

end number_of_solutions_l53_5350


namespace athena_total_spent_l53_5368

noncomputable def cost_sandwiches := 4 * 3.25
noncomputable def cost_fruit_drinks := 3 * 2.75
noncomputable def cost_cookies := 6 * 1.50
noncomputable def cost_chips := 2 * 1.85

noncomputable def total_cost := cost_sandwiches + cost_fruit_drinks + cost_cookies + cost_chips

theorem athena_total_spent : total_cost = 33.95 := 
by 
  simp [cost_sandwiches, cost_fruit_drinks, cost_cookies, cost_chips, total_cost]
  sorry

end athena_total_spent_l53_5368


namespace list_price_of_article_l53_5392

theorem list_price_of_article 
(paid_price : ℝ) 
(first_discount second_discount : ℝ)
(list_price : ℝ)
(h_paid_price : paid_price = 59.22)
(h_first_discount : first_discount = 0.10)
(h_second_discount : second_discount = 0.06000000000000002)
(h_final_price : paid_price = (1 - first_discount) * (1 - second_discount) * list_price) :
  list_price = 70 := 
by
  sorry

end list_price_of_article_l53_5392


namespace perpendicular_vectors_implies_k_eq_2_l53_5309

variable (k : ℝ)
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, k)

theorem perpendicular_vectors_implies_k_eq_2 (h : (2 : ℝ) * (-1 : ℝ) + (1 : ℝ) * k = 0) : k = 2 := by
  sorry

end perpendicular_vectors_implies_k_eq_2_l53_5309


namespace tan_of_acute_angle_and_cos_pi_add_alpha_l53_5387

theorem tan_of_acute_angle_and_cos_pi_add_alpha (α : ℝ) (h1 : 0 < α ∧ α < π/2)
  (h2 : Real.cos (π + α) = -Real.sqrt (3) / 2) : 
  Real.tan α = Real.sqrt (3) / 3 :=
by
  sorry

end tan_of_acute_angle_and_cos_pi_add_alpha_l53_5387


namespace necessary_but_not_sufficient_condition_l53_5325

variable (a b : ℝ) (lna lnb : ℝ)

theorem necessary_but_not_sufficient_condition (h1 : lna < lnb) (h2 : lna = Real.log a) (h3 : lnb = Real.log b) :
  (a > 0 ∧ b > 0 ∧ a < b ∧ a ^ 3 < b ^ 3) ∧ ¬(a ^ 3 < b ^ 3 → 0 < a ∧ a < b ∧ 0 < b) :=
by {
  sorry
}

end necessary_but_not_sufficient_condition_l53_5325


namespace sister_sandcastle_height_l53_5334

theorem sister_sandcastle_height (miki_height : ℝ)
                                (height_diff : ℝ)
                                (h_miki : miki_height = 0.8333333333333334)
                                (h_diff : height_diff = 0.3333333333333333) :
  miki_height - height_diff = 0.5 :=
by
  sorry

end sister_sandcastle_height_l53_5334


namespace both_firms_participate_l53_5369

-- Definitions based on the conditions
variable (V IC : ℝ) (α : ℝ)
-- Assumptions
variable (hα : 0 < α ∧ α < 1)
-- Part (a) condition transformation
def participation_condition := α * (1 - α) * V + 0.5 * α^2 * V ≥ IC

-- Given values for part (b)
def V_value : ℝ := 24
def α_value : ℝ := 0.5
def IC_value : ℝ := 7

-- New definitions for given values
def part_b_condition := (α_value * (1 - α_value) * V_value + 0.5 * α_value^2 * V_value) ≥ IC_value

-- Profits for part (c) comparison
def profit_when_both := 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
def profit_when_one := α * V - IC

-- Proof problem statement in Lean 4
theorem both_firms_participate (hV : V = 24) (hα : α = 0.5) (hIC : IC = 7) :
    (α * (1 - α) * V + 0.5 * α^2 * V) ≥ IC ∧ profit_when_both V alpha IC > profit_when_one V α IC := by
  sorry

end both_firms_participate_l53_5369


namespace section_b_students_can_be_any_nonnegative_integer_l53_5306

def section_a_students := 36
def avg_weight_section_a := 30
def avg_weight_section_b := 30
def avg_weight_whole_class := 30

theorem section_b_students_can_be_any_nonnegative_integer (x : ℕ) :
  let total_weight_section_a := section_a_students * avg_weight_section_a
  let total_weight_section_b := x * avg_weight_section_b
  let total_weight_whole_class := (section_a_students + x) * avg_weight_whole_class
  (total_weight_section_a + total_weight_section_b = total_weight_whole_class) :=
by 
  sorry

end section_b_students_can_be_any_nonnegative_integer_l53_5306


namespace steve_fraction_of_skylar_l53_5324

variables (S : ℤ) (Stacy Skylar Steve : ℤ)

-- Given conditions
axiom h1 : 32 = 3 * Steve + 2 -- Stacy's berries = 2 + 3 * Steve's berries
axiom h2 : Skylar = 20        -- Skylar has 20 berries
axiom h3 : Stacy = 32         -- Stacy has 32 berries

-- Final goal
theorem steve_fraction_of_skylar (h1: 32 = 3 * Steve + 2) (h2: 20 = Skylar) (h3: Stacy = 32) :
  Steve = Skylar / 2 := 
sorry

end steve_fraction_of_skylar_l53_5324


namespace quadratic_function_coefficient_nonzero_l53_5381

theorem quadratic_function_coefficient_nonzero (m : ℝ) :
  (y = (m + 2) * x * x + m) ↔ (m ≠ -2 ∧ (m^2 + m - 2 = 0) → m = 1) := by
  sorry

end quadratic_function_coefficient_nonzero_l53_5381


namespace combined_gross_profit_correct_l53_5311

def calculate_final_selling_price (initial_price : ℝ) (markup : ℝ) (discounts : List ℝ) : ℝ :=
  let marked_up_price := initial_price * (1 + markup)
  let final_price := List.foldl (λ price discount => price * (1 - discount)) marked_up_price discounts
  final_price

def calculate_gross_profit (initial_price : ℝ) (markup : ℝ) (discounts : List ℝ) : ℝ :=
  calculate_final_selling_price initial_price markup discounts - initial_price

noncomputable def combined_gross_profit : ℝ :=
  let earrings_gross_profit := calculate_gross_profit 240 0.25 [0.15]
  let bracelet_gross_profit := calculate_gross_profit 360 0.30 [0.10, 0.05]
  let necklace_gross_profit := calculate_gross_profit 480 0.40 [0.20, 0.05]
  let ring_gross_profit := calculate_gross_profit 600 0.35 [0.10, 0.05, 0.02]
  let pendant_gross_profit := calculate_gross_profit 720 0.50 [0.20, 0.03, 0.07]
  earrings_gross_profit + bracelet_gross_profit + necklace_gross_profit + ring_gross_profit + pendant_gross_profit

theorem combined_gross_profit_correct : combined_gross_profit = 224.97 :=
  by
  sorry

end combined_gross_profit_correct_l53_5311


namespace max_books_borrowed_l53_5356

theorem max_books_borrowed (students_total : ℕ) (students_no_books : ℕ) 
  (students_1_book : ℕ) (students_2_books : ℕ) (students_at_least_3_books : ℕ) 
  (average_books_per_student : ℝ) (H1 : students_total = 60) 
  (H2 : students_no_books = 4) 
  (H3 : students_1_book = 18) 
  (H4 : students_2_books = 20) 
  (H5 : students_at_least_3_books = students_total - (students_no_books + students_1_book + students_2_books)) 
  (H6 : average_books_per_student = 2.5) : 
  ∃ max_books : ℕ, max_books = 41 :=
by
  sorry

end max_books_borrowed_l53_5356


namespace g_max_value_l53_5378

def g (n : ℕ) : ℕ :=
if n < 15 then n + 15 else g (n - 7)

theorem g_max_value : ∃ N : ℕ, ∀ n : ℕ, g n ≤ N ∧ N = 29 := 
by 
  sorry

end g_max_value_l53_5378


namespace incorrect_directions_of_opening_l53_5317

-- Define the functions
def f (x : ℝ) : ℝ := 2 * (x - 3)^2
def g (x : ℝ) : ℝ := -2 * (x - 3)^2

-- The theorem (statement) to prove
theorem incorrect_directions_of_opening :
  ¬(∀ x, (f x > 0 ∧ g x > 0) ∨ (f x < 0 ∧ g x < 0)) :=
sorry

end incorrect_directions_of_opening_l53_5317


namespace ernie_circles_l53_5362

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes ali_circles : ℕ)
  (h1: boxes_per_circle_ali = 8)
  (h2: boxes_per_circle_ernie = 10)
  (h3: total_boxes = 80)
  (h4: ali_circles = 5) : 
  (total_boxes - ali_circles * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l53_5362


namespace horse_saddle_ratio_l53_5343

variable (H S : ℝ)
variable (m : ℝ)
variable (total_value saddle_value : ℝ)

theorem horse_saddle_ratio :
  total_value = 100 ∧ saddle_value = 12.5 ∧ H = m * saddle_value ∧ H + saddle_value = total_value → m = 7 :=
by
  sorry

end horse_saddle_ratio_l53_5343


namespace prime_ge_5_divisible_by_12_l53_5390

theorem prime_ge_5_divisible_by_12 (p : ℕ) (hp1 : p ≥ 5) (hp2 : Nat.Prime p) : 12 ∣ p^2 - 1 :=
by
  sorry

end prime_ge_5_divisible_by_12_l53_5390


namespace shift_parabola_l53_5372

theorem shift_parabola (x : ℝ) : 
  let y := -x^2
  let y_shifted_left := -((x + 3)^2)
  let y_shifted := y_shifted_left + 5
  y_shifted = -(x + 3)^2 + 5 := 
by {
  sorry
}

end shift_parabola_l53_5372


namespace bird_families_flew_away_l53_5319

def initial_families : ℕ := 41
def left_families : ℕ := 14

theorem bird_families_flew_away :
  initial_families - left_families = 27 :=
by
  -- This is a placeholder for the proof
  sorry

end bird_families_flew_away_l53_5319


namespace principal_amount_l53_5376

theorem principal_amount (P R : ℝ) : 
  (P + P * R * 2 / 100 = 850) ∧ (P + P * R * 7 / 100 = 1020) → P = 782 :=
by
  sorry

end principal_amount_l53_5376


namespace probability_at_least_eight_stayed_correct_l53_5385

noncomputable def probability_at_least_eight_stayed (n : ℕ) (c : ℕ) (p : ℚ) : ℚ :=
  let certain_count := c
  let unsure_count := n - c
  let k := 3
  let prob_eight := 
    (Nat.choose unsure_count k : ℚ) * (p^k) * ((1 - p)^(unsure_count - k))
  let prob_nine := p^unsure_count
  prob_eight + prob_nine

theorem probability_at_least_eight_stayed_correct :
  probability_at_least_eight_stayed 9 5 (3/7) = 513 / 2401 :=
by
  sorry

end probability_at_least_eight_stayed_correct_l53_5385


namespace arithmetic_geometric_mean_inequality_l53_5388

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (A : ℝ) (G : ℝ)
  (hA : A = (a + b) / 2) (hG : G = Real.sqrt (a * b)) : A ≥ G :=
by
  sorry

end arithmetic_geometric_mean_inequality_l53_5388


namespace prove_R36_div_R6_minus_R3_l53_5304

noncomputable def R (k : ℕ) : ℤ := (10^k - 1) / 9

theorem prove_R36_div_R6_minus_R3 :
  (R 36 / R 6) - R 3 = 100000100000100000100000100000099989 := sorry

end prove_R36_div_R6_minus_R3_l53_5304


namespace arithmetic_progression_sum_l53_5345

noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_progression_sum
    (a1 d : ℤ)
    (h : a 9 a1 d = a 12 a1 d / 2 + 3) :
  S 11 a1 d = 66 := 
by 
  sorry

end arithmetic_progression_sum_l53_5345


namespace pure_imaginary_solution_l53_5393

theorem pure_imaginary_solution (m : ℝ) (z : ℂ)
  (h1 : z = (m^2 - 1) + (m - 1) * I)
  (h2 : z.re = 0) : m = -1 :=
sorry

end pure_imaginary_solution_l53_5393


namespace domain_of_f_l53_5316

theorem domain_of_f (c : ℝ) :
  (∀ x : ℝ, -7 * x^2 + 5 * x + c ≠ 0) ↔ c < -25 / 28 :=
by
  sorry

end domain_of_f_l53_5316


namespace geo_seq_arith_seq_l53_5375

theorem geo_seq_arith_seq (a_n : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h_gp : ∀ n, a_n (n+1) = a_n n * q)
  (h_pos : ∀ n, a_n n > 0) (h_arith : a_n 4 - a_n 3 = a_n 5 - a_n 4) 
  (hq_pos : q > 0) (hq_neq1 : q ≠ 1) :
  S 6 / S 3 = 2 := by
  sorry

end geo_seq_arith_seq_l53_5375


namespace work_required_to_lift_satellite_l53_5303

noncomputable def satellite_lifting_work (m H R3 g : ℝ) : ℝ :=
  m * g * R3^2 * ((1 / R3) - (1 / (R3 + H)))

theorem work_required_to_lift_satellite :
  satellite_lifting_work (7.0 * 10^3) (200 * 10^3) (6380 * 10^3) 10 = 13574468085 :=
by sorry

end work_required_to_lift_satellite_l53_5303


namespace inequality_condition_necessary_not_sufficient_l53_5399

theorem inequality_condition (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  (1 / a > 1 / b) :=
by
  sorry

theorem necessary_not_sufficient (a b : ℝ) :
  (1 / a > 1 / b → 0 < a ∧ a < b) ∧ ¬ (0 < a ∧ a < b → 1 / a > 1 / b) :=
by
  sorry

end inequality_condition_necessary_not_sufficient_l53_5399


namespace islander_C_response_l53_5344

-- Define the types and assumptions
variables {Person : Type} (is_knight : Person → Prop) (is_liar : Person → Prop)
variables (A B C : Person)

-- Conditions from the problem
axiom A_statement : (is_liar A) ↔ (is_knight B = false ∧ is_knight C = false)
axiom B_statement : (is_knight B) ↔ (is_knight A ↔ ¬ is_knight C)

-- Conclusion we want to prove
theorem islander_C_response : is_knight C → (is_knight A ↔ ¬ is_knight C) := sorry

end islander_C_response_l53_5344


namespace grade_assignment_ways_l53_5318

-- Define the number of students and the number of grade choices
def students : ℕ := 12
def grade_choices : ℕ := 4

-- Define the number of ways to assign grades
def num_ways_to_assign_grades : ℕ := grade_choices ^ students

-- Prove that the number of ways to assign grades is 16777216
theorem grade_assignment_ways :
  num_ways_to_assign_grades = 16777216 :=
by
  -- Calculation validation omitted (proof step)
  sorry

end grade_assignment_ways_l53_5318


namespace evaluate_at_two_l53_5302

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem evaluate_at_two : f (g 2) + g (f 2) = 38 / 7 := by
  sorry

end evaluate_at_two_l53_5302


namespace prob_two_red_balls_l53_5351

-- Define the initial conditions for the balls in the bag
def red_balls : ℕ := 5
def blue_balls : ℕ := 6
def green_balls : ℕ := 2
def total_balls : ℕ := red_balls + blue_balls + green_balls

-- Define the probability of picking a red ball first
def prob_red1 : ℚ := red_balls / total_balls

-- Define the remaining number of balls and the probability of picking a red ball second
def remaining_red_balls : ℕ := red_balls - 1
def remaining_total_balls : ℕ := total_balls - 1
def prob_red2 : ℚ := remaining_red_balls / remaining_total_balls

-- Define the combined probability of both events
def prob_both_red : ℚ := prob_red1 * prob_red2

-- Statement of the theorem to be proved
theorem prob_two_red_balls : prob_both_red = 5 / 39 := by
  sorry

end prob_two_red_balls_l53_5351


namespace Cheerful_snakes_not_Green_l53_5354

variables {Snake : Type} (snakes : Finset Snake)
variable (Cheerful Green CanSing CanMultiply : Snake → Prop)

-- Conditions
axiom Cheerful_impl_CanSing : ∀ s, Cheerful s → CanSing s
axiom Green_impl_not_CanMultiply : ∀ s, Green s → ¬ CanMultiply s
axiom not_CanMultiply_impl_not_CanSing : ∀ s, ¬ CanMultiply s → ¬ CanSing s

-- Question
theorem Cheerful_snakes_not_Green : ∀ s, Cheerful s → ¬ Green s :=
by sorry

end Cheerful_snakes_not_Green_l53_5354


namespace winnie_lollipops_remainder_l53_5323

theorem winnie_lollipops_remainder :
  ∃ (k : ℕ), k = 505 % 14 ∧ k = 1 :=
by
  sorry

end winnie_lollipops_remainder_l53_5323


namespace weekly_tax_percentage_is_zero_l53_5313

variables (daily_expense : ℕ) (daily_revenue_fries : ℕ) (daily_revenue_poutine : ℕ) (weekly_net_income : ℕ)

def weekly_expense := daily_expense * 7
def weekly_revenue := daily_revenue_fries * 7 + daily_revenue_poutine * 7
def weekly_total_income := weekly_net_income + weekly_expense
def weekly_tax := weekly_total_income - weekly_revenue

theorem weekly_tax_percentage_is_zero
  (h1 : daily_expense = 10)
  (h2 : daily_revenue_fries = 12)
  (h3 : daily_revenue_poutine = 8)
  (h4 : weekly_net_income = 56) :
  weekly_tax = 0 :=
by sorry

end weekly_tax_percentage_is_zero_l53_5313


namespace find_f_zero_forall_x_f_pos_solve_inequality_l53_5365

variable {f : ℝ → ℝ}

-- Conditions
axiom condition_1 : ∀ x, x > 0 → f x > 1
axiom condition_2 : ∀ x y, f (x + y) = f x * f y
axiom condition_3 : f 2 = 3

-- Questions rewritten as Lean theorems

theorem find_f_zero : f 0 = 1 := sorry

theorem forall_x_f_pos : ∀ x, f x > 0 := sorry

theorem solve_inequality : ∀ x, f (7 + 2 * x) > 9 ↔ x > -3 / 2 := sorry

end find_f_zero_forall_x_f_pos_solve_inequality_l53_5365


namespace solve_for_a_l53_5330

theorem solve_for_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end solve_for_a_l53_5330


namespace egg_rolls_total_l53_5373

def total_egg_rolls (omar_rolls : ℕ) (karen_rolls : ℕ) : ℕ :=
  omar_rolls + karen_rolls

theorem egg_rolls_total :
  total_egg_rolls 219 229 = 448 :=
by
  sorry

end egg_rolls_total_l53_5373


namespace geometric_sequence_common_ratio_l53_5361

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = -4) : 
  q = -2 := 
by
  sorry

end geometric_sequence_common_ratio_l53_5361


namespace max_xy_l53_5353

theorem max_xy (x y : ℝ) (hxy_pos : x > 0 ∧ y > 0) (h : 5 * x + 8 * y = 65) : 
  xy ≤ 25 :=
by
  sorry

end max_xy_l53_5353


namespace baker_additional_cakes_l53_5397

theorem baker_additional_cakes (X : ℕ) : 
  (62 + X) - 144 = 67 → X = 149 :=
by
  intro h
  sorry

end baker_additional_cakes_l53_5397


namespace triangle_angle_type_l53_5308

theorem triangle_angle_type (a b c R : ℝ) (hc_max : c ≥ a ∧ c ≥ b) :
  (a^2 + b^2 + c^2 - 8 * R^2 > 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2) ∧
  (a^2 + b^2 + c^2 - 8 * R^2 = 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ (α = π / 2 ∨ β = π / 2 ∨ γ = π / 2)) ∧
  (a^2 + b^2 + c^2 - 8 * R^2 < 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2)) :=
sorry

end triangle_angle_type_l53_5308


namespace brick_length_proof_l53_5386

-- Defining relevant parameters and conditions
def width_of_brick : ℝ := 10 -- width in cm
def height_of_brick : ℝ := 7.5 -- height in cm
def wall_length : ℝ := 26 -- length in m
def wall_width : ℝ := 2 -- width in m
def wall_height : ℝ := 0.75 -- height in m
def num_bricks : ℝ := 26000 

-- Defining known volumes for conversion
def volume_of_wall_m3 : ℝ := wall_length * wall_width * wall_height
def volume_of_wall_cm3 : ℝ := volume_of_wall_m3 * 1000000 -- converting m³ to cm³

-- Volume of one brick given the unknown length L
def volume_of_one_brick (L : ℝ) : ℝ := L * width_of_brick * height_of_brick

-- Total volume of bricks is the volume of one brick times the number of bricks
def total_volume_of_bricks (L : ℝ) : ℝ := volume_of_one_brick L * num_bricks

-- The length of the brick is found by equating the total volume of bricks to the volume of the wall
theorem brick_length_proof : ∃ L : ℝ, total_volume_of_bricks L = volume_of_wall_cm3 ∧ L = 20 :=
by
  existsi 20
  sorry

end brick_length_proof_l53_5386


namespace solve_linear_system_l53_5315

theorem solve_linear_system :
  ∃ (x1 x2 x3 : ℚ), 
  (2 * x1 + 5 * x2 - 4 * x3 = 8) ∧ 
  (3 * x1 + 15 * x2 - 9 * x3 = 5) ∧ 
  (5 * x1 + 5 * x2 - 7 * x3 = 27) ∧
  (x1 = 19 / 3 + x3) ∧ 
  (x2 = -14 / 15 + 2 / 5 * x3) := 
by 
  sorry

end solve_linear_system_l53_5315


namespace desserts_brought_by_mom_l53_5364

-- Definitions for the number of each type of dessert
def num_coconut := 1
def num_meringues := 2
def num_caramel := 7

-- Conditions from the problem as definitions
def total_desserts := num_coconut + num_meringues + num_caramel = 10
def fewer_coconut_than_meringues := num_coconut < num_meringues
def most_caramel := num_caramel > num_meringues
def josef_jakub_condition := (num_coconut + num_meringues + num_caramel) - (4 * 2) = 1

-- We need to prove the answer based on these conditions
theorem desserts_brought_by_mom :
  total_desserts ∧ fewer_coconut_than_meringues ∧ most_caramel ∧ josef_jakub_condition → 
  num_coconut = 1 ∧ num_meringues = 2 ∧ num_caramel = 7 :=
by sorry

end desserts_brought_by_mom_l53_5364


namespace problem1_problem2_l53_5396

variable (α : ℝ) (tan_alpha_eq_one_over_three : Real.tan α = 1 / 3)

-- For the first proof problem
theorem problem1 : (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 :=
by sorry

-- For the second proof problem
theorem problem2 : Real.cos α ^ 2 - Real.sin (2 * α) = 3 / 10 :=
by sorry

end problem1_problem2_l53_5396


namespace cube_volume_increase_l53_5301

theorem cube_volume_increase (s : ℝ) (surface_area : ℝ) 
  (h1 : surface_area = 6 * s^2) (h2 : surface_area = 864) : 
  (1.5 * s)^3 = 5832 :=
by
  sorry

end cube_volume_increase_l53_5301


namespace quadrilateral_perimeter_l53_5370

theorem quadrilateral_perimeter (a b : ℝ) (h₁ : a = 10) (h₂ : b = 15)
  (h₃ : ∀ (ABD BCD ABC ACD : ℝ), ABD = BCD ∧ ABC = ACD) : a + a + b + b = 50 :=
by
  rw [h₁, h₂]
  linarith


end quadrilateral_perimeter_l53_5370


namespace smallest_x_condition_l53_5310

theorem smallest_x_condition (x : ℕ) : (∃ x > 0, (3 * x + 28)^2 % 53 = 0) -> x = 26 := 
by
  sorry

end smallest_x_condition_l53_5310


namespace new_average_doubled_l53_5326

theorem new_average_doubled (n : ℕ) (avg : ℝ) (h1 : n = 12) (h2 : avg = 50) :
  2 * avg = 100 := by
sorry

end new_average_doubled_l53_5326


namespace correct_answers_count_l53_5383

theorem correct_answers_count (total_questions correct_pts incorrect_pts final_score : ℤ)
  (h1 : total_questions = 26)
  (h2 : correct_pts = 8)
  (h3 : incorrect_pts = -5)
  (h4 : final_score = 0) :
  ∃ c i : ℤ, c + i = total_questions ∧ correct_pts * c + incorrect_pts * i = final_score ∧ c = 10 :=
by
  use 10, (26 - 10)
  simp
  sorry

end correct_answers_count_l53_5383


namespace population_increase_20th_century_l53_5347

theorem population_increase_20th_century (P : ℕ) :
  let population_mid_century := 3 * P
  let population_end_century := 12 * P
  (population_end_century - P) / P * 100 = 1100 :=
by
  sorry

end population_increase_20th_century_l53_5347


namespace rahim_books_second_shop_l53_5320

variable (x : ℕ)

-- Definitions of the problem's conditions
def total_cost : ℕ := 520 + 248
def total_books (x : ℕ) : ℕ := 42 + x
def average_price : ℕ := 12

-- The problem statement in Lean 4
theorem rahim_books_second_shop : x = 22 → total_cost / total_books x = average_price :=
  sorry

end rahim_books_second_shop_l53_5320


namespace remainder_when_divided_by_30_l53_5389

theorem remainder_when_divided_by_30 (x : ℤ) : 
  (4 + x) % 8 = 9 % 8 ∧
  (6 + x) % 27 = 4 % 27 ∧
  (8 + x) % 125 = 49 % 125 
  → x % 30 = 1 % 30 := by
  sorry

end remainder_when_divided_by_30_l53_5389


namespace solve_for_x_l53_5322

theorem solve_for_x (x : ℚ) : (2/3 : ℚ) - 1/4 = 1/x → x = 12/5 := 
by
  sorry

end solve_for_x_l53_5322


namespace rook_reaches_right_total_rook_reaches_right_seven_moves_l53_5314

-- Definition of the conditions for the problem
def rook_ways_total (n : Nat) :=
  2 ^ (n - 2)

def rook_ways_in_moves (n k : Nat) :=
  Nat.choose (n - 2) (k - 1)

-- Proof problem statements
theorem rook_reaches_right_total : rook_ways_total 30 = 2 ^ 28 := 
by sorry

theorem rook_reaches_right_seven_moves : rook_ways_in_moves 30 7 = Nat.choose 28 6 := 
by sorry

end rook_reaches_right_total_rook_reaches_right_seven_moves_l53_5314
