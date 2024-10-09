import Mathlib

namespace total_pears_picked_l220_22035

def mikes_pears : Nat := 8
def jasons_pears : Nat := 7
def freds_apples : Nat := 6

theorem total_pears_picked : (mikes_pears + jasons_pears) = 15 :=
by
  sorry

end total_pears_picked_l220_22035


namespace daughter_work_alone_12_days_l220_22070

/-- Given a man, his wife, and their daughter working together on a piece of work. The man can complete the work in 4 days, the wife in 6 days, and together with their daughter, they can complete it in 2 days. Prove that the daughter alone would take 12 days to complete the work. -/
theorem daughter_work_alone_12_days (h1 : (1/4 : ℝ) + (1/6) + D = 1/2) : D = 1/12 :=
by
  sorry

end daughter_work_alone_12_days_l220_22070


namespace red_tulips_for_smile_l220_22075

/-
Problem Statement:
Anna wants to plant red and yellow tulips in the shape of a smiley face. Given the following conditions:
1. Anna needs 8 red tulips for each eye.
2. She needs 9 times the number of red tulips in the smile to make the yellow background of the face.
3. The total number of tulips needed is 196.

Prove:
The number of red tulips needed for the smile is 18.
-/

-- Defining the conditions
def red_tulips_per_eye : Nat := 8
def total_tulips : Nat := 196
def yellow_multiplier : Nat := 9

-- Proving the number of red tulips for the smile
theorem red_tulips_for_smile (R : Nat) :
  2 * red_tulips_per_eye + R + yellow_multiplier * R = total_tulips → R = 18 :=
by
  sorry

end red_tulips_for_smile_l220_22075


namespace find_min_value_l220_22039

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1)

theorem find_min_value :
  (1 / (2 * a + 3 * b) + 1 / (2 * b + 3 * c) + 1 / (2 * c + 3 * a)) ≥ (9 / 5) :=
sorry

end find_min_value_l220_22039


namespace speed_A_correct_l220_22012

noncomputable def speed_A : ℝ :=
  200 / (19.99840012798976 * 60)

theorem speed_A_correct :
  speed_A = 0.16668 :=
sorry

end speed_A_correct_l220_22012


namespace intersection_on_circle_l220_22097

def parabola1 (X : ℝ) : ℝ := X^2 + X - 41
def parabola2 (Y : ℝ) : ℝ := Y^2 + Y - 40

theorem intersection_on_circle (X Y : ℝ) :
  parabola1 X = Y ∧ parabola2 Y = X → X^2 + Y^2 = 81 :=
by {
  sorry
}

end intersection_on_circle_l220_22097


namespace tan_alpha_sub_beta_l220_22023

theorem tan_alpha_sub_beta
  (α β : ℝ)
  (h1 : Real.tan (α + Real.pi / 5) = 2)
  (h2 : Real.tan (β - 4 * Real.pi / 5) = -3) :
  Real.tan (α - β) = -1 := 
sorry

end tan_alpha_sub_beta_l220_22023


namespace B_joined_with_54000_l220_22095

theorem B_joined_with_54000 :
  ∀ (x : ℕ),
    (36000 * 12) / (x * 4) = 2 → x = 54000 :=
by 
  intro x h
  sorry

end B_joined_with_54000_l220_22095


namespace value_of_expression_l220_22085

variable (x y : ℝ)

theorem value_of_expression (h1 : x + y = 6) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 228498 := by
  sorry

end value_of_expression_l220_22085


namespace meal_cost_with_tip_l220_22053

theorem meal_cost_with_tip 
  (cost_samosas : ℕ := 3 * 2)
  (cost_pakoras : ℕ := 4 * 3)
  (cost_lassi : ℕ := 2)
  (total_cost_before_tip := cost_samosas + cost_pakoras + cost_lassi)
  (tip : ℝ := 0.25 * total_cost_before_tip) :
  (total_cost_before_tip + tip = 25) :=
sorry

end meal_cost_with_tip_l220_22053


namespace arithmetic_seq_sum_l220_22008

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h2 : a 2 + a 5 + a 8 = 15) : a 3 + a 7 = 10 :=
sorry

end arithmetic_seq_sum_l220_22008


namespace find_n_l220_22043

theorem find_n : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 7615 [MOD 15] ∧ n = 10 := by
  use 10
  repeat { sorry }

end find_n_l220_22043


namespace quadratic_trinomial_bound_l220_22091

theorem quadratic_trinomial_bound (a b : ℤ) (f : ℝ → ℝ)
  (h_def : ∀ x : ℝ, f x = x^2 + a * x + b)
  (h_bound : ∀ x : ℝ, f x ≥ -9 / 10) :
  ∀ x : ℝ, f x ≥ -1 / 4 :=
sorry

end quadratic_trinomial_bound_l220_22091


namespace unique_bisecting_line_exists_l220_22071

noncomputable def triangle_area := 1 / 2 * 6 * 8
noncomputable def triangle_perimeter := 6 + 8 + 10

theorem unique_bisecting_line_exists :
  ∃ (line : ℝ → ℝ), 
    (∃ x y : ℝ, x + y = 12 ∧ x * y = 30 ∧ 
      1 / 2 * x * y * (24 / triangle_perimeter) = 12) ∧
    (∃ x' y' : ℝ, x' + y' = 12 ∧ x' * y' = 24 ∧ 
      1 / 2 * x' * y' * (24 / triangle_perimeter) = 12) ∧
    ((x = x' ∧ y = y') ∨ (x = y' ∧ y = x')) :=
sorry

end unique_bisecting_line_exists_l220_22071


namespace system_of_equations_is_B_l220_22045

-- Define the given conditions and correct answer
def condition1 (x y : ℝ) : Prop := 5 * x + y = 3
def condition2 (x y : ℝ) : Prop := x + 5 * y = 2
def correctAnswer (x y : ℝ) : Prop := 5 * x + y = 3 ∧ x + 5 * y = 2

theorem system_of_equations_is_B (x y : ℝ) : condition1 x y ∧ condition2 x y ↔ correctAnswer x y := by
  -- Proof goes here
  sorry

end system_of_equations_is_B_l220_22045


namespace roots_equal_when_m_l220_22040

noncomputable def equal_roots_condition (k n m : ℝ) : Prop :=
  1 + 4 * m^2 * k + 4 * m * n = 0

theorem roots_equal_when_m :
  equal_roots_condition 1 3 (-1.5 + Real.sqrt 2) ∧ 
  equal_roots_condition 1 3 (-1.5 - Real.sqrt 2) :=
by 
  sorry

end roots_equal_when_m_l220_22040


namespace tiling_rect_divisible_by_4_l220_22059

theorem tiling_rect_divisible_by_4 (m n : ℕ) (h : ∃ k l : ℕ, m = 4 * k ∧ n = 4 * l) : 
  (∃ a : ℕ, m = 4 * a) ∧ (∃ b : ℕ, n = 4 * b) :=
by 
  sorry

end tiling_rect_divisible_by_4_l220_22059


namespace part1_part2_l220_22055

noncomputable def f (x : ℝ) : ℝ :=
  abs (2 * x - 3) + abs (x - 5)

theorem part1 : { x : ℝ | f x ≥ 4 } = { x : ℝ | x ≥ 2 ∨ x ≤ 4 / 3 } :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x < a) ↔ a > 7 / 2 :=
by
  sorry

end part1_part2_l220_22055


namespace train_speed_correct_l220_22088

noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (time_seconds : ℝ) : ℝ :=
  (train_length + bridge_length) / time_seconds

theorem train_speed_correct :
  train_speed (400 : ℝ) (300 : ℝ) (45 : ℝ) = 700 / 45 :=
by
  sorry

end train_speed_correct_l220_22088


namespace charge_per_kilo_l220_22048

variable (x : ℝ)

theorem charge_per_kilo (h : 5 * x + 10 * x + 20 * x = 70) : x = 2 := by
  -- Proof goes here
  sorry

end charge_per_kilo_l220_22048


namespace A_alone_days_l220_22031

variable (r_A r_B r_C : ℝ)

-- Given conditions:
axiom cond1 : r_A + r_B = 1 / 3
axiom cond2 : r_B + r_C = 1 / 6
axiom cond3 : r_A + r_C = 4 / 15

-- Proposition stating the required proof, that A alone can do the job in 60/13 days:
theorem A_alone_days : r_A ≠ 0 → 1 / r_A = 60 / 13 :=
by
  intro h
  sorry

end A_alone_days_l220_22031


namespace fib_math_competition_l220_22020

theorem fib_math_competition :
  ∃ (n9 n8 n7 : ℕ), 
    n9 * 4 = n8 * 7 ∧ 
    n9 * 3 = n7 * 10 ∧ 
    n9 + n8 + n7 = 131 :=
sorry

end fib_math_competition_l220_22020


namespace pond_volume_extraction_l220_22002

/--
  Let length (l), width (w), and depth (h) be dimensions of a pond.
  Given:
  l = 20,
  w = 10,
  h = 5,
  Prove that the volume of the soil extracted from the pond is 1000 cubic meters.
-/
theorem pond_volume_extraction (l w h : ℕ) (hl : l = 20) (hw : w = 10) (hh : h = 5) :
  l * w * h = 1000 :=
  by
    sorry

end pond_volume_extraction_l220_22002


namespace ron_total_tax_l220_22007

def car_price : ℝ := 30000
def first_tier_level : ℝ := 10000
def first_tier_rate : ℝ := 0.25
def second_tier_rate : ℝ := 0.15

def first_tier_tax : ℝ := first_tier_level * first_tier_rate
def second_tier_tax : ℝ := (car_price - first_tier_level) * second_tier_rate
def total_tax : ℝ := first_tier_tax + second_tier_tax

theorem ron_total_tax : 
  total_tax = 5500 := by
  -- Proof will be provided here
  sorry

end ron_total_tax_l220_22007


namespace units_digit_p2_plus_3p_l220_22018

-- Define p
def p : ℕ := 2017^3 + 3^2017

-- Define the theorem to be proved
theorem units_digit_p2_plus_3p : (p^2 + 3^p) % 10 = 5 :=
by
  sorry -- Proof goes here

end units_digit_p2_plus_3p_l220_22018


namespace Linda_purchase_cost_l220_22030

def price_peanuts : ℝ := sorry
def price_berries : ℝ := sorry
def price_coconut : ℝ := sorry
def price_dates : ℝ := sorry

theorem Linda_purchase_cost:
  ∃ (p b c d : ℝ), 
    (p + b + c + d = 30) ∧ 
    (3 * p = d) ∧
    ((p + b) / 2 = c) ∧
    (b + c = 65 / 9) :=
sorry

end Linda_purchase_cost_l220_22030


namespace common_ratio_q_is_one_l220_22019

-- Define the geometric sequence {a_n}, and the third term a_3 and sum of first three terms S_3
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = a n * a 1

variables {a : ℕ → ℝ}
variable (q : ℝ)

-- Given conditions
axiom a_3 : a 3 = 3 / 2
axiom S_3 : a 1 * (1 + q + q^2) = 9 / 2

-- We need to prove q = 1
theorem common_ratio_q_is_one (h1 : is_geometric_sequence a) : q = 1 := sorry

end common_ratio_q_is_one_l220_22019


namespace middle_segment_proportion_l220_22069

theorem middle_segment_proportion (a b c : ℝ) (h_a : a = 1) (h_b : b = 3) :
  (a / c = c / b) → c = Real.sqrt 3 :=
by
  sorry

end middle_segment_proportion_l220_22069


namespace initial_tomato_count_l220_22089

variable (T : ℝ)
variable (H1 : T - (1 / 4 * T + 20 + 40) = 15)

theorem initial_tomato_count : T = 100 :=
by
  sorry

end initial_tomato_count_l220_22089


namespace speed_of_boat_in_still_water_l220_22006

-- Define a structure for the conditions
structure BoatConditions where
  V_b : ℝ    -- Speed of the boat in still water
  V_s : ℝ    -- Speed of the stream
  goes_along_stream : V_b + V_s = 11
  goes_against_stream : V_b - V_s = 5

-- Define the target theorem
theorem speed_of_boat_in_still_water (c : BoatConditions) : c.V_b = 8 :=
by
  sorry

end speed_of_boat_in_still_water_l220_22006


namespace math_problem_l220_22028

theorem math_problem :
  2537 + 240 * 3 / 60 - 347 = 2202 :=
by
  sorry

end math_problem_l220_22028


namespace geometric_sequence_b_l220_22063

theorem geometric_sequence_b (b : ℝ) (r : ℝ) (hb : b > 0)
  (h1 : 10 * r = b)
  (h2 : b * r = 10 / 9)
  (h3 : (10 / 9) * r = 10 / 81) :
  b = 10 :=
sorry

end geometric_sequence_b_l220_22063


namespace locus_of_Q_l220_22078

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2/a^2 + y^2/b^2 = 1

def A_vertice (a b : ℝ) (x y : ℝ) : Prop :=
  (x = a ∧ y = 0) ∨ (x = -a ∧ y = 0)

def chord_parallel_y_axis (x : ℝ) : Prop :=
  -- Assuming chord's x coordinate is given
  True

def lines_intersect_at_Q (a b Qx Qy : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse a b x y ∧
  A_vertice a b x y ∧
  chord_parallel_y_axis x ∧
  (
    ( (Qy - y) / (Qx - (-a)) = (Qy - 0) / (Qx - a) ) ∨ -- A'P slope-comp
    ( (Qy - (-y)) / (Qx - a) = (Qy - 0) / (Qx - (-a)) ) -- AP' slope-comp
  )

theorem locus_of_Q (a b Qx Qy : ℝ) :
  (lines_intersect_at_Q a b Qx Qy) →
  (Qx^2 / a^2 - Qy^2 / b^2 = 1) := by
  sorry

end locus_of_Q_l220_22078


namespace complement_of_M_is_34_l220_22003

open Set

noncomputable def U : Set ℝ := univ
def M : Set ℝ := {x | (x - 3) / (4 - x) < 0}
def complement_M (U : Set ℝ) (M : Set ℝ) : Set ℝ := U \ M

theorem complement_of_M_is_34 : complement_M U M = {x | 3 ≤ x ∧ x ≤ 4} := 
by sorry

end complement_of_M_is_34_l220_22003


namespace tim_score_in_math_l220_22005

def even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14]

def sum_even_numbers (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem tim_score_in_math : sum_even_numbers even_numbers = 56 := by
  -- Proof steps would be here
  sorry

end tim_score_in_math_l220_22005


namespace lice_checks_l220_22029

theorem lice_checks (t_first t_second t_third t_total t_per_check n_first n_second n_third n_total n_per_check n_kg : ℕ) 
 (h1 : t_first = 19 * t_per_check)
 (h2 : t_second = 20 * t_per_check)
 (h3 : t_third = 25 * t_per_check)
 (h4 : t_total = 3 * 60)
 (h5 : t_per_check = 2)
 (h6 : n_first = t_first / t_per_check)
 (h7 : n_second = t_second / t_per_check)
 (h8 : n_third = t_third / t_per_check)
 (h9 : n_total = (t_total - (t_first + t_second + t_third)) / t_per_check) :
 n_total = 26 :=
sorry

end lice_checks_l220_22029


namespace g_675_eq_42_l220_22082

theorem g_675_eq_42 
  (g : ℕ → ℕ) 
  (h_mul : ∀ x y : ℕ, x > 0 → y > 0 → g (x * y) = g x + g y) 
  (h_g15 : g 15 = 18) 
  (h_g45 : g 45 = 24) : g 675 = 42 :=
by
  sorry

end g_675_eq_42_l220_22082


namespace current_price_after_increase_and_decrease_l220_22024

-- Define constants and conditions
def initial_price_RAM : ℝ := 50
def percent_increase : ℝ := 0.30
def percent_decrease : ℝ := 0.20

-- Define intermediate and final values based on conditions
def increased_price_RAM : ℝ := initial_price_RAM * (1 + percent_increase)
def final_price_RAM : ℝ := increased_price_RAM * (1 - percent_decrease)

-- Theorem stating the final result
theorem current_price_after_increase_and_decrease 
  (init_price : ℝ) 
  (inc : ℝ) 
  (dec : ℝ) 
  (final_price : ℝ) :
  init_price = 50 ∧ inc = 0.30 ∧ dec = 0.20 → final_price = 52 := 
  sorry

end current_price_after_increase_and_decrease_l220_22024


namespace words_added_to_removed_ratio_l220_22036

-- Conditions in the problem
def Yvonnes_words : ℕ := 400
def Jannas_extra_words : ℕ := 150
def words_removed : ℕ := 20
def words_needed : ℕ := 1000 - 930

-- Definitions derived from the conditions
def Jannas_words : ℕ := Yvonnes_words + Jannas_extra_words
def total_words_before_editing : ℕ := Yvonnes_words + Jannas_words
def total_words_after_removal : ℕ := total_words_before_editing - words_removed
def words_added : ℕ := words_needed

-- The theorem we need to prove
theorem words_added_to_removed_ratio :
  (words_added : ℚ) / words_removed = 7 / 2 :=
sorry

end words_added_to_removed_ratio_l220_22036


namespace arithmetic_seq_a10_l220_22057

variable (a : ℕ → ℚ)
variable (S : ℕ → ℚ)
variable (d : ℚ := 1)

def is_arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

def sum_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem arithmetic_seq_a10 (h_seq : is_arithmetic_seq a d)
                          (h_sum : sum_first_n_terms a S)
                          (h_condition : S 8 = 4 * S 4) :
  a 10 = 19/2 := 
sorry

end arithmetic_seq_a10_l220_22057


namespace find_m_eq_4_l220_22052

theorem find_m_eq_4 (m : ℝ) (h₁ : ∃ (A B C : ℝ × ℝ), A = (m, -m+3) ∧ B = (2, m-1) ∧ C = (-1, 4)) (h₂ : (4 - (-m+3)) / (-1-m) = 3 * ((m-1) - 4) / (2 - (-1))) : m = 4 :=
sorry

end find_m_eq_4_l220_22052


namespace original_price_of_article_l220_22033

theorem original_price_of_article (selling_price : ℝ) (loss_percent : ℝ) (P : ℝ) 
  (h1 : selling_price = 450)
  (h2 : loss_percent = 25)
  : selling_price = (1 - loss_percent / 100) * P → P = 600 :=
by
  sorry

end original_price_of_article_l220_22033


namespace product_of_469111_and_9999_l220_22041

theorem product_of_469111_and_9999 : 469111 * 9999 = 4690418889 := 
by 
  sorry

end product_of_469111_and_9999_l220_22041


namespace time_to_pass_tree_l220_22072

noncomputable def length_of_train : ℝ := 275
noncomputable def speed_in_kmh : ℝ := 90
noncomputable def speed_in_m_per_s : ℝ := speed_in_kmh * (5 / 18)

theorem time_to_pass_tree : (length_of_train / speed_in_m_per_s) = 11 :=
by {
  sorry
}

end time_to_pass_tree_l220_22072


namespace evaluate_expression_l220_22050

theorem evaluate_expression (c d : ℝ) (h_c : c = 3) (h_d : d = 2) : 
  (c^2 + d + 1)^2 - (c^2 - d - 1)^2 = 80 := by 
  sorry

end evaluate_expression_l220_22050


namespace percentage_of_managers_l220_22077

theorem percentage_of_managers (P : ℝ) :
  (200 : ℝ) * (P / 100) - 99.99999999999991 = 0.98 * (200 - 99.99999999999991) →
  P = 99 := 
sorry

end percentage_of_managers_l220_22077


namespace find_Y_l220_22060

-- Definition of the problem.
def arithmetic_sequence (a d n : ℕ) : ℕ := a + d * (n - 1)

-- Conditions provided in the problem.
-- Conditions of the first row
def first_row (a₁ a₄ : ℕ) : Prop :=
  a₁ = 4 ∧ a₄ = 16

-- Conditions of the last row
def last_row (a₁' a₄' : ℕ) : Prop :=
  a₁' = 10 ∧ a₄' = 40

-- Value of Y (the second element of the second row from the second column)
def center_top_element (Y : ℕ) : Prop :=
  Y = 12

-- The theorem to prove.
theorem find_Y (a₁ a₄ a₁' a₄' Y : ℕ) (h1 : first_row a₁ a₄) (h2 : last_row a₁' a₄') (h3 : center_top_element Y) : Y = 12 := 
by 
  sorry -- proof to be provided.

end find_Y_l220_22060


namespace prob_first_given_defective_correct_l220_22068

-- Definitions from problem conditions
def first_box : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def second_box : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
def defective_first_box : Set ℕ := {1, 2, 3}
def defective_second_box : Set ℕ := {1, 2}

-- Probability values as defined
def prob_first_box : ℚ := 1 / 2
def prob_second_box : ℚ := 1 / 2
def prob_defective_given_first : ℚ := 3 / 10
def prob_defective_given_second : ℚ := 1 / 10

-- Calculation of total probability of defective component
def prob_defective : ℚ := (prob_first_box * prob_defective_given_first) + (prob_second_box * prob_defective_given_second)

-- Bayes' Theorem application to find the required probability
def prob_first_given_defective : ℚ := (prob_first_box * prob_defective_given_first) / prob_defective

-- Lean statement to verify the computed probability is as expected
theorem prob_first_given_defective_correct : prob_first_given_defective = 3 / 4 :=
by
  unfold prob_first_given_defective prob_defective
  sorry

end prob_first_given_defective_correct_l220_22068


namespace solve_congruence_y37_x3_11_l220_22098

theorem solve_congruence_y37_x3_11 (p : ℕ) (hp_pr : Nat.Prime p) (hp_le100 : p ≤ 100) : 
  ∃ (x y : ℕ), y^37 ≡ x^3 + 11 [MOD p] := 
sorry

end solve_congruence_y37_x3_11_l220_22098


namespace arithmetic_sequence_geometric_sequence_l220_22062

-- Arithmetic sequence proof problem
theorem arithmetic_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n - a (n - 1) = 2) :
  ∀ n, a n = 2 * n - 1 :=
by 
  sorry

-- Geometric sequence proof problem
theorem geometric_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n / a (n - 1) = 2) :
  ∀ n, a n = 2 ^ (n - 1) :=
by 
  sorry

end arithmetic_sequence_geometric_sequence_l220_22062


namespace bus_interval_duration_l220_22058

-- Definition of the conditions
def total_minutes : ℕ := 60
def total_buses : ℕ := 11
def intervals : ℕ := total_buses - 1

-- Theorem stating the interval between each bus departure
theorem bus_interval_duration : total_minutes / intervals = 6 := 
by
  -- The proof is omitted. 
  sorry

end bus_interval_duration_l220_22058


namespace number_of_non_empty_proper_subsets_of_A_l220_22080

noncomputable def A : Set ℤ := { x : ℤ | -1 < x ∧ x ≤ 2 }

theorem number_of_non_empty_proper_subsets_of_A : 
  (∃ (A : Set ℤ), A = { x : ℤ | -1 < x ∧ x ≤ 2 }) → 
  ∃ (n : ℕ), n = 6 := by
  sorry

end number_of_non_empty_proper_subsets_of_A_l220_22080


namespace fill_tank_without_leak_l220_22044

theorem fill_tank_without_leak (T : ℕ) : 
  (1 / T - 1 / 110 = 1 / 11) ↔ T = 10 :=
by 
  sorry

end fill_tank_without_leak_l220_22044


namespace intersection_A_B_l220_22073

noncomputable def set_A : Set ℝ := { x | 2 ≤ x ∧ x < 4 }
noncomputable def set_B : Set ℝ := { x | 3 ≤ x }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 3 ≤ x ∧ x < 4 } := 
sorry

end intersection_A_B_l220_22073


namespace find_income_l220_22049

-- Define the condition for savings
def savings_formula (income expenditure savings : ℝ) : Prop :=
  income - expenditure = savings

-- Define the ratio between income and expenditure
def ratio_condition (income expenditure : ℝ) : Prop :=
  income = 5 / 4 * expenditure

-- Given:
-- savings: Rs. 3400
-- We need to prove the income is Rs. 17000
theorem find_income (savings : ℝ) (income expenditure : ℝ) :
  savings_formula income expenditure savings →
  ratio_condition income expenditure →
  savings = 3400 →
  income = 17000 :=
sorry

end find_income_l220_22049


namespace maximum_value_when_t_is_2_solve_for_t_when_maximum_value_is_2_l220_22094

def f (x : ℝ) (t : ℝ) : ℝ := abs (2 * x - 1) - abs (t * x + 3)

theorem maximum_value_when_t_is_2 :
  ∃ x : ℝ, (f x 2) ≤ 4 ∧ ∀ y : ℝ, (f y 2) ≤ (f x 2) := sorry

theorem solve_for_t_when_maximum_value_is_2 :
  ∃ t : ℝ, t > 0 ∧ (∀ x : ℝ, (f x t) ≤ 2 ∧ (∃ y : ℝ, (f y t) = 2)) → t = 6 := sorry

end maximum_value_when_t_is_2_solve_for_t_when_maximum_value_is_2_l220_22094


namespace actual_time_between_two_and_three_l220_22032

theorem actual_time_between_two_and_three (x y : ℕ) 
  (h1 : 2 ≤ x ∧ x < 3)
  (h2 : 60 * y + x = 60 * x + y - 55) : 
  x = 2 ∧ y = 5 + 5 / 11 := 
sorry

end actual_time_between_two_and_three_l220_22032


namespace star_3_4_equals_8_l220_22021

def star (a b : ℕ) : ℕ := 4 * a + 5 * b - 2 * a * b

theorem star_3_4_equals_8 : star 3 4 = 8 := by
  sorry

end star_3_4_equals_8_l220_22021


namespace rectangular_prism_volume_l220_22064

theorem rectangular_prism_volume
  (l w h : ℝ)
  (h1 : l * w = 15)
  (h2 : w * h = 10)
  (h3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end rectangular_prism_volume_l220_22064


namespace sylvia_carla_together_time_l220_22093

-- Define the conditions
def sylviaRate := 1 / 45
def carlaRate := 1 / 30

-- Define the combined work rate and the time taken to complete the job together
def combinedRate := sylviaRate + carlaRate
def timeTogether := 1 / combinedRate

-- Theorem stating the desired result
theorem sylvia_carla_together_time : timeTogether = 18 := by
  sorry

end sylvia_carla_together_time_l220_22093


namespace scale_model_height_is_correct_l220_22092

noncomputable def height_of_scale_model (h_real : ℝ) (V_real : ℝ) (V_scale : ℝ) : ℝ :=
  h_real / (V_real / V_scale)^(1/3:ℝ)

theorem scale_model_height_is_correct :
  height_of_scale_model 90 500000 0.2 = 0.66 :=
by
  sorry

end scale_model_height_is_correct_l220_22092


namespace discount_percentage_l220_22015

theorem discount_percentage (C M A : ℝ) (h1 : M = 1.40 * C) (h2 : A = 1.05 * C) :
    (M - A) / M * 100 = 25 :=
by
  sorry

end discount_percentage_l220_22015


namespace system1_solution_system2_solution_l220_22022

theorem system1_solution (x y : ℤ) : 
  (x - y = 3) ∧ (x = 3 * y - 1) → (x = 5) ∧ (y = 2) :=
by
  sorry

theorem system2_solution (x y : ℤ) : 
  (2 * x + 3 * y = -1) ∧ (3 * x - 2 * y = 18) → (x = 4) ∧ (y = -3) :=
by
  sorry

end system1_solution_system2_solution_l220_22022


namespace complex_addition_zero_l220_22037

theorem complex_addition_zero (a b : ℝ) (i : ℂ) (h1 : (1 + i) * i = a + b * i) (h2 : i * i = -1) : a + b = 0 :=
sorry

end complex_addition_zero_l220_22037


namespace tickets_won_whack_a_mole_l220_22013

variable (t : ℕ)

def tickets_from_skee_ball : ℕ := 9
def cost_per_candy : ℕ := 6
def number_of_candies : ℕ := 7
def total_tickets_needed : ℕ := cost_per_candy * number_of_candies

theorem tickets_won_whack_a_mole : t + tickets_from_skee_ball = total_tickets_needed → t = 33 :=
by
  intro h
  have h1 : total_tickets_needed = 42 := by sorry
  have h2 : tickets_from_skee_ball = 9 := by rfl
  rw [h2, h1] at h
  sorry

end tickets_won_whack_a_mole_l220_22013


namespace part_I_part_II_l220_22061

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log x - k * x

theorem part_I (k : ℝ) (hk : k = 1) :
  (∀ x, 0 < x ∧ x < 1 → 0 < f 1 x - f 1 1)
  ∧ (∀ x, 1 < x → f 1 1 > f 1 x)
  ∧ f 1 1 = 0 :=
by
  sorry

theorem part_II (k : ℝ) (h_no_zeros : ∀ x, f k x ≠ 0) :
  k > 1 / exp 1 :=
by
  sorry

end part_I_part_II_l220_22061


namespace simplify_expression_l220_22000

theorem simplify_expression : (1 / (1 + Real.sqrt 2)) * (1 / (1 - Real.sqrt 2)) = -1 := by
  sorry

end simplify_expression_l220_22000


namespace divisor_is_20_l220_22096

theorem divisor_is_20 (D : ℕ) 
  (h1 : 242 % D = 11) 
  (h2 : 698 % D = 18) 
  (h3 : 940 % D = 9) :
  D = 20 :=
sorry

end divisor_is_20_l220_22096


namespace unique_four_digit_number_l220_22079

theorem unique_four_digit_number (N : ℕ) (a : ℕ) (x : ℕ) :
  (N = 1000 * a + x) ∧ (N = 7 * x) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (1 ≤ a ∧ a ≤ 9) →
  N = 3500 :=
by sorry

end unique_four_digit_number_l220_22079


namespace langsley_commute_time_l220_22056

theorem langsley_commute_time (first_bus: ℕ) (first_wait: ℕ) (second_bus: ℕ) (second_wait: ℕ) (third_bus: ℕ) (total_time: ℕ)
  (h1: first_bus = 40)
  (h2: first_wait = 10)
  (h3: second_bus = 50)
  (h4: second_wait = 15)
  (h5: third_bus = 95)
  (h6: total_time = first_bus + first_wait + second_bus + second_wait + third_bus) :
  total_time = 210 := 
by 
  sorry

end langsley_commute_time_l220_22056


namespace monotonicity_and_k_range_l220_22047

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x + (1 / 2 : ℝ) * x^2 - x

theorem monotonicity_and_k_range :
  (∀ x : ℝ, x ≥ 0 → f x ≥ k * x - 2) ↔ k ∈ Set.Iic (-2) := sorry

end monotonicity_and_k_range_l220_22047


namespace jimin_initial_candies_l220_22067

theorem jimin_initial_candies : 
  let candies_given_to_yuna := 25
  let candies_given_to_sister := 13
  candies_given_to_yuna + candies_given_to_sister = 38 := 
  by 
    sorry

end jimin_initial_candies_l220_22067


namespace books_taken_out_on_monday_l220_22025

-- Define total number of books initially
def total_books_init := 336

-- Define books taken out on Monday
variable (x : ℕ)

-- Define books brought back on Tuesday
def books_brought_back := 22

-- Define books present after Tuesday
def books_after_tuesday := 234

-- Theorem statement
theorem books_taken_out_on_monday :
  total_books_init - x + books_brought_back = books_after_tuesday → x = 124 :=
by sorry

end books_taken_out_on_monday_l220_22025


namespace compute_nested_operation_l220_22090

def my_op (a b : ℚ) : ℚ := (a^2 - b^2) / (1 - a * b)

theorem compute_nested_operation : my_op 1 (my_op 2 (my_op 3 4)) = -18 := by
  sorry

end compute_nested_operation_l220_22090


namespace fraction_of_water_in_mixture_l220_22076

theorem fraction_of_water_in_mixture (r : ℚ) (h : r = 2 / 3) : (3 / (2 + 3) : ℚ) = 3 / 5 :=
by
  sorry

end fraction_of_water_in_mixture_l220_22076


namespace janet_more_cards_than_brenda_l220_22086

theorem janet_more_cards_than_brenda : ∀ (J B M : ℕ), M = 2 * J → J + B + M = 211 → M = 150 - 40 → J - B = 9 :=
by
  intros J B M h1 h2 h3
  sorry

end janet_more_cards_than_brenda_l220_22086


namespace jose_internet_speed_l220_22026

-- Define the given conditions
def file_size : ℕ := 160
def upload_time : ℕ := 20

-- Define the statement we need to prove
theorem jose_internet_speed : file_size / upload_time = 8 :=
by
  -- Proof should be provided here
  sorry

end jose_internet_speed_l220_22026


namespace maximize_revenue_at_175_l220_22011

def price (x : ℕ) : ℕ :=
  if x ≤ 150 then 200 else 200 - (x - 150)

def revenue (x : ℕ) : ℕ :=
  price x * x

theorem maximize_revenue_at_175 :
  ∀ x : ℕ, revenue 175 ≥ revenue x := 
sorry

end maximize_revenue_at_175_l220_22011


namespace distance_between_islands_l220_22051

theorem distance_between_islands (AB : ℝ) (angle_BAC angle_ABC : ℝ) : 
  AB = 20 ∧ angle_BAC = 60 ∧ angle_ABC = 75 → 
  (∃ BC : ℝ, BC = 10 * Real.sqrt 6) := by
  intro h
  sorry

end distance_between_islands_l220_22051


namespace bennett_brothers_count_l220_22046

theorem bennett_brothers_count :
  ∃ B, B = 2 * 4 - 2 ∧ B = 6 :=
by
  sorry

end bennett_brothers_count_l220_22046


namespace percentage_of_other_sales_l220_22009

theorem percentage_of_other_sales :
  let pensPercentage := 20
  let pencilsPercentage := 15
  let notebooksPercentage := 30
  let totalPercentage := 100
  totalPercentage - (pensPercentage + pencilsPercentage + notebooksPercentage) = 35 :=
by
  sorry

end percentage_of_other_sales_l220_22009


namespace total_seniors_is_161_l220_22065

def total_students : ℕ := 240

def percentage_statistics : ℚ := 0.45
def percentage_geometry : ℚ := 0.35
def percentage_calculus : ℚ := 0.20

def percentage_stats_and_calc : ℚ := 0.10
def percentage_geom_and_calc : ℚ := 0.05

def percentage_seniors_statistics : ℚ := 0.90
def percentage_seniors_geometry : ℚ := 0.60
def percentage_seniors_calculus : ℚ := 0.80

def students_in_statistics : ℚ := percentage_statistics * total_students
def students_in_geometry : ℚ := percentage_geometry * total_students
def students_in_calculus : ℚ := percentage_calculus * total_students

def students_in_stats_and_calc : ℚ := percentage_stats_and_calc * students_in_statistics
def students_in_geom_and_calc : ℚ := percentage_geom_and_calc * students_in_geometry

def unique_students_in_statistics : ℚ := students_in_statistics - students_in_stats_and_calc
def unique_students_in_geometry : ℚ := students_in_geometry - students_in_geom_and_calc
def unique_students_in_calculus : ℚ := students_in_calculus - students_in_stats_and_calc - students_in_geom_and_calc

def seniors_in_statistics : ℚ := percentage_seniors_statistics * unique_students_in_statistics
def seniors_in_geometry : ℚ := percentage_seniors_geometry * unique_students_in_geometry
def seniors_in_calculus : ℚ := percentage_seniors_calculus * unique_students_in_calculus

def total_seniors : ℚ := seniors_in_statistics + seniors_in_geometry + seniors_in_calculus

theorem total_seniors_is_161 : total_seniors = 161 :=
by
  sorry

end total_seniors_is_161_l220_22065


namespace find_k_percent_l220_22042

theorem find_k_percent (k : ℝ) : 0.2 * 30 = 6 → (k / 100) * 25 = 6 → k = 24 := by
  intros h1 h2
  sorry

end find_k_percent_l220_22042


namespace fireflies_joined_l220_22027

theorem fireflies_joined (x : ℕ) : 
  let initial_fireflies := 3
  let flew_away := 2
  let remaining_fireflies := 9
  initial_fireflies + x - flew_away = remaining_fireflies → x = 8 := by
  sorry

end fireflies_joined_l220_22027


namespace ratio_of_canoes_to_kayaks_l220_22034

theorem ratio_of_canoes_to_kayaks 
    (canoe_cost kayak_cost total_revenue : ℕ) 
    (canoe_to_kayak_ratio extra_canoes : ℕ)
    (h1 : canoe_cost = 14)
    (h2 : kayak_cost = 15)
    (h3 : total_revenue = 288)
    (h4 : extra_canoes = 4)
    (h5 : canoe_to_kayak_ratio = 3) 
    (c k : ℕ)
    (h6 : c = k + extra_canoes)
    (h7 : c = canoe_to_kayak_ratio * k)
    (h8 : canoe_cost * c + kayak_cost * k = total_revenue) :
    c / k = 3 := 
sorry

end ratio_of_canoes_to_kayaks_l220_22034


namespace intersection_sets_l220_22081

def setA : Set ℝ := { x | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3 }
def setB : Set ℝ := { x | (x - 3) / (2 * x) ≤ 0 }

theorem intersection_sets (x : ℝ) : x ∈ setA ∧ x ∈ setB ↔ 0 < x ∧ x ≤ 1 := by
  sorry

end intersection_sets_l220_22081


namespace quadratic_function_value_l220_22054

theorem quadratic_function_value (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a - b + c = 9) :
  a + 3 * b + c = 1 := 
by 
  sorry

end quadratic_function_value_l220_22054


namespace calculate_expression_l220_22004

theorem calculate_expression : 6^3 - 5 * 7 + 2^4 = 197 := 
by
  -- Generally, we would provide the proof here, but it's not required.
  sorry

end calculate_expression_l220_22004


namespace greatest_product_from_sum_2004_l220_22014

theorem greatest_product_from_sum_2004 : ∃ (x y : ℤ), x + y = 2004 ∧ x * y = 1004004 :=
by
  sorry

end greatest_product_from_sum_2004_l220_22014


namespace age_difference_l220_22017

theorem age_difference (A B C : ℕ) (h1 : A + B > B + C) (h2 : C = A - 17) : (A + B) - (B + C) = 17 :=
by
  sorry

end age_difference_l220_22017


namespace absolute_value_half_angle_cosine_l220_22016

theorem absolute_value_half_angle_cosine (x : ℝ) (h1 : Real.sin x = -5 / 13) (h2 : ∀ n : ℤ, (2 * n) * Real.pi < x ∧ x < (2 * n + 1) * Real.pi) :
  |Real.cos (x / 2)| = Real.sqrt 26 / 26 :=
sorry

end absolute_value_half_angle_cosine_l220_22016


namespace jacob_twice_as_old_l220_22084

theorem jacob_twice_as_old (x : ℕ) : 18 + x = 2 * (9 + x) → x = 0 := by
  intro h
  linarith

end jacob_twice_as_old_l220_22084


namespace minute_hand_length_l220_22010

theorem minute_hand_length 
  (arc_length : ℝ) (r : ℝ) (h : arc_length = 20 * (2 * Real.pi / 60) * r) :
  r = 1/2 :=
  sorry

end minute_hand_length_l220_22010


namespace points_in_groups_l220_22087

theorem points_in_groups (n1 n2 : ℕ) (h_total : n1 + n2 = 28) 
  (h_lines_diff : (n1*(n1 - 1) / 2) - (n2*(n2 - 1) / 2) = 81) : 
  (n1 = 17 ∧ n2 = 11) ∨ (n1 = 11 ∧ n2 = 17) :=
by
  sorry

end points_in_groups_l220_22087


namespace find_a_l220_22066

theorem find_a (a : ℤ) :
  (∃! x : ℤ, |a * x + a + 2| < 2) ↔ a = 3 ∨ a = -3 := 
sorry

end find_a_l220_22066


namespace total_pens_left_l220_22099

def initial_blue_pens := 9
def removed_blue_pens := 4
def initial_black_pens := 21
def removed_black_pens := 7
def initial_red_pens := 6

def remaining_blue_pens := initial_blue_pens - removed_blue_pens
def remaining_black_pens := initial_black_pens - removed_black_pens
def remaining_red_pens := initial_red_pens

def total_remaining_pens := remaining_blue_pens + remaining_black_pens + remaining_red_pens

theorem total_pens_left : total_remaining_pens = 25 :=
by
  -- Proof will be provided here
  sorry

end total_pens_left_l220_22099


namespace geometric_sequence_a1_l220_22083

theorem geometric_sequence_a1 (a1 a2 a3 S3 : ℝ) (q : ℝ)
  (h1 : S3 = a1 + (1 / 2) * a2)
  (h2 : a3 = (1 / 4))
  (h3 : S3 = a1 * (1 + q + q^2))
  (h4 : a2 = a1 * q)
  (h5 : a3 = a1 * q^2) :
  a1 = 1 :=
sorry

end geometric_sequence_a1_l220_22083


namespace rent_change_percent_l220_22074

open Real

noncomputable def elaine_earnings_last_year (E : ℝ) : ℝ :=
E

noncomputable def elaine_rent_last_year (E : ℝ) : ℝ :=
0.2 * E

noncomputable def elaine_earnings_this_year (E : ℝ) : ℝ :=
1.15 * E

noncomputable def elaine_rent_this_year (E : ℝ) : ℝ :=
0.25 * (1.15 * E)

noncomputable def rent_percentage_change (E : ℝ) : ℝ :=
(elaine_rent_this_year E) / (elaine_rent_last_year E) * 100

theorem rent_change_percent (E : ℝ) :
  rent_percentage_change E = 143.75 :=
by
  sorry

end rent_change_percent_l220_22074


namespace fraction_videocassette_recorders_l220_22038

variable (H : ℝ) (F : ℝ)

-- Conditions
variable (cable_TV_frac : ℝ := 1 / 5)
variable (both_frac : ℝ := 1 / 20)
variable (neither_frac : ℝ := 0.75)

-- Main theorem statement
theorem fraction_videocassette_recorders (H_pos : 0 < H) 
  (cable_tv : cable_TV_frac * H > 0)
  (both : both_frac * H > 0) 
  (neither : neither_frac * H > 0) :
  F = 1 / 10 :=
by
  sorry

end fraction_videocassette_recorders_l220_22038


namespace percentage_is_40_l220_22001

variables (num : ℕ) (perc : ℕ)

-- Conditions
def ten_percent_eq_40 : Prop := 10 * num = 400
def certain_percentage_eq_160 : Prop := perc * num = 160 * 100

-- Statement to prove
theorem percentage_is_40 (h1 : ten_percent_eq_40 num) (h2 : certain_percentage_eq_160 num perc) : perc = 40 :=
sorry

end percentage_is_40_l220_22001
