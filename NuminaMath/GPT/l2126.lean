import Mathlib

namespace range_of_a_l2126_212612

def A : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a < x ∧ x < a + 1}

theorem range_of_a (a : ℝ)
  (h₀ : a < 1)
  (h₁ : B a ⊆ A) :
  a ∈ {x : ℝ | x ≤ -2 ∨ (1 / 2 ≤ x ∧ x < 1)} :=
by
  sorry

end range_of_a_l2126_212612


namespace clive_money_l2126_212613

noncomputable def clive_initial_money : ℝ  :=
  let total_olives := 80
  let olives_per_jar := 20
  let cost_per_jar := 1.5
  let change := 4
  let jars_needed := total_olives / olives_per_jar
  let total_cost := jars_needed * cost_per_jar
  total_cost + change

theorem clive_money (h1 : clive_initial_money = 10) : clive_initial_money = 10 :=
by sorry

end clive_money_l2126_212613


namespace arithmetic_seq_a9_l2126_212605

theorem arithmetic_seq_a9 (a : ℕ → ℤ) (h1 : a 3 - a 2 = -2) (h2 : a 7 = -2) : a 9 = -6 := 
by sorry

end arithmetic_seq_a9_l2126_212605


namespace donut_selection_count_l2126_212655

def num_donut_selections : ℕ :=
  Nat.choose 9 3

theorem donut_selection_count : num_donut_selections = 84 := 
by
  sorry

end donut_selection_count_l2126_212655


namespace sum_of_roots_l2126_212680

theorem sum_of_roots 
  (a b c : ℝ)
  (h1 : 1^2 + a * 1 + 2 = 0)
  (h2 : (∀ x : ℝ, x^2 + 5 * x + c = 0 → (x = a ∨ x = b))) :
  a + b + c = 1 :=
by
  sorry

end sum_of_roots_l2126_212680


namespace dogs_remaining_end_month_l2126_212650

theorem dogs_remaining_end_month :
  let initial_dogs := 200
  let dogs_arrive_w1 := 30
  let dogs_adopt_w1 := 40
  let dogs_arrive_w2 := 40
  let dogs_adopt_w2 := 50
  let dogs_arrive_w3 := 30
  let dogs_adopt_w3 := 30
  let dogs_adopt_w4 := 70
  let dogs_return_w4 := 20
  initial_dogs + (dogs_arrive_w1 - dogs_adopt_w1) + 
  (dogs_arrive_w2 - dogs_adopt_w2) +
  (dogs_arrive_w3 - dogs_adopt_w3) + 
  (-dogs_adopt_w4 - dogs_return_w4) = 90 := by
  sorry

end dogs_remaining_end_month_l2126_212650


namespace range_of_half_alpha_minus_beta_l2126_212646

theorem range_of_half_alpha_minus_beta (α β : ℝ) (hα : 1 < α ∧ α < 3) (hβ : -4 < β ∧ β < 2) :
  -3 / 2 < (1 / 2) * α - β ∧ (1 / 2) * α - β < 11 / 2 :=
sorry

end range_of_half_alpha_minus_beta_l2126_212646


namespace bill_profit_difference_l2126_212614

theorem bill_profit_difference (P : ℝ) 
  (h1 : 1.10 * P = 549.9999999999995)
  (h2 : ∀ NP NSP, NP = 0.90 * P ∧ NSP = 1.30 * NP →
  NSP - 549.9999999999995 = 35) :
  true :=
by {
  sorry
}

end bill_profit_difference_l2126_212614


namespace smaller_angle_at_3_45_l2126_212653

def minute_hand_angle : ℝ := 270
def hour_hand_angle : ℝ := 90 + 0.75 * 30

theorem smaller_angle_at_3_45 :
  min (|minute_hand_angle - hour_hand_angle|) (360 - |minute_hand_angle - hour_hand_angle|) = 202.5 := 
by
  sorry

end smaller_angle_at_3_45_l2126_212653


namespace total_journey_distance_l2126_212673

-- Definitions of the conditions

def journey_time : ℝ := 40
def first_half_speed : ℝ := 20
def second_half_speed : ℝ := 30

-- Proof statement
theorem total_journey_distance : ∃ D : ℝ, (D / first_half_speed + D / second_half_speed = journey_time) ∧ (D = 960) :=
by 
  sorry

end total_journey_distance_l2126_212673


namespace find_digits_l2126_212699

theorem find_digits (a b c d : ℕ) 
  (h₀ : 0 ≤ a ∧ a ≤ 9)
  (h₁ : 0 ≤ b ∧ b ≤ 9)
  (h₂ : 0 ≤ c ∧ c ≤ 9)
  (h₃ : 0 ≤ d ∧ d ≤ 9)
  (h₄ : (10 * a + c) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 17 / 37) :
  1000 * a + 100 * b + 10 * c + d = 2315 :=
by
  sorry

end find_digits_l2126_212699


namespace part1_part2_l2126_212682

-- Define the parabola C as y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line l with slope k passing through point P(-2, 1)
def line (x y k : ℝ) : Prop := y - 1 = k * (x + 2)

-- Part 1: Prove the range of k for which line l intersects parabola C at two points is -1 < k < -1/2
theorem part1 (k : ℝ) : 
  (∃ x y, parabola x y ∧ line x y k) ∧ (∃ u v, parabola u v ∧ u ≠ x ∧ line u v k) ↔ -1 < k ∧ k < -1/2 := sorry

-- Part 2: Prove the equations of line l when it intersects parabola C at only one point are y = 1, y = -x - 1, and y = -1/2 x
theorem part2 (k : ℝ) : 
  (∃! x y, parabola x y ∧ line x y k) ↔ (k = 0 ∨ k = -1 ∨ k = -1/2) := sorry

end part1_part2_l2126_212682


namespace three_digit_number_value_l2126_212674

theorem three_digit_number_value (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
    (h4 : a > b) (h5 : b > c)
    (h6 : (10 * a + b) + (10 * b + a) = 55)  
    (h7 : 1300 < 222 * (a + b + c) ∧ 222 * (a + b + c) < 1400) : 
    (100 * a + 10 * b + c) = 321 := 
sorry

end three_digit_number_value_l2126_212674


namespace find_cuboid_length_l2126_212620

theorem find_cuboid_length
  (b : ℝ) (h : ℝ) (S : ℝ)
  (hb : b = 10) (hh : h = 12) (hS : S = 960) :
  ∃ l : ℝ, 2 * (l * b + b * h + h * l) = S ∧ l = 16.36 :=
by
  sorry

end find_cuboid_length_l2126_212620


namespace necessary_but_not_sufficient_l2126_212624

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a + b > 4) ↔ (¬ (a > 2 ∧ b > 2)) ∧ ((a > 2 ∧ b > 2) → (a + b > 4)) :=
by
  sorry

end necessary_but_not_sufficient_l2126_212624


namespace perimeter_of_figure_l2126_212696

variable (x y : ℝ)
variable (lengths : Set ℝ)
variable (perpendicular_adjacent : Prop)
variable (area : ℝ)

-- Conditions
def condition_1 : Prop := ∀ l ∈ lengths, l = x ∨ l = y
def condition_2 : Prop := perpendicular_adjacent
def condition_3 : Prop := area = 252
def condition_4 : Prop := x = 2 * y

-- Problem statement
theorem perimeter_of_figure
  (h1 : condition_1 x y lengths)
  (h2 : condition_2 perpendicular_adjacent)
  (h3 : condition_3 area)
  (h4 : condition_4 x y) :
  ∃ perimeter : ℝ, perimeter = 96 := by
  sorry

end perimeter_of_figure_l2126_212696


namespace solve_phi_l2126_212687

theorem solve_phi (n : ℕ) : 
  (∃ (x y z : ℕ), 5 * x + 2 * y + z = 10 * n) → 
  (∃ (φ : ℕ), φ = 5 * n^2 + 4 * n + 1) :=
by 
  sorry

end solve_phi_l2126_212687


namespace economical_speed_l2126_212692

variable (a k : ℝ)
variable (ha : 0 < a) (hk : 0 < k)

theorem economical_speed (v : ℝ) : 
  v = (a / (2 * k))^(1/3) :=
sorry

end economical_speed_l2126_212692


namespace LCM_4_6_15_is_60_l2126_212670

def prime_factors (n : ℕ) : List ℕ :=
  [] -- placeholder, definition of prime_factor is not necessary for the problem statement, so we leave it abstract

def LCM (a b : ℕ) : ℕ := 
  sorry -- placeholder, definition of LCM not directly necessary for the statement

theorem LCM_4_6_15_is_60 : LCM (LCM 4 6) 15 = 60 := 
  sorry

end LCM_4_6_15_is_60_l2126_212670


namespace solve_for_x_l2126_212689

theorem solve_for_x :
  ∀ x : ℝ, (1 / 6 + 7 / x = 15 / x + 1 / 15 + 2) → x = -80 / 19 :=
by
  intros x h
  sorry

end solve_for_x_l2126_212689


namespace binom_13_11_eq_78_l2126_212651

theorem binom_13_11_eq_78 : Nat.choose 13 11 = 78 := by
  sorry

end binom_13_11_eq_78_l2126_212651


namespace raghu_investment_l2126_212694

theorem raghu_investment (R : ℝ) 
  (h1 : ∀ T : ℝ, T = 0.9 * R) 
  (h2 : ∀ V : ℝ, V = 0.99 * R) 
  (h3 : R + 0.9 * R + 0.99 * R = 6069) : 
  R = 2100 := 
by
  sorry

end raghu_investment_l2126_212694


namespace gcd_eq_55_l2126_212664

theorem gcd_eq_55 : Nat.gcd 5280 12155 = 55 := sorry

end gcd_eq_55_l2126_212664


namespace book_pages_total_l2126_212616

-- Define the conditions
def pagesPerNight : ℝ := 120.0
def nights : ℝ := 10.0

-- State the theorem to prove
theorem book_pages_total : pagesPerNight * nights = 1200.0 := by
  sorry

end book_pages_total_l2126_212616


namespace f_f_2_equals_l2126_212626

def f (x : ℕ) : ℕ := 4 * x ^ 3 - 6 * x + 2

theorem f_f_2_equals :
  f (f 2) = 42462 :=
by
  sorry

end f_f_2_equals_l2126_212626


namespace problem_solution_l2126_212698

theorem problem_solution :
  ∀ (x y z : ℤ),
  4 * x + y + z = 80 →
  3 * x + y - z = 20 →
  x = 20 →
  2 * x - y - z = 40 :=
by
  intros x y z h1 h2 hx
  rw [hx] at h1 h2
  -- Here you could continue solving but we'll use sorry to indicate the end as no proof is requested.
  sorry

end problem_solution_l2126_212698


namespace valid_param_a_valid_param_c_l2126_212608

/-
The task is to prove that the goals provided are valid parameterizations of the given line.
-/

def line_eqn (x y : ℝ) : Prop := y = -7/4 * x + 21/4

def is_valid_param (p₀ : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line_eqn ((p₀.1 + t * d.1) : ℝ) ((p₀.2 + t * d.2) : ℝ)

theorem valid_param_a : is_valid_param (7, 0) (4, -7) :=
by
  sorry

theorem valid_param_c : is_valid_param (0, 21/4) (-4, 7) :=
by
  sorry


end valid_param_a_valid_param_c_l2126_212608


namespace odds_against_C_l2126_212623

theorem odds_against_C (pA pB : ℚ) (hA : pA = 1 / 5) (hB : pB = 2 / 3) :
  (1 - (1 - pA + 1 - pB)) / (1 - pA - pB) = 13 / 2 := 
sorry

end odds_against_C_l2126_212623


namespace range_of_a_l2126_212688

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Lean statement for the problem
theorem range_of_a (a : ℝ) (h : A ∪ B a = B a) : -1 < a ∧ a ≤ 1 := 
by
  -- Proof is skipped
  sorry

end range_of_a_l2126_212688


namespace tan_240_eq_sqrt3_l2126_212635

theorem tan_240_eq_sqrt3 : Real.tan (240 * Real.pi / 180) = Real.sqrt 3 :=
by sorry

end tan_240_eq_sqrt3_l2126_212635


namespace total_rowing_time_l2126_212607

theorem total_rowing_time (s_b : ℕ) (s_s : ℕ) (d : ℕ) : 
  s_b = 9 → s_s = 6 → d = 170 → 
  (d / (s_b + s_s) + d / (s_b - s_s)) = 68 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_rowing_time_l2126_212607


namespace exists_reals_condition_l2126_212630

-- Define the conditions in Lean
theorem exists_reals_condition (n : ℕ) (h₁ : n ≥ 3) : 
  (∃ a : Fin (n + 2) → ℝ, a 0 = a n ∧ a 1 = a (n + 1) ∧ 
  ∀ i : Fin n, a i * a (i + 1) + 1 = a (i + 2)) ↔ 3 ∣ n := 
sorry

end exists_reals_condition_l2126_212630


namespace total_wicks_l2126_212671

-- Amy bought a 15-foot spool of string.
def spool_length_feet : ℕ := 15

-- Since there are 12 inches in a foot, convert the spool length to inches.
def spool_length_inches : ℕ := spool_length_feet * 12

-- The string is cut into an equal number of 6-inch and 12-inch wicks.
def wick_pair_length : ℕ := 6 + 12

-- Prove that the total number of wicks she cuts is 20.
theorem total_wicks : (spool_length_inches / wick_pair_length) * 2 = 20 := by
  sorry

end total_wicks_l2126_212671


namespace tan_sub_theta_cos_double_theta_l2126_212662

variables (θ : ℝ)

-- Condition: given tan θ = 2
axiom tan_theta_eq_two : Real.tan θ = 2

-- Proof problem 1: Prove tan (π/4 - θ) = -1/3
theorem tan_sub_theta (h : Real.tan θ = 2) : Real.tan (Real.pi / 4 - θ) = -1/3 :=
by sorry

-- Proof problem 2: Prove cos 2θ = -3/5
theorem cos_double_theta (h : Real.tan θ = 2) : Real.cos (2 * θ) = -3/5 :=
by sorry

end tan_sub_theta_cos_double_theta_l2126_212662


namespace max_sum_length_le_98306_l2126_212675

noncomputable def L (k : ℕ) : ℕ := sorry

theorem max_sum_length_le_98306 (x y : ℕ) (hx : x > 1) (hy : y > 1) (hl : L x + L y = 16) : x + 3 * y < 98306 :=
sorry

end max_sum_length_le_98306_l2126_212675


namespace sum_of_three_distinct_integers_product_625_l2126_212684

theorem sum_of_three_distinct_integers_product_625 :
  ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 131 :=
by
  sorry

end sum_of_three_distinct_integers_product_625_l2126_212684


namespace number_of_planters_l2126_212603

variable (a b : ℕ)

-- Conditions
def tree_planting_condition_1 : Prop := a * b = 2013
def tree_planting_condition_2 : Prop := (a - 5) * (b + 2) < 2013
def tree_planting_condition_3 : Prop := (a - 5) * (b + 3) > 2013

-- Theorem stating the number of people who participated in the planting is 61
theorem number_of_planters (h1 : tree_planting_condition_1 a b) 
                           (h2 : tree_planting_condition_2 a b) 
                           (h3 : tree_planting_condition_3 a b) : 
                           a = 61 := 
sorry

end number_of_planters_l2126_212603


namespace range_of_m_l2126_212659

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 + 2 * x + m > 0) → m > 1 :=
by
  -- Proof goes here
  sorry

end range_of_m_l2126_212659


namespace university_students_l2126_212639

theorem university_students (total_students students_both math_students physics_students : ℕ) 
  (h1 : total_students = 75) 
  (h2 : total_students = (math_students - students_both) + (physics_students - students_both) + students_both)
  (h3 : math_students = 2 * physics_students) 
  (h4 : students_both = 10) : 
  math_students = 56 := by
  sorry

end university_students_l2126_212639


namespace overall_percentage_gain_l2126_212629

theorem overall_percentage_gain
    (original_price : ℝ)
    (first_increase : ℝ)
    (first_discount : ℝ)
    (second_discount : ℝ)
    (third_discount : ℝ)
    (final_increase : ℝ)
    (final_price : ℝ)
    (overall_gain : ℝ)
    (overall_percentage_gain : ℝ)
    (h1 : original_price = 100)
    (h2 : first_increase = original_price * 1.5)
    (h3 : first_discount = first_increase * 0.9)
    (h4 : second_discount = first_discount * 0.85)
    (h5 : third_discount = second_discount * 0.8)
    (h6 : final_increase = third_discount * 1.1)
    (h7 : final_price = final_increase)
    (h8 : overall_gain = final_price - original_price)
    (h9 : overall_percentage_gain = (overall_gain / original_price) * 100) :
  overall_percentage_gain = 0.98 := by
  sorry

end overall_percentage_gain_l2126_212629


namespace juniper_remaining_bones_l2126_212601

-- Conditions
def initial_bones : ℕ := 4
def doubled_bones (b : ℕ) : ℕ := 2 * b
def stolen_bones (b : ℕ) : ℕ := b - 2

-- Theorem Statement
theorem juniper_remaining_bones : stolen_bones (doubled_bones initial_bones) = 6 := by
  -- Proof is omitted, only the statement is required as per instructions
  sorry

end juniper_remaining_bones_l2126_212601


namespace one_corresponds_to_36_l2126_212669

-- Define the given conditions
def corresponds (n : Nat) (s : String) : Prop :=
match n with
| 2  => s = "36"
| 3  => s = "363"
| 4  => s = "364"
| 5  => s = "365"
| 36 => s = "2"
| _  => False

-- Statement for the proof problem: Prove that 1 corresponds to 36
theorem one_corresponds_to_36 : corresponds 1 "36" :=
by
  sorry

end one_corresponds_to_36_l2126_212669


namespace wifi_cost_per_hour_l2126_212647

-- Define the conditions as hypotheses
def ticket_cost : ℝ := 11
def snacks_cost : ℝ := 3
def headphones_cost : ℝ := 16
def hourly_income : ℝ := 12
def trip_duration : ℝ := 3
def total_expenses : ℝ := ticket_cost + snacks_cost + headphones_cost
def total_earnings : ℝ := hourly_income * trip_duration

-- Translate the proof problem to Lean 4 statement
theorem wifi_cost_per_hour: 
  (total_earnings - total_expenses) / trip_duration = 2 :=
by sorry

end wifi_cost_per_hour_l2126_212647


namespace isosceles_triangle_perimeter_l2126_212695

def is_isosceles_triangle (A B C : ℝ) : Prop :=
  (A = B ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (A = C ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (B = C ∧ A + B > C ∧ A + C > B ∧ B + C > A)

theorem isosceles_triangle_perimeter {A B C : ℝ} 
  (h : is_isosceles_triangle A B C) 
  (h1 : A = 3 ∨ A = 7) 
  (h2 : B = 3 ∨ B = 7) 
  (h3 : C = 3 ∨ C = 7) : 
  A + B + C = 17 := 
sorry

end isosceles_triangle_perimeter_l2126_212695


namespace determine_F_l2126_212604

def f1 (x : ℝ) : ℝ := x^2 + x
def f2 (x : ℝ) : ℝ := 2 * x^2 - x
def f3 (x : ℝ) : ℝ := x^2 + x

def g1 (x : ℝ) : ℝ := x - 2
def g2 (x : ℝ) : ℝ := 2 * x
def g3 (x : ℝ) : ℝ := x + 2

def h (x : ℝ) : ℝ := x

theorem determine_F (F1 F2 F3 : ℕ) : 
  (F1 = 0 ∧ F2 = 0 ∧ F3 = 1) :=
by
  sorry

end determine_F_l2126_212604


namespace possible_ways_to_choose_gates_l2126_212685

theorem possible_ways_to_choose_gates : 
  ∃! (ways : ℕ), ways = 20 := 
by
  sorry

end possible_ways_to_choose_gates_l2126_212685


namespace find_natural_pairs_l2126_212622

-- Definitions
def is_natural (n : ℕ) : Prop := n > 0
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def satisfies_equation (x y : ℕ) : Prop := 2 * x^2 + 5 * x * y + 3 * y^2 = 41 * x + 62 * y + 21

-- Problem statement
theorem find_natural_pairs (x y : ℕ) (hx : is_natural x) (hy : is_natural y) (hrel : relatively_prime x y) :
  satisfies_equation x y ↔ (x = 2 ∧ y = 19) ∨ (x = 19 ∧ y = 2) :=
by
  sorry

end find_natural_pairs_l2126_212622


namespace championship_outcomes_l2126_212625

theorem championship_outcomes (students events : ℕ) (hs : students = 5) (he : events = 3) :
  ∃ outcomes : ℕ, outcomes = 5 ^ 3 := by
  sorry

end championship_outcomes_l2126_212625


namespace altitude_length_l2126_212693

noncomputable def length_of_altitude (l w : ℝ) : ℝ :=
  2 * l * w / Real.sqrt (l ^ 2 + w ^ 2)

theorem altitude_length (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  ∃ h : ℝ, h = length_of_altitude l w := by
  sorry

end altitude_length_l2126_212693


namespace ap_number_of_terms_l2126_212657

theorem ap_number_of_terms (a d : ℕ) (n : ℕ) (ha1 : (n - 1) * d = 12) (ha2 : a + 2 * d = 6)
  (h_odd_sum : (n / 2) * (2 * a + (n - 2) * d) = 36) (h_even_sum : (n / 2) * (2 * a + n * d) = 42) :
    n = 12 :=
by
  sorry

end ap_number_of_terms_l2126_212657


namespace num_even_multiple_5_perfect_squares_lt_1000_l2126_212676

theorem num_even_multiple_5_perfect_squares_lt_1000 : 
  ∃ n, n = 3 ∧ ∀ x, (x < 1000) ∧ (x > 0) ∧ (∃ k, x = 100 * k^2) → (n = 3) := by 
  sorry

end num_even_multiple_5_perfect_squares_lt_1000_l2126_212676


namespace find_b_l2126_212636

variable (x : ℝ)

noncomputable def d : ℝ := 3

theorem find_b (b c : ℝ) :
  (7 * x^2 - 5 * x + 11 / 4) * (d * x^2 + b * x + c) = 21 * x^4 - 26 * x^3 + 34 * x^2 - 55 / 4 * x + 33 / 4 →
  b = -11 / 7 :=
by
  sorry

end find_b_l2126_212636


namespace fraction_of_l2126_212690

theorem fraction_of (a b : ℚ) (h_a : a = 3/4) (h_b : b = 1/6) : b / a = 2/9 :=
by
  sorry

end fraction_of_l2126_212690


namespace paul_mowing_lawns_l2126_212641

theorem paul_mowing_lawns : 
  ∃ M : ℕ, 
    (∃ money_made_weeating : ℕ, money_made_weeating = 13) ∧
    (∃ spending_per_week : ℕ, spending_per_week = 9) ∧
    (∃ weeks_last : ℕ, weeks_last = 9) ∧
    (M + 13 = 9 * 9) → 
    M = 68 := by
sorry

end paul_mowing_lawns_l2126_212641


namespace find_number_l2126_212600

variable (x : ℕ)

theorem find_number (h : (10 + 20 + x) / 3 = ((10 + 40 + 25) / 3) + 5) : x = 60 :=
by
  sorry

end find_number_l2126_212600


namespace average_temperature_l2126_212645

theorem average_temperature (T_NY T_Miami T_SD : ℝ) (h1 : T_NY = 80) (h2 : T_Miami = T_NY + 10) (h3 : T_SD = T_Miami + 25) :
  (T_NY + T_Miami + T_SD) / 3 = 95 :=
by
  sorry

end average_temperature_l2126_212645


namespace min_x_div_y_l2126_212683

theorem min_x_div_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + y = 2) : ∃c: ℝ, c = 1 ∧ ∀(a: ℝ), x = a → y = 1 → a/y ≥ c :=
by
  sorry

end min_x_div_y_l2126_212683


namespace num_positive_integer_solutions_l2126_212619

theorem num_positive_integer_solutions : 
  ∃ n : ℕ, (∀ x : ℕ, x ≤ n → x - 1 < Real.sqrt 5) ∧ n = 3 :=
by
  sorry

end num_positive_integer_solutions_l2126_212619


namespace football_club_initial_balance_l2126_212642

noncomputable def initial_balance (final_balance income expense : ℕ) : ℕ :=
  final_balance + income - expense

theorem football_club_initial_balance :
  initial_balance 60 (2 * 10) (4 * 15) = 20 := by
sorry

end football_club_initial_balance_l2126_212642


namespace mary_baseball_cards_count_l2126_212618

def mary_initial_cards := 18
def mary_torn_cards := 8
def fred_gift_cards := 26
def mary_bought_cards := 40
def exchange_with_tom := 0
def mary_lost_cards := 5
def trade_with_lisa_gain := 1
def exchange_with_alex_loss := 2

theorem mary_baseball_cards_count : 
  mary_initial_cards - mary_torn_cards
  + fred_gift_cards
  + mary_bought_cards 
  + exchange_with_tom
  - mary_lost_cards
  + trade_with_lisa_gain 
  - exchange_with_alex_loss 
  = 70 := 
by
  sorry

end mary_baseball_cards_count_l2126_212618


namespace fraction_of_full_tank_used_l2126_212663

-- Define the initial conditions as per the problem statement
def speed : ℝ := 50 -- miles per hour
def time : ℝ := 5   -- hours
def miles_per_gallon : ℝ := 30
def full_tank_capacity : ℝ := 15 -- gallons

-- We need to prove that the fraction of gasoline used is 5/9
theorem fraction_of_full_tank_used : 
  ((speed * time) / miles_per_gallon) / full_tank_capacity = 5 / 9 := by
sorry

end fraction_of_full_tank_used_l2126_212663


namespace average_book_width_is_3_point_9375_l2126_212634

def book_widths : List ℚ := [3, 4, 3/4, 1.5, 7, 2, 5.25, 8]
def number_of_books : ℚ := 8
def total_width : ℚ := List.sum book_widths
def average_width : ℚ := total_width / number_of_books

theorem average_book_width_is_3_point_9375 :
  average_width = 3.9375 := by
  sorry

end average_book_width_is_3_point_9375_l2126_212634


namespace repeating_prime_exists_l2126_212679

open Nat

theorem repeating_prime_exists (p : Fin 2021 → ℕ) 
  (prime_seq : ∀ i : Fin 2021, Nat.Prime (p i))
  (diff_condition : ∀ i : Fin 2019, (p (i + 1) - p i = 6 ∨ p (i + 1) - p i = 12) ∧ (p (i + 2) - p (i + 1) = 6 ∨ p (i + 2) - p (i + 1) = 12)) : 
  ∃ i j : Fin 2021, i ≠ j ∧ p i = p j := by
  sorry

end repeating_prime_exists_l2126_212679


namespace time_to_meet_l2126_212668

variable (distance : ℕ)
variable (speed1 speed2 time : ℕ)

-- Given conditions
def distanceAB := 480
def speedPassengerCar := 65
def speedCargoTruck := 55

-- Sum of the speeds of the two vehicles
def sumSpeeds := speedPassengerCar + speedCargoTruck

-- Prove that the time it takes for the two vehicles to meet is 4 hours
theorem time_to_meet : sumSpeeds * time = distanceAB → time = 4 :=
by
  sorry

end time_to_meet_l2126_212668


namespace solution_set_of_inequality_l2126_212666

theorem solution_set_of_inequality:
  {x : ℝ | |x - 5| + |x + 1| < 8} = {x : ℝ | -2 < x ∧ x < 6} :=
sorry

end solution_set_of_inequality_l2126_212666


namespace find_possible_values_of_b_l2126_212621

def good_number (x : ℕ) : Prop :=
  ∃ p n : ℕ, Nat.Prime p ∧ n ≥ 2 ∧ x = p^n

theorem find_possible_values_of_b (b : ℕ) : 
  (b ≥ 4) ∧ good_number (b^2 - 2 * b - 3) ↔ b = 87 := sorry

end find_possible_values_of_b_l2126_212621


namespace smallest_possible_c_minus_a_l2126_212631

theorem smallest_possible_c_minus_a :
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a * b * c = Nat.factorial 9 ∧ c - a = 216 := 
by
  sorry

end smallest_possible_c_minus_a_l2126_212631


namespace remainder_when_divided_by_7_l2126_212644

theorem remainder_when_divided_by_7 :
  let a := -1234
  let b := 1984
  let c := -1460
  let d := 2008
  (a * b * c * d) % 7 = 0 :=
by
  sorry

end remainder_when_divided_by_7_l2126_212644


namespace cream_ratio_l2126_212615

variable (servings : ℕ) (fat_per_serving : ℕ) (fat_per_cup : ℕ)
variable (h_servings : servings = 4) (h_fat_per_serving : fat_per_serving = 11) (h_fat_per_cup : fat_per_cup = 88)

theorem cream_ratio (total_fat : ℕ) (h_total_fat : total_fat = fat_per_serving * servings) :
  (total_fat : ℚ) / fat_per_cup = 1 / 2 :=
by
  sorry

end cream_ratio_l2126_212615


namespace top_triangle_is_multiple_of_5_l2126_212633

-- Definitions of the conditions given in the problem

def lower_left_triangle := 12
def lower_right_triangle := 3

-- Let a, b, c, d be the four remaining numbers in the bottom row
variables (a b c d : ℤ)

-- Conditions that the sums of triangles must be congruent to multiples of 5
def second_lowest_row : Prop :=
  (3 - a) % 5 = 0 ∧
  (-a - b) % 5 = 0 ∧
  (-b - c) % 5 = 0 ∧
  (-c - d) % 5 = 0 ∧
  (2 - d) % 5 = 0

def third_lowest_row : Prop :=
  (2 + 2*a + b) % 5 = 0 ∧
  (a + 2*b + c) % 5 = 0 ∧
  (b + 2*c + d) % 5 = 0 ∧
  (3 + c + 2*d) % 5 = 0

def fourth_lowest_row : Prop :=
  (3 + 2*a + 2*b - c) % 5 = 0 ∧
  (-a + 2*b + 2*c - d) % 5 = 0 ∧
  (2 - b + 2*c + 2*d) % 5 = 0

def second_highest_row : Prop :=
  (2 - a + b - c + d) % 5 = 0 ∧
  (3 + a - b + c - d) % 5 = 0

def top_triangle : Prop :=
  (2 - a + b - c + d + 3 + a - b + c - d) % 5 = 0

theorem top_triangle_is_multiple_of_5 (a b c d : ℤ) :
  second_lowest_row a b c d →
  third_lowest_row a b c d →
  fourth_lowest_row a b c d →
  second_highest_row a b c d →
  top_triangle a b c d →
  ∃ k : ℤ, (2 - a + b - c + d + 3 + a - b + c - d) = 5 * k :=
by sorry

end top_triangle_is_multiple_of_5_l2126_212633


namespace a_n_values_l2126_212697

noncomputable def a : ℕ → ℕ := sorry
noncomputable def S : ℕ → ℕ := sorry

axiom Sn_property (n : ℕ) (hn : n > 0) : S n = 2 * (a n) - n

theorem a_n_values : a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 7 ∧ ∀ n : ℕ, n > 0 → a n = 2^n - 1 := 
by sorry

end a_n_values_l2126_212697


namespace volume_of_snow_l2126_212667

theorem volume_of_snow (L W H : ℝ) (hL : L = 30) (hW : W = 3) (hH : H = 0.75) :
  L * W * H = 67.5 := by
  sorry

end volume_of_snow_l2126_212667


namespace integer_solutions_of_log_inequality_l2126_212611

def log_inequality_solution_set : Set ℤ := {0, 1, 2}

theorem integer_solutions_of_log_inequality (x : ℤ) (h : 2 < Real.log (x + 5) / Real.log 2 ∧ Real.log (x + 5) / Real.log 2 < 3) :
    x ∈ log_inequality_solution_set :=
sorry

end integer_solutions_of_log_inequality_l2126_212611


namespace fraction_of_raisins_l2126_212649

-- Define the cost of a single pound of raisins
variables (R : ℝ) -- R represents the cost of one pound of raisins

-- Conditions
def mixed_raisins := 5 -- Chris mixed 5 pounds of raisins
def mixed_nuts := 4 -- with 4 pounds of nuts
def nuts_cost_ratio := 3 -- A pound of nuts costs 3 times as much as a pound of raisins

-- Statement to prove
theorem fraction_of_raisins
  (R_pos : R > 0) : (5 * R) / ((5 * R) + (4 * (3 * R))) = 5 / 17 :=
by
  -- The proof is omitted here.
  sorry

end fraction_of_raisins_l2126_212649


namespace octagon_perimeter_l2126_212643

-- Definitions based on conditions
def is_octagon (n : ℕ) : Prop := n = 8
def side_length : ℕ := 12

-- The proof problem statement
theorem octagon_perimeter (n : ℕ) (h : is_octagon n) : n * side_length = 96 := by
  sorry

end octagon_perimeter_l2126_212643


namespace digit_in_base_l2126_212609

theorem digit_in_base (t : ℕ) (h1 : t ≤ 9) (h2 : 5 * 7 + t = t * 9 + 3) : t = 4 := by
  sorry

end digit_in_base_l2126_212609


namespace shaltaev_boltaev_proof_l2126_212610

variable (S B : ℝ)

axiom cond1 : 175 * S > 125 * B
axiom cond2 : 175 * S < 126 * B

theorem shaltaev_boltaev_proof : 3 * S + B ≥ 1 :=
by {
  sorry
}

end shaltaev_boltaev_proof_l2126_212610


namespace find_y_l2126_212691

theorem find_y (y : ℤ) (h : (15 + 26 + y) / 3 = 23) : y = 28 :=
by sorry

end find_y_l2126_212691


namespace sum_pairwise_relatively_prime_integers_eq_160_l2126_212627

theorem sum_pairwise_relatively_prime_integers_eq_160
  (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h_prod : a * b * c = 27000)
  (h_coprime_ab : Nat.gcd a b = 1)
  (h_coprime_bc : Nat.gcd b c = 1)
  (h_coprime_ac : Nat.gcd a c = 1) :
  a + b + c = 160 :=
by
  sorry

end sum_pairwise_relatively_prime_integers_eq_160_l2126_212627


namespace credit_extended_l2126_212632

noncomputable def automobile_installment_credit (total_consumer_credit : ℝ) : ℝ :=
  0.43 * total_consumer_credit

noncomputable def extended_by_finance_companies (auto_credit : ℝ) : ℝ :=
  0.25 * auto_credit

theorem credit_extended (total_consumer_credit : ℝ) (h : total_consumer_credit = 465.1162790697675) :
  extended_by_finance_companies (automobile_installment_credit total_consumer_credit) = 50.00 :=
by
  rw [h]
  sorry

end credit_extended_l2126_212632


namespace volume_of_revolved_region_l2126_212617

theorem volume_of_revolved_region :
  let R := {p : ℝ × ℝ | |8 - p.1| + p.2 ≤ 10 ∧ 3 * p.2 - p.1 ≥ 15}
  let volume := (1 / 3) * Real.pi * (7 / Real.sqrt 10)^2 * (7 * Real.sqrt 10 / 4)
  let m := 343
  let n := 12
  let p := 10
  m + n + p = 365 := by
  sorry

end volume_of_revolved_region_l2126_212617


namespace game_show_prize_guess_l2126_212681

noncomputable def total_possible_guesses : ℕ :=
  (Nat.choose 8 3) * (Nat.choose 5 3) * (Nat.choose 2 2) * (Nat.choose 7 3)

theorem game_show_prize_guess :
  total_possible_guesses = 19600 :=
by
  -- Omitted proof steps
  sorry

end game_show_prize_guess_l2126_212681


namespace tan_75_degrees_eq_l2126_212638

noncomputable def tan_75_degrees : ℝ := Real.tan (75 * Real.pi / 180)

theorem tan_75_degrees_eq : tan_75_degrees = 2 + Real.sqrt 3 := by
  sorry

end tan_75_degrees_eq_l2126_212638


namespace crackers_shared_equally_l2126_212672

theorem crackers_shared_equally : ∀ (matthew_crackers friends_crackers left_crackers friends : ℕ),
  matthew_crackers = 23 →
  left_crackers = 11 →
  friends = 2 →
  matthew_crackers - left_crackers = friends_crackers →
  friends_crackers = friends * 6 :=
by
  intro matthew_crackers friends_crackers left_crackers friends
  sorry

end crackers_shared_equally_l2126_212672


namespace certain_fraction_exists_l2126_212686

theorem certain_fraction_exists (a b : ℚ) (h : a / b = 3 / 4) :
  (a / b) / (1 / 5) = (3 / 4) / (2 / 5) :=
by
  sorry

end certain_fraction_exists_l2126_212686


namespace find_intersection_l2126_212648

def A : Set ℝ := {x | abs (x + 1) = x + 1}

def B : Set ℝ := {x | x^2 + x < 0}

def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem find_intersection : intersection A B = {x | -1 < x ∧ x < 0} :=
by
  sorry

end find_intersection_l2126_212648


namespace school_growth_difference_l2126_212658

theorem school_growth_difference (X Y : ℕ) (H₁ : Y = 2400)
  (H₂ : X + Y = 4000) : (X + 7 * X / 100 - X) - (Y + 3 * Y / 100 - Y) = 40 :=
by
  sorry

end school_growth_difference_l2126_212658


namespace car_R_average_speed_l2126_212661

theorem car_R_average_speed :
  ∃ (v : ℕ), (600 / v) - 2 = 600 / (v + 10) ∧ v = 50 :=
by sorry

end car_R_average_speed_l2126_212661


namespace brianne_january_savings_l2126_212660

theorem brianne_january_savings (S : ℝ) (h : 16 * S = 160) : S = 10 :=
sorry

end brianne_january_savings_l2126_212660


namespace transformed_sum_l2126_212652

open BigOperators -- Open namespace to use big operators like summation

theorem transformed_sum (n : ℕ) (x : Fin n → ℝ) (s : ℝ) 
  (h_sum : ∑ i, x i = s) : 
  ∑ i, ((3 * (x i + 10)) - 10) = 3 * s + 20 * n :=
by
  sorry

end transformed_sum_l2126_212652


namespace total_distance_traveled_l2126_212665

noncomputable def totalDistance
  (d1 d2 : ℝ) (s1 s2 : ℝ) (average_speed : ℝ) (total_time : ℝ) : ℝ := 
  average_speed * total_time

theorem total_distance_traveled :
  let d1 := 160
  let s1 := 64
  let d2 := 160
  let s2 := 80
  let average_speed := 71.11111111111111
  let total_time := d1 / s1 + d2 / s2
  totalDistance d1 d2 s1 s2 average_speed total_time = 320 :=
by
  -- This is the main statement theorem
  sorry

end total_distance_traveled_l2126_212665


namespace problem_statement_l2126_212602

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (-1, 1)

-- Define vector addition
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- Define dot product
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define perpendicular condition
def perp (u v : ℝ × ℝ) : Prop := dot_prod u v = 0

theorem problem_statement : perp (vec_add a b) a :=
by
  sorry

end problem_statement_l2126_212602


namespace range_of_a_plus_b_l2126_212654

theorem range_of_a_plus_b 
  (a b : ℝ)
  (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) ≥ -1) : 
  -1 ≤ a + b ∧ a + b ≤ 2 :=
sorry

end range_of_a_plus_b_l2126_212654


namespace arithmetic_geometric_sum_l2126_212628

theorem arithmetic_geometric_sum {n : ℕ} (a : ℕ → ℤ) (S : ℕ → ℚ) 
  (h1 : ∀ k, a (k + 1) = a k + 2) 
  (h2 : (a 1) * (a 1 + a 4) = (a 1 + a 2) ^ 2 / 2) :
  S n = 6 - (4 * n + 6) / 2^n :=
by
  sorry

end arithmetic_geometric_sum_l2126_212628


namespace solve_equation1_solve_equation2_l2126_212656

-- Define the first equation as a condition
def equation1 (x : ℝ) : Prop :=
  3 * x + 20 = 4 * x - 25

-- Prove that x = 45 satisfies equation1
theorem solve_equation1 : equation1 45 :=
by 
  -- Proof steps would go here
  sorry

-- Define the second equation as a condition
def equation2 (x : ℝ) : Prop :=
  (2 * x - 1) / 3 = 1 - (2 * x - 1) / 6

-- Prove that x = 3/2 satisfies equation2
theorem solve_equation2 : equation2 (3 / 2) :=
by 
  -- Proof steps would go here
  sorry

end solve_equation1_solve_equation2_l2126_212656


namespace sum_first_8_terms_64_l2126_212606

-- Define the problem conditions
def isArithmeticSeq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def isGeometricSeq (a : ℕ → ℤ) : Prop :=
  ∀ (m n k : ℕ), m < n → n < k → (a n)^2 = a m * a k

-- Given arithmetic sequence with a common difference 2
def arithmeticSeqWithDiff2 (a : ℕ → ℤ) : Prop :=
  isArithmeticSeq a ∧ (∃ d : ℤ, d = 2 ∧ ∀ (n : ℕ), a (n + 1) = a n + d)

-- Given a₁, a₂, a₅ form a geometric sequence
def a1_a2_a5_formGeometricSeq (a: ℕ → ℤ) : Prop :=
  (a 2)^2 = (a 1) * (a 5)

-- Sum of the first 8 terms of the arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a (n - 1)) / 2

-- Main statement
theorem sum_first_8_terms_64 (a : ℕ → ℤ) (h1 : arithmeticSeqWithDiff2 a) (h2 : a1_a2_a5_formGeometricSeq a) : 
  sum_of_first_n_terms a 8 = 64 := 
sorry

end sum_first_8_terms_64_l2126_212606


namespace sum_of_7a_and_3b_l2126_212637

theorem sum_of_7a_and_3b (a b : ℤ) (h : a + b = 1998) : 7 * a + 3 * b ≠ 6799 :=
by sorry

end sum_of_7a_and_3b_l2126_212637


namespace percent_yz_of_x_l2126_212678

theorem percent_yz_of_x (x y z : ℝ) 
  (h₁ : 0.6 * (x - y) = 0.3 * (x + y))
  (h₂ : 0.4 * (x + z) = 0.2 * (y + z))
  (h₃ : 0.5 * (x - z) = 0.25 * (x + y + z)) :
  y + z = 0.0 * x :=
sorry

end percent_yz_of_x_l2126_212678


namespace solve_inequalities_l2126_212640

theorem solve_inequalities (x : ℤ) :
  (1 ≤ x ∧ x < 3) ↔ 
  ((↑x - 1) / 2 < (↑x : ℝ) / 3 ∧ 2 * (↑x : ℝ) - 5 ≤ 3 * (↑x : ℝ) - 6) :=
by
  sorry

end solve_inequalities_l2126_212640


namespace total_days_off_l2126_212677

-- Definitions for the problem conditions
def days_off_personal (months_in_year : ℕ) (days_per_month : ℕ) : ℕ :=
  days_per_month * months_in_year

def days_off_professional (months_in_year : ℕ) (days_per_month : ℕ) : ℕ :=
  days_per_month * months_in_year

def days_off_teambuilding (quarters_in_year : ℕ) (days_per_quarter : ℕ) : ℕ :=
  days_per_quarter * quarters_in_year

-- Main theorem to prove
theorem total_days_off
  (months_in_year : ℕ) (quarters_in_year : ℕ)
  (days_per_month_personal : ℕ) (days_per_month_professional : ℕ) (days_per_quarter_teambuilding: ℕ)
  (h_months : months_in_year = 12) (h_quarters : quarters_in_year = 4) 
  (h_days_personal : days_per_month_personal = 4) (h_days_professional : days_per_month_professional = 2) (h_days_teambuilding : days_per_quarter_teambuilding = 1) :
  days_off_personal months_in_year days_per_month_personal
  + days_off_professional months_in_year days_per_month_professional
  + days_off_teambuilding quarters_in_year days_per_quarter_teambuilding
  = 76 := 
by {
  -- Calculation
  sorry
}

end total_days_off_l2126_212677
