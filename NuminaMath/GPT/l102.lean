import Mathlib

namespace intersection_eq_l102_102828

def A := {x : ℝ | |x| = x}
def B := {x : ℝ | x^2 + x ≥ 0}

theorem intersection_eq : A ∩ B = {x : ℝ | 0 ≤ x} := by
  sorry

end intersection_eq_l102_102828


namespace solve_for_x_l102_102493

noncomputable def g (x : ℝ) : ℝ := (Real.sqrt (x + 2) / 5) ^ (1 / 4)

theorem solve_for_x : 
  ∃ x : ℝ, g (3 * x) = 3 * g x ∧ x = -404 / 201 := 
by {
  sorry
}

end solve_for_x_l102_102493


namespace two_roots_iff_a_gt_neg1_l102_102188

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l102_102188


namespace ratio_of_buyers_l102_102290

theorem ratio_of_buyers (B Y T : ℕ) (hB : B = 50) 
  (hT : T = Y + 40) (hTotal : B + Y + T = 140) : 
  (Y : ℚ) / B = 1 / 2 :=
by 
  sorry

end ratio_of_buyers_l102_102290


namespace time_comparison_l102_102434

noncomputable def pedestrian_speed : Real := 6.5
noncomputable def cyclist_speed : Real := 20.0
noncomputable def distance_between_points_B_A : Real := 4 * Real.pi - 6.5
noncomputable def alley_distance : Real := 4 * Real.pi - 6.5
noncomputable def combined_speed_3 : Real := pedestrian_speed + cyclist_speed
noncomputable def combined_speed_2 : Real := 21.5
noncomputable def time_scenario_3 : Real := (4 * Real.pi - 6.5) / combined_speed_3
noncomputable def time_scenario_2 : Real := (10.5 - 2 * Real.pi) / combined_speed_2

theorem time_comparison : time_scenario_2 < time_scenario_3 :=
by
  sorry

end time_comparison_l102_102434


namespace union_complement_eq_l102_102694

def U := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 5}

def complement (univ : Set ℕ) (s : Set ℕ) : Set ℕ :=
  {x ∈ univ | x ∉ s}

theorem union_complement_eq :
  N ∪ (complement U M) = {2, 3, 5} :=
by sorry

end union_complement_eq_l102_102694


namespace carol_age_l102_102555

theorem carol_age (B C : ℕ) (h1 : B + C = 66) (h2 : C = 3 * B + 2) : C = 50 :=
sorry

end carol_age_l102_102555


namespace power_increased_by_four_l102_102895

-- Definitions from the conditions
variables (F k v : ℝ) (initial_force_eq_resistive : F = k * v)

-- Define the new conditions with double the force
variables (new_force : ℝ) (new_velocity : ℝ) (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F)

-- The theorem statement
theorem power_increased_by_four (initial_force_eq_resistive : F = k * v) 
  (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F) :
  new_velocity = 2 * v → 
  (new_force * new_velocity) = 4 * (F * v) :=
sorry

end power_increased_by_four_l102_102895


namespace alice_speed_exceed_l102_102427

theorem alice_speed_exceed (d : ℝ) (t₁ t₂ : ℝ) (t₃ : ℝ) :
  d = 220 →
  t₁ = 220 / 40 →
  t₂ = t₁ - 0.5 →
  t₃ = 220 / t₂ →
  t₃ = 44 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_speed_exceed_l102_102427


namespace total_volume_of_four_cubes_l102_102299

theorem total_volume_of_four_cubes (s : ℝ) (h_s : s = 5) : 4 * s^3 = 500 :=
by
  sorry

end total_volume_of_four_cubes_l102_102299


namespace sin_2A_value_l102_102489

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h₁ : a / (2 * Real.cos A) = b / (3 * Real.cos B))
variable (h₂ : b / (3 * Real.cos B) = c / (6 * Real.cos C))

theorem sin_2A_value (h₃ : a / (2 * Real.cos A) = c / (6 * Real.cos C)) :
  Real.sin (2 * A) = 3 * Real.sqrt 11 / 10 := sorry

end sin_2A_value_l102_102489


namespace number_of_questions_in_test_l102_102864

variable (n : ℕ) -- the total number of questions
variable (correct_answers : ℕ) -- the number of correct answers
variable (sections : ℕ) -- number of sections in the test
variable (questions_per_section : ℕ) -- number of questions per section
variable (percentage_correct : ℚ) -- percentage of correct answers

-- Given conditions
def conditions := 
  correct_answers = 32 ∧ 
  sections = 5 ∧ 
  questions_per_section * sections = n ∧ 
  (70 : ℚ) < percentage_correct ∧ 
  percentage_correct < 77 ∧ 
  percentage_correct * n = 3200

-- The main statement to prove
theorem number_of_questions_in_test : conditions n correct_answers sections questions_per_section percentage_correct → 
  n = 45 :=
by
  sorry

end number_of_questions_in_test_l102_102864


namespace opposite_neg_two_l102_102281

def opposite (x : Int) : Int := -x

theorem opposite_neg_two : opposite (-2) = 2 := by
  sorry

end opposite_neg_two_l102_102281


namespace calculate_ggg1_l102_102520

def g (x : ℕ) : ℕ := 7 * x + 3

theorem calculate_ggg1 : g (g (g 1)) = 514 := 
by
  sorry

end calculate_ggg1_l102_102520


namespace sheets_taken_l102_102976

noncomputable def remaining_sheets_mean (b c : ℕ) : ℚ :=
  (b * (2 * b + 1) + (100 - 2 * (b + c)) * (2 * (b + c) + 101)) / 2 / (100 - 2 * c)

theorem sheets_taken (b c : ℕ) (h1 : 100 = 2 * 50) 
(h2 : ∀ n, n > 0 → 2 * n = n + n) 
(hmean : remaining_sheets_mean b c = 31) : 
  c = 17 := 
sorry

end sheets_taken_l102_102976


namespace largest_possible_d_l102_102969

theorem largest_possible_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 := 
sorry

end largest_possible_d_l102_102969


namespace one_thirds_in_nine_halves_l102_102654

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 13 := by
  sorry

end one_thirds_in_nine_halves_l102_102654


namespace problem_statement_l102_102910

theorem problem_statement :
  (∃ n : ℕ, n = 8 ∧ ∀ a : ℕ, a < 15 → (∃ x : ℤ, a * x ≡ 1 [MOD 15]) ↔ gcd a 15 = 1) :=
by
  use 8
  intro a
  intro ha
  split
  sorry

end problem_statement_l102_102910


namespace determine_functions_l102_102178

noncomputable def f : (ℝ → ℝ) := sorry

theorem determine_functions (f : ℝ → ℝ)
  (h_domain: ∀ x, 0 < x → 0 < f x)
  (h_eq: ∀ w x y z, 0 < w → 0 < x → 0 < y → 0 < z → w * x = y * z →
    (f w)^2 + (f x)^2 = (f (y^2) + f (z^2)) * (w^2 + x^2) / (y^2 + z^2)) :
  (∀ x, 0 < x → (f x = x ∨ f x = 1 / x)) :=
by
  intros x hx
  sorry

end determine_functions_l102_102178


namespace average_speed_round_trip_l102_102309

variable (D : ℝ) (u v : ℝ)
  
theorem average_speed_round_trip (h1 : u = 96) (h2 : v = 88) : 
  (2 * u * v) / (u + v) = 91.73913043 := 
by 
  sorry

end average_speed_round_trip_l102_102309


namespace find_amount_with_r_l102_102003

variable (p q r : ℝ)

-- Condition 1: p, q, and r have Rs. 6000 among themselves.
def total_amount : Prop := p + q + r = 6000

-- Condition 2: r has two-thirds of the total amount with p and q.
def r_amount : Prop := r = (2 / 3) * (p + q)

theorem find_amount_with_r (h1 : total_amount p q r) (h2 : r_amount p q r) : r = 2400 := by
  sorry

end find_amount_with_r_l102_102003


namespace symmetric_circle_eq_of_given_circle_eq_l102_102549

theorem symmetric_circle_eq_of_given_circle_eq
  (x y : ℝ)
  (eq1 : (x - 1)^2 + (y - 2)^2 = 1)
  (line_eq : y = x) :
  (x - 2)^2 + (y - 1)^2 = 1 := by
  sorry

end symmetric_circle_eq_of_given_circle_eq_l102_102549


namespace range_of_a_l102_102224

variable (a : ℝ)
variable (x y : ℝ)

def system_of_equations := 
  (5 * x + 2 * y = 11 * a + 18) ∧ 
  (2 * x - 3 * y = 12 * a - 8) ∧
  (x > 0) ∧ 
  (y > 0)

theorem range_of_a (h : system_of_equations a x y) : 
  - (2:ℝ) / 3 < a ∧ a < 2 :=
sorry

end range_of_a_l102_102224


namespace sum_of_coefficients_l102_102214

def polynomial : ℕ → ℤ
| 8 := -3
| 5 := 6
| 3 := -12
| 0 := 45

theorem sum_of_coefficients :
  polynomial 8 + polynomial 5 + polynomial 3 + polynomial 0 = 45 :=
by
  sorry

end sum_of_coefficients_l102_102214


namespace automobile_credit_percentage_at_end_of_year_x_l102_102331

noncomputable def calculate_percentage (auto_finance_credit : ℝ) (consumer_credit : ℝ) : ℝ :=
  let total_auto_credit := 3 * auto_finance_credit
  (total_auto_credit / consumer_credit) * 100

theorem automobile_credit_percentage_at_end_of_year_x :
  let auto_finance_credit := 35
  let total_consumer_credit := 291.6666666666667
  calculate_percentage auto_finance_credit total_consumer_credit = 36 := by
sorry

end automobile_credit_percentage_at_end_of_year_x_l102_102331


namespace infinite_power_tower_equation_l102_102415

noncomputable def infinite_power_tower (x : ℝ) : ℝ :=
  x ^ x ^ x ^ x ^ x -- continues infinitely

theorem infinite_power_tower_equation (x : ℝ) (h_pos : 0 < x) (h_eq : infinite_power_tower x = 2) : x = Real.sqrt 2 :=
  sorry

end infinite_power_tower_equation_l102_102415


namespace range_of_solutions_l102_102615

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end range_of_solutions_l102_102615


namespace find_x_l102_102524

def star (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + q.1, p.2 - q.2)

theorem find_x : ∃ x : ℤ, ∃ y : ℤ, star (4, 5) (1, 3) = star (x, y) (2, 1) ∧ x = 3 :=
by 
  sorry

end find_x_l102_102524


namespace relationship_between_problems_geometry_problem_count_steve_questions_l102_102407

variable (x y W A G : ℕ)

def word_problems (x : ℕ) : ℕ := x / 2
def addition_and_subtraction_problems (x : ℕ) : ℕ := x / 3
def geometry_problems (x W A : ℕ) : ℕ := x - W - A

theorem relationship_between_problems :
  W = word_problems x ∧
  A = addition_and_subtraction_problems x ∧
  G = geometry_problems x W A →
  W + A + G = x :=
by
  sorry

theorem geometry_problem_count :
  W = word_problems x ∧
  A = addition_and_subtraction_problems x →
  G = geometry_problems x W A →
  G = x / 6 :=
by
  sorry

theorem steve_questions :
  y = x / 2 - 4 :=
by
  sorry

end relationship_between_problems_geometry_problem_count_steve_questions_l102_102407


namespace circle_equation_determine_a_l102_102818

noncomputable theory

open Real

def point := (ℝ × ℝ)

def circle (D E F : ℝ) := ∀ p : point, (p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0)

def distance (p1 p2 : point) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def orthogonal (p1 p2 p3 : point) : Prop := (p2.1 - p1.1) * (p3.1 - p1.1) + (p2.2 - p1.2) * (p3.2 - p1.2) = 0

theorem circle_equation
  (P Q R : point)
  (C : D E F : ℝ)
  (h1 : P = (3 + 2*sqrt 2, 0))
  (h2 : Q = (3 - 2*sqrt 2, 0))
  (h3 : R = (0, 1))
  (h_circle : circle D E F)
  :
  circle D E F :=
by
  sorry

theorem determine_a
  (a : ℝ)
  (A B O : point)
  (circle_eq : x^2 + y^2 - 6 * x - 2 * y + 1 = 0)
  (line_eq : A.1 - A.2 + a = 0 ∧ B.1 - B.2 + a = 0)
  (orth : orthogonal O A B)
  :
  a = 1 ∨ a = -5 :=
by
  sorry

end circle_equation_determine_a_l102_102818


namespace average_rainfall_per_hour_in_June_1882_l102_102948

open Real

theorem average_rainfall_per_hour_in_June_1882 
  (total_rainfall : ℝ) (days_in_June : ℕ) (hours_per_day : ℕ)
  (H1 : total_rainfall = 450) (H2 : days_in_June = 30) (H3 : hours_per_day = 24) :
  total_rainfall / (days_in_June * hours_per_day) = 5 / 8 :=
by
  sorry

end average_rainfall_per_hour_in_June_1882_l102_102948


namespace parabola_sequence_l102_102557

theorem parabola_sequence (m: ℝ) (n: ℕ):
  (∀ t s: ℝ, t * s = -1/4) →
  (∀ x y: ℝ, y^2 = (1/(3^n)) * m * (x - (m / 4) * (1 - (1/(3^n))))) :=
sorry

end parabola_sequence_l102_102557


namespace eval_expression_l102_102350

theorem eval_expression (x y z : ℝ) 
  (h1 : z = y - 11) 
  (h2 : y = x + 3) 
  (h3 : x = 5)
  (h4 : x + 2 ≠ 0) 
  (h5 : y - 3 ≠ 0) 
  (h6 : z + 7 ≠ 0) : 
  ( (x + 3) / (x + 2) * (y - 1) / (y - 3) * (z + 9) / (z + 7) ) = 2.4 := 
by
  sorry

end eval_expression_l102_102350


namespace avg_weight_section_B_l102_102417

theorem avg_weight_section_B 
  (W_B : ℝ) 
  (num_students_A : ℕ := 36) 
  (avg_weight_A : ℝ := 30) 
  (num_students_B : ℕ := 24) 
  (total_students : ℕ := 60) 
  (avg_weight_class : ℝ := 30) 
  (h1 : num_students_A * avg_weight_A + num_students_B * W_B = total_students * avg_weight_class) :
  W_B = 30 :=
sorry

end avg_weight_section_B_l102_102417


namespace lines_parallel_if_perpendicular_to_plane_l102_102016

variables {α β γ : Plane} {m n : Line}

-- Define the properties of perpendicular lines to planes and parallel lines
def perpendicular_to (l : Line) (p : Plane) : Prop := 
sorry -- definition skipped

def parallel_to (l1 l2 : Line) : Prop := 
sorry -- definition skipped

-- Theorem Statement (equivalent translation of the given question and its correct answer)
theorem lines_parallel_if_perpendicular_to_plane 
  (h1 : perpendicular_to m α) 
  (h2 : perpendicular_to n α) : parallel_to m n :=
sorry

end lines_parallel_if_perpendicular_to_plane_l102_102016


namespace max_product_sum_1976_l102_102207

theorem max_product_sum_1976 (a : ℕ) (P : ℕ → ℕ) (h : ∀ n, P n > 0 → a = 1976) :
  ∃ (k l : ℕ), (2 * k + 3 * l = 1976) ∧ (P 1976 = 2 * 3 ^ 658) := sorry

end max_product_sum_1976_l102_102207


namespace ellipse_properties_l102_102924

-- Define the conditions of the problem
def condition1 (E : ℝ → ℝ → Prop) : Prop :=
  (E 2 1) ∧ (E (2 * Real.sqrt 2) 0) ∧ (∀ x y, E x y → (x^2 / 8 + y^2 / 2 = 1))

-- Define the statement to be proved
theorem ellipse_properties :
  ∃ E : ℝ → ℝ → Prop,
  condition1 E ∧
  (∀ (t : ℝ) (A B : ℝ × ℝ), 
    (A = (x, y)) ∧ (B = (x', y')) → y = (1 / 2) * x + t ∧
    y' = (1 / 2) * x' + t →
    let k1 := (y - 1) / (x - 2) in
    let k2 := (y' - 1) / (x' - 2) in
    k1 + k2 = 0) :=
sorry

end ellipse_properties_l102_102924


namespace theater_queue_arrangement_l102_102670

theorem theater_queue_arrangement : 
  let n := 7 -- total number of people
  let pair := 2 -- Alice and Bob considered as one unit
  let k := n - pair + 1 -- reducing to 6 units
  (Nat.factorial k) * (Nat.factorial pair) = 1440 :=
by
  let n := 7
  let pair := 2
  let k := n - pair + 1
  have h_k : k = 6 := rfl
  have h1 : Nat.factorial k = Nat.factorial 6 := by rw [h_k]
  have h_fac6 : Nat.factorial 6 = 720 := by norm_num
  have h_fac2 : Nat.factorial 2 = 2 := by norm_num
  rw [h1, h_fac6, h_fac2]
  norm_num
  exact rfl

end theater_queue_arrangement_l102_102670


namespace Shara_savings_l102_102590

theorem Shara_savings (P : ℝ) (d : ℝ) (paid : ℝ):
  d = 0.08 → paid = 184 → P = 200 → (P * (1 - d) = paid) → (P - paid = 16) :=
by
  intros hd hpaid hP heq
  -- It follows from the conditions given
  sorry

end Shara_savings_l102_102590


namespace slope_y_intercept_sum_l102_102359

theorem slope_y_intercept_sum 
  (m b : ℝ) 
  (h1 : (2 : ℝ) * m + b = -1) 
  (h2 : (5 : ℝ) * m + b = 2) : 
  m + b = -2 := 
sorry

end slope_y_intercept_sum_l102_102359


namespace repeating_decimal_to_fraction_l102_102351

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + (6 / 10) / 9) : x = 11 / 30 :=
by
  sorry

end repeating_decimal_to_fraction_l102_102351


namespace tutors_next_together_in_360_days_l102_102918

open Nat

-- Define the intervals for each tutor
def evan_interval := 5
def fiona_interval := 6
def george_interval := 9
def hannah_interval := 8
def ian_interval := 10

-- Statement to prove
theorem tutors_next_together_in_360_days :
  Nat.lcm (Nat.lcm evan_interval fiona_interval) (Nat.lcm george_interval (Nat.lcm hannah_interval ian_interval)) = 360 :=
by
  sorry

end tutors_next_together_in_360_days_l102_102918


namespace container_volume_ratio_l102_102166

theorem container_volume_ratio
  (A B C : ℝ)
  (h1 : (3 / 4) * A - (5 / 8) * B = (7 / 8) * C - (1 / 2) * C)
  (h2 : B =  (5 / 8) * B)
  (h3 : (5 / 8) * B =  (3 / 8) * C)
  (h4 : A =  (24 / 40) * C) : 
  A / C = 4 / 5 := sorry

end container_volume_ratio_l102_102166


namespace range_of_m_l102_102065

theorem range_of_m (x m : ℝ)
  (h1 : (x + 2) / (10 - x) ≥ 0)
  (h2 : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (h3 : m < 0)
  (h4 : ∀ (x : ℝ), (x + 2) / (10 - x) ≥ 0 → (x^2 - 2 * x + 1 - m^2 ≤ 0)) :
  -3 ≤ m ∧ m < 0 :=
sorry

end range_of_m_l102_102065


namespace rotate_right_triangle_along_right_angle_produces_cone_l102_102993

-- Define a right triangle and the conditions for its rotation
structure RightTriangle (α β γ : ℝ) :=
  (zero_angle : α = 0)
  (ninety_angle_1 : β = 90)
  (ninety_angle_2 : γ = 90)
  (sum_180 : α + β + γ = 180)

-- Define the theorem for the resulting shape when rotating the right triangle
theorem rotate_right_triangle_along_right_angle_produces_cone
  (T : RightTriangle α β γ) (line_of_rotation_contains_right_angle : α = 90 ∨ β = 90 ∨ γ = 90) :
  ∃ shape, shape = "cone" :=
sorry

end rotate_right_triangle_along_right_angle_produces_cone_l102_102993


namespace total_amount_l102_102053

-- Definitions based on the problem conditions
def jack_amount : ℕ := 26
def ben_amount : ℕ := jack_amount - 9
def eric_amount : ℕ := ben_amount - 10

-- Proof statement
theorem total_amount : jack_amount + ben_amount + eric_amount = 50 :=
by
  -- Sorry serves as a placeholder for the actual proof
  sorry

end total_amount_l102_102053


namespace compute_x_squared_y_plus_xy_squared_l102_102927

theorem compute_x_squared_y_plus_xy_squared 
  (x y : ℝ)
  (h1 : (1 / x) + (1 / y) = 4)
  (h2 : x * y + x + y = 7) :
  x^2 * y + x * y^2 = 49 := 
  sorry

end compute_x_squared_y_plus_xy_squared_l102_102927


namespace correct_answers_is_36_l102_102002

noncomputable def num_correct_answers (c w : ℕ) : Prop :=
  (c + w = 50) ∧ (4 * c - w = 130)

theorem correct_answers_is_36 (c w : ℕ) (h : num_correct_answers c w) : c = 36 :=
by
  sorry

end correct_answers_is_36_l102_102002


namespace min_balls_to_draw_l102_102128

theorem min_balls_to_draw (black white red : ℕ) (h_black : black = 10) (h_white : white = 9) (h_red : red = 8) :
  ∃ n, n = 20 ∧
  ∀ k, (k < 20) → ¬ (∃ b w r, b + w + r = k ∧ b ≤ black ∧ w ≤ white ∧ r ≤ red ∧ r > 0 ∧ w > 0) :=
by {
  sorry
}

end min_balls_to_draw_l102_102128


namespace determine_parallel_planes_l102_102112

-- Definition of planes and lines with parallelism
structure Plane :=
  (points : Set (ℝ × ℝ × ℝ))

structure Line :=
  (point1 point2 : ℝ × ℝ × ℝ)
  (in_plane : Plane)

def parallel_planes (α β : Plane) : Prop :=
  ∀ (l1 : Line) (l2 : Line), l1.in_plane = α → l2.in_plane = β → (l1 = l2)

def parallel_lines (l1 l2 : Line) : Prop :=
  ∀ p1 p2, l1.point1 = p1 → l1.point2 = p2 → l2.point1 = p1 → l2.point2 = p2


theorem determine_parallel_planes (α β γ : Plane)
  (h1 : parallel_planes γ α)
  (h2 : parallel_planes γ β)
  (l1 l2 : Line)
  (l1_in_alpha : l1.in_plane = α)
  (l2_in_alpha : l2.in_plane = α)
  (parallel_l1_l2 : ¬ (l1 = l2) → parallel_lines l1 l2)
  (l1_parallel_beta : ∀ l, l.in_plane = β → parallel_lines l l1)
  (l2_parallel_beta : ∀ l, l.in_plane = β → parallel_lines l l2) :
  parallel_planes α β := 
sorry

end determine_parallel_planes_l102_102112


namespace total_flour_needed_l102_102515

noncomputable def katie_flour : ℝ := 3

noncomputable def sheila_flour : ℝ := katie_flour + 2

noncomputable def john_flour : ℝ := 1.5 * sheila_flour

theorem total_flour_needed :
  katie_flour + sheila_flour + john_flour = 15.5 :=
by
  sorry

end total_flour_needed_l102_102515


namespace cone_prism_volume_ratio_l102_102892

-- Define the volumes and the ratio proof problem
theorem cone_prism_volume_ratio (r h : ℝ) (h_pos : 0 < r) (h_height : 0 < h) :
    let V_cone := (1 / 12) * π * r^2 * h
    let V_prism := 3 * r^2 * h
    (V_cone / V_prism) = (π / 36) :=
by
    -- Here we define the volumes of the cone and prism as given in the problem
    let V_cone := (1 / 12) * π * r^2 * h
    let V_prism := 3 * r^2 * h
    -- We then assert the ratio condition based on the solution
    sorry

end cone_prism_volume_ratio_l102_102892


namespace one_thirds_in_nine_halves_l102_102655

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 13 := by
  sorry

end one_thirds_in_nine_halves_l102_102655


namespace apples_after_operations_l102_102719

-- Define the initial conditions
def initial_apples : ℕ := 38
def used_apples : ℕ := 20
def bought_apples : ℕ := 28

-- State the theorem we want to prove
theorem apples_after_operations : initial_apples - used_apples + bought_apples = 46 :=
by
  sorry

end apples_after_operations_l102_102719


namespace cos_identity_l102_102423

theorem cos_identity (α : ℝ) : 
  3.4028 * (Real.cos α)^4 + 4 * (Real.cos α)^3 - 8 * (Real.cos α)^2 - 3 * (Real.cos α) + 1 = 
  2 * (Real.cos (7 * α / 2)) * (Real.cos (α / 2)) := 
by sorry

end cos_identity_l102_102423


namespace equation_two_roots_iff_l102_102203

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l102_102203


namespace count_valid_B_l102_102706

open Finset

def valid_B (B : Finset ℕ) : Prop :=
  B ⊆ {1, 2, 3, 4, 5, 6, 7, 8} ∧
  B.card = 3 ∧
  ∀ a b ∈ B, a ≠ b → a + b ≠ 9

theorem count_valid_B : (univ.filter valid_B).card = 10 :=
sorry

end count_valid_B_l102_102706


namespace compare_neg_fractions_l102_102453

theorem compare_neg_fractions : (- (3 / 2) < -1) :=
by sorry

end compare_neg_fractions_l102_102453


namespace total_ticket_cost_l102_102443

theorem total_ticket_cost (x y : ℕ) 
  (h1 : x + y = 380) 
  (h2 : y = x + 240) 
  (cost_orchestra : ℕ := 12) 
  (cost_balcony : ℕ := 8): 
  12 * x + 8 * y = 3320 := 
by 
  sorry

end total_ticket_cost_l102_102443


namespace relay_race_total_distance_l102_102838

theorem relay_race_total_distance
  (Sadie_speed : ℝ) (Sadie_time : ℝ) (Ariana_speed : ℝ) (Ariana_time : ℝ) (Sarah_speed : ℝ) (total_race_time : ℝ)
  (h1 : Sadie_speed = 3) (h2 : Sadie_time = 2)
  (h3 : Ariana_speed = 6) (h4 : Ariana_time = 0.5)
  (h5 : Sarah_speed = 4) (h6 : total_race_time = 4.5) :
  (Sadie_speed * Sadie_time + Ariana_speed * Ariana_time + Sarah_speed * (total_race_time - (Sadie_time + Ariana_time))) = 17 :=
by
  sorry

end relay_race_total_distance_l102_102838


namespace meal_cost_per_person_l102_102328

/-
Problem Statement:
Prove that the cost per meal is $3 given the conditions:
- There are 2 adults and 5 children.
- The total bill is $21.
-/

theorem meal_cost_per_person (total_adults : ℕ) (total_children : ℕ) (total_bill : ℝ) 
(total_people : ℕ) (cost_per_meal : ℝ) : 
total_adults = 2 → total_children = 5 → total_bill = 21 → total_people = total_adults + total_children →
cost_per_meal = total_bill / total_people → 
cost_per_meal = 3 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end meal_cost_per_person_l102_102328


namespace passing_percentage_correct_l102_102815

-- Define the conditions
def max_marks : ℕ := 500
def candidate_marks : ℕ := 180
def fail_by : ℕ := 45

-- Define the passing_marks based on given conditions
def passing_marks : ℕ := candidate_marks + fail_by

-- Theorem to prove: the passing percentage is 45%
theorem passing_percentage_correct : 
  (passing_marks / max_marks) * 100 = 45 := 
sorry

end passing_percentage_correct_l102_102815


namespace total_flour_l102_102528

def cups_of_flour (flour_added : ℕ) (flour_needed : ℕ) : ℕ :=
  flour_added + flour_needed

theorem total_flour :
  ∀ (flour_added flour_needed : ℕ), flour_added = 3 → flour_needed = 6 → cups_of_flour flour_added flour_needed = 9 :=
by 
  intros flour_added flour_needed h_added h_needed
  rw [h_added, h_needed]
  rfl

end total_flour_l102_102528


namespace new_supervisor_salary_correct_l102_102577

noncomputable def salary_new_supervisor
  (avg_salary_old : ℝ)
  (old_supervisor_salary : ℝ)
  (avg_salary_new : ℝ)
  (workers_count : ℝ)
  (total_salary_workers : ℝ := (avg_salary_old * (workers_count + 1)) - old_supervisor_salary)
  (new_supervisor_salary : ℝ := (avg_salary_new * (workers_count + 1)) - total_salary_workers)
  : ℝ :=
  new_supervisor_salary

theorem new_supervisor_salary_correct :
  salary_new_supervisor 430 870 420 8 = 780 :=
by
  simp [salary_new_supervisor]
  sorry

end new_supervisor_salary_correct_l102_102577


namespace Masha_gathers_5_mushrooms_l102_102474

def mushrooms_collected (B G : list ℕ) : ℕ :=
  B.sum + G.sum

def unique_girls (G : list ℕ) : Prop :=
  ∀ i j, i ≠ j → G.get_or_else i 0 ≠ G.get_or_else j 0

def at_least_43_mushrooms (B : list ℕ) : Prop :=
  ∀ i j k, B.get_or_else i 0 + B.get_or_else j 0 + B.get_or_else k 0 ≥ 43 

def within_5_times (A : list ℕ) : Prop :=
  ∀ i j, (max (A.get_or_else i 0) (A.get_or_else j 0)) ≤ 5 * (min (A.get_or_else i 0) (A.get_or_else j 0))

noncomputable def Masha_collects_most_mushrooms (G : list ℕ) : ℕ :=
  G.maximum_def 0

theorem Masha_gathers_5_mushrooms (B G : list ℕ) (h_size_B : B.length = 4) (h_size_G : G.length = 3) 
  (h_total : mushrooms_collected B G = 70) 
  (h_unique : unique_girls G) 
  (h_at_least_43 : at_least_43_mushrooms B) 
  (h_within_5times : within_5_times (B ++ G)) : 
  Masha_collects_most_mushrooms G = 5 := 
sorry

end Masha_gathers_5_mushrooms_l102_102474


namespace relationship_of_arithmetic_progression_l102_102664

theorem relationship_of_arithmetic_progression (x y z d : ℝ) (h1 : x + (y - z) + d = y + (z - x))
    (h2 : y + (z - x) + d = z + (x - y))
    (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
    x = y + d / 2 ∧ z = y + d := by
  sorry

end relationship_of_arithmetic_progression_l102_102664


namespace ravi_work_alone_days_l102_102269

theorem ravi_work_alone_days (R : ℝ) (h1 : 1 / 75 + 1 / R = 1 / 30) : R = 50 :=
sorry

end ravi_work_alone_days_l102_102269


namespace problem1_problem2_l102_102149

namespace ProofProblems

-- Problem 1: Prove the inequality
theorem problem1 (x : ℝ) (h : x + |2 * x - 1| < 3) : -2 < x ∧ x < 4 / 3 := 
sorry

-- Problem 2: Prove the value of x + y + z 
theorem problem2 (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2 * y + 3 * z = Real.sqrt 14) : 
  x + y + z = 3 * Real.sqrt 14 / 7 := 
sorry

end ProofProblems

end problem1_problem2_l102_102149


namespace second_divisor_l102_102358

theorem second_divisor (x : ℕ) : (282 % 31 = 3) ∧ (282 % x = 3) → x = 9 :=
by
  sorry

end second_divisor_l102_102358


namespace geometric_sequence_l102_102803

theorem geometric_sequence (q : ℝ) (a : ℕ → ℝ) (h1 : q > 0) (h2 : a 2 = 1)
  (h3 : a 2 * a 10 = 2 * (a 5)^2) : ∀ n, a n = 2^((n-2:ℝ)/2) := by
  sorry

end geometric_sequence_l102_102803


namespace gcd_of_sum_and_squares_l102_102980

theorem gcd_of_sum_and_squares {a b : ℤ} (h : Int.gcd a b = 1) : 
  Int.gcd (a^2 + b^2) (a + b) = 1 ∨ Int.gcd (a^2 + b^2) (a + b) = 2 := 
by
  sorry

end gcd_of_sum_and_squares_l102_102980


namespace wash_time_difference_l102_102699

def C := 30
def T := 2 * C
def total_time := 135

theorem wash_time_difference :
  ∃ S, C + T + S = total_time ∧ T - S = 15 :=
by
  sorry

end wash_time_difference_l102_102699


namespace washing_machine_capacity_l102_102537

theorem washing_machine_capacity 
  (shirts : ℕ) (sweaters : ℕ) (loads : ℕ) (total_clothing : ℕ) (n : ℕ)
  (h1 : shirts = 43) (h2 : sweaters = 2) (h3 : loads = 9)
  (h4 : total_clothing = shirts + sweaters)
  (h5 : total_clothing / loads = n) :
  n = 5 :=
sorry

end washing_machine_capacity_l102_102537


namespace tens_digit_of_8_pow_2048_l102_102875

theorem tens_digit_of_8_pow_2048 : (8^2048 % 100) / 10 = 8 := 
by
  sorry

end tens_digit_of_8_pow_2048_l102_102875


namespace factor_square_difference_l102_102353

theorem factor_square_difference (t : ℝ) : t^2 - 121 = (t - 11) * (t + 11) := 
  sorry

end factor_square_difference_l102_102353


namespace boat_travel_time_difference_l102_102430

noncomputable def travel_time_difference (v : ℝ) : ℝ :=
  let d := 90
  let t_downstream := 2.5191640969412834
  let t_upstream := d / (v - 3)
  t_upstream - t_downstream

theorem boat_travel_time_difference :
  ∃ v : ℝ, travel_time_difference v = 0.5088359030587166 := 
by
  sorry

end boat_travel_time_difference_l102_102430


namespace son_l102_102145

theorem son's_age (S M : ℕ) (h₁ : M = S + 25) (h₂ : M + 2 = 2 * (S + 2)) : S = 23 := by
  sorry

end son_l102_102145


namespace albums_either_but_not_both_l102_102762

-- Definition of the problem conditions
def shared_albums : Nat := 11
def andrew_total_albums : Nat := 20
def bob_exclusive_albums : Nat := 8

-- Calculate Andrew's exclusive albums
def andrew_exclusive_albums : Nat := andrew_total_albums - shared_albums

-- Question: Prove the total number of albums in either Andrew's or Bob's collection but not both is 17
theorem albums_either_but_not_both : 
  andrew_exclusive_albums + bob_exclusive_albums = 17 := 
by
  sorry

end albums_either_but_not_both_l102_102762


namespace sequence_period_9_l102_102280

def sequence_periodic (x : ℕ → ℤ) : Prop :=
  ∀ n > 1, x (n + 1) = |x n| - x (n - 1)

theorem sequence_period_9 (x : ℕ → ℤ) :
  sequence_periodic x → ∃ p, p = 9 ∧ ∀ n, x (n + p) = x n :=
by
  sorry

end sequence_period_9_l102_102280


namespace two_roots_iff_a_greater_than_neg1_l102_102195

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l102_102195


namespace river_width_l102_102893

theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) : depth = 5 → flow_rate_kmph = 2 → volume_per_minute = 5833.333333333333 → 
  (volume_per_minute / ((flow_rate_kmph * 1000 / 60) * depth) = 35) :=
by 
  intros h_depth h_flow_rate h_volume
  sorry

end river_width_l102_102893


namespace pipe_C_draining_rate_l102_102264

noncomputable def pipe_rate := 25

def tank_capacity := 2000
def pipe_A_rate := 200
def pipe_B_rate := 50
def pipe_C_duration_per_cycle := 2
def pipe_A_duration := 1
def pipe_B_duration := 2
def cycle_duration := pipe_A_duration + pipe_B_duration + pipe_C_duration_per_cycle
def total_time := 40
def number_of_cycles := total_time / cycle_duration
def water_filled_per_cycle := (pipe_A_rate * pipe_A_duration) + (pipe_B_rate * pipe_B_duration)
def total_water_filled := number_of_cycles * water_filled_per_cycle
def excess_water := total_water_filled - tank_capacity 
def pipe_C_rate := excess_water / (pipe_C_duration_per_cycle * number_of_cycles)

theorem pipe_C_draining_rate :
  pipe_C_rate = pipe_rate := by
  sorry

end pipe_C_draining_rate_l102_102264


namespace accurate_river_length_l102_102010

-- Define the given conditions
def length_GSA := 402
def length_AWRA := 403
def error_margin := 0.5
def probability_of_error := 0.04

-- State the theorem based on these conditions
theorem accurate_river_length : 
  ∀ Length_GSA Length_AWRA error_margin probability_of_error, 
  Length_GSA = 402 → 
  Length_AWRA = 403 → 
  error_margin = 0.5 → 
  probability_of_error = 0.04 → 
  (this based on independent measurements with above error margins)
  combined_length = 402.5 ∧ combined_probability_of_error = 0.04 :=
by 
  -- Proof to be completed
  sorry

end accurate_river_length_l102_102010


namespace cube_volume_total_four_boxes_l102_102295

theorem cube_volume_total_four_boxes :
  ∀ (length : ℕ), (length = 5) → (4 * (length^3) = 500) :=
begin
  intros length h,
  rw h,
  norm_num,
end

end cube_volume_total_four_boxes_l102_102295


namespace arithmetic_sum_example_l102_102932

def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sum_example (a1 d : ℤ) 
  (S20_eq_340 : S 20 a1 d = 340) :
  a 6 a1 d + a 9 a1 d + a 11 a1 d + a 16 a1 d = 68 :=
by
  sorry

end arithmetic_sum_example_l102_102932


namespace number_of_ways_to_choose_one_book_is_correct_l102_102129

-- Definitions of the given problem conditions
def number_of_chinese_books : Nat := 10
def number_of_english_books : Nat := 7
def number_of_math_books : Nat := 5

-- Theorem stating the proof problem
theorem number_of_ways_to_choose_one_book_is_correct : 
  number_of_chinese_books + number_of_english_books + number_of_math_books = 22 := by
  -- This proof is left as an exercise.
  sorry

end number_of_ways_to_choose_one_book_is_correct_l102_102129


namespace expand_product_l102_102463

theorem expand_product (x : ℝ) : (x + 2) * (x^2 + 3 * x + 4) = x^3 + 5 * x^2 + 10 * x + 8 := 
by
  sorry

end expand_product_l102_102463


namespace f_always_positive_l102_102981

noncomputable def f (x : ℝ) : ℝ := x^8 - x^5 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, 0 < f x := by
  sorry

end f_always_positive_l102_102981


namespace greatest_divisor_less_than_30_l102_102872

theorem greatest_divisor_less_than_30 :
  (∃ d, d ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} ∧ ∀ m, m ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} → m ≤ d) → 
  18 ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} :=
by
  sorry

end greatest_divisor_less_than_30_l102_102872


namespace john_treats_patients_per_year_l102_102090

theorem john_treats_patients_per_year :
  (let 
    patients_first_hospital_per_day := 20,
    patients_second_hospital_per_day := patients_first_hospital_per_day + (20 / 100 * patients_first_hospital_per_day),
    days_per_week := 5,
    weeks_per_year := 50,
    patients_first_hospital_per_week := patients_first_hospital_per_day * days_per_week,
    patients_second_hospital_per_week := patients_second_hospital_per_day * days_per_week,
    total_patients_per_week := patients_first_hospital_per_week + patients_second_hospital_per_week,
    total_patients_per_year := total_patients_per_week * weeks_per_year
  in total_patients_per_year = 11000) :=
by 
  sorry

end john_treats_patients_per_year_l102_102090


namespace determine_number_of_students_l102_102272

theorem determine_number_of_students 
  (n : ℕ) 
  (h1 : n < 600) 
  (h2 : n % 25 = 24) 
  (h3 : n % 19 = 15) : 
  n = 399 :=
by
  -- The proof will be provided here.
  sorry

end determine_number_of_students_l102_102272


namespace rain_probability_at_most_3_days_l102_102287

open BigOperators

def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def rain_probability := (1:ℝ)/5
noncomputable def no_rain_probability := (4:ℝ)/5

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p^k) * ((1-p)^(n-k))

theorem rain_probability_at_most_3_days :
  ∑ k in Finset.range 4, binomial_probability 31 k rain_probability = 0.544 :=
by
  sorry

end rain_probability_at_most_3_days_l102_102287


namespace distance_to_x_axis_l102_102952

theorem distance_to_x_axis (x y : ℝ) (h : (x, y) = (3, -4)) : abs y = 4 := sorry

end distance_to_x_axis_l102_102952


namespace quadratic_equation_real_roots_k_value_l102_102665

theorem quadratic_equation_real_roots_k_value :
  (∀ k : ℕ, (∃ x : ℝ, k * x^2 - 3 * x + 2 = 0) <-> k = 1) :=
by
  sorry
  
end quadratic_equation_real_roots_k_value_l102_102665


namespace ellipse_major_minor_axis_condition_l102_102365

theorem ellipse_major_minor_axis_condition (h1 : ∀ x y : ℝ, x^2 + m * y^2 = 1) 
                                          (h2 : ∀ a b : ℝ, a = 2 * b) :
  m = 1 / 4 :=
sorry

end ellipse_major_minor_axis_condition_l102_102365


namespace number_of_questions_in_test_l102_102863

variable (n : ℕ) -- the total number of questions
variable (correct_answers : ℕ) -- the number of correct answers
variable (sections : ℕ) -- number of sections in the test
variable (questions_per_section : ℕ) -- number of questions per section
variable (percentage_correct : ℚ) -- percentage of correct answers

-- Given conditions
def conditions := 
  correct_answers = 32 ∧ 
  sections = 5 ∧ 
  questions_per_section * sections = n ∧ 
  (70 : ℚ) < percentage_correct ∧ 
  percentage_correct < 77 ∧ 
  percentage_correct * n = 3200

-- The main statement to prove
theorem number_of_questions_in_test : conditions n correct_answers sections questions_per_section percentage_correct → 
  n = 45 :=
by
  sorry

end number_of_questions_in_test_l102_102863


namespace total_sum_spent_l102_102989

theorem total_sum_spent (b gift : ℝ) (friends tanya : ℕ) (extra_payment : ℝ)
  (h1 : friends = 10)
  (h2 : tanya = 1)
  (h3 : extra_payment = 3)
  (h4 : gift = 15)
  (h5 : b = 270)
  : (b + gift) = 285 :=
by {
  -- Given:
  -- friends = 10 (number of dinner friends),
  -- tanya = 1 (Tanya who forgot to pay),
  -- extra_payment = 3 (extra payment by each of the remaining 9 friends),
  -- gift = 15 (cost of the gift),
  -- b = 270 (total bill for the dinner excluding the gift),

  -- We need to prove:
  -- total sum spent by the group is $285, i.e., (b + gift) = 285

  sorry 
}

end total_sum_spent_l102_102989


namespace cyclic_quadrilateral_area_l102_102836

variable (a b c d R : ℝ)
noncomputable def p : ℝ := (a + b + c + d) / 2
noncomputable def Brahmagupta_area : ℝ := Real.sqrt ((p a b c d - a) * (p a b c d - b) * (p a b c d - c) * (p a b c d - d))

theorem cyclic_quadrilateral_area :
  Brahmagupta_area a b c d = Real.sqrt ((a * b + c * d) * (a * d + b * c) * (a * c + b * d)) / (4 * R) := sorry

end cyclic_quadrilateral_area_l102_102836


namespace rational_inequality_solution_l102_102858

theorem rational_inequality_solution {x : ℝ} : (4 / (x + 1) ≤ 1) → (x ∈ Set.Iic (-1) ∪ Set.Ici 3) :=
by 
  sorry

end rational_inequality_solution_l102_102858


namespace calculate_error_percentage_l102_102755

theorem calculate_error_percentage (x : ℝ) (hx : x > 0) (x_eq_9 : x = 9) :
  (abs ((x * (x - 8)) / (8 * x)) * 100) = 12.5 := by
  sorry

end calculate_error_percentage_l102_102755


namespace complement_union_l102_102525

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

theorem complement_union : 
  U = {0, 1, 2, 3, 4} →
  (U \ A = {1, 2}) →
  B = {1, 3} →
  (A ∪ B = {0, 1, 3, 4}) :=
by
  intros hU hA hB
  sorry

end complement_union_l102_102525


namespace percentage_not_speak_french_l102_102157

open Nat

theorem percentage_not_speak_french (students_surveyed : ℕ)
  (speak_french_and_english : ℕ) (speak_only_french : ℕ) :
  students_surveyed = 200 →
  speak_french_and_english = 25 →
  speak_only_french = 65 →
  ((students_surveyed - (speak_french_and_english + speak_only_french)) * 100 / students_surveyed) = 55 :=
by
  intros h1 h2 h3
  sorry

end percentage_not_speak_french_l102_102157


namespace total_flying_days_l102_102958

-- Definitions for the conditions
def days_fly_south_winter := 40
def days_fly_north_summer := 2 * days_fly_south_winter
def days_fly_east_spring := 60

-- Theorem stating the total flying days
theorem total_flying_days : 
  days_fly_south_winter + days_fly_north_summer + days_fly_east_spring = 180 :=
  by {
    -- This is where we would prove the theorem
    sorry
  }

end total_flying_days_l102_102958


namespace coefficient_of_x3_in_expansion_l102_102710

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def expansion_coefficient_x3 : ℤ :=
  let term1 := (-1 : ℤ) ^ 3 * binomial_coefficient 6 3
  let term2 := (1 : ℤ) * binomial_coefficient 6 2
  term1 + term2

theorem coefficient_of_x3_in_expansion :
  expansion_coefficient_x3 = -5 := by
  sorry

end coefficient_of_x3_in_expansion_l102_102710


namespace exactly_two_roots_iff_l102_102185

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l102_102185


namespace remainder_of_83_div_9_l102_102104

theorem remainder_of_83_div_9 : ∃ r : ℕ, 83 = 9 * 9 + r ∧ r = 2 :=
by {
  sorry
}

end remainder_of_83_div_9_l102_102104


namespace lambda_value_l102_102219

-- Definitions provided in the conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (e1 e2 : V) (A B C D : V)
-- Non-collinear vectors e1 and e2
variables (h_non_collinear : ∃ a b : ℝ, a ≠ b ∧ a • e1 + b • e2 ≠ 0)
-- Given vectors AB, BC, CD
variables (AB BC CD : V)
variables (lambda : ℝ)
-- Vector definitions based on given conditions
variables (h1 : AB = 2 • e1 + e2)
variables (h2 : BC = -e1 + 3 • e2)
variables (h3 : CD = lambda • e1 - e2)
-- Collinearity condition of points A, B, D
variables (collinear : ∃ β : ℝ, AB = β • (BC + CD))

-- The proof goal
theorem lambda_value (h1 : AB = 2 • e1 + e2) (h2 : BC = -e1 + 3 • e2) (h3 : CD = lambda • e1 - e2) (collinear : ∃ β : ℝ, AB = β • (BC + CD)) : lambda = 5 := 
sorry

end lambda_value_l102_102219


namespace expression_values_l102_102279

theorem expression_values (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ b)
  (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 1 ∨ (3 * a - b) / (a + 5 * b) = 3 := 
sorry

end expression_values_l102_102279


namespace total_volume_of_four_cubes_l102_102300

theorem total_volume_of_four_cubes (s : ℝ) (h_s : s = 5) : 4 * s^3 = 500 :=
by
  sorry

end total_volume_of_four_cubes_l102_102300


namespace fourth_guard_distance_l102_102442

theorem fourth_guard_distance 
  (length : ℝ) (width : ℝ)
  (total_distance_three_guards: ℝ)
  (P : ℝ := 2 * (length + width)) 
  (total_distance_four_guards : ℝ := P)
  (total_three : total_distance_three_guards = 850)
  (length_value : length = 300)
  (width_value : width = 200) :
  ∃ distance_fourth_guard : ℝ, distance_fourth_guard = 150 :=
by 
  sorry

end fourth_guard_distance_l102_102442


namespace unique_right_triangle_construction_l102_102044

noncomputable def right_triangle_condition (c f : ℝ) : Prop :=
  f < c / 2

theorem unique_right_triangle_construction (c f : ℝ) (h_c : 0 < c) (h_f : 0 < f) :
  right_triangle_condition c f :=
  sorry

end unique_right_triangle_construction_l102_102044


namespace original_number_unique_l102_102304

theorem original_number_unique (x : ℝ) (h_pos : 0 < x) 
  (h_condition : 100 * x = 9 / x) : x = 3 / 10 :=
by
  sorry

end original_number_unique_l102_102304


namespace total_race_distance_l102_102839

theorem total_race_distance :
  let sadie_time := 2
  let sadie_speed := 3
  let ariana_time := 0.5
  let ariana_speed := 6
  let total_time := 4.5
  let sarah_speed := 4
  let sarah_time := total_time - sadie_time - ariana_time
  let sadie_distance := sadie_speed * sadie_time
  let ariana_distance := ariana_speed * ariana_time
  let sarah_distance := sarah_speed * sarah_time
  let total_distance := sadie_distance + ariana_distance + sarah_distance
  total_distance = 17 :=
by
  sorry

end total_race_distance_l102_102839


namespace one_thirds_of_nine_halfs_l102_102649

theorem one_thirds_of_nine_halfs : (9 / 2) / (1 / 3) = 27 / 2 := 
by sorry

end one_thirds_of_nine_halfs_l102_102649


namespace comb_identity_l102_102939

theorem comb_identity (n : Nat) (h : 0 < n) (h_eq : Nat.choose n 2 = Nat.choose (n-1) 2 + Nat.choose (n-1) 3) : n = 5 := by
  sorry

end comb_identity_l102_102939


namespace area_of_triangle_ABC_l102_102327

theorem area_of_triangle_ABC 
  (BD DC : ℕ) 
  (h_ratio : BD / DC = 4 / 3)
  (S_BEC : ℕ) 
  (h_BEC : S_BEC = 105) :
  ∃ (S_ABC : ℕ), S_ABC = 315 := 
sorry

end area_of_triangle_ABC_l102_102327


namespace repeating_decimal_arithmetic_l102_102338

def x : ℚ := 0.234 -- repeating decimal 0.234
def y : ℚ := 0.567 -- repeating decimal 0.567
def z : ℚ := 0.891 -- repeating decimal 0.891

theorem repeating_decimal_arithmetic :
  x - y + z = 186 / 333 := 
sorry

end repeating_decimal_arithmetic_l102_102338


namespace bob_walking_rate_is_12_l102_102105

-- Definitions for the problem
def yolanda_distance := 24
def yolanda_rate := 3
def bob_distance_when_met := 12
def time_yolanda_walked := 2

-- The theorem we need to prove
theorem bob_walking_rate_is_12 : 
  (bob_distance_when_met / (time_yolanda_walked - 1) = 12) :=
by sorry

end bob_walking_rate_is_12_l102_102105


namespace one_thirds_in_nine_halves_l102_102652

theorem one_thirds_in_nine_halves : (9/2) / (1/3) = 27/2 := by
  sorry

end one_thirds_in_nine_halves_l102_102652


namespace investment_time_l102_102891

variable (P : ℝ) (R : ℝ) (SI : ℝ)

theorem investment_time (hP : P = 800) (hR : R = 0.04) (hSI : SI = 160) :
  SI / (P * R) = 5 := by
  sorry

end investment_time_l102_102891


namespace screen_to_body_ratio_increases_l102_102994

theorem screen_to_body_ratio_increases
  (a b m : ℝ)
  (h1 : a > b)
  (h2 : 0 < m)
  (h3 : m < 1) :
  (b + m) / (a + m) > b / a :=
by
  sorry

end screen_to_body_ratio_increases_l102_102994


namespace greatest_common_divisor_of_three_divisors_l102_102565

theorem greatest_common_divisor_of_three_divisors (m : ℕ) (h1 : ∃ x, x ∣ 120 ∧ x ∣ m ∧ x > 0 ∧ (∀ d, d ∣ x → d = 1 ∨ d = 2 ∨ d = 4))
  : gcd 120 m = 4 :=
begin
  sorry,
end

end greatest_common_divisor_of_three_divisors_l102_102565


namespace three_digit_integers_sat_f_n_eq_f_2005_l102_102254

theorem three_digit_integers_sat_f_n_eq_f_2005 
  (f : ℕ → ℕ)
  (h1 : ∀ m n : ℕ, f (m + n) = f (f m + n))
  (h2 : f 6 = 2)
  (h3 : f 6 ≠ f 9)
  (h4 : f 6 ≠ f 12)
  (h5 : f 6 ≠ f 15)
  (h6 : f 9 ≠ f 12)
  (h7 : f 9 ≠ f 15)
  (h8 : f 12 ≠ f 15) :
  ∃! n, 100 ≤ n ∧ n ≤ 999 ∧ f n = f 2005 → n = 225 := 
  sorry

end three_digit_integers_sat_f_n_eq_f_2005_l102_102254


namespace exactly_two_roots_iff_l102_102183

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l102_102183


namespace solve_for_x_l102_102633

theorem solve_for_x (x : ℝ) (h : (x^2 + 4*x - 5)^0 = 1) : x^2 - 5*x + 5 = 1 → x = 4 := 
by
  intro h2
  have : ∀ x, (x^2 + 4*x - 5 = 0) ↔ false := sorry
  exact sorry

end solve_for_x_l102_102633


namespace calculate_p_l102_102806

variable (m n : ℤ) (p : ℤ)

theorem calculate_p (h1 : 3 * m - 2 * n = -2) (h2 : p = 3 * (m + 405) - 2 * (n - 405)) : p = 2023 := 
  sorry

end calculate_p_l102_102806


namespace solution_interval_l102_102604

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end solution_interval_l102_102604


namespace problem_statement_l102_102233

-- Define function f(x) given parameter m
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- Define even function condition
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Define the monotonic decreasing interval condition
def is_monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) :=
 ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≥ f y

theorem problem_statement :
  (∀ x : ℝ, f m x = f m (-x)) → is_monotonically_decreasing (f 0) {x | 0 < x} :=
by 
  sorry

end problem_statement_l102_102233


namespace combination_2586_1_eq_2586_l102_102905

noncomputable def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem combination_2586_1_eq_2586 : combination 2586 1 = 2586 := by
  sorry

end combination_2586_1_eq_2586_l102_102905


namespace students_per_class_l102_102399

theorem students_per_class (total_cupcakes : ℕ) (num_classes : ℕ) (pe_students : ℕ) 
  (h1 : total_cupcakes = 140) (h2 : num_classes = 3) (h3 : pe_students = 50) : 
  (total_cupcakes - pe_students) / num_classes = 30 :=
by
  sorry

end students_per_class_l102_102399


namespace flutes_tried_out_l102_102098

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

end flutes_tried_out_l102_102098


namespace math_expression_evaluation_l102_102450

theorem math_expression_evaluation :
  36 + (120 / 15) + (15 * 19) - 150 - (450 / 9) = 129 :=
by
  sorry

end math_expression_evaluation_l102_102450


namespace count_integers_satisfying_conditions_l102_102781

theorem count_integers_satisfying_conditions :
  (∃ (s : Finset ℤ), s.card = 3 ∧
  ∀ x : ℤ, x ∈ s ↔ (-5 ≤ x ∧ x ≤ -3)) :=
by {
  sorry
}

end count_integers_satisfying_conditions_l102_102781


namespace greatest_power_of_two_factor_l102_102568

theorem greatest_power_of_two_factor (n m : ℕ) (h1 : n = 12) (h2 : m = 8) :
  ∃ k, k = 1209 ∧ 2^k ∣ n^603 - m^402 :=
by
  sorry

end greatest_power_of_two_factor_l102_102568


namespace arithmetic_expression_eval_l102_102015

theorem arithmetic_expression_eval :
  ((26.3 * 12 * 20) / 3) + 125 = 2229 :=
sorry

end arithmetic_expression_eval_l102_102015


namespace sum_of_specific_coefficients_of_polynomial_l102_102227

theorem sum_of_specific_coefficients_of_polynomial :
  let P := (1 + 2 * x)^5 -- define the polynomial (1 + 2x)^5
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℕ), 
    P = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5
    ∧ a_0 + a_1 + a_3 + a_5 = 123 :=
begin
  sorry
end

end sum_of_specific_coefficients_of_polynomial_l102_102227


namespace find_int_less_than_neg3_l102_102445

theorem find_int_less_than_neg3 : 
  ∃ x ∈ ({-4, -2, 0, 3} : Set Int), x < -3 ∧ x = -4 := 
by
  -- formal proof goes here
  sorry

end find_int_less_than_neg3_l102_102445


namespace frequency_total_students_l102_102420

noncomputable def total_students (known : ℕ) (freq : ℝ) : ℝ :=
known / freq

theorem frequency_total_students (known : ℕ) (freq : ℝ) (h1 : known = 40) (h2 : freq = 0.8) :
  total_students known freq = 50 :=
by
  rw [total_students, h1, h2]
  norm_num

end frequency_total_students_l102_102420


namespace probability_of_both_red_is_one_sixth_l102_102744

noncomputable def probability_both_red (red blue green : ℕ) (balls_picked : ℕ) : ℚ :=
  if balls_picked = 2 ∧ red = 4 ∧ blue = 3 ∧ green = 2 then (4 / 9) * (3 / 8) else 0

theorem probability_of_both_red_is_one_sixth :
  probability_both_red 4 3 2 2 = 1 / 6 :=
by
  unfold probability_both_red
  split_ifs
  · sorry
  · contradiction

end probability_of_both_red_is_one_sixth_l102_102744


namespace statement_1_statement_2_statement_3_statement_4_main_proof_l102_102409

noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem statement_1 : ¬ ∃ x, x = 2 ∧ ∀ y, f y ≤ f x := sorry

theorem statement_2 : ∃! x, f x - x = 0 := sorry

theorem statement_3 : ¬ ∃ k > 0, ∀ x > 0, f x > k * x := sorry

theorem statement_4 : ∀ x1 x2 : ℝ, x2 > x1 ∧ f x1 = f x2 → x1 + x2 > 4 := sorry

theorem main_proof : (¬ ∃ x, x = 2 ∧ ∀ y, f y ≤ f x) ∧ 
                     (∃! x, f x - x = 0) ∧ 
                     (¬ ∃ k > 0, ∀ x > 0, f x > k * x) ∧ 
                     (∀ x1 x2 : ℝ, x2 > x1 ∧ f x1 = f x2 → x1 + x2 > 4) := 
by
  apply And.intro
  · exact statement_1
  · apply And.intro
    · exact statement_2
    · apply And.intro
      · exact statement_3
      · exact statement_4

end statement_1_statement_2_statement_3_statement_4_main_proof_l102_102409


namespace product_of_consecutive_even_numbers_divisible_by_8_l102_102532

theorem product_of_consecutive_even_numbers_divisible_by_8 (n : ℤ) : 
  8 ∣ (2 * n) * (2 * n + 2) :=
by sorry

end product_of_consecutive_even_numbers_divisible_by_8_l102_102532


namespace additional_charge_per_segment_l102_102964

theorem additional_charge_per_segment :
  ∀ (initial_fee total_charge distance : ℝ), 
    initial_fee = 2.35 →
    total_charge = 5.5 →
    distance = 3.6 →
    (total_charge - initial_fee) / (distance / (2 / 5)) = 0.35 :=
by
  intros initial_fee total_charge distance h_initial_fee h_total_charge h_distance
  sorry

end additional_charge_per_segment_l102_102964


namespace largest_result_l102_102852

theorem largest_result (a b c : ℕ) (h1 : a = 0 / 100) (h2 : b = 0 * 100) (h3 : c = 100 - 0) : 
  c > a ∧ c > b :=
by
  sorry

end largest_result_l102_102852


namespace x_coord_sum_l102_102405

noncomputable def sum_x_coordinates (x : ℕ) : Prop :=
  (0 ≤ x ∧ x < 20) ∧ (∃ y, y ≡ 7 * x + 3 [MOD 20] ∧ y ≡ 13 * x + 18 [MOD 20])

theorem x_coord_sum : ∃ (x : ℕ), sum_x_coordinates x ∧ x = 15 := by 
  sorry

end x_coord_sum_l102_102405


namespace tricia_age_is_5_l102_102728

theorem tricia_age_is_5 :
  (∀ Amilia Yorick Eugene Khloe Rupert Vincent : ℕ,
    Tricia = 5 ∧
    (3 * Tricia = Amilia) ∧
    (4 * Amilia = Yorick) ∧
    (2 * Eugene = Yorick) ∧
    (Eugene / 3 = Khloe) ∧
    (Khloe + 10 = Rupert) ∧
    (Vincent = 22)) → 
  Tricia = 5 :=
by
  sorry

end tricia_age_is_5_l102_102728


namespace simultaneous_equations_solution_exists_l102_102060

theorem simultaneous_equations_solution_exists (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 6 ∧ y = (2 * m - 3) * x + 9) ↔ m ≠ 3 :=
by
  sorry

end simultaneous_equations_solution_exists_l102_102060


namespace equation_two_roots_iff_l102_102202

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l102_102202


namespace exactly_two_roots_iff_l102_102186

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l102_102186


namespace division_of_fractions_l102_102729

theorem division_of_fractions :
  (5 : ℚ) / 6 / ((2 : ℚ) / 3) = (5 : ℚ) / 4 :=
by
  sorry

end division_of_fractions_l102_102729


namespace product_of_numbers_is_178_5_l102_102996

variables (a b c d : ℚ)

def sum_eq_36 := a + b + c + d = 36
def first_num_cond := a = 3 * (b + c + d)
def second_num_cond := b = 5 * c
def fourth_num_cond := d = (1 / 2) * c

theorem product_of_numbers_is_178_5 (h1 : sum_eq_36 a b c d)
  (h2 : first_num_cond a b c d) (h3 : second_num_cond b c) (h4 : fourth_num_cond d c) :
  a * b * c * d = 178.5 :=
by
  sorry

end product_of_numbers_is_178_5_l102_102996


namespace grace_pennies_l102_102749

theorem grace_pennies :
  let dime_value := 10
  let coin_value := 5
  let dimes := 10
  let coins := 10
  dimes * dime_value + coins * coin_value = 150 :=
by
  let dime_value := 10
  let coin_value := 5
  let dimes := 10
  let coins := 10
  sorry

end grace_pennies_l102_102749


namespace find_quadruples_l102_102916

theorem find_quadruples (a b p n : ℕ) (h_prime : Prime p) (h_eq : a^3 + b^3 = p^n) :
  ∃ k : ℕ, (a, b, p, n) = (2^k, 2^k, 2, 3*k + 1) ∨ 
           (a, b, p, n) = (3^k, 2 * 3^k, 3, 3*k + 2) ∨ 
           (a, b, p, n) = (2 * 3^k, 3^k, 3, 3*k + 2) :=
sorry

end find_quadruples_l102_102916


namespace cadence_total_earnings_l102_102449

/-- Cadence's total earnings in both companies. -/
def total_earnings (old_salary_per_month new_salary_per_month : ℕ) (old_company_months new_company_months : ℕ) : ℕ :=
  (old_salary_per_month * old_company_months) + (new_salary_per_month * new_company_months)

theorem cadence_total_earnings :
  let old_salary_per_month := 5000
  let old_company_years := 3
  let months_per_year := 12
  let old_company_months := old_company_years * months_per_year
  let new_salary_per_month := old_salary_per_month + (old_salary_per_month * 20 / 100)
  let new_company_extra_months := 5
  let new_company_months := old_company_months + new_company_extra_months
  total_earnings old_salary_per_month new_salary_per_month old_company_months new_company_months = 426000 := by
sorry

end cadence_total_earnings_l102_102449


namespace rod_mass_equilibrium_l102_102859

variable (g : ℝ) (m1 : ℝ) (l : ℝ) (S : ℝ)

-- Given conditions
axiom m1_value : m1 = 1
axiom l_value  : l = 0.5
axiom S_value  : S = 0.1

-- The goal is to find m2 such that the equilibrium condition holds
theorem rod_mass_equilibrium (m2 : ℝ) :
  (m1 * S = m2 * l) → m2 = 0.2 :=
by
  sorry

end rod_mass_equilibrium_l102_102859


namespace Bill_tossed_objects_l102_102768

theorem Bill_tossed_objects (Ted_sticks Ted_rocks Bill_sticks Bill_rocks : ℕ)
  (h1 : Bill_sticks = Ted_sticks + 6)
  (h2 : Ted_rocks = 2 * Bill_rocks)
  (h3 : Ted_sticks = 10)
  (h4 : Ted_rocks = 10) :
  Bill_sticks + Bill_rocks = 21 :=
by
  sorry

end Bill_tossed_objects_l102_102768


namespace pencils_per_student_l102_102575

theorem pencils_per_student (total_pencils : ℕ) (students : ℕ) (pencils_per_student : ℕ) 
  (h_total : total_pencils = 125) 
  (h_students : students = 25) 
  (h_div : pencils_per_student = total_pencils / students) : 
  pencils_per_student = 5 :=
by
  sorry

end pencils_per_student_l102_102575


namespace three_digit_number_divisible_by_7_l102_102945

theorem three_digit_number_divisible_by_7
  (a b : ℕ)
  (h1 : (a + b) % 7 = 0) :
  (100 * a + 10 * b + a) % 7 = 0 :=
sorry

end three_digit_number_divisible_by_7_l102_102945


namespace binary_to_base4_representation_l102_102871

def binary_to_base4 (n : ℕ) : ℕ :=
  -- Assuming implementation that converts binary number n to its base 4 representation 
  sorry

theorem binary_to_base4_representation :
  binary_to_base4 0b10110110010 = 23122 :=
by sorry

end binary_to_base4_representation_l102_102871


namespace lion_turn_angles_l102_102890

-- Define the radius of the circle
def radius (r : ℝ) := r = 10

-- Define the path length the lion runs in meters
def path_length (d : ℝ) := d = 30000

-- Define the final goal: The sum of all the angles of its turns is at least 2998 radians
theorem lion_turn_angles (r d : ℝ) (α : ℝ) (hr : radius r) (hd : path_length d) (hα : d ≤ 10 * α) : α ≥ 2998 := 
sorry

end lion_turn_angles_l102_102890


namespace toby_change_is_7_l102_102136

def cheeseburger_cost : ℝ := 3.65
def milkshake_cost : ℝ := 2
def coke_cost : ℝ := 1
def large_fries_cost : ℝ := 4
def cookie_cost : ℝ := 0.5
def tax : ℝ := 0.2
def toby_funds : ℝ := 15

def total_food_cost_before_tax : ℝ := 
  2 * cheeseburger_cost + milkshake_cost + coke_cost + large_fries_cost + 3 * cookie_cost

def total_bill_with_tax : ℝ := total_food_cost_before_tax + tax

def each_person_share : ℝ := total_bill_with_tax / 2

def toby_change : ℝ := toby_funds - each_person_share

theorem toby_change_is_7 : toby_change = 7 := by
  sorry

end toby_change_is_7_l102_102136


namespace number_of_questions_in_test_l102_102861

theorem number_of_questions_in_test (x : ℕ) (sections questions_correct : ℕ)
  (h_sections : sections = 5)
  (h_questions_correct : questions_correct = 32)
  (h_percentage : 0.70 < (questions_correct : ℚ) / x ∧ (questions_correct : ℚ) / x < 0.77) 
  (h_multiple_of_sections : x % sections = 0) : 
  x = 45 :=
sorry

end number_of_questions_in_test_l102_102861


namespace similar_triangles_perimeter_l102_102288

theorem similar_triangles_perimeter
  (height_ratio : ℚ)
  (smaller_perimeter larger_perimeter : ℚ)
  (h_ratio : height_ratio = 3 / 5)
  (h_smaller_perimeter : smaller_perimeter = 12)
  : larger_perimeter = 20 :=
by
  sorry

end similar_triangles_perimeter_l102_102288


namespace smallest_n_value_existence_l102_102325

-- Define a three-digit positive integer n such that the conditions hold
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def problem_conditions (n : ℕ) : Prop :=
  n % 9 = 3 ∧ n % 6 = 3

-- Main statement: There exists a three-digit positive integer n satisfying the conditions and is equal to 111
theorem smallest_n_value_existence : ∃ n : ℕ, is_three_digit n ∧ problem_conditions n ∧ n = 111 :=
by
  sorry

end smallest_n_value_existence_l102_102325


namespace length_of_the_train_l102_102759

noncomputable def length_of_train (s1 s2 : ℝ) (t1 t2 : ℕ) : ℝ :=
  (s1 * t1 + s2 * t2) / 2

theorem length_of_the_train :
  ∀ (s1 s2 : ℝ) (t1 t2 : ℕ), s1 = 25 → t1 = 8 → s2 = 100 / 3 → t2 = 6 → length_of_train s1 s2 t1 t2 = 200 :=
by
  intros s1 s2 t1 t2 hs1 ht1 hs2 ht2
  rw [hs1, ht1, hs2, ht2]
  simp [length_of_train]
  norm_num

end length_of_the_train_l102_102759


namespace average_price_per_person_excluding_gratuity_l102_102312

def total_cost_with_gratuity : ℝ := 207.00
def gratuity_rate : ℝ := 0.15
def number_of_people : ℕ := 15

theorem average_price_per_person_excluding_gratuity :
  (total_cost_with_gratuity / (1 + gratuity_rate) / number_of_people) = 12.00 :=
by
  sorry

end average_price_per_person_excluding_gratuity_l102_102312


namespace cost_per_kg_after_30_l102_102168

theorem cost_per_kg_after_30 (l m : ℝ) 
  (hl : l = 20) 
  (h1 : 30 * l + 3 * m = 663) 
  (h2 : 30 * l + 6 * m = 726) : 
  m = 21 :=
by
  -- Proof will be written here
  sorry

end cost_per_kg_after_30_l102_102168


namespace inequality_problem_l102_102681

variable {a b : ℕ}

theorem inequality_problem (a : ℕ) (b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_neq_1_a : a ≠ 1) (h_neq_1_b : b ≠ 1) :
  ((a^5 - 1:ℚ) / (a^4 - 1)) * ((b^5 - 1) / (b^4 - 1)) > (25 / 64 : ℚ) * (a + 1) * (b + 1) :=
by
  sorry

end inequality_problem_l102_102681


namespace repeating_decimals_expr_as_fraction_l102_102336

-- Define the repeating decimals as fractions
def a : ℚ := 234 / 999
def b : ℚ := 567 / 999
def c : ℚ := 891 / 999

-- Lean 4 statement to prove the equivalence
theorem repeating_decimals_expr_as_fraction : a - b + c = 186 / 333 := by
  sorry

end repeating_decimals_expr_as_fraction_l102_102336


namespace available_floor_space_equals_110_sqft_l102_102745

-- Definitions for the conditions
def tile_side_in_feet : ℝ := 0.5
def width_main_section_tiles : ℕ := 15
def length_main_section_tiles : ℕ := 25
def width_alcove_tiles : ℕ := 10
def depth_alcove_tiles : ℕ := 8
def width_pillar_tiles : ℕ := 3
def length_pillar_tiles : ℕ := 5

-- Conversion of tiles to feet
def width_main_section_feet : ℝ := width_main_section_tiles * tile_side_in_feet
def length_main_section_feet : ℝ := length_main_section_tiles * tile_side_in_feet
def width_alcove_feet : ℝ := width_alcove_tiles * tile_side_in_feet
def depth_alcove_feet : ℝ := depth_alcove_tiles * tile_side_in_feet
def width_pillar_feet : ℝ := width_pillar_tiles * tile_side_in_feet
def length_pillar_feet : ℝ := length_pillar_tiles * tile_side_in_feet

-- Area calculations
def area_main_section : ℝ := width_main_section_feet * length_main_section_feet
def area_alcove : ℝ := width_alcove_feet * depth_alcove_feet
def total_area : ℝ := area_main_section + area_alcove
def area_pillar : ℝ := width_pillar_feet * length_pillar_feet
def available_floor_space : ℝ := total_area - area_pillar

-- Proof statement
theorem available_floor_space_equals_110_sqft 
  (h1 : width_main_section_feet = width_main_section_tiles * tile_side_in_feet)
  (h2 : length_main_section_feet = length_main_section_tiles * tile_side_in_feet)
  (h3 : width_alcove_feet = width_alcove_tiles * tile_side_in_feet)
  (h4 : depth_alcove_feet = depth_alcove_tiles * tile_side_in_feet)
  (h5 : width_pillar_feet = width_pillar_tiles * tile_side_in_feet)
  (h6 : length_pillar_feet = length_pillar_tiles * tile_side_in_feet) 
  (h7 : area_main_section = width_main_section_feet * length_main_section_feet)
  (h8 : area_alcove = width_alcove_feet * depth_alcove_feet)
  (h9 : total_area = area_main_section + area_alcove)
  (h10 : area_pillar = width_pillar_feet * length_pillar_feet)
  (h11 : available_floor_space = total_area - area_pillar) : 
  available_floor_space = 110 := 
by 
  sorry

end available_floor_space_equals_110_sqft_l102_102745


namespace average_age_of_school_l102_102501

theorem average_age_of_school 
  (total_students : ℕ)
  (average_age_boys : ℕ)
  (average_age_girls : ℕ)
  (number_of_girls : ℕ)
  (number_of_boys : ℕ := total_students - number_of_girls)
  (total_age_boys : ℕ := average_age_boys * number_of_boys)
  (total_age_girls : ℕ := average_age_girls * number_of_girls)
  (total_age_students : ℕ := total_age_boys + total_age_girls) :
  total_students = 640 →
  average_age_boys = 12 →
  average_age_girls = 11 →
  number_of_girls = 160 →
  (total_age_students : ℝ) / (total_students : ℝ) = 11.75 :=
by
  intros h1 h2 h3 h4
  sorry

end average_age_of_school_l102_102501


namespace incorrect_scientific_statement_is_D_l102_102460

-- Define the number of colonies screened by Student A and other students
def studentA_colonies := 150
def other_students_colonies := 50

-- Define the descriptions
def descriptionA := "The reason Student A had such results could be due to different soil samples or problems in the experimental operation."
def descriptionB := "Student A's prepared culture medium could be cultured without adding soil as a blank control, to demonstrate whether the culture medium is contaminated."
def descriptionC := "If other students use the same soil as Student A for the experiment and get consistent results with Student A, it can be proven that Student A's operation was without error."
def descriptionD := "Both experimental approaches described in options B and C follow the principle of control in the experiment."

-- The incorrect scientific statement identified
def incorrect_statement := descriptionD

-- The main theorem statement
theorem incorrect_scientific_statement_is_D : incorrect_statement = descriptionD := by
  sorry

end incorrect_scientific_statement_is_D_l102_102460


namespace minimum_value_is_1297_l102_102798

noncomputable def find_minimum_value (a b c n : ℕ) : ℕ :=
  if (a + b ≠ b + c) ∧ (b + c ≠ c + a) ∧ (a + b ≠ c + a) ∧
     ((a + b = n^2 ∧ b + c = (n + 1)^2 ∧ c + a = (n + 2)^2) ∨
      (a + b = (n + 1)^2 ∧ b + c = (n + 2)^2 ∧ c + a = n^2) ∨
      (a + b = (n + 2)^2 ∧ b + c = n^2 ∧ c + a = (n + 1)^2)) then
    a^2 + b^2 + c^2
  else
    0

theorem minimum_value_is_1297 (a b c n : ℕ) :
  a ≠ b → b ≠ c → c ≠ a → (∃ a b c n, (a + b = n^2 ∧ b + c = (n + 1)^2 ∧ c + a = (n + 2)^2) ∨
                                  (a + b = (n + 1)^2 ∧ b + c = (n + 2)^2 ∧ c + a = n^2) ∨
                                  (a + b = (n + 2)^2 ∧ b + c = n^2 ∧ c + a = (n + 1)^2)) →
  (∃ a b c, a^2 + b^2 + c^2 = 1297) :=
by sorry

end minimum_value_is_1297_l102_102798


namespace union_complement_eq_l102_102689

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l102_102689


namespace sitting_break_frequency_l102_102398

theorem sitting_break_frequency (x : ℕ) (h1 : 240 % x = 0) (h2 : 240 / 20 = 12) (h3 : 240 / x + 10 = 12) : x = 120 := 
sorry

end sitting_break_frequency_l102_102398


namespace triangle_inequality_difference_l102_102396

theorem triangle_inequality_difference :
  ∀ (x : ℤ), (x + 8 > 3) → (x + 3 > 8) → (8 + 3 > x) →
  ( 10 - 6 = 4 ) :=
by sorry

end triangle_inequality_difference_l102_102396


namespace coin_toss_min_n_l102_102021

theorem coin_toss_min_n (n : ℕ) :
  (1 : ℝ) - (1 / (2 : ℝ)) ^ n ≥ 15 / 16 → n ≥ 4 :=
by
  sorry

end coin_toss_min_n_l102_102021


namespace distinct_remainders_l102_102093

theorem distinct_remainders (n : ℕ) (hn : 0 < n) : 
  ∀ (i j : ℕ), (i < n) → (j < n) → (2 * i + 1 ≠ 2 * j + 1) → 
  ((2 * i + 1) ^ (2 * i + 1) % 2^n ≠ (2 * j + 1) ^ (2 * j + 1) % 2^n) :=
by
  sorry

end distinct_remainders_l102_102093


namespace equation_two_roots_iff_l102_102205

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l102_102205


namespace negation_of_exists_proposition_l102_102471

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
sorry

end negation_of_exists_proposition_l102_102471


namespace find_angle_B_find_a_plus_c_l102_102497

variable (A B C a b c S : Real)

-- Conditions
axiom h1 : a = (1 / 2) * c + b * Real.cos C
axiom h2 : S = Real.sqrt 3
axiom h3 : b = Real.sqrt 13

-- Questions (Proving the answers from the problem)
theorem find_angle_B (hA : A = Real.pi - (B + C)) : 
  B = Real.pi / 3 := by
  sorry

theorem find_a_plus_c (hac : (1 / 2) * a * c * Real.sin (Real.pi / 3) = Real.sqrt 3) : 
  a + c = 5 := by
  sorry

end find_angle_B_find_a_plus_c_l102_102497


namespace find_k_l102_102790

theorem find_k (a b c k : ℝ) 
  (h : ∀ x : ℝ, 
    (a * x^2 + b * x + c + b * x^2 + a * x - 7 + k * x^2 + c * x + 3) / (x^2 - 2 * x - 5) = (x^2 - 2*x - 5)) :
  k = 2 :=
by
  sorry

end find_k_l102_102790


namespace max_fraction_value_l102_102465

theorem max_fraction_value :
  ∀ (x y : ℝ), (1/4 ≤ x ∧ x ≤ 3/5) ∧ (1/5 ≤ y ∧ y ≤ 1/2) → 
    xy / (x^2 + y^2) ≤ 2/5 :=
by
  sorry

end max_fraction_value_l102_102465


namespace mod_99_equal_sum_pairs_mod_99_l102_102986

theorem mod_99_equal_sum_pairs_mod_99 (N : ℕ) (digits : list ℕ) 
  (hN_digits : digits.map (λ i, i / 10) = digits.zip_with (λ x y, x * 10 + y) digits.tail digits.tail.tail) :
  let sum_pairs := (list.sum (digits.zip_with (λ x y, x * 10 + y) digits.tail digits.tail.tail) + digits.head) in
  N % 99 = sum_pairs % 99 :=
by {
  sorry
}

end mod_99_equal_sum_pairs_mod_99_l102_102986


namespace sum_of_three_squares_l102_102084

theorem sum_of_three_squares (n : ℕ) (h : n = 100) : 
  ∃ (a b c : ℕ), a = 4 ∧ b^2 + c^2 = 84 ∧ a^2 + b^2 + c^2 = 100 ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c ∨ (b = c ∧ a ≠ b)) ∧
  (4^2 + 7^2 + 6^2 = 100 ∧ 4^2 + 8^2 + 5^2 = 100 ∧ 4^2 + 9^2 + 1^2 = 100) ∧
  (4^2 + 6^2 + 7^2 ≠ 100 ∧ 4^2 + 5^2 + 8^2 ≠ 100 ∧ 4^2 + 1^2 + 9^2 ≠ 100 ∧ 
   4^2 + 4^2 + 8^2 ≠ 100 ∨ 4^2 + 8^2 + 4^2 ≠ 100) :=
sorry

end sum_of_three_squares_l102_102084


namespace option_D_correct_option_A_incorrect_option_B_incorrect_option_C_incorrect_l102_102306

-- Define the variables
variables (m : ℤ)

-- State the conditions as hypotheses
theorem option_D_correct (m : ℤ) : 
  (m * (m - 1) = m^2 - m) :=
by {
    -- Proof sketch (not implemented):
    -- Use distributive property to demonstrate that both sides are equal.
    sorry
}

theorem option_A_incorrect (m : ℤ) : 
  ¬ (m^4 + m^3 = m^7) :=
by {
    -- Proof sketch (not implemented):
    -- Demonstrate that exponents can't be added this way when bases are added.
    sorry
}

theorem option_B_incorrect (m : ℤ) : 
  ¬ ((m^4)^3 = m^7) :=
by {
    -- Proof sketch (not implemented):
    -- Show that raising m^4 to the power of 3 results in m^12.
    sorry
}

theorem option_C_incorrect (m : ℤ) : 
  ¬ (2 * m^5 / m^3 = m^2) :=
by {
    -- Proof sketch (not implemented):
    -- Show that dividing results in 2m^2.
    sorry
}

end option_D_correct_option_A_incorrect_option_B_incorrect_option_C_incorrect_l102_102306


namespace volume_of_four_cubes_l102_102297

theorem volume_of_four_cubes (edge_length : ℕ) (num_cubes : ℕ) (h_edge : edge_length = 5) (h_num : num_cubes = 4) :
  num_cubes * (edge_length ^ 3) = 500 :=
by 
  sorry

end volume_of_four_cubes_l102_102297


namespace least_clock_equivalent_l102_102101

theorem least_clock_equivalent (x : ℕ) : 
  x > 3 ∧ x % 12 = (x * x) % 12 → x = 12 := 
by
  sorry

end least_clock_equivalent_l102_102101


namespace smallest_m_n_sum_l102_102850

theorem smallest_m_n_sum (m n : ℕ) (h_m : 1 < m) (h_pos : 0 < n) 
  (interval_length : (m^2 - 1) / (m * n) = 1 / 4033) : 
  m + n = 48421 :=
sorry

end smallest_m_n_sum_l102_102850


namespace greatest_divisor_of_three_consecutive_odds_l102_102403

theorem greatest_divisor_of_three_consecutive_odds (n : ℕ) : 
  ∃ (d : ℕ), (∀ (k : ℕ), k = 2*n + 1 ∨ k = 2*n + 3 ∨ k = 2*n + 5 → d ∣ (2*n + 1) * (2*n + 3) * (2*n + 5)) ∧ d = 3 :=
by
  sorry

end greatest_divisor_of_three_consecutive_odds_l102_102403


namespace problem_l102_102929

variable (a : ℕ → ℝ) -- {a_n} is a sequence
variable (S : ℕ → ℝ) -- S_n represents the sum of the first n terms
variable (d : ℝ) -- non-zero common difference
variable (a1 : ℝ) -- first term of the sequence

-- Define an arithmetic sequence with common difference d and first term a1
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (a1 : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of the first n terms of an arithmetic sequence
def sum_of_arithmetic_sequence (S a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem problem 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (d : ℝ) 
  (a1 : ℝ) 
  (h_non_zero : d ≠ 0)
  (h_sequence : is_arithmetic_sequence a d a1)
  (h_sum : sum_of_arithmetic_sequence S a)
  (h_S5_eq_S6 : S 5 = S 6) :
  S 11 = 0 := 
sorry

end problem_l102_102929


namespace friend_spending_l102_102422

-- Definitions based on conditions
def total_spent (you friend : ℝ) : Prop := you + friend = 15
def friend_spent (you friend : ℝ) : Prop := friend = you + 1

-- Prove that the friend's spending equals $8 given the conditions
theorem friend_spending (you friend : ℝ) (htotal : total_spent you friend) (hfriend : friend_spent you friend) : friend = 8 :=
by
  sorry

end friend_spending_l102_102422


namespace smallest_period_find_a_l102_102797

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + a

theorem smallest_period (a : ℝ) : 
  ∃ T > 0, ∀ x, f x a = f (x + T) a ∧ (∀ T' > 0, (∀ x, f x a = f (x + T') a) → T ≤ T') :=
by
  sorry

theorem find_a :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) ∧ (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = 4) ∧ a = 1 :=
by
  sorry

end smallest_period_find_a_l102_102797


namespace fish_worth_rice_l102_102238

variables (f l r : ℝ)

-- Conditions based on the problem statement
def fish_for_bread : Prop := 3 * f = 2 * l
def bread_for_rice : Prop := l = 4 * r

-- Statement to be proven
theorem fish_worth_rice (h₁ : fish_for_bread f l) (h₂ : bread_for_rice l r) : f = (8 / 3) * r :=
  sorry

end fish_worth_rice_l102_102238


namespace symmetric_line_equation_l102_102851

theorem symmetric_line_equation :
  (∃ line : ℝ → ℝ, ∀ x y, x + 2 * y - 3 = 0 → line 1 = 1 ∧ (∃ b, line 0 = b → x - 2 * y + 1 = 0)) :=
sorry

end symmetric_line_equation_l102_102851


namespace shelves_needed_l102_102579

theorem shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) (remaining_books : ℕ) (shelves : ℕ) :
  total_books = 34 →
  books_taken = 7 →
  books_per_shelf = 3 →
  remaining_books = total_books - books_taken →
  shelves = remaining_books / books_per_shelf →
  shelves = 9 :=
by
  intros h_total h_taken h_per_shelf h_remaining h_shelves
  rw [h_total, h_taken, h_per_shelf] at *
  sorry

end shelves_needed_l102_102579


namespace concert_tickets_full_price_revenue_l102_102029

theorem concert_tickets_full_price_revenue :
  ∃ (f p d : ℕ), f + d = 200 ∧ f * p + d * (p / 3) = 2688 ∧ f * p = 2128 :=
by
  -- We need to find the solution steps are correct to establish the existence
  sorry

end concert_tickets_full_price_revenue_l102_102029


namespace remainder_of_99_times_101_divided_by_9_is_0_l102_102874

theorem remainder_of_99_times_101_divided_by_9_is_0 : (99 * 101) % 9 = 0 :=
by
  sorry

end remainder_of_99_times_101_divided_by_9_is_0_l102_102874


namespace domain_of_function_l102_102357

theorem domain_of_function :
  { x : ℝ | -2 ≤ x ∧ x < 4 } = { x : ℝ | (x + 2 ≥ 0) ∧ (4 - x > 0) } :=
by
  sorry

end domain_of_function_l102_102357


namespace cole_cost_l102_102452

def length_of_sides := 15
def length_of_back := 30
def cost_per_foot_side := 4
def cost_per_foot_back := 5
def cole_installation_fee := 50

def neighbor_behind_contribution := (length_of_back * cost_per_foot_back) / 2
def neighbor_left_contribution := (length_of_sides * cost_per_foot_side) / 3

def total_cost := 
  2 * length_of_sides * cost_per_foot_side + 
  length_of_back * cost_per_foot_back

def cole_contribution := 
  total_cost - neighbor_behind_contribution - neighbor_left_contribution + cole_installation_fee

theorem cole_cost (h : cole_contribution = 225) : cole_contribution = 225 := by
  sorry

end cole_cost_l102_102452


namespace number_of_students_l102_102576

theorem number_of_students (N : ℕ) (T : ℕ)
  (h1 : T = 80 * N)
  (h2 : (T - 160) / (N - 8) = 90) :
  N = 56 :=
sorry

end number_of_students_l102_102576


namespace h_inverse_left_h_inverse_right_l102_102097

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1
noncomputable def h (x : ℝ) : ℝ := f (g x)
noncomputable def h_inv (y : ℝ) : ℝ := 1 + (Real.sqrt (3 * y + 12)) / 4 -- Correct answer

-- Theorem statements to prove the inverse relationship
theorem h_inverse_left (x : ℝ) : h (h_inv x) = x :=
by
  sorry -- Proof of the left inverse

theorem h_inverse_right (y : ℝ) : h_inv (h y) = y :=
by
  sorry -- Proof of the right inverse

end h_inverse_left_h_inverse_right_l102_102097


namespace arithmetic_seq_sum_l102_102067

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h2 : a 2 + a 5 + a 8 = 15) : a 3 + a 7 = 10 :=
sorry

end arithmetic_seq_sum_l102_102067


namespace sum_abc_l102_102968

variable {a b c : ℝ}
variables (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0)
variables (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (c + a))

theorem sum_abc (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (c + a))
   (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) :
   a + b + c = 1128 / 35 := 
sorry

end sum_abc_l102_102968


namespace find_k_l102_102997

variable {a_n : ℕ → ℤ}    -- Define the arithmetic sequence as a function from natural numbers to integers
variable {a1 d : ℤ}        -- a1 is the first term, d is the common difference

-- Conditions
axiom seq_def : ∀ n, a_n n = a1 + (n - 1) * d
axiom sum_condition : 9 * a1 + 36 * d = 4 * a1 + 6 * d
axiom ak_a4_zero (k : ℕ): a_n 4 + a_n k = 0

-- Problem Statement to prove
theorem find_k : ∃ k : ℕ, a_n 4 + a_n k = 0 → k = 10 :=
by
  use 10
  intro h
  -- proof omitted
  sorry

end find_k_l102_102997


namespace repeating_decimal_arithmetic_l102_102337

def x : ℚ := 0.234 -- repeating decimal 0.234
def y : ℚ := 0.567 -- repeating decimal 0.567
def z : ℚ := 0.891 -- repeating decimal 0.891

theorem repeating_decimal_arithmetic :
  x - y + z = 186 / 333 := 
sorry

end repeating_decimal_arithmetic_l102_102337


namespace geometric_sequence_a7_l102_102500

noncomputable def a (n : ℕ) : ℝ := sorry -- Definition of the sequence

theorem geometric_sequence_a7 :
  a 3 = 1 → a 11 = 25 → a 7 = 5 := 
by
  intros h3 h11
  sorry

end geometric_sequence_a7_l102_102500


namespace birdhouse_distance_l102_102558

theorem birdhouse_distance (car_distance : ℕ) (lawnchair_distance : ℕ) (birdhouse_distance : ℕ) 
  (h1 : car_distance = 200) 
  (h2 : lawnchair_distance = 2 * car_distance) 
  (h3 : birdhouse_distance = 3 * lawnchair_distance) : 
  birdhouse_distance = 1200 :=
by
  sorry

end birdhouse_distance_l102_102558


namespace solution_interval_l102_102602

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end solution_interval_l102_102602


namespace problem_l102_102659

theorem problem (x y : ℝ) (h : (3 * x - y + 5)^2 + |2 * x - y + 3| = 0) : x + y = -3 := 
by
  sorry

end problem_l102_102659


namespace countEquilateralTriangles_l102_102711

-- Define the problem conditions
def numSmallTriangles := 18  -- The number of small equilateral triangles
def includesMarkedTriangle: Prop := True  -- All counted triangles include the marked triangle "**"

-- Define the main question as a proposition
def totalEquilateralTriangles : Prop :=
  (numSmallTriangles = 18 ∧ includesMarkedTriangle) → (1 + 4 + 1 = 6)

-- The theorem stating the number of equilateral triangles containing the marked triangle
theorem countEquilateralTriangles : totalEquilateralTriangles :=
  by
    sorry

end countEquilateralTriangles_l102_102711


namespace how_many_more_cups_of_sugar_l102_102259

def required_sugar : ℕ := 11
def required_flour : ℕ := 9
def added_flour : ℕ := 12
def added_sugar : ℕ := 10

theorem how_many_more_cups_of_sugar :
  required_sugar - added_sugar = 1 :=
by
  sorry

end how_many_more_cups_of_sugar_l102_102259


namespace ln_sqrt2_lt_sqrt2_div2_ln_sin_cos_sum_l102_102313

theorem ln_sqrt2_lt_sqrt2_div2 : Real.log (Real.sqrt 2) < Real.sqrt 2 / 2 :=
sorry

theorem ln_sin_cos_sum : 2 * Real.log (Real.sin (1/8) + Real.cos (1/8)) < 1 / 4 :=
sorry

end ln_sqrt2_lt_sqrt2_div2_ln_sin_cos_sum_l102_102313


namespace complex_number_solution_l102_102485

open Complex

theorem complex_number_solution (z : ℂ) (h : (z - 2 * I) * (2 - I) = 5) : z = 2 + 3 * I :=
  sorry

end complex_number_solution_l102_102485


namespace gardening_project_total_cost_l102_102333

theorem gardening_project_total_cost :
  let 
    num_rose_bushes := 20
    cost_per_rose_bush := 150
    gardener_hourly_rate := 30
    gardener_hours_per_day := 5
    gardener_days := 4
    soil_volume := 100
    soil_cost_per_cubic_foot := 5

    cost_of_rose_bushes := num_rose_bushes * cost_per_rose_bush
    gardener_total_hours := gardener_hours_per_day * gardener_days
    cost_of_gardener := gardener_hourly_rate * gardener_total_hours
    cost_of_soil := soil_volume * soil_cost_per_cubic_foot

    total_cost := cost_of_rose_bushes + cost_of_gardener + cost_of_soil
  in
    total_cost = 4100 := 
  by
    intros
    simp [num_rose_bushes, cost_per_rose_bush, gardener_hourly_rate, gardener_hours_per_day, gardener_days, soil_volume, soil_cost_per_cubic_foot]
    rw [mul_comm num_rose_bushes, mul_comm gardener_total_hours, mul_comm soil_volume]
    sorry -- place for proof steps

end gardening_project_total_cost_l102_102333


namespace translate_point_left_l102_102726

def initial_point : ℝ × ℝ := (-2, -1)
def translation_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ := (p.1 - units, p.2)

theorem translate_point_left :
  translation_left initial_point 2 = (-4, -1) :=
by
  -- By definition and calculation
  -- Let p = initial_point
  -- x' = p.1 - 2,
  -- y' = p.2
  -- translation_left (-2, -1) 2 = (-4, -1)
  sorry

end translate_point_left_l102_102726


namespace set_problems_l102_102404

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}

theorem set_problems :
  (A ∩ B = ({4} : Set ℤ)) ∧
  (A ∪ B = ({1, 2, 4, 5, 6, 7, 8, 9, 10} : Set ℤ)) ∧
  (U \ (A ∪ B) = ({3} : Set ℤ)) ∧
  ((U \ A) ∩ (U \ B) = ({3} : Set ℤ)) :=
by
  sorry

end set_problems_l102_102404


namespace avg_writing_speed_l102_102033

theorem avg_writing_speed 
  (words1 hours1 words2 hours2 : ℕ)
  (h_words1 : words1 = 30000)
  (h_hours1 : hours1 = 60)
  (h_words2 : words2 = 50000)
  (h_hours2 : hours2 = 100) :
  (words1 + words2) / (hours1 + hours2) = 500 :=
by {
  sorry
}

end avg_writing_speed_l102_102033


namespace complement_U_A_eq_two_l102_102740

open Set

universe u

def U : Set ℕ := { x | x ≥ 2 }
def A : Set ℕ := { x | x^2 ≥ 5 }
def comp_U_A : Set ℕ := U \ A

theorem complement_U_A_eq_two : comp_U_A = {2} :=
by 
  sorry

end complement_U_A_eq_two_l102_102740


namespace num_coprime_to_15_l102_102908

theorem num_coprime_to_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.toFinset.card = 8 :=
by
  sorry

end num_coprime_to_15_l102_102908


namespace maggie_fraction_caught_l102_102527

theorem maggie_fraction_caught :
  let total_goldfish := 100
  let allowed_to_take_home := total_goldfish / 2
  let remaining_goldfish_to_catch := 20
  let goldfish_caught := allowed_to_take_home - remaining_goldfish_to_catch
  (goldfish_caught / allowed_to_take_home : ℚ) = 3 / 5 :=
by
  sorry

end maggie_fraction_caught_l102_102527


namespace replace_asterisks_l102_102421

theorem replace_asterisks (x : ℝ) (h : (x / 21) * (x / 84) = 1) : x = 42 :=
sorry

end replace_asterisks_l102_102421


namespace one_thirds_in_nine_halves_l102_102656

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 13 := by
  sorry

end one_thirds_in_nine_halves_l102_102656


namespace find_m_l102_102553

-- Mathematical definitions from the given conditions
def condition1 (m : ℝ) : Prop := m^2 - 2 * m - 2 = 1
def condition2 (m : ℝ) : Prop := m + 1/2 * m^2 > 0

-- The proof problem summary
theorem find_m (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = 3 :=
by
  sorry

end find_m_l102_102553


namespace negation_of_exists_statement_l102_102856

theorem negation_of_exists_statement :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by
  sorry

end negation_of_exists_statement_l102_102856


namespace perimeter_ratio_l102_102321

theorem perimeter_ratio (w l : ℕ) (hfold : w = 8) (lfold : l = 6) 
(folded_w : w / 2 = 4) (folded_l : l / 2 = 3) 
(hcut : w / 4 = 1) (lcut : l / 2 = 3) 
(perimeter_small : ℕ) (perimeter_large : ℕ)
(hperim_small : perimeter_small = 2 * (3 + 4)) 
(hperim_large : perimeter_large = 2 * (6 + 4)) :
(perimeter_small : ℕ) / (perimeter_large : ℕ) = 7 / 10 := sorry

end perimeter_ratio_l102_102321


namespace polar_distance_l102_102956

/-
Problem:
In the polar coordinate system, it is known that A(2, π / 6), B(4, 5π / 6). Then, the distance between points A and B is 2√7.

Conditions:
- Point A in polar coordinates: A(2, π / 6)
- Point B in polar coordinates: B(4, 5π / 6)
-/

/-- The distance between two points in the polar coordinate system A(2, π / 6) and B(4, 5π / 6) is 2√7. -/
theorem polar_distance :
  let A_ρ := 2
  let A_θ := π / 6
  let B_ρ := 4
  let B_θ := 5 * π / 6
  let A_x := A_ρ * Real.cos A_θ
  let A_y := A_ρ * Real.sin A_θ
  let B_x := B_ρ * Real.cos B_θ
  let B_y := B_ρ * Real.sin B_θ
  let distance := Real.sqrt ((B_x - A_x)^2 + (B_y - A_y)^2)
  distance = 2 * Real.sqrt 7 := by
  sorry

end polar_distance_l102_102956


namespace total_trees_after_planting_l102_102865

-- Definitions based on conditions
def initial_trees : ℕ := 34
def trees_to_plant : ℕ := 49

-- Statement to prove the total number of trees after planting
theorem total_trees_after_planting : initial_trees + trees_to_plant = 83 := 
by 
  sorry

end total_trees_after_planting_l102_102865


namespace find_rs_l102_102266

noncomputable def r : ℝ := sorry
noncomputable def s : ℝ := sorry
def cond1 := r > 0 ∧ s > 0
def cond2 := r^2 + s^2 = 1
def cond3 := r^4 + s^4 = (3 : ℝ) / 4

theorem find_rs (h1 : cond1) (h2 : cond2) (h3 : cond3) : r * s = Real.sqrt 2 / 4 :=
by sorry

end find_rs_l102_102266


namespace coins_in_pockets_l102_102503

theorem coins_in_pockets : (Nat.choose (5 + 3 - 1) (3 - 1)) = 21 := by
  sorry

end coins_in_pockets_l102_102503


namespace sufficient_m_value_l102_102686

theorem sufficient_m_value (m : ℕ) : 
  ((8 = m ∨ 9 = m) → 
  (m^2 + m^4 + m^6 + m^8 ≥ 6^3 + 6^5 + 6^7 + 6^9)) := 
by 
  sorry

end sufficient_m_value_l102_102686


namespace assignment_statement_increases_l102_102116

theorem assignment_statement_increases (N : ℕ) : (N + 1 = N + 1) :=
sorry

end assignment_statement_increases_l102_102116


namespace sub_from_square_l102_102232

theorem sub_from_square (n : ℕ) (h : n = 17) : (n * n - n) = 272 :=
by 
  -- Proof goes here
  sorry

end sub_from_square_l102_102232


namespace complement_M_l102_102974

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x | (x - 1) * (x - 4) = 0}

theorem complement_M :
  (U \ M) = {2, 3} := by
  sorry

end complement_M_l102_102974


namespace toby_change_l102_102135

theorem toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2
  let coke_cost := 1
  let large_fries_cost := 4
  let cookie_cost := 0.5
  let num_cookies := 3
  let tax := 0.2
  let initial_amount := 15
  let total_cost := 2 * cheeseburger_cost + milkshake_cost + coke_cost + large_fries_cost + num_cookies * cookie_cost + tax
  let cost_per_person := total_cost / 2
  let toby_change := initial_amount - cost_per_person
  in toby_change = 7 :=
sorry

end toby_change_l102_102135


namespace measure_of_angle_A_l102_102814

-- Define the conditions as assumptions
variable (B : Real) (angle1 angle2 A : Real)
-- Angle B is 120 degrees
axiom h1 : B = 120
-- One of the angles formed by the dividing line is 50 degrees
axiom h2 : angle1 = 50
-- Angles formed sum up to 180 degrees as they are supplementary
axiom h3 : angle2 = 180 - angle1
-- Vertical angles are equal
axiom h4 : A = angle2

theorem measure_of_angle_A (B angle1 angle2 A : Real) 
    (h1 : B = 120) (h2 : angle1 = 50) (h3 : angle2 = 180 - angle1) (h4 : A = angle2) : A = 130 := 
by
    sorry

end measure_of_angle_A_l102_102814


namespace average_salary_l102_102951

theorem average_salary (T_salary : ℕ) (R_salary : ℕ) (total_salary : ℕ) (T_count : ℕ) (R_count : ℕ) (total_count : ℕ) :
    T_salary = 12000 * T_count →
    R_salary = 6000 * R_count →
    total_salary = T_salary + R_salary →
    T_count = 6 →
    R_count = total_count - T_count →
    total_count = 18 →
    (total_salary / total_count) = 8000 :=
by
  intros
  sorry

end average_salary_l102_102951


namespace quartic_to_quadratic_l102_102934

-- Defining the statement of the problem
theorem quartic_to_quadratic (a b c x : ℝ) (y : ℝ) :
  a * x^4 + b * x^3 + c * x^2 + b * x + a = 0 →
  y = x + 1 / x →
  ∃ y1 y2, (a * y^2 + b * y + (c - 2 * a) = 0) ∧
           (x^2 - y1 * x + 1 = 0 ∨ x^2 - y2 * x + 1 = 0) :=
by
  sorry

end quartic_to_quadratic_l102_102934


namespace Shara_savings_l102_102591

theorem Shara_savings (P : ℝ) (d : ℝ) (paid : ℝ):
  d = 0.08 → paid = 184 → P = 200 → (P * (1 - d) = paid) → (P - paid = 16) :=
by
  intros hd hpaid hP heq
  -- It follows from the conditions given
  sorry

end Shara_savings_l102_102591


namespace Anne_height_l102_102446

-- Define the conditions
variables (S : ℝ)   -- Height of Anne's sister
variables (A : ℝ)   -- Height of Anne
variables (B : ℝ)   -- Height of Bella

-- Define the relations according to the problem's conditions
def condition1 (S : ℝ) := A = 2 * S
def condition2 (S : ℝ) := B = 3 * A
def condition3 (S : ℝ) := B - S = 200

-- Theorem statement to prove Anne's height
theorem Anne_height (S : ℝ) (A : ℝ) (B : ℝ)
(h1 : A = 2 * S) (h2 : B = 3 * A) (h3 : B - S = 200) : A = 80 :=
by sorry

end Anne_height_l102_102446


namespace solve_for_x_l102_102569

theorem solve_for_x : (∃ x : ℝ, ((10 - 2 * x) ^ 2 = 4 * x ^ 2 + 16) ∧ x = 2.1) :=
by
  sorry

end solve_for_x_l102_102569


namespace profit_without_discount_l102_102424

theorem profit_without_discount
  (CP SP_with_discount : ℝ) 
  (H1 : CP = 100) -- Assume cost price is 100
  (H2 : SP_with_discount = CP + 0.216 * CP) -- Selling price with discount
  (H3 : SP_with_discount = 0.95 * SP_without_discount) -- SP with discount is 95% of SP without discount
  : (SP_without_discount - CP) / CP * 100 = 28 := 
by
  -- proof goes here
  sorry

end profit_without_discount_l102_102424


namespace volume_ratio_of_cubes_l102_102623

theorem volume_ratio_of_cubes 
  (P_A P_B : ℕ) 
  (h_A : P_A = 40) 
  (h_B : P_B = 64) : 
  (∃ s_A s_B V_A V_B, 
    s_A = P_A / 4 ∧ 
    s_B = P_B / 4 ∧ 
    V_A = s_A^3 ∧ 
    V_B = s_B^3 ∧ 
    (V_A : ℚ) / V_B = 125 / 512) := 
by
  sorry

end volume_ratio_of_cubes_l102_102623


namespace simplify_fraction_l102_102844

theorem simplify_fraction :
  ∀ (x y : ℝ), x = 1 → y = 1 → (x^3 + y^3) / (x + y) = 1 :=
by
  intros x y hx hy
  rw [hx, hy]
  simp
  sorry

end simplify_fraction_l102_102844


namespace find_angle_A_l102_102947

theorem find_angle_A (A B C : ℝ) (a b c : ℝ) :
  (a * Real.sin A = b * Real.sin B + (c - b) * Real.sin C)
  → (A = π / 3) :=
sorry

end find_angle_A_l102_102947


namespace max_checkers_on_board_l102_102701

-- Define the size of the board.
def board_size : ℕ := 8

-- Define the max number of checkers per row/column.
def max_checkers_per_line : ℕ := 3

-- Define the conditions of the board.
structure BoardConfiguration :=
  (rows : Fin board_size → Fin (max_checkers_per_line + 1))
  (columns : Fin board_size → Fin (max_checkers_per_line + 1))
  (valid : ∀ (i : Fin board_size), rows i ≤ max_checkers_per_line ∧ columns i ≤ max_checkers_per_line)

-- Define the function to calculate the total number of checkers.
def total_checkers (config : BoardConfiguration) : ℕ :=
  Finset.univ.sum (λ i => config.rows i + config.columns i)

-- The theorem which states that the maximum number of checkers is 30.
theorem max_checkers_on_board : ∃ (config : BoardConfiguration), total_checkers config = 30 :=
  sorry

end max_checkers_on_board_l102_102701


namespace tan_alpha_value_l102_102634

open Real

theorem tan_alpha_value (α : ℝ) (h1 : sin α + cos α = -1 / 2) (h2 : 0 < α ∧ α < π) : tan α = -1 / 3 :=
sorry

end tan_alpha_value_l102_102634


namespace total_patients_in_a_year_l102_102091

-- Define conditions from the problem
def patients_per_day_first : ℕ := 20
def percent_increase_second : ℕ := 20
def working_days_per_week : ℕ := 5
def working_weeks_per_year : ℕ := 50

-- Lean statement for the problem
theorem total_patients_in_a_year (patients_per_day_first : ℕ) (percent_increase_second : ℕ) (working_days_per_week : ℕ) (working_weeks_per_year : ℕ) :
  (patients_per_day_first + ((patients_per_day_first * percent_increase_second) / 100)) * working_days_per_week * working_weeks_per_year = 11000 :=
by
  sorry

end total_patients_in_a_year_l102_102091


namespace num_girls_l102_102669

theorem num_girls (boys girls : ℕ) (h1 : girls = boys + 228) (h2 : boys = 469) : girls = 697 :=
sorry

end num_girls_l102_102669


namespace angle_bisector_segment_conditional_equality_l102_102478

theorem angle_bisector_segment_conditional_equality
  (a1 b1 a2 b2 : ℝ)
  (h1 : ∃ (P : ℝ), ∃ (e1 e2 : ℝ → ℝ), (e1 P = a1 ∧ e2 P = b1) ∧ (e1 P = a2 ∧ e2 P = b2)) :
  (1 / a1 + 1 / b1 = 1 / a2 + 1 / b2) :=
by 
  sorry

end angle_bisector_segment_conditional_equality_l102_102478


namespace range_of_solutions_l102_102605

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end range_of_solutions_l102_102605


namespace negation_exists_implies_forall_l102_102552

theorem negation_exists_implies_forall : 
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) :=
by
  sorry

end negation_exists_implies_forall_l102_102552


namespace real_solutions_eq59_l102_102468

theorem real_solutions_eq59 :
  (∃ (x: ℝ), -50 ≤ x ∧ x ≤ 50 ∧ (x / 50) = sin x) ∧
  (∃! (S: ℕ), S = 59) :=
sorry

end real_solutions_eq59_l102_102468


namespace EricBenJackMoneySum_l102_102050

noncomputable def EricBenJackTotal (E B J : ℕ) :=
  (E + B + J : ℕ)

theorem EricBenJackMoneySum :
  ∀ (E B J : ℕ), (E = B - 10) → (B = J - 9) → (J = 26) → (EricBenJackTotal E B J) = 50 :=
by
  intros E B J
  intro hE hB hJ
  rw [hJ] at hB
  rw [hB] at hE
  sorry

end EricBenJackMoneySum_l102_102050


namespace one_angle_not_greater_than_60_l102_102307

theorem one_angle_not_greater_than_60 (A B C : ℝ) (h : A + B + C = 180) : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 := 
sorry

end one_angle_not_greater_than_60_l102_102307


namespace hotdogs_sold_l102_102750

-- Definitions of initial and remaining hotdogs
def initial : ℕ := 99
def remaining : ℕ := 97

-- The statement that needs to be proven
theorem hotdogs_sold : initial - remaining = 2 :=
by
  sorry

end hotdogs_sold_l102_102750


namespace cube_volume_total_four_boxes_l102_102296

theorem cube_volume_total_four_boxes :
  ∀ (length : ℕ), (length = 5) → (4 * (length^3) = 500) :=
begin
  intros length h,
  rw h,
  norm_num,
end

end cube_volume_total_four_boxes_l102_102296


namespace percentage_l_75_m_l102_102811

theorem percentage_l_75_m
  (j k l m : ℝ)
  (x : ℝ)
  (h1 : 1.25 * j = 0.25 * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : (x / 100) * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 175 :=
by
  sorry

end percentage_l_75_m_l102_102811


namespace no_such_number_exists_l102_102791

theorem no_such_number_exists :
  ¬ ∃ n : ℕ, 529 < n ∧ n < 538 ∧ 16 ∣ n :=
by sorry

end no_such_number_exists_l102_102791


namespace min_value_of_b_minus_2c_plus_1_over_a_l102_102071

theorem min_value_of_b_minus_2c_plus_1_over_a
  (a b c : ℝ)
  (h₁ : (a ≠ 0))
  (h₂ : ∀ x, -1 < x ∧ x < 3 → ax^2 + bx + c < 0) :
  b - 2 * c + (1 / a) = 4 :=
sorry

end min_value_of_b_minus_2c_plus_1_over_a_l102_102071


namespace man_swim_distance_downstream_l102_102433

noncomputable def DistanceDownstream (Vm : ℝ) (Vupstream : ℝ) (time : ℝ) : ℝ :=
  let Vs := Vm - Vupstream
  let Vdownstream := Vm + Vs
  Vdownstream * time

theorem man_swim_distance_downstream :
  let Vm : ℝ := 3  -- speed of man in still water in km/h
  let time : ℝ := 6 -- time taken in hours
  let d_upstream : ℝ := 12 -- distance swum upstream in km
  let Vupstream : ℝ := d_upstream / time
  DistanceDownstream Vm Vupstream time = 24 := sorry

end man_swim_distance_downstream_l102_102433


namespace largest_integer_divisible_example_1748_largest_n_1748_l102_102730

theorem largest_integer_divisible (n : ℕ) (h : (n + 12) ∣ (n^3 + 160)) : n ≤ 1748 :=
by
  sorry

theorem example_1748 : 1748^3 + 160 = 1760 * 3045738 :=
by
  sorry

theorem largest_n_1748 (n : ℕ) (h : 1748 ≤ n) : (n + 12) ∣ (n^3 + 160) :=
by
  sorry

end largest_integer_divisible_example_1748_largest_n_1748_l102_102730


namespace tournament_games_l102_102673

theorem tournament_games (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 5) : 
  (n * (n - 1) / 2) * k = 2175 := by
  sorry

end tournament_games_l102_102673


namespace number_of_diagonals_octagon_heptagon_diff_l102_102495

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem number_of_diagonals_octagon_heptagon_diff :
  let A := number_of_diagonals 8
  let B := number_of_diagonals 7
  A - B = 6 :=
by
  sorry

end number_of_diagonals_octagon_heptagon_diff_l102_102495


namespace union_of_sets_l102_102801

def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 5, 7}
def union_result : Set ℕ := {1, 3, 5, 7}

theorem union_of_sets : A ∪ B = union_result := by
  sorry

end union_of_sets_l102_102801


namespace candy_problem_l102_102036

theorem candy_problem
  (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999)
  (h3 : n + 7 ≡ 0 [MOD 9])
  (h4 : n - 9 ≡ 0 [MOD 6]) :
  n = 101 :=
sorry

end candy_problem_l102_102036


namespace base_7_is_good_number_l102_102684

def is_good_number (m: ℕ) : Prop :=
  ∃ (p: ℕ) (n: ℕ), Prime p ∧ n ≥ 2 ∧ m = p^n

theorem base_7_is_good_number : 
  ∀ b: ℕ, (is_good_number (b^2 - (2 * b + 3))) → b = 7 :=
by
  intro b h
  sorry

end base_7_is_good_number_l102_102684


namespace EricBenJackMoneySum_l102_102051

noncomputable def EricBenJackTotal (E B J : ℕ) :=
  (E + B + J : ℕ)

theorem EricBenJackMoneySum :
  ∀ (E B J : ℕ), (E = B - 10) → (B = J - 9) → (J = 26) → (EricBenJackTotal E B J) = 50 :=
by
  intros E B J
  intro hE hB hJ
  rw [hJ] at hB
  rw [hB] at hE
  sorry

end EricBenJackMoneySum_l102_102051


namespace find_a_10_l102_102124

-- Definitions and conditions from the problem
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

variable (a : ℕ → ℕ)

-- Conditions given
axiom a_3 : a 3 = 3
axiom S_3 : S a 3 = 6
axiom arithmetic_seq : is_arithmetic_sequence a

-- Proof problem statement
theorem find_a_10 : a 10 = 10 := 
sorry

end find_a_10_l102_102124


namespace cannot_tile_remaining_with_dominoes_l102_102406

def can_tile_remaining_board (pieces : List (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), ∃ (piece : ℕ × ℕ), piece ∈ pieces ∧ piece.1 = i ∧ piece.2 = j

theorem cannot_tile_remaining_with_dominoes : 
  ∃ (pieces : List (ℕ × ℕ)), (∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 10) ∧ (1 ≤ j ∧ j ≤ 10) → ∃ (piece : ℕ × ℕ), piece ∈ pieces ∧ piece.1 = i ∧ piece.2 = j) ∧ ¬ can_tile_remaining_board pieces :=
sorry

end cannot_tile_remaining_with_dominoes_l102_102406


namespace equation_two_roots_iff_l102_102204

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l102_102204


namespace maximum_value_of_d_l102_102972

theorem maximum_value_of_d (a b c d : ℝ) 
  (h₁ : a + b + c + d = 10)
  (h₂ : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end maximum_value_of_d_l102_102972


namespace overall_average_score_l102_102830

theorem overall_average_score
  (mean_morning mean_evening : ℕ)
  (ratio_morning_evening : ℚ) 
  (h1 : mean_morning = 90)
  (h2 : mean_evening = 80)
  (h3 : ratio_morning_evening = 4 / 5) : 
  ∃ overall_mean : ℚ, overall_mean = 84 :=
by
  sorry

end overall_average_score_l102_102830


namespace shaded_areas_sum_l102_102447

theorem shaded_areas_sum (triangle_area : ℕ) (parts : ℕ)
  (h1 : triangle_area = 18)
  (h2 : parts = 9) :
  3 * (triangle_area / parts) = 6 :=
by
  sorry

end shaded_areas_sum_l102_102447


namespace fraction_of_70cm_ropes_l102_102847

theorem fraction_of_70cm_ropes (R : ℕ) (avg_all : ℚ) (avg_70 : ℚ) (avg_85 : ℚ) (total_len : R * avg_all = 480) 
  (total_ropes : R = 6) : 
  ∃ f : ℚ, f = 1 / 3 ∧ f * R * avg_70 + (R - f * R) * avg_85 = R * avg_all :=
by
  sorry

end fraction_of_70cm_ropes_l102_102847


namespace area_inequality_l102_102316

open Real

variables (AB CD AD BC S : ℝ) (alpha beta : ℝ)
variables (α_pos : 0 < α ∧ α < π) (β_pos : 0 < β ∧ β < π)
variables (S_pos : 0 < S) (H1 : ConvexQuadrilateral AB CD AD BC S)

theorem area_inequality :
  AB * CD * sin α + AD * BC * sin β ≤ 2 * S ∧ 2 * S ≤ AB * CD + AD * BC :=
sorry

end area_inequality_l102_102316


namespace tomas_first_month_distance_l102_102562

theorem tomas_first_month_distance 
  (distance_n_5 : ℝ := 26.3)
  (double_distance_each_month : ∀ (n : ℕ), n ≥ 1 → (distance_n : ℝ) = distance_n_5 / (2 ^ (5 - n)))
  : distance_n_5 / (2 ^ (5 - 1)) = 1.64375 :=
by
  sorry

end tomas_first_month_distance_l102_102562


namespace gcd_47_power5_1_l102_102341
-- Import the necessary Lean library

-- Mathematically equivalent proof problem in Lean 4
theorem gcd_47_power5_1 (a b : ℕ) (h1 : a = 47^5 + 1) (h2 : b = 47^5 + 47^3 + 1) :
  Nat.gcd a b = 1 :=
by
  sorry

end gcd_47_power5_1_l102_102341


namespace distance_from_origin_l102_102218

open Real

theorem distance_from_origin (x y : ℝ) (h_parabola : y ^ 2 = 4 * x) (h_focus : sqrt ((x - 1) ^ 2 + y ^ 2) = 4) : dist (x, y) (0, 0) = sqrt 21 :=
by
  sorry

end distance_from_origin_l102_102218


namespace minimum_value_of_f_l102_102057

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 5)

theorem minimum_value_of_f : ∃ (x : ℝ), x > 5 ∧ f x = 20 :=
by
  use 10
  sorry

end minimum_value_of_f_l102_102057


namespace problem_1_problem_2_problem_3_l102_102594

-- Definition and proof state for problem 1
theorem problem_1 (a b m n : ℕ) (h₀ : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n := by
  sorry

-- Definition and proof state for problem 2
theorem problem_2 (a m n : ℕ) (h₀ : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = 13 ∨ a = 7 := by
  sorry

-- Definition and proof state for problem 3
theorem problem_3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 := by
  sorry

end problem_1_problem_2_problem_3_l102_102594


namespace smallest_n_value_existence_l102_102326

-- Define a three-digit positive integer n such that the conditions hold
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def problem_conditions (n : ℕ) : Prop :=
  n % 9 = 3 ∧ n % 6 = 3

-- Main statement: There exists a three-digit positive integer n satisfying the conditions and is equal to 111
theorem smallest_n_value_existence : ∃ n : ℕ, is_three_digit n ∧ problem_conditions n ∧ n = 111 :=
by
  sorry

end smallest_n_value_existence_l102_102326


namespace fraction_irreducible_l102_102108

theorem fraction_irreducible (n : ℕ) (hn : 0 < n) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by sorry

end fraction_irreducible_l102_102108


namespace range_of_solutions_l102_102614

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end range_of_solutions_l102_102614


namespace sum_of_coefficients_l102_102213

theorem sum_of_coefficients :
  let p := -3 * (Polynomial.C (-6) + 4 * Polynomial.X^3 - 2 * Polynomial.X^5 + Polynomial.X^8)
              + 5 * (3 * Polynomial.X^2 + Polynomial.X^4)
              - 4 * (Polynomial.C 5 - Polynomial.X^6)
  in p.eval 1 = 45 :=
by
  let p := -3 * (Polynomial.C (-6) + 4 * Polynomial.X^3 - 2 * Polynomial.X^5 + Polynomial.X^8)
              + 5 * (3 * Polynomial.X^2 + Polynomial.X^4)
              - 4 * (Polynomial.C 5 - Polynomial.X^6)
  show p.eval 1 = 45
  sorry

end sum_of_coefficients_l102_102213


namespace max_value_m_n_squared_sum_l102_102373

theorem max_value_m_n_squared_sum (m n : ℤ) (h1 : 1 ≤ m ∧ m ≤ 1981) (h2 : 1 ≤ n ∧ n ≤ 1981) (h3 : (n^2 - m * n - m^2)^2 = 1) :
  m^2 + n^2 ≤ 3524578 :=
sorry

end max_value_m_n_squared_sum_l102_102373


namespace valid_k_for_triangle_l102_102181

theorem valid_k_for_triangle (k : ℕ) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ b + c > a ∧ c + a > b)) → k ≥ 6 :=
by
  sorry

end valid_k_for_triangle_l102_102181


namespace smallest_natural_number_k_l102_102459

theorem smallest_natural_number_k :
  ∃ k : ℕ, k = 4 ∧ ∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ 1 ≤ n → a^(k) * (1 - a)^(n) < 1 / (n + 1)^3 :=
by
  sorry

end smallest_natural_number_k_l102_102459


namespace least_number_when_increased_by_6_is_divisible_l102_102311

theorem least_number_when_increased_by_6_is_divisible :
  ∃ n : ℕ, 
    (n + 6) % 24 = 0 ∧ 
    (n + 6) % 32 = 0 ∧ 
    (n + 6) % 36 = 0 ∧ 
    (n + 6) % 54 = 0 ∧ 
    n = 858 :=
by
  sorry

end least_number_when_increased_by_6_is_divisible_l102_102311


namespace solution_set_abs_inequality_l102_102554

theorem solution_set_abs_inequality (x : ℝ) :
  (|2 - x| ≥ 1) ↔ (x ≤ 1 ∨ x ≥ 3) :=
by
  sorry

end solution_set_abs_inequality_l102_102554


namespace value_of_a_plus_b_l102_102722

-- Define the given nested fraction expression
def nested_expr := 1 + 1 / (1 + 1 / (1 + 1))

-- Define the simplified form of the expression
def simplified_form : ℚ := 13 / 8

-- The greatest common divisor condition
def gcd_condition : ℕ := Nat.gcd 13 8

-- The ultimate theorem to prove
theorem value_of_a_plus_b : 
  nested_expr = simplified_form ∧ gcd_condition = 1 → 13 + 8 = 21 := 
by 
  sorry

end value_of_a_plus_b_l102_102722


namespace apples_in_each_box_l102_102418

theorem apples_in_each_box (x : ℕ) :
  (5 * x - (60 * 5)) = (2 * x) -> x = 100 :=
by
  sorry

end apples_in_each_box_l102_102418


namespace basket_can_hold_40_fruits_l102_102584

-- Let us define the number of oranges as 10
def oranges : ℕ := 10

-- There are 3 times as many apples as oranges
def apples : ℕ := 3 * oranges

-- The total number of fruits in the basket
def total_fruits : ℕ := oranges + apples

theorem basket_can_hold_40_fruits (h₁ : oranges = 10) (h₂ : apples = 3 * oranges) : total_fruits = 40 :=
by
  -- We assume the conditions and derive the conclusion
  sorry

end basket_can_hold_40_fruits_l102_102584


namespace four_times_sum_of_cubes_gt_cube_sum_l102_102077

theorem four_times_sum_of_cubes_gt_cube_sum
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 :=
by
  sorry

end four_times_sum_of_cubes_gt_cube_sum_l102_102077


namespace ways_to_draw_balls_eq_total_ways_l102_102888

noncomputable def ways_to_draw_balls (n : Nat) :=
  if h : n = 15 then (15 * 14 * 13 * 12) else 0

noncomputable def valid_combinations : Nat := sorry

noncomputable def total_ways_to_draw : Nat :=
  valid_combinations * 24

theorem ways_to_draw_balls_eq_total_ways :
  ways_to_draw_balls 15 = total_ways_to_draw :=
sorry

end ways_to_draw_balls_eq_total_ways_l102_102888


namespace caleb_caught_trouts_l102_102339

theorem caleb_caught_trouts (C : ℕ) (h1 : 3 * C = C + 4) : C = 2 :=
by {
  sorry
}

end caleb_caught_trouts_l102_102339


namespace security_deposit_percentage_l102_102092

theorem security_deposit_percentage
    (daily_rate : ℝ) (pet_fee : ℝ) (service_fee_rate : ℝ) (days : ℝ) (security_deposit : ℝ)
    (total_cost : ℝ) (expected_percentage : ℝ) :
    daily_rate = 125.0 →
    pet_fee = 100.0 →
    service_fee_rate = 0.20 →
    days = 14 →
    security_deposit = 1110 →
    total_cost = daily_rate * days + pet_fee + (daily_rate * days + pet_fee) * service_fee_rate →
    expected_percentage = (security_deposit / total_cost) * 100 →
    expected_percentage = 50 :=
by
  intros
  sorry

end security_deposit_percentage_l102_102092


namespace product_modulo_7_l102_102776

theorem product_modulo_7 : 
  (2007 % 7 = 4) ∧ (2008 % 7 = 5) ∧ (2009 % 7 = 6) ∧ (2010 % 7 = 0) →
  (2007 * 2008 * 2009 * 2010) % 7 = 0 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end product_modulo_7_l102_102776


namespace power_increase_fourfold_l102_102896

theorem power_increase_fourfold 
    (F v : ℝ)
    (k : ℝ)
    (R : ℝ := k * v)
    (P_initial : ℝ := F * v)
    (v' : ℝ := 2 * v)
    (F' : ℝ := 2 * F)
    (R' : ℝ := k * v')
    (P_final : ℝ := F' * v') :
    P_final = 4 * P_initial := 
by
  sorry

end power_increase_fourfold_l102_102896


namespace express_y_in_terms_of_x_l102_102216

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 4) : y = 2 * x - 4 :=
by
  sorry

end express_y_in_terms_of_x_l102_102216


namespace quadratic_inequality_l102_102247

theorem quadratic_inequality (a x1 x2 : ℝ) (h_eq : x1 ^ 2 - a * x1 + a = 0) (h_eq' : x2 ^ 2 - a * x2 + a = 0) :
  x1^2 + x2^2 ≥ 2 * (x1 + x2) :=
sorry

end quadratic_inequality_l102_102247


namespace park_area_l102_102436

theorem park_area (w : ℝ) (h1 : 2 * (w + 3 * w) = 72) : w * (3 * w) = 243 :=
by
  sorry

end park_area_l102_102436


namespace integer_roots_of_poly_l102_102159

-- Define the polynomial
def poly (x : ℤ) (b1 b2 : ℤ) : ℤ :=
  x^3 + b2 * x ^ 2 + b1 * x + 18

-- The list of possible integer roots
def possible_integer_roots := [-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18]

-- Statement of the theorem
theorem integer_roots_of_poly (b1 b2 : ℤ) :
  ∀ x : ℤ, poly x b1 b2 = 0 → x ∈ possible_integer_roots :=
sorry

end integer_roots_of_poly_l102_102159


namespace find_num_candies_bought_l102_102823

-- Conditions
def cost_per_candy := 80
def sell_price_per_candy := 100
def num_sold := 48
def profit := 800

-- Question equivalence
theorem find_num_candies_bought (x : ℕ) 
  (hc : cost_per_candy = 80)
  (hs : sell_price_per_candy = 100)
  (hn : num_sold = 48)
  (hp : profit = 800) :
  48 * 100 - 80 * x = 800 → x = 50 :=
  by
  sorry

end find_num_candies_bought_l102_102823


namespace find_weight_per_square_inch_l102_102618

-- Define the TV dimensions and other given data
def bill_tv_width : ℕ := 48
def bill_tv_height : ℕ := 100
def bob_tv_width : ℕ := 70
def bob_tv_height : ℕ := 60
def weight_difference_pounds : ℕ := 150
def ounces_per_pound : ℕ := 16

-- Compute areas
def bill_tv_area := bill_tv_width * bill_tv_height
def bob_tv_area := bob_tv_width * bob_tv_height

-- Assume weight per square inch
def weight_per_square_inch : ℕ := 4

-- Total weight computation given in ounces
def bill_tv_weight := bill_tv_area * weight_per_square_inch
def bob_tv_weight := bob_tv_area * weight_per_square_inch
def weight_difference_ounces := weight_difference_pounds * ounces_per_pound

-- The theorem to prove
theorem find_weight_per_square_inch : 
  bill_tv_weight - bob_tv_weight = weight_difference_ounces → weight_per_square_inch = 4 :=
by
  intros
  /- Proof by computation -/
  sorry

end find_weight_per_square_inch_l102_102618


namespace right_triangle_perimeter_l102_102438

/-- A right triangle has an area of 150 square units,
and one leg with a length of 30 units. Prove that the
perimeter of the triangle is 40 + 10 * sqrt 10 units. -/
theorem right_triangle_perimeter :
  ∃ (x c : ℝ), (1 / 2) * 30 * x = 150 ∧ c^2 = 30^2 + x^2 ∧ 30 + x + c = 40 + 10 * sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l102_102438


namespace roots_sum_product_l102_102122

theorem roots_sum_product (p q : ℝ) (h_sum : p / 3 = 8) (h_prod : q / 3 = 12) : p + q = 60 := 
by 
  sorry

end roots_sum_product_l102_102122


namespace inequality_proof_l102_102139

theorem inequality_proof 
  (x1 x2 y1 y2 z1 z2 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : 0 < x2)
  (hxy1 : x1 * y1 > z1 ^ 2)
  (hxy2 : x2 * y2 > z2 ^ 2) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) ≤
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) :=
sorry

end inequality_proof_l102_102139


namespace elvins_fixed_charge_l102_102786

theorem elvins_fixed_charge (F C : ℝ) 
  (h1 : F + C = 40) 
  (h2 : F + 2 * C = 76) : F = 4 := 
by 
  sorry

end elvins_fixed_charge_l102_102786


namespace trig_sum_roots_l102_102810

theorem trig_sum_roots {θ a : Real} (hroots : ∀ x, x^2 - a * x + a = 0 → x = Real.sin θ ∨ x = Real.cos θ) :
  Real.cos (θ - 3 * Real.pi / 2) + Real.sin (3 * Real.pi / 2 + θ) = Real.sqrt 2 - 1 :=
by
  sorry

end trig_sum_roots_l102_102810


namespace problem_xyz_inequality_l102_102832

theorem problem_xyz_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h_eq : x^2 + y^2 + z^2 + x * y * z = 4) :
  x * y * z ≤ x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ x * y * z + 2 :=
by 
  sorry

end problem_xyz_inequality_l102_102832


namespace one_thirds_of_nine_halfs_l102_102650

theorem one_thirds_of_nine_halfs : (9 / 2) / (1 / 3) = 27 / 2 := 
by sorry

end one_thirds_of_nine_halfs_l102_102650


namespace simplify_trig_identity_l102_102845

theorem simplify_trig_identity (α : ℝ) :
  (Real.cos (Real.pi / 3 + α) + Real.sin (Real.pi / 6 + α)) = Real.cos α :=
by
  sorry

end simplify_trig_identity_l102_102845


namespace problem1_problem2_l102_102074

-- Define the conditions: f is an odd and decreasing function on [-1, 1]
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_decreasing : ∀ x y, x ≤ y → f y ≤ f x)

-- The domain of interest is [-1, 1]
variable (x1 x2 : ℝ)
variable (h_x1 : x1 ∈ Set.Icc (-1 : ℝ) 1)
variable (h_x2 : x2 ∈ Set.Icc (-1 : ℝ) 1)

-- Proof Problem 1
theorem problem1 : (f x1 + f x2) * (x1 + x2) ≤ 0 := by
  sorry

-- Assume condition for Problem 2
variable (a : ℝ)
variable (h_ineq : f (1 - a) + f (1 - a ^ 2) < 0)
variable (h_dom : ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → x ∈ Set.Icc (-1 : ℝ) 1)

-- Proof Problem 2
theorem problem2 : 0 < a ∧ a < 1 := by
  sorry

end problem1_problem2_l102_102074


namespace two_points_same_color_at_distance_one_l102_102118

theorem two_points_same_color_at_distance_one (color : ℝ × ℝ → ℕ) (h : ∀p : ℝ × ℝ, color p < 3) :
  ∃ (p q : ℝ × ℝ), dist p q = 1 ∧ color p = color q :=
sorry

end two_points_same_color_at_distance_one_l102_102118


namespace find_divisor_l102_102880

-- Defining the conditions
def dividend : ℕ := 181
def quotient : ℕ := 9
def remainder : ℕ := 1

-- The statement to prove
theorem find_divisor : ∃ (d : ℕ), dividend = (d * quotient) + remainder ∧ d = 20 := by
  sorry

end find_divisor_l102_102880


namespace range_of_solutions_l102_102607

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end range_of_solutions_l102_102607


namespace intersection_A_B_l102_102808

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | abs (x - 2) ≥ 1}
def answer : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_A_B :
  A ∩ B = answer :=
sorry

end intersection_A_B_l102_102808


namespace find_line_equation_l102_102793

theorem find_line_equation : 
  ∃ c : ℝ, (∀ x y : ℝ, 2*x + 4*y + c = 0 ↔ x + 2*y - 8 = 0) ∧ (2*2 + 4*3 + c = 0) :=
sorry

end find_line_equation_l102_102793


namespace lance_read_yesterday_l102_102401

-- Definitions based on conditions
def total_pages : ℕ := 100
def pages_tomorrow : ℕ := 35
def pages_yesterday (Y : ℕ) : ℕ := Y
def pages_today (Y : ℕ) : ℕ := Y - 5

-- The statement that we need to prove
theorem lance_read_yesterday (Y : ℕ) (h : pages_yesterday Y + pages_today Y + pages_tomorrow = total_pages) : Y = 35 :=
by sorry

end lance_read_yesterday_l102_102401


namespace tangent_range_of_a_l102_102072

theorem tangent_range_of_a 
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + a * x + 2 * y + a^2 = 0)
  (A : ℝ × ℝ) 
  (A_eq : A = (1, 2)) :
  -2 * Real.sqrt 3 / 3 < a ∧ a < 2 * Real.sqrt 3 / 3 :=
by
  sorry

end tangent_range_of_a_l102_102072


namespace mult_mod_7_zero_l102_102774

theorem mult_mod_7_zero :
  (2007 ≡ 5 [MOD 7]) →
  (2008 ≡ 6 [MOD 7]) →
  (2009 ≡ 0 [MOD 7]) →
  (2010 ≡ 1 [MOD 7]) →
  (2007 * 2008 * 2009 * 2010 ≡ 0 [MOD 7]) :=
by
  intros h1 h2 h3 h4
  sorry

end mult_mod_7_zero_l102_102774


namespace abc_order_l102_102475

open Real

noncomputable def a : ℝ := exp 0.11
noncomputable def b : ℝ := (1.1)^(1.1)
def c : ℝ := 1.11

theorem abc_order :
  a > b ∧ b > c :=
by
  -- Proof steps will be filled here
  sorry

end abc_order_l102_102475


namespace total_shells_correct_l102_102738

def morning_shells : ℕ := 292
def afternoon_shells : ℕ := 324

theorem total_shells_correct : morning_shells + afternoon_shells = 616 := by
  sorry

end total_shells_correct_l102_102738


namespace sphere_triangle_distance_l102_102283

theorem sphere_triangle_distance
  (P X Y Z : Type)
  (radius : ℝ)
  (h1 : radius = 15)
  (dist_XY : ℝ)
  (h2 : dist_XY = 6)
  (dist_YZ : ℝ)
  (h3 : dist_YZ = 8)
  (dist_ZX : ℝ)
  (h4 : dist_ZX = 10)
  (distance_from_P_to_triangle : ℝ)
  (h5 : distance_from_P_to_triangle = 10 * Real.sqrt 2) :
  let a := 10
  let b := 2
  let c := 1
  let result := a + b + c
  result = 13 :=
by
  sorry

end sphere_triangle_distance_l102_102283


namespace translation_2_units_left_l102_102282

-- Define the initial parabola
def parabola1 (x : ℝ) : ℝ := x^2 + 1

-- Define the translated parabola
def parabola2 (x : ℝ) : ℝ := x^2 + 4 * x + 5

-- State that parabola2 is obtained by translating parabola1
-- And prove that this translation is 2 units to the left
theorem translation_2_units_left :
  ∀ x : ℝ, parabola2 x = parabola1 (x + 2) := 
by
  sorry

end translation_2_units_left_l102_102282


namespace part1_part2_part3_l102_102535

-- Part 1
theorem part1 :
  ∀ x : ℝ, (4 * x - 3 = 1) → (x = 1) ↔ 
    (¬(x - 3 > 3 * x - 1) ∧ (4 * (x - 1) ≤ 2) ∧ (x + 2 > 0 ∧ 3 * x - 3 ≤ 1)) :=
by sorry

-- Part 2
theorem part2 :
  ∀ (m n q : ℝ), (m + 2 * n = 6) → (2 * m + n = 3 * q) → (m + n > 1) → q > -1 :=
by sorry

-- Part 3
theorem part3 :
  ∀ (k m n : ℝ), (k < 3) → (∃ x : ℝ, (3 * (x - 1) = k) ∧ (4 * x + n < x + 2 * m)) → 
    (m + n ≥ 0) → (∃! n : ℝ, ∀ x : ℝ, (2 ≤ m ∧ m < 5 / 2)) :=
by sorry

end part1_part2_part3_l102_102535


namespace box_width_is_target_width_l102_102757

-- Defining the conditions
def cube_volume : ℝ := 27
def box_length : ℝ := 8
def box_height : ℝ := 12
def max_cubes : ℕ := 24

-- Defining the target width we want to prove
def target_width : ℝ := 6.75

-- The proof statement
theorem box_width_is_target_width :
  ∃ w : ℝ,
  (∀ v : ℝ, (v = max_cubes * cube_volume) →
   ∀ l : ℝ, (l = box_length) →
   ∀ h : ℝ, (h = box_height) →
   v = l * w * h) →
   w = target_width :=
by
  sorry

end box_width_is_target_width_l102_102757


namespace sum_complex_l102_102784

-- Define the given complex numbers
def z1 : ℂ := ⟨2, 5⟩
def z2 : ℂ := ⟨3, -7⟩

-- State the theorem to prove the sum
theorem sum_complex : z1 + z2 = ⟨5, -2⟩ :=
by
  sorry

end sum_complex_l102_102784


namespace repeating_decimals_expr_as_fraction_l102_102335

-- Define the repeating decimals as fractions
def a : ℚ := 234 / 999
def b : ℚ := 567 / 999
def c : ℚ := 891 / 999

-- Lean 4 statement to prove the equivalence
theorem repeating_decimals_expr_as_fraction : a - b + c = 186 / 333 := by
  sorry

end repeating_decimals_expr_as_fraction_l102_102335


namespace mike_average_points_per_game_l102_102260

theorem mike_average_points_per_game (total_points games_played points_per_game : ℕ) 
  (h1 : games_played = 6) 
  (h2 : total_points = 24) 
  (h3 : total_points = games_played * points_per_game) : 
  points_per_game = 4 :=
by
  rw [h1, h2] at h3  -- Substitute conditions h1 and h2 into the equation
  sorry  -- the proof goes here

end mike_average_points_per_game_l102_102260


namespace combined_river_length_estimate_l102_102013

def river_length_GSA := 402 
def river_error_GSA := 0.5 
def river_prob_error_GSA := 0.04 

def river_length_AWRA := 403 
def river_error_AWRA := 0.5 
def river_prob_error_AWRA := 0.04 

/-- 
Given the measurements from GSA and AWRA, 
the combined estimate of the river's length, Rio-Coralio, is 402.5 km,
and the probability of error for this combined estimate is 0.04.
-/
theorem combined_river_length_estimate :
  ∃ l : ℝ, l = 402.5 ∧ ∀ p : ℝ, (p = 0.04) :=
sorry

end combined_river_length_estimate_l102_102013


namespace jack_weight_52_l102_102821

theorem jack_weight_52 (Sam Jack : ℕ) (h1 : Sam + Jack = 96) (h2 : Jack = Sam + 8) : Jack = 52 := 
by
  sorry

end jack_weight_52_l102_102821


namespace find_m_l102_102251

theorem find_m (m x1 x2 : ℝ) 
  (h1 : x1 * x1 - 2 * (m + 1) * x1 + m^2 + 2 = 0)
  (h2 : x2 * x2 - 2 * (m + 1) * x2 + m^2 + 2 = 0)
  (h3 : (x1 + 1) * (x2 + 1) = 8) : 
  m = 1 :=
sorry

end find_m_l102_102251


namespace min_value_of_expression_l102_102314

noncomputable def min_expression_value (a b c d : ℝ) : ℝ :=
  (a ^ 8) / ((a ^ 2 + b) * (a ^ 2 + c) * (a ^ 2 + d)) +
  (b ^ 8) / ((b ^ 2 + c) * (b ^ 2 + d) * (b ^ 2 + a)) +
  (c ^ 8) / ((c ^ 2 + d) * (c ^ 2 + a) * (c ^ 2 + b)) +
  (d ^ 8) / ((d ^ 2 + a) * (d ^ 2 + b) * (d ^ 2 + c))

theorem min_value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_expression_value a b c d = 1 / 2 :=
by
  -- Proof is omitted.
  sorry

end min_value_of_expression_l102_102314


namespace train_speed_l102_102737

theorem train_speed (L : ℝ) (T : ℝ) (hL : L = 200) (hT : T = 20) :
  L / T = 10 := by
  rw [hL, hT]
  norm_num
  done

end train_speed_l102_102737


namespace linear_dependence_k_l102_102045

theorem linear_dependence_k :
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
    (a * (2 : ℝ) + b * (5 : ℝ) = 0) ∧ 
    (a * (3 : ℝ) + b * k = 0) →
  k = 15 / 2 := by
  sorry

end linear_dependence_k_l102_102045


namespace second_yellow_probability_l102_102767

-- Define the conditions in Lean
def BagA : Type := {marble : Int // marble ≥ 0}
def BagB : Type := {marble : Int // marble ≥ 0}
def BagC : Type := {marble : Int // marble ≥ 0}
def BagD : Type := {marble : Int // marble ≥ 0}

noncomputable def marbles_in_A := 4 + 5 + 2
noncomputable def marbles_in_B := 7 + 5
noncomputable def marbles_in_C := 3 + 7
noncomputable def marbles_in_D := 8 + 2

-- Probabilities of drawing specific colors from Bag A
noncomputable def prob_white_A := 4 / 11
noncomputable def prob_black_A := 5 / 11
noncomputable def prob_red_A := 2 / 11

-- Probabilities of drawing a yellow marble from Bags B, C and D
noncomputable def prob_yellow_B := 7 / 12
noncomputable def prob_yellow_C := 3 / 10
noncomputable def prob_yellow_D := 8 / 10

-- Expected probability that the second marble is yellow
noncomputable def prob_second_yellow : ℚ :=
  (prob_white_A * prob_yellow_B) + (prob_black_A * prob_yellow_C) + (prob_red_A * prob_yellow_D)

/-- Prove that the total probability the second marble drawn is yellow is 163/330. -/
theorem second_yellow_probability :
  prob_second_yellow = 163 / 330 := sorry

end second_yellow_probability_l102_102767


namespace repeating_decimal_to_fraction_l102_102352

theorem repeating_decimal_to_fraction : (x : ℚ) (h : x = 0.3666) : x = 11 / 30 :=
sorry

end repeating_decimal_to_fraction_l102_102352


namespace range_of_p_l102_102936

theorem range_of_p (p : ℝ) (a_n b_n : ℕ → ℝ)
  (ha : ∀ n, a_n n = -n + p)
  (hb : ∀ n, b_n n = 3^(n-4))
  (C_n : ℕ → ℝ)
  (hC : ∀ n, C_n n = if a_n n ≥ b_n n then a_n n else b_n n)
  (hc : ∀ n : ℕ, n ≥ 1 → C_n n > C_n 4) :
  4 < p ∧ p < 7 :=
sorry

end range_of_p_l102_102936


namespace xyz_sum_eq_48_l102_102385

theorem xyz_sum_eq_48 (x y z : ℕ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : x * z + y = 47) : 
  x + y + z = 48 := by
  sorry

end xyz_sum_eq_48_l102_102385


namespace loss_percentage_25_l102_102017

variable (C S : ℝ)
variable (h : 15 * C = 20 * S)

theorem loss_percentage_25 (h : 15 * C = 20 * S) : (C - S) / C * 100 = 25 := by
  sorry

end loss_percentage_25_l102_102017


namespace cos_double_angle_example_l102_102061

theorem cos_double_angle_example (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (2 * θ) = -7 / 9 := by
  sorry

end cos_double_angle_example_l102_102061


namespace walked_8_miles_if_pace_4_miles_per_hour_l102_102657

-- Define the conditions
def walked_some_miles_in_2_hours (d : ℝ) : Prop :=
  d = 2

def pace_same_4_miles_per_hour (p : ℝ) : Prop :=
  p = 4

-- Define the proof problem
theorem walked_8_miles_if_pace_4_miles_per_hour :
  ∀ (d p : ℝ), walked_some_miles_in_2_hours d → pace_same_4_miles_per_hour p → (p * d = 8) :=
by
  intros d p h1 h2
  rw [h1, h2]
  exact sorry

end walked_8_miles_if_pace_4_miles_per_hour_l102_102657


namespace bronson_cost_per_bushel_is_12_l102_102770

noncomputable def cost_per_bushel 
  (sale_price_per_apple : ℝ := 0.40)
  (apples_per_bushel : ℕ := 48)
  (profit_from_100_apples : ℝ := 15)
  (number_of_apples_sold : ℕ := 100) 
  : ℝ :=
  let revenue := number_of_apples_sold * sale_price_per_apple
  let cost := revenue - profit_from_100_apples
  let number_of_bushels := (number_of_apples_sold : ℝ) / apples_per_bushel
  cost / number_of_bushels

theorem bronson_cost_per_bushel_is_12 :
  cost_per_bushel = 12 :=
by
  sorry

end bronson_cost_per_bushel_is_12_l102_102770


namespace Ian_hours_worked_l102_102226

theorem Ian_hours_worked (money_left: ℝ) (hourly_rate: ℝ) (spent: ℝ) (earned: ℝ) (hours: ℝ) :
  money_left = 72 → hourly_rate = 18 → spent = earned / 2 → earned = money_left * 2 → 
  earned = hourly_rate * hours → hours = 8 :=
by
  intros h1 h2 h3 h4 h5
  -- Begin mathematical validation process here
  sorry

end Ian_hours_worked_l102_102226


namespace probability_of_sphere_in_cube_l102_102435

noncomputable def cube_volume : Real :=
  (4 : Real)^3

noncomputable def sphere_volume : Real :=
  (4 / 3) * Real.pi * (2 : Real)^3

noncomputable def probability : Real :=
  sphere_volume / cube_volume

theorem probability_of_sphere_in_cube : probability = Real.pi / 6 := by
  sorry

end probability_of_sphere_in_cube_l102_102435


namespace maximum_value_expression_l102_102942

theorem maximum_value_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 :=
sorry

end maximum_value_expression_l102_102942


namespace original_paint_intensity_l102_102541

theorem original_paint_intensity
  (I : ℝ) -- Original intensity of the red paint
  (f : ℝ) -- Fraction of the original paint replaced
  (new_intensity : ℝ) -- Intensity of the new paint
  (replacement_intensity : ℝ) -- Intensity of the replacement red paint
  (hf : f = 2 / 3)
  (hreplacement_intensity : replacement_intensity = 0.30)
  (hnew_intensity : new_intensity = 0.40)
  : I = 0.60 := 
sorry

end original_paint_intensity_l102_102541


namespace union_complement_eq_l102_102690

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l102_102690


namespace distance_between_A_and_B_l102_102419

theorem distance_between_A_and_B 
  (v t t1 : ℝ)
  (h1 : 5 * v * t + 4 * v * t = 9 * v * t)
  (h2 : t1 = 10 / (4.8 * v))
  (h3 : 10 / 4.8 = 25 / 12):
  (9 * v * t + 4 * v * t1) = 450 :=
by 
  -- Proof to be completed
  sorry

end distance_between_A_and_B_l102_102419


namespace angle_and_area_of_triangle_l102_102819

variable {A B C : ℝ}
variable {a b c S : ℝ}

-- Conditions
def triangle_sides (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ (a b c : ℝ) (A B C : ℝ), A + B + C = π ∧ a = 5 ∧ S = (√3 / 2) * (b * c * cos A)

-- Part I: Determine the magnitude of angle A
def angle_A_eq_pi_over_3 (A : ℝ) : Prop := -- Angle A equals π/3
  A = π / 3

-- Part II: Find the area of the triangle with given values of sides and angles
def triangle_area (b c S : ℝ) : Prop :=
  let sin_A := sin (π / 3) in
  let bc := 6 in
  S = (1 / 2) * b * c * sin_A

theorem angle_and_area_of_triangle :
  triangle_sides a b c A B C S ∧ (b + c = 5) ∧ (a = √7) → angle_A_eq_pi_over_3 A ∧ triangle_area b c S := by
  sorry

end angle_and_area_of_triangle_l102_102819


namespace divisible_by_12_for_all_integral_n_l102_102783

theorem divisible_by_12_for_all_integral_n (n : ℤ) : 12 ∣ (2 * n ^ 3 - 2 * n) :=
sorry

end divisible_by_12_for_all_integral_n_l102_102783


namespace trigonometric_identity_l102_102062

variable {α β γ n : Real}

-- Condition:
axiom h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)

-- Statement to be proved:
theorem trigonometric_identity : 
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = (n + 1) / (n - 1) :=
by
  sorry

end trigonometric_identity_l102_102062


namespace area_to_paint_l102_102901

def height_of_wall : ℝ := 10
def length_of_wall : ℝ := 15
def window_height : ℝ := 3
def window_length : ℝ := 3
def door_height : ℝ := 1
def door_length : ℝ := 7

theorem area_to_paint : 
  let total_wall_area := height_of_wall * length_of_wall
  let window_area := window_height * window_length
  let door_area := door_height * door_length
  let area_to_paint := total_wall_area - window_area - door_area
  area_to_paint = 134 := 
by 
  sorry

end area_to_paint_l102_102901


namespace race_completion_times_l102_102950

theorem race_completion_times :
  ∃ (Patrick Manu Amy Olivia Sophie Jack : ℕ),
  Patrick = 60 ∧
  Manu = Patrick + 12 ∧
  Amy = Manu / 2 ∧
  Olivia = (2 * Amy) / 3 ∧
  Sophie = Olivia - 10 ∧
  Jack = Sophie + 8 ∧
  Manu = 72 ∧
  Amy = 36 ∧
  Olivia = 24 ∧
  Sophie = 14 ∧
  Jack = 22 := 
by
  -- proof here
  sorry

end race_completion_times_l102_102950


namespace measure_of_angle_BCD_l102_102509

-- Define angles and sides as given in the problem
variables (α β : ℝ)

-- Conditions: angles and side equalities
axiom angle_ABD_eq_BDC : α = β
axiom angle_DAB_eq_80 : α = 80
axiom side_AB_eq_AD : ∀ AB AD : ℝ, AB = AD
axiom side_DB_eq_DC : ∀ DB DC : ℝ, DB = DC

-- Prove that the measure of angle BCD is 65 degrees
theorem measure_of_angle_BCD : β = 65 :=
sorry

end measure_of_angle_BCD_l102_102509


namespace additional_charge_per_segment_l102_102963

theorem additional_charge_per_segment :
  ∀ (initial_fee total_charge distance : ℝ), 
    initial_fee = 2.35 →
    total_charge = 5.5 →
    distance = 3.6 →
    (total_charge - initial_fee) / (distance / (2 / 5)) = 0.35 :=
by
  intros initial_fee total_charge distance h_initial_fee h_total_charge h_distance
  sorry

end additional_charge_per_segment_l102_102963


namespace result_after_subtraction_l102_102324

-- Define the conditions
def x : ℕ := 40
def subtract_value : ℕ := 138

-- The expression we will evaluate
def result (x : ℕ) : ℕ := 6 * x - subtract_value

-- The theorem stating the evaluated result
theorem result_after_subtraction : result 40 = 102 :=
by
  unfold result
  rw [← Nat.mul_comm]
  simp
  sorry -- Proof placeholder

end result_after_subtraction_l102_102324


namespace blood_drops_per_liter_l102_102753

def mosquito_drops : ℕ := 20
def fatal_blood_loss_liters : ℕ := 3
def mosquitoes_to_kill : ℕ := 750

theorem blood_drops_per_liter (D : ℕ) (total_drops : ℕ) : 
  (total_drops = mosquitoes_to_kill * mosquito_drops) → 
  (fatal_blood_loss_liters * D = total_drops) → 
  D = 5000 := 
  by 
    intros h1 h2
    sorry

end blood_drops_per_liter_l102_102753


namespace simplify_and_evaluate_l102_102538

theorem simplify_and_evaluate :
  let a := (-1: ℝ) / 3
  let b := (-3: ℝ)
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 :=
by
  have a_def : a = (-1: ℝ) / 3 := rfl
  have b_def : b = (-3: ℝ) := rfl
  sorry

end simplify_and_evaluate_l102_102538


namespace choose_copresidents_l102_102748

theorem choose_copresidents (total_members : ℕ) (departments : ℕ) (members_per_department : ℕ) 
    (h1 : total_members = 24) (h2 : departments = 4) (h3 : members_per_department = 6) :
    ∃ ways : ℕ, ways = 54 :=
by
  sorry

end choose_copresidents_l102_102748


namespace problem_1_problem_2_problem_3_problem_4_l102_102317

theorem problem_1 : 3 * Real.sqrt 20 - Real.sqrt 45 - Real.sqrt (1/5) = 14 * Real.sqrt 5 / 5 :=
by sorry

theorem problem_2 : (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 - 1 = 2 :=
by sorry

theorem problem_3 : Real.sqrt 16 + 327 - 2 * Real.sqrt (1/4) = 330 :=
by sorry

theorem problem_4 : (Real.sqrt 3 - Real.sqrt 5) * (Real.sqrt 5 + Real.sqrt 3) - (Real.sqrt 5 - Real.sqrt 3) ^ 2 = 2 * Real.sqrt 15 - 6 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l102_102317


namespace total_trees_planted_l102_102444

/-- A yard is 255 meters long, with a tree at each end and trees planted at intervals of 15 meters. -/
def yard_length : ℤ := 255

def tree_interval : ℤ := 15

def total_trees : ℤ := 18

theorem total_trees_planted (L : ℤ) (d : ℤ) (n : ℤ) : 
  L = yard_length →
  d = tree_interval →
  n = total_trees →
  n = (L / d) + 1 :=
by
  intros hL hd hn
  rw [hL, hd, hn]
  sorry

end total_trees_planted_l102_102444


namespace inequality_comparison_l102_102481

theorem inequality_comparison 
  (a b : ℝ) (x y : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : x^2 / a^2 + y^2 / b^2 ≤ 1) :
  a^2 + b^2 ≥ (x + y)^2 :=
sorry

end inequality_comparison_l102_102481


namespace loss_percentage_is_10_l102_102020

-- Define the conditions
def cost_price (CP : ℝ) : Prop :=
  (550 : ℝ) = 1.1 * CP

def selling_price (SP : ℝ) : Prop :=
  SP = 450

-- Define the main proof statement
theorem loss_percentage_is_10 (CP SP : ℝ) (HCP : cost_price CP) (HSP : selling_price SP) :
  ((CP - SP) / CP) * 100 = 10 :=
by
  -- Translation of the condition into Lean statement
  sorry

end loss_percentage_is_10_l102_102020


namespace xy_difference_squared_l102_102386

theorem xy_difference_squared (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 :=
by
  -- the proof goes here
  sorry

end xy_difference_squared_l102_102386


namespace xy_eq_zero_l102_102294

theorem xy_eq_zero (x y : ℝ) (h1 : x - y = 3) (h2 : x^3 - y^3 = 27) : x * y = 0 := by
  sorry

end xy_eq_zero_l102_102294


namespace min_value_reciprocal_sum_l102_102068

theorem min_value_reciprocal_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) : 
  (∃ c, (∀ x y, x > 0 → y > 0 → x + y = 1 → (1/x + 1/y) ≥ c) ∧ (1/a + 1/b = c)) 
:= 
sorry

end min_value_reciprocal_sum_l102_102068


namespace perimeter_of_given_triangle_l102_102437

structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (right_triangle : (a^2 + b^2 = c^2))

def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

def given_triangle := {a := 10, b := 30, c := 10 * real.sqrt 10, right_triangle := by simp [pow_two, mul_self_sqrt, pow_two]}

theorem perimeter_of_given_triangle : perimeter given_triangle = 40 + 10 * real.sqrt 10 := sorry

end perimeter_of_given_triangle_l102_102437


namespace father_current_age_l102_102708

theorem father_current_age (F S : ℕ) 
  (h₁ : F - 6 = 5 * (S - 6)) 
  (h₂ : (F + 6) + (S + 6) = 78) : 
  F = 51 := 
sorry

end father_current_age_l102_102708


namespace evaluate_expression_l102_102039

theorem evaluate_expression (x : ℤ) (h : x = 2) : 20 - 2 * (3 * x^2 - 4 * x + 8) = -4 :=
by
  rw [h]
  sorry

end evaluate_expression_l102_102039


namespace estimate_probability_concave_l102_102043

noncomputable def times_thrown : ℕ := 1000
noncomputable def frequency_convex : ℝ := 0.44

theorem estimate_probability_concave :
  (1 - frequency_convex) = 0.56 := by
  sorry

end estimate_probability_concave_l102_102043


namespace expand_product_l102_102464

theorem expand_product : ∀ (x : ℝ), (3 * x - 4) * (2 * x + 9) = 6 * x^2 + 19 * x - 36 :=
by
  intro x
  sorry

end expand_product_l102_102464


namespace geometric_sequence_properties_l102_102720

noncomputable def geometric_sequence_sum (a : ℝ) (r : ℝ := a / 120) : ℝ :=
  120 + a + (a * r)

theorem geometric_sequence_properties (a : ℝ) (h_pos : 0 < a) (h_third_term : 120 * r = a)
  (h_third_term_is : a * r = 45 / 28) : 
  a = 5 * Real.sqrt (135 / 7) ∧ geometric_sequence_sum a = 184 :=
by sorry

end geometric_sequence_properties_l102_102720


namespace problem_solution_l102_102734

open Nat

def sum_odd (n : ℕ) : ℕ :=
  n ^ 2

def sum_even (n : ℕ) : ℕ :=
  n * (n + 1)

theorem problem_solution : 
  sum_odd 1010 - sum_even 1009 = 1010 :=
by
  -- Here the proof would go
  sorry

end problem_solution_l102_102734


namespace boat_speed_5_kmh_l102_102746

noncomputable def boat_speed_in_still_water (V_s : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
  (d / t) - V_s

theorem boat_speed_5_kmh :
  boat_speed_in_still_water 5 10 100 = 5 :=
by
  sorry

end boat_speed_5_kmh_l102_102746


namespace seating_arrangements_valid_count_l102_102085

theorem seating_arrangements_valid_count :
  let Martians := 6;
  let Venusians := 5;
  let Earthlings := 4;
  (∃ (N : ℕ),
    (N = 1) ∧
    Martians! * Venusians! * Earthlings! = 720 * 120 * 24) :=
begin
  sorry
end

end seating_arrangements_valid_count_l102_102085


namespace interior_angle_sum_of_regular_polygon_l102_102713

theorem interior_angle_sum_of_regular_polygon (h: ∀ θ, θ = 45) :
  ∃ s, s = 1080 := by
  sorry

end interior_angle_sum_of_regular_polygon_l102_102713


namespace number_of_ordered_pairs_l102_102907

theorem number_of_ordered_pairs (h : ∀ (m n : ℕ), 0 < m → 0 < n → 6/m + 3/n = 1 → true) : 
∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ ∀ (x : ℕ × ℕ), x ∈ s → 0 < x.1 ∧ 0 < x.2 ∧ 6 / ↑x.1 + 3 / ↑x.2 = 1 :=
by
-- Sorry, skipping the proof
  sorry

end number_of_ordered_pairs_l102_102907


namespace doug_lost_marbles_l102_102048

-- Definitions based on the conditions
variables (D D' : ℕ) -- D is the number of marbles Doug originally had, D' is the number Doug has now

-- Condition 1: Ed had 10 more marbles than Doug originally.
def ed_marble_initial (D : ℕ) : ℕ := D + 10

-- Condition 2: Ed had 45 marbles originally.
axiom ed_initial_marble_count : ed_marble_initial D = 45

-- Solve for D from condition 2
noncomputable def doug_initial_marble_count : ℕ := 45 - 10

-- Condition 3: Ed now has 21 more marbles than Doug.
axiom ed_current_marble_difference : 45 = D' + 21

-- Translate what we need to prove
theorem doug_lost_marbles : (doug_initial_marble_count - D') = 11 :=
by
    -- Insert math proof steps here
    sorry

end doug_lost_marbles_l102_102048


namespace number_of_ways_split_2000_cents_l102_102250

theorem number_of_ways_split_2000_cents : 
  ∃ n : ℕ, n = 357 ∧ (∃ (nick d q : ℕ), 
    nick > 0 ∧ d > 0 ∧ q > 0 ∧ 5 * nick + 10 * d + 25 * q = 2000) :=
sorry

end number_of_ways_split_2000_cents_l102_102250


namespace sequence_term_condition_l102_102377

theorem sequence_term_condition (n : ℕ) : (n^2 - 8 * n + 15 = 3) ↔ (n = 2 ∨ n = 6) :=
by 
  sorry

end sequence_term_condition_l102_102377


namespace trig_identity_l102_102928

theorem trig_identity (x : ℝ) (h : Real.cos (x - π / 3) = 1 / 3) :
  Real.cos (2 * x - 5 * π / 3) + Real.sin (π / 3 - x)^2 = 5 / 3 :=
by
  sorry

end trig_identity_l102_102928


namespace accurate_river_length_l102_102011

-- Define the given conditions
def length_GSA := 402
def length_AWRA := 403
def error_margin := 0.5
def probability_of_error := 0.04

-- State the theorem based on these conditions
theorem accurate_river_length : 
  ∀ Length_GSA Length_AWRA error_margin probability_of_error, 
  Length_GSA = 402 → 
  Length_AWRA = 403 → 
  error_margin = 0.5 → 
  probability_of_error = 0.04 → 
  (this based on independent measurements with above error margins)
  combined_length = 402.5 ∧ combined_probability_of_error = 0.04 :=
by 
  -- Proof to be completed
  sorry

end accurate_river_length_l102_102011


namespace final_value_of_S_is_10_l102_102983

-- Define the initial value of S
def initial_S : ℕ := 1

-- Define the sequence of I values
def I_values : List ℕ := [1, 3, 5]

-- Define the update operation on S
def update_S (S : ℕ) (I : ℕ) : ℕ := S + I

-- Final value of S after all updates
def final_S : ℕ := (I_values.foldl update_S initial_S)

-- The theorem stating that the final value of S is 10
theorem final_value_of_S_is_10 : final_S = 10 :=
by
  sorry

end final_value_of_S_is_10_l102_102983


namespace Rio_Coralio_Length_Estimate_l102_102009

def RioCoralioLength := 402.5
def GSA_length := 402
def AWRA_length := 403
def error_margin := 0.5
def error_probability := 0.04

theorem Rio_Coralio_Length_Estimate :
  ∀ (L_GSA L_AWRA : ℝ) (margin error_prob : ℝ),
  L_GSA = GSA_length ∧ L_AWRA = AWRA_length ∧ 
  margin = error_margin ∧ error_prob = error_probability →
  (RioCoralioLength = 402.5) ∧ (error_probability = 0.04) := 
by 
  intros L_GSA L_AWRA margin error_prob h,
  sorry

end Rio_Coralio_Length_Estimate_l102_102009


namespace power_increase_fourfold_l102_102897

theorem power_increase_fourfold 
    (F v : ℝ)
    (k : ℝ)
    (R : ℝ := k * v)
    (P_initial : ℝ := F * v)
    (v' : ℝ := 2 * v)
    (F' : ℝ := 2 * F)
    (R' : ℝ := k * v')
    (P_final : ℝ := F' * v') :
    P_final = 4 * P_initial := 
by
  sorry

end power_increase_fourfold_l102_102897


namespace masha_mushrooms_l102_102473

theorem masha_mushrooms (B1 B2 B3 B4 G1 G2 G3 : ℕ) (total : B1 + B2 + B3 + B4 + G1 + G2 + G3 = 70)
  (girls_distinct : G1 ≠ G2 ∧ G1 ≠ G3 ∧ G2 ≠ G3)
  (boys_threshold : ∀ {A B C D : ℕ}, (A = B1 ∨ A = B2 ∨ A = B3 ∨ A = B4) →
                    (B = B1 ∨ B = B2 ∨ B = B3 ∨ B = B4) →
                    (C = B1 ∨ C = B2 ∨ C = B3 ∨ C = B4) → 
                    (A ≠ B ∧ A ≠ C ∧ B ≠ C) →
                    A + B + C ≥ 43)
  (diff_no_more_than_five_times : ∀ {x y : ℕ}, (x = B1 ∨ x = B2 ∨ x = B3 ∨ x = B4 ∨ x = G1 ∨ x = G2 ∨ x = G3) →
                                  (y = B1 ∨ y = B2 ∨ y = B3 ∨ y = B4 ∨ y = G1 ∨ y = G2 ∨ y = G3) →
                                  x ≠ y → x ≤ 5 * y ∧ y ≤ 5 * x)
  (masha_max_girl : G3 = max G1 (max G2 G3))
  : G3 = 5 :=
sorry

end masha_mushrooms_l102_102473


namespace ant_travel_distance_l102_102132

theorem ant_travel_distance (r1 r2 r3 : ℝ) (h1 : r1 = 5) (h2 : r2 = 10) (h3 : r3 = 15) :
  let A_large := (1/3) * 2 * Real.pi * r3
  let D_radial := (r3 - r2) + (r2 - r1)
  let A_middle := (1/3) * 2 * Real.pi * r2
  let D_small := 2 * r1
  let A_small := (1/2) * 2 * Real.pi * r1
  A_large + D_radial + A_middle + D_small + A_small = (65 * Real.pi / 3) + 20 :=
by
  sorry

end ant_travel_distance_l102_102132


namespace brads_running_speed_proof_l102_102977

noncomputable def brads_speed (distance_between_homes : ℕ) (maxwells_speed : ℕ) (maxwells_time : ℕ) (brad_start_delay : ℕ) : ℕ :=
  let distance_covered_by_maxwell := maxwells_speed * maxwells_time
  let distance_covered_by_brad := distance_between_homes - distance_covered_by_maxwell
  let brads_time := maxwells_time - brad_start_delay
  distance_covered_by_brad / brads_time

theorem brads_running_speed_proof :
  brads_speed 54 4 6 1 = 6 := 
by
  unfold brads_speed
  rfl

end brads_running_speed_proof_l102_102977


namespace trigonometric_identity_l102_102174

theorem trigonometric_identity :
  1 / Real.sin (70 * Real.pi / 180) - Real.sqrt 2 / Real.cos (70 * Real.pi / 180) = 
  -2 * (Real.sin (25 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
sorry

end trigonometric_identity_l102_102174


namespace faction_with_more_liars_than_truth_tellers_l102_102394

theorem faction_with_more_liars_than_truth_tellers 
  (r1 r2 r3 l1 l2 l3 : ℕ) 
  (H1 : r1 + r2 + r3 + l1 + l2 + l3 = 2016)
  (H2 : r1 + l2 + l3 = 1208)
  (H3 : r2 + l1 + l3 = 908)
  (H4 : r3 + l1 + l2 = 608) :
  l3 - r3 = 100 :=
by
  sorry

end faction_with_more_liars_than_truth_tellers_l102_102394


namespace breakfast_plate_contains_2_eggs_l102_102766

-- Define the conditions
def breakfast_plate := Nat
def num_customers := 14
def num_bacon_strips := 56

-- Define the bacon strips per plate
def bacon_strips_per_plate (num_bacon_strips num_customers : Nat) : Nat :=
  num_bacon_strips / num_customers

-- Define the number of eggs per plate given twice as many bacon strips as eggs
def eggs_per_plate (bacon_strips_per_plate : Nat) : Nat :=
  bacon_strips_per_plate / 2

-- The main theorem we need to prove
theorem breakfast_plate_contains_2_eggs :
  eggs_per_plate (bacon_strips_per_plate 56 14) = 2 :=
by
  sorry

end breakfast_plate_contains_2_eggs_l102_102766


namespace extremum_f_range_a_for_no_zeros_l102_102073

noncomputable def f (a b x : ℝ) : ℝ :=
  (a * (x - 1) + b * Real.exp x) / Real.exp x

theorem extremum_f (a b : ℝ) (h_a_ne_zero : a ≠ 0) :
  (∃ (x : ℝ), a = -1 ∧ b = 0 ∧ f a b x = -1 / Real.exp 2) := sorry

theorem range_a_for_no_zeros (a : ℝ) :
  (∀ x : ℝ, a * x - a + Real.exp x ≠ 0) ↔ (-Real.exp 2 < a ∧ a < 0) := sorry

end extremum_f_range_a_for_no_zeros_l102_102073


namespace boys_girls_relationship_l102_102330

theorem boys_girls_relationship (b g : ℕ): (4 + 2 * b = g) → (b = (g - 4) / 2) :=
by
  intros h
  sorry

end boys_girls_relationship_l102_102330


namespace intersection_A_B_l102_102827

-- Define sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

-- The theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 3} :=
by
  sorry -- proof is skipped as instructed

end intersection_A_B_l102_102827


namespace profit_margin_A_cost_price_B_units_purchased_l102_102320

variables (cost_price_A selling_price_A selling_price_B profit_margin_B total_units total_cost : ℕ)
variables (units_A units_B : ℕ)

-- Conditions
def condition1 : cost_price_A = 40 := sorry
def condition2 : selling_price_A = 60 := sorry
def condition3 : selling_price_B = 80 := sorry
def condition4 : profit_margin_B = 60 := sorry
def condition5 : total_units = 50 := sorry
def condition6 : total_cost = 2200 := sorry

-- Proof statements 
theorem profit_margin_A (h1 : cost_price_A = 40) (h2 : selling_price_A = 60) :
  (selling_price_A - cost_price_A) * 100 / cost_price_A = 50 :=
by sorry

theorem cost_price_B (h3 : selling_price_B = 80) (h4 : profit_margin_B = 60) :
  (selling_price_B * 100) / (100 + profit_margin_B) = 50 :=
by sorry

theorem units_purchased (h5 : 40 * units_A + 50 * units_B = 2200)
  (h6 : units_A + units_B = 50) :
  units_A = 30 ∧ units_B = 20 :=
by sorry


end profit_margin_A_cost_price_B_units_purchased_l102_102320


namespace gardening_project_cost_l102_102334

noncomputable def totalCost : Nat :=
  let roseBushes := 20
  let costPerRoseBush := 150
  let gardenerHourlyRate := 30
  let gardenerHoursPerDay := 5
  let gardenerDays := 4
  let soilCubicFeet := 100
  let soilCostPerCubicFoot := 5

  let costOfRoseBushes := costPerRoseBush * roseBushes
  let gardenerTotalHours := gardenerDays * gardenerHoursPerDay
  let costOfGardener := gardenerHourlyRate * gardenerTotalHours
  let costOfSoil := soilCostPerCubicFoot * soilCubicFeet

  costOfRoseBushes + costOfGardener + costOfSoil

theorem gardening_project_cost : totalCost = 4100 := by
  sorry

end gardening_project_cost_l102_102334


namespace sum_first_five_terms_l102_102799

-- Define the geometric sequence
noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^n

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n
  else a1 * (1 - q^(n + 1)) / (1 - q)

-- Given conditions
def a1 : ℝ := 1
def q : ℝ := 2
def n : ℕ := 5

-- The theorem to be proven
theorem sum_first_five_terms : sum_geometric_sequence a1 q (n-1) = 31 := by
  sorry

end sum_first_five_terms_l102_102799


namespace additional_charge_per_segment_l102_102962

variable (initial_fee : ℝ := 2.35)
variable (total_charge : ℝ := 5.5)
variable (distance : ℝ := 3.6)
variable (segment_length : ℝ := (2/5 : ℝ))

theorem additional_charge_per_segment :
  let number_of_segments := distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  let additional_charge_per_segment := charge_for_distance / number_of_segments
  additional_charge_per_segment = 0.35 :=
by
  sorry

end additional_charge_per_segment_l102_102962


namespace fraction_ratio_x_div_y_l102_102383

theorem fraction_ratio_x_div_y (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : y / (x + z) = (x - y) / z) 
(h5 : y / (x + z) = x / (y + 2 * z)) :
  x / y = 2 / 3 := 
  sorry

end fraction_ratio_x_div_y_l102_102383


namespace tenth_term_in_arithmetic_sequence_l102_102733

theorem tenth_term_in_arithmetic_sequence :
  let a := (1:ℚ) / 2
  let d := (1:ℚ) / 3
  a + 9 * d = 7 / 2 :=
by
  sorry

end tenth_term_in_arithmetic_sequence_l102_102733


namespace greatest_common_divisor_three_divisors_l102_102563

theorem greatest_common_divisor_three_divisors (m : ℕ) (h : ∃ (D : set ℕ), D = {d | d ∣ 120 ∧ d ∣ m} ∧ D.card = 3) : 
  ∃ p : ℕ, p.prime ∧ greatest_dvd_set {d | d ∣ 120 ∧ d ∣ m} = p^2 := 
sorry

end greatest_common_divisor_three_divisors_l102_102563


namespace angle_B_is_40_degrees_l102_102674

theorem angle_B_is_40_degrees (angle_A angle_B angle_C : ℝ)
  (h1 : angle_A = 3 * angle_B)
  (h2 : angle_B = 2 * angle_C)
  (triangle_sum : angle_A + angle_B + angle_C = 180) :
  angle_B = 40 :=
by
  sorry

end angle_B_is_40_degrees_l102_102674


namespace find_a_find_m_l102_102643

-- Conditions: f(x) is an odd function with domain \(\mathbb{R}\)
def f (x : ℝ) (a : ℝ) : ℝ := (a - real.exp x) / (real.exp x + a)

-- Proof problem 1: Prove that a = 1 given f(x) is odd
theorem find_a (a : ℝ) (h : ∀ x : ℝ, f x a = - f (-x) a) : a = 1 :=
sorry

-- Conditions: f(2^(x+1) - 4^x) + f(1 - m) > 0 always holds for all x ∈ [1, 2]
def g (x : ℝ) : ℝ := f (2^(x + 1) - 4^x) 1 -- f uses the value a = 1

-- Proof problem 2: Prove that the range of real number m is (1, ∞)
theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → g x + f (1 - m) 1 > 0) : 1 < m :=
sorry

end find_a_find_m_l102_102643


namespace circle_tangent_line_l102_102792

theorem circle_tangent_line 
    (center : ℝ × ℝ) (line_eq : ℝ → ℝ → ℝ) 
    (tangent_eq : ℝ) :
    center = (-1, 1) →
    line_eq 1 (-1)= 0 →
    tangent_eq = 2 :=
  let h := -1;
  let k := 1;
  let radius := Real.sqrt 2;
  sorry

end circle_tangent_line_l102_102792


namespace range_of_solutions_l102_102616

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end range_of_solutions_l102_102616


namespace range_of_k_for_distinct_real_roots_l102_102941

theorem range_of_k_for_distinct_real_roots (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 - 2 * x + 1 = 0) → (k < 2 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_for_distinct_real_roots_l102_102941


namespace remainder_783245_div_7_l102_102732

theorem remainder_783245_div_7 :
  783245 % 7 = 1 :=
sorry

end remainder_783245_div_7_l102_102732


namespace part1_part2_part3_l102_102596

-- Part 1
theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem part2 (a m n : ℤ) (h1 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) (h2 : 0 < a) (h3 : 0 < m) (h4 : 0 < n) : 
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem part3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 :=
sorry

end part1_part2_part3_l102_102596


namespace solution_interval_l102_102601

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end solution_interval_l102_102601


namespace mode_of_list_is_five_l102_102510

def list := [3, 4, 5, 5, 5, 5, 7, 11, 21]

def occurrence_count (l : List ℕ) (x : ℕ) : ℕ :=
  l.count x

def is_mode (l : List ℕ) (x : ℕ) : Prop :=
  ∀ y : ℕ, occurrence_count l x ≥ occurrence_count l y

theorem mode_of_list_is_five : is_mode list 5 := by
  sorry

end mode_of_list_is_five_l102_102510


namespace quadratic_solution_l102_102812

theorem quadratic_solution (m n : ℝ) (h1 : m ≠ 0) (h2 : m * 1^2 + n * 1 - 1 = 0) : m + n = 1 :=
sorry

end quadratic_solution_l102_102812


namespace max_term_of_sequence_l102_102456

noncomputable def a_n (n : ℕ) : ℚ := (n^2 : ℚ) / (2^n : ℚ)

theorem max_term_of_sequence :
  ∃ n : ℕ, (∀ m : ℕ, a_n n ≥ a_n m) ∧ a_n n = 9 / 8 :=
sorry

end max_term_of_sequence_l102_102456


namespace ratio_of_red_to_total_l102_102647

def hanna_erasers : Nat := 4
def tanya_total_erasers : Nat := 20

def rachel_erasers (hanna_erasers : Nat) : Nat :=
  hanna_erasers / 2

def tanya_red_erasers (rachel_erasers : Nat) : Nat :=
  2 * (rachel_erasers + 3)

theorem ratio_of_red_to_total (hanna_erasers tanya_total_erasers : Nat)
  (hanna_has_4 : hanna_erasers = 4) 
  (tanya_total_is_20 : tanya_total_erasers = 20) 
  (twice_as_many : hanna_erasers = 2 * (rachel_erasers hanna_erasers)) 
  (three_less_than_half : rachel_erasers hanna_erasers = (1 / 2:Rat) * (tanya_red_erasers (rachel_erasers hanna_erasers)) - 3) :
  (tanya_red_erasers (rachel_erasers hanna_erasers)) / tanya_total_erasers = 1 / 2 := by
  sorry

end ratio_of_red_to_total_l102_102647


namespace percentage_error_in_area_l102_102598

theorem percentage_error_in_area (S : ℝ) (h : S > 0) :
  let S' := S * 1.06
  let A := S^2
  let A' := (S')^2
  (A' - A) / A * 100 = 12.36 := by
  sorry

end percentage_error_in_area_l102_102598


namespace one_half_of_scientific_notation_l102_102140

theorem one_half_of_scientific_notation :
  (1 / 2) * (1.2 * 10 ^ 30) = 6.0 * 10 ^ 29 :=
by
  sorry

end one_half_of_scientific_notation_l102_102140


namespace find_skirts_l102_102142

variable (blouses : ℕ) (skirts : ℕ) (slacks : ℕ)
variable (blouses_in_hamper : ℕ) (slacks_in_hamper : ℕ) (skirts_in_hamper : ℕ)
variable (clothes_in_hamper : ℕ)

-- Given conditions
axiom h1 : blouses = 12
axiom h2 : slacks = 8
axiom h3 : blouses_in_hamper = (75 * blouses) / 100
axiom h4 : slacks_in_hamper = (25 * slacks) / 100
axiom h5 : skirts_in_hamper = 3
axiom h6 : clothes_in_hamper = blouses_in_hamper + slacks_in_hamper + skirts_in_hamper
axiom h7 : clothes_in_hamper = 11

-- Proof goal: proving the total number of skirts
theorem find_skirts : skirts_in_hamper = (50 * skirts) / 100 → skirts = 6 :=
by sorry

end find_skirts_l102_102142


namespace problem1_l102_102318

theorem problem1
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h₁ : (3*x - 2)^(6) = a₀ + a₁ * (2*x - 1) + a₂ * (2*x - 1)^2 + a₃ * (2*x - 1)^3 + a₄ * (2*x - 1)^4 + a₅ * (2*x - 1)^5 + a₆ * (2*x - 1)^6)
  (h₂ : a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1)
  (h₃ : a₀ - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ = 64) :
  (a₁ + a₃ + a₅) / (a₀ + a₂ + a₄ + a₆) = -63 / 65 := by
  sorry

end problem1_l102_102318


namespace solution_l102_102342

noncomputable def F (a b c : ℝ) := a * (b ^ 3) + c

theorem solution (a : ℝ) (h : F a 2 3 = F a 3 10) : a = -7 / 19 := sorry

end solution_l102_102342


namespace find_principal_l102_102876

theorem find_principal
  (R : ℝ) (T : ℕ) (interest_less_than_principal : ℝ) : 
  R = 0.05 → 
  T = 10 → 
  interest_less_than_principal = 3100 → 
  ∃ P : ℝ, P - ((P * R * T): ℝ) = P - interest_less_than_principal ∧ P = 6200 :=
by
  sorry

end find_principal_l102_102876


namespace marked_box_in_second_row_l102_102626

theorem marked_box_in_second_row:
  ∀ a b c d e f g h : ℕ, 
  (e = a + b) → 
  (f = b + c) →
  (g = c + d) →
  (h = a + 2 * b + c) →
  ((a = 5) ∧ (d = 6)) →
  ((a = 3) ∨ (b = 3) ∨ (c = 3) ∨ (d = 3)) →
  (f = 3) :=
by
  sorry

end marked_box_in_second_row_l102_102626


namespace runner_speed_ratio_l102_102869

theorem runner_speed_ratio (d s u v_f v_s : ℝ) (hs : s ≠ 0) (hu : u ≠ 0)
  (H1 : (v_f + v_s) * s = d) (H2 : (v_f - v_s) * u = v_s * u) :
  v_f / v_s = 2 :=
by
  sorry

end runner_speed_ratio_l102_102869


namespace range_of_a_l102_102379

/--
Given the parabola \(x^2 = y\), points \(A\) and \(B\) are on the parabola and located on both sides of the y-axis,
and the line \(AB\) intersects the y-axis at point \((0, a)\). If \(\angle AOB\) is an acute angle (where \(O\) is the origin),
then the real number \(a\) is greater than 1.
-/
theorem range_of_a (a : ℝ) (x1 x2 : ℝ) : (x1^2 = x2^2) → (x1 * x2 = -a) → ((-a + a^2) > 0) → (1 < a) :=
by 
  sorry

end range_of_a_l102_102379


namespace bricks_needed_for_room_floor_l102_102310

-- Conditions
def length : ℕ := 4
def breadth : ℕ := 5
def bricks_per_square_meter : ℕ := 17

-- Question and Answer (Proof Problem)
theorem bricks_needed_for_room_floor : 
  (length * breadth) * bricks_per_square_meter = 340 := by
  sorry

end bricks_needed_for_room_floor_l102_102310


namespace roots_eq_two_iff_a_gt_neg1_l102_102197

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l102_102197


namespace problem_ab_value_l102_102476

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x^2 - 4 * x else a * x^2 + b * x

theorem problem_ab_value (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-x) a b) → a * b = 12 :=
by
  intro h
  let f_eqn := h 1 -- Checking the function equality for x = 1
  sorry

end problem_ab_value_l102_102476


namespace mass_percentage_of_O_in_dichromate_l102_102208

noncomputable def molar_mass_Cr : ℝ := 52.00
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def molar_mass_Cr2O7_2_minus : ℝ := (2 * molar_mass_Cr) + (7 * molar_mass_O)

theorem mass_percentage_of_O_in_dichromate :
  (7 * molar_mass_O / molar_mass_Cr2O7_2_minus) * 100 = 51.85 := 
by
  sorry

end mass_percentage_of_O_in_dichromate_l102_102208


namespace max_difference_of_mean_505_l102_102826

theorem max_difference_of_mean_505 (x y : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : 100 ≤ y ∧ y ≤ 999) (h3 : (x + y) / 2 = 505) : 
  x - y ≤ 810 :=
sorry

end max_difference_of_mean_505_l102_102826


namespace wire_not_used_is_20_l102_102349

def initial_wire_length : ℕ := 50
def number_of_parts : ℕ := 5
def parts_used : ℕ := 3

def length_of_each_part (total_length : ℕ) (parts : ℕ) : ℕ := total_length / parts
def length_used (length_each_part : ℕ) (used_parts : ℕ) : ℕ := length_each_part * used_parts
def wire_not_used (total_length : ℕ) (used_length : ℕ) : ℕ := total_length - used_length

theorem wire_not_used_is_20 : 
  wire_not_used initial_wire_length 
    (length_used 
      (length_of_each_part initial_wire_length number_of_parts) 
    parts_used) = 20 := by
  sorry

end wire_not_used_is_20_l102_102349


namespace solve_for_x_l102_102658

theorem solve_for_x (x y : ℚ) (h1 : 3 * x - 2 * y = 8) (h2 : x + 3 * y = 7) : x = 38 / 11 :=
by
  sorry

end solve_for_x_l102_102658


namespace shop_owner_cheat_selling_percentage_l102_102164

noncomputable def percentage_cheat_buying : ℝ := 12
noncomputable def profit_percentage : ℝ := 40
noncomputable def percentage_cheat_selling : ℝ := 20

theorem shop_owner_cheat_selling_percentage 
  (percentage_cheat_buying : ℝ := 12)
  (profit_percentage : ℝ := 40) :
  percentage_cheat_selling = 20 := 
sorry

end shop_owner_cheat_selling_percentage_l102_102164


namespace find_y_l102_102096

variable (a b y : ℝ)
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(3 * b) = a^b * y^b)

theorem find_y : y = 27 * a^2 :=
  by sorry

end find_y_l102_102096


namespace solution_range_l102_102611

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end solution_range_l102_102611


namespace mathlib_problem_l102_102516

/-- Given positive integers a, b, c, define d = gcd(a, b, c) and a = dx, b = dy, c = dz.
Prove that there exists a positive integer N such that
  a ∣ Nbc + b + c,
  b ∣ Nca + c + a,
  c ∣ Nab + a + b
if and only if 
  x, y, z are pairwise coprime, 
  gcd(d, xyz) ∣ x + y + z. 
-/
theorem mathlib_problem (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let d := Nat.gcd (Nat.gcd a b) c in
  let x := a / d in
  let y := b / d in
  let z := c / d in
  (∃ (N : ℕ), a ∣ N * b * c + b + c ∧ b ∣ N * c * a + c + a ∧ c ∣ N * a * b + a + b) ↔ 
  Nat.coprime x y ∧ Nat.coprime y z ∧ Nat.coprime z x ∧ Nat.gcd d (x * y * z) ∣ (x + y + z) :=
by
  sorry

end mathlib_problem_l102_102516


namespace arithmetic_sequence_ratio_l102_102920

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : ∀ n, S n / a n = (n + 1) / 2) :
  (a 2 / a 3 = 2 / 3) :=
sorry

end arithmetic_sequence_ratio_l102_102920


namespace remainder_of_division_l102_102209

def compute_remainder (dividend divisor : Polynomial ℤ) (remainder : Polynomial ℤ) : Prop :=
  dividend % divisor = remainder

theorem remainder_of_division :
  compute_remainder (Polynomial.C 1 + Polynomial.X^4) (Polynomial.C 6 - Polynomial.C 4 * Polynomial.X + Polynomial.X^2) (Polynomial.C (-59) + Polynomial.C 16 * Polynomial.X) :=
by sorry

end remainder_of_division_l102_102209


namespace number_of_ways_to_place_coins_l102_102506

theorem number_of_ways_to_place_coins :
  (nat.choose 7 2) = 21 :=
by
  sorry

end number_of_ways_to_place_coins_l102_102506


namespace area_of_rectangle_is_270_l102_102854

noncomputable def side_of_square := Real.sqrt 2025

noncomputable def radius_of_circle := side_of_square

noncomputable def length_of_rectangle := (2/5 : ℝ) * radius_of_circle

noncomputable def initial_breadth_of_rectangle := (1/2 : ℝ) * length_of_rectangle + 5

noncomputable def breadth_of_rectangle := if (length_of_rectangle + initial_breadth_of_rectangle) % 3 = 0 
                                          then initial_breadth_of_rectangle 
                                          else initial_breadth_of_rectangle + 1

noncomputable def area_of_rectangle := length_of_rectangle * breadth_of_rectangle

theorem area_of_rectangle_is_270 :
  area_of_rectangle = 270 := by
  sorry

end area_of_rectangle_is_270_l102_102854


namespace constant_S13_l102_102629

noncomputable def S (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem constant_S13 (a d p : ℝ) 
  (h : a + a + 3 * d + a + 7 * d = p) : 
  S a d 13 = 13 * p / 18 :=
by
  unfold S
  sorry

end constant_S13_l102_102629


namespace calculate_a2_b2_c2_l102_102946

theorem calculate_a2_b2_c2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = -3) (h3 : a * b * c = 2) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end calculate_a2_b2_c2_l102_102946


namespace mult_mod_7_zero_l102_102773

theorem mult_mod_7_zero :
  (2007 ≡ 5 [MOD 7]) →
  (2008 ≡ 6 [MOD 7]) →
  (2009 ≡ 0 [MOD 7]) →
  (2010 ≡ 1 [MOD 7]) →
  (2007 * 2008 * 2009 * 2010 ≡ 0 [MOD 7]) :=
by
  intros h1 h2 h3 h4
  sorry

end mult_mod_7_zero_l102_102773


namespace minimum_expression_value_l102_102315

theorem minimum_expression_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_sum : a + b + c + d = 4) :
  (∃ x, x = (1/2) ∨
   (∃ y, y ≤ (∑ e in [a, b, c, d], e^8 / prod (λ f, if f = e then (e^2 + 1) else f^2 + e) [a, b, c, d]) ∧ y = x)) := sorry

end minimum_expression_value_l102_102315


namespace paul_oil_change_rate_l102_102263

theorem paul_oil_change_rate (P : ℕ) (h₁ : 8 * (P + 3) = 40) : P = 2 :=
by
  sorry

end paul_oil_change_rate_l102_102263


namespace largest_divisor_69_86_l102_102305

theorem largest_divisor_69_86 (n : ℕ) (h₁ : 69 % n = 5) (h₂ : 86 % n = 6) : n = 16 := by
  sorry

end largest_divisor_69_86_l102_102305


namespace find_a_l102_102384

theorem find_a (a : ℕ) (h : a * 2 * 2^3 = 2^6) : a = 4 := 
by 
  sorry

end find_a_l102_102384


namespace valid_range_of_x_l102_102242

theorem valid_range_of_x (x : ℝ) (h1 : 2 - x ≥ 0) (h2 : x + 1 ≠ 0) : x ≤ 2 ∧ x ≠ -1 :=
sorry

end valid_range_of_x_l102_102242


namespace wire_not_used_l102_102347

variable (total_wire length_cut_parts parts_used : ℕ)

theorem wire_not_used (h1 : total_wire = 50) (h2 : length_cut_parts = 5) (h3 : parts_used = 3) : 
  total_wire - (parts_used * (total_wire / length_cut_parts)) = 20 := 
  sorry

end wire_not_used_l102_102347


namespace joker_probability_l102_102752

-- Definition of the problem parameters according to the conditions
def total_cards := 54
def jokers := 2

-- Calculate the probability
def probability (favorable : Nat) (total : Nat) : ℚ :=
  favorable / total

-- State the theorem that we want to prove
theorem joker_probability : probability jokers total_cards = 1 / 27 := by
  sorry

end joker_probability_l102_102752


namespace triangle_possible_sides_l102_102137

theorem triangle_possible_sides (a b c : ℕ) (h₁ : a + b + c = 7) (h₂ : a + b > c) (h₃ : a + c > b) (h₄ : b + c > a) :
  a = 1 ∨ a = 2 ∨ a = 3 :=
by {
  sorry
}

end triangle_possible_sides_l102_102137


namespace solve_quadratic_equation_1_solve_quadratic_equation_2_l102_102987

theorem solve_quadratic_equation_1 (x : ℝ) :
  3 * x^2 + 2 * x - 1 = 0 ↔ x = 1/3 ∨ x = -1 :=
by sorry

theorem solve_quadratic_equation_2 (x : ℝ) :
  (x + 2) * (x - 3) = 5 * x - 15 ↔ x = 3 :=
by sorry

end solve_quadratic_equation_1_solve_quadratic_equation_2_l102_102987


namespace sum_S_r_is_zero_matrix_l102_102824

open Matrix

variables {n r : ℕ}

def S_r (n r : ℕ) [fact (1 ≤ r)] [fact (r ≤ n)] : set (matrix (fin n) (fin n) (zmod 2)) :=
{A | rank A = r}

theorem sum_S_r_is_zero_matrix (n r : ℕ) [fact (2 ≤ n)] [fact (1 ≤ r)] [fact (r ≤ n)]
    : ∑ X in (S_r n r), X = 0 := 
by
  sorry

end sum_S_r_is_zero_matrix_l102_102824


namespace number_of_boys_is_60_l102_102005

-- Definitions based on conditions
def total_students : ℕ := 150

def number_of_boys (x : ℕ) : Prop :=
  ∃ g : ℕ, x + g = total_students ∧ g = (x * total_students) / 100

-- Theorem statement
theorem number_of_boys_is_60 : number_of_boys 60 := 
sorry

end number_of_boys_is_60_l102_102005


namespace find_m_for_parallel_lines_l102_102225

-- The given lines l1 and l2
def line1 (m: ℝ) : Prop := ∀ x y : ℝ, (3 + m) * x - 4 * y = 5 - 3 * m
def line2 : Prop := ∀ x y : ℝ, 2 * x - y = 8

-- Definition for parallel lines
def parallel_lines (l₁ l₂ : Prop) : Prop := 
  ∃ m : ℝ, (3 + m) / 4 = 2

-- The main theorem to prove
theorem find_m_for_parallel_lines (m: ℝ) (h: parallel_lines (line1 m) line2) : m = 5 :=
by sorry

end find_m_for_parallel_lines_l102_102225


namespace mean_temperature_l102_102278

def temperatures : List ℚ := [80, 79, 81, 85, 87, 89, 87, 90, 89, 88]

theorem mean_temperature :
  let n := temperatures.length
  let sum := List.sum temperatures
  (sum / n : ℚ) = 85.5 :=
by
  sorry

end mean_temperature_l102_102278


namespace point_B_coordinates_l102_102508

theorem point_B_coordinates :
  ∃ (B : ℝ × ℝ), (B.1 < 0) ∧ (|B.2| = 4) ∧ (|B.1| = 5) ∧ (B = (-5, 4) ∨ B = (-5, -4)) :=
sorry

end point_B_coordinates_l102_102508


namespace intersection_point_lines_distance_point_to_line_l102_102763

-- Problem 1
theorem intersection_point_lines :
  ∃ (x y : ℝ), (x - y + 2 = 0) ∧ (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 1) :=
sorry

-- Problem 2
theorem distance_point_to_line :
  ∀ (x y : ℝ), (x = 1) ∧ (y = -2) → ∃ d : ℝ, d = 3 ∧ (d = abs (3 * x + 4 * y - 10) / (Real.sqrt (3^2 + 4^2))) :=
sorry

end intersection_point_lines_distance_point_to_line_l102_102763


namespace intersect_A_B_l102_102371

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -1 < x ∧ x ≤ 1}

theorem intersect_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersect_A_B_l102_102371


namespace proportional_function_quadrants_l102_102235

theorem proportional_function_quadrants (k : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = k * x) ∧ (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = k * x) → k < 0 :=
by
  sorry

end proportional_function_quadrants_l102_102235


namespace sum_of_two_numbers_l102_102127

theorem sum_of_two_numbers (a b : ℝ) (h1 : a + b = 25) (h2 : a * b = 144) (h3 : |a - b| = 7) : a + b = 25 := 
  by
  sorry

end sum_of_two_numbers_l102_102127


namespace circle_radius_value_l102_102921

theorem circle_radius_value (k : ℝ) :
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + k = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25) → k = 16 :=
by
  sorry

end circle_radius_value_l102_102921


namespace ground_beef_lean_beef_difference_l102_102789

theorem ground_beef_lean_beef_difference (x y z : ℕ) 
  (h1 : x + y + z = 20) 
  (h2 : y + 2 * z = 18) :
  x - z = 2 :=
sorry

end ground_beef_lean_beef_difference_l102_102789


namespace percentage_students_enrolled_in_bio_l102_102900

-- Problem statement
theorem percentage_students_enrolled_in_bio (total_students : ℕ) (students_not_in_bio : ℕ) 
    (h1 : total_students = 880) (h2 : students_not_in_bio = 462) : 
    ((total_students - students_not_in_bio : ℚ) / total_students) * 100 = 47.5 := by 
  -- Proof is omitted
  sorry

end percentage_students_enrolled_in_bio_l102_102900


namespace three_integers_same_parity_l102_102764

theorem three_integers_same_parity (a b c : ℤ) : 
  (∃ i j, i ≠ j ∧ (i = a ∨ i = b ∨ i = c) ∧ (j = a ∨ j = b ∨ j = c) ∧ (i % 2 = j % 2)) :=
by
  sorry

end three_integers_same_parity_l102_102764


namespace final_value_of_A_l102_102760

theorem final_value_of_A (A : ℤ) (h₁ : A = 15) (h₂ : A = -A + 5) : A = -10 := 
by 
  sorry

end final_value_of_A_l102_102760


namespace num_sheets_in_stack_l102_102756

-- Definitions coming directly from the conditions
def thickness_ream := 4 -- cm
def num_sheets_ream := 400
def height_stack := 10 -- cm

-- The final proof statement
theorem num_sheets_in_stack : (height_stack / (thickness_ream / num_sheets_ream)) = 1000 :=
by
  sorry

end num_sheets_in_stack_l102_102756


namespace distribute_teachers_l102_102344

theorem distribute_teachers :
  let schools := {A, B, C, D}
  let teachers := 6
  let min_teachers_A := 2
  let min_teachers_B := 1
  let min_teachers_C := 1
  let min_teachers_D := 1
  ∃ (distribution : (ℕ × ℕ × ℕ × ℕ)),
    (distribution.1 + distribution.2 + distribution.3 + distribution.4 = teachers) ∧
    (distribution.1 ≥ min_teachers_A) ∧
    (distribution.2 ≥ min_teachers_B) ∧
    (distribution.3 ≥ min_teachers_C) ∧
    (distribution.4 ≥ min_teachers_D) ∧
    (∃ ways : ℕ, ways = 660) :=
sorry

end distribute_teachers_l102_102344


namespace quadratic_transform_l102_102715

theorem quadratic_transform : ∀ (x : ℝ), x^2 = 3 * x + 1 ↔ x^2 - 3 * x - 1 = 0 :=
by
  sorry

end quadratic_transform_l102_102715


namespace coprime_count_15_l102_102912

theorem coprime_count_15 :
  {a : ℕ | a < 15 ∧ Nat.gcd 15 a = 1}.card = 8 :=
by
sorry

end coprime_count_15_l102_102912


namespace length_of_chord_EF_l102_102454

noncomputable def chord_length (theta_1 theta_2 : ℝ) : ℝ :=
  let x_1 := 2 * Real.cos theta_1
  let y_1 := Real.sin theta_1
  let x_2 := 2 * Real.cos theta_2
  let y_2 := Real.sin theta_2
  Real.sqrt ((x_2 - x_1)^2 + (y_2 - y_1)^2)

theorem length_of_chord_EF :
  ∀ (theta_1 theta_2 : ℝ), 
  (2 * Real.cos theta_1) + (Real.sin theta_1) + Real.sqrt 3 = 0 →
  (2 * Real.cos theta_2) + (Real.sin theta_2) + Real.sqrt 3 = 0 →
  (2 * Real.cos theta_1)^2 + 4 * (Real.sin theta_1)^2 = 4 →
  (2 * Real.cos theta_2)^2 + 4 * (Real.sin theta_2)^2 = 4 →
  chord_length theta_1 theta_2 = 8 / 5 :=
by
  intros theta_1 theta_2 h1 h2 h3 h4
  sorry

end length_of_chord_EF_l102_102454


namespace river_ratio_l102_102550

theorem river_ratio (total_length straight_length crooked_length : ℕ) 
  (h1 : total_length = 80) (h2 : straight_length = 20) 
  (h3 : crooked_length = total_length - straight_length) : 
  (straight_length / Nat.gcd straight_length crooked_length) = 1 ∧ (crooked_length / Nat.gcd straight_length crooked_length) = 3 := 
by
  sorry

end river_ratio_l102_102550


namespace sqrt_product_l102_102621

theorem sqrt_product (h1 : Real.sqrt 81 = 9) 
                     (h2 : Real.sqrt 16 = 4) 
                     (h3 : Real.sqrt (Real.sqrt (Real.sqrt 64)) = 2 * Real.sqrt 2) : 
                     Real.sqrt 81 * Real.sqrt 16 * Real.sqrt (Real.sqrt (Real.sqrt 64)) = 72 * Real.sqrt 2 :=
by
  sorry

end sqrt_product_l102_102621


namespace coats_collected_elem_schools_correct_l102_102542

-- Conditions
def total_coats_collected : ℕ := 9437
def coats_collected_high_schools : ℕ := 6922

-- Definition to find coats collected from elementary schools
def coats_collected_elementary_schools : ℕ := total_coats_collected - coats_collected_high_schools

-- Theorem statement
theorem coats_collected_elem_schools_correct : 
  coats_collected_elementary_schools = 2515 := sorry

end coats_collected_elem_schools_correct_l102_102542


namespace elevator_time_l102_102172

theorem elevator_time :
  ∀ (floors steps_per_floor steps_per_second extra_time : ℕ) (elevator_time_sec elevator_time_min : ℚ),
    floors = 8 →
    steps_per_floor = 30 →
    steps_per_second = 3 →
    extra_time = 30 →
    elevator_time_sec = ((floors * steps_per_floor) / steps_per_second) - extra_time →
    elevator_time_min = elevator_time_sec / 60 →
    elevator_time_min = 0.833 :=
by
  intros floors steps_per_floor steps_per_second extra_time elevator_time_sec elevator_time_min
  intros h_floors h_steps_per_floor h_steps_per_second h_extra_time h_elevator_time_sec h_elevator_time_min
  rw [h_floors, h_steps_per_floor, h_steps_per_second, h_extra_time] at *
  sorry

end elevator_time_l102_102172


namespace total_pies_l102_102152

-- Define the number of each type of pie.
def apple_pies : Nat := 2
def pecan_pies : Nat := 4
def pumpkin_pies : Nat := 7

-- Prove the total number of pies.
theorem total_pies : apple_pies + pecan_pies + pumpkin_pies = 13 := by
  sorry

end total_pies_l102_102152


namespace cost_price_of_apple_l102_102881

variable (CP SP: ℝ)
variable (loss: ℝ)
variable (h1: SP = 18)
variable (h2: loss = CP / 6)
variable (h3: SP = CP - loss)

theorem cost_price_of_apple : CP = 21.6 :=
by
  sorry

end cost_price_of_apple_l102_102881


namespace problem_1_problem_2_problem_3_l102_102595

-- Definition and proof state for problem 1
theorem problem_1 (a b m n : ℕ) (h₀ : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n := by
  sorry

-- Definition and proof state for problem 2
theorem problem_2 (a m n : ℕ) (h₀ : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = 13 ∨ a = 7 := by
  sorry

-- Definition and proof state for problem 3
theorem problem_3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 := by
  sorry

end problem_1_problem_2_problem_3_l102_102595


namespace first_sequence_general_term_second_sequence_general_term_l102_102472

-- For the first sequence
def first_sequence_sum : ℕ → ℚ
| n => n^2 + 1/2 * n

theorem first_sequence_general_term (n : ℕ) : 
  (first_sequence_sum (n+1) - first_sequence_sum n) = (2 * (n+1) - 1/2) := 
sorry

-- For the second sequence
def second_sequence_sum : ℕ → ℚ
| n => 1/4 * n^2 + 2/3 * n + 3

theorem second_sequence_general_term (n : ℕ) : 
  (second_sequence_sum (n+1) - second_sequence_sum n) = 
  if n = 0 then 47/12 
  else (6 * (n+1) + 5)/12 := 
sorry

end first_sequence_general_term_second_sequence_general_term_l102_102472


namespace two_roots_iff_a_greater_than_neg1_l102_102193

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l102_102193


namespace sum_of_three_different_squares_l102_102816

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def existing_list (ns : List Nat) : Prop :=
  ∀ n ∈ ns, is_perfect_square n

theorem sum_of_three_different_squares (a b c : Nat) :
  existing_list [a, b, c] →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 128 →
  false :=
by
  intros
  sorry

end sum_of_three_different_squares_l102_102816


namespace total_birds_l102_102886

-- Definitions from conditions
def num_geese : ℕ := 58
def num_ducks : ℕ := 37

-- Proof problem statement
theorem total_birds : num_geese + num_ducks = 95 := by
  sorry

end total_birds_l102_102886


namespace evaluate_f_1990_l102_102638

def binom (n k : ℕ) : ℕ := Nat.choose n k  -- Define the binomial coefficient function

theorem evaluate_f_1990 :
  let f (n : ℕ) := ∑ k in Finset.range n, (-1) ^ k * (binom n k) ^ 2
  f 1990 = -binom 1990 995 :=
by
  sorry

end evaluate_f_1990_l102_102638


namespace volume_of_wedge_l102_102030

theorem volume_of_wedge (h : 2 * Real.pi * r = 18 * Real.pi) :
  let V := (4 / 3) * Real.pi * (r ^ 3)
  let V_wedge := V / 6
  V_wedge = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l102_102030


namespace hulk_jump_distance_l102_102709

theorem hulk_jump_distance :
  ∃ n : ℕ, 3^n > 1500 ∧ ∀ m < n, 3^m ≤ 1500 := 
sorry

end hulk_jump_distance_l102_102709


namespace mean_minus_median_is_two_ninths_l102_102630

theorem mean_minus_median_is_two_ninths :
  let missed_days := [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6]
  let n := missed_days.length
  let mean_days := (missed_days.sum : ℚ) / n
  let median_days := list.nth_le (missed_days.sort) (n / 2) (by linarith) 
  mean_days - median_days = 2 / 9 := by
  sorry

end mean_minus_median_is_two_ninths_l102_102630


namespace k_value_l102_102957

noncomputable def find_k (AB BC AC BD : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_AC : AC = 5) (h_BD : BD = (12/7) * Real.sqrt 2) : ℝ :=
  12 / 7

theorem k_value (AB BC AC BD : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_AC : AC = 5) (h_BD : BD = (12/7) * Real.sqrt 2) : 
  find_k AB BC AC BD h_AB h_BC h_AC h_BD = 12 / 7 :=
by
  sorry

end k_value_l102_102957


namespace two_roots_iff_a_gt_neg1_l102_102187

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l102_102187


namespace number_of_games_in_division_l102_102583

theorem number_of_games_in_division (P Q : ℕ) (h1 : P > 2 * Q) (h2 : Q > 6) (schedule_eq : 4 * P + 5 * Q = 82) : 4 * P = 52 :=
by sorry

end number_of_games_in_division_l102_102583


namespace option_C_cannot_form_right_triangle_l102_102735

def is_right_triangle_sides (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem option_C_cannot_form_right_triangle :
  ¬ (is_right_triangle_sides 1.5 2 3) :=
by
  -- This is intentionally left incomplete as per instructions
  sorry

end option_C_cannot_form_right_triangle_l102_102735


namespace solution_range_l102_102610

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end solution_range_l102_102610


namespace asymptote_of_hyperbola_l102_102176

theorem asymptote_of_hyperbola (x y : ℝ) (h : (x^2 / 16) - (y^2 / 25) = 1) : 
  y = (5 / 4) * x :=
sorry

end asymptote_of_hyperbola_l102_102176


namespace sum_of_areas_squares_l102_102032

theorem sum_of_areas_squares (a : ℝ) : 
  (∑' n : ℕ, (a^2 / 4^n)) = (4 * a^2 / 3) :=
by
  sorry

end sum_of_areas_squares_l102_102032


namespace two_roots_iff_a_greater_than_neg1_l102_102192

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l102_102192


namespace lucas_fence_painting_l102_102663

-- Define the conditions
def total_time := 60
def time_painting := 12
def rate_per_minute := 1 / total_time

-- State the theorem
theorem lucas_fence_painting :
  let work_done := rate_per_minute * time_painting
  work_done = 1 / 5 :=
by
  -- Proof omitted
  sorry

end lucas_fence_painting_l102_102663


namespace min_value_of_seq_l102_102645

theorem min_value_of_seq 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (m a₁ : ℝ)
  (h1 : ∀ n, a n + a (n + 1) = n * (-1) ^ ((n * (n + 1)) / 2))
  (h2 : m + S 2015 = -1007)
  (h3 : a₁ * m > 0) :
  ∃ x, x = (1 / a₁) + (4 / m) ∧ x = 9 :=
by
  sorry

end min_value_of_seq_l102_102645


namespace part1_part2_l102_102381

variable (a b : ℝ)
def A : ℝ := 2 * a * b - a
def B : ℝ := -a * b + 2 * a + b

theorem part1 : 5 * A a b - 2 * B a b = 12 * a * b - 9 * a - 2 * b := by
  sorry

theorem part2 : (∀ b : ℝ, 5 * A a b - 2 * B a b = 12 * a * b - 9 * a - 2 * b) -> a = 1 / 6 := by
  sorry

end part1_part2_l102_102381


namespace blocks_for_tower_l102_102408

theorem blocks_for_tower (total_blocks : ℕ) (house_blocks : ℕ) (extra_blocks : ℕ) (tower_blocks : ℕ) 
  (h1 : total_blocks = 95) 
  (h2 : house_blocks = 20) 
  (h3 : extra_blocks = 30) 
  (h4 : tower_blocks = house_blocks + extra_blocks) : 
  tower_blocks = 50 :=
sorry

end blocks_for_tower_l102_102408


namespace sum_first_9_terms_l102_102240

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable (a_1 a_2 a_3 a_4 a_5 a_6 : ℝ)

-- Conditions
axiom h1 : a 1 + a 5 = 10
axiom h2 : a 2 + a 6 = 14

-- Calculations
axiom h3 : a 3 = 5
axiom h4 : a 4 = 7
axiom h5 : d = 2
axiom h6 : a 5 = 9

-- The sum of the first 9 terms
axiom h7 : S 9 = 9 * a 5

theorem sum_first_9_terms : S 9 = 81 :=
by {
  sorry
}

end sum_first_9_terms_l102_102240


namespace tan_angle_addition_l102_102228

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 3) : Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end tan_angle_addition_l102_102228


namespace michael_lap_time_l102_102180

theorem michael_lap_time :
  ∃ T : ℝ, (∀ D : ℝ, D = 45 → (9 * T = 10 * D) → T = 50) :=
by
  sorry

end michael_lap_time_l102_102180


namespace simplify_to_linear_form_l102_102410

theorem simplify_to_linear_form (p : ℤ) : 
  ((7 * p + 3) - 3 * p * 6) * 5 + (5 - 2 / 4) * (8 * p - 12) = -19 * p - 39 := 
by 
  sorry

end simplify_to_linear_form_l102_102410


namespace count_coprime_with_15_lt_15_l102_102909

theorem count_coprime_with_15_lt_15 :
  {a : ℕ // a < 15 ∧ Nat.coprime 15 a}.to_finset.card = 8 := 
sorry

end count_coprime_with_15_lt_15_l102_102909


namespace cos_of_angle_sum_l102_102080

variable (θ : ℝ)

-- Given condition
axiom sin_theta : Real.sin θ = 1 / 4

-- To prove
theorem cos_of_angle_sum : Real.cos (3 * Real.pi / 2 + θ) = -1 / 4 :=
by
  sorry

end cos_of_angle_sum_l102_102080


namespace base_prime_representation_450_l102_102870

-- Define prime factorization property for number 450
def prime_factorization_450 := (450 = 2^1 * 3^2 * 5^2)

-- Define base prime representation concept
def base_prime_representation (n : ℕ) : ℕ := 
  if n = 450 then 122 else 0

-- Prove that the base prime representation of 450 is 122
theorem base_prime_representation_450 : 
  prime_factorization_450 →
  base_prime_representation 450 = 122 :=
by
  intros
  sorry

end base_prime_representation_450_l102_102870


namespace markup_calculation_l102_102004

def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.25
def net_profit : ℝ := 12

def overhead := purchase_price * overhead_percentage
def total_cost := purchase_price + overhead
def selling_price := total_cost + net_profit
def markup := selling_price - purchase_price

theorem markup_calculation : markup = 24 := by
  sorry

end markup_calculation_l102_102004


namespace telephone_call_duration_l102_102573

theorem telephone_call_duration (x : ℝ) :
  (0.60 + 0.06 * (x - 4) = 0.08 * x) → x = 18 :=
by
  sorry

end telephone_call_duration_l102_102573


namespace sin_theta_value_l102_102938

theorem sin_theta_value {θ : ℝ} (h₁ : 9 * (Real.tan θ)^2 = 4 * Real.cos θ) (h₂ : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ = 1 / 3 :=
by
  sorry

end sin_theta_value_l102_102938


namespace edward_lawns_forgotten_l102_102049

theorem edward_lawns_forgotten (dollars_per_lawn : ℕ) (total_lawns : ℕ) (total_earned : ℕ) (lawns_mowed : ℕ) (lawns_forgotten : ℕ) :
  dollars_per_lawn = 4 →
  total_lawns = 17 →
  total_earned = 32 →
  lawns_mowed = total_earned / dollars_per_lawn →
  lawns_forgotten = total_lawns - lawns_mowed →
  lawns_forgotten = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end edward_lawns_forgotten_l102_102049


namespace windows_per_floor_is_3_l102_102526

-- Given conditions
variables (W : ℕ)
def windows_each_floor (W : ℕ) : Prop :=
  (3 * 2 * W) - 2 = 16

-- Correct answer
theorem windows_per_floor_is_3 : windows_each_floor 3 :=
by 
  sorry

end windows_per_floor_is_3_l102_102526


namespace find_angles_and_area_l102_102486

noncomputable def angles_in_arithmetic_progression (A B C : ℝ) : Prop :=
  A + C = 2 * B ∧ A + B + C = 180

noncomputable def side_ratios (a b : ℝ) : Prop :=
  a / b = Real.sqrt 2 / Real.sqrt 3

noncomputable def triangle_area (a b c A B C : ℝ) : ℝ :=
  (1/2) * a * c * Real.sin B

theorem find_angles_and_area :
  ∃ (A B C a b c : ℝ), 
    angles_in_arithmetic_progression A B C ∧ 
    side_ratios a b ∧ 
    c = 2 ∧ 
    A = 45 ∧ 
    B = 60 ∧ 
    C = 75 ∧ 
    triangle_area a b c A B C = 3 - Real.sqrt 3 :=
sorry

end find_angles_and_area_l102_102486


namespace smallest_positive_n_l102_102141

theorem smallest_positive_n (n : ℕ) (h : 1023 * n % 30 = 2147 * n % 30) : n = 15 :=
by
  sorry

end smallest_positive_n_l102_102141


namespace monotonic_increasing_intervals_l102_102855

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 2*x + 1)
noncomputable def f' (x : ℝ) : ℝ := Real.exp x * (x^2 + 4*x + 3)

theorem monotonic_increasing_intervals :
  ∀ x, f' x > 0 ↔ (x < -3 ∨ x > -1) :=
by
  intro x
  -- proof omitted
  sorry

end monotonic_increasing_intervals_l102_102855


namespace min_unattainable_score_l102_102000

theorem min_unattainable_score : ∀ (score : ℕ), (¬ ∃ (a b c : ℕ), 
  (a = 1 ∨ a = 3 ∨ a = 8 ∨ a = 12 ∨ a = 0) ∧ 
  (b = 1 ∨ b = 3 ∨ b = 8 ∨ b = 12 ∨ b = 0) ∧ 
  (c = 1 ∨ c = 3 ∨ c = 8 ∨ c = 12 ∨ c = 0) ∧ 
  score = a + b + c) ↔ score = 22 := 
by
  sorry

end min_unattainable_score_l102_102000


namespace roots_conditions_l102_102220

theorem roots_conditions (α β m n : ℝ) (h_pos : β > 0)
  (h1 : α + 2 * β = -m)
  (h2 : 2 * α * β + β^2 = -3)
  (h3 : α * β^2 = -n)
  (h4 : α^2 + 2 * β^2 = 6) : 
  m = 0 ∧ n = 2 := by
  sorry

end roots_conditions_l102_102220


namespace find_n_positive_integer_l102_102917

theorem find_n_positive_integer:
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, 2^n + 12^n + 2011^n = k^2) ↔ n = 1 := 
by
  sorry

end find_n_positive_integer_l102_102917


namespace simplify_tan_cot_fraction_l102_102843

theorem simplify_tan_cot_fraction :
  let tan45 := 1
  let cot45 := 1
  (tan45^3 + cot45^3) / (tan45 + cot45) = 1 := by
    sorry

end simplify_tan_cot_fraction_l102_102843


namespace symmetric_point_l102_102544

theorem symmetric_point (x y : ℝ) (hx : x = -2) (hy : y = 3) (a b : ℝ) (hne : y = x + 1)
  (halfway : (a = (x + (-2)) / 2) ∧ (b = (y + 3) / 2) ∧ (2 * b = 2 * a + 2) ∧ (2 * b = 1)):
  (a, b) = (0, 1) :=
by
  sorry

end symmetric_point_l102_102544


namespace arith_seq_100th_term_l102_102800

noncomputable def arithSeq (a : ℤ) (n : ℕ) : ℤ :=
  a - 1 + (n - 1) * ((a + 1) - (a - 1))

theorem arith_seq_100th_term (a : ℤ) : arithSeq a 100 = 197 := by
  sorry

end arith_seq_100th_term_l102_102800


namespace find_fraction_l102_102678

noncomputable def distinct_real_numbers (a b : ℝ) : Prop :=
  a ≠ b

noncomputable def equation_condition (a b : ℝ) : Prop :=
  (2 * a / (3 * b)) + ((a + 12 * b) / (3 * b + 12 * a)) = (5 / 3)

theorem find_fraction (a b : ℝ) (h1 : distinct_real_numbers a b) (h2 : equation_condition a b) : a / b = -93 / 49 :=
by
  sorry

end find_fraction_l102_102678


namespace simplify_tan_cot_expression_l102_102842

theorem simplify_tan_cot_expression
  (h1 : Real.tan (Real.pi / 4) = 1)
  (h2 : Real.cot (Real.pi / 4) = 1) :
  (Real.tan (Real.pi / 4))^3 + (Real.cot (Real.pi / 4))^3 = 1 := by
  sorry

end simplify_tan_cot_expression_l102_102842


namespace johns_climb_height_correct_l102_102675

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

end johns_climb_height_correct_l102_102675


namespace rain_probability_at_most_3_days_l102_102286

open BigOperators

def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def rain_probability := (1:ℝ)/5
noncomputable def no_rain_probability := (4:ℝ)/5

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p^k) * ((1-p)^(n-k))

theorem rain_probability_at_most_3_days :
  ∑ k in Finset.range 4, binomial_probability 31 k rain_probability = 0.544 :=
by
  sorry

end rain_probability_at_most_3_days_l102_102286


namespace division_remainder_is_7_l102_102548

theorem division_remainder_is_7 (d q D r : ℕ) (hd : d = 21) (hq : q = 14) (hD : D = 301) (h_eq : D = d * q + r) : r = 7 :=
by
  sorry

end division_remainder_is_7_l102_102548


namespace least_number_divisible_by_6_has_remainder_4_is_40_l102_102873

-- Define the least number N which leaves a remainder of 4 when divided by 6
theorem least_number_divisible_by_6_has_remainder_4_is_40 :
  ∃ (N : ℕ), (∀ (k : ℕ), N = 6 * k + 4) ∧ N = 40 := by
  sorry

end least_number_divisible_by_6_has_remainder_4_is_40_l102_102873


namespace cost_price_computer_table_l102_102882

theorem cost_price_computer_table :
  ∃ CP : ℝ, CP * 1.25 = 5600 ∧ CP = 4480 :=
by
  sorry

end cost_price_computer_table_l102_102882


namespace Rio_Coralio_Length_Estimate_l102_102008

def RioCoralioLength := 402.5
def GSA_length := 402
def AWRA_length := 403
def error_margin := 0.5
def error_probability := 0.04

theorem Rio_Coralio_Length_Estimate :
  ∀ (L_GSA L_AWRA : ℝ) (margin error_prob : ℝ),
  L_GSA = GSA_length ∧ L_AWRA = AWRA_length ∧ 
  margin = error_margin ∧ error_prob = error_probability →
  (RioCoralioLength = 402.5) ∧ (error_probability = 0.04) := 
by 
  intros L_GSA L_AWRA margin error_prob h,
  sorry

end Rio_Coralio_Length_Estimate_l102_102008


namespace one_thirds_of_nine_halfs_l102_102648

theorem one_thirds_of_nine_halfs : (9 / 2) / (1 / 3) = 27 / 2 := 
by sorry

end one_thirds_of_nine_halfs_l102_102648


namespace largest_integer_b_l102_102906

theorem largest_integer_b (b : ℤ) : (b^2 < 60) → b ≤ 7 :=
by sorry

end largest_integer_b_l102_102906


namespace store_profit_l102_102026

theorem store_profit {C : ℝ} (h₁ : C > 0) : 
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.80 * SP2
  SPF - C = 0.20 * C := 
by 
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.80 * SP2
  sorry

end store_profit_l102_102026


namespace rectangle_area_l102_102668

theorem rectangle_area (a : ℕ) (h : 2 * (3 * a + 2 * a) = 160) : 3 * a * 2 * a = 1536 :=
by
  sorry

end rectangle_area_l102_102668


namespace other_factor_of_LCM_l102_102273

-- Definitions and conditions
def A : ℕ := 624
def H : ℕ := 52 
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Hypotheses based on the problem statement
axiom h_hcf : HCF A 52 = 52

-- The desired statement to prove
theorem other_factor_of_LCM (B : ℕ) (y : ℕ) : HCF A B = H → (A * y = 624) → y = 1 := 
by 
  intro h1 h2
  -- Actual proof steps are omitted
  sorry

end other_factor_of_LCM_l102_102273


namespace eggs_left_in_box_l102_102999

theorem eggs_left_in_box (initial_eggs : ℕ) (taken_eggs : ℕ) (remaining_eggs : ℕ) : 
  initial_eggs = 47 → taken_eggs = 5 → remaining_eggs = initial_eggs - taken_eggs → remaining_eggs = 42 :=
by
  sorry

end eggs_left_in_box_l102_102999


namespace difference_between_mean_and_median_l102_102834

def percentage_students (p70 p80 p90 p100 : ℝ) : Prop :=
  p70 + p80 + p90 + p100 = 1

def median_score (p70 p80 p90 p100 : ℝ) (s70 s80 s90 s100 : ℕ) : ℕ :=
  if (p70 + p80) < 0.5 then s100
  else if (p70 < 0.5) then s90 
  else if (p70 > 0.5) then s70
  else s80

def mean_score (p70 p80 p90 p100 : ℝ) (s70 s80 s90 s100 : ℕ) : ℝ :=
  p70 * s70 + p80 * s80 + p90 * s90 + p100 * s100

theorem difference_between_mean_and_median
  (p70 p80 p90 p100 : ℝ)
  (h_sum : percentage_students p70 p80 p90 p100)
  (s70 s80 s90 s100 : ℕ) :
  median_score p70 p80 p90 p100 s70 s80 s90 s100 - mean_score p70 p80 p90 p100 s70 s80 s90 s100 = 3 :=
  sorry

end difference_between_mean_and_median_l102_102834


namespace sticks_picked_up_l102_102572

variable (original_sticks left_sticks picked_sticks : ℕ)

theorem sticks_picked_up :
  original_sticks = 99 → left_sticks = 61 → picked_sticks = original_sticks - left_sticks → picked_sticks = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sticks_picked_up_l102_102572


namespace find_y_value_l102_102795

theorem find_y_value : 
  (15^2 * 8^3) / y = 450 → y = 256 :=
by
  sorry

end find_y_value_l102_102795


namespace probability_divisible_by_4_l102_102024

def set_of_numbers := Finset.range (800 - 100 + 1) |>.map (λ n => n + 100)
def set_of_multiples_of_4 := set_of_numbers.filter (λ n => n % 4 == 0)

theorem probability_divisible_by_4 :
  (set_of_multiples_of_4.card : ℚ) / (set_of_numbers.card : ℚ) = 176 / 701 :=
by
-- We start with the range [100, 800]
have card_set_of_numbers : set_of_numbers.card = 701 :=
  by simp [set_of_numbers, Finset.range, Finset.card, Nat.sub_add_cancel (200 - 100)]

-- Calculate the number of elements divisible by 4
have card_set_of_multiples_of_4 : set_of_multiples_of_4.card = 176 :=
  by
    -- Arithmetic progression starting at 100, ending at 800 with common difference 4
    have : set_of_multiples_of_4 = Finset.filter (λ n => n % 4 == 0) (Finset.range (800 - 100 + 1) |>.map (λ n => n + 100)) :=
      by simp [set_of_multiples_of_4, set_of_numbers]
    rw this
    simp [Finset.filter_card, Finset.count, Finset.card_map, Finset.range, Nat.sub_add_cancel (800 - 100)]
    have count_multiples := (Nat.div_eq_iff_eq_mul_right.mpr (by norm_num : 4 ≠ 0)).2
    have smallest_multiple_by_n := 100 -- Smallest n / common multiple ... 
    sorry -- Calculation skipped, provided accurate steps here for guidance

-- Prove final probability
rw [card_set_of_numbers, card_set_of_multiples_of_4]
simp [Rat.div_def, Rat.mk_eq_div, Int.mul_by_distrib_left]
norm_num

end probability_divisible_by_4_l102_102024


namespace max_bag_weight_l102_102829

-- Let's define the conditions first
def green_beans_weight := 4
def milk_weight := 6
def carrots_weight := 2 * green_beans_weight
def additional_capacity := 2

-- The total weight of groceries
def total_groceries_weight := green_beans_weight + milk_weight + carrots_weight

-- The maximum weight the bag can hold is the total weight of groceries plus the additional capacity
theorem max_bag_weight : (total_groceries_weight + additional_capacity) = 20 := by
  sorry

end max_bag_weight_l102_102829


namespace technicians_count_l102_102990

-- Variables
variables (T R : ℕ)
-- Conditions from the problem
def avg_salary_all := 8000
def avg_salary_tech := 12000
def avg_salary_rest := 6000
def total_workers := 30
def total_salary := avg_salary_all * total_workers

-- Equations based on conditions
def eq1 : T + R = total_workers := sorry
def eq2 : avg_salary_tech * T + avg_salary_rest * R = total_salary := sorry

-- Proof statement (external conditions are reused for clarity)
theorem technicians_count : T = 10 :=
by sorry

end technicians_count_l102_102990


namespace jordan_purchase_total_rounded_l102_102514

theorem jordan_purchase_total_rounded :
  let p1 := 2.49
  let p2 := 6.51
  let p3 := 11.49
  let r1 := 2 -- rounded value of p1
  let r2 := 7 -- rounded value of p2
  let r3 := 11 -- rounded value of p3
  r1 + r2 + r3 = 20 :=
by
  let p1 := 2.49
  let p2 := 6.51
  let p3 := 11.49
  let r1 := 2
  let r2 := 7
  let r3 := 11
  show r1 + r2 + r3 = 20
  sorry

end jordan_purchase_total_rounded_l102_102514


namespace greatest_value_l102_102055

theorem greatest_value (x : ℝ) : -x^2 + 9 * x - 18 ≥ 0 → x ≤ 6 :=
by
  sorry

end greatest_value_l102_102055


namespace solution_for_system_l102_102937
open Real

noncomputable def solve_system (a b x y : ℝ) : Prop :=
  (a * x + b * y = 7 ∧ b * x + a * y = 8)

noncomputable def solve_linear (a b m n : ℝ) : Prop :=
  (a * (m + n) + b * (m - n) = 7 ∧ b * (m + n) + a * (m - n) = 8)

theorem solution_for_system (a b : ℝ) : solve_system a b 2 3 → solve_linear a b (5/2) (-1/2) :=
by {
  sorry
}

end solution_for_system_l102_102937


namespace lending_period_C_l102_102586

theorem lending_period_C (P_B P_C : ℝ) (R : ℝ) (T_B I_total : ℝ) (T_C_months : ℝ) :
  P_B = 5000 ∧ P_C = 3000 ∧ R = 0.10 ∧ T_B = 2 ∧ I_total = 2200 ∧ 
  T_C_months = (2 / 3) * 12 → T_C_months = 8 := by
  intros h
  sorry

end lending_period_C_l102_102586


namespace num_valid_4x4_arrays_l102_102175

open Matrix

def is_increasing_row {α : Type*} [LinearOrder α] (m : Matrix (Fin 4) (Fin 4) α) : Prop :=
  ∀ i : Fin 4, ∀ j k : Fin 4, j < k → m i j < m i k

def is_increasing_col {α : Type*} [LinearOrder α] (m : Matrix (Fin 4) (Fin 4) α) : Prop :=
  ∀ j : Fin 4, ∀ i k : Fin 4, i < k → m i j < m k j

def valid_4x4_array (m : Matrix (Fin 4) (Fin 4) ℕ) : Prop :=
  (∀ i j, 1 ≤ m i j ∧ m i j ≤ 16) ∧
  is_increasing_row m ∧ is_increasing_col m

theorem num_valid_4x4_arrays : Finset.card {m : Matrix (Fin 4) (Fin 4) ℕ // valid_4x4_array m} = 120 :=
sorry

end num_valid_4x4_arrays_l102_102175


namespace correct_average_marks_l102_102275

theorem correct_average_marks (n : ℕ) (average initial_wrong current_correct : ℕ) 
  (h_n : n = 10) 
  (h_avg : average = 100) 
  (h_wrong : initial_wrong = 60)
  (h_correct : current_correct = 10) : 
  (average * n - initial_wrong + current_correct) / n = 95 := 
by
  -- This is where the proof would go
  sorry

end correct_average_marks_l102_102275


namespace parabola_expression_correct_area_triangle_ABM_correct_l102_102805

-- Given conditions
def pointA : ℝ × ℝ := (-1, 0)
def pointB : ℝ × ℝ := (3, 0)
def pointC : ℝ × ℝ := (0, 3)

-- Analytical expression of the parabola as y = -x^2 + 2x + 3
def parabola_eqn (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Definition of the vertex M of the parabola (derived from calculations)
def vertexM : ℝ × ℝ := (1, 4)

-- Calculation of distance AB
def distance_AB : ℝ := 4

-- Calculation of area of triangle ABM
def triangle_area_ABM : ℝ := 8

theorem parabola_expression_correct :
  (∀ x y, (y = parabola_eqn x ↔ (parabola_eqn x = y))) ∧
  (parabola_eqn pointC.1 = pointC.2) :=
by
  sorry

theorem area_triangle_ABM_correct :
  (1 / 2 * distance_AB * vertexM.2 = 8) :=
by
  sorry

end parabola_expression_correct_area_triangle_ABM_correct_l102_102805


namespace money_left_after_shopping_l102_102099

def initial_amount : ℕ := 26
def cost_jumper : ℕ := 9
def cost_tshirt : ℕ := 4
def cost_heels : ℕ := 5

theorem money_left_after_shopping : initial_amount - (cost_jumper + cost_tshirt + cost_heels) = 8 :=
by
  sorry

end money_left_after_shopping_l102_102099


namespace real_solutions_of_equation_l102_102466

theorem real_solutions_of_equation :
  ∃ n : ℕ, n ≈ 59 ∧ (∀ x : ℝ, x ∈ Icc (-50) 50 → (x / 50 = Real.sin x → x ∈ real_roots_of_eq))
    where
      real_roots_of_eq := {x : ℝ | x / 50 = Real.sin x} :=
sorry

end real_solutions_of_equation_l102_102466


namespace find_x_plus_y_l102_102662

theorem find_x_plus_y :
  ∀ (x y : ℝ), (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 :=
by
  intros x y h
  sorry

end find_x_plus_y_l102_102662


namespace B_spends_85_percent_salary_l102_102019

theorem B_spends_85_percent_salary (A_s B_s : ℝ) (A_savings : ℝ) :
  A_s + B_s = 2000 →
  A_s = 1500 →
  A_savings = 0.05 * A_s →
  (B_s - (B_s * (1 - 0.05))) = A_savings →
  (1 - 0.85) * B_s = 0.15 * B_s := 
by
  intros h1 h2 h3 h4
  sorry

end B_spends_85_percent_salary_l102_102019


namespace Toby_change_l102_102134

def change (orders_cost per_person total_cost given_amount : ℝ) : ℝ :=
  given_amount - per_person

def total_cost (cheeseburgers milkshake coke fries cookies tax : ℝ) : ℝ :=
  cheeseburgers + milkshake + coke + fries + cookies + tax

theorem Toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2.0
  let coke_cost := 1.0
  let fries_cost := 4.0
  let cookie_cost := 3 * 0.5 -- Total cost for three cookies
  let tax := 0.2
  let total := total_cost (2 * cheeseburger_cost) milkshake_cost coke_cost fries_cost cookie_cost tax
  let per_person := total / 2
  let toby_arrival := 15.0
  change total per_person total toby_arrival = 7 :=
by
  sorry

end Toby_change_l102_102134


namespace red_trace_larger_sphere_area_l102_102322

-- Defining the parameters and the given conditions
variables {R1 R2 : ℝ} (A1 : ℝ) (A2 : ℝ)
def smaller_sphere_radius := 4
def larger_sphere_radius := 6
def red_trace_smaller_sphere_area := 37

theorem red_trace_larger_sphere_area :
  R1 = smaller_sphere_radius → R2 = larger_sphere_radius → 
  A1 = red_trace_smaller_sphere_area → 
  A2 = A1 * (R2 / R1) ^ 2 → 
  A2 = 83.25 := 
  by
  intros hR1 hR2 hA1 hA2
  -- Use the given values and solve the assertion
  sorry

end red_trace_larger_sphere_area_l102_102322


namespace multiple_of_5_l102_102082

theorem multiple_of_5 (a : ℤ) (h : ¬ (5 ∣ a)) : 5 ∣ (a^12 - 1) :=
by
  sorry

end multiple_of_5_l102_102082


namespace unattainable_value_of_y_l102_102059

noncomputable def f (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

theorem unattainable_value_of_y :
  ∃ y : ℝ, y = -(1 / 3) ∧ ∀ x : ℝ, 3 * x + 4 ≠ 0 → f x ≠ y :=
by
  sorry

end unattainable_value_of_y_l102_102059


namespace image_length_interval_two_at_least_four_l102_102028

noncomputable def quadratic_function (p q r : ℝ) : ℝ → ℝ :=
  fun x => p * (x - q)^2 + r

theorem image_length_interval_two_at_least_four (p q r : ℝ)
  (h : ∀ I : Set ℝ, (∀ a b : ℝ, I = Set.Icc a b ∨ I = Set.Ioo a b → |b - a| = 1 → |quadratic_function p q r b - quadratic_function p q r a| ≥ 1)) :
  ∀ I' : Set ℝ, (∀ a b : ℝ, I' = Set.Icc a b ∨ I' = Set.Ioo a b → |b - a| = 2 → |quadratic_function p q r b - quadratic_function p q r a| ≥ 4) :=
by
  sorry


end image_length_interval_two_at_least_four_l102_102028


namespace midpoint_AB_l102_102094

noncomputable def s (x t : ℝ) : ℝ := (x + t)^2 + (x - t)^2

noncomputable def CP (x : ℝ) : ℝ := x * Real.sqrt 3 / 2

theorem midpoint_AB (x : ℝ) (P : ℝ) : 
    (s x 0 = 2 * CP x ^ 2) ↔ P = x :=
by
    sorry

end midpoint_AB_l102_102094


namespace area_of_cross_l102_102545

-- Definitions based on the conditions
def congruent_squares (n : ℕ) := n = 5
def perimeter_of_cross (p : ℕ) := p = 72

-- Targeting the proof that the area of the cross formed by the squares is 180 square units
theorem area_of_cross (n p : ℕ) (h1 : congruent_squares n) (h2 : perimeter_of_cross p) : 
  5 * (p / 12) ^ 2 = 180 := 
by 
  sorry

end area_of_cross_l102_102545


namespace solution_l102_102578

noncomputable def inequality_prove (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : Prop :=
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5)

noncomputable def equality_condition (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : Prop :=
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5) ↔ (x = 2 ∧ y = 2)

theorem solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : 
  inequality_prove x y h1 h2 h3 ∧ equality_condition x y h1 h2 h3 := by
  sorry

end solution_l102_102578


namespace principal_amount_l102_102794

theorem principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) : 
  A = 1120 → r = 0.05 → t = 6 → P = 1120 / (1 + 0.05 * 6) :=
by
  intros h1 h2 h3
  sorry

end principal_amount_l102_102794


namespace total_race_distance_l102_102840

theorem total_race_distance :
  let sadie_time := 2
  let sadie_speed := 3
  let ariana_time := 0.5
  let ariana_speed := 6
  let total_time := 4.5
  let sarah_speed := 4
  let sarah_time := total_time - sadie_time - ariana_time
  let sadie_distance := sadie_speed * sadie_time
  let ariana_distance := ariana_speed * ariana_time
  let sarah_distance := sarah_speed * sarah_time
  let total_distance := sadie_distance + ariana_distance + sarah_distance
  total_distance = 17 :=
by
  sorry

end total_race_distance_l102_102840


namespace union_complement_eq_l102_102691

def U := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 5}

def complement (univ : Set ℕ) (s : Set ℕ) : Set ℕ :=
  {x ∈ univ | x ∉ s}

theorem union_complement_eq :
  N ∪ (complement U M) = {2, 3, 5} :=
by sorry

end union_complement_eq_l102_102691


namespace opposite_of_neg_five_halves_l102_102717

theorem opposite_of_neg_five_halves : -(- (5 / 2: ℝ)) = 5 / 2 :=
by
    sorry

end opposite_of_neg_five_halves_l102_102717


namespace randy_quiz_score_l102_102268

theorem randy_quiz_score (q1 q2 q3 q5 : ℕ) (q4 : ℕ) :
  q1 = 90 → q2 = 98 → q3 = 94 → q5 = 96 → (q1 + q2 + q3 + q4 + q5) / 5 = 94 → q4 = 92 :=
by
  intros h1 h2 h3 h5 h_avg
  sorry

end randy_quiz_score_l102_102268


namespace find_general_formula_l102_102257

theorem find_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h₀ : n > 0)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, S (n + 1) = 2 * S n + n + 1)
  (h₃ : ∀ n, S (n + 1) - S n = a (n + 1)) :
  a n = 2^n - 1 :=
sorry

end find_general_formula_l102_102257


namespace area_relation_l102_102580

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  0.5 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_relation (A B C A' B' C' : ℝ × ℝ) (hAA'BB'CC'parallel: 
  ∃ k : ℝ, (A'.1 - A.1 = k * (B'.1 - B.1)) ∧ (A'.2 - A.2 = k * (B'.2 - B.2)) ∧ 
           (B'.1 - B.1 = k * (C'.1 - C.1)) ∧ (B'.2 - B.2 = k * (C'.2 - C.2))) :
  3 * (area_triangle A B C + area_triangle A' B' C') = 
    area_triangle A B' C' + area_triangle B C' A' + area_triangle C A' B' +
    area_triangle A' B C + area_triangle B' C A + area_triangle C' A B := 
sorry

end area_relation_l102_102580


namespace closest_to_fraction_l102_102787

theorem closest_to_fraction (n d : ℝ) (h_n : n = 510) (h_d : d = 0.125) :
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 5000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 6000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 7000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 8000 :=
by
  sorry

end closest_to_fraction_l102_102787


namespace total_flying_days_l102_102959

-- Definitions for the conditions
def days_fly_south_winter := 40
def days_fly_north_summer := 2 * days_fly_south_winter
def days_fly_east_spring := 60

-- Theorem stating the total flying days
theorem total_flying_days : 
  days_fly_south_winter + days_fly_north_summer + days_fly_east_spring = 180 :=
  by {
    -- This is where we would prove the theorem
    sorry
  }

end total_flying_days_l102_102959


namespace value_of_y_when_x_is_neg2_l102_102241

theorem value_of_y_when_x_is_neg2 :
  ∃ (k b : ℝ), (k + b = 2) ∧ (-k + b = -4) ∧ (∀ x, y = k * x + b) ∧ (x = -2) → (y = -7) := 
sorry

end value_of_y_when_x_is_neg2_l102_102241


namespace probability_snow_at_least_once_l102_102531

-- Define the probabilities given in the conditions
def p_day_1_3 : ℚ := 1 / 3
def p_day_4_7 : ℚ := 1 / 4
def p_day_8_10 : ℚ := 1 / 2

-- Define the complementary no-snow probabilities
def p_no_snow_day_1_3 : ℚ := 2 / 3
def p_no_snow_day_4_7 : ℚ := 3 / 4
def p_no_snow_day_8_10 : ℚ := 1 / 2

-- Compute the total probability of no snow for all ten days
def p_no_snow_all_days : ℚ :=
  (p_no_snow_day_1_3 ^ 3) * (p_no_snow_day_4_7 ^ 4) * (p_no_snow_day_8_10 ^ 3)

-- Define the proof problem: Calculate probability of at least one snow day
theorem probability_snow_at_least_once : (1 - p_no_snow_all_days) = 2277 / 2304 := by
  sorry

end probability_snow_at_least_once_l102_102531


namespace large_diagonal_proof_l102_102125

variable (a b : ℝ) (α : ℝ)
variable (h₁ : a < b)
variable (h₂ : 1 < a) -- arbitrary positive scalar to make obtuse properties hold

noncomputable def large_diagonal_length : ℝ :=
  Real.sqrt (a^2 + b^2 + 2 * b * (Real.cos α * Real.sqrt (a^2 - b^2 * Real.sin α^2) + b * Real.sin α^2))

theorem large_diagonal_proof
  (h₃ : 90 < α + Real.arcsin (b * Real.sin α / a)) :
  large_diagonal_length a b α = Real.sqrt (a^2 + b^2 + 2 * b * (Real.cos α * Real.sqrt (a^2 - b^2 * Real.sin α^2) + b * Real.sin α^2)) :=
sorry

end large_diagonal_proof_l102_102125


namespace significant_figures_and_precision_l102_102714

-- Definition of the function to count significant figures
def significant_figures (n : Float) : Nat :=
  -- Implementation of a function that counts significant figures
  -- Skipping actual implementation, assuming it is correct.
  sorry

-- Definition of the function to determine precision
def precision (n : Float) : String :=
  -- Implementation of a function that returns the precision
  -- Skipping actual implementation, assuming it is correct.
  sorry

-- The target number
def num := 0.03020

-- The properties of the number 0.03020
theorem significant_figures_and_precision :
  significant_figures num = 4 ∧ precision num = "ten-thousandth" :=
by
  sorry

end significant_figures_and_precision_l102_102714


namespace real_solutions_eq_31_l102_102467

noncomputable def number_of_real_solutions : ℕ :=
  let zero := 0
  let fifty := 50
  let neg_fifty := -50
  let num_intervals := 8
  let num_solutions_per_interval := 2
  let total_solutions := num_intervals * num_solutions_per_interval * 2 - 1
  total_solutions

theorem real_solutions_eq_31 : number_of_real_solutions = 31 := by
  sorry

end real_solutions_eq_31_l102_102467


namespace determine_k_a_l102_102785

theorem determine_k_a (k a : ℝ) (h : k - a ≠ 0) : (k = 0 ∧ a = 1 / 2) ↔ 
  (∀ x : ℝ, (x + 2) / (kx - ax - 1) = x → x = -2) :=
by
  sorry

end determine_k_a_l102_102785


namespace two_roots_iff_a_gt_neg1_l102_102191

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l102_102191


namespace perpendicular_bisector_eq_l102_102078

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 2)

-- Theorem stating that the perpendicular bisector has the specified equation
theorem perpendicular_bisector_eq : ∀ (x y : ℝ), (y = -2 * x + 3) ↔ ∃ (a b : ℝ), (a, b) = A ∨ (a, b) = B ∧ (y = -2 * x + 3) :=
by
  sorry

end perpendicular_bisector_eq_l102_102078


namespace bicycle_parking_income_l102_102034

theorem bicycle_parking_income (x : ℝ) (y : ℝ) 
    (h1 : 0 ≤ x ∧ x ≤ 2000)
    (h2 : y = 0.5 * x + 0.8 * (2000 - x)) : 
    y = -0.3 * x + 1600 := by
  sorry

end bicycle_parking_income_l102_102034


namespace original_decimal_number_l102_102302

theorem original_decimal_number (x : ℝ) (h₁ : 0 < x) (h₂ : 100 * x = 9 * (1 / x)) : x = 3 / 10 :=
by
  sorry

end original_decimal_number_l102_102302


namespace simplify_and_evaluate_l102_102411

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 + 1) : 
  ( ( (2 * x + 1) / x - 1 ) / ( (x^2 - 1) / x ) ) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l102_102411


namespace parabola_point_ordinate_l102_102361

-- The definition of the problem as a Lean 4 statement
theorem parabola_point_ordinate (a : ℝ) (x₀ y₀ : ℝ) 
  (h₀ : 0 < a)
  (h₁ : x₀^2 = (1 / a) * y₀)
  (h₂ : dist (0, 1 / (4 * a)) (0, -1 / (4 * a)) = 1)
  (h₃ : dist (x₀, y₀) (0, 1 / (4 * a)) = 5) :
  y₀ = 9 / 2 := 
sorry

end parabola_point_ordinate_l102_102361


namespace project_inflation_cost_increase_l102_102585

theorem project_inflation_cost_increase :
  let original_lumber_cost := 450
  let original_nails_cost := 30
  let original_fabric_cost := 80
  let lumber_inflation := 0.2
  let nails_inflation := 0.1
  let fabric_inflation := 0.05
  
  let new_lumber_cost := original_lumber_cost * (1 + lumber_inflation)
  let new_nails_cost := original_nails_cost * (1 + nails_inflation)
  let new_fabric_cost := original_fabric_cost * (1 + fabric_inflation)
  
  let total_increased_cost := (new_lumber_cost - original_lumber_cost) 
                            + (new_nails_cost - original_nails_cost) 
                            + (new_fabric_cost - original_fabric_cost)
  total_increased_cost = 97 := sorry

end project_inflation_cost_increase_l102_102585


namespace quadratic_function_coefficient_nonzero_l102_102234

theorem quadratic_function_coefficient_nonzero (m : ℝ) :
  (y = (m + 2) * x * x + m) ↔ (m ≠ -2 ∧ (m^2 + m - 2 = 0) → m = 1) := by
  sorry

end quadratic_function_coefficient_nonzero_l102_102234


namespace geometric_sequence_k_value_l102_102395

theorem geometric_sequence_k_value
  (k : ℤ)
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (h1 : ∀ n, S n = 3 * 2^n + k)
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n - 1))
  (h3 : ∃ r, ∀ n, a (n + 1) = r * a n) : k = -3 :=
sorry

end geometric_sequence_k_value_l102_102395


namespace low_degree_polys_condition_l102_102354

theorem low_degree_polys_condition :
  ∃ (f : Polynomial ℤ), ∃ (g : Polynomial ℤ), ∃ (h : Polynomial ℤ),
    (f = Polynomial.X ^ 3 + Polynomial.X ^ 2 + Polynomial.X + 1 ∨
          f = Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + 2 * Polynomial.X + 2 ∨
          f = 2 * Polynomial.X ^ 3 + Polynomial.X ^ 2 + 2 * Polynomial.X + 1 ∨
          f = 2 * Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + Polynomial.X + 2) ∧
          f ^ 4 + 2 * f + 2 = (Polynomial.X ^ 4 + 2 * Polynomial.X ^ 2 + 2) * g + 3 * h := 
sorry

end low_degree_polys_condition_l102_102354


namespace find_divisor_l102_102574

theorem find_divisor (D N : ℕ) (k l : ℤ)
  (h1 : N % D = 255)
  (h2 : (2 * N) % D = 112) :
  D = 398 := by
  -- Proof here
  sorry

end find_divisor_l102_102574


namespace price_per_book_sold_l102_102153

-- Definitions based on the given conditions
def total_books_before_sale : ℕ := 3 * 50
def books_sold : ℕ := 2 * 50
def total_amount_received : ℕ := 500

-- Target statement to be proved
theorem price_per_book_sold :
  (total_amount_received : ℚ) / books_sold = 5 :=
sorry

end price_per_book_sold_l102_102153


namespace total_amount_l102_102052

-- Definitions based on the problem conditions
def jack_amount : ℕ := 26
def ben_amount : ℕ := jack_amount - 9
def eric_amount : ℕ := ben_amount - 10

-- Proof statement
theorem total_amount : jack_amount + ben_amount + eric_amount = 50 :=
by
  -- Sorry serves as a placeholder for the actual proof
  sorry

end total_amount_l102_102052


namespace zero_a_if_square_every_n_l102_102765

theorem zero_a_if_square_every_n (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 := 
sorry

end zero_a_if_square_every_n_l102_102765


namespace trigonometric_inequality_equality_conditions_l102_102360

theorem trigonometric_inequality
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2)) ≥ 9 :=
sorry

theorem equality_conditions
  (α β : ℝ)
  (hα : α = Real.arctan (Real.sqrt 2))
  (hβ : β = π / 4) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2)) = 9 :=
sorry

end trigonometric_inequality_equality_conditions_l102_102360


namespace no_such_function_exists_l102_102533

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ m n : ℕ, (m + f n)^2 ≥ 3 * (f m)^2 + n^2 :=
by 
  sorry

end no_such_function_exists_l102_102533


namespace inequality_holds_for_real_numbers_l102_102741

theorem inequality_holds_for_real_numbers (a1 a2 a3 a4 : ℝ) (h1 : 1 < a1) 
  (h2 : 1 < a2) (h3 : 1 < a3) (h4 : 1 < a4) : 
  8 * (a1 * a2 * a3 * a4 + 1) ≥ (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) :=
by sorry

end inequality_holds_for_real_numbers_l102_102741


namespace local_maximum_at_negative_one_l102_102802

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * a * x + 2

theorem local_maximum_at_negative_one
  (a : ℝ)
  (h_min : ∀ f' : ℝ → ℝ, (f' = λ x, 3 * x^2 - 3 * a) → f' 1 = 0) :
  f a (-1) = 4 :=
by
  -- Definitions and hypotheses are in place; proof is omitted.
  sorry

end local_maximum_at_negative_one_l102_102802


namespace river_length_GSA_AWRA_l102_102006

-- Define the main problem statement
noncomputable def river_length_estimate (GSA_length AWRA_length GSA_error AWRA_error error_prob : ℝ) : Prop :=
  (GSA_length = 402) ∧ (AWRA_length = 403) ∧ 
  (GSA_error = 0.5) ∧ (AWRA_error = 0.5) ∧ 
  (error_prob = 0.04) ∧ 
  (abs (402.5 - GSA_length) ≤ GSA_error) ∧ 
  (abs (402.5 - AWRA_length) ≤ AWRA_error) ∧ 
  (error_prob = 1 - (2 * 0.02))

-- The main theorem statement
theorem river_length_GSA_AWRA :
  river_length_estimate 402 403 0.5 0.5 0.04 :=
by
  sorry

end river_length_GSA_AWRA_l102_102006


namespace find_trousers_l102_102388

variables (S T Ti : ℝ) -- Prices of shirt, trousers, and tie respectively
variables (x : ℝ)      -- The number of trousers in the first scenario

-- Conditions given in the problem
def condition1 : Prop := 6 * S + x * T + 2 * Ti = 80
def condition2 : Prop := 4 * S + 2 * T + 2 * Ti = 140
def condition3 : Prop := 5 * S + 3 * T + 2 * Ti = 110

-- Theorem to prove
theorem find_trousers : condition1 S T Ti x ∧ condition2 S T Ti ∧ condition3 S T Ti → x = 4 :=
by
  sorry

end find_trousers_l102_102388


namespace polynomial_remainder_division_l102_102210

theorem polynomial_remainder_division (x : ℝ) : 
  (x^4 + 1) % (x^2 - 4 * x + 6) = 16 * x - 59 := 
sorry

end polynomial_remainder_division_l102_102210


namespace union_complement_eq_l102_102688

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l102_102688


namespace speeding_tickets_l102_102102

theorem speeding_tickets (p1 p2 : ℝ)
  (h1 : p1 = 16.666666666666664)
  (h2 : p2 = 40) :
  (p1 * (100 - p2) / 100 = 10) :=
by sorry

end speeding_tickets_l102_102102


namespace find_rho_squared_l102_102256

theorem find_rho_squared:
  ∀ (a b : ℝ), (0 < a) → (0 < b) →
  (a^2 - 2 * b^2 = 0) →
  (∃ (x y : ℝ), 
    (0 ≤ x ∧ x < a) ∧ 
    (0 ≤ y ∧ y < b) ∧ 
    (a^2 + y^2 = b^2 + x^2) ∧ 
    ((a - x)^2 + (b - y)^2 = b^2 + x^2) ∧ 
    (x^2 + y^2 = b^2)) → 
  (∃ (ρ : ℝ), ρ = a / b ∧ ρ^2 = 2) :=
by
  intros a b ha hb hab hsol
  sorry  -- Proof to be provided later

end find_rho_squared_l102_102256


namespace increased_area_l102_102718

variable (r : ℝ)

theorem increased_area (r : ℝ) : 
  let initial_area : ℝ := π * r^2
  let final_area : ℝ := π * (r + 3)^2
  final_area - initial_area = 6 * π * r + 9 * π := by
sorry

end increased_area_l102_102718


namespace prime_ge_7_div_30_l102_102494

theorem prime_ge_7_div_30 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
sorry

end prime_ge_7_div_30_l102_102494


namespace find_x_l102_102877

theorem find_x (x : ℝ) (h : (3 * x) / 4 = 24) : x = 32 :=
by
  sorry

end find_x_l102_102877


namespace intersect_A_B_l102_102370

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -1 < x ∧ x ≤ 1}

theorem intersect_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersect_A_B_l102_102370


namespace february_five_sundays_in_twenty_first_century_l102_102262

/-- 
  Define a function to check if a year is a leap year
-/
def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

/-- 
  Define the specific condition for the problem: 
  Given a year, whether February 1st for that year is a Sunday
-/
def february_first_is_sunday (year : ℕ) : Prop :=
  -- This is a placeholder logic. In real applications, you would
  -- calculate the exact weekday of February 1st for the provided year.
  sorry

/-- 
  The list of years in the 21st century where February has 5 Sundays is 
  exactly {2004, 2032, 2060, and 2088}.
-/
theorem february_five_sundays_in_twenty_first_century :
  {year : ℕ | is_leap_year year ∧ february_first_is_sunday year ∧ (2001 ≤ year ∧ year ≤ 2100)} =
  {2004, 2032, 2060, 2088} := sorry

end february_five_sundays_in_twenty_first_century_l102_102262


namespace part1_min_value_of_f_when_a_is_1_part2_range_of_a_for_f_ge_x_l102_102221

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - Real.log x

theorem part1_min_value_of_f_when_a_is_1 : 
  (∃ x : ℝ, f 1 x = 1 / 2 ∧ (∀ y : ℝ, f 1 y ≥ f 1 x)) :=
sorry

theorem part2_range_of_a_for_f_ge_x :
  (∀ x : ℝ, x > 0 → f a x ≥ x) ↔ a ≥ 2 :=
sorry

end part1_min_value_of_f_when_a_is_1_part2_range_of_a_for_f_ge_x_l102_102221


namespace abc_inequality_l102_102255

-- Required conditions and proof statement
theorem abc_inequality 
  {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c = 1 / 8) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + a^2 * c^2 + b^2 * c^2 ≥ 15 / 16 := 
sorry

end abc_inequality_l102_102255


namespace problem_statement_l102_102696

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l102_102696


namespace rationalize_denominator_theorem_l102_102704

noncomputable def rationalize_denominator : Prop :=
  let num := 5
  let den := 2 + Real.sqrt 5
  let conj := 2 - Real.sqrt 5
  let expr := (num * conj) / (den * conj)
  expr = -10 + 5 * Real.sqrt 5

theorem rationalize_denominator_theorem : rationalize_denominator :=
  sorry

end rationalize_denominator_theorem_l102_102704


namespace percentage_vanaspati_after_adding_ghee_l102_102671

theorem percentage_vanaspati_after_adding_ghee :
  ∀ (original_quantity new_pure_ghee percentage_ghee percentage_vanaspati : ℝ),
    original_quantity = 30 →
    percentage_ghee = 0.5 →
    percentage_vanaspati = 0.5 →
    new_pure_ghee = 20 →
    (percentage_vanaspati * original_quantity) /
    (original_quantity + new_pure_ghee) * 100 = 30 :=
by
  intros original_quantity new_pure_ghee percentage_ghee percentage_vanaspati
  sorry

end percentage_vanaspati_after_adding_ghee_l102_102671


namespace sum_of_cubes_eq_91_l102_102126

theorem sum_of_cubes_eq_91 (a b : ℤ) (h₁ : a^3 + b^3 = 91) (h₂ : a * b = 12) : a^3 + b^3 = 91 :=
by
  exact h₁

end sum_of_cubes_eq_91_l102_102126


namespace two_roots_iff_a_greater_than_neg1_l102_102196

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l102_102196


namespace basketball_scores_l102_102150

theorem basketball_scores : ∃ (scores : Finset ℕ), 
  scores = { x | ∃ a b : ℕ, a + b = 7 ∧ x = 2 * a + 3 * b } ∧ scores.card = 8 :=
by
  sorry

end basketball_scores_l102_102150


namespace Megan_total_earnings_two_months_l102_102979

-- Define the conditions
def hours_per_day : ℕ := 8
def wage_per_hour : ℝ := 7.50
def days_per_month : ℕ := 20

-- Define the main question and correct answer
theorem Megan_total_earnings_two_months : 
  (2 * (days_per_month * (hours_per_day * wage_per_hour))) = 2400 := 
by
  -- In the problem statement, we are given conditions so we just state sorry because the focus is on the statement, not the solution steps.
  sorry

end Megan_total_earnings_two_months_l102_102979


namespace number_of_questions_in_test_l102_102862

theorem number_of_questions_in_test (x : ℕ) (sections questions_correct : ℕ)
  (h_sections : sections = 5)
  (h_questions_correct : questions_correct = 32)
  (h_percentage : 0.70 < (questions_correct : ℚ) / x ∧ (questions_correct : ℚ) / x < 0.77) 
  (h_multiple_of_sections : x % sections = 0) : 
  x = 45 :=
sorry

end number_of_questions_in_test_l102_102862


namespace roots_eq_two_iff_a_gt_neg1_l102_102201

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l102_102201


namespace squares_triangles_product_l102_102546

theorem squares_triangles_product :
  let S := 7
  let T := 10
  S * T = 70 :=
by
  let S := 7
  let T := 10
  show (S * T = 70)
  sorry

end squares_triangles_product_l102_102546


namespace time_walking_each_day_l102_102975

variable (days : Finset ℕ) (d1 : ℕ) (d2 : ℕ) (W : ℕ)

def time_spent_parking (days : Finset ℕ) : ℕ :=
  5 * days.card

def time_spent_metal_detector : ℕ :=
  2 * 30 + 3 * 10

def total_timespent (d1 d2 W : ℕ) : ℕ :=
  d1 + d2 + W

theorem time_walking_each_day (total_minutes : ℕ) (total_days : ℕ):
  total_timespent (time_spent_parking days) (time_spent_metal_detector) (total_minutes - time_spent_metal_detector - 5 * total_days)
  = total_minutes → W = 3 := by
  sorry

end time_walking_each_day_l102_102975


namespace inequality_proof_l102_102109

variable (a : ℝ)

theorem inequality_proof (a : ℝ) : 
  (a^2 + a + 2) / (Real.sqrt (a^2 + a + 1)) ≥ 2 :=
sorry

end inequality_proof_l102_102109


namespace find_number_l102_102570

theorem find_number (x : ℝ) : 14 * x + 15 * x + 18 * x + 11 = 152 → x = 3 := by
  sorry

end find_number_l102_102570


namespace algebraic_expression_correct_l102_102846

theorem algebraic_expression_correct (x y : ℝ) :
  (x - y)^2 - (x^2 - y^2) = (x - y)^2 - (x^2 - y^2) :=
by
  sorry

end algebraic_expression_correct_l102_102846


namespace calvin_wins_l102_102402

-- Definitions based on the conditions
variables 
  (k N : ℕ)
  (h_k : k ≥ 1)
  (h_N : N > 1)

-- 2N + 1 coins on a circle, all initially showing heads
def coins := list.repeat tt (2 * N + 1)

-- Calvin can turn any coin from heads to tails, Hobbes can turn at most one adjacent coin from tails to heads
-- Calvin wins if at any moment there are k coins showing tails after Hobbes has made his move
theorem calvin_wins : k ≤ N + 1 :=
by
  sorry

end calvin_wins_l102_102402


namespace triangle_area_ratio_l102_102666

noncomputable def area_ratio (AD DC : ℝ) (h : ℝ) : ℝ :=
  (1 / 2) * AD * h / ((1 / 2) * DC * h)

theorem triangle_area_ratio (AD DC : ℝ) (h : ℝ) (condition1 : AD = 5) (condition2 : DC = 7) :
  area_ratio AD DC h = 5 / 7 :=
by
  sorry

end triangle_area_ratio_l102_102666


namespace range_of_m_l102_102119

theorem range_of_m (m : ℝ) : (-1 : ℝ) ≤ m ∧ m ≤ 3 ∧ ∀ x y : ℝ, x - ((m^2) - 2 * m + 4) * y - 6 > 0 → (x, y) ≠ (-1, -1) := 
by sorry

end range_of_m_l102_102119


namespace cylinder_inscribed_in_sphere_l102_102031

noncomputable def sphere_volume (r : ℝ) : ℝ := 
  (4 / 3) * Real.pi * r^3

theorem cylinder_inscribed_in_sphere 
  (r_cylinder : ℝ)
  (h₁ : r_cylinder > 0)
  (height_cylinder : ℝ)
  (radius_sphere : ℝ)
  (h₂ : radius_sphere = r_cylinder + 2)
  (h₃ : height_cylinder = r_cylinder + 1)
  (h₄ : 2 * radius_sphere = Real.sqrt ((2 * r_cylinder)^2 + (height_cylinder)^2))
  : sphere_volume 17 = 6550 * 2 / 3 * Real.pi :=
by
  -- solution steps and proof go here
  sorry

end cylinder_inscribed_in_sphere_l102_102031


namespace find_C_coordinates_l102_102820

variables {A B M L C : ℝ × ℝ}

def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def on_line_bisector (L B : ℝ × ℝ) : Prop :=
  B.1 = 6  -- Vertical line through B

theorem find_C_coordinates
  (A := (2, 8))
  (M := (4, 11))
  (L := (6, 6))
  (hM : is_midpoint M A B)
  (hL : on_line_bisector L B) :
  C = (6, 14) :=
sorry

end find_C_coordinates_l102_102820


namespace one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero_l102_102081

theorem one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a < 1 / b) ↔ ((a * b) / (a^3 - b^3) > 0) := 
by
  sorry

end one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero_l102_102081


namespace average_speed_second_day_l102_102769

theorem average_speed_second_day
  (t v : ℤ)
  (h1 : 2 * t + 2 = 18)
  (h2 : (v + 5) * (t + 2) + v * t = 680) :
  v = 35 :=
by
  sorry

end average_speed_second_day_l102_102769


namespace total_instruments_correct_l102_102261

def fingers : Nat := 10
def hands : Nat := 2
def heads : Nat := 1

def trumpets := fingers - 3
def guitars := hands + 2
def trombones := heads + 2
def french_horns := guitars - 1
def violins := trumpets / 2
def saxophones := trombones / 3

theorem total_instruments_correct : 
  (trumpets + guitars = trombones + violins + saxophones) →
  trumpets + guitars + trombones + french_horns + violins + saxophones = 21 := by
  sorry

end total_instruments_correct_l102_102261


namespace triangle_area_l102_102457

noncomputable def area_triangle_ACD (t p : ℝ) : ℝ :=
  1 / 2 * p * (t - 2)

theorem triangle_area (t p : ℝ) (ht : 0 < t ∧ t < 12) (hp : 0 < p ∧ p < 12) :
  area_triangle_ACD t p = 1 / 2 * p * (t - 2) :=
sorry

end triangle_area_l102_102457


namespace factor_expression_l102_102771

variables (b : ℝ)

theorem factor_expression :
  (8 * b ^ 3 + 45 * b ^ 2 - 10) - (-12 * b ^ 3 + 5 * b ^ 2 - 10) = 20 * b ^ 2 * (b + 2) :=
by
  sorry

end factor_expression_l102_102771


namespace wire_not_used_is_20_l102_102348

def initial_wire_length : ℕ := 50
def number_of_parts : ℕ := 5
def parts_used : ℕ := 3

def length_of_each_part (total_length : ℕ) (parts : ℕ) : ℕ := total_length / parts
def length_used (length_each_part : ℕ) (used_parts : ℕ) : ℕ := length_each_part * used_parts
def wire_not_used (total_length : ℕ) (used_length : ℕ) : ℕ := total_length - used_length

theorem wire_not_used_is_20 : 
  wire_not_used initial_wire_length 
    (length_used 
      (length_of_each_part initial_wire_length number_of_parts) 
    parts_used) = 20 := by
  sorry

end wire_not_used_is_20_l102_102348


namespace river_length_GSA_AWRA_l102_102007

-- Define the main problem statement
noncomputable def river_length_estimate (GSA_length AWRA_length GSA_error AWRA_error error_prob : ℝ) : Prop :=
  (GSA_length = 402) ∧ (AWRA_length = 403) ∧ 
  (GSA_error = 0.5) ∧ (AWRA_error = 0.5) ∧ 
  (error_prob = 0.04) ∧ 
  (abs (402.5 - GSA_length) ≤ GSA_error) ∧ 
  (abs (402.5 - AWRA_length) ≤ AWRA_error) ∧ 
  (error_prob = 1 - (2 * 0.02))

-- The main theorem statement
theorem river_length_GSA_AWRA :
  river_length_estimate 402 403 0.5 0.5 0.04 :=
by
  sorry

end river_length_GSA_AWRA_l102_102007


namespace sum_of_first_n_terms_l102_102636

variable (a : ℕ → ℤ) (b : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Given conditions
axiom a_n_arith : ∀ n, a (n + 1) - a n = a 2 - a 1
axiom a_3 : a 3 = -6
axiom a_6 : a 6 = 0
axiom b_1 : b 1 = -8
axiom b_2 : b 2 = a 1 + a 2 + a 3

-- Correct answer to prove
theorem sum_of_first_n_terms : S n = 4 * (1 - 3^n) := sorry

end sum_of_first_n_terms_l102_102636


namespace megan_total_earnings_l102_102978

-- Define the constants
def work_hours_per_day := 8
def earnings_per_hour := 7.50
def work_days_per_month := 20

-- Define Megan's total earnings for two months
def total_earnings (work_hours_per_day : ℕ) (earnings_per_hour : ℝ) (work_days_per_month : ℕ) : ℝ :=
  2 * (work_hours_per_day * earnings_per_hour * work_days_per_month)

-- Prove that the total earnings for two months is $2400
theorem megan_total_earnings : total_earnings work_hours_per_day earnings_per_hour work_days_per_month = 2400 :=
by
  sorry

end megan_total_earnings_l102_102978


namespace smallest_share_arith_seq_l102_102848

theorem smallest_share_arith_seq (a1 d : ℚ) (h1 : 5 * a1 + 10 * d = 100) (h2 : (3 * a1 + 9 * d) * (1 / 7) = 2 * a1 + d) : a1 = 5 / 3 :=
by
  sorry

end smallest_share_arith_seq_l102_102848


namespace simplify_expression_l102_102540

theorem simplify_expression (x : ℝ) : 3 * x + 4 - x + 8 = 2 * x + 12 :=
by
  sorry

end simplify_expression_l102_102540


namespace train_length_l102_102416
-- Import all necessary libraries from Mathlib

-- Define the given conditions and prove the target
theorem train_length (L_t L_p : ℝ) (h1 : L_t = L_p) (h2 : 54 * (1000 / 3600) * 60 = 2 * L_t) : L_t = 450 :=
by
  -- Proof goes here
  sorry

end train_length_l102_102416


namespace lateral_surface_area_cut_off_l102_102754

theorem lateral_surface_area_cut_off {a b c d : ℝ} (h₁ : a = 4) (h₂ : b = 25) 
(h₃ : c = (2/5 : ℝ)) (h₄ : d = 2 * (4 / 25) * b) : 
4 + 10 + (1/4 * b) = 20.25 :=
by
  sorry

end lateral_surface_area_cut_off_l102_102754


namespace most_suitable_survey_l102_102879

-- Define the options as a type
inductive SurveyOption
| A -- Understanding the crash resistance of a batch of cars
| B -- Surveying the awareness of the "one helmet, one belt" traffic regulations among citizens in our city
| C -- Surveying the service life of light bulbs produced by a factory
| D -- Surveying the quality of components of the latest stealth fighter in our country

-- Define a function determining the most suitable for a comprehensive survey
def mostSuitableForCensus : SurveyOption :=
  SurveyOption.D

-- Theorem statement that Option D is the most suitable for a comprehensive survey
theorem most_suitable_survey :
  mostSuitableForCensus = SurveyOption.D :=
  sorry

end most_suitable_survey_l102_102879


namespace largest_possible_n_l102_102925

theorem largest_possible_n (k : ℕ) (hk : k > 0) : ∃ n, n = 3 * k - 1 := 
  sorry

end largest_possible_n_l102_102925


namespace sin_sum_ge_sin_sum_l102_102534

-- Define the conditions with appropriate constraints
variables {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxπ : x ≤ π) (hyπ : y ≤ π) (hzπ : z ≤ π)

-- State the theorem to prove
theorem sin_sum_ge_sin_sum (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (hxπ : x ≤ π) (hyπ : y ≤ π) (hzπ : z ≤ π) : 
  (sin x) + (sin y) + (sin z) ≥ sin (x + y + z) := 
sorry

end sin_sum_ge_sin_sum_l102_102534


namespace arithmetic_sequence_general_geometric_sequence_sum_l102_102581

theorem arithmetic_sequence_general (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d) 
  (h_a3 : a 3 = -6) 
  (h_a6 : a 6 = 0) :
  ∀ n, a n = 2 * n - 12 := 
sorry

theorem geometric_sequence_sum (a b : ℕ → ℤ) 
  (r : ℤ) 
  (S : ℕ → ℤ)
  (h_geom : ∀ n : ℕ, b (n + 1) = b n * r) 
  (h_b1 : b 1 = -8) 
  (h_b2 : b 2 = a 0 + a 1 + a 2) 
  (h_a1 : a 0 = -10) 
  (h_a2 : a 1 = -8) 
  (h_a3 : a 2 = -6) :
  ∀ n, S n = 4 * (1 - 3 ^ n) := 
sorry

end arithmetic_sequence_general_geometric_sequence_sum_l102_102581


namespace laura_saves_more_with_promotion_A_l102_102156

def promotion_A_cost (pair_price : ℕ) : ℕ :=
  let second_pair_price := pair_price / 2
  pair_price + second_pair_price

def promotion_B_cost (pair_price : ℕ) : ℕ :=
  let discount := pair_price * 20 / 100
  pair_price + (pair_price - discount)

def savings (pair_price : ℕ) : ℕ :=
  promotion_B_cost pair_price - promotion_A_cost pair_price

theorem laura_saves_more_with_promotion_A :
  savings 50 = 15 :=
  by
  -- The detailed proof will be added here
  sorry

end laura_saves_more_with_promotion_A_l102_102156


namespace range_of_solutions_l102_102606

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end range_of_solutions_l102_102606


namespace distinct_pairs_l102_102248

-- Definitions of rational numbers and distinctness.
def is_distinct (x y : ℚ) : Prop := x ≠ y

-- Conditions
variables {a b r s : ℚ}

-- Main theorem: prove that there is only 1 distinct pair (a, b)
theorem distinct_pairs (h_ab_distinct : is_distinct a b)
  (h_rs_distinct : is_distinct r s)
  (h_eq : ∀ z : ℚ, (z - r) * (z - s) = (z - a * r) * (z - b * s)) : 
    ∃! (a b : ℚ), ∀ z : ℚ, (z - r) * (z - s) = (z - a * r) * (z - b * s) :=
  sorry

end distinct_pairs_l102_102248


namespace parabola_translation_correct_l102_102727

variable (x : ℝ)

def original_parabola : ℝ := 5 * x^2

def translated_parabola : ℝ := 5 * (x - 2)^2 + 3

theorem parabola_translation_correct :
  translated_parabola x = 5 * (x - 2)^2 + 3 :=
by
  sorry

end parabola_translation_correct_l102_102727


namespace parallel_lines_constant_l102_102646

theorem parallel_lines_constant (a : ℝ) : 
  (∀ x y : ℝ, (a - 1) * x + 2 * y + 3 = 0 → x + a * y + 3 = 0) → a = -1 :=
by sorry

end parallel_lines_constant_l102_102646


namespace age_of_john_l102_102513

theorem age_of_john (J S : ℕ) 
  (h1 : S = 2 * J)
  (h2 : S + (50 - J) = 60) :
  J = 10 :=
sorry

end age_of_john_l102_102513


namespace flat_fee_for_solar_panel_equipment_l102_102133

theorem flat_fee_for_solar_panel_equipment
  (land_acreage : ℕ)
  (land_cost_per_acre : ℕ)
  (house_cost : ℕ)
  (num_cows : ℕ)
  (cow_cost_per_cow : ℕ)
  (num_chickens : ℕ)
  (chicken_cost_per_chicken : ℕ)
  (installation_hours : ℕ)
  (installation_cost_per_hour : ℕ)
  (total_cost : ℕ)
  (total_spent : ℕ) :
  land_acreage * land_cost_per_acre + house_cost +
  num_cows * cow_cost_per_cow + num_chickens * chicken_cost_per_chicken +
  installation_hours * installation_cost_per_hour = total_spent →
  total_cost = total_spent →
  total_cost - (land_acreage * land_cost_per_acre + house_cost +
  num_cows * cow_cost_per_cow + num_chickens * chicken_cost_per_chicken +
  installation_hours * installation_cost_per_hour) = 26000 := by 
  sorry

end flat_fee_for_solar_panel_equipment_l102_102133


namespace no_solutions_to_equation_l102_102117

theorem no_solutions_to_equation : ¬∃ x : ℝ, (x ≠ 0) ∧ (x ≠ 5) ∧ ((2 * x ^ 2 - 10 * x) / (x ^ 2 - 5 * x) = x - 3) :=
by
  sorry

end no_solutions_to_equation_l102_102117


namespace exactly_two_roots_iff_l102_102182

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l102_102182


namespace school_club_profit_l102_102162

theorem school_club_profit : 
  let purchase_price_per_bar := 3 / 4
  let selling_price_per_bar := 2 / 3
  let total_bars := 1200
  let bars_with_discount := total_bars - 1000
  let discount_per_bar := 0.10
  let total_cost := total_bars * purchase_price_per_bar
  let total_revenue_without_discount := total_bars * selling_price_per_bar
  let total_discount := bars_with_discount * discount_per_bar
  let adjusted_revenue := total_revenue_without_discount - total_discount
  let profit := adjusted_revenue - total_cost
  profit = -116 :=
by sorry

end school_club_profit_l102_102162


namespace geometric_sequence_a5_l102_102378

theorem geometric_sequence_a5 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_a3 : a 3 = -1)
  (h_a7 : a 7 = -9) : a 5 = -3 := 
sorry

end geometric_sequence_a5_l102_102378


namespace percentage_students_passed_l102_102502

theorem percentage_students_passed
    (total_students : ℕ)
    (students_failed : ℕ)
    (students_passed : ℕ)
    (percentage_passed : ℕ)
    (h1 : total_students = 840)
    (h2 : students_failed = 546)
    (h3 : students_passed = total_students - students_failed)
    (h4 : percentage_passed = (students_passed * 100) / total_students) :
    percentage_passed = 35 := by
  sorry

end percentage_students_passed_l102_102502


namespace max_value_min_value_l102_102056

noncomputable def y (x : ℝ) : ℝ := 2 * Real.sin (3 * x + (Real.pi / 3))

theorem max_value (x : ℝ) : (∃ k : ℤ, x = (2 * k * Real.pi) / 3 + Real.pi / 18) ↔ y x = 2 :=
sorry

theorem min_value (x : ℝ) : (∃ k : ℤ, x = (2 * k * Real.pi) / 3 - 5 * Real.pi / 18) ↔ y x = -2 :=
sorry

end max_value_min_value_l102_102056


namespace problem_l102_102253

theorem problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by sorry

end problem_l102_102253


namespace compare_fractions_l102_102782

theorem compare_fractions : (6/29 : ℚ) < (8/25 : ℚ) ∧ (8/25 : ℚ) < (11/31 : ℚ):=
by
  have h1 : (6/29 : ℚ) < (8/25 : ℚ) := sorry
  have h2 : (8/25 : ℚ) < (11/31 : ℚ) := sorry
  exact ⟨h1, h2⟩

end compare_fractions_l102_102782


namespace b_1001_value_l102_102680

theorem b_1001_value (b : ℕ → ℝ)
  (h1 : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)) 
  (h2 : b 1 = 3 + Real.sqrt 11)
  (h3 : b 888 = 17 + Real.sqrt 11) : 
  b 1001 = 7 * Real.sqrt 11 - 20 := sorry

end b_1001_value_l102_102680


namespace union_complement_eq_l102_102692

def U := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 5}

def complement (univ : Set ℕ) (s : Set ℕ) : Set ℕ :=
  {x ∈ univ | x ∉ s}

theorem union_complement_eq :
  N ∪ (complement U M) = {2, 3, 5} :=
by sorry

end union_complement_eq_l102_102692


namespace total_distance_travelled_l102_102138

/-- Proving that the total horizontal distance traveled by the centers of two wheels with radii 1 m and 2 m 
    after one complete revolution is 6π meters. -/
theorem total_distance_travelled (R1 R2 : ℝ) (h1 : R1 = 1) (h2 : R2 = 2) : 
    2 * Real.pi * R1 + 2 * Real.pi * R2 = 6 * Real.pi :=
by
  sorry

end total_distance_travelled_l102_102138


namespace two_roots_iff_a_gt_neg1_l102_102189

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l102_102189


namespace one_thirds_in_nine_halves_l102_102653

theorem one_thirds_in_nine_halves : (9/2) / (1/3) = 27/2 := by
  sorry

end one_thirds_in_nine_halves_l102_102653


namespace find_b_l102_102566

-- Definitions from the conditions
variables (a b : ℝ)

-- Theorem statement using the conditions and the correct answer
theorem find_b (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 :=
by
  sorry

end find_b_l102_102566


namespace range_of_m_l102_102496

noncomputable def unique_zero_point (m : ℝ) : Prop :=
  ∀ x : ℝ, m * (1/4)^x - (1/2)^x + 1 = 0 → ∀ x' : ℝ, m * (1/4)^x' - (1/2)^x' + 1 = 0 → x = x'

theorem range_of_m (m : ℝ) : unique_zero_point m → (m ≤ 0 ∨ m = 1/4) :=
sorry

end range_of_m_l102_102496


namespace curve_C1_general_equation_curve_C2_cartesian_equation_minimum_distance_P1P2_l102_102507

noncomputable def parametric_curve_C1 (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 2 * Real.sin α)

noncomputable def polar_curve_C2 (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 4) = 3 * Real.sqrt 2

theorem curve_C1_general_equation (x y : ℝ) (α : ℝ) :
  (2 * Real.cos α = x) ∧ (Real.sqrt 2 * Real.sin α = y) →
  x^2 / 4 + y^2 / 2 = 1 :=
sorry

theorem curve_C2_cartesian_equation (ρ θ : ℝ) (x y : ℝ) :
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ polar_curve_C2 ρ θ →
  x + y = 6 :=
sorry

theorem minimum_distance_P1P2 (P1 P2 : ℝ × ℝ) (d : ℝ) :
  (∃ α, P1 = parametric_curve_C1 α) ∧ (∃ x y, P2 = (x, y) ∧ x + y = 6) →
  d = (3 * Real.sqrt 2 - Real.sqrt 3) :=
sorry

end curve_C1_general_equation_curve_C2_cartesian_equation_minimum_distance_P1P2_l102_102507


namespace length_of_A_l102_102518

structure Point := (x : ℝ) (y : ℝ)

noncomputable def length (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

theorem length_of_A'B' (A A' B B' C : Point) 
    (hA : A = ⟨0, 6⟩)
    (hB : B = ⟨0, 10⟩)
    (hC : C = ⟨3, 6⟩)
    (hA'_line : A'.y = A'.x)
    (hB'_line : B'.y = B'.x) 
    (hA'C : ∃ m b, ((C.y = m * C.x + b) ∧ (C.y = b) ∧ (A.y = b))) 
    (hB'C : ∃ m b, ((C.y = m * C.x + b) ∧ (B.y = m * B.x + b)))
    : length A' B' = (12 / 7) * Real.sqrt 2 :=
by
  sorry

end length_of_A_l102_102518


namespace always_meaningful_fraction_l102_102470

theorem always_meaningful_fraction {x : ℝ} : (∀ x, ∃ option : ℕ, 
  (option = 1 ∧ (x ≠ 1 ∧ x ≠ -1)) ∨ 
  (option = 2 ∧ True) ∨ 
  (option = 3 ∧ x ≠ 0) ∨ 
  (option = 4 ∧ x ≠ 1)) → option = 2 :=
sorry

end always_meaningful_fraction_l102_102470


namespace area_of_field_l102_102160

theorem area_of_field : ∀ (L W : ℕ), L = 20 → L + 2 * W = 88 → L * W = 680 :=
by
  intros L W hL hEq
  rw [hL] at hEq
  sorry

end area_of_field_l102_102160


namespace expression_value_l102_102933

theorem expression_value
  (x y z : ℝ)
  (hx : x = -5 / 4)
  (hy : y = -3 / 2)
  (hz : z = Real.sqrt 2) :
  -2 * x ^ 3 - y ^ 2 + Real.sin z = 53 / 32 + Real.sin (Real.sqrt 2) :=
by
  rw [hx, hy, hz]
  sorry

end expression_value_l102_102933


namespace group_photo_arrangements_grouping_methods_selection_methods_with_at_least_one_male_l102_102587

-- Question 1
theorem group_photo_arrangements {M F : ℕ} (hM : M = 3) (hF : F = 5) :
  ∃ arrangements : ℕ, arrangements = 14400 := 
sorry

-- Question 2
theorem grouping_methods {N : ℕ} (hN : N = 8) :
  ∃ methods : ℕ, methods = 2520 := 
sorry

-- Question 3
theorem selection_methods_with_at_least_one_male {M F : ℕ} (hM : M = 3) (hF : F = 5) :
  ∃ methods : ℕ, methods = 1560 := 
sorry

end group_photo_arrangements_grouping_methods_selection_methods_with_at_least_one_male_l102_102587


namespace one_thirds_in_nine_halves_l102_102651

theorem one_thirds_in_nine_halves : (9/2) / (1/3) = 27/2 := by
  sorry

end one_thirds_in_nine_halves_l102_102651


namespace probability_of_neighboring_points_l102_102292

theorem probability_of_neighboring_points (n : ℕ) (h : n ≥ 3) : 
  (2 / (n - 1) : ℝ) = (n / (n * (n - 1) / 2) : ℝ) :=
by sorry

end probability_of_neighboring_points_l102_102292


namespace domain_of_f_l102_102356

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan (Real.arcsin (x^2))

theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ↔ x ∈ {x : ℝ | f x = f x} :=
by
  sorry

end domain_of_f_l102_102356


namespace num_new_students_l102_102413

theorem num_new_students 
  (original_avg_age : ℕ) 
  (original_num_students : ℕ) 
  (new_avg_age : ℕ) 
  (age_decrease : ℕ) 
  (total_age_orginal : ℕ := original_num_students * original_avg_age) 
  (total_new_students : ℕ := (original_avg_age - age_decrease) * (original_num_students + 12))
  (x : ℕ := total_new_students - total_age_orginal) :
  original_avg_age = 40 → 
  original_num_students = 12 →
  new_avg_age = 32 →
  age_decrease = 4 →
  x = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end num_new_students_l102_102413


namespace josh_money_left_l102_102245

def initial_amount : ℝ := 20
def cost_hat : ℝ := 10
def cost_pencil : ℝ := 2
def number_of_cookies : ℝ := 4
def cost_per_cookie : ℝ := 1.25

theorem josh_money_left : initial_amount - cost_hat - cost_pencil - (number_of_cookies * cost_per_cookie) = 3 := by
  sorry

end josh_money_left_l102_102245


namespace monotonic_increasing_interval_l102_102277

noncomputable def log_base := (1 / 4 : ℝ)

def quad_expression (x : ℝ) : ℝ := -x^2 + 2*x + 3

def is_defined (x : ℝ) : Prop := quad_expression x > 0

theorem monotonic_increasing_interval : ∀ (x : ℝ), 
  is_defined x → 
  ∃ (a b : ℝ), 1 < a ∧ a ≤ x ∧ x < b ∧ b < 3 :=
by
  sorry

end monotonic_increasing_interval_l102_102277


namespace book_spending_fraction_l102_102822

-- Define the conditions
def earnings_per_week := 10
def num_weeks := 4
def total_savings := earnings_per_week * num_weeks
def money_spent_video_game := total_savings / 2
def remaining_money_after_video_game := total_savings - money_spent_video_game
def remaining_money_after_book := 15
def money_spent_book := remaining_money_after_video_game - remaining_money_after_book

-- Prove the fraction spent on the book is 1/4
theorem book_spending_fraction :
  (money_spent_book / remaining_money_after_video_game) = 1 / 4 :=
by
  sorry

end book_spending_fraction_l102_102822


namespace integer_solutions_exist_l102_102682

theorem integer_solutions_exist (a : ℕ) (ha : 0 < a) :
  ∃ x y : ℤ, x^2 - y^2 = a^3 := 
sorry

end integer_solutions_exist_l102_102682


namespace sedrich_more_jelly_beans_l102_102831

-- Define the given conditions
def napoleon_jelly_beans : ℕ := 17
def mikey_jelly_beans : ℕ := 19
def sedrich_jelly_beans (x : ℕ) : ℕ := napoleon_jelly_beans + x

-- Define the main theorem to be proved
theorem sedrich_more_jelly_beans (x : ℕ) :
  2 * (napoleon_jelly_beans + sedrich_jelly_beans x) = 4 * mikey_jelly_beans → x = 4 :=
by
  -- Proving the theorem
  sorry

end sedrich_more_jelly_beans_l102_102831


namespace max_area_triangle_l102_102089

theorem max_area_triangle (A B C : ℝ) (a b c : ℝ) (h1 : Real.sqrt 2 * Real.sin A = Real.sqrt 3 * Real.cos A) (h2 : a = Real.sqrt 3) :
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / (8 * Real.sqrt 5) := 
sorry

end max_area_triangle_l102_102089


namespace area_large_sphere_trace_l102_102323

-- Define the conditions
def radius_small_sphere : ℝ := 4
def radius_large_sphere : ℝ := 6
def area_small_sphere_trace : ℝ := 37

-- Define the mathematically equivalent proof problem
theorem area_large_sphere_trace :
  let r1 := radius_small_sphere,
      r2 := radius_large_sphere,
      a1 := area_small_sphere_trace,
      ratio := (r2 / r1) ^ 2 in
  a1 * ratio = 83.25 := by
sorry

end area_large_sphere_trace_l102_102323


namespace new_weights_inequality_l102_102023

theorem new_weights_inequality (W : ℝ) (x y : ℝ) (h_avg_increase : (8 * W - 2 * 68 + x + y) / 8 = W + 5.5)
  (h_sum_new_weights : x + y ≤ 180) : x > W ∧ y > W :=
by {
  sorry
}

end new_weights_inequality_l102_102023


namespace hyperbola_m_range_l102_102239

-- Given conditions
def is_hyperbola_equation (m : ℝ) : Prop :=
  ∃ x y : ℝ, (4 - m) ≠ 0 ∧ (2 + m) ≠ 0 ∧ x^2 / (4 - m) - y^2 / (2 + m) = 1

-- Prove the range of m is -2 < m < 4
theorem hyperbola_m_range (m : ℝ) : is_hyperbola_equation m → (-2 < m ∧ m < 4) :=
by
  sorry

end hyperbola_m_range_l102_102239


namespace tan_ratio_l102_102677

variable (a b : Real)

theorem tan_ratio (h1 : Real.sin (a + b) = 5 / 8) (h2 : Real.sin (a - b) = 1 / 4) : 
  (Real.tan a) / (Real.tan b) = 7 / 3 := 
by 
  sorry

end tan_ratio_l102_102677


namespace solution_range_l102_102612

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end solution_range_l102_102612


namespace roots_eq_two_iff_a_gt_neg1_l102_102198

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l102_102198


namespace concert_people_count_l102_102146

variable {W M : ℕ}

theorem concert_people_count (h1 : W * 2 = M) (h2 : (W - 12) * 3 = M - 29) : W + M = 21 := 
sorry

end concert_people_count_l102_102146


namespace statement_B_statement_C_statement_D_l102_102143

-- Statement B
theorem statement_B (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a^3 * c < b^3 * c :=
sorry

-- Statement C
theorem statement_C (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) : (a / (c - a)) > (b / (c - b)) :=
sorry

-- Statement D
theorem statement_D (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ b < 0 :=
sorry

end statement_B_statement_C_statement_D_l102_102143


namespace katie_total_earnings_l102_102676

-- Define the conditions
def bead_necklaces := 4
def gem_necklaces := 3
def price_per_necklace := 3

-- The total money earned
def total_money_earned := bead_necklaces + gem_necklaces * price_per_necklace = 21

-- The statement to prove
theorem katie_total_earnings : total_money_earned :=
by
  sorry

end katie_total_earnings_l102_102676


namespace probability_three_mn_sub_m_sub_n_multiple_of_five_l102_102707

open Finset

noncomputable def is_multiple_of_five (m n : ℤ) : Prop :=
  (3 * m * n - m - n) % 5 = 0

noncomputable def probability : ℚ :=
  let s := {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let total_pairs := (s.product s).filter (λ p, p.1 ≠ p.2)
  let valid_pairs := total_pairs.filter (λ p, is_multiple_of_five p.1 p.2)
  (valid_pairs.card : ℚ) / (total_pairs.card : ℚ)

theorem probability_three_mn_sub_m_sub_n_multiple_of_five :
  probability = (2 : ℚ) / (9 : ℚ) :=
sorry

end probability_three_mn_sub_m_sub_n_multiple_of_five_l102_102707


namespace g_does_not_pass_second_quadrant_l102_102644

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x+5) + 4

def M : ℝ × ℝ := (-5, 5)

noncomputable def g (x : ℝ) : ℝ := -5 + (5 : ℝ)^(x)

theorem g_does_not_pass_second_quadrant (a : ℝ) (x : ℝ) 
  (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hM : f a (-5) = 5) : 
  ∀ x < 0, g x < 0 :=
by
  sorry

end g_does_not_pass_second_quadrant_l102_102644


namespace range_of_solutions_l102_102613

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end range_of_solutions_l102_102613


namespace compute_fraction_sum_l102_102679

theorem compute_fraction_sum
  (a b c : ℝ)
  (h : a^3 - 6 * a^2 + 11 * a = 12)
  (h : b^3 - 6 * b^2 + 11 * b = 12)
  (h : c^3 - 6 * c^2 + 11 * c = 12) :
  (ab : ℝ) / c + (bc : ℝ) / a + (ca : ℝ) / b = -23 / 12 := by
  sorry

end compute_fraction_sum_l102_102679


namespace all_flowers_bloom_simultaneously_l102_102949

-- Define days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

-- Define bloom conditions for the flowers
def sunflowers_bloom (d : Day) : Prop :=
  d ≠ Tuesday ∧ d ≠ Thursday ∧ d ≠ Sunday

def lilies_bloom (d : Day) : Prop :=
  d ≠ Thursday ∧ d ≠ Saturday

def peonies_bloom (d : Day) : Prop :=
  d ≠ Sunday

-- Define the main theorem
theorem all_flowers_bloom_simultaneously : ∃ d : Day, 
  sunflowers_bloom d ∧ lilies_bloom d ∧ peonies_bloom d ∧
  (∀ d', d' ≠ d → ¬ (sunflowers_bloom d' ∧ lilies_bloom d' ∧ peonies_bloom d')) :=
by
  sorry

end all_flowers_bloom_simultaneously_l102_102949


namespace range_of_a_l102_102926

noncomputable def setM (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def setN : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem range_of_a (a : ℝ) : setM a ∪ setN = setN ↔ (-2 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l102_102926


namespace roots_eq_two_iff_a_gt_neg1_l102_102199

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l102_102199


namespace tangent_same_at_origin_l102_102274

noncomputable def f (x : ℝ) := Real.exp (3 * x) - 1
noncomputable def g (x : ℝ) := 3 * Real.exp x - 3

theorem tangent_same_at_origin :
  (deriv f 0 = deriv g 0) ∧ (f 0 = g 0) :=
by
  sorry

end tangent_same_at_origin_l102_102274


namespace probability_four_friends_same_group_l102_102163

open ProbabilityTheory

theorem probability_four_friends_same_group :
  let n := 800
  let groups := 4
  let friends := 4
  let p := (1 / groups) ^ (friends - 1)
  p = 1 / 64 :=
begin
  sorry
end

end probability_four_friends_same_group_l102_102163


namespace initial_volume_shampoo_l102_102593

theorem initial_volume_shampoo (V : ℝ) 
  (replace_rate : ℝ)
  (use_rate : ℝ)
  (t : ℝ) 
  (hot_sauce_fraction : ℝ) 
  (hot_sauce_amount : ℝ) : 
  replace_rate = 1/2 → 
  use_rate = 1 → 
  t = 4 → 
  hot_sauce_fraction = 0.25 → 
  hot_sauce_amount = t * replace_rate → 
  hot_sauce_amount = hot_sauce_fraction * V → 
  V = 8 :=
by 
  intro h_replace_rate h_use_rate h_t h_hot_sauce_fraction h_hot_sauce_amount h_hot_sauce_amount_eq
  sorry

end initial_volume_shampoo_l102_102593


namespace a_range_l102_102807

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2
def g (x a : ℝ) : ℝ := 2 * x + a

theorem a_range :
  (∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (1 / 2) 2 ∧ x2 ∈ Set.Icc (1 / 2) 2 ∧ f x1 = g x2 a) ↔ -5 ≤ a ∧ a ≤ 0 := 
by 
  sorry

end a_range_l102_102807


namespace cylinder_volume_l102_102640

theorem cylinder_volume (r h : ℝ) (π : ℝ) 
  (h_pos : 0 < π) 
  (cond1 : 2 * π * r * h = 100 * π) 
  (cond2 : 4 * r^2 + h^2 = 200) : 
  (π * r^2 * h = 250 * π) := 
by 
  sorry

end cylinder_volume_l102_102640


namespace range_of_a_l102_102935

noncomputable def f (a x : ℝ) := a * Real.log x + x - 1

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → f a x ≥ 0) : a ≥ -1 := by
  sorry

end range_of_a_l102_102935


namespace sum_of_coeffs_in_expansion_l102_102672

theorem sum_of_coeffs_in_expansion (n : ℕ) : 
  (1 - 2 : ℤ)^n = (-1 : ℤ)^n :=
by
  sorry

end sum_of_coeffs_in_expansion_l102_102672


namespace min_x2_plus_y2_l102_102943

theorem min_x2_plus_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : x^2 + y^2 ≥ 1 :=
sorry

end min_x2_plus_y2_l102_102943


namespace jogging_path_diameter_l102_102391

theorem jogging_path_diameter 
  (d_pond : ℝ)
  (w_flowerbed : ℝ)
  (w_jogging_path : ℝ)
  (h_pond : d_pond = 20)
  (h_flowerbed : w_flowerbed = 10)
  (h_jogging_path : w_jogging_path = 12) :
  2 * (d_pond / 2 + w_flowerbed + w_jogging_path) = 64 :=
by
  sorry

end jogging_path_diameter_l102_102391


namespace arithmetic_geometric_sequence_ratio_l102_102923

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℕ) (d : ℕ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_positive_d : d > 0)
  (h_geometric : a 6 ^ 2 = a 2 * a 12) :
  (a 12) / (a 2) = 9 / 4 :=
sorry

end arithmetic_geometric_sequence_ratio_l102_102923


namespace ages_when_john_is_50_l102_102965

variable (age_john age_alice age_mike : ℕ)

-- Given conditions:
-- John is 10 years old
def john_is_10 : age_john = 10 := by sorry

-- Alice is twice John's age
def alice_is_twice_john : age_alice = 2 * age_john := by sorry

-- Mike is 4 years younger than Alice
def mike_is_4_years_younger : age_mike = age_alice - 4 := by sorry

-- Prove that when John is 50 years old, Alice will be 60 years old, and Mike will be 56 years old
theorem ages_when_john_is_50 : age_john = 50 → age_alice = 60 ∧ age_mike = 56 := 
by 
  intro h
  sorry

end ages_when_john_is_50_l102_102965


namespace savings_fraction_l102_102592

variable (P : ℝ) 
variable (S : ℝ)
variable (E : ℝ)
variable (T : ℝ)

theorem savings_fraction :
  (12 * P * S) = 2 * P * (1 - S) → S = 1 / 7 :=
by
  intro h
  sorry

end savings_fraction_l102_102592


namespace inequality_am_gm_l102_102107

theorem inequality_am_gm (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/2) :
  (1 - a) * (1 - b) ≤ 9 / 16 := 
by
  sorry

end inequality_am_gm_l102_102107


namespace simplify_and_evaluate_l102_102539

theorem simplify_and_evaluate 
  (a b : ℚ) (h_a : a = -1/3) (h_b : b = -3) : 
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := 
  by 
    rw [h_a, h_b]
    sorry

end simplify_and_evaluate_l102_102539


namespace necessary_and_sufficient_condition_l102_102482

theorem necessary_and_sufficient_condition (x : ℝ) : (x > 0) ↔ (1 / x > 0) :=
by
  sorry

end necessary_and_sufficient_condition_l102_102482


namespace mult_mod_7_zero_l102_102772

theorem mult_mod_7_zero :
  (2007 ≡ 5 [MOD 7]) →
  (2008 ≡ 6 [MOD 7]) →
  (2009 ≡ 0 [MOD 7]) →
  (2010 ≡ 1 [MOD 7]) →
  (2007 * 2008 * 2009 * 2010 ≡ 0 [MOD 7]) :=
by
  intros h1 h2 h3 h4
  sorry

end mult_mod_7_zero_l102_102772


namespace prob_dominant_trait_one_child_prob_at_least_one_dominant_trait_two_children_l102_102600

-- Define the probability of a genotype given two mixed genotype (rd) parents producing a child.
def prob_genotype_dd : ℚ := (1/2) * (1/2)
def prob_genotype_rr : ℚ := (1/2) * (1/2)
def prob_genotype_rd : ℚ := 2 * (1/2) * (1/2)

-- Assertion that the probability of a child displaying the dominant characteristic (dd or rd) is 3/4.
theorem prob_dominant_trait_one_child : 
  prob_genotype_dd + prob_genotype_rd = 3/4 := sorry

-- Define the probability of two children both being rr.
def prob_both_rr_two_children : ℚ := prob_genotype_rr * prob_genotype_rr

-- Assertion that the probability of at least one of two children displaying the dominant characteristic is 15/16.
theorem prob_at_least_one_dominant_trait_two_children : 
  1 - prob_both_rr_two_children = 15/16 := sorry

end prob_dominant_trait_one_child_prob_at_least_one_dominant_trait_two_children_l102_102600


namespace number_of_ways_to_choose_one_class_number_of_ways_to_choose_one_class_each_grade_number_of_ways_to_choose_two_classes_different_grades_l102_102329

-- Define the number of classes in each grade.
def num_classes_first_year : ℕ := 14
def num_classes_second_year : ℕ := 14
def num_classes_third_year : ℕ := 15

-- Prove the number of different ways to choose students from 1 class.
theorem number_of_ways_to_choose_one_class :
  (num_classes_first_year + num_classes_second_year + num_classes_third_year) = 43 := 
by {
  -- Numerical calculation
  sorry
}

-- Prove the number of different ways to choose students from one class in each grade.
theorem number_of_ways_to_choose_one_class_each_grade :
  (num_classes_first_year * num_classes_second_year * num_classes_third_year) = 2940 := 
by {
  -- Numerical calculation
  sorry
}

-- Prove the number of different ways to choose students from 2 classes from different grades.
theorem number_of_ways_to_choose_two_classes_different_grades :
  (num_classes_first_year * num_classes_second_year + num_classes_first_year * num_classes_third_year + num_classes_second_year * num_classes_third_year) = 616 := 
by {
  -- Numerical calculation
  sorry
}

end number_of_ways_to_choose_one_class_number_of_ways_to_choose_one_class_each_grade_number_of_ways_to_choose_two_classes_different_grades_l102_102329


namespace exactly_two_roots_iff_l102_102184

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l102_102184


namespace sqrt_inequality_l102_102982

theorem sqrt_inequality : (Real.sqrt 3 + Real.sqrt 7) < 2 * Real.sqrt 5 := 
  sorry

end sqrt_inequality_l102_102982


namespace age_product_difference_l102_102169

theorem age_product_difference 
  (age_today : ℕ) 
  (Arnold_age : age_today = 6) 
  (Danny_age : age_today = 6) : 
  (7 * 7) - (6 * 6) = 13 := 
by
  sorry

end age_product_difference_l102_102169


namespace cosine_double_angle_l102_102063

theorem cosine_double_angle (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cosine_double_angle_l102_102063


namespace range_of_a_l102_102372

-- Definitions of sets and the problem conditions
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}
def condition (a : ℝ) : Prop := P ∪ M a = P

-- The theorem stating what needs to be proven
theorem range_of_a (a : ℝ) (h : condition a) : -1 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l102_102372


namespace table_area_l102_102867

/-- Given the combined area of three table runners is 224 square inches, 
     overlapping the runners to cover 80% of a table results in exactly 24 square inches being covered by 
     two layers, and the area covered by three layers is 30 square inches,
     prove that the area of the table is 175 square inches. -/
theorem table_area (A : ℝ) (S T H : ℝ) (h1 : S + 2 * T + 3 * H = 224)
   (h2 : 0.80 * A = S + T + H) (h3 : T = 24) (h4 : H = 30) : A = 175 := 
sorry

end table_area_l102_102867


namespace volume_of_four_cubes_l102_102298

theorem volume_of_four_cubes (edge_length : ℕ) (num_cubes : ℕ) (h_edge : edge_length = 5) (h_num : num_cubes = 4) :
  num_cubes * (edge_length ^ 3) = 500 :=
by 
  sorry

end volume_of_four_cubes_l102_102298


namespace wire_not_used_l102_102346

variable (total_wire length_cut_parts parts_used : ℕ)

theorem wire_not_used (h1 : total_wire = 50) (h2 : length_cut_parts = 5) (h3 : parts_used = 3) : 
  total_wire - (parts_used * (total_wire / length_cut_parts)) = 20 := 
  sorry

end wire_not_used_l102_102346


namespace count_f_compositions_l102_102973

noncomputable def count_special_functions : Nat :=
  let A := Finset.range 6
  let f := (Set.univ : Set (A → A))
  sorry

theorem count_f_compositions (f : Fin 6 → Fin 6) 
  (h : ∀ x : Fin 6, (f ∘ f ∘ f) x = x) :
  count_special_functions = 81 :=
sorry

end count_f_compositions_l102_102973


namespace missing_number_in_proportion_l102_102086

theorem missing_number_in_proportion (x : ℝ) :
  (2 / x) = ((4 / 3) / (10 / 3)) → x = 5 :=
by sorry

end missing_number_in_proportion_l102_102086


namespace range_of_g_l102_102519

noncomputable def f (x : ℝ) : ℝ := 2 * x - 3

noncomputable def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → -29 ≤ g x ∧ g x ≤ 3) :=
sorry

end range_of_g_l102_102519


namespace find_x_y_l102_102490

theorem find_x_y (x y : ℝ) 
  (h1 : 3 * x = 0.75 * y)
  (h2 : x + y = 30) : x = 6 ∧ y = 24 := 
by
  sorry  -- Proof is omitted

end find_x_y_l102_102490


namespace airport_distance_l102_102779

theorem airport_distance (d t : ℝ) (h1 : d = 45 * (t + 0.75))
                         (h2 : d - 45 = 65 * (t - 1.25)) :
  d = 241.875 :=
by
  sorry

end airport_distance_l102_102779


namespace sum_infinite_series_eq_l102_102628

theorem sum_infinite_series_eq : 
  ∑' n : ℕ, (n + 1) * (1 / 999 : ℝ) ^ n = 1000 / 998 := by
sorry

end sum_infinite_series_eq_l102_102628


namespace beth_red_pill_cost_l102_102617

noncomputable def red_pill_cost (blue_pill_cost : ℝ) : ℝ := blue_pill_cost + 3

theorem beth_red_pill_cost :
  ∃ (blue_pill_cost : ℝ), 
  (21 * (red_pill_cost blue_pill_cost + blue_pill_cost) = 966) 
  → 
  red_pill_cost blue_pill_cost = 24.5 :=
by
  sorry

end beth_red_pill_cost_l102_102617


namespace geometric_sequence_common_ratio_l102_102393

theorem geometric_sequence_common_ratio (a_1 q : ℝ) 
  (h1 : a_1 * q^2 = 9) 
  (h2 : a_1 * (1 + q) + 9 = 27) : 
  q = 1 ∨ q = -1/2 := 
by
  sorry

end geometric_sequence_common_ratio_l102_102393


namespace lowest_common_denominator_l102_102853

theorem lowest_common_denominator (a b c : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : c = 18) : Nat.lcm (Nat.lcm a b) c = 36 :=
by
  -- Introducing the given conditions
  rw [h1, h2, h3]
  -- Compute the LCM of the provided values
  sorry

end lowest_common_denominator_l102_102853


namespace math_proof_problem_l102_102887

-- Define constants
def x := 2000000000000
def y := 1111111111111

-- Prove the main statement
theorem math_proof_problem :
  2 * (x - y) = 1777777777778 := 
  by
    sorry

end math_proof_problem_l102_102887


namespace cubic_three_real_roots_l102_102712

theorem cubic_three_real_roots (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧
   x₁ ^ 3 - 3 * x₁ - a = 0 ∧
   x₂ ^ 3 - 3 * x₂ - a = 0 ∧
   x₃ ^ 3 - 3 * x₃ - a = 0) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end cubic_three_real_roots_l102_102712


namespace cost_of_bananas_l102_102868

theorem cost_of_bananas (A B : ℝ) (n : ℝ) (Tcost: ℝ) (Acost: ℝ): 
  (A * n + B = Tcost) → (A * (1 / 2 * n) + B = Acost) → (Tcost = 7) → (Acost = 5) → B = 3 :=
by
  intros hTony hArnold hTcost hAcost
  sorry

end cost_of_bananas_l102_102868


namespace find_C_given_eq_statement_max_area_triangle_statement_l102_102088

open Real

noncomputable def find_C_given_eq (a b c A : ℝ) (C : ℝ) : Prop :=
  (2 * a = sqrt 3 * c * sin A - a * cos C) → 
  C = 2 * π / 3

noncomputable def max_area_triangle (a b c : ℝ) (C : ℝ) : Prop :=
  C = 2 * π / 3 →
  c = sqrt 3 →
  ∃ S, S = (sqrt 3 / 4) * a * b ∧ 
  ∀ a b : ℝ, a * b ≤ 1 → S = (sqrt 3 / 4)

-- Lean statements
theorem find_C_given_eq_statement (a b c A C : ℝ) : find_C_given_eq a b c A C := 
by sorry

theorem max_area_triangle_statement (a b c : ℝ) (C : ℝ) : max_area_triangle a b c C := 
by sorry

end find_C_given_eq_statement_max_area_triangle_statement_l102_102088


namespace calories_in_dressing_l102_102243

noncomputable def lettuce_calories : ℝ := 50
noncomputable def carrot_calories : ℝ := 2 * lettuce_calories
noncomputable def crust_calories : ℝ := 600
noncomputable def pepperoni_calories : ℝ := crust_calories / 3
noncomputable def cheese_calories : ℝ := 400

noncomputable def salad_calories : ℝ := lettuce_calories + carrot_calories
noncomputable def pizza_calories : ℝ := crust_calories + pepperoni_calories + cheese_calories

noncomputable def salad_eaten : ℝ := salad_calories / 4
noncomputable def pizza_eaten : ℝ := pizza_calories / 5

noncomputable def total_eaten : ℝ := salad_eaten + pizza_eaten

theorem calories_in_dressing : ((330 : ℝ) - total_eaten) = 52.5 := by
  sorry

end calories_in_dressing_l102_102243


namespace geometric_sum_l102_102252

open BigOperators

noncomputable def geom_sequence (a q : ℚ) (n : ℕ) : ℚ := a * q ^ n

noncomputable def sum_geom_sequence (a q : ℚ) (n : ℕ) : ℚ := 
  if q = 1 then a * n
  else a * (1 - q ^ (n + 1)) / (1 - q)

theorem geometric_sum (a q : ℚ) (h_a : a = 1) (h_S3 : sum_geom_sequence a q 2 = 3 / 4) :
  sum_geom_sequence a q 3 = 5 / 8 :=
sorry

end geometric_sum_l102_102252


namespace find_A_l102_102571

def A : ℕ := 7 * 5 + 3

theorem find_A : A = 38 :=
by
  sorry

end find_A_l102_102571


namespace verify_differential_eq_l102_102985

noncomputable def y (x : ℝ) : ℝ := (2 + 3 * x - 3 * x^2)^(1 / 3 : ℝ)
noncomputable def y_prime (x : ℝ) : ℝ := 
  1 / 3 * (2 + 3 * x - 3 * x^2)^(-2 / 3 : ℝ) * (3 - 6 * x)

theorem verify_differential_eq (x : ℝ) :
  y x * y_prime x = (1 - 2 * x) / y x :=
by
  sorry

end verify_differential_eq_l102_102985


namespace total_money_correct_l102_102110

-- Define the number of pennies and quarters Sam has
def pennies : ℕ := 9
def quarters : ℕ := 7

-- Define the value of one penny and one quarter
def penny_value : ℝ := 0.01
def quarter_value : ℝ := 0.25

-- Calculate the total value of pennies and quarters Sam has
def total_value : ℝ := pennies * penny_value + quarters * quarter_value

-- Proof problem: Prove that the total value of money Sam has is $1.84
theorem total_money_correct : total_value = 1.84 :=
sorry

end total_money_correct_l102_102110


namespace solution_interval_l102_102603

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end solution_interval_l102_102603


namespace max_sum_of_diagonals_l102_102161

theorem max_sum_of_diagonals (a b : ℝ) (h_side : a^2 + b^2 = 25) (h_bounds1 : 2 * a ≤ 6) (h_bounds2 : 2 * b ≥ 6) : 2 * (a + b) = 14 :=
sorry

end max_sum_of_diagonals_l102_102161


namespace fiona_observe_pairs_l102_102499

def classroom_pairs (n : ℕ) : ℕ :=
  if n > 1 then n - 1 else 0

theorem fiona_observe_pairs :
  classroom_pairs 12 = 11 :=
by
  sorry

end fiona_observe_pairs_l102_102499


namespace quadrilateral_area_l102_102363

noncomputable def AB : ℝ := 3
noncomputable def BC : ℝ := 3
noncomputable def CD : ℝ := 4
noncomputable def DA : ℝ := 8
noncomputable def angle_DAB_add_angle_ABC : ℝ := 180

theorem quadrilateral_area :
  AB = 3 ∧ BC = 3 ∧ CD = 4 ∧ DA = 8 ∧ angle_DAB_add_angle_ABC = 180 →
  ∃ area : ℝ, area = 13.2 :=
by {
  sorry
}

end quadrilateral_area_l102_102363


namespace bulb_illumination_l102_102291

theorem bulb_illumination (n : ℕ) (h : n = 6) : 
  (2^n - 1) = 63 := by {
  sorry
}

end bulb_illumination_l102_102291


namespace probability_prime_sum_l102_102345

noncomputable def firstEightPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def isPrimeSum (a b : ℕ) : Bool :=
  Nat.prime (a + b)

def validPairs : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 11), (2, 17)]

def totalPairs : ℕ :=
  Nat.choose 8 2

def favorablePairs : ℕ :=
  4 -- Manually counted from solution step 3

theorem probability_prime_sum :
  Rat.mk favorablePairs totalPairs = Rat.mk 1 7 := by
  sorry

end probability_prime_sum_l102_102345


namespace solution_set_inequality_l102_102469

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 :=
sorry

end solution_set_inequality_l102_102469


namespace find_missing_number_l102_102461

theorem find_missing_number (square boxplus boxtimes boxminus : ℕ) :
  square = 423 / 47 ∧
  1448 = 282 * boxminus + (boxminus * 10 + boxtimes) ∧
  423 * (boxplus / 3) = 282 →
  square = 9 ∧
  boxminus = 5 ∧
  boxtimes = 8 ∧
  boxplus = 2 ∧
  9 = 9 :=
by
  intro h
  sorry

end find_missing_number_l102_102461


namespace initial_percentage_increase_l102_102588

theorem initial_percentage_increase 
  (W R : ℝ) 
  (P : ℝ)
  (h1 : R = W * (1 + P/100)) 
  (h2 : R * 0.70 = W * 1.18999999999999993) :
  P = 70 :=
by sorry

end initial_percentage_increase_l102_102588


namespace find_y_find_x_l102_102955

-- Define vectors as per the conditions
def a : ℝ × ℝ := (3, -2)
def b (y : ℝ) : ℝ × ℝ := (-1, y)
def c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition for perpendicular vectors
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Define the condition for parallel vectors
def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Question 1 Proof Statement
theorem find_y : ∀ (y : ℝ), is_perpendicular a (b y) → y = 3 / 2 :=
by
  intros y h
  unfold is_perpendicular at h
  unfold dot_product at h
  sorry

-- Question 2 Proof Statement
theorem find_x : ∀ (x : ℝ), is_parallel a (c x) → x = 15 / 2 :=
by
  intros x h
  unfold is_parallel at h
  sorry

end find_y_find_x_l102_102955


namespace distribution_of_books_l102_102148

theorem distribution_of_books :
  let A := 2 -- number of identical art albums (type A)
  let B := 3 -- number of identical stamp albums (type B)
  let friends := 4 -- number of friends
  let total_ways := 5 -- total number of ways to distribute books 
  (A + B) = friends + 1 →
  total_ways = 5 := 
by
  intros A B friends total_ways h
  sorry

end distribution_of_books_l102_102148


namespace unique_solution_implies_d_999_l102_102265

variable (a b c d x y : ℤ)

theorem unique_solution_implies_d_999
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : 3 * x + y = 3005)
  (h5 : y = |x-a| + |x-b| + |x-c| + |x-d|)
  (h6 : ∃! x, 3 * x + |x-a| + |x-b| + |x-c| + |x-d| = 3005) :
  d = 999 :=
sorry

end unique_solution_implies_d_999_l102_102265


namespace identity_function_l102_102095

theorem identity_function (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : ∀ n : ℕ, f n = n :=
by
  sorry

end identity_function_l102_102095


namespace smallest_possible_value_m_l102_102915

theorem smallest_possible_value_m (r y b : ℕ) (h : 16 * r = 18 * y ∧ 18 * y = 20 * b) : 
  ∃ m : ℕ, 30 * m = 16 * r ∧ 30 * m = 720 ∧ m = 24 :=
by {
  sorry
}

end smallest_possible_value_m_l102_102915


namespace negation_at_most_three_l102_102551

theorem negation_at_most_three :
  ¬ (∀ n : ℕ, n ≤ 3) ↔ (∃ n : ℕ, n ≥ 4) :=
by
  sorry

end negation_at_most_three_l102_102551


namespace krishan_money_l102_102883

theorem krishan_money (R G K : ℕ) (hR : R = 637) (hRG : R * 17 = G * 7) (hGK : G * 17 = K * 7) : K = 3774 :=
by {
  sorry -- Proof not required as per the instructions
}

end krishan_money_l102_102883


namespace xiaoming_grandfather_age_l102_102027

def grandfather_age (x xm_diff : ℕ) :=
  xm_diff = 60 ∧ x > 7 * (x - xm_diff) ∧ x < 70

theorem xiaoming_grandfather_age (x : ℕ) (h_cond : grandfather_age x 60) : x = 69 :=
by
  sorry

end xiaoming_grandfather_age_l102_102027


namespace right_triangle_perimeter_l102_102441

noncomputable theory

def perimeter_of_triangle (a b c : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter
  (a : ℝ) (area : ℝ) (b : ℝ) :
  area = 150 ∧ a = 30 →
  ∃ x c : ℝ, (1 / 2) * a * x = area ∧
             c^2 = a^2 + x^2 ∧
             perimeter_of_triangle a x c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end right_triangle_perimeter_l102_102441


namespace problem_statement_l102_102177

noncomputable def least_period (f : ℝ → ℝ) (P : ℝ) :=
  ∀ x : ℝ, f (x + P) = f x

theorem problem_statement (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 5) + f (x - 5) = f x) :
  least_period f 30 :=
sorry

end problem_statement_l102_102177


namespace maximum_value_of_d_l102_102971

theorem maximum_value_of_d (a b c d : ℝ) 
  (h₁ : a + b + c + d = 10)
  (h₂ : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end maximum_value_of_d_l102_102971


namespace min_value_inequality_l102_102522

theorem min_value_inequality (p q r : ℝ) (h₀ : 0 < p) (h₁ : 0 < q) (h₂ : 0 < r) :
  ( 3 * r / (p + 2 * q) + 3 * p / (2 * r + q) + 2 * q / (p + r) ) ≥ (29 / 6) := 
sorry

end min_value_inequality_l102_102522


namespace chords_intersect_probability_l102_102271

noncomputable def probability_chords_intersect (n m : ℕ) : ℚ :=
  if (n > 6 ∧ m = 2023) then
    1 / 72
  else
    0

theorem chords_intersect_probability :
  probability_chords_intersect 6 2023 = 1 / 72 :=
by
  sorry

end chords_intersect_probability_l102_102271


namespace find_a_of_exponential_passing_point_l102_102488

theorem find_a_of_exponential_passing_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_point : a^2 = 4) : a = 2 :=
by
  -- Proof will be filled in here
  sorry

end find_a_of_exponential_passing_point_l102_102488


namespace find_x_plus_y_l102_102428

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c p : V) (x y : ℝ)

-- Conditions: Definitions as the given problem requires
-- Basis definitions
def basis1 := [a, b, c]
def basis2 := [a + b, a - b, c]

-- Conditions on p
def condition1 : p = 3 • a + b + c := sorry
def condition2 : p = x • (a + b) + y • (a - b) + c := sorry

-- The proof statement
theorem find_x_plus_y (h1 : p = 3 • a + b + c) (h2 : p = x • (a + b) + y • (a - b) + c) :
  x + y = 3 :=
sorry

end find_x_plus_y_l102_102428


namespace problem_l102_102517

theorem problem (a : ℕ → ℝ) (h0 : a 1 = 0) (h9 : a 9 = 0)
  (h2_8 : ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a i > 0) (h_nonneg : ∀ n, 1 ≤ n ∧ n ≤ 9 → a n ≥ 0) : 
  (∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i-1) + a (i+1) < 2 * a i) ∧ (∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i-1) + a (i+1) < 1.9 * a i) := 
sorry

end problem_l102_102517


namespace linear_equation_condition_l102_102992

theorem linear_equation_condition (a : ℝ) :
  (∃ x : ℝ, (a - 2) * x ^ (|a|⁻¹ + 3) = 0) ↔ a = -2 := 
by
  sorry

end linear_equation_condition_l102_102992


namespace limit_T_div_S_l102_102919

open Real

noncomputable def S (a : ℝ) : ℝ :=
  (1/3) * a^3

noncomputable def T (a : ℝ) : ℝ :=
  let b := (2 * sqrt 3 * a - 1) / (2 * a + sqrt 3) - a
  ((a - b)^3) / 6

theorem limit_T_div_S (a : ℝ) (h : 0 < a) :
  tendsto (λ a, T a / S a) at_top (𝓝 4) :=
sorry

end limit_T_div_S_l102_102919


namespace probability_event_A_probability_event_B_probability_event_C_l102_102561

-- Define the total number of basic events for three dice
def total_basic_events : ℕ := 6 * 6 * 6

-- Define events and their associated basic events
def event_A_basic_events : ℕ := 2 * 3 * 3
def event_B_basic_events : ℕ := 2 * 3 * 6
def event_C_basic_events : ℕ := 6 * 6 * 3

-- Define probabilities for each event
def P_A : ℚ := event_A_basic_events / total_basic_events
def P_B : ℚ := event_B_basic_events / total_basic_events
def P_C : ℚ := event_C_basic_events / total_basic_events

-- Statement to be proven
theorem probability_event_A : P_A = 1 / 12 := by
  sorry

theorem probability_event_B : P_B = 1 / 6 := by
  sorry

theorem probability_event_C : P_C = 1 / 2 := by
  sorry

end probability_event_A_probability_event_B_probability_event_C_l102_102561


namespace maximum_angle_B_in_triangle_l102_102014

theorem maximum_angle_B_in_triangle
  (A B C M : ℝ × ℝ)
  (hM : midpoint ℝ A B = M)
  (h_angle_MAC : ∃ angle_MAC : ℝ, angle_MAC = 15) :
  ∃ angle_B : ℝ, angle_B = 105 := 
by
  sorry

end maximum_angle_B_in_triangle_l102_102014


namespace josh_money_left_l102_102246

def initial_amount : ℝ := 20
def cost_hat : ℝ := 10
def cost_pencil : ℝ := 2
def number_of_cookies : ℝ := 4
def cost_per_cookie : ℝ := 1.25

theorem josh_money_left : initial_amount - cost_hat - cost_pencil - (number_of_cookies * cost_per_cookie) = 3 := by
  sorry

end josh_money_left_l102_102246


namespace range_of_x_l102_102780

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x :
  {x : ℝ | odot x (x - 2) < 0} = {x : ℝ | -2 < x ∧ x < 1} := 
by sorry

end range_of_x_l102_102780


namespace square_of_999_l102_102058

theorem square_of_999 : 999 * 999 = 998001 := by
  sorry

end square_of_999_l102_102058


namespace inequality_proof_l102_102930

variables {x y a b ε m : ℝ}

theorem inequality_proof (h1 : |x - a| < ε / (2 * m))
                        (h2 : |y - b| < ε / (2 * |a|))
                        (h3 : 0 < y ∧ y < m) :
                        |x * y - a * b| < ε :=
sorry

end inequality_proof_l102_102930


namespace machines_job_time_l102_102230

theorem machines_job_time (D : ℝ) (h1 : 15 * D = D * 20 * (3 / 4)) : ¬ ∃ t : ℝ, t = D :=
by
  sorry

end machines_job_time_l102_102230


namespace sum_of_smallest_and_second_smallest_l102_102724

-- Define the set of numbers
def numbers : Set ℕ := {10, 11, 12, 13}

-- Define the smallest and second smallest numbers
def smallest_number : ℕ := 10
def second_smallest_number : ℕ := 11

-- Prove the sum of the smallest and the second smallest numbers
theorem sum_of_smallest_and_second_smallest : smallest_number + second_smallest_number = 21 := by
  sorry

end sum_of_smallest_and_second_smallest_l102_102724


namespace car_rental_cost_l102_102151

def daily_rental_rate : ℝ := 29
def per_mile_charge : ℝ := 0.08
def rental_duration : ℕ := 1
def distance_driven : ℝ := 214.0

theorem car_rental_cost : 
  (daily_rental_rate * rental_duration + per_mile_charge * distance_driven) = 46.12 := 
by 
  sorry

end car_rental_cost_l102_102151


namespace tangent_line_at_zero_range_of_a_l102_102641

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * Real.sin x - 1

theorem tangent_line_at_zero (h : ∀ x, f 1 x = Real.exp x - Real.sin x - 1) :
  ∀ x, Real.exp x - Real.sin x - 1 = f 1 x :=
by
  sorry

theorem range_of_a (h : ∀ x, f a x ≥ 0) : a ∈ Set.Iic 1 :=
by
  sorry

end tangent_line_at_zero_range_of_a_l102_102641


namespace max_value_fraction_l102_102484

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2 * x + y) + y / (x + 2 * y)) <= (2 / 3) := 
sorry

end max_value_fraction_l102_102484


namespace find_a8_l102_102479

variable {a : ℕ → ℝ} -- Assuming the sequence is real-valued for generality

-- Defining the necessary properties and conditions of the arithmetic sequence.
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a 0 + n * (a 1 - a 0)

-- Given conditions as hypothesis
variable (h_seq : arithmetic_sequence a) 
variable (h_sum : a 3 + a 6 + a 10 + a 13 = 32)

-- The proof statement
theorem find_a8 : a 8 = 8 :=
by
  sorry -- The proof itself

end find_a8_l102_102479


namespace p_evaluation_l102_102685

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else 2 * x + 2 * y

theorem p_evaluation : p (p 3 (-4)) (p (-7) 0) = 40 := by
  sorry

end p_evaluation_l102_102685


namespace count_positive_numbers_is_three_l102_102167

def negative_three := -3
def zero := 0
def negative_three_squared := (-3) ^ 2
def absolute_negative_nine := |(-9)|
def negative_one_raised_to_four := -1 ^ 4

def number_list : List Int := [ -negative_three, zero, negative_three_squared, absolute_negative_nine, negative_one_raised_to_four ]

def count_positive_numbers (lst: List Int) : Nat :=
  lst.foldl (λ acc x => if x > 0 then acc + 1 else acc) 0

theorem count_positive_numbers_is_three : count_positive_numbers number_list = 3 :=
by
  -- The proof will go here.
  sorry

end count_positive_numbers_is_three_l102_102167


namespace quadratic_transform_l102_102716

theorem quadratic_transform : ∀ (x : ℝ), x^2 = 3 * x + 1 ↔ x^2 - 3 * x - 1 = 0 :=
by
  sorry

end quadratic_transform_l102_102716


namespace relay_race_total_distance_l102_102837

theorem relay_race_total_distance
  (Sadie_speed : ℝ) (Sadie_time : ℝ) (Ariana_speed : ℝ) (Ariana_time : ℝ) (Sarah_speed : ℝ) (total_race_time : ℝ)
  (h1 : Sadie_speed = 3) (h2 : Sadie_time = 2)
  (h3 : Ariana_speed = 6) (h4 : Ariana_time = 0.5)
  (h5 : Sarah_speed = 4) (h6 : total_race_time = 4.5) :
  (Sadie_speed * Sadie_time + Ariana_speed * Ariana_time + Sarah_speed * (total_race_time - (Sadie_time + Ariana_time))) = 17 :=
by
  sorry

end relay_race_total_distance_l102_102837


namespace integral_cos_2x_eq_half_l102_102742

theorem integral_cos_2x_eq_half :
  ∫ x in (0:ℝ)..(Real.pi / 4), Real.cos (2 * x) = 1 / 2 := by
sorry

end integral_cos_2x_eq_half_l102_102742


namespace frank_money_left_l102_102215

theorem frank_money_left (initial_money : ℝ) (spent_groceries : ℝ) (spent_magazine : ℝ) :
  initial_money = 600 →
  spent_groceries = (1/5) * initial_money →
  spent_magazine = (1/4) * (initial_money - spent_groceries) →
  initial_money - spent_groceries - spent_magazine = 360 := 
by
  intro h1 h2 h3
  rw [h1] at *
  rw [h2] at *
  rw [h3] at *
  sorry

end frank_money_left_l102_102215


namespace problem_statement_l102_102695

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l102_102695


namespace solve_inequality_l102_102111

theorem solve_inequality (x : ℝ) : 1 + 2 * (x - 1) ≤ 3 → x ≤ 2 :=
by
  sorry

end solve_inequality_l102_102111


namespace sequence_neither_arithmetic_nor_geometric_l102_102066

noncomputable def Sn (n : ℕ) : ℕ := 3 * n + 2
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 5 else Sn n - Sn (n - 1)

def not_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ¬∃ d, ∀ n, a (n + 1) = a n + d

def not_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ¬∃ r, ∀ n, a (n + 1) = r * a n

theorem sequence_neither_arithmetic_nor_geometric :
  not_arithmetic_sequence a ∧ not_geometric_sequence a :=
sorry

end sequence_neither_arithmetic_nor_geometric_l102_102066


namespace total_distance_l102_102400

open Real

theorem total_distance :
  let jonathan_distance := 7.5
  let mercedes_distance := 2.5 * jonathan_distance
  let davonte_distance := real.sqrt (3.25 * mercedes_distance)
  let felicia_distance := davonte_distance - 1.75
  let average_distance := (jonathan_distance + davonte_distance + felicia_distance) / 3
  let emilia_distance := average_distance ^ 2
  in
      mercedes_distance + davonte_distance + felicia_distance + emilia_distance ≈ 83.321 := 
by
  let jonathan_distance := 7.5
  let mercedes_distance := 2.5 * jonathan_distance
  let davonte_distance := real.sqrt (3.25 * mercedes_distance)
  let felicia_distance := davonte_distance - 1.75
  let average_distance := (jonathan_distance + davonte_distance + felicia_distance) / 3
  let emilia_distance := average_distance ^ 2
  have mercedes_dist_calc : mercedes_distance = 18.75 := by norm_num [jonathan_distance]
  -- More steps to simplify and verify the distances
  have davonte_dist_calc : davonte_distance = sqrt 60.9375 := by norm_num [mercedes_distance]
  -- More steps to simplify and verify the distances
  have felicia_dist_calc : felicia_distance = davonte_distance - 1.75 := by norm_num [davonte_distance]
  have avg_dist_calc : average_distance = (jonathan_distance + davonte_distance + felicia_distance) / 3 := by norm_num
  have emilia_dist_calc : emilia_distance = average_distance ^ 2 := by norm_num [average_distance]
  have total_distance := mercedes_distance + davonte_distance + felicia_distance + emilia_distance
  -- Verify total_distance is approximately equal to 83.321
  have : abs (total_distance - 83.321) < 0.001,
  sorry

end total_distance_l102_102400


namespace probability_no_defective_pens_l102_102667

theorem probability_no_defective_pens
  (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ)
  (h_total : total_pens = 10)
  (h_defective : defective_pens = 2)
  (h_selected : selected_pens = 2) :
  let non_defective_pens := total_pens - defective_pens in
  let prob_first_non_defective := (non_defective_pens : ℚ) / total_pens in
  let prob_second_non_defective := ((non_defective_pens - 1) : ℚ) / (total_pens - 1) in
  prob_first_non_defective * prob_second_non_defective = 28 / 45 :=
by
  sorry

end probability_no_defective_pens_l102_102667


namespace combined_river_length_estimate_l102_102012

def river_length_GSA := 402 
def river_error_GSA := 0.5 
def river_prob_error_GSA := 0.04 

def river_length_AWRA := 403 
def river_error_AWRA := 0.5 
def river_prob_error_AWRA := 0.04 

/-- 
Given the measurements from GSA and AWRA, 
the combined estimate of the river's length, Rio-Coralio, is 402.5 km,
and the probability of error for this combined estimate is 0.04.
-/
theorem combined_river_length_estimate :
  ∃ l : ℝ, l = 402.5 ∧ ∀ p : ℝ, (p = 0.04) :=
sorry

end combined_river_length_estimate_l102_102012


namespace domain_of_f_l102_102355

def domain_f (x : ℝ) : Prop := f x = Real.tan (Real.arcsin (x^2))

theorem domain_of_f :
  ∀ x : ℝ, (domain_f x → -1 ≤ x ∧ x ≤ 1) ∧ (0 ≤ Real.arcsin (x^2) ∧ Real.arcsin (x^2) ≤ Real.pi / 2) :=
by 
  sorry -- Proof to be done.

end domain_of_f_l102_102355


namespace johns_age_l102_102512

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 70) : j = 20 := by
  sorry

end johns_age_l102_102512


namespace total_fundamental_particles_l102_102898

def protons := 9
def neutrons := 19 - protons
def electrons := protons
def total_particles := protons + neutrons + electrons

theorem total_fundamental_particles : total_particles = 28 := by
  sorry

end total_fundamental_particles_l102_102898


namespace additional_charge_per_segment_l102_102961

variable (initial_fee : ℝ := 2.35)
variable (total_charge : ℝ := 5.5)
variable (distance : ℝ := 3.6)
variable (segment_length : ℝ := (2/5 : ℝ))

theorem additional_charge_per_segment :
  let number_of_segments := distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  let additional_charge_per_segment := charge_for_distance / number_of_segments
  additional_charge_per_segment = 0.35 :=
by
  sorry

end additional_charge_per_segment_l102_102961


namespace find_p_q_sum_l102_102121

theorem find_p_q_sum (p q : ℝ) 
  (sum_condition : p / 3 = 8) 
  (product_condition : q / 3 = 12) : 
  p + q = 60 :=
by
  sorry

end find_p_q_sum_l102_102121


namespace find_r_cubed_and_reciprocal_cubed_l102_102491

variable (r : ℝ)
variable (h : (r + 1 / r) ^ 2 = 5)

theorem find_r_cubed_and_reciprocal_cubed (r : ℝ) (h : (r + 1 / r) ^ 2 = 5) : r ^ 3 + 1 / r ^ 3 = 2 * Real.sqrt 5 := by
  sorry

end find_r_cubed_and_reciprocal_cubed_l102_102491


namespace factor_expression_l102_102042

theorem factor_expression (x : ℝ) : 
  (10 * x^3 + 45 * x^2 - 5 * x) - (-5 * x^3 + 10 * x^2 - 5 * x) = 5 * x^2 * (3 * x + 7) :=
by 
  sorry

end factor_expression_l102_102042


namespace equation_two_roots_iff_l102_102206

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l102_102206


namespace range_of_a_l102_102483

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧ 
  (∀ x > 1, f x = a^x) ∧ 
  (∀ x ≤ 1, f x = (4 - (a / 2)) * x + 2) → 
  4 ≤ a ∧ a < 8 :=
by
  sorry

end range_of_a_l102_102483


namespace right_triangle_perimeter_l102_102439

theorem right_triangle_perimeter (a b : ℝ) (h₁ : a = 30) (h₂ : 0 < b) (h₃ : 0.5 * a * b = 150) : 
  let c := Real.sqrt (a^2 + b^2)
  in a + b + c = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l102_102439


namespace sum_of_digits_product_is_13_l102_102902

def base_eight_to_base_ten (n : ℕ) : ℕ := sorry
def product_base_eight (n1 n2 : ℕ) : ℕ := sorry
def digits_sum_base_ten (n : ℕ) : ℕ := sorry

theorem sum_of_digits_product_is_13 :
  let N1 := base_eight_to_base_ten 35
  let N2 := base_eight_to_base_ten 42
  let product := product_base_eight N1 N2
  digits_sum_base_ten product = 13 :=
by
  sorry

end sum_of_digits_product_is_13_l102_102902


namespace arc_length_of_sector_l102_102940

noncomputable def central_angle := 36
noncomputable def radius := 15

theorem arc_length_of_sector : (central_angle * Real.pi * radius / 180 = 3 * Real.pi) :=
by
  sorry

end arc_length_of_sector_l102_102940


namespace carlos_more_miles_than_dana_after_3_hours_l102_102547

-- Define the conditions
variable (carlos_total_distance : ℕ)
variable (carlos_advantage : ℕ)
variable (dana_total_distance : ℕ)
variable (time_hours : ℕ)

-- State the condition values that are given in the problem
def conditions : Prop :=
  carlos_total_distance = 50 ∧
  carlos_advantage = 5 ∧
  dana_total_distance = 40 ∧
  time_hours = 3

-- State the proof goal
theorem carlos_more_miles_than_dana_after_3_hours
  (h : conditions carlos_total_distance carlos_advantage dana_total_distance time_hours) :
  carlos_total_distance - dana_total_distance = 10 :=
by
  sorry

end carlos_more_miles_than_dana_after_3_hours_l102_102547


namespace probability_rain_at_most_3_days_in_july_l102_102285

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_rain_at_most_3_days_in_july :
  let p := 1 / 5
  let n := 31
  let sum_prob := binomial_probability n 0 p + binomial_probability n 1 p + binomial_probability n 2 p + binomial_probability n 3 p
  abs (sum_prob - 0.125) < 0.001 :=
by
  sorry

end probability_rain_at_most_3_days_in_july_l102_102285


namespace katya_classmates_l102_102702

-- Let N be the number of Katya's classmates
variable (N : ℕ)

-- Let K be the number of candies Artyom initially received
variable (K : ℕ)

-- Condition 1: After distributing some candies, Katya had 10 more candies left than Artyom
def condition_1 := K + 10

-- Condition 2: Katya gave each child, including herself, one more candy, so she gave out N + 1 candies in total
def condition_2 := N + 1

-- Condition 3: After giving out these N + 1 candies, everyone in the class has the same number of candies.
def condition_3 : Prop := (K + 1) = (condition_1 K - condition_2 N) / (N + 1)


-- Goal: Prove the number of Katya's classmates N is 9.
theorem katya_classmates : N = 9 :=
by
  -- Restate the conditions in Lean
  
  -- Apply the conditions to find that the only viable solution is N = 9
  sorry

end katya_classmates_l102_102702


namespace intersection_of_A_and_B_l102_102369

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, (x ∈ A ∩ B) ↔ (x = 0 ∨ x = 1) := by
  sorry

end intersection_of_A_and_B_l102_102369


namespace greatest_common_divisor_of_120_and_m_l102_102564

theorem greatest_common_divisor_of_120_and_m (m : ℕ) (h : (∀ d, d ∣ 120 ∧ d ∣ m → d = 1 ∨ d = 2 ∨ d = 4)) : gcd 120 m = 4 :=
by
  sorry

end greatest_common_divisor_of_120_and_m_l102_102564


namespace geometric_common_ratio_l102_102070

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_common_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : q > 0) 
  (h2 : geometric_seq a q) (h3 : a 3 * a 7 = 4 * (a 4)^2) : q = 2 := 
by 
  sorry

end geometric_common_ratio_l102_102070


namespace jason_initial_pears_l102_102960

-- Define the initial number of pears Jason picked.
variable (P : ℕ)

-- Conditions translated to Lean:
-- Jason gave Keith 47 pears and received 12 from Mike, leaving him with 11 pears.
variable (h1 : P - 47 + 12 = 11)

-- The theorem stating the problem:
theorem jason_initial_pears : P = 46 :=
by
  sorry

end jason_initial_pears_l102_102960


namespace largest_possible_d_l102_102970

theorem largest_possible_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 := 
sorry

end largest_possible_d_l102_102970


namespace ann_older_than_susan_l102_102037

variables (A S : ℕ)

theorem ann_older_than_susan (h1 : S = 11) (h2 : A + S = 27) : A - S = 5 := by
  -- Proof is skipped
  sorry

end ann_older_than_susan_l102_102037


namespace max_value_proof_l102_102106

noncomputable def max_value (x y z : ℝ) : ℝ :=
  1 / x + 2 / y + 3 / z

theorem max_value_proof (x y z : ℝ) (h1 : 2 / 5 ≤ z ∧ z ≤ min x y)
    (h2 : x * z ≥ 4 / 15) (h3 : y * z ≥ 1 / 5) : max_value x y z ≤ 13 := 
by
  sorry

end max_value_proof_l102_102106


namespace area_of_inscribed_hexagon_in_square_is_27sqrt3_l102_102747

noncomputable def side_length_of_triangle : ℝ := 6
noncomputable def radius_of_circle (a : ℝ) : ℝ := (a * Real.sqrt 2) / 2
noncomputable def side_length_of_square (r : ℝ) : ℝ := 2 * r
noncomputable def side_length_of_hexagon_in_square (s : ℝ) : ℝ := s / (Real.sqrt 2)
noncomputable def area_of_hexagon (side_hexagon : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * side_hexagon^2

theorem area_of_inscribed_hexagon_in_square_is_27sqrt3 :
  ∀ (a r s side_hex : ℝ), 
    a = side_length_of_triangle →
    r = radius_of_circle a →
    s = side_length_of_square r →
    side_hex = side_length_of_hexagon_in_square s →
    area_of_hexagon side_hex = 27 * Real.sqrt 3 :=
by
  intros a r s side_hex h_a h_r h_s h_side_hex
  sorry

end area_of_inscribed_hexagon_in_square_is_27sqrt3_l102_102747


namespace domino_covering_l102_102622

theorem domino_covering (m n : ℕ) (m_eq : (m, n) ∈ [(5, 5), (4, 6), (3, 7), (5, 6), (3, 8)]) :
  (m * n % 2 = 1) ↔ (m = 5 ∧ n = 5) ∨ (m = 3 ∧ n = 7) :=
by
  sorry

end domino_covering_l102_102622


namespace probability_at_least_one_black_ball_l102_102998

theorem probability_at_least_one_black_ball :
  let total_balls := 6
  let red_balls := 2
  let white_ball := 1
  let black_balls := 3
  let total_combinations := Nat.choose total_balls 2
  let non_black_combinations := Nat.choose (total_balls - black_balls) 2
  let probability := 1 - (non_black_combinations / total_combinations : ℚ)
  probability = 4 / 5 :=
by
  sorry

end probability_at_least_one_black_ball_l102_102998


namespace range_of_solutions_l102_102608

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end range_of_solutions_l102_102608


namespace real_solutions_l102_102627

-- Given the condition (equation)
def quadratic_equation (x y : ℝ) : Prop :=
  x^2 + 2 * x * Real.sin (x * y) + 1 = 0

-- The main theorem statement proving the solutions for x and y
theorem real_solutions (x y : ℝ) (k : ℤ) :
  quadratic_equation x y ↔
  (x = 1 ∧ (y = (Real.pi / 2 + 2 * k * Real.pi) ∨ y = (-Real.pi / 2 + 2 * k * Real.pi))) ∨
  (x = -1 ∧ (y = (-Real.pi / 2 + 2 * k * Real.pi) ∨ y = (Real.pi / 2 + 2 * k * Real.pi))) :=
by
  sorry

end real_solutions_l102_102627


namespace celsius_to_fahrenheit_conversion_l102_102529

theorem celsius_to_fahrenheit_conversion (k b : ℝ) :
  (∀ C : ℝ, (C * k + b = C * 1.8 + 32)) → (k = 1.8 ∧ b = 32) :=
by
  intro h
  sorry

end celsius_to_fahrenheit_conversion_l102_102529


namespace value_of_expression_eq_33_l102_102721

theorem value_of_expression_eq_33 : (3^2 + 7^2 - 5^2 = 33) := by
  sorry

end value_of_expression_eq_33_l102_102721


namespace roots_eq_two_iff_a_gt_neg1_l102_102200

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end roots_eq_two_iff_a_gt_neg1_l102_102200


namespace find_number_l102_102429

theorem find_number (x : ℝ) (h : 0.26 * x = 93.6) : x = 360 := sorry

end find_number_l102_102429


namespace oliver_spent_amount_l102_102530

theorem oliver_spent_amount :
  ∀ (S : ℕ), (33 - S + 32 = 61) → S = 4 :=
by
  sorry

end oliver_spent_amount_l102_102530


namespace shortest_distance_from_ln_curve_to_line_l102_102211

noncomputable def shortest_distance_curve_to_line : ℝ :=
  let y := λ x : ℝ, Real.log (x - 1),
      line := λ x y : ℝ, x - y + 2
  in 2 * Real.sqrt 2

theorem shortest_distance_from_ln_curve_to_line :
  ∀ x y : ℝ, y = Real.log (x - 1) → x - y + 2 = 0 →
  shortest_distance_curve_to_line = 2 * Real.sqrt 2 :=
by
  intros x y h hxy,
  exact sorry

end shortest_distance_from_ln_curve_to_line_l102_102211


namespace michael_robots_l102_102332

-- Conditions
def tom_robots := 3
def times_more := 4

-- Theorem to prove
theorem michael_robots : (times_more * tom_robots) + tom_robots = 15 := by
  sorry

end michael_robots_l102_102332


namespace blue_dress_difference_l102_102244

theorem blue_dress_difference 
(total_space : ℕ)
(red_dresses : ℕ)
(blue_dresses : ℕ)
(h1 : total_space = 200)
(h2 : red_dresses = 83)
(h3 : blue_dresses = total_space - red_dresses) :
blue_dresses - red_dresses = 34 :=
by
  rw [h1, h2] at h3
  sorry -- Proof details go here.

end blue_dress_difference_l102_102244


namespace cubic_sum_l102_102231

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cubic_sum_l102_102231


namespace sum_zero_opposites_l102_102237

theorem sum_zero_opposites {a b : ℝ} (h : a + b = 0) : a = -b :=
by sorry

end sum_zero_opposites_l102_102237


namespace meal_combinations_l102_102736

theorem meal_combinations (MenuA_items : ℕ) (MenuB_items : ℕ) : MenuA_items = 15 ∧ MenuB_items = 12 → MenuA_items * MenuB_items = 180 :=
by
  sorry

end meal_combinations_l102_102736


namespace find_difference_l102_102703

theorem find_difference (L S : ℕ) (h1: L = 2 * S + 3) (h2: L + S = 27) (h3: L = 19) : L - 2 * S = 3 :=
by
  sorry

end find_difference_l102_102703


namespace tan_double_angle_difference_l102_102635

variable {α β : Real}

theorem tan_double_angle_difference (h1 : Real.tan α = 1 / 2) (h2 : Real.tan (α - β) = 1 / 5) :
  Real.tan (2 * α - β) = 7 / 9 := 
sorry

end tan_double_angle_difference_l102_102635


namespace abs_diff_gt_half_prob_l102_102984

noncomputable def probability_abs_diff_gt_half : ℝ :=
  ((1 / 4) * (1 / 8) + 
   (1 / 8) * (1 / 2) + 
   (1 / 8) * 1) * 2

theorem abs_diff_gt_half_prob : probability_abs_diff_gt_half = 5 / 16 := by 
  sorry

end abs_diff_gt_half_prob_l102_102984


namespace total_notebooks_correct_l102_102131

-- Definitions based on conditions
def total_students : ℕ := 28
def half_students : ℕ := total_students / 2
def notebooks_per_student_group1 : ℕ := 5
def notebooks_per_student_group2 : ℕ := 3

-- Total notebooks calculation
def total_notebooks : ℕ :=
  (half_students * notebooks_per_student_group1) + (half_students * notebooks_per_student_group2)

-- Theorem to be proved
theorem total_notebooks_correct : total_notebooks = 112 := by
  sorry

end total_notebooks_correct_l102_102131


namespace inequality_solution_set_l102_102212

theorem inequality_solution_set :
  {x : ℝ | (x^2 - x - 6) / (x - 1) > 0} = {x : ℝ | (-2 < x ∧ x < 1) ∨ (3 < x)} := by
  sorry

end inequality_solution_set_l102_102212


namespace necessary_but_not_sufficient_condition_l102_102739

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ( (2*x - 1)*x = 0 → x = 0 ) ∧ ( x = 0 → (2*x - 1)*x = 0 ) :=
by
  sorry

end necessary_but_not_sufficient_condition_l102_102739


namespace cubic_identity_l102_102492

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 := 
by
  sorry

end cubic_identity_l102_102492


namespace ratio_of_areas_of_squares_l102_102995

theorem ratio_of_areas_of_squares (a b : ℕ) (hC : a = 24) (hD : b = 30) :
  (a^2 : ℚ) / (b^2 : ℚ) = 16 / 25 := 
by
  sorry

end ratio_of_areas_of_squares_l102_102995


namespace stickers_total_correct_l102_102723

-- Define the conditions
def stickers_per_page : ℕ := 10
def pages_total : ℕ := 22

-- Define the total number of stickers
def total_stickers : ℕ := pages_total * stickers_per_page

-- The statement we want to prove
theorem stickers_total_correct : total_stickers = 220 :=
by {
  sorry
}

end stickers_total_correct_l102_102723


namespace problem_statement_l102_102698

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l102_102698


namespace middle_part_division_l102_102047

theorem middle_part_division 
  (x : ℝ) 
  (x_pos : x > 0) 
  (H : x + (1 / 4) * x + (1 / 8) * x = 96) :
  (1 / 4) * x = 17 + 21 / 44 :=
by
  sorry

end middle_part_division_l102_102047


namespace selection_schemes_correct_l102_102796

-- Define the problem parameters
def number_of_selection_schemes (persons : ℕ) (cities : ℕ) (persons_cannot_visit : ℕ) : ℕ :=
  let choices_for_paris := persons - persons_cannot_visit
  let remaining_people := persons - 1
  choices_for_paris * remaining_people * (remaining_people - 1) * (remaining_people - 2)

-- Define the example constants
def total_people : ℕ := 6
def total_cities : ℕ := 4
def cannot_visit_paris : ℕ := 2

-- The statement to be proved
theorem selection_schemes_correct : 
  number_of_selection_schemes total_people total_cities cannot_visit_paris = 240 := by
  sorry

end selection_schemes_correct_l102_102796


namespace calculate_total_difference_in_miles_l102_102560

def miles_bus_a : ℝ := 1.25
def miles_walk_1 : ℝ := 0.35
def miles_bus_b : ℝ := 2.68
def miles_walk_2 : ℝ := 0.47
def miles_bus_c : ℝ := 3.27
def miles_walk_3 : ℝ := 0.21

def total_miles_on_buses : ℝ := miles_bus_a + miles_bus_b + miles_bus_c
def total_miles_walked : ℝ := miles_walk_1 + miles_walk_2 + miles_walk_3
def total_difference_in_miles : ℝ := total_miles_on_buses - total_miles_walked

theorem calculate_total_difference_in_miles :
  total_difference_in_miles = 6.17 := by
  sorry

end calculate_total_difference_in_miles_l102_102560


namespace patrick_savings_ratio_l102_102835

theorem patrick_savings_ratio (S : ℕ) (bike_cost : ℕ) (lent_amt : ℕ) (remaining_amt : ℕ)
  (h1 : bike_cost = 150)
  (h2 : lent_amt = 50)
  (h3 : remaining_amt = 25)
  (h4 : S = remaining_amt + lent_amt) :
  (S / bike_cost : ℚ) = 1 / 2 := 
sorry

end patrick_savings_ratio_l102_102835


namespace number_of_people_prefer_soda_l102_102390

-- Given conditions
def total_people : ℕ := 600
def central_angle_soda : ℝ := 198
def full_circle_angle : ℝ := 360

-- Problem statement
theorem number_of_people_prefer_soda : 
  (total_people : ℝ) * (central_angle_soda / full_circle_angle) = 330 := by
  sorry

end number_of_people_prefer_soda_l102_102390


namespace AM_GM_contradiction_l102_102069

open Real

theorem AM_GM_contradiction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
      ¬ (6 < a + 4 / b ∧ 6 < b + 9 / c ∧ 6 < c + 16 / a) := by
  sorry

end AM_GM_contradiction_l102_102069


namespace find_p_q_sum_l102_102120

theorem find_p_q_sum (p q : ℝ) 
  (sum_condition : p / 3 = 8) 
  (product_condition : q / 3 = 12) : 
  p + q = 60 :=
by
  sorry

end find_p_q_sum_l102_102120


namespace cone_volume_l102_102022

theorem cone_volume (V_cylinder V_frustum V_cone : ℝ)
  (h₁ : V_cylinder = 9)
  (h₂ : V_frustum = 63) :
  V_cone = 64 :=
sorry

end cone_volume_l102_102022


namespace base_sequence_count_l102_102866

theorem base_sequence_count :
  let A := 1
  let C := 2
  let G := 3
  (nat.choose 6 A) * (nat.choose 5 C) * (nat.choose 3 G) = 60 :=
by
  sorry

end base_sequence_count_l102_102866


namespace find_x_plus_y_l102_102661

theorem find_x_plus_y :
  ∀ (x y : ℝ), (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 :=
by
  intros x y h
  sorry

end find_x_plus_y_l102_102661


namespace number_of_three_digit_integers_l102_102487

def digits : Finset Nat := {1, 3, 5, 9}

def count_distinct_three_digit_integers (s : Finset Nat) : Nat :=
  (s.card) * (s.card - 1) * (s.card - 2)

theorem number_of_three_digit_integers :
  count_distinct_three_digit_integers digits = 24 :=
by
  -- Proof would go here
  sorry

end number_of_three_digit_integers_l102_102487


namespace sets_tossed_per_show_l102_102620

-- Definitions
def sets_used_per_show : ℕ := 5
def number_of_shows : ℕ := 30
def total_sets_used : ℕ := 330

-- Statement to prove
theorem sets_tossed_per_show : 
  (total_sets_used - (sets_used_per_show * number_of_shows)) / number_of_shows = 6 := 
by
  sorry

end sets_tossed_per_show_l102_102620


namespace minimum_pipe_length_l102_102751

theorem minimum_pipe_length 
  (M S : ℝ × ℝ) 
  (horiz_dist : abs (M.1 - S.1) = 160)
  (vert_dist : abs (M.2 - S.2) = 120) :
  dist M S = 200 :=
by {
  sorry
}

end minimum_pipe_length_l102_102751


namespace two_roots_iff_a_greater_than_neg1_l102_102194

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l102_102194


namespace simplest_square_root_l102_102878

theorem simplest_square_root :
  ∀ (a b c d : Real),
  a = Real.sqrt 0.2 →
  b = Real.sqrt (1 / 2) →
  c = Real.sqrt 6 →
  d = Real.sqrt 12 →
  c = Real.sqrt 6 :=
by
  intros a b c d ha hb hc hd
  simp [ha, hb, hc, hd]
  sorry

end simplest_square_root_l102_102878


namespace intersection_of_A_and_B_l102_102222

open Set

noncomputable def A : Set ℝ := { x | (x - 2) / (x + 5) < 0 }
noncomputable def B : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | -5 < x ∧ x ≤ -1 } :=
sorry

end intersection_of_A_and_B_l102_102222


namespace gcf_of_36_and_54_l102_102567

theorem gcf_of_36_and_54 : Nat.gcd 36 54 = 18 := 
by
  sorry

end gcf_of_36_and_54_l102_102567


namespace problem1_problem2_l102_102173

-- Equivalent proof statement for part (1)
theorem problem1 : 2023^2 - 2022 * 2024 = 1 := by
  sorry

-- Equivalent proof statement for part (2)
theorem problem2 (m : ℝ) (h : m ≠ 1) (h1 : m ≠ -1) : 
  (m / (m^2 - 1)) / ((m^2 - m) / (m^2 - 2*m + 1)) = 1 / (m + 1) := by
  sorry

end problem1_problem2_l102_102173


namespace range_of_a_l102_102375

theorem range_of_a (a : ℝ) (h : ¬ ∃ t : ℝ, t^2 - a * t - a < 0) : -4 ≤ a ∧ a ≤ 0 :=
by 
  sorry

end range_of_a_l102_102375


namespace percent_preferred_apples_l102_102543

def frequencies : List ℕ := [75, 80, 45, 100, 50]
def frequency_apples : ℕ := 75
def total_frequency : ℕ := frequency_apples + frequencies[1] + frequencies[2] + frequencies[3] + frequencies[4]

theorem percent_preferred_apples :
  (frequency_apples * 100) / total_frequency = 21 := by
  -- Proof steps go here
  sorry

end percent_preferred_apples_l102_102543


namespace stock_yield_calculation_l102_102018

theorem stock_yield_calculation (par_value market_value annual_dividend : ℝ)
  (h1 : par_value = 100)
  (h2 : market_value = 80)
  (h3 : annual_dividend = 0.04 * par_value) :
  (annual_dividend / market_value) * 100 = 5 :=
by
  sorry

end stock_yield_calculation_l102_102018


namespace graph_shift_sine_l102_102725

theorem graph_shift_sine :
  ∀ x : ℝ, y = sin (2 * x + π / 4) ↔ y = sin (2 * (x + π / 8)) :=
by
  sorry

end graph_shift_sine_l102_102725


namespace balloon_count_l102_102130

theorem balloon_count (gold_balloon silver_balloon black_balloon blue_balloon green_balloon total_balloon : ℕ) (h1 : gold_balloon = 141) 
                      (h2 : silver_balloon = (gold_balloon / 3) * 5) 
                      (h3 : black_balloon = silver_balloon / 2) 
                      (h4 : blue_balloon = black_balloon / 2) 
                      (h5 : green_balloon = (blue_balloon / 4) * 3) 
                      (h6 : total_balloon = gold_balloon + silver_balloon + black_balloon + blue_balloon + green_balloon): 
                      total_balloon = 593 :=
by 
  sorry

end balloon_count_l102_102130


namespace sum_of_possible_remainders_l102_102113

theorem sum_of_possible_remainders (n : ℕ) (h_even : ∃ k : ℕ, n = 2 * k) : 
  let m := 1000 * (2 * n + 6) + 100 * (2 * n + 4) + 10 * (2 * n + 2) + (2 * n)
  let remainder (k : ℕ) := (1112 * k + 6420) % 29
  23 + 7 + 20 = 50 :=
  by
  sorry

end sum_of_possible_remainders_l102_102113


namespace segment_equality_l102_102480

variables {Point : Type} [AddGroup Point]

-- Define the points A, B, C, D, E, F
variables (A B C D E F : Point)

-- Given conditions
variables (AC CE BD DF AD CF : Point)
variable (h1 : AC = CE)
variable (h2 : BD = DF)
variable (h3 : AD = CF)

-- Theorem statement
theorem segment_equality (h1 : A - C = C - E)
                         (h2 : B - D = D - F)
                         (h3 : A - D = C - F) :
  (C - D) = (A - B) ∧ (C - D) = (E - F) :=
by
  sorry

end segment_equality_l102_102480


namespace nth_equation_l102_102100

theorem nth_equation (n : ℕ) : 
  1 + 6 * n = (3 * n + 1) ^ 2 - 9 * n ^ 2 := 
by 
  sorry

end nth_equation_l102_102100


namespace basketball_points_total_l102_102498

variable (Tobee_points Jay_points Sean_points Remy_points Alex_points : ℕ)

def conditions := 
  Tobee_points = 4 ∧
  Jay_points = 2 * Tobee_points + 6 ∧
  Sean_points = Jay_points / 2 ∧
  Remy_points = Tobee_points + Jay_points - 3 ∧
  Alex_points = Sean_points + Remy_points + 4

theorem basketball_points_total 
  (h : conditions Tobee_points Jay_points Sean_points Remy_points Alex_points) :
  Tobee_points + Jay_points + Sean_points + Remy_points + Alex_points = 66 :=
by sorry

end basketball_points_total_l102_102498


namespace total_sample_size_is_72_l102_102154

-- Definitions based on the given conditions:
def production_A : ℕ := 600
def production_B : ℕ := 1200
def production_C : ℕ := 1800
def total_production : ℕ := production_A + production_B + production_C
def sampled_B : ℕ := 2

-- Main theorem to prove the sample size:
theorem total_sample_size_is_72 : 
  ∃ (n : ℕ), 
    (∃ s_A s_B s_C, 
      s_A = (production_A * sampled_B * total_production) / production_B^2 ∧ 
      s_B = sampled_B ∧ 
      s_C = (production_C * sampled_B * total_production) / production_B^2 ∧
      n = s_A + s_B + s_C) ∧ 
  (n = 72) :=
sorry

end total_sample_size_is_72_l102_102154


namespace mike_pumpkins_l102_102705

def pumpkins : ℕ :=
  let sandy_pumpkins := 51
  let total_pumpkins := 74
  total_pumpkins - sandy_pumpkins

theorem mike_pumpkins : pumpkins = 23 :=
by
  sorry

end mike_pumpkins_l102_102705


namespace evaluate_expressions_for_pos_x_l102_102054

theorem evaluate_expressions_for_pos_x :
  (∀ x : ℝ, x > 0 → 6^x * x^3 = 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → (3 * x)^(3 * x) ≠ 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → 3^x * x^6 ≠ 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → (6 * x)^x ≠ 6^x * x^3) →
  ∃ n : ℕ, n = 1 := 
by
  sorry

end evaluate_expressions_for_pos_x_l102_102054


namespace conjunction_used_in_proposition_l102_102087

theorem conjunction_used_in_proposition (x : ℝ) (h : x^2 = 4) :
  (x = 2 ∨ x = -2) :=
sorry

end conjunction_used_in_proposition_l102_102087


namespace min_value_of_z_l102_102521

-- Define the conditions as separate hypotheses.
variable (x y : ℝ)

def condition1 : Prop := x - y + 1 ≥ 0
def condition2 : Prop := x + y - 1 ≥ 0
def condition3 : Prop := x ≤ 3

-- Define the objective function.
def z : ℝ := 2 * x - 3 * y

-- State the theorem to prove the minimum value of z given the conditions.
theorem min_value_of_z (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x) :
  ∃ x y, condition1 x y ∧ condition2 x y ∧ condition3 x ∧ z x y = -6 :=
sorry

end min_value_of_z_l102_102521


namespace roots_sum_product_l102_102123

theorem roots_sum_product (p q : ℝ) (h_sum : p / 3 = 8) (h_prod : q / 3 = 12) : p + q = 60 := 
by 
  sorry

end roots_sum_product_l102_102123


namespace possible_values_f_one_l102_102683

noncomputable def f (x : ℝ) : ℝ := sorry

variables (a b : ℝ)
axiom f_equation : ∀ x y : ℝ, 
  f ((x - y) ^ 2) = a * (f x)^2 - 2 * x * f y + b * y^2

theorem possible_values_f_one : f 1 = 1 ∨ f 1 = 2 :=
sorry

end possible_values_f_one_l102_102683


namespace initial_balance_l102_102966

theorem initial_balance (X : ℝ) : 
  (X - 60 - 30 - 0.25 * (X - 60 - 30) - 10 = 100) ↔ (X = 236.67) := 
  by
    sorry

end initial_balance_l102_102966


namespace joan_spent_on_thursday_l102_102700

theorem joan_spent_on_thursday : 
  ∀ (n : ℕ), 
  2 * (4 + n) = 18 → 
  n = 14 := 
by 
  sorry

end joan_spent_on_thursday_l102_102700


namespace find_radius_of_sector_l102_102884

noncomputable def radius_of_sector (P : ℝ) (θ : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem find_radius_of_sector :
  radius_of_sector 144 180 = 144 / (Real.pi + 2) :=
by
  unfold radius_of_sector
  sorry

end find_radius_of_sector_l102_102884


namespace intersection_of_A_and_B_l102_102368

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, (x ∈ A ∩ B) ↔ (x = 0 ∨ x = 1) := by
  sorry

end intersection_of_A_and_B_l102_102368


namespace cylindrical_can_increase_l102_102431

theorem cylindrical_can_increase (R H y : ℝ)
  (h₁ : R = 5)
  (h₂ : H = 4)
  (h₃ : π * (R + y)^2 * (H + y) = π * (R + 2*y)^2 * H) :
  y = Real.sqrt 76 - 5 :=
by
  sorry

end cylindrical_can_increase_l102_102431


namespace sacks_per_day_proof_l102_102382

-- Definitions based on the conditions in the problem
def totalUnripeOranges : ℕ := 1080
def daysOfHarvest : ℕ := 45

-- Mathematical statement to prove
theorem sacks_per_day_proof : totalUnripeOranges / daysOfHarvest = 24 :=
by sorry

end sacks_per_day_proof_l102_102382


namespace matrix_determinant_equality_l102_102967

open Complex Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_determinant_equality (A B : Matrix n n ℂ) (x : ℂ) 
  (h1 : A ^ 2 + B ^ 2 = 2 * A * B) :
  det (A - x • 1) = det (B - x • 1) :=
  sorry

end matrix_determinant_equality_l102_102967


namespace number_of_exercise_books_l102_102425

theorem number_of_exercise_books (pencils pens exercise_books : ℕ) (h_ratio : (14 * pens = 4 * pencils) ∧ (14 * exercise_books = 3 * pencils)) (h_pencils : pencils = 140) : exercise_books = 30 :=
by
  sorry

end number_of_exercise_books_l102_102425


namespace relation_between_x_and_y_l102_102387

open Real

noncomputable def x (t : ℝ) : ℝ := t^(1 / (t - 1))
noncomputable def y (t : ℝ) : ℝ := t^(t / (t - 1))

theorem relation_between_x_and_y (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1) : (y t)^(x t) = (x t)^(y t) :=
by sorry

end relation_between_x_and_y_l102_102387


namespace max_value_of_f_l102_102631

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∀ x : ℝ, x > 0 → f x ≤ (Real.log (Real.exp 1)) / (Real.exp 1) :=
by
  sorry

end max_value_of_f_l102_102631


namespace Yihana_uphill_walking_time_l102_102001

theorem Yihana_uphill_walking_time :
  let t1 := 3
  let t2 := 2
  let t_total := t1 + t2
  t_total = 5 :=
by
  let t1 := 3
  let t2 := 2
  let t_total := t1 + t2
  show t_total = 5
  sorry

end Yihana_uphill_walking_time_l102_102001


namespace min_value_ineq_least_3_l102_102639

noncomputable def min_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) : ℝ :=
  1 / (x + y) + (x + y) / z

theorem min_value_ineq_least_3 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  min_value_ineq x y z h1 h2 h3 h4 ≥ 3 :=
sorry

end min_value_ineq_least_3_l102_102639


namespace solution_range_l102_102609

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end solution_range_l102_102609


namespace tile_difference_correct_l102_102397

def initial_blue_tiles := 23
def initial_green_tiles := 16
def first_border_green_tiles := 6 * 1
def second_border_green_tiles := 6 * 2
def total_green_tiles := initial_green_tiles + first_border_green_tiles + second_border_green_tiles
def difference_tiling := total_green_tiles - initial_blue_tiles

theorem tile_difference_correct : difference_tiling = 11 := by
  sorry

end tile_difference_correct_l102_102397


namespace polynomial_multiplication_l102_102857

theorem polynomial_multiplication (x a : ℝ) : (x - a) * (x^2 + a * x + a^2) = x^3 - a^3 :=
by
  sorry

end polynomial_multiplication_l102_102857


namespace problem_statement_l102_102523

variable {R : Type*} [LinearOrderedField R]

theorem problem_statement
  (x1 x2 x3 y1 y2 y3 : R)
  (h1 : x1 + x2 + x3 = 0)
  (h2 : y1 + y2 + y3 = 0)
  (h3 : x1 * y1 + x2 * y2 + x3 * y3 = 0)
  (h4 : (x1^2 + x2^2 + x3^2) * (y1^2 + y2^2 + y3^2) > 0) :
  (x1^2 / (x1^2 + x2^2 + x3^2) + y1^2 / (y1^2 + y2^2 + y3^2) = 2 / 3) := 
sorry

end problem_statement_l102_102523


namespace arthur_walking_distance_l102_102170

/-- Arthur walks 8 blocks west and 10 blocks south, 
    each block being 1/4 mile -/
theorem arthur_walking_distance 
  (blocks_west : ℕ) (blocks_south : ℕ) (block_distance : ℚ)
  (h1 : blocks_west = 8) (h2 : blocks_south = 10) (h3 : block_distance = 1/4) :
  (blocks_west + blocks_south) * block_distance = 4.5 := 
by
  sorry

end arthur_walking_distance_l102_102170


namespace commission_percentage_proof_l102_102758

-- Let's define the problem conditions in Lean

-- Condition 1: Commission on first Rs. 10,000
def commission_first_10000 (sales : ℕ) : ℕ :=
  if sales ≤ 10000 then
    5 * sales / 100
  else
    500

-- Condition 2: Amount remitted to company after commission
def amount_remitted (total_sales : ℕ) (commission : ℕ) : ℕ :=
  total_sales - commission

-- Condition 3: Function to calculate commission on exceeding amount
def commission_exceeding (sales : ℕ) (x : ℕ) : ℕ :=
  x * sales / 100

-- The main hypothesis as per the given problem
def correct_commission_percentage (total_sales : ℕ) (remitted : ℕ) (x : ℕ) :=
  commission_first_10000 10000 + commission_exceeding (total_sales - 10000) x
  = total_sales - remitted

-- Problem statement to prove the percentage of commission on exceeding Rs. 10,000 is 4%
theorem commission_percentage_proof : correct_commission_percentage 32500 31100 4 := 
  by sorry

end commission_percentage_proof_l102_102758


namespace find_digits_l102_102914

theorem find_digits (A B C D : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
(h2 : 1 ≤ A ∧ A ≤ 9)
(h3 : 1 ≤ B ∧ B ≤ 9)
(h4 : 1 ≤ C ∧ C ≤ 9)
(h5 : 1 ≤ D ∧ D ≤ 9)
(h6 : (10 * A + B) * (10 * C + B) = 111 * D)
(h7 : (10 * A + B) < (10 * C + B)) :
A = 2 ∧ B = 7 ∧ C = 3 ∧ D = 9 :=
sorry

end find_digits_l102_102914


namespace eggs_per_box_l102_102743

-- Conditions
def num_eggs : ℝ := 3.0
def num_boxes : ℝ := 2.0

-- Theorem statement
theorem eggs_per_box (h1 : num_eggs = 3.0) (h2 : num_boxes = 2.0) : (num_eggs / num_boxes = 1.5) :=
sorry

end eggs_per_box_l102_102743


namespace complement_P_subset_PQ_intersection_PQ_eq_Q_l102_102223

open Set

variable {R : Type*} [OrderedCommRing R]

def P (x : R) : Prop := -2 ≤ x ∧ x ≤ 10
def Q (m x : R) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem complement_P : (compl (setOf P)) = {x | x < -2} ∪ {x | x > 10} :=
by {
  sorry
}

theorem subset_PQ (m : R) : (∀ x, P x → Q m x) ↔ m ≥ 9 :=
by {
  sorry
}

theorem intersection_PQ_eq_Q (m : R) : (∀ x, Q m x → P x) ↔ m ≤ 9 :=
by {
  sorry
}

end complement_P_subset_PQ_intersection_PQ_eq_Q_l102_102223


namespace original_number_unique_l102_102303

theorem original_number_unique (x : ℝ) (h_pos : 0 < x) 
  (h_condition : 100 * x = 9 / x) : x = 3 / 10 :=
by
  sorry

end original_number_unique_l102_102303


namespace similar_triangles_perimeter_l102_102289

theorem similar_triangles_perimeter
  {k : ℕ} (h_ratio : 3 = 3) (p_small : 12 = 12) :
  let p_large := 20
  in p_large = 20 := by
  sorry

end similar_triangles_perimeter_l102_102289


namespace square_of_99_l102_102426

theorem square_of_99 : 99 * 99 = 9801 :=
by sorry

end square_of_99_l102_102426


namespace find_a_for_min_l102_102376

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 - 6 * a * x + 2

theorem find_a_for_min {a x0 : ℝ} (hx0 : 1 < x0 ∧ x0 < 3) (h : ∀ x : ℝ, deriv (f a) x0 = 0) : a = -2 :=
by
  sorry

end find_a_for_min_l102_102376


namespace max_PA_PB_l102_102217

noncomputable def max_distance (PA PB : ℝ) : ℝ :=
  PA + PB

theorem max_PA_PB {A B : ℝ × ℝ} (m : ℝ) :
  A = (0, 0) ∧
  B = (1, 3) ∧
  dist A B = 10 →
  max_distance (dist A B) (dist (1, 3) B) = 2 * Real.sqrt 5 :=
by
  sorry

end max_PA_PB_l102_102217


namespace parabola_ellipse_focus_l102_102931

theorem parabola_ellipse_focus (p : ℝ) :
  (∃ (x y : ℝ), x^2 = 2 * p * y ∧ y = -1 ∧ x = 0) →
  p = -2 :=
by
  sorry

end parabola_ellipse_focus_l102_102931


namespace probability_pink_correct_l102_102885

def total_flowers_a : ℕ := 6 + 3
def total_flowers_b : ℕ := 2 + 7

def pink_flowers_a : ℕ := 3
def pink_flowers_b : ℕ := 7

def probability_pink_a : ℚ := pink_flowers_a / total_flowers_a
def probability_pink_b : ℚ := pink_flowers_b / total_flowers_b

def probability_pink : ℚ := (probability_pink_a + probability_pink_b) / 2

theorem probability_pink_correct : probability_pink = 5 / 9 := by
  sorry

end probability_pink_correct_l102_102885


namespace area_of_four_triangles_l102_102991

theorem area_of_four_triangles (a b : ℕ) (h1 : 2 * b = 28) (h2 : a + 2 * b = 30) :
    4 * (1 / 2 * a * b) = 56 := by
  sorry

end area_of_four_triangles_l102_102991


namespace binary_add_mul_l102_102619

def x : ℕ := 0b101010
def y : ℕ := 0b11010
def z : ℕ := 0b1110
def result : ℕ := 0b11000000000

theorem binary_add_mul : ((x + y) * z) = result := by
  sorry

end binary_add_mul_l102_102619


namespace logistics_center_correct_l102_102953

noncomputable def rectilinear_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-6, 9)
def C : ℝ × ℝ := (-3, -8)

def logistics_center : ℝ × ℝ := (-5, 0)

theorem logistics_center_correct : 
  ∀ L : ℝ × ℝ, 
  (rectilinear_distance L A = rectilinear_distance L B) ∧ 
  (rectilinear_distance L B = rectilinear_distance L C) ∧
  (rectilinear_distance L A = rectilinear_distance L C) → 
  L = logistics_center := sorry

end logistics_center_correct_l102_102953


namespace arccos_cos_9_eq_2_717_l102_102903

-- Statement of the proof problem
theorem arccos_cos_9_eq_2_717 : Real.arccos (Real.cos 9) = 2.717 :=
by
  sorry

end arccos_cos_9_eq_2_717_l102_102903


namespace roots_distribution_l102_102455

noncomputable def polynomial_roots : Polynomial ℝ :=
  Polynomial.Coeff (x^3 + 3x^2 - 4x + 12)
  
theorem roots_distribution : 
  (polynomial_roots.has_one_positive_real_root ∧ polynomial_roots.has_two_negative_real_roots) :=
sorry

end roots_distribution_l102_102455


namespace double_inequality_pos_reals_equality_condition_l102_102147

theorem double_inequality_pos_reals (x y z : ℝ) (x_pos: 0 < x) (y_pos: 0 < y) (z_pos: 0 < z):
  0 < (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ≤ (1 / 8) :=
  sorry

theorem equality_condition (x y z : ℝ) :
  ((1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) = (1 / 8)) ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
  sorry

end double_inequality_pos_reals_equality_condition_l102_102147


namespace total_cats_received_l102_102448

-- Defining the constants and conditions
def total_adult_cats := 150
def fraction_female_cats := 2 / 3
def fraction_litters := 2 / 5
def kittens_per_litter := 5

-- Defining the proof problem
theorem total_cats_received :
  let number_female_cats := (fraction_female_cats * total_adult_cats : ℤ)
  let number_litters := (fraction_litters * number_female_cats : ℤ)
  let number_kittens := number_litters * kittens_per_litter
  number_female_cats + number_kittens + (total_adult_cats - number_female_cats) = 350 := 
by
  sorry

end total_cats_received_l102_102448


namespace lily_typing_break_time_l102_102258

theorem lily_typing_break_time :
  ∃ t : ℝ, (15 * t + 15 * t = 255) ∧ (19 = 2 * t + 2) ∧ (t = 8) := 
sorry

end lily_typing_break_time_l102_102258


namespace imaginary_part_of_i_mul_root_l102_102813

theorem imaginary_part_of_i_mul_root
  (z : ℂ) (hz : z^2 - 4 * z + 5 = 0) : (i * z).im = 2 := 
sorry

end imaginary_part_of_i_mul_root_l102_102813


namespace range_of_a_l102_102825

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (2^x - 2^(-x))
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * (2^x + 2^(-x))

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * f x + g (2 * x) ≥ 0) ↔ a ≥ -17 / 6 :=
by
  sorry

end range_of_a_l102_102825


namespace power_increased_by_four_l102_102894

-- Definitions from the conditions
variables (F k v : ℝ) (initial_force_eq_resistive : F = k * v)

-- Define the new conditions with double the force
variables (new_force : ℝ) (new_velocity : ℝ) (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F)

-- The theorem statement
theorem power_increased_by_four (initial_force_eq_resistive : F = k * v) 
  (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F) :
  new_velocity = 2 * v → 
  (new_force * new_velocity) = 4 * (F * v) :=
sorry

end power_increased_by_four_l102_102894


namespace line_eq1_line_eq2_l102_102158

-- Define the line equations
def l1 (x y : ℝ) : Prop := 4 * x + y + 6 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 6 = 0

-- Theorem for when midpoint is at (0, 0)
theorem line_eq1 : ∀ x y : ℝ, (x + 6 * y = 0) ↔
  ∃ (a : ℝ), 
    l1 a (-(a / 6)) ∧
    l2 (-a) ((a / 6)) ∧
    (a + -a = 0) ∧ (-(a / 6) + a / 6 = 0) := 
by 
  sorry

-- Theorem for when midpoint is at (0, 1)
theorem line_eq2 : ∀ x y : ℝ, (x + 2 * y - 2 = 0) ↔
  ∃ (b : ℝ),
    l1 b (-b / 2 + 1) ∧
    l2 (-b) (1 - (-b / 2)) ∧
    (b + -b = 0) ∧ (-b / 2 + 1 + (1 - (-b / 2)) = 2) := 
by 
  sorry

end line_eq1_line_eq2_l102_102158


namespace range_of_m_l102_102236

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, abs (x - m) < 1 ↔ (1/3 < x ∧ x < 1/2)) ↔ (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  sorry

end range_of_m_l102_102236


namespace shaded_regions_area_sum_l102_102778

theorem shaded_regions_area_sum (side_len : ℚ) (radius : ℚ) (a b c : ℤ) :
  side_len = 16 → radius = side_len / 2 →
  a = (64 / 3) ∧ b = 32 ∧ c = 3 →
  (∃ x : ℤ, x = a + b + c ∧ x = 99) :=
by
  intros hside_len hradius h_constituents
  sorry

end shaded_regions_area_sum_l102_102778


namespace count_valid_n_l102_102249

theorem count_valid_n :
  let q_range := Finset.Icc 200 999
  let r_range := Finset.Icc 0 99
  ∃ (n : ℕ), n ∈ q_range ∧ n ∈ r_range ∧
             ∃ (count_n : ℕ), count_n = 6400 ∧
             ∀ (n' : ℕ), (∃ q r, n' = 100 * q + r ∧ q ∈ q_range ∧ r ∈ r_range ∧ (q + r) % 13 = 0) → count_n = 6400 :=
begin
  sorry
end

end count_valid_n_l102_102249


namespace find_m_n_and_max_value_l102_102642

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + 3 * m + n

-- Define a predicate for the function being even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define the conditions and what we want to prove
theorem find_m_n_and_max_value :
  ∀ m n : ℝ,
    is_even_function (f m n) →
    (m - 1 ≤ 2 * m) →
      (m = 1 / 3 ∧ n = 0) ∧ 
      (∀ x : ℝ, -2 / 3 ≤ x ∧ x ≤ 2 / 3 → f (1/3) 0 x ≤ 31 / 27) :=
by
  sorry

end find_m_n_and_max_value_l102_102642


namespace problem_statement_l102_102697

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l102_102697


namespace find_a4_l102_102860

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1) / 2) * d

theorem find_a4 (a d : ℤ)
    (h₁ : sum_first_n_terms a d 5 = 15)
    (h₂ : sum_first_n_terms a d 9 = 63) :
  arithmetic_sequence a d 4 = 5 :=
sorry

end find_a4_l102_102860


namespace angle_A_in_triangle_l102_102083

theorem angle_A_in_triangle :
  ∀ (A B C : ℝ) (a b c : ℝ),
  a = 2 * Real.sqrt 3 → b = 2 * Real.sqrt 2 → B = π / 4 → 
  (A = π / 3 ∨ A = 2 * π / 3) :=
by
  intros A B C a b c ha hb hB
  sorry

end angle_A_in_triangle_l102_102083


namespace dodecahedron_edges_l102_102559

noncomputable def regular_dodecahedron := Type

def faces : regular_dodecahedron → ℕ := λ _ => 12
def edges_per_face : regular_dodecahedron → ℕ := λ _ => 5
def shared_edges : regular_dodecahedron → ℕ := λ _ => 2

theorem dodecahedron_edges (d : regular_dodecahedron) :
  (faces d * edges_per_face d) / shared_edges d = 30 :=
by
  sorry

end dodecahedron_edges_l102_102559


namespace simplify_polynomial_l102_102270

variable {R : Type} [CommRing R] (s : R)

theorem simplify_polynomial :
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 4) = -4 * s + 1 :=
by
  sorry

end simplify_polynomial_l102_102270


namespace started_with_l102_102899

-- Define the conditions
def total_eggs : ℕ := 70
def bought_eggs : ℕ := 62

-- Define the statement to prove
theorem started_with (initial_eggs : ℕ) : initial_eggs = total_eggs - bought_eggs → initial_eggs = 8 := by
  intro h
  sorry

end started_with_l102_102899


namespace find_fraction_l102_102319

-- Let f be a real number representing the fraction
theorem find_fraction (f : ℝ) (h : f * 12 + 5 = 11) : f = 1 / 2 := 
by
  sorry

end find_fraction_l102_102319


namespace carl_city_mileage_l102_102340

noncomputable def city_mileage (miles_city mpg_highway cost_per_gallon total_cost miles_highway : ℝ) : ℝ :=
  let total_gallons := total_cost / cost_per_gallon
  let gallons_highway := miles_highway / mpg_highway
  let gallons_city := total_gallons - gallons_highway
  miles_city / gallons_city

theorem carl_city_mileage :
  city_mileage 60 40 3 42 200 = 20 / 3 := by
  sorry

end carl_city_mileage_l102_102340


namespace quiz_score_difference_l102_102833

theorem quiz_score_difference :
  let percentage_70 := 0.10
  let percentage_80 := 0.35
  let percentage_90 := 0.30
  let percentage_100 := 0.25
  let mean_score := (percentage_70 * 70) + (percentage_80 * 80) + (percentage_90 * 90) + (percentage_100 * 100)
  let median_score := 90
  mean_score = 87 → median_score - mean_score = 3 :=
by
  sorry

end quiz_score_difference_l102_102833


namespace train_speed_is_30_kmh_l102_102582

noncomputable def speed_of_train (train_length : ℝ) (cross_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / cross_time
  let train_speed_ms := relative_speed + man_speed_ms
  train_speed_ms * (3600 / 1000)

theorem train_speed_is_30_kmh :
  speed_of_train 400 59.99520038396929 6 = 30 :=
by
  -- Using the approximation mentioned in the solution, hence no computation proof required.
  sorry

end train_speed_is_30_kmh_l102_102582


namespace exponent_inequality_l102_102632

theorem exponent_inequality (a b c : ℝ) (h1 : a ≠ 1) (h2 : b ≠ 1) (h3 : c ≠ 1) (h4 : a > b) (h5 : b > c) (h6 : c > 0) : a ^ b > c ^ b :=
  sorry

end exponent_inequality_l102_102632


namespace union_complement_eq_l102_102687

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l102_102687


namespace part1_part2_l102_102477

noncomputable def point_M (m : ℝ) : ℝ × ℝ := (2 * m + 1, m - 4)
def point_N : ℝ × ℝ := (5, 2)

theorem part1 (m : ℝ) (h : m - 4 = 2) : point_M m = (13, 2) := by
  sorry

theorem part2 (m : ℝ) (h : 2 * m + 1 = 3) : point_M m = (3, -3) := by
  sorry

end part1_part2_l102_102477


namespace evaluate_seventy_two_square_minus_twenty_four_square_l102_102462

theorem evaluate_seventy_two_square_minus_twenty_four_square :
  72 ^ 2 - 24 ^ 2 = 4608 := 
by {
  sorry
}

end evaluate_seventy_two_square_minus_twenty_four_square_l102_102462


namespace symmetric_point_l102_102849

theorem symmetric_point : ∃ (x0 y0 : ℝ), 
  (x0 = -6 ∧ y0 = -3) ∧ 
  (∃ (m1 m2 : ℝ), 
    m1 = -1 ∧ 
    m2 = (y0 - 2) / (x0 + 1) ∧ 
    m1 * m2 = -1) ∧ 
  (∃ (x_mid y_mid : ℝ), 
    x_mid = (x0 - 1) / 2 ∧ 
    y_mid = (y0 + 2) / 2 ∧ 
    x_mid + y_mid + 4 = 0) := 
sorry

end symmetric_point_l102_102849


namespace product_modulo_7_l102_102777

theorem product_modulo_7 : 
  (2007 % 7 = 4) ∧ (2008 % 7 = 5) ∧ (2009 % 7 = 6) ∧ (2010 % 7 = 0) →
  (2007 * 2008 * 2009 * 2010) % 7 = 0 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end product_modulo_7_l102_102777


namespace original_decimal_number_l102_102301

theorem original_decimal_number (x : ℝ) (h₁ : 0 < x) (h₂ : 100 * x = 9 * (1 / x)) : x = 3 / 10 :=
by
  sorry

end original_decimal_number_l102_102301


namespace translate_down_three_units_l102_102954

def original_function (x : ℝ) : ℝ := 3 * x + 2

def translated_function (x : ℝ) : ℝ := 3 * x - 1

theorem translate_down_three_units :
  ∀ x : ℝ, translated_function x = original_function x - 3 :=
by
  intro x
  simp [original_function, translated_function]
  sorry

end translate_down_three_units_l102_102954


namespace sequence_problem_l102_102637

theorem sequence_problem
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : 1 + a1 + a1 = a1 + a1)
  (h2 : b1 * b1 = b2)
  (h3 : 4 = b2 * b2):
  (a1 + a2) / b2 = 2 :=
by
  -- The proof would go here
  sorry

end sequence_problem_l102_102637


namespace determine_m_l102_102075

theorem determine_m (m : ℝ) : (∀ x : ℝ, (m * x = 1 → x = 1 ∨ x = -1)) ↔ (m = 0 ∨ m = 1 ∨ m = -1) :=
by sorry

end determine_m_l102_102075


namespace triangle_perimeter_l102_102440

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 30) 
  (h2 : b = 10) 
  (h3 : c = real.sqrt (a^2 + b^2)) 
  (h4 : (1 / 2) * a * b = 150) : 
  a + b + c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end triangle_perimeter_l102_102440


namespace function_through_point_l102_102276

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x

theorem function_through_point (a : ℝ) (x : ℝ) (hx : (2 : ℝ) = x) (h : f 2 a = 4) : f x 2 = 2^x :=
by sorry

end function_through_point_l102_102276


namespace problem_l102_102660

theorem problem (x y : ℝ) (h : (3 * x - y + 5)^2 + |2 * x - y + 3| = 0) : x + y = -3 := 
by
  sorry

end problem_l102_102660


namespace minimum_value_sum_l102_102179

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / (3 * b) + b / (5 * c) + c / (6 * a)) >= (3 / (90^(1/3))) :=
by 
  sorry

end minimum_value_sum_l102_102179


namespace car_sharing_problem_l102_102392

theorem car_sharing_problem 
  (x : ℕ)
  (cond1 : ∃ c : ℕ, x = 4 * c + 4)
  (cond2 : ∃ c : ℕ, x = 3 * c + 9):
  (x / 4 + 1 = (x - 9) / 3) :=
by sorry

end car_sharing_problem_l102_102392


namespace alice_commission_percentage_l102_102035

-- Definitions from the given problem
def basic_salary : ℝ := 240
def total_sales : ℝ := 2500
def savings : ℝ := 29
def savings_percentage : ℝ := 0.10

-- The target percentage we want to prove
def commission_percentage : ℝ := 0.02

-- The statement we aim to prove
theorem alice_commission_percentage :
  commission_percentage =
  (savings / savings_percentage - basic_salary) / total_sales := 
sorry

end alice_commission_percentage_l102_102035


namespace probability_rain_at_most_3_days_in_july_l102_102284

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_rain_at_most_3_days_in_july :
  let p := 1 / 5
  let n := 31
  let sum_prob := binomial_probability n 0 p + binomial_probability n 1 p + binomial_probability n 2 p + binomial_probability n 3 p
  abs (sum_prob - 0.125) < 0.001 :=
by
  sorry

end probability_rain_at_most_3_days_in_july_l102_102284


namespace ways_to_divide_day_l102_102155

theorem ways_to_divide_day : 
  ∃ nm_count: ℕ, nm_count = 72 ∧ ∀ n m: ℕ, 0 < n ∧ 0 < m ∧ n * m = 72000 → 
  ∃ nm_pairs: ℕ, nm_pairs = 72 * 2 :=
sorry

end ways_to_divide_day_l102_102155


namespace complement_of_M_in_U_l102_102076

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | x^2 - 5*x + 6 = 0}
def C_U (M : Set ℕ) (U : Set ℕ) : Set ℕ := U \ M

theorem complement_of_M_in_U : C_U M U = {1, 4} :=
by
  sorry

end complement_of_M_in_U_l102_102076


namespace triangle_region_areas_l102_102165

open Real

theorem triangle_region_areas (A B C : ℝ) 
  (h1 : 20^2 + 21^2 = 29^2)
  (h2 : ∃ (triangle_area : ℝ), triangle_area = 210)
  (h3 : C > A)
  (h4 : C > B)
  : A + B + 210 = C := 
sorry

end triangle_region_areas_l102_102165


namespace arccos_cos_nine_l102_102904

theorem arccos_cos_nine : 
  ∀ (x : ℝ), x = 9 - 2 * Real.pi → Real.arccos (Real.cos 9) = x :=
by
  assume x h
  rw h
  sorry

end arccos_cos_nine_l102_102904


namespace calculate_expr_l102_102040

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem calculate_expr : ((x^3 * y^2)^2 * (x / y^3)) = x^7 * y :=
by sorry

end calculate_expr_l102_102040


namespace sum_of_factors_l102_102788

theorem sum_of_factors (x y : ℕ) :
  let exp := (27 * x ^ 6 - 512 * y ^ 6)
  let factor1 := (3 * x ^ 2 - 8 * y ^ 2)
  let factor2 := (3 * x ^ 2 + 8 * y ^ 2)
  let factor3 := (9 * x ^ 4 - 24 * x ^ 2 * y ^ 2 + 64 * y ^ 4)
  let sum := 3 + (-8) + 3 + 8 + 9 + (-24) + 64
  (factor1 * factor2 * factor3 = exp) ∧ (sum = 55) := 
by
  sorry

end sum_of_factors_l102_102788


namespace intervals_of_increase_of_f_l102_102624

theorem intervals_of_increase_of_f :
  ∀ k : ℤ,
  ∀ x y : ℝ,
  k * π - (5 / 8) * π ≤ x ∧ x ≤ y ∧ y ≤ k * π - (1 / 8) * π →
  3 * Real.sin ((π / 4) - 2 * x) - 2 ≤ 3 * Real.sin ((π / 4) - 2 * y) - 2 :=
by
  sorry

end intervals_of_increase_of_f_l102_102624


namespace sum_of_heights_less_than_perimeter_l102_102267

theorem sum_of_heights_less_than_perimeter
  (a b c h1 h2 h3 : ℝ) 
  (H1 : h1 ≤ b) 
  (H2 : h2 ≤ c) 
  (H3 : h3 ≤ a) 
  (H4 : h1 < b ∨ h2 < c ∨ h3 < a) : 
  h1 + h2 + h3 < a + b + c :=
by {
  sorry
}

end sum_of_heights_less_than_perimeter_l102_102267


namespace wrench_force_inv_proportional_l102_102114

theorem wrench_force_inv_proportional (F₁ : ℝ) (L₁ : ℝ) (F₂ : ℝ) (L₂ : ℝ) (k : ℝ)
  (h₁ : F₁ * L₁ = k) (h₂ : L₁ = 12) (h₃ : F₁ = 300) (h₄ : L₂ = 18) :
  F₂ = 200 :=
by
  sorry

end wrench_force_inv_proportional_l102_102114


namespace total_berries_l102_102988

theorem total_berries (S_stacy S_steve S_skylar : ℕ) 
  (h1 : S_stacy = 800)
  (h2 : S_stacy = 4 * S_steve)
  (h3 : S_steve = 2 * S_skylar) :
  S_stacy + S_steve + S_skylar = 1100 :=
by
  sorry

end total_berries_l102_102988


namespace find_prices_and_max_basketballs_l102_102293

def unit_price_condition (x : ℕ) (y : ℕ) : Prop :=
  y = 2*x - 30

def cost_ratio_condition (x : ℕ) (y : ℕ) : Prop :=
  3 * x = 2 * y - 60

def total_cost_condition (total_cost : ℕ) (num_basketballs : ℕ) (num_soccerballs : ℕ) : Prop :=
  total_cost ≤ 15500 ∧ num_basketballs + num_soccerballs = 200

theorem find_prices_and_max_basketballs
  (x y : ℕ) (total_cost : ℕ) (num_basketballs : ℕ) (num_soccerballs : ℕ)
  (h1 : unit_price_condition x y)
  (h2 : cost_ratio_condition x y)
  (h3 : total_cost_condition total_cost num_basketballs num_soccerballs)
  (h4 : total_cost = 90 * num_basketballs + 60 * num_soccerballs)
  : x = 60 ∧ y = 90 ∧ num_basketballs ≤ 116 :=
sorry

end find_prices_and_max_basketballs_l102_102293


namespace quadratic_equal_real_roots_l102_102362

theorem quadratic_equal_real_roots (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 = 0 ∧ (x = a*x / 2)) ↔ a = 2 ∨ a = -2 :=
by sorry

end quadratic_equal_real_roots_l102_102362


namespace sum_of_digits_of_product_in_base9_l102_102038

def base9_to_decimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  d1 * 9 + d0

def base10_to_base9 (n : ℕ) : ℕ :=
  let d0 := n % 9
  let d1 := (n / 9) % 9
  let d2 := (n / 81) % 9
  d2 * 100 + d1 * 10 + d0

def sum_of_digits_base9 (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 + d1 + d0

theorem sum_of_digits_of_product_in_base9 :
  let n1 := base9_to_decimal 36
  let n2 := base9_to_decimal 21
  let product := n1 * n2
  let base9_product := base10_to_base9 product
  sum_of_digits_base9 base9_product = 19 :=
by
  sorry

end sum_of_digits_of_product_in_base9_l102_102038


namespace vector_subtraction_l102_102380

def a : Real × Real := (2, -1)
def b : Real × Real := (-2, 3)

theorem vector_subtraction :
  a.1 - 2 * b.1 = 6 ∧ a.2 - 2 * b.2 = -7 := by
  sorry

end vector_subtraction_l102_102380


namespace derivative_f_l102_102115

noncomputable def f (x : ℝ) : ℝ := x + (1 / x)

theorem derivative_f (x : ℝ) (hx : x ≠ 0) :
  deriv f x = 1 - (1 / (x ^ 2)) :=
by
  -- The proof goes here
  sorry

end derivative_f_l102_102115


namespace total_coins_received_l102_102308

theorem total_coins_received (coins_first_day coins_second_day : ℕ) 
  (h_first_day : coins_first_day = 22) 
  (h_second_day : coins_second_day = 12) : 
  coins_first_day + coins_second_day = 34 := 
by 
  sorry

end total_coins_received_l102_102308


namespace find_m_l102_102809

def A := {x : ℝ | x^2 - 3 * x + 2 = 0}
def C (m : ℝ) := {x : ℝ | x^2 - m * x + 2 = 0}

theorem find_m (m : ℝ) (h : A ∩ C m = C m) : 
  m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) :=
by sorry

end find_m_l102_102809


namespace diagonal_crosses_700_cubes_l102_102889

noncomputable def num_cubes_crossed (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd c a + Nat.gcd (Nat.gcd a b) c

theorem diagonal_crosses_700_cubes :
  num_cubes_crossed 200 300 350 = 700 :=
sorry

end diagonal_crosses_700_cubes_l102_102889


namespace exists_indices_l102_102364

-- Define the sequence condition
def is_sequence_of_all_positive_integers (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, a m = n) ∧ (∀ n m1 m2 : ℕ, a m1 = n ∧ a m2 = n → m1 = m2)

-- Main theorem statement
theorem exists_indices 
  (a : ℕ → ℕ) 
  (h : is_sequence_of_all_positive_integers a) :
  ∃ (ℓ m : ℕ), 1 < ℓ ∧ ℓ < m ∧ (a 0 + a m = 2 * a ℓ) :=
by
  sorry

end exists_indices_l102_102364


namespace red_blood_cells_surface_area_l102_102556

-- Define the body surface area of an adult
def body_surface_area : ℝ := 1800

-- Define the multiplying factor for the surface areas of red blood cells
def multiplier : ℝ := 2000

-- Define the sum of the surface areas of all red blood cells
def sum_surface_area : ℝ := multiplier * body_surface_area

-- Define the expected sum in scientific notation
def expected_sum : ℝ := 3.6 * 10^6

-- The theorem that needs to be proved
theorem red_blood_cells_surface_area :
  sum_surface_area = expected_sum :=
by
  sorry

end red_blood_cells_surface_area_l102_102556


namespace min_ab_min_expr_min_a_b_l102_102367

-- Define the conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hln : Real.log a + Real.log b = Real.log (a + 9 * b))

-- 1. The minimum value of ab
theorem min_ab : ab = 36 :=
sorry

-- 2. The minimum value of (81 / a^2) + (1 / b^2)
theorem min_expr : (81 / a^2) + (1 / b^2) = (1 / 2) :=
sorry

-- 3. The minimum value of a + b
theorem min_a_b : a + b = 16 :=
sorry

end min_ab_min_expr_min_a_b_l102_102367


namespace find_A_l102_102079

def clubsuit (A B : ℤ) : ℤ := 4 * A + 2 * B + 6

theorem find_A : ∃ A : ℤ, clubsuit A 6 = 70 → A = 13 := 
by
  sorry

end find_A_l102_102079


namespace num_coprime_to_15_l102_102913

theorem num_coprime_to_15 : (filter (fun a => Nat.gcd a 15 = 1) (List.range 15)).length = 8 := by
  sorry

end num_coprime_to_15_l102_102913


namespace polynomial_satisfies_condition_l102_102343

-- Define P as a real polynomial
def P (a : ℝ) (X : ℝ) : ℝ := a * X

-- Define a statement that needs to be proven
theorem polynomial_satisfies_condition (P : ℝ → ℝ) :
  (∀ X : ℝ, P (2 * X) = 2 * P X) ↔ ∃ a : ℝ, ∀ X : ℝ, P X = a * X :=
by
  sorry

end polynomial_satisfies_condition_l102_102343


namespace necessary_but_not_sufficient_l102_102064

variable {a b : ℝ}

theorem necessary_but_not_sufficient : (a < b + 1) ∧ ¬ (a < b + 1 → a < b) :=
by
  sorry

end necessary_but_not_sufficient_l102_102064


namespace domain_of_f2x_l102_102804

theorem domain_of_f2x (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = f x) : 
  ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, f y = f (2 * x) :=
by
  sorry

end domain_of_f2x_l102_102804


namespace molecular_weight_of_compound_l102_102731

def hydrogen_atomic_weight : ℝ := 1.008
def chromium_atomic_weight : ℝ := 51.996
def oxygen_atomic_weight : ℝ := 15.999

def compound_molecular_weight (h_atoms : ℕ) (cr_atoms : ℕ) (o_atoms : ℕ) : ℝ :=
  h_atoms * hydrogen_atomic_weight + cr_atoms * chromium_atomic_weight + o_atoms * oxygen_atomic_weight

theorem molecular_weight_of_compound :
  compound_molecular_weight 2 1 4 = 118.008 :=
by
  sorry

end molecular_weight_of_compound_l102_102731


namespace average_salary_feb_mar_apr_may_l102_102414

theorem average_salary_feb_mar_apr_may 
  (average_jan_feb_mar_apr : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_months_1 : ℤ)
  (total_months_2 : ℤ)
  (total_sum_jan_apr : average_jan_feb_mar_apr * (total_months_1:ℝ) = 32000)
  (january_salary: salary_jan = 4700)
  (may_salary: salary_may = 6500)
  (total_months_1_eq: total_months_1 = 4)
  (total_months_2_eq: total_months_2 = 4):
  average_jan_feb_mar_apr * (total_months_1:ℝ) - salary_jan + salary_may/total_months_2 = 8450 :=
by
  sorry

end average_salary_feb_mar_apr_may_l102_102414


namespace triangle_side_ratio_range_l102_102511

theorem triangle_side_ratio_range (A B C a b c : ℝ) (h1 : A + 4 * B = 180) (h2 : C = 3 * B) (h3 : 0 < B ∧ B < 45) 
  (h4 : a / b = Real.sin (4 * B) / Real.sin B) : 
  1 < a / b ∧ a / b < 3 := 
sorry

end triangle_side_ratio_range_l102_102511


namespace last_two_non_zero_digits_of_75_factorial_l102_102625

theorem last_two_non_zero_digits_of_75_factorial : 
  ∃ (d : ℕ), d = 32 := sorry

end last_two_non_zero_digits_of_75_factorial_l102_102625


namespace solve_for_x_l102_102412

theorem solve_for_x (x : ℝ) (hx₁ : x ≠ 3) (hx₂ : x ≠ -2) 
  (h : (x + 5) / (x - 3) = (x - 2) / (x + 2)) : x = -1 / 3 :=
by
  sorry

end solve_for_x_l102_102412


namespace dots_per_ladybug_l102_102761

-- Define the conditions as variables
variables (m t : ℕ) (total_dots : ℕ) (d : ℕ)

-- Setting actual values for the variables based on the given conditions
def m_val : ℕ := 8
def t_val : ℕ := 5
def total_dots_val : ℕ := 78

-- Defining the total number of ladybugs and the average dots per ladybug
def total_ladybugs : ℕ := m_val + t_val

-- To prove: Each ladybug has 6 dots on average
theorem dots_per_ladybug : total_dots_val / total_ladybugs = 6 :=
by
  have m := m_val
  have t := t_val
  have total_dots := total_dots_val
  have d := 6
  sorry

end dots_per_ladybug_l102_102761


namespace maximum_rectangle_area_in_circle_l102_102041

theorem maximum_rectangle_area_in_circle (ω : ℝ → ℝ → Prop) (A B C D E : ℝ × ℝ) 
  (h1 : ω (A.fst, A.snd)) (h2 : ω (B.fst, B.snd)) (h3 : ω (C.fst, C.snd)) (h4 : ω (D.fst, D.snd))
  (h_int : ω (E.fst, E.snd)) 
  (h_AE : (A, E).fst - A.snd = 8) 
  (h_BE : (B, E).fst - B.snd = 2) 
  (h_CD : C.fst - D.fst + C.snd - D.snd = 10) 
  (h_AEC : E.fst - A.fst = E.fst - C.fst ∧ 
            E.snd - A.snd = E.snd - C.snd) : 
  ∃ R : ℝ, ∀ x y : ℝ, R = 6 * Real.sqrt 17 :=
begin
  use 6 * Real.sqrt 17,
  intros x y,
  sorry
end

end maximum_rectangle_area_in_circle_l102_102041


namespace coins_in_pockets_l102_102504

theorem coins_in_pockets : (Nat.choose (5 + 3 - 1) (3 - 1)) = 21 := by
  sorry

end coins_in_pockets_l102_102504


namespace part1_part2_part3_l102_102597

-- Part 1
theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem part2 (a m n : ℤ) (h1 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) (h2 : 0 < a) (h3 : 0 < m) (h4 : 0 < n) : 
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem part3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 :=
sorry

end part1_part2_part3_l102_102597


namespace desired_average_sale_is_5600_l102_102432

-- Define the sales for five consecutive months
def sale1 : ℕ := 5266
def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029

-- Define the required sale for the sixth month
def sale6 : ℕ := 4937

-- Calculate total sales for the first five months
def total_five_months := sale1 + sale2 + sale3 + sale4 + sale5

-- Calculate total sales for six months
def total_six_months := total_five_months + sale6

-- Calculate the desired average sale for six months
def desired_average := total_six_months / 6

-- The theorem statement: desired average sale for the six months
theorem desired_average_sale_is_5600 : desired_average = 5600 :=
by
  sorry

end desired_average_sale_is_5600_l102_102432


namespace conditional_probability_l102_102366

-- Given probabilities:
def p_a : ℚ := 5/23
def p_b : ℚ := 7/23
def p_c : ℚ := 1/23
def p_a_and_b : ℚ := 2/23
def p_a_and_c : ℚ := 1/23
def p_b_and_c : ℚ := 1/23
def p_a_and_b_and_c : ℚ := 1/23

-- Theorem statement to prove:
theorem conditional_probability : p_a_and_b_and_c / p_a_and_c = 1 :=
by
  sorry

end conditional_probability_l102_102366


namespace division_quotient_l102_102103

theorem division_quotient (dividend divisor remainder quotient : Nat) 
  (h_dividend : dividend = 109)
  (h_divisor : divisor = 12)
  (h_remainder : remainder = 1)
  (h_division_equation : dividend = divisor * quotient + remainder)
  : quotient = 9 := 
by
  sorry

end division_quotient_l102_102103


namespace sally_picked_peaches_l102_102841

theorem sally_picked_peaches (original_peaches total_peaches picked_peaches : ℕ)
  (h_orig : original_peaches = 13)
  (h_total : total_peaches = 55)
  (h_picked : picked_peaches = total_peaches - original_peaches) :
  picked_peaches = 42 :=
by
  sorry

end sally_picked_peaches_l102_102841


namespace find_a2_a3_sequence_constant_general_formula_l102_102922

-- Definition of the sequence and its sum Sn
variables (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
axiom a1_eq : a 1 = 2
axiom S_eq : ∀ n, S (n + 1) = 4 * a n - 2

-- Prove that a_2 = 4 and a_3 = 8
theorem find_a2_a3 : a 2 = 4 ∧ a 3 = 8 :=
sorry

-- Prove that the sequence {a_n - 2a_{n-1}} is constant
theorem sequence_constant {n : ℕ} (hn : n ≥ 2) :
  ∃ c, ∀ k ≥ 2, a k - 2 * a (k - 1) = c :=
sorry

-- Find the general formula for the sequence
theorem general_formula :
  ∀ n, a n = 2^n :=
sorry

end find_a2_a3_sequence_constant_general_formula_l102_102922


namespace number_of_oddly_powerful_integers_lt_500_l102_102458

noncomputable def count_oddly_powerful_integers_lt_500 : ℕ :=
  let count_cubes := 7 -- we counted cubes: 1^3, 2^3, 3^3, 4^3, 5^3, 6^3, 7^3
  let count_fifth_powers := 1 -- the additional fifth power not a cube: 3^5
  count_cubes + count_fifth_powers

theorem number_of_oddly_powerful_integers_lt_500 : count_oddly_powerful_integers_lt_500 = 8 :=
  sorry

end number_of_oddly_powerful_integers_lt_500_l102_102458


namespace find_some_number_l102_102025

theorem find_some_number (some_number : ℕ) : 
  ( ∃ n:ℕ, n = 54 ∧ (n / 18) * (n / some_number) = 1 ) ∧ some_number = 162 :=
by {
  sorry
}

end find_some_number_l102_102025


namespace two_roots_iff_a_gt_neg1_l102_102190

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l102_102190


namespace largest_sum_fraction_l102_102451

theorem largest_sum_fraction :
  max 
    ((1/3) + (1/2))
    (max 
      ((1/3) + (1/5))
      (max 
        ((1/3) + (1/6))
        (max 
          ((1/3) + (1/9))
          ((1/3) + (1/10))
        )
      )
    ) = 5/6 :=
by sorry

end largest_sum_fraction_l102_102451


namespace snail_returns_to_starting_point_l102_102589

-- Define the variables and conditions
variables (a1 a2 b1 b2 : ℕ)

-- Prove that snail can return to starting point after whole number of hours
theorem snail_returns_to_starting_point (h1 : a1 = a2) (h2 : b1 = b2) : (a1 + b1 : ℕ) = (a1 + b1 : ℕ) :=
by sorry

end snail_returns_to_starting_point_l102_102589


namespace union_complement_eq_l102_102693

def U := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 5}

def complement (univ : Set ℕ) (s : Set ℕ) : Set ℕ :=
  {x ∈ univ | x ∉ s}

theorem union_complement_eq :
  N ∪ (complement U M) = {2, 3, 5} :=
by sorry

end union_complement_eq_l102_102693


namespace rhombus_area_l102_102374

theorem rhombus_area (x y : ℝ)
  (h1 : x^2 + y^2 = 113) 
  (h2 : x = y + 8) : 
  1 / 2 * (2 * y) * (2 * (y + 4)) = 97 := 
by 
  -- Assume x and y are the half-diagonals of the rhombus
  sorry

end rhombus_area_l102_102374


namespace minimum_value_g_l102_102046

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if a > 1 then 
    a * (-1/a) + 1 
  else 
    if 0 < a then 
      a^2 + 1 
    else 
      0  -- adding a default value to make it computable

theorem minimum_value_g (a : ℝ) (m : ℝ) : 0 < a ∧ a < 2 ∧ ∃ x₀, f x₀ a = m → m ≥ 5 / 2 :=
by
  sorry

end minimum_value_g_l102_102046


namespace product_modulo_7_l102_102775

theorem product_modulo_7 : 
  (2007 % 7 = 4) ∧ (2008 % 7 = 5) ∧ (2009 % 7 = 6) ∧ (2010 % 7 = 0) →
  (2007 * 2008 * 2009 * 2010) % 7 = 0 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end product_modulo_7_l102_102775


namespace jim_age_is_55_l102_102171

-- Definitions of the conditions
def jim_age (t : ℕ) : ℕ := 3 * t + 10

def sum_ages (j t : ℕ) : Prop := j + t = 70

-- Statement of the proof problem
theorem jim_age_is_55 : ∃ t : ℕ, jim_age t = 55 ∧ sum_ages (jim_age t) t :=
by
  sorry

end jim_age_is_55_l102_102171


namespace frank_total_points_l102_102144

def points_defeating_enemies (enemies : ℕ) (points_per_enemy : ℕ) : ℕ :=
  enemies * points_per_enemy

def total_points (points_from_enemies : ℕ) (completion_points : ℕ) : ℕ :=
  points_from_enemies + completion_points

theorem frank_total_points :
  let enemies := 6
  let points_per_enemy := 9
  let completion_points := 8
  let points_from_enemies := points_defeating_enemies enemies points_per_enemy
  total_points points_from_enemies completion_points = 62 :=
by
  let enemies := 6
  let points_per_enemy := 9
  let completion_points := 8
  let points_from_enemies := points_defeating_enemies enemies points_per_enemy
  -- Placeholder for proof
  sorry

end frank_total_points_l102_102144


namespace sum_of_products_lt_zero_l102_102389

theorem sum_of_products_lt_zero (a b c d e f : ℤ) (h : ∃ (i : ℕ), i ≤ 6 ∧ i ≠ 6 ∧ (∀ i ∈ [a, b, c, d, e, f], i < 0 → i ≤ i)) :
  ab + cdef < 0 :=
sorry

end sum_of_products_lt_zero_l102_102389


namespace tan_angle_addition_l102_102229

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 3) : Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end tan_angle_addition_l102_102229


namespace rearrangementCount_l102_102817

-- Define the sequence as Finsets to leverage Lean's combinatorial libraries
def sequence := {1, 2, 3, 4, 5, 6}

-- Define the condition that numbers 5 and 6 must be together in any permutation
def consecutive56 (l : List Nat) : Prop :=
  ∃ a b c, l = a ++ [5, 6] ++ b ++ c ∨
           l = a ++ [6, 5] ++ b ++ c

-- Define the condition that no three consecutive terms are either increasing or decreasing
def noThreeConsecIncrDec (l : List Nat) : Prop :=
  ∀ i, i + 2 < l.length →
  ¬ (l.nthLe i (by sorry) < l.nthLe (i + 1) (by sorry) < l.nthLe (i + 2) (by sorry)) ∧
  ¬ (l.nthLe i (by sorry) > l.nthLe (i + 1) (by sorry) > l.nthLe (i + 2) (by sorry))

noncomputable def countValidPermutations : Nat :=
  (Finset.permList sequence.toList).filter (λ l, consecutive56 l ∧ noThreeConsecIncrDec l).card

theorem rearrangementCount : countValidPermutations = 20 := by sorry

end rearrangementCount_l102_102817


namespace square_area_percentage_error_l102_102599

theorem square_area_percentage_error (s : ℝ) (h : s > 0) : 
  let measured_s := 1.06 * s in
  let actual_area := s^2 in
  let calculated_area := measured_s^2 in
  let error_area := calculated_area - actual_area in
  let percentage_error := (error_area / actual_area) * 100 in
  percentage_error = 12.36 := 
by
  sorry

end square_area_percentage_error_l102_102599


namespace value_of_a_l102_102944

theorem value_of_a
  (x y a : ℝ)
  (h1 : x + 2 * y = 2 * a - 1)
  (h2 : x - y = 6)
  (h3 : x = -y)
  : a = -1 :=
by
  sorry

end value_of_a_l102_102944


namespace count_coprimes_15_l102_102911

def count_coprimes_less_than (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ a => Nat.gcd a n = 1).card

theorem count_coprimes_15 :
  count_coprimes_less_than 15 = 8 :=
by
  sorry

end count_coprimes_15_l102_102911


namespace number_of_ways_to_place_coins_l102_102505

theorem number_of_ways_to_place_coins :
  (nat.choose 7 2) = 21 :=
by
  sorry

end number_of_ways_to_place_coins_l102_102505


namespace rebecca_eggs_l102_102536

theorem rebecca_eggs (groups : ℕ) (eggs_per_group : ℕ) (total_eggs : ℕ) 
  (h1 : groups = 3) (h2 : eggs_per_group = 3) : total_eggs = 9 :=
by
  sorry

end rebecca_eggs_l102_102536
