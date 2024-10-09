import Mathlib

namespace clubs_students_equal_l953_95383

theorem clubs_students_equal
  (C E : ℕ)
  (h1 : ∃ N, N = 3 * C)
  (h2 : ∃ N, N = 3 * E) :
  C = E :=
by
  sorry

end clubs_students_equal_l953_95383


namespace smallest_class_size_l953_95338

theorem smallest_class_size (n : ℕ) (x : ℕ) (h1 : n > 50) (h2 : n = 4 * x + 2) : n = 54 :=
by
  sorry

end smallest_class_size_l953_95338


namespace smallest_z_l953_95313

theorem smallest_z 
  (x y z : ℕ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h1 : x + y = z) 
  (h2 : x * y < z^2) 
  (ineq : (27^z) * (5^x) > (3^24) * (2^y)) :
  z = 10 :=
by
  sorry

end smallest_z_l953_95313


namespace max_value_of_xy_expression_l953_95337

theorem max_value_of_xy_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + 3 * y < 60) : 
  xy * (60 - 4 * x - 3 * y) ≤ 2000 / 3 := 
sorry

end max_value_of_xy_expression_l953_95337


namespace calculate_final_number_l953_95384

theorem calculate_final_number (initial increment times : ℕ) (h₀ : initial = 540) (h₁ : increment = 10) (h₂ : times = 6) : initial + increment * times = 600 :=
by
  sorry

end calculate_final_number_l953_95384


namespace find_M_l953_95397

theorem find_M : ∃ M : ℕ, M > 0 ∧ 18 ^ 2 * 45 ^ 2 = 15 ^ 2 * M ^ 2 ∧ M = 54 := by
  use 54
  sorry

end find_M_l953_95397


namespace square_side_length_range_l953_95363

theorem square_side_length_range (a : ℝ) (h : a^2 = 30) : 5.4 < a ∧ a < 5.5 :=
sorry

end square_side_length_range_l953_95363


namespace number_of_solutions_l953_95302

theorem number_of_solutions :
  (∃ (a b c : ℕ), 4 * a = 6 * c ∧ 168 * a = 6 * a * b * c) → 
  ∃ (s : Finset ℕ), s.card = 6 :=
by sorry

end number_of_solutions_l953_95302


namespace find_x_rational_l953_95341

theorem find_x_rational (x : ℝ) (h1 : ∃ (a : ℚ), x + Real.sqrt 3 = a)
  (h2 : ∃ (b : ℚ), x^2 + Real.sqrt 3 = b) :
  x = (1 / 2 : ℝ) - Real.sqrt 3 :=
sorry

end find_x_rational_l953_95341


namespace velocity_division_l953_95398

/--
Given a trapezoidal velocity-time graph with bases V and U,
determine the velocity W that divides the area under the graph into
two regions such that the areas are in the ratio 1:k.
-/
theorem velocity_division (V U k : ℝ) (h_k : k ≠ -1) : 
  ∃ W : ℝ, W = (V^2 + k * U^2) / (k + 1) :=
by
  sorry

end velocity_division_l953_95398


namespace unique_solutions_of_system_l953_95381

def system_of_equations (x y : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3) = 0 ∧
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|) = 0

theorem unique_solutions_of_system :
  ∀ (x y : ℝ), system_of_equations x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4) :=
by sorry

end unique_solutions_of_system_l953_95381


namespace distinct_positive_and_conditions_l953_95307

theorem distinct_positive_and_conditions (a b : ℕ) (h_distinct: a ≠ b) (h_pos1: 0 < a) (h_pos2: 0 < b) (h_eq: a^3 - b^3 = a^2 - b^2) : 
  ∃ (c : ℕ), c = 9 * a * b ∧ (c = 1 ∨ c = 2 ∨ c = 3) :=
by
  sorry

end distinct_positive_and_conditions_l953_95307


namespace number_of_non_congruent_triangles_perimeter_18_l953_95367

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l953_95367


namespace overlap_region_area_l953_95364

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

noncomputable def overlap_area : ℝ := 
  let A : ℝ × ℝ := (0, 0);
  let B : ℝ × ℝ := (6, 2);
  let C : ℝ × ℝ := (2, 6);
  let D : ℝ × ℝ := (6, 6);
  let E : ℝ × ℝ := (0, 2);
  let F : ℝ × ℝ := (2, 0);
  let P1 : ℝ × ℝ := (2, 2);
  let P2 : ℝ × ℝ := (4, 2);
  let P3 : ℝ × ℝ := (3, 3);
  let P4 : ℝ × ℝ := (2, 3);
  1/2 * abs (P1.1 * (P2.2 - P4.2) + P2.1 * (P3.2 - P1.2) + P3.1 * (P4.2 - P2.2) + P4.1 * (P1.2 - P3.2))

theorem overlap_region_area :
  let A : ℝ × ℝ := (0, 0);
  let B : ℝ × ℝ := (6, 2);
  let C : ℝ × ℝ := (2, 6);
  let D : ℝ × ℝ := (6, 6);
  let E : ℝ × ℝ := (0, 2);
  let F : ℝ × ℝ := (2, 0);
  triangle_area A B C > 0 →
  triangle_area D E F > 0 →
  overlap_area = 0.5 :=
by { sorry }

end overlap_region_area_l953_95364


namespace wrench_turns_bolt_l953_95354

theorem wrench_turns_bolt (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (Real.sqrt 3 / Real.sqrt 2 < b / a) ∧ (b / a ≤ 3 - Real.sqrt 3) :=
sorry

end wrench_turns_bolt_l953_95354


namespace product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240_l953_95316

theorem product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240
 (p : ℕ) (prime_p : Prime p) (prime_p_plus_2 : Prime (p + 2)) (p_gt_7 : p > 7) :
  240 ∣ ((p - 1) * p * (p + 1)) := by
  sorry

end product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240_l953_95316


namespace total_games_played_l953_95387

theorem total_games_played (jerry_wins dave_wins ken_wins : ℕ)
  (h1 : jerry_wins = 7)
  (h2 : dave_wins = jerry_wins + 3)
  (h3 : ken_wins = dave_wins + 5) :
  jerry_wins + dave_wins + ken_wins = 32 :=
by
  sorry

end total_games_played_l953_95387


namespace bus_people_count_l953_95319

-- Define the initial number of people on the bus
def initial_people_on_bus : ℕ := 34

-- Define the number of people who got off the bus
def people_got_off : ℕ := 11

-- Define the number of people who got on the bus
def people_got_on : ℕ := 24

-- Define the final number of people on the bus
def final_people_on_bus : ℕ := (initial_people_on_bus - people_got_off) + people_got_on

-- Theorem: The final number of people on the bus is 47.
theorem bus_people_count : final_people_on_bus = 47 := by
  sorry

end bus_people_count_l953_95319


namespace inequality_of_ab_l953_95392

theorem inequality_of_ab (a b : ℝ) (h₁ : a < 0) (h₂ : -1 < b ∧ b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_of_ab_l953_95392


namespace transformer_coils_flawless_l953_95306

theorem transformer_coils_flawless (x y : ℕ) (hx : x + y = 8200)
  (hdef : (2 * x / 100) + (3 * y / 100) = 216) :
  ((x = 3000 ∧ y = 5200) ∧ ((x * 98 / 100) = 2940) ∧ ((y * 97 / 100) = 5044)) :=
by
  sorry

end transformer_coils_flawless_l953_95306


namespace expr_value_l953_95327

-- Define the given expression
def expr : ℕ := 11 - 10 / 2 + (8 * 3) - 7 / 1 + 9 - 6 * 2 + 4 - 3

-- Assert the proof goal
theorem expr_value : expr = 21 := by
  sorry

end expr_value_l953_95327


namespace find_width_of_jordan_rectangle_l953_95368

theorem find_width_of_jordan_rectangle (width : ℕ) (h1 : 12 * 15 = 9 * width) : width = 20 :=
by
  sorry

end find_width_of_jordan_rectangle_l953_95368


namespace wuyang_volleyball_team_members_l953_95339

theorem wuyang_volleyball_team_members :
  (Finset.filter Nat.Prime (Finset.range 50)).card = 15 :=
by
  sorry

end wuyang_volleyball_team_members_l953_95339


namespace number_of_shelves_l953_95335

/-- Adam could fit 11 action figures on each shelf -/
def action_figures_per_shelf : ℕ := 11

/-- Adam's shelves could hold a total of 44 action figures -/
def total_action_figures_on_shelves : ℕ := 44

/-- Prove the number of shelves in Adam's room -/
theorem number_of_shelves:
  total_action_figures_on_shelves / action_figures_per_shelf = 4 := 
by {
    sorry
}

end number_of_shelves_l953_95335


namespace difference_english_math_l953_95333

/-- There are 30 students who pass in English and 20 students who pass in Math. -/
axiom passes_in_english : ℕ
axiom passes_in_math : ℕ
axiom both_subjects : ℕ
axiom only_english : ℕ
axiom only_math : ℕ

/-- Definitions based on the problem conditions -/
axiom number_passes_in_english : only_english + both_subjects = 30
axiom number_passes_in_math : only_math + both_subjects = 20

/-- The difference between the number of students who pass only in English
    and the number of students who pass only in Math is 10. -/
theorem difference_english_math : only_english - only_math = 10 :=
by
  sorry

end difference_english_math_l953_95333


namespace equation_of_circle_C_equation_of_line_l_l953_95361

-- Condition: The center of the circle lies on the line y = x + 1.
def center_on_line (a b : ℝ) : Prop :=
  b = a + 1

-- Condition: The circle is tangent to the x-axis.
def tangent_to_x_axis (a b r : ℝ) : Prop :=
  r = b

-- Condition: Point P(-5, -2) lies on the circle.
def point_on_circle (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Condition: Point Q(-4, -5) lies outside the circle.
def point_outside_circle (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 > r^2

-- Proof (1): Find the equation of the circle.
theorem equation_of_circle_C :
  ∃ (a b r : ℝ), center_on_line a b ∧ tangent_to_x_axis a b r ∧ point_on_circle a b r (-5) (-2) ∧ point_outside_circle a b r (-4) (-5) ∧ (∀ x y, (x - a)^2 + (y - b)^2 = r^2 ↔ (x + 3)^2 + (y + 2)^2 = 4) :=
sorry

-- Proof (2): Find the equation of the line l.
theorem equation_of_line_l (a b r : ℝ) (ha : center_on_line a b) (hb : tangent_to_x_axis a b r) (hc : point_on_circle a b r (-5) (-2)) (hd : point_outside_circle a b r (-4) (-5)) :
  ∃ (k : ℝ), ∀ x y, ((k = 0 ∧ x = -2) ∨ (k ≠ 0 ∧ y + 4 = -3/4 * (x + 2))) ↔ ((x = -2) ∨ (3 * x + 4 * y + 22 = 0)) :=
sorry

end equation_of_circle_C_equation_of_line_l_l953_95361


namespace triangle_angle_sum_l953_95396

theorem triangle_angle_sum (x : ℝ) :
  let a := 40
  let b := 60
  let sum_of_angles := 180
  a + b + x = sum_of_angles → x = 80 :=
by
  intros
  sorry

end triangle_angle_sum_l953_95396


namespace negation_of_existence_l953_95331

variable (Triangle : Type) (has_circumcircle : Triangle → Prop)

theorem negation_of_existence :
  ¬ (∃ t : Triangle, ¬ has_circumcircle t) ↔ ∀ t : Triangle, has_circumcircle t :=
by sorry

end negation_of_existence_l953_95331


namespace points_on_circle_l953_95380

theorem points_on_circle (t : ℝ) : 
  ( (2 - 3 * t^2) / (2 + t^2) )^2 + ( 3 * t / (2 + t^2) )^2 = 1 := 
by 
  sorry

end points_on_circle_l953_95380


namespace coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_l953_95370

theorem coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5 :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (x - 2) ^ 5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 → a_2 = -80 := by
  sorry

end coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_l953_95370


namespace complement_A_is_interval_l953_95353

def U : Set ℝ := {x | True}
def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def compl_U_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem complement_A_is_interval : compl_U_A = {x | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end complement_A_is_interval_l953_95353


namespace find_sin_B_l953_95309

variables (a b c : ℝ) (A B C : ℝ)

def sin_law_abc (a b : ℝ) (sinA : ℝ) (sinB : ℝ) : Prop := 
  (a / sinA) = (b / sinB)

theorem find_sin_B {a b : ℝ} (sinA : ℝ) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hA : sinA = 1 / 3) :
  ∃ sinB : ℝ, (sinB = 5 / 9) ∧ sin_law_abc a b sinA sinB :=
by
  use 5 / 9
  simp [sin_law_abc, ha, hb, hA]
  sorry

end find_sin_B_l953_95309


namespace find_fraction_l953_95329

theorem find_fraction (n d : ℕ) (h1 : n / (d + 1) = 1 / 2) (h2 : (n + 1) / d = 1) : n / d = 2 / 3 := 
by 
  sorry

end find_fraction_l953_95329


namespace values_of_n_for_replaced_constant_l953_95362

theorem values_of_n_for_replaced_constant (n : ℤ) (x : ℤ) :
  (∀ n : ℤ, 4 * n + x > 1 ∧ 4 * n + x < 60) → x = 8 → 
  (∀ n : ℤ, 4 * n + 8 > 1 ∧ 4 * n + 8 < 60) :=
by
  sorry

end values_of_n_for_replaced_constant_l953_95362


namespace solid_circles_count_2006_l953_95321

def series_of_circles (n : ℕ) : List Char :=
  if n ≤ 0 then []
  else if n % 5 == 0 then '●' :: series_of_circles (n - 1)
  else '○' :: series_of_circles (n - 1)

def count_solid_circles (l : List Char) : ℕ :=
  l.count '●'

theorem solid_circles_count_2006 : count_solid_circles (series_of_circles 2006) = 61 := 
by
  sorry

end solid_circles_count_2006_l953_95321


namespace quadratic_minimum_eq_one_l953_95328

variable (p q : ℝ)

theorem quadratic_minimum_eq_one (hq : q = 1 + p^2 / 18) : 
  ∃ x : ℝ, 3 * x^2 + p * x + q = 1 :=
by
  sorry

end quadratic_minimum_eq_one_l953_95328


namespace possible_values_of_m_l953_95342

def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem possible_values_of_m (m : ℝ) : (∀ x, S x m → P x) ↔ (m = -1 ∨ m = 1 ∨ m = 3) :=
by
  sorry

end possible_values_of_m_l953_95342


namespace harkamal_paid_amount_l953_95371

variable (grapesQuantity : ℕ)
variable (grapesRate : ℕ)
variable (mangoesQuantity : ℕ)
variable (mangoesRate : ℕ)

theorem harkamal_paid_amount (h1 : grapesQuantity = 8) (h2 : grapesRate = 70) (h3 : mangoesQuantity = 9) (h4 : mangoesRate = 45) :
  (grapesQuantity * grapesRate + mangoesQuantity * mangoesRate) = 965 := by
  sorry

end harkamal_paid_amount_l953_95371


namespace tangent_line_at_0_l953_95322

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x + Real.sin x

theorem tangent_line_at_0 :
  ∃ (m b : ℝ), ∀ (x : ℝ), f 0 = 1 ∧ (f' : ℝ → ℝ) 0 = 1 ∧ (f' x = Real.exp x + 2 * x - 1 + Real.cos x) ∧ 
  (m = 1) ∧ (b = (m * 0 + 1)) ∧ (∀ x : ℝ, y = m * x + b) :=
by
  sorry

end tangent_line_at_0_l953_95322


namespace minimum_m_n_1978_l953_95394

-- Define the conditions given in the problem
variables (m n : ℕ) (h1 : n > m) (h2 : m > 1)
-- Define the condition that the last three digits of 1978^m and 1978^n are identical
def same_last_three_digits (a b : ℕ) : Prop :=
  (a % 1000 = b % 1000)

-- Define the problem statement: under the conditions, prove that m + n = 106 when minimized
theorem minimum_m_n_1978 (h : same_last_three_digits (1978^m) (1978^n)) : m + n = 106 :=
sorry   -- Proof will be provided here

end minimum_m_n_1978_l953_95394


namespace tom_age_ratio_l953_95326

variable (T M : ℕ)
variable (h1 : T = T) -- Tom's age is equal to the sum of the ages of his four children
variable (h2 : T - M = 3 * (T - 4 * M)) -- M years ago, Tom's age was three times the sum of his children's ages then

theorem tom_age_ratio : (T / M) = 11 / 2 := 
by
  sorry

end tom_age_ratio_l953_95326


namespace distance_with_tide_60_min_l953_95375

variable (v_m v_t : ℝ)

axiom man_with_tide : (v_m + v_t) = 5
axiom man_against_tide : (v_m - v_t) = 4

theorem distance_with_tide_60_min : (v_m + v_t) = 5 := by
  sorry

end distance_with_tide_60_min_l953_95375


namespace chris_pounds_of_nuts_l953_95382

theorem chris_pounds_of_nuts :
  ∀ (R : ℝ) (x : ℝ),
  (∃ (N : ℝ), N = 4 * R) →
  (∃ (total_mixture_cost : ℝ), total_mixture_cost = 3 * R + 4 * R * x) →
  (3 * R = 0.15789473684210525 * total_mixture_cost) →
  x = 4 :=
by
  intros R x hN htotal_mixture_cost hRA
  sorry

end chris_pounds_of_nuts_l953_95382


namespace calculate_interest_rate_l953_95390

variables (A : ℝ) (R : ℝ)

-- Conditions as definitions in Lean 4
def compound_interest_condition (A : ℝ) (R : ℝ) : Prop :=
  (A * (1 + R)^20 = 4 * A)

-- Theorem statement
theorem calculate_interest_rate (A : ℝ) (R : ℝ) (h : compound_interest_condition A R) : 
  R = (4)^(1/20) - 1 := 
sorry

end calculate_interest_rate_l953_95390


namespace gcd_306_522_l953_95314

theorem gcd_306_522 : Nat.gcd 306 522 = 18 := 
  by sorry

end gcd_306_522_l953_95314


namespace cricket_target_runs_l953_95332

-- Define the conditions
def first_20_overs_run_rate : ℝ := 4.2
def remaining_30_overs_run_rate : ℝ := 8
def overs_20 : ℤ := 20
def overs_30 : ℤ := 30

-- State the proof problem
theorem cricket_target_runs : 
  (first_20_overs_run_rate * (overs_20 : ℝ)) + (remaining_30_overs_run_rate * (overs_30 : ℝ)) = 324 :=
by
  sorry

end cricket_target_runs_l953_95332


namespace range_of_m_l953_95325

theorem range_of_m (f g : ℝ → ℝ) (h1 : ∃ m : ℝ, ∀ x : ℝ, f x = m * (x - m) * (x + m + 3))
  (h2 : ∀ x : ℝ, g x = 2 ^ x - 4)
  (h3 : ∀ x : ℝ, f x < 0 ∨ g x < 0) :
  ∃ m : ℝ, -5 < m ∧ m < 0 :=
sorry

end range_of_m_l953_95325


namespace number_of_true_propositions_l953_95315

-- Let's state the propositions
def original_proposition (P Q : Prop) := P → Q
def converse_proposition (P Q : Prop) := Q → P
def inverse_proposition (P Q : Prop) := ¬P → ¬Q
def contrapositive_proposition (P Q : Prop) := ¬Q → ¬P

-- Main statement we need to prove
theorem number_of_true_propositions (P Q : Prop) (hpq : original_proposition P Q) 
  (hc: contrapositive_proposition P Q) (hev: converse_proposition P Q)  (hbv: inverse_proposition P Q) : 
  (¬(P ↔ Q) ∨ (¬¬P ↔ ¬¬Q) ∨ (¬Q → ¬P) ∨ (P → Q)) := sorry

end number_of_true_propositions_l953_95315


namespace roof_length_width_diff_l953_95320

variable (w l : ℝ)
variable (h1 : l = 4 * w)
variable (h2 : l * w = 676)

theorem roof_length_width_diff :
  l - w = 39 :=
by
  sorry

end roof_length_width_diff_l953_95320


namespace pow_equation_sum_l953_95379

theorem pow_equation_sum (x y : ℕ) (hx : 2 ^ 11 * 6 ^ 5 = 4 ^ x * 3 ^ y) : x + y = 13 :=
  sorry

end pow_equation_sum_l953_95379


namespace area_union_example_l953_95305

noncomputable def area_union_square_circle (s r : ℝ) : ℝ :=
  let A_square := s ^ 2
  let A_circle := Real.pi * r ^ 2
  let A_overlap := (1 / 4) * A_circle
  A_square + A_circle - A_overlap

theorem area_union_example : (area_union_square_circle 10 10) = 100 + 75 * Real.pi :=
by
  sorry

end area_union_example_l953_95305


namespace lawn_width_is_60_l953_95347

theorem lawn_width_is_60
  (length : ℕ)
  (width : ℕ)
  (road_width : ℕ)
  (cost_per_sq_meter : ℕ)
  (total_cost : ℕ)
  (area_of_lawn : ℕ)
  (total_area_of_roads : ℕ)
  (intersection_area : ℕ)
  (area_cost_relation : total_area_of_roads * cost_per_sq_meter = total_cost)
  (intersection_included : (road_width * length + road_width * width - intersection_area) = total_area_of_roads)
  (length_eq : length = 80)
  (road_width_eq : road_width = 10)
  (cost_eq : cost_per_sq_meter = 2)
  (total_cost_eq : total_cost = 2600)
  (intersection_area_eq : intersection_area = road_width * road_width)
  : width = 60 :=
by
  sorry

end lawn_width_is_60_l953_95347


namespace sqrt_factorial_mul_squared_l953_95312

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l953_95312


namespace monotonic_decreasing_interval_l953_95348

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

def decreasing_interval (a b : ℝ) := 
  ∀ x : ℝ, a < x ∧ x < b → deriv f x < 0

theorem monotonic_decreasing_interval : decreasing_interval 0 1 :=
sorry

end monotonic_decreasing_interval_l953_95348


namespace initial_red_marbles_l953_95350

theorem initial_red_marbles
    (r g : ℕ)
    (h1 : 3 * r = 5 * g)
    (h2 : 2 * (r - 15) = g + 18) :
    r = 34 := by
  sorry

end initial_red_marbles_l953_95350


namespace m_is_perfect_square_l953_95330

theorem m_is_perfect_square (n : ℕ) (m : ℤ) (h1 : m = 2 + 2 * Int.sqrt (44 * n^2 + 1) ∧ Int.sqrt (44 * n^2 + 1) * Int.sqrt (44 * n^2 + 1) = 44 * n^2 + 1) :
  ∃ k : ℕ, m = k^2 :=
by
  sorry

end m_is_perfect_square_l953_95330


namespace simplify_fraction_l953_95356

theorem simplify_fraction (a b c : ℕ) (h1 : 222 = 2 * 111) (h2 : 999 = 3 * 333) (h3 : 111 = 3 * 37) :
  (222 / 999 * 111) = 74 :=
by
  sorry

end simplify_fraction_l953_95356


namespace simplify_expression_l953_95372

theorem simplify_expression (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y^2) - 5 * (2 + 3 * y) = -4 * y^2 - 17 * y - 8 :=
by
  sorry

end simplify_expression_l953_95372


namespace yuna_candy_days_l953_95324

theorem yuna_candy_days (total_candies : ℕ) (daily_candies_week : ℕ) (days_week : ℕ) (remaining_candies : ℕ) (daily_candies_future : ℕ) :
  total_candies = 60 →
  daily_candies_week = 6 →
  days_week = 7 →
  remaining_candies = total_candies - (daily_candies_week * days_week) →
  daily_candies_future = 3 →
  remaining_candies / daily_candies_future = 6 :=
by
  intros h_total h_daily_week h_days_week h_remaining h_daily_future
  sorry

end yuna_candy_days_l953_95324


namespace area_times_breadth_l953_95345

theorem area_times_breadth (b l A : ℕ) (h1 : b = 11) (h2 : l - b = 10) (h3 : A = l * b) : A / b = 21 := 
by
  sorry

end area_times_breadth_l953_95345


namespace growth_rate_yield_per_acre_l953_95378

theorem growth_rate_yield_per_acre (x : ℝ) (a_i y_i y_f : ℝ) (h1 : a_i = 5) (h2 : y_i = 10000) (h3 : y_f = 30000) 
  (h4 : y_f = 5 * (1 + 2 * x) * (y_i / a_i) * (1 + x)) : x = 0.5 := 
by
  -- Insert the proof here
  sorry

end growth_rate_yield_per_acre_l953_95378


namespace trucks_and_goods_l953_95301

variable (x : ℕ) -- Number of trucks
variable (goods : ℕ) -- Total tons of goods

-- Conditions
def condition1 : Prop := goods = 3 * x + 5
def condition2 : Prop := goods = 4 * (x - 5)

theorem trucks_and_goods (h1 : condition1 x goods) (h2 : condition2 x goods) : x = 25 ∧ goods = 80 :=
by
  sorry

end trucks_and_goods_l953_95301


namespace four_digit_number_divisible_by_18_l953_95318

theorem four_digit_number_divisible_by_18 : ∃ n : ℕ, (n % 2 = 0) ∧ (10 + n) % 9 = 0 ∧ n = 8 :=
by
  sorry

end four_digit_number_divisible_by_18_l953_95318


namespace prove_equal_values_l953_95343

theorem prove_equal_values :
  (-2: ℝ)^3 = -(2: ℝ)^3 :=
by sorry

end prove_equal_values_l953_95343


namespace contrary_implies_mutually_exclusive_contrary_sufficient_but_not_necessary_l953_95366

variable {A B : Prop}

def contrary (A : Prop) : Prop := A ∧ ¬A
def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B)

theorem contrary_implies_mutually_exclusive (A : Prop) : contrary A → mutually_exclusive A (¬A) :=
by sorry

theorem contrary_sufficient_but_not_necessary (A B : Prop) :
  (∃ (A : Prop), contrary A) → mutually_exclusive A B →
  (∃ (A : Prop), contrary A ∧ mutually_exclusive A B) :=
by sorry

end contrary_implies_mutually_exclusive_contrary_sufficient_but_not_necessary_l953_95366


namespace additive_inverse_commutativity_l953_95395

section
  variable {R : Type} [Ring R] (h : ∀ x : R, x ^ 2 = x)

  theorem additive_inverse (x : R) : -x = x := by
    sorry

  theorem commutativity (x y : R) : x * y = y * x := by
    sorry
end

end additive_inverse_commutativity_l953_95395


namespace meal_arrangement_exactly_two_correct_l953_95349

noncomputable def meal_arrangement_count : ℕ :=
  let total_people := 13
  let meal_types := ["B", "B", "B", "B", "C", "C", "C", "F", "F", "F", "V", "V", "V"]
  let choose_2_people := (total_people.choose 2)
  let derangement_7 := 1854  -- Derangement of BBCCCVVV
  let derangement_9 := 133496  -- Derangement of BBCCFFFVV
  choose_2_people * (derangement_7 + derangement_9)

theorem meal_arrangement_exactly_two_correct : meal_arrangement_count = 10482600 := by
  sorry

end meal_arrangement_exactly_two_correct_l953_95349


namespace max_area_of_fenced_rectangle_l953_95399

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l953_95399


namespace lcm_852_1491_l953_95358

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end lcm_852_1491_l953_95358


namespace quadratic_trinomial_prime_l953_95346

theorem quadratic_trinomial_prime (p x : ℤ) (hp : p > 1) (hx : 0 ≤ x ∧ x < p)
  (h_prime : Prime (x^2 - x + p)) : x = 0 ∨ x = 1 :=
by
  sorry

end quadratic_trinomial_prime_l953_95346


namespace time_for_D_to_complete_job_l953_95393

-- Definitions for conditions
def A_rate : ℚ := 1 / 6
def combined_rate : ℚ := 1 / 4

-- We need to find D_rate
def D_rate : ℚ := combined_rate - A_rate

-- Now we state the theorem
theorem time_for_D_to_complete_job :
  D_rate = 1 / 12 :=
by
  /-
  We want to show that given the conditions:
  1. A_rate = 1 / 6
  2. A_rate + D_rate = 1 / 4
  it results in D_rate = 1 / 12.
  -/
  sorry

end time_for_D_to_complete_job_l953_95393


namespace tan_sum_l953_95300

theorem tan_sum (x y : ℝ)
  (h1 : Real.sin x + Real.sin y = 72 / 65)
  (h2 : Real.cos x + Real.cos y = 96 / 65) : 
  Real.tan x + Real.tan y = 868 / 112 := 
by sorry

end tan_sum_l953_95300


namespace regression_line_equation_l953_95351

-- Define the conditions in the problem
def slope_of_regression_line : ℝ := 1.23
def center_of_sample_points : ℝ × ℝ := (4, 5)

-- The proof problem to show that the equation of the regression line is y = 1.23x + 0.08
theorem regression_line_equation :
  ∃ b : ℝ, (∀ x y : ℝ, (y = slope_of_regression_line * x + b) 
  → (4, 5) = (x, y)) → b = 0.08 :=
sorry

end regression_line_equation_l953_95351


namespace cost_of_pen_l953_95355

theorem cost_of_pen :
  ∃ p q : ℚ, (3 * p + 4 * q = 264) ∧ (4 * p + 2 * q = 230) ∧ (p = 39.2) :=
by
  sorry

end cost_of_pen_l953_95355


namespace min_value_inequality_equality_condition_l953_95303

theorem min_value_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1)) ≥ 8 :=
sorry

theorem equality_condition (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1) = 8) ↔ ((a = 2) ∧ (b = 2)) :=
sorry

end min_value_inequality_equality_condition_l953_95303


namespace angle_A_range_l953_95391

open Real

theorem angle_A_range (A : ℝ) (h1 : sin A + cos A > 0) (h2 : tan A < sin A) (h3 : 0 < A ∧ A < π) : 
  π / 2 < A ∧ A < 3 * π / 4 :=
by
  sorry

end angle_A_range_l953_95391


namespace order_of_nums_l953_95385

variable (a b : ℝ)

theorem order_of_nums (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := 
sorry

end order_of_nums_l953_95385


namespace cos_beta_of_tan_alpha_and_sin_alpha_plus_beta_l953_95374

theorem cos_beta_of_tan_alpha_and_sin_alpha_plus_beta 
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_tanα : Real.tan α = 3) (h_sin_alpha_beta : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 10 / 10 := 
sorry

end cos_beta_of_tan_alpha_and_sin_alpha_plus_beta_l953_95374


namespace pumps_280_gallons_in_30_minutes_l953_95304

def hydraflow_rate_per_hour := 560 -- gallons per hour
def time_fraction_in_hour := 1 / 2

theorem pumps_280_gallons_in_30_minutes : hydraflow_rate_per_hour * time_fraction_in_hour = 280 := by
  sorry

end pumps_280_gallons_in_30_minutes_l953_95304


namespace calc_pow_expression_l953_95334

theorem calc_pow_expression : (27^3 * 9^2) / 3^15 = 1 / 9 := 
by sorry

end calc_pow_expression_l953_95334


namespace arielle_age_l953_95340

theorem arielle_age (E A : ℕ) (h1 : E = 10) (h2 : E + A + E * A = 131) : A = 11 := by 
  sorry

end arielle_age_l953_95340


namespace joes_fast_food_cost_l953_95323

noncomputable def cost_of_sandwich (n : ℕ) : ℝ := n * 4
noncomputable def cost_of_soda (m : ℕ) : ℝ := m * 1.50
noncomputable def total_cost (n m : ℕ) : ℝ :=
  if n >= 10 then cost_of_sandwich n - 5 + cost_of_soda m else cost_of_sandwich n + cost_of_soda m

theorem joes_fast_food_cost :
  total_cost 10 6 = 44 := by
  sorry

end joes_fast_food_cost_l953_95323


namespace first_digit_of_base16_representation_l953_95311

-- Firstly we define the base conversion from base 4 to base 10 and from base 10 to base 16.
-- For simplicity, we assume that the required functions exist and skip their implementations.

-- Assume base 4 to base 10 conversion function
def base4_to_base10 (n : String) : Nat :=
  sorry

-- Assume base 10 to base 16 conversion function that gives the first digit
def first_digit_base16 (n : Nat) : Nat :=
  sorry

-- Given the base 4 number as string
def y_base4 : String := "20313320132220312031"

-- Define the final statement
theorem first_digit_of_base16_representation :
  first_digit_base16 (base4_to_base10 y_base4) = 5 :=
by
  sorry

end first_digit_of_base16_representation_l953_95311


namespace area_ratio_none_of_these_l953_95352

theorem area_ratio_none_of_these (h r a : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) (a_pos : 0 < a) (h_square_a_square : h^2 > a^2) :
  ¬ (∃ ratio, ratio = (π * r / (h + r)) ∨
               ratio = (π * r^2 / (a + h)) ∨
               ratio = (π * a * r / (h + 2 * r)) ∨
               ratio = (π * r / (a + r))) :=
by sorry

end area_ratio_none_of_these_l953_95352


namespace unattainable_y_value_l953_95373

theorem unattainable_y_value (y : ℝ) (x : ℝ) (h : x ≠ -4 / 3) : ¬ (y = -1 / 3) :=
by {
  -- The proof is omitted for now. 
  -- We're only constructing the outline with necessary imports and conditions.
  sorry
}

end unattainable_y_value_l953_95373


namespace marbles_percentage_l953_95376

def solid_color_other_than_yellow (total_marbles : ℕ) (solid_color_percent solid_yellow_percent : ℚ) : ℚ :=
  solid_color_percent - solid_yellow_percent

theorem marbles_percentage (total_marbles : ℕ) (solid_color_percent solid_yellow_percent : ℚ) :
  solid_color_percent = 90 / 100 →
  solid_yellow_percent = 5 / 100 →
  solid_color_other_than_yellow total_marbles solid_color_percent solid_yellow_percent = 85 / 100 :=
by
  intro h1 h2
  rw [h1, h2]
  norm_num
  sorry

end marbles_percentage_l953_95376


namespace sec_neg_450_undefined_l953_95365

theorem sec_neg_450_undefined : ¬ ∃ x, x = 1 / Real.cos (-450 * Real.pi / 180) :=
by
  -- Proof skipped using 'sorry'
  sorry

end sec_neg_450_undefined_l953_95365


namespace evaluate_expression_l953_95359

theorem evaluate_expression (x : ℕ) (h : x = 3) : 5^3 - 2^x * 3 + 4^2 = 117 :=
by
  rw [h]
  sorry

end evaluate_expression_l953_95359


namespace age_ratio_l953_95389

/-- Given that Sandy's age after 6 years will be 30 years,
    and Molly's current age is 18 years, 
    prove that the current ratio of Sandy's age to Molly's age is 4:3. -/
theorem age_ratio (M S : ℕ) 
  (h1 : M = 18) 
  (h2 : S + 6 = 30) : 
  S / gcd S M = 4 ∧ M / gcd S M = 3 :=
by
  sorry

end age_ratio_l953_95389


namespace basket_A_apples_count_l953_95344

-- Conditions
def total_baskets : ℕ := 5
def avg_fruits_per_basket : ℕ := 25
def fruits_in_B : ℕ := 30
def fruits_in_C : ℕ := 20
def fruits_in_D : ℕ := 25
def fruits_in_E : ℕ := 35

-- Calculation of total number of fruits
def total_fruits : ℕ := total_baskets * avg_fruits_per_basket
def other_baskets_fruits : ℕ := fruits_in_B + fruits_in_C + fruits_in_D + fruits_in_E

-- Question and Proof Goal
theorem basket_A_apples_count : total_fruits - other_baskets_fruits = 15 := by
  sorry

end basket_A_apples_count_l953_95344


namespace brandon_textbooks_weight_l953_95377

-- Define the weights of Jon's textbooks
def jon_textbooks : List ℕ := [2, 8, 5, 9]

-- Define the weight ratio between Jon's and Brandon's textbooks
def weight_ratio : ℕ := 3

-- Define the total weight of Jon's textbooks
def weight_jon : ℕ := jon_textbooks.sum

-- Define the weight of Brandon's textbooks to be proven
def weight_brandon : ℕ := weight_jon / weight_ratio

-- The theorem to be proven
theorem brandon_textbooks_weight : weight_brandon = 8 :=
by sorry

end brandon_textbooks_weight_l953_95377


namespace Seokhyung_drank_the_most_l953_95360

-- Define the conditions
def Mina_Amount := 0.6
def Seokhyung_Amount := 1.5
def Songhwa_Amount := Seokhyung_Amount - 0.6

-- Statement to prove that Seokhyung drank the most cola
theorem Seokhyung_drank_the_most : Seokhyung_Amount > Mina_Amount ∧ Seokhyung_Amount > Songhwa_Amount :=
by
  -- Proof skipped
  sorry

end Seokhyung_drank_the_most_l953_95360


namespace initial_condition_proof_move_to_1_proof_move_to_2_proof_recurrence_relation_proof_p_99_proof_p_100_proof_l953_95310

variable (p : ℕ → ℚ)

-- Given conditions
axiom initial_condition : p 0 = 1
axiom move_to_1 : p 1 = 1 / 2
axiom move_to_2 : p 2 = 3 / 4
axiom recurrence_relation : ∀ n : ℕ, 2 ≤ n → n ≤ 99 → p n - p (n - 1) = - 1 / 2 * (p (n - 1) - p (n - 2))
axiom p_99_cond : p 99 = 2 / 3 - 1 / (3 * 2^99)
axiom p_100_cond : p 100 = 1 / 3 + 1 / (3 * 2^99)

-- Proof that initial conditions are met
theorem initial_condition_proof : p 0 = 1 :=
sorry

theorem move_to_1_proof : p 1 = 1 / 2 :=
sorry

theorem move_to_2_proof : p 2 = 3 / 4 :=
sorry

-- Proof of the recurrence relation
theorem recurrence_relation_proof : ∀ n : ℕ, 2 ≤ n → n ≤ 99 → p n - p (n - 1) = - 1 / 2 * (p (n - 1) - p (n - 2)) :=
sorry

-- Proof of p_99
theorem p_99_proof : p 99 = 2 / 3 - 1 / (3 * 2^99) :=
sorry

-- Proof of p_100
theorem p_100_proof : p 100 = 1 / 3 + 1 / (3 * 2^99) :=
sorry

end initial_condition_proof_move_to_1_proof_move_to_2_proof_recurrence_relation_proof_p_99_proof_p_100_proof_l953_95310


namespace tuesday_more_than_monday_l953_95369

variable (M T W Th x : ℕ)

-- Conditions
def monday_dinners : M = 40 := by sorry
def tuesday_dinners : T = M + x := by sorry
def wednesday_dinners : W = T / 2 := by sorry
def thursday_dinners : Th = W + 3 := by sorry
def total_dinners : M + T + W + Th = 203 := by sorry

-- Proof problem: How many more dinners were sold on Tuesday than on Monday?
theorem tuesday_more_than_monday : x = 32 :=
by
  sorry

end tuesday_more_than_monday_l953_95369


namespace ping_pong_ball_probability_l953_95336

noncomputable def multiple_of_6_9_or_both_probability : ℚ :=
  let total_numbers := 72
  let multiples_of_6 := 12
  let multiples_of_9 := 8
  let multiples_of_both := 4
  (multiples_of_6 + multiples_of_9 - multiples_of_both) / total_numbers

theorem ping_pong_ball_probability :
  multiple_of_6_9_or_both_probability = 2 / 9 :=
by
  sorry

end ping_pong_ball_probability_l953_95336


namespace Iain_pennies_problem_l953_95357

theorem Iain_pennies_problem :
  ∀ (P : ℝ), 200 - 30 = 170 →
             170 - (P / 100) * 170 = 136 →
             P = 20 :=
by
  intros P h1 h2
  sorry

end Iain_pennies_problem_l953_95357


namespace intersection_of_A_and_B_l953_95308

def setA : Set ℝ := { x | x - 2 ≥ 0 }
def setB : Set ℝ := { x | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 2 }

theorem intersection_of_A_and_B :
  setA ∩ setB = { x | 2 ≤ x ∧ x < 4 } :=
sorry

end intersection_of_A_and_B_l953_95308


namespace lowest_temperature_in_january_2023_l953_95386

theorem lowest_temperature_in_january_2023 
  (T_Beijing T_Shanghai T_Shenzhen T_Jilin : ℝ)
  (h_Beijing : T_Beijing = -5)
  (h_Shanghai : T_Shanghai = 6)
  (h_Shenzhen : T_Shenzhen = 19)
  (h_Jilin : T_Jilin = -22) :
  T_Jilin < T_Beijing ∧ T_Jilin < T_Shanghai ∧ T_Jilin < T_Shenzhen :=
by
  sorry

end lowest_temperature_in_january_2023_l953_95386


namespace problem_final_value_l953_95388

theorem problem_final_value (x y z : ℝ) (hz : z ≠ 0) 
  (h1 : 3 * x - 2 * y - 2 * z = 0) 
  (h2 : x - 4 * y + 8 * z = 0) :
  (3 * x^2 - 2 * x * y) / (y^2 + 4 * z^2) = 120 / 269 := 
by 
  sorry

end problem_final_value_l953_95388


namespace molecular_weight_calc_l953_95317

theorem molecular_weight_calc (total_weight : ℕ) (num_moles : ℕ) (one_mole_weight : ℕ) :
  total_weight = 1170 → num_moles = 5 → one_mole_weight = total_weight / num_moles → one_mole_weight = 234 :=
by
  intros h1 h2 h3
  sorry

end molecular_weight_calc_l953_95317
