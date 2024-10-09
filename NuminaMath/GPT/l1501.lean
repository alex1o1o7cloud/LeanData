import Mathlib

namespace hawks_total_points_l1501_150167

def touchdowns : ℕ := 3
def points_per_touchdown : ℕ := 7
def total_points (t : ℕ) (p : ℕ) : ℕ := t * p

theorem hawks_total_points : total_points touchdowns points_per_touchdown = 21 := 
by 
  sorry

end hawks_total_points_l1501_150167


namespace ratio_of_wealth_l1501_150105

theorem ratio_of_wealth (W P : ℝ) 
  (h1 : 0 < P) (h2 : 0 < W) 
  (pop_X : ℝ := 0.4 * P) 
  (wealth_X : ℝ := 0.6 * W) 
  (top50_pop_X : ℝ := 0.5 * pop_X) 
  (top50_wealth_X : ℝ := 0.8 * wealth_X) 
  (pop_Y : ℝ := 0.2 * P) 
  (wealth_Y : ℝ := 0.3 * W) 
  (avg_wealth_top50_X : ℝ := top50_wealth_X / top50_pop_X) 
  (avg_wealth_Y : ℝ := wealth_Y / pop_Y) : 
  avg_wealth_top50_X / avg_wealth_Y = 1.6 := 
by sorry

end ratio_of_wealth_l1501_150105


namespace trajectory_of_P_is_right_branch_of_hyperbola_l1501_150147

-- Definitions of the given points F1 and F2
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Definition of the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Definition of point P satisfying the condition
def P (x y : ℝ) : Prop :=
  abs (distance (x, y) F1 - distance (x, y) F2) = 8

-- Trajectory of point P is the right branch of the hyperbola
theorem trajectory_of_P_is_right_branch_of_hyperbola :
  ∀ (x y : ℝ), P x y → True := -- Trajectory is hyperbola (right branch)
by
  sorry

end trajectory_of_P_is_right_branch_of_hyperbola_l1501_150147


namespace teorema_dos_bicos_white_gray_eq_angle_x_l1501_150137

-- Define the problem statement
theorem teorema_dos_bicos_white_gray_eq
    (n : ℕ)
    (AB CD : ℝ)
    (peaks : Fin n → ℝ)
    (white_angles gray_angles : Fin n → ℝ)
    (h_parallel : AB = CD)
    (h_white_angles : ∀ i, white_angles i = peaks i)
    (h_gray_angles : ∀ i, gray_angles i = peaks i):
    (Finset.univ.sum white_angles) = (Finset.univ.sum gray_angles) := sorry

theorem angle_x
    (AB CD : ℝ)
    (x : ℝ)
    (h_parallel : AB = CD):
    x = 32 := sorry

end teorema_dos_bicos_white_gray_eq_angle_x_l1501_150137


namespace positive_difference_of_two_numbers_l1501_150155

theorem positive_difference_of_two_numbers :
  ∃ (x y : ℤ), (x + y = 40) ∧ (3 * y - 2 * x = 8) ∧ (|y - x| = 4) :=
by
  sorry

end positive_difference_of_two_numbers_l1501_150155


namespace salt_solution_concentration_l1501_150166

theorem salt_solution_concentration :
  ∀ (C : ℝ),
  (∀ (mix_vol : ℝ) (pure_water : ℝ) (salt_solution_vol : ℝ),
    mix_vol = 1.5 →
    pure_water = 1 →
    salt_solution_vol = 0.5 →
    1.5 * 0.15 = 0.5 * (C / 100) →
    C = 45) :=
by
  intros C mix_vol pure_water salt_solution_vol h_mix h_pure h_salt h_eq
  sorry

end salt_solution_concentration_l1501_150166


namespace find_hourly_rate_l1501_150143

-- Defining the conditions
def hours_worked : ℝ := 7.5
def overtime_factor : ℝ := 1.5
def total_hours_worked : ℝ := 10.5
def total_earnings : ℝ := 48

-- Proving the hourly rate
theorem find_hourly_rate (R : ℝ) (h : 7.5 * R + (10.5 - 7.5) * 1.5 * R = 48) : R = 4 := by
  sorry

end find_hourly_rate_l1501_150143


namespace tickets_sold_correctly_l1501_150111

theorem tickets_sold_correctly :
  let total := 620
  let cost_per_ticket := 4
  let tickets_sold := 155
  total / cost_per_ticket = tickets_sold :=
by
  sorry

end tickets_sold_correctly_l1501_150111


namespace cubic_sum_l1501_150102

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end cubic_sum_l1501_150102


namespace fraction_equals_decimal_l1501_150184

theorem fraction_equals_decimal : (3 : ℝ) / 2 = 1.5 := 
sorry

end fraction_equals_decimal_l1501_150184


namespace Eunji_score_equals_56_l1501_150169

theorem Eunji_score_equals_56 (Minyoung_score Yuna_score : ℕ) (Eunji_score : ℕ) 
  (h1 : Minyoung_score = 55) (h2 : Yuna_score = 57)
  (h3 : Eunji_score > Minyoung_score) (h4 : Eunji_score < Yuna_score) : Eunji_score = 56 := by
  -- Given the hypothesis, it is a fact that Eunji's score is 56.
  sorry

end Eunji_score_equals_56_l1501_150169


namespace sum_PS_TV_l1501_150136

theorem sum_PS_TV 
  (P V : ℝ) 
  (hP : P = 3) 
  (hV : V = 33)
  (n : ℕ) 
  (hn : n = 6) 
  (Q R S T U : ℝ) 
  (hPR : P < Q ∧ Q < R ∧ R < S ∧ S < T ∧ T < U ∧ U < V)
  (h_divide : ∀ i : ℕ, i ≤ n → P + i * (V - P) / n = P + i * 5) :
  (P, V, Q, R, S, T, U) = (3, 33, 8, 13, 18, 23, 28) → (S - P) + (V - T) = 25 :=
by {
  sorry
}

end sum_PS_TV_l1501_150136


namespace cupcakes_per_package_calculation_l1501_150197

noncomputable def sarah_total_cupcakes := 38
noncomputable def cupcakes_eaten_by_todd := 14
noncomputable def number_of_packages := 3
noncomputable def remaining_cupcakes := sarah_total_cupcakes - cupcakes_eaten_by_todd
noncomputable def cupcakes_per_package := remaining_cupcakes / number_of_packages

theorem cupcakes_per_package_calculation : cupcakes_per_package = 8 := by
  sorry

end cupcakes_per_package_calculation_l1501_150197


namespace li_bai_initial_wine_l1501_150106

theorem li_bai_initial_wine (x : ℕ) 
  (h : (((((x * 2 - 2) * 2 - 2) * 2 - 2) * 2 - 2) = 2)) : 
  x = 2 :=
by
  sorry

end li_bai_initial_wine_l1501_150106


namespace solution_set_empty_range_a_l1501_150185

theorem solution_set_empty_range_a (a : ℝ) :
  (∀ x : ℝ, ¬((a - 1) * x^2 + 2 * (a - 1) * x - 4 ≥ 0)) ↔ -3 < a ∧ a ≤ 1 :=
by
  sorry

end solution_set_empty_range_a_l1501_150185


namespace conditional_prob_correct_l1501_150160

/-- Define the events A and B as per the problem -/
def event_A (x y : ℕ) : Prop := (x + y) % 2 = 0

def event_B (x y : ℕ) : Prop := (x % 2 = 0 ∨ y % 2 = 0) ∧ x ≠ y

/-- Define the probability of event A -/
def prob_A : ℚ := 1 / 2

/-- Define the combined probability of both events A and B occurring -/
def prob_A_and_B : ℚ := 1 / 6

/-- Calculate the conditional probability P(B | A) -/
def conditional_prob : ℚ := prob_A_and_B / prob_A

theorem conditional_prob_correct : conditional_prob = 1 / 3 := by
  -- This is where you would provide the proof if required
  sorry

end conditional_prob_correct_l1501_150160


namespace f_leq_2x_l1501_150162

noncomputable def f : ℝ → ℝ := sorry
axiom f_nonneg {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : 0 ≤ f x
axiom f_one : f 1 = 1
axiom f_superadditive {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hxy : x + y ≤ 1) : f (x + y) ≥ f x + f y

-- The theorem statement to be proved
theorem f_leq_2x {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : f x ≤ 2 * x := sorry

end f_leq_2x_l1501_150162


namespace regular_octagon_angle_ABG_l1501_150129

-- Definition of a regular octagon
structure RegularOctagon (V : Type) :=
(vertices : Fin 8 → V)

def angleABG (O : RegularOctagon ℝ) : ℝ :=
  22.5

-- The statement: In a regular octagon ABCDEFGH, the measure of ∠ABG is 22.5°
theorem regular_octagon_angle_ABG (O : RegularOctagon ℝ) : angleABG O = 22.5 :=
  sorry

end regular_octagon_angle_ABG_l1501_150129


namespace max_a_such_that_f_geq_a_min_value_under_constraint_l1501_150180

-- Problem (1)
theorem max_a_such_that_f_geq_a :
  ∃ (a : ℝ), (∀ (x : ℝ), |x - (5/2)| + |x - a| ≥ a) ∧ a = 5 / 4 := sorry

-- Problem (2)
theorem min_value_under_constraint :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2 * y + 3 * z = 1 ∧
  (3 / x + 2 / y + 1 / z) = 16 + 8 * Real.sqrt 3 := sorry

end max_a_such_that_f_geq_a_min_value_under_constraint_l1501_150180


namespace complement_of_A_is_negatives_l1501_150114

theorem complement_of_A_is_negatives :
  let U := Set.univ (α := ℝ)
  let A := {x : ℝ | x ≥ 0}
  (U \ A) = {x : ℝ | x < 0} :=
by
  sorry

end complement_of_A_is_negatives_l1501_150114


namespace sum_of_coefficients_l1501_150158

noncomputable def polynomial (x : ℝ) : ℝ := x^3 + 3*x^2 - 4*x - 12
noncomputable def simplified_polynomial (x : ℝ) (A B C : ℝ) : ℝ := A*x^2 + B*x + C

theorem sum_of_coefficients : 
  ∃ (A B C D : ℝ), 
    (∀ x ≠ D, simplified_polynomial x A B C = (polynomial x) / (x + 3)) ∧ 
    (A + B + C + D = -6) :=
by
  sorry

end sum_of_coefficients_l1501_150158


namespace smallest_positive_integer_cube_ends_368_l1501_150179

theorem smallest_positive_integer_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 368 ∧ n = 34 :=
by
  sorry

end smallest_positive_integer_cube_ends_368_l1501_150179


namespace apples_in_basket_l1501_150115

theorem apples_in_basket (x : ℕ) (h1 : 22 * x = (x + 45) * 13) : 22 * x = 1430 :=
by
  sorry

end apples_in_basket_l1501_150115


namespace calculate_expression_l1501_150126

theorem calculate_expression :
  36 + (150 / 15) + (12 ^ 2 * 5) - 300 - (270 / 9) = 436 := by
  sorry

end calculate_expression_l1501_150126


namespace graph_f_intersects_x_eq_1_at_most_once_l1501_150176

-- Define a function f from ℝ to ℝ
def f : ℝ → ℝ := sorry  -- Placeholder for the actual function

-- Define the domain of the function f (it's a generic function on ℝ for simplicity)
axiom f_unique : ∀ x y : ℝ, f x = f y → x = y  -- If f(x) = f(y), then x must equal y

-- Prove that the graph of y = f(x) intersects the line x = 1 at most once
theorem graph_f_intersects_x_eq_1_at_most_once : ∃ y : ℝ, (f 1 = y) ∨ (¬∃ y : ℝ, f 1 = y) :=
by
  -- Proof goes here
  sorry

end graph_f_intersects_x_eq_1_at_most_once_l1501_150176


namespace max_ratio_square_l1501_150135

variables {a b c x y : ℝ}
-- Assume a, b, c are positive real numbers
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
-- Assume the order of a, b, c: a ≥ b ≥ c
variable (h_order : a ≥ b ∧ b ≥ c)
-- Define the system of equations
variable (h_system : a^2 + y^2 = c^2 + x^2 ∧ c^2 + x^2 = (a - x)^2 + (c - y)^2)
-- Assume the constraints on x and y
variable (h_constraints : 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < c)

theorem max_ratio_square :
  ∃ (ρ : ℝ), ρ = (a / c) ∧ ρ^2 = 4 / 3 :=
sorry

end max_ratio_square_l1501_150135


namespace fifth_student_gold_stickers_l1501_150187

theorem fifth_student_gold_stickers :
  ∀ s1 s2 s3 s4 s5 s6 : ℕ,
  s1 = 29 →
  s2 = 35 →
  s3 = 41 →
  s4 = 47 →
  s6 = 59 →
  (s2 - s1 = 6) →
  (s3 - s2 = 6) →
  (s4 - s3 = 6) →
  (s6 - s4 = 12) →
  s5 = s4 + (s2 - s1) →
  s5 = 53 := by
  intros s1 s2 s3 s4 s5 s6 hs1 hs2 hs3 hs4 hs6 hd1 hd2 hd3 hd6 heq
  subst_vars
  sorry

end fifth_student_gold_stickers_l1501_150187


namespace find_x_when_y_is_20_l1501_150133

-- Definition of the problem conditions.
def constant_ratio (x y : ℝ) : Prop := ∃ k, (3 * x - 4) = k * (y + 7)

-- Main theorem statement.
theorem find_x_when_y_is_20 :
  (constant_ratio x 5 → constant_ratio 3 5) → 
  (constant_ratio x 20 → x = 5.0833) :=
  by sorry

end find_x_when_y_is_20_l1501_150133


namespace compute_expr_l1501_150183

-- Definitions
def a := 150 / 5
def b := 40 / 8
def c := 16 / 32
def d := 3

def expr := 20 * (a - b + c + d)

-- Theorem
theorem compute_expr : expr = 570 :=
by
  sorry

end compute_expr_l1501_150183


namespace melissa_total_commission_l1501_150142

def sale_price_coupe : ℝ := 30000
def sale_price_suv : ℝ := 2 * sale_price_coupe
def sale_price_luxury_sedan : ℝ := 80000

def commission_rate_coupe_and_suv : ℝ := 0.02
def commission_rate_luxury_sedan : ℝ := 0.03

def commission (rate : ℝ) (price : ℝ) : ℝ := rate * price

def total_commission : ℝ :=
  commission commission_rate_coupe_and_suv sale_price_coupe +
  commission commission_rate_coupe_and_suv sale_price_suv +
  commission commission_rate_luxury_sedan sale_price_luxury_sedan

theorem melissa_total_commission :
  total_commission = 4200 := by
  sorry

end melissa_total_commission_l1501_150142


namespace fred_baseball_cards_l1501_150110

variable (initial_cards : ℕ)
variable (bought_cards : ℕ)

theorem fred_baseball_cards (h1 : initial_cards = 5) (h2 : bought_cards = 3) : initial_cards - bought_cards = 2 := by
  sorry

end fred_baseball_cards_l1501_150110


namespace largest_consecutive_multiple_of_3_l1501_150188

theorem largest_consecutive_multiple_of_3 (n : ℕ) 
  (h : 3 * n + 3 * (n + 1) + 3 * (n + 2) = 72) : 3 * (n + 2) = 27 :=
by 
  sorry

end largest_consecutive_multiple_of_3_l1501_150188


namespace two_buttons_diff_size_color_l1501_150107

variables (box : Type) 
variable [Finite box]
variables (Big Small White Black : box → Prop)

axiom big_ex : ∃ x, Big x
axiom small_ex : ∃ x, Small x
axiom white_ex : ∃ x, White x
axiom black_ex : ∃ x, Black x
axiom size : ∀ x, Big x ∨ Small x
axiom color : ∀ x, White x ∨ Black x

theorem two_buttons_diff_size_color : 
  ∃ x y, x ≠ y ∧ (Big x ∧ Small y ∨ Small x ∧ Big y) ∧ (White x ∧ Black y ∨ Black x ∧ White y) := 
by
  sorry

end two_buttons_diff_size_color_l1501_150107


namespace triple_solutions_l1501_150146

theorem triple_solutions (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 2 ∧ b = 2 ∧ c = 2) ↔ a! + b! = 2 ^ c! :=
by
  sorry

end triple_solutions_l1501_150146


namespace units_digit_3968_805_l1501_150123

theorem units_digit_3968_805 : 
  (3968 ^ 805) % 10 = 8 := 
by
  -- Proof goes here
  sorry

end units_digit_3968_805_l1501_150123


namespace digits_difference_l1501_150175

theorem digits_difference (d A B : ℕ) (h1 : d > 6) (h2 : (B + A) * d + 2 * A = d^2 + 7 * d + 2)
  (h3 : B + A = 10) (h4 : 2 * A = 8) : A - B = 3 :=
by 
  sorry

end digits_difference_l1501_150175


namespace find_fraction_result_l1501_150104

open Complex

theorem find_fraction_result (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
    (h1 : x + y + z = 30)
    (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
    (x^3 + y^3 + z^3) / (x * y * z) = 33 := 
    sorry

end find_fraction_result_l1501_150104


namespace probability_all_quitters_same_tribe_l1501_150178

theorem probability_all_quitters_same_tribe :
  ∀ (people : Finset ℕ) (tribe1 tribe2 : Finset ℕ) (choose : ℕ → ℕ → ℕ) (prob : ℚ),
  people.card = 20 →
  tribe1.card = 10 →
  tribe2.card = 10 →
  tribe1 ∪ tribe2 = people →
  tribe1 ∩ tribe2 = ∅ →
  choose 20 3 = 1140 →
  choose 10 3 = 120 →
  prob = (2 * choose 10 3) / choose 20 3 →
  prob = 20 / 95 :=
by
  intro people tribe1 tribe2 choose prob
  intros hp20 ht1 ht2 hu hi hchoose20 hchoose10 hprob
  sorry

end probability_all_quitters_same_tribe_l1501_150178


namespace min_diff_proof_l1501_150134

noncomputable def triangleMinDiff : ℕ :=
  let PQ := 666
  let QR := 667
  let PR := 2010 - PQ - QR
  if (PQ < QR ∧ QR < PR ∧ PQ + QR > PR ∧ PQ + PR > QR ∧ PR + QR > PQ) then QR - PQ else 0

theorem min_diff_proof :
  ∃ PQ QR PR : ℕ, PQ + QR + PR = 2010 ∧ PQ < QR ∧ QR < PR ∧ (PQ + QR > PR) ∧ (PQ + PR > QR) ∧ (PR + QR > PQ) ∧ (QR - PQ = triangleMinDiff) := sorry

end min_diff_proof_l1501_150134


namespace math_club_members_count_l1501_150181

theorem math_club_members_count 
    (n_books : ℕ) 
    (n_borrow_each_member : ℕ) 
    (n_borrow_each_book : ℕ) 
    (total_borrow_count_books : n_books * n_borrow_each_book = 36) 
    (total_borrow_count_members : 2 * x = 36) 
    : x = 18 := 
by
  sorry

end math_club_members_count_l1501_150181


namespace like_terms_m_eq_2_l1501_150170

theorem like_terms_m_eq_2 (m : ℕ) :
  (∀ (x y : ℝ), 3 * x^m * y^3 = 3 * x^2 * y^3) -> m = 2 :=
by
  intros _
  sorry

end like_terms_m_eq_2_l1501_150170


namespace distinct_flavors_count_l1501_150117

theorem distinct_flavors_count (red_candies : ℕ) (green_candies : ℕ)
  (h_red : red_candies = 0 ∨ red_candies = 1 ∨ red_candies = 2 ∨ red_candies = 3 ∨ red_candies = 4 ∨ red_candies = 5 ∨ red_candies = 6)
  (h_green : green_candies = 0 ∨ green_candies = 1 ∨ green_candies = 2 ∨ green_candies = 3 ∨ green_candies = 4 ∨ green_candies = 5) :
  ∃ unique_flavors : Finset (ℚ), unique_flavors.card = 25 :=
by
  sorry

end distinct_flavors_count_l1501_150117


namespace selection_ways_l1501_150119

namespace CulturalPerformance

-- Define basic conditions
def num_students : ℕ := 6
def can_sing : ℕ := 3
def can_dance : ℕ := 2
def both_sing_and_dance : ℕ := 1

-- Define the proof statement
theorem selection_ways :
  ∃ (ways : ℕ), ways = 15 := by
  sorry

end CulturalPerformance

end selection_ways_l1501_150119


namespace f_2_eq_4_l1501_150121

def f (n : ℕ) : ℕ := (List.range (n + 1)).sum + (List.range n).sum

theorem f_2_eq_4 : f 2 = 4 := by
  sorry

end f_2_eq_4_l1501_150121


namespace fg_neg_two_l1501_150173

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x + 3

theorem fg_neg_two : f (g (-2)) = 2 := by
  sorry

end fg_neg_two_l1501_150173


namespace minimum_p_l1501_150152

-- Define the problem constants and conditions
noncomputable def problem_statement :=
  ∃ p q : ℕ, 
    0 < p ∧ 0 < q ∧ 
    (2008 / 2009 < p / (q : ℚ)) ∧ (p / (q : ℚ) < 2009 / 2010) ∧ 
    (∀ p' q' : ℕ, (0 < p' ∧ 0 < q' ∧ (2008 / 2009 < p' / (q' : ℚ)) ∧ (p' / (q' : ℚ) < 2009 / 2010)) → p ≤ p') 

-- The proof
theorem minimum_p (h : problem_statement) :
  ∃ p q : ℕ, 
    0 < p ∧ 0 < q ∧ 
    (2008 / 2009 < p / (q : ℚ)) ∧ (p / (q : ℚ) < 2009 / 2010) ∧
    p = 4017 :=
sorry

end minimum_p_l1501_150152


namespace tangent_line_at_1_l1501_150120

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, f 1)

-- Define the slope of the tangent line at x=1
def slope_at_1 : ℝ := f' 1

-- Define the tangent line equation at x=1
def tangent_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Theorem that the tangent line to f at x=1 is 2x - y + 1 = 0
theorem tangent_line_at_1 :
  tangent_line 1 (f 1) :=
by
  sorry

end tangent_line_at_1_l1501_150120


namespace fraction_is_two_thirds_l1501_150198

noncomputable def fraction_of_price_of_ballet_slippers (f : ℚ) : Prop :=
  let price_high_heels := 60
  let num_ballet_slippers := 5
  let total_cost := 260
  price_high_heels + num_ballet_slippers * f * price_high_heels = total_cost

theorem fraction_is_two_thirds : fraction_of_price_of_ballet_slippers (2 / 3) := by
  sorry

end fraction_is_two_thirds_l1501_150198


namespace plastic_bag_co2_release_l1501_150177

def total_co2_canvas_bag_lb : ℕ := 600
def total_co2_canvas_bag_oz : ℕ := 9600
def plastic_bags_per_trip : ℕ := 8
def shopping_trips : ℕ := 300

theorem plastic_bag_co2_release :
  total_co2_canvas_bag_oz = 2400 * 4 :=
by
  sorry

end plastic_bag_co2_release_l1501_150177


namespace find_integer_k_l1501_150125

theorem find_integer_k (k : ℤ) : (∃ k : ℤ, (k = 6) ∨ (k = 2) ∨ (k = 0) ∨ (k = -4)) ↔ (∃ k : ℤ, (2 * k^2 + k - 8) % (k - 1) = 0) :=
by
  sorry

end find_integer_k_l1501_150125


namespace smallest_b_for_q_ge_half_l1501_150112

open Nat

def binomial (n k : ℕ) : ℕ := if h : k ≤ n then n.choose k else 0

def q (b : ℕ) : ℚ := (binomial (32 - b) 2 + binomial (b - 1) 2) / (binomial 38 2 : ℕ)

theorem smallest_b_for_q_ge_half : ∃ (b : ℕ), b = 18 ∧ q b ≥ 1 / 2 :=
by
  -- Prove and find the smallest b such that q(b) ≥ 1/2
  sorry

end smallest_b_for_q_ge_half_l1501_150112


namespace piesEatenWithForksPercentage_l1501_150100

def totalPies : ℕ := 2000
def notEatenWithForks : ℕ := 640
def eatenWithForks : ℕ := totalPies - notEatenWithForks

def percentageEatenWithForks := (eatenWithForks : ℚ) / totalPies * 100

theorem piesEatenWithForksPercentage : percentageEatenWithForks = 68 := by
  sorry

end piesEatenWithForksPercentage_l1501_150100


namespace problem1_problem2_l1501_150194

theorem problem1 : |2 - Real.sqrt 3| - 2^0 - Real.sqrt 12 = 1 - 3 * Real.sqrt 3 := 
by 
  sorry
  
theorem problem2 : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1) ^ 2 = 14 + 4 * Real.sqrt 3 := 
by 
  sorry

end problem1_problem2_l1501_150194


namespace Thomas_speed_greater_than_Jeremiah_l1501_150108

-- Define constants
def Thomas_passes_kilometers_per_hour := 5
def Jeremiah_passes_kilometers_per_hour := 6

-- Define speeds (in meters per hour)
def Thomas_speed := Thomas_passes_kilometers_per_hour * 1000
def Jeremiah_speed := Jeremiah_passes_kilometers_per_hour * 1000

-- Define hypothetical additional distances
def Thomas_hypothetical_additional_distance := 600 * 2
def Jeremiah_hypothetical_additional_distance := 50 * 2

-- Define effective distances traveled
def Thomas_effective_distance := Thomas_speed + Thomas_hypothetical_additional_distance
def Jeremiah_effective_distance := Jeremiah_speed + Jeremiah_hypothetical_additional_distance

-- Theorem to prove
theorem Thomas_speed_greater_than_Jeremiah : Thomas_effective_distance > Jeremiah_effective_distance := by
  -- Placeholder for the proof
  sorry

end Thomas_speed_greater_than_Jeremiah_l1501_150108


namespace factorial_square_gt_power_l1501_150145

theorem factorial_square_gt_power {n : ℕ} (h : n > 2) : (n! * n!) > n^n :=
sorry

end factorial_square_gt_power_l1501_150145


namespace solve_for_x_l1501_150132

def delta (x : ℝ) : ℝ := 5 * x + 9
def phi (x : ℝ) : ℝ := 7 * x + 6

theorem solve_for_x (x : ℝ) (h : delta (phi x) = -4) : x = -43 / 35 :=
by
  sorry

end solve_for_x_l1501_150132


namespace work_together_days_l1501_150103

theorem work_together_days (ravi_days prakash_days : ℕ) (hr : ravi_days = 50) (hp : prakash_days = 75) : 
  (ravi_days * prakash_days) / (ravi_days + prakash_days) = 30 :=
sorry

end work_together_days_l1501_150103


namespace max_gold_coins_l1501_150138

theorem max_gold_coins (n : ℤ) (h₁ : ∃ k : ℤ, n = 13 * k + 3) (h₂ : n < 150) : n ≤ 146 :=
by {
  sorry -- Proof not required as per instructions
}

end max_gold_coins_l1501_150138


namespace find_y_l1501_150159

open Real

variable {x y : ℝ}

theorem find_y (h1 : x * y = 25) (h2 : x / y = 36) (hx : 0 < x) (hy : 0 < y) :
  y = 5 / 6 :=
by
  sorry

end find_y_l1501_150159


namespace coefficient_of_ab_is_correct_l1501_150199

noncomputable def a : ℝ := 15 / 7
noncomputable def b : ℝ := 15 / 2
noncomputable def ab : ℝ := 674.9999999999999
noncomputable def coeff_ab := ab / (a * b)

theorem coefficient_of_ab_is_correct :
  coeff_ab = 674.9999999999999 / ((15 * 15) / (7 * 2)) := sorry

end coefficient_of_ab_is_correct_l1501_150199


namespace rectangle_side_difference_l1501_150130

theorem rectangle_side_difference (p d x y : ℝ) (h1 : 2 * x + 2 * y = p)
                                   (h2 : x^2 + y^2 = d^2)
                                   (h3 : x = 2 * y) :
    x - y = p / 6 := 
sorry

end rectangle_side_difference_l1501_150130


namespace odd_function_f_l1501_150139

noncomputable def f : ℝ → ℝ
| x => if hx : x ≥ 0 then x * (1 - x) else x * (1 + x)

theorem odd_function_f {f : ℝ → ℝ}
  (h_odd : ∀ x : ℝ, f (-x) = - f x)
  (h_pos : ∀ x : ℝ, 0 ≤ x → f x = x * (1 - x)) :
  ∀ x : ℝ, x ≤ 0 → f x = x * (1 + x) := by
  intro x hx
  sorry

end odd_function_f_l1501_150139


namespace evaluate_expression_l1501_150101

theorem evaluate_expression : ((3 ^ 2) ^ 3) - ((2 ^ 3) ^ 2) = 665 := by
  sorry

end evaluate_expression_l1501_150101


namespace correct_operation_l1501_150156

theorem correct_operation (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by
  sorry

end correct_operation_l1501_150156


namespace age_difference_l1501_150116

variable (A B C : ℕ)

theorem age_difference (h₁ : C = A - 20) : (A + B) = (B + C) + 20 := 
sorry

end age_difference_l1501_150116


namespace min_value_expression_l1501_150118

theorem min_value_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : x * y * z = 1/2) :
  x^3 + 4 * x * y + 16 * y^3 + 8 * y * z + 3 * z^3 ≥ 18 :=
sorry

end min_value_expression_l1501_150118


namespace triangle_is_right_triangle_l1501_150154

theorem triangle_is_right_triangle
  (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 10 * a - 6 * b - 8 * c + 50 = 0) :
  a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2 :=
sorry

end triangle_is_right_triangle_l1501_150154


namespace dice_probability_l1501_150113

noncomputable def probability_each_number_appears_at_least_once : ℝ :=
  1 - (6 * (5/6)^10 - 15 * (4/6)^10 + 20 * (3/6)^10 - 15 * (2/6)^10 + 6 * (1/6)^10)

theorem dice_probability : probability_each_number_appears_at_least_once = 0.272 :=
by
  sorry

end dice_probability_l1501_150113


namespace work_completion_l1501_150141

theorem work_completion (d : ℝ) :
  (9 * (1 / d) + 8 * (1 / 20) = 1) ↔ (d = 15) :=
by
  sorry

end work_completion_l1501_150141


namespace min_value_expression_l1501_150153

theorem min_value_expression (n : ℕ) (h : 0 < n) : 
  ∃ (m : ℕ), (m = n) ∧ (∀ k > 0, (k = n) -> (n / 3 + 27 / n) = 6) := 
sorry

end min_value_expression_l1501_150153


namespace arcsin_sqrt2_over_2_eq_pi_over_4_l1501_150161

theorem arcsin_sqrt2_over_2_eq_pi_over_4 :
  Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
sorry

end arcsin_sqrt2_over_2_eq_pi_over_4_l1501_150161


namespace f_one_eq_zero_l1501_150164

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Conditions for the problem
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom periodic_function : ∀ x : ℝ, f (x + 2) = f (x)

-- Goal: Prove that f(1) = 0
theorem f_one_eq_zero : f 1 = 0 :=
by
  sorry

end f_one_eq_zero_l1501_150164


namespace value_of_Y_l1501_150151

-- Definitions for the conditions in part a)
def M := 2021 / 3
def N := M / 4
def Y := M + N

-- The theorem stating the question and its correct answer
theorem value_of_Y : Y = 843 := by
  sorry

end value_of_Y_l1501_150151


namespace height_of_spruce_tree_l1501_150122

theorem height_of_spruce_tree (t : ℚ) (h1 : t = 25 / 64) :
  (∃ s : ℚ, s = 3 / (1 - t) ∧ s = 64 / 13) :=
by
  sorry

end height_of_spruce_tree_l1501_150122


namespace problem_1_problem_2_l1501_150182

-- Definitions and conditions for the problems
def A : Set ℝ := { x | abs (x - 2) < 3 }
def B (m : ℝ) : Set ℝ := { x | x^2 - 2 * x - m < 0 }

-- Problem (I)
theorem problem_1 : (A ∩ (Set.univ \ B 3)) = { x | 3 ≤ x ∧ x < 5 } :=
sorry

-- Problem (II)
theorem problem_2 (m : ℝ) : (A ∩ B m = { x | -1 < x ∧ x < 4 }) → m = 8 :=
sorry

end problem_1_problem_2_l1501_150182


namespace charlie_rope_first_post_l1501_150193

theorem charlie_rope_first_post (X : ℕ) (h : X + 20 + 14 + 12 = 70) : X = 24 :=
sorry

end charlie_rope_first_post_l1501_150193


namespace find_a_b_l1501_150148

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 + Complex.I) 
  (h : (z^2 + a*z + b) / (z^2 - z + 1) = 1 - Complex.I) : a = -1 ∧ b = 2 :=
by
  sorry

end find_a_b_l1501_150148


namespace fraction_classification_l1501_150127

theorem fraction_classification (x y : ℤ) :
  (∃ a b : ℤ, a/b = x/(x+1)) ∧ ¬(∃ a b : ℤ, a/b = x/2 + 1) ∧ ¬(∃ a b : ℤ, a/b = x/2) ∧ ¬(∃ a b : ℤ, a/b = xy/3) :=
by sorry

end fraction_classification_l1501_150127


namespace complex_square_identity_l1501_150174

theorem complex_square_identity (i : ℂ) (h_i_squared : i^2 = -1) :
  i * (1 + i)^2 = -2 :=
by
  sorry

end complex_square_identity_l1501_150174


namespace find_y_eq_1_div_5_l1501_150191

theorem find_y_eq_1_div_5 (b : ℝ) (y : ℝ) (h1 : b > 2) (h2 : y > 0) (h3 : (3 * y)^(Real.log 3 / Real.log b) - (5 * y)^(Real.log 5 / Real.log b) = 0) :
  y = 1 / 5 :=
by
  sorry

end find_y_eq_1_div_5_l1501_150191


namespace fabric_cut_l1501_150172

/-- Given a piece of fabric that is 2/3 meter long,
we can cut a piece measuring 1/2 meter
by folding the original piece into four equal parts and removing one part. -/
theorem fabric_cut :
  ∃ (f : ℚ), f = (2/3 : ℚ) → ∃ (half : ℚ), half = (1/2 : ℚ) ∧ half = f * (3/4 : ℚ) :=
by
  sorry

end fabric_cut_l1501_150172


namespace tickets_distribution_l1501_150190

theorem tickets_distribution (people tickets : ℕ) (h_people : people = 9) (h_tickets : tickets = 24)
  (h_each_gets_at_least_one : ∀ (i : ℕ), i < people → (1 : ℕ) ≤ 1) :
  ∃ (count : ℕ), count ≥ 4 ∧ ∃ (f : ℕ → ℕ), (∀ i, i < people → 1 ≤ f i ∧ f i ≤ tickets) ∧ (∀ i < people, ∃ j < people, f i = f j) :=
  sorry

end tickets_distribution_l1501_150190


namespace ellipse_equation_l1501_150196

theorem ellipse_equation (a b c : ℝ) :
  (2 * a = 10) ∧ (c / a = 4 / 5) →
  ((x:ℝ)^2 / 25 + (y:ℝ)^2 / 9 = 1) ∨ ((x:ℝ)^2 / 9 + (y:ℝ)^2 / 25 = 1) :=
by
  sorry

end ellipse_equation_l1501_150196


namespace teams_same_matches_l1501_150165

theorem teams_same_matches (n : ℕ) (h : n = 30) : ∃ (i j : ℕ), i ≠ j ∧ ∀ (m : ℕ), m ≤ n - 1 → (some_number : ℕ) = (some_number : ℕ) :=
by {
  sorry
}

end teams_same_matches_l1501_150165


namespace kindergarten_children_count_l1501_150186

theorem kindergarten_children_count (D B C : ℕ) (hD : D = 18) (hB : B = 6) (hC : C + B = 12) : D + C + B = 30 :=
by
  sorry

end kindergarten_children_count_l1501_150186


namespace initial_lychees_count_l1501_150109

theorem initial_lychees_count (L : ℕ) (h1 : L / 2 = 2 * 100 * 5 / 5 * 5) : L = 500 :=
by sorry

end initial_lychees_count_l1501_150109


namespace price_of_cheaper_book_l1501_150131

theorem price_of_cheaper_book
    (total_cost : ℕ)
    (sets : ℕ)
    (price_more_expensive_book_increase : ℕ)
    (h1 : total_cost = 21000)
    (h2 : sets = 3)
    (h3 : price_more_expensive_book_increase = 300) :
  ∃ x : ℕ, 3 * ((x + (x + price_more_expensive_book_increase))) = total_cost ∧ x = 3350 :=
by
  sorry

end price_of_cheaper_book_l1501_150131


namespace rhombus_shorter_diagonal_l1501_150168

theorem rhombus_shorter_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d2 = 20) (h2 : area = 120) (h3 : area = (d1 * d2) / 2) : d1 = 12 :=
by 
  sorry

end rhombus_shorter_diagonal_l1501_150168


namespace initial_ratio_of_milk_to_water_l1501_150140

variable (M W : ℕ)
noncomputable def M_initial := 45 - W
noncomputable def W_new := W + 9

theorem initial_ratio_of_milk_to_water :
  M_initial = 36 ∧ W = 9 →
  M_initial / (W + 9) = 2 ↔ 4 = M_initial / W := 
sorry

end initial_ratio_of_milk_to_water_l1501_150140


namespace maximize_S_n_l1501_150149

variable (a_1 d : ℝ)
noncomputable def S (n : ℕ) := n * a_1 + (n * (n - 1) / 2) * d

theorem maximize_S_n {n : ℕ} (h1 : S 17 > 0) (h2 : S 18 < 0) : n = 9 := sorry

end maximize_S_n_l1501_150149


namespace y_comparison_l1501_150157

theorem y_comparison :
  let y1 := (-1)^2 - 2*(-1) + 3
  let y2 := (-2)^2 - 2*(-2) + 3
  y2 > y1 := by
  sorry

end y_comparison_l1501_150157


namespace min_value_ineq_l1501_150192

theorem min_value_ineq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 3) : 
  (1 / a) + (4 / b) ≥ 3 :=
sorry

end min_value_ineq_l1501_150192


namespace least_five_digit_congruent_6_mod_17_l1501_150128

theorem least_five_digit_congruent_6_mod_17 : ∃ n: ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 6 ∧ ∀ m: ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m :=
sorry

end least_five_digit_congruent_6_mod_17_l1501_150128


namespace total_amount_correct_l1501_150171

def num_2won_bills : ℕ := 8
def value_2won_bills : ℕ := 2
def num_1won_bills : ℕ := 2
def value_1won_bills : ℕ := 1

theorem total_amount_correct :
  (num_2won_bills * value_2won_bills) + (num_1won_bills * value_1won_bills) = 18 :=
by
  sorry

end total_amount_correct_l1501_150171


namespace hyperbola_focus_to_asymptote_distance_l1501_150124

theorem hyperbola_focus_to_asymptote_distance :
  ∀ (x y : ℝ), (x ^ 2 - y ^ 2 = 1) →
  ∃ c : ℝ, (c = 1) :=
by
  sorry

end hyperbola_focus_to_asymptote_distance_l1501_150124


namespace arithmetic_sequence_z_l1501_150189

-- Define the arithmetic sequence and value of z
theorem arithmetic_sequence_z (z : ℤ) (arith_seq : 9 + 27 = 2 * z) : z = 18 := 
by 
  sorry

end arithmetic_sequence_z_l1501_150189


namespace total_sum_is_750_l1501_150195

-- Define the individual numbers
def joyce_number : ℕ := 30

def xavier_number (joyce : ℕ) : ℕ :=
  4 * joyce

def coraline_number (xavier : ℕ) : ℕ :=
  xavier + 50

def jayden_number (coraline : ℕ) : ℕ :=
  coraline - 40

def mickey_number (jayden : ℕ) : ℕ :=
  jayden + 20

def yvonne_number (xavier joyce : ℕ) : ℕ :=
  xavier + joyce

-- Prove the total sum is 750
theorem total_sum_is_750 :
  joyce_number + xavier_number joyce_number + coraline_number (xavier_number joyce_number) +
  jayden_number (coraline_number (xavier_number joyce_number)) +
  mickey_number (jayden_number (coraline_number (xavier_number joyce_number))) +
  yvonne_number (xavier_number joyce_number) joyce_number = 750 :=
by {
  -- Proof omitted for brevity
  sorry
}

end total_sum_is_750_l1501_150195


namespace gcd_1755_1242_l1501_150163

theorem gcd_1755_1242 : Nat.gcd 1755 1242 = 27 := 
by
  sorry

end gcd_1755_1242_l1501_150163


namespace example_proof_l1501_150144

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom axiom1 (x y : ℝ) : f (x - y) = f x * g y - g x * f y
axiom axiom2 (x : ℝ) : f x ≠ 0
axiom axiom3 : f 1 = f 2

theorem example_proof : g (-1) + g 1 = 1 := by
  sorry

end example_proof_l1501_150144


namespace ticket_sales_l1501_150150

-- Definitions of the conditions
theorem ticket_sales (adult_cost child_cost total_people child_count : ℕ)
  (h1 : adult_cost = 8)
  (h2 : child_cost = 1)
  (h3 : total_people = 22)
  (h4 : child_count = 18) :
  (child_count * child_cost + (total_people - child_count) * adult_cost = 50) := by
  sorry

end ticket_sales_l1501_150150
