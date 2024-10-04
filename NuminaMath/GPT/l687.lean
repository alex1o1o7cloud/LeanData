import Mathlib

namespace minimum_distance_between_points_l687_687852

noncomputable def pointA : ℝ × ℝ := (-1, 0)

def theta_half : ℝ := 2 * θ

def sin_theta_half_eq : Prop := sin θ/2 = -4/5
def cos_theta_half_eq : Prop := cos θ/2 = 3/5

def pointB_on_terminal_side (B : ℝ × ℝ) : Prop :=
  ∃ (x : ℝ), B.2 = (24/7) * B.1

def minimum_distance_condition (B : ℝ × ℝ) : ℝ :=
  dist pointA B

theorem minimum_distance_between_points 
  (h₁ : sin_theta_half_eq) (h₂ : cos_theta_half_eq) :
  ∃ B : ℝ × ℝ, pointB_on_terminal_side B ∧ minimum_distance_condition B = 24/25 :=
sorry

end minimum_distance_between_points_l687_687852


namespace non_real_root_exists_l687_687447

noncomputable def P (n : ℕ) (coeffs : Fin n → ℤ) : Polynomial ℝ := sorry

theorem non_real_root_exists 
  (n : ℕ)
  (h1 : n ≥ 4)
  (coeffs : Fin (n + 1) → Int)
  (h2 : ∀ i, coeffs i ∈ {-1, 0, 1})
  (h3 : (P n coeffs).eval 0 ≠ 0) :
  ∃ z : ℂ, ¬ (z.im = 0) ∧ z ∈ (P n coeffs).roots :=
begin
  sorry
end

end non_real_root_exists_l687_687447


namespace sin_bound_l687_687984

theorem sin_bound (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < sin x ∧ sin x < x := 
sorry

end sin_bound_l687_687984


namespace find_ordered_triple_l687_687063

theorem find_ordered_triple
  (a b c : ℝ)
  (h1 : a > 2)
  (h2 : b > 2)
  (h3 : c > 2)
  (h4 : (a + 3) ^ 2 / (b + c - 3) + (b + 5) ^ 2 / (c + a - 5) + (c + 7) ^ 2 / (a + b - 7) = 48) :
  (a, b, c) = (7, 5, 3) :=
by {
  sorry
}

end find_ordered_triple_l687_687063


namespace morning_rowers_l687_687480

theorem morning_rowers (total_rowers afternoon_rowers : ℕ) (h1 : total_rowers = 32) (h2 : afternoon_rowers = 17) :
  total_rowers - afternoon_rowers = 15 :=
by
  rw [h1, h2]
  exact Nat.sub_self h2.symm ▸ rfl
  
#eval morning_rowers 32 17 rfl rfl

end morning_rowers_l687_687480


namespace shirt_price_l687_687518

theorem shirt_price (T S : ℝ) (h1 : T + S = 80.34) (h2 : T = S - 7.43) : T = 36.455 :=
by 
sorry

end shirt_price_l687_687518


namespace product_of_solutions_l687_687934

theorem product_of_solutions 
  (h : ∀ x : ℝ, (2 * x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) → x = 0 ∨ x = -5) : 
  ∏ x in (Finset.filter (λ x, (2 * x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) (Finset.range 2)), x = 0 := 
by
  sorry

end product_of_solutions_l687_687934


namespace correct_multiplication_result_l687_687969

theorem correct_multiplication_result (x : ℕ) (h : 9 * x = 153) : 6 * x = 102 :=
by {
  -- We would normally provide a detailed proof here, but as per instruction, we add sorry.
  sorry
}

end correct_multiplication_result_l687_687969


namespace min_k_for_cube_root_difference_l687_687527

theorem min_k_for_cube_root_difference : 
  ∀ (s : Finset ℕ), s.card = 13 → (∀ {a b : ℕ}, a ∈ s → b ∈ s → a ≠ b → |Real.cbrt a - Real.cbrt b| < 1) :=
by
  sorry

end min_k_for_cube_root_difference_l687_687527


namespace cyclic_A_F_D_E_l687_687443

open EuclideanGeometry

theorem cyclic_A_F_D_E
  (ABC : Triangle)
  (omega1 : Circle)
  (I : Point)
  (S : Point)
  (omega2 : Circle)
  (D E F A B C : Point)
  (h1ABC : Circumcircle omega1 ABC)
  (h2I : Incenter I ABC)
  (h3S : Circumcenter S (Triangle B C I))
  (h4omega2 : Circumcircle omega2 (Triangle B C I))
  (h5D : SecondIntersection omega2 (LineThrough B S) D)
  (h6E : SecondIntersection omega2 (LineThrough C S) E)
  (h7F : OnArc F (ArcNotContaining omega1 B C S))
  (h8Angle : Angle B S A = Angle F S C) :
  Cyclic {A, F, D, E} :=
sorry

end cyclic_A_F_D_E_l687_687443


namespace explicit_formulas_l687_687658

noncomputable def generateRandomVariables (r_i r_i' : ℝ) (h_ri : r_i ∈ Icc 0 1) (h_ri' : r_i' ∈ Icc 0 1) : (ℝ × ℝ) :=
  let x_i := r_i^(1/3)
  let y_i := x_i * sqrt r_i'
  (x_i, y_i)

theorem explicit_formulas (f : ℝ × ℝ → ℝ)
  (h_f : ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ x → f (x, y) = 6 * y) :
  ∃ (r_i r_i' : ℝ), r_i ∈ Icc 0 1 ∧ r_i' ∈ Icc 0 1 ∧
  generateRandomVariables r_i r_i' (Icc_0_least r_i) (Icc_0_least r_i') = (r_i^(1/3), (r_i^(1/3)) * sqrt r_i') :=
by
  sorry

end explicit_formulas_l687_687658


namespace general_formula_an_sum_Tn_l687_687794

-- Problem 1: General formula for the n-th term of the sequence {a_n}
theorem general_formula_an (a_n S_n : ℕ → ℝ) (h_seq_arith : ∀ n, 2 * a_n n = S_n n + 1 / 2)
  (h_S1 : S_n 1 = a_n 1) :
  a_n = λ n, 2^(n-2) := 
by 
  sorry

-- Problem 2: Sum of the first n terms of the sequence {1 / b_n}
theorem sum_Tn (a_n b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (h_an : ∀ n, a_n n = 2^(n-2))
  (h_bn : ∀ n, b_n n = (log 2 (a_n (2*n + 1))) * (log 2 (a_n (2*n + 3))))
  (h_Tn_def : T_n = λ n, ∑ i in range n, (1 / b_n i)) :
  T_n n = n / (2*n + 1) := 
by 
  sorry

end general_formula_an_sum_Tn_l687_687794


namespace max_value_of_f_l687_687799

def f (x : ℕ) : ℕ := 2 * x - 3

theorem max_value_of_f : ∀ x ∈ ({1, 2, 3} : set ℕ), f x ≤ 3 ∧ (∃ y ∈ ({1, 2, 3} : set ℕ), f y = 3) :=
by
  sorry

end max_value_of_f_l687_687799


namespace binomial_expected_value_l687_687353

theorem binomial_expected_value (X : Type) (n : ℕ) (p : ℝ) 
  (hX : X ∼ binomial n p) (h_n : n = 6) (h_p : p = 1 / 4) : 
  (EX : ℝ) = n * p := by
  sorry

end binomial_expected_value_l687_687353


namespace general_formula_a_n_sum_T_n_l687_687329

-- Definitions of the sequences
def a (n : ℕ) : ℕ := 4 + (n - 1) * 1
def S (n : ℕ) : ℕ := n / 2 * (2 * 4 + (n - 1) * 1)
def b (n : ℕ) : ℕ := 2 ^ (a n - 3)
def T (n : ℕ) : ℕ := 2 * (2 ^ n - 1)

-- Given conditions
axiom a4_eq_7 : a 4 = 7
axiom S2_eq_9 : S 2 = 9

-- Theorems to prove
theorem general_formula_a_n : ∀ n, a n = n + 3 := 
by sorry

theorem sum_T_n : ∀ n, T n = 2 ^ (n + 1) - 2 := 
by sorry

end general_formula_a_n_sum_T_n_l687_687329


namespace supplement_complement_l687_687965

theorem supplement_complement (angle1 angle2 : ℝ) 
  (h_complementary : angle1 + angle2 = 90) : 
   180 - angle1 = 90 + angle2 := by
  sorry

end supplement_complement_l687_687965


namespace range_of_a_l687_687719

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687719


namespace line_product_l687_687636

theorem line_product (b m : ℝ) (h1: b = -1) (h2: m = 2) : m * b = -2 :=
by
  rw [h1, h2]
  norm_num


end line_product_l687_687636


namespace collinearity_of_P_Q_H_l687_687871

theorem collinearity_of_P_Q_H
(triangle ABC : Type)
(H : point)
(M : point)
(N : point)
(P Q : point)
(on_segment_AB : M ∈ segment AB)
(on_segment_AC : N ∈ segment AC)
(is_orthocenter : is_orthocenter H ABC)
(circles_intersect_PQ : P ∈ circle (diameter BN) ∧ Q ∈ circle (diameter CM)):
collinear P Q H := 
sorry

end collinearity_of_P_Q_H_l687_687871


namespace exists_composite_power_sum_l687_687093

def is_composite (n : ℕ) : Prop := ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q 

theorem exists_composite_power_sum (a : ℕ) (h1 : 1 < a) (h2 : a ≤ 100) : 
  ∃ n, (n > 0) ∧ (n ≤ 6) ∧ is_composite (a ^ (2 ^ n) + 1) :=
by
  sorry

end exists_composite_power_sum_l687_687093


namespace smallest_k_l687_687526

theorem smallest_k (s : Finset ℕ) (h1 : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 2016) (n : ℕ) (h2 : s.card = n) :
  ∃ (a b ∈ s), a ≠ b ∧ abs (Nat.cbrt a - Nat.cbrt b) < 1 ↔ n ≥ 13 := by
  sorry

end smallest_k_l687_687526


namespace weekly_deficit_is_2800_l687_687863

def daily_intake (day : String) : ℕ :=
  if day = "Monday" then 2500 else 
  if day = "Tuesday" then 2600 else 
  if day = "Wednesday" then 2400 else 
  if day = "Thursday" then 2700 else 
  if day = "Friday" then 2300 else 
  if day = "Saturday" then 3500 else 
  if day = "Sunday" then 2400 else 0

def daily_expenditure (day : String) : ℕ :=
  if day = "Monday" then 3000 else 
  if day = "Tuesday" then 3200 else 
  if day = "Wednesday" then 2900 else 
  if day = "Thursday" then 3100 else 
  if day = "Friday" then 2800 else 
  if day = "Saturday" then 3000 else 
  if day = "Sunday" then 2700 else 0

def daily_deficit (day : String) : ℤ :=
  daily_expenditure day - daily_intake day

def weekly_caloric_deficit : ℤ :=
  daily_deficit "Monday" +
  daily_deficit "Tuesday" +
  daily_deficit "Wednesday" +
  daily_deficit "Thursday" +
  daily_deficit "Friday" +
  daily_deficit "Saturday" +
  daily_deficit "Sunday"

theorem weekly_deficit_is_2800 : weekly_caloric_deficit = 2800 := by
  sorry

end weekly_deficit_is_2800_l687_687863


namespace cubic_expression_solution_l687_687452

theorem cubic_expression_solution (r s : ℝ) (h₁ : 3 * r^2 - 4 * r - 7 = 0) (h₂ : 3 * s^2 - 4 * s - 7 = 0) :
  (3 * r^3 - 3 * s^3) / (r - s) = 37 / 3 :=
sorry

end cubic_expression_solution_l687_687452


namespace row_sum_lt_518_l687_687586

noncomputable theory

variables (a : ℕ → ℕ → ℕ)
variable (non_neg : ∀ i j, 0 ≤ a i j)
variable (sum_1956 : (∑ i in finset.range 8, ∑ j in finset.range 8, a i j) = 1956)
variable (diag_sum_112 : (∑ i in finset.range 8, a i i + a i (7 - i)) = 112)
variable (symmetry : ∀ i j, a i j = a j i)

theorem row_sum_lt_518 (r : ℕ) (hr : r < 8) : (∑ j in finset.range 8, a r j) < 518 :=
sorry

end row_sum_lt_518_l687_687586


namespace total_profit_equals_1_35_million_l687_687219

def y_production (x : ℝ) : ℝ := 200 * x - 100
def y_demand (x : ℝ) : ℝ := -20 * x^2 + 100 * x + 900
def x_as_function_of_t (t : ℝ) : ℝ := t + 1
def z_as_function_of_t (t : ℝ) : ℝ := (1/8) * t^2 + 3/2

theorem total_profit_equals_1_35_million (t x z y_production y_demand : ℝ) (h1 : x = t + 1)
  (h2 : z = (1/8) * t^2 + 3/2) (h3 : y_production = 200 * x - 100) (h4 : y_demand = -20 * x^2 + 100 * x + 900)
  (h5 : y_production = y_demand):
  900 * (5 - 7/2) * 1000 = 1.35 * 10^6 :=
by 
  sorry

end total_profit_equals_1_35_million_l687_687219


namespace susan_ate_6_candies_l687_687239

-- Definitions for the conditions
def candies_tuesday : ℕ := 3
def candies_thursday : ℕ := 5
def candies_friday : ℕ := 2
def candies_left : ℕ := 4

-- The total number of candies bought during the week
def total_candies_bought : ℕ := candies_tuesday + candies_thursday + candies_friday

-- The number of candies Susan ate during the week
def candies_eaten : ℕ := total_candies_bought - candies_left

-- Theorem statement
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  unfold candies_eaten total_candies_bought candies_tuesday candies_thursday candies_friday candies_left
  sorry

end susan_ate_6_candies_l687_687239


namespace solve_dog_weights_l687_687487

def brown_dog_weight : ℕ := 8

def black_dog_weight : ℕ := brown_dog_weight + Nat.sqrt 64

def white_dog_weight : ℕ := 2.5 * brown_dog_weight

def grey_dog_weight : ℕ := (black_dog_weight * black_dog_weight) / 2

def yellow_dog_weight : ℕ := ((grey_dog_weight / 4) + 2 * (grey_dog_weight / 4))

def pink_dog_weight : ℕ := (3 * white_dog_weight / 4) + 5

def blue_dog_weight : ℕ := yellow_dog_weight + 2 + 3

def purple_dog_weight : ℕ := ((brown_dog_weight + white_dog_weight) / 2) * 1.5

def dog_weights : List ℕ := [brown_dog_weight,
  black_dog_weight,
  white_dog_weight,
  grey_dog_weight,
  yellow_dog_weight,
  pink_dog_weight,
  blue_dog_weight,
  purple_dog_weight]

def median_weight (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· < ·)
  (sorted.get! (l.length / 2 - 1) + sorted.get! (l.length / 2)) / 2

def mode_weight (l : List ℕ) : ℕ :=
  l.foldl (λ (acc, max_freq, freq_map) n =>
    let freq := freq_map.find n |>.getD 0 + 1
    let freq_map := freq_map.insert n freq
    if freq > max_freq then (n, freq, freq_map)
    else acc) (0, 0, RBMap.empty)._1

theorem solve_dog_weights :
  median_weight dog_weights = 20.5 ∧
  mode_weight dog_weights = 20 := by
  sorry

end solve_dog_weights_l687_687487


namespace sin_330_is_neg_half_l687_687645

def sin_given_conditions (θ : ℝ) : Prop :=
  sin 330 = sin (360 - 30) ∧ 
  sin (-30) = -sin 30 ∧ 
  sin 30 = 1 / 2

theorem sin_330_is_neg_half (θ : ℝ) (h : sin_given_conditions θ) : sin 330 = -1 / 2 := 
by
  sorry

end sin_330_is_neg_half_l687_687645


namespace find_five_digit_number_l687_687949

theorem find_five_digit_number (x : ℕ) (hx : 10000 ≤ x ∧ x < 100000)
  (h : 10 * x + 1 = 3 * (100000 + x) ∨ 3 * (10 * x + 1) = 100000 + x) :
  x = 42857 :=
sorry

end find_five_digit_number_l687_687949


namespace residues_mod_p_l687_687440

theorem residues_mod_p (p : ℕ) (k : ℕ) (x y : ℕ) (hx : x < p) (hy : y < p) (h_prime : Nat.Prime p) (h_form : p = 4 * k + 3) (hx_coprime : Nat.gcd x p = 1) (hy_coprime : Nat.gcd y p = 1) :
  ∃ n : ℕ, n = (p - 1) / 2 ∧ (Set.image (λ z, (z^2 + y^2)^2 % p) {z | Nat.gcd z p = 1}).card = n :=
sorry

end residues_mod_p_l687_687440


namespace problem_cos_angle_YXW_l687_687856

noncomputable def cos_angle_YXW (XY XZ YZ : ℝ) (W : Point) [IncidenceGeometry] [Triangle X Y Z]
  (h1 : XY = 4)
  (h2 : XZ = 8)
  (h3 : YZ = 10)
  (hW : OnLine W (Line Y Z))
  (hXW_bisects : AngleBisector (Angle X Y Z) W) : ℝ :=
  \frac{\sqrt{88}}{16}

theorem problem_cos_angle_YXW :
  cos_angle_YXW 4 8 10 W = \frac{\sqrt{88}}{16} :=
  sorry

end problem_cos_angle_YXW_l687_687856


namespace range_of_a_l687_687783

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687783


namespace range_of_a_l687_687727

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687727


namespace number_of_students_in_club_l687_687417

variable (y : ℕ) -- Number of girls

def total_stickers_given (y : ℕ) : ℕ := y * y + (y + 3) * (y + 3)

theorem number_of_students_in_club :
  (total_stickers_given y = 640) → (2 * y + 3 = 35) := 
by
  intro h1
  sorry

end number_of_students_in_club_l687_687417


namespace distance_from_point_to_line_eq_17_over_7_l687_687288

section

variables (p : ℝ × ℝ × ℝ) (a b : ℝ × ℝ × ℝ)

def distance_from_point_to_line (p : ℝ × ℝ × ℝ) (a b : ℝ × ℝ × ℝ) : ℝ :=
  let v := (b.1 - a.1, b.2 - a.2, b.3 - a.3) in
  let t := ((p.1 - a.1) * v.1 + (p.2 - a.2) * v.2 + (p.3 - a.3) * v.3) /
           (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) in
  let closest_point := (a.1 + t * v.1, a.2 + t * v.2, a.3 + t * v.3) in
  let delta := (p.1 - closest_point.1, p.2 - closest_point.2, p.3 - closest_point.3) in
  real.sqrt (delta.1 * delta.1 + delta.2 * delta.2 + delta.3 * delta.3)

theorem distance_from_point_to_line_eq_17_over_7 :
  distance_from_point_to_line (2, -2, 3) (1, 1, -2) (3, -3, -1) = 17 / 7 := sorry

end

end distance_from_point_to_line_eq_17_over_7_l687_687288


namespace absolute_value_inequality_solution_l687_687477

theorem absolute_value_inequality_solution (x : ℝ) :
  |x - 2| + |x - 4| ≤ 3 ↔ (3 / 2 ≤ x ∧ x < 4) :=
by
  sorry

end absolute_value_inequality_solution_l687_687477


namespace min_k_for_cube_root_difference_l687_687530

theorem min_k_for_cube_root_difference (cards : Finset ℕ) (h : cards = Finset.range 2017) (selected : Finset ℕ) (h_selected : selected.card = 13) : 
  ∃ (a b : ℕ), a ∈ selected ∧ b ∈ selected ∧ a ≠ b ∧ (|real.cbrt a - real.cbrt b| < 1) :=
by
  sorry

end min_k_for_cube_root_difference_l687_687530


namespace thursday_production_production_difference_total_production_total_wage_l687_687589

open scoped BigOperators

-- Defining the daily deviations
def daily_deviations : List ℤ := [5, -2, -4, 13, -6, 6, -3]

-- Average daily production
def average_daily_production : ℤ := 100

-- Thursday's production
theorem thursday_production : 
  (average_daily_production + daily_deviations.nthLe 3 (by simp)) = 113 := sorry

-- Difference between highest and lowest production
theorem production_difference :
  ((average_daily_production + daily_deviations.max') - 
  (average_daily_production + daily_deviations.min')) = 19 :=
sorry

-- Total production for the week
theorem total_production :
  (700 + (daily_deviations.sum)) = 709 := sorry

-- Total wage calculation
def piece_rate_per_toy : ℕ := 20
def bonus_per_extra_toy : ℕ := 5
def deduction_per_less_toy : ℕ := 4
def planned_production : ℕ := 700

theorem total_wage :
  let actual_production := planned_production + daily_deviations.sum.toNat
  let base_wage := planned_production * piece_rate_per_toy
  let over_production_bonus := 
    if actual_production > planned_production then
      (actual_production - planned_production) * (piece_rate_per_toy + bonus_per_extra_toy)
    else 0
  let under_production_deduction :=
    if actual_production < planned_production then
      (planned_production - actual_production) * deduction_per_less_toy
    else 0
  in (base_wage + over_production_bonus - under_production_deduction) = 14225 :=
sorry

end thursday_production_production_difference_total_production_total_wage_l687_687589


namespace cone_volume_l687_687326

theorem cone_volume (central_angle : ℝ) (sector_area : ℝ) (h1 : central_angle = 120) (h2 : sector_area = 3 * Real.pi) :
  ∃ V : ℝ, V = (2 * Real.sqrt 2 * Real.pi) / 3 :=
by
  -- We acknowledge the input condition where the angle is 120° and sector area is 3π
  -- The problem requires proving the volume of the cone
  sorry

end cone_volume_l687_687326


namespace max_additional_plates_l687_687483

def initial_plates (A B C : ℕ) : ℕ := A * B * C

def plates_scenario_1 (A B C : ℕ) (X : ℕ) : ℕ := (A + X) * B * C
def plates_scenario_2 (A B C : ℕ) (Y : ℕ) : ℕ := A * B * (C + Y)
def plates_scenario_3 (A B C : ℕ) (X Y : ℕ) : ℕ := (A + X) * B * (C + Y)
def plates_scenario_4 (A B C D X Y : ℕ) : ℕ := (A + X) * (B + Y) * C

theorem max_additional_plates :
  let A := 3 in
  let B := 2 in
  let C := 4 in
  let initial := initial_plates A B C in
  let max_plates := max (max (plates_scenario_1 A B C 2)
                           (plates_scenario_2 A B C 2))
                        (max (plates_scenario_3 A B C 1 1)
                             (plates_scenario_4 A B C 1 1)) in
  max_plates - initial = 24 :=
by
  sorry

end max_additional_plates_l687_687483


namespace Mark_marbles_correct_l687_687633

def Connie_marbles : ℕ := 323
def Juan_marbles : ℕ := Connie_marbles + 175
def Mark_marbles : ℕ := 3 * Juan_marbles

theorem Mark_marbles_correct : Mark_marbles = 1494 := 
by
  sorry

end Mark_marbles_correct_l687_687633


namespace no_nat_number_exists_l687_687665

def lcm_upto (n : ℕ) : ℕ :=
  List.lcm (List.range (n+1))

theorem no_nat_number_exists (m : ℕ) : ¬ (lcm_upto (m + 1) = 4 * lcm_upto m) :=
  sorry

end no_nat_number_exists_l687_687665


namespace range_of_a_l687_687693

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687693


namespace days_worked_together_l687_687972

theorem days_worked_together (W : ℝ) (h1 : ∀ (a b : ℝ), (a + b) * 40 = W) 
                             (h2 : ∀ a, a * 16 = W) 
                             (x : ℝ) 
                             (h3 : (x * (W / 40) + 12 * (W / 16)) = W) : 
                             x = 10 := 
by
  sorry

end days_worked_together_l687_687972


namespace arithmetic_sequence_sum_l687_687485

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic property of the sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (h1 : is_arithmetic_sequence a d)
  (h2 : a 2 + a 4 + a 7 + a 11 = 44) :
  a 3 + a 5 + a 10 = 33 := 
sorry

end arithmetic_sequence_sum_l687_687485


namespace range_of_x_f_greater_than_4_l687_687886

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else x^2

theorem range_of_x_f_greater_than_4 :
  { x : ℝ | f x > 4 } = { x : ℝ | x < -2 ∨ x > 2 } :=
by
  sorry

end range_of_x_f_greater_than_4_l687_687886


namespace proof_problem_l687_687681

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * sin (2 * x + π / 6)
noncomputable def g (x : ℝ) : ℝ := sqrt 2 * cos (2 * x)

theorem proof_problem (hx : ∀ x, f (x - π / 6) = g x) :
  (∀ x, f x + g x = 0 → x = -2*π/3 ∨ x = 4*π/3) ∧
  (∀ x ∈ set.Icc (2 * π / 3) (5 * π / 6), monotone_on f (set.Icc (2 * π / 3) (5 * π / 6)) ∧
   monotone_on g (set.Icc (2 * π / 3) (5 * π / 6))) ∧
  (∀ x, f (x - π / 6) = g x) :=
begin
  sorry
end

end proof_problem_l687_687681


namespace probability_eighth_roll_is_last_l687_687015

def roll_probability : ℝ := 
  let p := (5 / 6) ^ 6 * (1 / 6)
  Float.round (p * 1000) / 1000

theorem probability_eighth_roll_is_last 
  : roll_probability = 0.027 := sorry

end probability_eighth_roll_is_last_l687_687015


namespace factor_expression_l687_687268

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l687_687268


namespace total_students_in_class_l687_687142

def number_of_girls := 9
def number_of_boys := 16
def total_students := number_of_girls + number_of_boys

theorem total_students_in_class : total_students = 25 :=
by
  -- The proof will go here
  sorry

end total_students_in_class_l687_687142


namespace general_term_l687_687354

-- Assume a sequence of real numbers representing the arithmetic sequence
variable (a : ℕ → ℝ)

-- Definitions based on the provided conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) : Prop :=
  a 2 + a 7 = 12

def condition2 (a : ℕ → ℝ) : Prop :=
  a 4 * a 5 = 35

-- The main statement that needs proof
theorem general_term (a : ℕ → ℝ) :
  arithmetic_sequence a → condition1 a → condition2 a →
  ∃ f : ℕ → ℝ, (∀ n, a n = f n) ∧ (f = (λ n, 2 * n - 3) ∨ f = (λ n, 15 - 2 * n)) :=
by
  sorry

end general_term_l687_687354


namespace triangle_area_lt_1_l687_687213

-- Triangle definitions and area calculations
structure Triangle :=
  (A B C : Point)
  (bisector_A A1 : Line)
  (bisector_B B1 : Line)
  (bisector_C C1 : Line)
  (height_BP : Line)

variables {ABC : Triangle}

-- Conditions provided in the problem
def conditions (ABC : Triangle) :=
  let A1_length := Segment_Length ABC.A ABC.A1 in
  let B1_length := Segment_Length ABC.B ABC.B1 in
  let C1_length := Segment_Length ABC.C ABC.C1 in
  A1_length < 1 ∧ B1_length < 1 ∧ C1_length < 1

-- Final statement to be proven
theorem triangle_area_lt_1 (ABC : Triangle) (h : conditions ABC) : 
  let S := Triangle_Area ABC in
  S < 1 :=
sorry

end triangle_area_lt_1_l687_687213


namespace bahs_from_yahs_l687_687825

theorem bahs_from_yahs (b r y : ℝ) 
  (h1 : 18 * b = 30 * r) 
  (h2 : 10 * r = 25 * y) : 
  1250 * y = 300 * b := 
by
  sorry

end bahs_from_yahs_l687_687825


namespace rupert_jumps_more_than_ronald_l687_687099

theorem rupert_jumps_more_than_ronald :
  ∃ R : ℕ, R > 157 ∧ R + 157 = 400 ∧ (R - 157) = 86 := 
by {
  let R := 243,
  use R,
  split,
  { exact nat.lt_of_add_lt_add_right (by norm_num [R]) },
  split,
  { norm_num [R] },
  { norm_num [R] }
}

end rupert_jumps_more_than_ronald_l687_687099


namespace number_of_subsets_of_B_l687_687000

theorem number_of_subsets_of_B :
  let A := {1, 2, 3}
  let B := {(x, y) | x y : ℕ // x ∈ A ∧ y ∈ A ∧ x + y ∈ A}
  ∃ (n : ℕ), (∀ (b : B), b ∈ B) ∧ (2 ^ 3 = 8) :=
by
  let A := {1, 2, 3}
  let B := {(x, y) | x y : ℕ // x ∈ A ∧ y ∈ A ∧ x + y ∈ A}
  existsi 3
  split
  · intros b
    exact b.2
  · simp only [pow_succ, pow_zero, mul_one, nat.pow_succ] at *
    exact 2 ^ 3 = 8
  sorry

end number_of_subsets_of_B_l687_687000


namespace range_of_a_l687_687699

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687699


namespace vector_addition_l687_687671

theorem vector_addition :
  let a := (2, 1)
  let b := (0, -1)
  2 * b + 3 * a = (6, 1) := by
  sorry

end vector_addition_l687_687671


namespace trapezoid_greatest_int_l687_687608

noncomputable def trapezoid_area_sum (a b c d : ℚ) (n1 n2 : ℕ) (r1 r2 r3 : ℚ) : ℚ :=
  r1 + r2 + r3 + n1 + n2 

theorem trapezoid_greatest_int (r1 r2 r3 : ℚ) (n1 n2 : ℕ) 
  (h1 : r1 = 35/2) 
  (h2 : r2 = 32/3)
  (h3 : r3 = 27) 
  (h4 : n1 = 3)
  (h5 : n2 = 5) :
  (⌊r1 + r2 + r3 + n1 + n2⌋ = 63) :=
by
  have sum := trapezoid_area_sum r1 r2 r3 n1 n2
  have : ∀ x : ℚ, ∃ k : ℤ, (k : ℚ) ≤ x ∧ x < (k + 1 : ℚ),
    from floor_spec
  sorry

end trapezoid_greatest_int_l687_687608


namespace smallest_k_l687_687525

theorem smallest_k (s : Finset ℕ) (h1 : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 2016) (n : ℕ) (h2 : s.card = n) :
  ∃ (a b ∈ s), a ≠ b ∧ abs (Nat.cbrt a - Nat.cbrt b) < 1 ↔ n ≥ 13 := by
  sorry

end smallest_k_l687_687525


namespace smallest_positive_period_axis_of_symmetry_monotonic_intervals_max_min_values_l687_687358

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

theorem smallest_positive_period : ∀ x : ℝ, f (x + Real.pi) = f x := sorry

theorem axis_of_symmetry (k : ℤ) : ∀ x : ℝ, x = k * Real.pi / 2 + Real.pi / 8 → f x = f (2 * k * Real.pi / 2 + Real.pi / 8 - x) := sorry

theorem monotonic_intervals (k : ℤ) :
  (∀ x, k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 8 → f x ≤ f (x + Real.pi)) ∧
  (∀ x, k * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 8 → f x ≥ f (x + Real.pi)) := sorry

theorem max_min_values :
  let interval := Set.Icc 0 (Real.pi / 2) in
  ∃ x1 x2 ∈ interval, f x1 = 2 ∧ f x2 = -Real.sqrt 2 :=
sorry

end smallest_positive_period_axis_of_symmetry_monotonic_intervals_max_min_values_l687_687358


namespace proposition3_symmetric_proposition3_correct_l687_687924

open Real

def f (x : ℝ) : ℝ := 4 * sin (2 * x + π / 3)

theorem proposition3_symmetric :
  ∃ x₀ : ℝ, (x₀ = -π / 6) ∧ (f (-π / 6) = 0) :=
by
  use -π / 6
  split
  . rfl
  . sorry

theorem proposition3_correct : 
  proposition3_symmetric :=
by
  apply proposition3_symmetric
  sorry

end proposition3_symmetric_proposition3_correct_l687_687924


namespace solve_equation_l687_687107

-- Define the equation and the conditions
def problem_equation (x : ℝ) : Prop :=
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 2)

def valid_solution (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6

-- State the theorem that solutions x = 3 and x = -4 solve the problem under the conditions
theorem solve_equation : ∀ x : ℝ, valid_solution x → (x = 3 ∨ x = -4 ∧ problem_equation x) :=
by
  sorry

end solve_equation_l687_687107


namespace ratio_of_x_to_y_l687_687555

theorem ratio_of_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
by
  sorry

end ratio_of_x_to_y_l687_687555


namespace num_ways_to_place_squares_l687_687460

/-- The number of ways to place 9 identical square pieces of paper on a table 
forming at least a two-layered rectangular shape, with the property that each 
upper layer square has two vertices located at the midpoints of the sides of 
squares in the layer directly beneath, is exactly 25. -/
theorem num_ways_to_place_squares (n : ℕ) (h : n = 9) : 
  ∃ k : ℕ, k = 25 :=
by
  use 25
  rw h
  sorry

end num_ways_to_place_squares_l687_687460


namespace triangle_inequality_l687_687971

theorem triangle_inequality 
  (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
begin
  sorry -- proof goes here
end

end triangle_inequality_l687_687971


namespace minimize_cost_l687_687110

open Real

-- Definitions
def tons_A := 15
def tons_B := 15
def req_X := 16
def req_Y := 14
def cost_A_to_X := 50
def cost_A_to_Y := 30
def cost_B_to_X := 60
def cost_B_to_Y := 45

-- Relationships and constraints
def transport_x := (x: ℝ) (h: 1 ≤ x ∧ x ≤ 15) : Prop :=
  let tons_from_A_to_X := x
  let tons_from_A_to_Y := tons_A - x
  let tons_from_B_to_X := req_X - x
  let tons_from_B_to_Y := x - 1
  let y := cost_A_to_X * tons_from_A_to_X + cost_A_to_Y * tons_from_A_to_Y +
            cost_B_to_X * tons_from_B_to_X + cost_B_to_Y * tons_from_B_to_Y
  y = 5 * x + 1365

-- Lean statement for the proof problem
theorem minimize_cost (x: ℝ) (h: 1 ≤ x ∧ x ≤ 15) : 
  ∃ x_min, (y_min : ℝ) (h_min : x_min = 1) (h_ymin : y_min = 1370),
    ∀ x, (1 ≤ x ∧ x ≤ 15) → 5 * x + 1365 ≥ y_min :=
by
  sorry

end minimize_cost_l687_687110


namespace start_of_setX_l687_687474

-- Definitions based on the given conditions
def setX (x_start : ℤ) : set ℤ := {x | x_start ≤ x ∧ x ≤ 12}
def setY : set ℤ := {y | 0 ≤ y ∧ y ≤ 20}
def common_elements (x_start : ℤ) : set ℤ := setX x_start ∩ setY

-- Theorem to prove that the starting number of set X is 1
theorem start_of_setX (x_start : ℤ) 
  (hX : ∀ x, x ∈ setX x_start → x_start ≤ x ∧ x ≤ 12)
  (hY : ∀ y, y ∈ setY → 0 ≤ y ∧ y ≤ 20)
  (h_common : common_elements x_start = {x | x_start ≤ x ∧ x ≤ 12}) :
  x_start = 1 := 
sorry

end start_of_setX_l687_687474


namespace sum_inequality_l687_687580

theorem sum_inequality
  (n : ℕ) (hn : n ≥ 2)
  (a b : ℕ → ℝ)
  (ha : ∀ i j, i ≤ j → a i ≥ a j)
  (hb : ∀ i j, i ≤ j → b i ≥ b j)
  (ha_nonneg : ∀ i, 0 ≤ a i)
  (hb_nonneg : ∀ i, 0 ≤ b i)
  (hprod : ∏ i in (finset.range n).filter (λi, true), a i = ∏ i in (finset.range n).filter (λi, true), b i)
  (hsum_diff : ∑ i in finset.range n, ∑ j in (finset.range (n - i)), (a i - a j) 
                ≤ ∑ i in finset.range n, ∑ j in (finset.range (n - i)), (b i - b j)) :
  ∑ i in finset.range n, a i ≤ (n - 1) * ∑ i in finset.range n, b i := 
sorry

end sum_inequality_l687_687580


namespace value_of_f_at_pi_over_6_l687_687802

def f (x : ℝ) (w : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (w * x + ϕ)

theorem value_of_f_at_pi_over_6 (w ϕ : ℝ) 
    (h : ∀ x, f (π / 6 + x) w ϕ = f (π / 6 - x) w ϕ) :
    f (π / 6) w ϕ = 2 ∨ f (π / 6) w ϕ = -2 :=
by
  sorry

end value_of_f_at_pi_over_6_l687_687802


namespace students_neither_math_physics_l687_687456

theorem students_neither_math_physics (total_students math_students physics_students both_students : ℕ) 
  (h1 : total_students = 120)
  (h2 : math_students = 80)
  (h3 : physics_students = 50)
  (h4 : both_students = 15) : 
  total_students - (math_students - both_students + physics_students - both_students + both_students) = 5 :=
by
  -- Each of the hypotheses are used exactly as given in the conditions.
  -- We omit the proof as requested.
  sorry

end students_neither_math_physics_l687_687456


namespace isosceles_triangle_angles_l687_687407

noncomputable def angle_opposite (a b c : ℝ) := real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

theorem isosceles_triangle_angles :
  let a := 5 in
  let b := 5 in
  let c := real.sqrt 17 - real.sqrt 5 in
  let θ := angle_opposite a b c in
  let φ := (180 - θ) / 2 in
  θ = real.arccos ((14 + real.sqrt 85) / 25) ∧ φ = (180 - θ) / 2 :=
by
  sorry

end isosceles_triangle_angles_l687_687407


namespace find_quadratic_polynomial_with_real_coefficients_l687_687297

theorem find_quadratic_polynomial_with_real_coefficients 
  (a b: ℝ)
  (h1: a ≠ 0)
  (h2: b ≠ 0)
  (h3: (a + bi) = 3 - 4 * complex.I)
  (h4: ∀ c d : ℝ, c≠0 → d≠0 → (a + bi = c - d * complex.I ∨ a + bi = c + d * complex.I))
  (h5: ∃ k: ℝ, k * -6 = 10) :
  ∃ k: ℝ, k * (x^2 - 6 * x + 25) = -5/3 * x^2 + 10 * x - 125 / 3 :=
sorry

end find_quadratic_polynomial_with_real_coefficients_l687_687297


namespace nina_total_cost_l687_687086

-- Define the cost of the first pair of shoes
def first_pair_cost : ℕ := 22

-- Define the cost of the second pair of shoes
def second_pair_cost : ℕ := first_pair_cost + (first_pair_cost / 2)

-- Define the total cost for both pairs of shoes
def total_cost : ℕ := first_pair_cost + second_pair_cost

-- The formal statement of the problem
theorem nina_total_cost : total_cost = 55 := by
  sorry

end nina_total_cost_l687_687086


namespace hyperbola_focus_larger_x_l687_687502

noncomputable def a : ℝ := 3
noncomputable def b : ℝ := 10
noncomputable def h : ℝ := 4
noncomputable def k : ℝ := 15
noncomputable def c : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_focus_larger_x :
  ∃ (x y : ℝ), 
    (x = h + c) ∧
    (y = k) ∧
    ((x - 4)^2 / (3:ℝ)^2 - (y - 15)^2 / (10:ℝ)^2 = 1) :=
by
  use h + c
  use k
  split
  · rfl
  split
  · rfl
  · sorry

end hyperbola_focus_larger_x_l687_687502


namespace james_ate_slices_l687_687047

variable (NumPizzas : ℕ) (SlicesPerPizza : ℕ) (FractionEaten : ℚ)
variable (TotalSlices : ℕ := NumPizzas * SlicesPerPizza)
variable (JamesSlices : ℚ := FractionEaten * TotalSlices)

theorem james_ate_slices (h1 : NumPizzas = 2) (h2 : SlicesPerPizza = 6) (h3 : FractionEaten = 2 / 3) :
    JamesSlices = 8 := 
by 
  simp [JamesSlices, TotalSlices]
  rw [h1, h2, h3]
  norm_num
  sorry

end james_ate_slices_l687_687047


namespace root_in_interval_l687_687561

def f (x : ℝ) : ℝ := real.log x + 2 * x - 7

theorem root_in_interval : ∃ x ∈ set.Ioo 2 3, f x = 0 := by
  sorry

end root_in_interval_l687_687561


namespace terminating_decimals_count_l687_687307

noncomputable def int_counts_terminating_decimals : ℕ :=
  let n_limit := 500
  let denominator := 2100
  Nat.floor (n_limit / 21)

theorem terminating_decimals_count :
  int_counts_terminating_decimals = 23 :=
by
  /- Proof will be here eventually -/
  sorry

end terminating_decimals_count_l687_687307


namespace a4_value_sum_first_2016_terms_l687_687853

-- Define the sequence recursively with initial conditions
def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧
  a 2 = 10 ∧
  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n)

-- Prove a₄ = -2 given the sequence definition
theorem a4_value (a : ℕ → ℤ) (h : sequence a) : a 4 = -2 :=
by sorry

-- Prove the sum of the first 2016 terms is 0
theorem sum_first_2016_terms (a : ℕ → ℤ) (h : sequence a) : (Finset.range 2016).sum (λ n, a (n + 1)) = 0 :=
by sorry

end a4_value_sum_first_2016_terms_l687_687853


namespace apples_left_l687_687453

def Mike_apples : ℝ := 7.0
def Nancy_apples : ℝ := 3.0
def Keith_ate_apples : ℝ := 6.0

theorem apples_left : Mike_apples + Nancy_apples - Keith_ate_apples = 4.0 := by
  sorry

end apples_left_l687_687453


namespace triangle_CGA_right_at_G_ratio_EG_CF_l687_687882

-- Define the setup conditions of the quadrilateral
variables {A B C D E F G : Type}
variables (AD BC AB CD AC BD : ℝ)
variables (a b : ℝ)
-- Assume the given conditions
variables (h_eq_AD_BC : AD = BC)
variables (h_parallel_AB_CD : AB ∥ CD)
variables (h_AB_gt_CD : AB > CD)
variables (h_midpoint_E : E = midpoint A C)
variables (h_intersection_F : F = intersection AC BD)
variables (h_parallel_line_EG_BD : line_parallel (passing_through E) (BD))
variables (h_intersection_G : G = intersection (line_parallel (passing_through E) (BD)) CD)

-- The first question: Show that triangle \(CGA\) is a right triangle at \(G\)
theorem triangle_CGA_right_at_G :
  ∀ (C G A : Type), isosceles_trapezoid AD BC AB CD AC BD ∧ midpoint AC = E ∧ intersection AC BD = F ∧ parallel_through E (BD) = line E G → right_triangle_at G C A :=
by sorry

-- The second question: Calculate the ratio EG/CF as a function of a and b
theorem ratio_EG_CF (CD b AB a : ℝ) :
  CD = b ∧ AB = a ∧ ∃ G F, EG / CF = (a + b) / (2 * b) :=
by sorry

end triangle_CGA_right_at_G_ratio_EG_CF_l687_687882


namespace m_plus_n_composite_l687_687507

theorem m_plus_n_composite (m n : ℕ) (h : 88 * m = 81 * n) : ¬ prime (m + n) :=
by 
  sorry

end m_plus_n_composite_l687_687507


namespace part1_part2_l687_687990

-- Part (1)
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 1) : (x - x^2 < Real.sin x) ∧ (Real.sin x < x) :=
sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = Real.cos (a * x) - Real.log (1 - x^2)) ∧ 
  (f' 0 = 0) ∧ (∃ x_max : ℝ, (f'' x_max < 0) ∧ (x_max = 0))) : a < -Real.sqrt 2 ∨ Real.sqrt 2 < a :=
sorry

end part1_part2_l687_687990


namespace tiling_remainder_l687_687587

theorem tiling_remainder :
  let N := ∑ (k in {3, 4, 5, 6, 7}), (combinations 6 (k - 1)) * (3^k - 3 * 2^k + 3)
  in N % 1000 = 106 := 
by
  -- Define the combinations function
  def combinations (n k : ℕ) := nat.choose n k
  
  -- Calculate the combinations and colorings
  let div_ways := λ k, combinations 6 (k - 1)
  let colorings := λ k, 3^k - 3 * 2^k + 3
  let ways := ∑ k in {3, 4, 5, 6, 7}, div_ways k * colorings k
  
  -- Sum the number of ways and find the remainder
  have h : (∑ k in {3, 4, 5, 6, 7}, div_ways k * colorings k) % 1000 = 106 := sorry
  exact h

end tiling_remainder_l687_687587


namespace sequence_b_sum_l687_687064

def sequence_b (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | 2 => 3
  | 3 => 5
  | n + 4 =>
    let p := sequence_b (n + 1)
    let q := sequence_b (n + 2) * sequence_b (n + 3)
    if (9 * p * p / 4 - q) < 0 then 0
    else if (9 * p * p / 4 - q) = 0 then 2
    else 4

theorem sequence_b_sum : sequence_b 1 + sequence_b 2 + sequence_b 3 + (4 * 17) = 78 :=
  sorry

end sequence_b_sum_l687_687064


namespace cos_arcsin_l687_687623

theorem cos_arcsin (h : 8^2 + 15^2 = 17^2) : Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
by
  have opp_hyp_pos : 0 ≤ 8 / 17 := by norm_num
  have hyp_pos : 0 < 17 := by norm_num
  have opp_le_hyp : 8 / 17 ≤ 1 := by norm_num
  sorry

end cos_arcsin_l687_687623


namespace find_line_eq_l687_687124

theorem find_line_eq (k b : ℝ) (h1 : -k + b = 0) (h2 : b = 3) : y = 3 * x + 3 :=
by
  have H : k = 3 :=
  sorry
  rewrite [H, h2]
  have line_eq : y = 3 * x + 3 :=
  sorry
  exact line_eq

end find_line_eq_l687_687124


namespace find_triples_l687_687057

def f (a b c : ℤ) : ℤ × ℤ × ℤ := (a + b + c, ab + bc + ca, abc)

theorem find_triples (a b c : ℤ) :
  (f (f a b c).1 (f a b c).2.1 (f a b c).2.2 = (a, b, c)) ↔ 
  ((b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = -1 ∧ c = 1)) :=
by {
  sorry
}

end find_triples_l687_687057


namespace inequality_floor_div_sum_l687_687675

noncomputable def floor_div_sequence_sum (x : ℝ) (n : ℕ) :=
  (∑ k in Finset.range n.succ, (⌊ ( x * k : ℝ) ⌋ / k : ℝ))

theorem inequality_floor_div_sum (x : ℝ) (n : ℕ) (hx : 0 < x) (hn : 0 < n) : 
  (⌊x * n⌋ : ℝ) ≥ floor_div_sequence_sum x n :=
sorry

end inequality_floor_div_sum_l687_687675


namespace circumcircle_property_l687_687446

/-- Let O be the center and r be the radius of the circumcircle of triangle ABC.
    Show that 1/AA' + 1/BB' + 1/CC' = 1/r if the extensions of AO, BO, and CO 
    intersect the circles passing through points {B, O, C}; {C, O, A}; and {A, O, B} 
    at points A', B', and C' respectively. -/
theorem circumcircle_property 
  (O : Point)
  (A B C A' B' C' : Point)
  (r : ℝ)
  (hO : IsCircumcenter O A B C)
  (hR : Circumradius O A B C = r)
  (hA' : Extension AO intersectsCirclesAt A' {B, O, C})
  (hB' : Extension BO intersectsCirclesAt B' {C, O, A})
  (hC' : Extension CO intersectsCirclesAt C' {A, O, B}) :
  (1 / dist A A') + (1 / dist B B') + (1 / dist C C') = 1 / r := 
sorry

end circumcircle_property_l687_687446


namespace factor_expression_l687_687269

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l687_687269


namespace find_c_and_general_formula_l687_687854

noncomputable def seq (a : ℕ → ℕ) (c : ℕ) := ∀ n : ℕ, a (n + 1) = a n + c * 2^n

theorem find_c_and_general_formula : 
  ∀ (c : ℕ) (a : ℕ → ℕ),
    (a 1 = 2) →
    (seq a c) →
    ((a 3) = (a 1) * ((a 2) / (a 1))^2) →
    ((a 2) = (a 1) * (a 2) / (a 1)) →
    c = 1 ∧ (∀ n, a n = 2^n) := 
by
  sorry

end find_c_and_general_formula_l687_687854


namespace area_of_rectangle_l687_687406

-- Define the points E, F, and H
noncomputable def E : ℝ × ℝ := (4, -20)
noncomputable def F : ℝ × ℝ := (1004, 200)
noncomputable def H (y : ℝ) : ℝ × ℝ := (6, y)

-- Area calculation
theorem area_of_rectangle (y : ℝ) (h : y = -320 / 11) : 
   (sqrt ((1004 - 4)^2 + (200 - (-20))^2) * sqrt ((6 - 4)^2 + (y - (-20))^2)) = 205824 / 11 :=
by 
suffices : sqrt ((1004 - 4)^2 + (200 - (-20))^2) = 1024 / (y + 20), sorry 
suffices : sqrt ((6 - 4)^2 + (y - (-20))^2) = 1024 / (100 - y), sorry 
suffices : (sqrt ((1004 - 4)^2 + (200 - (-20))^2)) = sqrt 1048400, sorry 
suffices : sqrt ((6 - 4)^2 + (y - (-20))^2) = sqrt ( 40484 / 11), sorry 
end

end area_of_rectangle_l687_687406


namespace probability_of_triangle_or_circle_l687_687457

-- Definitions (conditions)
def total_figures : ℕ := 12
def triangles : ℕ := 4
def circles : ℕ := 3
def squares : ℕ := 5
def figures : ℕ := triangles + circles + squares

-- Probability calculation
def probability_triangle_circle := (triangles + circles) / total_figures

-- Theorem statement (problem)
theorem probability_of_triangle_or_circle : probability_triangle_circle = 7 / 12 :=
by
  -- The proof is omitted, insert the proof here when necessary.
  sorry

end probability_of_triangle_or_circle_l687_687457


namespace distinct_real_roots_unique_l687_687345

theorem distinct_real_roots_unique :
  (∃ k : ℕ, (|x| - (4 / x) = (3 * |x|) / x) → k = 1) := sorry

end distinct_real_roots_unique_l687_687345


namespace limit_d_n_l687_687864

noncomputable def A (n : ℕ) : Matrix (Fin n) (Fin n) ℝ :=
  λ i j => Real.cos (1 + i * n + j)

def d (n : ℕ) : ℝ := (A n).det

theorem limit_d_n : Tendsto (λ n => d n) atTop (𝓝 0) :=
  sorry

end limit_d_n_l687_687864


namespace percentage_difference_l687_687976

def percent1 : ℕ := 60
def value1 : ℕ := 50
def percent2 : ℕ := 50
def value2 : ℕ := 30

theorem percentage_difference : (percent1 / 100.0) * value1 - (percent2 / 100.0) * value2 = 15 :=
by
  sorry

end percentage_difference_l687_687976


namespace polar_to_rectangular_l687_687231

theorem polar_to_rectangular (ρ : ℝ) (x y : ℝ) (h : ρ = 2) (h_rel : x^2 + y^2 = ρ^2) : x^2 + y^2 = 4 :=
by
  rw [h, sq] at h_rel
  exact h_rel

end polar_to_rectangular_l687_687231


namespace dot_product_AB_BC_in_triangle_l687_687419

theorem dot_product_AB_BC_in_triangle :
  ∀ (A B C : EuclideanSpace ℝ (Fin 2)),
  dist A B = 7 →
  dist B C = 5 →
  dist C A = 6 →
  ∥ (B - A) ∥ * ∥ (C - B) ∥ * (⟪B - A, C - B⟫) = -19 :=
by
  -- Naming the points in some finite-dimensional Euclidean space
  intros A B C hAB hBC hCA
  -- Some steps to prove the theorem will go here, but for now we use sorry
  sorry

end dot_product_AB_BC_in_triangle_l687_687419


namespace cos_arcsin_l687_687625

theorem cos_arcsin (h : 8^2 + 15^2 = 17^2) : Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
by
  have opp_hyp_pos : 0 ≤ 8 / 17 := by norm_num
  have hyp_pos : 0 < 17 := by norm_num
  have opp_le_hyp : 8 / 17 ≤ 1 := by norm_num
  sorry

end cos_arcsin_l687_687625


namespace area_triangle_PRT_l687_687399

theorem area_triangle_PRT :
  ∀ (A B C P Q R T : Point)
    (hABC : Triangle A B C)
    (hAngles : ∠ B = 60 ∧ ∠ C = 90)
    (hAB : dist A B = 1)
    (hEquilBCP : EquilateralTriangle B C P)
    (hEquilCAQ : EquilateralTriangle C A Q)
    (hEquilABR : EquilateralTriangle A B R)
    (hQR_meet_AB : Collinear B Q R T ∧ Intersection Q R AB T),
  area (Triangle P R T) = (9 * sqrt 3) / 32 := sorry

end area_triangle_PRT_l687_687399


namespace average_speed_correct_l687_687574

noncomputable def total_distance := 120 + 70
noncomputable def total_time := 2
noncomputable def average_speed := total_distance / total_time

theorem average_speed_correct :
  average_speed = 95 := by
  sorry

end average_speed_correct_l687_687574


namespace range_of_a_l687_687764

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l687_687764


namespace monotonically_increasing_interval_l687_687806

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * (sin x) * (cos x) + (sin x) ^ 2

noncomputable def g (x : ℝ) : ℝ := sin (x - π / 3) + 1 / 2

theorem monotonically_increasing_interval :
  ∃ (a b : ℝ), a = -π / 6 ∧ b = 5 * π / 6 ∧ ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → g x ≤ g y :=
sorry

end monotonically_increasing_interval_l687_687806


namespace binary_subtraction_result_l687_687661

theorem binary_subtraction_result :
  let x := 0b1101101 -- binary notation for 109
  let y := 0b11101   -- binary notation for 29
  let z := 0b101010  -- binary notation for 42
  let product := x * y
  let result := product - z
  result = 0b10000010001 := -- binary notation for 3119
by
  sorry

end binary_subtraction_result_l687_687661


namespace range_of_a_l687_687770

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l687_687770


namespace a1_greater_than_floor_2n_over_3_l687_687981

theorem a1_greater_than_floor_2n_over_3
  (n : ℕ)
  (a : ℕ → ℕ)
  (h1 : ∀ i j : ℕ, i < j → i ≤ n ∧ j ≤ n → a i < a j)
  (h2 : ∀ i j : ℕ, i ≠ j → i ≤ n ∧ j ≤ n → lcm (a i) (a j) > 2 * n)
  (h_max : ∀ i : ℕ, i ≤ n → a i ≤ 2 * n) :
  a 1 > (2 * n) / 3 :=
by
  sorry

end a1_greater_than_floor_2n_over_3_l687_687981


namespace find_a_plus_d_l687_687357

variables (a b c d e : ℝ)

theorem find_a_plus_d :
  a + b = 12 ∧ b + c = 9 ∧ c + d = 3 ∧ d + e = 7 ∧ e + a = 10 → a + d = 6 :=
by
  intros h
  have h1 : a + b = 12 := h.1
  have h2 : b + c = 9 := h.2.1
  have h3 : c + d = 3 := h.2.2.1
  have h4 : d + e = 7 := h.2.2.2.1
  have h5 : e + a = 10 := h.2.2.2.2
  sorry

end find_a_plus_d_l687_687357


namespace coefficient_x5_expansion_l687_687848

noncomputable def polynomial := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

theorem coefficient_x5_expansion :
  polynomial.coeff 5 = -21 :=
by sorry

end coefficient_x5_expansion_l687_687848


namespace increasing_on_0_1_iff_decreasing_on_3_4_l687_687176

variable {f : ℝ → ℝ}

-- Conditions on the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

-- Definitions of increasing and decreasing 
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

-- Theorem statement
theorem increasing_on_0_1_iff_decreasing_on_3_4
  (h_even : is_even f) 
  (h_periodic : is_periodic f 2) : 
  (is_increasing_on f 0 1 ↔ is_decreasing_on f 3 4) :=
begin
  sorry
end

end increasing_on_0_1_iff_decreasing_on_3_4_l687_687176


namespace cosine_of_arcsine_l687_687630

theorem cosine_of_arcsine (h : -1 ≤ (8 : ℝ) / 17 ∧ (8 : ℝ) / 17 ≤ 1) : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
sorry

end cosine_of_arcsine_l687_687630


namespace star_3_2_l687_687510

-- Definition of the operation
def star (a b : ℤ) : ℤ := a * b^3 - b^2 + 2

-- The proof problem
theorem star_3_2 : star 3 2 = 22 :=
by
  sorry

end star_3_2_l687_687510


namespace sum_series_eq_l687_687638

open BigOperators

theorem sum_series_eq : 
  ∑ n in Finset.range 256, (1 : ℝ) / ((2 * (n + 1 : ℕ) - 3) * (2 * (n + 1 : ℕ) + 1)) = -257 / 513 := 
by 
  sorry

end sum_series_eq_l687_687638


namespace zeros_in_99_999_998_squared_l687_687383

theorem zeros_in_99_999_998_squared (n : ℕ) (h : n = 10^8 - 2) : 
  ∃ k : ℕ, k = 8 ∧ count_zeros (n^2) = k := 
sorry

def count_zeros (x : ℕ) : ℕ := 
sorry

end zeros_in_99_999_998_squared_l687_687383


namespace find_uv_non_integer_l687_687436

def p (b : Fin 14 → ℚ) (x y : ℚ) : ℚ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4 + b 12 * x^3 * y^2 + b 13 * y^3 * x^2

variables (b : Fin 14 → ℚ)
variables (u v : ℚ)

def zeros_at_specific_points :=
  p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧
  p b 0 1 = 0 ∧ p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧
  p b (-1) (-1) = 0 ∧ p b 2 2 = 0 ∧ 
  p b 2 (-2) = 0 ∧ p b (-2) 2 = 0

theorem find_uv_non_integer
  (h : zeros_at_specific_points b) :
  p b (5/19) (16/19) = 0 :=
sorry

end find_uv_non_integer_l687_687436


namespace find_pq_l687_687866

-- Define the conditions and the problem statement
variable (AB CD AD BC : ℝ)
variable (P Q : ℝ)
variable (PC QB PQ : ℝ)

-- Assumptions
axiom ab_parallel_cd : AB ∥ CD
axiom ad_perp_ab : AD ⊥ AB
axiom p_touch_cd : true
axiom q_touch_ab : true
axiom pc_val : PC = 36
axiom qb_val : QB = 49

-- Main theorem statement
theorem find_pq (h1 : ab_parallel_cd) (h2 : ad_perp_ab) (h3 : p_touch_cd) (h4 : q_touch_ab) (h5 : pc_val) (h6 : qb_val) :
  PQ = 84 :=
sorry

end find_pq_l687_687866


namespace max_togs_possible_l687_687617

def tag_cost : ℕ := 3
def tig_cost : ℕ := 4
def tog_cost : ℕ := 8
def total_budget : ℕ := 100
def min_tags : ℕ := 1
def min_tigs : ℕ := 1
def min_togs : ℕ := 1

theorem max_togs_possible : 
  ∃ (tags tigs togs : ℕ), tags ≥ min_tags ∧ tigs ≥ min_tigs ∧ togs ≥ min_togs ∧ 
  tag_cost * tags + tig_cost * tigs + tog_cost * togs = total_budget ∧ togs = 11 :=
sorry

end max_togs_possible_l687_687617


namespace max_distance_correct_ring_area_correct_l687_687667

def r : ℝ := 26
def n : ℝ := 7
def w : ℝ := 20
def θ : ℝ := 180 / n

noncomputable def max_distance : ℝ := 24 / sin (Real.pi * θ / 180)

noncomputable def ring_area : ℝ :=
  960 * Real.pi / tan (Real.pi * θ / 180)

theorem max_distance_correct : max_distance = 24 / sin (Real.pi * θ / 180) :=
  by sorry

theorem ring_area_correct : ring_area = 960 * Real.pi / tan (Real.pi * θ / 180) :=
  by sorry

end max_distance_correct_ring_area_correct_l687_687667


namespace sum_of_numerator_and_denominator_l687_687558

theorem sum_of_numerator_and_denominator (x : ℚ) (h : x = 0.36) : 
  let frac : ℚ := (36 / 99).in_lowest_terms
  in frac.num + frac.denom = 15 := 
by sorry

end sum_of_numerator_and_denominator_l687_687558


namespace dimension_tolerance_l687_687494

theorem dimension_tolerance (base_dim : ℝ) (pos_tolerance : ℝ) (neg_tolerance : ℝ) 
  (max_dim : ℝ) (min_dim : ℝ) 
  (h_base : base_dim = 7) 
  (h_pos_tolerance : pos_tolerance = 0.05) 
  (h_neg_tolerance : neg_tolerance = 0.02) 
  (h_max_dim : max_dim = base_dim + pos_tolerance) 
  (h_min_dim : min_dim = base_dim - neg_tolerance) :
  max_dim = 7.05 ∧ min_dim = 6.98 :=
by
  sorry

end dimension_tolerance_l687_687494


namespace X_on_radical_axis_of_ACQ_and_BDP_l687_687921

-- Definitions based on conditions
variable {A B C D M P Q X : Type}
variable [cyclic_quadrilateral A B C D]
variable (MA MD : Line)
variable (ω : Circle)
variable (circum_ABCD : Circle)
variable [touches_segment ω MA at P]
variable [touches_segment ω MD at Q]
variable [touches_circle ω circum_ABCD at X]
variable [meet_diagonals A B C D M]

-- The theorem to be proved
theorem X_on_radical_axis_of_ACQ_and_BDP :
  lies_on_radical_axis X (circle ACQ) (circle BDP) :=
sorry -- Proof to be filled in

end X_on_radical_axis_of_ACQ_and_BDP_l687_687921


namespace total_games_l687_687454

-- Defining the conditions.
def games_this_month : ℕ := 9
def games_last_month : ℕ := 8
def games_next_month : ℕ := 7

-- Theorem statement to prove the total number of games.
theorem total_games : games_this_month + games_last_month + games_next_month = 24 := by
  sorry

end total_games_l687_687454


namespace area_of_gergonne_triangle_l687_687209

variables (T r s A B C : ℝ)

-- Conditions of the problem
def area_of_triangle : ℝ := T
def inradius_of_triangle : ℝ := r

-- Gergonne triangle area calculation
def gergonne_triangle_area (A B C : ℝ) : ℝ := 
    r^2 * (Real.cot (A / 2) + Real.cot (B / 2) + Real.cot (C / 2))

-- Main statement of the problem
theorem area_of_gergonne_triangle (h1 : r = T / s) (h2 : T = r * s) :
    ∃ A' : ℝ, A' = r^2 * (Real.cot (A / 2) + Real.cot (B / 2) + Real.cot (C / 2)) :=
    sorry

end area_of_gergonne_triangle_l687_687209


namespace maximum_daily_profit_l687_687203

variable (x : ℝ) (p : ℝ) (d : ℝ)

def daily_profit : ℝ := [200 - 10 * (x - 50)] * (x - 40)

theorem maximum_daily_profit :
  daily_profit 55 = 2250 := by
  sorry

end maximum_daily_profit_l687_687203


namespace ratatouille_cost_per_quart_l687_687473

theorem ratatouille_cost_per_quart :
  let eggplant_pounds := 5
  let zucchini_pounds := 4
  let tomato_pounds := 4
  let onion_pounds := 3
  let basil_pounds := 1
  let eggplant_zucchini_cost_per_pound := 2.0
  let tomato_cost_per_pound := 3.5
  let onion_cost_per_pound := 1.0
  let basil_cost_per_half_pound := 2.5
  let yield_quarts := 4
  let eggplant_zucchini_total_cost := (eggplant_pounds + zucchini_pounds) * eggplant_zucchini_cost_per_pound
  let tomato_total_cost := tomato_pounds * tomato_cost_per_pound
  let onion_total_cost := onion_pounds * onion_cost_per_pound
  let basil_total_cost := (basil_pounds / 0.5) * basil_cost_per_half_pound
  let total_cost := eggplant_zucchini_total_cost + tomato_total_cost + onion_total_cost + basil_total_cost
  total_cost / yield_quarts = 10 :=
by
  let eggplant_pounds := 5
  let zucchini_pounds := 4
  let tomato_pounds := 4
  let onion_pounds := 3
  let basil_pounds := 1
  let eggplant_zucchini_cost_per_pound := 2.0
  let tomato_cost_per_pound := 3.5
  let onion_cost_per_pound := 1.0
  let basil_cost_per_half_pound := 2.5
  let yield_quarts := 4
    
  let eggplant_zucchini_total_cost := (eggplant_pounds + zucchini_pounds) * eggplant_zucchini_cost_per_pound
  let tomato_total_cost := tomato_pounds * tomato_cost_per_pound
  let onion_total_cost := onion_pounds * onion_cost_per_pound
  let basil_total_cost := (basil_pounds / 0.5) * basil_cost_per_half_pound
  let total_cost := eggplant_zucchini_total_cost + tomato_total_cost + onion_total_cost + basil_total_cost

  have h_total_cost : total_cost = 40.0 := by sorry
  have h_cost_per_quart : total_cost / yield_quarts = 10 := by 
    rw [h_total_cost]
    exact sorry

  show total_cost / yield_quarts = 10 from h_cost_per_quart
  sorry

end ratatouille_cost_per_quart_l687_687473


namespace number_of_subtractions_resulting_in_1_l687_687872

def is_binary (n : ℕ) : Prop := ∀ d, ¬ d ∈ digits 2 n ∨ (d = 0 ∨ d = 1)

def binary_number_with_5_zeros_and_8_ones (n : ℕ) : Prop :=
  is_binary n ∧ (count 0 (digits 2 n) = 5) ∧ (count 1 (digits 2 n) = 8)

def set_B : set ℕ := {n | binary_number_with_5_zeros_and_8_ones n}

theorem number_of_subtractions_resulting_in_1 :
  (∃ s1 s2 ∈ set_B, s1 - s2 = 1) ∧ card {s | ∃ s1 s2 ∈ set_B, s1 - s2 = 1} = 330 :=
sorry

end number_of_subtractions_resulting_in_1_l687_687872


namespace John_next_birthday_age_l687_687431

variable (John Mike Lucas : ℝ)

def John_is_25_percent_older_than_Mike := John = 1.25 * Mike
def Mike_is_30_percent_younger_than_Lucas := Mike = 0.7 * Lucas
def sum_of_ages_is_27_point_3_years := John + Mike + Lucas = 27.3

theorem John_next_birthday_age 
  (h1 : John_is_25_percent_older_than_Mike John Mike) 
  (h2 : Mike_is_30_percent_younger_than_Lucas Mike Lucas) 
  (h3 : sum_of_ages_is_27_point_3_years John Mike Lucas) : 
  John + 1 = 10 := 
sorry

end John_next_birthday_age_l687_687431


namespace required_run_rate_l687_687415

noncomputable def run_rate_needed : ℕ → ℕ → ℕ → ℕ → ℚ :=
  λ target runs20 first_overs remaining_overs,
  (target - runs20) / remaining_overs

theorem required_run_rate (target runs20 first_overs remaining_overs : ℕ)
  (run_rate20 : ℚ) (h1 : run_rate20 * first_overs = runs20)
  (h2 : target = 350) (h3 : first_overs = 20) (h4 : run_rate20 = 4.5)
  (h5 : remaining_overs = 30) :
  run_rate_needed target runs20 first_overs remaining_overs = 8.67 :=
by {
  sorry
}

end required_run_rate_l687_687415


namespace range_of_a_l687_687708

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l687_687708


namespace a_plus_b_eq_six_l687_687392

open Set

variable (a b : ℝ)

def setA : Set ℝ := {5, Real.log a / Real.log 2}
def setB : Set ℝ := {b}

theorem a_plus_b_eq_six (h1 : setA a b ∩ setB b = {2}) : a + b = 6 := 
by
  sorry

end a_plus_b_eq_six_l687_687392


namespace cosine_of_arcsine_l687_687631

theorem cosine_of_arcsine (h : -1 ≤ (8 : ℝ) / 17 ∧ (8 : ℝ) / 17 ≤ 1) : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
sorry

end cosine_of_arcsine_l687_687631


namespace num_monograms_correct_l687_687892

def n : ℕ := 26  -- Total letters in the alphabet
def k : ℕ := 2   -- Number of letters to choose for first and middle initials

def num_monograms : ℕ :=
  Nat.choose (n - 1) k  -- Choosing 2 letters from the remaining 25

theorem num_monograms_correct : num_monograms = 300 :=
  by
    have : num_monograms = Nat.choose 25 2 := by rfl
    have : Nat.choose 25 2 = 300 := by sorry  -- Skip the proof for the combination formula
    rw [this]
    rfl

end num_monograms_correct_l687_687892


namespace find_missing_number_l687_687132

-- Define the set of given numbers
def given_numbers : List ℕ := [1, 2, 3, 5, 7, 8, 9]

-- Define the assertion that adding the missing number 8 makes the median 8
theorem find_missing_number :
  let complete_numbers := 1 :: 2 :: 3 :: 5 :: 7 :: 8 :: 8 :: 9 :: []
  let sorted_numbers := complete_numbers.qsort (≤)
  median_equals_8 : (sorted_numbers.nth (sorted_numbers.length / 2)).iget = 8
  :=
  sorry

end find_missing_number_l687_687132


namespace range_of_a_l687_687766

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l687_687766


namespace range_of_a_l687_687769

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l687_687769


namespace log_neither_even_nor_odd_l687_687043

noncomputable def log_function (x : ℝ) : ℝ := real.log x

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem log_neither_even_nor_odd : ¬is_even log_function ∧ ¬is_odd log_function :=
by
  sorry

end log_neither_even_nor_odd_l687_687043


namespace range_of_a_l687_687780

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687780


namespace point_B_third_quadrant_l687_687846

theorem point_B_third_quadrant (m n : ℝ) (hm : m < 0) (hn : n < 0) :
  (-m * n < 0) ∧ (m < 0) :=
by
  sorry

end point_B_third_quadrant_l687_687846


namespace power_mod_remainder_l687_687960

theorem power_mod_remainder {a b c : ℕ} (h : a = 17) (h1 : b = 1501) (h2 : c = 19) : a^b % c = 4 :=
by {
  rw [h, h1, h2],
  sorry
}

end power_mod_remainder_l687_687960


namespace angle_MKF_is_45_l687_687835

/-
Definitions of the problem conditions
-/

def triangle (A B C : Type) := ∃ (α β γ : ℝ), α + β + γ = 180

structure triangle_with_points (D E F K M : Type) := 
  (angle_D : ℝ)
  (angle_E : ℝ)
  (angle_F : ℝ)
  (altitude_DK : Prop)
  (median_EM : Prop)

noncomputable def measure_angle_MKF (angles : triangle_with_points D E F K M) : ℝ :=
  let ⟨_, α, β, γ, sum_angles⟩ := angles in
  90 - 45 -- replace this with the calculation steps if proof is needed

/-
Statement of the proof problem
-/

theorem angle_MKF_is_45 (D E F K M : Type) 
  (angles : triangle_with_points D E F K M)
  (h1 : angles.angle_D = 70) 
  (h2 : angles.angle_E = 60)
  (h3 : angles.angle_F = 50)
  (h4 : angles.altitude_DK)
  (h5 : angles.median_EM) : 
  measure_angle_MKF angles = 45 := 
  by sorry

end angle_MKF_is_45_l687_687835


namespace part1_part2_l687_687993

-- Proof that for 0 < x < 1, x - x^2 < sin x < x
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := 
sorry

-- Proof that if x = 0 is a local maximum of f(x) = cos(ax) - ln(1 - x^2), then a is in the specified range.
theorem part2 (a : ℝ) (h : ∀ x, (cos(a * x) - log(1 - x^2))' (0) = 0 ∧ (cos(a * x) - log(1 - x^2))'' (0) < 0) : 
  a ∈ Set.Ioo (-∞) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) (∞) := 
sorry

end part1_part2_l687_687993


namespace mean_of_remaining_students_l687_687028

noncomputable def mean_remaining_students (k : ℕ) (h : k > 18) (mean_class : ℚ) (mean_18_students : ℚ) : ℚ :=
  (12 * k - 360) / (k - 18)

theorem mean_of_remaining_students (k : ℕ) (h : k > 18) (mean_class_eq : mean_class = 12) (mean_18_eq : mean_18_students = 20) :
  mean_remaining_students k h mean_class mean_18_students = (12 * k - 360) / (k - 18) :=
by sorry

end mean_of_remaining_students_l687_687028


namespace at_least_100_valid_pairs_l687_687523

-- Define the conditions
def boots_distribution (L41 L42 L43 R41 R42 R43 : ℕ) : Prop :=
  L41 + L42 + L43 = 300 ∧ R41 + R42 + R43 = 300 ∧
  (L41 = 200 ∨ L42 = 200 ∨ L43 = 200) ∧
  (R41 = 200 ∨ R42 = 200 ∨ R43 = 200)

-- Define the theorem to be proven
theorem at_least_100_valid_pairs (L41 L42 L43 R41 R42 R43 : ℕ) :
  boots_distribution L41 L42 L43 R41 R42 R43 → 
  (L41 ≥ 100 ∧ R41 ≥ 100 ∨ L42 ≥ 100 ∧ R42 ≥ 100 ∨ L43 ≥ 100 ∧ R43 ≥ 100) → 100 ≤ min L41 R41 ∨ 100 ≤ min L42 R42 ∨ 100 ≤ min L43 R43 :=
  sorry

end at_least_100_valid_pairs_l687_687523


namespace fit_seven_rectangles_l687_687083

theorem fit_seven_rectangles (s : ℝ) (a : ℝ) : (s > 0) → (a > 0) → (14 * a ^ 2 ≤ s ^ 2 ∧ 2 * a ≤ s) → 
  (∃ (rectangles : Fin 7 → (ℝ × ℝ)), ∀ i, rectangles i = (a, 2 * a) ∧
   ∀ i j, i ≠ j → rectangles i ≠ rectangles j) :=
sorry

end fit_seven_rectangles_l687_687083


namespace find_b_monotonic_decreasing_range_of_m_solution_l687_687322

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := (-2^x - b) / (2^(x+1) + 2)

theorem find_b (h : ∀ x, f (-x) b = -f x b) : b = -1 :=
sorry

noncomputable def g (x : ℝ) : ℝ := (1 - 2^x) / (2^(x+1) + 2)

theorem monotonic_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → g x1 > g x2 :=
sorry

theorem range_of_m_solution : ∀ m : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ g x = m) ↔ m ∈ Icc (-1/6 : ℝ) 0 :=
sorry

end find_b_monotonic_decreasing_range_of_m_solution_l687_687322


namespace correct_statement_A_incorrect_statement_B_incorrect_statement_C_incorrect_statement_D_l687_687155

-- Define conditions for each statement
def condition_A (m : ℝ) : Prop := ∀ x : ℝ, m > 0 ∧ (m > 1/2 ∨ m < -1/2) → mx^2 + x + m > 0
def condition_B (x y : ℝ) : Prop := x^2 + y^2 ≥ 4 → x ≥ 2 ∧ y ≥ 2
def condition_C (a : ℝ) : Prop := (a > 1 → a^2 > a) ∧ (a^2 > a → a > 1)
def condition_D (a : ℝ) : Prop := (∀ x ∈ Set.Icc 0 1, x + a > 0) → a ∈ Set.Ioi 0

-- Correctness of statements
theorem correct_statement_A : condition_A :=
sorry

theorem incorrect_statement_B : ¬∃ x y : ℝ, condition_B x y :=
sorry

theorem incorrect_statement_C : ¬∃ a : ℝ, condition_C a :=
sorry

theorem incorrect_statement_D : ¬∃ a : ℝ, condition_D a :=
sorry

end correct_statement_A_incorrect_statement_B_incorrect_statement_C_incorrect_statement_D_l687_687155


namespace range_of_a_l687_687700

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687700


namespace remainder_largest_multiple_12_mod_100_l687_687444

theorem remainder_largest_multiple_12_mod_100 :
  ∃ M : ℕ, (∀ d ∈ finset.range 1 10, d ∈ digits 10 M) ∧
           (nat.gcd M 12 = 12) ∧
           (∀ m, (∀ d ∈ finset.range 1 10, d ∈ digits 10 m) ∧ (nat.gcd m 12 = 12) → m ≤ M) ∧
           M % 100 = 12 :=
sorry

end remainder_largest_multiple_12_mod_100_l687_687444


namespace range_of_x0_l687_687081

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 2^(-x) - 1 else x^2

theorem range_of_x0 (x0 : ℝ) (h : f x0 > 1) : 
  x0 ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo 1 ∞ := by
  sorry -- Proof goes here

end range_of_x0_l687_687081


namespace JimsInvestment_l687_687434

variable (totalInvestment : Int) 
variable (ratioJohn ratioJames ratioJim : Int)

-- Given conditions of the problem:
def investment_ratings := (ratioJohn = 4) ∧ (ratioJames = 7) ∧ (ratioJim = 9)
def total_investment := totalInvestment = 80000

-- The theorem to prove Jim's investment:
theorem JimsInvestment (totalInvestment : Int) (ratioJohn ratioJames ratioJim : Int) 
  (hratios : investment_ratings ∧ total_investment totalInvestment) : 
  totalInvestment * ratioJim / (ratioJohn + ratioJames + ratioJim) = 36000 :=
by 
  sorry

end JimsInvestment_l687_687434


namespace max_knight_path_length_5x5_l687_687248

def is_valid_move (start finish : ℕ × ℕ) : Prop :=
  (abs (start.1 - finish.1) = 1 ∧ abs (start.2 - finish.2) = 3) ∨
  (abs (start.1 - finish.1) = 3 ∧ abs (start.2 - finish.2) = 1)

def is_valid_path (board_size : ℕ) (path : List (ℕ × ℕ)) : Prop :=
  path.Nodup ∧ (∀ i, i < path.length - 1 → is_valid_move (path.get! i) (path.get! (i + 1))) ∧
  (∀ sq, sq ∈ path → sq.1 ≥ 1 ∧ sq.1 ≤ board_size ∧ sq.2 ≥ 1 ∧ sq.2 ≤ board_size)

theorem max_knight_path_length_5x5 : ∃ path : List (ℕ × ℕ), length path = 13 ∧ is_valid_path 5 path :=
sorry

end max_knight_path_length_5x5_l687_687248


namespace tensor_value_l687_687642

variables (h : ℝ)

def tensor (x y : ℝ) : ℝ := x^2 - y^2

theorem tensor_value : tensor h (tensor h h) = h^2 :=
by 
-- Complete proof body not required, 'sorry' is used for omitted proof
sorry

end tensor_value_l687_687642


namespace total_words_in_week_l687_687620

def typing_minutes_MWF : ℤ := 260
def typing_minutes_TTh : ℤ := 150
def typing_minutes_Sat : ℤ := 240
def typing_speed_MWF : ℤ := 50
def typing_speed_TTh : ℤ := 40
def typing_speed_Sat : ℤ := 60

def words_per_day_MWF : ℤ := typing_minutes_MWF * typing_speed_MWF
def words_per_day_TTh : ℤ := typing_minutes_TTh * typing_speed_TTh
def words_Sat : ℤ := typing_minutes_Sat * typing_speed_Sat

def total_words_week : ℤ :=
  (words_per_day_MWF * 3) + (words_per_day_TTh * 2) + words_Sat + 0

theorem total_words_in_week :
  total_words_week = 65400 :=
by
  sorry

end total_words_in_week_l687_687620


namespace shells_arrangement_count_l687_687433

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem shells_arrangement_count : factorial 14 / 7 = 87178291200 :=
by
  sorry

end shells_arrangement_count_l687_687433


namespace square_side_to_diagonal_ratio_l687_687201

theorem square_side_to_diagonal_ratio (s : ℝ) : 
  s / (s * Real.sqrt 2) = Real.sqrt 2 / 2 :=
by
  sorry

end square_side_to_diagonal_ratio_l687_687201


namespace total_cost_of_shoes_l687_687088

theorem total_cost_of_shoes 
  (cost_first_pair : ℕ)
  (percentage_increase : ℕ)
  (price_first : cost_first_pair = 22)
  (percentage_increase_eq : percentage_increase = 50) :
  let additional_cost := (percentage_increase * cost_first_pair) / 100
  let cost_second_pair := cost_first_pair + additional_cost
  let total_cost := cost_first_pair + cost_second_pair
  in total_cost = 55 :=
by
  sorry

end total_cost_of_shoes_l687_687088


namespace student_attends_all_three_l687_687845

open Finset

variables (F G C : Finset ℕ) (n : ℕ)

theorem student_attends_all_three (hF : F.card = 22) (hG : G.card = 21) 
                               (hC : C.card = 18) (hn : n = 30) : 
  ∃ s, s ∈ F ∧ s ∈ G ∧ s ∈ C :=
sorry

end student_attends_all_three_l687_687845


namespace star_evaluation_l687_687308

def star (a b : ℝ) : ℝ :=
  if a <= b then b else real.sqrt (a^2 - b^2)

theorem star_evaluation : star (real.sqrt 7) (star (real.sqrt 2) (real.sqrt 3)) = 2 * real.sqrt 10 := by
  sorry

end star_evaluation_l687_687308


namespace right_triangle_area_l687_687901

theorem right_triangle_area (c : ℝ) (h1 : c > 0)
  (h2 : ∃ (a b : ℝ), a^2 + b^2 = c^2 ∧ (a / c = (cos (real.pi / 12)))) :
  1 / 2 * a * b = 1 / 8 * c^2 :=
by
  sorry

end right_triangle_area_l687_687901


namespace smallest_k_l687_687524

theorem smallest_k (s : Finset ℕ) (h1 : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 2016) (n : ℕ) (h2 : s.card = n) :
  ∃ (a b ∈ s), a ≠ b ∧ abs (Nat.cbrt a - Nat.cbrt b) < 1 ↔ n ≥ 13 := by
  sorry

end smallest_k_l687_687524


namespace intersection_of_sets_l687_687373

def set_A (x : ℝ) := x + 1 ≤ 3
def set_B (x : ℝ) := 4 - x^2 ≤ 0

theorem intersection_of_sets : {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | x ≤ -2} ∪ {2} :=
by
  sorry

end intersection_of_sets_l687_687373


namespace value_of_x_l687_687013

-- Let a and b be real numbers.
variable (a b : ℝ)

-- Given conditions
def cond_1 : 10 * a = 6 * b := sorry
def cond_2 : 120 * a * b = 800 := sorry

theorem value_of_x (x : ℝ) (h1 : 10 * a = x) (h2 : 6 * b = x) (h3 : 120 * a * b = 800) : x = 20 :=
sorry

end value_of_x_l687_687013


namespace factor_expression_l687_687255

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687255


namespace factor_expression_l687_687267

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l687_687267


namespace max_participants_l687_687842

variable (A B C : Set ℕ) (ps : |A| = 8) (ms : |B| = 7) (pr : |C| = 11)
variable (A_cap_B : |A ∩ B| ≥ 2) (B_cap_C : |B ∩ C| ≥ 3) (A_cap_C : |A ∩ C| ≥ 4)

theorem max_participants (x : ℕ) (A_int_B_int_C : |A ∩ B ∩ C| = x) : A ∪ B ∪ C = 19 := by
  sorry

end max_participants_l687_687842


namespace establishmentYear_is_BingShen_l687_687114

def heavenlyStems : List String :=
  ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]

def earthlyBranches : List String :=
  ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

def heavenlyStemAt (yearDifference : Nat) : String :=
  heavenlyStems[(10 - (yearDifference % 10)) % 10]

def earthlyBranchAt (yearDifference : Nat) : String :=
  earthlyBranches[(12 - (yearDifference % 12)) % 12]

theorem establishmentYear_is_BingShen :
  ∀ (currentYear : Nat) (currentStem : String) (currentBranch : String),
  currentYear = 2023 →
  currentStem = "Gui" →
  currentBranch = "Mao" →
  heavenlyStemAt 67 = "Bing" ∧ earthlyBranchAt 67 = "Shen" :=
by
  intros currentYear currentStem currentBranch hYear hStem hBranch
  sorry

end establishmentYear_is_BingShen_l687_687114


namespace melanie_total_amount_l687_687084

theorem melanie_total_amount :
  let g1 := 12
  let g2 := 15
  let g3 := 8
  let g4 := 10
  let g5 := 20
  g1 + g2 + g3 + g4 + g5 = 65 :=
by
  sorry

end melanie_total_amount_l687_687084


namespace increasing_on_positive_reals_l687_687126

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem increasing_on_positive_reals : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y := by
  sorry

end increasing_on_positive_reals_l687_687126


namespace talent_show_l687_687940

theorem talent_show (B G : ℕ) (h1 : G = B + 22) (h2 : G + B = 34) : G = 28 :=
by
  sorry

end talent_show_l687_687940


namespace train_crossing_time_in_seconds_l687_687606

-- Given conditions
def train_length : ℕ := 160
def bridge_length : ℕ := 215
def train_speed_km_hr : ℝ := 45

-- Derived quantities
def total_distance : ℕ := train_length + bridge_length
def train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600

-- The theorem we want to prove
theorem train_crossing_time_in_seconds : total_distance / train_speed_m_s = 30 := by
  sorry

end train_crossing_time_in_seconds_l687_687606


namespace var_power_eight_l687_687017

variable (k j : ℝ)
variable {x y z : ℝ}

theorem var_power_eight (hx : x = k * y^4) (hy : y = j * z^2) : ∃ c : ℝ, x = c * z^8 :=
by
  sorry

end var_power_eight_l687_687017


namespace ship_divisibility_by_4_l687_687953

/-- Definition of a ship. -/
def is_ship (s : set (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ), ((x, y) ∈ s) → (∃ (x' y' : ℕ), abs (x - x') + abs (y - y') = 1 ∧ (x', y') ∈ s)

/-- Definition of "different ships". -/
def are_different_ships (s1 s2 : set (ℕ × ℕ)) : Prop :=
  ¬ (∃ (θ : ℤ), s1 = (rotate θ s2)) ∧ ¬ (s1 = (reflect s2))

/-- Predicate for an odd number of ships with n unit squares. -/
def odd_number_of_ships (n : ℕ) : Prop :=
  ∃ (S : set (set (ℕ × ℕ))), is_ship S ∧ S.card = n ∧ finset.card S % 2 = 1

/-- The main theorem: if there is an odd number of possible different ships
    consisting of n unit squares on a 10×10 board, then n is divisible by 4. -/
theorem ship_divisibility_by_4 (n : ℕ) (h : odd_number_of_ships n) : n % 4 = 0 :=
by {
  sorry
}

end ship_divisibility_by_4_l687_687953


namespace range_of_a_l687_687722

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687722


namespace cakesServedDuringDinner_today_is_6_l687_687601

def cakesServedDuringDinner (x : ℕ) : Prop :=
  5 + x + 3 = 14

theorem cakesServedDuringDinner_today_is_6 : cakesServedDuringDinner 6 :=
by
  unfold cakesServedDuringDinner
  -- The proof is omitted
  sorry

end cakesServedDuringDinner_today_is_6_l687_687601


namespace incorrect_statement_B_l687_687153

theorem incorrect_statement_B (A B C : ℝ) (hAB : A * B < 0) (hBC : B * C < 0) : ¬ ∀ (x y : ℝ), x * y + A * x + B * y + C = 0 → (x < 0 ∧ y < 0) :=
by
  sorry

end incorrect_statement_B_l687_687153


namespace no_ordered_triples_satisfy_conditions_l687_687232

theorem no_ordered_triples_satisfy_conditions :
  ¬ (∃ a b c : ℤ, a ≥ 3 ∧ b ≥ 2 ∧ c ≥ 0 ∧ log a b = 4^c ∧ a + b + c = 2023) :=
sorry

end no_ordered_triples_satisfy_conditions_l687_687232


namespace range_of_a_l687_687753

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687753


namespace total_selling_price_correct_l687_687596

noncomputable def calculateSellingPrice (price1 price2 price3 loss1 loss2 loss3 taxRate overheadCost : ℝ) : ℝ :=
  let totalPurchasePrice := price1 + price2 + price3
  let tax := taxRate * totalPurchasePrice
  let sellingPrice1 := price1 - (loss1 * price1)
  let sellingPrice2 := price2 - (loss2 * price2)
  let sellingPrice3 := price3 - (loss3 * price3)
  let totalSellingPrice := sellingPrice1 + sellingPrice2 + sellingPrice3
  totalSellingPrice + overheadCost + tax

theorem total_selling_price_correct :
  calculateSellingPrice 750 1200 500 0.10 0.15 0.05 0.05 300 = 2592.5 :=
by 
  -- The proof of this theorem is skipped.
  sorry

end total_selling_price_correct_l687_687596


namespace range_of_a_l687_687714

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l687_687714


namespace logarithmic_sufficient_exponential_l687_687668

theorem logarithmic_sufficient_exponential (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (log 2 a > log 2 b) → (2^a > 2^b) ∧ ¬((2^a > 2^b) → (log 2 a > log 2 b)) :=
by 
  sorry

end logarithmic_sufficient_exponential_l687_687668


namespace sqrt_domain_l687_687538

theorem sqrt_domain (x : ℝ) : 1 - x ≥ 0 → x ≤ 1 := by
  sorry

end sqrt_domain_l687_687538


namespace parabola_vertex_properties_l687_687178

theorem parabola_vertex_properties :
  ∃ (d e f : ℝ), 
    (∀ (x y : ℝ), x = d * y^2 + e * y + f) ∧ 
    (∀ (y : ℝ), x = d * (y + 6)^2 + 7) ∧
    (x = 2 ∧ y = -3) → 
    d + e + f = -182 / 9 :=
by
  sorry

end parabola_vertex_properties_l687_687178


namespace fraction_of_spotted_brown_toads_l687_687492

-- Definitions for the conditions
def green_toads_per_acre : ℕ := 8
def brown_toads_per_green_toad : ℕ := 25
def spotted_brown_toads_per_acre : ℕ := 50

-- Theorem statement
theorem fraction_of_spotted_brown_toads :
  let brown_toads_per_acre := green_toads_per_acre * brown_toads_per_green_toad in
  let fraction_of_spotted_brown_toads := (spotted_brown_toads_per_acre : ℚ) / brown_toads_per_acre in
  fraction_of_spotted_brown_toads = 1 / 4 :=
by
  sorry

end fraction_of_spotted_brown_toads_l687_687492


namespace find_ellipse_equation_find_segment_length_l687_687351

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (a^2 - b^2 = (a * (sqrt 2 / 2))^2) ∧ (a - a * (sqrt 2 / 2) = sqrt 2 - 1) ∧ (b^2 = 1)

theorem find_ellipse_equation :
  ∃ a b : ℝ, ellipse_equation a b ∧ (a^2 = 2) := sorry

noncomputable def segment_length (a b : ℝ) (k : ℝ) : ℝ :=
  let x1 := -8 * k / (1 + 2 * k^2)
  let x2 := 6 / (1 + 2 * k^2)
  (abs (x1 - x2) * sqrt (1 + k^2))

noncomputable def check_area (a b : ℝ) (k : ℝ) : Prop :=
  ellipse_equation a b ∧ a = sqrt 2 ∧ b = 1 ∧ sorry -- additional details to confirm k condition

theorem find_segment_length :
  ∃ k : ℝ, check_area (sqrt 2) 1 k ∧ segment_length (sqrt 2) 1 k = (3 / 2) := sorry

end find_ellipse_equation_find_segment_length_l687_687351


namespace value_of_a_l687_687821

theorem value_of_a (x : ℝ) (h : (1 - x^32) ≠ 0):
  (8 * a / (1 - x^32) = 
   2 / (1 - x) + 2 / (1 + x) + 
   4 / (1 + x^2) + 8 / (1 + x^4) + 
   16 / (1 + x^8) + 32 / (1 + x^16)) → 
  a = 8 := sorry

end value_of_a_l687_687821


namespace cubic_polynomial_at_2_and_neg2_l687_687073

-- Definitions based on the conditions
def Q (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Hypotheses from the conditions
variables (Q : ℝ → ℝ) (m a b c d : ℝ)
hypothesis h0 : Q 0 = 2 * m
hypothesis h1 : Q 1 = 5 * m
hypothesis h2 : Q (-1) = 7 * m

-- Proving the required result
theorem cubic_polynomial_at_2_and_neg2 : Q 2 + Q (-2) = 36 * m :=
by 
  -- To be solved
  sorry

end cubic_polynomial_at_2_and_neg2_l687_687073


namespace factor_expression_l687_687252

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687252


namespace math_proof_problem_l687_687038

-- Definition of the parametric equation of the curve
def parametric_equation_of_C (θ : ℝ) : ℝ × ℝ :=
  (1 + sqrt 7 * cos θ, sqrt 7 * sin θ)

-- Definition to establish polar system, converting parametric to polar
def polar_equation_of_C(ρ θ : ℝ) : Prop :=
  ρ^2 - 2 * ρ * cos θ - 6 = 0

-- Line l_1 and Ray l_2
def line_l1 (ρ θ : ℝ) : Prop :=
  2 * ρ * sin (θ + π / 3) - sqrt 3 = 0

def ray_l2 (ρ θ : ℝ) : Prop :=
  θ = π / 3

-- Problem (II) conditions
def point_P := (3, π / 3)
def point_Q := (1, π / 3)

-- The length of segment PQ
def segment_PQ_length (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1).abs

-- Theorem to prove
theorem math_proof_problem :
  (∀ θ ρ, parametric_equation_of_C θ =
          (ρ * cos θ, ρ * sin θ) → polar_equation_of_C ρ θ) ∧
  segment_PQ_length point_P point_Q = 2 :=
by sorry

end math_proof_problem_l687_687038


namespace hyperbola_eccentricity_l687_687828

noncomputable theory
open_locale classical
open_locale real

-- Definitions
def parabola_focus : ℝ × ℝ := (0, 1)
def hyperbola_asymptote (a b : ℝ) : (ℝ × ℝ) → ℝ := λ (x y), b * x + a * y
def distance_from_point_to_line (a b x₀ y₀ : ℝ) : ℝ := a / real.sqrt (a^2 + b^2)

-- Conditions
def condition_distance (a b : ℝ) : Prop := 
  distance_from_point_to_line a b 0 1 = 1 / 3 ∧ 0 < a ∧ 0 < b

-- Theorem to be proven
theorem hyperbola_eccentricity (a b : ℝ) (h: condition_distance a b) : 
  (real.sqrt (a^2 + b^2) / a) = real.sqrt 3 :=
sorry

end hyperbola_eccentricity_l687_687828


namespace eval_definite_integral_l687_687649

variable {a b : ℝ}
variable {f : ℝ → ℝ}
variable [Differentiable ℝ f]

theorem eval_definite_integral (h_diff : ∀ x, HasDerivAt f (deriv f x) x)
  (h_integ : IntervalIntegrable (deriv f) volume (3 * a) (3 * b)) :
  ∫ x in a..b, deriv f (3 * x) = (1 / 3) * (f (3 * b) - f (3 * a)) :=
by
  sorry

end eval_definite_integral_l687_687649


namespace range_of_a_l687_687742

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687742


namespace identify_irrational_number_l687_687214

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def A := Real.sqrt 2
def B := 3.1415926
def C := 2 / 11
def D := Real.cbrt 8

theorem identify_irrational_number :
  is_irrational A ∧ ¬ is_irrational B ∧ ¬ is_irrational C ∧ ¬ is_irrational D :=
by
  sorry

end identify_irrational_number_l687_687214


namespace locate_point_P_l687_687442

variable {A B C P : Type} [EuclideanGeometry A B C P]

-- Definitions based on conditions provided in the problem
-- Define the triangle ABC, with vertices A, B, and C
-- Define that the triangle is not isosceles at vertex A
def triangle_ABC_not_isosceles_at_A (A B C : Type) : Prop :=
  ∃ a b c : ℝ, a ≠ c ∧ True

-- Define an angle bisector from vertex A
def angle_bisector (A B C : Type) [EuclideanGeometry A B C P] : Prop :=
  let bisector := angle_bisects A B C
  True

-- Define perpendicular bisector of BC
def perpendicular_bisector_BC (B C P : Type) [EuclideanGeometry A B C P] : Prop :=
  let pb := perp_bisects B C
  True

-- Statement: proving P lies on the circumcircle of triangle ABC
theorem locate_point_P {A B C P : Type} [EuclideanGeometry A B C P]
    (h1 : triangle_ABC_not_isosceles_at_A A B C)
    (h2 : ∃ P, angle_bisector A B C ∧ perpendicular_bisector_BC B C P)
  : lies_on_circumcircle A B C P :=
  sorry

end locate_point_P_l687_687442


namespace range_of_a_l687_687756

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687756


namespace exists_group_of_three_friends_l687_687402

-- Defining the context of the problem
def people := Fin 10 -- a finite set of 10 people
def quarrel (x y : people) : Prop := -- a predicate indicating a quarrel between two people
sorry

-- Given conditions
axiom quarreled_pairs : ∃ S : Finset (people × people), S.card = 14 ∧ 
  ∀ {x y : people}, (x, y) ∈ S → x ≠ y ∧ quarrel x y

-- Question: Prove there exists a set of 3 friends among these 10 people
theorem exists_group_of_three_friends (p : Finset people):
  ∃ (group : Finset people), group.card = 3 ∧ ∀ {x y : people}, 
  x ∈ group → y ∈ group → x ≠ y → ¬ quarrel x y :=
sorry

end exists_group_of_three_friends_l687_687402


namespace area_and_cost_of_path_l687_687568

def field_length : ℕ := 95
def field_width : ℕ := 55
def path_width : ℕ := 5 / 2
def cost_per_sq_m : ℕ := 2

theorem area_and_cost_of_path :
  let total_length := field_length + 2 * path_width in
  let total_width := field_width + 2 * path_width in
  let total_area := total_length * total_width in
  let field_area := field_length * field_width in
  let path_area := total_area - field_area in
  let cost := path_area * cost_per_sq_m in
  path_area = 775 ∧ cost = 1550 :=
by
  sorry

end area_and_cost_of_path_l687_687568


namespace perp_lines_parallel_lines_l687_687374

noncomputable def line1 (a : ℝ) : ℝ → ℝ → Prop :=
λ x y, a * x + 2 * y + 6 = 0

noncomputable def line2 (a : ℝ) : ℝ → ℝ → Prop :=
λ x y, x + (a - 1) * y + a^2 - 1 = 0

-- Prove that when l1 is perpendicular to l2, then a = 2/3
theorem perp_lines (a : ℝ) : 
  (∃ x y, line1 a x y ∧ line2 a x y ∧ (a * 1 + 2 * (a - 1) = 0)) ↔ (a = 2 / 3) :=
sorry

-- Prove that when l1 is parallel to l2, then a = -1
theorem parallel_lines (a : ℝ) : 
  (∃ x y, line1 a x y ∧ line2 a x y ∧ (a / 1 = 2 / (a - 1) ∧ a ≠ 1)) ↔ (a = -1) :=
sorry

end perp_lines_parallel_lines_l687_687374


namespace range_a_l687_687079

open Real

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def f (a : ℝ) (x : ℝ) : ℝ := log a x

def q (a : ℝ) : Prop := ∀ x : ℝ, 0 < x → (∀ y : ℝ, 0 < y → (log a x < log a y ↔ x < y))

theorem range_a (a : ℝ) : (∃ (a : ℝ), (p a ∨ q a) ∧ ¬ (p a ∧ q a)) ↔ a ∈ Ioc (-2 : ℝ) 1 ∪ Icc 2 ∞ := 
by
  sorry

end range_a_l687_687079


namespace polygon_cyclic_iff_l687_687437

variables {n : ℕ} (A : Fin n → ℝ × ℝ) (dist : (ℝ × ℝ) → (ℝ × ℝ) → ℝ)
variables {b c : (Fin n) → ℝ}

-- Convex polygon condition
def convex_polygon (A : Fin n → ℝ × ℝ) : Prop := sorry

-- Cyclic polygon condition
def cyclic_polygon (A : Fin n → ℝ × ℝ) : Prop := sorry

-- Distance definition, given the coordinate pairs A_i and A_j
def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Proof statement
theorem polygon_cyclic_iff (h_convex : convex_polygon A) (hn : 4 ≤ n) :
    (cyclic_polygon A) ↔ (∀ (i j : Fin n), i.val < j.val → dist (A i) (A j) = b j * c i - b i * c j) := sorry

end polygon_cyclic_iff_l687_687437


namespace talent_show_girls_count_l687_687937

theorem talent_show_girls_count (B G : ℕ) (h1 : B + G = 34) (h2 : G = B + 22) : G = 28 :=
by
  sorry

end talent_show_girls_count_l687_687937


namespace range_of_a_l687_687768

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l687_687768


namespace total_cost_of_shoes_l687_687087

theorem total_cost_of_shoes 
  (cost_first_pair : ℕ)
  (percentage_increase : ℕ)
  (price_first : cost_first_pair = 22)
  (percentage_increase_eq : percentage_increase = 50) :
  let additional_cost := (percentage_increase * cost_first_pair) / 100
  let cost_second_pair := cost_first_pair + additional_cost
  let total_cost := cost_first_pair + cost_second_pair
  in total_cost = 55 :=
by
  sorry

end total_cost_of_shoes_l687_687087


namespace range_of_a_l687_687710

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l687_687710


namespace min_coin_flips_no_three_in_a_row_l687_687544

theorem min_coin_flips_no_three_in_a_row (flip_count : ℕ) : 
  ∃ (f : ℕ × ℕ → bool), 
    (∀ i j, i ≠ j → (f (i, j) ≠ tt) ∧ (f (i, j) ≠ ff)) →
    flip_count = 4 :=
by sorry

end min_coin_flips_no_three_in_a_row_l687_687544


namespace james_ate_eight_slices_l687_687049

-- Define the conditions
def num_pizzas := 2
def slices_per_pizza := 6
def fraction_james_ate := 2 / 3
def total_slices := num_pizzas * slices_per_pizza

-- Define the statement to prove
theorem james_ate_eight_slices : fraction_james_ate * total_slices = 8 :=
by
  sorry

end james_ate_eight_slices_l687_687049


namespace sum_max_elements_l687_687637

namespace MathProof

def S (j : ℕ) : set ℕ := {n : ℕ | n > 0 ∧ n ≤ j}

def maxElement (A : set ℕ) : ℕ := if h : A.nonempty then classical.some (finset.max' (set.finite.toFinset (set.finite_of_nonempty_of_finite h)) sorry) else 0

theorem sum_max_elements (j : ℕ) :
  (∑ A in (set.powerset (S j)).toFinset, maxElement A) = (j - 1) * 2^j + 1 :=
by
  sorry

end MathProof

end sum_max_elements_l687_687637


namespace range_of_a_l687_687752

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687752


namespace triangle_klm_angles_l687_687042

theorem triangle_klm_angles 
  (α β γ : ℝ)
  (h_eq : α + β + γ = 60) 
  (h_kab : γ = 15) 
  (h_mbc : α = 20) 
  (h_mac : β = 25)
  : ∃ (x y z : ℝ), x = 3 * α ∧ y = 3 * β ∧ z = 3 * γ ∧ x + y + z = 180 ∧ x = 60 ∧ y = 75 ∧ z = 45 :=
begin
  use [3 * α, 3 * β, 3 * γ],
  split, { refl },
  split, { refl },
  split, { refl },
  split,
  { linarith [h_eq], },
  split, { linarith [h_mbc], },
  split, { linarith [h_mac], },
  { linarith [h_kab] }
end

end triangle_klm_angles_l687_687042


namespace complex_real_number_condition_l687_687313

theorem complex_real_number_condition (a : ℝ) (h : ((-a : ℂ) + complex.I) / (1 - complex.I) ∈ ℝ) : a = 1 :=
by
  sorry

end complex_real_number_condition_l687_687313


namespace zeros_in_factorial_base9_l687_687008

theorem zeros_in_factorial_base9 (n : ℕ) (hn₁ : n = 15) : 
  (nat_trailing_zeros (15 !) 9) = 3 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 15! = 1307674368000 := by norm_num
  have h3 : ∑ k in finsup 15 (λ k : ℕ, ⌊ 15 / 3^k ⌋) = 6 := by norm_num
  have h4 : ∀ m, trailing_zero_bound (15!, 9) = m := by sorry
  sorry

end zeros_in_factorial_base9_l687_687008


namespace range_of_a_l687_687711

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l687_687711


namespace factor_expression_l687_687257

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687257


namespace range_of_a_l687_687743

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687743


namespace f_log2_12_plus_f_1_l687_687361

def f (x : ℝ) : ℝ := if x < 2 then log x (3 - x) else 2^(x - 2)

theorem f_log2_12_plus_f_1 : f (log 2 12) + f 1 = 4 :=
by
  -- No concrete proof provided; assuming steps are correct.
  sorry

end f_log2_12_plus_f_1_l687_687361


namespace carlos_marbles_l687_687621

theorem carlos_marbles:
  ∃ M, M > 1 ∧ 
       M % 5 = 1 ∧ 
       M % 7 = 1 ∧ 
       M % 11 = 1 ∧ 
       M % 4 = 2 ∧ 
       M = 386 := by
  sorry

end carlos_marbles_l687_687621


namespace clock_hands_angle_96_degrees_l687_687618

variables (t : ℝ) (angle : ℝ)
noncomputable def minute_hand_position (t : ℝ) : ℝ := 6 * t
noncomputable def hour_hand_position (t : ℝ) : ℝ := 210 + 0.5 * t

theorem clock_hands_angle_96_degrees :
  ∃ t₁ t₂ ∈ set.Icc (0:ℝ) 60,
  let θ_m₁ := minute_hand_position t₁,
      θ_h₁ := hour_hand_position t₁,
      θ_m₂ := minute_hand_position t₂,
      θ_h₂ := hour_hand_position t₂ in
  (abs (θ_m₁ - θ_h₁) = 96 ∨ abs (θ_m₁ - θ_h₁) = 360 - 96) ∧
  (abs (θ_m₂ - θ_h₂) = 96 ∨ abs (θ_m₂ - θ_h₂) = 360 - 96) :=
by
  sorry

end clock_hands_angle_96_degrees_l687_687618


namespace range_of_a_l687_687787

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687787


namespace value_of_y_l687_687560

theorem value_of_y :
  ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end value_of_y_l687_687560


namespace probability_two_blue_gumballs_l687_687426

theorem probability_two_blue_gumballs (p_pink : ℝ) (h_pink : p_pink = 0.33333333333333337) : 
  let p_blue := 1 - p_pink in
  p_blue * p_blue = 0.4444444444444445 :=
by
  sorry

end probability_two_blue_gumballs_l687_687426


namespace tan_alpha_in_4th_quadrant_l687_687334

theorem tan_alpha_in_4th_quadrant (α : ℝ) (h1 : 0 > α ∧ α > -π/2) 
  (h2 : cos(π/2 + α) = 4/5) : tan α = -4/3 :=
sorry

end tan_alpha_in_4th_quadrant_l687_687334


namespace sum_of_distinct_prime_divisors_1728_l687_687961

theorem sum_of_distinct_prime_divisors_1728 : 
  (2 + 3 = 5) :=
sorry

end sum_of_distinct_prime_divisors_1728_l687_687961


namespace compute_pqr_l687_687824

theorem compute_pqr
  (p q r : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (h_sum : p + q + r = 30)
  (h_eq : 1 / p + 1 / q + 1 / r + 240 / (p * q * r) = 1) :
  p * q * r = 1080 := by
  sorry

end compute_pqr_l687_687824


namespace james_ate_eight_slices_l687_687048

-- Define the conditions
def num_pizzas := 2
def slices_per_pizza := 6
def fraction_james_ate := 2 / 3
def total_slices := num_pizzas * slices_per_pizza

-- Define the statement to prove
theorem james_ate_eight_slices : fraction_james_ate * total_slices = 8 :=
by
  sorry

end james_ate_eight_slices_l687_687048


namespace true_discount_example_l687_687519

/-- Definition of true_discount is provided -/
def true_discount (A R T : ℚ) : ℚ := 
  (A * R * T) / (100 + (R * T))

/-- Prove the specific case of true_discount for given conditions -/
theorem true_discount_example :
  true_discount 1764 16 (¾) = 189 :=
by
  sorry

end true_discount_example_l687_687519


namespace sin_bound_l687_687985

theorem sin_bound (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < sin x ∧ sin x < x := 
sorry

end sin_bound_l687_687985


namespace jellybean_mass_l687_687604

noncomputable def cost_per_gram : ℚ := 7.50 / 250
noncomputable def mass_for_180_cents : ℚ := 1.80 / cost_per_gram

theorem jellybean_mass :
  mass_for_180_cents = 60 := 
  sorry

end jellybean_mass_l687_687604


namespace real_roots_of_quadratic_l687_687393

theorem real_roots_of_quadratic (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ m ≤ 4 / 3 :=
by
  sorry

end real_roots_of_quadratic_l687_687393


namespace line_through_midpoint_l687_687385

theorem line_through_midpoint (x y : ℝ) (P : x = 2 ∧ y = -1) :
  (∃ l : ℝ, ∀ t : ℝ, 
  (1 + 5 * Real.cos t = x) ∧ (5 * Real.sin t = y) →
  (x - y = 3)) :=
by
  sorry

end line_through_midpoint_l687_687385


namespace find_regular_rate_r_equals_18_l687_687182

noncomputable def regular_rate (R : ℝ) : ℝ :=
  let regular_hours := 40
  let overtime_hours := 8.12698412698413
  let overtime_rate := 1.75 * R
  let total_hours := regular_hours + overtime_hours
  let total_compensation := 976
  total_compensation = (regular_hours * R) + (overtime_hours * overtime_rate)

theorem find_regular_rate_r_equals_18 :
  ∃ (R : ℝ), regular_rate R ∧ abs (R - 18) < 1e-6 :=
by
  sorry

end find_regular_rate_r_equals_18_l687_687182


namespace max_min_values_of_function_l687_687293

theorem max_min_values_of_function :
  let f := (fun x : ℝ => 3 * x^4 + 4 * x^3 + 34)
  ∃ (max min : ℝ), (∀ x ∈ Icc (-2 : ℝ) 1, f x ≤ max) ∧ (∀ x ∈ Icc (-2 : ℝ) 1, min ≤ f x) ∧
                   max = f (-2) ∧ max = 50 ∧
                   min = f (-1) ∧ min = 33 :=
by
  let f := (fun x : ℝ => 3 * x^4 + 4 * x^3 + 34)
  use 50, 33
  have h₁ : ∀ x, f' x = 12 * x^2 * (x + 1), from sorry,
  have critical_points : {x | f' x = 0} = {0, -1}, from sorry,
  -- Check values at endpoints and critical points
  have h_f_neg2 : f (-2) = 50 := by simp [f],
  have h_f_1 : f 1 = 41 := by simp [f],
  have h_f_neg1 : f (-1) = 33 := by simp [f],
  have h_f_0 : f 0 = 34 := by simp [f],
  split,
  -- Proving max value
  { intro x,
    intro hx,
    by_cases hx0 : x = -2 ∨ x = 1 ∨ x = -1 ∨ x = 0,
    any_goals { finish },
    -- f(-2) = 50, rest points have lower values
    show f x ≤ 50, from sorry,},
  -- Proving min value
  split,
  { intro x,
    intro hx,
    by_cases hx0 : x = -2 ∨ x = 1 ∨ x = -1 ∨ x = 0,
    any_goals { finish },
    -- f(-1) = 33, rest points have higher values
    show 33 ≤ f x, from sorry,},
  -- Verifying calculated max and min points are as expected
  repeat {split}; assumption
  sorry

end max_min_values_of_function_l687_687293


namespace a_plus_b_minus_c_in_S_l687_687685

-- Define the sets P, Q, and S
def P := {x : ℤ | ∃ k : ℤ, x = 3 * k}
def Q := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def S := {x : ℤ | ∃ k : ℤ, x = 3 * k - 1}

-- Define the elements a, b, and c as members of sets P, Q, and S respectively
variables (a b c : ℤ)
variable (ha : a ∈ P) -- a ∈ P
variable (hb : b ∈ Q) -- b ∈ Q
variable (hc : c ∈ S) -- c ∈ S

-- Theorem statement proving the question
theorem a_plus_b_minus_c_in_S : a + b - c ∈ S := sorry

end a_plus_b_minus_c_in_S_l687_687685


namespace diagonals_bisect_each_other_l687_687493

theorem diagonals_bisect_each_other 
  (ABCD : Type) [convex_quadrilateral ABCD]
  (O : Point)     
  (intersects_diagonals : diagonals_intersect_at O) 
  (area_condition : ∀ A B C D, 
    area (triangle O A B) + area (triangle O C D) = area (triangle O B C) + area (triangle O D A)): 
  ∃ AC BD, bisects AC BD :=
by
  sorry

end diagonals_bisect_each_other_l687_687493


namespace chickens_in_farm_l687_687614

theorem chickens_in_farm (c b : ℕ) (h1 : c + b = 9) (h2 : 2 * c + 4 * b = 26) : c = 5 := by sorry

end chickens_in_farm_l687_687614


namespace distance_increase_between_blocks_l687_687146

theorem distance_increase_between_blocks
  (m : ℝ) -- mass of each block
  (Π : ℝ) -- potential energy of the compressed spring
  (μ : ℝ) -- coefficient of friction
  (g : ℝ) -- acceleration due to gravity
  (h_g_pos : 0 < g) (h_m_pos : 0 < m) (h_Π_pos : 0 < Π) (h_μ_pos : 0 < μ) :
  -- The increase in the distance between the blocks is:
  let L := Π / (μ * m * g) in
  L = Π / (μ * m * g) :=
by
  -- Add a sorry to indicate the proof is omitted.
  sorry

end distance_increase_between_blocks_l687_687146


namespace multiple_of_four_of_sum_zero_l687_687827

theorem multiple_of_four_of_sum_zero (n : ℕ) (a : ℕ → ℤ) 
  (h1 : ∀ i, a i = 1 ∨ a i = -1)
  (h2 : (∑ i in Finset.range n, a i * a ((i + 1) % n) ) = 0) : 
  n % 4 = 0 := 
sorry

end multiple_of_four_of_sum_zero_l687_687827


namespace num_true_statements_l687_687644

-- Definitions for each statement as propositions
def statement1 (l1 l2 : Line) (p : Plane) : Prop :=
  (Parallel l1 p ∧ Parallel l2 p) → Parallel l1 l2

def statement2 (p1 p2 p3 : Plane) : Prop :=
  (Parallel p1 p3 ∧ Parallel p2 p3) → Parallel p1 p2

def statement3 (l1 l2 : Line) (p : Plane) : Prop :=
  (Perpendicular l1 p ∧ Perpendicular l2 p) → Parallel l1 l2

def statement4 (p1 p2 p3 : Plane) : Prop :=
  (Perpendicular p1 p3 ∧ Perpendicular p2 p3) → Perpendicular p1 p2

-- Theorem stating the number of true statements is 2
theorem num_true_statements (l1 l2 : Line) (p1 p2 p3 : Plane) :
  (¬statement1 l1 l2 p1) →
  (statement2 p1 p2 p3) →
  (statement3 l1 l2 p1) →
  (¬statement4 p1 p2 p3) →
  (2 : ℕ) :=
by
  intros h1 h2 h3 h4
  sorry

end num_true_statements_l687_687644


namespace intersecting_points_in_circle_l687_687097

theorem intersecting_points_in_circle :
  let polygons := [4, 6, 8, 9]
  in ∀ (is_inscribed : ∀ n ∈ polygons, ∃ (polygon : finset point), -- Here we assume point is a predefined type representing points on plane
                                             (is_regular_polygon polygon n ∧
                                              ∀ p ∈ polygon, p ∈ unit_circle)), -- we assume polygon vertices are in the unit circle
     no_shared_vertex : (∀ p ∈ polygons, ∀ q ∈ polygons, p ≠ q → only_intersects_at p q set.empty),
     no_three_sides_intersect : (∀ p ∈ polygons, ∀ q ∈ polygons, ∀ r ∈ polygons, p ≠ q → q ≠ r → p ≠ r → sides_intersect_at_most_twice p q r),
     (∑ p1 p2, p1 < p2 → (∀ (side1 : p1.sides) (side2 : p2.sides), intersects_inside_circle side1 side2)) = 64 :=
sorry

end intersecting_points_in_circle_l687_687097


namespace how_many_candies_eaten_l687_687246

variable (candies_tuesday candies_thursday candies_friday candies_left : ℕ)

def total_candies (candies_tuesday candies_thursday candies_friday : ℕ) : ℕ :=
  candies_tuesday + candies_thursday + candies_friday

theorem how_many_candies_eaten (h_tuesday : candies_tuesday = 3)
                               (h_thursday : candies_thursday = 5)
                               (h_friday : candies_friday = 2)
                               (h_left : candies_left = 4) :
  (total_candies candies_tuesday candies_thursday candies_friday) - candies_left = 6 :=
by
  sorry

end how_many_candies_eaten_l687_687246


namespace y_intercept_of_tangent_line_l687_687676

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem y_intercept_of_tangent_line (a : ℝ) : 
  let f' x := (a - 1 / x)
  let tangent_line := λ x y : ℝ, y - f a 1 = f' 1 * (x - 1)
  (tangent_line 0 _) = 1 :=
by
  let f' x := (a - 1 / x)
  let tangent_line := λ x y : ℝ, y - f a 1 = f' 1 * (x - 1)
  have : tangent_line 0 (tangent_line 0 1) = 1 := sorry
  exact this

end y_intercept_of_tangent_line_l687_687676


namespace intersection_S_T_l687_687333

def S : Set ℝ := {x | x > -2}

def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_S_T : S ∩ T = {x | -2 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_S_T_l687_687333


namespace arithmetic_sequence_general_term_l687_687330

theorem arithmetic_sequence_general_term:
  ∃ (a : ℕ → ℕ), 
    (∀ n, a n + 1 > a n) ∧
    (a 1 = 2) ∧ 
    ((a 2) ^ 2 = a 5 + 6) ∧ 
    (∀ n, a n = 2 * n) :=
by
  sorry

end arithmetic_sequence_general_term_l687_687330


namespace range_of_a_l687_687696

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687696


namespace cost_of_pumpkin_seeds_l687_687191

theorem cost_of_pumpkin_seeds (P : ℝ)
    (h1 : ∃(P_tomato P_chili : ℝ), P_tomato = 1.5 ∧ P_chili = 0.9) 
    (h2 : 3 * P + 4 * 1.5 + 5 * 0.9 = 18) 
    : P = 2.5 :=
by sorry

end cost_of_pumpkin_seeds_l687_687191


namespace range_of_a_l687_687703

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687703


namespace grid_Y_value_l687_687217

/-- The specific $4 \times 4$ grid problem where we need to find the center top-left value $Y$ -/
theorem grid_Y_value 
  (arithmetic_seq : ∀ (a b : ℕ), Prop) 
  (first_row : ∀ (n : ℕ) (a d : ℝ), ∀ t ∈ finset.range 4, (first_row n a d t) := λ n a d t, a + t * d)
  (fourth_row : ∀ (n : ℕ) (a d : ℝ), ∀ t ∈ finset.range 4, (fourth_row n a d t) := λ n a d t, a + t * d)
  (grid_structure : ∀ (i j : ℕ), grid_structure i j := first_row ∨ fourth_row)
  (y_pos : grid_structure 1 1 = first_row) 
  : (14.33 : ℝ) :=
sorry

end grid_Y_value_l687_687217


namespace tetrahedrons_formed_l687_687375

def triangle_eq : Triangle ->
  (Triangle.sides = [3, 4, 5]
  ∨ Triangle.sides = [4, 5, sqrt 41]
  ∨ Triangle.sides = [5 / 6 * sqrt 2, 4, 5]) :=
sorry

def tetrahedron_eq : Nat :=
1

theorem tetrahedrons_formed :
  ∃ t : Tetrahedron, (∀ f ∈ t.faces, triangle_eq f) ∧ count t = tetrahedron_eq :=
sorry

end tetrahedrons_formed_l687_687375


namespace shirt_price_correct_l687_687516

noncomputable def sweater_price := 43.885
noncomputable def shirt_price := 36.455
noncomputable def total_cost := 80.34
noncomputable def price_difference := 7.43

theorem shirt_price_correct :
  (shirt_price + sweater_price = total_cost) ∧ (sweater_price - shirt_price = price_difference) →
  shirt_price = 36.455 :=
by {
  intros h,
  sorry
}

end shirt_price_correct_l687_687516


namespace angle_remains_acute_l687_687198

noncomputable theory

structure Quadrilateral (α : Type*) [InnerProductSpace ℝ α] :=
(A B C D : α)
(midpoints_acute_angle : ∀ (P Q R : α), 
  (P = (A + B) / 2 ∧ Q = (B + C) / 2 ∧ R = (C + D) / 2) → 
  inner (Q - P) (R - Q) > 0)

theorem angle_remains_acute {α : Type*} [InnerProductSpace ℝ α] (quad : Quadrilateral α) :
  ∀ (P Q R : α), (P = (quad.A + quad.B) / 2 ∧ Q = (quad.B + quad.C) / 2 ∧ R = (quad.C + quad.D) / 2) →
  inner (Q - P) (R - Q) > 0 :=
sorry

end angle_remains_acute_l687_687198


namespace open_sets_l687_687643

-- Given definitions
def is_open (A : set (ℝ × ℝ)) : Prop :=
  ∀ (x₀ y₀ : ℝ), (x₀, y₀) ∈ A → ∃ r > 0, ∀ (x y : ℝ), sqrt ((x - x₀)^2 + (y - y₀)^2) < r → (x, y) ∈ A

-- Set ③
def set3 : set (ℝ × ℝ) := {p | |p.1| + |p.2| < 1}

-- Set ④
def set4 : set (ℝ × ℝ) := {p | 0 < p.1^2 + (p.2 - 1)^2 ∧ p.1^2 + (p.2 - 1)^2 < 1}

-- Theorem to prove
theorem open_sets : is_open set3 ∧ is_open set4 := by
  sorry

end open_sets_l687_687643


namespace part1_l687_687584

theorem part1 : log 2.5 6.25 + log 10 0.01 + log (real.sqrt e) - real.pow 2 (1 + log 2 3) = -11 / 2 :=
by
  sorry

end part1_l687_687584


namespace f_neg2_f_3_l687_687077

def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x + 4 else 10 - 3 * x

theorem f_neg2 : f (-2) = 0 := 
by 
  unfold f
  split_ifs
  · simp
  · sorry

theorem f_3 : f 3 = 1 :=
by 
  unfold f
  split_ifs
  · sorry
  · simp

end f_neg2_f_3_l687_687077


namespace negation_of_universal_sin_l687_687128

theorem negation_of_universal_sin (h : ∀ x : ℝ, Real.sin x > 0) : ∃ x : ℝ, Real.sin x ≤ 0 :=
sorry

end negation_of_universal_sin_l687_687128


namespace cost_per_quart_proof_l687_687471

-- Definitions of costs and quantities
def cost_of_eggplants (ep_pounds : ℕ) (ep_per_pound : ℝ) := ep_pounds * ep_per_pound
def cost_of_zucchini (zu_pounds : ℕ) (zu_per_pound : ℝ) := zu_pounds * zu_per_pound
def cost_of_tomatoes (to_pounds : ℕ) (to_per_pound : ℝ) := to_pounds * to_per_pound
def cost_of_onions (on_pounds : ℕ) (on_per_pound : ℝ) := on_pounds * on_per_pound
def cost_of_basil (ba_pounds : ℕ) (ba_per_half_pound : ℝ) := ba_pounds * (ba_per_half_pound / 0.5)

-- Total cost of vegetables
def total_cost :=
  cost_of_eggplants 5 2.0 +
  cost_of_zucchini 4 2.0 +
  cost_of_tomatoes 4 3.5 +
  cost_of_onions 3 1.0 +
  cost_of_basil 1 2.5

-- Number of quarts of ratatouille
def number_of_quarts := 4

-- Cost per quart calculation
def cost_per_quart := total_cost / number_of_quarts

-- The proof statement
theorem cost_per_quart_proof : cost_per_quart = 10.0 :=
by
  unfold cost_per_quart total_cost cost_of_eggplants cost_of_zucchini cost_of_tomatoes cost_of_onions cost_of_basil
  norm_num
  sorry

end cost_per_quart_proof_l687_687471


namespace necessary_and_sufficient_condition_l687_687018

def A : set ℝ := { x | x / (x - 1) ≤ 0 }
def B : set ℝ := { x | x ^ 2 < 2 * x }

theorem necessary_and_sufficient_condition (x : ℝ) :
  (x ∈ A ∩ B) ↔ (x ∈ Ioo 0 1) :=
sorry

end necessary_and_sufficient_condition_l687_687018


namespace part1_part2_l687_687989

-- Part (1)
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 1) : (x - x^2 < Real.sin x) ∧ (Real.sin x < x) :=
sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = Real.cos (a * x) - Real.log (1 - x^2)) ∧ 
  (f' 0 = 0) ∧ (∃ x_max : ℝ, (f'' x_max < 0) ∧ (x_max = 0))) : a < -Real.sqrt 2 ∨ Real.sqrt 2 < a :=
sorry

end part1_part2_l687_687989


namespace vector_properties_l687_687376

variables (a b : ℝ^3)

def a := (1, 1, 1: ℝ)
def b := (-1, 0, 2: ℝ)

theorem vector_properties :
  |a| = sqrt 3 ∧
  (a / |a|) = (sqrt 3 / 3, sqrt 3 / 3, sqrt 3 / 3) ∧
  (a ⋅ b) ≠ -1 ∧
  (a ⋅ b) / (|a| * |b|) = sqrt 15 / 15 :=
by
  sorry

end vector_properties_l687_687376


namespace circle_equation_unique_l687_687591

theorem circle_equation_unique 
    (C : ℝ × ℝ → Prop)
    (P : ℝ × ℝ) (Q : ℝ × ℝ)
    (P_on_C : C P)
    (Q_on_C : C Q)
    (chord_lengths_equal : ∀ x y, C (x, 0) ↔ C (0, x)) :
  (C = λ z, (z.1 + 2)^2 + (z.2 + 2)^2 = 25) ∨ 
  (C = λ z, (z.1 + 1)^2 + (z.2 - 1)^2 = 5) := 
by
  -- Placeholder for the actual proof
  sorry

end circle_equation_unique_l687_687591


namespace marble_probability_l687_687458

theorem marble_probability :
  let p_other := 0.4 in
  let draws := 5 in
  (\Sigma i in range draws, p_other) = 2 :=
by
  sorry

end marble_probability_l687_687458


namespace g_is_odd_l687_687044

def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_is_odd_l687_687044


namespace cricket_average_increase_l687_687188

theorem cricket_average_increase (initial_innings : ℕ) (initial_average : ℕ) (next_innings_runs : ℕ) :
  initial_innings = 12 →
  initial_average = 48 →
  next_innings_runs = 178 →
  ((initial_average * initial_innings + next_innings_runs) / (initial_innings + 1) - initial_average) = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end cricket_average_increase_l687_687188


namespace part1_part2_l687_687996

-- Proof that for 0 < x < 1, x - x^2 < sin x < x
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := 
sorry

-- Proof that if x = 0 is a local maximum of f(x) = cos(ax) - ln(1 - x^2), then a is in the specified range.
theorem part2 (a : ℝ) (h : ∀ x, (cos(a * x) - log(1 - x^2))' (0) = 0 ∧ (cos(a * x) - log(1 - x^2))'' (0) < 0) : 
  a ∈ Set.Ioo (-∞) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) (∞) := 
sorry

end part1_part2_l687_687996


namespace coin_collection_problem_l687_687970

theorem coin_collection_problem (n : ℕ) 
  (quarters : ℕ := n / 2)
  (half_dollars : ℕ := 2 * (n / 2))
  (value_nickels : ℝ := 0.05 * n)
  (value_quarters : ℝ := 0.25 * (n / 2))
  (value_half_dollars : ℝ := 0.5 * (2 * (n / 2)))
  (total_value : ℝ := value_nickels + value_quarters + value_half_dollars) :
  total_value = 67.5 ∨ total_value = 135 :=
sorry

end coin_collection_problem_l687_687970


namespace problem_min_value_problem_inequality_range_l687_687339

theorem problem_min_value (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) :
  (1 / a + 4 / b) ≥ 9 :=
sorry

theorem problem_inequality_range (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) (x : ℝ) :
  (1 / a + 4 / b) ≥ |2 * x - 1| - |x + 1| ↔ -7 ≤ x ∧ x ≤ 11 :=
sorry

end problem_min_value_problem_inequality_range_l687_687339


namespace compute_f_f_3_l687_687362

-- Definition of the function f
def f (x : ℝ) : ℝ :=
  if x >= 1 then 2 - x else x^2

-- Proof statement: f(f(3)) = 1
theorem compute_f_f_3 : f (f 3) = 1 :=
by
  -- To be completed in the proof
  sorry

end compute_f_f_3_l687_687362


namespace h_h_3_eq_2915_l687_687876

def h (x : ℕ) : ℕ := 3 * x^2 + x + 1

theorem h_h_3_eq_2915 : h (h 3) = 2915 := by
  sorry

end h_h_3_eq_2915_l687_687876


namespace part1_part2_l687_687988

-- Part (1)
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 1) : (x - x^2 < Real.sin x) ∧ (Real.sin x < x) :=
sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = Real.cos (a * x) - Real.log (1 - x^2)) ∧ 
  (f' 0 = 0) ∧ (∃ x_max : ℝ, (f'' x_max < 0) ∧ (x_max = 0))) : a < -Real.sqrt 2 ∨ Real.sqrt 2 < a :=
sorry

end part1_part2_l687_687988


namespace complete_the_square_l687_687640

theorem complete_the_square (x : ℝ) : (x^2 - 8 * x + 10 = 0) → (x - 4)^2 = 6 :=
by 
  intro h,
  sorry

end complete_the_square_l687_687640


namespace sum_geometric_series_2_sum_geometric_series_3_sum_geometric_series_a_l687_687096

theorem sum_geometric_series_2 : (1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7 + 2^8 + 2^9 = 2^10 - 1) := 
by sorry

theorem sum_geometric_series_3 : (3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9 + 3^{10} = (3^{11} - 3) / 2) :=
by sorry

theorem sum_geometric_series_a (a : ℕ) (n : ℕ) (h_pos_a : 0 < a) (h_neq : a ≠ 1) : 
  (1 + a + a^2 + a^3 + a^4 + a^5 + a^6 + a^7 + a^8 + a^9 + a^n = (a^(n+1) - 1) / (a - 1)) := 
by sorry

end sum_geometric_series_2_sum_geometric_series_3_sum_geometric_series_a_l687_687096


namespace range_of_a_for_decreasing_function_l687_687803

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x + 4 else 3 * a / x

theorem range_of_a_for_decreasing_function :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≥ f a x2) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end range_of_a_for_decreasing_function_l687_687803


namespace volume_of_bounded_convex_body_l687_687545

/-- We are given a unit cube. 
Each plane forms a 45 degree angle with adjacent faces and does not intersect the cube. 
We want to show that the volume of the convex body bounded by these planes is 2. -/
theorem volume_of_bounded_convex_body : 
  ∃ V : ℝ, 
    (∀ (cube : set ℝ^3) (is_unit_cube : is_unit_cube cube) 
       (planes : set (set ℝ^3)) 
       (planes_conditions : planes_are_at_45_deg_to_adj_faces planes cube) 
       (do_not_intersect_cube : do_not_intersect_planes planes cube), 
       V = volume_of_bounded_body cube planes) 
    ∧ V = 2 := 
by
  sorry

end volume_of_bounded_convex_body_l687_687545


namespace range_of_a_l687_687751

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687751


namespace possible_measures_for_angle_A_l687_687929

-- Definition of angles A and B, and their relationship
def is_supplementary_angles (A B : ℕ) : Prop := A + B = 180

def is_multiple_of (A B : ℕ) : Prop := ∃ k : ℕ, k ≥ 1 ∧ A = k * B

-- Prove there are 17 possible measures for angle A.
theorem possible_measures_for_angle_A : 
  (∀ (A B : ℕ), (A > 0) ∧ (B > 0) ∧ is_multiple_of A B ∧ is_supplementary_angles A B → 
  A = B * 17) := 
sorry

end possible_measures_for_angle_A_l687_687929


namespace range_of_a_l687_687750

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687750


namespace evaluate_polynomial_given_condition_l687_687651

theorem evaluate_polynomial_given_condition :
  ∀ x : ℝ, x > 0 → x^2 - 2 * x - 8 = 0 → (x^3 - 2 * x^2 - 8 * x + 4 = 4) := 
by
  intro x hx hcond
  sorry

end evaluate_polynomial_given_condition_l687_687651


namespace multiples_of_41_l687_687534

theorem multiples_of_41 {a : ℕ → ℤ} (h : ∀ k : ℕ, (∑ i in finset.range 41, a ((k + i) % 1000) ^ 2) % (41^2) = 0) : ∀ n, a n % 41 = 0 :=
  sorry

end multiples_of_41_l687_687534


namespace problem_statement_equality_condition_l687_687974

theorem problem_statement (x y z : ℝ) (hx : 0 <= x) (hy : 0 <= y) (hz : 0 <= z) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) >= 2 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : 0 <= x) (hy : 0 <= y) (hz : 0 <= z) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) = 2 ↔ x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end problem_statement_equality_condition_l687_687974


namespace painting_price_after_5_years_l687_687931

variable (P : ℝ)
-- Conditions on price changes over the years
def year1_price (P : ℝ) := P * 1.30
def year2_price (P : ℝ) := year1_price P * 0.80
def year3_price (P : ℝ) := year2_price P * 1.25
def year4_price (P : ℝ) := year3_price P * 0.90
def year5_price (P : ℝ) := year4_price P * 1.15

theorem painting_price_after_5_years (P : ℝ) :
  year5_price P = 1.3455 * P := by
  sorry

end painting_price_after_5_years_l687_687931


namespace scientific_notation_240000_l687_687920

theorem scientific_notation_240000 :
  240000 = 2.4 * 10^5 :=
by
  sorry

end scientific_notation_240000_l687_687920


namespace round_robin_cycles_l687_687200

theorem round_robin_cycles (n teams : Finset ℕ) (games_won : ℕ → ℕ) (plays_every_other : ∀ t ∈ teams, ∀ t' ∈ teams, t ≠ t' → played_once t t')
  (wins_losses : ∀ t ∈ teams, games_won t = 12 ∧ (24 - games_won t) = 12) : 
  ∃ sets_of_three : Finset (Finset ℕ), |sets_of_three| = 650 ∧ 
    ∀ s ∈ sets_of_three, s.card = 3 ∧ ∃ a b c ∈ s, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (wins a b ∧ wins b c ∧ wins c a) :=
sorry

end round_robin_cycles_l687_687200


namespace mutually_exclusive_complementary_l687_687192

noncomputable def group : (ℕ × ℕ) := (3, 2)

def selection (n : ℕ × ℕ): set (ℕ × ℕ) :=
  {x | x.1 + x.2 = 2 ∧ x.1 ≤ n.1 ∧ x.2 ≤ n.2}

theorem mutually_exclusive_complementary (n : ℕ × ℕ) (h : n = group) :
  let E_1 := {x : ℕ × ℕ | x.2 ≥ 1 ∧ x.1 + x.2 = 2}
  let E_2 := {x : ℕ × ℕ | x.1 = 2 ∧ x.1 + x.2 = 2} in
  (E_1 ∩ E_2 = ∅) ∧ ((E_1 ∪ E_2) = selection n) :=
by
  sorry

end mutually_exclusive_complementary_l687_687192


namespace find_y_l687_687011

theorem find_y (x y : ℝ) (h₁ : x = 51) (h₂ : x^3 * y - 2 * x^2 * y + x * y = 51000) : y = 2 / 5 := by
  sorry

end find_y_l687_687011


namespace geometric_seq_sum_a5_a7_l687_687395

noncomputable def geometric_seq (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (q ^ (n - 1))

theorem geometric_seq_sum_a5_a7 :
  ∀ (a1 q : ℝ),
    geometric_seq a1 q 2 + geometric_seq a1 q 4 = 20 →
    geometric_seq a1 q 3 + geometric_seq a1 q 5 = 40 →
    geometric_seq a1 q 5 + geometric_seq a1 q 7 = 160 :=
by
  intros a1 q h1 h2
  sorry

#eval geometric_seq_sum_a5_a7 2 2

end geometric_seq_sum_a5_a7_l687_687395


namespace find_k_l687_687379

def vector_a := (2, 1 : ℝ × ℝ)
def vector_b (k : ℝ) := (-1, k : ℝ)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) (h : dot_product vector_a (2 • vector_a - vector_b k) = 0) : k = 12 := by
  sorry

end find_k_l687_687379


namespace range_of_a_l687_687784

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687784


namespace midpoint_coords_l687_687347

noncomputable def F1 : (ℝ × ℝ) := (-2 * Real.sqrt 2, 0)
noncomputable def F2 : (ℝ × ℝ) := (2 * Real.sqrt 2, 0)
def major_axis_length : ℝ := 6
def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  let a := 3
  let b := 1
  (x^2) / (a^2) + y^2 / (b^2) = 1

theorem midpoint_coords :
  ∃ (A B : ℝ × ℝ), ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧ line_eq A.1 A.2 ∧ line_eq B.1 B.2 →
  (A.1 + B.1) / 2 = -9 / 5 ∧ (A.2 + B.2) / 2 = 1 / 5 :=
by
  sorry

end midpoint_coords_l687_687347


namespace count_ordered_triples_l687_687448

def S := { n : ℕ | 1 ≤ n ∧ n ≤ 25 }

def succ (a b : ℕ) : Prop := (0 < a - b ∧ a - b ≤ 12) ∨ (b - a > 12)

theorem count_ordered_triples :
  (finset.filter (λ (xyz : ℕ × ℕ × ℕ),
    let (x, y, z) := xyz in
    x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    succ x y ∧ succ y z ∧ succ z x) 
  ({ x : ℕ × ℕ × ℕ | true }.to_finset)).card = 1950 :=
sorry

end count_ordered_triples_l687_687448


namespace repeating_decimal_to_fraction_l687_687956

theorem repeating_decimal_to_fraction : 
  (let x := 0.2 * (10:ℚ)^(0) + 13 * (10:ℚ)^(-2) * (1/(1-(10:ℚ)^(-2)))
  in x = 523 / 2475) :=
by
  sorry

end repeating_decimal_to_fraction_l687_687956


namespace union_A_B_range_of_a_l687_687371

-- Define sets A and B
def A : set ℝ := { x | -4 ≤ x ∧ x ≤ 0 }
def B : set ℝ := { x | x > -2 }

-- Define C and condition C ∩ A = C implies C ⊆ A
def C (a : ℝ) : set ℝ := { x | a < x ∧ x < a + 1 }

-- Prove that A ∪ B = { x | x ≥ -4 }
theorem union_A_B : A ∪ B = { x : ℝ | x ≥ -4 } :=
by sorry

-- Prove that a ∈ [-4, -1] given C ∩ A = C
theorem range_of_a (a : ℝ) (h : C a ∩ A = C a) : a ∈ Icc (-4) (-1) :=
by sorry

end union_A_B_range_of_a_l687_687371


namespace min_Re_z1_z2_min_expression_l687_687170

open Complex

variables {z₁ z₂ : ℂ}

def conditions (z₁ z₂ : ℂ) : Prop :=
  (Re z₁ > 0) ∧ (Re z₂ > 0) ∧ (Re (z₁^2) = 2) ∧ (Re (z₂^2) = 2)

theorem min_Re_z1_z2 : ∃ m : ℝ, (∀ (z₁ z₂ : ℂ), conditions z₁ z₂ → Re (z₁ * z₂) ≥ m) ∧ 
                                                     (∀ (z₁ z₂ : ℂ), conditions z₁ z₂ → m = 0) :=
by {
  -- Proof to be provided
  sorry
}

theorem min_expression : ∃ n : ℝ, (∀ (z₁ z₂ : ℂ), conditions z₁ z₂ → 
                                          (Complex.abs (z₁ + 2) + Complex.abs (conj z₂ + 2) - Complex.abs (conj z₁ - z₂) ≥ n)) ∧ 
                                                  (∀ (z₁ z₂ : ℂ), conditions z₁ z₂ → n = 0) :=
by {
  -- Proof to be provided
  sorry
}

end min_Re_z1_z2_min_expression_l687_687170


namespace cricket_team_right_handed_players_l687_687978

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (non_throwers : ℕ := total_players - throwers)
  (left_handed_non_throwers : ℕ := non_throwers / 3)
  (right_handed_throwers : ℕ := throwers)
  (right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers)
  (total_right_handed : ℕ := right_handed_throwers + right_handed_non_throwers)
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : left_handed_non_throwers = non_throwers / 3) :
  total_right_handed = 59 :=
by
  rw [h1, h2] at *
  -- The remaining parts of the proof here are omitted for brevity.
  sorry

end cricket_team_right_handed_players_l687_687978


namespace sum_inequality_proof_l687_687076

theorem sum_inequality_proof (p n : ℕ) 
  (C : Π (h k : ℕ), 0 ≤ h → h ≤ n → 0 ≤ k → k ≤ p * h → ℝ) 
  (hC : ∀ (h k : ℕ) (Hh₁ : 0 ≤ h) (Hh₂ : h ≤ n) (Hk₁ : 0 ≤ k) (Hk₂ : k ≤ p * h), 0 ≤ C h k ∧ C h k ≤ 1) :
  (∑ h in finset.range n.succ, ∑ k in finset.range (p * h.succ), (C h.succ k.val (nat.zero_le _) (nat.succ_le_succ h.prop) (nat.zero_le _) k.is_lt) / h.succ) ^ 2
  ≤ 2 * p * (∑ h in finset.range n.succ, ∑ k in finset.range (p * h.succ), C h.succ k.val (nat.zero_le _) (nat.succ_le_succ h.prop) (nat.zero_le _) k.is_lt) := 
sorry

end sum_inequality_proof_l687_687076


namespace work_completion_time_l687_687163

-- Define the constants for work rates and times
def W : ℚ := 1
def P_rate : ℚ := W / 20
def Q_rate : ℚ := W / 12
def initial_days : ℚ := 4

-- Define the amount of work done by P in the initial 4 days
def work_done_initial : ℚ := initial_days * P_rate

-- Define the remaining work after initial 4 days
def remaining_work : ℚ := W - work_done_initial

-- Define the combined work rate of P and Q
def combined_rate : ℚ := P_rate + Q_rate

-- Define the time taken to complete the remaining work
def remaining_days : ℚ := remaining_work / combined_rate

-- Define the total time taken to complete the work
def total_days : ℚ := initial_days + remaining_days

-- The theorem to prove
theorem work_completion_time :
  total_days = 10 := 
by
  -- these term can be the calculation steps
  sorry

end work_completion_time_l687_687163


namespace range_of_a_l687_687704

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687704


namespace find_z_value_l687_687484

theorem find_z_value (k : ℝ) (hy5 : ∀ y z, y^5 * real.root z 5 = k) 
  (h_initial : hy5 3 8) : 
  real.root (find_z_value k 6) 5 = 1 / 16 :=
by
  have h_3_8 : 3^5 * real.root 8 5 = k := h_initial
  have k_value : k = 486 := by
    sorry
  have h_6_z : 6^5 * real.root (find_z_value k 6) 5 = 486 := hy5 6 (find_z_value k 6)
  have value_root : real.root (find_z_value k 6) 5 = 1 / 16 := by
    sorry
  exact value_root

end find_z_value_l687_687484


namespace factor_expression_l687_687256

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687256


namespace factor_expression_l687_687260

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687260


namespace route_evaluation_l687_687946

section RouteEvaluation

-- Conditions based on the table and provided data
def male_total : ℕ := 120
def female_total : ℕ := 180
def total : ℕ := 300

def male_good_A : ℕ := 30
def male_average_A : ℕ := 20
def male_good_B : ℕ := 55
def male_average_B : ℕ := 15

def female_good_A : ℕ := 70
def female_average_A : ℕ := 30
def female_good_B : ℕ := 20
def female_average_B : ℕ := 60

def total_good_A : ℕ := male_good_A + female_good_A -- 100
def total_average_A : ℕ := male_average_A + female_average_A -- 50
def total_good_B : ℕ := male_good_B + female_good_B -- 75
def total_average_B : ℕ := male_average_B + female_average_B -- 75

def table_correct : Prop :=
  (male_good_A + male_average_A + male_good_B + male_average_B = male_total) ∧
  (female_good_A + female_average_A + female_good_B + female_average_B = female_total) ∧
  (total_good_A + total_average_A + total_good_B + total_average_B = total)

-- Given chi-square formula 
def χ_squared (a b c d n : ℕ) : ℚ :=
  (n * (a*d - b*c)^2) / ((a+b)*(c+d)*(a+c)*(b+d))

def independence_test (a b c d : ℕ) : Prop :=
  (χ_squared a b c d total > 10.828)

noncomputable def expected_value (good_avg : ℕ → ℚ) (scores : list (ℕ × ℚ)) : ℚ :=
  scores.foldl (λ acc xy, acc + (xy.1 * xy.2)) 0 / good_avg

-- Determine the route choice based on expected value
def preferred_route (exp_A exp_B : ℚ) : Prop :=
  (exp_A > exp_B)

-- Lean proof statement
theorem route_evaluation :
  table_correct ∧
  independence_test male_good_A male_good_B female_good_A female_good_B ∧
  preferred_route (expected_value 100 [(6, 1/27), (9, 2/9), (12, 4/9), (15, 8/27)])
                  (expected_value 75 [(6, 1/8), (9, 3/8), (12, 3/8), (15, 1/8)]) :=
by sorry

end RouteEvaluation

end route_evaluation_l687_687946


namespace negation_of_existential_negation_example_l687_687130

theorem negation_of_existential (P : ℝ → Prop) :
  (∃ x : ℝ, P x) ↔ ¬ (∀ x : ℝ, ¬ P x) :=
begin
  exact ⟨λ ⟨x, hx⟩ h, h x hx, λ h, ⟨_, classical.not_forall.mp h⟩⟩
end

theorem negation_example :
  ¬ (∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ ∀ x : ℝ, x^3 - 2*x + 1 ≠ 0 :=
begin
  split,
  {
    intros h x hx,
    exact h ⟨x, hx⟩,
  },
  {
    intros h ⟨x, hx⟩,
    exact h x hx,
  }
end

lemma negation_specific_example :
  ¬ (∃ x : ℝ, x^3 - 2*x + 1 = 0) = (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) :=
by exact negation_example

end negation_of_existential_negation_example_l687_687130


namespace start_of_range_l687_687941

variable (x : ℕ)

theorem start_of_range (h : ∃ (n : ℕ), n ≤ 79 ∧ n % 11 = 0 ∧ x = 79 - 3 * 11) 
(h4 : ∀ (k : ℕ), 0 ≤ k ∧ k < 4 → ∃ (y : ℕ), y = 79 - (k * 11) ∧ y % 11 = 0) :
  x = 44 := by
  sorry

end start_of_range_l687_687941


namespace exists_vertex_with_degree_at_most_five_l687_687462

variable {V E F : ℕ}

theorem exists_vertex_with_degree_at_most_five
  (handshaking_lemma : ∑ v in {v : ℕ // true}, nat_degree v = 2 * E)
  (euler_formula : V - E + F = 2)
  (faces_edges_relation : 3 * F ≤ 2 * E) :
  ∃ v, (∃G : simple_graph (fin V), is_planar G) → ((G.degree v ≤ 5) : Prop) :=
by
  sorry

end exists_vertex_with_degree_at_most_five_l687_687462


namespace range_of_a_l687_687734

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l687_687734


namespace odd_even_divisor_ratio_l687_687445

def N := 56 * 56 * 95 * 405

theorem odd_even_divisor_ratio :
  let sum_of_all_divisors := (1 + 2 + 4 + 8 + 16 + 32 + 64) 
                                  * (1 + 3 + 3^2 + 3^3 + 3^4)
                                  * (1 + 5 + 5^2)
                                  * (1 + 7 + 7^2)
                                  * (1 + 19),
      sum_of_odd_divisors := (1 + 3 + 3^2 + 3^3 + 3^4) 
                               * (1 + 5 + 5^2)
                               * (1 + 7 + 7^2)
                               * (1 + 19),
      sum_of_even_divisors := sum_of_all_divisors - sum_of_odd_divisors
  in (sum_of_odd_divisors / sum_of_even_divisors) = (1 / 126) := by
    sorry

end odd_even_divisor_ratio_l687_687445


namespace sum_smallest_largest_prime_mul_count_eq_l687_687609

theorem sum_smallest_largest_prime_mul_count_eq :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let smallest := 2
  let largest := 47
  let num_primes := 15
  (smallest + largest) * num_primes = 735 :=
by
  -- Definitions as per identified conditions
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  have smallest : Nat := primes.head -- Smallest prime is 2
  have largest : Nat := primes.getLast (by decide) -- Largest prime is 47
  have num_primes : Nat := primes.length -- Number of primes is 15
  -- Begin proof
  calc
    (smallest + largest) * num_primes = (2 + 47) * 15 : by
      simp only [smallest, largest, num_primes]
    _ = 49 * 15 : by simp
    _ = 735 : by norm_num

end sum_smallest_largest_prime_mul_count_eq_l687_687609


namespace winning_probability_correct_l687_687928

-- Define the conditions
def numPowerBalls : ℕ := 30
def numLuckyBalls : ℕ := 49
def numChosenBalls : ℕ := 6

-- Define the probability of picking the correct PowerBall
def powerBallProb : ℚ := 1 / numPowerBalls

-- Define the combination function for choosing LuckyBalls
noncomputable def combination (n k : ℕ) : ℕ := n.choose k

-- Define the probability of picking the correct LuckyBalls
noncomputable def luckyBallProb : ℚ := 1 / (combination numLuckyBalls numChosenBalls)

-- Define the total winning probability
noncomputable def totalWinningProb : ℚ := powerBallProb * luckyBallProb

-- State the theorem to prove
theorem winning_probability_correct : totalWinningProb = 1 / 419512480 :=
by
  sorry

end winning_probability_correct_l687_687928


namespace alex_walking_distance_l687_687388

theorem alex_walking_distance
  (distance : ℝ)
  (time_45 : ℝ)
  (walking_rate : distance = 1.5 ∧ time_45 = 45):
  ∃ distance_90, distance_90 = 3 :=
by 
  sorry

end alex_walking_distance_l687_687388


namespace f1_flat_bottomed_f2_not_flat_bottomed_g_flat_bottomed_m_1_n_1_l687_687321

-- Definitions of the functions
def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)
def f2 (x : ℝ) : ℝ := x + abs (x - 2)
def g (m n x : ℝ) : ℝ := m * x + sqrt (x^2 + 2 * x + n)

-- General definition for a flat-bottomed function
def is_flat_bottomed (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∃ a b c, [a, b] ⊆ D ∧ (∀ x ∈ [a, b], f x = c) ∧ (∀ x ∈ D, x ∉ [a, b] → f x > c)

-- Statement that f1 is a flat-bottomed function on ℝ
theorem f1_flat_bottomed : is_flat_bottomed f1 set.univ :=
sorry  -- Proof omitted

-- Statement that f2 is not a flat-bottomed function on ℝ
theorem f2_not_flat_bottomed : ¬ is_flat_bottomed f2 set.univ :=
sorry  -- Proof omitted

-- Statement that g is a flat-bottomed function on [-2, +∞) with m = 1 and n = 1
theorem g_flat_bottomed_m_1_n_1 : 
  is_flat_bottomed (g 1 1) (set.Ici (-2)) :=
sorry  -- Proof omitted

end f1_flat_bottomed_f2_not_flat_bottomed_g_flat_bottomed_m_1_n_1_l687_687321


namespace John_walking_time_l687_687430

theorem John_walking_time (d1 d2 t1 t2 : ℝ) (h1 : d1 = 2) (h2 : t1 = 6) (h3 : d2 = 4) (h4 : t2 = 12) :
  (d1/t1 = (d2/t2)) :=
by
  subst_vars
  simp
  sorry

end John_walking_time_l687_687430


namespace translation_motion_l687_687563

-- Define the motion types
def up_and_down_elevator : Prop := true
def swinging_on_swing : Prop := false
def closing_textbook : Prop := false
def swinging_pendulum : Prop := false

-- Theorem: Only Option A describes translation motion.
theorem translation_motion :
  up_and_down_elevator ∧ ¬swinging_on_swing ∧ ¬closing_textbook ∧ ¬swinging_pendulum :=
by
  split;
  try { exact true.intro };
  try { exact false.elim sorry };
  sorry

end translation_motion_l687_687563


namespace toy_ratio_l687_687427

variable (Jaxon : ℕ) (Gabriel : ℕ) (Jerry : ℕ)

theorem toy_ratio (h1 : Jerry = Gabriel + 8) 
                  (h2 : Jaxon = 15)
                  (h3 : Gabriel + Jerry + Jaxon = 83) :
                  Gabriel / Jaxon = 2 := 
by
  sorry

end toy_ratio_l687_687427


namespace abs_diff_eq_2_point_1_l687_687441

noncomputable def floor (z : ℝ) : ℤ := int.floor z
noncomputable def frac (z : ℝ) : ℝ := z - floor z

theorem abs_diff_eq_2_point_1 (x y : ℝ)
  (h1 : floor x - frac y = 3.7)
  (h2 : frac x + floor y = 6.2) :
  |x - y| = 2.1 :=
sorry

end abs_diff_eq_2_point_1_l687_687441


namespace hyperbola_asymptote_condition_intersect_line_hyperbola_l687_687791

def is_asymptote (line_eq : ℝ → ℝ → Prop) (h_eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line_eq x y → ∀ ε > 0, ∃ δ > 0, ∀ t > δ, h_eq (x + t * cos 45) (y + t * sin 45) 

def point_on_hyperbola (p : ℝ × ℝ) (h_eq : ℝ → ℝ → Prop) : Prop :=
  h_eq p.1 p.2

def standard_eq_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 4

theorem hyperbola_asymptote_condition:
  (is_asymptote (λ x y, y = x) (standard_eq_hyperbola)) ∧ 
  (point_on_hyperbola (sqrt 5, 1) (standard_eq_hyperbola)) →
  ∀ x y, standard_eq_hyperbola x y := 
sorry

theorem intersect_line_hyperbola:
  ∀ k : ℝ,
  (∀ x y, y = k*x - 1 → standard_eq_hyperbola x y → x^2 - y^2 = 4 → (1 - k^2) * x^2 + 2 * k * x - 5 = 0) →
  ∃ k : ℝ, k = 1 ∨ k = -1 ∨ k = sqrt 5 / 2 ∨ k = -sqrt 5 / 2 :=
sorry

end hyperbola_asymptote_condition_intersect_line_hyperbola_l687_687791


namespace part_a_even_function_part_b_not_odd_function_part_c_minimum_value_part_d_no_three_real_roots_l687_687801

def f (x a : ℝ) : ℝ := x^2 + 2 * (abs (x - a))

theorem part_a_even_function (a : ℝ) (h : a = 0) : 
  ∀ x, f x a = f (-x) a := 
begin
  assume x,
  rw h,
  simp [f],
end

theorem part_b_not_odd_function : ¬ ∃ a : ℝ, ∀ x, f x a = -f (-x) a := 
begin
  assume h, 
  cases h with a ha,
  specialize ha a,
  sorry  -- Detailed proof skipped
end

theorem part_c_minimum_value (a : ℝ) (h1 : -1 < a) (h2 : a < 1) : 
  ∀ x, f x a ≥ a^2 ∧ (∃ x', f x' a = a^2) := 
begin
  assume x,
  split,
  -- proving f(x) >= a^2
  sorry,  -- Detailed proof skipped
  -- proving min value a^2 exists
  existsi a,
  simp [f],
end

theorem part_d_no_three_real_roots (a m : ℝ) : 
  ¬ ∃ x1 x2 x3, f x1 a = m ∧ f x2 a = m ∧ f x3 a = m := 
begin
  assume h, 
  cases h with x1 hx1,
  cases hx1 with x2 hx2,
  cases hx2 with x3 hx3,
  sorry  -- Detailed proof skipped
end

end part_a_even_function_part_b_not_odd_function_part_c_minimum_value_part_d_no_three_real_roots_l687_687801


namespace incorrect_statement_B_l687_687365

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem incorrect_statement_B
  (ω φ : ℝ)
  (h_ω_pos : ω > 0)
  (h_φ_range : 0 < φ ∧ φ < π)
  (h_f_neg_pi_by_10 : f ω φ (-π / 10) = 0)
  (h_f_le_max_at_2pi_by_5 : ∀ x, f ω φ x ≤ |f ω φ (2 * π / 5)|)
  (h_monotonic : ∀ x1 x2, -π / 5 < x1 ∧ x1 < x2 ∧ x2 < π / 10 → f ω φ x1 < f ω φ x2)
  : ¬ (φ = 3 * π / 5) :=
sorry

end incorrect_statement_B_l687_687365


namespace smallest_constant_M_l687_687300

theorem smallest_constant_M (x y z w : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) :
    sqrt (x / (y + z + w)) + sqrt (y / (x + z + w)) + sqrt (z / (x + y + w)) + sqrt (w / (x + y + z)) < 4 / sqrt 3 :=
by
  sorry

end smallest_constant_M_l687_687300


namespace max_min_of_f_on_interval_l687_687292

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 4 + 4 * x ^ 3 + 34

theorem max_min_of_f_on_interval :
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ 50) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 50) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, 33 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 33) :=
by
  sorry

end max_min_of_f_on_interval_l687_687292


namespace vector_properties_l687_687813

-- Define the vectors structurally
def a (λ θ : ℝ) : ℝ × ℝ := (Real.cos (λ * θ), Real.cos ((10 - λ) * θ))
def b (λ θ : ℝ) : ℝ × ℝ := (Real.sin ((10 - λ) * θ), Real.sin (λ * θ))

-- State the main theorem
theorem vector_properties (λ θ : ℝ) :
    ((a λ θ).fst^2 + (a λ θ).snd^2) + ((b λ θ).fst^2 + (b λ θ).snd^2) = 2
    ∧ ((a λ θ).fst * (b λ θ).fst + (a λ θ).snd * (b λ θ).snd = 0 → ∃ k : ℤ, θ = k * Real.pi / 10)
    ∧ (θ = Real.pi / 20 → ((a λ θ).fst * (b λ θ).fst + (a λ θ).snd * (b λ θ).snd = 0)) :=
by
  sorry

end vector_properties_l687_687813


namespace mathematics_class_size_l687_687223

theorem mathematics_class_size (total_students : ℕ) (total_students_eq : total_students = 75)
  (math_or_physics : ∀ s, s ∈ (Mathematics ∪ Physics))
  (math_size : ∀ physics_class math_class : ℕ, physics_class * 2 = math_class)
  (both_classes : ℕ) (both_classes_eq : both_classes = 10) :
  ∃ math_size_total, math_size_total = 170 / 3 := 
sorry

end mathematics_class_size_l687_687223


namespace sin_bound_l687_687983

theorem sin_bound (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < sin x ∧ sin x < x := 
sorry

end sin_bound_l687_687983


namespace vectors_parallel_l687_687830

variable (x : ℝ)
def a := (1 : ℝ, x)
def b := (-2 : ℝ, 1)

theorem vectors_parallel (h : ∃ k : ℝ, a = k • b) : x = -1 / 2 :=
by sorry

end vectors_parallel_l687_687830


namespace susan_ate_candies_l687_687242

theorem susan_ate_candies (candies_tuesday candies_thursday candies_friday candies_left : ℕ) 
  (h_tuesday : candies_tuesday = 3) 
  (h_thursday : candies_thursday = 5) 
  (h_friday : candies_friday = 2) 
  (h_left : candies_left = 4) : candies_tuesday + candies_thursday + candies_friday - candies_left = 6 := by
  sorry

end susan_ate_candies_l687_687242


namespace range_of_a_l687_687737

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l687_687737


namespace minimum_g_value_l687_687885

noncomputable def tetrahedron_PQRS (P Q R S : Point) : Prop :=
  distance P Q = 30 ∧
  distance R S = 30 ∧
  distance P R = 46 ∧
  distance Q S = 46 ∧
  distance P S = 50 ∧
  distance Q R = 50

def g (X P Q R S : Point) : ℝ :=
  distance P X + distance Q X + distance R X + distance S X

theorem minimum_g_value {P Q R S X : Point} (h : tetrahedron_PQRS P Q R S) :
  ∃ a b : ℕ, (a + b = 638 ∧ g (some_point_on_TU) P Q R S = (a * (Real.sqrt b)) :=
sorry

end minimum_g_value_l687_687885


namespace orange_orchard_land_l687_687183

theorem orange_orchard_land (F H : ℕ) 
  (h1 : F + H = 120) 
  (h2 : ∃ x : ℕ, x + (2 * x + 1) = 10) 
  (h3 : ∃ x : ℕ, 2 * x + 1 = H)
  (h4 : ∃ x : ℕ, F = x) 
  (h5 : ∃ y : ℕ, H = 2 * y + 1) :
  F = 36 ∧ H = 84 :=
by
  sorry

end orange_orchard_land_l687_687183


namespace range_of_a_l687_687725

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687725


namespace terminating_decimals_count_l687_687305

theorem terminating_decimals_count :
  let n_values := {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∃ k, n = 21 * k)} in
  n_values.finite.count = 23 :=
by {
  sorry
}

end terminating_decimals_count_l687_687305


namespace find_a_l687_687888

theorem find_a (a : ℝ) (h : ({-1, 1, 3} ∩ {a + 2, a^2 + 4} = {3})) : a = 1 := by
  sorry

end find_a_l687_687888


namespace range_of_x_l687_687394

noncomputable def f : ℝ → ℝ
| x := if (-1 ≤ x ∧ x ≤ 1) then 3 * (1 - 2 ^ x) / (2 ^ x + 1)
       else -1 / 4 * (x ^ 3 + 3 * x)

theorem range_of_x : 
  (∀ m, m ∈ set.Icc (-3 : ℝ) 2 → f (m * x - 1) + f x > 0) ↔ (-1/2 < x) ∧ (x < 1/3) := 
sorry

end range_of_x_l687_687394


namespace collapsing_fraction_l687_687089

-- Define the total number of homes on Gotham St as a variable.
variable (T : ℕ)

/-- Fraction of homes on Gotham Street that are termite-ridden. -/
def fraction_termite_ridden (T : ℕ) : ℚ := 1 / 3

/-- Fraction of homes on Gotham Street that are termite-ridden but not collapsing. -/
def fraction_termite_not_collapsing (T : ℕ) : ℚ := 1 / 10

/-- Fraction of termite-ridden homes that are collapsing. -/
theorem collapsing_fraction :
  (fraction_termite_ridden T - fraction_termite_not_collapsing T) = 7 / 30 :=
by
  sorry

end collapsing_fraction_l687_687089


namespace range_of_a_l687_687697

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687697


namespace trapezoid_inequality_l687_687418

noncomputable def trapezoid (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
  ∃ (θ1 θ2 : ℝ), θ1 < θ2 ∧ θ2 < π / 2 

theorem trapezoid_inequality 
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (h_trap : trapezoid A B C D) 
  (h_angle : ∃ (θ1 θ2 : ℝ), θ1 < θ2 ∧ θ2 < π / 2) :
  dist A C > dist B D := 
sorry

end trapezoid_inequality_l687_687418


namespace sqrt_eq_seven_implies_x_eq_six_l687_687172

theorem sqrt_eq_seven_implies_x_eq_six (x : ℝ) (h : sqrt (4 + 9 + x^2) = 7) : x = 6 ∨ x = -6 :=
by
  sorry

end sqrt_eq_seven_implies_x_eq_six_l687_687172


namespace student_tickets_sold_l687_687537

theorem student_tickets_sold (S NS : ℕ) (h1 : 9 * S + 11 * NS = 20960) (h2 : S + NS = 2000) : S = 520 :=
by
  sorry

end student_tickets_sold_l687_687537


namespace books_shelved_in_fiction_section_l687_687425

def calculate_books_shelved_in_fiction_section (total_books : ℕ) (remaining_books : ℕ) (books_shelved_in_history : ℕ) (books_shelved_in_children : ℕ) (books_added_back : ℕ) : ℕ :=
  let total_shelved := total_books - remaining_books
  let adjusted_books_shelved_in_children := books_shelved_in_children - books_added_back
  let total_shelved_in_history_and_children := books_shelved_in_history + adjusted_books_shelved_in_children
  total_shelved - total_shelved_in_history_and_children

theorem books_shelved_in_fiction_section:
  calculate_books_shelved_in_fiction_section 51 16 12 8 4 = 19 :=
by 
  -- Definition of the function gives the output directly so proof is trivial.
  rfl

end books_shelved_in_fiction_section_l687_687425


namespace color_regions_with_two_colors_l687_687405

theorem color_regions_with_two_colors (n : ℕ) (lines : Fin n → Set (ℝ × ℝ)) :
  ∃ (color : Set (ℝ × ℝ) → Fin 2), ∀ r1 r2 : Set (ℝ × ℝ), 
    (Adjacent r1 r2 lines → color r1 ≠ color r2) :=
sorry

/- Definitions and assumptions needed -/
def Adjacent (r1 r2 : Set (ℝ × ℝ)) (lines : Set (Set (ℝ × ℝ))) : Prop :=
∃ l ∈ lines, (r1 ∩ l).Nonempty ∧ (r2 ∩ l).Nonempty

end color_regions_with_two_colors_l687_687405


namespace total_books_in_classroom_l687_687401

-- Define the given conditions using Lean definitions
def num_children : ℕ := 15
def books_per_child : ℕ := 12
def additional_books : ℕ := 22

-- Define the hypothesis and the corresponding proof statement
theorem total_books_in_classroom : num_children * books_per_child + additional_books = 202 := 
by sorry

end total_books_in_classroom_l687_687401


namespace terminating_decimals_count_l687_687306

noncomputable def int_counts_terminating_decimals : ℕ :=
  let n_limit := 500
  let denominator := 2100
  Nat.floor (n_limit / 21)

theorem terminating_decimals_count :
  int_counts_terminating_decimals = 23 :=
by
  /- Proof will be here eventually -/
  sorry

end terminating_decimals_count_l687_687306


namespace machineB_rate_correct_l687_687891

-- Define the rate of Machine A
def rateA : ℝ := 6000 / 3  -- envelopes per hour

-- Define the combined rate of Machines B and C
def rateB_C : ℝ := 6000 / 2.5  -- envelopes per hour

-- Define the combined rate of Machines A and C
def rateA_C : ℝ := 3000 / 1  -- envelopes per hour

-- Define the rate of Machine B we want to verify
def rateB : ℝ := rateB_C - (rateA_C - rateA)

-- The theorem that asserts Machine B processes envelopes at a rate of 1400 envelopes/hour
theorem machineB_rate_correct : rateB = 1400 :=
by
  unfold rateA rateB_C rateA_C rateB
  norm_num
  sorry

end machineB_rate_correct_l687_687891


namespace polynomial_divisibility_l687_687899

variable {R : Type*} [CommRing R] {p q r : R[X]}

theorem polynomial_divisibility (h : Irreducible p) (hqr : p ∣ q * r) : p ∣ q ∨ p ∣ r :=
  sorry

end polynomial_divisibility_l687_687899


namespace problem_statement_l687_687069

noncomputable def cubic_poly := Polynomial.Cubic 3 (-4) 200 (-5)

theorem problem_statement (p q r : ℝ) (h_roots : p = cubic_poly.root1 ∧ q = cubic_poly.root2 ∧ r = cubic_poly.root3) :
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 403 := 
sorry

end problem_statement_l687_687069


namespace sin_bound_l687_687987

theorem sin_bound (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < sin x ∧ sin x < x := 
sorry

end sin_bound_l687_687987


namespace tournament_committee_count_l687_687414

-- Given conditions
def num_teams : ℕ := 5
def members_per_team : ℕ := 8
def committee_size : ℕ := 11
def nonhost_member_selection (n : ℕ) : ℕ := (n.choose 2) -- Selection of 2 members from non-host teams
def host_member_selection (n : ℕ) : ℕ := (n.choose 2)   -- Selection of 2 members from the remaining members of the host team; captain not considered in this choose as it's already selected

-- The total number of ways to form the required tournament committee
def total_committee_selections : ℕ :=
  num_teams * host_member_selection 7 * (nonhost_member_selection 8)^4

-- Proof stating the solution to the problem
theorem tournament_committee_count :
  total_committee_selections = 64534080 := by
  sorry

end tournament_committee_count_l687_687414


namespace infinite_union_finite_or_countable_has_same_cardinality_l687_687900

variable {X Y : Set}

-- Define properties/infinite set X
variable [Infinite X] -- X is an infinite set
variable (Y_countable_or_finite : (Countable Y ∨ Finite Y)) -- Y is finite or countable

theorem infinite_union_finite_or_countable_has_same_cardinality 
  (X_infinite : Infinite X) 
  (Y_countable_or_finite : ∀ Y, Countable Y ∨ Finite Y) :
  # (X ∪ Y) = # X := sorry

end infinite_union_finite_or_countable_has_same_cardinality_l687_687900


namespace range_of_a_l687_687755

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687755


namespace problem_one_problem_two_axis_of_symmetry_problem_two_center_of_symmetry_problem_three_l687_687062

def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)
def b (x : ℝ) : ℝ × ℝ := (Real.sin x - Real.cos x, -1)

def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1 / 2

theorem problem_one (x : ℝ) : f(x) = (Real.sqrt 2 / 2) * Real.sin (2 * x - Real.pi / 4) := sorry

theorem problem_two_axis_of_symmetry (x : ℝ) (k : ℤ) : 
  x = (k * Real.pi / 2 + 3 * Real.pi / 8) := sorry

theorem problem_two_center_of_symmetry (x : ℝ) (k : ℤ) : 
  (x, 0) = (k * Real.pi / 2 + Real.pi / 8, 0) := sorry

theorem problem_three (x : ℝ) (k : ℤ) : 
  f(x) ≥ 1 / 2 ↔ (Real.pi / 4 + k * Real.pi) ≤ x ∧ x ≤ (Real.pi / 2 + k * Real.pi) := sorry

end problem_one_problem_two_axis_of_symmetry_problem_two_center_of_symmetry_problem_three_l687_687062


namespace ma_m_gt_mb_l687_687314

theorem ma_m_gt_mb (a b : ℝ) (m : ℝ) (h : a < b) : ¬ (m * a > m * b) → m ≥ 0 := 
  sorry

end ma_m_gt_mb_l687_687314


namespace how_many_candies_eaten_l687_687244

variable (candies_tuesday candies_thursday candies_friday candies_left : ℕ)

def total_candies (candies_tuesday candies_thursday candies_friday : ℕ) : ℕ :=
  candies_tuesday + candies_thursday + candies_friday

theorem how_many_candies_eaten (h_tuesday : candies_tuesday = 3)
                               (h_thursday : candies_thursday = 5)
                               (h_friday : candies_friday = 2)
                               (h_left : candies_left = 4) :
  (total_candies candies_tuesday candies_thursday candies_friday) - candies_left = 6 :=
by
  sorry

end how_many_candies_eaten_l687_687244


namespace range_of_a_l687_687728

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l687_687728


namespace sqrt_sum_eq_seven_l687_687111

theorem sqrt_sum_eq_seven (y : ℝ) (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) : 
  sqrt (64 - y^2) + sqrt (36 - y^2) = 7 :=
sorry

end sqrt_sum_eq_seven_l687_687111


namespace largest_integer_less_log_sum_l687_687547

theorem largest_integer_less_log_sum : 
  let s := (∑ i in (Finset.range 10010).filter (λi, i > 0), Real.log2 (i + 1) - Real.log2 i) in
  s = Real.log2 10010 →
  floor s = 13 :=
by
  sorry

end largest_integer_less_log_sum_l687_687547


namespace minimum_N_to_maintain_80_percent_win_rate_l687_687115

theorem minimum_N_to_maintain_80_percent_win_rate :
  ∀ (N : ℕ), (N ≥ 1) ↔ ((3 + N : ℝ) / (4 + N : ℝ) ≥ (4 / 5)) :=
by
  intro N
  split
  { intro h
    rw [ge_from_le, ge_from_le]
    exact_mod_cast (calc
      5 * (3 + N) ≥ 4 * (4 + N) : by linarith [h]
      )
  }
  { intro h
    have h₁ : (4 : ℝ) < 4 + N := by linarith
    rw le_div_iff h₁ at h
    exact_mod_cast (calc
      5 * (3 + N) ≥ 4 * (4 + N) : by linarith
      )
  }
sorry

end minimum_N_to_maintain_80_percent_win_rate_l687_687115


namespace volume_Q3_eq_156035_over_65536_p_plus_q_eq_221571_l687_687327

theorem volume_Q3_eq_156035_over_65536 :
  let Q : ℕ → ℝ := λ i, if i = 0 then 2 else Q (i - 1) + 4 * (3 / 4) ^ 3 * Q (i - 1)
  Q 3 = 156035 / 65536 := 
by
  sorry

theorem p_plus_q_eq_221571 : 
  let Q : ℕ → ℝ := λ i, if i = 0 then 2 else Q (i - 1) + 4 * (3 / 4) ^ 3 * Q (i - 1),
      p : ℤ := 156035,
      q : ℤ := 65536
  Q 3 = p / q → p + q = 221571 :=
by
  sorry

end volume_Q3_eq_156035_over_65536_p_plus_q_eq_221571_l687_687327


namespace ones_digit_of_8_pow_47_l687_687552

theorem ones_digit_of_8_pow_47 :
  (8^47) % 10 = 2 :=
by
  sorry

end ones_digit_of_8_pow_47_l687_687552


namespace calc_t_f_7_l687_687878

noncomputable def t (x : ℝ) : ℝ := Real.sqrt(4 * x^2 + 1)
noncomputable def f (x : ℝ) : ℝ := 7 - t(x)

theorem calc_t_f_7 : t(f(7)) = Real.sqrt(985 - 56 * Real.sqrt(197)) := by
  sorry

end calc_t_f_7_l687_687878


namespace logarithmic_inequality_l687_687674

theorem logarithmic_inequality (m n : ℝ) (hmn : m > n) (hn1 : n > 1) :
  log n m > log m n ∧ log m n > log n (1 / n * m) :=
sorry

end logarithmic_inequality_l687_687674


namespace magic_square_sum_l687_687416

theorem magic_square_sum (x y z w v: ℕ) (h1: 27 + w + 22 = 49 + w)
  (h2: 27 + 18 + x = 45 + x) (h3: 22 + 24 + y = 46 + y)
  (h4: 49 + w = 46 + y) (hw: w = y - 3) (hx: x = y + 1)
  (hz: z = x + 3) : x + z = 45 :=
by {
  sorry
}

end magic_square_sum_l687_687416


namespace inequality_proof_l687_687108

noncomputable def inequality_solution : Set ℝ :=
  {-∞ < x | x < -2016} ∪ {-1009 < x | x < 1007}

theorem inequality_proof (x : ℝ) :
  (x ∈ inequality_solution) ↔ ((|x + 3| + |1 - x|) / (x + 2016) < 1) :=
sorry

end inequality_proof_l687_687108


namespace find_k_l687_687915

open Complex

noncomputable def possible_values_of_k (a b c d e : ℂ) (k : ℂ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∧
  (a * k^4 + b * k^3 + c * k^2 + d * k + e = 0) ∧
  (b * k^4 + c * k^3 + d * k^2 + e * k + a = 0)

theorem find_k (a b c d e : ℂ) (k : ℂ) :
  possible_values_of_k a b c d e k → k^5 = 1 :=
by
  intro h
  sorry

#check find_k

end find_k_l687_687915


namespace finite_commutation_l687_687865

open Group

variables {G : Type*} [Group G]

def finite_order_elements (G : Type*) [Group G] : Set G :=
  { a : G | ∃ (n : ℕ), 0 < n ∧ a ^ n = 1 }

theorem finite_commutation (F : Set G) (hF : F = finite_order_elements G) (hF_fin : F.finite) :
  ∃ (n : ℕ), n > 0 ∧ ∀ (x : G) (y : G), y ∈ F → x ^ n * y = y * x ^ n :=
sorry

end finite_commutation_l687_687865


namespace positive_difference_median_mode_l687_687959

/-!
  Given the stem and leaf plot data:
  4 | 2 4 4 5 5 5
  5 | 1 1 1 3 3 3
  6 | 2 4 6 6 7 8
  7 | 0 4 4 5 5 6
  8 | 1 2 5 8 9 9

  We need to show that the positive difference between the median and the mode is 23.
-/
theorem positive_difference_median_mode :
  let data := [42, 44, 44, 45, 45, 45,
               51, 51, 51, 53, 53, 53,
               62, 64, 66, 66, 67, 68,
               70, 74, 74, 75, 75, 76,
               81, 82, 85, 88, 89, 89] in
  let median := 74 in
  let mode := 51 in
  (median - mode) = 23 :=
by
  sorry

end positive_difference_median_mode_l687_687959


namespace min_next_score_to_increase_avg_l687_687082

def Liam_initial_scores : List ℕ := [72, 85, 78, 66, 90, 82]

def current_average (scores: List ℕ) : ℚ :=
  (scores.sum / scores.length : ℚ)

def next_score_requirement (initial_scores: List ℕ) (desired_increase: ℚ) : ℚ :=
  let current_avg := current_average initial_scores
  let desired_avg := current_avg + desired_increase
  let total_tests := initial_scores.length + 1
  let total_required := desired_avg * total_tests
  total_required - initial_scores.sum

theorem min_next_score_to_increase_avg :
  next_score_requirement Liam_initial_scores 5 = 115 := by
  sorry

end min_next_score_to_increase_avg_l687_687082


namespace part1_part2_l687_687991

-- Part (1)
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 1) : (x - x^2 < Real.sin x) ∧ (Real.sin x < x) :=
sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = Real.cos (a * x) - Real.log (1 - x^2)) ∧ 
  (f' 0 = 0) ∧ (∃ x_max : ℝ, (f'' x_max < 0) ∧ (x_max = 0))) : a < -Real.sqrt 2 ∨ Real.sqrt 2 < a :=
sorry

end part1_part2_l687_687991


namespace largest_possible_b_l687_687140

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 12 :=
by
  sorry

end largest_possible_b_l687_687140


namespace projection_of_a_onto_b_eq_neg_sqrt_2_l687_687312

noncomputable def projection (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / magnitude_b

theorem projection_of_a_onto_b_eq_neg_sqrt_2 :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (-1, 1)
  projection a b = -Real.sqrt 2 :=
by
  sorry

end projection_of_a_onto_b_eq_neg_sqrt_2_l687_687312


namespace range_of_a_l687_687760

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l687_687760


namespace Ceva_concurrence_proof_l687_687503

-- Definitions of the conditions in Lean
variables {A B C A1 B1 C1 X A2 B2 C2 : Type}

-- Assumptions
axiom incircle_touches_sides : touches (incircle ABC) A1 B1 C1
axiom X_inside_triangle : inside_triangle X ABC
axiom AX_intersects_arc : intersects (line AX) (arc B1 C1) A2
axiom BX_intersects_arc : intersects (line BX) (arc C1 A1) B2
axiom CX_intersects_arc : intersects (line CX) (arc A1 B1) C2

-- The theorem to prove
theorem Ceva_concurrence_proof :
  concurrent (line A1 A2) (line B1 B2) (line C1 C2) :=
sorry

end Ceva_concurrence_proof_l687_687503


namespace min_distance_parabola_l687_687686

open Real

theorem min_distance_parabola {P : ℝ × ℝ} (hP : P.2^2 = 4 * P.1) : ∃ m : ℝ, m = 2 * sqrt 3 ∧ ∀ Q : ℝ × ℝ, Q = (4, 0) → dist P Q ≥ m :=
by sorry

end min_distance_parabola_l687_687686


namespace problem_f_2008_mod_100_l687_687894

theorem problem_f_2008_mod_100 : 
    let f : ℕ → ℕ := λ n, if n = 1 then 1 
                        else if n = 2 then 1 
                        else f (n - 1) + f (n - 2)
    in (f 2008) % 100 = 71 :=
by
  let f := λ n, if n = 1 then 1 else if n = 2 then 1 else ((Nat.fib (n - 1)) + (Nat.fib (n - 2)))
  have h : f(1) = 1 := rfl
  have h2 : f(2) = 1 := rfl
  -- Further proof steps involve using the properties and precomputed result
  -- We assume the provided solution as fact for the Lean statement
  sorry

end problem_f_2008_mod_100_l687_687894


namespace decagon_diagonals_l687_687508

-- Definition of the number of diagonals in a polygon with n sides.
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- The proof problem statement
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l687_687508


namespace evaluate_difference_of_squares_l687_687249

theorem evaluate_difference_of_squares :
  (50^2 - 30^2 = 1600) :=
by sorry

end evaluate_difference_of_squares_l687_687249


namespace tadpoles_to_fish_ratio_l687_687942

theorem tadpoles_to_fish_ratio 
  (F : ℕ) (T : ℕ) (F' : ℕ) 
  (hF : F = 50) 
  (hF' : F' = F - 7) 
  (hT : T / 2 = F' + 32) : 
  T / F = 3 := 
by 
  rw [hF, hF'] at hT
  have h : T / 2 = 43 + 32 := hT
  have h' : T / 2 = 75 := by norm_num at h
  have t_value : T = 150 := by linarith at h'
  have ratio := T / F
  rw [hF, t_value] at ratio
  have ratio' : 150 / 50 = 3 := by norm_num
  exact ratio'

end tadpoles_to_fish_ratio_l687_687942


namespace cylinder_volume_from_rectangle_l687_687664

theorem cylinder_volume_from_rectangle (width length : ℝ) (h₁ : width = 8) (h₂ : length = 20) :
  let r := width / 2
  let h := length
  volume := π * r^2 * h
  volume = 320 * π :=
by
  sorry

end cylinder_volume_from_rectangle_l687_687664


namespace inscribed_circle_radius_l687_687948

-- Definitions and assumptions based on the problem statement.
def isosceles_triangle (a b c: ℝ) :=
a = b

def congruent_triangles (a b: ℝ) :=
a = b

def midpoint (x y m: ℝ) :=
m = (x + y) / 2

def perpendicular (x y: ℝ) :=
x * y = 0

-- Given conditions
variables (EF FG GH HE: ℝ)
variables (EG FH: ℝ)
variables (FI: ℝ)

-- Assume the lengths given in the problem
axiom length_conditions: EF = 13 ∧ FG = 13 ∧ GH = 13 ∧ HE = 13 ∧ EG = 24

-- I is the midpoint of EG
axiom midpoint_condition: midpoint 0 EG 12 -- Using the fact that I is the midpoint

-- FI is perpendicular to EG
axiom perpendicular_condition: perpendicular FI EG

-- The radius of the inscribed circle in triangle EFI is given
theorem inscribed_circle_radius:
  (∃ r: ℝ, ∀ (r = 2), -- Radius of the inscribed circle is 2
  sorry

end inscribed_circle_radius_l687_687948


namespace range_of_a_l687_687716

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l687_687716


namespace tan_alpha_result_l687_687822

theorem tan_alpha_result (α : ℝ) (h : Real.tan (α - Real.pi / 4) = 1 / 6) : Real.tan α = 7 / 5 :=
by
  sorry

end tan_alpha_result_l687_687822


namespace exists_person_with_exactly_2008_acquaintances_l687_687677

-- Definitions
variable (G : Type) -- Type representing the gathering
variable [inhabited G] [fintype G] -- Assume G is non-empty and finite
variable acquaintances : G → set G -- Relation representing acquaintances

-- Conditions
def no_common_acquaintances_if_same_number (x y : G) : Prop :=
  (finite.to_finset (acquaintances x)).card = (finite.to_finset (acquaintances y)).card →
  (finite.to_finset (acquaintances x ∩ acquaintances y)).card = 0

-- Statement to prove
theorem exists_person_with_exactly_2008_acquaintances (h : ∃ x : G, (finite.to_finset (acquaintances x)).card ≥ 2008)
  (h_cond : ∀ x y : G, no_common_acquaintances_if_same_number acquaintances x y) :
  ∃ x : G, (finite.to_finset (acquaintances x)).card = 2008 :=
sorry

end exists_person_with_exactly_2008_acquaintances_l687_687677


namespace equilateral_triangle_dot_product_l687_687033

noncomputable def dot_product_sum (a b c : ℝ) := 
  a * b + b * c + c * a

theorem equilateral_triangle_dot_product 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A = 1)
  (h2 : B = 1)
  (h3 : C = 1)
  (h4 : a = 1)
  (h5 : b = 1)
  (h6 : c = 1) :
  dot_product_sum a b c = 1 / 2 :=
by 
  sorry

end equilateral_triangle_dot_product_l687_687033


namespace expand_product_l687_687652

def poly1 (x : ℝ) := 4 * x + 2
def poly2 (x : ℝ) := 3 * x - 1
def poly3 (x : ℝ) := x + 6

theorem expand_product (x : ℝ) :
  (poly1 x) * (poly2 x) * (poly3 x) = 12 * x^3 + 74 * x^2 + 10 * x - 12 :=
by
  sorry

end expand_product_l687_687652


namespace evans_family_children_count_l687_687488

-- Let the family consist of the mother, the father, two grandparents, and children.
-- This proof aims to show x, the number of children, is 1.

theorem evans_family_children_count
  (m g y : ℕ) -- m = mother's age, g = average age of two grandparents, y = average age of children
  (x : ℕ) -- x = number of children
  (avg_family_age : (m + 50 + 2 * g + x * y) / (4 + x) = 30)
  (father_age : 50 = 50)
  (avg_non_father_age : (m + 2 * g + x * y) / (3 + x) = 25) :
  x = 1 :=
sorry

end evans_family_children_count_l687_687488


namespace slices_per_banana_l687_687247

-- Define conditions
def yogurts : ℕ := 5
def slices_per_yogurt : ℕ := 8
def bananas : ℕ := 4
def total_slices_needed : ℕ := yogurts * slices_per_yogurt

-- Statement to prove
theorem slices_per_banana : total_slices_needed / bananas = 10 := by sorry

end slices_per_banana_l687_687247


namespace range_of_a_l687_687935

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - (a^2 + a) * x + a^3 > 0 ↔ (x < a^2 ∨ x > a)) → (0 ≤ a ∧ a ≤ 1) :=
by
  intros h
  sorry

end range_of_a_l687_687935


namespace sum_first_2017_terms_eq_l687_687349

variable {α : Type*}
variable [LinearOrderedField α]
variable [OrderedRing α]

-- Define the conditions
def f : α → α := sorry
axiom symmetry_f : ∀ x : α, f x = f (2 - x)
axiom monotone_f : ∀ x y : α, 1 ≤ x → x ≤ y → f x ≤ f y

noncomputable def a : ℕ → α := sorry
axiom a_arithmetic_seq : ∃ d ≠ 0, ∀ n : ℕ, a (n + 1) = a n + d
axiom f_equal_at : f (a 6) = f (a 2012)

-- Problem statement to prove
theorem sum_first_2017_terms_eq : ∑ i in Finset.range 2017, a i = 2017 := 
sorry

end sum_first_2017_terms_eq_l687_687349


namespace johns_raise_percentage_l687_687572

variable (original new earned_diff increase : ℝ)

def percentage_increase (orig new : ℝ) : ℝ := ((new - orig) / orig) * 100

theorem johns_raise_percentage :
  original = 60 ∧ new = 120 →
  percentage_increase original new = 100 :=
by
  assume h,
  cases h with h_orig h_new,
  have earned_diff : new - original = 60 := by
    rw [h_orig, h_new],
    norm_num,
  have increase : (new - original) / original = 1 := by
    rw [h_orig, earned_diff],
    norm_num,
  rw [percentage_increase, h_orig, h_new, increase],
  norm_num,
  sorry

end johns_raise_percentage_l687_687572


namespace range_of_a_l687_687741

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687741


namespace brand_a_percentage_difference_brand_b_percentage_difference_brand_c_percentage_difference_l687_687143

def repair_cost_per_year (repair_cost : ℝ) (repair_lifetime : ℝ) : ℝ :=
  repair_cost / repair_lifetime

def new_purchase_cost_per_year (purchase_price : ℝ) (purchase_lifetime : ℝ) : ℝ :=
  purchase_price / purchase_lifetime

def percentage_difference (repair_cost_per_year : ℝ) (new_cost_per_year : ℝ) : ℝ :=
  let difference := new_cost_per_year - repair_cost_per_year in
  (difference / new_cost_per_year) * 100

theorem brand_a_percentage_difference :
  let repair_cost_a := 12.00
  let repair_lifetime_a := 1.0
  let purchase_price_a := 35.0
  let purchase_lifetime_a := 2.5
  let repair_a := repair_cost_per_year repair_cost_a repair_lifetime_a
  let new_a := new_purchase_cost_per_year purchase_price_a purchase_lifetime_a
  percentage_difference repair_a new_a = 14.29 := sorry

theorem brand_b_percentage_difference :
  let repair_cost_b := 15.50
  let repair_lifetime_b := 1.5
  let purchase_price_b := 45.0
  let purchase_lifetime_b := 3.0
  let repair_b := repair_cost_per_year repair_cost_b repair_lifetime_b
  let new_b := new_purchase_cost_per_year purchase_price_b purchase_lifetime_b
  percentage_difference repair_b new_b = 31.13 := sorry

theorem brand_c_percentage_difference :
  let repair_cost_c := 18.00
  let repair_lifetime_c := 1.75
  let purchase_price_c := 55.0
  let purchase_lifetime_c := 4.0
  let repair_c := repair_cost_per_year repair_cost_c repair_lifetime_c
  let new_c := new_purchase_cost_per_year purchase_price_c purchase_lifetime_c
  percentage_difference repair_c new_c = 25.16 := sorry

end brand_a_percentage_difference_brand_b_percentage_difference_brand_c_percentage_difference_l687_687143


namespace range_of_a_l687_687759

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l687_687759


namespace smallest_positive_integer_x_l687_687663

theorem smallest_positive_integer_x :
  ∃ (x : ℕ), 0 < x ∧ (45 * x + 13) % 17 = 5 % 17 ∧ ∀ y : ℕ, 0 < y ∧ (45 * y + 13) % 17 = 5 % 17 → y ≥ x := 
sorry

end smallest_positive_integer_x_l687_687663


namespace range_of_a_l687_687740

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687740


namespace sum_of_factors_coefficients_l687_687497

theorem sum_of_factors_coefficients (a b c d e f g h i j k l m n o p : ℤ) :
  (81 * x^8 - 256 * y^8 = (a * x + b * y) *
                        (c * x^2 + d * x * y + e * y^2) *
                        (f * x^3 + g * x * y^2 + h * y^3) *
                        (i * x + j * y) *
                        (k * x^2 + l * x * y + m * y^2) *
                        (n * x^3 + o * x * y^2 + p * y^3)) →
  a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p = 40 :=
by
  sorry

end sum_of_factors_coefficients_l687_687497


namespace special_num_closed_l687_687476

-- Define a structure to represent the form a + b * sqrt(2) + c * sqrt(4th root of 2) + d * sqrt(4th root of 8)
structure SpecialNum where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ

-- Closed under addition
def closed_under_add (x y : SpecialNum) : SpecialNum :=
  { a := x.a + y.a,
    b := x.b + y.b,
    c := x.c + y.c,
    d := x.d + y.d }

-- Closed under subtraction
def closed_under_sub (x y : SpecialNum) : SpecialNum :=
  { a := x.a - y.a,
    b := x.b - y.b,
    c := x.c - y.c,
    d := x.d - y.d }

-- Closed under multiplication
noncomputable def closed_under_mul (x y : SpecialNum) : SpecialNum :=
  { a := x.a * y.a + 2 * x.b * y.b + 2 * x.c * y.d + 2 * x.d * y.c,
    b := x.a * y.b + x.b * y.a + x.c * y.c + 2 * x.d * y.d,
    c := x.a * y.c + 2 * x.b * y.d + x.c * y.a + 2 * x.d * y.b,
    d := x.a * y.d + x.b * y.c + x.c * y.b + x.d * y.a }

-- Closed under division (Reciprocal must be considered)
noncomputable def closed_under_div (x y : SpecialNum) (h : y.a * y.a + 2 * y.b * y.b - 4 * y.c * y.d ≠ 0) : SpecialNum :=
  let denom := y.a * y.a + 2 * y.b * y.b - 4 * y.c * y.d + sqrt 2 * (y.c * y.c + 2 * y.d * y.d - 2 * y.a * y.b)
  let num := ClosedUnderReciprocal_mul x (SpecialNum.mk y.a y.b (-y.c) (-y.d))
  { a := num.a / denom,
    b := num.b / denom,
    c := num.c / denom,
    d := num.d / denom }

-- Prove that SpecialNum is closed under the four operations
theorem special_num_closed (x y : SpecialNum) (h : y.a * y.a + 2 * y.b * y.b - 4 * y.c * y.d ≠ 0) : 
  ∃ z : SpecialNum, 
    z = closed_under_add x y ∨ 
    z = closed_under_sub x y ∨ 
    z = closed_under_mul x y ∨ 
    z = closed_under_div x y h :=
  sorry

end special_num_closed_l687_687476


namespace problem_I_problem_II_l687_687980

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 3)

theorem problem_I (x : ℝ) : (f x > 7 - x) ↔ (x < -6 ∨ x > 2) := 
by 
  sorry

theorem problem_II (m : ℝ) : (∃ x : ℝ, f x ≤ abs (3 * m - 2)) ↔ (m ≤ -1 ∨ m ≥ 7 / 3) := 
by 
  sorry

end problem_I_problem_II_l687_687980


namespace range_of_a_l687_687758

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l687_687758


namespace geometric_sequence_seventh_term_l687_687498

theorem geometric_sequence_seventh_term
  (a r : ℝ)
  (h1 : a * r^4 = 16)
  (h2 : a * r^10 = 4) :
  a * r^6 = 4 * (2^(2/3)) :=
by
  sorry

end geometric_sequence_seventh_term_l687_687498


namespace range_of_a_l687_687762

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l687_687762


namespace distinct_real_roots_unique_l687_687346

theorem distinct_real_roots_unique :
  (∃ k : ℕ, (|x| - (4 / x) = (3 * |x|) / x) → k = 1) := sorry

end distinct_real_roots_unique_l687_687346


namespace range_of_f_sin_alpha_minus_beta_l687_687363

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt 3 * real.sin (2 * x) - 2 * (real.cos x)^2 + 1

theorem range_of_f :
  set.image (λ x, f x) (set.Ico (-π/12 : ℝ) (π/2 : ℝ)) = set.Icc (-real.sqrt 3) 2 :=
sorry

variable (α β : ℝ)

theorem sin_alpha_minus_beta 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2)
  (h3 : f ((1/2) * α + (π/12)) = 10/13)
  (h4 : f ((1/2) * β + (π/3)) = 6/5) :
  real.sin (α - β) = -33/65 :=
sorry

end range_of_f_sin_alpha_minus_beta_l687_687363


namespace parallelogram_z_l687_687895

noncomputable def z (A B C : Complex) (D : Complex) : Complex := sorry

theorem parallelogram_z :
  let A := Complex.mk (-3) (-2)
  let B := Complex.mk (-4) 5
  let C := Complex.mk 2 1
  let D := Complex.mk 1 8
  (Complex.Re (B - A) = Complex.Re (D - C)) ∧ (Complex.Im (B - A) = Complex.Im (D - C))
  →
  z A B C D = Complex.mk 1 8 :=
by
  intros
  sorry

end parallelogram_z_l687_687895


namespace stream_speed_l687_687180

theorem stream_speed (v : ℝ) : 
  (∀ (speed_boat_in_still_water distance time : ℝ), 
    speed_boat_in_still_water = 25 ∧ distance = 90 ∧ time = 3 →
    distance = (speed_boat_in_still_water + v) * time) →
  v = 5 :=
by
  intro h
  have h1 := h 25 90 3 ⟨rfl, rfl, rfl⟩
  sorry

end stream_speed_l687_687180


namespace terminating_decimals_count_l687_687304

theorem terminating_decimals_count :
  let n_values := {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∃ k, n = 21 * k)} in
  n_values.finite.count = 23 :=
by {
  sorry
}

end terminating_decimals_count_l687_687304


namespace range_of_a_l687_687738

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687738


namespace f_0_1_f_even_f_periodic_and_sum_l687_687634

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ (m n : ℝ), f (m + n) + f (m - n) = 2 * f m * f n
axiom h2 : ∀ (m : ℝ), f (1 + m) = f (1 - m)
axiom h3_1 : ∃ x₀ : ℝ, f x₀ ≠ 0
axiom h3_2 : ∀ x ∈ set.Ioc 0 1, f x < 1

theorem f_0_1 : f 0 = 1 ∧ f 1 = -1 := sorry

theorem f_even : ∀ x : ℝ, f (-x) = f x := sorry

theorem f_periodic_and_sum :
  (∀ x : ℝ, f (x + 2) = f x) ∧ 
  ∑ k in finset.range 2017, f (k / 3) = 1 / 2 := sorry

end f_0_1_f_even_f_periodic_and_sum_l687_687634


namespace least_possible_k_l687_687391

-- Define the conditions
def prime_factor_form (k : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ k = 2^a * 3^b * 5^c

def divisible_by_1680 (k : ℕ) : Prop :=
  (k ^ 4) % 1680 = 0

-- Define the proof problem
theorem least_possible_k (k : ℕ) (h_div : divisible_by_1680 k) (h_prime : prime_factor_form k) : k = 210 :=
by
  -- Statement of the problem, proof to be filled
  sorry

end least_possible_k_l687_687391


namespace range_of_a_l687_687775

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l687_687775


namespace prove_f2_l687_687065

def func_condition (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x ^ 2 - y) + 2 * c * f x * y

theorem prove_f2 (c : ℝ) (f : ℝ → ℝ)
  (hf : func_condition f c) :
  (f 2 = 0 ∨ f 2 = 4) ∧ (2 * (if f 2 = 0 then 4 else if f 2 = 4 then 4 else 0) = 8) :=
by {
  sorry
}

end prove_f2_l687_687065


namespace pairs_form_1a1_sum_palindrome_l687_687635

def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

def num_of_pairs_palindrome_sum : ℕ :=
  (List.range 10).sum (λ a => (List.range (10 - a)).length)

theorem pairs_form_1a1_sum_palindrome :
  num_of_pairs_palindrome_sum = 55 :=
by
  sorry -- Proof is omitted.

end pairs_form_1a1_sum_palindrome_l687_687635


namespace range_of_a_l687_687731

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l687_687731


namespace muffins_divide_equally_l687_687428

theorem muffins_divide_equally (friends : ℕ) (total_muffins : ℕ) (Jessie_and_friends : ℕ) (muffins_per_person : ℕ) :
  friends = 6 →
  total_muffins = 35 →
  Jessie_and_friends = friends + 1 →
  muffins_per_person = total_muffins / Jessie_and_friends →
  muffins_per_person = 5 :=
by
  intros h_friends h_muffins h_people h_division
  sorry

end muffins_divide_equally_l687_687428


namespace factor_expression_l687_687279

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687279


namespace range_of_a_l687_687730

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l687_687730


namespace connect_points_l687_687682

-- Define the setup
variable (P : Type) [Fintype P]  -- Points on the plane
variable (is_plane : function.some_relation P)  -- Relational axiom, encode plane structure
variable (blue red : Fin 100 → P)  -- 100 blue and 100 red points

-- Define the key properties
def no_three_collinear (S : Fin 100 → P) : Prop := 
  ∀ (a b c : Fin 100), (a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  ¬ is_collinear (S a) (S b) (S c)  -- shorthand for collinearity axiom

-- Main proof statement
theorem connect_points : 
  no_three_collinear blue → no_three_collinear red →
  (∃ f : Fin 100 → Fin 100, 
    ∀ i j, i ≠ j → 
    (blue i ≠ blue j) ∧ 
    (red i ≠ red j) ∧ 
    ¬ is_intersect (blue i, red (f i)) (blue j, red (f j))) := 
begin
  sorry
end

end connect_points_l687_687682


namespace distance_point_to_plane_3D_l687_687850

theorem distance_point_to_plane_3D :
  let x0 := 2
  let y0 := 4
  let z0 := 1
  let A := 1
  let B := 2
  let C := 2
  let D := 3
  d = (abs (A * x0 + B * y0 + C * z0 + D)) / real.sqrt (A^2 + B^2 + C^2)
  d = 5 :=
by
  let x0 := 2
  let y0 := 4
  let z0 := 1
  let A := 1
  let B := 2
  let C := 2
  let D := 3
  let num := abs (A * x0 + B * y0 + C * z0 + D)
  let denom := real.sqrt (A^2 + B^2 + C^2)
  let d := num / denom
  sorry

end distance_point_to_plane_3D_l687_687850


namespace domain_of_f_l687_687234

def my_log (x : ℝ) : ℝ := Real.log x

def f (x : ℝ) : ℝ := my_log (2 ^ x - 1)

theorem domain_of_f : {x : ℝ | 2 ^ x > 1} = {x : ℝ | x > 0} := by
  sorry

end domain_of_f_l687_687234


namespace range_of_f_l687_687662

-- Define the function
def f (x : ℝ) : ℝ := (3 * x + 5) / (x + 4)

-- State the theorem about the range of the function
theorem range_of_f : set.range f = (set.Iio 3) ∪ (set.Ioi 3) :=
by
  sorry

end range_of_f_l687_687662


namespace sum_a1_a11_l687_687820

theorem sum_a1_a11 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℤ) 
  (h1 : a_0 = -512) 
  (h2 : -2 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11) 
  : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 510 :=
sorry

end sum_a1_a11_l687_687820


namespace number_of_unique_sums_l687_687615

def bagA : Set ℕ := {1, 4, 5, 8}
def bagB : Set ℕ := {2, 3, 7, 9}

theorem number_of_unique_sums : 
  let sums := {x + y | x ∈ bagA, y ∈ bagB} in
  sums.card = 12 := 
sorry

end number_of_unique_sums_l687_687615


namespace path_exists_probability_l687_687041

-- Let the maze segments be randomly colored black or white
open ProbabilityTheory

noncomputable def maze_path_probability :=
  let event_a_b : Event (PathExists 'white A B) := sorry
  let event_c_d : Event (PathExists 'black C D) := sorry
  Pr[event_a_b] = 1 / 2 ∧ Pr[event_c_d] = 1 / 2

theorem path_exists_probability :
  maze_path_probability :=
by sorry

end path_exists_probability_l687_687041


namespace find_x_l687_687578

-- Conditions
variables (x y k : ℝ)

-- Inverse square relationship
def inversely_as_square (x y : ℝ) : Prop := x = k / y^2

-- Given values
def given_conditions : Prop := inversely_as_square 1 3 ∧ k = 9

-- Statement to prove: Given the conditions, we need to prove that x = 0.5625 when y = 4
theorem find_x (h : given_conditions k) : inversely_as_square x 4 → x = 0.5625 :=
by sorry

end find_x_l687_687578


namespace sum_of_n_values_l687_687536

def is_divisible (a b : Nat) : Prop := b ∣ a

def n_values : Set Nat := { n | n > 0 ∧ is_divisible 180 n ∧ is_divisible (15 * n) 12 ∧ is_divisible (12 * n) 15 }

theorem sum_of_n_values : ( ∑ n in n_values, n ) = 260 := by
  sorry

end sum_of_n_values_l687_687536


namespace parabola_directrix_dist_l687_687797

-- Definition of the directrix line l1
def line_l1 (x y : ℝ) : Prop := x - y - 3 = 0

-- Definition of the parabola
def parabola (x y a : ℝ) : Prop := x^2 = a * y

-- Definition of the focus of the parabola
def parabola_focus (a : ℝ) : ℝ × ℝ := (0, a / 4)

-- Distance from point to line formula
def distance_point_to_line (px py a b c : ℝ) : ℝ :=
  abs (a * px + b * py + c) / real.sqrt (a^2 + b^2)

-- Main theorem statement
theorem parabola_directrix_dist (a : ℝ) (h : 0 < a) :
  distance_point_to_line 0 (a / 4) 1 (-1) (-3) = 2 * real.sqrt 2 ↔ a = 4 :=
by
  sorry

end parabola_directrix_dist_l687_687797


namespace terminal_side_in_fourth_quadrant_l687_687336

theorem terminal_side_in_fourth_quadrant 
  (h_sin_half : Real.sin (α / 2) = 3 / 5)
  (h_cos_half : Real.cos (α / 2) = -4 / 5) : 
  (Real.sin α < 0) ∧ (Real.cos α > 0) :=
by
  sorry

end terminal_side_in_fourth_quadrant_l687_687336


namespace cross_section_equilateral_triangle_l687_687504

-- Definitions and conditions
structure Cone where
  r : ℝ -- radius of the base circle
  R : ℝ -- radius of the semicircle
  h : ℝ -- slant height

axiom lateral_surface_unfolded (c : Cone) : c.R = 2 * c.r

def CrossSectionIsEquilateral (c : Cone) : Prop :=
  (c.h ^ 2 = (c.r * c.h)) ∧ (c.h = 2 * c.r)

-- Problem statement with conditions
theorem cross_section_equilateral_triangle (c : Cone) (h_equals_diameter : c.R = 2 * c.r) : CrossSectionIsEquilateral c :=
by
  sorry

end cross_section_equilateral_triangle_l687_687504


namespace three_sum_eq_nine_seven_five_l687_687823

theorem three_sum_eq_nine_seven_five {a b c : ℝ} 
    (h1 : b + c = 15 - 2 * a)
    (h2 : a + c = -10 - 4 * b)
    (h3 : a + b = 8 - 2 * c) : 
    3 * a + 3 * b + 3 * c = 9.75 := 
by
    sorry

end three_sum_eq_nine_seven_five_l687_687823


namespace number_of_subsets_B_l687_687002

def A := {1, 2, 3}

def B := {p : ℕ × ℕ | p.1 ∈ A ∧ p.2 ∈ A ∧ (p.1 + p.2) ∈ A}

theorem number_of_subsets_B : (∃ (n : ℕ), n = 8 ∧ ∃ (s : Finset (ℕ × ℕ)), s = B ∧ s.powerset.card = 2 ^ s.card) :=
by
  use 8
  split
  · refl
  · use B
    split
    · refl
    · sorry

end number_of_subsets_B_l687_687002


namespace count_positive_integers_b_log_b_1024_l687_687818

theorem count_positive_integers_b_log_b_1024 : 
    { b : ℕ // log b 1024 ∈ ℕ ∧ log b 1024 > 0 }.card = 4 :=
by
  sorry

end count_positive_integers_b_log_b_1024_l687_687818


namespace remainder_when_multiplied_by_2003_2004_2005_2006_2007_divided_by_17_is_zero_l687_687151

theorem remainder_when_multiplied_by_2003_2004_2005_2006_2007_divided_by_17_is_zero : 
  (2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 :=
by {
  have h1 : 2003 % 17 = 15 := sorry,
  have h2 : 2004 % 17 = 16 := sorry,
  have h3 : 2005 % 17 = 0 := sorry,
  have h4 : 2006 % 17 = 1 := sorry,
  have h5 : 2007 % 17 = 2 := sorry,
  calc
    (2003 * 2004 * 2005 * 2006 * 2007) % 17
        = (15 * 16 * 0 * 1 * 2) % 17 : by rw [h1, h2, h3, h4, h5]
    ... = 0 % 17 : by norm_num
    ... = 0 : by norm_num
}

end remainder_when_multiplied_by_2003_2004_2005_2006_2007_divided_by_17_is_zero_l687_687151


namespace distance_between_cities_l687_687539

theorem distance_between_cities (x : ℝ) (h1 : x ≥ 100) (t : ℝ)
  (A_speed : ℝ := 12) (B_speed : ℝ := 0.05 * x)
  (condition_A : 7 + A_speed * t + B_speed * t = x)
  (condition_B : t = (x - 7) / (A_speed + B_speed)) :
  x = 140 :=
sorry

end distance_between_cities_l687_687539


namespace range_of_a_l687_687698

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687698


namespace root_exists_in_interval_l687_687022

def f (x : ℝ) := x^3 + 3*x - 1

theorem root_exists_in_interval : ∃ c ∈ set.Icc (0 : ℝ) 1, f c = 0 :=
by
  sorry

end root_exists_in_interval_l687_687022


namespace area_of_shaded_region_l687_687847

noncomputable def area_of_semicircle (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2

theorem area_of_shaded_region :
  let A B C D E F : ℝ := 6
  let small_semicircle_area := area_of_semicircle (A / 2)
  let big_semicircle_area := area_of_semicircle (A * 5 / 2)
  (big_semicircle_area - small_semicircle_area * 4) = 108 * Real.pi :=
by
  sorry

end area_of_shaded_region_l687_687847


namespace find_n_l687_687957

theorem find_n {
    n : ℤ
   } (h1 : 0 ≤ n) (h2 : n < 103) (h3 : 99 * n ≡ 72 [ZMOD 103]) :
    n = 52 :=
sorry

end find_n_l687_687957


namespace complex_problem_l687_687315

section
variable (a b : ℝ)
noncomputable def i : ℂ := complex.I

theorem complex_problem (h : (1 - 2 * i) * (2 + a * i) = (b - 2 * i)) : a + b = 8 :=
by
  sorry
end

end complex_problem_l687_687315


namespace min_length_segment_l687_687320

theorem min_length_segment (a b : ℝ) (hb : b >= (a * (Real.sqrt 2 + 1)) / 2) :
  let R1 := a / Real.sqrt 2,
      R2 := Real.sqrt (b^2 - a * b + (3 * a^2) / 4) in
  abs (R1 - R2) = minLength where
  minLength = if b >= a * (Real.sqrt 2 + 1) / 2 then R2 - R1 else Real.sqrt (4 * b^2 - 8 * a * b + 7 * a^2 - 2 * a^2 * Real.sqrt 2)
  := sorry

end min_length_segment_l687_687320


namespace range_of_a_l687_687773

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l687_687773


namespace limit_p_n_is_half_l687_687230

theorem limit_p_n_is_half : 
  ∀ (p_n : ℕ → ℝ), (∀ n, p_n n = probability_special_number (2 * n)) → 
  tendsto p_n at_top (𝓝 (1 / 2)) :=
sorry

end limit_p_n_is_half_l687_687230


namespace trajectory_of_B_l687_687324

-- Define the points and the line for the given conditions
def A : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (2, -3)
def D_line (x : ℝ) (y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the statement to be proved
theorem trajectory_of_B (x y : ℝ) :
  D_line x y → ∃ Bx By, (3 * Bx - By - 20 = 0) :=
sorry

end trajectory_of_B_l687_687324


namespace part_I_min_value_part_II_sin_x_l687_687005

-- Define the vectors
def vec_m (x : Real) : Real := Real.cos (x / 2) - 1
def vec_n (x : Real) : (Real × Real) := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)

-- Define the function f(x)
noncomputable def f (x : Real) : Real :=
  let m := vec_m x
  let n := vec_n x
  m * n.1 + m * n.2 + 1

-- Problem (I): Proving the minimum value condition for f(x)
theorem part_I_min_value : (∀ x ∈ Set.Icc (Real.pi / 2) Real.pi, f x ≥ 1) ∧
  (∃! x ∈ Set.Icc (Real.pi / 2) Real.pi, f x = 1) :=
by
  sorry

-- Problem (II): Finding sin(x) given f(x) = 11/10
theorem part_II_sin_x {x : Real} (hx : x ∈ Set.Icc 0 (Real.pi / 2)) (hf : f x = 11 / 10) :
  Real.sin x = (3 * Real.sqrt 3 + 4) / 10 :=
by
  sorry

end part_I_min_value_part_II_sin_x_l687_687005


namespace parallel_lines_a_value_l687_687331

noncomputable def slope (line : ℝ → ℝ → Prop) : ℝ := 
  classical.some (Classical.some_spec (exists_unique_slope line))

theorem parallel_lines_a_value :
  ∀ (a : ℝ), (∀ (x y : ℝ), x + a * y + 3 = 0) → 
  (∀ (x y : ℝ), (a - 2) * x + 3 * y + a = 0) → 
  a = -1 :=
by
  sorry

end parallel_lines_a_value_l687_687331


namespace bracelet_ratio_l687_687227

theorem bracelet_ratio : ∃ (x : ℕ), 
  (5 + x - 1/3 * (5 + x) = 6) ∧ (x / 16 = 1 / 4) := 
by
  have h1 : ∀ x : ℕ, 5 + x - 1/3 * (5 + x) = 6 ↔ 2 * x + 10 = 18 := 
    sorry
  have h2 : ∃ x : ℕ, 2 * x + 10 = 18 := 
    sorry
  obtain ⟨x, hx⟩ := h2
  use [x],
  exact ⟨by rw [← h1 x]; exact hx, sorry⟩

end bracelet_ratio_l687_687227


namespace books_about_outer_space_l687_687147

variable (x : ℕ)

theorem books_about_outer_space :
  160 + 48 + 16 * x = 224 → x = 1 :=
by
  intro h
  sorry

end books_about_outer_space_l687_687147


namespace range_of_a_l687_687757

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687757


namespace company_picnic_employees_l687_687224

theorem company_picnic_employees (managers : ℕ) (teams : ℕ) (people_per_team : ℕ) (total_players : ℕ) : 
  teams = 6 → 
  people_per_team = 5 → 
  total_players = teams * people_per_team → 
  managers = 23 → 
  total_players - managers = 7 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h3
  have ht : 6 * 5 = 30 := by norm_num
  rw ht at h3
  rw h3 at h4
  rw h4
  norm_num
  sorry

end company_picnic_employees_l687_687224


namespace rectangle_area_l687_687511

theorem rectangle_area (y : ℝ) (h_rect : (5 - (-3)) * (y - (-1)) = 48) (h_pos : 0 < y) : y = 5 :=
by
  sorry

end rectangle_area_l687_687511


namespace solve_system_l687_687479

variables (a b c d : ℝ)

theorem solve_system :
  (a + c = -4) ∧
  (a * c + b + d = 6) ∧
  (a * d + b * c = -5) ∧
  (b * d = 2) →
  ((a = -3 ∧ b = 2 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = -3 ∧ d = 2)) :=
by
  intro h
  -- Insert proof here
  sorry

end solve_system_l687_687479


namespace max_participants_l687_687840

-- Define the sets of people who know each subject
constants (A B C : Finset ℕ)

-- Define the given conditions
axiom h1 : A.card = 8
axiom h2 : B.card = 7
axiom h3 : C.card = 11
axiom h4 : (A ∩ B).card ≥ 2
axiom h5 : (B ∩ C).card ≥ 3
axiom h6 : (A ∩ C).card ≥ 4

-- The theorem stating the maximum number of participants
theorem max_participants : (A ∪ B ∪ C).card = 19 := sorry

end max_participants_l687_687840


namespace pages_per_book_l687_687859

theorem pages_per_book (P : ℕ) 
  (books_last_month : ℕ := 5) 
  (books_this_month : ℕ := 2 * books_last_month) 
  (total_pages : ℕ := 150) 
  (total_books : ℕ := books_last_month + books_this_month) 
  (total_pages_calc : ℕ := books_last_month * P + books_this_month * P) : 
  total_pages = total_pages_calc → P = 10 := 
by 
  intro h
  unfold books_last_month books_this_month total_books total_pages_calc at h
  rw [add_mul, mul_comm 2] at h
  linarith
  sorry

end pages_per_book_l687_687859


namespace simplify_and_evaluate_l687_687103

noncomputable def a := 3

theorem simplify_and_evaluate : (a^2 / (a + 1) - 1 / (a + 1)) = 2 := by
  sorry

end simplify_and_evaluate_l687_687103


namespace f_symm_l687_687438

def f (a b : ℕ) : ℕ :=
  {s : fin a → ℤ // (finset.univ.sum $ λ i, |s i|) ≤ b}.card

theorem f_symm (a b : ℕ) : f a b = f b a := 
  sorry

end f_symm_l687_687438


namespace coefficient_of_x_in_binomial_expansion_l687_687233

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x_in_binomial_expansion :
  (∃ r : ℕ, 5 - 2 * r = 1 ∧ binomial_coeff 5 r * (-2)^r = 40) :=
by
  sorry

end coefficient_of_x_in_binomial_expansion_l687_687233


namespace fraction_length_EF_of_GH_l687_687898

theorem fraction_length_EF_of_GH (GH GE EH GF FH EF : ℝ)
  (h1 : GE = 3 * EH)
  (h2 : GF = 4 * FH)
  (h3 : GE + EH = GH)
  (h4 : GF + FH = GH) :
  EF / GH = 1 / 20 := by 
  sorry

end fraction_length_EF_of_GH_l687_687898


namespace amusement_park_plan_l687_687051

noncomputable def calculate_tickets (ticket_per_attraction : ℕ → ℕ) (tickets_given : ℕ) : ℕ :=
  (ticket_per_attraction 1 + ticket_per_attraction 2 + ticket_per_attraction 3 + ticket_per_attraction 4 + ticket_per_attraction 5) - tickets_given

def ride_time : ℝ := 1.5

theorem amusement_park_plan (entrance_fee lunch_min lunch_max souvenir_min souvenir_max budget : ℝ) 
  (ticket_bundle_cost : ℕ → ℝ) (tickets_needed tickets_given : ℕ) (total_ride_time activity_time : ℝ) :
  entrance_fee + 22 + lunch_min + souvenir_min ≤ budget ∧
  total_ride_time ≤ ride_time := 
begin
  sorry
end

end amusement_park_plan_l687_687051


namespace part1_part2_l687_687994

-- Proof that for 0 < x < 1, x - x^2 < sin x < x
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := 
sorry

-- Proof that if x = 0 is a local maximum of f(x) = cos(ax) - ln(1 - x^2), then a is in the specified range.
theorem part2 (a : ℝ) (h : ∀ x, (cos(a * x) - log(1 - x^2))' (0) = 0 ∧ (cos(a * x) - log(1 - x^2))'' (0) < 0) : 
  a ∈ Set.Ioo (-∞) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) (∞) := 
sorry

end part1_part2_l687_687994


namespace maximum_remains_8sqrt6_l687_687509

def initial_numbers := (1:ℕ) :: (2:ℕ) :: (3:ℕ) :: (4:ℕ) :: (5:ℕ) :: 
                       (6:ℕ) :: (7:ℕ) :: (8:ℕ) :: (9:ℕ) :: (10:ℕ) :: []

def process (l : list ℕ) : list ℕ :=
  match l with
  | a :: b :: c :: rest => (nat.sqrt (a * a + b * b + c * c) :: rest)
  | _ => l

noncomputable def maximum_remaining_number (l : list ℕ) : ℕ :=
  sorry

theorem maximum_remains_8sqrt6 (l : list ℕ) :
  l = initial_numbers → maximum_remaining_number l = 8 * nat.sqrt 6 :=
  sorry

end maximum_remains_8sqrt6_l687_687509


namespace computer_table_cost_price_l687_687979

theorem computer_table_cost_price (CP : ℝ) 
  (h1 : ∃ (x : ℝ), x = 7967) 
  (h2 : ∃ (y : ℝ), y = 1.24): 
  CP = 7967 / 1.24 :=
by
  obtain ⟨x, hx⟩ := h1
  obtain ⟨y, hy⟩ := h2
  rw [hx, hy]
  exact div_eq_of_eq_mul CP sorry

-- Note: This is a simplified version and requires to deal with roundings and inequalities in the actual proof.

end computer_table_cost_price_l687_687979


namespace range_of_a_l687_687761

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l687_687761


namespace meet_time_l687_687109

def sophie_speed : ℝ := 15 -- mph
def alex_speed : ℝ := 20 -- mph
def distance_AB : ℝ := 70 -- miles
def sophie_start_time : ℝ := 7.75 -- 7:45 AM in hours
def alex_start_time : ℝ := 8.33333 -- 8:20 AM in hours ( 8 + 20/60 = 8.33333 )

theorem meet_time : ∃ t : ℝ, sophie_start_time + t = 10.25 ∧
                            sophie_speed * t + alex_speed * (t - (alex_start_time - sophie_start_time)) = distance_AB :=
begin
  -- Assertion: Sophie and Alex meet at 10:15 AM which is 10.25 hours from midnight
  -- Assertion: Their combined equivalent distance equations should sum up to 70 miles
  sorry
end

end meet_time_l687_687109


namespace negation_of_P_l687_687127

-- Defining the original proposition
def P : Prop := ∃ x₀ : ℝ, x₀^2 = 1

-- The problem is to prove the negation of the proposition
theorem negation_of_P : (¬P) ↔ (∀ x : ℝ, x^2 ≠ 1) :=
  by sorry

end negation_of_P_l687_687127


namespace problem_solution_l687_687337

noncomputable def a_seq (n : ℕ) : ℝ := 3 * n - 1
noncomputable def b_seq (n : ℕ) : ℝ := (1 : ℝ) / (3 ^ (n - 1))

theorem problem_solution (n : ℕ) :
  a_seq n = 3 * n - 1 ∧
  (∑ i in Finset.range n, b_seq (i + 1)) = (3 / 2) - (1 / (2 * 3 ^ (n - 1))) := by
  sorry

end problem_solution_l687_687337


namespace factor_expression_l687_687271

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687271


namespace distinct_solutions_of_system_l687_687421

theorem distinct_solutions_of_system (a : ℝ) (x y : ℝ) :
  a ∈ Ioo 4 (4.5) ∨ a ∈ Icc (4.5) (16/3) →
  (log (abs (x + 3)) (a * x + 4 * a) = 2 * log (abs (x + 3)) (x + y) ∧
  x + 1 + sqrt (x^2 + 2 * x + y - 4) = 0) →
  (x = (a-10 + sqrt(a^2 - 4 * a))/2 ∧ y = 5) ∨ 
  (x = (a-10 - sqrt(a^2 - 4 * a))/2 ∧ y = 5) :=
sorry

end distinct_solutions_of_system_l687_687421


namespace distinct_real_roots_eq_one_l687_687344

theorem distinct_real_roots_eq_one : 
  (∃ x : ℝ, |x| - 4/x = (3 * |x|) / x) ∧ 
  ¬∃ x1 x2 : ℝ, 
    x1 ≠ x2 ∧ 
    (|x1| - 4/x1 = (3 * |x1|) / x1) ∧ 
    (|x2| - 4/x2 = (3 * |x2|) / x2) :=
sorry

end distinct_real_roots_eq_one_l687_687344


namespace problem_solution_l687_687228

def expression := (1/3)⁻¹ - 2 * Math.cos (Real.pi / 6) - |2 - Real.sqrt 3| - (4 - Real.pi)⁰

theorem problem_solution : expression = 0 :=
by
  sorry

end problem_solution_l687_687228


namespace largest_value_of_polynomial_l687_687482

noncomputable def P : Polynomial ℝ := sorry

theorem largest_value_of_polynomial :
  (∀ t : ℝ, P.eval t = P.eval 1 * t^2 + P.eval (P.eval 1) * t + P.eval (P.eval (P.eval 1))) →
  P.eval (P.eval (P.eval (P.eval 1))) = 1 / 9 :=
sorry

end largest_value_of_polynomial_l687_687482


namespace susan_ate_candies_l687_687243

theorem susan_ate_candies (candies_tuesday candies_thursday candies_friday candies_left : ℕ) 
  (h_tuesday : candies_tuesday = 3) 
  (h_thursday : candies_thursday = 5) 
  (h_friday : candies_friday = 2) 
  (h_left : candies_left = 4) : candies_tuesday + candies_thursday + candies_friday - candies_left = 6 := by
  sorry

end susan_ate_candies_l687_687243


namespace range_of_a_l687_687748

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687748


namespace ones_digit_of_8_pow_47_l687_687550

theorem ones_digit_of_8_pow_47 : (8^47) % 10 = 2 := 
  sorry

end ones_digit_of_8_pow_47_l687_687550


namespace susan_ate_6_candies_l687_687238

-- Definitions for the conditions
def candies_tuesday : ℕ := 3
def candies_thursday : ℕ := 5
def candies_friday : ℕ := 2
def candies_left : ℕ := 4

-- The total number of candies bought during the week
def total_candies_bought : ℕ := candies_tuesday + candies_thursday + candies_friday

-- The number of candies Susan ate during the week
def candies_eaten : ℕ := total_candies_bought - candies_left

-- Theorem statement
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  unfold candies_eaten total_candies_bought candies_tuesday candies_thursday candies_friday candies_left
  sorry

end susan_ate_6_candies_l687_687238


namespace range_of_a_l687_687778

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687778


namespace factor_expression_l687_687265

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l687_687265


namespace f_properties_l687_687364

def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 4*x + 5

theorem f_properties:
  let tangent_line_at_P := (λ x : ℝ, 3 * x + 1) in
  let extreme_value_at_minus2 := -2 in
  let interval := set.Icc (-3 : ℝ) (1 : ℝ) in
  -- Verify the tangent line at point P(1, f(1))
  (∀ x, f x = x^3 + 2*x^2 - 4*x + 5) ∧
  tangent_line_at_P 1 = f 1 ∧
  3 + 2*2 - 4 = 3 ∧
  -- Extreme value at x = -2
  (extreme_value_at_minus2 = -2) ∧
  -- Maximum value on the interval [-3, 1]
  (∀ x ∈ interval, f x ≤ 13) := 
by
  sorry

end f_properties_l687_687364


namespace given_trig_identity_l687_687814

variable {x : ℂ} {α : ℝ} {n : ℕ}

theorem given_trig_identity (h : x + 1/x = 2 * Real.cos α) : x^n + 1/x^n = 2 * Real.cos (n * α) :=
sorry

end given_trig_identity_l687_687814


namespace acrobats_count_l687_687400

theorem acrobats_count
  (a e c : ℕ)
  (h1 : 2 * a + 4 * e + 2 * c = 58)
  (h2 : a + e + c = 25) :
  a = 11 :=
by
  -- Proof skipped
  sorry

end acrobats_count_l687_687400


namespace tangent_line_eq_at_point_l687_687496

def curve (x : ℝ) : ℝ := x^3

def point : ℝ × ℝ := (2, 8)

theorem tangent_line_eq_at_point :
  ∃ m b : ℝ, 
  (∀ x : ℝ, (curve x = m * (x - point.1) + b) → 
           (m = 12 ∧ b = -16)) :=
begin
  sorry
end

end tangent_line_eq_at_point_l687_687496


namespace minimum_xyz_product_of_point_on_or_within_RQS_l687_687680

theorem minimum_xyz_product_of_point_on_or_within_RQS (A B C D E F P : Type) 
  (equilateral_triangle_ABC : EquilateralTriangle ABC)
  (side_length_ABC : ∀ (a b : ABC), distance a b = 4)
  (points_1_1_condition : distance A E = 1 ∧ distance B F = 1 ∧ distance C D = 1)
  (triangle_RQS : Triangle (Intersection AD BE CF))
  (P_within_or_on_RQS : ∀ (vertex : Vertex triangle_RQS), InVertex P triangle_RQS)
  (distances_xyz : ∀ (side : Side ABC), distance P side = xyz_distance x y z) :
  ∀ (vertex : Vertex triangle_RQS), 
    ∃ (P : Point), (∀ (selected_vertex : vertex), minimum_value (xyz x y z) (P = vertex)) ∧
                  (xyz x y z = \(\frac{648}{2197} \sqrt{3})) :=
sorry

end minimum_xyz_product_of_point_on_or_within_RQS_l687_687680


namespace smallest_positive_b_l687_687912

def periodic_10 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 10) = f x

theorem smallest_positive_b
  (f : ℝ → ℝ)
  (h : periodic_10 f) :
  ∀ x, f ((x - 20) / 2) = f (x / 2) :=
by
  sorry

end smallest_positive_b_l687_687912


namespace modulus_of_z_l687_687795

-- Given conditions
def z : ℂ := (2 + Complex.i) / (1 + Complex.i)^2

-- Prove that the modulus of z is sqrt(5) / 2
theorem modulus_of_z : Complex.abs z = (Real.sqrt 5) / 2 := 
by 
  sorry

end modulus_of_z_l687_687795


namespace leading_digit_not_necessarily_one_l687_687975

-- Define a condition to check if the leading digit of a number is the same
def same_leading_digit (x: ℕ) (n: ℕ) : Prop :=
  (Nat.digits 10 x).head? = (Nat.digits 10 (x^n)).head?

-- Theorem stating the digit does not need to be 1 under given conditions
theorem leading_digit_not_necessarily_one :
  (∃ x: ℕ, x > 1 ∧ same_leading_digit x 2 ∧ same_leading_digit x 3) ∧ 
  (∃ x: ℕ, x > 1 ∧ ∀ n: ℕ, 1 ≤ n ∧ n ≤ 2015 → same_leading_digit x n) :=
sorry

end leading_digit_not_necessarily_one_l687_687975


namespace problem_statement_l687_687356

variable {ℝ : Type} [LinearOrderedField ℝ] [NormedField ℝ] [CompleteSpace ℝ]
variable {f g : ℝ → ℝ}

theorem problem_statement (diff_f : Differentiable ℝ f) (diff_g : Differentiable ℝ g)
    (cond1 : ∀ x > 1, deriv f x > deriv g x) 
    (cond2 : ∀ x < 1, deriv f x < deriv g x) :
    f 2 - f 1 > g 2 - g 1 := 
by sorry

end problem_statement_l687_687356


namespace tan_half_angles_ratio_l687_687897

variables (a b : ℝ)
def c := Real.sqrt (a^2 + b^2)
variables (α β : ℝ)

theorem tan_half_angles_ratio (P : ℝ × ℝ)
  (hyp1 : a ≠ 0) (hyp2 : b ≠ 0)
  (hyp3 : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (hyp4 : ∀ {F1 F2 : ℝ × ℝ},
    F1 = (-c, 0) ∧ F2 = (c, 0) ∧
    ∃ α β, α = Real.arctan (P.1 / P.2) ∧ β = Real.arctan ((P.1 - c) / P.2)) :
  Real.tan (α / 2) / Real.tan (β / 2) = (c - a) / (c + a) :=
sorry

end tan_half_angles_ratio_l687_687897


namespace range_of_a_l687_687776

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l687_687776


namespace number_of_intersections_l687_687930

noncomputable def y1 (x: ℝ) : ℝ := (x - 1) ^ 4
noncomputable def y2 (x: ℝ) : ℝ := 2 ^ (abs x) - 2

theorem number_of_intersections : (∃ x₁ x₂ x₃ x₄ : ℝ, y1 x₁ = y2 x₁ ∧ y1 x₂ = y2 x₂ ∧ y1 x₃ = y2 x₃ ∧ y1 x₄ = y2 x₄ ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :=
sorry

end number_of_intersections_l687_687930


namespace area_of_triangle_ABC_l687_687285

-- Define the conditions
def angle_BAC : ℝ := 90
def angle_ABC : ℝ := 45
def length_AC : ℝ := 14

-- The theorem stating the problem
theorem area_of_triangle_ABC :
  (∠BAC = angle_BAC) → 
  (∠ABC = angle_ABC) → 
  (length AC = length_AC) →
  (area_of_triangle_ABC = 49) :=
by
  sorry

end area_of_triangle_ABC_l687_687285


namespace polynomial_integer_condition_l687_687197

noncomputable def is_integer_valued_polynomial (f : ℤ → ℤ) (n : ℕ) : Prop :=
  ∀ x : ℤ, f x ∈ ℤ

theorem polynomial_integer_condition (f : ℤ → ℤ) (n : ℕ) : 
  (∀ x : ℤ, f x ∈ ℤ) ↔ (∀ k : ℤ, ∀ i in list.range (n + 1), f (k + i) ∈ ℤ) :=
  sorry

end polynomial_integer_condition_l687_687197


namespace value_of_expression_l687_687521

theorem value_of_expression : (0.3 : ℝ)^2 + 0.1 = 0.19 := 
by sorry

end value_of_expression_l687_687521


namespace zero_point_interval_l687_687289

def f (x : Real) : Real := x - 4 * (1/2) ^ x

theorem zero_point_interval : ∃ c ∈ (1 : Real, 2), f c = 0 :=
by
  have f_cont : Continuous f := sorry
  have eval_at_1 : f 1 = -1 := by simp [f]
  have eval_at_2 : f 2 = 1 := by simp [f]
  have interval : f 1 * f 2 < 0 := by norm_num
  sorry

end zero_point_interval_l687_687289


namespace max_participants_l687_687839

-- Define the sets of people who know each subject
constants (A B C : Finset ℕ)

-- Define the given conditions
axiom h1 : A.card = 8
axiom h2 : B.card = 7
axiom h3 : C.card = 11
axiom h4 : (A ∩ B).card ≥ 2
axiom h5 : (B ∩ C).card ≥ 3
axiom h6 : (A ∩ C).card ≥ 4

-- The theorem stating the maximum number of participants
theorem max_participants : (A ∪ B ∪ C).card = 19 := sorry

end max_participants_l687_687839


namespace weight_of_new_person_l687_687573

theorem weight_of_new_person (A : ℝ) : (∀ w : ℝ, (8 * (A + 2)) - (8 * A) + 65 = w ↔ w = 81) :=
by 
  intros w
  split
  { intro h,
    replace h := calc
      (8 * (A + 2)) - (8 * A) + 65 = 8A + 16 - 8A + 65 : by simp
        ... = 81 : by linarith,
    exact h }
  { intro h,
    rw h,
    calc
      (8 * (A + 2)) - (8 * A) + 65 = 8A + 16 - 8A + 65 : by simp
        ... = 81 : by linarith}

end weight_of_new_person_l687_687573


namespace range_of_a_l687_687785

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687785


namespace time_to_cross_l687_687952

def train_length_1 : ℝ := 200  -- Length of the first train in meters
def train_length_2 : ℝ := 160  -- Length of the second train in meters
def speed_kmph_1 : ℝ := 68     -- Speed of the first train in km/h
def speed_kmph_2 : ℝ := 40     -- Speed of the second train in km/h
def angle_elevation : ℝ := 15  -- Angle of elevation in degrees

def speed_mps (kmph : ℝ) : ℝ := kmph * 1000 / 3600

theorem time_to_cross : 
  let speed_1 := speed_mps speed_kmph_1,
      speed_2 := speed_mps speed_kmph_2,
      relative_speed := speed_1 + speed_2,
      total_length := train_length_1 + train_length_2 in
  total_length / relative_speed = 12 := 
by
  -- Definitions and calculations go here...
  sorry

end time_to_cross_l687_687952


namespace distinct_real_roots_eq_one_l687_687343

theorem distinct_real_roots_eq_one : 
  (∃ x : ℝ, |x| - 4/x = (3 * |x|) / x) ∧ 
  ¬∃ x1 x2 : ℝ, 
    x1 ≠ x2 ∧ 
    (|x1| - 4/x1 = (3 * |x1|) / x1) ∧ 
    (|x2| - 4/x2 = (3 * |x2|) / x2) :=
sorry

end distinct_real_roots_eq_one_l687_687343


namespace rationalizing_factor_sqrt2_sub_1_simplify_fraction_compare_sqrt_diff_series_calculation_l687_687903

-- Problem (1): Rationalizing factor
theorem rationalizing_factor_sqrt2_sub_1 : 
  ∃ x, x = (sqrt 2 + 1) :=
sorry

-- Problem (2): Simplify fraction by eliminating square root in the denominator
theorem simplify_fraction : 
  ∃ x, x = (3 + sqrt 6) ∧ (3 / (3 - sqrt 6)) = x :=
sorry

-- Problem (3): Compare √2019 - √2018 and √2018 - √2017
theorem compare_sqrt_diff : 
  (sqrt 2019 - sqrt 2018) < (sqrt 2018 - sqrt 2017) :=
sorry

-- Problem (4): Series calculation
theorem series_calculation : 
  (∑ n in finset.range (2024 - 2 + 1), 1 / (sqrt (n + 2) + sqrt (n + 1))) * (sqrt 2024 + 1) = 2023 :=
sorry

end rationalizing_factor_sqrt2_sub_1_simplify_fraction_compare_sqrt_diff_series_calculation_l687_687903


namespace range_of_a_l687_687720

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687720


namespace exists_infinitely_many_coprime_sums_of_primes_l687_687666

open Nat

def sum_of_primes_less_than (n : ℕ) : ℕ :=
  (Finset.filter (λ p, pnat.prime p ∧ p < n) (Finset.range n)).sum

def coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem exists_infinitely_many_coprime_sums_of_primes :
  ∃ᶠ n in at_top, (n ≥ 3) → coprime (sum_of_primes_less_than n) n :=
sorry

end exists_infinitely_many_coprime_sums_of_primes_l687_687666


namespace integer_root_count_l687_687007

theorem integer_root_count (b : ℝ) :
  (∃ r s : ℤ, r + s = b ∧ r * s = 8 * b) ↔
  b = -9 ∨ b = 0 ∨ b = 9 :=
sorry

end integer_root_count_l687_687007


namespace measure_angle_D_l687_687212

def is_square (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] :=
∀(a b c d : A), dist a b = dist b c ∧ dist b c = dist c d ∧ dist c d = dist d a ∧
               ∀(angleA angleB : ℝ), angleA = 90 ∧ angleB = 90 → angleA = angleB

theorem measure_angle_D 
  (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  (is_rect : ∀ (a b c d : A), dist a b = dist b c ∧ dist b c = dist c d ∧ dist c d = dist d a) 
  (angles_equal : ∀ (A B : ℝ), A = 90 ∧ B = 90) : 
  ∀ (D : ℝ), D = 90 :=
by
  sorry

end measure_angle_D_l687_687212


namespace range_of_a_l687_687779

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687779


namespace find_omega_find_area_triangle_l687_687004

-- Definitions and given conditions
def vector_a (ω x : ℝ) : ℝ × ℝ := (sin (ω * x) + cos (ω * x), sqrt 3 * cos (ω * x))
def vector_b (ω x : ℝ) : ℝ × ℝ := (cos (ω * x) - sin (ω * x), 2 * sin (ω * x))
def f (ω x : ℝ) : ℝ := (vector_a ω x).1 * (vector_b ω x).1 + (vector_a ω x).2 * (vector_b ω x).2
def period (ω : ℝ) : ℝ := π / ω

-- Solutions for the questions
theorem find_omega (ω : ℝ) (h : ω > 0) (T : ℝ) (hx : period ω = π) : ω = 1 := 
by {
  sorry 
}

-- Additional conditions for the triangle problem
variables {A B C a b c : ℝ}

predicate triangle_abc : Prop := 
  f 1 C = 1 ∧
  C = π / 3 ∧ 
  a * a + b * b - 2 * a * b * cos (π/3) = 4 ∧
  sin C + sin (B - A) = 3 * sin (2 * A)

theorem find_area_triangle (A B C a b c : ℝ) (ha : a * a + b * b - 2 * a * b * cos (C) = c * c) 
  (hb : sin C + sin (B - A) = 3 * sin (2 * A)) 
  (hy : triangle_abc) : 
  c = 2 → 
  A = π/2 ∨ true → 
  (abs ((c * b) / 2 * sqrt 3 / 6) = abs (sqrt 3)) := 
by {
  sorry 
}

end find_omega_find_area_triangle_l687_687004


namespace cost_per_quart_proof_l687_687470

-- Definitions of costs and quantities
def cost_of_eggplants (ep_pounds : ℕ) (ep_per_pound : ℝ) := ep_pounds * ep_per_pound
def cost_of_zucchini (zu_pounds : ℕ) (zu_per_pound : ℝ) := zu_pounds * zu_per_pound
def cost_of_tomatoes (to_pounds : ℕ) (to_per_pound : ℝ) := to_pounds * to_per_pound
def cost_of_onions (on_pounds : ℕ) (on_per_pound : ℝ) := on_pounds * on_per_pound
def cost_of_basil (ba_pounds : ℕ) (ba_per_half_pound : ℝ) := ba_pounds * (ba_per_half_pound / 0.5)

-- Total cost of vegetables
def total_cost :=
  cost_of_eggplants 5 2.0 +
  cost_of_zucchini 4 2.0 +
  cost_of_tomatoes 4 3.5 +
  cost_of_onions 3 1.0 +
  cost_of_basil 1 2.5

-- Number of quarts of ratatouille
def number_of_quarts := 4

-- Cost per quart calculation
def cost_per_quart := total_cost / number_of_quarts

-- The proof statement
theorem cost_per_quart_proof : cost_per_quart = 10.0 :=
by
  unfold cost_per_quart total_cost cost_of_eggplants cost_of_zucchini cost_of_tomatoes cost_of_onions cost_of_basil
  norm_num
  sorry

end cost_per_quart_proof_l687_687470


namespace sine_transform_correct_l687_687944

def transform_A : ℝ → ℝ := 
  λ x, 4 * x - (π / 3)

def transform_sin_shift_scale : ℝ → ℝ := 
  λ x, Real.sin (4 * (x + π / 6) - π / 3)

theorem sine_transform_correct :
  ∀ x, (Real.sin (x + π / 6)) = (Real.sin (transform_A x)) :=
by
  intro x
  -- Here we need to prove the transformation correctness
  sorry

end sine_transform_correct_l687_687944


namespace max_colored_cells_l687_687836

theorem max_colored_cells (n : ℕ) (h : n = 100) : 
  ∀ (M : matrix (fin n) (fin n) bool), 
    (∀ i j, M i j → (∃ k, (M i k ∧ ¬∀ k ≠ j, ¬M i k) ∨ (M k j ∧ ¬∀ k ≠ i, ¬M k j))) →
    nat.card {i | ∃ j, M i j} ≤ 198 :=
sorry

end max_colored_cells_l687_687836


namespace octagon_area_ratio_l687_687037

def ratio_of_areas (A B C D E F G H P Q R S T U V W : Point) (s : ℝ) : Prop :=
  regular_octagon A B C D E F G H s ∧
  is_on_side P A B ∧
  is_on_side Q B C ∧
  is_on_side R C D ∧
  is_on_side S D E ∧
  is_on_side T E F ∧
  is_on_side U F G ∧
  is_on_side V G H ∧
  is_on_side W H A ∧
  parallel_lines [A, G] [P, W] [Q, T] [U, R] [S, V] ∧
  equally_spaced_lines [A, G] [P, W] [Q, T] [U, R] [S, V] → 
  area (octagon P Q R S T U V W) / area (octagon A B C D E F G H) = 9 / 16

theorem octagon_area_ratio :
  ∀ (A B C D E F G H P Q R S T U V W : Point) (s : ℝ),
    ratio_of_areas A B C D E F G H P Q R S T U V W s :=
sorry

end octagon_area_ratio_l687_687037


namespace proposition_A_proposition_B_proposition_C_proposition_D_l687_687154

section
variable {z1 z2 : ℂ} {n : ℕ}

/-- Proposition A: If z1 and z2 are conjugate complex numbers, then z1 * z2 is a real number. -/
theorem proposition_A (h_conjugate: z1.conj = z2) : (z1 * z2).im = 0 := by
  sorry

/-- Proposition B: For the imaginary unit i and positive integer n, i^(4n + 3) is not equal to i. -/
theorem proposition_B (h_pos: 0 < n) : Complex.i^(4*n + 3) ≠ Complex.i := by
  sorry

/-- Proposition C: The point corresponding to the complex number -2 - i is in the third quadrant. -/
theorem proposition_C : let p := (-2 : ℂ, -1 : ℂ) in p.1 < 0 ∧ p.2 < 0 := by
  sorry

/-- Proposition D: There exist complex numbers z1 and z2 such that |z1| = |z2| and z1 ≠ z2. -/
theorem proposition_D : ∃ (z1 z2 : ℂ), |z1| = |z2| ∧ z1 ≠ z2 := by
  sorry

end

end proposition_A_proposition_B_proposition_C_proposition_D_l687_687154


namespace GregPPO_reward_correct_l687_687815

-- Define the maximum ProcGen reward
def maxProcGenReward : ℕ := 240

-- Define the maximum CoinRun reward in the more challenging version
def maxCoinRunReward : ℕ := maxProcGenReward / 2

-- Define the percentage reward obtained by Greg's PPO algorithm
def percentageRewardObtained : ℝ := 0.9

-- Calculate the reward obtained by Greg's PPO algorithm
def rewardGregPPO : ℝ := percentageRewardObtained * maxCoinRunReward

-- The theorem to prove the correct answer
theorem GregPPO_reward_correct : rewardGregPPO = 108 := by
  sorry

end GregPPO_reward_correct_l687_687815


namespace power_function_at_2_l687_687809

noncomputable def power_function (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_at_2 (α : ℝ) : power_function 2 α = (real.sqrt 2) / 2 → 
  ∃ α, power_function 4 α = 1 / 2 :=
by
  intro h
  use -1/2
  simp [power_function, h]
  sorry

end power_function_at_2_l687_687809


namespace shirt_cost_l687_687014

variable (J S : ℝ)

def condition1 : Prop := 3 * J + 2 * S = 69
def condition2 : Prop := 2 * J + 3 * S = 66
def cost_of_shirt : Prop := S = 12

theorem shirt_cost (h1 : condition1 J S) (h2 : condition2 J S) : cost_of_shirt J S :=
by
  sorry

end shirt_cost_l687_687014


namespace vector_properties_l687_687377

variables (a b : ℝ^3)

def a := (1, 1, 1: ℝ)
def b := (-1, 0, 2: ℝ)

theorem vector_properties :
  |a| = sqrt 3 ∧
  (a / |a|) = (sqrt 3 / 3, sqrt 3 / 3, sqrt 3 / 3) ∧
  (a ⋅ b) ≠ -1 ∧
  (a ⋅ b) / (|a| * |b|) = sqrt 15 / 15 :=
by
  sorry

end vector_properties_l687_687377


namespace length_of_AB_l687_687369

-- Define the parabola and the conditions
def parabola : Set (ℝ × ℝ) := {p | ∃ x y, y^2 = 4 * x ∧ p = (x, y)}

def focus : ℝ × ℝ := (1, 0)

-- Define the points A and B on the parabola
def on_parabola (p : ℝ × ℝ) : Prop := p ∈ parabola

-- Define the distance from a point to the y-axis (x = 0)
def distance_to_axis (p : ℝ × ℝ) : ℝ := p.1

theorem length_of_AB {A B : ℝ × ℝ} 
  (hA : on_parabola A) (hB : on_parabola B) 
  (dA : distance_to_axis A = 3) (dB : distance_to_axis B = 7)
  (h : ∃ l, F ∈ l ∧ A ∈ l ∧ B ∈ l) :
  dist A B = 10 := 
by sorry

end length_of_AB_l687_687369


namespace binom_sum_eq_l687_687158

theorem binom_sum_eq {n : ℕ} :
  (∑ k in Finset.range(n+1), (2^k) * (Nat.choose n k) * (Nat.choose (n-k) (n-k / 2))) = (Nat.choose (2*n + 1) n) :=
by
  sorry

end binom_sum_eq_l687_687158


namespace rationalize_denominator_sqrt_l687_687467

theorem rationalize_denominator_sqrt (x y : ℝ) (hx : x = 5) (hy : y = 12) :
  Real.sqrt (x / y) = Real.sqrt 15 / 6 :=
by
  rw [hx, hy]
  sorry

end rationalize_denominator_sqrt_l687_687467


namespace solution_eq_293_l687_687072

noncomputable def f : ℕ → ℕ
| 1 := 1
| (2 * n + 1) := f (2 * n) + 1
| (2 * n) := 3 * f n

theorem solution_eq_293 (k l : ℕ) (h1 : k < l) (h2 : f k + f l = 293) : 
  (k = 5 ∧ l = 47) ∨
  (k = 13 ∧ l = 39) ∨
  (k = 7 ∧ l = 45) ∨
  (k = 15 ∧ l = 37) := sorry

end solution_eq_293_l687_687072


namespace toothpick_count_300th_stage_l687_687923

theorem toothpick_count_300th_stage :
  let a1 := 6
  let d := 4
  let n := 300
  let a_n := a1 + (n - 1) * d in
  a_n = 1202 :=
by
  let a1 := 6
  let d := 4
  let n := 300
  let a_n := a1 + (n - 1) * d
  show a_n = 1202
  sorry

end toothpick_count_300th_stage_l687_687923


namespace sum_of_altitudes_l687_687506

theorem sum_of_altitudes (x y : ℝ) (h : 12 * x + 5 * y = 60) :
  let a := (if y = 0 then x else 0)
  let b := (if x = 0 then y else 0)
  let c := (60 / (Real.sqrt (12^2 + 5^2)))
  a + b + c = 281 / 13 :=
sorry

end sum_of_altitudes_l687_687506


namespace inequality_solution_l687_687218

theorem inequality_solution (x : ℝ) : x^2 + x - 20 < 0 ↔ -5 < x ∧ x < 4 := 
by
  sorry

end inequality_solution_l687_687218


namespace range_of_a_l687_687729

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l687_687729


namespace evening_more_than_morning_l687_687098

theorem evening_more_than_morning:
  let remy_morning_sales := 55
  let nick_morning_sales := remy_morning_sales - 6
  let price_per_bottle := 0.5
  let remy_revenue_morning := remy_morning_sales * price_per_bottle
  let nick_revenue_morning := nick_morning_sales * price_per_bottle
  let total_morning_sales := remy_revenue_morning + nick_revenue_morning
  let total_evening_sales := 55
  total_evening_sales - total_morning_sales = 3 :=
begin
  sorry
end

end evening_more_than_morning_l687_687098


namespace max_participants_l687_687841

variable (A B C : Set ℕ) (ps : |A| = 8) (ms : |B| = 7) (pr : |C| = 11)
variable (A_cap_B : |A ∩ B| ≥ 2) (B_cap_C : |B ∩ C| ≥ 3) (A_cap_C : |A ∩ C| ≥ 4)

theorem max_participants (x : ℕ) (A_int_B_int_C : |A ∩ B ∩ C| = x) : A ∪ B ∪ C = 19 := by
  sorry

end max_participants_l687_687841


namespace abs_a_lt_abs_b_sub_abs_c_l687_687317

theorem abs_a_lt_abs_b_sub_abs_c (a b c : ℝ) (h : |a + c| < b) : |a| < |b| - |c| :=
sorry

end abs_a_lt_abs_b_sub_abs_c_l687_687317


namespace solve_for_x_l687_687909

theorem solve_for_x (x : ℝ) : 2^(x-3) * 8^(x-1) = 4^(x+2) ↔ x = 5 :=
by
  sorry

end solve_for_x_l687_687909


namespace possible_pirate_counts_max_coins_after_redistribution_l687_687595

-- Part (a)
theorem possible_pirate_counts (n : ℕ) (a1 : ℕ) (h1 : n > 1) (h2 : n * a1 + (n * (n - 1) / 2) = 2009) :
  n = 7 ∨ n = 41 ∨ n = 49 :=
by
  sorry

-- Part (b)
theorem max_coins_after_redistribution (a1 : ℕ) (h : a1 = 284) :
  let a := [a1, a1 + 1, a1 + 2, a1 + 3, a1 + 4, a1 + 5, a1 + 6]
  in ∃ n', n' ∈ a ∧ n' = 1996 :=
by
  sorry

end possible_pirate_counts_max_coins_after_redistribution_l687_687595


namespace shortest_distance_ln_curve_to_line_l687_687513

open Real

noncomputable def f (x : ℝ) : ℝ := log (2 * x - 1)

theorem shortest_distance_ln_curve_to_line : 
  ∃ (P : ℝ × ℝ), P = (1, 0) ∧ f P.1 = P.2 ∧ 
  let distance := abs (2 * P.1 - P.2 + 3) / sqrt (2 ^ 2 + (-1) ^ 2) in 
  distance = sqrt 5 :=
by sorry

end shortest_distance_ln_curve_to_line_l687_687513


namespace students_neither_math_physics_l687_687455

theorem students_neither_math_physics (total_students math_students physics_students both_students : ℕ) 
  (h1 : total_students = 120)
  (h2 : math_students = 80)
  (h3 : physics_students = 50)
  (h4 : both_students = 15) : 
  total_students - (math_students - both_students + physics_students - both_students + both_students) = 5 :=
by
  -- Each of the hypotheses are used exactly as given in the conditions.
  -- We omit the proof as requested.
  sorry

end students_neither_math_physics_l687_687455


namespace assign_positive_numbers_possible_l687_687907

noncomputable def assign_positive_numbers_to_regions (n : ℕ) (lines : fin n → PlanarLine) 
  (no_parallel: ∀ i j : fin n, i ≠ j → ¬parallel (lines i) (lines j)) 
  (no_concurrent: ∀ i j k : fin n, i ≠ j → j ≠ k → i ≠ k → ¬concurrent (lines i) (lines j) (lines k)) : 
  Prop :=
∃ (assignments : Finset Region → ℕ), 
  ∀ l : PlanarLine, sum_of_regions_left l assignments = sum_of_regions_right l assignments

theorem assign_positive_numbers_possible 
  (n : ℕ) 
  (lines : fin n → PlanarLine) 
  (no_parallel: ∀ i j : fin n, i ≠ j → ¬parallel (lines i) (lines j)) 
  (no_concurrent: ∀ i j k : fin n, i ≠ j → j ≠ k → i ≠ k → ¬concurrent (lines i) (lines j) (lines k)) : 
  assign_positive_numbers_to_regions n lines no_parallel no_concurrent :=
sorry

end assign_positive_numbers_possible_l687_687907


namespace shirt_price_correct_l687_687515

noncomputable def sweater_price := 43.885
noncomputable def shirt_price := 36.455
noncomputable def total_cost := 80.34
noncomputable def price_difference := 7.43

theorem shirt_price_correct :
  (shirt_price + sweater_price = total_cost) ∧ (sweater_price - shirt_price = price_difference) →
  shirt_price = 36.455 :=
by {
  intros h,
  sorry
}

end shirt_price_correct_l687_687515


namespace abcd_formula_l687_687914

-- Conditions definitions
def f (x : ℝ) := a * x^2 + b * x + c
def f_shifted (x : ℝ) := 5 * x^2 + 2 * x + 6

-- Theorem statement
theorem abcd_formula (a b c : ℝ) (h1 : ∀ x : ℝ, f (x + 2) = f_shifted x) : a + 2 * b + 3 * c = 35 :=
by
  sorry

end abcd_formula_l687_687914


namespace p_divisible_by_1979_l687_687318

theorem p_divisible_by_1979 (p q : ℕ)
  (h : (p : ℚ) / q = ∑ i in (Finset.range 1320).filter (λ n, n % 2 = 0), (-1) ^ (i + 1) / i) :
  1979 ∣ p :=
begin
  sorry
end

end p_divisible_by_1979_l687_687318


namespace possible_values_of_quadratic_l687_687137

theorem possible_values_of_quadratic (x : ℝ) (h : x^2 - 5 * x + 4 < 0) : 10 < x^2 + 4 * x + 5 ∧ x^2 + 4 * x + 5 < 37 :=
by
  sorry

end possible_values_of_quadratic_l687_687137


namespace a_and_b_are_skew_l687_687889

-- Define the basic properties and conditions
variables (Point Line Plane : Type)
variables (lies_on : Line → Plane → Prop) (intersects_at : Line → Plane → Point → Prop)
variables (A : Point) (a b : Line) (α β : Plane)

-- A point is not on a line
def not_on_line (P : Point) (l : Line) : Prop := ¬ ∃ (x : Point), x = P ∧ lies_on x l

-- Inserting the given conditions
axiom line_a_in_plane_α : lies_on a α
axiom line_b_intersects_plane_α_at_A : intersects_at b α A
axiom point_A_not_on_line_a : not_on_line A a

-- The statement to prove
theorem a_and_b_are_skew :
  (¬ (∃ (P : Point), P ≠ A ∧ lies_on P a ∧ lies_on P b)) → ¬ (∃ (x : Line), (lies_on x α) ∧ (lies_on x β) ∧ x = a ∧ x = b) := 
sorry

end a_and_b_are_skew_l687_687889


namespace not_equivalent_to_0_point_000045_l687_687966

theorem not_equivalent_to_0_point_000045 :
  let A := 4.5 * 10^(-5)
  let B := (9:ℝ) / 2 * 10^(-5)
  let C := 45 * 10^(-7)
  let D := 1 / 22500
  let E := 45 / 10^6
  C ≠ 0.000045 := by
    sorry

end not_equivalent_to_0_point_000045_l687_687966


namespace range_of_a_l687_687771

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l687_687771


namespace invalid_votes_percentage_is_15_l687_687410

def percentage_invalid_votes (total_votes valid_votes polled_votes_by_A : ℕ) : ℝ :=
  let percentage_valid_votes := (polled_votes_by_A:ℝ) / (total_votes:ℝ)
  let percentage_invalid_votes := 1 - percentage_valid_votes / 0.8
  percentage_invalid_votes * 100

theorem invalid_votes_percentage_is_15 (total_votes : ℕ) (polled_votes_by_A : ℕ) :
  total_votes = 560000 ∧ polled_votes_by_A = 380800 → percentage_invalid_votes total_votes (polled_votes_by_A * 100 / 80) polled_votes_by_A = 15 := by
  sorry

end invalid_votes_percentage_is_15_l687_687410


namespace factor_expression_l687_687259

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687259


namespace altitude_minus_one_eq_l687_687851

def side_length : ℝ := 6
def altitude (s : ℝ) : ℝ := (real.sqrt 3 / 2) * s
def radius_inscribed (h : ℝ) : ℝ := (1 / 2) * h
def x (h : ℝ) : ℝ := h - 1

theorem altitude_minus_one_eq :
  let h := altitude side_length in
  let _ := radius_inscribed h in
  x h = 3 * real.sqrt 3 - 1 :=
by
  sorry

end altitude_minus_one_eq_l687_687851


namespace transformed_quadratic_l687_687094

theorem transformed_quadratic (a b c n x : ℝ) (h : a * x^2 + b * x + c = 0) :
  a * x^2 + n * b * x + n^2 * c = 0 :=
sorry

end transformed_quadratic_l687_687094


namespace square_of_radius_l687_687184

theorem square_of_radius 
  (AP PB CQ QD : ℝ) 
  (hAP : AP = 25)
  (hPB : PB = 35)
  (hCQ : CQ = 30)
  (hQD : QD = 40) 
  : ∃ r : ℝ, r^2 = 13325 := 
sorry

end square_of_radius_l687_687184


namespace exponent_multiplication_l687_687152

theorem exponent_multiplication :
  (10^(3/4)) * (10^(-0.25)) * (10^(1.5)) = 10^2 :=
by sorry

end exponent_multiplication_l687_687152


namespace problem_statement_l687_687350

-- Definitions of conditions
def a : ℝ := 5
def b : ℝ := 3 - 2 * Real.sqrt 2

-- Main goal: prove that the expression equals -1
theorem problem_statement : 
  (2 + Real.sqrt 2) / (2 - Real.sqrt 2) = (a + (1 - b)) → 
  (∃ (a : ℝ), (floor ((2 + Real.sqrt 2) / (2 - Real.sqrt 2)) = a) ∧ 
  (∃ (b : ℝ), (fract ((2 + Real.sqrt 2) / (2 - Real.sqrt 2)) = 1 - b) ∧ 
  ((b - 1) * (5 - b)) / Real.sqrt (5^2 - 3^2) = -1)) :=
by
  sorry

end problem_statement_l687_687350


namespace area_of_quadrilateral_l687_687036

noncomputable def AreaQuad (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] := 
  distance A B = 4 ∧ -- AB = 4
  distance B C = 5 ∧ -- BC = 5
  distance C D = 6 ∧ -- CD = 6
  angle B C = 120 ∧  -- m∠C = 120°
  angle A B = 120    -- m∠B = 120°

theorem area_of_quadrilateral {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :
  AreaQuad A B C D → 
  (area A B C + area B C D) = 12.5 * Real.sqrt 3 :=
by
  intros h
  sorry -- Detailed proof goes here

end area_of_quadrilateral_l687_687036


namespace sherry_loaves_l687_687908

theorem sherry_loaves (batter_loaves : ℕ) (bananas_needed : ℕ) (total_bananas : ℕ) :
  batter_loaves = 3 → bananas_needed = 1 → total_bananas = 33 →
  (total_bananas * batter_loaves / bananas_needed) = 99 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end sherry_loaves_l687_687908


namespace range_of_a_l687_687777

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l687_687777


namespace G_even_l687_687673

-- Definitions as conditions from the problem
variable (a : ℝ) (F : ℝ → ℝ)
variable (h1 : a > 0) (h2 : a ≠ 1)
variable (hF : ∀ x, F (-x) = - F x)

-- Definition of G(x)
def G (x : ℝ) : ℝ := F x * (1 / (a^x - 1) + 1/2)

-- Theorem that G(x) is an even function
theorem G_even : ∀ x, G a F x = G a F (-x) :=
by
  sorry

end G_even_l687_687673


namespace geom_seq_result_l687_687789

-- Given conditions about the geometric sequence
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_a1_a2 : a 1 + a 2 = 4 / 9)
variable (h_a3_to_a6 : a 3 + a 4 + a 5 + a 6 = 40)

-- Defining the geometric sequence with the first term a_1 and common ratio q
def geom_seq (a_first q : ℝ) (n : ℕ) : ℝ := a_first * q^n

-- Problem statement to be proven
theorem geom_seq_result :
  (∀ n, a n = geom_seq (a 1) q n) →
  (q > 0) →
  a 1 = 1 / 9 →
  q = 3 →
  (a 7 + a 8 + a 9) / 9 = 117 :=
begin
  intros h_geom_seq h_q_pos h_a1 h_q,
  sorry,
end

end geom_seq_result_l687_687789


namespace maximum_edges_no_triangle_l687_687486

theorem maximum_edges_no_triangle (A : Fin 6 → Point) 
  (h_no_collinear : ∀ (i j k : Fin 6), ¬collinear (A i) (A j) (A k)) 
  (h_no_triangle : ∀ (i j k : Fin 6), ¬(connected (A i) (A j) ∧ connected (A j) (A k) ∧ connected (A k) (A i))):
  ∃ E : ℕ, E = 9 ∧ (∀ (i j : Fin 6), i ≠ j → (connected (A i) (A j) ↔ E ≤ 9)) :=
begin
  sorry
end

end maximum_edges_no_triangle_l687_687486


namespace range_of_a_l687_687747

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687747


namespace gcd_of_sum_and_fraction_l687_687016

theorem gcd_of_sum_and_fraction (p : ℕ) (a b : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1)
  (hcoprime : Nat.gcd a b = 1) : Nat.gcd (a + b) ((a^p + b^p) / (a + b)) = p := 
sorry

end gcd_of_sum_and_fraction_l687_687016


namespace range_of_a_l687_687781

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687781


namespace range_of_a_l687_687706

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687706


namespace cyclist_speed_l687_687566

def distance_m : ℝ := 750 -- Distance in meters
def time_min : ℝ := 2 + 30 / 60 -- Time in minutes (2 minutes 30 seconds)
def time_hr : ℝ := time_min / 60 -- Convert time to hours
def distance_km : ℝ := distance_m / 1000 -- Convert distance to kilometers

theorem cyclist_speed :
  (distance_km / time_hr ≈ 18) :=
by
  sorry

end cyclist_speed_l687_687566


namespace odd_integers_solution_l687_687880

theorem odd_integers_solution (a b c d k m : ℤ) (h_odd_a : odd a) (h_odd_b : odd b) (h_odd_c : odd c) (h_odd_d : odd d)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : a * d = b * c) (h6 : a + d = 2 * k) (h7 : b + c = 2 * m) :
  a = 1 := by
  sorry

end odd_integers_solution_l687_687880


namespace count_integers_abs_le_pisq_l687_687817

theorem count_integers_abs_le_pisq : 
  { x : ℤ | abs x ≤ Real.pi ^ 2 }.finite.to_finset.card = 19 := 
by
  sorry

end count_integers_abs_le_pisq_l687_687817


namespace paper_cost_l687_687575
noncomputable section

variables (P C : ℝ)

theorem paper_cost (h : 100 * P + 200 * C = 6.00) : 
  20 * P + 40 * C = 1.20 :=
sorry

end paper_cost_l687_687575


namespace factor_expression_l687_687274

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687274


namespace segments_parallel_and_equal_l687_687171

def pentagon_points (A B C D E P Q R S M N : Type) (k1 k2 : ℝ) : Prop :=
  (∃ (AP PB DR RC BQ QC ES SD PM MR SN NQ : Type),
  (AP / PB = k1) ∧ (DR / RC = k1) ∧ (BQ / QC = k2) ∧ (ES / SD = k2) ∧ 
  (PM / MR = k2) ∧ (SN / NQ = k1) ∧ 
  -- Placeholder for the segment MN as a definition
  (∃ (MN AE : Type), MN // Placeholder for parallel relation and length equality:
    MN ∥ AE ∧ MN = AE / (k1 + 1) * (k2 + 1)))

-- Formalize the claim to be proven
theorem segments_parallel_and_equal (A B C D E P Q R S M N : Type) (k1 k2 : ℝ) 
  (h : pentagon_points A B C D E P Q R S M N k1 k2) : 
  ∃ (MN AE : Type), 
    MN ∥ AE ∧ 
    MN = (AE / ((k1 + 1) * (k2 + 1))) :=
  sorry

end segments_parallel_and_equal_l687_687171


namespace gain_percent_l687_687567

theorem gain_percent (CP SP : ℝ) (hCP : CP = 20) (hSP : SP = 35) : 
  (SP - CP) / CP * 100 = 75 :=
by
  rw [hCP, hSP]
  sorry

end gain_percent_l687_687567


namespace all_nat_numbers_appear_once_l687_687461

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom a_pos_irrational : 0 < a ∧ ¬(∃ (m n : ℕ), a = m / n)
axiom b_pos_irrational : 0 < b ∧ ¬(∃ (m n : ℕ), b = m / n)
axiom ab_condition : (1 / a) + (1 / b) = 1

theorem all_nat_numbers_appear_once :
  (∀ k : ℕ, ∃ m n : ℕ, k = ⌊m * a⌋ ∨ k = ⌊n * b⌋) :=
begin
  sorry
end

end all_nat_numbers_appear_once_l687_687461


namespace range_of_a_l687_687712

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l687_687712


namespace factor_expression_l687_687263

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687263


namespace four_people_in_five_chairs_l687_687034

theorem four_people_in_five_chairs : ∃ (n : ℕ), n = 5 * 4 * 3 * 2 ∧ n = 120 :=
by
  use 120
  split 
  · norm_num
  · refl

end four_people_in_five_chairs_l687_687034


namespace sequence_terminates_l687_687602

-- Definition of the sequence construction rules
def next_term (a : ℕ) : Option ℕ :=
  if a % 10 ≤ 5 then
    if a / 10 = 0 then none else some (a / 10)
  else
    some (9 * a)

-- The main theorem stating that the sequence cannot be infinite
theorem sequence_terminates (a0 : ℕ) : ∃ n, next_term^[n] (some a0) = none :=
by
  sorry

end sequence_terminates_l687_687602


namespace p_neither_sufficient_nor_necessary_l687_687877

theorem p_neither_sufficient_nor_necessary (x y : ℝ) :
  (x > 1 ∧ y > 1) ↔ ¬((x > 1 ∧ y > 1) → (x + y > 3)) ∧ ¬((x + y > 3) → (x > 1 ∧ y > 1)) :=
by
  sorry

end p_neither_sufficient_nor_necessary_l687_687877


namespace game_ends_after_36_rounds_l687_687090

noncomputable def game_tokens (A B C D : ℕ) (rounds : ℕ) :=
  ∃ rounds : ℕ, (A = 16 ∧ B = 15 ∧ C = 14 ∧ D = 13) ∧ 
  (∀ r < rounds, 

    (let (max_player, max_tokens) := 
      if A >= B ∧ A >= C ∧ A >= D then ('A', A)
      else if B >= A ∧ B >= C ∧ B >= D then ('B', B)
      else if C >= A ∧ C >= B ∧ C >= D then ('C', C)
      else ('D', D) 

    in (if max_player = 'A' then
        (A - (r * 4) = 0 ∨ B - (r * 4) = 0 ∨ C - (r * 4) = 0 ∨ D - (r * 4) = 0)
    else
        (max_tokens - 4 * r = 0)))

theorem game_ends_after_36_rounds : game_tokens 16 15 14 13 36 :=
begin
  sorry
end

end game_ends_after_36_rounds_l687_687090


namespace chessboard_game_winner_l687_687193

theorem chessboard_game_winner (m n : ℕ) (initial_position : ℕ × ℕ) :
  (m * n) % 2 = 0 → (∃ A_wins : Prop, A_wins) ∧ 
  (m * n) % 2 = 1 → (∃ B_wins : Prop, B_wins) :=
by
  sorry

end chessboard_game_winner_l687_687193


namespace second_tap_empties_cistern_l687_687187

theorem second_tap_empties_cistern (t_fill: ℝ) (x: ℝ) (t_net: ℝ) : 
  (1 / 6) - (1 / x) = (1 / 12) → x = 12 := 
by
  sorry

end second_tap_empties_cistern_l687_687187


namespace neither_prime_nor_composite_probability_l687_687829

/-- Prove that the probability of drawing a number from 1 to 96 that is neither prime nor composite
     equals 1/96, given that numbers 1 to 96 are written on 96 pieces of paper and one piece is picked at random. -/
theorem neither_prime_nor_composite_probability : 
  let s : Finset ℕ := Finset.range 97 in
  let neither_prime_nor_composite (n : ℕ) : bool :=
    (n = 0) || (n = 1)
  in
  (∑ i in s, (ite (neither_prime_nor_composite i) 1 0)) / (s.card) = 1 / 96 :=
by
  sorry

end neither_prime_nor_composite_probability_l687_687829


namespace range_of_a_l687_687692

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687692


namespace unique_solution_solve_inequation_l687_687868

open Complex

-- Conditions setup
variables (a b : ℝ) (z : ℂ)
axiom z_not_real : z ∉ ℝ
axiom condition_ab : abs (a - b) = abs (a + b - 2 * z)

-- Part (a): Prove the equation has a unique solution
theorem unique_solution (x : ℝ) : 
  abs (z - (a : ℂ))^x + abs (conj z - (b : ℂ))^x = abs (a - b)^x ↔ x = 2 :=
sorry

-- Part (b): Solve the inequation
theorem solve_inequation (x : ℝ) : 
  abs (z - ↑a)^x + abs (conj z - ↑b)^x ≤ abs (a - b)^x ↔ x ≥ 2 :=
sorry

end unique_solution_solve_inequation_l687_687868


namespace range_of_a_l687_687745

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687745


namespace fraction_of_shaded_quilt_block_l687_687603

theorem fraction_of_shaded_quilt_block :
  let total_area := 16
  let shaded_area := 8
  shaded_area / total_area = (1 / 2) := 
by
  let total_area := 16
  let shaded_area := 8
  show shaded_area / total_area = (1 / 2)
  by sorry

end fraction_of_shaded_quilt_block_l687_687603


namespace range_of_quadratic_func_l687_687807

def quadratic_func (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem range_of_quadratic_func : 
  set.range (quadratic_func) = set.Icc 1 17 :=
by 
  sorry

end range_of_quadratic_func_l687_687807


namespace how_many_candies_eaten_l687_687245

variable (candies_tuesday candies_thursday candies_friday candies_left : ℕ)

def total_candies (candies_tuesday candies_thursday candies_friday : ℕ) : ℕ :=
  candies_tuesday + candies_thursday + candies_friday

theorem how_many_candies_eaten (h_tuesday : candies_tuesday = 3)
                               (h_thursday : candies_thursday = 5)
                               (h_friday : candies_friday = 2)
                               (h_left : candies_left = 4) :
  (total_candies candies_tuesday candies_thursday candies_friday) - candies_left = 6 :=
by
  sorry

end how_many_candies_eaten_l687_687245


namespace factor_expression_l687_687276

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687276


namespace runners_regroup_time_l687_687669

theorem runners_regroup_time :
  ∃ t : ℕ,
    (∃ t1 : ℕ, t1 = 1200 / 1.5 ∧ 1.5 * t1 = 1200) ∧
    (∃ t2 : ℕ, t2 = 1200 / 2.5 ∧ 2.5 * t2 = 1200) ∧
    (∃ t3 : ℕ, t3 = 1200 / 3 ∧ 3 * t3 = 1200) ∧
    (∃ t : ℕ, t = Nat.lcm (Nat.lcm 400 240) 200) ∧
    t = 1200 := by
  -- Proof goes here
  sorry

end runners_regroup_time_l687_687669


namespace largest_consecutive_interesting_numbers_l687_687078

def is_interesting (k : ℕ) : Prop :=
  let primes := (finset.range (k + 1)).map (λ n, (Nat.prime (n + 1)).to_nth_prime)
  (primes.prod % k = 0)

theorem largest_consecutive_interesting_numbers :
  (∀ n : ℕ, is_interesting n →
    is_interesting (n+1) → is_interesting (n+2) →
    (n+3) = 3) :=
begin
  sorry
end

end largest_consecutive_interesting_numbers_l687_687078


namespace range_of_a_l687_687749

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687749


namespace find_second_sum_l687_687161

theorem find_second_sum (S : ℝ) (x : ℝ) (h : S = 2704 ∧ 24 * x / 100 = 15 * (S - x) / 100) : (S - x) = 1664 := 
  sorry

end find_second_sum_l687_687161


namespace problem_I_problem_II_l687_687790

variable {θ : Real}
variable {m : Real}

-- Conditions
def valid_angle (θ : Real) := 0 < θ ∧ θ < π / 2
noncomputable def sin_cos_sum (θ : Real) := sin θ + cos θ = (sqrt 3 + 1) / 2
noncomputable def sin_cos_prod (θ : Real) (m : Real) := sin θ * cos θ = m / 2

-- Problem
theorem problem_I (θ : Real) (h1 : valid_angle θ) (h2 : sin_cos_sum θ) :
  (sin θ / (1 - 1 / tan θ) + cos θ / (1 - tan θ)) = (sqrt 3 + 1) / 2 := sorry

theorem problem_II (θ : Real) (m : Real) (h1 : valid_angle θ) (h2 : sin_cos_sum θ) (h3 : sin_cos_prod θ m) :
  m = sqrt 3 / 2 ∧ (θ = π / 6 ∨ θ = π / 3) := sorry

end problem_I_problem_II_l687_687790


namespace duration_of_loan_l687_687904

def simple_interest_duration (P : ℕ) (SI : ℕ) (R : ℕ) : ℕ :=
  (SI * 100) / (P * R)

theorem duration_of_loan (P SI R : ℕ) (H1 : P = 1100) (H2 : SI = 704) (H3 : R = 8) :
  simple_interest_duration P SI R = 8 :=
by
  rw [H1, H2, H3]
  simp [simple_interest_duration]
  sorry

end duration_of_loan_l687_687904


namespace no_airline_has_50_cities_connected_l687_687027

noncomputable def cities := 200
noncomputable def airlines := 8
axiom connected_flight : sorry -- This describes that every pair of cities is connected by flights operated by one of these eight airlines.

theorem no_airline_has_50_cities_connected :
  ∃ airline_idx : ℕ, airline_idx < airlines →
  ∀ subset_cities : set ℕ, subset_cities.size > 50 →
  ¬ ∀ (c₁ c₂ ∈ subset_cities), connected_flight (airline_idx, c₁, c₂) :=
sorry

end no_airline_has_50_cities_connected_l687_687027


namespace concyclicity_APST_l687_687408

variable {Point : Type}

-- Given an acute triangle ABC with circumcenter O and orthocenter H.
variable (A B C O H P T S : Point)
variable [Geometry Point]

-- A set of hypotheses stating:
-- 1. ABC is an acute triangle.
-- 2. O is the circumcenter of triangle ABC.
-- 3. H is the orthocenter of triangle ABC.
-- 4. P is the midpoint of segment OH.
-- 5. T is the midpoint of segment AO.
-- 6. The perpendicular bisector of AO intersects BC at point S.
-- Together imply that points A, P, S, T are concyclic.

axiom acute_triangle (h : Triangle A B C) : AcuteTriangle A B C
axiom circumcenter (h : Triangle A B C) (O) : Circumcenter A B C O
axiom orthocenter (h : Triangle A B C) (H) : Orthocenter A B C H
axiom midpoint_OH (P : Point) (O H : Point) : P = Midpoint O H
axiom midpoint_AO (T : Point) (A O : Point) : T = Midpoint A O
axiom perp_bisector_AO (S : Point) (A O B C : Point) : S = PerpBisectorIntersection A O B C

theorem concyclicity_APST : ∀ (A B C O H P T S : Point),
  AcuteTriangle A B C →
  Circumcenter A B C O →
  Orthocenter A B C H →
  P = Midpoint O H →
  T = Midpoint A O →
  S = PerpBisectorIntersection A O B C →
  Concyclic A P S T :=
by
  intros
  sorry

end concyclicity_APST_l687_687408


namespace part1_part2_l687_687229

noncomputable def complex_prod1 : ℂ := (1 - 2 * I) * (3 + 4 * I) * (-2 + I)
theorem part1 : complex_prod1 = -20 + 15 * I :=
by sorry

noncomputable def complex_div1 : ℂ := (1 + 2 * I) / (3 - 4 * I)
theorem part2 : complex_div1 = (-1 / 5) + (2 / 5) * I :=
by sorry

end part1_part2_l687_687229


namespace thief_speed_l687_687206

-- Define constants for the problem
def initial_distance : ℕ := 160 -- initial distance in meters
def speed_policeman : ℕ := 10 -- speed in km/hr
def distance_thief : ℕ := 640 -- distance thief runs before being caught in meters

-- Define the speed of the thief we need to prove
def speed_thief : ℕ := 8 -- speed in km/hr

-- Prove that the speed of the thief is 8 km/hr given the conditions
theorem thief_speed : speed_thief = 8 :=
by
  -- Convert distances from meters to kilometers
  let d_km := initial_distance / 1000 + distance_thief / 1000
  let d_t_km := distance_thief / 1000
  -- Set up the proportion
  have proportion : d_km / d_t_km = speed_policeman / speed_thief,
    sorry
  -- Solve for speed_thief
  have solve_v : speed_thief = 8,
    sorry
  -- Conclude the theorem
  exact solve_v

end thief_speed_l687_687206


namespace factor_expression_l687_687264

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l687_687264


namespace product_divisible_by_odd_prime_l687_687439

variables {m n : ℕ} (h_pos : 0 < n) (h_mn : m > n)

noncomputable def x (k : ℕ) : ℕ := (m + k) / (n + k)

theorem product_divisible_by_odd_prime (h_int : ∀ k ∈ finset.range (n + 1), ∃ x_k ∈ finset.range (m + k), x k = x_k):
  ∃ p : ℕ, prime p ∧ p % 2 = 1 ∧ p ∣ ∏ k in finset.range (n + 1), x k - 1 :=
sorry

end product_divisible_by_odd_prime_l687_687439


namespace platform_length_l687_687207

theorem platform_length (train_length : ℕ) (train_speed_kmh : ℚ) (pass_time_s : ℕ) 
    (H1 : train_length = 360)
    (H2 : train_speed_kmh = 45)
    (H3 : pass_time_s = 40) :
    ∃ platform_length : ℕ, platform_length = 140 := 
by 
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  have speed_in_ms : train_speed_ms = 12.5, by norm_num[H2]
  let total_distance := train_speed_ms * pass_time_s
  have total_distance_calculated : total_distance = 500, by norm_num[speed_in_ms, H3]
  let platform_length := total_distance - train_length
  have platform_length_calculated : platform_length = 140, by norm_num[total_distance_calculated, H1]
  use platform_length
  exact platform_length_calculated

end platform_length_l687_687207


namespace triangle_area_l687_687061

-- Define the vectors
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-6, 3)

-- Function to calculate the determinant of a 2x2 matrix
def det_2x2 (x1 x2 : ℝ) (y1 y2 : ℝ) : ℝ :=
  x1 * y2 - x2 * y1

-- Calculate the area of the triangle implicitly
def area_triangle (a b : ℝ × ℝ) : ℝ :=
  (det_2x2 a.1 b.1 a.2 b.2).abs / 2

-- The statement to be proved
theorem triangle_area : area_triangle a b = 3 := by
  sorry

end triangle_area_l687_687061


namespace no_concat_perfect_cube_l687_687463

theorem no_concat_perfect_cube (n : ℕ) (h₁ : n = 2001) :
  ∀ nums (h₂ : nums = list.range (n + 1) \ [0]), 
  let concatenated_num := nums.foldl (λ acc x, acc * 10^(nat.log10 x + 1) + x) 0 in
  (concatenated_num % 9 ≠ 0 ∧ concatenated_num % 9 ≠ 1 ∧ concatenated_num % 9 ≠ 8) :=
by 
  sorry

end no_concat_perfect_cube_l687_687463


namespace nina_total_cost_l687_687085

-- Define the cost of the first pair of shoes
def first_pair_cost : ℕ := 22

-- Define the cost of the second pair of shoes
def second_pair_cost : ℕ := first_pair_cost + (first_pair_cost / 2)

-- Define the total cost for both pairs of shoes
def total_cost : ℕ := first_pair_cost + second_pair_cost

-- The formal statement of the problem
theorem nina_total_cost : total_cost = 55 := by
  sorry

end nina_total_cost_l687_687085


namespace estimate_correct_l687_687600

noncomputable def estimatedTangerines : ℝ :=
  let μ := 90
  let σ := 2
  let P := Real.normDist μ σ
  let probability := P.cdf 96
  10000 * probability

theorem estimate_correct :
  estimatedTangerines = 9987 :=
begin
  sorry
end

end estimate_correct_l687_687600


namespace area_of_rectangular_garden_l687_687927

-- Definitions based on conditions
def width : ℕ := 15
def length : ℕ := 3 * width
def area : ℕ := length * width

-- The theorem we want to prove
theorem area_of_rectangular_garden : area = 675 :=
by sorry

end area_of_rectangular_garden_l687_687927


namespace find_a_of_extreme_at_1_l687_687499

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - x - Real.log x

theorem find_a_of_extreme_at_1 :
  (∃ a : ℝ, ∃ f' : ℝ -> ℝ, (f' x = 3 * a * x^2 - 1 - 1/x) ∧ f' 1 = 0) →
  ∃ a : ℝ, a = 2 / 3 :=
by
  sorry

end find_a_of_extreme_at_1_l687_687499


namespace correct_parentheses_l687_687367

theorem correct_parentheses : (1 * 2 * 3 + 4) * 5 = 50 := by
  sorry

end correct_parentheses_l687_687367


namespace arithmetic_progression_cube_non_square_l687_687611

def is_arithmetic_progression (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a n + (m - n) * d = a m ∧ n < m

theorem arithmetic_progression_cube_non_square (a : ℕ → ℕ) (d : ℕ) (h_arith_prog : is_arithmetic_progression a) 
    (h_cube : ∃ n : ℕ, ∃ k : ℕ, a n = k^3) :
    ∃ m : ℕ, ∃ l : ℕ, l^3 = a m ∧ ¬(∃ (k : ℕ), l^2 = k) :=
sorry

end arithmetic_progression_cube_non_square_l687_687611


namespace perfect_square_product_l687_687512

theorem perfect_square_product (A : Finset ℕ) (hA : A.card = 2016)
  (hprime : ∀ a ∈ A, ∀ p, Nat.Prime p → p ∣ a → p < 30) :
  ∃ a b c d ∈ A, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ Nat.IsSquare (a * b * c * d) :=
by
  sorry

end perfect_square_product_l687_687512


namespace points_collinear_l687_687678

noncomputable theory

open EuclideanGeometry

variables {A B C H M N P Q : Point}

def orthocenter (A B C H : Point) : Prop := is_orthocenter A B C H

def on_segment (P A B : Point) : Prop := is_on_segment P A B

def circle_with_diameter (A B : Point) (P : Point) : Prop := ∃ O R, is_circle O R ∧ diameter (A B) O R ∧ is_on_circle P O R

def collinear (A B C : Point) : Prop := is_collinear A B C

theorem points_collinear 
  (ABC_triangle : Triangle A B C)
  (H_is_orthocenter : orthocenter A B C H)
  (M_on_segment_AB : on_segment M A B)
  (N_on_segment_AC : on_segment N A C)
  (P_on_circle_BN : circle_with_diameter B N P)
  (Q_on_circle_CM : circle_with_diameter C M Q) :
  collinear P Q H :=
sorry

end points_collinear_l687_687678


namespace bela_wins_iff_n_is_odd_l687_687616

theorem bela_wins_iff_n_is_odd (n : ℕ) (h : n > 6) : (∃ x, 0 ≤ x ∧ x ≤ n) ∧ (∀ x y, x ≠ y → |x - y| > 1.5 ∨ y = n + 1) →
  (∃ bela_wins, bela_wins ↔ n % 2 = 1) :=
  sorry

end bela_wins_iff_n_is_odd_l687_687616


namespace factor_expression_l687_687266

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l687_687266


namespace bisect_BK_l687_687592

variable {A B C K O D : Point}
variable (tri : Triangle ABC)

-- Condition: A circle is inscribed in triangle ABC
def inscribed_circle (tri : Triangle ABC) : Circle 
#check inscribed_circle 

-- Condition: The inscribed circle touches side AC at point K
def touch_point (c : Circle) (P Q : Point) : Point 
#check touch_point

def incircle_touch_AC := touch_point (inscribed_circle tri) A C

-- Condition: D is the midpoint of side AC
def midpoint (P Q : Point) : Point 
#check midpoint

def midpoint_AC := midpoint A C

-- Condition: The center of the inscribed circle is O
def circle_center (c : Circle) : Point 
#check circle_center

def center_incircle := circle_center (inscribed_circle tri)

-- Question: The line connecting midpoint of AC (D) with center O bisects the segment BK
theorem bisect_BK 
  (h1 : incircle_touch_AC = K)
  (h2 : midpoint_AC = D) 
  (h3 : center_incircle = O) : 
  midpoint (B K) = intersection_line (line D O) (line B K) := 
by 
  sorry

end bisect_BK_l687_687592


namespace total_puppies_tyler_l687_687543

/-- Proves that the total number of puppies Tyler now has is 235 given the conditions. -/
theorem total_puppies_tyler (total_dogs : ℕ) 
  (dogs_with_5_5_puppies : ℕ)
  (dogs_with_8_puppies : ℕ)
  (additional_puppies : ℕ → ℝ) 
  (additional_dogs : ℕ) 
  (total_puppies : ℝ) 
  (total_dogs = 35)
  (dogs_with_5_5_puppies = 15)
  (dogs_with_8_puppies = 10) 
  (additional_dogs = 5)
  (additional_puppies additional_dogs = 2.5)
  (total_puppies = 235) :
  (dogs_with_5_5_puppies * 5.5 + dogs_with_8_puppies * 8 + (total_dogs - dogs_with_5_5_puppies - dogs_with_8_puppies) * 6 + additional_dogs * additional_puppies additional_dogs) = total_puppies :=
by
  sorry

end total_puppies_tyler_l687_687543


namespace bacon_cost_is_correct_l687_687429

variable (total_money : ℝ)
variable (hummus_price : ℝ)
variable (chicken_price : ℝ)
variable (vegetable_price : ℝ)
variable (apple_price : ℝ)
variable (num_apples : ℕ)

def cost_of_bacon (total_money : ℝ) (hummus_price : ℝ) (chicken_price : ℝ)
  (vegetable_price : ℝ) (apple_price : ℝ) (num_apples : ℕ) : ℝ :=
  let total_cost := (2 * hummus_price) + chicken_price + vegetable_price in
  let remaining_money := total_money - total_cost in
  let apples_cost := num_apples * apple_price in
  remaining_money - apples_cost

theorem bacon_cost_is_correct :
  ∀ (total_money hummus_price chicken_price vegetable_price apple_price : ℝ)
    (num_apples : ℕ),
    total_money = 60 →
    hummus_price = 5 →
    chicken_price = 20 →
    vegetable_price = 10 →
    apple_price = 2 →
    num_apples = 5 →
    cost_of_bacon total_money hummus_price chicken_price vegetable_price apple_price num_apples = 10 :=
by
  intros
  unfold cost_of_bacon
  simp
  sorry

end bacon_cost_is_correct_l687_687429


namespace determine_n_for_square_l687_687235

theorem determine_n_for_square (n : ℕ) : (∃ a : ℕ, 5^n + 4 = a^2) ↔ n = 1 :=
by
-- The proof will be included here, but for now, we just provide the structure
sorry

end determine_n_for_square_l687_687235


namespace apples_add_up_l687_687906

variables (Sean_initial_apples Susan_apples total_apples : ℕ)

def Sean_has_initial_apples : Prop :=
  Sean_initial_apples = 9

def Susan_gives_apples : Prop :=
  Susan_apples = 8

def total_after_giving : Prop :=
  total_apples = Sean_initial_apples + Susan_apples

theorem apples_add_up
  (h1 : Sean_has_initial_apples)
  (h2 : Susan_gives_apples)
  : total_after_giving :=
by sorry

end apples_add_up_l687_687906


namespace max_cardinality_l687_687660

-- Define the conditions as Lean statements
def is_valid_set (S : Set ℕ) : Prop := (∀ x ∈ S, 1 ≤ x ∧ x ≤ 100) ∧
  (∀ (a b c : ℕ), a ≠ b → a ∈ S → b ∈ S → c ∈ S → Nat.gcd (a + b) c = 1 ∨ Nat.gcd (a + b) c > 1)

-- Define the theorem stating the maximum cardinality of S
theorem max_cardinality (S : Set ℕ) : is_valid_set S → Set.card S ≤ 50 := sorry

end max_cardinality_l687_687660


namespace liars_count_l687_687522

-- Definitions
def person : Type := ℕ -- Represent each person as a natural number from 0 to 99.
def is_knight (p : person) : Prop := sorry
def is_liar (p : person) : Prop := sorry
def is_eccentric (p : person) : Prop := sorry

axiom table_size : ∀ p : person, p < 100

axiom roles_distinct (p : person) : 
  (is_knight p ∧ ¬ is_liar p ∧ ¬ is_eccentric p) ∨
  (¬ is_knight p ∧ is_liar p ∧ ¬ is_eccentric p) ∨
  (¬ is_knight p ∧ ¬ is_liar p ∧ is_eccentric p)

axiom knight_says (p : person) 
  (next : person) : 
  is_knight p → next = (p + 1) % 100 → is_liar next

axiom liar_says (p : person) 
  (next : person) : 
  is_liar p → next = (p + 1) % 100 → ¬ is_liar next

axiom eccentric_with_liar (p : person) 
  (prev next : person) : 
  is_eccentric p → prev = (p + 99) % 100 → next = (p + 1) % 100 → is_liar prev → is_liar next

axiom eccentric_with_knight (p : person) 
  (prev next : person) : 
  is_eccentric p → prev = (p + 99) % 100 → next = (p + 1) % 100 → is_knight prev → ¬ is_liar next

axiom eccentric_with_eccentric (p : person) 
  (prev next : person) : 
  is_eccentric p → prev = (p + 99) % 100 → next = (p + 1) % 100 → is_eccentric prev → sorry

-- Theorem statement
theorem liars_count (n : ℕ) : (n = 33) ∨ (n = 34) :=
sorry

end liars_count_l687_687522


namespace coeff_of_monomial_l687_687919

-- Define the given monomial
def monomial (a b : ℝ) : ℝ := -((2 * Real.pi * a * b^2) / 3)

-- Statement of the problem: Prove the coefficient of the monomial is -2π/3
theorem coeff_of_monomial (a b : ℝ) : (monomial a b) = (- (2 * Real.pi / 3) * (a * b^2)) :=
by
  -- Proof would follow here
  sorry

end coeff_of_monomial_l687_687919


namespace cos_arcsin_of_fraction_l687_687627

theorem cos_arcsin_of_fraction : ∀ x, x = 8 / 17 → x ∈ set.Icc (-1:ℝ) 1 → Real.cos (Real.arcsin x) = 15 / 17 :=
by
  intros x hx h_range
  rw hx
  have h : (x:ℝ)^2 + Real.cos (Real.arcsin x)^2 = 1 := Real.sin_sq_add_cos_sq (Real.arcsin x)
  sorry

end cos_arcsin_of_fraction_l687_687627


namespace minimum_mn_l687_687805

noncomputable def f (x : ℝ) (n m : ℝ) : ℝ := Real.log x - n * x + Real.log m + 1

noncomputable def f' (x : ℝ) (n : ℝ) : ℝ := 1/x - n

theorem minimum_mn (m n x_0 : ℝ) (h_m : m > 1) (h_tangent : 2*x_0 - (f x_0 n m) + 1 = 0) :
  mn = e * ((1/x_0 - 1) ^ 2 - 1) :=
sorry

end minimum_mn_l687_687805


namespace permutations_mod_1000_l687_687873

def count_permutations (str : List Char) (cond1 cond2 cond3 : List Char → Prop) : Nat :=
  sorry -- Placeholder for the actual calculation function

theorem permutations_mod_1000 :
  let str := ['D', 'D', 'D', 'D', 'D', 'D', 'E', 'E', 'E', 'E', 'E', 'F', 'F', 'F', 'F', 'F', 'F', 'F']
  let cond1 (sub : List Char) := ∀ c ∈ sub.take 5, c ≠ 'D'
  let cond2 (sub : List Char) := ∀ c ∈ sub.drop 5. take 5, c ≠ 'E'
  let cond3 (sub : List Char) := ∀ c ∈ sub.drop 10, c ≠ 'F'
  let M := count_permutations str cond1 cond2 cond3
  M % 1000 = 406 :=
by
  sorry -- Placeholder for the actual proof

end permutations_mod_1000_l687_687873


namespace probability_X_greater_than_0_l687_687902

noncomputable def normal_pdf (μ σ x : ℝ) : ℝ :=
  (1 / (σ * real.sqrt (2 * real.pi))) * real.exp (-(x - μ)^2 / (2 * σ^2))

def X_max_pdf_value : Prop :=
  ∃ (μ σ : ℝ), (normal_pdf μ σ (real.sqrt 2) = (1 / (2 * real.sqrt real.pi)))

axiom P_one_sigma (μ σ : ℝ) : 
  (P_normal (μ - σ) (μ + σ) = 0.6827)

axiom P_two_sigma (μ σ : ℝ) :
  (P_normal (μ - 2 * σ) (μ + 2 * σ) = 0.9545)

theorem probability_X_greater_than_0 (μ σ : ℝ) (hμ : μ = real.sqrt 2) (hσ : σ = real.sqrt 2) :
  ∃ p, p = 0.84135 ∧ P_normal (0, ∞) = p :=
begin
  sorry
end

end probability_X_greater_than_0_l687_687902


namespace prob_at_least_one_hit_prob_at_least_three_hits_prob_no_more_than_one_hit_l687_687136

-- Definition of the probabilities
def p : ℝ := 0.8
def q : ℝ := 0.2
def n : ℕ := 4

-- Binomial probability formula
noncomputable def binom (n k : ℕ) : ℝ :=
  ((nat.choose n k : ℕ) : ℝ) * p^k * q^(n-k)

-- Part (a): Probability of at least one hit
theorem prob_at_least_one_hit : ∑ k in finset.range (n+1), if k = 0 then binom n k else 0 = 0.9984 := sorry

-- Part (b): Probability of at least three hits
theorem prob_at_least_three_hits : ∑ k in finset.range (n+1), if k >= 3 then binom n k else 0 = 0.8192 := sorry

-- Part (c): Probability of no more than one hit
theorem prob_no_more_than_one_hit : ∑ k in finset.range (n+1), if k <= 1 then binom n k else 0 = 0.0272 := sorry

end prob_at_least_one_hit_prob_at_least_three_hits_prob_no_more_than_one_hit_l687_687136


namespace rose_bush_arrangement_possible_l687_687435

theorem rose_bush_arrangement_possible (bushes : Set ℕ) (rows : Set (Set ℕ)) :
  bushes.card = 15 ∧ (∀ r ∈ rows, r.card = 5) ∧ rows.card = 6 →
  (∃ arrangement : bushes ⊆ finset.univ, rows.card = 6) :=
by
  sorry

end rose_bush_arrangement_possible_l687_687435


namespace matchstick_triangles_l687_687963

theorem matchstick_triangles (perimeter : ℕ) (h_perimeter : perimeter = 30) : 
  ∃ n : ℕ, n = 17 ∧ 
  (∀ a b c : ℕ, a + b + c = perimeter → a > 0 → b > 0 → c > 0 → 
                a + b > c ∧ a + c > b ∧ b + c > a → 
                a ≤ b ∧ b ≤ c → n = 17) := 
sorry

end matchstick_triangles_l687_687963


namespace school_student_difference_l687_687101

theorem school_student_difference 
  (B_A : ℕ)      -- Number of boys in School A
  (G_B : ℕ)      -- Number of girls in School B
  (A_m B_m : ℕ)  -- Number of male students respectively in Schools A and B
  (A_f G_A : ℕ)  -- Number of female students respectively in Schools A and B
  (h1 : B_A = 217)
  (h2 : G_B = 196)
  (h3 : B_A + B_m = A_f + G_B) :
  |B_m - A_f| = 21 := 
by
  sorry

end school_student_difference_l687_687101


namespace range_of_a_l687_687765

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l687_687765


namespace pens_difference_proof_l687_687071

variables (A B M N X Y : ℕ)

-- Initial number of pens for Alex and Jane
def Alex_initial (A : ℕ) := A
def Jane_initial (B : ℕ) := B

-- Weekly multiplication factors for Alex and Jane
def Alex_weekly_growth (X : ℕ) := X
def Jane_weekly_growth (Y : ℕ) := Y

-- Number of pens after 4 weeks
def Alex_after_4_weeks (A X : ℕ) := A * X^4
def Jane_after_4_weeks (B Y : ℕ) := B * Y^4

-- Proving the difference in the number of pens
theorem pens_difference_proof (hM : M = A * X^4) (hN : N = B * Y^4) :
  M - N = (A * X^4) - (B * Y^4) :=
by sorry

end pens_difference_proof_l687_687071


namespace main_theorem_l687_687465

open Polynomial

noncomputable theory

variables {P Q : Polynomial ℂ} {z : ℂ}

def is_even_polynomial (P : Polynomial ℂ) : Prop :=
  ∀ z : ℂ, P(z) = P(-z)

def exists_even_composite (P : Polynomial ℂ) : Prop :=
  ∃ Q : Polynomial ℂ, ∀ z : ℂ, P(z) = Q(z) * Q(-z)

theorem main_theorem (P : Polynomial ℂ) : 
  is_even_polynomial P ↔ exists_even_composite P := 
by 
  sorry

end main_theorem_l687_687465


namespace find_m_of_symmetric_l687_687311

-- Define the function
def f (x : ℝ) (m : ℝ) : ℝ := 4^x / (4^x + m)

-- Define the symmetry condition
def symmetric_about (a b : ℝ) (f : ℝ → ℝ) := ∀ x, f(x) + f(2*a - x) = 2*b

theorem find_m_of_symmetric (m : ℝ) :
  symmetric_about (1/2) (1/2) (λ x, f x m) → m = 2 :=
by
  intro h
  -- skipping the actual proof, assume the answer is correct
  sorry

end find_m_of_symmetric_l687_687311


namespace medium_ceiling_lights_count_l687_687035

theorem medium_ceiling_lights_count (S M L : ℕ) 
  (h1 : L = 2 * M) 
  (h2 : S = M + 10) 
  (h_bulbs : S + 2 * M + 3 * L = 118) : M = 12 :=
by
  -- Proof omitted
  sorry

end medium_ceiling_lights_count_l687_687035


namespace arithmetic_sequence_sum_l687_687060

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a_1 d : ℤ) 
  (h1: S 3 = (3 * a_1) + (3 * (2 * d) / 2))
  (h2: S 7 = (7 * a_1) + (7 * (6 * d) / 2)) :
  S 5 = (5 * a_1) + (5 * (4 * d) / 2) := by
  sorry

end arithmetic_sequence_sum_l687_687060


namespace sara_total_cents_l687_687905

def number_of_quarters : ℕ := 11
def value_per_quarter : ℕ := 25

theorem sara_total_cents : number_of_quarters * value_per_quarter = 275 := by
  sorry

end sara_total_cents_l687_687905


namespace coefficient_x3_in_binomial_expansion_l687_687040

theorem coefficient_x3_in_binomial_expansion :
  ∃ c : ℕ, (c = 19600 * 2^47) ∧ 
            (∀ (x : ℤ), (x + 2)^50 = ∑ k in finset.range 51, (nat.choose 50 k * 2^(50 - k) * x^k) ∧
            (nat.choose 50 3 * 2^47 = c)) :=
by
  sorry

end coefficient_x3_in_binomial_expansion_l687_687040


namespace john_can_ensure_win_in_coin_game_l687_687541

theorem john_can_ensure_win_in_coin_game :
  ∀ (table_coins bill_coins john_coins : ℕ), 
  (bill_coins = 74) → (john_coins = 74) → table_coins = 0 →
  (∀ n, n ≤ bill_coins ∧ n ≤ john_coins →
            (1 ≤ n ∧ n ≤ 3) →
            ((bill_coins - n, john_coins - n) → 
                 bill_coins + john_coins + table_coins = 99 →
                     (john turns → table_coins = 100) → john wins)
sorry

end john_can_ensure_win_in_coin_game_l687_687541


namespace range_of_a_l687_687772

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l687_687772


namespace cannot_be_2009_l687_687156

theorem cannot_be_2009 (a b c : ℕ) (h : b * 1234^2 + c * 1234 + a = c * 1234^2 + a * 1234 + b) : (b * 1^2 + c * 1 + a ≠ 2009) :=
by
  sorry

end cannot_be_2009_l687_687156


namespace second_quadrilateral_property_l687_687950

-- Define points and segments in the context of inscribed quadrilaterals
variables {Point : Type} [MetricSpace Point]
variables {A B C D E F G H : Point}
variables {AB BC CD DA EF FG GH HE : ℝ}

-- Inscribed quadrilaterals condition
axiom inscribed_in_circle (A B C D : Point) : True
axiom inscribed_in_circle' (E F G H : Point) : True

-- Given condition for the first quadrilateral
axiom first_quadrilateral_condition : AB * CD = BC * DA

-- Define the segments in terms of lengths
variable (AB: ℝ := dist A B)
variable (BC: ℝ := dist B C)
variable (CD: ℝ := dist C D)
variable (DA: ℝ := dist D A)
variable (EF: ℝ := dist E F)
variable (FG: ℝ := dist F G)
variable (GH: ℝ := dist G H)
variable (HE: ℝ := dist H E)

-- Define the proof problem
theorem second_quadrilateral_property :
  inscribed_in_circle A B C D →
  inscribed_in_circle' E F G H →
  AB * CD = BC * DA →
  EF * GH = FG * HE :=
by
  sorry

end second_quadrilateral_property_l687_687950


namespace radius_increase_l687_687964

/-- Proving that the radius increases by 7/π inches when the circumference increases from 50 inches to 64 inches -/
theorem radius_increase (C₁ C₂ : ℝ) (h₁ : C₁ = 50) (h₂ : C₂ = 64) :
  (C₂ / (2 * Real.pi) - C₁ / (2 * Real.pi)) = 7 / Real.pi :=
by
  sorry

end radius_increase_l687_687964


namespace ones_digit_of_8_pow_47_l687_687554

theorem ones_digit_of_8_pow_47 :
  (8^47) % 10 = 2 :=
by
  sorry

end ones_digit_of_8_pow_47_l687_687554


namespace hyperbola_eccentricity_l687_687325

variable (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
variable (F1 P F2 : ℝ × ℝ)
variable (hyp : (P.1 * P.1 / (a * a)) - (P.2 * P.2 / (b * b)) = 1)
variable (angle_condition : (dist P F1) * (dist P F2) = 0)
variable (dist_condition : dist P F1 = 2 * dist P F2)
variable (dist_definition: (dist P F1) - (dist P F2) = 2 * a)
variable (dist_foci : 2 * c = dist F1 F2)

theorem hyperbola_eccentricity : (dist P F1)^2 + (dist P F2)^2 = (dist F1 F2)^2 ∧ (5 * a^2 = c^2) -> (c / a = sqrt 5) := by
  sorry

end hyperbola_eccentricity_l687_687325


namespace sum_perimeters_is_64_l687_687116

variable (x : ℝ)

def area_first_square : ℝ := x^2 + 12 * x + 36

def area_second_square : ℝ := 4 * x^2 - 12 * x + 9

def side_first_square : ℝ := x + 6

def side_second_square : ℝ := 2 * x - 3

def perimeter_first_square : ℝ := 4 * (x + 6)

def perimeter_second_square : ℝ := 4 * (2 * x - 3)

def sum_perimeters : ℝ := perimeter_first_square x + perimeter_second_square x

theorem sum_perimeters_is_64 (hx : x = 4.333333333333333) :
  sum_perimeters x = 64 := by
  rw [← hx]
  dsimp [sum_perimeters, perimeter_first_square, perimeter_second_square, side_first_square, side_second_square]
  sorry -- Proof follows here

end sum_perimeters_is_64_l687_687116


namespace range_of_a_l687_687718

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687718


namespace ball_hits_ground_l687_687922

theorem ball_hits_ground : 
  ∃ t : ℚ, -4.9 * t^2 + 4 * t + 10 = 0 ∧ t = 10 / 7 :=
by sorry

end ball_hits_ground_l687_687922


namespace product_nonreal_roots_l687_687296

namespace PolynomialProof

noncomputable def polynomial_equation : Polynomial ℂ := 
  Polynomial.monomial 4 1 - Polynomial.monomial 3 6 
  + Polynomial.monomial 2 15 - Polynomial.monomial 1 20

theorem product_nonreal_roots :
  let p := polynomial_equation - Polynomial.C 2009 in
  ∃ (a b : ℂ), ∀ (x : ℂ), 
  x^4 - 6*x^3 + 15*x^2 - 20*x = 2009 →
  (x = a ∨ x = b) →
  (x = 2 + Complex.i * Complex.root 2033 4 ∨ x = 2 - Complex.i * Complex.root 2033 4) →
  a * b = 4 + Complex.root 2033 2 :=
sorry

end PolynomialProof

end product_nonreal_roots_l687_687296


namespace AI_midpoint_of_AHAM_l687_687679

variable {X B C A_H A_I A_M : Type} [MetricSpace X] [MetricSpace B] [MetricSpace C] [MetricSpace A_H] [MetricSpace A_I] [MetricSpace A_M]

-- Assumptions
variables (h_triangle : is_triangle X B C)
           (h_distinct : A_H ≠ A_I ∧ A_I ≠ A_M ∧ A_H ≠ A_M)
           (h_orthocenter : is_orthocenter X A_H B C)
           (h_incenter : is_incenter X A_I B C)
           (h_centroid : is_centroid X A_M B C)
           (h_parallel : is_parallel (line A_H A_M) (line B C))

-- Statement to prove
theorem AI_midpoint_of_AHAM (h_triangle h_distinct h_orthocenter h_incenter h_centroid h_parallel) :
          is_midpoint A_I A_H A_M := sorry

end AI_midpoint_of_AHAM_l687_687679


namespace tug_of_war_tournament_selection_l687_687916

theorem tug_of_war_tournament_selection (strengths : Fin 20 → ℕ) (h_distinct : Function.Injective strengths) :
  ∃ representative : Fin 20, ∀ k, tourn_match winning_ties_strengths representative k ∧ representative ≠ (weakest_player strengths) :=
by
  sorry

end tug_of_war_tournament_selection_l687_687916


namespace sum_prob_27_l687_687593

def die1 : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
def die2 : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

theorem sum_prob_27 : 
  (∃ n1 ∈ die1, ∃ n2 ∈ die2, n1 + n2 = 27) ∧ 
  (Finset.card die1 * Finset.card die2 = 400) → 
  (∃ n1 ∈ die1, ∃ n2 ∈ die2, n1 + n2 = 27) / (Finset.card die1 * Finset.card die2) = 1 / 40 := 
by 
  sorry

end sum_prob_27_l687_687593


namespace range_of_a_l687_687763

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l687_687763


namespace probability_points_one_unit_apart_l687_687106

theorem probability_points_one_unit_apart :
  let total_points := 16
  let total_pairs := (total_points * (total_points - 1)) / 2
  let favorable_pairs := 12
  let probability := favorable_pairs / total_pairs
  probability = (1 : ℚ) / 10 :=
by
  sorry

end probability_points_one_unit_apart_l687_687106


namespace range_of_a_l687_687767

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l687_687767


namespace ball_distribution_l687_687095

theorem ball_distribution : 
  let ways := (∃ x y z : ℕ, x + y + z = 20 ∧ x ≥ 1 ∧ y ≥ 2 ∧ z ≥ 3) in 
  ways = (∑ i in (Finset.range (20 - 1 + 1)).image ((· + 1 : ℕ → ℕ) ∘ Finset.filter (λ n => n ≥ 2).card), 1) :=
begin
  sorry
end

end ball_distribution_l687_687095


namespace journey_total_distance_l687_687860

-- Define the conditions
def miles_already_driven : ℕ := 642
def miles_to_drive : ℕ := 558

-- The total distance of the journey
def total_distance : ℕ := miles_already_driven + miles_to_drive

-- Prove that the total distance of the journey equals 1200 miles
theorem journey_total_distance : total_distance = 1200 := 
by
  -- here the proof would go
  sorry

end journey_total_distance_l687_687860


namespace ratio_of_largest_to_smallest_root_in_geometric_progression_l687_687469

theorem ratio_of_largest_to_smallest_root_in_geometric_progression 
    (a b c d : ℝ) (r s t : ℝ) 
    (h_poly : 81 * r^3 - 243 * r^2 + 216 * r - 64 = 0)
    (h_geo_prog : r > 0 ∧ s > 0 ∧ t > 0 ∧ ∃ (k : ℝ),  k > 0 ∧ s = r * k ∧ t = s * k) :
    ∃ (k : ℝ), k = r^2 ∧ s = r * k ∧ t = s * k := 
sorry

end ratio_of_largest_to_smallest_root_in_geometric_progression_l687_687469


namespace circle_equation_exists_l687_687793

theorem circle_equation_exists :
  ∃ (x_c y_c r : ℝ), 
  x_c > 0 ∧ y_c > 0 ∧ 0 < r ∧ r < 5 ∧ (∀ x y : ℝ, (x - x_c)^2 + (y - y_c)^2 = r^2) :=
sorry

end circle_equation_exists_l687_687793


namespace range_of_a_l687_687702

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687702


namespace coin_stack_height_l687_687404

def alpha_thickness : ℝ := 1.25
def beta_thickness : ℝ := 2.00
def gamma_thickness : ℝ := 0.90
def delta_thickness : ℝ := 1.60
def stack_height : ℝ := 18.00

theorem coin_stack_height :
  (∃ n : ℕ, stack_height = n * beta_thickness) ∨ (∃ n : ℕ, stack_height = n * gamma_thickness) :=
sorry

end coin_stack_height_l687_687404


namespace sin_coterminal_angle_l687_687236

theorem sin_coterminal_angle :
  sin (7 * Real.pi / 3) = sqrt 3 / 2 :=
by sorry

end sin_coterminal_angle_l687_687236


namespace abs_simplify_l687_687831

theorem abs_simplify (x : ℝ) (h : x < 0) : |3 * x + real.sqrt (x^2)| = -2 * x :=
by
  sorry

end abs_simplify_l687_687831


namespace min_S_n_is_10_l687_687409

noncomputable def arithmetic_sequence_minimizes_S (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : Prop :=
  (d > 0) ∧ (a 10 + a 11 < 0) ∧ (a 10 * a 11 < 0) ∧ (∀ n, S n = ∑ i in finset.range n, a i)

theorem min_S_n_is_10 (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h : arithmetic_sequence_minimizes_S a d S) : ∃ n, n = 10 ∧ ∀ m, S n ≤ S m :=
by {
  -- Proof is skipped
  sorry
}

end min_S_n_is_10_l687_687409


namespace range_of_a_l687_687298

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ Real.pi / 2 →
    (x + 3 + 2 * Real.sin θ * Real.cos θ) ^ 2 +
    (x + a * Real.sin θ + a * Real.cos θ) ^ 2 ≥ 1 / 8) ↔
  (a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6) :=
by
  sorry

end range_of_a_l687_687298


namespace total_books_proof_l687_687588

def initial_books : ℝ := 41.0
def added_books_first : ℝ := 33.0
def added_books_next : ℝ := 2.0

theorem total_books_proof : initial_books + added_books_first + added_books_next = 76.0 :=
by
  sorry

end total_books_proof_l687_687588


namespace smallest_n_for_purple_l687_687622

-- The conditions as definitions
def red := 18
def green := 20
def blue := 22
def purple_cost := 24

-- The mathematical proof problem statement
theorem smallest_n_for_purple : 
  ∃ n : ℕ, purple_cost * n = Nat.lcm (Nat.lcm red green) blue ∧
            ∀ m : ℕ, (purple_cost * m = Nat.lcm (Nat.lcm red green) blue → m ≥ n) ↔ n = 83 := 
by
  sorry

end smallest_n_for_purple_l687_687622


namespace linear_function_through_point_decreasing_l687_687091

theorem linear_function_through_point_decreasing (k b : ℝ) (h1 : k ≠ 0) (h2 : b = 2) (h3 : k < 0) :
  ∃ (f : ℝ → ℝ), f = fun x => k * x + b ∧ f 0 = 2 ∧ ∀ x1 x2, x1 < x2 → f x1 > f x2 :=
sorry

end linear_function_through_point_decreasing_l687_687091


namespace eleven_y_minus_x_l687_687167

theorem eleven_y_minus_x (x y : ℤ) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 11 * y - x = 1 := by
  sorry

end eleven_y_minus_x_l687_687167


namespace problem_statement_l687_687003

def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) : ℝ := 3 * x - 2

theorem problem_statement : f (g (f (g 2))) = 7189058 := by
  sorry

end problem_statement_l687_687003


namespace solve_log_eq_l687_687910

theorem solve_log_eq (x : ℝ) : log 3 (9^x - 4) = x + 1 ↔ x = log 3 4 :=
sorry

end solve_log_eq_l687_687910


namespace union_of_sets_l687_687883

def M := {x : ℝ | -1 < x ∧ x < 1}
def N := {x : ℝ | x^2 - 3 * x ≤ 0}

theorem union_of_sets : M ∪ N = {x : ℝ | -1 < x ∧ x ≤ 3} :=
by sorry

end union_of_sets_l687_687883


namespace polynomial_divisibility_l687_687012

theorem polynomial_divisibility (p q : ℝ) :
    (∀ x, x = -2 ∨ x = 3 → (x^6 - x^5 + x^4 - p*x^3 + q*x^2 - 7*x - 35) = 0) →
    (p, q) = (6.86, -36.21) :=
by
  sorry

end polynomial_divisibility_l687_687012


namespace range_of_a_l687_687721

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687721


namespace circle_tangent_sum_radii_l687_687185

theorem circle_tangent_sum_radii :
  let r1 := 6 + 2 * Real.sqrt 6
  let r2 := 6 - 2 * Real.sqrt 6
  r1 + r2 = 12 :=
by
  sorry

end circle_tangent_sum_radii_l687_687185


namespace min_k_for_cube_root_difference_l687_687531

theorem min_k_for_cube_root_difference (cards : Finset ℕ) (h : cards = Finset.range 2017) (selected : Finset ℕ) (h_selected : selected.card = 13) : 
  ∃ (a b : ℕ), a ∈ selected ∧ b ∈ selected ∧ a ≠ b ∧ (|real.cbrt a - real.cbrt b| < 1) :=
by
  sorry

end min_k_for_cube_root_difference_l687_687531


namespace susan_ate_candies_l687_687241

theorem susan_ate_candies (candies_tuesday candies_thursday candies_friday candies_left : ℕ) 
  (h_tuesday : candies_tuesday = 3) 
  (h_thursday : candies_thursday = 5) 
  (h_friday : candies_friday = 2) 
  (h_left : candies_left = 4) : candies_tuesday + candies_thursday + candies_friday - candies_left = 6 := by
  sorry

end susan_ate_candies_l687_687241


namespace trajectory_equation_l687_687792

def point (α : Type*) := α × α

def parabola_equation (x : ℝ) : ℝ := 2 * x^2 + 1

def section_formula (r1 r2 : ℝ) (A P : point ℝ) : point ℝ :=
  ((r1 * A.1 + r2 * P.1) / (r1 + r2), (r1 * A.2 + r2 * P.2) / (r1 + r2))

theorem trajectory_equation :
  ∀ (P : point ℝ) (A : point ℝ) (r1 r2 : ℝ),
    P.2 = parabola_equation P.1 →
    A = (0, 1) →
    (r1 = 2) →
    (r2 = 1) →
    let M := section_formula r1 r2 A P in
    M.2 = 6 * M.1^2 + 1 := 
by
  sorry

end trajectory_equation_l687_687792


namespace max_min_values_of_function_l687_687294

theorem max_min_values_of_function :
  let f := (fun x : ℝ => 3 * x^4 + 4 * x^3 + 34)
  ∃ (max min : ℝ), (∀ x ∈ Icc (-2 : ℝ) 1, f x ≤ max) ∧ (∀ x ∈ Icc (-2 : ℝ) 1, min ≤ f x) ∧
                   max = f (-2) ∧ max = 50 ∧
                   min = f (-1) ∧ min = 33 :=
by
  let f := (fun x : ℝ => 3 * x^4 + 4 * x^3 + 34)
  use 50, 33
  have h₁ : ∀ x, f' x = 12 * x^2 * (x + 1), from sorry,
  have critical_points : {x | f' x = 0} = {0, -1}, from sorry,
  -- Check values at endpoints and critical points
  have h_f_neg2 : f (-2) = 50 := by simp [f],
  have h_f_1 : f 1 = 41 := by simp [f],
  have h_f_neg1 : f (-1) = 33 := by simp [f],
  have h_f_0 : f 0 = 34 := by simp [f],
  split,
  -- Proving max value
  { intro x,
    intro hx,
    by_cases hx0 : x = -2 ∨ x = 1 ∨ x = -1 ∨ x = 0,
    any_goals { finish },
    -- f(-2) = 50, rest points have lower values
    show f x ≤ 50, from sorry,},
  -- Proving min value
  split,
  { intro x,
    intro hx,
    by_cases hx0 : x = -2 ∨ x = 1 ∨ x = -1 ∨ x = 0,
    any_goals { finish },
    -- f(-1) = 33, rest points have higher values
    show 33 ≤ f x, from sorry,},
  -- Verifying calculated max and min points are as expected
  repeat {split}; assumption
  sorry

end max_min_values_of_function_l687_687294


namespace find_consecutive_integers_for_negative_sum_l687_687303

theorem find_consecutive_integers_for_negative_sum :
  ∃ n : ℤ, ((n+1) = n + 1) ∧ ((n^2 - 13 * n + 36) + ((n + 1)^2 - 13 * (n + 1) + 36) < 0) :=
by
  use 4
  split
  case left => rfl
  case right => norm_num
  sorry

end find_consecutive_integers_for_negative_sum_l687_687303


namespace probability_of_Xiao_Bing_winning_l687_687968

noncomputable def probability_Xiao_Bing_winning : ℚ :=
let outcomes := (6 * 6 : ℚ),
    same := 6,
    diff := 30 in
(diff / outcomes * 2) / ((same / outcomes * 10) + (diff / outcomes * 2))

theorem probability_of_Xiao_Bing_winning :
  probability_Xiao_Bing_winning = 1 / 2 :=
by
  sorry

end probability_of_Xiao_Bing_winning_l687_687968


namespace positive_real_as_sum_l687_687475

theorem positive_real_as_sum (k : ℝ) (hk : k > 0) : 
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ n, a n < a (n + 1)) ∧ (∑' n, 1 / 10 ^ a n = k) :=
sorry

end positive_real_as_sum_l687_687475


namespace isosceles_triangle_l687_687381

-- Define the regular pentagon and its center
def regular_pentagon (A1 A2 A3 A4 A5 O : Point) : Prop :=
  ∀ x y, 
    (x = O ∧ dist O A1 = dist O x) ∨ 
    (is_adjacent A1 x y ∧ dist x y = dist A1 A2) ∨ 
    (is_diagonal x y ∧ dist x y = dist A1 A3)

-- Define the main theorem to prove
theorem isosceles_triangle (A1 A2 A3 A4 A5 O : Point) :
  regular_pentagon A1 A2 A3 A4 A5 O →
  ∀ (x y z : Point), 
  x ∈ {A1, A2, A3, A4, A5, O} → 
  y ∈ {A1, A2, A3, A4, A5, O} → 
  z ∈ {A1, A2, A3, A4, A5, O} →
  (x ≠ y ∧ y ≠ z ∧ z ≠ x) →
  isosceles_triangle x y z :=
by
  sorry

end isosceles_triangle_l687_687381


namespace value_of_fraction_l687_687023

theorem value_of_fraction (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 :=
by
  sorry

end value_of_fraction_l687_687023


namespace spherical_to_rectangular_correct_l687_687639

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin(phi) * Real.cos(theta), rho * Real.sin(phi) * Real.sin(theta), rho * Real.cos(phi))

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 4 (Real.pi / 4) (Real.pi / 6) = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) :=
by
  sorry

end spherical_to_rectangular_correct_l687_687639


namespace count_symmetric_pentominoes_l687_687006

def pentominoes : List (Set (Int × Int)) := [
  -- Definitions of all 12 pentominoes as sets of coordinates (not provided in detail)
]

def hasReflectionalSymmetry (pentomino : Set (Int × Int)) : Prop :=
  -- Definition of reflectional symmetry for a set of coordinates (not provided in detail)

theorem count_symmetric_pentominoes : 
  (count hasReflectionalSymmetry pentominoes) = 6 :=
sorry

end count_symmetric_pentominoes_l687_687006


namespace quadratic_common_root_l687_687870

theorem quadratic_common_root 
    (p q r : ℝ) 
    (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
    (h_not_all_equal : p ≠ q ∨ q ≠ r ∨ r ≠ p)
    (α : ℝ) 
    (h1 : p * α^2 + 2 * q * α + r = 0) 
    (h2 : q * α^2 + 2 * r * α + p = 0) :
  α < 0 ∧ ∃ β γ : ℂ, β ≠ γ ∧ (real.sqrt (r^2 - 4 * r * q) < 0) := 
sorry

end quadratic_common_root_l687_687870


namespace range_of_a_l687_687709

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l687_687709


namespace period_tan_sec_l687_687150

noncomputable def y (x : ℝ) : ℝ := (Real.tan x) + (Real.sec x)

theorem period_tan_sec : ∀ x, y(x + 2 * Real.pi) = y(x) :=
by
  sorry

end period_tan_sec_l687_687150


namespace problem1_problem2_l687_687798

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2 - a)*(x - 1) - 2*Real.log x

theorem problem1 (x : ℝ) : 
  (∀ x > 0, f x 1 < f 2 1) ∧ (∀ x > 2, f x 1 > f 2 1) :=
sorry

theorem problem2 (x : ℝ) : 
  (∀ x ∈ Ioo 0 (1 / 2), f x (2 - 4 * Real.log 2) > 0) :=
sorry

end problem1_problem2_l687_687798


namespace cos_angle_AND_l687_687175

noncomputable def regular_tetrahedron (A B C D : ℝ^3) :=
  (dist A B = dist A C) ∧
  (dist A B = dist A D) ∧
  (dist A B = dist B C) ∧
  (dist A B = dist B D) ∧
  (dist A B = dist C D)

noncomputable def midpoint (P Q M : ℝ^3) :=
  M = (P + Q) / 2

theorem cos_angle_AND {A B C D N : ℝ^3} 
  (h_tetra : regular_tetrahedron A B C D)
  (h_midpoint : midpoint B C N) : 
  real.cos (angle A N D) = 2 / 3 := 
sorry

end cos_angle_AND_l687_687175


namespace students_uncool_parents_only_child_l687_687533

theorem students_uncool_parents_only_child :
  ∀ (total students_with_cool_dads students_with_cool_moms both_cool_parents_and_siblings : ℕ),
    total = 40 →
    students_with_cool_dads = 20 →
    students_with_cool_moms = 22 →
    both_cool_parents_and_siblings = 10 →
    (total - (students_with_cool_dads + students_with_cool_moms - both_cool_parents_and_siblings)) = 8 :=
by {
  -- Conditions from the problem
  intros total students_with_cool_dads students_with_cool_moms both_cool_parents_and_siblings,
  assume h1 : total = 40,
  assume h2 : students_with_cool_dads = 20,
  assume h3 : students_with_cool_moms = 22,
  assume h4 : both_cool_parents_and_siblings = 10,
  -- Convert the problem to the given conditions ensuring we reach the required conclusion
  rw [h1, h2, h3, h4],
  -- Simplify the resulting equality
  sorry
}

end students_uncool_parents_only_child_l687_687533


namespace min_pipes_to_match_capacity_l687_687598

theorem min_pipes_to_match_capacity :
  ∀ (h : ℝ),
  let r_big := 6
  let V_big := λ h, π * r_big^2 * h
  let r_small := 1.5
  let V_small := λ h, π * r_small^2 * h 
  in V_big h = 16 * V_small h :=
by
  sorry

end min_pipes_to_match_capacity_l687_687598


namespace volume_pyramid_ABFG_is_4_3_l687_687582

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def AB := 2
def BF := 2
def baseArea := (1/2) * AB * BF
def height := 2

def pyramidVolume (baseArea height : ℝ) := (1 / 3) * baseArea * height

theorem volume_pyramid_ABFG_is_4_3 :
    let A := Point3D.mk 0 0 0
    let B := Point3D.mk 2 0 0
    let F := Point3D.mk 2 0 2
    let G := Point3D.mk 2 2 2
    let volume := pyramidVolume baseArea height
    volume = 4 / 3 := sorry

end volume_pyramid_ABFG_is_4_3_l687_687582


namespace intersection_M_N_l687_687884

def M : Set ℕ := {1, 2, 4, 8}
def N : Set ℕ := {x | x ∣ 4 ∧ 0 < x}

theorem intersection_M_N :
  M ∩ N = {1, 2, 4} :=
sorry

end intersection_M_N_l687_687884


namespace ten_differences_le_100_exists_l687_687670

theorem ten_differences_le_100_exists (s : Finset ℤ) (h_card : s.card = 101) (h_range : ∀ x ∈ s, 0 ≤ x ∧ x ≤ 1000) :
∃ S : Finset ℕ, S.card = 10 ∧ (∀ y ∈ S, y ≤ 100) :=
by {
  sorry
}

end ten_differences_le_100_exists_l687_687670


namespace cyclic_proportion_l687_687092

variable {A B C p q r : ℝ}

theorem cyclic_proportion (h1 : A / B = p) (h2 : B / C = q) (h3 : C / A = r) :
  ∃ x y z, A = x ∧ B = y ∧ C = z ∧ x / y = p ∧ y / z = q ∧ z / x = r ∧
  x = (p^2 * q / r)^(1/3:ℝ) ∧ y = (q^2 * r / p)^(1/3:ℝ) ∧ z = (r^2 * p / q)^(1/3:ℝ) :=
by sorry

end cyclic_proportion_l687_687092


namespace prime_polynomial_condition_l687_687173

theorem prime_polynomial_condition (p : ℕ) (hp : Nat.Prime p) :
  (∀ (P Q : Polynomial (Fin p)), 
    (∀ n : ℤ, eval (eval (n : Polynomial ℤ) Q) P % p = n % p) 
    → P.degree = Q.degree) 
  ↔ p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 :=
sorry

end prime_polynomial_condition_l687_687173


namespace no_all_prime_l687_687947

open Nat

-- Let's define the digits and the cards
def digits := {1, 3, 7, 9}

-- Let's assume there are four positions. On each card there are two sides.
def card1_side1 := 1
def card1_side2 := 3
def card2_side1 := 7
def card2_side2 := 9

-- Two-digit numbers formed
def numbers_formed := {
  10 * card1_side1 + card2_side1,
  10 * card1_side1 + card2_side2,
  10 * card1_side2 + card2_side1,
  10 * card1_side2 + card2_side2,
  10 * card2_side1 + card1_side1,
  10 * card2_side1 + card1_side2,
  10 * card2_side2 + card1_side1,
  10 * card2_side2 + card1_side2
}

-- Check if all numbers formed are prime
theorem no_all_prime:
  ∀ n ∈ numbers_formed, ¬ prime n :=
by {
  -- Numbers formed include 39 and 91, both are composite.
  sorry
}

end no_all_prime_l687_687947


namespace sum_of_sequence_l687_687342

noncomputable def f (x : ℝ) : ℝ := x^2 + x

theorem sum_of_sequence (n : ℕ) (hn : n > 0) :
  (finset.sum (finset.range n.succ) (λ k, 1 / f (k + 1))) = n / (n + 1) :=
by 
  sorry

end sum_of_sequence_l687_687342


namespace interest_rate_annual_l687_687973

theorem interest_rate_annual :
  ∃ R : ℝ, 
    (5000 * 2 * R / 100) + (3000 * 4 * R / 100) = 2640 ∧ 
    R = 12 :=
sorry

end interest_rate_annual_l687_687973


namespace factor_expression_l687_687270

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687270


namespace removed_number_is_1011_l687_687310

theorem removed_number_is_1011 : 
  ∃ x ∈ finset.range 2022, ∑ i in finset.range 2022, i - x % 2022 = 0 :=
sorry

end removed_number_is_1011_l687_687310


namespace value_of_b_area_of_triangle_l687_687834

noncomputable def f (x B : ℝ) := sin (2*x + B) + sqrt 3 * cos (2*x + B)
def is_even_function {f : ℝ → ℝ} := ∀ x, f (-x) = f x

theorem value_of_b (B : ℝ) (hB : B = π / 6) (hf : is_even_function (f _ B)) :
  f (π / 12) B = sqrt 3 :=
sorry

theorem area_of_triangle (a b : ℝ) (hB : B = π / 6) (ha : a = 3) (hb : b = sqrt 3)
  (hA : ∀ A, A = π / 3 ∨ A = 2 * π / 3) :
  ∃ S, (S = (3 * sqrt 3) / 2 ∨ S = (3 * sqrt 3) / 4) :=
sorry

end value_of_b_area_of_triangle_l687_687834


namespace g_x_distinct_values_l687_687066

noncomputable def g (x : ℝ) : ℝ :=
  ∑ k in finset.range 13, (⌊2 * (k + 3) * x⌋ - 2 * (k + 3) * ⌊x⌋)

theorem g_x_distinct_values : ∃ n : ℕ, n = 50 ∧ ∀ x : ℝ, x ≥ 0 → (∃ (a b : ℝ), g a ≠ g b ∧ a ≠ b) → finset.card (finset.range (int.ceil (classical.some (g x)))) = n :=
sorry

end g_x_distinct_values_l687_687066


namespace tournament_game_count_l687_687855

/-- In a tournament with 25 players where each player plays 4 games against each other,
prove that the total number of games played is 1200. -/
theorem tournament_game_count : 
  let n := 25
  let games_per_pair := 4
  let total_games := (n * (n - 1) / 2) * games_per_pair
  total_games = 1200 :=
by
  -- Definitions based on the conditions
  let n := 25
  let games_per_pair := 4

  -- Calculating the total number of games
  let total_games := (n * (n - 1) / 2) * games_per_pair

  -- This is the main goal to prove
  have h : total_games = 1200 := sorry
  exact h

end tournament_game_count_l687_687855


namespace cost_percentage_l687_687120

-- Define the original and new costs
def original_cost (t b : ℝ) : ℝ := t * b^4
def new_cost (t b : ℝ) : ℝ := t * (2 * b)^4

-- Define the theorem to prove the percentage relationship
theorem cost_percentage (t b : ℝ) (C R : ℝ) (h1 : C = original_cost t b) (h2 : R = new_cost t b) :
  (R / C) * 100 = 1600 :=
by sorry

end cost_percentage_l687_687120


namespace inscribed_circle_diameter_l687_687546

-- Define the side lengths of the triangle
variables (AB AC BC : ℝ)
-- Define the semiperimeter s and area K of the triangle
noncomputable def s := (AB + AC + BC) / 2
noncomputable def area := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
-- Define the radius and diameter of the inscribed circle
noncomputable def r := area / s
noncomputable def diameter := 2 * r

-- The theorem to be proved
theorem inscribed_circle_diameter (hAB : AB = 11) (hAC : AC = 6) (hBC : BC = 7) : diameter = Real.sqrt 10 :=
by 
  have semiperimeter : s = 12 := by sorry
  have area_value : area = 6 * Real.sqrt 10 := by sorry
  have radius_value : r = Real.sqrt 10 / 2 := by sorry
  show diameter = Real.sqrt 10, from by sorry

end inscribed_circle_diameter_l687_687546


namespace perimeter_non_shaded_region_l687_687489

-- Definitions
def outer_rectangle : ℝ × ℝ := (12, 10) -- (width, height)
def attached_rectangle : ℝ × ℝ := (3, 4) -- (width, height)
def inner_shaded_rectangle : ℝ × ℝ := (3, 5) -- (width, height)
def shaded_area : ℝ := 120
def right_angle (θ : ℝ) : Prop := θ = 90

-- Statement to prove
theorem perimeter_non_shaded_region : 
  let total_area := (outer_rectangle.1 * outer_rectangle.2) + (attached_rectangle.1 * attached_rectangle.2),
      non_shaded_area := total_area - shaded_area,
      perimeter := (2 * ((outer_rectangle.1 - inner_shaded_rectangle.1) / 2 + inner_shaded_rectangle.2)) + 
                   (2 * ((outer_rectangle.2 - inner_shaded_rectangle.2) + attached_rectangle.2))
  in perimeter = 19 := 
by
  sorry

end perimeter_non_shaded_region_l687_687489


namespace methane_reaction_l687_687962

noncomputable def methane_reacts_with_chlorine
  (moles_CH₄ : ℕ)
  (moles_Cl₂ : ℕ)
  (moles_CCl₄ : ℕ)
  (moles_HCl_produced : ℕ) : Prop :=
  moles_CH₄ = 3 ∧ 
  moles_Cl₂ = 12 ∧ 
  moles_CCl₄ = 3 ∧ 
  moles_HCl_produced = 12

theorem methane_reaction : 
  methane_reacts_with_chlorine 3 12 3 12 :=
by sorry

end methane_reaction_l687_687962


namespace sum_of_perimeters_of_squares_l687_687542

theorem sum_of_perimeters_of_squares
  (x y : ℝ)
  (h1 : x^2 + y^2 = 130)
  (h2 : x^2 / y^2 = 4) :
  4*x + 4*y = 12*Real.sqrt 26 := by
  sorry

end sum_of_perimeters_of_squares_l687_687542


namespace max_possible_x_l687_687295

noncomputable section

def tan_deg (x : ℕ) : ℝ := Real.tan (x * Real.pi / 180)

theorem max_possible_x (x y : ℕ) (h₁ : tan_deg x - tan_deg y = 1 + tan_deg x * tan_deg y)
  (h₂ : tan_deg x * tan_deg y = 1) (h₃ : x = 98721) : x = 98721 := sorry

end max_possible_x_l687_687295


namespace min_cos_x_l687_687879

open Real

theorem min_cos_x (x y z : ℝ) (h1 : sin x = cot y) (h2 : sin y = cot z) (h3 : sin z = cot x) : 
  cos x = sqrt ((3 - sqrt 5) / 2) :=
sorry

end min_cos_x_l687_687879


namespace a_finishes_remaining_work_in_7_days_l687_687181

/-- Proof problem: Given that A can finish a work in 21 days and B can do the same work in 15 
days. B worked for 10 days and left the job. Prove that A alone can finish the remaining work 
in 7 days. -/
theorem a_finishes_remaining_work_in_7_days (work : Type*) (total_work A_work_rate B_work_rate : ℝ) 
  (A_days : 21 = total_work / A_work_rate) 
  (B_days : 15 = total_work / B_work_rate) 
  (B_work_done_in_10_days : B_work_rate * 10 = (2/3) * total_work) 
  : (total_work - (2 / 3) * total_work) / A_work_rate = 7 := 
by 
  conv at A_days {to_rhs, rw ← mul_div_assoc, rw mul_comm A_work_rate, rw mul_div}
  sorry

end a_finishes_remaining_work_in_7_days_l687_687181


namespace monthly_subscription_cheaper_l687_687026

variable (x : ℕ) (x_val : x = 20)

def cost_per_minute (time : ℕ) : ℝ := (0.05 + 0.02) * 60 * time

def cost_subscription (time : ℕ) : ℝ := 50 + (0.02 * 60 * time)

theorem monthly_subscription_cheaper (h : x = 20) :
  cost_subscription x < cost_per_minute x :=
by
  have h1 : cost_per_minute x = 4.2 * x,
  { unfold cost_per_minute, ring_nf },
  have h2 : cost_subscription x = 50 + 1.2 * x,
  { unfold cost_subscription, ring_nf },
  rw [h1, h2, h],
  norm_num,
  sorry

end monthly_subscription_cheaper_l687_687026


namespace unique_extremum_range_l687_687021

noncomputable def f (a x : ℝ) : ℝ := a * (x-2) * real.exp x + real.log x + (1/x)

theorem unique_extremum_range (a : ℝ) (h_extremum : ∃! x, x > 0 ∧ (∀ y, y > 0 → (y < x → f a y < f a x) ∧ (y > x → f a y > f a x))) :
  0 ≤ a ∧ a < 1 / real.exp 1 :=
by
  sorry

end unique_extremum_range_l687_687021


namespace max_additional_pies_l687_687590

theorem max_additional_pies (initial_cherries used_cherries cherries_per_pie : ℕ) 
  (h₀ : initial_cherries = 500) 
  (h₁ : used_cherries = 350) 
  (h₂ : cherries_per_pie = 35) :
  (initial_cherries - used_cherries) / cherries_per_pie = 4 := 
by
  sorry

end max_additional_pies_l687_687590


namespace count_ordered_triples_l687_687632

theorem count_ordered_triples :
  (∃ (a b c : ℕ), 
    1 ≤ a ∧ a ≤ 12 ∧
    1 ≤ b ∧ b ≤ 12 ∧
    1 ≤ c ∧ c ≤ 12 ∧
    (a * b * (b * c + 1) ≡ -1 [MOD 13])) -> 
    (cardinal.mk {p : ℕ × ℕ × ℕ | 
      p.1.1 ∈ (finset.range 12).image (+1) ∧ 
      p.1.2 ∈ (finset.range 12).image (+1) ∧ 
      p.2 ∈ (finset.range 12).image (+1) ∧ 
      (p.1.1 * p.1.2 * (p.1.2 * p.2 + 1) ≡ -1 [MOD 13])} = 132) :=
begin
  sorry
end

end count_ordered_triples_l687_687632


namespace equation_of_circumscribed_circle_ABC_l687_687370

-- Define the points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 0 0
def B := Point.mk 4 0
def C := Point.mk 0 6

-- Define the equation of the circumscribed circle
def circumscribed_circle (center : Point) (radius : ℝ) : (ℝ × ℝ) → Prop :=
  λ p, (p.1 - center.x)^2 + (p.2 - center.y)^2 = radius^2

-- Define the midpoint of BC and the radius of the circumscribed circle
def midpoint (p₁ p₂ : Point) : Point :=
  Point.mk ((p₁.x + p₂.x) / 2) ((p₁.y + p₂.y) / 2)

def distance (p₁ p₂ : Point) : ℝ :=
  real.sqrt ((p₂.x - p₁.x)^2 + (p₂.y - p₁.y)^2)

-- The statement to prove
theorem equation_of_circumscribed_circle_ABC :
  circumscribed_circle (midpoint B C) (distance B C / 2) = λ p, (p.1 - 2)^2 + (p.2 - 3)^2 = 13 :=
by
  sorry

end equation_of_circumscribed_circle_ABC_l687_687370


namespace ada_original_seat_l687_687105

theorem ada_original_seat 
  (Bea_initial : ℕ) (Ceci_initial : ℕ) (Dee_initial : ℕ) (Edie_initial : ℕ) (Fifi_initial : ℕ) 
  (Bea_final = Bea_initial - 1)
  (Ceci_final = Ceci_initial + 2)
  (Dee_final = Edie_initial) 
  (Edie_final = Dee_initial)
  (Fifi_final = 1) 
  (Fifi_initial = 4):
  (Ada_initial = 4) :=
sorry

end ada_original_seat_l687_687105


namespace factor_expression_l687_687280

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687280


namespace min_k_for_cube_root_difference_l687_687529

theorem min_k_for_cube_root_difference : 
  ∀ (s : Finset ℕ), s.card = 13 → (∀ {a b : ℕ}, a ∈ s → b ∈ s → a ≠ b → |Real.cbrt a - Real.cbrt b| < 1) :=
by
  sorry

end min_k_for_cube_root_difference_l687_687529


namespace range_of_a_l687_687717

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l687_687717


namespace cosine_of_arcsine_l687_687629

theorem cosine_of_arcsine (h : -1 ≤ (8 : ℝ) / 17 ∧ (8 : ℝ) / 17 ≤ 1) : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
sorry

end cosine_of_arcsine_l687_687629


namespace triangle_AB_eq_2DE_l687_687867

variables {A B C D E : Type} [Euclidean_geometry A B C D E]

theorem triangle_AB_eq_2DE 
  (h1 : is_altitude A D B C) 
  (h2 : is_median A E B C) 
  (h3 : angle B = 2 * angle C) : 
  side_length A B = 2 * side_length D E :=
sorry

end triangle_AB_eq_2DE_l687_687867


namespace inequality_solution_set_l687_687355

noncomputable def cond := ∀ (a b x : ℝ), (ax - b > 0) ↔ (x > 1 / 2)

theorem inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, (a * x - b > 0) ↔ (x > 1 / 2)) :
    {x | (a * x - 2 * b) / (-x + 5) > 0} = {x | 1 < x ∧ x < 5} :=
sorry

end inequality_solution_set_l687_687355


namespace simplest_quadratic_radical_l687_687562

theorem simplest_quadratic_radical : 
  (∀ x ∈ {sqrt (1/2), sqrt 3, sqrt 8, sqrt 4}, 
    (IsSimplestQuadraticRadical x = (x = sqrt 3)))
  := by sorry

namespace Helper 

def IsSimplestQuadraticRadical (x : Real) : Prop :=
  -- Definition of simplest quadratic radical:
  -- A quadratic radical is in simplest form if it is not simplifiable or is a prime square root.
  sqrt x = x
  
end Helper

end simplest_quadratic_radical_l687_687562


namespace fourth_vertex_parallelogram_coordinates_l687_687020

def fourth_vertex_of_parallelogram (A B C : ℝ × ℝ) :=
  ∃ D : ℝ × ℝ, (D = (11, 4) ∨ D = (-1, 12) ∨ D = (3, -12))

theorem fourth_vertex_parallelogram_coordinates :
  fourth_vertex_of_parallelogram (1, 0) (5, 8) (7, -4) :=
by
  sorry

end fourth_vertex_parallelogram_coordinates_l687_687020


namespace fraction_of_girls_at_concert_l687_687221

variable (g b : ℕ)
variable (h_eq : g = b)

theorem fraction_of_girls_at_concert
  (h_girls_at_concert : 5 / 6 * g)
  (h_boys_at_concert : 3 / 4 * b) :
  (5 / 6 * g) / ((5 / 6 * g) + (3 / 4 * b)) = 30 / 57 := by
  sorry

end fraction_of_girls_at_concert_l687_687221


namespace price_per_can_of_spam_l687_687380

-- Definitions of conditions
variable (S : ℝ) -- The price per can of Spam
def cost_peanut_butter := 3 * 5 -- 3 jars of peanut butter at $5 each
def cost_bread := 4 * 2 -- 4 loaves of bread at $2 each
def total_cost := 59 -- Total amount paid

-- Proof problem to verify the price per can of Spam
theorem price_per_can_of_spam :
  12 * S + cost_peanut_butter + cost_bread = total_cost → S = 3 :=
by
  sorry

end price_per_can_of_spam_l687_687380


namespace value_of_n_l687_687577

theorem value_of_n {k n : ℕ} (h1 : k = 71 * n + 11) (h2 : (k : ℝ) / (n : ℝ) = 71.2) : n = 55 :=
sorry

end value_of_n_l687_687577


namespace simplify_expr1_simplify_expr2_l687_687104

theorem simplify_expr1 : (-4)^2023 * (-0.25)^2024 = -0.25 :=
by 
  sorry

theorem simplify_expr2 : 23 * (-4 / 11) + (-5 / 11) * 23 - 23 * (2 / 11) = -23 :=
by 
  sorry

end simplify_expr1_simplify_expr2_l687_687104


namespace points_of_intersection_l687_687812

noncomputable def intersection_points : Prop :=
  let f := λ (x : ℝ), 3 * x^2 - 6 * x + 5
  let g := λ (x : ℝ), 2 * x + 1
  ∃ x1 y1 x2 y2 : ℝ, 
    (f x1 = y1 ∧ g x1 = y1 ∧ f x2 = y2 ∧ g x2 = y2) ∧
    (x1, y1) = (2, 5) ∧ (x2, y2) = (2 / 3, 7 / 3)

theorem points_of_intersection : intersection_points :=
  sorry

end points_of_intersection_l687_687812


namespace find_angle_C_find_side_c_l687_687397


-- Define the problem conditions
variables {A B C : Real} {a b c : Real} (hC_acute : 0 < C ∧ C < Real.pi / 2) (hC_nonzero_cos : Real.cos C ≠ 0)

-- The problem: 1. Find the measure of angle C
theorem find_angle_C (h : Real.sin (2 * C) = Real.sqrt 3 * Real.cos C) : C = Real.pi / 3 :=
sorry

-- The problem: 2. Find the length of side c
theorem find_side_c (ha : a = 1) (hb : b = 4) (hC : C = Real.pi / 3) : c = Real.sqrt 13 :=
sorry

end find_angle_C_find_side_c_l687_687397


namespace arithmetic_seq_third_sum_l687_687413

-- Define the arithmetic sequence using its first term and common difference
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * n

theorem arithmetic_seq_third_sum
  (a₁ d : ℤ)
  (h1 : (a₁ + (a₁ + 3 * d) + (a₁ + 6 * d) = 39))
  (h2 : ((a₁ + d) + (a₁ + 4 * d) + (a₁ + 7 * d) = 33)) :
  ((a₁ + 2 * d) + (a₁ + 5 * d) + (a₁ + 8 * d) = 27) :=
by
  sorry

end arithmetic_seq_third_sum_l687_687413


namespace total_number_of_edges_bound_l687_687328

def in_general_position (L : set Line) : Prop :=
  (∀ l₁ l₂ ∈ L, l₁ ≠ l₂ → ¬ parallel l₁ l₂) ∧
  (∀ l₁ l₂ l₃ ∈ L, l₁ ≠ l₂ → l₂ ≠ l₃ → l₁ ≠ l₃ →
     ¬ concurrent {l₁, l₂, l₃})

theorem total_number_of_edges_bound (L : set Line) (hL : in_general_position L) (ℓ : Line) :
  let n := |L|
  in total_edges_intersected_by ℓ ≤ 6 * n := 
sorry

end total_number_of_edges_bound_l687_687328


namespace talent_show_girls_count_l687_687938

theorem talent_show_girls_count (B G : ℕ) (h1 : B + G = 34) (h2 : G = B + 22) : G = 28 :=
by
  sorry

end talent_show_girls_count_l687_687938


namespace range_of_a_l687_687701

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687701


namespace mean_home_runs_is_7_29_l687_687118

-- Conditions (definition of the problem)
def num_home_runs : List (ℕ × ℕ) := [(5, 5), (3, 9), (4, 7), (2, 11)]

-- Calculating the total number of home runs
def total_home_runs : ℕ := num_home_runs.foldl (λ acc p, acc + p.1 * p.2) 0

-- Calculating the total number of players
def total_players : ℕ := num_home_runs.foldl (λ acc p, acc + p.1) 0

-- The mean number of home runs
def mean_home_runs : ℕ → ℕ → ℝ := λ total players, total / players.toReal

-- The theorem to prove the mean number of home runs is 7.29
theorem mean_home_runs_is_7_29 : mean_home_runs total_home_runs total_players = 7.29 := 
by
  unfold mean_home_runs total_home_runs total_players
  -- We will simplify here to get the answer (using the definition and Lean's algebraic capabilities)
  sorry

end mean_home_runs_is_7_29_l687_687118


namespace range_of_a_l687_687695

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687695


namespace speed_of_slower_train_is_correct_l687_687951

-- Definitions based on conditions
def length_of_each_train : ℝ := 60  -- Length of each train in meters
def speed_of_faster_train : ℝ := 48  -- Speed of the faster train in km/hr
def time_to_pass_slower_train : ℝ := 36  -- Time to pass the slower train in seconds
def distance_to_pass : ℝ := length_of_each_train * 2  -- Distance to pass the slower train

-- Conversion factor
def kmh_to_mps (speed : ℝ) : ℝ := speed * (5 / 18)

-- Prove the speed of the slower train
theorem speed_of_slower_train_is_correct (v : ℝ) (h : kmh_to_mps (speed_of_faster_train - v) = distance_to_pass / time_to_pass_slower_train) : v = 36 := 
  sorry

end speed_of_slower_train_is_correct_l687_687951


namespace factorize_expression_l687_687654

theorem factorize_expression (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by sorry

end factorize_expression_l687_687654


namespace range_of_a_l687_687754

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687754


namespace range_of_a_l687_687690

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687690


namespace meaningful_expression_l687_687933

-- Define the condition and the statement to prove
theorem meaningful_expression (x : ℝ) : x ≠ 2023 → ∃ (y : ℝ), y = 1 / (x - 2023) :=
begin
  intro h,
  use 1 / (x - 2023),
  simp,
  exact h,
end

end meaningful_expression_l687_687933


namespace find_unit_prices_and_max_additional_purchase_l687_687943

noncomputable def unit_price_of_type_A_B (p_A p_B : ℕ) : Prop :=
p_A = 4 ∧ p_B = 2.5

theorem find_unit_prices_and_max_additional_purchase :
  (∃ x y, x = 4 ∧ y = 2.5 ∧ 
    x = y + 1.5 ∧ 
    (8000 / x) = (5000 / y)) ∧ 
  (∃ m, m = 800 ∧ 
    (2.5 * 2 * m + 4 * m ≤ 7200) ∧ 
    (2.5 * 2 * m + 4 * m).to_nat = m.to_nat * 9) :=
begin
  sorry,
end

end find_unit_prices_and_max_additional_purchase_l687_687943


namespace range_g_l687_687299

def g (x : ℝ) : ℝ := if x ≠ 1 then 3 * (x + 5) else 0

theorem range_g :
  {y : ℝ | ∃ x : ℝ, x ≠ 1 ∧ g x = y} = {y : ℝ | y ≠ 18} := by
  sorry

end range_g_l687_687299


namespace percent_membership_voted_winning_l687_687569

-- Definitions based on conditions
def total_members : ℕ := 1600
def votes_cast : ℕ := 525
def percent_votes_cast_winning : ℝ := 0.60

-- Problem statement
theorem percent_membership_voted_winning :
  let votes_received := percent_votes_cast_winning * votes_cast in
  (votes_received / total_members) * 100 = 19.6875 :=
by
  let votes_received := percent_votes_cast_winning * votes_cast
  sorry

end percent_membership_voted_winning_l687_687569


namespace knights_selection_l687_687581

/-- 
Given:
1. There are 12 knights sitting at a round table.
2. Each knight is an enemy with the two adjacent knights.
3. We need to choose 5 knights such that no two chosen knights are enemies.

Prove that the number of ways to make this selection is 36.
-/

theorem knights_selection (n k : ℕ) (hn : n = 12) (hk : k = 5) :
  (∃ (f : finset (fin n)), f.card = k ∧ (∀ (x ∈ f) (y ∈ f), abs (x - y) ≠ 1 ∧ abs (x - y) ≠ n - 1))
  → (reflect 36 : ℕ) :=
by
  sorry

end knights_selection_l687_687581


namespace negation_proposition_of_cube_of_odd_is_odd_l687_687129

def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem negation_proposition_of_cube_of_odd_is_odd :
  (¬ ∀ n : ℤ, odd n → odd (n^3)) ↔ (∃ n : ℤ, odd n ∧ ¬ odd (n^3)) :=
by
  sorry

end negation_proposition_of_cube_of_odd_is_odd_l687_687129


namespace work_completion_days_l687_687570

theorem work_completion_days :
  (∀ (M : ℕ → ℕ → ℕ), M 8 20) →
  (∀ (W : ℕ → ℕ → ℕ), W 12 20) →
  (∀ (combined_work_days : ℕ → ℕ → ℕ), combined_work_days 6 11 = 12) :=
by
  intros M W
  have man_rate := 1 / (8 * 20)
  have woman_rate := 1 / (12 * 20)
  have combined_rate := 6 * man_rate + 11 * woman_rate
  have one_day_work := combined_rate / 1
  have required_days := 1 / one_day_work
  assumption
  sorry

end work_completion_days_l687_687570


namespace johny_travelled_South_distance_l687_687862

theorem johny_travelled_South_distance :
  ∃ S : ℝ, S + (S + 20) + 2 * (S + 20) = 220 ∧ S = 40 :=
by
  sorry

end johny_travelled_South_distance_l687_687862


namespace critical_point_value_and_nature_monotonic_intervals_l687_687122

noncomputable def f (x : ℝ) (a : ℝ) := exp x * (x^2 - a*x - a)

theorem critical_point_value_and_nature (a : ℝ) :
  (∃ c : ℝ, (∃ x : ℝ, x = 1 ∧ x = c) ∧ (∀ x, deriv (f x a) x = (f x a) + exp x * (2 * x - a)) ∧ deriv (f 1 a) 1 = 0) ↔ 
  (a = 1 ∧ ∀ x, f' x = exp x * (x + 2) * (x - a) ∧ x = 1 -> is_min (f x a) 1) :=
sorry

theorem monotonic_intervals (a : ℝ) :
  (∀ x, f' x = exp x * (x + 2) * (x - a)) ∧ (-∞ < x < a ∨ -2 < x < ∞) ∧ (x > a ∨ x < -2 → f' x > 0) ∧ (x < a ∨ -2 < x < ∞ → f' x < 0) ↔ 
  (if a = -2, ∀ x, f x > 0; if a < -2, increasing_on (f x) (-∞, a) ∧ (-2, ∞) decreasing_on (f x) (a, -2); if a > -2, increasing_on (f x) (-∞, -2) (a, ∞) decreasing_on (f x) (-2, a)) :=
sorry

end critical_point_value_and_nature_monotonic_intervals_l687_687122


namespace find_price_of_100_apples_l687_687194

noncomputable def price_of_100_apples (P : ℕ) : Prop :=
  (12000 / P) - (12000 / (P + 4)) = 5

theorem find_price_of_100_apples : price_of_100_apples 96 :=
by
  sorry

end find_price_of_100_apples_l687_687194


namespace missing_digit_divisibility_l687_687514

theorem missing_digit_divisibility (x : ℕ) (h1 : x < 10) :
  3 ∣ (1 + 3 + 5 + 7 + x + 2) ↔ x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
  sorry

end missing_digit_divisibility_l687_687514


namespace hyperbola_probability_l687_687466

open Function

def possible_points : List (ℕ × ℕ) := [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]

def on_hyperbola (p : ℕ × ℕ) : Bool :=
  match p with
  | (m, n) => n = 6 / m

def favorable_points : List (ℕ × ℕ) :=
  possible_points.filter on_hyperbola

theorem hyperbola_probability : 
  (favorable_points.length.toRat / possible_points.length.toRat = 1 / 3) :=
  sorry

end hyperbola_probability_l687_687466


namespace james_ate_slices_l687_687046

variable (NumPizzas : ℕ) (SlicesPerPizza : ℕ) (FractionEaten : ℚ)
variable (TotalSlices : ℕ := NumPizzas * SlicesPerPizza)
variable (JamesSlices : ℚ := FractionEaten * TotalSlices)

theorem james_ate_slices (h1 : NumPizzas = 2) (h2 : SlicesPerPizza = 6) (h3 : FractionEaten = 2 / 3) :
    JamesSlices = 8 := 
by 
  simp [JamesSlices, TotalSlices]
  rw [h1, h2, h3]
  norm_num
  sorry

end james_ate_slices_l687_687046


namespace number_of_sweaters_l687_687844

theorem number_of_sweaters 
(total_price_shirts : ℝ)
(total_shirts : ℕ)
(total_price_sweaters : ℝ)
(price_difference : ℝ) :
total_price_shirts = 400 ∧ total_shirts = 25 ∧ total_price_sweaters = 1500 ∧ price_difference = 4 →
(total_price_sweaters / ((total_price_shirts / total_shirts) + price_difference) = 75) :=
by
  intros
  sorry

end number_of_sweaters_l687_687844


namespace factor_expression_l687_687262

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687262


namespace problem_statement_l687_687070

noncomputable def cubic_poly := Polynomial.Cubic 3 (-4) 200 (-5)

theorem problem_statement (p q r : ℝ) (h_roots : p = cubic_poly.root1 ∧ q = cubic_poly.root2 ∧ r = cubic_poly.root3) :
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 403 := 
sorry

end problem_statement_l687_687070


namespace zhang_qiang_exercise_time_l687_687157

noncomputable def angle_between_hands (minute: ℕ) : ℝ :=
  abs (6 * minute - (180 + 0.5 * minute))

theorem zhang_qiang_exercise_time :
  ∃ (t: ℝ), angle_between_hands t = 110 ∧ 0 < t ∧ t < 60 ∧ abs (5.5 * t - 180) = 110 :=
by
  sorry

end zhang_qiang_exercise_time_l687_687157


namespace unique_real_root_analytic_expression_max_min_values_interval_range_of_a_l687_687687

variables {a b : ℝ} {f : ℝ → ℝ}

-- (I) Proof
theorem unique_real_root_analytic_expression
  (h0 : 2 * a + b = 0)
  (h1 : ∀ x : ℝ, f x = a * x^2 + b * x)
  (unique_root : ∃! x, f x - x = 0) :
  f = λ x, - (1 / 2) * x^2 + x := by
  sorry

-- (II) Proof
theorem max_min_values_interval
  (h0 : a = 1)
  (h1 : ∀ x : ℝ, f x = a * (x^2 - 2 * x))
  (x : ℝ)
  (hx : x ∈ set.Icc (-1 : ℝ) 2) :
  (∀ y ∈ set.Icc (-1 : ℝ) 2, f y ≤ f(-1)) ∧ (∀ y ∈ set.Icc (-1 : ℝ) 2, f(1) ≤ f y) := by
  sorry

-- (III) Proof
theorem range_of_a
  (h0 : ∀ x ≥ 2, f x = a * (x^2 - 2 * x))
  (h1 : ∀ x ≥ 2, f x ≥ 2 - a) :
  2 ≤ a := by
  sorry

end unique_real_root_analytic_expression_max_min_values_interval_range_of_a_l687_687687


namespace asymptotes_of_hyperbola_with_same_foci_l687_687810

noncomputable def hyperbola_equation (m : ℝ) : (ℝ × ℝ) → Prop := 
  λ coords, m * (coords.snd)^2 - (coords.fst)^2 = 1

noncomputable def ellipse_equation : (ℝ × ℝ) → Prop := 
  λ coords, (coords.snd)^2 / 5 + (coords.fst)^2 = 1

def foci_hyperbola (m : ℝ) : set (ℝ × ℝ) := 
  {p | p.1 = 0 ∧ (p.2 = sqrt (1 / m + 1) ∨ p.2 = -sqrt (1 / m + 1))}

def foci_ellipse : set (ℝ × ℝ) := 
  {p | p.1 = 0 ∧ (p.2 = 2 ∨ p.2 = -2)}

theorem asymptotes_of_hyperbola_with_same_foci :
  ∀ (m : ℝ), (foci_hyperbola m = foci_ellipse) → m = (1 / 3) ∧ 
  ∀ (x y : ℝ), (hyperbola_equation m (x, y)) → (y = sqrt(3) * x ∨ y = - sqrt(3) * x) :=
by
  -- Proof not required
  sorry

end asymptotes_of_hyperbola_with_same_foci_l687_687810


namespace chuck_team_leads_by_2_l687_687655

open Nat

noncomputable def chuck_team_score_first_quarter := 9 * 2 + 5 * 1
noncomputable def yellow_team_score_first_quarter := 7 * 2 + 4 * 3

noncomputable def chuck_team_score_second_quarter := 6 * 2 + 3 * 3
noncomputable def yellow_team_score_second_quarter := 5 * 2 + 2 * 3 + 3 * 1

noncomputable def chuck_team_score_third_quarter := 4 * 2 + 2 * 3 + 6 * 1
noncomputable def yellow_team_score_third_quarter := 6 * 2 + 2 * 3

noncomputable def chuck_team_score_fourth_quarter := 8 * 2 + 1 * 3
noncomputable def yellow_team_score_fourth_quarter := 4 * 2 + 3 * 3 + 2 * 1

noncomputable def chuck_team_technical_fouls := 3
noncomputable def yellow_team_technical_fouls := 2

noncomputable def total_chuck_team_score :=
  chuck_team_score_first_quarter + chuck_team_score_second_quarter + 
  chuck_team_score_third_quarter + chuck_team_score_fourth_quarter + 
  chuck_team_technical_fouls

noncomputable def total_yellow_team_score :=
  yellow_team_score_first_quarter + yellow_team_score_second_quarter + 
  yellow_team_score_third_quarter + yellow_team_score_fourth_quarter + 
  yellow_team_technical_fouls

noncomputable def chuck_team_lead :=
  total_chuck_team_score - total_yellow_team_score

theorem chuck_team_leads_by_2 :
  chuck_team_lead = 2 :=
by
  sorry

end chuck_team_leads_by_2_l687_687655


namespace shirt_price_l687_687517

theorem shirt_price (T S : ℝ) (h1 : T + S = 80.34) (h2 : T = S - 7.43) : T = 36.455 :=
by 
sorry

end shirt_price_l687_687517


namespace problem_conditions_l687_687468

noncomputable def f (x : ℝ) : ℝ := (2 * x - x^2) * Real.exp x

theorem problem_conditions :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 2) ∧
  (∃ x_max, x_max = Real.sqrt 2 ∧ (∀ y, f y ≤ f x_max)) ∧
  ¬(∃ x_min, ∀ y, f x_min ≤ f y) :=
by sorry

end problem_conditions_l687_687468


namespace solution_l687_687387

-- Definition of t
def t : ℝ := 1 / (2 - Real.cbrt 3)

-- Statement to be proved
theorem solution : t = (2 + Real.cbrt 3) * (2 + Real.sqrt 3) :=
by sorry

end solution_l687_687387


namespace least_matching_pair_l687_687141

namespace PlateProblem

-- Definitions based on conditions
def white_plates : Nat := 2
def green_plates : Nat := 6
def red_plates : Nat := 8
def pink_plates : Nat -- We assume there exists some number of pink plates.
def purple_plates : Nat := 10

-- Total number of plates
def total_plates : Nat := white_plates + green_plates + red_plates + pink_plates + purple_plates

-- Question to prove: Prove that pulling out 6 plates ensures a matching pair given the conditions.
theorem least_matching_pair (pink_plates non_neg: Nat) : 
  total_plates >= 27 → ∃ p1 p2, p1 ≠ p2 ∧ p1 = p2 :=
by
  -- Proof is not required per instructions, so we add sorry
  sorry

end PlateProblem

end least_matching_pair_l687_687141


namespace factor_expression_l687_687253

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687253


namespace solve_diophantine_eq_l687_687911

def diophantine_equation (x y z : ℤ) : Prop :=
  2 * x^2 + 2 * x^2 * z^2 + z^2 + 7 * y^2 - 42 * y + 33 = 0

theorem solve_diophantine_eq 
  : ∃ x y z : ℤ, diophantine_equation x y z ∧ 
    ((x = 1 ∧ y = 5 ∧ z = 0) ∨
     (x = -1 ∧ y = 5 ∧ z = 0) ∨
     (x = 1 ∧ y = 1 ∧ z = 0) ∨
     (x = -1 ∧ y = 1 ∧ z = 0)) :=
begin
  sorry
end

end solve_diophantine_eq_l687_687911


namespace factor_expression_l687_687275

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687275


namespace cos_arcsin_of_fraction_l687_687626

theorem cos_arcsin_of_fraction : ∀ x, x = 8 / 17 → x ∈ set.Icc (-1:ℝ) 1 → Real.cos (Real.arcsin x) = 15 / 17 :=
by
  intros x hx h_range
  rw hx
  have h : (x:ℝ)^2 + Real.cos (Real.arcsin x)^2 = 1 := Real.sin_sq_add_cos_sq (Real.arcsin x)
  sorry

end cos_arcsin_of_fraction_l687_687626


namespace time_to_pass_l687_687597

-- Defining the given conditions
def speed_man : ℝ := 50 / 3.6  -- Converting to meters per second
def speed_goods : ℝ := 62 / 3.6  -- Converting to meters per second
def length_goods : ℝ := 280  -- Length in meters

-- The theorem to prove the time taken for the goods train to pass the man's train
theorem time_to_pass : (280 / (speed_man + speed_goods)) ≈ 9 := 
sorry

end time_to_pass_l687_687597


namespace coeff_x2_term_l687_687450

noncomputable def max_value_of_function := 2

theorem coeff_x2_term
    (f : ℝ → ℝ)
    (h : ∀ x : ℝ, f x = sin x + (sqrt 3) * cos x)
    (a : ℝ)
    (ha : a = max_value_of_function) :
    coefficient (a * sqrt x - (1 / sqrt x))^6 x^2 = -192 :=
sorry

end coeff_x2_term_l687_687450


namespace find_smallest_m_l687_687449

def is_in_set_S (z : ℂ) : Prop :=
  ∃ (x y : ℝ), z = x + y * I ∧ (Real.sqrt 2 / 2) ≤ x ∧ x ≤ (Real.sqrt 3 / 2)

def is_root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

theorem find_smallest_m (m : ℕ) : m = 16 ↔
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, is_in_set_S z ∧ is_root_of_unity z n :=
sorry

end find_smallest_m_l687_687449


namespace minimum_c_value_l687_687893

noncomputable def minC (a b c : ℕ) : ℕ :=
  if a < b ∧ b < c ∧ (∃! x, 2 * x + (|x - a| + |x - b| + |x - c|) = 2019) then c else 0

theorem minimum_c_value : 
  ∀ (a b c : ℕ), a < b → b < c → 
  (∃! x, 2 * x + (abs (x - a) + abs (x - b) + abs (x - c)) = 2019) → 
  c = 1010 :=
sorry

end minimum_c_value_l687_687893


namespace find_X_l687_687579

def operation (X Y : Int) : Int := X + 2 * Y 

lemma property_1 (X : Int) : operation X 0 = X := 
by simp [operation]

lemma property_2 (X Y : Int) : operation X (Y - 1) = (operation X Y) - 2 := 
by simp [operation]; linarith

lemma property_3 (X Y : Int) : operation X (Y + 1) = (operation X Y) + 2 := 
by simp [operation]; linarith

theorem find_X (X : Int) : operation X X = -2019 ↔ X = -673 :=
by sorry

end find_X_l687_687579


namespace greg_original_seat_l687_687646

-- Defining the initial seating arrangement and movements
def movement_fn (seat : ℕ) (move : ℕ) : ℕ := seat + move

-- Define the final positions based on given movements
def iris_final_seat (initial_seat : ℕ) : ℕ := movement_fn initial_seat 1
def jamal_final_seat (initial_seat : ℕ) : ℕ := movement_fn initial_seat (-2)
def kim_final_seat (initial_kim_seat initial_leo_seat : ℕ) : ℕ := initial_leo_seat
def leo_final_seat (initial_kim_seat initial_leo_seat : ℕ) : ℕ := initial_kim_seat

-- Lean statement to find Greg's original seat
theorem greg_original_seat 
  (initial_seats : Fin 5 → ℕ)
  (Iris_initial_seat : ℕ) 
  (Jamal_initial_seat : ℕ) 
  (Kim_initial_seat : ℕ) 
  (Leo_initial_seat : ℕ) 
  (Greg_initial_seat : ℕ)
  (Iris_final := iris_final_seat Iris_initial_seat)
  (Jamal_final := jamal_final_seat Jamal_initial_seat) 
  (Kim_final := kim_final_seat Kim_initial_seat Leo_initial_seat)
  (Leo_final := leo_final_seat Kim_initial_seat Leo_initial_seat) : 
  (Iris_final, Jamal_final, Kim_final, Leo_final, 1) = 
  (initial_seats 0, initial_seats 1, initial_seats 2, initial_seats 3, initial_seats 4) → 
  Greg_initial_seat = 2 := sorry

end greg_original_seat_l687_687646


namespace factor_expression_l687_687273

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687273


namespace range_of_a_l687_687732

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l687_687732


namespace find_s_l687_687237

theorem find_s (s : ℝ) : (9 = 3^(4 * s + 2)) → s = 0 :=
by
  intros h
  sorry

end find_s_l687_687237


namespace larger_segment_l687_687138

theorem larger_segment (a b c y x : ℝ) (h1 : a = 26) (h2 : b = 60) (h3 : c = 64) :
  (a ^ 2 = x ^ 2 + y ^ 2) →
  (b ^ 2 = (c - x) ^ 2 + y ^ 2) →
  (c - x ≈ 55) :=
by sorry

end larger_segment_l687_687138


namespace fraction_equivalence_l687_687557

theorem fraction_equivalence (n : ℤ) : (4 + n) * 3 = (7 + n) * 2 → n = 2 := by
  intros h
  have h : (12 + 3 * n) = 14 + 2 * n := by simpa [mul_add]
  linarith

example : ∃ (n : ℤ), (4 + n) * 3 = (7 + n) * 2 ∧ n = 2 := by
  use 2
  split
  · simp
  · rfl

end fraction_equivalence_l687_687557


namespace number_of_subsets_B_l687_687001

def A := {1, 2, 3}

def B := {p : ℕ × ℕ | p.1 ∈ A ∧ p.2 ∈ A ∧ (p.1 + p.2) ∈ A}

theorem number_of_subsets_B : (∃ (n : ℕ), n = 8 ∧ ∃ (s : Finset (ℕ × ℕ)), s = B ∧ s.powerset.card = 2 ^ s.card) :=
by
  use 8
  split
  · refl
  · use B
    split
    · refl
    · sorry

end number_of_subsets_B_l687_687001


namespace circle_center_sum_l687_687286

/-- Given the equation of a circle, prove that the sum of the x and y coordinates of the center is -1. -/
theorem circle_center_sum (x y : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 4 * x - 6 * y + 9) → x + y = -1 :=
by 
  sorry

end circle_center_sum_l687_687286


namespace exists_real_x_l687_687059

noncomputable theory
open_locale classical

-- Let P and Q be monic polynomials of degree 2014
-- such that for all real numbers x, P(x) ≠ Q(x).
-- Show that there exists a real number x such that P(x-1) = Q(x+1).

theorem exists_real_x (P Q : Polynomial ℝ) (hP : P.monic) (hQ : Q.monic)
  (degP : P.degree = 2014) (degQ : Q.degree = 2014)
  (h : ∀ x : ℝ, P.eval x ≠ Q.eval x) :
  ∃ x : ℝ, P.eval (x - 1) = Q.eval (x + 1) :=
sorry

end exists_real_x_l687_687059


namespace no_pairs_of_a_and_d_l687_687451

theorem no_pairs_of_a_and_d :
  ∀ (a d : ℝ), (∀ (x y: ℝ), 4 * x + a * y + d = 0 ↔ d * x - 3 * y + 15 = 0) -> False :=
by 
  sorry

end no_pairs_of_a_and_d_l687_687451


namespace spatial_relationship_l687_687338

variables {a b c : Type}          -- Lines a, b, c
variables {α β γ : Type}          -- Planes α, β, γ

-- Parallel relationship between planes
def plane_parallel (α β : Type) : Prop := sorry
-- Perpendicular relationship between planes
def plane_perpendicular (α β : Type) : Prop := sorry
-- Parallel relationship between lines and planes
def line_parallel_plane (a α : Type) : Prop := sorry
-- Perpendicular relationship between lines and planes
def line_perpendicular_plane (a α : Type) : Prop := sorry
-- Parallel relationship between lines
def line_parallel (a b : Type) : Prop := sorry
-- The angle formed by a line and a plane
def angle (a : Type) (α : Type) : Type := sorry

theorem spatial_relationship :
  (plane_parallel α γ ∧ plane_parallel β γ → plane_parallel α β) ∧
  ¬ (line_parallel_plane a α ∧ line_parallel_plane b α → line_parallel a b) ∧
  ¬ (plane_perpendicular α γ ∧ plane_perpendicular β γ → plane_parallel α β) ∧
  ¬ (line_perpendicular_plane a c ∧ line_perpendicular_plane b c → line_parallel a b) ∧
  (line_parallel a b ∧ plane_parallel α β → angle a α = angle b β) :=
sorry

end spatial_relationship_l687_687338


namespace find_period_and_center_and_area_l687_687808
noncomputable def vector_dot_product (a b : (ℝ × ℝ)) : ℝ :=
a.1 * b.1 + a.2 * b.2

def f (x : ℝ) : ℝ :=
vector_dot_product (⟨2 * Real.sin x, 1⟩ : ℝ × ℝ) (⟨Real.sin (x + Real.pi / 3), -1 / 2⟩ : ℝ × ℝ)

def ABC_area (B : ℝ) (A : ℝ) (a b : ℝ) : ℝ :=
1 / 2 * a * 2 * Real.sin B * Real.sqrt 3 / 2

theorem find_period_and_center_and_area :
(∀ x, f x = f (x + Real.pi)) ∧ 
(∀ k ∈ ℤ, x = 1 / 2 * k * Real.pi + Real.pi / 12 → f x = 0) ∧ 
(∀ B A a, f B = 1 ∧ b = 2 ∧ b / a = (Real.cos B + 1) / (2 - Real.cos A) 
→ ABC_area B A a 2 = Real.sqrt 3) := sorry

end find_period_and_center_and_area_l687_687808


namespace problem_statement_l687_687913

noncomputable def solveProblem : ℝ :=
  let a := 2
  let b := -3
  let c := 1
  a + b + c

-- The theorem statement to ensure a + b + c equals 0
theorem problem_statement : solveProblem = 0 := by
  sorry

end problem_statement_l687_687913


namespace crayon_selection_l687_687837

theorem crayon_selection :
  let total_crayons := 15
  let red_crayons := 3
  let select_crayons := 5
  let non_red_crayons := total_crayons - red_crayons
  let choose (n k : ℕ) : ℕ := nat.choose n k in
  choose total_crayons select_crayons
  = (choose red_crayons 1 * choose non_red_crayons 4)
  + (choose red_crayons 2 * choose non_red_crayons 3)
  + (choose red_crayons 3 * choose non_red_crayons 2) :=
by
  sorry

end crayon_selection_l687_687837


namespace f_at_3_2_l687_687075

def f (x y : ℝ) : ℝ :=
  (x + y) / (2 * x - 3 * y + 1)

theorem f_at_3_2 : f 3 2 = 5 :=
  sorry

end f_at_3_2_l687_687075


namespace place_seven_coins_l687_687540

-- Definitions of regular 12-gon and tangent property
def regular_dodecagon (points : list Point) : Prop :=
  points.length = 12 ∧ ∀ i, (dist points[i] points[(i+1) % 12]) = 1

def tangent_property (points : list Point) : Prop :=
  ∀ i, Tangent (points[i]) (points[(i+1) % 12])

-- Proposition proving it's possible to put 7 coins inside a ring of 12 coins
theorem place_seven_coins :
  ∀ (points : list Point),
  (regular_dodecagon points) ∧ (tangent_property points) →
  ∃ (ys : list Point), ys.length = 7 ∧ (∀ i j, dist ys[i] ys[j] = 2) :=
  by
    intros points H
    sorry -- Detailed proof steps go here

end place_seven_coins_l687_687540


namespace find_m_l687_687352

variables {a : ℝ × ℝ} {m : ℝ}

def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

noncomputable def perp_condition (a b : ℝ × ℝ) : Prop :=
(a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2) = 0 

theorem find_m (h₁ : magnitude a = 1) (h₂ : b = (1/2, m)) (h₃ : perp_condition a b) : m = sqrt (3)/2 ∨ m = -sqrt (3)/2 := 
sorry

end find_m_l687_687352


namespace ratio_e_f_l687_687788

theorem ratio_e_f (a b c d e f : ℚ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.25) :
  e / f = 9 / 4 :=
sorry

end ratio_e_f_l687_687788


namespace no_real_solutions_abs_eq_l687_687819

theorem no_real_solutions_abs_eq (y : ℝ) :
  ¬ ∃ (y : ℝ), | y - 2 | = | y - 1 | + | y - 4 | := 
sorry

end no_real_solutions_abs_eq_l687_687819


namespace solve_for_x_l687_687619

theorem solve_for_x : ∃ x : ℝ, 32^10 + 2 * 32^10 = 2^x ∧ x = log 2 3 + 50 :=
by
  sorry

end solve_for_x_l687_687619


namespace range_of_a_l687_687733

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l687_687733


namespace cost_of_fencing_l687_687162

theorem cost_of_fencing (d : ℝ) (rate : ℝ) : d = 26 → rate = 1.50 → 
  let C := Real.pi * d in
  let cost := C * rate in
  cost ≈ 122.52 :=
by
  intros h1 h2
  let C := Real.pi * d
  let cost := C * rate
  have h3 : C = Real.pi * 26 := by rw [h1]
  rw [h3] at cost
  have h4 : cost = (Real.pi * 26) * 1.50 := by rw [h2]
  rw [h4]
  calc (Real.pi * 26) * 1.50
      = 81.6814 * 1.50 : by norm_num1
      = 122.5221      : by norm_num1
  approx 122.5221 ≈ 122.52 : sorry

end cost_of_fencing_l687_687162


namespace find_length_BC_in_acute_triangle_l687_687032

theorem find_length_BC_in_acute_triangle 
  {A B C : Type} [euclidean_geometry.triangle A B C]
  (area_ABC : euclidean_geometry.area A B C = 10 * real.sqrt 3)
  (AB : dist A B = 5)
  (AC : dist A C = 8) :
  dist B C = 7 := by
sorry

end find_length_BC_in_acute_triangle_l687_687032


namespace part_1_conditions_part_2_min_value_l687_687368

theorem part_1_conditions
  (a b x : ℝ)
  (h1: 2 * a * x^2 - 8 * x - 3 * a^2 < 0)
  (h2: ∀ x, -1 < x -> x < b)
  : a = 2 ∧ b = 3 := sorry

theorem part_2_min_value
  (a b x y : ℝ)
  (h1: x > 0)
  (h2: y > 0)
  (h3: a = 2)
  (h4: b = 3)
  (h5: (a / x) + (b / y) = 1)
  : ∃ min_val : ℝ, min_val = 3 * x + 2 * y ∧ min_val = 24 := sorry

end part_1_conditions_part_2_min_value_l687_687368


namespace range_of_a_l687_687705

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687705


namespace cannot_be_smallest_l687_687220

theorem cannot_be_smallest (a b c : ℝ) (h_a : a ≠ 0)
  (symm_cond : ∀ t : ℝ, f (2 + t) = f (2 - t)) :
    ¬ (f(1) = min (f(-1)) (min (f(1)) (min (f(2)) (f(5)))))
    := sorry

where f (x : ℝ) : ℝ := a * x^2 + b * x + c

end cannot_be_smallest_l687_687220


namespace find_other_discount_l687_687125

def other_discount (list_price final_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : Prop :=
  let price_after_first_discount := list_price - (first_discount / 100) * list_price
  final_price = price_after_first_discount - (second_discount / 100) * price_after_first_discount

theorem find_other_discount : 
  other_discount 70 59.22 10 6 :=
by
  sorry

end find_other_discount_l687_687125


namespace range_of_a_l687_687746

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687746


namespace rectangle_area_l687_687284

noncomputable def width := 14
noncomputable def length := width + 6
noncomputable def perimeter := 2 * width + 2 * length
noncomputable def area := width * length

theorem rectangle_area (h1 : length = width + 6) (h2 : perimeter = 68) : area = 280 := 
by 
  have hw : width = 14 := by sorry 
  have hl : length = 20 := by sorry 
  have harea : area = 280 := by sorry
  exact harea

end rectangle_area_l687_687284


namespace replaced_person_weight_l687_687491

theorem replaced_person_weight (W : ℝ) (increase : ℝ) (new_weight : ℝ) (average_increase : ℝ) (number_of_persons : ℕ) :
  average_increase = 2.5 →
  new_weight = 70 →
  number_of_persons = 8 →
  increase = number_of_persons * average_increase →
  W + increase = W - replaced_weight + new_weight →
  replaced_weight = 50 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end replaced_person_weight_l687_687491


namespace water_volume_per_minute_l687_687199

theorem water_volume_per_minute 
  (depth : ℝ) (width : ℝ) (flow_kmph : ℝ)
  (h_depth : depth = 8) (h_width : width = 25) (h_flow_rate : flow_kmph = 8) :
  (width * depth * (flow_kmph * 1000 / 60)) = 26666.67 :=
by 
  have flow_m_per_min := flow_kmph * 1000 / 60
  have area := width * depth
  have volume_per_minute := area * flow_m_per_min
  sorry

end water_volume_per_minute_l687_687199


namespace fit_jack_apples_into_jill_basket_l687_687052

-- Conditions:
def jack_basket_full : ℕ := 12
def jack_basket_space : ℕ := 4
def jack_current_apples : ℕ := jack_basket_full - jack_basket_space
def jill_basket_capacity : ℕ := 2 * jack_basket_full

-- Proof statement:
theorem fit_jack_apples_into_jill_basket : jill_basket_capacity / jack_current_apples = 3 :=
by {
  sorry
}

end fit_jack_apples_into_jill_basket_l687_687052


namespace polynomials_with_rational_values_at_rationals_are_rational_coefficient_l687_687148

open Polynomial

theorem polynomials_with_rational_values_at_rationals_are_rational_coefficient
  (P : ℚ[X]) :
  (∀ q : ℚ, eval q P ∈ ℚ) → (∀ i : ℕ, coeff P i ∈ ℚ) :=
sorry

end polynomials_with_rational_values_at_rationals_are_rational_coefficient_l687_687148


namespace cos_arcsin_of_fraction_l687_687628

theorem cos_arcsin_of_fraction : ∀ x, x = 8 / 17 → x ∈ set.Icc (-1:ℝ) 1 → Real.cos (Real.arcsin x) = 15 / 17 :=
by
  intros x hx h_range
  rw hx
  have h : (x:ℝ)^2 + Real.cos (Real.arcsin x)^2 = 1 := Real.sin_sq_add_cos_sq (Real.arcsin x)
  sorry

end cos_arcsin_of_fraction_l687_687628


namespace find_x_l687_687302

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : x * floor x = 50) : x = 7.142857 :=
by
  sorry

end find_x_l687_687302


namespace perpendicular_line_l687_687495

theorem perpendicular_line 
  (a b c : ℝ) 
  (p : ℝ × ℝ) 
  (h₁ : p = (-1, 3)) 
  (h₂ : a * (-1) + b * 3 + c = 0) 
  (h₃ : a * p.fst + b * p.snd + c = 0) 
  (hp : a = 1 ∧ b = -2 ∧ c = 3) : 
  ∃ a₁ b₁ c₁ : ℝ, 
  a₁ * (-1) + b₁ * 3 + c₁ = 0 ∧ a₁ = 2 ∧ b₁ = 1 ∧ c₁ = -1 := 
by 
  sorry

end perpendicular_line_l687_687495


namespace part_1_a_part_1_b_part_2_part_3_l687_687925

-- Define the given data and the target calculations as constants and definitions in Lean
noncomputable def a (n : ℕ) (rate : ℕ → ℕ → ℝ) : ℕ := (rate n 1 * n).nat_floor
noncomputable def b (m n : ℕ) : ℝ := m / n
noncomputable def p (rates : List ℝ) : ℝ := (rates.sum) / (rates.length)
noncomputable def s (required : ℕ) (p : ℝ) : ℕ := (required / p).nat_floor

-- Example of specific cases
def rate_200 := 0.955
def rate_1000 := 0.954
def germination_rates := [0.94, rate_200, 0.946, rate_1000, 0.953, 0.9496]
def required_seedlings := 9500

-- Proof statements
theorem part_1_a : a 200 (λ _ _, rate_200) = 191 := by
  -- Skip actual proof
  sorry

theorem part_1_b : b 954 1000 = 0.954 := by
  -- Skip actual proof
  sorry

theorem part_2 : p germination_rates = 0.95 := by
  -- Skip actual proof
  sorry

theorem part_3 : s required_seedlings (p germination_rates) = 10000 := by
  -- Skip actual proof
  sorry

end part_1_a_part_1_b_part_2_part_3_l687_687925


namespace age_ratio_l687_687936

theorem age_ratio (my_age : ℕ) (mother_age : ℕ) 
  (h1 : my_age + mother_age = 40) (h2 : my_age = 10) : 
  mother_age.toRat / my_age.toRat = 3 :=
by
  sorry

end age_ratio_l687_687936


namespace sum_of_squares_of_roots_l687_687301

theorem sum_of_squares_of_roots:
  let a := 1
  let b := -15
  let c := 7
  let Δ := b^2 - 4*a*c
  let r1 := (-b + Real.sqrt Δ) / (2*a)
  let r2 := (-b - Real.sqrt Δ) / (2*a)
  r1^2 + r2^2 = 211 :=
by
  let a := 1
  let b := -15
  let c := 7
  let Δ := b^2 - 4*a*c
  let r1 := (-b + Real.sqrt Δ) / (2*a)
  let r2 := (-b - Real.sqrt Δ) / (2*a)
  have sum_of_roots : r1 + r2 = -b / a := sorry
  have prod_of_roots : r1 * r2 = c / a := sorry
  calc
    r1^2 + r2^2
        = (r1 + r2)^2 - 2 * r1 * r2 : sorry
    ... = (15)^2 - 2 * 7 : sorry
    ... = 225 - 14 : sorry
    ... = 211 : sorry

end sum_of_squares_of_roots_l687_687301


namespace distance_focus_to_directrix_l687_687811

variable (p : ℝ) (y : ℝ)

-- We define the parabola and the conditions given in the problem.
def parabola := ∃ (p : ℝ), p > 0 ∧ ∀ y : ℝ, (y^2 = 2 * p * 6)

-- We display the condition of distance from the point with x = 6 to the focus being 10.
def distance_to_focus := 10 = real.sqrt ((6 - p/2)^2 + y^2)

-- The assertion we aim to prove.
theorem distance_focus_to_directrix : parabola p → distance_to_focus p y → p = 8 :=
by
  sorry

end distance_focus_to_directrix_l687_687811


namespace susan_ate_6_candies_l687_687240

-- Definitions for the conditions
def candies_tuesday : ℕ := 3
def candies_thursday : ℕ := 5
def candies_friday : ℕ := 2
def candies_left : ℕ := 4

-- The total number of candies bought during the week
def total_candies_bought : ℕ := candies_tuesday + candies_thursday + candies_friday

-- The number of candies Susan ate during the week
def candies_eaten : ℕ := total_candies_bought - candies_left

-- Theorem statement
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  unfold candies_eaten total_candies_bought candies_tuesday candies_thursday candies_friday candies_left
  sorry

end susan_ate_6_candies_l687_687240


namespace system_of_equations_solution_l687_687478

theorem system_of_equations_solution (x y : ℝ) (h1 : 2 * x ^ 2 - 5 * x + 3 = 0) (h2 : y = 3 * x + 1) : 
  (x = 1.5 ∧ y = 5.5) ∨ (x = 1 ∧ y = 4) :=
sorry

end system_of_equations_solution_l687_687478


namespace midpoint_parameter_eq_l687_687896

-- Definitions for given functions and parameters
def parametric_eq_x (a t θ : ℝ) : ℝ := a + t * Real.cos θ
def parametric_eq_y (b t θ : ℝ) : ℝ := b + t * Real.sin θ

variable (a b θ t1 t2 : ℝ)

-- Points B and C corresponding to parameter values t1 and t2
def x_B : ℝ := parametric_eq_x a t1 θ
def y_B : ℝ := parametric_eq_y b t1 θ
def x_C : ℝ := parametric_eq_x a t2 θ
def y_C : ℝ := parametric_eq_y b t2 θ

-- Midpoint M of segment BC
def x_M : ℝ := (x_B + x_C) / 2
def y_M : ℝ := (y_B + y_C) / 2

-- Target statement: midpoint parameter is (t1 + t2) / 2
theorem midpoint_parameter_eq : 
  ∃ tM, (parametric_eq_x a tM θ = x_M ∧ parametric_eq_y b tM θ = y_M) ↔ tM = (t1 + t2) / 2 :=
by 
  sorry -- The actual proof is not required

end midpoint_parameter_eq_l687_687896


namespace polyhedron_not_necessarily_prism_l687_687390

namespace Proof

-- Define the conditions
def isPolyhedronWithSpecifiedConditions (P : Type) [Polyhedron P] : Prop :=
  (∃ F1 F2 : Set Point, F1 ≠ F2 ∧ F1 ∥ F2) ∧
  ∀ (F : Set Point), F ≠ F1 ∧ F ≠ F2 → isParallelogram F

-- Define what it means for a polyhedron to be a prism
def isPrism (P : Type) [Polyhedron P] : Prop :=
  (∃ F1 F2 : Set Point, F1 ≠ F2 ∧ F1 ∥ F2 ∧ F1 ≅ F2) ∧
  ∀ (F : Set Point), F ≠ F1 ∧ F ≠ F2 → isParallelogram F

-- Prove that a polyhedron with the described conditions does not necessarily have to be a prism
theorem polyhedron_not_necessarily_prism (P : Type) [Polyhedron P] :
  isPolyhedronWithSpecifiedConditions P → ¬isPrism P :=
sorry

end Proof

end polyhedron_not_necessarily_prism_l687_687390


namespace eccentricity_of_hyperbola_l687_687366

noncomputable def hyperbola_eccentricity (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : b^2 = 4 * a^2) : ℝ :=
let c := Real.sqrt (a^2 + b^2) in
c / a

theorem eccentricity_of_hyperbola (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : b^2 = 4 * a^2) : hyperbola_eccentricity a b h₀ h₁ h₂ = Real.sqrt 5 :=
by sorry

end eccentricity_of_hyperbola_l687_687366


namespace pioneers_club_attendance_l687_687585

theorem pioneers_club_attendance :
  ∃ (A B : (Fin 11)), A ≠ B ∧
  (∃ (clubs_A clubs_B : Finset (Fin 5)), clubs_A = clubs_B) :=
by
  sorry

end pioneers_club_attendance_l687_687585


namespace seth_spent_more_l687_687102

def cost_ice_cream (cartons : ℕ) (price : ℕ) := cartons * price
def cost_yogurt (cartons : ℕ) (price : ℕ) := cartons * price
def amount_spent (cost_ice : ℕ) (cost_yog : ℕ) := cost_ice - cost_yog

theorem seth_spent_more :
  amount_spent (cost_ice_cream 20 6) (cost_yogurt 2 1) = 118 := by
  sorry

end seth_spent_more_l687_687102


namespace scalene_triangle_no_division_l687_687858

-- Define a scalene triangle
structure ScaleneTriangle (α : Type) [LinearOrder α] :=
(a b c : α)
(h1 : a ≠ b)
(h2 : b ≠ c)
(h3 : c ≠ a)

-- The theorem to prove that a scalene triangle cannot be divided into two equal triangles
theorem scalene_triangle_no_division (α : Type) [LinearOrder α] (T : ScaleneTriangle α) : 
  ¬ ∃ (L : α → Prop), 
    (∀ x, L x → x ∈ (set.univ : set α)) ∧ 
    (∃ A B C : α, 
      T.a = A ∧ T.b = B ∧ T.c = C ∧ 
      (T.a ≤ T.b ∧ T.b ≤ T.c ∧ 
       ∃ P Q : α, P ≠ Q ∧
        (L P ∧ ¬ L Q ∧ (P = Q → T.a = T.b)) ∧ 
        (L Q ∧ ¬ L P ∧ (Q = P → T.b = T.c))) ∧ 
      (∃ A' B' C' : α, 
        A' = A ∧ B' = C ∧ C' = B ∧ 
        (A' ≤ B' ∧ B' ≤ C' ∧ 
         ∃ P' Q' : α, P' ≠ Q' ∧
          (L P' ∧ ¬ L Q' ∧ (P' = Q' → A' = B')) ∧ 
          (L Q' ∧ ¬ L P' ∧ (Q' = P' → B' = C'))))) :=
sorry

end scalene_triangle_no_division_l687_687858


namespace social_media_phone_ratio_l687_687250

/-- 
Given that Jonathan spends 8 hours on his phone daily and 28 hours on social media in a week, 
prove that the ratio of the time spent on social media to the total time spent on his phone daily is \( 1 : 2 \).
-/
theorem social_media_phone_ratio (daily_phone_hours : ℕ) (weekly_social_media_hours : ℕ) 
  (h1 : daily_phone_hours = 8) (h2 : weekly_social_media_hours = 28) :
  (weekly_social_media_hours / 7) / daily_phone_hours = 1 / 2 := 
by
  sorry

end social_media_phone_ratio_l687_687250


namespace circumcircles_intersect_at_one_point_l687_687168

variables {A B C D P Q N M : Type} [Points A B C D P Q N M]

-- Conditions
variables (ABCD_convex : Convex ABCD)
variables (P_midpoint_AC : Midpoint P A C)
variables (Q_midpoint_BD : Midpoint Q B D)
variables (PQ_intersection_AB_CD : Intersection PQ AB = N ∧ Intersection PQ CD = M)

-- Hypotheses for each triangle mentioned
variables (circle_ANP : Circumcircle ANP)
variables (circle_BNQ : Circumcircle BNQ)
variables (circle_CMP : Circumcircle CMP)
variables (circle_DMQ : Circumcircle DMQ)

-- Statement to prove that the circumcircles of triangles intersect at one point
theorem circumcircles_intersect_at_one_point :
  ∃ R, IsOnCircle R circle_ANP ∧ IsOnCircle R circle_BNQ ∧ IsOnCircle R circle_CMP ∧ IsOnCircle R circle_DMQ :=
sorry

end circumcircles_intersect_at_one_point_l687_687168


namespace gcd_diophantine_solutions_l687_687283
open Nat

theorem gcd_diophantine_solutions :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ (z + y^2 + x^3 = x * y * z) ∧ (x = gcd y z) ∧
  ((x = 1 ∧ y = 2 ∧ z = 5) ∨ (x = 1 ∧ y = 3 ∧ z = 5) ∨ (x = 2 ∧ y = 2 ∧ z = 4) ∨ (x = 2 ∧ y = 6 ∧ z = 4)) :=
by
  sorry

end gcd_diophantine_solutions_l687_687283


namespace min_value_frac_add_x_l687_687174

theorem min_value_frac_add_x (x : ℝ) (h : x > 3) : (∃ m, (∀ (y : ℝ), y > 3 → (4 / y - 3 + y) ≥ m) ∧ m = 7) :=
sorry

end min_value_frac_add_x_l687_687174


namespace find_a_b_find_max_a_l687_687359

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - x^2 - a * x

theorem find_a_b (h_tangent : ∀ x : ℝ, x = 0 → Deriv (f x - 2 * x) = b) : ∃ b : ℝ, a = -1 ∧ b = 1 :=
by
  -- Proof goes here
  sorry

theorem find_max_a (h_increasing : ∀ x : ℝ, Deriv (f x a) ≥ 0) : a ≤ (2 - 2 * Real.log 2) :=
by
  -- Proof goes here
  sorry

end find_a_b_find_max_a_l687_687359


namespace color123123123_l687_687648

open Function

abbreviation Ticket := String                 -- Define a ticket as a string of digits.
abbreviation Digit := Char                    -- Each digit can be a character 1, 2, or 3.
inductive Color | red | blue | green          -- Define the possible colors.

def isNineDigitUsing123 (s : Ticket) : Prop :=
  s.length = 9 ∧ (∀ c ∈ s.toList, c = '1' ∨ c = '2' ∨ c = '3')  -- Validate digits.

def differInAllNinePlaces (t1 t2 : Ticket) : Prop :=
  t1.length = 9 ∧ t2.length = 9 ∧ (∀ i, i < 9 → t1.get i ≠ t2.get i)  -- Check all nine places.

axiom ticketColor : Ticket → Color         -- Assume we have a function mapping tickets to colors.

-- Define the given facts as axioms:
axiom ticket1_is_red : ticketColor "122222222" = Color.red
axiom ticket2_is_green : ticketColor "222222222" = Color.green
axiom differentTicketsDifferentColors :
  ∀ t1 t2 : Ticket, differInAllNinePlaces t1 t2 → ticketColor t1 ≠ ticketColor t2

-- Define the main theorem: Determine the color of ticket "123123123"
theorem color123123123 :
  ticketColor "123123123" = Color.red := by
  sorry

end color123123123_l687_687648


namespace part1_part2_l687_687804

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + (a * x) / (x + 1)

theorem part1 (a : ℝ) : (∀ x ∈ Ioo 0 4, 0 ≤ ((x + 1)^2 + a * x)) ↔ a ≥ -4 :=
sorry

theorem part2 (a : ℝ) : (∃ x₀ y₀ : ℝ, y₀ = 2 * x₀ ∧ (∂ f a / ∂ x) x₀ = 2 ∧ y₀ = log x₀ + (a * x₀) / (x₀ + 1)) ↔ a = 4 :=
sorry

end part1_part2_l687_687804


namespace inequality_proof_l687_687074

theorem inequality_proof (a b c : ℝ) (n : ℕ) (habc : a * b * c = 1) (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hn : n ≥ 2) :
  (a / real.pow (b + c) (1 / n : ℝ)) + (b / real.pow (c + a) (1 / n : ℝ)) + (c / real.pow (a + b) (1 / n : ℝ)) ≥ 3 / real.pow 2 (1 / n : ℝ) :=
sorry

end inequality_proof_l687_687074


namespace g_is_even_and_symmetric_l687_687501

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin (2 * x) - Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (4 * x)

theorem g_is_even_and_symmetric :
  (∀ x : ℝ, g x = g (-x)) ∧ (∀ k : ℤ, g ((2 * k - 1) * π / 8) = 0) :=
by
  sorry

end g_is_even_and_symmetric_l687_687501


namespace probability_of_no_adjacent_standing_is_123_over_1024_l687_687113

def total_outcomes : ℕ := 2 ^ 10

 -- Define the recursive sequence a_n
def a : ℕ → ℕ 
| 0 => 1
| 1 => 1
| n + 2 => a (n + 1) + a n

lemma a_10_val : a 10 = 123 := by
  sorry

def probability_no_adjacent_standing (n : ℕ): ℚ :=
  a n / total_outcomes

theorem probability_of_no_adjacent_standing_is_123_over_1024 :
  probability_no_adjacent_standing 10 = 123 / 1024 := by
  rw [probability_no_adjacent_standing, total_outcomes, a_10_val]
  norm_num

end probability_of_no_adjacent_standing_is_123_over_1024_l687_687113


namespace g_property_g_26_values_count_and_sum_l687_687875

noncomputable def g : ℕ → ℕ :=
sorry

theorem g_property (a b : ℕ) : 2 * g (a^2 + b^2 + 1) = g a ^ 2 + g b ^ 2 :=
sorry

theorem g_26_values_count_and_sum :
  ∃ n s : ℕ, (let possible_values : Finset ℕ := {0, 1} in
               n = possible_values.card ∧
               s = possible_values.sum id ∧
               n * s = 2) :=
begin
  use 2,
  use 1,
  dsimp,
  split,
  { refl, },
  split,
  { refl, },
  { norm_num, }
end

end g_property_g_26_values_count_and_sum_l687_687875


namespace part1_part2_l687_687992

-- Part (1)
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 1) : (x - x^2 < Real.sin x) ∧ (Real.sin x < x) :=
sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = Real.cos (a * x) - Real.log (1 - x^2)) ∧ 
  (f' 0 = 0) ∧ (∃ x_max : ℝ, (f'' x_max < 0) ∧ (x_max = 0))) : a < -Real.sqrt 2 ∨ Real.sqrt 2 < a :=
sorry

end part1_part2_l687_687992


namespace ratio_of_areas_is_4_l687_687165

-- Using the given conditions:
def smaller_side_length (s : ℝ) := s
def smaller_diagonal (s : ℝ) : ℝ := s * Real.sqrt 2
def larger_diagonal (s : ℝ) : ℝ := 2 * smaller_diagonal s
def larger_side_length (s : ℝ) : ℝ := 2 * s
def smaller_area (s : ℝ) : ℝ := s^2
def larger_area (s : ℝ) : ℝ := (larger_side_length s)^2
def area_ratio (s : ℝ) : ℝ := larger_area s / smaller_area s

-- Prove the ratio of the areas is 4.
theorem ratio_of_areas_is_4 (s : ℝ) (h_pos : s > 0) : area_ratio s = 4 :=
by 
  -- Proof goes here
  sorry

end ratio_of_areas_is_4_l687_687165


namespace arithmetic_mean_neg8_to_6_l687_687958

theorem arithmetic_mean_neg8_to_6 : 
  let s := (-8:ℤ), e := 6
  let count := e - s + 1
  let integer_sum := ∑ i in Finset.range (e - s + 1), s + i
  let mean := integer_sum.toReal / count.toReal
  mean = -1.0 :=
by
  sorry

end arithmetic_mean_neg8_to_6_l687_687958


namespace bus_capacity_l687_687571

noncomputable def total_people_in_bus : ℕ :=
let left_seats := 15 in
let right_seats := left_seats - 3 in
let seat_capacity := 3 in
let back_seat_capacity := 8 in
(left_seats * seat_capacity) + (right_seats * seat_capacity) + back_seat_capacity

theorem bus_capacity : total_people_in_bus = 89 :=
by 
  let left_seats := 15 in
  let right_seats := left_seats - 3 in
  let seat_capacity := 3 in
  let back_seat_capacity := 8 in
  have people_left := left_seats * seat_capacity,
  have people_right := right_seats * seat_capacity,
  have total_people := people_left + people_right + back_seat_capacity,
  sorry

end bus_capacity_l687_687571


namespace cyclist_time_approx_l687_687189

-- Given definitions from conditions:
def length_hill : ℝ := 800 -- in meters
def speed_ascent_kmh : ℝ := 12.4 -- in km/h
def speed_ascent_ms : ℝ := (12.4 * 1000) / 3600 -- in m/s
def speed_descent_ms : ℝ := 3 * speed_ascent_ms -- in m/s

-- Translate the main claim into a Lean theorem statement:
theorem cyclist_time_approx (length_hill : ℝ) (speed_ascent_kmh : ℝ) (speed_ascent_ms : ℝ) (speed_descent_ms : ℝ) :
  (length_hill / speed_ascent_ms + length_hill / speed_descent_ms) / 60 ≈ 5.168 :=
by
  sorry

end cyclist_time_approx_l687_687189


namespace cannot_cut_all_heads_l687_687423

theorem cannot_cut_all_heads
  (heads_initial : ℕ)
  (heads_initial = 100)
  (sword1_cut : ℕ)
  (sword1_cut = 21)
  (sword2_cut : ℕ)
  (sword2_cut = 4)
  (sword2_growth : ℕ)
  (sword2_growth = 2006) :
  ∀ n : ℕ, (heads_initial - sword1_cut * n) % 7 = 2 → (heads_initial - sword2_cut + sword2_growth) % 7 = 2 → (heads_initial) % 7 = 2 → ¬ ∃ k : ℕ, heads_initial - sword1_cut * k = 0 :=
by
  sorry

end cannot_cut_all_heads_l687_687423


namespace problem_statement_l687_687348

namespace MathProof

variables {R : Type*} [linear_ordered_field R] {f : R → R}

/-- 
Given that the function f(x) is decreasing on (8, +∞) and y = f(x + 8) is an even function, 
we need to prove that f(7) > f(10).
-/
theorem problem_statement (h_decreasing : ∀ x y : R, 8 < x ∧ x < y → f y < f x)
  (h_even : ∀ x : R, f (8 - x) = f (8 + x)) : f 7 > f 10 :=
sorry

end MathProof

end problem_statement_l687_687348


namespace find_a_l687_687796

noncomputable theory

open Real

def differentiate (f : ℝ → ℝ) (x : ℝ) : ℝ := (f(x + 1e-8) - f(x)) / 1e-8

def correct_a (f' : ℝ → ℝ) (a : ℝ) := f'(1) = -1

theorem find_a : ∃ a b : ℝ, correct_a (λ x, 3 * x^2 - 2 * a * x) a :=
by
  use [2, 0] -- use correct a and arbitrary b
  have f' : ℝ → ℝ := λ x, 3 * x^2 - 2 * 2 * x
  have h₁ : f'(1) = -1 := by sorry
  exact ⟨2, h₁⟩

end find_a_l687_687796


namespace functions_are_the_same_l687_687215

def f : ℝ → ℝ := fun x => (x - 1) ^ 0
def g : ℝ → ℝ := fun x => 1 / (x - 1) ^ 0
def domain (x : ℝ) : Prop := x ≠ 1

theorem functions_are_the_same : ∀ x, domain x → f x = g x := by
  intros x hx
  sorry

end functions_are_the_same_l687_687215


namespace remaining_nails_after_repairs_l687_687190

def fraction_used (perc : ℤ) (total : ℤ) : ℤ :=
  (total * perc) / 100

def after_kitchen (nails : ℤ) : ℤ :=
  nails - fraction_used 35 nails

def after_fence (nails : ℤ) : ℤ :=
  let remaining := after_kitchen nails
  remaining - fraction_used 75 remaining

def after_table (nails : ℤ) : ℤ :=
  let remaining := after_fence nails
  remaining - fraction_used 55 remaining

def after_floorboard (nails : ℤ) : ℤ :=
  let remaining := after_table nails
  remaining - fraction_used 30 remaining

theorem remaining_nails_after_repairs :
  after_floorboard 400 = 21 :=
by
  sorry

end remaining_nails_after_repairs_l687_687190


namespace cube_root_division_l687_687653

theorem cube_root_division (h₁ : (16.2:ℚ) = 81 / 5) :
  Real.cbrt (9 / 16.2) = Real.cbrt (5) / 3 :=
by
  sorry

end cube_root_division_l687_687653


namespace alex_silver_tokens_l687_687211

-- Definitions and conditions
def initialRedTokens : ℕ := 100
def initialBlueTokens : ℕ := 50
def firstBoothRedChange (x : ℕ) : ℕ := 3 * x
def firstBoothSilverGain (x : ℕ) : ℕ := 2 * x
def firstBoothBlueGain (x : ℕ) : ℕ := x
def secondBoothBlueChange (y : ℕ) : ℕ := 2 * y
def secondBoothSilverGain (y : ℕ) : ℕ := y
def secondBoothRedGain (y : ℕ) : ℕ := y

-- Final conditions when no more exchanges are possible
def finalRedTokens (x y : ℕ) : ℕ := initialRedTokens - firstBoothRedChange x + secondBoothRedGain y
def finalBlueTokens (x y : ℕ) : ℕ := initialBlueTokens + firstBoothBlueGain x - secondBoothBlueChange y

-- Total silver tokens calculation
def totalSilverTokens (x y : ℕ) : ℕ := firstBoothSilverGain x + secondBoothSilverGain y

-- Proof that in the end, Alex has 147 silver tokens
theorem alex_silver_tokens : 
  ∃ (x y : ℕ), finalRedTokens x y = 2 ∧ finalBlueTokens x y = 1 ∧ totalSilverTokens x y = 147 :=
by
  -- the proof logic will be filled here
  sorry

end alex_silver_tokens_l687_687211


namespace cannot_reach_reverse_order_l687_687403

open Equiv

def initial_permutation : Equiv.perm (Fin 3) := Equiv.mk id (λ x, x) (λ x, rfl) (λ x, rfl)

def valid_permutations : set (Equiv.perm (Fin 3)) := 
  {sigma : Equiv.perm (Fin 3) | ∀ i, sigma i ≠ i}

def no_fixpoint (sigma : Equiv.perm (Fin 3)) : Prop := 
  ∀ i, sigma i ≠ i

theorem cannot_reach_reverse_order (sigma : Equiv.perm (Fin 3)) (h : no_fixpoint sigma) :
  ¬ (sigma^(100) = fun _ => Fin.rotate 2 fin3 2) :=
by
  sorry

end cannot_reach_reverse_order_l687_687403


namespace range_of_a_l687_687713

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l687_687713


namespace PQRS_is_cyclic_l687_687881

-- Define the given cyclic quadrilateral and points
variables {A B C D P Q R S : Point}

-- Define the lengths of the segments
variable (hAB : dist A B = 7)
variable (hCD : dist C D = 8)
variable (hAP : dist A P = 3)
variable (hBQ : dist B Q = 3)
variable (hCR : dist C R = 2)
variable (hDS : dist D S = 2)

-- Define the cyclic nature of the quadrilateral ABCD
variable (hcyclic : cyclic_quad A B C D)

-- The goal is to prove that PQRS is a cyclic quadrilateral
theorem PQRS_is_cyclic : cyclic_quad P Q R S :=
sorry

end PQRS_is_cyclic_l687_687881


namespace area_square_WXYZ_is_89_l687_687411

-- Definitions based on problem conditions
variables (W X Y Z O M N : Type)
variables [square : Square W X Y Z]
variables [on_WZ : On M W Z] [on_WX : On N W X]
variables [orth_1 : Perpendicular WM NZ]

-- Given constants
variables (WO MO : ℝ)
variables [WO_is_8 : WO = 8] [MO_is_5 : MO = 5]

-- Proof statement
theorem area_square_WXYZ_is_89 :
  square_area W X Y Z = 89 := 
sorry

end area_square_WXYZ_is_89_l687_687411


namespace range_of_a_l687_687723

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687723


namespace minimum_value_expression_l687_687874

open Real

theorem minimum_value_expression (α β : ℝ) :
  ∃ x y : ℝ, x = 3 * cos α + 4 * sin β ∧ y = 3 * sin α + 4 * cos β ∧
    ((x - 7) ^ 2 + (y - 12) ^ 2) = 242 - 14 * sqrt 193 :=
sorry

end minimum_value_expression_l687_687874


namespace sprinklers_cover_proportion_l687_687843

noncomputable def proportion_of_lawn_covered (a : ℝ) : ℝ :=
  (real.pi + 3 - 3 * real.sqrt 3) / 3

theorem sprinklers_cover_proportion (a : ℝ) (h_pos : 0 < a) :
  (∑ i in (finset.range 4), 1/4 * π * a^2) / (a^2) = proportion_of_lawn_covered a :=
sorry

end sprinklers_cover_proportion_l687_687843


namespace inclination_angle_of_line_l687_687123

theorem inclination_angle_of_line (α : ℝ) (hα : 0 ≤ α ∧ α < Real.pi) :
  ∃ α, (sqrt 3 * x + 3 * y + 1 = 0) → tan α = - (sqrt 3 / 3) → α = 5 * Real.pi / 6 :=
by
  sorry


end inclination_angle_of_line_l687_687123


namespace cos_arcsin_l687_687624

theorem cos_arcsin (h : 8^2 + 15^2 = 17^2) : Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
by
  have opp_hyp_pos : 0 ≤ 8 / 17 := by norm_num
  have hyp_pos : 0 < 17 := by norm_num
  have opp_le_hyp : 8 / 17 ≤ 1 := by norm_num
  sorry

end cos_arcsin_l687_687624


namespace paint_rate_proof_l687_687505

def length : ℝ := 21.633307652783934
def breadth : ℝ := length / 3
def area : ℝ := length * breadth
def total_cost : ℝ := 624
def rate_per_sqm : ℝ := total_cost / area

theorem paint_rate_proof : rate_per_sqm = 4 := by
  -- This is where the proof would go
  sorry

end paint_rate_proof_l687_687505


namespace Kim_shirts_left_l687_687056

def initial_shirts : ℚ := 4.5 * 12
def bought_shirts : ℚ := 7
def lost_shirts : ℚ := 2
def fraction_given : ℚ := 2 / 5

theorem Kim_shirts_left :
  (initial_shirts + bought_shirts - lost_shirts) - ((fraction_given : ℚ) * (initial_shirts + bought_shirts - lost_shirts)).toInt = 36 := 
by
  sorry

end Kim_shirts_left_l687_687056


namespace toms_remaining_speed_l687_687945

-- Defining the constants and conditions
def total_distance : ℝ := 100
def first_leg_distance : ℝ := 50
def first_leg_speed : ℝ := 20
def avg_speed : ℝ := 28.571428571428573

-- Proving Tom's speed during the remaining part of the trip
theorem toms_remaining_speed :
  ∃ (remaining_leg_speed : ℝ),
    (remaining_leg_speed = 50) ∧
    (total_distance = first_leg_distance + 50) ∧
    ((first_leg_distance / first_leg_speed + 50 / remaining_leg_speed) = total_distance / avg_speed) :=
by
  sorry

end toms_remaining_speed_l687_687945


namespace num_customers_after_family_l687_687205

-- Definitions
def soft_taco_price : ℕ := 2
def hard_taco_price : ℕ := 5
def family_hard_tacos : ℕ := 4
def family_soft_tacos : ℕ := 3
def total_income : ℕ := 66

-- Intermediate values which can be derived
def family_cost : ℕ := (family_hard_tacos * hard_taco_price) + (family_soft_tacos * soft_taco_price)
def remaining_income : ℕ := total_income - family_cost

-- Proposition: Number of customers after the family
def customers_after_family : ℕ := remaining_income / (2 * soft_taco_price)

-- Theorem to prove the number of customers is 10
theorem num_customers_after_family : customers_after_family = 10 := by
  sorry

end num_customers_after_family_l687_687205


namespace percentage_increase_20_percent_edges_l687_687576

def percentage_increase_in_surface_area (L : ℝ) : ℝ :=
  let SA_original := 6 * L^2
  let L_new := 1.20 * L
  let SA_new := 6 * (L_new)^2
  let percentage_increase := ((SA_new - SA_original) / SA_original) * 100
  percentage_increase

theorem percentage_increase_20_percent_edges (L : ℝ) (hL : 0 < L) : 
  percentage_increase_in_surface_area L = 44 := 
by
  sorry

end percentage_increase_20_percent_edges_l687_687576


namespace factor_expression_l687_687281

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687281


namespace balanced_integers_count_l687_687612

def is_balanced_integer (n : ℕ) : Prop := 
  let (a,bc) := (n / 100, n % 100)
  let (b, c) := (bc / 10, bc % 10)
  a = b + c ∧ c % 2 = 0

noncomputable def count_balanced_integers : ℕ := 
  (Finset.range 900).filter (λ n => is_balanced_integer (n + 100)).card

theorem balanced_integers_count :
  count_balanced_integers = 18 :=
sorry

end balanced_integers_count_l687_687612


namespace platform_length_l687_687159

theorem platform_length 
  (train_length : ℝ) 
  (time_crossing_platform : ℝ) 
  (time_crossing_pole : ℝ) 
  (V : ℝ := train_length / time_crossing_pole)
  (L : ℝ := V * time_crossing_platform - train_length) :
  train_length = 300 → 
  time_crossing_platform = 39 → 
  time_crossing_pole = 24 → 
  L = 187.5 := 
by
  intros h_train_length h_time_platform h_time_pole
  simp only [V, L, h_train_length, h_time_platform, h_time_pole]
  sorry

end platform_length_l687_687159


namespace number_of_quarters_l687_687144

-- Defining constants for the problem
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_quarter : ℝ := 0.25

-- Given conditions
def total_dimes : ℝ := 3
def total_nickels : ℝ := 4
def total_pennies : ℝ := 200
def total_amount : ℝ := 5.00

-- Theorem stating the number of quarters found
theorem number_of_quarters :
  (total_amount - (total_dimes * value_dime + total_nickels * value_nickel + total_pennies * value_penny)) / value_quarter = 10 :=
by
  sorry

end number_of_quarters_l687_687144


namespace sequence_property_increasing_sequence_base8_representation_determine_a1998_l687_687869

noncomputable def a (n : ℕ) : ℕ := sorry -- Definition of the sequence which needs a formal proof later.

theorem sequence_property (n : ℕ) (i j k : ℕ) :
  ∃! (a : ℕ), a = (a n) + 2 * (a i) + 4 * (a j) :=
sorry -- Placeholder for the proof of unique representation requirement.

theorem increasing_sequence (m n : ℕ) (hmn : m < n) : (a m) < (a n) :=
sorry -- Placeholder for the proof that the sequence is increasing.

theorem base8_representation (n : ℕ) : ∀ b, b ∈ (a n)  → b = 0 ∨ b = 1 :=
sorry -- Placeholder for the proof regarding the base8 representation (only 0s and 1s).

theorem determine_a1998 : a 1998 = 17716 :=
sorry -- The final result that needs proof.

end sequence_property_increasing_sequence_base8_representation_determine_a1998_l687_687869


namespace vector_range_values_l687_687335

open Real

noncomputable theory

variables (a b c : ℝ × ℝ)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (h_perp : (a.1 * b.1 + a.2 * b.2) = 0)
variables (h_cond : ‖(c.1 - a.1 - b.1, c.2 - a.2 - b.2)‖ = 2)

theorem vector_range_values :
  2 - sqrt 2 ≤ ‖c‖ ∧ ‖c‖ ≤ 2 + sqrt 2 := sorry

end vector_range_values_l687_687335


namespace subset_implication_l687_687372

noncomputable def M (x : ℝ) : Prop := -2 * x + 1 ≥ 0
noncomputable def N (a x : ℝ) : Prop := x < a

theorem subset_implication (a : ℝ) :
  (∀ x, M x → N a x) → a > 1 / 2 :=
by
  sorry

end subset_implication_l687_687372


namespace tourist_tax_l687_687605

theorem tourist_tax (total_value : ℕ) (non_taxable_amount : ℕ) (tax_rate : ℚ) (tax : ℚ) : 
  total_value = 1720 → 
  non_taxable_amount = 600 → 
  tax_rate = 0.12 → 
  tax = (total_value - non_taxable_amount : ℕ) * tax_rate → 
  tax = 134.40 := 
by 
  intros total_value_eq non_taxable_amount_eq tax_rate_eq tax_eq
  sorry

end tourist_tax_l687_687605


namespace min_k_for_cube_root_difference_l687_687528

theorem min_k_for_cube_root_difference : 
  ∀ (s : Finset ℕ), s.card = 13 → (∀ {a b : ℕ}, a ∈ s → b ∈ s → a ≠ b → |Real.cbrt a - Real.cbrt b| < 1) :=
by
  sorry

end min_k_for_cube_root_difference_l687_687528


namespace factor_expression_l687_687261

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687261


namespace chord_length_of_circle_line_intersection_l687_687290

theorem chord_length_of_circle_line_intersection :
  let center := (2 : ℝ, -2 : ℝ)
  let radius := 3
  let line := (2 : ℝ, -1 : ℝ, -1 : ℝ) -- Ax + By + C = 0
  let d := |2 * 2 + (-1) * (-2) + (-1) | / Real.sqrt (2 ^ 2 + (-1) ^ 2)
  let L := 2 * Real.sqrt (radius ^ 2 - d ^ 2)
in
  L = 4 := by
  let center := (2 : ℝ, -2 : ℝ)
  let radius := (3 : ℝ)
  let line := (2 : ℝ, -1 : ℝ, -1 : ℝ)
  let d := |2 * 2 + (-1) * (-2) + (-1) | / Real.sqrt (2 ^ 2 + (-1) ^ 2)
  let L := 2 * Real.sqrt (radius ^ 2 - d ^ 2)
  sorry

end chord_length_of_circle_line_intersection_l687_687290


namespace tetrahedron_volume_l687_687169

theorem tetrahedron_volume (AB BC CD DA AC BD : ℝ)
  (hAB : AB = 1)
  (hBC : BC = 2 * real.sqrt 6)
  (hCD : CD = 5)
  (hDA : DA = 7)
  (hAC : AC = 5)
  (hBD : BD = 7) :
  let volume := (real.sqrt 66) / 2 in
  volume = (real.sqrt 66) / 2 := 
sorry

end tetrahedron_volume_l687_687169


namespace abcd_not_2012_l687_687422

theorem abcd_not_2012 
    (a b c d : ℤ) 
    (h : (a - b) * (c + d) = (a + b) * (c - d)) : 
    a * b * c * d ≠ 2012 :=
begin
  sorry
end

end abcd_not_2012_l687_687422


namespace talent_show_l687_687939

theorem talent_show (B G : ℕ) (h1 : G = B + 22) (h2 : G + B = 34) : G = 28 :=
by
  sorry

end talent_show_l687_687939


namespace max_point_derivative_condition_l687_687360

open Function

theorem max_point_derivative_condition (a : ℝ) (f : ℝ → ℝ) 
  (h_deriv : ∀ x, deriv f x = a * (x + 1) * (x - a)) 
  (h_max : ∀ x, x = a → f x = max (λ y, f y)) 
  (h_a_neg : a < 0) : -1 < a ∧ a < 0 :=
by
  sorry

end max_point_derivative_condition_l687_687360


namespace solution_set_of_inequality_l687_687672

def f (x : ℝ) : ℝ :=
if 0 ≤ x then 1 else -1

theorem solution_set_of_inequality (x : ℝ) :
  x + (x + 2) * f (x + 2) ≤ 5 ↔ x ∈ Set.Iic (3 / 2) := 
by {
  sorry
}

end solution_set_of_inequality_l687_687672


namespace john_min_pizzas_l687_687861

theorem john_min_pizzas (p : ℕ) :
  (∀ (earnings_per_pizza gas_per_pizza car_cost : ℕ), earnings_per_pizza = 10 → gas_per_pizza = 3 → car_cost = 5000 → p ≤ (car_cost * 7⁻¹).ceil → p = 715 → false) :=
  sorry

end john_min_pizzas_l687_687861


namespace jane_book_pages_l687_687050

theorem jane_book_pages (x : ℝ) :
  (x - (1 / 4 * x + 10) - (1 / 5 * (x - (1 / 4 * x + 10)) + 20) - (1 / 2 * (x - (1 / 4 * x + 10) - (1 / 5 * (x - (1 / 4 * x + 10)) + 20)) + 25) = 75) → x = 380 :=
by
  sorry

end jane_book_pages_l687_687050


namespace max_min_of_f_on_interval_l687_687291

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 4 + 4 * x ^ 3 + 34

theorem max_min_of_f_on_interval :
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ 50) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 50) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, 33 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 33) :=
by
  sorry

end max_min_of_f_on_interval_l687_687291


namespace maximum_daily_profit_l687_687204

variable (x : ℝ) (p : ℝ) (d : ℝ)

def daily_profit : ℝ := [200 - 10 * (x - 50)] * (x - 40)

theorem maximum_daily_profit :
  daily_profit 55 = 2250 := by
  sorry

end maximum_daily_profit_l687_687204


namespace ratatouille_cost_per_quart_l687_687472

theorem ratatouille_cost_per_quart :
  let eggplant_pounds := 5
  let zucchini_pounds := 4
  let tomato_pounds := 4
  let onion_pounds := 3
  let basil_pounds := 1
  let eggplant_zucchini_cost_per_pound := 2.0
  let tomato_cost_per_pound := 3.5
  let onion_cost_per_pound := 1.0
  let basil_cost_per_half_pound := 2.5
  let yield_quarts := 4
  let eggplant_zucchini_total_cost := (eggplant_pounds + zucchini_pounds) * eggplant_zucchini_cost_per_pound
  let tomato_total_cost := tomato_pounds * tomato_cost_per_pound
  let onion_total_cost := onion_pounds * onion_cost_per_pound
  let basil_total_cost := (basil_pounds / 0.5) * basil_cost_per_half_pound
  let total_cost := eggplant_zucchini_total_cost + tomato_total_cost + onion_total_cost + basil_total_cost
  total_cost / yield_quarts = 10 :=
by
  let eggplant_pounds := 5
  let zucchini_pounds := 4
  let tomato_pounds := 4
  let onion_pounds := 3
  let basil_pounds := 1
  let eggplant_zucchini_cost_per_pound := 2.0
  let tomato_cost_per_pound := 3.5
  let onion_cost_per_pound := 1.0
  let basil_cost_per_half_pound := 2.5
  let yield_quarts := 4
    
  let eggplant_zucchini_total_cost := (eggplant_pounds + zucchini_pounds) * eggplant_zucchini_cost_per_pound
  let tomato_total_cost := tomato_pounds * tomato_cost_per_pound
  let onion_total_cost := onion_pounds * onion_cost_per_pound
  let basil_total_cost := (basil_pounds / 0.5) * basil_cost_per_half_pound
  let total_cost := eggplant_zucchini_total_cost + tomato_total_cost + onion_total_cost + basil_total_cost

  have h_total_cost : total_cost = 40.0 := by sorry
  have h_cost_per_quart : total_cost / yield_quarts = 10 := by 
    rw [h_total_cost]
    exact sorry

  show total_cost / yield_quarts = 10 from h_cost_per_quart
  sorry

end ratatouille_cost_per_quart_l687_687472


namespace brothers_children_eq_2_l687_687045

def molly_package_cost : ℕ := 5
def molly_parents : ℕ := 2
def molly_brothers : ℕ := 3
def molly_total_cost : ℕ := 70

theorem brothers_children_eq_2
  (package_cost : ℕ := molly_package_cost)
  (parents : ℕ := molly_parents)
  (brothers : ℕ := molly_brothers)
  (total_cost : ℕ := molly_total_cost) :
  let spouses := brothers,
      total_immediate_family_and_spouses_packages := parents + brothers + spouses,
      cost_immediate_family_and_spouses := total_immediate_family_and_spouses_packages * package_cost,
      remaining_cost := total_cost - cost_immediate_family_and_spouses,
      total_children_packages := remaining_cost / package_cost,
      children_per_brother := total_children_packages / brothers in
  children_per_brother = 2 := by
  sorry

end brothers_children_eq_2_l687_687045


namespace Piper_gym_sessions_end_on_Monday_l687_687459

theorem Piper_gym_sessions_end_on_Monday : 
  (starts_on_Monday : Bool) → 
  (sessions : ℕ) → 
  (alternations : Bool) → 
  (each_day_except_Sunday : Bool) → 
  starts_on_Monday = true → sessions = 35 → alternations = true → each_day_except_Sunday = true → 
  day_of_week (70 % 7) = Monday :=
by
  sorry

end Piper_gym_sessions_end_on_Monday_l687_687459


namespace determine_c_l687_687179

noncomputable def parabola_c (a b c x y : ℝ) := y = a * x^2 + b * x + c

def vertex := (3 : ℝ, -5 : ℝ)
def passing_point := (1 : ℝ, -3 : ℝ)

theorem determine_c (a b c : ℝ) :
  parabola_c a b c 0 c → (vertex.1 - 3)^2 * a + vertex.2 = −5 →
  parabola_c a b c passing_point.1 passing_point.2 → c = -0.5 := 
sorry

end determine_c_l687_687179


namespace factor_expression_l687_687278

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687278


namespace minimum_value_of_a_plus_b_l687_687009

theorem minimum_value_of_a_plus_b :
  ∃ a b: ℕ, a > 0 ∧ b > 0 ∧ 4a + b = 13 ∧ ∀ a' b' : ℕ, a' > 0 ∧ b' > 0 ∧ 4a' + b' = 13 → (a + b ≤ a' + b') := 
begin
  sorry
end

end minimum_value_of_a_plus_b_l687_687009


namespace variance_angle_a_variance_angle_b_l687_687196

noncomputable def variance_of_angles (α β γ : ℝ) : ℝ :=
  (1 / 3) * ((α - 2 * π / 3) ^ 2 + (β - 2 * π / 3) ^ 2 + (γ - 2 * π / 3) ^ 2)

theorem variance_angle_a (α β γ : ℝ) (h : α + β + γ = 2 * π) :
  variance_of_angles α β γ < 10 * π ^ 2 / 27 :=
sorry

theorem variance_angle_b (α β γ : ℝ) (h : α + β + γ = 2 * π) :
  variance_of_angles α β γ < 2 * π ^ 2 / 9 :=
sorry

end variance_angle_a_variance_angle_b_l687_687196


namespace ball_travel_approximately_80_l687_687177

noncomputable def ball_travel_distance : ℝ :=
  let h₀ := 20
  let ratio := 2 / 3
  h₀ + -- first descent
  h₀ * ratio + -- first ascent
  h₀ * ratio + -- second descent
  h₀ * ratio^2 + -- second ascent
  h₀ * ratio^2 + -- third descent
  h₀ * ratio^3 + -- third ascent
  h₀ * ratio^3 + -- fourth descent
  h₀ * ratio^4 -- fourth ascent

theorem ball_travel_approximately_80 :
  abs (ball_travel_distance - 80) < 1 :=
sorry

end ball_travel_approximately_80_l687_687177


namespace min_k_for_cube_root_difference_l687_687532

theorem min_k_for_cube_root_difference (cards : Finset ℕ) (h : cards = Finset.range 2017) (selected : Finset ℕ) (h_selected : selected.card = 13) : 
  ∃ (a b : ℕ), a ∈ selected ∧ b ∈ selected ∧ a ≠ b ∧ (|real.cbrt a - real.cbrt b| < 1) :=
by
  sorry

end min_k_for_cube_root_difference_l687_687532


namespace product_of_conversions_l687_687251

-- Define the binary number 1101
def binary_number := 1101

-- Convert binary 1101 to decimal
def binary_to_decimal : ℕ := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the ternary number 212
def ternary_number := 212

-- Convert ternary 212 to decimal
def ternary_to_decimal : ℕ := 2 * 3^2 + 1 * 3^1 + 2 * 3^0

-- Statement to prove
theorem product_of_conversions : (binary_to_decimal) * (ternary_to_decimal) = 299 := by
  sorry

end product_of_conversions_l687_687251


namespace range_of_dot_product_l687_687019

-- Definitions based on given conditions
def side_length : ℝ := 1

def point_P (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : ℝ × ℝ := (x, 0)
def point_C : ℝ × ℝ := (1, 1)

-- Vector definitions directly derived from conditions
def vector_AP (x : ℝ) : ℝ × ℝ := point_P x ⟨x, le_rfl⟩
def vector_PB (x : ℝ) : ℝ × ℝ := (1 - x, 1)
def vector_PD (x : ℝ) : ℝ × ℝ := (x - 1, 0)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- The theorem to be proven
theorem range_of_dot_product (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : 
  dot_product (vector_AP x) (vector_PB x + vector_PD x) = 0 :=
by
  sorry

end range_of_dot_product_l687_687019


namespace min_distance_y_axis_l687_687332

open Real

noncomputable def distance (A B : ℝ × ℝ) : ℝ := 
  (sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))

theorem min_distance_y_axis (P : ℝ × ℝ) (h_on_y_axis : P.1 = 0) (A := (2, 5) : ℝ × ℝ) (B := (4, -1) : ℝ × ℝ) : 
  let B' := (-B.1, B.2) in
  let P' := (0, 3) in
  ∀ P, P = P' ∧ distance A P + distance B P = distance A P' + distance B P' :=
by
  sorry

end min_distance_y_axis_l687_687332


namespace smallest_number_after_removal_l687_687412

theorem smallest_number_after_removal (A : ℕ) (digits : List ℕ) 
  (hA : A = 12345678987654321) 
  (hSum : digits.sum = 60) 
  (hSubset : List.filter (λ d, ¬ d ∈ digits) (A.digits) = [4, 8, 9]) : 
  ∃ B : ℕ, B = 489 := 
by
  -- Detailed proof skipped
  sorry

end smallest_number_after_removal_l687_687412


namespace average_variance_correct_l687_687031

def avg_variance_scores (scores : List ℝ) :=
  let sorted_scores := scores.qsort (λ x y => x < y)
  let trimmed_scores := sorted_scores.drop 1 |>.dropLast 1
  let mean := trimmed_scores.sum / trimmed_scores.length
  let variance := (trimmed_scores.map (λ x => (x - mean) ^ 2)).sum / trimmed_scores.length
  (mean, variance)

theorem average_variance_correct : 
  avg_variance_scores [9.4, 8.4, 9.4, 9.9, 9.6, 9.4, 9.7] = (9.5, 0.016) :=
  sorry

end average_variance_correct_l687_687031


namespace range_of_a_l687_687694

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687694


namespace sum_of_differences_l687_687490

theorem sum_of_differences (x : ℝ) (h : (45 + x) / 2 = 38) : abs (x - 45) + abs (x - 30) = 15 := by
  sorry

end sum_of_differences_l687_687490


namespace range_of_a_l687_687688

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687688


namespace number_of_valid_subcommittees_l687_687134

theorem number_of_valid_subcommittees : 
  let total_members := 12
  let professors := 5
  let subcommittee_size := 4
  let total_subcommittees := Nat.choose total_members subcommittee_size
  let zero_prof_subcommittees := Nat.choose (total_members - professors) subcommittee_size
  let one_prof_subcommittees := professors * Nat.choose (total_members - professors) (subcommittee_size - 1)
  let non_valid_subcommittees := zero_prof_subcommittees + one_prof_subcommittees
  valid_subcommittees := 285 in
  total_subcommittees - non_valid_subcommittees = valid_subcommittees :=
by 
  sorry

end number_of_valid_subcommittees_l687_687134


namespace max_checkers_on_chessboard_l687_687548

theorem max_checkers_on_chessboard : 
  ∃ (w b : ℕ), (∀ r c : ℕ, r < 8 ∧ c < 8 → w = 2 * b) ∧ (8 * (w + b) = 48) ∧ (w + b) * 8 ≤ 64 :=
by sorry

end max_checkers_on_chessboard_l687_687548


namespace range_of_a_l687_687689

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687689


namespace area_of_ADEC_l687_687849

theorem area_of_ADEC (A B C D E F : Type) [HasCoordinates A Real]
  [HasCoordinates B Real] [HasCoordinates C Real] [HasCoordinates D Real]
  [HasCoordinates E Real] [HasCoordinates F Real] 
  (h1 : ∠ C = 90)
  (h2 : distance A D = distance D B)
  (h3 : line_through D E = perpendicular (line_through A B))
  (h4 : line_through C F = perpendicular (line_through A B))
  (h5 : F ≠ D)
  (h6 : distance A B = 24)
  (h7 : distance A C = 15) :
  area_of_quadrilateral A D E C = 135 := by 
  sorry

end area_of_ADEC_l687_687849


namespace ones_digit_of_8_pow_47_l687_687549

theorem ones_digit_of_8_pow_47 : (8^47) % 10 = 2 := 
  sorry

end ones_digit_of_8_pow_47_l687_687549


namespace triangle_ABC_BC_length_l687_687857

theorem triangle_ABC_BC_length :
  ∀ (A B C Y : Type) (AB AC : ℝ), 
  (AB = 60) ∧ 
  (AC = 85) ∧
  -- Circle with center A and radius AB intersects BC at B and Y
  (∀ (a : ℝ), a = AB → (BC = a ∧ BY + CY = BC)) ∧
  -- BY and CY are integers
  (∃ (BY CY : ℤ), BC = BY + CY) →
  (BC = 145) :=
by
  intros,
  sorry

end triangle_ABC_BC_length_l687_687857


namespace compute_expression_l687_687067

-- Define the roots of the polynomial.
variables (p q r : ℝ)
-- Define the polynomial.
def poly := (3 * x^3 - 4 * x^2 + 200 * x - 5) = 0

-- Vieta's formulas give us the sum of the roots.
def roots_sum := p + q + r = (4 / 3)

-- We need to prove the final expression equals the computed value.
theorem compute_expression 
  (h1 : poly p) (h2 : poly q) (h3 : poly r) (sum_h : roots_sum p q r) :
  ((p + q - 2) ^ 3 + (q + r - 2) ^ 3 + (r + p - 2) ^ 3) = (184 / 9) := 
sorry -- Proof to be filled in later.

end compute_expression_l687_687067


namespace speed_ratio_l687_687481

variable (d_A d_B : ℝ) (t_A t_B : ℝ)

-- Define the conditions
def condition1 : Prop := d_A = (1 + 1/5) * d_B
def condition2 : Prop := t_B = (1 - 1/11) * t_A

-- State the theorem that the speed ratio is 12:11
theorem speed_ratio (h1 : condition1 d_A d_B) (h2 : condition2 t_A t_B) :
  (d_A / t_A) / (d_B / t_B) = 12 / 11 :=
sorry

end speed_ratio_l687_687481


namespace ones_digit_of_8_pow_47_l687_687553

theorem ones_digit_of_8_pow_47 :
  (8^47) % 10 = 2 :=
by
  sorry

end ones_digit_of_8_pow_47_l687_687553


namespace integral_inequality_l687_687340

def continuous_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ set.Icc a b, continuous_at f x

theorem integral_inequality 
  (a b h : ℝ)
  (f g : ℝ → ℝ)
  (hf : continuous_on f a b)
  (hg : continuous_on g a b)
  (H1 : ∫ t in set.Icc 0 1, f t ∂ measure_theory.measure_space.volume 
       ≥ ∫ t in set.Icc 0 1, g t ∂ measure_theory.measure_space.volume)
  (H2 : ∫ t in set.Icc 0 h, f t ∂ measure_theory.measure_space.volume 
               = ∫ t in set.Icc a b, g t ∂ measure_theory.measure_space.volume) :
  (∫ x in set.Icc a b, x * f x ∂ measure_theory.measure_space.volume 
     ≤ ∫ x in set.Icc a b, x * g x ∂ measure_theory.measure_space.volume) :=
sorry

end integral_inequality_l687_687340


namespace percentage_passed_l687_687977

-- Definitions corresponding to the conditions
def F_H : ℝ := 25
def F_E : ℝ := 35
def F_B : ℝ := 40

-- Main theorem stating the question's proof.
theorem percentage_passed :
  (100 - (F_H + F_E - F_B)) = 80 :=
by
  -- we can transcribe the remaining process here if needed.
  sorry

end percentage_passed_l687_687977


namespace cream_ratio_l687_687053

theorem cream_ratio (john_coffee_initial jane_coffee_initial : ℕ)
  (john_drank john_added_cream jane_added_cream jane_drank : ℕ) :
  john_coffee_initial = 20 →
  jane_coffee_initial = 20 →
  john_drank = 3 →
  john_added_cream = 4 →
  jane_added_cream = 3 →
  jane_drank = 5 →
  john_added_cream / (jane_added_cream * 18 / (23 * 1)) = (46 / 27) := 
by
  intros
  sorry

end cream_ratio_l687_687053


namespace number_of_possible_tower_heights_l687_687282

-- Axiom for the possible increment values when switching brick orientations
def possible_increments : Set ℕ := {4, 7}

-- Base height when all bricks contribute the smallest dimension
def base_height (num_bricks : ℕ) (smallest_side : ℕ) : ℕ :=
  num_bricks * smallest_side

-- Check if a given height can be achieved by changing orientations of the bricks
def can_achieve_height (h : ℕ) (n : ℕ) (increments : Set ℕ) : Prop :=
  ∃ m k : ℕ, h = base_height n 2 + m * 4 + k * 7

-- Final proof statement
theorem number_of_possible_tower_heights :
  (50 : ℕ) = 50 →
  (∀ k : ℕ, (100 + k * 4 <= 450) → can_achieve_height (100 + k * 4) 50 possible_increments) →
  ∃ (num_possible_heights : ℕ), num_possible_heights = 90 :=
by
  sorry

end number_of_possible_tower_heights_l687_687282


namespace sin_bound_l687_687986

theorem sin_bound (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < sin x ∧ sin x < x := 
sorry

end sin_bound_l687_687986


namespace range_of_a_l687_687786

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687786


namespace professionals_not_using_any_l687_687225

-- Let L be the set of professionals using laptops.
-- Let T be the set of professionals using tablets.
-- Let C be the set of professionals drinking coffee.
-- Total number of professionals is represented by N.

theorem professionals_not_using_any:
  let N := 40
  let L := 18
  let T := 14
  let C := 16
  let LT := 7
  let TC := 4
  let LC := 5
  let LTC := 3 in
  let using_none := N - (L + T + C - LT - TC - LC + LTC) in
  using_none = 5 :=
by
  sorry

end professionals_not_using_any_l687_687225


namespace john_total_spent_in_usd_l687_687054

def cost_umbrella_eur : ℝ := 8
def cost_raincoat_eur : ℝ := 15
def cost_bag_eur : ℝ := 25

def discount_umbrella : ℝ := 0.10
def discount_raincoat : ℝ := 0.10
def discount_bag : ℝ := 0.05

def num_umbrella : ℕ := 2
def num_raincoat : ℕ := 3
def num_bag : ℕ := 1

def purchase_conversion_rate : ℝ := 1.15
def refund_conversion_rate : ℝ := 1.17

def restocking_fee_percent : ℝ := 0.20

def total_cost_usd_after_refund (n_umbrella n_raincoat n_bag : ℕ)
  (cost_u cost_r cost_b discount_u discount_r discount_b restocking_fee purch_conv_rate refund_conv_rate : ℝ) : ℝ :=
  let total_cost_before_discount_eur := n_umbrella * cost_u + n_raincoat * cost_r + n_bag * cost_b in
  let total_discount_eur := n_umbrella * cost_u * discount_u + n_raincoat * cost_r * discount_r + n_bag * cost_b * discount_b in
  let total_cost_after_discount_eur := total_cost_before_discount_eur - total_discount_eur in
  let total_cost_usd := total_cost_after_discount_eur * purch_conv_rate in

  let defective_item_cost_eur := (1 - discount_r) * cost_r in
  let refund_eur := (1 - restocking_fee) * defective_item_cost_eur in
  let refund_usd := refund_eur * refund_conv_rate in

  total_cost_usd - refund_usd

theorem john_total_spent_in_usd 
  : total_cost_usd_after_refund num_umbrella num_raincoat num_bag cost_umbrella_eur cost_raincoat_eur cost_bag_eur discount_umbrella discount_raincoat discount_bag restocking_fee_percent purchase_conversion_rate refund_conversion_rate = 77.81 := 
by 
  sorry

end john_total_spent_in_usd_l687_687054


namespace parametric_eq_line_product_of_distances_l687_687323

-- Definitions representing conditions of the problem
def point_P : ℝ × ℝ := (-1, 2)
def inclination_angle : ℝ := (2 * Real.pi) / 3
noncomputable def circle_equation (theta : ℝ) : ℝ := 2 * Real.cos (theta + Real.pi / 3)

-- Line l parametric equation
theorem parametric_eq_line :
  ∃ (t : ℝ), ∀ (x y : ℝ), x = -1 - (1/2) * t ∧ y = 2 + (Real.sqrt 3 / 2) * t :=
  sorry

-- Proposition about the product of distances |PM| * |PN|
theorem product_of_distances :
  ∃ (t1 t2 : ℝ), (t1 * t2 = 6 + 2 * Real.sqrt 3) := 
  sorry

end parametric_eq_line_product_of_distances_l687_687323


namespace mutually_exclusive_but_not_complementary_l687_687833

def EventAtLeastOneBlack (selection : set ball) : Prop :=
  ∃ b ∈ selection, b = black

def EventBothRed (selection : set ball) : Prop :=
  ∀ b ∈ selection, b = red

def EventBothBlack (selection : set ball) : Prop :=
  ∀ b ∈ selection, b = black

def EventAtLeastOneRed (selection : set ball) : Prop :=
  ∃ r ∈ selection, r = red

def EventExactlyOneBlack (selection : set ball) : Prop :=
  ∃ b ∈ selection, b = black ∧ ∃ r ∈ selection, r = red

def EventExactlyTwoBlack (selection : set ball) : Prop :=
  ∀ b ∈ selection, b = black ∧ set.card selection = 2

theorem mutually_exclusive_but_not_complementary :
  ∀ (selection : set ball),
    (EventExactlyOneBlack selection → ¬ EventExactlyTwoBlack selection) ∧
    (¬ EventExactlyOneBlack selection → ¬ EventExactlyTwoBlack selection) :=
sorry

end mutually_exclusive_but_not_complementary_l687_687833


namespace tangent_line_at_point_l687_687659

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 4 * x + 2
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 4 * x - 4

theorem tangent_line_at_point :
  f 1 = -3 ∧ f' 1 = -5 →
  ∃ k b, (λ x, k * x + b) == (λ x, 5 * x + -2) :=
by
  sorry

end tangent_line_at_point_l687_687659


namespace exists_n_for_all_digits_l687_687309

theorem exists_n_for_all_digits (a : ℕ) (n : ℕ) (h : n ≥ 4) :
  ∀ d, d ∈ digits 10 (n * (n + 1) / 2) → d = a ↔ a = 5 ∨ a = 6 :=
by
  sorry

end exists_n_for_all_digits_l687_687309


namespace distinct_diff_count_proof_l687_687816

open Set

def distinct_diff_count : Nat := 9

theorem distinct_diff_count_proof :
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let diffs := {d | ∃ a b ∈ s, a ≠ b ∧ d = abs (a - b)}
  countable (diffs \ {0}) = distinct_diff_count :=
by
  sorry

end distinct_diff_count_proof_l687_687816


namespace equal_telephone_bills_l687_687166

theorem equal_telephone_bills (m : ℕ) : 
  (7 + 0.25 * m = 12 + 0.20 * m) -> m = 100 :=
by
  sorry

end equal_telephone_bills_l687_687166


namespace max_value_of_g_l687_687926

def g : ℕ → ℕ 
| n := if n < 15 then n + 15 else g (n - 7)

theorem max_value_of_g : ∃ M, (∀ n, g n ≤ M) ∧ (∀ m, ((∀ n, g n ≤ m) → M ≤ m)) ∧ M = 29 := by
  sorry

end max_value_of_g_l687_687926


namespace quadratic_equation_with_root_l687_687656

theorem quadratic_equation_with_root :
  ∃ (a b c : ℚ), (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (a = 1) ∧ (b = 2) ∧ (c = -2) ∧
  (∀ x : ℝ, x = (√5 - 3) → a * x^2 + b * x + c = 0) :=
by
  existsi (1 : ℚ)
  existsi (2 : ℚ)
  existsi (-2 : ℚ)
  repeat { split }
  -- Coefficients verification
  · exact one_ne_zero
  · exact two_ne_zero
  · exact neg_ne_zero.mpr two_ne_zero
  -- Quadratic term coefficient
  · reflexivity
  -- Linear term coefficient
  · reflexivity
  -- Constant term coefficient
  · reflexivity
  -- Root verification
  · intros x hx
    rw [hx]
    ring_nf
    sorry -- Completing the proof is not necessary as per instructions

end quadratic_equation_with_root_l687_687656


namespace ellipse_hyperbola_same_foci_l687_687121

theorem ellipse_hyperbola_same_foci (n : ℝ) (h_pos : 0 < n)
    (h_ellipse : ∀ x y : ℝ, x^2 / 16 + y^2 / n^2 = 1)
    (h_hyperbola : ∀ x y : ℝ, x^2 / n^2 - y^2 / 4 = 1) :
    n = real.sqrt 6 :=
sorry

end ellipse_hyperbola_same_foci_l687_687121


namespace pet_store_problem_l687_687195

theorem pet_store_problem 
  (initial_puppies : ℕ) 
  (sold_day1 : ℕ) 
  (sold_day2 : ℕ) 
  (sold_day3 : ℕ) 
  (sold_day4 : ℕ)
  (sold_day5 : ℕ) 
  (puppies_per_cage : ℕ)
  (initial_puppies_eq : initial_puppies = 120) 
  (sold_day1_eq : sold_day1 = 25) 
  (sold_day2_eq : sold_day2 = 10) 
  (sold_day3_eq : sold_day3 = 30) 
  (sold_day4_eq : sold_day4 = 15) 
  (sold_day5_eq : sold_day5 = 28) 
  (puppies_per_cage_eq : puppies_per_cage = 6) : 
  (initial_puppies - (sold_day1 + sold_day2 + sold_day3 + sold_day4 + sold_day5)) / puppies_per_cage = 2 := 
by 
  sorry

end pet_store_problem_l687_687195


namespace find_g7_l687_687500

namespace ProofProblem

variable (g : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, g (x + y) = g x + g y)
variable (h2 : g 6 = 8)

theorem find_g7 : g 7 = 28 / 3 := by
  sorry

end ProofProblem

end find_g7_l687_687500


namespace minimum_value_of_f_l687_687080

def z1 (α : ℝ) : ℂ := complex.of_real (real.sin α) + complex.I * 2
def z2 (α : ℝ) : ℂ := complex.of_real 1 + complex.I * real.cos α

noncomputable def f (α : ℝ) : ℝ := (13 - complex.abs2 (z1 α + complex.I * z2 α)) / complex.abs (z1 α - complex.I * z2 α)

theorem minimum_value_of_f : ∃ α : ℝ, f α = 2 :=
by
  sorry

end minimum_value_of_f_l687_687080


namespace dryer_cost_l687_687210

theorem dryer_cost (washer_dryer_total_cost washer_cost dryer_cost : ℝ) (h1 : washer_dryer_total_cost = 1200) (h2 : washer_cost = dryer_cost + 220) :
  dryer_cost = 490 :=
by
  sorry

end dryer_cost_l687_687210


namespace sixth_result_l687_687164

theorem sixth_result (A : ℕ → ℕ) (h1 : (∑ i in finset.range 11, A i) = 660)
  (h2 : (∑ i in finset.range 6, A i) = 348) (h3 : (∑ i in finset.range 6 \shift_right 5, A (i + 5)) = 378) :
  A 5 = 66 :=
by
  sorry

end sixth_result_l687_687164


namespace coordinates_of_B_l687_687135

structure Point where
  x : Float
  y : Float

def symmetricWithRespectToY (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = A.y

theorem coordinates_of_B (A B : Point) 
  (hA : A.x = 2 ∧ A.y = -5)
  (h_sym : symmetricWithRespectToY A B) :
  B.x = -2 ∧ B.y = -5 :=
by
  sorry

end coordinates_of_B_l687_687135


namespace probability_prime_is_0_2389_l687_687599

-- Define a constant representing the range limit
def limit_n : ℕ := 120

-- Assume the probabilities p and 2p
constant p : ℝ

-- Assume the total sum of probabilities for all numbers must be 1
axiom h1 : 60 * p + 60 * (2 * p) = 1

-- Calculate p based on the axiom
def prob_p : ℝ := 1 / 180

-- Assume the number of primes in the range 1..120
def primes_less_or_equal_60 : ℕ := 17
def primes_greater_60 : ℕ := 13

-- Probabilities for primes in the two ranges
def prob_primes_leq_60 := (primes_less_or_equal_60 : ℝ) * prob_p
def prob_primes_gt_60 := (primes_greater_60 : ℝ) * 2 * prob_p

-- Total probability of choosing a prime number
def total_prob_prime : ℝ := prob_primes_leq_60 + prob_primes_gt_60 

-- Prove the total probability is approximately 0.2389
theorem probability_prime_is_0_2389 : total_prob_prime ≈ 0.2389 := sorry

end probability_prime_is_0_2389_l687_687599


namespace range_of_a_l687_687715

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l687_687715


namespace polynomial_remainder_l687_687556

-- Define the polynomials f(x) and g(x)
def f (x : ℚ) : ℚ := 3 * x^4 - 8 * x^3 + 20 * x^2 - 7 * x + 13
def g (x : ℚ) : ℚ := x^2 + 5 * x - 3

-- State the theorem that the remainder when dividing f(x) by g(x) is a specific polynomial
theorem polynomial_remainder :
  ∃ (r : ℚ[X]), degree r < degree (X^2 + 5 * X - 3 : ℚ[X]) ∧ f(X) = (X^2 + 5 * X - 3) * (leading_coeff (f(X)) / leading_coeff (X^2 + 5 * X - 3) * X^2) + r ∧ r = (168 * X^2 + 44 * X + 85 : ℚ[X]) :=
by
  -- Placeholder for the eventual proof steps
  sorry

end polynomial_remainder_l687_687556


namespace count_valid_subsets_l687_687131

theorem count_valid_subsets:
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∀ M : set ℕ, M ⊆ S ∧ M ≠ ∅ ∧ (∀ x ∈ M, (x + 3 ∈ M) ∨ (even x ∧ x / 2 ∈ M)) → 
    {M | M ⊆ S ∧ M ≠ ∅ ∧ ∀ x ∈ M, (x + 3 ∈ M) ∨ (even x ∧ x / 2 ∈ M)}.to_finset.card = 118 := 
sorry

end count_valid_subsets_l687_687131


namespace range_of_a_l687_687739

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687739


namespace meeting_point_l687_687890

-- Define the coordinates of Lucas and Jane
def Lucas_position : ℝ × ℝ := (2, 5)
def Jane_position : ℝ × ℝ := (10, 1)

def section_formula (p1 p2 : ℝ × ℝ) (m n : ℝ) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  ( (m * x2 + n * x1) / (m + n), (m * y2 + n * y1) / (m + n) )

theorem meeting_point :
  section_formula Lucas_position Jane_position 1 3 = (4, 4) :=
  sorry

end meeting_point_l687_687890


namespace average_and_variance_of_new_data_set_l687_687917

theorem average_and_variance_of_new_data_set
  (avg : ℝ) (var : ℝ) (constant : ℝ)
  (h_avg : avg = 2.8)
  (h_var : var = 3.6)
  (h_const : constant = 60) :
  (avg + constant = 62.8) ∧ (var = 3.6) :=
sorry

end average_and_variance_of_new_data_set_l687_687917


namespace volume_of_pyramid_l687_687117

-- Definitions and assumptions based on conditions
def isosceles_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  ∃ AK : RealNumber, AK = 9 ∧ ∃ BC : RealNumber, BC = 6

-- Definitions for the pyramid
def pyramid (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] : Prop :=
  ∃ DA DB DC : RealNumber, DA = 13 ∧ DB = 13 ∧ DC = 13

-- Theorem to prove the volume of the pyramid
theorem volume_of_pyramid (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (h₁ : isosceles_triangle A B C) (h₂ : pyramid A B C D) : 
  ∃ V : RealNumber, V = 108 :=
sorry

end volume_of_pyramid_l687_687117


namespace part1_part2_l687_687999

theorem part1 (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) : 
  x - x^2 < sin x ∧ sin x < x := sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (hf : f = λ x, cos (a * x) - log (1 - x^2)) 
  (hmax : (∀ (x : ℝ), 0 < x → x < 1 → f x < f 0) ∧ (∀ (x : ℝ), -1 < x → x < 0 → f x < f 0)) : 
  a < -sqrt 2 ∨ sqrt 2 < a := sorry

end part1_part2_l687_687999


namespace cost_of_hiring_actors_l687_687594

theorem cost_of_hiring_actors
  (A : ℕ)
  (CostOfFood : ℕ := 150)
  (EquipmentRental : ℕ := 300 + 2 * A)
  (TotalCost : ℕ := 3 * A + 450)
  (SellingPrice : ℕ := 10000)
  (Profit : ℕ := 5950) :
  TotalCost = SellingPrice - Profit → A = 1200 :=
by
  intro h
  sorry

end cost_of_hiring_actors_l687_687594


namespace balls_probability_l687_687535

theorem balls_probability : 
  let p := (∑ n in (range ⊤), (2:ℝ)^-(2+9*n))
  ∃ p, p = 1 / 2044 :=
by
  sorry

end balls_probability_l687_687535


namespace question1_question2_l687_687341

theorem question1 (m : ℝ) (x : ℝ) :
  (∀ x, x^2 - m * x + (m - 1) ≥ 0) → m = 2 :=
by
  sorry

theorem question2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (n = (a + 1 / b) * (2 * b + 1 / (2 * a))) → n ≥ (9 / 2) :=
by
  sorry

end question1_question2_l687_687341


namespace wooden_toy_price_l687_687932

noncomputable def price_of_hat : ℕ := 10
noncomputable def total_money : ℕ := 100
noncomputable def hats_bought : ℕ := 3
noncomputable def change_received : ℕ := 30
noncomputable def total_spent := total_money - change_received
noncomputable def cost_of_hats := hats_bought * price_of_hat

theorem wooden_toy_price :
  ∃ (W : ℕ), total_spent = 2 * W + cost_of_hats ∧ W = 20 := 
by 
  sorry

end wooden_toy_price_l687_687932


namespace original_number_of_movies_l687_687160

theorem original_number_of_movies (x : ℕ) (dvd blu_ray : ℕ)
  (h1 : dvd = 17 * x)
  (h2 : blu_ray = 4 * x)
  (h3 : 17 * x / (4 * x - 4) = 9 / 2) :
  dvd + blu_ray = 378 := by
  sorry

end original_number_of_movies_l687_687160


namespace average_of_x_l687_687010

theorem average_of_x (x : ℝ) (h : sqrt (3 * x^2 + 2) = sqrt 50) : 
    (1 / 2 * (x + - x)) = 0 :=
by
  sorry

end average_of_x_l687_687010


namespace hypotenuse_length_l687_687613

theorem hypotenuse_length (a c : ℝ) (h_perimeter : 2 * a + c = 36) (h_area : (1 / 2) * a^2 = 24) : c = 4 * Real.sqrt 6 :=
by
  sorry

end hypotenuse_length_l687_687613


namespace constant_term_expansion_l687_687119

theorem constant_term_expansion :
  let expr1 := (λ x : ℝ, x^2 - 3*x + 4/x)
  let expr2 := (λ x : ℝ, (1 - 1 / Real.sqrt x)^5)
  ∀ x : ℝ, x ≠ 0 → (expr1 x * expr2 x).is_constant_term (-25) :=
by
  sorry

end constant_term_expansion_l687_687119


namespace probability_sum_eq_9_l687_687565

open Set

def set_a : Set ℕ := {2, 3, 4, 5}
def set_b : Set ℕ := {4, 5, 6, 7, 8}

theorem probability_sum_eq_9 :
  ((countable_set { (x, y) | x ∈ set_a ∧ y ∈ set_b ∧ x + y = 9 }.to_finset.card : ℚ) /
  (countable_set { (x, y) | x ∈ set_a ∧ y ∈ set_b }.to_finset.card)) = 1 / 5 :=
by
  sorry

end probability_sum_eq_9_l687_687565


namespace coefficient_x2_y7_expansion_l687_687287

theorem coefficient_x2_y7_expansion : 
  let exp := (x + y) * (x - y)^8 in
  (coeff (exp : polynomial _) (x^2 * y^7) = 20) := 
  sorry

end coefficient_x2_y7_expansion_l687_687287


namespace range_of_a_l687_687724

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687724


namespace range_of_a_l687_687691

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l687_687691


namespace range_m_l687_687316

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (x m : ℝ) : ℝ := (1 / 2) ^ x - m

theorem range_m (m : ℝ) :
  (∀ x1 ∈ set.Icc (-1 : ℝ) 3, ∃ x2 ∈ set.Icc (0 : ℝ) 2, f x1 ≥ g x2 m) →
  m ≥ 1 / 4 :=
by
  sorry

end range_m_l687_687316


namespace question1_question2_l687_687982

-- Define required symbols and parameters
variables {x : ℝ} {b c : ℝ}

-- Statement 1: Proving b + c given the conditions on the inequality
theorem question1 (h : ∀ x, -1 < x ∧ x < 3 → 5*x^2 - b*x + c < 0) : b + c = -25 := sorry

-- Statement 2: Proving the solution set for the given inequality
theorem question2 (h : ∀ x, (2 * x - 5) / (x + 4) ≥ 0 → (x ≥ 5 / 2 ∨ x < -4)) : 
  {x | (2 * x - 5) / (x + 4) ≥ 0} = {x | x ≥ 5/2 ∨ x < -4} := sorry

end question1_question2_l687_687982


namespace part1_part2_l687_687995

-- Proof that for 0 < x < 1, x - x^2 < sin x < x
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := 
sorry

-- Proof that if x = 0 is a local maximum of f(x) = cos(ax) - ln(1 - x^2), then a is in the specified range.
theorem part2 (a : ℝ) (h : ∀ x, (cos(a * x) - log(1 - x^2))' (0) = 0 ∧ (cos(a * x) - log(1 - x^2))'' (0) < 0) : 
  a ∈ Set.Ioo (-∞) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) (∞) := 
sorry

end part1_part2_l687_687995


namespace part1_part2_l687_687997

-- Proof that for 0 < x < 1, x - x^2 < sin x < x
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := 
sorry

-- Proof that if x = 0 is a local maximum of f(x) = cos(ax) - ln(1 - x^2), then a is in the specified range.
theorem part2 (a : ℝ) (h : ∀ x, (cos(a * x) - log(1 - x^2))' (0) = 0 ∧ (cos(a * x) - log(1 - x^2))'' (0) < 0) : 
  a ∈ Set.Ioo (-∞) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) (∞) := 
sorry

end part1_part2_l687_687997


namespace cos_solution_l687_687384

theorem cos_solution (A : ℝ) (h : tan A + sec A = 3) : cos A = 3 / 5 :=
by
  sorry

end cos_solution_l687_687384


namespace triangle_ABC_obtuse_and_not_isosceles_l687_687800

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

def is_arithmetic_sequence (a b c : ℝ) : Prop := 2 * b = a + c

def on_curve (p : ℝ × ℝ) : Prop := p.snd = f p.fst

def is_increasing_order (a b c : ℝ) : Prop := a < b ∧ b < c

theorem triangle_ABC_obtuse_and_not_isosceles 
    (a b c : ℝ) 
    (ha : on_curve (a, f a)) 
    (hb : on_curve (b, f b)) 
    (hc : on_curve (c, f c)) 
    (h_seq : is_arithmetic_sequence a b c) 
    (h_incr : is_increasing_order a b c) 
    : (obtuse_triangle ((a, f a)) ((b, f b)) ((c, f c))) ∧ 
      (¬ right_triangle ((a, f a)) ((b, f b)) ((c, f c))) ∧ 
      (¬ isosceles_triangle ((a, f a)) ((b, f b)) ((c, f c))) := 
sorry


end triangle_ABC_obtuse_and_not_isosceles_l687_687800


namespace speed_of_train_is_correct_l687_687607

-- Definitions based on given conditions
def train_length : ℝ := 200
def platform_length : ℝ := 300.04
def crossing_time : ℝ := 25
def total_distance := train_length + platform_length
def speed_in_m_per_s := total_distance / crossing_time
def speed_in_km_per_h := speed_in_m_per_s * 3.6

-- Proof statement: The speed of the train is approximately 72.01 km/h
theorem speed_of_train_is_correct : abs (speed_in_km_per_h - 72.01) < 0.01 := 
by sorry

end speed_of_train_is_correct_l687_687607


namespace cone_surface_area_ratio_l687_687832

theorem cone_surface_area_ratio (l : ℝ) (h_l_pos : 0 < l) :
  let θ := (120 * Real.pi) / 180 -- converting 120 degrees to radians
  let side_area := (1/2) * l^2 * θ
  let r := l / 3
  let base_area := Real.pi * r^2
  let surface_area := side_area + base_area
  side_area ≠ 0 → 
  surface_area / side_area = 4 / 3 := 
by
  -- Provide the proof here
  sorry

end cone_surface_area_ratio_l687_687832


namespace game_is_unfair_l687_687564

def pencil_game_unfair : Prop :=
∀ (take1 take2 : ℕ → ℕ),
  take1 1 = 1 ∨ take1 1 = 2 →
  take2 2 = 1 ∨ take2 2 = 2 →
  ∀ n : ℕ,
    n = 5 → (∃ first_move : ℕ, (take1 first_move = 2) ∧ (take2 (take1 first_move) = 1 ∨ take2 (take1 first_move) = 2) ∧ (take1 (take2 (n - take1 first_move)) = 1 ∨ take1 (take2 (n - take1 first_move)) = 2) ∧
    ∀ second_move : ℕ, (second_move = n - first_move - take2 (n - take1 first_move)) → 
    n - first_move - take2 (n - take1 first_move) = 1 ∨ n - first_move - take2 (n - take1 first_move) = 2)

theorem game_is_unfair : pencil_game_unfair := 
sorry

end game_is_unfair_l687_687564


namespace ones_digit_of_8_pow_47_l687_687551

theorem ones_digit_of_8_pow_47 : (8^47) % 10 = 2 := 
  sorry

end ones_digit_of_8_pow_47_l687_687551


namespace finite_odd_divisors_condition_l687_687657

theorem finite_odd_divisors_condition (k : ℕ) (hk : 0 < k) :
  (∃ N : ℕ, ∀ n : ℕ, n > N → ¬ (n % 2 = 1 ∧ n ∣ k^n + 1)) ↔ (∃ c : ℕ, k + 1 = 2^c) :=
by sorry

end finite_odd_divisors_condition_l687_687657


namespace original_commission_l687_687559

theorem original_commission 
    (advance_fee incentive given_amount : ℝ) 
    (h1 : advance_fee = 8280) 
    (h2 : incentive = 1780) 
    (h3 : given_amount = 18500) : 
    ∃ C, C + incentive - advance_fee = given_amount ∧ C = 25000 :=
by
  use 25000
  rw [h1, h2, h3]
  norm_num  -- Simplify numeric expressions
  split
  · norm_num  -- First part of the split: check equality of expressions
  · rfl       -- Second part of the split: assert that C = 25000

end original_commission_l687_687559


namespace total_simple_interest_l687_687202

theorem total_simple_interest (P R T : ℝ) (hP : P = 6178.846153846154) (hR : R = 0.13) (hT : T = 5) :
    P * R * T = 4011.245192307691 := by
  rw [hP, hR, hT]
  norm_num
  sorry

end total_simple_interest_l687_687202


namespace part1_part2_l687_687998

theorem part1 (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) : 
  x - x^2 < sin x ∧ sin x < x := sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (hf : f = λ x, cos (a * x) - log (1 - x^2)) 
  (hmax : (∀ (x : ℝ), 0 < x → x < 1 → f x < f 0) ∧ (∀ (x : ℝ), -1 < x → x < 0 → f x < f 0)) : 
  a < -sqrt 2 ∨ sqrt 2 < a := sorry

end part1_part2_l687_687998


namespace no_real_m_divisible_by_x_minus_2_has_integer_roots_l687_687887

noncomputable def p (m : ℝ) (x : ℝ) : ℝ :=
  x^4 - 2 * x^3 - (m^2 + 3 * m) * x^2 + (6 * m^2 + 12 * m + 8) * x - 24 * m^2 - 48 * m - 32

theorem no_real_m_divisible_by_x_minus_2_has_integer_roots :
  ∀ (m : ℝ), ¬ (p m 2 = 0 ∧ (∀ x : ℝ, p m x = 0 → x ∈ Int)) :=
by
  sorry

end no_real_m_divisible_by_x_minus_2_has_integer_roots_l687_687887


namespace count_n_that_satisfy_conditions_l687_687382

theorem count_n_that_satisfy_conditions :
  {n : ℕ | n % 6 = 0 ∧ lcm (nat.factorial 6) n = 6 * gcd (nat.factorial 12) n}.to_finset.card = 270 :=
by sorry

end count_n_that_satisfy_conditions_l687_687382


namespace find_point_M_l687_687039

-- Definitions for conditions
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 4

def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (0, 4)
def axis_of_symmetry : ℝ := 1
def D : ℝ × ℝ := (1, 3)

theorem find_point_M (M : ℝ × ℝ) (a b : ℝ) :
  parabola a b -2 = 0 → 
  parabola a b 4 = 0 →
  parabola a b 0 = 4 →
  a = -1 / 2 →
  b = 1 →
  (M = (1, -1) ∨ M = (1, 7)) →
  (| D.snd - M.snd | = 4) →
  True := sorry

end find_point_M_l687_687039


namespace minimum_value_l687_687319

-- Defining the conditions
variables (x y : ℝ)

-- Defining the hypotheses
def conditions (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 1) :=
  ∀ (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 1), True

-- Goal statement
theorem minimum_value (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 1) :
  (1 / x + 1 / y) ≥ 4 + 2 * real.sqrt 3 :=
by
  sorry

end minimum_value_l687_687319


namespace factor_expression_l687_687254

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687254


namespace parallel_if_perp_to_plane_l687_687216

variable {α m n : Type}

variables (plane : α) (line_m line_n : m)

-- Define what it means for lines to be perpendicular to a plane
def perpendicular_to_plane (line : m) (pl : α) : Prop := sorry

-- Define what it means for lines to be parallel
def parallel (line1 line2 : m) : Prop := sorry

-- The conditions
axiom perp_1 : perpendicular_to_plane line_m plane
axiom perp_2 : perpendicular_to_plane line_n plane

-- The theorem to prove
theorem parallel_if_perp_to_plane : parallel line_m line_n := sorry

end parallel_if_perp_to_plane_l687_687216


namespace tomcat_next_to_thinner_queen_l687_687222

variables {Tomcat : Type} {Queen : Type}
variables [HasWeight Tomcat] [HasWeight Queen]
variables (T : List Tomcat) (Q : List Queen)
variables (thinner : Queen → Tomcat → Prop)

def fatter (t1 t2 : Tomcat) : Prop := weight t1 > weight t2
def thinner (thinner : Queen → Tomcat → Prop) (q : Queen) (t : Tomcat) : Prop := thinner q t

-- Given Conditions
-- There are 10 tomcats and 19 queens arranged in a row
axiom h1 : T.length = 10
axiom h2 : Q.length = 19
-- Each tomcat is fatter than each queen sitting next to her
axiom h3 : ∀ (t : Tomcat) (q : Queen), thinner q t

-- Question: Prove that next to each tomcat, there is a queen who is thinner than him.
theorem tomcat_next_to_thinner_queen :
  ∀ t ∈ T, ∃ q ∈ Q, thinner q t :=
by sorry

end tomcat_next_to_thinner_queen_l687_687222


namespace minimum_squared_distance_l687_687683

theorem minimum_squared_distance (a b c d : ℝ) 
    (h : (b + a^2 - 3 * real.log a)^2 + (c - d + 2)^2 = 0) :
    (a - c)^2 + (b - d)^2 = 8 :=
sorry

end minimum_squared_distance_l687_687683


namespace factor_expression_l687_687258

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687258


namespace repeating_decimal_to_fraction_l687_687149

theorem repeating_decimal_to_fraction :
  let x := (2 + 34/99) / 10 in
  x = 116 / 495 := by
  sorry

end repeating_decimal_to_fraction_l687_687149


namespace households_selected_l687_687838

theorem households_selected (H : ℕ) (M L S n h : ℕ)
  (h1 : H = 480)
  (h2 : M = 200)
  (h3 : L = 160)
  (h4 : H = M + L + S)
  (h5 : h = 6)
  (h6 : (h : ℚ) / n = (S : ℚ) / H) : n = 24 :=
by
  sorry

end households_selected_l687_687838


namespace factor_expression_l687_687272

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687272


namespace find_a_for_odd_function_l687_687386

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (2^x + 1)

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_a_for_odd_function :
  ∃ a : ℝ, (is_odd_function (f a)) ∧ f a 0 = 0 :=
begin
  use -1/2,
  split,
  { intros x,
    -- Use the given function definition to show that it is odd
    sorry },
  { -- Show that f(0) = 0
    sorry }
end

end find_a_for_odd_function_l687_687386


namespace right_angled_triangle_side_length_l687_687396

theorem right_angled_triangle_side_length :
  ∃ c : ℕ, (c = 5) ∧ (3^2 + 4^2 = c^2) ∧ (c = 4 + 1) := by
  sorry

end right_angled_triangle_side_length_l687_687396


namespace freken_bok_weight_l687_687055

variables (K F M : ℕ)

theorem freken_bok_weight 
  (h1 : K + F = M + 75) 
  (h2 : F + M = K + 45) : 
  F = 60 :=
sorry

end freken_bok_weight_l687_687055


namespace sqrt119_product_l687_687520

theorem sqrt119_product : (∃ (a b : ℕ), (a = 10) ∧ (b = 11) ∧ (√119 < 11) ∧ (√119 > 10) ∧ (a * b = 110)) :=
by
  sorry

end sqrt119_product_l687_687520


namespace pyramid_top_l687_687918

theorem pyramid_top (a₁ a₂ a₃ : ℕ) (h₁ : a₁ = 7) (h₂ : a₂ = 12) (h₃ : a₃ = 4) :
  let b₁ := a₁ + a₂,
      b₂ := a₂ + a₃,
      c₁ := b₁ + b₂
  in c₁ = 35 :=
by {
  intros,
  rw [h₁, h₂, h₃],
  dsimp [b₁, b₂, c₁],
  norm_num,
}

end pyramid_top_l687_687918


namespace car_deceleration_time_l687_687389

variables {V V₀ B k t : ℝ} -- Declare the variables as real numbers

theorem car_deceleration_time 
  (h1 : V = V₀ - B * t)
  (h2 : S = V₀ * t - (1/2) * k * t ^ 2) :
  t = (V₀ - V) / B := 
by sorry

end car_deceleration_time_l687_687389


namespace number_of_subsets_with_property_M_l687_687058

def has_property_M (A : Set ℤ) : Prop :=
  (∃ k ∈ A, k + 1 ∈ A) ∧ (∀ k ∈ A, k - 2 ∉ A)

def S : Set ℤ := {1, 2, 3, 4, 5, 6}

theorem number_of_subsets_with_property_M : 
  {t : Set ℤ | t ⊆ S ∧ t.card = 3 ∧ has_property_M t}.card = 6 :=
by
  sorry

end number_of_subsets_with_property_M_l687_687058


namespace range_of_a_l687_687736

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l687_687736


namespace range_of_a_l687_687744

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687744


namespace sams_speed_l687_687100

/-
Lean statement equivalent to the mathematical proof problem:
Prove that Sam's speed is 50 meters per minute given the specified conditions.
-/

theorem sams_speed (d_AC d_CB : ℕ) (time_total_minutes : ℕ) (d_AC_eq : d_AC = 600) (d_CB_eq : d_CB = 400) (time_total_eq : time_total_minutes = 20) : 
  let d_AB := d_AC + d_CB in 
  d_AB = 1000 → 
  let speed := d_AB / time_total_minutes in 
  speed = 50 :=
by
  intros 
  let d_AB := d_AC + d_CB
  have h1 : d_AB = 1000 := by rw [d_AC_eq, d_CB_eq]; exact rfl
  let speed := d_AB / time_total_minutes
  have h2 : speed = 50 := by rw [h1, time_total_eq]; norm_num
  exact h2
  sorry

end sams_speed_l687_687100


namespace triangle_ABC_area_l687_687024

open_locale big_operators

structure Triangle (α : Type*) :=
(A B C : α)

variables {α : Type*} [linear_ordered_field α] [normed_space α ℝ]

def segment_ratio (P Q R : α) (ratio : ℝ) :=
  P = (1 / (1 + ratio)) * Q + (ratio / (1 + ratio)) * R

variables (T : Triangle α)
def point_G := segment_ratio T.B T.C 3
def point_H := segment_ratio T.C T.A 3
def point_I := segment_ratio point_G T.A 2
noncomputable def area (triangle : Triangle α) : ℝ := sorry -- Assume a function that calculates area


theorem triangle_ABC_area {T : Triangle ℝ}
    (G : ℝ) (H : ℝ) (I : ℝ)
    (area_GHI : ℝ) :
    segment_ratio G T.B T.C 3 → 
    segment_ratio H T.A T.C 3 → 
    segment_ratio I T.A G 2 → 
    area ⟨G, H, I⟩ = area_GHI → 
    area_GHI = 10 → 
    area T = 120 :=
by
    sorry

end triangle_ABC_area_l687_687024


namespace probability_blue_or_green_is_two_thirds_l687_687955

-- Definitions for the given conditions
def blue_faces := 3
def red_faces := 2
def green_faces := 1
def total_faces := blue_faces + red_faces + green_faces
def successful_outcomes := blue_faces + green_faces

-- Probability definition
def probability_blue_or_green := (successful_outcomes : ℚ) / total_faces

-- The theorem we want to prove
theorem probability_blue_or_green_is_two_thirds :
  probability_blue_or_green = (2 / 3 : ℚ) :=
by
  -- here would be the proof steps, but we replace them with sorry as per the instructions
  sorry

end probability_blue_or_green_is_two_thirds_l687_687955


namespace range_of_a_l687_687774

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l687_687774


namespace cube_has_8_equilateral_triangles_l687_687133

/-- 
Given the geometric properties of a cube:
- it has 6 square faces
- it has 8 vertices such that each vertex is shared by three faces,
- the edges meeting at each vertex are mutually perpendicular,
- each face's diagonal can be used,

Prove that there are 8 equilateral triangles formed by three of its vertices.
-/
def number_of_equilateral_triangles : ℕ := 8

theorem cube_has_8_equilateral_triangles 
  (cube : Type) 
  (faces : ℕ := 6) 
  (vertices : ℕ := 8) 
  (vertex_shared_faces : ∀ v : vertices, 3) 
  (edges_perpendicular : ∀ v : vertices, mutually_perpendicular_edges v) 
  (diagonal_exists : ∀ f : faces, diagonal_of_square_face f)
 : number_of_equilateral_triangles = 8 := sorry

-- Definitions to satisfy the conditions
def mutually_perpendicular_edges (v : ℕ) : Prop := sorry
def diagonal_of_square_face (f : ℕ) : Prop := sorry

end cube_has_8_equilateral_triangles_l687_687133


namespace total_distance_l687_687432

def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def A : point := (-3, 6)
def B : point := (7, -3)
def C : point := (3, 3)
def O : point := (0, 0)

theorem total_distance :
  distance A C + distance C O + distance O B = 3 * real.sqrt 5 + 3 * real.sqrt 2 + real.sqrt 58 := by
  sorry

end total_distance_l687_687432


namespace range_of_a_l687_687726

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687726


namespace cost_of_milkshake_l687_687424

theorem cost_of_milkshake
  (initial_money : ℝ)
  (remaining_after_cupcakes : ℝ)
  (remaining_after_sandwich : ℝ)
  (remaining_after_toy : ℝ)
  (final_remaining : ℝ)
  (money_spent_on_milkshake : ℝ) :
  initial_money = 20 →
  remaining_after_cupcakes = initial_money - (1 / 4) * initial_money →
  remaining_after_sandwich = remaining_after_cupcakes - 0.30 * remaining_after_cupcakes →
  remaining_after_toy = remaining_after_sandwich - (1 / 5) * remaining_after_sandwich →
  final_remaining = 3 →
  money_spent_on_milkshake = remaining_after_toy - final_remaining →
  money_spent_on_milkshake = 5.40 :=
by
  intros 
  sorry

end cost_of_milkshake_l687_687424


namespace ratio_less_than_one_l687_687954

def prod_increasing (k j : ℕ) : ℕ := (list.range j).map (λ n, k + n).prod

def a := prod_increasing 2020 4
def b := prod_increasing 2120 4

theorem ratio_less_than_one : (a : ℝ) / (b : ℝ) < 1 := 
by {
  sorry
}

end ratio_less_than_one_l687_687954


namespace shaded_region_area_l687_687186

-- Definitions from the problem conditions
def small_circle_radius : ℝ := 4
def big_circle_radius : ℝ := small_circle_radius * 2
def small_circle_area : ℝ := real.pi * (small_circle_radius ^ 2)
def big_circle_area : ℝ := real.pi * (big_circle_radius ^ 2)

-- Lean theorem based on the math proof problem
theorem shaded_region_area :
  big_circle_area - small_circle_area = 48 * real.pi :=
by sorry

end shaded_region_area_l687_687186


namespace compute_expression_l687_687068

-- Define the roots of the polynomial.
variables (p q r : ℝ)
-- Define the polynomial.
def poly := (3 * x^3 - 4 * x^2 + 200 * x - 5) = 0

-- Vieta's formulas give us the sum of the roots.
def roots_sum := p + q + r = (4 / 3)

-- We need to prove the final expression equals the computed value.
theorem compute_expression 
  (h1 : poly p) (h2 : poly q) (h3 : poly r) (sum_h : roots_sum p q r) :
  ((p + q - 2) ^ 3 + (q + r - 2) ^ 3 + (r + p - 2) ^ 3) = (184 / 9) := 
sorry -- Proof to be filled in later.

end compute_expression_l687_687068


namespace arithmetic_mean_l687_687139

variables (x y z : ℝ)

def condition1 : Prop := 1 / (x * y) = y / (z - x + 1)
def condition2 : Prop := 1 / (x * y) = 2 / (z + 1)

theorem arithmetic_mean (h1 : condition1 x y z) (h2 : condition2 x y z) : x = (z + y) / 2 :=
by
  sorry

end arithmetic_mean_l687_687139


namespace max_min_values_l687_687684

open Real

noncomputable def circle_condition (x y : ℝ) :=
  (x - 3) ^ 2 + (y - 3) ^ 2 = 6

theorem max_min_values (x y : ℝ) (hx : circle_condition x y) :
  ∃ k k' d d', 
    k = 3 + 2 * sqrt 2 ∧
    k' = 3 - 2 * sqrt 2 ∧
    k = y / x ∧
    k' = y / x ∧
    d = sqrt ((x - 2) ^ 2 + y ^ 2) ∧
    d' = sqrt ((x - 2) ^ 2 + y ^ 2) ∧
    d = sqrt (10) + sqrt (6) ∧
    d' = sqrt (10) - sqrt (6) :=
sorry

end max_min_values_l687_687684


namespace chord_line_of_ellipse_through_point_l687_687826

theorem chord_line_of_ellipse_through_point
  (E F : Type)
  [AddCommGroup E] [AffineSpace E F]
  (p : E) (pt : E) 
  (H_ellipse_eq : ∀ x y : ℝ, (x^2 / 4) + (y^2 / 2) = 1)
  (H_bisected : ∀ (x1 y1 x2 y2 : ℝ), ∃ (p : affine_plane),
    ( (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 1 ) 
    ∧ ( (x1^2 / 4) + (y1^2 / 2) = 1 )
    ∧ ( (x2^2 / 4) + (y2^2 / 2) = 1 )):
  ∃ (k1 k2 : ℝ), (k1 * p + k2 * pt - 3 = 0) := sorry

end chord_line_of_ellipse_through_point_l687_687826


namespace triangle_side_lengths_l687_687145

-- Define the given constants and necessary hypotheses
variables {A B C F M G O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace F] [MetricSpace M] [MetricSpace G] [MetricSpace O]

-- Define the geometric and trigonometric properties of the triangles and points
def isosceles_triangle (ABC : Triangle A B C) : Prop :=
  (dist A C = dist B C)

def orthocenter (M : A) : Prop :=
  -- Assume definition of orthocenter
  sorry

def midpoint (F : A) (AB : Seg A B) : Prop :=
  -- Assume definition of midpoint
  sorry

def centroid_in_incircle (G : A) (incenter : Circle G) : Prop :=
  -- Assume definition of centroid being on incircle
  sorry

-- Define the given distance condition
def FM_distance (F M : A) : Prop :=
  dist F M = √6

-- Define the conclusion for side lengths
def side_lengths (AB : Seg A B) (AC BC : Seg A C) (a c : ℝ) : Prop :=
  dist A C = a ∧ dist B C = a ∧ dist A B = c

theorem triangle_side_lengths
  (ABC : Triangle A B C)
  (AB AC BC : Seg A B)
  (F M G : A)
  (h_iso : isosceles_triangle ABC)
  (h_midpoint : midpoint F AB)
  (h_ortho : orthocenter M)
  (h_centroid_incircle : centroid_in_incircle G inscribed_circle)
  (h_FM : FM_distance F M) :
  ∃ c a : ℝ, side_lengths AB AC BC a c :=
begin
  -- State the conclusion with the known side lengths
  use [24, 60],
  sorry
end

end triangle_side_lengths_l687_687145


namespace find_theta_l687_687378

-- Define vectors a, b and constraints on vector c
def a : (ℝ × ℝ) := (1, 3)
def b : (ℝ × ℝ) := (-2, -6)

variables (c : ℝ × ℝ)
noncomputable def c_mag := real.sqrt 10

-- Define the condition (a + b) · c = 5
def condition : Prop := 
  let ab := (a.1 + b.1, a.2 + b.2)
  ab.1 * c.1 + ab.2 * c.2 = 5

-- Define the angle between vectors a and c
def angle_between (u v : ℝ × ℝ) : ℝ := 
  real.acos ((u.1 * v.1 + u.2 * v.2) / (real.sqrt (u.1^2 + u.2^2) * c_mag))

-- Declare the theorem stating the angle θ between a and c is 120 degrees (in radians)
theorem find_theta {c : ℝ × ℝ} (h : condition) (hc : real.sqrt (c.1^2 + c.2^2) = c_mag):
  angle_between a c = real.pi * 2 / 3 :=
sorry

end find_theta_l687_687378


namespace david_weighted_average_l687_687641

def weighted_average_score (english_mark math_mark physics_mark chemistry_mark biology_mark : ℕ)
(weights : list ℝ) : ℝ :=
  let marks := [english_mark, math_mark, physics_mark, chemistry_mark, biology_mark].map (λ x, x / 100.0)
  in (weights.zip marks).sum (λ (w, m), w * m)

theorem david_weighted_average :
  weighted_average_score 70 60 78 60 65 [0.25, 0.20, 0.30, 0.15, 0.10] = 68.4 :=
  sorry

end david_weighted_average_l687_687641


namespace determinant_zero_l687_687650

def matrix_determinant (x y z : ℝ) : ℝ :=
  Matrix.det ![
    ![1, x, y + z],
    ![1, x + y, z],
    ![1, x + z, y]
  ]

theorem determinant_zero (x y z : ℝ) : matrix_determinant x y z = 0 := 
by
  sorry

end determinant_zero_l687_687650


namespace combined_points_correct_l687_687025

-- Definitions for the points scored by each player
def points_Lemuel := 7 * 2 + 5 * 3 + 4
def points_Marcus := 4 * 2 + 6 * 3 + 7
def points_Kevin := 9 * 2 + 4 * 3 + 5
def points_Olivia := 6 * 2 + 3 * 3 + 6

-- Definition for the combined points scored by both teams
def combined_points := points_Lemuel + points_Marcus + points_Kevin + points_Olivia

-- Theorem statement to prove combined points equals 128
theorem combined_points_correct : combined_points = 128 :=
by
  -- Lean proof goes here
  sorry

end combined_points_correct_l687_687025


namespace exists_positive_n_with_m_zeros_l687_687464

theorem exists_positive_n_with_m_zeros (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, 0 < n ∧ ∃ k : ℕ, 7^n = k * 10^m :=
sorry

end exists_positive_n_with_m_zeros_l687_687464


namespace angle_CBD_supplementary_to_ABC_l687_687420

theorem angle_CBD_supplementary_to_ABC
  (ABC : Type)
  [triangle ABC]
  (BAC ABC ACB CBD : angle ABC)
  (h1 : BAC = ABC - 20)
  (h2 : ACB = 50)
  (h3 : supplementary ABC CBD) :
  CBD = 105 := sorry

end angle_CBD_supplementary_to_ABC_l687_687420


namespace valid_pair_l687_687029

-- Definitions of the animals
inductive Animal
| lion
| tiger
| leopard
| elephant

open Animal

-- Given conditions
def condition1 (selected : Animal → Prop) : Prop :=
  selected lion → selected tiger

def condition2 (selected : Animal → Prop) : Prop :=
  ¬selected leopard → ¬selected tiger

def condition3 (selected : Animal → Prop) : Prop :=
  selected leopard → ¬selected elephant

-- Main theorem to prove
theorem valid_pair (selected : Animal → Prop) (pair : Animal × Animal) :
  (pair = (tiger, leopard)) ↔ 
  (condition1 selected ∧ condition2 selected ∧ condition3 selected) :=
sorry

end valid_pair_l687_687029


namespace dilute_solution_l687_687610

variable (original_solution_weight : ℝ) (original_solution_concentration : ℝ) (final_solution_concentration : ℝ)
variable (W : ℝ) (essence_amount : ℝ)

-- Condition 1: The original solution contains 60% essence
def condition1 : original_solution_concentration = 0.6 := rfl

-- Condition 2: The original solution weighs 15 ounces
def condition2 : original_solution_weight = 15 := rfl

-- Condition 3: The amount of essence in the original solution is 9 ounces
def condition3 : essence_amount = 9 := rfl

-- Condition 4: The final solution should contain 40% essence
def condition4 : final_solution_concentration = 0.4 := rfl

-- The theorem to prove given these conditions
theorem dilute_solution :
  essence_amount = final_solution_concentration * (original_solution_weight + W) → W = 7.5 :=
by
  intros
  sorry

end dilute_solution_l687_687610


namespace factor_expression_l687_687277

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l687_687277


namespace range_of_a_l687_687782

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l687_687782


namespace distance_from_O1_to_DE_l687_687398

open Real EuclideanGeometry

-- Given conditions
variables (O1 O2 A B C D E : Point)
variables (r1 r2 : ℝ) -- radii of circles O1 and O2
variables (AC AD : ℝ) (hAC : AC = 3) (hAD : AD = 6) (hr1 : r1 = 2)
variables (hO1 : ∃ (O1 : Point), is_circle O1 r1)
variables (hO2 : ∃ (O2 : Point), is_circle O2 r2)
variables (h1 h2 : ∃ (A B : Point), circle_intersections O1 r1 O2 r2 A B)
variables (hC : ∃ (C : Point), on_circle C O1 r1 ∧ ¬ on_circle C O2 r2)
variables (hD : ∃ (D : Point), line_meets_circle_at D O2 r2 (extended_line_through C A))
variables (hE : ∃ (E : Point), line_meets_circle_at E O2 r2 (extended_line_through C B))

-- Target statement: the distance from O1 to the line DE
theorem distance_from_O1_to_DE : 
  ∃ (O1G : ℝ), O1G = 19 / 4 := by 
  sorry

end distance_from_O1_to_DE_l687_687398


namespace triplet_D_not_sum_2_triplet_E_not_sum_2_l687_687967

def triplet_A : (ℚ × ℚ × ℚ) := (1/4, 1/2, 5/4)
def triplet_B : (ℚ × ℚ × ℚ) := (3, -1, 0)
def triplet_C : (ℚ × ℚ × ℚ) := (0.2, 0.5, 1.3)
def triplet_D : (ℚ × ℚ × ℚ) := (1.5, -2.4, 0.9)
def triplet_E : (ℚ × ℚ × ℚ) := (-7/3, -2/3, 3)

theorem triplet_D_not_sum_2 : (triplet_D.1 + triplet_D.2 + triplet_D.3) ≠ 2 :=
by {
  -- proof here
  sorry
}

theorem triplet_E_not_sum_2 : (triplet_E.1 + triplet_E.2 + triplet_E.3) ≠ 2 :=
by {
  -- proof here
  sorry
}

end triplet_D_not_sum_2_triplet_E_not_sum_2_l687_687967


namespace susan_money_left_l687_687647

-- Define the amount earned from jobs
def earnings_from_swimming : ℝ := 1200
def earnings_from_babysitting : ℝ := 600

-- Total earnings
def total_earnings : ℝ := earnings_from_swimming + earnings_from_babysitting

-- Amount spent on clothes and accessories
def spent_on_clothes : ℝ := 0.40 * total_earnings

-- Money left after buying clothes and accessories
def left_after_clothes : ℝ := total_earnings - spent_on_clothes

-- Amount spent on books
def spent_on_books : ℝ := 0.25 * left_after_clothes

-- Money left after buying books
def left_after_books : ℝ := left_after_clothes - spent_on_books

-- Amount spent on gifts
def spent_on_gifts : ℝ := 0.15 * left_after_books

-- Final amount of money left
def money_left : ℝ := left_after_books - spent_on_gifts

-- The theorem statement
theorem susan_money_left : money_left = 688.50 :=
by
  let total_earnings := earnings_from_swimming + earnings_from_babysitting
  let spent_on_clothes := 0.40 * total_earnings
  let left_after_clothes := total_earnings - spent_on_clothes
  let spent_on_books := 0.25 * left_after_clothes
  let left_after_books := left_after_clothes - spent_on_books
  let spent_on_gifts := 0.15 * left_after_books
  let money_left := left_after_books - spent_on_gifts
  calc
    money_left = left_after_books - spent_on_gifts := rfl
    ... = (left_after_clothes - 0.25 * left_after_clothes) - 0.15 * (left_after_clothes - 0.25 * left_after_clothes) := by ring
    ... = (left_after_clothes - 0.25 * left_after_clothes) - 0.15 * 0.75 * left_after_clothes := by rw [left_after_books]
    ... = (total_earnings - spent_on_clothes - 0.25 * (total_earnings - spent_on_clothes)) - 0.15 * 0.75 * (total_earnings - spent_on_clothes) := by rw [left_after_clothes]
    ... = (total_earnings - 0.4 * total_earnings - 0.25 * (total_earnings - 0.4 * total_earnings)) - 0.15 * 0.75 * (total_earnings - 0.4 * total_earnings) := by rw [spent_on_clothes]
    ... = (1 - 0.4 - 0.25 * (1 - 0.4)) * total_earnings - 0.15 * 0.75 * (1 - 0.4) * total_earnings := by ring
    ... = (0.6 - 0.25 * 0.6) * total_earnings - 0.15 * 0.75 * 0.6 * total_earnings := by ring
    ... = (0.6 - 0.15) * total_earnings - 0.0675 * total_earnings := by ring
    ... = 0.45 * total_earnings - 0.0675 * total_earnings := by ring
    ... = 0.3825 * total_earnings := by ring
    ... = 0.3825 * 1800 := by simp [total_earnings]
    ... = 688.5 := by norm_num

end susan_money_left_l687_687647


namespace average_mpg_proof_l687_687226

noncomputable def odometer_start := 58300
noncomputable def fuel_initial := 10
noncomputable def fuel_first_refill := 15
noncomputable def odometer_first_refill := 58700
noncomputable def fuel_second_refill := 25
noncomputable def odometer_end := 59275

def total_distance_traveled := odometer_end - odometer_start

def total_fuel_used := fuel_first_refill + fuel_second_refill

def average_mpg := total_distance_traveled / total_fuel_used

def rounded_average_mpg := Real.round (average_mpg * 10) / 10

theorem average_mpg_proof : rounded_average_mpg = 24.4 := 
by simp [odometer_start, fuel_initial, fuel_first_refill, odometer_first_refill, fuel_second_refill, odometer_end, total_distance_traveled, total_fuel_used, average_mpg, rounded_average_mpg]; sorry

end average_mpg_proof_l687_687226


namespace angle_OQE_iff_QE_eq_QF_l687_687583

open EuclideanGeometry

variables {A B C N P O Q E F : Point}
variables [IsAngleBisectorAngleBAC : Angle (A N O) = Angle (A N P)] [AngleANP : Angle (A N P) = 90] [AngleAPO : Angle (A P O) = 90]
variables [OnLineAB : on_line A B P] [OnLineAN : on_line A N O] [OnLineNP : on_line N P Q] [OnLineThroughQ : on_line Q E] [OnLineThroughQ : on_line Q F]
variables [IntersectionAB : on_line A B E] [IntersectionAC : on_line A C F]

theorem angle_OQE_iff_QE_eq_QF : angle (O Q E) = 90 ↔ dist Q E = dist Q F := 
by
  sorry

end angle_OQE_iff_QE_eq_QF_l687_687583


namespace seokjin_class_students_l687_687112

variables (J S T : ℕ)

-- Conditions
def cond1 : Prop := T = J + 3
def cond2 : Prop := J = S - 2
def cond3 : Prop := T = 35

-- Statement to prove
theorem seokjin_class_students (h1 : cond1) (h2 : cond2) (h3 : cond3) : S = 34 :=
by { sorry }

end seokjin_class_students_l687_687112


namespace length_of_train_l687_687208

-- Define the conditions
def speed_kmph := 54
def time_seconds := 12

-- Convert speed from km/h to m/s
def speed_mps := (speed_kmph : ℝ) * (5 / 18)

-- Define the theorem to prove the length of the train
theorem length_of_train : speed_mps * (time_seconds : ℝ) = 180 := 
by
  -- Calculation for verification
  calc
    speed_mps * (time_seconds : ℝ)
      = (54 * (5 / 18)) * 12 : by sorry  -- This stands for the proof steps of actual calculations
      ... = 15 * 12 : by sorry
      ... = 180 : by sorry

end length_of_train_l687_687208


namespace frog_escape_probability_l687_687030

def P (N : ℕ) : ℚ :=
match N with
| 0     => 0
| 12    => 1
| N + 1 => (N / 12) * P (N - 1) + (1 - (N / 12)) * P (N + 1)

theorem frog_escape_probability : P 3 = 16 / 37 := sorry

end frog_escape_probability_l687_687030


namespace range_of_a_l687_687735

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l687_687735


namespace range_of_a_l687_687707

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l687_687707
