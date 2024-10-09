import Mathlib

namespace al_original_portion_l169_16901

theorem al_original_portion {a b c d : ℕ} 
  (h1 : a + b + c + d = 2000)
  (h2 : a - 150 + 3 * b + 3 * c + d - 50 = 2500)
  (h3 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  a = 450 :=
sorry

end al_original_portion_l169_16901


namespace snack_eaters_initial_count_l169_16999

-- Define all variables and conditions used in the problem
variables (S : ℕ) (initial_people : ℕ) (new_outsiders_1 : ℕ) (new_outsiders_2 : ℕ) (left_after_first_half : ℕ) (left_after_second_half : ℕ) (remaining_snack_eaters : ℕ)

-- Assign the specific values according to conditions
def conditions := 
  initial_people = 200 ∧
  new_outsiders_1 = 20 ∧
  new_outsiders_2 = 10 ∧
  left_after_first_half = (S + new_outsiders_1) / 2 ∧
  left_after_second_half = left_after_first_half + new_outsiders_2 - 30 ∧
  remaining_snack_eaters = left_after_second_half / 2 ∧
  remaining_snack_eaters = 20

-- State the theorem to prove
theorem snack_eaters_initial_count (S : ℕ) (initial_people new_outsiders_1 new_outsiders_2 left_after_first_half left_after_second_half remaining_snack_eaters : ℕ) :
  conditions S initial_people new_outsiders_1 new_outsiders_2 left_after_first_half left_after_second_half remaining_snack_eaters → S = 100 :=
by sorry

end snack_eaters_initial_count_l169_16999


namespace cupcakes_left_over_l169_16985

def total_cupcakes := 40
def ms_delmont_class := 18
def mrs_donnelly_class := 16
def ms_delmont := 1
def mrs_donnelly := 1
def school_nurse := 1
def school_principal := 1

def total_given_away := ms_delmont_class + mrs_donnelly_class + ms_delmont + mrs_donnelly + school_nurse + school_principal

theorem cupcakes_left_over : total_cupcakes - total_given_away = 2 := by
  sorry

end cupcakes_left_over_l169_16985


namespace time_worked_together_l169_16992

noncomputable def combined_rate (P_rate Q_rate : ℝ) : ℝ :=
  P_rate + Q_rate

theorem time_worked_together (P_rate Q_rate : ℝ) (t additional_time job_completed : ℝ) :
  P_rate = 1 / 4 ∧ Q_rate = 1 / 15 ∧ additional_time = 1 / 5 ∧ job_completed = (additional_time * P_rate) →
  (t * combined_rate P_rate Q_rate + job_completed = 1) → 
  t = 3 :=
sorry

end time_worked_together_l169_16992


namespace problem1_problem2_l169_16914

namespace ArithmeticSequence

-- Part (1)
theorem problem1 (a1 : ℚ) (d : ℚ) (S_n : ℚ) (n : ℕ) (a_n : ℚ) 
  (h1 : a1 = 5 / 6) 
  (h2 : d = -1 / 6) 
  (h3 : S_n = -5) 
  (h4 : S_n = n * (2 * a1 + (n - 1) * d) / 2) 
  (h5 : a_n = a1 + (n - 1) * d) : 
  (n = 15) ∧ (a_n = -3 / 2) :=
sorry

-- Part (2)
theorem problem2 (d : ℚ) (n : ℕ) (a_n : ℚ) (a1 : ℚ) (S_n : ℚ)
  (h1 : d = 2) 
  (h2 : n = 15) 
  (h3 : a_n = -10) 
  (h4 : a_n = a1 + (n - 1) * d) 
  (h5 : S_n = n * (2 * a1 + (n - 1) * d) / 2) : 
  (a1 = -38) ∧ (S_n = -360) :=
sorry

end ArithmeticSequence

end problem1_problem2_l169_16914


namespace min_value_l169_16935

-- Given points A, B, and C and their specific coordinates
def A : (ℝ × ℝ) := (1, 3)
def B (a : ℝ) : (ℝ × ℝ) := (a, 1)
def C (b : ℝ) : (ℝ × ℝ) := (-b, 0)

-- Conditions
axiom a_pos (a : ℝ) : a > 0
axiom b_pos (b : ℝ) : b > 0
axiom collinear (a b : ℝ) : 3 * a + 2 * b = 1

-- The theorem to prove
theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hcollinear : 3 * a + 2 * b = 1) : 
  ∃ z, z = 11 + 6 * Real.sqrt 2 ∧ ∀ (x y : ℝ), (x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 1) -> (3 / x + 1 / y) ≥ z :=
by sorry -- Proof to be provided

end min_value_l169_16935


namespace operation_is_double_l169_16942

theorem operation_is_double (x : ℝ) (operation : ℝ → ℝ) (h1: x^2 = 25) (h2: operation x = x / 5 + 9) : operation x = 2 * x :=
by
  sorry

end operation_is_double_l169_16942


namespace two_cos_45_eq_sqrt_2_l169_16976

theorem two_cos_45_eq_sqrt_2 : 2 * Real.cos (pi / 4) = Real.sqrt 2 := by
  sorry

end two_cos_45_eq_sqrt_2_l169_16976


namespace minimum_value_expr_l169_16941

noncomputable def expr (x : ℝ) : ℝ := (x^2 + 11) / Real.sqrt (x^2 + 5)

theorem minimum_value_expr : ∃ x : ℝ, expr x = 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_expr_l169_16941


namespace intersection_P_Q_l169_16946

def P := {x : ℤ | x^2 - 16 < 0}
def Q := {x : ℤ | ∃ n : ℤ, x = 2 * n}

theorem intersection_P_Q :
  P ∩ Q = {-2, 0, 2} :=
sorry

end intersection_P_Q_l169_16946


namespace find_c_of_parabola_l169_16989

theorem find_c_of_parabola (a b c : ℝ) (h_vertex : ∀ x, y = a * (x - 3)^2 - 5)
                           (h_point : ∀ x y, (x = 1) → (y = -3) → y = a * (x - 3)^2 - 5)
                           (h_standard_form : ∀ x, y = a * x^2 + b * x + c) :
  c = -0.5 :=
sorry

end find_c_of_parabola_l169_16989


namespace largest_n_polynomials_l169_16986

theorem largest_n_polynomials :
  ∃ (P : ℕ → (ℝ → ℝ)), (∀ i j, i ≠ j → ∀ x, P i x + P j x ≠ 0) ∧ (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ∃ x, P i x + P j x + P k x = 0) ↔ n = 3 := 
sorry

end largest_n_polynomials_l169_16986


namespace carol_spending_l169_16907

noncomputable def savings (S : ℝ) : Prop :=
∃ (X : ℝ) (stereo_spending television_spending : ℝ), 
  stereo_spending = (1 / 4) * S ∧
  television_spending = X * S ∧
  stereo_spending + television_spending = 0.25 * S ∧
  (stereo_spending - television_spending) / S = 0.25

theorem carol_spending (S : ℝ) : savings S :=
sorry

end carol_spending_l169_16907


namespace smallest_possible_b_l169_16987

theorem smallest_possible_b (a b c : ℚ) (h1 : a < b) (h2 : b < c)
    (arithmetic_seq : 2 * b = a + c) (geometric_seq : c^2 = a * b) :
    b = 1 / 2 :=
by
  let a := 4 * b
  let c := 2 * b - a
  -- rewrite and derived equations will be done in the proof
  sorry

end smallest_possible_b_l169_16987


namespace votes_combined_l169_16926

theorem votes_combined (vote_A vote_B : ℕ) (h_ratio : vote_A = 2 * vote_B) (h_A_votes : vote_A = 14) : vote_A + vote_B = 21 :=
by
  sorry

end votes_combined_l169_16926


namespace calculate_expression_l169_16958

variable (x y : ℝ)

theorem calculate_expression :
  (-2 * x^2 * y)^3 = -8 * x^6 * y^3 :=
by 
  sorry

end calculate_expression_l169_16958


namespace find_number_l169_16920

theorem find_number (x : ℕ) (h : 3 * (2 * x + 8) = 84) : x = 10 :=
by
  sorry

end find_number_l169_16920


namespace analytical_expression_of_f_l169_16923

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b

theorem analytical_expression_of_f (a b : ℝ) (h_a : a > 0)
  (h_max : (∃ x_max : ℝ, f x_max a b = 5 ∧ (∀ x : ℝ, f x_max a b ≥ f x a b)))
  (h_min : (∃ x_min : ℝ, f x_min a b = 1 ∧ (∀ x : ℝ, f x_min a b ≤ f x a b))) :
  f x 3 1 = x^3 + 3 * x^2 + 1 := 
sorry

end analytical_expression_of_f_l169_16923


namespace fraction_solution_l169_16934

theorem fraction_solution (x : ℝ) (h : 4 - 9 / x + 4 / x^2 = 0) : 3 / x = 12 ∨ 3 / x = 3 / 4 :=
by
  -- Proof to be written here
  sorry

end fraction_solution_l169_16934


namespace locus_points_eq_distance_l169_16968

def locus_is_parabola (x y : ℝ) : Prop :=
  (y - 1) ^ 2 = 16 * (x - 2)

theorem locus_points_eq_distance (x y : ℝ) :
  locus_is_parabola x y ↔ (x, y) = (4, 1) ∨
    dist (x, y) (4, 1) = dist (x, y) (0, y) :=
by
  sorry

end locus_points_eq_distance_l169_16968


namespace suitable_altitude_range_l169_16922

theorem suitable_altitude_range :
  ∀ (temperature_at_base : ℝ) (temp_decrease_per_100m : ℝ) (suitable_temp_low : ℝ) (suitable_temp_high : ℝ) (altitude_at_base : ℝ),
  (22 = temperature_at_base) →
  (0.5 = temp_decrease_per_100m) →
  (18 = suitable_temp_low) →
  (20 = suitable_temp_high) →
  (0 = altitude_at_base) →
  400 ≤ ((temperature_at_base - suitable_temp_high) / temp_decrease_per_100m * 100) ∧ ((temperature_at_base - suitable_temp_low) / temp_decrease_per_100m * 100) ≤ 800 :=
by
  intros temperature_at_base temp_decrease_per_100m suitable_temp_low suitable_temp_high altitude_at_base
  intro h1 h2 h3 h4 h5
  sorry

end suitable_altitude_range_l169_16922


namespace prove_system_of_inequalities_l169_16952

theorem prove_system_of_inequalities : 
  { x : ℝ | x / (x - 2) ≥ 0 ∧ 2 * x + 1 ≥ 0 } = Set.Icc (-(1:ℝ)/2) 0 ∪ Set.Ioi 2 := 
by
  sorry

end prove_system_of_inequalities_l169_16952


namespace max_three_m_plus_four_n_l169_16918

theorem max_three_m_plus_four_n (m n : ℕ) 
  (h : m * (m + 1) + n ^ 2 = 1987) : 3 * m + 4 * n ≤ 221 :=
sorry

end max_three_m_plus_four_n_l169_16918


namespace expenditure_representation_l169_16916

def income_represented_pos (income : ℤ) : Prop := income > 0

def expenditure_represented_neg (expenditure : ℤ) : Prop := expenditure < 0

theorem expenditure_representation (income expenditure : ℤ) (h_income: income_represented_pos income) (exp_value: expenditure = 3) : expenditure_represented_neg expenditure := 
sorry

end expenditure_representation_l169_16916


namespace problem_1_problem_2a_problem_2b_l169_16980

noncomputable def v_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def v_b : ℝ × ℝ := (3, -Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (v_a x).1 * (v_b).1 + (v_a x).2 * (v_b).2

theorem problem_1 (x : ℝ) (h : x ∈ Set.Icc 0 Real.pi) : 
  (v_a x).1 * (v_b).2 = (v_a x).2 * (v_b).1 → x = (5 * Real.pi / 6) :=
by
  sorry

theorem problem_2a : 
  ∃ x ∈ Set.Icc 0 Real.pi, f x = 3 ∧ ∀ y ∈ Set.Icc 0 Real.pi, f y ≤ 3 :=
by
  sorry

theorem problem_2b :
  ∃ x ∈ Set.Icc 0 Real.pi, f x = -2 * Real.sqrt 3 ∧ ∀ y ∈ Set.Icc 0 Real.pi, f y ≥ -2 * Real.sqrt 3 :=
by
  sorry

end problem_1_problem_2a_problem_2b_l169_16980


namespace part1_part2_l169_16917

theorem part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) : 
  (1 / a) + (1 / (b + 1)) ≥ 4 / 5 := 
by 
  sorry

theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 8) : 
  a + b ≥ 4 := 
by 
  sorry

end part1_part2_l169_16917


namespace solve_equation_l169_16921

noncomputable def smallest_solution : ℝ :=
(15 - Real.sqrt 549) / 6

theorem solve_equation :
  ∃ x : ℝ, 
    (3 * x / (x - 3) + (3 * x^2 - 27) / x = 18) ∧
    x = smallest_solution :=
by
  sorry

end solve_equation_l169_16921


namespace prob_x_lt_y_is_correct_l169_16971

open Set

noncomputable def prob_x_lt_y : ℝ :=
  let rectangle := Icc (0: ℝ) 4 ×ˢ Icc (0: ℝ) 3
  let area_rectangle := 4 * 3
  let triangle := {p : ℝ × ℝ | p.1 ∈ Icc (0: ℝ) 3 ∧ p.2 ∈ Icc (0: ℝ) 3 ∧ p.1 < p.2}
  let area_triangle := 1 / 2 * 3 * 3
  let probability := area_triangle / area_rectangle
  probability

-- To state as a theorem using Lean's notation
theorem prob_x_lt_y_is_correct : prob_x_lt_y = 3 / 8 := sorry

end prob_x_lt_y_is_correct_l169_16971


namespace megan_markers_l169_16900

theorem megan_markers (initial_markers : ℕ) (new_markers : ℕ) (total_markers : ℕ) :
  initial_markers = 217 →
  new_markers = 109 →
  total_markers = 326 →
  initial_markers + new_markers = 326 :=
by
  sorry

end megan_markers_l169_16900


namespace value_of_f_prime_at_2_l169_16962

theorem value_of_f_prime_at_2 :
  ∃ (f' : ℝ → ℝ), 
  (∀ (x : ℝ), f' x = 2 * x + 3 * f' 2 + 1 / x) →
  f' 2 = - (9 / 4) := 
by 
  sorry

end value_of_f_prime_at_2_l169_16962


namespace quadratic_one_real_root_l169_16930

theorem quadratic_one_real_root (m : ℝ) : 
  (∃ x : ℝ, (x^2 - 6*m*x + 2*m = 0) ∧ 
    (∀ y : ℝ, (y^2 - 6*m*y + 2*m = 0) → y = x)) → 
  m = 2 / 9 :=
by
  sorry

end quadratic_one_real_root_l169_16930


namespace distinct_arrangements_l169_16906

-- Definitions based on the conditions
def boys : ℕ := 4
def girls : ℕ := 4
def total_people : ℕ := boys + girls
def arrangements : ℕ := Nat.factorial boys * Nat.factorial (total_people - 2) * Nat.factorial 6

-- Main statement: Verify the number of distinct arrangements
theorem distinct_arrangements : arrangements = 8640 := by
  -- We will replace this proof with our Lean steps (which is currently omitted)
  sorry

end distinct_arrangements_l169_16906


namespace range_of_a_l169_16959

noncomputable def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
noncomputable def setB : Set ℝ := {x | x < -1 ∨ x > 3}

theorem range_of_a (a : ℝ) :
  ((setA a ∩ setB) = setA a) ∧ (∃ x, x ∈ (setA a ∩ setB)) →
  (a < -3 ∨ a > 3) ∧ (a < -1 ∨ a > 1) :=
by sorry

end range_of_a_l169_16959


namespace find_principal_l169_16964

theorem find_principal
  (P : ℝ)
  (R : ℝ := 4)
  (T : ℝ := 5)
  (SI : ℝ := (P * R * T) / 100) 
  (h : SI = P - 2400) : 
  P = 3000 := 
sorry

end find_principal_l169_16964


namespace isosceles_triangle_l169_16948

theorem isosceles_triangle (a c : ℝ) (A C : ℝ) (h : a * Real.sin A = c * Real.sin C) : a = c → Isosceles :=
sorry

end isosceles_triangle_l169_16948


namespace disjunction_of_false_is_false_l169_16928

-- Given conditions
variables (p q : Prop)

-- We are given the assumption that both p and q are false propositions
axiom h1 : ¬ p
axiom h2 : ¬ q

-- We want to prove that the disjunction p ∨ q is false
theorem disjunction_of_false_is_false (p q : Prop) (h1 : ¬ p) (h2 : ¬ q) : ¬ (p ∨ q) := 
by
  sorry

end disjunction_of_false_is_false_l169_16928


namespace minimum_value_of_f_l169_16993

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 4

theorem minimum_value_of_f : ∃ x : ℝ, f x = -5 ∧ ∀ y : ℝ, f y ≥ -5 :=
by
  sorry

end minimum_value_of_f_l169_16993


namespace binom_floor_divisible_l169_16944

theorem binom_floor_divisible {p n : ℕ}
  (hp : Prime p) :
  (Nat.choose n p - n / p) % p = 0 := 
by
  sorry

end binom_floor_divisible_l169_16944


namespace relationship_of_y_coordinates_l169_16937

theorem relationship_of_y_coordinates (b y1 y2 y3 : ℝ):
  (y1 = 3 * -2.3 + b) → (y2 = 3 * -1.3 + b) → (y3 = 3 * 2.7 + b) → (y1 < y2 ∧ y2 < y3) := 
by 
  intros h1 h2 h3
  sorry

end relationship_of_y_coordinates_l169_16937


namespace trihedral_angle_sum_gt_180_l169_16956

theorem trihedral_angle_sum_gt_180
    (a' b' c' α β γ : ℝ)
    (Sabc : Prop)
    (h1 : b' = π - α)
    (h2 : c' = π - β)
    (h3 : a' = π - γ)
    (triangle_inequality : a' + b' + c' < 2 * π) :
    α + β + γ > π :=
by
  sorry

end trihedral_angle_sum_gt_180_l169_16956


namespace larger_number_is_70380_l169_16961

theorem larger_number_is_70380 (A B : ℕ) 
    (hcf : Nat.gcd A B = 20) 
    (lcm : Nat.lcm A B = 20 * 9 * 17 * 23) :
    max A B = 70380 :=
  sorry

end larger_number_is_70380_l169_16961


namespace a_n_formula_b_n_formula_l169_16996

namespace SequenceFormulas

theorem a_n_formula (n : ℕ) (h_pos : 0 < n) : 
  (∃ S : ℕ → ℕ, S n = 2 * n^2 + 2 * n) → ∃ a : ℕ → ℕ, a n = 4 * n :=
by
  sorry

theorem b_n_formula (n : ℕ) (h_pos : 0 < n) : 
  (∃ T : ℕ → ℕ, T n = 2 - (if n > 1 then T (n-1) else 1)) → ∃ b : ℕ → ℝ, b n = (1/2)^(n-1) :=
by
  sorry

end SequenceFormulas


end a_n_formula_b_n_formula_l169_16996


namespace smallest_N_l169_16929

theorem smallest_N (p q r s t u : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (ht : 0 < t) (hu : 0 < u)
  (h_sum : p + q + r + s + t + u = 2023) :
  ∃ N : ℕ, N = max (max (max (max (p + q) (q + r)) (r + s)) (s + t)) (t + u) ∧ N = 810 :=
sorry

end smallest_N_l169_16929


namespace direct_proportional_function_point_l169_16947

theorem direct_proportional_function_point 
    (h₁ : ∃ k : ℝ, ∀ x : ℝ, (2, -3).snd = k * (2, -3).fst)
    (h₂ : ∃ k : ℝ, ∀ x : ℝ, (4, -6).snd = k * (4, -6).fst)
    : (∃ k : ℝ, k = -(3 / 2)) :=
by
  sorry

end direct_proportional_function_point_l169_16947


namespace rectangle_diagonal_length_l169_16972

theorem rectangle_diagonal_length (P : ℝ) (L W D : ℝ) 
  (hP : P = 72) 
  (h_ratio : 3 * W = 2 * L) 
  (h_perimeter : 2 * (L + W) = P) :
  D = Real.sqrt (L * L + W * W) :=
sorry

end rectangle_diagonal_length_l169_16972


namespace cos_alpha_minus_half_beta_l169_16954

theorem cos_alpha_minus_half_beta
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (π / 4 + α) = 1 / 3)
  (h4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α - β / 2) = Real.sqrt 6 / 3 :=
by
  sorry

end cos_alpha_minus_half_beta_l169_16954


namespace even_of_even_square_sqrt_two_irrational_l169_16981

-- Problem 1: Prove that if \( p^2 \) is even, then \( p \) is even given \( p \in \mathbb{Z} \).
theorem even_of_even_square (p : ℤ) (h : Even (p * p)) : Even p := 
sorry 

-- Problem 2: Prove that \( \sqrt{2} \) is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ Int.gcd a b = 1 ∧ (a : ℝ) / b = Real.sqrt 2 :=
sorry

end even_of_even_square_sqrt_two_irrational_l169_16981


namespace compound_interest_l169_16933

theorem compound_interest (SI : ℝ) (P : ℝ) (R : ℝ) (T : ℝ) (CI : ℝ) :
  SI = 50 →
  R = 5 →
  T = 2 →
  P = (SI * 100) / (R * T) →
  CI = P * (1 + R / 100)^T - P →
  CI = 51.25 :=
by
  intros
  exact sorry -- This placeholder represents the proof that would need to be filled in 

end compound_interest_l169_16933


namespace mass_percentage_Br_HBrO3_l169_16936

theorem mass_percentage_Br_HBrO3 (molar_mass_H : ℝ) (molar_mass_Br : ℝ) (molar_mass_O : ℝ)
  (molar_mass_HBrO3 : ℝ) (mass_percentage_H : ℝ) (mass_percentage_Br : ℝ) :
  molar_mass_H = 1.01 →
  molar_mass_Br = 79.90 →
  molar_mass_O = 16.00 →
  molar_mass_HBrO3 = molar_mass_H + molar_mass_Br + 3 * molar_mass_O →
  mass_percentage_H = 0.78 →
  mass_percentage_Br = (molar_mass_Br / molar_mass_HBrO3) * 100 → 
  mass_percentage_Br = 61.98 :=
sorry

end mass_percentage_Br_HBrO3_l169_16936


namespace solution_set_of_inequality1_solution_set_of_inequality2_l169_16988

-- First inequality problem
theorem solution_set_of_inequality1 :
  {x : ℝ | x^2 + 3*x + 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
sorry

-- Second inequality problem
theorem solution_set_of_inequality2 :
  {x : ℝ | -3*x^2 + 2*x + 2 < 0} =
  {x : ℝ | x ∈ Set.Iio ((1 - Real.sqrt 7) / 3) ∪ Set.Ioi ((1 + Real.sqrt 7) / 3)} :=
sorry

end solution_set_of_inequality1_solution_set_of_inequality2_l169_16988


namespace sequence_expression_l169_16991

noncomputable def a_n (n : ℕ) : ℤ :=
if n = 1 then -1 else 1 - 2^n

def S_n (a_n : ℕ → ℤ) (n : ℕ) : ℤ :=
2 * a_n n + n

theorem sequence_expression :
  ∀ n : ℕ, n > 0 → (a_n n = 1 - 2^n) :=
by
  intro n hn
  sorry

end sequence_expression_l169_16991


namespace boat_travel_time_downstream_l169_16912

theorem boat_travel_time_downstream
  (v c: ℝ)
  (h1: c = 1)
  (h2: 24 / (v - c) = 6): 
  24 / (v + c) = 4 := 
by
  sorry

end boat_travel_time_downstream_l169_16912


namespace expand_and_simplify_l169_16931

theorem expand_and_simplify (y : ℚ) (h : y ≠ 0) :
  (3/4 * (8/y - 6*y^2 + 3*y)) = (6/y - 9*y^2/2 + 9*y/4) :=
by
  sorry

end expand_and_simplify_l169_16931


namespace least_number_to_add_l169_16974

theorem least_number_to_add (n : ℕ) (d : ℕ) (h1 : n = 907223) (h2 : d = 577) : (d - (n % d) = 518) := 
by
  rw [h1, h2]
  sorry

end least_number_to_add_l169_16974


namespace transform_quadratic_equation_l169_16913

theorem transform_quadratic_equation :
  ∀ x : ℝ, (x^2 - 8 * x - 1 = 0) → ((x - 4)^2 = 17) :=
by
  intro x
  intro h
  sorry

end transform_quadratic_equation_l169_16913


namespace root_equation_l169_16905

variable (m : ℝ)
theorem root_equation (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2020 = 2023 := by
  sorry

end root_equation_l169_16905


namespace neg_p_iff_a_in_0_1_l169_16943

theorem neg_p_iff_a_in_0_1 (a : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) ∧ (0 < a ∧ a < 1) :=
sorry

end neg_p_iff_a_in_0_1_l169_16943


namespace G_is_even_l169_16904

noncomputable def G (F : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := 
  F x * (1 / (a^x - 1) + 1 / 2)

theorem G_is_even (a : ℝ) (F : ℝ → ℝ) 
  (h₀ : a > 0) 
  (h₁ : a ≠ 1)
  (hF : ∀ x : ℝ, F (-x) = - F x) : 
  ∀ x : ℝ, G F a (-x) = G F a x :=
by 
  sorry

end G_is_even_l169_16904


namespace max_abs_sum_on_circle_l169_16994

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by sorry

end max_abs_sum_on_circle_l169_16994


namespace car_initial_time_l169_16983

variable (t : ℝ)

theorem car_initial_time (h : 80 = 720 / (3/2 * t)) : t = 6 :=
sorry

end car_initial_time_l169_16983


namespace find_y_if_x_l169_16945

theorem find_y_if_x (x : ℝ) (hx : x^2 + 8 * (x / (x - 3))^2 = 53) :
  (∃ y, y = (x - 3)^3 * (x + 4) / (2 * x - 5) ∧ y = 17000 / 21) :=
  sorry

end find_y_if_x_l169_16945


namespace min_value_S_max_value_m_l169_16990

noncomputable def S (x : ℝ) : ℝ := abs (x - 2) + abs (x - 4)

theorem min_value_S : ∃ x, S x = 2 ∧ ∀ x, S x ≥ 2 := by
  sorry

theorem max_value_m : ∀ x y, S x ≥ m * (-y^2 + 2*y) → 0 ≤ m ∧ m ≤ 2 := by
  sorry

end min_value_S_max_value_m_l169_16990


namespace deal_saves_customer_two_dollars_l169_16970

-- Define the conditions of the problem
def movie_ticket_price : ℕ := 8
def popcorn_price : ℕ := movie_ticket_price - 3
def drink_price : ℕ := popcorn_price + 1
def candy_price : ℕ := drink_price / 2

def normal_total_price : ℕ := movie_ticket_price + popcorn_price + drink_price + candy_price
def deal_price : ℕ := 20

-- Prove the savings
theorem deal_saves_customer_two_dollars : normal_total_price - deal_price = 2 :=
by
  -- We will fill in the proof here
  sorry

end deal_saves_customer_two_dollars_l169_16970


namespace math_problem_solution_l169_16910

open Real

noncomputable def math_problem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a + b + c + d = 4) : Prop :=
  (b / sqrt (a + 2 * c) + c / sqrt (b + 2 * d) + d / sqrt (c + 2 * a) + a / sqrt (d + 2 * b)) ≥ (4 * sqrt 3) / 3

theorem math_problem_solution (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 4) :
  math_problem a b c d ha hb hc hd h := by sorry

end math_problem_solution_l169_16910


namespace max_value_product_focal_distances_l169_16967

theorem max_value_product_focal_distances {a b c : ℝ} 
  (h1 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h2 : ∀ x : ℝ, -a ≤ x ∧ x ≤ a) 
  (e : ℝ) :
  (∀ x : ℝ, (a - e * x) * (a + e * x) ≤ a^2) :=
sorry

end max_value_product_focal_distances_l169_16967


namespace height_of_triangle_on_parabola_l169_16978

open Real

theorem height_of_triangle_on_parabola
  (x0 x1 : ℝ)
  (y0 y1 : ℝ)
  (hA : y0 = x0^2)
  (hB : y0 = (-x0)^2)
  (hC : y1 = x1^2)
  (hypotenuse_parallel : y0 = y1 + 1):
  y0 - y1 = 1 := 
by
  sorry

end height_of_triangle_on_parabola_l169_16978


namespace andy_diana_weight_l169_16909

theorem andy_diana_weight :
  ∀ (a b c d : ℝ),
  a + b = 300 →
  b + c = 280 →
  c + d = 310 →
  a + d = 330 := by
  intros a b c d h₁ h₂ h₃
  -- Proof goes here
  sorry

end andy_diana_weight_l169_16909


namespace f_2017_equals_neg_one_fourth_l169_16911

noncomputable def f : ℝ → ℝ := sorry -- Original definition will be derived from the conditions

axiom symmetry_about_y_axis : ∀ (x : ℝ), f (-x) = f x
axiom periodicity : ∀ (x : ℝ), f (x + 3) = -f x
axiom specific_interval : ∀ (x : ℝ), (3/2 < x ∧ x < 5/2) → f x = (1/2)^x

theorem f_2017_equals_neg_one_fourth : f 2017 = -1/4 :=
by sorry

end f_2017_equals_neg_one_fourth_l169_16911


namespace sequence_length_arithmetic_sequence_l169_16949

theorem sequence_length_arithmetic_sequence :
  ∃ n : ℕ, ∀ (a d : ℕ), a = 2 → d = 3 → a + (n - 1) * d = 2014 ∧ n = 671 :=
by {
  sorry
}

end sequence_length_arithmetic_sequence_l169_16949


namespace woman_traveled_by_bus_l169_16925

noncomputable def travel_by_bus : ℕ :=
  let total_distance := 1800
  let distance_by_plane := total_distance / 4
  let distance_by_train := total_distance / 6
  let distance_by_taxi := total_distance / 8
  let remaining_distance := total_distance - (distance_by_plane + distance_by_train + distance_by_taxi)
  let distance_by_rental := remaining_distance * 2 / 3
  distance_by_rental / 2

theorem woman_traveled_by_bus :
  travel_by_bus = 275 :=
by 
  sorry

end woman_traveled_by_bus_l169_16925


namespace part_1_part_2_l169_16938

noncomputable def f (x a : ℝ) : ℝ := x^2 * |x - a|

theorem part_1 (a : ℝ) (h : a = 2) : {x : ℝ | f x a = x} = {0, 1, 1 + Real.sqrt 2} :=
by 
  sorry

theorem part_2 (a : ℝ) : 
  ∃ m : ℝ, m = 
    if a ≤ 1 then 1 - a 
    else if 1 < a ∧ a ≤ 2 then 0 
    else if 2 < a ∧ a ≤ (7 / 3 : ℝ) then 4 * (a - 2) 
    else a - 1 :=
by 
  sorry

end part_1_part_2_l169_16938


namespace Gordons_heavier_bag_weight_l169_16963

theorem Gordons_heavier_bag_weight :
  ∀ (G : ℝ), (5 * 2 = 3 + G) → G = 7 :=
by
  intro G h
  sorry

end Gordons_heavier_bag_weight_l169_16963


namespace inequality_proof_l169_16903

theorem inequality_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3 / 4 :=
by {
  sorry
}

end inequality_proof_l169_16903


namespace geometric_sequence_divisible_l169_16984

theorem geometric_sequence_divisible (a1 a2 : ℝ) (h1 : a1 = 5 / 8) (h2 : a2 = 25) :
  ∃ n : ℕ, n = 7 ∧ (40^(n-1) * (5/8)) % 10^7 = 0 :=
by
  sorry

end geometric_sequence_divisible_l169_16984


namespace range_of_m_l169_16940

theorem range_of_m (x y m : ℝ) 
  (h1: 3 * x + y = 1 + 3 * m) 
  (h2: x + 3 * y = 1 - m) 
  (h3: x + y > 0) : 
  m > -1 :=
sorry

end range_of_m_l169_16940


namespace cell_phone_total_cost_l169_16995

def base_cost : ℕ := 25
def text_cost_per_message : ℕ := 3
def extra_minute_cost_per_minute : ℕ := 15
def included_hours : ℕ := 40
def messages_sent_in_february : ℕ := 200
def hours_talked_in_february : ℕ := 41

theorem cell_phone_total_cost :
  base_cost + (messages_sent_in_february * text_cost_per_message) / 100 + 
  ((hours_talked_in_february - included_hours) * 60 * extra_minute_cost_per_minute) / 100 = 40 :=
by
  sorry

end cell_phone_total_cost_l169_16995


namespace sum_ak_div_k2_ge_sum_inv_k_l169_16977

open BigOperators

theorem sum_ak_div_k2_ge_sum_inv_k
  (n : ℕ)
  (a : Fin n → ℕ)
  (hpos : ∀ k, 0 < a k)
  (hdist : Function.Injective a) :
  ∑ k : Fin n, (a k : ℝ) / (k + 1 : ℝ)^2 ≥ ∑ k : Fin n, 1 / (k + 1 : ℝ) := sorry

end sum_ak_div_k2_ge_sum_inv_k_l169_16977


namespace meera_fraction_4kmh_l169_16924

noncomputable def fraction_of_time_at_4kmh (total_time : ℝ) (x : ℝ) : ℝ :=
  x / total_time

theorem meera_fraction_4kmh (total_time x : ℝ) (h1 : x = total_time / 14) :
  fraction_of_time_at_4kmh total_time x = 1 / 14 :=
by
  sorry

end meera_fraction_4kmh_l169_16924


namespace find_m_l169_16939

def point (α : Type) := (α × α)

def collinear {α : Type} [LinearOrderedField α] 
  (p1 p2 p3 : point α) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

theorem find_m {m : ℚ} 
  (h : collinear (4, 10) (-3, m) (-12, 5)) : 
  m = 125 / 16 :=
by sorry

end find_m_l169_16939


namespace sales_tax_reduction_difference_l169_16965

def sales_tax_difference (original_rate new_rate market_price : ℝ) : ℝ :=
  (market_price * original_rate) - (market_price * new_rate)

theorem sales_tax_reduction_difference :
  sales_tax_difference 0.035 0.03333 10800 = 18.36 :=
by
  -- This is where the proof would go, but it is not required for this task.
  sorry

end sales_tax_reduction_difference_l169_16965


namespace probability_event_A_l169_16966

def probability_of_defective : Real := 0.3
def probability_of_all_defective : Real := 0.027
def probability_of_event_A : Real := 0.973

theorem probability_event_A :
  1 - probability_of_all_defective = probability_of_event_A :=
by
  sorry

end probability_event_A_l169_16966


namespace rides_total_l169_16951

theorem rides_total (rides_day1 rides_day2 : ℕ) (h1 : rides_day1 = 4) (h2 : rides_day2 = 3) : rides_day1 + rides_day2 = 7 := 
by 
  sorry

end rides_total_l169_16951


namespace vector_addition_subtraction_identity_l169_16902

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (BC AB AC : V)

theorem vector_addition_subtraction_identity : BC + AB - AC = 0 := 
by sorry

end vector_addition_subtraction_identity_l169_16902


namespace traceable_edges_l169_16973

-- Define the vertices of the rectangle
def vertex (x y : ℕ) : ℕ × ℕ := (x, y)

-- Define the edges of the rectangle
def edges : List (ℕ × ℕ) :=
  [vertex 0 0, vertex 0 1,    -- vertical edges
   vertex 1 0, vertex 1 1,
   vertex 2 0, vertex 2 1,
   vertex 0 0, vertex 1 0,    -- horizontal edges
   vertex 1 0, vertex 2 0,
   vertex 0 1, vertex 1 1,
   vertex 1 1, vertex 2 1]

-- Define the theorem to be proved
theorem traceable_edges :
  ∃ (count : ℕ), count = 61 :=
by
  sorry

end traceable_edges_l169_16973


namespace range_of_a_l169_16997

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def increasing_on_negative (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → y ≤ 0 → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (ha : even_function f) (hb : increasing_on_negative f) 
  (hc : ∀ a : ℝ, f a ≤ f (2 - a)) : ∀ a : ℝ, a < 1 → false :=
by
  sorry

end range_of_a_l169_16997


namespace initial_amount_in_cookie_jar_l169_16919

theorem initial_amount_in_cookie_jar (doris_spent : ℕ) (martha_spent : ℕ) (amount_left : ℕ) (spent_eq_martha : martha_spent = doris_spent / 2) (amount_left_eq : amount_left = 12) (doris_spent_eq : doris_spent = 6) : (doris_spent + martha_spent + amount_left = 21) :=
by
  sorry

end initial_amount_in_cookie_jar_l169_16919


namespace min_value_M_l169_16915

theorem min_value_M (a b : ℕ) (ha: 0 < a) (hb: 0 < b) : ∃ a b, M = 3 * a^2 - a * b^2 - 2 * b - 4 ∧ M = 2 := sorry

end min_value_M_l169_16915


namespace intersection_is_14_l169_16957

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem intersection_is_14 : A ∩ B = {1, 4} := 
by sorry

end intersection_is_14_l169_16957


namespace ratio_of_work_completed_by_a_l169_16998

theorem ratio_of_work_completed_by_a (A B W : ℝ) (ha : (A + B) * 6 = W) :
  (A * 3) / W = 1 / 2 :=
by 
  sorry

end ratio_of_work_completed_by_a_l169_16998


namespace sum_f_1_to_2017_l169_16932

noncomputable def f (x : ℝ) : ℝ :=
  if x % 6 < -1 then -(x % 6 + 2) ^ 2 else x % 6

theorem sum_f_1_to_2017 : (List.sum (List.map f (List.range' 1 2017))) = 337 :=
  sorry

end sum_f_1_to_2017_l169_16932


namespace total_time_spent_l169_16979

def one_round_time : ℕ := 30
def saturday_initial_rounds : ℕ := 1
def saturday_additional_rounds : ℕ := 10
def sunday_rounds : ℕ := 15

theorem total_time_spent :
  one_round_time * (saturday_initial_rounds + saturday_additional_rounds + sunday_rounds) = 780 := by
  sorry

end total_time_spent_l169_16979


namespace principal_amount_l169_16960

theorem principal_amount
  (P : ℝ)
  (r : ℝ := 0.05)
  (t : ℝ := 2)
  (H : P * (1 + r)^t - P - P * r * t = 17) :
  P = 6800 :=
by sorry

end principal_amount_l169_16960


namespace regular_polygon_sides_l169_16908

theorem regular_polygon_sides (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (angle_A angle_B angle_C : ℝ)
  (is_circle_inscribed_triangle : angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A ∧ angle_B + angle_C + angle_A = 180)
  (n : ℕ)
  (is_regular_polygon : B = C ∧ angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A) :
  n = 9 := sorry

end regular_polygon_sides_l169_16908


namespace waiter_earnings_l169_16969

def num_customers : ℕ := 9
def num_no_tip : ℕ := 5
def tip_per_customer : ℕ := 8
def num_tipping_customers := num_customers - num_no_tip

theorem waiter_earnings : num_tipping_customers * tip_per_customer = 32 := by
  sorry

end waiter_earnings_l169_16969


namespace neg_p_implies_neg_q_l169_16982

variable {x : ℝ}

def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

theorem neg_p_implies_neg_q (h : ¬ p x) : ¬ q x :=
sorry

end neg_p_implies_neg_q_l169_16982


namespace compute_problem_l169_16975

theorem compute_problem : (19^12 / 19^8)^2 = 130321 := by
  sorry

end compute_problem_l169_16975


namespace A_worked_days_l169_16955

theorem A_worked_days 
  (W : ℝ)                              -- Total work in arbitrary units
  (A_work_days : ℕ)                    -- Days A can complete the work 
  (B_work_days_remaining : ℕ)          -- Days B takes to complete remaining work
  (B_work_days : ℕ)                    -- Days B can complete the work alone
  (hA : A_work_days = 15)              -- A can do the work in 15 days
  (hB : B_work_days_remaining = 12)    -- B completes the remaining work in 12 days
  (hB_alone : B_work_days = 18)        -- B alone can do the work in 18 days
  :
  ∃ (x : ℕ), x = 5                     -- A worked for 5 days before leaving the job
  := 
  sorry                                 -- Proof not provided

end A_worked_days_l169_16955


namespace remainder_of_product_mod_10_l169_16950

theorem remainder_of_product_mod_10 :
  (1265 * 4233 * 254 * 1729) % 10 = 0 := by
  sorry

end remainder_of_product_mod_10_l169_16950


namespace min_value_of_a_l169_16927

theorem min_value_of_a (a : ℝ) (x : ℝ) (h1: 0 < a) (h2: a ≠ 1) (h3: 1 ≤ x → a^x ≥ a * x) : a ≥ Real.exp 1 :=
by
  sorry

end min_value_of_a_l169_16927


namespace number_of_regular_soda_bottles_l169_16953

-- Define the total number of bottles and the number of diet soda bottles
def total_bottles : ℕ := 30
def diet_soda_bottles : ℕ := 2

-- Define the number of regular soda bottles
def regular_soda_bottles : ℕ := total_bottles - diet_soda_bottles

-- Statement of the main proof problem
theorem number_of_regular_soda_bottles : regular_soda_bottles = 28 := by
  -- Proof goes here
  sorry

end number_of_regular_soda_bottles_l169_16953
