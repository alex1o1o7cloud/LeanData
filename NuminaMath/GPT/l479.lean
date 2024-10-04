import Mathlib

namespace find_AC_l479_479131

def angle_A := 60
def angle_B := 45
def side_BC := 12
def sin_60 := (Real.sin (Real.pi / 3)) -- 60 degrees in radians
def sin_45 := (Real.sin (Real.pi / 4)) -- 45 degrees in radians

theorem find_AC : 
  let AC := side_BC * sin_45 / sin_60 in
  AC = 4 * Real.sqrt 6 :=
sorry

end find_AC_l479_479131


namespace average_salary_l479_479220

theorem average_salary (total_workers technicians other_workers technicians_avg_salary other_workers_avg_salary total_salary : ℝ)
  (h_workers : total_workers = 21)
  (h_technicians : technicians = 7)
  (h_other_workers : other_workers = total_workers - technicians)
  (h_technicians_avg_salary : technicians_avg_salary = 12000)
  (h_other_workers_avg_salary : other_workers_avg_salary = 6000)
  (h_total_technicians_salary : total_salary = (technicians * technicians_avg_salary + other_workers * other_workers_avg_salary))
  (h_total_other_salary : total_salary = 168000) :
  total_salary / total_workers = 8000 := by
    sorry

end average_salary_l479_479220


namespace triangle_ratio_l479_479877

noncomputable def EG_GF_ratio (A B C M E F G : Point) (hM : midpoint M B C) 
  (hAB : dist A B = 15) (hAC : dist A C = 18) (hAE_AF : dist A E = 3 * dist A F)
  (hG_intersect : intersects G (line A M) (line E F)) : ℝ :=
(dist E G) / (dist G F)

theorem triangle_ratio (A B C M E F G : Point) 
  (hM : midpoint M B C) 
  (hAB : dist A B = 15) 
  (hAC : dist A C = 18) 
  (hAE_AF : dist A E = 3 * dist A F) 
  (hG_intersect : intersects G (line A M) (line E F)) : 
  EG_GF_ratio A B C M E F G hM hAB hAC hAE_AF hG_intersect = 2 := sorry

end triangle_ratio_l479_479877


namespace brown_mice_count_l479_479139

-- Definitions based on conditions
def total_mice (white_mice : ℕ) (fraction_white : ℚ) : ℕ :=
  (white_mice : ℚ) / fraction_white

def brown_mice (total_mice white_mice : ℕ) : ℕ :=
  total_mice - white_mice

-- Given conditions
def fraction_white : ℚ := 2 / 3
def white_mice : ℕ := 14

-- Problem statement to be proven
theorem brown_mice_count : brown_mice (total_mice white_mice fraction_white) white_mice = 7 :=
by
  sorry

end brown_mice_count_l479_479139


namespace domain_of_function_tan_l479_479956

def domain_of_tan (x : ℝ) : Prop := ∀ k : ℤ, x ≠ (π / 3) + (k * π / 2)

theorem domain_of_function_tan {x : ℝ} : domain_of_tan x ↔ 
  ∀ k : ℤ, x ≠ (π / 3) + (k * π / 2) :=
sorry

end domain_of_function_tan_l479_479956


namespace problem_inequality_l479_479175

theorem problem_inequality 
  (a b c : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) : 
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) :=
by
  sorry

end problem_inequality_l479_479175


namespace range_of_m_l479_479734

variable (x m : ℝ)

-- Definitions
def p : Prop := abs (x - 3) ≤ 2
def q : Prop := (x - m + 1) * (x - m - 1) ≤ 0
def not_p : Prop := ¬ p
def not_q : Prop := ¬ q

-- Sufficient but not necessary condition
def sufficient_but_not_necessary (A B : Prop) : Prop := (A → B) ∧ ¬ (B → A)

theorem range_of_m : sufficient_but_not_necessary (not_p x) (not_q x m) → 2 ≤ m ∧ m ≤ 4 :=
sorry

end range_of_m_l479_479734


namespace necessary_condition_not_sufficient_condition_l479_479066

noncomputable def f (x : ℝ) : ℝ := sorry -- Define the function f as needed
variable x₀ : ℝ

-- Define the condition: differentiability of f at x₀
def F_has_derivative : Prop := DifferentiableAt ℝ f x₀

-- Define the necessary condition: f'(x₀) = 0
def p : Prop := deriv f x₀ = 0

-- Define the local extremum condition
def q : Prop :=
  IsLocalMin f x₀ ∨ IsLocalMax f x₀

-- The theorem stating the necessary condition
theorem necessary_condition : q → p :=
begin
  sorry
end

-- The theorem stating that p is not sufficient for q
theorem not_sufficient_condition : ¬(p → q) :=
begin
  sorry
end

end necessary_condition_not_sufficient_condition_l479_479066


namespace no_reassembly_of_disks_l479_479934

theorem no_reassembly_of_disks
  (disks : list (set Point))
  (h_overlap : ∀ d1 d2 ∈ disks, d1 ≠ d2 → ∃ p, p ∈ d1 ∩ d2)
  (h_no_containment : ∀ d1 d2 ∈ disks, ¬(d1 ⊆ d2)) :
  ¬ ∃ configuration : list (set Point), configuration ≠ disks ∧
  (∀ c ∈ configuration, c.distinct ∧
   ∀ c1 c2 ∈ configuration, c1 ≠ c2 → ∃ p, p ∈ c1 ∩ c2) :=
sorry

end no_reassembly_of_disks_l479_479934


namespace volume_of_prism_l479_479219

-- Given dimensions a, b, and c, with the following conditions:
variables (a b c : ℝ)
axiom ab_eq_30 : a * b = 30
axiom ac_eq_40 : a * c = 40
axiom bc_eq_60 : b * c = 60

-- The volume of the prism is given by:
theorem volume_of_prism : a * b * c = 120 * Real.sqrt 5 :=
by
  sorry

end volume_of_prism_l479_479219


namespace decimal_irrational_l479_479369

theorem decimal_irrational (n : ℕ) (hn : 0 < n) :
  let f (x : ℕ) := x^n
  in irrational (decimal_concat (f 1) (f 2) (f 3) ...) :=
sorry

end decimal_irrational_l479_479369


namespace range_of_positive_integers_in_list_k_l479_479915

-- Define the list of consecutive integers
def consecutive_list (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

-- Define the problem conditions
def list_k : List ℤ := consecutive_list (-12) 25

-- Define a predicate to check if an integer is positive
def is_positive (x : ℤ) : Prop := x > 0

-- Get the positive integers from the list
def positive_integers_in_list (l : List ℤ) : List ℤ :=
  l.filter is_positive

-- Calculate the range of a list of integers
def range_of_list (l : List ℤ) : ℤ :=
  l.maximum' - l.minimum'

-- The main theorem
theorem range_of_positive_integers_in_list_k : range_of_list (positive_integers_in_list list_k) = 12 := by
  sorry

end range_of_positive_integers_in_list_k_l479_479915


namespace acorns_given_is_correct_l479_479186

-- Define initial conditions
def initial_acorns : ℕ := 16
def remaining_acorns : ℕ := 9

-- Define the number of acorns given to her sister
def acorns_given : ℕ := initial_acorns - remaining_acorns

-- Theorem statement
theorem acorns_given_is_correct : acorns_given = 7 := by
  sorry

end acorns_given_is_correct_l479_479186


namespace polynomials_with_rational_values_are_rational_coeffs_l479_479354

noncomputable def polynomial_with_complex_coeff_and_rational_result (P : Polynomial ℂ) : Prop :=
  ∀ q : ℚ, (P.eval q).im = 0

theorem polynomials_with_rational_values_are_rational_coeffs (P : Polynomial ℂ) :
  polynomial_with_complex_coeff_and_rational_result P → ∀ i, isRat (P.coeff i) :=
by
  sorry

end polynomials_with_rational_values_are_rational_coeffs_l479_479354


namespace number_of_five_digit_numbers_l479_479796

def count_five_identical_digits: Nat := 9
def count_two_different_digits: Nat := 1215
def count_three_different_digits: Nat := 6480
def count_four_different_digits: Nat := 22680
def count_five_different_digits: Nat := 27216

theorem number_of_five_digit_numbers :
  count_five_identical_digits + count_two_different_digits +
  count_three_different_digits + count_four_different_digits +
  count_five_different_digits = 57600 :=
by
  sorry

end number_of_five_digit_numbers_l479_479796


namespace train_passes_jogger_in_40_seconds_l479_479301

-- Define the given constants and conditions
def jogger_speed_kmph : ℝ := 9
def train_speed_kmph : ℝ := 45
def jogger_ahead_distance_m : ℝ := 280
def train_length_m : ℝ := 120

-- Convert speeds from kmph to m/s
def jogger_speed_mps : ℝ := jogger_speed_kmph * (1000 / 3600)
def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

-- Calculate the relative speed of the train with respect to the jogger
def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps

-- Calculate the total distance the train needs to cover to pass the jogger
def total_distance_m : ℝ := jogger_ahead_distance_m + train_length_m

-- Time taken for the train to pass the jogger
def time_to_pass_seconds : ℝ := total_distance_m / relative_speed_mps

-- The proof that the time to pass the jogger is 40 seconds
theorem train_passes_jogger_in_40_seconds :
  time_to_pass_seconds = 40 := by
    sorry

end train_passes_jogger_in_40_seconds_l479_479301


namespace number_of_points_marked_l479_479521

theorem number_of_points_marked (a₁ a₂ b₁ b₂ : ℕ) 
  (h₁ : a₁ * a₂ = 50) (h₂ : b₁ * b₂ = 56) (h₃ : a₁ + a₂ = b₁ + b₂) : 
  (a₁ + a₂ + 1 = 16) :=
sorry

end number_of_points_marked_l479_479521


namespace Q_2_over_Q_neg_2_l479_479967

noncomputable def g (x : ℝ) : ℝ := x^5 - 12 * x^4 + 5

lemma distinct_roots {s : ℕ → ℝ} (h_distinct : ∀ i j, i ≠ j → s i ≠ s j) :
  ∃ s_1 s_2 s_3 s_4 s_5, g(s_1) = 0 ∧ g(s_2) = 0 ∧ g(s_3) = 0 ∧ g(s_4) = 0 ∧ g(s_5) = 0 :=
sorry

lemma Q_property (Q : ℝ → ℝ) (k : ℝ) (hQ_deg : ∀ z, Q(z) = k * ∏ j in finset.range 5, (z - (s j ^ 2 + 1 / s j ^ 2))) :
  ∀ j, Q(s j ^ 2 + 1 / s j ^ 2) = 0 :=
sorry

theorem Q_2_over_Q_neg_2 (Q : ℝ → ℝ) (s : ℕ → ℝ) (k : ℝ) (hQ_deg : ∀ z, Q(z) = k * ∏ j in finset.range 5, (z - (s j ^ 2 + 1 / s j ^ 2)))
  (h_distinct : ∀ i j, i ≠ j → s i ≠ s j) : 
  (Q 2) / (Q (-2)) = 1 :=
sorry

end Q_2_over_Q_neg_2_l479_479967


namespace triangle_ABC_area_l479_479952

theorem triangle_ABC_area :
  let r := 3
  let BD := 4
  let AD := 10
  let ED := 6
  let area_ABC := (1 / 2) * ((sqrt 136 + 6) / 2) * sqrt (36 - ((sqrt 136 + 6) / 2)^2)
  in area_ABC = 120 / 31 :=
by
  sorry

end triangle_ABC_area_l479_479952


namespace max_volume_of_right_triangle_prism_l479_479856

-- Definitions and conditions
def right_triangle_prism_base_area (a b : ℝ) : ℝ := (1 / 2) * a * b
def right_triangle_prism_lateral_area (a b h : ℝ) : ℝ := a * h + b * h

-- Main theorem statement
theorem max_volume_of_right_triangle_prism
  (a b h : ℝ)
  (h_right_triangle : a > 0 ∧ b > 0 ∧ h > 0)
  (sum_adj_faces : right_triangle_prism_lateral_area a b h + right_triangle_prism_base_area a b = 30) :
  ∃ V, (V = (1 / 2) * a * b * h ∧ V ≤ 50) :=
begin
  sorry,
end

end max_volume_of_right_triangle_prism_l479_479856


namespace symmetric_point_lies_on_H1H2_l479_479424

open EuclideanGeometry

noncomputable def problem_statement (circle1 circle2: Circle) (P Q A B C D: Point) (H1 H2: Point) : Prop :=
  circles_intersect circle1 circle2 P Q ∧
  chord circle1 A C ∧
  chord circle2 B D ∧
  passes_through P A C ∧
  passes_through P B D ∧
  orthocenter P C D H1 ∧
  orthocenter P A B H2 ∧
  point_symmetric Q A C

theorem symmetric_point_lies_on_H1H2
    (circle1 circle2: Circle) (P Q A B C D: Point) (H1 H2: Point) :
    problem_statement circle1 circle2 P Q A B C D H1 H2 →
    lies_on (point_symmetric Q A C) (line H1 H2) :=
by
  intros h,
  sorry

end symmetric_point_lies_on_H1H2_l479_479424


namespace triangle_area_l479_479315

theorem triangle_area : 
  ∀ C : ℝ × ℝ, 
  C.1 + C.2 = 7 → 
  let A := (3 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 3 : ℝ)
  let area := 1 / 2 * real.sqrt ((3 - 0) ^ 2 + (0 - 3) ^ 2) * (4 / real.sqrt (1 ^ 2 + 1 ^ 2))
  area = 6 := 
by
  intros C hC
  let A := (3 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 3 : ℝ)
  let AB := real.sqrt ((3 - 0) ^ 2 + (0 - 3) ^ 2)
  let h := 4 / real.sqrt (1 ^ 2 + 1 ^ 2)
  let area := 1 / 2 * AB * h
  have hAB : AB = 3 * real.sqrt 2 := by sorry
  have hh : h = 2 * real.sqrt 2 := by sorry
  rw [hAB, hh]
  have h_area : area = 1 / 2 * (3 * real.sqrt 2) * (2 * real.sqrt 2) := by rfl
  rw h_area
  norm_num
  rfl
#align triangle_area

end triangle_area_l479_479315


namespace sin_double_angle_l479_479376

theorem sin_double_angle {α : ℝ} (h : sin α - cos α = 4 / 3) : sin (2 * α) = -7 / 9 :=
sorry

end sin_double_angle_l479_479376


namespace find_AC_l479_479134

noncomputable def isTriangle (A B C : Type) : Type := sorry

-- Define angles and lengths.
variables (A B C : Type)
variables (angle_A angle_B : ℝ)
variables (BC AC : ℝ)

-- Assume the given conditions.
axiom angle_A_60 : angle_A = 60 * Real.pi / 180
axiom angle_B_45 : angle_B = 45 * Real.pi / 180
axiom BC_12 : BC = 12

-- Statement to prove.
theorem find_AC 
  (h_triangle : isTriangle A B C)
  (h_angle_A : angle_A = 60 * Real.pi / 180)
  (h_angle_B : angle_B = 45 * Real.pi / 180)
  (h_BC : BC = 12) :
  ∃ AC : ℝ, AC = 8 * Real.sqrt 3 / 3 :=
sorry

end find_AC_l479_479134


namespace limit_of_difference_quotient_l479_479756

noncomputable def f : ℝ → ℝ := sorry

axiom f_deriv : ∀ x, has_deriv_at f (f' x) x
axiom f'_at_1 : f' 1 = 2

theorem limit_of_difference_quotient :
  (tendsto (λ Δx : ℝ, (f (1 - Δx) - f (1 + Δx)) / Δx) (𝓝 0) (𝓝 (-4))) :=
by
  sorry

end limit_of_difference_quotient_l479_479756


namespace minimum_points_to_form_isosceles_triangle_l479_479650

-- Define the conditions
variable (A B C : Point)
variable (side_length : ℝ)
variable (is_regular_triangle : IsRegularTriangle ABC)
variable (division_points : Finset Point)
variable (num_points : ℕ := 10)
variable (n : ℕ)

-- State the theorem
theorem minimum_points_to_form_isosceles_triangle 
  (h_side_length : side_length = 3)
  (h_num_divisions : division_points.card = num_points)
  (h_min_points : n ≥ 5) :
  ∃ s : Finset Point, s.card = n ∧ 
    (∀ (p q r : Point), p ∈ s → q ∈ s → r ∈ s → IsIsoscelesTriangle p q r) := sorry

end minimum_points_to_form_isosceles_triangle_l479_479650


namespace machine_A_production_rate_l479_479916

noncomputable def machine_A_sprockets_per_hour
  (P R Q : ℕ → ℝ) (time_Q : ℝ) :=
  let time_P := time_Q + 12
  let time_R := time_P - 8
  let sprockets := 990
  let production_rate_A := 990 / time_P
  let production_rate_Q := 1.12 * production_rate_A
  let production_rate_R := 1.08 * production_rate_A
  production_rate_A = 1.105

theorem machine_A_production_rate
  (time_Q : ℝ) (production_rate_A : ℝ)
  (production_rate_Q : ℝ) (production_rate_R : ℝ)
  (sprockets : ℝ) (time_P : ℝ)
  (time_R : ℝ) :
  (time_P  = time_Q + 12) →
  (time_R = time_P - 8) →
  (production_rate_Q = 1.12 * production_rate_A) →
  (production_rate_R = 1.08 * production_rate_A) →
  time_P * production_rate_A = sprockets →
  time_Q * production_rate_Q = sprockets →
  time_R * production_rate_R = sprockets →
  production_rate_A = 1.105 :=
begin
  sorry
end

end machine_A_production_rate_l479_479916


namespace find_b_for_square_binomial_l479_479702

theorem find_b_for_square_binomial 
  (b : ℝ)
  (u t : ℝ)
  (h₁ : u^2 = 4)
  (h₂ : 2 * t * u = 8)
  (h₃ : b = t^2) : b = 4 := 
  sorry

end find_b_for_square_binomial_l479_479702


namespace minimum_value_l479_479493

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y)

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  3 ≤ min_value_expr x y :=
  sorry

end minimum_value_l479_479493


namespace cost_of_pen_is_51_l479_479937

-- Definitions of variables and conditions
variables {p q : ℕ}
variables (h1 : 6 * p + 2 * q = 348)
variables (h2 : 3 * p + 4 * q = 234)

-- Goal: Prove the cost of a pen (p) is 51 cents
theorem cost_of_pen_is_51 : p = 51 :=
by
  -- placeholder for the proof
  sorry

end cost_of_pen_is_51_l479_479937


namespace integer_pairs_perfect_squares_l479_479686

theorem integer_pairs_perfect_squares (a b : ℤ) :
  (∃ k : ℤ, (a, b) = (k^2, 0) ∨ (a, b) = (0, k^2) ∨ (a, b) = (k, 1-k) ∨ (a, b) = (-6, -5) ∨ (a, b) = (-5, -6) ∨ (a, b) = (-4, -4))
  ↔ 
  (∃ x1 x2 : ℤ, a^2 + 4*b = x1^2 ∧ b^2 + 4*a = x2^2) :=
sorry

end integer_pairs_perfect_squares_l479_479686


namespace find_c_d_l479_479111

theorem find_c_d (y : ℝ) (c d : ℕ) (hy : y^2 + 4*y + 4/y + 1/y^2 = 35)
  (hform : ∃ (c d : ℕ), y = c + Real.sqrt d) : c + d = 42 :=
sorry

end find_c_d_l479_479111


namespace log_problem_l479_479631

theorem log_problem
  (x : ℝ)
  (h : log 2 (16 - 2^x) = x) :
  x = 3 :=
sorry

end log_problem_l479_479631


namespace find_value_l479_479634

theorem find_value (number remainder certain_value : ℕ) (h1 : number = 26)
  (h2 : certain_value / 2 = remainder) 
  (h3 : remainder = ((number + 20) * 2 / 2) - 2) :
  certain_value = 88 :=
by
  sorry

end find_value_l479_479634


namespace parametric_curve_length_6pi_l479_479692

noncomputable def parametric_curve_length (t : ℝ) : ℝ × ℝ :=
  (3 * Real.sin(t - (Real.pi / 4)), 3 * Real.cos(t - (Real.pi / 4)))

theorem parametric_curve_length_6pi :
  ∫ (t : ℝ) in 0..(2*Real.pi), Real.sqrt ((parametric_curve_length t).fst' * Real.sin(t - (Real.pi / 4)) + (parametric_curve_length t).snd' * Real.cos(t - (Real.pi / 4))) = 6 * Real.pi := 
by
  sorry

end parametric_curve_length_6pi_l479_479692


namespace smallest_positive_multiple_of_45_l479_479606

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ 45 * x = 45 :=
by {
  use 1,
  rw mul_one,
  exact nat.one_pos,
  sorry
}

end smallest_positive_multiple_of_45_l479_479606


namespace Jake_weight_loss_l479_479805

variables (J K x : ℕ)

theorem Jake_weight_loss : 
  J = 198 ∧ J + K = 293 ∧ J - x = 2 * K → x = 8 := 
by {
  sorry
}

end Jake_weight_loss_l479_479805


namespace triangle_angles_l479_479595

noncomputable def angle_triangle (E : ℝ) :=
if E = 45 then (90, 45, 45) else if E = 36 then (72, 72, 36) else (0, 0, 0)

theorem triangle_angles (E : ℝ) :
  (∃ E, E = 45 → angle_triangle E = (90, 45, 45))
  ∨
  (∃ E, E = 36 → angle_triangle E = (72, 72, 36)) :=
by
    sorry

end triangle_angles_l479_479595


namespace entry_cost_proof_l479_479472

variable (hitting_rate : ℕ → ℝ)
variable (entry_cost : ℝ)
variable (total_hits : ℕ)
variable (money_lost : ℝ)

-- Conditions
axiom hitting_rate_condition : hitting_rate 200 = 0.025
axiom total_hits_condition : total_hits = 300
axiom money_lost_condition : money_lost = 7.5

-- Question: Prove that the cost to enter the contest equals $10.00
theorem entry_cost_proof : entry_cost = 10 := by
  sorry

end entry_cost_proof_l479_479472


namespace composite_sum_of_ab_l479_479966

theorem composite_sum_of_ab (a b : ℕ) (h : 31 * a = 54 * b) : ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ a + b = k * l :=
sorry

end composite_sum_of_ab_l479_479966


namespace sin_cos_shift_l479_479259

noncomputable def cos_max := (0 : ℝ, 1 : ℝ)
noncomputable def sin_max := (Real.pi / 2, 1 : ℝ)

theorem sin_cos_shift :
  cos_max.1 + Real.pi / 2 = sin_max.1 ∧ cos_max.2 = sin_max.2 :=
by
  sorry

end sin_cos_shift_l479_479259


namespace proposition_p_true_l479_479776

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := 1/2 * x^2 + x + 1

theorem proposition_p_true : ∀ x ≥ 0, f x ≥ g x :=
by
  intros x hx
  let h := f x - g x
  have : ∀ x, h = Real.exp x - (1/2 * x^2 + x + 1) := by
    intro x
    calc
      f x - g x = Real.exp x - (1/2 * x^2 + x + 1) : by rfl
  sorry

end proposition_p_true_l479_479776


namespace scientific_notation_correct_l479_479861

theorem scientific_notation_correct :
  1200000000 = 1.2 * 10^9 := 
by
  sorry

end scientific_notation_correct_l479_479861


namespace total_houses_in_neighborhood_l479_479447

-- Definition of the function f
def f (x : ℕ) : ℕ := x^2 + 3*x

-- Given conditions
def x := 40

-- The theorem states that the total number of houses in Mariam's neighborhood is 1760.
theorem total_houses_in_neighborhood : (x + f x) = 1760 :=
by
  sorry

end total_houses_in_neighborhood_l479_479447


namespace circle_through_intersections_and_point_l479_479358

-- Definitions of given circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

-- Given point (1, -1)
def P1 : (ℝ × ℝ) := (1, -1)

-- Proof problem statement
theorem circle_through_intersections_and_point : 
  ∃ (λ : ℝ), 
    (∀ (x y : ℝ), (x^2 + y^2 + 4 * x - 4 * y + λ * (x^2 + y^2 - x) = 0) → 
     (C1 x y) ∧ (C2 x y)) ∧
    (let (a, b) := P1 in (a^2 + b^2 + 4 * a - 4 * b + λ * (a^2 + b^2 - a) = 0)) ∧ 
    (9 * x^2 + 9 * y^2 - 14 * x + 4 * y = 0) := sorry

end circle_through_intersections_and_point_l479_479358


namespace generalized_AM_GM_inequality_l479_479213

theorem generalized_AM_GM_inequality {n : ℕ} (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) :
  (∀ i, i < n → 0 < i) →
  (Finset.univ.sum a / n) ≥ ∏ i in Finset.univ, a i ^ (1 / n : ℝ) :=
by
  sorry

end generalized_AM_GM_inequality_l479_479213


namespace common_chord_and_angle_l479_479407

theorem common_chord_and_angle 
  (h1 : ∀ x y : ℝ, x^2 - 4 * x + y^2 - 2 * y = 8) 
  (h2 : ∀ x y : ℝ, x^2 - 6 * x + y^2 - 4 * y = -8) :
  (∀ x y : ℝ, x + y - 4 = 0) ∧ ((∃ θ : ℝ, θ = 45 ∧ tan θ = 1)) :=
by
  sorry

end common_chord_and_angle_l479_479407


namespace rory_more_jellybeans_l479_479931

-- Definitions based on the conditions
def G : ℕ := 15 -- Gigi has 15 jellybeans
def LorelaiConsumed (R G : ℕ) : ℕ := 3 * (R + G) -- Lorelai has already eaten three times the total number of jellybeans

theorem rory_more_jellybeans {R : ℕ} (h1 : LorelaiConsumed R G = 180) : (R - G) = 30 :=
  by
    -- we can skip the proof here with sorry, as we are only interested in the statement for now
    sorry

end rory_more_jellybeans_l479_479931


namespace area_AIHFGD_eq_25_l479_479151

-- Definitions of squares and points
def is_square (ABCD : set (ℝ × ℝ)) (A B C D : ℝ × ℝ) : Prop :=
  ∃ (s : ℝ), s = 5 ∧
    -- Assuming A, B, C, D are in counterclockwise order
    A = (0, 0) ∧
    B = (s, 0) ∧
    C = (s, s) ∧
    D = (0, s)

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Problem statement
theorem area_AIHFGD_eq_25 (sq1 sq2 : set (ℝ × ℝ)) (A B C D E F G H I : ℝ × ℝ) :
  is_square sq1 A B C D →
  is_square sq2 E F G D →
  midpoint B C = H →
  midpoint E F = H →
  midpoint A B = I →
  area_of_polygon [A, I, H, F, G, D] = 25 :=
sorry

end area_AIHFGD_eq_25_l479_479151


namespace conical_well_volume_l479_479639

noncomputable def volume_conical_well (diameter_top diameter_base height : ℝ) : ℝ :=
  let radius_top := diameter_top / 2
  let radius_base := diameter_base / 2
  let radius_avg := (radius_top + radius_base) / 2
  (1 / 3) * Real.pi * radius_avg^2 * height

theorem conical_well_volume :
  volume_conical_well 2 1 8 ≈ 4.71239 :=
by
  unfold volume_conical_well
  simp only [div_mul_eq_mul_div, div_pow]
  have approx_pi : Real.pi ≈ 3.14159 := sorry
  have calc_volume : (1 / 3) * 3.14159 * (0.75^2) * 8 ≈ 4.71239 := sorry
  rw [approx_pi]
  exact calc_volume

end conical_well_volume_l479_479639


namespace largest_integer_product_l479_479716

-- Define the conditions
def valid_integer (n : ℕ) : Prop :=
  let digits := (n.digits 10) in
  digits ≠ [] ∧
  (∀ i < digits.length - 1, digits.nth i < digits.nth (i + 1)) ∧
  digits.foldl (λ acc d, acc + d^2) 0 = 85

-- Define the problem to prove the product of digits
def product_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (λ acc d, acc * d) 1

-- Proposed main theorem statement
theorem largest_integer_product : ∃ n : ℕ, valid_integer n ∧ product_of_digits n = 64 :=
  sorry

end largest_integer_product_l479_479716


namespace tunnel_length_l479_479313

theorem tunnel_length (x : ℕ) (y : ℕ) 
  (h1 : 300 + x = 60 * y) 
  (h2 : x - 300 = 30 * y) : 
  x = 900 := 
by
  sorry

end tunnel_length_l479_479313


namespace cellphone_loading_time_approximately_l479_479157

noncomputable def cellphone_loading_time_minutes : ℝ :=
  let T := 533.78 -- Solution for T from solving the given equation
  T / 60

theorem cellphone_loading_time_approximately :
  abs (cellphone_loading_time_minutes - 8.90) < 0.01 :=
by 
  -- The proof goes here, but we are just required to state it
  sorry

end cellphone_loading_time_approximately_l479_479157


namespace price_per_bottle_is_half_l479_479204

theorem price_per_bottle_is_half (P : ℚ) 
  (Remy_bottles_morning : ℕ) (Nick_bottles_morning : ℕ) 
  (Total_sales_evening : ℚ) (Evening_more : ℚ) : 
  Remy_bottles_morning = 55 → 
  Nick_bottles_morning = Remy_bottles_morning - 6 → 
  Total_sales_evening = 55 → 
  Evening_more = 3 → 
  104 * P + 3 = 55 → 
  P = 1 / 2 := 
by
  intros h_remy_55 h_nick_remy h_total_55 h_evening_3 h_sales_eq
  sorry

end price_per_bottle_is_half_l479_479204


namespace zeros_of_y_in_interval_l479_479757

noncomputable def f (x : ℝ) : ℝ := 
  if h : 1 ≤ x ∧ x < 2 then 1 - |2 * x - 3| 
  else if 2 ≤ x then 1 / 2 * f (1 / 2 * x) 
  else 0 -- this covers the definition for all x in [1, +∞)

def y (x : ℝ) : ℝ := 2 * x * f x - 3

theorem zeros_of_y_in_interval :
  ∃ s : set ℝ, s = {x | 1 < x ∧ x < 2017 ∧ y x = 0} ∧ s.card = 11 :=
sorry

end zeros_of_y_in_interval_l479_479757


namespace P_plus_Q_l479_479108

theorem P_plus_Q (P Q : ℕ) (h1 : 4 / 7 = P / 49) (h2 : 4 / 7 = 84 / Q) : P + Q = 175 :=
by
  sorry

end P_plus_Q_l479_479108


namespace probability_of_red_ball_on_fourth_draw_with_replacement_probability_of_red_ball_on_fourth_draw_without_replacement_l479_479444

-- Definition of the probabilities
def probability_with_replacement : ℚ :=
  let red_probability := 6 / 10
  let white_probability := 4 / 10
  (
    white_probability^3 * red_probability
  )

def probability_without_replacement : ℚ :=
  (4 / 10) * (3 / 9) * (2 / 8) * (6 / 7)

-- Theorem statements based on the conditions and answers
theorem probability_of_red_ball_on_fourth_draw_with_replacement :
    probability_with_replacement = 24 / 625 :=
begin
  sorry
end

theorem probability_of_red_ball_on_fourth_draw_without_replacement :
    probability_without_replacement = 1 / 70 :=
begin
  sorry
end

end probability_of_red_ball_on_fourth_draw_with_replacement_probability_of_red_ball_on_fourth_draw_without_replacement_l479_479444


namespace gain_percent_l479_479278

theorem gain_percent (CP SP : ℝ) (hCP : CP = 840) (hSP : SP = 1220) : 
  (SP - CP) / CP * 100 ≈ 45.24 :=
by
  sorry

end gain_percent_l479_479278


namespace largest_M_value_l479_479176

noncomputable theory

variables (x y z : ℝ)

def satisfies_conditions (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x + y + z = 20 ∧
  xy + yz + zx = 78

def M (x y z : ℝ) : ℝ := min (x * y) (min (y * z) (z * x))

theorem largest_M_value :
  ∃ (x y z : ℝ), satisfies_conditions x y z ∧ M x y z = 400 / 9 :=
by {
  sorry,
}

end largest_M_value_l479_479176


namespace sum_x_coords_above_line_l479_479191

theorem sum_x_coords_above_line :
  let points := [(4, 15), (6, 25), (12, 40), (18, 45), (21, 60), (25, 70)]
  let line := fun x => 3 * x + 5
  (points.filter (fun p => p.2 > line p.1)).map (fun p => p.1) = [6] :=
by
  -- points
  have points := [(4, 15), (6, 25), (12, 40), (18, 45), (21, 60), (25, 70)]
  -- line equation
  let line := fun x => 3 * x + 5
  -- check each point
  have check1 : (4, 15).2 <= line (4, 15).1 := by norm_num
  have check2 : (6, 25).2 > line (6, 25).1 := by norm_num
  have check3 : (12, 40).2 <= line (12, 40).1 := by norm_num
  have check4 : (18, 45).2 <= line (18, 45).1 := by norm_num
  have check5 : (21, 60).2 <= line (21, 60).1 := by norm_num
  have check6 : (25, 70).2 <= line (25, 70).1 := by norm_num
  -- filter and map
  have filtered := points.filter (fun p => p.2 > line p.1)
  have mapped := filtered.map (fun p => p.1)
  -- result
  exact mapped = [6]


end sum_x_coords_above_line_l479_479191


namespace find_derivative_at_one_l479_479811

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := x^3 - c * x^2 + 3

theorem find_derivative_at_one (c : ℝ) (h : ∀ x : ℝ, Derivative (f c) x = 3 * x^2 - 2 * f (c) 1 * x)
  : f c 1 = 1 := sorry

end find_derivative_at_one_l479_479811


namespace marbles_difference_l479_479886

def lostMarbles : ℕ := 8
def foundMarbles : ℕ := 10

theorem marbles_difference (lostMarbles foundMarbles : ℕ) : foundMarbles - lostMarbles = 2 := 
by
  sorry

end marbles_difference_l479_479886


namespace greatest_number_of_sets_l479_479503

-- We define the number of logic and visual puzzles.
def n_logic : ℕ := 18
def n_visual : ℕ := 9

-- The theorem states that the greatest number of identical sets Mrs. Wilson can create is the GCD of 18 and 9.
theorem greatest_number_of_sets : gcd n_logic n_visual = 9 := by
  sorry

end greatest_number_of_sets_l479_479503


namespace shaded_area_l479_479867

theorem shaded_area (r : ℝ) (h : r = 5) 
  (perp_diam : true) 
  (quarter_circle_arcs : true) : 
  let tau := 2 * Real.pi in
  let circle_area := Real.pi * r^2 in
  let triangle_area := (1/2) * r * r in
  let sector_area := (1/4) * circle_area in 
  2 * triangle_area + 2 * sector_area = 25 + 12.5 * Real.pi := 
by
  have triangle_total : 2 * triangle_area = 25 := by 
    sorry
  have sector_total : 2 * sector_area = 12.5 * Real.pi := by 
    sorry
  calc
    2 * triangle_area + 2 * sector_area
        = 25 + 12.5 * Real.pi by
    rw [triangle_total, sector_total]

end shaded_area_l479_479867


namespace pairwise_disjoint_arith_seqs_l479_479735

theorem pairwise_disjoint_arith_seqs:
  ∃ (a : Fin 100 → ℤ), 
    ∀ (i j : Fin 100), 
      i ≠ j → 
        (∀ k1 k2 : ℕ, a i + k1 * (n i) ≠ a j + k2 * (n j)) :=
sorry

end pairwise_disjoint_arith_seqs_l479_479735


namespace speed_of_slower_train_is_36_l479_479996

-- Definitions used in the conditions
def length_of_train := 25 -- meters
def combined_length_of_trains := 2 * length_of_train -- meters
def time_to_pass := 18 -- seconds
def speed_of_faster_train := 46 -- km/hr
def conversion_factor := 1000 / 3600 -- to convert from km/hr to m/s

-- Prove that speed of the slower train is 36 km/hr
theorem speed_of_slower_train_is_36 :
  ∃ v : ℕ, v = 36 ∧ ((combined_length_of_trains : ℝ) = ((speed_of_faster_train - v) * conversion_factor * time_to_pass)) :=
sorry

end speed_of_slower_train_is_36_l479_479996


namespace find_n_divisors_l479_479705

open BigOperators

def divisor_count (n : ℕ) : ℕ :=
  number_of_divisors n -- Assume a function that provides the number of divisors of a number

def cube_root (x : ℕ) : ℕ :=
  -- Assume a function that computes the integer cube root of x
  nat_root 3 x

theorem find_n_divisors:
  ∀ n : ℕ, divisor_count n = cube_root (4 * n) ↔ n = 2 ∨ n = 128 ∨ n = 4000 :=
by
  sorry

end find_n_divisors_l479_479705


namespace num_children_l479_479253

-- Definitions of the conditions as per a)
def sum_of_ages (n : ℕ) : ℕ := n / 2 * (8 + 3 * (n - 1))  -- Using n * (4 + 4 + (n-1)*3)

-- The problem to prove is that the number of children is 5 given the conditions
theorem num_children : ∃ (n : ℕ), sum_of_ages n = 50 ∧ n = 5 :=
by
  use 5
  unfold sum_of_ages
  norm_num
  apply and.intro
  { norm_num }
  { rfl }

end num_children_l479_479253


namespace coefficient_of_x3_term_l479_479408

theorem coefficient_of_x3_term (n : ℕ) (h₁ : (x - (1 / x^(1/2)))^n = 512) :
  (nat.choose 9 4) = 126 := 
sorry

end coefficient_of_x3_term_l479_479408


namespace clock_angle_230_l479_479600

theorem clock_angle_230 (h12 : ℕ := 12) (deg360 : ℕ := 360) 
  (hour_mark_deg : ℕ := deg360 / h12) (hour_halfway : ℕ := hour_mark_deg / 2)
  (hour_deg_230 : ℕ := hour_mark_deg * 3) (total_angle : ℕ := hour_halfway + hour_deg_230) :
  total_angle = 105 := 
by
  sorry

end clock_angle_230_l479_479600


namespace find_f_expression_range_of_m_l479_479248

-- Given conditions for the function f
variables {R : Type*} [ordered_ring R]
variables (f : R → R) (x : R)

-- Definition of the function based on conditions
def f_satisfies_conditions : Prop :=
  (∀ x, f(x + 1) - f(x) = 2 * x) ∧ (f 0 = 1)

-- The first part: expression for f(x)
theorem find_f_expression (hf : f_satisfies_conditions f) : f = λ x, x^2 - x + 1 :=
sorry

-- The second part: determining range of m
def always_above (f g : R → R) (a b : R) : Prop :=
  ∀ x, a ≤ x → x ≤ b → f x > g x

theorem range_of_m (f : R → R) (m : R) (hf : f = λ x, x^2 - x + 1) :
  always_above f (λ x, 2 * x + m) (-1) 1 ↔ m < -1 :=
sorry

end find_f_expression_range_of_m_l479_479248


namespace probability_at_least_one_multiple_of_4_l479_479331

-- Define the condition
def random_integer_between_1_and_60 : set ℤ := {n : ℤ | 1 ≤ n ∧ n ≤ 60}

-- Define the probability theorems and the proof for probability calculation
theorem probability_at_least_one_multiple_of_4 :
  (∀ (n1 n2 : ℤ), (n1 ∈ random_integer_between_1_and_60) ∧ (n2 ∈ random_integer_between_1_and_60) → 
  (∃ k, n1 = 4 * k ∨ ∃ k, n2 = 4 * k)) ∧ 
  (1 / 60 * 1 / 60) * (15 / 60) ^ 2 = 7 / 16 :=
by
  sorry

end probability_at_least_one_multiple_of_4_l479_479331


namespace count_zeros_fraction_l479_479794

theorem count_zeros_fraction : 
  ∀ (a b c d : ℕ), (a = 3) → (b = 2^7) → (d = 5^10) → ((a : ℚ) / (b * d) = c / 10^10) → (c = 24) → 
  ∃ n : ℕ, n = 8 ∧ 10^n * (c / 10^10) = c :=
by
  intros a b c d ha hb hd hfrac hc
  use 8
  split
  { exact rfl }
  { calc 10^8 * (c / 10^10) = (10^8 * c) / 10^10 : by field_simp
                        ... = 24 / 10^2 : by rw [hc, pow_add, pow_add, mul_assoc, mul_comm 10 24, mul_comm 10 100000000, mul_div_mul_left, div_eq_div_iff]
                        ... = 24 / 100 },
  sorry

end count_zeros_fraction_l479_479794


namespace ratio_of_cream_l479_479471

def initial_coffee := 12
def joe_drank := 2
def cream_added := 2
def joann_cream_added := 2
def joann_drank := 2

noncomputable def joe_coffee_after_drink_add := initial_coffee - joe_drank + cream_added
noncomputable def joe_cream := cream_added

noncomputable def joann_initial_mixture := initial_coffee + joann_cream_added
noncomputable def joann_portion_before_drink := joann_cream_added / joann_initial_mixture
noncomputable def joann_remaining_coffee := joann_initial_mixture - joann_drank
noncomputable def joann_cream_after_drink := joann_portion_before_drink * joann_remaining_coffee
noncomputable def joann_cream := joann_cream_after_drink

theorem ratio_of_cream : joe_cream / joann_cream = 7 / 6 :=
by sorry

end ratio_of_cream_l479_479471


namespace contrapositive_of_proposition_is_false_l479_479225

theorem contrapositive_of_proposition_is_false (x y : ℝ) 
  (h₀ : (x + y > 0) → (x > 0 ∧ y > 0)) : 
  ¬ ((x ≤ 0 ∨ y ≤ 0) → (x + y ≤ 0)) :=
by
  sorry

end contrapositive_of_proposition_is_false_l479_479225


namespace most_stable_machine_l479_479986

noncomputable def var_A : ℝ := 10.3
noncomputable def var_B : ℝ := 6.9
noncomputable def var_C : ℝ := 3.5

theorem most_stable_machine :
  (var_C < var_B) ∧ (var_C < var_A) :=
by
  sorry

end most_stable_machine_l479_479986


namespace hyperbola_equation_l479_479777

theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : c = Real.sqrt 5) (h4 : b / a = 1 / 2) : 
  (x : ℝ) (y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → 
  (x^2 / 4 - y^2 = 1) :=
by
  sorry

end hyperbola_equation_l479_479777


namespace number_of_obtuse_triangles_count_obtuse_k_values_l479_479567

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a^2 + b^2 < c^2)

def valid_k_values (lower upper k : ℕ) : ℕ :=
  (upper - lower) + 1

theorem number_of_obtuse_triangles (k : ℕ) (h1 : k > 4 ∧ k ≤ 10 ∨ 21 ≤ k ∧ k < 28): 
  (5 ≤ k ∧ k ≤ 10 ∧ is_obtuse_triangle 12 k 16) ∨ (21 ≤ k ∧ k < 28 ∧ is_obtuse_triangle 12 16 k) :=
  sorry
     
theorem count_obtuse_k_values : ℕ := 
  valid_k_values 5 10 + valid_k_values 21 27 = 13

end number_of_obtuse_triangles_count_obtuse_k_values_l479_479567


namespace projected_area_sum_l479_479855

theorem projected_area_sum
  (ABC A1B1C1 : Triangle)
  (AB AC BC A1B1 A1C1 B1C1 : LineSegment)
  (D E : Point)
  (area_ABC : Real)
  (area_A1B1C1 : Real)
  (H_parallel : Parallel DE BC)
  (H_D_on_AB : On D AB)
  (H_E_on_AC : On E AC) :
  area_ABC = √3 →
  area_A1B1C1 = 4 * √3 →
  sum_of_projected_areas A1DE B1C1ED A1B1C1 = (8 * √3) / 3 :=
by
  sorry

end projected_area_sum_l479_479855


namespace solution_set_f_x_lt_0_l479_479755

variable (f : ℝ → ℝ)

-- Define the conditions
def condition1 := ∀ x, f' x + 2 * f x > 0
def condition2 := f (-1) = 0

-- The theorem we need to prove
theorem solution_set_f_x_lt_0 (h1 : condition1 f) (h2 : condition2 f) :
    { x : ℝ | f x < 0 } = set.Iio (-1) :=
sorry

end solution_set_f_x_lt_0_l479_479755


namespace grid_satisfies_conditions_l479_479840

-- Define the range of consecutive integers
def consecutive_integers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Define a function to check mutual co-primality condition for two given numbers
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the 3x3 grid arrangement as a matrix
@[reducible]
def grid : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![8, 9, 10],
    ![5, 7, 11],
    ![6, 13, 12]]

-- Define a predicate to check if the grid cells are mutually coprime for adjacent elements
def mutually_coprime_adjacent (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
    ∀ (i j k l : Fin 3), 
    (|i - k| + |j - l| = 1 ∨ |i - k| = |j - l| ∧ |i - k| = 1) → coprime (m i j) (m k l)

-- The main theorem statement
theorem grid_satisfies_conditions : 
  (∀ (i j : Fin 3), grid i j ∈ consecutive_integers) ∧ 
  mutually_coprime_adjacent grid :=
by
  sorry

end grid_satisfies_conditions_l479_479840


namespace unique_solution_for_series_l479_479020

theorem unique_solution_for_series (a m n : ℕ) (h₀: 0 < a) (h₁: 0 < m) (h₂: 0 < n)
  (h₃: (a + 1) * (nat.geometricSeries a 2 3) * ... * (nat.geometricSeries a n (n + 1)) = nat.geometricSeries a m (m + 1)) :
  (a = 1 ∧ m = (nat.fact (n+1) - 1) ∧ n > 0) :=
by
  sorry

end unique_solution_for_series_l479_479020


namespace find_m_if_perpendicular_l479_479402

theorem find_m_if_perpendicular 
  (m : ℝ)
  (h : ∀ m (slope1 : ℝ) (slope2 : ℝ), 
    (slope1 = -m) → 
    (slope2 = (-1) / (3 - 2 * m)) → 
    slope1 * slope2 = -1)
  : m = 3 := 
by
  sorry

end find_m_if_perpendicular_l479_479402


namespace odd_function_condition_l479_479005

noncomputable def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) ↔ (a = 0 ∧ b = 0) :=
by
  sorry

end odd_function_condition_l479_479005


namespace statement1_statement2_statement3_number_of_correct_statements_is_one_l479_479093

-- Definitions based on the conditions
def lines_parallel_to_planes (m n: Line) (α β: Plane) :=
  m ≠ n ∧
  ∀ l: Line, l ∈ {m, n} →
  (l ∩ α = ∅ ∧ l ∩ β = ∅ ∧ l ∥ α ∧ l ∥ β)

def condition1 (m n: Line) (α β: Plane) :=
  (m ≠ n) ∧ (intersect m n) ∧ (∀ l∈ {m, n}, (l ∩ α = ∅) ∧ (l ∩ β = ∅) ∧ (l ∥ α) ∧ (l ∥ β))

def condition2 (m: Line) (α β: Plane) :=
  m ∥ α ∧ m ∥ β

def condition3 (m n: Line) (α β: Plane) :=
  (m ∥ α) ∧ (n ∥ β) ∧ (m ∥ n)

-- Theorems to validate each condition.
theorem statement1 {m n: Line} {α β: Plane} (h: condition1 m n α β) : α ∥ β :=
sorry

theorem statement2 {m: Line} {α β: Plane} (h: condition2 m α β) : ¬ (α ∥ β) :=
sorry

theorem statement3 {m n: Line} {α β: Plane} (h: condition3 m n α β) : ¬ (α ∥ β) :=
sorry

-- Final theorem stating that only statement 1 is correct
theorem number_of_correct_statements_is_one (m n: Line) (α β: Plane) :
  (statement1 h1 as true) ∧ (statement2 h2 as false) ∧ (statement3 h3 as false) → 1 :=
sorry

end statement1_statement2_statement3_number_of_correct_statements_is_one_l479_479093


namespace coefficient_x3_product_l479_479024

noncomputable def P (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + x^2 - 3
noncomputable def Q (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 4

theorem coefficient_x3_product :
  (∃ c : ℝ, c = 13 ∧ ∀ (x : ℝ), P(x) * Q(x) = c * x^3 + ...) sorry

end coefficient_x3_product_l479_479024


namespace find_missing_ratio_l479_479949

def compounded_ratio (x y : ℚ) : ℚ := (x / y) * (6 / 11) * (11 / 2)

theorem find_missing_ratio (x y : ℚ) (h : compounded_ratio x y = 2) :
  x / y = 2 / 3 :=
sorry

end find_missing_ratio_l479_479949


namespace snow_probability_at_least_once_l479_479969

def probability_day1 : Real := 3 / 4
def probability_day4 : Real := 1 / 2

theorem snow_probability_at_least_once :
  let decrease := (probability_day1 - probability_day4) / 3
  let probability_day2 := probability_day1 - decrease
  let probability_day3 := probability_day2 - decrease
  let p_not_snow_day1 := 1 - probability_day1
  let p_not_snow_day2 := 1 - probability_day2
  let p_not_snow_day3 := 1 - probability_day3
  let p_not_snow_day4 := 1 - probability_day4
  let p_not_snow_all_days := p_not_snow_day1 * p_not_snow_day2 * p_not_snow_day3 * p_not_snow_day4
  let p_snow_at_least_once := 1 - p_not_snow_all_days
  p_snow_at_least_once = 283 / 288 := by sorry

end snow_probability_at_least_once_l479_479969


namespace min_value_hyperbola_l479_479722

theorem min_value_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ e : ℝ, e = 2 ∧ (b^2 = (e * a)^2 - a^2)) :
  (a * 3 + 1 / a) = 2 * Real.sqrt 3 :=
by
  sorry

end min_value_hyperbola_l479_479722


namespace convert_base8_to_base10_l479_479684

def base8_to_base10 (n : Nat) : Nat := 
  -- Assuming a specific function that converts from base 8 to base 10
  sorry 

theorem convert_base8_to_base10 :
  base8_to_base10 5624 = 2964 :=
by
  sorry

end convert_base8_to_base10_l479_479684


namespace min_distance_from_M_to_C_l479_479873

-- Definitions based on conditions from the problem
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  ⟨ρ * Real.cos θ, ρ * Real.sin θ⟩

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Given conditions
def M := polar_to_cartesian (4 * Real.sqrt 2) (Real.pi / 4)
def A := (1, 0)
def r := Real.sqrt 2
def C (α : ℝ) : ℝ × ℝ :=
  (1 + Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α)

-- The theorem we want to prove
theorem min_distance_from_M_to_C : 
  ∃ m ∈ Set.range C, distance M m = 5 - Real.sqrt 2 := 
sorry

end min_distance_from_M_to_C_l479_479873


namespace sum_of_coordinates_B_l479_479198

theorem sum_of_coordinates_B
  (x y : ℤ)
  (Mx My : ℤ)
  (Ax Ay : ℤ)
  (M : Mx = 2 ∧ My = -3)
  (A : Ax = -4 ∧ Ay = -5)
  (midpoint_x : (x + Ax) / 2 = Mx)
  (midpoint_y : (y + Ay) / 2 = My) :
  x + y = 7 :=
by
  sorry

end sum_of_coordinates_B_l479_479198


namespace matrix_invertibility_and_fraction_sum_l479_479390

theorem matrix_invertibility_and_fraction_sum
  (a b c : ℝ)
  (h1 : a^3 + b^3 + c^3 = 3 * a * b * c)
  (h2 : det (Matrix.of ![
    ![a, b, c],
    ![c, a, b],
    ![b, c, a]]) = 0) :
  ∃ (x : ℝ), (x = (a / (b + c) + b / (c + a) + c / (a + b))) ∧ (x = 3 / 2) :=
by
  sorry

end matrix_invertibility_and_fraction_sum_l479_479390


namespace total_weight_is_correct_l479_479497

def siblings_suitcases : Nat := 1 + 2 + 3 + 4 + 5 + 6
def weight_per_sibling_suitcase : Nat := 10
def total_weight_siblings : Nat := siblings_suitcases * weight_per_sibling_suitcase

def parents : Nat := 2
def suitcases_per_parent : Nat := 3
def weight_per_parent_suitcase : Nat := 12
def total_weight_parents : Nat := parents * suitcases_per_parent * weight_per_parent_suitcase

def grandparents : Nat := 2
def suitcases_per_grandparent : Nat := 2
def weight_per_grandparent_suitcase : Nat := 8
def total_weight_grandparents : Nat := grandparents * suitcases_per_grandparent * weight_per_grandparent_suitcase

def other_relatives_suitcases : Nat := 8
def weight_per_other_relatives_suitcase : Nat := 15
def total_weight_other_relatives : Nat := other_relatives_suitcases * weight_per_other_relatives_suitcase

def total_weight_all_suitcases : Nat := total_weight_siblings + total_weight_parents + total_weight_grandparents + total_weight_other_relatives

theorem total_weight_is_correct : total_weight_all_suitcases = 434 := by {
  sorry
}

end total_weight_is_correct_l479_479497


namespace rachel_picked_apples_l479_479203

-- Define relevant variables based on problem conditions
variable (trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ)
variable (total_apples_picked : ℕ)

-- Assume the given conditions
axiom num_trees : trees = 4
axiom apples_each_tree : apples_per_tree = 7
axiom apples_left : remaining_apples = 29

-- Define the number of apples picked
def total_apples_picked_def := trees * apples_per_tree

-- State the theorem to prove the total apples picked
theorem rachel_picked_apples :
  total_apples_picked_def trees apples_per_tree = 28 :=
by
  -- Proof omitted
  sorry

end rachel_picked_apples_l479_479203


namespace tangent_circles_and_locus_l479_479223

theorem tangent_circles_and_locus
  (A B M : Point) (C C' : Circle) (T : Line)
  (hC : is_diameter C A B)
  (hT : is_tangent T C B)
  (h_point_on_circ : M ≠ A ∧ is_on_circle M C)
  (hC' : is_tangent C' T ∧ is_tangent C' C M) :
  ∃ N : Point, is_on_line N T ∧ is_on_circle N C' ∧
               locus_centers_parabola A B ∧
               orthogonal_circle_exists A B C' :=
begin
  sorry
end

end tangent_circles_and_locus_l479_479223


namespace min_value_condition_inequality_condition_l479_479775

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem min_value_condition (a : ℝ) (h : ∃ x, f x a = 2) : a = -1 ∨ a = -5 :=
sorry

theorem inequality_condition (a : ℝ) (h : ∀ x ∈ set.Icc 0 1, f x a ≤ |5 + x|) : -1 ≤ a ∧ a ≤ 2 :=
sorry

end min_value_condition_inequality_condition_l479_479775


namespace constant_a_value_l479_479818

theorem constant_a_value :
  ∀ (a : ℝ), ¬ ∃ x²_coeff : ℝ,
  ∀ (x : ℝ), (x + 1) * (x^2 - 5 * a * x + a) = x^3 + x²_coeff * x^2 + other_terms
  → x²_coeff = 0 → a = 1 / 5 :=
by
  intro a h
  cases h with x²_coeff h₁
  use (0 : ℝ)
  intros x hx
  rw [mul_add, mul_sub, mul_assoc, mul_assoc, add_mul, mul_sub, mul_add]
  simp at hx
  sorry

end constant_a_value_l479_479818


namespace problem_statement_l479_479477

def R' : Set ℕ := { r | ∃ n : ℕ, r = (2 ^ n % 500) }

def S' : ℕ := ∑ r in R', r

theorem problem_statement : S' % 500 = 7 := sorry

end problem_statement_l479_479477


namespace geometric_series_sum_l479_479482

theorem geometric_series_sum :
  let a := 6
  let r := - (2 / 5)
  s = ∑' n, (a * r ^ n) ↔ s = 30 / 7 :=
sorry

end geometric_series_sum_l479_479482


namespace triangle_with_sides_5x_12x_13x_is_right_l479_479977

theorem triangle_with_sides_5x_12x_13x_is_right (x : ℝ) (hx : 0 < x) :
  let a := 5 * x,
      b := 12 * x,
      c := 13 * x in
  a^2 + b^2 = c^2 :=
by
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  calc
    a^2 + b^2 = (5 * x)^2 + (12 * x)^2 : by rfl
           ... = 25 * x^2 + 144 * x^2  : by ring
           ... = 169 * x^2            : by ring
           ... = (13 * x)^2           : by rfl

end triangle_with_sides_5x_12x_13x_is_right_l479_479977


namespace domain_of_function_tan_l479_479957

def domain_of_tan (x : ℝ) : Prop := ∀ k : ℤ, x ≠ (π / 3) + (k * π / 2)

theorem domain_of_function_tan {x : ℝ} : domain_of_tan x ↔ 
  ∀ k : ℤ, x ≠ (π / 3) + (k * π / 2) :=
sorry

end domain_of_function_tan_l479_479957


namespace family_travel_distance_l479_479621

noncomputable def distance_travelled(t1 t2 s1 s2 T : ℝ) : ℝ :=
  let D := 2 * T / ((1/s1) + (1/s2)) in D

theorem family_travel_distance : distance_travelled 1 1 35 40 12 = 448 :=
by
  sorry

end family_travel_distance_l479_479621


namespace collinearity_of_intersections_of_corresponding_sides_l479_479926

theorem collinearity_of_intersections_of_corresponding_sides
    (A B C A' B' C' L P Q R : Point)
    (h1 : Line A A' ∧ Line B B' ∧ Line C C')
    (h2 : ∃ L, Line A A' L ∧ Line B B' L ∧ Line C C' L) :
    (Collinear P Q R ↔ 
    (∃ P Q R : Point, 
    Line B C intersect B' C' = P ∧
    Line C A intersect C' A' = Q ∧
    Line A B intersect A' B' = R ∧
    Point L)) :=
sorry

end collinearity_of_intersections_of_corresponding_sides_l479_479926


namespace max_value_of_f_range_of_k_l479_479083

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (log x) / x - k / x

def slope_at_1 (f : ℝ → ℝ) : ℝ := 10
def slope_condition (k : ℝ) : Prop := deriv (λ x, (log x) / x - k / x) 1 = 10

def inequality1 (x k : ℝ) : Prop := x * x * f x k + 1 / (x + 1) ≥ 0
def inequality2 (x k : ℝ) : Prop := k ≥ 1 / 2 * x * x + (exp 2 - 2) * x - exp x - 7

def max_value_at_exp_eleven (k : ℝ) : ℝ := (log (exp 10)) / (exp 10) - k / (exp 10)

theorem max_value_of_f (k : ℝ) 
  (h₁ : slope_condition k)
  (h₂ : ∀ x ≥ 1, inequality1 x k)
  (h₃ : ∀ x ≥ 1, inequality2 x k) :
  max_value_at_exp_eleven 9 = 1 / (exp 10) := by
  sorry

theorem range_of_k (k : ℝ) 
  (h₁ : ∀ x ≥ 1, inequality1 x k) 
  (h₂ : ∀ x ≥ 1, inequality2 x k) :
  e ^ 2 - 9 ≤ k ∧ k ≤ 1 / 2 := by
  sorry

end max_value_of_f_range_of_k_l479_479083


namespace four_digit_integer_l479_479232

theorem four_digit_integer (a b c d : ℕ) 
    (h1 : a + b + c + d = 16) 
    (h2 : b + c = 10) 
    (h3 : a - d = 2) 
    (h4 : (a - b + c - d) % 11 = 0) : 
    a = 4 ∧ b = 4 ∧ c = 6 ∧ d = 2 :=
sorry

end four_digit_integer_l479_479232


namespace arrange_consecutive_integers_no_common_divisors_l479_479848

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def adjacent (i j : ℕ) (n m : ℕ) : Prop :=
  (abs (i - n) = 1 ∧ j = m) ∨
  (i = n ∧ abs (j - m) = 1) ∨
  (abs (i - n) = 1 ∧ abs (j - m) = 1)

theorem arrange_consecutive_integers_no_common_divisors :
  let grid := [[8, 9, 10], [5, 7, 11], [6, 13, 12]] in
  ∀ i j n m, i < 3 → j < 3 → n < 3 → m < 3 →
  adjacent i j n m →
  is_coprime (grid[i][j]) (grid[n][m]) :=
by
  -- This is where you would prove the theorem
  sorry

end arrange_consecutive_integers_no_common_divisors_l479_479848


namespace angle_CFE_alpha_l479_479875

variables (A B C D E F : Type) [geometry affine_space Type] 
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]

-- Conditions
def trapezoid_ABCD (AB AD BC : ℝ) (is_perpendicular_to_AB_AD : Perp AB AD) (is_perpendicular_to_AB_BC: Perp AB BC) : Prop :=
is_perpendicular_to_AB_AD ∧ is_perpendicular_to_AB_BC

def diagonals_intersect_at_E (P QR: Type) [is_triangulation P QR] (E : P) : Prop :=
triangulation_diagonal_intersection P QR E

def foot_perpendicular_E_to_AB (E AB F : Type) (foot : F) : Prop :=
foot = foot_perpendicular E AB

def angle_DFE_alpha (DFE : Type) (angle : ℝ) (alpha : ℝ): Prop := 
vertex_angle DFE = alpha

-- Problem statement
theorem angle_CFE_alpha (trapezoid_perpendicular: ∀(AB AD BC : ℝ), trapezoid_ABCD AB AD BC) 
(diagonals_intersect: ∀(P QR: Type), diagonals_intersect_at_E P QR E) 
(foot_perpendicular: ∀(E AB F : Type), foot_perpendicular_E_to_AB E AB F)
(alpha_def : ∀(DFE : Type) (angle : ℝ), angle_DFE_alpha DFE angle alpha) :
∃ (angle : ℝ), vertex_angle CFE = alpha := 
sorry

end angle_CFE_alpha_l479_479875


namespace smart_mart_puzzle_sales_l479_479888

variable (s : ℕ) (s_this_week : ℕ) (p : ℕ)

theorem smart_mart_puzzle_sales (h1 : s = 45) 
                                (h2 : s_this_week = Nat.floor (s * 1.2)) 
                                (h3 : p = Nat.floor (s_this_week * 0.70)) : 
                                p = 16 := 
by sorry

end smart_mart_puzzle_sales_l479_479888


namespace problem_part1_problem_part2_l479_479730

theorem problem_part1 (α : ℝ) (h : Real.tan α = -2) :
    (3 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = -4 / 7 := 
    sorry

theorem problem_part2 (α : ℝ) (h : Real.tan α = -2) :
    3 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = -5 := 
    sorry

end problem_part1_problem_part2_l479_479730


namespace sum_first_3m_terms_l479_479978

variable (m : ℕ) (a₁ d : ℕ)

def S (n : ℕ) := n * a₁ + (n * (n - 1)) / 2 * d

-- Given conditions
axiom sum_first_m_terms : S m = 0
axiom sum_first_2m_terms : S (2 * m) = 0

-- Theorem to be proved
theorem sum_first_3m_terms : S (3 * m) = 210 :=
by
  sorry

end sum_first_3m_terms_l479_479978


namespace remainder_when_divided_l479_479717

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 + x^3 + 1

-- The statement to be proved
theorem remainder_when_divided (x : ℝ) : (p 2) = 25 :=
by
  sorry

end remainder_when_divided_l479_479717


namespace find_m_l479_479461

-- Define the vectors a and b
def a (m : ℝ) : EuclideanSpace ℝ (Fin 3) := ![-2, m, 3]
def b : EuclideanSpace ℝ (Fin 3) := ![3, 1, 2]

-- Define the condition that a is orthogonal to b
def orthogonal (a b : EuclideanSpace ℝ (Fin 3)) : Prop := InnerProductSpace.inner a b = 0

-- The statement to be proved: If a and b are orthogonal, then m = 0
theorem find_m (m : ℝ) (h : orthogonal (a m) b) : m = 0 :=
by
  -- Using our conditions as the definitions
  have ha : a m = ![-2, m, 3] := rfl
  have hb : b = ![3, 1, 2] := rfl
  -- The hypothesis and the definition of orthogonality lead to m = 0
  sorry

end find_m_l479_479461


namespace roots_formula_l479_479764

theorem roots_formula (x₁ x₂ p : ℝ)
  (h₁ : x₁ + x₂ = 6 * p)
  (h₂ : x₁ * x₂ = p^2)
  (h₃ : ∀ x, x ^ 2 - 6 * p * x + p ^ 2 = 0 → x = x₁ ∨ x = x₂) :
  (1 / (x₁ + p) + 1 / (x₂ + p) = 1 / p) :=
by
  sorry

end roots_formula_l479_479764


namespace product_bounds_l479_479750

open Real

theorem product_bounds (n : ℕ) (x : Fin n → ℝ) (h1 : 2 ≤ n)
  (h2 : ∀ i, x i ≥ 1 / n)
  (h3 : ∑ i, (x i) ^ 2 = 1) :
  let P := ∏ i, x i in
  P ≤ (1 / (n:ℝ) ^ (n / 2:ℝ)) ∧
  P ≥ (Real.sqrt (n ^ 2 - n + 1) / (n : ℝ) ^ n) :=
  sorry

end product_bounds_l479_479750


namespace gain_percent_is_80_l479_479226

noncomputable def cost_price_per_type (c : ℝ) := (1.5 * c, 2 * c, c)

noncomputable def total_cost_price (countA countB countC : ℕ) (c : ℝ) :=
  1.5 * c * countA + 2 * c * countB + c * countC

def ratio := (3, 4, 5)
def total_chocolates := 81
def selling_price_count := 45

-- Number of each type of chocolates
def count_total := fst ratio + snd ratio + nth ratio 2
def countA := total_chocolates * (fst ratio) / count_total
def countB := total_chocolates * (snd ratio) / count_total
def countC := total_chocolates * (nth ratio 2) / count_total

-- Calculating total cost price
def CP_total (c : ℝ) := total_cost_price countA countB countC c

-- Selling price is given for 45 chocolates, which is equal to the cost price of 81 chocolates
def selling_price_per_chocolate (c : ℝ) := (CP_total c) / selling_price_count

-- Cost price for 45 chocolates
def CP_45 (c : ℝ) := selling_price_per_chocolate c * selling_price_count

-- Gain calculation
def gain (c : ℝ) := selling_price_per_chocolate c * /* number of chocolates */  selling_price_count - CP_45 c

-- Gain percent calculation
def gain_percent (c : ℝ) := (gain c) / CP_45 c * 100

-- The goal statement in Lean 4.
theorem gain_percent_is_80 :
  ∀ c : ℝ, gain_percent c = 80 :=
by
  intro c
  sorry

end gain_percent_is_80_l479_479226


namespace find_n_l479_479854

noncomputable def positive_geometric_sequence := ℕ → ℝ

def is_geometric_sequence (a : positive_geometric_sequence) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def conditions (a : positive_geometric_sequence) :=
  is_geometric_sequence a ∧
  a 0 * a 1 * a 2 = 4 ∧
  a 3 * a 4 * a 5 = 12 ∧
  ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324

theorem find_n (a : positive_geometric_sequence) (h : conditions a) : ∃ n : ℕ, n = 14 :=
by
  sorry

end find_n_l479_479854


namespace flower_growth_l479_479921

theorem flower_growth (total_seeds : ℕ) (seeds_per_bed : ℕ) (max_grow_per_bed : ℕ) (h1 : total_seeds = 55) (h2 : seeds_per_bed = 15) (h3 : max_grow_per_bed = 60) : total_seeds ≤ 55 :=
by
  -- use the given conditions
  have h4 : total_seeds = 55 := h1
  sorry -- Proof goes here, omitted as instructed

end flower_growth_l479_479921


namespace arrangement_is_correct_l479_479839

-- Definition of adjacency in a 3x3 matrix.
def adjacent (i j k l : Nat) : Prop := 
  (i = k ∧ j = l + 1) ∨ (i = k ∧ j = l - 1) ∨ -- horizontal adjacency
  (i = k + 1 ∧ j = l) ∨ (i = k - 1 ∧ j = l) ∨ -- vertical adjacency
  (i = k + 1 ∧ j = l + 1) ∨ (i = k - 1 ∧ j = l - 1) ∨ -- diagonal adjacency
  (i = k + 1 ∧ j = l - 1) ∨ (i = k - 1 ∧ j = l + 1)   -- diagonal adjacency

-- Definition to check if two numbers share a common divisor greater than 1.
def coprime (x y : Nat) : Prop := Nat.gcd x y = 1

-- The arrangement of the numbers in the 3x3 grid.
def grid := 
  [ [8, 9, 10],
    [5, 7, 11],
    [6, 13, 12] ]

-- Ensure adjacents numbers are coprime
def correctArrangement :=
  (coprime grid[0][0] grid[0][1]) ∧ (coprime grid[0][1] grid[0][2]) ∧
  (coprime grid[1][0] grid[1][1]) ∧ (coprime grid[1][1] grid[1][2]) ∧
  (coprime grid[2][0] grid[2][1]) ∧ (coprime grid[2][1] grid[2][2]) ∧
  (adjacent 0 0 1 1 → coprime grid[0][0] grid[1][1]) ∧
  (adjacent 0 2 1 1 → coprime grid[0][2] grid[1][1]) ∧
  (adjacent 1 0 2 1 → coprime grid[1][0] grid[2][1]) ∧
  (adjacent 1 2 2 1 → coprime grid[1][2] grid[2][1]) ∧
  (coprime grid[0][1] grid[1][0]) ∧ (coprime grid[0][1] grid[1][2]) ∧
  (coprime grid[2][1] grid[1][0]) ∧ (coprime grid[2][1] grid[1][2])

-- Statement to be proven
theorem arrangement_is_correct : correctArrangement := 
  sorry

end arrangement_is_correct_l479_479839


namespace kitten_food_consumption_l479_479935

-- Definitions of the given conditions
def k : ℕ := 4  -- Number of kittens
def ac : ℕ := 3  -- Number of adult cats
def f : ℕ := 7  -- Initial cans of food
def af : ℕ := 35  -- Additional cans of food needed
def days : ℕ := 7  -- Total number of days

-- Definition of the food consumption per adult cat per day
def food_per_adult_cat_per_day : ℕ := 1

-- Definition of the correct answer: food per kitten per day
def food_per_kitten_per_day : ℚ := 0.75

-- Proof statement
theorem kitten_food_consumption (k : ℕ) (ac : ℕ) (f : ℕ) (af : ℕ) (days : ℕ) (food_per_adult_cat_per_day : ℕ) :
  (ac * food_per_adult_cat_per_day * days + k * food_per_kitten_per_day * days = f + af) → 
  food_per_kitten_per_day = 0.75 :=
sorry

end kitten_food_consumption_l479_479935


namespace complex_number_solution_l479_479382

theorem complex_number_solution (z : ℂ) (i : ℂ) (h_i : i * i = -1) (h : z + z * i = 1 + 5 * i) : z = 3 + 2 * i :=
sorry

end complex_number_solution_l479_479382


namespace money_left_after_bike_purchase_l479_479159

-- Definitions based on conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def quarter_value : ℝ := 0.25
def bike_cost : ℝ := 180

-- The theorem statement
theorem money_left_after_bike_purchase : (jars * quarters_per_jar * quarter_value) - bike_cost = 20 := by
  sorry

end money_left_after_bike_purchase_l479_479159


namespace factor_expression_l479_479680

theorem factor_expression (x : ℝ) : 16 * x^4 - 4 * x^2 = 4 * x^2 * (2 * x + 1) * (2 * x - 1) :=
sorry

end factor_expression_l479_479680


namespace domain_tan_2x_minus_pi_over_6_l479_479954

def tan_undefined (θ : ℝ) : Prop := ∃ k : ℤ, θ = (π / 2) + k * π

theorem domain_tan_2x_minus_pi_over_6 :
  ∀ x : ℝ, ¬ tan_undefined (2 * x - π / 6) ↔ ¬ ∃ k : ℤ, x = (π / 3) + k * (π / 2) :=
by sorry

end domain_tan_2x_minus_pi_over_6_l479_479954


namespace ramesh_installation_cost_l479_479927

-- Definitions of the given conditions
def labelled_price : ℝ := 12500 / 0.80
def transport_cost : ℝ := 125
def no_discount_selling_price : ℝ := labelled_price * 1.12
def required_selling_price : ℝ := 17920

-- Installation cost calculation
def extra_cost : ℝ := required_selling_price - no_discount_selling_price
def installation_cost : ℝ := extra_cost - transport_cost

-- Proof statement
theorem ramesh_installation_cost :
  ∀ (labelled_price = 12500 / 0.80),
  ∀ (transport_cost = 125),
  ∀ (no_discount_selling_price = labelled_price * 1.12),
  ∀ (required_selling_price = 17920),
  installation_cost = 295 := 
by 
  sorry

end ramesh_installation_cost_l479_479927


namespace length_of_EH_l479_479863

noncomputable def EF : ℝ := 7
noncomputable def FG : ℝ := 6
noncomputable def GH : ℝ := 24
noncomputable def FJ : ℝ := GH - EF 
noncomputable def EJ : ℝ := Real.sqrt (EF^2 + FJ^2)
noncomputable def EH : ℝ := EJ

theorem length_of_EH :
  EH = Real.sqrt 338 := by
  have h_EF : EF = 7 := rfl
  have h_GH : GH = 24 := rfl
  have h_FJ : FJ = GH - EF := rfl
  have h_calculate_FJ : FJ = 17 := by simp [FJ, GH, EF, h_EF, h_GH]
  have h_Pythagorean : EJ^2 = EF^2 + FJ^2 := by simp [EJ, EF, FJ]
  have h_calculate_EJ : EJ = Real.sqrt (EF^2 + FJ^2) := by simp [EJ, EF, FJ]
  have h_EJ : EJ = Real.sqrt 338 := by simp [EJ, h_calculate_EJ, h_calculate_FJ]
  have h_EH : EH = EJ := rfl
  rw [h_EJ, h_EH]
  simp

end length_of_EH_l479_479863


namespace equal_tangents_l479_479141

-- Definitions
variables {A B C D O : Type}
variables [hQuad : ConvexQuadrilateral ABCD]
variables [hDiagIntersect : DiagonalsIntersectAt AC BD O]
variables [hAngle1 : AngleEqual BAC CBD]
variables [hAngle2 : AngleEqual BCA CDB]

-- Theorem statement
theorem equal_tangents (A B C D O : Type) [hQuad : ConvexQuadrilateral ABCD]
  [hDiagIntersect : DiagonalsIntersectAt AC BD O]
  [hAngle1 : AngleEqual BAC CBD]
  [hAngle2 : AngleEqual BCA CDB] :
  equal_tangents_to_circumcircle B C (circumcircle A O D) :=
sorry

end equal_tangents_l479_479141


namespace tangent_line_circle_l479_479804

theorem tangent_line_circle (r : ℝ) (h₁ : 0 < r) (h₂ : tangent (λ p : ℝ × ℝ, p.1 + p.2 = r) (λ p : ℝ × ℝ, p.1^2 + p.2^2 = r)) : r = 2 :=
sorry

end tangent_line_circle_l479_479804


namespace find_f_prime_at_1_l479_479817

noncomputable def f (x : ℝ) : ℝ := x^3 - f' 1 * x^2 + 3

theorem find_f_prime_at_1 : deriv f 1 = 1 := by
  sorry

end find_f_prime_at_1_l479_479817


namespace CL_tangent_C2_l479_479948

theorem CL_tangent_C2
  (C1 C2 : Circle) (O1 O2 : Point)
  (h1 : C1.center = O1) (h2 : C2.center = O2)
  (h3 : externally_tangent C1 C2)
  (A B : Point) (h4 : tangent_point C1 A) (h5 : tangent_point C2 B)
  (h6 : not_intersect (segment O1 O2) (tangent C1 C2))
  (C : Point) (h7 : reflection A (line O1 O2) = C)
  (P : Point) (h8 : intersection (line A C) (line O1 O2) = P)
  (L : Point) (h9 : second_intersection (line B P) C2 = L) :
  is_tangent (line C L) C2 :=
sorry

end CL_tangent_C2_l479_479948


namespace base9_number_perfect_square_l479_479737

theorem base9_number_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : 0 ≤ d ∧ d ≤ 8) (n : ℕ) 
  (h3 : n = 729 * a + 81 * b + 45 + d) (h4 : ∃ k : ℕ, k * k = n) : d = 0 := 
sorry

end base9_number_perfect_square_l479_479737


namespace geometric_series_sum_l479_479347

theorem geometric_series_sum:
  let a := 1
  let r := 5
  let n := 5
  (1 - r^n) / (1 - r) = 781 :=
by
  let a := 1
  let r := 5
  let n := 5
  sorry

end geometric_series_sum_l479_479347


namespace sequence_general_term_l479_479380

theorem sequence_general_term (n : ℕ) (h : n > 0) : 
  (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2 * n) 
  : a n = n * n - n + 1 :=
sorry

end sequence_general_term_l479_479380


namespace value_of_expression_l479_479127

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end value_of_expression_l479_479127


namespace derivative_f_l479_479417

def f (x : ℝ) : ℝ := Real.cos (2 * x) * Real.log x

theorem derivative_f :
  ∀ x, (f' x) = -2 * Real.sin(2 * x) * Real.log(x) + Real.cos(2 * x) / x :=
by
  sorry

end derivative_f_l479_479417


namespace cultural_performance_l479_479577
open Classical

variable (total : ℕ) (singing : ℕ) (dancing : ℕ) (sing_and_dance : ℕ)
variable (number_of_ways : ℕ)

theorem cultural_performance : 
  total = 6 →
  singing = 3 →
  dancing = 2 →
  sing_and_dance = 1 →
  number_of_ways = (choose 3 2) * (choose 2 1) →
  number_of_ways = 15 := by
  intros h1 h2 h3 h4 h5
  rw h5
  simp
  sorry

end cultural_performance_l479_479577


namespace monica_individual_students_l479_479918

-- Definitions of each class's student counts and overlaps:
def class1 : ℕ := 20
def class2 : ℕ := 25
def class3 : ℕ := 25
def class4 : ℕ := class1 / 2
def class5 : ℕ := 28
def class6 : ℕ := 28

def overlap1_and_2 : ℕ := 5
def overlap4_and_5 : ℕ := 3
def overlap2_3_and_6 : ℕ := 6
def overlap3_and_6_average : ℕ := 10 -- already counted, so not subtracted again
def overlap5_and_6_average : ℕ := 8

-- Total student spots without considering overlaps:
def total_spots : ℕ := class1 + class2 + class3 + class4 + class5 + class6

-- Total overlaps that need to be subtracted:
def total_overlaps : ℕ := overlap1_and_2 + overlap4_and_5 + overlap2_3_and_6 + overlap5_and_6_average

-- We are to prove:
theorem monica_individual_students : total_spots - total_overlaps = 114 :=
by
  have h1 : total_spots = 20 + 25 + 25 + 10 + 28 + 28 := rfl
  have h2 : total_overlaps = 5 + 3 + 6 + 8 := rfl
  have h3 : 136 - 22 = 114 := rfl
  rw [h1, h2, h3]
  sorry

end monica_individual_students_l479_479918


namespace negative_represents_backward_l479_479950

-- Definitions based on conditions
def forward (distance : Int) : Int := distance
def backward (distance : Int) : Int := -distance

-- The mathematical equivalent proof problem
theorem negative_represents_backward
  (distance : Int)
  (h : forward distance = 5) :
  backward distance = -5 :=
sorry

end negative_represents_backward_l479_479950


namespace wise_men_hat_guessing_l479_479530

theorem wise_men_hat_guessing :
  ∃ (strategy : (fin 300 → fin 25) → (fin 300 → fin 25)), ∀ (hats : fin 300 → fin 25),
  (∑ i, if strategy hats i = hats i then 1 else 0) ≥ 150 := 
sorry

end wise_men_hat_guessing_l479_479530


namespace cos_2θ_eq_neg_half_λ_range_l479_479053

noncomputable def z1 (θ : ℝ) : ℂ := 2 * sin θ - complex.I * √3
noncomputable def z2 (θ : ℝ) : ℂ := 1 + complex.I * (2 * cos θ)

def θ_range (θ : ℝ) : Prop := θ ∈ set.Icc (π / 3) (π / 2)
def is_real (z : ℂ) : Prop := z.im = 0

def a (θ : ℝ) : ℝ × ℝ := (2 * sin θ, -√3)
def b (θ : ℝ) : ℝ × ℝ := (1, 2 * cos θ)

def dot (v w : ℝ × ℝ) : ℝ :=
  prod.fst v * prod.fst w + prod.snd v * prod.snd w

theorem cos_2θ_eq_neg_half (θ : ℝ) (hθ : θ_range θ) (H : is_real (z1 θ * z2 θ)) : cos (2 * θ) = -1 / 2 :=
sorry

theorem λ_range (θ : ℝ) (hθ : θ_range θ) (λ : ℝ)
  (H : dot (λ • a θ - b θ) (a θ - λ • b θ) = 0) :
  λ ∈ set.Iic (2 - √3) ∪ set.Ici (2 + √3) :=
sorry

end cos_2θ_eq_neg_half_λ_range_l479_479053


namespace sum_of_sequence_l479_479335

noncomputable def sequence_sum : Int :=
  let terms := List.range' 2020 (-15) 134
  terms.zipWith (fun (i : Nat) (x : Int) => if i % 2 = 0 then x else -x) (List.range 0 134) |>.sum

theorem sum_of_sequence :
  sequence_sum = 1035 :=
by
  sorry

end sum_of_sequence_l479_479335


namespace alice_winning_strategy_l479_479492

theorem alice_winning_strategy (n : ℕ) (hn : n ≥ 2) : 
  (Alice_has_winning_strategy ↔ n % 4 = 3) :=
sorry

end alice_winning_strategy_l479_479492


namespace sum_sequence_l479_479031

theorem sum_sequence (n : ℕ) : 
    ∑ k in Finset.range(n + 1), (3 * k + 2) - 3 = (3 * n^2 + 7 * n - 2) / 2 := 
by 
    sorry

end sum_sequence_l479_479031


namespace find_least_m_in_interval_l479_479001

noncomputable def x : ℕ → ℚ
| 0     := 7
| (n+1) := (x n ^ 2 + 7 * x n + 6) / (x n + 8)

def limit_value : ℚ := 5 + 1 / 2 ^ 18

def m := Nat.find (λ n, x n ≤ limit_value)

theorem find_least_m_in_interval :
  71 ≤ m ∧ m ≤ 210 :=
by
  have x0 := 7
  have recursive_def: ∀ n, x (n+1) = (x n ^ 2 + 7 * x n + 6) / (x n + 8) := calc
    ∀ n, x (n + 1) = by sorry
  
  -- Proof omitted (insert steps here to show that 71 ≤ m and m ≤ 210)
  sorry

end find_least_m_in_interval_l479_479001


namespace city_graph_degree_bound_l479_479850

/-- In a certain country with initially 2002 cities, each city connected through
cannot be isolated from other cities by blocking a city and each year new city 
is built and connected such that it breaks cycles maintaining acyclic graph which 
finally results in degree >= 2002 for every city. 
Prove that at this point, the number of roads leading to the outside from any 
one city is no less than 2002. -/
theorem city_graph_degree_bound (n : ℕ) (G : Graph) (h_initial : G.order = 2002)
  (h_connected : ∀ (v : G.vertex), is_connectivity_preserved G v)
  (h_break_cycles : ∀ k G' (hk : k < n) (hG'_ : acyclic G') (v : G'.vertex),
    acyclic (add_vertex_and_break_cycle G' v)) :
  ∀ G' (hn : G'.order = n), degree_bound G' :=
sorry

end city_graph_degree_bound_l479_479850


namespace projection_vector_l479_479028

-- Definitions of the vectors involved
def vector_a : ℝ × ℝ × ℝ := (3, 2, 4)
def direction_vector_d : ℝ × ℝ × ℝ := (1, -1/2, 1/3)

-- Statement to prove the projection
theorem projection_vector :
  let scalar_proj := ( (3 * 1 + 2 * (-1 / 2) + 4 * (1 / 3)) / (1^2 + (-1/2)^2 + (1/3)^2) )
  scalar_proj * 1 = 168 / 37 
  ∧ scalar_proj * (-1 / 2) = -84 / 37 
  ∧ scalar_proj * (1 / 3) = 56 / 37
:=
by {
  let scalar_proj := ( (3 * 1 + 2 * (-1 / 2) + 4 * (1 / 3)) / (1^2 + (-1/2)^2 + (1/3)^2) ),
  split,
  {
    calc
      scalar_proj * 1 = 168 / 37 : sorry
  },
  split,
  {
    calc
      scalar_proj * (-1 / 2) = -84 / 37 : sorry
  },
  {
    calc
      scalar_proj * (1 / 3) = 56 / 37 : sorry
  }
}

end projection_vector_l479_479028


namespace arithmetic_progression_sum_l479_479760

theorem arithmetic_progression_sum (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 0 + n * d)
  (h2 : a 0 = 2)
  (h3 : a 1 + a 2 = 13) :
  a 3 + a 4 + a 5 = 42 :=
sorry

end arithmetic_progression_sum_l479_479760


namespace monotonicity_of_f_l479_479766

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 1

theorem monotonicity_of_f (a x : ℝ) :
  (a > 0 → ((∀ x, (x < -2 * a / 3 → f a x' > f a x) ∧ (x > 0 → f a x' > f a x)) ∧ (∀ x, (-2 * a / 3 < x ∧ x < 0 → f a x' < f a x)))) ∧
  (a = 0 → ∀ x, f a x' > f a x) ∧
  (a < 0 → ((∀ x, (x < 0 → f a x' > f a x) ∧ (x > -2 * a / 3 → f a x' > f a x)) ∧ (∀ x, (0 < x ∧ x < -2 * a / 3 → f a x' < f a x)))) :=
sorry

end monotonicity_of_f_l479_479766


namespace face_opposite_violet_is_blue_l479_479640

-- Conditions encoded as definitions
def faces : Type := {orange, black, yellow, violet, blue, pink}

-- Views encoded as sets of observations
def view1 := (top = blue ∧ front = yellow ∧ right = orange)
def view2 := (top = blue ∧ front = pink ∧ right = orange)
def view3 := (top = blue ∧ front = black ∧ right = orange)

-- Define the problem statement
theorem face_opposite_violet_is_blue : 
  (∀ (top front right bot left back : faces), 
  (view1 ∧ view2 ∧ view3) → 
  (bot = violet → top = blue)) :=
sorry

end face_opposite_violet_is_blue_l479_479640


namespace dice_probability_l479_479190

-- Definitions of events for even number roll and prime number roll
def even_numbers := {2, 4, 6}
def prime_numbers := {2, 3, 5}

-- Given conditions:
-- 1. Two six-sided dice are rolled
-- 2. Probabilities involved
noncomputable def probability_even_first : ℚ := 3 / 6
noncomputable def probability_prime_second : ℚ := 3 / 6

-- Question: Prove that the combined probability is 1/4
theorem dice_probability : probability_even_first * probability_prime_second = 1 / 4 :=
by sorry

end dice_probability_l479_479190


namespace difference_of_numbers_l479_479536

noncomputable def larger_num : ℕ := 1495
noncomputable def quotient : ℕ := 5
noncomputable def remainder : ℕ := 4

theorem difference_of_numbers :
  ∃ S : ℕ, larger_num = quotient * S + remainder ∧ (larger_num - S = 1197) :=
by 
  sorry

end difference_of_numbers_l479_479536


namespace smallest_positive_multiple_of_45_l479_479603

theorem smallest_positive_multiple_of_45 : ∃ (n : ℕ), n > 0 ∧ ∃ (x : ℕ), x > 0 ∧ n = 45 * x ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l479_479603


namespace find_derivative_at_one_l479_479810

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := x^3 - c * x^2 + 3

theorem find_derivative_at_one (c : ℝ) (h : ∀ x : ℝ, Derivative (f c) x = 3 * x^2 - 2 * f (c) 1 * x)
  : f c 1 = 1 := sorry

end find_derivative_at_one_l479_479810


namespace length_FD_l479_479869

-- Definitions of the conditions in Lean
def is_square (A B C D : Point) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C D = side_length ∧ dist D A = side_length ∧
  ∠ ABC = π / 2 ∧ ∠ BCD = π / 2 ∧ ∠ CDA = π / 2 ∧ ∠ DAB = π / 2

-- Given points A, B, C, D forming a square
variable {A B C D E F G : Point}

-- Assume ABCD is a square with each side equal to 8 cm
axiom h1 : is_square A B C D 8

-- Given E located \(\frac{1}{4}\) along \(\overline{AD}\) from \(D\)
axiom h2 : E = point_on_line_segment D A (1 - 1/4)

-- Given F is on \(\overline{CD}\) and \(\overline{CF}\) coincides with \(\overline{EF}\)
axiom h3 : F ∈ line_segment C D ∧ CF = EF

-- To Prove: The length of \(\overline{FD}\) is \(\frac{15}{4} \ \text{cm}\)
theorem length_FD : dist F D = 15 / 4 := by
  sorry

end length_FD_l479_479869


namespace rook_visits_all_squares_even_n_l479_479866

theorem rook_visits_all_squares_even_n (n : ℕ) (is_even : n % 2 = 0) :
  ∃ path : list (ℕ × ℕ), 
    (∀ p ∈ path, 1 ≤ p.1 ∧ p.1 ≤ n ∧ 1 ≤ p.2 ∧ p.2 ≤ n) ∧
    path.length = n * n ∧
    (list.chain' (λ a b, (a.1 = b.1 ∧ abs (a.2 - b.2) = 1) ∨ (a.2 = b.2 ∧ abs (a.1 - b.2) = 1)) path) ∧
    list.head path = some (1, 1) ∧
    list.last path = some (1, 1) :=
sorry

end rook_visits_all_squares_even_n_l479_479866


namespace smallest_M_equals_35_l479_479454

noncomputable def smallest_M (P : Fin 6 → Nat) (M : Nat) : Prop :=
∀ i j : Fin 6,
  let c := fun i => P i % M + 1
  in if i = 0 then c 0 = 6 * c 1 - 5 else
     if i = 1 then c 1 + M = 6 * c 0 - 5 else
     if i = 2 then c 2 + 2 * M = 6 * c 3 - 3 else
     if i = 3 then c 3 + 3 * M = 6 * c 2 - 3 else
     if i = 4 then c 4 + 4 * M = 6 * c 5 - 1 else
     if i = 5 then c 5 + 5 * M = 6 * c 4 - 1 else
     True

theorem smallest_M_equals_35 : ∃ M, smallest_M (fun i => i.val * 6 + 1) M ∧ M = 35 :=
by
  use 35
  sorry

end smallest_M_equals_35_l479_479454


namespace exists_fixed_point_f_l479_479085

def f (x : ℝ) : ℝ := x^3 - x^2 + (x / 2) + (1 / 4)

theorem exists_fixed_point_f : ∃ x₀ ∈ (set.Ioo 0 (1 / 2)), f x₀ = x₀ :=
by
  sorry

end exists_fixed_point_f_l479_479085


namespace regular_tire_price_l479_479656

theorem regular_tire_price 
  (x : ℝ) 
  (h1 : 3 * x + x / 2 = 300) 
  : x = 600 / 7 := 
sorry

end regular_tire_price_l479_479656


namespace find_n_mod_l479_479362

theorem find_n_mod (n : ℤ) : n ≡ 27514 [MOD 16] ∧ 0 ≤ n ∧ n ≤ 15 ↔ n = 10 := by
  sorry

end find_n_mod_l479_479362


namespace digits_same_l479_479516

theorem digits_same (n : ℕ) (h₁ : 2^2003 < 10^2003) 
  (h₂ : 1990^2003 = 10^(3 * 2003) * n): 
  nat.digits 10 (1990^2003) = nat.digits 10 (1990^2003 + 2^2003) := 
sorry

end digits_same_l479_479516


namespace bess_throw_distance_l479_479670

-- Definitions based on the conditions
def bess_throws (x : ℝ) : ℝ := 4 * 2 * x
def holly_throws : ℝ := 5 * 8
def total_throws (x : ℝ) : ℝ := bess_throws x + holly_throws

-- Lean statement for the proof
theorem bess_throw_distance (x : ℝ) (h : total_throws x = 200) : x = 20 :=
by 
  sorry

end bess_throw_distance_l479_479670


namespace sum_f_values_l479_479000

noncomputable def f : ℝ → ℝ :=
  λ x, if h₁ : -3 ≤ x ∧ x < -1 then -(x+2)^2 
       else if h₂ : -1 ≤ x ∧ x < 3 then x 
       else f (x - 6 * ⌊x/6⌋) -- Making use of periodicity

theorem sum_f_values :
  (∑ i in finset.range 2012, f (i + 1)) = 338 :=
sorry

end sum_f_values_l479_479000


namespace inequality_subtraction_l479_479045

theorem inequality_subtraction {a b c : ℝ} (h : a > b) : a - c > b - c := 
sorry

end inequality_subtraction_l479_479045


namespace calculate_ff1_l479_479728

def f (x : ℝ) : ℝ :=
  if x > 1 then x + 5
  else 2 * x ^ 2 + 1

theorem calculate_ff1 : f (f 1) = 8 := by
  sorry

end calculate_ff1_l479_479728


namespace amount_subtracted_is_15_l479_479703

theorem amount_subtracted_is_15 (n x : ℕ) (h1 : 7 * n - x = 2 * n + 10) (h2 : n = 5) : x = 15 :=
by 
  sorry

end amount_subtracted_is_15_l479_479703


namespace derivative_at_one_l479_479814

def f (x : ℝ) : ℝ := x^3 - f' 1 * x^2 + 3

theorem derivative_at_one :
  (deriv f 1) = 1 := by
sorry

end derivative_at_one_l479_479814


namespace find_m_value_l479_479236

-- Definitions of the hyperbola and its focus condition
def hyperbola_eq (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / m) - (y^2 / (3 + m)) = 1

def focus_condition (m : ℝ) : Prop :=
  4 = (m) + (3 + m)

-- Theorem stating the value of m
theorem find_m_value (m : ℝ) : hyperbola_eq m → focus_condition m → m = 1 / 2 :=
by
  intros
  sorry

end find_m_value_l479_479236


namespace minimum_value_of_F_inequality_f_g_l479_479746

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x a : ℝ) : ℝ := x + a / x
noncomputable def F (x a : ℝ) : ℝ := Real.log x + a / x

theorem minimum_value_of_F 
  (a : ℝ) 
  (h : ∃ x ∈ Set.Icc 1 Real.exp 1, F x a = 3 / 2) : 
  a = Real.sqrt (Real.exp 1) := sorry

theorem inequality_f_g 
  (a : ℝ)
  (h : ∀ x ≥ 1, Real.log x ≤ x + a / x) : 
  a ≥ -1 := sorry

end minimum_value_of_F_inequality_f_g_l479_479746


namespace monotonicity_and_extreme_points_l479_479412

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^2 + a * log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - 4 * x + 2

theorem monotonicity_and_extreme_points (a x0 : ℝ) (h1 : 0 < a) (h2 : a < 1) (hx0 : 1 / 2 < x0) (hx0_upper : x0 < 1) 
  (h_min : ∀ x ∈ set.Ioi 0, deriv (g x a) x = 0 → x = x0 ∨ ∃ x1, x1 ≠ x0 ∧ deriv (g x a) x1 = 0) :
  g x0 a > 1 / 2 - log 2 :=
sorry

end monotonicity_and_extreme_points_l479_479412


namespace total_people_attended_l479_479318

theorem total_people_attended (A C : ℕ) (ticket_price_adult ticket_price_child : ℕ) (total_receipts : ℕ) 
  (number_of_children : ℕ) (h_ticket_prices : ticket_price_adult = 60 ∧ ticket_price_child = 25)
  (h_total_receipts : total_receipts = 140 * 100) (h_children : C = 80) 
  (h_equation : ticket_price_adult * A + ticket_price_child * C = total_receipts) : 
  A + C = 280 :=
by
  sorry

end total_people_attended_l479_479318


namespace least_possible_value_f_1998_l479_479691

theorem least_possible_value_f_1998 
  (f : ℕ → ℕ)
  (h : ∀ m n, f (n^2 * f m) = m * (f n)^2) : 
  f 1998 = 120 :=
sorry

end least_possible_value_f_1998_l479_479691


namespace radius_of_given_spherical_circle_l479_479563
noncomputable def circle_radius_spherical_coords : Real :=
  let spherical_to_cartesian (rho theta phi : Real) : (Real × Real × Real) :=
    (rho * (Real.sin phi) * (Real.cos theta), rho * (Real.sin phi) * (Real.sin theta), rho * (Real.cos phi))
  let (rho, theta, phi) := (1, 0, Real.pi / 3)
  let (x, y, z) := spherical_to_cartesian rho theta phi
  let radius := Real.sqrt (x^2 + y^2)
  radius

theorem radius_of_given_spherical_circle :
  circle_radius_spherical_coords = (Real.sqrt 3) / 2 :=
sorry

end radius_of_given_spherical_circle_l479_479563


namespace students_before_Yoongi_l479_479311

theorem students_before_Yoongi (total_students : ℕ) (students_after_Yoongi : ℕ) 
  (condition1 : total_students = 20) (condition2 : students_after_Yoongi = 11) :
  total_students - students_after_Yoongi - 1 = 8 :=
by 
  sorry

end students_before_Yoongi_l479_479311


namespace rectangle_area_in_triangle_l479_479314

-- definitions corresponding to the conditions
variables (b h x : ℝ)
def triangle_base := b
def triangle_altitude := 3 * h
def rectangle_height := x

-- theorem statement
theorem rectangle_area_in_triangle
  (hb : triangle_base = b)
  (h3h : triangle_altitude = 3 * h)
  (hx : rectangle_height = x) :
  ∃ (A : ℝ), A = (b * x / (3 * h)) * (3 * h - x) := 
by
  use (b * x / (3 * h)) * (3 * h - x)
  rw [hb, h3h, hx]
  sorry

end rectangle_area_in_triangle_l479_479314


namespace find_solution_set_l479_479708

theorem find_solution_set (x : ℝ) : 
  (1 / (x + 2) + 8 / (x + 6) ≥ 1)
  ↔ (x ∈ set.Ioc (-6 : ℝ) 5 ∨ x = 5) := 
sorry

end find_solution_set_l479_479708


namespace tan_alpha_not_unique_l479_479107

theorem tan_alpha_not_unique (α : ℝ) (h1 : α > 0) (h2 : α < Real.pi) (h3 : (Real.sin α)^2 + Real.cos (2 * α) = 1) :
  ¬(∃ t : ℝ, Real.tan α = t) :=
by
  sorry

end tan_alpha_not_unique_l479_479107


namespace isosceles_AC1A1_l479_479433

theorem isosceles_AC1A1 {A B C A1 C1 : Type*}
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace A1] [MetricSpace C1]
  (triangle_isosceles : AB = BC)
  (angle_bisectors : IsAngleBisector A A1 ∧ IsAngleBisector C C1) :
  IsIsoscelesTriangle A C1 A1 :=
sorry

end isosceles_AC1A1_l479_479433


namespace bounded_sequence_l479_479370

noncomputable def exists_unique_sequence (a : ℝ) (n : ℕ) (h : n > 0) :
  ∃! (x : Fin (n + 2) → ℝ), x 0 = 0 ∧ x (Fin.last (n + 1)) = 0 ∧
  ∀ i : Fin n, 1/2 * (x i + x ⟨i + 1, Nat.ltOfLtOfLe (Fin.is_lt i) (Nat.le_succ n)⟩) = x i + x i ^ 3 - a ^ 3 :=
sorry

theorem bounded_sequence (a : ℝ) (n : ℕ) (h : n > 0) :
  ∀ (x : Fin (n + 2) → ℝ), (x 0 = 0 ∧ x (Fin.last (n + 1)) = 0 ∧
  ∀ i : Fin n, 1/2 * (x i + x ⟨i + 1, Nat.ltOfLtOfLe (Fin.is_lt i) (Nat.le_succ n)⟩) = x i + x i ^ 3 - a ^ 3) →
  ∀ i : Fin (n + 2), |x i| ≤ |a| :=
sorry

end bounded_sequence_l479_479370


namespace four_digit_integer_l479_479231

theorem four_digit_integer (a b c d : ℕ) 
    (h1 : a + b + c + d = 16) 
    (h2 : b + c = 10) 
    (h3 : a - d = 2) 
    (h4 : (a - b + c - d) % 11 = 0) : 
    a = 4 ∧ b = 4 ∧ c = 6 ∧ d = 2 :=
sorry

end four_digit_integer_l479_479231


namespace corrected_mean_of_observations_l479_479549

theorem corrected_mean_of_observations :
  (mean : ℝ) (n : ℕ) (inc_obs corr_obs : ℝ)
  (h1 : n = 50)
  (h2 : mean = 36)
  (h3 : inc_obs = 23)
  (h4 : corr_obs = 45) :
  (mean * n - inc_obs + corr_obs) / n = 36.44 :=
by
  sorry

end corrected_mean_of_observations_l479_479549


namespace total_students_l479_479526

theorem total_students (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hshake : (2 * m * n - m - n) = 252) : m * n = 72 :=
  sorry

end total_students_l479_479526


namespace digits_in_8_20_3_30_base_12_l479_479693

def digits_in_base (n b : ℕ) : ℕ :=
  if n = 0 then 1 else 1 + Nat.log b n

theorem digits_in_8_20_3_30_base_12 : digits_in_base (8^20 * 3^30) 12 = 31 :=
by
  sorry

end digits_in_8_20_3_30_base_12_l479_479693


namespace algebraic_expression_value_l479_479124

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 :=
by
  sorry

end algebraic_expression_value_l479_479124


namespace equilateral_triangle_area_l479_479154

noncomputable def is_point_in_triangle (Q X Y Z : ℝ × ℝ) : Prop := sorry

theorem equilateral_triangle_area
  (X Y Z : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (h1 : dist Q X = 10)
  (h2 : dist Q Y = 12)
  (h3 : dist Q Z = 14)
  (h4 : dist X Y = dist Y Z ∧ dist Y Z = dist Z X) :
  (let A := (dist X Y) in let area := (sqrt 3 / 4) * (A^2) in round(area) = 158) :=
sorry

end equilateral_triangle_area_l479_479154


namespace find_k_l479_479286

def a : ℕ := 786
def b : ℕ := 74
def c : ℝ := 1938.8

theorem find_k (k : ℝ) : (a * b) / k = c → k = 30 :=
by
  intro h
  sorry

end find_k_l479_479286


namespace product_of_coordinates_l479_479064

open_locale classical

variables {a b : ℝ}

def P1 := (a, 5 : ℝ)
def P2 := (-4, b : ℝ)

-- Symmetry condition: The points P1 and P2 are symmetrical about the x-axis
def symmetrical_about_x_axis (P1 P2 : ℝ × ℝ) : Prop :=
P1.1 = - P2.1 ∧ P1.2 = - P2.2

theorem product_of_coordinates (h : symmetrical_about_x_axis (a, 5) (-4, b)) : a * b = -20 :=
sorry

end product_of_coordinates_l479_479064


namespace find_derivative_at_one_l479_479809

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := x^3 - c * x^2 + 3

theorem find_derivative_at_one (c : ℝ) (h : ∀ x : ℝ, Derivative (f c) x = 3 * x^2 - 2 * f (c) 1 * x)
  : f c 1 = 1 := sorry

end find_derivative_at_one_l479_479809


namespace eccentricity_is_sqrt_5_l479_479087

-- Define the hyperbola and related properties
variables {a b : ℝ} (ha : a > 0) (hb : b > 0)

-- Define the hyperbola equation and asymptote condition
def hyperbola_eq : Prop := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
def asymptote_cond : Prop := ∀ x y : ℝ, 2 * x = y

-- Define the eccentricity of the hyperbola based on given conditions
def eccentricity_of_hyperbola (a b c e : ℝ) : Prop :=
  b = 2 * a ∧ c = √(a^2 + b^2) ∧ e = c / a

-- Main theorem: Given the conditions, prove the eccentricity is sqrt(5)
theorem eccentricity_is_sqrt_5 {a b c e : ℝ} (ha : a > 0) (hb : b > 0) :
  hyperbola_eq ha hb → asymptote_cond ha hb → eccentricity_of_hyperbola a b c e → e = √5 :=
by
  -- Since we only need to state the theorem without proof, we add 'sorry' to skip the proof.
  sorry

end eccentricity_is_sqrt_5_l479_479087


namespace periodic_f_l479_479490

noncomputable def f : ℝ → ℝ := sorry

variable (a : ℝ) (h_a : a > 0) (h_f : ∀ x : ℝ, f(x + a) = 1 / 2 + sqrt (f x - f x ^ 2))

theorem periodic_f : ∀ x : ℝ, f(x + 2 * a) = f(x) :=
by 
  -- proof
  sorry

end periodic_f_l479_479490


namespace probability_one_tail_given_at_least_one_head_l479_479991

theorem probability_one_tail_given_at_least_one_head :
  (∑ ω in {ω : Finset (Fin 2) // ∑ i, (ite (ω i = 1) 1 0) = 1}, 1) /
  (∑ ω in {ω : Finset (Fin 2) // ∑ i, (ite (ω i = 1) 1 0) > 0}, 1) =
    3 / 7 :=
by sorry

end probability_one_tail_given_at_least_one_head_l479_479991


namespace arrangement_is_correct_l479_479838

-- Definition of adjacency in a 3x3 matrix.
def adjacent (i j k l : Nat) : Prop := 
  (i = k ∧ j = l + 1) ∨ (i = k ∧ j = l - 1) ∨ -- horizontal adjacency
  (i = k + 1 ∧ j = l) ∨ (i = k - 1 ∧ j = l) ∨ -- vertical adjacency
  (i = k + 1 ∧ j = l + 1) ∨ (i = k - 1 ∧ j = l - 1) ∨ -- diagonal adjacency
  (i = k + 1 ∧ j = l - 1) ∨ (i = k - 1 ∧ j = l + 1)   -- diagonal adjacency

-- Definition to check if two numbers share a common divisor greater than 1.
def coprime (x y : Nat) : Prop := Nat.gcd x y = 1

-- The arrangement of the numbers in the 3x3 grid.
def grid := 
  [ [8, 9, 10],
    [5, 7, 11],
    [6, 13, 12] ]

-- Ensure adjacents numbers are coprime
def correctArrangement :=
  (coprime grid[0][0] grid[0][1]) ∧ (coprime grid[0][1] grid[0][2]) ∧
  (coprime grid[1][0] grid[1][1]) ∧ (coprime grid[1][1] grid[1][2]) ∧
  (coprime grid[2][0] grid[2][1]) ∧ (coprime grid[2][1] grid[2][2]) ∧
  (adjacent 0 0 1 1 → coprime grid[0][0] grid[1][1]) ∧
  (adjacent 0 2 1 1 → coprime grid[0][2] grid[1][1]) ∧
  (adjacent 1 0 2 1 → coprime grid[1][0] grid[2][1]) ∧
  (adjacent 1 2 2 1 → coprime grid[1][2] grid[2][1]) ∧
  (coprime grid[0][1] grid[1][0]) ∧ (coprime grid[0][1] grid[1][2]) ∧
  (coprime grid[2][1] grid[1][0]) ∧ (coprime grid[2][1] grid[1][2])

-- Statement to be proven
theorem arrangement_is_correct : correctArrangement := 
  sorry

end arrangement_is_correct_l479_479838


namespace tan_half_sum_of_angles_l479_479731

variables {α β : ℝ}
hypothesis (tan_alpha : Real.tan α = 2)
hypothesis (tan_beta : Real.tan β = 3)
hypothesis (alpha_acute : 0 < α ∧ α < Real.pi / 2)
hypothesis (beta_acute : 0 < β ∧ β < Real.pi / 2)

theorem tan_half_sum_of_angles : Real.tan ((α + β) / 2) = 1 + Real.sqrt 2 :=
by
  sorry

end tan_half_sum_of_angles_l479_479731


namespace quadrilateral_angle_E_approx_l479_479218

theorem quadrilateral_angle_E_approx :
  ∃ (E : ℝ), (E = 3 * (E / 3)) ∧ (E = 4 * (E / 4)) ∧ (E = 6 * (E / 6)) ∧ (E + E / 3 + E / 4 + E / 6 = 360) ∧ (E ≈ 206) :=
sorry

end quadrilateral_angle_E_approx_l479_479218


namespace total_volume_of_ice_cream_l479_479652

def r : ℝ := 3    -- radius of the opening of the cone, hemisphere, and scoop
def hCone : ℝ := 12  -- height of the cone
def hCylinder : ℝ := 2  -- height of the cylindrical scoop

def V_cone : ℝ := (1/3) * π * r^2 * hCone  -- volume of the cone
def V_hemisphere : ℝ := (2/3) * π * r^3  -- volume of the hemisphere
def V_cylinder : ℝ := π * r^2 * hCylinder  -- volume of the cylindrical scoop

def V_total : ℝ := V_cone + V_hemisphere + V_cylinder  -- total volume

theorem total_volume_of_ice_cream : V_total = 72 * π := by
  sorry

end total_volume_of_ice_cream_l479_479652


namespace tv_price_reduction_percentage_l479_479246

noncomputable def price_reduction (x : ℝ) : Prop :=
  (1 - x / 100) * 1.80 = 1.44000000000000014

theorem tv_price_reduction_percentage : price_reduction 20 :=
by
  sorry

end tv_price_reduction_percentage_l479_479246


namespace angle_equality_l479_479589

-- Define the geometric setup including the circles and their intersection
variable {α : Type*} [EuclideanAffineSpace α] -- Assuming Euclidean affine space for geometry

variable (Circle1 Circle2 : Set (Point α))
variable (K A B C D : Point α)
variable {l : Line α}

-- Define the conditions
def conditions : Prop :=
  (Circle1 ∩ Circle2 = {K}) ∧
  (K ∈ Circle1) ∧ (K ∈ Circle2) ∧
  (A ∈ Circle1) ∧ (B ∈ Circle1) ∧
  (C ∈ Circle2) ∧ (D ∈ Circle2) ∧
  (A ∈ l) ∧ (B ∈ l) ∧ (C ∈ l) ∧ (D ∈ l) ∧
  (A ≠ B) ∧ (C ≠ D) ∧
  (l ∩ LineThrough K A = {A}) ∧
  (l ∩ LineThrough K B = {B}) ∧
  (l ∩ LineThrough K C = {C}) ∧
  (l ∩ LineThrough K D = {D}) ∧
  (C ∈ Segment A D)

-- Define the theorem statement
theorem angle_equality (h : conditions Circle1 Circle2 K A B C D l) : ∠ A K C = ∠ B K D := 
by
  sorry -- The proof is omitted

end angle_equality_l479_479589


namespace f_f_2_eq_one_ninth_l479_479411

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x > 0 then x - 4 else 3^x

-- Theorem stating that f(f(2)) = 1/9
theorem f_f_2_eq_one_ninth : f (f 2) = 1 / 9 := by
  sorry

end f_f_2_eq_one_ninth_l479_479411


namespace particle_acceleration_non_zero_and_radial_l479_479554

noncomputable def acceleration_properties (P : Type) [NormedAddCommGroup P] [NormedSpace ℝ P] 
  (O A B : P) (k T : ℝ) (path : ℝ → P) : Prop :=
  (path 0 = A) ∧ (path T = B) ∧ (∀ t, ∥path t - O∥ = k) ∧ (∥gradient path 0∥ = 0) ∧ (∀ t, ∀ (h : 0 < t ∧ t < T), ∥gradient path t∥ ≠ 0) ∧ 
  (∃ t, 0 < t ∧ t < T ∧ gradient (gradient path) t = 0)

theorem particle_acceleration_non_zero_and_radial 
  (P : Type) [NormedAddCommGroup P] [NormedSpace ℝ P] (O A B : P) (k T : ℝ) 
  (path : ℝ → P) (h : acceleration_properties P O A B k T path) : 
  (∀ t, 0 ≤ t ∧ t ≤ T → ∥gradient path t∥ ≠ 0) ∧ 
  (∃ t, 0 < t ∧ t < T ∧ gradient (gradient path) t = 0 ∧ acceleration_properties P O A B k T path) :=
by
  -- Complex proof body omitted
  sorry

end particle_acceleration_non_zero_and_radial_l479_479554


namespace probability_of_team_with_2_girls_2_boys_l479_479987

open Nat

-- Define the combinatorics function for binomial coefficients
def binomial (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_of_team_with_2_girls_2_boys :
  let total_women := 8
  let total_men := 6
  let team_size := 4
  let ways_to_choose_2_girls := binomial total_women 2
  let ways_to_choose_2_boys := binomial total_men 2
  let total_ways_to_form_team := binomial (total_women + total_men) team_size
  let favorable_outcomes := ways_to_choose_2_girls * ways_to_choose_2_boys
  (favorable_outcomes : ℚ) / total_ways_to_form_team = 60 / 143 := 
by sorry

end probability_of_team_with_2_girls_2_boys_l479_479987


namespace find_a_b_l479_479568

theorem find_a_b (a b : ℝ) : 
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - a * x - b < 0) → a = 5 ∧ b = -6 :=
sorry

end find_a_b_l479_479568


namespace range_of_a_l479_479773

def f (a x : ℝ) : ℝ := x^2 + a * x + 3 - a

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ Icc (-2 : ℝ) (2 : ℝ) → f a x ≥ 0) ↔ a ∈ Icc (-7 : ℝ) (2 : ℝ) :=
by
  sorry

end range_of_a_l479_479773


namespace radius_of_given_spherical_circle_l479_479562
noncomputable def circle_radius_spherical_coords : Real :=
  let spherical_to_cartesian (rho theta phi : Real) : (Real × Real × Real) :=
    (rho * (Real.sin phi) * (Real.cos theta), rho * (Real.sin phi) * (Real.sin theta), rho * (Real.cos phi))
  let (rho, theta, phi) := (1, 0, Real.pi / 3)
  let (x, y, z) := spherical_to_cartesian rho theta phi
  let radius := Real.sqrt (x^2 + y^2)
  radius

theorem radius_of_given_spherical_circle :
  circle_radius_spherical_coords = (Real.sqrt 3) / 2 :=
sorry

end radius_of_given_spherical_circle_l479_479562


namespace quadratic_intersects_x_axis_l479_479068

theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersects_x_axis_l479_479068


namespace angle_AED_eq_90_l479_479858

-- Isosceles triangle and parallel lines setting
variables {A B C X Y D E : Type}
variables (h_isosceles : AB = AC)
variables (h_parallel : XY || AB)
variables (h_circumcenter : circumcenter C X Y D)
variables (h_midpoint : midpoint E BY)

-- Prove the angle condition
theorem angle_AED_eq_90 (h_isosceles : AB = AC)
                         (h_parallel : XY || AB)
                         (h_circumcenter : circumcenter C X Y D)
                         (h_midpoint : midpoint E BY) :
  ∠AED = 90 :=
sorry

end angle_AED_eq_90_l479_479858


namespace arithmetic_sequence_sum_l479_479495

-- Definitions based on problem conditions
variable (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ) -- terms of the sequence
variable (S_3 S_6 S_9 : ℤ)

-- Given conditions
variable (h1 : S_3 = 3 * a_1 + 3 * (a_2 - a_1))
variable (h2 : S_6 = 6 * a_1 + 15 * (a_2 - a_1))
variable (h3 : S_3 = 9)
variable (h4 : S_6 = 36)

-- Theorem to prove
theorem arithmetic_sequence_sum : S_9 = 81 :=
by
  -- We just state the theorem here and will provide a proof later
  sorry

end arithmetic_sequence_sum_l479_479495


namespace least_expensive_trip_is_1627_44_l479_479512

noncomputable def least_expensive_trip_cost : ℝ :=
  let distance_DE := 4500
  let distance_DF := 4000
  let distance_EF := Real.sqrt (distance_DE ^ 2 - distance_DF ^ 2)
  let cost_bus (distance : ℝ) : ℝ := distance * 0.20
  let cost_plane (distance : ℝ) : ℝ := distance * 0.12 + 120
  let cost_DE := min (cost_bus distance_DE) (cost_plane distance_DE)
  let cost_EF := min (cost_bus distance_EF) (cost_plane distance_EF)
  let cost_DF := min (cost_bus distance_DF) (cost_plane distance_DF)
  cost_DE + cost_EF + cost_DF

theorem least_expensive_trip_is_1627_44 :
  least_expensive_trip_cost = 1627.44 := sorry

end least_expensive_trip_is_1627_44_l479_479512


namespace multiples_of_seven_between_10_and_150_l479_479429

def is_multiple_of (a : ℕ) (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * a

def count_multiples_in_range (a n m : ℕ) : ℕ :=
  let start := n / a + 1
  let end := m / a
  end - start + 1

theorem multiples_of_seven_between_10_and_150 : count_multiples_in_range 7 10 150 = 20 :=
by
  sorry

end multiples_of_seven_between_10_and_150_l479_479429


namespace grid_is_valid_l479_479834

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Check if all adjacent cells (by side or diagonal) in a 3x3 grid are coprime -/
def valid_grid (grid : Array (Array ℕ)) : Prop :=
  let adjacent_pairs := 
    [(0, 0, 0, 1), (0, 1, 0, 2), 
     (1, 0, 1, 1), (1, 1, 1, 2), 
     (2, 0, 2, 1), (2, 1, 2, 2),
     (0, 0, 1, 0), (0, 1, 1, 1), (0, 2, 1, 2),
     (1, 0, 2, 0), (1, 1, 2, 1), (1, 2, 2, 2),
     (0, 0, 1, 1), (0, 1, 1, 2), 
     (1, 0, 2, 1), (1, 1, 2, 2), 
     (0, 2, 1, 1), (1, 2, 2, 1), 
     (0, 1, 1, 0), (1, 1, 2, 0)] 
  adjacent_pairs.all (λ ⟨r1, c1, r2, c2⟩, is_coprime (grid[r1][c1]) (grid[r2][c2]))

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10],
    #[5, 7, 11],
    #[6, 13, 12]]

theorem grid_is_valid : valid_grid grid :=
  by
    -- Proof omitted, placeholder
    sorry

end grid_is_valid_l479_479834


namespace axis_of_symmetry_l479_479939

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (4 * x + (Real.pi / 6))

-- Define the transformation
def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x - (Real.pi / 6))

theorem axis_of_symmetry : ∃ k : ℤ, ∃ x : ℝ, g x = g (-x + (↑k * Real.pi / 2 + Real.pi / 3)) :=
by
  -- We state the existence of an axis of symmetry using the given transformation.
  sorry

end axis_of_symmetry_l479_479939


namespace complement_A_is_1_and_5_l479_479091

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set A
def A : Set ℤ := {x | abs (x - 3) < 2}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := U \ {n ∈ U | (n : ℤ) ∈ A}

-- Prove the statement
theorem complement_A_is_1_and_5 : complement_U_A = {1, 5} := 
by
  sorry

end complement_A_is_1_and_5_l479_479091


namespace existence_of_circle_set_l479_479514

def isRational (x : ℚ) (y : ℚ) : Prop :=
  ∃ (p q : ℤ), q > 0 ∧ gcd p q = 1 ∧ x = p / q ∧ y = 0

def circle_eq (p q : ℤ) : (ℚ × ℚ) → Prop := λ (x, y),
  (x - (p : ℚ) / q)^2 + (y - (1 : ℚ) / (2 * q^2))^2 = (1 : ℚ) / (4 * q^4)

def property_one (C : set ((ℚ × ℚ) → Prop)) : Prop :=
  ∀ (r : ℚ), isRational r 0 -> (∃ circle ∈ C, ∃(x, y : ℚ), (x = r ∧ circle (x, y)))

def property_two (C : set ((ℚ × ℚ) → Prop)) : Prop :=
  ∀ (c1 c2 ∈ C), c1 ≠ c2 → ∃! (x, y : ℚ), c1 (x, y) ∧ c2 (x, y)

theorem existence_of_circle_set : 
  ∃ C : set ((ℚ × ℚ) → Prop), property_one C ∧ property_two C :=
sorry

end existence_of_circle_set_l479_479514


namespace min_distance_trajectory_to_plane_zero_l479_479857

/-- In a three-dimensional space, we have three mutually perpendicular planes: α, β, and r. 
1. Let there be a point A on plane α.
2. Point A is at a distance of 1 from both planes β and r.
3. Let P be a variable point on plane α such that the distance from P to plane β is √2 times 
the distance from P to point A. 
Given these conditions, prove that the minimum distance from points on the trajectory of P to plane r is 0. -/

theorem min_distance_trajectory_to_plane_zero
    (α β r : Plane)
    (A : Point)
    (P : Point → Prop)
    (h1 : A ∈ α)
    (h2 : distance A β = 1)
    (h3 : distance A r = 1)
    (h4 : ∀ P, P ∈ α → distance P β = sqrt 2 * distance P A) :
    ∃ P, P ∈ α ∧ distance P r = 0 := 
sorry

end min_distance_trajectory_to_plane_zero_l479_479857


namespace grid_satisfies_conditions_l479_479843

-- Define the range of consecutive integers
def consecutive_integers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Define a function to check mutual co-primality condition for two given numbers
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the 3x3 grid arrangement as a matrix
@[reducible]
def grid : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![8, 9, 10],
    ![5, 7, 11],
    ![6, 13, 12]]

-- Define a predicate to check if the grid cells are mutually coprime for adjacent elements
def mutually_coprime_adjacent (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
    ∀ (i j k l : Fin 3), 
    (|i - k| + |j - l| = 1 ∨ |i - k| = |j - l| ∧ |i - k| = 1) → coprime (m i j) (m k l)

-- The main theorem statement
theorem grid_satisfies_conditions : 
  (∀ (i j : Fin 3), grid i j ∈ consecutive_integers) ∧ 
  mutually_coprime_adjacent grid :=
by
  sorry

end grid_satisfies_conditions_l479_479843


namespace range_of_a_l479_479121

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) → -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l479_479121


namespace MapleLeafHigh_points_l479_479824

def MapleLeafHigh (x y : ℕ) : Prop :=
  (1/3 * x + 3/8 * x + 18 + y = x) ∧ (10 ≤ y) ∧ (y ≤ 30)

theorem MapleLeafHigh_points : ∃ y, MapleLeafHigh 104 y ∧ y = 21 := 
by
  use 21
  sorry

end MapleLeafHigh_points_l479_479824


namespace katya_female_classmates_l479_479164

theorem katya_female_classmates (g b : ℕ) (h1 : b = 2 * g) (h2 : b = g + 7) :
  g - 1 = 6 :=
by
  sorry

end katya_female_classmates_l479_479164


namespace maximize_A_l479_479371

noncomputable def A (n : ℕ) : ℝ := (20^n + 11^n) / nat.factorial n

theorem maximize_A : ∃ n : ℕ, (∀ m : ℕ, A n ≥ A m) ∧ n = 19 :=
by
  sorry

end maximize_A_l479_479371


namespace period_ending_time_l479_479041

theorem period_ending_time (start_time : ℕ) (rain_duration : ℕ) (no_rain_duration : ℕ) (end_time : ℕ) :
  start_time = 8 ∧ rain_duration = 4 ∧ no_rain_duration = 5 ∧ end_time = 8 + rain_duration + no_rain_duration
  → end_time = 17 :=
by
  sorry

end period_ending_time_l479_479041


namespace min_distance_ellipse_to_line_l479_479965

theorem min_distance_ellipse_to_line :
  let ellipse (x y : ℝ) := x^2 / 16 + y^2 / 12 = 1
  let line (x y : ℝ) := x - 2 * y - 12 = 0
  ∃ d : ℝ, (∀ x y : ℝ, ellipse x y → (∃ θ : ℝ, x = 4 * Real.cos θ ∧ y = 2 * Real.sqrt 3 * Real.sin θ) → 
  d = | 4 * Real.cos θ - 4 * Real.sqrt 3 * Real.sin θ - 12 | / Real.sqrt 5) ∧ 
  (∀ θ : ℝ, d = 4 * Real.sqrt 5 / 5 * | 2 * Real.cos (θ + π / 3) - 3 |) ∧ 
  (∃ θ : ℝ, 2 * Real.cos (θ + π / 3) = 1)
  → d = 4 * Real.sqrt 5 / 5 :=
by
  sorry

end min_distance_ellipse_to_line_l479_479965


namespace minimum_value_proof_l479_479178

noncomputable def minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem minimum_value_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  minimum_value x y z hx hy hz h ≥ 4 :=
sorry

end minimum_value_proof_l479_479178


namespace product_sum_of_g3_eq_zero_l479_479899

def g (x : ℝ) : ℝ := sorry

-- Premise: for all x, y in ℝ, g(xg(y) - x) = xy - g(x)
axiom g_eq : ∀ x y : ℝ, g (x * g y - x) = x * y - g x

theorem product_sum_of_g3_eq_zero :
  let distinct_vals := {g_3 : ℝ | ∃ g : ℝ → ℝ, (∀ x y : ℝ, g (x * g y - x) = x * y - g x) ∧ (g 3 = g_3)},
      num_vals := distinct_vals.to_finset.size,
      sum_vals := distinct_vals.sum id in
  num_vals * sum_vals = 0 :=
by
  sorry

end product_sum_of_g3_eq_zero_l479_479899


namespace find_ellipse_and_fixed_point_l479_479051

variable {x y a b c x₀ y₀ k : ℝ}
variable {F1 F2 : ℝ × ℝ}
variable {O : ℝ × ℝ} := (0, 0)
variable {S : ℝ × ℝ} := (0, -1/3)
variable {P : ℝ × ℝ}

def is_foci (F1 F2 : ℝ × ℝ) (a : ℝ) (c : ℝ) : Prop :=
  F1 = (-c, 0) ∧ F2 = (c, 0)

def on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity (a c : ℝ) : Prop :=
  c / a = (Real.sqrt 2) / 2

def distance_from_origin (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 7/4

def dot_product_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  ((F1.1 - P.1), (F1.2 - P.2)) ∘ ((F2.1 - P.1), (F2.2 - P.2)) = 3 / 4

def line_through_Slope (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 1/3

def intersection_points (a b k : ℝ) (x1 x2 y1 y2 : ℝ) : Prop :=
  on_ellipse a b x1 y1 ∧ on_ellipse a b x2 y2 ∧ line_through_Slope k x1 y1 ∧ line_through_Slope k x2 y2

theorem find_ellipse_and_fixed_point (a b c k x₀ y₀ x₁ x₂ y₁ y₂ : ℝ) (F1 F2 : ℝ × ℝ) : 
  a > b → b > 0 → 
  eccentricity a c →
  distance_from_origin (x₀, y₀) → 
  dot_product_condition (x₀, y₀) F1 F2 → 
  is_foci F1 F2 a c →
  intersection_points a b k x₁ x₂ y₁ y₂ →
  on_ellipse a b x₀ y₀ →
  a^2 = 2 ∧ b^2 = 1 →
  (∃ m : ℝ, (m = 1) ∧ 
  ∀ (k : ℝ), let x₁ := (-1/3 - m + k * x₁), x₂ := (-1/3 - m + k * x₂) in 
    let y₁ := k * x₁ - 1/3 - m, y₂ := k * x₂ - 1/3 - m in 
    ((x₁ * x₂ + y₁ * y₂ - m * (y₁ + y₂) + m^2) = 0)) :=
begin
  sorry
end

end find_ellipse_and_fixed_point_l479_479051


namespace grandmaster_plays_21_games_l479_479641

theorem grandmaster_plays_21_games (a : ℕ → ℕ) (n : ℕ) :
  (∀ i, 1 ≤ a (i + 1) - a i) ∧ (∀ i, a (i + 7) - a i ≤ 10) →
  ∃ (i j : ℕ), i < j ∧ (a j - a i = 21) :=
sorry

end grandmaster_plays_21_games_l479_479641


namespace exist_distinct_a_b_c_d_l479_479976

noncomputable def A : set ℕ := sorry    -- Definition for the set A

axiom in_every_block_of_100 (k : ℕ) : ∃ n ∈ A, k*100 + 1 ≤ n ∧ n ≤ (k+1)*100

theorem exist_distinct_a_b_c_d (ha : ∀ k, ∃ n ∈ A, k*100 + 1 ≤ n ∧ n ≤ (k+1)*100) :
  ∃ a b c d ∈ A, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d :=
sorry

end exist_distinct_a_b_c_d_l479_479976


namespace food_per_ant_l479_479919

open Real

theorem food_per_ant (A : ℕ) (C_ounce C_job C_leaf : ℝ) (L J : ℕ)
  (hA : A = 400)
  (hC_ounce : C_ounce = 0.1)
  (hC_job : C_job = 5)
  (hC_leaf : C_leaf = 0.01)
  (hL : L = 6000)
  (hJ : J = 4) :
  800 / A = 2 :=
by
  let T_J := J * C_job
  let T_L := L * C_leaf
  let T := T_J + T_L
  have hT_J : T_J = 20 := by sorry
  have hT_L : T_L = 60 := by sorry
  have hT : T = 80 := by sorry
  have hOunces : T / C_ounce = 800 := by sorry
  show 800 / A = 2, by
    rw [hA]
    exact (div_eq_iff (ne_of_gt (show 400 > 0 by norm_num))).mpr rfl

end food_per_ant_l479_479919


namespace range_of_m_l479_479393

-- Define the sets P and S as described in the problem statement
def P : Set ℝ := {x | x^2 - 8 * x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Define the necessary condition
def nec_cond (x : ℝ) : Prop := x ∈ (Pᶜ)

-- Define the necessary but not sufficient condition
def nec_but_not_suff_cond (x : ℝ) : Prop := nec_cond x → x ∈ (Sᶜ m)

theorem range_of_m (m : ℝ) : (∀ x : ℝ, nec_but_not_suff_cond x) → m ∈ Set.Ici 9 :=
by
  sorry

end range_of_m_l479_479393


namespace complement_of_A_in_U_l479_479423

open Set

variable {α : Type*}

def U : Set ℝ := {x | x ≥ 0}
def A : Set ℝ := {x | x ≥ 1}
def complement_U_A : Set ℝ := U \ A

theorem complement_of_A_in_U :
  complement_U_A = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end complement_of_A_in_U_l479_479423


namespace number_is_7612_l479_479573

-- Definitions of the conditions
def digits_correct_wrong_positions (guess : Nat) (num_correct : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10, 
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → 
      (num_correct = 2 ∧
      (digits_placed 0 ≠ (guess / 1000) % 10 ∧ 
      digits_placed 1 ≠ (guess / 100) % 10 ∧ 
      digits_placed 2 ≠ (guess / 10) % 10 ∧ 
      digits_placed 3 ≠ guess % 10)))

def digits_correct_positions (guess : Nat) (num_correct : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10,
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → 
      (num_correct = 2 ∧
      (digits_placed 0 = (guess / 1000) % 10 ∨ 
      digits_placed 1 = (guess / 100) % 10 ∨ 
      digits_placed 2 = (guess / 10) % 10 ∨ 
      digits_placed 3 = guess % 10)))

def digits_not_correct (guess : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10,
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → False)

-- The main theorem to prove
theorem number_is_7612 :
  digits_correct_wrong_positions 8765 2 ∧
  digits_correct_wrong_positions 1023 2 ∧
  digits_correct_positions 8642 2 ∧
  digits_not_correct 5430 →
  ∃ (num : Nat), 
    (num / 1000) % 10 = 7 ∧
    (num / 100) % 10 = 6 ∧
    (num / 10) % 10 = 1 ∧
    num % 10 = 2 ∧
    num = 7612 :=
sorry

end number_is_7612_l479_479573


namespace rate_of_water_flow_l479_479930

-- Define the initial conditions and the final state
variables (initial_water final_water time : ℝ)

-- State the conditions in Lean
def initial_conditions : Prop := initial_water = 100 ∧ final_water = 280 ∧ time = 90

-- Prove the rate of water flow
theorem rate_of_water_flow (h : initial_conditions) : (final_water - initial_water) / time = 2 :=
by
  cases h with hi hf,
  cases hf with hf ht,
  rw [hi, hf, ht],
  simp,
  norm_num,
  sorry

end rate_of_water_flow_l479_479930


namespace trigonometric_expression_evaluation_l479_479100

theorem trigonometric_expression_evaluation (θ : ℝ) 
  (h : (cot θ ^ 2000 + 2) / (sin θ + 1) = 1) : 
  (sin θ + 2) ^ 2 * (cos θ + 1) = 9 := 
by
  sorry

end trigonometric_expression_evaluation_l479_479100


namespace newYorkTimeCorrect_l479_479980

-- Definitions of the conditions
def timeDifferenceNYBeijing : ℤ := -13
def timeBeijing : (ℕ × ℕ × ℕ) := (2023, 10, 11)  -- (year, month, day, hour)
def timeInBeijingHour : ℕ := 8

-- Function to subtract hours from a given time, considering day changes
noncomputable def subtractHours (date : (ℕ × ℕ × ℕ)) (hours : ℕ) (sub : ℤ) : (ℕ × ℕ × ℕ) × ℕ :=
  match sub + (-(hours : ℤ)) with
  | h if h >= 0 => (date, h.to_nat)
  | h => ((date.fst, date.snd, date.third - 1), ((24 : ℤ) + h).to_nat)  -- Simplistic day decrease, not generic

theorem newYorkTimeCorrect :
  let ny_time := subtractHours timeBeijing timeInBeijingHour timeDifferenceNYBeijing in
  ny_time = ((2023, 10, 10), 7) :=
by
  sorry

end newYorkTimeCorrect_l479_479980


namespace third_median_is_7_l479_479465

noncomputable def triangle_third_median_length (a b : ℝ) (area : ℝ) : ℝ :=
  -- We define the required length of the third median with given medians a and b, and area
  -- as per the given conditions.
  if H : a = 4 ∧ b = 8 ∧ area = 4 * real.sqrt 15 then 7 else sorry

-- Here we are proving the third median length in triangle DEF under given conditions is 7.
theorem third_median_is_7 :
  ∀ (a b area : ℝ), a = 4 → b = 8 → area = 4 * real.sqrt 15 → triangle_third_median_length a b area = 7 :=
by
  intros a b area ha hb harea
  unfold triangle_third_median_length
  rw [if_pos (and.intro ha (and.intro hb harea))]
  refl

end third_median_is_7_l479_479465


namespace parabola_hyperbola_eqn_l479_479081

-- Given conditions
variables (a b x y : ℝ)
def parabola_vertex_origin : Prop := 
  ∃ p > 0, y^2 = 2 * p * x

def hyperbola_focus_directrix : Prop := 
  y^2 = 4 * x ∧ 4 * x^2 - 4 * y^2 / 3 = 1

def parabola_hyperbola_intersect (p : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ (x = 3/2 ∧ y = sqrt 6)

-- Prove the equations of the parabola and hyperbola
theorem parabola_hyperbola_eqn (a b : ℝ) : 
  parabola_vertex_origin a y x →
  hyperbola_focus_directrix a b x y →
  parabola_hyperbola_intersect p x y →
  y^2 = 4 * x ∧ 4 * x^2 - (4 / 3) * y^2 = 1 :=
by
  intros h h1 h2
  sorry

end parabola_hyperbola_eqn_l479_479081


namespace grid_satisfies_conditions_l479_479842

-- Define the range of consecutive integers
def consecutive_integers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Define a function to check mutual co-primality condition for two given numbers
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the 3x3 grid arrangement as a matrix
@[reducible]
def grid : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![8, 9, 10],
    ![5, 7, 11],
    ![6, 13, 12]]

-- Define a predicate to check if the grid cells are mutually coprime for adjacent elements
def mutually_coprime_adjacent (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
    ∀ (i j k l : Fin 3), 
    (|i - k| + |j - l| = 1 ∨ |i - k| = |j - l| ∧ |i - k| = 1) → coprime (m i j) (m k l)

-- The main theorem statement
theorem grid_satisfies_conditions : 
  (∀ (i j : Fin 3), grid i j ∈ consecutive_integers) ∧ 
  mutually_coprime_adjacent grid :=
by
  sorry

end grid_satisfies_conditions_l479_479842


namespace arrangement_is_correct_l479_479837

-- Definition of adjacency in a 3x3 matrix.
def adjacent (i j k l : Nat) : Prop := 
  (i = k ∧ j = l + 1) ∨ (i = k ∧ j = l - 1) ∨ -- horizontal adjacency
  (i = k + 1 ∧ j = l) ∨ (i = k - 1 ∧ j = l) ∨ -- vertical adjacency
  (i = k + 1 ∧ j = l + 1) ∨ (i = k - 1 ∧ j = l - 1) ∨ -- diagonal adjacency
  (i = k + 1 ∧ j = l - 1) ∨ (i = k - 1 ∧ j = l + 1)   -- diagonal adjacency

-- Definition to check if two numbers share a common divisor greater than 1.
def coprime (x y : Nat) : Prop := Nat.gcd x y = 1

-- The arrangement of the numbers in the 3x3 grid.
def grid := 
  [ [8, 9, 10],
    [5, 7, 11],
    [6, 13, 12] ]

-- Ensure adjacents numbers are coprime
def correctArrangement :=
  (coprime grid[0][0] grid[0][1]) ∧ (coprime grid[0][1] grid[0][2]) ∧
  (coprime grid[1][0] grid[1][1]) ∧ (coprime grid[1][1] grid[1][2]) ∧
  (coprime grid[2][0] grid[2][1]) ∧ (coprime grid[2][1] grid[2][2]) ∧
  (adjacent 0 0 1 1 → coprime grid[0][0] grid[1][1]) ∧
  (adjacent 0 2 1 1 → coprime grid[0][2] grid[1][1]) ∧
  (adjacent 1 0 2 1 → coprime grid[1][0] grid[2][1]) ∧
  (adjacent 1 2 2 1 → coprime grid[1][2] grid[2][1]) ∧
  (coprime grid[0][1] grid[1][0]) ∧ (coprime grid[0][1] grid[1][2]) ∧
  (coprime grid[2][1] grid[1][0]) ∧ (coprime grid[2][1] grid[1][2])

-- Statement to be proven
theorem arrangement_is_correct : correctArrangement := 
  sorry

end arrangement_is_correct_l479_479837


namespace total_blocks_l479_479880

-- Define lengths of the fort
def length (l : Nat) := l = 15
def width  (w : Nat) := w = 12
def height (h : Nat) := h = 6

-- Deducting the wall thickness of 1 ft on each side
def inner_length (l : Nat) (i_l : Nat) := inner_length = l - 2
def inner_width  (w : Nat) (i_w : Nat) := inner_width = w - 2
def inner_height (h : Nat) (i_h : Nat) := inner_height = h - 2

-- The volume of the fort
def volume (l w h V : Nat) := V = l * w * h

-- The volume of the interior space
def inner_volume (i_l i_w i_h V_int : Nat) := V_int = i_l * i_w * i_h

-- Total number of blocks used
theorem total_blocks : ∀ l w h i_l i_w i_h V V_int,
  length l → width w → height h →
  inner_length l i_l → inner_width w i_w → inner_height h i_h →
  volume l w h V →
  inner_volume i_l i_w i_h V_int →
  (V - V_int = 560) :=
by
  sorry

end total_blocks_l479_479880


namespace pipes_height_l479_479038

theorem pipes_height (d : ℝ) (h : ℝ) (r : ℝ) (s : ℝ)
  (hd : d = 12)
  (hs : s = d)
  (hr : r = d / 2)
  (heq : h = 6 * Real.sqrt 3 + r) :
  h = 6 * Real.sqrt 3 + 6 :=
by
  sorry

end pipes_height_l479_479038


namespace cubic_sum_eq_product_l479_479583

variable {R : Type*} [CommRing R] 
variable {x1 x2 x3 p q : R}

theorem cubic_sum_eq_product (h1 : x1^3 + x1 * p + q = 0) 
                             (h2 : x2^3 + x2 * p + q = 0) 
                             (h3 : x3^3 + x3 * p + q = 0) 
                             (hx12 : x1 ≠ x2) (hx13 : x1 ≠ x3) (hx23 : x2 ≠ x3) :
                             x1^3 + x2^3 + x3^3 = 3 * x1 * x2 * x3 := 
begin
  sorry
end

end cubic_sum_eq_product_l479_479583


namespace quadratic_intersection_l479_479076

theorem quadratic_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersection_l479_479076


namespace slopes_of_reflected_light_l479_479150

noncomputable theory

open Real

-- Definitions pulled from the problem statement
def circle : set (ℝ × ℝ) := { p | (p.1 - 3)^2 + (p.2 + 2)^2 = 1 }
def point_A := (2 : ℝ, 3 : ℝ)
def point_B := (-2 : ℝ, 3 : ℝ)
def is_slope_of_tangent (k : ℝ) := ∃ (A B : ℝ), (∃ (x : ℝ), A = k * x + B) ∧ circle ⟨A, B⟩

-- The theorem to be proved
theorem slopes_of_reflected_light : 
    is_slope_of_tangent (-4/3) ∨ is_slope_of_tangent (-3/4) := sorry

end slopes_of_reflected_light_l479_479150


namespace persons_in_first_group_is_18_l479_479118

noncomputable def num_persons_first_group (x : ℕ) := 
  ∃ y : ℕ, y = 42 * x / 140 ∧ y * 10 = 30 * 6

theorem persons_in_first_group_is_18 : num_persons_first_group 18 :=
by 
  existsi 18
  have h1 : 42 * 18 / 140 = 5.4 by sorry
  have h2 : 5.4 * 10 = 30 * 6 by sorry
  exact ⟨h1, h2⟩

end persons_in_first_group_is_18_l479_479118


namespace fraction_of_b_eq_three_tenths_a_l479_479620

theorem fraction_of_b_eq_three_tenths_a (a b : ℝ) (h1 : a + b = 100) (h2 : b = 60) :
  (3 / 10) * a = (1 / 5) * b :=
by 
  have h3 : a = 40 := by linarith [h1, h2]
  rw [h2, h3]
  linarith

end fraction_of_b_eq_three_tenths_a_l479_479620


namespace product_of_fractions_l479_479333

theorem product_of_fractions :
  (1 / 2) * (2 / 3) * (3 / 4) * (3 / 2) = 3 / 8 := by
  sorry

end product_of_fractions_l479_479333


namespace t_shirts_per_package_l479_479617

theorem t_shirts_per_package (total_tshirts : ℕ) (num_packages : ℕ) (h1 : total_tshirts = 426) (h2 : num_packages = 71) :
    total_tshirts / num_packages = 6 :=
by
  rw [h1, h2]
  norm_num
  done

end t_shirts_per_package_l479_479617


namespace initial_workers_45_l479_479525

variable (W : ℕ)  -- Define the initial number of workers as a natural number
variable (R : ℝ)  -- Define the work rate as a real number
variable (work_done : ℕ → ℝ → ℝ → ℝ)  -- Define work done as a function of workers, time, and rate 

-- Define the conditions given in the problem
def condition1 := work_done W 8 R = 30
def condition2 := work_done (W + 15) 6 R = 30

-- Define the main theorem to be proved
theorem initial_workers_45 (h1 : condition1) (h2 : condition2) : W = 45 :=
by
  sorry

end initial_workers_45_l479_479525


namespace B_2023_coordinates_l479_479871

def is_square (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  (angle A B C = π/2) ∧ (angle B C D = π/2) ∧ (angle C D A = π/2) ∧ (angle D A B = π/2)

def A_line (n : ℕ) : Point :=
  ⟨n, n + 1⟩

def C_line (n : ℕ) : Point :=
  ⟨n, 0⟩

theorem B_2023_coordinates :
  ∃ B : Point, B.x = 2^2023 - 1 := 
sorry

end B_2023_coordinates_l479_479871


namespace distance_probability_l479_479531

noncomputable def probability_distance_under_8000 : ℚ := 4 / 5

variables (Bangkok_CapeTown Bangkok_Honolulu Bangkok_London Bangkok_Tokyo CapeTown_Honolulu CapeTown_London CapeTown_Tokyo Honolulu_London Honolulu_Tokyo London_Tokyo : ℕ)

axiom chart_distances : 
  Bangkok_CapeTown = 6300 ∧
  Bangkok_Honolulu = 6609 ∧
  Bangkok_London = 5944 ∧
  Bangkok_Tokyo = 5200 ∧
  CapeTown_Honolulu = 11535 ∧
  CapeTown_London = 5989 ∧
  CapeTown_Tokyo = 9500 ∧
  Honolulu_London = 7240 ∧
  Honolulu_Tokyo = 4100 ∧
  London_Tokyo = 6000

theorem distance_probability : 
  let total_pairs := 10 in
  let pairs_under_8000 := 8 in
  pairs_under_8000 / total_pairs = probability_distance_under_8000 :=
by
  -- Proof omitted
  sorry

end distance_probability_l479_479531


namespace number_of_digits_in_2_15_mul_3_2_mul_5_10_l479_479694

theorem number_of_digits_in_2_15_mul_3_2_mul_5_10 : 
  ∃ (d : ℕ), d = 13 ∧ (nat.digits 10 (2^15 * 3^2 * 5^10)).length = d :=
by
  use 13
  split
  <|> {
    sorry -- proof not needed
  }

end number_of_digits_in_2_15_mul_3_2_mul_5_10_l479_479694


namespace log_equation_solution_l479_479109

theorem log_equation_solution
  (b x : ℝ)
  (h1 : b > 0)
  (h2 : b ≠ 1)
  (h3 : x ≠ 1)
  (h4 : log (x) / log (b^3) + log (b) / log (x^3) = 1) :
  x = b^( (3 + Real.sqrt 5) / 2) ∨ x = b^((3 - Real.sqrt 5) / 2) :=
sorry

end log_equation_solution_l479_479109


namespace AngleBPCIsRight_l479_479239

-- Define the basic setup of the triangle and inscribed circle
variables {A B C M N P : Point}
variables [inscribed_circle : InscribedCircle (Triangle.mk A B C) M N]
variables [angle_bisector_intersection : AngleBisectorIntersection (Line.mk M N) (Angle.mk B A C) P]

-- Define the theorem to prove
theorem AngleBPCIsRight :
  angle_eq (Angle.mk B P C) (90 : Degree) :=
sorry

end AngleBPCIsRight_l479_479239


namespace smallest_positive_multiple_of_45_l479_479604

theorem smallest_positive_multiple_of_45 : ∃ (n : ℕ), n > 0 ∧ ∃ (x : ℕ), x > 0 ∧ n = 45 * x ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l479_479604


namespace max_blue_cells_n2_l479_479222

theorem max_blue_cells_n2 :
  ∀ (table : ℕ) (n colors : ℕ), 
  table = 40 → n = 2 → 
  (∀ i j, (∃ x, 0 ≤ x ∧ x < n ∧ ∀ k, k < table → (table (i, k) = x ∨ table (k, j) = x)) → 
  (table.blue_cells ≤ 800)) := 
sorry

end max_blue_cells_n2_l479_479222


namespace problem_1_problem_2_problem_3_l479_479367

noncomputable def sec (x : ℝ) := 1 / cos x
noncomputable def tan (x : ℝ) := sin x / cos x

theorem problem_1 : sec (50 * real.pi / 180) + tan (10 * real.pi / 180) = sqrt 3 := 
sorry

theorem problem_2 : cos (2 * real.pi / 7) + cos (4 * real.pi / 7) + cos (6 * real.pi / 7) = -1 / 2 := 
sorry

theorem problem_3 : tan (6 * real.pi / 180) * tan (42 * real.pi / 180) * tan (66 * real.pi / 180) * tan (78 * real.pi / 180) = 1 := 
sorry

end problem_1_problem_2_problem_3_l479_479367


namespace max_null_entries_l479_479383
-- Import necessary libraries

-- Define the problem conditions
variable (A : Matrix (Fin 3) (Fin 3) ℝ)
variable (A_symm : A.isSymmetric)
variable (A_no_zero_entries : ∀ i j, A i j ≠ 0)

-- Define f(A) following the problem's definition
noncomputable def f (A : Matrix (Fin 3) (Fin 3) ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  let eigenvalues := A.eigenvalues -- Assuming a function to get eigenvalues
  Matrix.diag (Fin 3) (λ i, eigenvalues.sum - eigenvalues i)

-- Define the sequence Aₙ
noncomputable def sequence (n : ℕ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Nat.recOn n A (λ n A_n, f A_n)

-- Define the main theorem to prove
theorem max_null_entries : (∃ (n1 n2 : ℕ), n1 ≠ n2 ∧ n1 ≠ n2 + 2 ∧ n2 ≠ n1 + 2 ∧
  (∀ i j, sequence A i j = 0) ∧ (∀ i j, sequence A i j = 1)) → False := 
sorry

end max_null_entries_l479_479383


namespace circle_circumference_l479_479023

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Prove that circumference of the circle is 2 * sqrt 2 * pi
theorem circle_circumference : 
  (∀ x y : ℝ, circle_eq x y) → (∀ r : ℝ, r = □2) → 2 * pi * r = 2 * (sqrt 2) * pi :=
by sorry

end circle_circumference_l479_479023


namespace distance_between_points_l479_479355

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_between_points :
  distance (2, 3, 1) (5, 9, 6) = real.sqrt 70 := by
  sorry

end distance_between_points_l479_479355


namespace HorseKeepsPower_l479_479120

/-- If the Little Humpbacked Horse does not eat for seven days or does not sleep for seven days,
    he will lose his magic power. Suppose he did not eat or sleep for a whole week. 
    Prove that by the end of the seventh day, he must do the activity he did not do right before 
    the start of the first period of seven days in order to keep his power. -/
theorem HorseKeepsPower (eat sleep : ℕ → Prop) :
  (∀ (n : ℕ), (n ≥ 7 → ¬eat n) ∨ (n ≥ 7 → ¬sleep n)) →
  (∀ (n : ℕ), n < 7 → (¬eat n ∧ ¬sleep n)) →
  ∃ (t : ℕ), t > 7 → (eat t ∨ sleep t) :=
sorry

end HorseKeepsPower_l479_479120


namespace jerrys_age_l479_479187

theorem jerrys_age (M J : ℕ) (h1 : M = 3 * J - 4) (h2 : M = 14) : J = 6 :=
by 
  sorry

end jerrys_age_l479_479187


namespace cube_surface_area_l479_479252

open Real

theorem cube_surface_area (V : ℝ) (a : ℝ) (S : ℝ)
  (h1 : V = a ^ 3)
  (h2 : a = 4)
  (h3 : V = 64) :
  S = 6 * a ^ 2 :=
by
  sorry

end cube_surface_area_l479_479252


namespace m_separable_l479_479513

variables (e : ℕ)

theorem m_separable (C A : set α) (hC : ∀ a b ∈ A, a ≠ b → ∃ (seq : list α), a ∈ seq ∧ b ∈ seq ∧ (∀ i, i < seq.length - 1 → (seq.nth i, seq.nth (i + 1)) ∈ C)) :
  ∃ m : ℕ, m ≤ (1 / 2 + real.sqrt (2 * e + 1 / 4)) :=
begin
  sorry
end

end m_separable_l479_479513


namespace option_A_right_angle_option_B_not_right_angle_option_C_not_right_angle_option_D_not_right_angle_l479_479958

theorem option_A_right_angle (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) :
  a^2 + b^2 = c^2 :=
by {
  rw [h1, h2, h3],
  norm_num
}

theorem option_B_not_right_angle (a b c : ℕ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  a^2 + b^2 ≠ c^2 :=
by {
  rw [h1, h2, h3],
  norm_num
}

theorem option_C_not_right_angle (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) :
  a^2 + b^2 ≠ c^2 :=
by {
  rw [h1, h2, h3],
  norm_num
}

theorem option_D_not_right_angle (a b c : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 6) :
  a^2 + b^2 ≠ c^2 :=
by {
  rw [h1, h2, h3],
  norm_num
}

end option_A_right_angle_option_B_not_right_angle_option_C_not_right_angle_option_D_not_right_angle_l479_479958


namespace fraction_of_female_gives_birth_is_correct_l479_479943

-- Given conditions
def total_dogs : Int := 40
def female_fraction : Rat := 0.6
def puppies_per_dog : Int := 10
def puppies_remaining : Int := 50
def puppies_donated : Int := 130

-- Calculate the number of female dogs
def female_dogs : Int := total_dogs * female_fraction
def total_puppies_before_donation : Int := puppies_remaining + puppies_donated

-- Define the fraction of female dogs that give birth to puppies
def fraction_of_female_gives_birth (F : Rat) : Prop := 
  F * female_dogs * puppies_per_dog = total_puppies_before_donation

theorem fraction_of_female_gives_birth_is_correct : fraction_of_female_gives_birth (3/4) :=
by
  have h1 : female_dogs = 24 := by
    unfold female_dogs
    exact calc
      total_dogs * female_fraction = 40 * 0.6 : by rfl
      ... = 24 : by norm_num
  
  have h2 : total_puppies_before_donation = 180 := by
    unfold total_puppies_before_donation
    exact calc
      puppies_remaining + puppies_donated = 50 + 130 : by rfl
      ... = 180 : by norm_num

  -- Main calculation to show F = 3/4 satisfies the condition
  unfold fraction_of_female_gives_birth
  calc 
    (3/4) * female_dogs * puppies_per_dog
      = (3/4) * 24 * 10 : by rw h1
      ... = 180 : by norm_num
  exact sorry

end fraction_of_female_gives_birth_is_correct_l479_479943


namespace ratio_of_largest_element_to_sum_of_others_l479_479340

theorem ratio_of_largest_element_to_sum_of_others :
  let s := (∑ i in Finset.range 8, 12^i)
  in (12^8 : ℕ) / (s : ℕ) = 11 :=
by
  let s := ∑ i in Finset.range 8, 12^i
  sorry

end ratio_of_largest_element_to_sum_of_others_l479_479340


namespace intervals_of_increase_equilateral_triangle_l479_479770

-- Define the function f
def f (x : Real) : Real :=
  sin (2 * x + π / 3) - cos (2 * x + π / 6) - sqrt 3 * cos (2 * x)

-- Prove the intervals of monotonic increase
theorem intervals_of_increase :
  ∀ k : ℤ, (∀ x, f' x ≥ 0) ↔ (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) := 
sorry

-- Define the conditions for the triangle problem
variables {A B C : Real} (hB : B = π / 3) (AC : Real) (BC : Real) (AB : Real)
  (hp : AC = sqrt 3) (hperim : AB + BC + AC = 3 * sqrt 3)

-- Prove the triangle is equilateral and AB = BC = sqrt 3
theorem equilateral_triangle :
  ∃ AB BC : Real,
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3 ∧
  AB = BC = sqrt 3 ∧
  AB + BC + AC = 3 * sqrt 3 := 
sorry

end intervals_of_increase_equilateral_triangle_l479_479770


namespace problem_solution_l479_479768

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 4 then (1/2) ^ x else sorry

theorem problem_solution : f (2 + real.log 3 / real.log 2) = 1 / 24 :=
by rw [f]; linarith

end problem_solution_l479_479768


namespace worth_of_stuff_l479_479337

theorem worth_of_stuff (x : ℝ)
  (h1 : 1.05 * x - 8 = 34) :
  x = 40 :=
by
  sorry

end worth_of_stuff_l479_479337


namespace infinite_solutions_l479_479724

theorem infinite_solutions (b : ℤ) : 
  (∀ x : ℤ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := 
by sorry

end infinite_solutions_l479_479724


namespace problem_solution_l479_479404

def complex_eq (a b : ℂ) : Prop := 
  a.re = b.re ∧ a.im = b.im

theorem problem_solution :
  let z := (1 - complex.i) ^ 2 - 3 * (1 + complex.i) / (2 - complex.i)
  in ∃ (z' : ℂ) (a b : ℚ),
   z' = -1/5 - 13 / 5 * complex.i ∧ complex.conj(z') = -1/5 + 13 / 5 * complex.i ∧
   complex_eq ((5 / 13 : ℚ) * z' + (14 / 13 : ℚ)) (1 - complex.i) :=
by
  sorry

end problem_solution_l479_479404


namespace number_53_in_sequence_l479_479885

/-- 
Jo and Blair take turns counting from 1. Jo starts by saying 1. 
Blair follows by saying the next number. Every third turn when it's Jo's turn, 
she says a number that is two more than the last number said by Blair. 
What is the 53rd number said?
-/
theorem number_53_in_sequence : 
  let sequence : ℕ → ℕ
  | 0     := 1 -- Jo says 1 on the first turn
  | 1     := 2 -- Blair says 2 on the second turn
  | (n+2) := if (n + 2 + 1) % 6 = 0 then sequence (n + 1) + 2 else sequence (n + 1) + 1
  in sequence 52 = 72 :=
by {
  -- The detailed proof or construction of the sequence is omitted.
  sorry
}

end number_53_in_sequence_l479_479885


namespace white_washing_cost_correct_l479_479626

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

def door_length : ℝ := 6
def door_width : ℝ := 3

def window_length : ℝ := 4
def window_width : ℝ := 3

def cost_per_sq_ft : ℝ := 8

def calculate_white_washing_cost : ℝ :=
  let total_wall_area := 2 * (room_length * room_height) + 2 * (room_width * room_height)
  let door_area := door_length * door_width
  let window_area := 3 * (window_length * window_width)
  let effective_area := total_wall_area - door_area - window_area
  effective_area * cost_per_sq_ft

theorem white_washing_cost_correct : calculate_white_washing_cost = 7248 := by
  sorry

end white_washing_cost_correct_l479_479626


namespace smallest_n_root_of_unity_l479_479271

-- Define the polynomial z^3 - z + 1
def poly : Polynomial ℂ := Polynomial.X ^ 3 - Polynomial.X + 1

-- Define a statement to capture that the smallest positive integer n 
-- such that all roots of poly are n-th roots of unity is 5.
theorem smallest_n_root_of_unity : ∃ n : ℕ, 0 < n ∧ ∀ z : ℂ, z^3 - z + 1 = 0 → z^n = 1 :=
by
  use 5
  sorry

end smallest_n_root_of_unity_l479_479271


namespace keio_university_1977_l479_479710

-- Define a continuous function
noncomputable def is_continuous (f : ℝ → ℝ) : Prop :=
  ∀ x ε > 0, ∃ δ > 0, ∀ y, abs (y - x) < δ → abs (f y - f x) < ε

-- The mathematical statement of the problem 
theorem keio_university_1977 :
  let f := fun x => x in
  is_continuous f →
  (∀ x : ℝ, ∫ t in 0..x, sin t * f (x - t) = f x - sin x) :=
by
  sorry

end keio_university_1977_l479_479710


namespace angle_at_230_is_105_degrees_l479_479597

def degree_between_hands (h m : ℕ) : ℚ :=
  let minute_angle := m * 6 in
  let hour_angle := (h % 12) * 30 + (m / 2) in
  let angle := abs (hour_angle - minute_angle) in
  min angle (360 - angle)

theorem angle_at_230_is_105_degrees :
  degree_between_hands 2 30 = 105 := 
sorry

end angle_at_230_is_105_degrees_l479_479597


namespace amount_paid_to_B_l479_479658

-- Definition of the conditions
def total_payment : ℝ := 529
def fraction_ac_completed : ℝ := 19 / 23
def fraction_b_completed : ℝ := 4 / 23

-- Question translated to Lean statement
theorem amount_paid_to_B : fraction_b_completed * total_payment = 92 := by
  sorry

end amount_paid_to_B_l479_479658


namespace painting_clock_57_painting_clock_1913_l479_479156

-- Part (a)
theorem painting_clock_57 (h : ∀ n : ℕ, (n = 12 ∨ (exists k : ℕ, n = (12 + k * 57) % 12))) :
  ∃ m : ℕ, m = 4 :=
by { sorry }

-- Part (b)
theorem painting_clock_1913 (h : ∀ n : ℕ, (n = 12 ∨ (exists k : ℕ, n = (12 + k * 1913) % 12))) :
  ∃ m : ℕ, m = 12 :=
by { sorry }

end painting_clock_57_painting_clock_1913_l479_479156


namespace ball_hits_ground_time_l479_479234

theorem ball_hits_ground_time :
  ∃ t : ℝ, -20 * t^2 + 30 * t + 60 = 0 ∧ t = (3 + Real.sqrt 57) / 4 :=
by 
  sorry

end ball_hits_ground_time_l479_479234


namespace sum_of_sequence_is_maximal_l479_479080

noncomputable def max_sum_of_first_n_terms (a : ℕ → ℝ) (a_pos : ∀ n, a n > 0) (a_ne_one : ∀ n, a n ≠ 1) :=
let b := λ n, Real.log (a n) in
  ∃ n : ℕ, n = 11 ∧ b 3 = 18 ∧ b 6 = 12 ∧ 
  ∑ i in Finset.range (n + 1), b i = 132

theorem sum_of_sequence_is_maximal {a : ℕ → ℝ} (a_pos : ∀ n, a n > 0) (a_ne_one : ∀ n, a n ≠ 1) :
  let b := λ n, Real.log (a n) in
  b 3 = 18 → b 6 = 12 → 
  ∑ i in Finset.range 11, b i = 132 :=
by
  sorry

end sum_of_sequence_is_maximal_l479_479080


namespace non_collinear_condition_isosceles_right_triangle_condition_l479_479095

-- Vectors definitions
def vector_OA := (3 : ℝ, -4 : ℝ)
def vector_OB := (6 : ℝ, -3 : ℝ)
def vector_OC (x y : ℝ) := (5 - x, -3 - y)

-- Vectors AB and AC
def vector_AB := ((6 : ℝ) - 3, (-3 : ℝ) - (-4))
def vector_AC (x y : ℝ) := ((5 - x) - 3, (-3 - y) - (-4))

theorem non_collinear_condition (x y : ℝ) : 
  3 * y - x ≠ 1 ↔ ¬ (vector_AB.1 * vector_AC x y.2 = vector_AB.2 * vector_AC x y.1) :=
sorry

theorem isosceles_right_triangle_condition (x y : ℝ) : 
  (x = 0 ∧ y = -3) ∨ (x = -2 ∧ y = 3) ↔ 
    (vector_AB.1 * (vector_OB.1 - vector_OC x y.1) + vector_AB.2 * (vector_OB.2 - vector_OC x y.2) = 0)
    ∧ ((vector_AB.1)^2 + (vector_AB.2)^2 = (vector_OB.1 - vector_OC x y.1)^2 + (vector_OB.2 - vector_OC x y.2)^2) :=
sorry

end non_collinear_condition_isosceles_right_triangle_condition_l479_479095


namespace pyramid_four_triangular_faces_area_l479_479677

noncomputable def pyramid_total_area (base_edge lateral_edge : ℝ) : ℝ :=
  if base_edge = 8 ∧ lateral_edge = 7 then 16 * Real.sqrt 33 else 0

theorem pyramid_four_triangular_faces_area :
  pyramid_total_area 8 7 = 16 * Real.sqrt 33 :=
by
  -- Proof omitted
  sorry

end pyramid_four_triangular_faces_area_l479_479677


namespace quad_intersects_x_axis_l479_479071

theorem quad_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x ^ 2 - 7 * x - 7 = 0) ↔ (k ≥ -7 / 4 ∧ k ≠ 0) :=
by sorry

end quad_intersects_x_axis_l479_479071


namespace circle_equation_l479_479360

theorem circle_equation :
  ∃ (r : ℝ) (h : y - 2 = r),
    r = 4 ∧
    (∀ (p : ℝ × ℝ), p ∈ ({ (4, 2), (-4, 2) } : set (ℝ × ℝ)) → (p.1 - 0) ^ 2 + (p.2 - 2) ^ 2 = r ^ 2) ∧
    (∀ (y : ℝ), (y = -4) → (∀ (p : ℝ × ℝ), (p = (0, r)) → (p.2 = 2))) := sorry

end circle_equation_l479_479360


namespace expectation_of_xi_l479_479291

noncomputable def expectation_xi : ℚ :=
  let possible_values := [3, 4, 5]
  let probabilities := [1/10, 3/10, 6/10]
  ∑ i in range(3), (possible_values.nth i).getOrElse 0 * (probabilities.nth i).getOrElse 0

theorem expectation_of_xi :
  let xi := ∑ i in finset.range 3,
              [3, 4, 5].nth i.getOrElse 0 * [1 / 10, 3 / 10, 6 / 10].nth i.getOrElse 0
  xi = 9 / 2 :=
by
  sorry

end expectation_of_xi_l479_479291


namespace correct_option_l479_479740

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
axiom sum_of_first_n_terms {n : ℕ} : S n = ∑ i in Finset.range n, a i
axiom recurrence_relation {b : ℝ} {n : ℕ} : a (n+1) = b * a n

-- Question and correct answer
theorem correct_option (b_eq_zero : b = 0) (a1 : ℝ) :
  ∀ n, S n = a 0 :=
sorry

end correct_option_l479_479740


namespace volume_pyramid_MABCD_l479_479476

theorem volume_pyramid_MABCD (ABCD : Type) (A B C D M : ABCD) 
  (MA MC MB DM : ℝ) 
  (h1 : DM ⟂ plane ABCD)
  (h2 : ∃ d : ℤ, DM = d)
  (b : ℕ) 
  (h3 : b % 2 = 0) 
  (h4 : MA = b ∧ MC = b + 2 ∧ MB = b + 4) :
  volume_pyramid M A B C D = 64 / 3 :=
by sorry

end volume_pyramid_MABCD_l479_479476


namespace sandy_siding_cost_l479_479932

noncomputable def total_cost_of_siding (wall_width wall_height roof_width roof_height siding_width siding_height siding_cost : ℕ) : ℕ :=
  let wall_area := wall_width * wall_height
  let roof_section_area := roof_width * roof_height
  let total_roof_area := 2 * roof_section_area
  let total_area := wall_area + total_roof_area
  let siding_area := siding_width * siding_height
  let number_of_sections := (total_area + siding_area - 1) / siding_area -- equivalent to ceil(total_area / siding_area)
  in number_of_sections * siding_cost

theorem sandy_siding_cost :
  total_cost_of_siding 10 7 10 6 10 15 35 = 70 :=
by
  -- Proof is omitted
  sorry

end sandy_siding_cost_l479_479932


namespace backyard_area_l479_479941

theorem backyard_area {length width : ℝ} 
  (h1 : 30 * length = 1500) 
  (h2 : 12 * (2 * (length + width)) = 1500) : 
  length * width = 625 :=
by
  sorry

end backyard_area_l479_479941


namespace find_a_l479_479613

theorem find_a (a x : ℝ) (h1: a - 2 ≤ x) (h2: x ≤ a + 1) (h3 : -x^2 + 2 * x + 3 = 3) :
  a = 2 := sorry

end find_a_l479_479613


namespace angle_at_230_is_105_degrees_l479_479598

def degree_between_hands (h m : ℕ) : ℚ :=
  let minute_angle := m * 6 in
  let hour_angle := (h % 12) * 30 + (m / 2) in
  let angle := abs (hour_angle - minute_angle) in
  min angle (360 - angle)

theorem angle_at_230_is_105_degrees :
  degree_between_hands 2 30 = 105 := 
sorry

end angle_at_230_is_105_degrees_l479_479598


namespace arrangement_is_correct_l479_479836

-- Definition of adjacency in a 3x3 matrix.
def adjacent (i j k l : Nat) : Prop := 
  (i = k ∧ j = l + 1) ∨ (i = k ∧ j = l - 1) ∨ -- horizontal adjacency
  (i = k + 1 ∧ j = l) ∨ (i = k - 1 ∧ j = l) ∨ -- vertical adjacency
  (i = k + 1 ∧ j = l + 1) ∨ (i = k - 1 ∧ j = l - 1) ∨ -- diagonal adjacency
  (i = k + 1 ∧ j = l - 1) ∨ (i = k - 1 ∧ j = l + 1)   -- diagonal adjacency

-- Definition to check if two numbers share a common divisor greater than 1.
def coprime (x y : Nat) : Prop := Nat.gcd x y = 1

-- The arrangement of the numbers in the 3x3 grid.
def grid := 
  [ [8, 9, 10],
    [5, 7, 11],
    [6, 13, 12] ]

-- Ensure adjacents numbers are coprime
def correctArrangement :=
  (coprime grid[0][0] grid[0][1]) ∧ (coprime grid[0][1] grid[0][2]) ∧
  (coprime grid[1][0] grid[1][1]) ∧ (coprime grid[1][1] grid[1][2]) ∧
  (coprime grid[2][0] grid[2][1]) ∧ (coprime grid[2][1] grid[2][2]) ∧
  (adjacent 0 0 1 1 → coprime grid[0][0] grid[1][1]) ∧
  (adjacent 0 2 1 1 → coprime grid[0][2] grid[1][1]) ∧
  (adjacent 1 0 2 1 → coprime grid[1][0] grid[2][1]) ∧
  (adjacent 1 2 2 1 → coprime grid[1][2] grid[2][1]) ∧
  (coprime grid[0][1] grid[1][0]) ∧ (coprime grid[0][1] grid[1][2]) ∧
  (coprime grid[2][1] grid[1][0]) ∧ (coprime grid[2][1] grid[1][2])

-- Statement to be proven
theorem arrangement_is_correct : correctArrangement := 
  sorry

end arrangement_is_correct_l479_479836


namespace symmetry_axes_intersect_at_centroid_l479_479925

-- Define key components
variable {V : Type} [AddCommGroup V] [Module ℝ V]

structure Polygon (V : Type) [AddCommGroup V] [Module ℝ V] where
  vertices : List V

def centroid (P : Polygon V) : V :=
  let n := (P.vertices.length : ℝ)
  (1 / n • P.vertices.sum)

def is_symmetry_axis (P : Polygon V) (l : AffineSubspace ℝ V) : Prop :=
  ∀ v ∈ P.vertices, l.symmMap v ∈ P.vertices

-- The main theorem
theorem symmetry_axes_intersect_at_centroid (P : Polygon V) (l1 l2 : AffineSubspace ℝ V)
  (h1 : is_symmetry_axis P l1) (h2 : is_symmetry_axis P l2) :
  l1 = l2 := sorry

end symmetry_axes_intersect_at_centroid_l479_479925


namespace num_integer_points_strictly_between_A_B_l479_479642

-- Define points A and B
def A : ℝ × ℝ := (3, 5)
def B : ℝ × ℝ := (100, 405)

-- Define a function to calculate the number of integer coordinate points strictly between A and B on a line
noncomputable def num_integer_points (A B : ℝ × ℝ) : ℕ :=
  let slope := (B.2 - A.2) / (B.1 - A.1) in
  let y_intercept := A.2 - slope * A.1 in
  have h : slope = 400 / 97 := 
    sorry, -- Given slope calculation
  have h' : (∀ x, x ∈ ℤ ∧ 3 < x ∧ x < 100 → 
    ∃ y, y ∈ ℤ ∧ y = slope * x + y_intercept) := 
    sorry, -- Ensuring integer coordinates for points strictly between A and B
  96 -- Based on manual calculation of number of points from 4 to 99

-- Theorem statement
theorem num_integer_points_strictly_between_A_B : 
  num_integer_points A B = 96 :=
sorry

end num_integer_points_strictly_between_A_B_l479_479642


namespace percentage_sum_of_first_three_with_respect_to_last_two_l479_479440

theorem percentage_sum_of_first_three_with_respect_to_last_two 
    (X : ℝ) 
    (n1 : ℝ := 0.075 * X) 
    (n2 : ℝ := 0.135 * X) 
    (n3 : ℝ := 0.21 * X) 
    (n4 : ℝ := 0.295 * X) 
    (n5 : ℝ := 0.35 * X) 
    :
    ((n1 + n2 + n3) / (n4 + n5)) * 100 ≈ 65.12 :=
by
  sorry

end percentage_sum_of_first_three_with_respect_to_last_two_l479_479440


namespace area_triangle_ABC_l479_479458

-- Definitions and conditions
def Trapezoid (A B C D : Type) := 
  AB // CD ∧ 
  (∀ a b c d : ℝ, AB = a ∧ CD = c ∧ c = 3 * a) ∧ 
  (∀ area : ℝ, area = 27)

-- The theorem to prove
theorem area_triangle_ABC 
  {A B C D : Type} 
  [trapezoid: Trapezoid A B C D] :
    ∃ area_ABC : ℝ, area_ABC = 6.75 
by 
  sorry

end area_triangle_ABC_l479_479458


namespace initial_men_count_l479_479581

theorem initial_men_count (M : ℕ) :
  let total_food := M * 22
  let food_after_2_days := total_food - 2 * M
  let remaining_food := 20 * M
  let new_total_men := M + 190
  let required_food_for_16_days := new_total_men * 16
  (remaining_food = required_food_for_16_days) → M = 760 :=
by
  intro h
  sorry

end initial_men_count_l479_479581


namespace probability_not_square_or_cube_l479_479552

-- Define the set of numbers from 1 to 200
def numbers_set : Set ℕ := { n | 1 ≤ n ∧ n ≤ 200 }

-- Define a predicate to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Define a predicate to check if a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

-- Define a predicate to check if a number is a perfect sixth power
def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

-- Define the counting function for perfect squares, cubes, and sixth powers within a range
def count_perfect_squares : ℕ := { n ∈ numbers_set | is_perfect_square n }.toFinset.card
def count_perfect_cubes : ℕ := { n ∈ numbers_set | is_perfect_cube n }.toFinset.card
def count_perfect_sixth_powers : ℕ := { n ∈ numbers_set | is_perfect_sixth_power n }.toFinset.card

-- Define the Lean proof problem statement
theorem probability_not_square_or_cube :
  (200 - (count_perfect_squares + count_perfect_cubes - count_perfect_sixth_powers)) / 200 = 183 / 200 :=
by
  sorry

end probability_not_square_or_cube_l479_479552


namespace triangle_angle_B_l479_479464

noncomputable def R (A B C : ℝ) (a b c : ℝ) : ℝ := a / sin A

theorem triangle_angle_B (A B C a b c : ℝ)
  (h1 : a = 2 * R A B C a b c * sin A)
  (h2 : b = 2 * R A B C a b c * sin B)
  (h3 : sqrt 3 * a = 2 * b * sin A) :
  B = π / 3 ∨ B = 2 * π / 3 :=
begin
  sorry
end

end triangle_angle_B_l479_479464


namespace triangle_with_angle_ratio_is_right_triangle_l479_479445

theorem triangle_with_angle_ratio_is_right_triangle (A B C : Angle) (h : A + B + C = 180) 
  (h_ratio : A = k ∧ B = 2 * k ∧ C = 3 * k) : C = 90 := 
by
  sorry

end triangle_with_angle_ratio_is_right_triangle_l479_479445


namespace solve_for_x_l479_479523

theorem solve_for_x (x : ℝ) : 
  (x - 35) / 3 = (3 * x + 10) / 8 → x = -310 := by
  sorry

end solve_for_x_l479_479523


namespace radius_of_circle_formed_by_spherical_coords_l479_479565

theorem radius_of_circle_formed_by_spherical_coords :
  (∃ θ : ℝ, radius_of_circle (1, θ, π / 3) = sqrt 3 / 2) :=
sorry

end radius_of_circle_formed_by_spherical_coords_l479_479565


namespace A_lt_B_l479_479628

noncomputable def A := ∑ i in Finset.range 44, (2*i + 1) / ((i + 1) * (i + 2))^2

noncomputable def B := (Real.sqrt (3) - 1) ^ (1/3) * (Real.sqrt (3) + 1) ^ (1/3) / (2 ^ (1/3))

theorem A_lt_B : A < B := by
  have A_val : A = ∑ i in Finset.range 44, (2 * i + 1) / ((i + 1) * (i + 2))^2 := 
    by sorry
  have B_val : B = 1 :=
    by sorry
  calc
    A = ∑ i in Finset.range 44, (2 * i + 1) / ((i + 1) * (i + 2))^2 := A_val
    ... < 1 := by sorry
    B = 1 := B_val
    ... := by sorry

end A_lt_B_l479_479628


namespace iced_coffee_cost_correct_l479_479504

-- Definitions based on the conditions 
def coffee_cost_per_day (iced_coffee_cost : ℝ) : ℝ := 3 + iced_coffee_cost
def total_spent (days : ℕ) (iced_coffee_cost : ℝ) : ℝ := days * coffee_cost_per_day iced_coffee_cost

-- Proof statement
theorem iced_coffee_cost_correct (iced_coffee_cost : ℝ) (h : total_spent 20 iced_coffee_cost = 110) : iced_coffee_cost = 2.5 :=
by
  sorry

end iced_coffee_cost_correct_l479_479504


namespace max_value_of_function_in_interval_l479_479244

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem max_value_of_function_in_interval :
  ∃ x ∈ Icc (-4 : ℝ) (4 : ℝ), f x = 10 ∧ ∀ y ∈ Icc (-4) (4), f y ≤ f x :=
by
  sorry -- Proof omitted

end max_value_of_function_in_interval_l479_479244


namespace collinear_vectors_x_eq_neg_two_l479_479790

theorem collinear_vectors_x_eq_neg_two (x : ℝ) (a b : ℝ×ℝ) :
  a = (1, 2) → b = (x, -4) → a.1 * b.2 = a.2 * b.1 → x = -2 :=
by
  intro ha hb hc
  sorry

end collinear_vectors_x_eq_neg_two_l479_479790


namespace probability_sum_even_l479_479582

theorem probability_sum_even : 
  let faces := {1, 2, 3, 4, 5, 6}
  let p_even := 3 / 6
  let p_odd := 3 / 6
  let p_all_even := p_even ^ 3
  let p_two_odd_one_even := 3 * p_even * p_odd * p_odd
  let total_probability := p_all_even + p_two_odd_one_even
  in total_probability = 3 / 8 :=
by
  sorry

end probability_sum_even_l479_479582


namespace tan_sum_angle_identity_l479_479802

theorem tan_sum_angle_identity
  (α β : ℝ)
  (h1 : Real.tan (α + 2 * β) = 2)
  (h2 : Real.tan β = -3) :
  Real.tan (α + β) = -1 := sorry

end tan_sum_angle_identity_l479_479802


namespace probability_multiple_of_4_l479_479323

-- Definition of the problem conditions
def random_integer (n : ℕ) := ∀ i, 0 < i ∧ i ≤ n → Prop

def multiple_of_4 (i : ℕ) : Prop := i % 4 = 0

def count_multiples_of_4 (n : ℕ) : ℕ := (finset.range n).filter (λ x, multiple_of_4 x).card

-- Given problem conditions
def ben_choose_random_integer : Prop :=
  ∃ x y : ℕ, random_integer 60 x ∧ random_integer 60 y

-- Required proof statement
theorem probability_multiple_of_4 :
  (count_multiples_of_4 60 = 15) →
  (ben_choose_random_integer) →
  let probability := 1 - (3/4) * (3/4)
  in probability = 7/16 :=
begin
  intros h_multiples h_ben_choose,
  sorry
end

end probability_multiple_of_4_l479_479323


namespace probability_at_least_one_multiple_of_4_l479_479329

-- Define the condition
def random_integer_between_1_and_60 : set ℤ := {n : ℤ | 1 ≤ n ∧ n ≤ 60}

-- Define the probability theorems and the proof for probability calculation
theorem probability_at_least_one_multiple_of_4 :
  (∀ (n1 n2 : ℤ), (n1 ∈ random_integer_between_1_and_60) ∧ (n2 ∈ random_integer_between_1_and_60) → 
  (∃ k, n1 = 4 * k ∨ ∃ k, n2 = 4 * k)) ∧ 
  (1 / 60 * 1 / 60) * (15 / 60) ^ 2 = 7 / 16 :=
by
  sorry

end probability_at_least_one_multiple_of_4_l479_479329


namespace problem_statement_l479_479211

variables (a b c : ℝ)
variables (ha : 0 < a) (hab : a < b) (hbc : b < c)

theorem problem_statement : 
  ab < bc ∧ 
  ac < bc ∧ 
  ab < ac ∧ 
  a + c < b + c ∧ 
  c / a > 1 :=
by
  sorry

end problem_statement_l479_479211


namespace common_ratio_value_l479_479079

open_locale big_operators

noncomputable def common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n * q

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  2 * a 2 = a 0 + a 1

theorem common_ratio_value (a : ℕ → ℝ) (q : ℝ) (h_geom : common_ratio a q) (h_arith : arithmetic_sequence a) (h_q_ne_1 : q ≠ 1) : 
  q = -1/2 :=
sorry

end common_ratio_value_l479_479079


namespace school_annual_growth_rate_l479_479636

def average_annual_growth_rate (investment_2021 : ℝ) (investment_2023 : ℝ) (growth_rate : ℝ) : Prop :=
  investment_2023 = investment_2021 * (1 + growth_rate)^2

theorem school_annual_growth_rate : average_annual_growth_rate 10000 14400 0.2 :=
by 
  -- define the given investments and growth rate
  let investment_2021 := 10000
  let investment_2023 := 14400
  let growth_rate := 0.2
  -- now we write the equation given in the problem
  have h : investment_2023 = investment_2021 * (1 + growth_rate)^2 := sorry
  -- assume it fulfills the condition
  exact h

end school_annual_growth_rate_l479_479636


namespace mul_exponent_property_l479_479334

variable (m : ℕ)  -- Assuming m is a natural number for simplicity

theorem mul_exponent_property : m^2 * m^3 = m^5 := 
by {
  sorry
}

end mul_exponent_property_l479_479334


namespace grid_satisfies_conditions_l479_479841

-- Define the range of consecutive integers
def consecutive_integers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Define a function to check mutual co-primality condition for two given numbers
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the 3x3 grid arrangement as a matrix
@[reducible]
def grid : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![8, 9, 10],
    ![5, 7, 11],
    ![6, 13, 12]]

-- Define a predicate to check if the grid cells are mutually coprime for adjacent elements
def mutually_coprime_adjacent (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
    ∀ (i j k l : Fin 3), 
    (|i - k| + |j - l| = 1 ∨ |i - k| = |j - l| ∧ |i - k| = 1) → coprime (m i j) (m k l)

-- The main theorem statement
theorem grid_satisfies_conditions : 
  (∀ (i j : Fin 3), grid i j ∈ consecutive_integers) ∧ 
  mutually_coprime_adjacent grid :=
by
  sorry

end grid_satisfies_conditions_l479_479841


namespace angle_between_vecOA_vecOC_l479_479397

noncomputable def vecOA : ℝ := 1
noncomputable def vecOB : ℝ := 2
noncomputable def angleAOB : ℝ := 2 * Real.pi / 3
noncomputable def vecOC (vecOA vecOB : ℝ) : ℝ := (1 / 2) * vecOA + (1 / 4) * vecOB

noncomputable def cos_angle (vecOC vecOA : ℝ) :=
  vecOC * vecOA / (Real.sqrt(3) / 2 * vecOA)

noncomputable def arccos_deg (cos_val : ℝ) : ℝ := Real.acos (cos_val) * 180 / Real.pi

theorem angle_between_vecOA_vecOC : 
  arccos_deg (\cos_angle (vecOC vecOA vecOB) vecOA) = 60 :=
sorry

end angle_between_vecOA_vecOC_l479_479397


namespace diagonals_length_l479_479727

variable {R : Type*} [RealField R]

-- Define the rhombus and the given conditions
structure Rhombus (A B C D : R) :=
(acute_angle : ∀ (angle : R), angle < 90)
(perpendiculars_length : ∀ {P Q : R}, P = 3 ∧ Q = 3)
(distance_between_bases : R)
(B_distance : distance_between_bases = 3 * sqrt 3)
-- Calculation of the diagonals
theorem diagonals_length {A B C D : R} (rhombus : Rhombus A B C D) :
  ∃ (AC : R) (BD : R), AC = 6 ∧ BD = 2 * sqrt 3 :=
sorry

end diagonals_length_l479_479727


namespace stable_table_configurations_l479_479654

noncomputable def numberOfStableConfigurations (n : ℕ) : ℕ :=
  1 / 3 * (n + 1) * (2 * n ^ 2 + 4 * n + 3)

theorem stable_table_configurations (n : ℕ) (hn : 0 < n) :
  numberOfStableConfigurations n = 
    (1 / 3 * (n + 1) * (2 * n ^ 2 + 4 * n + 3)) :=
by
  sorry

end stable_table_configurations_l479_479654


namespace problem1_solution_problem2_solution_l479_479678

-- Problem 1
theorem problem1_solution (x y : ℝ) (h1 : 2 * x + 3 * y = 8) (h2 : x = y - 1) : x = 1 ∧ y = 2 := by
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) (h1 : 2 * x - y = -1) (h2 : x + 3 * y = 17) : x = 2 ∧ y = 5 := by
  sorry

end problem1_solution_problem2_solution_l479_479678


namespace magic_square_y_minus_x_l479_479457

theorem magic_square_y_minus_x :
  ∀ (x y : ℝ), 
    (x - 2 = 2 * y + y) ∧ (x - 2 = -2 + y + 6) →
    y - x = -6 :=
by 
  intros x y h
  sorry

end magic_square_y_minus_x_l479_479457


namespace haleys_car_distance_l479_479974

theorem haleys_car_distance (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used : ℕ) (distance_covered : ℕ) 
   (h_ratio : fuel_ratio = 4) (h_distance_ratio : distance_ratio = 7) (h_fuel_used : fuel_used = 44) :
   distance_covered = 77 := by
  -- Proof to be filled in
  sorry

end haleys_car_distance_l479_479974


namespace find_positions_l479_479267

def first_column (m : ℕ) : ℕ := 4 + 3*(m-1)

def table_element (m n : ℕ) : ℕ := first_column m + (n-1)*(2*m + 1)

theorem find_positions :
  (∀ m n, table_element m n ≠ 1994) ∧
  (∃ m n, table_element m n = 1995 ∧ ((m = 6 ∧ n = 153) ∨ (m = 153 ∧ n = 6))) :=
by
  sorry

end find_positions_l479_479267


namespace solution_y_amount_l479_479208

theorem solution_y_amount
  (solution_x : ℕ) (alcohol_x : ℕ) (solution_y : ℕ) (alcohol_y : ℕ) 
  (total_volume : ℕ) (desired_concentration : ℚ)
  (hx : alcohol_x = 10)
  (sx : solution_x = 100)
  (hy : alcohol_y = 30)
  (dc : desired_concentration = 0.25)
  (V_total : total_volume = solution_x + solution_y) :
  (10 + 0.3 * solution_y) / (100 + solution_y) = 0.25 → solution_y = 300 := 
begin
  sorry
end

end solution_y_amount_l479_479208


namespace round_robin_games_l479_479217

theorem round_robin_games (n : ℕ) (h : n = 18) :
  (n * (n - 1)) / 2 = 153 :=
by
  rw h
  norm_num
  sorry

end round_robin_games_l479_479217


namespace sum_of_roots_of_quadratic_l479_479801

theorem sum_of_roots_of_quadratic :
  ∀ x : ℝ, (x - 1) * (x + 4) = 18 -> (∃ a b c : ℝ, a = 1 ∧ b = 3 ∧ c = -22 ∧ ((a * x^2 + b * x + c = 0) ∧ (-b / a = -3))) :=
by
  sorry

end sum_of_roots_of_quadratic_l479_479801


namespace area_of_BEIH_l479_479475

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def point_on_segment {A B : ℝ × ℝ} (ratio : ℝ) : ℝ × ℝ :=
  ((1 - ratio) * A.1 + ratio * B.1, (1 - ratio) * A.2 + ratio * B.2)

noncomputable def shoelace_area (vertices : List (ℝ × ℝ)) : ℝ :=
  0.5 * abs (
    vertices.zip (vertices.drop 1 ++ [vertices.nth 0].get!) |
    (λ ⟨(x1, y1), (x2, y2)⟩, (x1 * y2) - (x2 * y1))
  ).sum

theorem area_of_BEIH :
  let 
    B : ℝ × ℝ := (0, 0),
    A : ℝ × ℝ := (0, 3),
    D : ℝ × ℝ := (3, 3),
    C : ℝ × ℝ := (3, 0),
    E : ℝ × ℝ := midpoint A B,
    F : ℝ × ℝ := midpoint B C,
    G : ℝ × ℝ := (3, 2), -- G such that CG = 2GD on CD
    I_x := (2 - 3) / (-3), -- solve x for y = 2 and line AF
    I : ℝ × ℝ := (1 / 3, 2),
    H : ℝ × ℝ := (3 / 4, 3 / 4)
  in 
    shoelace_area [B, E, I, H] = 7 / 8 :=
by 
  sorry

end area_of_BEIH_l479_479475


namespace rajesh_walked_distance_l479_479518

theorem rajesh_walked_distance (H : ℝ) (D_R : ℝ) 
  (h1 : D_R = 4 * H - 10)
  (h2 : H + D_R = 25) :
  D_R = 18 :=
by
  sorry

end rajesh_walked_distance_l479_479518


namespace validate_triples_l479_479690

theorem validate_triples :
  ∀ (m n k : ℕ), 
    m ≥ 2 → 
    n ≥ 2 → 
    k ≥ 3 → 
    ∃ (d : fin k → ℕ) (d' : fin k → ℕ),
      (∀ (i : fin k), d 0 = 1 ∧ d (k-1) = m) → 
      (∀ (i : fin k), d' 0 = 1 ∧ d' (k-1) = n) →
      (∀ (i : fin (k-2)) (h2_hk : 2 ≤ i + 2 ∧ i + 2 ≤ k - 1), d'.val (i + 2) = d.val (i + 2) + 1) →
      ∃ p, p = (4, 9, 3) ∨ p = (8, 15, 4) :=
by
  sorry

end validate_triples_l479_479690


namespace proportion_spotted_females_horned_males_l479_479451

def total_cows : ℕ := 300
def females_from_males (M : ℕ) : ℕ := 2 * M
def spots_eq_horns {F M : ℕ} (S : ℚ) : Prop := (S * F : ℚ) = (S * M : ℚ) + 50
def proportion_in_field (S : ℚ) (F M : ℕ) : Prop := S = 0.5

theorem proportion_spotted_females_horned_males :
  ∃ (S : ℚ) (M F : ℕ), M + F = total_cows ∧ F = females_from_males M ∧ spots_eq_horns S ∧ S = 0.5 :=
begin
  use 0.5,
  use 100,
  use 200,
  split,
  { exact (by norm_num : 100 + 200 = 300) },
  split,
  { exact (by norm_num : 200 = 2 * 100) },
  split,
  { norm_num,
    exact (by norm_num : 0.5 * 200 = 0.5 * 100 + 50) },
  { refl }
end

end proportion_spotted_females_horned_males_l479_479451


namespace remainder_of_M_div_45_l479_479170

-- Define the number M as described in the conditions
def M : ℤ := parseInt ((List.range (50 + 1)).tail.map Show.show).mkString

-- Theorem stating that M modulo 45 is equal to 15
theorem remainder_of_M_div_45 : (M % 45) = 15 :=
by
  sorry

end remainder_of_M_div_45_l479_479170


namespace quadrilateral_is_spatial_l479_479389

variable {V : Type*} [InnerProductSpace ℝ V]

variables {A B C D : V}

def isSpatialQuadrilateral (A B C D : V) : Prop :=
  (∃ u v w x : V, A = u ∧ B = v ∧ C = w ∧ D = x ∧
  (u ≠ v) ∧ (v ≠ w) ∧ (w ≠ x) ∧ (x ≠ u)) 

theorem quadrilateral_is_spatial (h1 : ⟪(B - A), (C - B)⟫ > 0)
                                 (h2 : ⟪(C - B), (D - C)⟫ > 0)
                                 (h3 : ⟪(D - C), (A - D)⟫ > 0)
                                 (h4 : ⟪(A - D), (B - A)⟫ > 0) :
  isSpatialQuadrilateral A B C D :=
sorry

end quadrilateral_is_spatial_l479_479389


namespace eight_faucets_fill_time_l479_479373

theorem eight_faucets_fill_time :
  (∀ t1 v1 f1 t2 v2 f2 : ℝ, f1 * t1 / v1 = f2 * t2 / v2) →
  (∀ t1 v1 f1 t2 v2 f2 : ℝ, v1 / t1 = f1 * (v2 / t2) / f2) →
  (∀ v t : ℝ, v / t = (200 / 8)) →
  (∀ f : ℝ, (200 / 8) / f = 6.25) →
  (∀ f : ℝ, 6.25 * f = 50) →
  (∀ v : ℝ, v / 50 = 1) →
  1 := 
sorry

end eight_faucets_fill_time_l479_479373


namespace john_next_birthday_l479_479162

variables {j e l : ℝ}

-- Conditions
def john_older_emily : Prop := j = 1.25 * e
def emily_younger_lucas : Prop := e = 0.7 * l
def sum_ages : Prop := j + e + l = 32

-- Statements to prove
theorem john_next_birthday
  (h1 : john_older_emily)
  (h2 : emily_younger_lucas)
  (h3 : sum_ages) :
  floor j + 1 = 11 :=
sorry

end john_next_birthday_l479_479162


namespace factorial_power_of_two_l479_479285

theorem factorial_power_of_two solutions (a b c : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_equation : a.factorial + b.factorial = 2^(c.factorial)) :
  solutions = [(1, 1, 1), (2, 2, 2)] :=
sorry

end factorial_power_of_two_l479_479285


namespace moles_of_NaCl_formed_l479_479714

-- Define the conditions
def moles_NaOH : ℕ := 3
def moles_HCl : ℕ := 3

-- Define the balanced chemical equation as a relation
def reaction (NaOH HCl NaCl H2O : ℕ) : Prop :=
  NaOH = HCl ∧ HCl = NaCl ∧ H2O = NaCl

-- Define the proof problem
theorem moles_of_NaCl_formed :
  ∀ (NaOH HCl NaCl H2O : ℕ), NaOH = 3 → HCl = 3 → reaction NaOH HCl NaCl H2O → NaCl = 3 :=
by
  intros NaOH HCl NaCl H2O hNa hHCl hReaction
  sorry

end moles_of_NaCl_formed_l479_479714


namespace relationship_among_a_b_c_l479_479379

noncomputable def a := 2 ^ (0.5 : ℝ)
noncomputable def b := Real.log 2
noncomputable def c := Real.logb 2 (Real.sin (2 * Real.pi / 5))

theorem relationship_among_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_among_a_b_c_l479_479379


namespace length_of_BD_in_right_triangle_l479_479487

/-- Given a right triangle \( \triangle ABC \) with a right angle at B, and a circle with 
diameter BC intersecting AC at point D, if the area of \( \triangle ABC \) is 180 and \( AC = 30 \), 
then the length of \( BD \), the altitude from B to AC, is 12. -/
theorem length_of_BD_in_right_triangle :
  ∀ (A B C D : Type) (AC BC BD : ℝ),
  (B is_right_angle_at B) ∧ (BC = diameter_of_circle) ∧
  (intersect AC D) ∧ (area_of_triangle ABC = 180) -> (AC = 30) -> 
  (BD = 12) :=
by
  sorry

end length_of_BD_in_right_triangle_l479_479487


namespace both_can_decode_neither_can_decode_exactly_one_can_decode_l479_479922

def probability (A B : Prop) : Rat := sorry

axiom decode_probability_A : probability decode_A decode_B = 1/3
axiom decode_probability_B : probability decode_A decode_B = 1/4
axiom decode_independent : independent decode_A decode_B

theorem both_can_decode : probability (decode_A ∧ decode_B) = 1/12 := sorry

theorem neither_can_decode : probability (¬decode_A ∧ ¬decode_B) = 1/2 := sorry

theorem exactly_one_can_decode : probability ((decode_A ∧ ¬decode_B) ∨ (¬decode_A ∧ decode_B)) = 1/3 := sorry

end both_can_decode_neither_can_decode_exactly_one_can_decode_l479_479922


namespace sufficient_condition_for_no_real_roots_l479_479906

noncomputable def eq_cubic_roots_no_real (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h_geom : b^2 = a * c) : Prop :=
  let delta3 := (c^2 - 16) in delta3 < 0

theorem sufficient_condition_for_no_real_roots 
  (a b : ℝ) (ha : a^2 ≥ 4) (hb : b^2 < 8) : 
  eq_cubic_roots_no_real a b (b^2 / a) (gt_of_ge_of_gt ha (by linarith)) (by linarith) (div_pos hb (by linarith))
    (by linarith [pow_two b, pow_two a]) :=
  by sorry

end sufficient_condition_for_no_real_roots_l479_479906


namespace necessary_but_not_sufficient_condition_l479_479224

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (|x - 1| < 2) → (x(x - 3) < 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l479_479224


namespace axis_of_symmetry_of_parabola_l479_479759

theorem axis_of_symmetry_of_parabola (a b c : ℝ) :
  (a * (-2)^2 + b * (-2) + c = 3) ∧ (a * 4^2 + b * 4 + c = 3) →
  ∃ x : ℝ, x = 1 :=
by
  intro h
  use 1
  sorry

end axis_of_symmetry_of_parabola_l479_479759


namespace intersect_distance_l479_479442

open Real

variables (p x y : ℝ)

-- Conditions
def line_eq (p x : ℝ) := 2 * x + p / 2
def para_eq (p y : ℝ) := 2 * p * y

-- Statement
theorem intersect_distance (h_pos : p > 0) (h_line : y = line_eq p x) (h_para : x^2 = para_eq p y) :
  ∃ A B : ℝ, dist A B = 10 * p :=
sorry

end intersect_distance_l479_479442


namespace number_of_positive_integers_with_positive_log_l479_479104

theorem number_of_positive_integers_with_positive_log (b : ℕ) (h : ∃ n : ℕ, n > 0 ∧ b ^ n = 1024) : 
  ∃ L, L = 4 :=
sorry

end number_of_positive_integers_with_positive_log_l479_479104


namespace sum_inequality_l479_479900

theorem sum_inequality (n : ℕ) (a : ℕ → ℝ) (h : n > 3) (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) : 
  1 < ∑ i in Finset.range n, (a i) / (a (i - 1) + a i + a (i + 1)) ∧ 
  ∑ i in Finset.range n, (a i) / (a (i - 1) + a i + a (i + 1)) < ⌊(n : ℝ) / 2⌋ :=
sorry

end sum_inequality_l479_479900


namespace probability_of_at_least_one_multiple_of_4_l479_479326

open ProbabilityTheory

def prob_at_least_one_multiple_of_4 : ℚ := 7 / 16

theorem probability_of_at_least_one_multiple_of_4 :
  let S := Finset.range 60
  let multiples_of_4 := S.filter (λ x, (x + 1) % 4 = 0)
  let prob (a b : ℕ) := (a : ℚ) / b
  let prob_neither_multiple_4 := (prob (60 - multiples_of_4.card) 60) ^ 2
  1 - prob_neither_multiple_4 = prob_at_least_one_multiple_of_4 := by
  sorry

end probability_of_at_least_one_multiple_of_4_l479_479326


namespace sum_of_coefficients_l479_479729

theorem sum_of_coefficients 
  (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℕ)
  (h : (3 * x - 1) ^ 10 = a + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 + a_6 * x ^ 6 + a_7 * x ^ 7 + a_8 * x ^ 8 + a_9 * x ^ 9 + a_10 * x ^ 10) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 1023 := 
sorry

end sum_of_coefficients_l479_479729


namespace quadratic_pos_in_interval_l479_479619

theorem quadratic_pos_in_interval (m n : ℤ)
  (h2014 : (2014:ℤ)^2 + m * 2014 + n > 0)
  (h2015 : (2015:ℤ)^2 + m * 2015 + n > 0) :
  ∀ x : ℝ, 2014 ≤ x ∧ x ≤ 2015 → (x^2 + (m:ℝ) * x + (n:ℝ)) > 0 :=
by
  sorry

end quadratic_pos_in_interval_l479_479619


namespace piercing_line_exists_l479_479653

def cube_dimension := 20
def brick_count := 2000
def brick_dimensions := (2, 2, 1)

def can_pierce_without_intersecting (C_dim : ℕ) (brick_num : ℕ) (brick_dims : ℕ × ℕ × ℕ) : Prop :=
  ∃ (line : ℝ → ℝ × ℝ × ℝ), (line (C_dim / 2) = (C_dim / 2, C_dim / 2, C_dim))

theorem piercing_line_exists : can_pierce_without_intersecting cube_dimension brick_count brick_dimensions :=
  sorry

end piercing_line_exists_l479_479653


namespace point_minimizes_expression_l479_479197

variables {A B C P D E F : Type} [Euclidean_geometry]
variables {triangle_ABC : triangle A B C}

def is_feet_of_perpendicular (P D : Type) (side : line) : Prop :=
  perpendicular P side ∧ foot_of_perpendicular P side = D

noncomputable def minimizes_perpendicular_distance (P : Type) (triangle : triangle A B C) :=
  ∀ P', ∑ dis in feet tri P', (length side_tri / length dis) ≥ ∑ dis in feet tri P, (length side_tri / length dis)

theorem point_minimizes_expression (P : Type) (triangle : triangle A B C) (H1: in_interior_triangle P triangle)
  (H2: ∀ side feet, is_feet_of_perpendicular P feet side) :
  minimizes_perpendicular_distance P triangle ↔ incenter_of_triangle P triangle :=
sorry

end point_minimizes_expression_l479_479197


namespace cos_shift_sin_l479_479989

theorem cos_shift_sin (x : ℝ) : 
  ∃ (δ : ℝ), (λ x, cos (x + π/3)) x = (λ x, sin (x + δ)) x ∧ δ = -5*π/6 :=
sorry

end cos_shift_sin_l479_479989


namespace minimum_value_expression_l479_479054

theorem minimum_value_expression 
  (a b c d : ℝ)
  (h1 : (2 * a^2 - Real.log a) / b = 1)
  (h2 : (3 * c - 2) / d = 1) :
  ∃ min_val : ℝ, min_val = (a - c)^2 + (b - d)^2 ∧ min_val = 1 / 10 :=
by {
  sorry
}

end minimum_value_expression_l479_479054


namespace find_min_intersection_l479_479032

def num_subsets (S : Set α) : ℕ := 2 ^ (Set.card S)

theorem find_min_intersection (A B C : Set α) (card_A : Set.card A = 100)
    (card_B : Set.card B = 101) (intersect_ge : Set.card (A ∩ B) ≥ 95)
    (subsets_eq : num_subsets A + num_subsets B + num_subsets C = num_subsets (A ∪ B ∪ C)) :
    ∃ (k : ℕ), k = 96 ∧ Set.card (A ∩ B ∩ C) ≥ k :=
begin
  sorry 
end

end find_min_intersection_l479_479032


namespace discount_percentage_l479_479227

theorem discount_percentage (M C S : ℝ) (hC : C = 0.64 * M) (hS : S = C * 1.28125) :
  ((M - S) / M) * 100 = 18.08 := 
by
  sorry

end discount_percentage_l479_479227


namespace value_of_r_when_n_is_2_l479_479481

-- Define the given conditions
def s : ℕ := 2 ^ 2 + 1
def r : ℤ := 3 ^ s - s

-- Prove that r equals 238 when n = 2
theorem value_of_r_when_n_is_2 : r = 238 := by
  sorry

end value_of_r_when_n_is_2_l479_479481


namespace count_distinct_ordered_pairs_l479_479763

theorem count_distinct_ordered_pairs : 
  { n : ℕ // n = 19 } :=
by
  let a_values := {a // a % 2 = 0 ∧ 1 ≤ a ∧ a ≤ 38}
  have h_a_card : a_values.card = 19, sorry -- This is solved with arithmetic sequence reasoning
  let b_values := {b // ∃ a ∈ a_values, a + b = 40}
  have h_b_card : ∀ (a : ℕ) (ha : a ∈ a_values), 40 - a > 0, 
  { intros a ha,
    have : b = 40 - a,
    rw [this, gt_iff_lt],
    exact sorry -- Remaining inequalities to ensure b > 0
  }
  exact ⟨19, h_a_card⟩

end count_distinct_ordered_pairs_l479_479763


namespace valid_sentences_count_l479_479946

inductive GnollishWord
| splargh
| glumph
| amr
| gazak

open GnollishWord

def is_valid_sentence : list GnollishWord → Prop
| (splargh :: glumph :: _) := false
| (_ :: amr :: gazak :: _) := false
| _ := true

def count_valid_sentences : ℕ :=
  (list.replicate 3 [splargh, glumph, amr, gazak]).product.filter is_valid_sentence.length

theorem valid_sentences_count : count_valid_sentences = 50 :=
by
  -- proof needs to be filled here
  sorry

end valid_sentences_count_l479_479946


namespace minimum_greeting_pairs_l479_479983

def minimum_mutual_greetings (n: ℕ) (g: ℕ) : ℕ :=
  (n * g - (n * (n - 1)) / 2)

theorem minimum_greeting_pairs :
  minimum_mutual_greetings 400 200 = 200 :=
by 
  sorry

end minimum_greeting_pairs_l479_479983


namespace find_fraction_l479_479289

-- Define the constants as given in the problem
def half_number := 945.0000000000013
def number := 2 * half_number

def lhs := (4 / 15) * (5 / 7) * number
def rhs (F : ℝ) := (4 / 9) * F * number + 24

-- State the theorem to solve for the fraction (F)
theorem find_fraction : ∃ F : ℝ, lhs = rhs F :=
by
  let n := number
  have h_n : half_number * 2 = n := rfl
  have h_lhs : lhs = (4 / 15) * (5 / 7) * n := rfl
  have h_rhs : ∀ F : ℝ, rhs F = (4 / 9) * F * n + 24 := by intros; refl
  use 0.4
  have n_val := (4 / 15) * (5 / 7) * 2 * 945.0000000000013
  have lhs_val := (4 / 15) * (5 / 7) * n
  rw [←h_lhs, h_rhs 0.4]
  sorry

end find_fraction_l479_479289


namespace range_of_a_l479_479911

variables (a : ℝ)
def A : set ℝ := if a > 1 then set.univ else {x | x ≤ a ∨ x ≥ 2 - a}
def B : set ℤ := set.univ

theorem range_of_a 
  (ha : a ∈ ℝ) 
  (hB : B = {x : ℤ | ∃ n : ℤ, x = n}) 
  (hCA_card : ∀ x ∈ A, ¬ x ∈ B ∧ fintype.card (Aᶜ ∩ B) = 3) 
  : 0 ≤ a ∧ a < 1 := 
sorry

end range_of_a_l479_479911


namespace reduction_amount_is_250_l479_479968

-- Definitions from the conditions
def original_price : ℝ := 500
def reduction_rate : ℝ := 0.5

-- The statement to be proved
theorem reduction_amount_is_250 : (reduction_rate * original_price) = 250 := by
  sorry

end reduction_amount_is_250_l479_479968


namespace sin_cos_expr_l479_479099

theorem sin_cos_expr (θ : ℝ) (h : (cot θ) ^ 2000 + 2 = sin θ + 1) : (sin θ + 2)^2 * (cos θ + 1) = 9 :=
sorry

end sin_cos_expr_l479_479099


namespace g_is_correct_range_of_a_correct_l479_479772

section
variable {a : ℝ}

/-- Define the function f -/
def f (x : ℝ) : ℝ := a * x - abs (x + 1)

/-- Define the odd function g -/
def g (x : ℝ) : ℝ :=
  if x > 0 then (a-1) * x - 1
  else if x < 0 then (a-1) * x + 1
  else 0

/-- Define the range of a for which f has a maximum value -/
def range_of_a (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1

theorem g_is_correct :
  ∀ x, g x = if x > 0 then (a-1) * x - 1 else if x < 0 then (a-1) * x + 1 else 0 := by
  sorry

theorem range_of_a_correct :
  (∃ (x : ℝ), ∃ (y : ℝ), f (x) ≤ y) → range_of_a a := by
  sorry

end

end g_is_correct_range_of_a_correct_l479_479772


namespace letters_to_word_l479_479205

def letter_order (P O K V M B S q : ℕ) : list ℕ :=
  if P > O ∧ O > K ∧ V > M ∧ M < S ∧ V > O ∧ O > B then
    [K, B, M, S, O, P, V]
  else
    []

theorem letters_to_word (P O K V M B S q : ℕ) :
  P > O ∧ O > K ∧ V > M ∧ M < S ∧ V > O ∧ O > B ∧
  ([K, B, M, S, O, P, V] = [1, 2, 3, 4, 5, 6, 7]) →
  "КОМПЬЮТЕР" := sorry

end letters_to_word_l479_479205


namespace probability_increase_l479_479147

theorem probability_increase (p q : ℝ) (k l : ℕ) (hpq: q = 1 - p) (hk : 0 ≤ k ∧ k ≤ 14) (hl : 0 ≤ l ∧ l ≤ 14) :
  let increase := binom (k + l) k * q^(l + 1) * p^k in
  increase = binom (k + l) k * q^(l + 1) * p^k := 
by
  sorry

end probability_increase_l479_479147


namespace max_gcd_15n_plus_4_8n_plus_1_l479_479665

theorem max_gcd_15n_plus_4_8n_plus_1 (n : ℕ) (h : n > 0) : 
  ∃ g, g = gcd (15 * n + 4) (8 * n + 1) ∧ g ≤ 17 :=
sorry

end max_gcd_15n_plus_4_8n_plus_1_l479_479665


namespace sum_of_distinct_a_values_l479_479767

def f (x : ℝ) : ℝ :=
  ∑ i in (Finset.range 2019).image (fun i => i + 1), |x + i| +
  ∑ i in (Finset.range 2019).image (fun i => -(i + 1)), |x + i|

theorem sum_of_distinct_a_values
  (h1 : ∀ a : ℝ, f (a^2 - 3 * a + 2) = f (a - 1)) :
  ∑ a in (finset.filter (λ a, a ∈ ℝ ∧ f (a^2 - 3 * a + 2) = f (a - 1)) (finset.range 2019).image (λ i, i + 1)), a = 6 :=
by
  sorry

end sum_of_distinct_a_values_l479_479767


namespace find_f_at_10_over_3_l479_479060

-- Definitions based on conditions
def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 8^x - 1
  else if 0 ≤ 2 - x ∧ 2 - x ≤ 1 then f (2 - x)
  else if x < 0 then -f (-x)
  else f (x mod 4) -- Modulo for periodic property

-- Statement to prove
theorem find_f_at_10_over_3 : f (10 / 3) = -3 := sorry

end find_f_at_10_over_3_l479_479060


namespace gcd_of_8247_13619_29826_l479_479361

theorem gcd_of_8247_13619_29826 : Nat.gcd (Nat.gcd 8247 13619) 29826 = 3 := 
sorry

end gcd_of_8247_13619_29826_l479_479361


namespace cycling_sequences_reappear_after_28_cycles_l479_479241

/-- Cycling pattern of letters and digits. Letter cycle length is 7; digit cycle length is 4.
Prove that the LCM of 7 and 4 is 28, which is the first line on which both sequences will reappear -/
theorem cycling_sequences_reappear_after_28_cycles 
  (letters_cycle_length : ℕ) (digits_cycle_length : ℕ) 
  (h_letters : letters_cycle_length = 7) 
  (h_digits : digits_cycle_length = 4) 
  : Nat.lcm letters_cycle_length digits_cycle_length = 28 :=
by
  rw [h_letters, h_digits]
  sorry

end cycling_sequences_reappear_after_28_cycles_l479_479241


namespace radius_of_circle_formed_by_spherical_coords_l479_479564

theorem radius_of_circle_formed_by_spherical_coords :
  (∃ θ : ℝ, radius_of_circle (1, θ, π / 3) = sqrt 3 / 2) :=
sorry

end radius_of_circle_formed_by_spherical_coords_l479_479564


namespace tina_money_left_l479_479587

theorem tina_money_left :
  let june_savings := 27
  let july_savings := 14
  let august_savings := 21
  let books_spending := 5
  let shoes_spending := 17
  june_savings + july_savings + august_savings - (books_spending + shoes_spending) = 40 :=
by
  sorry

end tina_money_left_l479_479587


namespace integer_solutions_count_l479_479797

theorem integer_solutions_count : 
  {x : ℤ // (x - 2) ^ (13 - x ^ 2) = 1}.to_finset.card = 2 := 
sorry

end integer_solutions_count_l479_479797


namespace tony_needs_46_gallons_l479_479990

def column_radius (d : ℝ) : ℝ := d / 2
def lateral_area (r h : ℝ) : ℝ := 2 * Real.pi * r * h
def top_bottom_area (r : ℝ) : ℝ := Real.pi * r^2
def total_area (r h : ℝ) : ℝ := lateral_area r h + 2 * top_bottom_area r
def total_paint_area (r h : ℝ) (n : ℕ) : ℝ := n * total_area r h
def gallons_of_paint (area : ℝ) (coverage : ℝ) : ℝ := area / coverage
def round_up (x : ℝ) : ℕ := ⌈x⌉

theorem tony_needs_46_gallons :
  let diameter := 12
  let height := 24
  let num_columns := 12
  let coverage := 300
  let r := column_radius diameter
  let area := total_paint_area r height num_columns
  let gallons_needed := gallons_of_paint area coverage
  round_up gallons_needed = 46 := by
  sorry

end tony_needs_46_gallons_l479_479990


namespace max_value_of_distances_l479_479733

noncomputable def point_intersection (m : ℝ) : ℝ × ℝ :=
  let x := 1 in
  let y := 3 in
  (x, y)

def dist (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.fst - A.fst)^2 + (B.snd - A.snd)^2)

theorem max_value_of_distances (m : ℝ) :
  let A := (0, 0)
  let B := (1, 3)
  let M := point_intersection m
  dist A M^2 + dist M B^2 = dist A B^2 →
  dist A B^2 = 10 →
  2 * dist A M * dist M B ≤ 10 :=
by
  sorry

end max_value_of_distances_l479_479733


namespace jake_snakes_l479_479884

theorem jake_snakes (S : ℕ) 
  (h1 : 2 * S + 1 = 6) 
  (h2 : 2250 = 5 * 250 + 1000) :
  S = 3 := 
by
  sorry

end jake_snakes_l479_479884


namespace more_students_than_guinea_pigs_l479_479010

theorem more_students_than_guinea_pigs (students_per_classroom guinea_pigs_per_classroom classrooms : ℕ)
  (h1 : students_per_classroom = 24) 
  (h2 : guinea_pigs_per_classroom = 3) 
  (h3 : classrooms = 6) : 
  (students_per_classroom * classrooms) - (guinea_pigs_per_classroom * classrooms) = 126 := 
by
  sorry

end more_students_than_guinea_pigs_l479_479010


namespace colored_squares_possible_l479_479266

theorem colored_squares_possible :
  ∀ (n : ℕ), (n = 99 ∨ n = 100) →
  ∃ (assembly : Fin n → Fin n → (Color × Color × Color × Color)),
  (∀ i j, assembly i j = (C1, C2, C3, C4) ∨ assembly i j = (C1', C2', C3', C4')) ∧
  (∀ k, ∃ c, color_of_side (outer_side assembly k) = c ∧
    ∃ k', color_of_side (outer_side assembly k') ≠ color_of_side (outer_side assembly k) ∧
    color_of_side (outer_side assembly k') ≠ c) :=
begin
  sorry
end

end colored_squares_possible_l479_479266


namespace diagonals_in_polygon_l479_479795

theorem diagonals_in_polygon (n : Nat) (h : n = 27) : 
  let diagonals := (n * (n - 3)) / 2 in 
  diagonals = 324 := 
by
  rw h
  have h1 : (27 * 24) / 2 = 324 := by norm_num
  exact h1
  sorry

end diagonals_in_polygon_l479_479795


namespace radius_of_given_spherical_circle_l479_479561
noncomputable def circle_radius_spherical_coords : Real :=
  let spherical_to_cartesian (rho theta phi : Real) : (Real × Real × Real) :=
    (rho * (Real.sin phi) * (Real.cos theta), rho * (Real.sin phi) * (Real.sin theta), rho * (Real.cos phi))
  let (rho, theta, phi) := (1, 0, Real.pi / 3)
  let (x, y, z) := spherical_to_cartesian rho theta phi
  let radius := Real.sqrt (x^2 + y^2)
  radius

theorem radius_of_given_spherical_circle :
  circle_radius_spherical_coords = (Real.sqrt 3) / 2 :=
sorry

end radius_of_given_spherical_circle_l479_479561


namespace volume_of_cube_l479_479509

theorem volume_of_cube (a : ℕ) (h : a^3 - (a^3 - 4 * a) = 12) : a^3 = 27 :=
by 
  sorry

end volume_of_cube_l479_479509


namespace a_plus_b_equals_neg_four_l479_479732

-- Conditions as definitions
def a : ℝ := sorry -- place-holder for 'a', will be defined during proof
def b : ℝ := sorry -- place-holder for 'b', will be defined during proof
def i : ℂ := Complex.I -- imaginary unit

-- Given condition
def cond : Proof (a - 2 * i = b + a * i) := sorry

-- Statement to prove
theorem a_plus_b_equals_neg_four (h : cond) : a + b = -4 :=
  sorry

end a_plus_b_equals_neg_four_l479_479732


namespace three_digit_number_condition_l479_479709

theorem three_digit_number_condition (x y z : ℕ) (h₀ : 1 ≤ x ∧ x ≤ 9) (h₁ : 0 ≤ y ∧ y ≤ 9) (h₂ : 0 ≤ z ∧ z ≤ 9)
(h₃ : 100 * x + 10 * y + z = 34 * (x + y + z)) : 
100 * x + 10 * y + z = 102 ∨ 100 * x + 10 * y + z = 204 ∨ 100 * x + 10 * y + z = 306 ∨ 100 * x + 10 * y + z = 408 :=
sorry

end three_digit_number_condition_l479_479709


namespace grid_is_valid_l479_479830

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Check if all adjacent cells (by side or diagonal) in a 3x3 grid are coprime -/
def valid_grid (grid : Array (Array ℕ)) : Prop :=
  let adjacent_pairs := 
    [(0, 0, 0, 1), (0, 1, 0, 2), 
     (1, 0, 1, 1), (1, 1, 1, 2), 
     (2, 0, 2, 1), (2, 1, 2, 2),
     (0, 0, 1, 0), (0, 1, 1, 1), (0, 2, 1, 2),
     (1, 0, 2, 0), (1, 1, 2, 1), (1, 2, 2, 2),
     (0, 0, 1, 1), (0, 1, 1, 2), 
     (1, 0, 2, 1), (1, 1, 2, 2), 
     (0, 2, 1, 1), (1, 2, 2, 1), 
     (0, 1, 1, 0), (1, 1, 2, 0)] 
  adjacent_pairs.all (λ ⟨r1, c1, r2, c2⟩, is_coprime (grid[r1][c1]) (grid[r2][c2]))

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10],
    #[5, 7, 11],
    #[6, 13, 12]]

theorem grid_is_valid : valid_grid grid :=
  by
    -- Proof omitted, placeholder
    sorry

end grid_is_valid_l479_479830


namespace system_has_real_solution_l479_479687

theorem system_has_real_solution (k : ℝ) : 
  (∃ x y : ℝ, y = k * x + 4 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 :=
by
  sorry

end system_has_real_solution_l479_479687


namespace area_of_ABC_l479_479596

noncomputable def points_A_B_C_D_E_coplanar : Prop :=
  ∃ (A B C D E : ℝ × ℝ), coplanar {A, B, C, D, E}

noncomputable def right_angle_D (A D C : ℝ × ℝ) : Prop :=
  angle A D C = π / 2

noncomputable def segment_lengths (A B C D E : ℝ × ℝ) : Prop :=
  dist A C = 10 ∧ dist A B = 17 ∧ dist D C = 6 ∧ dist D E = 8 ∧ E ∈ line_through D C

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem area_of_ABC 
  (A B C D E : ℝ × ℝ)
  (h0 : points_A_B_C_D_E_coplanar)
  (h1 : right_angle_D A D C) 
  (h2 : segment_lengths A B C D E) 
  : area_triangle A B C = 84 := 
sorry

end area_of_ABC_l479_479596


namespace midpoint_of_XY_l479_479169

noncomputable def midpoint_trapezoid (A B C D M P X Q Y : Point) : Prop :=
  (trapezoid A B C D) ∧
  (midpoint M A B) ∧
  P ∈ segment B C ∧
  X = line_intersection (line_through P D) (line_through A B) ∧
  Q = line_intersection (line_through P M) (line_through B D) ∧
  Y = line_intersection (line_through P Q) (line_through A B) →
  midpoint M X Y

-- The theorem statement
theorem midpoint_of_XY {A B C D M P X Q Y : Point} (h : midpoint_trapezoid A B C D M P X Q Y) : 
  midpoint M X Y :=
begin
  -- proof goes here
  sorry
end

end midpoint_of_XY_l479_479169


namespace vanessa_needs_more_quarters_l479_479264

theorem vanessa_needs_more_quarters :
    let total_amount := 213.75
    let quarter_value := 0.25
    let quarters_per_soda := 7
    let total_quarters := total_amount / quarter_value
    let sodas_bought := total_quarters / quarters_per_soda
    let remainder_quarters := total_quarters % quarters_per_soda
    7 - remainder_quarters = 6 :=
by
  let total_amount := 213.75
  let quarter_value := 0.25
  let quarters_per_soda := 7
  let total_quarters := total_amount / quarter_value
  let sodas_bought := total_quarters / quarters_per_soda
  let remainder_quarters := total_quarters % quarters_per_soda
  show 7 - remainder_quarters = 6, from sorry

end vanessa_needs_more_quarters_l479_479264


namespace minimum_dot_product_parallelogram_PABQ_l479_479055

-- Problem 1: Minimum value of the dot product
theorem minimum_dot_product :
  ∀ (x0 y0 : ℝ), (x0 ^ 2 / 3 + y0 ^ 2 / 2 = 1) →
  let F1 := (-1, 0)
  let F2 := (1, 0)
  let PF1 := (-1 - x0, -y0)
  let PF2 := (1 - x0, -y0)
  (PF1.1 * PF2.1 + PF1.2 * PF2.2) = x0 ^ 2 + y0 ^ 2 - 1 →
  x0 ^ 2 + y0 ^ 2 - 1 = 1 :=
sorry

-- Problem 2: Equation of the line for parallelogram PABQ
theorem parallelogram_PABQ :
  let P := (-1, 2 * real.sqrt 3 / 3) in
  let k := - real.sqrt 3 / 3 in
  let l := (x : ℝ) (y : ℝ) := y = k * (x + 1) in
  ∀ (Q : ℝ × ℝ), (Q ≠ P ∧ (Q.2 - P.2 = k * (Q.1 - P.1)) ∧ ((Q.1 ^ 2 / 3) + (Q.2 ^ 2 / 2) = 1) →
  l (Q.1) (Q.2)) →
  x + real.sqrt 3 * y + 1 = 0 :=
sorry

end minimum_dot_product_parallelogram_PABQ_l479_479055


namespace number_of_students_in_class_l479_479498

theorem number_of_students_in_class
  (total_stickers : ℕ) (stickers_to_friends : ℕ) (stickers_left : ℝ) (students_each : ℕ → ℝ)
  (n_friends : ℕ) (remaining_stickers : ℝ) :
  total_stickers = 300 →
  stickers_to_friends = (n_friends * (n_friends + 1)) / 2 →
  stickers_left = 7.5 →
  ∀ n, n_friends = 10 →
  remaining_stickers = total_stickers - stickers_to_friends - (students_each n_friends) * (n - n_friends - 1) →
  (∃ n : ℕ, remaining_stickers = 7.5 ∧
              total_stickers - (stickers_to_friends + (students_each (n - n_friends - 1) * (n - n_friends - 1))) = 7.5) :=
by
  sorry

end number_of_students_in_class_l479_479498


namespace reachable_pair_D_l479_479723

noncomputable def fA (x : ℝ) : ℝ := Real.cos x
noncomputable def gA (x : ℝ) : ℝ := 2

noncomputable def fB (x : ℝ) : ℝ := Real.log x^2 - 2*x + 5
noncomputable def gB (x : ℝ) : ℝ := Real.sin (Real.pi/2 * x)

noncomputable def fC (x : ℝ) : ℝ := Real.sqrt (4 - x^2)
noncomputable def gC (x : ℝ) : ℝ := (3/4)*x + 15/4

noncomputable def fD (x : ℝ) : ℝ := x + 2/x
noncomputable def gD (x : ℝ) : ℝ := Real.log x + 2

/-- The pairs (fA, gA), (fB, gB), and (fC, gC) are not reachable, but (fD, gD) is reachable. -/
theorem reachable_pair_D: 
  (∀ x, |fA x - gA x| ≥ 1) ∧
  (∀ x, |fB x - gB x| ≥ 1) ∧
  (∀ x, |fC x - gC x| ≥ 1) ∧
  (∃ x, |fD x - gD x| < 1) :=
by
  sorry

end reachable_pair_D_l479_479723


namespace number_of_solution_pairs_l479_479695

theorem number_of_solution_pairs : ∃! (n : ℕ), ∃ (x y : ℕ), 4 * x + 7 * y = 600 ∧ 1 ≤ x ∧ 1 ≤ y ∧ n = 22 :=
begin
  sorry
end

end number_of_solution_pairs_l479_479695


namespace problem_equivalence_l479_479097

-- Define vectors a and b
def vector_a (m : ℝ) : ℝ × ℝ × ℝ := (m, 1, 0)
def vector_b : ℝ × ℝ × ℝ := (2, 1, 1)

-- Definition to check length of a vector
def vec_len (v : ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Dot product of two vectors
def dot_product (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

-- Projection of vector a onto vector b
def proj_vector_a_on_b (m : ℝ) : ℝ × ℝ × ℝ :=
  let a := vector_a m
  let b := vector_b
  let scale := dot_product a b / dot_product b b
  (scale * b.1, scale * b.2, scale * b.3)

-- Lean 4 statement of the proof problem
theorem problem_equivalence :
  ∀ (m : ℝ),
    (vec_len (vector_a m) = real.sqrt 5 → m = 2 ∨ m = -2) ∧
    (dot_product (vector_a m) vector_b = 0 → m = -1/2) ∧
    (m = 1 → proj_vector_a_on_b m = (1, 1/2, 1/2)) :=
by
  sorry

end problem_equivalence_l479_479097


namespace isosceles_triangle_exists_l479_479050

-- Definitions for a triangle vertex and side lengths
structure Triangle :=
  (A B C : ℝ × ℝ) -- Vertices A, B, C
  (AB AC BC : ℝ)  -- Sides AB, AC, BC

-- Definition for all sides being less than 1 unit
def sides_less_than_one (T : Triangle) : Prop :=
  T.AB < 1 ∧ T.AC < 1 ∧ T.BC < 1

-- Definition for isosceles triangle containing the original one
def exists_isosceles_containing (T : Triangle) : Prop :=
  ∃ (T' : Triangle), 
    (T'.AB = T'.AC ∨ T'.AB = T'.BC ∨ T'.AC = T'.BC) ∧
    T'.A = T.A ∧ -- T'.A vertex is same as T.A
    (T'.AB < 1 ∧ T'.AC < 1 ∧ T'.BC < 1) ∧
    (∃ (B1 : ℝ × ℝ), -- There exists point B1 such that new triangle T' incorporates B1
      T'.B = B1 ∧
      T'.C = T.C) -- T' also has vertex C of original triangle

-- Complete theorem statement
theorem isosceles_triangle_exists (T : Triangle) (hT : sides_less_than_one T) : exists_isosceles_containing T :=
by 
  sorry

end isosceles_triangle_exists_l479_479050


namespace contrapositive_example_l479_479533

theorem contrapositive_example (x : ℝ) : (¬ (2^x + 1 < 3) → ¬ (x < 1)) ↔ (x ≥ 1 → 2^x + 1 ≥ 3) :=
begin
  sorry
end

end contrapositive_example_l479_479533


namespace rabbit_travel_time_l479_479649

theorem rabbit_travel_time :
  ∀ (speed distance : ℝ), speed = 10 ∧ distance = 3 → (distance / speed) * 60 = 18 :=
by
  intros speed distance h
  cases h with h_speed h_distance
  rw [h_speed, h_distance]
  norm_num
  sorry

end rabbit_travel_time_l479_479649


namespace smallest_nine_consecutive_sum_l479_479250

theorem smallest_nine_consecutive_sum (n : ℕ) (h : (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) + (n+8) = 2007)) : n = 219 :=
sorry

end smallest_nine_consecutive_sum_l479_479250


namespace ribbon_tying_length_l479_479985

theorem ribbon_tying_length :
  let l1 := 36
  let l2 := 42
  let l3 := 48
  let cut1 := l1 / 6
  let cut2 := l2 / 6
  let cut3 := l3 / 6
  let rem1 := l1 - cut1
  let rem2 := l2 - cut2
  let rem3 := l3 - cut3
  let total_rem := rem1 + rem2 + rem3
  let final_length := 97
  let tying_length := total_rem - final_length
  tying_length = 8 :=
by
  sorry

end ribbon_tying_length_l479_479985


namespace mod_inverse_correct_l479_479363

def mod_inverse_of_30_mod_31 : ℤ :=
  30

theorem mod_inverse_correct : ∃ a : ℤ, 30 * a % 31 = 1 ∧ a = mod_inverse_of_30_mod_31 :=
by
  use 30
  split
  · sorry
  · rfl

end mod_inverse_correct_l479_479363


namespace find_value_of_a_l479_479413

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then 2^x + 1 else x^2 + a * x

theorem find_value_of_a (a : ℝ) (h : f a (f a 0) = 4 * a) : a = 2 :=
by {
  sorry
}

end find_value_of_a_l479_479413


namespace rajesh_walked_distance_l479_479517

theorem rajesh_walked_distance (H : ℝ) (D_R : ℝ) 
  (h1 : D_R = 4 * H - 10)
  (h2 : H + D_R = 25) :
  D_R = 18 :=
by
  sorry

end rajesh_walked_distance_l479_479517


namespace domain_tan_2x_minus_pi_over_6_l479_479955

def tan_undefined (θ : ℝ) : Prop := ∃ k : ℤ, θ = (π / 2) + k * π

theorem domain_tan_2x_minus_pi_over_6 :
  ∀ x : ℝ, ¬ tan_undefined (2 * x - π / 6) ↔ ¬ ∃ k : ℤ, x = (π / 3) + k * (π / 2) :=
by sorry

end domain_tan_2x_minus_pi_over_6_l479_479955


namespace distance_M_to_midpoint_KL_l479_479876

-- Define the lengths and the points on the sides of triangle ABC
variables (AK KB BL LC CM MA : ℝ)

-- Given lengths of segments on sides of the triangle
axiom AK_eq_5 : AK = 5
axiom KB_eq_3 : KB = 3
axiom BL_eq_2 : BL = 2
axiom LC_eq_7 : LC = 7
axiom CM_eq_1 : CM = 1
axiom MA_eq_6 : MA = 6

-- The goal is to prove that the distance from point M to the midpoint of KL is \frac{1}{2} \sqrt{\frac{3529}{21}}
theorem distance_M_to_midpoint_KL :
  let KL_midpoint_distance := (1 / 2) * Real.sqrt (3529 / 21) in
  KL_midpoint_distance = (1 / 2) * Real.sqrt (3529 / 21) :=
by
  -- Variables (AK, KB, BL, LC, CM, MA) are given, and conditions are applied;
  -- the theorem's proof would show that the distance matches the given result.
  -- This is currently just a placeholder, the actual proof is not inferred by Lean.
  sorry

end distance_M_to_midpoint_KL_l479_479876


namespace total_boxes_produced_l479_479618

-- Define the rates of machines A and B
def rate_A (x : ℝ) : ℝ := x / 10
def rate_B (x : ℝ) : ℝ := 2 * x / 5

-- Combined rate of machines A and B
def combined_rate (x : ℝ) : ℝ := rate_A x + rate_B x

-- Time they work together
def time := 14

-- Prove that the total boxes produced is 7x
theorem total_boxes_produced (x : ℝ) : combined_rate x * time = 7 * x :=
by
  -- proof will be inserted here
  sorry

end total_boxes_produced_l479_479618


namespace problem_solving_example_l479_479394

theorem problem_solving_example (α β : ℝ) (h1 : α + β = 3) (h2 : α * β = 1) (h3 : α^2 - 3 * α + 1 = 0) (h4 : β^2 - 3 * β + 1 = 0) :
  7 * α^5 + 8 * β^4 = 1448 :=
sorry

end problem_solving_example_l479_479394


namespace four_lines_intersections_l479_479891

theorem four_lines_intersections :
  ∃ S : set ℕ, S = {0, 1, 3, 4, 5, 6} ∧
    ∀ (l1 l2 l3 l4 : ℝ → ℝ → Prop),
      (∀ i j, i ≠ j → ∃ P : ℝ × ℝ, l1 P ∧ l2 P ∧ l3 P ∧ l4 P → 
        (∃ n ∈ S, number_of_intersections l1 l2 l3 l4 = n)) :=
begin
  sorry
end

end four_lines_intersections_l479_479891


namespace arithmetic_sequence_terms_l479_479123

theorem arithmetic_sequence_terms (a d n : ℕ) 
  (h_sum_first_3 : 3 * a + 3 * d = 34)
  (h_sum_last_3 : 3 * a + 3 * d * (n - 1) = 146)
  (h_sum_all : n * (2 * a + (n - 1) * d) = 2 * 390) : 
  n = 13 :=
by
  sorry

end arithmetic_sequence_terms_l479_479123


namespace judys_school_week_l479_479163

theorem judys_school_week
  (pencils_used : ℕ)
  (packs_cost : ℕ)
  (total_cost : ℕ)
  (days_period : ℕ)
  (pencils_per_pack : ℕ)
  (pencils_in_school_days : ℕ)
  (total_pencil_use : ℕ) :
  (total_cost / packs_cost * pencils_per_pack = total_pencil_use) →
  (total_pencil_use / days_period = pencils_used) →
  (pencils_in_school_days / pencils_used = 5) :=
sorry

end judys_school_week_l479_479163


namespace picnic_condition_a_picnic_condition_b_l479_479997

def village_A := (0 : ℝ, 0 : ℝ)
def village_B := (3 : ℝ, 0 : ℝ)
def radius := 2

def circle (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2

def union_of_circles (p : ℝ × ℝ) :=
  circle village_A radius p ∨ circle village_B radius p

def intersection_of_circles (p : ℝ × ℝ) :=
  circle village_A radius p ∧ circle village_B radius p

theorem picnic_condition_a (p : ℝ × ℝ) :
  (∃ (p : ℝ × ℝ), p ∈ set_of union_of_circles ) :=
sorry

theorem picnic_condition_b (p : ℝ × ℝ) :
  (∃ (p : ℝ × ℝ), p ∈ set_of intersection_of_circles ) :=
sorry

end picnic_condition_a_picnic_condition_b_l479_479997


namespace find_cost_price_l479_479951

theorem find_cost_price (SP : ℝ) (loss_percent : ℝ) (CP : ℝ) (h1 : SP = 1260) (h2 : loss_percent = 16) : CP = 1500 :=
by
  sorry

end find_cost_price_l479_479951


namespace melanie_phil_ages_l479_479820

theorem melanie_phil_ages (A B : ℕ) 
  (h : (A + 10) * (B + 10) = A * B + 400) :
  (A + 6) + (B + 6) = 42 :=
by
  sorry

end melanie_phil_ages_l479_479820


namespace minimize_distance_optimal_C_l479_479923

variable (l : ℝ)

theorem minimize_distance (C : ℝ) (H1 : 0 ≤ C) (H2 : C ≤ l) :
  let A := 0
  let B := l
  let x : ℝ := C
  let distance := l^2 + (3 * (2*x - l)^2) / 4 in
  distance ≥ l^2 :=
begin
  sorry
end

theorem optimal_C (H : 0 ≤ l): ∃ C, C = l / 2 :=
begin
  use l / 2,
  sorry
end

end minimize_distance_optimal_C_l479_479923


namespace value_of_expression_l479_479126

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end value_of_expression_l479_479126


namespace sequence_2020th_term_l479_479304

theorem sequence_2020th_term :
  (∀ n : ℕ, (x : ℕ → ℤ), 
    x 1 = 2 ∧ (∀ k : ℕ, x (k + 1) = x k ^ 2 - 2 * x k + 1) → x 2020 = 1) :=
  sorry

end sequence_2020th_term_l479_479304


namespace trigonometric_identity_l479_479043

-- Definition for the given condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = 2

-- The proof goal
theorem trigonometric_identity (α : ℝ) (h : tan_alpha α) : 
  Real.cos (π + α) * Real.cos (π / 2 + α) = 2 / 5 :=
by
  sorry

end trigonometric_identity_l479_479043


namespace four_digit_integer_l479_479230

theorem four_digit_integer (a b c d : ℕ) 
  (h1 : a + b + c + d = 16) 
  (h2 : b + c = 10) 
  (h3 : a - d = 2)
  (h4 : (a - b + c - d) % 11 = 0) 
  : 1000 * a + 100 * b + 10 * c + d = 4642 := 
begin
  sorry
end

end four_digit_integer_l479_479230


namespace find_m_interval_l479_479343

def seq (x : ℕ → ℚ) : Prop :=
  (x 0 = 7) ∧ (∀ n : ℕ, x (n + 1) = (x n ^ 2 + 8 * x n + 9) / (x n + 7))

def m_spec (x : ℕ → ℚ) (m : ℕ) : Prop :=
  (x m ≤ 5 + 1 / 2^15)

theorem find_m_interval :
  ∃ (x : ℕ → ℚ) (m : ℕ), seq x ∧ m_spec x m ∧ 81 ≤ m ∧ m ≤ 242 :=
sorry

end find_m_interval_l479_479343


namespace find_length_of_tiger_l479_479655

noncomputable def length_of_tiger (L : ℝ) : Prop :=
  (∀ t : ℝ, t > 0 → t = 1 → L / t = L) ∧
  (∀ θ : ℝ, θ = π / 6 → ∀ d : ℝ, d = 20 * Real.cos θ → ∀ T : ℝ, T = 5 → L = d / T)

theorem find_length_of_tiger : ∃ L : ℝ, length_of_tiger L ∧ L = 2 * Real.sqrt 3 :=
sorry

end find_length_of_tiger_l479_479655


namespace polyhedron_volume_l479_479463

noncomputable def volume_polyhedron (A B C A1 B1 C1 : ℝ^3) : ℝ :=
  -- Formula for the volume based on given vertices needs definition
  sorry

-- Assume all necessary conditions:
variables (A B C A1 B1 C1 : ℝ^3)
variables (h_eq_triangle : (dist A B = 1) ∧ (dist B C = 1) ∧ (dist C A = 1))
variables (h_perpendiculars : (dist A A1 = 4) ∧ (dist B B1 = 5) ∧ (dist C C1 = 6))
variables (h_plane_side : true) -- Points on the same side of the plane, assume it's always true for simplicity

theorem polyhedron_volume :
  volume_polyhedron A B C A1 B1 C1 = (3 * Real.sqrt 3) / 2 :=
by
  -- Proof steps go here
  sorry

end polyhedron_volume_l479_479463


namespace arithmetic_sequence_geometric_sequence_ratio_lambda_exists_l479_479048

-- Problem 1
theorem arithmetic_sequence (a : ℕ → ℝ) (k : ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_eq : ∀ n, a (n + 1) ^ 2 = a n * a (n + 2) + (a 2 - a 1) ^ 2) :
  a 1 + a 3 = 2 * a 2 := 
sorry

-- Problem 2
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_eq : ∀ n, a (n + 1) ^ 2 = a n * a (n + 2))
  (h_arith : 2 * a 4 = a 2 + a 5) :
  ∃ q, q = 1 ∨ q = (1 + Real.sqrt 5) / 2 ∧ a 2 = q * a 1 :=
sorry

-- Problem 3
theorem lambda_exists (a b k : ℝ) :
  ∃ λ, λ = (a^2 + b^2 - k) / (a * b) ∧ ∀ n, a n + a (n + 2) = λ * a (n + 1) := 
sorry

end arithmetic_sequence_geometric_sequence_ratio_lambda_exists_l479_479048


namespace inverse_of_157_mod_263_l479_479332

theorem inverse_of_157_mod_263 :
  ∃ b : ℤ, (157 * b ≡ 1 [MOD 263]) ∧ b ≡ 197 [MOD 263] :=
by
  sorry

end inverse_of_157_mod_263_l479_479332


namespace new_machine_rate_l479_479645

-- Define the conditions as assumptions
variables (R : ℝ) -- Rate of the new machine in bolts per hour
variables (time_minutes : ℝ) (total_bolts : ℝ) (old_rate : ℝ)

-- Given conditions
def conditions := (time_minutes = 96) ∧ (total_bolts = 400) ∧ (old_rate = 100)

-- Theorem to prove the rate of the new machine
theorem new_machine_rate (h : conditions) : R = 150 :=
by
  sorry

end new_machine_rate_l479_479645


namespace find_f_log_l479_479386

def even_function (f : ℝ → ℝ) :=
  ∀ (x : ℝ), f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ (x : ℝ), f (x + p) = f x

theorem find_f_log (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 2)
  (h_condition : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 0 → f x = 3 * x + 4 / 9) :
  f (Real.log 5 / Real.log (1 / 3)) = -5 / 9 :=
by
  sorry

end find_f_log_l479_479386


namespace is_geometric_sequence_general_term_a_n_sum_first_n_terms_l479_479741

-- Definition of the sequence {a_n}
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * (a n) + n

-- (1) Proving that {a_n + n} is a geometric sequence
theorem is_geometric_sequence (n : ℕ) :
  (a (n + 1) + (n + 1)) = 2 * (a n + n) :=
sorry

-- (2) Finding the general term formula for {a_n}
theorem general_term_a_n (n : ℕ) :
  a n = 2^n - n :=
sorry

-- Definition of the sequence {b_n}
def b (n : ℕ) : ℚ := (2 * n - 1) / 2^n

-- Sum of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℚ := finset.sum (finset.range n.succ) (λ i, b i)

-- (3) Finding the sum of the first n terms {b_n}
theorem sum_first_n_terms (n : ℕ) :
  S n = 3 - (2 * n + 3) / 2^n :=
sorry

end is_geometric_sequence_general_term_a_n_sum_first_n_terms_l479_479741


namespace simplify_expression_l479_479630

theorem simplify_expression (h : (Real.pi / 2) < 2 ∧ 2 < Real.pi) : 
  Real.sqrt (1 - 2 * Real.sin 2 * Real.cos 2) = Real.sin 2 - Real.cos 2 :=
sorry

end simplify_expression_l479_479630


namespace selection_with_median_l479_479143

theorem selection_with_median (students : Fin 19 → ℝ) (h_diff : ∀ i j, i ≠ j → students i ≠ students j) (Xiaohong_height : ℝ) :
  (Xiaohong_height ≥ (students (Fin.of_nat 9))) ↔ (∃ i, i < 10 ∧ students i = Xiaohong_height) :=
sorry

end selection_with_median_l479_479143


namespace mass_percentage_O_mixture_l479_479027

noncomputable def molar_mass_Al2O3 : ℝ := (2 * 26.98) + (3 * 16.00)
noncomputable def molar_mass_Cr2O3 : ℝ := (2 * 51.99) + (3 * 16.00)
noncomputable def mass_of_O_in_Al2O3 : ℝ := 3 * 16.00
noncomputable def mass_of_O_in_Cr2O3 : ℝ := 3 * 16.00
noncomputable def mass_percentage_O_in_Al2O3 : ℝ := (mass_of_O_in_Al2O3 / molar_mass_Al2O3) * 100
noncomputable def mass_percentage_O_in_Cr2O3 : ℝ := (mass_of_O_in_Cr2O3 / molar_mass_Cr2O3) * 100
noncomputable def mass_percentage_O_in_mixture : ℝ := (0.50 * mass_percentage_O_in_Al2O3) + (0.50 * mass_percentage_O_in_Cr2O3)

theorem mass_percentage_O_mixture : mass_percentage_O_in_mixture = 39.325 := by
  sorry

end mass_percentage_O_mixture_l479_479027


namespace farm_horses_cows_l479_479282

variables (H C : ℕ)

theorem farm_horses_cows (H C : ℕ) (h1 : H = 6 * C) (h2 : (H - 15) = 3 * (C + 15)) : (H - 15) - (C + 15) = 70 :=
by {
  sorry
}

end farm_horses_cows_l479_479282


namespace positive_difference_solutions_l479_479715

theorem positive_difference_solutions : 
  (abs (sqrt 162 - (-sqrt 162))) = 18 * sqrt 2 :=
by
  sorry

end positive_difference_solutions_l479_479715


namespace sum_and_product_of_solutions_l479_479718

theorem sum_and_product_of_solutions (x : ℤ) :
  (∀ (x : ℤ), x^4 - 36 * x^2 + 225 = 0 → (x = 5 ∨ x = -5 ∨ x = 3 ∨ x = -3)) →
  (finset.sum (List.toFinset [-5, -3, 3, 5]) id = 0) ∧
  (finset.prod (List.toFinset [-5, -3, 3, 5]) id = 225) :=
by {
  sorry
}

end sum_and_product_of_solutions_l479_479718


namespace divide_friends_among_teams_l479_479106

theorem divide_friends_among_teams :
  let friends_num := 8
  let teams_num := 4
  (teams_num ^ friends_num) = 65536 := by
  sorry

end divide_friends_among_teams_l479_479106


namespace haley_car_distance_l479_479971

theorem haley_car_distance (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used_gallons : ℕ) (distance_covered_miles : ℕ) :
  fuel_ratio = 4 → distance_ratio = 7 → fuel_used_gallons = 44 → distance_covered_miles = (distance_ratio * fuel_used_gallons / fuel_ratio)
  → distance_covered_miles = 77 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  dsimp at h4
  linarith

end haley_car_distance_l479_479971


namespace find_x_l479_479791

variables (x : ℝ)
def a := (2, 4)
def b := (x, 3)

def dot_product (u v : ℝ × ℝ) : ℝ := u.fst * v.fst + u.snd * v.snd

theorem find_x (h : dot_product (a.1 - b.1, a.2 - b.2) b = 0) : x = -1 ∨ x = 3 :=
by
  sorry

end find_x_l479_479791


namespace rectangle_area_error_percentage_l479_479860

theorem rectangle_area_error_percentage (L W : ℝ) : 
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let actual_area := L * W
  let measured_area := measured_length * measured_width
  let error := measured_area - actual_area
  let error_percentage := (error / actual_area) * 100
  error_percentage = 0.7 := 
by
  sorry

end rectangle_area_error_percentage_l479_479860


namespace prize_winner_l479_479144

def contestant_won_prize (A B C D : Prop) (statement_A : Prop) (statement_B : Prop) (statement_C : Prop) (statement_D : Prop) : Prop :=
  ∃ x, (x = A ∨ x = B ∨ x = C ∨ x = D) ∧ ((statement_A ↔ A = B ∨ A = C) → false) ∧ (statement_B ↔ ¬A ∧ ¬C) ∧ (statement_C ↔ C) ∧ (statement_D ↔ B) ∧
       (by cases' [A, B, C, D]; exact statement_A ↔ (A ∨ B ∨ C ∨ ⊥) → false)

-- Define each statement for the contestants
variables (A B C D : Prop)

def statement_A := B ∨ C
def statement_B := ¬A ∧ ¬C
def statement_C := C
def statement_D := B

-- The theorem which proves who won the prize from the given conditions
theorem prize_winner : 
  contestant_won_prize A B C D statement_A statement_B statement_C statement_D → C :=
sorry

end prize_winner_l479_479144


namespace louie_payment_is_329_l479_479183

noncomputable def louie_monthly_payment : ℕ :=
  let P := 1000
  let r := 0.12
  let n := 1
  let t := 0.5
  let FV := P * (1 + r/n)^(n*t*12)
  let monthly_payment := FV / 6
  Int.toNat (Float.floor (monthly_payment + 0.5)) -- rounding to the nearest dollar

theorem louie_payment_is_329 : louie_monthly_payment = 329 :=
  by
  -- Here, the proof would generally go to show that the calculated monthly_payment rounds to 329.
  -- Proof steps would consist of explicitly calculating FV and then the monthly payment division,
  -- and demonstrating the result is roughly 328.97, and upon rounding, it becomes 329.
  sorry

end louie_payment_is_329_l479_479183


namespace solve_quadratic_eq_solve_equal_squares_l479_479938

theorem solve_quadratic_eq (x : ℝ) : 
    (4 * x^2 - 2 * x - 1 = 0) ↔ 
    (x = (1 + Real.sqrt 5) / 4 ∨ x = (1 - Real.sqrt 5) / 4) := 
by
  sorry

theorem solve_equal_squares (y : ℝ) :
    ((y + 1)^2 = (3 * y - 1)^2) ↔ 
    (y = 1 ∨ y = 0) := 
by
  sorry

end solve_quadratic_eq_solve_equal_squares_l479_479938


namespace min_sum_radii_l479_479466

theorem min_sum_radii (x y : ℝ)
  (h_ap : 3 * x = dist A P)
  (h_bq : 3 * y = dist B Q)
  (h_tetrahedron : regular_tetrahedron A B C D 1) :
  x + y = \(\frac{\sqrt{6}-1}{5}\) :=
sorry

end min_sum_radii_l479_479466


namespace distance_from_point_to_line_l479_479538

theorem distance_from_point_to_line 
    (a b c x1 y1 : ℝ)
    (h_line : a * x1 + b * y1 + c = 0) :
    let d := |a * x1 + b * y1 + c| / sqrt (a^2 + b^2) in
    a = 1 ∧ b = 1 ∧ c = -1 ∧ x1 = 1 ∧ y1 = 1 → d = sqrt 2 / 2 :=
by
  intro a b c x1 y1 h_line
  dsimp
  intro h
  rcases h with ⟨ha, hb, hc, hx1, hy1⟩
  rw [ha, hb, hc, hx1, hy1]
  have : d = |1 * 1 + 1 * 1 - 1| / sqrt(1^2 + 1^2) := rfl
  norm_num
  exact sqrt 2 / 2

#check distance_from_point_to_line

end distance_from_point_to_line_l479_479538


namespace conjugate_of_z_l479_479761

noncomputable def z : ℂ := (1 - complex.I) / (1 + complex.I)

theorem conjugate_of_z :
  (conj z) = complex.I :=
by
  -- The detailed proof is omitted
  sorry

end conjugate_of_z_l479_479761


namespace sum_of_distinct_real_values_l479_479346

theorem sum_of_distinct_real_values (x : ℝ) (h : abs_sum_2017_abs_x_eq_1 x) : 
  x = 1 / 2017 ∨ x = 1 / 2015 → 
  ∑ x, (x = 1 / 2017 ∨ x = 1 / 2015) = 4032 / (2017 * 2015) :=
by 
  sorry

-- Definition to represent the condition
def abs_sum_2017_abs_x_eq_1 (x : ℝ) : Prop :=
  abs (abs (.. (2017 times) .. abs x + x + .. x + x + x)) = 1
  

end sum_of_distinct_real_values_l479_479346


namespace profit_percentage_correct_l479_479663

-- Definitions based on the conditions given
def cost_price := 95
def marked_price := 125
def discount_rate : ℝ := 0.05

-- Calculate discount
def discount := discount_rate * marked_price

-- Calculate selling price
def selling_price := marked_price - discount

-- Calculate profit
def profit := selling_price - cost_price

-- Calculate profit percentage
def profit_percentage := (profit / cost_price) * 100

theorem profit_percentage_correct : profit_percentage = 25 := by
  -- Definitions expanded
  -- Using definitions to get to the final form
  have discount_def : discount = discount_rate * marked_price := rfl
  have selling_price_def : selling_price = marked_price - discount := rfl
  have profit_def : profit = selling_price - cost_price := rfl
  have profit_percentage_def : profit_percentage = (profit / cost_price) * 100 := rfl
  -- Calculation steps as per proof
  rw [discount_def, selling_price_def, profit_def, profit_percentage_def]
  have : discount = 6.25 := by norm_num
  have : selling_price = 118.75 := by norm_num
  have : profit = 23.75 := by norm_num
  have : profit_percentage = 25 := by norm_num
  assumption -- final step to conclude the proof

end profit_percentage_correct_l479_479663


namespace problem_part_one_problem_part_two_l479_479456

variable {A B C : ℝ}
variable {a b c : ℝ}

def is_acute_triangle (A B C : ℝ) : Prop :=
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2

theorem problem_part_one
  (h₀ : a = 2 * b * Real.sin A)
  (hac : is_acute_triangle A B C)
  (h₁ : a = 3 * Real.sqrt 3)
  (h₂ : c = 5) :
  B = π / 6 :=
sorry

theorem problem_part_two
  (h₀ : a = 3 * Real.sqrt 3)
  (h₁ : c = 5)
  (h₂ : B = π / 6) :
  1 / 2 * a * c * Real.sin B = 15 * Real.sqrt 3 / 4 ∧
  (c^2 + a^2 - 2 * c * a * Real.cos (π / 6) = 7) :=
sorry


end problem_part_one_problem_part_two_l479_479456


namespace perimeter_of_rectangle_l479_479571

-- Define the properties of the rectangle based on the given conditions
variable (l w : ℝ)
axiom h1 : l + w = 7
axiom h2 : 2 * l + w = 9.5

-- Define the function for perimeter of the rectangle
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

-- Formal statement of the proof problem
theorem perimeter_of_rectangle : perimeter l w = 14 := by
  -- Given conditions
  have h3 : l = 2.5 := sorry
  have h4 : w = 4.5 := sorry
  -- Conclusion based on the conditions
  show perimeter l w = 14 from sorry

end perimeter_of_rectangle_l479_479571


namespace range_of_a_l479_479035

theorem range_of_a (a x y : ℝ) (h1: x + 3 * y = 2 + a) (h2: 3 * x + y = -4 * a) (h3: x + y > 2) : a < -2 :=
sorry

end range_of_a_l479_479035


namespace angle_B_triangle_perimeter_l479_479821

def triangle_angles (A B C : ℕ) := A + B + C = 180

def area (a c B : ℝ) := (1/2) * a * c * real.sin B = real.sqrt 3

theorem angle_B (a b c B C : ℝ) (h1 : (2 * a - c) * real.cos B = b * real.cos C) :
    B = 60 :=
begin
  sorry
end

theorem triangle_perimeter (a b c : ℝ) (h1 : (1/2) * a * c * real.sin 60 = real.sqrt 3)
  (h2 : a + c = 6) (h3 : b = real.sqrt 24) : 
    a + b + c = 6 + 2 * real.sqrt 6 :=
begin
  sorry
end

end angle_B_triangle_perimeter_l479_479821


namespace sequence_property_l479_479004

/-- Define the sequence a -/
def a : ℕ → ℤ
| 1     := 1
| 2     := b
| (n+2) := 2 * a (n + 1) - a n + 2

/-- Prove the given mathematical problem -/
theorem sequence_property (b : ℤ) (hb : 0 < b) : ∀ n : ℕ, ∃ m : ℕ, a n * a (n + 1) = a m :=
by
  sorry

end sequence_property_l479_479004


namespace problem_statement_l479_479391

open Set

variable (a : ℕ)
variable (A : Set ℕ := {2, 3, 4})
variable (B : Set ℕ := {a + 2, a})

theorem problem_statement (hB : B ⊆ A) : (A \ B) = {3} :=
sorry

end problem_statement_l479_479391


namespace solution_set_of_inequality_l479_479569

theorem solution_set_of_inequality {x : ℝ} :
  {x | |x| * (1 - 2 * x) > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | 0 < x ∧ x < 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l479_479569


namespace points_for_blue_zone_l479_479580

noncomputable def calculate_points_blue_zone (points_bullseye : ℝ) (ratio : ℝ) : ℝ :=
  points_bullseye / ratio

theorem points_for_blue_zone 
  (radius_bullseye : ℝ) 
  (points_bullseye : ℝ) 
  (width_ring : ℝ) 
  (area_blue : ℝ) 
  (area_bullseye : ℝ) 
  (prob_ratio : ℝ) 
  (ratio_points : ℝ)
  (inner_radius_blue : ℝ)
  (outer_radius_blue : ℝ): 
  radius_bullseye = 1 →
  points_bullseye = 315 →
  width_ring = radius_bullseye →
  inner_radius_blue = 3 →
  outer_radius_blue = 4 →
  area_blue = (real.pi * (outer_radius_blue^2 - inner_radius_blue^2)) →
  area_bullseye = (real.pi * (radius_bullseye^2)) →
  prob_ratio = area_blue / area_bullseye →
  ratio_points = 1 / prob_ratio →
  calculate_points_blue_zone points_bullseye prob_ratio = 45 :=
by
  intros h_radius_bullseye h_points_bullseye h_width_ring h_inner_radius_blue h_outer_radius_blue 
         h_area_blue h_area_bullseye h_prob_ratio h_ratio_points
  sorry

end points_for_blue_zone_l479_479580


namespace increasing_m_range_l479_479439

noncomputable def f (x m : ℝ) : ℝ := x^2 + Real.log x - 2 * m * x

theorem increasing_m_range (m : ℝ) : 
  (∀ x > 0, (2 * x + 1 / x - 2 * m ≥ 0)) → m ≤ Real.sqrt 2 :=
by
  intros h
  -- Proof steps would go here
  sorry

end increasing_m_range_l479_479439


namespace grid_is_valid_l479_479832

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Check if all adjacent cells (by side or diagonal) in a 3x3 grid are coprime -/
def valid_grid (grid : Array (Array ℕ)) : Prop :=
  let adjacent_pairs := 
    [(0, 0, 0, 1), (0, 1, 0, 2), 
     (1, 0, 1, 1), (1, 1, 1, 2), 
     (2, 0, 2, 1), (2, 1, 2, 2),
     (0, 0, 1, 0), (0, 1, 1, 1), (0, 2, 1, 2),
     (1, 0, 2, 0), (1, 1, 2, 1), (1, 2, 2, 2),
     (0, 0, 1, 1), (0, 1, 1, 2), 
     (1, 0, 2, 1), (1, 1, 2, 2), 
     (0, 2, 1, 1), (1, 2, 2, 1), 
     (0, 1, 1, 0), (1, 1, 2, 0)] 
  adjacent_pairs.all (λ ⟨r1, c1, r2, c2⟩, is_coprime (grid[r1][c1]) (grid[r2][c2]))

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10],
    #[5, 7, 11],
    #[6, 13, 12]]

theorem grid_is_valid : valid_grid grid :=
  by
    -- Proof omitted, placeholder
    sorry

end grid_is_valid_l479_479832


namespace expansion_of_product_l479_479701

theorem expansion_of_product (x : ℝ) :
  (7 * x + 3) * (5 * x^2 + 2 * x + 4) = 35 * x^3 + 29 * x^2 + 34 * x + 12 := 
by
  sorry

end expansion_of_product_l479_479701


namespace contrapositive_of_square_inequality_l479_479532

theorem contrapositive_of_square_inequality (x y : ℝ) :
  (x > y → x^2 > y^2) ↔ (x^2 ≤ y^2 → x ≤ y) :=
sorry

end contrapositive_of_square_inequality_l479_479532


namespace find_ab_l479_479623

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 :=
by
  sorry -- Proof to be provided

end find_ab_l479_479623


namespace remainder_x_pow_150_mod_x_minus_1_pow_3_l479_479676

theorem remainder_x_pow_150_mod_x_minus_1_pow_3 :
  ∃ (p : Polynomial ℤ), (p ≡ 11175 * X ^ 2 - 22200 * X + 11026 [X] % (X - 1) ^ 3) ∧ p = X ^ 150 :=
sorry

end remainder_x_pow_150_mod_x_minus_1_pow_3_l479_479676


namespace part1_part2_l479_479446

variable (A B C a b c : ℝ)
variable (sin cos : ℝ → ℝ)

-- Prove B = π/3 from the given equation
theorem part1 (h : b * sin A = sqrt 3 * a * cos B) : B = (π / 3) := by
  sorry

-- Prove the area of triangle ABC with given values of a, b, and B
theorem part2 (h : b = 3) (h1 : a = 2) (h2 : B = π / 3) :
  let c : ℝ := 1 + sqrt 6
  let area := (1 / 2) * a * c * (sqrt 3 / 2)
  area = (sqrt 3 + 3 * sqrt 2) / 2 := by
  sorry

end part1_part2_l479_479446


namespace maximal_possible_planes_l479_479575

-- Define the conditions
constant n : ℕ -- number of lines
constant lines : finset (ℝ × ℝ × ℝ × ℝ) -- each line can be represented by a point and a direction vector

-- Assume conditions
axiom distinct_lines : lines.card = n
axiom no_two_parallel : ∀ l1 l2 ∈ lines, l1 ≠ l2 → ¬ ∃ k : ℝ, k ≠ 0 ∧ (l2.2 = k • l1.2)
axiom no_three_meet_at_one_point : ∀ l1 l2 l3 ∈ lines, l1 ≠ l2 → l2 ≠ l3 → l1 ≠ l3 → ¬ coplanar l1 l2 l3

-- Define the proof problem
theorem maximal_possible_planes : (n.choose 2) = n * (n - 1) / 2 := 
by sorry

end maximal_possible_planes_l479_479575


namespace num_values_ffx_eq_0_l479_479807

def f (x : ℝ) : ℝ :=
if x ≥ -3 then x^2 - 9 else x + 6

theorem num_values_ffx_eq_0 : 
  let f (x : ℝ) : ℝ := if x ≥ -3 then x^2 - 9 else x + 6 
  in (card {x : ℝ | f (f x) = 0}) = 3 :=
by
  -- Proof is not provided
  sorry

end num_values_ffx_eq_0_l479_479807


namespace diameter_of_circular_field_l479_479025

noncomputable def diameter (C : ℝ) : ℝ := C / Real.pi

theorem diameter_of_circular_field :
  let cost_per_meter := 3
  let total_cost := 376.99
  let circumference := total_cost / cost_per_meter
  diameter circumference = 40 :=
by
  let cost_per_meter : ℝ := 3
  let total_cost : ℝ := 376.99
  let circumference : ℝ := total_cost / cost_per_meter
  have : circumference = 125.66333333333334 := by sorry
  have : diameter circumference = 40 := by sorry
  sorry

end diameter_of_circular_field_l479_479025


namespace quadratic_intersection_l479_479075

theorem quadratic_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersection_l479_479075


namespace smallest_multiple_of_7_greater_than_500_l479_479365

theorem smallest_multiple_of_7_greater_than_500 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n > 500 ∧ n = 504 := 
by
  sorry

end smallest_multiple_of_7_greater_than_500_l479_479365


namespace asymptotes_of_hyperbola_l479_479539

theorem asymptotes_of_hyperbola (a b x y : ℝ) (h : a = 5 ∧ b = 2) :
  (x^2 / 25 - y^2 / 4 = 1) → (y = (2 / 5) * x ∨ y = -(2 / 5) * x) :=
by
  sorry

end asymptotes_of_hyperbola_l479_479539


namespace complement_P_eq_Ioo_l479_479422

def U : Set ℝ := Set.univ
def P : Set ℝ := { x | x^2 - 5 * x - 6 ≥ 0 }
def complement_of_P_in_U : Set ℝ := Set.Ioo (-1) 6

theorem complement_P_eq_Ioo :
  (U \ P) = complement_of_P_in_U :=
by sorry

end complement_P_eq_Ioo_l479_479422


namespace area_triangle_DEF_l479_479148

structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ)
  (AB_length : dist A B = AB)
  (BC_length : dist B C = BC)
  (AC_length : dist A C = AC)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ := (p1.2 - p2.2) / (p1.1 - p2.1)

def perpendicular (m1 m2 : ℝ) := m1 * m2 = -1

structure PointOnLine (P1 P2 : ℝ × ℝ) :=
  (P : ℝ × ℝ)
  (on_line : ∃ t : ℝ, P = (P1.1 + t * (P2.1 - P1.1), P1.2 + t * (P2.2 - P1.2)))

theorem area_triangle_DEF (A B C : ℝ × ℝ) (AB BC AC : ℝ)
  (triangle_ABC : Triangle)
  (midpoint_D : midpoint A C = (75, 0))
  (E_on_BC : PointOnLine B C E)
  (F_on_AB : PointOnLine A B F)
  (perpendicular_DE_BC : perpendicular (slope (midpoint A C) E) (slope B C))
  (perpendicular_DF_AB : perpendicular (slope (midpoint A C) F) (slope A B)) :
  let area := (1 / 2) * abs ((midpoint A C).1 * (E.2 - F.2) + E.1 * (F.2 - (midpoint A C).2) + F.1 * ((midpoint A C).2 - E.2))
  in area = 378.96 := by
  sorry

end area_triangle_DEF_l479_479148


namespace sin_cos_expr_l479_479098

theorem sin_cos_expr (θ : ℝ) (h : (cot θ) ^ 2000 + 2 = sin θ + 1) : (sin θ + 2)^2 * (cos θ + 1) = 9 :=
sorry

end sin_cos_expr_l479_479098


namespace sum_of_numbers_l479_479970

noncomputable def sum_two_numbers (x y : ℝ) : ℝ :=
  x + y

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 16) (h2 : (1 / x) = 3 * (1 / y)) : 
  sum_two_numbers x y = (16 * Real.sqrt 3) / 3 := 
by 
  sorry

end sum_of_numbers_l479_479970


namespace maximum_chord_length_l479_479302

theorem maximum_chord_length (t : ℝ) (h_t : t^2 < 5) :
  let l := λ x : ℝ, (x + t)
  let ellipse := λ (x y : ℝ), (x^2) / 4 + y^2 = 1
  let chord_length := 2 * 1 * real.sqrt(1 - 5 / 4 * x^2 + t^2 / 4 + (t^2 - 1))
  max_chord_length = λ (|AB| maximum), |AB| :=
  maximum length_of |AB| = 4 * real.sqrt(10) / 5 :=
by
  sorry

end maximum_chord_length_l479_479302


namespace vertex_x_coord_l479_479959

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Conditions based on given points
def conditions (a b c : ℝ) : Prop :=
  quadratic a b c 2 = 4 ∧
  quadratic a b c 8 =4 ∧
  quadratic a b c 10 = 13

-- Statement to prove the x-coordinate of the vertex is 5
theorem vertex_x_coord (a b c : ℝ) (h : conditions a b c) : 
  (-(b) / (2 * a)) = 5 :=
by
  sorry

end vertex_x_coord_l479_479959


namespace circle_equation_l479_479357

-- Defining the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

-- Defining the point
def point : ℝ × ℝ := (1, -1)

-- Proving the equation of the new circle that passes through the intersection points 
-- of the given circles and the given point
theorem circle_equation (x y : ℝ) :
  (circle1 x y ∧ circle2 x y ∧ x = 1 ∧ y = -1) → 9 * x^2 + 9 * y^2 - 14 * x + 4 * y = 0 :=
sorry

end circle_equation_l479_479357


namespace value_of_a_and_simplified_expression_l479_479999

theorem value_of_a_and_simplified_expression (a : ℝ) (x y : ℝ) :
  let expr := 3 * (3 * x^2 + 4 * x * y) - a * (2 * x^2 + 3 * x * y - 1) in
  (∀ y, (expr.does_not_contain_y) → (a = 4) ∧ (expr = x^2 + 4)) :=
by sorry

end value_of_a_and_simplified_expression_l479_479999


namespace find_a_100_l479_479975

open Nat

-- Define the sequence a_n
def a : ℕ → ℕ
| 1       => 1
| (n + m) => a n + a m + n * m

-- The proof statement
theorem find_a_100 : a 100 = 5050 :=
sorry

end find_a_100_l479_479975


namespace interest_earned_l479_479215

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (hP : P = 1500) (hr : r = 0.02) (hn : n = 5) :
  let A := P * (1 + r)^n
  in (A - P) ≈ 156 :=
by
  sorry

end interest_earned_l479_479215


namespace percentage_increase_in_length_is_10_l479_479964

variables (L B : ℝ) -- original length and breadth
variables (length_increase_percentage breadth_increase_percentage area_increase_percentage : ℝ)

noncomputable def new_length (x : ℝ) : ℝ := L * (1 + x / 100)
noncomputable def new_breadth : ℝ := B * 1.25
noncomputable def new_area (x : ℝ) : ℝ := new_length L B x * new_breadth B
noncomputable def increased_area : ℝ := L * B * 1.375

theorem percentage_increase_in_length_is_10 :
 (breadth_increase_percentage = 25) →
 (area_increase_percentage = 37.5) →
 (new_area L B 10 = increased_area L B) → length_increase_percentage = 10
:= by
  sorry

end percentage_increase_in_length_is_10_l479_479964


namespace infinite_primes_dividing_at_least_one_term_l479_479898

def is_strictly_increasing (a : ℕ → ℕ) :=
  ∀ n : ℕ, a n < a (n + 1)

def polynomial_with_real_coefficients (f : ℝ → ℝ) := 
  ∃ a b : list ℝ, f = λ x => polynomial.eval x (polynomial.of_list a) / (polynomial.eval x (polynomial.of_list b))

def satisfies_condition (f : ℝ → ℝ) (a : ℕ → ℕ) :=
  ∀ n : ℕ, a n ≤ f n

theorem infinite_primes_dividing_at_least_one_term
(f : ℝ → ℝ) 
(hf : polynomial_with_real_coefficients f)
(a : ℕ → ℕ)
(ha : is_strictly_increasing a)
(h_condition : satisfies_condition f a) :
  ∃ infinitely_many_primes : ℕ → Prop, 
  (∀ p, Prime p → infinitely_many_primes p → ∃ n, p ∣ a n) ∧ 
  (Set.Infinite {p : ℕ | Prime p ∧ infinitely_many_primes p}) :=
sorry

end infinite_primes_dividing_at_least_one_term_l479_479898


namespace pump_problem_l479_479277

theorem pump_problem : ∃ (A : ℝ), (1 / A + 1 / 2 = 3 / 4) ∧ A = 4 := 
by 
  use 4
  split
  · rw [inv_div, inv_div]
    simp
  · ring
  sorry

end pump_problem_l479_479277


namespace probability_red_or_white_is_19_over_25_l479_479290

-- Definitions for the conditions
def totalMarbles : ℕ := 50
def blueMarbles : ℕ := 12
def redMarbles : ℕ := 18
def whiteMarbles : ℕ := totalMarbles - (blueMarbles + redMarbles)

-- Define the probability calculation
def probabilityRedOrWhite : ℚ := (redMarbles + whiteMarbles) / totalMarbles

-- The theorem we need to prove
theorem probability_red_or_white_is_19_over_25 :
  probabilityRedOrWhite = 19 / 25 :=
by
  -- Sorry to skip the proof
  sorry

end probability_red_or_white_is_19_over_25_l479_479290


namespace smallest_positive_multiple_of_45_l479_479605

theorem smallest_positive_multiple_of_45 : ∃ (n : ℕ), n > 0 ∧ ∃ (x : ℕ), x > 0 ∧ n = 45 * x ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l479_479605


namespace scientific_notation_9600000_l479_479960

theorem scientific_notation_9600000 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 9600000 = a * 10 ^ n ∧ a = 9.6 ∧ n = 6 :=
by
  exists 9.6
  exists 6
  simp
  sorry

end scientific_notation_9600000_l479_479960


namespace f_correct_l479_479664

def f (x : ℝ) : ℝ :=
  if x > 0 then 2 / x - 1
  else if x < 0 then 2 / x + 1
  else 0

theorem f_correct (a : ℝ) (h : f a = 3) : a = 1 / 2 :=
sorry

end f_correct_l479_479664


namespace length_AB_l479_479713

-- Define a triangle with the given conditions
structure Triangle :=
  (A B C : Type)
  (AC : ℝ)
  (angleBAC : ℝ)
  (AC_length : AC = 10 * real.sqrt 2)
  (angleBAC_value : angleBAC = 30)

-- The statement to prove
theorem length_AB (T : Triangle) :
  T.AC = 10 * real.sqrt 2 → T.angleBAC = 30 → (∃ AB : ℝ, AB = 5 * real.sqrt 2) :=
by
  sorry

end length_AB_l479_479713


namespace triangle_area_is_correct_l479_479262

-- Define the points
def point1 : (ℝ × ℝ) := (0, 3)
def point2 : (ℝ × ℝ) := (5, 0)
def point3 : (ℝ × ℝ) := (0, 6)
def point4 : (ℝ × ℝ) := (4, 0)

-- Define a function to calculate the area based on the intersection points
noncomputable def area_of_triangle (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  let slope1 := (p2.2 - p1.2) / (p2.1 - p1.1)
  let intercept1 := p1.2 - slope1 * p1.1
  let slope2 := (p4.2 - p3.2) / (p4.1 - p3.1)
  let intercept2 := p3.2 - slope2 * p3.1
  let x_intersect := (intercept2 - intercept1) / (slope1 - slope2)
  let y_intersect := slope1 * x_intersect + intercept1
  let base := x_intersect
  let height := y_intersect
  (1 / 2) * base * height

-- The proof problem statement in Lean
theorem triangle_area_is_correct :
  area_of_triangle point1 point2 point3 point4 = 5 / 3 :=
by
  sorry

end triangle_area_is_correct_l479_479262


namespace diane_stamp_arrangements_l479_479348

def diane_stamps := Σ (s1 : ℕ) (s2 : ℕ) (s4 : ℕ) (s10 : ℕ), 
  s1 * 1 + s2 * 2 + s4 * 4 + s10 * 10 = 15 ∧
  s1 ≤ 1 ∧ s2 ≤ 2 ∧ s4 ≤ 4 ∧ s10 ≤ 5

theorem diane_stamp_arrangements : 
    (∑ s in (Finset.univ : Finset diane_stamps), 1) = 45 := 
sorry

end diane_stamp_arrangements_l479_479348


namespace find_f_prime_at_1_l479_479815

noncomputable def f (x : ℝ) : ℝ := x^3 - f' 1 * x^2 + 3

theorem find_f_prime_at_1 : deriv f 1 = 1 := by
  sorry

end find_f_prime_at_1_l479_479815


namespace range_of_a_l479_479235

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 4 → (x^2 + 2 * (a - 1) * x + 2) ≥ (y^2 + 2 * (a - 1) * y + 2)) →
  a ≤ -3 :=
by
  sorry

end range_of_a_l479_479235


namespace find_d_l479_479894

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := c * x + 1

theorem find_d (c d : ℝ) (hx : ∀ x, f (g x c) c = 15 * x + d) : d = 8 :=
sorry

end find_d_l479_479894


namespace area_of_triangle_ABC_is_169_div_sqrt_21_l479_479295

/-
A circle of radius 2 is tangent to a circle of radius 3. The sides of ΔABC are tangent to these circles, and the sides AB and AC are congruent.
-/
variables {O O' : Type}
variables (r1 r2 : ℝ) (A B C : Type) [metric_space O] [metric_space O']
variables (radius2_circle : ∀ (x : O), dist x (O : O) = 2)
variables (radius3_circle : ∀ (y : O'), dist y (O' : O') = 3)
variables (tangent_to_2 : ∀ (x : A) (y : O), dist x y = 2 → ∃ (z : O), x = z ∧ dist y z = 2)
variables (tangent_to_3 : ∀ (x : A) (y : O'), dist x y = 3 → ∃ (z : O'), x = z ∧ dist y z = 3)
variables (ab_congruent_ac : dist A B = dist A C)

theorem area_of_triangle_ABC_is_169_div_sqrt_21 :
  ∃ (ΔABC : Type), ΔABC = sqrt21 (169 : ℝ) :=
sorry

end area_of_triangle_ABC_is_169_div_sqrt_21_l479_479295


namespace range_positive_of_odd_increasing_l479_479052

-- Define f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Define f as an increasing function on (-∞,0)
def is_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < 0 → f (x) < f (y)

-- Given an odd function that is increasing on (-∞,0) and f(-1) = 0, prove the range of x for which f(x) > 0 is (-1, 0) ∪ (1, +∞)
theorem range_positive_of_odd_increasing (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_increasing : is_increasing_on_neg f)
  (h_f_neg_one : f (-1) = 0) :
  {x : ℝ | f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end range_positive_of_odd_increasing_l479_479052


namespace number_of_spacy_subsets_l479_479800

noncomputable def spacy_subsets : ℕ → ℕ
| 1 := 2
| 2 := 3
| 3 := 4
| n := spacy_subsets (n - 1) + spacy_subsets (n - 3)

theorem number_of_spacy_subsets : spacy_subsets 15 = 406 :=
by sorry

end number_of_spacy_subsets_l479_479800


namespace value_of_expression_l479_479435

variable {a b c d e f : ℝ}

theorem value_of_expression :
  a * b * c = 130 →
  b * c * d = 65 →
  c * d * e = 1000 →
  d * e * f = 250 →
  (a * f) / (c * d) = 1 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end value_of_expression_l479_479435


namespace sum_of_max_min_elements_l479_479785

theorem sum_of_max_min_elements (T : Set (ℝ × ℝ)) :
  T = { p : ℝ × ℝ | |p.1 + 1| + |p.2 - 2| ≤ 3 } →
  let M := { z : ℝ | ∃ p : ℝ × ℝ, p ∈ T ∧ z = p.1 + 2 * p.2 } in
  (Sup M + Inf M = 6) :=
by
  intro hT
  let M := { z : ℝ | ∃ p : ℝ × ℝ, p ∈ T ∧ z = p.1 + 2 * p.2 }
  have hT_def : T = { p : ℝ × ℝ | |p.1 + 1| + |p.2 - 2| ≤ 3 } := hT
  have h_sup : Sup M = 9 := sorry
  have h_inf : Inf M = -3 := sorry
  rw [h_sup, h_inf]
  exact rfl

end sum_of_max_min_elements_l479_479785


namespace parabola_c_value_l479_479089

theorem parabola_c_value :
  ∀ (a b c : ℝ),
    (vertex : ℝ × ℝ) (point_on_parabola : ℝ × ℝ),
    vertex = (5, -1) →
    point_on_parabola = (3, 1) →
    (λ y, a * y^2 + b * y + c) = λ y, a * (y + 1)^2 + 5 →
    c = 9 / 2 :=
by
  intros a b c vertex point_on_parabola h_vertex h_point_on_parabola h_eqn
  sorry

end parabola_c_value_l479_479089


namespace equation_has_solutions_l479_479629

theorem equation_has_solutions :
  ∀ x : ℝ, 2.21 * (root 3 ((5 + x)^2)) + 4 * (root 3 ((5 - x)^2)) = 5 * (root 3 (25 - x)) ↔
  (x = 0 ∨ x = 63 / 13) := by
  sorry

end equation_has_solutions_l479_479629


namespace probability_multiple_of_4_l479_479324

-- Definition of the problem conditions
def random_integer (n : ℕ) := ∀ i, 0 < i ∧ i ≤ n → Prop

def multiple_of_4 (i : ℕ) : Prop := i % 4 = 0

def count_multiples_of_4 (n : ℕ) : ℕ := (finset.range n).filter (λ x, multiple_of_4 x).card

-- Given problem conditions
def ben_choose_random_integer : Prop :=
  ∃ x y : ℕ, random_integer 60 x ∧ random_integer 60 y

-- Required proof statement
theorem probability_multiple_of_4 :
  (count_multiples_of_4 60 = 15) →
  (ben_choose_random_integer) →
  let probability := 1 - (3/4) * (3/4)
  in probability = 7/16 :=
begin
  intros h_multiples h_ben_choose,
  sorry
end

end probability_multiple_of_4_l479_479324


namespace eight_p_plus_one_is_composite_l479_479924

theorem eight_p_plus_one_is_composite (p : ℕ) (hp : Nat.Prime p) (h8p1 : Nat.Prime (8 * p - 1)) : ¬ Nat.Prime (8 * p + 1) :=
by
  sorry

end eight_p_plus_one_is_composite_l479_479924


namespace volume_of_solid_T_def_l479_479015

noncomputable def volume_of_solid_T : ℝ :=
  let s := λ (x y z : ℝ), abs x + abs y ≤ 2 ∧ abs x + abs z ≤ 2 ∧ abs y + abs z ≤ 2 in
  8 * (1 / 6 * 2 * 2 * 2)

theorem volume_of_solid_T_def :
  volume_of_solid_T = 32 / 3 :=
by
  sorry

end volume_of_solid_T_def_l479_479015


namespace area_of_hexagon_l479_479245

theorem area_of_hexagon (DEF : Type) [number DEF]
  (perimeter_DEF : DEF)
  (radius_circumcircle : DEF)
  (perpendicular_bisectors_meet_circumcircle :
    ∀ d e f : DEF, d + e + f = perimeter_DEF ∧ radius_circumcircle = 10) :
  (perimeter_DEF = 40) →
  ∃ hexagon_area : DEF, hexagon_area = 50 :=
by
  intro h
  have perimeter := h
  have area_hex := 50
  exact ⟨area_hex, rfl⟩
  sorry

end area_of_hexagon_l479_479245


namespace coprime_3x3_grid_l479_479825

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def valid_3x3_grid (grid : Array (Array ℕ)) : Prop :=
  -- Check row-wise, column-wise, and diagonal adjacencies for coprimality
  ∀ i j, i < 3 → j < 3 →
  (if j < 2 then is_coprime (grid[i][j]) (grid[i][j+1]) else True) ∧
  (if i < 2 then is_coprime (grid[i][j]) (grid[i+1][j]) else True) ∧
  (if i < 2 ∧ j < 2 then is_coprime (grid[i][j]) (grid[i+1][j+1]) else True) ∧
  (if i < 2 ∧ j > 0 then is_coprime (grid[i][j]) (grid[i+1][j-1]) else True)

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10], #[5, 7, 11], #[6, 13, 12]]

theorem coprime_3x3_grid : valid_3x3_grid grid := sorry

end coprime_3x3_grid_l479_479825


namespace line_equation_through_point_line_equation_sum_of_intercepts_l479_479864

theorem line_equation_through_point (x y : ℝ) (h : y = 2 * x + 5)
  (hx : x = -2) (hy : y = 1) : 2 * x - y + 5 = 0 :=
by {
  sorry
}

theorem line_equation_sum_of_intercepts (x y : ℝ) (h : y = 2 * x + 6)
  (hx : x = -3) (hy : y = 3) : 2 * x - y + 6 = 0 :=
by {
  sorry
}

end line_equation_through_point_line_equation_sum_of_intercepts_l479_479864


namespace convex_polyhedron_inequality_l479_479467

-- Definitions of the given conditions
variable (R : ℝ) (n : ℕ)
variable (l : Fin n → ℝ) (ϕ : Fin n → ℝ)

-- Main statement to be proven
theorem convex_polyhedron_inequality : 
  (∑ i in Finset.finRange n, l i * (Real.pi - ϕ i)) ≤ 8 * Real.pi * R :=
by
  sorry

end convex_polyhedron_inequality_l479_479467


namespace candies_bought_is_18_l479_479667

-- Define the original number of candies
def original_candies : ℕ := 9

-- Define the total number of candies after buying more
def total_candies : ℕ := 27

-- Define the function to calculate the number of candies bought
def candies_bought (o t : ℕ) : ℕ := t - o

-- The main theorem stating that the number of candies bought is 18
theorem candies_bought_is_18 : candies_bought original_candies total_candies = 18 := by
  -- This is where the proof would go
  sorry

end candies_bought_is_18_l479_479667


namespace no_odd_even_composition_equivalence_l479_479201

theorem no_odd_even_composition_equivalence 
  (even_reflections : ℕ) (odd_reflections : ℕ)
  (glide_reflection_isometry : Prop)
  (glide_reflection_properties :
    ∀ l : ℝ,
      (∀ p : ℝ, translates_point_along_line l p) ∧ 
      (∃ q : ℝ, maps_no_parallel_line_to_itself l q) ∧
      (∀ r : ℝ, no_fixed_points_or_fixed_on_line l r))
  (isometry_movements :
    ∀ m : ℝ,
      ((rotation m) ∧ (fixed_point_exists_at_center m)) ∨ 
      ((translation m) ∧ (maps_all_parallel_lines_to_themselves m)))
  (even_composition_preserves_orientation : Prop)
  (odd_composition_reverses_orientation : Prop) : 
  ¬ (can_be_even_and_odd_composition even_reflections odd_reflections) :=
by 
  sorry

end no_odd_even_composition_equivalence_l479_479201


namespace general_formula_b_n_T_n_lt_3_over_2_l479_479385

variable {n : ℕ}
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Condition: a_n = 5S_n + 1
def a_n (n : ℕ) : ℝ := 5 * S n + 1

-- Define b_n
def b_n (n : ℕ) : ℝ := (4 + a n) / (1 - a n)

-- Question 1: Prove the general formula for b_n
theorem general_formula_b_n (n : ℕ) : b_n n = (4 + (-1/4)^n) / (1 - (-1/4)^n) := sorry

-- Define C_n
def C_n (n : ℕ) : ℝ := b n * 2 - b (n * 2 - 1)

-- Define T_n as the sum of the first n terms of C_n
def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, C_n i

-- Question 2: Prove that T_n < 3/2 for any positive integer n
theorem T_n_lt_3_over_2 (n : ℕ) : T_n n < 3 / 2 := sorry

end general_formula_b_n_T_n_lt_3_over_2_l479_479385


namespace prove_x_eq_1_l479_479166

-- Definitions
variables {a b c x : ℝ}

-- Conditions
variable h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a
variable h_coincide : (ax^2 + bx + c = ax^2 + cx + b ∨ ax^2 + bx + c = bx^2 + cx + a ∨ ax^2 + bx + c = bx^2 + ax + c ∨ ax^2 + bx + c = cx^2 + ax + b ∨ ax^2 + bx + c = cx^2 + bx + a ∨
                      ax^2 + cx + b = bx^2 + cx + a ∨ ax^2 + cx + b = bx^2 + ax + c ∨ ax^2 + cx + b = cx^2 + ax + b ∨ ax^2 + cx + b = cx^2 + bx + a ∨
                      bx^2 + cx + a = bx^2 + ax + c ∨ bx^2 + cx + a = cx^2 + ax + b ∨ bx^2 + cx + a = cx^2 + bx + a ∨
                      bx^2 + ax + c = cx^2 + ax + b ∨ bx^2 + ax + c = cx^2 + bx + a ∨
                      cx^2 + ax + b = cx^2 + bx + a)

-- Statement to Prove
theorem prove_x_eq_1 : x = 1 :=
  by
  apply sorry -- Proof not included

end prove_x_eq_1_l479_479166


namespace polynomial_roots_arithmetic_progression_l479_479707

theorem polynomial_roots_arithmetic_progression :
  ∃ b : ℝ, (∃ a : ℂ, (a - a.i * sqrt 6, a, a + a.i * sqrt 6) ∈ ({x | polynomial.eval x (polynomial.C b + x^3 - 9*x^2 + 33*x) = 0}) ∧ polynomial.C b + x^3 - 9*x^2 + 33*x = 0) → b = -15 :=
by
  sorry

end polynomial_roots_arithmetic_progression_l479_479707


namespace probability_increasing_function_g_l479_479058

theorem probability_increasing_function_g (a : ℝ) (h : 0 ≤ a ∧ a ≤ 10) : 
  ∃ p : ℝ, p = 1 / 5 ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → ∀ a, (0 ≤ a ∧ a < 2) → g a x₁ < g a x₂ ) :=
by
  let g (a x : ℝ) : ℝ := (a - 2) / x
  use 1 / 5
  sorry

end probability_increasing_function_g_l479_479058


namespace inequality_of_inradii_l479_479261

-- Definitions and assumptions from the conditions
variables {A B C D A1 B1 C1 : Type*}
variables (triangle_ABC : Triangle A B C) (point_D : Point D)
variables (point_A1 : Point A1) (area_S1 area_S2 : ℝ)
variables (r1 r2 : ℝ)
variables (incenter_I1 incenter_I2 : Point)

-- The Lean statement representing the problem
theorem inequality_of_inradii (h1 : points_lie_on BC D A1)
    (h2 : D_inside_triangle ABC D)
    (h3 : incenter triangle_ABC I1)
    (h4 : incenter (Triangle A1 B1 C1) I2)
    (S1_area : area triangle_ABC = area_S1)
    (S2_area : area (Triangle A1 B1 C1) = area_S2)
    (radii_condition : inradii_relation r1 r2) :
    r1 > r2 := 
sorry

end inequality_of_inradii_l479_479261


namespace point_distance_difference_l479_479149

theorem point_distance_difference
    (t : ℝ)
    (x y : ℝ)
    (x_A y_A x_B y_B : ℝ)
    (t_A t_B : ℝ): 
    (x = 1 + 2 * t) → 
    (y = 2 + t) → 
    (t_A + t_B = -8/5) → 
    (t_A * t_B = 1/5) → 
    (P : ℝ × ℝ) (P = (-1, 1)) → 
    (A : ℝ × ℝ) (A = (1 + 2 * t_A, 2 + t_A)) →
    (B : ℝ × ℝ) (B = (1 + 2 * t_B, 2 + t_B)) →
    ∥(P.1 - A.1) + (P.2 - A.2) - ((P.1 - B.1) + (P.2 - B.2))∥ = 2 * sqrt(5) / 5 :=
sorry

end point_distance_difference_l479_479149


namespace max_acute_angles_convex_polygon_l479_479296

theorem max_acute_angles_convex_polygon (n : ℕ) (h1 : 3 ≤ n)
  (sum_exterior_angles : ∑ i in range n, exterior_angle i = 360)
  (supplementary_interior_exterior : ∀ i, exterior_angle i + interior_angle i = 180) :
  ∃ m ≤ 3, ∀ i, acute_angle (interior_angle i) → i < m := 
sorry

end max_acute_angles_convex_polygon_l479_479296


namespace y1_less_than_y2_l479_479441

noncomputable def y1 : ℝ := 2 * (-5) + 1
noncomputable def y2 : ℝ := 2 * 3 + 1

theorem y1_less_than_y2 : y1 < y2 := by
  sorry

end y1_less_than_y2_l479_479441


namespace songs_performed_l479_479452

variable (R L S M : ℕ)
variable (songs_total : ℕ)

def conditions := 
  R = 9 ∧ L = 6 ∧ (6 ≤ S ∧ S ≤ 9) ∧ (6 ≤ M ∧ M ≤ 9) ∧ songs_total = (R + L + S + M) / 3

theorem songs_performed (h : conditions R L S M songs_total) :
  songs_total = 9 ∨ songs_total = 10 ∨ songs_total = 11 :=
sorry

end songs_performed_l479_479452


namespace time_on_wednesday_is_40_minutes_l479_479502

def hours_to_minutes (h : ℚ) : ℚ := h * 60

def time_monday : ℚ := hours_to_minutes (3 / 4)
def time_tuesday : ℚ := hours_to_minutes (1 / 2)
def time_wednesday (w : ℚ) : ℚ := w
def time_thursday : ℚ := hours_to_minutes (5 / 6)
def time_friday : ℚ := 75
def total_time : ℚ := hours_to_minutes 4

theorem time_on_wednesday_is_40_minutes (w : ℚ) 
    (h1 : time_monday = 45) 
    (h2 : time_tuesday = 30) 
    (h3 : time_thursday = 50) 
    (h4 : time_friday = 75)
    (h5 : total_time = 240) 
    (h6 : total_time = time_monday + time_tuesday + time_wednesday w + time_thursday + time_friday) 
    : w = 40 := 
by 
  sorry

end time_on_wednesday_is_40_minutes_l479_479502


namespace find_AC_l479_479133

noncomputable def isTriangle (A B C : Type) : Type := sorry

-- Define angles and lengths.
variables (A B C : Type)
variables (angle_A angle_B : ℝ)
variables (BC AC : ℝ)

-- Assume the given conditions.
axiom angle_A_60 : angle_A = 60 * Real.pi / 180
axiom angle_B_45 : angle_B = 45 * Real.pi / 180
axiom BC_12 : BC = 12

-- Statement to prove.
theorem find_AC 
  (h_triangle : isTriangle A B C)
  (h_angle_A : angle_A = 60 * Real.pi / 180)
  (h_angle_B : angle_B = 45 * Real.pi / 180)
  (h_BC : BC = 12) :
  ∃ AC : ℝ, AC = 8 * Real.sqrt 3 / 3 :=
sorry

end find_AC_l479_479133


namespace inequality_f_c_f_a_f_b_l479_479758

-- Define the function f and the conditions
def f : ℝ → ℝ := sorry

noncomputable def a : ℝ := Real.log (1 / Real.pi)
noncomputable def b : ℝ := (Real.log Real.pi) ^ 2
noncomputable def c : ℝ := Real.log (Real.sqrt Real.pi)

-- Theorem statement
theorem inequality_f_c_f_a_f_b :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2) →
  f c > f a ∧ f a > f b :=
by
  -- Proof omitted
  sorry

end inequality_f_c_f_a_f_b_l479_479758


namespace coprime_3x3_grid_l479_479828

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def valid_3x3_grid (grid : Array (Array ℕ)) : Prop :=
  -- Check row-wise, column-wise, and diagonal adjacencies for coprimality
  ∀ i j, i < 3 → j < 3 →
  (if j < 2 then is_coprime (grid[i][j]) (grid[i][j+1]) else True) ∧
  (if i < 2 then is_coprime (grid[i][j]) (grid[i+1][j]) else True) ∧
  (if i < 2 ∧ j < 2 then is_coprime (grid[i][j]) (grid[i+1][j+1]) else True) ∧
  (if i < 2 ∧ j > 0 then is_coprime (grid[i][j]) (grid[i+1][j-1]) else True)

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10], #[5, 7, 11], #[6, 13, 12]]

theorem coprime_3x3_grid : valid_3x3_grid grid := sorry

end coprime_3x3_grid_l479_479828


namespace high_probability_event_is_C_l479_479247

-- Define the probabilities of events A, B, and C
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.1
def prob_C : ℝ := 0.9

-- Statement asserting Event C has the high possibility of occurring
theorem high_probability_event_is_C : prob_C > prob_A ∧ prob_C > prob_B :=
by
  sorry

end high_probability_event_is_C_l479_479247


namespace area_of_quadrilateral_PQRS_l479_479862

-- Define the points and distances
variable (P Q R S : Type) [PointClass P Q R S]
variable (dist : P → P → ℝ)

-- Given conditions
variable (hQR_right : ∠QRS = 90)
variable (hPQ : dist P Q = 15)
variable (hQR : dist Q R = 5)
variable (hRS : dist R S = 12)
variable (hPS : dist P S = 17)

-- Define the problem to prove the area of the quadrilateral PQRS
theorem area_of_quadrilateral_PQRS : 
    area_of_quadrilateral PQRS = 162 :=
sorry

end area_of_quadrilateral_PQRS_l479_479862


namespace chocolate_for_Noah_l479_479496

theorem chocolate_for_Noah :
  let total_chocolate : ℚ := 60 / 7 in
  let divided_pile : ℚ := total_chocolate / 5 in
  let chocolate_for_Noah : ℚ := 3 * divided_pile in
  chocolate_for_Noah = 36 / 7 :=
by
  sorry

end chocolate_for_Noah_l479_479496


namespace max_sides_subdivision_13_max_sides_subdivision_1950_l479_479450

-- Part (a)
theorem max_sides_subdivision_13 (n : ℕ) (h : n = 13) : 
  ∃ p : ℕ, p ≤ n ∧ p = 13 := 
sorry

-- Part (b)
theorem max_sides_subdivision_1950 (n : ℕ) (h : n = 1950) : 
  ∃ p : ℕ, p ≤ n ∧ p = 1950 := 
sorry

end max_sides_subdivision_13_max_sides_subdivision_1950_l479_479450


namespace certain_number_x_l479_479436

-- Definitions and conditions
def least_possible_k := 4.9956356288922485

def int_k (k : ℤ) := k > least_possible_k

def x := 0.00101

-- The statement to be proved
theorem certain_number_x (k : ℤ) (hk : int_k k) : x * 10 ^ k > 100 := 
sorry

end certain_number_x_l479_479436


namespace probability_of_at_least_one_multiple_of_4_l479_479328

open ProbabilityTheory

def prob_at_least_one_multiple_of_4 : ℚ := 7 / 16

theorem probability_of_at_least_one_multiple_of_4 :
  let S := Finset.range 60
  let multiples_of_4 := S.filter (λ x, (x + 1) % 4 = 0)
  let prob (a b : ℕ) := (a : ℚ) / b
  let prob_neither_multiple_4 := (prob (60 - multiples_of_4.card) 60) ^ 2
  1 - prob_neither_multiple_4 = prob_at_least_one_multiple_of_4 := by
  sorry

end probability_of_at_least_one_multiple_of_4_l479_479328


namespace problem_statement_l479_479275

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem problem_statement :
  (is_irrational (sqrt 10)) ∧
  ¬(is_irrational (1 / 2)) ∧
  ¬(is_irrational (sqrt (-1))) ∧
  ¬(is_irrational 0.23) :=
by
  sorry

end problem_statement_l479_479275


namespace radius_of_circle_formed_by_spherical_coords_l479_479566

theorem radius_of_circle_formed_by_spherical_coords :
  (∃ θ : ℝ, radius_of_circle (1, θ, π / 3) = sqrt 3 / 2) :=
sorry

end radius_of_circle_formed_by_spherical_coords_l479_479566


namespace water_left_in_bucket_l479_479672

theorem water_left_in_bucket (initial_amount poured_amount : ℝ) (h1 : initial_amount = 0.8) (h2 : poured_amount = 0.2) : initial_amount - poured_amount = 0.6 := by
  sorry

end water_left_in_bucket_l479_479672


namespace sum_of_common_divisors_of_list_is_28_l479_479366

noncomputable def divisors (n : ℤ) : List ℕ := 
  (List.range (n.natAbs + 1)).filter (λ d, d > 0 ∧ n % d = 0)

def commonDivisors (lst : List ℤ) : List ℕ :=
  lst.tail.getD [] |> List.foldl (λ acc x, List.intersect acc (divisors x)) (divisors lst.head)

def sumCommonDivisors (lst : List ℤ) : ℕ :=
  (commonDivisors lst).sum

theorem sum_of_common_divisors_of_list_is_28 : 
  sumCommonDivisors [48, 96, -24, 144, 192] = 28 := sorry

end sum_of_common_divisors_of_list_is_28_l479_479366


namespace Mahdi_tennis_on_Monday_l479_479917

def plays_sport_each_day (sports : List String) : Prop := 
  sports.length = 7

def plays_basketball (sports : List String) : Prop :=
  sports.get! 2 = "Basketball"

def plays_golf (sports : List String) : Prop :=
  sports.get! 4 = "Golf"

def runs_three_days_no_consecutive (sports : List String) : Prop :=
  sports.count "Running" = 3 ∧ ∀ i, sports.get? i = "Running" → sports.get? (i + 1) ≠ "Running"

def no_tennis_day_after_running_or_swimming (sports : List String) : Prop :=
  ∀ i, (matches sports i "Running" ∨ matches sports i "Swimming") → matches sports (i + 1) ≠ "Tennis"

def matches (sports : List String) (i : Nat) (sport : String) : Bool :=
  i < sports.length ∧ sports.get! i = sport

def Mahdi_plays_tennis_on_Monday (sports : List String) : Prop :=
  sports.get! 0 = "Tennis"

theorem Mahdi_tennis_on_Monday (sports : List String) :
  plays_sport_each_day sports ∧
  plays_basketball sports ∧
  plays_golf sports ∧
  runs_three_days_no_consecutive sports ∧
  no_tennis_day_after_running_or_swimming sports →
  Mahdi_plays_tennis_on_Monday sports :=
by 
  sorry

end Mahdi_tennis_on_Monday_l479_479917


namespace sphere_diameter_twice_volume_l479_479535

-- Define the volume of a sphere based on its radius
def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Prove that for a sphere with radius 9, the sphere with double its volume has diameter 18 * ∛2
theorem sphere_diameter_twice_volume {a b : ℕ} (h_radius : ∀ r, r = 9) (h_volume : ∀ V, V = sphere_volume 9) : a + b = 20 :=
by
  sorry -- Skipping the proof

end sphere_diameter_twice_volume_l479_479535


namespace cost_of_sandwiches_and_smoothies_l479_479914

-- Define the cost of sandwiches and smoothies
def sandwich_cost := 4
def smoothie_cost := 3

-- Define the discount applicable
def sandwich_discount := 1
def total_sandwiches := 6
def total_smoothies := 7

-- Calculate the effective cost per sandwich considering discount
def effective_sandwich_cost := if total_sandwiches > 4 then sandwich_cost - sandwich_discount else sandwich_cost

-- Calculate the total cost for sandwiches
def sandwiches_cost := total_sandwiches * effective_sandwich_cost

-- Calculate the total cost for smoothies
def smoothies_cost := total_smoothies * smoothie_cost

-- Calculate the total cost
def total_cost := sandwiches_cost + smoothies_cost

-- The main statement to prove
theorem cost_of_sandwiches_and_smoothies : total_cost = 39 := by
  -- skip the proof
  sorry

end cost_of_sandwiches_and_smoothies_l479_479914


namespace prime_pairs_sum_40_l479_479105

open Nat

def is_prime (n : ℕ) : Prop := prime n

def valid_prime_pairs (n : ℕ) (s : ℕ) : List (ℕ × ℕ) :=
  List.filter (λ (p : ℕ × ℕ), is_prime p.1 ∧ is_prime p.2 ∧ p.1 + p.2 = s) 
              ((List.range n).bind (λ a, (List.range (a + 1)).map (λ b => (a, b))))

theorem prime_pairs_sum_40 :
  List.length (valid_prime_pairs 20 40) = 3 :=
by
  sorry

end prime_pairs_sum_40_l479_479105


namespace min_distance_sum_l479_479165

open Real

theorem min_distance_sum (A B C D E F P : Point)
    (h1 : dist A B = 80) 
    (h2 : dist B C = 100)
    (h3 : dist A C = 60)
    (h4 : (D : OnLine BC) ∧ dist C D = 10)
    (h5 : (E : OnLine AC) ∧ dist A E = 45)
    (h6 : (F : OnLine AB) ∧ dist B F = 60)
    : ∃ (x y z : ℕ), (AP + BP + CP + DP + EP + FP = Real.sqrt x + Real.sqrt y + Real.sqrt z) ∧ (x + y + z = 5725) :=
sorry

end min_distance_sum_l479_479165


namespace noncongruent_triangles_count_l479_479199

-- Define the points A, B, C, P, Q, R and the conditions on these points
variables (A B C P Q R : Type)
variable [metric_space ℝ]
variables (dist_AB : dist A B = dist A C)
variables (is_midpoint_P : P = midpoint A B)
variables (is_one_third_Q : Q = one_third_point A C)
variables (is_midpoint_R : R = midpoint B C)

-- Statement of the problem in Lean
theorem noncongruent_triangles_count : 
  ∃ t : set (set Type), (t = {{A, B, C}, {A, B, P}, {A, C, Q}, {B, C, R}, {P, Q, R}, 
               {P, Q, C}, {P, Q, B}, {P, R, A}, {P, R, B}, {Q, R, C}}) ∧ 
               (∀ Δ1 Δ2 ∈ t, Δ1 ≠ Δ2 → ¬is_congruent Δ1 Δ2) ∧ 
               (card t = 10) :=
sorry

end noncongruent_triangles_count_l479_479199


namespace annas_mean_score_l479_479368

theorem annas_mean_score (scores : List ℚ)
    (h_scores : scores = [88, 90, 92, 95, 96, 98, 100, 102, 105])
    (h_tim_mean : ∃ T : List ℚ, T ⊂ scores ∧ T.length = 5 ∧ (T.sum / T.length) = 95) :
    ∃ A : List ℚ, A ⊂ scores ∧ A.length = 4 ∧ (A.sum / A.length) = 97.75 := 
begin
  sorry
end

end annas_mean_score_l479_479368


namespace eggs_per_hen_l479_479263

theorem eggs_per_hen (total_chickens : ℕ) (num_roosters : ℕ) (non_laying_hens : ℕ) (total_eggs : ℕ) :
  total_chickens = 440 →
  num_roosters = 39 →
  non_laying_hens = 15 →
  total_eggs = 1158 →
  (total_eggs / (total_chickens - num_roosters - non_laying_hens) = 3) :=
by
  intros
  sorry

end eggs_per_hen_l479_479263


namespace trigonometric_identity_l479_479936

variable (x y : ℝ)

theorem trigonometric_identity (x y : ℝ) :
  cos x ^ 2 + cos (x - y) ^ 2 - 2 * cos x * cos y * cos (x - y) = sin y ^ 2 :=
sorry

end trigonometric_identity_l479_479936


namespace number_of_members_l479_479550

-- Definitions based on conditions in the problem
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 7
def cap_cost : ℕ := tshirt_cost

def home_game_cost_per_member : ℕ := sock_cost + tshirt_cost
def away_game_cost_per_member : ℕ := sock_cost + tshirt_cost + cap_cost
def total_cost_per_member : ℕ := home_game_cost_per_member + away_game_cost_per_member

def total_league_cost : ℕ := 4324

-- Statement to be proved
theorem number_of_members (m : ℕ) (h : total_league_cost = m * total_cost_per_member) : m = 85 :=
sorry

end number_of_members_l479_479550


namespace find_three_digit_number_in_decimal_l479_479529

theorem find_three_digit_number_in_decimal :
  ∃ (A B C : ℕ), ∀ (hA : A ≠ 0 ∧ A < 7) (hB : B ≠ 0 ∧ B < 7) (hC : C ≠ 0 ∧ C < 7) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
    (h1 : (7 * A + B) + C = 7 * C)
    (h2 : (7 * A + B) + (7 * B + A) = 7 * B + 6), 
    A * 100 + B * 10 + C = 425 :=
by
  sorry

end find_three_digit_number_in_decimal_l479_479529


namespace plane_eq_unique_l479_479542

open Int 

def plane_eq (A B C D x y z : ℤ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_eq_unique (x y z : ℤ) (A B C D : ℤ)
  (h₁ : x = 8) 
  (h₂ : y = -6) 
  (h₃ : z = 2) 
  (h₄ : A > 0)
  (h₅ : gcd (|A|) (gcd (|B|) (gcd (|C|) (|D|))) = 1) :
  plane_eq 4 (-3) 1 (-52) x y z :=
by
  sorry

end plane_eq_unique_l479_479542


namespace correct_answers_l479_479182

noncomputable def ellipse := {x : ℝ × ℝ // (x.1 ^ 2) / 2 + x.2 ^ 2 = 1}

def foci_dist_sum (P : ellipse) : ℝ :=
  let c := 1 in -- c is already calculated as 1
  let F1 := (-c, 0) in
  let F2 := (c, 0) in
  (Real.sqrt ((P.val.1 + c)^2 + P.val.2^2)) + (Real.sqrt ((P.val.1 - c)^2 + P.val.2^2))

theorem correct_answers (P : ellipse) : 
(foci_dist_sum P = 2 * Real.sqrt 2) ∧ (∀ (θ : ℝ), (θ < real.pi / 2 → False) → 
(False := θ) :=
sorry

end correct_answers_l479_479182


namespace statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_l479_479276

theorem statement_A_correct :
  (∃ x0 : ℝ, x0^2 + 2 * x0 + 2 < 0) ↔ (¬ ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0) :=
sorry

theorem statement_B_incorrect :
  ¬ (∀ x y : ℝ, x > y → |x| > |y|) :=
sorry

theorem statement_C_incorrect :
  ¬ ∀ x : ℤ, x^2 > 0 :=
sorry

theorem statement_D_correct :
  (∀ m : ℝ, (∃ x1 x2 : ℝ, x1 + x2 = 2 ∧ x1 * x2 = m ∧ x1 * x2 > 0) ↔ m < 0) :=
sorry

end statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_l479_479276


namespace trigonometric_expression_evaluation_l479_479101

theorem trigonometric_expression_evaluation (θ : ℝ) 
  (h : (cot θ ^ 2000 + 2) / (sin θ + 1) = 1) : 
  (sin θ + 2) ^ 2 * (cos θ + 1) = 9 := 
by
  sorry

end trigonometric_expression_evaluation_l479_479101


namespace find_angle_B_correct_l479_479881

noncomputable def find_angle_B : Real :=
  let A : ℝ := ∠3 ABC
  let B : ℝ := ∠3 A_1B_1C_1
  have h1 : AB = BC := by sorry
  have h2 : A_1B_1 = B_1C_1 := by sorry
  have h3 : is_similar ABC A_1B_1C_1 := by sorry
  have h4 : A_3 AB A_1B_1 = 2 := by sorry
  have h5 : (∠2 A_1B_1) ⊥ AC := by sorry
   
  -- Calculating the angle B
  angle B := arccos (1 / 8)

-- Proving that given the conditions, the angle B is equal to arccos(1 / 8)
theorem find_angle_B_correct :
  find_angle_B = arccos (1 / 8) :=
  by sorry

end find_angle_B_correct_l479_479881


namespace clock_angle_230_l479_479599

theorem clock_angle_230 (h12 : ℕ := 12) (deg360 : ℕ := 360) 
  (hour_mark_deg : ℕ := deg360 / h12) (hour_halfway : ℕ := hour_mark_deg / 2)
  (hour_deg_230 : ℕ := hour_mark_deg * 3) (total_angle : ℕ := hour_halfway + hour_deg_230) :
  total_angle = 105 := 
by
  sorry

end clock_angle_230_l479_479599


namespace total_acres_cleaned_l479_479300

theorem total_acres_cleaned (A D : ℕ) (h1 : (D - 1) * 90 + 30 = A) (h2 : D * 80 = A) : A = 480 :=
sorry

end total_acres_cleaned_l479_479300


namespace general_term_sum_bound_l479_479874

variable {a : Nat → ℝ}
variable {n : Nat}

def a_rec (n : Nat) : ℝ :=
  if n = 1 then 1 else
  if n = 2 then 1 / 4 else
  (n - 2) * a (n - 1) / (n - 1 - a (n - 1))

noncomputable def a_series (n : Nat) : ℝ :=
  if n = 0 then 0 else a n + a_series (n - 1)

theorem general_term (n : ℕ) : a n = 1 / (3 * n - 2) := by
  sorry

theorem sum_bound (n : ℕ) : (∑ k in Finset.range n, (a k) ^ 2) < 7 / 6 := by
  sorry

end general_term_sum_bound_l479_479874


namespace time_on_sideline_l479_479138

def total_game_time : ℕ := 90
def time_mark_played_first_period : ℕ := 20
def time_mark_played_second_period : ℕ := 35
def total_time_mark_played : ℕ := time_mark_played_first_period + time_mark_played_second_period

theorem time_on_sideline : total_game_time - total_time_mark_played = 35 := by
  sorry

end time_on_sideline_l479_479138


namespace arithmetic_sequence_sum_cubes_l479_479251

theorem arithmetic_sequence_sum_cubes (x : ℤ) (k : ℕ) (h : ∀ i, 0 <= i ∧ i <= k → (x + 2 * i : ℤ)^3 =
  -1331) (hk : k > 3) : k = 6 :=
sorry

end arithmetic_sequence_sum_cubes_l479_479251


namespace problem1_problem2_l479_479056

variable (α : ℝ)

-- First problem statement
theorem problem1 (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6 / 13 :=
by 
  sorry

-- Second problem statement
theorem problem2 (h : Real.tan α = 2) :
  3 * (Real.sin α)^2 + 3 * Real.sin α * Real.cos α - 2 * (Real.cos α)^2 = 16 / 5 :=
by 
  sorry

end problem1_problem2_l479_479056


namespace three_liters_to_gallons_l479_479398

theorem three_liters_to_gallons :
  (0.5 : ℝ) * 3 * 0.1319 = 0.7914 := by
  sorry

end three_liters_to_gallons_l479_479398


namespace painted_numbers_eq_l479_479648

/-- A pair of positive integers (m, n) is called guerrera if there exist positive integers a, b, c, d such that
      m = ab, n = cd, and a + b = c + d -/
def is_guerrera (m n : ℕ) : Prop :=
  ∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ m = a * b ∧ n = c * d ∧ a + b = c + d

/-- Predicate to indicate whether a number is painted -/
def is_painted (painted_numbers : set ℕ) (x : ℕ) : Prop :=
  x ∈ painted_numbers ∨ ∃ y : ℕ, y ∈ painted_numbers ∧ is_guerrera x y

noncomputable def painted_numbers : set ℕ :=
  { x : ℕ | x ≥ 3 }

theorem painted_numbers_eq :
  { x : ℕ | ∃ y, is_painted {3, 5} x } = { x : ℕ | x ≥ 3 } :=
sorry

end painted_numbers_eq_l479_479648


namespace a_star_b_value_l479_479003

theorem a_star_b_value (a b : ℤ) (h1 : a + b = 12) (h2 : a * b = 32) (h3 : b = 8) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 3 / 8 := by
sorry

end a_star_b_value_l479_479003


namespace perimeter_of_triangle_l479_479303

noncomputable def triangle_perimeter : ℝ :=
  let line1 := (x : ℝ) -> 1
  let line2 := (x : ℝ) -> 1 + (1/2) * x
  let origin := (0, 0)
  let intersection1 := (1, line1 1)
  let intersection2 := (1, line2 1)
  let vertical_length := |intersection2.2 - intersection1.2|
  let horizontal_length := Real.sqrt ((1 - origin.1)^2 + (intersection2.2 - origin.2)^2)
  let hypotenuse_length := Real.sqrt ((1 - origin.1)^2 + (intersection1.2 - origin.2)^2)
  vertical_length + horizontal_length + hypotenuse_length

theorem perimeter_of_triangle :
  triangle_perimeter = Real.sqrt 3.25 + Real.sqrt 5 + 3.5 :=
by
  sorry

end perimeter_of_triangle_l479_479303


namespace tiles_needed_for_square_l479_479430

theorem tiles_needed_for_square : 
  ∀ (width height : ℕ), 
    width = 12 → height = 15 → 
    let lcm_width_height := nat.lcm width height in
    let side_length_square := lcm_width_height in
    let area_square := side_length_square * side_length_square in
    let area_tile := width * height in
    area_square / area_tile = 20 :=
by
  intros width height h_width h_height
  let lcm_width_height := nat.lcm width height
  let side_length_square := lcm_width_height
  let area_square := side_length_square * side_length_square
  let area_tile := width * height
  sorry

end tiles_needed_for_square_l479_479430


namespace four_digit_integer_l479_479229

theorem four_digit_integer (a b c d : ℕ) 
  (h1 : a + b + c + d = 16) 
  (h2 : b + c = 10) 
  (h3 : a - d = 2)
  (h4 : (a - b + c - d) % 11 = 0) 
  : 1000 * a + 100 * b + 10 * c + d = 4642 := 
begin
  sorry
end

end four_digit_integer_l479_479229


namespace players_quit_game_l479_479988

variable (total_players initial num_lives players_left players_quit : Nat)
variable (each_player_lives : Nat)

theorem players_quit_game :
  (initial = 8) →
  (each_player_lives = 3) →
  (num_lives = 15) →
  players_left = num_lives / each_player_lives →
  players_quit = initial - players_left →
  players_quit = 3 :=
by
  intros h_initial h_each_player_lives h_num_lives h_players_left h_players_quit
  sorry

end players_quit_game_l479_479988


namespace no_tetrahedron_with_all_edges_obtuse_angle_l479_479349

def tetrahedron := Π (face1 face2 face3 face4 : set (ℝ × ℝ × ℝ)), 
  -- Defining the structure as four triangular faces
  ∀ (edge : set (ℝ × ℝ × ℝ)), 
  -- Each edge
  edge ∈ face1 ∩ face2 ∨ edge ∈ face2 ∩ face3 ∨ edge ∈ face3 ∩ face4 ∨ edge ∈ face4 ∩ face1

noncomputable def dihedral_angle (f1 f2 : set (ℝ × ℝ × ℝ)) : ℝ := 
  -- Dihedral angle between two faces
  real.arccos (-1 / 3)

def is_obtuse (angle : ℝ) : Prop :=
  90 < angle ∧ angle < 180

theorem no_tetrahedron_with_all_edges_obtuse_angle :
  ¬ ∃ (T : tetrahedron), ∀ (e : set (ℝ × ℝ × ℝ)) (f1 f2 : set (ℝ × ℝ × ℝ)), 
    -- For any edge and the faces it belongs to define the angle
    (e ∈ f1 ∩ f2) → is_obtuse (dihedral_angle f1 f2) :=
sorry

end no_tetrahedron_with_all_edges_obtuse_angle_l479_479349


namespace comic_books_arrangement_l479_479501

theorem comic_books_arrangement :
  let spiderman := 7!
  let archie := 6!
  let garfield := 4!
  let group_arrangements := 2 * 2 * 1 (Archie ≠ bottom)
  (spiderman * archie * garfield * group_arrangements) = 19353600 :=
by {
  sorry
}

end comic_books_arrangement_l479_479501


namespace point_to_line_distance_l479_479537

theorem point_to_line_distance :
  let circle_center : ℝ×ℝ := (0, 1)
  let A : ℝ := -1
  let B : ℝ := 1
  let C : ℝ := -2
  let line_eq (x y : ℝ) := A * x + B * y + C == 0
  ∀ (x0 : ℝ) (y0 : ℝ),
    circle_center = (x0, y0) →
    (|A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)) = (Real.sqrt 2 / 2) := 
by 
  intros
  -- Proof goes here
  sorry -- Placeholder for the proof.

end point_to_line_distance_l479_479537


namespace least_positive_integer_for_multiple_of_five_l479_479602

theorem least_positive_integer_for_multiple_of_five (x : ℕ) (h_pos : 0 < x) (h_multiple : (625 + x) % 5 = 0) : x = 5 :=
sorry

end least_positive_integer_for_multiple_of_five_l479_479602


namespace decreasing_power_function_unique_m_l479_479273

-- Define the conditions
def decreasing_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y ≤ f x

def power_function (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - m - 1) * x^m

-- The theorem stating the problem
theorem decreasing_power_function_unique_m (m : ℝ) :
  (∀ x ∈ Ioi 0, differentiable_at ℝ (λ x, power_function m x) x)
  → decreasing_function (λ x, power_function m x) 0 ⊤
  → m = -1 :=
begin
  sorry,
end

end decreasing_power_function_unique_m_l479_479273


namespace sum_of_valid_n_l479_479612

theorem sum_of_valid_n :
  let divisors := [1, -1, 3, -3, 9, -9, 2, -2, 6, -6, 18, -18] in
  let valid_n_list := divisors.filter_map (fun d => 
    if (2 * ((d + 1) / 2) - 1 = d) then some ((d + 1) / 2) else none) in
  valid_n_list.sum = 3 :=
by sorry

end sum_of_valid_n_l479_479612


namespace find_a7_a8_l479_479459

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (g : geometric_sequence a q)

def sum_1_2 : ℝ := a 1 + a 2
def sum_3_4 : ℝ := a 3 + a 4

theorem find_a7_a8
  (h1 : sum_1_2 = 30)
  (h2 : sum_3_4 = 60)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 7 + a 8 = (a 1 + a 2) * (q ^ 6) := 
sorry

end find_a7_a8_l479_479459


namespace smallest_positive_multiple_of_45_l479_479607

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ 45 * x = 45 :=
by {
  use 1,
  rw mul_one,
  exact nat.one_pos,
  sorry
}

end smallest_positive_multiple_of_45_l479_479607


namespace compute_cd_l479_479912

-- Define the variables c and d as real numbers
variables (c d : ℝ)

-- Define the conditions
def condition1 : Prop := c + d = 10
def condition2 : Prop := c^3 + d^3 = 370

-- State the theorem we need to prove
theorem compute_cd (h1 : condition1 c d) (h2 : condition2 c d) : c * d = 21 :=
by
  sorry

end compute_cd_l479_479912


namespace solve_percentage_increase_length_l479_479961

def original_length (L : ℝ) : Prop := true
def original_breadth (B : ℝ) : Prop := true

def new_breadth (B' : ℝ) (B : ℝ) : Prop := B' = 1.25 * B

def new_length (L' : ℝ) (L : ℝ) (x : ℝ) : Prop := L' = L * (1 + x / 100)

def original_area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

def new_area (A' : ℝ) (A : ℝ) : Prop := A' = 1.375 * A

def percentage_increase_length (x : ℝ) : Prop := x = 10

theorem solve_percentage_increase_length (L B A A' L' B' x : ℝ)
  (hL : original_length L)
  (hB : original_breadth B)
  (hB' : new_breadth B' B)
  (hL' : new_length L' L x)
  (hA : original_area L B A)
  (hA' : new_area A' A)
  (h_eqn : L' * B' = A') :
  percentage_increase_length x :=
by
  sorry

end solve_percentage_increase_length_l479_479961


namespace triangle_area_proof_l479_479022

structure Point :=
(x : ℝ)
(y : ℝ)

def area_of_triangle (A B C : Point) : ℝ :=
  1 / 2 * abs ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x))

theorem triangle_area_proof :
  let A := ⟨2, -3⟩;
      B := ⟨-1, 1⟩;
      C := ⟨0, -4⟩
  in area_of_triangle A B C = 11 / 2 :=
by
  -- Proof skipped
  sorry

end triangle_area_proof_l479_479022


namespace find_number_of_olives_l479_479945

theorem find_number_of_olives (O : ℕ)
  (lettuce_choices : 2 = 2)
  (tomato_choices : 3 = 3)
  (soup_choices : 2 = 2)
  (total_combos : 2 * 3 * O * 2 = 48) :
  O = 4 :=
by
  sorry

end find_number_of_olives_l479_479945


namespace triangle_largest_angle_l479_479078

theorem triangle_largest_angle {k : ℝ} (h1 : k > 0)
  (h2 : k + 2 * k + 3 * k = 180) : 3 * k = 90 := 
sorry

end triangle_largest_angle_l479_479078


namespace slope_of_line_through_PQ_is_4_l479_479803

theorem slope_of_line_through_PQ_is_4
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a4 : a 4 = 15)
  (h_a9 : a 9 = 55) :
  let a3 := a 3
  let a8 := a 8
  (a 9 - a 4) / (9 - 4) = 8 → (a 8 - a 3) / (13 - 3) = 4 := by
  sorry

end slope_of_line_through_PQ_is_4_l479_479803


namespace quad_intersects_x_axis_l479_479073

theorem quad_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x ^ 2 - 7 * x - 7 = 0) ↔ (k ≥ -7 / 4 ∧ k ≠ 0) :=
by sorry

end quad_intersects_x_axis_l479_479073


namespace provisions_duration_initially_l479_479142

theorem provisions_duration_initially :
  ∀ (D : ℕ),
    let soldiers_initial := 1200,
        daily_consumption_initial := 3,
        soldiers_after_joining := 1728,
        daily_consumption_after_joining := 2.5,
        duration_after_joining := 25 in
    (soldiers_initial * daily_consumption_initial * D = 
    soldiers_after_joining * daily_consumption_after_joining * duration_after_joining) → 
    D = 30 :=
by
  intros D soldiers_initial daily_consumption_initial soldiers_after_joining daily_consumption_after_joining duration_after_joining h
  sorry

end provisions_duration_initially_l479_479142


namespace total_valid_price_arrangements_l479_479299

theorem total_valid_price_arrangements (S : Multiset ℕ) (prices : List ℕ) (valid_prices : ∀ p ∈ prices, 1 ≤ p ∧ p ≤ 9999) 
  (at_least_one_1000 : ∃ p ∈ prices, p ≥ 1000) : 
  S = {1, 1, 1, 2, 2, 3, 3, 3, 3} → 
  prices.length = 3 →
  multiset.recur_on S (λ _, 26460 = 0)
  (λ b n m, 26460 = 
    (multiset.filter (λ (a : ℕ), a = 1) S).card.factorial / 
    ((multiset.count 1 S).factorial * (multiset.count 2 S).factorial * 
      (multiset.count 3 S).factorial)) := sorry

end total_valid_price_arrangements_l479_479299


namespace right_triangle_30_60_90_l479_479057

theorem right_triangle_30_60_90 {A B C : Type} [InnerProductSpace ℝ A] 
  (h₁ : ∠ C = 90°) (h₂ : ∠ B = 30°) (h₃ : dist A B = 2) : dist A C = 1 := 
sorry

end right_triangle_30_60_90_l479_479057


namespace matrices_are_inverses_l479_479545

-- Define the two matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[3, -8], [-4, 11]]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[11, 8], [4, 3]]

-- Prove that A and B are inverses
theorem matrices_are_inverses :
  A * B = 1 := 
  sorry

end matrices_are_inverses_l479_479545


namespace pythagorean_theorem_l479_479207

theorem pythagorean_theorem {a b c p q : ℝ} 
  (h₁ : p * c = a ^ 2) 
  (h₂ : q * c = b ^ 2)
  (h₃ : p + q = c) : 
  c ^ 2 = a ^ 2 + b ^ 2 := 
by 
  sorry

end pythagorean_theorem_l479_479207


namespace souvenirs_total_l479_479666

theorem souvenirs_total (n : ℕ) (h : n = 45) :
  let seq := [1, 3, 5, 7],
      m := n / 4,
      r := n % 4 in
  4 * m * (1 + 3 + 5 + 7) / 4 + list.nthLe seq r (by linarith) = 177 :=
by
  -- additional necessary setup
  sorry

end souvenirs_total_l479_479666


namespace find_x_minus_y_l479_479806

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : x - y = 3 / 2 :=
by
  sorry

end find_x_minus_y_l479_479806


namespace grid_satisfies_conditions_l479_479844

-- Define the range of consecutive integers
def consecutive_integers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Define a function to check mutual co-primality condition for two given numbers
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the 3x3 grid arrangement as a matrix
@[reducible]
def grid : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![8, 9, 10],
    ![5, 7, 11],
    ![6, 13, 12]]

-- Define a predicate to check if the grid cells are mutually coprime for adjacent elements
def mutually_coprime_adjacent (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
    ∀ (i j k l : Fin 3), 
    (|i - k| + |j - l| = 1 ∨ |i - k| = |j - l| ∧ |i - k| = 1) → coprime (m i j) (m k l)

-- The main theorem statement
theorem grid_satisfies_conditions : 
  (∀ (i j : Fin 3), grid i j ∈ consecutive_integers) ∧ 
  mutually_coprime_adjacent grid :=
by
  sorry

end grid_satisfies_conditions_l479_479844


namespace product_uv_l479_479528

theorem product_uv (γ δ m n u v : ℝ) 
  (h1 : ∀ x : ℝ, x^2 - m * x + n = 0 → (x = Real.tan γ ∨ x = Real.tan δ))
  (h2 : ∀ x : ℝ, x^2 - u * x + v = 0 → (x = Real.cot γ ∨ x = Real.cot δ)) :
  uv = m / n^2 :=
by {
  sorry
}

end product_uv_l479_479528


namespace range_of_a_l479_479415

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.abs x) + x^2

theorem range_of_a (a : ℝ) : f (3 * a - 2) > f (a - 1) → a ∈ Set.Ioo (⊤, 1 / 2) ∪ Set.Ioo (3 / 4, ⊤) :=
by
  intro h
  sorry

end range_of_a_l479_479415


namespace no_solution_for_n_gt_9_l479_479706

theorem no_solution_for_n_gt_9 :
  ∀ (n k : ℕ), nat.prime k ∨ n ≤ k ∧ k ≤ n / 2 → 
  (n > 9 ∧ k > 1 ∧ k ≠ n ∧ k ≠ 1 ∧ (∃ p q : ℕ, 1 < p ∧ 1 < q ∧ k = p * q) → ¬ (n ≤ n! - k^n ∧ n! - k^n ≤ k * n)) :=
by
  sorry

end no_solution_for_n_gt_9_l479_479706


namespace find_g_of_one_fifth_l479_479909

variable {g : ℝ → ℝ}

theorem find_g_of_one_fifth (h₀ : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1)
    (h₁ : g 0 = 0)
    (h₂ : ∀ {x y}, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y)
    (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x)
    (h₄ : ∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2) :
  g (1 / 5) = 1 / 4 :=
by
  sorry

end find_g_of_one_fifth_l479_479909


namespace problem_inequality_l479_479905

variables {n : ℕ} (x : fin (n + 1) → ℝ) (h : ∀ i, 0 < x i)

theorem problem_inequality :
  ∑ i in finset.range (n + 1), (x i / x ((i + 1) % (n + 1))) ^ n ≥ 
  ∑ i in finset.range (n + 1), x i / x ((i + 1) % (n + 1)) :=
sorry

end problem_inequality_l479_479905


namespace third_term_one_is_at_n_13_term_2015_is_31_l479_479780

section sequence_problem

-- Define the sequence a_n
def seq (n : ℕ) : ℕ × ℕ :=
  let group := Nat.find (λ k, (k * (k + 1)) / 2 ≥ n)
  let index := n - (group * (group - 1)) / 2 - 1
  (index + 1, group - index)

-- Define the term equal to 1
def is_one (a : ℕ × ℕ) : Prop := a.1 = a.2

-- Problem (1): Prove that the third term that is 1 occurs at position 13
theorem third_term_one_is_at_n_13 : 
  let positions : List ℕ := List.filterMap (λ n, if is_one (seq n) then some n else none) (List.range 20)
  positions.liftNth! 2 = 13 := by
  sorry

-- Problem (2): Prove that the 2015th term of the sequence is 31
theorem term_2015_is_31 : seq 2015 = (31, 1) := by
  sorry

end sequence_problem

end third_term_one_is_at_n_13_term_2015_is_31_l479_479780


namespace number_of_true_propositions_is_1_l479_479551

axiom line : Type
axiom plane : Type
axiom parallel : line → line → Prop
axiom subset : line → plane → Prop
axiom countless : plane → set line
axiom parallel_to_plane : line → plane → Prop

def statement1 (l : line) (α : plane) : Prop :=
  (∀ l', l' ∈ countless α → parallel l l') → parallel_to_plane l α

def statement2 (a : line) (α : plane) : Prop :=
  ¬ subset a α → parallel_to_plane a α

def statement3 (a b : line) (α : plane) : Prop :=
  (parallel a b ∧ subset b α) → parallel_to_plane a α

def statement4 (a b : line) (α : plane) : Prop :=
  (parallel a b ∧ subset b α) → (∀ l', l' ∈ countless α → parallel a l')

theorem number_of_true_propositions_is_1 :
  (∃! (i : ℕ), i = 4) :=
sorry

end number_of_true_propositions_is_1_l479_479551


namespace average_of_five_integers_l479_479437

theorem average_of_five_integers :
  ∃ (A : ℕ) (l s : ℕ),
    (l = 68) ∧
    (s = l - 10) ∧
    (A = (4 * s + l) / 5) ∧
    A = 60 :=
begin
  sorry
end

end average_of_five_integers_l479_479437


namespace arrangement_ends_l479_479576

theorem arrangement_ends (P : Fin 5 → Prop) (A B : Fin 5) :
  (∃ (σ : Equiv.Perm (Fin 5)), (σ 0 = A ∨ σ 0 = B ∨ σ 4 = A ∨ σ 4 = B)) ∧
  ∀ i, P i → ⦃σ : Equiv.Perm (Fin 5) | σ i = i⦄ → count σ = 5! - 36 :=
  sorry

end arrangement_ends_l479_479576


namespace no_larger_triangle_from_two_acute_triangles_l479_479995

theorem no_larger_triangle_from_two_acute_triangles (T1 T2 : Triangle) (hT1 : is_acute_triangle T1) (hT2 : is_acute_triangle T2) :
  ¬ ∃ (T : Triangle) (s : Side) (A1 : Angle) (A2 : Angle), 
    (T1.has_side s) ∧ (T2.has_side s) ∧ 
    (T1.opposite_angle s = A1) ∧ (T2.opposite_angle s = A2) ∧ 
    (A1 + A2 = 180°) :=
sorry

end no_larger_triangle_from_two_acute_triangles_l479_479995


namespace quadratic_intersects_x_axis_l479_479070

theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersects_x_axis_l479_479070


namespace cos_theta_sub_pi_over_4_l479_479434

theorem cos_theta_sub_pi_over_4 (θ : Real) (h : Real.tan (π * Real.cos θ) = Real.cot (π * Real.sin θ)) :
  Real.cos (θ - π / 4) = ± (Math.sqrt 2 / 4) :=
sorry

end cos_theta_sub_pi_over_4_l479_479434


namespace coprime_3x3_grid_l479_479827

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def valid_3x3_grid (grid : Array (Array ℕ)) : Prop :=
  -- Check row-wise, column-wise, and diagonal adjacencies for coprimality
  ∀ i j, i < 3 → j < 3 →
  (if j < 2 then is_coprime (grid[i][j]) (grid[i][j+1]) else True) ∧
  (if i < 2 then is_coprime (grid[i][j]) (grid[i+1][j]) else True) ∧
  (if i < 2 ∧ j < 2 then is_coprime (grid[i][j]) (grid[i+1][j+1]) else True) ∧
  (if i < 2 ∧ j > 0 then is_coprime (grid[i][j]) (grid[i+1][j-1]) else True)

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10], #[5, 7, 11], #[6, 13, 12]]

theorem coprime_3x3_grid : valid_3x3_grid grid := sorry

end coprime_3x3_grid_l479_479827


namespace find_m_for_increasing_function_l479_479556

theorem find_m_for_increasing_function :
  ∀ m : ℝ, (∀ x > 0, (m^2 - m - 1) * x^(m^2 + 2m - 3) > 0) → m = 2 :=
by sorry

end find_m_for_increasing_function_l479_479556


namespace part1_part2_l479_479392

def A (a : ℝ) : set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def B : set ℝ := {x | x^2 - 5 * x + 6 = 0}
def C : set ℝ := {x | x^2 + 2 * x - 8 = 0}

-- Part (1)
theorem part1 (a : ℝ) (h : A a ∩ B = A a ∪ B) : a = 5 := 
sorry

-- Part (2)
theorem part2 (a : ℝ) (h1 : (A a ∩ B) ≠ ∅) (h2 : (A a ∩ C) = ∅) : a = -2 := 
sorry

end part1_part2_l479_479392


namespace radius_of_circle_from_spherical_coords_l479_479558

theorem radius_of_circle_from_spherical_coords :
  ∀ (θ: ℝ), let ρ := 1, φ := π / 3 in
  (√(ρ * sin φ * cos θ)^2 + (ρ * sin φ * sin θ)^2) = √3 / 2 :=
by
  intros θ
  let ρ := 1
  let φ := π / 3
  sorry

end radius_of_circle_from_spherical_coords_l479_479558


namespace true_propositions_l479_479557

def f1 (x : ℝ) : ℝ := sin (|x|)
def f2 (x : ℝ) : ℝ := sin x
def f3 (x : ℝ) : ℝ := cos (-x)
def f4 (x : ℝ) : ℝ := cos (|x|)
def f5 (x : ℝ) : ℝ := |sin x|
def f6 (x : ℝ) : ℝ := cos x
def f7 (x : ℝ) : ℝ := sin (-x)

theorem true_propositions :
  ({(λ x, f3 x = f4 x), (λ x, f6 x = cos (-x))} : set (ℝ → Prop)) = {true, true} :=
by
  sorry

end true_propositions_l479_479557


namespace percent_motorists_receive_tickets_l479_479507

theorem percent_motorists_receive_tickets (n : ℕ) (h1 : (25 : ℕ) % 100 = 25) (h2 : (20 : ℕ) % 100 = 20) :
  (75 * n / 100) = (20 * n / 100) :=
by
  sorry

end percent_motorists_receive_tickets_l479_479507


namespace zoes_apartment_number_units_digit_is_1_l479_479374

-- Defining the conditions as the initial problem does
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def has_digit_two (n : ℕ) : Prop :=
  n / 10 = 2 ∨ n % 10 = 2

def three_out_of_four (n : ℕ) : Prop :=
  (is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬ has_digit_two n) ∨
  (is_square n ∧ is_odd n ∧ ¬ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (is_square n ∧ ¬ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (¬ is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n)

theorem zoes_apartment_number_units_digit_is_1 : ∃ n : ℕ, is_two_digit_number n ∧ three_out_of_four n ∧ n % 10 = 1 :=
by
  sorry

end zoes_apartment_number_units_digit_is_1_l479_479374


namespace smallest_non_prime_in_gap_l479_479021

theorem smallest_non_prime_in_gap :
  ∃ n : ℕ, 40 < n ∧ n + 4 < 60 ∧ (∀ k : ℕ, n ≤ k ∧ k < n + 5 → ¬ prime k) ∧ n = 48 :=
sorry

end smallest_non_prime_in_gap_l479_479021


namespace exists_close_points_l479_479381

theorem exists_close_points (r : ℝ) (h : r > 0) (points : Fin 5 → EuclideanSpace ℝ (Fin 3)) (hf : ∀ i, dist (points i) (0 : EuclideanSpace ℝ (Fin 3)) = r) :
  ∃ i j : Fin 5, i ≠ j ∧ dist (points i) (points j) ≤ r * Real.sqrt 2 :=
by 
  sorry

end exists_close_points_l479_479381


namespace employees_with_advanced_degrees_l479_479851

theorem employees_with_advanced_degrees (total_employees : ℕ) (num_females : ℕ) (males_college_only : ℕ) :
  total_employees = 200 ∧ num_females = 120 ∧ males_college_only = 40 →
  (∃ num_advanced_degrees : ℕ,
    num_advanced_degrees = total_employees - males_college_only ∧
    num_advanced_degrees = 160) :=
begin
  intros h,
  sorry
end

end employees_with_advanced_degrees_l479_479851


namespace inequality_holds_l479_479200

theorem inequality_holds (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) : 
  ((2 + x)/(1 + x))^2 + ((2 + y)/(1 + y))^2 ≥ 9/2 := 
sorry

end inequality_holds_l479_479200


namespace max_value_of_expr_l479_479172

noncomputable def max_expr (a b : ℝ) (h : a + b = 5) : ℝ :=
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4

theorem max_value_of_expr (a b : ℝ) (h : a + b = 5) : max_expr a b h ≤ 6084 / 17 :=
sorry

end max_value_of_expr_l479_479172


namespace value_of_N_l479_479979

theorem value_of_N (a b c N : ℚ) (h1 : a + b + c = 120) (h2 : a - 10 = N) (h3 : 10 * b = N) (h4 : c - 10 = N) : N = 1100 / 21 := 
sorry

end value_of_N_l479_479979


namespace cost_of_black_men_tshirt_l479_479887

def men_tshirt_cost (color : String) : Real := sorry
def women_tshirt_cost (color : String) : Real := men_tshirt_cost color - 5
def white_men_tshirt_cost : Real := 20
def total_employees : Real := 40
def total_cost : Real := 660
def each_sector_employees : Real := total_employees / 2

theorem cost_of_black_men_tshirt :
  (∀ color, women_tshirt_cost color = men_tshirt_cost color - 5) →
  white_men_tshirt_cost = 20 →
  total_employees = 40 →
  total_cost = 660 →
  each_sector_employees = 20 →
  let B := men_tshirt_cost "black" in
  200 + 150 + 10 * B + 10 * (B - 5) = 660 →
  B = 18 :=
sorry

end cost_of_black_men_tshirt_l479_479887


namespace card_A_union_B_l479_479168

-- Define sets and their cardinalities
def A : Set ℕ := {x | 1000 ≤ x ∧ x < 10000 ∧ x % 2 = 1}
def B : Set ℕ := {x | 1000 ≤ x ∧ x < 10000 ∧ x % 5 = 0}

-- Known cardinalities
axiom card_A : A.card = 4500
axiom card_B : B.card = 1800
axiom card_A_inter_B : (A ∩ B).card = 900

-- The theorem to prove
theorem card_A_union_B : (A ∪ B).card = 5400 :=
by {
  have h : (A ∪ B).card = A.card + B.card - (A ∩ B).card,
    from Set.card_union A B,
  rw [card_A, card_B, card_A_inter_B] at h,
  norm_num at h,
  exact h,
}

end card_A_union_B_l479_479168


namespace evaluate_trigonometric_expression_l479_479287

theorem evaluate_trigonometric_expression :
  (sin (120 * (π/180))) ^ 2 + cos (180 * (π/180)) + tan (45 * (π/180)) - (cos (-330 * (π/180))) ^ 2 + sin (-210 * (π/180)) = 1 / 2 := 
  sorry

end evaluate_trigonometric_expression_l479_479287


namespace haleys_car_distance_l479_479973

theorem haleys_car_distance (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used : ℕ) (distance_covered : ℕ) 
   (h_ratio : fuel_ratio = 4) (h_distance_ratio : distance_ratio = 7) (h_fuel_used : fuel_used = 44) :
   distance_covered = 77 := by
  -- Proof to be filled in
  sorry

end haleys_car_distance_l479_479973


namespace part1_probability_infected_given_not_infected_part2_minimum_efficiency_injection_l479_479594

-- Conditions for the first part of the problem
def rabbits_infected_in_group1 := {2}
def rabbits_infected_in_group2 := {6, 7, 10}
def total_rabbits := 10
def total_infected_group1 := rabbits_infected_in_group1.size
def total_infected_group2 := rabbits_infected_in_group2.size
def size_of_group := 5

def probability_first_infected := 
  (1 / 2 * (total_infected_group1 / size_of_group)) + 
  (1 / 2 * (total_infected_group2 / size_of_group))

def probability_both_infected_not_infected :=
  (1 / 2 * (total_infected_group1 / size_of_group) *
   (1 / 2 + 1 / 2 * (total_infected_group1 / size_of_group))) +
  (1 / 2 * (total_infected_group2 / size_of_group) *
   ((1 / 2 * ((size_of_group - total_infected_group2) / size_of_group)) +
   (1 / 2 * (total_infected_group2 / size_of_group))))

def conditional_probability := probability_both_infected_not_infected / probability_first_infected

theorem part1_probability_infected_given_not_infected:
  conditional_probability = 51 / 80 :=
sorry

-- Conditions for the second part of the problem
def one_injection_effectiveness : ℝ := 0.6
def two_injection_effectiveness := 1 - (1 - one_injection_effectiveness) * (1 - one_injection_effectiveness)

def minimum_effectiveness_per_injection_needed := 
  1 - (1 - x)^2 ≥ 0.96

theorem part2_minimum_efficiency_injection : 
  minimum_effectiveness_per_injection_needed = 0.8 :=
sorry

end part1_probability_infected_given_not_infected_part2_minimum_efficiency_injection_l479_479594


namespace Mike_exercises_l479_479188

theorem Mike_exercises :
  let pull_ups_per_visit := 2
  let push_ups_per_visit := 5
  let squats_per_visit := 10
  let office_visits_per_day := 5
  let kitchen_visits_per_day := 8
  let living_room_visits_per_day := 7
  let days_per_week := 7
  let total_pull_ups := pull_ups_per_visit * office_visits_per_day * days_per_week
  let total_push_ups := push_ups_per_visit * kitchen_visits_per_day * days_per_week
  let total_squats := squats_per_visit * living_room_visits_per_day * days_per_week
  total_pull_ups = 70 ∧ total_push_ups = 280 ∧ total_squats = 490 :=
by
  let pull_ups_per_visit := 2
  let push_ups_per_visit := 5
  let squats_per_visit := 10
  let office_visits_per_day := 5
  let kitchen_visits_per_day := 8
  let living_room_visits_per_day := 7
  let days_per_week := 7
  let total_pull_ups := pull_ups_per_visit * office_visits_per_day * days_per_week
  let total_push_ups := push_ups_per_visit * kitchen_visits_per_day * days_per_week
  let total_squats := squats_per_visit * living_room_visits_per_day * days_per_week
  have h1 : total_pull_ups = 2 * 5 * 7 := rfl
  have h2 : total_push_ups = 5 * 8 * 7 := rfl
  have h3 : total_squats = 10 * 7 * 7 := rfl
  show total_pull_ups = 70 ∧ total_push_ups = 280 ∧ total_squats = 490
  sorry

end Mike_exercises_l479_479188


namespace range_of_a_l479_479751

noncomputable def set_A : Set ℝ := { x | x^2 - 5 * x - 24 < 0 }

noncomputable def set_B (a : ℝ) : Set ℝ := { x | (x - 2 * a) * (x - a) < 0 }

theorem range_of_a (a : ℝ) : (set_A ∪ set_B a = set_A) ↔ (a ∈ Icc (-(3 / 2)) 4) :=
by
  sorry

end range_of_a_l479_479751


namespace necessary_and_sufficient_condition_l479_479721

variable (p q : Prop)

theorem necessary_and_sufficient_condition (hp : p) (hq : q) : ¬p ∨ ¬q = False :=
by {
    -- You are requested to fill out the proof here.
    sorry
}

end necessary_and_sufficient_condition_l479_479721


namespace alex_felicia_volume_ratio_l479_479661

theorem alex_felicia_volume_ratio :
  let volume (l w h : ℕ) := l * w * h in
  let V_A := volume 8 6 12 in
  let V_F := volume 12 6 8 in
  V_A = V_F :=
by
  sorry

end alex_felicia_volume_ratio_l479_479661


namespace original_population_l479_479633

-- Define the initial setup
variable (P : ℝ)

-- The conditions given in the problem
axiom ten_percent_died (P : ℝ) : (1 - 0.1) * P = 0.9 * P
axiom twenty_percent_left (P : ℝ) : (1 - 0.2) * (0.9 * P) = 0.9 * P * 0.8

-- Define the final condition
axiom final_population (P : ℝ) : 0.9 * P * 0.8 = 3240

-- The proof problem
theorem original_population : P = 4500 :=
by
  sorry

end original_population_l479_479633


namespace range_of_a_l479_479037

theorem range_of_a (a x y : ℝ)
  (h1 : x + 3 * y = 2 + a)
  (h2 : 3 * x + y = -4 * a)
  (hxy : x + y > 2) : a < -2 := 
sorry

end range_of_a_l479_479037


namespace jenn_money_left_over_l479_479161

-- Definitions based on problem conditions
def num_jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def value_per_quarter : ℚ := 0.25   -- Rational number to represent $0.25
def cost_of_bike : ℚ := 180         -- Rational number to represent $180

-- Statement to prove that Jenn will have $20 left after buying the bike
theorem jenn_money_left_over : 
  (num_jars * quarters_per_jar * value_per_quarter) - cost_of_bike = 20 :=
by
  sorry

end jenn_money_left_over_l479_479161


namespace b_10_value_l479_479789

noncomputable def a_seq : ℕ → ℝ
| 1 := 1
| n + 2 := (2^(n + 1)) / a_seq (n + 1) 

noncomputable def b_seq (n : ℕ) : ℕ → ℝ :=
λ n, a_seq n + a_seq (n + 1)

theorem b_10_value : b_seq 10 = 64 :=
by
  sorry

end b_10_value_l479_479789


namespace visit_patients_on_straight_road_l479_479698

-- Define points representing the patients along a road

structure Point :=
  (x : ℝ)

def cow : Point := ⟨0⟩   -- Location of the cow
def sheWolf : Point := ⟨1⟩   -- Location of the she-wolf
def beetle : Point := ⟨2⟩   -- Location of the beetle
def worm : Point := ⟨6⟩   -- Location of the worm

-- Define the distances Dr. Aibolit walks starting from each patient
def distance (p1 p2 : Point) : ℝ := abs (p2.x - p1.x)

def shortest_route_starting_at (start : Point) : ℝ :=
  if start = cow then distance cow worm
  else if start = sheWolf then distance sheWolf worm + distance cow sheWolf
  else if start = beetle then distance beetle worm + distance beetle sheWolf + distance beetle cow
  else distance worm worm   -- This case should not occur as worm is picked last.

theorem visit_patients_on_straight_road :
  shortest_route_starting_at cow = 6 ∧
  shortest_route_starting_at sheWolf = 7 ∧
  shortest_route_starting_at beetle = 8 :=
by {
  sorry
}

end visit_patients_on_straight_road_l479_479698


namespace unique_number_not_in_range_l479_479173

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : g a b c d 13 = 13)
  (h2 : g a b c d 31 = 31)
  (h3 : ∀ x, x ≠ -d / c → g a b c d (g a b c d x) = x) :
  ∀ y, ∃! x, g a b c d x = y :=
by {
  sorry
}

end unique_number_not_in_range_l479_479173


namespace birds_in_park_l479_479853

theorem birds_in_park (pigeon_ratio sparrow_ratio crow_ratio parakeet_ratio : ℚ)
  (h_pigeon : pigeon_ratio = 0.4)
  (h_sparrow : sparrow_ratio = 0.2)
  (h_crow : crow_ratio = 0.15)
  (h_remainder : parakeet_ratio = 1 - pigeon_ratio - sparrow_ratio - crow_ratio) :
  let total_birds := 100 in
  let non_sparrow_birds := total_birds * (1 - sparrow_ratio) in
  (crow_ratio * total_birds / non_sparrow_birds) * 100 = 18.75 :=
by {
  sorry
}

end birds_in_park_l479_479853


namespace enrollment_difference_and_average_l479_479998

noncomputable def Summit_Ridge_students := 1560
noncomputable def Pine_Hills_students := 1150
noncomputable def Oak_Valley_students := 1950
noncomputable def Maple_Town_students := 1840

theorem enrollment_difference_and_average :
  let largest := max (max Summit_Ridge_students Pine_Hills_students) (max Oak_Valley_students Maple_Town_students),
      smallest := min (min Summit_Ridge_students Pine_Hills_students) (min Oak_Valley_students Maple_Town_students),
      sum := Summit_Ridge_students + Pine_Hills_students + Oak_Valley_students + Maple_Town_students,
      average := sum / 4 in
  largest - smallest = 800 ∧ average = 1625 := 
by 
  let largest := max (max Summit_Ridge_students Pine_Hills_students) (max Oak_Valley_students Maple_Town_students),
      smallest := min (min Summit_Ridge_students Pine_Hills_students) (min Oak_Valley_students Maple_Town_students),
      sum := Summit_Ridge_students + Pine_Hills_students + Oak_Valley_students + Maple_Town_students,
      average := sum / 4
  show largest - smallest = 800 ∧ average = 1625
  sorry

end enrollment_difference_and_average_l479_479998


namespace remainder_2519_div_6_l479_479305

theorem remainder_2519_div_6 : ∃ q r, 2519 = 6 * q + r ∧ 0 ≤ r ∧ r < 6 ∧ r = 5 := 
by
  sorry

end remainder_2519_div_6_l479_479305


namespace positive_slope_asymptote_l479_479683

-- Define the foci points A and B and the given equation of the hyperbola
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-3, 1)
def hyperbola_eqn (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y - 1)^2) - Real.sqrt ((x + 3)^2 + (y - 1)^2) = 4

-- State the theorem about the positive slope of the asymptote
theorem positive_slope_asymptote (x y : ℝ) (h : hyperbola_eqn x y) : 
  ∃ b a : ℝ, b = Real.sqrt 5 ∧ a = 2 ∧ (b / a) = Real.sqrt 5 / 2 :=
by
  sorry

end positive_slope_asymptote_l479_479683


namespace part_1_part_2a_part_2b_l479_479896

namespace InequalityProofs

-- Definitions extracted from the problem
def quadratic_function (m x : ℝ) : ℝ := m * x^2 + (1 - m) * x + m - 2

-- Lean statement for Part 1
theorem part_1 (m : ℝ) : (∀ x : ℝ, quadratic_function m x ≥ -2) ↔ m ∈ Set.Ici (1 / 3) :=
sorry

-- Lean statement for Part 2, breaking into separate theorems for different ranges of m
theorem part_2a (m : ℝ) (h : m < -1) :
  (∀ x : ℝ, quadratic_function m x < m - 1) → 
  (∀ x : ℝ, x ∈ (Set.Iic (-1 / m) ∪ Set.Ici 1)) :=
sorry

theorem part_2b (m : ℝ) (h : -1 < m ∧ m < 0) :
  (∀ x : ℝ, quadratic_function m x < m - 1) → 
  (∀ x : ℝ, x ∈ (Set.Iic 1 ∪ Set.Ici (-1 / m))) :=
sorry

end InequalityProofs

end part_1_part_2a_part_2b_l479_479896


namespace complement_of_M_l479_479443

open Set

def U : Set ℝ := univ

def M : Set ℝ := { x | x^2 - x ≥ 0 }

theorem complement_of_M :
  compl M = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end complement_of_M_l479_479443


namespace correct_statement_is_A_l479_479616

theorem correct_statement_is_A :
  (A : ℕ → ℕ → Prop) ∧
  (B : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ → Prop) ∧
  (C : ℝ → ℝ → Prop) ∧
  (D : ℝ → Prop) →
  A 1 2 := by -- replace 1 and 2 with meaningful parameters if needed
sorry

end correct_statement_is_A_l479_479616


namespace range_of_a_l479_479034

theorem range_of_a (a x y : ℝ) (h1: x + 3 * y = 2 + a) (h2: 3 * x + y = -4 * a) (h3: x + y > 2) : a < -2 :=
sorry

end range_of_a_l479_479034


namespace circumference_of_square_is_correct_l479_479268

-- Conditions: The area of the square is 324 square meters.
def area_of_square (side : ℝ) : ℝ := side^2

-- Question: What is the circumference of the square?
def circumference_of_square (side : ℝ) : ℝ := 4 * side

-- Given: area = 324
-- We are to prove: circumference == 72
theorem circumference_of_square_is_correct :
  ∃ (side : ℝ), area_of_square side = 324 ∧ circumference_of_square side = 72 :=
by
  have area := area_of_square 18
  have circumference := circumference_of_square 18
  use 18
  split
  repeat {simp [area_of_square, circumference_of_square]}
  sorry

end circumference_of_square_is_correct_l479_479268


namespace probability_sum_nine_l479_479257

theorem probability_sum_nine :
  let outcomes := [(a, b, c) | a ← [1, 2, 3, 4, 5, 6], b ← [1, 2, 3, 4, 5, 6], c ← [1, 2, 3, 4, 5, 6]]
  let favorable := (outcomes.filter (λ (abc : ℕ × ℕ × ℕ), abc.1 + abc.2 + abc.3 = 9)).length
  let total := outcomes.length
  let probability := favorable / total
  probability = 19 / 216 :=
by
  sorry

end probability_sum_nine_l479_479257


namespace prove_circle_equation_prove_max_value_of_difference_l479_479736

-- Part (Ⅰ):
theorem prove_circle_equation (t : ℝ) (ht : t ≠ 0) 
(h1 : ∀ x y, x^2 + y^2 = 0) 
(h2 : ∀ x y, (x - 3)^2 + (y - 1)^2 = 10 ∨ (x + 3)^2 + (y + 1)^2 = 10) 
(h3 : ¬∀ x y, (x + 3)^2 + (y + 1)^2 = 10) : 
(∀ x y, (x - 3)^2 + (y - 1)^2 = 10) 
:= by
  sorry

-- Part (Ⅱ):
theorem prove_max_value_of_difference (B : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ)
(hB : B = (0, 2)) 
(hline_l : ∀ Px Py, Px + Py + 2 = 0)
(hcircle : ∀ Qx Qy, (Qx - 3)^2 + (Qy - 1)^2 = 10)
: ∃ P, ∀ Q, ‖P - Q‖ - ‖P - B‖ = 2 * real.sqrt 10 ∧ P = (-6, 4)
:= by
  sorry

end prove_circle_equation_prove_max_value_of_difference_l479_479736


namespace boat_upstream_speed_l479_479292

-- Lean 4 statement for the proof problem
theorem boat_upstream_speed (V_b : ℝ) (V_s : ℝ) (downstream_speed : ℝ) :
  (V_b = 7) → (downstream_speed = 10) → (V_b - V_s = 4) :=
by
  intros h1 h2
  let V_s := downstream_speed - V_b
  have h3 : V_s = 3 := by
    rw [h1, h2]
    rfl
  sorry

end boat_upstream_speed_l479_479292


namespace bowling_ball_surface_area_l479_479293

theorem bowling_ball_surface_area (diameter : ℝ) (h : diameter = 9) :
    let r := diameter / 2
    let surface_area := 4 * Real.pi * r^2
    surface_area = 81 * Real.pi := by
  sorry

end bowling_ball_surface_area_l479_479293


namespace celina_total_cost_l479_479679

def hoodieCost : ℝ := 80
def hoodieTaxRate : ℝ := 0.05

def flashlightCost := 0.20 * hoodieCost
def flashlightTaxRate : ℝ := 0.10

def bootsInitialCost : ℝ := 110
def bootsDiscountRate : ℝ := 0.10
def bootsTaxRate : ℝ := 0.05

def waterFilterCost : ℝ := 65
def waterFilterDiscountRate : ℝ := 0.25
def waterFilterTaxRate : ℝ := 0.08

def campingMatCost : ℝ := 45
def campingMatDiscountRate : ℝ := 0.15
def campingMatTaxRate : ℝ := 0.08

def backpackCost : ℝ := 105
def backpackTaxRate : ℝ := 0.08

def totalCost : ℝ := 
  let hoodieTotal := (hoodieCost * (1 + hoodieTaxRate))
  let flashlightTotal := (flashlightCost * (1 + flashlightTaxRate))
  let bootsTotal := ((bootsInitialCost * (1 - bootsDiscountRate)) * (1 + bootsTaxRate))
  let waterFilterTotal := ((waterFilterCost * (1 - waterFilterDiscountRate)) * (1 + waterFilterTaxRate))
  let campingMatTotal := ((campingMatCost * (1 - campingMatDiscountRate)) * (1 + campingMatTaxRate))
  let backpackTotal := (backpackCost * (1 + backpackTaxRate))
  hoodieTotal + flashlightTotal + bootsTotal + waterFilterTotal + campingMatTotal + backpackTotal

theorem celina_total_cost: totalCost = 413.91 := by
  sorry

end celina_total_cost_l479_479679


namespace findObtuseAngle_l479_479364

-- Condition: given that α is obtuse
def isObtuse (α : Real) := 90 < α ∧ α < 180

-- Definition of the given condition
def givenCondition (α : Real) := sin α * (1 + sqrt 3 * tan (10 * Real.pi / 180)) = 1

-- The theorem to prove
theorem findObtuseAngle :
  isObtuse (140 * Real.pi / 180) ∧ givenCondition (140 * Real.pi / 180) :=
sorry

end findObtuseAngle_l479_479364


namespace cylindrical_to_rectangular_example_l479_479342

noncomputable def cylindricalToRectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * cos θ, r * sin θ, z)

theorem cylindrical_to_rectangular_example :
  cylindricalToRectangular 4 π 6 = (-4, 0, 6) :=
by
  sorry

end cylindrical_to_rectangular_example_l479_479342


namespace sum_of_values_of_a_l479_479579

theorem sum_of_values_of_a : 
  (∀ (a : ℝ), (a + 8) ^ 2 - 4 * 4 * 9 = 0 → 
  (a^2 + 16 * a - 80 = 0 → a = -20 ∨ a = 4)) →
  wsnd_of_roots_sum ((-1) * coeff_a) = -16 := 
by
  intros h1 h2
  rw [coeff_a, -1 * 16]
  sorry

end sum_of_values_of_a_l479_479579


namespace coprime_3x3_grid_l479_479829

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def valid_3x3_grid (grid : Array (Array ℕ)) : Prop :=
  -- Check row-wise, column-wise, and diagonal adjacencies for coprimality
  ∀ i j, i < 3 → j < 3 →
  (if j < 2 then is_coprime (grid[i][j]) (grid[i][j+1]) else True) ∧
  (if i < 2 then is_coprime (grid[i][j]) (grid[i+1][j]) else True) ∧
  (if i < 2 ∧ j < 2 then is_coprime (grid[i][j]) (grid[i+1][j+1]) else True) ∧
  (if i < 2 ∧ j > 0 then is_coprime (grid[i][j]) (grid[i+1][j-1]) else True)

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10], #[5, 7, 11], #[6, 13, 12]]

theorem coprime_3x3_grid : valid_3x3_grid grid := sorry

end coprime_3x3_grid_l479_479829


namespace general_solution_of_differential_eq_l479_479712

noncomputable def y (x : ℝ) (C1 C2 : ℝ) : ℝ :=
  C1 * exp (-x) + C2 * exp (-2 * x) + ((-3 / 10) * x + 17 / 50) * cos x + ((1 / 10) * x + 3 / 25) * sin x

theorem general_solution_of_differential_eq (C1 C2 : ℝ) :
  ∀ (x : ℝ), 
    (derivative (derivative (y x C1 C2)) + 3 * derivative (y x C1 C2) + 2 * y x C1 C2) = x * sin x := 
by 
  sorry

end general_solution_of_differential_eq_l479_479712


namespace proof_problem_l479_479432

variable (a b c : ℝ)

-- Condition for Option A
def option_A_condition (a b : ℝ) : Prop :=
ab ≠ 0 ∧ a < b

def option_A_statement (a b : ℝ) : Prop :=
1 / a > 1 / b

-- Condition for Option B
def option_B_condition (a : ℝ) : Prop :=
0 < a ∧ a < 1

def option_B_statement (a : ℝ) : Prop :=
a^3 < a

-- Condition for Option C
def option_C_condition (a b : ℝ) : Prop :=
a > b ∧ b > 0

def option_C_statement (a b : ℝ) : Prop :=
(b + 1) / (a + 1) < b / a

-- Condition for Option D
def option_D_condition (a b c : ℝ) : Prop :=
c < b ∧ b < a ∧ ac < 0

def option_D_statement (a b c : ℝ) : Prop :=
 cb^2 < ab^2

theorem proof_problem :
  ¬option_A_statement a b ↔ option_A_condition a b ∧
  option_B_statement a ↔ option_B_condition a ∧
  ¬option_C_statement a b ↔ option_C_condition a b ∧
  ¬option_D_statement a b c ↔ option_D_condition a b c :=
by
  intros 
  sorry

end proof_problem_l479_479432


namespace henry_books_donation_l479_479426

theorem henry_books_donation
  (initial_books : ℕ := 99)
  (room_books : ℕ := 21)
  (coffee_table_books : ℕ := 4)
  (cookbook_books : ℕ := 18)
  (boxes : ℕ := 3)
  (picked_up_books : ℕ := 12)
  (final_books : ℕ := 23) :
  (initial_books - final_books + picked_up_books - (room_books + coffee_table_books + cookbook_books)) / boxes = 15 :=
by
  sorry

end henry_books_donation_l479_479426


namespace algebraic_expression_value_l479_479125

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 :=
by
  sorry

end algebraic_expression_value_l479_479125


namespace find_k_l479_479819

-- Define the equation for the root
def equation (x : ℝ) : Prop := Real.log10 x = 2 - x

-- Define the interval condition
def interval_condition (x₀ : ℝ) (k : ℤ) : Prop := x₀ ∈ Set.Ioo (k - 1 : ℝ) k

-- Main statement in Lean: proof that k = 2
theorem find_k (x₀ : ℝ) (k : ℤ) (hk : k ∈ Set.univ) : equation x₀ → interval_condition x₀ k → k = 2 :=
by
  sorry

end find_k_l479_479819


namespace derivative_at_one_l479_479813

def f (x : ℝ) : ℝ := x^3 - f' 1 * x^2 + 3

theorem derivative_at_one :
  (deriv f 1) = 1 := by
sorry

end derivative_at_one_l479_479813


namespace range_of_a_l479_479396

theorem range_of_a (f : ℝ → ℝ) (a : ℝ):
  (∀ x, f x = f (-x)) →
  (∀ x y, 0 ≤ x → x < y → f x ≤ f y) →
  (∀ x, 1/2 ≤ x ∧ x ≤ 1 → f (a * x + 1) ≤ f (x - 2)) →
  -2 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l479_479396


namespace smallest_positive_multiple_of_45_l479_479611

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ (∀ y : ℕ, y > 0 → 45 * y ≥ 45 * x) ∧ 45 * x = 45 :=
by
  use 1
  split
  · apply Nat.one_pos
  · split
    · intros y hy
      apply mul_le_mul
      · apply Nat.one_le_of_lt hy
      · apply le_refl
      · apply le_of_lt
        apply Nat.pos_of_ne_zero
        intro h
        exact hy (h.symm ▸ rfl)
      · apply le_of_lt
        apply Nat.pos_of_ne_zero
        intro h
        exact hy (h.symm ▸ rfl)
    · apply rfl
  sorry

end smallest_positive_multiple_of_45_l479_479611


namespace part_1_part_2_l479_479782

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 9 / 4 ∧ (∀ n ≥ 1, 2 * a (n + 1) * a n - 7 * a (n + 1) - 3 * a n + 12 = 0)

def seq_c (a c : ℕ → ℝ) : Prop :=
  (∀ n, c n = a n - 2)

theorem part_1 (a : ℕ → ℝ) (c : ℕ → ℝ) (h_a : seq_a a) (h_c : seq_c a c) :
  ∀ n ≥ 1, c n = 1 / (3^n + 1) := 
sorry

def seq_b (a b : ℕ → ℝ) : Prop :=
  ∀ n, b n = (n^2 / (n + 1)) * a n

def floor_sum_b (b : ℕ → ℝ) : ℕ → ℝ 
| 0 := 0
| (n + 1) := floor (b (n + 1)) + floor_sum_b n

theorem part_2 (a b : ℕ → ℝ) (h_a : seq_a a) (h_b : seq_b a b) :
  ∀ n, floor_sum_b b n ≤ 2019 → n ≤ 45 :=
sorry

end part_1_part_2_l479_479782


namespace arrange_consecutive_integers_no_common_divisors_l479_479846

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def adjacent (i j : ℕ) (n m : ℕ) : Prop :=
  (abs (i - n) = 1 ∧ j = m) ∨
  (i = n ∧ abs (j - m) = 1) ∨
  (abs (i - n) = 1 ∧ abs (j - m) = 1)

theorem arrange_consecutive_integers_no_common_divisors :
  let grid := [[8, 9, 10], [5, 7, 11], [6, 13, 12]] in
  ∀ i j n m, i < 3 → j < 3 → n < 3 → m < 3 →
  adjacent i j n m →
  is_coprime (grid[i][j]) (grid[n][m]) :=
by
  -- This is where you would prove the theorem
  sorry

end arrange_consecutive_integers_no_common_divisors_l479_479846


namespace find_shares_l479_479039

def shareA (B : ℝ) : ℝ := 3 * B
def shareC (B : ℝ) : ℝ := B - 25
def shareD (A B : ℝ) : ℝ := A + B - 10
def total_share (A B C D : ℝ) : ℝ := A + B + C + D

theorem find_shares :
  ∃ (A B C D : ℝ),
  A = 744.99 ∧
  B = 248.33 ∧
  C = 223.33 ∧
  D = 983.32 ∧
  A = shareA B ∧
  C = shareC B ∧
  D = shareD A B ∧
  total_share A B C D = 2200 := 
sorry

end find_shares_l479_479039


namespace maximize_expression_l479_479484

noncomputable def S (n : ℕ) : ℝ := (1 / n : ℝ)

theorem maximize_expression : ∃ n : ℕ, n > 0 ∧ 
  ∀ m : ℕ, m > 0 → 
    (S m m * S m (S m)) / (1 + 10 * (S m S m)^2) ≤ 
    (S n n * S n (S n)) / (1 + 10 * (S n S n)^2) := by
  sorry

end maximize_expression_l479_479484


namespace dropped_score_l479_479281

variable (A B C D : ℕ)

theorem dropped_score (h1 : A + B + C + D = 180) (h2 : A + B + C = 150) : D = 30 := by
  sorry

end dropped_score_l479_479281


namespace mark_sideline_time_l479_479135

def total_game_time : ℕ := 90
def initial_play : ℕ := 20
def second_play : ℕ := 35
def total_play_time : ℕ := initial_play + second_play
def sideline_time : ℕ := total_game_time - total_play_time

theorem mark_sideline_time : sideline_time = 35 := by
  sorry

end mark_sideline_time_l479_479135


namespace A_runs_4_times_faster_than_B_l479_479294

theorem A_runs_4_times_faster_than_B (v_A v_B : ℕ) (k : ℕ) (h1 : v_A = k * v_B) 
    (h2 : ∀ t: nat, t > 0 →  (100 : ℕ) / v_A = (25 : ℕ) / v_B) : k = 4 := 
sorry

end A_runs_4_times_faster_than_B_l479_479294


namespace triangle_cos_sin_eq_l479_479474

-- Lean code equivalent to the given proof problem

theorem triangle_cos_sin_eq {A B C : ℝ} (h_triangle: A + B + C = π) 
  (C_obtuse: C > π / 2)
  (h1: cos A ^ 2 + cos C ^ 2 + 2 * sin A * sin C * cos B = 17 / 9)
  (h2: cos C ^ 2 + cos B ^ 2 + 2 * sin C * sin B * cos A = 16 / 7) :
  cos B ^ 2 + cos A ^ 2 + 2 * sin B * sin A * cos C = (197 - 30 * Real.sqrt 35) / 441 :=
sorry

end triangle_cos_sin_eq_l479_479474


namespace simple_interest_rate_l479_479279

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (I : ℝ) : 
  T = 6 → I = (7/6) * P - P → I = P * R * T / 100 → R = 100 / 36 :=
by
  intros T_eq I_eq simple_interest_eq
  sorry

end simple_interest_rate_l479_479279


namespace range_of_a_l479_479036

theorem range_of_a (a x y : ℝ)
  (h1 : x + 3 * y = 2 + a)
  (h2 : 3 * x + y = -4 * a)
  (hxy : x + y > 2) : a < -2 := 
sorry

end range_of_a_l479_479036


namespace line_passes_fixed_point_l479_479044

theorem line_passes_fixed_point (a b : ℝ) (h : a + 2 * b = 1) : 
  a * (1/2) + 3 * (-1/6) + b = 0 :=
by
  sorry

end line_passes_fixed_point_l479_479044


namespace impossible_partition_l479_479468

theorem impossible_partition (S : Finset ℕ) (hS : S = Finset.range 34 \ Finset.singleton 0) :
  ¬(∃ (G : Finset (Finset (ℕ × ℕ × ℕ))),
      (∀ g ∈ G, ∃ a b c, g = Finset.singleton (a, b, c) ∧ (a ∈ S ∧ b ∈ S ∧ c ∈ S) ∧ (a = b + c ∨ b = a + c ∨ c = a + b)) ∧ 
      (∀ x ∈ S, ∃! g ∈ G, ∃ a b c, g = Finset.singleton (a, b, c) ∧ (x = a ∨ x = b ∨ x = c)) ∧
      G.card = 11) :=
by sorry

end impossible_partition_l479_479468


namespace problem_statement_l479_479181

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.exp x + a * Real.sin x + b

theorem problem_statement (a b : ℝ) : 
  (∀ x : ℝ, 0 ≤ x → f x 1 b ≥ 0) ∧ (∃ a b, f' 0 = 1 ∧ f 0 = 1 - b) ∧ (∃ y, y = Real.exp y - 1) → b ≥ -1 ∧ a = 0 ∧ b = -2 ∧ ∀ x : ℝ, 0 < x → f x 0 (-2) > Real.log x :=
by
  sorry

end problem_statement_l479_479181


namespace part1_part2_part3_l479_479384

-- Define the sequence a_n with the given condition
noncomputable def a_n (n : ℕ) : ℝ := 
  if n = 0 then 0 else classical.some (exists_unique_of_exists_of_unique (
    exists_unique_of_exists_of_unique (λ x : ℝ, x^n + n*x = 1) 
  ))

-- 1. Prove that a_1 = 1 / 2 and a_2 = sqrt(2) - 1
theorem part1: 
  a_n 1 = 1 / 2 ∧ a_n 2 = Real.sqrt 2 - 1 := 
sorry

-- 2. Prove that 0 < a_n < 1 for all positive n
theorem part2 (n : ℕ) (hn : n > 0): 
  0 < a_n n ∧ a_n n < 1 := 
sorry

-- 3. Prove that a_1^2 + a_2^2 + ... + a_n^2 < 1 for all positive n
theorem part3 (n : ℕ) (hn : n > 0): 
  ∑ i in Finset.range n, (a_n (i+1))^2 < 1 := 
sorry

end part1_part2_part3_l479_479384


namespace number_of_tangent_lines_l479_479243

noncomputable def circle_center : ℝ × ℝ := (3, 3)
noncomputable def circle_radius : ℝ := 2 * Real.sqrt 2

-- Here we define what it means for a line to have equal intercepts on both axes
def equal_intercepts (line : ℝ → ℝ → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, line x y ↔ x + y = a

-- The statement that we need to prove: there are four such tangent lines
theorem number_of_tangent_lines :
  let C := circle_center,
      radius := circle_radius,
      tangent_lines_count := { line : ℝ → ℝ → Prop // equal_intercepts line ∧ (∀ x y : ℝ, line x y → (x - 3)^2 + (y - 3)^2 = 8) } in
  tangent_lines_count.card = 4 :=
by sorry

end number_of_tangent_lines_l479_479243


namespace real_solution_count_l479_479006

theorem real_solution_count : 
  ∃ x : ℝ → ℂ, 3 * x^3 - 21 * ⌊ x ⌋ - 54 = 0 ∧ x ∈ ℝ := 
begin
  sorry
end

end real_solution_count_l479_479006


namespace arithmetic_geometric_sum_l479_479742

def a (n : ℕ) : ℕ := 3 * n - 2
def b (n : ℕ) : ℕ := 3 ^ (n - 1)

theorem arithmetic_geometric_sum :
  a (b 1) + a (b 2) + a (b 3) = 33 := by
  sorry

end arithmetic_geometric_sum_l479_479742


namespace haley_car_distance_l479_479972

theorem haley_car_distance (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used_gallons : ℕ) (distance_covered_miles : ℕ) :
  fuel_ratio = 4 → distance_ratio = 7 → fuel_used_gallons = 44 → distance_covered_miles = (distance_ratio * fuel_used_gallons / fuel_ratio)
  → distance_covered_miles = 77 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  dsimp at h4
  linarith

end haley_car_distance_l479_479972


namespace find_x_gx_eq_g3gx_l479_479479

-- Define the function g(x)
def g (x : ℝ) : ℝ := -3 * Real.cos (Real.pi * x)

-- Formulate the Lean 4 statement of the problem
theorem find_x_gx_eq_g3gx :
  ∃ (x1 x2 : ℝ), -1 ≤ x1 ∧ x1 ≤ 1 ∧ -1 ≤ x2 ∧ x2 ≤ 1 ∧ x1 ≠ x2 ∧
  g (g (g x1)) = g x1 ∧ g (g (g x2)) = g x2 :=
sorry

end find_x_gx_eq_g3gx_l479_479479


namespace bethany_ride_hours_over_two_weeks_l479_479671

theorem bethany_ride_hours_over_two_weeks :
  -- Definitions based on conditions
  (let monWedFriHours := 1 in
   let tueThuHours := 0.5 in
   let satHours := 2 in
   -- Calculate total hours per week
   let weeklyTotal := (3 * monWedFriHours) + (2 * tueThuHours) + (1 * satHours) in
   -- Calculate total hours over two weeks
   2 * weeklyTotal = 12) := sorry

end bethany_ride_hours_over_two_weeks_l479_479671


namespace find_point_P_l479_479762

noncomputable def P := 
  let a := \frac{\sqrt{3}}{3}
  let b :=  -\frac{2\sqrt{6}}{3}
  let c := -\frac{\sqrt{3}}{3}
  let d := \frac{2\sqrt{6}}{3}
  (a, b, c, d)

theorem find_point_P :
  ∃ P ∈ { (x, y) | x^2 + \frac{y^2}{4} = 1 }, 
  dist P (0, \sqrt{3}) = dist_to_line P ( \sqrt{2} x + 2 \sqrt{3} ) :=
begin
  let F := (0, \sqrt{3}),
  let l := (y, x) | y = \sqrt{2} x + 2 \sqrt{3},
  let P1 := P.1,
  let P2 := P.2,
  let A := \sqrt{ (P1 - 0)^2 + (P2 - \sqrt{3})^2 },
  let B := (|\sqrt{2} P1 - 2 P2 + 2 \sqrt{3}| / \sqrt{3}),
  use (P.1, P.2),
  simp *,
  exact A = B,
end

end find_point_P_l479_479762


namespace absent_children_count_l479_479192

theorem absent_children_count (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ)
    (absent_children : ℕ) (total_bananas : ℕ) (present_children : ℕ) :
    total_children = 640 →
    bananas_per_child = 2 →
    extra_bananas_per_child = 2 →
    total_bananas = (total_children * bananas_per_child) →
    present_children = (total_children - absent_children) →
    total_bananas = (present_children * (bananas_per_child + extra_bananas_per_child)) →
    absent_children = 320 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end absent_children_count_l479_479192


namespace circumscribed_circle_radius_l479_479194

-- Define the square ABCD with side length 1
structure Square (A B C D : Type) :=
(side_eq : A = B)
(side_len : dist A B = 1 ∧ dist B C = 1 ∧ dist C D = 1 ∧ dist D A = 1)
(right_angles : ∀ (P Q R : Type), P ∈ [A, B, C, D] ∧ Q ∈ [A, B, C, D] ∧ R ∈ [A, B, C, D] ∧ dist P Q = 1 ∧ dist Q R = 1 → angle P Q R = π/2)

-- Define the equilateral triangle AKB on side AB
structure EquilateralTriangle (A K B : Type) :=
(side_eq : A = K = B)
(side_akb : dist A K = dist K B = dist A B = 1)

-- Define the EquilateralTriangle CKD
structure EquilateralTriangle' (C K D : Type) :=
(side_eq' : C = K = D)
(dist_ckd : dist C K = dist K D = dist C D = 1)

-- Define the circumscribed circle around triangle CKD
def circum_circle_radius (C K D : Type) (t : EquilateralTriangle' C K D) : Type :=
  ∃ r : Type, r = 1 ∧ ∀ x : Type, x ∈ [C, K, D] → dist (circumcenter C K D) x = r

-- Main statement
theorem circumscribed_circle_radius (A B C D K : Type) (sq : Square A B C D) (eqt : EquilateralTriangle A K B) : ∃ (r : Type), r = 1 ∧ circum_circle_radius C K D (EquilateralTriangle' C K D) :=
  sorry

end circumscribed_circle_radius_l479_479194


namespace total_payment_360_l479_479283

noncomputable def q : ℝ := 12
noncomputable def p_wage : ℝ := 1.5 * q
noncomputable def p_hourly_rate : ℝ := q + 6
noncomputable def h : ℝ := 20
noncomputable def total_payment_p : ℝ := p_wage * h -- The total payment when candidate p is hired
noncomputable def total_payment_q : ℝ := q * (h + 10) -- The total payment when candidate q is hired

theorem total_payment_360 : 
  p_wage = p_hourly_rate ∧ 
  total_payment_p = total_payment_q ∧ 
  total_payment_p = 360 := by
  sorry

end total_payment_360_l479_479283


namespace remainder_equiv_l479_479614

theorem remainder_equiv (x : ℤ) (h : ∃ k : ℤ, x = 95 * k + 31) : ∃ m : ℤ, x = 19 * m + 12 := 
sorry

end remainder_equiv_l479_479614


namespace find_pink_roses_l479_479872

variable (red_f_1 pink_f_2 yellow_f_3 orange_f_4 total_roses l_pick_red l_pick_pink l_pick_yellow l_pick_orange : ℕ)
variable (h1 : red_f_1 = 12)
variable (h2 : yellow_f_3 = 20)
variable (h3 : orange_f_4 = 8)
variable (h4 : total_roses = 22)
variable (h5 : l_pick_red = (0.5 * red_f_1).toInt)
variable (h6 : l_pick_pink = (0.5 * pink_f_2).toInt)
variable (h7 : l_pick_yellow = (0.25 * yellow_f_3).toInt)
variable (h8 : l_pick_orange = (0.25 * orange_f_4).toInt)

theorem find_pink_roses (h1 : red_f_1 = 12) (h2 : yellow_f_3 = 20) (h3 : orange_f_4 = 8)
  (h4 : total_roses = 22) (h5 : l_pick_red = (0.5 * red_f_1).toInt) 
  (h6 : l_pick_pink = (0.5 * pink_f_2).toInt) (h7 : l_pick_yellow = (0.25 * yellow_f_3).toInt)
  (h8 : l_pick_orange = (0.25 * orange_f_4).toInt) : 
  pink_f_2 = 18 := 
by
  sorry

end find_pink_roses_l479_479872


namespace height_of_intersection_of_lines_l479_479129

theorem height_of_intersection_of_lines
  (h₁ : ℝ) (h₂ : ℝ) (d : ℝ) :
  h₁ = 30 → h₂ = 50 → d = 150 → 
  let y1 := (-1 / 5) * d + h₁ in
  let y2 := (1 / 3) * d in
  y1 = y2 → y1 = 1875 / 100 := 
by
  intros h1_val h2_val d_val eq_intersect
  sorry

end height_of_intersection_of_lines_l479_479129


namespace product_fraction_l479_479336

theorem product_fraction :
  (∏ n in (Finset.range 98).map (fun x => x + 3), (1 - (1:ℚ)/n)) = (1:ℚ)/50 :=
by
  sorry

end product_fraction_l479_479336


namespace compute_expression_l479_479682

theorem compute_expression : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end compute_expression_l479_479682


namespace percentage_increase_in_length_is_10_l479_479963

variables (L B : ℝ) -- original length and breadth
variables (length_increase_percentage breadth_increase_percentage area_increase_percentage : ℝ)

noncomputable def new_length (x : ℝ) : ℝ := L * (1 + x / 100)
noncomputable def new_breadth : ℝ := B * 1.25
noncomputable def new_area (x : ℝ) : ℝ := new_length L B x * new_breadth B
noncomputable def increased_area : ℝ := L * B * 1.375

theorem percentage_increase_in_length_is_10 :
 (breadth_increase_percentage = 25) →
 (area_increase_percentage = 37.5) →
 (new_area L B 10 = increased_area L B) → length_increase_percentage = 10
:= by
  sorry

end percentage_increase_in_length_is_10_l479_479963


namespace removable_discontinuity_f_at_2_removable_discontinuity_g_at_4_jump_discontinuity_phi_at_0_jump_discontinuity_h_at_1_l479_479879

-- Problem for function f with removable discontinuity at x = 2
theorem removable_discontinuity_f_at_2 :
  let f : ℝ → ℝ := λ x, if x = 2 then 1 else 3 * x in
  ∃ a, ∀ (ε > 0), ∃ (δ > 0), ∀ x, |x - 2| < δ → |f x - a| < ε ∧ a ≠ f 2 := sorry

-- Problem for function g with removable discontinuity at x = 4
theorem removable_discontinuity_g_at_4 :
  let g : ℝ → ℝ := λ x, (x^2 - 16) / (x - 4) in
  ∃ a, ∀ (ε > 0), ∃ (δ > 0), ∀ x, |x - 4| < δ → |g x - a| < ε ∧ a ≠ g 4 := sorry

-- Problem for function φ with jump discontinuity at x = 0
theorem jump_discontinuity_phi_at_0 :
  let φ : ℝ → ℝ := λ x, if x ≤ 0 then x^2 else x + 1 in
  ∃ l₁ l₂, l₁ ≠ l₂ ∧ (∀ (ε > 0), ∃ (δ > 0), ∀ x, 0 < |x| < δ → |x| ≤ 0 → |φ x - l₁| < ε) ∧ (∀ (ε > 0), ∃ (δ > 0), ∀ x, 0 < |x| < δ → x > 0 → |φ x - l₂| < ε) := sorry

-- Problem for function h with jump discontinuity at x = 1
theorem jump_discontinuity_h_at_1 :
  let h : ℝ → ℝ := λ x, (|x - 1|) / (x - 1) in
  ∃ l₁ l₂, l₁ ≠ l₂ ∧ (∀ (ε > 0), ∃ (δ > 0), ∀ x, 0 < |x - 1| < δ → x < 1 → |h x - l₁| < ε) ∧ (∀ (ε > 0), ∃ (δ > 0), ∀ x, 0 < |x - 1| < δ → x > 1 → |h x - l₂| < ε) := sorry

end removable_discontinuity_f_at_2_removable_discontinuity_g_at_4_jump_discontinuity_phi_at_0_jump_discontinuity_h_at_1_l479_479879


namespace min_value_is_four_l479_479180

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_is_four (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z h1 h2 h3 h4 = 4 :=
sorry

end min_value_is_four_l479_479180


namespace ellipse_equation_no_parallel_line_l479_479743

theorem ellipse_equation (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : ∃ x y : ℝ, x = 2 ∧ y = 3 ∧ ((x^2) / (a^2)) + ((y^2) / (b^2)) = 1)
  (h₄ : ∀ x y : ℝ, ((x - 2)^2 + y^2 = 4) → (x, y) ≠ (2, 0)) :
  (a = 4) ∧ (b^2 = 12) ∧ (∀ x y, ((x^2) / 16) + ((y^2) / 12) = 1) := 
sorry

theorem no_parallel_line (t : ℝ)
  (h₁ : ∀ x y : ℝ, ((x^2) / 16) + ((y^2) / 12) = 1 → ∃ x : ℝ, y = (3/2) * x + t)
  (h₂ : -4 * real.sqrt 3 ≤ t ∧ t ≤ 4 * real.sqrt 3)
  (h₃ : ∀ t, 4 = (abs t) / (real.sqrt (9 / 4 + 1)) → t = 2 * real.sqrt 13 ∨ t = -2 * real.sqrt 13) :
  ¬∀ t, t = 2 * real.sqrt 13 ∨ t = -2 * real.sqrt 13  := 
sorry

end ellipse_equation_no_parallel_line_l479_479743


namespace roberto_starting_salary_l479_479929

-- Given conditions as Lean definitions
def current_salary : ℝ := 134400
def previous_salary (S : ℝ) : ℝ := 1.40 * S

-- The proof problem statement
theorem roberto_starting_salary (S : ℝ) 
    (h1 : current_salary = 1.20 * previous_salary S) : 
    S = 80000 :=
by
  -- We will insert the proof here
  sorry

end roberto_starting_salary_l479_479929


namespace sum_of_d_values_l479_479555

theorem sum_of_d_values (d : ℝ) :
  (∃ d : ℝ, (d^2 + d^2 = (12 - d)^2 + (2d - 6)^2)) →
  ∑ x in {d | (d^2 + d^2 = (12 - d)^2 + (2d - 6)^2)}, x = 16 :=
by
  sorry

end sum_of_d_values_l479_479555


namespace hyperbola_equation_hyperbola_eccentricity_l479_479030

-- Define the geometric properties and conditions
def asymptote : ∀ (x y : ℝ), 3 * x + 4 * y = 0 := λ x y, 3 * x + 4 * y = 0

def focus : (ℝ × ℝ) := (4, 0)

-- Define the statements to be proved
theorem hyperbola_equation : 
  ∃ a b : ℝ, a = 4 ∧ b = 16 / 3 ∧  
  (∀ x y : ℝ, ((x^2) / (a^2)) - ((y^2) / (b^2)) = 1) :=
begin
  -- Here, provide the necessary proof steps according to the given conditions, 
  -- but we are skipping the proof steps since only the statement is required.
  sorry,
end

theorem hyperbola_eccentricity :
  ∃ e : ℝ, e = 5 / 4 :=
begin
  -- Here, provide the necessary proof steps according to the given conditions,
  -- but we are skipping the proof steps since only the statement is required.
  sorry,
end

end hyperbola_equation_hyperbola_eccentricity_l479_479030


namespace triangle_ABC_is_right_triangle_l479_479047

variable (A B C O : Type)
variable [normed_add_comm_group O]
variables (OB OC OA : O)
variable (ABC : Triangle A B C)

-- Condition: Given point O and the vectors \
def condition : Prop := 
  ∥ OB - OC ∥ = ∥ OB + OC - 2 * OA ∥

-- Prove that triangle ABC is a right triangle
theorem triangle_ABC_is_right_triangle 
  (h : condition OB OC OA) : 
  IsRightTriangle ABC :=
sorry

end triangle_ABC_is_right_triangle_l479_479047


namespace min_value_is_four_l479_479179

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_is_four (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z h1 h2 h3 h4 = 4 :=
sorry

end min_value_is_four_l479_479179


namespace green_pill_cost_is_21_l479_479659

-- Definitions based on conditions
def number_of_days : ℕ := 21
def total_cost : ℕ := 819
def daily_cost : ℕ := total_cost / number_of_days
def green_pill_cost (pink_pill_cost : ℕ) : ℕ := pink_pill_cost + 3

-- Given pink pill cost is x, then green pill cost is x + 3
-- We need to prove that for some x, the daily cost of the pills equals 39, and thus green pill cost is 21

theorem green_pill_cost_is_21 (pink_pill_cost : ℕ) (h : daily_cost = (green_pill_cost pink_pill_cost) + pink_pill_cost) :
    green_pill_cost pink_pill_cost = 21 :=
by
  sorry

end green_pill_cost_is_21_l479_479659


namespace quadrilateral_AB_length_l479_479902

/-- Let ABCD be a quadrilateral with BC = CD = DA = 1, ∠DAB = 135°, and ∠ABC = 75°. 
    Prove that AB = (√6 - √2) / 2.
-/
theorem quadrilateral_AB_length (BC CD DA : ℝ) (angle_DAB angle_ABC : ℝ) (h1 : BC = 1)
    (h2 : CD = 1) (h3 : DA = 1) (h4 : angle_DAB = 135) (h5 : angle_ABC = 75) :
    AB = (Real.sqrt 6 - Real.sqrt 2) / 2 := by
    sorry

end quadrilateral_AB_length_l479_479902


namespace intervals_of_monotonicity_k_neg_intervals_of_monotonicity_k_pos_range_of_a_l479_479769

-- Define the function f(x) = k * x^2 / e^x
def f (k x : ℝ) : ℝ := (k * x^2) / Real.exp x

-- Problem 1: Intervals of monotonicity
-- For k < 0, prove the intervals of increasing and decreasing
theorem intervals_of_monotonicity_k_neg (k x : ℝ) (hk : k < 0) :
  (f' k x > 0 ↔ x < 0 ∨ x > 2) ∧ (f' k x < 0 ↔ 0 < x ∧ x < 2) :=
sorry

-- For k > 0, prove the intervals of increasing and decreasing
theorem intervals_of_monotonicity_k_pos (k x : ℝ) (hk : k > 0) :
  (f' k x > 0 ↔ 0 < x ∧ x < 2) ∧ (f' k x < 0 ↔ x < 0 ∨ x > 2) :=
sorry

-- Problem 2: Range of a when k = 1
-- Prove that there exists x > 0 such that ln(f(x)) > ax implies a < (2/e) - 1
theorem range_of_a (x a : ℝ) (hk : k = 1) (hx : x > 0) (h : Real.log (f 1 x) > a * x) :
  a < (2 / Real.exp 1) - 1 :=
sorry

end intervals_of_monotonicity_k_neg_intervals_of_monotonicity_k_pos_range_of_a_l479_479769


namespace option_A_incorrect_l479_479788

variables (a b : Type) [linear_ordered_field a] [linear_ordered_field b]
variables (α β γ : Type)

-- Conditions from the problem
variable (p1 : a ∥ α)
variable (p2 : a ∥ b)
variable (p3 : ¬(b ∥ α))

-- Incorrect statement to prove as a condition leads to what answer, which means we need that:
-- Prove: From the conditions: (a ∥ α), (a ∥ b), ¬(b ∥ α) => we get what?
theorem option_A_incorrect : (a ∥ α) → (a ∥ b) → ¬(b ∥ α) → ¬(b ⊥ α) :=
by
    sorry

end option_A_incorrect_l479_479788


namespace solve_percentage_increase_length_l479_479962

def original_length (L : ℝ) : Prop := true
def original_breadth (B : ℝ) : Prop := true

def new_breadth (B' : ℝ) (B : ℝ) : Prop := B' = 1.25 * B

def new_length (L' : ℝ) (L : ℝ) (x : ℝ) : Prop := L' = L * (1 + x / 100)

def original_area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

def new_area (A' : ℝ) (A : ℝ) : Prop := A' = 1.375 * A

def percentage_increase_length (x : ℝ) : Prop := x = 10

theorem solve_percentage_increase_length (L B A A' L' B' x : ℝ)
  (hL : original_length L)
  (hB : original_breadth B)
  (hB' : new_breadth B' B)
  (hL' : new_length L' L x)
  (hA : original_area L B A)
  (hA' : new_area A' A)
  (h_eqn : L' * B' = A') :
  percentage_increase_length x :=
by
  sorry

end solve_percentage_increase_length_l479_479962


namespace coefficient_x3y_in_expansion_l479_479868

theorem coefficient_x3y_in_expansion : 
  coefficient (expand (x^2 + x - y)^3) (x^3 * y) = -6 := 
sorry

end coefficient_x3y_in_expansion_l479_479868


namespace cinema_total_cost_l479_479510

theorem cinema_total_cost 
  (total_students : ℕ)
  (ticket_cost : ℕ)
  (half_price_interval : ℕ)
  (free_interval : ℕ)
  (half_price_cost : ℕ)
  (free_cost : ℕ)
  (total_cost : ℕ)
  (H_total_students : total_students = 84)
  (H_ticket_cost : ticket_cost = 50)
  (H_half_price_interval : half_price_interval = 12)
  (H_free_interval : free_interval = 35)
  (H_half_price_cost : half_price_cost = ticket_cost / 2)
  (H_free_cost : free_cost = 0)
  (H_total_cost : total_cost = 3925) :
  total_cost = ((total_students / half_price_interval) * half_price_cost +
                (total_students / free_interval) * free_cost +
                (total_students - (total_students / half_price_interval + total_students / free_interval)) * ticket_cost) :=
by 
  sorry

end cinema_total_cost_l479_479510


namespace arithmetic_sequence_value_l479_479865

variables {α : Type*} [linear_ordered_comm_ring α]
variables (a : ℕ → α)

noncomputable def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n m : ℕ, ∃ d : α, a (n + m) = a n + m * d

theorem arithmetic_sequence_value {a : ℕ → ℝ}
  (h_arith : is_arithmetic_sequence a)
  (h : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
sorry

end arithmetic_sequence_value_l479_479865


namespace count_equilateral_triangles_in_lattice_l479_479011

-- Conditions: Define the enlarged hexagonal lattice with unit distance and an additional outer layer.

-- Definitions for the proof
structure HexagonalLattice where
  layer : ℕ → Set (ℕ × ℕ)

def vertices_distance_one (v1 v2 : (ℕ × ℕ)) : Prop :=
  -- Definition specifying distance between vertices is one unit.
  sorry

def is_equilateral_triangle (t : (ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ)) : Prop :=
  -- Definition specifying an equilateral triangle in the lattice.
  sorry

noncomputable def enlargedHexagonalLattice : HexagonalLattice :=
  -- Definition of the enlarged hexagonal lattice with an additional outer layer.
  sorry

-- The question requiring proof.
theorem count_equilateral_triangles_in_lattice :
  (finset.card {t : (ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ) | 
    t ∈ enlargedHexagonalLattice.layer 2 ∧ is_equilateral_triangle t}) = 20 := by
  sorry

end count_equilateral_triangles_in_lattice_l479_479011


namespace dot_product_eq_norm_eq_l479_479403

variable {ℝ : Type} [inner_product_space ℝ ℝ]
variables (a b : ℝ)

-- Define the conditions
constants (m : ℝ) (n : ℝ) (angle_ab : ℝ)
hypothesis length_a : ∥a∥ = 2
hypothesis length_b : ∥b∥ = 1
hypothesis angle_condition : angle a b = real.pi * (2/3)

-- Prove that (2a - b) · a = 9
theorem dot_product_eq :
  (2 • a - b) ⬝ a = 9 :=
sorry

-- Prove that |a + 2b| = 2
theorem norm_eq :
  ∥a + 2 • b∥ = 2 :=
sorry

end dot_product_eq_norm_eq_l479_479403


namespace max_min_K_max_min_2x_plus_y_l479_479399

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

theorem max_min_K (x y : ℝ) (h : circle_equation x y) : 
  - (Real.sqrt 3) / 3 ≤ (y - 1) / (x - 2) ∧ (y - 1) / (x - 2) ≤ (Real.sqrt 3) / 3 :=
by sorry

theorem max_min_2x_plus_y (x y : ℝ) (h : circle_equation x y) :
  1 - Real.sqrt 5 ≤ 2 * x + y ∧ 2 * x + y ≤ 1 + Real.sqrt 5 :=
by sorry

end max_min_K_max_min_2x_plus_y_l479_479399


namespace largest_consecutive_integers_sum_to_45_l479_479240

theorem largest_consecutive_integers_sum_to_45 (x n : ℕ) (h : 45 = n * (2 * x + n - 1) / 2) : n ≤ 9 :=
sorry

end largest_consecutive_integers_sum_to_45_l479_479240


namespace range_of_m_l479_479094

-- Define the two vectors a and b
def vector_a := (1, 2)
def vector_b (m : ℝ) := (m, 3 * m - 2)

-- Define the condition for non-collinearity
def non_collinear (m : ℝ) := ¬ (m / 1 = (3 * m - 2) / 2)

theorem range_of_m (m : ℝ) : non_collinear m ↔ m ≠ 2 :=
  sorry

end range_of_m_l479_479094


namespace harry_total_hours_l479_479012

variable (x h y : ℕ)

theorem harry_total_hours :
  ((h + 2 * y) = 42) → ∃ t, t = h + y :=
  by
    sorry -- Proof is omitted as per the instructions

end harry_total_hours_l479_479012


namespace arrangement_is_correct_l479_479835

-- Definition of adjacency in a 3x3 matrix.
def adjacent (i j k l : Nat) : Prop := 
  (i = k ∧ j = l + 1) ∨ (i = k ∧ j = l - 1) ∨ -- horizontal adjacency
  (i = k + 1 ∧ j = l) ∨ (i = k - 1 ∧ j = l) ∨ -- vertical adjacency
  (i = k + 1 ∧ j = l + 1) ∨ (i = k - 1 ∧ j = l - 1) ∨ -- diagonal adjacency
  (i = k + 1 ∧ j = l - 1) ∨ (i = k - 1 ∧ j = l + 1)   -- diagonal adjacency

-- Definition to check if two numbers share a common divisor greater than 1.
def coprime (x y : Nat) : Prop := Nat.gcd x y = 1

-- The arrangement of the numbers in the 3x3 grid.
def grid := 
  [ [8, 9, 10],
    [5, 7, 11],
    [6, 13, 12] ]

-- Ensure adjacents numbers are coprime
def correctArrangement :=
  (coprime grid[0][0] grid[0][1]) ∧ (coprime grid[0][1] grid[0][2]) ∧
  (coprime grid[1][0] grid[1][1]) ∧ (coprime grid[1][1] grid[1][2]) ∧
  (coprime grid[2][0] grid[2][1]) ∧ (coprime grid[2][1] grid[2][2]) ∧
  (adjacent 0 0 1 1 → coprime grid[0][0] grid[1][1]) ∧
  (adjacent 0 2 1 1 → coprime grid[0][2] grid[1][1]) ∧
  (adjacent 1 0 2 1 → coprime grid[1][0] grid[2][1]) ∧
  (adjacent 1 2 2 1 → coprime grid[1][2] grid[2][1]) ∧
  (coprime grid[0][1] grid[1][0]) ∧ (coprime grid[0][1] grid[1][2]) ∧
  (coprime grid[2][1] grid[1][0]) ∧ (coprime grid[2][1] grid[1][2])

-- Statement to be proven
theorem arrangement_is_correct : correctArrangement := 
  sorry

end arrangement_is_correct_l479_479835


namespace minimum_omega_l479_479117

theorem minimum_omega (ω : ℕ) (h_pos : ω ∈ {n : ℕ | n > 0}) (h_cos_center : ∃ k : ℤ, ω * (π / 6) + (π / 6) = k * π + π / 2) :
  ω = 2 :=
by { sorry }

end minimum_omega_l479_479117


namespace number_of_students_not_liking_any_sport_l479_479140

namespace ProofProblem

variables (total_students : ℕ) (B : ℕ) (C : ℕ) (F : ℕ)
variables (BC : ℕ) (CF : ℕ) (BF : ℕ) (BCF : ℕ)

theorem number_of_students_not_liking_any_sport :
  B = 20 →
  C = 18 →
  F = 12 →
  BC = 8 →
  CF = 6 →
  BF = 5 →
  BCF = 3 →
  total_students = 50 →
  total_students - (B + C + F - BC - CF - BF + BCF) = 16 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8,
  rw [h1, h2, h3, h4, h5, h6, h7, h8],
  norm_num,
end

end ProofProblem

end number_of_students_not_liking_any_sport_l479_479140


namespace tina_savings_l479_479585

theorem tina_savings :
  let june_savings : ℕ := 27
  let july_savings : ℕ := 14
  let august_savings : ℕ := 21
  let books_spending : ℕ := 5
  let shoes_spending : ℕ := 17
  let total_savings := june_savings + july_savings + august_savings
  let total_spending := books_spending + shoes_spending
  let remaining_money := total_savings - total_spending
  remaining_money = 40 :=
by
  sorry

end tina_savings_l479_479585


namespace periodic_function_l479_479489

open Real

theorem periodic_function (f : ℝ → ℝ) 
  (h_bound : ∀ x : ℝ, |f x| ≤ 1)
  (h_func_eq : ∀ x : ℝ, f (x + 13 / 42) + f x = f (x + 1 / 6) + f (x + 1 / 7)) : 
  ∀ x : ℝ, f (x + 1) = f x := 
  sorry

end periodic_function_l479_479489


namespace calculation_not_minus_one_l479_479274

theorem calculation_not_minus_one :
  (-1 : ℤ) * 1 ≠ 1 ∧
  (-1 : ℤ) / (-1) = 1 ∧
  (-2015 : ℤ) / 2015 ≠ 1 ∧
  (-1 : ℤ)^9 * (-1 : ℤ)^2 ≠ 1 := by 
  sorry

end calculation_not_minus_one_l479_479274


namespace probability_at_least_one_multiple_of_4_l479_479330

-- Define the condition
def random_integer_between_1_and_60 : set ℤ := {n : ℤ | 1 ≤ n ∧ n ≤ 60}

-- Define the probability theorems and the proof for probability calculation
theorem probability_at_least_one_multiple_of_4 :
  (∀ (n1 n2 : ℤ), (n1 ∈ random_integer_between_1_and_60) ∧ (n2 ∈ random_integer_between_1_and_60) → 
  (∃ k, n1 = 4 * k ∨ ∃ k, n2 = 4 * k)) ∧ 
  (1 / 60 * 1 / 60) * (15 / 60) ^ 2 = 7 / 16 :=
by
  sorry

end probability_at_least_one_multiple_of_4_l479_479330


namespace selecting_female_probability_l479_479455

theorem selecting_female_probability (female male : ℕ) (total : ℕ)
  (h_female : female = 4)
  (h_male : male = 6)
  (h_total : total = female + male) :
  (female / total : ℚ) = 2 / 5 := 
by
  -- Insert proof steps here
  sorry

end selecting_female_probability_l479_479455


namespace prove_a_star_b_l479_479553

variable (a b : ℤ)
variable (h1 : a + b = 12)
variable (h2 : a * b = 35)

def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

theorem prove_a_star_b : star a b = 12 / 35 :=
by
  sorry

end prove_a_star_b_l479_479553


namespace sum_G_2_to_1003_l479_479033

-- Definition of G(n)
def G (n : ℕ) : ℕ := if n % 2 = 0 then 2 * n else 2 * n - 1

-- Theorem statement
theorem sum_G_2_to_1003 : (∑ n in Finset.range (1004 - 2) + 2, G n) = 1004004 :=
by
  sorry

end sum_G_2_to_1003_l479_479033


namespace count_valid_four_digit_numbers_l479_479428

def is_valid_four_digit (n : ℕ) : Prop :=
  let u := n % 10 in
  let t := (n / 10) % 10 in
  let h := (n / 1000) % 10 in
  1000 ≤ n ∧ n < 10000 ∧ -- n is a four-digit number
  u ≥ 2 * t ∧ -- units digit is at least twice the tens digit
  h % 2 = 1 -- thousands digit is odd

theorem count_valid_four_digit_numbers : 
  (Finset.filter is_valid_four_digit (Finset.Icc 1000 9999)).card = 1500 :=
  by sorry

end count_valid_four_digit_numbers_l479_479428


namespace visitors_yesterday_l479_479319

-- Definitions based on the given conditions
def visitors_today : ℕ := 583
def visitors_total : ℕ := 829

-- Theorem statement to prove the number of visitors the day before Rachel visited
theorem visitors_yesterday : ∃ v_yesterday: ℕ, v_yesterday = visitors_total - visitors_today ∧ v_yesterday = 246 :=
by
  sorry

end visitors_yesterday_l479_479319


namespace intersection_is_isosceles_right_triangle_l479_479897

def U := {x | ∃ (t : Type), x = t ∧ t = triangle}
def M := {x | x = right_triangle}
def N := {x | x = isosceles_triangle}

theorem intersection_is_isosceles_right_triangle : 
  M ∩ N = {x | x = isosceles_right_triangle} := 
by
  sorry

end intersection_is_isosceles_right_triangle_l479_479897


namespace general_term_a_sum_first_n_terms_a_sum_first_20_terms_b_l479_479420

-- Definition of the sequence {a_n}
def a_sequence : ℕ → ℕ
| 0        := 1
| (n+1)    := 3 * a_sequence n

-- General term formula for {a_n}
theorem general_term_a (n : ℕ) : a_sequence n = 3^n :=
by sorry

-- Sum of the first n terms for {a_n}
def S_n (n : ℕ) : ℕ :=
(3^n - 1) / 2

-- Sum formula for {S_n}
theorem sum_first_n_terms_a (n : ℕ) : (∑ i in range (n + 1), a_sequence i) = S_n n :=
by sorry

-- Conditions for sequence {b_n}
def b_1 : ℕ := a_sequence 1
def b_3 : ℕ := a_sequence 0 + a_sequence 1 + a_sequence 2

-- Define the arithmetic sequence {b_n}
def b_sequence (n : ℕ) : ℕ :=
3 + 5 * n - 5

-- Sum of the first n terms for {b_n}
def T_n (n : ℕ) : ℕ :=
n * (b_sequence 1 + b_sequence n) / 2

-- Sum formula for T_20
theorem sum_first_20_terms_b : T_n 20 = 1010 :=
by sorry

end general_term_a_sum_first_n_terms_a_sum_first_20_terms_b_l479_479420


namespace angle_DMN_and_circumcenter_l479_479317

variables {A B C D E F G H I J M N P: Type}

/-- Statement of the problem: 
Given,
  1. ABC is a triangle.
  2. D is the midpoint of the arc BC not containing A.
  3. E is the midpoint of the arc CA not containing B.
  4. F is the midpoint of the arc AB not containing C.
  5. Line DE meets BC at G and AC at H.
  6. M is the midpoint of GH.
  7. Line DF meets BC at I and AB at J.
  8. N is the midpoint of IJ.
  9. AD meets EF at P.
Prove,
  1. ∠DMN = (∠A / 2) + (∠B / 2)
  2. The circumcenter of triangle DMN lies on the circumcircle of triangle PMN.
-/
theorem angle_DMN_and_circumcenter (ABC : Type) (triangle : Type)
  [ht : ABC → triangle]
  (D E F G H I J M N P: Type)
  (ABC_triangle : triangle)
  (D_midpoint : D = midpoint_of_arc_hidden BC A)
  (E_midpoint : E = midpoint_of_arc_hidden CA B)
  (F_midpoint : F = midpoint_of_arc_hidden AB C)
  (GH_line : G = intersection_of_lines DE BC ∧ H = intersection_of_lines DE AC)
  (M_midpoint : M = midpoint G H)
  (IJ_line : I = intersection_of_lines DF BC ∧ J = intersection_of_lines DF AB)
  (N_midpoint : N = midpoint I J)
  (P_intersection : P = intersection_point AD EF) :
  ∠DMN = (∠A / 2) + (∠B / 2) ∧ circumcenter_DMN_on_circumcircle PMN :=
sorry

end angle_DMN_and_circumcenter_l479_479317


namespace central_angle_of_sector_l479_479400

theorem central_angle_of_sector (alpha : ℝ) (l : ℝ) (A : ℝ) (h1 : l = 2 * Real.pi) (h2 : A = 5 * Real.pi) : 
  alpha = 72 :=
by
  sorry

end central_angle_of_sector_l479_479400


namespace brooke_milk_price_per_gallon_l479_479673

theorem brooke_milk_price_per_gallon :
  -- Conditions
  (total_cows = 12) →
  (milk_per_cow = 4) →
  (total_customers = 6) →
  (milk_per_customer = 6) →
  (butter_per_gallon = 2) →
  (price_per_butter = 1.5) →
  (total_earnings = 144) →
  -- Conclusion: price per gallon of milk
  (price_per_gallon : ℝ) →
  total_cows * milk_per_cow = 48 ∧
  total_customers * milk_per_customer = 36 ∧
  48 - 36 = 12 ∧
  12 * butter_per_gallon = 24 ∧
  24 * price_per_butter = 36 ∧
  total_earnings - 36 = 108 ∧
  108 / 36 = price_per_gallon →
  price_per_gallon = 3 :=
begin
  -- Proof omitted
  sorry
end

end brooke_milk_price_per_gallon_l479_479673


namespace largest_integer_same_cost_l479_479260

def sum_decimal_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sum_ternary_digits (n : ℕ) : ℕ :=
  n.digits 3 |>.sum

theorem largest_integer_same_cost :
  ∃ n : ℕ, n < 1000 ∧ sum_decimal_digits n = sum_ternary_digits n ∧ ∀ m : ℕ, m < 1000 ∧ sum_decimal_digits m = sum_ternary_digits m → m ≤ n := 
  sorry

end largest_integer_same_cost_l479_479260


namespace eq1_no_solution_eq2_solutions_l479_479209

-- Define the first problem: No solution for the first equation
theorem eq1_no_solution (x : ℝ) :
  (x + 1) / (x - 1) - 4 / (x ^ 2 - 1) = 1 → false :=
by intro h; sorry

-- Define the second problem: Find the solutions for the quadratic equation
theorem eq2_solutions :
  ∀ (x : ℝ), (x ^ 2 + 3 * x - 2 = 0) ↔ (x = -3/2 - sqrt 17 / 2 ∨ x = -3/2 + sqrt 17 / 2) :=
by intro x; sorry

end eq1_no_solution_eq2_solutions_l479_479209


namespace complex_arithmetic_l479_479522

theorem complex_arithmetic :
  ((6 - 3 * complex.I) - (2 + 4 * complex.I)) * (2 * complex.I) = 14 + 8 * complex.I := by
  sorry

end complex_arithmetic_l479_479522


namespace divisor_equation_l479_479488

def number_of_divisors (n : ℕ) : ℕ :=
  Nat.divisors n |>.length 

theorem divisor_equation (n p q : ℕ) (h1 : 2 ≤ n)
  (h2 : ∀ (n : ℕ), d (n) = number_of_divisors n)
  (h3 : ∀ (n : ℕ), d (n^3) = number_of_divisors (n^3))
  (h4 : p.prime) (h5 : q.prime) (h6 : (p ≠ q))
  (h7 : number_of_divisors (n^3) = 5 * number_of_divisors n) :
  n = p^3 * q :=
sorry

end divisor_equation_l479_479488


namespace temperature_at_night_l479_479448

theorem temperature_at_night 
  (T_morning : ℝ) 
  (T_rise_noon : ℝ) 
  (T_drop_night : ℝ) 
  (h1 : T_morning = 22) 
  (h2 : T_rise_noon = 6) 
  (h3 : T_drop_night = 10) : 
  (T_morning + T_rise_noon - T_drop_night = 18) :=
by 
  sorry

end temperature_at_night_l479_479448


namespace f_monotonic_intervals_and_range_l479_479774

-- Definition of the function f(x)
def f (x : ℝ) := (4 * x^2 - 12 * x - 3) / (2 * x + 1)

-- Definition of the function g(x, a)
def g (x a : ℝ) := -x - 2 * a

-- Conditions that we need to prove for the monotonicity and range of f(x) alongside finding a suitable value of a
theorem f_monotonic_intervals_and_range :
  (∀ x ∈ set.Icc 0 (1 / 2 : ℝ), deriv f x < 0) ∧
  (∀ x ∈ set.Icc (1 / 2 : ℝ) 1, deriv f x > 0) ∧
  (set.range f = set.Icc (-4 : ℝ) (-3 : ℝ)) ∧
  (∀ (x₁ x₂ : ℝ), x₁ ∈ set.Icc 0 1 → x₂ ∈ set.Icc 0 1 → g x₂ (3 / 2) = f x₁) :=
by sorry

end f_monotonic_intervals_and_range_l479_479774


namespace rectangle_perimeter_l479_479992

open Real

noncomputable def perimeter_of_rectangle {a b c w : ℝ} (h₀ : a = 9) (h₁ : b = 12) (h₂ : c = 15)
  (h₃ : w = 6) (h₄ : a^2 + b^2 = c^2) : ℝ :=
  let area_triangle := (1 / 2) * a * b
  let l := area_triangle / w
  2 * (w + l)

-- Statement: The perimeter of the rectangle is 30 units.
theorem rectangle_perimeter : perimeter_of_rectangle (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num : 9^2 + 12^2 = 15^2) = 30 :=
sorry

end rectangle_perimeter_l479_479992


namespace num_values_n_l479_479720

-- Define the sum of digits function
def S (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem num_values_n:
  ∃ (n₁ n₂ : ℕ), 
  n₁ + S(n₁) + S(S(n₁)) = 2050 ∧ 
  n₂ + S(n₂) + S(S(n₂)) = 2050 ∧ 
  2012 ≤ n₁ ∧ n₁ ≤ 2050 ∧ 
  2012 ≤ n₂ ∧ n₂ ≤ 2050 ∧ 
  n₁ ≠ n₂ :=
sorry

end num_values_n_l479_479720


namespace probability_of_picking_letter_from_MATHEMATICS_l479_479113

theorem probability_of_picking_letter_from_MATHEMATICS : 
  (8 : ℤ) / 26 = (4 : ℤ) / 13 :=
by
  norm_num

end probability_of_picking_letter_from_MATHEMATICS_l479_479113


namespace part1_part2_l479_479771

noncomputable def f (x: ℝ) (a: ℝ) (b: ℝ) : ℝ := a * Real.log x - b * x

theorem part1 (a : ℝ) : 
  (∀ x > 0, x ∈ ℝ → f x a 1 = a * Real.log x - x) →
  (a ≤ 0 → ∀ x > 0, DifferentiableAt ℝ (f x a 1) x → Deriv (f x a 1) x < 0) ∧ 
  (a > 0 → (∀ x ∈ Ioo 0 a, DifferentiableAt ℝ (f x a 1) x → Deriv (f x a 1) x > 0) ∧ 
  (∀ x > a, DifferentiableAt ℝ (f x a 1) x → Deriv (f x a 1) x < 0)) :=
sorry

noncomputable def g (x: ℝ) (m: ℝ) : ℝ := x * Real.exp x - (m + 1) * x - 1

theorem part2 (m: ℝ) : 
  (∀ x > 0, 1 * Real.log x - b * x ≤ -1 ∧ 1 * Real.log x - b * x ≤ x * Real.exp x - (m + 1) * x - 1) →
  m ≤ 1 :=
sorry

end part1_part2_l479_479771


namespace find_chord_line_l479_479748

noncomputable def hyperbola_line_equation (P : ℝ × ℝ) (h : P = (8, 1)) : Prop :=
  P = (8, 1) → (∀ A B : ℝ × ℝ, (A.1 + B.1 = 16) ∧ (A.2 + B.2 = 2) ∧ (A.1^2 - 4 * A.2^2 = 4) ∧ (B.1^2 - 4 * B.2^2 = 4) →
    (∃ k : ℝ, k = 2 ∧ ∀ x y : ℝ, y = k * x + (1 - 8 * k) = 2 * x - y - 15))

theorem find_chord_line :
  hyperbola_line_equation (8, 1) (rfl) := 
  sorry

end find_chord_line_l479_479748


namespace convert_binary_to_decimal_l479_479341

theorem convert_binary_to_decimal : (1 * 2^2 + 1 * 2^1 + 1 * 2^0) = 7 := by
  sorry

end convert_binary_to_decimal_l479_479341


namespace kneading_time_is_correct_l479_479185

def total_time := 280
def rising_time_per_session := 120
def number_of_rising_sessions := 2
def baking_time := 30

def total_rising_time := rising_time_per_session * number_of_rising_sessions
def total_non_kneading_time := total_rising_time + baking_time
def kneading_time := total_time - total_non_kneading_time

theorem kneading_time_is_correct : kneading_time = 10 := by
  have h1 : total_rising_time = 240 := by
    sorry
  have h2 : total_non_kneading_time = 270 := by
    sorry
  have h3 : kneading_time = 10 := by
    sorry
  exact h3

end kneading_time_is_correct_l479_479185


namespace union_complement_l479_479090

open Set

def U : Set ℤ := {x | -3 < x ∧ x < 3}

def A : Set ℤ := {1, 2}

def B : Set ℤ := {-2, -1, 2}

theorem union_complement :
  A ∪ (U \ B) = {0, 1, 2} := by
  sorry

end union_complement_l479_479090


namespace ramu_profit_percent_correct_l479_479625

def ramu_initial_cost : ℝ := 42000
def ramu_repair_cost : ℝ := 13000
def ramu_selling_price : ℝ := 60900
def ramu_total_cost : ℝ := ramu_initial_cost + ramu_repair_cost
def ramu_profit : ℝ := ramu_selling_price - ramu_total_cost
def ramu_profit_percent : ℝ := (ramu_profit / ramu_total_cost) * 100

theorem ramu_profit_percent_correct : ramu_profit_percent ≈ 10.73 := 
by {
  sorry
}

end ramu_profit_percent_correct_l479_479625


namespace find_quadruples_l479_479019

def valid_quadruple (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + x2 * x3 * x4 = 2 ∧ 
  x2 + x3 * x4 * x1 = 2 ∧ 
  x3 + x4 * x1 * x2 = 2 ∧ 
  x4 + x1 * x2 * x3 = 2

theorem find_quadruples (x1 x2 x3 x4 : ℝ) :
  valid_quadruple x1 x2 x3 x4 ↔ (x1, x2, x3, x4) = (1, 1, 1, 1) ∨ 
                                   (x1, x2, x3, x4) = (3, -1, -1, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, 3, -1, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, -1, 3, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, -1, -1, 3) := by
  sorry

end find_quadruples_l479_479019


namespace grid_is_valid_l479_479831

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Check if all adjacent cells (by side or diagonal) in a 3x3 grid are coprime -/
def valid_grid (grid : Array (Array ℕ)) : Prop :=
  let adjacent_pairs := 
    [(0, 0, 0, 1), (0, 1, 0, 2), 
     (1, 0, 1, 1), (1, 1, 1, 2), 
     (2, 0, 2, 1), (2, 1, 2, 2),
     (0, 0, 1, 0), (0, 1, 1, 1), (0, 2, 1, 2),
     (1, 0, 2, 0), (1, 1, 2, 1), (1, 2, 2, 2),
     (0, 0, 1, 1), (0, 1, 1, 2), 
     (1, 0, 2, 1), (1, 1, 2, 2), 
     (0, 2, 1, 1), (1, 2, 2, 1), 
     (0, 1, 1, 0), (1, 1, 2, 0)] 
  adjacent_pairs.all (λ ⟨r1, c1, r2, c2⟩, is_coprime (grid[r1][c1]) (grid[r2][c2]))

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10],
    #[5, 7, 11],
    #[6, 13, 12]]

theorem grid_is_valid : valid_grid grid :=
  by
    -- Proof omitted, placeholder
    sorry

end grid_is_valid_l479_479831


namespace tan_alpha_third_quadrant_l479_479395

theorem tan_alpha_third_quadrant 
    (α : ℝ) 
    (h1 : π < α ∧ α < 3 * π / 2) 
    (h2 : 3 * Real.cos (2 * α) + Real.sin α = 2) : 
    Real.tan α = Real.sqrt 2 / 4 := 
begin
    sorry
end

end tan_alpha_third_quadrant_l479_479395


namespace coprime_3x3_grid_l479_479826

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def valid_3x3_grid (grid : Array (Array ℕ)) : Prop :=
  -- Check row-wise, column-wise, and diagonal adjacencies for coprimality
  ∀ i j, i < 3 → j < 3 →
  (if j < 2 then is_coprime (grid[i][j]) (grid[i][j+1]) else True) ∧
  (if i < 2 then is_coprime (grid[i][j]) (grid[i+1][j]) else True) ∧
  (if i < 2 ∧ j < 2 then is_coprime (grid[i][j]) (grid[i+1][j+1]) else True) ∧
  (if i < 2 ∧ j > 0 then is_coprime (grid[i][j]) (grid[i+1][j-1]) else True)

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10], #[5, 7, 11], #[6, 13, 12]]

theorem coprime_3x3_grid : valid_3x3_grid grid := sorry

end coprime_3x3_grid_l479_479826


namespace sequence_exists_l479_479889

variables (G : Type*) [group G] [fintype G] (a b : G)

theorem sequence_exists (n : ℕ) (hG : fintype.card G = n)
  (hGen : ∀ g ∈ G, ∃ k l : ℕ, g = a ^ k * b ^ l) :
  ∃ (g : fin (2 * n) → G),
    (∀ x : G, ∃! i : fin (2 * n), g i = x) ∧
    (∀ i : fin (2 * n), g ⟨i + 1, by simp only [nat.add_one, add_lt_add_iff_right, fin.is_lt]⟩ = g i * a ∨ g i * b) :=
sorry

end sequence_exists_l479_479889


namespace find_b_l479_479622

variable (a b : ℕ)
variable (total_profit managing_fee remaining_profit a_profit : ℕ)

-- Conditions based on step a)
def conditions (a b total_profit managing_fee remaining_profit a_profit : ℕ) : Prop :=
  a = 3500 ∧
  total_profit = 9600 ∧
  managing_fee = 0.10 * total_profit ∧
  remaining_profit = total_profit - managing_fee ∧
  a_profit = 6000 - managing_fee ∧
  a * remaining_profit / (a + b) = a_profit

-- Given the above conditions, find the value of b
theorem find_b (a b total_profit managing_fee remaining_profit a_profit : ℕ) :
  conditions 3500 b 9600 960 8640 5040 →
  b = 2500 :=
by
  intro h
  sorry

end find_b_l479_479622


namespace tina_money_left_l479_479586

theorem tina_money_left :
  let june_savings := 27
  let july_savings := 14
  let august_savings := 21
  let books_spending := 5
  let shoes_spending := 17
  june_savings + july_savings + august_savings - (books_spending + shoes_spending) = 40 :=
by
  sorry

end tina_money_left_l479_479586


namespace wholesale_cost_l479_479651

theorem wholesale_cost (W R : ℝ) (h1 : R = 1.20 * W) (h2 : 0.70 * R = 168) : W = 200 :=
by
  sorry

end wholesale_cost_l479_479651


namespace ratio_correct_l479_479669

-- Definitions based on the problem conditions
def initial_cards_before_eating (X : ℤ) : ℤ := X
def cards_bought_new : ℤ := 4
def cards_left_after_eating : ℤ := 34

-- Definition of the number of cards eaten by the dog
def cards_eaten_by_dog (X : ℤ) : ℤ := X + cards_bought_new - cards_left_after_eating

-- Definition of the ratio of the number of cards eaten to the total number of cards before being eaten
def ratio_cards_eaten_to_total (X : ℤ) : ℚ := (cards_eaten_by_dog X : ℚ) / (X + cards_bought_new : ℚ)

-- Statement to prove
theorem ratio_correct (X : ℤ) : ratio_cards_eaten_to_total X = (X - 30) / (X + 4) := by
  sorry

end ratio_correct_l479_479669


namespace identify_true_propositions_l479_479082

-- Definitions based on the conditions
def prop1 (l1 l2 : Line) (p1 p2 : Plane) : Prop :=
(∀ {p : Plane}, IsParallel l1 p → IsParallel l2 p → IsParallel p1 p2)

def prop2 (l : Line) (p1 p2 : Plane) : Prop :=
PassesThrough l p1 → IsPerpendicular l p2 → IsPerpendicular p1 p2

def prop3 (l1 l2 l3 : Line) (p : Plane) : Prop :=
(InPlane l1 p ∧ InPlane l2 p) → (IsPerpendicular l1 l3 ∧ IsPerpendicular l2 l3) → IsParallel l1 l2

def prop4 (l : Line) (p1 p2 : Plane) : Prop :=
IsPerpendicular p1 p2 → (InPlane l p1 ∧ ¬ IsPerpendicular l (LineOfIntersection p1 p2)) → ¬ IsPerpendicular l p2

-- Lean statement of the problem with the correct propositions
theorem identify_true_propositions {l l1 l2 l3 : Line} {p1 p2 p3 : Plane} :
  (prop2 l p1 p2 ∧ prop4 l p1 p2) ∧ ¬ (prop1 l1 l2 p1 p2 ∨ prop3 l1 l2 l3 p3) :=
by { sorry }

end identify_true_propositions_l479_479082


namespace range_of_a_l479_479419

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, x^2 - a * x + 1 = 0) ↔ -2 < a ∧ a < 2 :=
by sorry

end range_of_a_l479_479419


namespace solve_expr_l479_479014
noncomputable section

open Real

-- Define the expression to simplify.
def expr : ℝ :=
  (log 2) ^ 2 + (log 5) * (log 20) + (2014 ^ (1 / 2) - 2) ^ 0 + 0.064 ^ (-2 / 3) * (1 / 4) ^ (-2)

-- State the theorem to be proved.
theorem solve_expr : expr = 102 := by
  sorry

end solve_expr_l479_479014


namespace smallest_positive_multiple_of_45_l479_479608

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ 45 * x = 45 :=
by {
  use 1,
  rw mul_one,
  exact nat.one_pos,
  sorry
}

end smallest_positive_multiple_of_45_l479_479608


namespace determine_x_l479_479994

def is_ohara_triple (a b x : ℤ) : Prop :=
  (Real.sqrt (Int.natAbs a) + Real.sqrt (Int.natAbs b) = x)

theorem determine_x : is_ohara_triple (-49) 64 15 :=
by {
  unfold is_ohara_triple,
  sorry -- proof goes here
}

end determine_x_l479_479994


namespace minimum_value_of_function_on_interval_l479_479061

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x ^ 3 - 6 * x ^ 2 + m

theorem minimum_value_of_function_on_interval :
  (∃ m : ℝ, (∀ x ∈ ([-2, 2] : set ℝ), f x m ≤ 2 * (real.sqrt x) - 6 * x^2 + m) ∧ (f 0 m = 3)) →
  (∀ x ∈ ([-2, 2] : set ℝ), f x 3 ≥ -37) :=
sorry

end minimum_value_of_function_on_interval_l479_479061


namespace count_odd_perfect_squares_less_than_16000_l479_479798

theorem count_odd_perfect_squares_less_than_16000 : 
  ∃ n : ℕ, n = 31 ∧ ∀ k < 16000, 
    ∃ b : ℕ, b = 2 * n + 1 ∧ k = (4 * n + 3) ^ 2 ∧ (∃ m : ℕ, m = b + 1 ∧ m % 2 = 0) := 
sorry

end count_odd_perfect_squares_less_than_16000_l479_479798


namespace quadratic_minimum_l479_479688

-- Define the constants p and q as positive real numbers
variables (p q : ℝ) (hp : 0 < p) (hq : 0 < q)

-- Define the quadratic function f
def f (x : ℝ) : ℝ := 3 * x^2 + p * x + q

-- Assertion to prove: the function f reaches its minimum at x = -p / 6
theorem quadratic_minimum : 
  ∃ x : ℝ, x = -p / 6 ∧ (∀ y : ℝ, f y ≥ f x) :=
sorry

end quadratic_minimum_l479_479688


namespace congruence_proof_l479_479593

-- Define the given functions
def f1 (x : ℝ) : ℝ := 2 * real.log (x + 2) / real.log 2
def f2 (x : ℝ) : ℝ := real.log (x + 2) / real.log 2
def f3 (x : ℝ) : ℝ := (real.log (x + 2) / real.log 2)^2
def f4 (x : ℝ) : ℝ := real.log (2 * x) / real.log 2

-- Define congruence condition
def congruent (f g : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f (x + a) + b = g x

-- Prove that f2 and f4 are congruent
theorem congruence_proof : congruent f2 f4 :=
  sorry

end congruence_proof_l479_479593


namespace find_h_squared_l479_479681

noncomputable def complex_magnitudes (a b c : ℂ) (p q : ℂ) : Prop :=
  -- The polynomial condition
  (∀ z, z^3 + p*z + q = 0 → z = a ∨ z = b ∨ z = c) ∧
  -- The magnitude sum condition
  (|a|^2 + |b|^2 + |c|^2 = 360) ∧
  -- The centroid and right angle condition
  (a + b + c = 0) ∧
  -- Right triangle condition
  (let x := |b - c| in let y := |a - c| in let z := |a - b| in x^2 = y^2 + z^2)

theorem find_h_squared {a b c p q : ℂ} (h : ℝ) :
  complex_magnitudes a b c p q →
  h^2 = |b - c|^2 :=
begin
  intro h_conditions,
  cases h_conditions with _ h_right_triangle,
  simp only [*, pow_two],
  have hyp := h_right_triangle.right.right.right,
  rw hyp,
  sorry
end

end find_h_squared_l479_479681


namespace inclination_angle_of_line_l479_479786

theorem inclination_angle_of_line
  (α : ℝ) (h1 : α > 0) (h2 : α < 180)
  (hslope : Real.tan α = - (Real.sqrt 3) / 3) :
  α = 150 :=
sorry

end inclination_angle_of_line_l479_479786


namespace tina_savings_l479_479584

theorem tina_savings :
  let june_savings : ℕ := 27
  let july_savings : ℕ := 14
  let august_savings : ℕ := 21
  let books_spending : ℕ := 5
  let shoes_spending : ℕ := 17
  let total_savings := june_savings + july_savings + august_savings
  let total_spending := books_spending + shoes_spending
  let remaining_money := total_savings - total_spending
  remaining_money = 40 :=
by
  sorry

end tina_savings_l479_479584


namespace probability_of_hitting_target_at_least_twice_l479_479308

theorem probability_of_hitting_target_at_least_twice :
  let p : ℝ := 0.6 in
  let q : ℝ := 0.4 in
  let n : ℕ := 3 in
  ∑ k in {2, 3}, nat.choose n k * p^k * q^(n-k) = 81 / 125 :=
by
  sorry

end probability_of_hitting_target_at_least_twice_l479_479308


namespace rohan_salary_correct_l479_479206

def rohan_savings (S : ℕ) : Prop :=
  let food_expenses := 0.30 * S
  let house_rent := 0.20 * S
  let entertainment_expenses := 0.5 * food_expenses
  let conveyance_expenses := entertainment_expenses + 0.25 * entertainment_expenses
  let education_expenses := 0.05 * S
  let utilities_expenses := 0.10 * S
  let total_expenses := food_expenses + house_rent + entertainment_expenses + conveyance_expenses + education_expenses + utilities_expenses
  let savings := S - total_expenses
  savings = 2500

noncomputable def rohan_monthly_salary : ℕ :=
  2500 / 0.0125 -- Calculating the monthly salary based on savings

theorem rohan_salary_correct : rohan_savings 200000 := 
by 
  unfold rohan_savings 
  -- skipping the actual arithmetic proof
  sorry

end rohan_salary_correct_l479_479206


namespace calculate_area_l479_479519

noncomputable def area_inside_rect_outside_circles (EF EG: ℝ) (rE rF rG: ℝ) : ℝ :=
  let area_rectangle := EF * EG
  let area_quarter_circles := (rE^2 * Real.pi) / 4 + (rF^2 * Real.pi) / 4 + (rG^2 * Real.pi) / 4
  area_rectangle - area_quarter_circles

theorem calculate_area :
  area_inside_rect_outside_circles 4 6 2 3 1.5 ≈ 12.0 := by
  sorry

end calculate_area_l479_479519


namespace subset_implies_a_lt_neg_one_nonempty_intersection_implies_a_lt_three_l479_479042

open Set

variable (a : ℝ)
def U : Set ℝ := univ
def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | x > a }

-- Problem 1: Prove that a < -1 given A ⊆ B
theorem subset_implies_a_lt_neg_one (h : A ⊆ B) : a < -1 :=
by sorry

-- Problem 2: Prove that a < 3 given A ∩ B ≠ ∅
theorem nonempty_intersection_implies_a_lt_three (h : A ∩ B ≠ ∅) : a < 3 :=
by sorry

end subset_implies_a_lt_neg_one_nonempty_intersection_implies_a_lt_three_l479_479042


namespace book_collection_increase_l479_479822

theorem book_collection_increase 
  (N : ℕ) 
  (h1 : N * 1.004 * 1.008 < 50000) 
  : (N * (1.004 * 1.008 - 1.004)).floor = 251 :=
by contradiction

end book_collection_increase_l479_479822


namespace question1_question2_l479_479779

-- Definitions based on the conditions
def f (x m : ℝ) : ℝ := x^2 + 4*x + m

theorem question1 (m : ℝ) (h1 : m ≠ 0) (h2 : 16 - 4 * m > 0) : m < 4 :=
  sorry

theorem question2 (m : ℝ) (hx : ∀ x : ℝ, f x m = 0 → f (-x - 4) m = 0) 
  (h_circ : ∀ (x y : ℝ), x^2 + y^2 + 4*x - (m + 1) * y + m = 0 → (x = 0 ∧ y = 1) ∨ (x = -4 ∧ y = 1)) :
  (∀ (x y : ℝ), x^2 + y^2 + 4*x - (m + 1) * y + m = 0 → (x = 0 ∧ y = 1)) ∨ (∀ (x y : ℝ), (x = -4 ∧ y = 1)) :=
  sorry

end question1_question2_l479_479779


namespace function_satisfies_conditions_l479_479704

/-- Define the set S, for generality assume S is reals, can be adjusted as needed
    to fit the specific definition of set S in your context. -/
def S := ℝ

noncomputable def f (x : S) : S := -x / (1 + x)

/-- Prove that the function f satisfies the given functional equation and increasing condition. -/
theorem function_satisfies_conditions :
  (∀ (x y : S), f (x + f y + x * f y) = y + f x + y * f x) ∧
  (∀ a b : S, (-1 < a ∧ a < 0 ∧ -1 < b ∧ b < 0 ∨ 0 < a ∧ 0 < b ) → a < b → f a / a < f b / b) :=
by
  sorry

end function_satisfies_conditions_l479_479704


namespace find_n_l479_479307

theorem find_n (x : ℝ) (h1 : x = 596.95) (h2 : ∃ n : ℝ, n + 11.95 - x = 3054) : ∃ n : ℝ, n = 3639 :=
by
  sorry

end find_n_l479_479307


namespace motorboat_speed_relative_to_water_l479_479646

noncomputable def speed_of_current : ℝ := 2.28571428571
noncomputable def time_upstream : ℝ := 20 / 60
noncomputable def time_downstream : ℝ := 15 / 60

theorem motorboat_speed_relative_to_water (v : ℝ) :
  let c := speed_of_current in
  let d_up := (v - c) * time_upstream in
  let d_down := (v + c) * time_downstream in
  d_up = d_down → v = 16 :=
by
  sorry

end motorboat_speed_relative_to_water_l479_479646


namespace probability_of_at_least_one_multiple_of_4_l479_479327

open ProbabilityTheory

def prob_at_least_one_multiple_of_4 : ℚ := 7 / 16

theorem probability_of_at_least_one_multiple_of_4 :
  let S := Finset.range 60
  let multiples_of_4 := S.filter (λ x, (x + 1) % 4 = 0)
  let prob (a b : ℕ) := (a : ℚ) / b
  let prob_neither_multiple_4 := (prob (60 - multiples_of_4.card) 60) ^ 2
  1 - prob_neither_multiple_4 = prob_at_least_one_multiple_of_4 := by
  sorry

end probability_of_at_least_one_multiple_of_4_l479_479327


namespace find_other_number_l479_479216

theorem find_other_number (A B : ℕ) (H1 : Nat.lcm A B = 2310) (H2 : Nat.gcd A B = 30) (H3 : A = 770) : B = 90 :=
  by
  sorry

end find_other_number_l479_479216


namespace collatz_conjecture_m_values_l479_479214

def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2
  else 3 * n + 1

def iter_collatz (n : ℕ) : ℕ → ℕ
| 0 => n
| (k+1) => collatz (iter_collatz n k)

theorem collatz_conjecture_m_values :
  ∃ m : ℕ, iter_collatz m 6 = 1 ∧ (m = 64 ∨ m = 10 ∨ m = 1 ∨ m = 8) :=
begin
  sorry
end

end collatz_conjecture_m_values_l479_479214


namespace inequality_equivalence_l479_479749

theorem inequality_equivalence (a : ℝ) :
  (∀ (x : ℝ), |x + 1| + |x - 1| ≥ a) ↔ (a ≤ 2) :=
sorry

end inequality_equivalence_l479_479749


namespace q_poly_correct_l479_479942

open Polynomial

noncomputable def q : Polynomial ℚ := 
  -(C 1) * X^6 + C 4 * X^4 + C 21 * X^3 + C 15 * X^2 + C 14 * X + C 3

theorem q_poly_correct : 
  ∀ x : Polynomial ℚ,
  q + (X^6 + 4 * X^4 + 5 * X^3 + 12 * X) = 
  (8 * X^4 + 26 * X^3 + 15 * X^2 + 26 * X + C 3) := by sorry

end q_poly_correct_l479_479942


namespace value_of_n_l479_479401

theorem value_of_n (a : ℝ) (n : ℕ) (h : ∃ (k : ℕ), (n - 2 * k = 0) ∧ (k = 4)) : n = 8 :=
sorry

end value_of_n_l479_479401


namespace distance_traveled_l479_479130

theorem distance_traveled 
    (walking_time_min : ℕ) (walking_rate_mph : ℕ)
    (running_time_min : ℕ) (running_rate_mph : ℕ)
    (total_time_min : ℕ)
    (h_walking_time : walking_time_min = 30)
    (h_walking_rate : walking_rate_mph = 3)
    (h_running_time : running_time_min = 45)
    (h_running_rate : running_rate_mph = 8)
    (h_total_time : total_time_min = 75) :
    (walking_rate_mph * walking_time_min / 60) + (running_rate_mph * running_time_min / 60) = 7.5 :=
by
  sorry

end distance_traveled_l479_479130


namespace volume_of_tetrahedron_after_folding_l479_479249

theorem volume_of_tetrahedron_after_folding 
  (AB BC CA : ℝ)
  (h1 : AB = 11)
  (h2 : BC = 20)
  (h3 : CA = 21) :
  volume_of_folding_tetrahedron AB BC CA = 45 :=
sorry

end volume_of_tetrahedron_after_folding_l479_479249


namespace gcd_1729_78945_is_1_l479_479272

theorem gcd_1729_78945_is_1 :
  ∃ m n : ℤ, 1729 * m + 78945 * n = 1 := sorry

end gcd_1729_78945_is_1_l479_479272


namespace polygon_sides_l479_479893

variables {b : ℝ} (hb : b > 0)

def set_T (b : ℝ) := {p : ℝ × ℝ | 
  b ≤ p.1 ∧ p.1 ≤ 3b ∧ 
  b ≤ p.2 ∧ p.2 ≤ 3b ∧ 
  p.1 + p.2 ≥ 2b ∧ 
  p.1 + 2b ≥ p.2 ∧ 
  p.2 + b ≥ p.1 ∧ 
  p.1 + p.2 ≤ 4b}

theorem polygon_sides (b : ℝ) (hb : b > 0): 
  ∃ vertices,  set.count vertices = 4 
               ∧ ∀ p1 p2 ∈ vertices, 
                 p1 ≠ p2 → (p1, p2) ∈ set_T b := 
sorry

end polygon_sides_l479_479893


namespace Micheal_work_rate_l479_479499

theorem Micheal_work_rate 
    (M A : ℕ) 
    (h1 : 1 / M + 1 / A = 1 / 20)
    (h2 : 9 / 200 = 1 / A) : M = 200 :=
by
    sorry

end Micheal_work_rate_l479_479499


namespace perimeter_of_rectangle_l479_479572

-- Define the properties of the rectangle based on the given conditions
variable (l w : ℝ)
axiom h1 : l + w = 7
axiom h2 : 2 * l + w = 9.5

-- Define the function for perimeter of the rectangle
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

-- Formal statement of the proof problem
theorem perimeter_of_rectangle : perimeter l w = 14 := by
  -- Given conditions
  have h3 : l = 2.5 := sorry
  have h4 : w = 4.5 := sorry
  -- Conclusion based on the conditions
  show perimeter l w = 14 from sorry

end perimeter_of_rectangle_l479_479572


namespace expression_value_l479_479574

theorem expression_value : (8 * 6) - (4 / 2) = 46 :=
by
  sorry

end expression_value_l479_479574


namespace Ryan_hours_l479_479016

variables (English Chinese Spanish : ℕ)
variables (hEnglish : English = 6) (hChinese : Chinese = 7) (hSpanish : Spanish = 4)

theorem Ryan_hours :
  abs (Chinese - (English + Spanish)) = 3 :=
by
  rw [hEnglish, hChinese, hSpanish]
  norm_num
  sorry

end Ryan_hours_l479_479016


namespace company_employee_decrease_l479_479506

-- Let E be the total number of employees before July 1
-- Let S be the average salary per employee before July 1
-- Let E' be the total number of employees after July 1

variable {E E' : ℝ} (S : ℝ)

-- Conditions
def condition1 (E E' : ℝ) (S : ℝ): Prop :=
  -- The total salary remains the same before and after July 1
  E' * 1.1 * S = E * S

-- Define the percent decrease
def percent_decrease (E E' : ℝ): ℝ :=
  (E - E') / E * 100

-- Statement of the theorem
theorem company_employee_decrease (S : ℝ) (E E' : ℝ) 
  (h : condition1 E E' S) : 
  percent_decrease E E' = 9.09 :=
by
  -- Apply the given condition
  sorry

end company_employee_decrease_l479_479506


namespace rackets_packed_l479_479944

theorem rackets_packed (total_cartons : ℕ) (cartons_3 : ℕ) (cartons_2 : ℕ) 
  (h1 : total_cartons = 38) 
  (h2 : cartons_3 = 24) 
  (h3 : cartons_2 = total_cartons - cartons_3) :
  3 * cartons_3 + 2 * cartons_2 = 100 := 
by
  -- The proof is omitted
  sorry

end rackets_packed_l479_479944


namespace radius_of_circle_from_spherical_coords_l479_479559

theorem radius_of_circle_from_spherical_coords :
  ∀ (θ: ℝ), let ρ := 1, φ := π / 3 in
  (√(ρ * sin φ * cos θ)^2 + (ρ * sin φ * sin θ)^2) = √3 / 2 :=
by
  intros θ
  let ρ := 1
  let φ := π / 3
  sorry

end radius_of_circle_from_spherical_coords_l479_479559


namespace functional_equation_holds_l479_479344

noncomputable def f (x : ℝ) : ℝ :=
  (x + 1) / (x - 1)

theorem functional_equation_holds (x : ℝ) (hx : x ∉ set.Icc 0 1) :
  f(x) + f(1 / (1 - x)) = 2 * (1 - 2 * x) / (x * (1 - x)) :=
by
  sorry

end functional_equation_holds_l479_479344


namespace roots_of_quadratic_are_coprime_powers_l479_479062

theorem roots_of_quadratic_are_coprime_powers 
  (p : ℤ) (h1 : odd p)
  (x1 x2 : ℂ) (h2 : x1^2 + ↑p * x1 - 1 = 0) (h3 : x2^2 + ↑p * x2 - 1 = 0):
  ∀ n : ℕ, coprime (x1^n + x2^n) (x1^(n + 1) + x2^(n + 1)) :=
by 
  sorry

end roots_of_quadratic_are_coprime_powers_l479_479062


namespace Mike_does_not_need_additional_space_l479_479189

theorem Mike_does_not_need_additional_space :
  let available_space := 28 -- GB
  let backup_files := 26 -- GB
  let software_installation := 4 -- GB
  let new_files := 6 -- GB
  let compression_ratio := 0.65
  let compressed_backup_files := backup_files * compression_ratio
  let total_needed_space := compressed_backup_files + software_installation + new_files
  total_needed_space <= available_space :=
by 
  let available_space := 28 -- GB
  let backup_files := 26 -- GB
  let software_installation := 4 -- GB
  let new_files := 6 -- GB
  let compression_ratio := 0.65
  let compressed_backup_files := backup_files * compression_ratio
  let total_needed_space := compressed_backup_files + software_installation + new_files
  have h1 : total_needed_space = 26 * 0.65 + 4 + 6, by unfold compressed_backup_files
  have h2 : 26 * 0.65 = 16.9, by norm_num
  have h3 : total_needed_space = 16.9 + 4 + 6, by rw [h1, h2]
  have h4 : 16.9 + 4 = 20.9, by norm_num
  have h5 : total_needed_space = 20.9 + 6, by rw h3
  have h6 : 20.9 + 6 = 26.9, by norm_num
  have h7 : total_needed_space = 26.9, by rw h5
  have h8 : 28 >= 26.9, by norm_num
  exact h8

end Mike_does_not_need_additional_space_l479_479189


namespace sin_omega_not_monotonic_interval_l479_479752

/-- 
Find the number of values of ω for which the function y = sin(ω x) 
is not monotonic on the interval [π/4, π/3].
Given that ω ∈ ℕ* and ω ≤ 15.
-/
theorem sin_omega_not_monotonic_interval : 
  ∃ (n : ℕ), n = 8 ∧ ∀ ω : ℕ, (1 ≤ ω ∧ ω ≤ 15) → 
  ¬ (monotonic_on (λ x : ℝ, Real.sin (ω * x)) (Set.Icc (π / 4) (π / 3))) ↔ ω ∈ {5, 8, 9, 11, 12, 13, 14, 15} :=
by
  sorry

end sin_omega_not_monotonic_interval_l479_479752


namespace time_on_sideline_l479_479137

def total_game_time : ℕ := 90
def time_mark_played_first_period : ℕ := 20
def time_mark_played_second_period : ℕ := 35
def total_time_mark_played : ℕ := time_mark_played_first_period + time_mark_played_second_period

theorem time_on_sideline : total_game_time - total_time_mark_played = 35 := by
  sorry

end time_on_sideline_l479_479137


namespace mass_percentage_of_Al_in_Al₂_SO₄₃_l479_479345

-- Atomic masses for the elements used in the compound
constant Al_mass : ℝ := 26.98 -- g/mol
constant S_mass : ℝ := 32.07 -- g/mol
constant O_mass : ℝ := 16.00 -- g/mol

-- Composition of aluminum sulfate, Al₂(SO₄)₃
def Al₂_SO₄₃_molar_mass := 2 * Al_mass + 3 * S_mass + 12 * O_mass

def Al₂_SO₄₃_mass_percentage_of_Al := (2 * Al_mass / Al₂_SO₄₃_molar_mass) * 100

theorem mass_percentage_of_Al_in_Al₂_SO₄₃ : 
  Al₂_SO₄₃_mass_percentage_of_Al ≈ 15.77 := 
sorry

end mass_percentage_of_Al_in_Al₂_SO₄₃_l479_479345


namespace divide_into_groups_l479_479901

theorem divide_into_groups (n m k : ℕ) (h1 : m ≥ n) (h2 : (n * (n + 1)) / 2 = m * k) :
  ∃ groups : list (list ℕ), (∀ g ∈ groups, m = g.sum) ∧ (∃ g, groups = g.permutations.symm, 
  g = [1..n]) :=
  sorry

end divide_into_groups_l479_479901


namespace circle_through_intersections_and_point_l479_479359

-- Definitions of given circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

-- Given point (1, -1)
def P1 : (ℝ × ℝ) := (1, -1)

-- Proof problem statement
theorem circle_through_intersections_and_point : 
  ∃ (λ : ℝ), 
    (∀ (x y : ℝ), (x^2 + y^2 + 4 * x - 4 * y + λ * (x^2 + y^2 - x) = 0) → 
     (C1 x y) ∧ (C2 x y)) ∧
    (let (a, b) := P1 in (a^2 + b^2 + 4 * a - 4 * b + λ * (a^2 + b^2 - a) = 0)) ∧ 
    (9 * x^2 + 9 * y^2 - 14 * x + 4 * y = 0) := sorry

end circle_through_intersections_and_point_l479_479359


namespace mark_sideline_time_l479_479136

def total_game_time : ℕ := 90
def initial_play : ℕ := 20
def second_play : ℕ := 35
def total_play_time : ℕ := initial_play + second_play
def sideline_time : ℕ := total_game_time - total_play_time

theorem mark_sideline_time : sideline_time = 35 := by
  sorry

end mark_sideline_time_l479_479136


namespace corrected_mean_l479_479546

theorem corrected_mean (n : ℕ) (original_mean incorrect_correct : ℕ -> ℕ) (incorrect : ℕ) (correct : ℕ) :
  n = 50 ->
  original_mean n = 36 ->
  incorrect = 23 ->
  correct = 45 ->
  incorrect_correct = λ incorrect correct, (original_mean n * n - incorrect + correct) / n ->
  incorrect_correct incorrect correct = 36.44 :=
by
  intros n_eq original_mean_eq incorrect_eq correct_eq incorrect_correct_def
  sorry

end corrected_mean_l479_479546


namespace corrected_mean_of_observations_l479_479548

theorem corrected_mean_of_observations :
  (mean : ℝ) (n : ℕ) (inc_obs corr_obs : ℝ)
  (h1 : n = 50)
  (h2 : mean = 36)
  (h3 : inc_obs = 23)
  (h4 : corr_obs = 45) :
  (mean * n - inc_obs + corr_obs) / n = 36.44 :=
by
  sorry

end corrected_mean_of_observations_l479_479548


namespace exhibition_adult_child_ratio_l479_479947

theorem exhibition_adult_child_ratio (a c : ℕ) 
  (h1 : 30 * a + 15 * c = 2250) 
  (h2 : a ≥ 1) 
  (h3 : c ≥ 1) 
  : a / c = 1 := by
  -- Prove the result
  sorry

end exhibition_adult_child_ratio_l479_479947


namespace passengers_on_first_stop_l479_479984

theorem passengers_on_first_stop (x : ℕ) : 
    (∀ init_passengers first_stop net_change final_passengers, 
        init_passengers = 50 → 
        net_change = 22 - 5 → 
        final_passengers = 49 → 
        init_passengers + x - net_change = final_passengers) → 
    x = 16 :=
by 
    intro h
    have init_eq : 50 = 50 := rfl
    have net_eq : 17 = 17 := rfl
    have final_eq : 49 = 49 := rfl
    specialize h 50 16 17 49 init_eq net_eq final_eq
    exact h

end passengers_on_first_stop_l479_479984


namespace find_omega_intervals_of_increase_l479_479494

noncomputable def f (ω x : ℝ) : ℝ := (sin (ω * x) + cos (ω * x))^2 + 2 * (cos (ω * x))^2

theorem find_omega (ω : ℝ) (h₁ : ω > 0) (h₂ : ∀ x, f ω (x + (2 * π / 3)) = f ω x) : ω = 3 / 2 :=
sorry

noncomputable def g (ω x : ℝ) : ℝ := f ω (x - π / 2)

theorem intervals_of_increase (ω : ℝ) (h₁ : ω = 3 / 2) :
  ∀ k : ℤ, (∃ a b : ℝ, (a = 2 / 3 * k * π + π / 4) ∧ (b = 2 / 3 * k * π + 7 * π / 12) ∧
           ∀ x, (a ≤ x ∧ x ≤ b) → (g ω).derivative x > 0) :=
sorry

end find_omega_intervals_of_increase_l479_479494


namespace ab_sum_l479_479112

theorem ab_sum (a b : ℝ) (h : |a - 4| + (b + 1)^2 = 0) : a + b = 3 :=
by
  sorry -- this is where the proof would go

end ab_sum_l479_479112


namespace arrange_consecutive_integers_no_common_divisors_l479_479845

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def adjacent (i j : ℕ) (n m : ℕ) : Prop :=
  (abs (i - n) = 1 ∧ j = m) ∨
  (i = n ∧ abs (j - m) = 1) ∨
  (abs (i - n) = 1 ∧ abs (j - m) = 1)

theorem arrange_consecutive_integers_no_common_divisors :
  let grid := [[8, 9, 10], [5, 7, 11], [6, 13, 12]] in
  ∀ i j n m, i < 3 → j < 3 → n < 3 → m < 3 →
  adjacent i j n m →
  is_coprime (grid[i][j]) (grid[n][m]) :=
by
  -- This is where you would prove the theorem
  sorry

end arrange_consecutive_integers_no_common_divisors_l479_479845


namespace seventeen_power_seven_mod_eleven_l479_479527

-- Define the conditions
def mod_condition : Prop := 17 % 11 = 6

-- Define the main goal (to prove the correct answer)
theorem seventeen_power_seven_mod_eleven (h : mod_condition) : (17^7) % 11 = 8 := by
  -- Proof goes here
  sorry

end seventeen_power_seven_mod_eleven_l479_479527


namespace imaginary_part_of_z_l479_479116

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4 * I) * z = abs (4 + 3 * I)) : im z = 4 / 5 :=
sorry

end imaginary_part_of_z_l479_479116


namespace total_revenue_correct_l479_479647

noncomputable def total_ticket_revenue : ℕ :=
  let revenue_2pm := 180 * 6 + 20 * 5 + 60 * 4 + 20 * 3 + 20 * 5
  let revenue_5pm := 95 * 8 + 30 * 7 + 110 * 5 + 15 * 6
  let revenue_8pm := 122 * 10 + 74 * 7 + 29 * 8
  revenue_2pm + revenue_5pm + revenue_8pm

theorem total_revenue_correct : total_ticket_revenue = 5160 := by
  sorry

end total_revenue_correct_l479_479647


namespace number_of_ways_two_girls_together_l479_479462

theorem number_of_ways_two_girls_together
  (boys girls : ℕ)
  (total_people : ℕ)
  (ways : ℕ) :
  boys = 3 →
  girls = 3 →
  total_people = boys + girls →
  ways = 432 :=
by
  intros
  sorry

end number_of_ways_two_girls_together_l479_479462


namespace initial_amount_l479_479040

theorem initial_amount (x : ℕ) (h1 : x - 3 + 14 = 22) : x = 11 :=
sorry

end initial_amount_l479_479040


namespace ratio_of_radii_l479_479590

theorem ratio_of_radii (r R : ℝ) (k : ℝ) (h1 : R > r) (h2 : π * R^2 - π * r^2 = k * π * r^2) :
  R / r = Real.sqrt (k + 1) :=
sorry

end ratio_of_radii_l479_479590


namespace maximum_omega_l479_479421

theorem maximum_omega
  (ω : ℝ)
  (hω : 0 < ω)
  (h_mono_incr : ∀ x y : ℝ, -π/3 < x ∧ x < π/2 → 
                             -π/3 < y ∧ y < π/2 → 
                             x < y → tan (ω * x + π/4) < tan (ω * y + π/4)) :
  ω ≤ 1/2 :=
by
  sorry

end maximum_omega_l479_479421


namespace houses_with_only_one_pet_l479_479453

theorem houses_with_only_one_pet (h_total : ∃ t : ℕ, t = 75)
                                 (h_dogs : ∃ d : ℕ, d = 40)
                                 (h_cats : ∃ c : ℕ, c = 30)
                                 (h_dogs_and_cats : ∃ dc : ℕ, dc = 10)
                                 (h_birds : ∃ b : ℕ, b = 8)
                                 (h_cats_and_birds : ∃ cb : ℕ, cb = 5)
                                 (h_no_dogs_and_birds : ∀ db : ℕ, ¬ (∃ db : ℕ, db = 1)) :
  ∃ n : ℕ, n = 48 :=
by
  have only_dogs := 40 - 10
  have only_cats := 30 - 10 - 5
  have only_birds := 8 - 5
  have result := only_dogs + only_cats + only_birds
  exact ⟨result, sorry⟩

end houses_with_only_one_pet_l479_479453


namespace bound_diff_sqrt_two_l479_479491

theorem bound_diff_sqrt_two (a b k m : ℝ) (h : ∀ x ∈ Set.Icc a b, abs (x^2 - k * x - m) ≤ 1) : b - a ≤ 2 * Real.sqrt 2 := sorry

end bound_diff_sqrt_two_l479_479491


namespace perpendicular_vectors_lambda_l479_479792

variable (λ : ℝ)
def vector_a := (λ, -2)
def vector_b := (λ - 1, 1)

theorem perpendicular_vectors_lambda :
  (vector_a λ).fst * (vector_b λ).fst + (vector_a λ).snd * (vector_b λ).snd = 0 ↔ λ = -1 ∨ λ = 2 := by
  dsimp [vector_a, vector_b]
  sorry

end perpendicular_vectors_lambda_l479_479792


namespace does_not_pass_through_second_quadrant_l479_479242

def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

theorem does_not_pass_through_second_quadrant :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ x < 0 ∧ y > 0 :=
sorry

end does_not_pass_through_second_quadrant_l479_479242


namespace train_length_l479_479312

noncomputable def speed_in_m_s (speed_km_hr : ℝ) : ℝ :=
  speed_km_hr * (1000 / 3600)

def length_of_train (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  speed_in_m_s speed_km_hr * time_seconds

theorem train_length 
  (time_crossing_pole : ℝ)
  (speed_km_hr : ℝ)
  (H1 : time_crossing_pole = 9.99920006399488)
  (H2 : speed_km_hr = 36) :
  length_of_train speed_km_hr time_crossing_pole ≈ 99.9920006399488 :=
by
  sorry

end train_length_l479_479312


namespace number_of_correct_statements_l479_479541

theorem number_of_correct_statements :
  (∀ (triangle : Type) [is_equilateral_triangle : equilateral triangle], 
    (triangle_interior_angle : ℝ) (h1 : ∀ (a b c : ℝ), a = b ∧ b = c → a + b + c = 180) ∧
    (triangle_angle_eq_60 : ℝ) (h2 : ∀ (a b c : ℝ), (a = 60 ∧ b = 60 ∧ c = 60) → a + b + c = 180) ∧ 
    (triangle_all_angle_eq : ℝ) (h3 : ∀ (a b c : ℝ), (a = 60 ∧ b = 60 ∧ c = 60) → ∃ (equilateral triangle : Type), equilateral triangle = triangle) ∧
    (isosceles_eq_60 : ℝ) (h4 : ∀ (a b c : ℝ), (iso_triangle : a = b ∧ (a + b + c = 180) ∧ (c = 60)) → equilateral_triangle)
  ) →
  4 :=
sorry

end number_of_correct_statements_l479_479541


namespace max_real_roots_l479_479338

noncomputable def p (n : ℕ) (x : ℝ) : ℝ :=
  ∑ i in (Finset.range (n+1)), (ite (even i) 1 (-1)) * (i + 1) * (x ^ (n - i))

theorem max_real_roots (n : ℕ) (n_pos : 0 < n) :
  ∃ (r : ℕ), r ≤ n ∧ ∀ x : ℝ, p n x = 0 → x ∈ Finset.range (n + 1) := 
sorry

end max_real_roots_l479_479338


namespace cos_angle_relation_l479_479754

theorem cos_angle_relation (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) : Real.cos (2 * α - 2 * π / 3) = -7 / 9 := by 
  sorry

end cos_angle_relation_l479_479754


namespace avg_production_last_5_days_l479_479624

-- Define the conditions
def avg_production_first_25_days : ℕ := 65
def days_first_period : ℕ := 25
def avg_production_month : ℕ := 60
def total_days_month : ℕ := 30
def days_second_period : ℕ := 5

-- Formulate the theorem
theorem avg_production_last_5_days :
  let total_production_first_25_days := days_first_period * avg_production_first_25_days,
      total_monthly_production := total_days_month * avg_production_month,
      total_production_last_5_days := total_monthly_production - total_production_first_25_days,
      avg_production_last_5_days := total_production_last_5_days / days_second_period
  in avg_production_last_5_days = 35 :=
by {
  let total_production_first_25_days := days_first_period * avg_production_first_25_days,
  let total_monthly_production := total_days_month * avg_production_month,
  let total_production_last_5_days := total_monthly_production - total_production_first_25_days,
  let avg_production_last_5_days := total_production_last_5_days / days_second_period,
  have h : total_production_first_25_days = 1625 := rfl,
  have h1 : total_monthly_production = 1800 := rfl,
  have h2 : total_production_last_5_days = 175 := rfl,
  have h3 : avg_production_last_5_days = 35 := rfl,
  exact h3,
}

end avg_production_last_5_days_l479_479624


namespace fixed_point_l479_479543

noncomputable def f (a x : ℝ) : ℝ := log a (2 * x - 3) + 1

theorem fixed_point (a : ℝ) (ha : 0 < a ∧ a ≠ 1) : f a 2 = 1 :=
by
  -- property log_a(1) = 0 simplifies logging 1 to 0
  -- 2x - 3 = 1 at x = 2 means f 2 = 1
  sorry

end fixed_point_l479_479543


namespace price_of_expensive_feed_l479_479258

theorem price_of_expensive_feed
  (total_weight : ℝ)
  (mix_price_per_pound : ℝ)
  (cheaper_feed_weight : ℝ)
  (cheaper_feed_price_per_pound : ℝ)
  (expensive_feed_price_per_pound : ℝ) :
  total_weight = 27 →
  mix_price_per_pound = 0.26 →
  cheaper_feed_weight = 14.2105263158 →
  cheaper_feed_price_per_pound = 0.17 →
  expensive_feed_price_per_pound = 0.36 :=
by
  intros h1 h2 h3 h4
  sorry

end price_of_expensive_feed_l479_479258


namespace distance_between_parallel_lines_l479_479122

theorem distance_between_parallel_lines :
  let A1 := 3
  let B1 := 4
  let C1 := -3
  let A2 := 6
  let B2 := 8
  let C2 := -14

  let d := abs (C2 - C1) / sqrt (A1^2 + B1^2)
  (A1 = A2) ∧ (B1 = B2) ∧ (A1 * B2 = A2 * B1) →
  d = 2 :=
by
  sorry

end distance_between_parallel_lines_l479_479122


namespace ratio_sum_odd_even_divisors_l479_479892

noncomputable def N : ℕ := 38 * 38 * 91 * 210

theorem ratio_sum_odd_even_divisors :
  let sum_odd_divisors := ∑ d in (Nat.divisors N).filter (λ x, ¬ 2 ∣ x), d,
      sum_even_divisors := ∑ d in (Nat.divisors N).filter (λ x, 2 ∣ x), d
  in sum_odd_divisors * 14 = sum_even_divisors :=
by
  sorry

end ratio_sum_odd_even_divisors_l479_479892


namespace hannah_highest_score_l479_479102

def total_questions : Nat := 50
def student1_score : Nat := 47
def student2_score : Nat := 47
def student3_score : Nat := 46
def student4_score : Nat := 46

theorem hannah_highest_score 
  (T : Nat := total_questions)
  (S1 : Nat := student1_score)
  (S2 : Nat := student2_score)
  (S3 : Nat := student3_score)
  (S4 : Nat := student4_score)
  : Nat :=
  let highest_other_score := max (max (max S1 S2) S3) S4
  highest_other_score + 1

-- Here we define hannah_highest_score to be 48 as calculated.
#eval hannah_highest_score -- 48

end hannah_highest_score_l479_479102


namespace domain_of_log_function_l479_479953

-- Define the problematic quadratic function
def quadratic_fn (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Define the domain condition for our function
def domain_condition (x : ℝ) : Prop := quadratic_fn x > 0

-- The actual statement to prove, stating that the domain is (1, 3)
theorem domain_of_log_function :
  {x : ℝ | domain_condition x} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end domain_of_log_function_l479_479953


namespace prob_a_greater_than_b_l479_479505

noncomputable def probability_of_team_a_finishing_with_more_points (n_teams : ℕ) (initial_win : Bool) : ℚ :=
  if initial_win ∧ n_teams = 9 then
    39203 / 65536
  else
    0 -- This is a placeholder and not accurate for other cases

theorem prob_a_greater_than_b (n_teams : ℕ) (initial_win : Bool) (hp : initial_win ∧ n_teams = 9) :
  probability_of_team_a_finishing_with_more_points n_teams initial_win = 39203 / 65536 :=
by
  sorry

end prob_a_greater_than_b_l479_479505


namespace compute_expression_l479_479904

theorem compute_expression (p q : ℝ) (h1 : p + q = 6) (h2 : p * q = 10) : 
  p^3 + p^4 * q^2 + p^2 * q^4 + p * q^3 + p^5 * q^3 = 38676 := by
  -- Proof goes here
  sorry

end compute_expression_l479_479904


namespace standard_deviation_transformation_l479_479981

noncomputable def variance (data : List ℝ) := sorry -- Define the variance function (this is just a placeholder)

-- The theorem we intend to prove
theorem standard_deviation_transformation (a : List ℝ) (S : ℝ) (h : variance a = S^2) :
    variance (a.map (fun x => 2 * x - 3)) = 4 * S^2 ∧ real.sqrt (4 * S^2) = 2 * S := by
  sorry

end standard_deviation_transformation_l479_479981


namespace greatest_base_five_digit_sum_l479_479269

theorem greatest_base_five_digit_sum (n : ℕ) (h₁ : 0 < n) (h₂ : n < 3139) : 
  ∃ s : ℕ, s = (Option.get (n.digits 5)).sum ∧ s = 16 := 
sorry

end greatest_base_five_digit_sum_l479_479269


namespace distribute_numbers_l479_479088

theorem distribute_numbers :
  ∃ (A B C : Set ℕ), 
    A ∪ B ∪ C = {1, 10, 11, 15, 18, 37, 40} ∧
    A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧
    (∑ x in A, x = 33 ∧ ∑ x in B, x = 99 ∧ (A ∪ B ∪ C).card = 7) :=
sorry

end distribute_numbers_l479_479088


namespace composite_sum_of_divides_l479_479388

open Nat

theorem composite_sum_of_divides (a b c d e f : ℕ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
    (hS_div1 : (a + b + c + d + e + f) ∣ (a * b * c + d * e * f))
    (hS_div2 : (a + b + c + d + e + f) ∣ (a * b + b * c + c * a - d * e - e * f - f * d))
    (S := a + b + c + d + e + f) : ¬Prime S := 
sorry

end composite_sum_of_divides_l479_479388


namespace chameleons_can_become_blue_l479_479508

-- Definitions for the problem conditions
-- Let's assume some abstract definition of colors and the biting rule
inductive Color
| Red
| Blue
| Green
| Yellow
| Purple

-- Abstract biting rule
def bite : Color → Color → Color
| Color.Red, Color.Green => Color.Blue
| Color.Green, Color.Red => Color.Red
| Color.Red, Color.Red => Color.Yellow
-- Other combinations as necessary

-- Define the main theorem statement
theorem chameleons_can_become_blue (k : ℕ) (h : k ≥ 5) :
  ∀ (r : ℕ), r = 5 → 
  ∃ (sequence : list (Color × Color)), 
    ∀ (start_colors : vector Color r), 
    (∀ i, start_colors.nth i = Color.Red) →
    (let end_colors := start_colors.foldl (λ colors rule, colors.map (λ c, bite (fst rule) c)) start_colors 
    in ∀ i, end_colors.nth i = Color.Blue) :=
begin
  sorry
end

end chameleons_can_become_blue_l479_479508


namespace count_he_numbers_l479_479046

noncomputable def f (n : ℕ) : ℝ :=
  Real.logBase (n+1) (n+2)

def is_he_number (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ∃ m : ℕ, Real.logBase 2 (k + 2) = m

def in_interval (n : ℕ) : Prop :=
  1 < n ∧ n < 2015

theorem count_he_numbers : 
  (finset.univ
    .filter (λ n, in_interval n ∧ is_he_number n)).card = 9 :=
by
  sorry

end count_he_numbers_l479_479046


namespace identify_fake_bag_l479_479660

theorem identify_fake_bag (bags : Fin 10 → Fin 2) (weights : Fin 2 → ℕ) :
  (∀ i, weights (bags i) = if bags i = 0 then 10 else 9) →
  let S := ∑ i, (i + 1) * weights (bags i)
  in S = 550 → bags (S - 550) = 1 := 
by sorry

end identify_fake_bag_l479_479660


namespace infinite_geometric_series_proof_l479_479351

-- Define the conditions
def first_term (a : ℚ) : Prop := a = 1 / 2
def common_ratio (r : ℚ) : Prop := r = 1 / 4

-- Define the target: the sum of the infinite geometric series
def infinite_geometric_series_sum (a r S : ℚ) : Prop :=
  S = a / (1 - r)

-- Prove the problem statement
theorem infinite_geometric_series_proof :
  ∃ S : ℚ, first_term (1 / 2) ∧ common_ratio (1 / 4) ∧ infinite_geometric_series_sum (1 / 2) (1 / 4) S :=
begin
  use 2 / 3,
  split, { refl },
  split, { refl },
  sorry
end

end infinite_geometric_series_proof_l479_479351


namespace no_such_numbers_l479_479155

/-- For every n ≥ 2021, there do not exist n integer numbers such that the 
square of each number is equal to the sum of all other numbers, and not 
all the numbers are equal. -/
theorem no_such_numbers (n : ℕ) (h : n ≥ 2021) :
  ¬ ∃ (x : ℕ → ℤ), (∀ i, x i ^ 2 = (finset.univ.sum x) - x i) ∧ (¬ ∀ i j, x i = x j) :=
by
  sorry

end no_such_numbers_l479_479155


namespace notebook_cost_l479_479644

theorem notebook_cost :
  let mean_expenditure := 500
  let daily_expenditures := [450, 600, 400, 500, 550, 300]
  let cost_earphone := 620
  let cost_pen := 30
  let total_days := 7
  let total_expenditure := mean_expenditure * total_days
  let sum_other_days := daily_expenditures.sum
  let expenditure_friday := total_expenditure - sum_other_days
  let cost_notebook := expenditure_friday - (cost_earphone + cost_pen)
  cost_notebook = 50 := by
  sorry

end notebook_cost_l479_479644


namespace abs_x_minus_one_sufficient_but_not_necessary_for_quadratic_l479_479483

theorem abs_x_minus_one_sufficient_but_not_necessary_for_quadratic (x : ℝ) :
  (|x - 1| < 2) → (x^2 - 4 * x - 5 < 0) ∧ ¬(x^2 - 4 * x - 5 < 0 → |x - 1| < 2) :=
by
  sorry

end abs_x_minus_one_sufficient_but_not_necessary_for_quadratic_l479_479483


namespace compare_a_n_inequality_Sn_Tn_l479_479627

variables (n : ℕ) (Rn Rn1 : ℝ) (a_n a_n1 : ℝ) (Sn Tn : ℝ)

-- Establish conditions given in the problem
variables (C_n : ℝ → ℝ → ℝ)
variables (M N : ℝ × ℝ)
variable (intersects_y_axis : M = (0, Rn))
variable (intersects_curve : N = (1 / n, real.sqrt (1 / n)))
variable (point_A : ℝ × ℝ)
variable (line_MN_intersects_x_axis : point_A = (a_n, 0))
variable (Sn : ℝ)
variable (Tn : ℝ)

-- Define sequences
variables (S : ℕ → ℝ) (T : ℕ → ℝ)

-- Prove that a_n > a_{n+1} > 2
theorem compare_a_n : a_n > a_n1 ∧ a_n1 > 2 :=
sorry

-- Prove that 7/5 < (S_n - 2n) / T_n < 3/2
theorem inequality_Sn_Tn : 7 / 5 < (Sn - 2 * n) / Tn ∧ (Sn - 2 * n) / Tn < 3 / 2 :=
sorry

end compare_a_n_inequality_Sn_Tn_l479_479627


namespace complement_intersection_l479_479787
noncomputable def U : set ℝ := {x | x < 0}
noncomputable def M : set ℝ := {x | x + 1 < 0}
noncomputable def N : set ℝ := {x | 1 / 8 < 2^x ∧ 2^x < 1}

theorem complement_intersection :
  (U \ M) ∩ N = { x : ℝ | -1 ≤ x ∧ x < 0 } :=
by
  sorry

end complement_intersection_l479_479787


namespace smallest_positive_multiple_of_45_l479_479610

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ (∀ y : ℕ, y > 0 → 45 * y ≥ 45 * x) ∧ 45 * x = 45 :=
by
  use 1
  split
  · apply Nat.one_pos
  · split
    · intros y hy
      apply mul_le_mul
      · apply Nat.one_le_of_lt hy
      · apply le_refl
      · apply le_of_lt
        apply Nat.pos_of_ne_zero
        intro h
        exact hy (h.symm ▸ rfl)
      · apply le_of_lt
        apply Nat.pos_of_ne_zero
        intro h
        exact hy (h.symm ▸ rfl)
    · apply rfl
  sorry

end smallest_positive_multiple_of_45_l479_479610


namespace frog_arrangements_l479_479578

theorem frog_arrangements : 
  let num_frogs := 8 in
  let num_green := 3 in
  let num_red := 4 in
  let num_blue := 1 in
  let factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1) in
  let arrangements_per_grouping := 2 * factorial num_green * factorial num_red in
  let total_arrangements := 2 * arrangements_per_grouping in
  total_arrangements = 576 :=
by
  sorry

end frog_arrangements_l479_479578


namespace find_multiple_of_numerator_l479_479534

theorem find_multiple_of_numerator
  (n d k : ℕ)
  (h1 : d = k * n - 1)
  (h2 : (n + 1) / (d + 1) = 3 / 5)
  (h3 : (n : ℚ) / d = 5 / 9) : k = 2 :=
sorry

end find_multiple_of_numerator_l479_479534


namespace exists_polynomial_T_l479_479473

variable {R : Type*} [CommRing R] [Algebra R $Polynomial R]

theorem exists_polynomial_T
  {P Q R S : Polynomial R} 
  (h1 : ¬P.is_constant)
  (h2 : ¬Q.is_constant)
  (h3 : ¬R.is_constant)
  (h4 : ¬S.is_constant)
  (hPQRS : P.eval₂ (C : R → Polynomial R) (Q : Polynomial R) = R.eval₂ (C : R → Polynomial R) (S : Polynomial R))
  (hdeg : P.degree % R.degree = 0) :
  ∃ T : Polynomial R, P = R.comp T :=
sorry

end exists_polynomial_T_l479_479473


namespace sum_infinite_series_l479_479002

noncomputable def G : ℕ → ℚ
| 0       := 0
| 1       := 5 / 4
| (n + 2) := 3 * G (n + 1) - (1 / 2) * G n

theorem sum_infinite_series : 
  (∑' n : ℕ, 1 / (3^(3^n) - 1/(3^(3^n)))) = 1 :=
sorry

end sum_infinite_series_l479_479002


namespace interval_second_bell_l479_479675

theorem interval_second_bell 
  (T : ℕ)
  (h1 : ∀ n : ℕ, n ≠ 0 → 630 % n = 0)
  (h2 : gcd T 630 = T)
  (h3 : lcm 9 (lcm 14 18) = lcm 9 (lcm 14 18))
  (h4 : 630 % lcm 9 (lcm 14 18) = 0) : 
  T = 5 :=
sorry

end interval_second_bell_l479_479675


namespace pigeonhole_principle_useful_inequality_l479_479167

open Finset

theorem pigeonhole_principle_useful_inequality (n : ℕ) (A : Finset ℕ) (hA : A = range (2^(n+1))) (B : Finset ℕ) (hB : B ⊆ A) (h_cardB : B.card = 2 * n + 1) :
  ∃ a b c ∈ B, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ b * c < 2 * a^2 ∧ 2 * a^2 < 4 * b * c := 
by
  sorry

end pigeonhole_principle_useful_inequality_l479_479167


namespace flour_needed_for_one_batch_l479_479195

theorem flour_needed_for_one_batch (F : ℝ) (h1 : 8 * F + 8 * 1.5 = 44) : F = 4 := 
by
    sorry

end flour_needed_for_one_batch_l479_479195


namespace exists_equal_subinterval_l479_479908

open Set Metric Function

variable {a b : ℝ}
variable {f : ℕ → ℝ → ℝ}
variable {n m : ℕ}

-- Define the conditions
def continuous_on_interval (f : ℕ → ℝ → ℝ) (a b : ℝ) :=
  ∀ n, ContinuousOn (f n) (Icc a b)

def root_cond (f : ℕ → ℝ → ℝ) (a b : ℝ) :=
  ∀ x ∈ Icc a b, ∃ m n, m ≠ n ∧ f m x = f n x

-- The main theorem statement
theorem exists_equal_subinterval (f : ℕ → ℝ → ℝ) (a b : ℝ) 
  (h_cont : continuous_on_interval f a b) 
  (h_root : root_cond f a b) : 
  ∃ (c d : ℝ), c < d ∧ Icc c d ⊆ Icc a b ∧ ∃ m n, m ≠ n ∧ ∀ x ∈ Icc c d, f m x = f n x := 
sorry

end exists_equal_subinterval_l479_479908


namespace max_visible_sum_four_cubes_l479_479372

def face_numbers : List ℕ := [1, 3, 9, 27, 81, 243]

def cube (faces : List ℕ) : Prop := faces.length = 6 ∧ ∀ x ∈ faces, x ∈ face_numbers

def cube_stack (cubes : List (List ℕ)) : Prop :=
  cubes.length = 4 ∧ ∀ c ∈ cubes, cube c

noncomputable def max_visible_sum (cubes : List (List ℕ)) : ℕ :=
  List.foldl (λ acc c, acc + List.sum (List.take 5 c)) 0 cubes

theorem max_visible_sum_four_cubes (cubes : List (List ℕ)) (h : cube_stack cubes) :
  max_visible_sum cubes = 1444 :=
sorry

end max_visible_sum_four_cubes_l479_479372


namespace cost_of_blackberry_jam_l479_479013

theorem cost_of_blackberry_jam
  (B J N : ℕ) (hN : 1 < N) (h_eq : N * (6 * B + 7 * J) = 396) :
  7 * J * N / 100 = 3.78 :=
by
  sorry

end cost_of_blackberry_jam_l479_479013


namespace derivative_at_one_l479_479812

def f (x : ℝ) : ℝ := x^3 - f' 1 * x^2 + 3

theorem derivative_at_one :
  (deriv f 1) = 1 := by
sorry

end derivative_at_one_l479_479812


namespace probability_multiple_of_4_l479_479325

-- Definition of the problem conditions
def random_integer (n : ℕ) := ∀ i, 0 < i ∧ i ≤ n → Prop

def multiple_of_4 (i : ℕ) : Prop := i % 4 = 0

def count_multiples_of_4 (n : ℕ) : ℕ := (finset.range n).filter (λ x, multiple_of_4 x).card

-- Given problem conditions
def ben_choose_random_integer : Prop :=
  ∃ x y : ℕ, random_integer 60 x ∧ random_integer 60 y

-- Required proof statement
theorem probability_multiple_of_4 :
  (count_multiples_of_4 60 = 15) →
  (ben_choose_random_integer) →
  let probability := 1 - (3/4) * (3/4)
  in probability = 7/16 :=
begin
  intros h_multiples h_ben_choose,
  sorry
end

end probability_multiple_of_4_l479_479325


namespace complex_value_of_z_l479_479065

theorem complex_value_of_z (a b : ℝ) (z : ℂ) : 
  (a + b * complex.i = 0) ∧ (b^2 + 4 * b + 4 = 0) → z = 2 - 2 * complex.i :=
by
  intro h
  have ha : a = -b := by sorry
  have hb : b = -2 := by sorry
  rw [ha, hb]
  exact sorry

end complex_value_of_z_l479_479065


namespace repeating_decimal_eq_fraction_l479_479026

theorem repeating_decimal_eq_fraction :
  let a := (85 : ℝ) / 100
  let r := (1 : ℝ) / 100
  (∑' n : ℕ, a * (r ^ n)) = 85 / 99 := by
  let a := (85 : ℝ) / 100
  let r := (1 : ℝ) / 100
  exact sorry

end repeating_decimal_eq_fraction_l479_479026


namespace mean_and_sum_l479_479570

-- Define the sum of five numbers to be 1/3
def sum_of_five_numbers : ℚ := 1 / 3

-- Define the mean of these five numbers
def mean_of_five_numbers : ℚ := sum_of_five_numbers / 5

-- State the theorem
theorem mean_and_sum (h : sum_of_five_numbers = 1 / 3) :
  mean_of_five_numbers = 1 / 15 ∧ (mean_of_five_numbers + sum_of_five_numbers = 2 / 5) :=
by
  sorry

end mean_and_sum_l479_479570


namespace volume_pyramid_PABC_l479_479544

-- Definitions based on the given conditions
-- Representation of volume calculation problem in Lean 4
noncomputable def volume_of_pyramid 
  (PA PB PC : ℝ)
  (base_equilateral : Prop)
  (equal_lateral_areas : Prop) : ℝ :=
  if PA = 2 ∧ PB = 2 ∧ PC = 3 ∧ base_equilateral ∧ equal_lateral_areas then
    (5 / (16 * sqrt(2)))
  else
    0

-- Stating the theorem
theorem volume_pyramid_PABC :
  let PA := 2
  let PB := 2
  let PC := 3
  (base_equilateral : Prop) (equal_lateral_areas : Prop) :
  base_equilateral → equal_lateral_areas →
  volume_of_pyramid PA PB PC base_equilateral equal_lateral_areas = (5 / (16 * sqrt(2))) :=
by
  intros
  -- The actual proof would go here
  sorry

end volume_pyramid_PABC_l479_479544


namespace derivative_of_f_l479_479765

def f (x : ℝ) : ℝ := x^3 + 3^x + Real.log 3

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 3 * x^2 + 3^x * Real.log 3 :=
by
  sorry

end derivative_of_f_l479_479765


namespace problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l479_479115

-- Problem G6.1
theorem problem_G6_1 : (21 ^ 3 - 11 ^ 3) / (21 ^ 2 + 21 * 11 + 11 ^ 2) = 10 := 
  sorry

-- Problem G6.2
theorem problem_G6_2 (p q : ℕ) (h1 : (p : ℚ) * 6 = 4 * (q : ℚ)) : q = 3 * p / 2 := 
  sorry

-- Problem G6.3
theorem problem_G6_3 (q r : ℕ) (h1 : q % 7 = 3) (h2 : r % 7 = 5) (h3 : 18 < r) (h4 : r < 26) : r = 24 := 
  sorry

-- Problem G6.4
def star (a b : ℕ) : ℕ := a * b + 1

theorem problem_G6_4 : star (star 3 4) 2 = 27 := 
  sorry

end problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l479_479115


namespace jenn_money_left_over_l479_479160

-- Definitions based on problem conditions
def num_jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def value_per_quarter : ℚ := 0.25   -- Rational number to represent $0.25
def cost_of_bike : ℚ := 180         -- Rational number to represent $180

-- Statement to prove that Jenn will have $20 left after buying the bike
theorem jenn_money_left_over : 
  (num_jars * quarters_per_jar * value_per_quarter) - cost_of_bike = 20 :=
by
  sorry

end jenn_money_left_over_l479_479160


namespace general_term_sum_sequence_l479_479739

noncomputable theory

-- Define the sequence
def a (n : ℕ) : ℝ :=
  if n = 0 then 0 else real.sqrt (2 * n - 1)

-- Define the sequence b
def b (n : ℕ) : ℝ :=
  if n = 0 then 0 else (a n) ^ 2 / (2 ^ n)

-- Define the sum of the first n terms of the sequence b
def S (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, b (k + 1))

-- Prove that the general term a_n = sqrt(2n - 1)
theorem general_term (n : ℕ) (h : n > 0) : a n = real.sqrt (2 * n - 1) :=
begin
  sorry
end

-- Prove the sum Sn = 3 - (2n + 3) / 2^n
theorem sum_sequence (n : ℕ) : S n = 3 - (2 * n + 3) / (2 ^ n) :=
begin
  sorry
end

end general_term_sum_sequence_l479_479739


namespace number_of_blue_parrots_l479_479920

noncomputable def total_parrots : ℕ := 108
def fraction_blue_parrots : ℚ := 1 / 6

theorem number_of_blue_parrots : (fraction_blue_parrots * total_parrots : ℚ) = 18 := 
by
  sorry

end number_of_blue_parrots_l479_479920


namespace initial_invitation_count_l479_479638

def people_invited (didnt_show : ℕ) (num_tables : ℕ) (people_per_table : ℕ) : ℕ :=
  didnt_show + num_tables * people_per_table

theorem initial_invitation_count (didnt_show : ℕ) (num_tables : ℕ) (people_per_table : ℕ)
    (h1 : didnt_show = 35) (h2 : num_tables = 5) (h3 : people_per_table = 2) :
  people_invited didnt_show num_tables people_per_table = 45 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end initial_invitation_count_l479_479638


namespace max_value_of_squares_l479_479903

theorem max_value_of_squares (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 8) :
  a^2 + b^2 + c^2 + d^2 ≤ 4 :=
sorry

end max_value_of_squares_l479_479903


namespace radius_of_inscribed_circle_eq_l479_479696

def semiperimeter (a : ℝ) := 3 * a / 2

def area_of_equilateral_triangle (a : ℝ) := (real.sqrt 3 / 4) * a ^ 2

def inscribed_circle_radius (a : ℝ) := 
  let s := semiperimeter a
  let K := area_of_equilateral_triangle a
  K / s

theorem radius_of_inscribed_circle_eq (a : ℝ) (ha : a = 8) : 
  inscribed_circle_radius a = 4 * real.sqrt 3 / 3 := 
by
  rw[ha]
  sorry

end radius_of_inscribed_circle_eq_l479_479696


namespace find_a_of_binomial_coefficient_l479_479438

theorem find_a_of_binomial_coefficient :
  (∃ (a : ℝ), let c := (2 * (ⅇ : ℝ) - a * (ⅇ⁻¹ : ℝ)^1) in
  (binom.binomial 7 5) * (2 ^ 2) * (-(a^5)) = 84) → a = -1 :=
  by
  sorry

end find_a_of_binomial_coefficient_l479_479438


namespace minimum_sequence_length_l479_479486

def S : Set ℕ := {1, 2, 3, 4}

theorem minimum_sequence_length :
  ∃ (n : ℕ) (q : ℕ → ℕ), (∀ (B : Set ℕ), B ⊆ S ∧ B ≠ ∅ → 
  ∃ (k : ℕ), (k + B.card ≤ n) ∧ (B = {q i | i ∈ finset.range k (k + B.card)})) ∧ n = 7 :=
sorry

end minimum_sequence_length_l479_479486


namespace difference_by_one_exists_l479_479726

theorem difference_by_one_exists (s : Finset ℕ) (h1 : ∀ n ∈ s, 1 ≤ n ∧ n ≤ 50) (h2 : s.card = 26) :
  ∃ (x y ∈ s), x ≠ y ∧ (x = y + 1 ∨ x = y - 1) :=
by sorry

end difference_by_one_exists_l479_479726


namespace eight_faucets_fill_30_gallons_in_60_seconds_l479_479725

-- Constants and assumptions based on the problem conditions
constant four_faucets_fill_time : ℝ := 8     -- in minutes
constant four_faucets_volume : ℝ := 120      -- in gallons
constant target_volume : ℝ := 30             -- in gallons
constant number_of_faucets : ℕ := 8

-- Given four faucets fill a 120-gallon tub in 8 minutes:
def rate_per_faucet := four_faucets_volume / (4 * four_faucets_fill_time)     -- in gallons per minute per faucet

-- Combined rate of eight faucets:
def combined_rate := number_of_faucets * rate_per_faucet                     -- in gallons per minute

-- Time required to fill the target volume with the combined rate, in minutes:
def time_to_fill := target_volume / combined_rate                            -- in minutes

-- Convert time to seconds:
def time_to_fill_seconds := time_to_fill * 60                                -- in seconds

-- Prove that the time to fill the 30-gallon tub with eight faucets is 60 seconds:
theorem eight_faucets_fill_30_gallons_in_60_seconds :
  time_to_fill_seconds = 60 :=
by
  sorry

end eight_faucets_fill_30_gallons_in_60_seconds_l479_479725


namespace tank_capacity_l479_479511

theorem tank_capacity 
  (A_rate : ℕ) (B_rate : ℕ) (C_rate : ℕ) (A_time : ℕ) (B_time : ℕ) (C_time : ℕ) (total_time : ℕ)
  (hA_rate : A_rate = 200) (hB_rate : B_rate = 50) (hC_rate : C_rate = 25)
  (hA_time : A_time = 1) (hB_time : B_time = 2) (hC_time : C_time = 2) (h_total_time : total_time = 100) :
  let cycle_fill := (A_rate * A_time) + (B_rate * B_time) - (C_rate * C_time) in
  let cycle_time := A_time + B_time + C_time in
  let num_cycles := total_time / cycle_time in
  let total_fill := num_cycles * cycle_fill in
  total_fill = 5000 :=
by 
  sorry

end tank_capacity_l479_479511


namespace ranges_of_a_l479_479414

def f (a x : ℝ) : ℝ :=
if x > 1 then a^x else (4 - a / 2) * x + 2

theorem ranges_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → f a x1 ≤ f a x2) → (4 ≤ a ∧ a < 8) :=
by
  sorry

end ranges_of_a_l479_479414


namespace brigade_newspapers_members_l479_479009

theorem brigade_newspapers_members :
  ∃ (n m : ℕ),
  (∀ (members newsltrs : Type) (reads : members → set newsltrs),
    (∀ x : members, (reads x).finite ∧ (reads x).to_finset.card = 2)    ∧
    (∀ y : newsltrs, ∃ (S : finset members), S.finite ∧ S.card = 5 ∧
    ∀ x ∈ S, y ∈ reads x)                                                ∧
    (∀ (y1 y2 : newsltrs), y1 ≠ y2 → (∃! x : members, y1 ∈ reads x ∧ y2 ∈ reads x))
  ) →
  (n = 6 ∧ m = 15) :=
begin
  sorry
end

end brigade_newspapers_members_l479_479009


namespace probability_of_prime_l479_479306

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_in_range (a b : ℕ) : list ℕ :=
  (list.range' a (b - a + 1)).filter is_prime

noncomputable def probability_prime_in_set : ℚ :=
  let total_elements := 200 - 40 + 1 in
  let prime_count := (primes_in_range 40 200).length in
  prime_count / total_elements

theorem probability_of_prime :
  probability_prime_in_set = 35 / 161 :=
sorry

end probability_of_prime_l479_479306


namespace angle_FDE_eq_60_l479_479890

open EuclideanGeometry

variables {A B C D E F P Q : Point}

-- Conditions setup
-- Points P, D, E, and F within triangle ABC
-- Line AP intersects BC at D
-- Line BP intersects CA at E
-- Line CP intersects AB at F
-- Point Q on the ray [BE such that E ∈ [BQ
-- Given that ∠EDQ = ∠BDF
-- BE and AD are perpendicular
-- |DQ| = 2|BD|

noncomputable def triangle_condition (A B C P D E F Q : Point) : Prop :=
  inside_triangle P A B C ∧
  collinear [A, P, D] ∧
  collinear [B, P, E] ∧
  collinear [C, P, F] ∧
  E ∈ ray BE ∧
  angle E D Q = angle B D F ∧
  perp B E A D ∧
  dist D Q = 2 * dist B D

theorem angle_FDE_eq_60 
  (A B C P D E F Q : Point)
  (h : triangle_condition A B C P D E F Q) :
  angle F D E = 60 :=
sorry -- Proof steps to be filled in

end angle_FDE_eq_60_l479_479890


namespace min_mobots_required_to_mow_lawn_l479_479515

theorem min_mobots_required_to_mow_lawn (m n : ℕ) : 
  ∀ (mobots : list ℕ), 
  (∀ i, i < m → mobots i < n → (mobots i) = i) 
  ↔ 
  ∃ k, k = min m n ∧ length mobots = k :=
sorry

end min_mobots_required_to_mow_lawn_l479_479515


namespace smallest_possible_AC_l479_479339

-- Constants and assumptions
variables (AC CD : ℕ)
def BD_squared : ℕ := 68

-- Prime number constraint for CD
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Given facts
axiom eq_ab_ac (AB : ℕ) : AB = AC
axiom perp_bd_ac (BD AC : ℕ) : BD^2 = BD_squared
axiom int_ac_cd : AC = (CD^2 + BD_squared) / (2 * CD)

theorem smallest_possible_AC :
  ∃ AC : ℕ, (∃ CD : ℕ, is_prime CD ∧ CD < 10 ∧ AC = (CD^2 + BD_squared) / (2 * CD)) ∧ AC = 18 :=
by
  sorry

end smallest_possible_AC_l479_479339


namespace distance_to_place_equals_2_point_25_l479_479643

-- Definitions based on conditions
def rowing_speed : ℝ := 4
def river_speed : ℝ := 2
def total_time_hours : ℝ := 1.5

-- Downstream speed = rowing_speed + river_speed
def downstream_speed : ℝ := rowing_speed + river_speed
-- Upstream speed = rowing_speed - river_speed
def upstream_speed : ℝ := rowing_speed - river_speed

-- Define the distance d
def distance (d : ℝ) : Prop :=
  (d / downstream_speed + d / upstream_speed = total_time_hours)

-- The theorem statement
theorem distance_to_place_equals_2_point_25 :
  ∃ d : ℝ, distance d ∧ d = 2.25 :=
by
  sorry

end distance_to_place_equals_2_point_25_l479_479643


namespace modulo_inverse_expression_58_l479_479353

theorem modulo_inverse_expression_58 :
  (4 * 43 + 12 * 37 - 6 * 53) % 60 = 58 :=
by {
  have h1 : 7 * 43 % 60 = 1,
  { sorry },
  have h2 : 13 * 37 % 60 = 1,
  { sorry },
  have h3 : 17 * 53 % 60 = 1,
  { sorry },
  calc
    (4 * 43 + 12 * 37 - 6 * 53) % 60
        = 298 % 60 : by sorry
    ... = 58 : by sorry
}

end modulo_inverse_expression_58_l479_479353


namespace basketball_card_cost_l479_479882

def glasses_cost := 50
def jeans_cost := 100
def mary_discount_rate := 0.10
def mary_total_spending := glasses_cost * 2 + jeans_cost
def jack_sweater_cost := 80
def jack_football_cost := 40
def jack_watch_cost := 65
def jack_tax_rate := 0.08
def rose_shoes_cost := 150

theorem basketball_card_cost :
  let total_spent := mary_total_spending * (1 - mary_discount_rate) in
  let jack_total_spent := (jack_sweater_cost + jack_football_cost + jack_watch_cost) * (1 + jack_tax_rate) in
  ∃ cost_of_one_deck : ℝ,
  let rose_total_spent := rose_shoes_cost + 3 * cost_of_one_deck in
  total_spent = jack_total_spent ∧
  total_spent = rose_total_spent →
  cost_of_one_deck = 10 :=
by
  sorry

end basketball_card_cost_l479_479882


namespace Cheapest_Taxi_l479_479470

noncomputable def Jim_charge : ℝ := 2.25 + 9 * 0.35
noncomputable def Susan_charge : ℝ := 3.00 + 11 * 0.40
noncomputable def John_charge : ℝ := 1.75 + 15 * 0.30

theorem Cheapest_Taxi :
  Jim_charge < Susan_charge ∧ Jim_charge < John_charge := by
sorry

end Cheapest_Taxi_l479_479470


namespace students_excelling_in_physics_and_chemistry_not_math_l479_479449

-- Definitions and known conditions.
variables (B1 B2 B3 : Finset ℕ) -- Sets representing students excelling in each subject.

variables (total : ℕ) 
variables (n1 n2 n3 n12 n13 n123 : ℕ)

-- Assumptions from the problem statement.
axiom h1 : total = 100
axiom h2 : B1.card = 70
axiom h3 : B2.card = 65
axiom h4 : B3.card = 75
axiom h5 : (B1 ∩ B2).card = 40
axiom h6 : (B1 ∩ B3).card = 45
axiom h7 : (B1 ∩ B2 ∩ B3).card = 25

-- Define the subset of students excelling in physics and chemistry but not in mathematics.
def physics_and_chemistry_not_math := B2 ∩ B3 \ B1

-- The proof statement to be proven.
theorem students_excelling_in_physics_and_chemistry_not_math : physics_and_chemistry_not_math.card = 25 :=
by sorry

end students_excelling_in_physics_and_chemistry_not_math_l479_479449


namespace travel_time_approximation_l479_479668

def average_speed_highways : ℝ := 50
def average_speed_urban : ℝ := 30
def rest_stops_count : ℕ := 3
def rest_stop_duration_minutes : ℕ := 45
def total_distance : ℝ := 790
def urban_distance : ℝ := 120

def time_in_urban_areas : ℝ := urban_distance / average_speed_urban
def highway_distance : ℝ := total_distance - urban_distance
def time_on_highways : ℝ := highway_distance / average_speed_highways
def total_rest_stop_time_hours : ℝ := (rest_stops_count * rest_stop_duration_minutes) / 60

def total_travel_time : ℝ := time_in_urban_areas + time_on_highways + total_rest_stop_time_hours

theorem travel_time_approximation : abs(total_travel_time - 19.65) < 0.1 := by
  sorry

end travel_time_approximation_l479_479668


namespace ferry_tourists_total_l479_479298

theorem ferry_tourists_total 
  (n : ℕ)
  (a d : ℕ)
  (sum_arithmetic_series : ℕ → ℕ → ℕ → ℕ)
  (trip_count : n = 5)
  (first_term : a = 85)
  (common_difference : d = 3) :
  sum_arithmetic_series n a d = 455 :=
by
  sorry

end ferry_tourists_total_l479_479298


namespace quad_intersects_x_axis_l479_479072

theorem quad_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x ^ 2 - 7 * x - 7 = 0) ↔ (k ≥ -7 / 4 ∧ k ≠ 0) :=
by sorry

end quad_intersects_x_axis_l479_479072


namespace max_value_b_squared_over_a_sq_plus_c_sq_l479_479738

theorem max_value_b_squared_over_a_sq_plus_c_sq
  (a b c : ℝ) (h1 : a > 0)
  (h2 : ∀ x : ℝ, a * x^2 + (b - 2 * a) * x + (c - b) ≥ 0) :
  ∃ k : ℝ, k = 2 * sqrt 2 - 2 ∧ 
           ∀ (a b c : ℝ), (a > 0) → 
           (∀ x : ℝ, a * x^2 + (b - 2 * a) * x + (c - b) ≥ 0) → 
           (b^2 / (a^2 + c^2) ≤ k) := 
sorry

end max_value_b_squared_over_a_sq_plus_c_sq_l479_479738


namespace people_eat_only_vegetarian_l479_479852

theorem people_eat_only_vegetarian (non_veg_only : ℕ) (both : ℕ) (total_veg : ℕ) : 
  non_veg_only = 8 → both = 6 → total_veg = 19 → total_veg - both = 13 :=
by {
  intros h_non_veg_only h_both h_total_veg,
  rw [h_both, h_total_veg],
  norm_num,
  }

end people_eat_only_vegetarian_l479_479852


namespace LCM_14_21_l479_479270

theorem LCM_14_21 : Nat.lcm 14 21 = 42 := 
by
  sorry

end LCM_14_21_l479_479270


namespace problem_correct_propositions_proof_l479_479747

noncomputable def number_of_correct_propositions (l m n : Line) (α β : Plane) :=
  let p1 := ¬ (α ⟂ β ∧ l ⟂ α → l ∥ β)
  let p2 := ¬ (α ⟂ β ∧ l ⊆ α → l ⟂ β)
  let p3 := ¬ (l ⟂ m ∧ m ⟂ n → l ∥ n)
  let p4 := m ⟂ α ∧ α ∥ β ∧ n ∥ β → m ⟂ n
  cond := lines_and_planes_are_different l m n α β
  p1 + p2 + p3 + p4 = 1

axiom lines_and_planes_are_different (l m n : Line) (α β : Plane) : l ≠ m ∧ l ≠ n ∧ m ≠ n ∧ α ≠ β

theorem problem_correct_propositions_proof (l m n : Line) (α β : Plane) (h : lines_and_planes_are_different l m n α β) :
  number_of_correct_propositions l m n α β = 1 :=
sorry

end problem_correct_propositions_proof_l479_479747


namespace triangle_sides_l479_479878

theorem triangle_sides (a d : ℝ) (BC AE BD : ℝ) (h1 : BC = a)
  (h2 : AE^2 + BD^2 = d^2)
  (h3 : ∃ AE BD x y : ℝ, ∃ O : ℝ, AO = 2x ∧ BO = 2y ∧ AE = 3x ∧ BD = 3y ∧ x^2 + y^2 = d^2 / 9
      ∧ 4x^2 + 4y^2 = (2d / 3)^2 ∧ 4x^2 + y^2 = (sqrt(20d^2 / 9 - a^2))^2) : 
  ∃ AB AC : ℝ, AB = 2d / 3 ∧ AC = sqrt(20d^2 / 9 - a^2) :=
by
  sorry

end triangle_sides_l479_479878


namespace problem_condition_seq_is_arithmetic_and_sum_is_bound_l479_479781

theorem problem_condition_seq_is_arithmetic_and_sum_is_bound (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (a 1 = 1) →
  (∀ n, S n = ∑ i in range n, a (i + 1)) →
  (∀ n, a n = 2 * (S n) ^ 2 / (2 * S n - 1)) →
  (∀ n, 1 / S n = 1 + 2 * (n - 1)) →
  S 1 + (∑ k in (range n), 1 / (k + 1) * S (k + 1)) < 3 / 2 :=
by
  intros h1 hs ha harith
  sorry

end problem_condition_seq_is_arithmetic_and_sum_is_bound_l479_479781


namespace find_ratio_of_intersections_l479_479193

-- Definition of the geometrical setup and proof statement
theorem find_ratio_of_intersections
  (A B C D A1 B1 C1 D1 P : Type)
  [tetrahedron : euclidean_space.tetrahedron A B C D]
  (cond1 : intersect_line_plane A A1 (face B C D) = true)
  (cond2 : intersect_line_plane B B1 (face A C D) = true)
  (cond3 : intersect_line_plane C C1 (face A B D) = true)
  (cond4 : intersect_line_plane D D1 (face A B C) = true)
  (interP : is_intersection_point [line A A1, line B B1, line C C1, line D D1] P)
  (ratio_eq : ∀ r, (dist A P / dist A1 P = r) ∧ 
                 (dist B P / dist B1 P = r) ∧ 
                 (dist C P / dist C1 P = r) ∧ 
                 (dist D P / dist D1 P = r)) :
  r = 3 :=
by
  -- This is only the problem statement, proof is not required
  sorry

end find_ratio_of_intersections_l479_479193


namespace BC_length_l479_479870

theorem BC_length (A B C P Q R S : EuclideanGeometry) (h1 : square AQPB) (h2 : square ASRC) (h3 : equilateral_triangle AQS) (h4 : length QS = 4) : length BC = 4 :=
sorry

end BC_length_l479_479870


namespace count_zero_expressions_l479_479409

/-- Given four specific vector expressions, prove that exactly two of them evaluate to the zero vector. --/
theorem count_zero_expressions
(AB BC CA MB BO OM AC BD CD OA OC CO : ℝ × ℝ)
(H1 : AB + BC + CA = 0)
(H2 : AB + (MB + BO + OM) ≠ 0)
(H3 : AB - AC + BD - CD = 0)
(H4 : OA + OC + BO + CO ≠ 0) :
  (∃ count, count = 2 ∧
      ((AB + BC + CA = 0) → count = count + 1) ∧
      ((AB + (MB + BO + OM) = 0) → count = count + 1) ∧
      ((AB - AC + BD - CD = 0) → count = count + 1) ∧
      ((OA + OC + BO + CO = 0) → count = count + 1)) :=
sorry

end count_zero_expressions_l479_479409


namespace satisfy_fn_l479_479808

noncomputable def f : ℤ → ℤ
| n => if n = 6 then 1 else f(n-1) - n

theorem satisfy_fn (n : ℤ) (h1: f(6) = 1) (h2: f(n) = 4) : f(n-1) = 4 + n :=
sorry

end satisfy_fn_l479_479808


namespace smallest_perfect_square_sum_l479_479077

theorem smallest_perfect_square_sum (a b : ℕ) (ha : nat.factors_count a = 15) (hb : nat.factors_count b = 20) (hab : ∃ n : ℕ, a + b = n ^ 2) : a + b = 576 :=
sorry

end smallest_perfect_square_sum_l479_479077


namespace find_room_dimension_l479_479233

/-- Variables for the room dimensions and constant values. -/
variables (x : ℕ)

/-- Given conditions from the problem. -/
def height := 12 -- feet
def width := 25 -- feet
def door_area := 6 * 3 -- square feet
def window_area := 4 * 3 -- square feet
def num_windows := 3
def cost_per_sqft := 5 -- Rs. per square feet
def total_cost := 4530 -- Rs.

/-- Calculate the total area of the walls to be whitewashed. -/
def walls_area (x : ℕ) : ℕ :=
  2 * (width * height) + 2 * (x * height) - door_area - num_windows * window_area

/-- The total cost of whitewashing the walls. -/
def calculate_cost (x : ℕ) : ℕ :=
  cost_per_sqft * (walls_area x)

/-- Main theorem to prove the unknown dimension of the room. -/
theorem find_room_dimension : x = 15 :=
by
  have h1 : calculate_cost x = total_cost := sorry
  exact sorry

end find_room_dimension_l479_479233


namespace simple_interest_rate_l479_479280

theorem simple_interest_rate (P : ℝ) (SI : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : SI = P / 5)
  (h2 : SI = P * R * T / 100)
  (h3 : T = 7) : 
  R = 20 / 7 :=
by 
  sorry

end simple_interest_rate_l479_479280


namespace maximize_net_income_l479_479635

noncomputable def net_income (x : ℕ) : ℤ :=
  if 60 ≤ x ∧ x ≤ 90 then 750 * x - 1700
  else if 90 < x ∧ x ≤ 300 then -3 * x * x + 1020 * x - 1700
  else 0

theorem maximize_net_income :
  (∀ x : ℕ, 60 ≤ x ∧ x ≤ 300 →
    net_income x ≤ net_income 170) ∧
  net_income 170 = 85000 := 
sorry

end maximize_net_income_l479_479635


namespace r_squared_minus_p_squared_l479_479110

theorem r_squared_minus_p_squared (p r : ℝ) (α β : ℝ) :
  (tan α = root (polynomial.C (-1) + polynomial.C p^2 * X^(2: ℕ)) (2*p*X)) → 
  (tan β = root (polynomial.C (-1) + polynomial.C p^2 * X^(2: ℕ)) (2*p*X)) →
  (cot α = root (polynomial.C (-1) + polynomial.C r^2 * X^(2: ℕ)) (2*r*X)) → 
  (cot β = root (polynomial.C (-1) + polynomial.C r^2 * X^(2: ℕ)) (2*r*X)) → 
  r^2 - p^2 = (2 * p^2 - p^4) / (p^2 - 1) :=
by 
  sorry

end r_squared_minus_p_squared_l479_479110


namespace min_value_ab_l479_479059

theorem min_value_ab (b : ℝ) (h : b > 0) (perp : (b^2 + 1) * (-b^(-2)) = -a) : ab = 2 :=
by
  -- Proof goes here
  sorry

end min_value_ab_l479_479059


namespace thirteen_fifth_mod_seven_l479_479212

theorem thirteen_fifth_mod_seven : 
  ∃ m : ℕ, 13^5 ≡ m [MOD 7] ∧ 0 ≤ m ∧ m < 7 ∧ m = 0 :=
by
  existsi 0
  split
  { sorry } -- Proof that 13^5 ≡ 0 [MOD 7]
  split
  { exact nat.zero_le 0 } -- Proof that 0 ≤ 0
  split
  { exact nat.zero_lt_succ 6 } -- Proof that 0 < 7 (i.e., 0 < 7)
  { refl } -- Proof that m = 0

end thirteen_fifth_mod_seven_l479_479212


namespace find_f_prime_at_1_l479_479816

noncomputable def f (x : ℝ) : ℝ := x^3 - f' 1 * x^2 + 3

theorem find_f_prime_at_1 : deriv f 1 = 1 := by
  sorry

end find_f_prime_at_1_l479_479816


namespace mathemetics_combinations_l479_479153

theorem mathemetics_combinations : 
  let vowels := ['A', 'E', 'A', 'I']
  let consonants := ['M', 'T', 'H', 'M', 'T', 'C', 'S'] 
  let total_vowels := vowels.length
  let total_consonants := consonants.length
  (total_vowels = 4) → 
  (total_consonants = 7) → 
  (∀ x ∈ consonants, x = 'T' → count_in_list consonants 'T' = 2) →
  let vowels_choices := choose total_vowels 3
  let consonant_case1 := (choose 2 1 + choose 2 0) * (choose 5 3 + choose 5 4)
  let consonant_case2 := choose 5 2
  let total_cases := vowels_choices * (consonant_case1 + consonant_case2)
  total_cases = 220 := 
sorry

end mathemetics_combinations_l479_479153


namespace percentile_80th_is_10_8_l479_479049

noncomputable def data_set : list ℝ := [8.6, 8.9, 9.1, 9.6, 9.7, 9.8, 9.9, 10.2, 10.6, 10.8, 11.2, 11.7]

noncomputable def percentile (p : ℝ) (l : list ℝ) : ℝ :=
  let sorted := list.sort (≤) l
  let n := list.length l
  let pos := (n : ℝ) * p
  if pos.fract = 0 then
    list.nth_le sorted (pos.to_nat - 1) (by simp only [sorted, n, list.length_pos_of_mem (list.mem_cons_self (list.nth_le sorted 0 sorry) sorted)])
  else
    list.nth_le sorted pos.ceil.to_nat.pred (by simp only [sorted, n, list.length_pos_of_mem (list.mem_cons_self (list.nth_le sorted 0 sorry) sorted)])

theorem percentile_80th_is_10_8 : percentile 0.8 data_set = 10.8 := by
  sorry

end percentile_80th_is_10_8_l479_479049


namespace geometry_problem_l479_479387

variables {Point : Type}
variables (A B C D P Q : Point)
variables [MetricSpace Point]
variables [HasLattice OrderOfMetricSpace.FilterOrder]

-- Assuming points A, B, C, D form a unit square
def unit_square (A B C D : Point) :=
  dist A B = 1 ∧ dist B C = 1 ∧ dist C D = 1 ∧ dist D A = 1 ∧
  dist A C = sqrt 2 ∧ dist B D = sqrt 2

-- Assuming points P and Q are inside the unit square ABCD
def points_in_square (P Q A B C D : Point) :=
  dist P A + dist P B <= 1 ∧ dist P B + dist P C <= 1 ∧ dist P C + dist P D <= 1 ∧ dist P D + dist P A <= 1 ∧
  dist Q A + dist Q B <= 1 ∧ dist Q B + dist Q C <= 1 ∧ dist Q C + dist Q D <= 1 ∧ dist Q D + dist Q A <= 1

theorem geometry_problem (h1 : unit_square A B C D)
                         (h2 : points_in_square P Q A B C D) :
  13 * (dist P A + dist Q C) + 14 * dist P Q + 15 * (dist P B + dist Q D) > 38 :=
sorry

end geometry_problem_l479_479387


namespace max_curved_sides_l479_479540

theorem max_curved_sides (n : ℕ) (h : 2 ≤ n) : 
  ∃ m, m = 2 * n - 2 :=
sorry

end max_curved_sides_l479_479540


namespace minimum_value_proof_l479_479177

noncomputable def minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem minimum_value_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  minimum_value x y z hx hy hz h ≥ 4 :=
sorry

end minimum_value_proof_l479_479177


namespace hyperbola_eccentricity_range_l479_479778

noncomputable def hyperbola := {x y a b : ℝ // x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0}
noncomputable def parabola := {y x a : ℝ // y^2 = -2 * a * x}

theorem hyperbola_eccentricity_range (a b : ℝ) (C1 : hyperbola) (C2 : parabola) :
  ∃ (e : ℝ), (1 < e ∧ e ≤ 3 * real.sqrt 2 / 4) :=
sorry

end hyperbola_eccentricity_range_l479_479778


namespace dominic_average_speed_l479_479008

noncomputable def average_speed (total_distance total_time : ℚ) : ℚ :=
  total_distance / total_time

theorem dominic_average_speed :
  let total_distance := (20 : ℚ) + (40 : ℚ) + (30 : ℚ) + (94 : ℚ) in
  let time_to_post_office := (2 : ℚ) in
  let time_to_grocery_store := (3 : ℚ) - time_to_post_office in
  let time_to_friend := (5 : ℚ) - (3 : ℚ) in
  let time_from_friend_to_shop := (94 : ℚ) / (45 : ℚ) in
  let total_stop_time := (3 * 0.5 : ℚ) in
  let total_time := time_to_post_office + time_to_grocery_store + time_to_friend + total_stop_time + time_from_friend_to_shop in
  average_speed total_distance total_time ≈ (21.42 : ℚ) := 
sorry

#eval dominic_average_speed

end dominic_average_speed_l479_479008


namespace day_of_week_366th_day_2004_l479_479119

theorem day_of_week_366th_day_2004 :
  ∀ (day_of_week: ℕ),   -- day_of_week: 0 (Sunday), 1 (Monday), ..., 6 (Saturday)
  ((45 % 7 = 2) ∧ -- 45th day of 2004 is Tuesday, so 2 (Tuesday)
  true) → 
  (366 % 7 = 2) :=   -- 366th day of 2004 should be Monday
  
begin
  sorry
end

end day_of_week_366th_day_2004_l479_479119


namespace min_dist_AB_is_2_l479_479460

-- Define curves in polar coordinates
def curve_C1 (ρ θ : ℝ) : Prop := ρ = 2 * sin (θ + π / 3)
def curve_C2 (ρ θ : ℝ) : Prop := ρ * sin (θ + π / 3) = 4

-- Definition of points A and B on respective curves
def A_on_C1 (A : ℝ × ℝ) : Prop := ∃ θ, A = (2 * sin (θ + π / 3) * cos θ, 2 * sin (θ + π / 3) * sin θ)
def B_on_C2 (B : ℝ × ℝ) : Prop := ∃ θ, B = (4 / sin (θ + π / 3) * cos θ, 4 / sin (θ + π / 3) * sin θ)

-- Minimum distance formula in rectangular coordinates
noncomputable def dist (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Prove the minimum distance between points A on C1 and B on C2 is 2
theorem min_dist_AB_is_2 : ∃ A B, A_on_C1 A ∧ B_on_C2 B ∧ dist A B = 2 := 
sorry

end min_dist_AB_is_2_l479_479460


namespace positive_number_property_l479_479309

theorem positive_number_property (x : ℝ) (h_pos : 0 < x) (h_property : (x^2 / 100) = 9) : x = 30 :=
by
  sorry

end positive_number_property_l479_479309


namespace fraction_of_quarters_1840_1849_equals_4_over_15_l479_479662

noncomputable def fraction_of_states_from_1840s (total_states : ℕ) (states_from_1840s : ℕ) : ℚ := 
  states_from_1840s / total_states

theorem fraction_of_quarters_1840_1849_equals_4_over_15 :
  fraction_of_states_from_1840s 30 8 = 4 / 15 := 
by
  sorry

end fraction_of_quarters_1840_1849_equals_4_over_15_l479_479662


namespace geometric_sequence_and_general_term_l479_479017

noncomputable def a : ℕ → ℝ
noncomputable def S : ℕ → ℝ
noncomputable def b (n : ℕ) : ℝ := a (n+1) - 2 * a n

axiom S_condition (n : ℕ) : S (n+1) = 4 * a n + 2
axiom a_initial : a 1 = 1

theorem geometric_sequence_and_general_term :
  (∀ n : ℕ, b (n + 1) = 2 * b n) ∧ (b 1 = 3) ∧ (∀ n : ℕ, b (n + 1) = 3 * 2^n) :=
sorry

end geometric_sequence_and_general_term_l479_479017


namespace price_per_pound_of_fruits_and_vegetables_is_4_l479_479469

theorem price_per_pound_of_fruits_and_vegetables_is_4 (wage_per_hour min_wage meat_price_per_pound bread_price_per_pound
                                                       janitorial_hours janitorial_wage hours_worked
                                                       wasted_meat_weight wasted_bread_weight wasted_fruits_vegetables_weight:
    ℝ) 
  (h_minimum_wage: min_wage = 8) 
  (h_meat_price: meat_price_per_pound = 5) 
  (h_bread_price: bread_price_per_pound = 1.5) 
  (h_janitorial_hours: janitorial_hours = 10) 
  (h_janitorial_wage: janitorial_wage = 10) 
  (h_hours_worked: hours_worked = 50)
  (h_wasted_meat_weight: wasted_meat_weight = 20) 
  (h_wasted_bread_weight: wasted_bread_weight = 60) 
  (h_wasted_fruits_vegetables_weight: wasted_fruits_vegetables_weight = 15) : 
  (price_per_pound : ℝ) 
  (h_price: price_per_pound = 4) := 
by
  -- skpping the proof
  sorry

end price_per_pound_of_fruits_and_vegetables_is_4_l479_479469


namespace total_slices_is_78_l479_479674

-- Definitions based on conditions
def ratio_buzz_waiter (x : ℕ) : Prop := (5 * x) + (8 * x) = 78
def waiter_condition (x : ℕ) : Prop := (8 * x) - 20 = 28

-- Prove that the total number of slices is 78 given conditions
theorem total_slices_is_78 (x : ℕ) (h1 : ratio_buzz_waiter x) (h2 : waiter_condition x) : (5 * x) + (8 * x) = 78 :=
by
  sorry

end total_slices_is_78_l479_479674


namespace monotonicity_f_range_of_k_existence_of_m_l479_479084

-- Definitions required based on conditions
def f (x : ℝ) : ℝ := x * Real.log x

-- (1) Monotonicity of f
theorem monotonicity_f : 
    (∀ x, 0 < x ∧ x < 1 / Real.exp(1) → f' x < 0) ∧
    (∀ x, x > 1 / Real.exp(1) → f' x > 0) := sorry

-- (2) Range of k
theorem range_of_k (x : ℝ) (hx : 0 < x) :
    ∃ k, ∀ x > 0, f x > k * x - 1 / 2 ∧ k ≤ 1 - Real.log 2 := sorry

-- (3) Existence of smallest positive constant m
theorem existence_of_m :
    ∃ (m : ℝ), 0 < m ∧ (∀ a > m, ∀ x > 0, f(a + x) < f(a) * Real.exp x) := sorry

end monotonicity_f_range_of_k_existence_of_m_l479_479084


namespace probability_of_sum_17_of_three_8_faced_dice_l479_479128

theorem probability_of_sum_17_of_three_8_faced_dice : 
  let d1 := [1, 2, 3, 4, 5, 6, 7, 8],
      d2 := [1, 2, 3, 4, 5, 6, 7, 8],
      d3 := [1, 2, 3, 4, 5, 6, 7, 8] in
  (∃ sum = 17, (∑ (i, j, k) ∈ (list.product d1 (list.product d2 d3)), 
               if i + j + k = sum then 1 else 0) / (list.length d1 * list.length d2 * list.length d3) = 27 / 512) := sorry

end probability_of_sum_17_of_three_8_faced_dice_l479_479128


namespace mabel_tomatoes_l479_479184

theorem mabel_tomatoes (x : ℕ)
  (plant_1_bore : ℕ)
  (plant_2_bore : ℕ := x + 4)
  (total_first_two_plants : ℕ := x + plant_2_bore)
  (plant_3_bore : ℕ := 3 * total_first_two_plants)
  (plant_4_bore : ℕ := 3 * total_first_two_plants)
  (total_tomatoes : ℕ)
  (h1 : total_first_two_plants = 2 * x + 4)
  (h2 : plant_3_bore = 3 * (2 * x + 4))
  (h3 : plant_4_bore = 3 * (2 * x + 4))
  (h4 : total_tomatoes = x + plant_2_bore + plant_3_bore + plant_4_bore)
  (h5 : total_tomatoes = 140) :
   x = 8 :=
by
  sorry

end mabel_tomatoes_l479_479184


namespace find_k_such_that_f_is_odd_l479_479418

def f (k : ℝ) (x : ℝ) : ℝ := (k - 2^x) / (1 + k * 2^x)

theorem find_k_such_that_f_is_odd (k : ℝ) :
  (∀ x : ℝ, f k (-x) = -f k x) ↔ (k = 1 ∨ k = -1) := by
  sorry

end find_k_such_that_f_is_odd_l479_479418


namespace rectangle_perimeters_l479_479196

theorem rectangle_perimeters (w h : ℝ) 
  (h1 : 2 * (w + h) = 20)
  (h2 : 2 * (4 * w + h) = 56) : 
  4 * (w + h) = 40 ∧ 2 * (w + 4 * h) = 44 := 
by
  sorry

end rectangle_perimeters_l479_479196


namespace points_are_concyclic_l479_479238

-- Initial Definitions and Assumptions
variables {A B C : Type} [inner_product_space ℝ A]
variables {X Y Z X' Y' P Q : A}
variables (Ω : circle) (ω : circle)
variables (I : A)

-- Given conditions
def incircle_touches_sides (ω : circle) (ABC : triangle) (X Y Z : A) : Prop := sorry
def feet_of_altitudes (XYZ : triangle) (X' Y' : A) : Prop := sorry
def line_intersects_circumcircle (X' Y' A B C P Q : A) (Ω : circle) : Prop := sorry

-- Main theorem
theorem points_are_concyclic
  (h1 : incircle_touches_sides ω ⟨A, B, C⟩ X Y Z)
  (h2 : feet_of_altitudes ⟨X, Y, Z⟩ X' Y')
  (h3 : line_intersects_circumcircle X' Y' A B C P Q Ω) :
  cyclic_quadrilateral X Y P Q :=
sorry

end points_are_concyclic_l479_479238


namespace part_a_l479_479632

theorem part_a (A B C D P : Point) (h1 : PointsAreColinear A D B)
               (h2 : PointsAreColinear A D C) (h3 : DistanceEqual A B C D) :
  PA + PD ≥ PB + PC := sorry

end part_a_l479_479632


namespace ellipse_problem_l479_479405

-- Condition for the ellipse equation
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧ (y^2 / a^2 + x^2 / b^2 = 1)

-- Condition for the eccentricity
def eccentricity (a b c : ℝ) : Prop :=
  c / a = sqrt 3 / 2

-- Condition for the area of Δ MNF_2
def area_of_triangle (a b c : ℝ) : Prop :=
  2 * b^2 * c / a = sqrt 3

-- The correct answer for the standard equation of the ellipse
def standard_equation_of_ellipse : Prop :=
  ∀ x y, (a > b) ∧ (b > 0) → (x^2 + y^2 / 4 = 1)

-- The correct answer for the range of m
def range_of_m (m : ℝ) : Prop :=
  m ∈ set.Ioo (-2) (-1) ∪ set.Ioo 1 2 ∪ {0}

-- Main theorem statement
theorem ellipse_problem (a b c : ℝ) (x y m : ℝ) :
  (a > b ∧ b > 0) →
  eccentricity a b c →
  area_of_triangle a b c →
  (∃ f : {x // x = (standard_equation_of_ellipse x y)}) ∧
  (∀ m, range_of_m m) :=
by sorry

end ellipse_problem_l479_479405


namespace true_weight_of_C_l479_479254

theorem true_weight_of_C (A1 B1 C1 A2 B2 : ℝ) (l1 l2 m1 m2 A B C : ℝ)
  (hA1 : (A + m1) * l1 = (A1 + m2) * l2)
  (hB1 : (B + m1) * l1 = (B1 + m2) * l2)
  (hC1 : (C + m1) * l1 = (C1 + m2) * l2)
  (hA2 : (A2 + m1) * l1 = (A + m2) * l2)
  (hB2 : (B2 + m1) * l1 = (B + m2) * l2) :
  C = (C1 - A1) * Real.sqrt ((A2 - B2) / (A1 - B1)) + 
      (A1 * Real.sqrt (A2 - B2) + A2 * Real.sqrt (A1 - B1)) / 
      (Real.sqrt (A1 - B1) + Real.sqrt (A2 - B2)) :=
sorry

end true_weight_of_C_l479_479254


namespace number_of_children_l479_479288

theorem number_of_children (n m : ℕ) (h1 : 11 * (m + 6) + n * m = n^2 + 3 * n - 2) : n = 9 :=
sorry

end number_of_children_l479_479288


namespace solve_triangle_given_median_and_angles_l479_479210

variables (α β k3 : ℝ) -- Define variables for given angles α, β and median k3
variables (a b c γ₁ γ₂ : ℝ) -- Define variables for the unknown sides a, b, c and angles γ₁, γ₂

-- Define the problem statement
theorem solve_triangle_given_median_and_angles :
  (k3 = sqrt (2 * b^2 + 2 * c^2 - a^2) / 2) →
  (tan ((γ₁ - γ₂) / 2) = (tan ((α - β) / 2) / tan ((α + β) / 2)) * cot ((α + β) / 2)) →
  ∃ (a b c : ℝ), true :=
by
  intro h1 h2
  -- Further steps and definitions are omitted
  -- The proof will use h1 and h2 to show the existence of a, b, and c satisfying the above conditions.
  sorry

end solve_triangle_given_median_and_angles_l479_479210


namespace percentage_increase_in_y_is_160_percent_l479_479940

-- Given conditions
variables (x y c : ℝ) (h1 : y = c * x)

-- x increases by 30%
def x_prime := 1.3 * x

-- New value of y given x' and proportionality constant c
def y_prime := c * x_prime

-- Given that the new value of y is 260% of its original value
def y_prime_given : ℝ := 2.6 * y

-- Prove that the percentage increase in y is 160%
theorem percentage_increase_in_y_is_160_percent :
  y_prime = y_prime_given → (y_prime - y) / y = 1.6 := by
sorry

end percentage_increase_in_y_is_160_percent_l479_479940


namespace cyclic_quadrilateral_tangent_circle_sum_eq_l479_479637

theorem cyclic_quadrilateral_tangent_circle_sum_eq (A B C D : Point) (ω : Circle) (hAB : ω.center ∈ segment A B)
  (hBC : tangent_to_circle ω (segment B C))
  (hCD : tangent_to_circle ω (segment C D))
  (hDA : tangent_to_circle ω (segment D A))
  (h_cyclic : is_cyclic_quadrilateral A B C D) :
  length (segment A D) + length (segment B C) = length (segment A B) := 
sorry

end cyclic_quadrilateral_tangent_circle_sum_eq_l479_479637


namespace beach_trip_time_l479_479883

noncomputable def totalTripTime (driveTime eachWay : ℝ) (beachTimeFactor : ℝ) : ℝ :=
  let totalDriveTime := eachWay * 2
  totalDriveTime + (totalDriveTime * beachTimeFactor)

theorem beach_trip_time :
  totalTripTime 2 2 2.5 = 14 := 
by
  sorry

end beach_trip_time_l479_479883


namespace Q_ratio_l479_479910

-- Define the given polynomial f
def f (x : ℝ) : ℝ := x^2005 + 13 * x^2004 + 2

-- Assume distinct zeroes
axiom has_distinct_zeroes : ∃ s : (Fin 2005) → ℝ, ∀ i ≠ j, s i ≠ s j ∧ f (s i) = 0

-- Define the polynomial Q such that Q(s_j + 2/s_j) = 0 for each root s_j of f
def Q (z : ℝ) : ℝ := ∏ i in Finset.range 2005, (z - ((s i) + 2 / (s i)))

theorem Q_ratio : Q 1 / Q (-1) = 1 := by
  sorry

end Q_ratio_l479_479910


namespace f_even_function_g_range_m_inequality_l479_479410

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(-x)
noncomputable def g (a : ℝ) : ℝ → ℝ := λ x, 2^(2 * x) + 2^(-2 * x) - 2 * a * f x
noncomputable def h (m x : ℝ) : Prop := m * f x ≤ 2^(-x) + m - 1

theorem f_even_function : ∀ x : ℝ, f (-x) = f x := by
  sorry

theorem g_range (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x → (2 - 4 * a ≤ g a x ∧ a ≤ 2) 
  ∨ (-a^2 - 2 ≤ g a x ∧ 2 < a)) := by
  sorry

theorem m_inequality (m : ℝ) : 
  (∀ x : ℝ, 0 < x → h m x) → m ≤ -1/3 := by
  sorry

end f_even_function_g_range_m_inequality_l479_479410


namespace minimum_value_of_expression_l479_479378

theorem minimum_value_of_expression
  (a b c : ℝ)
  (h : 2 * a + 2 * b + c = 8) :
  ∃ x, (x = (a - 1)^2 + (b + 2)^2 + (c - 3)^2) ∧ x ≥ (49 / 9) :=
sorry

end minimum_value_of_expression_l479_479378


namespace find_parcera_triples_l479_479320

noncomputable def is_prime (n : ℕ) : Prop := sorry
noncomputable def parcera_triple (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧
  p ∣ q^2 - 4 ∧ q ∣ r^2 - 4 ∧ r ∣ p^2 - 4

theorem find_parcera_triples : 
  {t : ℕ × ℕ × ℕ | parcera_triple t.1 t.2.1 t.2.2} = 
  {(2, 2, 2), (5, 3, 7), (7, 5, 3), (3, 7, 5)} :=
sorry

end find_parcera_triples_l479_479320


namespace continuous_zero_point_condition_l479_479067

theorem continuous_zero_point_condition (f : ℝ → ℝ) {a b : ℝ} (h_cont : ContinuousOn f (Set.Icc a b)) :
  (f a * f b < 0) → (∃ c ∈ Set.Ioo a b, f c = 0) ∧ ¬ (∃ c ∈ Set.Ioo a b, f c = 0 → f a * f b < 0) :=
sorry

end continuous_zero_point_condition_l479_479067


namespace sequence_invariant_eq_l479_479783

theorem sequence_invariant_eq (a : ℝ) (h : a > 0)
  (h_seq : ∀ n : ℕ, a_{n+1} = (a_n / 2) + (1 / a_n)) :
  ∃ m : ℕ, m > 0 ∧ ∀ n : ℤ, a_{m + n} = a_n :=
sorry

end sequence_invariant_eq_l479_479783


namespace money_left_after_bike_purchase_l479_479158

-- Definitions based on conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def quarter_value : ℝ := 0.25
def bike_cost : ℝ := 180

-- The theorem statement
theorem money_left_after_bike_purchase : (jars * quarters_per_jar * quarter_value) - bike_cost = 20 := by
  sorry

end money_left_after_bike_purchase_l479_479158


namespace corrected_mean_l479_479547

theorem corrected_mean (n : ℕ) (original_mean incorrect_correct : ℕ -> ℕ) (incorrect : ℕ) (correct : ℕ) :
  n = 50 ->
  original_mean n = 36 ->
  incorrect = 23 ->
  correct = 45 ->
  incorrect_correct = λ incorrect correct, (original_mean n * n - incorrect + correct) / n ->
  incorrect_correct incorrect correct = 36.44 :=
by
  intros n_eq original_mean_eq incorrect_eq correct_eq incorrect_correct_def
  sorry

end corrected_mean_l479_479547


namespace find_m_l479_479719

noncomputable def g (x : ℝ) := Real.cot (x / 3) - Real.cot x

theorem find_m : ∃ m : ℝ, ∀ x : ℝ, g x = (sin (m * x)) / (sin (x / 3) * sin x) ↔ m = 2 / 3 :=
begin
  sorry
end

end find_m_l479_479719


namespace tripod_new_height_l479_479350

noncomputable def tripod_height (a b L L' : ℝ) : ℝ :=
  let total_height : ℝ := a - b
  in total_height

theorem tripod_new_height :
  (L : ℝ) = 5 ∧ 
  (h : ℝ) = 4 ∧ 
  (L' : ℝ) = 4 ∧ 
  (a^2 + b^2 = L^2) ∧ 
  (a^2 + bʹ^2 = L'^2) ∧ 
  angles_equal_fixed (legs tripod) (angles tripod) ∧ 
  height_initial (tripod) = h
  → 
  height_new_leg_cut tripod = (144 / (√1585)) := 
sorry

end tripod_new_height_l479_479350


namespace radius_of_circle_from_spherical_coords_l479_479560

theorem radius_of_circle_from_spherical_coords :
  ∀ (θ: ℝ), let ρ := 1, φ := π / 3 in
  (√(ρ * sin φ * cos θ)^2 + (ρ * sin φ * sin θ)^2) = √3 / 2 :=
by
  intros θ
  let ρ := 1
  let φ := π / 3
  sorry

end radius_of_circle_from_spherical_coords_l479_479560


namespace rotation_matrix_det_75_degrees_l479_479478

open Matrix Real

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![cos θ, -sin θ; sin θ, cos θ]

theorem rotation_matrix_det_75_degrees :
  let S := rotation_matrix (75 * π / 180) in det S = 1 :=
by
  let θ := (75 * π / 180) -- defining 75 degrees in radians
  let S := rotation_matrix θ
  have hS : S = !![cos θ, -sin θ; sin θ, cos θ] := rfl
  have hdet : det S = cos θ * cos θ + sin θ * sin θ := Matrix.det_fin_two
  have pythagorean_identity : cos θ * cos θ + sin θ * sin θ = 1 := sorry
  rw [hdet, pythagorean_identity]
  exact rfl

end rotation_matrix_det_75_degrees_l479_479478


namespace smallest_positive_multiple_of_45_l479_479609

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ (∀ y : ℕ, y > 0 → 45 * y ≥ 45 * x) ∧ 45 * x = 45 :=
by
  use 1
  split
  · apply Nat.one_pos
  · split
    · intros y hy
      apply mul_le_mul
      · apply Nat.one_le_of_lt hy
      · apply le_refl
      · apply le_of_lt
        apply Nat.pos_of_ne_zero
        intro h
        exact hy (h.symm ▸ rfl)
      · apply le_of_lt
        apply Nat.pos_of_ne_zero
        intro h
        exact hy (h.symm ▸ rfl)
    · apply rfl
  sorry

end smallest_positive_multiple_of_45_l479_479609


namespace shortest_wire_length_l479_479591

noncomputable def wire_length (d_small d_large h_offset : ℝ) : ℝ :=
  let r_small := d_small / 2
  let r_large := d_large / 2
  let vertical_offset := h_offset + r_small
  let horizontal_separation := r_large - r_small
  let straight_section := 2 * real.sqrt (horizontal_separation^2 + vertical_offset^2)
  let angle_small := real.arctan (vertical_offset / horizontal_separation)
  let angle_large := 2 * angle_small
  let arc_small := (angle_large / (2 * real.pi)) * (2 * real.pi * r_small)
  let arc_large := (angle_large / (2 * real.pi)) * (2 * real.pi * r_large)
  straight_section + arc_small + arc_large

theorem shortest_wire_length :
  wire_length 5 20 4 = 43.089 := sorry

end shortest_wire_length_l479_479591


namespace tangency_divides_ratios_l479_479202

open EuclideanGeometry

variables {A B C D E1 E2 E3 E4 : Point}

def is_cyclic_quadrilateral (quad : Quadrilateral) : Prop :=
  cyclic quad

def is_tangential_quadrilateral (quad : Quadrilateral) : Prop :=
  tangential quad

noncomputable def incircle_tangent_points (quad : Quadrilateral) : Point × Point × Point × Point :=
  let O := incircle_center quad
  (tangent_point O (side quad AB), tangent_point O (side quad BC), tangent_point O (side quad CD), tangent_point O (side quad DA))

theorem tangency_divides_ratios {Q : Quadrilateral} (h_cyclic : is_cyclic_quadrilateral Q) (h_tangential : is_tangential_quadrilateral Q) :
    let (E1, E2, E3, E4) := incircle_tangent_points Q in
    ratio (segment A E1) (segment E1 B) = ratio (segment D E3) (segment E3 C) :=
by
  intro E1 E2 E3 E4
  sorry

end tangency_divides_ratios_l479_479202


namespace restore_salary_l479_479657

variable (W : ℝ) -- Define the initial wage as a real number
variable (newWage : ℝ := 0.7 * W) -- New wage after a 30% reduction

-- Define the hypothesis for the initial wage reduction
theorem restore_salary : (100 * (W / (0.7 * W) - 1)) = 42.86 :=
by
  sorry

end restore_salary_l479_479657


namespace inverse_of_f_l479_479086

noncomputable def f (x : ℝ) : ℝ := (x - 1) ^ 2 - 1

theorem inverse_of_f :
  ∃ (t : ℝ), t ∈ Iic 0 ∧ f t = -1 / 2 ∧ t = -Real.sqrt 2 / 2 :=
by
{
  use -Real.sqrt 2 / 2
  split
  { linarith [Real.sqrt 2_pos], }
  split
  { sorry, }
  { refl, }
}

end inverse_of_f_l479_479086


namespace polynomial_degree_l479_479601

noncomputable def polynomial1 : Polynomial ℤ := 3 * Polynomial.monomial 5 1 + 2 * Polynomial.monomial 4 1 - Polynomial.monomial 1 1 + Polynomial.C 5
noncomputable def polynomial2 : Polynomial ℤ := 4 * Polynomial.monomial 11 1 - 2 * Polynomial.monomial 8 1 + 5 * Polynomial.monomial 5 1 - Polynomial.C 9
noncomputable def polynomial3 : Polynomial ℤ := (Polynomial.monomial 2 1 - Polynomial.C 3) ^ 9

theorem polynomial_degree :
  (polynomial1 * polynomial2 - polynomial3).degree = 18 := by
  sorry

end polynomial_degree_l479_479601


namespace part1_part2_l479_479416

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.log x

theorem part1 (x : ℝ) (hx : x > 0) : f x ≤ x^2 :=
sorry

theorem part2 (x : ℝ) (hx : x > 0) (c : ℝ) (hc : c ≥ -1) : f x ≤ 2 * x + c :=
sorry

end part1_part2_l479_479416


namespace number_of_children_l479_479500

def total_granola_bars : ℕ := 250
def fraction_eaten_by_parents : ℚ := 3 / 5
def granola_bars_per_child : ℕ := 25

theorem number_of_children (total_granola_bars : ℕ) (fraction_eaten_by_parents : ℚ) (granola_bars_per_child : ℕ) :
  ∃ (children : ℕ), (total_granola_bars - total_granola_bars * fraction_eaten_by_parents) / granola_bars_per_child = children := by
  sorry

#eval number_of_children 250 (3/5 : ℚ) 25

end number_of_children_l479_479500


namespace area_quadrilateral_ABCD_l479_479174

-- Given conditions for the hexagon and areas
variable (ABCDEF : convex_hexagon)
variable (A B C D E F : Point)
variable (h1 : trisection(AC AE ∠BAF))
variable (h2 : parallel(BE CD) ∧ parallel(CF DE))
variable (h3 : distance(A B) = 2 * distance(A C) ∧ distance(A C) = 2 * distance(A E) ∧ distance(A E) = 2 * distance(A F))
variable (area_ACDE : ℝ) (h4 : area_ACDE = 2014)
variable (area_ADEF : ℝ) (h5 : area_ADEF = 1400)

-- Prove the area of quadrilateral ABCD
theorem area_quadrilateral_ABCD : ∀ (ABCDEF : convex_hexagon) (A B C D E F : Point) 
  (h1 : trisection(AC AE ∠BAF)) 
  (h2 : parallel(BE CD) ∧ parallel(CF DE))
  (h3 : distance(A B) = 2 * distance(A C) 
     ∧ distance(A C) = 2 * distance(A E) 
     ∧ distance(A E) = 2 * distance(A F))
    (area_ACDE = 2014) 
    (area_ADEF = 1400),
  area (quadrilateral A B C D) = 7295 :=
by sorry

end area_quadrilateral_ABCD_l479_479174


namespace find_f_2024_l479_479745

noncomputable def f (x : ℝ) : ℝ :=   -- Define the function f
  if 1 ≤ x ∧ x < 2 then real.log x + 2 else sorry -- Specify the given condition in the problem statement

axiom odd_f : ∀ x : ℝ, f(-x) = -f(x)                                    -- Define the odd function property
axiom symmetry_property : ∀ x : ℝ, f(x) = f(2 - x)                      -- Define the symmetry property 

theorem find_f_2024 : f 2024 = 0 := 
by
  sorry                                                                 -- A placeholder for the proof

end find_f_2024_l479_479745


namespace post_spacing_change_l479_479265

theorem post_spacing_change :
  ∀ (posts : ℕ → ℝ) (constant_spacing : ℝ), 
  (∀ n, 1 ≤ n ∧ n < 16 → posts (n + 1) - posts n = constant_spacing) →
  posts 16 - posts 1 = 48 → 
  posts 28 - posts 16 = 36 →
  ∃ (k : ℕ), 16 < k ∧ k ≤ 28 ∧ posts (k + 1) - posts k ≠ constant_spacing ∧ posts (k + 1) - posts k = 2.9 ∧ k = 20 := 
  sorry

end post_spacing_change_l479_479265


namespace function_quadrants_l479_479431

theorem function_quadrants (a b : ℝ) (h_a : a > 1) (h_b : b < -1) :
  (∀ x : ℝ, a^x + b > 0 → ∃ x1 : ℝ, a^x1 + b < 0 → ∃ x2 : ℝ, a^x2 + b < 0) :=
sorry

end function_quadrants_l479_479431


namespace stickers_distribution_l479_479103

theorem stickers_distribution : 
  (card {l : List ℕ | l.sum = 10 ∧ l.length = 5}) = 30 := 
sorry

end stickers_distribution_l479_479103


namespace no_two_or_more_consecutive_sum_30_l479_479799

theorem no_two_or_more_consecutive_sum_30 :
  ∀ (a n : ℕ), n ≥ 2 → (n * (2 * a + n - 1) = 60) → false :=
by
  intro a n hn h
  sorry

end no_two_or_more_consecutive_sum_30_l479_479799


namespace probability_X_greater_than_2_l479_479310

variables {X : Type} [Measure_theory.ProbabilityMeasure X]
constants (f : ℝ → ℝ)


-- Definition of the density function f(x)
def density := ∀ x : ℝ, f x = (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-((x - 1)^2 / 2))

-- Condition: ∫ from 0 to 1 of the density function = a
def integral_condition (a : ℝ) := ∫ x in 0..1, f x = a

-- The theorem to be proved
theorem probability_X_greater_than_2 (a : ℝ) (h1 : density) (h2 : integral_condition a) : 
  Measure_theory.ProbabilityMeasure.measure {x : ℝ | x > 2} = 1/2 - a :=
sorry

end probability_X_greater_than_2_l479_479310


namespace ratio_proof_l479_479352

variables (x y m n : ℝ)

def ratio_equation1 (x y m n : ℝ) : Prop :=
  (5 * x + 7 * y) / (3 * x + 2 * y) = m / n

def target_equation (x y m n : ℝ) : Prop :=
  (13 * x + 16 * y) / (2 * x + 5 * y) = (2 * m + n) / (m - n)

theorem ratio_proof (x y m n : ℝ) (h: ratio_equation1 x y m n) :
  target_equation x y m n :=
by
  sorry

end ratio_proof_l479_479352


namespace find_diameter_of_wheel_l479_479316

noncomputable def diameter_of_wheel (π d : ℝ) (total_distance number_of_revolutions : ℝ) :=
  total_distance = number_of_revolutions * π * d

theorem find_diameter_of_wheel :
  ∃ d : ℝ, diameter_of_wheel real.pi d 1672 19.017288444040037 :=
by {
  use 28.000,
  simp [diameter_of_wheel, real.pi],
  -- This follows from the given problem and approximate calculation 
  sorry
}

end find_diameter_of_wheel_l479_479316


namespace find_AC_l479_479132

def angle_A := 60
def angle_B := 45
def side_BC := 12
def sin_60 := (Real.sin (Real.pi / 3)) -- 60 degrees in radians
def sin_45 := (Real.sin (Real.pi / 4)) -- 45 degrees in radians

theorem find_AC : 
  let AC := side_BC * sin_45 / sin_60 in
  AC = 4 * Real.sqrt 6 :=
sorry

end find_AC_l479_479132


namespace three_lines_through_one_point_l479_479699

theorem three_lines_through_one_point
  (lines : Fin 9 → (ℝ × ℝ) → Prop)
  (h_divide : ∀ i, ∃ (x1 y1 x2 y2 : ℝ),
    lines i (x1, y1) ∧ lines i (x2, y2) ∧
    divide_square_in_area_ratio (x1, y1) (x2, y2) (2 / 5) (3 / 5)) :
  ∃ p : (ℝ × ℝ), ∃ (i j k : Fin 9), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
    lines i p ∧ lines j p ∧ lines k p := sorry

end three_lines_through_one_point_l479_479699


namespace smallest_k_divisibility_l479_479895

open Nat

def largest_prime_2023_digits : ℕ :=
  sorry -- placeholder for the actual largest prime with 2023 digits

theorem smallest_k_divisibility (k : ℕ) (h_prime : prime largest_prime_2023_digits) (h_large : largest_prime_2023_digits.digits = 2023) : 
  (largest_prime_2023_digits ^ 2 - k) % 24 = 0 ↔ k = 1 :=
by sorry

end smallest_k_divisibility_l479_479895


namespace arrange_consecutive_integers_no_common_divisors_l479_479849

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def adjacent (i j : ℕ) (n m : ℕ) : Prop :=
  (abs (i - n) = 1 ∧ j = m) ∨
  (i = n ∧ abs (j - m) = 1) ∨
  (abs (i - n) = 1 ∧ abs (j - m) = 1)

theorem arrange_consecutive_integers_no_common_divisors :
  let grid := [[8, 9, 10], [5, 7, 11], [6, 13, 12]] in
  ∀ i j n m, i < 3 → j < 3 → n < 3 → m < 3 →
  adjacent i j n m →
  is_coprime (grid[i][j]) (grid[n][m]) :=
by
  -- This is where you would prove the theorem
  sorry

end arrange_consecutive_integers_no_common_divisors_l479_479849


namespace greatest_ratio_distinct_points_on_circle_l479_479697

-- Define integer points on the circle x^2 + y^2 = 100
def on_circle (P : ℤ × ℤ) := P.1 * P.1 + P.2 * P.2 = 100

-- Define whether a distance is irrational
def is_irrational_dist (P Q : ℤ × ℤ) : Prop :=
  ¬(∃ r : ℚ, r * r = (P.1 - Q.1) * (P.1 - Q.1) + (P.2 - Q.2) * (P.2 - Q.2))

-- Define the distances PQ and RS
def dist (P Q : ℤ × ℤ) :=
  real.sqrt ((P.1 - Q.1) * (P.1 - Q.1) + (P.2 - Q.2) * (P.2 - Q.2))

-- The main problem statement
theorem greatest_ratio_distinct_points_on_circle 
  (P Q R S : ℤ × ℤ)
  (hP : on_circle P) 
  (hQ : on_circle Q) 
  (hR : on_circle R) 
  (hS : on_circle S)
  (hDistinct : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S)
  (hIrrationalPQ : is_irrational_dist P Q)
  (hIrrationalRS : is_irrational_dist R S)
  : dist P Q / dist R S = 1 := begin
  sorry
end

end greatest_ratio_distinct_points_on_circle_l479_479697


namespace a_is_not_p_oscillating_sequence_b_is_p_oscillating_sequence_c_is_p_oscillating_sequence_l479_479685

-- 1. Definitions and theorems for sequences a_n and b_n
def p_oscillating_sequence (x : ℕ → ℝ) (p : ℝ) : Prop :=
  ∀ n : ℕ, (x (n + 1) - p) * (x n - p) < 0

def a_n (n : ℕ) : ℝ := 2 * n - 1
def b_n (q : ℝ) (n : ℕ) : ℝ := q ^ n

theorem a_is_not_p_oscillating_sequence :
  ¬ ∃ p : ℝ, p_oscillating_sequence a_n p := 
sorry

theorem b_is_p_oscillating_sequence (q : ℝ) (hq : -1 < q ∧ q < 0) :
  p_oscillating_sequence (b_n q) 0 :=
sorry

-- 2. Definitions and theorems for sequences c_n
def c_n : ℕ → ℝ
| 0     := 1
| (n+1) := 1 / (c_n n + 1)

theorem c_is_p_oscillating_sequence :
  ∃ p : ℝ, p = (Real.sqrt 5 - 1) / 2 ∧ p_oscillating_sequence c_n p :=
sorry

end a_is_not_p_oscillating_sequence_b_is_p_oscillating_sequence_c_is_p_oscillating_sequence_l479_479685


namespace first_candidate_percentage_l479_479146

-- Conditions
def total_votes : ℕ := 600
def second_candidate_votes : ℕ := 240
def first_candidate_votes : ℕ := total_votes - second_candidate_votes

-- Question and correct answer
theorem first_candidate_percentage : (first_candidate_votes * 100) / total_votes = 60 := by
  sorry

end first_candidate_percentage_l479_479146


namespace smallest_product_not_factor_60_l479_479592

theorem smallest_product_not_factor_60 : ∃ (a b : ℕ), a ≠ b ∧ a ∣ 60 ∧ b ∣ 60 ∧ ¬ (a * b) ∣ 60 ∧ a * b = 8 := sorry

end smallest_product_not_factor_60_l479_479592


namespace point_division_ratio_l479_479615

variables {E F G H O : Type}
variables (EG EO GO EH OH OF : ℝ)

def rotation_condition (theta : ℝ) : Prop :=
  theta = real.arccos (1 / 3)

def point_on_side (O E G : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ k < 1 ∧ O = k * E + (1 - k) * G

def move_condition (O : ℝ) (E G F H : Type) [has_dist O E G F H] :
  Prop := dist F E = 0 ∧ dist G H = dist G O ∧ H ∈ closed_segment F G

def similar_triangles (theta : ℝ) (FO EO GO HO : ℝ) : Prop :=
  rotation_condition theta ∧ 
  FO/GO = EO/HO ∧ FO = EO ∧ GO = HO

theorem point_division_ratio (theta : ℝ) (O : ℝ) (E G F H : Type) 
  [has_dist O E] :
  rotation_condition theta →
  point_on_side O E G →
  move_condition O E G F H →
  similar_triangles theta FO EO GO HO →
  FO / OG = 3 :=
sorry

end point_division_ratio_l479_479615


namespace right_triangle_area_l479_479237

-- Definitions of conditions
def right_triangle_hypotenuse : ℝ := 10 * Real.sqrt 3
def angle_30_degrees : ℝ := Real.pi / 6 -- 30 degrees in radians

-- Assertion: the area of the triangle with given conditions
theorem right_triangle_area : 
  ∃ (s l : ℝ), 
  s = right_triangle_hypotenuse / 2 ∧ 
  l = (Real.sqrt 3 / 2) * right_triangle_hypotenuse ∧ 
  0.5 * s * l = 37.5 * Real.sqrt 3 :=
sorry

end right_triangle_area_l479_479237


namespace determine_conflicting_pairs_l479_479982

structure EngineerSetup where
  n : ℕ
  barrels : Fin (2 * n) → Reactant
  conflicts : Fin n → (Reactant × Reactant)

def testTubeBurst (r1 r2 : Reactant) (conflicts : Fin n → (Reactant × Reactant)) : Prop :=
  ∃ i, conflicts i = (r1, r2) ∨ conflicts i = (r2, r1)

theorem determine_conflicting_pairs (setup : EngineerSetup) :
  ∃ pairs : Fin n → (Reactant × Reactant),
  (∀ i, pairs i ∈ { p | ∃ j, setup.conflicts j = p ∨ setup.conflicts j = (p.snd, p.fst) }) ∧
  (∀ i j, i ≠ j → pairs i ≠ pairs j) := 
sorry

end determine_conflicting_pairs_l479_479982


namespace find_total_perimeter_l479_479520

-- Define the conditions of the problem
def unit_circle_radius := 1
def perimeter_semicircle (r : ℝ) := π * r
def perimeter_circle (r : ℝ) := 2 * π * r
def smaller_radius (r : ℝ) := r
def larger_radius (r : ℝ) := unit_circle_radius - r

-- Define the total perimeter of the shaded portion of the logo
def total_perimeter (r : ℝ) :=
  2 * perimeter_semicircle (smaller_radius r) + 
  2 * perimeter_semicircle (larger_radius r) + 
  perimeter_circle unit_circle_radius

-- The main theorem statement
theorem find_total_perimeter : total_perimeter 1 = 4 * π :=
sorry

end find_total_perimeter_l479_479520


namespace sum_g_l479_479480

def g (x : ℝ) : ℝ := 2 / (4 ^ x + 2)

theorem sum_g (n : ℝ) (h₁ : n = 999) (h₂: ∑ k in (finset.range n), g((k+1)/1000) = 498.5) : 
  (finset.range n).sum (λ k, g ((k+1)/1000)) = 498.5 :=
by 
  rw [h₁, h₂]
  sorry

end sum_g_l479_479480


namespace golu_distance_travelled_l479_479793

theorem golu_distance_travelled 
  (b : ℝ) (c : ℝ) (h : c^2 = x^2 + b^2) : x = 8 := by
  sorry

end golu_distance_travelled_l479_479793


namespace correct_statements_l479_479744

def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom monotonic_increasing_on_neg1_0 : ∀ ⦃x y : ℝ⦄, -1 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y
axiom functional_eqn (x : ℝ) : f (1 - x) + f (1 + x) = 0

theorem correct_statements :
  (∀ x, f (1 - x) = -f (1 + x)) ∧ f 2 ≤ f x :=
by
  sorry

end correct_statements_l479_479744


namespace train_crossing_time_l479_479427

theorem train_crossing_time (length_train : ℕ) (speed_train_kmph : ℕ) (length_bridge : ℕ) : ℝ :=
  let total_distance := length_train + length_bridge in
  let speed_train_mps := speed_train_kmph * (1000 / 3600 : ℝ) in
  let time_to_cross := total_distance / speed_train_mps in
  time_to_cross = 72.5

end train_crossing_time_l479_479427


namespace max_image_cardinality_eq_l479_479485

-- Assume S is a finite set.
variable (S : Type) [Fintype S]

-- Define the condition f : P(S) → ℝ, with f(X ∩ Y) = min(f(X), f(Y))
def valid_function (f : Set S → ℝ) : Prop :=
  ∀ X Y : Set S, f(X ∩ Y) = min (f X) (f Y)

-- Define the image of a function.
def image_of_function (f : Set S → ℝ) : Set ℝ :=
  Set.range f

-- Define F as the set of valid functions.
def F : Set (Set S → ℝ) :=
  {f | valid_function S f}

-- Define the cardinality of the image of a function.
def image_cardinality (f : Set S → ℝ) : ℕ :=
  (image_of_function S f).to_finset.card

-- Definition of the maximum image cardinality.
def max_image_cardinality : ℕ :=
  ⨆ f ∈ F, image_cardinality S f

-- Main theorem to prove.
theorem max_image_cardinality_eq : max_image_cardinality S = Fintype.card S + 1 :=
sorry

end max_image_cardinality_eq_l479_479485


namespace number_of_smaller_cubes_l479_479297

theorem number_of_smaller_cubes (N : ℕ) : 
  (∀ a : ℕ, ∃ n : ℕ, n * a^3 = 125) ∧
  (∀ b : ℕ, b ≤ 5 → ∃ m : ℕ, m * b^3 ≤ 125) ∧
  (∃ x y : ℕ, x ≠ y) → 
  N = 118 :=
sorry

end number_of_smaller_cubes_l479_479297


namespace sandwiches_provided_l479_479255

theorem sandwiches_provided (original_count sold_out : ℕ) (h1 : original_count = 9) (h2 : sold_out = 5) : (original_count - sold_out = 4) :=
by
  sorry

end sandwiches_provided_l479_479255


namespace quadratic_polynomial_l479_479029

noncomputable def p (x : ℝ) : ℝ := -3 * x^2 - 9 * x + 84

axiom p_at_neg7 : p (-7) = 0
axiom p_at_4 : p 4 = 0
axiom p_at_5 : p 5 = -36

theorem quadratic_polynomial : p (-7) = 0 ∧ p 4 = 0 ∧ p 5 = -36 :=
by {
  split,
  { exact p_at_neg7 },
  split,
  { exact p_at_4 },
  { exact p_at_5 },
  sorry
}

end quadratic_polynomial_l479_479029


namespace count_triples_l479_479171

open Nat

def lcm (m n : ℕ) : ℕ := (m * n) / (gcd m n)

theorem count_triples :
  let S := { (a, b, c) : ℕ × ℕ × ℕ // 0 < a ∧ 0 < b ∧ 0 < c ∧ lcm a b = 1200 ∧ lcm b c = 2400 ∧ lcm c a = 3000} in
  S.card = 96 :=
by sorry

end count_triples_l479_479171


namespace quadratic_intersects_x_axis_l479_479069

theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersects_x_axis_l479_479069


namespace k_lucky_arrangement_exists_for_n_geq_3_l479_479913

def isKLucky (N : ℕ) (positions : Fin N → ℝ) (k : ℕ) : Prop :=
  ∃ new_positions : Fin N → ℝ, 
    (∀ i j : Fin N, i ≠ j → positions i ≠ new_positions i) ∧
    (∑ i j : Fin N, |positions i - positions j|) ≥ k * (∑ i j : Fin N, |new_positions i - new_positions j|)

theorem k_lucky_arrangement_exists_for_n_geq_3 :
  (∀ k : ℕ, ∃ positions : Fin (N : ℕ) → ℝ, isKLucky N positions k) ↔ N ≥ 3 := by
    sorry

end k_lucky_arrangement_exists_for_n_geq_3_l479_479913


namespace angle_EHX_is_12_l479_479145

open EuclideanGeometry

noncomputable def acute_triangle (DEF : Triangle) : Prop :=
  DEF.is_acute

noncomputable def angle_DEF (DEF : Triangle) : angle :=
  63

noncomputable def angle_DFE (DEF : Triangle) : angle :=
  78

noncomputable def altitudes_intersect_at_H (DEF : Triangle) (DX EY : Line) (H : Point) : Prop :=
  DEF.is_altitude DX ∧ DEF.is_altitude EY ∧ DX ≠ EY ∧ H ∈ DX ∧ H ∈ EY

theorem angle_EHX_is_12 (DEF : Triangle) (DX EY : Line) (H : Point)
  (h₁ : acute_triangle DEF)
  (h₂ : altitudes_intersect_at_H DEF DX EY H)
  (h₃ : DEF.angle_def = 63)
  (h₄ : DEF.angle_dfe = 78) :
  ∠EHX = 12 :=
sorry

end angle_EHX_is_12_l479_479145


namespace PQ_ratio_l479_479228

-- Definitions
def hexagon_area : ℕ := 7
def base_of_triangle : ℕ := 4

-- Conditions
def PQ_bisects_area (A : ℕ) : Prop :=
  A = hexagon_area / 2

def area_below_PQ (U T : ℚ) : Prop :=
  U + T = hexagon_area / 2 ∧ U = 1

def triangle_area (T b : ℚ) : ℚ :=
  1/2 * b * (5/4)

def XQ_QY_ratio (XQ QY : ℚ) : ℚ :=
  XQ / QY

-- Theorem Statement
theorem PQ_ratio (XQ QY : ℕ) (h1 : PQ_bisects_area (hexagon_area / 2))
  (h2 : area_below_PQ 1 (triangle_area (5/2) base_of_triangle))
  (h3 : XQ + QY = base_of_triangle) : XQ_QY_ratio XQ QY = 1 := sorry

end PQ_ratio_l479_479228


namespace hannah_strawberries_l479_479425

def april_days : ℕ := 30
def odd_day_harvest : ℕ := 5
def even_day_harvest : ℕ := 7
def giveaway_percentage : ℚ := 0.25
def spoilage_percentage : ℚ := 0.15
def stolen_strawberries : ℕ := 20

theorem hannah_strawberries : 
  let total_strawberries := (15 * odd_day_harvest) + (15 * even_day_harvest)
  let given_away := giveaway_percentage * total_strawberries
  let remaining_after_giveaway := total_strawberries - given_away
  let spoiled := spoilage_percentage * remaining_after_giveaway
  let remaining_after_spoilage := remaining_after_giveaway - spoiled
  in remaining_after_spoilage - stolen_strawberries = 95 :=
  sorry

end hannah_strawberries_l479_479425


namespace symmetric_point_exists_l479_479711

-- Define the point P and line equation.
structure Point (α : Type*) := (x : α) (y : α)
def P : Point ℝ := ⟨5, -2⟩
def line_eq (x y : ℝ) : Prop := x - y + 5 = 0

-- Define a function for the line PQ being perpendicular to the given line.
def is_perpendicular (P Q : Point ℝ) : Prop :=
  (Q.y - P.y) / (Q.x - P.x) = -1

-- Define a function for the midpoint of PQ lying on the given line.
def midpoint_on_line (P Q : Point ℝ) : Prop :=
  line_eq ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

-- Define the symmetry function based on the provided conditions.
def is_symmetric (Q : Point ℝ) : Prop :=
  is_perpendicular P Q ∧ midpoint_on_line P Q

-- State the main theorem to be proved: there exists a point Q that satisfies the 
-- conditions and is symmetric to P with respect to the given line.
theorem symmetric_point_exists : ∃ Q : Point ℝ, is_symmetric Q ∧ Q = ⟨-7, 10⟩ :=
by
  sorry

end symmetric_point_exists_l479_479711


namespace quadratic_intersection_l479_479074

theorem quadratic_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersection_l479_479074


namespace nat_forms_6n_plus_1_or_5_prod_6n_plus_1_prod_6n_plus_5_prod_6n_plus_1_and_5_l479_479321

theorem nat_forms_6n_plus_1_or_5 (x : ℕ) (h1 : ¬ (x % 2 = 0) ∧ ¬ (x % 3 = 0)) :
  ∃ n : ℕ, x = 6 * n + 1 ∨ x = 6 * n + 5 := 
sorry

theorem prod_6n_plus_1 (m n : ℕ) :
  (6 * m + 1) * (6 * n + 1) = 6 * (6 * m * n + m + n) + 1 :=
sorry

theorem prod_6n_plus_5 (m n : ℕ) :
  (6 * m + 5) * (6 * n + 5) = 6 * (6 * m * n + 5 * m + 5 * n + 4) + 1 :=
sorry

theorem prod_6n_plus_1_and_5 (m n : ℕ) :
  (6 * m + 1) * (6 * n + 5) = 6 * (6 * m * n + 5 * m + n) + 5 :=
sorry

end nat_forms_6n_plus_1_or_5_prod_6n_plus_1_prod_6n_plus_5_prod_6n_plus_1_and_5_l479_479321


namespace three_students_with_A_l479_479700

-- Define the statements of the students
variables (Eliza Fiona George Harry : Prop)

-- Conditions based on the problem statement
axiom Fiona_implies_Eliza : Fiona → Eliza
axiom George_implies_Fiona : George → Fiona
axiom Harry_implies_George : Harry → George

-- There are exactly three students who scored an A
theorem three_students_with_A (hE : Bool) : 
  (Eliza = false) → (Fiona = true) → (George = true) → (Harry = true) :=
by
  sorry

end three_students_with_A_l479_479700


namespace range_of_m_l479_479784

theorem range_of_m (A : Set ℝ) (m : ℝ) (h₁ : A = { x | x^2 + sqrt m * x + 1 = 0 })
  (h₂ : A ∩ Set.Univ = ∅) : 0 ≤ m ∧ m < 4 := by
  sorry

end range_of_m_l479_479784


namespace mean_of_remaining_two_numbers_l479_479933

theorem mean_of_remaining_two_numbers :
  let n1 := 1871
  let n2 := 1997
  let n3 := 2023
  let n4 := 2029
  let n5 := 2113
  let n6 := 2125
  let n7 := 2137
  let total_sum := n1 + n2 + n3 + n4 + n5 + n6 + n7
  let known_mean := 2100
  let mean_of_other_two := 1397.5
  total_sum = 13295 →
  5 * known_mean = 10500 →
  total_sum - 10500 = 2795 →
  2795 / 2 = mean_of_other_two :=
by
  intros
  sorry

end mean_of_remaining_two_numbers_l479_479933


namespace domain_of_f_l479_479689

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 2*x - 3)

theorem domain_of_f : 
  {x : ℝ | (x^2 - 2*x - 3) ≠ 0} = {x : ℝ | x < -1} ∪ {x : ℝ | -1 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l479_479689


namespace exists_c_number_of_bad_integers_l479_479092

-- Definitions and assumptions
variables (p q : ℕ)
variables (coprime : Nat.gcd p q = 1)

-- Definition of Good and Bad integers
def is_good (n : ℕ) : Prop :=
  ∃ x y : ℕ, n = p * x + q * y

def is_bad (n : ℕ) : Prop :=
  ¬ is_good p q n

-- Statement 1
theorem exists_c (h_coprime : Nat.gcd p q = 1) : 
  ∃ c, ∀ n, (is_good p q n ∨ is_good p q (c - n)) ∧ ¬ (is_good p q n ∧ is_good p q (c - n)) :=
sorry

-- Statement 2
theorem number_of_bad_integers (h_coprime : Nat.gcd p q = 1) : 
  ∃ m, m = (p - 1) * (q - 1) / 2 ∧ m = { n | is_bad p q n ∧ n < pq - p - q }.card :=
sorry

end exists_c_number_of_bad_integers_l479_479092


namespace distance_ratio_l479_479588

variable (d_RB d_BC : ℝ)

theorem distance_ratio
    (h1 : d_RB / 60 + d_BC / 20 ≠ 0)
    (h2 : 36 * (d_RB / 60 + d_BC / 20) = d_RB + d_BC) : 
    d_RB / d_BC = 2 := 
sorry

end distance_ratio_l479_479588


namespace problem1_solve_eq_l479_479524

theorem problem1_solve_eq (x : ℝ) : x * (x - 5) = 3 * x - 15 ↔ (x = 5 ∨ x = 3) := by
  sorry

end problem1_solve_eq_l479_479524


namespace intersection_points_l479_479993

noncomputable def y1 : ℝ := 3 * ((9 - Real.sqrt 37) / 2)^2 - 15 * ((9 - Real.sqrt 37) / 2) - 15
noncomputable def y2 : ℝ := 3 * ((9 + Real.sqrt 37) / 2)^2 - 15 * ((9 + Real.sqrt 37) / 2) - 15

theorem intersection_points :
  ∃ (x1 x2 x3 y1 y2 : ℝ), 
    (y1 = 3 * ((9 - Real.sqrt 37) / 2)^2 - 15 * ((9 - Real.sqrt 37) / 2) - 15) ∧
    (y2 = 3 * ((9 + Real.sqrt 37) / 2)^2 - 15 * ((9 + Real.sqrt 37) / 2) - 15) ∧
    (x1, (y = 3 * x1^2 - 15 * x1 - 15) ∧ (y = x1^3 - 5 * x1^2 + 8 * x1 - 4) →
       (x2 = (9 - Real.sqrt 37) / 2 ∧ (y = 3 * x2^2 - 15 * x2 - 15) ∧ (y = x2^3 - 5 * x2^2 + 8 * x2 - 4)) →
       (x3 = (9 + Real.sqrt 37) / 2 ∧ (y = 3 * x3^2 - 15 * x3 - 15) ∧ (y = x3^3 - 5 * x3^2 + 8 * x3 - 4))) :=
(-1, 3),
((9 - Real.sqrt 37) / 2, y1),
((9 + Real.sqrt 37) / 2, y2) sorry

end intersection_points_l479_479993


namespace grid_is_valid_l479_479833

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Check if all adjacent cells (by side or diagonal) in a 3x3 grid are coprime -/
def valid_grid (grid : Array (Array ℕ)) : Prop :=
  let adjacent_pairs := 
    [(0, 0, 0, 1), (0, 1, 0, 2), 
     (1, 0, 1, 1), (1, 1, 1, 2), 
     (2, 0, 2, 1), (2, 1, 2, 2),
     (0, 0, 1, 0), (0, 1, 1, 1), (0, 2, 1, 2),
     (1, 0, 2, 0), (1, 1, 2, 1), (1, 2, 2, 2),
     (0, 0, 1, 1), (0, 1, 1, 2), 
     (1, 0, 2, 1), (1, 1, 2, 2), 
     (0, 2, 1, 1), (1, 2, 2, 1), 
     (0, 1, 1, 0), (1, 1, 2, 0)] 
  adjacent_pairs.all (λ ⟨r1, c1, r2, c2⟩, is_coprime (grid[r1][c1]) (grid[r2][c2]))

def grid : Array (Array ℕ) :=
  #[#[8, 9, 10],
    #[5, 7, 11],
    #[6, 13, 12]]

theorem grid_is_valid : valid_grid grid :=
  by
    -- Proof omitted, placeholder
    sorry

end grid_is_valid_l479_479833


namespace sin_2x_value_l479_479753

theorem sin_2x_value (x : ℝ) (h : sin x + cos x = 1 / 2) : sin (2 * x) = -3 / 4 :=
by
  sorry

end sin_2x_value_l479_479753


namespace expression_value_l479_479063

theorem expression_value :
  3 * 12^2 - 3 * 13 + 2 * 16 * 11^2 = 4265 :=
by
  sorry

end expression_value_l479_479063


namespace false_proposition_C_l479_479377

variable (a b c : ℝ)
noncomputable def f (x : ℝ) := a * x^2 + b * x + c

theorem false_proposition_C 
  (ha : a > 0)
  (x0 : ℝ)
  (hx0 : x0 = -b / (2 * a)) :
  ¬ ∀ x : ℝ, f a b c x ≤ f a b c x0 :=
by
  sorry

end false_proposition_C_l479_479377


namespace circle_equation_l479_479356

-- Defining the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

-- Defining the point
def point : ℝ × ℝ := (1, -1)

-- Proving the equation of the new circle that passes through the intersection points 
-- of the given circles and the given point
theorem circle_equation (x y : ℝ) :
  (circle1 x y ∧ circle2 x y ∧ x = 1 ∧ y = -1) → 9 * x^2 + 9 * y^2 - 14 * x + 4 * y = 0 :=
sorry

end circle_equation_l479_479356


namespace geometric_sequence_product_proof_l479_479152

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geometric_sequence_product_proof (a : ℕ → ℝ) (q : ℝ)
  (h_geo : geometric_sequence a q) 
  (h1 : a 2010 * a 2011 * a 2012 = 3)
  (h2 : a 2013 * a 2014 * a 2015 = 24) :
  a 2016 * a 2017 * a 2018 = 192 :=
sorry

end geometric_sequence_product_proof_l479_479152


namespace no_rect_with_odd_grid_intersections_l479_479859

theorem no_rect_with_odd_grid_intersections 
  (rectangle_on_grid : ℝ × ℝ × ℝ × ℝ)
  (is_45_deg_rotated : ∀ p1 p2 : ℝ × ℝ, p1 ≠ p2 → 
    (exists x' y' : ℝ, (x' ≠ p1.1 ∧ x' ≠ p2.1) ∧ (y' ≠ p1.2 ∧ y' ≠ p2.2)))
  (no_vertices_on_grid : ∀ p : ℝ × ℝ, (p.1 ∉ set_of (λ x, ∃ n : ℤ, x = n)) ∧ (p.2 ∉ set_of (λ y, ∃ n : ℤ, y = n))) :
  ¬(∀ side : ℝ × ℝ × ℝ × ℝ, ∃ t : ℤ, odd t ∧ side.intersect_grid_lines t) :=
sorry

end no_rect_with_odd_grid_intersections_l479_479859


namespace projection_of_a_onto_b_l479_479375

def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let (ax, ay) := a
  let (bx, by) := b
  let dot_ab := ax * bx + ay * by
  let mag_b2 := bx * bx + by * by
  (dot_ab / mag_b2 * bx, dot_ab / mag_b2 * by)

theorem projection_of_a_onto_b :
  vector_projection (1, 0) (1, 1) = (1/2, 1/2) :=
by
  sorry

end projection_of_a_onto_b_l479_479375


namespace Roberto_outfits_count_l479_479928

-- Define the problem conditions
def num_trousers : ℕ := 5
def num_shirts : ℕ := 7
def num_jackets : ℕ := 4
def num_belts : ℕ := 2
def restricted_jacket_trousers : ℕ := 2

-- Define the required statement
theorem Roberto_outfits_count : 
  let total_combinations_without_restriction := num_trousers * num_shirts * num_jackets * num_belts in
  let restricted_combinations := (num_jackets - 1) * num_trousers * num_shirts * num_belts - restricted_jacket_trousers * num_shirts * num_belts in
    total_combinations_without_restriction - restricted_combinations = 168 :=
by
  sorry

end Roberto_outfits_count_l479_479928


namespace relationship_between_mean_median_modes_l479_479823

def dates : List ℕ := (List.replicate 12 1) ++ (List.replicate 12 2) ++ (List.replicate 12 3) ++
                       (List.replicate 12 4) ++ (List.replicate 12 5) ++ (List.replicate 12 6) ++
                       (List.replicate 12 7) ++ (List.replicate 12 8) ++ (List.replicate 12 9) ++
                       (List.replicate 12 10) ++ (List.replicate 12 11) ++ (List.replicate 12 12) ++
                       (List.replicate 12 13) ++ (List.replicate 12 14) ++ (List.replicate 12 15) ++
                       (List.replicate 12 16) ++ (List.replicate 12 17) ++ (List.replicate 12 18) ++
                       (List.replicate 12 19) ++ (List.replicate 12 20) ++ (List.replicate 12 21) ++
                       (List.replicate 12 22) ++ (List.replicate 12 23) ++ (List.replicate 12 24) ++
                       (List.replicate 12 25) ++ (List.replicate 12 26) ++ (List.replicate 12 27) ++
                       (List.replicate 12 28) ++ (List.replicate 12 29) ++ (List.replicate 11 30) ++
                       (List.replicate 7 31)

noncomputable def mean (xs : List ℕ) : ℚ :=
  (xs.foldl (· + ·) 0 : ℚ) / (xs.length : ℚ)

noncomputable def median (xs : List ℕ) : ℚ :=
  let sorted_xs := xs.qsort (· ≤ ·)
  if sorted_xs.length % 2 = 0 then
    ((sorted_xs.get (sorted_xs.length / 2 - 1) + sorted_xs.get (sorted_xs.length / 2)) : ℚ) / 2
  else
    sorted_xs.get (sorted_xs.length / 2)

def modes (xs : List ℕ) : List ℕ :=
  let counts := xs.groupBy id |>.map (·.length)
  let max_count := xs.groupBy id |>.map (·.length) |>.maximumD 0
  xs.groupBy id |>.filter (λ g => g.length = max_count) |>.map head |>.join

noncomputable def median_of_modes (xs : List ℕ) : ℚ :=
  median (modes xs)

theorem relationship_between_mean_median_modes :
  let μ := mean dates
  let M := median dates
  let d := median_of_modes dates
  d < μ ∧ μ < M :=
by
  -- Proof omitted
  sorry

end relationship_between_mean_median_modes_l479_479823


namespace competition_participants_bounds_l479_479322

theorem competition_participants_bounds :
  ∀ (n c : ℕ),
    (10.35 ≤ c / n ∧ c / n < 10.45) →
    (10.55 ≤ (c + 4) / n ∧ (c + 4) / n < 10.65) →
    14 ≤ n ∧ n ≤ 39 :=
by
  intros n c h h_corrected
  sorry

end competition_participants_bounds_l479_479322


namespace proof_equivalence_l479_479018

noncomputable def compute_expression (N : ℕ) (M : ℕ) : ℚ :=
  ((N - 3)^3 + (N - 2)^3 + (N - 1)^3 + N^3 + (N + 1)^3 + (N + 2)^3 + (N + 3)^3) /
  ((M - 3) * (M - 2) + (M - 1) * M + M * (M + 1) + (M + 2) * (M + 3))

theorem proof_equivalence:
  let N := 65536
  let M := 32768
  compute_expression N M = 229376 := 
  by
    sorry

end proof_equivalence_l479_479018


namespace area_shaded_quad_correct_l479_479256

-- Define the side lengths of the squares
def side_length_small : ℕ := 3
def side_length_middle : ℕ := 5
def side_length_large : ℕ := 7

-- Define the total base length
def total_base_length : ℕ := side_length_small + side_length_middle + side_length_large

-- The height of triangle T3, equal to the side length of the largest square
def height_T3 : ℕ := side_length_large

-- The height-to-base ratio for each triangle
def height_to_base_ratio : ℚ := height_T3 / total_base_length

-- The heights of T1 and T2
def height_T1 : ℚ := side_length_small * height_to_base_ratio
def height_T2 : ℚ := (side_length_small + side_length_middle) * height_to_base_ratio

-- The height of the trapezoid, which is the side length of the middle square
def trapezoid_height : ℕ := side_length_middle

-- The bases of the trapezoid
def base1 : ℚ := height_T1
def base2 : ℚ := height_T2

-- The area of the trapezoid formula
def area_shaded_quad : ℚ := (trapezoid_height * (base1 + base2)) / 2

-- Assertion that the area of the shaded quadrilateral is equal to 77/6
theorem area_shaded_quad_correct : area_shaded_quad = 77 / 6 := by sorry

end area_shaded_quad_correct_l479_479256


namespace max_ratio_ellipse_l479_479406

theorem max_ratio_ellipse (a : ℝ) (h : a > Real.sqrt 3) : 
  (a = 2) ↔ 
  (let F : ℝ × ℝ := (-Real.sqrt (a^2 - 3), 0),
       G : ℝ × ℝ := (-a, 0),
       H : ℝ × ℝ := (-a^2 / Real.sqrt (a^2 - 3), 0),
       FG : ℝ := a - Real.sqrt (a^2 - 3),
       OH : ℝ := a^2 / Real.sqrt (a^2 - 3),
       ratio := FG / OH
   in   ∀ x : ℝ, x ∈ Set.Icc 0 1 → (-((x - 1 / 2)^2) + 1 / 4)) → ratio
sorry

end max_ratio_ellipse_l479_479406


namespace find_other_number_l479_479284

theorem find_other_number (A B : ℕ) (LCM HCF : ℕ) (h1 : LCM = 2310) (h2 : HCF = 83) (h3 : A = 210) (h4 : LCM * HCF = A * B) : B = 913 :=
by
  sorry

end find_other_number_l479_479284


namespace paulas_dimes_l479_479221

noncomputable theory

variables (n d : ℕ)
def average_before (total_value n : ℕ) : ℕ := total_value / n
def average_after (total_value : ℕ) : ℕ := (total_value + 25) / (n + 1)

theorem paulas_dimes :
  average_before (20 * n) n = 20 →
  average_after (20 * n) = 21 →
  d = 0 :=
by
  intros h1 h2
  have eq1 : 20 * n + 25 = 21 * (n + 1), by linarith
  have eq2 : n = 4, by linarith
  have coin_composition := (3 * 25) + (1 * 5)
  have total_value_composition := coin_composition = 80, by linarith
  have h3 : total_value_composition = 80 → d= 0, by sorry
  exact h3 total_value_composition

end paulas_dimes_l479_479221


namespace parallel_vectors_values_l479_479096

noncomputable section

open Real

variables (t : ℝ) (a b : ℝ × ℝ)

def vector_a := (1 : ℝ, t)
def vector_b := (t : ℝ, 4)

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

theorem parallel_vectors_values (h : vectors_parallel (1, t) (t, 4)) :
  t = 2 ∨ t = -2 := by
  sorry

end parallel_vectors_values_l479_479096


namespace product_of_possible_values_l479_479114

theorem product_of_possible_values (x : ℚ) (h : abs ((18 : ℚ) / (2 * x) - 4) = 3) : (x = 9 ∨ x = 9/7) → (9 * (9/7) = 81/7) :=
by
  intros
  sorry

end product_of_possible_values_l479_479114


namespace arrange_consecutive_integers_no_common_divisors_l479_479847

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def adjacent (i j : ℕ) (n m : ℕ) : Prop :=
  (abs (i - n) = 1 ∧ j = m) ∨
  (i = n ∧ abs (j - m) = 1) ∨
  (abs (i - n) = 1 ∧ abs (j - m) = 1)

theorem arrange_consecutive_integers_no_common_divisors :
  let grid := [[8, 9, 10], [5, 7, 11], [6, 13, 12]] in
  ∀ i j n m, i < 3 → j < 3 → n < 3 → m < 3 →
  adjacent i j n m →
  is_coprime (grid[i][j]) (grid[n][m]) :=
by
  -- This is where you would prove the theorem
  sorry

end arrange_consecutive_integers_no_common_divisors_l479_479847


namespace sqrt_6_plus_sqrt_7_gt_2sqrt2_plus_sqrt5_l479_479007

theorem sqrt_6_plus_sqrt_7_gt_2sqrt2_plus_sqrt5 : 
  (real.sqrt 6) + (real.sqrt 7) > (2 * (real.sqrt 2)) + (real.sqrt 5) := 
by
  have h1 : (real.sqrt 42) > (real.sqrt 40) := sorry,
  have a2 : (real.sqrt 6) + (real.sqrt 7) = real.sqrt ((real.sqrt 6) ^ 2 + 2 * (real.sqrt 6) * (real.sqrt 7) + (real.sqrt 7) ^ 2) := sorry,
  have a3 : (2 * (real.sqrt 2)) + (real.sqrt 5) = real.sqrt ((2 * (real.sqrt 2)) ^ 2 + 2 * (2 * (real.sqrt 2)) * (real.sqrt 5) + (real.sqrt 5) ^ 2) := sorry,
  have h2 : real.sqrt (13 + 2 * (real.sqrt 42)) > real.sqrt (13 + 2 * (real.sqrt 40)) := sorry,
  exact h2

end sqrt_6_plus_sqrt_7_gt_2sqrt2_plus_sqrt5_l479_479007


namespace real_part_of_z_l479_479907

noncomputable def z : ℂ := (2 + complex.I) / ((1 + complex.I) ^ 2)

theorem real_part_of_z : z.re = 1 / 2 := 
by
  sorry

end real_part_of_z_l479_479907
