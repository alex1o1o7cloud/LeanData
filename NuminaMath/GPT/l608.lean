import Mathlib

namespace boys_more_than_girls_l608_60809

def numGirls : ℝ := 28.0
def numBoys : ℝ := 35.0

theorem boys_more_than_girls : numBoys - numGirls = 7.0 := by
  sorry

end boys_more_than_girls_l608_60809


namespace largest_circle_area_l608_60870

theorem largest_circle_area (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) :
  ∃ r : ℝ, (2 * π * r = 60) ∧ (π * r ^ 2 = 900 / π) := 
sorry

end largest_circle_area_l608_60870


namespace spending_less_l608_60810

-- Define the original costs in USD for each category.
def cost_A_usd : ℝ := 520
def cost_B_usd : ℝ := 860
def cost_C_usd : ℝ := 620

-- Define the budget cuts for each category.
def cut_A : ℝ := 0.25
def cut_B : ℝ := 0.35
def cut_C : ℝ := 0.30

-- Conversion rate from USD to EUR.
def conversion_rate : ℝ := 0.85

-- Sales tax rate.
def tax_rate : ℝ := 0.07

-- Calculate the reduced cost after budget cuts for each category.
def reduced_cost_A_usd := cost_A_usd * (1 - cut_A)
def reduced_cost_B_usd := cost_B_usd * (1 - cut_B)
def reduced_cost_C_usd := cost_C_usd * (1 - cut_C)

-- Convert costs from USD to EUR.
def reduced_cost_A_eur := reduced_cost_A_usd * conversion_rate
def reduced_cost_B_eur := reduced_cost_B_usd * conversion_rate
def reduced_cost_C_eur := reduced_cost_C_usd * conversion_rate

-- Calculate the total reduced cost in EUR before tax.
def total_reduced_cost_eur := reduced_cost_A_eur + reduced_cost_B_eur + reduced_cost_C_eur

-- Calculate the tax amount on the reduced cost.
def tax_reduced_cost := total_reduced_cost_eur * tax_rate

-- Total reduced cost in EUR after tax.
def total_reduced_cost_with_tax := total_reduced_cost_eur + tax_reduced_cost

-- Calculate the original costs in EUR without any cuts.
def original_cost_A_eur := cost_A_usd * conversion_rate
def original_cost_B_eur := cost_B_usd * conversion_rate
def original_cost_C_eur := cost_C_usd * conversion_rate

-- Calculate the total original cost in EUR before tax.
def total_original_cost_eur := original_cost_A_eur + original_cost_B_eur + original_cost_C_eur

-- Calculate the tax amount on the original cost.
def tax_original_cost := total_original_cost_eur * tax_rate

-- Total original cost in EUR after tax.
def total_original_cost_with_tax := total_original_cost_eur + tax_original_cost

-- Difference in spending.
def spending_difference := total_original_cost_with_tax - total_reduced_cost_with_tax

-- Prove the company must spend €561.1615 less.
theorem spending_less : spending_difference = 561.1615 := 
by 
  sorry

end spending_less_l608_60810


namespace original_salary_l608_60872

-- Given conditions as definitions
def salaryAfterRaise (x : ℝ) : ℝ := 1.10 * x
def salaryAfterReduction (x : ℝ) : ℝ := salaryAfterRaise x * 0.95
def finalSalary : ℝ := 1045

-- Statement to prove
theorem original_salary (x : ℝ) (h : salaryAfterReduction x = finalSalary) : x = 1000 :=
by
  sorry

end original_salary_l608_60872


namespace maximize_f_l608_60830

noncomputable def f (x y z : ℝ) := x * y^2 * z^3

theorem maximize_f :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 1 →
  f x y z ≤ 1 / 432 ∧ (f x y z = 1 / 432 → x = 1/6 ∧ y = 1/3 ∧ z = 1/2) :=
by
  sorry

end maximize_f_l608_60830


namespace angle_C_45_l608_60851

theorem angle_C_45 (A B C : ℝ) 
(h : (Real.cos A + Real.sin A) * (Real.cos B + Real.sin B) = 2) 
(HA : 0 ≤ A) (HB : 0 ≤ B) (HC : 0 ≤ C):
A + B + C = π → 
A = B →
C = π / 2 - B →
C = π / 4 := 
by
  intros;
  sorry

end angle_C_45_l608_60851


namespace enumerate_set_l608_60840

open Set

def is_positive_integer (x : ℕ) : Prop := x > 0

theorem enumerate_set :
  { p : ℕ × ℕ | p.1 + p.2 = 4 ∧ is_positive_integer p.1 ∧ is_positive_integer p.2 } =
  { (1, 3), (2, 2), (3, 1) } := by 
sorry

end enumerate_set_l608_60840


namespace train_B_speed_l608_60877

-- Given conditions
def speed_train_A := 70 -- km/h
def time_after_meet_A := 9 -- hours
def time_after_meet_B := 4 -- hours

-- Proof statement
theorem train_B_speed : 
  ∃ (V_b : ℕ),
    V_b * time_after_meet_B + V_b * s = speed_train_A * time_after_meet_A + speed_train_A * s ∧
    V_b = speed_train_A := 
sorry

end train_B_speed_l608_60877


namespace triangle_inequality_cosine_rule_l608_60843

theorem triangle_inequality_cosine_rule (a b c : ℝ) (A B C : ℝ)
  (hA : Real.cos A = (b^2 + c^2 - a^2) / (2 * b * c))
  (hB : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c))
  (hC : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  a^3 * Real.cos A + b^3 * Real.cos B + c^3 * Real.cos C ≤ (3 / 2) * a * b * c := 
sorry

end triangle_inequality_cosine_rule_l608_60843


namespace mary_spent_total_amount_l608_60882

def cost_of_berries := 11.08
def cost_of_apples := 14.33
def cost_of_peaches := 9.31
def total_cost := 34.72

theorem mary_spent_total_amount :
  cost_of_berries + cost_of_apples + cost_of_peaches = total_cost :=
by
  sorry

end mary_spent_total_amount_l608_60882


namespace sum_nk_l608_60863

theorem sum_nk (n k : ℕ) (h₁ : 3 * n - 4 * k = 4) (h₂ : 4 * n - 5 * k = 13) : n + k = 55 := by
  sorry

end sum_nk_l608_60863


namespace ratio_of_water_level_increase_l608_60885

noncomputable def volume_narrow_cone (h₁ : ℝ) : ℝ := (16 / 3) * Real.pi * h₁
noncomputable def volume_wide_cone (h₂ : ℝ) : ℝ := (64 / 3) * Real.pi * h₂
noncomputable def volume_marble_narrow : ℝ := (32 / 3) * Real.pi
noncomputable def volume_marble_wide : ℝ := (4 / 3) * Real.pi

theorem ratio_of_water_level_increase :
  ∀ (h₁ h₂ h₁' h₂' : ℝ),
  h₁ = 4 * h₂ →
  h₁' = h₁ + 2 →
  h₂' = h₂ + (1 / 16) →
  volume_narrow_cone h₁ = volume_wide_cone h₂ →
  volume_narrow_cone h₁ + volume_marble_narrow = volume_narrow_cone h₁' →
  volume_wide_cone h₂ + volume_marble_wide = volume_wide_cone h₂' →
  (h₁' - h₁) / (h₂' - h₂) = 32 :=
by
  intros h₁ h₂ h₁' h₂' h₁_eq_4h₂ h₁'_eq_h₁_add_2 h₂'_eq_h₂_add_1_div_16 vol_h₁_eq_vol_h₂ vol_nar_eq vol_wid_eq
  sorry

end ratio_of_water_level_increase_l608_60885


namespace correct_average_marks_l608_60846

theorem correct_average_marks 
  (n : ℕ) (average initial_wrong new_correct : ℕ) 
  (h_num_students : n = 30)
  (h_average_marks : average = 100)
  (h_initial_wrong : initial_wrong = 70)
  (h_new_correct : new_correct = 10) :
  (average * n - (initial_wrong - new_correct)) / n = 98 := 
by
  sorry

end correct_average_marks_l608_60846


namespace range_of_m_l608_60833

theorem range_of_m (m : ℝ) (f : ℝ → ℝ) 
(hf : ∀ x, f x = (Real.sqrt 3) * Real.sin ((Real.pi * x) / m))
(exists_extremum : ∃ x₀, (deriv f x₀ = 0) ∧ (x₀^2 + (f x₀)^2 < m^2)) :
(m > 2) ∨ (m < -2) :=
sorry

end range_of_m_l608_60833


namespace dice_sum_not_18_l608_60822

theorem dice_sum_not_18 (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6) (h2 : 1 ≤ d2 ∧ d2 ≤ 6) 
    (h3 : 1 ≤ d3 ∧ d3 ≤ 6) (h4 : 1 ≤ d4 ∧ d4 ≤ 6) (h_prod : d1 * d2 * d3 * d4 = 144) : 
    d1 + d2 + d3 + d4 ≠ 18 := 
sorry

end dice_sum_not_18_l608_60822


namespace calculate_actual_distance_l608_60817

-- Definitions corresponding to the conditions
def map_scale : ℕ := 6000000
def map_distance_cm : ℕ := 5

-- The theorem statement corresponding to the proof problem
theorem calculate_actual_distance :
  (map_distance_cm * map_scale / 100000) = 300 := 
by
  sorry

end calculate_actual_distance_l608_60817


namespace seq_1964_l608_60805

theorem seq_1964 (a : ℕ → ℤ) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = -1)
  (h4 : ∀ n ≥ 4, a n = a (n - 1) * a (n - 3)) :
  a 1964 = -1 :=
by {
  sorry
}

end seq_1964_l608_60805


namespace pure_imaginary_complex_solution_l608_60845

theorem pure_imaginary_complex_solution (a : Real) :
  (a ^ 2 - 1 = 0) ∧ ((a - 1) ≠ 0) → a = -1 := by
  sorry

end pure_imaginary_complex_solution_l608_60845


namespace problem_l608_60811

theorem problem (a b c d e : ℝ) (h0 : a ≠ 0)
  (h1 : 625 * a + 125 * b + 25 * c + 5 * d + e = 0)
  (h2 : -81 * a + 27 * b - 9 * c + 3 * d + e = 0)
  (h3 : 16 * a + 8 * b + 4 * c + 2 * d + e = 0) :
  (b + c + d) / a = -6 :=
by
  sorry

end problem_l608_60811


namespace worker_usual_time_l608_60864

theorem worker_usual_time (S T : ℝ) (D : ℝ) (h1 : D = S * T)
    (h2 : D = (3/4) * S * (T + 8)) : T = 24 :=
by
  sorry

end worker_usual_time_l608_60864


namespace geometric_sequence_term_l608_60883

theorem geometric_sequence_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) :
  (∀ n, S_n n = 3^n - 1) →
  (a_n n = S_n n - S_n (n - 1)) →
  (a_n n = 2 * 3^(n - 1)) :=
by
  intros h1 h2
  sorry

end geometric_sequence_term_l608_60883


namespace find_x_l608_60813

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x n : ℕ) (h₀ : n = 4) (h₁ : ¬(is_prime (2 * n + x))) : x = 1 :=
by
  sorry

end find_x_l608_60813


namespace handshake_problem_7_boys_21_l608_60871

theorem handshake_problem_7_boys_21 :
  let n := 7
  let total_handshakes := n * (n - 1) / 2
  total_handshakes = 21 → (n - 1) = 6 :=
by
  -- Let n be the number of boys (7 in this case)
  let n := 7
  
  -- Define the total number of handshakes equation
  let total_handshakes := n * (n - 1) / 2
  
  -- Assume the total number of handshakes is 21
  intro h
  -- Proof steps would go here
  sorry

end handshake_problem_7_boys_21_l608_60871


namespace find_x_in_terms_of_abc_l608_60868

variable {x y z a b c : ℝ}

theorem find_x_in_terms_of_abc
  (h1 : xy / (x + y + 1) = a)
  (h2 : xz / (x + z + 1) = b)
  (h3 : yz / (y + z + 1) = c) :
  x = 2 * a * b * c / (a * b + a * c - b * c) := 
sorry

end find_x_in_terms_of_abc_l608_60868


namespace range_of_f_l608_60889

noncomputable def f (x : ℝ) : ℝ := x + |x - 2|

theorem range_of_f : Set.range f = Set.Ici 2 :=
sorry

end range_of_f_l608_60889


namespace N_subseteq_M_l608_60849

/--
Let M = { x | ∃ n ∈ ℤ, x = n / 2 + 1 } and
N = { y | ∃ m ∈ ℤ, y = m + 0.5 }.
Prove that N is a subset of M.
-/
theorem N_subseteq_M : 
  let M := { x : ℝ | ∃ n : ℤ, x = n / 2 + 1 }
  let N := { y : ℝ | ∃ m : ℤ, y = m + 0.5 }
  N ⊆ M := sorry

end N_subseteq_M_l608_60849


namespace math_problem_l608_60884

/-- Given a function definition f(x) = 2 * x * f''(1) + x^2,
    Prove that the second derivative f''(0) is equal to -4. -/
theorem math_problem (f : ℝ → ℝ) (h1 : ∀ x, f x = 2 * x * (deriv^[2] (f) 1) + x^2) :
  (deriv^[2] f) 0 = -4 :=
  sorry

end math_problem_l608_60884


namespace andrew_total_travel_time_l608_60823

theorem andrew_total_travel_time :
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  subway_time + train_time + bike_time = 38 :=
by
  let subway_time := 10
  let train_time := 2 * subway_time
  let bike_time := 8
  sorry

end andrew_total_travel_time_l608_60823


namespace total_assembly_time_l608_60852

def chairs := 2
def tables := 2
def bookshelf := 1
def tv_stand := 1

def time_per_chair := 8
def time_per_table := 12
def time_per_bookshelf := 25
def time_per_tv_stand := 35

theorem total_assembly_time : (chairs * time_per_chair) + (tables * time_per_table) + (bookshelf * time_per_bookshelf) + (tv_stand * time_per_tv_stand) = 100 := by
  sorry

end total_assembly_time_l608_60852


namespace orange_weight_l608_60858

variable (A O : ℕ)

theorem orange_weight (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 :=
  sorry

end orange_weight_l608_60858


namespace eval_dagger_l608_60886

noncomputable def dagger (m n p q : ℕ) : ℚ := 
  (m * p) * (q / n)

theorem eval_dagger : dagger 5 16 12 5 = 75 / 4 := 
by 
  sorry

end eval_dagger_l608_60886


namespace nabla_value_l608_60831

def nabla (a b c d : ℕ) : ℕ := a * c + b * d

theorem nabla_value : nabla 3 1 4 2 = 14 :=
by
  sorry

end nabla_value_l608_60831


namespace find_m_l608_60841

-- Defining the sets and conditions
def A (m : ℝ) : Set ℝ := {1, m-2}
def B : Set ℝ := {x | x = 2}

theorem find_m (m : ℝ) (h : A m ∩ B = {2}) : m = 4 := by
  sorry

end find_m_l608_60841


namespace impossible_to_achieve_target_l608_60800

def initial_matchsticks := (1, 0, 0, 0)  -- Initial matchsticks at vertices (A, B, C, D)
def target_matchsticks := (1, 9, 8, 9)   -- Target matchsticks at vertices (A, B, C, D)

def S (a1 a2 a3 a4 : ℕ) : ℤ := a1 - a2 + a3 - a4

theorem impossible_to_achieve_target : 
  ¬∃ (f : ℕ × ℕ × ℕ × ℕ → ℕ × ℕ × ℕ × ℕ), 
    (f initial_matchsticks = target_matchsticks) ∧ 
    (∀ (a1 a2 a3 a4 : ℕ) k, 
      f (a1, a2, a3, a4) = (a1 - k, a2 + k, a3, a4 + k) ∨ 
      f (a1, a2, a3, a4) = (a1, a2 - k, a3 + k, a4 + k) ∨ 
      f (a1, a2, a3, a4) = (a1 + k, a2 - k, a3 - k, a4) ∨ 
      f (a1, a2, a3, a4) = (a1 - k, a2, a3 + k, a4 - k)) := sorry

end impossible_to_achieve_target_l608_60800


namespace notebooks_per_child_if_half_l608_60865

theorem notebooks_per_child_if_half (C N : ℕ) 
    (h1 : N = C / 8) 
    (h2 : C * N = 512) : 
    512 / (C / 2) = 16 :=
by
    sorry

end notebooks_per_child_if_half_l608_60865


namespace parabola_equation_l608_60898

theorem parabola_equation (h_axis : ∃ p > 0, x = p / 2) :
  ∃ p > 0, y^2 = -2 * p * x :=
by 
  -- proof steps will be added here
  sorry

end parabola_equation_l608_60898


namespace find_radius_l608_60814

-- Define the given values
def arc_length : ℝ := 4
def central_angle : ℝ := 2

-- We need to prove this statement
theorem find_radius (radius : ℝ) : arc_length = radius * central_angle → radius = 2 := 
by
  sorry

end find_radius_l608_60814


namespace general_term_formula_l608_60887

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d, ∀ n, a (n + 1) = a n + d

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 5 = (2 / 7) * (a 3) ^ 2) (h2 : S 7 = 63) :
  ∀ n, a n = 2 * n + 1 := by
  sorry

end general_term_formula_l608_60887


namespace necessary_but_not_sufficient_l608_60820

noncomputable def isEllipseWithFociX (a b : ℝ) : Prop :=
  ∃ (C : ℝ → ℝ → Prop), (∀ (x y : ℝ), C x y ↔ (x^2 / a + y^2 / b = 1)) ∧ (a > b ∧ a > 0 ∧ b > 0)

theorem necessary_but_not_sufficient (a b : ℝ) :
  (∀ (C : ℝ → ℝ → Prop), (∀ x y : ℝ, C x y ↔ (x^2 / a + y^2 / b = 1))
    → ((a > b ∧ a > 0 ∧ b > 0) → isEllipseWithFociX a b))
  ∧ ¬ (a > b → ∀ (C : ℝ → ℝ → Prop), (∀ x y : ℝ, C x y ↔ (x^2 / a + y^2 / b = 1)) → isEllipseWithFociX a b) :=
sorry

end necessary_but_not_sufficient_l608_60820


namespace third_trial_point_l608_60819

variable (a b : ℝ) (x₁ x₂ x₃ : ℝ)

axiom experimental_range : a = 2 ∧ b = 4
axiom method_0618 : ∀ x1 x2, (x1 = 2 + 0.618 * (4 - 2) ∧ x2 = 2 + (4 - x1)) ∨ 
                              (x1 = (2 + (4 - 3.236)) ∧ x2 = 3.236)
axiom better_result (x₁ x₂ : ℝ) : x₁ > x₂  -- Assuming better means strictly greater

axiom x1_value : x₁ = 3.236 ∨ x₁ = 2.764
axiom x2_value : x₂ = 2.764 ∨ x₂ = 3.236
axiom x3_cases : (x₃ = 4 - 0.618 * (4 - x₁)) ∨ (x₃ = 2 + (4 - x₂))

theorem third_trial_point : x₃ = 3.528 ∨ x₃ = 2.472 :=
by
  sorry

end third_trial_point_l608_60819


namespace cos_BHD_correct_l608_60836

noncomputable def cos_BHD : ℝ :=
  let DB := 2
  let DC := 2 * Real.sqrt 2
  let AB := Real.sqrt 3
  let DH := DC
  let HG := DH * Real.sin (Real.pi / 6)  -- 30 degrees in radians
  let FB := AB
  let HB := FB * Real.sin (Real.pi / 4)  -- 45 degrees in radians
  let law_of_cosines :=
    DB^2 = DH^2 + HB^2 - 2 * DH * HB * Real.cos (Real.pi / 3)
  let expected_cos := (Real.sqrt 3) / 12
  expected_cos

theorem cos_BHD_correct :
  cos_BHD = (Real.sqrt 3) / 12 :=
by
  sorry

end cos_BHD_correct_l608_60836


namespace jessica_quarters_l608_60896

theorem jessica_quarters (quarters_initial quarters_given : Nat) (h_initial : quarters_initial = 8) (h_given : quarters_given = 3) :
  quarters_initial + quarters_given = 11 := by
  sorry

end jessica_quarters_l608_60896


namespace f_bounds_l608_60888

noncomputable def f (x1 x2 x3 x4 : ℝ) := 1 - (x1^3 + x2^3 + x3^3 + x4^3) - 6 * (x1 * x2 * x3 + x1 * x2 * x4 + x1 * x3 * x4 + x2 * x3 * x4)

theorem f_bounds (x1 x2 x3 x4 : ℝ) (h : x1 + x2 + x3 + x4 = 1) :
  0 < f x1 x2 x3 x4 ∧ f x1 x2 x3 x4 ≤ 3 / 4 :=
by
  -- Proof steps go here
  sorry

end f_bounds_l608_60888


namespace range_of_a_l608_60834

theorem range_of_a (a : ℝ) :
  (∀ x, a * x^2 - x + (1 / 16 * a) > 0 → a > 2) →
  (0 < a - 3 / 2 ∧ a - 3 / 2 < 1 → 3 / 2 < a ∧ a < 5 / 2) →
  (¬ ((∀ x, a * x^2 - x + (1 / 16 * a) > 0) ∧ (0 < a - 3 / 2 ∧ a - 3 / 2 < 1))) →
  ((3 / 2 < a) ∧ (a ≤ 2)) ∨ (a ≥ 5 / 2) :=
by
  sorry

end range_of_a_l608_60834


namespace minimize_f_a_n_distance_l608_60869

noncomputable def f (x : ℝ) : ℝ :=
  2^x + Real.log x

noncomputable def a (n : ℕ) : ℝ :=
  0.1 * n

theorem minimize_f_a_n_distance :
  ∃ n : ℕ, n = 110 ∧ ∀ m : ℕ, (m > 0) -> |f (a 110) - 2012| ≤ |f (a m) - 2012| :=
by
  sorry

end minimize_f_a_n_distance_l608_60869


namespace sequence_general_term_l608_60894

theorem sequence_general_term :
  ∀ (a : ℕ → ℝ), (a 1 = 1) →
    (∀ n : ℕ, n > 0 → (Real.sqrt (a n) - Real.sqrt (a (n + 1)) = Real.sqrt (a n * a (n + 1)))) →
    (∀ n : ℕ, n > 0 → a n = 1 / (n ^ 2)) :=
by
  intros a ha1 hrec n hn
  sorry

end sequence_general_term_l608_60894


namespace new_students_count_l608_60899

-- Define the conditions as given in the problem statement.
def original_average_age := 40
def original_number_students := 17
def new_students_average_age := 32
def decreased_age := 36  -- Since the average decreases by 4 years from 40 to 36

-- Let x be the number of new students, the proof problem is to find x.
def find_new_students (x : ℕ) : Prop :=
  original_average_age * original_number_students + new_students_average_age * x = decreased_age * (original_number_students + x)

-- Prove that find_new_students(x) holds for x = 17
theorem new_students_count : find_new_students 17 :=
by
  sorry -- the proof goes here

end new_students_count_l608_60899


namespace functional_eq_solution_l608_60848

theorem functional_eq_solution (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2) :
  g 10 = -48 :=
sorry

end functional_eq_solution_l608_60848


namespace problem_a_even_triangles_problem_b_even_triangles_l608_60808

-- Definition for problem (a)
def square_divided_by_triangles_3_4_even (a : ℕ) : Prop :=
  let area_triangle := 3 * 4 / 2
  let area_square := a * a
  let k := area_square / area_triangle
  (k % 2 = 0)

-- Definition for problem (b)
def rectangle_divided_by_triangles_1_2_even (l w : ℕ) : Prop :=
  let area_triangle := 1 * 2 / 2
  let area_rectangle := l * w
  let k := area_rectangle / area_triangle
  (k % 2 = 0)

-- Theorem for problem (a)
theorem problem_a_even_triangles {a : ℕ} (h : a > 0) :
  square_divided_by_triangles_3_4_even a :=
sorry

-- Theorem for problem (b)
theorem problem_b_even_triangles {l w : ℕ} (hl : l > 0) (hw : w > 0) :
  rectangle_divided_by_triangles_1_2_even l w :=
sorry

end problem_a_even_triangles_problem_b_even_triangles_l608_60808


namespace find_missing_number_l608_60878

theorem find_missing_number
  (a b c d e : ℝ) (mean : ℝ) (f : ℝ)
  (h1 : a = 13) 
  (h2 : b = 8)
  (h3 : c = 13)
  (h4 : d = 7)
  (h5 : e = 23)
  (hmean : mean = 14.2) :
  (a + b + c + d + e + f) / 6 = mean → f = 21.2 :=
by
  sorry

end find_missing_number_l608_60878


namespace evaluate_expression_correct_l608_60857

noncomputable def evaluate_expression : ℤ :=
  6 - 8 * (9 - 4 ^ 2) * 5 + 2

theorem evaluate_expression_correct : evaluate_expression = 288 := by
  sorry

end evaluate_expression_correct_l608_60857


namespace ball_hits_ground_l608_60853

theorem ball_hits_ground : 
  ∃ t : ℚ, -4.9 * t^2 + 4 * t + 10 = 0 ∧ t = 10 / 7 :=
by sorry

end ball_hits_ground_l608_60853


namespace crayons_per_unit_l608_60862

theorem crayons_per_unit :
  ∀ (units : ℕ) (cost_per_crayon : ℕ) (total_cost : ℕ),
    units = 4 →
    cost_per_crayon = 2 →
    total_cost = 48 →
    (total_cost / cost_per_crayon) / units = 6 :=
by
  intros units cost_per_crayon total_cost h_units h_cost_per_crayon h_total_cost
  sorry

end crayons_per_unit_l608_60862


namespace compute_expression_l608_60893

-- Given Conditions
variables (a b c : ℕ)
variable (h : 2^a * 3^b * 5^c = 36000)

-- Proof Statement
theorem compute_expression (h : 2^a * 3^b * 5^c = 36000) : 3 * a + 4 * b + 6 * c = 41 :=
sorry

end compute_expression_l608_60893


namespace find_a_b_min_l608_60842

def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem find_a_b_min (a b : ℝ) :
  (∃ a b, f 1 a b = 10 ∧ deriv (f · a b) 1 = 0) →
  a = 4 ∧ b = -11 ∧ ∀ x ∈ Set.Icc (-4:ℝ) 3, f x a b ≥ f 1 4 (-11) := 
by
  -- Skipping the proof
  sorry

end find_a_b_min_l608_60842


namespace cost_of_adult_ticket_is_10_l608_60860

-- Definitions based on the problem's conditions
def num_adults : ℕ := 5
def num_children : ℕ := 2
def cost_concessions : ℝ := 12
def total_cost : ℝ := 76
def cost_child_ticket : ℝ := 7

-- Statement to prove the cost of an adult ticket being $10
theorem cost_of_adult_ticket_is_10 :
  ∃ A : ℝ, (num_adults * A + num_children * cost_child_ticket + cost_concessions = total_cost) ∧ A = 10 :=
by
  sorry

end cost_of_adult_ticket_is_10_l608_60860


namespace distance_covered_by_center_of_circle_l608_60881

-- Definition of the sides of the triangle
def side1 : ℕ := 5
def side2 : ℕ := 12
def hypotenuse : ℕ := 13

-- Definition of the circle's radius
def radius : ℕ := 2

-- Define a function that calculates the perimeter of the smaller triangle
noncomputable def smallerTrianglePerimeter (s1 s2 hyp r : ℕ) : ℕ :=
  (s1 - 2 * r) + (s2 - 2 * r) + (hyp - 2 * r)

-- Main theorem statement
theorem distance_covered_by_center_of_circle :
  smallerTrianglePerimeter side1 side2 hypotenuse radius = 18 :=
by
  sorry

end distance_covered_by_center_of_circle_l608_60881


namespace constant_term_in_quadratic_eq_l608_60847

theorem constant_term_in_quadratic_eq : 
  ∀ (x : ℝ), (x^2 - 5 * x = 2) → (∃ a b c : ℝ, a = 1 ∧ a * x^2 + b * x + c = 0 ∧ c = -2) :=
by
  sorry

end constant_term_in_quadratic_eq_l608_60847


namespace find_a_for_parallel_lines_l608_60891

def direction_vector_1 (a : ℝ) : ℝ × ℝ × ℝ :=
  (2 * a, 3, 2)

def direction_vector_2 : ℝ × ℝ × ℝ :=
  (2, 3, 2)

theorem find_a_for_parallel_lines : ∃ a : ℝ, direction_vector_1 a = direction_vector_2 :=
by
  use 1
  unfold direction_vector_1
  sorry  -- proof omitted

end find_a_for_parallel_lines_l608_60891


namespace point_P_in_second_quadrant_l608_60812

-- Define what it means for a point to lie in a certain quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- The coordinates of point P
def point_P : ℝ × ℝ := (-2, 3)

-- Prove that the point P is in the second quadrant
theorem point_P_in_second_quadrant : in_second_quadrant (point_P.1) (point_P.2) :=
by
  sorry

end point_P_in_second_quadrant_l608_60812


namespace cube_side_ratio_l608_60875

theorem cube_side_ratio (a b : ℝ) (h : (6 * a^2) / (6 * b^2) = 36) : a / b = 6 :=
by
  sorry

end cube_side_ratio_l608_60875


namespace trip_to_office_duration_l608_60895

noncomputable def distance (D : ℝ) : Prop :=
  let T1 := D / 58
  let T2 := D / 62
  T1 + T2 = 3

theorem trip_to_office_duration (D : ℝ) (h : distance D) : D / 58 = 1.55 :=
by sorry

end trip_to_office_duration_l608_60895


namespace find_e_l608_60804

-- Definitions of the problem conditions
def Q (x : ℝ) (f d e : ℝ) := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) :
  (∀ x : ℝ, Q x f d e = 3 * x^3 + d * x^2 + e * x + f) →
  (f = 9) →
  ((∃ p q r : ℝ, p + q + r = - d / 3 ∧ p * q * r = - f / 3
    ∧ 1 / (p + q + r) = -3
    ∧ 3 + d + e + f = p * q * r) →
    e = -16) :=
by
  intros hQ hf hroots
  sorry

end find_e_l608_60804


namespace polynomial_has_one_real_root_l608_60854

theorem polynomial_has_one_real_root (a : ℝ) :
  (∃! x : ℝ, x^3 - 2 * a * x^2 + 3 * a * x + a^2 - 2 = 0) :=
sorry

end polynomial_has_one_real_root_l608_60854


namespace number_of_integers_congruent_7_mod_9_lessthan_1000_l608_60855

theorem number_of_integers_congruent_7_mod_9_lessthan_1000 : 
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → 7 + 9 * n < 1000 → k + 1 = 111 :=
by
  sorry

end number_of_integers_congruent_7_mod_9_lessthan_1000_l608_60855


namespace eq_of_plane_contains_points_l608_60897

noncomputable def plane_eq (p q r : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let ⟨px, py, pz⟩ := p
  let ⟨qx, qy, qz⟩ := q
  let ⟨rx, ry, rz⟩ := r
  -- Vector pq
  let pq := (qx - px, qy - py, qz - pz)
  let ⟨pqx, pqy, pqz⟩ := pq
  -- Vector pr
  let pr := (rx - px, ry - py, rz - pz)
  let ⟨prx, pry, prz⟩ := pr
  -- Normal vector via cross product
  let norm := (pqy * prz - pqz * pry, pqz * prx - pqx * prz, pqx * pry - pqy * prx)
  let ⟨nx, ny, nz⟩ := norm
  -- Use normalized normal vector (1, 2, -2)
  (1, 2, -2, -(1 * px + 2 * py + -2 * pz))

theorem eq_of_plane_contains_points : 
  plane_eq (-2, 5, -3) (2, 5, -1) (4, 3, -2) = (1, 2, -2, -14) :=
by
  sorry

end eq_of_plane_contains_points_l608_60897


namespace sufficient_condition_for_parallel_lines_l608_60807

-- Define the condition for lines to be parallel
def lines_parallel (a b c d e f : ℝ) : Prop :=
(∃ k : ℝ, a = k * c ∧ b = k * d)

-- Define the specific lines given in the problem
def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + y - 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 5

theorem sufficient_condition_for_parallel_lines (a : ℝ) :
  (lines_parallel (a) (1) (-1) (1) (-1) (1 + 5)) ↔ (a = -1) :=
sorry

end sufficient_condition_for_parallel_lines_l608_60807


namespace find_solution_pairs_l608_60874

theorem find_solution_pairs (m n : ℕ) (t : ℕ) (ht : t > 0) (hcond : 2 ≤ m ∧ 2 ≤ n ∧ n ∣ (1 + m^(3^n) + m^(2 * 3^n))) : 
  ∃ t : ℕ, t > 0 ∧ m = 3 * t - 2 ∧ n = 3 :=
by sorry

end find_solution_pairs_l608_60874


namespace cos_graph_symmetric_l608_60832

theorem cos_graph_symmetric :
  ∃ (x0 : ℝ), x0 = (Real.pi / 3) ∧ ∀ y, (∃ x, y = Real.cos (2 * x + Real.pi / 3)) ↔ (∃ x, y = Real.cos (2 * (2 * x0 - x) + Real.pi / 3)) :=
by
  -- Let x0 = π / 3
  let x0 := Real.pi / 3
  -- Show symmetry about x = π / 3
  exact ⟨x0, by norm_num, sorry⟩

end cos_graph_symmetric_l608_60832


namespace find_cube_edge_length_l608_60815

-- Define parameters based on the problem conditions
def is_solution (n : ℕ) : Prop :=
  n > 4 ∧
  (6 * (n - 4)^2 = (n - 4)^3)

-- The main theorem statement
theorem find_cube_edge_length : ∃ n : ℕ, is_solution n ∧ n = 10 :=
by
  use 10
  sorry

end find_cube_edge_length_l608_60815


namespace find_flour_amount_l608_60844

variables (F S C : ℕ)

-- Condition 1: Proportions must remain constant
axiom proportion : 11 * S = 7 * F ∧ 7 * C = 5 * S

-- Condition 2: Mary needs 2 more cups of flour than sugar
axiom flour_sugar : F = S + 2

-- Condition 3: Mary needs 1 more cup of sugar than cocoa powder
axiom sugar_cocoa : S = C + 1

-- Question: How many cups of flour did she put in?
theorem find_flour_amount : F = 8 :=
by
  sorry

end find_flour_amount_l608_60844


namespace regular_polygon_sides_l608_60825

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l608_60825


namespace find_f_2_pow_2011_l608_60826

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_positive (x : ℝ) : x > 0 → f x > 0

axiom f_initial_condition : f 1 + f 2 = 10

axiom f_functional_equation (a b : ℝ) : f a + f b = f (a+b) - 2 * Real.sqrt (f a * f b)

theorem find_f_2_pow_2011 : f (2^2011) = 2^4023 := 
by 
  sorry

end find_f_2_pow_2011_l608_60826


namespace visibility_time_correct_l608_60867

noncomputable def visibility_time (r : ℝ) (d : ℝ) (v_j : ℝ) (v_k : ℝ) : ℝ :=
  (d / (v_j + v_k)) * (r / (r * (v_j / v_k + 1)))

theorem visibility_time_correct :
  visibility_time 60 240 4 2 = 120 :=
by
  sorry

end visibility_time_correct_l608_60867


namespace gcd_subtract_ten_l608_60835

theorem gcd_subtract_ten (a b : ℕ) (h₁ : a = 720) (h₂ : b = 90) : (Nat.gcd a b) - 10 = 80 := by
  sorry

end gcd_subtract_ten_l608_60835


namespace jessy_initial_earrings_l608_60816

theorem jessy_initial_earrings (E : ℕ) (h₁ : 20 + E + (2 / 3 : ℚ) * E + (2 / 15 : ℚ) * E = 57) : E = 20 :=
by
  sorry

end jessy_initial_earrings_l608_60816


namespace problem_1_problem_2_l608_60802

open Real

-- Step 1: Define the line and parabola conditions
def line_through_focus (k n : ℝ) : Prop := ∀ (x y : ℝ),
  y = k * (x - 1) ∧ (y = 0 → x = 1)
noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Step 2: Prove x_1 x_2 = 1 if line passes through the focus
theorem problem_1 (k n : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_line_thru_focus : line_through_focus k 1)
  (h_parabola_points : parabola x1 y1 ∧ parabola x2 y2)
  (h_intersection : y1 = k * (x1 - 1) ∧ y2 = k * (x2 - 1))
  (h_non_zero : x1 * x2 ≠ 0) :
  x1 * x2 = 1 :=
sorry

-- Step 3: Prove n = 4 if x_1 x_2 + y_1 y_2 = 0
theorem problem_2 (k n : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_line_thru_focus : line_through_focus k n)
  (h_parabola_points : parabola x1 y1 ∧ parabola x2 y2)
  (h_intersection : y1 = k * (x1 - n) ∧ y2 = k * (x2 - n))
  (h_product_relate : x1 * x2 + y1 * y2 = 0) :
  n = 4 :=
sorry

end problem_1_problem_2_l608_60802


namespace field_day_difference_l608_60806

def class_students (girls boys : ℕ) := girls + boys

def grade_students 
  (class1_girls class1_boys class2_girls class2_boys class3_girls class3_boys : ℕ) :=
  (class1_girls + class2_girls + class3_girls, class1_boys + class2_boys + class3_boys)

def diff_students (g1 b1 g2 b2 g3 b3 : ℕ) := 
  b1 + b2 + b3 - (g1 + g2 + g3)

theorem field_day_difference :
  let g3_1 := 10   -- 3rd grade first class girls
  let b3_1 := 14   -- 3rd grade first class boys
  let g3_2 := 12   -- 3rd grade second class girls
  let b3_2 := 10   -- 3rd grade second class boys
  let g3_3 := 11   -- 3rd grade third class girls
  let b3_3 :=  9   -- 3rd grade third class boys
  let g4_1 := 12   -- 4th grade first class girls
  let b4_1 := 13   -- 4th grade first class boys
  let g4_2 := 15   -- 4th grade second class girls
  let b4_2 := 11   -- 4th grade second class boys
  let g4_3 := 14   -- 4th grade third class girls
  let b4_3 := 12   -- 4th grade third class boys
  let g5_1 :=  9   -- 5th grade first class girls
  let b5_1 := 13   -- 5th grade first class boys
  let g5_2 := 10   -- 5th grade second class girls
  let b5_2 := 11   -- 5th grade second class boys
  let g5_3 := 11   -- 5th grade third class girls
  let b5_3 := 14   -- 5th grade third class boys
  diff_students (g3_1 + g3_2 + g3_3 + g4_1 + g4_2 + g4_3 + g5_1 + g5_2 + g5_3)
                (b3_1 + b3_2 + b3_3 + b4_1 + b4_2 + b4_3 + b5_1 + b5_2 + b5_3) = 3 :=
by
  sorry

end field_day_difference_l608_60806


namespace sum_of_two_digit_numbers_l608_60859

/-- Given two conditions regarding multiplication mistakes, we prove the sum of the numbers. -/
theorem sum_of_two_digit_numbers
  (A B C D : ℕ)
  (h1 : (10 * A + B) * (60 + D) = 2496)
  (h2 : (10 * A + B) * (20 + D) = 936) :
  (10 * A + B) + (10 * C + D) = 63 :=
by
  -- Conditions and necessary steps for solving the problem would go here.
  -- We're focusing on stating the problem, not the solution.
  sorry

end sum_of_two_digit_numbers_l608_60859


namespace disjoint_subsets_same_sum_l608_60818

theorem disjoint_subsets_same_sum (s : Finset ℕ) (h₁ : s.card = 10) (h₂ : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 100) :
  ∃ A B : Finset ℕ, A ⊆ s ∧ B ⊆ s ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end disjoint_subsets_same_sum_l608_60818


namespace probability_of_winning_l608_60838

theorem probability_of_winning (P_lose P_tie P_win : ℚ) (h_lose : P_lose = 5/11) (h_tie : P_tie = 1/11)
  (h_total : P_lose + P_win + P_tie = 1) : P_win = 5/11 := 
by
  sorry

end probability_of_winning_l608_60838


namespace age_difference_l608_60821

theorem age_difference (O Y : ℕ) (h₀ : O = 38) (h₁ : Y + O = 74) : O - Y = 2 := by
  sorry

end age_difference_l608_60821


namespace total_price_of_basic_computer_and_printer_l608_60873

-- Definitions for the conditions
def basic_computer_price := 2000
def enhanced_computer_price (C : ℕ) := C + 500
def printer_price (C : ℕ) (P : ℕ) := 1/6 * (C + 500 + P)

-- The proof problem statement
theorem total_price_of_basic_computer_and_printer (C P : ℕ) 
  (h1 : C = 2000)
  (h2 : printer_price C P = P) : 
  C + P = 2500 :=
sorry

end total_price_of_basic_computer_and_printer_l608_60873


namespace trigonometric_signs_l608_60866

noncomputable def terminal_side (θ α : ℝ) : Prop :=
  ∃ k : ℤ, θ = α + 2 * k * Real.pi

theorem trigonometric_signs :
  ∀ (α θ : ℝ), 
    (∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 5) ∧ terminal_side θ α →
    (Real.sin θ < 0) ∧ (Real.cos θ > 0) ∧ (Real.tan θ < 0) →
    (Real.sin θ / abs (Real.sin θ) + Real.cos θ / abs (Real.cos θ) + Real.tan θ / abs (Real.tan θ) = -1) :=
by intros
   sorry

end trigonometric_signs_l608_60866


namespace algebraic_expression_value_l608_60879

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y = 2) : 2 * x + 4 * y - 1 = 3 :=
sorry

end algebraic_expression_value_l608_60879


namespace minimize_distances_l608_60829

/-- Given points P = (6, 7), Q = (3, 4), and R = (0, m),
    find the value of m that minimizes the sum of distances PR and QR. -/
theorem minimize_distances (m : ℝ) :
  let P := (6, 7)
  let Q := (3, 4)
  ∃ m : ℝ, 
    ∀ m' : ℝ, 
    (dist (6, 7) (0, m) + dist (3, 4) (0, m)) ≤ (dist (6, 7) (0, m') + dist (3, 4) (0, m'))
:= ⟨5, sorry⟩

end minimize_distances_l608_60829


namespace relation_among_a_b_c_l608_60850

theorem relation_among_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = (3 / 5)^4)
  (h2 : b = (3 / 5)^3)
  (h3 : c = Real.log (3 / 5) / Real.log 3) :
  c < a ∧ a < b :=
by
  sorry

end relation_among_a_b_c_l608_60850


namespace train_crossing_time_l608_60876

-- Defining a structure for our problem context
structure TrainCrossing where
  length : Real -- length of the train in meters
  speed_kmh : Real -- speed of the train in km/h
  conversion_factor : Real -- conversion factor from km/h to m/s

-- Given the conditions in the problem
def trainData : TrainCrossing :=
  ⟨ 280, 50.4, 0.27778 ⟩

-- The main theorem statement:
theorem train_crossing_time (data : TrainCrossing) : 
  data.length / (data.speed_kmh * data.conversion_factor) = 20 := 
by
  sorry

end train_crossing_time_l608_60876


namespace simplify_and_evaluate_expression_l608_60839

theorem simplify_and_evaluate_expression :
  let a := 2 * Real.sin (Real.pi / 3) + 3
  (a + 1) / (a - 3) - (a - 3) / (a + 2) / ((a^2 - 6 * a + 9) / (a^2 - 4)) = Real.sqrt 3 := by
  sorry

end simplify_and_evaluate_expression_l608_60839


namespace certain_number_is_gcd_l608_60801

theorem certain_number_is_gcd (x : ℕ) (h1 : ∃ k : ℕ, 72 * 14 = k * x) (h2 : x = Nat.gcd 1008 72) : x = 72 :=
sorry

end certain_number_is_gcd_l608_60801


namespace fraction_sum_equals_l608_60861

theorem fraction_sum_equals :
  (1 / 20 : ℝ) + (2 / 10 : ℝ) + (4 / 40 : ℝ) = 0.35 :=
by
  sorry

end fraction_sum_equals_l608_60861


namespace batsman_average_l608_60890

/-- The average after 12 innings given that the batsman makes a score of 115 in his 12th innings,
     increases his average by 3 runs, and he had never been 'not out'. -/
theorem batsman_average (A : ℕ) (h1 : 11 * A + 115 = 12 * (A + 3)) : A + 3 = 82 := 
by
  sorry

end batsman_average_l608_60890


namespace range_of_a_decreasing_l608_60828

def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a_decreasing (a : ℝ) : (∀ x y : ℝ, x ∈ Set.Iic 4 → y ∈ Set.Iic 4 → x ≤ y → f x a ≥ f y a) ↔ a ≤ -3 :=
by
  sorry

end range_of_a_decreasing_l608_60828


namespace tigers_losses_l608_60827

theorem tigers_losses (L T : ℕ) (h1 : 56 = 38 + L + T) (h2 : T = L / 2) : L = 12 :=
by sorry

end tigers_losses_l608_60827


namespace mascots_arrangement_count_l608_60803

-- Define the entities
def bing_dung_dung_mascots := 4
def xue_rong_rong_mascots := 3

-- Define the conditions
def xue_rong_rong_a_and_b_adjacent := true
def xue_rong_rong_c_not_adjacent_to_ab := true

-- Theorem stating the problem and asserting the answer
theorem mascots_arrangement_count : 
  (xue_rong_rong_a_and_b_adjacent ∧ xue_rong_rong_c_not_adjacent_to_ab) →
  (number_of_arrangements = 960) := by
  sorry

end mascots_arrangement_count_l608_60803


namespace probability_class_4_drawn_first_second_l608_60892

noncomputable def P_1 : ℝ := 1 / 10
noncomputable def P_2 : ℝ := 9 / 100

theorem probability_class_4_drawn_first_second :
  P_1 = 1 / 10 ∧ P_2 = 9 / 100 := by
  sorry

end probability_class_4_drawn_first_second_l608_60892


namespace quadratic_inequality_solution_l608_60880

theorem quadratic_inequality_solution (m : ℝ) :
    (∃ x : ℝ, x^2 - m * x + 1 ≤ 0) ↔ m ≥ 2 ∨ m ≤ -2 := by
  sorry

end quadratic_inequality_solution_l608_60880


namespace jeremy_total_earnings_l608_60837

theorem jeremy_total_earnings :
  let steven_rate : ℚ := 12 / 3
  let mark_rate : ℚ := 10 / 4
  let steven_rooms : ℚ := 8 / 3
  let mark_rooms : ℚ := 9 / 4
  let steven_payment : ℚ := steven_rate * steven_rooms
  let mark_payment : ℚ := mark_rate * mark_rooms
  steven_payment + mark_payment = 391 / 24 :=
by
  let steven_rate : ℚ := 12 / 3
  let mark_rate : ℚ := 10 / 4
  let steven_rooms : ℚ := 8 / 3
  let mark_rooms : ℚ := 9 / 4
  let steven_payment : ℚ := steven_rate * steven_rooms
  let mark_payment : ℚ := mark_rate * mark_rooms
  sorry

end jeremy_total_earnings_l608_60837


namespace base7_divisible_by_19_l608_60824

theorem base7_divisible_by_19 (y : ℕ) (h : y ≤ 6) :
  (7 * y + 247) % 19 = 0 ↔ y = 0 :=
by sorry

end base7_divisible_by_19_l608_60824


namespace sin_double_angle_l608_60856

noncomputable def unit_circle_point :=
  (1 / 2, Real.sqrt (1 - (1 / 2) ^ 2))

theorem sin_double_angle 
  (α : Real)
  (h1 : (1 / 2, Real.sqrt (1 - (1 / 2) ^ 2)) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 })
  (h2 : α = (Real.arccos (1 / 2)) ∨ α = -(Real.arccos (1 / 2))) :
  Real.sin (π / 2 + 2 * α) = -1 / 2 :=
by
  sorry

end sin_double_angle_l608_60856
