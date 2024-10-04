import Mathlib

namespace permutations_of_seven_digits_l341_341408

-- Definitions based on conditions
def seven_digits := [2, 2, 5, 5, 5, 9, 3]
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The proof problem statement in Lean 4
theorem permutations_of_seven_digits :
  (factorial 7) / ((factorial (count seven_digits 2)) * (factorial (count seven_digits 5))) = 420 := by
  sorry

end permutations_of_seven_digits_l341_341408


namespace interesting_numbers_count_eq_three_l341_341677

def g1 (n : ℕ) : ℕ :=
  if n = 1 then 1
  else let prime_factors : List (ℕ × ℕ) := n.factorization.toList in
    prime_factors.foldl (λ acc pf => acc * (pf.fst + 2)^(pf.snd - 1)) 1

def gm : ℕ → ℕ → ℕ
| 1, n => g1 n
| m+1, n => g1 (gm m n)

def is_interesting (n : ℕ) : Prop :=
∀ m, gm m n > 0

def interesting_numbers_count : ℕ :=
Fin.range 500.succ.fst.countb (fun N => is_interesting (N + 1))

theorem interesting_numbers_count_eq_three : interesting_numbers_count = 3 := sorry

end interesting_numbers_count_eq_three_l341_341677


namespace third_dimension_of_box_l341_341540

theorem third_dimension_of_box (h : ℕ) (H : (151^2 - 150^2) * h + 151^2 = 90000) : h = 223 :=
sorry

end third_dimension_of_box_l341_341540


namespace limit_sine_power_l341_341659

open Real Filter

theorem limit_sine_power (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f x = (sin (2 * x) / x) ^ (1 + x)) →
  ∃ l : ℝ, tendsto (λ x : ℝ, f x) (𝓝 0) (𝓝 l) ∧ l = 2 :=
begin
  intro h_f,
  have h_sin : ∀ x, sin (2 * x) = 2 * x * sin x / x,
    from sorry,
  have h_tendsto_base : tendsto (λ x, (sin (2 * x) / x)) (𝓝 0) (𝓝 2),
    from sorry,
  have h_tendsto_exponent : tendsto (λ x, (1 + x)) (𝓝 0) (𝓝 1),
    from tendsto_add tendsto_const_nhds tendsto_id,
  rw ← h_f,
  rw ← tendsto.comp h_tendsto_base h_tendsto_exponent,
  rw pow_one,
  exact 2,
end

end limit_sine_power_l341_341659


namespace sum_of_remainders_l341_341839

open Finset

def R : Finset (Zmod 500) :=
  (range 100).image (λ n, (3 ^ n : Zmod 500))

def S : Zmod 500 :=
  R.sum id

theorem sum_of_remainders :
  S = 0 := sorry

end sum_of_remainders_l341_341839


namespace monthly_income_A_l341_341242

theorem monthly_income_A (A B C : ℝ) :
  A + B = 10100 ∧ B + C = 12500 ∧ A + C = 10400 →
  A = 4000 :=
by
  intro h
  have h1 : A + B = 10100 := h.1
  have h2 : B + C = 12500 := h.2.1
  have h3 : A + C = 10400 := h.2.2
  sorry

end monthly_income_A_l341_341242


namespace nth_power_identity_l341_341469

variable (θ : ℝ) (x : ℂ) (n : ℕ)
hypothesis (hθ : 0 < θ ∧ θ < Real.pi)
hypothesis (hx : x + 1 / x = 2 * Real.sin θ)

theorem nth_power_identity : x^n + 1 / x^n = 2 * Real.cos (n * θ + Real.pi / 2) :=
sorry

end nth_power_identity_l341_341469


namespace remainder_two_disjoint_subsets_l341_341103

theorem remainder_two_disjoint_subsets (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) :
  (let n := (3 ^ 12 - 2 * 2 ^ 12 + 1) / 2 in n % 1000 = 625) :=
by
  sorry

end remainder_two_disjoint_subsets_l341_341103


namespace steve_writes_24_pages_per_month_l341_341525

/-- Calculate the number of pages Steve writes in a month given the conditions. -/
theorem steve_writes_24_pages_per_month :
  (∃ (days_in_month : ℕ) (letter_interval : ℕ) (letter_minutes : ℕ) (page_minutes : ℕ) 
      (long_letter_factor : ℕ) (long_letter_minutes : ℕ) (total_pages : ℕ),
    days_in_month = 30 ∧ 
    letter_interval = 3 ∧ 
    letter_minutes = 20 ∧ 
    page_minutes = 10 ∧ 
    long_letter_factor = 2 ∧ 
    long_letter_minutes = 80 ∧ 
    total_pages = 24 ∧ 
    (days_in_month / letter_interval * (letter_minutes / page_minutes)
      + long_letter_minutes / (long_letter_factor * page_minutes) = total_pages)) :=
sorry

end steve_writes_24_pages_per_month_l341_341525


namespace range_of_c_proof_l341_341358

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def range_of_c (a b c : V) (ha : ‖a‖ = 3) (hb : ‖b‖ = 4) (h_ab : ⟪a, b⟫ = 0) (h_acbc : ⟪(a - c), (b - c)⟫ = 0) : set ℝ :=
{‖c‖ | 0 ≤ ‖c‖ ∧ ‖c‖ ≤ 5}

theorem range_of_c_proof (a b c : V) (ha : ‖a‖ = 3) (hb : ‖b‖ = 4) (h_ab : ⟪a, b⟫ = 0) (h_acbc : ⟪(a - c), (b - c)⟫ = 0) : range_of_c a b c ha hb h_ab h_acbc = { x | 0 ≤ x ∧ x ≤ 5 } :=
by sorry

end range_of_c_proof_l341_341358


namespace magnitude_of_sum_l341_341529

variables (a b : ℝ × ℝ)
variables (θ : ℝ)
variables (h1 : θ = π / 3) (h2 : (a.1^2 + a.2^2) = 4) (h3 : (b.1^2 + b.2^2) = 1)

theorem magnitude_of_sum (h_dot : a.1 * b.1 + a.2 * b.2 = cos θ) :
  (a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2 = 12 :=
by
  sorry

end magnitude_of_sum_l341_341529


namespace gcd_204_85_l341_341185

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := 
by sorry

end gcd_204_85_l341_341185


namespace distance_product_root_five_l341_341906

theorem distance_product_root_five (x : ℝ) (h : x ≠ 0) :
  let P := (x, 2*x + (5/x))
  let d1 := |2*x - (2*x + (5/x))| / Real.sqrt 5
  let d2 := |x|
  d1 * d2 = Real.sqrt 5 :=
by
  let P := (x, 2 * x + (5 / x))
  let d1 := abs (2 * x - (2 * x + (5 / x))) / Real.sqrt 5
  let d2 := abs x
  have h1: d1 = Real.sqrt 5 / abs x := by
    sorry
  have h2: d2 = abs x := by
    sorry
  rw [h1, h2]
  ring
  sorry

end distance_product_root_five_l341_341906


namespace K_lies_on_diagonal_AC_l341_341583

/- Define the geometrical entities and assumptions -/
variables {α : Type*} [metric_space α] [inner_product_space ℝ α]

-- Parallelogram ABCD
variables {A B C D : α}
variable [parallelogram A B C D]

-- Centers of the circles
variables {O_1 O_2 : α}
-- Point of tangency
variable {K : α}

-- Tangency conditions
variable (tangent_to_AB_AD : tangent_circle_to_sides O_1 A B A D)
variable (tangent_to_CB_CD : tangent_circle_to_sides O_2 C B C D)

-- Touching condition
variable (touching : external_tangent_circle O_1 O_2 K)

-- Prove that K lies on the diagonal AC
theorem K_lies_on_diagonal_AC (hABCD : parallelogram A B C D)
(hO1 : tangent_circle_to_sides O_1 A B A D)
(hO2 : tangent_circle_to_sides O_2 C B C D)
(hK : external_tangent_circle O_1 O_2 K) :
  K ∈ line_through A C :=
sorry

end K_lies_on_diagonal_AC_l341_341583


namespace max_area_100_max_fence_length_l341_341995

noncomputable def maximum_allowable_area (x y : ℝ) : Prop :=
  40 * x + 2 * 45 * y + 20 * x * y ≤ 3200

theorem max_area_100 (x y S : ℝ) (h : maximum_allowable_area x y) :
  S <= 100 :=
sorry

theorem max_fence_length (x y : ℝ) (h : maximum_allowable_area x y) (h1 : x * y = 100) :
  x = 15 :=
sorry

end max_area_100_max_fence_length_l341_341995


namespace brick_length_l341_341254

theorem brick_length (L : ℝ) :
  (∀ (V_wall V_brick : ℝ),
    V_wall = 29 * 100 * 2 * 100 * 0.75 * 100 ∧
    V_wall = 29000 * V_brick ∧
    V_brick = L * 10 * 7.5) →
  L = 20 :=
by
  intro h
  sorry

end brick_length_l341_341254


namespace condition_necessary_and_sufficient_l341_341731

-- Definitions for the conditions provided in the problem statement
def exp_gt (x y : ℝ) : Prop := real.exp x > real.exp y
def cuberoot_gt (x y : ℝ) : Prop := x^(1/3 : ℝ) > y^(1/3 : ℝ)

-- The theorem we need to prove
theorem condition_necessary_and_sufficient (x y : ℝ) :
  (x > y) ↔ (exp_gt x y) ∧ (cuberoot_gt x y) :=
by
  sorry

end condition_necessary_and_sufficient_l341_341731


namespace abs_expr_simplification_l341_341423

theorem abs_expr_simplification (x : ℝ) (h : x > 1) : 
  |x + Real.sqrt ((x + 2)^2)| = 2x + 2 := 
by 
  sorry

end abs_expr_simplification_l341_341423


namespace range_of_k_l341_341370

noncomputable def f (k x : ℝ) : ℝ := k * 4^x - k * 2^(x + 1) + 6 * (k - 5)

theorem range_of_k (k : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), f k x ≠ 0) → (k < 5 ∨ 6 < k) :=
by
  sorry

end range_of_k_l341_341370


namespace each_vaccine_costs_45_l341_341934

theorem each_vaccine_costs_45
    (num_vaccines : ℕ)
    (doctor_visit_cost : ℝ)
    (insurance_coverage : ℝ)
    (trip_cost : ℝ)
    (total_payment : ℝ) :
    num_vaccines = 10 ->
    doctor_visit_cost = 250 ->
    insurance_coverage = 0.80 ->
    trip_cost = 1200 ->
    total_payment = 1340 ->
    (∃ (vaccine_cost : ℝ), vaccine_cost = 45) :=
by {
    sorry
}

end each_vaccine_costs_45_l341_341934


namespace domain_fx_l341_341678

def quadratic_nonnegative (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c ≥ 0

def no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

def domain_of_function (f : ℝ → ℝ) (domain : set ℝ) : Prop :=
  ∀ x, x ∈ domain ↔ ∃ y, f y = x

theorem domain_fx :
  ∀ x : ℝ,
    (quadratic_nonnegative 1 (-5) 6 x) ∧ (no_real_roots 1 (-2) 3) ↔ x ∈ set.Iic 2 ∪ set.Ici 3 :=
begin
  sorry
end

end domain_fx_l341_341678


namespace matrix_A_pow_50_l341_341828

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, 1], ![-16, -3]]

theorem matrix_A_pow_50 :
  A ^ 50 = ![![201, 50], ![-800, -199]] :=
sorry

end matrix_A_pow_50_l341_341828


namespace fouad_age_l341_341643

theorem fouad_age (F : ℕ) (Ahmed_current_age : ℕ) (H : Ahmed_current_age = 11) (H2 : F + 4 = 2 * Ahmed_current_age) : F = 18 :=
by
  -- We do not need to write the proof steps, just a placeholder.
  sorry

end fouad_age_l341_341643


namespace plane_coloring_l341_341889

theorem plane_coloring (n : ℕ) (h : n ≥ 1) :
  ∃ (color : ℝ → ℝ → ℕ), (∀ x y, color x y = 0 ∨ color x y = 1) ∧
    (∀ (P Q : ℝ × ℝ), adjacent P Q → color P.fst P.snd ≠ color Q.fst Q.snd) :=
sorry

end plane_coloring_l341_341889


namespace range_of_m_l341_341865

def M := {y : ℝ | ∃ (x : ℝ), y = (1/2)^x}
def N (m : ℝ) := {y : ℝ | ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ y = ((1/(m-1) + 1) * (x - 1) + (|m| - 1) * (x - 2))}

theorem range_of_m (m : ℝ) : (∀ y ∈ N m, y ∈ M) ↔ -1 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l341_341865


namespace evaluate_g_at_2_l341_341783

def g (x : ℝ) : ℝ := x^3 - 2

theorem evaluate_g_at_2 : g 2 = 6 :=
by
  unfold g
  simp
  norm_num
  done

end evaluate_g_at_2_l341_341783


namespace minimum_value_of_M_l341_341011

theorem minimum_value_of_M (a1 a2 a3 : ℝ) (x y : ℝ) (h_ne_zero : ¬(a1 = 0 ∧ a2 = 0 ∧ a3 = 0)) 
                           (h_x_pos : 0 < x) (h_y_pos : 0 < y) (h_sum : x + y = 2) :
  let M := (x * a1 * a2 + y * a2 * a3) / (a1^2 + a2^2 + a3^2) in
  M ≤ (sqrt 2) / 2 :=
by
  let f := fun x y => sqrt (x^2 + y^2) / 2
  let min_f := sqrt 2 / 2
  sorry

end minimum_value_of_M_l341_341011


namespace problem_statement_l341_341687

-- Define the problem statement, given the constant 42 and known perfect squares
noncomputable def sqrt_42_minus_1_range : Prop :=
  5 < Real.sqrt 42 - 1 ∧ Real.sqrt 42 - 1 < 6

-- Proof statement
theorem problem_statement : sqrt_42_minus_1_range :=
by
  have h₁ : 36 < 42 := by norm_num
  have h₂ : 42 < 49 := by norm_num
  have h_sqrt_42 : 6 < Real.sqrt 42 := by 
    apply Real.sqrt_lt.2
    use 36
    use 42
    exact ⟨by norm_num, by norm_num⟩
  have h_sqrt_42' : Real.sqrt 42 < 7 := by
    apply Real.lt_sqrt.2
    use 42
    use 49
    exact ⟨by norm_num, by norm_num⟩
  have h_range : 5 < Real.sqrt 42 - 1 ∧ Real.sqrt 42 - 1 < 6 :=
    ⟨ by linarith, by linarith ⟩
  exact h_range

end problem_statement_l341_341687


namespace triangle_congruence_l341_341290

theorem triangle_congruence (A B C D E G F : Type) [square A B C D]
  (H1 : E ∈ line_segment B D) 
  (H2 : perpendicular (line_segment D G) (line_segment A E) at G) 
  (H3 : G ∈ line_segment A E)
  (H4 : F = intersection_point (line_segment D G) (line_segment A C)) :
  congruent (triangle C D F) (triangle D A E) :=
sorry

end triangle_congruence_l341_341290


namespace tan_double_angle_plus_pi_l341_341024

theorem tan_double_angle_plus_pi 
  (α : ℝ) 
  (h : ∀ x, tan (atan (sqrt 3 * x / x)) = sqrt 3) : 
  tan (2 * α + π) = -sqrt 3 :=
sorry

end tan_double_angle_plus_pi_l341_341024


namespace intersection_points_form_square_figure_is_square_l341_341697

noncomputable def prove_square (x y : ℝ) (h_hyperbola : x * y = 18) (h_circle : x^2 + y^2 = 36) : Prop :=
  if (x = 3 * Real.sqrt 2 ∧ y = 3 * Real.sqrt 2) ∨
     (x = -3 * Real.sqrt 2 ∧ y = -3 * Real.sqrt 2) ∨
     (x = 3 * Real.sqrt 2 ∧ y = -3 * Real.sqrt 2) ∨
     (x = -3 * Real.sqrt 2 ∧ y = 3 * Real.sqrt 2)
  then
    true
  else
    false

theorem intersection_points_form_square :
  ∀ (x y : ℝ), (x * y = 18) ∧ (x^2 + y^2 = 36) →
  (x = 3 * Real.sqrt 2 ∧ y = 3 * Real.sqrt 2) ∨ 
  (x = -3 * Real.sqrt 2 ∧ y = -3 * Real.sqrt 2) ∨ 
  (x = 3 * Real.sqrt 2 ∧ y = -3 * Real.sqrt 2) ∨ 
  (x = -3 * Real.sqrt 2 ∧ y = 3 * Real.sqrt 2) :=
by
  sorry

theorem figure_is_square :
  ∀ (a b c d : ℝ × ℝ),
    (a = (3 * Real.sqrt 2, 3 * Real.sqrt 2) ∧
     b = (-3 * Real.sqrt 2, -3 * Real.sqrt 2) ∧
     c = (3 * Real.sqrt 2, -3 * Real.sqrt 2) ∧
     d = (-3 * Real.sqrt 2, 3 * Real.sqrt 2)) →
    dist a c = dist b d
    ∧ dist a b = dist c d
    ∧ dist a d = dist b c
    ∧ ((dist a b = dist a d) → (dist a b = 6 * Real.sqrt 2))
    ∧ ((dist a c = dist b c) → (dist a c = 6 * Real.sqrt 2))
    ∧ ((dist a d = dist b d) → (dist a d = 6 * Real.sqrt 2))
    ∧ (angle a b c = Real.pi / 2 ∧
       angle a c d = Real.pi / 2 ∧
       angle b c d = Real.pi / 2 ∧
       angle c d a = Real.pi / 2)
    ∧ true :=
by
  sorry

end intersection_points_form_square_figure_is_square_l341_341697


namespace necessary_but_not_sufficient_condition_l341_341752

theorem necessary_but_not_sufficient_condition
  (a b m : ℝ) (x y : ℝ)
  (h1 : x^2 / a^2 + y^2 / b^2 = 1)
  (h2 : ∀ x, 2^x + m - 1 = 0 → m < 1)
  (h3 : ∀ m, 0 < m → m < 1 → ∀ x > 0, monotone_decreasing (λ x, log m x)) :
  (∀ x, 2^x + m - 1 = 0 → m < 1) →
  (∀ m, 0 < m → m < 1 → ∀ x > 0, monotone_decreasing (λ x, log m x)) →
  (m < 1 → 0 < m ∧ m < 1) :=
by 
  sorry

end necessary_but_not_sufficient_condition_l341_341752


namespace no_such_polynomials_exist_l341_341452

theorem no_such_polynomials_exist 
    (n : ℕ) (hn: n > 0)
    (f : Polynomial ℚ) (hf_coeffs_non_int: ¬(∀ coeff in f.coeffs, coeff ∈ ℤ))
    (g : Polynomial ℤ)
    (S : Finset ℤ) (hS_card : S.card = n + 1) :
    ¬(∀ t ∈ S, Polynomial.eval t f = Polynomial.eval t g) :=
sorry

end no_such_polynomials_exist_l341_341452


namespace cost_price_250_l341_341636

theorem cost_price_250 (C : ℝ) (h1 : 0.90 * C = C - 0.10 * C) (h2 : 1.10 * C = C + 0.10 * C) (h3 : 1.10 * C - 0.90 * C = 50) : C = 250 := 
by
  sorry

end cost_price_250_l341_341636


namespace prob_general_term_and_sum_l341_341735

noncomputable def a_n (n : ℕ) : ℤ :=
  2 * n - 1

noncomputable def b_n (n : ℕ) : ℤ :=
  (-3)^(n-1)

noncomputable def S_n (n : ℕ) : ℤ :=
  (n * (2 + (n-1) * 2)) / 2 -- sum of first n terms of an arithmetic sequence

noncomputable def T_n (n : ℕ) : ℤ :=
  (List.range n).sum (λ k => (a_n (k + 1) * b_n (k + 1)))

theorem prob_general_term_and_sum :
  (∑ k in Finset.range 4, a_n (k + 1)) - b_n 4 = 43 ∧
  a_4 + b_4 = -20 ∧
  (∀ n, a_n = 2 * n - 1 ∧ b_n = (-3)^(n-1)) ∧
  T_n = -((4 * n - 1) * (-3)^n + 1) / 8 :=
by
  sorry

end prob_general_term_and_sum_l341_341735


namespace relationship_among_abc_l341_341849

noncomputable def a : ℝ := 3 ^ 0.4
noncomputable def b : ℝ := Real.log 0.4 / Real.log 3
noncomputable def c : ℝ := 0.4 ^ 3

theorem relationship_among_abc : a > c ∧ c > b :=
by
  -- The proof is to determine this statement is correct
  sorry

end relationship_among_abc_l341_341849


namespace range_of_f_l341_341815

def custom_op (a b : ℝ) : ℝ :=
if a >= b then a else b^2

def f (x : ℝ) : ℝ :=
(custom_op 1 x) * x - (custom_op 2 x)

theorem range_of_f : Set.image f (Set.Icc (-2) 2) = Set.Icc (-4) 6 :=
sorry

end range_of_f_l341_341815


namespace weight_12m_rod_l341_341057

-- Define the weight of a 6 meters long rod
def weight_of_6m_rod : ℕ := 7

-- Given the condition that the weight is proportional to the length
def weight_of_rod (length : ℕ) : ℕ := (length / 6) * weight_of_6m_rod

-- Prove the weight of a 12 meters long rod
theorem weight_12m_rod : weight_of_rod 12 = 14 := by
  -- Calculation skipped, proof required here
  sorry

end weight_12m_rod_l341_341057


namespace volume_of_parallelepiped_l341_341062

theorem volume_of_parallelepiped
  (a b : ℝ)
  (acute_angle : ℝ)
  (h1 : acute_angle = π / 3)
  (longer_diagonal_eq_shorter_diagonal : 
    (sqrt (a^2 + b^2 + a * b)) = sqrt (a^2 + b^2 - a * b + (sqrt (2 * a * b)) ^ 2)) :
  (a * b * (sqrt (3)/2) * sqrt (2 * a * b)) = 1/2 * a * b * sqrt (6 * a * b) :=
begin
  sorry
end

end volume_of_parallelepiped_l341_341062


namespace cos_angle_C_arithmetic_seq_l341_341796

variable {R : Type*} [LinearOrderedField R]

open Real

theorem cos_angle_C_arithmetic_seq (a b c A C : R) (h1 : a + c = 2 * b) (h2 : A = 3 * C) :
  cos C = (1 + sqrt 33) / 8 :=
by sorry

end cos_angle_C_arithmetic_seq_l341_341796


namespace tangent_line_at_5_l341_341018

theorem tangent_line_at_5 
  (f : ℝ → ℝ)
  (h : tangent_line_at f (λ x, -x + 8) 5) :
  f 5 = 3 ∧ deriv f 5 = -1 :=
by
  sorry

end tangent_line_at_5_l341_341018


namespace three_digit_number_count_l341_341157

theorem three_digit_number_count :
  (∑ x in {0, 2, 4}, ∑ y in {1, 3, 5}, ∑ z in {1, 3, 5} \ {y},
     (x ≠ 0 → 2 * 1! * 2!) + (x = 0 → 1! * 2 * 2!)) = 48 :=
by
  sorry

end three_digit_number_count_l341_341157


namespace compute_perimeter_of_rectangle_l341_341572

def diameter_of_circle_diameter_q := 6
def congruent_circles {P Q R : Type*} (circle_q_diameter : ℝ) (circle_touch : ℝ): Prop :=
  circle_q_diameter = 6 ∧ circle_touch = 6

theorem compute_perimeter_of_rectangle 
  (diameter_of_circle : ℝ) 
  (rect_height : ℝ) 
  (rect_width : ℝ) 
  (touching_circles : congruent_circles diameter_of_circle rect_height ∧ rect_width = 12): 
  2 * (rect_height + rect_width) = 36 :=
by
  sorry

end compute_perimeter_of_rectangle_l341_341572


namespace original_price_of_cycle_l341_341999

variable (P : ℝ)
variable (S : ℝ := 1125)
variable (gain : ℝ := 0.25)
variable (one_hundred : ℝ := 1)

theorem original_price_of_cycle :
  S = P * (one_hundred + gain) → P = 900 :=
by
  assume h : S = P * (one_hundred + gain)
  have step1 : one_hundred + gain = 1.25 := by norm_num
  rw step1 at h
  have step2 : 1125 = P * 1.25 := by simp [h]
  have step3 : 900 = 1125 / 1.25 := by norm_num
  rw ←step3
  field_simp [h]
  ring
  exact (by norm_num : (1125 : ℝ) / 1.25 = (900 : ℝ))

end original_price_of_cycle_l341_341999


namespace remainder_of_product_l341_341494

theorem remainder_of_product (k m : ℤ) :
  let x := 315 * k + 53 in
  let y := 385 * m + 41 in
  (x * y) % 21 = 10 :=
by
  let x := 315 * k + 53
  let y := 385 * m + 41
  sorry

end remainder_of_product_l341_341494


namespace word_problems_count_l341_341568

variable (total_questions : ℕ) (add_sub_questions : ℕ) (steve_answerable : ℕ) (diff : ℕ)

theorem word_problems_count :
  total_questions = 45 →
  add_sub_questions = 28 →
  steve_answerable = 38 →
  diff = 7 →
  (total_questions - add_sub_questions) = 17 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2]
  exact sorry

end word_problems_count_l341_341568


namespace cube_root_21952_is_28_l341_341087

theorem cube_root_21952_is_28 :
  ∃ n : ℕ, n^3 = 21952 ∧ n = 28 :=
sorry

end cube_root_21952_is_28_l341_341087


namespace m_plus_n_l341_341262

noncomputable def P : ℕ × ℕ → ℚ
| (0, 0) := 1
| (x, 0) := 0
| (0, y) := 0
| (x, y) := if x > 0 ∧ y > 0 then (2/5) * P (x-1, y) + (2/5) * P (x, y-1) + (1/5) * P (x-1, y-1) else 0

open Nat

theorem m_plus_n (m n : ℕ) (h₁ : m ∈ ℕ ∧ n ∈ ℕ) (h₂ : ¬(5 ∣ m)) :
  P (5, 5) = m / 5 ^ n → m + n = ? := sorry

end m_plus_n_l341_341262


namespace problem_solution_l341_341252

def white_ball_condition (n : ℕ) : Prop :=
  ∀ (k : ℕ), let i := (k % 5) in
  2 ≤ (if i = 0 then 1 else if i = 1 then 1 else 0) +

       (if i = 1 then 1 else if i = 2 then 1 else 0) +

       (if i = 2 then 1 else if i = 3 then 1 else 0) +

       (if i = 3 then 1 else if i = 4 then 1 else 0) +

       (if i = 4 then 1 else if i = 0 then 1 else 0) ∧
  (if k < n then true else false) = 2

theorem problem_solution {n : ℕ} (h1 : n ≠ 0) (h2 : n ≠ 2021) (h3 : n ≠ 2022) (h4 : n ≠ 2023) (h5 : n ≠ 2024) :
  (¬ (n % 5 = 0)) ∧ white_ball_condition n → false :=
by {
  sorry
}

end problem_solution_l341_341252


namespace inverse_of_3_mod_199_l341_341331

theorem inverse_of_3_mod_199 : (3 * 133) % 199 = 1 :=
by
  sorry

end inverse_of_3_mod_199_l341_341331


namespace collinear_O_a_O_b_O_c_l341_341008

set_option pp.all true

noncomputable def circumcenter (A B C : Point) : Point := sorry

theorem collinear_O_a_O_b_O_c 
  (A B C L_a L_b L_c A_b A_c B_a B_c C_a C_b : Point)
  (h1 : collinear A B C) 
  (h2 : line_intersects l (BC) L_a) 
  (h3 : line_intersects l (AC) L_b) 
  (h4 : line_intersects l (AB) L_c)
  (h5 : perpendicular L_a (BC) A_b A_c) 
  (h6 : perpendicular L_b (AC) B_a B_c) 
  (h7 : perpendicular L_c (AB) C_a C_b) 
  (hO_a : O_a = circumcenter A A_b A_c)
  (hO_b : O_b = circumcenter B B_a B_c)
  (hO_c : O_c = circumcenter C C_a C_b) : 
  collinear O_a O_b O_c := 
sorry

end collinear_O_a_O_b_O_c_l341_341008


namespace find_lambda_l341_341039

noncomputable def vectorPerpendicular (a b : ℝ) (angle : ℝ) (lambda : ℝ) : Prop :=
  let dot_prod := (2 * 2 * (Real.cos angle)) - lambda in
  (0 : ℝ) = (4 + (2 * lambda - 1) * dot_prod - lambda)

theorem find_lambda
  (a b : ℝ := 2)
  (angle : ℝ := Real.pi * 2 / 3) :
  vectorPerpendicular a b angle 3 :=
by 
  sorry

end find_lambda_l341_341039


namespace speed_of_stream_l341_341260

theorem speed_of_stream (downstream_speed upstream_speed : ℕ) (h1 : downstream_speed = 12) (h2 : upstream_speed = 8) : 
  (downstream_speed - upstream_speed) / 2 = 2 :=
by
  sorry

end speed_of_stream_l341_341260


namespace range_of_g_l341_341345

noncomputable def g (x : ℝ) : ℝ :=
  (real.arccos (x / 3))^2 + (real.pi / 2) * real.arcsin (x / 3) - (real.arcsin (x / 3))^2 +
  (real.pi^2 / 18) * (x^2 + 9 * x + 27)

theorem range_of_g :
  set.range g = (set.Icc (real.pi^2 / 6) (4 * real.pi^2 / 3)) :=
sorry

end range_of_g_l341_341345


namespace cubes_with_all_three_faces_l341_341508

theorem cubes_with_all_three_faces (total_cubes red_cubes blue_cubes green_cubes: ℕ) 
  (h_total: total_cubes = 100)
  (h_red: red_cubes = 80)
  (h_blue: blue_cubes = 85)
  (h_green: green_cubes = 75) :
  40 ≤ total_cubes - ((total_cubes - red_cubes) + (total_cubes - blue_cubes) + (total_cubes - green_cubes)) ∧ (total_cubes - ((total_cubes - red_cubes) + (total_cubes - blue_cubes) + (total_cubes - green_cubes))) ≤ 75 :=
by {
  sorry
}

end cubes_with_all_three_faces_l341_341508


namespace parabola_p_value_line_through_focus_l341_341395

-- First part of the problem: Proving \( p = 2 \) given the conditions.
theorem parabola_p_value (p : ℝ) (h : (2 : ℝ)^2 = 2 * p * (1 : ℝ)) : p = 2 :=
by sorry

-- Helper predicate to define the parabola 
def is_on_parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Second part of the problem: Finding the equation of line \( l \) given the conditions.
theorem line_through_focus (k : ℝ) (h1 : k^2 = 2 / 3) : 
by sorry

end parabola_p_value_line_through_focus_l341_341395


namespace probability_of_product_divisible_by_3_l341_341348

noncomputable def probability_product_divisible_by_3 : ℚ :=
  let outcomes := 6 * 6
  let favorable_outcomes := outcomes - (4 * 4)
  favorable_outcomes / outcomes

theorem probability_of_product_divisible_by_3 (d1 d2 : Fin 6) :
  (d1 * d2) % 3 = 0 :=
sorry

end probability_of_product_divisible_by_3_l341_341348


namespace right_triangle_hypotenuse_l341_341879

noncomputable def find_BC (a b h : ℝ) : Prop :=
  let pq := 2
  let pr := 3
  let ps := 4
  (3 * a + 1.5 * b = (1 / 2) * a * b) ∧ (a^2 + b^2 = h^2) ∧ b = 2 * a ∧ h = 6 * Real.sqrt 5

theorem right_triangle_hypotenuse (a b h : ℝ) (P : point) (Q R S : foot_point) :
  P ∈ triangle_abc ∧ 
  Q.foot = perpendicular_from P to AB ∧ 
  R.foot = perpendicular_from P to BC ∧ 
  S.foot = perpendicular_from P to AC ∧
  Q.length = 2 ∧ 
  R.length = 3 ∧ 
  S.length = 4 →
  (find_BC a b h) →
  h = 6 * Real.sqrt 5 :=
sorry

end right_triangle_hypotenuse_l341_341879


namespace number_of_students_in_all_events_l341_341439

variable (T A B : ℕ)

-- Defining given conditions
-- Total number of students in the class
def total_students : ℕ := 45
-- Number of students participating in the Soccer event
def soccer_students : ℕ := 39
-- Number of students participating in the Basketball event
def basketball_students : ℕ := 28

-- Main theorem to prove
theorem number_of_students_in_all_events
  (h_total : T = total_students)
  (h_soccer : A = soccer_students)
  (h_basketball : B = basketball_students) :
  ∃ x : ℕ, x = A + B - T := sorry

end number_of_students_in_all_events_l341_341439


namespace distance_AB_l341_341473

/-- Definition of polar to Cartesian transformation for points A and B -/
structure PolarPoint :=
  (r : ℝ)
  (θ : ℝ)

def A := PolarPoint.mk 4 θ1
def B := PolarPoint.mk 5 θ2

/-- The distance AB in Cartesian coordinates given the polar coordinates and condition -/
theorem distance_AB (θ1 θ2 : ℝ) (h : θ1 - θ2 = π / 3) :
  let x1 := 4 * Real.cos θ1
  let y1 := 4 * Real.sin θ1
  let x2 := 5 * Real.cos θ2
  let y2 := 5 * Real.sin θ2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = Real.sqrt 21 :=
by
  sorry

end distance_AB_l341_341473


namespace rockets_win_probability_in_7_games_l341_341806

theorem rockets_win_probability_in_7_games :
  (∃ p, p = 3 / 4 ∧
       ( ∀ games : ℕ,
           games = 7 → 
           ( ∃ probability, 
               probability = ((\binom(6, 3) : ℚ) * (1/4)^3 * (3/4)^3) * (1/4) ∧ 
               (probability = (27 / 16384))))) := sorry

end rockets_win_probability_in_7_games_l341_341806


namespace three_inv_mod_199_l341_341329

theorem three_inv_mod_199 : ∃ x : ℤ, 3 * x ≡ 1 [MOD 199] ∧ (0 ≤ x ∧ x < 199) :=
by
  use 133
  split
  · show 3 * 133 ≡ 1 [MOD 199]
    sorry
  · split
    · show 0 ≤ 133
      linarith
    · show 133 < 199
      linarith

end three_inv_mod_199_l341_341329


namespace complement_union_M_N_eq_16_l341_341402

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subsets M and N
def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {4, 5}

-- Define the union of M and N
def unionMN : Set ℕ := M ∪ N

-- Define the complement of M ∪ N in U
def complementUnionMN : Set ℕ := U \ unionMN

-- State the theorem that the complement is {1, 6}
theorem complement_union_M_N_eq_16 : complementUnionMN = {1, 6} := by
  sorry

end complement_union_M_N_eq_16_l341_341402


namespace sugar_needed_in_two_minutes_l341_341577

-- Let a be the amount of sugar needed per chocolate bar.
def sugar_per_chocolate_bar : ℝ := 1.5

-- Let b be the number of chocolate bars produced per minute.
def chocolate_bars_per_minute : ℕ := 36

-- Let t be the time in minutes.
def time_in_minutes : ℕ := 2

theorem sugar_needed_in_two_minutes : 
  let sugar_in_one_minute := chocolate_bars_per_minute * sugar_per_chocolate_bar
  let total_sugar := sugar_in_one_minute * time_in_minutes
  total_sugar = 108 := by
  sorry

end sugar_needed_in_two_minutes_l341_341577


namespace distance_run_by_worker_l341_341618

/-- A construction worker sets a delayed explosion fuse scheduled to go off in 40 seconds.
    He starts running at a speed of 5 yards per second after setting the fuse.
    The speed of sound is 1080 feet per second.
    Prove that the distance in yards that the construction worker has run when he hears the
    explosion is 203 yards.
-/
theorem distance_run_by_worker  : 
  let time_heard := 40 + 43200 / 1065 in
  let distance_yds := (15 * time_heard) / 3 in
  distance_yds = 203 :=
by
  sorry

end distance_run_by_worker_l341_341618


namespace zebra_difference_is_zebra_l341_341869

/-- 
A zebra number is a non-negative integer in which the digits strictly alternate between even and odd.
Given two 100-digit zebra numbers, prove that their difference is still a 100-digit zebra number.
-/
theorem zebra_difference_is_zebra 
  (A B : ℕ) 
  (hA : (∀ i, (A / 10^i % 10) % 2 = i % 2) ∧ (A / 10^100 = 0) ∧ (A > 10^99))
  (hB : (∀ i, (B / 10^i % 10) % 2 = i % 2) ∧ (B / 10^100 = 0) ∧ (B > 10^99)) 
  : (∀ j, (((A - B) / 10^j) % 10) % 2 = j % 2) ∧ ((A - B) / 10^100 = 0) ∧ ((A - B) > 10^99) :=
sorry

end zebra_difference_is_zebra_l341_341869


namespace range_of_x_l341_341791

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  x - 4 * real.sqrt y = 2 * real.sqrt (x - y)

-- Define the mathematical statement to prove
theorem range_of_x (x y : ℝ) (h : conditions x y) : x ∈ ({0} ∪ set.Icc 4 20) :=
sorry

end range_of_x_l341_341791


namespace ellipse_equation_slope_sum_zero_max_quadrilateral_area_l341_341010

theorem ellipse_equation (P : ℝ × ℝ) (a b : ℝ) (h1 : P = (sqrt 3, 1/2)) (h2 : a > b) (h3 : b > 0) 
  (ellipse_eq : (P.1)^2 / a^2 + (P.2)^2 / b^2 = 1) :  
  a = 2 ∧ b = 1 ∧ (∀ x y : ℝ, x^2 / 4 + y^2 = 1 → (∃ F : ℝ × ℝ, F.1 = sqrt 3 ∧ F.2 = 0 ∧ F.1 = sqrt(3))) := 
  sorry

theorem slope_sum_zero (A B C : ℝ × ℝ) (k_AB k_BC : ℝ) 
  (h1 : ∀ P, (P.1)^2 / 4 + (P.2)^2 = 1) (h2 : ∀ P, k_AB = if P.1 ≠ 0 then P.2 / P.1 else 0) (h3 : ∀ P, k_BC = if P.1 ≠ 0 then P.2 / P.1 else 0)
  (h_intersect : ∀ A B C, (A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 0 ∧ B.2 = 0 ∨ C.1 = 0 ∧ C.2 = 0)) 
  (h_dot : A.1 * B.1 + A.2 * B.2 = 5 * A.2 * B.2) :
  k_AB + k_BC = 0 := 
  sorry

theorem max_quadrilateral_area (A B C D : ℝ × ℝ) 
  (h1 : ∀ P, (P.1)^2 / 4 + (P.2)^2 = 1) (h2 : ∃ A B C D : ℝ × ℝ, A.1 * B.1 + A.2 * B.2 = 5 * A.2 * B.2) :
  ∃ max_area : ℝ, max_area = 4 := 
  sorry

end ellipse_equation_slope_sum_zero_max_quadrilateral_area_l341_341010


namespace sum_of_divisors_of_47_l341_341956

theorem sum_of_divisors_of_47 : 
  ∑ d in {1, 47}, d = 48 := 
by 
  sorry

end sum_of_divisors_of_47_l341_341956


namespace find_multiple_of_nada_l341_341644

def multiple_of_nadas_money_john_has (total_money ali_less_than_nada john_money : ℕ) (k : ℕ) : Prop :=
  ∃ N : ℕ, total_money = N + (N - ali_less_than_nada) + john_money ∧ john_money = k * N

theorem find_multiple_of_nada (h : multiple_of_nadas_money_john_has 67 5 48 4) : ∃ N, 48 = 4 * N :=
by apply h; sorry

end find_multiple_of_nada_l341_341644


namespace cone_height_l341_341256

theorem cone_height (r : ℝ) (n : ℕ) (sec_r : ℝ) (h : ℝ) (slant_height : ℝ) ( base_radius : ℝ):
  (r = 10) → (n = 4) → (sec_r = (2 * r * Real.pi) / n) → (slant_height = r) →
  (base_radius = sec_r / (2 * Real.pi)) → 
  (h = Real.sqrt (slant_height ^ 2 - base_radius ^ 2)) →
  h = Real.sqrt 93.75 := 
by {
  intros r_val n_val sec_r_val slant_height_val base_radius_val height_eq,
  exact height_eq,
  sorry
}

end cone_height_l341_341256


namespace sin_cos_monotonic_increasing_interval_l341_341534

theorem sin_cos_monotonic_increasing_interval : 
  ∃ (a b : ℝ), a = -π / 8 ∧ b = 3 * π / 8 ∧
  ∀ x y : ℝ, (a ≤ x ∧ x < y ∧ y ≤ b) → (sin (2 * x) - cos (2 * x) ≤ sin (2 * y) - cos (2 * y)) := by
  sorry

end sin_cos_monotonic_increasing_interval_l341_341534


namespace value_of_x_plus_y_l341_341464

variables {x y : ℝ}

theorem value_of_x_plus_y (h1 : y = 4 * ⌊x⌋ + 1) (h2 : y = 2 * ⌊x + 3⌋ + 7) (h3 : x ∉ ℤ) :
  31 < x + y ∧ x + y < 32 :=
by
  sorry

end value_of_x_plus_y_l341_341464


namespace dan_balloons_l341_341709

theorem dan_balloons (fred_balloons sam_balloons total_balloons dan_balloons : ℕ) 
  (h₁ : fred_balloons = 10) 
  (h₂ : sam_balloons = 46) 
  (h₃ : total_balloons = 72) : 
  dan_balloons = total_balloons - (fred_balloons + sam_balloons) :=
by
  sorry

end dan_balloons_l341_341709


namespace sum_of_remainders_l341_341838

open Finset

def R : Finset (Zmod 500) :=
  (range 100).image (λ n, (3 ^ n : Zmod 500))

def S : Zmod 500 :=
  R.sum id

theorem sum_of_remainders :
  S = 0 := sorry

end sum_of_remainders_l341_341838


namespace equations_of_motion_and_speed_of_L_l341_341308

-- Define the conditions
def omega : ℝ := 10 -- angular velocity in rad/s
def OA : ℝ := 90   -- length of OA in cm
def AB : ℝ := 90   -- length of AB in cm
def AL : ℝ := OA / 3 -- length of AL which is 1/3 of AB

-- Translate the mathematical expressions
noncomputable def x_L (t : ℝ) : ℝ := AL * Real.cos (omega * t)
noncomputable def y_L (t : ℝ) : ℝ := AL * Real.sin (omega * t)

-- Define the speed function for point L
noncomputable def v_L (t : ℝ) : ℝ :=
  Real.sqrt ((-AL * omega * Real.sin (omega * t))^2 + (AL * omega * Real.cos (omega * t))^2)

-- The theorem we need to prove
theorem equations_of_motion_and_speed_of_L (t : ℝ) :
  (x_L t = 30 * Real.cos (10 * t)) ∧
  (y_L t = 30 * Real.sin (10 * t)) ∧
  (v_L t = 300) :=
by {
  sorry
}

end equations_of_motion_and_speed_of_L_l341_341308


namespace ravi_overall_profit_l341_341885

-- Define the purchase prices
def refrigerator_purchase_price := 15000
def mobile_phone_purchase_price := 8000

-- Define the percentages
def refrigerator_loss_percent := 2
def mobile_phone_profit_percent := 10

-- Define the calculations for selling prices
def refrigerator_loss_amount := (refrigerator_loss_percent / 100) * refrigerator_purchase_price
def refrigerator_selling_price := refrigerator_purchase_price - refrigerator_loss_amount

def mobile_phone_profit_amount := (mobile_phone_profit_percent / 100) * mobile_phone_purchase_price
def mobile_phone_selling_price := mobile_phone_purchase_price + mobile_phone_profit_amount

-- Define the total purchase and selling prices
def total_purchase_price := refrigerator_purchase_price + mobile_phone_purchase_price
def total_selling_price := refrigerator_selling_price + mobile_phone_selling_price

-- Define the overall profit calculation
def overall_profit := total_selling_price - total_purchase_price

-- Statement of the theorem
theorem ravi_overall_profit :
  overall_profit = 500 := by
  sorry

end ravi_overall_profit_l341_341885


namespace one_divisor_of_factorial_ten_is_ten_times_factorial_nine_l341_341410

theorem one_divisor_of_factorial_ten_is_ten_times_factorial_nine :
  let f10 := 10!
  let f9 := 9!
  let d := 10 * f9
  (d ∣ f10 ∧ ∀ e, (e ∣ f10 ∧ e = 10 * f9) → e = d) →
  ∃! d, d ∣ 10! ∧ d = 10 * 9! :=
by
  sorry

end one_divisor_of_factorial_ten_is_ten_times_factorial_nine_l341_341410


namespace maximum_true_statements_l341_341107

-- Define the conditions and the statements
variables {a b : ℤ} (hab : a < b) (ha_neg : a < 0) (hb_neg : b < 0)

-- Statements identified from the problem
def statement1 : Prop := (1 : ℚ) / a < (1 : ℚ) / b
def statement2 : Prop := a^2 > b^2
def statement3 : Prop := a < b
def statement4 : Prop := a < 0
def statement5 : Prop := b < 0
def statement6 : Prop := |a| > |b|

-- Final theorem to prove that at most 5 statements can be true simultaneously
theorem maximum_true_statements : ¬statement1 ∧ statement2 ∧ statement3 ∧ statement4 ∧ statement5 ∧ statement6 :=
by
  sorry

end maximum_true_statements_l341_341107


namespace vision_data_approx_l341_341165

theorem vision_data_approx (L V : ℝ) (hL : L = 5 + Real.log10 V) (hL_value : L = 4.9) :
  V ≈ 0.8 :=
by
  -- Define the approximation constant
  let c : ℝ := 1.259
  -- Assume the given approximate value
  assume h_c : Real.rpow (10 : ℝ) (1 / 10) = c
  -- Begin proof (skipped)
  sorry

end vision_data_approx_l341_341165


namespace range_of_a_l341_341479

theorem range_of_a (a : ℝ) :
  let A := (-2, 3)
  let B := (0, a)
  let circle := (x + 3)^2 + (y + 2)^2 = 1
  ∃ a ∈ (1/3 : ℝ)..(3/2 : ℝ), intersects_symmetrical_line (A, B, circle) := 
sorry

end range_of_a_l341_341479


namespace num_two_digit_values_satisfying_condition_l341_341101

def digit_sum (x : ℕ) : ℕ := x.digitsSum

def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99

theorem num_two_digit_values_satisfying_condition :
  {x : ℕ // is_two_digit x ∧ digit_sum (digit_sum x) = 4 }.card = 9 :=
by
  sorry

end num_two_digit_values_satisfying_condition_l341_341101


namespace probability_drawing_diamond_not_ace_and_ace_is_24_over_1275_l341_341630

noncomputable def probability_diamond_non_ace_first_and_ace_second (cards : Finset (card))
  (remaining_deck : Finset (card)) 
  (num_diamonds_not_ace : Nat)
  (num_aces : Nat)
  (total_remaining_cards : Nat)
  (Conditional_probability_diamond_not_ace : ℚ)
  (Conditional_probability_ace_after_drawing_diamond_not_ace : ℚ) : ℚ :=
  Conditional_probability_diamond_not_ace * Conditional_probability_ace_after_drawing_diamond_not_ace

theorem probability_drawing_diamond_not_ace_and_ace_is_24_over_1275:
  let num_diamonds_not_ace := 12 in
  let num_aces := 4 in
  let total_remaining_cards := 51 in
  let total_remaining_cards_after_one_drawn := 50 in
  let Conditional_probability_diamond_not_ace := 12 / 51 in
  let Conditional_probability_ace_after_drawing_diamond_not_ace := 4 / 50 in
  probability_diamond_non_ace_first_and_ace_second cards remaining_deck num_diamonds_not_ace num_aces total_remaining_cards Conditional_probability_diamond_not_ace Conditional_probability_ace_after_drawing_diamond_not_ace = 24 / 1275 :=
by sorry

end probability_drawing_diamond_not_ace_and_ace_is_24_over_1275_l341_341630


namespace value_of_expression_l341_341775

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end value_of_expression_l341_341775


namespace percent_calculation_l341_341609

-- Define the conditions
def part : ℝ := 6.2
def whole : ℝ := 1000.0

-- Define the expected percentage as the solution
def expected_percent : ℝ := 0.62

-- Define the formula to compute the percentage
def compute_percent (p w : ℝ) : ℝ := (p / w) * 100

-- Statement of the problem to be proved
theorem percent_calculation : compute_percent part whole = expected_percent := 
by 
  -- The proof is omitted
  sorry

end percent_calculation_l341_341609


namespace real_part_proof_l341_341124

noncomputable def real_part_of_fraction (z : ℂ) (h : |z| = 2 ∧ z.im ≠ 0) : ℝ :=
  let x := z.re in
  let y := z.im in
  (2 - x) / (8 - 4 * x + x^2)

theorem real_part_proof (z : ℂ) (h : |z| = 2 ∧ z.im ≠ 0) :
  (real_part_of_fraction z h) = (2 - z.re) / (8 - 4 * z.re + z.re^2) :=
by
  sorry

end real_part_proof_l341_341124


namespace cos_2pi_over_3_minus_alpha_l341_341420

theorem cos_2pi_over_3_minus_alpha (α : ℝ) (h : sin (π / 6 - α) = 2 / 3) :
  cos (2 * π / 3 - α) = - (2 / 3) :=
sorry

end cos_2pi_over_3_minus_alpha_l341_341420


namespace log_sum_equals_four_l341_341801

-- Assuming the conditions
axiom exists_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (h_pos: ∀ n, a n > 0) :
  (∀ n, a (n + 1) = a n * r) ∧ a 3 * a 11 = 16

-- The theorem we want to prove
theorem log_sum_equals_four {a : ℕ → ℝ} {r : ℝ} (h_seq: ∀ n, a (n + 1) = a n * r) 
  (h_pos: ∀ n, a n > 0) (h_prod: a 3 * a 11 = 16) :
  Real.log 2 (a 2) + Real.log 2 (a 12) = 4 := 
by
  sorry

end log_sum_equals_four_l341_341801


namespace tan_half_angle_product_l341_341771

theorem tan_half_angle_product (a b : ℝ) (h : 3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ (x : ℝ), x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) := 
sorry

end tan_half_angle_product_l341_341771


namespace Shiela_neighbors_l341_341158

theorem Shiela_neighbors (total_drawings drawings_per_neighbor : ℕ) (h1 : total_drawings = 54) (h2 : drawings_per_neighbor = 9) :
  ∃ n : ℕ, total_drawings = drawings_per_neighbor * n ∧ n = 6 :=
by {
  use 6,
  split,
  { rw [h1, h2],
    exact rfl },
  { exact rfl }
}

end Shiela_neighbors_l341_341158


namespace four_digit_numbers_with_sum_20_l341_341707

def is_valid_digit (a b c d : ℕ) : Prop :=
  a >= 1 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧ c >= 0 ∧ c <= 9 ∧ d >= 0 ∧ d <= 9 ∧ a + b + c + d = 20

theorem four_digit_numbers_with_sum_20 : 
  (∃ a b c d : ℕ, is_valid_digit a b c d) = 12 := 
sorry

end four_digit_numbers_with_sum_20_l341_341707


namespace sum_of_first_45_natural_numbers_l341_341979

theorem sum_of_first_45_natural_numbers : (45 * (45 + 1)) / 2 = 1035 := by
  sorry

end sum_of_first_45_natural_numbers_l341_341979


namespace find_teacher_age_l341_341902

/-- Given conditions: 
1. The class initially has 30 students with an average age of 10.
2. One student aged 11 leaves the class.
3. The average age of the remaining 29 students plus the teacher is 11.
Prove that the age of the teacher is 30 years.
-/
theorem find_teacher_age (total_students : ℕ) (avg_age : ℕ) (left_student_age : ℕ) 
  (remaining_avg_age : ℕ) (teacher_age : ℕ) :
  total_students = 30 →
  avg_age = 10 →
  left_student_age = 11 →
  remaining_avg_age = 11 →
  289 + teacher_age = 29 * remaining_avg_age + teacher_age →
  teacher_age = 30 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end find_teacher_age_l341_341902


namespace ellipse_C_eq_lambda_value_max_area_triangle_l341_341070

-- Definitions of conditions as given in the problem
def ellipse_C (x y : ℝ) : Prop := (x^2)/4 + y^2 = 1
def ellipse_E (x y : ℝ) : Prop := (x^2)/16 + (y^2)/4 = 1

variables (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ)
def point_on_C : Prop := ellipse_C x₀ y₀
def line_through_P (k m : ℝ) : Prop := ∀ x y, y = k * x + m → ellipse_E x y
def point_Q (λ : ℝ) : Prop := ∀ (x y : ℝ), λ = 2 ∧ ellipse_E (-λ * x₀) (-λ * y₀)

-- Proof statements
theorem ellipse_C_eq : ellipse_C x y ↔ (x^2)/4 + y^2 = 1 := sorry

theorem lambda_value : point_Q 2 := sorry

theorem max_area_triangle : ∃ S, S = 6 * real.sqrt 3 := sorry

end ellipse_C_eq_lambda_value_max_area_triangle_l341_341070


namespace min_value_formula_l341_341737

variable (a b : ℝ)

noncomputable def geometric_mean_condition := 8^a * 2^b = 2

theorem min_value_formula
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : geometric_mean_condition a b) : 
  ∃ x, x = 5 + 2 * Real.sqrt 6 ∧ (∀ y, y = (1 / a + 2 / b) → y ≥ x) :=
sorry

end min_value_formula_l341_341737


namespace even_function_property_l341_341759

variable (f : ℝ → ℝ)

def h (x : ℝ) : ℝ := f x + x + 2

theorem even_function_property (h_even : ∀ x, h x = h (-x))
    (f_value_at_2 : f 2 = 3) : f (-2) = 7 :=
by
  have h_2_eq : h 2 = f 2 + 2 + 2 := rfl
  rw [f_value_at_2] at h_2_eq
  have h_2_value : h 2 = 7 := by rw [h_2_eq, add_comm, add_assoc]
  have h_minus_2_eq : h (-2) = h 2 := h_even 2
  rw [h_2_value] at h_minus_2_eq
  have h_minus_2_value : h (-2) = f (-2) - 2 + 2 := rfl
  rw [h_minus_2_value] at h_minus_2_eq
  rw [add_comm, add_assoc] at h_minus_2_eq
  exact h_minus_2_eq.symm.trans sorry

end even_function_property_l341_341759


namespace correct_values_of_k_l341_341072

noncomputable def integer_point_solution_sets (k : ℤ) : Prop :=
    let x := (-(k + 2) : ℤ) / (k - 1) in
    x - 2 = k * x + k ∧ (∀ k ∈ {0, 2, 4, -2}, (x, x - 2) ∈ ℤ × ℤ)

theorem correct_values_of_k (k : ℤ) : 
  integer_point_solution_sets k ↔ k = 0 ∨ k = 2 ∨ k = 4 ∨ k = -2 := by sorry

end correct_values_of_k_l341_341072


namespace pa_pb_value_l341_341816

noncomputable def polar_curve_rect_eq : ℝ × ℝ → Prop :=
λ (p : ℝ × ℝ), (p.1 - 1) ^ 2 + (p.2) ^ 2 = 1

def parametric_line_eq : ℝ → ℝ × ℝ :=
λ t, (⟨sqrt 3 t / 2, -1 + t / 2⟩)

def polar_to_cartesian : ℝ × ℝ → ℝ × ℝ
| (ρ, θ) := (ρ * cos θ, ρ * sin θ)

def point_P := polar_to_cartesian (1, 3 * real.pi / 2)

def intersection_points : ℝ × ℝ → ℝ → ℝ → Prop :=
λ (P : ℝ × ℝ) (t1 t2 : ℝ), 
  let A := parametric_line_eq t1,
      B := parametric_line_eq t2 in
  polar_curve_rect_eq A ∧ polar_curve_rect_eq B ∧
  ((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2) = t1 ∧
  ((P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2) = t2

theorem pa_pb_value 
  (P : ℝ × ℝ) 
  (t1 t2 : ℝ) 
  (H : intersection_points P t1 t2) : 
  (|t1 + 1| + |t2 + 1|) = sqrt 3 + 3 := sorry

end pa_pb_value_l341_341816


namespace travelTimeDifference_l341_341213

def departureArrivalTimes := {train_109T := (19 * 60 + 33, 34 * 60 + 26), train_1461 := (11 * 60 + 58, 32 * 60 + 1)}

theorem travelTimeDifference (d_a : departureArrivalTimes) :
  let T1_start := d_a.train_109T.1
  let T1_end := d_a.train_109T.2
  let T2_start := d_a.train_1461.1
  let T2_end := d_a.train_1461.2
  let duration_T1 := T1_end - T1_start
  let duration_T2 := T2_end - T2_start
  duration_T1 = 893 ∧ duration_T2 = 1203 →
  abs (duration_T2 - duration_T1) = 310 := 
by
  intros
  sorry

end travelTimeDifference_l341_341213


namespace peaches_after_7_days_l341_341992

def initial_peaches := 18
def initial_ripe := 4
def initial_unripe := initial_peaches - initial_ripe
def peaches_ripen_each_day (n : ℕ) := 2 + n
def peaches_eaten_each_day (n : ℕ) := n + 1

theorem peaches_after_7_days :
  let 
    rec peaches (day: ℕ): ℕ × ℕ :=
      match day with
      | 0 => (initial_ripe, initial_unripe)
      | n + 1 =>
        let (ripe, unripe) := peaches n
        let ripe_today := peaches_ripen_each_day n
        let eaten_today := peaches_eaten_each_day n
        let new_ripe := ripe + (if ripe_today ≤ unripe then ripe_today else unripe) - eaten_today
        let new_unripe := unripe - (if ripe_today ≤ unripe then ripe_today else unripe)
        (if new_ripe < 0 then 0 else new_ripe, if new_unripe < 0 then 0 else new_unripe)
  in peaches 7 = (0, 0) :=
by sorry

end peaches_after_7_days_l341_341992


namespace polynomial_inequality_l341_341829

noncomputable def P : ℝ[X] := sorry -- Polynomial with real coefficients
noncomputable def Q : ℝ[X] := sorry -- Polynomial with real coefficients
noncomputable def S (x : ℝ[X]) := P * Q -- Polynomial with integer coefficients

lemma P0_positive : P.coeff 0 > 0 := sorry 
lemma S_integer_coefficients : ∀ n, (S P).coeff n ∈ ℤ := sorry 

theorem polynomial_inequality (x : ℝ) (hx : x > 0) :
  S (x^2) - (S x)^2 ≤ (1/4) * (P (x^3)^2 + Q (x^3)) :=
sorry

end polynomial_inequality_l341_341829


namespace math_problem_l341_341493

def f (x : ℝ) : ℝ := 2 * x - Real.cos x

noncomputable def arithmetic_sequence (a_1 : ℝ) : ℕ → ℝ
| 0     => a_1
| (n+1) => a_1 + (n : ℝ) * (Real.pi / 8)

theorem math_problem 
  (a_1 : ℝ) 
  (h_sum : f (arithmetic_sequence a_1 0) + f (arithmetic_sequence a_1 1) + f (arithmetic_sequence a_1 2) + f (arithmetic_sequence a_1 3) + f (arithmetic_sequence a_1 4) = 5 * Real.pi) :
  (f (arithmetic_sequence a_1 2)) ^ 2 - (arithmetic_sequence a_1 0) * (arithmetic_sequence a_1 4) = 13 * (Real.pi ^ 2) / 16 := 
  sorry

end math_problem_l341_341493


namespace triangle_is_isosceles_if_sin2A_eq_sin2B_triangle_is_right_if_tanA_tanB_eq_1_triangle_is_isosceles_if_a_eq_2c_cosB_l341_341846

-- Lean statement for Statement A
theorem triangle_is_isosceles_if_sin2A_eq_sin2B
  (A B C : ℝ) (a b c : ℝ) (angle_A : A = a) (angle_B : B = b) (angle_C : C = c)
  (tri_sum : A + B + C = π) (sin2A_eq_sin2B : sin (2 * A) = sin (2 * B)) :
  a = b :=
sorry

-- Lean statement for Statement B
theorem triangle_is_right_if_tanA_tanB_eq_1
  (A B : ℝ) (tanA_tanB_eq_1 : tan A * tan B = 1) :
  A + B = π / 2 :=
sorry

-- Lean statement for Statement C
theorem triangle_is_isosceles_if_a_eq_2c_cosB
  (A B C : ℝ) (a b c : ℝ) (angle_A : A = a) (angle_B : B = b) (angle_C : C = c)
  (tri_sum : A + B + C = π) (a_eq_2c_cosB : a = 2 * c * cos B) :
  a = c :=
sorry

end triangle_is_isosceles_if_sin2A_eq_sin2B_triangle_is_right_if_tanA_tanB_eq_1_triangle_is_isosceles_if_a_eq_2c_cosB_l341_341846


namespace length_of_LC_l341_341998

noncomputable def edge_length : ℝ := 12
noncomputable def initial_fill_fraction : ℝ := 5 / 8
noncomputable def liquid_height : ℝ := initial_fill_fraction * edge_length
noncomputable def area_cross_section : ℝ := edge_length * liquid_height
noncomputable def LC (KB : ℝ) : ℝ := 2 * KB

theorem length_of_LC (KB : ℝ) (h : LC KB * edge_length / 2 = area_cross_section) : LC KB = 2 * 5 := by
  have h1 : liquid_height = 7.5 := by sorry
  have h2 : area_cross_section = 90 := by sorry
  have h3 : edge_length = 12 := by sorry
  have h4 : 3 * KB * 12 / 2 = 90 := by sorry
  have KB_res : KB = 5 := by sorry
  show LC KB = 10 from sorry

end length_of_LC_l341_341998


namespace two_point_questions_count_l341_341597

theorem two_point_questions_count (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 :=
sorry

end two_point_questions_count_l341_341597


namespace partition_into_clique_independent_set_l341_341669

variable {V : Type} [Fintype V] (E : V → V → Prop) [DecidableRel E]

/-- 
The main theorem stating that given any assembly of n people,
if any subset of 4 people contains 3 who either all know each other 
or 3 who do not know each other, then it is possible to partition 
them into a clique and an independent set. 
-/
theorem partition_into_clique_independent_set
  (h : ∀ (A : Finset V), A.card = 4 → 
       ∃ (B : Finset V), B ⊆ A ∧ B.card = 3 ∧ (∀ (x y ∈ B), E x y) ∨ (∀ (x y ∈ B), ¬E x y)) :
  ∃ (C D : Finset V), (∀ (x y ∈ C, E x y) ∧ ∀ (x y ∈ D, ¬E x y) ∧ C ∪ D = Finset.univ) :=
sorry

end partition_into_clique_independent_set_l341_341669


namespace number_of_factors_of_M_l341_341462

theorem number_of_factors_of_M : 
  let M := 35^5 + 5 * 35^4 + 10 * 35^3 + 10 * 35^2 + 5 * 35 + 1 in
  ∃ (n : ℕ), (count (M.factors) = n ∧ n = 121) := 
by {
  sorry
}

end number_of_factors_of_M_l341_341462


namespace range_of_values_for_a_l341_341485

open Real

def point_A : Point := (-2, 3)

def point_B (a : ℝ) : Point := (0, a)

def circle_center : Point := (-3, -2)

noncomputable def symmetric_line_distance (a : ℝ) : ℝ :=
  let A := 3 - a
  let B := -2
  let C := 2 * a
  let x_0 := -3
  let y_0 := -2
  abs (A * x_0 + B * y_0 + C) / sqrt (A ^ 2 + B ^ 2)

def valid_range : set ℝ := set.Icc (1 / 3) (3 / 2)

theorem range_of_values_for_a (a : ℝ) : symmetric_line_distance a ≤ 1 → a ∈ valid_range :=
  sorry

end range_of_values_for_a_l341_341485


namespace smallest_possible_degree_p_l341_341535

theorem smallest_possible_degree_p (p : Polynomial ℝ) :
  (∀ x, 0 < |x| → ∃ C, |((3 * x^7 + 2 * x^6 - 4 * x^3 + x - 5) / (p.eval x)) - C| < ε)
  → (Polynomial.degree p) ≥ 7 := by
  sorry

end smallest_possible_degree_p_l341_341535


namespace find_rate_of_interest_l341_341284

theorem find_rate_of_interest (P R : ℝ) (H1: 17640 = P * (1 + R / 100)^2) (H2: 21168 = P * (1 + R / 100)^3) : R = 6.27 :=
by 
  sorry

end find_rate_of_interest_l341_341284


namespace number_of_elements_complement_intersection_l341_341013

open Set

-- Definitions of sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {0, 1, 2, 4, 5}
def U : Set ℕ := A ∪ B

-- Statement to prove
theorem number_of_elements_complement_intersection :
  (U \ (A ∩ B)).toFinset.card = 3 :=
by
  sorry

end number_of_elements_complement_intersection_l341_341013


namespace appliance_costs_l341_341641

noncomputable def costs_equiv : Prop :=
  let D := 2000 / 10.5 in
  let W := 3 * D in
  let R := 2 * W in
  let M := D / 2 in
  D + W + R + M = 2000

theorem appliance_costs :
  let D := 2000 / 10.5 in
  let W := 3 * D in
  let R := 2 * W in
  let M := D / 2 in
  D = 2000 / 10.5 ∧ W = 3 * D ∧ R = 2 * W ∧ M = D / 2 ∧ D + W + R + M = 2000 :=
by
  let D := 2000 / 10.5
  let W := 3 * D
  let R := 2 * W
  let M := D / 2
  exact ⟨rfl, rfl, rfl, rfl, by norm_num⟩

end appliance_costs_l341_341641


namespace circle_area_increase_l341_341053

theorem circle_area_increase (r : ℝ) :
  let A_initial := Real.pi * r^2
  let A_new := Real.pi * (2*r)^2
  let delta_A := A_new - A_initial
  let percentage_increase := (delta_A / A_initial) * 100
  percentage_increase = 300 := by
  sorry

end circle_area_increase_l341_341053


namespace number_of_squares_sharing_two_vertices_l341_341100

-- Given conditions
def right_isosceles_triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (AB BC AC : ℝ),
  AB = BC ∧
  angle A B C = π / 2 ∧
  AB ^ 2 + BC ^ 2 = AC ^ 2

-- Desired proof problem
theorem number_of_squares_sharing_two_vertices
  (A B C : ℝ × ℝ) (h : right_isosceles_triangle A B C) :
  ∃ n, n = 2 :=
begin
  sorry
end

end number_of_squares_sharing_two_vertices_l341_341100


namespace napkin_coloring_l341_341518

structure Napkin where
  top : ℝ
  bottom : ℝ
  left : ℝ
  right : ℝ

def intersects_vertically (n1 n2 : Napkin) : Prop :=
  n1.left ≤ n2.right ∧ n2.left ≤ n1.right

def intersects_horizontally (n1 n2 : Napkin) : Prop :=
  n1.bottom ≤ n2.top ∧ n2.bottom ≤ n1.top

def can_be_crossed_by_line (n1 n2 : Napkin) : Prop :=
  intersects_vertically n1 n2 ∨ intersects_horizontally n1 n2

theorem napkin_coloring
  (blue_napkins green_napkins : List Napkin)
  (h_cross : ∀ (b : Napkin) (g : Napkin), 
    b ∈ blue_napkins → g ∈ green_napkins → can_be_crossed_by_line b g) :
  ∃ (color : String) (h1 h2 : ℝ) (v : ℝ), 
    (color = "blue" ∧ ∀ b ∈ blue_napkins, (b.bottom ≤ h1 ∧ h1 ≤ b.top) ∨ (b.bottom ≤ h2 ∧ h2 ≤ b.top) ∨ (b.left ≤ v ∧ v ≤ b.right)) ∨
    (color = "green" ∧ ∀ g ∈ green_napkins, (g.bottom ≤ h1 ∧ h1 ≤ g.top) ∨ (g.bottom ≤ h2 ∧ h2 ≤ g.top) ∨ (g.left ≤ v ∧ v ≤ g.right)) :=
sorry

end napkin_coloring_l341_341518


namespace f_le_g_for_a_eq_neg1_l341_341753

noncomputable def f (a : ℝ) (b : ℝ) (x : ℝ) : ℝ :=
  (a * x + b) * Real.exp x

noncomputable def g (t : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * x - Real.log x + t

theorem f_le_g_for_a_eq_neg1 (t : ℝ) :
  let b := 3
  ∃ x ∈ Set.Ioi 0, f (-1) b x ≤ g t x ↔ t ≤ Real.exp 2 - 1 / 2 :=
by
  sorry

end f_le_g_for_a_eq_neg1_l341_341753


namespace area_above_line_is_zero_l341_341218

-- Define the circle: (x - 4)^2 + (y - 5)^2 = 12
def circle (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 12

-- Define the line: y = x - 2
def line (x y : ℝ) : Prop := y = x - 2

-- Prove that the area of the circle above this line is 0
theorem area_above_line_is_zero : 
  ∀ x y : ℝ, circle x y ∧ y > x - 2 → false := sorry

end area_above_line_is_zero_l341_341218


namespace pen_cost_l341_341521

theorem pen_cost
  (p q : ℕ)
  (h1 : 6 * p + 5 * q = 380)
  (h2 : 3 * p + 8 * q = 298) :
  p = 47 :=
sorry

end pen_cost_l341_341521


namespace find_n_l341_341857

theorem find_n (n : ℕ) (h_pos : n > 0) (h_ineq : n < Real.sqrt 65 ∧ Real.sqrt 65 < n + 1) : n = 8 := by sorry

end find_n_l341_341857


namespace factorize_expression_l341_341325

theorem factorize_expression (x : ℝ) : x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 :=
by sorry

end factorize_expression_l341_341325


namespace expenditure_ratio_l341_341923

variable (P1 P2 : Type)
variable (I1 I2 E1 E2 : ℝ)
variable (R_incomes : I1 / I2 = 5 / 4)
variable (S1 S2 : ℝ)
variable (S_equal : S1 = S2)
variable (I1_fixed : I1 = 4000)
variable (Savings : S1 = 1600)

theorem expenditure_ratio :
  (I1 - E1 = 1600) → 
  (I2 * 4 / 5 - E2 = 1600) →
  I2 = 3200 →
  E1 / E2 = 3 / 2 :=
by
  intro P1_savings P2_savings I2_calc
  -- proof steps go here
  sorry

end expenditure_ratio_l341_341923


namespace area_of_W_l341_341192

def W (x y : ℝ) : Prop := (|x| + |4 - |y|| - 4)^2 ≤ 4

theorem area_of_W : measure (λ (p : ℝ × ℝ), W p.1 p.2) = 120 := 
sorry

end area_of_W_l341_341192


namespace gcd_204_85_l341_341181

theorem gcd_204_85 : Nat.gcd 204 85 = 17 :=
sorry

end gcd_204_85_l341_341181


namespace petya_cannot_win_l341_341200

theorem petya_cannot_win (n : ℕ) (h : n ≥ 3) : ¬ ∃ strategy : ℕ → ℕ → Prop, 
  (∀ k, strategy k (k+1) ∧ strategy k (k-1))
  ∧ ∀ m, ¬ strategy n m :=
sorry

end petya_cannot_win_l341_341200


namespace last_digit_of_2_pow_2010_l341_341875

theorem last_digit_of_2_pow_2010 : (2 ^ 2010) % 10 = 4 :=
by
  sorry

end last_digit_of_2_pow_2010_l341_341875


namespace optionA_optionB_optionC_optionD_l341_341971

variables {α : Type*} [nontrivial α] [linear_ordered_ring α] (a b x y c : α)

-- Define each transformation as a Lean theorem
theorem optionA (h : a = b) : a * c = b * c :=
by rw [h]

theorem optionB (h : a * (x^2 + 1) = b * (x^2 + 1)) (hx : x^2 + 1 ≠ 0) : a = b :=
by simpa using (mul_left_cancel₀ hx h)

theorem optionC (h : a = b) (hc : c ≠ 0) : a / (c^2) = b / (c^2) :=
by rw [h]

theorem optionD (h : x = y) : x - 3 = y - 3 :=
by rw [h]

end optionA_optionB_optionC_optionD_l341_341971


namespace probability_correct_l341_341599

noncomputable def probability_of_getting_number_greater_than_4 : ℚ :=
  let favorable_outcomes := 2
  let total_outcomes := 6
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_getting_number_greater_than_4 = 1 / 3 := by sorry

end probability_correct_l341_341599


namespace money_spent_l341_341768

def initial_money (Henry : Type) : ℤ := 11
def birthday_money (Henry : Type) : ℤ := 18
def final_money (Henry : Type) : ℤ := 19

theorem money_spent (Henry : Type) : (initial_money Henry + birthday_money Henry - final_money Henry = 10) := 
by sorry

end money_spent_l341_341768


namespace quadratic_eq_real_roots_l341_341432

theorem quadratic_eq_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 4 * x + 2 = 0) →
  (∃ y : ℝ, a * y^2 - 4 * y + 2 = 0) →
  a ≤ 2 ∧ a ≠ 0 :=
by sorry

end quadratic_eq_real_roots_l341_341432


namespace common_root_sum_k_l341_341951

theorem common_root_sum_k :
  (∃ x : ℝ, (x^2 - 4 * x + 3 = 0) ∧ (x^2 - 6 * x + k = 0)) → 
  (∃ (k₁ k₂ : ℝ), (k₁ = 5) ∧ (k₂ = 9) ∧ (k₁ + k₂ = 14)) :=
by
  sorry

end common_root_sum_k_l341_341951


namespace variable_swap_l341_341245

theorem variable_swap (x y t : Nat) (h1 : x = 5) (h2 : y = 6) (h3 : t = x) (h4 : x = y) (h5 : y = t) : 
  x = 6 ∧ y = 5 := 
by
  sorry

end variable_swap_l341_341245


namespace correct_formulas_l341_341405

noncomputable def S (a x : ℝ) := (a^x - a^(-x)) / 2
noncomputable def C (a x : ℝ) := (a^x + a^(-x)) / 2

variable {a x y : ℝ}

axiom h1 : a > 0
axiom h2 : a ≠ 1

theorem correct_formulas : S a (x + y) = S a x * C a y + C a x * S a y ∧ S a (x - y) = S a x * C a y - C a x * S a y :=
by 
  sorry

end correct_formulas_l341_341405


namespace sum_of_digits_base_3_333_l341_341313

theorem sum_of_digits_base_3_333 : 
  let n := 333 in 
  let digit_sum := (1:ℕ) + 1 + 0 + 1 + 0 + 0 in 
  sum_of_digits_base_3 n = digit_sum := 
by 
  -- Proof here (skipped with sorry)
  sorry

-- auxiliary definition for sum of digits in base 3
def sum_of_digits_base_3 (n : ℕ) : ℕ := 
  let rec digitsum (x : ℕ) (sum : ℕ) : ℕ :=
    if x = 0 then sum else digitsum (x / 3) (sum + x % 3) 
  in digitsum n 0

end sum_of_digits_base_3_333_l341_341313


namespace main_theorem_l341_341460

-- Definition of S and L
def S (m n : ℕ) : set (ℕ × ℕ) := 
  { p | let (x, y) := p in 1 ≤ x ∧ x ≤ m + n - 1 ∧ 1 ≤ y ∧ y ≤ m + n - 1 ∧ m + 1 ≤ x + y ∧ x + y ≤ 2 * m + n - 1 }

def L (m n : ℕ) : set (set (ℕ × ℕ)) :=
  { l | ∃ k : ℕ, (∀ x, (x, k) ∈ l) ∨ (∀ y, (k, y) ∈ l) ∨ (∀ p : ℕ × ℕ, p.1 + p.2 = k ∧ p ∈ l) }

-- Structure of a subset T of S
structure subset_T (m n : ℕ) :=
  (T : set (ℕ × ℕ))
  (ht_cons : T ⊆ S m n)
  (ht_lines : ∀ l ∈ L m n, ∃ p ∈ l, p ∈ T)

-- Main theorem statement
theorem main_theorem (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∃ t : subset_T m n, ∀ l ∈ L m n, (t.T ∩ l).to_finset.card % 2 = 1) ↔ 
  (n + m) % 4 = 1 ∨ (n - m) % 4 = 0 :=
sorry

end main_theorem_l341_341460


namespace abc_divisibility_l341_341003

theorem abc_divisibility (a b c : ℕ) (h₁ : a ∣ (b * c - 1)) (h₂ : b ∣ (c * a - 1)) (h₃ : c ∣ (a * b - 1)) : 
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 1 ∧ b = 1 ∧ ∃ n : ℕ, n ≥ 1 ∧ c = n) :=
by
  sorry

end abc_divisibility_l341_341003


namespace angle_opposite_c_zero_l341_341064

theorem angle_opposite_c_zero (a b c : ℝ) 
  (h : (a + b + c) * (a + b - c) = 4 * a * b) : 
  ∠C = 0 :=
sorry

end angle_opposite_c_zero_l341_341064


namespace tangent_line_eq_at_a1_f_above_g_in_interval_1_2_max_h_in_interval_neg1_1_l341_341390

section
variable (a : ℝ)

def f (x : ℝ) : ℝ := x^3 - 3 * a * x
def g (x : ℝ) : ℝ := Real.log x
def t (x : ℝ) : ℝ := g x - f x
def h (x : ℝ) := |f x|

theorem tangent_line_eq_at_a1 : 
  t 1 = 2 ∧ (∃ m : ℝ, t 1 = 1 ∧ t' 1 = m) → x - y + 1 = 0 := 
sorry

theorem f_above_g_in_interval_1_2 : 
  (∀ x ∈ (Set.Icc 1 2), f x ≥ g x) → a ≤ (1 / 3) := 
sorry

theorem max_h_in_interval_neg1_1 :
  F a = max (h 0) (h 1) → 
  F a = 
    if a ≤ (1 / 4) then 1 - 3 * a
    else if (1 / 4) < a ∧ a < 1 then 2 * a * Real.sqrt a
    else 3 * a - 1 := 
sorry

end

end tangent_line_eq_at_a1_f_above_g_in_interval_1_2_max_h_in_interval_neg1_1_l341_341390


namespace problem_1_problem_2_l341_341744

-- Conditions for the problem
def domain_Q := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def set_P (a : ℝ) := {x : ℝ | a + 1 ≤ x ∧ x ≤ 2 * a + 1}

-- Proof problems translated to Lean statements

-- (1) If a = 3, determine the intersection of the complement of P in the real numbers with Q
theorem problem_1 : 
  let a := 3 in
  let P := set_P a in 
  ∀ x : ℝ, x ∈ (set.univ \ P) ∩ domain_Q ↔ -2 ≤ x ∧ x < 4 :=
by sorry

-- (2) If P ⊆ Q, find the range of the real number a.
theorem problem_2 : 
  ∀ a : ℝ, 
  (set_P a ⊆ domain_Q) → a ∈ set.Iic 2 :=
by sorry

end problem_1_problem_2_l341_341744


namespace range_of_m_solve_inequality_l341_341034

open Real Set

noncomputable def f (x: ℝ) := -abs (x - 2)
noncomputable def g (x: ℝ) (m: ℝ) := -abs (x - 3) + m

-- Problem 1: Prove the range of m given the condition
theorem range_of_m (h : ∀ x : ℝ, f x > g x m) : m < 1 :=
  sorry

-- Problem 2: Prove the set of solutions for f(x) + a - 1 > 0
theorem solve_inequality (a : ℝ) :
  (if a = 1 then {x : ℝ | x ≠ 2}
   else if a > 1 then univ
   else {x : ℝ | x < 1 + a} ∪ {x : ℝ | x > 3 - a}) = {x : ℝ | f x + a - 1 > 0} :=
  sorry

end range_of_m_solve_inequality_l341_341034


namespace convex_quadrilaterals_count_l341_341937

theorem convex_quadrilaterals_count : combinatorial.choose 12 4 = 495 := by
  sorry

end convex_quadrilaterals_count_l341_341937


namespace range_of_a_l341_341421

def f (a x : ℝ) : ℝ :=
if x > 1 then a^x else (4 - a / 2) * x + 2

theorem range_of_a (a : ℝ) (h1 : ∀ x y : ℝ, x < y → f a x ≤ f a y) :
  4 ≤ a ∧ a < 8 :=
sorry

end range_of_a_l341_341421


namespace distance_focus_asymptote_hyperbola_l341_341761

theorem distance_focus_asymptote_hyperbola :
  ∀ (x y : ℝ),
  (x, y) = (5, 0) →
  (∃ a b : ℝ, a = 3 ∧ b = 4 ∧ 3*x + 4*y = 0 ∨ 3*x - 4*y = 0) →
  (∃ k l L : ℝ, L = 3 * k + 4 * l ∧ ∀ d : ℝ, d = |3 * 5 + 4 * 0| / real.sqrt (3^2 + 4^2) ∧ d = 3) :=
begin
  sorry
end

end distance_focus_asymptote_hyperbola_l341_341761


namespace arithmetic_sequence_length_l341_341411

theorem arithmetic_sequence_length 
  (a1 : ℤ) (d : ℤ) (an : ℤ) (n : ℤ) 
  (h_start : a1 = -5) 
  (h_diff : d = 5) 
  (h_end : an = 50) 
  (h_formula : an = a1 + (n - 1) * d) : 
  n = 12 := 
by
  -- Given conditions
  have hs : a1 = -5 := h_start,
  have hd : d = 5 := h_diff,
  have ha : an = 50 := h_end,
  -- Use the arithmetic sequence formula
  have hf : an = a1 + (n - 1) * d := h_formula,
  -- Solve for n
  sorry

end arithmetic_sequence_length_l341_341411


namespace real_part_proof_l341_341125

noncomputable def real_part_of_fraction (z : ℂ) (h : |z| = 2 ∧ z.im ≠ 0) : ℝ :=
  let x := z.re in
  let y := z.im in
  (2 - x) / (8 - 4 * x + x^2)

theorem real_part_proof (z : ℂ) (h : |z| = 2 ∧ z.im ≠ 0) :
  (real_part_of_fraction z h) = (2 - z.re) / (8 - 4 * z.re + z.re^2) :=
by
  sorry

end real_part_proof_l341_341125


namespace parabola_axis_l341_341028

section
variable (x y : ℝ)

-- Condition: Defines the given parabola equation.
def parabola_eq (x y : ℝ) : Prop := x = (1 / 4) * y^2

-- The Proof Problem: Prove that the axis of this parabola is x = -1/2.
theorem parabola_axis (h : parabola_eq x y) : x = - (1 / 2) := 
sorry
end

end parabola_axis_l341_341028


namespace find_monic_cubic_polynomial_l341_341338

/-- There exists a monic cubic polynomial Q(x) with integer coefficients such that Q(∛3 + 1) = 0
    and Q(x) = x^3 - 3x^2 + 3x - 4. -/
theorem find_monic_cubic_polynomial :
  ∃ Q : Polynomial ℤ, Polynomial.monic Q ∧ Q.eval (Real.cbrt 3 + 1) = 0 ∧
    Q = Polynomial.X^3 - 3 * Polynomial.X^2 + 3 * Polynomial.X - 4 :=
by
  sorry

end find_monic_cubic_polynomial_l341_341338


namespace sum_of_integers_square_256_sum_of_integers_solution_l341_341560

theorem sum_of_integers_square_256 (x : ℤ) (h : x^2 = x + 256) : x = 16 ∨ x = -16 :=
by
  sorry

theorem sum_of_integers_solution : (16 + (-16) = 0) : Prop :=
by
  exact rfl

end sum_of_integers_square_256_sum_of_integers_solution_l341_341560


namespace strict_decreasing_l341_341178

-- Define the function f and its properties
variable (f : ℝ → ℝ)

-- Condition 1: f(x + y) ≤ f(x) + f(y) for any x, y ∈ ℝ
def condition1 := ∀ (x y : ℝ), f(x + y) ≤ f(x) + f(y)

-- Condition 2: f(x) < 0 for any x > 0
def condition2 := ∀ (x : ℝ), x > 0 → f(x) < 0

-- The theorem we want to prove: f is strictly decreasing
theorem strict_decreasing (h1 : condition1 f) (h2 : condition2 f) : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f(x₁) > f(x₂) := by
  sorry

end strict_decreasing_l341_341178


namespace fraction_product_cube_l341_341216

theorem fraction_product_cube :
  ((5 : ℚ) / 8)^3 * ((4 : ℚ) / 9)^3 = (125 : ℚ) / 5832 :=
by
  sorry

end fraction_product_cube_l341_341216


namespace maximal_n_odd_even_solution_equality_l341_341246

-- Define the main equation and its properties
def main_equation (n : ℕ) (xs : Fin n → ℕ) : Prop :=
  (∑ i in Finset.range n, (i+1) * xs ⟨i, Fin.is_lt i n⟩) = 2017

-- Define property of x_i being positive integers
def positive_integers (n : ℕ) (xs : Fin n → ℕ) : Prop :=
  ∀ i, 0 < xs i

-- Main conjecture stating the existence of maximal n
theorem maximal_n : 
  ∃ n xs, main_equation n xs ∧ positive_integers n xs ∧ n = 63 := by
  sorry

-- Define the set A_n for positive integer solutions
def A_n (n : ℕ) : Set (Fin n → ℕ) :=
  {xs | main_equation n xs ∧ positive_integers n xs}

-- Define odd and even solutions
def odd_solutions (n : ℕ) : Set (Fin n → ℕ) :=
  if n % 2 = 1 then A_n n else ∅

def even_solutions (n : ℕ) : Set (Fin n → ℕ) :=
  if n % 2 = 0 then A_n n else ∅

-- Prove that the number of odd solutions equals the number of even solutions
theorem odd_even_solution_equality :
  ∃ n, ∀ (n : ℕ), (n % 2 = 1 → |odd_solutions n| = |even_solutions n + 1|) ∧ 
                   (n % 2 = 0 → |even_solutions n| = |odd_solutions n + 1|) := by
  sorry

end maximal_n_odd_even_solution_equality_l341_341246


namespace total_loss_l341_341654

theorem total_loss (P : ℝ) (A : ℝ) (L : ℝ) (h1 : A = (1/9) * P) (h2 : 603 = (P / (A + P)) * L) : 
  L = 670 :=
by
  sorry

end total_loss_l341_341654


namespace find_angle_B_l341_341653

variable (α : Type) [AddGroup α] [LinearOrder α] [OrderedAddCommGroup α]

def angle := {x : α // 0 ≤ x ∧ x ≤ 180}

theorem find_angle_B (angle1 angle2 angle3 : angle) (B : α)
  (h1 : angle2.val = 180 - B)
  (h2 : angle3.val = 180 - angle1.val) :
  B = 30 := 
  sorry

end find_angle_B_l341_341653


namespace slant_height_of_cone_l341_341190

-- Definitions of given conditions
def lateral_area (L : ℝ) := L = 10 * Real.pi
def radius (r : ℝ) := r = 2

-- Definition of base circumference
def base_circumference (r : ℝ) := 2 * Real.pi * r

-- Definition of lateral area in terms of slant height
def lateral_area_formula (r l lateral_area_ : ℝ) := lateral_area_ = 0.5 * base_circumference(r) * l

-- The theorem to prove the length of the slant height
theorem slant_height_of_cone (L : ℝ) (r : ℝ) (l : ℝ)
  (hL : lateral_area L) (hr : radius r) (hL_formula : lateral_area_formula r l L) :
  l = 5 := by
  sorry

end slant_height_of_cone_l341_341190


namespace find_common_ratio_l341_341074

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

-- Given conditions
lemma a2_eq_8 (a₁ q : ℝ) : geometric_sequence a₁ q 2 = 8 :=
by sorry

lemma a5_eq_64 (a₁ q : ℝ) : geometric_sequence a₁ q 5 = 64 :=
by sorry

-- The common ratio q
theorem find_common_ratio (a₁ q : ℝ) (hq : 0 < q) :
  (geometric_sequence a₁ q 2 = 8) → (geometric_sequence a₁ q 5 = 64) → q = 2 :=
by sorry

end find_common_ratio_l341_341074


namespace intersection_of_P_and_Q_l341_341765

open Set

theorem intersection_of_P_and_Q :
  let P := {0, 2, 4, 6} : Set ℕ
  let Q := {x : ℕ | x ≤ 3}
  P ∩ Q = {0, 2} :=
by
  let P := {0, 2, 4, 6} : Set ℕ
  let Q := {x : ℕ | x ≤ 3}
  sorry

end intersection_of_P_and_Q_l341_341765


namespace three_times_diameter_l341_341255

theorem three_times_diameter (A : ℝ) (π : ℝ) (r : ℝ) (d : ℝ) :
  A = 16 * π → π * r^2 = A → d = 2 * r → 3 * d = 24 :=
by
  intros hA hπ hr
  rw [hA, hπ] at *,
  sorry

end three_times_diameter_l341_341255


namespace sqrt_sum_bounds_l341_341461

theorem sqrt_sum_bounds (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
    4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2 - b)^2) + 
                   Real.sqrt (b^2 + (2 - c)^2) + 
                   Real.sqrt (c^2 + (2 - d)^2) + 
                   Real.sqrt (d^2 + (2 - a)^2) ∧
    Real.sqrt (a^2 + (2 - b)^2) + 
    Real.sqrt (b^2 + (2 - c)^2) + 
    Real.sqrt (c^2 + (2 - d)^2) + 
    Real.sqrt (d^2 + (2 - a)^2) ≤ 8 :=
sorry

end sqrt_sum_bounds_l341_341461


namespace range_of_a1_l341_341743

theorem range_of_a1 (a : ℕ → ℕ) (S : ℕ → ℕ) (h_seq : ∀ n, 12 * S n = 4 * a (n + 1) + 5^n - 13)
  (h_S4 : ∀ n, S n ≤ S 4):
  13 / 48 ≤ a 1 ∧ a 1 ≤ 59 / 64 :=
sorry

end range_of_a1_l341_341743


namespace sqrt_monotonic_increasing_l341_341594

theorem sqrt_monotonic_increasing (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) :
  sqrt x < sqrt y :=
by sorry

end sqrt_monotonic_increasing_l341_341594


namespace total_students_l341_341569

theorem total_students (groups students_per_group : ℕ) (h : groups = 6) (k : students_per_group = 5) :
  groups * students_per_group = 30 := 
by
  sorry

end total_students_l341_341569


namespace amy_tips_calculation_l341_341649

theorem amy_tips_calculation 
  (hourly_wage : ℝ) (hours_worked : ℝ) (total_earnings : ℝ) 
  (h_wage : hourly_wage = 2)
  (h_hours : hours_worked = 7)
  (h_total : total_earnings = 23) : 
  total_earnings - (hourly_wage * hours_worked) = 9 := 
sorry

end amy_tips_calculation_l341_341649


namespace ratio_of_unit_prices_l341_341194

theorem ratio_of_unit_prices (price_A : ℝ) (sheets_A : ℕ) (price_B : ℝ) (sheets_B : ℕ)
  (hA : price_A = 3 ∧ sheets_A = 2) (hB : price_B = 2 ∧ sheets_B = 3) :
  (price_A / sheets_A) / (price_B / sheets_B) = 9 / 4 :=
by
  -- conditions
  rcases hA with ⟨hA₁, hA₂⟩
  rcases hB with ⟨hB₁, hB₂⟩
  -- calculations
  have h1: (price_A / sheets_A) = 3 / 2 := by rw [hA₁, hA₂, div_eq_mul_inv, ← mul_assoc, mul_inv_cancel (two_ne_zero), one_mul]
  have h2: (price_B / sheets_B) = 2 / 3 := by rw [hB₁, hB₂, div_eq_mul_inv, ← mul_assoc, mul_inv_cancel (three_ne_zero), one_mul]
  rw [h1, h2, div_div, div_eq_mul_inv, mul_comm (3 / 2), mul_assoc (3 / 2), inv_mul_cancel (ne_of_gt (show 2/3 > 0 by norm_num))]
  norm_num -- calculates 9/4
  exact rfl
  -- sorry

end ratio_of_unit_prices_l341_341194


namespace sum_of_solutions_sum_of_integers_satisfying_eq_l341_341546

theorem sum_of_solutions (x : ℤ) (h : x^2 = x + 256) : x = 16 ∨ x = -16 := sorry

theorem sum_of_integers_satisfying_eq : ∑ x in finset.filter (λ x, x^2 = x + 256) (finset.Icc -16 16), x = 0 := 
begin
  sorry
end

end sum_of_solutions_sum_of_integers_satisfying_eq_l341_341546


namespace rectangle_inscription_l341_341083

theorem rectangle_inscription (circle : Type) (A B : circle) :
  ∃ n : ℕ ∪ {∞}, n = number_of_possible_rectangles circle A B 
    ∧ n >= 0 
    ∧ n <= ∞ := 
sorry

end rectangle_inscription_l341_341083


namespace eccentricity_of_specific_ellipse_l341_341020

noncomputable def eccentricity_of_ellipse 
  (a b c : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_ab : a > b)
  (E : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
  (line : ∀ x y, x - 2 * y + 2 = 0 → (x, y) = (a, 0) ∨ x^2 - c^2 = a^2 - b^2) : 
  ℝ :=
begin
  let e := c / a,
  have h1 : (b^2 / (c^2) = 1 / 4), from sorry,
  have h2 : (a^2 / c^2 = 5 / 4), from sorry,
  have h3 : c / a = 2 * real.sqrt 5 / 5, from sorry,
  exact e
end

theorem eccentricity_of_specific_ellipse 
  (ecc : eccentricity_of_ellipse 5 3 2 (by norm_num) (by norm_num) (by norm_num) (by { intros, sorry }) 
  (by { intros, sorry })) 
  (h_eq : ecc = 2 * real.sqrt 5 / 5) : Prop :=
begin
  exact ecc = 2 * real.sqrt 5 / 5
end

end eccentricity_of_specific_ellipse_l341_341020


namespace complex_number_real_imag_equal_l341_341054

theorem complex_number_real_imag_equal (a : ℝ) (h : (a + 6) = (3 - 2 * a)) : a = -1 :=
by
  sorry

end complex_number_real_imag_equal_l341_341054


namespace sum_of_integers_squared_greater_sum_of_integers_satisfying_condition_l341_341550

theorem sum_of_integers_squared_greater (x : ℤ) (hx : x ^ 2 = 256 + x) : x = 16 ∨ x = -16 :=
begin
  have : x ^ 2 - x - 256 = 0,
  { rw ← hx, ring },
  apply quadratic_eq_zero_iff.mpr,
  use [16, -16],
  split;
  linarith,
end

theorem sum_of_integers_satisfying_condition : ∑ x in ({16, -16} : finset ℤ), x = 0 :=
by simp

end sum_of_integers_squared_greater_sum_of_integers_satisfying_condition_l341_341550


namespace binom_9_5_equals_126_l341_341305

theorem binom_9_5_equals_126 : Nat.binom 9 5 = 126 := 
by 
  sorry

end binom_9_5_equals_126_l341_341305


namespace cannot_form_62_cents_with_six_coins_l341_341893

-- Define the coin denominations and their values
structure Coin :=
  (value : ℕ)
  (count : ℕ)

def penny : Coin := ⟨1, 6⟩
def nickel : Coin := ⟨5, 6⟩
def dime : Coin := ⟨10, 6⟩
def quarter : Coin := ⟨25, 6⟩
def halfDollar : Coin := ⟨50, 6⟩

-- Define the main theorem statement
theorem cannot_form_62_cents_with_six_coins :
  ¬ (∃ (p n d q h : ℕ),
      p + n + d + q + h = 6 ∧
      1 * p + 5 * n + 10 * d + 25 * q + 50 * h = 62) :=
sorry

end cannot_form_62_cents_with_six_coins_l341_341893


namespace Bolyai_János_reading_ways_l341_341069

open Nat

def ways_to_read_Bolyai_János : ℕ :=
  binomial 10 5

theorem Bolyai_János_reading_ways :
  ways_to_read_Bolyai_János = 252 := by
  sorry

end Bolyai_János_reading_ways_l341_341069


namespace general_formula_an_sum_first_n_terms_cn_l341_341736

-- Define sequences and conditions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n, a (n + 1) = a n + d
def geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop := ∀ n, b (n + 1) = b n * r

variables (a : ℕ → ℤ) (b : ℕ → ℤ)

-- Given conditions
axiom a_is_arithmetic : arithmetic_sequence a 2
axiom b_is_geometric : geometric_sequence b 3
axiom b2_eq_3 : b 2 = 3
axiom b3_eq_9 : b 3 = 9
axiom a1_eq_b1 : a 1 = b 1
axiom a14_eq_b4 : a 14 = b 4

-- Results to be proved
theorem general_formula_an : ∀ n, a n = 2 * n - 1 := sorry
theorem sum_first_n_terms_cn : ∀ n, (∑ i in Finset.range n, (a i + b i)) = n^2 + (3^n - 1) / 2 := sorry

end general_formula_an_sum_first_n_terms_cn_l341_341736


namespace min_sum_of_squares_of_distances_l341_341451

-- Define the coordinate system and standard lengths
variable {point : Type*} [MetricSpace point]
variables (P A B C M : point)
variables (PA PB PC : ℝ) 

-- Given conditions
def conditions : Prop :=
  (PA = dist P A) ∧ (PB = dist P B) ∧ (PC = dist P C) ∧
  (dist P A = 3) ∧ (dist P B = 3) ∧ (dist P C = 4) ∧
  (angle P A C = π / 2) ∧ (angle C P B = π / 2) ∧ (angle B P A = π / 2)

-- The theorem to prove
theorem min_sum_of_squares_of_distances (H : conditions P A B C PA PB PC) :
  ∃ M : point, 
  let dM_PAB := dist M (line P A B),
      dM_PBC := dist M (line P B C),
      dM_PCA := dist M (line P C A)
  in dM_PAB^2 + dM_PBC^2 + dM_PCA^2 = 144 / 41 := sorry

end min_sum_of_squares_of_distances_l341_341451


namespace find_angle_opposite_side_l341_341067

noncomputable def angle_opposite_side {a b c : ℝ} (h : (a + b + c) * (a + b - c) = 4 * a * b) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  let C := 0 in if hC : ∃ C, C = 0 then 0 else 0

theorem find_angle_opposite_side {a b c : ℝ} (h : (a + b + c) * (a + b - c) = 4 * a * b) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  angle_opposite_side h ha hb hc = 0 := by
  sorry

end find_angle_opposite_side_l341_341067


namespace rationalize_fraction1_rationalize_fraction2_product_of_fractions_l341_341982

variables (a b c : ℝ) (h_c : c > 0)

theorem rationalize_fraction1 (h_c: c > 0) :
  (a / (b - real.sqrt c + a) = (a * (b + a + real.sqrt c)) / ((b + a)^2 - c)) :=
sorry

variables (x y z u : ℝ) (h_y : y > 0) (h_u : u > 0)

theorem rationalize_fraction2 :
  (a / (x + real.sqrt y + z + real.sqrt u) =
  (a * (x + z - real.sqrt y - real.sqrt u) * ((x + z)^2 - y - u + 2 * real.sqrt (y * u))) / 
  (((x + z)^2 - y - u)^2 - 4 * y * u)) :=
sorry

theorem product_of_fractions :
  (2 * (2 / real.sqrt 2) * (2 / real.sqrt (2 + real.sqrt 2)) * 
  (2 / real.sqrt (2 + real.sqrt (2 + real.sqrt 2))) * 
  (2 / real.sqrt (2 + real.sqrt (2 + real.sqrt (2 + real.sqrt 2)))) = 
  2^4 * real.sqrt (2 - real.sqrt (2 + real.sqrt (2 + real.sqrt 2)))) :=
sorry

end rationalize_fraction1_rationalize_fraction2_product_of_fractions_l341_341982


namespace monthly_fixed_costs_l341_341617

theorem monthly_fixed_costs (cost_per_component shipping_cost_per_unit components_sold price_per_component : ℝ) 
  (total_variable_cost total_revenue F : ℝ) 
  (h_cost_comp : cost_per_component = 80) 
  (h_shipping_comp : shipping_cost_per_unit = 4) 
  (h_components_sold : components_sold = 150) 
  (h_price_comp : price_per_component = 193.33) 
  (h_total_variable_cost : total_variable_cost = cost_per_component + shipping_cost_per_unit) 
  (h_total_revenue : total_revenue = components_sold * price_per_component) 
  (h_total_var_costs_150 : total_variable_cost * components_sold = 12600) :
  F = total_revenue - h_total_var_costs_150 := 
by { rw [h_total_variable_cost, h_cost_comp, h_shipping_comp, h_components_sold, h_price_comp, h_total_revenue],
     norm_num,
     sorry }

end monthly_fixed_costs_l341_341617


namespace correct_statements_l341_341647

theorem correct_statements:
  ¬(∀ (l1 l2 : Line), (l1.slope = l2.slope) → l1.parallel l2) ∧
  ¬(∀ (l1 l2 : Line), l1.parallel l2 → (l1.slope = l2.slope) ∨ (l1.slope = none ∧ l2.slope = none)) ∧
  ¬(∀ (l1 l2 : Line), (l1.slope = none ∧ l2.slope ≠ none) ∨ (l2.slope = none ∧ l1.slope ≠ none) → l1.perpendicular l2) ∧
  (∀ (l1 l2 : Line), (l1.slope = none ∧ l2.slope = none ∧ ¬(l1 = l2)) → l1.parallel l2) :=
sorry

end correct_statements_l341_341647


namespace find_extreme_value_at_pi_over_3_l341_341382

noncomputable def f (x : ℝ) : ℝ := Real.log x - Real.sin x

theorem find_extreme_value_at_pi_over_3 :
  (∃ a : ℝ, ∀ x : ℝ, f(x) = Real.log x - Real.sin x → (f (Real.pi / 3) = 0) → a = Real.pi / 6) := sorry

end find_extreme_value_at_pi_over_3_l341_341382


namespace solution_inequality_l341_341056

variable (a x : ℝ)

theorem solution_inequality (h : ∀ x, |x - a| + |x + 4| ≥ 1) : a ≤ -5 ∨ a ≥ -3 := by
  sorry

end solution_inequality_l341_341056


namespace sqrt_pattern_l341_341374

theorem sqrt_pattern {a b : ℕ} (h1 : a = 8^2 - 1) (h2 : b = 8) :
  (√(b + b / a) = b * √(b / a)) :=
by
  sorry

end sqrt_pattern_l341_341374


namespace trigonometric_identity_l341_341712

variable {a b c A B C : ℝ}

theorem trigonometric_identity (h1 : 2 * c^2 - 2 * a^2 = b^2) 
  (cos_A : ℝ) (cos_C : ℝ) 
  (h_cos_A : cos_A = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_C : cos_C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  2 * c * cos_A - 2 * a * cos_C = b := 
sorry

end trigonometric_identity_l341_341712


namespace Eliane_schedule_count_l341_341686

-- Define the conditions as given in the problem
def morning_classes_days := ['Monday, 'Tuesday, 'Wednesday, 'Thursday, 'Friday, 'Saturday]
def morning_classes_times := ['9AM, '10AM, '11AM]
def afternoon_classes_days := ['Monday, 'Tuesday, 'Wednesday, 'Thursday, 'Friday]
def afternoon_classes_times := ['5PM, '6PM]

-- Define the constraints
def valid_schedule (morning_day : String) (afternoon_day : String) : Prop :=
  morning_day ≠ afternoon_day ∧ (morning_day, afternoon_day) ∉ 
    [('Monday, 'Tuesday), ('Tuesday, 'Wednesday), ('Wednesday, 'Thursday), ('Thursday, 'Friday), ('Friday, 'Saturday)]

-- Prove that the total number of schedules satisfying the conditions is 96
theorem Eliane_schedule_count :
  ∃ num_ways : ℕ, num_ways = 96 ∧ (
    ∀ (morning_day : String) (morning_time : String) 
      (afternoon_day : String) (afternoon_time : String),
      morning_day ∈ morning_classes_days →
      morning_time ∈ morning_classes_times →
      afternoon_day ∈ afternoon_classes_days →
      afternoon_time ∈ afternoon_classes_times →
      valid_schedule morning_day afternoon_day →
      ∃ ways : ℕ, ways = num_ways
  ) :=
sorry

end Eliane_schedule_count_l341_341686


namespace find_a_l341_341349

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - y = 3) : a = 2 :=
by
  sorry

end find_a_l341_341349


namespace expected_score_is_6_l341_341058

-- Define the probabilities of making a shot
def p : ℝ := 0.5

-- Define the scores for each scenario
def score_first_shot : ℝ := 8
def score_second_shot : ℝ := 6
def score_third_shot : ℝ := 4
def score_no_shot : ℝ := 0

-- Compute the expected value
def expected_score : ℝ :=
  p * score_first_shot +
  (1 - p) * p * score_second_shot +
  (1 - p) * (1 - p) * p * score_third_shot +
  (1 - p) * (1 - p) * (1 - p) * score_no_shot

theorem expected_score_is_6 : expected_score = 6 := by
  sorry

end expected_score_is_6_l341_341058


namespace article_cost_price_l341_341241

theorem article_cost_price :
  ∃ C : ℝ, 
  (1.05 * C) - 2 = (1.045 * C) ∧ 
  ∃ C_new : ℝ, C_new = (0.95 * C) ∧ ((1.045 * C) = (C_new + 0.1 * C_new)) ∧ C = 400 := 
sorry

end article_cost_price_l341_341241


namespace sin2a_minus_cos2a_half_l341_341014

theorem sin2a_minus_cos2a_half (a : ℝ) (h : Real.tan (a - Real.pi / 4) = 1 / 2) :
  Real.sin (2 * a) - Real.cos a ^ 2 = 1 / 2 := 
sorry

end sin2a_minus_cos2a_half_l341_341014


namespace bisection_method_root_interval_bisection_method_second_calculation_l341_341231

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_method_root_interval (h0 : f 0 < 0) (h05 : f 0.5 > 0) : 
  ∃ x0, 0 < x0 ∧ x0 < 0.5 ∧ f x0 = 0 := 
by
  sorry

theorem bisection_method_second_calculation : 
  let x_mid : ℝ := 0.25 in 
  f x_mid = f (0.25) := 
by
  rfl

end bisection_method_root_interval_bisection_method_second_calculation_l341_341231


namespace renovation_time_l341_341933

def bedrooms := 3
def bedroom_time := 4 -- time in hours per bedroom
def kitchen_time := bedroom_time * 1.5
def combined_time := bedrooms * bedroom_time + kitchen_time
def living_room_time := combined_time * 2
def total_time := combined_time + living_room_time

theorem renovation_time : total_time = 54 := by
  -- Calculations leading to result verification skipped
  sorry

end renovation_time_l341_341933


namespace basketball_volume_64_times_baseball_l341_341991

-- Define the conditions
def radius_relation (r : ℝ) (R : ℝ) : Prop :=
  R = 4 * r

def volume_of_sphere (r : ℝ) : ℝ :=
  (4/3) * π * r^3

-- Define the statement to be proved
theorem basketball_volume_64_times_baseball (r R : ℝ) (h : radius_relation r R) :
  volume_of_sphere R = 64 * volume_of_sphere r :=
by
  sorry

end basketball_volume_64_times_baseball_l341_341991


namespace binary_string_to_power_of_two_l341_341668

-- Define the binary string and its properties
def binary_string := list ℕ

-- Predicate to check if a list is a binary string
def is_binary_string (b: binary_string) : Prop :=
  ∀ d ∈ b, d = 0 ∨ d = 1

-- Predicate to count the number of ones in a binary string
def count_ones (b: binary_string) : ℕ :=
  b.count (λ d => d = 1)

-- Predicate to check if the sum of a decomposed binary string is a power of two
def is_power_of_two (n: ℕ) : Prop :=
  ∃ k, n = 2^k

-- The main theorem statement
theorem binary_string_to_power_of_two (b : binary_string) (h1 : is_binary_string b) (h2 : count_ones b ≥ 2017) :
  ∃ (s : binary_string → list ℕ → ℕ), (∀ plus_signs between digits of b, by perform sum in base 2 result in power of two) :=
sorry

end binary_string_to_power_of_two_l341_341668


namespace smallest_consecutive_integer_sum_l341_341562

-- Definitions based on conditions
def consecutive_integer_sum (n : ℕ) := 20 * n + 190

-- Theorem statement
theorem smallest_consecutive_integer_sum : 
  ∃ (n k : ℕ), (consecutive_integer_sum n = k^3) ∧ (∀ m l : ℕ, (consecutive_integer_sum m = l^3) → k^3 ≤ l^3) :=
sorry

end smallest_consecutive_integer_sum_l341_341562


namespace unique_root_when_abs_t_gt_2_l341_341247

theorem unique_root_when_abs_t_gt_2 (t : ℝ) (h : |t| > 2) :
  ∃! x : ℝ, x^3 - 3 * x = t ∧ |x| > 2 :=
sorry

end unique_root_when_abs_t_gt_2_l341_341247


namespace Jerry_throw_away_time_l341_341453

theorem Jerry_throw_away_time :
  (let num_cans := 35 in
   let cans_per_trip := 3 in
   let drain_time_per_trip := 30 in
   let walk_time_per_way := 10 in
   let num_trips := (35.to_real / 3).ceil.to_nat in
   let total_drain_time := num_trips * drain_time_per_trip in
   let total_walk_time := num_trips * (2 * walk_time_per_way) in
   let total_time_seconds := total_drain_time + total_walk_time in
   let total_time_minutes := total_time_seconds / 60 in
   total_time_minutes = 10) :=
by
  sorry

end Jerry_throw_away_time_l341_341453


namespace distance_from_point_to_plane_l341_341077

-- Defining conditions
variables (A B C D : Point)
variable h1 : equilateral_triangle A B C
variable h2 : distance A D = 2
variable h3 : distance B D = 2
variable h4 : distance C D = 2
variable h5 : perpendicular A D B D
variable h6 : perpendicular A D C D

-- Finding the distance from D to the plane ABC
theorem distance_from_point_to_plane (A B C D : Point) (h1 : equilateral_triangle A B C) 
  (h2 : distance A D = 2) (h3 : distance B D = 2) (h4 : distance C D = 2)
  (h5 : perpendicular A D B D) (h6 : perpendicular A D C D) :
  distance_to_plane D (plane_of_triangle A B C) = 2 * sqrt 3 / 3 := sorry

end distance_from_point_to_plane_l341_341077


namespace hyperbola_equation_given_conditions_l341_341719

theorem hyperbola_equation_given_conditions :
  ∀ (a b : ℝ), a > 0 → b > 0 → 
  ( ∃ F : ℝ × ℝ, F = (2, 0) ) →
  ( ∃ h₁ h₂ : ℝ → ℝ, h₁ x = (b / a) * x ∧ h₂ x = -(b / a) * x ) → 
  ( ∀ C : (ℝ × ℝ), C = (2, 0) ) → 
  ( ∃ rc : ℝ, rc = √3) →
  ( ∀ (A B : ℝ), A = a^2 ∧ B = b^2 →  
  ( ∀ E : ℝ, E = A + B → ( ∀ D : ℝ, D = b * b - a * a ) )) →
  ( ∑ k, k = 4 ) →
  (A = 1 ∧ B = 3) → 
  (x^2 - (y^2 / 3) = 1) := 
by
  sorry

end hyperbola_equation_given_conditions_l341_341719


namespace impossible_to_move_A_to_B_l341_341193

/-- The plane is divided into equilateral triangles with side length 1 by three infinite series of equally spaced parallel lines.
    M is the set of all vertices of these triangles.
    Given two vertices A and B of one triangle, it is allowed to rotate the plane by 120 degrees around any vertex in set M.
    Prove that it is impossible to move point A to point B with a finite number of such transformations. -/
theorem impossible_to_move_A_to_B
  (side_length : ℝ := 1)
  (M : set (ℝ × ℝ))
  (A B : ℝ × ℝ)
  (hA : A ∈ M)
  (hB : B ∈ M)
  (hTriangle : ∃ T : finset (ℝ × ℝ), {A, B} ⊆ T ∧ T.card = 3 ∧ (∀ pair ∈ T.pairs, dist pair.fst pair.snd = side_length))
  : ¬(∃ n : ℕ, ∃ transformations : fin n → (ℝ × ℝ) → (ℝ × ℝ), (transformations 0) A = B) := sorry

end impossible_to_move_A_to_B_l341_341193


namespace usual_time_56_l341_341604

theorem usual_time_56 (S : ℝ) (T : ℝ) (h : (T + 24) * S = T * (0.7 * S)) : T = 56 :=
by sorry

end usual_time_56_l341_341604


namespace polynomial_proof_l341_341339

noncomputable def Q (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 3

theorem polynomial_proof : Q (real.cbrt 3 + 1) = 0 ∧ 
                           (∀ (x : ℝ), Q(x).leadingCoeff = 1 ∧ 
                             (∀ (a b c : ℤ), Q(a + b * x + c * x^2).denom = 1)) :=
by
  sorry

end polynomial_proof_l341_341339


namespace action_movies_rented_l341_341683

-- Defining the conditions as hypotheses
theorem action_movies_rented (a M A D : ℝ) (h1 : 0.64 * M = 10 * a)
                             (h2 : D = 5 * A)
                             (h3 : D + A = 0.36 * M) :
    A = 0.9375 * a :=
sorry

end action_movies_rented_l341_341683


namespace increasing_iff_integral_condition_l341_341519

variable {f : ℝ → ℝ}

theorem increasing_iff_integral_condition (h_continuous : continuous f) :
  (∀ (a b c : ℝ), a < b → b < c → (c - b) * (∫ x in a..b, f x) ≤ (b - a) * (∫ x in b..c, f x)) ↔
  (∀ x y : ℝ, x < y → f x ≤ f y) :=
sorry

end increasing_iff_integral_condition_l341_341519


namespace sum_of_positive_divisors_of_prime_l341_341960

theorem sum_of_positive_divisors_of_prime (h_prime : Nat.prime 47) : 1 + 47 = 48 :=
by
  have d1 : 1 ∣ 47 := Nat.one_dvd 47
  have d47 : 47 ∣ 47 := Nat.dvd_refl 47
  have divisors := [1, 47]
  have sum_divisors := List.sum divisors
  rw [List.sum_cons, List.sum_nil] at sum_divisors
  simp at sum_divisors
  exact sum_divisors

end sum_of_positive_divisors_of_prime_l341_341960


namespace angle_between_generatrix_and_base_l341_341276

-- Definitions from conditions
variables (R1 R2 : ℝ) (frustumSphere : Bool)
variables (base1_area base2_area : ℝ) (base_ratio : base2_area = 4 * base1_area)

-- Topic-specific variables
variables (slant_height : ℝ) (angle_generatrix_base : ℝ)

-- The actual statement
theorem angle_between_generatrix_and_base
  (h1 : base2_area = 4 * base1_area)
  (h2 : ∀ {R2}, R2 = 2 * R1)
  : angle_generatrix_base = Real.arccos(1 / Real.sqrt 5) :=
sorry

end angle_between_generatrix_and_base_l341_341276


namespace geometric_sequence_relation_l341_341607

-- Arithmetic Sequence Problem
variable (a : ℕ → ℝ) (a_10_zero : a 10 = 0)

theorem geometric_sequence_relation
  (b : ℕ → ℝ) (b_9_one : b 9 = 1) :
  ∀ n : ℕ, n < 17 → (∏ i in Finset.range n, b i) = (∏ i in Finset.range (17 - n), b i) :=
sorry

-- Tetrahedron Problem
variable (V : ℝ) (S1 S2 S3 S4 : ℝ)

def radius_inscribed_sphere : ℝ :=
  3 * V / (S1 + S2 + S3 + S4)
  
example : radius_inscribed_sphere V S1 S2 S3 S4 = 3 * V / (S1 + S2 + S3 + S4) :=
by refl

end geometric_sequence_relation_l341_341607


namespace distance_between_points_on_line_l341_341770

theorem distance_between_points_on_line 
  (p q r s : ℝ)
  (line_eq : q = 2 * p + 3) 
  (s_eq : s = 2 * r + 6) :
  Real.sqrt ((r - p)^2 + (s - q)^2) = Real.sqrt (5 * (r - p)^2 + 12 * (r - p) + 9) :=
sorry

end distance_between_points_on_line_l341_341770


namespace odd_function_formula_l341_341739

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2*x + 3
else if x < 0 then - (x^2 + 2*x + 3)
else 0

theorem odd_function_formula :
  (∀ x : ℝ, f (-x) = - f x) →
  (∀ x : ℝ, (x > 0) → f x = x^2 - 2*x + 3) →
  ∀ x : ℝ, f x = 
  if x > 0 then x^2 - 2*x + 3
  else if x < 0 then - (x^2 + 2*x + 3)
  else 0 :=
by
  intro h_odd h_pos
  funext x
  split_ifs
  case h_1 =>
    apply h_pos, exact h_1
  case h_2 =>
    have h : x ≠ 0 := by simp [h]
    rw [←neg_pos, ←(h_odd (-x)), h_pos]
    simp [h]
  case h_3 =>
    have h : x = 0 := by
      simp [lt_or_gt_of_ne h]
    simp [h]

end odd_function_formula_l341_341739


namespace find_k_l341_341912

theorem find_k : 
  ∃ x y k : ℝ, y = 7 * x - 2 ∧ y = -3 * x + 14 ∧ y = 4 * x + k ∧ k = 2.8 :=
by
  sorry

end find_k_l341_341912


namespace ant_edge_same_direction_once_l341_341650

-- Definitions for the problem context
def dodecahedron (V : Type) := 
  ∃ (E : set (V × V)), 
    (card V = 20) ∧ 
    (card E = 30) ∧ 
    (∀ v ∈ V, ∃! (e1 e2 e3 ∈ E), (v ∈ fst e1 ∧ v ∈ fst e2 ∧ v ∈ fst e3))

-- The closed path on the dodecahedron edges
structure ant_path (V : Type) (E : set (V × V)) :=
  (path : list (V × V))
  (closed : (path.head = path.last))
  (no_turn_back : ∀ (i : ℕ) (h : i < path.length), ¬((path.nth_le i h).snd = (path.nth_le (i + 1) (lt_of_succ_lt h)).fst))
  (cross_exactly_twice : ∀ e ∈ E, list.count e path = 2)

-- The goal statement for Lean
theorem ant_edge_same_direction_once (V : Type) (d : dodecahedron V) : 
  ∃ (E : set (V × V)) (p : ant_path V E), 
    ∃ e ∈ E, list.adjacent (λ e1 e2 : (V × V), e1 = e2) p.path e := 
begin
  sorry
end

end ant_edge_same_direction_once_l341_341650


namespace sum_of_integers_square_256_sum_of_integers_solution_l341_341559

theorem sum_of_integers_square_256 (x : ℤ) (h : x^2 = x + 256) : x = 16 ∨ x = -16 :=
by
  sorry

theorem sum_of_integers_solution : (16 + (-16) = 0) : Prop :=
by
  exact rfl

end sum_of_integers_square_256_sum_of_integers_solution_l341_341559


namespace amy_uploaded_photos_l341_341648

theorem amy_uploaded_photos (albums photos_per_album : ℕ) (h1 : albums = 9) (h2 : photos_per_album = 20) :
  albums * photos_per_album = 180 :=
by {
  sorry
}

end amy_uploaded_photos_l341_341648


namespace one_positive_real_solution_l341_341412

theorem one_positive_real_solution : 
    ∃! x : ℝ, 0 < x ∧ (x ^ 10 + 7 * x ^ 9 + 14 * x ^ 8 + 1729 * x ^ 7 - 1379 * x ^ 6 = 0) :=
sorry

end one_positive_real_solution_l341_341412


namespace compute_f100_l341_341312

def is_multiple_of (a b : ℕ) : Prop := ∃ k, b = a * k

noncomputable def f : ℕ → ℝ
| x := if is_multiple_of 3 x ∧ ¬ is_multiple_of 5 x then real.log x / real.log 3 else
       if is_multiple_of 5 x ∧ ¬ is_multiple_of 3 x then real.log x / real.log 5 else
       if is_multiple_of 15 x then real.log x / real.log 15 else
       1 + f (x + 1)

theorem compute_f100 : f 100 = 7 :=
by
  sorry

end compute_f100_l341_341312


namespace sum_of_integers_satisfying_equation_l341_341555

theorem sum_of_integers_satisfying_equation : (∑ x in { x : ℤ | x^2 = 256 + x }, x) = 0 := 
sorry

end sum_of_integers_satisfying_equation_l341_341555


namespace sum_first_11_elems_s11_l341_341448

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

-- Definitions based on the problem
variables {a : ℕ → ℝ} [h_arith : is_arithmetic_sequence a]

-- Given condition
axiom condition : a 2 + a 4 + a 6 + a 8 + a 10 = 80

-- Question to be proven
theorem sum_first_11_elems_s11 : ∑ i in finset.range 11, a i = 176 :=
begin
  sorry
end

end sum_first_11_elems_s11_l341_341448


namespace quadratic_has_one_solution_l341_341163

theorem quadratic_has_one_solution (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0) ∧ (∀ x₁ x₂ : ℝ, (3 * x₁^2 - 6 * x₁ + m = 0) → (3 * x₂^2 - 6 * x₂ + m = 0) → x₁ = x₂) → m = 3 :=
by
  -- intricate steps would go here
  sorry

end quadratic_has_one_solution_l341_341163


namespace probability_all_black_after_rotation_l341_341990

-- Define the conditions
def num_unit_squares : ℕ := 16
def num_colors : ℕ := 3
def prob_per_color : ℚ := 1 / 3

-- Define the type for probabilities
def prob_black_grid : ℚ := (1 / 81) * (11 / 27) ^ 12

-- The statement to be proven
theorem probability_all_black_after_rotation :
  (prob_black_grid =
    ((1 / 3) ^ 4) * ((11 / 27) ^ 12)) :=
sorry

end probability_all_black_after_rotation_l341_341990


namespace max_distance_ac_l341_341403

variables {a b c : ℝ × ℝ}
def vector_length (v : ℝ × ℝ) := real.sqrt (v.1 * v.1 + v.2 * v.2)
def dot_product (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2

axiom length_a : vector_length a = 4
axiom length_b : vector_length b = 2 * real.sqrt 2
axiom angle_ab : dot_product a b / (vector_length a * vector_length b) ≥ real.cos (real.pi / 4)
axiom condition_c : dot_product (c - a) (c - b) = -1

theorem max_distance_ac : ∀ (a b c : ℝ × ℝ),
  vector_length a = 4 →
  vector_length b = 2 * real.sqrt 2 →
  dot_product a b / (vector_length a * vector_length b) ≥ real.cos (real.pi / 4) →
  dot_product (c - a) (c - b) = -1 →
  vector_length (c - a) ≤ real.sqrt 2 + 1 :=
by sorry

end max_distance_ac_l341_341403


namespace value_of_a_l341_341398

theorem value_of_a (a : ℝ) (A : Set ℝ) (B : Set ℝ) (hA : A = {-3}) (hB : B = { x | a * x + 1 = 0 }) (hSubset : B ⊆ A) : a = 1 / 3 :=
sorry

end value_of_a_l341_341398


namespace sum_two_digit_divisors_of_135_with_remainder_9_l341_341108

theorem sum_two_digit_divisors_of_135_with_remainder_9 (d : ℕ) (hd_pos : d > 0) (hd_cond : 144 % d = 9) : 
∃ (s : ℕ), s = 60 ∧ ((d | 135) → (10 ≤ d ∧ d < 100)) → (s = 15 + 45) := sorry

end sum_two_digit_divisors_of_135_with_remainder_9_l341_341108


namespace exists_large_subset_l341_341844

open Nat

theorem exists_large_subset (X : Finset ℕ) (n m : ℕ) 
  (A : (Fin n)) (Ai : Finset ℕ → Finset ℕ) 
  (hX_card : X.card = n) 
  (hAi_card : ∀ i, (Ai i).card = 3) 
  (hAi_inter : ∀ i j, i ≠ j → (Ai i ∩ Ai j).card ≤ 1) :
  ∃ B : Finset ℕ, B ⊆ X ∧ B.card ≥ ⌊ sqrt (2 * n : ℝ) ⌋₊ ∧ ∀ i, (Ai i) ∩ B = ∅ :=
sorry

end exists_large_subset_l341_341844


namespace no_solutions_distinct_l341_341672

theorem no_solutions_distinct :
  ∀ (a b c d e f g h : ℕ), 
    (∀ x y, x ≠ y → x ∈ {a, b, c, d, e, f, g, h} → y ∈ {a, b, c, d, e, f, g, h}) ∧ 
    (a > 0 ∧ a < 9) ∧ (b > 0 ∧ b < 9) ∧ (c > 0 ∧ c < 9) ∧ (d > 0 ∧ d < 9) ∧ 
    (e > 0 ∧ e < 9) ∧ (f > 0 ∧ f < 9) ∧ (g > 0 ∧ g < 9) ∧ (h > 0 ∧ h < 9) → 
    (a / 10^1 + b / 10^2 + c / 10^3 + d / 10^4 + e / 10^1 + f / 10^2 + g / 10^3 + h / 10^4 ≠ 1) := 
sorry

end no_solutions_distinct_l341_341672


namespace find_m_for_symmetry_l341_341392

def symmetric_about_y_eq_x (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = x → ∃ y, f y = x

theorem find_m_for_symmetry :
  ∃ m : ℝ, (∀ x : ℝ, (λ x, (x - 5) / (2 * x + m)) x = x → (∃ y : ℝ, (λ x, (x - 5) / (2 * x + m)) y = x)) → m = -1 :=
begin
  let f := λ x, (x - 5) / (2 * x + m),
  have sym_f := ∀ x, f x = x → ∃ y, f y = x,
  sorry
end

end find_m_for_symmetry_l341_341392


namespace arthur_leftover_money_l341_341289

theorem arthur_leftover_money :
  (∀ cards_worth comic_book_cost num_cards : ℕ, 
    cards_worth = 5 ∧ comic_book_cost = 600 ∧ num_cards = 2000 →
    (let total_money := (cards_worth * num_cards) / 100 in
     let num_comic_books := total_money / comic_book_cost in 
     let cost_comic_books := num_comic_books * comic_book_cost in
     total_money - cost_comic_books / 100 = 4)) :=
λ cards_worth comic_book_cost num_cards ⟨h1, h2, h3⟩,
  let total_money := (cards_worth * num_cards) / 100 in
  let num_comic_books := total_money / comic_book_cost in 
  let cost_comic_books := num_comic_books * comic_book_cost in
  have leftover := total_money - cost_comic_books / 100, 
  sorry

end arthur_leftover_money_l341_341289


namespace max_value_cos_sin_sum_l341_341699

theorem max_value_cos_sin_sum (θ₁ θ₂ θ₃ θ₄ : ℝ) :
  ∃ θ₁ θ₂ θ₃ θ₄, ((cos θ₁ * sin θ₂) + (cos θ₂ * sin θ₃) + (cos θ₃ * sin θ₄) + (cos θ₄ * sin θ₁)) = 2 := 
sorry

end max_value_cos_sin_sum_l341_341699


namespace real_part_of_inverse_l341_341118

def z : ℂ := sorry

theorem real_part_of_inverse (hz: |z| = 2 ∧ im z ≠ 0) : real_part (1 / (2 - z)) = 1 / 2 := 
by
  sorry

end real_part_of_inverse_l341_341118


namespace circles_tangent_to_y_axis_and_line_l341_341664

theorem circles_tangent_to_y_axis_and_line:
  ∃ d e (f : ℕ), 
  ∃ (n : ℝ) (h_pos : 0 < n) (h_format : n = (d * real.sqrt e) / f),
  let r1 := c / (d * real.sqrt e) / f,
      r2 := d / (d * real.sqrt e) / f in
  let cir1 := (10 - c)^2 + (8 - c / ((d * real.sqrt e) / f))^2 = (c / ((d * real.sqrt e) / f))^2,
      cir2 := (10 - d)^2 + (8 - d / ((d * real.sqrt e) / f))^2 = (d / ((d * real.sqrt e) / f))^2 in
  cir1 ∧ cir2 ∧ (r1 * r2 = 50) → d + e + f = 24 := 
sorry

end circles_tangent_to_y_axis_and_line_l341_341664


namespace remainder_2456789_div_7_l341_341228

theorem remainder_2456789_div_7 :
  2456789 % 7 = 6 := 
by 
  sorry

end remainder_2456789_div_7_l341_341228


namespace number_of_true_propositions_is_one_l341_341283

-- Define the propositions
def P1 : Prop := ∀ (a b : ℝ), a = b → (a and b are vertical angles)
def P2 : Prop := ∀ (a b : ℝ), (a and b are corresponding angles) → a = b
def P3 : Prop := ∀ (a b : ℝ), a = b → (180 - a) = (180 - b)
def P4 : Prop := ∀ (x y : ℝ), x^2 = y^2 → x = y

-- Prove that the number of true propositions is 1
theorem number_of_true_propositions_is_one :
  (¬P1) ∧ (¬P2) ∧ P3 ∧ (¬P4) → (1 = 1) :=
by
  sorry

end number_of_true_propositions_is_one_l341_341283


namespace total_teaching_time_l341_341685

def teaching_times :=
  let eduardo_math_time := 3 * 60
  let eduardo_science_time := 4 * 90
  let eduardo_history_time := 2 * 120
  let total_eduardo_time := eduardo_math_time + eduardo_science_time + eduardo_history_time

  let frankie_math_time := 2 * (3 * 60)
  let frankie_science_time := 2 * (4 * 90)
  let frankie_history_time := 2 * (2 * 120)
  let total_frankie_time := frankie_math_time + frankie_science_time + frankie_history_time

  let georgina_math_time := 3 * (3 * 80)
  let georgina_science_time := 3 * (4 * 100)
  let georgina_history_time := 3 * (2 * 150)
  let total_georgina_time := georgina_math_time + georgina_science_time + georgina_history_time

  total_eduardo_time + total_frankie_time + total_georgina_time

theorem total_teaching_time : teaching_times = 5160 := by
  -- calculations omitted
  sorry

end total_teaching_time_l341_341685


namespace find_ABC_sum_l341_341179

-- Conditions
def poly (A B C : ℤ) (x : ℤ) := x^3 + A * x^2 + B * x + C
def roots_condition (A B C : ℤ) := poly A B C (-1) = 0 ∧ poly A B C 3 = 0 ∧ poly A B C 4 = 0

-- Proof goal
theorem find_ABC_sum (A B C : ℤ) (h : roots_condition A B C) : A + B + C = 11 :=
sorry

end find_ABC_sum_l341_341179


namespace jonathans_weekly_caloric_deficit_l341_341090

def daily_calorie_intake : ℕ → ℕ
| 0 := 2500  -- Monday
| 1 := 2600  -- Tuesday
| 2 := 2400  -- Wednesday
| 3 := 2700  -- Thursday
| 4 := 2300  -- Friday
| 5 := 3500  -- Saturday
| 6 := 2400  -- Sunday
| _ := 0      -- In case of an invalid day input

def calories_burned_via_exercise : ℕ → ℕ
| 0 := 1000  -- Monday
| 1 := 1200  -- Tuesday
| 2 := 1300  -- Wednesday
| 3 := 1600  -- Thursday
| 4 := 1000  -- Friday
| 5 := 0     -- Saturday
| 6 := 1200  -- Sunday
| _ := 0      -- In case of an invalid day input

def net_daily_calories (d : ℕ) : ℕ := daily_calorie_intake d - calories_burned_via_exercise d

def total_net_weekly_calories : ℕ :=
  (net_daily_calories 0) + (net_daily_calories 1) + (net_daily_calories 2) + 
  (net_daily_calories 3) + (net_daily_calories 4) + (net_daily_calories 5) + 
  (net_daily_calories 6)

def total_weekly_calories_consumed : ℕ :=
  daily_calorie_intake 0 + daily_calorie_intake 1 + daily_calorie_intake 2 + 
  daily_calorie_intake 3 + daily_calorie_intake 4 + daily_calorie_intake 5 + 
  daily_calorie_intake 6

theorem jonathans_weekly_caloric_deficit : total_weekly_calories_consumed - total_net_weekly_calories = 6800 :=
  by sorry

end jonathans_weekly_caloric_deficit_l341_341090


namespace quadratic_distinct_zeros_l341_341035

theorem quadratic_distinct_zeros (m : ℝ) : 
  (x^2 + m * x + (m + 3)) = 0 → 
  (0 < m^2 - 4 * (m + 3)) ↔ (m < -2) ∨ (m > 6) :=
sorry

end quadratic_distinct_zeros_l341_341035


namespace range_of_a_l341_341050

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → (1/3 * x^3 - (a/2) * x^2 + (a-1) * x + 1) < 0) ∧
  (∀ x : ℝ, x > 6 → (1/3 * x^3 - (a/2) * x^2 + (a-1) * x + 1) > 0)
  ↔ (5 < a ∧ a < 7) :=
sorry

end range_of_a_l341_341050


namespace triangle_problem_l341_341016

noncomputable def triangle_area (a b C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : c = 2 * a) 
  (h2 : b = 2 * Real.sqrt 3)
  (h3 : Real.sqrt 3 * c * Real.sin A = a * Real.cos C) :
  C = Real.pi / 6 ∧ triangle_area a b (Real.pi / 6) = (Real.sqrt 15 - Real.sqrt 3) / 2 :=
begin
  sorry
end

end triangle_problem_l341_341016


namespace B_set_l341_341860

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

def A (a b : ℝ) : set ℝ := {x | f x a b = x}

def B (a b : ℝ) : set ℝ := {x | f (f x a b) a b = x}

theorem B_set (a b : ℝ) (h : A a b = {-1, 3}) : B a b = {-1, real.sqrt 3, -real.sqrt 3, 3} := 
sorry

end B_set_l341_341860


namespace factorize_x4_minus_1_l341_341326

theorem factorize_x4_minus_1 (x : ℂ) : x^4 - 1 = (x + Complex.i) * (x - Complex.i) * (x - 1) * (x + 1) :=
  sorry

end factorize_x4_minus_1_l341_341326


namespace sum_of_perimeters_l341_341924

theorem sum_of_perimeters (x y : Real) 
  (h1 : x^2 + y^2 = 85)
  (h2 : x^2 - y^2 = 45) :
  4 * (Real.sqrt 65 + 2 * Real.sqrt 5) = 4 * x + 4 * y := by
  sorry

end sum_of_perimeters_l341_341924


namespace symmetrical_line_intersects_circle_l341_341486

variables (A_x A_y B_x B_y a x y : ℝ)

def points_A_B (A_x A_y B_x B_y : ℝ) : Prop :=
  A_x = -2 ∧ A_y = 3 ∧ B_x = 0

def circle (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y + 2) ^ 2 = 1

theorem symmetrical_line_intersects_circle (a : ℝ) :
  (∀ x y, points_A_B (-2) 3 0 a → (circle x y → ∃ (x y : ℝ), True)) →
  (a ∈ set.Icc (1 / 3 : ℝ) (3 / 2 : ℝ)) :=
by
  intros h
  sorry

end symmetrical_line_intersects_circle_l341_341486


namespace locus_of_endpoints_l341_341656

-- Define the given conditions as premises
variables (r1 r2 S α1 α2 : ℝ) (A : ℝ × ℝ)
variable (α : ℝ) (h_α : α < 2 * Real.pi)

-- Define distances traveled for each arc
def s1 := r1 * α1
def s2 := r2 * α2
def total_distance := s1 + s2

-- Define circles W1 and W2 representing the loci of possible endpoints
noncomputable def W1 := metric.sphere ((2 * (r1 - r2) * (Real.sin α2)), 0) (2 * (r1 - r2) * (Real.sin α2))
noncomputable def W2 := metric.sphere (0, (2 * (r1 - r2) * (Real.sin α1))) (2 * (r1 - r2) * (Real.sin α1))

-- Define the intersection of the two circles as the possible positions
def possible_positions := set.inter W1 W2

-- Theorem statement: The locus of endpoints given the conditions
theorem locus_of_endpoints :
  ∃(P : ℝ × ℝ), P ∈ possible_positions :=
sorry

end locus_of_endpoints_l341_341656


namespace lengths_of_sides_l341_341811

-- Define the necessary components of the problem
variables {A B C D : Type} [EuclideanGeometry]

-- Given conditions
axiom isosceles_triangle (ABC : Triangle) (AB BC : Real) :
  is_isosceles ABC AB BC

axiom vertex_angle (ABC : Triangle) :
  angle ABC = 36

axiom angle_bisector_eq (AD : Segment) :
  AD.length = sqrt 20

axiom angle_bisector (BAC : Angle) (D : Point) :
  is_angle_bisector BAC D

-- The goal is to find the side lengths
theorem lengths_of_sides {AB BC AC : Real} :
  isosceles_triangle ABC AB BC →
  vertex_angle ABC →
  angle_bisector_eq AD →
  angle_bisector BAC D →
  AB = BC ∧ 
  AB = 5 + sqrt 5 ∧ 
  AC = 2 * sqrt 5 :=
sorry

end lengths_of_sides_l341_341811


namespace algebraic_expression_value_l341_341002

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x - y = -2) 
  (h2 : 2 * x + y = -1) : 
  (x - y)^2 - (x - 2 * y) * (x + 2 * y) = 7 :=
by {
  sorry
}

end algebraic_expression_value_l341_341002


namespace smallest_value_of_a_minus_b_l341_341244

theorem smallest_value_of_a_minus_b (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) : a - b = 4 :=
by
  sorry

end smallest_value_of_a_minus_b_l341_341244


namespace quadrilateral_perimeter_l341_341814

theorem quadrilateral_perimeter
    (right_angle_triangle_1 : ∃ A B E : Point, right_triangle A B E ∧ ∠AEB = 45)
    (right_angle_triangle_2 : ∃ B C E : Point, right_triangle B C E ∧ ∠BEC = 45)
    (right_angle_triangle_3 : ∃ C D E : Point, right_triangle C D E ∧ ∠CED = 45)
    (AE_eq_16 : AE = 16) :
    perimeter ABCD = 96 + 16 * sqrt 2 := 
sorry

end quadrilateral_perimeter_l341_341814


namespace binom_9_5_equals_126_l341_341306

theorem binom_9_5_equals_126 : Nat.binom 9 5 = 126 := 
by 
  sorry

end binom_9_5_equals_126_l341_341306


namespace evaluate_g_x_plus_2_l341_341109

theorem evaluate_g_x_plus_2 (x : ℝ) (h₁ : x ≠ -3/2) (h₂ : x ≠ 2) : 
  (2 * (x + 2) + 3) / ((x + 2) - 2) = (2 * x + 7) / x :=
by 
  sorry

end evaluate_g_x_plus_2_l341_341109


namespace intercept_sum_l341_341075

-- Define the equation of the line and the condition on the intercepts.
theorem intercept_sum (c : ℚ) (x y : ℚ) (h1 : 3 * x + 5 * y + c = 0) (h2 : x + y = 55/4) : 
  c = 825/32 :=
sorry

end intercept_sum_l341_341075


namespace sum_of_integers_satisfying_equation_l341_341554

theorem sum_of_integers_satisfying_equation : (∑ x in { x : ℤ | x^2 = 256 + x }, x) = 0 := 
sorry

end sum_of_integers_satisfying_equation_l341_341554


namespace fixed_point_of_line_l341_341155

theorem fixed_point_of_line (m : ℝ) : 
  ∃ (x y : ℝ), ((m - 1) * x - y + 2 * m + 1 = 0) ∧ (x = -2) ∧ (y = 3) := 
by
  have h1 : ∀ m : ℝ, (m - 1) * (-2) - 3 + 2 * m + 1 = 0 := sorry
  use [-2, 3]
  split
  · exact sorry
  · split <;> refl

end fixed_point_of_line_l341_341155


namespace find_n_l341_341134

theorem find_n (n : ℕ) (M : ℕ) (A : ℕ) 
  (hM : M = n - 11) 
  (hA : A = n - 2) 
  (hM_ge_one : M ≥ 1) 
  (hA_ge_one : A ≥ 1) 
  (hM_plus_A_lt_n : M + A < n) : 
  n = 12 := 
by 
  sorry

end find_n_l341_341134


namespace intersection_of_A_and_B_l341_341732

-- Definitions based on conditions
def A : Set ℝ := { x | x + 2 = 0 }
def B : Set ℝ := { x | x^2 - 4 = 0 }

-- Theorem statement proving the question == answer given conditions
theorem intersection_of_A_and_B : A ∩ B = {-2} :=
by 
  sorry

end intersection_of_A_and_B_l341_341732


namespace duck_cow_legs_heads_diff_l341_341441

theorem duck_cow_legs_heads_diff :
  ∀ (D : ℕ), let C := 20 in
  let L := 2 * D + 4 * C in
  let H := D + C in
  L = 2 * H + 40 :=
by
  intro D
  let C := 20
  let L := 2 * D + 4 * C
  let H := D + C
  sorry

end duck_cow_legs_heads_diff_l341_341441


namespace range_of_h_eq_l341_341335

def h (x : ℝ) : ℝ := 3 / (1 + 3 * x^3)

theorem range_of_h_eq (a b : ℝ) 
  (h_range : ∀ y : ℝ, y ∈ set.Ioc a b ↔ ∃ x : ℝ, h x = y) : 
  a + b = 3 := 
sorry

end range_of_h_eq_l341_341335


namespace part1_l341_341983

theorem part1 (a b : Real) (h : sqrt (a - 1) + sqrt ((9 + b)^2) = 0) : real.cbrt (a + b) = -2 :=
sorry

end part1_l341_341983


namespace tom_candy_count_l341_341932

def initial_pieces := 2
def friend_gift := 7
def bought_more := 10
def fraction_given_to_sister := 1 / 3
def percentage_eaten := 0.20

def total_after_friend_gift := initial_pieces + friend_gift
def total_after_buying := total_after_friend_gift + bought_more
def given_to_sister := floor (total_after_buying * fraction_given_to_sister)
def remaining_after_sister := total_after_buying - given_to_sister
def eaten := floor (remaining_after_sister * percentage_eaten)
def remaining_candies := remaining_after_sister - eaten

theorem tom_candy_count : 
  remaining_candies = 11 := 
sorry

end tom_candy_count_l341_341932


namespace problem_inequality_l341_341830

theorem problem_inequality (a : ℝ) (h_pos : 0 < a) : 
  ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → a^(Real.sin x) * (a + 1)^(Real.cos x) ≥ a :=
by 
  sorry

end problem_inequality_l341_341830


namespace largest_number_with_sum_18_l341_341587

def digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def different_digits (n : Nat) : Prop :=
  ∀ i j : Nat, i ≠ j → (Nat.digits 10 n).get? i ≠ (Nat.digits 10 n).get? j

def sum_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

theorem largest_number_with_sum_18 : ∃ n : Nat, sum_digits n = 18 ∧ different_digits n ∧ ∀ m : Nat, (sum_digits m = 18 ∧ different_digits m) → n ≥ m :=
begin
  use 543210,
  split,
  { -- Proving the sum of digits of 543210 is 18
    sorry },
  split,
  { -- Proving the digits of 543210 are all different
    sorry },
  { -- Proving that 543210 is the largest number under given conditions
    sorry }
end

end largest_number_with_sum_18_l341_341587


namespace intersection_complement_l341_341133

variable (U : Set ℝ) (M N : Set ℝ)

def U : Set ℝ := Set.univ -- Universal set, ℝ
def M : Set ℝ := {x | -2 < x ∧ x < 2} -- M = { x | -2 < x < 2 }
def N : Set ℝ := {x | ∃ y, y = 2^x - 1} -- N = { y | y = 2^x - 1 }

theorem intersection_complement : M \cap (U \N) = {x | -2 < x ∧ x <= -1} := by
    sorry

end intersection_complement_l341_341133


namespace savings_correct_l341_341910

noncomputable def savings (income expenditure : ℕ) : ℕ :=
income - expenditure

theorem savings_correct (I E : ℕ) (h_ratio :  I / E = 10 / 4) (h_income : I = 19000) :
  savings I E = 11400 :=
sorry

end savings_correct_l341_341910


namespace integral_min_value_l341_341782

noncomputable def f : ℝ → ℝ := sorry

theorem integral_min_value :
  (∀ x, f(x - 1) + f(x + 1) ≥ x + f(x)) →
  ∫ x in 1..2005, f x = 2010012 := by
sorry

end integral_min_value_l341_341782


namespace motherGaveMoney_l341_341973

-- Define the given constants and fact
def initialMoney : Real := 0.85
def foundMoney : Real := 0.50
def toyCost : Real := 1.60
def remainingMoney : Real := 0.15

-- Define the unknown amount given by his mother
def motherMoney (M : Real) := initialMoney + M + foundMoney - toyCost = remainingMoney

-- Statement to prove
theorem motherGaveMoney : ∃ M : Real, motherMoney M ∧ M = 0.40 :=
by
  sorry

end motherGaveMoney_l341_341973


namespace oil_height_ratio_l341_341623

theorem oil_height_ratio
  (r h h' : ℝ)
  (h0 : h ≠ 0)
  (h1 : r ≠ 0)
  (h2 : h' = ((1 / 4) * real.pi * r^2 - (1 / 2) * r^2) * h / (real.pi * r^2)) :
  h' / h = 1 / 4 - 1 / (2 * real.pi) :=
sorry

end oil_height_ratio_l341_341623


namespace smallest_prime_dividing_sum_l341_341591

-- Define the mathematical problem using Lean 4 syntax
theorem smallest_prime_dividing_sum : 
    ∃ p : ℕ, nat.prime p ∧ p ∣ (3 ^ 11 + 5 ^ 13) ∧ ∀ q : ℕ, nat.prime q → q ∣ (3 ^ 11 + 5 ^ 13) → p ≤ q :=
by
  sorry

end smallest_prime_dividing_sum_l341_341591


namespace sam_initial_balloons_l341_341516

theorem sam_initial_balloons:
  ∀ (S : ℕ), (S - 10 + 16 = 52) → S = 46 :=
by
  sorry

end sam_initial_balloons_l341_341516


namespace marty_paint_combinations_l341_341497

theorem marty_paint_combinations :
  let colors := 6
  let tools := 4
  ∀ (room_tool box_tool : ℕ),
  room_tool ≠ box_tool →
  colors * tools * (tools - 1) = 72 :=
by { intros, simp [colors, tools], sorry }

end marty_paint_combinations_l341_341497


namespace sum_of_divisors_of_47_l341_341957

theorem sum_of_divisors_of_47 : 
  ∑ d in {1, 47}, d = 48 := 
by 
  sorry

end sum_of_divisors_of_47_l341_341957


namespace max_area_of_quadrilateral_MPNQ_l341_341446

/-- Define the parameters and conditions based on the problem statement. --/
def C1 (r : ℝ) (θ : ℝ) : Prop := (0 < r ∧ r < 4 ∧ θ = θ)
def C2 (θ : ℝ) : ℝ × ℝ := (2 + 2 * sqrt 2 * cos θ, 2 + 2 * sqrt 2 * sin θ)
def pointN (r α : ℝ) : ℝ × ℝ := (r * cos α, r * sin α)
def pointQ (r α : ℝ) : ℝ × ℝ := (r * cos (α + π / 4), r * sin (α + π / 4))
def pointP (α : ℝ) : ℝ × ℝ := (4 * sqrt 2 * sin (α + π / 4), 4 * sqrt 2 * sin (α + π / 4))
def pointM (α : ℝ) : ℝ × ℝ := (4 * sqrt 2 * sin (α + π / 2), 4 * sqrt 2 * sin (α + π / 4))

/-- Main theorem to prove the maximum area of quadrilateral MPNQ --/
theorem max_area_of_quadrilateral_MPNQ {r α : ℝ}
    (hα : 0 < α ∧ α < π / 2)
    (hr : r = 2 * sqrt 2)
    (hN : pointN r α = (r * cos α, r * sin α))
    (hQ : pointQ r (α + π / 4) = (r * cos (α + π / 4), r * sin (α + π / 4)))
    (hP : pointP α = (4 * sqrt 2 * sin (α + π / 4), 4 * sqrt 2 * sin (α + π / 4)))
    (hM : pointM α = (4 * sqrt 2 * sin (α + π / 2), 4 * sqrt 2 * sin (α + π / 4))) :
  (let S := λ a b : ℝ, 1 / 2 * a * b * sin (π / 4) in
  S (4 * sqrt 2 * sin (α + π / 4)) (4 * sqrt 2 * sin (α + π / 2)) -
  S r r = 4 + 2 * sqrt 2) :=
sorry

end max_area_of_quadrilateral_MPNQ_l341_341446


namespace q_evaluation_l341_341676

def q (x y : ℤ) : ℤ :=
if x >= 0 ∧ y >= 0 then x - y
else if x < 0 ∧ y < 0 then x + 3 * y
else 2 * x + 2 * y

theorem q_evaluation : q (q 1 (-1)) (q (-2) (-3)) = -22 := by
sorry

end q_evaluation_l341_341676


namespace number_of_seats_l341_341166

-- Define the given conditions
def total_people : ℕ := 20
def people_per_seat : ℕ := 5

-- Define the conclusion we want to prove
theorem number_of_seats (h1 : total_people = 20) (h2 : people_per_seat = 5) :
  total_people / people_per_seat = 4 :=
by {
  rw [h1, h2], -- Use conditions
  norm_num, -- Perform numeric simplifications
  sorry -- Placeholder for the proof
}

end number_of_seats_l341_341166


namespace longest_segment_in_cylinder_l341_341619

theorem longest_segment_in_cylinder
  (r : ℝ) (h : ℝ) (diameter: ℝ := 2 * r) : 
  r = 5 ∧ h = 6 → 
  ∃ longest_segment : ℝ, longest_segment = Real.sqrt (diameter^2 + h^2) ∧ longest_segment = Real.sqrt 136 :=
by
  sorry

end longest_segment_in_cylinder_l341_341619


namespace max_blocks_fit_l341_341222

def block_dimensions : ℝ × ℝ × ℝ := (3, 1, 1)
def box_dimensions : ℝ × ℝ × ℝ := (4, 3, 2)

theorem max_blocks_fit :
  let (block_l, block_w, block_h) := block_dimensions in
  let (box_l, box_w, box_h) := box_dimensions in
  (block_l ≤ box_l ∧ block_w ≤ box_w ∧ block_h ≤ box_h) →
  block_l * block_w * block_h > 0 →
  ∃ k : ℕ, k = 6 :=
by
  sorry

end max_blocks_fit_l341_341222


namespace negate_existential_l341_341915

theorem negate_existential (p : Prop) : (¬(∃ x : ℝ, x^2 - 2 * x + 2 ≤ 0)) ↔ ∀ x : ℝ, x^2 - 2 * x + 2 > 0 :=
by sorry

end negate_existential_l341_341915


namespace solve_for_b_l341_341674

theorem solve_for_b 
  (b : ℝ)
  (h : (25 * b^2) - 84 = 0) :
  b = (2 * Real.sqrt 21) / 5 ∨ b = -(2 * Real.sqrt 21) / 5 :=
by sorry

end solve_for_b_l341_341674


namespace pawns_placement_5x5_l341_341435

def number_of_ways_to_place_pawns (n : ℕ) := 
  (finset.range n).card.factorial * (finset.range n).card.factorial

theorem pawns_placement_5x5 : number_of_ways_to_place_pawns 5 = 14400 :=
by sorry

end pawns_placement_5x5_l341_341435


namespace find_h_l341_341794

theorem find_h: 
  ∃ h k, (∀ x, 2 * x ^ 2 + 6 * x + 11 = 2 * (x - h) ^ 2 + k) ∧ h = -3 / 2 :=
by
  sorry

end find_h_l341_341794


namespace necessary_but_not_sufficient_l341_341716

def is_hyperbola (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, a * x^2 + b * y^2 = c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b < 0

theorem necessary_but_not_sufficient (a b c : ℝ) (p : a * b < 0) (q : is_hyperbola a b c) :
  (∀ (a b c : ℝ), is_hyperbola a b c → a * b < 0) ∧ (¬ ∀ (a b c : ℝ), a * b < 0 → is_hyperbola a b c) :=
by
  sorry

end necessary_but_not_sufficient_l341_341716


namespace partition_convex_hull_boundary_l341_341843

-- Definition of the problem
variable (S : Finset (ℝ × ℝ))
variable [Finite S]

-- Given conditions:
-- 1. S is a finite set of points in the plane (already implied by the type Finset of ℝ × ℝ).
-- 2. |S| is even.
axiom h_even_cardinality : 2 ∣ Finset.card S

-- Target statement to prove
theorem partition_convex_hull_boundary :
  ∃ (S₁ S₂ : Finset (ℝ × ℝ)), 
  S₁ ∪ S₂ = S ∧
  Finset.card (convexHull S₁).boundary = Finset.card (convexHull S₂).boundary :=
sorry

end partition_convex_hull_boundary_l341_341843


namespace trigonometric_identity_l341_341383

theorem trigonometric_identity (α : ℝ) (m : ℝ) 
  (h₁ : m^2 + (3/5)^2 = 1) 
  (h₂ : m < 0) 
  (h₃ : sin α = 3/5) 
  (h₄ : cos α = m) : 
  (sin α - 2 * cos α) / (sin α + cos α) = -11 := 
sorry

end trigonometric_identity_l341_341383


namespace number_of_fully_covered_squares_is_48_l341_341996

-- Define the setup of the checkerboard and the disc
def checkerboard_side : ℝ := 10
def square_side : ℝ := D
def disc_diameter : ℝ := 5 * D
def disc_radius : ℝ := disc_diameter / 2

-- Condition defining what it means for a square to be fully covered
def is_fully_covered_square (i j : ℕ) : Prop :=
  let distance := Math.sqrt ((i * square_side - checkerboard_side / 2) ^ 2 +
                             (j * square_side - checkerboard_side / 2) ^ 2)
  distance ≤ disc_radius

-- The total number of squares on one quadrant fully covered
def number_of_covered_squares_per_quadrant : ℕ := 
  sorry -- the counting logic goes here

-- The total number of fully covered squares on the whole board
def total_fully_covered_squares : ℕ :=
  4 * number_of_covered_squares_per_quadrant

-- The theorem to prove
theorem number_of_fully_covered_squares_is_48 : total_fully_covered_squares = 48 :=
  sorry

end number_of_fully_covered_squares_is_48_l341_341996


namespace integral_equation_solution_l341_341895

noncomputable def φ (x : ℝ) : ℝ := 1 + x^2 / 2

theorem integral_equation_solution (x : ℝ) (φ : ℝ → ℝ) 
    (h : ∫ t in 0..x, cos (x - t) * φ t = x) : φ x = 1 + x^2 / 2 := 
    sorry

end integral_equation_solution_l341_341895


namespace vision_data_approx_l341_341164

theorem vision_data_approx (L V : ℝ) (hL : L = 5 + Real.log10 V) (hL_value : L = 4.9) :
  V ≈ 0.8 :=
by
  -- Define the approximation constant
  let c : ℝ := 1.259
  -- Assume the given approximate value
  assume h_c : Real.rpow (10 : ℝ) (1 / 10) = c
  -- Begin proof (skipped)
  sorry

end vision_data_approx_l341_341164


namespace centroid_locus_entire_space_centroid_locus_line_l341_341404

theorem centroid_locus_entire_space (l1 l2 l3 : set (ℝ × ℝ × ℝ)) (A B C : ℝ × ℝ × ℝ)
  (hA : A ∈ l1) (hB : B ∈ l2) (hC : C ∈ l3) :
  ∀ G : ℝ × ℝ × ℝ, (∃ A B C,
    G = (1 / 3) • (A + B + C) ∧ A ∈ l1 ∧ B ∈ l2 ∧ C ∈ l3) → G ∈ (ℝ × ℝ × ℝ) :=
sorry

theorem centroid_locus_line (l1 l2 l3 : set (ℝ × ℝ × ℝ)) (A B C : ℝ × ℝ × ℝ) (π : set (ℝ × ℝ × ℝ))
  (hA : A ∈ l1) (hB : B ∈ l2) (hC : C ∈ l3) (parallel_planes : ∀ (P : set (ℝ × ℝ × ℝ)), P ∈ l1 ∨ P ∈ l2 ∨ P ∈ l3 → P ∥ π) :
  ∀ G : ℝ × ℝ × ℝ, (∃ A B C,
    G = (1 / 3) • (A + B + C) ∧ A ∈ l1 ∧ B ∈ l2 ∧ C ∈ l3) → (∃ l : set (ℝ × ℝ × ℝ), G ∈ l ∧ l ∥ π) :=
sorry

end centroid_locus_entire_space_centroid_locus_line_l341_341404


namespace least_four_digit_number_l341_341223

theorem least_four_digit_number :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
    (5 ∈ (Int.digits 10 n).toList) ∧
    ((Int.digits 10 n).toList.nodup) ∧
    (∀ d ∈ (Int.digits 10 n).toList, n % d = 0) ∧
    n % 8 = 0 ∧
    n = 5136 :=
by
  sorry

end least_four_digit_number_l341_341223


namespace probability_of_A_l341_341177

open ProbabilityTheory

variable (Ω : Type) [ProbabilitySpace Ω]

variable (A B : Event Ω)

-- Events A and B are independent.
axiom indep : Independent A B

-- Conditions
axiom h1 : 0 < P A
axiom h2 : P A = 2 * P B
axiom h3 : P (A ∪ B) = 5 * P (A ∩ B)

-- Goal: Probability that Event A occurs is 1/2.
theorem probability_of_A : P A = 1/2 :=
by
  sorry

end probability_of_A_l341_341177


namespace sum_exponents_outside_radical_l341_341232

noncomputable def simplified_expression : ℕ → ℕ → ℕ → ℕ :=
  λ a b c, do
    let expr := nat.cbrt(24 * a ^ 4 * b ^ 6 * c ^ 11)
    let outside_a := 1
    let outside_b := 2
    let outside_c := 3
    outside_a + outside_b + outside_c

theorem sum_exponents_outside_radical : simplified_expression 4 6 11 = 6 :=
by sorry

end sum_exponents_outside_radical_l341_341232


namespace find_a_l341_341352

-- Define the variables and conditions
variable (a x y : ℤ)

-- Given conditions
def x_value := (x = 2)
def y_value := (y = 1)
def equation := (a * x - y = 3)

-- The theorem to prove
theorem find_a : x_value x → y_value y → equation a x y → a = 2 :=
by
  intros
  sorry

end find_a_l341_341352


namespace train_speed_is_36_kph_l341_341638

-- Define the given conditions
def distance_meters : ℕ := 1800
def time_minutes : ℕ := 3

-- Convert distance from meters to kilometers
def distance_kilometers : ℕ -> ℕ := fun d => d / 1000
-- Convert time from minutes to hours
def time_hours : ℕ -> ℚ := fun t => (t : ℚ) / 60

-- Calculate speed in kilometers per hour
def speed_kph (d : ℕ) (t : ℕ) : ℚ :=
  let d_km := d / 1000
  let t_hr := (t : ℚ) / 60
  d_km / t_hr

-- The theorem to prove the speed
theorem train_speed_is_36_kph :
  speed_kph distance_meters time_minutes = 36 := by
  sorry

end train_speed_is_36_kph_l341_341638


namespace tangent_BD_circumcircle_ADZ_l341_341470

variable {Point : Type}
variable (A B C D E X Y Z : Point)
variable (ω : Set Point -- Circumcircle of \( \triangle ABC \))
variable (BC BE AX BY ADZ : Set Point) -- Lines defined by points

-- Conditions as per the problem statement
variables
  (h1 : ∠ABC = 90)
  (h2 : ∠B < ∠C)
  (h3 : TangentAt A ω D)
  (h4 : Reflect A BC E)
  (h5 : FootPerpendicular A BE X)
  (h6 : Midpoint AX Y)
  (h7 : Meet ω BY Z)

-- Proof problem: Given the conditions, prove that \( BD \) is tangent to the circumcircle of \( ADZ \)
theorem tangent_BD_circumcircle_ADZ :
  Tangent (Line BD) (Circumcircle (Triangle ADZ)) :=
sorry

end tangent_BD_circumcircle_ADZ_l341_341470


namespace new_profit_is_220_percent_l341_341627

noncomputable def cost_price (CP : ℝ) : ℝ := 100

def initial_profit_percentage : ℝ := 60

noncomputable def initial_selling_price (CP : ℝ) : ℝ :=
  CP + (initial_profit_percentage / 100) * CP

noncomputable def new_selling_price (SP : ℝ) : ℝ :=
  2 * SP

noncomputable def new_profit_percentage (CP SP2 : ℝ) : ℝ :=
  ((SP2 - CP) / CP) * 100

theorem new_profit_is_220_percent : 
  new_profit_percentage (cost_price 100) (new_selling_price (initial_selling_price (cost_price 100))) = 220 :=
by
  sorry

end new_profit_is_220_percent_l341_341627


namespace max_value_xy_l341_341415

theorem max_value_xy (x y : ℝ) (h : x + y = 1) : xy_max_value := 
  xy_max_value = 1 / 4 :=
sorry

end max_value_xy_l341_341415


namespace solve_inequality_l341_341692

theorem solve_inequality : 
  {x : ℝ | (x^2 - 1) / (x-4)^2 ≥ 0} = (Iio (-1) ∪ Icc (-1 : ℝ) 1 ∪ Ioc 1 4 ∪ Ioi 4) := 
by 
  -- Proof will be here 
  sorry

end solve_inequality_l341_341692


namespace geometric_sequence_problem_l341_341440

variable {a : ℕ → ℝ} -- Considering the sequence is a real number sequence
variable {q : ℝ} -- Common ratio

-- Conditions
axiom a2a6_eq_16 : a 2 * a 6 = 16
axiom a4_plus_a8_eq_8 : a 4 + a 8 = 8

-- Geometric sequence definition
axiom geometric_sequence : ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_problem : a 20 / a 10 = 1 :=
  by
  sorry

end geometric_sequence_problem_l341_341440


namespace inequality_symmetric_poly_l341_341117

-- Definition of elementary symmetric polynomials (Σ_k)
noncomputable def sigma_k (k : ℕ) (xs : List ℝ) :=
  (xs.powerset.filter (λ s, s.length = k)).map (λ s, s.prod id).sum

theorem inequality_symmetric_poly (n : ℕ) (xs : List ℝ) (h_len: xs.length = n) (h_n: 2 ≤ n) :
  ∀ k, 1 ≤ k → k < n →
  (sigma_k k xs) ^ 2 ≥ (sigma_k (k-1) xs) * (sigma_k (k+1) xs) :=
by
  intros k h1 h2
  sorry

end inequality_symmetric_poly_l341_341117


namespace area_triangle_ABC_l341_341805

-- Definitions for the given conditions
def side_length_square : ℝ := real.sqrt 2
def side_length_pentagon : ℝ := real.sqrt 2

-- The theorem statement
theorem area_triangle_ABC (s : ℝ) (h_s : s = side_length_square) : 
  let A := side_length_pentagon 
    in A = s →
  let R := s / (2 * real.sin (real.pi / 5))
    in let side_triangle_ABC := 2 * R 
    in let area_triangle := (real.sqrt 3 / 4) * (side_triangle_ABC)^2
    in area_triangle = real.sqrt 3 / (2 * real.sin (real.pi / 5))^2 := 
by 
  intros 
  sorry

end area_triangle_ABC_l341_341805


namespace divisible_by_n_sequence_l341_341153

theorem divisible_by_n_sequence (n : ℕ) (h1 : n > 1) (h2 : n % 2 = 1) : 
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 ∧ n ∣ (2^k - 1) :=
by {
  sorry
}

end divisible_by_n_sequence_l341_341153


namespace ellipse_area_l341_341285

def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 9 * y^2 - 36 * y + 36 = 0

theorem ellipse_area :
  (∀ x y : ℝ, ellipse_equation x y → true) →
  (π * 1 * (4/3) = 4 * π / 3) :=
by
  intro h
  norm_num
  sorry

end ellipse_area_l341_341285


namespace expression_that_gives_value_8_l341_341784

theorem expression_that_gives_value_8 (a b : ℝ) 
  (h_eq1 : a = 2) 
  (h_eq2 : b = 2) 
  (h_roots : ∀ x, (x - a) * (x - b) = x^2 - 4 * x + 4) : 
  2 * (a + b) = 8 :=
by
  sorry

end expression_that_gives_value_8_l341_341784


namespace valid_programs_l341_341635

def course := {English : Type} ∨ {Algebra : Type} ∨ {Geometry : Type} ∨ {History : Type} ∨ {Art : Type} ∨ {Latin : Type} ∨ {Biology : Type} ∨ {Chemistry : Type}

def total_courses := 8

def required_courses := 5

def english := 1

def math_courses := 2

def science_courses := 2

def remaining_courses := total_courses - english

def total_combinations := Nat.choose remaining_courses (required_courses - english)

def non_math_combinations := Nat.choose (remaining_courses - math_courses) (required_courses - english)

def non_science_combinations := Nat.choose (remaining_courses - science_courses) (required_courses - english)

def valid_combinations := total_combinations - non_math_combinations - non_science_combinations

theorem valid_programs : valid_combinations = 25 := by
  sorry

end valid_programs_l341_341635


namespace radius_of_original_circle_l341_341277

noncomputable def original_circle_radius (L B : ℝ) (h_ratio: L / B = 6 / 5) (h_area: L * B = 29.975864606614373) : ℝ :=
  let P := 2 * (L + B)
  let r := P / (2 * Real.pi)
  r

theorem radius_of_original_circle :
  ∃ r : ℝ, original_circle_radius 5.9975862 4.9979895 (by norm_num) (by norm_num) = 3.5 :=
sorry

end radius_of_original_circle_l341_341277


namespace seating_problem_l341_341997

def seating_condition(n : ℕ) : Prop :=
  ∃ (f : Fin n → Fin n), bijective f ∧ 
    ∀ i j : Fin n, i ≠ j → (dist i j ≠ dist (f i) (f j))

theorem seating_problem :
  (¬ seating_condition 4) ∧ seating_condition 5 ∧ (¬ seating_condition 6) ∧ seating_condition 7 :=
by {
  sorry
}

def dist (i j : Fin n) : ℕ :=
  min ((i.val - j.val).natAbs) (n - (i.val - j.val).natAbs)

end seating_problem_l341_341997


namespace total_animals_l341_341657

variable (rats chihuahuas : ℕ)
variable (h1 : rats = 60)
variable (h2 : rats = 6 * chihuahuas)

theorem total_animals (rats : ℕ) (chihuahuas : ℕ) (h1 : rats = 60) (h2 : rats = 6 * chihuahuas) : rats + chihuahuas = 70 := by
  sorry

end total_animals_l341_341657


namespace inequality_solution_set_impossible_l341_341887

theorem inequality_solution_set_impossible (a b : ℝ) (h_b : b ≠ 0) : ¬ (a = 0 ∧ ∀ x, ax + b > 0 ∧ x > (b / a)) :=
by {
  sorry
}

end inequality_solution_set_impossible_l341_341887


namespace planes_parallel_l341_341596

theorem planes_parallel (n1 n2 : ℝ × ℝ × ℝ)
  (h1 : n1 = (2, -1, 0)) 
  (h2 : n2 = (-4, 2, 0)) :
  ∃ k : ℝ, n2 = k • n1 := by
  -- Proof is beyond the scope of this exercise.
  sorry

end planes_parallel_l341_341596


namespace sum_of_integers_squared_greater_sum_of_integers_satisfying_condition_l341_341551

theorem sum_of_integers_squared_greater (x : ℤ) (hx : x ^ 2 = 256 + x) : x = 16 ∨ x = -16 :=
begin
  have : x ^ 2 - x - 256 = 0,
  { rw ← hx, ring },
  apply quadratic_eq_zero_iff.mpr,
  use [16, -16],
  split;
  linarith,
end

theorem sum_of_integers_satisfying_condition : ∑ x in ({16, -16} : finset ℤ), x = 0 :=
by simp

end sum_of_integers_squared_greater_sum_of_integers_satisfying_condition_l341_341551


namespace right_triangle_HD_HA_ratio_l341_341196

-- Define the triangle with sides 9, 12, and 15
structure RightTriangle where
  a b c : ℝ
  h : a^2 + b^2 = c^2

-- Define the orthocenter H as a point where altitudes intersect
structure Orthocenter (T : RightTriangle) where
  H : ℝ × ℝ

-- Define the altitude AD to the side of length 12
structure Altitude (T : RightTriangle) where
  AD : ℝ
  altitude_condition : T.b = 12

-- Define the relationship between the distances HD and HA
def ratio_HD_HA (T : RightTriangle) (H : Orthocenter T) (alt : Altitude T) : Prop :=
  let D := (T.b / 2, 0)
  let A := (0, T.a)
  let HA := T.a
  let HD := (T.b / 2)
  HD / HA = 2/3

-- Formalize the proof problem as a theorem
theorem right_triangle_HD_HA_ratio :
  ∃ (T : RightTriangle), T.a = 9 ∧ T.b = 12 ∧ T.c = 15 ∧
  ∃ (H : Orthocenter T), 
  ∃ (alt : Altitude T), alt.altitude_condition ∧
  ratio_HD_HA T H alt :=
sorry

end right_triangle_HD_HA_ratio_l341_341196


namespace percentage_growth_in_thirty_years_l341_341437

noncomputable def population_growth_percentage : ℕ :=
  let a := 2  -- Starting cube root such that a^3 = 2^3 = 8
  let b := 5  -- Cube root after first increment
  let c := 7  -- Cube root after second increment
  let d := 8  -- Cube root after the third increment
  let initial_population := a^3 -- Population in 1991
  let final_population := d^3 -- Population in 2021
  let growth := final_population - initial_population
  let percentage_gain := (growth * 100) / initial_population
  in percentage_gain

theorem percentage_growth_in_thirty_years : 
  population_growth_percentage = 824240700 := by
  sorry

end percentage_growth_in_thirty_years_l341_341437


namespace measure_of_angle_C_l341_341818

theorem measure_of_angle_C
  (A B C : ℝ)
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1)
  (h3 : A + B + C = Real.pi) :
  C = Real.pi / 6 := 
sorry

end measure_of_angle_C_l341_341818


namespace factorize_m_square_minus_16_l341_341327

-- Define the expression
def expr (m : ℝ) : ℝ := m^2 - 16

-- Define the factorized form
def factorized_expr (m : ℝ) : ℝ := (m + 4) * (m - 4)

-- State the theorem
theorem factorize_m_square_minus_16 (m : ℝ) : expr m = factorized_expr m :=
by
  sorry

end factorize_m_square_minus_16_l341_341327


namespace inequality_holds_l341_341365

variables {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem inequality_holds
  (h_even : is_even f)
  (h_cond : ∀ x ∈ Ico 0 (π/2), f'' x * cos x + f x * sin x > 0) :
  f (π/6) < sqrt 3 * f (π/3) :=
sorry

end inequality_holds_l341_341365


namespace sum_of_divisors_of_prime_47_l341_341965

theorem sum_of_divisors_of_prime_47 : ∀ n : ℕ, prime n → n = 47 → (∑ d in (finset.filter (λ x, n % x = 0) (finset.range (n + 1))), d) = 48 := 
by {
  intros n prime_n n_is_47,
  sorry -- Proof is omitted
}

end sum_of_divisors_of_prime_47_l341_341965


namespace complex_solutions_count_l341_341407

open Complex

noncomputable def countComplexSolutions : ℕ :=
  let S := {z : ℂ | abs z < 15 ∧ exp z = (z - 2) / (z + 2)}
  S.card

theorem complex_solutions_count :
  countComplexSolutions = 2 :=
by
  sorry

end complex_solutions_count_l341_341407


namespace angle_alpha_set_l341_341897

theorem angle_alpha_set :
  ∀ (α : ℝ), (∃ (k : ℤ), α = k * π - π / 3) ↔ 
  (vertex_at_origin α ∧ initial_side_non_negative_x_axis α ∧ terminal_side_on_line α (-sqrt 3)) := 
sorry

-- Definitions of conditions (auxiliary definitions)
def vertex_at_origin (α : ℝ) : Prop :=
  True  -- Given, no specific property to define here

def initial_side_non_negative_x_axis (α : ℝ) : Prop :=
  True  -- Given, no specific property to define here

def terminal_side_on_line (α : ℝ) (m : ℝ) : Prop :=
  y = m * x  -- stating that the terminal side lies on the line y = mx

end angle_alpha_set_l341_341897


namespace min_crossing_time_proof_l341_341506

def min_crossing_time (times : List ℕ) : ℕ :=
  -- Function to compute the minimum crossing time. Note: Actual implementation skipped.
sorry

theorem min_crossing_time_proof
  (times : List ℕ)
  (h_times : times = [2, 4, 8, 16]) :
  min_crossing_time times = 30 :=
sorry

end min_crossing_time_proof_l341_341506


namespace best_value_is_cranberry_juice_l341_341273

def unit_cost (price : ℕ) (volume : ℕ) : ℕ :=
  price / volume

def best_value_cost (price_c : ℕ) (volume_c : ℕ)
                    (price_a : ℕ) (volume_a : ℕ)
                    (price_o : ℕ) (volume_o : ℕ) : ℕ :=
  min (unit_cost price_c volume_c) (min (unit_cost price_a volume_a) (unit_cost price_o volume_o))

theorem best_value_is_cranberry_juice : 
  ∀ (price_c price_a price_o : ℕ) (volume_c volume_a volume_o : ℕ),
  price_c = 84 → volume_c = 12 →
  price_a = 120 → volume_a = 16 →
  price_o = 75 → volume_o = 10 →
  best_value_cost price_c volume_c price_a volume_a price_o volume_o = unit_cost price_c volume_c :=
by
  intros
  rw [best_value_cost, unit_cost, unit_cost, unit_cost]
  rw [←Nat.div_eq_of_lt_le, ←Nat.div_eq_of_lt_le, ←Nat.div_eq_of_lt_le]
  sorry -- Proof details would follow from here

end best_value_is_cranberry_juice_l341_341273


namespace find_monic_poly_l341_341606

def monic_poly_degree_two (g : ℝ → ℝ) :=
  ∃ b c : ℝ, g = λ x, x^2 + b*x + c

theorem find_monic_poly (g : ℝ → ℝ) (h_monic : monic_poly_degree_two g) (h_g0 : g 0 = 2) (h_g1 : g 1 = 6) :
  g = (λ x, x^2 + 3*x + 2) :=
sorry

end find_monic_poly_l341_341606


namespace compute_pq_absolute_value_l341_341909

noncomputable def given_conditions : Prop :=
  let foci_ellipse := 5 in
  let foci_hyperbola := 8 in
  ∃ p q : ℝ, 
    (q^2 - p^2 = foci_ellipse^2) ∧
    (p^2 + q^2 = foci_hyperbola^2)

theorem compute_pq_absolute_value :
  given_conditions →
  ∃ p q : ℝ, |p * q| = (Real.sqrt 12371) / 2 :=
begin
  sorry
end

end compute_pq_absolute_value_l341_341909


namespace value_of_expression_l341_341774

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end value_of_expression_l341_341774


namespace y_not_multiple_of_7_y_not_multiple_of_14_y_not_multiple_of_21_y_not_multiple_of_28_l341_341835

def y : ℕ := 42 + 98 + 210 + 333 + 175 + 28

theorem y_not_multiple_of_7 : ¬ (7 ∣ y) := sorry
theorem y_not_multiple_of_14 : ¬ (14 ∣ y) := sorry
theorem y_not_multiple_of_21 : ¬ (21 ∣ y) := sorry
theorem y_not_multiple_of_28 : ¬ (28 ∣ y) := sorry

end y_not_multiple_of_7_y_not_multiple_of_14_y_not_multiple_of_21_y_not_multiple_of_28_l341_341835


namespace sum_of_divisors_of_47_l341_341955

theorem sum_of_divisors_of_47 : 
  ∑ d in {1, 47}, d = 48 := 
by 
  sorry

end sum_of_divisors_of_47_l341_341955


namespace total_marks_l341_341899

variable (E S M : Nat)

-- Given conditions
def thrice_as_many_marks_in_English_as_in_Science := E = 3 * S
def ratio_of_marks_in_English_and_Maths            := M = 4 * E
def marks_in_Science                               := S = 17

-- Proof problem statement
theorem total_marks (h1 : E = 3 * S) (h2 : M = 4 * E) (h3 : S = 17) :
  E + S + M = 272 :=
by
  sorry

end total_marks_l341_341899


namespace remainder_two_disjoint_subsets_l341_341102

theorem remainder_two_disjoint_subsets (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) :
  (let n := (3 ^ 12 - 2 * 2 ^ 12 + 1) / 2 in n % 1000 = 625) :=
by
  sorry

end remainder_two_disjoint_subsets_l341_341102


namespace three_digit_numbers_divisible_by_5_l341_341042

theorem three_digit_numbers_divisible_by_5 (digits : Finset ℕ) (h_digits : digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) : 
  let count := ((digits \ {0}).card * digits.card) * 2 
  in count = 180 := 
by
  sorry

end three_digit_numbers_divisible_by_5_l341_341042


namespace part1_part2_part3_l341_341729

-- Given conditions and definitions
def A : ℝ := 1
def B : ℝ := 3
def y1 : ℝ := sorry  -- simply a placeholder value as y1 == y2
def y2 : ℝ := y1
def y (x m n : ℝ) : ℝ := x^2 + m * x + n

-- (1) Proof of m = -4
theorem part1 (n : ℝ) (h1 : y A m n = y1) (h2 : y B m n = y2) : m = -4 := sorry

-- (2) Proof of n = 4 when the parabola intersects the x-axis at one point
theorem part2 (h : ∃ n, ∀ x : ℝ, y x (-4) n = 0 → x = (x - 2)^2) : n = 4 := sorry

-- (3) Proof of the range of real number values for a
theorem part3 (a : ℝ) (b1 b2 : ℝ) (n : ℝ) (h1 : y a (-4) n = b1) 
  (h2 : y B (-4) n = b2) (h3 : b1 > b2) : a < 1 ∨ a > 3 := sorry

end part1_part2_part3_l341_341729


namespace mia_bought_more_pencils_l341_341870

theorem mia_bought_more_pencils (p : ℝ) (n1 n2 : ℕ) 
  (price_pos : p > 0.01)
  (liam_spent : 2.10 = p * n1)
  (mia_spent : 2.82 = p * n2) :
  (n2 - n1) = 12 := 
by
  sorry

end mia_bought_more_pencils_l341_341870


namespace min_height_is_4_l341_341238

noncomputable def height_min_material (V : ℝ) : ℝ :=
let x := (V ^ (2/3)) in
V / (x^2)

theorem min_height_is_4 :
  ∀ (V : ℝ), V = 256 -> height_min_material V = 4 :=
by
  intro V hV
  rw [height_min_material]
  have h1: x = 8, from (by sorry)
  have h2: heuristic height_min_material 256 = 4, from (by sorry)
  exact h2

end min_height_is_4_l341_341238


namespace count_numbers_with_digit_3_l341_341414

def contains_digit_3 (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.contains 3

theorem count_numbers_with_digit_3 :
  let nums := List.range' 300 300 in
  (nums.filter contains_digit_3).length = 138 :=
by
  sorry

end count_numbers_with_digit_3_l341_341414


namespace points_coincide_l341_341154

variables {V : Type*} [inner_product_space ℝ V]

-- Define orthocenter function for triangle formed by three points
def orthocenter (A B C : V) : V := sorry

-- Define midpoint function for segment formed by two points
def midpoint (A B : V) : V := (A + B) / 2

theorem points_coincide {A B C D : V} :
  let H_a := orthocenter B C D in
  let M_a := midpoint A H_a in
  let H_b := orthocenter A C D in
  let M_b := midpoint B H_b in
  let H_c := orthocenter A B D in
  let M_c := midpoint C H_c in
  let H_d := orthocenter A B C in
  let M_d := midpoint D H_d in
  M_a = M_b ∧ M_b = M_c ∧ M_c = M_d :=
begin
  sorry
end

end points_coincide_l341_341154


namespace carol_optimal_choice_l341_341280

theorem carol_optimal_choice :
  (∃ c : ℝ, 0 ≤ c ∧ c ≤ 1 ∧ ∀ c', (0 ≤ c' ∧ c' ≤ 1) → 
  (probability_of_winning Alice Bob Carol Dave c ≥
   probability_of_winning Alice Bob Carol Dave c') → c = 13 / 24) :=
begin
  sorry
end

noncomputable def probability_of_winning (alice : ℝ) (bob : ℝ) (carol : ℝ) (dave : ℝ) (c : ℝ) : ℝ :=
  if 0 < c ∧ c < 1/3 then 3 * c^2
  else if 1/3 ≤ c ∧ c ≤ 2/3 then -18 * c^2 + 18 * c - 3
  else if 2/3 < c ∧ c < 1 then 1 - c
  else 0

end carol_optimal_choice_l341_341280


namespace range_of_a_l341_341480

theorem range_of_a (a : ℝ) :
  let A := (-2, 3)
  let B := (0, a)
  let circle := (x + 3)^2 + (y + 2)^2 = 1
  ∃ a ∈ (1/3 : ℝ)..(3/2 : ℝ), intersects_symmetrical_line (A, B, circle) := 
sorry

end range_of_a_l341_341480


namespace find_x_minus_y_l341_341360

-- Variables and conditions
variables (x y : ℝ)
def abs_x_eq_3 := abs x = 3
def y_sq_eq_one_fourth := y^2 = 1 / 4
def x_plus_y_neg := x + y < 0

-- Proof problem stating that x - y must equal one of the two possible values
theorem find_x_minus_y (h1 : abs x = 3) (h2 : y^2 = 1 / 4) (h3 : x + y < 0) : 
  x - y = -7 / 2 ∨ x - y = -5 / 2 :=
  sorry

end find_x_minus_y_l341_341360


namespace largest_C_property_l341_341989

noncomputable def largest_C : ℕ := 506

theorem largest_C_property :
  ∀ (a : ℕ → ℤ), (∀ i, i < 2022 → a i = 1 ∨ a i = -1) →
  ∃ (k : ℕ) (t : ℕ → ℕ), 1 ≤ k ∧ k ≤ 2022 ∧
  (∀ i, 1 ≤ i ∧ i < k → t (i + 1) - t i ≤ 2) ∧
  largest_C ≤ |∑ i in finset.range k, a (t i)| := 
sorry

end largest_C_property_l341_341989


namespace square_area_with_circles_l341_341272

theorem square_area_with_circles 
  (radius : ℝ) 
  (circle_count : ℕ) 
  (side_length : ℝ) 
  (total_area : ℝ)
  (h1 : radius = 7)
  (h2 : circle_count = 4)
  (h3 : side_length = 2 * (2 * radius))
  (h4 : total_area = side_length * side_length)
  : total_area = 784 :=
sorry

end square_area_with_circles_l341_341272


namespace dynamo_goal_distribution_l341_341322

theorem dynamo_goal_distribution (players : Finset ℕ) (goals : ℕ → ℕ)
  (h_card : players.card = 9)
  (h_goals_ge_1: ∀ p ∈ players, goals p ≥ 1)
  (h_total_goals: (players.sum goals) = 47)
  (h_max_goals: ∃ p ∈ players, goals p = 12) :
  ∃ p1 p2 ∈ players, p1 ≠ p2 ∧ goals p1 = goals p2 :=
by
  sorry

end dynamo_goal_distribution_l341_341322


namespace determine_m_l341_341789

theorem determine_m (a b : ℝ) (m : ℝ) :
  (a^2 + 2 * a * b - b^2) - (a^2 + m * a * b + 2 * b^2) = (2 - m) * a * b - 3 * b^2 →
  (∀ a b : ℝ, (2 - m) * a * b = 0) →
  m = 2 :=
by
  sorry

end determine_m_l341_341789


namespace pencil_distribution_l341_341705

theorem pencil_distribution :
  ∃ (ways : ℕ), ways = 31 ∧
  ∃ (friends : Fin 5 → ℕ), (∑ i, friends i = 10) ∧ 
  (∀ i, 1 ≤ friends i) ∧ 
  (∀ i j, abs (friends i - friends j) ≤ 3) := 
sorry

end pencil_distribution_l341_341705


namespace quadratic_roots_identity_l341_341376

theorem quadratic_roots_identity (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0) (hmn : m * n = -5) (hm_plus_n : m + n = -2) : m^2 + m * n + 2 * m = 0 :=
by {
    sorry
}

end quadratic_roots_identity_l341_341376


namespace find_a5_l341_341393

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def condition1 := a 2 * a 3 * a 4 = 1
def condition2 := a 6 * a 7 * a 8 = 64

-- General term formula of the geometric sequence
def geo_seq (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : Prop := ∃ c, (∀ n, a n = c * q ^ n)

-- Problem statement
theorem find_a5 (h1 : condition1) (h2 : condition2) (h3 : geo_seq a q) : a 5 = 2 :=
sorry

end find_a5_l341_341393


namespace ones_digit_sum_l341_341947

theorem ones_digit_sum :
  (∑ n in Finset.range 2014, (n + 1) ^ 2013) % 10 = 1 := by
  -- Define the range
  let range := Finset.range 2014
  -- Define the ones digit cycle condition
  have cycle_cond : ∀ n ∈ range, (n + 1) % 10 = ((n + 1) ^ 2013) % 10 := by sorry
  -- Apply the ones digit cycle condition to the sum
  calc
    (∑ n in range, (n + 1) ^ 2013) % 10
        = (∑ n in range, (n + 1)) % 10 : by sorry
      ... = 1 : by sorry

end ones_digit_sum_l341_341947


namespace gcd_204_85_l341_341180

theorem gcd_204_85 : Nat.gcd 204 85 = 17 :=
sorry

end gcd_204_85_l341_341180


namespace monotonic_intervals_max_value_on_interval_ratio_less_than_ae_l341_341031

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x / a - Real.exp x

theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (∀ x, x < Real.log (1 / a) → (f' x a > 0)) ∧ (∀ x, x > Real.log (1 / a) → (f' x a < 0)) := 
sorry

theorem max_value_on_interval (a : ℝ) (h : a > 0) : 
  if Real.log (1 / a) ≥ 2 ∨ (0 < a ∧ a ≤ 1 / Real.exp 2)
  then f 2 a = 2 / a - Real.exp 2
  else if 1 < Real.log (1 / a) ∧ Real.log (1 / a) < 2 ∨ (1 / Real.exp 2 < a ∧ a < 1 / Real.exp 1)
  then f (Real.log (1 / a)) a = Real.log (1 / a) / a - 1 / a
  else f 1 a = 1 / a - Real.exp 1 :=
sorry

theorem ratio_less_than_ae (x1 x2 a : ℝ) (h1 : a > 0) (h2 : x1 < x2) (h3 : f x1 a = 0) (h4 : f x2 a = 0) : 
  x1 / x2 < a * Real.exp 1 := 
sorry

end monotonic_intervals_max_value_on_interval_ratio_less_than_ae_l341_341031


namespace cost_of_materials_l341_341645

theorem cost_of_materials (initial_bracelets given_away : ℕ) (sell_price profit : ℝ)
  (h1 : initial_bracelets = 52) 
  (h2 : given_away = 8) 
  (h3 : sell_price = 0.25) 
  (h4 : profit = 8) :
  let remaining_bracelets := initial_bracelets - given_away
  let total_revenue := remaining_bracelets * sell_price
  let cost_of_materials := total_revenue - profit
  cost_of_materials = 3 := 
by
  sorry

end cost_of_materials_l341_341645


namespace negation_exists_lt_zero_l341_341916

variable {f : ℝ → ℝ}

theorem negation_exists_lt_zero :
  ¬ (∃ x : ℝ, f x < 0) → ∀ x : ℝ, 0 ≤ f x := by
  sorry

end negation_exists_lt_zero_l341_341916


namespace inequality_inequality_hold_l341_341831

theorem inequality_inequality_hold (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab_sum : a^2 + b^2 = 1/2) :
  (1 / (1 - a)) + (1 / (1 - b)) ≥ 4 :=
by
  sorry

end inequality_inequality_hold_l341_341831


namespace difference_between_maximum_and_minimum_possible_values_l341_341291

-- Defining the percentage of students who answered "Yes" and "No" initially and finally
def initial_yes_percentage : ℕ := 60
def initial_no_percentage : ℕ := 40
def final_yes_percentage : ℕ := 80
def final_no_percentage : ℕ := 20

-- Define the hypothesis for percentages adding to 100%
def percentages_are_valid (initial_yes_percentage initial_no_percentage final_yes_percentage final_no_percentage : ℕ) :=
  initial_yes_percentage + initial_no_percentage = 100 ∧ final_yes_percentage + final_no_percentage = 100

-- Define the percentage y of students who changed their answers
def percentage_changed_answers (initial_yes_percentage initial_no_percentage final_yes_percentage final_no_percentage : ℕ) : ℕ :=
  final_yes_percentage - initial_yes_percentage + (initial_no_percentage - final_no_percentage)

-- The proof problem
theorem difference_between_maximum_and_minimum_possible_values :
  percentages_are_valid initial_yes_percentage initial_no_percentage final_yes_percentage final_no_percentage →
  percentage_changed_answers initial_yes_percentage initial_no_percentage final_yes_percentage final_no_percentage = 40 :=
by
  intros h,
  sorry

end difference_between_maximum_and_minimum_possible_values_l341_341291


namespace find_log_sum_l341_341397

open Real

/-- Given conditions of the sequence -/
def seq_condition (x : ℕ → ℝ) : Prop := ∀ n : ℕ, n > 0 → log (x (n + 1)) = 1 + log (x n)

/-- The sum condition for the first 100 terms -/
def sum_condition (x : ℕ → ℝ) : Prop := (finset.range 100).sum (λ n, x (n + 1)) = 1

/-- Main theorem to be proven -/
theorem find_log_sum (x : ℕ → ℝ)
  (h1 : seq_condition x)
  (h2 : sum_condition x) :
  log ((finset.Icc 101 200).sum x) = 100 :=
sorry

end find_log_sum_l341_341397


namespace binom_9_5_equals_126_l341_341304

theorem binom_9_5_equals_126 : Nat.binom 9 5 = 126 := 
by 
  sorry

end binom_9_5_equals_126_l341_341304


namespace paul_min_correct_answers_l341_341914

theorem paul_min_correct_answers (c i u : ℕ) (total_questions : ℕ) (total_score : ℤ) 
  (correct_points incorrect_points unanswered_points : ℤ)
  (answered questions_left : ℕ) :
  unanswered_points = 2 ->
  correct_points = 7 ->
  incorrect_points = -2 ->
  total_questions = 25 ->
  questions_left = 7 ->
  answered = 18 ->
  total_score = 90 ->
  (correct_points * (c : ℤ) + incorrect_points * (i : ℤ) + unanswered_points * (u : ℤ) ≥ total_score)
  := 
begin
  intros h_unanswered_points h_correct_points h_incorrect_points h_total_questions
         h_questions_left h_answered h_total_score,
  let points_from_unanswered := unanswered_points * questions_left,
  let points_needed := total_score - points_from_unanswered,
  have h_points_from_attempted : correct_points * (c : ℤ) + incorrect_points * (i : ℤ) 
    - unanswered_points * questions_left ≥ total_score - unanswered_points * questions_left, 
  sorry
end

end paul_min_correct_answers_l341_341914


namespace arrange_cyclic_sequence_l341_341620

-- Variables and definitions based on the problem conditions
variable {G : Type*} [Group G] [Fintype G] (g h : G) [IsGenerated G {g, h}]
 
-- Lean statement for proof problem
theorem arrange_cyclic_sequence (n : ℕ) (G : Type*) [Group G] [Fintype G]
  (hn : Fintype.card G = n) (g h : G) [IsGeneratedBy G {g, h}] :
  ∃ (s : Fin 2n → G), 
    (∀ i : Fin 2n, s ⟨i + 1 % 2n, (Nat.mod_lt _ (Nat.succ_pos')).1⟩ = g * s ⟨i, sorry⟩ ∨ s ⟨i + 1 % 2n, _⟩ = h * s ⟨i, sorry⟩) ∧
    (s 0 = g * s ⟨2n-1, sorry⟩ ∨ s 0 = h * s ⟨2n-1, sorry⟩) :=
  sorry

end arrange_cyclic_sequence_l341_341620


namespace transformation_matrix_is_correct_l341_341006

noncomputable def rotation_matrix_30 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    [Real.sqrt 3 / 2, -1 / 2],
    [1 / 2, Real.sqrt 3 / 2]
  ]

def scaling_matrix_2 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    [2, 0],
    [0, 2]
  ]

def combined_transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  scaling_matrix_2 ⬝ rotation_matrix_30

theorem transformation_matrix_is_correct :
  combined_transformation_matrix = ![
    [Real.sqrt 3, -1],
    [1, Real.sqrt 3]
  ] :=
sorry

end transformation_matrix_is_correct_l341_341006


namespace binom_9_5_l341_341303

open Nat

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Theorem to prove binom(9, 5) = 126
theorem binom_9_5 : binom 9 5 = 126 := by
  sorry

end binom_9_5_l341_341303


namespace bisection_interval_length_l341_341427

theorem bisection_interval_length (n : ℕ) : 
  (1 / (2:ℝ)^n) ≤ 0.01 → n ≥ 7 :=
by 
  sorry

end bisection_interval_length_l341_341427


namespace alice_original_number_l341_341279

theorem alice_original_number : ∃ y : ℤ, 3 * (3 * y + 15) = 135 ∧ y = 10 :=
by
  use 10
  split
  · calc
      3 * (3 * 10 + 15)
        = 3 * 45 : by norm_num
    ... = 135 : by norm_num
  · norm_num

end alice_original_number_l341_341279


namespace remainder_eq_x_minus_1_l341_341227

def f (x : ℂ) : ℂ := (x - 1) ^ 1001
def g (x : ℂ) : ℂ := x ^ 2 + x + 1

theorem remainder_eq_x_minus_1 : ∀ x : ℂ, ∃ q r : ℂ, f(x) = q * g(x) + r ∧ r = x - 1 :=
begin
  sorry
end

end remainder_eq_x_minus_1_l341_341227


namespace find_common_ratio_l341_341724

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

def Sn (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q ≠ 1 then a₁ * (1 - q ^ n) / (1 - q) else n * a₁

theorem find_common_ratio
    (a₁ : ℝ) (an : ℝ) (Sn_n : ℝ) (q : ℝ) (n : ℕ)
    (h₀ : a₁ = 2)
    (h₁ : an = -64)
    (h₂ : Sn_n = -42)
    (h₃ : an = geometric_sequence a₁ q n)
    (h₄ : Sn_n = Sn a₁ q n) :
  q = -2 :=
by
  sorry

end find_common_ratio_l341_341724


namespace digit_3_appears_300_times_l341_341966

-- Define the function that counts the occurrences of digit 3 in range from 1 to n
def count_digit_3 (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc x, acc + (String.toNat (String.filter (λ c, c = '3') (toString x)).headOr 0)) 0

-- Statement to prove
theorem digit_3_appears_300_times {n : ℕ} : count_digit_3 n = 300 → n = 1000 :=
sorry

end digit_3_appears_300_times_l341_341966


namespace min_value_of_func_l341_341422

theorem min_value_of_func (x : ℝ) : 
  ∃ y ∈ set.range (λ x : ℝ, (Real.cos x) ^ 2 + Real.sin x), y = -1 :=
sorry

end min_value_of_func_l341_341422


namespace ratio_avg_speeds_l341_341324

-- Definitions based on the problem conditions
def distance_A_B := 600
def time_Eddy := 3
def distance_A_C := 460
def time_Freddy := 4

-- Definition of average speeds
def avg_speed_Eddy := distance_A_B / time_Eddy
def avg_speed_Freddy := distance_A_C / time_Freddy

-- Theorem statement
theorem ratio_avg_speeds : avg_speed_Eddy / avg_speed_Freddy = 40 / 23 := 
sorry

end ratio_avg_speeds_l341_341324


namespace value_of_expression_l341_341779

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  rw [h1, h2]
  norm_num
  sorry

end value_of_expression_l341_341779


namespace odot_subtraction_l341_341917
-- Import the Mathlib library to access necessary mathematical definitions and operations

-- Define the custom operation \odot for all nonzero numbers
def odot (a b : ℝ) : ℝ := a^3 / (b + 1)

-- State the proof problem: prove the given equation
theorem odot_subtraction :
  ((odot (odot 3 2) 1) - (odot 3 (odot 2 1)) = (3591 / 10)) :=
by
  -- Skip the proof
  sorry

end odot_subtraction_l341_341917


namespace sum_f_eq_sum_g_l341_341264

-- A partition of n is a sequence of integers that sum to n
def is_partition (n : ℕ) (p : List ℕ) : Prop :=
  p.sum = n

-- f(p) is the number of 1's in p
def f (p : List ℕ) : ℕ :=
  p.count 1

-- g(p) is the number of distinct integers in p
def g (p : List ℕ) : ℕ :=
  p.toFinset.card

-- Prove that sum of f(p) equals sum of g(p) over all partitions of n
theorem sum_f_eq_sum_g (n : ℕ) :
  (finset.univ.filter (is_partition n)).sum f = (finset.univ.filter (is_partition n)).sum g :=
by
  sorry

end sum_f_eq_sum_g_l341_341264


namespace right_triangle_area_l341_341711

theorem right_triangle_area (a b : ℝ) (h : sqrt (a - 5) + (b - 4)^2 = 0) : 
  let area1 : ℝ := 6
  let area2 : ℝ := 10
  (∃ hypotenuse : ℝ, a^2 + b^2 = hypotenuse^2 ∧ (1 / 2) * a * b = area2) ∨ 
  (∃ leg : ℝ, b^2 + leg^2 = a^2 ∧ (1 / 2) * b * leg = area1) :=
by
  sorry

end right_triangle_area_l341_341711


namespace objects_meet_probability_l341_341138

structure Object where
  start_pos : ℕ × ℕ
  steps : (ℕ × ℕ) → List (ℕ × ℕ) → ℕ

noncomputable def probability_of_meeting : ℚ :=
  let a_paths := λ i, Nat.choose 4 i
  let b_paths := λ i, Nat.choose 4 i
  let prob := (0:4).sum (λ i, (a_paths i) * (b_paths i) / 2^8)
  prob

theorem objects_meet_probability :
  let A := Object.mk (0, 0) (λ pos lst, lst.length)
  let B := Object.mk (4, 5) (λ pos lst, lst.length)
  probability_of_meeting = 0.27 := by
  sorry

end objects_meet_probability_l341_341138


namespace multiples_of_number_l341_341976

theorem multiples_of_number (a b : ℤ) (q : set ℤ) (h₁ : ∀ x ∈ q, ∃ y, x = n * y)
  (h₂ : q = {x | a ≤ x ∧ x ≤ b}) (h₃ : set.card ({x ∈ q | ∃ y, x = n * y} : set ℤ) = 11)
  (h₄ : set.card ({x ∈ q | ∃ z, x = 7 * z} : set ℤ) = 21) : n = 14 := by
  sorry

end multiples_of_number_l341_341976


namespace solve_for_x_l341_341526

theorem solve_for_x (x : ℚ) :
  (3 + 1 / (2 + 1 / (3 + 3 / (4 + x)))) = 225 / 73 ↔ x = -647 / 177 :=
by sorry

end solve_for_x_l341_341526


namespace min_value_is_neg2032188_l341_341467

noncomputable def min_expression_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_neq: x ≠ y) (h_cond: x + y + 1/x + 1/y = 2022) : ℝ :=
(x + 1/y) * (x + 1/y - 2016) + (y + 1/x) * (y + 1/x - 2016)

theorem min_value_is_neg2032188 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_neq: x ≠ y) (h_cond: x + y + 1/x + 1/y = 2022) :
  min_expression_value x y h_pos_x h_pos_y h_neq h_cond = -2032188 := 
sorry

end min_value_is_neg2032188_l341_341467


namespace quadratic_eq_real_roots_l341_341431

theorem quadratic_eq_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 4 * x + 2 = 0) →
  (∃ y : ℝ, a * y^2 - 4 * y + 2 = 0) →
  a ≤ 2 ∧ a ≠ 0 :=
by sorry

end quadratic_eq_real_roots_l341_341431


namespace three_digit_non_multiples_of_8_or_6_l341_341041

theorem three_digit_non_multiples_of_8_or_6 : 
  let count_8 := 124 - 13 + 1,
      count_6 := 166 - 17 + 1,
      count_24 := 41 - 5 + 1,
      count_either_8_or_6 := count_8 + count_6 - count_24,
      total_three_digit_numbers := 999 - 100 + 1
  in total_three_digit_numbers - count_either_8_or_6 = 675 :=
by sorry

end three_digit_non_multiples_of_8_or_6_l341_341041


namespace problem_solution_l341_341718

-- Definition of the geometric sequence and the arithmetic condition
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = q * a n
def arithmetic_condition (a : ℕ → ℕ) := 2 * (a 3 + 1) = a 2 + a 4

-- Definitions used in the proof
def a_n (n : ℕ) : ℕ := 2^(n-1)
def b_n (n : ℕ) := a_n n + n
def S_5 := b_n 1 + b_n 2 + b_n 3 + b_n 4 + b_n 5

-- Proof statement
theorem problem_solution : 
  (∃ a : ℕ → ℕ, geometric_sequence a 2 ∧ arithmetic_condition a ∧ a 1 = 1 ∧ (∀ n, a n = 2^(n-1))) ∧
  S_5 = 46 :=
by
  sorry

end problem_solution_l341_341718


namespace cylinder_volume_l341_341633

theorem cylinder_volume (h : ℝ) (C : ℝ) (π_ne_zero : π ≠ 0) :
  h = 10 → C = 20 →  volume = 1000 / π := by
  -- Definitions
  let r := C / (2 * π)
  let volume := π * r^2 * h

  -- Proof steps
  intro h_eq
  intro C_eq

  -- Substituting known values
  rw [h_eq, C_eq]
  have : r = 10 / π := calc
    r = 20 / (2 * π) : by rw C_eq
    ... = 10 / π : by ring

  rw this
  show π * (10 / π)^2 * 10 = 1000 / π
  field_simp
  ring

end cylinder_volume_l341_341633


namespace train_crossing_time_l341_341637

noncomputable def time_to_cross : ℝ :=
  let train1_length := 150 -- meters
  let train1_speed := 25 * (1000 / 3600) -- converting kmph to m/s
  let train2_length := 200 -- meters
  let train2_speed := 35 * (1000 / 3600) -- converting kmph to m/s
  let relative_speed := train2_speed - train1_speed
  let total_length := train1_length + train2_length
  total_length / relative_speed

theorem train_crossing_time :
  time_to_cross ≈ 125.9 := by
  sorry

end train_crossing_time_l341_341637


namespace binom_9_5_l341_341301

open Nat

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Theorem to prove binom(9, 5) = 126
theorem binom_9_5 : binom 9 5 = 126 := by
  sorry

end binom_9_5_l341_341301


namespace proof_S6_l341_341747

-- Definitions based on conditions
def seq (a : ℕ → ℝ) := ∀ n : ℕ, a n > 0
def geometric_step (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) / a n < 1
def a3_plus_a5 (a : ℕ → ℝ) := a 3 + a 5 = 20
def a2_times_a6 (a : ℕ → ℝ) := a 2 * a 6 = 64
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n + 1), a i

-- Main theorem to prove
theorem proof_S6 (a : ℕ → ℝ) 
  (h_seq : seq a)
  (h_geom_step : geometric_step a)
  (h_a3a5 : a3_plus_a5 a)
  (h_a2a6 : a2_times_a6 a)
  : S a 6 = 126 :=
by 
  sorry

end proof_S6_l341_341747


namespace rational_of_cubic_root_sum_one_l341_341890

theorem rational_of_cubic_root_sum_one {x y : ℚ} (hx : is_rational x) (hy : is_rational y)
  (h : real.cbrt x + real.cbrt y = 1) : is_rational (real.cbrt x) ∧ is_rational (real.cbrt y) :=
sorry

end rational_of_cubic_root_sum_one_l341_341890


namespace sum_of_x_l341_341859

open Real

def f (x : ℝ) := 3 * x - 2
def f_inv (x : ℝ) := (x + 2) / 3
def f_x_inv (x : ℝ) := f (x⁻¹)

theorem sum_of_x (h : ∀ x, f_inv x = f_x_inv x → x = 1 ∨ x = -9 / 7) :
  (1 : ℝ) + -9 / 7 = -2 / 7 :=
by
  sorry

end sum_of_x_l341_341859


namespace quadratic_rewrite_as_square_of_binomial_plus_integer_l341_341764

theorem quadratic_rewrite_as_square_of_binomial_plus_integer :
    ∃ a b, ∀ x, x^2 + 16 * x + 72 = (x + a)^2 + b ∧ b = 8 :=
by
  sorry

end quadratic_rewrite_as_square_of_binomial_plus_integer_l341_341764


namespace molly_class_girls_l341_341136

theorem molly_class_girls (G : ℕ) (total_students : ℕ)
  (like_green like_yellow : ℕ) (like_pink : ℕ) :
  total_students = 30 →
  like_green = 15 →
  like_yellow = 9 →
  like_pink = (1/3 : ℚ) * G →
  like_pink + like_green + like_yellow = total_students →
  G = 18 :=
by
  intros h_total h_green h_yellow h_pink h_eq
  rw [h_total, h_green, h_yellow, h_pink] at h_eq
  norm_num at h_eq
  sorry

end molly_class_girls_l341_341136


namespace problem_statement_l341_341833

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem problem_statement:
  let r := ∑ k in finset.filter (λ n, ∑ i in finset.range (n + 1), (nat.num_divisors (i + 1)) % 2 = 1) (finset.range 2023), 1
  in sum_of_digits r = 18 :=
sorry

end problem_statement_l341_341833


namespace current_population_l341_341068

theorem current_population (P_initial : ℕ) (P_initial = 7145) 
  (died_percent : ℝ) (died_percent = 0.15)
  (left_percent : ℝ) (left_percent = 0.25) :
  ∃ P_current : ℕ, P_current = 4555 :=
by
  let died := (died_percent * P_initial).round
  let P_after_died := P_initial - died
  let left := (left_percent * P_after_died).round
  let P_current := P_after_died - left
  use P_current
  sorry

end current_population_l341_341068


namespace probability_at_least_half_correct_l341_341212

noncomputable def probability_of_at_least_half_correct_guesses : ℚ :=
  let p := (1 : ℚ) / 4;
  (∑ k in finset.range (21 - 10), nat.choose 20 (k + 10) * (p ^ (k + 10)) * ((1 - p) ^ (20 - (k + 10))))

theorem probability_at_least_half_correct :
  probability_of_at_least_half_correct_guesses = 
    (∑ k in finset.range (21 - 10), (nat.choose 20 (k + 10 : ℕ) : ℚ) *
    ((1 / 4 : ℚ) ^ (k + 10)) * ((3 / 4 : ℚ) ^ (20 - (k + 10)))) :=
sorry

end probability_at_least_half_correct_l341_341212


namespace range_of_a_l341_341481

theorem range_of_a (a : ℝ) :
  let A := (-2, 3)
  let B := (0, a)
  let circle := (x + 3)^2 + (y + 2)^2 = 1
  ∃ a ∈ (1/3 : ℝ)..(3/2 : ℝ), intersects_symmetrical_line (A, B, circle) := 
sorry

end range_of_a_l341_341481


namespace parabola_properties_l341_341734

theorem parabola_properties (p : ℝ) (hp : p = 1/2) :
  let A : ℝ × ℝ := (1, 1)
  let B : ℝ × ℝ := (0, -1)
  ∀ (P Q : ℝ × ℝ),
    P.1^2 = 2 * p * P.2 ∧ Q.1^2 = 2 * p * Q.2 ∧
    ∃ (k : ℝ), P.2 = k * P.1 - 1 ∧ Q.2 = k * Q.1 - 1 →
  
  -- Corresponds to option B: Line AB is tangent to C
  let m_AB := (A.2 - B.2) / (A.1 - B.1) in
  let line_AB := λ x : ℝ, m_AB * x + B.2 in
  (A.2 = line_AB A.1) →
  
  -- Corresponds to option C: |OP| * |OQ| ≥ |OA|^2
  let OA := real.sqrt (A.1^2 + A.2^2) in
  let OP := real.sqrt (P.1^2 + P.2^2) in
  let OQ := real.sqrt (Q.1^2 + Q.2^2) in
  OP * OQ ≥ OA^2 ∧
  
  -- Corresponds to option D: |BP| * |BQ| > |BA|^2
  let BA := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) in
  let BP := real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2 + 1)^2) in
  let BQ := real.sqrt ((Q.1 - B.1)^2 + (Q.2 - B.2 + 1)^2) in
  BP * BQ > BA^2 :=
sorry

end parabola_properties_l341_341734


namespace remainder_S_div_500_l341_341841

def R : Set ℕ := { r | ∃ n : ℕ, r = (3^n) % 500 }

def S : ℕ := ∑ r in R.to_finset, r

theorem remainder_S_div_500 : S % 500 = 453 :=
by sorry

end remainder_S_div_500_l341_341841


namespace count_squares_within_region_l341_341769

noncomputable def countSquares : Nat := sorry

theorem count_squares_within_region :
  countSquares = 45 :=
sorry

end count_squares_within_region_l341_341769


namespace angle_BCA_is_60_l341_341443

theorem angle_BCA_is_60 (a b c : ℝ) (h : 2 * (a + c) = (c - a) ^ 2 + b ^ 2) :
  ∠ BCA = 60 := 
by {
  sorry
}

end angle_BCA_is_60_l341_341443


namespace worker_bees_hive_empty_l341_341059

theorem worker_bees_hive_empty:
  ∀ (initial_worker: ℕ) (leave_nectar: ℕ) (reassign_guard: ℕ) (return_trip: ℕ) (multiplier: ℕ),
  initial_worker = 400 →
  leave_nectar = 28 →
  reassign_guard = 30 →
  return_trip = 15 →
  multiplier = 5 →
  ((initial_worker - leave_nectar - reassign_guard + return_trip) * (1 - multiplier)) = 0 :=
by
  intros initial_worker leave_nectar reassign_guard return_trip multiplier
  sorry

end worker_bees_hive_empty_l341_341059


namespace symmetrical_line_intersects_circle_l341_341487

variables (A_x A_y B_x B_y a x y : ℝ)

def points_A_B (A_x A_y B_x B_y : ℝ) : Prop :=
  A_x = -2 ∧ A_y = 3 ∧ B_x = 0

def circle (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y + 2) ^ 2 = 1

theorem symmetrical_line_intersects_circle (a : ℝ) :
  (∀ x y, points_A_B (-2) 3 0 a → (circle x y → ∃ (x y : ℝ), True)) →
  (a ∈ set.Icc (1 / 3 : ℝ) (3 / 2 : ℝ)) :=
by
  intros h
  sorry

end symmetrical_line_intersects_circle_l341_341487


namespace polynomial_range_l341_341046

def p (x : ℝ) : ℝ := x^4 - 4*x^3 + 8*x^2 - 8*x + 5

theorem polynomial_range : ∀ x : ℝ, p x ≥ 2 :=
by
sorry

end polynomial_range_l341_341046


namespace fx_greater_than_2_l341_341760

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log x

theorem fx_greater_than_2 :
  ∀ x : ℝ, x > 0 → f x > 2 :=
by {
  sorry
}

end fx_greater_than_2_l341_341760


namespace find_a_l341_341351

-- Define the variables and conditions
variable (a x y : ℤ)

-- Given conditions
def x_value := (x = 2)
def y_value := (y = 1)
def equation := (a * x - y = 3)

-- The theorem to prove
theorem find_a : x_value x → y_value y → equation a x y → a = 2 :=
by
  intros
  sorry

end find_a_l341_341351


namespace minimal_distance_sum_l341_341344

open EuclideanGeometry

def Point := ℝ × ℝ

structure Quadrilateral :=
  (A B C D : Point)
  (convex : ConvexPolygon [A, B, C, D])

def intersection_of_diagonals (q : Quadrilateral) : Point :=
  sorry -- Assume a function that computes the intersection

theorem minimal_distance_sum (q : Quadrilateral) :
  let M := intersection_of_diagonals q in
  ∀ K : Point, 
    let dist := EuclideanDistance in
    dist K q.A + dist K q.B + dist K q.C + dist K q.D ≥ 
    dist M q.A + dist M q.B + dist M q.C + dist M q.D :=
sorry

end minimal_distance_sum_l341_341344


namespace geometric_sequence_length_l341_341921

theorem geometric_sequence_length (a : ℕ → ℝ) (n : ℕ) (r : ℝ)
  (h1 : a 0 * a 1 * a 2 = 2)
  (h2 : a (n - 3) * a (n - 2) * a (n - 1) = 4)
  (h3 : (∏ i in Finset.range n, a i) = 64)
  (h_geom : ∀ i, a (i + 1) = r * a i) : n = 12 :=
sorry

end geometric_sequence_length_l341_341921


namespace pyramid_signs_combination_count_l341_341442

-- Define the conditions
def bottom_cells (a b c d e : Int) : Prop :=
  (a = 1 ∨ a = -1) ∧ (b = 1 ∨ b = -1) ∧ (c = 1 ∨ c = -1) ∧ (d = 1 ∨ d = -1) ∧ (e = 1 ∨ e = -1)

def signs_match (a b : Int) : Int := if a = b then 1 else -1

def top_cell_value (a b c d e : Int) : Int :=
  let r2_1 := signs_match a b
  let r2_2 := signs_match b c
  let r2_3 := signs_match c d
  let r2_4 := signs_match d e
  let r3_1 := signs_match r2_1 r2_2
  let r3_2 := signs_match r2_2 r2_3
  let r3_3 := signs_match r2_3 r2_4
  let r4_1 := signs_match r3_1 r3_2
  let r4_2 := signs_match r3_2 r3_3
  signs_match r4_1 r4_2

def odd_minus_signs (a b c d e : Int) : Prop :=
  (a + b + c + d + e) % 2 ≠ 0

-- The theorem
theorem pyramid_signs_combination_count :
  ∃ (a b c d e : Int), bottom_cells a b c d e ∧ top_cell_value a b c d e = 1 ∧ odd_minus_signs a b c d e ∧ 
  fintype.card { x : Int × Int × Int × Int × Int // bottom_cells x.1 x.2 x.3 x.4 x.5 ∧ top_cell_value x.1 x.2 x.3 x.4 x.5 = 1 ∧ odd_minus_signs x.1 x.2 x.3 x.4 x.5 } = 8 :=
sorry

end pyramid_signs_combination_count_l341_341442


namespace slope_angle_line_l341_341545

theorem slope_angle_line (a : ℝ) : ∃ α : ℝ, 0 ≤ α ∧ α < 180 ∧ tan α = sqrt 3 ∧ α = 60 := 
by
  sorry

end slope_angle_line_l341_341545


namespace sum_of_integers_satisfying_equation_l341_341556

theorem sum_of_integers_satisfying_equation : (∑ x in { x : ℤ | x^2 = 256 + x }, x) = 0 := 
sorry

end sum_of_integers_satisfying_equation_l341_341556


namespace sum_of_possible_values_l341_341862

theorem sum_of_possible_values (a b c d : ℝ) (h1 : |a - b| = 3) (h2 : |b - c| = 4) (h3 : |c - d| = 5) (ha : a = 0) :
  set.sum { |a - d| | ∃ b c, |a - b| = 3 ∧ |b - c| = 4 ∧ |c - d| = 5 } = 24 :=
by
  sorry

end sum_of_possible_values_l341_341862


namespace budget_circle_salaries_degrees_l341_341240

theorem budget_circle_salaries_degrees :
  let transportation := 20
  let research_development := 9
  let utilities := 5
  let equipment := 4
  let supplies := 2
  let total_percent := 100
  let full_circle_degrees := 360
  let total_allocated_percent := transportation + research_development + utilities + equipment + supplies
  let salaries_percent := total_percent - total_allocated_percent
  let salaries_degrees := (salaries_percent * full_circle_degrees) / total_percent
  salaries_degrees = 216 :=
by
  sorry

end budget_circle_salaries_degrees_l341_341240


namespace binom_9_5_l341_341298

theorem binom_9_5 : nat.binomial 9 5 = 126 := by
  sorry

end binom_9_5_l341_341298


namespace problem1_extr_vals_l341_341986

-- Definitions from conditions
def f (x : ℝ) := x^3 - 2 * x^2 + x
def g (x : ℝ) := f x + 1

theorem problem1_extr_vals :
  ∃ a b : ℝ, a = g (1/3) ∧ b = g 1 ∧ a = 31/27 ∧ b = 1 :=
by
  sorry

end problem1_extr_vals_l341_341986


namespace circle_length_l341_341522

theorem circle_length (n : ℕ) (arm_span : ℝ) (overlap : ℝ) (contribution : ℝ) (total_length : ℝ) :
  n = 16 ->
  arm_span = 10.4 ->
  overlap = 3.5 ->
  contribution = arm_span - overlap ->
  total_length = n * contribution ->
  total_length = 110.4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end circle_length_l341_341522


namespace enemy_plane_hit_probability_l341_341610

theorem enemy_plane_hit_probability : 
  let p_A_hit := 0.6
  let p_B_hit := 0.5
  let p_A_miss := 1 - p_A_hit
  let p_B_miss := 1 - p_B_hit
  let p_both_miss := p_A_miss * p_B_miss
  let p_hit := 1 - p_both_miss
  p_hit = 0.8 :=
by
  simp only [p_A_hit, p_B_hit, p_A_miss, p_B_miss, p_both_miss, p_hit]
  norm_num
  sorry

end enemy_plane_hit_probability_l341_341610


namespace monotonically_decreasing_range_l341_341389

def f (x a : ℝ) : ℝ := Real.log x - x * Real.exp x + a * x

theorem monotonically_decreasing_range (a : ℝ) :
  (∀ x ∈ Set.Ici 1, deriv (λ x, f x a) x ≤ 0) → a ≤ 2 * Real.exp 1 - 1 :=
sorry

end monotonically_decreasing_range_l341_341389


namespace inequality_proof_l341_341471

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + b^2 + c^2 + a * b * c = 4) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 + a * b * c ≤ 4 := by
  sorry

end inequality_proof_l341_341471


namespace initial_distance_between_jack_and_christina_l341_341820

theorem initial_distance_between_jack_and_christina
  (jack_speed : ℝ)
  (christina_speed : ℝ)
  (lindy_speed : ℝ)
  (lindy_total_distance : ℝ)
  (meeting_time : ℝ)
  (combined_speed : ℝ) :
  jack_speed = 5 ∧
  christina_speed = 3 ∧
  lindy_speed = 9 ∧
  lindy_total_distance = 270 ∧
  meeting_time = lindy_total_distance / lindy_speed ∧
  combined_speed = jack_speed + christina_speed →
  meeting_time = 30 ∧
  combined_speed = 8 →
  (combined_speed * meeting_time) = 240 :=
by
  sorry

end initial_distance_between_jack_and_christina_l341_341820


namespace yella_computer_usage_l341_341974

theorem yella_computer_usage (last_week_usage : ℕ) (usage_reduction : ℕ) (days_in_week : ℕ) :
  last_week_usage = 91 → usage_reduction = 35 → days_in_week = 7 →
  (last_week_usage - usage_reduction) / days_in_week = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end yella_computer_usage_l341_341974


namespace trajectory_of_P_l341_341369

-- Definitions
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (4, 0)
def line_m (P : ℝ × ℝ) (k1 : ℝ) : Prop := P.2 = k1 * (P.1 - A.1)
def line_n (P : ℝ × ℝ) (k2 : ℝ) : Prop := P.2 = k2 * (P.1 - B.1)
def line_l (l : ℝ → ℝ) := ∀ x, l x = k * (x - 2)

-- Proof statement
theorem trajectory_of_P (k1 k2 a : ℝ) (h : k1 * k2 = a) : 
  let E_equation :=  ∀ x y, y ≠ 0 → (y^2 / (x^2 - 16) = a) ↔ (x^2 / 16 - y^2 / (16 * a) = 1) 
  ∧ (a > 0 → "Hyperbola excluding points (-4,0) and (4,0)")
  ∧ (-1 < a ∧ a < 0 → "Ellipse with foci on x-axis")
  ∧ (a = -1 → "Circle")
  ∧ (a < -1 → "Ellipse with foci on y-axis") 
  ∧ a = -1/4 → (∀ line_l k, 
            ∀ C D : ℝ × ℝ, 
            "No such line l exists such that the circle with diameter CD passes through B") := 
by
  sorry

end trajectory_of_P_l341_341369


namespace school_travel_time_is_12_l341_341093

noncomputable def time_to_school (T : ℕ) : Prop :=
  let extra_time := 6
  let total_distance_covered := 2 * extra_time
  T = total_distance_covered

theorem school_travel_time_is_12 :
  ∃ T : ℕ, time_to_school T ∧ T = 12 :=
by
  sorry

end school_travel_time_is_12_l341_341093


namespace sqrt_four_root_l341_341690

theorem sqrt_four_root (x y z : ℝ) (h1 : x = 8) (h2 : y = 24.75) (h3 : z = 99) :
  sqrt (sqrt (sqrt (sqrt (x / y)))) = 2 / (sqrt (sqrt (sqrt (sqrt z)))) :=
by
  sorry

end sqrt_four_root_l341_341690


namespace find_B_squared_l341_341332

noncomputable def g (x : ℝ) : ℝ := real.sqrt 28 + 64 / x

theorem find_B_squared : 
  let B := (abs ((real.sqrt 28 + real.sqrt 284) / 2) + abs ((real.sqrt 28 - real.sqrt 284) / 2)) in
  B^2 = 284 := 
by
  sorry

end find_B_squared_l341_341332


namespace cost_price_percentage_l341_341905

/-- The cost price (CP) as a percentage of the marked price (MP) given 
that the discount is 18% and the gain percent is 28.125%. -/
theorem cost_price_percentage (MP CP : ℝ) (h1 : CP / MP = 0.64) : 
  (CP / MP) * 100 = 64 :=
by
  sorry

end cost_price_percentage_l341_341905


namespace unique_increasing_function_on_neg_infinity_to_zero_l341_341281

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 1
noncomputable def f_B (x : ℝ) : ℝ := 1 - 1 / x
noncomputable def f_C (x : ℝ) : ℝ := x^2 - 5 * x - 6
noncomputable def f_D (x : ℝ) : ℝ := 3 - x

theorem unique_increasing_function_on_neg_infinity_to_zero :
  ∀ (x : ℝ), x < 0 → 
    (f_B x ≥ f_B (-1) : ℝ → f_A x < f_A (-1) ∧ f_C x < f_C (-1) ∧ f_D x < f_D (-1)) :=
by
  sorry

end unique_increasing_function_on_neg_infinity_to_zero_l341_341281


namespace PA_PB_eq_two_l341_341450

noncomputable def parametricL (t : ℝ) : ℝ × ℝ :=
  (1 / 2 * t, 1 + (sqrt 3) / 2 * t)

noncomputable def parametricC (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 2 * Real.sin θ)

def P : ℝ × ℝ := (0, 1)

theorem PA_PB_eq_two : ∃ t1 t2 : ℝ, 
  (let (x1, y1) := parametricL t1 in
  let (x2, y2) := parametricL t2 in
  (sqrt ((x1 - P.1) ^ 2 + (y1 - P.2) ^ 2)) * 
  (sqrt ((x2 - P.1) ^ 2 + (y2 - P.2) ^ 2)) = 2) ∧
  (x1-1)^2 + y1^2 = 4 ∧
  (x2-1)^2 + y2^2 = 4 := 
by sorry

end PA_PB_eq_two_l341_341450


namespace value_of_expression_l341_341926

def expr : ℕ :=
  8 + 2 * (3^2)

theorem value_of_expression : expr = 26 :=
  by
  sorry

end value_of_expression_l341_341926


namespace probability_odd_product_greater_than_15_l341_341892

-- Definitions of conditions
def balls := {1, 2, 3, 4, 5, 6}

-- Lean proof problem statement
theorem probability_odd_product_greater_than_15 :
  let outcomes : Finset (ℕ × ℕ) := Finset.product (Finset.of_set balls) (Finset.of_set balls),
  odds := {n ∈ balls | n % 2 = 1},
  success := ((5, 5) ∈ outcomes)
  in ((Finset.card success).toNat : ℚ) / (Finset.card outcomes).toNat = 1 / 36 := by 
sorry

end probability_odd_product_greater_than_15_l341_341892


namespace bisect_segments_l341_341080

-- Define the given conditions
variables {A B C A₁ B₁ C₁ Cₐ C_b : Point}
variables (hA₁ : IsAltitude A₁ A B C)
variables (hB₁ : IsAltitude B₁ B A C)
variables (hC₁ : IsAltitude C₁ C A B)
variables (hCₐ : Proj C₁ A C A₁)
variables (hC_b : Proj C₁ B C B₁)

-- Define what we want to prove
theorem bisect_segments :
  Bisects_segment Cₐ C_b C₁ A₁ ∧ Bisects_segment Cₐ C_b C₁ B₁ :=
sorry

end bisect_segments_l341_341080


namespace range_of_a_l341_341476

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 1 ∧ y = a + (3 - a)/2 * x) →
  (a ∈ set.Icc (1 / 3) (3 / 2)) :=
by
  sorry

end range_of_a_l341_341476


namespace sam_more_than_sarah_l341_341078

-- Defining the conditions
def street_width : ℤ := 25
def block_length : ℤ := 450
def block_width : ℤ := 350
def alleyway : ℤ := 25

-- Defining the distances run by Sarah and Sam
def sarah_long_side : ℤ := block_length + alleyway
def sarah_short_side : ℤ := block_width
def sam_long_side : ℤ := block_length + 2 * street_width
def sam_short_side : ℤ := block_width + 2 * street_width

-- Defining the total distance run by Sarah and Sam in one lap
def sarah_total_distance : ℤ := 2 * sarah_long_side + 2 * sarah_short_side
def sam_total_distance : ℤ := 2 * sam_long_side + 2 * sam_short_side

-- Proving the difference between Sam's and Sarah's running distances
theorem sam_more_than_sarah : sam_total_distance - sarah_total_distance = 150 := by
  -- The proof is omitted
  sorry

end sam_more_than_sarah_l341_341078


namespace ratio_of_men_to_women_l341_341515

-- Define constants
def total_people : ℕ := 60
def men_in_meeting : ℕ := 4
def women_in_meeting : ℕ := 6
def women_reduction_percentage : ℕ := 20

-- Statement of the problem
theorem ratio_of_men_to_women (total_people men_in_meeting women_in_meeting women_reduction_percentage: ℕ)
  (total_people_eq : total_people = 60)
  (men_in_meeting_eq : men_in_meeting = 4)
  (women_in_meeting_eq : women_in_meeting = 6)
  (women_reduction_percentage_eq : women_reduction_percentage = 20) :
  (men_in_meeting + ((total_people - men_in_meeting - women_in_meeting) * women_reduction_percentage / 100)) 
  = total_people / 2 :=
sorry

end ratio_of_men_to_women_l341_341515


namespace sum_of_k_values_l341_341949

theorem sum_of_k_values :
  (∃ k, ∃ x, x^2 - 4*x + 3 = 0 ∧ x^2 - 6*x + k = 0) →
  (perfect_values = {5, 9}) →
  (∑ i in perfect_values, i = 14) := by
  sorry

end sum_of_k_values_l341_341949


namespace tangent_line_at_zero_l341_341696

noncomputable def tangent_line_equation (f : ℝ → ℝ) (f' : ℝ → ℝ) (x0 : ℝ) : ℝ → ℝ :=
  λ x, f x0 + f' x0 * (x - x0)

def function_f (x : ℝ) : ℝ := x^2 + Real.exp x
def derivative_f (x : ℝ) : ℝ := 2 * x + Real.exp x
def point_of_tangency := (0 : ℝ, 1 : ℝ)
def expected_tangent_line (x : ℝ) : ℝ := x + 1

theorem tangent_line_at_zero :
  tangent_line_equation function_f derivative_f 0 = expected_tangent_line := 
  by 
    sorry

end tangent_line_at_zero_l341_341696


namespace ball_arrangement_divisibility_l341_341250

theorem ball_arrangement_divisibility :
  ∀ (n : ℕ), (∀ (i : ℕ), i < n → (∃ j k l m : ℕ, j < k ∧ k < l ∧ l < m ∧ m < n ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ j
    ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m)) →
  ¬((n = 2021) ∨ (n = 2022) ∨ (n = 2023) ∨ (n = 2024)) :=
sorry

end ball_arrangement_divisibility_l341_341250


namespace chromatic_number_inequality_l341_341097

variables (G : Type) [graph G] (m : ℕ) -- Assuming we have a graph G and m represents the number of edges.

noncomputable def chromatic_number : ℕ := sorry -- We define the chromatic_number function, it is noncomputable since we skip its proof.

theorem chromatic_number_inequality (G : Type) [graph G] (m : ℕ) 
  (h_edges : number_of_edges G = m) :
  chromatic_number G ≤ (1 / 2) + sqrt(2 * m + 1 / 4) :=
begin
  sorry
end

end chromatic_number_inequality_l341_341097


namespace male_salmon_count_l341_341571

theorem male_salmon_count (total_salmon : ℕ) (female_salmon : ℕ) (male_salmon : ℕ) 
  (h1 : total_salmon = 971639) 
  (h2 : female_salmon = 259378) 
  (h3 : male_salmon = total_salmon - female_salmon) : 
  male_salmon = 712261 :=
by
  sorry

end male_salmon_count_l341_341571


namespace range_of_values_for_a_l341_341482

open Real

def point_A : Point := (-2, 3)

def point_B (a : ℝ) : Point := (0, a)

def circle_center : Point := (-3, -2)

noncomputable def symmetric_line_distance (a : ℝ) : ℝ :=
  let A := 3 - a
  let B := -2
  let C := 2 * a
  let x_0 := -3
  let y_0 := -2
  abs (A * x_0 + B * y_0 + C) / sqrt (A ^ 2 + B ^ 2)

def valid_range : set ℝ := set.Icc (1 / 3) (3 / 2)

theorem range_of_values_for_a (a : ℝ) : symmetric_line_distance a ≤ 1 → a ∈ valid_range :=
  sorry

end range_of_values_for_a_l341_341482


namespace parallel_lines_condition_l341_341249

variable {a : ℝ}

theorem parallel_lines_condition (a_is_2 : a = 2) :
  (∀ x y : ℝ, a * x + 2 * y = 0 → x + y = 1) ∧ (∀ x y : ℝ, x + y = 1 → a * x + 2 * y = 0) :=
by
  sorry

end parallel_lines_condition_l341_341249


namespace trees_saved_by_schools_l341_341886

def trees_saved_per_tonne_paper : ℕ := 24

def tonnes_paper_recycled_per_school : ℚ := 3 / 4

def number_of_schools : ℕ := 4

theorem trees_saved_by_schools :
  let total_tonnes_recycled := number_of_schools * tonnes_paper_recycled_per_school in
  total_tonnes_recycled * trees_saved_per_tonne_paper = 72 :=
by
  sorry

end trees_saved_by_schools_l341_341886


namespace total_passengers_wearing_hats_l341_341876

theorem total_passengers_wearing_hats
(total_passengers : ℕ)
(proportion_men : ℚ)
(proportion_women_hat : ℚ)
(proportion_men_hat : ℚ)
(h1 : total_passengers = 1500)
(h2 : proportion_men = 0.40)
(h3 : proportion_women_hat = 0.15)
(h4 : proportion_men_hat = 0.12) :
  let men := proportion_men * total_passengers in
  let women := total_passengers - men in
  let women_wearing_hats := proportion_women_hat * women in
  let men_wearing_hats := proportion_men_hat * men in
  women_wearing_hats + men_wearing_hats = 207 :=
by sorry

end total_passengers_wearing_hats_l341_341876


namespace probability_of_rerolling_three_dice_l341_341089

/--
Jason rolls four fair six-sided dice. He then looks at the rolls and chooses a subset of the dice 
(possibly empty, possibly all four dice) to reroll. He wins if and only if the sum of the numbers 
face up on the four dice is exactly 9. Jason always plays to optimize his chances of winning.
Prove that the probability that he chooses to reroll exactly three of the dice is 11/216.
-/
theorem probability_of_rerolling_three_dice : 
  let dice_rolls := {sum : ℕ // 4 ≤ sum ∧ sum ≤ 24} -- Possible sums of four 6-sided dice.
  (∃ (reroll_strategy : finset ℕ → finset ℕ),
    (∀ (rolls : finset ℕ), reroll_strategy rolls = {sum | sum = 9 - sum(rolls)}) ∧
    (probability (reroll_strategy = 3) = 11 / 216)) :=
sorry

end probability_of_rerolling_three_dice_l341_341089


namespace prob_D_correct_l341_341642

def prob_A : ℚ := 1 / 4
def prob_B : ℚ := 1 / 3
def prob_C : ℚ := 1 / 6
def total_prob (prob_D : ℚ) : Prop := prob_A + prob_B + prob_C + prob_D = 1

theorem prob_D_correct : ∃ (prob_D : ℚ), total_prob prob_D ∧ prob_D = 1 / 4 :=
by
  -- Proof omitted
  sorry

end prob_D_correct_l341_341642


namespace ratio_of_areas_l341_341150

variables {A B C O : Type}

-- Conditions
axiom O_inside_triangle : ∃ A B C O : Type, 
  vector3 A B C O /\ (vector3.oa + vector3.ob + vector3.oc = 0)

-- Ratio of areas
theorem ratio_of_areas (h : vector3.oa + vector3.ob + vector3.oc = 0) : 
  area(triangle_abc) / area(concave_quad_aboc) = 3 / 2 :=
sorry

end ratio_of_areas_l341_341150


namespace circumcenter_equidistant_l341_341919
-- Import all necessary libraries from Mathlib

-- Define the context for a triangle, its vertices, and the claim about the circumcenter and equidistance.
theorem circumcenter_equidistant {A B C : Point} (T : Triangle A B C) :
  let circumcenter := circumcenter T in
  (dist circumcenter A = dist circumcenter B) ∧ (dist circumcenter B = dist circumcenter C) :=
sorry

end circumcenter_equidistant_l341_341919


namespace imaginary_part_of_z_l341_341385

open Complex

theorem imaginary_part_of_z : let z := (1 - I) / (1 + 3 * I) in
  Complex.im z = -2 / 5 :=
by
  let z := ((1 : ℂ) - I) / ((1 : ℂ) + 3 * I)
  show Complex.im z = -2 / 5
  sorry

end imaginary_part_of_z_l341_341385


namespace triangle_statements_correct_l341_341380

theorem triangle_statements_correct :
  ∀ (A B C : ℝ) (a b c : ℝ),
  (a = 1 ∧ b = 2 → A ≠ π / 3) ∧
  (A = π / 6 ∧ a = 1 ∧ c = sqrt 3 → b ≠ 1) ∧
  (∀ A B C : ℝ, (A < π / 2 ∧ B < π / 2 ∧ C < π / 2) ∧ a = 2 ∧ b = 3 → sqrt 5 < c ∧ c < sqrt 13) ∧
  (∀ A B C : ℝ, sin A ^ 2 ≤ sin B ^ 2 + sin C ^ 2 - sin B * sin C → 0 < A ∧ A ≤ π / 3) :=
by
  sorry

end triangle_statements_correct_l341_341380


namespace remainder_S_div_500_l341_341840

def R : Set ℕ := { r | ∃ n : ℕ, r = (3^n) % 500 }

def S : ℕ := ∑ r in R.to_finset, r

theorem remainder_S_div_500 : S % 500 = 453 :=
by sorry

end remainder_S_div_500_l341_341840


namespace quadratic_has_real_roots_l341_341347

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 + 4 * x + k = 0) ↔ k ≤ 4 := by
  sorry

end quadratic_has_real_roots_l341_341347


namespace chromatic_number_plane_l341_341532

open Real

def chromatic_number (plane : ℝ × ℝ → ℕ) : ℕ :=
  ∃ (χ : ℕ), (∀ x y : ℝ × ℝ, (dist x y = 1) → (plane x ≠ plane y))
  ∧ (∀ x : ℝ × ℝ, plane x < χ)
  
theorem chromatic_number_plane (χ : ℕ) :
  4 ≤ χ ∧ χ ≤ 7 :=
begin
  sorry
end

end chromatic_number_plane_l341_341532


namespace quadratic_has_real_roots_l341_341763

-- Define the condition that a quadratic equation has real roots given ac < 0

variable {a b c : ℝ}

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_real_roots (h : a * c < 0) : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by
  sorry

end quadratic_has_real_roots_l341_341763


namespace lambda_range_decreasing_difference_l341_341055

section

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ)

-- Condition: a is a decreasing difference sequence
def decreasing_difference (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → a (n + 1) - a n < a (n + 2) - a (n + 1)

-- Condition: The sequence a_n and the sum S_n satisfy the given equation
def satisfies_sum_equation (a S : ℕ → ℝ) (λ : ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → 2 * S n = 3 * a n + 2 * λ - 1

theorem lambda_range_decreasing_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ)
  (h1 : decreasing_difference a) 
  (h2 : satisfies_sum_equation a S λ) : 
  λ > 1 / 2 :=
sorry

end

end lambda_range_decreasing_difference_l341_341055


namespace index_cards_per_student_l341_341294

theorem index_cards_per_student
    (periods_per_day : ℕ)
    (students_per_class : ℕ)
    (cost_per_pack : ℕ)
    (total_spent : ℕ)
    (cards_per_pack : ℕ)
    (total_packs : ℕ)
    (total_index_cards : ℕ)
    (total_students : ℕ)
    (index_cards_per_student : ℕ)
    (h1 : periods_per_day = 6)
    (h2 : students_per_class = 30)
    (h3 : cost_per_pack = 3)
    (h4 : total_spent = 108)
    (h5 : cards_per_pack = 50)
    (h6 : total_packs = total_spent / cost_per_pack)
    (h7 : total_index_cards = total_packs * cards_per_pack)
    (h8 : total_students = periods_per_day * students_per_class)
    (h9 : index_cards_per_student = total_index_cards / total_students) :
    index_cards_per_student = 10 := 
  by
    sorry

end index_cards_per_student_l341_341294


namespace sum_of_integers_satisfying_equation_l341_341557

theorem sum_of_integers_satisfying_equation : (∑ x in { x : ℤ | x^2 = 256 + x }, x) = 0 := 
sorry

end sum_of_integers_satisfying_equation_l341_341557


namespace binom_9_5_l341_341296

theorem binom_9_5 : nat.binomial 9 5 = 126 := by
  sorry

end binom_9_5_l341_341296


namespace y_intercept_of_line_l341_341693

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by sorry

end y_intercept_of_line_l341_341693


namespace monotonic_intervals_range_of_a_l341_341714

noncomputable def f : ℝ → ℝ := λ x, x^3 - (1/2) * x^2 - 2 * x + 5

theorem monotonic_intervals :
  ( ∀ x : ℝ, x < -2/3 → f' x > 0 ) ∧
  ( ∀ x : ℝ, -2/3 < x → x < 1 → f' x < 0 ) ∧
  ( ∀ x : ℝ, x > 1 → f' x > 0 ) :=
sorry

theorem range_of_a :
  ∀ a : ℝ, (5 < a ∧ a < 5 + 1/216) ↔
  ( ∃ x0 : ℝ, (0, a) ∈ lineThrough (x0, f x0) ) := 
sorry

end monotonic_intervals_range_of_a_l341_341714


namespace angle_CED_90_l341_341436

-- Data for the problem
variables {A B C M O D E : Type}
variables (triangle : Triangle A B C)
variables (M_midpoint : Midpoint M A B)
variables (O_circumcenter : Circumcenter O triangle)
variables (R r : ℝ)
variables (OM_eq : distance O M = R - r)
variables (D_bisector_A : ExternalAngleBisector D A B C)
variables (E_bisector_C : ExternalAngleBisector E C B A)

-- Main theorem statement
theorem angle_CED_90 (h1 : Midpoint M A B) (h2 : Circumcenter O triangle) 
(h3 : distance O M = R - r) (h4 : ExternalAngleBisector D A B C) 
(h5 : ExternalAngleBisector E C B A) : 
  angle C E D = 90 := 
  by
  sorry

end angle_CED_90_l341_341436


namespace right_triangle_hypotenuse_median_and_BC_l341_341721

/-- Definitions for the right triangle vertices -/
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (1, sqrt 3)
def C : ℝ × ℝ := (4, 0)

/-- Equation of the line containing side BC -/
def equation_of_BC : Prop :=
  ∃ (x y : ℝ), y - sqrt 3 = -4 * sqrt 3 * (x - 1)

/-- A simplified version of the equation for side BC after rearrangement -/
def simplified_equation_of_BC : Prop :=
  ∃ (x y : ℝ), sqrt 3 * x + 4 * y = 4 * sqrt 3

/-- Equation of the line containing the median to the hypotenuse of triangle ABC from the origin to the midpoint of AC -/
def median_to_hypotenuse : Prop :=
  ∀ (x y : ℝ), y = sqrt 3 * x

/-- Main theorem containing both proof statements -/
theorem right_triangle_hypotenuse_median_and_BC :
  equation_of_BC ↔ simplified_equation_of_BC ∧ median_to_hypotenuse :=
begin
  sorry
end

end right_triangle_hypotenuse_median_and_BC_l341_341721


namespace gcd_204_85_l341_341186

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l341_341186


namespace consecutive_odd_integers_sum_l341_341204

theorem consecutive_odd_integers_sum (a b c : ℤ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) (h3 : c % 2 = 1) (h4 : a < b) (h5 : b < c) (h6 : c = -47) : a + b + c = -141 := 
sorry

end consecutive_odd_integers_sum_l341_341204


namespace symmetrical_line_intersects_circle_l341_341488

variables (A_x A_y B_x B_y a x y : ℝ)

def points_A_B (A_x A_y B_x B_y : ℝ) : Prop :=
  A_x = -2 ∧ A_y = 3 ∧ B_x = 0

def circle (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y + 2) ^ 2 = 1

theorem symmetrical_line_intersects_circle (a : ℝ) :
  (∀ x y, points_A_B (-2) 3 0 a → (circle x y → ∃ (x y : ℝ), True)) →
  (a ∈ set.Icc (1 / 3 : ℝ) (3 / 2 : ℝ)) :=
by
  intros h
  sorry

end symmetrical_line_intersects_circle_l341_341488


namespace red_crayons_count_l341_341207

variable (R : ℕ) -- Number of red crayons
variable (B : ℕ) -- Number of blue crayons
variable (Y : ℕ) -- Number of yellow crayons

-- Conditions
axiom h1 : B = R + 5
axiom h2 : Y = 2 * B - 6
axiom h3 : Y = 32

-- Statement to prove
theorem red_crayons_count : R = 14 :=
by
  sorry

end red_crayons_count_l341_341207


namespace infinite_series_floor_sum_l341_341605

-- Define the intermediate result
lemma floor_two_x_eq_floor_x_add_floor_x_half (x : ℝ) : 
  ⌊2 * x⌋ = ⌊x⌋ + ⌊x + (1 / 2)⌋ := sorry

-- Define the main theorem
theorem infinite_series_floor_sum (x : ℝ) (h : ∀ k : ℕ, 2^k > x → ⌊(x + 2^k) / 2^(k+1)⌋ = 0) :
  ∑' k, ⌊ (x + 2^k) / 2^(k+1) ⌋ = ⌊x⌋ :=
begin
  -- Use the intermediate lemma
  have intermediate := floor_two_x_eq_floor_x_add_floor_x_half x,
  sorry -- the rest of the proof
end

end infinite_series_floor_sum_l341_341605


namespace real_part_proof_l341_341122

noncomputable def real_part_of_fraction (z : ℂ) (h : |z| = 2 ∧ z.im ≠ 0) : ℝ :=
  let x := z.re in
  let y := z.im in
  (2 - x) / (8 - 4 * x + x^2)

theorem real_part_proof (z : ℂ) (h : |z| = 2 ∧ z.im ≠ 0) :
  (real_part_of_fraction z h) = (2 - z.re) / (8 - 4 * z.re + z.re^2) :=
by
  sorry

end real_part_proof_l341_341122


namespace isosceles_triangle_PQT_l341_341113

noncomputable def cyclic_points (A B C D E : Point) : Prop :=
  collinear A C D ∧ collinear A C B ∧ collinear C D E ∧ collinear B D E

noncomputable def is_intersection (P : Point) (l1 l2 : Line) : Prop :=
  P ∈ l1 ∧ P ∈ l2

theorem isosceles_triangle_PQT {A B C D E P Q T : Point}
  (hCircle : cyclic_points A B C D E)
  (hAB_BC : dist A B = dist B C)
  (hCD_DE : dist C D = dist D E)
  (hP_intersect : is_intersection P (line A D) (line B E))
  (hQ_intersect : is_intersection Q (line A C) (line B D))
  (hT_intersect : is_intersection T (line B D) (line C E))
  : dist P Q = dist P T ∨ dist Q P = dist Q T ∨ dist T P = dist T Q := sorry

end isosceles_triangle_PQT_l341_341113


namespace sum_of_two_lowest_scores_l341_341706

theorem sum_of_two_lowest_scores
  (scores : Fin 5 → ℝ) -- there are five test scores
  (mean : ∀ s, (∑ i, scores i) / 5 = 92) -- mean of the five scores is 92
  (median : ∀ s, ∃ (a b : ℝ), a ≤ b ∧ b = 95 ∧ a ≤ scores 2 ∧ scores 2 ≤ b) -- median of the five scores is 95
  (mode : ∀ s, ∀ x : ℝ, x = 94 → (∃ i j, i ≠ j ∧ scores i = x ∧ scores j = x)) -- mode of the five scores is 94
  : (∑ i in Finset.filter (fun i => scores i < 95) Finset.univ, scores i) = 177 :=
by
  sorry

end sum_of_two_lowest_scores_l341_341706


namespace part1_part2_l341_341738

variables {a b c : ℝ}
variables {A B C: ℝ}
variables {m n : ℝ × ℝ}

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Given conditions
axiom h1 : m = (3 * a, 3)
axiom h2 : n = (-2 * Real.sin B, b)
axiom h3 : vector_dot m n = 0
axiom acute_angles : 0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2
axiom triangle_angles : A + B + C = Real.pi

-- Prove the value of A
theorem part1 : A = Real.pi / 6 :=
by 
  -- Placeholder for the proof
  sorry

-- Given that a = 2 and the perimeter of the triangle is 6
axiom a_eq_2 : a = 2
axiom perimeter : a + b + c = 6

-- Prove the area of the triangle
theorem part2 : 
  let s := (a + b + c) / 2 in
  √(s * (s - a) * (s - b) * (s - c)) = 6 - 3 * Real.sqrt 3:=
by 
  -- Placeholder for the proof
  sorry

end part1_part2_l341_341738


namespace slope_of_line_through_origin_l341_341079

open Real

def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def perpendicular_slope (m : ℝ) : ℝ := -1 / m

theorem slope_of_line_through_origin (P Q : ℝ × ℝ) (hP : P = (4, 6)) (hQ : Q = (6, 2)) :
  slope (0, 0) (5, 4) = 1 / 2 :=
by
  sorry

end slope_of_line_through_origin_l341_341079


namespace incorrect_statement_B_l341_341970

-- Define the conditions
def statement_A (a : ℝ) : Prop :=
  a^2 = (-a)^2

def statement_B (a : ℝ) : Prop :=
  |a| = -|a|

def statement_C (a : ℝ) : Prop :=
  (|3a| / |a|) = -(|3(-a)| / |-a|)

def statement_D (a : ℝ) : Prop :=
  |a| = -|a|

-- Theorem stating that statement B is incorrect
theorem incorrect_statement_B (a : ℝ) : 
  statement_A a → ¬ statement_B a → statement_C a → statement_D a → ¬ statement_B a :=
by
  -- proof omitted
  sorry

end incorrect_statement_B_l341_341970


namespace no_three_nat_sum_pair_is_pow_of_three_l341_341288

theorem no_three_nat_sum_pair_is_pow_of_three :
  ¬ ∃ (a b c : ℕ) (m n p : ℕ), a + b = 3 ^ m ∧ b + c = 3 ^ n ∧ c + a = 3 ^ p := 
by 
  sorry

end no_three_nat_sum_pair_is_pow_of_three_l341_341288


namespace speed_of_second_train_l341_341215

theorem speed_of_second_train 
  (time_to_pass : ℝ) (length_first_train : ℝ) (length_second_train : ℝ)
  (speed_first_train : ℝ) :
  time_to_pass = 90 →
  length_first_train = 125 →
  length_second_train = 125.02 →
  speed_first_train = 50 →
  ∃ V2 : ℝ, V2 = 60.0008 :=
by intro h1 h2 h3 h4
   let relative_speed := (V2 - speed_first_train) * (1 / 3.6)
   have hs1 : relative_speed = 250.02 / 90,
   calc
     relative_speed : (V2 - speed_first_train) * (1 / 3.6) = (250.02 / 90) : sorry
   let V2 := (250.02 / 90) * 3.6 + speed_first_train
   have hs2 : V2 = 60.0008,
   calc
     V2 : (250.02 / 90) * 3.6 + speed_first_train = 60.0008 : sorry
   exact ⟨V2, hs2⟩

end speed_of_second_train_l341_341215


namespace three_integers_desc_order_l341_341237

theorem three_integers_desc_order (a b c : ℤ) : ∃ a' b' c' : ℤ, 
  (a = a' ∨ a = b' ∨ a = c') ∧
  (b = a' ∨ b = b' ∨ b = c') ∧
  (c = a' ∨ c = b' ∨ c = c') ∧ 
  (a' ≠ b' ∨ a' ≠ c' ∨ b' ≠ c') ∧
  a' ≥ b' ∧ b' ≥ c' :=
sorry

end three_integers_desc_order_l341_341237


namespace max_expression_value_l341_341468

noncomputable def complex_abs (z : ℂ) : ℝ := complex.abs z

noncomputable def expression (z : ℂ) : ℝ := complex_abs ((z - 2) ^ 2 * (z + 2))

theorem max_expression_value (z : ℂ) (hz : complex_abs z = 2) : 
  ∃ m : ℝ, m = 64 ∧ ∀ w : ℂ, complex_abs w = 2 → expression w ≤ m :=
by {
  sorry
}

end max_expression_value_l341_341468


namespace find_b_l341_341378

variable {x1 x2 b : ℝ}

-- Conditions
def sum_roots := x1 + x2 = -b
def product_roots := x1 * x2 = 4
def condition := x1 - x1 * x2 + x2 = 2

theorem find_b (h_sum_roots : sum_roots) (h_product_roots : product_roots) (h_condition : condition) : b = -6 := 
by 
  sorry

end find_b_l341_341378


namespace real_part_of_inverse_is_half_l341_341127

noncomputable def real_part_of_inverse_expression (z : ℂ) (h : ∥z∥ = 2) (hnonreal : ¬ real z) : ℝ :=
  real_part (1 / (2 - z))

theorem real_part_of_inverse_is_half (z : ℂ) (h : ∥z∥ = 2) (hnonreal : ¬ real z) : 
  real_part_of_inverse_expression z h hnonreal = 1 / 2 :=
by
  sorry

end real_part_of_inverse_is_half_l341_341127


namespace distinct_arrangements_of_digits_l341_341161

theorem distinct_arrangements_of_digits (grid : fin 2 × fin 2 → fin 4) (digits : fin 4 → ℕ) :
  (digits 0 = 1 ∧ digits 1 = 1 ∧ digits 2 = 2 ∧ digits 3 = 3) →
  (∃ i j : fin 2 × fin 2, grid i = 3 ∧ grid j = 1 ∧ adjacent i j) →
  12 = ∑ f : (fin 2 × fin 2 ↦ fin 4), (∃ i j : fin 2 × fin 2, grid i = 3 ∧ grid j = 1 ∧ adjacent i j) :=
sorry

-- Define adjacency in the grid
def adjacent (i j : fin 2 × fin 2) : Prop :=
  (i.1 = j.1 ∧ (i.2 = j.2 + 1 ∨ i.2 + 1 = j.2)) ∨
  (i.2 = j.2 ∧ (i.1 = j.1 + 1 ∨ i.1 + 1 = j.1))

noncomputable def grid_example : (fin 2 × fin 2 → fin 4) :=
  sorry  -- Example grid function

#check grid_example

end distinct_arrangements_of_digits_l341_341161


namespace find_a_value_l341_341704

noncomputable def f (x a : ℝ) : ℝ := (x^2 + a) / (x + 1)

def slope_of_tangent_line (a : ℝ) : Prop :=
  (deriv (fun x => f x a) 1) = -1

theorem find_a_value : ∃ a : ℝ, slope_of_tangent_line a ∧ a = 7 := by
  sorry

end find_a_value_l341_341704


namespace gcd_204_85_l341_341184

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := 
by sorry

end gcd_204_85_l341_341184


namespace find_x_minus_y_l341_341359

-- Variables and conditions
variables (x y : ℝ)
def abs_x_eq_3 := abs x = 3
def y_sq_eq_one_fourth := y^2 = 1 / 4
def x_plus_y_neg := x + y < 0

-- Proof problem stating that x - y must equal one of the two possible values
theorem find_x_minus_y (h1 : abs x = 3) (h2 : y^2 = 1 / 4) (h3 : x + y < 0) : 
  x - y = -7 / 2 ∨ x - y = -5 / 2 :=
  sorry

end find_x_minus_y_l341_341359


namespace ellipse_with_conditions_l341_341026

open Real

variables {a b x y : ℝ}
def ellipse (a b : ℝ) (C : Set (ℝ × ℝ)) : Prop := 
  ∀ (p : ℝ × ℝ), p ∈ C ↔ p.1^2 / a^2 + p.2^2 / b^2 = 1

variables {F B1 B2 : ℝ × ℝ}

theorem ellipse_with_conditions :
  ∀ (a b : ℝ), a > b → b > 0 →
  ellipse a b
    {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1} →
  (F = (1,0)) →
  (B1 = (0, -b)) →
  (B2 = (0, b)) →
  ((F.1 - B1.1) * (F.1 - B2.1) + (F.2 - B1.2) * (F.2 - B2.2)) = -a →
  (C.Point x y : {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}) →
  (0 < |DP x y| / |MN x y| ∧ |DP x y| / |MN x y| < 1/4) :=
sorry

end ellipse_with_conditions_l341_341026


namespace mary_baseball_cards_l341_341498

theorem mary_baseball_cards :
  let initial_cards := 18
  let torn_cards := 8
  let fred_gifted_cards := 26
  let bought_cards := 40
  let exchanged_cards := 10
  let lost_cards := 5
  
  let remaining_cards := initial_cards - torn_cards
  let after_gift := remaining_cards + fred_gifted_cards
  let after_buy := after_gift + bought_cards
  let after_exchange := after_buy - exchanged_cards + exchanged_cards
  let final_count := after_exchange - lost_cards
  
  final_count = 71 :=
by
  sorry

end mary_baseball_cards_l341_341498


namespace polynomial_coeff_sum_l341_341096

theorem polynomial_coeff_sum (A B C D : ℤ) 
  (h : ∀ x : ℤ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 :=
by 
  sorry

end polynomial_coeff_sum_l341_341096


namespace problem_i_solution_problem_ii_part_i_solution_problem_ii_part_ii_solution_l341_341621

section problem_i

variables {b : ℕ → ℝ}
-- Conditions for Problem (i)
def symmetric_sequence (n : ℕ) (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, i < n / 2 → a i = a (n - i - 1)

def problem_i_condition_1 := symmetric_sequence 7 b

def problem_i_condition_2 := b 1 = 3

def problem_i_condition_3 := b 4 = 1

def problem_i_answer := b 0 = 9 ∧ b 1 = 3 ∧ b 2 = 1 ∧ b 3 = 1/3 ∧ b 4 = 1 ∧ b 5 = 3 ∧ b 6 = 9

theorem problem_i_solution : problem_i_condition_1 ∧ problem_i_condition_2 ∧ problem_i_condition_3 → problem_i_answer :=
sorry

end problem_i

section problem_ii_part_i

variables {c : ℕ → ℝ} {k : ℕ}

def symmetric_sequence_even (n : ℕ) (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, i < n / 2 → a i = a (n - i - 1)

def problem_ii_condition_1 := symmetric_sequence_even (2 * k - 1) c

def problem_ii_condition_2 := ∀ n : ℕ, n < k - 1 → |c (n + 1) - c n| = 2

def problem_ii_condition_3 := ∃ k : ℕ, c k = 2017 ∧ (∀ n : ℕ, n < k → c n < c (n + 1))

def sum_of_sequence_2k1 (c : ℕ → ℝ) (k : ℕ) : ℝ := 
  2 * (k * (c 1 + c k) / 2) - c k

def problem_ii_part_i_answer (k : ℕ) := maximizing_sum := sum_of_sequence_2k1 c k = 2036162

theorem problem_ii_part_i_solution : problem_ii_condition_1 ∧ problem_ii_condition_2 ∧ problem_ii_condition_3 → problem_ii_part_i_answer 1009 :=
sorry

end problem_ii_part_i

section problem_ii_part_ii

variables {c : ℕ → ℝ} {k : ℕ}

def problem_ii_condition_4 := c 1 = 2018

def sum_of_sequence_2k1_equal (c : ℕ → ℝ) (k : ℕ) : ℝ :=
  2 * (k * (c 1 + c k) / 2) - c k

def problem_ii_part_ii_answer (k : ℕ) := sum_of_sequence_2k1_equal c = 2018

theorem problem_ii_part_ii_solution : ∃ k : ℕ, problem_ii_condition_1 ∧ problem_ii_condition_2 ∧ problem_ii_condition_4 ∧ problem_ii_part_ii_answer k → k = 2019 :=
sorry

end problem_ii_part_ii

end problem_i_solution_problem_ii_part_i_solution_problem_ii_part_ii_solution_l341_341621


namespace value_of_expression_l341_341776

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end value_of_expression_l341_341776


namespace proposition_D_l341_341466

variables (m n : Line) (α β : Plane)

-- Conditions
axiom m_diff_n : m ≠ n
axiom α_diff_β : α ≠ β

-- Given conditions for Proposition D
axiom m_perp_β : m ⊥ β
axiom n_perp_β : n ⊥ β
axiom n_perp_α : n ⊥ α

-- Prove
theorem proposition_D : m ⊥ α :=
by
  sorry

end proposition_D_l341_341466


namespace complex_problem_l341_341426

-- Define the complex number z
def z : ℂ := (4 + 3 * complex.I) / (2 - complex.I)

-- Define the imaginary part condition
def imaginary_part (z : ℂ) : ℂ := z.im

-- Define the conjugate and product condition
def conjugate_times (z : ℂ) (w : ℂ) : ℂ := (conjugate z) * w

-- Define the modulus condition
def modulus (z : ℂ) : ℝ := complex.abs z

theorem complex_problem :
  (imaginary_part z = 2) ∧
  (modulus (conjugate_times z (2 - complex.I)) = 5) :=
by
  sorry

end complex_problem_l341_341426


namespace cost_of_whistle_l341_341873

theorem cost_of_whistle (cost_yoyo : ℕ) (total_spent : ℕ) (cost_yoyo_equals : cost_yoyo = 24) (total_spent_equals : total_spent = 38) : (total_spent - cost_yoyo) = 14 :=
by
  sorry

end cost_of_whistle_l341_341873


namespace unique_real_root_count_l341_341700

theorem unique_real_root_count :
  ∃! x : ℝ, (x^12 + 1) * (x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 12 * x^11 := by
  sorry

end unique_real_root_count_l341_341700


namespace symmetrical_line_intersects_circle_l341_341489

variables (A_x A_y B_x B_y a x y : ℝ)

def points_A_B (A_x A_y B_x B_y : ℝ) : Prop :=
  A_x = -2 ∧ A_y = 3 ∧ B_x = 0

def circle (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y + 2) ^ 2 = 1

theorem symmetrical_line_intersects_circle (a : ℝ) :
  (∀ x y, points_A_B (-2) 3 0 a → (circle x y → ∃ (x y : ℝ), True)) →
  (a ∈ set.Icc (1 / 3 : ℝ) (3 / 2 : ℝ)) :=
by
  intros h
  sorry

end symmetrical_line_intersects_circle_l341_341489


namespace multiple_of_righty_points_l341_341827

theorem multiple_of_righty_points :
  ∀ (X : ℕ), (Lefty Righty Other : ℕ → Prop) 
  (Lefty 20) 
  (Righty = fun n => 20 / 2) 
  (Other = fun n => 10 * X)
  (average_score = 30), 
  (TeamPoints = Lefty + Righty + Other) / 3 = average_score → X = 6 := 
by 
  intro X Lefty Righty Other average_score TeamPoints
  sorry

end multiple_of_righty_points_l341_341827


namespace ship_direction_reciprocal_l341_341049

theorem ship_direction_reciprocal (A B : Type) 
  (direction_from_B_to_A : B → A → ℝ) 
  (h1 : direction_from_B_to_A B A = 35) : 
  direction_from_B_to_A A B = 215 := 
sorry

end ship_direction_reciprocal_l341_341049


namespace inequality_for_integers_l341_341152

theorem inequality_for_integers (n : ℕ) (hn : n > 1) : 
  2 * n < ∑ k in Finset.range n + 1, 4 ^ (k / n) ∧ ∑ k in Finset.range n + 1, 4 ^ (k / n) ≤ 3 * n :=
by
  sorry

end inequality_for_integers_l341_341152


namespace domain_of_p_l341_341162

theorem domain_of_p (h : ℝ → ℝ)
  (h_domain : ∀ x, -12 ≤ x ∧ x ≤ 6 → true) :
  set_of (λ x, -(-12 : ℝ) + 1 ≤ -3 * x + 1 ∧ -3 * x + 1 ≤ 6) = {x : ℝ | -5 / 3 ≤ x ∧ x ≤ 13 / 3} :=
by
  sorry

end domain_of_p_l341_341162


namespace petya_no_win_implies_draw_or_lost_l341_341201

noncomputable def petya_cannot_win (n : ℕ) (h : n ≥ 3) : Prop :=
  ∀ (Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ),
    ∃ m : ℕ, Petya_strategy m ≠ Vasya_strategy m

theorem petya_no_win_implies_draw_or_lost (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ, 
    (∀ m : ℕ, Petya_strategy m = Vasya_strategy m) :=
by {
  sorry
}

end petya_no_win_implies_draw_or_lost_l341_341201


namespace f_2017_eq_2018_l341_341977

def f (n : ℕ) : ℕ := sorry

theorem f_2017_eq_2018 (f : ℕ → ℕ) (h1 : ∀ n, f (f n) + f n = 2 * n + 3) (h2 : f 0 = 1) : f 2017 = 2018 :=
sorry

end f_2017_eq_2018_l341_341977


namespace sum_of_g_9_values_l341_341850

noncomputable def f (x : ℝ) : ℝ := x^2 - 5*x + 8
noncomputable def g (x : ℝ) : ℝ := 3*x + 2

theorem sum_of_g_9_values : 
  let x1 := (5 + Real.sqrt 29) / 2 in
  let x2 := (5 - Real.sqrt 29) / 2 in
  f x1 = 9 ∧ f x2 = 9 →
  g x1 + g x2 = 19 :=
by
  intro h,
  have hx1 := h.1,
  have hx2 := h.2,
  sorry

end sum_of_g_9_values_l341_341850


namespace adi_baller_prob_l341_341278

theorem adi_baller_prob (a b : ℕ) (p : ℝ) (h_prime: Nat.Prime a) (h_pos_b: 0 < b)
  (h_p: p = (1 / 2) ^ (1 / 35)) : a + b = 37 :=
sorry

end adi_baller_prob_l341_341278


namespace meeting_lamppost_l341_341981

-- Define the initial conditions of the problem
def lampposts : ℕ := 400
def start_alla : ℕ := 1
def start_boris : ℕ := 400
def meet_alla : ℕ := 55
def meet_boris : ℕ := 321

-- Define a theorem that we need to prove: Alla and Boris will meet at the 163rd lamppost
theorem meeting_lamppost : ∃ (n : ℕ), n = 163 := 
by {
  sorry -- Proof goes here
}

end meeting_lamppost_l341_341981


namespace cos_alpha_third_quadrant_l341_341373

theorem cos_alpha_third_quadrant (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h : tan α = 5 / 12) : cos α = -12 / 13 :=
sorry

end cos_alpha_third_quadrant_l341_341373


namespace beach_ball_problem_l341_341271

noncomputable def change_in_radius (C₁ C₂ : ℝ) : ℝ := (C₂ - C₁) / (2 * Real.pi)

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

noncomputable def percentage_increase_in_volume (V₁ V₂ : ℝ) : ℝ := (V₂ - V₁) / V₁ * 100

theorem beach_ball_problem (C₁ C₂ : ℝ) (hC₁ : C₁ = 30) (hC₂ : C₂ = 36) :
  change_in_radius C₁ C₂ = 3 / Real.pi ∧
  percentage_increase_in_volume (volume (C₁ / (2 * Real.pi))) (volume (C₂ / (2 * Real.pi))) = 72.78 :=
by
  sorry

end beach_ball_problem_l341_341271


namespace min_convex_polygons_of_union_l341_341265

theorem min_convex_polygons_of_union (n : ℕ) (hn : 2 ≤ n) :
  let M := regular_ngon n 1
      M' := rotate_ngon M (π / n)
  in min_convex_polygons (M ∪ M') = n + 1 :=
sorry

end min_convex_polygons_of_union_l341_341265


namespace cylindrical_to_rectangular_point_l341_341175

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_point :
  cylindrical_to_rectangular (Real.sqrt 2) (Real.pi / 4) 1 = (1, 1, 1) :=
by
  sorry

end cylindrical_to_rectangular_point_l341_341175


namespace find_y_l341_341795

theorem find_y (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 :=
sorry

end find_y_l341_341795


namespace last_tuesday_of_april_l341_341904

def is_tuesday (date : ℕ) : Prop :=
  -- Assuming date is in the format of April (1 to 30) and considering April 1 is a Tuesday
  (date % 7 = 3)

def cafe_open_days (start_date end_date : ℕ) : ℕ :=
  -- Counting total open days, assuming 6 open days a week with Monday off
  let weeks := (end_date - start_date + 1) / 7
  let remaining_days := (end_date - start_date + 1) % 7
  weeks * 6 + remaining_days - if (start_date % 7 = 0) then 1 else 0

theorem last_tuesday_of_april
  (H1 : is_tuesday 1)
  (H2 : cafe_open_days 1 20 = 17)
  (H3 : cafe_open_days 10 30 = 18)
  (H4 : ∃! d ∈ { H2, H3 }, d = false)
  : ∃ date, date = 29 ∧ is_tuesday date := 
sorry

end last_tuesday_of_april_l341_341904


namespace eddie_games_l341_341323

-- Define the study block duration in minutes
def study_block_duration : ℕ := 60

-- Define the homework time in minutes
def homework_time : ℕ := 25

-- Define the time for one game in minutes
def game_time : ℕ := 5

-- Define the total time Eddie can spend playing games
noncomputable def time_for_games : ℕ := study_block_duration - homework_time

-- Define the number of games Eddie can play
noncomputable def number_of_games : ℕ := time_for_games / game_time

-- Theorem stating the number of games Eddie can play while completing his homework
theorem eddie_games : number_of_games = 7 := by
  sorry

end eddie_games_l341_341323


namespace unique_games_count_l341_341295

noncomputable def total_games_played (n : ℕ) (m : ℕ) : ℕ :=
  (n * m) / 2

theorem unique_games_count (students : ℕ) (games_per_student : ℕ) (h1 : students = 9) (h2 : games_per_student = 6) :
  total_games_played students games_per_student = 27 :=
by
  rw [h1, h2]
  -- This partially evaluates total_games_played using the values from h1 and h2.
  -- Performing actual proof steps is not necessary, so we'll use sorry.
  sorry

end unique_games_count_l341_341295


namespace price_of_tuna_pack_l341_341655

-- Defining the conditions as given in the problem
def packs_sold_peak_hour : ℕ := 6
def packs_sold_low_hour : ℕ := 4
def hours_sold : ℕ := 15
def extra_money_made_high_season : ℕ := 1800

-- Statement to prove the price of each tuna pack
theorem price_of_tuna_pack :
  let packs_sold_peak_day := packs_sold_peak_hour * hours_sold,
      packs_sold_low_day := packs_sold_low_hour * hours_sold,
      extra_packs_sold := packs_sold_peak_day - packs_sold_low_day in
  (extra_money_made_high_season / extra_packs_sold) = 60 :=
by
  -- The proof is omitted; inserting a placeholder
  sorry

end price_of_tuna_pack_l341_341655


namespace geom_seq_sum_5_l341_341198

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

theorem geom_seq_sum_5 (a_n : ℕ → ℝ) (q a₃ a₅ a₂ a₆ : ℝ) (n : ℕ) :
  a_n > 0 →
  q > 1 →
  a_n 2 * a_n 6 = 64 →
  a_n 3 + a_n 5 = 20 →
  let S₅ := geometric_sum (a₂ / q) q 5 in
  S₅ = 31 :=
by
  sorry

end geom_seq_sum_5_l341_341198


namespace excircles_touch_midline_properties_l341_341533

noncomputable def midpoint (P Q : Point) : Point := 
  sorry

theorem excircles_touch_midline_properties 
(A B C K L : Point)
(h1 : ExcircleTouches AC B K)
(h2 : ExcircleTouches BC A L) :
∃ M N : Point,
  (is_midpoint M K L) ∧ (is_midpoint N A B) ∧
  (divides_perimeter_half (segment M N) (triangle A B C)) ∧
  (parallel (segment M N) (angle_bisector ∠ACB)) :=
by
  sorry

end excircles_touch_midline_properties_l341_341533


namespace total_initial_collection_l341_341871

variable (marco strawberries father strawberries_lost : ℕ)
variable (marco : ℕ := 12)
variable (father : ℕ := 16)
variable (strawberries_lost : ℕ := 8)
variable (total_initial_weight : ℕ := marco + father + strawberries_lost)

theorem total_initial_collection : total_initial_weight = 36 :=
by
  sorry

end total_initial_collection_l341_341871


namespace sum_of_first_seven_terms_l341_341009

variable {α : Type} [LinearOrderedField α]

def arithmeticSeq (a_4 : α) := (λ n : ℕ, a_4 + (n - 4) * d)

def sumFirstNTerms (a_1 : α) (a_n : α) (n : ℕ) : α :=
  n * (a_1 + a_n) / 2

theorem sum_of_first_seven_terms (a_4 : α) (S_7 : α) :
  a_4 = 4 → S_7 = (7 / 2) * (a_4 + (a_4 + 3 * d)) →
  S_7 = 28 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end sum_of_first_seven_terms_l341_341009


namespace no_solution_fermat_like_l341_341987

theorem no_solution_fermat_like (x y z k : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) 
  (hxk : x < k) (hyk : y < k) (hxk_eq : x ^ k + y ^ k = z ^ k) : false :=
sorry

end no_solution_fermat_like_l341_341987


namespace find_min_f_l341_341710

open Real

def f (x : ℝ) : ℝ := (x^2 - 6*x - 3) / (x + 1)

def domain := Icc 0 1

theorem find_min_f : ∃ m, (∀ x ∈ domain, f x ≥ m) ∧ (∃ x ∈ domain, f x = -4) :=
by
  sorry

end find_min_f_l341_341710


namespace a₀_value_sum_even_coeffs_l341_341418

open Polynomial

noncomputable def poly := (X - 3)^3 * (2 * X + 1)^5
noncomputable def expansion := C a₀ + C a₁ * X + C a₂ * X^2 + C a₃ * X^3 + C a₄ * X^4 + C a₅ * X^5 + C a₆ * X^6 + C a₇ * X^7 + C a₈ * X^8

-- Prove that the constant term is -27
theorem a₀_value : (poly.eval 0) = (-27 : ℤ) := sorry

-- Prove that the sum a₀ + a₂ + ... + a₈ is -940
theorem sum_even_coeffs : 
  a₀ + a₂ + a₄ + a₆ + a₈ = (-940 : ℤ) := sorry

end a₀_value_sum_even_coeffs_l341_341418


namespace modulo_residue_l341_341226

theorem modulo_residue :
  (247 + 5 * 39 + 7 * 143 + 4 * 15) % 13 = 8 := by
  have h1 : 247 % 13 = 0 := by sorry
  have h2 : (5 * 39) % 13 = 0 := by sorry
  have h3 : (7 * 143) % 13 = 0 := by sorry
  have h4 : (4 * 15) % 13 = 8 := by sorry
  calc
    (247 + 5 * 39 + 7 * 143 + 4 * 15) % 13
      = (0 + 0 + 0 + 8) % 13 := by
        rw [h1, h2, h3, h4]
      _ = 8 := by simp

end modulo_residue_l341_341226


namespace find_value_of_c_l341_341628

noncomputable def parabola (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem find_value_of_c (b c : ℝ) 
    (h1 : parabola b c 1 = 2)
    (h2 : parabola b c 5 = 2) :
    c = 7 :=
by
  sorry

end find_value_of_c_l341_341628


namespace sum_of_seven_vectors_is_zero_l341_341205

theorem sum_of_seven_vectors_is_zero
  (a : ℕ → ℝ^3)
  (h : ∀ i j k l m n o : ℕ, j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ n ∧ n ≠ o ∧ o ≠ j →
    ||a j + a k + a l|| = ||a m + a n + a o|| ∧
    j < 7 ∧ k < 7 ∧ l < 7 ∧ m < 7 ∧ n < 7 ∧ o < 7) :
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 0 :=
by
  sorry

end sum_of_seven_vectors_is_zero_l341_341205


namespace convex_func_equiv_l341_341520

noncomputable def convex_func_property (f : ℝ → ℝ) : Prop :=
∀ (k : ℕ) (u v : Fin k → ℝ),
  (∀ i j, i < j → u i ≥ u j) ∧ 
  (∀ i j, i < j → v i ≥ v j) ∧
  (∀ i, i < k → u i < v i) ∧ 
  (∀ j, 1 ≤ j → (∑ i in Finset.range j, u i) < (∑ i in Finset.range j, v i)) ∧
  (∑ i in Finset.range k, u i = ∑ i in Finset.range k, v i) →
  (∑ i in Finset.range k, f (u i) < ∑ i in Finset.range k, f (v i))

def is_convex (f : ℝ → ℝ) : Prop :=
∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 → 
  f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

theorem convex_func_equiv (f : ℝ → ℝ) : 
  convex_func_property f ↔ is_convex f :=
sorry

end convex_func_equiv_l341_341520


namespace infinity_solutions_l341_341511

theorem infinity_solutions (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) :
  ∃ (a b : ℕ) (infinitely_many : ℕ → Prop), infinitely_many (λ k, a = fib (2*k+2) ∧ b = fib (2*k+1)) ∧ a^2 - b^2 = a * b - 1 := sorry

end infinity_solutions_l341_341511


namespace part1_part2_l341_341757

noncomputable def f (a x : ℝ) : ℝ := (a * Real.exp x - a - x) * Real.exp x

theorem part1 (a : ℝ) (h0 : a ≥ 0) (h1 : ∀ x : ℝ, f a x ≥ 0) : a = 1 := 
sorry

theorem part2 (h1 : ∀ x : ℝ, f 1 x ≥ 0) :
  ∃! x0 : ℝ, (∀ x : ℝ, x0 = x → 
  (f 1 x0) = (f 1 x)) ∧ (0 < f 1 x0 ∧ f 1 x0 < 1/4) :=
sorry

end part1_part2_l341_341757


namespace rational_coefficients_terms_count_l341_341589

theorem rational_coefficients_terms_count : 
  (∃ s : Finset ℕ, ∀ k ∈ s, k % 20 = 0 ∧ k ≤ 725 ∧ s.card = 37) :=
by
  -- Translates to finding the set of all k satisfying the condition and 
  -- ensuring it has a cardinality of 37.
  sorry

end rational_coefficients_terms_count_l341_341589


namespace max_diff_in_grid_l341_341698

theorem max_diff_in_grid (n : ℕ) (h : n ≤ 209) : 
  ∀ grid : array (fin 20) (array (fin 20) ℕ), 
  (∀ i j, grid[i][j] ∈ finset.range 1 401) →
  ∃ row col, ((∃ i₁ i₂, (i₁ < i₂) ∧ |grid[i₁][col] - grid[i₂][col]| ≥ n) ∨ 
              (∃ j₁ j₂, (j₁ < j₂) ∧ |grid[row][j₁] - grid[row][j₂]| ≥ n)) :=
sorry

end max_diff_in_grid_l341_341698


namespace binom_9_5_l341_341300

open Nat

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Theorem to prove binom(9, 5) = 126
theorem binom_9_5 : binom 9 5 = 126 := by
  sorry

end binom_9_5_l341_341300


namespace find_fourth_number_l341_341608

variable (x : ℝ)

theorem find_fourth_number
  (h : 3 + 33 + 333 + x = 399.6) :
  x = 30.6 :=
sorry

end find_fourth_number_l341_341608


namespace upstream_time_proof_l341_341148

-- Define the speeds and times
variables (x y t_ACd t_CAu t_ACnew : ℝ)

-- Initial conditions
def initial_conditions : Prop :=
  t_ACd = 6 ∧ t_CAu = 7 ∧ t_ACnew = 5.5

-- Goal: Prove the upstream travel time from C to A under new conditions
def upstream_time_C_to_A_new : ℝ := 7.7

theorem upstream_time_proof (h : initial_conditions) : 
  upstream_time_C_to_A_new = 7.7 :=
sorry

end upstream_time_proof_l341_341148


namespace omega_value_interval_monotonic_increase_l341_341029

def f (ω : ℝ) (x : ℝ) : ℝ := sin (2 * ω * x) + 2 * (cos (ω * x)) ^ 2

-- Given ω > 0 and the smallest positive period of f is π
axiom ω_gt_zero (ω : ℝ) : ω > 0
axiom period_π (ω : ℝ) : ∃ T > 0, T = π ∧ ∀ x, f ω (x + T) = f ω x

-- Prove ω = 1
theorem omega_value (ω : ℝ) : ω = 1 :=
sorry

def g (x : ℝ) : ℝ := (λ x, sqrt 2 * sin (x - π / 12) + 1) x 

-- Prove the interval of monotonic increase of g
theorem interval_monotonic_increase (k : ℤ) (x : ℝ) :
  2 * k * π - 5 * π / 12 ≤ x ∧ x ≤ 2 * k * π + 7 * π / 12 ↔ ∀ x1 x2, 2 * k * π - 5 * π / 12 ≤ x1 → x1 ≤ x2 → x2 ≤ 2 * k * π + 7 * π / 12 → g x1 ≤ g x2 :=
sorry

end omega_value_interval_monotonic_increase_l341_341029


namespace math_problem_proof_l341_341868

def y1 := 2
def y2 := 3
def z1 := 3
def z2 := 5
def z3 := 2
def r1 := 4
def r2 := 6
def x1 := 4
def x2 := 1

def y := 3 * y1^(y2 + 2) + 4
def z := 2 * z1^3 - (z2^2) * z3
def r := real.sqrt (r1^3 + r2 + 2)
def x := 2 * x1 * y1^3 - x2^2 + 10

theorem math_problem_proof : y = 100 ∧ z = 4 ∧ abs (r - 8.4852) < 0.0001 ∧ x = 73 := by
  sorry

end math_problem_proof_l341_341868


namespace harrys_fish_count_l341_341140

theorem harrys_fish_count : 
  let sam_fish := 7
  let joe_fish := 8 * sam_fish
  let harry_fish := 4 * joe_fish
  harry_fish = 224 :=
by
  sorry

end harrys_fish_count_l341_341140


namespace find_alpha_l341_341792

-- Definitions of conditions
def terminal_side_symmetric (α : ℝ) : Prop :=
  ∃ k : ℤ, α = (2 * k + 1) * π / 3

def in_interval (α : ℝ) : Prop :=
  -4 * π < α ∧ α < -2 * π

-- Proof problem statement
theorem find_alpha (α : ℝ) (h1 : terminal_side_symmetric α) (h2 : in_interval α) :
  α = -11 * π / 3 ∨ α = -5 * π / 3 :=
by
  sorry

end find_alpha_l341_341792


namespace super_domino_double_probability_l341_341274

theorem super_domino_double_probability :
  ∃ (dominos : set (ℕ × ℕ)),
    (∀ a b : ℕ, a ∈ finset.range 15 ∧ b ∈ finset.range 15 → (a, b) ∈ dominos) ∧
    (∀ d ∈ finset.range 15, (d, d) ∈ dominos) →
    (∃ n : ℚ, n = 15 / (Nat.choose 15 2 + 15) ∧ n = 1 / 8) :=
by
  -- Placeholder for the proof steps
  sorry

end super_domino_double_probability_l341_341274


namespace certain_event_l341_341967

def conditionA : Prop := "The temperature in Aojiang on June 1st this year is 30 degrees."
def conditionB : Prop := ∀ (n : Nat), ¬ (n = 10) → (True → False) -- Representing the impossible event
def conditionC : Prop := ∀ (x : ℝ), (if x = 1 then True else False) -- Representing the certain event that a stone will fall
def conditionD : Prop := "In this math competition, every participating student will score full marks."

theorem certain_event (A: Prop) (B: Prop) (C: Prop) (D: Prop):
  C = ∀ (x : ℝ), (if x = 1 then True else False) :=
by
  sorry

end certain_event_l341_341967


namespace binom_9_5_equals_126_l341_341307

theorem binom_9_5_equals_126 : Nat.binom 9 5 = 126 := 
by 
  sorry

end binom_9_5_equals_126_l341_341307


namespace find_spy_l341_341073

-- Define types for Knight, Liar, and Spy
inductive Knight | mk : Knight
inductive Liar | mk : Liar
inductive Spy | mk : Spy

-- Define statements
def statement_A (x : Knight ∨ Liar ∨ Spy) (named : A -> bool) : Prop :=
  named A = true

def statement_B (x : Knight ∨ Liar ∨ Spy) (statement_A_result : Prop) : Prop :=
  statement_A_result

def statement_C (x : Knight ∨ Liar ∨ Spy) (named : C -> bool) : Prop :=
  named C = true

-- Conditions
axiom cond_spy_is_named (x : Knight ∨ Liar ∨ Spy) : (∃ y, (Spy.mk ↔ y = Murdock)) 

axiom cond_unique_roles : 
  ∃ (A B C : Prop), 
  (statement_A (A ∨ B ∨ C) (λ x, x = Murdock)) ∧ 
  (statement_B (A ∨ B ∨ C) (statement_A (A ∨ B ∨ C) (λ x, x = Murdock))) ∧ 
  (statement_C (A ∨ B ∨ C) (λ x, x = Murdock)) ∧ 
  (A ↔ Knight.mk) ∧ (B ↔ Liar.mk) ∧ (C ↔ Spy.mk)

-- Proving that A is the spy
theorem find_spy : (Spy.mk = A) :=
  sorry

end find_spy_l341_341073


namespace Thelma_cuts_each_tomato_into_8_slices_l341_341211

-- Conditions given in the problem
def slices_per_meal := 20
def family_size := 8
def tomatoes_needed := 20

-- The quantity we want to prove
def slices_per_tomato := 8

-- Statement to be proven: Thelma cuts each green tomato into the correct number of slices
theorem Thelma_cuts_each_tomato_into_8_slices :
  (slices_per_meal * family_size) = (tomatoes_needed * slices_per_tomato) :=
by 
  sorry

end Thelma_cuts_each_tomato_into_8_slices_l341_341211


namespace problem_1_problem_2_problem_3_l341_341867

-- (Ⅰ) p = 1/2, q = -2/3, find b_3
theorem problem_1 : 
  let p := (1:ℚ) / 2,
      q := -(2:ℚ) / 3 
  in b_m = find_b 3 :=
begin
  sorry
end

-- (Ⅱ) p = 2, q = -1, sum of first 2m terms of b_m
theorem problem_2 (m : ℕ) : 
  let p := (2:ℚ),
      q := -(1:ℚ) 
  in sum_first_2m_terms  = m^2 + 2*m :=
begin
  sorry
end

-- (Ⅲ) p = 1/4, -1/4 ≤ q < 0 such that b_m = 4m + 1
theorem problem_3 : 
  ∃ p q : ℚ, p = (1:ℚ) / 4 ∧ -1/4 ≤ q ∧ q < 0 ∧ 
    ∀ m : ℕ, b_m = 4*m + 1 :=
begin
  sorry
end

end problem_1_problem_2_problem_3_l341_341867


namespace intersection_A_B_l341_341036

def setA := {0, 1, 2, 3}
def setB := { x | ∃ n ∈ setA, x = n ^ 2 }
def P := setA ∩ setB

theorem intersection_A_B :
  P = {0, 1} :=
by
  sorry

end intersection_A_B_l341_341036


namespace total_bike_count_l341_341825

def total_bikes (bikes_jungkook bikes_yoongi : Nat) : Nat :=
  bikes_jungkook + bikes_yoongi

theorem total_bike_count : total_bikes 3 4 = 7 := 
  by 
  sorry

end total_bike_count_l341_341825


namespace breanna_books_count_l341_341935

theorem breanna_books_count 
  (tony_books : ℕ)
  (dean_books : ℕ)
  (shared_books_td : ℕ)
  (shared_books_all : ℕ)
  (total_books : ℕ)
  (tony_books = 23)
  (dean_books = 12)
  (shared_books_td = 3)
  (shared_books_all = 1)
  (total_books = 47) : 
  ∃ (breanna_books : ℕ), breanna_books = 20 := 
by
  let unique_books_tony := tony_books - shared_books_td - shared_books_all
  let unique_books_dean := dean_books - shared_books_td - shared_books_all
  let breanna_books := total_books - unique_books_tony - unique_books_dean
  use breanna_books
  sorry

end breanna_books_count_l341_341935


namespace max_perimeter_convex_quadrilateral_l341_341799

theorem max_perimeter_convex_quadrilateral :
  ∃ (AB BC AD CD AC BD : ℝ), 
    AB = 1 ∧ BC = 1 ∧
    AD ≤ 1 ∧ CD ≤ 1 ∧ AC ≤ 1 ∧ BD ≤ 1 ∧
    2 + 4 * Real.sin (Real.pi / 12) = 
      AB + BC + AD + CD :=
sorry

end max_perimeter_convex_quadrilateral_l341_341799


namespace problem_solution_l341_341111

theorem problem_solution (n : ℝ) (a b c : ℕ) (h_eqn : (4 / (n - 2)) + (6 / (n - 6)) + (18 / (n - 18)) + (20 / (n - 20)) = n^2 - 13 * n - 8)
(h_n : n = a + Real.sqrt (b + Real.sqrt c))
(h_largest : ∀ m, (4 / (m - 2)) + (6 / (m - 6)) + (18 / (m - 18)) + (20 / (m - 20)) = m^2 - 13 * m - 8 → m ≤ n) :
a + b + c = 82 :=
begin
  sorry
end

end problem_solution_l341_341111


namespace binom_9_5_l341_341302

open Nat

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Theorem to prove binom(9, 5) = 126
theorem binom_9_5 : binom 9 5 = 126 := by
  sorry

end binom_9_5_l341_341302


namespace max_diff_surface_area_and_lateral_area_of_prism_l341_341063

theorem max_diff_surface_area_and_lateral_area_of_prism :
  let r := 2
  let sphere_area := 4 * π * r^2
  ∃ (a h : ℝ), 
  (r = real.sqrt ((h / 2)^2 + (real.sqrt 2 / 2 * a)^2)) →
  sphere_area - 4 * a * h = 16 * (π - real.sqrt 2) :=
by 
  sorry

end max_diff_surface_area_and_lateral_area_of_prism_l341_341063


namespace mr_smiths_sixth_child_not_represented_l341_341499

def car_plate_number := { n : ℕ // ∃ a b : ℕ, n = 1001 * a + 110 * b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 }
def mr_smith_is_45 (n : ℕ) := (n % 100) = 45
def divisible_by_children_ages (n : ℕ) : Prop := ∀ i : ℕ, 1 ≤ i ∧ i ≤ 9 → n % i = 0

theorem mr_smiths_sixth_child_not_represented :
    ∃ n : car_plate_number, mr_smith_is_45 n.val ∧ divisible_by_children_ages n.val → ¬ (6 ∣ n.val) :=
by
  sorry

end mr_smiths_sixth_child_not_represented_l341_341499


namespace sqrt_sum_ineq_l341_341832

open Real

theorem sqrt_sum_ineq (a b c d : ℝ) (h : a ≥ 0) (h1 : b ≥ 0) (h2 : c ≥ 0) (h3 : d ≥ 0)
  (h4 : a + b + c + d = 4) : 
  sqrt (a + b + c) + sqrt (b + c + d) + sqrt (c + d + a) + sqrt (d + a + b) ≥ 6 :=
sorry

end sqrt_sum_ineq_l341_341832


namespace triangle_inequality_lt_l341_341465

theorem triangle_inequality_lt {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a < b + c) (h2 : b < a + c) (h3 : c < a + b) : a^2 + b^2 + c^2 < 2 * (a*b + b*c + c*a) := 
sorry

end triangle_inequality_lt_l341_341465


namespace a_n_general_formula_b_n_general_formula_T_n_less_than_half_l341_341723

noncomputable theory

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∃ d, ∀ n, a (n + 1) = a n + d

def forms_geometric_sequence (x y z : ℤ) : Prop :=
  y ^ 2 = x * z

def is_b_sequence (a b : ℕ → ℤ) : Prop :=
  b 1 = 1 ∧ ∀ n, b (n + 1) = b n + a n

def a_n (n : ℕ) : ℤ := 2 * n - 1
def b_n (n : ℕ) : ℤ := n ^ 2 - 2 * n + 2

def partial_sum (f : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ k in range n, f k

def T_n (n : ℕ) : ℚ :=
  partial_sum (λ n, 1 / ((a_n n) * (a_n (n + 1)))) n

-- Main theorem statements
theorem a_n_general_formula : 
  is_arithmetic_sequence a_n :=
  sorry

theorem b_n_general_formula :
  is_b_sequence a_n b_n :=
  sorry

theorem T_n_less_than_half (n : ℕ) :
  T_n n < 1 / 2 :=
  sorry

end a_n_general_formula_b_n_general_formula_T_n_less_than_half_l341_341723


namespace crayon_selection_l341_341797

theorem crayon_selection :
  let total_crayons := 15
  let red_crayons := 3
  let selections_for_2_red := Nat.choose 3 2
  let selections_for_3_non_red := Nat.choose 12 3
in total_crayons = 15 ∧ red_crayons = 3 →
   selections_for_2_red * selections_for_3_non_red = 660 :=
by
  sorry

end crayon_selection_l341_341797


namespace set_M_roster_method_l341_341132

open Set

theorem set_M_roster_method :
  {a : ℤ | ∃ (n : ℕ), 6 = n * (5 - a)} = {-1, 2, 3, 4} := by
  sorry

end set_M_roster_method_l341_341132


namespace third_number_is_42_l341_341930

variable (x : ℕ)

def number1 : ℕ := 5 * x
def number2 : ℕ := 6 * x
def number3 : ℕ := 8 * x

theorem third_number_is_42 (h : number1 x + number3 x = number2 x + 49) : number2 x = 42 :=
by
  sorry

end third_number_is_42_l341_341930


namespace arithmetic_progression_number_of_terms_l341_341808

variable (a d : ℕ)
variable (n : ℕ) (h_n_even : n % 2 = 0)
variable (h_sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 60)
variable (h_sum_even : (n / 2) * (2 * (a + d) + (n - 2) * d) = 80)
variable (h_diff : (n - 1) * d = 16)

theorem arithmetic_progression_number_of_terms : n = 8 :=
by
  sorry

end arithmetic_progression_number_of_terms_l341_341808


namespace nine_digit_divisible_by_11_l341_341430

theorem nine_digit_divisible_by_11 (m : ℕ) (k : ℤ) (h1 : 8 + 4 + m + 6 + 8 = 26 + m)
(h2 : 5 + 2 + 7 + 1 = 15)
(h3 : 26 + m - 15 = 11 + m)
(h4 : 11 + m = 11 * k) :
m = 0 := by
  sorry

end nine_digit_divisible_by_11_l341_341430


namespace function_monotonicity_l341_341787

noncomputable def f (x a : ℝ) : ℝ := (x^2 + 2 * x + a) / (x + 1)

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

theorem function_monotonicity (a : ℝ):
  (is_monotonically_increasing (λ x, f x a) (set.Ici 1)) →
  a ≤ 5 :=
sorry

end function_monotonicity_l341_341787


namespace area_of_triangle_l341_341170

theorem area_of_triangle (S_x S_y S_z S : ℝ)
  (hx : S_x = Real.sqrt 7) (hy : S_y = Real.sqrt 6)
  (hz : ∃ k : ℕ, S_z = k) (hs : ∃ n : ℕ, S = n)
  : S = 7 := by
  sorry

end area_of_triangle_l341_341170


namespace find_a_l341_341756

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h : f = λ x, a * x^3 - 2 * x) (p : f (-1) = 4) : a = -2 :=
by
  sorry

end find_a_l341_341756


namespace decimal_to_base5_equiv_l341_341310

def base5_representation (n : ℕ) : ℕ := -- Conversion function (implementation to be filled later)
  sorry

theorem decimal_to_base5_equiv : base5_representation 88 = 323 :=
by
  -- Proof steps go here.
  sorry

end decimal_to_base5_equiv_l341_341310


namespace polynomial_inequality_l341_341858

open Polynomial

variables {R : Type*} [OrderedRing R] (P : Polynomial R) (n : ℕ)

theorem polynomial_inequality 
  (hP_deg : P.degree = n) 
  (hP_roots_real : ∀ r ∈ P.roots, r ∈ ℝ) :
  (n - 1 : ℕ) * ((P.derivative.eval x)^2) ≥ n * (P.eval x * (P.derivative.derivative.eval x)) ∧ 
  ((n - 1 : ℕ) * ((P.derivative.eval x)^2) = n * (P.eval x * (P.derivative.derivative.eval x)) ↔ 
   ∃ c a : R, P = Polynomial.C c * (X - Polynomial.C a)^n) :=
sorry

end polynomial_inequality_l341_341858


namespace domain_of_f_l341_341658

def f (x : ℝ) := sqrt (2 - sqrt (4 - sqrt (x + 1)))

theorem domain_of_f : ∀ x, -1 ≤ x ∧ x ≤ 15 → ∃ y, y = f x := sorry

end domain_of_f_l341_341658


namespace min_volume_when_equal_volume_surface_area_l341_341946

open BigOperators

noncomputable def volume (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def surface_area (r h : ℝ) : ℝ := 2 * π * r^2 + 2 * π * r * h

theorem min_volume_when_equal_volume_surface_area : 
  ∃ r h : ℝ, surface_area r h = volume r h ∧ 
  (∀ r' h' : ℝ, surface_area r' h' = volume r' h' → volume r' h' ≥ volume r h) ∧
  volume 3 (2 * 3 / (3 - 2)) = 54 * π := 
sorry

end min_volume_when_equal_volume_surface_area_l341_341946


namespace ratio_speeds_l341_341984

-- Definitions for conditions
variable (v_A v_B : ℝ)

-- The given conditions
def condition_1 : Prop := True
def condition_2 : Prop := A at O and B at (0, -500)
def condition_3 : Prop := dist (2 * v_A, 0) = dist (0, -500 + 2 * v_B)
def condition_4 : Prop := dist (10 * v_A, 0) = dist (0, -500 + 10 * v_B)

-- Defining the distance function
def dist (x : ℝ, y : ℝ) : ℝ := sqrt (x^2 + y^2)

-- The target theorem to prove
theorem ratio_speeds (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) : 
  v_A / v_B = 5 / 6 := sorry

end ratio_speeds_l341_341984


namespace evaluate_trig_expression_l341_341375

theorem evaluate_trig_expression (α : ℝ) (h : Real.tan α = -4/3) : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1 / 7 :=
by
  sorry

end evaluate_trig_expression_l341_341375


namespace petya_cannot_win_l341_341199

theorem petya_cannot_win (n : ℕ) (h : n ≥ 3) : ¬ ∃ strategy : ℕ → ℕ → Prop, 
  (∀ k, strategy k (k+1) ∧ strategy k (k-1))
  ∧ ∀ m, ¬ strategy n m :=
sorry

end petya_cannot_win_l341_341199


namespace triangles_congruent_l341_341512

variable {Point : Type} [EuclideanGeometry Point]

-- Definitions of points
variables (A B C A1 B1 C1 D D1 : Point)

-- Definitions of angles and distances
variables (∠BAC ∠B1A1C1 : Angle)
variables (AD A1D1 AB A1B1 : ℝ)

-- Conditions
axiom angle_equal : ∠BAC = ∠B1A1C1
axiom angle_bisectors : is_angle_bisector A B C D ∧ is_angle_bisector A1 B1 C1 D1
axiom angle_bisectors_equal : AD = A1D1
axiom side_equal : AB = A1B1

theorem triangles_congruent : congruent (triangle A B C) (triangle A1 B1 C1) := 
by
  -- Insert the proof here
  sorry

end triangles_congruent_l341_341512


namespace work_problem_l341_341542

theorem work_problem (x : ℕ) (h : (5 * x) * 18 = 3 * x * 30) : 3 * x * 30 = 5 * x * 18 :=
begin
  exact h.symm,
end

end work_problem_l341_341542


namespace find_non_perfect_square_l341_341595

noncomputable def is_perfect_square (n : ℝ) : Prop := ∃ x : ℝ, x^2 = n

theorem find_non_perfect_square :
  ¬ is_perfect_square (6 ^ 2041) ∧ 
  is_perfect_square (3 ^ 2040) ∧ 
  is_perfect_square (7 ^ 2042) ∧ 
  is_perfect_square (8 ^ 2043) ∧ 
  is_perfect_square (9 ^ 2044) :=
begin
  -- Insert proof here
  sorry
end

end find_non_perfect_square_l341_341595


namespace zephyr_island_population_capacity_reach_l341_341445

-- Definitions for conditions
def acres := 30000
def acres_per_person := 2
def initial_year := 2023
def initial_population := 500
def population_growth_rate := 4
def growth_period := 20

-- Maximum population supported by the island
def max_population := acres / acres_per_person

-- Function to calculate population after a given number of years
def population (years : ℕ) : ℕ := initial_population * (population_growth_rate ^ (years / growth_period))

-- The Lean statement to prove that the population will reach or exceed max_capacity in 60 years
theorem zephyr_island_population_capacity_reach : ∃ t : ℕ, t ≤ 60 ∧ population t ≥ max_population :=
by
  sorry

end zephyr_island_population_capacity_reach_l341_341445


namespace range_of_a_l341_341433

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then -x + 5 else 2 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x, 3 ≤ f x a) ∧ (0 < a) ∧ (a ≠ 1) → 1 < a ∧ a ≤ 2 :=
by
  intro h
  sorry

end range_of_a_l341_341433


namespace sugar_needed_in_two_minutes_l341_341579

-- Let a be the amount of sugar needed per chocolate bar.
def sugar_per_chocolate_bar : ℝ := 1.5

-- Let b be the number of chocolate bars produced per minute.
def chocolate_bars_per_minute : ℕ := 36

-- Let t be the time in minutes.
def time_in_minutes : ℕ := 2

theorem sugar_needed_in_two_minutes : 
  let sugar_in_one_minute := chocolate_bars_per_minute * sugar_per_chocolate_bar
  let total_sugar := sugar_in_one_minute * time_in_minutes
  total_sugar = 108 := by
  sorry

end sugar_needed_in_two_minutes_l341_341579


namespace carnival_candies_l341_341091

theorem carnival_candies :
  ∃ (c : ℕ), c % 5 = 4 ∧ c % 6 = 3 ∧ c % 8 = 5 ∧ c < 150 ∧ c = 69 :=
by
  sorry

end carnival_candies_l341_341091


namespace gcd_204_85_l341_341187

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l341_341187


namespace log_inequality_l341_341095

open Real

theorem log_inequality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  log a ((b^2 / (a * c)) - b + (a * c)) * log b ((c^2 / (a * b)) - c + (a * b)) * log c ((a^2 / (b * c)) - a + (b * c)) ≥ 1 :=
begin
  sorry
end

end log_inequality_l341_341095


namespace lines_intersect_at_same_point_l341_341539

theorem lines_intersect_at_same_point : 
  (∃ (x y : ℝ), y = 2 * x - 1 ∧ y = -3 * x + 4 ∧ y = 4 * x + m) → m = -3 :=
by
  sorry

end lines_intersect_at_same_point_l341_341539


namespace polynomial_root_p_value_l341_341419

theorem polynomial_root_p_value (p : ℝ) : (3 : ℝ) ^ 3 + p * (3 : ℝ) - 18 = 0 → p = -3 :=
by
  intro h
  sorry

end polynomial_root_p_value_l341_341419


namespace donuts_selection_l341_341878

theorem donuts_selection :
  (∃ g c p : ℕ, g + c + p = 6 ∧ g ≥ 1 ∧ c ≥ 1 ∧ p ≥ 1) →
  ∃ k : ℕ, k = 10 :=
by {
  -- The mathematical proof steps are omitted according to the instructions
  sorry
}

end donuts_selection_l341_341878


namespace statement_c_false_l341_341234

theorem statement_c_false : ¬ ∃ (x y : ℝ), x^2 + y^2 < 0 := by
  sorry

end statement_c_false_l341_341234


namespace solve_for_a_l341_341377

theorem solve_for_a (a x : ℝ) (h : x = 1 ∧ 2 * a * x - 2 = a + 3) : a = 5 :=
by
  sorry

end solve_for_a_l341_341377


namespace number_of_disjoint_subsets_remainder_l341_341104

-- Define the set S
def S : set ℕ := {i | 1 ≤ i ∧ i ≤ 12}

-- The main theorem statement
theorem number_of_disjoint_subsets_remainder :
  let n := 1 / 2 * (3^12 - 2 * 2^12 + 1) in
  n % 1000 = 625 :=
by
  sorry

end number_of_disjoint_subsets_remainder_l341_341104


namespace domain_f_l341_341586

def f (x : ℝ) : ℝ := Real.sqrt (x - 5) + Real.cbrt (x - 7)

theorem domain_f :
  {x : ℝ | ∃ y : ℝ, y = f x} = set.Ici 5 :=
sorry

end domain_f_l341_341586


namespace inverse_of_3_mod_199_l341_341330

theorem inverse_of_3_mod_199 : (3 * 133) % 199 = 1 :=
by
  sorry

end inverse_of_3_mod_199_l341_341330


namespace boys_trees_l341_341318

theorem boys_trees (avg_per_person trees_per_girl trees_per_boy : ℕ) :
  avg_per_person = 6 →
  trees_per_girl = 15 →
  (1 / trees_per_boy + 1 / trees_per_girl = 1 / avg_per_person) →
  trees_per_boy = 10 :=
by
  intros h_avg h_girl h_eq
  -- We will provide the proof here eventually
  sorry

end boys_trees_l341_341318


namespace range_of_a_l341_341881

noncomputable def prop_p (a x : ℝ) : Prop := 3 * a < x ∧ x < a

noncomputable def prop_q (x : ℝ) : Prop := x^2 - x - 6 < 0

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, ¬ prop_p a x) ∧ ¬ (∃ x : ℝ, ¬ prop_p a x) → ¬ (∃ x : ℝ, ¬ prop_q x) → -2/3 ≤ a ∧ a < 0 := 
by
  sorry

end range_of_a_l341_341881


namespace james_puzzle_completion_time_l341_341822

theorem james_puzzle_completion_time :
  let num_puzzles := 2
  let pieces_per_puzzle := 2000
  let pieces_per_10_minutes := 100
  let total_pieces := num_puzzles * pieces_per_puzzle
  let sets_of_100_pieces := total_pieces / pieces_per_10_minutes
  let total_minutes := sets_of_100_pieces * 10
  total_minutes = 400 :=
by
  -- Definitions based on conditions
  let num_puzzles := 2
  let pieces_per_puzzle := 2000
  let pieces_per_10_minutes := 100
  let total_pieces := num_puzzles * pieces_per_puzzle
  let sets_of_100_pieces := total_pieces / pieces_per_10_minutes
  let total_minutes := sets_of_100_pieces * 10

  -- Using sorry to skip proof
  sorry

end james_puzzle_completion_time_l341_341822


namespace nell_cards_left_l341_341502

def initial_cards : ℕ := 304
def cards_given : ℕ := 28
def cards_left : ℕ := initial_cards - cards_given

theorem nell_cards_left : cards_left = 276 := by
  rw [cards_left, initial_cards, cards_given]
  sorry

end nell_cards_left_l341_341502


namespace trigonometric_identity_l341_341985

theorem trigonometric_identity : sin (20 * real.pi / 180) * cos (10 * real.pi / 180) - cos (160 * real.pi / 180) * sin (10 * real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l341_341985


namespace a33_is_3_l341_341354

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 6 ∧ ∀ n, a (n + 2) = a (n + 1) - a n

theorem a33_is_3 (a : ℕ → ℤ) (h : sequence a) : a 33 = 3 := by
  sorry

end a33_is_3_l341_341354


namespace total_money_raised_l341_341570

-- Assume there are 30 students in total
def total_students := 30

-- Assume 10 students raised $20 each
def students_raising_20 := 10
def money_raised_per_20 := 20

-- The rest of the students raised $30 each
def students_raising_30 := total_students - students_raising_20
def money_raised_per_30 := 30

-- Prove that the total amount raised is $800
theorem total_money_raised :
  (students_raising_20 * money_raised_per_20) +
  (students_raising_30 * money_raised_per_30) = 800 :=
by
  sorry

end total_money_raised_l341_341570


namespace probability_X_less_than_0_l341_341745

open Real

noncomputable def X : Type := sorry  -- Define the random variable X

axiom X_normal_distribution (σ : ℝ) : X follows normal_distribution(2, σ^2)
axiom P_X_less_than_4 : P(X < 4) = 0.8

theorem probability_X_less_than_0 (σ : ℝ) [h1 : X follows normal_distribution(2, σ^2)] (h2 : P(X < 4) = 0.8) : P(X < 0) = 0.2 :=
sorry

end probability_X_less_than_0_l341_341745


namespace ones_digit_of_sum_of_powers_l341_341590

theorem ones_digit_of_sum_of_powers :
  (1^2011 + 2^2011 + 3^2011 + 4^2011 + 5^2011 + 6^2011 + 7^2011 + 8^2011 + 9^2011 + 10^2011) % 10 = 5 :=
by
  sorry

end ones_digit_of_sum_of_powers_l341_341590


namespace remainder_div_8_l341_341980

theorem remainder_div_8 (x : ℤ) (h : ∃ k : ℤ, x = 63 * k + 27) : x % 8 = 3 :=
by
  sorry

end remainder_div_8_l341_341980


namespace largest_angle_in_isosceles_triangle_l341_341810

-- Definitions of the conditions from the problem
def isosceles_triangle (A B C : ℕ) : Prop :=
  A = B ∨ B = C ∨ A = C

def angle_opposite_equal_side (θ : ℕ) : Prop :=
  θ = 50

-- The proof problem statement
theorem largest_angle_in_isosceles_triangle (A B C : ℕ) (θ : ℕ)
  : isosceles_triangle A B C → angle_opposite_equal_side θ → ∃ γ, γ = 80 :=
by
  sorry

end largest_angle_in_isosceles_triangle_l341_341810


namespace minimum_value_l341_341848

theorem minimum_value (
  (a b c d e f : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f) 
  (h_sum : a + b + c + d + e + f = 9)
  ) : 
  ∃ x : ℝ, x = (1/a + 9/b + 25/c + 49/d + 81/e + 121/f) ∧ x ≥ 286^2 / 9 := 
sorry

end minimum_value_l341_341848


namespace f_minus_exp_increasing_min_x_value_l341_341853

-- Definitions and conditions from the problem
variables {f : ℝ → ℝ}

-- Condition 1: f' exists and f'(x) * exp(-x) > 1 for all x in (0, +∞)
axiom h1 : ∀ x > 0, (has_deriv_at f x) ∧ (f' x) * real.exp (-x) > 1

-- Condition 2: f(ln x) ≥ x + sqrt(e)
axiom h2 : ∀ x > 0, f (real.log x) ≥ x + real.sqrt real.exp 1

-- Condition 3: f(1/2) = 2 * sqrt(e)
axiom h3 : f (1 / 2) = 2 * real.sqrt real.exp 1

-- Theorem 1: f(x) - exp(x) is increasing for x in (0, +∞)
theorem f_minus_exp_increasing : ∀ x > 0, deriv (λ (x : ℝ), f x - real.exp x) x > 0 :=
by
  assume x hx,
  have h_deriv : has_deriv_at (λ x, f x - real.exp x) (f' x - real.exp x) x := (has_deriv_at.sub (has_deriv_at_of_has_deriv_at f x) (has_deriv_at_exp x)),
  exact (lt_of_lt_of_le (lt_of_mul_gt_of_pos hx (h1 x hx)) (sub_nonneg_of_le (le_refl (real.exp x))))

-- Theorem 2: The minimum value of x such that x ≥ sqrt(e)
theorem min_x_value : ∃ x > 0, x = real.sqrt real.exp 1 ∧ f (real.log x) ≥ x + real.sqrt real.exp 1 :=
by
  use real.sqrt real.exp 1,
  split,
  exact real.sqrt_pos_of_pos real.exp_pos,
  split,
  refl,
  exact (h2 (real.sqrt real.exp 1) (real.sqrt_pos_of_pos real.exp_pos))

end f_minus_exp_increasing_min_x_value_l341_341853


namespace angle_opposite_c_zero_l341_341065

theorem angle_opposite_c_zero (a b c : ℝ) 
  (h : (a + b + c) * (a + b - c) = 4 * a * b) : 
  ∠C = 0 :=
sorry

end angle_opposite_c_zero_l341_341065


namespace union_of_A_B_l341_341399

open Set

def A : Set ℝ := { x | x^2 + 2x - 3 < 0 }
def B : Set ℝ := { x | x^2 - 4x ≤ 0 }

theorem union_of_A_B : A ∪ B = { x | -3 < x ∧ x ≤ 4 } := 
by 
  sorry

end union_of_A_B_l341_341399


namespace find_a_n_l341_341742

variable {α : Type*} [LinearOrderedField α] (d : α) (a_n : ℕ → α)

axiom arithmetic_sequence : ∀ n, a_n (n + 1) = a_n n + d
axiom common_difference_nonzero : d ≠ 0
axiom a_3_eq_5 : a_n 3 = 5
axiom geom_sequence : (a_n 2)^2 = (a_n 1) * (a_n 4)

theorem find_a_n : a_n = λ n, 2 * n - 1 :=
by
  sorry

end find_a_n_l341_341742


namespace at_most_5_opposite_signs_l341_341052

def opposite_sign_bound (l : List ℤ) : Prop :=
  l.length = 6 → 
  ∀ (opposite_count ≤ 5), 
  (opposite_count = l.count (λ x, x < 0)).xor (l.count (λ x, x < 0) % 2 = 1)

theorem at_most_5_opposite_signs (l : List ℤ) : 
  opposite_sign_bound l := 
by sorry

end at_most_5_opposite_signs_l341_341052


namespace find_parallel_line_l341_341695

theorem find_parallel_line (m : ℝ) (x1 y1 : ℝ) (h1 : y1 = (3 / 2) * x1 + m) :
  (∃ k b : ℝ, k = (3 / 2) ∧ b = - 11 / 2 ∧ ∀ x y : ℝ, y = k * x + b ↔ 3 * x - 2 * y - 11 = 0) :=
by
  use (3 / 2)
  use (-11 / 2)
  split
  { rw [eq_self_iff_true] }
  split
  { rw [eq_self_iff_true] }
  intro x y
  split
  { intro h
    calc
      3 * x - 2 * y - 11
        = 3 * x - 2 * ((3 / 2) * x + - (11 / 2)) - 11 : by rw [h]
    ... = 3 * x - (3 * x) - (-11) - 11               : by ring
    ... = 0                                         : by norm_num }
  { intro h
    have : y = (3 / 2) * x - 11 / 2 := by
      calc
        y
          = (3 / 2) * x + - 11 / 2 : by sorry
    rw [this] }

end find_parallel_line_l341_341695


namespace period_of_f_max_min_values_of_f_l341_341032

noncomputable def f (x : ℝ) : ℝ :=
  sin(x)^2 - sin(x - π / 6)^2

theorem period_of_f : real.periodic f π :=
by sorry

theorem max_min_values_of_f :
  (x ∈ set.Icc (-(π / 3)) (π / 4)) →
  (f x ≤ sqrt 3 / 4) ∧ (f x ≥ -1 / 2) :=
by sorry

end period_of_f_max_min_values_of_f_l341_341032


namespace sum_of_solutions_sum_of_integers_satisfying_eq_l341_341548

theorem sum_of_solutions (x : ℤ) (h : x^2 = x + 256) : x = 16 ∨ x = -16 := sorry

theorem sum_of_integers_satisfying_eq : ∑ x in finset.filter (λ x, x^2 = x + 256) (finset.Icc -16 16), x = 0 := 
begin
  sorry
end

end sum_of_solutions_sum_of_integers_satisfying_eq_l341_341548


namespace oil_to_water_ratio_in_bottle_D_l341_341927

noncomputable def bottle_oil_water_ratio (CA : ℝ) (CB : ℝ) (CC : ℝ) (CD : ℝ) : ℝ :=
  let oil_A := (1 / 2) * CA
  let water_A := (1 / 2) * CA
  let oil_B := (1 / 4) * CB
  let water_B := (1 / 4) * CB
  let total_water_B := CB - oil_B - water_B
  let oil_C := (1 / 3) * CC
  let water_C := 0.4 * CC
  let total_water_C := CC - oil_C - water_C
  let total_capacity_D := CD
  let total_oil_D := oil_A + oil_B + oil_C
  let total_water_D := water_A + total_water_B + water_C + total_water_C
  total_oil_D / total_water_D

theorem oil_to_water_ratio_in_bottle_D (CA : ℝ) :
  let CB := 2 * CA
  let CC := 3 * CA
  let CD := CA + CC
  bottle_oil_water_ratio CA CB CC CD = (2 / 3.7) :=
by 
  sorry

end oil_to_water_ratio_in_bottle_D_l341_341927


namespace rooms_in_store_l341_341248

theorem rooms_in_store (x : ℕ) (h1 : (7 * x + 7)) (h2 : (9 * (x - 1))) : x = 8 :=
by sorry

end rooms_in_store_l341_341248


namespace exists_solution_in_interal_l341_341334

theorem exists_solution_in_interal :
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ (t + ((t + 1 / Real.sqrt 3) / (1 - t / Real.sqrt 3)) + (3 * t - t^3) / (1 - 3 * (t^2)) = 0) :=
sorry

end exists_solution_in_interal_l341_341334


namespace shaded_area_is_24_l341_341530

-- Define the problem conditions
def square_Area (s : ℝ) := s * s

-- Given data
variables (ABCD_area : ℝ)
def s_ABCD := real.sqrt ABCD_area

-- EFGH square conditions
def diagonal_EFGH := s_ABCD
def s_EFGH := diagonal_EFGH / real.sqrt 2
def EFGH_area := square_Area s_EFGH

-- Shaded region conditions
def shaded_fraction := (6 / 8 : ℚ)
def shaded_area := shaded_fraction * EFGH_area

theorem shaded_area_is_24 (h : ABCD_area = 64) : shaded_area ABCD_area = 24 := by
  -- Statements derived from given conditions
  have s_ABCD_sqr : s_ABCD = real.sqrt 64 := by 
    rw [h]
  have s_ABCD_val : s_ABCD = 8 := by 
    rw [s_ABCD_sqr, real.sqrt_eq_rpow, ←real.rpow_nat_cast 64 2] 
    norm_num
  
  -- Calculating lengths
  have diag_EFGH_val : diagonal_EFGH ABCD_area = 8 := by 
    rw [s_ABCD_val, diagonal_EFGH]
  have s_EFGH_val : s_EFGH ABCD_area = 4 * real.sqrt 2 := by 
    rw [diag_EFGH_val, ←real.mul_to_mul_sub, real.div_eq_mul_one_div, 
        real.one_div_sqrt, real.mul_sqrt_cancel zero_lt_two]
    norm_num
  
  -- Calculating areas
  have EFGH_area_val : EFGH_area ABCD_area = 32 := by 
    rw [s_EFGH_val, real.sq, mul_comm, ←mul_assoc, 
        EFGH_area, ←mul_eq_mul_left_iff]
    norm_num
  
  -- Final area of shaded region
  rw [shaded_area, shaded_fraction, EFGH_area_val]
  norm_num
  sorry -- insert the final necessary condition
  
-- Note: The exact implementation may require additional work on correct area calculations and imports.

end shaded_area_is_24_l341_341530


namespace distance_from_point_to_line_l341_341343

structure Point (R : Type) [Real R] :=
  (x y z : R)

def line (s : ℝ) : Point ℝ :=
  { x := 4 + s,
    y := 6 + 3 * s,
    z := 5 - s }

def distance (p1 p2 : Point ℝ) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem distance_from_point_to_line :
  distance (Point.mk 2 3 4) (line (-4 / 3)) = Real.sqrt 62 / 3 :=
by
  sorry

end distance_from_point_to_line_l341_341343


namespace expected_number_of_returns_l341_341092

noncomputable def expected_returns_to_zero : ℝ :=
  let p_move := 1 / 3
  let expected_value := -1 + (3 / (Real.sqrt 5))
  expected_value

theorem expected_number_of_returns : expected_returns_to_zero = (3 * Real.sqrt 5 - 5) / 5 :=
  by sorry

end expected_number_of_returns_l341_341092


namespace UN_anniversary_day_l341_341167

/--
The United Nations was founded on October 24, 1945, which was a Wednesday.
Prove that the 75th anniversary of this event occurred on a Friday in the year 2020.
-/
theorem UN_anniversary_day
  (start_day : ℕ := 3) -- Wednesday is represented by 3
  (years_diff : ℕ := 75)
  (leap_years : ℕ := 18)
  (regular_years : ℕ := 57)
  (days_in_week : ℕ := 7)
  (mod_days : ℕ := 2) : 
  (start_day + regular_years + 2 * leap_years) % days_in_week = 5 := 
by sorry

end UN_anniversary_day_l341_341167


namespace bushes_needed_l341_341321

theorem bushes_needed (bushes_yield : ℕ) (container_to_zucchini : ℕ → ℕ → ℕ) :
  bushes_yield = 10 →
  container_to_zucchini 6 3 = 2 →
  ∃ bushes_needed: ℕ, bushes_needed = 12 :=
by
  intros bushes_yield_eq container_trade_eq
  use 12
  sorry

end bushes_needed_l341_341321


namespace quadrilateral_is_parallelogram_l341_341726

variables (x1 y1 x2 y2 h k : ℝ)
variables (k_ne_midpoint_y : k ≠ (y1 + y2) / 2)
variables (P : ℝ × ℝ) (Q : ℝ × ℝ) (M : ℝ × ℝ) (S : ℝ × ℝ) (A : ℝ × ℝ)

def isParallelogram (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) / (D.1 - C.1) = (B.2 - A.2) / (D.2 - C.2) ∧
  (C.1 - B.1) / (D.1 - A.1) = (C.2 - B.2) / (D.2 - A.2)

theorem quadrilateral_is_parallelogram :
  let P := (x1, y1),
      Q := (x2, y2),
      M := ((x1 + x2) / 2, (y1 + y2) / 2),
      S := ((x1 + x2) / 2, k),
      A := (h, k) in
  isParallelogram A P M S :=
by sorry

end quadrilateral_is_parallelogram_l341_341726


namespace minor_axis_length_l341_341671

-- Define the five points
def point1 : ℝ × ℝ := (-5 / 2, 2)
def point2 : ℝ × ℝ := (0, 0)
def point3 : ℝ × ℝ := (0, 3)
def point4 : ℝ × ℝ := (4, 0)
def point5 : ℝ × ℝ := (4, 3)

-- Define the center of the rectangle formed by four points
def center : ℝ × ℝ := (2, 1.5)

-- Define the distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Assume we have the ellipse passing through these points,
-- with its axes parallel to the coordinate axes.
theorem minor_axis_length :
  ∀ (A B : ℝ), 
  (A ≠ B) →
  distance point1 center = 5 * Real.sqrt 2 / 2 →
  let b := Real.sqrt (56.25 / 17) in
  2 * b = 15 * Real.sqrt 17 / 17 :=
by
  intros
  sorry

end minor_axis_length_l341_341671


namespace olympic_emblem_cards_correct_l341_341616

-- Define the number of cards and the probability condition
def num_total_cards : ℕ := 8

def prob_not_both_olympic_emblem : ℚ := 25 / 28

-- Define the combinatorial functions
def C (n k : ℕ) : ℕ := nat.choose n k

noncomputable def olympic_emblem_cards : ℕ :=
  let n := nat.choose num_total_cards 2 in sorry

-- The probability distribution
def prob_xi_eq_1 : ℚ := (C 5 2 : ℚ) / C num_total_cards 2
def prob_xi_eq_2 : ℚ := ((C 3 2 : ℚ) / C num_total_cards 2 * (C 5 2 : ℚ) / C 6 2) + ((C 3 1 * C 5 1: ℚ) / C num_total_cards 2 * (C 4 2 : ℚ) / C 6 2)
def prob_xi_eq_3 : ℚ := ((C 3 2 : ℚ) / C num_total_cards 2 * (C 1 1 * C 5 1: ℚ) / C 6 2 * (C 4 2 : ℚ) / C 4 2) +
                           ((C 3 1 * C 5 1: ℚ) / C num_total_cards 2 * (C 2 2 : ℚ) / C 6 2 * (C 4 2 : ℚ) / C 4 2) +
                           ((C 3 1 * C 5 1: ℚ) / C num_total_cards 2 * (C 2 1 * C 4 1: ℚ) / C 6 2 * (C 3 2 : ℚ) / C 4 2)
def prob_xi_eq_4 : ℚ := ((C 3 1 * C 5 1: ℚ) / C 6 2 * (C 2 1 * C 4 1: ℚ) / C 6 2 * (C 1 1 * C 3 1: ℚ) / C 4 2 * (C 2 2 : ℚ) / C 2 2)

-- Expected value
noncomputable def expected_value : ℚ :=
  (1 * prob_xi_eq_1) + (2 * prob_xi_eq_2) + (3 * prob_xi_eq_3) + (4 * prob_xi_eq_4)

-- The main theorem combining both parts
theorem olympic_emblem_cards_correct :
  olympic_emblem_cards = 3 ∧ prob_xi_eq_1 = 5 / 14 ∧ prob_xi_eq_2 = 2 / 7 ∧ prob_xi_eq_3 = 3 / 14 ∧ prob_xi_eq_4 = 1 / 7 ∧ expected_value = 15 / 7 :=
sorry


end olympic_emblem_cards_correct_l341_341616


namespace sum_of_integers_squared_greater_sum_of_integers_satisfying_condition_l341_341552

theorem sum_of_integers_squared_greater (x : ℤ) (hx : x ^ 2 = 256 + x) : x = 16 ∨ x = -16 :=
begin
  have : x ^ 2 - x - 256 = 0,
  { rw ← hx, ring },
  apply quadratic_eq_zero_iff.mpr,
  use [16, -16],
  split;
  linarith,
end

theorem sum_of_integers_satisfying_condition : ∑ x in ({16, -16} : finset ℤ), x = 0 :=
by simp

end sum_of_integers_squared_greater_sum_of_integers_satisfying_condition_l341_341552


namespace sugar_used_in_two_minutes_l341_341575

-- Definitions according to conditions
def sugar_per_bar : ℝ := 1.5
def bars_per_minute : ℝ := 36
def minutes : ℝ := 2

-- Theorem statement
theorem sugar_used_in_two_minutes : bars_per_minute * sugar_per_bar * minutes = 108 :=
by
  -- We add sorry here to complete the proof later.
  sorry

end sugar_used_in_two_minutes_l341_341575


namespace parallelepiped_to_cube_l341_341911

-- Define the conditions of the problem
def edge_lengths (a b c : ℕ) := (a = 8 ∧ b = 8 ∧ c = 27)

-- Define the goal of the proof: dividing the parallelepiped into four parts that form a cube
theorem parallelepiped_to_cube (a b c : ℕ) (cube_side : ℕ) :
  edge_lengths a b c → cube_side = 12 :=
by
  intros h,
  cases h with ha hb,
  cases hb with hb hc,
  sorry -- provides the proof later

end parallelepiped_to_cube_l341_341911


namespace find_x_l341_341864
noncomputable def statement (k : ℤ) : Prop := 
  ∃ x : ℝ, (x - (real.pi / 3)) = (real.pi / 2 + k * real.pi + 4 * complex.I) ∧ x = 2

theorem find_x (k : ℤ) : statement k :=
    sorry

end find_x_l341_341864


namespace john_personal_payment_l341_341455

-- Definitions of the conditions
def cost_of_one_hearing_aid : ℕ := 2500
def number_of_hearing_aids : ℕ := 2
def insurance_coverage_percent : ℕ := 80

-- Derived definitions based on conditions
def total_cost : ℕ := cost_of_one_hearing_aid * number_of_hearing_aids
def insurance_coverage_amount : ℕ := total_cost * insurance_coverage_percent / 100
def johns_share : ℕ := total_cost - insurance_coverage_amount

-- Theorem statement (proof not included)
theorem john_personal_payment : johns_share = 1000 :=
sorry

end john_personal_payment_l341_341455


namespace number_of_valid_ellipses_l341_341751

def is_valid_ellipse (m n : ℕ) : Prop :=
  n > m ∧ m ∈ {1, 2, 3, 4, 5} ∧ n ∈ {1, 2, 3, 4, 5, 6, 7}

def count_valid_ellipses : ℕ :=
  (Finset.univ.filter (λ n, n ∈ {1, 2, 3, 4, 5, 6, 7})).sum (λ n,
    (Finset.univ.filter (λ m, is_valid_ellipse m n)).card)

theorem number_of_valid_ellipses : count_valid_ellipses = 20 :=
by sorry

end number_of_valid_ellipses_l341_341751


namespace arithmetic_sequence_general_term_bn_sum_l341_341363

section
variable {α : Type*} [LinearOrderedField α]

-- Definitions regarding the sequence {a_n}
def Sn (n : ℕ) (a : ℕ → α) : α := n * a n - n * (n - 1)
def a1 := (1 : α)
def a (n : ℕ) : α := 2 * n - 1

-- Prove that the general term of the sequence a_n is 2n - 1
theorem arithmetic_sequence_general_term (a : ℕ → α) (n : ℕ) (h1 : Sn n a = n * a n - n * (n - 1)) (h2 : a 1 = 1) :
  a n = 2 * n - 1 := sorry

-- Definitions regarding the sequence {b_n}
def bn (n : ℕ) (a : ℕ → α) : α := 2 / (a n * a (n + 1))
def Tn (n : ℕ) (b : ℕ → α) : α := ∑ k in finset.range n + 1, b k

-- Prove that Tn = ∑_{k=1}^{n} b_k = 2n / (2n + 1)
theorem bn_sum (a : ℕ → α) (b : ℕ → α) (n : ℕ) (h1 : ∀ n, a n = 2 * n - 1) (h2 : b n = 2 / (a n * a (n + 1))) :
  Tn n b = 2 * n / (2 * n + 1) := sorry

end

end arithmetic_sequence_general_term_bn_sum_l341_341363


namespace altitudes_sixth_degree_polynomial_l341_341544

theorem altitudes_sixth_degree_polynomial (a b c d : ℚ) (r1 r2 r3 : ℝ)
    (h1 : r1^3 + b*r1^2 + c*r1 + d = 0)
    (h2 : r2^3 + b*r2^2 + c*r2 + d = 0)
    (h3 : r3^3 + b*r3^2 + c*r3 + d = 0)
    (triangle_cond : r1 + r2 > r3 ∧ r1 + r3 > r2 ∧ r2 + r3 > r1) :
    ∃ (p : ℚ[X]), p.degree = 6 ∧ 
    (p.eval (h1 * h1) = 0 ∧ p.eval (h2 * h2) = 0 ∧ p.eval (h3 * h3) = 0) :=
sorry

end altitudes_sixth_degree_polynomial_l341_341544


namespace max_distance_is_15_6_miles_l341_341823

-- Definitions of speeds and directions as per the conditions
def jay_speed : ℝ := 1 / 20 -- miles per minute
def paul_speed : ℝ := 3.2 / 40 -- miles per minute
def naomi_speed : ℝ := 2 / 30 -- miles per minute

-- Time duration in minutes
def time_minutes : ℝ := 2 * 60

-- Calculations for distances covered after 2 hours
def jay_distance : ℝ := jay_speed * time_minutes -- 6 miles
def paul_distance : ℝ := paul_speed * time_minutes -- 9.6 miles
def naomi_distance : ℝ := naomi_speed * time_minutes -- 8 miles

-- Combined horizontal distance between Jay and Paul
def horizontal_distance : ℝ := jay_distance + paul_distance -- 15.6 miles

theorem max_distance_is_15_6_miles : horizontal_distance = 15.6 := 
by
  -- Placeholder for actual proof steps
  sorry

end max_distance_is_15_6_miles_l341_341823


namespace max_split_subsets_correct_l341_341099

noncomputable def max_split_subsets (n : ℕ) (h : n ≥ 3) : ℕ :=
  n - 2

theorem max_split_subsets_correct (n : ℕ) (h : n ≥ 3)
  (S : Finset (Finset (Fin n)))
  (hS : ∀ s ∈ S, 2 ≤ s.card ∧ s.card ≤ n - 1) :
  ∃ P : List (Fin n), 
  ∀ s ∈ S, ∃ a b c ∈ P.permutations, (a ∈ s ∧ c ∈ s ∧ b ∉ s ∧ List.index_of b P < List.index_of c P ∧ List.index_of c P < List.index_of a P.Permutations.contains) :=
begin
  let m := max_split_subsets n h,
  have : S.card ≤ m,
  sorry
end

end max_split_subsets_correct_l341_341099


namespace sum_of_positive_divisors_of_prime_l341_341961

theorem sum_of_positive_divisors_of_prime (h_prime : Nat.prime 47) : 1 + 47 = 48 :=
by
  have d1 : 1 ∣ 47 := Nat.one_dvd 47
  have d47 : 47 ∣ 47 := Nat.dvd_refl 47
  have divisors := [1, 47]
  have sum_divisors := List.sum divisors
  rw [List.sum_cons, List.sum_nil] at sum_divisors
  simp at sum_divisors
  exact sum_divisors

end sum_of_positive_divisors_of_prime_l341_341961


namespace range_of_omega_monotonic_decreasing_l341_341428

def omega_range (ω: ℝ) : Prop :=
  ω ∈ Set.Icc (1/2) (5/4)

def sine_monotonically_decreasing (ω: ℝ) (f: ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Ioo (π/2) π, f(x) < f(x + 1)

theorem range_of_omega_monotonic_decreasing :
  ∀ (ω : ℝ), (0 < ω ∧ sine_monotonically_decreasing ω (fun x => Real.sin(ω * x + π / 4))) → omega_range ω :=
sorry

end range_of_omega_monotonic_decreasing_l341_341428


namespace number_of_diet_soda_l341_341259

variable (d r : ℕ)

-- Define the conditions of the problem
def condition1 : Prop := r = d + 79
def condition2 : Prop := r = 83

-- State the theorem we want to prove
theorem number_of_diet_soda (h1 : condition1 d r) (h2 : condition2 r) : d = 4 :=
by
  sorry

end number_of_diet_soda_l341_341259


namespace sum_xyz_l341_341047

theorem sum_xyz (x y z : ℝ) (h1 : x + y = 1) (h2 : y + z = 1) (h3 : z + x = 1) : x + y + z = 3 / 2 :=
  sorry

end sum_xyz_l341_341047


namespace ratio_second_shop_to_shirt_l341_341495

-- Define the initial conditions in Lean
def initial_amount : ℕ := 55
def spent_on_shirt : ℕ := 7
def final_amount : ℕ := 27

-- Define the amount spent in the second shop calculation
def spent_in_second_shop (i_amt s_shirt f_amt : ℕ) : ℕ :=
  (i_amt - s_shirt) - f_amt

-- Define the ratio calculation
def ratio (a b : ℕ) : ℕ := a / b

-- Lean 4 statement proving the ratio of amounts
theorem ratio_second_shop_to_shirt : 
  ratio (spent_in_second_shop initial_amount spent_on_shirt final_amount) spent_on_shirt = 3 := 
by
  sorry

end ratio_second_shop_to_shirt_l341_341495


namespace triangle_inequality_l341_341209
-- Import necessary libraries

-- Define the problem
theorem triangle_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (α β γ : ℝ) (h_alpha : α = 2 * Real.sqrt (b * c)) (h_beta : β = 2 * Real.sqrt (c * a)) (h_gamma : γ = 2 * Real.sqrt (a * b)) :
  (a / α) + (b / β) + (c / γ) ≥ (3 / 2) :=
by
  sorry

end triangle_inequality_l341_341209


namespace inequality_has_no_solution_l341_341197

theorem inequality_has_no_solution (x : ℝ) : -x^2 + 2*x - 2 > 0 → false :=
by
  sorry

end inequality_has_no_solution_l341_341197


namespace weight_of_green_peppers_l341_341040

-- Definitions for conditions and question
def total_weight : ℝ := 0.6666666667
def is_split_equally (x y : ℝ) : Prop := x = y

-- Theorem statement that needs to be proved
theorem weight_of_green_peppers (g r : ℝ) (h_split : is_split_equally g r) (h_total : g + r = total_weight) :
  g = 0.33333333335 :=
by sorry

end weight_of_green_peppers_l341_341040


namespace sum_even_divisors_of_180_l341_341953

theorem sum_even_divisors_of_180 : 
  let n := 180 in 
  ∑ d in finset.filter (λ x, x % 2 = 0) (finset.divisors n), d = 468 := by
  let n := 180
  have h_factorization : n = 2^2 * 3^2 * 5 := by rfl
  sorry

end sum_even_divisors_of_180_l341_341953


namespace tan_half_angle_l341_341023

-- Define the point (3, -4) and the associated angle α
variables (α : ℝ)

-- Given conditions and definitions
def terminal_side_at_point (x y : ℝ) (α : ℝ) : Prop := 
  ∃ (r : ℝ), r > 0 ∧ (x = r * Real.cos α) ∧ (y = r * Real.sin α)

-- Assertions based on given data
axiom angle_through_point : terminal_side_at_point 3 -4 α

-- Goal: Prove that tan(α / 2) = -1/2
theorem tan_half_angle : 
  terminal_side_at_point 3 -4 α → Real.tan (α / 2) = -1/2 :=
by
  intro h
  sorry

end tan_half_angle_l341_341023


namespace hexagon_side_or_diagonal_le_one_l341_341913

-- Define the type for points in 2D space (R^2)
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a ConvexHexagon structure
structure ConvexHexagon :=
  (A B C D E F : Point)
  -- Add any necessary conditions to ensure it's convex as needed
  -- (This abstraction skips detailed convexity conditions for simplicity)

-- Define the distance formula between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a function to compute the length of the longest diagonal in a hexagon
noncomputable def longest_diagonal (hex : ConvexHexagon) : ℝ :=
max (distance hex.A hex.D)
(max (distance hex.B hex.E)
(max (distance hex.C hex.F)
(max (distance hex.A hex.C)
(max (distance hex.B hex.D)
(max (distance hex.C hex.E)
(max (distance hex.D hex.F)
(max (distance hex.E hex.A) (distance hex.F hex.B))))))))

-- Define a function to check if any side or diagonal length <= 1
noncomputable def has_side_or_diagonal_le_one (hex : ConvexHexagon) : Prop :=
  ∃ (p1 p2 : Point), (p1 = hex.A ∨ p1 = hex.B ∨ p1 = hex.C ∨ p1 = hex.D ∨ p1 = hex.E ∨ p1 = hex.F) ∧ 
                      (p2 = hex.A ∨ p2 = hex.B ∨ p2 = hex.C ∨ p2 = hex.D ∨ p2 = hex.E ∨ p2 = hex.F) ∧
                      distance p1 p2 ≤ 1

-- State the theorem
theorem hexagon_side_or_diagonal_le_one (hex : ConvexHexagon) (h : longest_diagonal hex = 2) : 
  has_side_or_diagonal_le_one hex :=
sorry

end hexagon_side_or_diagonal_le_one_l341_341913


namespace checkerboard_rearrangement_impossible_l341_341505

theorem checkerboard_rearrangement_impossible :
  ∀ (board : Fin 5 → Fin 5 → bool),
    (∀ i j, ∃ x y, (x, y) ≠ (i, j) ∧ (board i j) = (¬board x y)) →
    (∃ (f : Fin 5 × Fin 5 → Fin 5 × Fin 5),
      ∀ p, ((f p).1 = p.1 ∨ (f p).2 = p.2) ∧ (board p.1 p.2 = ¬board (f p).1 (f p).2)) →
      false :=
by
  sorry

end checkerboard_rearrangement_impossible_l341_341505


namespace min_value_inequality_l341_341012

theorem min_value_inequality (x y : ℝ) (h1 : x^2 + y^2 = 3) (h2 : |x| ≠ |y|) :
  ∃ (m : ℝ), m = (1 / (2*x + y)^2 + 4 / (x - 2*y)^2) ∧ m = 3 / 5 :=
by
  sorry

end min_value_inequality_l341_341012


namespace bags_filled_on_saturday_l341_341156

-- Definitions of the conditions
def bags_sat (S : ℕ) := S
def bags_sun := 4
def cans_per_bag := 9
def total_cans := 63

-- The statement to prove
theorem bags_filled_on_saturday (S : ℕ) 
  (h : total_cans = (bags_sat S + bags_sun) * cans_per_bag) : 
  S = 3 :=
by sorry

end bags_filled_on_saturday_l341_341156


namespace median_number_of_moons_is_three_l341_341588

/-- Define the list of the number of moons for each celestial body -/
def number_of_moons : List ℕ := [0, 0, 1, 2, 3, 3, 4, 17, 18, 25]

/-- Statement to prove that the median number of moons per celestial body is 3 -/
theorem median_number_of_moons_is_three : 
  ((number_of_moons.sorted.nth 4).iget + (number_of_moons.sorted.nth 5).iget) / 2 = 3 :=
by 
  sorry

end median_number_of_moons_is_three_l341_341588


namespace part1_part2_l341_341715

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1| + 2 * |x + 2|

theorem part1 : ∀ x : ℝ, f x ≥ 5 := sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, 15 - 2 * f x < a^2 + 9 / (a^2 + 1)) ↔ a ≠ sqrt 2 ∧ a ≠ -sqrt 2 := sorry

end part1_part2_l341_341715


namespace termites_ate_12_black_cells_l341_341900

-- Definitions of the chessboard and the sections that were eaten
def chessboard_8x8 : matrix (fin 8) (fin 8) (bool) :=
λ i j, (i + j) % 2 = 0

def section1 (i j : fin 8) : Prop := i < 3 ∧ j < 3
def section2 (i j : fin 8) : Prop := i > 4 ∧ i < 8 ∧ j > 4 ∧ j < 8

-- Predicate for identifying black cells
def is_black (i j : fin 8) : Prop := chessboard_8x8 i j

-- Defining the total number of black cells eaten by termites
def eaten_black_cells : nat :=
(finset.univ.filter (λ ⟨i, j⟩, (section1 i j ∨ section2 i j) ∧ is_black i j)).card

-- The theorem statement to prove the number of eaten black cells is 12
theorem termites_ate_12_black_cells : 
  eaten_black_cells = 12 := sorry

end termites_ate_12_black_cells_l341_341900


namespace total_additions_in_2_hours_30_minutes_l341_341263

def additions_rate : ℕ := 15000

def time_in_seconds : ℕ := 2 * 3600 + 30 * 60

def total_additions : ℕ := additions_rate * time_in_seconds

theorem total_additions_in_2_hours_30_minutes :
  total_additions = 135000000 :=
by
  -- Non-trivial proof skipped
  sorry

end total_additions_in_2_hours_30_minutes_l341_341263


namespace sally_picked_11_pears_l341_341888

theorem sally_picked_11_pears (total_pears : ℕ) (pears_picked_by_Sara : ℕ) (pears_picked_by_Sally : ℕ) 
    (h1 : total_pears = 56) (h2 : pears_picked_by_Sara = 45) :
    pears_picked_by_Sally = total_pears - pears_picked_by_Sara := by
  sorry

end sally_picked_11_pears_l341_341888


namespace equilateral_triangle_area_perpendiculars_l341_341364

variable {A B C P D E F : Point}
variable {ABC : Triangle}

theorem equilateral_triangle_area_perpendiculars
  (h1 : equilateral_triangle ABC 2)
  (h2 : point_in_triangle P ABC)
  (h3 : perpendicular P (side BC ABC) D)
  (h4 : perpendicular P (side CA ABC) E)
  (h5 : perpendicular P (side AB ABC) F) :
  area (triangle DEF) = sqrt 3 / 2 := sorry

end equilateral_triangle_area_perpendiculars_l341_341364


namespace sum_mod_remainder_l341_341463

theorem sum_mod_remainder : 
  let T := ∑ n in Finset.range 334, (-1) ^ n * Nat.choose 2010 (3 * n)
  in T % 1001 = 486 :=
by
  sorry

end sum_mod_remainder_l341_341463


namespace find_breadth_of_rectangle_l341_341191

theorem find_breadth_of_rectangle
  (L R S : ℝ)
  (h1 : L = (2 / 5) * R)
  (h2 : R = S)
  (h3 : S ^ 2 = 625)
  (A : ℝ := 100)
  (h4 : A = L * B) :
  B = 10 := sorry

end find_breadth_of_rectangle_l341_341191


namespace range_of_values_for_a_l341_341484

open Real

def point_A : Point := (-2, 3)

def point_B (a : ℝ) : Point := (0, a)

def circle_center : Point := (-3, -2)

noncomputable def symmetric_line_distance (a : ℝ) : ℝ :=
  let A := 3 - a
  let B := -2
  let C := 2 * a
  let x_0 := -3
  let y_0 := -2
  abs (A * x_0 + B * y_0 + C) / sqrt (A ^ 2 + B ^ 2)

def valid_range : set ℝ := set.Icc (1 / 3) (3 / 2)

theorem range_of_values_for_a (a : ℝ) : symmetric_line_distance a ≤ 1 → a ∈ valid_range :=
  sorry

end range_of_values_for_a_l341_341484


namespace tangent_line_equation_l341_341030

def f (x : ℝ) : ℝ := x^2

theorem tangent_line_equation :
  let x := (1 : ℝ)
  let y := f x
  ∃ m b : ℝ, m = 2 ∧ b = 1 ∧ (2*x - y - 1 = 0) := by
  sorry

end tangent_line_equation_l341_341030


namespace cos_double_angle_l341_341434

theorem cos_double_angle (α : ℝ) (h : Real.tan α = -3) : Real.cos (2 * α) = -4 / 5 := sorry

end cos_double_angle_l341_341434


namespace sum_of_solutions_sum_of_integers_satisfying_eq_l341_341549

theorem sum_of_solutions (x : ℤ) (h : x^2 = x + 256) : x = 16 ∨ x = -16 := sorry

theorem sum_of_integers_satisfying_eq : ∑ x in finset.filter (λ x, x^2 = x + 256) (finset.Icc -16 16), x = 0 := 
begin
  sorry
end

end sum_of_solutions_sum_of_integers_satisfying_eq_l341_341549


namespace fraction_of_outer_circle_not_covered_l341_341598

-- Define the radii based on the given diameters
def outer_diameter : ℝ := 30
def inner_diameter : ℝ := 24

-- Compute the radii
def outer_radius : ℝ := outer_diameter / 2
def inner_radius : ℝ := inner_diameter / 2

-- Compute the areas
def area_outer : ℝ := Real.pi * outer_radius ^ 2
def area_inner : ℝ := Real.pi * inner_radius ^ 2

-- Compute the area not covered
def area_not_covered : ℝ := area_outer - area_inner

-- Compute the fraction not covered
def fraction_not_covered : ℝ := area_not_covered / area_outer

-- The theorem we want to prove
theorem fraction_of_outer_circle_not_covered :
  fraction_not_covered = 9 / 25 :=
by
  -- All required values and computations are defined above, 
  -- the theorem statement is provided as required without proof.
  sorry

end fraction_of_outer_circle_not_covered_l341_341598


namespace exists_equal_abs_difference_in_permutation_l341_341458

theorem exists_equal_abs_difference_in_permutation (a : Fin 2011 → Fin 2012) 
  (h_permutation : Function.Bijective a) :
  ∃ (j k : Fin 2011), j < k ∧ |(a j).val - j.val| = |(a k).val - k.val| :=
by
  sorry

end exists_equal_abs_difference_in_permutation_l341_341458


namespace evaluate_expression_l341_341689

theorem evaluate_expression : 
  ∀ (x y z : ℚ), 
  z = y - 11 → 
  y = x + 3 → 
  x = 5 → 
  (x + 2) ≠ 0 ∧ (y - 3) ≠ 0 ∧ (z + 7) ≠ 0 →
  (x + 3) / (x + 2) * (y - 2) / (y - 3) * (z + 9) / (z + 7) = 72 / 35 := 
by {
  intros x y z hz hy hx nz,
  sorry
}

end evaluate_expression_l341_341689


namespace imaginary_part_eq_fraction_l341_341536

theorem imaginary_part_eq_fraction :
  complex.im ( (1 : ℂ) + (1 : ℂ) * complex.I / ((4 : ℂ) + (3 : ℂ) * complex.I) ) = 1 / 25 := 
sorry

end imaginary_part_eq_fraction_l341_341536


namespace sin_shifted_angle_l341_341748

theorem sin_shifted_angle (α : ℝ) (h : ∃ x y : ℝ, x = -5 ∧ y = -12 ∧ (x^2 + y^2 = 169) ∧ atan2 y x = α) :
  sin (3 * Real.pi / 2 + α) = 5 / 13 :=
sorry

end sin_shifted_angle_l341_341748


namespace solve_for_a_and_b_l341_341130

noncomputable theory

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x + 14

theorem solve_for_a_and_b (a b : ℝ) (h1 : f a = 1) (h2 : f b = 19) : a + b = -2 :=
by
  sorry

end solve_for_a_and_b_l341_341130


namespace determine_radius_of_circles_l341_341941

noncomputable def radius_of_circles_tangent_to_ellipse : ℝ := sorry

-- The problem statement and conditions
axiom ellipse_equation (x y : ℝ) : 9 * x^2 + 4 * y^2 = 36
axiom circles_tangent_to_each_other (r : ℝ) : (∃ x y : ℝ, (x - r)^2 + y^2 = r^2) 

theorem determine_radius_of_circles (r : ℝ) 
  (h1 : circles_tangent_to_each_other r) 
  (h2 : ∀ x y : ℝ, ellipse_equation x y → (∃ t : ℝ, x = r ∧ y = 0)):
  r = 2 := sorry


end determine_radius_of_circles_l341_341941


namespace limit_equivalence_l341_341336

noncomputable def limit_example : Real :=
  lim (fun x : Real => (3 * x^4 + 2 * x^3 - x^2 + 5 * x + 5) / (x^3 + 1)) (-1)

theorem limit_equivalence : limit_example = -1 / 3 := by
  sorry

end limit_equivalence_l341_341336


namespace find_f_0_abs_l341_341855

noncomputable def f : ℝ → ℝ := sorry -- f is a second-degree polynomial with real coefficients

axiom h1 : ∀ (x : ℝ), x = 1 → |f x| = 9
axiom h2 : ∀ (x : ℝ), x = 2 → |f x| = 9
axiom h3 : ∀ (x : ℝ), x = 3 → |f x| = 9

theorem find_f_0_abs : |f 0| = 9 := sorry

end find_f_0_abs_l341_341855


namespace inscribe_rectangle_in_circle_l341_341082

theorem inscribe_rectangle_in_circle (C : set (ℝ × ℝ)) (hC : ∃ r : ℝ, ∀ p : ℝ × ℝ, p ∈ C ↔ (p.1 ^ 2 + p.2 ^ 2 = r ^ 2))
  (A B : ℝ × ℝ) (hA : A ∈ interior C) (hB : B ∈ interior C) :
  ∃ (rect : set (ℝ × ℝ)), (is_rectangle rect) ∧ (rect ⊆ C) ∧ (A ∈ boundary rect) ∧ (B ∈ boundary rect) :=
by { sorry }

end inscribe_rectangle_in_circle_l341_341082


namespace average_last_four_numbers_l341_341172

theorem average_last_four_numbers (numbers : List ℝ) 
  (h1 : numbers.length = 7)
  (h2 : (numbers.sum / 7) = 62)
  (h3 : (numbers.take 3).sum / 3 = 58) : 
  ((numbers.drop 3).sum / 4) = 65 :=
by
  sorry

end average_last_four_numbers_l341_341172


namespace axis_of_symmetry_l341_341891

def original_func (x : ℝ) : ℝ := sin (x + π / 6)

def shrunken_func (x : ℝ) : ℝ := sin (2 * x + π / 6)

def shifted_func (x : ℝ) : ℝ := sin (2 * x - π / 2)

theorem axis_of_symmetry : shifted_func (-π / 2) = 0 :=
by
  sorry

end axis_of_symmetry_l341_341891


namespace sqrt_a_expansion_l341_341416

theorem sqrt_a_expansion (a : ℝ) (C : ℝ) (h : ((λ (x : ℝ), ((x - (sqrt a / x ^ 2)) ^ 6).coeff 0) = C)) : 
  C = 60 → a = 4 :=
by
  sorry

end sqrt_a_expansion_l341_341416


namespace sqrt_diff_inequality_l341_341361

theorem sqrt_diff_inequality (x : ℝ) (hx : x ≥ -3) : 
  sqrt (x + 5) - sqrt (x + 3) > sqrt (x + 6) - sqrt (x + 4) :=
sorry

end sqrt_diff_inequality_l341_341361


namespace propositions_correct_l341_341387

theorem propositions_correct (k : ℤ) (x a ω : ℝ) (h₁ : ∃ (k : ℤ), 
  (k * π + 5*π/12) ≤ x ∧ x ≤ (k * π + 11*π/12)) 
  (h₂ : ∀ x, ∃ t, y = sin(x + π/3)) 
  (h₃ : a ≤ -2) 
  (h₄ : ω > 0 → ω ≥ (399 / 2) * π) 
  (h₅ : ∀ k, (k * π - π/2) < x ∧ x < (k * π + π/4)) : 
  {2, 3, 4, 5} = { n | n ∈ {1, 2, 3, 4, 5} ∧ True → (n ≠ 1) ∧ (n = 2) ∧ (n = 3) ∧ (n = 4) ∧ (n = 5)} :=
by
  sorry

end propositions_correct_l341_341387


namespace cole_average_speed_l341_341665

noncomputable def average_speed_to_work 
  (round_trip_time : ℝ) 
  (return_speed : ℝ) 
  (time_to_work : ℝ) 
  : ℝ := 
  let distance := return_speed * (round_trip_time - time_to_work)
  in distance / time_to_work

theorem cole_average_speed :
  average_speed_to_work 1 105 (35.0 / 60) = 75 := 
by sorry

end cole_average_speed_l341_341665


namespace Cevas_theorem_l341_341882

theorem Cevas_theorem (A B C A1 B1 C1 O : Type)
  (mA mB mC : ℝ)
  (hA_mass : mA = 1)
  (hB_mass : mB = p)
  (hC_mass : mC = pq)
  (hIntersection : line(A, A1) ∩ line(C, C1) = { O })
  (hCenter_C1 : centerOfMass A (mA) B (mB) = C1)
  (hCenter_A1 : centerOfMass B (mB) C (mC) = A1)
  (hCenter_O : centerOfMass3 A (mA) B (mB) C (mC) = O) : 
  line(B, B1) ∩ { O } ↔ (CB1 / B1A = 1 / (p*q)) := 
sorry

end Cevas_theorem_l341_341882


namespace angle_in_first_quadrant_l341_341901

def angle := -999 - 30 / 60 -- defining the angle as -999°30'
def coterminal (θ : Real) : Real := θ + 3 * 360 -- function to compute a coterminal angle

theorem angle_in_first_quadrant : 
  let θ := coterminal angle
  0 <= θ ∧ θ < 90 :=
by
  -- Exact proof steps would go here, but they are omitted as per instructions.
  sorry

end angle_in_first_quadrant_l341_341901


namespace city_mileage_per_tankful_l341_341993

theorem city_mileage_per_tankful :
  ∀ (T : ℝ), 
  ∃ (city_miles : ℝ),
    (462 = T * (32 + 12)) ∧
    (city_miles = 32 * T) ∧
    (city_miles = 336) :=
by
  sorry

end city_mileage_per_tankful_l341_341993


namespace problem_statement_l341_341388

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x else x^2

theorem problem_statement : f (f (-2)) = 8 := by
  sorry

end problem_statement_l341_341388


namespace triangle_area_from_squares_l341_341727

/-- 
Given three squares with areas 36, 64, and 100, 
prove that the area of the triangle formed by their side lengths is 24.
-/
theorem triangle_area_from_squares :
  let a := 6
  let b := 8
  let c := 10
  let area_a := a ^ 2
  let area_b := b ^ 2
  let area_c := c ^ 2
  (area_a = 36) →
  (area_b = 64) →
  (area_c = 100) →
  (a^2 + b^2 = c^2) →
  (1 / 2 * a * b = 24) :=
by
  intros a b c area_a area_b area_c h1 h2 h3 h4
  exact h4
  sorry

end triangle_area_from_squares_l341_341727


namespace trig_identity_l341_341044

theorem trig_identity (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
by
  sorry

end trig_identity_l341_341044


namespace function_increasing_and_min_value_l341_341851

noncomputable def f (x : ℝ) : ℝ := sorry
def f' (x : ℝ) : ℝ := sorry

theorem function_increasing_and_min_value :
  (∀ x > 0, f'(x) * Real.exp(-x) > 1) →
  (∀ x > 0, f(x) - Real.exp(x) = (f(x) - Real.exp(x))) →
  f (Real.log x) ≥ x + Real.sqrt Real.exp 1 →
  f(1/2) = 2 * Real.sqrt Real.exp 1 →
  (∀ x > 0, f(x) - Real.exp(x) increases on (0, ∞)) ∧ 
  ∃ x, x ≥ Real.sqrt (Real.exp 1) :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end function_increasing_and_min_value_l341_341851


namespace radius_of_inscribed_circle_l341_341517

-- Definitions based on the problem conditions
def sector_OAB_third_circle (r: ℝ) := (r = 6 / 3)
def inscribed_radius : ℝ := 6 * real.sqrt 2 - 6

-- The main theorem statement
theorem radius_of_inscribed_circle (r : ℝ) 
  (sector_OAB_condition : sector_OAB_third_circle 6)
  (inscribed_condition : inscribed_radius = r) :
  r = 6 * real.sqrt 2 - 6 :=
  sorry

end radius_of_inscribed_circle_l341_341517


namespace part_I_part_II_part_III_l341_341754

def f (x : ℝ) : ℝ := (4^x) / (4^x + 2)

theorem part_I (x : ℝ) : f(x) + f(1 - x) = 1 := by
  sorry

theorem part_II (a_n : ℕ → ℝ) (n : ℕ) : 
  a_n n = ∑ k in finset.range (n+1), f ((k:ℝ) / n) → 
  a_n n = (n + 1) / 2 := by
  sorry

theorem part_III (b_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ n, b_n n = 2^(n+1) * ((n + 1) / 2)) : 
  S_n n = finset.sum (finset.range n) b_n → 
  S_n n = n * 2^(n+1) := by
  sorry

end part_I_part_II_part_III_l341_341754


namespace train_passing_time_l341_341585

-- Define the constants: speeds and lengths
def speed1 := 46 -- km/hr
def speed2 := 36 -- km/hr
def length := 50 -- meters

-- Define conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s : ℝ := 5 / 18

-- Define the relative speed in m/s
def relative_speed := (speed1 - speed2) * km_per_hr_to_m_per_s

-- Define the total distance needed to pass in meters
def total_distance := 2 * length -- because both trains are of equal length

-- Define the time it takes for the faster train to pass the slower train
def time_to_pass := total_distance / relative_speed

theorem train_passing_time :
  time_to_pass = 36 := by
  sorry

end train_passing_time_l341_341585


namespace sum_of_coefficients_l341_341417

theorem sum_of_coefficients :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ,
    (1 - 2 * x)^9 = a_9 * x^9 + a_8 * x^8 + a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
    a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 = -2 :=
by
  sorry

end sum_of_coefficients_l341_341417


namespace pieces_info_at_most_two_identical_digits_l341_341798

def num_pieces_of_information_with_at_most_two_positions_as_0110 : Nat :=
  (Nat.choose 4 2 + Nat.choose 4 1 + Nat.choose 4 0)

theorem pieces_info_at_most_two_identical_digits :
  num_pieces_of_information_with_at_most_two_positions_as_0110 = 11 :=
by
  sorry

end pieces_info_at_most_two_identical_digits_l341_341798


namespace find_b_l341_341314

theorem find_b (b : ℝ) (x : ℝ) (h : 5^(x + 8) = 9^x) : b = 9 / 5 :=
  sorry

end find_b_l341_341314


namespace range_of_a_l341_341475

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 1 ∧ y = a + (3 - a)/2 * x) →
  (a ∈ set.Icc (1 / 3) (3 / 2)) :=
by
  sorry

end range_of_a_l341_341475


namespace polynomial_evaluation_l341_341631

theorem polynomial_evaluation (p q r s : ℝ) (f := λ x : ℝ, p * x ^ 3 + q * x ^ 2 + r * x + s)
  (h : f 3 = 4) : 6 * p - 3 * q + r - 2 * s = 60 * p + 15 * q + 7 * r - 8 :=
by {
  -- the proof would go here
  sorry
}

end polynomial_evaluation_l341_341631


namespace distance_CP_eq_2_sqrt_3_l341_341762

noncomputable def find_distance_CP : ℝ := 
  let ρ (θ : ℝ) := 4 * Real.cos θ
  let C := (2:ℝ, 0:ℝ)
  let polar_to_rect (r θ : ℝ) := (r * Real.cos θ, r * Real.sin θ)
  let P := polar_to_rect 2 (2 * Real.pi / 3)
  Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)

theorem distance_CP_eq_2_sqrt_3 : find_distance_CP = 2 * Real.sqrt 3 := by 
  sorry

end distance_CP_eq_2_sqrt_3_l341_341762


namespace problem_statement_l341_341282

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

noncomputable def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x < y → x ∈ s → y ∈ s → f x < f y

noncomputable def f_C (x : ℝ) : ℝ := -2 * Real.log (Real.abs x)

theorem problem_statement :
  is_even_function f_C ∧ is_increasing_on f_C {x | x < 0} :=
by
  sorry

end problem_statement_l341_341282


namespace extremum_m_eq_neg1_monotonic_decreasing_interval_m_l341_341001

noncomputable def f (m : ℝ) (x : ℝ) := (x^2 + m * x + m) * real.exp x

theorem extremum_m_eq_neg1 :
  let f_neg1 := λ x, f (-1) x in
  (∀ x, (x < -2 → deriv f_neg1 x > 0) ∧ (-2 < x ∧ x < 1 → deriv f_neg1 x < 0) ∧ (x > 1 → deriv f_neg1 x > 0)) →
  f_neg1 (-2) = 5 * real.exp (-2) ∧ f_neg1 1 = -real.exp 1 :=
by sorry

theorem monotonic_decreasing_interval_m : 
  (∀ x : ℝ, f (4) x = (x^2 + 6 * x + 8) * real.exp x) ∧ ((quadratic_roots (x^2 + (4 + 2) * x + 8) = (-4, -2)) → ∃ m : ℝ, m = 4) :=
by sorry

def quadratic_roots (a b c : ℝ) : prod ℝ ℝ :=
  if h : b^2 - 4*a*c ≥ 0 then
    let d := (b^2 - 4*a*c).sqrt in
    ((-b + d) / (2*a), (-b - d) / (2*a))
  else
    (0, 0)

end extremum_m_eq_neg1_monotonic_decreasing_interval_m_l341_341001


namespace traveler_distance_l341_341236

theorem traveler_distance 
    (distance_north_south : ℕ) (distance_west_east : ℕ) :
    distance_north_south = 18 - 6 → distance_west_east = 11 - 6 →
    sqrt (distance_north_south^2 + distance_west_east^2) = 13 :=
by
    intros h_ns h_we
    rw [h_ns, h_we]
    dsimp [distance_north_south, distance_west_east]
    norm_num
    simp
    sorry

end traveler_distance_l341_341236


namespace complex_number_is_purely_imaginary_l341_341425

noncomputable def purely_imaginary_complex_number (a b : ℝ) (h : b ≠ 0) : Prop :=
  let z := (a + b * complex.i) / (4 + 3 * complex.i) in
  z.re = 0 ∧ z.im ≠ 0

theorem complex_number_is_purely_imaginary (a b : ℝ) (hb : b ≠ 0) :
  purely_imaginary_complex_number a b hb → a / b = -3 / 4 :=
sorry

end complex_number_is_purely_imaginary_l341_341425


namespace max_value_expression_l341_341225

noncomputable theory

open Real

def expression (t : ℝ) : ℝ := ((3^t - 5*t) * t) / (9^t)

theorem max_value_expression : ∃ t : ℝ, expression(t) = 1 / 20 :=
sorry

end max_value_expression_l341_341225


namespace probability_event_1_eq_half_probability_event_2_not_eq_half_probability_event_3_eq_half_probability_event_4_not_eq_half_l341_341509

-- Define the number of tosses for players A and B
def num_tosses_A : ℕ := 2017
def num_tosses_B : ℕ := 2016

-- Define the random variables for the number of heads for players A and B
def heads_A : ℕ := num_tosses_A // 2 -- Assuming a fair division of heads and tails
def heads_B : ℕ := num_tosses_B // 2

-- Define the events
def event_1 : Prop := heads_A > heads_B
def event_2 : Prop := (num_tosses_A - heads_A) < heads_B
def event_3 : Prop := (num_tosses_A - heads_A) > heads_A
def event_4 : Prop := heads_B = (num_tosses_B - heads_B)

-- The conjectures we need to prove
theorem probability_event_1_eq_half : probability event_1 = 0.5 := sorry
theorem probability_event_2_not_eq_half : probability event_2 ≠ 0.5 := sorry
theorem probability_event_3_eq_half : probability event_3 = 0.5 := sorry
theorem probability_event_4_not_eq_half : probability event_4 ≠ 0.5 := sorry

end probability_event_1_eq_half_probability_event_2_not_eq_half_probability_event_3_eq_half_probability_event_4_not_eq_half_l341_341509


namespace min_value_of_expression_proof_l341_341780

noncomputable def min_value_of_expression : Prop :=
  ∀ x y : ℝ, (x > 0 ∧ y > 0 ∧ x + y = 1) → (xy + 2 / xy) = 33 / 4

theorem min_value_of_expression_proof : min_value_of_expression :=
sorry

end min_value_of_expression_proof_l341_341780


namespace harrys_fish_count_l341_341139

theorem harrys_fish_count : 
  let sam_fish := 7
  let joe_fish := 8 * sam_fish
  let harry_fish := 4 * joe_fish
  harry_fish = 224 :=
by
  sorry

end harrys_fish_count_l341_341139


namespace arithmetic_sequence_sum_l341_341447

-- Given the conditions in the problem
theorem arithmetic_sequence_sum :
  (∃ (a : ℕ → ℕ), 
    (∃ b c : ℕ, 
      polynomial.roots (polynomial.X ^ 2 - polynomial.C 6 * polynomial.X + polynomial.C 8) = {a 3, a 15}) ∧ 
    (∑ i in finset.range 5, a (i + 7)) = 15) :=
sorry

end arithmetic_sequence_sum_l341_341447


namespace Annika_hiked_east_correctly_l341_341651

variables (minutes_per_km_flat : ℝ) 
          (uphill_speed_decrease : ℝ) 
          (downhill_speed_increase : ℝ) 
          (distance_flat_terrain : ℝ) 
          (distance_uphill : ℝ)
          (total_time : ℝ)
          (total_distance_east : ℝ)

def hike_time_flat := minutes_per_km_flat * distance_flat_terrain
def minutes_per_km_uphill := minutes_per_km_flat * (1 + uphill_speed_decrease)
def hike_time_uphill := minutes_per_km_uphill * distance_uphill

def remaining_time := total_time - (hike_time_flat + hike_time_uphill)

def minutes_per_km_downhill := minutes_per_km_flat * (1 - downhill_speed_increase)
def total_hike_time := λ (x : ℝ), (x * minutes_per_km_downhill) + (x * minutes_per_km_uphill) + ((distance_flat_terrain - x) * minutes_per_km_flat)

theorem Annika_hiked_east_correctly :
  minutes_per_km_flat = 12 ∧
  uphill_speed_decrease = 0.20 ∧
  downhill_speed_increase = 0.30 ∧
  distance_flat_terrain = 1.5 ∧
  distance_uphill = 1.25 ∧
  total_time = 51 ∧
  total_distance_east = 2.75 →
  let x := -3 / (22.8 - 12) in
  x < 0 →
  total_distance_east = 2.75 :=
sorry

end Annika_hiked_east_correctly_l341_341651


namespace inscribe_rectangle_in_circle_l341_341081

theorem inscribe_rectangle_in_circle (C : set (ℝ × ℝ)) (hC : ∃ r : ℝ, ∀ p : ℝ × ℝ, p ∈ C ↔ (p.1 ^ 2 + p.2 ^ 2 = r ^ 2))
  (A B : ℝ × ℝ) (hA : A ∈ interior C) (hB : B ∈ interior C) :
  ∃ (rect : set (ℝ × ℝ)), (is_rectangle rect) ∧ (rect ⊆ C) ∧ (A ∈ boundary rect) ∧ (B ∈ boundary rect) :=
by { sorry }

end inscribe_rectangle_in_circle_l341_341081


namespace complex_product_quadrant_l341_341728

def z1 : ℂ := 2 + complex.i
def z2 : ℂ := 1 - complex.i

theorem complex_product_quadrant :
  (z1 * z2).im < 0 ∧ (z1 * z2).re > 0 :=
by sorry

end complex_product_quadrant_l341_341728


namespace sum_of_k_values_l341_341950

theorem sum_of_k_values :
  (∃ k, ∃ x, x^2 - 4*x + 3 = 0 ∧ x^2 - 6*x + k = 0) →
  (perfect_values = {5, 9}) →
  (∑ i in perfect_values, i = 14) := by
  sorry

end sum_of_k_values_l341_341950


namespace compare_flavors_l341_341614

def flavor_ratings_A := [7, 9, 8, 6, 10]
def flavor_ratings_B := [5, 6, 10, 10, 9]

def mean (l : List ℕ) : ℚ := l.sum / l.length

def variance (l : List ℕ) : ℚ :=
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem compare_flavors : 
  mean flavor_ratings_A = mean flavor_ratings_B ∧ variance flavor_ratings_A < variance flavor_ratings_B := by
  sorry

end compare_flavors_l341_341614


namespace magnitude_of_difference_l341_341004

variables (m : ℝ)
def vector_a := (m, 1)
def vector_b := (2, -6)
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_difference :
  perpendicular (vector_a m) vector_b →
  magnitude (vector_a m.1 - vector_b) = 5 * real.sqrt 2 :=
sorry

end magnitude_of_difference_l341_341004


namespace christopher_stroll_time_l341_341663

variable (distance : ℝ) (speed : ℝ)

def time_stroll (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem christopher_stroll_time:
  distance = 5 → speed = 4 → time_stroll distance speed = 1.25 :=
by
  intros h_distance h_speed
  rw [h_distance, h_speed]
  rfl

end christopher_stroll_time_l341_341663


namespace max_min_f_inequality_g_l341_341000

noncomputable def f (x : ℝ) := x * Real.log x - x

theorem max_min_f :
  let a := (1 / Real.e : ℝ)
  let b := Real.e
  let f_min := -1
  let f_max := 0
  f a = -2 / Real.e ∧ f b = 0 ∧ f 1 = -1 :=
by {
  have h := let a := 1 / Real.e, 
            in calc
              f a = -2 / Real.e : by sorry
  exact h,
}

noncomputable def g (x : ℝ) := 1 / Real.exp x - 3 / (2 * x) - Real.log x

theorem inequality_g :
  ∀ x ∈ Set.Icc (1 / Real.e) Real.e, g(x) < -1 :=
by {
  assume x hx,
  have h := by {
    intros,
    calc g(1 / Real.e) < -1 : by sorry
  },
  exact h,
}

end max_min_f_inequality_g_l341_341000


namespace sum_of_integers_square_256_sum_of_integers_solution_l341_341558

theorem sum_of_integers_square_256 (x : ℤ) (h : x^2 = x + 256) : x = 16 ∨ x = -16 :=
by
  sorry

theorem sum_of_integers_solution : (16 + (-16) = 0) : Prop :=
by
  exact rfl

end sum_of_integers_square_256_sum_of_integers_solution_l341_341558


namespace intersection_M_N_l341_341400

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | |x| > 1}

theorem intersection_M_N : M ∩ N = {2} := by
  sorry

end intersection_M_N_l341_341400


namespace cost_price_of_apple_l341_341268

theorem cost_price_of_apple (C : ℚ) (h1 : 19 = 5/6 * C) : C = 22.8 := by
  sorry

end cost_price_of_apple_l341_341268


namespace op_4_6_l341_341772

-- Define the operation @ in Lean
def op (a b : ℕ) : ℤ := 2 * (a : ℤ)^2 - 2 * (b : ℤ)^2

-- State the theorem to prove
theorem op_4_6 : op 4 6 = -40 :=
by sorry

end op_4_6_l341_341772


namespace power_fraction_example_l341_341333

theorem power_fraction_example : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := 
by
  sorry

end power_fraction_example_l341_341333


namespace range_of_a_l341_341717

def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≥ 2 then x / (2 * x^2 + 8)
else (1 / 2) ^ |x - a|

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Ici 2, ∃! x2 ∈ set.Iio 2, f x1 a = f x2 a) ↔ -1 ≤ a ∧ a < 5 :=
sorry

end range_of_a_l341_341717


namespace sum_of_divisors_of_prime_47_l341_341962

theorem sum_of_divisors_of_prime_47 : ∀ n : ℕ, prime n → n = 47 → (∑ d in (finset.filter (λ x, n % x = 0) (finset.range (n + 1))), d) = 48 := 
by {
  intros n prime_n n_is_47,
  sorry -- Proof is omitted
}

end sum_of_divisors_of_prime_47_l341_341962


namespace trig_eq_solutions_l341_341239

theorem trig_eq_solutions (x : ℝ) (k n : ℤ) :
  (cos (x / 2) ≠ 0 ∧ sin (x / 2) ≠ 0) ∧ 
    (tan (x / 2)^2 + sin (x / 2)^2 * tan (x / 2) + cos (x / 2)^2 * cot (x / 2) + cot (x / 2)^2 + sin x = 4) → 
    (x = (-1)^(k + 1) * arcsin (2/3) + k * real.pi ∨ x = real.pi / 2 + 2 * n * real.pi) := 
sorry

end trig_eq_solutions_l341_341239


namespace sugar_used_in_two_minutes_l341_341574

-- Definitions according to conditions
def sugar_per_bar : ℝ := 1.5
def bars_per_minute : ℝ := 36
def minutes : ℝ := 2

-- Theorem statement
theorem sugar_used_in_two_minutes : bars_per_minute * sugar_per_bar * minutes = 108 :=
by
  -- We add sorry here to complete the proof later.
  sorry

end sugar_used_in_two_minutes_l341_341574


namespace selection_ways_l341_341928

def Worker : Type := {A : String} ⊕ {B : String} ⊕ {C : String}

def num_ways_selection : Nat :=
  have available_workers : List Worker := [Worker.inl ⟨"A"⟩, Worker.inr (Sum.inl ⟨"B"⟩), Worker.inr (Sum.inr ⟨"C"⟩)]
  available_workers.length * (available_workers.length - 1)

theorem selection_ways : num_ways_selection = 6 :=
  by
  sorry

end selection_ways_l341_341928


namespace evaluate_expression_l341_341688

theorem evaluate_expression :
  (2^4 + 2^3 * 2) / (2^(-1) * 2^2 + 2^(-3) + 2^(-5)) = 1024 / 69 :=
by
  sorry

end evaluate_expression_l341_341688


namespace can_arrange_20_coins_l341_341938

theorem can_arrange_20_coins : 
  ∃ arrange : ℕ × ℕ → ℕ, 
    (∀ i j, 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4 → arrange (i, j) = 1) ∧ 
    (∀ i j, (1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4) → arrange (i, j) = 1 + if (i, j) ∈ 
      [(1, 1), (2, 3), (3, 4), (4, 2)] then 1 else 0) := 
begin
  sorry
end

end can_arrange_20_coins_l341_341938


namespace value_range_l341_341564

noncomputable def f (x : ℝ) : ℝ := 1 - 4 * x - 2 * x^2

theorem value_range (x : ℝ) (h : 1 < x) :
  ∃ y : ℝ, y = f x ∧ y ∈ Iio (-5) :=
by
  sorry

end value_range_l341_341564


namespace particle_at_point3_l341_341883

def reflect_screen_prob (a_n b_n c_n d_n : ℕ → ℝ) : Prop :=
  (∀ n, a_n = b_(n-1) * 0.5) ∧ 
  (∀ n, b_n = a_(n-1) * 1 + c_(n-1) * 0.5) ∧ 
  (∀ n, c_n = b_(n-1) * 0.5) ∧
  (∀ n, d_n = d_(n-1) * 1 + c_(n-1) * 0.5) ∧
  (a_0 = 0) ∧ (b_0 = 1) ∧ (c_0 = 0) ∧ (d_0 = 0)

theorem particle_at_point3 (a_n b_n c_n d_n : ℕ → ℝ) (n : ℕ) (h : reflect_screen_prob a_n b_n c_n d_n) :
  ∃ (d : ℝ), d = d_n := 
sorry

end particle_at_point3_l341_341883


namespace problem1_problem2_l341_341662

-- Proof problem for the first calculation
theorem problem1 : sqrt 6 * sqrt (2 / 3) / sqrt 2 = sqrt 2 := 
by sorry

-- Proof problem for the second calculation
theorem problem2 : (sqrt 2 + sqrt 5) ^ 2 - (sqrt 2 + sqrt 5) * (sqrt 2 - sqrt 5) = 10 + 2 * sqrt 10 := 
by sorry

end problem1_problem2_l341_341662


namespace walnut_trees_currently_in_park_l341_341208

-- Definitions from the conditions
def total_trees : ℕ := 77
def trees_to_be_planted : ℕ := 44

-- Statement to prove: number of current trees = 33
theorem walnut_trees_currently_in_park : total_trees - trees_to_be_planted = 33 :=
by
  sorry

end walnut_trees_currently_in_park_l341_341208


namespace sum_of_reciprocals_l341_341725

-- Define the relevant conditions and proof statement
theorem sum_of_reciprocals (n : ℕ) (h : n > 2) :
  (∑ (a b : ℕ) in {p : ℕ × ℕ | p.1 < p.2 ∧ p.2 ≤ n ∧ p.1 + p.2 > n ∧ Nat.gcd p.1 p.2 = 1}.to_finset, (1 : ℚ) / (a * b)) = 1 / 2 :=
sorry

end sum_of_reciprocals_l341_341725


namespace petya_no_win_implies_draw_or_lost_l341_341202

noncomputable def petya_cannot_win (n : ℕ) (h : n ≥ 3) : Prop :=
  ∀ (Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ),
    ∃ m : ℕ, Petya_strategy m ≠ Vasya_strategy m

theorem petya_no_win_implies_draw_or_lost (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ, 
    (∀ m : ℕ, Petya_strategy m = Vasya_strategy m) :=
by {
  sorry
}

end petya_no_win_implies_draw_or_lost_l341_341202


namespace probability_all_six_draws_white_l341_341612

theorem probability_all_six_draws_white :
  let total_balls := 14
  let white_balls := 7
  let single_draw_white_probability := (white_balls : ℚ) / total_balls
  (single_draw_white_probability ^ 6 = (1 : ℚ) / 64) :=
by
  sorry

end probability_all_six_draws_white_l341_341612


namespace number_of_triangles_and_squares_l341_341203

theorem number_of_triangles_and_squares (x y : ℕ) (h1 : x + y = 13) (h2 : 3 * x + 4 * y = 47) : 
  x = 5 ∧ y = 8 :=
by
  sorry

end number_of_triangles_and_squares_l341_341203


namespace divisors_problem_l341_341632

theorem divisors_problem (n : ℕ) (k : ℕ) (m : ℕ)
  (h1 : n = 5^k * m)
  (h2 : nat.coprime 5 m)
  (h3 : nat.divisors_count n = 72)
  (h4 : nat.divisors_count (5 * n) = 90) :
  k = 3 :=
sorry

end divisors_problem_l341_341632


namespace light_bulb_probabilities_l341_341566

variables (totalBulbs defectiveBulbs nonDefectiveBulbs : ℕ)
variables (totalOutcomes defectiveOutcomes mixedOutcomes : ℚ)

-- Define the conditions
def bulbs_conditions : Prop :=
  totalBulbs = 6 ∧ defectiveBulbs = 2 ∧ nonDefectiveBulbs = 4

-- Probability that both bulbs are defective
lemma prob_both_defective (H : bulbs_conditions) : 
  (defectiveOutcomes = (defectiveBulbs * defectiveBulbs) / (totalBulbs * totalBulbs)) :=
sorry

-- Probability that one bulb is defective and the other is non-defective
lemma prob_one_def_one_nondef (H : bulbs_conditions) : 
  (mixedOutcomes = (defectiveBulbs * nonDefectiveBulbs + nonDefectiveBulbs * defectiveBulbs) / (totalBulbs * totalBulbs)) :=
sorry

-- Probability that at least one bulb is non-defective
lemma prob_at_least_one_nondefective (H : bulbs_conditions) :
  (totalOutcomes = 1 - (defectiveOutcomes / (totalBulbs * totalBulbs))) :=
sorry

-- Ensure proofs for the required probabilities
theorem light_bulb_probabilities (H : bulbs_conditions) :
  (∃ p1 p2 p3 : ℚ,
    p1 = (defectiveBulbs * defectiveBulbs) / (totalBulbs * totalBulbs) ∧
    p1 = 1 / 9 ∧
    p2 = (defectiveBulbs * nonDefectiveBulbs + nonDefectiveBulbs * defectiveBulbs) / (totalBulbs * totalBulbs) ∧
    p2 = 4 / 9 ∧
    p3 = 1 - p1 ∧
    p3 = 8 / 9) :=
sorry

end light_bulb_probabilities_l341_341566


namespace problem_statement_l341_341071

open Real

-- Define the parametric equations of Line l.
def line_l_parametric (t : Real) : Real × Real :=
  (t * cos (2 * π / 3), 4 + t * sin (2 * π / 3))

-- Define the polar equation of Curve C.
def curve_C_polar (θ : Real) : Real :=
  4

-- Define the Cartesian equation of Curve C.
def curve_C_cartesian (x y : Real) : Prop :=
  x^2 + y^2 = 16

-- Define the standard equation of line l.
def line_l_standard (x y : Real) : Prop :=
  √3 * x + y - 4 = 0

-- Prove the statements.
theorem problem_statement :
  (∀ t : Real, let (x, y) := line_l_parametric t in line_l_standard x y) ∧
  (∀ (ρ θ : Real), (ρ = curve_C_polar θ) → curve_C_cartesian (ρ * cos θ) (ρ * sin θ)) ∧
  (∃ A B : (ℝ × ℝ), curve_C_cartesian A.1 A.2 ∧ curve_C_cartesian B.1 B.2 ∧ line_l_standard A.1 A.2 ∧ line_l_standard B.1 B.2 ∧
    ∃ θ : Real, 0 < θ ∧ θ < π ∧ 2 * θ = ∠ (0, 0) A B ∧ θ = π / 3 ∧ 2 * θ = 2 * π / 3) :=
by
  sorry

-- Angle calculation.
def ∠ (O A B : (ℝ × ℝ)) : Real :=
  acos ((A.1 * B.1 + A.2 * B.2) / (sqrt (A.1^2 + A.2^2) * sqrt (B.1^2 + B.2^2)))


end problem_statement_l341_341071


namespace area_of_square_l341_341147

theorem area_of_square {PQRS : Type*} [metric_space PQRS] [has_mk_square PQRS]
  (P Q R S M N : PQRS) 
  (h1 : is_square P Q R S) 
  (h2 : lies_on_side M P Q) 
  (h3 : lies_on_side N R S) 
  (h4 : dist P M = 20) 
  (h5 : dist M N = 20) 
  (h6 : dist N R = 20) : 
  square_area P Q R S = 3600 :=
sorry

end area_of_square_l341_341147


namespace min_expression_value_l341_341112

def distinct_elements (s : Set ℤ) : Prop := s = {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_expression_value :
  ∃ (p q r s t u v w : ℤ),
    distinct_elements {p, q, r, s, t, u, v, w} ∧
    (p + q + r + s) ≥ 5 ∧
    (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
     q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
     r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
     s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
     t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
     u ≠ v ∧ u ≠ w ∧
     v ≠ w) →
    (p + q + r + s)^2 + (t + u + v + w)^2 = 26 :=
sorry

end min_expression_value_l341_341112


namespace sugar_used_in_two_minutes_l341_341576

-- Definitions according to conditions
def sugar_per_bar : ℝ := 1.5
def bars_per_minute : ℝ := 36
def minutes : ℝ := 2

-- Theorem statement
theorem sugar_used_in_two_minutes : bars_per_minute * sugar_per_bar * minutes = 108 :=
by
  -- We add sorry here to complete the proof later.
  sorry

end sugar_used_in_two_minutes_l341_341576


namespace sugar_used_in_two_minutes_l341_341582

-- Define constants for problem conditions
def sugar_per_bar : Float := 1.5
def bars_per_minute : Nat := 36
def minutes : Nat := 2

-- Define the total sugar used in two minutes
def total_sugar_used : Float := (bars_per_minute * sugar_per_bar) * minutes

-- State the theorem and its proof
theorem sugar_used_in_two_minutes : total_sugar_used = 108 := by
  sorry

end sugar_used_in_two_minutes_l341_341582


namespace pizza_area_percentage_increase_l341_341169

theorem pizza_area_percentage_increase :
  let r1 := 6
  let r2 := 4
  let A1 := Real.pi * r1^2
  let A2 := Real.pi * r2^2
  let deltaA := A1 - A2
  let N := (deltaA / A2) * 100
  N = 125 := by
  sorry

end pizza_area_percentage_increase_l341_341169


namespace find_5x_plus_7y_l341_341812

variables (A B C D E F G : Type) [AffineSpace A]
variables (AB AD BE BD : A → A)
variables (λ x y : ℝ)

-- Definitions for the conditions
def isParallelogram (ABCD : A → A) : Prop := 
  ∃ (A B C D : A),
  (ABCD B = AB + AD) ∧
  (ABCD C = AB - AD) ∧
  (ABCD D = BE) ∧ 
  (ABCD E = BD + AB)

def reflection (B D E : A) : Prop :=
  ∃ (B D E : A), 
  BE = 2 * BD

def vector_relation(AF FC : A) : Prop := 
  ∃ (AF FC : A),
  AF = 3 * FC

def segment_point (G E F : A) : Prop :=
  ∃ (λ : ℝ), λ ∈ set.Icc 0 1 ∧ EG = λ * EF

-- Lean 4 statement to prove the desired property
theorem find_5x_plus_7y (h_parallelogram: isParallelogram A B C D)
  (h_reflection: reflection B D E)
  (h_relation: vector_relation A F C)
  (h_segment: segment_point G E F)
  (h_AG: ∃ (x y : ℝ), G = x * AB + y * AD) :
  5 * x + 7 * y = 9 :=
sorry

end find_5x_plus_7y_l341_341812


namespace problem1_problem2_monotonic_increasing_problem2_bounded_above_l341_341856

-- Definition and monotonicity of the function f
def f (x : ℝ) : ℝ := x * Real.log (1 + 1 / x)

theorem problem1 :
  ∀ x : ℝ, (x < -1 ∨ x > 0) → monotone_on f (set.Ioi x) :=
sorry

-- Definition and properties of the sequence a_n
def a_n (n : ℕ) : ℝ := (1 + 1 / n) ^ n

theorem problem2_monotonic_increasing (n : ℕ) (h : n ≠ 0) :
  a_n n < a_n (n + 1) :=
sorry

theorem problem2_bounded_above (n : ℕ) (h : n ≠ 0) :
  a_n n < Real.exp 1 :=
sorry

end problem1_problem2_monotonic_increasing_problem2_bounded_above_l341_341856


namespace sum_of_integers_squared_greater_sum_of_integers_satisfying_condition_l341_341553

theorem sum_of_integers_squared_greater (x : ℤ) (hx : x ^ 2 = 256 + x) : x = 16 ∨ x = -16 :=
begin
  have : x ^ 2 - x - 256 = 0,
  { rw ← hx, ring },
  apply quadratic_eq_zero_iff.mpr,
  use [16, -16],
  split;
  linarith,
end

theorem sum_of_integers_satisfying_condition : ∑ x in ({16, -16} : finset ℤ), x = 0 :=
by simp

end sum_of_integers_squared_greater_sum_of_integers_satisfying_condition_l341_341553


namespace max_acute_angles_convex_octagon_l341_341945

def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

theorem max_acute_angles_convex_octagon (angles : Fin 8 → ℝ) (h_convex : ∀ i, 0 < angles i ∧ angles i < 180) (h_sum : ∑ i, angles i = sum_of_interior_angles 8) : 
  ∃ (n : ℕ), n = 4 ∧ ∀ i, i < n → angles i < 90 ∧ ∀ j, j ≥ n → angles j ≥ 90 :=
sorry

end max_acute_angles_convex_octagon_l341_341945


namespace collinear_points_addition_l341_341793

variables (a b : ℝ)

-- Points definitions
def P1 := (2, a, b)
def P2 := (a, 3, b)
def P3 := (a, b, 4)

-- Collinearity condition definition
def collinear (P Q R : ℝ × ℝ × ℝ) : Prop :=
∃ (k : ℝ), P.1 = k * Q.1 + (1 - k) * R.1 ∧ P.2 = k * Q.2 + (1 - k) * R.2 ∧ P.3 = k * Q.3 + (1 - k) * R.3

-- The theorem we need to prove
theorem collinear_points_addition : collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 :=
by sorry

end collinear_points_addition_l341_341793


namespace three_inv_mod_199_l341_341328

theorem three_inv_mod_199 : ∃ x : ℤ, 3 * x ≡ 1 [MOD 199] ∧ (0 ≤ x ∧ x < 199) :=
by
  use 133
  split
  · show 3 * 133 ≡ 1 [MOD 199]
    sorry
  · split
    · show 0 ≤ 133
      linarith
    · show 133 < 199
      linarith

end three_inv_mod_199_l341_341328


namespace terminal_side_quadrant_l341_341740

def α : Real := - (2 * Real.pi / 3)

theorem terminal_side_quadrant :
  ∃ quadrant : ℕ, quadrant = 3 ∧ (α + 2 * Real.pi * 1) / Real.pi = 4 / 3 := 
sorry

end terminal_side_quadrant_l341_341740


namespace route_down_distance_l341_341622

theorem route_down_distance
  (rate_up : ℕ)
  (time_up : ℕ)
  (rate_down_rate_factor : ℚ)
  (time_down : ℕ)
  (h1 : rate_up = 4)
  (h2 : time_up = 2)
  (h3 : rate_down_rate_factor = (3 / 2))
  (h4 : time_down = time_up) :
  rate_down_rate_factor * rate_up * time_up = 12 := 
by
  rw [h1, h2, h3]
  sorry

end route_down_distance_l341_341622


namespace no_integer_solutions_to_equation_l341_341680

theorem no_integer_solutions_to_equation : ∀ x y : ℤ, 2^(2*x) - 3^(2*y) ≠ 85 := by
  sorry

end no_integer_solutions_to_equation_l341_341680


namespace dan_marble_counts_l341_341675

theorem dan_marble_counts :
  ∀ (green_start green_taken green_returned : ℕ)
    (violet_start violet_taken violet_returned : ℕ)
    (blue_start blue_taken blue_returned : ℕ),
  green_start = 32 → violet_start = 38 → blue_start = 46 →
  green_taken = 23 → violet_taken = 15 → blue_taken = 31 →
  green_returned = 10 → violet_returned = 8 → blue_returned = 17 →
  let final_green := green_start - green_taken + green_returned,
      final_violet := violet_start - violet_taken + violet_returned,
      final_blue := blue_start - blue_taken + blue_returned
  in final_green = 19 ∧ final_violet = 31 ∧ final_blue = 32 :=
by intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _;
   simp;
   sorry

end dan_marble_counts_l341_341675


namespace ball_arrangement_divisibility_l341_341251

theorem ball_arrangement_divisibility :
  ∀ (n : ℕ), (∀ (i : ℕ), i < n → (∃ j k l m : ℕ, j < k ∧ k < l ∧ l < m ∧ m < n ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ j
    ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m)) →
  ¬((n = 2021) ∨ (n = 2022) ∨ (n = 2023) ∨ (n = 2024)) :=
sorry

end ball_arrangement_divisibility_l341_341251


namespace smallest_winning_N_and_digit_sum_l341_341293

-- Definitions of operations
def B (x : ℕ) : ℕ := 3 * x
def S (x : ℕ) : ℕ := x + 100

/-- The main theorem confirming the smallest winning number and sum of its digits -/
theorem smallest_winning_N_and_digit_sum :
  ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 999 ∧ (900 ≤ 9 * N + 400 ∧ 9 * N + 400 < 1000) ∧ (N = 56) ∧ (5 + 6 = 11) :=
by {
  -- Proof skipped
  sorry
}

end smallest_winning_N_and_digit_sum_l341_341293


namespace ball_reaches_height_less_than_one_l341_341611

noncomputable def find_bounces : ℕ :=
  Nat.find (λ k => 500 * (1 / 3 : ℝ) ^ k < 1)

theorem ball_reaches_height_less_than_one :
  find_bounces = 6 :=
by
  sorry

end ball_reaches_height_less_than_one_l341_341611


namespace length_of_third_side_l341_341149

theorem length_of_third_side (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 12) (h2 : c = 18) (h3 : B = 2 * C) :
  ∃ a, a = 15 :=
by {
  sorry
}

end length_of_third_side_l341_341149


namespace sqrt_equality_condition_l341_341037

theorem sqrt_equality_condition {a b c : ℝ} :
  sqrt (4 * a^2 + 9 * b^2) = 2 * a + 3 * b + c ↔ (12 * a * b + 4 * a * c + 6 * b * c + c^2 = 0 ∧ 2 * a + 3 * b + c ≥ 0) :=
begin
  sorry
end

end sqrt_equality_condition_l341_341037


namespace deg_q_l341_341160

-- Define the polynomials and their degrees
variables (p q : Polynomial ℝ) -- Generic polynomials p and q
variable (i : Polynomial ℝ := p.comp(q)^2 - q^3) -- Define i(x) = p(q(x))^2 - q(x)^3

-- State the conditions
axiom deg_p : p.degree = 4
axiom deg_i : i.degree = 12

-- Define the statement to prove
theorem deg_q : q.degree = 4 :=
by
  sorry -- Proof is omitted as per instructions

end deg_q_l341_341160


namespace compute_sum_squares_l341_341667

theorem compute_sum_squares :
  let N : ℕ := ∑ k in range 20, ((120 - 3 * k) ^ 2 + (119 - 3 * k) ^ 2 + (118 - 3 * k) ^ 2) - ((117 - 3 * k) ^ 2 + (116 - 3 * k) ^ 2 + (115 - 3 * k) ^ 2)
  N = 2250 :=
by
  let N := ∑ k in range 20, ((120 - 3 * k) ^ 2 + (119 - 3 * k) ^ 2 + (118 - 3 * k) ^ 2) - ((117 - 3 * k) ^ 2 + (116 - 3 * k) ^ 2 + (115 - 3 * k) ^ 2)
  have hN : N = 2250 := sorry
  exact hN

end compute_sum_squares_l341_341667


namespace gcd_204_85_l341_341182

theorem gcd_204_85 : Nat.gcd 204 85 = 17 :=
sorry

end gcd_204_85_l341_341182


namespace pyramid_volume_l341_341514

-- Define the given geometry conditions
structure RegularOctagon (A B C D E F G H : Point ℝ) :=
(side_length : ℝ)
(is_regular : side_length = 5)

structure EquilateralTriangle (P A B : Point ℝ) :=
(side_length : ℝ)
(is_equilateral : side_length = 10)

-- Define the coordinates for the points of the pyramid
variable (P A B C D E F G H : Point ℝ)

-- Define theorems for properties we use in the conditions
axiom regular_octagon (h : RegularOctagon A B C D E F G H)
axiom equilateral_triangle (h₁ : EquilateralTriangle P A B)

-- Define the statement for the volume of the pyramid
theorem pyramid_volume (h₁ : RegularOctagon A B C D E F G H) (h₂ : EquilateralTriangle P A B) :
  volume_of_pyramid P A B C D E F G H = 250 :=
sorry

end pyramid_volume_l341_341514


namespace white_square_area_l341_341135

def edge_length : ℝ := 10
def total_green_paint : ℝ := 300
def number_of_faces : ℝ := 6
def total_face_area : ℝ := number_of_faces * (edge_length * edge_length)
def area_each_face_green_paint : ℝ := total_green_paint / number_of_faces
def total_area_each_face : ℝ := edge_length * edge_length

theorem white_square_area :
  total_area_each_face - area_each_face_green_paint = 50 := by
  sorry

end white_square_area_l341_341135


namespace solve_for_x_l341_341925

-- Define the operation *
def op (a b : ℝ) : ℝ := 2 * a - b

-- The theorem statement
theorem solve_for_x :
  (∃ x : ℝ, op x (op 1 3) = 2) ∧ (∀ x, op x -1 = 2)
  → x = 1/2 := by
  sorry

end solve_for_x_l341_341925


namespace real_part_of_inverse_l341_341121

def z : ℂ := sorry

theorem real_part_of_inverse (hz: |z| = 2 ∧ im z ≠ 0) : real_part (1 / (2 - z)) = 1 / 2 := 
by
  sorry

end real_part_of_inverse_l341_341121


namespace part1_part2_l341_341391

noncomputable def f (x : ℝ) : ℝ :=
  |2 * x - 4| + |x + 1|

def g (x : ℝ) : ℝ :=
  f(x) - |x - 2|

theorem part1 (x : ℝ) : f(x) ≥ 4 ↔ (x ≤ 1 ∨ x ≥ 7/3) :=
  sorry

theorem part2 (a b c : ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h_sum : a + b + c = 3) : 
  (1/a + 1/b + 1/c) ≥ 3 :=
  sorry

end part1_part2_l341_341391


namespace seqs_continue_indefinitely_no_step_sums_to_zero_l341_341563

noncomputable def x_seq : ℕ → ℝ
| 0 := 2
| (n + 1) := 2 * (x_seq n) / ((x_seq n) ^ 2 - 1)

noncomputable def y_seq : ℕ → ℝ
| n := 4  -- Note: y_0 is not defined, we'll use y_1 as base case here
| (n + 1) := 2 * (y_seq n) / ((y_seq n) ^ 2 - 1)

noncomputable def z_seq : ℕ → ℝ
| n := 6 / 7  -- Note: similarly, use z_1 as base case
| (n + 1) := 2 * (z_seq n) / ((z_seq n) ^ 2 - 1)

theorem seqs_continue_indefinitely :
  ∀ n, (x_seq n ≠ 1 ∧ x_seq n ≠ -1) ∧ (y_seq n ≠ 1 ∧ y_seq n ≠ -1) ∧ (z_seq n ≠ 1 ∧ z_seq n ≠ -1) := 
by sorry

theorem no_step_sums_to_zero :
  ∀ n, x_seq n + y_seq n + z_seq n ≠ 0 := 
by sorry

end seqs_continue_indefinitely_no_step_sums_to_zero_l341_341563


namespace range_of_values_for_a_l341_341483

open Real

def point_A : Point := (-2, 3)

def point_B (a : ℝ) : Point := (0, a)

def circle_center : Point := (-3, -2)

noncomputable def symmetric_line_distance (a : ℝ) : ℝ :=
  let A := 3 - a
  let B := -2
  let C := 2 * a
  let x_0 := -3
  let y_0 := -2
  abs (A * x_0 + B * y_0 + C) / sqrt (A ^ 2 + B ^ 2)

def valid_range : set ℝ := set.Icc (1 / 3) (3 / 2)

theorem range_of_values_for_a (a : ℝ) : symmetric_line_distance a ≤ 1 → a ∈ valid_range :=
  sorry

end range_of_values_for_a_l341_341483


namespace problem_statement_l341_341969

def is_opposite (a b : ℤ) := a + b = 0

theorem problem_statement :
  (¬ is_opposite 2 (1/2 : ℚ)) ∧
  (is_opposite (-1 ^ 2022 : ℤ) 1) ∧
  (¬ is_opposite 1 (-1 ^ 2022 : ℤ)) ∧
  (¬ is_opposite 2 (|-2|)) :=
by
  sorry

end problem_statement_l341_341969


namespace range_of_a_l341_341477

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 1 ∧ y = a + (3 - a)/2 * x) →
  (a ∈ set.Icc (1 / 3) (3 / 2)) :=
by
  sorry

end range_of_a_l341_341477


namespace find_general_formula_prove_inequality_l341_341817

-- Define the sequence condition
def sequence_condition (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (finset.range n).sum (λ k, a k * (1 / (n + 1))) = n^2 + n

-- Define the general formula
def general_formula (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = 2 * n * (n + 1)

-- The inequality to prove
def inequality (a : ℕ → ℕ) : Prop :=
  (finset.range n).sum (λ k, k / ((k + 2) * a k)) < 1 / 4

-- Theorems to prove
theorem find_general_formula : ∃ a : ℕ → ℕ, general_formula a :=
begin
  sorry
end

theorem prove_inequality (a : ℕ → ℕ) (h : general_formula a) : inequality a :=
begin
  sorry
end

end find_general_formula_prove_inequality_l341_341817


namespace sum_of_integers_square_256_sum_of_integers_solution_l341_341561

theorem sum_of_integers_square_256 (x : ℤ) (h : x^2 = x + 256) : x = 16 ∨ x = -16 :=
by
  sorry

theorem sum_of_integers_solution : (16 + (-16) = 0) : Prop :=
by
  exact rfl

end sum_of_integers_square_256_sum_of_integers_solution_l341_341561


namespace correct_average_l341_341602

theorem correct_average (S' : ℝ) (a a' b b' c c' : ℝ) (n : ℕ) 
  (incorrect_avg : S' / n = 22) 
  (a_eq : a = 52) (a'_eq : a' = 32)
  (b_eq : b = 47) (b'_eq : b' = 27) 
  (c_eq : c = 68) (c'_eq : c' = 45)
  (n_eq : n = 12) 
  : ((S' - (a' + b' + c') + (a + b + c)) / 12 = 27.25) := 
by
  sorry

end correct_average_l341_341602


namespace sugar_used_in_two_minutes_l341_341581

-- Define constants for problem conditions
def sugar_per_bar : Float := 1.5
def bars_per_minute : Nat := 36
def minutes : Nat := 2

-- Define the total sugar used in two minutes
def total_sugar_used : Float := (bars_per_minute * sugar_per_bar) * minutes

-- State the theorem and its proof
theorem sugar_used_in_two_minutes : total_sugar_used = 108 := by
  sorry

end sugar_used_in_two_minutes_l341_341581


namespace sugar_needed_in_two_minutes_l341_341578

-- Let a be the amount of sugar needed per chocolate bar.
def sugar_per_chocolate_bar : ℝ := 1.5

-- Let b be the number of chocolate bars produced per minute.
def chocolate_bars_per_minute : ℕ := 36

-- Let t be the time in minutes.
def time_in_minutes : ℕ := 2

theorem sugar_needed_in_two_minutes : 
  let sugar_in_one_minute := chocolate_bars_per_minute * sugar_per_chocolate_bar
  let total_sugar := sugar_in_one_minute * time_in_minutes
  total_sugar = 108 := by
  sorry

end sugar_needed_in_two_minutes_l341_341578


namespace distance_between_lines_l341_341362

-- Define the edge length of the cube
variables (a : ℝ)

-- Define points of the cube
def A := (0, 0, 0)
def B := (a, 0, 0)
def C := (a, a, 0)
def D := (0, a, 0)
def A1 := (0, 0, a)
def B1 := (a, 0, a)
def C1 := (a, a, a)
def D1 := (0, a, a)

-- Define the lines BD1 and DC1
def line_BD1 := (B, D1)
def line_DC1 := (D, C1)

-- Distance between two skew lines in 3D space
noncomputable def distance_skew_lines (l1 l2 : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ) : ℝ := 
  sorry -- This function will calculate the distance between two skew lines

-- Proof statement
theorem distance_between_lines : distance_skew_lines line_BD1 line_DC1 = a * sqrt 6 / 6 := 
  sorry

end distance_between_lines_l341_341362


namespace sum_of_100th_row_l341_341309

def triangularArraySum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2^(n+1) - 3*n

theorem sum_of_100th_row :
  triangularArraySum 100 = 2^100 - 297 :=
by
  sorry

end sum_of_100th_row_l341_341309


namespace solve_for_s_l341_341666

noncomputable def compute_s : Set ℝ :=
  { s | ∀ (x : ℝ), (x ≠ -1) → ((s * x - 3) / (x + 1) = x ↔ x^2 + (1 - s) * x + 3 = 0) ∧
    ((1 - s) ^ 2 - 4 * 3 = 0) }

theorem solve_for_s (h : ∀ s ∈ compute_s, s = 1 + 2 * Real.sqrt 3 ∨ s = 1 - 2 * Real.sqrt 3) :
  compute_s = {1 + 2 * Real.sqrt 3, 1 - 2 * Real.sqrt 3} :=
by
  sorry

end solve_for_s_l341_341666


namespace sum_of_positive_divisors_of_prime_l341_341959

theorem sum_of_positive_divisors_of_prime (h_prime : Nat.prime 47) : 1 + 47 = 48 :=
by
  have d1 : 1 ∣ 47 := Nat.one_dvd 47
  have d47 : 47 ∣ 47 := Nat.dvd_refl 47
  have divisors := [1, 47]
  have sum_divisors := List.sum divisors
  rw [List.sum_cons, List.sum_nil] at sum_divisors
  simp at sum_divisors
  exact sum_divisors

end sum_of_positive_divisors_of_prime_l341_341959


namespace Jordan_income_l341_341060

theorem Jordan_income (q A : ℝ) (h : A > 30000)
  (h1 : (q / 100 * 30000 + (q + 3) / 100 * (A - 30000) - 600) = (q + 0.5) / 100 * A) :
  A = 60000 :=
by
  sorry

end Jordan_income_l341_341060


namespace sum_and_product_of_roots_l341_341195

theorem sum_and_product_of_roots (m p : ℝ) 
    (h₁ : ∀ α β : ℝ, (3 * α^2 - m * α + p = 0 ∧ 3 * β^2 - m * β + p = 0) → α + β = 9)
    (h₂ : ∀ α β : ℝ, (3 * α^2 - m * α + p = 0 ∧ 3 * β^2 - m * β + p = 0) → α * β = 14) :
    m + p = 69 := 
sorry

end sum_and_product_of_roots_l341_341195


namespace max_blocks_fit_l341_341220

-- Defining the dimensions of the box and blocks
def box_length : ℝ := 4
def box_width : ℝ := 3
def box_height : ℝ := 2

def block_length : ℝ := 3
def block_width : ℝ := 1
def block_height : ℝ := 1

-- Theorem stating the maximum number of blocks that fit
theorem max_blocks_fit : (24 / 3 = 8) ∧ (1 * 3 * 2 = 6) → 6 = 6 := 
by
  sorry

end max_blocks_fit_l341_341220


namespace cuboid_volume_l341_341257

theorem cuboid_volume (a b c : ℕ) (h_incr_by_2_becomes_cube : c + 2 = a)
  (surface_area_incr : 2*a*(a + a + c + 2) - 2*a*(c + a + b) = 56) : a * b * c = 245 :=
sorry

end cuboid_volume_l341_341257


namespace difference_si_ci_l341_341978

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def compound_interest (P : ℝ) (R : ℝ) (n : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R / (n * 100))^(n * T) - P

theorem difference_si_ci (P R T n : ℝ) (hP : P = 1200) (hR : R = 10) (hT : T = 1) (hn : n = 2) :
  let SI := simple_interest P R T,
      CI := compound_interest P R n T in
  CI - SI = -59.25 :=
by
  intros
  rw [hP, hR, hT, hn]
  dsimp [simple_interest, compound_interest]
  rw [real.div_mul_eq_div_div, real.div_div_eq_div_mul]
  have h1 : (1 + 0.05) = 1.05, by norm_num
  have h2 : (1.05)^2 = 1.1025, by norm_num
  simp only [h1, h2]
  norm_num
  sorry

end difference_si_ci_l341_341978


namespace f_minus_exp_increasing_min_x_value_l341_341854

-- Definitions and conditions from the problem
variables {f : ℝ → ℝ}

-- Condition 1: f' exists and f'(x) * exp(-x) > 1 for all x in (0, +∞)
axiom h1 : ∀ x > 0, (has_deriv_at f x) ∧ (f' x) * real.exp (-x) > 1

-- Condition 2: f(ln x) ≥ x + sqrt(e)
axiom h2 : ∀ x > 0, f (real.log x) ≥ x + real.sqrt real.exp 1

-- Condition 3: f(1/2) = 2 * sqrt(e)
axiom h3 : f (1 / 2) = 2 * real.sqrt real.exp 1

-- Theorem 1: f(x) - exp(x) is increasing for x in (0, +∞)
theorem f_minus_exp_increasing : ∀ x > 0, deriv (λ (x : ℝ), f x - real.exp x) x > 0 :=
by
  assume x hx,
  have h_deriv : has_deriv_at (λ x, f x - real.exp x) (f' x - real.exp x) x := (has_deriv_at.sub (has_deriv_at_of_has_deriv_at f x) (has_deriv_at_exp x)),
  exact (lt_of_lt_of_le (lt_of_mul_gt_of_pos hx (h1 x hx)) (sub_nonneg_of_le (le_refl (real.exp x))))

-- Theorem 2: The minimum value of x such that x ≥ sqrt(e)
theorem min_x_value : ∃ x > 0, x = real.sqrt real.exp 1 ∧ f (real.log x) ≥ x + real.sqrt real.exp 1 :=
by
  use real.sqrt real.exp 1,
  split,
  exact real.sqrt_pos_of_pos real.exp_pos,
  split,
  refl,
  exact (h2 (real.sqrt real.exp 1) (real.sqrt_pos_of_pos real.exp_pos))

end f_minus_exp_increasing_min_x_value_l341_341854


namespace gcd_204_85_l341_341188

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l341_341188


namespace noncongruent_triangles_count_l341_341880

theorem noncongruent_triangles_count :
  ∀ (A B C M N O : Point),
    is_isosceles_triangle A B C ∧
    midpoint A B M ∧ midpoint B C N ∧ midpoint C A O →
    number_of_noncongruent_triangles {A, B, C, M, N, O} = 10 :=
by
  intros
  sorry

end noncongruent_triangles_count_l341_341880


namespace sugar_used_in_two_minutes_l341_341580

-- Define constants for problem conditions
def sugar_per_bar : Float := 1.5
def bars_per_minute : Nat := 36
def minutes : Nat := 2

-- Define the total sugar used in two minutes
def total_sugar_used : Float := (bars_per_minute * sugar_per_bar) * minutes

-- State the theorem and its proof
theorem sugar_used_in_two_minutes : total_sugar_used = 108 := by
  sorry

end sugar_used_in_two_minutes_l341_341580


namespace sum_of_positive_divisors_of_prime_l341_341958

theorem sum_of_positive_divisors_of_prime (h_prime : Nat.prime 47) : 1 + 47 = 48 :=
by
  have d1 : 1 ∣ 47 := Nat.one_dvd 47
  have d47 : 47 ∣ 47 := Nat.dvd_refl 47
  have divisors := [1, 47]
  have sum_divisors := List.sum divisors
  rw [List.sum_cons, List.sum_nil] at sum_divisors
  simp at sum_divisors
  exact sum_divisors

end sum_of_positive_divisors_of_prime_l341_341958


namespace triangle_area_126_altitude_largest_side_9_l341_341005

-- Define the sides and vertices of the triangle
variables (A B C : Type)
variables (AB BC AC : ℝ)
variables (non_right_angled : ¬(A - B = B - C ∨ B - C = C - A ∨ C - A = A - B))
variables (h_AB : AB = 41) (h_BC : BC = 28) (h_AC : AC = 15)

-- Define the Cosine of angle C based on the law of cosines
noncomputable def cos_C := (AC^2 + BC^2 - AB^2) / (2 * AC * BC)

-- Define the sine of angle C using sin^2 = 1 - cos^2
noncomputable def sin_C := Real.sqrt(1 - cos_C^2)

-- Calculate the area of the triangle using the formula: (1/2) * a * b * sin(C)
noncomputable def area := (1 / 2) * AC * BC * sin_C

-- Calculate the altitude corresponding to the largest side BC
noncomputable def altitude := (2 * area) / BC

theorem triangle_area_126 : area = 126 := by
  sorry

theorem altitude_largest_side_9 : altitude = 9 := by
  sorry

end triangle_area_126_altitude_largest_side_9_l341_341005


namespace n_times_s_eq_9001_div_9_l341_341098

noncomputable def T := {x : ℝ // x > 0}

-- Given function f that satisfies the given condition
def f (x : T) : ℝ := sorry

-- Condition the function f must satisfy
axiom f_condition : ∀ x y : T, f x * f y = f ⟨x.val * y.val, mul_pos x.property y.property⟩ + 1000 * (1 / x.val ^ 2 + 1 / y.val ^ 2 + 1000)

-- Definition of the proof theorem
theorem n_times_s_eq_9001_div_9 (n s : ℝ) (h_n_s : n = 1 ∧ s = 9001 / 9) : n * s = 9001 / 9 := 
by { 
  cases h_n_s,
  rw [h_n_s_left, h_n_s_right] 
}

end n_times_s_eq_9001_div_9_l341_341098


namespace solve_for_Q_l341_341523

noncomputable def Q_solution (Q : ℝ) : Prop :=
  sqrt (Q^3) = 10 * root 4 100

theorem solve_for_Q : ∃ Q : ℝ, Q_solution Q ∧ Q = 10 :=
by
  let Q := 10
  have h1: Q_solution Q := by
    calc
      sqrt (Q^3) = sqrt (10^3) : by rw [Q, pow3_eq_mul_self_sqr]
      ... = 10 * root 4 100 : sorry
  exact ⟨Q, h1, rfl⟩

end solve_for_Q_l341_341523


namespace arithmetic_sequence_26th_term_l341_341907

theorem arithmetic_sequence_26th_term (a d : ℤ) (h1 : a = 3) (h2 : a + d = 13) (h3 : a + 2 * d = 23) : 
  a + 25 * d = 253 :=
by
  -- specifications for variables a, d, and hypotheses h1, h2, h3
  sorry

end arithmetic_sequence_26th_term_l341_341907


namespace h_h_neg1_l341_341773

def h (x: ℝ) : ℝ := 3 * x^2 - x + 1

theorem h_h_neg1 : h (h (-1)) = 71 := by
  sorry

end h_h_neg1_l341_341773


namespace units_digit_2_pow_2015_l341_341503

theorem units_digit_2_pow_2015 : ∃ u : ℕ, (2 ^ 2015 % 10) = u ∧ u = 8 := 
by
  sorry

end units_digit_2_pow_2015_l341_341503


namespace real_part_of_inverse_l341_341120

def z : ℂ := sorry

theorem real_part_of_inverse (hz: |z| = 2 ∧ im z ≠ 0) : real_part (1 / (2 - z)) = 1 / 2 := 
by
  sorry

end real_part_of_inverse_l341_341120


namespace f_of_3_l341_341357

theorem f_of_3 (f : ℕ → ℕ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 3 = 7 := 
sorry

end f_of_3_l341_341357


namespace day_of_week_dec_26_l341_341319

theorem day_of_week_dec_26 (nov_26_is_thu : true) : true :=
sorry

end day_of_week_dec_26_l341_341319


namespace coef_x4_pq_l341_341944

noncomputable def p (x : ℤ) : ℤ :=
  x^5 - 2*x^4 + 4*x^3 - 5*x^2 + 2*x - 1

noncomputable def q (x : ℤ) : ℤ :=
  3*x^4 - x^3 + 2*x^2 + 6*x - 5

theorem coef_x4_pq : coeff_x4 (p * q) = 19 :=
  sorry

end coef_x4_pq_l341_341944


namespace max_blocks_fit_l341_341221

def block_dimensions : ℝ × ℝ × ℝ := (3, 1, 1)
def box_dimensions : ℝ × ℝ × ℝ := (4, 3, 2)

theorem max_blocks_fit :
  let (block_l, block_w, block_h) := block_dimensions in
  let (box_l, box_w, box_h) := box_dimensions in
  (block_l ≤ box_l ∧ block_w ≤ box_w ∧ block_h ≤ box_h) →
  block_l * block_w * block_h > 0 →
  ∃ k : ℕ, k = 6 :=
by
  sorry

end max_blocks_fit_l341_341221


namespace find_angle_opposite_side_l341_341066

noncomputable def angle_opposite_side {a b c : ℝ} (h : (a + b + c) * (a + b - c) = 4 * a * b) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  let C := 0 in if hC : ∃ C, C = 0 then 0 else 0

theorem find_angle_opposite_side {a b c : ℝ} (h : (a + b + c) * (a + b - c) = 4 * a * b) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  angle_opposite_side h ha hb hc = 0 := by
  sorry

end find_angle_opposite_side_l341_341066


namespace nine_div_repeating_decimal_l341_341217

noncomputable def repeating_decimal := 1 / 3

theorem nine_div_repeating_decimal : 9 / repeating_decimal = 27 := by
  sorry

end nine_div_repeating_decimal_l341_341217


namespace problem_solution_l341_341253

def white_ball_condition (n : ℕ) : Prop :=
  ∀ (k : ℕ), let i := (k % 5) in
  2 ≤ (if i = 0 then 1 else if i = 1 then 1 else 0) +

       (if i = 1 then 1 else if i = 2 then 1 else 0) +

       (if i = 2 then 1 else if i = 3 then 1 else 0) +

       (if i = 3 then 1 else if i = 4 then 1 else 0) +

       (if i = 4 then 1 else if i = 0 then 1 else 0) ∧
  (if k < n then true else false) = 2

theorem problem_solution {n : ℕ} (h1 : n ≠ 0) (h2 : n ≠ 2021) (h3 : n ≠ 2022) (h4 : n ≠ 2023) (h5 : n ≠ 2024) :
  (¬ (n % 5 = 0)) ∧ white_ball_condition n → false :=
by {
  sorry
}

end problem_solution_l341_341253


namespace sum_of_divisors_of_47_l341_341954

theorem sum_of_divisors_of_47 : 
  ∑ d in {1, 47}, d = 48 := 
by 
  sorry

end sum_of_divisors_of_47_l341_341954


namespace magnitude_of_sum_of_perpendicular_vectors_l341_341051

theorem magnitude_of_sum_of_perpendicular_vectors (x : ℝ) :
  let a := (2, 1 : ℝ × ℝ)
  let b := (x - 1, -x : ℝ × ℝ)
  (2 * (x - 1) + 1 * (-x)) = 0 →
  ‖((2, 1) + (x - 1, -x) : ℚ × ℚ)‖ = real.sqrt 10 :=
by 
  intro a b h,
  sorry

end magnitude_of_sum_of_perpendicular_vectors_l341_341051


namespace arrange_plants_l341_341652

-- Define the problem statement
theorem arrange_plants (b : Fin 5) (t : Fin 3) : 
  let total_plants := 5 + 3
  let group := 6 -- Tomato group as a single entity + 5 basil plants
  let ways_arrange_in_circle := (group - 1)!
  let ways_arrange_tomato_group := (3)!
  (ways_arrange_in_circle * ways_arrange_tomato_group) = 720 :=
by 
    sorry

end arrange_plants_l341_341652


namespace centroid_iff_equal_areas_l341_341151

variables {A B C N : Type} [innTriangle : IsInTriangle N A B C]

def area (X Y Z : Type) : Type := sorry -- Define area function for triangles.

theorem centroid_iff_equal_areas 
  (h1 : area N A B = area N B C)
  (h2 : area N B C = area N C A) :
  is_centroid N A B C :=
sorry

end centroid_iff_equal_areas_l341_341151


namespace gcd_204_85_l341_341183

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := 
by sorry

end gcd_204_85_l341_341183


namespace circle_ellipse_tangent_radius_l341_341584

theorem circle_ellipse_tangent_radius :
  ∀ (r : ℝ), (∀ x y : ℝ, (x - r)^2 + y^2 = r^2 → x^2 + 4y^2 = 5) →
  r = real.sqrt(15) / 4 :=
by
  -- The proof goes here.
  sorry

end circle_ellipse_tangent_radius_l341_341584


namespace sum_of_divisors_of_prime_47_l341_341964

theorem sum_of_divisors_of_prime_47 : ∀ n : ℕ, prime n → n = 47 → (∑ d in (finset.filter (λ x, n % x = 0) (finset.range (n + 1))), d) = 48 := 
by {
  intros n prime_n n_is_47,
  sorry -- Proof is omitted
}

end sum_of_divisors_of_prime_47_l341_341964


namespace right_isosceles_triangle_acute_angle_l341_341802

noncomputable def measure_of_acute_angle (θ : ℝ) : Prop :=
  let a := 1 -- any positive value works due to scaling properties, assume a = 1 for simplification
  let h := a * Real.sqrt 2
  let sin_θ := Real.sin θ in
  h^2 = 3 * a * sin_θ → θ = 45

theorem right_isosceles_triangle_acute_angle 
  (a h : ℝ) (θ : ℝ) (h_square_eq : h^2 = 3 * a * Real.sin θ)
  (isosceles : h = a * Real.sqrt 2) : θ = 45 :=
by
  sorry

end right_isosceles_triangle_acute_angle_l341_341802


namespace rectangle_length_l341_341346

theorem rectangle_length {x w : ℝ} 
  (area_PQRS : 5 * (x * w) = 4000) 
  (dimension_relation : 3 * w = 2 * x) : 
  abs (x - 35) < 1 :=
by 
  have w_def : w = 2 / 3 * x := by linarith
  have area_eq : 5 * (x * (2 / 3 * x)) = 4000 := by rw [w_def, mul_assoc]
  simp at area_eq
  have x_squared_eq : x ^ 2 = 1200 := by linarith
  have x_value : x = real.sqrt 1200 := by rw [real.sqrt_eq, x_squared_eq]
  have approx_x : real.sqrt 1200 ≈ 34.6 := sorry
  show abs (x - 35) < 1, by linarith

end rectangle_length_l341_341346


namespace zoo_animal_arrangement_l341_341813

/-- In the Sunny Farm Zoo, there are 4 elephants, 3 rabbits, and 5 parrots.
    Prove that the number of unique ways to arrange the 12 animals in a line
    such that all of the animals from each species are grouped together
    is 103680. -/
theorem zoo_animal_arrangement :
  let elephants := 4
  let rabbits := 3
  let parrots := 5
  let total_animals := 12
  let animals := [elephants, rabbits, parrots]
  (fact 3) * (fact elephants) * (fact rabbits) * (fact parrots) = 103680 :=
by {
  let elephants := 4
  let rabbits := 3
  let parrots := 5
  let animals := [elephants, rabbits, parrots]
  show (fact 3) * (fact elephants) * (fact rabbits) * (fact parrots) = 103680,
  sorry
}

end zoo_animal_arrangement_l341_341813


namespace cats_left_l341_341629

theorem cats_left (siamese_cats : ℕ) (house_cats : ℕ) (cats_sold : ℕ) (total_initial_cats : ℕ) (remaining_cats : ℕ) :
  siamese_cats = 15 → house_cats = 49 → cats_sold = 19 → total_initial_cats = siamese_cats + house_cats → remaining_cats = total_initial_cats - cats_sold → remaining_cats = 45 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h4, h3] at h5
  exact h5

end cats_left_l341_341629


namespace min_angle_B_l341_341381

-- Definitions using conditions from part a)
def triangle (A B C : ℝ) : Prop := A + B + C = Real.pi
def arithmetic_sequence_prop (A B C : ℝ) : Prop := 
  Real.tan A + Real.tan C = 2 * (1 + Real.sqrt 2) * Real.tan B

-- Main theorem to prove
theorem min_angle_B (A B C : ℝ) (h1 : triangle A B C) (h2 : arithmetic_sequence_prop A B C) :
  B ≥ Real.pi / 4 :=
sorry

end min_angle_B_l341_341381


namespace steven_sum_l341_341159

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10
def diff_digits (n : ℕ) : ℤ := tens_digit n - units_digit n

theorem steven_sum : (∑ n in Finset.range 90 \ Finset.range 10, diff_digits (n + 10)) = 45 :=
by
  sorry

end steven_sum_l341_341159


namespace expected_value_of_remainder_mod_64_l341_341472

noncomputable def M (a b c d e f : ℕ) : ℕ := a + 2 * b + 4 * c + 8 * d + 16 * e + 32 * f

theorem expected_value_of_remainder_mod_64 (a b c d e f : ℕ) (ha : 1 ≤ a ∧ a ≤ 100) (hb : 1 ≤ b ∧ b ≤ 100) (hc : 1 ≤ c ∧ c ≤ 100) (hd : 1 ≤ d ∧ d ≤ 100) (he : 1 ≤ e ∧ e ≤ 100) (hf : 1 ≤ f ∧ f ≤ 100) : 
  (∑ i in finset.range 64, i) / 64 = 31.5 :=
by 
  sorry

end expected_value_of_remainder_mod_64_l341_341472


namespace parabola_focus_distance_l341_341785

theorem parabola_focus_distance :
  let p := 4
  let a := √(3)
  let b := 1
  let c := √(a^2 + b^2)
  let h_focus := (2, 0)
  let para_focus := (2, 0)
  let P := (2, b')
  let distance := (λ P1 P2: ℝ × ℝ, Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2))
  in h_focus = para_focus → distance P para_focus = 4 :=
by
  sorry

end parabola_focus_distance_l341_341785


namespace distinct_mappings_count_l341_341679

-- Define the sets M and N
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {1, 2, 3, 4}

-- Define the number of elements in each set
def |M| : ℕ := 3
def |N| : ℕ := 4

-- Prove that the number of distinct mappings from M to N is 64
theorem distinct_mappings_count : (|N| ^ |M|) = 64 :=
by
  sorry

end distinct_mappings_count_l341_341679


namespace max_blocks_fit_l341_341219

-- Defining the dimensions of the box and blocks
def box_length : ℝ := 4
def box_width : ℝ := 3
def box_height : ℝ := 2

def block_length : ℝ := 3
def block_width : ℝ := 1
def block_height : ℝ := 1

-- Theorem stating the maximum number of blocks that fit
theorem max_blocks_fit : (24 / 3 = 8) ∧ (1 * 3 * 2 = 6) → 6 = 6 := 
by
  sorry

end max_blocks_fit_l341_341219


namespace not_always_possible_even_distribution_l341_341567

theorem not_always_possible_even_distribution :
  ¬ ∀ (a : Fin 23 → ℕ), (∀ i, a i ≥ 0) →
  (∑ i, a i) = 46 → 
  ∃ b : Fin 23 → ℕ, (∀ i, b i = 2) ∧
  (∑ i, i.val * b i) % 23 = (∑ i, i.val * a i) % 23 :=
begin
  sorry
end

end not_always_possible_even_distribution_l341_341567


namespace triangle_DEF_area_10_l341_341936

-- Definitions of vertices and line
def D : ℝ × ℝ := (4, 0)
def E : ℝ × ℝ := (0, 4)
def line (x y : ℝ) : Prop := x + y = 9

-- Definition of point F lying on the given line
axiom F_on_line (F : ℝ × ℝ) : line (F.1) (F.2)

-- The proof statement of the area of triangle DEF being 10
theorem triangle_DEF_area_10 : ∃ F : ℝ × ℝ, line F.1 F.2 ∧ 
  (1 / 2) * abs (D.1 - F.1) * abs E.2 = 10 :=
by
  sorry

end triangle_DEF_area_10_l341_341936


namespace positive_difference_of_perimeters_l341_341940

/-- 
  Define the perimeters of the two figures based on given conditions:
  - Rect1: 4x1 rectangle.
  - Rect2: A 6x1 rectangle divided into two 3x1 sections with an extra vertical shift of 1 unit.
-/
def perimeter_rect1 : ℕ := 2 * (4 + 1)
def perimeter_rect2 : ℕ := 2 * 6 + 4

/-- 
  The problem statement: the positive difference of the perimeters of the two figures is 6 units.
-/
theorem positive_difference_of_perimeters : abs (perimeter_rect2 - perimeter_rect1) = 6 := 
sorry

end positive_difference_of_perimeters_l341_341940


namespace number_of_elderly_sampled_l341_341615

def total_elderly : ℕ := 30
def total_middle_aged : ℕ := 90
def total_young : ℕ := 60
def sample_size : ℕ := 36

theorem number_of_elderly_sampled : 
  let total_population := total_elderly + total_middle_aged + total_young in
  sample_size = total_population / 5 →
  sample_size * total_elderly / total_population = 6 :=
sorry

end number_of_elderly_sampled_l341_341615


namespace perimeter_of_regular_pentagon_is_75_l341_341929

-- Define the side length and the property of the figure
def side_length : ℝ := 15
def is_regular_pentagon : Prop := true  -- assuming this captures the regular pentagon property

-- Define the perimeter calculation based on the conditions
def perimeter (n : ℕ) (side_length : ℝ) := n * side_length

-- The theorem to prove
theorem perimeter_of_regular_pentagon_is_75 :
  is_regular_pentagon → perimeter 5 side_length = 75 :=
by
  intro _ -- We don't need to use is_regular_pentagon directly
  rw [side_length]
  norm_num
  sorry

end perimeter_of_regular_pentagon_is_75_l341_341929


namespace real_part_proof_l341_341123

noncomputable def real_part_of_fraction (z : ℂ) (h : |z| = 2 ∧ z.im ≠ 0) : ℝ :=
  let x := z.re in
  let y := z.im in
  (2 - x) / (8 - 4 * x + x^2)

theorem real_part_proof (z : ℂ) (h : |z| = 2 ∧ z.im ≠ 0) :
  (real_part_of_fraction z h) = (2 - z.re) / (8 - 4 * z.re + z.re^2) :=
by
  sorry

end real_part_proof_l341_341123


namespace number_of_disjoint_subsets_remainder_l341_341105

-- Define the set S
def S : set ℕ := {i | 1 ≤ i ∧ i ≤ 12}

-- The main theorem statement
theorem number_of_disjoint_subsets_remainder :
  let n := 1 / 2 * (3^12 - 2 * 2^12 + 1) in
  n % 1000 = 625 :=
by
  sorry

end number_of_disjoint_subsets_remainder_l341_341105


namespace proof_question1_proof_question2_l341_341076

variables (city_A : Type)
variables (national_store_survey : Prop)
variables (beverage_survey : Prop)
variables (popular_colors_autumn : Prop)
variables (mode_color_city_A_red : Prop)
variables (national_released_color_coffee : Prop)
variables (chain_store_sample : Prop)
variables (representative_sample : Prop)

noncomputable def question1 (h1 : national_store_survey)
                            (h2 : beverage_survey)
                            (h3 : popular_colors_autumn)
                            (h4 : mode_color_city_A_red)
                            (h5 : national_released_color_coffee)
                            (h6 : chain_store_sample)
                            : Prop := ¬ representative_sample -- This translates to "No"

noncomputable def question2 (h1 : national_store_survey)
                            (h2 : beverage_survey)
                            (h3 : popular_colors_autumn)
                            (h4 : mode_color_city_A_red)
                            (h5 : national_released_color_coffee)
                            (h6 : chain_store_sample)
                            : Prop := chain_store_sample   -- This translates to "The representativeness of the sample"

theorem proof_question1 (h1 : national_store_survey)
                        (h2 : beverage_survey)
                        (h3 : popular_colors_autumn)
                        (h4 : mode_color_city_A_red)
                        (h5 : national_released_color_coffee)
                        (h6 : chain_store_sample)
                        : question1 h1 h2 h3 h4 h5 h6 :=
sorry

theorem proof_question2 (h1 : national_store_survey)
                        (h2 : beverage_survey)
                        (h3 : popular_colors_autumn)
                        (h4 : mode_color_city_A_red)
                        (h5 : national_released_color_coffee)
                        (h6 : chain_store_sample)
                        : question2 h1 h2 h3 h4 h5 h6 :=
sorry

end proof_question1_proof_question2_l341_341076


namespace jackson_miles_l341_341292

theorem jackson_miles (beka_miles jackson_miles : ℕ) (h1 : beka_miles = 873) (h2 : beka_miles = jackson_miles + 310) : jackson_miles = 563 := by
  sorry

end jackson_miles_l341_341292


namespace binom_9_5_l341_341299

theorem binom_9_5 : nat.binomial 9 5 = 126 := by
  sorry

end binom_9_5_l341_341299


namespace lambda_inequality_l341_341367

noncomputable def findLambda (P : Fin 5 → ℝ×ℝ) : ℝ :=
  let distances := {d | ∃ i j, i ≠ j ∧ d = ((P i).1 - (P j).1)^2 + ((P i).2 - (P j).2)^2}
  let d_max := distances.sup (λ x, x)
  let d_min := distances.inf (λ x, x)
  d_max / d_min

theorem lambda_inequality (P : Fin 5 → ℝ×ℝ) :
  findLambda P ≥ 2 * Real.sin (Real.pi * 54 / 180) :=
sorry

end lambda_inequality_l341_341367


namespace corrected_mean_l341_341243

theorem corrected_mean (incorrect_mean : ℕ) (num_observations : ℕ) (wrong_value actual_value : ℕ) : 
  (50 * 36 + (43 - 23)) / 50 = 36.4 :=
by
  sorry

end corrected_mean_l341_341243


namespace sphere_prism_area_diff_l341_341634

noncomputable def maxLateralAreaDiff (r a : ℝ) (h : ℝ) : ℝ :=
  if h^2 + 2 * a^2 = 16 then 16 * (π - real.sqrt 2) else 0

theorem sphere_prism_area_diff :
  maxLateralAreaDiff 2 2 (real.sqrt 12) = 16 * (π - real.sqrt 2) :=
by 
  sorry

end sphere_prism_area_diff_l341_341634


namespace sequence_14th_term_l341_341396

-- Definition for the sequence
def a (n : ℕ) : ℝ := real.sqrt (6 * n - 3)

theorem sequence_14th_term (n : ℕ) (h : n = 14) : a n = 9 :=
by {
  rw h,
  unfold a,
  norm_num,
}


end sequence_14th_term_l341_341396


namespace real_part_of_inverse_is_half_l341_341126

noncomputable def real_part_of_inverse_expression (z : ℂ) (h : ∥z∥ = 2) (hnonreal : ¬ real z) : ℝ :=
  real_part (1 / (2 - z))

theorem real_part_of_inverse_is_half (z : ℂ) (h : ∥z∥ = 2) (hnonreal : ¬ real z) : 
  real_part_of_inverse_expression z h hnonreal = 1 / 2 :=
by
  sorry

end real_part_of_inverse_is_half_l341_341126


namespace fifth_equation_l341_341379

-- Define the conditions
def condition1 : Prop := 2^1 * 1 = 2
def condition2 : Prop := 2^2 * 1 * 3 = 3 * 4
def condition3 : Prop := 2^3 * 1 * 3 * 5 = 4 * 5 * 6

-- The statement to prove
theorem fifth_equation (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 :=
sorry

end fifth_equation_l341_341379


namespace house_rent_fraction_l341_341261

def fraction_spent_on_house_rent (S : ℝ) : ℝ :=
  let exp_food := (3/10) * S
  let exp_conveyance := (1/8) * S
  let total_exp := exp_food + exp_conveyance
  let house_rent := S - (1400 + 3400)
  let fraction := house_rent / S
  fraction

theorem house_rent_fraction: ∀ (S : ℝ), 
  (3/10) * S + (1/8) * S = 3400 → 
  (S - (3400 + 1400)) / S = 2 / 5 := by
    intro S
    intros h
    sorry

end house_rent_fraction_l341_341261


namespace hunter_rabbit_game_l341_341624

-- Define necessary constants and conditions
constant A : ℕ → ℝ × ℝ -- Rabbit position for n-th round
constant B : ℕ → ℝ × ℝ -- Hunter position for n-th round
constant P : ℕ → ℝ × ℝ -- Tracking device reported position for n-th round

-- Initial positions
axiom A0_eq_B0 : A 0 = B 0

-- Movement rules
axiom rabbit_move : ∀ n, dist (A n) (A (n + 1)) = 1
axiom tracking_device : ∀ n, dist (P (n + 1)) (A (n + 1)) ≤ 1
axiom hunter_move : ∀ n, dist (B n) (B (n + 1)) = 1

-- Main theorem to prove
theorem hunter_rabbit_game :
  ¬ (∃ f : (ℕ → (ℝ × ℝ)), (∀ i, dist (B i) (B (i + 1)) = 1) ∧ (dist (B 10^9) (A 10^9) ≤ 100)) := sorry

end hunter_rabbit_game_l341_341624


namespace third_box_weight_l341_341145

def b1 : ℕ := 2
def difference := 11

def weight_third_box (b1 b3 difference : ℕ) : Prop :=
  b3 - b1 = difference

theorem third_box_weight : weight_third_box b1 13 difference :=
by
  simp [b1, difference]
  sorry

end third_box_weight_l341_341145


namespace sandals_sold_l341_341541

theorem sandals_sold (S : ℕ) (shoes_to_sandals_ratio : ℕ × ℕ) (shoes_sold : ℕ) (h_ratio : shoes_to_sandals_ratio = (15, 8)) (h_shoes_sold : shoes_sold = 135) :
  15 * S = 8 * shoes_sold := 
by
  sorry

# Define specific values used in the theorem
def sandals_sold_specific (S : ℕ) (h_S : S = 72) : 15 * S = 8 * 135 := 
by
  sorry

end sandals_sold_l341_341541


namespace evaluate_expression_l341_341660

theorem evaluate_expression : 
  let E := (√3 - 1)^2 + (√3 - √2) * (√2 + √3) + (√2 + 1) / (√2 - 1) - 3 * √(1 / 2)
  in E = 8 - 2 * √3 + (√2 / 2) :=
by
  sorry

end evaluate_expression_l341_341660


namespace real_part_of_inverse_is_half_l341_341129

noncomputable def real_part_of_inverse_expression (z : ℂ) (h : ∥z∥ = 2) (hnonreal : ¬ real z) : ℝ :=
  real_part (1 / (2 - z))

theorem real_part_of_inverse_is_half (z : ℂ) (h : ∥z∥ = 2) (hnonreal : ¬ real z) : 
  real_part_of_inverse_expression z h hnonreal = 1 / 2 :=
by
  sorry

end real_part_of_inverse_is_half_l341_341129


namespace value_of_expression_l341_341778

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  rw [h1, h2]
  norm_num
  sorry

end value_of_expression_l341_341778


namespace frequency_histogram_approaches_density_curve_l341_341513
-- Import all Mathlib

-- Declare the necessary definitions and assumptions as per the conditions
variable {α : Type*} -- Assume a type for our sample and population data

-- Definitions as per the problem
def sample_size_increases_indefinitely (samples : ℕ → set α) : Prop :=
  ∀ n, ∃ N, n ≤ N ∧ finite (samples N)

def class_interval_decreases_infinitely (classes : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, classes n ≤ ε

def frequency_distribution_histogram
(samples : ℕ → set α) (classes : ℕ → ℝ) : ℕ → ℝ
| n := -- some function here, we can't infer from problem so keeping placeholder
  sorry

def population_density_curve (population : set α) : ℝ :=
  -- some function here, we can't infer from problem so keeping placeholder
  sorry

-- The theorem statement to prove
theorem frequency_histogram_approaches_density_curve
(population : set α)
(samples : ℕ → set α)
(classes : ℕ → ℝ)
(h_increases : sample_size_increases_indefinitely samples)
(h_decreases : class_interval_decreases_infinitely classes) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N,
    ∀ x ∈ population, abs (frequency_distribution_histogram samples classes n x - population_density_curve population x) < ε :=
sorry

end frequency_histogram_approaches_density_curve_l341_341513


namespace centroid_construction_possible_l341_341115

-- Define the conditions.
def is_divisible_by (m k : ℕ) : Prop := ∃ d, m = d * k
def has_divisible_centroid (n m : ℕ) : Prop :=
  n ≥ 3 ∧ (∀ p, p ∣ n → is_divisible_by m p) ∧ is_divisible_by m 2

-- Lean statement for the problem.
theorem centroid_construction_possible (n m : ℕ) (h₁ : n ≥ 3)
  (h₂ : ∀ p : ℕ, p.prime → p ∣ n → ∃ d, m = d * p)
  (h₃ : ∃ d, m = d * 2) :
  has_divisible_centroid n m :=
by
  sorry

end centroid_construction_possible_l341_341115


namespace area_of_triangle_ABC_l341_341490

-- Define the problem conditions
variables (O A B C : Type) [euclidean_space O A B C]
          (orig : O = (0, 0, 0))
          (Ox : A = (√⁴ 48, 0, 0))
          (Oy : B = (0, OA, 0))
          (Oz : C = (0, 0, OC))
          (angle_BAC : ∠BAC = 45°)

-- State the theorem to conclude the area of the triangle ABC
theorem area_of_triangle_ABC : 
  let a := (sqrt[4] 48 : ℝ) in
  let area_triangle_ABC := (1 / 2) * (a^2) * (sqrt 3 / 2) in
  area_triangle_ABC = 2 * sqrt 3 :=
  sorry

end area_of_triangle_ABC_l341_341490


namespace det_proof_l341_341834

variable {R : Type*} [Field R] [Inhabited R]

def matrix_2x2 (a b c d : R) : Matrix (Fin 2) (Fin 2) R :=
  !![a, b; c, d]

theorem det_proof (x : R) (A : Matrix (Fin 2) (Fin 2) R) (h₁ : 0 < x)
    (h₂ : A.is_square) (h₃ : A.det ≠ 0) (h₄ : (A^2 + (scalarMatrix 2 x)).det = 0) :
    (A^2 + A + (scalarMatrix 2 x)).det = x := by
  sorry

end det_proof_l341_341834


namespace harry_fish_count_l341_341142

theorem harry_fish_count
  (sam_fish : ℕ) (joe_fish : ℕ) (harry_fish : ℕ)
  (h1 : sam_fish = 7)
  (h2 : joe_fish = 8 * sam_fish)
  (h3 : harry_fish = 4 * joe_fish) :
  harry_fish = 224 :=
by
  sorry

end harry_fish_count_l341_341142


namespace train_length_l341_341275

open Real

theorem train_length (speed_kmph : ℝ) (bridge_length : ℝ) (time_sec : ℝ) (train_length : ℝ) : 
  (speed_kmph = 45) → (bridge_length = 235) → (time_sec = 30) → 
  ((speed_kmph * 1000 / 3600) * time_sec = train_length + bridge_length) → 
  train_length = 140 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have speed_mps: 45 * 1000 / 3600 = 12.5 := by norm_num
  rw speed_mps at h4
  have distance: 12.5 * 30 = 375 := by norm_num
  rw distance at h4
  linarith

end train_length_l341_341275


namespace trains_meet_simultaneously_trains_48_km_apart_l341_341287

/-- Define the speed of slow and fast trains and the distance between two stations --/
def speed_slow := 60 -- km/h
def speed_fast := 100 -- km/h
def distance_AB := 448 -- km

/-- Prove the time (in hours) it takes for the slow and fast trains to meet when they depart simultaneously. --/
theorem trains_meet_simultaneously : 
  (distance_AB / (speed_slow + speed_fast)) = 2.8 := 
by 
  sorry

/-- Define the time difference between the departures of slow and fast trains. --/
def time_difference := 32 / 60 -- 32 minutes earlier in hours

/-- Prove the times (in hours) it takes for the fast train to ensure the distance between them is 48 km 
    given the slow train departs 32 minutes earlier. --/
theorem trains_48_km_apart : 
  ∃ y,  (speed_slow * time_difference + speed_slow * y + speed_fast * y = distance_AB - 48) ∧ 
        (y = 2.3 ∨ y = 2.9) := 
by 
  sorry

end trains_meet_simultaneously_trains_48_km_apart_l341_341287


namespace maryann_rescue_time_l341_341872

def time_to_free_cheaph (minutes : ℕ) : ℕ := 6
def time_to_free_expenh (minutes : ℕ) : ℕ := 8
def num_friends : ℕ := 3

theorem maryann_rescue_time : (time_to_free_cheaph 6 + time_to_free_expenh 8) * num_friends = 42 := 
by
  sorry

end maryann_rescue_time_l341_341872


namespace proof_X_Y_properties_l341_341371

noncomputable def X : MeasureTheory.ProbabilityDistributions.PMF ℕ :=
  MeasureTheory.ProbabilityDistributions.PMF.binomial 10 0.6

def Y (X : ℕ) : ℕ := 8 - X

theorem proof_X_Y_properties :
  (MeasureTheory.AEStronglyMeasurable (λ ω, X.val) MeasureTheory.measure_space.volume) →
  (MeasureTheory.AEStronglyMeasurable (λ ω, Y (X.val)) MeasureTheory.measure_space.volume) →
  (MeasureTheory.ProbabilityDistribution.expectation X.toReal = 6) ∧
  (MeasureTheory.ProbabilityDistribution.expectation (λ ω, (Y (X.val)).toReal) = 2) ∧
  (MeasureTheory.ProbabilityDistribution.variance X = 2.4) ∧
  (MeasureTheory.ProbabilityDistribution.variance (λ ω, Y (X.val)) = 2.4) :=
by
  sorry

end proof_X_Y_properties_l341_341371


namespace problem_l341_341355

def f (x : ℝ) (a b : ℝ) := x^5 + a * x^3 + b * x - 2

-- We are given f(-2) = m
variables (a b m : ℝ)
theorem problem (h : f (-2) a b = m) : f 2 a b + f (-2) a b = -4 :=
by sorry

end problem_l341_341355


namespace gcd_binom_is_integer_l341_341975

theorem gcd_binom_is_integer
  (m n : ℕ) (h_mn : m ≤ n) (d : ℕ) (h_d : d = Nat.gcd m n) :
  (d / n) * Nat.choose n m ∈ ℤ := by
  sorry

end gcd_binom_is_integer_l341_341975


namespace trees_planted_along_path_l341_341214

theorem trees_planted_along_path (p d : ℕ) (h_p : p = 50) (h_d : d = 2) : 
  2 * (p / d + 1) = 52 := 
by {
  -- Substituting the initial conditions
  rw [h_p, h_d],
  -- Perform calculation
  simp,
  -- Intermediate steps are abstracted
  sorry
}

end trees_planted_along_path_l341_341214


namespace inequality_for_distinct_integers_l341_341114

-- Define the necessary variables and conditions
variable {a b c : ℤ}

-- Ensure a, b, and c are pairwise distinct integers
def pairwise_distinct (a b c : ℤ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

-- The main theorem statement
theorem inequality_for_distinct_integers 
  (h : pairwise_distinct a b c) : 
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + Real.sqrt (3 * (a * b + b * c + c * a + 1)) :=
by
  sorry

end inequality_for_distinct_integers_l341_341114


namespace find_x_l341_341767

theorem find_x (x : ℝ) (a b c : ℝ × ℝ)
  (ha : a = (x, 1))
  (hb : b = (2, x))
  (hc : c = (1, -2))
  (h_perpendicular : (a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2)) = 0) :
  x = 1 / 2 :=
sorry

end find_x_l341_341767


namespace abs_eq_sum_condition_l341_341781

theorem abs_eq_sum_condition (x y : ℝ) (h : |x - y^2| = x + y^2) : x = 0 ∧ y = 0 :=
  sorry

end abs_eq_sum_condition_l341_341781


namespace arithmetic_sequence_ratio_l341_341527

theorem arithmetic_sequence_ratio (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = (1/2) * n * (2 * a 1 + (n-1) * d))
  (h2 : ∀ n, T n = (1/2) * n * (2 * b 1 + (n-1) * d'))
  (h3 : ∀ n, S n / T n = 7*n / (n + 3)): a 5 / b 5 = 21 / 4 := 
by {
  sorry
}

end arithmetic_sequence_ratio_l341_341527


namespace total_female_officers_l341_341600

-- Define the conditions using Lean definitions
def duty_percentage := 0.17
def total_on_duty := 204
def half_on_duty_female := total_on_duty / 2

-- The proof problem to be solved
theorem total_female_officers (F : ℕ) :
  half_on_duty_female = duty_percentage * F → F = 600 :=
by
  -- math logic placeholder
  intros
  sorry

end total_female_officers_l341_341600


namespace total_crayons_l341_341206

theorem total_crayons (initial_crayons : ℕ) (added_crayons : ℕ) (h_initial : initial_crayons = 9) (h_added : added_crayons = 3) : 
  initial_crayons + added_crayons = 12 :=
by 
  rw [h_initial, h_added]
  sorry

end total_crayons_l341_341206


namespace real_part_of_inverse_is_half_l341_341128

noncomputable def real_part_of_inverse_expression (z : ℂ) (h : ∥z∥ = 2) (hnonreal : ¬ real z) : ℝ :=
  real_part (1 / (2 - z))

theorem real_part_of_inverse_is_half (z : ℂ) (h : ∥z∥ = 2) (hnonreal : ¬ real z) : 
  real_part_of_inverse_expression z h hnonreal = 1 / 2 :=
by
  sorry

end real_part_of_inverse_is_half_l341_341128


namespace nancy_yearly_payment_l341_341501

open Real

-- Define the monthly cost of the car insurance
def monthly_cost : ℝ := 80

-- Nancy's percentage contribution
def percentage : ℝ := 0.40

-- Calculate the monthly payment Nancy will make
def monthly_payment : ℝ := percentage * monthly_cost

-- Calculate the yearly payment Nancy will make
def yearly_payment : ℝ := 12 * monthly_payment

-- State the proof problem
theorem nancy_yearly_payment : yearly_payment = 384 :=
by
  -- Proof goes here
  sorry

end nancy_yearly_payment_l341_341501


namespace ratio_of_tax_revenue_to_cost_of_stimulus_l341_341908

-- Definitions based on the identified conditions
def bottom_20_percent_people (total_people : ℕ) : ℕ := (total_people * 20) / 100
def stimulus_per_person : ℕ := 2000
def total_people : ℕ := 1000
def government_profit : ℕ := 1600000

-- Cost of the stimulus
def cost_of_stimulus : ℕ := bottom_20_percent_people total_people * stimulus_per_person

-- Tax revenue returned to the government
def tax_revenue : ℕ := government_profit + cost_of_stimulus

-- The Proposition we need to prove
theorem ratio_of_tax_revenue_to_cost_of_stimulus :
  tax_revenue / cost_of_stimulus = 5 :=
by
  sorry

end ratio_of_tax_revenue_to_cost_of_stimulus_l341_341908


namespace sum_of_triangle_altitudes_l341_341702

theorem sum_of_triangle_altitudes : 
  let A := (6, 0) in let B := (0, 16) in let C := (0, 0) in
  let height_A := 48 / 3 in let height_B := 48 / 8 in 
  let base := sqrt ((6 : ℝ)^2 + (16 : ℝ)^2) in
  let height_O := 96 / base in
  height_A + height_B + height_O = 370 / sqrt 292  :=
by
  -- Definitions of points and heights
  let A := (6 : ℝ, 0)
  let B := (0, 16 : ℝ)
  let C := (0, 0 : ℝ)
  let height_A := 16
  let height_B := 6
  let base := sqrt ((6 : ℝ) * (6) + (16 : ℝ) * (16))
  let height_O := 96 / base
  
  -- Proof to be filled in later
  sorry

end sum_of_triangle_altitudes_l341_341702


namespace fill_40x41_table_l341_341819

-- Define the condition on integers in the table
def valid_integer_filling (m n : ℕ) (table : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < m → j < n →
    table i j =
    ((if i > 0 then if table i j = table (i - 1) j then 1 else 0 else 0) +
    (if j > 0 then if table i j = table i (j - 1) then 1 else 0 else 0) +
    (if i < m - 1 then if table i j = table (i + 1) j then 1 else 0 else 0) +
    (if j < n - 1 then if table i j = table i (j + 1) then 1 else 0 else 0))

-- Define the specific problem for a 40 × 41 table.
theorem fill_40x41_table :
  ∃ (table : ℕ → ℕ → ℕ), valid_integer_filling 40 41 table :=
by
  sorry

end fill_40x41_table_l341_341819


namespace rounded_wavelength_is_correctly_represented_l341_341565

def wavelength : ℝ := 0.000077

def rounded_wavelength_nearest_0.00001 : ℝ := 0.00008

def scientific_notation_form (n : ℝ) : Prop :=
  n = 8 * 10^(-5)

theorem rounded_wavelength_is_correctly_represented :
  scientific_notation_form rounded_wavelength_nearest_0.00001 :=
by
  sorry

end rounded_wavelength_is_correctly_represented_l341_341565


namespace reunion_handshakes_l341_341942

-- Condition: Number of boys in total
def total_boys : ℕ := 12

-- Condition: Number of left-handed boys
def left_handed_boys : ℕ := 4

-- Condition: Number of right-handed (not exclusively left-handed) boys
def right_handed_boys : ℕ := total_boys - left_handed_boys

-- Function to calculate combinations n choose 2 (number of handshakes in a group)
def combinations (n : ℕ) : ℕ := n * (n - 1) / 2

-- Condition: Number of handshakes among left-handed boys
def handshakes_left (n : ℕ) : ℕ := combinations left_handed_boys

-- Condition: Number of handshakes among right-handed boys
def handshakes_right (n : ℕ) : ℕ := combinations right_handed_boys

-- Problem statement: total number of handshakes
def total_handshakes (total_boys left_handed_boys right_handed_boys : ℕ) : ℕ :=
  handshakes_left left_handed_boys + handshakes_right right_handed_boys

theorem reunion_handshakes : total_handshakes total_boys left_handed_boys right_handed_boys = 34 :=
by sorry

end reunion_handshakes_l341_341942


namespace binom_9_5_l341_341297

theorem binom_9_5 : nat.binomial 9 5 = 126 := by
  sorry

end binom_9_5_l341_341297


namespace linear_regression_negatively_correlated_l341_341384

variables (x y : ℝ)

def regression_equation (x : ℝ) := 1 - 2 * x

theorem linear_regression_negatively_correlated :
  (∃ f : ℝ → ℝ, f = regression_equation ∧ ∀ x, f x = 1 - 2 * x) →
  ∃ b : ℝ, b < 0 ∧ (∃ a : ℝ, (∀ x, y = a + b * x)) →
  (¬ ∀ x y, x = y) :=
by
  intro h
  sorry

end linear_regression_negatively_correlated_l341_341384


namespace total_climbing_time_l341_341824

noncomputable def time_for_first_flight : ℕ := 30
noncomputable def time_difference : ℕ := 8
noncomputable def number_of_flights : ℕ := 7

theorem total_climbing_time : 
  let a := time_for_first_flight,
      d := time_difference,
      n := number_of_flights,
      l := a + (n - 1) * d,
      S_n := n * (a + l) / 2
  in S_n = 378 := 
by 
  sorry

end total_climbing_time_l341_341824


namespace distinct_products_l341_341110

def largest_m (n k : ℕ) (hn : n > 0) (hk : k > 0) : ℕ :=
  n + k - 2

theorem distinct_products (n k : ℕ) (hn : n > 0) (hk : k > 0) :
  ∃ m, (∀ A B : finset ℕ, A.card = k → B.card = n → 
    (A.product B).image (λ p, p.1 * p.2) ⊇ finset.range (largest_m n k hn hk)) :=
begin
  use largest_m n k hn hk,
  sorry
end

end distinct_products_l341_341110


namespace pages_read_on_Sunday_l341_341454

def total_pages : ℕ := 93
def pages_read_on_Saturday : ℕ := 30
def pages_remaining_after_Sunday : ℕ := 43

theorem pages_read_on_Sunday : total_pages - pages_read_on_Saturday - pages_remaining_after_Sunday = 20 := by
  sorry

end pages_read_on_Sunday_l341_341454


namespace problem1_question_problem1_contrapositive_problem1_negation_problem2_question_problem2_contrapositive_problem2_negation_l341_341972

-- Proof statement for problem 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem problem1_question (x y : ℕ) (h : ¬(is_odd x ∧ is_odd y)) : is_odd (x + y) := sorry

theorem problem1_contrapositive (x y : ℕ) (h : is_odd x ∧ is_odd y) : ¬ is_odd (x + y) := sorry

theorem problem1_negation : ∃ (x y : ℕ), ¬(is_odd x ∧ is_odd y) ∧ ¬ is_odd (x + y) := sorry

-- Proof statement for problem 2

structure Square : Type := (is_rhombus : Prop)

def all_squares_are_rhombuses : Prop := ∀ (sq : Square), sq.is_rhombus

theorem problem2_question : all_squares_are_rhombuses = true := sorry

theorem problem2_contrapositive : ¬ all_squares_are_rhombuses = false := sorry

theorem problem2_negation : ¬(∃ (sq : Square), ¬ sq.is_rhombus) = false := sorry

end problem1_question_problem1_contrapositive_problem1_negation_problem2_question_problem2_contrapositive_problem2_negation_l341_341972


namespace necessary_and_sufficient_conditions_l341_341027

open Real

def cubic_has_arithmetic_sequence_roots (a b c : ℝ) : Prop :=
∃ x y : ℝ,
  (x - y) * (x) * (x + y) + a * (x^2 + x - y + x + y) + b * x + c = 0 ∧
  3 * x = -a

theorem necessary_and_sufficient_conditions
  (a b c : ℝ) (h : cubic_has_arithmetic_sequence_roots a b c) :
  2 * a^3 - 9 * a * b + 27 * c = 0 ∧ a^2 - 3 * b ≥ 0 :=
sorry

end necessary_and_sufficient_conditions_l341_341027


namespace range_of_a_l341_341474

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 1 ∧ y = a + (3 - a)/2 * x) →
  (a ∈ set.Icc (1 / 3) (3 / 2)) :=
by
  sorry

end range_of_a_l341_341474


namespace find_smaller_number_l341_341507

theorem find_smaller_number (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : x = 18 :=
by
  sorry

end find_smaller_number_l341_341507


namespace problem1_problem2_l341_341661

-- Problem 1
theorem problem1 : 3^2 * (-1 + 3) - (-16) / 8 = 20 :=
by decide  -- automatically prove simple arithmetic

-- Problem 2
variables {x : ℝ} (hx1 : x ≠ 1) (hx2 : x ≠ -1)

theorem problem2 : ((x^2 / (x + 1)) - (1 / (x + 1))) * (x + 1) / (x - 1) = x + 1 :=
by sorry  -- proof to be completed

end problem1_problem2_l341_341661


namespace ratio_of_ages_three_years_ago_l341_341573

theorem ratio_of_ages_three_years_ago (k Y_c : ℕ) (h1 : 45 - 3 = k * (Y_c - 3)) (h2 : (45 + 7) + (Y_c + 7) = 83) : (45 - 3) / (Y_c - 3) = 2 :=
by {
  sorry
}

end ratio_of_ages_three_years_ago_l341_341573


namespace hemisphere_surface_area_l341_341174

theorem hemisphere_surface_area (C : ℝ) (hC : C = 36) : 
  let r := 18 / Real.pi in 
  let A := 3 * π * r^2 in 
  A = 972 / π :=
by 
  -- The proof
  sorry

end hemisphere_surface_area_l341_341174


namespace disjoint_subset_remainder_l341_341845

open Finset

noncomputable def S : Finset ℕ := { n | n ∈ range 1 13 }.toFinset

theorem disjoint_subset_remainder :
  let n := (3:ℕ)^12 - 2 * (2:ℕ)^12 + 1 in
  n / 2 % 500 = 125 :=
by
  let n := (3:ℕ)^12 - 2 * (2:ℕ)^12 + 1
  have h : n / 2 % 500 = 125 := by sorry
  exact h

end disjoint_subset_remainder_l341_341845


namespace correct_proposition_B_l341_341646

def angle (θ : ℝ) : Prop := θ ∈ set.Icc 0 (π / 2)
def first_quadrant (θ : ℝ) : Prop := θ ∈ set.Icc 0 π
def same_terminal_side (θ₁ θ₂ : ℝ) : Prop := ∃ k : ℤ, θ₁ = θ₂ + 2 * k * π
def quadrant (θ : ℝ) : Prop := ∃ k : ℤ, θ = k * π

theorem correct_proposition_B (θ : ℝ) :
  (forall θ, angle θ → first_quadrant θ) ∧
  (forall θ, first_quadrant θ → ¬angle θ) ∧
  (forall θ₁ θ₂, same_terminal_side θ₁ θ₂ → θ₁ ≠ θ₂) ∧
  (forall θ, ¬quadrant θ) →
  (*B:*) (forall θ, angle θ → first_quadrant θ) :=
begin
  sorry
end

end correct_proposition_B_l341_341646


namespace polar_equation_of_line_segment_l341_341788

theorem polar_equation_of_line_segment :
  ∃ (ρ θ : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 ∧ ρ = 1 / (cos θ + sin θ) :=
by
  sorry

end polar_equation_of_line_segment_l341_341788


namespace flight_time_l341_341922

theorem flight_time (r v : ℝ) (h_r : r = 3000) (h_v : v = 600) : 
  (2 * real.pi * r) / v = 10 * real.pi :=
by
  rw [h_r, h_v]
  sorry

end flight_time_l341_341922


namespace kevin_hopped_distance_after_four_hops_l341_341826

noncomputable def kevin_total_hopped_distance : ℚ :=
  let hop1 := 1
  let hop2 := 1 / 2
  let hop3 := 1 / 4
  let hop4 := 1 / 8
  hop1 + hop2 + hop3 + hop4

theorem kevin_hopped_distance_after_four_hops :
  kevin_total_hopped_distance = 15 / 8 :=
by
  sorry

end kevin_hopped_distance_after_four_hops_l341_341826


namespace common_root_sum_k_l341_341952

theorem common_root_sum_k :
  (∃ x : ℝ, (x^2 - 4 * x + 3 = 0) ∧ (x^2 - 6 * x + k = 0)) → 
  (∃ (k₁ k₂ : ℝ), (k₁ = 5) ∧ (k₂ = 9) ∧ (k₁ + k₂ = 14)) :=
by
  sorry

end common_root_sum_k_l341_341952


namespace number_playing_two_or_more_l341_341800

noncomputable def total_people : ℕ := 800
noncomputable def fraction_play_one_instrument : ℚ := 1/5
noncomputable def probability_exactly_one_instrument : ℚ := 0.16
noncomputable def number_playing_at_least_one := fraction_play_one_instrument * total_people
noncomputable def number_playing_exactly_one := probability_exactly_one_instrument * number_playing_at_least_one

theorem number_playing_two_or_more : (number_playing_at_least_one.to_int - number_playing_exactly_one.to_int) = 135 := by
  sorry

end number_playing_two_or_more_l341_341800


namespace find_N_l341_341842

-- Define the sets S and T
def S : Set ℕ := {k | ∃ n : ℕ, k = 2^n ∧ n ≤ 12}
def T : Set ℕ := {k | ∃ n : ℕ, k = 2^n - 1 ∧ n ≤ 12}

-- Define the function that calculates the total sum N of positive differences
def calculate_total_sum (S T : Set ℕ) : ℕ :=
  let positive_diffs_S := {d | ∃ x ∈ S, ∃ y ∈ S, x > y ∧ d = x - y},
      positive_diffs_T := {d | ∃ x ∈ T, ∃ y ∈ T, x > y ∧ d = x - y} in
  positive_diffs_S.sum + positive_diffs_T.sum

-- Declare the main theorem to prove
theorem find_N : ∃ N : ℕ, calculate_total_sum S T = N := sorry

end find_N_l341_341842


namespace booth_earnings_after_5_days_l341_341613

def booth_daily_popcorn_earnings := 50
def booth_daily_cotton_candy_earnings := 3 * booth_daily_popcorn_earnings
def booth_total_daily_earnings := booth_daily_popcorn_earnings + booth_daily_cotton_candy_earnings
def booth_total_expenses := 30 + 75

theorem booth_earnings_after_5_days :
  5 * booth_total_daily_earnings - booth_total_expenses = 895 :=
by
  sorry

end booth_earnings_after_5_days_l341_341613


namespace ellipse_equation_max_triangle_area_point_existence_l341_341021

-- Given conditions
def vertex : ℝ × ℝ := (0, -Real.sqrt 5)
def major_axis_orientation : Bool := true  -- major axis along x-axis
def ellipse_center : ℝ × ℝ := (0, 0)
def eccentricity : ℝ := Real.sqrt 6 / 6

-- Question 1: Equation of the ellipse
theorem ellipse_equation : ∃ a b : ℝ, (Real.sqrt 6 = a) ∧ (Real.sqrt 5 = b) ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (a = Real.sqrt 6) ∧ (b = Real.sqrt 5)) :=
sorry

-- Question 2: Maximum area of triangle MF_1F_2
theorem max_triangle_area : ∀ M : ℝ × ℝ, (M.1^2 / 6 + M.2^2 / 5 = 1) → ∃ S : ℝ, S = Real.sqrt 5 :=
sorry

-- Question 3: Existence of point P such that PF_1 ⋅ PF_2 = 0
theorem point_existence : ¬ ∃ P : ℝ × ℝ, (P.1^2 / 6 + P.2^2 / 5 = 1) ∧ ((P.1 - Real.sqrt 6 / 2) * (P.1 + Real.sqrt 6 / 2) + P.2^2 = 0) :=
sorry

end ellipse_equation_max_triangle_area_point_existence_l341_341021


namespace find_angle_A_and_area_l341_341836

section
variables (A B C : Real) (a b c : Real)

-- Given conditions
def conditions (α β γ : Real) (a b c : Real) :=
  α = B ∧
  β = C ∧
  γ = A ∧
  cos β * cos γ - sin β * sin γ = 1 / 2 ∧
  a = 2 * sqrt 3 ∧
  b + c = 4 ∧
  cos γ = -1 / 2

-- Problem statement
theorem find_angle_A_and_area (α β γ : Real) (a b c : Real)
  (h : conditions α β γ a b c) :
  A = 2 * Real.pi / 3 ∧
  (1 / 2 * b * c * sin A = sqrt 3) :=
sorry
end

end find_angle_A_and_area_l341_341836


namespace solve_inequality_l341_341342

-- Definition of the original condition
def num (x : ℝ) : ℝ := x^2 - 9
def denom (x : ℝ) : ℝ := x^2 - 4
def rational_expr (x : ℝ) : ℝ := num x / denom x

-- The Lean theorem to prove the inequality and its solution set
theorem solve_inequality (x : ℝ) : 
  (rational_expr x > 0) ↔ (x < -3 ∨ x > 3) := 
sorry

end solve_inequality_l341_341342


namespace number_of_correct_propositions_l341_341368

-- Define conditions as propositions
def prop1 : Prop := ∀ (P A B C : Point), plane_contains A ∧ plane_contains B ∧ plane_contains C → plane_contains P
def prop2 : Prop := ∀ (P A B C : Point) (α : Plane), ¬plane_contains α P ∧ plane_contains α A ∧ plane_contains α B ∧ plane_contains α C → ¬plane_contains_single_plane P A B C
def prop3 : Prop := ∀ (L1 L2 L3 : Line), pairwise_intersect L1 L2 ∧ pairwise_intersect L2 L3 ∧ pairwise_intersect L3 L1 → coplanar L1 L2 L3
def prop4 : Prop := ∀ (Q : Quadrilateral), has_equal_opposite_sides Q → parallelogram Q

-- Define question as a proposition
def correct_propositions : Nat := 0

-- Main theorem statement
theorem number_of_correct_propositions : 
  (prop1 ∧ prop2 ∧ prop3 ∧ prop4) = (correct_propositions = 0) :=
sorry

end number_of_correct_propositions_l341_341368


namespace necessary_not_sufficient_l341_341713

theorem necessary_not_sufficient (a b : ℝ) : (a > b - 1) ∧ ¬ (a > b - 1 → a > b) := 
sorry

end necessary_not_sufficient_l341_341713


namespace intersection_M_N_l341_341401

open Set Real

def M := {x : ℝ | x^2 + x - 6 < 0}
def N := {x : ℝ | abs (x - 1) ≤ 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l341_341401


namespace nancy_pictures_l341_341235

theorem nancy_pictures (z m b d : ℕ) (hz : z = 120) (hm : m = 75) (hb : b = 45) (hd : d = 93) :
  (z + m + b) - d = 147 :=
by {
  -- Theorem definition capturing the problem statement
  sorry
}

end nancy_pictures_l341_341235


namespace John_has_15_snakes_l341_341456

theorem John_has_15_snakes (S : ℕ)
  (H1 : ∀ M, M = 2 * S)
  (H2 : ∀ M L, L = M - 5)
  (H3 : ∀ L P, P = L + 8)
  (H4 : ∀ P D, D = P / 3)
  (H5 : S + (2 * S) + ((2 * S) - 5) + (((2 * S) - 5) + 8) + (((((2 * S) - 5) + 8) / 3)) = 114) :
  S = 15 :=
by sorry

end John_has_15_snakes_l341_341456


namespace marble_problem_l341_341625

-- Defining the problem in Lean statement
theorem marble_problem 
  (m : ℕ) (n k : ℕ) (hx : m = 220) (hy : n = 20) : 
  (∀ x : ℕ, (k = n + x) → (m / n = 11) → (m / k = 10)) → (x = 2) :=
by {
  sorry
}

end marble_problem_l341_341625


namespace speed_of_middle_point_l341_341266

theorem speed_of_middle_point (v1 v2 : ℝ) (h_v1 : v1 = 5) (h_v2 : v2 = 4) :
    let v_perp_A := Real.sqrt (v1^2 - v2^2)
    let v_perp_B := v_perp_A / 2
    let v_B := Real.sqrt (v2^2 + v_perp_B^2)
    v_B ≈ 4.3 :=
by
  sorry

end speed_of_middle_point_l341_341266


namespace find_common_difference_l341_341749

noncomputable def common_difference (a₁ d : ℤ) : Prop :=
  let a₂ := a₁ + d
  let a₃ := a₁ + 2 * d
  let S₅ := 5 * a₁ + 10 * d
  a₂ + a₃ = 8 ∧ S₅ = 25 → d = 2

-- Statement of the proof problem
theorem find_common_difference (a₁ d : ℤ) (h : common_difference a₁ d) : d = 2 :=
by sorry

end find_common_difference_l341_341749


namespace triple_f_of_3_l341_341045

def f (x : ℤ) : ℤ := -3 * x + 5

theorem triple_f_of_3 : f (f (f 3)) = -46 := by
  sorry

end triple_f_of_3_l341_341045


namespace sum_lcm_180_eq_292_l341_341593

noncomputable def sum_of_positive_integers_lcm_180 : ℕ :=
∑ ν in (Finset.filter (λ (ν : ℕ), Nat.lcm ν 45 = 180 ∧ ν > 0) (Finset.range 181)), ν

theorem sum_lcm_180_eq_292 : sum_of_positive_integers_lcm_180 = 292 := 
by
  sorry

end sum_lcm_180_eq_292_l341_341593


namespace minimum_value_point_l341_341786

-- Define the function f(x) for x > 2
def f (x : ℝ) : ℝ := x + 1 / (x - 2)

-- Define the condition that x > 2
def domain_condition (x : ℝ) : Prop := x > 2

-- Define the goal, which is to find the minimum value point of the function f(x)
theorem minimum_value_point :
  ∃ a > 2, (∀ x > 2, f a ≤ f x) ∧ a = 3 := sorry

end minimum_value_point_l341_341786


namespace find_functional_eq_solution_l341_341341

theorem find_functional_eq_solution :
  ∀ (f : ℝ → ℝ),
    (∀ x y z : ℝ, x > y → y > z → z > 0 → 
      f(x - y + z) = f(x) + f(y) + f(z) - x * y - y * z + x * z) → 
    (∀ x : ℝ, x > 0 → f(x) = x ^ 2 / 2) :=
by
  -- Since we only need the statement, the proof is omitted
  sorry

end find_functional_eq_solution_l341_341341


namespace cyclic_quadrilateral_square_l341_341896

open Real

theorem cyclic_quadrilateral_square
  (A B C D : Point) (O : Point) (R : ℝ) (hR : R = 1)
  (h_cyclic : ∀ P₁ P₂, angle (O - A) (O - P₁) + angle (O - P₂) (O - P₁) = π)
  (h_product : AB * BC * CD * DA ≥ 4) :
  is_square A B C D :=
sorry

end cyclic_quadrilateral_square_l341_341896


namespace dog_greatest_distance_is_15_l341_341877

noncomputable def greatest_distance_from_origin : Real :=
  let dog_position := (4, 3) : ℝ × ℝ
  let rope_length := 10 : ℝ
  let origin := (0, 0) : ℝ × ℝ
  let distance_to_center := Real.sqrt((dog_position.1 - origin.1)^2 + (dog_position.2 - origin.2)^2)
  distance_to_center + rope_length

theorem dog_greatest_distance_is_15 : greatest_distance_from_origin = 15 := by
  sorry

end dog_greatest_distance_is_15_l341_341877


namespace largest_connected_groups_four_l341_341269

-- Define the space station with 25 chambers and a combination of bidirectional and one-way tunnels
def space_station : SimpleGraph (Fin 25) :=
{ adj := λ x y => sorry, -- Adjacency relation (to be defined precisely in a real proof)
  sym := sorry, -- Symmetry condition (to be defined for undirected edges)
  loopless := sorry } -- Loopless condition (no loops)

-- Define the property of a group of four chambers being connected
def connected_group_of_four (s : Finset (Fin 25)) (G : SimpleGraph (Fin 25)) : Prop :=
  s.card = 4 ∧ G.induced_subgraph s.is_connected

-- Define the main theorem stating the largest number of connected groups of four chambers
theorem largest_connected_groups_four :
  ∃ (s : Finset (Fin 25)), (connected_group_of_four s space_station) ∧ s.card = 9650 :=
sorry

end largest_connected_groups_four_l341_341269


namespace polynomial_ratio_condition_l341_341673

theorem polynomial_ratio_condition (α : ℝ) :
  (∃ P Q : polynomial ℝ, (∀ x, P.coeff x ≥ 0) ∧ (∀ x, Q.coeff x ≥ 0) ∧ (P / Q = polynomial.C (1:ℝ) * (X^2 - (polynomial.C α) * X + polynomial.C (1:ℝ)))) ↔ α < 2 := by
  sorry

end polynomial_ratio_condition_l341_341673


namespace harrys_fish_count_l341_341141

theorem harrys_fish_count : 
  let sam_fish := 7
  let joe_fish := 8 * sam_fish
  let harry_fish := 4 * joe_fish
  harry_fish = 224 :=
by
  sorry

end harrys_fish_count_l341_341141


namespace set_equality_l341_341746

namespace ProofProblem

-- Define the set equality and the question to be proved
theorem set_equality (a b : ℚ) (h1 : {1, a, b / a} = {0, a^2, a + b}) : a ^ 2013 + b ^ 2012 = -1 := by
  sorry

end ProofProblem

end set_equality_l341_341746


namespace math_proof_problems_l341_341353

open Real

noncomputable def problem1 (α : ℝ) : Prop :=
  (sin (π - α) - 2 * sin (π / 2 + α) = 0) → (sin α * cos α + sin α ^ 2 = 6 / 5)

noncomputable def problem2 (α β : ℝ) : Prop :=
  (tan (α + β) = -1) → (tan α = 2) → (tan β = 3)

-- Example of how to state these problems as a theorem
theorem math_proof_problems (α β : ℝ) : problem1 α ∧ problem2 α β := by
  sorry

end math_proof_problems_l341_341353


namespace sum_base5_is_2112_l341_341701

noncomputable def base5toNat (n : ℕ) : ℕ :=
  let digits := n.digits 5
  digits.foldl (λ acc digit, acc * 5 + digit) 0

def summation_base5_eq : Prop :=
  base5toNat 1234 + base5toNat 234 + base5toNat 34 = base5toNat 2112

theorem sum_base5_is_2112 : summation_base5_eq :=
by
  sorry

end sum_base5_is_2112_l341_341701


namespace polynomial_proof_l341_341340

noncomputable def Q (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 3

theorem polynomial_proof : Q (real.cbrt 3 + 1) = 0 ∧ 
                           (∀ (x : ℝ), Q(x).leadingCoeff = 1 ∧ 
                             (∀ (a b c : ℤ), Q(a + b * x + c * x^2).denom = 1)) :=
by
  sorry

end polynomial_proof_l341_341340


namespace tangent_line_equation_l341_341019

noncomputable def f (x : ℝ) : ℝ := sorry

theorem tangent_line_equation
  (h : ∀ (x : ℝ), f (1 - x) - 2 * f x = x^2 - 1) :
  let y := f
  in (8, -3, 5) : (ℝ × ℝ × ℝ)
    ∃ (y : ℝ → ℝ),
        ∀ (m : ℝ) (c : ℝ → ℝ → ℝ) (p : ℝ × ℝ),
          m = (λ (x : ℝ), -2 * x + 2 / 3) (p.siwas.right)
        → c = λ m x, m * x
        → p = (-1, y (-1))
        → c m p.1 + y p.2 = 5 :=
by sorry

end tangent_line_equation_l341_341019


namespace min_diagonal_length_l341_341918

theorem min_diagonal_length (a b : ℝ) (h : a + b = 10) : 
  (λ d : ℝ, ∃ a b : ℝ, (a + b = 10) ∧ d^2 = a^2 + b^2 ∧ d = Real.sqrt 50).nonempty := 
by sorry

end min_diagonal_length_l341_341918


namespace length_PR_of_circle_l341_341510

theorem length_PR_of_circle (r : ℝ) (PQ : ℝ) (PR : ℝ) : 
  ∀ (P Q R : Point)
  (center : Point)
  (midpoint_R : is_midpoint_minor_arc P Q R)
  (on_circle : is_circle center r)
  (P_on_circle : is_on_circle P on_circle)
  (Q_on_circle : is_on_circle Q on_circle)
  (PQ_length : dist P Q = PQ),
  r = 10 →
  PQ = 12 →
  PR = 2 * sqrt 10 :=
by
  intros
  sorry

end length_PR_of_circle_l341_341510


namespace product_first_8_terms_l341_341015

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a_2 : a 2 = 3 := sorry
def a_7 : a 7 = 1 := sorry

-- Proof statement
theorem product_first_8_terms (h_geom : is_geometric_sequence a q) 
  (h_a2 : a 2 = 3) 
  (h_a7 : a 7 = 1) : 
  (a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 = 81) :=
sorry

end product_first_8_terms_l341_341015


namespace imaginary_part_of_conjugate_l341_341537

open Complex

def z : ℂ := Complex.i * (3 - Complex.i)

theorem imaginary_part_of_conjugate : (z.conj).im = -3 :=
by
  sorry

end imaginary_part_of_conjugate_l341_341537


namespace no_x_squared_term_l341_341790

theorem no_x_squared_term {m : ℚ} (h : (x+1) * (x^2 + 5*m*x + 3) = x^3 + (5*m + 1)*x^2 + (3 + 5*m)*x + 3) : 
  5*m + 1 = 0 → m = -1/5 := by sorry

end no_x_squared_term_l341_341790


namespace find_range_of_a_l341_341766

noncomputable def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 + a * x + a + 3 = 0 }

theorem find_range_of_a (a : ℝ) :
  B(a) ⊆ A → -2 ≤ a ∧ a < 6 := 
sorry

end find_range_of_a_l341_341766


namespace other_person_time_to_complete_job_l341_341528

-- Define the conditions
def SureshTime : ℕ := 15
def SureshWorkHours : ℕ := 9
def OtherPersonWorkHours : ℕ := 4

-- The proof problem: Prove that the other person can complete the job in 10 hours.
theorem other_person_time_to_complete_job (x : ℕ) 
  (h1 : ∀ SureshWorkHours SureshTime, SureshWorkHours * (1 / SureshTime) = (SureshWorkHours / SureshTime) ∧ 
       4 * (SureshWorkHours / SureshTime / 4) = 1) : 
  (x = 10) :=
sorry

end other_person_time_to_complete_job_l341_341528


namespace angle_MON_l341_341733

theorem angle_MON (O M N : ℝ × ℝ) (D : ℝ) :
  (O = (0, 0)) →
  (M = (-2, 2)) →
  (N = (2, 2)) →
  (x^2 + y^2 + D * x - 4 * y = 0) →
  (D = 0) →
  ∃ θ : ℝ, θ = 90 :=
by
  sorry

end angle_MON_l341_341733


namespace no_tetrahedron_with_given_heights_l341_341317

theorem no_tetrahedron_with_given_heights (h1 h2 h3 h4 : ℝ) (V : ℝ) (V_pos : V > 0)
    (S1 : ℝ := 3*V) (S2 : ℝ := (3/2)*V) (S3 : ℝ := V) (S4 : ℝ := V/2) :
    (h1 = 1) → (h2 = 2) → (h3 = 3) → (h4 = 6) → ¬ ∃ (S1 S2 S3 S4 : ℝ), S1 < S2 + S3 + S4 := by
  intros
  sorry

end no_tetrahedron_with_given_heights_l341_341317


namespace number_of_valid_colorings_l341_341146

-- Defining the color type with two possible values: red or blue
inductive Color
| red
| blue

open Color

-- Defining the conditions as given in the problem
def isValidColoring (color : ℤ → Color) : Prop :=
  (∀ x k : ℤ, color x = color (x + 7 * k)) ∧
  (color 20 = red) ∧ (color 14 = red) ∧
  (color 71 = blue) ∧ (color 143 = blue)

-- Proving the number of valid colorings is 8
theorem number_of_valid_colorings : 
  ∃ f : ℤ → Color, isValidColoring f ∧
  (∃ l : List (ℤ → Color), list.length l = 7 ∧ (∀ g ∈ l, isValidColoring g) ∧
  (∀ g1 g2 ∈ l, g1 ≠ g2) ∧ Henry.master = list.nodup l) :=
sorry

end number_of_valid_colorings_l341_341146


namespace cubic_equation_three_distinct_real_roots_l341_341708

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x^2 - a

theorem cubic_equation_three_distinct_real_roots (a : ℝ) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃
  ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ↔ -4 < a ∧ a < 0 :=
sorry

end cubic_equation_three_distinct_real_roots_l341_341708


namespace equilateral_triangle_l341_341807

noncomputable def midpoint (A B : Point) : Point := sorry
def distance_to_line (P A B : Point) : ℝ := sorry
def length (A B : Point) : ℝ := sorry

structure Triangle :=
(A B C : Point)

def is_midpoint (P A B : Point) : Prop := sorry

theorem equilateral_triangle (T : Triangle) [h1 : is_midpoint (midpoint T.B T.C) T.B T.C]
  [h2 : is_midpoint (midpoint T.A T.C) T.A T.C]
  [h3 : is_midpoint (midpoint T.A T.B) T.A T.B]
  (h4 : length T.A (midpoint T.B T.C) = distance_to_line (midpoint T.B T.C) T.A T.B +
    distance_to_line (midpoint T.B T.C) T.A T.C)
  (h5 : length T.B (midpoint T.A T.C) = distance_to_line (midpoint T.A T.C) T.B T.A +
    distance_to_line (midpoint T.A T.C) T.B (midpoint T.C T.B)) 
  (h6 : length T.C (midpoint T.A T.B) = distance_to_line (midpoint T.A T.B) T.C T.A +
    distance_to_line (midpoint T.A T.B) T.C (midpoint T.B T.C)) :
  length T.A T.B = length T.B T.C ∧ length T.B T.C = length T.C T.A := sorry

end equilateral_triangle_l341_341807


namespace cylinder_volume_l341_341258

theorem cylinder_volume (h : Real := 2) (A : Real := 4 * Real.pi) : 
  let r := A / (2 * Real.pi * h) in
  let V := Real.pi * r^2 * h in
  V = 2 * Real.pi := by
  sorry

end cylinder_volume_l341_341258


namespace train_crosses_bridge_in_30_seconds_l341_341639

theorem train_crosses_bridge_in_30_seconds :
  ∀ (length_of_train length_of_bridge : ℕ) (speed_of_train_kph : ℕ),
  length_of_train = 80 →
  length_of_bridge = 295 →
  speed_of_train_kph = 45 →
  train_crosses_bridge_in_seconds length_of_train length_of_bridge speed_of_train_kph = 30 :=
begin
  sorry
end

noncomputable def train_crosses_bridge_in_seconds (length_of_train length_of_bridge speed_of_train_kph : ℕ) : ℕ :=
let speed_of_train_mps := (speed_of_train_kph * 1000) / 3600 in
let total_distance := length_of_train + length_of_bridge in
total_distance / speed_of_train_mps

end train_crosses_bridge_in_30_seconds_l341_341639


namespace distance_A_to_C_through_B_l341_341176

-- Define the distances on the map
def Distance_AB_map : ℝ := 20
def Distance_BC_map : ℝ := 10

-- Define the scale of the map
def scale : ℝ := 5

-- Define the actual distances
def Distance_AB := Distance_AB_map * scale
def Distance_BC := Distance_BC_map * scale

-- Define the total distance from A to C through B
def Distance_AC_through_B := Distance_AB + Distance_BC

-- Theorem to be proved
theorem distance_A_to_C_through_B : Distance_AC_through_B = 150 := by
  sorry

end distance_A_to_C_through_B_l341_341176


namespace find_c_of_parabola_l341_341903

theorem find_c_of_parabola (a b c : ℝ) (h_eqn : ∀ y : ℝ, (5, 3) = (a * (y - 3)^2 + 3 * y + c)) 
  (h_vertex : (5, 3) = (a * (3-3)^2 + b * 3 + c))
  (h_passes : (3, 5) = (a * 5^2 + b * 5 + c))
  (h_a : a = -1) : 
  c = -4 :=
by
  sorry

end find_c_of_parabola_l341_341903


namespace find_a_l341_341750

-- Definition of the function f
def f (x : ℝ) : ℝ :=
  x * Real.sin x + 5

-- Derivative of f
def f' (x : ℝ) : ℝ :=
  Real.sin x + x * Real.cos x

-- The point of tangency
def x_perp : ℝ := Real.pi / 2

-- Slope of the tangent line at x = π/2
def slope_of_tangent_at_x_perp : ℝ :=
  f' x_perp

-- Equation of the line
def line_slope (a : ℝ) : ℝ :=
  -a / 4

-- The proof goal
theorem find_a (a : ℝ) (h : slope_of_tangent_at_x_perp * line_slope a = -1) : a = 4 :=
by
  -- [Add proof details here or skip with sorry.]
  sorry

end find_a_l341_341750


namespace river_width_l341_341939

theorem river_width (w : ℕ) (speed_const : ℕ) 
(meeting1_from_nearest_shore : ℕ) (meeting2_from_other_shore : ℕ)
(h1 : speed_const = 1) 
(h2 : meeting1_from_nearest_shore = 720) 
(h3 : meeting2_from_other_shore = 400)
(h4 : 3 * w = 3 * meeting1_from_nearest_shore)
(h5 : 2160 = 2 * w - meeting2_from_other_shore) :
w = 1280 :=
by
  {
      sorry
  }

end river_width_l341_341939


namespace function_increasing_and_min_value_l341_341852

noncomputable def f (x : ℝ) : ℝ := sorry
def f' (x : ℝ) : ℝ := sorry

theorem function_increasing_and_min_value :
  (∀ x > 0, f'(x) * Real.exp(-x) > 1) →
  (∀ x > 0, f(x) - Real.exp(x) = (f(x) - Real.exp(x))) →
  f (Real.log x) ≥ x + Real.sqrt Real.exp 1 →
  f(1/2) = 2 * Real.sqrt Real.exp 1 →
  (∀ x > 0, f(x) - Real.exp(x) increases on (0, ∞)) ∧ 
  ∃ x, x ≥ Real.sqrt (Real.exp 1) :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end function_increasing_and_min_value_l341_341852


namespace no_partition_possible_l341_341085

theorem no_partition_possible :
  ¬∃ (P : Type) (l w : P → ℕ), 
    (∀ (p : P), 2 * (l p + w p) = 18 ∨ 2 * (l p + w p) = 22 ∨ 2 * (l p + w p) = 26) ∧
    (∑ p, l p * w p = 35 * 35) :=
by
  sorry

end no_partition_possible_l341_341085


namespace complex_equation_solution_l341_341017

theorem complex_equation_solution (x y : ℝ) (i : ℂ) (h₁ : i * i = -1) (h₂ : x * i - y = -1 + i) : 
  (1 - i) * (x - y * i) = -2 * i := 
begin
  sorry
end

end complex_equation_solution_l341_341017


namespace dihedral_angle_l341_341861

-- Definitions based on the given problem conditions
def A := (0, 0, 0)
def A1 := (0, 0, 1)
def B := (4, 0, 0)
def C (theta : ℝ) := (4 * cos theta, 4 * sin theta, 0)
def B1 := (4, 0, 1)
def angleBAC := 60 * (π / 180) -- 60 degrees in radians

-- Statement: Prove that the dihedral angle between plane A1CB1 and the base of the cylinder AB is 30 degrees.
theorem dihedral_angle 
  (AA1 : ℝ = 1) 
  (AB : ℝ = 4)
  (angleBAC : ℝ = 60 * (π / 180))  -- 60 degrees in radians
  : dihedral_angle_of_planes (plane A1 C(θ) B1) (plane A B C(θ)) = 30 * (π / 180) := 
sorry

end dihedral_angle_l341_341861


namespace oplus_calculation_l341_341847

def my_oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem oplus_calculation : my_oplus 2 3 = 23 := 
by
    sorry

end oplus_calculation_l341_341847


namespace real_number_values_m_pure_imaginary_values_m_l341_341492

namespace Proof

def is_real_number (z : ℂ) : Prop :=
  z.im = 0

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

def complex_number (m : ℝ) : ℂ :=
  (m^2 - m - 2) + (m^2 + 3*m + 2)*complex.i

theorem real_number_values_m :
  ∀ m : ℝ, (is_real_number (complex_number m) ↔ m = -1 ∨ m = -2) := 
by
  intro m
  sorry

theorem pure_imaginary_values_m :
  ∀ m : ℝ, (is_pure_imaginary (complex_number m) ↔ m = 2) :=
by
  intro m
  sorry

end Proof

end real_number_values_m_pure_imaginary_values_m_l341_341492


namespace origin_not_in_A_point_M_in_A_l341_341543

def set_A : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ x + 2 * y - 1 ≥ 0 ∧ y ≤ x + 2 ∧ 2 * x + y - 5 ≤ 0}

theorem origin_not_in_A : (0, 0) ∉ set_A := by
  sorry

theorem point_M_in_A : (1, 1) ∈ set_A := by
  sorry

end origin_not_in_A_point_M_in_A_l341_341543


namespace kayla_scored_3_on_level_2_l341_341457

/--
Kayla scored 3 points on the second level given the following conditions:
1. Kayla scored 2 points on level 1.
2. Kayla scored 5 points on level 3.
3. Kayla scored 8 points on level 4.
4. Kayla scored 12 points on level 5.
5. Kayla will score 17 points on level 6.
-/
theorem kayla_scored_3_on_level_2 : 
  ∃ (points_on_level_2 : ℕ), 
    let points : ℕ → ℕ := λ n, if n = 1 then 2 else 
                              if n = 3 then 5 else 
                              if n = 4 then 8 else 
                              if n = 5 then 12 else 
                              if n = 6 then 17 else 
                              points_on_level_2 in
    points 2 = 3 :=
by sorry

end kayla_scored_3_on_level_2_l341_341457


namespace verify_option_a_l341_341968

-- Define Option A's condition
def option_a_condition (a : ℝ) : Prop :=
  2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2

-- State the theorem that Option A's factorization is correct
theorem verify_option_a (a : ℝ) : option_a_condition a := by sorry

end verify_option_a_l341_341968


namespace TobiasChargePerDrivewayShoveling_l341_341931

noncomputable def TobiasAllowancePerMonth : ℕ := 5
noncomputable def TobiasMonthsSaving : ℕ := 3
noncomputable def TobiasAllowanceTotal : ℕ := TobiasAllowancePerMonth * TobiasMonthsSaving

noncomputable def TobiasChargePerLawn : ℕ := 15
noncomputable def TobiasLawnsMowed : ℕ := 4
noncomputable def TobiasMoneyFromMowingLawns : ℕ := TobiasChargePerLawn * TobiasLawnsMowed

noncomputable def TobiasDrivewaysShoveled : ℕ := 5
noncomputable def TobiasTotalMoneyBeforeBuyingShoes : ℕ := 95 + 15

theorem TobiasChargePerDrivewayShoveling : ∃ x : ℕ, 15 + 60 + 5 * x = 110 ∧ x = 7 :=
by
  use 7
  split
  case h.left => 
    calc
      15 + 60 + 5 * 7 = 75 + 5 * 7 := by rfl
      ... = 75 + 35 := by rfl
      ... = 110 := by rfl
  case h.right => rfl

end TobiasChargePerDrivewayShoveling_l341_341931


namespace expression_value_is_one_l341_341538

def sequence (n : ℕ) : ℕ :=
  Nat.rec 1 (λ _, Nat.casesOn _ 2 (λ _ x, sequence _ * 5 + sequence (_ - 1))) n

def integer_part (x : ℝ) : ℕ := floor x
def fractional_part (x : ℝ) : ℝ := x - floor x

def expression_value : ℝ :=
  integer_part ((sequence 2 : ℝ) / (sequence 1 : ℝ)) *
  (list.prod (list.map (λ n, 
    (fractional_part (integer_part ((sequence (n + 3) : ℝ) / (sequence (n + 2) : ℝ)))) : ℝ))
    (list.range 2022))

theorem expression_value_is_one :
  expression_value = 1 :=
by
  sorry

end expression_value_is_one_l341_341538


namespace rectangle_inscription_l341_341084

theorem rectangle_inscription (circle : Type) (A B : circle) :
  ∃ n : ℕ ∪ {∞}, n = number_of_possible_rectangles circle A B 
    ∧ n >= 0 
    ∧ n <= ∞ := 
sorry

end rectangle_inscription_l341_341084


namespace three_equal_parts_l341_341720

open Real

noncomputable def parallelogram {A B C D : Point} (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ D) (h4 : D ≠ A) : Prop :=
(IsCollinear B C D ∧ IsCollinear A B D ∧ distance A B = distance C D ∧ distance A D = distance B C)

noncomputable def midpoint {A B E : Point} (h : E = (A + B) / 2) : Prop := true

noncomputable def intersection (A B M : Point) (h : line A B ∩ line M = M) : Prop := true

axiom existence {A B C D E F M N : Point}
  (h_parallelogram : parallelogram A B C D)
  (h_midpointE : midpoint B C E)
  (h_midpointF : midpoint C D F)
  (h_intersectionM : intersection A E M)
  (h_intersectionN : intersection A F N) :
  (distance B M = distance M N) ∧ (distance M N = distance N D)

theorem three_equal_parts {A B C D E F M N : Point}
  (h_parallelogram : parallelogram A B C D)
  (h_midpointE : midpoint B C E)
  (h_midpointF : midpoint C D F)
  (h_intersectionM : intersection A E M)
  (h_intersectionN : intersection A F N) :
  (distance B M = distance M N) ∧ (distance M N = distance N D) :=
sorry

end three_equal_parts_l341_341720


namespace minimum_value_of_K_l341_341131

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / Real.exp x

noncomputable def f_K (K x : ℝ) : ℝ :=
  if f x ≤ K then f x else K

theorem minimum_value_of_K :
  (∀ x > 0, f_K (1 / Real.exp 1) x = f x) → (∃ K : ℝ, K = 1 / Real.exp 1) :=
by
  sorry

end minimum_value_of_K_l341_341131


namespace probability_vowel_initials_l341_341137

/-- Mrs. Johnson teaches a special 30-student seminar where each student's initial begins with the same letter.
    The vowels are considered A, E, I, O, U, Y, and W. 
    Prove that the probability of randomly selecting a student whose initials are among these vowels is 7/30. -/
theorem probability_vowel_initials : 
  let students := 30
  let vowels := {'A', 'E', 'I', 'O', 'U', 'Y', 'W'}
  let total_initials := 26
  let num_vowel_initials := Set.card vowels
  students ∈ ℕ ∧ total_initials ∈ ℕ ∧ num_vowel_initials ∈ ℕ ∧
  num_vowel_initials / students = 7 / 30 :=
by
  sorry

end probability_vowel_initials_l341_341137


namespace fraction_to_decimal_l341_341691

theorem fraction_to_decimal (n d : ℕ) (hn : n = 53) (hd : d = 160) (gcd_nd : Nat.gcd n d = 1)
  (prime_factorization_d : ∃ k l : ℕ, d = 2^k * 5^l) : ∃ dec : ℚ, (n:ℚ) / (d:ℚ) = dec ∧ dec = 0.33125 :=
by sorry

end fraction_to_decimal_l341_341691


namespace new_students_joined_l341_341171

theorem new_students_joined 
  (original_avg_age : ℕ) (new_students_avg_age : ℕ) (decrease_in_avg : ℕ) (original_students : ℕ)
  (h1 : original_avg_age = 40)
  (h2 : new_students_avg_age = 32)
  (h3 : decrease_in_avg = 4)
  (h4 : original_students = 12) :
  let x := 12 in
  (original_students * original_avg_age + x * new_students_avg_age = 36 * (original_students + x)) :=
by
  sorry

end new_students_joined_l341_341171


namespace quadrilateral_angles_equal_l341_341803

variables (a a' α : ℝ)
variables (C B A A' B' F D G E : Type)

def right_angle_quadrilateral (AB AB' : ℝ) (α : ℝ) : Prop :=
  ∀ (C B A A' B' F D G E : Type), 
  C = A ∧ B = E ∧ A = G ∧ A' = D ∧ B' = F ∧ A' = E ∧ α = 45

def distances_and_angles_equal :=
  ∀ (a a' : ℝ) (α : ℝ),
  (right_angle_quadrilateral (2 * a) (2 * a') α) →
  ∀ (C B A A' B' F D G E : Type),
  (∠ C B A = ∠ C B' A)

theorem quadrilateral_angles_equal 
(a a' : ℝ) (α : ℝ) (h : right_angle_quadrilateral (2 * a) (2 * a') α) :
∠ C B A = ∠ C B' A :=
by {
  sorry
}

end quadrilateral_angles_equal_l341_341803


namespace find_monic_cubic_polynomial_l341_341337

/-- There exists a monic cubic polynomial Q(x) with integer coefficients such that Q(∛3 + 1) = 0
    and Q(x) = x^3 - 3x^2 + 3x - 4. -/
theorem find_monic_cubic_polynomial :
  ∃ Q : Polynomial ℤ, Polynomial.monic Q ∧ Q.eval (Real.cbrt 3 + 1) = 0 ∧
    Q = Polynomial.X^3 - 3 * Polynomial.X^2 + 3 * Polynomial.X - 4 :=
by
  sorry

end find_monic_cubic_polynomial_l341_341337


namespace average_price_per_book_l341_341601

theorem average_price_per_book
  (amount_spent_first_shop : ℕ)
  (amount_spent_second_shop : ℕ)
  (books_first_shop : ℕ)
  (books_second_shop : ℕ)
  (total_amount_spent : ℕ := amount_spent_first_shop + amount_spent_second_shop)
  (total_books_bought : ℕ := books_first_shop + books_second_shop)
  (average_price : ℕ := total_amount_spent / total_books_bought) :
  amount_spent_first_shop = 520 → amount_spent_second_shop = 248 →
  books_first_shop = 42 → books_second_shop = 22 →
  average_price = 12 :=
by
  intros
  sorry

end average_price_per_book_l341_341601


namespace remainder_of_expression_l341_341948

theorem remainder_of_expression :
  (9^5 + 8^7 + 7^6) % 5 = 1 :=
by
  have h1 : 9 % 5 = 4 := by norm_num
  have h2 : 8 % 5 = 3 := by norm_num
  have h3 : 7 % 5 = 2 := by norm_num
  have rem_9_pow_5 := calc (9^5 % 5) = (4^5 % 5) : by rw [←pow_mod, h1]
                        ... = 4 % 5 : by norm_num
  
  have rem_8_pow_7 := calc (8^7 % 5) = (3^7 % 5) : by rw [←pow_mod, h2]
                        ... = 3 % 5 : by norm_num
  
  have rem_7_pow_6 := calc (7^6 % 5) = (2^6 % 5) : by rw [←pow_mod, h3]
                        ... = 4 % 5 : by norm_num
  
  calc (9^5 + 8^7 + 7^6) % 5 = (4 + 3 + 4) % 5 : by rw [rem_9_pow_5, rem_8_pow_7, rem_7_pow_6]
                            ... = 11 % 5 : by norm_num
                            ... = 1 % 5 : by norm_num
                            ... = 1 : by norm_num

end remainder_of_expression_l341_341948


namespace percentage_decrease_correct_l341_341626

noncomputable def original_price : ℝ := 1200
noncomputable def price_increase_percentage : ℝ := 0.10
noncomputable def final_price : ℝ := original_price - 78

theorem percentage_decrease_correct :
  let increased_price := original_price * (1 + price_increase_percentage)
  let decrease_percentage := ((increased_price - final_price) / increased_price) * 100 in
  decrease_percentage = 15 := 
by
  sorry

end percentage_decrease_correct_l341_341626


namespace sum_of_solutions_sum_of_integers_satisfying_eq_l341_341547

theorem sum_of_solutions (x : ℤ) (h : x^2 = x + 256) : x = 16 ∨ x = -16 := sorry

theorem sum_of_integers_satisfying_eq : ∑ x in finset.filter (λ x, x^2 = x + 256) (finset.Icc -16 16), x = 0 := 
begin
  sorry
end

end sum_of_solutions_sum_of_integers_satisfying_eq_l341_341547


namespace value_of_expression_l341_341777

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  rw [h1, h2]
  norm_num
  sorry

end value_of_expression_l341_341777


namespace inscribed_circle_radius_l341_341640

theorem inscribed_circle_radius (a b c : ℕ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) :
  let p := (a + b + c) / 2 in
  let S := (a * b) / 2 in
  let r := S / p in
  r = 1 :=
by
  -- Declaration of variables according to the given conditions
  let a := 3
  let b := 4
  let c := 5
  -- Calculations
  let p := (a + b + c) / 2
  have hp : p = 6, by sorry
  let S := (a * b) / 2
  have hS : S = 6, by sorry
  let r := S / p
  -- Final assertion
  show r = 1, by
    rw [hp, hS]
    exact rfl

end inscribed_circle_radius_l341_341640


namespace ellipse_segment_length_l341_341386

noncomputable def length_of_segment_AB : ℝ :=
  sqrt(2) * (sqrt ((4 / 7) - (-16 / 7)^2))

theorem ellipse_segment_length :
  ∀ (x y : ℝ) (l : ℝ → ℝ) (focus : ℝ × ℝ),
  (focus = (-2, 0)) →
  (l = λ x, x + 2) →
  (x^2 / 4 + y^2 / 3 = 1) →
  length_of_segment_AB = 12 * sqrt 2 / 7 :=
begin
  intro x y l focus, intro h_focus, intro h_line, intro h_ellipse,
  sorry
end

end ellipse_segment_length_l341_341386


namespace harry_fish_count_l341_341143

theorem harry_fish_count
  (sam_fish : ℕ) (joe_fish : ℕ) (harry_fish : ℕ)
  (h1 : sam_fish = 7)
  (h2 : joe_fish = 8 * sam_fish)
  (h3 : harry_fish = 4 * joe_fish) :
  harry_fish = 224 :=
by
  sorry

end harry_fish_count_l341_341143


namespace sequence_periodic_l341_341459

def num_divisors (m : ℕ) : ℕ := 
  finset.card (finset.filter (λ d, m % d = 0) (finset.range (m + 1)))

def sequence (c : ℕ) (n : ℕ) : ℕ :=
nat.rec_on n 1 (λ n a_n, num_divisors a_n + c)

theorem sequence_periodic (c : ℕ) :
  ∃ k, ∀ n, sequence c (k + n) = sequence c k :=
sorry

end sequence_periodic_l341_341459


namespace right_triangle_hypotenuse_length_l341_341429

theorem right_triangle_hypotenuse_length :
  (∃ a b : ℝ, a^2 - 6 * a + 4 = 0 ∧ b^2 - 6 * b + 4 = 0 ∧
   (c : ℝ) = real.sqrt (a^2 + b^2) ∧ c = 2 * real.sqrt 7) :=
begin
  sorry
end

end right_triangle_hypotenuse_length_l341_341429


namespace inflated_balls_successfully_l341_341504

variable (total_balls : ℕ)
variable (hole_percentage : ℚ)
variable (overinflate_percentage : ℚ)
variable (defect_percentage : ℚ)

theorem inflated_balls_successfully :
  total_balls = 500 →
  hole_percentage = 0.65 →
  overinflate_percentage = 0.25 →
  defect_percentage = 0.10 →
  (total_balls - 
  ⌊total_balls * hole_percentage⌋.to_nat - 
  ⌊(total_balls - ⌊total_balls * hole_percentage⌋.to_nat) * overinflate_percentage⌋.to_nat - 
  ⌊((total_balls - ⌊total_balls * hole_percentage⌋.to_nat) - 
  ⌊(total_balls - ⌊total_balls * hole_percentage⌋.to_nat) * overinflate_percentage⌋.to_nat) * defect_percentage⌋.to_nat) =
  118 :=
by
  intros ht tb hopks defp
  sorry

end inflated_balls_successfully_l341_341504


namespace three_letter_initials_count_l341_341409

theorem three_letter_initials_count :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  (finset.univ.filter (λ l : list Char, l.length = 3 ∧ l.nodup ∧ 
                                            ∀ x ∈ l, x ∈ letters)).card = 720 :=
by
  sorry

end three_letter_initials_count_l341_341409


namespace painters_work_days_l341_341088

theorem painters_work_days :
  (∀ (r: ℝ) (d1 d2: ℝ), d1 * r = 9 → d2 * r * 1.5 → d2 = 3 / 2) → sorry

end painters_work_days_l341_341088


namespace sequence_even_odd_l341_341722

-- Definitions
def sequence (a : ℕ) : ℕ → ℕ
| 0       => a
| (n + 1) => (nat.floor (1.5 * sequence n) + 1)

-- Problem Statement
theorem sequence_even_odd :
  ∃ a_1 : ℕ, 
    (∀ n, n < 100000 → even (sequence a_1 n)) ∧ 
    odd (sequence a_1 100000) := 
sorry

end sequence_even_odd_l341_341722


namespace marla_colors_green_l341_341496

theorem marla_colors_green 
  (total_rows : ℕ) (total_cols : ℕ) 
  (red_rows : ℕ) (red_cols : ℕ) 
  (blue_rows : ℕ) : 
  total_rows = 20 → 
  total_cols = 30 → 
  red_rows = 8 → 
  red_cols = 12 → 
  blue_rows = 6 → 
  let total_squares := total_rows * total_cols in
  let red_squares := red_rows * red_cols in
  let blue_squares := blue_rows * total_cols in
  let green_squares := total_squares - red_squares - blue_squares in
  green_squares = 324 :=
by 
  intros h1 h2 h3 h4 h5; 
  let total_squares := 20 * 30;
  let red_squares := 8 * 12;
  let blue_squares := 6 * 30;
  have h_total : total_squares = 600 := rfl;
  have h_red : red_squares = 96 := rfl;
  have h_blue : blue_squares = 180 := rfl;
  let green_squares := total_squares - red_squares - blue_squares;
  have h_green : green_squares = 600 - 96 - 180 := rfl;
  have h_g_simp : green_squares = 324 := rfl;
  exact h_g_simp;

end marla_colors_green_l341_341496


namespace locus_of_points_proof_l341_341670

noncomputable def locus_of_points_in_triangle (s : ℝ) : set (ℝ × ℝ) :=
{P | dist P (0, 0) ^ 2 + dist P (s, 0) ^ 2 + dist P (0, s) ^ 2 < 2 * s ^ 2}

def centroid_circle (s : ℝ) : set (ℝ × ℝ) :=
{P | dist P (s / 3, s / 3) < 2 * s / 3}

theorem locus_of_points_proof (s : ℝ) (P : ℝ × ℝ) :
  (P ∈ locus_of_points_in_triangle s) → (P ∈ centroid_circle s) :=
sorry

end locus_of_points_proof_l341_341670


namespace smallest_positive_integer_x_for_2520x_eq_m_cubed_l341_341681

theorem smallest_positive_integer_x_for_2520x_eq_m_cubed :
  ∃ (M x : ℕ), x > 0 ∧ 2520 * x = M^3 ∧ (∀ y, y > 0 ∧ 2520 * y = M^3 → x ≤ y) :=
sorry

end smallest_positive_integer_x_for_2520x_eq_m_cubed_l341_341681


namespace percentage_stock_sold_l341_341531

/-!
# Problem Statement
Given:
1. The cash realized on selling a certain percentage stock is Rs. 109.25.
2. The brokerage is 1/4%.
3. The cash after deducting the brokerage is Rs. 109.

Prove:
The percentage of the stock sold is 100%.
-/

noncomputable def brokerage_fee (S : ℝ) : ℝ :=
  S * 0.0025

noncomputable def selling_price (realized_cash : ℝ) (fee : ℝ) : ℝ :=
  realized_cash + fee

theorem percentage_stock_sold (S : ℝ) (realized_cash : ℝ) (cash_after_brokerage : ℝ)
  (h1 : realized_cash = 109.25)
  (h2 : cash_after_brokerage = 109)
  (h3 : brokerage_fee S = S * 0.0025) :
  S = 109.25 :=
by
  sorry

end percentage_stock_sold_l341_341531


namespace percentage_passed_in_all_three_subjects_l341_341444

-- Define the given failed percentages as real numbers
def A : ℝ := 0.25  -- 25%
def B : ℝ := 0.48  -- 48%
def C : ℝ := 0.35  -- 35%
def AB : ℝ := 0.27 -- 27%
def AC : ℝ := 0.20 -- 20%
def BC : ℝ := 0.15 -- 15%
def ABC : ℝ := 0.10 -- 10%

-- State the theorem to prove the percentage of students who passed in all three subjects
theorem percentage_passed_in_all_three_subjects : 
  1 - (A + B + C - AB - AC - BC + ABC) = 0.44 :=
by
  sorry

end percentage_passed_in_all_three_subjects_l341_341444


namespace solve_given_equation_l341_341524

noncomputable def solve_equation (x : ℂ) : Prop :=
  (4*x^3 + 4*x^2 + 3*x + 2)/(x - 2) = 4*x^2 + 5*x + 4

theorem solve_given_equation :
  ∀ x : ℂ, solve_equation x → 
    (x = (-9 + complex.I * complex.sqrt 79) / 8) ∨ 
    (x = (-9 - complex.I * complex.sqrt 79) / 8) :=
by
  intro x h
  sorry

end solve_given_equation_l341_341524


namespace range_of_a_l341_341730

theorem range_of_a (a : ℝ) (p : ∀ x ∈ Icc 1 2, x^2 - a ≥ 0) (q : ∃ x : ℝ, x^2 + 2 * a * x + (2 - a) = 0) : ¬ (¬ p ∨ ¬ q) → (a ≤ -2 ∨ a = 1) :=
sorry

end range_of_a_l341_341730


namespace charity_total_amount_raised_l341_341438

theorem charity_total_amount_raised :
  let students_group_A := 10 in
  let race_A := students_group_A * 20 in
  let bake_sales_A := students_group_A * 5 in
  let total_A := race_A + bake_sales_A in
  
  let students_group_B := 12 in
  let race_B := students_group_B * 30 in
  let car_washes_B := students_group_B * 10 in
  let total_B := race_B + car_washes_B in
  
  let students_group_C := 8 in
  let race_C := students_group_C * 25 in
  let raffle_C := 150 in
  let total_C := race_C + raffle_C in

  let students_group_D := 15 in
  let race_D := students_group_D * 35 in
  let garage_sale_D := 200 in
  let total_D := race_D + garage_sale_D in
  
  let students_group_E := 5 in
  let race_E := students_group_E * 40 in
  let art_auction_E := 300 in
  let total_E := race_E + art_auction_E in
  
  let total_amount_raised := total_A + total_B + total_C + total_D + total_E in
  
  total_amount_raised = 2305 :=
by
  -- The proof will be added here.
  sorry

end charity_total_amount_raised_l341_341438


namespace shortest_chord_line_through_P_longest_chord_line_through_P_l341_341741

theorem shortest_chord_line_through_P (P : ℝ × ℝ) (circle : (ℝ × ℝ) → Prop) (hP : P = (-1, 2))
  (h_circle_eq : ∀ (x y : ℝ), circle (x, y) ↔ x ^ 2 + y ^ 2 = 8) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ (x y : ℝ), y = 1/2 * x + 5/2 → a * x + b * y + c = 0)
  ∧ (a = 1) ∧ (b = -2) ∧ (c = 5) := sorry

theorem longest_chord_line_through_P (P : ℝ × ℝ) (circle : (ℝ × ℝ) → Prop) (hP : P = (-1, 2))
  (h_circle_eq : ∀ (x y : ℝ), circle (x, y) ↔ x ^ 2 + y ^ 2 = 8) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ (x y : ℝ), y = -2 * x → a * x + b * y + c = 0)
  ∧ (a = 2) ∧ (b = 1) ∧ (c = 0) := sorry

end shortest_chord_line_through_P_longest_chord_line_through_P_l341_341741


namespace find_circle_equation_l341_341173

open Real

def is_on_line (p : ℝ × ℝ) (a b c : ℝ) : Prop :=
  a * p.1 + b * p.2 + c = 0

def is_on_y_axis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def equation_of_circle (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

theorem find_circle_equation :
  ∃ center : ℝ × ℝ, ∃ radius : ℝ,
  is_on_line center 2 (-1) (-7) ∧
  equation_of_circle center radius (0, -4) ∧
  equation_of_circle center radius (0, -2) ∧
  equation_of_circle center radius = (λ p, (p.1 - 2)^2 + (p.2 + 3)^2 = 5) :=
by
  sorry

end find_circle_equation_l341_341173


namespace probability_of_rolling_sum_12_with_four_dice_l341_341424

noncomputable def probability_sum_of_four_dice_is_12 : ℚ :=
  let outcomes := {x : Finset (List ℕ) | x ∈ ((Finset.range 6).image (λ x => x + 1)).product (Fun.repeat ((Finset.range 6).image (λ x => x + 1)) 3) ∧ x.sum = 12}
  in (outcomes.card : ℚ) / (6^4)

theorem probability_of_rolling_sum_12_with_four_dice :
  probability_sum_of_four_dice_is_12 = 73 / 1296 := by
  sorry

end probability_of_rolling_sum_12_with_four_dice_l341_341424


namespace option_d_always_holds_l341_341233

theorem option_d_always_holds (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b := by
  sorry

end option_d_always_holds_l341_341233


namespace heavy_operators_earn_129_dollars_per_day_l341_341286

noncomputable def heavy_operator_daily_wage (H : ℕ) : Prop :=
  let laborer_wage := 82
  let total_people := 31
  let total_payroll := 3952
  let laborers_count := 1
  let heavy_operators_count := total_people - laborers_count
  let heavy_operators_payroll := total_payroll - (laborer_wage * laborers_count)
  H = heavy_operators_payroll / heavy_operators_count

theorem heavy_operators_earn_129_dollars_per_day : heavy_operator_daily_wage 129 :=
by
  unfold heavy_operator_daily_wage
  sorry

end heavy_operators_earn_129_dollars_per_day_l341_341286


namespace coefficient_x7_l341_341449

theorem coefficient_x7 (a : ℝ) (h : (Nat.choose 10 3) * (-a)^3 = 15) : a = 1 / 2 :=
by
  have : (Nat.choose 10 3) = 120 := by norm_num
  have h1 : 120 * (-a)^3 = 15 := by rwa [this] at h
  sorry  -- Proof will be completed here.

end coefficient_x7_l341_341449


namespace find_angle_l341_341022

theorem find_angle (x : ℝ) (h : 180 - x = 6 * (90 - x)) : x = 72 := 
by 
    sorry

end find_angle_l341_341022


namespace angle_bisector_contradiction_l341_341603

theorem angle_bisector_contradiction
  (A B C K : Type)
  (triangle : ∃ (A B C : Type), A ∈ triangle ∧ B ∈ triangle ∧ C ∈ triangle)
  (angle_bisector_A : (∃ (K : Type), (AK : is_bisector_of_angle A K) ∧ K ∈ BC))
  (bisector_bisects_BC : AK ∈ bisector_of B K) :
  false :=
by {
  -- Use the provided conditions to lead this to a contradiction,
  -- thus proving the theorem.
  sorry
}

end angle_bisector_contradiction_l341_341603


namespace solve_g_l341_341094

def g (a b : ℚ) : ℚ :=
if a + b ≤ 4 then (a * b - 2 * a + 3) / (3 * a)
else (a * b - 3 * b - 1) / (-3 * b)

theorem solve_g :
  g 3 1 + g 1 5 = 11 / 15 :=
by
  -- Here we just set up the theorem statement. Proof is not included.
  sorry

end solve_g_l341_341094


namespace range_of_a_l341_341478

theorem range_of_a (a : ℝ) :
  let A := (-2, 3)
  let B := (0, a)
  let circle := (x + 3)^2 + (y + 2)^2 = 1
  ∃ a ∈ (1/3 : ℝ)..(3/2 : ℝ), intersects_symmetrical_line (A, B, circle) := 
sorry

end range_of_a_l341_341478


namespace triangle_inequality_l341_341007

variables {α β γ a b c : ℝ}
variable {n : ℕ}

theorem triangle_inequality (h_sum_angles : α + β + γ = Real.pi) (h_pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  (Real.pi / 3) ^ n ≤ (a * α ^ n + b * β ^ n + c * γ ^ n) / (a + b + c) ∧ 
  (a * α ^ n + b * β ^ n + c * γ ^ n) / (a + b + c) < (Real.pi ^ n / 2) :=
by
  sorry

end triangle_inequality_l341_341007


namespace contest_correct_answers_l341_341061

/-- 
In a mathematics contest with ten problems, a student gains 
5 points for a correct answer and loses 2 points for an 
incorrect answer. If Olivia answered every problem 
and her score was 29, how many correct answers did she have?
-/
theorem contest_correct_answers (c w : ℕ) (h1 : c + w = 10) (h2 : 5 * c - 2 * w = 29) : c = 7 :=
by 
  sorry

end contest_correct_answers_l341_341061


namespace math_problem_l341_341311

variable {R : Type*} [LinearOrder R] [Add R] [Neg R]

def odd_function (f : R → R) : Prop :=
  ∀ x, f (-x) = -f (x)

def decreasing_function (f : R → R) : Prop :=
  ∀ x y, x ≤ y → f (x) ≥ f (y)

noncomputable def f (x : R) : R := sorry

theorem math_problem (m n : R) (Hmn : m + n ≥ 0) 
    (Hodd : odd_function f) (Hdec : decreasing_function f) :
    (f(m) * f(-m) ≤ 0) ∧ (f(m) + f(n) ≤ f(-m) + f(-n)) :=
by
  sorry

end math_problem_l341_341311


namespace evaluate_M_plus_N_l341_341837

def permutations := 
  Multiset.toFinset (Multiset.perm (Multiset.ofList [1, 2, 3, 4, 6]))

def sum_of_products (p : List ℕ) : ℕ := 
  p.head! * p.tail!.head! + 
  p.tail!.head! * p.tail!.tail!.head! + 
  p.tail!.tail!.head! * p.tail!.tail!.tail!.head! + 
  p.tail!.tail!.tail!.head! * p.tail!.tail!.tail!.tail!.head! + 
  p.tail!.tail!.tail!.tail!.head! * p.head!

def M : ℕ := 
  Finset.max (Finset.image sum_of_products permutations)

def N : ℕ := 
  permutations.filter (λ p => sum_of_products p = M).card

theorem evaluate_M_plus_N : M + N = 65 := 
  by
    -- The proof goes here
    sorry

end evaluate_M_plus_N_l341_341837


namespace hospital_workers_count_l341_341821

theorem hospital_workers_count {N : ℕ} (hj : N = 5 - 3)
  (prob_jj : (2.choose 2) * 1 / (N.choose 2) = 0.1) : N = 5 :=
by
  -- Use the given conditions directly to prove the statement
  sorry

end hospital_workers_count_l341_341821


namespace tire_price_l341_341168

/-- 
Mark paid $255 for a set of four tires with the following promotion:
1. A 20% discount on three tires.
2. The fourth tire costs 5 dollars.

Prove that the regular price of one tire is 104.17 dollars.
--/
theorem tire_price (x : ℝ) (h1 : 3 * 0.8 * x + 5 = 255) : x = 104.17 := 
begin
  sorry
end

end tire_price_l341_341168


namespace unique_int_solution_ineq_m_eq_4_max_value_a2_b2_c2_l341_341394

-- Part I
theorem unique_int_solution_ineq_m_eq_4 {
  m : ℝ} (h : ∀ x : ℤ, |2 * (x : ℝ) - m| ≤ 1 → x = 2) : m = 4 := sorry

-- Part II
theorem max_value_a2_b2_c2 {
  a b c : ℝ} (h : 4 * a^4 + 4 * b^4 + 4 * c^4 = 4) : a^2 + b^2 + c^2 ≤ real.sqrt 3 := sorry

end unique_int_solution_ineq_m_eq_4_max_value_a2_b2_c2_l341_341394


namespace points_of_tangency_coplanar_l341_341270

-- Conditions:
variables {A B C D K L M N : Type}
variables [InnerProductSpace ℝ A] -- Assume A is a real inner product space since we're dealing with Euclidean geometry.
variables {sphere : Set A}
variables {AB BC CD DA : Set A}
variables {K L M N : A}

-- Assume K, L, M, N are points of tangency of the sphere with sides:
-- K on AB, L on BC, M on CD, N on DA
variable (tangent_points : sphere ∩ AB = {K} ∧ sphere ∩ BC = {L} ∧ sphere ∩ CD = {M} ∧ sphere ∩ DA = {N})

-- To prove:
theorem points_of_tangency_coplanar (hs : tangent_points):
  ∃ (P : AffineSubspace ℝ A), K ∈ P ∧ L ∈ P ∧ M ∈ P ∧ N ∈ P := sorry

end points_of_tangency_coplanar_l341_341270


namespace sum_of_divisors_of_prime_47_l341_341963

theorem sum_of_divisors_of_prime_47 : ∀ n : ℕ, prime n → n = 47 → (∑ d in (finset.filter (λ x, n % x = 0) (finset.range (n + 1))), d) = 48 := 
by {
  intros n prime_n n_is_47,
  sorry -- Proof is omitted
}

end sum_of_divisors_of_prime_47_l341_341963


namespace range_of_x_l341_341116

theorem range_of_x (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x < -1/2 ∨ x > 1/4 :=
by
  sorry

end range_of_x_l341_341116


namespace find_A_find_area_l341_341106

variables {A B C : ℝ}
variables {a b c h : ℝ}

-- Condition: a * sin(B) = sqrt(3) * b * cos(A)
def condition1 : Prop := a * Real.sin B = Real.sqrt 3 * b * Real.cos A

-- Condition: Altitude on side AB is sqrt(3), and a = 3.
def condition2 : Prop := h = Real.sqrt 3 ∧ a = 3

-- Prove that A = π / 3 given condition1.
theorem find_A (h1 : condition1) : A = (Real.pi / 3) :=
  sorry

-- Prove that the area of ABC is (sqrt(3) + 3 * sqrt(2)) / 2 given condition2.
theorem find_area (h2 : condition2) : 
  let b := 2 in
  let c := 1 + Real.sqrt 6 in
  (b * c * Real.sin (Real.pi / 3)) / 2 = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2 :=
  sorry

end find_A_find_area_l341_341106


namespace no_solution_abs_eq_l341_341413

theorem no_solution_abs_eq : ∀ y : ℝ, |y - 2| ≠ |y - 1| + |y - 4| :=
by
  intros y
  sorry

end no_solution_abs_eq_l341_341413


namespace decimal_zeros_l341_341230

theorem decimal_zeros (a b : ℕ) (h : a = 3 ∧ b = 1250) : 
  (number_of_zeros_between_decimal_point_and_first_nonzero_digit (a / b : ℚ) = 2) :=
sorry

end decimal_zeros_l341_341230


namespace Nicky_pace_5_mps_l341_341874

/-- Given the conditions:
  - Cristina runs at a pace of 5 meters per second.
  - Nicky runs for 30 seconds before Cristina catches up to him.
  Prove that Nicky’s pace is 5 meters per second. -/
theorem Nicky_pace_5_mps
  (Cristina_pace : ℝ)
  (time_Nicky : ℝ)
  (catchup : Cristina_pace * time_Nicky = 150)
  (def_Cristina_pace : Cristina_pace = 5)
  (def_time_Nicky : time_Nicky = 30) :
  (150 / 30) = 5 :=
by
  sorry

end Nicky_pace_5_mps_l341_341874


namespace spacesMovedBeforeSetback_l341_341898

-- Let's define the conditions as local constants
def totalSpaces : ℕ := 48
def firstTurnMove : ℕ := 8
def thirdTurnMove : ℕ := 6
def remainingSpacesToWin : ℕ := 37
def setback : ℕ := 5

theorem spacesMovedBeforeSetback (x : ℕ) : 
  (firstTurnMove + thirdTurnMove) + x - setback + remainingSpacesToWin = totalSpaces →
  x = 28 := by
  sorry

end spacesMovedBeforeSetback_l341_341898


namespace sum_of_reciprocals_of_roots_eq_17_div_8_l341_341703

theorem sum_of_reciprocals_of_roots_eq_17_div_8 :
  ∀ p q : ℝ, (p + q = 17) → (p * q = 8) → (1 / p + 1 / q = 17 / 8) :=
by
  intros p q h1 h2
  sorry

end sum_of_reciprocals_of_roots_eq_17_div_8_l341_341703


namespace distribution_schemes_correct_l341_341682

-- Define the number of volunteers, number of projects, and requirement of at least one volunteer per project
def volunteers : Nat := 6
def projects : Nat := 4

-- Function to calculate the number of ways to distribute volunteers to projects
noncomputable def distribution_schemes (v p : Nat) : Nat := 
  -- Case 1: Two groups of 2 volunteers and two groups of 1 volunteer each
  let case1 := (Nat.choose v 2 * Nat.choose (v - 2) 2) / Nat.factorial 2 +
               -- Case 2: One group of 3 volunteers and three groups of 1 volunteer each
               Nat.choose v 3
  in case1 * Nat.factorial p

-- Statement to be proven
theorem distribution_schemes_correct :
  distribution_schemes volunteers projects = 1560 := 
sorry

end distribution_schemes_correct_l341_341682


namespace range_of_a_l341_341755

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / (Real.log x) + a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f a x ≤ f a (x + ε))) → a ≤ -1/4 :=
sorry

end range_of_a_l341_341755


namespace non_seniors_play_instrument_count_l341_341804

/-
  There are 400 students in total.
  50% of the seniors play a musical instrument.
  20% of the non-seniors do not play a musical instrument.
  40% of the students do not play a musical instrument.

  Prove that the number of non-seniors who play a musical instrument is 106.
-/
noncomputable def number_of_non_seniors_playing_instrument (s n : ℕ) : ℕ :=
  n

theorem non_seniors_play_instrument_count (s n : ℕ) (h1 : s + n = 400)
    (h2 : 0.5 * s + 0.2 * n = 160) : number_of_non_seniors_playing_instrument s n = 106 := by
  -- Sorry is used here to skip the proof.
  sorry

end non_seniors_play_instrument_count_l341_341804


namespace no_solution_when_k_eq_7_l341_341315

theorem no_solution_when_k_eq_7 
  (x : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : 
  (∀ k : ℝ, (x - 3) / (x - 4) = (x - k) / (x - 8) → False) ↔ k = 7 :=
by
  sorry

end no_solution_when_k_eq_7_l341_341315


namespace initial_concentration_l341_341994

theorem initial_concentration (C : ℝ) 
  (hC : (C * 0.2222222222222221) + (0.25 * 0.7777777777777779) = 0.35) :
  C = 0.7 :=
sorry

end initial_concentration_l341_341994


namespace minimum_value_l341_341372

theorem minimum_value (x y : ℝ) (h1 : xy + 1 = 4x + y) (h2 : x > 1) : (x + 1) * (y + 2) ≥ 15 :=
sorry

end minimum_value_l341_341372


namespace probability_longer_piece_at_least_y_l341_341267

theorem probability_longer_piece_at_least_y (y : ℝ) (hy : y > 0) :
  let C : ℝ := uniform 0 2 in
  ∀ C ∈ set.univ, 
  let longer_piece := if C ≤ 1 then 2 - C else C - 2 + C in
  (probability ( (C ≤ (2 / (y + 1))) ∨ (C ≥ (2 * y / (y + 1)))) = 4 / (y + 1) ) :=
by
  sorry

end probability_longer_piece_at_least_y_l341_341267


namespace find_a_l341_341350

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - y = 3) : a = 2 :=
by
  sorry

end find_a_l341_341350


namespace factorial_product_trailing_zeros_l341_341920

def countTrailingZerosInFactorialProduct : ℕ :=
  let countFactorsOfFive (n : ℕ) : ℕ := 
    (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125) + (n / 15625) + (n / 78125) + (n / 390625) 
  List.range 100 -- Generates list [0, 1, ..., 99]
  |> List.map (fun k => countFactorsOfFive (k + 1)) -- Apply countFactorsOfFive to each k+1
  |> List.foldr (· + ·) 0 -- Sum all counts

theorem factorial_product_trailing_zeros : countTrailingZerosInFactorialProduct = 1124 := by
  sorry

end factorial_product_trailing_zeros_l341_341920


namespace harry_fish_count_l341_341144

theorem harry_fish_count
  (sam_fish : ℕ) (joe_fish : ℕ) (harry_fish : ℕ)
  (h1 : sam_fish = 7)
  (h2 : joe_fish = 8 * sam_fish)
  (h3 : harry_fish = 4 * joe_fish) :
  harry_fish = 224 :=
by
  sorry

end harry_fish_count_l341_341144


namespace books_read_in_one_week_l341_341500

constant books_per_day : Nat := 2
constant days_in_week : Nat := 7

theorem books_read_in_one_week : (books_per_day * days_in_week) = 14 := by
  sorry

end books_read_in_one_week_l341_341500


namespace real_part_of_inverse_l341_341119

def z : ℂ := sorry

theorem real_part_of_inverse (hz: |z| = 2 ∧ im z ≠ 0) : real_part (1 / (2 - z)) = 1 / 2 := 
by
  sorry

end real_part_of_inverse_l341_341119


namespace math_problem_l341_341491

theorem math_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^3 + y^3 = x - y) : x^2 + 4 * y^2 < 1 := 
sorry

end math_problem_l341_341491


namespace product_smallest_largest_approx_l341_341894

theorem product_smallest_largest_approx :
  (∃ (x : ℚ), 3 * x + 4 * x + 5 * x + 6 * x + 7 * x + 8 * x = 3850) →
  (∃ (product : ℚ), product ≈ 326666.666...) :=
by
  intro hx
  cases hx with x hx_eq
  let smallest := 3 * x
  let largest := 8 * x
  have prod := smallest * largest
  use prod
  sorry

end product_smallest_largest_approx_l341_341894


namespace aya_coloring_max_value_l341_341943

theorem aya_coloring_max_value (n : ℕ) (hn : n ≥ 1) :
  ∃ k : ℕ, (∀ (tokens : Matrix (Fin n) (Fin n) Bool), 
            (∀ i j, tokens i j → tokens i j → i = j ∨ i ≠ j ∧ j ≠ i) ∧
            (∃ unique_T : Matrix (Fin n) (Fin n) Unit, ∀ i, 
              ∃! j, tokens i j → colored (i, j) )
            → k = n * (n + 1) / 2) :=
by
  sorry

end aya_coloring_max_value_l341_341943


namespace sum_of_fractions_l341_341592

theorem sum_of_fractions : (3 / 20 : ℝ) + (5 / 50 : ℝ) + (7 / 2000 : ℝ) = 0.2535 :=
by sorry

end sum_of_fractions_l341_341592


namespace curve_is_line_l341_341694

-- Let r, θ be real numbers
variables (r θ : ℝ)

-- Define the polar equation
def polar_equation (θ : ℝ) : ℝ :=
  2 / (2 * Real.sin θ - 3 * Real.cos θ)

-- Define the Cartesian equations derived from the polar equation
noncomputable def x (r θ : ℝ) : ℝ := r * Real.cos θ
noncomputable def y (r θ : ℝ) : ℝ := r * Real.sin θ

theorem curve_is_line : 
  ∀ θ : ℝ, let r := polar_equation θ in (x r θ = 0) ∨ (5 * (x r θ) = 12 * (y r θ)) := 
by 
  sorry

end curve_is_line_l341_341694


namespace rational_sin_cos_iff_rational_or_undefined_tan_half_l341_341884

variable (α : ℝ)

def rational_sin_cos (α : ℝ) : Prop :=
  ∃ y x : ℚ, sin α = y ∧ cos α = x

def rational_or_undefined_tan_half (α : ℝ) : Prop :=
  ∃ m : ℚ, tan (α / 2) = m ∨ cos (α / 2) = 0 ∧ sin (α / 2) = 1 ∨ sin (α / 2) = -1

theorem rational_sin_cos_iff_rational_or_undefined_tan_half :
  rational_sin_cos α ↔ rational_or_undefined_tan_half α := by
  sorry

end rational_sin_cos_iff_rational_or_undefined_tan_half_l341_341884


namespace cosine_of_angle_l341_341025

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (4, -2) - a

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def cos_angle (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cosine_of_angle :
  cos_angle a b = Real.sqrt 5 / 5 :=
by
  sorry

end cosine_of_angle_l341_341025


namespace calculate_sin_product_l341_341086

theorem calculate_sin_product (α β : ℝ) (h1 : Real.sin (α + β) = 0.2) (h2 : Real.cos (α - β) = 0.3) :
  Real.sin (α + π/4) * Real.sin (β + π/4) = 0.25 :=
by
  sorry

end calculate_sin_product_l341_341086


namespace problem_l341_341758

-- Conditions
def f (a b x : ℝ) : ℝ := a * x^3 + (sqrt b) * x^2 - a^2 * x

theorem problem (a b x1 x2 : ℝ) (h_pos_a : 0 < a)
  (h_x1_lt_x2 : x1 < x2) 
  (h_f'_zero : 3 * a * x1^2 + 2 * (sqrt b) * x1 - a^2 = 0 ∧
               3 * a * x2^2 + 2 * (sqrt b) * x2 - a^2 = 0) 
  (h_abs_sum : |x1| + |x2| = 2) : 
  (0 < a ∧ a ≤ 3) ∧ (0 ≤ b ∧ b ≤ 12) := 
by
  sorry

end problem_l341_341758


namespace two_digit_congruent_to_one_mod_three_l341_341043

theorem two_digit_congruent_to_one_mod_three :
  {n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ n % 3 = 1}.card = 30 :=
by
  sorry

end two_digit_congruent_to_one_mod_three_l341_341043


namespace base_angle_of_isosceles_triangle_l341_341366

theorem base_angle_of_isosceles_triangle (A B C : ℝ) (h_triangle : A + B + C = 180) (h_isosceles : A = B ∨ B = C ∨ A = C) (h_angle : A = 42 ∨ B = 42 ∨ C = 42) :
  A = 42 ∨ A = 69 ∨ B = 42 ∨ B = 69 ∨ C = 42 ∨ C = 69 :=
by
  sorry

end base_angle_of_isosceles_triangle_l341_341366


namespace bugs_meet_l341_341809

/-- 
Given an isosceles trapezoid ABCD with AB > CD and diagonal AC, and two bugs moving with 
constant and identical speeds along the contours of the triangles ADC and ABC respectively,
they will eventually meet on the diagonal AC.
-/
theorem bugs_meet {a b c d : ℝ} (h1 : a > c) 
  (constant_speed : ℝ) 
  (cycle1 : A → C → D → A) 
  (cycle2 : A → B → C → A) :
  ∃ t : ℝ, ∃ x : ℝ, ∃ y : ℝ, (x = y) ∧ (cycle1 t = x) ∧ (cycle2 t = y) :=
sorry

end bugs_meet_l341_341809


namespace ab5_a2_c5_a2_inequality_l341_341863

theorem ab5_a2_c5_a2_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a ^ 5 - a ^ 2 + 3) * (b ^ 5 - b ^ 2 + 3) * (c ^ 5 - c ^ 2 + 3) ≥ (a + b + c) ^ 3 := 
by
  sorry

end ab5_a2_c5_a2_inequality_l341_341863


namespace average_profit_south_average_profit_north_prob_picking_one_third_grade_apple_chi_squared_test_l341_341210

-- Definitions for the problem conditions
def cost_per_kg := 5
def price_first_grade := 12
def price_second_grade := 8
def price_third_grade := 1

-- South Mountain apple distribution
def south_first_grade_kg := 40
def south_second_grade_kg := 150
def south_third_grade_kg := 10

-- North Mountain apple distribution
def north_first_grade_kg := 50
def north_second_grade_kg := 120
def north_third_grade_kg := 30

-- Number of apples per kilogram for each grade
def apples_per_kg_first_grade := 3
def apples_per_kg_second_grade := 4
def apples_per_kg_third_grade := 6

-- Event definitions for the Chi-squared test
def a := south_third_grade_kg
def b := south_first_grade_kg + south_second_grade_kg
def c := north_third_grade_kg
def d := north_first_grade_kg + north_second_grade_kg
def n := a + b + c + d

-- Chi-squared critical values
def chi_squared_critical_value_0_01 := 6.635

-- Lean 4 statement for the problem

theorem average_profit_south:
  (1 / 200.0) * ((price_first_grade * south_first_grade_kg) +
                 (price_second_grade * south_second_grade_kg) +
                 (price_third_grade * south_third_grade_kg) -
                 (cost_per_kg * 200)) = 3.45 := 
sorry

theorem average_profit_north:
  (1 / 200.0) * ((price_first_grade * north_first_grade_kg) +
                 (price_second_grade * north_second_grade_kg) +
                 (price_third_grade * north_third_grade_kg) -
                 (cost_per_kg * 200)) = 2.95 :=
sorry

theorem prob_picking_one_third_grade_apple:
  (let total_apples := (south_first_grade_kg * apples_per_kg_first_grade) +
                      (south_second_grade_kg * apples_per_kg_second_grade) +
                      (south_third_grade_kg * apples_per_kg_third_grade)
  in let sampling_ratio := 13 / total_apples.to_float
  in let selected_first := (south_first_grade_kg * apples_per_kg_first_grade * sampling_ratio).to_nnreal
  in let selected_second := (south_second_grade_kg * apples_per_kg_second_grade * sampling_ratio).to_nnreal
  in let selected_third := (south_third_grade_kg * apples_per_kg_third_grade * sampling_ratio).to_nnreal
  in let total_combinations := nat.choose 13 2
  in let success_combinations := nat.choose (selected_first + selected_second) 1 * nat.choose selected_third 1
  in success_combinations / total_combinations.to_float) = (2.0 / 13.0) :=
sorry

theorem chi_squared_test:
  (let K_squared := (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))
  in K_squared) > chi_squared_critical_value_0_01 :=
sorry

end average_profit_south_average_profit_north_prob_picking_one_third_grade_apple_chi_squared_test_l341_341210


namespace smallest_inverse_defined_l341_341229

theorem smallest_inverse_defined (n : ℤ) : n = 5 :=
by sorry

end smallest_inverse_defined_l341_341229


namespace total_sacks_l341_341406

theorem total_sacks (sacks_per_day : ℕ) (days : ℕ) (h1 : sacks_per_day = 38) (h2 : days = 49) :
  sacks_per_day * days = 1862 :=
by
  rw [h1, h2]
  exact rfl

end total_sacks_l341_341406


namespace min_value_of_A_div_B_l341_341316

noncomputable def A (g1 : Finset ℕ) : ℕ :=
  g1.prod id

noncomputable def B (g2 : Finset ℕ) : ℕ :=
  g2.prod id

theorem min_value_of_A_div_B : ∃ (g1 g2 : Finset ℕ), 
  g1 ∪ g2 = (Finset.range 31).erase 0 ∧ g1 ∩ g2 = ∅ ∧ A g1 % B g2 = 0 ∧ A g1 / B g2 = 1077205 :=
by
  sorry

end min_value_of_A_div_B_l341_341316


namespace interval_of_monotonic_increase_correct_l341_341048

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1/2) * Real.sin (2 * x + π / 3)

-- Define the function g(x) obtained by shifting f(x) to the left by π / 3 units
def g (x : ℝ) : ℝ := (1/2) * Real.sin (2 * (x + π / 3) + π / 3)

-- Define the target intervals
def interval_of_monotonic_increase (k : ℤ) : Set ℝ :=
  { x | k * π + π / 4 ≤ x ∧ x ≤ k * π + 3 * π / 4 }

-- State the main theorem to be proven
theorem interval_of_monotonic_increase_correct (k : ℤ) :
  (forall x, interval_of_monotonic_increase k x ⟺
  (g(x) = - (1/2) * Real.sin(2 * x) ∧
  k * π + π / 4 ≤ x ∧ x ≤ k * π + 3 * π / 4)) := by
    sorry

end interval_of_monotonic_increase_correct_l341_341048


namespace draw_from_unit_D_l341_341684

variable (d : ℕ)

-- Variables representing the number of questionnaires drawn from A, B, C, and D
def QA : ℕ := 30 - d
def QB : ℕ := 30
def QC : ℕ := 30 + d
def QD : ℕ := 30 + 2 * d

-- Total number of questionnaires drawn
def TotalDrawn : ℕ := QA d + QB + QC d + QD d

theorem draw_from_unit_D :
  (TotalDrawn d = 150) →
  QD d = 60 := sorry

end draw_from_unit_D_l341_341684


namespace solution_1_solution_2_l341_341033

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (2 * x + 3)

lemma f_piecewise (x : ℝ) : 
  f x = if x ≤ -3 / 2 then -4 * x - 2
        else if -3 / 2 < x ∧ x < 1 / 2 then 4
        else 4 * x + 2 := 
by
-- This lemma represents the piecewise definition of f(x)
sorry

theorem solution_1 : 
  (∀ x : ℝ, f x < 5 ↔ (-7 / 4 < x ∧ x < 3 / 4)) := 
by 
-- Proof of the inequality solution
sorry

theorem solution_2 : 
  (∀ t : ℝ, (∀ x : ℝ, f x - t ≥ 0) → t ≤ 4) :=
by
-- Proof that the maximum value of t is 4
sorry

end solution_1_solution_2_l341_341033


namespace find_k_range_l341_341866

def A (k : ℝ) : set ℝ := {x | k * x^2 - (k + 3) * x - 1 ≥ 0}

def B : set ℝ := {y | ∃ x : ℝ, y = 2 * x + 1}

def proof (k : ℝ) :=
  A k ∩ B = ∅ → -9 < k ∧ k < -1

theorem find_k_range (k : ℝ) : proof k := sorry

end find_k_range_l341_341866


namespace equation_of_l3_minimum_distance_l341_341038

-- Definitions for the given conditions, points A, and lines
def l1 (x y : ℝ) : Prop := 2 * x + y = 3
def l2 (x y : ℝ) : Prop := x - y = 0
def l4 (x y m : ℝ) : Prop := 4 * x + 2 * y + m^2 + 1 = 0

-- Intersection point of l1 and l2 is A
def A : ℝ × ℝ := (1, 1)

-- The proof goals
theorem equation_of_l3 (x y : ℝ) : 
  (∃ a b : ℝ, l1 a b ∧ l2 a b ∧ (x = a ∧ y = b) ∧ x - 2 * y + 1 = 0) :=
sorry

theorem minimum_distance (m : ℝ) : 
    let d := abs (m^2 + 7) / (2 * real.sqrt 5)
    in d = 7 * real.sqrt 5 / 10 :=
sorry

end equation_of_l3_minimum_distance_l341_341038


namespace find_number_l341_341988

theorem find_number (x : ℝ) : 61 + x * 12 / (180 / 3) = 62 → x = 5 :=
by
  sorry

end find_number_l341_341988


namespace zero_in_interval_l341_341189

noncomputable def f (x : ℝ) : ℝ := 3^x - 4

theorem zero_in_interval : ∃ x ∈ set.Ioo (1:ℝ) (2:ℝ), f x = 0 :=
by
  sorry

end zero_in_interval_l341_341189


namespace least_divisible_prime_product_l341_341224

theorem least_divisible_prime_product : ∃ n : ℕ, (n % 7 = 0) ∧ (n % 11 = 0) ∧ (n % 13 = 0) ∧ (n <= fact 13) ∧ ∀ m : ℕ, (m % 7 = 0) ∧ (m % 11 = 0) ∧ (m % 13 = 0) ∧ (m <= fact 13) → n <= m :=
begin
  let n := 7 * 11 * 13,
  use n,
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_mul_right 7 (11 * 13)) },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_mul_of_dvd_right (dvd_refl 11) 13) },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_mul_of_dvd_right (dvd_refl 13) 7) },
  split,
  { apply nat.le_of_dvd,
    { apply nat.fact_pos },
    { exact nat.lcm_dvd (by norm_num) (by norm_num) } },
  { intros m hm,
    exact nat.le_of_dvd (nat.pos_of_ne_zero (by norm_num)) (nat.gcd_lcm_dvd_iff.mpr ⟨hm.1, hm.2.1⟩) },
end

end least_divisible_prime_product_l341_341224


namespace f_is_odd_f_range_ff_domain_l341_341356

-- Define the function f
def f (x : ℝ) : ℝ := (Real.exp x + 1) / (Real.exp x - 1)

-- Prove that f is an odd function
theorem f_is_odd : ∀ x, f (-x) = -f x :=
by
  sorry

-- Prove the range of f(x) is (-∞, -1) ∪ (1, +∞)
theorem f_range : ∀ y, (∃ x, f x = y) ↔ y ∈ Set.Ioo (-∞ : ℝ) (-1) ∪ Set.Ioo (1 : ℝ) (∞) :=
by
  sorry

-- Prove the domain of f(f(x)) is { x | x ≠ 0 }
theorem ff_domain : ∀ x, x ≠ 0 ↔ f (f x) ∈ Set.univ :=
by
  sorry

end f_is_odd_f_range_ff_domain_l341_341356


namespace y_for_10min_y_range_6_10_max_time_13_5_liters_l341_341320

noncomputable def oxygen_consumption_10min (x : ℝ) := 
  (50 / x) * (x^2 / 100) + 10 * 0.3 + (100 / x) * 0.32

theorem y_for_10min (x : ℝ) (hx : x > 0) : 
  oxygen_consumption_10min x = x / 2 + 32 / x + 3 :=
sorry

noncomputable def oxygen_consumption_20min (x : ℝ) :=
  (50 / x) * (x^2 / 100) + 20 * 0.3 + (100 / x) * 0.32

theorem y_range_6_10 (x : ℝ) (hx : 6 ≤ x ∧ x ≤ 10) :
  14 ≤ oxygen_consumption_20min x ∧ oxygen_consumption_20min x ≤ 43 / 3 :=
sorry

def max_underwater_time (total_oxygen : ℝ) (consume_diving_return : ℝ) (consume_per_minute : ℝ) :=
  (total_oxygen - consume_diving_return) / consume_per_minute

theorem max_time_13_5_liters : max_underwater_time 13.5 8 0.3 ≈ 18 :=
sorry

end y_for_10min_y_range_6_10_max_time_13_5_liters_l341_341320
