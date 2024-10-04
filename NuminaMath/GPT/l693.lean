import Mathlib

namespace cos_832_eq_cos_l693_693249

theorem cos_832_eq_cos (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * Real.pi / 180) = Real.cos (832 * Real.pi / 180)) : n = 112 := 
  sorry

end cos_832_eq_cos_l693_693249


namespace smallest_scrumptious_l693_693429

def scrumptious (B : ℤ) : Prop :=
  ∃ (k : ℤ), (k > 0 ∧ (∀ n, n ≥ 0 ∧ n < k → ∑ i in range k, (B + i) = 2021))

theorem smallest_scrumptious : ∃ B, scrumptious B ∧ ∀ B', scrumptious B' → B ≤ B' :=
by
  let B := -2020
  existsi B
  split
  · use ∑ i in range 2021
  · sorry

end smallest_scrumptious_l693_693429


namespace ellipse_hyperbola_tangent_n_value_l693_693632

theorem ellipse_hyperbola_tangent_n_value :
  (∃ n : ℝ, (∀ x y : ℝ, 4 * x^2 + y^2 = 4 ∧ x^2 - n * (y - 1)^2 = 1) ↔ n = 3 / 2) :=
by
  sorry

end ellipse_hyperbola_tangent_n_value_l693_693632


namespace perimeter_of_triangle_DEF_eq_136_l693_693487

-- Define the conditions of the problem as hypotheses
variables (D E F P Q R : Type)
variable [Geometry (D E F P Q R)]  -- Assuming some Geometry instance to handle geometric concepts

-- Define given values
def radius_of_incircle (DEF : Triangle D E F) : ℝ := 13
def DP : ℝ := 17
def PE : ℝ := 31
def FQ : ℝ := 20

-- Define the perimeter calculation
def perimeter_of_triangle (DEF : Triangle D E F) : ℝ :=
  2 * (DP + PE + FQ)

-- The statement to be proved
theorem perimeter_of_triangle_DEF_eq_136 (DEF : Triangle D E F) :
  perimeter_of_triangle DEF = 136 := by
  -- The actual proof goes here; replacing with sorry for the template
  sorry

end perimeter_of_triangle_DEF_eq_136_l693_693487


namespace find_derivative_at_2_l693_693300

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x * (f' 1)
noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem find_derivative_at_2 :
  f' 2 = 0 := 
by 
  sorry

end find_derivative_at_2_l693_693300


namespace M_sufficient_not_necessary_for_N_l693_693407

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 2}

theorem M_sufficient_not_necessary_for_N (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → ¬ (a ∈ M)) :=
sorry

end M_sufficient_not_necessary_for_N_l693_693407


namespace solve_equation_l693_693045

theorem solve_equation (x : ℝ) (h₀ : x ≠ 1) (h₁ : (x + 1)^3 = 1 / (x - 1)) : 
  x = Real.cbrt 2 :=
by
  sorry

end solve_equation_l693_693045


namespace interval_solution_l693_693464

theorem interval_solution (x : ℝ) (h : |x - 1| + |x + 2| < 5) : x ∈ Ioo (-3 : ℝ) 2 := sorry

end interval_solution_l693_693464


namespace units_digit_sum_2_pow_a_5_pow_b_l693_693668

theorem units_digit_sum_2_pow_a_5_pow_b (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 100)
  (h2 : 1 ≤ b ∧ b ≤ 100) :
  (2 ^ a + 5 ^ b) % 10 ≠ 8 :=
sorry

end units_digit_sum_2_pow_a_5_pow_b_l693_693668


namespace intersection_of_M_and_N_l693_693286

variable (x : ℝ)

def M : set ℝ := {x | x + 2 ≥ 0}
def N : set ℝ := {x | x - 1 < 0}
def intersection : set ℝ := {x | -2 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N : M x ∩ N x = intersection x :=
sorry

end intersection_of_M_and_N_l693_693286


namespace not_consecutive_after_transform_l693_693074

theorem not_consecutive_after_transform (f : ℕ → ℕ) (g : ℕ → ℕ) (h : ∀ x y, (f (x + y))^2 + (g (x - y))^2 = 2 * (x^2 + y^2)) :
  ∀ (n : ℕ), n > 999 → ¬(∃ a : ℕ, ∃ (s : set ℕ), s = {a, a+1, ..., a+n-1}) :=
begin
  intros n hn,
  by_contradiction H,
  obtain ⟨a, s, hs⟩ := H,
  sorry
end

end not_consecutive_after_transform_l693_693074


namespace abs_c_minus_d_eq_56_l693_693674

def tau (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ d, n % d = 0).card + 1

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, tau (k + 1))

def c (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ k, S (k + 1) % 2 = 1).card

def d (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ k, S (k + 1) % 2 = 0).card

theorem abs_c_minus_d_eq_56 : ∀ n ≤ 1500, |c n - d n| = 56 :=
sorry

end abs_c_minus_d_eq_56_l693_693674


namespace value_can_be_10_l693_693336

theorem value_can_be_10 : 
  ∃ (f : List (ℕ → ℕ → ℕ)), 
  (f.length = 4 ∧
   (f = [has_add.add, has_sub.sub, has_mul.mul, has_div.div] ∨
    f = [has_add.add, has_sub.sub, has_div.div, has_mul.mul] ∨
    f = [has_add.add, has_mul.mul, has_sub.sub, has_div.div] ∨
    f = [has_add.add, has_mul.mul, has_div.div, has_sub.sub] ∨
    f = [has_add.add, has_div.div, has_sub.sub, has_mul.mul] ∨
    f = [has_add.add, has_div.div, has_mul.mul, has_sub.sub] ∨
    f = [has_sub.sub, has_add.add, has_mul.mul, has_div.div] ∨
    f = [has_sub.sub, has_add.add, has_div.div, has_mul.mul] ∨
    f = [has_sub.sub, has_mul.mul, has_add.add, has_div.div] ∨
    f = [has_sub.sub, has_mul.mul, has_div.div, has_add.add] ∨
    f = [has_sub.sub, has_div.div, has_add.add, has_mul.mul] ∨
    f = [has_sub.sub, has_div.div, has_mul.mul, has_add.add] ∨
    f = [has_mul.mul, has_add.add, has_sub.sub, has_div.div] ∨
    f = [has_mul.mul, has_add.add, has_div.div, has_sub.sub] ∨
    f = [has_mul.mul, has_sub.sub, has_add.add, has_div.div] ∨
    f = [has_mul.mul, has_sub.sub, has_div.div, has_add.add] ∨
    f = [has_mul.mul, has_div.div, has_add.add, has_sub.sub] ∨
    f = [has_mul.mul, has_div.div, has_sub.sub, has_add.add] ∨
    f = [has_div.div, has_add.add, has_sub.sub, has_mul.mul] ∨
    f = [has_div.div, has_add.add, has_mul.mul, has_sub.sub] ∨
    f = [has_div.div, has_sub.sub, has_add.add, has_mul.mul] ∨
    f = [has_div.div, has_sub.sub, has_mul.mul, has_add.add] ∨
    f = [has_div.div, has_mul.mul, has_add.add, has_sub.sub] ∨
    f = [has_div.div, has_mul.mul, has_sub.sub, has_add.add]) ∧
  (f.head 7 (f.tail.head 2 (f.tail.tail.head 3 (f.tail.tail.tail.head 4 5))) = 10) :=
sorry

end value_can_be_10_l693_693336


namespace area_triangle_HML_l693_693373

/-
  Given a triangle ABC with side lengths 13 cm, 14 cm, and 15 cm, and points H, M, and L being the intersection
  points of its altitudes, medians, and angle bisectors respectively, prove the area of triangle HML is 21 cm².
-/
theorem area_triangle_HML (A B C : Point) (H M L : Point)
  (hABC : triangle A B C)
  (h_side_AB : dist A B = 13) (h_side_BC : dist B C = 15) (h_side_AC : dist A C = 14)
  (hH : orthocenter H A B C) (hM : median M A B C) (hL : incenter L A B C) :
  area A B C / 4 = 21 := sorry

end area_triangle_HML_l693_693373


namespace square_perimeter_l693_693834

theorem square_perimeter (A : ℝ) (s : ℝ) (h1 : s^2 = 675) : 4 * s = 60 * real.sqrt 3 :=
  by
  sorry

end square_perimeter_l693_693834


namespace exists_a_l693_693318

noncomputable def solutions_of_quadratic (a : ℝ) : Set ℝ :=
  {x : ℝ | (a - 1) * x^2 + 3 * x - 2 = 0}

theorem exists_a (A : Set ℝ) :
  ∃ a : ℝ, (A = {x | (a - 1) * x^2 + 3 * x - 2 = 0}) ∧ (A = ∅ ∨ ∃ x, A = {x}) :=
begin
  sorry
end

end exists_a_l693_693318


namespace sum_of_partial_fraction_coeffs_l693_693393

noncomputable def polynomial := (λ x => x^3 - 18 * x^2 + 91 * x - 170)

theorem sum_of_partial_fraction_coeffs
  (a b c D E F : ℝ)
  (ha : polynomial a = 0)
  (hb : polynomial b = 0)
  (hc : polynomial c = 0)
  (hD : ∀ s, s ≠ a ∧ s ≠ b ∧ s ≠ c → (1 / polynomial s) = (D / (s - a) + E / (s - b) + F / (s - c))) :
  D + E + F = 0 := sorry

end sum_of_partial_fraction_coeffs_l693_693393


namespace expected_S_fraction_has_sum_l693_693158

-- Define the conditions of the problem.
def num_potatoes : ℕ := 2016
def num_chosen : ℕ := 1007

-- Define S based on the conditions in the problem.
noncomputable def S (p : ℕ) (remaining_potatoes : Finset ℕ) : ℚ :=
if ∃ q ∈ remaining_potatoes, q < p then
  let q := Classical.choose (exists_property (Finset.exists_mem_lt remaining_potatoes p)) in
  ↑(p * q)
else
  1

-- Define the expected value of S.
noncomputable def expected_S (chosen_potatoes remaining_potatoes : Finset ℕ) : ℚ :=
(Finset.sum (Finset.powersetLen num_chosen (Finset.range 2016)) 
  (λ chosen, S (chosen.min' (by simp)) (Finset.range 2016 \ chosen))) / 
↑(Finset.card (Finset.powersetLen num_chosen (Finset.range 2016)))

-- Prove that the expected value of S is equal to a fraction m/n and m+n = 2688.
theorem expected_S_fraction_has_sum (m n : ℕ) :
  (expected_S (Finset.range 2016) (Finset.range 2016 \ Finset.range 1007)) = (m / n : ℚ) → m + n = 2688 :=
by sorry

end expected_S_fraction_has_sum_l693_693158


namespace face_value_shares_l693_693561

theorem face_value_shares (market_value : ℝ) (dividend_rate desired_rate : ℝ) (FV : ℝ) 
  (h1 : dividend_rate = 0.09)
  (h2 : desired_rate = 0.12)
  (h3 : market_value = 36.00000000000001)
  (h4 : (dividend_rate * FV) = (desired_rate * market_value)) :
  FV = 48.00000000000001 :=
by
  sorry

end face_value_shares_l693_693561


namespace abs_ineq_sqrt_2005_l693_693035

theorem abs_ineq_sqrt_2005 (m n : ℤ) (hm : 0 < m) (hn : 0 < n) :
  |(n : ℝ) * real.sqrt 2005 - m| > 1 / (90 * n) :=
by
  sorry

end abs_ineq_sqrt_2005_l693_693035


namespace air_quality_levels_probabilities_average_number_of_people_exercising_contingency_table_and_relation_l693_693580

def air_quality_survey :=
  let excellent := (2 + 16 + 25)
  let good := (5 + 10 + 12)
  let mild_pollution := (6 + 7 + 8)
  let moderate_pollution := (7 + 2 + 0)
  let total_days := 100
  let prob_excellent := excellent / total_days
  let prob_good := good / total_days
  let prob_mild_pollution := mild_pollution / total_days
  let prob_moderate_pollution := moderate_pollution / total_days
  let average_exercise := 
    (1 / total_days) * (100 * (2 + 5 + 6 + 7) + 300 * (16 + 10 + 7 + 2) + 500 * (25 + 12 + 8))
  let table :=
    (33, 37, 70, 22, 8, 30, 55, 45, 100)
  let K_squared :=
    (total_days * (33 * 8 - 37 * 22)^2) / (70 * 30 * 55 * 45)
  let check_confidence :=
    K_squared > 3.841
  (prob_excellent, prob_good, prob_mild_pollution, prob_moderate_pollution, average_exercise, table, check_confidence)

theorem air_quality_levels_probabilities :
  let (prob_excellent, prob_good, prob_mild_pollution, prob_moderate_pollution, _, _, _) := air_quality_survey in
  prob_excellent = 0.43 ∧
  prob_good = 0.27 ∧
  prob_mild_pollution = 0.21 ∧
  prob_moderate_pollution = 0.09 := by
  sorry

theorem average_number_of_people_exercising :
  let (_, _, _, _, average_exercise, _, _) := air_quality_survey in
  average_exercise = 350 := by
  sorry

theorem contingency_table_and_relation :
  let (_, _, _, _, _, (ge_400, gt_400, total_g, le_400, lt_400, total_p, total_le_400, total_gt_400, total_t), check_confidence) := air_quality_survey in
  ge_400 = 33 ∧
  gt_400 = 37 ∧
  total_g = 70 ∧
  le_400 = 22 ∧
  lt_400 = 8 ∧
  total_p = 30 ∧
  total_le_400 = 55 ∧
  total_gt_400 = 45 ∧
  total_t = 100 ∧
  check_confidence = true := by
  sorry

end air_quality_levels_probabilities_average_number_of_people_exercising_contingency_table_and_relation_l693_693580


namespace largest_prime_factor_of_fact_sum_is_7_l693_693969

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693969


namespace complex_area_proof_l693_693366

noncomputable def complex_area_of_z (z : ℂ) : ℝ :=
if h : |z + complex.I| ≤ 1 then π else 0

theorem complex_area_proof (z : ℂ) (h : |z + complex.I| ≤ 1) : 
  complex_area_of_z z = π := 
by
  sorry

end complex_area_proof_l693_693366


namespace impossible_arrangement_of_300_numbers_in_circle_l693_693150

theorem impossible_arrangement_of_300_numbers_in_circle :
  ¬ ∃ (nums : Fin 300 → ℕ), (∀ i : Fin 300, nums i > 0) ∧
    ∃ unique_exception : Fin 300,
      ∀ i : Fin 300, i ≠ unique_exception → nums i = Int.natAbs (nums (Fin.mod (i.val - 1) 300) - nums (Fin.mod (i.val + 1) 300)) := 
sorry

end impossible_arrangement_of_300_numbers_in_circle_l693_693150


namespace problem_statement_l693_693387

open_locale classical

variables {A B C X Y N : Type*} [metric_space A] [metric_space B] [metric_space C]
[metric_space X] [metric_space Y] [metric_space N]
[is_triangle A B C] 

noncomputable def triangle_is_acute (ABC : triangle A B C) : Prop := 
ABC.is_acute ∧ dist A B < dist A C

noncomputable def points_on_minor_arc (X Y : Type*) 
[metric_space X] [metric_space Y]
(B C : Type*) [metric_space B] [metric_space C] : Prop := 
dist B X = dist X Y ∧ dist X Y = dist Y C

noncomputable def point_on_segment (N : Type*) 
[metric_space N] (A Y : Type*) 
[metric_space A] [metric_space Y] [segment A Y N] : Prop := 
dist A N = dist N C

theorem problem_statement (ABC : triangle A B C)
(X Y : points_on_minor_arc X Y B C)
(N : point_on_segment N A Y)
(h1 : triangle_is_acute ABC) 
(h2 : dist B X = dist X Y ∧ dist X Y = dist Y C)
(h3 : dist A N = dist N C ) :
∃ M, is_midpoint M A X ∧ line N C passes_through M :=
sorry

end problem_statement_l693_693387


namespace negation_proposition_l693_693063

theorem negation_proposition:
  (¬ (∀ x : ℝ, (1 ≤ x) → (x^2 - 2*x + 1 ≥ 0))) ↔ (∃ x : ℝ, (1 ≤ x) ∧ (x^2 - 2*x + 1 < 0)) := 
sorry

end negation_proposition_l693_693063


namespace find_solutions_eq_zero_l693_693800

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 5 * x - 7 else -3 * x + 12 

theorem find_solutions_eq_zero :
  (f (7 / 5) = 0) ∧ (f 4 = 0) :=
by
  sorry

end find_solutions_eq_zero_l693_693800


namespace ones_digit_of_9_pow_46_l693_693122

theorem ones_digit_of_9_pow_46 : (9 ^ 46) % 10 = 1 :=
by
  sorry

end ones_digit_of_9_pow_46_l693_693122


namespace largest_prime_factor_7fac_8fac_l693_693944

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693944


namespace circumference_of_tank_B_l693_693048

-- Define the problem conditions
variables (h_C : ℝ) (C_C : ℝ) (h_B : ℝ) (V_C : ℝ) (V_B : ℝ)
variables (r_C : ℝ) (r_B : ℝ) (r_C_def : r_C = C_C / (2 * Real.pi))
variables (V_C_def : V_C = Real.pi * r_C^2 * h_C)
variables (V_B_def : V_B = 5 * V_C / 4)
variables (r_B_def : r_B = Real.sqrt (V_B / (Real.pi * h_B)))
variables (C_B : ℝ) (C_B_def : C_B = 2 * Real.pi * r_B)

-- Define the conditions explicitly as hypotheses
hypothesis hC : h_C = 10
hypothesis CC  : C_C = 8
hypothesis hB  : h_B = 8
hypothesis VC : V_C = Real.pi * (C_C / (2 * Real.pi))^2 * h_C
hypothesis VB : V_B = (5 / 4) * V_C

-- Lean theorem stating the proof goal
theorem circumference_of_tank_B :
  C_B = 10 :=
by
  sorry

end circumference_of_tank_B_l693_693048


namespace simplify_fraction_l693_693439

theorem simplify_fraction (num denom : ℕ) (h_num : num = 90) (h_denom : denom = 150) : 
  num / denom = 3 / 5 := by
  rw [h_num, h_denom]
  norm_num
  sorry

end simplify_fraction_l693_693439


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693895

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693895


namespace isosceles_triangle_area_l693_693485

variable {A B C K L: Type}
variable [Real A] [Real B] [Real C] [Real K] [Real L]

noncomputable def triangle_area (H : ℝ) : ℝ := H^2 * Real.sqrt 3

theorem isosceles_triangle_area (H : ℝ) (AB AC : ℝ) (BK : ℝ) (BL : ℝ) (hp1 : AB = AC) (hp2 : BK = H) 
  (hp3 : BK = 2 * BL) :
  (∃ (S : ℝ), S = 1/2 * BK * (2 * (H * Real.sqrt 3)) → S = triangle_area H) :=
  sorry

end isosceles_triangle_area_l693_693485


namespace average_sale_six_months_l693_693556

-- Define the sales for the first five months
def sale_month1 : ℕ := 6335
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562

-- Define the required sale for the sixth month
def sale_month6 : ℕ := 5091

-- Proof that the desired average sale for the six months is 6500
theorem average_sale_six_months : 
  (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6500 :=
by
  sorry

end average_sale_six_months_l693_693556


namespace compute_63_times_57_l693_693614

theorem compute_63_times_57 : 63 * 57 = 3591 := 
by {
   have h : (60 + 3) * (60 - 3) = 60^2 - 3^2, from
     by simp [mul_add, add_mul, add_assoc, sub_mul, mul_sub, sub_add, sub_sub, add_sub, mul_self_sub],
   have h1 : 60^2 = 3600, from rfl,
   have h2 : 3^2 = 9, from rfl,
   have h3 : 60^2 - 3^2 = 3600 - 9, by rw [h1, h2],
   rw h at h3,
   exact h3,
}

end compute_63_times_57_l693_693614


namespace triangle_angles_l693_693187

noncomputable def angle_of_triangle (a b c : ℝ) : ℝ := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

theorem triangle_angles :
  let a := 3
  let b := 3
  let c := Real.sqrt 8
  ∃ A B C : ℝ, 
    A = 61.78 ∧
    B = 61.78 ∧
    C = 56.44 ∧
    angle_of_triangle a a c = C ∧
    angle_of_triangle a c a = A ∧
    angle_of_triangle c a a = B :=
by
  let a := 3
  let b := 3
  let c := Real.sqrt 8
  use 61.78, 61.78, 56.44
  split; [refl, split; [refl, split; [refl, sorry, sorry]]]

end triangle_angles_l693_693187


namespace blue_vs_yellow_bin_probability_l693_693546

theorem blue_vs_yellow_bin_probability :
  let prob_k (k : ℕ) := (1 / 3 : ℝ) ^ k in
  let prob_same_bin := ∑' k, (prob_k k) ^ 2 in
  let higher_bin_probability := (1 - prob_same_bin) / 2 in
  higher_bin_probability = (7 / 16 : ℝ) :=
by
  sorry

end blue_vs_yellow_bin_probability_l693_693546


namespace right_triangles_count_l693_693180

-- Define the grid points according to the problem conditions
def points : List (ℝ × ℝ) := [(0,0), (0,1), (1,0), (1,0.5), (1,1), (2,0), (2,1), (3,0), (3,0.5), (3,1)]

-- Define a right triangle where one side must be parallel to x-axis or y-axis
structure right_triangle (a b c : ℝ × ℝ) : Prop :=
(is_right_triangle : right_triangle_side_parallel a b c)

-- A placeholder for checking if one side is parallel to the x or y axis
def right_triangle_side_parallel (a b c : ℝ × ℝ) : Prop := sorry

-- The theorem statement
theorem right_triangles_count : 
  ∃ (triangles : Finset (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)), 
  (∀ t ∈ triangles, right_triangle t.1 t.2 t.3) ∧ 
  Finset.card triangles = 10 :=
sorry

end right_triangles_count_l693_693180


namespace sum_of_reciprocals_of_roots_l693_693700

noncomputable def polynomial_has_roots_cyclotomic_circle 
  (a b c d : ℝ) : Prop :=
  ∃ z1 z2 z3 z4 : ℂ, 
    (z1^4 + a * z1^3 + b * z1^2 + c * z1 + d = 0) ∧ 
    (z2^4 + a * z2^3 + b * z2^2 + c * z2 + d = 0) ∧ 
    (z3^4 + a * z3^3 + b * z3^2 + c * z3 + d = 0) ∧ 
    (z4^4 + a * z4^3 + b * z4^2 + c * z4 + d = 0) ∧ 
    |z1| = 1 ∧ |z2| = 1 ∧ |z3| = 1 ∧ |z4| = 1

theorem sum_of_reciprocals_of_roots 
    (a b c d : ℝ) 
    (h : polynomial_has_roots_cyclotomic_circle a b c d)
  : ∑ z : ℂ in {z1, z2, z3, z4}, z⁻¹ = -a := 
sorry

end sum_of_reciprocals_of_roots_l693_693700


namespace find_alpha_l693_693328

-- Given conditions
variables (α β : ℝ)
axiom h1 : α + β = 11
axiom h2 : α * β = 24
axiom h3 : α > β

-- Theorems to prove
theorem find_alpha : α = 8 :=
  sorry

end find_alpha_l693_693328


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693892

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693892


namespace parabola_equation_chord_midpoint_l693_693701

-- Proof Problem 1: Equation of the parabola E
theorem parabola_equation (p : ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) (E : ℝ → ℝ → Prop)
  (h1 : 0 < p)            -- p > 0
  (h2 : M = (2, M.2))      -- M(2, m)
  (h3 : F = (p / 2, 0))    -- Focus F of y^2 = 2px
  (h4 : |(M.1 - F.1, M.2 - F.2)| = 3) -- |MF| = 3
  (hE : ∀ x y, E x y ↔ y^2 = 2 * p * x) :
  (∀ y x, E x y ↔ y^2 = 4 * x) := -- Equation of parabola E is y^2 = 4x
sorry

-- Proof Problem 2: Equation of the line containing the chord
theorem chord_midpoint (E : ℝ → ℝ → Prop) (N : ℝ × ℝ) (k : ℝ) (line : ℝ → ℝ → Prop)
  (hE : ∀ x y, E x y ↔ y^2 = 4x) -- Equation of parabola E is y^2 = 4x
  (hN : N = (1, 1))             -- N(1,1) is the midpoint
  (hline : ∀ x y, line x y ↔ y - 1 = k * (x - 1)) (hk : k = 2) :
  (∀ x y, line x y ↔ 2 * x - y - 1 = 0) := -- Equation of the line is 2x - y - 1 = 0
sorry

end parabola_equation_chord_midpoint_l693_693701


namespace minimum_students_l693_693751

-- Define the variables and conditions
variables (b g : ℕ)
hypothesis (h1 : 1/2 * b = 2/3 * g)

-- Define the main theorem to proof the minimum possible number of students
theorem minimum_students (h1 : 1/2 * b = 2/3 * g) : b + g = 7 :=
sorry

end minimum_students_l693_693751


namespace parallelogram_AQ_AC_ratio_l693_693032

theorem parallelogram_AQ_AC_ratio (A B C D P Q : Point) (n : ℕ)
  (h_parallelogram: is_parallelogram A B C D)
  (h_P_on_AD: ∃ t : ℝ, 0 < t ∧ t < 1 ∧ t * vector_from A D = vector_from A P ∧ (AP:AD) = 1/n)
  (h_intersection_Q: ∃ t u : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ u ∧ u ≤ 1 ∧ (t * AC + (1 - t) * BP = Q)) :
  (AQ : AC) = 1 / (n + 1) := 
  sorry

end parallelogram_AQ_AC_ratio_l693_693032


namespace midpoint_diagonal_split_l693_693693

variable {Point : Type}
variable {Quadrilateral : Type}

structure Quadrilateral (ABCD : Quadrilateral) where
  A B C D E : Point
  is_divided_into_four_equal_area_triangles : ∀ (P Q R : Point → Prop), 
    is_triangle ABO ∧ is_triangle BCO ∧ is_triangle COD ∧ is_triangle DOA 
    → area ABO = area BCO ∧ area BCO = area COD ∧ area COD = area DOA

noncomputable def is_midpoint (E A C : Point) : Prop :=
  ∃ M, (M = midpoint A C) ∧ (E = M)

theorem midpoint_diagonal_split {ABCD : Quadrilateral} (h : Quadrilateral ABCD) :
  ∃ E, (is_midpoint E (h.A) (h.C)) ∨ (is_midpoint E (h.B) (h.D)) :=
sorry

end midpoint_diagonal_split_l693_693693


namespace largest_prime_factor_7fac_8fac_l693_693940

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693940


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693920

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693920


namespace calculate_regular_rate_l693_693077

def regular_hours_per_week : ℕ := 6 * 10
def total_weeks : ℕ := 4
def total_regular_hours : ℕ := regular_hours_per_week * total_weeks
def total_worked_hours : ℕ := 245
def overtime_hours : ℕ := total_worked_hours - total_regular_hours
def overtime_rate : ℚ := 4.20
def total_earning : ℚ := 525
def total_overtime_pay : ℚ := overtime_hours * overtime_rate
def total_regular_pay : ℚ := total_earning - total_overtime_pay
def regular_rate : ℚ := total_regular_pay / total_regular_hours

theorem calculate_regular_rate : regular_rate = 2.10 :=
by
  -- The proof would go here
  sorry

end calculate_regular_rate_l693_693077


namespace largest_prime_factor_7fac_8fac_l693_693937

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693937


namespace simplify_expression_l693_693461

theorem simplify_expression : 18 * (8 / 15) * (1 / 12) = 4 / 5 :=
by
  sorry

end simplify_expression_l693_693461


namespace simplify_fraction_l693_693435

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end simplify_fraction_l693_693435


namespace value_of_a_l693_693351

-- Define the three lines as predicates
def line1 (x y : ℝ) : Prop := x + y = 1
def line2 (x y : ℝ) : Prop := x - y = 1
def line3 (a x y : ℝ) : Prop := a * x + y = 1

-- Define the condition that the lines do not form a triangle
def lines_do_not_form_triangle (a x y : ℝ) : Prop :=
  (∀ x y, line1 x y → ¬line3 a x y) ∨
  (∀ x y, line2 x y → ¬line3 a x y) ∨
  (a = 1)

theorem value_of_a (a : ℝ) :
  (¬ ∃ x y, line1 x y ∧ line2 x y ∧ line3 a x y) →
  lines_do_not_form_triangle a 1 0 →
  a = -1 :=
by
  intro h1 h2
  sorry

end value_of_a_l693_693351


namespace non_repeating_combinations_count_l693_693724

/-- Given three sets of prime numbers: A = {3, 11}, B = {5, 41}, and C = {7, 29},
this theorem proves that the number of non-repeating combinations that can be formed
under the following conditions is 16:

1. No number is repeated within a single combination.
2. Each combination must use exactly two numbers from each set.
3. The combinations must start with a number from set A and end with a number from set C.
4. Set B numbers cannot be consecutive within the combinations.
-/
theorem non_repeating_combinations_count
  (A : Set ℕ)
  (hA : A = {3, 11})
  (B : Set ℕ)
  (hB : B = {5, 41})
  (C : Set ℕ)
  (hC : C = {7, 29})
  (no_repeat : ∀ (comb : List ℕ), comb.Nodup)
  (start_with_A : ∀ (comb : List ℕ), comb.head ∈ A)
  (end_with_C : ∀ (comb : List ℕ), comb.last ∈ C)
  (non_consecutive_B : ∀ (comb : List ℕ), ∀ i, comb.nth i ∈ B → comb.nth (i + 1) ∉ B) :
  ∃ n, n = 16 := 
sorry

end non_repeating_combinations_count_l693_693724


namespace problem1_problem2_l693_693287

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

variable {x : ℝ}

-- Problem 1
theorem problem1 (a : ℕ → ℝ) (h : (1 - x)^11 = ∑ k in finset.range 12, a k * x^k) :
  |a 6| + |a 7| + |a 8| + |a 9| + |a 10| + |a 11| = 1024 := 
  sorry

-- Problem 2
variable (n : ℕ) (a b : ℕ → ℝ) (S : ℕ → ℝ)

-- Assuming the binomial coefficients
variable (C : ℕ → ℕ → ℕ)
  [hC : ∀ (n k : ℕ), C n k = binomial_coefficient n k] 

theorem problem2 {n : ℕ} (h1 : n ≥ 2) 
  (ha: ∀ k, a (k + 1) = (-1)^k * C n (k + 1)) 
  (hb : ∀ k : ℕ, k ≤ n - 1 → b k = (k + 1) / (n - k) * a (k + 1))
  (hS : ∀ m, S m = ∑ j in finset.range (m + 1), b j)
  {m : ℕ} (hm : m ≤ n - 1) :
  abs (S m / C (n - 1) m) = 1 :=
  sorry

end problem1_problem2_l693_693287


namespace smallest_distance_condition_l693_693011

open Complex

noncomputable def a : ℂ := -2 - 4 * I
noncomputable def b : ℂ := 6 + 7 * I

theorem smallest_distance_condition (z w : ℂ) 
  (hz : abs (z - a) = 2) 
  (hw : abs (w - b) = 4) : 
  abs (z - w) ≥ real.sqrt 185 - 6 :=
sorry

end smallest_distance_condition_l693_693011


namespace length_PZ_proof_l693_693368

open Classical
variable {Point : Type} [EuclideanGeometry Point]

def parallel_segments {A B C D : Point} : Prop := 
  ∃ l, A ∈ l ∧ B ∈ l ∧ ∃ l', C ∈ l' ∧ D ∈ l' ∧ ∀ (p : Point), p ∈ l ∨ p ∈ l' 

def length (P Q : Point) : ℝ := 
  Real.sqrt ((Q.x - P.x) ^ 2 + (Q.y - P.y) ^ 2)

def CX := 50 : ℝ
def DP := 15 : ℝ
def PW := 30 : ℝ

variable (C D X W P Z : Point)

axiom parallel_CD_XW : parallel_segments C D X W
axiom e_CX : length C X = CX
axiom e_DP : length D P = DP
axiom e_PW : length P W = PW

def length_PZ (P Z : Point) := length P Z

theorem length_PZ_proof : length_PZ P Z = 35 := by
  sorry

end length_PZ_proof_l693_693368


namespace grunters_prob_win_5_out_of_6_l693_693475

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem grunters_prob_win_5_out_of_6 :
  binomial_probability 6 5 (4/5) = 6144 / 15625 := 
sorry

end grunters_prob_win_5_out_of_6_l693_693475


namespace frisbee_sales_total_receipts_l693_693575

theorem frisbee_sales_total_receipts 
  (total_frisbees : ℕ) 
  (price_3_frisbee : ℕ) 
  (price_4_frisbee : ℕ) 
  (sold_3 : ℕ) 
  (sold_4 : ℕ) 
  (total_receipts : ℕ) 
  (h1 : total_frisbees = 60) 
  (h2 : price_3_frisbee = 3)
  (h3 : price_4_frisbee = 4) 
  (h4 : sold_3 + sold_4 = total_frisbees) 
  (h5 : sold_4 ≥ 24)
  (h6 : total_receipts = sold_3 * price_3_frisbee + sold_4 * price_4_frisbee) :
  total_receipts = 204 :=
sorry

end frisbee_sales_total_receipts_l693_693575


namespace total_students_correct_l693_693491

-- Define the given conditions
variables (A B C : ℕ)

-- Number of students in class B
def B_def : ℕ := 25

-- Number of students in class A (B is 8 fewer than A)
def A_def : ℕ := B_def + 8

-- Number of students in class C (C is 5 times B)
def C_def : ℕ := 5 * B_def

-- The total number of students
def total_students : ℕ := A_def + B_def + C_def

-- The proof statement
theorem total_students_correct : total_students = 183 := by
  sorry

end total_students_correct_l693_693491


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693102

theorem frequencies_of_first_class_products 
  (aA bA aB bB : ℕ) 
  (total_A total_B total_products : ℕ) 
  (hA : aA = 150) (hB : bA = 50) (hC : aB = 120) (hD : bB = 80) 
  (hE : total_A = 200) (hF : total_B = 200) (h_total : total_products = total_A + total_B) : 
  aA / total_A = 3 / 4 ∧ aB / total_B = 3 / 5 := 
by
  sorry

theorem confidence_in_difference_of_quality 
  (n a b c d : ℕ) 
  (h_n : n = 400) (hA : a = 150) (hB : b = 50) (hC : c = 120) (hD : d = 80) 
  (K2 : ℝ) 
  (hK2 : K2 = n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))) : 
  K2 > 6.635 ∧ K2 < 10.828 := 
by
  sorry

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693102


namespace problem_solution_l693_693767

def parametric_x (t : ℝ) : ℝ := (Real.sqrt 2) / 2 * t
def parametric_y (t : ℝ) : ℝ := 2 + (Real.sqrt 2) / 2 * t

def polar_radius (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)
def polar_angle (x y : ℝ) : ℝ := Real.atan2 y x

def parametric_point (t : ℝ) : ℝ × ℝ := (parametric_x t, parametric_y t)
def polar_of_point (t : ℝ) : ℝ × ℝ :=
  let (x, y) := parametric_point t
  (polar_radius x y, polar_angle x y)

def curve_C_eq : Prop := ∃ x y : ℝ, x^2 + y^2 = 16

def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b^2 - 4 * a * c
  ((-b + Real.sqrt discriminant) / (2 * a), (-b - Real.sqrt discriminant) / (2 * a))

def intersection_point_dist (t : ℝ) : ℝ := Real.sqrt ((parametric_x t)^2 + (parametric_y t - 2)^2)

def reciprocal_sum_of_distances
  (t1 t2 : ℝ) : ℝ :=
  1 / (intersection_point_dist t1) + 1 / (intersection_point_dist t2)

theorem problem_solution :
  (parametric_point (-Real.sqrt 2) = (-1, 1) ∧ polar_of_point (-Real.sqrt 2) = (Real.sqrt 2, 3 * Real.pi / 4)) ∧ curve_C_eq ∧
  ∀ (t1 t2 : ℝ),
  (t1 + t2 = -2 * Real.sqrt 2) ∧ (t1 * t2 = -12) →
  reciprocal_sum_of_distances t1 t2 = Real.sqrt 14 / 6 :=
by sorry

end problem_solution_l693_693767


namespace probability_divisible_by_3_l693_693141

noncomputable def prime_digit_two_digit_integers : List ℕ :=
  [23, 25, 27, 32, 35, 37, 52, 53, 57, 72, 73, 75]

noncomputable def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- The statement
theorem probability_divisible_by_3 :
  let favorable_outcomes := prime_digit_two_digit_integers.filter is_divisible_by_3
  let total_outcomes := prime_digit_two_digit_integers.length
  let r := favorable_outcomes.length / total_outcomes
  r = 1 / 3 := 
sorry

end probability_divisible_by_3_l693_693141


namespace octal_subtraction_correct_l693_693657

theorem octal_subtraction_correct : 
  nat.ofDigits 8 [6, 5, 4, 3] - nat.ofDigits 8 [4, 3, 2, 1] = nat.ofDigits 8 [2, 2, 2, 2] :=
by sorry

end octal_subtraction_correct_l693_693657


namespace distance_from_center_to_line_l693_693659

theorem distance_from_center_to_line :
    let center := (-1, 0 : ℝ × ℝ)
    let line (p : ℝ × ℝ) := 2 * p.1 - p.2 + 3
    let distance (center : ℝ × ℝ) (line : ℝ × ℝ → ℝ) :=
        |line center| / real.sqrt (2^2 + (-1)^1)
    distance center line = real.sqrt 5 / 5 :=
by
    -- We have identified all conditions from the problem, and the expected answer is provided.
    -- The function 'distance' computes the point-to-line distance according to the formula provided.
    sorry

end distance_from_center_to_line_l693_693659


namespace positive_difference_x_coordinates_l693_693266

-- Definition of line p passing through points (0, 8) and (4, 0)
def line_p (x : ℝ) : ℝ := (-2) * x + 8

-- Definition of line q passing through points (0, 3) and (10, 0)
def line_q (x : ℝ) : ℝ := (-0.3) * x + 3

theorem positive_difference_x_coordinates : 
  let yp := 20
  let yq := 20
  let xp := (-line_p.1(yp - 8) / -2)
  let xq := (-line_q.1(yp - 3) / -0.3)
  abs(xp - xq) = 50.67 :=
by 
  let x_p := -6
  let x_q := -56.67
  have h₀ : xp = x_p := sorry
  have h₁ : xq = x_q := sorry
  calc
    abs(xp - xq)
        = abs(x_p - x_q) : by rw [h₀, h₁]
    ... = 50.67 : by sorry

end positive_difference_x_coordinates_l693_693266


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693100

theorem frequencies_of_first_class_products 
  (aA bA aB bB : ℕ) 
  (total_A total_B total_products : ℕ) 
  (hA : aA = 150) (hB : bA = 50) (hC : aB = 120) (hD : bB = 80) 
  (hE : total_A = 200) (hF : total_B = 200) (h_total : total_products = total_A + total_B) : 
  aA / total_A = 3 / 4 ∧ aB / total_B = 3 / 5 := 
by
  sorry

theorem confidence_in_difference_of_quality 
  (n a b c d : ℕ) 
  (h_n : n = 400) (hA : a = 150) (hB : b = 50) (hC : c = 120) (hD : d = 80) 
  (K2 : ℝ) 
  (hK2 : K2 = n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))) : 
  K2 > 6.635 ∧ K2 < 10.828 := 
by
  sorry

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693100


namespace decreasing_implies_increasing_l693_693788

variable (a : ℝ) (f g : ℝ → ℝ)

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f y < f x

def is_increasing (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → g x < g y

theorem decreasing_implies_increasing (ha : a > 0) (ha_ne : a ≠ 1) :
  (is_decreasing (λ x, a ^ x)) → (is_increasing (λ x, (2 - a) * x^3)) :=
by
  sorry

end decreasing_implies_increasing_l693_693788


namespace matrix_inversion_l693_693633

theorem matrix_inversion (d p : ℝ) (h₀ : d ≠ 3)
  (h₁ : (matrix.inv ![![1, 4], ![6, d]] = (p : ℝ) •![![1, 4], ![6, d]])) :
  (d = -1) ∧ (p = 1 / 25) := 
sorry

end matrix_inversion_l693_693633


namespace fraction_division_l693_693731

theorem fraction_division : (3 / 4) / (2 / 5) = 15 / 8 := by
  sorry

end fraction_division_l693_693731


namespace phase_shift_of_cosine_function_l693_693254

theorem phase_shift_of_cosine_function :
  ∀ (x : ℝ), ∃ (C : ℝ), C = -π / 4 ∧ 2 * cos (x + π / 4) = 2 * cos (x - (-π / 4)) := by
sorry

end phase_shift_of_cosine_function_l693_693254


namespace triangle_perimeter_l693_693066

theorem triangle_perimeter (MN NP MP : ℝ)
  (h1 : MN - NP = 18)
  (h2 : MP = 40)
  (h3 : MN / NP = 28 / 12) : 
  MN + NP + MP = 85 :=
by
  -- Proof is omitted
  sorry

end triangle_perimeter_l693_693066


namespace max_small_packages_l693_693489

theorem max_small_packages (L S : ℝ) (W : ℝ) (h1 : W = 12 * L) (h2 : W = 20 * S) :
  (∃ n_smalls, n_smalls = 5 ∧ W - 9 * L = n_smalls * S) :=
by
  sorry

end max_small_packages_l693_693489


namespace largest_prime_factor_l693_693982

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693982


namespace xy_correct_l693_693819

noncomputable def xy_value (x y : ℝ) (h1 : 2^x = 256^(y + 1)) (h2 : 27^y = 3^(x - 2)) : ℝ :=
  x * y

theorem xy_correct (x y : ℝ) (h1 : 2^x = 256^(y + 1)) (h2 : 27^y = 3^(x - 2)) : 
  xy_value x y h1 h2 = 48 / 25 :=
by
  sorry

end xy_correct_l693_693819


namespace basketball_game_first_half_points_l693_693558

theorem basketball_game_first_half_points (a b r d : ℕ) (H1 : a = b)
  (H2 : a * (1 + r + r^2 + r^3) = 4 * a + 6 * d + 1) 
  (H3 : 15 * a ≤ 100) (H4 : b + (b + d) + b + 2 * d + b + 3 * d < 100) : 
  (a + a * r + b + b + d) = 34 :=
by sorry

end basketball_game_first_half_points_l693_693558


namespace proof_a_b_sum_proof_a_b_diff_l693_693814

theorem proof_a_b_sum (a b c : ℝ) (α β γ : ℝ) (h₁ : a / sin α = c / sin γ) (h₂ : b / sin β = c / sin γ) :
  (a + b) / c = (cos ((α - β) / 2)) / (sin (γ / 2)) := 
sorry

theorem proof_a_b_diff (a b c : ℝ) (α β γ : ℝ) (h₁ : a / sin α = c / sin γ) (h₂ : b / sin β = c / sin γ) :
  (a - b) / c = (sin ((α - β) / 2)) / (cos (γ / 2)) := 
sorry

end proof_a_b_sum_proof_a_b_diff_l693_693814


namespace triangle_inequality_third_side_l693_693304

theorem triangle_inequality_third_side (a : ℝ) (h1 : 3 + a > 7) (h2 : 7 + a > 3) (h3 : 3 + 7 > a) : 
  4 < a ∧ a < 10 :=
by sorry

end triangle_inequality_third_side_l693_693304


namespace sum_of_three_base4_numbers_l693_693256

theorem sum_of_three_base4_numbers :
  let a := "203_4".to_list.base_to_nat 4
  let b := "112_4".to_list.base_to_nat 4
  let c := "330_4".to_list.base_to_nat 4
  (a + b + c).nat_to_base 4 = "13110_4".to_list.base_to_nat 4 :=
by sorry

end sum_of_three_base4_numbers_l693_693256


namespace sin_cos_difference_theorem_tan_theorem_l693_693678

open Real

noncomputable def sin_cos_difference (x : ℝ) : Prop :=
  -π / 2 < x ∧ x < 0 ∧ (sin x + cos x = 1 / 5) ∧ (sin x - cos x = - 7 / 5)

theorem sin_cos_difference_theorem (x : ℝ) (h : sin_cos_difference x) : 
  sin x - cos x = - 7 / 5 := by
  sorry

noncomputable def sin_cos_ratio (x : ℝ) : Prop :=
  -π / 2 < x ∧ x < 0 ∧ (sin x + cos x = 1 / 5) ∧ (sin x - cos x = - 7 / 5) ∧ (tan x = -3 / 4)

theorem tan_theorem (x : ℝ) (h : sin_cos_ratio x) :
  tan x = -3 / 4 := by
  sorry

end sin_cos_difference_theorem_tan_theorem_l693_693678


namespace beef_not_used_l693_693378

-- Define the context and necessary variables
variable (totalBeef : ℕ) (usedVegetables : ℕ)
variable (beefUsed : ℕ)

-- The conditions given in the problem
def initial_beef : Prop := totalBeef = 4
def used_vegetables : Prop := usedVegetables = 6
def relation_vegetables_beef : Prop := usedVegetables = 2 * beefUsed

-- The statement we need to prove
theorem beef_not_used
  (h1 : initial_beef totalBeef)
  (h2 : used_vegetables usedVegetables)
  (h3 : relation_vegetables_beef usedVegetables beefUsed) :
  (totalBeef - beefUsed) = 1 := by
  sorry

end beef_not_used_l693_693378


namespace sum_of_roots_tangential_eq_3pi_l693_693666

theorem sum_of_roots_tangential_eq_3pi :
  (∑ (x : ℝ) in (set.filter (λ x, 0 ≤ x ∧ x < 2 * real.pi) (set_of (λ x, tan x = 3 + real.sqrt 7 ∨ tan x = 3 - real.sqrt 7))), x) = 3 * real.pi :=
sorry

end sum_of_roots_tangential_eq_3pi_l693_693666


namespace triangle_fold_angle_l693_693597

theorem triangle_fold_angle (A B C D E : Type) (angle : A → ℝ) : 
  angle B = 74 ∧ angle A = 70 ∧ angle CEB = 20 → angle ADC = 92 := 
by
  sorry

end triangle_fold_angle_l693_693597


namespace key_count_l693_693499

theorem key_count (complexes : ℕ) (apartments_per_complex : ℕ) (keys_per_apartment : ℕ) 
  (h_complexes : complexes = 2) 
  (h_apartments_per_complex : apartments_per_complex = 12) 
  (h_keys_per_apartment : keys_per_apartment = 3) : 
  complexes * apartments_per_complex * keys_per_apartment = 72 := 
by 
  rw [h_complexes, h_apartments_per_complex, h_keys_per_apartment] 
  norm_num
  sorry

end key_count_l693_693499


namespace translate_point_D_l693_693430

-- Define the initial points and the translation vector
def A : ℝ × ℝ := (-1, 4)
def C : ℝ × ℝ := (3, 6)
def D : ℝ × ℝ := (-3, 2)
def translation_vector := (C.1 - A.1, C.2 - A.2)

-- Coordinate of the corresponding point of D after translation
def D_corresponding := (D.1 + translation_vector.1, D.2 + translation_vector.2)

-- Prove that the coordinates of the corresponding point of D are (1, 4)
theorem translate_point_D : D_corresponding = (1, 4) :=
by
  unfold translation_vector
  unfold D_corresponding
  simp
  done

end translate_point_D_l693_693430


namespace solve_problem_l693_693473

noncomputable def problem (n : ℕ) (a : ℕ) : Prop :=
  a = (10^(2*n) - 1) / (3 * (10^n + 1)) ∧ (a.digits.sum = 567) → n = 189

theorem solve_problem : ∃ n a, problem n a :=
  sorry

end solve_problem_l693_693473


namespace additional_profit_is_80000_l693_693056

-- Define the construction cost of a regular house
def construction_cost_regular (C : ℝ) : ℝ := C

-- Define the construction cost of the special house
def construction_cost_special (C : ℝ) : ℝ := C + 200000

-- Define the selling price of a regular house
def selling_price_regular : ℝ := 350000

-- Define the selling price of the special house
def selling_price_special : ℝ := 1.8 * 350000

-- Define the profit from selling a regular house
def profit_regular (C : ℝ) : ℝ := selling_price_regular - (construction_cost_regular C)

-- Define the profit from selling the special house
def profit_special (C : ℝ) : ℝ := selling_price_special - (construction_cost_special C)

-- Define the additional profit made by building and selling the special house compared to a regular house
def additional_profit (C : ℝ) : ℝ := (profit_special C) - (profit_regular C)

-- Theorem to prove the additional profit is $80,000
theorem additional_profit_is_80000 (C : ℝ) : additional_profit C = 80000 :=
sorry

end additional_profit_is_80000_l693_693056


namespace already_installed_windows_l693_693564

-- Definitions based on given conditions
def total_windows : ℕ := 9
def hours_per_window : ℕ := 6
def remaining_hours : ℕ := 18

-- Main statement to prove
theorem already_installed_windows : (total_windows - remaining_hours / hours_per_window) = 6 :=
by
  -- To prove: total_windows - (remaining_hours / hours_per_window) = 6
  -- This step is intentionally left incomplete (proof to be filled in by the user)
  sorry

end already_installed_windows_l693_693564


namespace max_c_for_range_l693_693662

theorem max_c_for_range (c : ℝ) :
  (∃ x : ℝ, (x^2 - 7*x + c = 2)) → c ≤ 57 / 4 :=
by
  sorry

end max_c_for_range_l693_693662


namespace pqr_value_l693_693471

theorem pqr_value (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h1 : p + q + r = 30) 
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + (420 : ℚ) / (p * q * r) = 1) : 
  p * q * r = 1800 := 
sorry

end pqr_value_l693_693471


namespace simplify_polynomial_l693_693119

theorem simplify_polynomial (x : ℝ) :
  3 + 5 * x - 7 * x^2 - 9 + 11 * x - 13 * x^2 + 15 - 17 * x + 19 * x^2 = 9 - x - x^2 := 
  by {
  -- placeholder for the proof
  sorry
}

end simplify_polynomial_l693_693119


namespace options_valid_l693_693706

noncomputable def sequence (n : ℕ) (λ : ℝ) : ℝ :=
  λ / (n + λ - 1)

def sum_first_n (n : ℕ) (λ : ℝ) : ℝ :=
  (finset.range n).sum (λ i, sequence (i + 1) λ)

theorem options_valid (λ : ℝ) (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a 1 = 1 ∧ a (n + 1) * a n = λ * (a n - a (n + 1)) ∧ a n ≠ 0) →
  (S n = sum_first_n n λ) →
  (∃ λ : ℝ, ∀ n, sequence (n + 1) λ > 0) ∧
  (∃ λ : ℝ, ∀ n, sequence (n + 1) λ < sequence n λ) ∧
  (∃ λ : ℝ, ∀ n, sum_first_n (n + 1) λ < sum_first_n n λ) :=
begin
  sorry
end

end options_valid_l693_693706


namespace simplify_fraction_90_150_l693_693445

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end simplify_fraction_90_150_l693_693445


namespace minimum_value_l693_693252

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x) + 4 / (1 - x)

theorem minimum_value : ∃ x ∈ Ioo (-1 : ℝ) (1 : ℝ), ∀ y ∈ Ioo (-1 : ℝ) (1 : ℝ), f x ≤ f y ∧ f x = 9 / 2 :=
by
  sorry

end minimum_value_l693_693252


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693930

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693930


namespace parallelogram_perimeter_l693_693703

theorem parallelogram_perimeter (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) : 2 * (a + b) = 16 :=
by {
  rw [h₁, h₂],
  norm_num,
  sorry
}

end parallelogram_perimeter_l693_693703


namespace complex_expression_value_l693_693403

noncomputable def z := Complex.exp (Complex.I * (4 * Real.pi / 7))

theorem complex_expression_value :
  (|((z / (1 + z^2)) + (z^2 / (1 + z^4)) + (z^3 / (1 + z^6)))|) = 2 :=
by
  have z_is_root_of_unity : z^7 = 1 := by
    sorry

  have z_root_property : z^6 + z^5 + z^4 + z^3 + z^2 + z + 1 = 0 := by
    sorry

  -- Final proof combining the conditions would go here
  sorry

end complex_expression_value_l693_693403


namespace no_ab_term_implies_m_eq_neg6_l693_693350

theorem no_ab_term_implies_m_eq_neg6 (a b : ℝ) (m : ℝ) : 
  (∀ a b, (3 * (a^2 - 2 * a * b - b^2) - (a^2 + m * a * b + 2 * b^2)) ≠ (C * a * b)) → m = -6 :=
begin
  -- proof omitted
  sorry
end

end no_ab_term_implies_m_eq_neg6_l693_693350


namespace diane_stamp_combinations_l693_693645

/-- Define the types of stamps Diane has --/
def diane_stamps : List ℕ := [1, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8]

/-- Define the condition for the correct number of different arrangements to sum exactly to 12 cents -/
noncomputable def count_arrangements (stamps : List ℕ) (sum : ℕ) : ℕ :=
  -- Implementation of the counting function goes here
  sorry

/-- Prove that the number of distinct arrangements to make exactly 12 cents is 13 --/
theorem diane_stamp_combinations : count_arrangements diane_stamps 12 = 13 :=
  sorry

end diane_stamp_combinations_l693_693645


namespace length_AX_l693_693771

-- Defining the given conditions as variables using Lean notation
variables (A B C X : Type)
variables (AC BC BX AX : ℝ)
variable (CX_bisects_∠ACB : true)

-- The given lengths
def AC_length : AC = 24 := sorry
def BC_length : BC = 36 := sorry
def BX_length : BX = 42 := sorry

-- State the theorem to prove AX = 28
theorem length_AX 
    (AC_length : AC = 24) 
    (BC_length : BC = 36) 
    (BX_length : BX = 42) 
    (CX_bisects_∠ACB : true) :
    AX = 28 := 
sorry

end length_AX_l693_693771


namespace y_squared_range_l693_693742

theorem y_squared_range (y : ℝ) 
  (h : Real.sqrt (Real.sqrt (y + 16)) - Real.sqrt (Real.sqrt (y - 16)) = 2) : 
  9200 ≤ y^2 ∧ y^2 ≤ 9400 := 
sorry

end y_squared_range_l693_693742


namespace simplify_fraction_l693_693452

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end simplify_fraction_l693_693452


namespace triangle_properties_l693_693293

theorem triangle_properties 
-- Conditions
  (a b c : ℝ) (A B C : ℝ) (S : ℝ)
  (ha : a^2 + c^2 = b^2 + a * c)
  (hb : b = 2)
  (hS : S = sqrt 3)
  -- Statements to prove
  (hB : B = 60)
  (h_eq_triangle : a = b ∧ b = c) :
  -- Validating the problem:
  ( a^2 + c^2 = b^2 + a * c ∧
    b = 2 ∧
    S = sqrt 3 ∧
    B = 60 ∧
    (a = b ∧ b = c)) :=
begin
  sorry
end

end triangle_properties_l693_693293


namespace total_hats_l693_693030

noncomputable def num_adults := 1500
noncomputable def proportion_men := (2 : ℚ) / 3
noncomputable def proportion_women := 1 - proportion_men
noncomputable def proportion_men_hats := (15 : ℚ) / 100
noncomputable def proportion_women_hats := (10 : ℚ) / 100

noncomputable def num_men := proportion_men * num_adults
noncomputable def num_women := proportion_women * num_adults
noncomputable def num_men_hats := proportion_men_hats * num_men
noncomputable def num_women_hats := proportion_women_hats * num_women

noncomputable def total_adults_with_hats := num_men_hats + num_women_hats

theorem total_hats : total_adults_with_hats = 200 := 
by
  sorry

end total_hats_l693_693030


namespace largest_prime_factor_of_fact_sum_is_7_l693_693965

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693965


namespace integral_of_f_l693_693713

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Set.Icc (-(Real.pi)) 0 then Real.sin x else Real.sqrt (1 - x^2)

theorem integral_of_f : ∫ x in Icc (-(Real.pi)) 1, f x = (Real.pi / 4) - 2 := by
  sorry

end integral_of_f_l693_693713


namespace sum_of_solutions_l693_693794

noncomputable def f : ℝ → ℝ := λ x, 12 * x + 5

def f_inv (x : ℝ) : ℝ := (x - 5) / 12

theorem sum_of_solutions : 
  let inv_comp (x : ℝ) := f_inv (f ((3 * x)⁻¹)) in 
  (∑ x in { x : ℝ | f_inv x = inv_comp x }.to_finset, id x) = 65 :=
by sorry

end sum_of_solutions_l693_693794


namespace smallest_possible_N_l693_693021

theorem smallest_possible_N (p q r s t : ℕ) (hp : p > 0) (hq : q > 0) 
  (hr : r > 0) (hs : s > 0) (ht : t > 0) (h_sum : p + q + r + s + t = 4020) :
  ∃ N, N = max (max (p + q) (q + r)) (max (r + s) (s + t)) ∧ N = 1005 :=
sorry

end smallest_possible_N_l693_693021


namespace gail_original_seat_l693_693042

-- Definitions of conditions
def initial_seats : Fin 7 → Option String :=
  fun n => match n with
           | 0 => some "Hank"
           | 1 => some "Ivy"
           | 2 => some "Gail"
           | 3 => some "Kali"
           | 4 => some "Lana"
           | 5 => some "Mo"
           | 6 => some "Jack"
           end

def move_right : Nat → Nat → Nat := fun seat n => (seat + n) % 7
def move_left : Nat → Nat → Nat := fun seat n => (seat + 7 - n) % 7

noncomputable def final_seats : Fin 7 → Option String :=
  fun n => match n.val with
           | 0 => initial_seats (move_left 2 2)  -- Gail's new possible position
           | 1 => initial_seats 1 -- Ivy moved left by 2
           | 2 => initial_seats (move_right 0 3) -- Hank moved right by 3
           | 3 => initial_seats (move_right 6 1) -- Jack moved right by 1
           | 4 => initial_seats 4 -- Kali/Lana's new positions (switched)
           | 5 => initial_seats 5 -- Mo hasn't moved
           | 6 => initial_seats 3
           end

theorem gail_original_seat :
  (final_seats 0 = initial_seats 2 ∨ final_seats 6 = initial_seats 2) →
  initial_seats 2 = some "Gail" :=
by
  intro h
  cases h
  sorry
  sorry

end gail_original_seat_l693_693042


namespace ellipse_focus_k_value_l693_693843

theorem ellipse_focus_k_value (k : ℝ) 
    (eq_ellipse : ∀ x y : ℝ, 5 * x^2 - k * y^2 = 5) 
    (focus : (0, 2)) : 
  k = -1 :=
sorry

end ellipse_focus_k_value_l693_693843


namespace both_roots_abs_less_than_one_l693_693116

open Real

theorem both_roots_abs_less_than_one {a b : ℝ} 
  (h1 : |a| + |b| < 1) 
  (h2 : a^2 - 4 * b ≥ 0) : 
  ∀ (x : ℝ), is_root (polynomial.quadratic a b) x → |x| < 1 :=
sorry

end both_roots_abs_less_than_one_l693_693116


namespace base_eight_to_base_ten_l693_693880

theorem base_eight_to_base_ten : ∃ n : ℕ, 47 = 4 * 8 + 7 ∧ n = 39 :=
by
  sorry

end base_eight_to_base_ten_l693_693880


namespace candies_needed_for_full_bag_l693_693078

theorem candies_needed_for_full_bag (total_candies : ℕ) (candies_per_bag : ℕ) : total_candies = 254 → candies_per_bag = 30 → (candies_per_bag - total_candies % candies_per_bag) = 16 := 
by 
  intros h_total h_bag
  rw [h_total, h_bag]
  sorry

end candies_needed_for_full_bag_l693_693078


namespace perpendicular_lines_m_values_l693_693725

theorem perpendicular_lines_m_values (m : ℝ) :
  (l1: linear_equation := λ x y, m * x + y - 2 = 0) →
  (l2: linear_equation := λ x y, (m + 1) * x - 2 * m * y + 1 = 0) →
  (is_perpendicular l1 l2) → (m = 0 ∨ m = 1) :=
by
  -- Define what it means for two lines to be perpendicular
  let is_perpendicular := λ l1 l2, 
    ∃ m1 m2, slope l1 = m1 ∧ slope l2 = m2 ∧ m1 * m2 = -1
  -- Definitions of slopes for given lines
  let slope_l1 := -m
  let slope_l2 := (m + 1) / (-2 * m)
  -- Assumptions from the problem conditions
  assume h1 : linear_equation l1,
  assume h2 : linear_equation l2,
  assume h3 : is_perpendicular l1 l2,
  sorry

end perpendicular_lines_m_values_l693_693725


namespace mother_l693_693136

def age_relations (P M : ℕ) : Prop :=
  P = (2 * M) / 5 ∧ P + 10 = (M + 10) / 2

theorem mother's_present_age (P M : ℕ) (h : age_relations P M) : M = 50 :=
by
  sorry

end mother_l693_693136


namespace total_dollars_l693_693807

theorem total_dollars (mark_dollars : ℚ) (carolyn_dollars : ℚ) (mark_money : mark_dollars = 7 / 8) (carolyn_money : carolyn_dollars = 2 / 5) :
  mark_dollars + carolyn_dollars = 1.275 := sorry

end total_dollars_l693_693807


namespace circles_tangent_to_three_general_lines_l693_693369

theorem circles_tangent_to_three_general_lines (L1 L2 L3 : set (ℝ × ℝ)) 
  (hL1 : ∀ x y : ℝ, (x, y) ∈ L1 → L1 = {p | ∃ m b, p = (x, y) ∧ y = m * x + b}) 
  (hL2 : ∀ x y : ℝ, (x, y) ∈ L2 → L2 = {p | ∃ m b, p = (x, y) ∧ y = m * x + b}) 
  (hL3 : ∀ x y : ℝ, (x, y) ∈ L3 → L3 = {p | ∃ m b, p = (x, y) ∧ y = m * x + b}) 
  (h_general_position : ¬ ((∃ p : ℝ × ℝ, p ∈ L1 ∧ p ∈ L2 ∧ p ∈ L3) ∨ 
                          (∀ p1 p2 : ℝ × ℝ, p1 ∈ L1 → p2 ∈ L1 → p1 = p2) ∨ 
                          (∀ p1 p2 : ℝ × ℝ, p1 ∈ L2 → p2 ∈ L2 → p1 = p2) ∨ 
                          (∀ p1 p2 : ℝ × ℝ, p1 ∈ L3 → p2 ∈ L3 → p1 = p2))) : 
  ∃ (C1 C2 C3 C4 : set (ℝ × ℝ)), 
  (∀ i : fin 4, ∃ r : ℝ, ∃ (x y : ℝ), Ci = {p | dist p (x, y) = r}) ∧ 
  ∀ i : fin 4, ∀ p : ℝ × ℝ, p ∈ Ci → 
  (dist p (x1, y1) = r1 ∧ dist p (x2, y2) = r2 ∧ dist p (x3, y3) = r3) := 
begin
  sorry
end

end circles_tangent_to_three_general_lines_l693_693369


namespace unique_sums_count_l693_693602

open Set

-- Defining the sets of chips in bags C and D
def BagC : Set ℕ := {1, 3, 7, 9}
def BagD : Set ℕ := {4, 6, 8}

-- The proof problem: show there are 7 unique sums
theorem unique_sums_count : (BagC ×ˢ BagD).image (λ p => p.1 + p.2) = {5, 7, 9, 11, 13, 15, 17} :=
by
  -- Proof omitted; complete proof would go here
  sorry

end unique_sums_count_l693_693602


namespace distance_between_towns_l693_693512

theorem distance_between_towns 
  (rate1 rate2 : ℕ) (time : ℕ) (distance : ℕ)
  (h_rate1 : rate1 = 48)
  (h_rate2 : rate2 = 42)
  (h_time : time = 5)
  (h_distance : distance = rate1 * time + rate2 * time) : 
  distance = 450 :=
by
  sorry

end distance_between_towns_l693_693512


namespace time_saved_l693_693563

theorem time_saved (speed_with_tide distance1 time1 distance2 time2: ℝ) 
  (h1: speed_with_tide = 5) 
  (h2: distance1 = 5) 
  (h3: time1 = 1) 
  (h4: distance2 = 40) 
  (h5: time2 = 10) : 
  time2 - (distance2 / speed_with_tide) = 2 := 
sorry

end time_saved_l693_693563


namespace stability_measure_variance_l693_693133

-- Given conditions
def scores : List ℕ := [86, 78, 80, 85, 92]

-- Mathematical equivalent proof problem statement
theorem stability_measure_variance : 
  (variance scores) = (choose_the_correct_measure scores) := 
sorry

end stability_measure_variance_l693_693133


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693932

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693932


namespace largest_prime_factor_of_7fact_8fact_l693_693908

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693908


namespace worms_split_count_eq_coprime_count_l693_693209

-- Define the conditions
def is_valid_segment (x1 y1 x2 y2 : ℕ) : Prop :=
  (x2 = x1 + 1 ∧ y2 = y1) ∨ (x2 = x1 ∧ y2 = y1 + 1)

def is_valid_path (path : list (ℕ × ℕ)) : Prop :=
  path.head = (0,0) ∧ ∀ (i : ℕ), i < path.length - 1 → is_valid_segment path.nth(i) path.nth(i+1)

-- Define the function to count the number of ways to split the worm
def number_of_domino_ways (path : list (ℕ × ℕ)) [is_valid_path path] : ℕ :=
  sorry

-- Define the function to count the number of natural numbers less than n and coprime with n
def count_coprime (n : ℕ) : ℕ :=
  (list.range n).filter (nat.coprime n).length

-- Main theorem statement
theorem worms_split_count_eq_coprime_count (n : ℕ) (hn : n > 2) :
  (number_of_domino_ways path = n) ↔ (count_coprime n) :=
  sorry

end worms_split_count_eq_coprime_count_l693_693209


namespace dice_probability_same_color_l693_693736

theorem dice_probability_same_color :
  let P_maroon := (5 / 20) * (4 / 20)
  let P_teal := (6 / 20) * (7 / 20)
  let P_cyan := (7 / 20) * (8 / 20)
  let P_sparkly := (2 / 20) * (1 / 20)
  P_maroon + P_teal + P_cyan + P_sparkly = 3 / 10 :=
by
  -- define individual probabilities
  let P_maroon := (5 / 20) * (4 / 20)
  let P_teal := (6 / 20) * (7 / 20)
  let P_cyan := (7 / 20) * (8 / 20)
  let P_sparkly := (2 / 20) * (1 / 20)
  -- calculate the total probability
  have H1 : P_maroon = 1 / 20 := by { ring }
  have H2 : P_teal = 21 / 200 := by { ring }
  have H3 : P_cyan = 7 / 50 := by { ring }
  have H4 : P_sparkly = 1 / 200 := by { ring }
  have total := H1 + H2 + H3 + H4
  -- combine and simplify the probabilities
  show 1 / 20 + 21 / 200 + 7 / 50 + 1 / 200 = 3 / 10
  sorry

end dice_probability_same_color_l693_693736


namespace ceil_neg_3_7_l693_693225

-- Define the ceiling function in Lean
def ceil (x : ℝ) : ℤ := int.ceil x

-- A predicate to represent the statement we want to prove
theorem ceil_neg_3_7 : ceil (-3.7) = -3 := by
  -- Provided conditions
  have h1 : ceil (-3.7) = int.ceil (-3.7) := rfl
  have h2 : int.ceil (-3.7) = -3 := by
    -- Lean's int.ceil function returns the smallest integer greater or equal to the input
    sorry  -- proof goes here

  -- The main statement
  exact h2

end ceil_neg_3_7_l693_693225


namespace constant_term_in_expansion_eq_neg40_l693_693479

-- Definition to represent the binomial expansion term
def binomial_expansion_term (n k : ℕ) (x : ℝ) : ℝ :=
  (2:ℝ)^(n-k) * (Real.choose n k) * x^(n-2*k)

-- Function to get the coefficients of terms that result in a constant when multiplied
def constant_term_coeff (x : ℝ) : ℝ :=
  let coeff_x_minus1 := binomial_expansion_term 5 3 x -- coefficient for x^{-1} term
  let coeff_x := binomial_expansion_term 5 2 x -- coefficient for x term
  coeff_x_minus1 - coeff_x

-- Main theorem statement to prove the constant term
theorem constant_term_in_expansion_eq_neg40 (x : ℝ) : constant_term_coeff x = -40 := by
  sorry

end constant_term_in_expansion_eq_neg40_l693_693479


namespace find_m_tan_periodicity_l693_693661

theorem find_m_tan_periodicity : ∃ m : ℤ, -180 < m ∧ m < 180 ∧ tan (m * Real.pi / 180) = tan (1230 * Real.pi / 180) ∧ m = -30 :=
begin
  use -30,
  split,
  { linarith, },  -- Prove -180 < -30 which is true
  split,
  { linarith, },  -- Prove -30 < 180 which is true
  split,
  { -- Prove the equality of tangents
    rw [←Real.tan_pi_div_two],
    repeat {rw Real.tan_periodic (Real.pi / 180)},
    simp,
    norm_num,
  },
  { refl },  -- m is expected to be -30
end

end find_m_tan_periodicity_l693_693661


namespace positive_integers_of_inequality_l693_693067

theorem positive_integers_of_inequality (x : ℕ) (h : 9 - 3 * x > 0) : x = 1 ∨ x = 2 :=
sorry

end positive_integers_of_inequality_l693_693067


namespace right_triangle_perimeter_l693_693569

theorem right_triangle_perimeter (area leg1 : ℕ) (h_area : area = 180) (h_leg1 : leg1 = 30) :
  ∃ leg2 hypotenuse perimeter, 
    (2 * area = leg1 * leg2) ∧ 
    (hypotenuse^2 = leg1^2 + leg2^2) ∧ 
    (perimeter = leg1 + leg2 + hypotenuse) ∧ 
    (perimeter = 42 + 2 * Real.sqrt 261) :=
by
  sorry

end right_triangle_perimeter_l693_693569


namespace distance_between_boy_and_friend_l693_693547

-- Definitions for the given conditions
def speed_of_car_kmhr : ℝ := 108
def time_taken_s : ℝ := 3.9669421487603307
def speed_of_sound_ms : ℝ := 330
def kmhr_to_ms_conversion : ℝ := 1/3.6

-- Conversion of the car's speed from km/hr to m/s
def speed_of_car_ms : ℝ := speed_of_car_kmhr * kmhr_to_ms_conversion

-- Calculation of the relative speed of sound
def relative_speed_of_sound_ms : ℝ := speed_of_sound_ms - speed_of_car_ms

-- Calculation of the distance
def distance_m (relative_speed : ℝ) (time : ℝ) : ℝ := relative_speed * time

-- Theorem to prove the distance between the boy and his friend
theorem distance_between_boy_and_friend :
  distance_m relative_speed_of_sound_ms time_taken_s = 1190.082644628099 :=
by
  sorry

end distance_between_boy_and_friend_l693_693547


namespace find_first_number_l693_693833

/-- The Least Common Multiple (LCM) of two numbers A and B is 2310,
    and their Highest Common Factor (HCF) is 30.
    Given one of the numbers B is 180, find the other number A. -/
theorem find_first_number (A B : ℕ) (LCM HCF : ℕ) (h1 : LCM = 2310) (h2 : HCF = 30) (h3 : B = 180) (h4 : A * B = LCM * HCF) :
  A = 385 :=
by sorry

end find_first_number_l693_693833


namespace largest_prime_factor_of_fact_sum_is_7_l693_693962

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693962


namespace age_of_vanya_and_kolya_l693_693080

theorem age_of_vanya_and_kolya (P V K : ℕ) (hP : P = 10)
  (hV : V = P - 1) (hK : K = P - 5 + 1) : V = 9 ∧ K = 6 :=
by
  sorry

end age_of_vanya_and_kolya_l693_693080


namespace assistant_stop_time_l693_693170

-- Define the start time for the craftsman
def craftsmanStartTime : Nat := 8 * 60 -- in minutes

-- Craftsman starts at 8:00 AM and stops at 12:00 PM
def craftsmanEndTime : Nat := 12 * 60 -- in minutes

-- Craftsman produces 6 bracelets every 20 minutes
def craftsmanProductionPerMinute : Nat := 6 / 20

-- Assistant starts working at 9:00 AM
def assistantStartTime : Nat := 9 * 60 -- in minutes

-- Assistant produces 8 bracelets every 30 minutes
def assistantProductionPerMinute : Nat := 8 / 30

-- Total production duration for craftsman in minutes
def craftsmanWorkDuration : Nat := craftsmanEndTime - craftsmanStartTime

-- Total bracelets produced by craftsman
def totalBraceletsCraftsman : Nat := craftsmanWorkDuration * craftsmanProductionPerMinute

-- Time it takes for the assistant to produce the same number of bracelets
def assistantWorkDuration : Nat := totalBraceletsCraftsman / assistantProductionPerMinute

-- Time the assistant will stop working
def assistantEndTime : Nat := assistantStartTime + assistantWorkDuration

-- Convert time in minutes to hours and minutes format (output as a string for clarity)
def formatTime (timeInMinutes: Nat) : String :=
  let hours := timeInMinutes / 60
  let minutes := timeInMinutes % 60
  s! "{hours}:{if minutes < 10 then "0" else ""}{minutes}"

-- Proof goal: assistant will stop working at "13:30" (or 1:30 PM)
theorem assistant_stop_time : 
  formatTime assistantEndTime = "13:30" := 
by
  sorry

end assistant_stop_time_l693_693170


namespace problem_statement_l693_693821

-- Define the sides of the original triangle
def side_5 := 5
def side_12 := 12
def side_13 := 13

-- Define the perimeters of the isosceles triangles
def P := 3 * side_5
def Q := 3 * side_12
def R := 3 * side_13

-- Statement we want to prove
theorem problem_statement : P + R = (3 / 2) * Q := by
  sorry

end problem_statement_l693_693821


namespace thickness_wall_is_034_l693_693589

noncomputable def thickness_of_hollow_iron_sphere_wall
  (q : ℝ)
  (f : ℝ)
  (s : ℝ) : ℝ :=
let pi := Real.pi in
let V := q / ((4 / 3) * pi) in
let R := (V * (3 / 4)) in
let R_cubed := R in
let R_cubed_val := R_cubed + (q * 30) / s in
let R_inner_cubed := R_cubed - 100.4 in
let R_ := R_cubed ** (1 / 3) in
let r := R_inner_cubed ** (1 / 3) in
R_ - r

theorem thickness_wall_is_034
  (q : ℝ) (hq : q = 3012)
  (f : ℝ) (hf : f = 3/4)
  (s : ℝ) (hs : s = 7.5) :
  thickness_of_hollow_iron_sphere_wall q f s = 0.34 :=
by
  sorry

end thickness_wall_is_034_l693_693589


namespace calculate_120ab_l693_693140

variable (a b : ℚ)

theorem calculate_120ab (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * (a * b) = 800 := by
  sorry

end calculate_120ab_l693_693140


namespace at_least_one_greater_than_one_l693_693791

theorem at_least_one_greater_than_one (a b : ℝ) (h2 : a + b > 2) (h5 : Real.log a b < 0) : a > 1 ∨ b > 1 :=
by
  sorry

end at_least_one_greater_than_one_l693_693791


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693923

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693923


namespace area_OAB_is_sqrt3_div_2_l693_693370

-- Definitions for polar coordinates and conversion to rectangular coordinates
noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

-- Points A and B in polar coordinates
def A_polar : ℝ × ℝ := (1, Real.pi / 6)
def B_polar : ℝ × ℝ := (2, Real.pi / 2)

-- Points A and B in rectangular coordinates
def A_rect : ℝ × ℝ := polar_to_rect A_polar.1 A_polar.2
def B_rect : ℝ × ℝ := polar_to_rect B_polar.1 B_polar.2

-- Area calculation using the formula 1/2 * base * height
def area_triangle (O A B : ℝ × ℝ) : ℝ :=
  1/2 * (B.1 - O.1) * A.2

theorem area_OAB_is_sqrt3_div_2 :
  area_triangle (0, 0) A_rect B_rect = Real.sqrt 3 / 2 :=
by
  sorry

end area_OAB_is_sqrt3_div_2_l693_693370


namespace ceil_neg_3_7_l693_693228

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := by
  sorry

end ceil_neg_3_7_l693_693228


namespace pentagon_triangle_sum_l693_693650

theorem pentagon_triangle_sum :
  let S := List.sum [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let S1 := 25
  let S2 := 16 in
  S = 55 ∧ 2 * S1 + 5 = S ∧ S2 * 5 = 2 * S1 + (S1 + 5) :=
begin
  sorry
end

end pentagon_triangle_sum_l693_693650


namespace value_of_K_l693_693127

theorem value_of_K :
  (32^5 * 4^3 = 2^(31)) :=
by
  -- Initial conditions: 32 = 2^5 and 4 = 2^2
  have h1: 32 = 2^5 := rfl
  have h2: 4 = 2^2 := rfl
  
  -- Express 32^5 and 4^3 using the conditions
  calc
    32^5 * 4^3 = (2^5)^5 * (2^2)^3   : by rw [←h1, ←h2]
           ... = 2^(5 * 5) * 2^(2 * 3) : by simp [pow_mul]
           ... = 2^25 * 2^6             : by norm_num
           ... = 2^(25 + 6)             : by rw [pow_add]
           ... = 2^31                   : by norm_num

  -- This implies the final result
  exact rfl

end value_of_K_l693_693127


namespace circumscribed_circle_exists_l693_693072

-- Definitions
variables {Point : Type} [metric_space Point]
variables (O A B C D P Q : Point)
variable (circle : set Point)
variable tangent_points : Point × Point
variable on_tangent : Point → Point → Point → Prop -- C and D lie on BP and BQ
variable (equidistant_from_center : Point → Point → Point → Prop) -- O is equidistant from C and D
variable (perpendicular_to_tangent : Point → Point → Point → Prop) -- Radii OP and OQ

-- Condition definitions
def is_center (O A : Point) := A = O

def lies_outside (B circle) := ¬(B ∈ circle)

def equidistant_from_O (O C D : Point) := dist O C = dist O D

-- Lean 4 statement for the proof problem
theorem circumscribed_circle_exists (h1 : is_center O A) 
  (h2 : lies_outside B circle) 
  (h3 : on_tangent C B P)
  (h4 : on_tangent D B Q)
  (h5 : C ∈ circle)
  (h6 : D ∈ circle)
  (h7 : tangent_points = (P, Q))
  (h8 : equidistant_from_O O C D)
  (h9 : perpendicular_to_tangent O P B ∧ perpendicular_to_tangent O Q B) :
  ∃ circle', circle' ∋ A ∧ circle' ∋ B ∧ circle' ∋ C ∧ circle' ∋ D :=
sorry

end circumscribed_circle_exists_l693_693072


namespace simplify_fraction_90_150_l693_693447

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end simplify_fraction_90_150_l693_693447


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693914

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693914


namespace divisible_by_1989_l693_693815

theorem divisible_by_1989 (n : ℕ) : 
  1989 ∣ (13 * (-50)^n + 17 * 40^n - 30) :=
by
  sorry

end divisible_by_1989_l693_693815


namespace single_discount_equivalence_l693_693173

variable (p : ℝ) (d1 d2 d3 : ℝ)

def apply_discount (price discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_multiple_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem single_discount_equivalence :
  p = 1200 →
  d1 = 0.15 →
  d2 = 0.10 →
  d3 = 0.05 →
  let final_price_multiple := apply_multiple_discounts p [d1, d2, d3]
  let single_discount := (p - final_price_multiple) / p
  single_discount = 0.27325 :=
by
  intros h1 h2 h3 h4
  let final_price_multiple := apply_multiple_discounts p [d1, d2, d3]
  let single_discount := (p - final_price_multiple) / p
  sorry

end single_discount_equivalence_l693_693173


namespace balance_equation_l693_693504

section
variables (workers_bolts workers_nuts total_workers bolts_per_worker nuts_per_worker per_bolt_requires_nuts : ℕ)
variables (x : ℕ)

-- Given conditions
def original_workers := 16
def bolts_per_worker := 1200
def nuts_per_worker := 2000
def nuts_per_bolt := 2

theorem balance_equation :
  2 * 1200 * x = 2000 * (original_workers - x) := sorry

end

end balance_equation_l693_693504


namespace find_cosine_F1_P_F2_l693_693288

variable (F1 F2 P : Point) -- Assume Point is a defined structure for points in 2D space

noncomputable def hyperbola_eq : Prop := 
  ∃ (x y : ℝ), x^2 - y^2 = 2 ∧ (P : Point) = (x, y)

noncomputable def condition1 : Prop := 
  ∃ (F1 F2 : Point), is_foci F1 F2 (hyperbola_eq F1 F2)

noncomputable def condition2 : Prop :=
  ∃ (P : Point), is_on_hyperbola P F1 F2

noncomputable def condition3 : Prop := 
  2 * (dist P F2) = dist P F1

noncomputable def cosine_angle := 
  cos_angle F1 P F2

theorem find_cosine_F1_P_F2 (F1 F2 P : Point) 
  (h1 : condition1 F1 F2) 
  (h2 : condition2 P) 
  (h3 : condition3 P F1 F2) : 
  cosine_angle F1 P F2 = 3 / 4 := 
sorry

end find_cosine_F1_P_F2_l693_693288


namespace max_elements_subset_S_l693_693004

open Set

noncomputable def max_size (S : Set ℕ) : ℕ := 
  if h : S ⊆ (range 100).succ then S.card else 0

theorem max_elements_subset_S :
  ∀ S : Set ℕ, (S ⊆ (range 100).succ) → (∀ a b ∈ S, a ≠ b → ¬ (5 ∣ a * b)) → max_size(S) = 80 :=
begin
  intros S h_sub h_div,
  sorry
end

end max_elements_subset_S_l693_693004


namespace find_negative_a_l693_693798

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 0 then -x else 3 * x - 22

theorem find_negative_a (a : ℝ) (ha : a < 0) :
  g (g (g 7)) = g (g (g a)) ↔ a = -23 / 3 :=
by
  sorry

end find_negative_a_l693_693798


namespace incorrect_statement_l693_693549

-- Defining the conditions
def price_tag : ℕ := 80
def discount : ℝ := 0.2
def profit_per_product : ℕ := 24
def base_units_sold_per_week : ℕ := 220
def additional_units_per_dollar_decrease : ℕ := 20

-- Denote the cost price of each product and the price reduction per unit
def cost_price : ℕ := price_tag - price_tag * 20 / 100 - profit_per_product
def x : ℕ -- price reduction, assumed as integer
def y : ℕ -- profit per week

-- Computing the selling price after the reduction, units sold, and profit
def selling_price_after_reduction : ℕ := price_tag * 80 / 100 - x
def units_sold_per_week : ℕ := base_units_sold_per_week + additional_units_per_dollar_decrease * x
def profit_per_week (x) : ℕ := (24 - x) * (base_units_sold_per_week + additional_units_per_dollar_decrease * x)

-- Statement to be proven
theorem incorrect_statement : (84 - x) * (base_units_sold_per_week + additional_units_per_dollar_decrease * x) ≠ profit_per_week x :=
sorry

end incorrect_statement_l693_693549


namespace find_original_salary_l693_693174

variable (S : ℝ)
variable (savings_percentage rent_percentage utilities_percentage groceries_percentage new_savings : ℝ)
variable (rent_increase_percentage utilities_increase_percentage groceries_increase_percentage : ℝ)

def original_savings := savings_percentage * S
def rent := rent_percentage * S
def utilities := utilities_percentage * S
def groceries := groceries_percentage * S

def increased_rent := rent + rent_increase_percentage * rent
def increased_utilities := utilities + utilities_increase_percentage * utilities
def increased_groceries := groceries + groceries_increase_percentage * groceries

def new_expenses := increased_rent + increased_utilities + increased_groceries
def original_expenses := rent + utilities + groceries

theorem find_original_salary : 
  (savings_percentage * S - new_savings = new_expenses - original_expenses) → 
  S = 3000 :=
by
  sorry

-- Assign values specifically adherent to the conditions
#eval let S := 3000
        let savings_percentage := 0.20
        let rent_percentage := 0.40
        let utilities_percentage := 0.30
        let groceries_percentage := 0.20
        let new_savings := 180
        let rent_increase_percentage := 0.15
        let utilities_increase_percentage := 0.20
        let groceries_increase_percentage := 0.10
        original_savings S savings_percentage

end find_original_salary_l693_693174


namespace sum_of_real_solutions_l693_693216

theorem sum_of_real_solutions : 
  (∀ x ∈ ℝ, |x^2 - 14*x + 51| = 3 → x = 8 ∨ x = 6) →
  (∑ x in {8, 6}, x) = 14 :=
by
  intro h
  -- The set of real solutions are {6, 8}. We just sum them up.
  have h_sum : ∑ x in {8, 6}, x = 14 := by simp
  exact h_sum

end sum_of_real_solutions_l693_693216


namespace harry_book_page_count_l693_693041

-- Definition of the entire setup of the problem
def selena_pages (x : ℝ) := x
def half_selena_pages (x : ℝ) := x / 2
def harry_pages (x y : ℝ) := half_selena_pages x - y

-- Theorem statement to prove Harry's page count
theorem harry_book_page_count (x y : ℝ) : harry_pages x y = x / 2 - y :=
by
  sorry

end harry_book_page_count_l693_693041


namespace triangle_BE_length_l693_693083

theorem triangle_BE_length :
  ∀ (A B C D E F : Type)
    [linear_order_metric_space A B C D E F]
    (AB BC CA : ℝ)
    (AB_lt_AD : ∀ AD AE, AB < AD ∧ AD < AE)
    (circumABD : Triangle_circumcircle ABC D)
    (circumEBC : Triangle_circumcircle E BC)
    (DF : ℝ := 3)
    (EF : ℝ := 4)
    (length_BE : ℝ := (17 / 5)),

    AB = 3 ∧ BC = 4 ∧ CA = 5 ∧ AD ∈ ray AB ∧ AE ∈ ray AB ∧ F ≠ C 
    → BE = length_BE := sorry

end triangle_BE_length_l693_693083


namespace certain_number_is_32_l693_693745

theorem certain_number_is_32 (k t : ℚ) (certain_number : ℚ) 
  (h1 : t = 5/9 * (k - certain_number))
  (h2 : t = 75) (h3 : k = 167) :
  certain_number = 32 :=
sorry

end certain_number_is_32_l693_693745


namespace price_reduction_equation_l693_693166

theorem price_reduction_equation:
  ∀ (P0 Pf : ℕ) (x : ℝ), P0 = 100 ∧ Pf = 80 → Pf = P0 * (1 - x)^2 :=
by  
  intros P0 Pf x h,
  cases h with hP0 hPf,
  rw [hP0, hPf],
  norm_num,
  sorry

end price_reduction_equation_l693_693166


namespace exists_two_with_divisible_square_diff_l693_693036

theorem exists_two_with_divisible_square_diff (S : Finset ℕ) (hS : S.card = 43) :
  ∃ a b ∈ S, a ≠ b ∧ (a^2 - b^2) % 100 = 0 :=
by {
  sorry
}

end exists_two_with_divisible_square_diff_l693_693036


namespace simplify_polynomial_l693_693459

theorem simplify_polynomial : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 3 * x^2 + 6 * x - 8) = x^3 + x^2 + 3 * x + 3 :=
by
  sorry

end simplify_polynomial_l693_693459


namespace moles_reaction_l693_693729

def reactants := {NH4NO3 : ℕ, NaOH : ℕ}
def products := {NaNO3 : ℕ, NH3 : ℕ, H2O : ℕ}

axiom balanced_equation :
    ∀ (NH4NO3 NaOH NaNO3 NH3 H2O : ℕ),
    NH4NO3 + NaOH = NaNO3 + NH3 + H2O →

    NH4NO3 = 1 ∧ NaOH = 1 ∧ NaNO3 = 1 ∧ NH3 = 1 ∧ H2O = 1

theorem moles_reaction :
    ∀ (NH4NO3 NaOH NaNO3 NH3 H2O : ℕ),
    balanced_equation NH4NO3 NaOH NaNO3 NH3 H2O →
    NaOH = 3 →
    NH4NO3 = 3 ∧ NaNO3 = 3 ∧ NH3 = 3 ∧ H2O = 3 :=
by intros; sorry

end moles_reaction_l693_693729


namespace point_in_second_quadrant_l693_693390

theorem point_in_second_quadrant : 
  let z := (1 + 2 * Complex.I) in
  let w := z * z in
  let a := w.re in 
  let b := w.im in
  w = Complex.mk a b → 
  a = -3 ∧ b = 4 ∧ a < 0 ∧ b > 0 :=
by
  let z := (1 + 2 * Complex.I)
  let w := z * z
  let a := w.re
  let b := w.im
  sorry

end point_in_second_quadrant_l693_693390


namespace largest_prime_factor_of_fact_sum_is_7_l693_693964

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693964


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693998

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693998


namespace num_diamonds_F25_l693_693481

def num_diamonds (n : ℕ) : ℕ := 
  match n with
  | 1 => 1
  | m + 2 => 2 * (m + 2) * (m + 1) + 2 * (m + 2) - 3

theorem num_diamonds_F25 : num_diamonds 25 = 1297 := by
  sorry

end num_diamonds_F25_l693_693481


namespace isosceles_trapezoid_perimeter_l693_693347

-- Define the concept of an isosceles trapezoid with given side lengths
structure IsoscelesTrapezoid (a b c : ℝ) :=
  (is_isosceles : a = b ∨ b = c ∨ a = c)
  (side_lengths : multiset (ℝ) := {a, b, c, c})

def perimeter (t : IsoscelesTrapezoid a b c) : ℝ :=
  t.side_lengths.sum

theorem isosceles_trapezoid_perimeter (a b c : ℝ) (h : IsoscelesTrapezoid a b c) :
   (perimeter h = 25 ∨ perimeter h = 30 ∨ perimeter h = 33) :=
  by sorry

end isosceles_trapezoid_perimeter_l693_693347


namespace distance_from_center_to_chord_l693_693357

-- Define the problem statement and conditions
theorem distance_from_center_to_chord
  (O : Point) (r : ℝ) (A : Point) (chord1_length1 chord1_length2 chord2_length1 chord2_length2 : ℝ)
  (h_perpendicular : chord1 ⟂ chord2)
  (h_chord1 : chord1_length1 = 3 ∧ chord1_length2 = 7)
  (h_chord2 : chord2_length1 = 3 ∧ chord2_length2 = 7) :
  exists (d : ℝ), d = 2 :=
by
  sorry

end distance_from_center_to_chord_l693_693357


namespace largest_prime_factor_l693_693976

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693976


namespace find_interest_rate_l693_693772

theorem find_interest_rate (P : ℝ) (n : ℕ) (l : ℝ) (r : ℝ) : 
    P = 7500 → n = 2 → l = 12.00000000000091 → 
    P * (1 + r)^2 - P * (1 + r * n) = l → 
    r ≈ 0.0012649110640673517 :=
by 
  intros hP hn hl heq 
  unfold approx_eq
  let r_approx := Real.sqrt (l / P)
  have hr : r = r_approx, from sorry
  rw hr
  have hr_app : |r_approx - 0.0012649110640673517| < ε, from sorry
  exact hr_app
  done

end find_interest_rate_l693_693772


namespace impossible_arrangement_of_300_numbers_in_circle_l693_693151

theorem impossible_arrangement_of_300_numbers_in_circle :
  ¬ ∃ (nums : Fin 300 → ℕ), (∀ i : Fin 300, nums i > 0) ∧
    ∃ unique_exception : Fin 300,
      ∀ i : Fin 300, i ≠ unique_exception → nums i = Int.natAbs (nums (Fin.mod (i.val - 1) 300) - nums (Fin.mod (i.val + 1) 300)) := 
sorry

end impossible_arrangement_of_300_numbers_in_circle_l693_693151


namespace radius_of_circle_touching_PQ_QR_PL_l693_693363

theorem radius_of_circle_touching_PQ_QR_PL (PQRS : Parallelogram) 
  (P Q R S L: ℝ)
  (h_bisector: Bisector_at_vertex P intersects RS at L)
  (angle_P : angle P = 80 * (pi / 180)) 
  (PQ : segment PQ = 7) :
  radius_of_circle_touching PQ QR PL = 7 * cos (40 * (pi / 180)) * tan (20 * (pi / 180)) :=
sorry

end radius_of_circle_touching_PQ_QR_PL_l693_693363


namespace frequencies_first_class_confidence_difference_quality_l693_693108

theorem frequencies_first_class (a b c d n : ℕ) (Ha : a = 150) (Hb : b = 50) 
                                (Hc : c = 120) (Hd : d = 80) (Hn : n = 400) 
                                (totalA : a + b = 200) 
                                (totalB : c + d = 200) :
  (a / (a + b) = 3 / 4) ∧ (c / (c + d) = 3 / 5) := by
sorry

theorem confidence_difference_quality (a b c d n : ℕ) (Ha : a = 150)
                                       (Hb : b = 50) (Hc : c = 120)
                                       (Hd : d = 80) (Hn : n = 400)
                                       (total : n = 400)
                                       (first_class_total : a + c = 270)
                                       (second_class_total : b + d = 130) :
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  k_squared > 6.635 := by
sorry

end frequencies_first_class_confidence_difference_quality_l693_693108


namespace symmetry_center_of_tan_l693_693071

-- Assuming the problem context
def f (x : ℝ) : ℝ := Real.tan (π * x + π / 4)

theorem symmetry_center_of_tan (k : ℤ) :
  ∃ x : ℝ, (π * x + π / 4 = k * π / 2) ∧ (x = (2 * k - 1) / 4) ∧ (f x = 0) :=
begin
  sorry
end

end symmetry_center_of_tan_l693_693071


namespace Dawn_has_10_CDs_l693_693778

-- Lean definition of the problem conditions
def Kristine_more_CDs (D K : ℕ) : Prop :=
  K = D + 7

def Total_CDs (D K : ℕ) : Prop :=
  D + K = 27

-- Lean statement of the proof
theorem Dawn_has_10_CDs (D K : ℕ) (h1 : Kristine_more_CDs D K) (h2 : Total_CDs D K) : D = 10 :=
by
  sorry

end Dawn_has_10_CDs_l693_693778


namespace volume_of_regular_quadrilateral_pyramid_l693_693057

-- The dihedral angle between adjacent lateral faces (alpha)
variable (α : ℝ)

-- The side length of the base (b)
variable (b : ℝ)

-- The angle between the lateral edge and the base plane (phi)
variable (φ : ℝ)

theorem volume_of_regular_quadrilateral_pyramid (h_alpha_phi_relation : α = 2 * φ):
  let V := (b^3 * Real.sqrt 2 * Real.tan φ) / 6 in
  V = V :=
by {
  sorry
}

end volume_of_regular_quadrilateral_pyramid_l693_693057


namespace trajectory_is_plane_l693_693708

/--
Given that the vertical coordinate of a moving point P is always 2, 
prove that the trajectory of the moving point P forms a plane in a 
three-dimensional Cartesian coordinate system.
-/
theorem trajectory_is_plane (P : ℝ × ℝ × ℝ) (hP : ∀ t : ℝ, ∃ x y, P = (x, y, 2)) :
  ∃ a b c d, a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧ (∀ x y, ∃ z, (a * x + b * y + c * z + d = 0) ∧ z = 2) :=
by
  -- This proof should show that there exist constants a, b, c, and d such that 
  -- the given equation represents a plane and the z-coordinate is always 2.
  sorry

end trajectory_is_plane_l693_693708


namespace at_least_n_minus_one_linear_indep_derivatives_l693_693217

variables {n : ℕ} {f : Fin n → ℝ → ℝ}
variables (hf : ∀ i : Fin n, Differentiable ℝ (f i))
variables (h_lin_indep : LinearIndependence ℝ (fun i : Fin n => f i))

theorem at_least_n_minus_one_linear_indep_derivatives :
  ∃ g : Fin (n - 1) → ℝ → ℝ, LinearIndependence ℝ (fun i => Deriv (f i)) :=
sorry

end at_least_n_minus_one_linear_indep_derivatives_l693_693217


namespace find_a_from_conditions_l693_693241

theorem find_a_from_conditions
  (a b : ℤ)
  (h₁ : 2584 * a + 1597 * b = 0)
  (h₂ : 1597 * a + 987 * b = -1) :
  a = 1597 :=
by sorry

end find_a_from_conditions_l693_693241


namespace solve_system_l693_693467

noncomputable def system_solution (C1 C2 : ℝ) : (ℝ → ℝ) × (ℝ → ℝ) :=
  λ t, (C1 * Real.exp (2 * t) + C2 * Real.exp (4 * t) - 4 * Real.exp (3 * t) - Real.exp (-t),
       C1 * Real.exp (2 * t) + (C2 / 3) * Real.exp (4 * t) - 2 * Real.exp (3 * t) - 2 * Real.exp (-t))

theorem solve_system (C1 C2 : ℝ) :
  ∃ (x y : ℝ → ℝ), 
    (∀ t : ℝ, (deriv x t = 5 * x t - 3 * y t + 2 * Real.exp (3 * t)) 
           ∧ (deriv y t = x t + y t + 5 * Real.exp (-t))) 
    ∧ x = (system_solution C1 C2).1 ∧ y = (system_solution C1 C2).2 :=
  sorry

end solve_system_l693_693467


namespace original_players_count_l693_693861

theorem original_players_count (n : ℕ) (W : ℕ) :
  (W = n * 103) →
  ((W + 110 + 60) = (n + 2) * 99) →
  n = 7 :=
by sorry

end original_players_count_l693_693861


namespace number_one_fourth_more_than_it_is_30_percent_less_than_80_l693_693081

theorem number_one_fourth_more_than_it_is_30_percent_less_than_80 :
    ∃ (n : ℝ), (5 / 4) * n = 56 ∧ n = 45 :=
by
  sorry

end number_one_fourth_more_than_it_is_30_percent_less_than_80_l693_693081


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693921

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693921


namespace range_of_a_l693_693710

theorem range_of_a 
  (a : ℝ) 
  (h : ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ x1 + x2 = 2 ∧ x1 * x2 = log a (a^2 - a)) : 
  1 < a ∧ a < (1 + Real.sqrt 5) / 2 := sorry

end range_of_a_l693_693710


namespace tangential_quadrilateral_sums_eq_l693_693817

theorem tangential_quadrilateral_sums_eq
  (A B C D P Q R S : Point)
  (circle : Circle)
  (h1 : tangent circle A P)
  (h2 : tangent circle B P)
  (h3 : tangent circle B Q)
  (h4 : tangent circle C Q)
  (h5 : tangent circle C R)
  (h6 : tangent circle D R)
  (h7 : tangent circle D S)
  (h8 : tangent circle A S)
  (AP_eq_AS : dist A P = dist A S)
  (BP_eq_BQ : dist B P = dist B Q)
  (CQ_eq_CR : dist C Q = dist C R)
  (DR_eq_DS : dist D R = dist D S) : 
  dist A B + dist C D = dist B C + dist D A :=
by 
  sorry

end tangential_quadrilateral_sums_eq_l693_693817


namespace trevor_coin_difference_l693_693866

theorem trevor_coin_difference:
  ∀ (total_coins quarters: ℕ),
  total_coins = 77 →
  quarters = 29 →
  (total_coins - quarters = 48) := by
  intros total_coins quarters h1 h2
  sorry

end trevor_coin_difference_l693_693866


namespace cows_value_increase_l693_693379

noncomputable def cow1_initial_weight := 732
noncomputable def cow2_initial_weight := 845
noncomputable def cow3_initial_weight := 912

noncomputable def cow1_increase_factor := 1.35
noncomputable def cow2_increase_factor := 1.28
noncomputable def cow3_increase_factor := 1.4

noncomputable def price_per_pound := 2.75

noncomputable def cow1_final_weight := cow1_initial_weight * cow1_increase_factor
noncomputable def cow2_final_weight := cow2_initial_weight * cow2_increase_factor
noncomputable def cow3_final_weight := cow3_initial_weight * cow3_increase_factor

noncomputable def cow1_value_before := cow1_initial_weight * price_per_pound
noncomputable def cow1_value_after := cow1_final_weight * price_per_pound

noncomputable def cow2_value_before := cow2_initial_weight * price_per_pound
noncomputable def cow2_value_after := cow2_final_weight * price_per_pound

noncomputable def cow3_value_before := cow3_initial_weight * price_per_pound
noncomputable def cow3_value_after := cow3_final_weight * price_per_pound

noncomputable def cow1_increase_value := cow1_value_after - cow1_value_before
noncomputable def cow2_increase_value := cow2_value_after - cow2_value_before
noncomputable def cow3_increase_value := cow3_value_after - cow3_value_before

noncomputable def total_increase_value := cow1_increase_value + cow2_increase_value + cow3_increase_value

theorem cows_value_increase (total_increase_value = 2358.40) : 
   cow1_value_before = cow1_initial_weight * price_per_pound 
   ∧ cow1_value_after = cow1_final_weight * price_per_pound 
   ∧ cow2_value_before = cow2_initial_weight * price_per_pound 
   ∧ cow2_value_after = cow2_final_weight * price_per_pound 
   ∧ cow3_value_before = cow3_initial_weight * price_per_pound 
   ∧ cow3_value_after = cow3_final_weight * price_per_pound 
   ∧ cow1_increase_value = cow1_value_after - cow1_value_before 
   ∧ cow2_increase_value = cow2_value_after - cow2_value_before 
   ∧ cow3_increase_value = cow3_value_after - cow3_value_before 
   ∧ total_increase_value = cow1_increase_value + cow2_increase_value + cow3_increase_value 
   ∧ total_increase_value = 2358.40 := 
by 
   sorry

end cows_value_increase_l693_693379


namespace floor_condition_sufficient_not_necessary_l693_693744

-- Define the floor function for real numbers
def floor (x : ℝ) : ℤ := Int.floor x

-- Define the proposition for sufficiency
def sufficient (x y : ℝ) : Prop := abs (x - y) < 1 → floor x = floor y

-- Define the proposition for non-necessity
def not_necessary (x y : ℝ) : Prop := ¬(abs (x - y) < 1) ∨ (floor x ≠ floor y)

theorem floor_condition_sufficient_not_necessary :
  (∀ x y : ℝ, floor x = floor y → abs (x - y) < 1) ∧ 
  (∃ x y : ℝ, abs (x - y) < 1 ∧ floor x ≠ floor y) :=
by
  sorry

end floor_condition_sufficient_not_necessary_l693_693744


namespace father_dig_time_l693_693808

-- Definitions based on the conditions
variable (T : ℕ) -- Time taken by the father to dig the hole in hours
variable (D : ℕ) -- Depth of the hole dug by the father in feet
variable (M : ℕ) -- Depth of the hole dug by Michael in feet

-- Conditions
def father_hole_depth : Prop := D = 4 * T
def michael_hole_depth : Prop := M = 2 * D - 400
def michael_dig_time : Prop := M = 4 * 700

-- The proof statement, proving T = 400 given the conditions
theorem father_dig_time (T D M : ℕ)
  (h1 : father_hole_depth T D)
  (h2 : michael_hole_depth D M)
  (h3 : michael_dig_time M) : T = 400 := 
by
  sorry

end father_dig_time_l693_693808


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693886

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693886


namespace g_interval_l693_693006

def g (a b c : ℝ) : ℝ :=
  (a ^ 2) / ((a ^ 2) + (b ^ 2)) +
  (b ^ 2) / ((b ^ 2) + (c ^ 2)) +
  (c ^ 2) / ((c ^ 2) + (a ^ 2))

theorem g_interval (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 < g a b c ∧ g a b c < 2 :=
sorry

end g_interval_l693_693006


namespace penelope_total_savings_l693_693418

/-- A definition of Penelope's daily savings. -/
def daily_savings (day : Nat) : ℕ :=
  if day % 7 = 5 ∨ day % 7 = 6 then 30 else 24

/-- The number of days in a non-leap year. -/
def days_in_year : ℕ := 365

/-- The total weekend savings for the year. -/
def total_weekend_savings : ℕ := 52 * 2 * 30

/-- The total weekday savings for the year. -/
def total_weekday_savings : ℕ := (days_in_year - 52 * 2) * 24

/-- The total subscription cost for the year. -/
def total_subscription_cost : ℕ := 12 * 45

/-- The total savings before interest for the year. -/
def total_savings_before_interest : ℕ :=
  total_weekend_savings + total_weekday_savings - total_subscription_cost

/-- The interest rate for the savings account. -/
def annual_interest_rate : ℝ := 0.03

/-- The interest earned for the year. -/
def interest_earned : ℝ :=
  total_savings_before_interest * annual_interest_rate.toNat

/-- The total savings at the end of the year. -/
def total_savings_at_end_of_year : ℝ :=
  total_savings_before_interest + interest_earned

/-- Theorem: Penelope's total savings at the end of the year is $9109.32. -/
theorem penelope_total_savings : total_savings_at_end_of_year = 9109.32 := by
  sorry

end penelope_total_savings_l693_693418


namespace sum_reciprocal_circumradius_equals_two_inverse_main_circumradius_l693_693278

-- Define the entities from the problem
variables (A B C O A' A1 A2 H_A : Type)
variables (triangle_ABC : Triangle A B C)
variables (R R_A R_B R_C : ℝ)

-- Define conditions
variable (isAcuteABC : AcuteTriangle triangle_ABC)
variable (circumcenterO : Circumcenter O triangle_ABC)
variable (angleEquality : ∀ X Y : Point, AngleXAO = AngleYAO)
variable (perpendicularA1 : Perpendicular A'A_1 AC)
variable (perpendicularA2 : Perpendicular A'A_2 AB)
variable (perpendicularH_A : Perpendicular AH_A BC)
variable (circumradiusRA : Circumradius R_A (Triangle H_A A1 A2))
variable (circumradiusRB : Circumradius R_B (Triangle H_B B1 B2))
variable (circumradiusRC : Circumradius R_C (Triangle H_C C1 C2))

-- Prove the required statement
theorem sum_reciprocal_circumradius_equals_two_inverse_main_circumradius :
  (1 / R_A) + (1 / R_B) + (1 / R_C) = 2 / R :=
sorry

end sum_reciprocal_circumradius_equals_two_inverse_main_circumradius_l693_693278


namespace max_shaded_area_l693_693773

open_locale classical

noncomputable def shaded_area_triangle (base height : ℝ) : ℝ := 
  (1 / 2) * base * height

noncomputable def shaded_area_rectangle (length width : ℝ) : ℝ :=
  length * width

theorem max_shaded_area (a b c d e : ℝ) (base_bases_sum : a + b + c + d + e = 1) :
  shaded_area_rectangle 1 1 > shaded_area_triangle 1 1 :=
by
  sorry

end max_shaded_area_l693_693773


namespace find_sum_of_a_b_c_l693_693068

def a := 8
def b := 2
def c := 2

theorem find_sum_of_a_b_c : a + b + c = 12 :=
by
  have ha : a = 8 := rfl
  have hb : b = 2 := rfl
  have hc : c = 2 := rfl
  sorry

end find_sum_of_a_b_c_l693_693068


namespace smallest_a_value_l693_693789

noncomputable def minimum_a (a b : ℝ) := a ≥ 0 ∧ b ≥ 0 ∧ (∀ x : ℤ, sin (a * x + b) = sin (17 * x)) ∧ ∀ c : ℝ, (c ≥ 0 ∧ ∀ x : ℤ, sin (c * x + b) = sin (17 * x)) → a ≤ c

theorem smallest_a_value (a b : ℝ) (h : minimum_a a b) : a = 17 :=
by
  sorry

end smallest_a_value_l693_693789


namespace triangle_base_l693_693344

theorem triangle_base (h : ℝ) (A : ℝ) (b : ℝ) (h_eq : h = 10) (A_eq : A = 46) (area_eq : A = (b * h) / 2) : b = 9.2 :=
by
  -- sorry to be replaced with the actual proof
  sorry

end triangle_base_l693_693344


namespace sufficient_condition_l693_693588

theorem sufficient_condition (a b : ℝ) (h1 : a > 1) (h2 : b > 1) : ab > 1 :=
sorry

end sufficient_condition_l693_693588


namespace largest_prime_factor_7fac_plus_8fac_l693_693948

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693948


namespace find_c_value_l693_693342

def f (c : ℝ) (x : ℝ) : ℝ := c * x^4 + (c^2 - 3) * x^2 + 1

theorem find_c_value (c : ℝ) :
  (∀ x < -1, deriv (f c) x < 0) ∧ 
  (∀ x, -1 < x → x < 0 → deriv (f c) x > 0) → 
  c = 1 :=
by 
  sorry

end find_c_value_l693_693342


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693928

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693928


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693922

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693922


namespace tangent_line_eq_l693_693844

theorem tangent_line_eq (b : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 1 = 0) → (∀ x y : ℝ, x^2 + y^2 = 5) →
  (∀ x y : ℝ, 2 * x - y + b = 0) → 
  (∀ d : ℝ, d = abs b / sqrt (2^2 + (-1)^2) → d = sqrt 5) →
  b = 5 ∨ b = -5 :=
by
  sorry

end tangent_line_eq_l693_693844


namespace min_dot_product_l693_693003

noncomputable def M (m : ℝ) : ℝ × ℝ := (m, 2 - m)
noncomputable def N (n : ℝ) : ℝ × ℝ := (n, 2 - n)
noncomputable def distance (A B : ℝ × ℝ) : ℝ := 
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
noncomputable def dot_product (O A B : ℝ × ℝ) : ℝ :=
  A.1 * B.1 + A.2 * B.2

theorem min_dot_product :
  ∀ (m n : ℝ), distance (M m) (N n) = real.sqrt 2 → 
  ∃ n, n = n ∧ dot_product (0, 0) (M (n + 1)) (N n) = 3/2 := 
sorry

end min_dot_product_l693_693003


namespace min_value_expression_l693_693007

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  infi (λ a b c : ℝ, (a + b + 1) / c + (a + c + 1) / b + (b + c + 1) / a) = 9 :=
sorry

end min_value_expression_l693_693007


namespace range_of_k_l693_693284

theorem range_of_k 
  (x1 x2 y1 y2 k : ℝ)
  (h1 : y1 = 2 * x1 - k * x1 + 1)
  (h2 : y2 = 2 * x2 - k * x2 + 1)
  (h3 : x1 ≠ x2)
  (h4 : (x1 - x2) * (y1 - y2) < 0) : k > 2 := 
sorry

end range_of_k_l693_693284


namespace value_of_f_2018_l693_693401

noncomputable def f : ℤ → ℤ := sorry

axiom even_function (x : ℤ) : f(x) = f(-x)
axiom value_at_one : f(1) = 1
axiom value_at_2017_ne_one : f(2017) ≠ 1
axiom inequality (x y : ℤ) : 2 * f(x + y) - f(x) - f(y) ≤ |f(x) - f(y)|

theorem value_of_f_2018 : f(2018) = 1 :=
by
  sorry

end value_of_f_2018_l693_693401


namespace volunteer_arrangement_l693_693670

def students : List String := ["A", "B", "C", "D", "E"]
def jobs : List String := ["translation", "tour guide", "etiquette", "driver"]

-- Definition of the conditions
def cannotDrive : String → Prop
| "A" := true
| "B" := true
| _   := false

def validAssignment (assignment : String → String) : Prop :=
  (∀ s, s ∈ students → assignment s ∈ jobs) ∧
  (∀ j, j ∈ jobs → ∃ s, s ∈ students ∧ assignment s = j) ∧
  (assignment "A" ≠ "driver") ∧
  (assignment "B" ≠ "driver")

-- Problem statement
theorem volunteer_arrangement :
  ∃ (assignments : {assignment : String → String // validAssignment assignment}), 
  true :=
sorry

end volunteer_arrangement_l693_693670


namespace sequence_increasing_l693_693245

theorem sequence_increasing (a : ℕ → ℝ) (a0 : ℝ) (h0 : a 0 = 1 / 5)
  (H : ∀ n : ℕ, a (n + 1) = 2^n - 3 * a n) :
  ∀ n : ℕ, a (n + 1) > a n :=
sorry

end sequence_increasing_l693_693245


namespace binomial_expansion_coeff_l693_693746

theorem binomial_expansion_coeff (a : ℝ) :
  (∀ x : ℝ, (7.choose 1) * a = 7) → a = 1 :=
by 
  intro h
  sorry

end binomial_expansion_coeff_l693_693746


namespace range_of_a_l693_693341

-- Define the function f as given in the problem
def f (x a : ℝ) : ℝ := (x^3) / 3 - (a / 2) * x^2 + x + 1

-- Define the derivative of the function f
def f_prime (x a : ℝ) : ℝ := x^2 - a * x + 1

-- The interval in question
def interval : set ℝ := set.Ioo (1 / 2) 3

-- Define the condition for having an extreme point in the interval
def has_extreme_point_in_interval (a : ℝ) : Prop :=
  ∃ x ∈ interval, f_prime x a = 0

-- The theorem to prove the range of a
theorem range_of_a (a : ℝ) : has_extreme_point_in_interval a ↔ (2 < a ∧ a < 10 / 3) :=
by sorry

end range_of_a_l693_693341


namespace angle_bisectors_right_triangle_l693_693248

structure RightTriangle (A B C : Type) [EuclideanSpace A B C] where
  (AC BC : ℝ)
  (right : isRightTriangle AC BC)

def length_of_hypotenuse (A B C : Type) [EuclideanSpace A B C] (t : RightTriangle A B C) : ℝ :=
  sqrt (t.AC^2 + t.BC^2)

def angle_bisector_length (A B C M : Type) [EuclideanSpace A B C] 
    (t : RightTriangle A B C) (bisector_fun : M → A → B) (leg : ℝ) : ℝ :=
  let opposite_segment := bisector_fun (0.5 * leg) t.AC t.BC
  sqrt (t.AC^2 + opposite_segment^2)

theorem angle_bisectors_right_triangle {A B C : Type} [EuclideanSpace A B C] (t : RightTriangle A B C) : 
  length_of_hypotenuse A B C t = 30 ∧ angle_bisector_length A B C (0.5 * 24) t.AC = 9 * sqrt 5 ∧ angle_bisector_length A B C (0.5 * 18) t.BC = 8 * sqrt 10 :=
  sorry

end angle_bisectors_right_triangle_l693_693248


namespace max_min_values_in_interval_l693_693251

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x + 5

def interval : Set ℝ := Icc (-5/2 : ℝ) (3/2)

theorem max_min_values_in_interval :
  (∃ x ∈ interval, f x = 9) ∧ (∃ x ∈ interval, f x = -11.25) :=
sorry

end max_min_values_in_interval_l693_693251


namespace smallest_possible_value_of_distance_l693_693014

noncomputable def smallest_distance (z w : ℂ) : ℝ :=
  complex.abs (z - w)

theorem smallest_possible_value_of_distance
  (z w : ℂ)
  (hz : complex.abs (z + 2 + 4 * complex.I) = 2)
  (hw : complex.abs (w - (6 + 7 * complex.I)) = 4) :
  smallest_distance z w = real.sqrt 185 - 6 := sorry

end smallest_possible_value_of_distance_l693_693014


namespace shaded_region_area_l693_693360

noncomputable def radius_of_circle (EH GH : ℝ) : ℝ := real.sqrt (EH^2 + GH^2)

noncomputable def area_of_circle (r : ℝ) : ℝ := real.pi * r^2

noncomputable def area_of_quarter_circle (area_of_circle : ℝ) : ℝ := area_of_circle / 4

noncomputable def area_of_rectangle (EH GH : ℝ) : ℝ := EH * GH

noncomputable def area_of_shaded_region (quarter_circle_area rectangle_area : ℝ) : ℝ :=
  quarter_circle_area - rectangle_area

theorem shaded_region_area :
  let EH := 5
  let GH := 12
  let r := radius_of_circle EH GH
  let circle_area := area_of_circle r
  let quarter_circle_area := area_of_quarter_circle circle_area
  let rectangle_area := area_of_rectangle EH GH
  let shaded_area := area_of_shaded_region quarter_circle_area rectangle_area
  shaded_area ≈ 72.665 :=
by sorry

end shaded_region_area_l693_693360


namespace at_least_one_truth_not_knight_l693_693511

-- Define roles
inductive Role
| knight
| liar
| normal

open Role

-- Statements
def A_statement (B_role : Role) : Bool :=
  B_role = knight

def B_statement (A_role : Role) : Bool :=
  A_role ≠ knight

-- The main theorem
theorem at_least_one_truth_not_knight (A B : Role) :
  (A ∈ [liar, normal] ∧ A_statement B = false ∧ (B = normal ∨ B = liar ∧ B_statement A = true)) ∨
  (B ∈ [normal] ∧ B_statement A = true) :=
by
  sorry

end at_least_one_truth_not_knight_l693_693511


namespace right_triangle_perimeter_l693_693571

theorem right_triangle_perimeter 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_area : 1/2 * 30 * b = 180)
  (h_pythagorean : c^2 = 30^2 + b^2)
  : a + b + c = 42 + 2 * Real.sqrt 261 :=
sorry

end right_triangle_perimeter_l693_693571


namespace no_such_function_exists_l693_693646

theorem no_such_function_exists (f : ℝ → ℝ) (Hf : ∀ x : ℝ, 2 * f (Real.cos x) = f (Real.sin x) + Real.sin x) : False :=
by
  sorry

end no_such_function_exists_l693_693646


namespace P_n_limit_l693_693785

-- The problem statement and conditions:
noncomputable def equilateral_triangle_side (a : ℝ) := a

noncomputable def point_on_AB (a : ℝ) := λ (n : ℕ), 0 ≤ n

noncomputable def BP_n_distance (a : ℝ) (n : ℕ) : ℝ :=
if n = 0 then a else let P_n_prev := BP_n_distance a (n - 1) in (3 / 4) * a - (1 / 8) * P_n_prev 

-- The final statement that needs to be proven:
theorem P_n_limit (a : ℝ) (n : ℕ) (h : 0 < a) : 
  ∃ L, L = (2 / 3) * a ∧ 
  ∀ ε > 0, ∃ N, ∀ m ≥ N, |BP_n_distance a m - L| < ε :=
begin
  sorry
end

end P_n_limit_l693_693785


namespace most_winning_team_l693_693053

section MostWinningTeam

open List

-- Define the input conditions
variables {N : ℕ}
variables (team_names : list (list string))

-- Ensure that N is within the specified range.
def valid_N : Prop := 1 ≤ N ∧ N ≤ 30

-- Ensure each winning team name list consists of exactly three names.
def valid_team_names : Prop :=
  ∀ team, team ∈ team_names → length team = 3 ∧ (∀ name, name ∈ team → 1 ≤ length name ∧ length name ≤ 10)

-- Define a function to sort a team (list of names) alphabetically
def sort_team (team : list string) : list string :=
  sort (<=) team

-- Count occurrences of each unique team name
def count_occurrences (team : list string) (teams : list (list string)) : ℕ :=
  count (λ t, t ~ team) teams

-- The theorem to prove
theorem most_winning_team
  (hN : valid_N)
  (hteam : valid_team_names team_names) :
  ∃ team_wins : (list string) × ℕ,
    team_wins.2 = maximum (map (λ team, (team, count_occurrences team (map sort_team team_names))) (map sort_team team_names)).snd :=
begin
  admit,
end

end MostWinningTeam

end most_winning_team_l693_693053


namespace extra_apples_l693_693851

-- Define the input parameters
def red_apples : ℕ := 25
def green_apples : ℕ := 17
def students_wanted_fruit : ℕ := 10

-- Define the problem
theorem extra_apples (total_apples : ℕ) (students_took_apples : ℕ) : total_apples - students_took_apples = 32 :=
by 
  let total_apples := red_apples + green_apples
  let students_took_apples := students_wanted_fruit
  have h1 : total_apples = 42 := by 
    rw [red_apples, green_apples]
    sorry
  have h2 : students_took_apples = 10 := by 
    rw [students_wanted_fruit]
    sorry
  rw [h1, h2]
  sorry

end extra_apples_l693_693851


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693099

theorem frequencies_of_first_class_products 
  (aA bA aB bB : ℕ) 
  (total_A total_B total_products : ℕ) 
  (hA : aA = 150) (hB : bA = 50) (hC : aB = 120) (hD : bB = 80) 
  (hE : total_A = 200) (hF : total_B = 200) (h_total : total_products = total_A + total_B) : 
  aA / total_A = 3 / 4 ∧ aB / total_B = 3 / 5 := 
by
  sorry

theorem confidence_in_difference_of_quality 
  (n a b c d : ℕ) 
  (h_n : n = 400) (hA : a = 150) (hB : b = 50) (hC : c = 120) (hD : d = 80) 
  (K2 : ℝ) 
  (hK2 : K2 = n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))) : 
  K2 > 6.635 ∧ K2 < 10.828 := 
by
  sorry

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693099


namespace max_distance_from_origin_to_line_l693_693718

variable (k : ℝ)

def line (x y : ℝ) : Prop := k * x + y + 1 = 0

theorem max_distance_from_origin_to_line :
  ∃ k : ℝ, ∀ x y : ℝ, line k x y -> dist (0, 0) (x, y) ≤ 1 := 
sorry

end max_distance_from_origin_to_line_l693_693718


namespace sum_of_all_possible_k_l693_693691

noncomputable def sum_possible_k : ℕ :=
  let valid_k := {k | ∃ (a_3 a_2 a_1 a_0 : ℕ), 0 ≤ a_0 ∧ a_0 ≤ 16 ∧ 0 ≤ a_1 ∧ a_1 ≤ 16 ∧
                                   0 ≤ a_2 ∧ a_2 ≤ 16 ∧ 1 ≤ a_3 ∧ a_3 ≤ 16 ∧
                                   (49 * a_3 - 8 * a_2 + a_1 = 0) ∧
                                   (k = -51 * a_3 + 17 * a_2 - 4 * a_1 + a_0)} in
  valid_k.to_finset.sum id

theorem sum_of_all_possible_k : sum_possible_k = 2568 := by sorry

end sum_of_all_possible_k_l693_693691


namespace cos_angle_identity_l693_693681

theorem cos_angle_identity (a : ℝ) (h : Real.sin (π / 6 - a) - Real.cos a = 1 / 3) :
  Real.cos (2 * a + π / 3) = 7 / 9 :=
by
  sorry

end cos_angle_identity_l693_693681


namespace compute_63_times_57_l693_693616

theorem compute_63_times_57 : 63 * 57 = 3591 := 
by {
   have h : (60 + 3) * (60 - 3) = 60^2 - 3^2, from
     by simp [mul_add, add_mul, add_assoc, sub_mul, mul_sub, sub_add, sub_sub, add_sub, mul_self_sub],
   have h1 : 60^2 = 3600, from rfl,
   have h2 : 3^2 = 9, from rfl,
   have h3 : 60^2 - 3^2 = 3600 - 9, by rw [h1, h2],
   rw h at h3,
   exact h3,
}

end compute_63_times_57_l693_693616


namespace seventh_observation_is_nine_l693_693476

variable (observations : Fin 6 → ℕ)

def average_six_observations_is_sixteen := (∑ i, observations i) = 6 * 16

def new_observation_decreases_average 
  (new_obs : ℕ) (h_avg_six : average_six_observations_is_sixteen observations) := 
  ((∑ i, observations i) + new_obs) = 7 * 15

theorem seventh_observation_is_nine 
  (observations : Fin 6 → ℕ) 
  (new_obs : ℕ) 
  (h_avg_six : average_six_observations_is_sixteen observations) 
  (h_new_avg : new_observation_decreases_average new_obs h_avg_six):
  new_obs = 9 := 
by 
  sorry

end seventh_observation_is_nine_l693_693476


namespace algebra_inequality_l693_693020

theorem algebra_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a^3 + b^3 + c^3 = 3) : 
  1 / (a^2 + a + 1) + 1 / (b^2 + b + 1) + 1 / (c^2 + c + 1) ≥ 1 := 
by 
  sorry

end algebra_inequality_l693_693020


namespace determine_set_B_l693_693802
open Set

/-- Given problem conditions and goal in Lean 4 -/
theorem determine_set_B (U A B : Set ℕ) (hU : U = { x | x < 10 } )
  (hA_inter_compl_B : A ∩ (U \ B) = {1, 3, 5, 7, 9} ) :
  B = {2, 4, 6, 8} :=
by
  sorry

end determine_set_B_l693_693802


namespace largest_prime_factor_7fac_plus_8fac_l693_693951

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693951


namespace angle_less_than_20_degrees_l693_693497

theorem angle_less_than_20_degrees (point : Type) (lines : Finset (point → Prop)) (h_lines : lines.card = 10) :
  ∃ θ : ℝ, θ < 20 ∧ θ ∈ (Finset.image (fun l => angle_between_lines l.1 l.2) lines) :=
by
  sorry

end angle_less_than_20_degrees_l693_693497


namespace right_triangle_perimeter_l693_693572

theorem right_triangle_perimeter 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_area : 1/2 * 30 * b = 180)
  (h_pythagorean : c^2 = 30^2 + b^2)
  : a + b + c = 42 + 2 * Real.sqrt 261 :=
sorry

end right_triangle_perimeter_l693_693572


namespace sum_y_coordinates_of_intersection_with_y_axis_l693_693208

-- Define the center and radius of the circle
def center : ℝ × ℝ := (-4, 5)
def radius : ℝ := 9

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + center.1)^2 + (y - center.2)^2 = radius^2

theorem sum_y_coordinates_of_intersection_with_y_axis : 
  ∃ y1 y2 : ℝ, circle_eq 0 y1 ∧ circle_eq 0 y2 ∧ y1 + y2 = 10 :=
by
  sorry

end sum_y_coordinates_of_intersection_with_y_axis_l693_693208


namespace steve_speed_back_l693_693538

theorem steve_speed_back :
  ∀ (v : ℝ), v > 0 → (20 / v + 20 / (2 * v) = 6) → 2 * v = 10 := 
by
  intros v v_pos h
  sorry

end steve_speed_back_l693_693538


namespace f_lt_g_l693_693402

noncomputable def f (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), 1 / real.sqrt (i.succ : ℝ)

noncomputable def g (n : ℕ) : ℝ := 2 * real.sqrt n

theorem f_lt_g (n : ℕ) (h1 : n > 2) : f n < g n := sorry

end f_lt_g_l693_693402


namespace fibonacci_eighth_term_l693_693753

def fibonacci_sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 2 ∧ a 2 = 9 ∧ ∀ n ≥ 2, a (n + 1) = a n + a (n - 1)

theorem fibonacci_eighth_term :
  ∀ a : ℕ → ℕ, fibonacci_sequence a → a 7 = 107 :=
by
  intro a
  intro h
  have h₁ : a 1 = a 2 - a 0, from sorry
  -- Additional steps and calculations would follow here.
  sorry

end fibonacci_eighth_term_l693_693753


namespace bob_expected_difference_leap_year_l693_693605

noncomputable def expected_difference (days : ℕ) : ℕ :=
let p_odd := 4/7,
    p_even := 3/7 in
nat.round ((p_odd - p_even) * days)

theorem bob_expected_difference_leap_year : expected_difference 366 = 52 := 
by 
  sorry

end bob_expected_difference_leap_year_l693_693605


namespace simplify_fraction_90_150_l693_693446

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end simplify_fraction_90_150_l693_693446


namespace largest_prime_factor_of_7fact_8fact_l693_693899

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693899


namespace part1_part2_l693_693699

variable (x : ℝ) (a : Fin 11 → ℝ)

noncomputable def polynomial_expansion := 
  (x^2 + 1) * (2*x - 1)^8 = ∑ i in Finset.range 11, a i * (x + 1)^i

theorem part1 : ∃ a : Fin 11 → ℝ, polynomial_expansion x a → ∑ i in Finset.range 11, a i = 1 :=
by
  sorry 

theorem part2 : 
  ∃ a : Fin 11 → ℝ, polynomial_expansion x a → (∑ i in Finset.filter (λ n, n % 2 = 0) (Finset.range 11), a i = (1 / 2) * 5^9 + 1 / 2) ∧ 
  (∑ i in Finset.filter (λ n, n % 2 = 1) (Finset.range 11), a i = (1 / 2) - (1 / 2) * 5^9) :=
by 
  sorry 

end part1_part2_l693_693699


namespace one_ge_one_of_a_or_b_l693_693422

-- Define the variables and hypotheses
variable (x : ℝ)
def a := |cos x - sin x|
def b := |cos x + sin x|

-- State the theorem
theorem one_ge_one_of_a_or_b :
  a ≥ 1 ∨ b ≥ 1 :=
sorry

end one_ge_one_of_a_or_b_l693_693422


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693098

theorem frequencies_of_first_class_products 
  (aA bA aB bB : ℕ) 
  (total_A total_B total_products : ℕ) 
  (hA : aA = 150) (hB : bA = 50) (hC : aB = 120) (hD : bB = 80) 
  (hE : total_A = 200) (hF : total_B = 200) (h_total : total_products = total_A + total_B) : 
  aA / total_A = 3 / 4 ∧ aB / total_B = 3 / 5 := 
by
  sorry

theorem confidence_in_difference_of_quality 
  (n a b c d : ℕ) 
  (h_n : n = 400) (hA : a = 150) (hB : b = 50) (hC : c = 120) (hD : d = 80) 
  (K2 : ℝ) 
  (hK2 : K2 = n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))) : 
  K2 > 6.635 ∧ K2 < 10.828 := 
by
  sorry

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693098


namespace greatest_fraction_lt_17_l693_693517

theorem greatest_fraction_lt_17 :
  ∃ (x : ℚ), x = 15 / 4 ∧ x^2 < 17 ∧ ∀ y : ℚ, y < 4 → y^2 < 17 → y ≤ 15 / 4 := 
by
  use 15 / 4
  sorry

end greatest_fraction_lt_17_l693_693517


namespace range_a_derivative_sqrt_value_at_t_l693_693846

variable {a x1 x2 t : ℝ}

-- Q1: Prove that the range of a such that the function intersects the x-axis at two distinct points is a > e^2.
theorem range_a (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ (exp x1 - a * x1 + a = 0) ∧ (exp x2 - a * x2 + a = 0)) : 
  a > Real.exp 2 := 
sorry

-- Q2: Given the function intersects the x-axis at A(x1, 0) and B(x2, 0) with x1 < x2, prove that f'(√(x1 * x2)) < 0.
theorem derivative_sqrt (h : exp x1 - a * x1 + a = 0) (h' : exp x2 - a * x2 + a = 0) (h'' : x1 < x2) : 
  (exp (Real.sqrt(x1 * x2)) - a) < 0 := 
sorry

-- Q3: Given √((x2 - 1) / (x1 - 1)) = t, find the value of (a - 1)(t - 1) for the function such that A(x1, 0), B(x2, 0), and C form an isosceles right triangle with C on the graph of f.
theorem value_at_t (h : exp x1 - a * x1 + a = 0) (h' : exp x2 - a * x2 + a = 0) (h'' : x1 < x2) (ht : Real.sqrt ((x2 - 1) / (x1 - 1)) = t): 
  (a - 1) * (t - 1) = 2 := 
sorry

end range_a_derivative_sqrt_value_at_t_l693_693846


namespace largest_prime_factor_7fac_plus_8fac_l693_693959

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693959


namespace base_eight_to_base_ten_l693_693877

theorem base_eight_to_base_ten (n : ℕ) (h : n = 4 * 8^1 + 7 * 8^0) : n = 39 := by
  sorry

end base_eight_to_base_ten_l693_693877


namespace problem_statement_l693_693258

variable {f : ℝ → ℝ} 
variable (hf : Differentiable ℝ f)
variable (hf' : ∀ x, x < -1 → deriv f x ≥ 0)

theorem problem_statement : f(0) + f(2) ≥ 2 * f(1) := by
  sorry

end problem_statement_l693_693258


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693101

theorem frequencies_of_first_class_products 
  (aA bA aB bB : ℕ) 
  (total_A total_B total_products : ℕ) 
  (hA : aA = 150) (hB : bA = 50) (hC : aB = 120) (hD : bB = 80) 
  (hE : total_A = 200) (hF : total_B = 200) (h_total : total_products = total_A + total_B) : 
  aA / total_A = 3 / 4 ∧ aB / total_B = 3 / 5 := 
by
  sorry

theorem confidence_in_difference_of_quality 
  (n a b c d : ℕ) 
  (h_n : n = 400) (hA : a = 150) (hB : b = 50) (hC : c = 120) (hD : d = 80) 
  (K2 : ℝ) 
  (hK2 : K2 = n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))) : 
  K2 > 6.635 ∧ K2 < 10.828 := 
by
  sorry

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693101


namespace samuel_total_distance_l693_693823

variable (speed1 time1 speed2 time2 remaining_distance : ℕ)

def distance1 : ℕ := speed1 * time1
def distance2 : ℕ := speed2 * time2

def total_distance : ℕ := distance1 + distance2 + remaining_distance

theorem samuel_total_distance (h1 : speed1 = 50) (h2 : time1 = 3)
                              (h3 : speed2 = 80) (h4 : time2 = 4)
                              (h5 : remaining_distance = 130) :
                              total_distance speed1 time1 speed2 time2 remaining_distance = 600 :=
by
  sorry

end samuel_total_distance_l693_693823


namespace factorize_a_squared_plus_2a_l693_693238

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2*a = a * (a + 2) :=
sorry

end factorize_a_squared_plus_2a_l693_693238


namespace arithmetic_mean_integers_neg5_to_6_l693_693873

theorem arithmetic_mean_integers_neg5_to_6 : 
  (let range := (-5:ℤ) to 6 in 
   let count := (6 - -5 + 1 : ℤ) in 
   let sum := (range.sum : ℤ) in
   (sum : ℤ) / (count : ℤ) = (0.5 : ℚ)) :=
by
  let range := List.range' (-5) (1 + 6 - -5 : ℤ).toNat
  let count := 12
  let sum := range.sum
  have : sum = 6, sorry
  have : sum / count = 0.5, sorry
  exact this

end arithmetic_mean_integers_neg5_to_6_l693_693873


namespace solve_equation_l693_693463

theorem solve_equation (x : ℝ) :
  (2 * x - 1)^2 - 25 = 0 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end solve_equation_l693_693463


namespace power_of_two_sequence_invariant_l693_693782

theorem power_of_two_sequence_invariant
  (n : ℕ)
  (a b : ℕ → ℕ)
  (h₀ : a 0 = 1)
  (h₁ : b 0 = n)
  (hi : ∀ i : ℕ, a i < b i → a (i + 1) = 2 * a i + 1 ∧ b (i + 1) = b i - a i - 1)
  (hj : ∀ i : ℕ, a i > b i → a (i + 1) = a i - b i - 1 ∧ b (i + 1) = 2 * b i + 1)
  (hk : ∀ i : ℕ, a i = b i → a (i + 1) = a i ∧ b (i + 1) = b i)
  (k : ℕ)
  (h : a k = b k) :
  ∃ m : ℕ, n + 3 = 2 ^ m :=
by
  sorry

end power_of_two_sequence_invariant_l693_693782


namespace power_of_two_representation_l693_693671

def f (n : ℕ) : ℕ :=
  -- This is a placeholder for the actual definition of f, which should be 
  -- the function counting the number of representations of n as sums of powers of 2.
  sorry 

theorem power_of_two_representation (n : ℕ) (h : n ≥ 3) : 
  2 ^ (n^2 / 4) < f(2^n) ∧ f(2^n) < 2 ^ (n^2 / 2) :=
sorry

end power_of_two_representation_l693_693671


namespace right_triangle_perimeter_l693_693570

theorem right_triangle_perimeter (area leg1 : ℕ) (h_area : area = 180) (h_leg1 : leg1 = 30) :
  ∃ leg2 hypotenuse perimeter, 
    (2 * area = leg1 * leg2) ∧ 
    (hypotenuse^2 = leg1^2 + leg2^2) ∧ 
    (perimeter = leg1 + leg2 + hypotenuse) ∧ 
    (perimeter = 42 + 2 * Real.sqrt 261) :=
by
  sorry

end right_triangle_perimeter_l693_693570


namespace tetrahedron_vector_dot_product_l693_693276

open EuclideanGeometry

theorem tetrahedron_vector_dot_product (
  (A B C D E F : Point)
  (tetra : regular_tetrahedron A B C D)
  (edge_length : dist A B = 1)
  (AE_eq_2EB : vector_relation A E 2 B)
  (AF_eq_2FD : vector_relation A F 2 D)
) : vector_dot_product (vector_sub E F) (vector_sub D C) = -1 / 3 := 
sorry

end tetrahedron_vector_dot_product_l693_693276


namespace jason_egg_consumption_l693_693235

-- Definition for the number of eggs Jason consumes per day
def eggs_per_day : ℕ := 3

-- Definition for the number of days in a week
def days_in_week : ℕ := 7

-- Definition for the number of weeks we are considering
def weeks : ℕ := 2

-- The statement we want to prove, which combines all the conditions and provides the final answer
theorem jason_egg_consumption : weeks * days_in_week * eggs_per_day = 42 := by
sorry

end jason_egg_consumption_l693_693235


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693883

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693883


namespace frequencies_of_first_class_quality_difference_confidence_l693_693114

section quality_comparison

variables (n a b c d : ℕ)

-- Given conditions
def total_products : ℕ := 400
def machine_a_total : ℕ := 200
def machine_a_first : ℕ := 150
def machine_a_second : ℕ := 50
def machine_b_total : ℕ := 200
def machine_b_first : ℕ := 120
def machine_b_second : ℕ := 80

-- Defining the K^2 calculation formula
def K_squared : ℚ :=
  (total_products * (machine_a_first * machine_b_second - machine_a_second * machine_b_first) ^ 2 : ℚ) /
  ((machine_a_first + machine_a_second) * (machine_b_first + machine_b_second) * (machine_a_first + machine_b_first) * (machine_a_second + machine_b_second))

-- Proof statement for Q1: Frequencies of first-class products
theorem frequencies_of_first_class :
  machine_a_first / machine_a_total = 3 / 4 ∧ 
  machine_b_first / machine_b_total = 3 / 5 := 
sorry

-- Proof statement for Q2: Confidence level of difference in quality
theorem quality_difference_confidence :
  K_squared = 10.256 ∧ 10.256 > 6.635 → 0.99 :=
sorry

end quality_comparison

end frequencies_of_first_class_quality_difference_confidence_l693_693114


namespace similar_triangles_MNP_l693_693498

theorem similar_triangles_MNP (A B C D : Point) (sphere : Sphere) :
  ∀ (M N P : Point),
  passes_through sphere A ∧ passes_through sphere B ∧ passes_through sphere C ∨
  passes_through sphere A ∧ passes_through sphere B ∧ passes_through sphere D ∨
  passes_through sphere A ∧ passes_through sphere C ∧ passes_through sphere D ∨
  passes_through sphere B ∧ passes_through sphere C ∧ passes_through sphere D →
  intersects_edge sphere A D M ∧ intersects_edge sphere B D N ∧ intersects_edge sphere C D P ∨
  intersects_edge sphere A B M ∧ intersects_edge sphere C B N ∧ intersects_edge sphere D B P ∨
  intersects_edge sphere A C M ∧ intersects_edge sphere B C N ∧ intersects_edge sphere D C P ∨
  intersects_edge sphere A D M ∧ intersects_edge sphere B D N ∧ intersects_edge sphere C D P →
  triangle_similar M N P (another_MNP : Point × Point × Point) :=
by
  sorry

end similar_triangles_MNP_l693_693498


namespace tangent_line_equations_minimum_area_quadrilateral_l693_693709

noncomputable def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

noncomputable def point_on_line (x y : ℝ) : Prop := x + y - 6 = 0

def tangent_line_through_point (m k : ℝ) : Prop :=
  (k = 0 → ∃ b, ∀ x, x = 3 → tangent_line x 0 3 b) ∧
  (k ≠ 0 → k = (m^2 - 1) / (2 * m) ∧ ∃ b, ∀ x y, y = k * x + b → tangent_line x y 3 (k * 3 + b))

theorem tangent_line_equations (m : ℝ) :
  (∀ x y, circle_equation x y) →
  ∃ k, tangent_line_through_point m k :=
sorry

theorem minimum_area_quadrilateral :
  ∃ S : ℝ, S = sqrt 7 ∧ ∃ x y : ℝ, point_on_line x y ∧ x = 4 ∧ y = 2 :=
sorry

end tangent_line_equations_minimum_area_quadrilateral_l693_693709


namespace num_subsets_of_set_l693_693730

theorem num_subsets_of_set 
  (A : set ℕ) (hA : A = {0, 1, 2}) : (nat.card {B : set ℕ // B ⊆ A} = 8) := 
begin
  sorry
end

end num_subsets_of_set_l693_693730


namespace subproblem1_l693_693201

theorem subproblem1 (a : ℝ) : a^3 * a + (2 * a^2)^2 = 5 * a^4 := 
by sorry

end subproblem1_l693_693201


namespace number_of_buckets_after_reduction_l693_693502

def initial_buckets : ℕ := 25
def reduction_factor : ℚ := 2 / 5

theorem number_of_buckets_after_reduction :
  (initial_buckets : ℚ) * (1 / reduction_factor) = 63 := by
  sorry

end number_of_buckets_after_reduction_l693_693502


namespace triangle_inequality_l693_693388

section TriangleInequality

variables (A B C : Type) [inner_product_space ℝ A]

-- Let ABC be an acute-angled triangle
variables {a b c : ℝ} (ABC : triangle A B C)
  (acute_ABC : (ABC.is_acute_angled))
  (AD : ray A B) (BE : ray B C) (CF : ray C A)
  (D E F : point ℝ) -- Points on BC, CA, AB respectively
  (interior_bisectors : interior_bisectors AD BE CF D E F)

-- Definition for the goal
theorem triangle_inequality
  (EF BC FD CA DE AB : ℝ) (r R : ℝ) :
  (EF / BC) + (FD / CA) + (DE / AB) ≥ 1 + (r / R) :=
sorry

end TriangleInequality

end triangle_inequality_l693_693388


namespace circle_touching_y_axis_radius_5_k_value_l693_693676

theorem circle_touching_y_axis_radius_5_k_value :
  ∃ k : ℝ, ∀ x y : ℝ, (x^2 + 8 * x + y^2 + 4 * y - k = 0) →
    (∃ r : ℝ, r = 5 ∧ (∀ c : ℝ × ℝ, (c.1 + 4)^2 + (c.2 + 2)^2 = r^2) ∧
      (∃ x : ℝ, x + 4 = 0)) :=
by
  sorry

end circle_touching_y_axis_radius_5_k_value_l693_693676


namespace fuel_for_first_third_l693_693804

def total_fuel : ℕ := 60
def fuel_second_third : ℕ := total_fuel / 3
def fuel_final_third : ℕ := fuel_second_third / 2
def fuel_first_third : ℕ := total_fuel - fuel_second_third - fuel_final_third

theorem fuel_for_first_third (total_fuel : ℕ) : 
  (total_fuel = 60) → 
  (fuel_first_third = total_fuel - (total_fuel / 3) - (total_fuel / 6)) →
  fuel_first_third = 30 := 
by 
  intros h1 h2
  rw h1 at h2
  norm_num at h2
  exact h2

end fuel_for_first_third_l693_693804


namespace problem_l693_693340

theorem problem
  (f : ℝ → ℝ)
  (x₁ x₂ : ℝ)
  (φ : ℝ)
  (h₀ : ∀ x, f x = sqrt 2 * sin (2 * x + φ))
  (h₁ : |φ| < π / 2)
  (h₂ : ∀ x, f x = f (π / 6 - x))
  (h₃ : x₁ ∈ Ioo (-17 * π / 12) (-2 * π / 3))
  (h₄ : x₂ ∈ Ioo (-17 * π / 12) (-2 * π / 3))
  (h₅ : x₁ ≠ x₂)
  (h₆ : f x₁ = f x₂)
 : f (x₁ + x₂) = sqrt 6 / 2 := 
sorry

end problem_l693_693340


namespace largest_prime_factor_of_fact_sum_is_7_l693_693961

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693961


namespace weight_of_new_student_l693_693537

theorem weight_of_new_student (avg_weight_19_students : ℝ) (num_19_students : ℕ) (new_avg_weight : ℝ) (num_20_students : ℕ) :
  avg_weight_19_students = 15 ∧ num_19_students = 19 ∧ new_avg_weight = 14.6 ∧ num_20_students = 20 →
  292 - 285 = 7 :=
by
  intros h
  have h1 : avg_weight_19_students = 15 := h.1
  have h2 : num_19_students = 19 := h.2.1
  have h3 : new_avg_weight = 14.6 := h.2.2.1
  have h4 : num_20_students = 20 := h.2.2.2
  -- The proof part is omitted
  exact 7 = 7

end weight_of_new_student_l693_693537


namespace product_quality_difference_l693_693087

variable (n a b c d : ℕ)
variable (P_K_2 : ℝ → ℝ)

def first_class_freq_A := a / (a + b : ℕ)
def first_class_freq_B := c / (c + d : ℕ)

def K2 := (n : ℝ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem product_quality_difference
  (ha : a = 150) (hb : b = 50) 
  (hc : c = 120) (hd : d = 80)
  (hn : n = 400)
  (hK : P_K_2 0.010 = 6.635) : 
  first_class_freq_A a b = 3 / 4 ∧
  first_class_freq_B c d = 3 / 5 ∧
  K2 n a b c d > P_K_2 0.010 :=
by {
  sorry
}

end product_quality_difference_l693_693087


namespace quadrilateralRhombus_l693_693015

variable (A B C D P Q R S: Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B]
variable (AB : A → B → ℝ)
variable (BC : B → C → ℝ)
variable (CD : C → D → ℝ)
variable (DA : D → A → ℝ)

-- Conditions
variable (isConvexQuadrilateral : ConvexQuadrilateral A B C D)
variable (midpointP : P = midpoint A B)
variable (midpointQ : Q = midpoint B C)
variable (midpointR : R = midpoint C D)
variable (midpointS : S = midpoint D A)
variable (equilateralAQR : EquilateralTriangle A Q R)
variable (equilateralCSP : EquilateralTriangle C S P)

-- Conclusion
theorem quadrilateralRhombus (isConvexQuadrilateral : ConvexQuadrilateral A B C D) 
(midpointP : P = midpoint A B) 
(midpointQ : Q = midpoint B C) 
(midpointR : R = midpoint C D) 
(midpointS : S = midpoint D A) 
(equilateralAQR : EquilateralTriangle A Q R) 
(equilateralCSP : EquilateralTriangle C S P) : 
(isRhombus A B C D) ∧ (angle A B C = 60) :=
sorry

end quadrilateralRhombus_l693_693015


namespace train_stop_time_l693_693139

theorem train_stop_time (v_excluding_stoppages v_including_stoppages : ℕ) 
    (h1 : v_excluding_stoppages = 45) 
    (h2 : v_including_stoppages = 30) : 
    ∃ (minutes_stopped : ℕ), minutes_stopped = 20 :=
by
  let speed_reduction := v_excluding_stoppages - v_including_stoppages
  have speed_reduction_calc : speed_reduction = 15 := by
    rw [h1, h2]
    rfl
  let hours_stopped := speed_reduction / v_excluding_stoppages
  have hours_stopped_calc : hours_stopped = 1 / 3 := by
    rw [speed_reduction_calc, h1]
    norm_num
  let minutes_stopped := hours_stopped * 60
  have minutes_stopped_calc : minutes_stopped = 20 := by
    rw [hours_stopped_calc]
    norm_num
  use minutes_stopped
  exact minutes_stopped_calc

end train_stop_time_l693_693139


namespace find_triangle_sides_l693_693492

variable (a b c : ℕ)
variable (P : ℕ)
variable (R : ℚ := 65 / 8)
variable (r : ℕ := 4)

theorem find_triangle_sides (h1 : R = 65 / 8) (h2 : r = 4) (h3 : P = a + b + c) : 
  a = 13 ∧ b = 14 ∧ c = 15 :=
  sorry

end find_triangle_sides_l693_693492


namespace triangle_midline_l693_693371

-- Definitions of the elements in the condition
variables (A B C D E : Type)
variables (AB AC BC DE : ℝ)
variables (is_midpoint : D = midpoint A B)
variables (is_parallel : is_parallel DE BC)
variables (length_DE : DE = 4)

-- Lean statement for the proof problem
theorem triangle_midline (h_midpoint : is_midpoint) (h_parallel : is_parallel) (h_DE : length_DE) : BC = 8 :=
sorry

end triangle_midline_l693_693371


namespace linear_eq_implies_k_eq_three_l693_693264

theorem linear_eq_implies_k_eq_three (k : ℝ) : 
  (∀ x : ℝ, (k - 3) * x^2 + 2 * x - 3 = 0 → (k - 3) = 0) → k = 3 :=
by
  intros h
  have hk : k - 3 = 0 := h 0 (by norm_num)
  linarith

end linear_eq_implies_k_eq_three_l693_693264


namespace frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693095

theorem frequency_machine_A (total_A first_class_A : ℕ) (h_total_A: total_A = 200) (h_first_class_A: first_class_A = 150) :
  first_class_A / total_A = 3 / 4 := by
  rw [h_total_A, h_first_class_A]
  norm_num

theorem frequency_machine_B (total_B first_class_B : ℕ) (h_total_B: total_B = 200) (h_first_class_B: first_class_B = 120) :
  first_class_B / total_B = 3 / 5 := by
  rw [h_total_B, h_first_class_B]
  norm_num

theorem chi_square_test_significance (n a b c d : ℕ) (h_n: n = 400) (h_a: a = 150) (h_b: b = 50) 
  (h_c: c = 120) (h_d: d = 80) :
  let K_squared := (n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d)))
  in K_squared > 6.635 := by
  rw [h_n, h_a, h_b, h_c, h_d]
  let num := 400 * (150 * 80 - 50 * 120)^2
  let denom := (150 + 50) * (120 + 80) * (150 + 120) * (50 + 80)
  have : K_squared = num / denom := rfl
  norm_num at this
  sorry

end frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693095


namespace find_a_l693_693243

theorem find_a (a b : ℤ) (h : ∀ x, x^2 - x - 1 = 0 → ax^18 + bx^17 + 1 = 0) : a = 1597 :=
sorry

end find_a_l693_693243


namespace howard_finances_l693_693384

theorem howard_finances (W D X Y : ℝ) (h_initial : 26) (h_final : 52) :
  26 + W + D - X - Y = 52 :=
sorry

end howard_finances_l693_693384


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693925

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693925


namespace num_planes_alpha_eq_32_l693_693280

-- We define four non-coplanar points A, B, C, D
variables (A B C D : Point)
-- The condition that distances between any two points are distinct
variable (distinct_distances : ∀ (P Q : Point), P ≠ Q → dist P Q ≠ dist Q P )
-- We define a plane α
variable (α : Plane)

-- Condition: Three points are equidistant from α, and the fourth is at a distance twice that of the three equidistant points

-- Define the property that a point is equidistant from a plane
def equidistant (p : Point) (α : Plane) (d : ℝ) : Prop :=
  dist p α = d

-- Define the property that a point is twice the distance from the plane as another point
def twice_distance (p q : Point) (α : Plane) : Prop :=
  dist p α = 2 * dist q α

-- Now we state the theorem to prove the number of such planes
theorem num_planes_alpha_eq_32 :
  ∃ (count : ℕ), count = 32 ∧ 
  (∃ (d : ℝ), equidistant A α d ∧ equidistant B α d ∧ equidistant C α d ∧ twice_distance D A α ∨
               equidistant A α d ∧ equidistant B α d ∧ twice_distance C D α ∨
               equidistant A α d ∧ twice_distance B C α ∨
               twice_distance A D α) :=
by
  sorry

end num_planes_alpha_eq_32_l693_693280


namespace determine_c_l693_693406

theorem determine_c (a b c : ℤ) (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h2 : ∀ x : ℤ, f x = x^3 + a * x^2 + b * x + c)
  (h3 : f a = a^3) (h4 : f b = b^3) : c = 16 :=
by 
  have g : (x : ℤ) → a * x^2 + b * x + c = 0 := sorry
  have h5 : a and b are roots of g := sorry
  have sum_of_roots : a + b = -b := sorry
  have product_of_roots : a * b = c := sorry
  have eliminated_b_using_vieta : c = 16 := sorry
  exact eliminated_b_using_vieta

end determine_c_l693_693406


namespace arithmetic_sequence_and_sum_proof_l693_693762

-- Define the arithmetic sequence given the conditions a₂ = 3 and a₈ = 9
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 2 = 3 ∧ a 8 = 9 ∧ ∀ n : ℕ, a (n + 1) = a n + 1

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n = 2 ^ (n + 1)

-- States that for a given sequence, the sum of the first n terms is the specified sum formula
def sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 2 ^ (n + 3) - 8

theorem arithmetic_sequence_and_sum_proof :
  ∃ (a : ℕ → ℕ) (S : ℕ → ℕ),
    arithmetic_sequence a ∧
    (∀ n : ℕ, a n = n + 1) ∧
    geometric_sequence (λ n, 2 ^ (a n)) ∧
    sum_of_first_n_terms (λ n, 2 ^ (a n)) S :=
by
  sorry

end arithmetic_sequence_and_sum_proof_l693_693762


namespace mork_effective_tax_rate_theorem_mindy_effective_tax_rate_theorem_combined_effective_tax_rate_theorem_l693_693027

noncomputable def Mork_base_income (M : ℝ) : ℝ := M
noncomputable def Mindy_base_income (M : ℝ) : ℝ := 4 * M
noncomputable def Mork_total_income (M : ℝ) : ℝ := 1.5 * M
noncomputable def Mindy_total_income (M : ℝ) : ℝ := 6 * M

noncomputable def Mork_total_tax (M : ℝ) : ℝ :=
  0.4 * M + 0.5 * 0.5 * M
noncomputable def Mindy_total_tax (M : ℝ) : ℝ :=
  0.3 * 4 * M + 0.35 * 2 * M

noncomputable def Mork_effective_tax_rate (M : ℝ) : ℝ :=
  (Mork_total_tax M) / (Mork_total_income M)

noncomputable def Mindy_effective_tax_rate (M : ℝ) : ℝ :=
  (Mindy_total_tax M) / (Mindy_total_income M)

noncomputable def combined_effective_tax_rate (M : ℝ) : ℝ :=
  (Mork_total_tax M + Mindy_total_tax M) / (Mork_total_income M + Mindy_total_income M)

theorem mork_effective_tax_rate_theorem (M : ℝ) : Mork_effective_tax_rate M = 43.33 / 100 := sorry
theorem mindy_effective_tax_rate_theorem (M : ℝ) : Mindy_effective_tax_rate M = 31.67 / 100 := sorry
theorem combined_effective_tax_rate_theorem (M : ℝ) : combined_effective_tax_rate M = 34 / 100 := sorry

end mork_effective_tax_rate_theorem_mindy_effective_tax_rate_theorem_combined_effective_tax_rate_theorem_l693_693027


namespace cosine_probability_l693_693565

noncomputable def probability_cos_between (x : ℝ) (h : -3/2 ≤ x ∧ x ≤ 3/2) : Prop :=
  let intervalLength := 3
  let targetIntervalLength := 2
  (∃ x, h ∧ (1/2 ≤ Real.cos (π / 3 * x) ∧ Real.cos (π / 3 * x) ≤ 1)) →
  (targetIntervalLength / intervalLength = 2 / 3)

theorem cosine_probability :
  ∀ x : ℝ, -3/2 ≤ x ∧ x ≤ 3/2 → probability_cos_between x sorry :=
sorry

end cosine_probability_l693_693565


namespace triangle_right_angle_l693_693584

theorem triangle_right_angle {A B C : ℝ} 
  (h1 : A + B + C = 180)
  (h2 : A = B)
  (h3 : A = (1/2) * C) :
  C = 90 :=
by 
  sorry

end triangle_right_angle_l693_693584


namespace largest_prime_factor_7fac_8fac_l693_693935

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693935


namespace line_intersects_x_axis_at_neg3_l693_693601

theorem line_intersects_x_axis_at_neg3 :
  ∃ (x y : ℝ), (5 * y - 7 * x = 21 ∧ y = 0) ↔ (x = -3 ∧ y = 0) :=
by
  sorry

end line_intersects_x_axis_at_neg3_l693_693601


namespace min_sum_of_factors_of_9_factorial_l693_693059

theorem min_sum_of_factors_of_9_factorial (p q r s : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (h : p * q * r * s = (9!)) : 
  p + q + r + s ≥ 132 := 
sorry

end min_sum_of_factors_of_9_factorial_l693_693059


namespace maximum_value_x2_add_3xy_add_y2_l693_693830

-- Define the conditions
variables {x y : ℝ}

-- State the theorem
theorem maximum_value_x2_add_3xy_add_y2 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : 3 * x^2 - 2 * x * y + 5 * y^2 = 12) :
  ∃ e f g h : ℕ,
    x^2 + 3 * x * y + y^2 = (1144 + 204 * Real.sqrt 15) / 91 ∧ e + f + g + h = 1454 :=
sorry

end maximum_value_x2_add_3xy_add_y2_l693_693830


namespace range_of_m_l693_693688

noncomputable def p (m : ℝ) : Prop := 
  abs (1 - m) / sqrt 2 < 1

noncomputable def q (m : ℝ) : Prop := 
  0 < m ∧ m < 4 ∧ ((m > 0) ∧ (f 0 := m * (0:ℝ)^2 - (0:ℝ) + m - 4 < 0)) ∨ 
  ((m < 0) ∧ (f 0 := m * (0:ℝ)^2 - (0:ℝ) + m - 4 > 0))

theorem range_of_m (m : ℝ) (hp : ¬p m) (hpq : p m ∨ q m) : 
  sqrt 2 + 1 ≤ m ∧ m < 4 := 
sorry

end range_of_m_l693_693688


namespace sum_x_coordinates_of_system_l693_693257

theorem sum_x_coordinates_of_system :
  let f (x : ℝ) := |x^2 - 8 * x + 15|
  let g (x : ℝ) := 8 - x
  let solutions := {x : ℝ | f x = g x}
  (∑ x in solutions, x) = 7 :=
by {
  let f (x : ℝ) := |x^2 - 8 * x + 15|,
  let g (x : ℝ) := 8 - x,
  let solutions := {x : ℝ | f x = g x},
  calc
    (∑ x in solutions, x) = 7 : sorry
}

end sum_x_coordinates_of_system_l693_693257


namespace base_eight_to_base_ten_l693_693881

theorem base_eight_to_base_ten : ∃ n : ℕ, 47 = 4 * 8 + 7 ∧ n = 39 :=
by
  sorry

end base_eight_to_base_ten_l693_693881


namespace characterize_functions_l693_693654

def meets_condition (f : ℚ → ℤ) : Prop :=
  ∀ (x : ℚ) (a : ℤ) (b : ℕ), 0 < b → f ((f x + a) / b) = f ((x + a) / b)

theorem characterize_functions (f : ℚ → ℤ) :
  meets_condition f →
  (∃ c : ℤ, ∀ x : ℚ, f x = c) ∨ (f = λ x : ℚ, rat.floor x) ∨ (f = λ x : ℚ, rat.ceil x) :=
sorry

end characterize_functions_l693_693654


namespace solution_l693_693792

noncomputable def problem (a b c x y z : ℝ) :=
  11 * x + b * y + c * z = 0 ∧
  a * x + 19 * y + c * z = 0 ∧
  a * x + b * y + 37 * z = 0 ∧
  a ≠ 11 ∧
  x ≠ 0

theorem solution (a b c x y z : ℝ) (h : problem a b c x y z) :
  (a / (a - 11)) + (b / (b - 19)) + (c / (c - 37)) = 1 :=
sorry

end solution_l693_693792


namespace simplify_fraction_l693_693455

theorem simplify_fraction : ∃ (a b : ℕ), a = 90 ∧ b = 150 ∧ (90:ℚ) / (150:ℚ) = (3:ℚ) / (5:ℚ) :=
by {
  use 90,
  use 150,
  split,
  refl,
  split,
  refl,
  sorry,
}

end simplify_fraction_l693_693455


namespace smallest_draw_probability_is_white_l693_693354

theorem smallest_draw_probability_is_white :
  let total_balls := 9 + 5 + 2 in
  let red_ball_prob := 9 / total_balls in
  let black_ball_prob := 5 / total_balls in
  let white_ball_prob := 2 / total_balls in
  white_ball_prob < red_ball_prob ∧ white_ball_prob < black_ball_prob :=
by
  sorry

end smallest_draw_probability_is_white_l693_693354


namespace even_tens_digit_n_sq_l693_693261

theorem even_tens_digit_n_sq (s : Finset ℕ) :
  (s = Finset.range 100 \ Finset.range 11).map (Finset.EmbeddingSubtype σ => σ.val + 100) →
  (Finset.filter
     (λ n : ℕ =>
       let tens_digit := ((n^2 / 10) % 10)
       tens_digit % 2 = 0) s).card = 60 := 
by
  intros hs
  unfold Finset.EmbeddingSubtype Finset.range Finset.filter
  sorry

end even_tens_digit_n_sq_l693_693261


namespace cos_17pi_over_4_eq_sqrt2_over_2_l693_693651

theorem cos_17pi_over_4_eq_sqrt2_over_2 : Real.cos (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_17pi_over_4_eq_sqrt2_over_2_l693_693651


namespace quadratic_complete_square_l693_693236

theorem quadratic_complete_square :
  ∀ x : ℝ, x^2 - 4 * x + 5 = (x - 2)^2 + 1 :=
by
  intro x
  sorry

end quadratic_complete_square_l693_693236


namespace root_inequality_l693_693685

noncomputable def f (a x : ℝ) : ℝ := 2 * x + 1 - Real.exp (a * x)

theorem root_inequality (a x1 x2 : ℝ) (h_distinct_roots : x1 ≠ x2) (h_roots : f a x1 = 1 ∧ f a x2 = 1) :
  x1 + x2 > 2 / a :=
  sorry

end root_inequality_l693_693685


namespace congruent_triangles_l693_693137

theorem congruent_triangles
  {A B C A1 B1 C1 A2 B2 C2 : Type*}
  [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
  [LinearOrderedField A1] [LinearOrderedField A2]
  [LineSegment AB BC CA A1B1 B1C1 C1A1 A2B2 B2C2 C2A2]
  (h1 : C1•A1 ⊥ BC) 
  (h2 : A1•B1 ⊥ CA) 
  (h3 : B1•C1 ⊥ AB) 
  (h4 : B2•A2 ⊥ BC) 
  (h5 : C2•B2 ⊥ CA) 
  (h6 : A2•C2 ⊥ AB) : 
  A1B1C1 ≈ A2B2C2 :=
sorry

end congruent_triangles_l693_693137


namespace largest_prime_factor_l693_693974

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693974


namespace circle_config_PH_length_l693_693598

theorem circle_config_PH_length (O A B C P D H : Point) (r : ℝ) (h : Circle O r) (h1 : diameter A B O r) 
(angle_COA : ∠COA = 60) (h2 : line_through A B) (h3 : extend_to P A B) 
(h4 : BP = 1 / 2 * BO) (h5 : intersects_semicircle C P D) 
(h6 : perpendicular_from_P A P intersects_extension_AD_at H) :
  length PH = 2 * sqrt(3) / 3 := 
sorry

end circle_config_PH_length_l693_693598


namespace tangent_line_eq_f_gt_2x_cubed_max_k_l693_693715

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem tangent_line_eq (x : ℝ) : y = 2 * x :=
sorry

theorem f_gt_2x_cubed (x : ℝ) (h : 0 < x ∧ x < 1) : f(x) > 2 * (x + x^3 / 3) :=
sorry

theorem max_k (x : ℝ) (h : 0 < x ∧ x < 1) : 
  ∃ k, ∀ x ∈ Ioo 0 1, (k < 2 → f(x) > k * (x + x^3 / 3)) ∧ (k ≥ 2 → ¬(f(x) > k * (x + x^3 / 3))) :=
sorry

end tangent_line_eq_f_gt_2x_cubed_max_k_l693_693715


namespace right_isosceles_triangle_areas_l693_693820

theorem right_isosceles_triangle_areas :
  let A := 25 / 2
  let B := 144 / 2
  let C := 169 / 2
  A + B = C :=
by
  dsimp [A, B, C]
  rfl

end right_isosceles_triangle_areas_l693_693820


namespace truck_gasoline_rate_l693_693609

theorem truck_gasoline_rate (gas_initial gas_final : ℕ) (dist_supermarket dist_farm_turn dist_farm_final : ℕ) 
    (total_miles gas_used : ℕ) : 
  gas_initial = 12 →
  gas_final = 2 →
  dist_supermarket = 10 →
  dist_farm_turn = 4 →
  dist_farm_final = 6 →
  total_miles = dist_supermarket + dist_farm_turn + dist_farm_final →
  gas_used = gas_initial - gas_final →
  total_miles / gas_used = 2 :=
by sorry

end truck_gasoline_rate_l693_693609


namespace problem_condition_sufficient_not_necessary_problem_condition_not_necessary_l693_693395

variable {R : Type*} [Field R] [DecidableEq R] [Zero R] [One R] [Add R] [Mul R] [HasPow R ℕ]

-- Definitions for geometric sequence
def is_geometric (a : ℕ → R) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i < n → a (i + 1) = a i * a 1 / a 0

-- Definitions for condition q
def condition_q (a : ℕ → R) (n : ℕ) : Prop :=
  (∑ i in Finset.range (n - 1), a i ^ 2) * (∑ i in Finset.range (n - 1), a (i + 1) ^ 2) = 
  (∑ i in Finset.range (n - 1), a i * a (i + 1)) ^ 2

theorem problem_condition_sufficient_not_necessary 
  (a : ℕ → R) (n : ℕ) (h_geometric : is_geometric a n) (h_n : n ≥ 3) : 
  condition_q a n := sorry

-- Conversely, we assert that condition_q does not necessarily imply is_geometric
theorem problem_condition_not_necessary 
  (a : ℕ → R) (n : ℕ) (h_q : condition_q a n) (h_n : n ≥ 3) : 
  ¬ is_geometric a n := sorry

end problem_condition_sufficient_not_necessary_problem_condition_not_necessary_l693_693395


namespace largest_prime_factor_of_fact_sum_is_7_l693_693970

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693970


namespace perpendicular_vectors_m_val_l693_693723

theorem perpendicular_vectors_m_val (m : ℝ) 
  (a : ℝ × ℝ := (-1, 2)) 
  (b : ℝ × ℝ := (m, 1)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  m = 2 := 
by 
  sorry

end perpendicular_vectors_m_val_l693_693723


namespace largest_prime_factor_7fac_8fac_l693_693936

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693936


namespace quadrilateral_area_l693_693425

open Real

/-- Given a quadrilateral ABCD with the following properties:
    - ∠ABC = 90°
    - ∠ACD = 90°
    - AC = 24
    - CD = 18
    - Diagonals AC and BD intersect at point E
    - AE = 6
  
    Prove that the area of quadrilateral ABCD is 378. -/
theorem quadrilateral_area 
  (A B C D E : Point)
  (h1 : ∠ B A C = 90)
  (h2 : ∠ A C D = 90)
  (h3 : dist A C = 24)
  (h4 : dist C D = 18)
  (h5 : E ∈ lineSegment A C)
  (h6 : E ∈ lineSegment B D)
  (h7 : dist A E = 6) :
  areaQuadrilateral A B C D = 378 :=
sorry

end quadrilateral_area_l693_693425


namespace mass_percentage_Cl_in_CCl4_l693_693250

noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_Cl : ℝ := 35.45

noncomputable def molar_mass_CCl4 : ℝ := atomic_mass_C + 4 * atomic_mass_Cl
noncomputable def total_mass_Cl_in_CCl4 : ℝ := 4 * atomic_mass_Cl

theorem mass_percentage_Cl_in_CCl4 : (total_mass_Cl_in_CCl4 / molar_mass_CCl4) * 100 ≈ 92.23 := by
  sorry

end mass_percentage_Cl_in_CCl4_l693_693250


namespace customer_wants_score_of_eggs_l693_693551

def Score := 20
def Dozen := 12

def options (n : Nat) : Prop :=
  n = Score ∨ n = 2 * Score ∨ n = 2 * Dozen ∨ n = 3 * Score

theorem customer_wants_score_of_eggs : 
  ∃ n, options n ∧ n = Score := 
by
  exists Score
  constructor
  apply Or.inl
  rfl
  rfl

end customer_wants_score_of_eggs_l693_693551


namespace simplify_fraction_l693_693434

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end simplify_fraction_l693_693434


namespace total_number_of_animals_l693_693362

variables (initial_elephants initial_hippos : ℕ)
variables (female_elephants : ℕ) (newborn_elephants : ℕ)
variables (female_hippos : ℕ) (newborn_hippos : ℕ)
variables (surviving_elephants : ℕ) (surviving_hippos : ℕ)

def init_conditions : Prop :=
  initial_elephants = 20 ∧
  initial_hippos = 35 ∧ 
  female_elephants = initial_elephants / 2 ∧ 
  newborn_elephants = female_elephants * 3 ∧ 
  female_hippos = initial_hippos * 5 / 7 ∧
  newborn_hippos = female_hippos * 5 ∧
  surviving_elephants = newborn_elephants * 80 / 100 ∧
  surviving_hippos = newborn_hippos * 70 / 100

theorem total_number_of_animals (h : init_conditions) :
  (initial_elephants + surviving_elephants + initial_hippos + surviving_hippos).nat_abs = 166 :=
by {
  sorry
}

end total_number_of_animals_l693_693362


namespace perfect_square_expression_l693_693540

theorem perfect_square_expression (x y z : ℤ) :
    9 * (x^2 + y^2 + z^2)^2 - 8 * (x + y + z) * (x^3 + y^3 + z^3 - 3 * x * y * z) =
      ((x + y + z)^2 - 6 * (x * y + y * z + z * x))^2 := 
by 
  sorry

end perfect_square_expression_l693_693540


namespace transformed_function_l693_693507

noncomputable def original_function (x : ℝ) : ℝ := -2 * x^2

theorem transformed_function :
  ∃ h : ℝ → ℝ, ( ∀ x : ℝ, h x = -2 * (x + 3)^2 + 2 ) ∧
               ( ∀ x : ℝ, h x = -2 * x^2 - 12 * x - 16 ) := 
by
  exists ( λ x, -2 * (x + 3)^2 + 2 )
  constructor
  · intro x
    rfl
  · intro x
    sorry

end transformed_function_l693_693507


namespace frequencies_first_class_confidence_difference_quality_l693_693105

theorem frequencies_first_class (a b c d n : ℕ) (Ha : a = 150) (Hb : b = 50) 
                                (Hc : c = 120) (Hd : d = 80) (Hn : n = 400) 
                                (totalA : a + b = 200) 
                                (totalB : c + d = 200) :
  (a / (a + b) = 3 / 4) ∧ (c / (c + d) = 3 / 5) := by
sorry

theorem confidence_difference_quality (a b c d n : ℕ) (Ha : a = 150)
                                       (Hb : b = 50) (Hc : c = 120)
                                       (Hd : d = 80) (Hn : n = 400)
                                       (total : n = 400)
                                       (first_class_total : a + c = 270)
                                       (second_class_total : b + d = 130) :
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  k_squared > 6.635 := by
sorry

end frequencies_first_class_confidence_difference_quality_l693_693105


namespace pow_mod_26_l693_693519

theorem pow_mod_26 (a b n : ℕ) (hn : n = 2023) (h₁ : a = 17) (h₂ : b = 26) :
  a ^ n % b = 7 := by
  sorry

end pow_mod_26_l693_693519


namespace combined_ages_multiple_of_50_in_28_years_l693_693840

theorem combined_ages_multiple_of_50_in_28_years :
  ∀ (Hurley Richard Kate : ℕ),
  (Hurley = 14) →
  (Richard = Hurley's age + 20) →
  (Kate = Richard's age - 10) →
  ∃ t : ℕ, ((Hurley + t) + (Richard + t) + (Kate + t)) % 50 = 0 ∧ t = 28 ∧
      Richard + t = 62 ∧ Hurley + t = 42 ∧ Kate + t = 52 :=
by
  sorry

end combined_ages_multiple_of_50_in_28_years_l693_693840


namespace simplify_fraction_l693_693438

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end simplify_fraction_l693_693438


namespace largest_prime_factor_of_7fact_8fact_l693_693902

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693902


namespace simplify_cube_root_l693_693043

theorem simplify_cube_root (a : ℝ) (h : 0 ≤ a) : (a * a^(1/2))^(1/3) = a^(1/2) :=
sorry

end simplify_cube_root_l693_693043


namespace range_of_a_l693_693272

theorem range_of_a {a : ℝ} (h : ∀ x ∈ set.Icc 1 4, abs (x + 4/x - a) + a ≤ 5) : a ∈ set.Iic (9 / 2) :=
sorry

end range_of_a_l693_693272


namespace point_on_x_axis_l693_693337

theorem point_on_x_axis (a : ℝ) (h : (1, a + 1).snd = 0) : a = -1 :=
by
  sorry

end point_on_x_axis_l693_693337


namespace team_A_match_win_probability_l693_693361

def probability_A_game_win : ℚ := 2/3

theorem team_A_match_win_probability :
  let P_A := probability_A_game_win,
      P_match := P_A * P_A + 2 * P_A * (1 - P_A) * P_A
  in P_match = 20 / 27 := by
  sorry

end team_A_match_win_probability_l693_693361


namespace sin_eq_cos_then_alpha_eq_62_degrees_l693_693330

-- Define the conditions
def angle := 28
def alpha := 62

-- Theorem statement
theorem sin_eq_cos_then_alpha_eq_62_degrees (h : sin (angle: ℝ * Real.pi / 180) = cos (alpha: ℝ * Real.pi / 180)) : alpha = 62 :=
by
  sorry

end sin_eq_cos_then_alpha_eq_62_degrees_l693_693330


namespace limit_difference_quotient_l693_693271

section
variable (f : ℝ → ℝ) (h : ∀ x, f(x) = 1 / x)

theorem limit_difference_quotient :
  (lim (fun (Δx : ℝ) => (f(2 + Δx) - f 2) / Δx) (𝓝 0)) = -1 / 4 :=
by 
  sorry
end

end limit_difference_quotient_l693_693271


namespace min_sum_a_l693_693270

-- Define the conditions and the proof statement in Lean
theorem min_sum_a (a : ℕ → ℕ) (H : ∀ k, 2 ≤ k ∧ k ≤ 8 → (a k = a (k - 1) + 1 ∨ a k = a (k + 1) - 1) → ∃! i, 2 ≤ i ∧ i ≤ 8 ∧ (a i = a (i - 1) + 1 ∨ a i = a (i + 1) -1)) 
  (Ha1 : a 1 = 6) (Ha9 : a 9 = 9) 
  (a_pos : ∀ i, 1 ≤ i ∧ i ≤ 9 → a i > 0) :
  (∑ i in finset.range 9, a (i + 1)) = 31 :=
begin
  sorry
end

end min_sum_a_l693_693270


namespace impossibility_of_arrangement_l693_693154

-- Definitions based on identified conditions in the problem
def isValidArrangement (arr : List ℕ) : Prop :=
  arr.length = 300 ∧
  (∀ i, i < 300 - 1 → arr.get i = |arr.get (i - 1) - arr.get (i + 1)|) ∧
  (arr.all (λ x => x > 0))

theorem impossibility_of_arrangement :
  ¬ (∃ arr : List ℕ, isValidArrangement arr) :=
sorry

end impossibility_of_arrangement_l693_693154


namespace sequence_general_term_sum_general_formula_l693_693640

noncomputable def harmonic_mean {n : ℕ} (a : ℕ → ℝ) : ℝ :=
  n / (∑ i in finset.range n, a i)

noncomputable def sequence_a (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, harmonic_mean a = 1 / (2 * n + 1)

noncomputable def sequence_d (a : ℕ → ℝ) (d : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, d n = 2^n * a n

noncomputable def sum_T (d : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T n = ∑ i in finset.range n, d i

theorem sequence_general_term (a : ℕ → ℝ) :
  sequence_a a → ∀ n : ℕ, a n = 4 * n - 1 :=
by
  sorry

theorem sum_general_formula (a d : ℕ → ℝ) (T : ℕ → ℝ) :
  sequence_a a → sequence_d a d → sum_T d T →
  ∀ n : ℕ, T n = (4 * n - 5) * 2^(n + 1) + 10 :=
by
  sorry

end sequence_general_term_sum_general_formula_l693_693640


namespace find_a_l693_693712

def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then 3^x + 1 else x^2 + a * x

theorem find_a (a : ℝ) :
  (f (f 0 a) a) = 6 → a = 1 :=
by
  intro h
  have h1 : f 0 a = 2 := by
    simp [f]
  have h2 : f 2 a = 4 + 2 * a := by
    simp [f]
  have h3 : 4 + 2 * a = 6 := by
    rw [h1] at h
    rw [h2] at h
    exact h
  have h4 : 2 * a = 2 := by
    exact eq_of_add_eq_add_right h3
  exact eq_of_mul_eq_mul_left (by norm_num) h4

end find_a_l693_693712


namespace sandwich_interval_is_40_minutes_l693_693049

-- Definitions
def total_jalapeno_peppers : ℕ := 48
def jalapeno_per_sandwich : ℕ := 4
def hours_in_a_day : ℕ := 8
def minutes_in_an_hour : ℕ := 60

-- Theorem
theorem sandwich_interval_is_40_minutes :
  (total_jalapeno_peppers / jalapeno_per_sandwich) = 12 ∧
  (hours_in_a_day * minutes_in_an_hour) = 480 ∧
  (480 / 12) = 40 :=
by
  have h1 : (total_jalapeno_peppers / jalapeno_per_sandwich) = 12 := by rw [total_jalapeno_peppers, jalapeno_per_sandwich]; norm_num
  have h2 : (hours_in_a_day * minutes_in_an_hour) = 480 := by rw [hours_in_a_day, minutes_in_an_hour]; norm_num
  have h3 : (480 / 12) = 40 := by norm_num
  exact ⟨h1, h2, h3⟩
sorry

end sandwich_interval_is_40_minutes_l693_693049


namespace problem_sum_divisors_l693_693267

open Nat

noncomputable def sum_odd_divisors_lt_sum_even_divisors (n : ℕ) (k : ℕ) : Prop :=
  n = 2^2 * 3 * 5^2 * k ∧
  (Σ d in (Finset.filter (λ d, d ≠ n) (Finset.filter (λ d, odd d) (nat.divisors n))), d) <
  (Σ d in (Finset.filter (λ d, d ≠ n) (Finset.filter (λ d, even d) (nat.divisors n))), d)

theorem problem_sum_divisors (n k : ℕ) (h1 : n = 2^2 * 3 * 5^2 * k)
  (h2 : 300 ∣ n) :
  sum_odd_divisors_lt_sum_even_divisors n k :=
by
  sorry

end problem_sum_divisors_l693_693267


namespace max_prism_volume_l693_693757

noncomputable def max_volume_prism (a h : ℝ) : ℝ :=
  (sqrt 3 / 4 * a^2) * h

theorem max_prism_volume :
  ∃ (a h : ℝ),
    2 * a * h + (sqrt 3 / 4) * a^2 = 30 ∧ 
    max_volume_prism a h = 6.90 :=
by
  sorry

end max_prism_volume_l693_693757


namespace integer_solutions_range_l693_693319

theorem integer_solutions_range (a : ℝ) :
  (∀ x : ℤ, x^2 - x + a - a^2 < 0 → x + 2 * a > 1) ↔ 1 < a ∧ a ≤ 2 := sorry

end integer_solutions_range_l693_693319


namespace factorial_divisible_by_3_power_l693_693018

theorem factorial_divisible_by_3_power :
  (\sum i in List.range 7, (2018 / 3^i).nat_floor) = 1004 := 
by
  sorry

end factorial_divisible_by_3_power_l693_693018


namespace impossibility_of_arrangement_l693_693153

-- Definitions based on identified conditions in the problem
def isValidArrangement (arr : List ℕ) : Prop :=
  arr.length = 300 ∧
  (∀ i, i < 300 - 1 → arr.get i = |arr.get (i - 1) - arr.get (i + 1)|) ∧
  (arr.all (λ x => x > 0))

theorem impossibility_of_arrangement :
  ¬ (∃ arr : List ℕ, isValidArrangement arr) :=
sorry

end impossibility_of_arrangement_l693_693153


namespace smallest_n_mod_l693_693124

theorem smallest_n_mod :
  ∃ n : ℕ, (23 * n ≡ 5678 [MOD 11]) ∧ (∀ m : ℕ, (23 * m ≡ 5678 [MOD 11]) → (0 < n) ∧ (n ≤ m)) :=
  by
  sorry

end smallest_n_mod_l693_693124


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693999

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693999


namespace part1_part2_l693_693717

theorem part1 (x : ℝ) : |x + 3| - 2 * x - 1 < 0 → 2 < x :=
by sorry

theorem part2 (m : ℝ) : (m > 0) →
  (∃ x : ℝ, |x - m| + |x + 1/m| = 2) → m = 1 :=
by sorry

end part1_part2_l693_693717


namespace exists_cell_with_both_pieces_l693_693469

-- Definition of the grid and placements
def Grid (n : ℕ) := array (array bool n) n -- A n x n grid with bool indicating black cells (true) or not (false).

structure Conditions (n : ℕ) (grid : Grid n) :=
  (odd_black_cells_in_rows : ∀ i < n, ∃ k, (array.filter (λ x, grid[i][x]) (finRange n)).length = 2*k + 1)
  (odd_black_cells_in_cols : ∀ j < n, ∃ k, (array.filter (λ x, grid[x][j]) (finRange n)).length = 2*k + 1)
  (distinct_red_columns : array (option (fin n)) n) -- Red pieces presence in different columns
  (distinct_blue_rows : array (option (fin n)) n) -- Blue pieces presence in different rows

-- The theorem statement
theorem exists_cell_with_both_pieces (n : ℕ) (grid : Grid n) (cond : Conditions n grid) :
  ∃ i j, cond.distinct_red_columns[i] = some j ∧ cond.distinct_blue_rows[j] = some i :=
sorry

end exists_cell_with_both_pieces_l693_693469


namespace adrian_greater_than_natasha_probability_l693_693411

noncomputable def probability_adrian_greater_natasha : ℝ :=
let x_dist := uniform_real_dist 0 1524 in
let y_dist := uniform_real_dist 0 3048 in
measure_space.measure.probability (λ (y x : ℝ), y > x) x_dist y_dist

theorem adrian_greater_than_natasha_probability :
  probability_adrian_greater_natasha = 3 / 4 :=
sorry

end adrian_greater_than_natasha_probability_l693_693411


namespace contrapositive_of_basis_l693_693194

-- Definitions required for the problem statement
variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

-- Define the set of vectors
def basis1 := {a, b, c}
def basis2 := {a + b, b + c, c + a}

-- Statement of the problem as a Lean theorem
theorem contrapositive_of_basis :
  ¬(LinearIndependent ℝ basis2 ∧ span ℝ basis2 = ⊤) →
  ¬(LinearIndependent ℝ basis1 ∧ span ℝ basis1 = ⊤) :=
sorry

end contrapositive_of_basis_l693_693194


namespace ellipse_equation_l693_693316

noncomputable def c : ℝ := real.sqrt (2^2 + (2 * real.sqrt 3)^2)

theorem ellipse_equation 
  (h_eq : ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1))
  (semi_major_hyp : ∃ a : ℝ, a = 2)
  (semi_minor_hyp : ∃ b : ℝ, b = real.sqrt 12)
  (focal_distance : ∃ c : ℝ, c = real.sqrt (2^2 + (2 * real.sqrt 3)^2))
  (semi_major_ellipse : ∃ a' : ℝ, a' = 4)
  (semi_minor_ellipse : ∃ b' : ℝ, b' = real.sqrt 12) :
  ∀ x y : ℝ, (x^2 / 16 + y^2 / 12 = 1) :=
by
  intros x y
  sorry

end ellipse_equation_l693_693316


namespace product_quality_difference_l693_693086

variable (n a b c d : ℕ)
variable (P_K_2 : ℝ → ℝ)

def first_class_freq_A := a / (a + b : ℕ)
def first_class_freq_B := c / (c + d : ℕ)

def K2 := (n : ℝ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem product_quality_difference
  (ha : a = 150) (hb : b = 50) 
  (hc : c = 120) (hd : d = 80)
  (hn : n = 400)
  (hK : P_K_2 0.010 = 6.635) : 
  first_class_freq_A a b = 3 / 4 ∧
  first_class_freq_B c d = 3 / 5 ∧
  K2 n a b c d > P_K_2 0.010 :=
by {
  sorry
}

end product_quality_difference_l693_693086


namespace simplify_fraction_l693_693443

theorem simplify_fraction (num denom : ℕ) (h_num : num = 90) (h_denom : denom = 150) : 
  num / denom = 3 / 5 := by
  rw [h_num, h_denom]
  norm_num
  sorry

end simplify_fraction_l693_693443


namespace largest_prime_factor_l693_693983

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693983


namespace sixty_three_times_fifty_seven_l693_693624

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 := by
  sorry

end sixty_three_times_fifty_seven_l693_693624


namespace sam_memorized_digits_l693_693822

theorem sam_memorized_digits (c s m : ℕ) 
  (h1 : s = c + 6) 
  (h2 : m = 6 * c)
  (h3 : m = 24) : 
  s = 10 :=
by
  sorry

end sam_memorized_digits_l693_693822


namespace prob_exactly_two_approve_l693_693177

def voter_approval_pmf (p : ℝ) : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.binomial 4 p

theorem prob_exactly_two_approve (p : ℝ) (h_p : p = 0.6) :
  voter_approval_pmf p 2 = 0.3456 := 
by
  have hp : p = 0.6 := h_p
  sorry

end prob_exactly_two_approve_l693_693177


namespace triangle_RS_l693_693019

theorem triangle_RS (A B C H R S : Point)
  (hCH : altitude CH A B C)
  (hR : tangent_point R (triangle_incircle A C H) CH)
  (hS : tangent_point S (triangle_incircle B C H) CH)
  (hAB : distance A B = 2021)
  (hAC : distance A C = 2020)
  (hBC : distance B C = 2019) :
  let RS := distance R S in
  ∃ (m n : ℕ), nat.gcd m n = 1 ∧ RS = m / n ∧ m + n = 6011 :=
by
  sorry

end triangle_RS_l693_693019


namespace ceil_neg_3_7_l693_693221

-- Define the ceiling function in Lean
def ceil (x : ℝ) : ℤ := int.ceil x

-- A predicate to represent the statement we want to prove
theorem ceil_neg_3_7 : ceil (-3.7) = -3 := by
  -- Provided conditions
  have h1 : ceil (-3.7) = int.ceil (-3.7) := rfl
  have h2 : int.ceil (-3.7) = -3 := by
    -- Lean's int.ceil function returns the smallest integer greater or equal to the input
    sorry  -- proof goes here

  -- The main statement
  exact h2

end ceil_neg_3_7_l693_693221


namespace original_digit_sum_six_and_product_is_1008_l693_693585

theorem original_digit_sum_six_and_product_is_1008 (x : ℕ) :
  (2 ∣ x / 10) → (4 ∣ x / 10) → 
  (x % 10 + (x / 10) = 6) →
  ((x % 10) * 10 + (x / 10)) * ((x / 10) * 10 + (x % 10)) = 1008 →
  x = 42 ∨ x = 24 :=
by
  intro h1 h2 h3 h4
  sorry


end original_digit_sum_six_and_product_is_1008_l693_693585


namespace find_f_expression_l693_693274

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_linear : ∃ (k b : ℝ), k ≠ 0 ∧ ∀ x, f(x) = k * x + b
axiom f_composition : ∀ x, f(f(x)) = 4 * x + 9

theorem find_f_expression : ∃ k b : ℝ, (k = 2 ∧ b = 3) ∨ (k = -2 ∧ b = -9) :=
sorry

end find_f_expression_l693_693274


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693917

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693917


namespace decimal_units_digit_sequence_is_rational_l693_693008

noncomputable def unit_digit (n : ℕ) : ℕ := n % 10

noncomputable def a_n (n : ℕ) : ℕ :=
  unit_digit (Finset.sum (Finset.range n) (λ k, (k + 1) ^ 2))

theorem decimal_units_digit_sequence_is_rational : 
  (∃ period : ℕ, ∀ n, a_n n = a_n (n + period)) → 
  (∃ p q : ℕ, (p/q : Real) = ∑ i in Finset.range 20, a_n i / 10^((i : ℕ) + 1)) := 
sorry

end decimal_units_digit_sequence_is_rational_l693_693008


namespace simplify_fraction_90_150_l693_693444

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end simplify_fraction_90_150_l693_693444


namespace kaleb_initial_games_l693_693776

-- Let n be the number of games Kaleb started out with
def initial_games (n : ℕ) : Prop :=
  let sold_games := 46
  let boxes := 6
  let games_per_box := 5
  n = sold_games + boxes * games_per_box

-- Now we state the theorem
theorem kaleb_initial_games : ∃ n, initial_games n ∧ n = 76 :=
  by sorry

end kaleb_initial_games_l693_693776


namespace W_sequence_palindrome_l693_693586

def is_palindrome (s : String) : Prop := s == s.reverse

def W (n : Nat) : String :=
  match n with
  | 0 => "a"
  | 1 => "b"
  | n + 2 => W (n) ++ W (n + 1)

theorem W_sequence_palindrome : ∀ n ≥ 1, is_palindrome (List.foldl (++) "" (List.map W (List.range (n + 1))))
  := sorry

end W_sequence_palindrome_l693_693586


namespace log_xyz_l693_693472

variables (t x y z : ℝ)

def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_xyz (h1 : log_base x t = 6) (h2 : log_base y t = 10) (h3 : log_base z t = 15) : log_base (x * y * z) t = 3 :=
by
  sorry

end log_xyz_l693_693472


namespace arrange_digits_l693_693334

-- Defining the number of boxes and digits
def boxes : ℕ := 5
def digits : Finset ℕ := {0, 1, 2, 3, 4}

-- The problem boils down to counting permutations of 5 unique digits in 5 boxes
noncomputable def count_ways : ℕ := (boxes * (4.factorial))

-- The main theorem stating the number of ways to arrange the digits
theorem arrange_digits: count_ways = 120 := by sorry

end arrange_digits_l693_693334


namespace f_t_minus_1_plus_f_t_neg_l693_693060

variable {t : ℝ}

-- We define the function f as given.
def f (x : ℝ) := x / (1 + x^2)

-- We state the theorem using the given conditions.
theorem f_t_minus_1_plus_f_t_neg (h_odd : ∀ x, f (-x) = -f x)
  (h_inc : ∀ x y, x < y → f x < f y)
  (h_t : 0 < t ∧ t < 1/2) :
  f (t - 1) + f t < 0 :=
by
  sorry

end f_t_minus_1_plus_f_t_neg_l693_693060


namespace man_speed_is_correct_l693_693583

noncomputable def train_length : ℝ := 165
noncomputable def train_speed_kmph : ℝ := 60
noncomputable def time_seconds : ℝ := 9

-- Function to convert speed from kmph to m/s
noncomputable def kmph_to_mps (speed_kmph: ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Function to convert speed from m/s to kmph
noncomputable def mps_to_kmph (speed_mps: ℝ) : ℝ :=
  speed_mps * 3600 / 1000

-- The speed of the train in m/s
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- The relative speed of the train with respect to the man in m/s
noncomputable def relative_speed_mps : ℝ := train_length / time_seconds

-- The speed of the man in m/s
noncomputable def man_speed_mps : ℝ := relative_speed_mps - train_speed_mps

-- The speed of the man in kmph
noncomputable def man_speed_kmph : ℝ := mps_to_kmph man_speed_mps

-- The statement to be proved
theorem man_speed_is_correct : man_speed_kmph = 5.976 := 
sorry

end man_speed_is_correct_l693_693583


namespace eq_of_condition_l693_693850

theorem eq_of_condition (x : ℝ) (h : x + 1/x = 3) : x^7 - 5 * x^5 + 3 * x^3 = 126 * x - 48 :=
begin
  sorry
end

end eq_of_condition_l693_693850


namespace mean_exercise_days_rounded_l693_693409

/-- Given the number of members and the number of days they exercised,
    prove that the mean number of days of exercise last month, rounded
    to the nearest hundredth, is 9.89. -/
theorem mean_exercise_days_rounded :
  let members_days := [(2, 2), (4, 5), (6, 8), (3, 10), (5, 12), (7, 15)] in
  let total_days := List.sum (members_days.map (λ (members, days) => members * days)) in
  let total_members := List.sum (members_days.map (λ (members, _) => members)) in
  (total_days.toReal / total_members.toReal).round * 100 / 100 = 9.89 :=
by
  sorry

end mean_exercise_days_rounded_l693_693409


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693927

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693927


namespace general_formula_for_seq_diff_constant_seq_value_Mn_less_than_5_over_2_l693_693023

open Nat

-- Definitions
def seq_a (a : ℝ) : ℕ → ℝ
| 0       => a
| (n + 1) => seq_a a n

def seq_b (a : ℝ) (c : ℝ) : ℕ → ℝ
| 0       => 1
| (n + 1) => (seq_a a n + seq_c a c n) / 2

def seq_c (a : ℝ) (b : ℝ) : ℕ → ℝ
| 0       => 3
| (n + 1) => (seq_a a n + seq_b a b n) / 2

def is_constant_seq {α : Type*} [Field α] [UniformSpace α] (f : ℕ → α) : Prop :=
  ∀ n m, f n = f m

def S_n (a : ℝ) (n : ℕ) : ℝ :=
  (range (n + 1)).sum (λ i, seq_b a (seq_c a (seq_b a (seq_c a (seq_b a (seq_c a (seq_b a 1))))) i))

def T_n (a : ℝ) (n : ℕ) : ℝ :=
  (range n).sum (λ i, seq_c a (seq_b a 1) i)

def M_n (a : ℝ) (n : ℕ) : ℝ :=
  2 * S_n a (n + 1) - T_n a n

-- Theorem (1)
theorem general_formula_for_seq_diff (a : ℝ) :
  ∀ n, (seq_c a (seq_b a 1) n - seq_b a (seq_c a (seq_b a 1)) n) = 2 * (-1/2)^(n-1) := sorry

-- Theorem (2)
theorem constant_seq_value (a : ℝ) :
  is_constant_seq (seq_a a) →
  is_constant_seq (λ n, seq_b a (seq_c a (seq_b a 1) n) + seq_c a (seq_b a 1) n) →
  a = 2 := sorry

-- Theorem (3)
theorem Mn_less_than_5_over_2 (a : ℝ) (h_geom : ∀ n, seq_a a n = a^n) :
  (∀ n, M_n a n < 5/2) →
  (-1 < a ∧ a < 0 ∨ 0 < a ∧ a ≤ 1/3) := sorry

end general_formula_for_seq_diff_constant_seq_value_Mn_less_than_5_over_2_l693_693023


namespace contradictory_events_l693_693175

-- Definitions for the conditions
def event_a : Prop := at_least_one_hit
def event_b : Prop := three_consecutive_misses

-- Statement of the problem in Lean 4
theorem contradictory_events :
  (∃ x, x ∈ event_a) ↔ ¬ (∃ y, y ∈ event_b)
sorry

end contradictory_events_l693_693175


namespace problem_implies_statement_l693_693755

-- Definitions for the sets
variables (Flog Hak Grep Jinx : Type)

-- Conditions provided
axiom Flogs_are_Greps : ∀ x : Flog, x ∈ Grep
axiom Haks_are_Greps : ∀ x : Hak, x ∈ Grep
axiom Haks_are_Jinx : ∀ x : Hak, x ∈ Jinx
axiom No_Flogs_are_Jinx : ∀ x : Flog, x ∉ Jinx

-- Theorem to be proved
theorem problem_implies_statement :
  (∀ x : Jinx, x ∈ Hak) ∧ (∃ x : Jinx, x ∉ Flog) :=
sorry

end problem_implies_statement_l693_693755


namespace T_2022_eq_T_2019_l693_693780

-- Define what T_n represents
def T (n : ℕ) := { 
  triangles : Set (ℕ × ℕ × ℕ) //
  ∀ (a b c: ℕ), a ∈ triangles → b ∈ triangles → c ∈ triangles → 
  (a + b + c = n ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ (a, b, c) ∈ triangles) 
}

-- Prove that T 2022 = T 2019
theorem T_2022_eq_T_2019 : T 2022 = T 2019 :=
sorry

end T_2022_eq_T_2019_l693_693780


namespace sixty_three_times_fifty_seven_l693_693629

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 :=
by
  let a := 60
  let b := 3
  have h : (a + b) * (a - b) = a^2 - b^2 := by sorry
  have h1 : 63 = a + b := by rfl
  have h2 : 57 = a - b := by rfl
  calc
    63 * 57 = (a + b) * (a - b) : by rw [h1, h2]
    ... = a^2 - b^2 : by rw h
    ... = 60^2 - 3^2 : by rfl
    ... = 3600 - 9 : by sorry
    ... = 3591 : by norm_num

end sixty_three_times_fifty_seven_l693_693629


namespace total_pizza_slices_correct_l693_693028

-- Define the conditions
def num_pizzas : Nat := 3
def slices_per_first_two_pizzas : Nat := 8
def num_first_two_pizzas : Nat := 2
def slices_third_pizza : Nat := 12

-- Define the total slices based on conditions
def total_slices : Nat := slices_per_first_two_pizzas * num_first_two_pizzas + slices_third_pizza

-- The theorem to be proven
theorem total_pizza_slices_correct : total_slices = 28 := by
  sorry

end total_pizza_slices_correct_l693_693028


namespace predict_monthly_savings_l693_693567

noncomputable def sum_x_i := 80
noncomputable def sum_y_i := 20
noncomputable def sum_x_i_y_i := 184
noncomputable def sum_x_i_sq := 720
noncomputable def n := 10
noncomputable def x_bar := sum_x_i / n
noncomputable def y_bar := sum_y_i / n
noncomputable def b := (sum_x_i_y_i - n * x_bar * y_bar) / (sum_x_i_sq - n * x_bar^2)
noncomputable def a := y_bar - b * x_bar
noncomputable def regression_eqn(x: ℝ) := b * x + a

theorem predict_monthly_savings :
  regression_eqn 7 = 1.7 :=
by
  sorry

end predict_monthly_savings_l693_693567


namespace coefficient_of_term_is_const_factor_l693_693054

-- Defining the term -π * x * y^3 / 5
def term (x y : ℝ) : ℝ := - (π * x * y^3) / 5

-- Theorem statement to prove the coefficient of the term -π * x * y^3 / 5 is -π / 5
theorem coefficient_of_term_is_const_factor (x y : ℝ) : 
  coeff_of_term (term x y) = -π / 5 :=
by 
  -- Since we only need the statement, we add sorry to skip the proof.
  sorry

-- Define what it means to be the coefficient of the term
def coeff_of_term (t : ℝ) : ℝ :=
  -π / 5  -- Coefficient is -π / 5 as shown in the solution steps

end coefficient_of_term_is_const_factor_l693_693054


namespace total_doughnuts_l693_693809

-- Definitions used in the conditions
def boxes : ℕ := 4
def doughnuts_per_box : ℕ := 12

theorem total_doughnuts : boxes * doughnuts_per_box = 48 :=
by
  sorry

end total_doughnuts_l693_693809


namespace wall_length_l693_693531

theorem wall_length (side_mirror : ℝ) (width_wall : ℝ) (length_wall : ℝ) 
  (h_mirror: side_mirror = 18) 
  (h_width: width_wall = 32)
  (h_area: (side_mirror ^ 2) * 2 = width_wall * length_wall):
  length_wall = 20.25 := 
by 
  -- The following 'sorry' is a placeholder for the proof
  sorry

end wall_length_l693_693531


namespace journey_time_l693_693161

noncomputable def velocity_of_stream : ℝ := 4
noncomputable def speed_of_boat_in_still_water : ℝ := 14
noncomputable def distance_A_to_B : ℝ := 180
noncomputable def distance_B_to_C : ℝ := distance_A_to_B / 2
noncomputable def downstream_speed : ℝ := speed_of_boat_in_still_water + velocity_of_stream
noncomputable def upstream_speed : ℝ := speed_of_boat_in_still_water - velocity_of_stream

theorem journey_time : (distance_A_to_B / downstream_speed) + (distance_B_to_C / upstream_speed) = 19 := by
  sorry

end journey_time_l693_693161


namespace bert_made_1_dollar_l693_693604

def bert_earnings (selling_price tax_rate markup : ℝ) : ℝ :=
  selling_price - (tax_rate * selling_price) - (selling_price - markup)

theorem bert_made_1_dollar :
  bert_earnings 90 0.1 10 = 1 :=
by 
  sorry

end bert_made_1_dollar_l693_693604


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693889

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693889


namespace factorize_a_squared_plus_2a_l693_693239

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2 * a = a * (a + 2) :=
  sorry

end factorize_a_squared_plus_2a_l693_693239


namespace simplify_fraction_l693_693450

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end simplify_fraction_l693_693450


namespace original_number_is_85_l693_693188

theorem original_number_is_85
  (x : ℤ) (h_sum : 10 ≤ x ∧ x < 100) 
  (h_condition1 : (x / 10) + (x % 10) = 13)
  (h_condition2 : 10 * (x % 10) + (x / 10) = x - 27) :
  x = 85 :=
by
  sorry

end original_number_is_85_l693_693188


namespace greatest_i_l693_693386

theorem greatest_i (a₀ d : ℕ) (h_a₀ : 0 < a₀) (h_d : 0 < d) :
  (∃ i, ∑ k in Finset.range (i+1), (a₀ + k * d) = 2010 ∧ i = 29) :=
sorry

end greatest_i_l693_693386


namespace largest_possible_n_l693_693595

-- Define the initial conditions and arithmetic sequences
variables {a_n b_n: ℕ → ℕ}  {x y: ℕ}

-- Define the initial values and common differences
def arithmetic_sequences (a_n b_n: ℕ → ℕ) (x y: ℕ) :=
  (a_n 1 = 1) ∧ (b_n 1 = 1) ∧
  (∀ n, a_n (n + 1) = a_n n + x) ∧ 
  (∀ n, b_n (n + 1) = b_n n + y) ∧
  (1 < a_n 2) ∧ (a_n 2 ≤ b_n 2)

-- Define the product condition
def product_condition (a_n b_n: ℕ → ℕ) :=
  ∃ n, a_n n * b_n n = 1764

-- The main theorem to be proved
theorem largest_possible_n (a_n b_n: ℕ → ℕ) (x y: ℕ) 
  (h_seq: arithmetic_sequences a_n b_n x y)
  (h_prod: product_condition a_n b_n) :
  ∃ n, n = 44 ∧ 
  (∀ m, (m > n → ¬ product_condition (λ k, a_n k) (λ k, b_n k))) :=
sorry

end largest_possible_n_l693_693595


namespace value_of_trig_expression_l693_693684

theorem value_of_trig_expression (α : Real) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -3 :=
by 
  sorry

end value_of_trig_expression_l693_693684


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693989

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693989


namespace second_largest_consecutive_odd_195_l693_693494

theorem second_largest_consecutive_odd_195 :
  ∃ x : Int, (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 195 ∧ (x + 2) = 41 := by
  sorry

end second_largest_consecutive_odd_195_l693_693494


namespace linear_eq_representation_l693_693037

noncomputable def vector_equation (A B C x y : ℝ) : ℝ :=
  let a := A * (complex.i) + B * (complex.j)
  let r := x * (complex.i) + y * (complex.j)
  let m := -C
  inner r a = m

noncomputable def normal_form (A B C x y : ℝ) : ℝ :=
  let p := -C / Real.sqrt (A^2 + B^2)
  let cosϕ := A / Real.sqrt (A^2 + B^2)
  let sinϕ := B / Real.sqrt (A^2 + B^2)
  cosϕ * x + sinϕ * y = p

theorem linear_eq_representation (A B C x y : ℝ) : 
  vector_equation A B C x y = normal_form A B C x y :=
sorry

end linear_eq_representation_l693_693037


namespace unique_solution_a_exists_l693_693734

open Real

noncomputable def equation (a x : ℝ) :=
  4 * a^2 + 3 * x * log x + 3 * (log x)^2 = 13 * a * log x + a * x

theorem unique_solution_a_exists : 
  ∃! a : ℝ, ∃ x : ℝ, 0 < x ∧ equation a x :=
sorry

end unique_solution_a_exists_l693_693734


namespace solution_l693_693824

-- Define the problem.
def problem (CD : ℝ) (hexagon_side : ℝ) (CY : ℝ) (BY : ℝ) : Prop :=
  CD = 2 ∧ hexagon_side = 2 ∧ CY = 4 * CD ∧ BY = 9 * Real.sqrt 2 → BY = 9 * Real.sqrt 2

theorem solution : problem 2 2 8 (9 * Real.sqrt 2) :=
by
  -- Contextualize the given conditions and directly link to the desired proof.
  intro h
  sorry

end solution_l693_693824


namespace harold_had_8_dollars_l693_693638

variables (D E F G H : ℕ)

theorem harold_had_8_dollars (h1 : abs (D - E) = 15)
                            (h2 : abs (E - F) = 9)
                            (h3 : abs (F - G) = 7)
                            (h4 : abs (G - H) = 6)
                            (h5 : abs (H - D) = 13)
                            (h6 : D + E + F + G + H = 72) :
  H = 8 :=
sorry

end harold_had_8_dollars_l693_693638


namespace probability_of_correct_dial_l693_693777

def area_codes : Finset ℕ := {407, 410, 415}

def digit_arrangements : Finset (Finset ℕ) := 
  { {0, 1, 2, 3, 4} }

lemma count_arrangements : 
  (digit_arrangements.card + (digit_arrangements.card * digit_arrangements.card)) = 120 :=
by sorry

theorem probability_of_correct_dial : 
  (1 : ℚ) / (area_codes.card * 5!) = 1 / 360 :=
by 
  sorry

end probability_of_correct_dial_l693_693777


namespace probability_of_event_a_l693_693298

-- Given conditions and question
variables (a b : Prop)
variables (p : Prop → ℝ)

-- Given conditions
axiom p_a : p a = 4 / 5
axiom p_b : p b = 2 / 5
axiom p_a_and_b_given : p (a ∧ b) = 0.32
axiom independent_a_b : p (a ∧ b) = p a * p b

-- The proof statement we need to prove: p a = 0.8
theorem probability_of_event_a :
  p a = 0.8 :=
sorry

end probability_of_event_a_l693_693298


namespace unique_solution_l693_693247

def is_solution (x y p : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ p.prime ∧ (xy^3) / (x + y) = p

theorem unique_solution :
  ∀ (x y p : ℕ), is_solution x y p → (x, y, p) = (14, 2, 7) :=
by
  sorry

end unique_solution_l693_693247


namespace correct_statement_D_l693_693528

/-- Define what it means for the distance from a point to a line -/
def distance_from_point_to_line {α : Type*} [metric_space α] (p : α) (ℓ : linear_map ℝ ℝ α) : ℝ :=
inf {d | ∃ q : α, q ∈ ℓ.range ∧ dist p q = d}

/-- Statement D: The distance from a point outside a line to the perpendicular segment on that line is called the distance from the point to the line. -/
theorem correct_statement_D (α : Type*) [metric_space α] (p : α) (ℓ : linear_map ℝ ℝ α) (h : ∃ q : α, q ∈ ℓ.range ∧ q ≠ p) :
  distance_from_point_to_line p ℓ = dist p (orthogonal_projection ℓ.range p) :=
sorry

end correct_statement_D_l693_693528


namespace largest_prime_factor_7fac_8fac_l693_693943

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693943


namespace modulo_11_residue_l693_693610

theorem modulo_11_residue : 
  (341 + 6 * 50 + 4 * 156 + 3 * 12^2) % 11 = 4 := 
by
  sorry

end modulo_11_residue_l693_693610


namespace salesman_past_income_l693_693178

theorem salesman_past_income (W1 W2 W3 W4 W5 : ℝ) :
  let W6 := 586 in
  let W7 := 586 in
  let W8 := 586 in
  let W9 := 586 in
  let W10 := 586 in
  (W1 + W2 + W3 + W4 + W5) + (W6 + W7 + W8 + W9 + W10) = 5000 → 
  W1 + W2 + W3 + W4 + W5 = 2070 :=
by 
  intros
  rw [W6, W7, W8, W9, W10] at *
  sorry

end salesman_past_income_l693_693178


namespace solving_inequality_l693_693478

theorem solving_inequality (x : ℝ) : 
  (x > 2 ∨ x < -2 ∨ (-1 < x ∧ x < 1)) ↔ ((x^2 - 4) / (x^2 - 1) > 0) :=
by 
  sorry

end solving_inequality_l693_693478


namespace sixty_three_times_fifty_seven_l693_693626

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 :=
by
  let a := 60
  let b := 3
  have h : (a + b) * (a - b) = a^2 - b^2 := by sorry
  have h1 : 63 = a + b := by rfl
  have h2 : 57 = a - b := by rfl
  calc
    63 * 57 = (a + b) * (a - b) : by rw [h1, h2]
    ... = a^2 - b^2 : by rw h
    ... = 60^2 - 3^2 : by rfl
    ... = 3600 - 9 : by sorry
    ... = 3591 : by norm_num

end sixty_three_times_fifty_seven_l693_693626


namespace fraction_division_l693_693123

theorem fraction_division (a b c d e : ℚ)
  (h1 : a = 3 / 7)
  (h2 : b = 1 / 3)
  (h3 : d = 2 / 5)
  (h4 : c = a + b)
  (h5 : e = c / d):
  e = 40 / 21 := by
  sorry

end fraction_division_l693_693123


namespace BoxMullerTransform_normal_independent_l693_693398

open MeasureTheory ProbabilityTheory

/-- Lean statement only defining conditions -/

noncomputable def BoxMullerTransform (U V : ℝ) : ℝ × ℝ := 
  (sqrt (- 2 * log V) * cos (2 * π * U), sqrt (- 2 * log V) * sin (2 * π * U))

/-- Main theorem to prove X and Y are independent and normally distributed -/

theorem BoxMullerTransform_normal_independent 
  (U V : MeasureTheory.MeasureSpace ℝ) 
  (hU : ProbabilityTheory.Independent U) 
  (hV : ProbabilityTheory.Independent V) 
  (hU_dist : MeasureTheory.ProbabilityMeasure U)
  (hV_dist : MeasureTheory.ProbabilityMeasure V)
  (hU_uniform : ∀ a b, ∫ x in set.Ioo a b, U.density x = (b - a))
  (hV_uniform : ∀ a b, ∫ x in set.Ioo a b, V.density x = (b - a)) :
  let (X, Y) := BoxMullerTransform U V in
  (ProbabilityTheory.Independent X Y 
  ∧ ProbabilityTheory.HasPDF X (λ x, exp (-(x ^ 2) / 2) / sqrt (2 * π))
  ∧ ProbabilityTheory.HasPDF Y (λ y, exp (-(y ^ 2) / 2) / sqrt (2 * π))) := 
sorry

end BoxMullerTransform_normal_independent_l693_693398


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693911

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693911


namespace probability_standard_bulb_l693_693025

structure FactoryConditions :=
  (P_H1 : ℝ)
  (P_H2 : ℝ)
  (P_H3 : ℝ)
  (P_A_H1 : ℝ)
  (P_A_H2 : ℝ)
  (P_A_H3 : ℝ)

theorem probability_standard_bulb (conditions : FactoryConditions) : 
  conditions.P_H1 = 0.45 → 
  conditions.P_H2 = 0.40 → 
  conditions.P_H3 = 0.15 →
  conditions.P_A_H1 = 0.70 → 
  conditions.P_A_H2 = 0.80 → 
  conditions.P_A_H3 = 0.81 → 
  (conditions.P_H1 * conditions.P_A_H1 + 
   conditions.P_H2 * conditions.P_A_H2 + 
   conditions.P_H3 * conditions.P_A_H3) = 0.7565 :=
by 
  intros h1 h2 h3 a_h1 a_h2 a_h3 
  sorry

end probability_standard_bulb_l693_693025


namespace largest_prime_factor_of_7fact_8fact_l693_693896

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693896


namespace divisible_by_9_l693_693421

theorem divisible_by_9 (n : ℕ) : 9 ∣ (4^n + 15 * n - 1) :=
by
  sorry

end divisible_by_9_l693_693421


namespace distance_between_foci_l693_693658

theorem distance_between_foci :
  (let x := ℝ in ∃ a b : ℝ, (a^2 = 9) ∧ (b^2 = 2.25) ∧ (2 * real.sqrt (a^2 - b^2) = 5.196)) :=
by
  sorry

end distance_between_foci_l693_693658


namespace impossible_300_numbers_l693_693147

theorem impossible_300_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) (hp : ∀ i, 0 < a i)
(hdiff : ∃ k, ∀ i ≠ k, a i = a ((i + 1) % n) - a ((i - 1 + n) % n)) 
: false :=
by {
  sorry
}

end impossible_300_numbers_l693_693147


namespace percentage_of_ore_contains_alloy_l693_693593

def ore_contains_alloy_iron (weight_ore weight_iron : ℝ) (P : ℝ) : Prop :=
  (P / 100 * weight_ore) * 0.9 = weight_iron

theorem percentage_of_ore_contains_alloy (w_ore : ℝ) (w_iron : ℝ) (P : ℝ) 
    (h_w_ore : w_ore = 266.6666666666667) (h_w_iron : w_iron = 60) 
    (h_ore_contains : ore_contains_alloy_iron w_ore w_iron P) 
    : P = 25 :=
by
  rw [h_w_ore, h_w_iron] at h_ore_contains
  sorry

end percentage_of_ore_contains_alloy_l693_693593


namespace find_a_from_conditions_l693_693242

theorem find_a_from_conditions
  (a b : ℤ)
  (h₁ : 2584 * a + 1597 * b = 0)
  (h₂ : 1597 * a + 987 * b = -1) :
  a = 1597 :=
by sorry

end find_a_from_conditions_l693_693242


namespace frequencies_first_class_confidence_difference_quality_l693_693104

theorem frequencies_first_class (a b c d n : ℕ) (Ha : a = 150) (Hb : b = 50) 
                                (Hc : c = 120) (Hd : d = 80) (Hn : n = 400) 
                                (totalA : a + b = 200) 
                                (totalB : c + d = 200) :
  (a / (a + b) = 3 / 4) ∧ (c / (c + d) = 3 / 5) := by
sorry

theorem confidence_difference_quality (a b c d n : ℕ) (Ha : a = 150)
                                       (Hb : b = 50) (Hc : c = 120)
                                       (Hd : d = 80) (Hn : n = 400)
                                       (total : n = 400)
                                       (first_class_total : a + c = 270)
                                       (second_class_total : b + d = 130) :
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  k_squared > 6.635 := by
sorry

end frequencies_first_class_confidence_difference_quality_l693_693104


namespace frequencies_of_first_class_quality_difference_confidence_l693_693109

section quality_comparison

variables (n a b c d : ℕ)

-- Given conditions
def total_products : ℕ := 400
def machine_a_total : ℕ := 200
def machine_a_first : ℕ := 150
def machine_a_second : ℕ := 50
def machine_b_total : ℕ := 200
def machine_b_first : ℕ := 120
def machine_b_second : ℕ := 80

-- Defining the K^2 calculation formula
def K_squared : ℚ :=
  (total_products * (machine_a_first * machine_b_second - machine_a_second * machine_b_first) ^ 2 : ℚ) /
  ((machine_a_first + machine_a_second) * (machine_b_first + machine_b_second) * (machine_a_first + machine_b_first) * (machine_a_second + machine_b_second))

-- Proof statement for Q1: Frequencies of first-class products
theorem frequencies_of_first_class :
  machine_a_first / machine_a_total = 3 / 4 ∧ 
  machine_b_first / machine_b_total = 3 / 5 := 
sorry

-- Proof statement for Q2: Confidence level of difference in quality
theorem quality_difference_confidence :
  K_squared = 10.256 ∧ 10.256 > 6.635 → 0.99 :=
sorry

end quality_comparison

end frequencies_of_first_class_quality_difference_confidence_l693_693109


namespace prob_A_given_at_least_one_hit_l693_693867

theorem prob_A_given_at_least_one_hit (P_A P_B : ℝ) (hA : P_A = 0.6) (hB : P_B = 0.5) :
  let P_at_least_one_hit := P_A * (1 - P_B) + (1 - P_A) * P_B + P_A * P_B in
  let P_A_and_at_least_one_hit := P_A * (1 - P_B) + P_A * P_B in
  P_A_and_at_least_one_hit / P_at_least_one_hit = 3 / 4 :=
by {
  sorry
}

end prob_A_given_at_least_one_hit_l693_693867


namespace arithmetic_mean_integers_neg5_to_6_l693_693872

theorem arithmetic_mean_integers_neg5_to_6 : 
  (let range := (-5:ℤ) to 6 in 
   let count := (6 - -5 + 1 : ℤ) in 
   let sum := (range.sum : ℤ) in
   (sum : ℤ) / (count : ℤ) = (0.5 : ℚ)) :=
by
  let range := List.range' (-5) (1 + 6 - -5 : ℤ).toNat
  let count := 12
  let sum := range.sum
  have : sum = 6, sorry
  have : sum / count = 0.5, sorry
  exact this

end arithmetic_mean_integers_neg5_to_6_l693_693872


namespace largest_prime_factor_l693_693977

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693977


namespace soccer_claim_fraction_l693_693198

theorem soccer_claim_fraction 
  (total_students enjoy_soccer do_not_enjoy_soccer claim_do_not_enjoy honesty fraction_3_over_11 : ℕ)
  (h1 : enjoy_soccer = total_students / 2)
  (h2 : do_not_enjoy_soccer = total_students / 2)
  (h3 : claim_do_not_enjoy = enjoy_soccer * 3 / 10)
  (h4 : honesty = do_not_enjoy_soccer * 8 / 10)
  (h5 : fraction_3_over_11 = enjoy_soccer * 3 / (10 * (enjoy_soccer * 3 / 10 + do_not_enjoy_soccer * 2 / 10)))
  : fraction_3_over_11 = 3 / 11 :=
sorry

end soccer_claim_fraction_l693_693198


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693997

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693997


namespace sin_alpha_value_l693_693679

theorem sin_alpha_value (α : ℝ) (h : 3 * sin (2 * α) = cos α) (h1 : cos α ≠ 0) : sin α = 1 / 6 := 
by
  sorry

end sin_alpha_value_l693_693679


namespace total_sum_of_subsets_eq_512_l693_693721

open Finset

def M : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem total_sum_of_subsets_eq_512 :
  let sum_sums := ∑ S in M.powerset \ {∅}, ∑ k in S, (-1)^k * k
  in sum_sums = 512 :=
by
  simp only [M]
  have sum_formula : ∑ k in M, (-1)^k * k = -2 + 4 - 6 + 8 - 10 + 12 - 14 + 16 - 18 := sorry -- the calculation step here
  have total_subsets : 2^7 = 128 := by norm_num
  exact 128 * sum_formula = 512 := sorry -- total sum calculation
  sorry

end total_sum_of_subsets_eq_512_l693_693721


namespace correct_function_satisfies_condition_l693_693542

theorem correct_function_satisfies_condition (a : ℝ) (h1 : a > 0) (h2: a ≠ 1):
  ∀ x y : ℝ, (let f := λ x, a^x in f x * f y = f (x + y)) :=
by
  intro x y
  let f := λ x, a^x
  sorry

end correct_function_satisfies_condition_l693_693542


namespace find_team_with_most_wins_l693_693051

def team_won_most (n : ℕ) (results : List (List String)) : String :=
  let sorted_teams := results.map fun team => team.sort
  let unique_teams := sorted_teams.eraseDup
  let counts := unique_teams.map fun team => (team, sorted_teams.count (· == team))
  let max_team := counts.maximumBy (·.snd)
  max_team.fst.intercalate " " ++ ":" ++ toString max_team.snd

theorem find_team_with_most_wins (n : ℕ) 
    (hn : 1 ≤ n ∧ n ≤ 30)
    (results : List (List String)) 
    (hresults : ∀ team ∈ results, team.length = 3 ∧ ∀ name ∈ team, 1 ≤ name.length ∧ name.length ≤ 10)  :
    ∃ team : String, ∃ wins : ℕ, (team_won_most n results = team ++ ":" ++ toString wins) :=
by
  sorry

end find_team_with_most_wins_l693_693051


namespace perimeter_of_triangle_l693_693349

theorem perimeter_of_triangle (x : ℕ) :
  let P := (x + (x + 1) + (x - 1))
  P = 21 → x = 7 :=
by
  intros P h
  have h1 : P = 3 * x, 
  { 
    unfold P,
    ring, 
  }
  rw h1 at h,
  linarith

end perimeter_of_triangle_l693_693349


namespace transformed_variance_specific_transformed_variance_l693_693496

theorem transformed_variance (S2 : ℝ) (a b : ℝ) (n : ℕ) (x : ℕ → ℝ)
  (h : S2 = 5)  :
  let S'2 := (1 : ℝ) / n * (Finset.sum (Finset.range n) (λ i, (a * x i + b - (a * (Finset.sum (Finset.range n) x / n) + b))^2))
  in S'2 = a^2 * S2 :=
by
  let original_variance := (1 : ℝ) / n * (Finset.sum (Finset.range n) (λ i, (x i - Finset.sum (Finset.range n) x / n) ^ 2))
  have h_variance : original_variance = S2, from sorry,
  show ((1 : ℝ) / n * (Finset.sum (Finset.range n) (λ i, (a * x i + b - (a * (Finset.sum (Finset.range n) x / n) + b))^2))) = a^2 * S2, from sorry
-- Using the given conditions
end

theorem specific_transformed_variance :
  transformed_variance 5 2 3 = 20 :=
by
  sorry

end transformed_variance_specific_transformed_variance_l693_693496


namespace num_squares_limit_removed_area_l693_693189

theorem num_squares (n : ℕ) : 
  let initial_squares := 1 in
  let remaining_squares := 8^n in
  remaining_squares = 8^n :=
by
  sorry

theorem limit_removed_area (n : ℕ) :
  let removed_area (k : ℕ) := 1 - (8 / 9) ^ k in
  let limit_removed_area := 1 in
  filter.tendsto removed_area filter.atTop (nhds 1) :=
by
  sorry

end num_squares_limit_removed_area_l693_693189


namespace triangle_inequality_check_triangle_sets_l693_693527

theorem triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem check_triangle_sets :
  ¬triangle_inequality 1 2 3 ∧
  triangle_inequality 2 2 2 ∧
  ¬triangle_inequality 2 2 4 ∧
  ¬triangle_inequality 1 3 5 :=
by
  sorry

end triangle_inequality_check_triangle_sets_l693_693527


namespace melting_point_of_ice_in_Celsius_l693_693515

theorem melting_point_of_ice_in_Celsius :
  ∀ (boiling_point_F boiling_point_C melting_point_F temperature_C temperature_F : ℤ),
    (boiling_point_F = 212) →
    (boiling_point_C = 100) →
    (melting_point_F = 32) →
    (temperature_C = 60) →
    (temperature_F = 140) →
    (5 * melting_point_F = 9 * 0 + 160) →         -- Using the given equation F = (9/5)C + 32 and C = 0
    melting_point_F = 32 ∧ 0 = 0 :=
by
  intros
  sorry

end melting_point_of_ice_in_Celsius_l693_693515


namespace total_arrangements_l693_693169

-- Defining the selection and arrangement problem conditions
def select_and_arrange (n m : ℕ) : ℕ :=
  Nat.choose n m * Nat.factorial m

-- Specifying the specific problem's constraints and results
theorem total_arrangements : select_and_arrange 8 2 * select_and_arrange 6 2 = 60 := by
  -- Proof omitted
  sorry

end total_arrangements_l693_693169


namespace air_quality_probabilities_average_exercise_exercise_air_quality_relationship_l693_693577

noncomputable def problem_data := ℕ
noncomputable def total_days := 100

-- Air Quality level data
def air_quality_data := [ (2, 16, 25), (5, 10, 12), (6, 7, 8), (7, 2, 0) ]
def air_quality_probs := [0.43, 0.27, 0.21, 0.09]

-- Midpoint values for the intervals
def exercise_midpoints := [100, 300, 500]

-- Summing the days for each air quality level
def air_quality_sums (data : List (Nat × Nat × Nat)) : List Nat :=
  data.map (λ (x : Nat × Nat × Nat) => x.1 + x.2 + x.3)

def air_quality_probs_computed : List Float :=
  (air_quality_sums air_quality_data).map (λ (x : Nat) => x / total_days.toFloat)

-- Estimated average number of people exercising in the park in a day
def average_exercise_calculated : Float := 
  1 / total_days.toFloat * ((exercise_midpoints[0] * (air_quality_data[0].1 + air_quality_data[1].1 + air_quality_data[2].1 + air_quality_data[3].1)) +
                            (exercise_midpoints[1] * (air_quality_data[0].2 + air_quality_data[1].2 + air_quality_data[2].2 + air_quality_data[3].2)) +
                            (exercise_midpoints[2] * (air_quality_data[0].3 + air_quality_data[1].3 + air_quality_data[2].3 + air_quality_data[3].3)))

def estimated_average_exercise := 350

-- 2x2 Contingency Table data
def contingency_table_counts := (33, 37, 22, 8)

-- Function to compute K^2 for contingency table
def compute_K2 (a b c d n : Nat) : Float :=
  n.toFloat * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d)).toFloat

-- K^2 computation result
def computed_K2_result : Float := compute_K2 33 37 22 8 100

-- Threshold for 95% confidence level
def confidence_threshold := 3.841

-- Relationship conclusion
def relationship : Bool :=
  computed_K2_result > confidence_threshold

theorem air_quality_probabilities :
  air_quality_probs_computed = air_quality_probs := sorry

theorem average_exercise :
  average_exercise_calculated = estimated_average_exercise := sorry

theorem exercise_air_quality_relationship :
  relationship = true := sorry

end air_quality_probabilities_average_exercise_exercise_air_quality_relationship_l693_693577


namespace intersection_sets_l693_693738

def setA : Set ℝ := { x | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3 }
def setB : Set ℝ := { x | (x - 3) / (2 * x) ≤ 0 }

theorem intersection_sets (x : ℝ) : x ∈ setA ∧ x ∈ setB ↔ 0 < x ∧ x ≤ 1 := by
  sorry

end intersection_sets_l693_693738


namespace shaded_area_l693_693756

-- Define the radii of the circles
def R : ℝ := 9        -- radius of the larger circle
def r : ℝ := R / 2    -- radius of each smaller circle (half the radius of the larger circle)

-- Define the areas of the circles
def area_large_circle : ℝ := Real.pi * R^2
def area_small_circle : ℝ := Real.pi * r^2
def total_area_small_circles : ℝ := 3 * area_small_circle

-- Prove the area of the shaded region
theorem shaded_area : area_large_circle - total_area_small_circles = 20.25 * Real.pi := by
  sorry

end shaded_area_l693_693756


namespace xy_equals_10_pow_100_l693_693420

variable (x y : ℝ)

noncomputable def sqrt_log_x := Real.sqrt (Real.log x)
noncomputable def sqrt_log_y := Real.sqrt (Real.log y)
noncomputable def log_sqrt_x := (Real.log (Real.sqrt x))
noncomputable def log_sqrt_y := (Real.log (Real.sqrt y))

-- Defining the main condition
def main_condition := sqrt_log_x + sqrt_log_y + log_sqrt_x + log_sqrt_y + 10 = 120

-- Proving the main statement
theorem xy_equals_10_pow_100 (h : x > 0) (hx : y > 0) (hc : main_condition x y) : x * y = 10 ^ 100 := 
sorry

end xy_equals_10_pow_100_l693_693420


namespace min_dot_product_on_hyperbola_l693_693702

theorem min_dot_product_on_hyperbola : 
  ∀ (m n : ℝ), (m^2 / 4 - n^2 = 1) → 
  let A := (-2, 0) in
  let B := (2, 0) in
  let PA := (-2 - m, -n) in
  let PB := (2 - m, -n) in
  (PA.1 * PB.1 + PA.2 * PB.2) = 0 :=
by
  intros m n h A B PA PB
  rw [A, B, PA, PB]
  sorry

end min_dot_product_on_hyperbola_l693_693702


namespace number_of_buckets_after_reduction_l693_693503

def initial_buckets : ℕ := 25
def reduction_factor : ℚ := 2 / 5

theorem number_of_buckets_after_reduction :
  (initial_buckets : ℚ) * (1 / reduction_factor) = 63 := by
  sorry

end number_of_buckets_after_reduction_l693_693503


namespace simplify_fraction_l693_693454

theorem simplify_fraction : ∃ (a b : ℕ), a = 90 ∧ b = 150 ∧ (90:ℚ) / (150:ℚ) = (3:ℚ) / (5:ℚ) :=
by {
  use 90,
  use 150,
  split,
  refl,
  split,
  refl,
  sorry,
}

end simplify_fraction_l693_693454


namespace find_number_l693_693130

theorem find_number (x : ℕ) (h : x / 3 = 3) : x = 9 :=
sorry

end find_number_l693_693130


namespace initial_volume_solution_l693_693581

theorem initial_volume_solution 
  (V : ℝ) -- initial volume of the solution
  (h_initial_concentration : 0.05 * V) -- 5 percent sodium chloride by volume initially
  (h_evaporation : V - 5500) -- 5500 gallons evaporate
  (h_final_concentration : 0.05 * V / (V - 5500) = 1 / 9) -- concentration becomes approximately 11.11111111111111 percent
  : V = 10000 := 
sorry

end initial_volume_solution_l693_693581


namespace hyperbola_distance_to_foci_l693_693346

theorem hyperbola_distance_to_foci
  (E : ∀ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1)
  (F1 F2 : ℝ)
  (P : ℝ)
  (dist_PF1 : P = 5)
  (a : ℝ)
  (ha : a = 3): 
  |P - F2| = 11 :=
by
  sorry

end hyperbola_distance_to_foci_l693_693346


namespace similar_triangles_segment_length_l693_693869

theorem similar_triangles_segment_length (BC AB DE : ℝ) (BAC EDF : ℝ)
  (h_similar : Triangle.similar ABC DEF)
  (h_BC : BC = 8)
  (h_AB : AB = 10)
  (h_DE : DE = 24)
  (h_BAC : BAC = 90)
  (h_EDF : EDF = 90) : 
  ∃ (EF : ℝ), EF = 19.2 :=
by
  sorry

end similar_triangles_segment_length_l693_693869


namespace simplify_fraction_l693_693451

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end simplify_fraction_l693_693451


namespace cos_double_angle_l693_693682

theorem cos_double_angle (α : ℝ) (h : Real.sin (Real.pi + α) = 1 / 3) : Real.cos (2 * α) = 7 / 9 := 
by 
  sorry

end cos_double_angle_l693_693682


namespace line_passes_through_center_line_intersects_circle_triangle_area_l693_693273

-- Define the circle C
def circle_eq : ℝ → ℝ → ℝ := λ x y, x^2 + y^2 - 4 * x + 2 * y + 1

-- Define the line l
def line_eq (k : ℝ) : ℝ → ℝ := λ x, k * x - 1

-- Define the function that checks if a point lies on the line
def is_on_line (x y k : ℝ) : Prop := y = k * x - 1

-- Define the function that checks if a point lies on the circle
def is_on_circle (x y : ℝ) : Prop := circle_eq x y = 0

-- Center of circle C
def center : ℝ × ℝ := (2, -1)

-- First theorem: Value of k when line l passes through the center of the circle
theorem line_passes_through_center (k : ℝ) : is_on_line 2 (-1) k → k = 0 :=
sorry

-- Second theorem: Line intersects the circle at two points and area of triangle = 2
theorem line_intersects_circle_triangle_area (k : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), is_on_circle x1 y1 ∧ is_on_circle x2 y2 ∧ x1 ≠ x2 ∧ line_eq k x1 = y1 ∧ line_eq k x2 = y2 ∧
    let a := (2, -1) in let b := (x1, y1) in let c := (x2, y2) in (|a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)| / 2) = 2) ↔ (k = 1 ∨ k = -1) :=
sorry

end line_passes_through_center_line_intersects_circle_triangle_area_l693_693273


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693915

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693915


namespace frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693096

theorem frequency_machine_A (total_A first_class_A : ℕ) (h_total_A: total_A = 200) (h_first_class_A: first_class_A = 150) :
  first_class_A / total_A = 3 / 4 := by
  rw [h_total_A, h_first_class_A]
  norm_num

theorem frequency_machine_B (total_B first_class_B : ℕ) (h_total_B: total_B = 200) (h_first_class_B: first_class_B = 120) :
  first_class_B / total_B = 3 / 5 := by
  rw [h_total_B, h_first_class_B]
  norm_num

theorem chi_square_test_significance (n a b c d : ℕ) (h_n: n = 400) (h_a: a = 150) (h_b: b = 50) 
  (h_c: c = 120) (h_d: d = 80) :
  let K_squared := (n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d)))
  in K_squared > 6.635 := by
  rw [h_n, h_a, h_b, h_c, h_d]
  let num := 400 * (150 * 80 - 50 * 120)^2
  let denom := (150 + 50) * (120 + 80) * (150 + 120) * (50 + 80)
  have : K_squared = num / denom := rfl
  norm_num at this
  sorry

end frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693096


namespace equal_radii_right_triangle_l693_693752

theorem equal_radii_right_triangle (CA CB : ℕ) (hCA : CA = 30) (hCB : CB = 16) (AB : ℕ) (hAB : AB = 34)
  (r : ℚ) (h_r_expr : r = 680 / 57) : 
  ∃ p q : ℕ, (p / q) = r ∧ Nat.gcd p q = 1 ∧ (p + q = 737) := by
  use (680, 57)
  split
  . rw [h_r_expr]
  . split
    . exact rfl
    . exact Nat.gcd_eq_one_of_rel_prime 680 57 sorry
  . rfl

#print equal_radii_right_triangle

end equal_radii_right_triangle_l693_693752


namespace abs_h_eq_sqrt_22_div_2_l693_693826

theorem abs_h_eq_sqrt_22_div_2 
  (h : ℝ)
  (x : ℝ)
  (quadratic_eqn : (x - h)^2 + 4 * h = 5 + x)
  (sum_squares_eq_20 : ∀ r s : ℝ, r + s = 2 * h + 1 ∧ r * s = h^2 + 4 * h - 5 → r^2 + s^2 = 20) 
  : |h| = sqrt 22 / 2 :=
sorry

end abs_h_eq_sqrt_22_div_2_l693_693826


namespace vase_sale_gain_l693_693410

theorem vase_sale_gain :
  let C1 := 3.50 / 1.25 in
  let C2 := 3.50 / 0.85 in
  let total_cost := C1 + C2 in
  let total_revenue := 3.50 + 3.50 in
  let net_result := total_revenue - total_cost in
  net_result = 0.08 :=
by
  sorry

end vase_sale_gain_l693_693410


namespace values_of_x_l693_693697

variable (x : ℤ)

def p : Prop := |x - 1| ≤ 1
def q : Prop := x ∉ ℤ

theorem values_of_x :
  (¬ (p x) ∧ ¬ (p x ∧ q x) = false → (0 ≤ x ∧ x ≤ 2)) :=
by
  sorry

end values_of_x_l693_693697


namespace probability_both_dice_greater_than_4_l693_693129

def ProbabilitySingleDieGreaterThan4 : ℚ := 2 / 6

theorem probability_both_dice_greater_than_4 :
  (ProbabilitySingleDieGreaterThan4 * ProbabilitySingleDieGreaterThan4) = (1 / 9) :=
by
  sorry

end probability_both_dice_greater_than_4_l693_693129


namespace problem_statement_l693_693157

variables {x y P Q : ℝ}

theorem problem_statement (h1 : x^2 + y^2 = (x + y)^2 + P) (h2 : x^2 + y^2 = (x - y)^2 + Q) : P = -2 * x * y ∧ Q = 2 * x * y :=
by
  sorry

end problem_statement_l693_693157


namespace problem1_problem2_problem3_l693_693314

namespace MathProofProblems

-- Define the function f(x)
def f (x a : ℝ) := x^2 - 2 * a * x

-- Problem 1: When a = 2, solve the inequality -3 < f(x) < 5 with respect to x
theorem problem1 : set_of (λ x : ℝ, -3 < f x 2 ∧ f x 2 < 5) = {x | (-1 < x ∧ x < 1) ∨ (3 < x ∧ x < 5)} :=
by
  sorry

-- Problem 2: Expression for M(a) given the conditions
def M (a : ℝ) : ℝ := if a > real.sqrt 5 then a - real.sqrt (a^2 - 5) else a + real.sqrt (a^2 + 5)

theorem problem2 (a : ℝ) (ha_pos : 0 < a) : ∀ x ∈ Icc 0 (M a), abs (f x a) ≤ 5 :=
by
  sorry

-- Problem 3: Values of a and t such that the function y = f(x) has specific max and min values
theorem problem3 (a t : ℝ) : 
  ((f t a = 0 ∧ f (t + 2) a = 0 ∧ (∀ x ∈ Icc t (t + 2), f x a ≤ 0 ∧ f x a ≥ -4)) ↔ (a = 2 ∧ (t = 0 ∨ t = 2))) :=
by
  sorry

end MathProofProblems

end problem1_problem2_problem3_l693_693314


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693931

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693931


namespace interest_rate_decrease_l693_693192

theorem interest_rate_decrease (initial_rate final_rate : ℝ) (x : ℝ) 
  (h_initial_rate : initial_rate = 2.25 * 0.01)
  (h_final_rate : final_rate = 1.98 * 0.01) :
  final_rate = initial_rate * (1 - x)^2 := 
  sorry

end interest_rate_decrease_l693_693192


namespace frequencies_first_class_confidence_difference_quality_l693_693103

theorem frequencies_first_class (a b c d n : ℕ) (Ha : a = 150) (Hb : b = 50) 
                                (Hc : c = 120) (Hd : d = 80) (Hn : n = 400) 
                                (totalA : a + b = 200) 
                                (totalB : c + d = 200) :
  (a / (a + b) = 3 / 4) ∧ (c / (c + d) = 3 / 5) := by
sorry

theorem confidence_difference_quality (a b c d n : ℕ) (Ha : a = 150)
                                       (Hb : b = 50) (Hc : c = 120)
                                       (Hd : d = 80) (Hn : n = 400)
                                       (total : n = 400)
                                       (first_class_total : a + c = 270)
                                       (second_class_total : b + d = 130) :
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  k_squared > 6.635 := by
sorry

end frequencies_first_class_confidence_difference_quality_l693_693103


namespace sum_of_monomials_same_type_l693_693348

theorem sum_of_monomials_same_type 
  (x y : ℝ) 
  (m n : ℕ) 
  (h1 : m = 1) 
  (h2 : 3 = n + 1) : 
  (2 * x ^ m * y ^ 3) + (-5 * x * y ^ (n + 1)) = -3 * x * y ^ 3 := 
by 
  sorry

end sum_of_monomials_same_type_l693_693348


namespace part_a_part_b_l693_693811

/-- Part (a) statement: -/
theorem part_a (x : Fin 100 → ℕ) :
  (∀ i j k a b c d : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d →
    x i + x j + x k < x a + x b + x c + x d) →
  (∀ i j a b c : Fin 100, i ≠ j ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    x i + x j < x a + x b + x c) :=
by
  sorry

/-- Part (b) statement: -/
theorem part_b (x : Fin 100 → ℕ) :
  (∀ i j a b c : Fin 100, i ≠ j ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    x i + x j < x a + x b + x c) →
  (∀ i j k a b c d : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d →
    x i + x j + x k < x a + x b + x c + x d) :=
by
  sorry

end part_a_part_b_l693_693811


namespace arc_length_independent_of_position_l693_693190

theorem arc_length_independent_of_position (A B C: Point) (O : Point) (r : ℝ) 
(h₁ : triangle_is_equilateral A B C) 
(h₂ : line_parallel_through_point (line BC) A O) 
(h₃ : touches_segment (circle O r) BC) : 
arc_length_of_circle_within_triangle (circle O r) (triangle A B C) = constant_length :=
sorry

end arc_length_independent_of_position_l693_693190


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693916

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693916


namespace sixty_three_times_fifty_seven_l693_693627

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 :=
by
  let a := 60
  let b := 3
  have h : (a + b) * (a - b) = a^2 - b^2 := by sorry
  have h1 : 63 = a + b := by rfl
  have h2 : 57 = a - b := by rfl
  calc
    63 * 57 = (a + b) * (a - b) : by rw [h1, h2]
    ... = a^2 - b^2 : by rw h
    ... = 60^2 - 3^2 : by rfl
    ... = 3600 - 9 : by sorry
    ... = 3591 : by norm_num

end sixty_three_times_fifty_seven_l693_693627


namespace roger_cookie_price_l693_693196

open Classical

theorem roger_cookie_price
  (art_base1 art_base2 art_height : ℕ) 
  (art_cookies_per_batch art_cookie_price roger_cookies_per_batch : ℕ)
  (art_area : ℕ := (art_base1 + art_base2) * art_height / 2)
  (total_dough : ℕ := art_cookies_per_batch * art_area)
  (roger_area : ℚ := total_dough / roger_cookies_per_batch)
  (art_total_earnings : ℚ := art_cookies_per_batch * art_cookie_price) :
  ∀ (roger_cookie_price : ℚ), roger_cookies_per_batch * roger_cookie_price = art_total_earnings →
  roger_cookie_price = 100 / 3 :=
sorry

end roger_cookie_price_l693_693196


namespace largest_prime_factor_of_7fact_8fact_l693_693898

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693898


namespace final_number_odd_l693_693413

theorem final_number_odd :
  (∃ n : ℕ, n < 1967 ∧ (∀ k : ℕ, (k ∈ {i | i ∈ finset.range 1967}) → odd k → odd (1 + k - 1))) :=
begin
  -- Definitions of initial conditions and operations
  let initial_numbers := finset.range 1967,
  let operations := λ (a b : ℕ), a - b,
  
  -- Final statement to prove
  let remaining_number := 1, -- it's implicitly defined as the number remaining after all operations
  have : ∀ (x y : ℕ), x ∈ initial_numbers → y ∈ initial_numbers → x ≠ y → 
         (∃ z : ℕ, z = operations x y ∧ (∃ c d ∈ initial_numbers, operations c d = z)),
  sorry
end

end final_number_odd_l693_693413


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693992

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693992


namespace largest_prime_factor_of_fact_sum_is_7_l693_693963

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693963


namespace sum_of_squares_of_distances_l693_693774

-- Definitions based on the conditions provided:
variables (A B C D X : Point)
variable (a : ℝ)
variable (h1 h2 h3 h4 : ℝ)

-- Conditions:
axiom square_side_length : a = 5
axiom area_ratios : (1/2 * a * h1) / (1/2 * a * h2) = 1 / 5 ∧ 
                    (1/2 * a * h2) / (1/2 * a * h3) = 5 / 9

-- Problem Statement to Prove:
theorem sum_of_squares_of_distances :
  h1^2 + h2^2 + h3^2 + h4^2 = 33 :=
sorry

end sum_of_squares_of_distances_l693_693774


namespace triangle_count_geq_l693_693001

noncomputable def A (n : ℕ) : finset (ℝ × ℝ) := sorry -- Define the set A (2n distinct points)
def segments (n k : ℕ) : finset (finset (ℝ × ℝ)) := sorry -- Define the segments set (n^2 + k segments)

theorem triangle_count_geq (n k : ℕ) (h_n_pos : 0 < n) (h_k_pos : 0 < k) 
  (h_distinct: (A n).card = 2 * n) (h_non_collinear: ∀ p1 p2 p3 ∈ A n, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 → ¬collinear p1 p2 p3)
  (h_segments : (segments n k).card = n^2 + k) :
  ∃ t : finset (finset (ℝ × ℝ)), t.card ≥ (4 / 3) * k^(3/2) ∧
  (∀ tri ∈ t, ∃ a b c ∈ (A n), {a, b, c} = tri ∧
  ∀ s ∈ tri.powerset.filter(λ s, s.card = 2), s ∈ (segments n k)) :=
sorry -- Proof this theorem

end triangle_count_geq_l693_693001


namespace number_of_valid_sets_l693_693641

-- Define the set U of 20 diagonals in the regular polygon P1P2P3P4P5P6P7P8
def is_diagonal (i j : ℕ) : Prop := (1 ≤ i ∧ i ≤ 8) ∧ (1 ≤ j ∧ j ≤ 8) ∧ i ≠ j ∧ abs (i - j) ≠ 1 ∧ abs (i - j) ≠ 7

def U := {P_ij : (ℕ × ℕ) // is_diagonal P_ij.1 P_ij.2}

-- Define the conditions for set S
def is_valid_set (S : finset (ℕ × ℕ)) : Prop :=
  (∀ P_ij ∈ S, is_diagonal P_ij.1 P_ij.2) ∧
  (∀ P_ij P_jk ∈ S, P_ij.2 = P_jk.1 → ∃ P_ik ∈ S, P_ik.1 = P_ij.1 ∧ P_ik.2 = P_jk.2)

-- The number of such valid sets
def count_valid_sets : ℕ := 715

theorem number_of_valid_sets :
  (finset.filter is_valid_set (finset.powerset U)).card = count_valid_sets := sorry

end number_of_valid_sets_l693_693641


namespace unique_solution_a_exists_l693_693735

open Real

noncomputable def equation (a x : ℝ) :=
  4 * a^2 + 3 * x * log x + 3 * (log x)^2 = 13 * a * log x + a * x

theorem unique_solution_a_exists : 
  ∃! a : ℝ, ∃ x : ℝ, 0 < x ∧ equation a x :=
sorry

end unique_solution_a_exists_l693_693735


namespace moles_H2_formed_l693_693253

/-- 
The reaction given is:
NaH + H₂O -> NaOH + H₂
Given 3 moles of NaH and 3 moles of H₂O, prove that the number of moles of H₂ formed is 3.
-/ 
theorem moles_H2_formed (n_NaH : ℕ) (n_H2O : ℕ) (n_NaH_val : n_NaH = 3) (n_H2O_val : n_H2O = 3) :
  let n_H2 := n_NaH in
  n_H2 = 3 :=
by
  sorry

end moles_H2_formed_l693_693253


namespace necessary_condition_l693_693055

variables {a b x : ℤ} 

theorem necessary_condition 
  (h1 : (∃ x : ℤ, x^2 + a * x + b = 0))
  (h2 : a ∈ ℤ ∧ b ∈ ℤ) :
  a + b ∈ ℤ :=
sorry

end necessary_condition_l693_693055


namespace simplify_fraction_l693_693457

theorem simplify_fraction : ∃ (a b : ℕ), a = 90 ∧ b = 150 ∧ (90:ℚ) / (150:ℚ) = (3:ℚ) / (5:ℚ) :=
by {
  use 90,
  use 150,
  split,
  refl,
  split,
  refl,
  sorry,
}

end simplify_fraction_l693_693457


namespace contestant_A_final_round_probability_distribution_xi_expectation_of_xi_l693_693548

open ProbabilityTheory MeasureTheory

noncomputable def prob_enter_final_round (p : ℚ) : ℚ := 
  p^3 + (3 * p^2 * (1 - p) * p) + (6 * p^2 * (1 - p)^2 * p)

theorem contestant_A_final_round_probability :
  let p := (2 : ℚ) / 3
  prob_enter_final_round p = 64 / 81 := 
sorry

def distribution_of_xi (p : ℚ) (k : ℕ) : ℚ :=
  match k with
  | 3 => p^3 + (1 - p)^3
  | 4 => (3 * p^2 * (1 - p) * p) + (3 * (1 - p)^2 * p * (1 - p))
  | 5 => (6 * p^2 * (1 - p)^2 * p) + (6 * p^2 * (1 - p)^2 * (1 - p))
  | _ => 0 -- only valid for 3, 4, 5 as per conditions

theorem distribution_xi : 
  let p := (2 : ℚ) / 3
  distribution_of_xi p 3 = 1 / 3 ∧
  distribution_of_xi p 4 = 10 / 27 ∧
  distribution_of_xi p 5 = 8 / 27 := 
sorry

theorem expectation_of_xi :
  let p := (2 : ℚ) / 3
  (3 * distribution_of_xi p 3 + 4 * distribution_of_xi p 4 + 5 * distribution_of_xi p 5) = 3 + 26 / 27 :=
sorry

end contestant_A_final_round_probability_distribution_xi_expectation_of_xi_l693_693548


namespace minimize_fuel_consumption_l693_693506

-- Define conditions as constants
def cargo_total : ℕ := 157
def cap_large : ℕ := 5
def cap_small : ℕ := 2
def fuel_large : ℕ := 20
def fuel_small : ℕ := 10

-- Define truck counts
def n_large : ℕ := 31
def n_small : ℕ := 1

-- Theorem: the number of large and small trucks that minimize fuel consumption
theorem minimize_fuel_consumption : 
  n_large * cap_large + n_small * cap_small = cargo_total ∧
  (∀ m_large m_small, m_large * cap_large + m_small * cap_small = cargo_total → 
    m_large * fuel_large + m_small * fuel_small ≥ n_large * fuel_large + n_small * fuel_small) :=
by
  -- Statement to be proven
  sorry

end minimize_fuel_consumption_l693_693506


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693097

theorem frequencies_of_first_class_products 
  (aA bA aB bB : ℕ) 
  (total_A total_B total_products : ℕ) 
  (hA : aA = 150) (hB : bA = 50) (hC : aB = 120) (hD : bB = 80) 
  (hE : total_A = 200) (hF : total_B = 200) (h_total : total_products = total_A + total_B) : 
  aA / total_A = 3 / 4 ∧ aB / total_B = 3 / 5 := 
by
  sorry

theorem confidence_in_difference_of_quality 
  (n a b c d : ℕ) 
  (h_n : n = 400) (hA : a = 150) (hB : b = 50) (hC : c = 120) (hD : d = 80) 
  (K2 : ℝ) 
  (hK2 : K2 = n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))) : 
  K2 > 6.635 ∧ K2 < 10.828 := 
by
  sorry

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l693_693097


namespace trailing_zeros_in_15_factorial_base_15_are_3_l693_693630

/--
Compute the number of trailing zeros in \( 15! \) when expressed in base 15.
-/
def compute_trailing_zeros_in_factorial_base_15 : ℕ :=
  let num_factors_3 := (15 / 3) + (15 / 9)
  let num_factors_5 := (15 / 5)
  min num_factors_3 num_factors_5

theorem trailing_zeros_in_15_factorial_base_15_are_3 :
  compute_trailing_zeros_in_factorial_base_15 = 3 :=
sorry

end trailing_zeros_in_15_factorial_base_15_are_3_l693_693630


namespace simplify_fraction_l693_693458

theorem simplify_fraction : ∃ (a b : ℕ), a = 90 ∧ b = 150 ∧ (90:ℚ) / (150:ℚ) = (3:ℚ) / (5:ℚ) :=
by {
  use 90,
  use 150,
  split,
  refl,
  split,
  refl,
  sorry,
}

end simplify_fraction_l693_693458


namespace car_speed_l693_693164

def distance := 390 -- km
def time := 4 -- hours
def speed := distance / time -- speed in km/h

theorem car_speed : speed = 97.5 :=
by
  sorry

end car_speed_l693_693164


namespace unique_solution_count_l693_693733

theorem unique_solution_count : 
  ∃! (a : ℝ), ∀ {x : ℝ}, (0 < x) → 
    4 * a ^ 2 + 3 * x * log x + 3 * (log x)^2 = 13 * a * log x + a * x :=
sorry

end unique_solution_count_l693_693733


namespace tessa_still_owes_greg_l693_693726

def initial_debt : ℝ := 40
def first_repayment : ℝ := 0.25 * initial_debt
def debt_after_first_repayment : ℝ := initial_debt - first_repayment
def second_borrowing : ℝ := 25
def debt_after_second_borrowing : ℝ := debt_after_first_repayment + second_borrowing
def second_repayment : ℝ := 0.5 * debt_after_second_borrowing
def debt_after_second_repayment : ℝ := debt_after_second_borrowing - second_repayment
def third_borrowing : ℝ := 30
def debt_after_third_borrowing : ℝ := debt_after_second_repayment + third_borrowing
def third_repayment : ℝ := 0.1 * debt_after_third_borrowing
def final_debt : ℝ := debt_after_third_borrowing - third_repayment

theorem tessa_still_owes_greg : final_debt = 51.75 := by
  sorry

end tessa_still_owes_greg_l693_693726


namespace frequencies_of_first_class_quality_difference_confidence_l693_693110

section quality_comparison

variables (n a b c d : ℕ)

-- Given conditions
def total_products : ℕ := 400
def machine_a_total : ℕ := 200
def machine_a_first : ℕ := 150
def machine_a_second : ℕ := 50
def machine_b_total : ℕ := 200
def machine_b_first : ℕ := 120
def machine_b_second : ℕ := 80

-- Defining the K^2 calculation formula
def K_squared : ℚ :=
  (total_products * (machine_a_first * machine_b_second - machine_a_second * machine_b_first) ^ 2 : ℚ) /
  ((machine_a_first + machine_a_second) * (machine_b_first + machine_b_second) * (machine_a_first + machine_b_first) * (machine_a_second + machine_b_second))

-- Proof statement for Q1: Frequencies of first-class products
theorem frequencies_of_first_class :
  machine_a_first / machine_a_total = 3 / 4 ∧ 
  machine_b_first / machine_b_total = 3 / 5 := 
sorry

-- Proof statement for Q2: Confidence level of difference in quality
theorem quality_difference_confidence :
  K_squared = 10.256 ∧ 10.256 > 6.635 → 0.99 :=
sorry

end quality_comparison

end frequencies_of_first_class_quality_difference_confidence_l693_693110


namespace final_tree_count_l693_693859

noncomputable def current_trees : ℕ := 39
noncomputable def trees_planted_today : ℕ := 41
noncomputable def trees_planted_tomorrow : ℕ := 20

theorem final_tree_count : current_trees + trees_planted_today + trees_planted_tomorrow = 100 := by
  sorry

end final_tree_count_l693_693859


namespace sum_of_solutions_eq_65_l693_693797

noncomputable def f (x : ℝ) : ℝ := 12 * x + 5

theorem sum_of_solutions_eq_65 :
  let S := {x : ℝ | f⁻¹'({x}) = f ((3 * x)⁻¹)} in
  (∑ x in S, x) = 65 :=
by
  sorry

end sum_of_solutions_eq_65_l693_693797


namespace frequencies_first_class_confidence_difference_quality_l693_693107

theorem frequencies_first_class (a b c d n : ℕ) (Ha : a = 150) (Hb : b = 50) 
                                (Hc : c = 120) (Hd : d = 80) (Hn : n = 400) 
                                (totalA : a + b = 200) 
                                (totalB : c + d = 200) :
  (a / (a + b) = 3 / 4) ∧ (c / (c + d) = 3 / 5) := by
sorry

theorem confidence_difference_quality (a b c d n : ℕ) (Ha : a = 150)
                                       (Hb : b = 50) (Hc : c = 120)
                                       (Hd : d = 80) (Hn : n = 400)
                                       (total : n = 400)
                                       (first_class_total : a + c = 270)
                                       (second_class_total : b + d = 130) :
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  k_squared > 6.635 := by
sorry

end frequencies_first_class_confidence_difference_quality_l693_693107


namespace power_function_value_l693_693847

theorem power_function_value (f : ℝ → ℝ) (h : ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a) (h₁ : f 4 = 1 / 2) :
  f (1 / 16) = 4 :=
sorry

end power_function_value_l693_693847


namespace sum_of_first_12_is_21_l693_693696

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_12_is_21 (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a) (h_sum : ∑ i in finset.range 12, a i = 21) :
  a 1 + a 4 + a 7 + a 10 = 7 :=
sorry

end sum_of_first_12_is_21_l693_693696


namespace largest_prime_factor_7fac_plus_8fac_l693_693960

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693960


namespace coeff_x4_product_l693_693120

theorem coeff_x4_product :
  let poly1 := 2 * x^5 - 4 * x^4 + 3 * x^3 - 5 * x^2 + x - 1
  let poly2 := 3 * x^4 - 2 * x^3 + x^2 + 5 * x + 6
  let product := poly1 * poly2
  polynomial.coeff (product) 4 = -19 := 
by
  sorry

end coeff_x4_product_l693_693120


namespace clock_chime_time_l693_693863

theorem clock_chime_time (t : ℕ) (h : t = 12) (k : 4 * (t / (4 - 1)) = 12) :
  12 * (t / (4 - 1)) - (12 - 1) * (t / (4 - 1)) = 44 :=
by {
  sorry
}

end clock_chime_time_l693_693863


namespace solve_for_x_l693_693825

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 14.7 -> x = 105 := by
  sorry

end solve_for_x_l693_693825


namespace quadrant_of_complex_number_l693_693689

theorem quadrant_of_complex_number (z : ℂ) (h : I * z = 2 + 3 * I) : 
  complex.re z > 0 ∧ complex.im z < 0 :=
sorry

end quadrant_of_complex_number_l693_693689


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693993

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693993


namespace ceil_neg_3_7_l693_693226

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := by
  sorry

end ceil_neg_3_7_l693_693226


namespace find_length_of_LN_l693_693758

theorem find_length_of_LN (LMN : Triangle)
  (angle_M_90 : is_right_triangle LMN M)
  (sin_N : sin (angle_at N) = 3 / 5)
  (LM_length : length_side LMN LM = 15) :
  length_side LMN LN = 25 := by
  sorry

end find_length_of_LN_l693_693758


namespace largest_prime_factor_of_fact_sum_is_7_l693_693967

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693967


namespace T1_l693_693694

constant maa : Type
constant pib : Type

-- Postulates
axiom P1 (p : pib) : Set maa
axiom P2 (p1 p2 p3 : pib) (h1: p1 ≠ p2) (h2: p1 ≠ p3) (h3: p2 ≠ p3) : 
  Set.inter (P1 p1) (Set.inter (P1 p2) (P1 p3)).card = 1
axiom P3 (m : maa) : {p : pib | m ∈ P1 p}.card = 3
axiom P4 : {p : pib | true}.card = 5

-- Theorem T1
theorem T1 : {m : maa | true}.card = 10 :=
sorry

-- Final proof structure
constant FinalProof : 
  (P1_type : ∀ p : pib, Set maa) → 
  (P2_type : ∀ (p1 p2 p3 : pib), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → Set.inter (P1_type p1) (Set.inter (P1_type p2) (P1_type p3)).card = 1) → 
  (P3_type : ∀ m : maa, {p : pib | m ∈ P1_type p}.card = 3) → 
  (P4_type : {p : pib | true}.card = 5) → 
  {m : maa | true}.card = 10 := 
λ P1 P2 P3 P4, T1

end T1_l693_693694


namespace smallest_and_second_smallest_diff_l693_693862

theorem smallest_and_second_smallest_diff :
  let nums := {10, 11, 12}
  let min := 10
  let second_min := 11
  second_min - min = 1 :=
by
  let nums := {10, 11, 12}
  let min := 10
  let second_min := 11
  have h : second_min - min = 1 := sorry
  exact h

end smallest_and_second_smallest_diff_l693_693862


namespace chord_length_of_dividing_hexagon_l693_693568

theorem chord_length_of_dividing_hexagon (r : ℝ) : 
  let a := 5
  let b := 7
  let l := r * sqrt 3
  hex_inscribed (r : ℝ) →
  hex_side_lengths [a, b, a, b, a, b] →
  (l = 7 * sqrt 3) := by
  sorry

end chord_length_of_dividing_hexagon_l693_693568


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693884

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693884


namespace maria_first_stop_distance_is_280_l693_693647

noncomputable def maria_travel_distance : ℝ := 560
noncomputable def first_stop_distance (x : ℝ) : ℝ := x
noncomputable def distance_after_first_stop (x : ℝ) : ℝ := maria_travel_distance - first_stop_distance x
noncomputable def second_stop_distance (x : ℝ) : ℝ := (1 / 4) * distance_after_first_stop x
noncomputable def remaining_distance : ℝ := 210

theorem maria_first_stop_distance_is_280 :
  ∃ x, first_stop_distance x = 280 ∧ second_stop_distance x + remaining_distance = distance_after_first_stop x := sorry

end maria_first_stop_distance_is_280_l693_693647


namespace bakery_storage_l693_693535

theorem bakery_storage (S F B : ℕ) (h1 : S * 8 = 3 * F) (h2 : F * 1 = 10 * B) (h3 : F * 1 = 8 * (B + 60)) : S = 900 :=
by
  -- We would normally put the proof steps here, but since it's specified to include only the statement
  sorry

end bakery_storage_l693_693535


namespace sum_of_positive_a_integers_with_integer_root_l693_693313

noncomputable def f (a x : ℕ) : ℕ := a * x * x + (1 - 2 * a) * x + a - 3

theorem sum_of_positive_a_integers_with_integer_root :
  (∑ a in ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ), if ∃ x : ℤ, f a x = 0 then a else 0) = 4 :=
  by
    sorry

end sum_of_positive_a_integers_with_integer_root_l693_693313


namespace plant_lamp_arrangements_l693_693040

/-- Rachel has two identical basil plants and an aloe plant.
Additionally, she has two identical white lamps, two identical red lamps, and 
two identical blue lamps she can put each plant under 
(she can put more than one plant under a lamp, but each plant is under exactly one lamp). 
-/
theorem plant_lamp_arrangements : 
  let plants := ["basil", "basil", "aloe"]
  let lamps := ["white", "white", "red", "red", "blue", "blue"]
  ∃ n, n = 27 := by
  sorry

end plant_lamp_arrangements_l693_693040


namespace jame_hourly_wage_is_530_l693_693376

def jame_old_hourly : ℝ := 16
def jame_old_hours_per_week : ℝ := 25
def jame_weeks_per_year : ℝ := 52
def jame_old_annual_income : ℝ := jame_old_hourly * jame_old_hours_per_week * jame_weeks_per_year

def jame_new_hours_per_week : ℝ := 40
def income_difference : ℝ := 20800

def jame_new_hourly_wage : ℝ :=
  (jame_old_annual_income + income_difference) / (jame_new_hours_per_week * jame_weeks_per_year)

theorem jame_hourly_wage_is_530 :
  jame_new_hourly_wage = 530 := sorry

end jame_hourly_wage_is_530_l693_693376


namespace premium_rate_is_three_l693_693184

-- Define the conditions
def original_value := 14000
def insured_fraction := 5 / 7
def premium_amount := 300
def insured_value := insured_fraction * original_value

-- Define the statement to be proven
def premium_rate : ℝ :=
  (premium_amount / insured_value) * 100

-- Prove that the premium rate is 3%
theorem premium_rate_is_three : premium_rate = 3 :=
by
  -- Skip the proof, just state the goal
  sorry

end premium_rate_is_three_l693_693184


namespace largest_prime_factor_of_7fact_8fact_l693_693900

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693900


namespace largest_even_number_l693_693749

theorem largest_even_number (x : ℤ) (h1 : 3 * x + 6 = (x + (x + 2) + (x + 4)) / 3 + 44) : 
  x + 4 = 24 := 
by 
  sorry

end largest_even_number_l693_693749


namespace find_interest_rate_l693_693183

-- Definitions of given values
def SimpleInterest : ℝ := 70
def Principal : ℝ := 400
def Time : ℝ := 5

-- Definition of expected interest rate
def ExpectedRate : ℝ := 3.5 / 100  -- Converting percentage to a decimal

-- Theorem statement
theorem find_interest_rate : (SimpleInterest = Principal * ExpectedRate * Time) :=
by
  sorry

end find_interest_rate_l693_693183


namespace time_interval_between_recordings_is_5_seconds_l693_693380

theorem time_interval_between_recordings_is_5_seconds
  (instances_per_hour : ℕ)
  (seconds_per_hour : ℕ)
  (h1 : instances_per_hour = 720)
  (h2 : seconds_per_hour = 3600) :
  seconds_per_hour / instances_per_hour = 5 :=
by
  -- proof omitted
  sorry

end time_interval_between_recordings_is_5_seconds_l693_693380


namespace total_legs_in_farm_l693_693075

def num_animals : Nat := 13
def num_chickens : Nat := 4
def legs_per_chicken : Nat := 2
def legs_per_buffalo : Nat := 4

theorem total_legs_in_farm : 
  (num_chickens * legs_per_chicken) + ((num_animals - num_chickens) * legs_per_buffalo) = 44 :=
by
  sorry

end total_legs_in_farm_l693_693075


namespace largest_prime_factor_7fac_8fac_l693_693945

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693945


namespace impossible_300_numbers_l693_693148

theorem impossible_300_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) (hp : ∀ i, 0 < a i)
(hdiff : ∃ k, ∀ i ≠ k, a i = a ((i + 1) % n) - a ((i - 1 + n) % n)) 
: false :=
by {
  sorry
}

end impossible_300_numbers_l693_693148


namespace partial_fraction_product_l693_693720

theorem partial_fraction_product (A B C : ℚ)
  (h_eq : ∀ x, (x^2 - 13) / ((x-2) * (x+2) * (x-3)) = A / (x-2) + B / (x+2) + C / (x-3))
  (h_A : A = 9 / 4)
  (h_B : B = -9 / 20)
  (h_C : C = -4 / 5) :
  A * B * C = 81 / 100 := 
by
  sorry

end partial_fraction_product_l693_693720


namespace correct_calculation_l693_693524

theorem correct_calculation (m n : ℝ) : 4 * m + 2 * n - (n - m) = 5 * m + n :=
by sorry

end correct_calculation_l693_693524


namespace largest_prime_factor_l693_693978

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693978


namespace minimum_teachers_required_l693_693599

theorem minimum_teachers_required :
  ∀ (math_teachers physics_teachers chemistry_teachers biology_teachers computer_science_teachers : ℕ) 
  (periods per_day : ℕ) (max_subjects_per_teacher : ℕ),
  math_teachers = 5 ∧
  physics_teachers = 4 ∧
  chemistry_teachers = 4 ∧
  biology_teachers = 4 ∧
  computer_science_teachers = 3 ∧
  periods = 6 ∧
  max_subjects_per_teacher = 2 →
  (let total_slots := (5 * periods) + (4 * periods) + (4 * periods) + (4 * periods) + (3 * periods) in
   let slots_per_teacher := 2 * periods in
   total_slots / slots_per_teacher = 10) :=
by sorry

end minimum_teachers_required_l693_693599


namespace tangent_line_eq_range_of_a_l693_693714

-- Define the function f(x) = ax + (2a-1)/x + 1 - 3a
def f (a x : ℝ) : ℝ := a * x + (2 * a - 1) / x + 1 - 3 * a

-- Part I: Prove the equation of the tangent line to the function y = f(x) at the point (2, f(2))
theorem tangent_line_eq (a : ℝ) (h : a = 1) :
  let f₁ := f 1
  ∀ x y : ℝ, y - f₁ 2 = (1 - (1 / (2^2))) * (x - 2) ↔ 3 * x - 4 * y - 4 = 0 :=
sorry

-- Part II: Prove the range of values for the real number a
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → f a x ≥ (1 - a) * Real.log x) :
  a ≥ 1 / 3 :=
sorry

end tangent_line_eq_range_of_a_l693_693714


namespace largest_prime_factor_7fac_plus_8fac_l693_693957

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693957


namespace A_number_is_35_l693_693545

theorem A_number_is_35 (A B : ℕ) 
  (h_sum_digits : A + B = 8) 
  (h_diff_numbers : 10 * B + A = 10 * A + B + 18) :
  10 * A + B = 35 :=
by {
  sorry
}

end A_number_is_35_l693_693545


namespace functional_eq_and_nonzero_implies_f3_eq_one_l693_693483

variable {R : Type*} [IsDomain R] (f : R → R)

theorem functional_eq_and_nonzero_implies_f3_eq_one
  (h1 : ∀ x y : R, f(x - y) = f(x) * f(y))
  (h2 : ∀ x : R, f(x) ≠ 0) : f 3 = 1 := 
  sorry

end functional_eq_and_nonzero_implies_f3_eq_one_l693_693483


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693919

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693919


namespace largest_prime_factor_7fac_8fac_l693_693939

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693939


namespace sixty_three_times_fifty_seven_l693_693623

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 := by
  sorry

end sixty_three_times_fifty_seven_l693_693623


namespace divides_if_not_divisible_by_4_l693_693433

theorem divides_if_not_divisible_by_4 (n : ℕ) :
  (¬ (4 ∣ n)) → (5 ∣ (1^n + 2^n + 3^n + 4^n)) :=
by sorry

end divides_if_not_divisible_by_4_l693_693433


namespace geom_seq_properties_l693_693854

-- Define the geometric sequence and its conditions
def geom_seq (a : ℕ → ℝ) : Prop :=
  (∀ n, 0 < a n) ∧ 
  (2 * a 3 = real.sqrt (a 2 * a 6)) ∧ 
  (2 * a 1 + 3 * a 2 = 16)

-- Define b_n
def b_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), real.log 2 (a i)

-- Define the sum of the first n terms of 1 / b_n
def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), 1 / (b_n a i)

-- The main statement combining the conclusions
theorem geom_seq_properties (a : ℕ → ℝ) (n : ℕ) : 
  geom_seq a → 
  (∀ m, a m = 2^m) ∧ 
  (S_n a n = (2 * n) / (n + 1)) :=
begin
  -- Skip the proof for now
  sorry
end

end geom_seq_properties_l693_693854


namespace even_number_in_every_row_from_third_onwards_l693_693156

-- Define the conditions and theorems
theorem even_number_in_every_row_from_third_onwards :
  ∀ (n : ℕ), (n ≥ 3) → ∃ (k : ℤ), (|k| < n) ∧ even (a n k) :=
by
  -- Definitions
  def a : ℕ → ℤ → ℤ
  | 1, 0 := 1
  | n+1, k := (a n (k-1) + a n k + a n (k+1))

  -- Proof statement placeholder
  sorry

end even_number_in_every_row_from_third_onwards_l693_693156


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693894

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693894


namespace quadratic_decreasing_condition_l693_693748

theorem quadratic_decreasing_condition (h : ℝ) (y : ℝ → ℝ) (x : ℝ) (hx : x < 1) 
    (hy : y = fun x => (x - h)^2 + 3) : 
    (∀ x1 x2 : ℝ, x1 < 1 → x2 < 1 → x1 < x2 → y x1 > y x2) → (h ≥ 1) :=
begin
  -- Proof omitted
  sorry
end

end quadratic_decreasing_condition_l693_693748


namespace triangle_area_is_13_l693_693392

def vector_a := ⟨7, 3⟩ : ℝ × ℝ
def vector_b := ⟨3, 5⟩ : ℝ × ℝ

theorem triangle_area_is_13 : 
  let matrix := [[vector_a.1, vector_b.1], [vector_a.2, vector_b.2]] in
  let determinant := (vector_a.1 * vector_b.2) - (vector_a.2 * vector_b.1) in
  abs determinant / 2 = 13 :=
by 
  let vector_a := (7, 3)
  let vector_b := (3, 5)
  let matrix := [[vector_a.1, vector_b.1], [vector_a.2, vector_b.2]]
  let determinant := (vector_a.1 * vector_b.2) - (vector_a.2 * vector_b.1)
  show abs determinant / 2 = 13
  sorry

end triangle_area_is_13_l693_693392


namespace D_equals_J_l693_693016

-- Definitions of parallel lines and collinear points
variables {A B C D E F G H I J : Type}
variables [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ D]
          [affine_space ℝ E] [affine_space ℝ F] [affine_space ℝ G] [affine_space ℝ H]
          [affine_space ℝ I] [affine_space ℝ J]

-- Assuming lines' parallelism
variables (ABC_triangle : affine_space ℝ (triangle A B C)) 
variables (DE_parallel_AB : parallel_line_segment (D, E) (A, B))
variables (EF_parallel_BC : parallel_line_segment (E, F) (B, C))
variables (FG_parallel_CA : parallel_line_segment (F, G) (C, A))
variables (GH_parallel_AB : parallel_line_segment (G, H) (A, B))
variables (HI_parallel_BC : parallel_line_segment (H, I) (B, C))
variables (IJ_parallel_CA : parallel_line_segment (I, J) (C, A))

-- Goal: To prove that the points D and J coincide.
theorem D_equals_J :
   D = J :=
by
sorry

end D_equals_J_l693_693016


namespace no_fractions_satisfy_condition_l693_693634

theorem no_fractions_satisfy_condition :
  ∀ (x y : ℕ), 
    x > 0 → y > 0 → Nat.gcd x y = 1 →
    (1.2 : ℚ) * (x : ℚ) / (y : ℚ) = (x + 2 : ℚ) / (y + 2 : ℚ) →
    False :=
by
  intros x y hx hy hrel hcond
  sorry

end no_fractions_satisfy_condition_l693_693634


namespace anna_not_lose_l693_693594

theorem anna_not_lose :
  ∀ (cards : Fin 9 → ℕ),
    ∃ (A B C D : ℕ),
      (A + B ≥ C + D) :=
by
  sorry

end anna_not_lose_l693_693594


namespace exponentiation_product_rule_l693_693131

theorem exponentiation_product_rule (a : ℝ) : (3 * a) ^ 2 = 9 * a ^ 2 :=
by
  sorry

end exponentiation_product_rule_l693_693131


namespace arithmetic_sequence_eighth_term_l693_693495

theorem arithmetic_sequence_eighth_term (a d : ℚ) 
  (h1 : 6 * a + 15 * d = 21) 
  (h2 : a + 6 * d = 8) : 
  a + 7 * d = 9 + 2/7 :=
by
  sorry

end arithmetic_sequence_eighth_term_l693_693495


namespace multiply_63_57_l693_693618

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l693_693618


namespace impossibility_of_arrangement_l693_693155

-- Definitions based on identified conditions in the problem
def isValidArrangement (arr : List ℕ) : Prop :=
  arr.length = 300 ∧
  (∀ i, i < 300 - 1 → arr.get i = |arr.get (i - 1) - arr.get (i + 1)|) ∧
  (arr.all (λ x => x > 0))

theorem impossibility_of_arrangement :
  ¬ (∃ arr : List ℕ, isValidArrangement arr) :=
sorry

end impossibility_of_arrangement_l693_693155


namespace son_work_rate_l693_693562

noncomputable def man_work_rate := 1/10
noncomputable def combined_work_rate := 1/5

theorem son_work_rate :
  ∃ S : ℝ, man_work_rate + S = combined_work_rate ∧ S = 1/10 := sorry

end son_work_rate_l693_693562


namespace muffins_apples_l693_693162

def apples_left_for_muffins (total_apples : ℕ) (pie_apples : ℕ) (refrigerator_apples : ℕ) : ℕ :=
  total_apples - (pie_apples + refrigerator_apples)

theorem muffins_apples (total_apples pie_apples refrigerator_apples : ℕ) (h_total : total_apples = 62) (h_pie : pie_apples = total_apples / 2) (h_refrigerator : refrigerator_apples = 25) : apples_left_for_muffins total_apples pie_apples refrigerator_apples = 6 := 
by 
  sorry

end muffins_apples_l693_693162


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693926

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693926


namespace product_of_fractions_gt_two_thirds_l693_693038

theorem product_of_fractions_gt_two_thirds :
  (∏ k in Finset.range 99 \ Finset.singleton 0 |+| 2, (k^3 - 1) / (k^3 + 1) > 2 / 3) :=
sorry

end product_of_fractions_gt_two_thirds_l693_693038


namespace largest_prime_factor_7fac_plus_8fac_l693_693949

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693949


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693885

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693885


namespace average_speed_round_trip_l693_693842

noncomputable def distance_AB : ℝ := 120
noncomputable def speed_AB : ℝ := 30
noncomputable def speed_BA : ℝ := 40

theorem average_speed_round_trip :
  (2 * distance_AB * speed_AB * speed_BA) / (distance_AB * (speed_AB + speed_BA)) = 34 := 
  by 
    sorry

end average_speed_round_trip_l693_693842


namespace value_of_m_squared_plus_2m_minus_3_l693_693295

theorem value_of_m_squared_plus_2m_minus_3 (m : ℤ) : 
  (∀ x : ℤ, 4 * (x - 1) - m * x + 6 = 8 → x = 3) →
  m^2 + 2 * m - 3 = 5 :=
by
  sorry

end value_of_m_squared_plus_2m_minus_3_l693_693295


namespace emily_garden_larger_l693_693775

-- Define the dimensions and conditions given in the problem
def john_length : ℕ := 30
def john_width : ℕ := 60
def emily_length : ℕ := 35
def emily_width : ℕ := 55

-- Define the effective area for John’s garden given the double space requirement
def john_usable_area : ℕ := (john_length * john_width) / 2

-- Define the total area for Emily’s garden
def emily_usable_area : ℕ := emily_length * emily_width

-- State the theorem to be proved
theorem emily_garden_larger : emily_usable_area - john_usable_area = 1025 :=
by
  sorry

end emily_garden_larger_l693_693775


namespace frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693092

theorem frequency_machine_A (total_A first_class_A : ℕ) (h_total_A: total_A = 200) (h_first_class_A: first_class_A = 150) :
  first_class_A / total_A = 3 / 4 := by
  rw [h_total_A, h_first_class_A]
  norm_num

theorem frequency_machine_B (total_B first_class_B : ℕ) (h_total_B: total_B = 200) (h_first_class_B: first_class_B = 120) :
  first_class_B / total_B = 3 / 5 := by
  rw [h_total_B, h_first_class_B]
  norm_num

theorem chi_square_test_significance (n a b c d : ℕ) (h_n: n = 400) (h_a: a = 150) (h_b: b = 50) 
  (h_c: c = 120) (h_d: d = 80) :
  let K_squared := (n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d)))
  in K_squared > 6.635 := by
  rw [h_n, h_a, h_b, h_c, h_d]
  let num := 400 * (150 * 80 - 50 * 120)^2
  let denom := (150 + 50) * (120 + 80) * (150 + 120) * (50 + 80)
  have : K_squared = num / denom := rfl
  norm_num at this
  sorry

end frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693092


namespace simplify_fraction_l693_693453

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end simplify_fraction_l693_693453


namespace most_winning_team_l693_693052

section MostWinningTeam

open List

-- Define the input conditions
variables {N : ℕ}
variables (team_names : list (list string))

-- Ensure that N is within the specified range.
def valid_N : Prop := 1 ≤ N ∧ N ≤ 30

-- Ensure each winning team name list consists of exactly three names.
def valid_team_names : Prop :=
  ∀ team, team ∈ team_names → length team = 3 ∧ (∀ name, name ∈ team → 1 ≤ length name ∧ length name ≤ 10)

-- Define a function to sort a team (list of names) alphabetically
def sort_team (team : list string) : list string :=
  sort (<=) team

-- Count occurrences of each unique team name
def count_occurrences (team : list string) (teams : list (list string)) : ℕ :=
  count (λ t, t ~ team) teams

-- The theorem to prove
theorem most_winning_team
  (hN : valid_N)
  (hteam : valid_team_names team_names) :
  ∃ team_wins : (list string) × ℕ,
    team_wins.2 = maximum (map (λ team, (team, count_occurrences team (map sort_team team_names))) (map sort_team team_names)).snd :=
begin
  admit,
end

end MostWinningTeam

end most_winning_team_l693_693052


namespace pool_filling_time_l693_693193

theorem pool_filling_time
    (V : ℝ)  -- Volume of the pool in liters
    (x : ℝ)  -- Flow rate of each hose in liters per hour
    (h_rate : 3 * x = V / 12)  -- Combined flow rate with three hoses filling the pool in 12 hours
    (h_stop_time : 11)  -- Time at which one hose stops working (11:00 a.m.)
    (h_remaining : 6 < h_stop_time) :  -- The hose stops after 6 hours (i.e., at 11:00 a.m.)
  let remaining_time := (V - (5 * 3 * x)) / (2 * x)
  in h_remaining_time : remaining_time = 10.5 → -- It takes 10.5 hours to fill the rest with 2 hoses
  11 + 10.5 = 21.5 :=  -- Pool will be filled 21.5 hours after start (9:30 p.m.)
  sorry

end pool_filling_time_l693_693193


namespace log_exp_identity_l693_693462

theorem log_exp_identity :
  log 3 18 - log 3 2 + Real.exp (Real.log 1) = 3 :=
by
  sorry

end log_exp_identity_l693_693462


namespace bobby_truck_gasoline_consumption_rate_l693_693606

variable {initial_gasoline : ℝ}
variable {final_gasoline : ℝ}
variable {dist_to_supermarket : ℝ}
variable {dist_to_farm : ℝ}
variable {dist_into_farm_trip : ℝ}
variable {returned_dist : ℝ}
variable {total_miles_driven : ℝ}
variable {total_gasoline_used : ℝ}
variable {rate_of_consumption : ℝ}

-- Conditions given in the problem
axiom initial_gasoline_is_12 : initial_gasoline = 12
axiom final_gasoline_is_2 : final_gasoline = 2
axiom dist_home_to_supermarket : dist_to_supermarket = 5
axiom dist_home_to_farm : dist_to_farm = 6
axiom dist_home_to_turnaround : dist_into_farm_trip = 2
axiom returned_distance : returned_dist = dist_into_farm_trip * 2

-- Distance calculations based on problem description
def dist_to_supermarket_round_trip : ℝ := dist_to_supermarket * 2
def dist_home_to_turnaround_round_trip : ℝ := returned_dist
def full_farm_trip : ℝ := dist_to_farm

-- Total Distance Calculation
axiom total_distance_is_22 : total_miles_driven = 
  dist_to_supermarket_round_trip + dist_home_to_turnaround_round_trip + full_farm_trip
axiom total_gasoline_used_is_10 : total_gasoline_used = initial_gasoline - final_gasoline

-- Question: Prove the rate of consumption is 2.2 miles per gallon
def rate_of_consumption_calculation (total_miles : ℝ) (total_gas : ℝ) : ℝ :=
  total_miles / total_gas

theorem bobby_truck_gasoline_consumption_rate :
    rate_of_consumption_calculation total_miles_driven total_gasoline_used = 2.2 := 
  sorry

end bobby_truck_gasoline_consumption_rate_l693_693606


namespace negation_of_universal_proposition_l693_693813

theorem negation_of_universal_proposition (x : ℝ) :
  (¬ (∀ x : ℝ, |x| < 0)) ↔ (∃ x_0 : ℝ, |x_0| ≥ 0) := by
  sorry

end negation_of_universal_proposition_l693_693813


namespace find_range_of_m_l693_693698

theorem find_range_of_m:
  (∀ x: ℝ, ¬ ∃ x: ℝ, x^2 + (m - 3) * x + 1 = 0) →
  (∀ y: ℝ, ¬ ∀ y: ℝ, x^2 + y^2 / (m - 1) = 1) → 
  1 < m ∧ m ≤ 2 :=
by
  sorry

end find_range_of_m_l693_693698


namespace first_digit_after_decimal_six_l693_693828

theorem first_digit_after_decimal_six (n : ℤ) (h : n ≥ 2) : 
  ∃ d : ℕ, d = 6 ∧ ((n : ℝ) + 0.6 < (n^3 + 2 * n^2 + n : ℝ)^(1 / 3) ∧ (n^3 + 2 * n^2 + n : ℝ)^(1 / 3) < (n : ℝ) + 0.7) :=
by
  have rational_root_approx : ((n^3 + 2 * n^2 + n : ℝ)^(1 / 3)) = 6.5 := sorry
  exact ⟨6, rational_root_approx, by sorry⟩

end first_digit_after_decimal_six_l693_693828


namespace sin_double_angle_value_l693_693325

theorem sin_double_angle_value (α : ℝ) (h₁ : Real.sin (π / 4 - α) = 3 / 5) (h₂ : 0 < α ∧ α < π / 4) : 
  Real.sin (2 * α) = 7 / 25 := 
sorry

end sin_double_angle_value_l693_693325


namespace parallelogram_area_l693_693415

theorem parallelogram_area (α : ℝ) (a b : ℝ) (hα: α = 150) (ha : a = 10) (hb : b = 20) :
  let h : ℝ := a * b * Real.sin (30 * Real.pi / 180) in
  h = 100 * Real.sqrt 3 :=
by 
  rw [hα, ha, hb, Real.sin_deg 30] 
  sorry

end parallelogram_area_l693_693415


namespace distance_between_points_MN_l693_693761

theorem distance_between_points_MN :
  let M := (1 : ℝ, 3 : ℝ)
  let N := (4 : ℝ, -1 : ℝ)
  dist M N = 5 := by
  sorry

end distance_between_points_MN_l693_693761


namespace smallest_a_value_l693_693790

noncomputable def minimum_a (a b : ℝ) := a ≥ 0 ∧ b ≥ 0 ∧ (∀ x : ℤ, sin (a * x + b) = sin (17 * x)) ∧ ∀ c : ℝ, (c ≥ 0 ∧ ∀ x : ℤ, sin (c * x + b) = sin (17 * x)) → a ≤ c

theorem smallest_a_value (a b : ℝ) (h : minimum_a a b) : a = 17 :=
by
  sorry

end smallest_a_value_l693_693790


namespace math_problem_correct_l693_693460

noncomputable def math_problem : Prop :=
  (1 / ((3 / (Real.sqrt 5 + 2)) - (1 / (Real.sqrt 4 + 1)))) = ((27 * Real.sqrt 5 + 57) / 40)

theorem math_problem_correct : math_problem := by
  sorry

end math_problem_correct_l693_693460


namespace ac_max_value_l693_693400

noncomputable def max_ac_value :=
  let a := 1010
  let b := 506
  let c := 505
  let d := -2021
  in (a, b, c, d)

theorem ac_max_value : 
  ∃ a b c d : ℤ, 
  a > b ∧ b > c ∧ c > d ∧ d ≥ -2021 ∧ (a + b) * (d + a) = (b + c) * (c + d) 
  ∧ b + c ≠ 0 ∧ d + a ≠ 0 
  ∧ a * c = 510050 := 
by 
  have h : ∃ a b c d : ℤ, 
    a = 1010 ∧ b = 506 ∧ c = 505 ∧ d = -2021 
    ∧ a > b 
    ∧ b > c 
    ∧ c > d 
    ∧ d ≥ -2021 
    ∧ (a + b) * (d + a) = (b + c) * (c + d) 
    ∧ b + c ≠ 0 
    ∧ d + a ≠ 0 := 
  by
    existsi (1010 : ℤ),
    existsi (506 : ℤ),
    existsi (505 : ℤ),
    existsi (-2021 : ℤ),
    refine ⟨rfl, rfl, rfl, rfl, _, _, _, _, _, _, _⟩,
    sorry,
  obtain ⟨a, b, c, d, ha, hb, hc, hd, he, hf, hg, hh⟩ := h,
  exact ⟨a, b, c, d, ha, hb, hc, hd, he, hf, hg, hh, rfl⟩

end ac_max_value_l693_693400


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693934

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693934


namespace largest_prime_factor_of_7fact_8fact_l693_693897

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693897


namespace double_persons_half_work_l693_693144

-- Definitions based on conditions
def persons : Type := ℕ
def work_days (P : persons) : ℕ := 24

-- Statement to prove based on the question and the correct answer
theorem double_persons_half_work {P : persons} (h : work_days P = 24) : work_days (2 * P) / 2 = 6 := 
by sorry

end double_persons_half_work_l693_693144


namespace problem1_part1_problem1_part2_l693_693845

noncomputable def f (x : ℝ) : ℝ := (1 / 2) - (1 / 2) * real.cos (2 * x)

theorem problem1_part1 : ∃ T > 0, ∀ x, f(x) = f(x + T) ∧ T = real.pi ∧ ∀ x, f(-x) = f(x) :=
begin
  have T_pos : ∃ T > 0, T = real.pi, by { use real.pi, norm_num },
  have even_f : ∀ x, f(-x) = f(x),
  {
    intro x,
    unfold f,
    rw real.cos_neg,
  },
  use [real.pi, real.pi_pos],
  split,
  { intro x, unfold f, rw real.cos_add, rw real.cos_pi },
  exact T_pos,
  exact even_f,
end

noncomputable def g (x : ℝ) : ℝ :=
if x ∈ [0, real.pi / 2] then (1 / 2) - f(x) else sorry

theorem problem1_part2 (x : ℝ) : x ∈ [-real.pi, 0] → g(x) =
  (if x ∈ [-real.pi / 2, 0] then -(1 / 2) * real.cos(2 * x) else (1 / 2) * real.cos(2 * x)) :=
begin
  sorry
end

end problem1_part1_problem1_part2_l693_693845


namespace smallest_five_digit_neg_int_congruent_to_2_mod_17_l693_693520

theorem smallest_five_digit_neg_int_congruent_to_2_mod_17 :
  ∃ x : ℤ, x < -9999 ∧ x ≡ 2 [MOD 17] ∧ ∀ y : ℤ, y < -9999 ∧ y ≡ 2 [MOD 17] → x ≤ y := 
begin
  use -10011,
  split,
  { norm_num, },
  split,
  { norm_num, exact rfl, },
  { intros y hy,
    cases hy with hy₁ hy₂,
    change y ≡ 2 [MOD 17] at hy₂,
    norm_num at hy₁,
    sorry, }
end

end smallest_five_digit_neg_int_congruent_to_2_mod_17_l693_693520


namespace sum_of_solutions_l693_693795

noncomputable def f : ℝ → ℝ := λ x, 12 * x + 5

def f_inv (x : ℝ) : ℝ := (x - 5) / 12

theorem sum_of_solutions : 
  let inv_comp (x : ℝ) := f_inv (f ((3 * x)⁻¹)) in 
  (∑ x in { x : ℝ | f_inv x = inv_comp x }.to_finset, id x) = 65 :=
by sorry

end sum_of_solutions_l693_693795


namespace radii_ratio_l693_693138

theorem radii_ratio (R1 R2 : ℝ) (α : ℝ) (O1 O2 : ℝ) (sinα : ℝ) 
  (area_quad : R1 * R2 * sinα) (area_tri : R2^2 * sinα * cos α)
  (h : area_quad / area_tri = 4 / 3) : R2 / R1 = 5 / 4 :=
by
  -- conditions
  have area_quad_eq : area_quad = R1 * R2 * sinα := sorry
  have area_tri_eq : area_tri = R2^2 * sinα * cos α := sorry

  -- calculation
  have ratio_eq : R1 / (R2 * cos α) = 4 / 3 := 
    by calc
      area_quad / area_tri = 4 / 3 : h
      (R1 * R2 * sinα) / (R2^2 * sinα * cos α) = 4 / 3 : by sorry
      R1 / (R2 * cos α) = 4 / 3 : by sorry
  
  -- solve for ratio
  have R1_eq : R1 = (4 / 3) * R2 * cos α := by sorry
  have R1_cos_eq : R1 * cos α = R2 := by sorry
  show R2 / R1 = 5 / 4 from by sorry


end radii_ratio_l693_693138


namespace speed_difference_is_42_l693_693191

/-- Lana and Spencer race conditions -/
def distance : ℝ := 12
def lana_time_minutes : ℝ := 15
def spencer_time_hours : ℝ := 2

/-- Convert Lana's travel time to hours -/
def lana_time_hours : ℝ := lana_time_minutes / 60

/-- Calculate Lana's speed -/
def lana_speed : ℝ := distance / lana_time_hours

/-- Calculate Spencer's speed -/
def spencer_speed : ℝ := distance / spencer_time_hours

/-- Prove the difference in speeds between Lana and Spencer -/
theorem speed_difference_is_42 :
  lana_speed - spencer_speed = 42 :=
sorry

end speed_difference_is_42_l693_693191


namespace impossible_300_numbers_l693_693149

theorem impossible_300_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) (hp : ∀ i, 0 < a i)
(hdiff : ∃ k, ∀ i ≠ k, a i = a ((i + 1) % n) - a ((i - 1 + n) % n)) 
: false :=
by {
  sorry
}

end impossible_300_numbers_l693_693149


namespace construct_parallel_equal_chord_l693_693529

-- Definitions for mathematical objects: circles, segments, chords, and parallelism
structure Circle := (center : Point) (radius : ℝ)
structure Segment := (start : Point) (end : Point)
def length (s : Segment) := dist s.start s.end
def is_parallel (a b : Segment) := sorry -- Define parallelism between segments

-- Given conditions
variables (S : Circle) (MN : Segment)

-- The statement of the problem
theorem construct_parallel_equal_chord :
  ∃ (A B : Point), (chord S A B) ∧ (Segment A B).length = length MN ∧ is_parallel (Segment A B) MN := 
sorry

end construct_parallel_equal_chord_l693_693529


namespace tan_pi_minus_alpha_l693_693683

theorem tan_pi_minus_alpha (α : ℝ) (h : Real.tan (Real.pi - α) = -2) : 
  (1 / (Real.cos (2 * α) + Real.cos α ^ 2) = -5 / 2) :=
by
  sorry

end tan_pi_minus_alpha_l693_693683


namespace common_difference_of_arithmetic_sequence_l693_693292

-- Define that a sequence is arithmetic if there exists a common difference
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
    ∀ n : ℕ, a n = a 0 + n * d

-- The main statement to prove
theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
    (h1 : is_arithmetic_sequence a d)
    (h2 : a 10 = 20)
    (h3 : ∑ i in Finset.range 10, a i = 1010) : d = 2 :=
by 
  sorry

end common_difference_of_arithmetic_sequence_l693_693292


namespace ceil_neg_3_7_l693_693230

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := by
  sorry

end ceil_neg_3_7_l693_693230


namespace digit_sum_10_pow_93_minus_937_l693_693719

-- Define a function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem digit_sum_10_pow_93_minus_937 :
  sum_of_digits (10^93 - 937) = 819 :=
by
  sorry

end digit_sum_10_pow_93_minus_937_l693_693719


namespace area_of_triangle_l693_693061

-- Define a structure for the triangle with necessary parameters
structure Triangle :=
(height : ℝ)
(base_part1 : ℝ)
(base_part2 : ℝ)
(angle_ratio : ℕ × ℕ)

def triangle_data : Triangle := {
  height := 2,
  base_part1 := 1,
  base_part2 := 5 / 3,
  angle_ratio := (2, 1)
}

-- Define the statement that proves the area of the triangle is 8/3 cm^2
theorem area_of_triangle (T : Triangle) (h : T.height = 2) (bp1 : T.base_part1 = 1) 
  (bp2 : T.base_part2 = 5 / 3) (ar : T.angle_ratio = (2,1)) : 
  (1 / 2) * (T.base_part1 + T.base_part2) * T.height = 8 / 3 :=
by {
  sorry
}

end area_of_triangle_l693_693061


namespace at_least_one_on_side_l693_693587

noncomputable def quadrilateral : Type := { A B C D : Point // is_convex A B C D }
def altitude_foot (A B C : Point) : Point := sorry -- Assume given functions
noncomputable def points_on_sides (A B C D : Point) : Prop := sorry -- Assume given condition

theorem at_least_one_on_side (A B C D : Point) (h : is_convex A B C D) :
  ∃ P, P ∈ {altitude_foot A B C, altitude_foot B C D, altitude_foot C D A, altitude_foot D A B,
           altitude_foot A C B, altitude_foot B D C, altitude_foot C A D, altitude_foot D B A,
           altitude_foot A B D, altitude_foot B C A, altitude_foot C D B, altitude_foot D A C} ∧ points_on_sides A B C D :=
begin
  -- proof omitted
  sorry
end

end at_least_one_on_side_l693_693587


namespace largest_prime_factor_of_fact_sum_is_7_l693_693968

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693968


namespace number_reported_2010th_l693_693669

noncomputable def seq (n : ℕ) : ℕ :=
if n = 0 then 2 else
if n = 1 then 3 else
(nat.of_digits 10 [seq (n - 1) * seq (n - 2)]) % 10

theorem number_reported_2010th : seq 2009 = 4 :=
sorry

end number_reported_2010th_l693_693669


namespace irrational_greater_than_2_l693_693297

def floor (x : ℝ) : ℤ := 
  ⌊x⌋

def A (x : ℝ) : set ℤ := 
  {n | ∃ (k : ℕ), k > 0 ∧ n = floor (k * x)}

theorem irrational_greater_than_2 (α : ℝ) (h_irr : irrational α) (h_gt_1 : 1 < α) :
  (∀ β : ℝ, 0 < β → A α ⊇ A β → ∃ (m : ℕ), β = m * α) ↔ (2 < α) := 
sorry

end irrational_greater_than_2_l693_693297


namespace nikita_productivity_l693_693084

theorem nikita_productivity 
  (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 7) 
  (h2 : 5 * x + 3 * y = 11) : 
  y = 2 := 
sorry

end nikita_productivity_l693_693084


namespace air_quality_levels_probabilities_average_number_of_people_exercising_contingency_table_and_relation_l693_693579

def air_quality_survey :=
  let excellent := (2 + 16 + 25)
  let good := (5 + 10 + 12)
  let mild_pollution := (6 + 7 + 8)
  let moderate_pollution := (7 + 2 + 0)
  let total_days := 100
  let prob_excellent := excellent / total_days
  let prob_good := good / total_days
  let prob_mild_pollution := mild_pollution / total_days
  let prob_moderate_pollution := moderate_pollution / total_days
  let average_exercise := 
    (1 / total_days) * (100 * (2 + 5 + 6 + 7) + 300 * (16 + 10 + 7 + 2) + 500 * (25 + 12 + 8))
  let table :=
    (33, 37, 70, 22, 8, 30, 55, 45, 100)
  let K_squared :=
    (total_days * (33 * 8 - 37 * 22)^2) / (70 * 30 * 55 * 45)
  let check_confidence :=
    K_squared > 3.841
  (prob_excellent, prob_good, prob_mild_pollution, prob_moderate_pollution, average_exercise, table, check_confidence)

theorem air_quality_levels_probabilities :
  let (prob_excellent, prob_good, prob_mild_pollution, prob_moderate_pollution, _, _, _) := air_quality_survey in
  prob_excellent = 0.43 ∧
  prob_good = 0.27 ∧
  prob_mild_pollution = 0.21 ∧
  prob_moderate_pollution = 0.09 := by
  sorry

theorem average_number_of_people_exercising :
  let (_, _, _, _, average_exercise, _, _) := air_quality_survey in
  average_exercise = 350 := by
  sorry

theorem contingency_table_and_relation :
  let (_, _, _, _, _, (ge_400, gt_400, total_g, le_400, lt_400, total_p, total_le_400, total_gt_400, total_t), check_confidence) := air_quality_survey in
  ge_400 = 33 ∧
  gt_400 = 37 ∧
  total_g = 70 ∧
  le_400 = 22 ∧
  lt_400 = 8 ∧
  total_p = 30 ∧
  total_le_400 = 55 ∧
  total_gt_400 = 45 ∧
  total_t = 100 ∧
  check_confidence = true := by
  sorry

end air_quality_levels_probabilities_average_number_of_people_exercising_contingency_table_and_relation_l693_693579


namespace binomial_sum_mod_p_l693_693397

theorem binomial_sum_mod_p (p : ℕ) [hp : Fact (Nat.Prime p)] (hp_gt3 : 3 < p) :
  let q := (p - 1) / 2
  in ∑ i in Finset.range (q + 1), Nat.choose (2 * i) i % p = 1 ∨ 
     ∑ i in Finset.range (q + 1), Nat.choose (2 * i) i % p = p - 1  :=
by
  sorry -- Proof goes here

end binomial_sum_mod_p_l693_693397


namespace smallest_whole_number_larger_than_sum_l693_693665

theorem smallest_whole_number_larger_than_sum : 
  ∃ n : ℕ, n > 3 + 4 + 5 + 6 + (1 / 7) + (1 / 8) + (1 / 9) + (1 / 10) ∧ ∀ m : ℕ, m > 3 + 4 + 5 + 6 + (1 / 7) + (1 / 8) + (1 / 9) + (1 / 10) → n ≤ m :=
  ∃ n : ℕ, n = 19 := 
sorry

end smallest_whole_number_larger_than_sum_l693_693665


namespace cheryl_material_left_l693_693612

theorem cheryl_material_left (
  h₁ : (5 : ℚ) / 11 + 2 / 3 = 37 / 33) 
  (h₂ : 0.6666666666666665 = 2 / 3 : ℚ)
  (total_used : 2 / 3 : ℚ)
  (h₃ : (37 / 33) - total_used = 5 / 11) : 
  ((5 : ℚ) / 11 + 2 / 3 - 2 / 3 = 5 / 11) :=
by {
  subst h₂,
  rw h₁,
  exact h₃,
}

end cheryl_material_left_l693_693612


namespace find_D_E_l693_693477

/--
Consider the circle given by \( x^2 + y^2 + D \cdot x + E \cdot y + F = 0 \) that is symmetrical with
respect to the line \( l_1: x - y + 4 = 0 \) and the line \( l_2: x + 3y = 0 \). Prove that the values 
of \( D \) and \( E \) are \( 12 \) and \( -4 \), respectively.
-/
theorem find_D_E (D E F : ℝ) (h1 : -D/2 + E/2 + 4 = 0) (h2 : -D/2 - 3*E/2 = 0) : D = 12 ∧ E = -4 :=
by
  sorry

end find_D_E_l693_693477


namespace sum_100_terms_sequence_l693_693573

-- Definition of the sequence using the given conditions
def sequence_term (n : ℕ) : ℤ :=
  (-1)^((n : ℤ) // nat.factorial n) * (((n : ℤ) ∈ set.range nat.factorial) sup (n : ℕ - nat.factorial n))

noncomputable def sum_first_100_terms : ℤ :=
  ∑ i in finset.range 100, sequence_term (i + 1)

theorem sum_100_terms_sequence : sum_first_100_terms = 35 :=
by
  sorry

end sum_100_terms_sequence_l693_693573


namespace smallest_m_to_mark_all_squares_l693_693468

theorem smallest_m_to_mark_all_squares (n : ℕ) :
  ∃ m, (m = (n - 1)^2 + 1) :=
begin
  -- We'll specify the steps in our proof (not necessary to fill, because we only need the statement)
  sorry
end

end smallest_m_to_mark_all_squares_l693_693468


namespace min_square_sum_l693_693294

-- Given conditions:
variables {a b : ℝ}
hypothesis h1 : a^2 + 2 * a * b - 3 * b^2 = 1

-- The problem to prove:
theorem min_square_sum : a^2 + b^2 ≥ (Real.sqrt 5 + 1) / 4 := by
  sorry

end min_square_sum_l693_693294


namespace cattle_transport_problem_l693_693172

noncomputable def truck_capacity 
    (total_cattle : ℕ)
    (distance_one_way : ℕ)
    (speed : ℕ)
    (total_time : ℕ) : ℕ :=
  total_cattle / (total_time / ((distance_one_way * 2) / speed))

theorem cattle_transport_problem :
  truck_capacity 400 60 60 40 = 20 := by
  -- The theorem statement follows the structure from the conditions and question
  sorry

end cattle_transport_problem_l693_693172


namespace semicircle_circumference_approx_l693_693145

noncomputable def rectangle_length : ℝ := 9
noncomputable def rectangle_breadth : ℝ := 6
noncomputable def square_perimeter : ℝ := 2 * (rectangle_length + rectangle_breadth)
noncomputable def square_side : ℝ := square_perimeter / 4
noncomputable def semicircle_diameter : ℝ := square_side
noncomputable def semicircle_circumference : ℝ := (Real.pi * semicircle_diameter) / 2 + semicircle_diameter

theorem semicircle_circumference_approx :
  (semicircle_circumference : ℝ) ≈ 19.28 :=
sorry

end semicircle_circumference_approx_l693_693145


namespace angle_ACB_obtuse_l693_693034

variable {α : Type _} [LinearOrderedField α] 

theorem angle_ACB_obtuse
  {A B C D : EuclideanGeometry.Point α}
  (h_distinct: A ≠ D ∧ B ≠ D ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (h_on_AB: EuclideanGeometry.collinear {A, B, D})
  (h_ratio : EuclideanGeometry.distance A D / EuclideanGeometry.distance D C = 
             EuclideanGeometry.distance A B / EuclideanGeometry.distance B C) : 
  ∠ A C B > π / 2 :=
by
  sorry

end angle_ACB_obtuse_l693_693034


namespace min_value_of_x_sq_plus_6x_l693_693125

theorem min_value_of_x_sq_plus_6x : ∃ x : ℝ, ∀ y : ℝ, y^2 + 6*y ≥ -9 :=
by
  sorry

end min_value_of_x_sq_plus_6x_l693_693125


namespace probability_exactly_one_shot_l693_693868

open ProbabilityTheory

variable (Ω : Type) [ProbabilitySpace Ω]
variables (A B : Event Ω)

-- Given conditions
def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.6

-- Prove the goal
theorem probability_exactly_one_shot :
  (ℙ[A] = prob_A) →
  (ℙ[B] = prob_B) →
  (ℙ[A ∧ (¬B)] + ℙ[(¬A) ∧ B] = 0.44) :=
by
  intros hA hB
  sorry

end probability_exactly_one_shot_l693_693868


namespace Damien_jogs_miles_over_three_weeks_l693_693211

theorem Damien_jogs_miles_over_three_weeks :
  (5 * 5) * 3 = 75 :=
by sorry

end Damien_jogs_miles_over_three_weeks_l693_693211


namespace sum_of_intervals_eq_one_l693_693259

def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def g (x : ℝ) : ℝ := 
  let n := floor x
  n * (2015 ^ (x - n) - 1)

theorem sum_of_intervals_eq_one :
  ∑ n in Finset.range 2014, Real.log 2015.toReal ((n + 1 : ℝ) / (n : ℝ)) = 1 :=
by
  sorry

end sum_of_intervals_eq_one_l693_693259


namespace simplify_fraction_l693_693437

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end simplify_fraction_l693_693437


namespace geometric_sequence_product_l693_693315

theorem geometric_sequence_product
  (x y z : ℝ)
  (geo_seq : ∀ (a b : ℕ), b = a + 1 → (a = 0 ∧ b = 1 ∧ y = x) ∨ (a = 1 ∧ b = 2 ∧ z = y * x) ∨ (a = 3 ∧ b = 4 ∧ -2 = z * y) ∨ (a = 4 ∧ b = 0 ∧ -1 = -2 / y))
  (y_squared : y ^ 2 = 2) :
  x * y * z = 2 * real.sqrt 2 :=
by
  sorry

end geometric_sequence_product_l693_693315


namespace max_piece_length_l693_693637

theorem max_piece_length (a b c : ℕ) (h1 : a = 60) (h2 : b = 75) (h3 : c = 90) :
  Nat.gcd (Nat.gcd a b) c = 15 :=
by 
  sorry

end max_piece_length_l693_693637


namespace max_value_of_f_l693_693214

def f (t : ℝ) : ℝ :=
  ((3 ^ t - 4 * t) * t) / (9 ^ t)

theorem max_value_of_f :
  ∃ t ∈ ℝ, f t = 1 / 16 :=
by
  sorry

end max_value_of_f_l693_693214


namespace train_arrival_time_l693_693185

  constant initial_arrival_time : Nat := 11 * 60 + 40  -- 11:40 in minutes
  constant delay_time : Nat := 25  -- 25 minutes

  -- exact_arrival_time calculates the new arrival time
  def exact_arrival_time : Nat := initial_arrival_time + delay_time

  -- Convert times back to the "hours and minutes" format for better readability (optional)
  def time_in_hours_and_minutes (time_in_minutes : Nat) : (Nat × Nat) :=
    (time_in_minutes / 60, time_in_minutes % 60)

  theorem train_arrival_time :
    time_in_hours_and_minutes exact_arrival_time = (12, 5) :=
  by
    sorry
  
end train_arrival_time_l693_693185


namespace net_difference_in_expenditure_l693_693146

variable (P A : ℝ)

def condition1 : ℝ := P * 1.25
def condition2 : ℝ := condition1 * 0.7 * A
def condition3 : ℝ := P * A

theorem net_difference_in_expenditure
  (h1 : P > 0)
  (h2 : A > 0) :
  ∃ net_difference : ℝ, net_difference = |condition2 - condition3| ∧ net_difference = 0.125 * P * A := 
by sorry

end net_difference_in_expenditure_l693_693146


namespace problem1_problem2_l693_693301

variable {ℝ : Type*} [LinearOrderedField ℝ]

def fun_derivative_condition (f: ℝ → ℝ) : Prop :=
  ∀ x, 0 < deriv f x ∧ deriv f x < 1

def fun_fixed_point (f: ℝ → ℝ) (α : ℝ) : Prop :=
  f α = α

theorem problem1 (f: ℝ → ℝ) (α: ℝ)
  (H1: fun_derivative_condition f)
  (H2: fun_fixed_point f α) :
  ∀ x, x > α → x > f x :=
sorry

theorem problem2 (f: ℝ → ℝ) (I: set ℝ) (α : ℝ)
  (H1: fun_derivative_condition f)
  (H2: fun_fixed_point f α)
  (H3: ∀ a b, (a ∈ I ∧ b ∈ I) → ∃ x, x ∈ I ∧ f(b) - f(a) = (b - a) * deriv f x):
  ∀ β, f β = β → β = α :=
sorry

end problem1_problem2_l693_693301


namespace base_eight_to_base_ten_l693_693882

theorem base_eight_to_base_ten : ∃ n : ℕ, 47 = 4 * 8 + 7 ∧ n = 39 :=
by
  sorry

end base_eight_to_base_ten_l693_693882


namespace min_sum_of_first_2012_terms_l693_693212

def is_absolute_sum_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, |a (n + 1)| + |a n| = d

def a : ℕ → ℝ := sorry -- assumed sequence will be defined as part of the proof

theorem min_sum_of_first_2012_terms :
  is_absolute_sum_sequence a 2 → a 1 = 2 → (finset.sum (finset.range 2012) a) = -2008 :=
by
  sorry

end min_sum_of_first_2012_terms_l693_693212


namespace polynomial_properties_l693_693876

noncomputable def p (x : ℝ) : ℝ := x^4 - 4 * x^3 - 26 * x^2 + 60 * x + 225

theorem polynomial_properties : 
  (p(-3) = 0) ∧ (p(5) = 0) ∧ (∀ x : ℝ, ∃ y : ℝ, (p'(x) = 0) → (p''(x) ≥ 0)) ∧ (∀ x : ℝ, p(x+1) = p(-x-1)) ∧ (∀ x : ℝ, p(x+1) ≤ 256) := 
by 
  sorry

end polynomial_properties_l693_693876


namespace geometric_sequence_third_term_l693_693555

theorem geometric_sequence_third_term (r : ℕ) (h_r : 5 * r ^ 4 = 1620) : 5 * r ^ 2 = 180 := by
  sorry

end geometric_sequence_third_term_l693_693555


namespace math_problem_l693_693680

theorem math_problem (x y : ℝ) :
  let A := x^3 + 3*x^2*y + y^3 - 3*x*y^2
  let B := x^2*y - x*y^2
  A - 3*B = x^3 + y^3 := by
  sorry

end math_problem_l693_693680


namespace sum_first_mk_terms_l693_693695

variable {n m k : ℕ} (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m k : ℕ, (n ≤ m ∧ m ≤ k) → a n - a m = a m - a k

-- Given conditions
variable (h1 : a m = 1 / k)
variable (h2 : a k = 1 / m)
variable (h_arith : is_arithmetic_sequence a)

-- The final proof goal
theorem sum_first_mk_terms : 
  (∑ i in finset.range (m * k), a i) = (1 + k * m) / 2 :=
  sorry

end sum_first_mk_terms_l693_693695


namespace positive_integer_n_prime_l693_693262

theorem positive_integer_n_prime :
  ∃! (n : ℕ), (n > 0) ∧ Prime (-n^4 + n^3 - 4*n^2 + 18*n - 19) :=
sorry

end positive_integer_n_prime_l693_693262


namespace nathalie_more_points_than_lizzie_l693_693353

theorem nathalie_more_points_than_lizzie :
  (L N A T : ℕ) → (x : ℕ) →
  L = 4 →
  N = 4 + x →
  A = 2 * (L + N) →
  T = 50 →
  (T = L + N + A + 17) →
  x = 3 :=
by
  intros L N A T x hL hN hA hT hTotal
  rw [hL] at hN
  rw [hL, hN] at hA
  rw [hL, hN, hA] at hTotal
  sorry

end nathalie_more_points_than_lizzie_l693_693353


namespace quadratic_real_roots_probability_l693_693465

theorem quadratic_real_roots_probability :
  let outcomes := (fin_fun (λ n : Fin 36, (1, 1) * (n + 1))),
  let B_C := finset.product (finset.of_finite outcomes) (finset.of_finite outcomes),
  let real_roots_count := B_C.filter (λ bc, let ⟨b, c⟩ := bc in b^2 - 4 * c ≥ 0),
  (real_roots_count.card : ℝ) / (B_C.card : ℝ) = 19 / 36 :=
by sorry

end quadratic_real_roots_probability_l693_693465


namespace tan_alpha_two_implies_fraction_eq_three_fourths_l693_693740

variable {α : ℝ}

theorem tan_alpha_two_implies_fraction_eq_three_fourths (h1 : Real.tan α = 2) (h2 : Real.cos α ≠ 0) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

end tan_alpha_two_implies_fraction_eq_three_fourths_l693_693740


namespace division_result_l693_693219

def expr := 180 / (12 + 13 * 2)

theorem division_result : expr = 90 / 19 := by
  sorry

end division_result_l693_693219


namespace fuel_first_third_l693_693806

-- Defining constants based on conditions
def total_fuel := 60
def fuel_second_third := total_fuel / 3
def fuel_final_third := fuel_second_third / 2

-- Defining what we need to prove
theorem fuel_first_third :
  total_fuel - (fuel_second_third + fuel_final_third) = 30 :=
by
  sorry

end fuel_first_third_l693_693806


namespace dilation_image_l693_693841

-- Define the given conditions
def center : ℂ := 2 - 3 * complex.i
def scale_factor : ℝ := 3
def initial_point : ℂ := - complex.i

-- Define the dilation function
def dilation (c : ℂ) (k : ℝ) (z : ℂ) : ℂ := c + k * (z - c)

-- State the theorem we need to prove
theorem dilation_image : dilation center scale_factor initial_point = -4 + 3 * complex.i :=
by
  -- The proof is omitted
  sorry

end dilation_image_l693_693841


namespace max_dot_product_l693_693322

variables (x y z : ℝ)

def a : ℝ × ℝ × ℝ := (1, 1, -2)
def b : ℝ × ℝ × ℝ := (x, y, z)

theorem max_dot_product (h : x^2 + y^2 + z^2 = 16) :
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 ≤ 4 * real.sqrt 6 := 
by
  sorry

end max_dot_product_l693_693322


namespace largest_prime_factor_l693_693980

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693980


namespace om_perpendicular_to_cc1_iff_sides_condition_l693_693799

variable {A B C O M : Type}
variable [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] -- Assuming they reside in real inner product spaces
variable (triangle : Triangle A B C)

-- Let O be the circumcenter of a triangle and M be the centroid.
def circumcenter (t : Triangle A B C) : Point ℝ :=
  sorry -- Definition of circumcenter

def centroid (t : Triangle A B C) : Point ℝ :=
  sorry -- Definition of centroid

-- Definitions for the sides' lengths of the triangle
def length (a b : Point ℝ) : ℝ :=
  dist a b

-- Definitions for the sides of the triangle
def a : ℝ := length B C
def b : ℝ := length C A
def c : ℝ := length A B

-- Let O be the circumcenter and M be the centroid
variable (O_is_circumcenter : O = circumcenter triangle)
variable (M_is_centroid : M = centroid triangle)

-- Define the condition a^2 + b^2 = 2c^2
variable (sides_condition : a^2 + b^2 = 2 * c^2)

-- Define the proof problem
theorem om_perpendicular_to_cc1_iff_sides_condition :
  let O := circumcenter triangle in
  let M := centroid triangle in
  (is_perpendicular (line O M) (median C)) ↔ (a^2 + b^2 = 2 * c^2) :=
sorry -- Proof not required

end om_perpendicular_to_cc1_iff_sides_condition_l693_693799


namespace fourth_column_rectangles_l693_693596

-- Definitions based on the conditions
def grid_size : ℕ := 7
def fourth_column := { i : ℕ // i < grid_size }

def is_rectangle (rect : set (ℕ × ℕ)) : Prop :=
  ∃ (a b c d : ℕ), a < c ∧ b < d ∧ rect = { p | p.1 ≥ a ∧ p.1 < c ∧ p.2 ≥ b ∧ p.2 < d }

def rectangles (rect : set (ℕ × ℕ)) (n : ℕ) : Prop :=
  is_rectangle rect ∧ ∃ (area : ℕ), n = (area : ℤ)

def belongs_to_rectangles (squares : finset (ℕ × ℕ)) (column : ℕ) (count : ℕ) : Prop :=
  ∃ rects : finset (set (ℕ × ℕ)),
    (∀ square ∈ squares, ∃ rect ∈ rects, square ∈ rect) ∧
    rects.card = count ∧
    ∀ rect ∈ rects, is_rectangle rect

-- Problem statement in Lean 4
theorem fourth_column_rectangles :
  ∃ rect_count : ℕ,
    rect_count = 4 ∧
    belongs_to_rectangles (finset.filter (λ p, p.2 = 3) ((finset.range 7).product (finset.range 7))) 3 rect_count :=
by {
  sorry
}

end fourth_column_rectangles_l693_693596


namespace correct_number_of_arrangements_l693_693265

-- Define the set of four people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

open Person

-- Define a function checking if a list of persons satisfies the conditions
def valid_arrangement (l : List Person) : Prop :=
  List.length l = 4 ∧
  (l.get! 0 ≠ A) ∧
  (l.get! 3 ≠ B)

-- Function to count valid arrangements of persons
def count_valid_arrangements : ℕ :=
  List.length (List.filter valid_arrangement (List.permutations [A, B, C, D]))

-- Theorem stating the number of valid arrangements is 14
theorem correct_number_of_arrangements : count_valid_arrangements = 14 :=
  by
    sorry

end correct_number_of_arrangements_l693_693265


namespace tan_period_l693_693852

noncomputable def smallest_positive_period (f : ℝ → ℝ) : ℝ :=
  Inf { T | T > 0 ∧ ∀ x, f (x + T) = f x }

def tan_function (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 4)

theorem tan_period : smallest_positive_period tan_function = Real.pi / 2 :=
  by
    sorry

end tan_period_l693_693852


namespace largest_prime_factor_7fac_8fac_l693_693942

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693942


namespace buckets_required_l693_693501

theorem buckets_required (C : ℝ) (hC : C > 0) :
  let original_bucket_count := 25
  let reduction_factor := 2 / 5
  let new_bucket_count := original_bucket_count / reduction_factor
  new_bucket_count.ceil = 63 :=
by
  -- Definitions and conditions
  let original_bucket_count := 25
  let reduction_factor := 2 / 5
  let total_capacity := original_bucket_count * C
  let new_bucket_capacity := reduction_factor * C
  let new_bucket_count := total_capacity / new_bucket_capacity
  
  -- Main goal
  have : new_bucket_count = (25 * C) / ((2 / 5) * C) := by sorry
  have : new_bucket_count = 25 / (2 / 5) := by sorry
  have : new_bucket_count = 25 * (5 \ 2) := by sorry
  have : new_bucket_count = 62.5 := by sorry
  exact ceil_eq 63 _.mpr sorry

end buckets_required_l693_693501


namespace arrangements_ABC_together_l693_693856

noncomputable def permutation_count_ABC_together (n : Nat) (unit_size : Nat) (remaining : Nat) : Nat :=
  (Nat.factorial unit_size) * (Nat.factorial (remaining + 1))

theorem arrangements_ABC_together : permutation_count_ABC_together 6 3 3 = 144 :=
by
  sorry

end arrangements_ABC_together_l693_693856


namespace largest_prime_factor_of_fact_sum_is_7_l693_693972

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693972


namespace find_team_with_most_wins_l693_693050

def team_won_most (n : ℕ) (results : List (List String)) : String :=
  let sorted_teams := results.map fun team => team.sort
  let unique_teams := sorted_teams.eraseDup
  let counts := unique_teams.map fun team => (team, sorted_teams.count (· == team))
  let max_team := counts.maximumBy (·.snd)
  max_team.fst.intercalate " " ++ ":" ++ toString max_team.snd

theorem find_team_with_most_wins (n : ℕ) 
    (hn : 1 ≤ n ∧ n ≤ 30)
    (results : List (List String)) 
    (hresults : ∀ team ∈ results, team.length = 3 ∧ ∀ name ∈ team, 1 ≤ name.length ∧ name.length ≤ 10)  :
    ∃ team : String, ∃ wins : ℕ, (team_won_most n results = team ++ ":" ++ toString wins) :=
by
  sorry

end find_team_with_most_wins_l693_693050


namespace gray_region_area_l693_693636

theorem gray_region_area (r : ℝ) : 
  let inner_circle_radius := r
  let outer_circle_radius := r + 3
  let inner_circle_area := Real.pi * (r ^ 2)
  let outer_circle_area := Real.pi * ((r + 3) ^ 2)
  let gray_region_area := outer_circle_area - inner_circle_area
  gray_region_area = 6 * Real.pi * r + 9 * Real.pi := 
by
  sorry

end gray_region_area_l693_693636


namespace merchants_and_cost_l693_693432

theorem merchants_and_cost (n C : ℕ) (h1 : 8 * n = C + 3) (h2 : 7 * n = C - 4) : n = 7 ∧ C = 53 := 
by 
  sorry

end merchants_and_cost_l693_693432


namespace feet_more_than_heads_l693_693355

def num_hens := 50
def num_goats := 45
def num_camels := 8
def num_keepers := 15

def feet_per_hen := 2
def feet_per_goat := 4
def feet_per_camel := 4
def feet_per_keeper := 2

def total_heads := num_hens + num_goats + num_camels + num_keepers
def total_feet := (num_hens * feet_per_hen) + (num_goats * feet_per_goat) + (num_camels * feet_per_camel) + (num_keepers * feet_per_keeper)

-- Theorem to prove:
theorem feet_more_than_heads : total_feet - total_heads = 224 := by
  -- proof goes here
  sorry

end feet_more_than_heads_l693_693355


namespace largest_prime_factor_7fac_plus_8fac_l693_693958

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693958


namespace exists_subset_S_l693_693639

def F : Set (Fin 2006 → ℤ) := {f | ∀ i, f i = 1 ∨ f i = -1}

noncomputable def exists_S (S : Set (Fin 2006 → ℤ)) [F S] [∃ S, S.card = 2006 ∧ (∀ (a : Fin 2006 → ℤ), F a → ∃ b : Fin 2006 → ℤ, S b ∧ ∑ i, a i * b i = 0)] : Prop := 
S.card = 2006 ∧ ∀ (a : Fin 2006 → ℤ), F a → ∃ b : Fin 2006 → ℤ, S b ∧ ∑ i, a i * b i = 0

theorem exists_subset_S : ∃ S, exists_S S :=
sorry

end exists_subset_S_l693_693639


namespace complex_coordinates_l693_693763

theorem complex_coordinates : (⟨(-1:ℝ), (-1:ℝ)⟩ : ℂ) = (⟨0,1⟩ : ℂ) * (⟨-2,0⟩ : ℂ) / (⟨1,1⟩ : ℂ) :=
by
  sorry

end complex_coordinates_l693_693763


namespace correct_statements_l693_693711

-- Definitions according to the problem statement
def is_radius (s : ℝ → set ℝ) (r : ℝ) := 
  ∀ x, x ∈ s r → (x - r) * (x - r) = r * r

def is_diameter (s : ℝ → set ℝ) (d : ℝ) := 
  ∀ x y, x ∈ s d → y ∈ s d → (x - y) * (x - y) = d * d

def cut_sphere (s : ℝ → set ℝ) (plane : set ℝ) := 
  ∃ c, c ∈ plane ∧ (∀ x y, x ∈ s c → y ∈ plane → (x - y) * (x - y) = c * c)

def sphere_rep (s : ℝ → set ℝ) (c : ℝ) :=
  s = λ r, {x | (x - c) * (x - c) = r * r}

-- Proof goals
theorem correct_statements (s : ℝ → set ℝ) (r d c : ℝ) (plane : set ℝ) :
  is_radius s r ∧
  ¬ is_diameter s d ∧
  cut_sphere s plane ∧
  sphere_rep s c :=
sorry

end correct_statements_l693_693711


namespace fuel_for_first_third_l693_693803

def total_fuel : ℕ := 60
def fuel_second_third : ℕ := total_fuel / 3
def fuel_final_third : ℕ := fuel_second_third / 2
def fuel_first_third : ℕ := total_fuel - fuel_second_third - fuel_final_third

theorem fuel_for_first_third (total_fuel : ℕ) : 
  (total_fuel = 60) → 
  (fuel_first_third = total_fuel - (total_fuel / 3) - (total_fuel / 6)) →
  fuel_first_third = 30 := 
by 
  intros h1 h2
  rw h1 at h2
  norm_num at h2
  exact h2

end fuel_for_first_third_l693_693803


namespace product_quality_difference_l693_693085

variable (n a b c d : ℕ)
variable (P_K_2 : ℝ → ℝ)

def first_class_freq_A := a / (a + b : ℕ)
def first_class_freq_B := c / (c + d : ℕ)

def K2 := (n : ℝ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem product_quality_difference
  (ha : a = 150) (hb : b = 50) 
  (hc : c = 120) (hd : d = 80)
  (hn : n = 400)
  (hK : P_K_2 0.010 = 6.635) : 
  first_class_freq_A a b = 3 / 4 ∧
  first_class_freq_B c d = 3 / 5 ∧
  K2 n a b c d > P_K_2 0.010 :=
by {
  sorry
}

end product_quality_difference_l693_693085


namespace simplify_fraction_l693_693456

theorem simplify_fraction : ∃ (a b : ℕ), a = 90 ∧ b = 150 ∧ (90:ℚ) / (150:ℚ) = (3:ℚ) / (5:ℚ) :=
by {
  use 90,
  use 150,
  split,
  refl,
  split,
  refl,
  sorry,
}

end simplify_fraction_l693_693456


namespace junior_score_is_90_l693_693754

variables (n : ℕ) -- total number of students
variables (j_score s_avg : ℝ) -- score of juniors, average score of seniors

-- Conditions
def pct_juniors := 0.2 * n
def pct_seniors := 0.8 * n
def avg_class_score := 78
def same_score_juniors := ∀ j : ℕ, j < pct_juniors → j_score = j_score -- all juniors have the same score
def avg_senior_score := 75

-- To prove: score of each junior is 90
theorem junior_score_is_90
  (H1 : pct_juniors = 0.2 * n)
  (H2 : pct_seniors = 0.8 * n)
  (H3 : avg_class_score = 78)
  (H4 : same_score_juniors)
  (H5 : avg_senior_score = 75) :
  j_score = 90 :=
sorry

end junior_score_is_90_l693_693754


namespace number_of_lemons_l693_693522

theorem number_of_lemons (
  (price_increase_per_lemon : ℝ) 
  (planned_lemon_price : ℝ) 
  (planned_grape_price : ℝ)
  (num_grapes : ℕ)
  (total_money_collected : ℝ)
  (new_lemon_price : ℝ) 
  (new_grape_price : ℝ)
  (money_from_grapes : ℝ)
  (money_from_lemons : ℝ)
  (num_lemons : ℕ)) :
    price_increase_per_lemon = 4 ∧
    planned_lemon_price = 8 ∧
    planned_grape_price = 7 ∧
    num_grapes = 140 ∧
    total_money_collected = 2220 ∧
    new_lemon_price = 12 ∧
    new_grape_price = 9 ∧
    money_from_grapes = 1260 ∧
    money_from_lemons = 960 ∧
    num_lemons = 80  :=
by
    repeat { sorry }

end number_of_lemons_l693_693522


namespace first_pile_weights_l693_693871

def weights := {1, 2, 3, 4, 8, 16}
def total_weight (s : Set ℕ) : ℕ := s.sum id

theorem first_pile_weights : ∃ (s1 s2 : Set ℕ), s1 = {1, 16} ∧ s2 = {2, 3, 4, 8} ∧ s1 ∪ s2 = weights ∧ total_weight s1 = total_weight s2 := by
  let s1 := {1, 16}
  let s2 := {2, 3, 4, 8}
  have h1 : s1.union s2 = weights := by -- Show that the union of s1 and s2 gives the original set of weights.
    sorry
  have h2 : total_weight s1 = 17 := by -- Show that the total weight of s1 is 17.
    sorry
  have h3 : total_weight s2 = 17 := by -- Show that the total weight of s2 is 17.
    sorry
  exact ⟨s1, s2, rfl, rfl, h1, h2.symm⟩

end first_pile_weights_l693_693871


namespace largest_prime_factor_l693_693985

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693985


namespace fourth_grade_students_agreed_l693_693865

theorem fourth_grade_students_agreed :
  ∀ (total_students third_grade_students fourth_grade_students : ℕ), 
  total_students = 391 ∧ 
  third_grade_students = 154 ∧ 
  fourth_grade_students = total_students - third_grade_students 
  → fourth_grade_students = 237 :=
by
  intros total_students third_grade_students fourth_grade_students h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3] at h4
  exact h4

end fourth_grade_students_agreed_l693_693865


namespace triangle_angle_sum_l693_693508

theorem triangle_angle_sum {A B C D : Type*}
  (isosceles_ABC : AB = BC)
  (isosceles_ADC : AD = DC)
  (D_in_ABC : D ∈ triangle ABC)
  (angle_ABC : ∠ ABC = 50)
  (angle_ADC : ∠ ADC = 130) :
  ∠ BAD = 40 := 
sorry

end triangle_angle_sum_l693_693508


namespace probability_other_child_girl_l693_693552

theorem probability_other_child_girl (h : True) : 
  let outcomes := ["BG", "GB", "GG"] in
  let total_outcomes := outcomes.length in
  let favorable_outcomes := ["GG"].length in
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 3 :=
by
  sorry

end probability_other_child_girl_l693_693552


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693893

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693893


namespace incorrect_statement_about_BFD_primeE_l693_693367

noncomputable def is_parallelogram (quad : Quadrilateral) : Prop :=
  sorry -- Definition needed for quadrilateral being a parallelogram

noncomputable def could_be_square (quad : Quadrilateral) : Prop :=
  sorry -- Definition needed for quadrilateral potentially being a square

noncomputable def could_be_rhombus (quad : Quadrilateral) : Prop :=
  sorry -- Definition needed for quadrilateral potentially being a rhombus

noncomputable def proj_to_square (quad : Quadrilateral) : Prop :=
  sorry -- Definition needed for projection onto base being a square

structure Cube :=
  (A B C D A' B' C' D' : Point)

def quadrilateral_BFD_primeE (cube : Cube) : Quadrilateral :=
  sorry -- Construction of quadrilateral BFD'E in terms of cube properties

theorem incorrect_statement_about_BFD_primeE
  (cube : Cube) 
  (quad := quadrilateral_BFD_primeE cube) :
  ¬(could_be_square quad) :=
sorry

end incorrect_statement_about_BFD_primeE_l693_693367


namespace happy_and_evil_is_power_of_4_l693_693118

-- Definition of happy integer
def is_happy (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > b ∧ b > 0 ∧ n = (a^2 * b) / (a - b)

-- Definition of evil integer
def is_evil (m : ℕ) : Prop :=
  ¬∃ (n : ℕ), is_happy(n) ∧ d(n) = m 

-- Main theorem: All integers that are both happy and evil are powers of 4
theorem happy_and_evil_is_power_of_4 (m : ℕ) : (is_happy m ∧ is_evil m) → (∃ (k : ℕ), m = 4^k) :=
by
  sorry

end happy_and_evil_is_power_of_4_l693_693118


namespace largest_prime_factor_of_7fact_8fact_l693_693907

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693907


namespace number_of_points_at_distances_l693_693653

-- Definitions of the distances (Conditions from a))
def distance_m : ℝ := sorry
def distance_n : ℝ := sorry
def distance_p : ℝ := sorry

-- The planes (Construct parallel planes at specific distances)

-- Proving the number of solutions to the problem given the distances
theorem number_of_points_at_distances (m n p : ℝ) : ∃! pts : point3d, (dist pts m = distance_m) ∧ (dist pts n = distance_n) ∧ (dist pts p = distance_p) → pts.card = 8 := sorry

end number_of_points_at_distances_l693_693653


namespace infinite_solutions_exists_l693_693818

/--
Prove that the equation x - y + z = 1 has infinitely many solutions in positive integers
such that x, y, z are pairwise distinct, and for any two of these numbers, their product
is divisible by the third number.
-/
theorem infinite_solutions_exists :
  ∃ᶠ (x y z : ℕ) in at_top, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x - y + z = 1 ∧ 
  (x * y ∣ z) ∧ (y * z ∣ x) ∧ (z * x ∣ y) := 
sorry

end infinite_solutions_exists_l693_693818


namespace largest_prime_factor_of_fact_sum_is_7_l693_693971

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693971


namespace minimize_PR_RQ_l693_693283

-- Define the points P, Q and the point R with x = 2 and y = m
def P : ℝ × ℝ := (-2, -3)
def Q : ℝ × ℝ := (5, 3)
def R (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the function to calculate distance between two points
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the total length PR + RQ
def total_length (m : ℝ) : ℝ :=
  (distance P (R m)) + (distance (R m) Q)

-- Define the theorem to prove the value of m that minimizes PR + RQ is 3/7
theorem minimize_PR_RQ : total_length (3/7) = (distance P (R (3/7))) + (distance (R (3/7)) Q) :=
  sorry

end minimize_PR_RQ_l693_693283


namespace least_positive_integer_form_l693_693002

theorem least_positive_integer_form (n : ℕ) (hn : 0 < n) :
  ∃ dn, (dn = 2 * (4^n - 1) / 3 + 1) ∧ 
        ∀ (a b : Fin n → ℕ), (∑ i, (if a i = 0 then 1 else -1 ) * 2^(b i) ≠ dn) :=
by
  sorry

end least_positive_integer_form_l693_693002


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693929

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693929


namespace heather_total_oranges_l693_693727

-- Define the initial conditions
def initial_oranges : ℝ := 60.0
def additional_oranges : ℝ := 35.0

-- Define the total number of oranges
def total_oranges : ℝ := initial_oranges + additional_oranges

-- State the theorem that needs to be proven
theorem heather_total_oranges : total_oranges = 95.0 := 
by
  sorry

end heather_total_oranges_l693_693727


namespace factorize_a_squared_plus_2a_l693_693240

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2 * a = a * (a + 2) :=
  sorry

end factorize_a_squared_plus_2a_l693_693240


namespace perpendicular_base_on_side_l693_693544

theorem perpendicular_base_on_side (n : ℕ) (h : n = 101) 
  (h_inscribed : ∀ (i : ℕ), i < n → ∃ (a_i : \fintype \real);
    ¬ ∃ (p : \real), 
    ∀ (j : ℕ), j < n → 
    (p ∈ opposite_side({a_i}) → (dropped_perpendicular(a_i, opposite_side(a_i such that i exists), circle) ∧ 
    ¬ extension_side_opposite( a_i, p)
:
  \fintype \real

:
   ∃ (i : ℕ), i < n → (not_on_extension(i))) :
  (101)) :=
sorry

end perpendicular_base_on_side_l693_693544


namespace probability_of_point_below_x_axis_l693_693416

noncomputable def probability_below_x_axis 
  (P Q R S : Point) 
  (hx1 : P = (4, 4)) 
  (hx2 : Q = (-2, -2)) 
  (hx3 : R = (-8, -2)) 
  (hx4 : S = (2, 4)) 
  : ℝ :=
  let area_PQRS := 24 in -- The total area of parallelogram PQRS
  let area_QTUR := 8 in -- The area of the part below the x-axis
  area_QTUR / area_PQRS

theorem probability_of_point_below_x_axis : 
  probability_below_x_axis (4, 4) (-2, -2) (-8, -2) (2, 4) = 1 / 3 := sorry

end probability_of_point_below_x_axis_l693_693416


namespace simplify_fraction_l693_693440

theorem simplify_fraction (num denom : ℕ) (h_num : num = 90) (h_denom : denom = 150) : 
  num / denom = 3 / 5 := by
  rw [h_num, h_denom]
  norm_num
  sorry

end simplify_fraction_l693_693440


namespace percentage_increase_in_average_visibility_l693_693870

theorem percentage_increase_in_average_visibility :
  let avg_visibility_without_telescope := (100 + 110) / 2
  let avg_visibility_with_telescope := (150 + 165) / 2
  let increase_in_avg_visibility := avg_visibility_with_telescope - avg_visibility_without_telescope
  let percentage_increase := (increase_in_avg_visibility / avg_visibility_without_telescope) * 100
  percentage_increase = 50 := by
  -- calculations are omitted; proof goes here
  sorry

end percentage_increase_in_average_visibility_l693_693870


namespace num_students_second_grade_l693_693179

structure School :=
(total_students : ℕ)
(prob_male_first_grade : ℝ)

def stratified_sampling (school : School) : ℕ := sorry

theorem num_students_second_grade (school : School) (total_selected : ℕ) : 
    school.total_students = 4000 →
    school.prob_male_first_grade = 0.2 →
    total_selected = 100 →
    stratified_sampling school = 30 :=
by
  intros
  sorry

end num_students_second_grade_l693_693179


namespace simplify_fraction_l693_693449

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end simplify_fraction_l693_693449


namespace quadrilateral_area_is_two_l693_693417

def A : (Int × Int) := (0, 0)
def B : (Int × Int) := (2, 0)
def C : (Int × Int) := (2, 3)
def D : (Int × Int) := (0, 2)

noncomputable def area (p1 p2 p3 p4 : (Int × Int)) : ℚ :=
  (1 / 2 : ℚ) * (abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2) - 
                      (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p4.1 + p4.2 * p1.1)))

theorem quadrilateral_area_is_two : 
  area A B C D = 2 := by
  sorry

end quadrilateral_area_is_two_l693_693417


namespace product_quality_difference_l693_693089

variable (n a b c d : ℕ)
variable (P_K_2 : ℝ → ℝ)

def first_class_freq_A := a / (a + b : ℕ)
def first_class_freq_B := c / (c + d : ℕ)

def K2 := (n : ℝ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem product_quality_difference
  (ha : a = 150) (hb : b = 50) 
  (hc : c = 120) (hd : d = 80)
  (hn : n = 400)
  (hK : P_K_2 0.010 = 6.635) : 
  first_class_freq_A a b = 3 / 4 ∧
  first_class_freq_B c d = 3 / 5 ∧
  K2 n a b c d > P_K_2 0.010 :=
by {
  sorry
}

end product_quality_difference_l693_693089


namespace max_trig_expression_l693_693663

theorem max_trig_expression (A : ℝ) : (2 * Real.sin (A / 2) + Real.cos (A / 2) ≤ Real.sqrt 3) :=
sorry

end max_trig_expression_l693_693663


namespace problem1_i_problem1_ii_problem1_iii_problem2_problem3_l693_693743

section double_bottom_sequences

def is_double_bottom (a : ℕ → ℝ) (p q : ℕ) (c : ℝ) : Prop :=
  p ≠ q ∧ a p = c ∧ a q = c ∧ ∀ n : ℕ, (n ≠ p ∧ n ≠ q) → a n > c

-- (1) (i)
theorem problem1_i : is_double_bottom (λ n, n + 6 / n) p q c :=
sorry

-- (1) (ii)
theorem problem1_ii : ¬is_double_bottom (λ n, Real.sin (n * Real.pi / 2)) p q c :=
sorry

-- (1) (iii)
theorem problem1_iii : is_double_bottom (λ n, abs ((n - 3) * (n - 5))) p q c :=
sorry

-- (2)
theorem problem2 (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℝ) :
  (∀ n, a n = if n ≤ 50 then 101 - 2 * n else (2^(n - 50) + m)) →
  is_double_bottom a 50 51 ((2 + m)) →
  m = -1 ∧ (∀ n, n ≤ 50 → S n = 100 * n - n^2) ∧
  (∀ n, n ≥ 51 → S n = 2^(n - 49) - n + 2548) :=
sorry

-- (3)
theorem problem3 (k : ℤ) :
  (∃ k : ℤ, is_double_bottom (λ n, (k * n + 3) * (0.9 ^ n)) p q c) →
  k = -1 ∨ k = -3 :=
sorry

end double_bottom_sequences

end problem1_i_problem1_ii_problem1_iii_problem2_problem3_l693_693743


namespace predict_sales_l693_693488

noncomputable def mean_x : ℝ := (2 + 3 + 4 + 5 + 6) / 5
noncomputable def mean_y : ℝ := (29 + 41 + 50 + 59 + 71) / 5
noncomputable def a_hat : ℝ := mean_y - 10.2 * mean_x
noncomputable def regression : ℝ → ℝ := λ x, 10.2 * x + a_hat

theorem predict_sales (x : ℝ) (h : x = 8) : regression x = 90.8 :=
by
  unfold regression
  sorry

end predict_sales_l693_693488


namespace visibility_count_in_square_l693_693176

open Nat
open scoped BigOperators

theorem visibility_count_in_square :
  let points := Finset.Icc (0, 0) (25, 25).filter (λ p, p ≠ (0, 0) ∧ gcd p.fst p.snd = 1)
  Finset.card points = 399 := 
by {
  let points := Finset.Icc (0, 0) (25, 25),
  let filtered_points := points.filter (λ p, p ≠ (0, 0) ∧ gcd p.fst p.snd = 1),
  have visible_points_count : Finset.card filtered_points = 1 + 2 * ∑ x in Finset.range(25).filter (λ x, x > 0), Nat.totient (x+1),
  sorry
}

end visibility_count_in_square_l693_693176


namespace domain_of_sqrt_cos_minus_half_correct_l693_693642

noncomputable def domain_of_sqrt_cos_minus_half (x : ℝ) : Prop :=
  ∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3

theorem domain_of_sqrt_cos_minus_half_correct :
  ∀ x, (∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3) ↔
    ∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3 :=
by sorry

end domain_of_sqrt_cos_minus_half_correct_l693_693642


namespace middle_term_in_ratio_l693_693766

theorem middle_term_in_ratio (a b : ℝ) (h1 : a = 4) (h2 : b = 9) : ∃ x : ℝ, x * x = a * b ∧ (x = 6 ∨ x = -6) :=
by {
  have h : a * b = 36 := by { rw [h1, h2], norm_num },
  use 6,
  split,
  { rw [←h, mul_self_sqrt],
    { norm_num },
    { norm_num }},
  { left, refl }
}

end middle_term_in_ratio_l693_693766


namespace part_a_part_b_l693_693532

-- Part (a) Lean 4 statement
theorem part_a (m n : ℕ) 
  (k : Fin n → ℕ) 
  (divisible : (Finset.univ.sum (fun i => 2 ^ (k i))) % (2 ^ m - 1) = 0) : 
  n ≥ m := 
sorry

-- Part (b) Lean 4 statement
theorem part_b (m : ℕ) : 
  ¬ ∃ (P : ℕ), 
    (P % (10^m - 1) / 9 = 0) ∧ 
    (nat.sum_digits P < m) := 
sorry

end part_a_part_b_l693_693532


namespace smallest_y_condition_l693_693128

theorem smallest_y_condition : ∃ y : ℕ, y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 ∧ y = 167 :=
by 
  sorry

end smallest_y_condition_l693_693128


namespace tissues_used_l693_693076

-- Define the conditions
def box_tissues : ℕ := 160
def boxes_bought : ℕ := 3
def tissues_left : ℕ := 270

-- Define the theorem that needs to be proven
theorem tissues_used (total_tissues := boxes_bought * box_tissues) : total_tissues - tissues_left = 210 := by
  sorry

end tissues_used_l693_693076


namespace last_two_digits_of_sum_l693_693202

theorem last_two_digits_of_sum :
  let factorials := list.range' 1 96
  let sum_factorials := factorials.filter (λ n, n % 6 = 0).map factorial.sum %
  100
  sum_factorials = 20 :=
by
  sorry

end last_two_digits_of_sum_l693_693202


namespace emily_annual_holidays_l693_693649

theorem emily_annual_holidays 
    (holidays_per_month : ℕ) 
    (months_in_year : ℕ) 
    (h1: holidays_per_month = 2)
    (h2: months_in_year = 12)
    : holidays_per_month * months_in_year = 24 := 
by
  sorry

end emily_annual_holidays_l693_693649


namespace divisor_of_first_division_l693_693143

theorem divisor_of_first_division (n d : ℕ) (hn_pos : 0 < n)
  (h₁ : (n + 1) % d = 4) (h₂ : n % 2 = 1) : 
  d = 6 :=
sorry

end divisor_of_first_division_l693_693143


namespace correct_statements_l693_693290

-- Definitions of vectors and planes
variables {V : Type*} [inner_product_space ℝ V]
variables (e n1 n2: V) (l : Line V) (α β : Plane V)

-- Hypotheses
variables (h1 : direction_vector l = e)
variables (h2 : normal_vector α = n1)
variables (h3 : normal_vector β = n2)
variables (h4 : ¬l ∈ α)
variables (h5 : ¬l ∈ β)
variables (h6 : α ≠ β)

-- Proof that the statements A, B, and C are correct
theorem correct_statements :
  (e ⊥ n1 ↔ l ∥ α) ∧ 
  (n1 ⊥ n2 ↔ α ⊥ β) ∧ 
  (n1 ∥ n2 ↔ α ∥ β) :=
sorry

end correct_statements_l693_693290


namespace find_a_l693_693244

theorem find_a (a b : ℤ) (h : ∀ x, x^2 - x - 1 = 0 → ax^18 + bx^17 + 1 = 0) : a = 1597 :=
sorry

end find_a_l693_693244


namespace no_valid_n_for_f_l693_693396

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).sum id

theorem no_valid_n_for_f (n : ℕ) : sum_of_divisors (sum_of_divisors n) ≠ n + 3 :=
by
  sorry

end no_valid_n_for_f_l693_693396


namespace y_completes_work_l693_693539

theorem y_completes_work (x_work_time y_remaining_work_time : ℕ) (total_work : ℚ) :
  x_work_time = 40 ∧ y_remaining_work_time = 24 ∧ total_work = 1 →
  let x_daily_work := 1 / 40 in
  let y_remaining_work := total_work - (8 * x_daily_work) in
  let y_total_time := y_remaining_work_time / y_remaining_work in
  y_total_time = 30 :=
sorry

end y_completes_work_l693_693539


namespace compute_63_times_57_l693_693617

theorem compute_63_times_57 : 63 * 57 = 3591 := 
by {
   have h : (60 + 3) * (60 - 3) = 60^2 - 3^2, from
     by simp [mul_add, add_mul, add_assoc, sub_mul, mul_sub, sub_add, sub_sub, add_sub, mul_self_sub],
   have h1 : 60^2 = 3600, from rfl,
   have h2 : 3^2 = 9, from rfl,
   have h3 : 60^2 - 3^2 = 3600 - 9, by rw [h1, h2],
   rw h at h3,
   exact h3,
}

end compute_63_times_57_l693_693617


namespace air_quality_probabilities_average_exercise_exercise_air_quality_relationship_l693_693578

noncomputable def problem_data := ℕ
noncomputable def total_days := 100

-- Air Quality level data
def air_quality_data := [ (2, 16, 25), (5, 10, 12), (6, 7, 8), (7, 2, 0) ]
def air_quality_probs := [0.43, 0.27, 0.21, 0.09]

-- Midpoint values for the intervals
def exercise_midpoints := [100, 300, 500]

-- Summing the days for each air quality level
def air_quality_sums (data : List (Nat × Nat × Nat)) : List Nat :=
  data.map (λ (x : Nat × Nat × Nat) => x.1 + x.2 + x.3)

def air_quality_probs_computed : List Float :=
  (air_quality_sums air_quality_data).map (λ (x : Nat) => x / total_days.toFloat)

-- Estimated average number of people exercising in the park in a day
def average_exercise_calculated : Float := 
  1 / total_days.toFloat * ((exercise_midpoints[0] * (air_quality_data[0].1 + air_quality_data[1].1 + air_quality_data[2].1 + air_quality_data[3].1)) +
                            (exercise_midpoints[1] * (air_quality_data[0].2 + air_quality_data[1].2 + air_quality_data[2].2 + air_quality_data[3].2)) +
                            (exercise_midpoints[2] * (air_quality_data[0].3 + air_quality_data[1].3 + air_quality_data[2].3 + air_quality_data[3].3)))

def estimated_average_exercise := 350

-- 2x2 Contingency Table data
def contingency_table_counts := (33, 37, 22, 8)

-- Function to compute K^2 for contingency table
def compute_K2 (a b c d n : Nat) : Float :=
  n.toFloat * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d)).toFloat

-- K^2 computation result
def computed_K2_result : Float := compute_K2 33 37 22 8 100

-- Threshold for 95% confidence level
def confidence_threshold := 3.841

-- Relationship conclusion
def relationship : Bool :=
  computed_K2_result > confidence_threshold

theorem air_quality_probabilities :
  air_quality_probs_computed = air_quality_probs := sorry

theorem average_exercise :
  average_exercise_calculated = estimated_average_exercise := sorry

theorem exercise_air_quality_relationship :
  relationship = true := sorry

end air_quality_probabilities_average_exercise_exercise_air_quality_relationship_l693_693578


namespace tan_theta_range_l693_693197

theorem tan_theta_range {d : ℝ} (hd : d ∈ set.Ico 0 1) :
  let α := real.atan (sqrt 2 * (1 - d) / sqrt (1 - d^2)),
      β := real.atan (sqrt 2 * (1 + d) / sqrt (1 - d^2)),
      θ := α + β,
      tan_θ := real.tan θ
  in tan_θ ∈ set.Iic (-2 * sqrt 2) :=
by
  let α := real.atan (sqrt 2 * (1 - d) / sqrt (1 - d^2)),
      β := real.atan (sqrt 2 * (1 + d) / sqrt (1 - d^2)),
      θ := α + β,
      tan_θ := real.tan θ
  show tan_θ ∈ set.Iic (-2 * sqrt 2), from sorry

end tan_theta_range_l693_693197


namespace number_of_people_l693_693835

-- Definition of given conditions
def increase_in_average_weight (N : ℕ) : Prop :=
  ∀ (W_old W_new : ℕ), W_old = 65 → W_new = 89 → 3 * N = (W_new - W_old)

-- Main statement to prove
theorem number_of_people (N : ℕ) (h : increase_in_average_weight N) : N = 8 :=
by {
  -- Call to the provided condition
  have h_condition := h 65 89 rfl rfl,
  -- Simplify the obtained equation
  simp at h_condition,
  -- The simplified equation should lead to the answer
  exact nat.eq_of_mul_eq_mul_left (by norm_num) h_condition.symm,
}

end number_of_people_l693_693835


namespace correct_calculation_l693_693523

theorem correct_calculation (m n : ℝ) : 4 * m + 2 * n - (n - m) = 5 * m + n :=
by sorry

end correct_calculation_l693_693523


namespace plants_needed_correct_l693_693381

def total_plants_needed (ferns palms succulents total_desired : ℕ) : ℕ :=
 total_desired - (ferns + palms + succulents)

theorem plants_needed_correct : total_plants_needed 3 5 7 24 = 9 := by
  sorry

end plants_needed_correct_l693_693381


namespace yvonne_probability_is_correct_l693_693530

def probability_of_success (p_X p_Y p_Z : ℚ) : ℚ :=
  p_X * p_Y * (1 - p_Z)

theorem yvonne_probability_is_correct :
  ∀ (p_X p_Z: ℚ) (p_XZ' : ℚ),
  p_X = 1 / 4 →
  p_Z = 5 / 8 →
  p_XZ' = 0.0625 →
  let p_Y := p_XZ' / (p_X * (1 - p_Z)) in
  p_Y = 1 / 16 :=
by
  intros p_X p_Z p_XZ' hX hZ hXZ'
  let p_Y := p_XZ' / (p_X * (1 - p_Z))
  sorry

end yvonne_probability_is_correct_l693_693530


namespace distance_between_locations_l693_693509

theorem distance_between_locations (speed_B : ℝ) (time : ℝ) (factor_A : ℝ)
  (hB : speed_B = 60) (h_time : time = 2.4) (h_factor : factor_A = 1.5) :
  let speed_A := speed_B * factor_A in
  let combined_speed := speed_B + speed_A in
  let distance := combined_speed * time in
  distance = 360 :=
by
  -- Placeholder for the mathematical proof
  sorry

end distance_between_locations_l693_693509


namespace new_paint_intensity_l693_693470

def I1 : ℝ := 0.50
def I2 : ℝ := 0.25
def F : ℝ := 0.2

theorem new_paint_intensity : (1 - F) * I1 + F * I2 = 0.45 := by
  sorry

end new_paint_intensity_l693_693470


namespace largest_prime_factor_l693_693979

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693979


namespace modulus_of_z_l693_693332

open Complex

theorem modulus_of_z (z : ℂ) (hz : (1 + I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l693_693332


namespace factorize_a_squared_plus_2a_l693_693237

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2*a = a * (a + 2) :=
sorry

end factorize_a_squared_plus_2a_l693_693237


namespace expressions_equal_l693_693644

theorem expressions_equal {x y z : ℤ} : (x + 2 * y * z = (x + y) * (x + 2 * z)) ↔ (x + y + 2 * z = 1) :=
by
  sorry

end expressions_equal_l693_693644


namespace smallest_solution_fraction_eq_l693_693255

theorem smallest_solution_fraction_eq (x : ℝ) (h : x ≠ 3) :
    3 * x / (x - 3) + (3 * x^2 - 27) / x = 16 ↔ x = (2 - Real.sqrt 31) / 3 := 
sorry

end smallest_solution_fraction_eq_l693_693255


namespace zoo_recovery_time_l693_693543

theorem zoo_recovery_time (lions rhinos recover_time : ℕ) (total_animals : ℕ) (total_time : ℕ)
    (h_lions : lions = 3) (h_rhinos : rhinos = 2) (h_recover_time : recover_time = 2)
    (h_total_animals : total_animals = lions + rhinos) (h_total_time : total_time = total_animals * recover_time) :
    total_time = 10 :=
by
  rw [h_lions, h_rhinos] at h_total_animals
  rw [h_total_animals, h_recover_time] at h_total_time
  exact h_total_time

end zoo_recovery_time_l693_693543


namespace simplify_fraction_l693_693442

theorem simplify_fraction (num denom : ℕ) (h_num : num = 90) (h_denom : denom = 150) : 
  num / denom = 3 / 5 := by
  rw [h_num, h_denom]
  norm_num
  sorry

end simplify_fraction_l693_693442


namespace sum_reciprocals_B_l693_693793

noncomputable def B : Set ℕ :=
  {n | ∃ a b d : ℕ, n = 2^a * 3^b * 7^d}

theorem sum_reciprocals_B : 
  (∑' n in B, 1 / (n : ℝ)) = 7 / 2 :=
sorry -- Proof omitted

end sum_reciprocals_B_l693_693793


namespace initial_ratio_of_milk_to_water_l693_693359

theorem initial_ratio_of_milk_to_water (M W : ℕ) (h1 : M + W = 60) (h2 : 2 * M = W + 60) : M / W = 2 :=
by
  sorry

end initial_ratio_of_milk_to_water_l693_693359


namespace tom_candies_left_is_ten_l693_693505

-- Define initial conditions
def initial_candies: ℕ := 2
def friend_gave_candies: ℕ := 7
def bought_candies: ℕ := 10

-- Define total candies before sharing
def total_candies := initial_candies + friend_gave_candies + bought_candies

-- Define the number of candies Tom gives to his sister
def candies_given := total_candies / 2

-- Define the number of candies Tom has left
def candies_left := total_candies - candies_given

-- Prove the final number of candies left
theorem tom_candies_left_is_ten : candies_left = 10 :=
by
  -- The proof is left as an exercise
  sorry

end tom_candies_left_is_ten_l693_693505


namespace f_analytic_expression_f_m_range_l693_693302

-- Define the function using the given conditions
def f (x : ℝ) : ℝ := 
  if x < 0 then (x - 1)^2 
  else if x = 0 then 0 
  else - (x + 1)^2

-- State that f(x) is an odd function
axiom f_odd : ∀ x : ℝ, f(-x) = -f(x)

-- The first proof problem, proving the analytical expression
theorem f_analytic_expression :
  ∀ x : ℝ, 
  f(x) = 
  if x < 0 then (x - 1)^2 
  else if x = 0 then 0 
  else - (x + 1)^2 := sorry

-- The second proof problem, proving the range of m given the condition
theorem f_m_range (m : ℝ) (h : f(m^2 + 2m) + f(m) > 0) :
  -3 < m ∧ m < 0 := sorry

end f_analytic_expression_f_m_range_l693_693302


namespace N_is_orthocenter_of_AYZ_l693_693000

variables {α : Type*} [EuclideanGeometry α]

/-- Acute triangle ABC with altitudes BE and CF -/
variables {A B C E F M N X Y Z : α}
variables (is_acute_triangle : ∀ {x y z : α}, is_acute x y z)
variables (is_altitude_BE : is_altitude B E A C)
variables (is_altitude_CF : is_altitude C F A B)
variables (midpoint_M : is_midpoint M B C)
variables (intersection_N : line_intersection N (line AM) (line EF))
variables (projection_X : is_projection X N (line BC))
variables (projection_Y : is_projection Y X (line AB))
variables (projection_Z : is_projection Z X (line AC))

/-- Prove N is the orthocenter of triangle AYZ -/
theorem N_is_orthocenter_of_AYZ : is_orthocenter N A Y Z :=
  sorry

end N_is_orthocenter_of_AYZ_l693_693000


namespace fraction_of_white_surface_area_l693_693171

/-- A cube has edges of 4 inches and is constructed using 64 smaller cubes, each with edges of 1 inch.
Out of these smaller cubes, 56 are white and 8 are black. The 8 black cubes fully cover one face of the larger cube.
Prove that the fraction of the surface area of the larger cube that is white is 5/6. -/
theorem fraction_of_white_surface_area 
  (total_cubes : ℕ := 64)
  (white_cubes : ℕ := 56)
  (black_cubes : ℕ := 8)
  (total_surface_area : ℕ := 96)
  (black_face_area : ℕ := 16)
  (white_surface_area : ℕ := 80) :
  white_surface_area / total_surface_area = 5 / 6 :=
sorry

end fraction_of_white_surface_area_l693_693171


namespace coefficient_x2_in_P_l693_693764

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def P (x : ℚ) : ℚ :=
  (1 - x^4) * (2 - x)^5

theorem coefficient_x2_in_P (c : ℚ) (h : c = 80) :
  (∀ x : ℚ, P(x)) → c = 80 := 
begin
  sorry,
end

end coefficient_x2_in_P_l693_693764


namespace arithmetic_sequence_nth_term_l693_693058

theorem arithmetic_sequence_nth_term (x n : ℕ) 
  (h1 : 3 * x - 4)
  (h2 : 6 * x - 14)
  (h3 : 4 * x + 2)
  (hnth : 4018) :
  n = 716 := by
  sorry

end arithmetic_sequence_nth_term_l693_693058


namespace truck_gasoline_rate_l693_693608

theorem truck_gasoline_rate (gas_initial gas_final : ℕ) (dist_supermarket dist_farm_turn dist_farm_final : ℕ) 
    (total_miles gas_used : ℕ) : 
  gas_initial = 12 →
  gas_final = 2 →
  dist_supermarket = 10 →
  dist_farm_turn = 4 →
  dist_farm_final = 6 →
  total_miles = dist_supermarket + dist_farm_turn + dist_farm_final →
  gas_used = gas_initial - gas_final →
  total_miles / gas_used = 2 :=
by sorry

end truck_gasoline_rate_l693_693608


namespace units_digit_subtraction_l693_693062

theorem units_digit_subtraction (a b c : ℕ) (h1 : a = c + 3) : (101c + 10b + 300 - (101c + 10b + 3)) % 10 = 7 := by
  sorry

end units_digit_subtraction_l693_693062


namespace probability_of_number_between_21_and_30_l693_693408

-- Define the success condition of forming a two-digit number between 21 and 30.
def successful_number (d1 d2 : Nat) : Prop :=
  let n1 := 10 * d1 + d2
  let n2 := 10 * d2 + d1
  (21 ≤ n1 ∧ n1 ≤ 30) ∨ (21 ≤ n2 ∧ n2 ≤ 30)

-- Calculate the probability of a successful outcome.
def probability_success (favorable total : Nat) : Nat :=
  favorable / total

-- The main theorem claiming the probability that Melinda forms a number between 21 and 30.
theorem probability_of_number_between_21_and_30 :
  let successful_counts := 10
  let total_possible := 36
  probability_success successful_counts total_possible = 5 / 18 :=
by
  sorry

end probability_of_number_between_21_and_30_l693_693408


namespace equal_distances_midpoint_conditions_l693_693281

noncomputable def distance_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / sqrt (a^2 + b^2)

theorem equal_distances (m : ℝ) :
  distance_to_line (1, m) 2 (-1) 2 = distance_to_line (1, m) 1 2 (-4) ↔
  m = -1 ∨ m = 7 / 3 :=
by
  sorry

theorem midpoint_conditions (A B : ℝ × ℝ) :
  let P := (1, 1),
      A := (-2/5, 6/5),
      B := (12/5, 4/5) in
  (P.1 = (A.1 + B.1) / 2) ∧ (P.2 = (A.2 + B.2) / 2) ↔
  (∃ k : ℝ, k = -1/7 ∧ (P.2 - 1) = k * (P.1 - 1)) :=
by
  sorry

end equal_distances_midpoint_conditions_l693_693281


namespace problem_l693_693677

variable (f : ℝ → ℝ)

-- Given condition
axiom h : ∀ x : ℝ, f (1 / x) = 1 / (x + 1)

-- Prove that f(2) = 2/3
theorem problem : f 2 = 2 / 3 :=
sorry

end problem_l693_693677


namespace largest_prime_factor_of_7fact_8fact_l693_693905

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693905


namespace problem_statement_l693_693327

def P := {x : ℝ | x < 1}
def Q := {x : ℝ | x > 1}
def C_R (S : set ℝ) := {x : ℝ | x ∉ S}

theorem problem_statement : Q ⊆ C_R P := 
by sorry

end problem_statement_l693_693327


namespace max_n_l693_693553

open Nat

def four_element_set (s : Finset ℕ) : Prop := Finset.card s = 4

def good_set (s : Finset ℕ) : Prop :=
∃ (a b c d : ℕ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (a * b % gcd c d = 0 ∨ a * c % gcd b d = 0 ∨ a * d % gcd b c = 0 ∨
   b * c % gcd a d = 0 ∨ b * d % gcd a c = 0 ∨ c * d % gcd a b = 0)

theorem max_n (n : ℕ) : n ≤ 230 ↔ ∀ (s : Finset ℕ), four_element_set s → 
  (∀ x ∈ s, x ≤ n) → good_set s :=
begin
  sorry
end

end max_n_l693_693553


namespace largest_prime_factor_7fac_plus_8fac_l693_693955

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693955


namespace percentage_drop_l693_693554

theorem percentage_drop (P N P' N' : ℝ) (h1 : N' = 1.60 * N) (h2 : P' * N' = 1.2800000000000003 * (P * N)) :
  P' = 0.80 * P :=
by
  sorry

end percentage_drop_l693_693554


namespace largest_prime_factor_of_7fact_8fact_l693_693904

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693904


namespace cube_max_vertex_sum_l693_693550

theorem cube_max_vertex_sum :
  ∀ (f : ℕ → ℕ), (∑ i in finset.univ.filter (λ i, i < 6), (f i)) = 60 ∧
  (∀ i, f i + f (5 - i) = 10) → 
  (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
   a + b + c = 22 ∧ 
   f a + f b + f c = 22) :=
by
  intros f h,
  obtain ⟨a, b, c, ha, hb, hc, h_sum⟩ := 
    exists_max_sum_of_vertex_numbers f h,
  use [f a, f b, f c],
  finish

end cube_max_vertex_sum_l693_693550


namespace triangle_is_obtuse_l693_693372

theorem triangle_is_obtuse (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) 
  (h_sum : A + B + C = π) (h_sin_cos : sin A * cos C < 0) : ∃ C, (π/2 < C ∧ C < π) :=
by
  sorry

end triangle_is_obtuse_l693_693372


namespace jason_two_weeks_eggs_l693_693233

-- Definitions of given conditions
def eggs_per_omelet := 3
def days_per_week := 7
def weeks := 2

-- Statement to prove
theorem jason_two_weeks_eggs : (eggs_per_omelet * (days_per_week * weeks)) = 42 := by
  sorry

end jason_two_weeks_eggs_l693_693233


namespace area_of_triangle_NPQ_l693_693769

-- Definitions of the given points and their properties
def triangle_area (X Y Z : Type) [triangle XYZ] : ℝ := 120

noncomputable def midpoint (A B : Type) [line AB] : Type := {M : Type // 2 * (distance M A) = distance A B}

noncomputable def P (X Z : Type) := midpoint X Z  -- Point M
noncomputable def N (X Y : Type) := midpoint X Y  -- Point N
noncomputable def P (M Z : Type) := midpoint M Z  -- Point P
noncomputable def Q (M Y : Type) := midpoint M Y  -- Point Q

-- The theorem we want to prove, stating the area of triangle NPQ
theorem area_of_triangle_NPQ (X Y Z : Type) [triangle XYZ] :
  triangle_area X Y Z :=
begin
  let M := midpoint X Z,
  let N := midpoint X Y,
  let P := midpoint M Z,
  let Q := midpoint M Y,
  sorry
end

end area_of_triangle_NPQ_l693_693769


namespace christina_total_payment_l693_693207

def item1_ticket_price : ℝ := 200
def item1_discount1 : ℝ := 0.25
def item1_discount2 : ℝ := 0.15
def item1_tax_rate : ℝ := 0.07

def item2_ticket_price : ℝ := 150
def item2_discount : ℝ := 0.30
def item2_tax_rate : ℝ := 0.10

def item3_ticket_price : ℝ := 100
def item3_discount : ℝ := 0.20
def item3_tax_rate : ℝ := 0.05

def expected_total : ℝ := 335.93

theorem christina_total_payment :
  let item1_final_price :=
    (item1_ticket_price * (1 - item1_discount1) * (1 - item1_discount2)) * (1 + item1_tax_rate)
  let item2_final_price :=
    (item2_ticket_price * (1 - item2_discount)) * (1 + item2_tax_rate)
  let item3_final_price :=
    (item3_ticket_price * (1 - item3_discount)) * (1 + item3_tax_rate)
  item1_final_price + item2_final_price + item3_final_price = expected_total :=
by
  sorry

end christina_total_payment_l693_693207


namespace cross_section_area_l693_693836

-- Define the given conditions of the problem
structure Pyramid (T A B C : Point) :=
(base_side_len : ℝ)        -- all sides of base triangle ABC are equal
(base_all_sides_eq : A.distance B = base_side_len ∧ B.distance C = base_side_len ∧ C.distance A = base_side_len)
(height : ℝ)               -- height of the pyramid TA
(height_eq_sqrt3 : T.distance B = height)
(height_value : height = sqrt 3)

structure Sphere (O : Point) :=
(center_on_O : T.distance O = sqrt 3 / 2)

-- Define the theorem which we want to prove
theorem cross_section_area (T A B C O : Point) (P : Pyramid T A B C) (S : Sphere O) 
  (angle_with_base : ℝ) (angle_eq_60 : angle_with_base = 60) 
  (plane_parallel_to_AD : Prop):
  ∃ (area : ℝ), area = 11 * sqrt 3 / 10 :=
  sorry

end cross_section_area_l693_693836


namespace correct_propositions_is_two_l693_693064

/-
Conditions:
1. Proposition ①: "The four vertices of a trapezoid are in the same plane."
2. Proposition ②: "Three parallel lines must be coplanar."
3. Proposition ③: "Two planes that have three common points must coincide."
4. Proposition ④: "Four straight lines, each of which intersect the others and have different points of intersection, must be coplanar."
-/

def is_correct (prop : ℕ → Prop) (n : ℕ) : Prop :=
  match n with
  | 1 => ∀ (a b c d : ℝ × ℝ × ℝ), trapezoid a b c d → coplanar {a, b, c, d}
  | 2 => ∀ (l1 l2 l3 : ℝ → ℝ → ℝ), parallel l1 l2 → parallel l2 l3 → parallel l1 l3 → coplanar {l1, l2, l3}
  | 3 => ∀ (P Q R : ℝ × ℝ × ℝ), collinear {P, Q, R} → ∀ (plane1 plane2 : ℝ → ℝ → ℝ), plane1 P → plane1 Q → plane1 R → plane2 P → plane2 Q → plane2 R → plane1 = plane2
  | 4 => ∀ (l1 l2 l3 l4 : ℝ → ℝ → ℝ), (∀ p, ∃ a b, intersection p a b → p ∈ {l1, l2, l3, l4}) → coplanar {l1, l2, l3, l4}
  | _ => false

def correct_propositions : ℕ := 
  (if is_correct 1 then 1 else 0) +
  (if is_correct 2 then 1 else 0) +
  (if is_correct 3 then 1 else 0) +
  (if is_correct 4 then 1 else 0)

theorem correct_propositions_is_two :
  correct_propositions = 2 :=
sorry

end correct_propositions_is_two_l693_693064


namespace mac_total_loss_is_correct_l693_693026

def day_1_value : ℝ := 6 * 0.075 + 2 * 0.0075
def day_2_value : ℝ := 10 * 0.0045 + 5 * 0.0036
def day_3_value : ℝ := 4 * 0.10 + 1 * 0.011
def day_4_value : ℝ := 7 * 0.013 + 5 * 0.038
def day_5_value : ℝ := 3 * 0.5 + 2 * 0.0019
def day_6_value : ℝ := 12 * 0.0072 + 3 * 0.0013
def day_7_value : ℝ := 8 * 0.045 + 6 * 0.0089

def total_value : ℝ := day_1_value + day_2_value + day_3_value + day_4_value + day_5_value + day_6_value + day_7_value

def daily_loss (total_value: ℝ): ℝ := total_value - 0.25

def total_loss : ℝ := daily_loss day_1_value + daily_loss day_2_value + daily_loss day_3_value + daily_loss day_4_value + daily_loss day_5_value + daily_loss day_6_value + daily_loss day_7_value

theorem mac_total_loss_is_correct : total_loss = 2.1619 := 
by 
  simp [day_1_value, day_2_value, day_3_value, day_4_value, day_5_value, day_6_value, day_7_value, daily_loss, total_loss]
  sorry

end mac_total_loss_is_correct_l693_693026


namespace frequencies_first_class_confidence_difference_quality_l693_693106

theorem frequencies_first_class (a b c d n : ℕ) (Ha : a = 150) (Hb : b = 50) 
                                (Hc : c = 120) (Hd : d = 80) (Hn : n = 400) 
                                (totalA : a + b = 200) 
                                (totalB : c + d = 200) :
  (a / (a + b) = 3 / 4) ∧ (c / (c + d) = 3 / 5) := by
sorry

theorem confidence_difference_quality (a b c d n : ℕ) (Ha : a = 150)
                                       (Hb : b = 50) (Hc : c = 120)
                                       (Hd : d = 80) (Hn : n = 400)
                                       (total : n = 400)
                                       (first_class_total : a + c = 270)
                                       (second_class_total : b + d = 130) :
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  k_squared > 6.635 := by
sorry

end frequencies_first_class_confidence_difference_quality_l693_693106


namespace quadrilateral_area_of_tangents_l693_693832

noncomputable def area_of_quadrilateral_tangents (a b : ℝ) (h_a : a = 5) (h_b : b = 3) (e : ℝ)
    (h_e : e = Real.sqrt (1 - (b^2 / a^2))) : ℝ :=
  let c := a * e in
  let d := b * e in
  let P := (25 / 4, 0) in
  let Q := (0, 3) in
  let R := (-25 / 4, 0) in
  let S := (0, -3) in
  62.5

theorem quadrilateral_area_of_tangents (h : 9 * (5 ^ 2) + 25 * (3 ^ 2) = 225) :
  area_of_quadrilateral_tangents 5 3 5 rfl 3 rfl (Real.sqrt (1 - (3^2 / 5^2))) (by simp : 1 - (3^2 / 5^2) = 16 / 25) = 62.5 :=
by sorry

end quadrilateral_area_of_tangents_l693_693832


namespace largest_prime_factor_7fac_plus_8fac_l693_693954

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693954


namespace ellipse_equation_and_max_area_l693_693303

theorem ellipse_equation_and_max_area
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (A : ℝ × ℝ)
  (h3 : A = (2, Real.sqrt 2))
  (F1 F2 : ℝ × ℝ)
  (h4 : F2.2 = 0) -- F2's y-coordinate is 0 (perpendicular to x-axis)
  (h5 : F1.1 = -2 ∧ F2.1 = 2) -- F1 is (-2, 0) and F2 is (2, 0)
  (ellipse : ℝ × ℝ → Prop)
  (h6 : ∀ p, ellipse p ↔ (p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1)) :
  ((a = 2 * Real.sqrt 2) ∧ (b = 2) ∧ (ellipse = λ p, p.1 ^ 2 / 8 + p.2 ^ 2 / 4 = 1) ∧ (max_triangle_area := 2 * Real.sqrt 2)) := sorry

end ellipse_equation_and_max_area_l693_693303


namespace cos_alpha_value_tan_sin_expression_l693_693541

-- First proof problem
theorem cos_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : sin (α + π / 4) = sqrt 2 / 10) : 
  cos α = -3 / 5 := 
by 
  sorry

-- Second proof problem
theorem tan_sin_expression : 
  (tan 10 * (π / 180) - sqrt 3) * sin 40 * (π / 180) = -1 := 
by 
  sorry

end cos_alpha_value_tan_sin_expression_l693_693541


namespace minimum_value_fraction_l693_693331

theorem minimum_value_fraction (a : ℝ) (h : a > 1) : (a^2 - a + 1) / (a - 1) ≥ 3 :=
by
  sorry

end minimum_value_fraction_l693_693331


namespace find_y_pow_x_l693_693296

theorem find_y_pow_x (x y : ℝ) (h : |x - 2| + (y + 5)^2 = 0) : y^x = 25 :=
by
  have hx : x = 2, from by linarith,
  have hy : y = -5, from by linarith,
  rw [hx, hy],
  norm_num

end find_y_pow_x_l693_693296


namespace ellipse_equation_and_trapezoid_l693_693308

variable {x y : ℝ}

/- Definitions -/
def ellipse (a b : ℝ) := (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)
def distance (A B: ℝ × ℝ) : ℝ := real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
def eccentricity (c a : ℝ) := c / a
def trapezoid (A P Q M : ℝ × ℝ) := 
  (A.2 = P.2 ∧ M.2 = Q.2) ∨ (A.1 = P.1 ∧ M.1 = Q.1)

/- Proof problem -/
theorem ellipse_equation_and_trapezoid : 
  ∃ a b c : ℝ, 
  a > b ∧ b > 0 ∧ distance (1, 0) (-1, 0) = 4 ∧
  eccentricity c a = 1/2 ∧
  (b^2 = a^2 - c^2) ∧
  ∀ (P Q : ℝ × ℝ), Q = (4, 0) → P.1 = 4 →
  (∃ M : ℝ × ℝ, ellipse a b ∧ trapezoid (2, 0) P Q M) → 
  (∃ P : ℝ × ℝ, (P = (4, real.sqrt 3) ∨ P = (4, -real.sqrt 3))) :=
begin
  sorry
end

end ellipse_equation_and_trapezoid_l693_693308


namespace quadrilateral_is_trapezoid_or_parallelogram_l693_693424

def quadrilateral (A B C D : Type) [linear_ordered_field A] (O : A) 
  (M : A) (N : A) (XY_parallel_BC_or_DA : Prop) : Prop :=
  XY_parallel_BC_or_DA

theorem quadrilateral_is_trapezoid_or_parallelogram 
  (A B C D : Type) [linear_ordered_field A] (O : A) 
  (M : A) (N : A) (XY_parallel_BC_or_DA : Prop)
  (segment_bisected_by_O : XY_parallel_BC_or_DA → Prop) :
  (quadrilateral A B C D O M N XY_parallel_BC_or_DA) →
  (segment_bisected_by_O XY_parallel_BC_or_DA) →
  (is_trapezoid A B C D ∨ is_parallelogram A B C D) :=
by
  sorry

end quadrilateral_is_trapezoid_or_parallelogram_l693_693424


namespace gcd_72_168_gcd_98_280_f_at_3_l693_693513

/-- 
Prove that the GCD of 72 and 168 using the method of mutual subtraction is 24.
-/
theorem gcd_72_168 : Nat.gcd 72 168 = 24 :=
sorry

/-- 
Prove that the GCD of 98 and 280 using the Euclidean algorithm is 14.
-/
theorem gcd_98_280 : Nat.gcd 98 280 = 14 :=
sorry

/-- 
Prove that the value of f(3) where f(x) = x^5 + x^3 + x^2 + x + 1 is 283 using Horner's method.
-/
def f (x : ℕ) : ℕ := x^5 + x^3 + x^2 + x + 1

theorem f_at_3 : f 3 = 283 :=
sorry

end gcd_72_168_gcd_98_280_f_at_3_l693_693513


namespace sum_c_eq_l693_693672

-- Define the sequence of positive terms {a_n} such that a_1 = 1.
def a : ℕ+ → ℕ
| ⟨1, _⟩ := 1
| ⟨n+2, h⟩ := a ⟨n+1, Nat.succ_pos (n+1)⟩ + 1

-- Define the sum of the first n terms S_n
def S : ℕ+ → ℚ
| ⟨n, h⟩ := (n * (n + 1)) / 2

-- Define the sequence {c_n} based on the given properties
def c : ℕ+ → ℚ
| ⟨n, h⟩ := (-1 : ℚ)^n * ((1 / n) + (1 / (n + 1)))

-- Define the sum of the first 2016 terms of the sequence {c_n}
def sum_c : ℚ :=
  (Finset.range 2016).sum (λ i, c ⟨i + 1, Nat.succ_pos (i)⟩)

theorem sum_c_eq : sum_c = -2016 / 2017 :=
by sorry

end sum_c_eq_l693_693672


namespace part1_part2_l693_693285

section Part1
variable (A : Set ℝ) (B : Set ℝ) (m : ℝ)
def setA : Set ℝ := {x | -2 < x ∧ x < 5}
def setB : ℝ → Set ℝ := λ m, {x | m+1 ≤ x ∧ x ≤ 2*m-1}
def complementR : Set ℝ := {x | x ≤ -2 ∨ x ≥ 5}
variable (m3 : m = 3)

theorem part1 (hm : m = 3) : (complementR ∩ setB m) = {5} := by
  sorry
end Part1

section Part2
variable (A : Set ℝ) (B : Set ℝ) (m : ℝ)
def setA : Set ℝ := {x | -2 < x ∧ x < 5}
def setB : ℝ → Set ℝ := λ m, {x | m+1 ≤ x ∧ x ≤ 2*m-1}

theorem part2 : A ∪ setB m = A → m < 3 := by
  sorry
end Part2

end part1_part2_l693_693285


namespace smallest_domain_size_l693_693117

def f (x : ℕ) : ℕ :=
  if x % 2 = 1 then 3 * x + 1 else x / 2

theorem smallest_domain_size :
  ∃ d : ℕ, (∀ a, a ∈ {7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1} ↔ f a ∈ {22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1})
  ∧ 
  (∀ m : ℕ, (∀ a, a ∈ {7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1} ↔ f a ∈ {22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1}) → m ≥ 15)
  ∧ 
  d = 15 :=
sorry

end smallest_domain_size_l693_693117


namespace sum_of_T_is_101110000_l693_693391

-- The set T consists of all positive integers with five digits in base 2
def T : Finset ℕ := Finset.Ico 16 32

-- Prove the sum of all elements in T is equal to 376 in decimal which is 101110000 in binary
theorem sum_of_T_is_101110000 :
  (∑ x in T, x) = 376 := by
  sorry

end sum_of_T_is_101110000_l693_693391


namespace product_of_odd_neg_ints_plus_five_eq_zero_l693_693121

theorem product_of_odd_neg_ints_plus_five_eq_zero :
  let product := ∏ i in finset.Ico (-2022 : ℤ) 0, if odd i then i else 1
  in product + 5 = 0 :=
by
  sorry

end product_of_odd_neg_ints_plus_five_eq_zero_l693_693121


namespace ellipse_properties_l693_693195

theorem ellipse_properties 
  (foci1 foci2 : ℝ × ℝ) 
  (point_on_ellipse : ℝ × ℝ) 
  (h k a b : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0) 
  (ellipse_condition : foci1 = (-4, 1) ∧ foci2 = (-4, 5) ∧ point_on_ellipse = (1, 3))
  (ellipse_eqn : (x y : ℝ) → ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1) :
  a + k = 8 :=
by
  sorry

end ellipse_properties_l693_693195


namespace product_quality_difference_l693_693090

variable (n a b c d : ℕ)
variable (P_K_2 : ℝ → ℝ)

def first_class_freq_A := a / (a + b : ℕ)
def first_class_freq_B := c / (c + d : ℕ)

def K2 := (n : ℝ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem product_quality_difference
  (ha : a = 150) (hb : b = 50) 
  (hc : c = 120) (hd : d = 80)
  (hn : n = 400)
  (hK : P_K_2 0.010 = 6.635) : 
  first_class_freq_A a b = 3 / 4 ∧
  first_class_freq_B c d = 3 / 5 ∧
  K2 n a b c d > P_K_2 0.010 :=
by {
  sorry
}

end product_quality_difference_l693_693090


namespace largest_prime_factor_l693_693975

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693975


namespace volume_of_rotated_square_l693_693181

theorem volume_of_rotated_square (s : ℝ) (h : s = 10) :
  ∃ V : ℝ, V = 250 * Real.pi :=
by {
 sorry,
}

end volume_of_rotated_square_l693_693181


namespace math_problem_l693_693394

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48

theorem math_problem (a b c d : ℝ)
  (h1 : a + b + c + d = 6)
  (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
  proof_problem a b c d :=
by
  sorry

end math_problem_l693_693394


namespace final_price_including_tax_l693_693065

noncomputable def increasedPrice (originalPrice : ℝ) (increasePercentage : ℝ) : ℝ :=
  originalPrice + originalPrice * increasePercentage

noncomputable def discountedPrice (increasedPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  increasedPrice - increasedPrice * discountPercentage

noncomputable def finalPrice (discountedPrice : ℝ) (salesTax : ℝ) : ℝ :=
  discountedPrice + discountedPrice * salesTax

theorem final_price_including_tax :
  let originalPrice := 200
  let increasePercentage := 0.30
  let discountPercentage := 0.30
  let salesTax := 0.07
  let incPrice := increasedPrice originalPrice increasePercentage
  let disPrice := discountedPrice incPrice discountPercentage
  finalPrice disPrice salesTax = 194.74 :=
by
  simp [increasedPrice, discountedPrice, finalPrice]
  sorry

end final_price_including_tax_l693_693065


namespace multiply_63_57_l693_693619

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l693_693619


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693990

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693990


namespace fraction_of_grid_covered_by_triangle_l693_693514

open Real

noncomputable def point := (ℝ × ℝ)

def P : point := (2, 6)
def Q : point := (8, 2)
def R : point := (7, 7)

def area_triangle (P Q R : point) : ℝ :=
  0.5 * |P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)|

def area_grid : ℝ := 8 * 9

def fraction_covered : ℝ := area_triangle P Q R / area_grid

-- The theorem we need to prove:
theorem fraction_of_grid_covered_by_triangle : fraction_covered = 13 / 72 :=
by
  sorry

end fraction_of_grid_covered_by_triangle_l693_693514


namespace problem_min_value_l693_693312

noncomputable def f (a : ℝ) (x : ℝ) := -x^3 + a * x^2 - 4
noncomputable def f' (a : ℝ) (x : ℝ) := -3 * x^2 + 2 * a * x

theorem problem_min_value (a : ℝ) (m n : ℝ) (h_extremum : f' a 2 = 0) (h_m : m ∈ set.Icc (-1 : ℝ) 1) (h_n : n ∈ set.Icc (-1 : ℝ) 1) :
  min (f a m + f' a n) = -13 :=
sorry

end problem_min_value_l693_693312


namespace scientific_notation_316000000_l693_693864

theorem scientific_notation_316000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 316000000 = a * 10 ^ n ∧ a = 3.16 ∧ n = 8 :=
by
  -- Proof would be here
  sorry

end scientific_notation_316000000_l693_693864


namespace part_I_part_II_part_III_l693_693705

-- Define the sum of the first n terms of sequence S_n
def S_n (n : ℕ) : ℝ := (3^n - 1) / 2

-- Define b_n
def b_n (a_n : ℝ) : ℝ := 2 * (1 + Real.log (a_n) / Real.log 3)

-- Define the sum of the first n terms of the sequence {a_n b_n}
noncomputable def T_n (n : ℕ) : ℝ :=
  2 * (1 + 2 * 3 + 3 * 3^2 + ... + n * 3^(n-1))

-- (I) Prove that T_n = (1 + (2n - 1) * 3^n) / 2
theorem part_I (n : ℕ) (a_n : ℕ → ℝ) :
  T_n n = (1 + (2 * n - 1) * 3^n) / 2 := sorry

-- (II) Prove that ∏_(i = 1 to n) (1 + b_i) / b_i < sqrt (2 * n + 1)
theorem part_II (n : ℕ) (a_n : ℕ → ℝ) :
  (∏ i in range (n+1), (1 + b_n (a_n i)) / b_n (a_n i)) < Real.sqrt (2 * n + 1) := sorry

-- (III) Prove that ∏_(i = 1 to n) ((b_i - 1) / b_i)^2 ≥ 1 / (4 * n)
theorem part_III (n : ℕ) (a_n : ℕ → ℝ) :
  (∏ i in range (n+1), ((b_n (a_n i) - 1) / b_n (a_n i))^2) ≥ 1 / (4 * n) := sorry

end part_I_part_II_part_III_l693_693705


namespace ceil_neg_3_7_l693_693222

-- Define the ceiling function in Lean
def ceil (x : ℝ) : ℤ := int.ceil x

-- A predicate to represent the statement we want to prove
theorem ceil_neg_3_7 : ceil (-3.7) = -3 := by
  -- Provided conditions
  have h1 : ceil (-3.7) = int.ceil (-3.7) := rfl
  have h2 : int.ceil (-3.7) = -3 := by
    -- Lean's int.ceil function returns the smallest integer greater or equal to the input
    sorry  -- proof goes here

  -- The main statement
  exact h2

end ceil_neg_3_7_l693_693222


namespace g_minus_1001_l693_693831

def g (x : ℝ) : ℝ := sorry

theorem g_minus_1001 :
  (∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x) →
  g 1 = 3 →
  g (-1001) = 1005 :=
by
  intros h1 h2
  sorry

end g_minus_1001_l693_693831


namespace find_seventh_score_l693_693385

def jacob_scores (s : Fin 8 → ℕ) : Prop :=
  (∀ i, 85 ≤ s i ∧ s i ≤ 100 ∧ ∀ j ≠ i, s i ≠ s j) ∧
  (∀ n : Fin 8, (1 ≤ n) → ((∑ i in Finset.range n.val, s ⟨i, sorry⟩) / n.val) = ∑ / n.val) ∧
  s 7 = 90

theorem find_seventh_score (s : Fin 8 → ℕ) (h : jacob_scores s) : s 6 = 92 := sorry

end find_seventh_score_l693_693385


namespace number_of_s_divisible_by_7_l693_693017

def f (x : ℤ) : ℤ := x^2 + 4 * x + 3

def S : set ℤ := {x | 0 ≤ x ∧ x ≤ 30}

theorem number_of_s_divisible_by_7 : 
  {s ∈ S | f s % 7 = 0}.card = 4 :=
begin
  sorry
end

end number_of_s_divisible_by_7_l693_693017


namespace solution_l693_693783

def A (a : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.1 - p.2 ≥ 0 ∧ a * p.1 + p.2 ≥ 2 ∧ p.1 - a * p.2 ≤ 2 }

theorem solution (a : ℝ) : a < 0 → (1, 1) ∉ A a := by
  intros ha
  have h1 : 1 - 1 ≥ 0 := by linarith
  have h2 : a + 1 ≥ 2 → False := by linarith
  have h3 : 1 - a ≤ 2 := by linarith
  sorry

end solution_l693_693783


namespace solve_system_eq_solve_system_ineq_l693_693466

-- For the system of equations:
theorem solve_system_eq (x y : ℝ) (h1 : x + 2 * y = 7) (h2 : 3 * x + y = 6) : x = 1 ∧ y = 3 :=
sorry

-- For the system of inequalities:
theorem solve_system_ineq (x : ℝ) (h1 : 2 * (x - 1) + 1 > -3) (h2 : x - 1 ≤ (1 + x) / 3) : -1 < x ∧ x ≤ 2 :=
sorry

end solve_system_eq_solve_system_ineq_l693_693466


namespace length_of_BD_l693_693768

/-- Triangle ABC is a right triangle with C as the right angle, AC = 9 and BC = 12. 
    Points D and E are on AB and BC respectively, with BED being a right angle, and DE = 10.
    This statement proves that BD = 50/3. -/
theorem length_of_BD {A B C D E : Type} [MetricSpace A]
  (angle_C_eq_90 : ∠C = 90)
  (AC_eq_9 : AC = 9)
  (BC_eq_12 : BC = 12)
  (D_on_AB : D ∈ segment A B)
  (E_on_BC : E ∈ segment B C)
  (angle_BED_eq_90 : ∠BED = 90)
  (DE_eq_10 : DE = 10) : BD = 50 / 3 :=
sorry

end length_of_BD_l693_693768


namespace find_a_for_perpendicular_lines_l693_693747

theorem find_a_for_perpendicular_lines :
  ∀ a : ℝ, 
  (let line_M := (2 * a + 5) * x + (a - 2) * y + 4 = 0 in
   let line_N := (2 - a) * x + (a + 3) * y - 1 = 0 in
   (line_M ∧ line_N)) → 
   (a = 2 ∨ a = -2) := 
begin
  sorry
end

end find_a_for_perpendicular_lines_l693_693747


namespace simple_interest_earned_l693_693182

variable (P R T SI : ℝ)

-- Definitions based on conditions
def total_sum : ℝ := 16065
def rate : ℝ := 5 / 100
def time : ℝ := 5
def simple_interest_formula (P R T : ℝ) : ℝ := P * R * T / 100

theorem simple_interest_earned :
  ∃ P : ℝ, total_sum = P + simple_interest_formula P rate time ∧
            simple_interest_formula P rate time = 3213 :=
by
  sorry

end simple_interest_earned_l693_693182


namespace impossible_arrangement_of_300_numbers_in_circle_l693_693152

theorem impossible_arrangement_of_300_numbers_in_circle :
  ¬ ∃ (nums : Fin 300 → ℕ), (∀ i : Fin 300, nums i > 0) ∧
    ∃ unique_exception : Fin 300,
      ∀ i : Fin 300, i ≠ unique_exception → nums i = Int.natAbs (nums (Fin.mod (i.val - 1) 300) - nums (Fin.mod (i.val + 1) 300)) := 
sorry

end impossible_arrangement_of_300_numbers_in_circle_l693_693152


namespace well_depth_l693_693167

noncomputable def radius (diameter : ℝ) : ℝ := diameter / 2

noncomputable def volume_of_well (depth : ℝ) : ℝ := Real.pi * (radius 4) ^ 2 * depth

theorem well_depth {depth : ℝ} (h_volume : volume_of_well depth = 301.59289474462014) :
  depth ≈ 24 :=
sorry

end well_depth_l693_693167


namespace largest_prime_factor_7fac_8fac_l693_693946

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693946


namespace find_polynomials_satisfying_functional_equation_l693_693667

noncomputable def P_1 (x : ℝ) : ℝ := x
noncomputable def P_2 (x : ℝ) : ℝ := x^2 + 1
noncomputable def P_3 (x : ℝ) : ℝ := x^4 + 2 * x^2 + 2

theorem find_polynomials_satisfying_functional_equation :
    ∃ P1 P2 P3 : ℝ → ℝ,
      (∀ x : ℝ, P1 x = x) ∧
      (∀ x : ℝ, P2 x = x^2 + 1) ∧
      (∀ x : ℝ, P3 x = x^4 + 2 * x^2 + 2) ∧
      (∀ P : ℝ → ℝ, 
          (∀ x : ℝ, P(x^2 + 1) = P(x)^2 + 1) → 
          (P = P1 ∨ P = P2 ∨ P = P3))
      :=
begin
  use [P_1, P_2, P_3],
  split,
  { intro x, refl },
  split,
  { intro x, refl },
  split,
  { intro x, refl },
  intros P hP,
  sorry
end

end find_polynomials_satisfying_functional_equation_l693_693667


namespace geometric_sum_s5_l693_693291

noncomputable def S_n (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * (1 - q^n) / (1 - q)

theorem geometric_sum_s5 (a_3 a_4 a_5 a_7 : ℝ) (h1 : a_3 > 0) (h2 : a_4 > 0) (h3 : a_5 > 0) (h4 : a_7 > 0) 
  (h_geom : Float.sqrt (a_3 * a_5) = 2) (h_arith : (a_4 + 2 * a_7) / 2 = 5 / 4) : 
  S_n (16 : ℝ) (1 / 2 : ℝ) 5 = 31 := by
  sorry

end geometric_sum_s5_l693_693291


namespace linear_system_solution_l693_693722

theorem linear_system_solution (a b x y : ℤ)
  (h1 : a * x - y = 4)
  (h2 : 3 * x + b * y = 4)
  (hx : x = 2)
  (hy : y = -2) :
  a + b = 2 :=
by {
  rw [hx, hy] at h1 h2,
  have ha : a = 1, {
    linarith,
  },
  have hb : b = 1, {
    linarith,
  },
  rw [ha, hb],
  exact rfl,
}

end linear_system_solution_l693_693722


namespace find_width_of_sheet_of_paper_l693_693566

def width_of_sheet_of_paper (W : ℝ) : Prop :=
  let margin := 1.5
  let length_of_paper := 10
  let area_covered := 38.5
  let width_of_picture := W - 2 * margin
  let length_of_picture := length_of_paper - 2 * margin
  width_of_picture * length_of_picture = area_covered

theorem find_width_of_sheet_of_paper : ∃ W : ℝ, width_of_sheet_of_paper W ∧ W = 8.5 :=
by
  -- Placeholder for the actual proof
  sorry

end find_width_of_sheet_of_paper_l693_693566


namespace total_number_of_questions_l693_693812

theorem total_number_of_questions (N : ℕ)
  (hp : 0.8 * N = (4 / 5 : ℝ) * N)
  (hv : 35 = 35)
  (hb : (N / 2 : ℕ) = 1 * (N.div 2))
  (ha : N - 7 = N - 7) : N = 60 :=
by
  sorry

end total_number_of_questions_l693_693812


namespace ginger_limeade_calories_in_300g_l693_693779

/-- Definitions and conditions -/
def lime_juice_weight : ℕ := 150
def honey_weight : ℕ := 120
def water_weight : ℕ := 450
def ginger_weight : ℕ := 30

def total_weight : ℕ := lime_juice_weight + honey_weight + water_weight + ginger_weight

def lime_juice_calories_per_100g : ℕ := 20
def honey_calories_per_100g : ℕ := 304
def ginger_calories_per_30g : ℕ := 2

def total_calories : ℝ :=
  (lime_juice_weight / 100.0) * lime_juice_calories_per_100g +
  (honey_weight / 100.0) * honey_calories_per_100g +
  (ginger_weight / 30.0) * ginger_calories_per_30g

def calories_per_gram : ℝ := total_calories / total_weight

/-- Main statement -/
theorem ginger_limeade_calories_in_300g : calories_per_gram * 300 ≈ 159 :=
by
  sorry

end ginger_limeade_calories_in_300g_l693_693779


namespace correct_propositions_l693_693079

theorem correct_propositions :
  ¬(∀ (α β : ℝ), (0 < α ∧ α < π/2) ∧ (0 < β ∧ β < π/2) ∧ (α > β) → sin α > sin β) ∧
  ¬(∀ (a : ℝ), (∃ k : ℤ, 2 * π / |a| = 4 * π) → (a = 1/2)) ∧
  ¬(∀ (x : ℝ), (x ≠ π/2 + 2 * k * π ∧ x ≠ (2 * (π/2 + 2 * k * π))) → y = (sin 2*x - sin x)/(sin x - 1)) ∧
  (∀ (x : ℝ), (0 ≤ x ∧ x ≤ π) → (y = sin (x - π/2) ∧ monotone_on sin (x-π/2))) ∧
  (∀ (x : ℝ), (π/4 ≤ x ∧ x ≤ π/2) → ((sin^2 x + √3 * sin x * cos x) ≤ 3/2)) :=
  sorry

end correct_propositions_l693_693079


namespace sum_of_prime_divisors_of_3_pow_7_minus_1_l693_693611

theorem sum_of_prime_divisors_of_3_pow_7_minus_1 :
  let n := 3 ^ 7 - 1
  n = 2 * 1093 → (nat.prime 2 ∧ nat.prime 1093 ∧ (2 + 1093 = 1095)) :=
by
  intro n hn
  sorry

end sum_of_prime_divisors_of_3_pow_7_minus_1_l693_693611


namespace distance_PQ_l693_693005

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def W : Point3D := ⟨0, 0, 0⟩
def X : Point3D := ⟨2, 0, 0⟩
def Y : Point3D := ⟨2, 3, 0⟩
def Z : Point3D := ⟨0, 3, 0⟩

def W' : Point3D := ⟨0, 0, 12⟩
def X' : Point3D := ⟨2, 0, 16⟩
def Y' : Point3D := ⟨2, 3, 24⟩
def Z' : Point3D := ⟨0, 3, 12⟩

def midpoint (a b : Point3D) : Point3D :=
⟨(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2⟩

def P : Point3D := midpoint W' Y'
def Q : Point3D := midpoint X' Z'

def distance (p q : Point3D) : ℝ :=
Real.sqrt ((p.x - q.x) ^ 2 + (p.y - q.y) ^ 2 + (p.z - q.z) ^ 2)

theorem distance_PQ : distance P Q = 4 := by
  sorry

end distance_PQ_l693_693005


namespace exist_adjacent_triangles_with_large_difference_l693_693590

theorem exist_adjacent_triangles_with_large_difference :
  ∀ (f : Fin 25 → Fin 25), ∃ i j : Fin 25, i ≠ j ∧ adjacent i j ∧ |f i - f j| > 3 := 
  sorry

end exist_adjacent_triangles_with_large_difference_l693_693590


namespace circle_geometry_problem_l693_693480

/-- Given conditions of the geometric problem:
    - AE = 169 cm
    - EC = 119 cm
    - BD bisects ∠CBA
    - AB is the diameter of the circle
    - Chords AC and BD intersect at E
    Prove that ED = 65 cm.
-/
theorem circle_geometry_problem 
  (A B C D E : Type)
  (circle : ∀ {X : Type}, X)
  (chord : ∀ {X Y : Type}, X → Y → Type)
  (AE EC ED : ℝ)
  (is_diameter : ∀ {X Y : Type}, X → Y → Prop)
  (bisects : ∀ {X Y Z : Type}, X → Y → Z → Prop)
  (intersects : ∀ {X Y Z : Type}, X → Y → Z → Prop)
  (H1 : AE = 169)
  (H2 : EC = 119)
  (H3 : bisects B D A)
  (H4 : is_diameter A B)
  (H5 : intersects A C E)
  (H6 : intersects B D E) :
  ED = 65 := sorry

end circle_geometry_problem_l693_693480


namespace multiply_63_57_l693_693620

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l693_693620


namespace perp_vectors_implies_lambda_l693_693321

/-- Defining vector a -/
def a : ℝ × ℝ := (2, 1)

/-- Defining vector b with variable lambda -/
def b (λ : ℝ) : ℝ × ℝ := (3, λ)

/-- Defining the dot product for ℝ × ℝ -/
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/-- The goal is to prove that if the dot product of vectors a and b is zero,
    then λ = -6. -/
theorem perp_vectors_implies_lambda (λ : ℝ) (h : dot_product a (b λ) = 0) : λ = -6 :=
by
  sorry

end perp_vectors_implies_lambda_l693_693321


namespace false_propositions_l693_693675

variables (α β : Plane) (m n : Line)
variable (three_non_collinear_points : ∃ (P Q R : Point), P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧ P ∈ α ∧ Q ∈ α ∧ R ∈ α ∧ (∀ x ∈ [P, Q, R], dist x β = d))

theorem false_propositions (h1 : m ⊥ α) (h2 : m ⊥ n) (h3 : m ∥ α) (h4 : α ⊥ β) (h5 : n ⊥ β) (h6 : three_non_collinear_points) :
  (¬ (n ∥ α) ∧ ¬ (m ⊥ β) ∧ ¬ (α ∥ β)) :=
sorry

end false_propositions_l693_693675


namespace real_seq_proof_l693_693781

noncomputable def real_seq_ineq (a : ℕ → ℝ) : Prop :=
  ∀ k m : ℕ, k > 0 → m > 0 → |a (k + m) - a k - a m| ≤ 1

theorem real_seq_proof (a : ℕ → ℝ) (h : real_seq_ineq a) :
  ∀ k m : ℕ, k > 0 → m > 0 → |a k / k - a m / m| < 1 / k + 1 / m :=
by
  sorry

end real_seq_proof_l693_693781


namespace coefficient_x3_in_expansion_l693_693837

theorem coefficient_x3_in_expansion : 
  (∃ c : ℤ, ∀ x : ℤ, (1 - 2 * (x : ℤ)) ^ 6 = ∑ r in finset.range 7, ((-2) ^ r * nat.choose 6 r * x ^ r) ∧ 
  (gett_coeff (x^3) (∑ r in finset.range 7, ((-2) ^ r * nat.choose 6 r * x ^ r)) = c)) :=
begin
  sorry
end

end coefficient_x3_in_expansion_l693_693837


namespace product_area_perimeter_square_l693_693031

-- Define the coordinates of points E, F, G, H
def E : ℝ × ℝ := (4, 5)
def F : ℝ × ℝ := (5, 2)
def G : ℝ × ℝ := (2, 1)
def H : ℝ × ℝ := (1, 4)

-- Define the Euclidean distance function
def euclidean_distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Prove that the product of the area and the perimeter of square EFGH is 40√10 cm³
theorem product_area_perimeter_square :
  let side_length := euclidean_distance E F in
  let area := side_length ^ 2 in
  let perimeter := 4 * side_length in
  area * perimeter = 40 * real.sqrt 10 :=
by
  sorry

end product_area_perimeter_square_l693_693031


namespace AK_eq_BK_l693_693419

open EuclideanGeometry

-- Definitions for vertices
variables (A B C D K : Point)

-- Conditions of the problem
variables (hD : D ∈ line_segment A C)
variables (hAngleEq : ∠ B D C = ∠ B A C)
variables (hBisectorK1 : K ∈ line_bisector B A C)
variables (hBisectorK2 : K ∈ line_bisector D B C)

-- Theorem statement
theorem AK_eq_BK : dist A K = dist B K :=
sorry

end AK_eq_BK_l693_693419


namespace smallest_distance_condition_l693_693012

open Complex

noncomputable def a : ℂ := -2 - 4 * I
noncomputable def b : ℂ := 6 + 7 * I

theorem smallest_distance_condition (z w : ℂ) 
  (hz : abs (z - a) = 2) 
  (hw : abs (w - b) = 4) : 
  abs (z - w) ≥ real.sqrt 185 - 6 :=
sorry

end smallest_distance_condition_l693_693012


namespace minimum_area_rectangle_l693_693275

noncomputable def minimum_rectangle_area (a : ℝ) : ℝ :=
  if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
  else if a < 1 / 2 then 1 - 2 * a
  else 0

theorem minimum_area_rectangle (a : ℝ) :
  minimum_rectangle_area a =
    if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
    else if a < 1 / 2 then 1 - 2 * a
    else 0 :=
by
  sorry

end minimum_area_rectangle_l693_693275


namespace ceil_neg_3_7_l693_693220

-- Define the ceiling function in Lean
def ceil (x : ℝ) : ℤ := int.ceil x

-- A predicate to represent the statement we want to prove
theorem ceil_neg_3_7 : ceil (-3.7) = -3 := by
  -- Provided conditions
  have h1 : ceil (-3.7) = int.ceil (-3.7) := rfl
  have h2 : int.ceil (-3.7) = -3 := by
    -- Lean's int.ceil function returns the smallest integer greater or equal to the input
    sorry  -- proof goes here

  -- The main statement
  exact h2

end ceil_neg_3_7_l693_693220


namespace correct_diagram_to_describe_production_steps_l693_693082

/-- To describe the production steps of a certain product in a factory, the correct diagram type needs to be chosen 
    from the options: Product Structure Diagram, Material Structure Diagram, Program Flowchart, and Process Flow Diagram. -/
theorem correct_diagram_to_describe_production_steps
  (diagram : Type)
  (Product_Structure_Diagram Material_Structure_Diagram Program_Flowchart Process_Flow_Diagram : diagram):
  (to_describe_production_steps : diagram)
  (h : Process_Flow_Diagram = to_describe_production_steps) :=
  to_describe_production_steps = Process_Flow_Diagram := 
sorry

end correct_diagram_to_describe_production_steps_l693_693082


namespace probability_interval_normal_distribution_l693_693474

noncomputable def normal_distribution (μ σ : ℝ) := ProbDensityFunction.mk (fun x => (1 / (σ * (2 * Real.pi)^(1/2))) * Real.exp (-(x - μ)^2 / (2 * σ^2))) sorry sorry

theorem probability_interval_normal_distribution (μ σ : ℝ)
  (h1 : Prob (event {x | x > 5}) (normal_distribution μ σ) = 0.2)
  (h2 : Prob (event {x | x < -1}) (normal_distribution μ σ) = 0.2) : 
  Prob (event {x | 2 < x ∧ x < 5}) (normal_distribution μ σ) = 0.3 :=
sorry

end probability_interval_normal_distribution_l693_693474


namespace largest_prime_factor_7fac_plus_8fac_l693_693953

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693953


namespace arithmetic_mean_of_ints_from_neg5_to_6_l693_693875

def int_range := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
def mean (nums : List Int) : Float := (nums.sum : Float) / (nums.length : Float)

theorem arithmetic_mean_of_ints_from_neg5_to_6 : 
  mean int_range = 0.5 := by
    sorry

end arithmetic_mean_of_ints_from_neg5_to_6_l693_693875


namespace original_class_size_l693_693536

theorem original_class_size
  (N : ℕ)
  (h1 : 40 * N = T)
  (h2 : T + 15 * 32 = 36 * (N + 15)) :
  N = 15 := by
  sorry

end original_class_size_l693_693536


namespace intersect_lines_single_point_l693_693277

variables {C A1 B1 A2 B2 : Point}
variable k : ℝ
variable e1 e2 : Line

-- Define segments CA1, CB1, CA2, CB2
noncomputable def CA1 := dist C A1 
noncomputable def CB1 := dist C B1 
noncomputable def CA2 := dist C A2 
noncomputable def CB2 := dist C B2 

-- Condition on reciprocals
axiom reciprocal_condition1 : 1 / CA1 + 1 / CB1 = k
axiom reciprocal_condition2 : 1 / CA2 + 1 / CB2 = k

-- Definition of the proof
theorem intersect_lines_single_point (reciprocal_condition1 : 1 / CA1 + 1 / CB1 = k)
                                     (reciprocal_condition2 : 1 / CA2 + 1 / CB2 = k) 
                                     (intersect_e1 : e1.intersect C A1 B1)
                                     (intersect_e2 : e2.intersect C A2 B2) : 
  ∃ (M : Point), (M ∈ e1) ∧ (M ∈ e2) :=
sorry

end intersect_lines_single_point_l693_693277


namespace unique_solution_count_l693_693732

theorem unique_solution_count : 
  ∃! (a : ℝ), ∀ {x : ℝ}, (0 < x) → 
    4 * a ^ 2 + 3 * x * log x + 3 * (log x)^2 = 13 * a * log x + a * x :=
sorry

end unique_solution_count_l693_693732


namespace trig_identity_solution_l693_693135

theorem trig_identity_solution (z : ℝ) : 
  (∃ k : ℤ, z = (k * Real.pi) / 3) ↔ (sin(z)^3 * sin(3 * z) + cos(z)^3 * cos(3 * z) = cos(4 * z)^3) := 
by
  sorry

end trig_identity_solution_l693_693135


namespace decrease_percent_revenue_l693_693853

theorem decrease_percent_revenue 
  (T C : ℝ) 
  (hT : T > 0) 
  (hC : C > 0) 
  (new_tax : ℝ := 0.65 * T) 
  (new_consumption : ℝ := 1.15 * C) 
  (original_revenue : ℝ := T * C) 
  (new_revenue : ℝ := new_tax * new_consumption) :
  100 * (original_revenue - new_revenue) / original_revenue = 25.25 :=
sorry

end decrease_percent_revenue_l693_693853


namespace second_pipe_fill_time_l693_693115

open Real

theorem second_pipe_fill_time (t : ℝ) :
    (1 / 18 + 1 / t - 1 / 45 = 1 / 12) → t = 20 :=
by
  -- Definition of conditions as per problem statement
  -- first pipe fills in 18 minutes, so rate is 1/18 tank per minute
  let rate_first := 1 / 18
  -- outlet pipe empties in 45 minutes, so rate is -1/45 tank per minute (negative since it empties)
  let rate_outlet := -1 / 45
  -- combined rate of all pipes together is 1/12 tank per minute
  let combined_rate := 1 / 12
  assume h
  -- Prove that the time it takes for the second pipe to fill the tank t = 20
  sorry

end second_pipe_fill_time_l693_693115


namespace largest_prime_factor_l693_693981

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693981


namespace fuel_first_third_l693_693805

-- Defining constants based on conditions
def total_fuel := 60
def fuel_second_third := total_fuel / 3
def fuel_final_third := fuel_second_third / 2

-- Defining what we need to prove
theorem fuel_first_third :
  total_fuel - (fuel_second_third + fuel_final_third) = 30 :=
by
  sorry

end fuel_first_third_l693_693805


namespace ball_hits_ground_in_approximately_2_7181_seconds_l693_693600

noncomputable def time_when_ball_hits_ground : ℝ :=
  let h (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 200
  exists_unique t in { t : ℝ | h t = 0} ∧ t ≥ 0

theorem ball_hits_ground_in_approximately_2_7181_seconds :
  ∃ t : ℝ, time_when_ball_hits_ground t ∧ t ≈ 2.7181 :=
sorry

end ball_hits_ground_in_approximately_2_7181_seconds_l693_693600


namespace derivative_at_pi_div_2_l693_693339

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x

theorem derivative_at_pi_div_2 : (deriv f (Real.pi / 2)) = 4 := 
by
  sorry

end derivative_at_pi_div_2_l693_693339


namespace find_AC_length_l693_693358

-- Definitions of the cyclic quadrilateral and given conditions
def cyclic_quadrilateral (A B C D : Type) [HasAngle A B] [HasAngle B C] [HasAngle C D] :=
  true -- cyclic quadrilateral implies the opposite angles sum to 180 degrees, etc.

def angle_ratio (α β γ : ℕ) := α / β = 2 / 3 ∧ β / γ = 3 / 4

def given_conditions (CD BC : ℕ) (angle_A angle_B angle_C : ℕ) :=
  CD = 10 ∧ BC = 12 * Real.sqrt 3 - 5 ∧ angle_A = 60 ∧ angle_B = 90 ∧ angle_C = 120

-- The main statement to be proved
theorem find_AC_length (A B C D : Type) [cyclic_quadrilateral A B C D]
  (α β γ : ℕ)
  (CD BC : ℕ) :
  angle_ratio α β γ →
  given_conditions CD BC α β γ →
  ∃ AC : ℝ, AC = 26 :=
by
  sorry

end find_AC_length_l693_693358


namespace parabola_and_points_l693_693482

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_and_points (x1 y1 x2 y2 : ℝ) 
  (hA_on_parabola : x1^2 = 4 * y1) 
  (hB_on_parabola : x2^2 = 4 * y2) 
  (h_distance_condition : abs (distance (x1, y1) (0, 1) - distance (x2, y2) (0, 1)) = 2) 
  : y1 + x1^2 - y2 - x2^2 = 10 :=
begin
  -- proof steps would go here
  sorry,
end

end parabola_and_points_l693_693482


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693995

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693995


namespace division_of_field_l693_693159

theorem division_of_field :
  (∀ (hectares : ℕ) (parts : ℕ), hectares = 5 ∧ parts = 8 →
  (1 / parts = 1 / 8) ∧ (hectares / parts = 5 / 8)) :=
by
  sorry


end division_of_field_l693_693159


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693991

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693991


namespace frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693094

theorem frequency_machine_A (total_A first_class_A : ℕ) (h_total_A: total_A = 200) (h_first_class_A: first_class_A = 150) :
  first_class_A / total_A = 3 / 4 := by
  rw [h_total_A, h_first_class_A]
  norm_num

theorem frequency_machine_B (total_B first_class_B : ℕ) (h_total_B: total_B = 200) (h_first_class_B: first_class_B = 120) :
  first_class_B / total_B = 3 / 5 := by
  rw [h_total_B, h_first_class_B]
  norm_num

theorem chi_square_test_significance (n a b c d : ℕ) (h_n: n = 400) (h_a: a = 150) (h_b: b = 50) 
  (h_c: c = 120) (h_d: d = 80) :
  let K_squared := (n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d)))
  in K_squared > 6.635 := by
  rw [h_n, h_a, h_b, h_c, h_d]
  let num := 400 * (150 * 80 - 50 * 120)^2
  let denom := (150 + 50) * (120 + 80) * (150 + 120) * (50 + 80)
  have : K_squared = num / denom := rfl
  norm_num at this
  sorry

end frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693094


namespace second_quadrant_angles_l693_693309

def is_in_second_quadrant (θ : ℝ) : Prop :=
  (90 < θ ∧ θ < 180) ∨ (450 < θ ∧ θ < 540)

def given_angles_set : set ℝ :=
  {-120, -240, 180, 495}

theorem second_quadrant_angles :
  {θ | θ ∈ given_angles_set ∧ is_in_second_quadrant θ} = {-240, 495} := by
  sorry

end second_quadrant_angles_l693_693309


namespace largest_prime_factor_of_7fact_8fact_l693_693903

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693903


namespace three_true_propositions_l693_693787

variable {Plane : Type}
variable {Line : Type}
variable (alpha beta gamma : Plane)
variable (m n : Line)

def proposition_1 (h1 : alpha ⊥ gamma) (h2 : beta ∥ gamma) : Prop :=
  alpha ⊥ beta

def proposition_2 (h1 : alpha ∥ gamma) (h2 : beta ∥ gamma) : Prop :=
  alpha ∥ beta

def proposition_3 (h1 : m ∥ alpha) (h2 : n ∥ alpha) : Prop :=
  m ∥ n

def proposition_4 (h1 : alpha ⊥ gamma) (h2 : beta ⊥ gamma) (h3 : alpha ∩ beta = m) : Prop :=
  m ⊥ gamma

theorem three_true_propositions :
  (∃ (h1 : alpha ⊥ gamma) (h2 : beta ∥ gamma), proposition_1 alpha beta gamma h1 h2) ∧
  (∃ (h1 : alpha ∥ gamma) (h2 : beta ∥ gamma), proposition_2 alpha beta gamma h1 h2) ∧
  ¬ (∀ (h1 : m ∥ alpha) (h2 : n ∥ alpha), proposition_3 m n alpha h1 h2) ∧
  (∃ (h1 : alpha ⊥ gamma) (h2 : beta ⊥ gamma) (h3 : alpha ∩ beta = m), proposition_4 alpha beta gamma m h1 h2 h3) :=
sorry

end three_true_propositions_l693_693787


namespace finite_tasty_integers_l693_693591

def is_terminating_decimal (a b : ℕ) : Prop :=
  ∃ (c : ℕ), (b = c * 2^a * 5^a)

def is_tasty (n : ℕ) : Prop :=
  n > 2 ∧ ∀ (a b : ℕ), a + b = n → (is_terminating_decimal a b ∨ is_terminating_decimal b a)

theorem finite_tasty_integers : 
  ∃ (N : ℕ), ∀ (n : ℕ), n > N → ¬ is_tasty n :=
sorry

end finite_tasty_integers_l693_693591


namespace yura_roma_sums_are_equal_l693_693134

theorem yura_roma_sums_are_equal (n : ℕ) :
  let yura_squares := (finset.range 1010).sum (λ i, (n + i)^2)
  let roma_squares := (finset.range 1009).sum (λ i, (n + 1010 + i)^2)
  yura_squares = roma_squares :=
by
  sorry

end yura_roma_sums_are_equal_l693_693134


namespace simplify_fraction_l693_693441

theorem simplify_fraction (num denom : ℕ) (h_num : num = 90) (h_denom : denom = 150) : 
  num / denom = 3 / 5 := by
  rw [h_num, h_denom]
  norm_num
  sorry

end simplify_fraction_l693_693441


namespace how_many_quarters_did_dad_give_l693_693427

theorem how_many_quarters_did_dad_give (quarters_initial quarters_total : ℕ) (h1 : quarters_initial = 21) (h2 : quarters_total = 70) :
  ∃ quarters_given : ℕ, quarters_given = quarters_total - quarters_initial ∧ quarters_given = 49 :=
by
  use (quarters_total - quarters_initial)
  split
  sorry
  sorry

end how_many_quarters_did_dad_give_l693_693427


namespace area_of_trapezoid_EFGH_l693_693426

def E := (0 : ℝ, 0 : ℝ)
def F := (5 : ℝ, 0 : ℝ)
def G := (5 : ℝ, 6 : ℝ)
def H := (15 : ℝ, 6 : ℝ)

def EF : ℝ := 5
def FG : ℝ := 6
def GH : ℝ := 10
def HE : ℝ := 8 

def right_angle := (F.1 - E.1) * (G.1 - F.1) + (F.2 - E.2) * (G.2 - F.2) = 0

theorem area_of_trapezoid_EFGH : right_angle → (1 / 2) * (EF + GH) * FG = 45 := 
by 
  sorry

end area_of_trapezoid_EFGH_l693_693426


namespace prime_factor_count_l693_693142

theorem prime_factor_count (n : ℕ) (hn : nat.factors_count n = 211) : 
  ∃ p : ℕ, prime p ∧ n = p ^ 210 :=
sorry

end prime_factor_count_l693_693142


namespace smallest_possible_value_of_distance_l693_693013

noncomputable def smallest_distance (z w : ℂ) : ℝ :=
  complex.abs (z - w)

theorem smallest_possible_value_of_distance
  (z w : ℂ)
  (hz : complex.abs (z + 2 + 4 * complex.I) = 2)
  (hw : complex.abs (w - (6 + 7 * complex.I)) = 4) :
  smallest_distance z w = real.sqrt 185 - 6 := sorry

end smallest_possible_value_of_distance_l693_693013


namespace problem_in_lean_l693_693311

variable (f : ℝ → ℝ)

def F (x : ℝ) : ℝ := f x / Real.exp x

axiom f_derivative (x : ℝ) : deriv f x < f x

theorem problem_in_lean : f 2 < Real.exp 2 * f 0 ∧ f 2012 < Real.exp 2012 * f 0 := by
  sorry

end problem_in_lean_l693_693311


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693924

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693924


namespace correct_operation_l693_693525

variables {a b : ℝ}

theorem correct_operation : (5 * a * b - 6 * a * b = -1 * a * b) := by
  sorry

end correct_operation_l693_693525


namespace circle_intersection_range_a_l693_693338

theorem circle_intersection_range_a :
  (∃ (a : ℝ), ∀ (x y : ℝ), (x + 2)^2 + (y - a)^2 = 1 ∧ (x - a)^2 + (y - 5)^2 = 16) → (1 < a ∧ a < 2) :=
begin
  sorry,
end

end circle_intersection_range_a_l693_693338


namespace part1_part2_l693_693760

-- Definitions used in problems
structure Point where
  x : ℝ
  y : ℝ

def vector (p1 p2 : Point) : (ℝ × ℝ) := (p2.x - p1.x, p2.y - p1.y)

def parallel (v1 v2 : (ℝ × ℝ)) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

def orthogonal (v1 v2 : (ℝ × ℝ)) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

noncomputable def magnitude (v : (ℝ × ℝ)) := real.sqrt (v.1^2 + v.2^2)

noncomputable def y_value (θ t : ℝ) := cos θ ^ 2 - cos θ + (t/4)^2

-- Part 1 statement
theorem part1 (θ t : ℝ) (A B : Point)
  (ha : vector (Point.mk 0 0) A = (2, 1))
  (hA : A = Point.mk 1 0)
  (hB : B = Point.mk (cos θ) t)
  (h_parallel : parallel (2, 1) (vector A B))
  (h_magnitude : magnitude (vector A B) = real.sqrt 5 * magnitude (vector (Point.mk 0 0) A)) :
  vector (Point.mk 0 0) B = (-1, -1) :=
sorry

-- Part 2 statement
theorem part2 (θ t : ℝ)
  (ha : (2, 1))
  (A : Point)
  (hA : A = Point.mk 1 0)
  (hB : B = Point.mk (cos θ) t)
  (h_orthogonal : orthogonal (2, 1) (vector A B)) :
  ∃ θ t, y_value θ t = -(1/5) :=
sorry

end part1_part2_l693_693760


namespace reach_one_from_45_reach_one_from_345_reach_one_from_any_nat_l693_693533

theorem reach_one_from_45 : ∃ (n : ℕ), n = 1 :=
by
  -- Start from 45 and follow the given steps to reach 1.
  sorry

theorem reach_one_from_345 : ∃ (n : ℕ), n = 1 :=
by
  -- Start from 345 and follow the given steps to reach 1.
  sorry

theorem reach_one_from_any_nat (n : ℕ) (h : n ≠ 0) : ∃ (k : ℕ), k = 1 :=
by
  -- Prove that starting from any non-zero natural number, you can reach 1.
  sorry

end reach_one_from_45_reach_one_from_345_reach_one_from_any_nat_l693_693533


namespace red_ball_count_l693_693849

theorem red_ball_count (n_white : ℕ) (white_ratio red_ratio : ℕ) (ratio_condition : white_ratio = 3 ∧ red_ratio = 2) (white_balls : n_white = 9) : Σ (n_red : ℕ), n_red = 6 :=
by
  sorry

end red_ball_count_l693_693849


namespace min_digits_for_fraction_l693_693215

theorem min_digits_for_fraction :
  let frac := (987654321 : ℚ) / (2^30 * 5^3)
  min_digits_required frac = 30 :=
begin
  sorry
end

end min_digits_for_fraction_l693_693215


namespace bobby_truck_gasoline_consumption_rate_l693_693607

variable {initial_gasoline : ℝ}
variable {final_gasoline : ℝ}
variable {dist_to_supermarket : ℝ}
variable {dist_to_farm : ℝ}
variable {dist_into_farm_trip : ℝ}
variable {returned_dist : ℝ}
variable {total_miles_driven : ℝ}
variable {total_gasoline_used : ℝ}
variable {rate_of_consumption : ℝ}

-- Conditions given in the problem
axiom initial_gasoline_is_12 : initial_gasoline = 12
axiom final_gasoline_is_2 : final_gasoline = 2
axiom dist_home_to_supermarket : dist_to_supermarket = 5
axiom dist_home_to_farm : dist_to_farm = 6
axiom dist_home_to_turnaround : dist_into_farm_trip = 2
axiom returned_distance : returned_dist = dist_into_farm_trip * 2

-- Distance calculations based on problem description
def dist_to_supermarket_round_trip : ℝ := dist_to_supermarket * 2
def dist_home_to_turnaround_round_trip : ℝ := returned_dist
def full_farm_trip : ℝ := dist_to_farm

-- Total Distance Calculation
axiom total_distance_is_22 : total_miles_driven = 
  dist_to_supermarket_round_trip + dist_home_to_turnaround_round_trip + full_farm_trip
axiom total_gasoline_used_is_10 : total_gasoline_used = initial_gasoline - final_gasoline

-- Question: Prove the rate of consumption is 2.2 miles per gallon
def rate_of_consumption_calculation (total_miles : ℝ) (total_gas : ℝ) : ℝ :=
  total_miles / total_gas

theorem bobby_truck_gasoline_consumption_rate :
    rate_of_consumption_calculation total_miles_driven total_gasoline_used = 2.2 := 
  sorry

end bobby_truck_gasoline_consumption_rate_l693_693607


namespace smallest_sum_B_b_l693_693326

theorem smallest_sum_B_b : ∃ (B b : ℕ), (BBB_to_base_10 B = base_b_to_decimal B b) ∧ B < 4 ∧ b > 5 ∧ B + b = 11 :=
by 
  let BBB_to_base_10 (B : ℕ) := 21 * B
  let base_b_to_decimal (B : ℕ) (b : ℕ) := 4 * (b + 1)
  sorry

end smallest_sum_B_b_l693_693326


namespace correct_operation_l693_693132

-- Define the operations given in the conditions
def optionA (m : ℝ) := m^2 + m^2 = 2 * m^4
def optionB (a : ℝ) := a^2 * a^3 = a^5
def optionC (m n : ℝ) := (m * n^2) ^ 3 = m * n^6
def optionD (m : ℝ) := m^6 / m^2 = m^3

-- Theorem stating that option B is the correct operation
theorem correct_operation (a m n : ℝ) : optionB a :=
by sorry

end correct_operation_l693_693132


namespace problem1_problem2_l693_693205

-- Problem 1
theorem problem1 : (-(64 : ℤ) : ℚ)^(1/3 : ℚ) + (16 : ℚ)^(1/2 : ℚ) * ((9 : ℚ) / 4)^(1/2 : ℚ) + (-((2 : ℚ)^(1/2 : ℚ)))^2 = 4 :=
by sorry

-- Problem 2
theorem problem2 : (27 : ℚ)^(1/3 : ℚ) - (0 : ℚ)^(1/2 : ℚ) + (1 / 8 : ℚ)^(1/3 : ℚ) = 3.5 :=
by sorry

end problem1_problem2_l693_693205


namespace value_of_expression_l693_693686

theorem value_of_expression (m : ℝ) (α : ℝ) (h : m < 0) (h_M : M = (3 * m, -m)) :
  let sin_alpha := -m / (Real.sqrt 10 * -m)
  let cos_alpha := 3 * m / (Real.sqrt 10 * -m)
  (1 / (2 * sin_alpha * cos_alpha + cos_alpha^2) = 10 / 3) :=
by
  sorry

end value_of_expression_l693_693686


namespace projective_map_exists_theorem_l693_693039

noncomputable def projective_map_exists
  (l0 l : Type) [ProjectiveLine l0] [ProjectiveLine l]
  (A0 B0 C0 : l0) (A B C : l) : Prop :=
  ∃ (P : l0 → l), projective_map P ∧ P A0 = A ∧ P B0 = B ∧ P C0 = C

-- Main theorem statement
theorem projective_map_exists_theorem
  (l0 l : Type) [ProjectiveLine l0] [ProjectiveLine l]
  (A0 B0 C0 : l0) (A B C : l) :
  projective_map_exists l0 l A0 B0 C0 A B C :=
sorry

end projective_map_exists_theorem_l693_693039


namespace frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693091

theorem frequency_machine_A (total_A first_class_A : ℕ) (h_total_A: total_A = 200) (h_first_class_A: first_class_A = 150) :
  first_class_A / total_A = 3 / 4 := by
  rw [h_total_A, h_first_class_A]
  norm_num

theorem frequency_machine_B (total_B first_class_B : ℕ) (h_total_B: total_B = 200) (h_first_class_B: first_class_B = 120) :
  first_class_B / total_B = 3 / 5 := by
  rw [h_total_B, h_first_class_B]
  norm_num

theorem chi_square_test_significance (n a b c d : ℕ) (h_n: n = 400) (h_a: a = 150) (h_b: b = 50) 
  (h_c: c = 120) (h_d: d = 80) :
  let K_squared := (n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d)))
  in K_squared > 6.635 := by
  rw [h_n, h_a, h_b, h_c, h_d]
  let num := 400 * (150 * 80 - 50 * 120)^2
  let denom := (150 + 50) * (120 + 80) * (150 + 120) * (50 + 80)
  have : K_squared = num / denom := rfl
  norm_num at this
  sorry

end frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693091


namespace senate_subcommittee_l693_693160

/-- 
Proof of the number of ways to form a Senate subcommittee consisting of 7 Republicans
and 2 Democrats from the available 12 Republicans and 6 Democrats.
-/
theorem senate_subcommittee (R D : ℕ) (choose_R choose_D : ℕ) (hR : R = 12) (hD : D = 6) 
  (h_choose_R : choose_R = 7) (h_choose_D : choose_D = 2) : 
  (Nat.choose R choose_R) * (Nat.choose D choose_D) = 11880 := by
  sorry

end senate_subcommittee_l693_693160


namespace square_inscribed_perimeter_l693_693576

theorem square_inscribed_perimeter (r : ℝ) (h : r = 1) : ∃ p : ℝ, p = 4 * Real.sqrt 2 :=
by
  have h_diam : 2 * r = 2 := by rw h
  have h_side : ∃ s : ℝ, s * Real.sqrt 2 = 2 := by
    use (2 / Real.sqrt 2)
    field_simp
  use 4 * Real.sqrt 2
  sorry

end square_inscribed_perimeter_l693_693576


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693913

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693913


namespace hyperbola_equation_l693_693707

noncomputable def foci_hyperbola := (λ c : ℝ, (F1 : ℝ × ℝ, F2 : ℝ × ℝ) =>
  F1 = (-real.sqrt c, 0) ∧ F2 = (real.sqrt c, 0))

noncomputable def asymptotes_hyperbola := (λ m : ℝ, (asym1 : ℝ → ℝ, asym2 : ℝ → ℝ) =>
  asym1 = (λ x, m * x) ∧ asym2 = (λ x, -m * x))

theorem hyperbola_equation (c m a b : ℝ) (h_foci : foci_hyperbola c (-real.sqrt 10, 0) (real.sqrt 10, 0))
  (h_asym : asymptotes_hyperbola (1/2) (λ x, (1/2) * x) (λ x, -(1/2) * x))
  (h1 : a^2 + b^2 = c)
  (h2 : m = b / a) :
  (a = 2 * real.sqrt 2) ∧ (b = real.sqrt 2) →
  ∀ x y, (x : ℝ) * (x / (2 * real.sqrt 2)) - (y : ℝ) * (y / real.sqrt 2) = 1 :=
sorry

end hyperbola_equation_l693_693707


namespace milan_rate_per_minute_l693_693260

-- Definitions based on the conditions
def monthly_fee : ℝ := 2.0
def total_bill : ℝ := 23.36
def total_minutes : ℕ := 178
def expected_rate_per_minute : ℝ := 0.12

-- Theorem statement based on the question
theorem milan_rate_per_minute :
  (total_bill - monthly_fee) / total_minutes = expected_rate_per_minute := 
by 
  sorry

end milan_rate_per_minute_l693_693260


namespace general_term_a_sum_of_bn_l693_693786

-- Define sequences a_n and b_n
noncomputable def a (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b (n : ℕ) : ℚ := 1 / ((2 * n + 1) * (2 * n + 3))

-- Conditions
lemma condition_1 (n : ℕ) : a n > 0 := by sorry
lemma condition_2 (n : ℕ) : (a n)^2 + 2 * (a n) = 4 * (n * (n + 1)) + 3 := 
  by sorry

-- Theorem for question 1
theorem general_term_a (n : ℕ) : a n = 2 * n + 1 := by sorry

-- Theorem for question 2
theorem sum_of_bn (n : ℕ) : 
  (Finset.range n).sum b = (n : ℚ) / (6 * n + 9) := by sorry

end general_term_a_sum_of_bn_l693_693786


namespace simplify_fraction_l693_693436

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end simplify_fraction_l693_693436


namespace workers_in_first_shift_l693_693648

-- Define the conditions
def employees_in_shifts
  (total_first_shift : ℕ)   -- F
  (total_second_shift : ℕ := 50)
  (total_third_shift : ℕ := 40)
  (pension_percent_first_shift : ℚ := 0.20)
  (pension_percent_second_shift : ℚ := 0.40)
  (pension_percent_third_shift : ℚ := 0.10)
  (total_pension_percent : ℚ := 0.24) : Prop :=
  (0.20 * (total_first_shift : ℚ) + 
   0.40 * (total_second_shift : ℚ) + 
   0.10 * (total_third_shift : ℚ) = 
   0.24 * (total_first_shift + total_second_shift + total_third_shift : ℚ))

-- The theorem we need to prove
theorem workers_in_first_shift: 
  ∃ F : ℕ, employees_in_shifts F ∧ F = 60 :=
begin
  -- the body of the proof goes here
  sorry
end

end workers_in_first_shift_l693_693648


namespace last_two_digits_l693_693518

theorem last_two_digits :
  let a_n := λ (n : ℕ), 4 * n + 7 in
  (a_n 1 = 11) ∧ (a_n 500 = 2007) ∧
  (∑ n in Finset.range 500, (a_n n) ^ 2) ^ 2 % 100 = 0 :=
by
  -- Definitions for arithmetic sequence
  let a_n := λ (n : ℕ), 4 * n + 7
  have seq_1 : a_n 1 = 11 := by simp [a_n]
  have seq_last : a_n 500 = 2007 := by simp [a_n]
  
  -- Convert the problem into the sum form and square it modulo 100
  have h : (∑ n in Finset.range 500, (a_n n) ^ 2) ^ 2 % 100 = 0 :=
  sorry

  -- The theorem to prove
  exact ⟨seq_1, seq_last, h⟩

end last_two_digits_l693_693518


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693918

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693918


namespace question1_question2_l693_693323

-- Definitions used in the conditions
def vector_a : ℝ × ℝ := (2, -1)
def vector_b_part1 : ℝ × ℝ := (1, 7)
def vector_b_part2 : ℝ × ℝ := (1, -3)
def length (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
def cos_angle (v1 v2 : ℝ × ℝ) : ℝ := dot_product v1 v2 / (length v1 * length v2)

-- Question 1: Prove the magnitude of vector_b given orthogonality condition
theorem question1 : dot_product vector_a (vector_a.1 + vector_b_part1.1, vector_a.2 + vector_b_part1.2) = 0 → length vector_b_part1 = 5 * real.sqrt 2 :=
by
  sorry

-- Question 2: Prove the angle between vectors given linear combination condition
theorem question2 : (vector_a.1 + 2 * vector_b_part2.1, vector_a.2 + 2 * vector_b_part2.2) = (4, -7) → real.arccos (cos_angle vector_a vector_b_part2) = real.pi / 4 :=
by
  sorry

end question1_question2_l693_693323


namespace other_number_remainder_l693_693521

theorem other_number_remainder (x : ℕ) (k n : ℤ) (hx : x > 0) (hk : 200 = k * x + 2) (hnk : n ≠ k) : ∃ m : ℤ, (n * ↑x + 2) = m * ↑x + 2 ∧ (n * ↑x + 2) % x = 2 := 
by
  sorry

end other_number_remainder_l693_693521


namespace ratio_payment_shared_side_l693_693613

variable (length_side length_back : ℕ) (cost_per_foot cole_payment : ℕ)
variables (neighbor_back_contrib neighbor_left_contrib total_cost_fence : ℕ)
variables (total_cost_shared_side : ℕ)

theorem ratio_payment_shared_side
  (h1 : length_side = 9)
  (h2 : length_back = 18)
  (h3 : cost_per_foot = 3)
  (h4 : cole_payment = 72)
  (h5 : neighbor_back_contrib = (length_back / 2) * cost_per_foot)
  (h6 : total_cost_fence = (2* length_side + length_back) * cost_per_foot)
  (h7 : total_cost_shared_side = length_side * cost_per_foot)
  (h8 : cole_left_total_payment = cole_payment + neighbor_back_contrib)
  (h9 : neighbor_left_contrib = cole_left_total_payment - cole_payment):
  neighbor_left_contrib / total_cost_shared_side = 1 := 
sorry

end ratio_payment_shared_side_l693_693613


namespace parallelogram_problem_l693_693759

noncomputable def vertex_O : (ℝ × ℝ) := (0, 0)
noncomputable def vertex_A : (ℝ × ℝ) := (3, 6)
noncomputable def vertex_B : (ℝ × ℝ) := (8, 6)

-- Distance between two points (x1, y1) and (x2, y2)
def distance (P Q : ℝ × ℝ) : ℝ := sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

-- Proof statement
theorem parallelogram_problem :
  (let O := vertex_O in
   let A := vertex_A in
   let B := vertex_B in
   let C := (5, 0) in
   let AC_ratio := distance O B / distance A C in
   AC_ratio = sqrt 2.5 ∧
   ((∀ x : ℝ, OC := (λ x : ℝ, 0)) = (λ x, 0)) ∧
   ((∀ x : ℝ, AB := (λ x : ℝ, 6)) = (λ x, 6)) ∧
   ((∀ x : ℝ, OA := (λ x : ℝ, 2*x)) = (λ x, 2 * x)) ∧
   ((∀ x : ℝ, BC := (λ x : ℝ, 2*x - 10)) = (λ x, 2 * x - 10)) ∧
   ((∀ x : ℝ, AC := (λ x : ℝ, -3*x + 15)) = (λ x, -3 * x + 15))) := by
sorry

end parallelogram_problem_l693_693759


namespace compute_63_times_57_l693_693615

theorem compute_63_times_57 : 63 * 57 = 3591 := 
by {
   have h : (60 + 3) * (60 - 3) = 60^2 - 3^2, from
     by simp [mul_add, add_mul, add_assoc, sub_mul, mul_sub, sub_add, sub_sub, add_sub, mul_self_sub],
   have h1 : 60^2 = 3600, from rfl,
   have h2 : 3^2 = 9, from rfl,
   have h3 : 60^2 - 3^2 = 3600 - 9, by rw [h1, h2],
   rw h at h3,
   exact h3,
}

end compute_63_times_57_l693_693615


namespace impossible_all_black_l693_693047

def initial_white_chessboard (n : ℕ) : Prop :=
  n = 0

def move_inverts_three (move : ℕ → ℕ) : Prop :=
  ∀ n, move n = n + 3 ∨ move n = n - 3

theorem impossible_all_black (move : ℕ → ℕ) (n : ℕ) (initial : initial_white_chessboard n) (invert : move_inverts_three move) : ¬ ∃ k, move^[k] n = 64 :=
by sorry

end impossible_all_black_l693_693047


namespace log_product_value_l693_693010

theorem log_product_value :
  let z := (Real.log 4 / Real.log 2) *
           (Real.log 8 / Real.log 4) *
           (Real.log 16 / Real.log 8) *
           (Real.log 32 / Real.log 16)
  in z = 5 :=
by
  let z := (Real.log 4 / Real.log 2) *
           (Real.log 8 / Real.log 4) *
           (Real.log 16 / Real.log 8) *
           (Real.log 32 / Real.log 16)
  exact sorry

end log_product_value_l693_693010


namespace arithmetic_mean_of_ints_from_neg5_to_6_l693_693874

def int_range := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
def mean (nums : List Int) : Float := (nums.sum : Float) / (nums.length : Float)

theorem arithmetic_mean_of_ints_from_neg5_to_6 : 
  mean int_range = 0.5 := by
    sorry

end arithmetic_mean_of_ints_from_neg5_to_6_l693_693874


namespace ceil_neg_3_7_l693_693229

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := by
  sorry

end ceil_neg_3_7_l693_693229


namespace constant_term_binomial_expansion_l693_693838

theorem constant_term_binomial_expansion :
  let general_term := λ r, (choose 6 r) * ((1/4 : ℝ)^6 * r) * ((-2 : ℝ) ^ r) * (1 / (r * sqrt x))
  (use_formula_to_find_r : ∃ r : ℤ, 6 - (3 / 2) * r = 0) → 
  (∃ r : ℕ, r = 4) →
  (constant_term_value = 15) :=
by
  -- The general term of the expansion of (x/4 - 2/sqrt(x))^6
  let general_term := λ r, (choose 6 r) * ((1/4 : ℝ)^(6 - r)) * ((-2 / sqrt x) ^ r)
  -- Solve for r where the term is constant
  assume h1 : ∃ r : ℕ, 6 - (3 / 2) * r = 0
  assume h2 : ∃ r : ℕ, r = 4
  have h3 : general_term 4 = 15
  sorry

end constant_term_binomial_expansion_l693_693838


namespace triangle_area_and_circumradius_l693_693186

theorem triangle_area_and_circumradius (a b c: ℝ) (h₁: a = 12) (h₂: b = 35) (h₃: c = 37) 
  (h_right_triangle: a^2 + b^2 = c^2) :
  let area := (1 / 2) * a * b in
  let circumradius := (1 / 2) * c in
  area = 210 ∧ circumradius = 37 / 2 := by
    -- Hypotheses and conditions
    have h₁ : a = 12 := by assumption,
    have h₂ : b = 35 := by assumption,
    have h₃ : c = 37 := by assumption,
    have h_right_triangle: a^2 + b^2 = c^2 := by assumption,
    -- Definitions and results to be proved
    let area := (1 / 2) * a * b,
    let circumradius := (1 / 2) * c,
    -- Required proofs to be completed
    sorry

end triangle_area_and_circumradius_l693_693186


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693996

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693996


namespace min_wins_for_playoffs_l693_693356

variable (wins : ℕ)
variable (total_games : ℕ := 32)
variable (points_for_win : ℕ := 3)
variable (points_for_loss : ℕ := -1)
variable (required_points : ℕ := 48)

theorem min_wins_for_playoffs (h : points_for_win * wins + points_for_loss * (total_games - wins) ≥ required_points) :
  wins ≥ 20 :=
by 
  sorry

end min_wins_for_playoffs_l693_693356


namespace emily_sixth_quiz_score_l693_693218

theorem emily_sixth_quiz_score (a1 a2 a3 a4 a5 : ℕ) (target_mean : ℕ) (sixth_score : ℕ) :
  a1 = 94 ∧ a2 = 97 ∧ a3 = 88 ∧ a4 = 90 ∧ a5 = 102 ∧ target_mean = 95 →
  sixth_score = (target_mean * 6 - (a1 + a2 + a3 + a4 + a5)) →
  sixth_score = 99 :=
by
  sorry

end emily_sixth_quiz_score_l693_693218


namespace ceil_neg_3_7_l693_693224

-- Define the ceiling function in Lean
def ceil (x : ℝ) : ℤ := int.ceil x

-- A predicate to represent the statement we want to prove
theorem ceil_neg_3_7 : ceil (-3.7) = -3 := by
  -- Provided conditions
  have h1 : ceil (-3.7) = int.ceil (-3.7) := rfl
  have h2 : int.ceil (-3.7) = -3 := by
    -- Lean's int.ceil function returns the smallest integer greater or equal to the input
    sorry  -- proof goes here

  -- The main statement
  exact h2

end ceil_neg_3_7_l693_693224


namespace complete_square_l693_693516

-- Definitions based on conditions
def row_sum_piece2 := 2 + 1 + 3 + 1
def total_sum_square := 4 * row_sum_piece2
def sum_piece1 := 7
def sum_piece2 := 8
def sum_piece3 := 8
def total_given_pieces := sum_piece1 + sum_piece2 + sum_piece3
def sum_missing_piece := total_sum_square - total_given_pieces

-- Statement to prove that the missing piece has the correct sum
theorem complete_square : (sum_missing_piece = 5) :=
by 
  -- It is a placeholder for the proof steps, the actual proof steps are not needed
  sorry

end complete_square_l693_693516


namespace ceil_neg_3_7_l693_693227

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := by
  sorry

end ceil_neg_3_7_l693_693227


namespace lamps_remain_lit_after_toggling_l693_693858

theorem lamps_remain_lit_after_toggling :
  let n := 1997
  let lcm_2_3_5 := Nat.lcm (Nat.lcm 2 3) 5
  let multiples_30 := n / 30
  let multiples_6 := n / (2 * 3)
  let multiples_15 := n / (3 * 5)
  let multiples_10 := n / (2 * 5)
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let pulled_three_times := multiples_30
  let pulled_twice := multiples_6 + multiples_15 + multiples_10 - 3 * pulled_three_times
  let pulled_once := multiples_2 + multiples_3 + multiples_5 - 2 * pulled_twice - 3 * pulled_three_times
  1997 - pulled_three_times - pulled_once = 999 := by
  let n := 1997
  let lcm_2_3_5 := Nat.lcm (Nat.lcm 2 3) 5
  let multiples_30 := n / 30
  let multiples_6 := n / (2 * 3)
  let multiples_15 := n / (3 * 5)
  let multiples_10 := n / (2 * 5)
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let pulled_three_times := multiples_30
  let pulled_twice := multiples_6 + multiples_15 + multiples_10 - 3 * pulled_three_times
  let pulled_once := multiples_2 + multiples_3 + multiples_5 - 2 * pulled_twice - 3 * pulled_three_times
  have h : 1997 - pulled_three_times - (pulled_once) = 999 := sorry
  exact h

end lamps_remain_lit_after_toggling_l693_693858


namespace inventory_total_base_10_l693_693163

theorem inventory_total_base_10 :
  let artifact_5 := 4213
  let sculpture_5 := 2431
  let coins_5 := 213
  let artifact_10 := 3*5^0 + 1*5^1 + 2*5^2 + 4*5^3
  let sculpture_10 := 1*5^0 + 3*5^1 + 4*5^2 + 2*5^3
  let coins_10 := 3*5^0 + 1*5^1 + 2*5^2
  in artifact_10 + sculpture_10 + coins_10 = 982 :=
by
  let artifact_5 := 4213
  let sculpture_5 := 2431
  let coins_5 := 213
  let artifact_10 := 3*5^0 + 1*5^1 + 2*5^2 + 4*5^3
  let sculpture_10 := 1*5^0 + 3*5^1 + 4*5^2 + 2*5^3
  let coins_10 := 3*5^0 + 1*5^1 + 2*5^2
  show artifact_10 + sculpture_10 + coins_10 = 982
  sorry

end inventory_total_base_10_l693_693163


namespace largest_prime_factor_l693_693984

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693984


namespace four_digit_numbers_from_2101_l693_693324

/-- The number of different four-digit numbers that can be formed by arranging 
the digits in 2101 is 9, provided that leading digits should not be 0 and repeating digits are considered. -/
theorem four_digit_numbers_from_2101 : 
  (number_of_unique_four_digit_numbers [2, 1, 0, 1] = 9) := 
sorry

/-- Define the function to count the number of unique four-digit numbers that can be formed
from the given digits, ensuring the leading digit is non-zero and accounting for repetitions. -/
def number_of_unique_four_digit_numbers (digits : List ℕ) : ℕ := 
sorry

end four_digit_numbers_from_2101_l693_693324


namespace find_number_l693_693073

-- Define the set of positive even integers less than a certain number that contain the digits 5 or 9.
def even_integers_with_5_or_9 (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x, ((x < n) ∧ (x % 2 = 0) ∧
     (x.toDigits 10).any (λ d, d = 5 ∨ d = 9)))
     (Finset.range n)

theorem find_number :
  ∃ n, even_integers_with_5_or_9 n = ({50, 52, 54, 56, 58, 90, 92, 94, 96, 98} : Finset ℕ) ∧
       n = 100 :=
by
  sorry

end find_number_l693_693073


namespace num_valid_10_digit_sequences_l693_693728

theorem num_valid_10_digit_sequences : 
  ∃ (n : ℕ), n = 64 ∧ 
  (∀ (seq : Fin 10 → Fin 3), 
    (∀ i : Fin 9, abs (seq i.succ - seq i) = 1) → 
    (∀ i : Fin 10, seq i < 3) →
    ∃ k : Nat, k = 10 ∧ seq 0 < 10 ∧ seq 1 < 10 ∧ seq 2 < 10 ∧ seq 3 < 10 ∧ 
      seq 4 < 10 ∧ seq 5 < 10 ∧ seq 6 < 10 ∧ seq 7 < 10 ∧ 
      seq 8 < 10 ∧ seq 9 < 10 ∧ k = 10 → n = 64) :=
sorry

end num_valid_10_digit_sequences_l693_693728


namespace gcd_of_14658_and_11241_l693_693660

open Nat

theorem gcd_of_14658_and_11241 :
  gcd 14658 11241 = 3 := by
  sorry

end gcd_of_14658_and_11241_l693_693660


namespace find_lambda_l693_693692

open Real

variables (k : ℝ) (n : ℕ) (a : ℕ → ℝ)
hypothesis (hk : k > 2)
hypothesis (hn : n ≥ 3)
hypothesis (ha_positive : ∀ i, 1 ≤ i ∧ i ≤ n → a i > 0)

theorem find_lambda (h : (finset.range n).sum (λ i, a i) * (finset.range n).sum (λ i, (a i)⁻¹) < (sqrt (k + 4/k + 5) + n - 3)^2) : 
  a 1 + a 2 < k * a 3 := 
sorry

end find_lambda_l693_693692


namespace frequencies_of_first_class_quality_difference_confidence_l693_693113

section quality_comparison

variables (n a b c d : ℕ)

-- Given conditions
def total_products : ℕ := 400
def machine_a_total : ℕ := 200
def machine_a_first : ℕ := 150
def machine_a_second : ℕ := 50
def machine_b_total : ℕ := 200
def machine_b_first : ℕ := 120
def machine_b_second : ℕ := 80

-- Defining the K^2 calculation formula
def K_squared : ℚ :=
  (total_products * (machine_a_first * machine_b_second - machine_a_second * machine_b_first) ^ 2 : ℚ) /
  ((machine_a_first + machine_a_second) * (machine_b_first + machine_b_second) * (machine_a_first + machine_b_first) * (machine_a_second + machine_b_second))

-- Proof statement for Q1: Frequencies of first-class products
theorem frequencies_of_first_class :
  machine_a_first / machine_a_total = 3 / 4 ∧ 
  machine_b_first / machine_b_total = 3 / 5 := 
sorry

-- Proof statement for Q2: Confidence level of difference in quality
theorem quality_difference_confidence :
  K_squared = 10.256 ∧ 10.256 > 6.635 → 0.99 :=
sorry

end quality_comparison

end frequencies_of_first_class_quality_difference_confidence_l693_693113


namespace water_height_is_20cb2_l693_693855

noncomputable def water_height_expression (radius height : ℝ) (full_percent : ℝ) : ℝ :=
  let volume_cone := (1 / 3) * Real.pi * radius^2 * height
  let volume_water := full_percent * volume_cone
  let x := (volume_water / volume_cone)^(1/3)
  height * x

theorem water_height_is_20cb2 :
  ∀ (radius height : ℝ) (full_percent : ℝ),
    radius = 20 → height = 100 → full_percent = 0.4 →
    (∃ a b : ℕ, ∧
      (height * (full_percent * (1 / 3) * Real.pi * radius^2 * height)^(1/3) = a * Real.cbrt b) ∧
      b = 2) ∧ a + b = 22 :=
by
  sorry

end water_height_is_20cb2_l693_693855


namespace cos_sin_fraction_l693_693269

theorem cos_sin_fraction (α β : ℝ) (h1 : Real.tan (α + β) = 2 / 5) 
                         (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := 
  sorry

end cos_sin_fraction_l693_693269


namespace calculate_single_trip_price_l693_693574

-- Definitions based on the problem conditions
def no_discount_limit := 200
def first_discount_limit := 500
def discount_90 := 0.90
def discount_70 := 0.70

def payment_total := 168 + 423 * discount_90 -- total payment from two trips

-- Final price calculation for a single trip scenario
def single_trip_price (total_price : ℝ) : ℝ :=
  if total_price <= no_discount_limit then
    total_price
  else if total_price <= first_discount_limit then
    total_price * discount_90
  else
    let first_part := first_discount_limit * discount_90
    let remaining := (total_price - first_discount_limit) * discount_70
    first_part + remaining

-- Given conditions for two trips
def total_price := 168 + (423 / discount_90 : ℝ) -- original price of the goods

-- The proof statement
theorem calculate_single_trip_price :
  single_trip_price total_price = 546.6 :=
by
  sorry

end calculate_single_trip_price_l693_693574


namespace sum_of_w_l693_693009

def g (y : ℝ) : ℝ := (2 * y)^3 - 2 * (2 * y) + 5

theorem sum_of_w (w1 w2 w3 : ℝ)
  (hw1 : g (2 * w1) = 13)
  (hw2 : g (2 * w2) = 13)
  (hw3 : g (2 * w3) = 13) :
  w1 + w2 + w3 = -1 / 4 :=
sorry

end sum_of_w_l693_693009


namespace renata_final_money_l693_693029

def initial_money := 10
def donation := 4
def prize := 90
def slot_machine_losses := [50, 10, 5]
def sunglasses_price := 15
def sunglasses_discount := 0.20
def water_cost := 1
def lottery_ticket_cost := 1
def lottery_prize := 65
def sandwich_price := 8
def sandwich_discount := 0.25
def latte_price := 4
def split_bill (total: ℕ) := total / 2

theorem renata_final_money :
  let remaining1 := initial_money - donation in
  let remaining2 := remaining1 + prize in
  let remaining3 := remaining2 - slot_machine_losses.sum in
  let sunglasses_cost := (sunglasses_price : ℕ) - (sunglasses_price * sunglasses_discount) in
  let remaining4 := remaining3 - sunglasses_cost in
  let remaining5 := remaining4 - water_cost - lottery_ticket_cost in
  let remaining6 := remaining5 + lottery_prize in
  let sandwich_cost := (sandwich_price : ℕ) - (sandwich_price * sandwich_discount) in
  let total_meal_cost := sandwich_cost + latte_price in
  let remaining7 := remaining6 - split_bill total_meal_cost in
  remaining7 = 77 :=
by
  sorry

end renata_final_money_l693_693029


namespace length_AB_l693_693750

-- Given a right triangle ABC with angle A equal to 90 degrees (π / 2 radians)
-- and the dot product of vectors AB and BC equal to -2,
-- prove that the length of vector AB is sqrt(2).
theorem length_AB (A B C : ℝ × ℝ) (hA : ∠ A B C = (π / 2)) (h_dot : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = -2) :
    dist A B = sqrt 2 :=
    sorry

end length_AB_l693_693750


namespace DE_squared_plus_EF_squared_plus_FD_squared_eq_five_l693_693365

theorem DE_squared_plus_EF_squared_plus_FD_squared_eq_five 
  (a b c : ℝ) (D E F : Point)
  (H : Point)
  (H1 : acute_triangle a b c)
  (H2 : altitude a D) (H3 : altitude b E) (H4 : altitude c F)
  (H5 : orthocenter H (triangle a b c))
  (H6 : bisects_area EF (triangle a b c))
  (H7 : a = 3) (H8 : b = 2 * Real.sqrt 2) (H9 : c = Real.sqrt 5):
  DE^2 + EF^2 + FD^2 = 5 :=
sorry

end DE_squared_plus_EF_squared_plus_FD_squared_eq_five_l693_693365


namespace polynomial_condition_l693_693246

theorem polynomial_condition (P : ℤ[X]) (hP : ∀ n : ℕ, n > 0 → P.eval n ≠ 0 ∧ ∃ k : ℤ, P.eval (10^(digits n) * n + n) = k * P.eval n) :
  ∃ m : ℕ, P = polynomial.C 1 * polynomial.X^m :=
begin
  sorry
end

end polynomial_condition_l693_693246


namespace sequence_arithmetic_not_geometric_l693_693268

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem sequence_arithmetic_not_geometric (a b c : ℝ) 
  (h1 : 3^a = 4) (h2 : 3^b = 12) (h3 : 3^c = 36) :
  (a + c = 2 * b) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) := by
sorry

end sequence_arithmetic_not_geometric_l693_693268


namespace simplify_fraction_90_150_l693_693448

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end simplify_fraction_90_150_l693_693448


namespace sixty_three_times_fifty_seven_l693_693628

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 :=
by
  let a := 60
  let b := 3
  have h : (a + b) * (a - b) = a^2 - b^2 := by sorry
  have h1 : 63 = a + b := by rfl
  have h2 : 57 = a - b := by rfl
  calc
    63 * 57 = (a + b) * (a - b) : by rw [h1, h2]
    ... = a^2 - b^2 : by rw h
    ... = 60^2 - 3^2 : by rfl
    ... = 3600 - 9 : by sorry
    ... = 3591 : by norm_num

end sixty_three_times_fifty_seven_l693_693628


namespace find_coordinates_of_B_l693_693282

-- Define points A and B, and vector a
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 5 }
def a : Point := { x := 2, y := 3 }

-- Define the proof problem
theorem find_coordinates_of_B (B : Point) 
  (h1 : B.x + 1 = 3 * a.x)
  (h2 : B.y - 5 = 3 * a.y) : 
  B = { x := 5, y := 14 } := 
sorry

end find_coordinates_of_B_l693_693282


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693910

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693910


namespace base_eight_to_base_ten_l693_693878

theorem base_eight_to_base_ten (n : ℕ) (h : n = 4 * 8^1 + 7 * 8^0) : n = 39 := by
  sorry

end base_eight_to_base_ten_l693_693878


namespace ceil_neg_3_7_l693_693231

theorem ceil_neg_3_7 : Real.ceil (-3.7) = -3 := by
  sorry

end ceil_neg_3_7_l693_693231


namespace frequencies_of_first_class_quality_difference_confidence_l693_693112

section quality_comparison

variables (n a b c d : ℕ)

-- Given conditions
def total_products : ℕ := 400
def machine_a_total : ℕ := 200
def machine_a_first : ℕ := 150
def machine_a_second : ℕ := 50
def machine_b_total : ℕ := 200
def machine_b_first : ℕ := 120
def machine_b_second : ℕ := 80

-- Defining the K^2 calculation formula
def K_squared : ℚ :=
  (total_products * (machine_a_first * machine_b_second - machine_a_second * machine_b_first) ^ 2 : ℚ) /
  ((machine_a_first + machine_a_second) * (machine_b_first + machine_b_second) * (machine_a_first + machine_b_first) * (machine_a_second + machine_b_second))

-- Proof statement for Q1: Frequencies of first-class products
theorem frequencies_of_first_class :
  machine_a_first / machine_a_total = 3 / 4 ∧ 
  machine_b_first / machine_b_total = 3 / 5 := 
sorry

-- Proof statement for Q2: Confidence level of difference in quality
theorem quality_difference_confidence :
  K_squared = 10.256 ∧ 10.256 > 6.635 → 0.99 :=
sorry

end quality_comparison

end frequencies_of_first_class_quality_difference_confidence_l693_693112


namespace area_sine_curve_l693_693656

theorem area_sine_curve : ∫ x in 0..π, sin x = 2 :=
by
  sorry

end area_sine_curve_l693_693656


namespace common_area_solution_l693_693364

universe u

def is_isosceles (O A B : ℝ × ℝ) : Prop := 
  let dist := λ P Q: ℝ × ℝ, (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2
  dist O A = dist A B

def in_first_quadrant (A : ℝ × ℝ) : Prop := A.1 > 0 ∧ A.2 > 0
def on_x_axis (B : ℝ × ℝ) : Prop := B.2 = 0

def area_of_triangle (O A B : ℝ × ℝ) : ℝ := 
  (O.1 * (A.2 - B.2) + A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2)) / 2

noncomputable def common_area (A B : ℝ × ℝ) (s : ℝ) : ℝ :=
  if s ≤ 1 then s
  else s - real.sqrt (s ^ 2 - s) - real.log (real.sqrt s - real.sqrt (s - 1))

theorem common_area_solution (O A B : ℝ × ℝ) (s : ℝ) 
  (h_isosceles : is_isosceles O A B) 
  (h_in_first : in_first_quadrant A) 
  (h_on_x_axis : on_x_axis B) 
  (h_area : area_of_triangle O A B = s) : 
  ∃ c : ℝ, common_area A B s = c :=
sorry

end common_area_solution_l693_693364


namespace square_b_perimeter_l693_693046

/-- Square A has an area of 121 square centimeters. Square B has a certain perimeter.
  If square B is placed within square A and a random point is chosen within square A,
  the probability that the point is not within square B is 0.8677685950413223.
  Prove the perimeter of square B is 16 centimeters. -/
theorem square_b_perimeter (area_A : ℝ) (prob : ℝ) (perimeter_B : ℝ) 
  (h1 : area_A = 121)
  (h2 : prob = 0.8677685950413223)
  (h3 : ∃ (a b : ℝ), area_A = a * a ∧ a * a - b * b = prob * area_A) :
  perimeter_B = 16 :=
sorry

end square_b_perimeter_l693_693046


namespace tetrahedron_planes_count_l693_693643

def tetrahedron_planes : ℕ :=
  let vertices := 4
  let midpoints := 6
  -- The total number of planes calculated by considering different combinations
  4      -- planes formed by three vertices
  + 6    -- planes formed by two vertices and one midpoint
  + 12   -- planes formed by one vertex and two midpoints
  + 7    -- planes formed by three midpoints

theorem tetrahedron_planes_count :
  tetrahedron_planes = 29 :=
by
  sorry

end tetrahedron_planes_count_l693_693643


namespace weekly_charge_for_motel_l693_693206

theorem weekly_charge_for_motel (W : ℝ) (h1 : ∀ t : ℝ, t = 3 * 4 → t = 12)
(h2 : ∀ cost_weekly : ℝ, cost_weekly = 12 * W)
(h3 : ∀ cost_monthly : ℝ, cost_monthly = 3 * 1000)
(h4 : cost_monthly + 360 = 12 * W) : 
W = 280 := 
sorry

end weekly_charge_for_motel_l693_693206


namespace abc_division_l693_693414

theorem abc_division (A B C S C1 C2: Point) (CA_B: Line CA) (CB_B: Line CB) 
(SC: Line S C) : 
triangle A B C 
→ square_on CA B outside
→ square_on CB B outside
→ opposite_sides_intersect CA_B CB_B S
→ ∃ C1 C2, S ∈ line_through C1 C2
→ ratio_divides (SC divides A B) ((AC len)^2) ((BC len)^2) :=
begin
  sorry
end

end abc_division_l693_693414


namespace derivative_sin_2x_l693_693839

-- Definition of the function y
def y (x : ℝ) : ℝ := sin (2 * x)

-- Theorem statement: Proving the derivative of y with respect to x.
theorem derivative_sin_2x : ∀ x : ℝ, deriv y x = 2 * cos (2 * x) :=
by
  sorry

end derivative_sin_2x_l693_693839


namespace overall_average_length_of_ropes_l693_693860

theorem overall_average_length_of_ropes :
  let ropes := 6
  let third_part := ropes / 3
  let average1 := 70
  let average2 := 85
  let length1 := third_part * average1
  let length2 := (ropes - third_part) * average2
  let total_length := length1 + length2
  let overall_average := total_length / ropes
  overall_average = 80 := by
sorry

end overall_average_length_of_ropes_l693_693860


namespace simplify_expr_l693_693044

noncomputable def example_expr : ℚ :=
  ((2 / 3) ^ 0) + (2 ^ (-2)) * ((9 / 16) ^ (-1 / 2)) + (Real.log10 8 + Real.log10 125)

/-- Proof that the simplification of the given mathematical expression equals 13 / 3. -/
theorem simplify_expr : example_expr = 13 / 3 :=
by
  sorry

end simplify_expr_l693_693044


namespace largest_prime_factor_7fac_plus_8fac_l693_693952

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693952


namespace min_tan_halfangle_C_l693_693770

theorem min_tan_halfangle_C (A B C : ℝ) (hABC : A + B + C = 180) 
  (h_condition : tan (A / 2) + tan (B / 2) = 1) : 
  tan (C / 2) ≥ 3 / 4 :=
sorry

end min_tan_halfangle_C_l693_693770


namespace buckets_required_l693_693500

theorem buckets_required (C : ℝ) (hC : C > 0) :
  let original_bucket_count := 25
  let reduction_factor := 2 / 5
  let new_bucket_count := original_bucket_count / reduction_factor
  new_bucket_count.ceil = 63 :=
by
  -- Definitions and conditions
  let original_bucket_count := 25
  let reduction_factor := 2 / 5
  let total_capacity := original_bucket_count * C
  let new_bucket_capacity := reduction_factor * C
  let new_bucket_count := total_capacity / new_bucket_capacity
  
  -- Main goal
  have : new_bucket_count = (25 * C) / ((2 / 5) * C) := by sorry
  have : new_bucket_count = 25 / (2 / 5) := by sorry
  have : new_bucket_count = 25 * (5 \ 2) := by sorry
  have : new_bucket_count = 62.5 := by sorry
  exact ceil_eq 63 _.mpr sorry

end buckets_required_l693_693500


namespace average_speed_last_segment_l693_693375

-- Definitions based on conditions
def total_distance : ℝ := 150
def total_time : ℝ := 2.5
def first_hour_speed : ℝ := 50
def second_hour_speed : ℝ := 55
def stop_time : ℝ := 15 / 60  -- 15 minutes converted to hours
def last_segment_time : ℝ := 0.5  -- 30 minutes converted to hours

-- Derived distances
def first_segment_distance : ℝ := first_hour_speed * 1
def second_segment_distance : ℝ := second_hour_speed * 1
def covered_distance : ℝ := first_segment_distance + second_segment_distance
def last_segment_distance : ℝ := total_distance - covered_distance

-- Theorem to prove the average speed for the last segment
theorem average_speed_last_segment :
  (last_segment_distance / last_segment_time) = 90 :=
by
  -- Calculations can be done in the proof part, which is skipped by sorry
  sorry

end average_speed_last_segment_l693_693375


namespace minimum_value_of_f_l693_693490

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2) + 2 * x

theorem minimum_value_of_f (h : ∀ x > 0, f x ≥ 3) : ∃ x, x > 0 ∧ f x = 3 :=
by
  sorry

end minimum_value_of_f_l693_693490


namespace largest_prime_factor_of_7fact_8fact_l693_693906

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693906


namespace equivalence_p_q_l693_693022

def proposition_p (x : ℝ) : Prop := sin (2 * x) = 1
def proposition_q (x : ℝ) : Prop := tan x = 1

theorem equivalence_p_q (x : ℝ) : proposition_p x ↔ proposition_q x := 
sorry

end equivalence_p_q_l693_693022


namespace cos_double_angle_l693_693307

theorem cos_double_angle (α : ℝ) (h_origin : α.vertex = (0,0)) 
  (h_initial_side : α.initial_side.coincides_with_non_negative_x_axis) 
  (h_point_A : α.terminal_side.contains_point (2, 3)) : cos(2 * α) = -5 / 13 :=
sorry

end cos_double_angle_l693_693307


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693890

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693890


namespace shaded_region_area_is_10_l693_693765

noncomputable def area_shaded_region : ℝ :=
let a := 4 -- side length of the small square
let b := 12 -- side length of the large square
let side_ratio := 3 / 4 -- from similarity of triangles DGF and AHF
let DG := side_ratio * a in
let area_triangle_DGF := 0.5 * DG * a in
let area_small_square := a * a in
area_small_square - area_triangle_DGF

theorem shaded_region_area_is_10 : area_shaded_region = 10 := by
  -- The actual proof would go here
  sorry

end shaded_region_area_is_10_l693_693765


namespace find_product_and_least_k_l693_693389

noncomputable def product_xk (x : List ℚ) (k : ℚ) : ℚ :=
  x.foldl (λ acc xi => acc * xi) 1

theorem find_product_and_least_k (k n : ℕ) (x : List ℚ)
  (hk : k > 1) (hn : n > 2018) (hn_odd : Odd n)
  (hx_nonzero : ∀ xi ∈ x, xi ≠ 0) (hx_length : x.length = n)
  (hx_condition : ∀ i ∈ Finset.range n, (x.nth i).getD 0 + (k / (x.nth (i + 1) % n).getD 0) = 
                               (x.nth ((i + 1) % n)).getD 0 + (k / (x.nth ((i + 2) % n)).getD 0))
  : product_xk x k = k ^ (n / 2) ∧ 4 ≤ k := sorry

end find_product_and_least_k_l693_693389


namespace largest_prime_factor_7fac_8fac_l693_693941

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693941


namespace optionA_optionC_l693_693279

-- Definitions
variables {a b c : ℝ}
variables (e : ℝ)
variables (B F O : ℝ)
variables (x y : ℝ)

-- Given Conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1)

def semi_focal_distance (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2

def angle_condition (B F O : ℝ) : Prop :=
  B > F

def eccentricity (a c e : ℝ) : Prop :=
  e = c / a

-- Statements to Prove
theorem optionA (h1 : ellipse a b x y) (h2 : semi_focal_distance a b c) (h3 : 2 * b > a + c) :
  b^2 > a * c :=
sorry

theorem optionC (h1 : ellipse a b x y) (h2 : semi_focal_distance a b c) (h3 : angle_condition B F O) :
  0 < e ∧ e < real.sqrt 2 / 2 :=
sorry

end optionA_optionC_l693_693279


namespace digits_after_decimal_l693_693203

theorem digits_after_decimal (n : ℕ) (h : 10^5 * 125 = 2^5 * 5^8) : 
  (↑(5:ℕ)^7 / (10:ℕ)^5 * 125).digits_after_decimal = 5 :=
by sorry

end digits_after_decimal_l693_693203


namespace ceil_neg_3_7_l693_693223

-- Define the ceiling function in Lean
def ceil (x : ℝ) : ℤ := int.ceil x

-- A predicate to represent the statement we want to prove
theorem ceil_neg_3_7 : ceil (-3.7) = -3 := by
  -- Provided conditions
  have h1 : ceil (-3.7) = int.ceil (-3.7) := rfl
  have h2 : int.ceil (-3.7) = -3 := by
    -- Lean's int.ceil function returns the smallest integer greater or equal to the input
    sorry  -- proof goes here

  -- The main statement
  exact h2

end ceil_neg_3_7_l693_693223


namespace students_with_green_eyes_l693_693810

theorem students_with_green_eyes 
  (total_students : ℕ) 
  (brown_to_green_ratio : ℕ → ℕ) 
  (both_brown_and_green : ℕ) 
  (neither : ℕ)
  (total_students_eq : total_students = 40)
  (brown_to_green_ratio_eq : brown_to_green_ratio = (λ x, 3 * x))
  (both_brown_and_green_eq : both_brown_and_green = 9)
  (neither_eq : neither = 4): 
  ∃ (green_eyes : ℕ), green_eyes = 9 := 
by
  -- Solution proof goes here
  sorry

end students_with_green_eyes_l693_693810


namespace principal_amount_l693_693534

variable (P : ℝ)

/-- Prove the principal amount P given that the simple interest at 4% for 5 years is Rs. 2400 less than the principal --/
theorem principal_amount : 
  (4/100) * P * 5 = P - 2400 → 
  P = 3000 := 
by 
  sorry

end principal_amount_l693_693534


namespace largest_prime_factor_of_fact_sum_is_7_l693_693973

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693973


namespace largest_prime_factor_7fac_plus_8fac_l693_693956

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693956


namespace random_divisor_multiple_of_10_pow_88_l693_693335

theorem random_divisor_multiple_of_10_pow_88 
  (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 99) (h2 : 0 ≤ b ∧ b ≤ 99)
  (h3 : 10^99 = 2^99 * 5^99) (h4 : 10^88 = 2^88 * 5^88)
  (h5 : ∀ a b : ℕ, 88 ≤ a ∧ a ≤ 99 → 88 ≤ b ∧ b ≤ 99 → 2^a * 5^b = 10^88 * k) : 
  ∃ m n : ℕ, m + n = 634 ∧ nat.coprime m n ∧ (m : ℚ) / n = 9 / 625 := 
by
  sorry

end random_divisor_multiple_of_10_pow_88_l693_693335


namespace sum_horse_distances_l693_693857

noncomputable def daily_distance (n : ℕ) : ℝ :=
  if n = 16 then 315
  else if n < 16 then
    315 / (1.05 ^ (16 - n))
  else 315 * (1.05 ^ (n - 16))

theorem sum_horse_distances :
  let sum_distances := (Finset.range 17).sum (λ n, daily_distance n)
  1.05^17 = 2.292 →
  abs (sum_distances - 7752) < 1 :=
by
  sorry

end sum_horse_distances_l693_693857


namespace function_values_at_mean_l693_693716

noncomputable def f (x : ℝ) : ℝ := x^2 - 10 * x + 16

theorem function_values_at_mean (x₁ x₂ : ℝ) (h₁ : x₁ = 8) (h₂ : x₂ = 2) :
  let x' := (x₁ + x₂) / 2
  let x'' := Real.sqrt (x₁ * x₂)
  f x' = -9 ∧ f x'' = -8 := by
  let x' := (x₁ + x₂) / 2
  let x'' := Real.sqrt (x₁ * x₂)
  have hx' : x' = 5 := sorry
  have hx'' : x'' = 4 := sorry
  have hf_x' : f x' = -9 := sorry
  have hf_x'' : f x'' = -8 := sorry
  exact ⟨hf_x', hf_x''⟩

end function_values_at_mean_l693_693716


namespace reasonable_distribution_of_game_cards_l693_693827

-- Define the conditions
def total_game_cards : ℕ := 12
def a_points : ℕ := 2
def b_points : ℕ := 1
def remaining_rounds : ℕ := 2

-- Define the probabilities
def probability_of_a_winning : ℚ := 3 / 4
def probability_of_b_winning : ℚ := 1 / 4

-- Define the distribution of cards based on winning probabilities
def distribution_of_game_cards (prob_a_win : ℚ) (prob_b_win : ℚ) (total: ℕ) : (ℕ × ℕ) := 
  let cards_for_a := (prob_a_win * total).natAbs
  let cards_for_b := (prob_b_win * total).natAbs
  (cards_for_a, cards_for_b)

-- Theorem statement
theorem reasonable_distribution_of_game_cards : 
  distribution_of_game_cards probability_of_a_winning probability_of_b_winning total_game_cards = (9, 3) :=
  by
    sorry

end reasonable_distribution_of_game_cards_l693_693827


namespace smallest_n_sum_gt_10_pow_5_l693_693070

theorem smallest_n_sum_gt_10_pow_5 :
  ∃ (n : ℕ), (n ≥ 142) ∧ (5 * n^2 + 4 * n ≥ 100000) :=
by
  use 142
  sorry

end smallest_n_sum_gt_10_pow_5_l693_693070


namespace seventh_team_cups_l693_693431

noncomputable theory

def total_cups := 2500
def team1_cups := 450
def team2_cups := 300
def team5_6_cups := team1_cups + team2_cups
def t34 (t7 : ℕ) := 2 * t7
def t7_cups : ℕ := 334  -- Given as the result derived in the solution

theorem seventh_team_cups : 
  team1_cups + team2_cups + t34 t7_cups + team5_6_cups + t7_cups = total_cups :=
by {
  -- This will be the place for the proof that computes and verifies the result
  -- Using sorry here to assume the proof for now.
  sorry
}

end seventh_team_cups_l693_693431


namespace parabolas_vertex_condition_l693_693510

theorem parabolas_vertex_condition (p q x₁ x₂ y₁ y₂ : ℝ) (h1: y₂ = p * (x₂ - x₁)^2 + y₁) (h2: y₁ = q * (x₁ - x₂)^2 + y₂) (h3: x₁ ≠ x₂) : p + q = 0 :=
sorry

end parabolas_vertex_condition_l693_693510


namespace jason_egg_consumption_l693_693234

-- Definition for the number of eggs Jason consumes per day
def eggs_per_day : ℕ := 3

-- Definition for the number of days in a week
def days_in_week : ℕ := 7

-- Definition for the number of weeks we are considering
def weeks : ℕ := 2

-- The statement we want to prove, which combines all the conditions and provides the final answer
theorem jason_egg_consumption : weeks * days_in_week * eggs_per_day = 42 := by
sorry

end jason_egg_consumption_l693_693234


namespace jason_two_weeks_eggs_l693_693232

-- Definitions of given conditions
def eggs_per_omelet := 3
def days_per_week := 7
def weeks := 2

-- Statement to prove
theorem jason_two_weeks_eggs : (eggs_per_omelet * (days_per_week * weeks)) = 42 := by
  sorry

end jason_two_weeks_eggs_l693_693232


namespace intersection_m_n_l693_693317

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := { -1, 0, 1, 2, 3 }

theorem intersection_m_n : M ∩ N = {0, 1, 2} := 
sorry

end intersection_m_n_l693_693317


namespace f_sum_neg_l693_693690

def f : ℝ → ℝ := sorry

theorem f_sum_neg (x₁ x₂ : ℝ)
  (h1 : ∀ x, f (4 - x) = - f x)
  (h2 : ∀ x, x < 2 → ∀ y, y < x → f y < f x)
  (h3 : x₁ + x₂ > 4)
  (h4 : (x₁ - 2) * (x₂ - 2) < 0)
  : f x₁ + f x₂ < 0 := 
sorry

end f_sum_neg_l693_693690


namespace lines_intersect_l693_693559

def line1 (s : ℚ) : ℚ × ℚ :=
  (1 + 2 * s, 4 - 3 * s)

def line2 (v : ℚ) : ℚ × ℚ :=
  (3 + 4 * v, 9 - v)

theorem lines_intersect :
  ∃ s v : ℚ, (line1 s) = (line2 v) ∧ (line1 s) = (-17/5, 53/5) := 
sorry

end lines_intersect_l693_693559


namespace possible_values_of_n_l693_693655

theorem possible_values_of_n (n : ℕ) (h1 : ∃ d₆ d₇ : ℕ, d₆ < d₇ ∧ d₆ = nth_divisor n 5 ∧ d₇ = nth_divisor n 6 ∧ n + 1 = d₆^2 + d₇^2) (h2 : nat_num_of_divisors(n) ≥ 7) : n = 144 ∨ n = 1984 :=
sorry

def nth_divisor (n : ℕ) (k : ℕ) : ℕ :=
  if k < number_of_divisors n then (sorted_divisors n).nth k else 0

def number_of_divisors (n : ℕ) : ℕ :=
  (list_iota (n+1)).count (λ d, n % d = 0)

def sorted_divisors (n : ℕ) : list ℕ :=
  (list_iota (n+1)).filter (λ d, n % d = 0)

def list_iota (n : ℕ) : list ℕ :=
  list.range n

end possible_values_of_n_l693_693655


namespace projection_of_vector_a_on_b_l693_693305

theorem projection_of_vector_a_on_b (a b : ℝ) (magnitude_a : ℝ) (magnitude_b : ℝ) (angle_ab : ℝ)
  (h1 : magnitude_a = 5) (h2 : magnitude_b = 3) (h3 : angle_ab = real.pi / 3) :
  (magnitude_a * real.cos angle_ab) / magnitude_b = 5 / 2 :=
by
  sorry

end projection_of_vector_a_on_b_l693_693305


namespace g_even_l693_693213

def g (x : ℝ) : ℝ := log (x^2)

theorem g_even : ∀ x : ℝ, g x = g (-x) :=
by sorry

end g_even_l693_693213


namespace sum_of_x_is_nine_l693_693801

noncomputable def sum_of_x (n : ℕ) : ℂ := 
∑ i in finset.range n, (λ i : ℕ, x_i)

theorem sum_of_x_is_nine
  (x y z : ℂ)
  (x_i : ℕ → ℂ)
  (y_i : ℕ → ℂ)
  (z_i : ℕ → ℂ)
  (h1 : ∀ i, x_i i + y_i i * z_i i = 9)
  (h2 : ∀ i, y_i i + x_i i * z_i i = 13)
  (h3 : ∀ i, z_i i + x_i i * y_i i = 12)
  (h4 : n = 2 ∨ n = 3) :
  sum_of_x n = 9 := sorry

end sum_of_x_is_nine_l693_693801


namespace find_d_plus_f_l693_693635

theorem find_d_plus_f (a b c d e f : ℂ) (h1 : b = 4) (h2 : e = -2 * a - c) (h3 : 2 * a + 4 * complex.I + c + 3 * d * complex.I + e + f * complex.I = 6 * complex.I) :
  d + f = 2 :=
sorry

end find_d_plus_f_l693_693635


namespace people_in_hall_l693_693033

noncomputable def number_of_people_in_hall (total_chairs : ℕ) (empty_chairs : ℕ) : ℕ :=
  let total_people := (total_chairs - empty_chairs) * 2
  in total_people

theorem people_in_hall (total_chairs : ℕ) (empty_chairs : ℕ) (half_people_seated : ℕ) (five_eighths_chairs_occupied : ℕ) :
  half_people_seated = 5 * total_chairs / 8 →
  empty_chairs = 8 →
  total_chairs = 64 →
  number_of_people_in_hall total_chairs empty_chairs = 80 :=
by
  intros h1 h2 h3
  rw [h2, h3]
  unfold number_of_people_in_hall
  simp
  sorry

end people_in_hall_l693_693033


namespace range_of_a_l693_693345

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| > a) → a < 3 :=
by
  sorry

end range_of_a_l693_693345


namespace intersection_eq_l693_693024

-- We need the definitions of the sets.
def M : Set ℝ := {x | 2^(x + 1) > 1}
def N : Set ℝ := {x | Real.log x ≤ 1}

-- The theorem to be proven.
theorem intersection_eq : M ∩ N = {x | 0 < x ∧ x ≤ Real.exp 1} :=
by
  sorry

end intersection_eq_l693_693024


namespace scheduling_methods_l693_693168
   
   theorem scheduling_methods :
     (∃ (morning_afternoon: Fin 6 → Prop),
       (∀ (i: Fin 3), morning_afternoon ⟨i, sorry⟩ = true) ∧
       (∀ (j: Fin 3), morning_afternoon ⟨j + 3, sorry⟩ = false) ∧ 
       (M ∈ morning_afternoon) ∧ (A ∉ morning_afternoon)) →
       (∃f: Fin 6 → String, 
       (∀ (i: Fin 3), f i = "Mathematics" ↔ i ∈ {0,1,2}) ∧
       (∀ (i: Fin 3), f (i + 3) = "Art" ↔ (i + 3) ∈ {3,4,5}) ∧ 
       ∃! σ: Perm (Fin 6), 
       set_repr (σ.to_fun ∘ f) ⊆ { "Chinese", "Mathematics", "English", "Physics", "Physical Education", "Art" }) →
     (9 * 24 = 216) :=
   by sorry
   
end scheduling_methods_l693_693168


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693988

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693988


namespace sum_of_areas_of_confetti_l693_693069

theorem sum_of_areas_of_confetti :
  let red_side := 11
  let blue_side := 5
  let red_area := red_side * red_side
  let blue_area := blue_side * blue_side
  red_area + blue_area = 146 :=
by
  let red_side := 11
  let blue_side := 5
  let red_area := red_side * red_side
  let blue_area := blue_side * blue_side
  have h1 : red_area = 121 := rfl
  have h2 : blue_area = 25 := rfl
  have h3 : red_area + blue_area = 146 := by simp only [h1, h2]; exact rfl
  exact h3

end sum_of_areas_of_confetti_l693_693069


namespace vector_parallel_l693_693739

open Function

variables {x y : ℝ}

def vector_a (x : ℝ) : ℝ × ℝ × ℝ := (2 * x, 1, 3)
def vector_b (y : ℝ) : ℝ × ℝ × ℝ := (1, -2 * y, 9)

def are_parallel {α β : Type*} [LinearOrder α] [LinearOrder β] 
  (v₁ : α × α × α) (v₂ : β × β × β) : Prop :=
v₁.1 / v₂.1 = v₁.2 / v₂.2 ∧ v₁.2 / v₂.2 = v₁.3 / v₂.3

theorem vector_parallel (h : are_parallel (vector_a x) (vector_b y)) : x * y = -1 / 4 := 
sorry

end vector_parallel_l693_693739


namespace Sue_final_answer_l693_693582

theorem Sue_final_answer (x : ℕ) (h : x = 8) : 
  let y := (x + 3) * 2 - 2 in
  (y + 1) * 2 + 4 = 46 :=
by
  sorry

end Sue_final_answer_l693_693582


namespace max_sqrt_sum_l693_693399

theorem max_sqrt_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 2013) :
  sqrt (3 * a + 12) + sqrt (3 * b + 12) + sqrt (3 * c + 12) ≤ 135 :=
sorry

end max_sqrt_sum_l693_693399


namespace sqrt_approx_half_cbrt_approx_x_l693_693412

-- Definition of given approximations
def sqrt_approx : ℝ := 7.071
def cbrt_approx1 : ℝ := 1.8308
def cbrt_approx2 : ℝ := 18.308
def target_cbrt : ℝ := -0.18308

-- 1. Prove that sqrt(50) ≈ 7.071 implies sqrt(0.5) ≈ 0.7071
theorem sqrt_approx_half : sqrt_approx ≈ 7.071 → sqrt 0.5 ≈ 0.7071 := by
  sorry

-- 2. Prove that the given conditions on cube roots implies x ≈ -0.006137
theorem cbrt_approx_x : (cbrt_approx1 ≈ 1.8308 ∧ cbrt_approx2 ≈ 18.308 ∧ (3√x) ≈ target_cbrt) →
  x ≈ -0.006137 := by
  sorry

end sqrt_approx_half_cbrt_approx_x_l693_693412


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693891

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693891


namespace range_f_on_interval_shortest_distance_between_intersections_l693_693310

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * sin x * cos x - 2 * sqrt 3 * cos x ^ 2 + sqrt 3

-- Define the interval [0, π / 2]
def interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ π / 2

-- Theorem stating the range of f(x) when x is in the interval
theorem range_f_on_interval : ∀ (x : ℝ), interval x → -sqrt 3 ≤ f x ∧ f x ≤ 2 :=
by
  intro x hx
  -- Proof omitted
  sorry

-- Theorem for the shortest distance between intersections of y = f(x) and y = 1
theorem shortest_distance_between_intersections : 
  ∃ d : ℝ, d = 2 * π / 3 ∧ (∃ x1 x2 : ℝ, interval x1 ∧ interval x2 ∧ f x1 = 1 ∧ f x2 = 1 ∧ x2 - x1 = d) :=
by
  -- Proof omitted
  sorry

end range_f_on_interval_shortest_distance_between_intersections_l693_693310


namespace correct_eccentricity_l693_693299

noncomputable def hyperbola_eccentricity (a b : ℝ) (P F1 F2 O : ℝ × ℝ)
  (ha : 0 < a) (hb : 0 < b)
  (hP : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (hF1 : F1.1 = -sqrt(a^2 + b^2) ∧ F1.2 = 0)
  (hF2 : F2.1 = sqrt(a^2 + b^2) ∧ F2.2 = 0)
  (hOrigin : O = (0, 0))
  (dot_product_zero : ((P.1, P.2) + (sqrt(a^2 + b^2), 0)) • (F2.1 - P.1, F2.2 - P.2) = 0)
  (dist_condition : dist (P.1, P.2) (F1.1, F1.2) = sqrt(3) * dist (P.1, P.2) (F2.1, F2.2)) :
  ℝ :=
  sqrt(3) + 1

theorem correct_eccentricity (a b : ℝ) (P F1 F2 O : ℝ)
  (ha : 0 < a) (hb : 0 < b)
  (hP : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (hF1 : F1.1 = -sqrt(a^2 + b^2) ∧ F1.2 = 0)
  (hF2 : F2.1 = sqrt(a^2 + b^2) ∧ F2.2 = 0)
  (hOrigin : O = (0, 0))
  (dot_product_zero : ((P.1, P.2) + (sqrt(a^2 + b^2), 0)) • (F2.1 - P.1, F2.2 - P.2) = 0)
  (dist_condition : dist (P.1, P.2) (F1.1, F1.2) = sqrt(3) * dist (P.1, P.2) (F2.1, F2.2)) :
  hyperbola_eccentricity a b P F1 F2 O ha hb hP hF1 hF2 hOrigin dot_product_zero dist_condition = sqrt(3) + 1 := 
sorry

end correct_eccentricity_l693_693299


namespace cats_average_weight_l693_693377

/-- Prove that the overall average weight of Janele's cats is 13.046 pounds -/
theorem cats_average_weight :
  let weights := [12.0, 12.0, 14.7, 9.3, 13.2, 15.8, (14 + 15.4 + 13.7 + 14.2) / 4]
  (weights.sum / weights.length = 13.046) :=
by
  let weights := [12.0, 12.0, 14.7, 9.3, 13.2, 15.8, (14 + 15.4 + 13.7 + 14.2) / 4]
  have h_avg_weight : (weights.sum / weights.length) = 13.046 := sorry
  exact h_avg_weight

end cats_average_weight_l693_693377


namespace frequencies_of_first_class_quality_difference_confidence_l693_693111

section quality_comparison

variables (n a b c d : ℕ)

-- Given conditions
def total_products : ℕ := 400
def machine_a_total : ℕ := 200
def machine_a_first : ℕ := 150
def machine_a_second : ℕ := 50
def machine_b_total : ℕ := 200
def machine_b_first : ℕ := 120
def machine_b_second : ℕ := 80

-- Defining the K^2 calculation formula
def K_squared : ℚ :=
  (total_products * (machine_a_first * machine_b_second - machine_a_second * machine_b_first) ^ 2 : ℚ) /
  ((machine_a_first + machine_a_second) * (machine_b_first + machine_b_second) * (machine_a_first + machine_b_first) * (machine_a_second + machine_b_second))

-- Proof statement for Q1: Frequencies of first-class products
theorem frequencies_of_first_class :
  machine_a_first / machine_a_total = 3 / 4 ∧ 
  machine_b_first / machine_b_total = 3 / 5 := 
sorry

-- Proof statement for Q2: Confidence level of difference in quality
theorem quality_difference_confidence :
  K_squared = 10.256 ∧ 10.256 > 6.635 → 0.99 :=
sorry

end quality_comparison

end frequencies_of_first_class_quality_difference_confidence_l693_693111


namespace inequality_of_power_sums_l693_693263

variable (a b c : ℝ)

theorem inequality_of_power_sums (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a < b + c) (h5 : b < c + a) (h6 : c < a + b) :
  a^4 + b^4 + c^4 < 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) := sorry

end inequality_of_power_sums_l693_693263


namespace reflection_line_coordinates_sum_l693_693486

theorem reflection_line_coordinates_sum (m b : ℝ)
  (h : ∀ (x y x' y' : ℝ), (x, y) = (-4, 2) → (x', y') = (2, 6) → 
  ∃ (m b : ℝ), y = m * x + b ∧ y' = m * x' + b ∧ ∀ (p q : ℝ), 
  (p, q) = ((x+x')/2, (y+y')/2) → p = ((-4 + 2)/2) ∧ q = ((2 + 6)/2)) :
  m + b = 1 :=
by
  sorry

end reflection_line_coordinates_sum_l693_693486


namespace find_y_l693_693652

-- Define the conditions
def OA : ℝ := 5
def OB : ℝ := 6
def OC : ℝ := 12
def OD : ℝ := 5
def BD : ℝ := 9
def angle_AOC_eq_angle_BOD : Prop := ∠ A O C = ∠ B O D

-- The proof problem statement
theorem find_y (h_cos_phi : cos (∠ B O D) = -7/30) : ∃ y : ℝ, y = sqrt 197 :=
by
  sorry

end find_y_l693_693652


namespace compute_expression_l693_693631

theorem compute_expression :
  8.1^0 - (1 / 2)^(-2) + Real.log 25 / Real.log 10 + 2 * (Real.log 2 / Real.log 10) = -1 := by
  sorry

end compute_expression_l693_693631


namespace basketball_game_first_half_points_l693_693557

theorem basketball_game_first_half_points (a b r d : ℕ) (H1 : a = b)
  (H2 : a * (1 + r + r^2 + r^3) = 4 * a + 6 * d + 1) 
  (H3 : 15 * a ≤ 100) (H4 : b + (b + d) + b + 2 * d + b + 3 * d < 100) : 
  (a + a * r + b + b + d) = 34 :=
by sorry

end basketball_game_first_half_points_l693_693557


namespace largest_prime_factor_l693_693986

def factorial : ℕ → ℕ 
  | 0     := 1
  | (n+1) := (n+1) * factorial n

def fact7 : ℕ := factorial 7
def fact8 : ℕ := factorial 8

theorem largest_prime_factor (n : ℕ) : 
  let sum_fact := fact7 + fact8 in
  ∃ p, nat.prime p ∧ p ∣ sum_fact ∧ ∀ q, nat.prime q → q ∣ sum_fact → q ≤ p := by
  -- Definitions of 7! and 8! and their properties.
  sorry

end largest_prime_factor_l693_693986


namespace bobby_last_10_throws_successful_l693_693200

theorem bobby_last_10_throws_successful :
    let initial_successful := 18 -- Bobby makes 18 successful throws out of his initial 30 throws.
    let total_throws := 30 + 10 -- Bobby makes a total of 40 throws.
    let final_successful := 0.64 * total_throws -- Bobby needs to make 64% of 40 throws to achieve a 64% success rate.
    let required_successful := 26 -- Adjusted to the nearest whole number.
    -- Bobby makes 8 successful throws in his last 10 attempts.
    required_successful - initial_successful = 8 := by
  sorry

end bobby_last_10_throws_successful_l693_693200


namespace point_slope_form_of_line_l693_693664

noncomputable def line_passing_through_point_slope_twice (P : ℝ × ℝ) (k : ℝ) : (ℝ → ℝ → Prop) :=
  ∃ b : ℝ, b = P.2 - k * P.1 ∧ ∀ x y : ℝ, y = k * x + b

theorem point_slope_form_of_line (P : ℝ × ℝ) (k : ℝ) (twice_slope : ∀ θ, tan θ = k → tan (2 * θ) = -sqrt(3))
    (line_eq : ∀ y x, y = sqrt(3) * x + sqrt(3)) :
  P = (3, 4) →
  line_passing_through_point_slope_twice P (-sqrt(3)) (λ x y, y - 4 = -sqrt(3) * (x - 3)) :=
by
  sorry

end point_slope_form_of_line_l693_693664


namespace baker_cakes_l693_693199

theorem baker_cakes : (62.5 + 149.25 - 144.75 = 67) :=
by
  sorry

end baker_cakes_l693_693199


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693912

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693912


namespace ellipse_equation_find_k_l693_693404

-- Lean statement for Problem I
theorem ellipse_equation
  (a b c : ℝ) 
  (a_gt_b : a > b) 
  (b_gt_zero : b > 0) 
  (eccentricity_condition : c / a = sqrt 3 / 3) 
  (perpendicular_segment : 2 * sqrt 6 * b / 3 = 4 * sqrt 3 / 3) :
  a = sqrt 3 ∧ b = sqrt 2 ∧ c = 1 ∧ ∀ x y, x^2 / 3 + y^2 / 2 = 1 :=
sorry

-- Lean statement for Problem II
theorem find_k
  (a b : ℝ) 
  (ellipse_equation : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) 
  (focus_point : F(-1, 0))
  (A : (-sqrt 3, 0)) 
  (B : (sqrt 3, 0)) 
  (dot_product_condition : 
    ∀ x1 y1 x2 y2 k, 
      (x1 + sqrt 3, y1) • (sqrt 3 - x2, -y2) + (x2 + sqrt 3, y2) • (sqrt 3 - x1, -y1) = 7) :
  k = sqrt 10 ∨ k = -sqrt 10 :=
sorry

end ellipse_equation_find_k_l693_693404


namespace a_range_condition_l693_693741

theorem a_range_condition (a : ℝ) : (∀ x ∈ set.Icc (0 : ℝ) (2 : ℝ), x^2 - 2 * a * x + a + 2 ≥ 0) ↔ a ∈ set.Icc (-2) (2) :=
sorry

end a_range_condition_l693_693741


namespace frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693093

theorem frequency_machine_A (total_A first_class_A : ℕ) (h_total_A: total_A = 200) (h_first_class_A: first_class_A = 150) :
  first_class_A / total_A = 3 / 4 := by
  rw [h_total_A, h_first_class_A]
  norm_num

theorem frequency_machine_B (total_B first_class_B : ℕ) (h_total_B: total_B = 200) (h_first_class_B: first_class_B = 120) :
  first_class_B / total_B = 3 / 5 := by
  rw [h_total_B, h_first_class_B]
  norm_num

theorem chi_square_test_significance (n a b c d : ℕ) (h_n: n = 400) (h_a: a = 150) (h_b: b = 50) 
  (h_c: c = 120) (h_d: d = 80) :
  let K_squared := (n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d)))
  in K_squared > 6.635 := by
  rw [h_n, h_a, h_b, h_c, h_d]
  let num := 400 * (150 * 80 - 50 * 120)^2
  let denom := (150 + 50) * (120 + 80) * (150 + 120) * (50 + 80)
  have : K_squared = num / denom := rfl
  norm_num at this
  sorry

end frequency_machine_A_frequency_machine_B_chi_square_test_significance_l693_693093


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693888

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693888


namespace no_determinable_cost_of_2_pans_l693_693382

def pots_and_pans_problem : Prop :=
  ∀ (P Q : ℕ), 3 * P + 4 * Q = 100 → ¬∃ Q_cost : ℕ, Q_cost = 2 * Q

theorem no_determinable_cost_of_2_pans : pots_and_pans_problem :=
by
  sorry

end no_determinable_cost_of_2_pans_l693_693382


namespace bert_made_1_dollar_l693_693603

def bert_earnings (selling_price tax_rate markup : ℝ) : ℝ :=
  selling_price - (tax_rate * selling_price) - (selling_price - markup)

theorem bert_made_1_dollar :
  bert_earnings 90 0.1 10 = 1 :=
by 
  sorry

end bert_made_1_dollar_l693_693603


namespace ratio_of_brownies_l693_693383

def total_brownies : ℕ := 15
def eaten_on_monday : ℕ := 5
def eaten_on_tuesday : ℕ := total_brownies - eaten_on_monday

theorem ratio_of_brownies : eaten_on_tuesday / eaten_on_monday = 2 := 
by
  sorry

end ratio_of_brownies_l693_693383


namespace gcd_permutation_pairs_l693_693423

theorem gcd_permutation_pairs (N : ℕ) (hN : N > 1000) (π : Fin N → Fin N) :
  (∃ S, S ⊆ {p : Fin N × Fin N | p.1 ≠ p.2 ∧ Nat.gcd p.1.val p.2.val = 1 ∧ Nat.gcd (π p.1).val (π p.2).val = 1} ∧
  S.card ≥ (11 * N.choose 2 / 100).toNat) := 
sorry

end gcd_permutation_pairs_l693_693423


namespace points_on_opposite_sides_l693_693306

theorem points_on_opposite_sides (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by sorry

end points_on_opposite_sides_l693_693306


namespace sum_of_solutions_eq_65_l693_693796

noncomputable def f (x : ℝ) : ℝ := 12 * x + 5

theorem sum_of_solutions_eq_65 :
  let S := {x : ℝ | f⁻¹'({x}) = f ((3 * x)⁻¹)} in
  (∑ x in S, x) = 65 :=
by
  sorry

end sum_of_solutions_eq_65_l693_693796


namespace pure_imaginary_root_magnitude_l693_693687

theorem pure_imaginary_root_magnitude :
  ∀ (m : ℝ), (∃ (z : ℂ), (z.im ≠ 0) ∧ z^2 - (2*m - 1 : ℂ)*z + (m^2 + 1 : ℂ) = 0) →
  ∃ (z : ℂ), (|z + m| = √6 / 2) := by
  intro m h
  sorry

end pure_imaginary_root_magnitude_l693_693687


namespace sum_of_coefficients_is_one_l693_693204

/--
Given the polynomial \(p(x) = 3(x^8 - 2x^5 + 4x^3 - 6) - 5(2x^4 + 3x - 7) + 6(x^6 - x^2 + 1)\),
prove that the sum of the coefficients of \(p(x)\) is equal to 1.
-/
noncomputable def p : Polynomial ℤ :=
  3 * (Polynomial.monomial 8 1 - 2 * Polynomial.monomial 5 1 + 4 * Polynomial.monomial 3 1 - Polynomial.C 6)
  - 5 * (2 * Polynomial.monomial 4 1 + 3 * Polynomial.monomial 1 1 - Polynomial.C 7)
  + 6 * (Polynomial.monomial 6 1 - Polynomial.monomial 2 1 + Polynomial.C 1)

theorem sum_of_coefficients_is_one : Polynomial.sum_of_coefficients p = 1 :=
  sorry

end sum_of_coefficients_is_one_l693_693204


namespace correct_statements_count_l693_693405

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

noncomputable def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f (x)

noncomputable def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f(x) > f(y)

noncomputable def has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ (f_inv : ℝ → ℝ), ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

noncomputable def has_zero_point (g : ℝ → ℝ) : Prop :=
  ∃ x, g (x) = 0

theorem correct_statements_count (f : ℝ → ℝ) :
  (is_odd f → is_odd (f ∘ f)) ∧
  (∃ T, is_periodic f T → is_periodic (f ∘ f) T) ∧
  (is_monotonically_decreasing f → ¬is_monotonically_decreasing (f ∘ f)) ∧
  (has_inverse f ∧ has_zero_point (λ x, f x - f⁻¹ x) → has_zero_point (λ x, f x - x)) →
  (3 : ℕ) := sorry

end correct_statements_count_l693_693405


namespace mrs_smith_class_boys_girls_ratio_l693_693352

theorem mrs_smith_class_boys_girls_ratio (total_students boys girls : ℕ) (h1 : boys / girls = 3 / 4) (h2 : boys + girls = 42) : girls = boys + 6 :=
by
  sorry

end mrs_smith_class_boys_girls_ratio_l693_693352


namespace base_eight_to_base_ten_l693_693879

theorem base_eight_to_base_ten (n : ℕ) (h : n = 4 * 8^1 + 7 * 8^0) : n = 39 := by
  sorry

end base_eight_to_base_ten_l693_693879


namespace length_LM_eq_5_l693_693592

variables (P Q R L M : Type) 

noncomputable def area_triang_PQR : ℝ := 200
noncomputable def area_trapezoid : ℝ := 150
noncomputable def height_PQR : ℝ := 40

theorem length_LM_eq_5 :
  let area_triang_PQR := (200 : ℝ),
      area_trapezoid := (150 : ℝ),
      height_PQR := (40 : ℝ),
      base_PQR := 2 * area_triang_PQR / height_PQR,
      area_triang_PLM := area_triang_PQR - area_trapezoid,
      ratio_areas := area_triang_PLM / area_triang_PQR,
      ratio_lengths := ratio_areas.sqrt
  in
  ratio_lengths * base_PQR = 5 :=
by
  -- Proof to be provided
  sorry

end length_LM_eq_5_l693_693592


namespace multiply_63_57_l693_693621

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l693_693621


namespace largest_prime_factor_7fac_plus_8fac_l693_693950

theorem largest_prime_factor_7fac_plus_8fac : 
  let fact7 := 7!
  let fact8 := 8!
  let num := fact7 + fact8
  prime_factor_larger_than num = 7 :=
by sorry

end largest_prime_factor_7fac_plus_8fac_l693_693950


namespace intersection_complement_l693_693320

universe u
variable {α : Type u}

-- Define the sets I, M, N, and their complement with respect to I
def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}
def complement_I (s : Set ℕ) : Set ℕ := { x ∈ I | x ∉ s }

-- Statement of the theorem
theorem intersection_complement :
  M ∩ (complement_I N) = {1} :=
by
  sorry

end intersection_complement_l693_693320


namespace coefficient_x2_in_expansion_eq_80_l693_693329

theorem coefficient_x2_in_expansion_eq_80 (a : ℝ) : (binom 5 3 * a^3 * x^2 = 80) → a = 2 :=
by 
  sorry

end coefficient_x2_in_expansion_eq_80_l693_693329


namespace function_is_constant_l693_693816

noncomputable def f (x α : ℝ) : ℝ :=
  (Real.cos x)^2 + (Real.cos (x + α))^2 - 2 * (Real.cos α) * (Real.cos x) * (Real.cos (x + α))

theorem function_is_constant (α : ℝ) : ∃ c : ℝ, ∀ x : ℝ, f x α = c :=
  exists.intro ((1 - (Real.cos (2 * α))) / 2) (
    by sorry
  )

end function_is_constant_l693_693816


namespace largest_prime_factor_of_7_fact_plus_8_fact_l693_693933

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else nat.find_greatest (λ p, is_prime p ∧ p ∣ n) n

theorem largest_prime_factor_of_7_fact_plus_8_fact : largest_prime_factor (factorial 7 + factorial 8) = 7 :=
by
  sorry

end largest_prime_factor_of_7_fact_plus_8_fact_l693_693933


namespace sum_reciprocal_bound_l693_693493

-- Define the sequence recursively
def x : ℕ → ℝ
| 0     := 1 -- note that Lean uses 0-based indexing, so this corresponds to x_1 in the problem
| (n+1) := x n + real.sqrt (x n)

-- Define the main theorem
theorem sum_reciprocal_bound : (∑ n in finset.range 2018, 1 / x n.succ) < 3 :=
by 
    -- This step calls on your intuition of how to transform the original problem into a mathematically equivalent one in Lean
    -- Begin the theorem here using these details, but specify the key elements like the summation bound
    sorry -- Proof will be provided here

end sum_reciprocal_bound_l693_693493


namespace pentagon_area_ratio_l693_693784

theorem pentagon_area_ratio (ABCDE PQRST : Set Point)
  (is_regular_pentagon_ABCDE : ∀ (A B C D E : Point),
    ABCDE = {A, B, C, D, E} ∧
    (segment A B).is_regular_pentagon ABCDE)
  (is_equilateral_outside_triangle : ∀ (AB BC CD DE EA : Segment) (A B C D E P Q R S T : Point),
    ABCDE = {A, B, C, D, E} ∧
    (P, Q, R, S, T are_centers_of_equilateral_triangles ABCDE))
  : 1 = 1 :=  -- Formula for the area ratio, which is based on simplified calculations
sorry

end pentagon_area_ratio_l693_693784


namespace largest_prime_factor_7fac_8fac_l693_693947

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693947


namespace circle_point_inside_radius_l693_693704

theorem circle_point_inside_radius {
  (O : Type) [metric_space O] [nonempty O] (P : O) (r : ℝ) (h_r : r = 3)
  (hP : dist P O < r) : dist P O = 2 :=
by sorry

end circle_point_inside_radius_l693_693704


namespace parry_secretary_or_treasurer_probability_l693_693165

theorem parry_secretary_or_treasurer_probability (members : Finset ℕ) 
  (h_card : members.card = 10) :
  let parry := 0 in   -- assume Parry's identifier is 0
  ∃ (P : members → ℚ), 
    (P secretary + P treasurer = 19 / 90) := 
by
  let parry := 0
  have h_president := card_pred_SymDiff.card_eq_coe members
  sorry -- proof goes here

end parry_secretary_or_treasurer_probability_l693_693165


namespace checkered_square_division_l693_693374

theorem checkered_square_division (m n k d m1 n1 : ℕ) (h1 : m^2 = n * k)
  (h2 : d = Nat.gcd m n) (hm : m = m1 * d) (hn : n = n1 * d)
  (h3 : Nat.gcd m1 n1 = 1) : 
  ∃ (part_size : ℕ), 
    part_size = n ∧ (∃ (pieces : ℕ), pieces = k) ∧ m^2 = pieces * part_size := 
sorry

end checkered_square_division_l693_693374


namespace sum_of_real_numbers_solution_l693_693126

theorem sum_of_real_numbers_solution :
  (∑ x in {x : ℝ | |x^2 - 12*x + 34| = 2}.to_finset, x) = 18 :=
by { sorry }

end sum_of_real_numbers_solution_l693_693126


namespace perfect_square_eq_m_val_l693_693333

theorem perfect_square_eq_m_val (m : ℝ) (h : ∃ a : ℝ, x^2 - m * x + 49 = (x - a)^2) : m = 14 ∨ m = -14 :=
by
  sorry

end perfect_square_eq_m_val_l693_693333


namespace sequence_floor_equality_l693_693210

noncomputable def u : ℕ → ℝ 
| 0     := 2
| 1     := 5 / 2
| (n+2) := (u (n+1)) * ((u n) ^ 2 - 2) - u 1

theorem sequence_floor_equality :
  ∀ n : ℕ, n > 0 → ⌊u n⌋ = 2 ^ ((2 ^ n - (-1) ^ n) / 3) :=
by sorry

end sequence_floor_equality_l693_693210


namespace largest_prime_factor_7_factorial_plus_8_factorial_l693_693909

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | Nat.succ n' => (Nat.succ n') * factorial n'

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_prime_factor (n : Nat) : Nat :=
  let factors := List.filter is_prime (List.range (n+1)).filter (λ m => m ∣ n)
  List.maximum factors |>.getD 0

theorem largest_prime_factor_7_factorial_plus_8_factorial :
  ∀ n, n = 7! + 8! → largest_prime_factor n = 7 :=
by
  intro n,
  assume h : n = 7! + 8!,
  sorry

end largest_prime_factor_7_factorial_plus_8_factorial_l693_693909


namespace product_quality_difference_l693_693088

variable (n a b c d : ℕ)
variable (P_K_2 : ℝ → ℝ)

def first_class_freq_A := a / (a + b : ℕ)
def first_class_freq_B := c / (c + d : ℕ)

def K2 := (n : ℝ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem product_quality_difference
  (ha : a = 150) (hb : b = 50) 
  (hc : c = 120) (hd : d = 80)
  (hn : n = 400)
  (hK : P_K_2 0.010 = 6.635) : 
  first_class_freq_A a b = 3 / 4 ∧
  first_class_freq_B c d = 3 / 5 ∧
  K2 n a b c d > P_K_2 0.010 :=
by {
  sorry
}

end product_quality_difference_l693_693088


namespace find_a_of_exp_function_l693_693343

theorem find_a_of_exp_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a ^ 2 = 9) : a = 3 :=
sorry

end find_a_of_exp_function_l693_693343


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693994

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693994


namespace sixty_three_times_fifty_seven_l693_693622

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 := by
  sorry

end sixty_three_times_fifty_seven_l693_693622


namespace largest_prime_factor_7fac_8fac_l693_693938

theorem largest_prime_factor_7fac_8fac : 
  let f := (7! + 8!)
  let prime_factors_7 := {2, 3, 5, 7}
  f = 7! * 9 ∧ largest_prime_factor f = 7 :=
by
  sorry

end largest_prime_factor_7fac_8fac_l693_693938


namespace largest_prime_factor_of_7fact_plus_8fact_l693_693887

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the given expression 7! + 8!
def expr : ℕ := factorial 7 + factorial 8

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Prove that 7 is the largest prime factor of 7! + 8!
theorem largest_prime_factor_of_7fact_plus_8fact : ∃ p, is_prime p ∧ p ∣ expr ∧ ∀ q, is_prime q ∧ q ∣ expr → q ≤ p := by
  sorry

end largest_prime_factor_of_7fact_plus_8fact_l693_693887


namespace largest_prime_factor_of_7fact_8fact_l693_693901

/-- The largest prime factor of 7! + 8! is 7. -/
theorem largest_prime_factor_of_7fact_8fact : ∃ p, p.prime ∧ p ∣ (nat.factorial 7 + nat.factorial 8) ∧ ∀ q, q.prime ∧ q ∣ (nat.factorial 7 + nat.factorial 8) → q ≤ p ∧ p = 7 :=
by
  sorry

end largest_prime_factor_of_7fact_8fact_l693_693901


namespace largest_prime_factor_of_7_plus_8_factorial_l693_693987

-- Define factorial function
def factorial : ℕ → ℕ 
| 0       => 1
| 1       => 1
| (n + 2) => (n + 2) * factorial (n + 1)

-- Define the problem
theorem largest_prime_factor_of_7_plus_8_factorial :
  ∀ p, prime p ∧ p ∣ (factorial 7 + factorial 8) → p ≤ 7 := 
by
  sorry

end largest_prime_factor_of_7_plus_8_factorial_l693_693987


namespace largest_prime_factor_of_fact_sum_is_7_l693_693966

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def seven_fact := factorial 7
def eight_fact := factorial 8

-- Proposition statement
theorem largest_prime_factor_of_fact_sum_is_7 (n : ℕ) (m : ℕ) (hn : seven_fact = 5040) (hm : eight_fact = 40320) (h_sum : n + m = 7! + 8!) :
  ∃ p, p.prime ∧ ¬ ∃ q, q > p ∧ q.prime ∧ q ∣ (n + m) := sorry

end largest_prime_factor_of_fact_sum_is_7_l693_693966


namespace exp_inequality_solution_l693_693737

theorem exp_inequality_solution (x : ℝ) (h : 1 < Real.exp x ∧ Real.exp x < 2) : 0 < x ∧ x < Real.log 2 :=
by
  sorry

end exp_inequality_solution_l693_693737


namespace tangent_distance_l693_693848

-- Define the variables involved
variables {A O T : Type} [MetricSpace A]
variables (r l AO OT : ℝ) [Nontrivial r]

-- Defining the condition from the problem
def condition1 : l = (4/3) * r := sorry

-- Required theorem statement
theorem tangent_distance (h1 : l = (4/3) * r) :
  (AO = (5/3) * r) → (AO - r = (2/3) * r) := 
by {
  intro hAO,
  calc
    AO - r = (5/3) * r - r : by rw hAO
        ... = (5/3) * r - (3/3) * r : by norm_num
        ... = (2/3) * r : by ring,
  sorry
}

end tangent_distance_l693_693848


namespace jim_saving_amount_l693_693428

theorem jim_saving_amount
    (sara_initial_savings : ℕ)
    (sara_weekly_savings : ℕ)
    (jim_weekly_savings : ℕ)
    (weeks_elapsed : ℕ)
    (sara_total_savings : ℕ := sara_initial_savings + weeks_elapsed * sara_weekly_savings)
    (jim_total_savings : ℕ := weeks_elapsed * jim_weekly_savings)
    (savings_equal: sara_total_savings = jim_total_savings)
    (sara_initial_savings_value : sara_initial_savings = 4100)
    (sara_weekly_savings_value : sara_weekly_savings = 10)
    (weeks_elapsed_value : weeks_elapsed = 820) :
    jim_weekly_savings = 15 := 
by
  sorry

end jim_saving_amount_l693_693428


namespace passengers_heads_l693_693560

theorem passengers_heads (S C Cap X total_legs heads_cats heads_sailors_and_cooks heads_captain legs_cats legs_sailors_and_cooks legs_captain)
  (h1 : total_legs = 41)
  (h2 : heads_cats = 5 ∗ 1)
  (h3 : legs_cats = 5 ∗ 4)
  (h4 : heads_sailors_and_cooks = X ∗ 1)
  (h5 : legs_sailors_and_cooks = X ∗ 2)
  (h6 : heads_captain = 1)
  (h7 : legs_captain = 1)
  (h8 : legs_cats + legs_sailors_and_cooks + legs_captain = total_legs)
  (h9 : X = 10):
  heads_cats + heads_sailors_and_cooks + heads_captain = 16 :=
by
  sorry

end passengers_heads_l693_693560


namespace log_base_2_x_to_neg_half_l693_693289

theorem log_base_2 (x : ℝ) (h : log x / log 2 = 3) : x = 8 :=
by {
  sorry
}

theorem x_to_neg_half (x : ℝ) (hx : x = 8) : x^(-1/2) = sqrt 2 / 4 :=
by {
  sorry
}

end log_base_2_x_to_neg_half_l693_693289


namespace correct_operation_l693_693526

variables {a b : ℝ}

theorem correct_operation : (5 * a * b - 6 * a * b = -1 * a * b) := by
  sorry

end correct_operation_l693_693526


namespace split_cube_l693_693673

theorem split_cube (m : ℕ) (hm : m > 1) (h : ∃ k, ∃ l, l > 0 ∧ (3 + 2 * (k - 1)) = 59 ∧ (k + l = (m * (m - 1)) / 2)) : m = 8 :=
sorry

end split_cube_l693_693673


namespace solve_equations_l693_693829

theorem solve_equations (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 4 * a + 3 * b = 39) : a + b = 82 / 7 := by
  sorry

end solve_equations_l693_693829


namespace sixty_three_times_fifty_seven_l693_693625

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 := by
  sorry

end sixty_three_times_fifty_seven_l693_693625


namespace graph_transformation_matches_B_l693_693484

noncomputable def f (x : ℝ) : ℝ :=
  if (-3 : ℝ) ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0 -- Define this part to handle cases outside the given range.

noncomputable def g (x : ℝ) : ℝ :=
  f ((1 - x) / 2)

theorem graph_transformation_matches_B :
  g = some_graph_function_B := 
sorry

end graph_transformation_matches_B_l693_693484
