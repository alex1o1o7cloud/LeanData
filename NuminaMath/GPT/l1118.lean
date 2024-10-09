import Mathlib

namespace polynomial_simplification_l1118_111881

def A (x : ℝ) := 5 * x^2 + 4 * x - 1
def B (x : ℝ) := -x^2 - 3 * x + 3
def C (x : ℝ) := 8 - 7 * x - 6 * x^2

theorem polynomial_simplification (x : ℝ) : A x - B x + C x = 4 :=
by
  simp [A, B, C]
  sorry

end polynomial_simplification_l1118_111881


namespace min_number_of_lucky_weights_l1118_111808

-- Definitions and conditions
def weight (n: ℕ) := n -- A weight is represented as a natural number.

def is_lucky (weights: Finset ℕ) (w: ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ weights ∧ b ∈ weights ∧ a ≠ b ∧ w = a + b
-- w is "lucky" if it's the sum of two other distinct weights in the set.

def min_lucky_guarantee (weights: Finset ℕ) (k: ℕ) : Prop :=
  ∀ (w1 w2 : ℕ), w1 ∈ weights ∧ w2 ∈ weights →
    ∃ (lucky_weights : Finset ℕ), lucky_weights.card = k ∧
    (is_lucky weights w1 ∧ is_lucky weights w2 ∧ (w1 ≥ 3 * w2 ∨ w2 ≥ 3 * w1))
-- The minimum number k of "lucky" weights ensures there exist two weights 
-- such that their masses differ by at least a factor of three.

-- The theorem to be proven
theorem min_number_of_lucky_weights (weights: Finset ℕ) (h_distinct: weights.card = 100) :
  ∃ k, min_lucky_guarantee weights k ∧ k = 87 := 
sorry

end min_number_of_lucky_weights_l1118_111808


namespace subset_implies_value_l1118_111894

theorem subset_implies_value (a : ℝ) : (∀ x ∈ ({0, -a} : Set ℝ), x ∈ ({1, -1, 2 * a - 2} : Set ℝ)) → a = 1 := by
  sorry

end subset_implies_value_l1118_111894


namespace count_true_propositions_l1118_111867

theorem count_true_propositions :
  let prop1 := false  -- Proposition ① is false
  let prop2 := true   -- Proposition ② is true
  let prop3 := true   -- Proposition ③ is true
  let prop4 := false  -- Proposition ④ is false
  (if prop1 then 1 else 0) + (if prop2 then 1 else 0) +
  (if prop3 then 1 else 0) + (if prop4 then 1 else 0) = 2 :=
by
  -- The theorem is expected to be proven here
  sorry

end count_true_propositions_l1118_111867


namespace arrange_polynomial_l1118_111813

theorem arrange_polynomial :
  ∀ (x y : ℝ), 2 * x^3 * y - 4 * y^2 + 5 * x^2 = 5 * x^2 + 2 * x^3 * y - 4 * y^2 :=
by
  sorry

end arrange_polynomial_l1118_111813


namespace train_length_is_approx_l1118_111886

noncomputable def train_length : ℝ :=
  let speed_kmh : ℝ := 54
  let conversion_factor : ℝ := 1000 / 3600
  let speed_ms : ℝ := speed_kmh * conversion_factor
  let time_seconds : ℝ := 11.999040076793857
  speed_ms * time_seconds

theorem train_length_is_approx : abs (train_length - 179.99) < 0.001 := 
by
  sorry

end train_length_is_approx_l1118_111886


namespace subtraction_correct_l1118_111883

def x : ℝ := 5.75
def y : ℝ := 1.46
def result : ℝ := 4.29

theorem subtraction_correct : x - y = result := 
by
  sorry

end subtraction_correct_l1118_111883


namespace a_m_power_m_divides_a_n_power_n_a1_does_not_divide_any_an_power_n_l1118_111872

theorem a_m_power_m_divides_a_n_power_n:
  ∀ (a : ℕ → ℕ) (m : ℕ), (a 1).gcd (a 2) = 1 ∧ (∀ n, a (n + 2) = a (n + 1) * a n + 1) ∧ m > 1 → ∃ n > m, (a m) ^ m ∣ (a n) ^ n := by 
  sorry

theorem a1_does_not_divide_any_an_power_n:
  ∀ (a : ℕ → ℕ), (a 1).gcd (a 2) = 1 ∧ (∀ n, a (n + 2) = a (n + 1) * a n + 1) → ¬ ∃ n > 1, (a 1) ∣ (a n) ^ n := by
  sorry

end a_m_power_m_divides_a_n_power_n_a1_does_not_divide_any_an_power_n_l1118_111872


namespace no_solution_for_eq_l1118_111849

theorem no_solution_for_eq (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -3) :
  (12 / (x^2 - 9) - 2 / (x - 3) = 1 / (x + 3)) → False :=
sorry

end no_solution_for_eq_l1118_111849


namespace shaded_region_perimeter_l1118_111846

theorem shaded_region_perimeter (r : ℝ) (h : r = 12 / Real.pi) :
  3 * (24 / 6) = 12 := 
by
  sorry

end shaded_region_perimeter_l1118_111846


namespace john_saves_money_l1118_111823

theorem john_saves_money :
  let original_spending := 4 * 2
  let new_price_per_coffee := 2 + (2 * 0.5)
  let new_coffees := 4 / 2
  let new_spending := new_coffees * new_price_per_coffee
  original_spending - new_spending = 2 :=
by
  -- calculations omitted
  sorry

end john_saves_money_l1118_111823


namespace geom_arith_seq_l1118_111853

theorem geom_arith_seq (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_arith : 2 * a 3 - (a 5 / 2) = (a 5 / 2) - 3 * a 1) (hq : q > 0) :
  (a 2 + a 5) / (a 9 + a 6) = 1 / 9 :=
by
  sorry

end geom_arith_seq_l1118_111853


namespace bowling_ball_weight_l1118_111803

theorem bowling_ball_weight (b k : ℝ) (h1 : 5 * b = 3 * k) (h2 : 4 * k = 120) : b = 18 :=
by
  sorry

end bowling_ball_weight_l1118_111803


namespace tank_C_capacity_is_80_percent_of_tank_B_l1118_111856

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := 
  Real.pi * r^2 * h

theorem tank_C_capacity_is_80_percent_of_tank_B :
  ∀ (h_C c_C h_B c_B : ℝ), 
    h_C = 10 ∧ c_C = 8 ∧ h_B = 8 ∧ c_B = 10 → 
    (volume_of_cylinder (c_C / (2 * Real.pi)) h_C) / 
    (volume_of_cylinder (c_B / (2 * Real.pi)) h_B) * 100 = 80 := 
by 
  intros h_C c_C h_B c_B h_conditions
  obtain ⟨h_C_10, c_C_8, h_B_8, c_B_10⟩ := h_conditions
  sorry

end tank_C_capacity_is_80_percent_of_tank_B_l1118_111856


namespace marina_total_cost_l1118_111895

theorem marina_total_cost (E P R X : ℕ) 
    (h1 : 15 + E + P = 47)
    (h2 : 15 + R + X = 58) :
    15 + E + P + R + X = 90 :=
by
  -- The proof will go here
  sorry

end marina_total_cost_l1118_111895


namespace problem_statement_l1118_111878

-- Definitions related to the given conditions
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi) / 6)

theorem problem_statement :
  (∀ x1 x2 : ℝ, (x1 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → (x2 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → x1 < x2 → f x1 < f x2) →
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) →
  f (-((5 * Real.pi) / 12)) = (Real.sqrt 3) / 2 :=
by
  intros h_mono h_symm
  sorry

end problem_statement_l1118_111878


namespace subset_condition_for_A_B_l1118_111826

open Set

theorem subset_condition_for_A_B {a : ℝ} (A B : Set ℝ) 
  (hA : A = {x | abs (x - 2) < a}) 
  (hB : B = {x | x^2 - 2 * x - 3 < 0}) :
  B ⊆ A ↔ 3 ≤ a :=
  sorry

end subset_condition_for_A_B_l1118_111826


namespace cost_of_each_ring_l1118_111858

theorem cost_of_each_ring (R : ℝ) 
  (h1 : 4 * 12 + 8 * R = 80) : R = 4 :=
by 
  sorry

end cost_of_each_ring_l1118_111858


namespace greatest_possible_value_q_minus_r_l1118_111807

theorem greatest_possible_value_q_minus_r : ∃ q r : ℕ, 1025 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q - r = 31 :=
by {
  sorry
}

end greatest_possible_value_q_minus_r_l1118_111807


namespace pyramid_top_block_l1118_111877

theorem pyramid_top_block (a b c d e : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
                         (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
                         (h : a * b ^ 4 * c ^ 6 * d ^ 4 * e = 140026320) : 
                         (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 5) ∨ 
                         (a = 1 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 5) ∨ 
                         (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 1) ∨ 
                         (a = 5 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 1) := 
sorry

end pyramid_top_block_l1118_111877


namespace acute_angle_ACD_l1118_111880

theorem acute_angle_ACD (α : ℝ) (h : α ≤ 120) :
  ∃ (ACD : ℝ), ACD = Real.arcsin ((Real.tan (α / 2)) / Real.sqrt 3) :=
sorry

end acute_angle_ACD_l1118_111880


namespace tenth_term_arithmetic_sequence_l1118_111801

theorem tenth_term_arithmetic_sequence :
  ∀ (a₁ : ℚ) (d : ℚ), 
  (a₁ = 3/4) → (d = 1/2) →
  (a₁ + 9 * d) = 21/4 :=
by
  intro a₁ d ha₁ hd
  rw [ha₁, hd]
  sorry

end tenth_term_arithmetic_sequence_l1118_111801


namespace g_domain_l1118_111889

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arcsin (x ^ 3))

theorem g_domain : {x : ℝ | -1 < x ∧ x < 1} = Set {x | ∃ y, g x = y} :=
by
  sorry

end g_domain_l1118_111889


namespace answer_one_answer_two_answer_three_l1118_111816

def point_condition (A B : ℝ) (P : ℝ) (k : ℝ) : Prop := |A - P| = k * |B - P|

def question_one : Prop :=
  let A := -3
  let B := 6
  let k := 2
  let P := 3
  point_condition A B P k

def question_two : Prop :=
  ∀ x k : ℝ, |x + 2| + |x - 1| = 3 → point_condition (-3) 6 x k → (1 / 8 ≤ k ∧ k ≤ 4 / 5)

def question_three : Prop :=
  let A := -3
  let B := 6
  ∃ t : ℝ, t = 3 / 2 ∧ point_condition A (-3 + t) (6 - 2 * t) 3

theorem answer_one : question_one := by sorry

theorem answer_two : question_two := by sorry

theorem answer_three : question_three := by sorry

end answer_one_answer_two_answer_three_l1118_111816


namespace decreasing_number_4312_max_decreasing_number_divisible_by_9_l1118_111888

-- Definitions and conditions
def is_decreasing_number (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d4 ≠ 0 ∧
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
  (10 * d1 + d2 - (10 * d2 + d3) = 10 * d3 + d4)

def is_divisible_by_9 (n m : ℕ) : Prop :=
  (n + m) % 9 = 0

-- Theorem Statements
theorem decreasing_number_4312 : 
  is_decreasing_number 4312 :=
sorry

theorem max_decreasing_number_divisible_by_9 : 
  ∀ n, is_decreasing_number n ∧ is_divisible_by_9 (n / 10) (n % 1000) → n ≤ 8165 :=
sorry

end decreasing_number_4312_max_decreasing_number_divisible_by_9_l1118_111888


namespace find_g_1_l1118_111857

theorem find_g_1 (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (2*x - 3) = 2*x^2 - x + 4) : 
  g 1 = 11.5 :=
sorry

end find_g_1_l1118_111857


namespace tins_of_beans_left_l1118_111863

theorem tins_of_beans_left (cases : ℕ) (tins_per_case : ℕ) (damage_percentage : ℝ) (h_cases : cases = 15)
  (h_tins_per_case : tins_per_case = 24) (h_damage_percentage : damage_percentage = 0.05) :
  let total_tins := cases * tins_per_case
  let damaged_tins := total_tins * damage_percentage
  let tins_left := total_tins - damaged_tins
  tins_left = 342 :=
by
  sorry

end tins_of_beans_left_l1118_111863


namespace parabola_satisfies_given_condition_l1118_111843

variable {p : ℝ}
variable {x1 x2 : ℝ}

-- Condition 1: The equation of the parabola is y^2 = 2px where p > 0.
def parabola_equation (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Condition 2: The parabola has a focus F.
-- Condition 3: A line passes through the focus F with an inclination angle of π/3.
def line_through_focus (p : ℝ) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - p / 2)

-- Condition 4 & 5: The line intersects the parabola at points A and B with distance |AB| = 8.
def intersection_points (p : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 ≠ x2 ∧ parabola_equation p x1 (Real.sqrt 3 * (x1 - p / 2)) ∧ parabola_equation p x2 (Real.sqrt 3 * (x2 - p / 2)) ∧
  abs (x1 - x2) * Real.sqrt (1 + 3) = 8

-- The proof statement
theorem parabola_satisfies_given_condition (hp : 0 < p) (hintersect : intersection_points p x1 x2) : 
  parabola_equation 3 x1 (Real.sqrt 3 * (x1 - 3 / 2)) ∧ parabola_equation 3 x2 (Real.sqrt 3 * (x2 - 3 / 2)) := sorry

end parabola_satisfies_given_condition_l1118_111843


namespace trigonometric_identity_proof_l1118_111828

theorem trigonometric_identity_proof (θ : ℝ) 
  (h : Real.tan (θ + Real.pi / 4) = -3) : 
  2 * Real.sin θ ^ 2 - Real.cos θ ^ 2 = 7 / 5 :=
sorry

end trigonometric_identity_proof_l1118_111828


namespace abs_inequality_solution_l1118_111814

theorem abs_inequality_solution (x : ℝ) : |x + 2| + |x - 1| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 :=
sorry

end abs_inequality_solution_l1118_111814


namespace damaged_potatoes_l1118_111806

theorem damaged_potatoes (initial_potatoes : ℕ) (weight_per_bag : ℕ) (price_per_bag : ℕ) (total_sales : ℕ) :
  initial_potatoes = 6500 →
  weight_per_bag = 50 →
  price_per_bag = 72 →
  total_sales = 9144 →
  ∃ damaged_potatoes : ℕ, damaged_potatoes = initial_potatoes - (total_sales / price_per_bag) * weight_per_bag ∧
                               damaged_potatoes = 150 :=
by
  intros _ _ _ _ 
  exact sorry

end damaged_potatoes_l1118_111806


namespace quilt_shaded_fraction_l1118_111875

theorem quilt_shaded_fraction :
  let total_squares := 16
  let shaded_full_square := 4
  let shaded_half_triangles_as_square := 2
  let total_area := total_squares
  let shaded_area := shaded_full_square + shaded_half_triangles_as_square
  shaded_area / total_area = 3 / 8 :=
by
  sorry

end quilt_shaded_fraction_l1118_111875


namespace solve_quadratic_abs_l1118_111845

theorem solve_quadratic_abs (x : ℝ) :
  x^2 - |x| - 1 = 0 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 ∨ 
                   x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
sorry

end solve_quadratic_abs_l1118_111845


namespace part_one_part_two_l1118_111885

noncomputable def a (n : ℕ) : ℚ := if n = 1 then 1 / 2 else 2 ^ (n - 1) / (1 + 2 ^ (n - 1))

noncomputable def b (n : ℕ) : ℚ := n / a n

noncomputable def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => b (i + 1))

/-Theorem:
1. Prove that for all n > 0, a(n) = 2^(n-1) / (1 + 2^(n-1)).
2. Prove that for all n ≥ 3, S(n) > n^2 / 2 + 4.
-/
theorem part_one (n : ℕ) (h : n > 0) : a n = 2 ^ (n - 1) / (1 + 2 ^ (n - 1)) := sorry

theorem part_two (n : ℕ) (h : n ≥ 3) : S n > n ^ 2 / 2 + 4 := sorry

end part_one_part_two_l1118_111885


namespace overall_profit_no_discount_l1118_111893

theorem overall_profit_no_discount:
  let C_b := 100
  let C_p := 100
  let C_n := 100
  let profit_b := 42.5 / 100
  let profit_p := 35 / 100
  let profit_n := 20 / 100
  let S_b := C_b + (C_b * profit_b)
  let S_p := C_p + (C_p * profit_p)
  let S_n := C_n + (C_n * profit_n)
  let TCP := C_b + C_p + C_n
  let TSP := S_b + S_p + S_n
  let OverallProfit := TSP - TCP
  let OverallProfitPercentage := (OverallProfit / TCP) * 100
  OverallProfitPercentage = 32.5 :=
by sorry

end overall_profit_no_discount_l1118_111893


namespace find_a8_l1118_111802

/-!
Let {a_n} be an arithmetic sequence, with S_n representing the sum of the first n terms.
Given:
1. S_6 = 8 * S_3
2. a_3 - a_5 = 8
Prove: a_8 = -26
-/

noncomputable def arithmetic_seq (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

noncomputable def sum_arithmetic_seq (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem find_a8 (a_1 d : ℤ)
  (h1 : sum_arithmetic_seq a_1 d 6 = 8 * sum_arithmetic_seq a_1 d 3)
  (h2 : arithmetic_seq a_1 d 3 - arithmetic_seq a_1 d 5 = 8) :
  arithmetic_seq a_1 d 8 = -26 :=
  sorry

end find_a8_l1118_111802


namespace trajectory_of_Q_l1118_111805

-- Define Circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define Line l
def lineL (x y : ℝ) : Prop := x + y = 2

-- Define Conditions based on polar definitions
def polarCircle (ρ θ : ℝ) : Prop := ρ = 2

def polarLine (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 2

-- Define points on ray OP
def pointP (ρ₁ θ : ℝ) : Prop := ρ₁ = 2 / (Real.cos θ + Real.sin θ)
def pointR (ρ₂ θ : ℝ) : Prop := ρ₂ = 2

-- Prove the trajectory of Q
theorem trajectory_of_Q (O P R Q : ℝ × ℝ)
  (ρ₁ θ ρ ρ₂ : ℝ)
  (h1: circleC O.1 O.2)
  (h2: lineL P.1 P.2)
  (h3: polarCircle ρ₂ θ)
  (h4: polarLine ρ₁ θ)
  (h5: ρ * ρ₁ = ρ₂^2) :
  ρ = 2 * (Real.cos θ + Real.sin θ) :=
by
  sorry

end trajectory_of_Q_l1118_111805


namespace sum_of_palindromes_l1118_111825

-- Define a three-digit palindrome predicate
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ b < 10 ∧ n = 100*a + 10*b + a

-- Define the product of the two palindromes equaling 436,995
theorem sum_of_palindromes (a b : ℕ) (h_a : is_palindrome a) (h_b : is_palindrome b) (h_prod : a * b = 436995) : 
  a + b = 1332 :=
sorry

end sum_of_palindromes_l1118_111825


namespace money_initial_amounts_l1118_111890

theorem money_initial_amounts (x : ℕ) (A B : ℕ) 
  (h1 : A = 8 * x) 
  (h2 : B = 5 * x) 
  (h3 : (A - 50) = 4 * (B + 100) / 5) : 
  A = 800 ∧ B = 500 := 
sorry

end money_initial_amounts_l1118_111890


namespace competition_participants_solved_all_three_l1118_111899

theorem competition_participants_solved_all_three
  (p1 p2 p3 : ℕ → Prop)
  (total_participants : ℕ)
  (h1 : ∃ n, n = 85 * total_participants / 100 ∧ ∀ k, k < n → p1 k)
  (h2 : ∃ n, n = 80 * total_participants / 100 ∧ ∀ k, k < n → p2 k)
  (h3 : ∃ n, n = 75 * total_participants / 100 ∧ ∀ k, k < n → p3 k) :
  ∃ n, n ≥ 40 * total_participants / 100 ∧ ∀ k, k < n → p1 k ∧ p2 k ∧ p3 k :=
by
  sorry

end competition_participants_solved_all_three_l1118_111899


namespace expression_value_l1118_111851

theorem expression_value : 2016 - 2017 + 2018 - 2019 + 2020 = 2018 := 
by 
  sorry

end expression_value_l1118_111851


namespace base_7_to_base_10_conversion_l1118_111870

theorem base_7_to_base_10_conversion :
  (6 * 7^2 + 5 * 7^1 + 3 * 7^0) = 332 :=
by sorry

end base_7_to_base_10_conversion_l1118_111870


namespace erin_trolls_count_l1118_111879

def forest_trolls : ℕ := 6
def bridge_trolls : ℕ := 4 * forest_trolls - 6
def plains_trolls : ℕ := bridge_trolls / 2
def total_trolls : ℕ := forest_trolls + bridge_trolls + plains_trolls

theorem erin_trolls_count : total_trolls = 33 := by
  -- Proof omitted
  sorry

end erin_trolls_count_l1118_111879


namespace area_of_circle_l1118_111810

theorem area_of_circle (C : ℝ) (hC : C = 36 * Real.pi) : 
  ∃ k : ℝ, (∃ r : ℝ, r = 18 ∧ k = r^2 ∧ (pi * r^2 = k * pi)) ∧ k = 324 :=
by
  sorry

end area_of_circle_l1118_111810


namespace online_textbooks_cost_l1118_111854

theorem online_textbooks_cost (x : ℕ) :
  (5 * 10) + x + 3 * x = 210 → x = 40 :=
by
  sorry

end online_textbooks_cost_l1118_111854


namespace sum_pqrs_eq_3150_l1118_111882

theorem sum_pqrs_eq_3150
  (p q r s : ℝ)
  (h1 : p ≠ q) (h2 : p ≠ r) (h3 : p ≠ s) (h4 : q ≠ r) (h5 : q ≠ s) (h6 : r ≠ s)
  (hroots1 : ∀ x : ℝ, x^2 - 14*p*x - 15*q = 0 → (x = r ∨ x = s))
  (hroots2 : ∀ x : ℝ, x^2 - 14*r*x - 15*s = 0 → (x = p ∨ x = q)) :
  p + q + r + s = 3150 :=
by
  sorry

end sum_pqrs_eq_3150_l1118_111882


namespace function_is_monotonically_decreasing_l1118_111860

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 3)

theorem function_is_monotonically_decreasing :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → deriv f x ≤ 0 :=
by
  sorry

end function_is_monotonically_decreasing_l1118_111860


namespace sum_of_remainders_mod_30_l1118_111896

theorem sum_of_remainders_mod_30 (a b c : ℕ) (h1 : a % 30 = 14) (h2 : b % 30 = 11) (h3 : c % 30 = 19) :
  (a + b + c) % 30 = 14 :=
by
  sorry

end sum_of_remainders_mod_30_l1118_111896


namespace new_ratio_first_term_l1118_111898

theorem new_ratio_first_term (x : ℕ) (r1 r2 : ℕ) (new_r1 : ℕ) :
  r1 = 4 → r2 = 15 → x = 29 → new_r1 = r1 + x → new_r1 = 33 :=
by
  intros h_r1 h_r2 h_x h_new_r1
  rw [h_r1, h_x] at h_new_r1
  exact h_new_r1

end new_ratio_first_term_l1118_111898


namespace volume_of_extended_parallelepiped_l1118_111827

theorem volume_of_extended_parallelepiped :
  let main_box_volume := 3 * 3 * 6
  let external_boxes_volume := 2 * (3 * 3 * 1 + 3 * 6 * 1 + 3 * 6 * 1)
  let spheres_volume := 8 * (1 / 8) * (4 / 3) * Real.pi * (1 ^ 3)
  let cylinders_volume := 12 * (1 / 4) * Real.pi * 1^2 * 3 + 12 * (1 / 4) * Real.pi * 1^2 * 6
  main_box_volume + external_boxes_volume + spheres_volume + cylinders_volume = (432 + 52 * Real.pi) / 3 :=
by
  sorry

end volume_of_extended_parallelepiped_l1118_111827


namespace least_integer_solution_l1118_111837

theorem least_integer_solution (x : ℤ) (h : x^2 = 2 * x + 98) : x = -7 :=
by {
  sorry
}

end least_integer_solution_l1118_111837


namespace abs_quadratic_inequality_solution_l1118_111874

theorem abs_quadratic_inequality_solution (x : ℝ) :
  |x^2 - 4 * x + 3| ≤ 3 ↔ 0 ≤ x ∧ x ≤ 4 :=
by sorry

end abs_quadratic_inequality_solution_l1118_111874


namespace saree_stripes_l1118_111864

theorem saree_stripes
  (G : ℕ) (B : ℕ) (Br : ℕ) (total_stripes : ℕ) (total_patterns : ℕ)
  (h1 : G = 3 * Br)
  (h2 : B = 5 * G)
  (h3 : Br = 4)
  (h4 : B + G + Br = 100)
  (h5 : total_stripes = 100)
  (h6 : total_patterns = total_stripes / 3) :
  B = 84 ∧ total_patterns = 33 := 
  by {
    sorry
  }

end saree_stripes_l1118_111864


namespace dryer_cost_l1118_111866

theorem dryer_cost (washer_dryer_total_cost washer_cost dryer_cost : ℝ) (h1 : washer_dryer_total_cost = 1200) (h2 : washer_cost = dryer_cost + 220) :
  dryer_cost = 490 :=
by
  sorry

end dryer_cost_l1118_111866


namespace digit_making_527B_divisible_by_9_l1118_111804

theorem digit_making_527B_divisible_by_9 (B : ℕ) : 14 + B ≡ 0 [MOD 9] → B = 4 :=
by
  intro h
  -- sorry is used in place of the actual proof.
  sorry

end digit_making_527B_divisible_by_9_l1118_111804


namespace arithmetic_sequence_check_l1118_111832

theorem arithmetic_sequence_check 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h : ∀ n : ℕ, a (n+1) = a n + d) 
  : (∀ n : ℕ, (a n + 1) - (a (n - 1) + 1) = d) 
    ∧ (∀ n : ℕ, 2 * a (n + 1) - 2 * a n = 2 * d)
    ∧ (∀ n : ℕ, a (n + 1) - (a n + n) = d + 1) := 
by
  sorry

end arithmetic_sequence_check_l1118_111832


namespace regular_price_of_tire_l1118_111829

theorem regular_price_of_tire (x : ℝ) (h : 3 * x + 3 = 240) : x = 79 :=
by
  sorry

end regular_price_of_tire_l1118_111829


namespace min_expr_value_l1118_111871

theorem min_expr_value (a b c : ℝ) (h₀ : b > c) (h₁ : c > a) (h₂ : a > 0) (h₃ : b ≠ 0) :
  (∀ (a b c : ℝ), b > c → c > a → a > 0 → b ≠ 0 → 
   (2 + 6 * a^2 = (a+b)^3 / b^2 + (b-c)^2 / b^2 + (c-a)^3 / b^2) →
   2 <= (a + b)^3 / b^2 + (b - c)^2 / b^2 + (c - a)^3 / b^2) :=
by 
  sorry

end min_expr_value_l1118_111871


namespace Jennifer_future_age_Jordana_future_age_Jordana_current_age_l1118_111859

variable (Jennifer_age_now Jordana_age_now : ℕ)

-- Conditions
def age_in_ten_years (current_age : ℕ) : ℕ := current_age + 10
theorem Jennifer_future_age : age_in_ten_years Jennifer_age_now = 30 := sorry
theorem Jordana_future_age : age_in_ten_years Jordana_age_now = 3 * age_in_ten_years Jennifer_age_now := sorry

-- Question to prove
theorem Jordana_current_age : Jordana_age_now = 80 := sorry

end Jennifer_future_age_Jordana_future_age_Jordana_current_age_l1118_111859


namespace fred_grew_38_cantaloupes_l1118_111811

/-
  Fred grew some cantaloupes. Tim grew 44 cantaloupes.
  Together, they grew a total of 82 cantaloupes.
  Prove that Fred grew 38 cantaloupes.
-/

theorem fred_grew_38_cantaloupes (T F : ℕ) (h1 : T = 44) (h2 : T + F = 82) : F = 38 :=
by
  rw [h1] at h2
  linarith

end fred_grew_38_cantaloupes_l1118_111811


namespace prove_box_problem_l1118_111873

noncomputable def boxProblem : Prop :=
  let height1 := 2
  let width1 := 4
  let length1 := 6
  let clay1 := 48
  let height2 := 3 * height1
  let width2 := 2 * width1
  let length2 := 1.5 * length1
  let volume1 := height1 * width1 * length1
  let volume2 := height2 * width2 * length2
  let n := (volume2 / volume1) * clay1
  n = 432

theorem prove_box_problem : boxProblem := by
  sorry

end prove_box_problem_l1118_111873


namespace probability_correct_l1118_111892

-- Defining the values on the spinner
inductive SpinnerValue
| Bankrupt
| Thousand
| EightHundred
| FiveThousand
| Thousand'

open SpinnerValue

-- Function to get value in number from SpinnerValue
def value (v : SpinnerValue) : ℕ :=
  match v with
  | Bankrupt => 0
  | Thousand => 1000
  | EightHundred => 800
  | FiveThousand => 5000
  | Thousand' => 1000

-- Total number of spins
def total_spins : ℕ := 3

-- Total possible outcomes
def total_outcomes : ℕ := (5 : ℕ) ^ total_spins

-- Number of favorable outcomes (count of permutations summing to 5800)
def favorable_outcomes : ℕ :=
  12  -- This comes from solution steps

-- The probability as a ratio of favorable outcomes to total outcomes
def probability_of_5800_in_three_spins : ℚ :=
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_5800_in_three_spins = 12 / 125 := by
  sorry

end probability_correct_l1118_111892


namespace fixed_monthly_charge_for_100_GB_l1118_111824

theorem fixed_monthly_charge_for_100_GB
  (fixed_charge M : ℝ)
  (extra_charge_per_GB : ℝ := 0.25)
  (total_bill : ℝ := 65)
  (GB_over : ℝ := 80)
  (extra_charge : ℝ := GB_over * extra_charge_per_GB) :
  total_bill = M + extra_charge → M = 45 :=
by sorry

end fixed_monthly_charge_for_100_GB_l1118_111824


namespace derivative_of_function_y_l1118_111818

noncomputable def function_y (x : ℝ) : ℝ := (x^2) / (x + 3)

theorem derivative_of_function_y (x : ℝ) :
  deriv function_y x = (x^2 + 6 * x) / ((x + 3)^2) :=
by 
  -- sorry since the proof is not required
  sorry

end derivative_of_function_y_l1118_111818


namespace compound_interest_years_l1118_111844

-- Define the parameters
def principal : ℝ := 7500
def future_value : ℝ := 8112
def annual_rate : ℝ := 0.04
def compounding_periods : ℕ := 1

-- Define the proof statement
theorem compound_interest_years :
  ∃ t : ℕ, future_value = principal * (1 + annual_rate / compounding_periods) ^ t ∧ t = 2 :=
by
  sorry

end compound_interest_years_l1118_111844


namespace find_x_l1118_111820

theorem find_x : ∃ (x : ℝ), x > 0 ∧ x + 17 = 60 * (1 / x) ∧ x = 3 :=
by
  sorry

end find_x_l1118_111820


namespace tank_capacity_l1118_111869

-- Definitions from conditions
def initial_fraction := (1 : ℚ) / 4  -- The tank is 1/4 full initially
def added_amount := 5  -- Adding 5 liters

-- The proof problem to show that the tank's total capacity c equals 60 liters
theorem tank_capacity
  (c : ℚ)  -- The total capacity of the tank in liters
  (h1 : c / 4 + added_amount = c / 3)  -- Adding 5 liters makes the tank 1/3 full
  : c = 60 := 
sorry

end tank_capacity_l1118_111869


namespace acute_triangle_l1118_111835

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ area, 
    (area = (1 / 2) * a * b * Real.sin C) ∧
    (a / Real.sin A = 2 * c / Real.sqrt 3) ∧
    (c = Real.sqrt 7) ∧
    (area = (3 * Real.sqrt 3) / 2)

theorem acute_triangle (a b c A B C : ℝ) (h : triangle_ABC a b c A B C) :
  C = 60 ∧ a^2 + b^2 = 13 :=
by
  obtain ⟨_, h_area, h_sine, h_c, h_area_eq⟩ := h
  sorry

end acute_triangle_l1118_111835


namespace avg_salary_officers_l1118_111812

-- Definitions of the given conditions
def avg_salary_employees := 120
def avg_salary_non_officers := 110
def num_officers := 15
def num_non_officers := 495

-- The statement to be proven
theorem avg_salary_officers : (15 * (15 * X) / (15 + 495)) = 450 :=
by
  sorry

end avg_salary_officers_l1118_111812


namespace sum_of_integers_eq_l1118_111831

-- We define the conditions
variables (x y : ℕ)
-- The conditions specified in the problem
def diff_condition : Prop := x - y = 16
def prod_condition : Prop := x * y = 63

-- The theorem stating that given the conditions, the sum is 2*sqrt(127)
theorem sum_of_integers_eq : diff_condition x y → prod_condition x y → x + y = 2 * Real.sqrt 127 :=
by
  sorry

end sum_of_integers_eq_l1118_111831


namespace find_a_purely_imaginary_l1118_111838

noncomputable def purely_imaginary_condition (a : ℝ) : Prop :=
    (2 * a - 1) / (a^2 + 1) = 0 ∧ (a + 2) / (a^2 + 1) ≠ 0

theorem find_a_purely_imaginary :
    ∀ (a : ℝ), purely_imaginary_condition a ↔ a = 1/2 := 
by
  sorry

end find_a_purely_imaginary_l1118_111838


namespace ryan_fraction_l1118_111887

-- Define the total amount of money
def total_money : ℕ := 48

-- Define that Ryan owns a fraction R of the total money
variable {R : ℚ}

-- Define the debts
def ryan_owes_leo : ℕ := 10
def leo_owes_ryan : ℕ := 7

-- Define the final amount Leo has after settling the debts
def leo_final_amount : ℕ := 19

-- Define the condition that Leo and Ryan together have $48
def leo_plus_ryan (leo_amount ryan_amount : ℚ) : Prop := 
  leo_amount + ryan_amount = total_money

-- Define Ryan's amount as a fraction R of the total money
def ryan_amount (R : ℚ) : ℚ := R * total_money

-- Define Leo's amount before debts were settled
def leo_amount_before_debts : ℚ := (leo_final_amount : ℚ) + leo_owes_ryan

-- Define the equation after settling debts
def leo_final_eq (leo_amount_before_debts : ℚ) : Prop :=
  (leo_amount_before_debts - ryan_owes_leo = leo_final_amount)

-- The Lean theorem that needs to be proved
theorem ryan_fraction :
  ∃ (R : ℚ), leo_plus_ryan (leo_amount_before_debts - ryan_owes_leo) (ryan_amount R)
  ∧ leo_final_eq leo_amount_before_debts
  ∧ R = 11 / 24 :=
sorry

end ryan_fraction_l1118_111887


namespace remainder_of_11_pow_2023_mod_33_l1118_111819

theorem remainder_of_11_pow_2023_mod_33 : (11 ^ 2023) % 33 = 11 := 
by
  sorry

end remainder_of_11_pow_2023_mod_33_l1118_111819


namespace proof_problem_l1118_111855

theorem proof_problem (a b A B : ℝ) (f : ℝ → ℝ)
  (h_f : ∀ θ : ℝ, f θ ≥ 0)
  (h_f_def : ∀ θ : ℝ, f θ = 1 + a * Real.cos θ + b * Real.sin θ + A * Real.sin (2 * θ) + B * Real.cos (2 * θ)) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end proof_problem_l1118_111855


namespace greatest_b_value_l1118_111836

theorem greatest_b_value (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b:ℝ) * x + 15 ≠ -9) ↔ b = 9 :=
sorry

end greatest_b_value_l1118_111836


namespace boat_speed_in_still_water_l1118_111861

-- Define the conditions
def speed_of_stream : ℝ := 3 -- (speed in km/h)
def time_downstream : ℝ := 1 -- (time in hours)
def time_upstream : ℝ := 1.5 -- (time in hours)

-- Define the goal by proving the speed of the boat in still water
theorem boat_speed_in_still_water : 
  ∃ V_b : ℝ, (V_b + speed_of_stream) * time_downstream = (V_b - speed_of_stream) * time_upstream ∧ V_b = 15 :=
by
  sorry -- (Proof will be provided here)

end boat_speed_in_still_water_l1118_111861


namespace alberto_spent_more_l1118_111848

-- Define the expenses of Alberto and Samara
def alberto_expenses : ℕ := 2457
def samara_oil_expense : ℕ := 25
def samara_tire_expense : ℕ := 467
def samara_detailing_expense : ℕ := 79
def samara_total_expenses : ℕ := samara_oil_expense + samara_tire_expense + samara_detailing_expense

-- State the theorem to prove the difference in expenses
theorem alberto_spent_more :
  alberto_expenses - samara_total_expenses = 1886 := by
  sorry

end alberto_spent_more_l1118_111848


namespace circle_people_count_l1118_111817

def num_people (n : ℕ) (a b : ℕ) : Prop :=
  a = 7 ∧ b = 18 ∧ (b = a + (n / 2))

theorem circle_people_count (n : ℕ) (a b : ℕ) (h : num_people n a b) : n = 24 :=
by
  sorry

end circle_people_count_l1118_111817


namespace set_A_membership_l1118_111830

theorem set_A_membership (U : Finset ℕ) (A : Finset ℕ) (B : Finset ℕ)
  (hU : U.card = 193)
  (hB : B.card = 49)
  (hneither : (U \ (A ∪ B)).card = 59)
  (hAandB : (A ∩ B).card = 25) :
  A.card = 110 := sorry

end set_A_membership_l1118_111830


namespace tan_pi_over_12_plus_tan_7pi_over_12_l1118_111852

theorem tan_pi_over_12_plus_tan_7pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (7 * Real.pi / 12)) = -4 * (3 - Real.sqrt 3) / 5 :=
by
  sorry

end tan_pi_over_12_plus_tan_7pi_over_12_l1118_111852


namespace flower_shop_percentage_l1118_111822

theorem flower_shop_percentage (C : ℕ) : 
  let V := (1/3 : ℝ) * C
  let T := (1/12 : ℝ) * C
  let R := T
  let total := C + V + T + R
  (C / total) * 100 = 66.67 := 
by
  sorry

end flower_shop_percentage_l1118_111822


namespace brownies_pieces_count_l1118_111800

theorem brownies_pieces_count:
  let pan_width := 24
  let pan_length := 15
  let piece_width := 3
  let piece_length := 2
  pan_width * pan_length / (piece_width * piece_length) = 60 := 
by
  sorry

end brownies_pieces_count_l1118_111800


namespace event_probability_l1118_111834

noncomputable def probability_event : ℝ :=
  let a : ℝ := (1 : ℝ) / 2
  let b : ℝ := (3 : ℝ) / 2
  let interval_length : ℝ := 2
  (b - a) / interval_length

theorem event_probability :
  probability_event = (3 : ℝ) / 4 :=
by
  -- Proof step will be supplied here
  sorry

end event_probability_l1118_111834


namespace sequence_bounded_l1118_111884

theorem sequence_bounded (a : ℕ → ℕ) (a1 : ℕ) (h1 : a 0 = a1)
  (heven : ∀ n : ℕ, ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ a (2 * n) = a (2 * n - 1) - d)
  (hodd : ∀ n : ℕ, ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ a (2 * n + 1) = a (2 * n) + d) :
  ∀ n : ℕ, a n ≤ 10 * a1 := 
by
  sorry

end sequence_bounded_l1118_111884


namespace percentage_profit_l1118_111876

theorem percentage_profit 
  (C S : ℝ) 
  (h : 29 * C = 24 * S) : 
  ((S - C) / C) * 100 = 20.83 := 
by
  sorry

end percentage_profit_l1118_111876


namespace volume_of_cuboid_is_250_cm3_l1118_111821

-- Define the edge length of the cube
def edge_length (a : ℕ) : ℕ := 5

-- Define the volume of a single cube
def cube_volume := (edge_length 5) ^ 3

-- Define the total volume of the cuboid formed by placing two such cubes in a line
def cuboid_volume := 2 * cube_volume

-- Theorem stating the volume of the cuboid formed
theorem volume_of_cuboid_is_250_cm3 : cuboid_volume = 250 := by
  sorry

end volume_of_cuboid_is_250_cm3_l1118_111821


namespace lauren_earnings_tuesday_l1118_111868

def money_from_commercials (num_commercials : ℕ) (rate_per_commercial : ℝ) : ℝ :=
  num_commercials * rate_per_commercial

def money_from_subscriptions (num_subscriptions : ℕ) (rate_per_subscription : ℝ) : ℝ :=
  num_subscriptions * rate_per_subscription

def total_money (num_commercials : ℕ) (rate_per_commercial : ℝ) (num_subscriptions : ℕ) (rate_per_subscription : ℝ) : ℝ :=
  money_from_commercials num_commercials rate_per_commercial + money_from_subscriptions num_subscriptions rate_per_subscription

theorem lauren_earnings_tuesday (num_commercials : ℕ) (rate_per_commercial : ℝ) (num_subscriptions : ℕ) (rate_per_subscription : ℝ) :
  num_commercials = 100 → rate_per_commercial = 0.50 → num_subscriptions = 27 → rate_per_subscription = 1.00 → 
  total_money num_commercials rate_per_commercial num_subscriptions rate_per_subscription = 77 :=
by
  intros h1 h2 h3 h4
  simp [money_from_commercials, money_from_subscriptions, total_money, h1, h2, h3, h4]
  sorry

end lauren_earnings_tuesday_l1118_111868


namespace sufficient_but_not_necessary_l1118_111891

theorem sufficient_but_not_necessary (x : ℝ) : (x < -2 → x ≤ 0) → ¬(x ≤ 0 → x < -2) :=
by
  sorry

end sufficient_but_not_necessary_l1118_111891


namespace determine_m_value_l1118_111847

theorem determine_m_value
  (m : ℝ)
  (h : ∀ x : ℝ, -7 < x ∧ x < -1 ↔ mx^2 + 8 * m * x + 28 < 0) :
  m = 4 := by
  sorry

end determine_m_value_l1118_111847


namespace compute_expression_l1118_111841

theorem compute_expression : (-1) ^ 2014 + (π - 3.14) ^ 0 - (1 / 2) ^ (-2) = -2 := by
  sorry

end compute_expression_l1118_111841


namespace mistaken_divisor_is_12_l1118_111842

theorem mistaken_divisor_is_12 (dividend : ℕ) (mistaken_divisor : ℕ) (correct_divisor : ℕ) 
  (mistaken_quotient : ℕ) (correct_quotient : ℕ) (remainder : ℕ) :
  remainder = 0 ∧ correct_divisor = 21 ∧ mistaken_quotient = 42 ∧ correct_quotient = 24 ∧ 
  dividend = mistaken_quotient * mistaken_divisor ∧ dividend = correct_quotient * correct_divisor →
  mistaken_divisor = 12 :=
by 
  sorry

end mistaken_divisor_is_12_l1118_111842


namespace lucas_seq_units_digit_M47_l1118_111839

def lucas_seq : ℕ → ℕ := 
  sorry -- skipped sequence generation for brevity

def M (n : ℕ) : ℕ :=
  if n = 0 then 3 else
  if n = 1 then 1 else
  lucas_seq n -- will call the lucas sequence generator

-- Helper function to get the units digit of a number
def units_digit (n: ℕ) : ℕ :=
  n % 10

theorem lucas_seq_units_digit_M47 : units_digit (M (M 6)) = 3 := 
sorry

end lucas_seq_units_digit_M47_l1118_111839


namespace parabola_latus_rectum_l1118_111840

theorem parabola_latus_rectum (p : ℝ) (H : ∀ y : ℝ, y^2 = 2 * p * -2) : p = 4 :=
sorry

end parabola_latus_rectum_l1118_111840


namespace quadratic_with_real_roots_l1118_111809

theorem quadratic_with_real_roots: 
  ∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4 * x₁ + k = 0 ∧ x₂^2 + 4 * x₂ + k = 0) ↔ (k ≤ 4) := 
by 
  sorry

end quadratic_with_real_roots_l1118_111809


namespace num_whole_numbers_between_sqrt_50_and_sqrt_200_l1118_111833

theorem num_whole_numbers_between_sqrt_50_and_sqrt_200 :
  let lower := Nat.ceil (Real.sqrt 50)
  let upper := Nat.floor (Real.sqrt 200)
  lower <= upper ∧ (upper - lower + 1) = 7 :=
by
  sorry

end num_whole_numbers_between_sqrt_50_and_sqrt_200_l1118_111833


namespace coordinates_of_A_l1118_111815

theorem coordinates_of_A 
  (a : ℝ)
  (h1 : (a - 1) = 3 + (3 * a - 2)) :
  (a - 1, 3 * a - 2) = (-2, -5) :=
by
  sorry

end coordinates_of_A_l1118_111815


namespace max_alpha_beta_square_l1118_111850

theorem max_alpha_beta_square (k : ℝ) (α β : ℝ)
  (h1 : α^2 - (k - 2) * α + (k^2 + 3 * k + 5) = 0)
  (h2 : β^2 - (k - 2) * β + (k^2 + 3 * k + 5) = 0)
  (h3 : α ≠ β) :
  (α^2 + β^2) ≤ 18 :=
sorry

end max_alpha_beta_square_l1118_111850


namespace exponential_inequality_l1118_111897

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ (a * b)^((a + b) / 2) :=
sorry

end exponential_inequality_l1118_111897


namespace avg_and_var_of_scaled_shifted_data_l1118_111865

-- Definitions of average and variance
noncomputable def avg (l: List ℝ) : ℝ := (l.sum) / l.length
noncomputable def var (l: List ℝ) : ℝ := (l.map (λ x => (x - avg l) ^ 2)).sum / l.length

theorem avg_and_var_of_scaled_shifted_data
  (n : ℕ)
  (x : Fin n → ℝ)
  (h_avg : avg (List.ofFn x) = 2)
  (h_var : var (List.ofFn x) = 3) :
  avg (List.ofFn (λ i => 2 * x i + 3)) = 7 ∧ var (List.ofFn (λ i => 2 * x i + 3)) = 12 := by
  sorry

end avg_and_var_of_scaled_shifted_data_l1118_111865


namespace minimum_value_of_expression_l1118_111862

theorem minimum_value_of_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 1 / a + 1 / b = 1) :
  ∃ (x : ℝ), x = (1 / (a - 1) + 9 / (b - 1)) ∧ x = 6 :=
by
  sorry

end minimum_value_of_expression_l1118_111862
