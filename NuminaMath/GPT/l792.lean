import Mathlib

namespace students_per_table_l792_792189

theorem students_per_table (total_students tables students_bathroom students_canteen added_students exchange_students : ℕ) 
  (h1 : total_students = 47)
  (h2 : tables = 6)
  (h3 : students_bathroom = 3)
  (h4 : students_canteen = 3 * students_bathroom)
  (h5 : added_students = 2 * 4)
  (h6 : exchange_students = 3 + 3 + 3) :
  (total_students - (students_bathroom + students_canteen + added_students + exchange_students)) / tables = 3 := 
by 
  sorry

end students_per_table_l792_792189


namespace angle_A_eq_angle_B_implies_rectangle_l792_792944

-- Define the quadrilateral ABCD with given properties
structure Quadrilateral (A B C D : Type) :=
  (AD_parallel_BC : Prop)
  (AB_eq_CD : Prop)

-- The conditions given in the problem
variable (A B C D : Type)

def quadrilateral_ABCD (A B C D : Type) : Quadrilateral A B C D :=
{ AD_parallel_BC := AD_parallel_BC A B C D,
  AB_eq_CD := AB_eq_CD A B C D }

-- Define the angles
variable (angle_A angle_B angle_C angle_D : ℝ)

-- The math proof statement
theorem angle_A_eq_angle_B_implies_rectangle 
  (h1 : quadrilateral_ABCD A B C D)
  (h2 : angle_A = angle_B) : 
  angle_A = 90 ∧ angle_B = 90 ∧ angle_C = 90 ∧ angle_D = 90 :=
sorry

end angle_A_eq_angle_B_implies_rectangle_l792_792944


namespace trajectory_circle_l792_792033

noncomputable def midpoint (A B : ℝ) : ℝ := (A + B) / 2

theorem trajectory_circle (A B P : ℝ) (M : ℝ) (hM : M = midpoint A B) (hP : (B - A) = 2 * (P - M)) :
  ∃ r, ∀ (x : ℝ), (x - M)^2 + (M - P)^2 = r^2 :=
by
  sorry

end trajectory_circle_l792_792033


namespace range_of_a_is_le_2_l792_792850

variable (a : ℝ)

def proposition_p : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log 0.5 (x^2 + 2 * x + a)
def proposition_q : Prop := ∀ x > 2, ∀ y > 0, y = (x - a)^2 → x > a

theorem range_of_a_is_le_2 (h_p : proposition_p a) (h_q : proposition_q a) :
  ∀ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) → a ≤ 2 :=
sorry

end range_of_a_is_le_2_l792_792850


namespace number_of_days_l792_792721

variables (S Wx Wy : ℝ)

-- Given conditions
def condition1 : Prop := S = 36 * Wx
def condition2 : Prop := S = 45 * Wy

-- The lean statement to prove the number of days D = 20
theorem number_of_days (h1 : condition1 S Wx) (h2 : condition2 S Wy) : 
  S / (Wx + Wy) = 20 :=
by
  sorry

end number_of_days_l792_792721


namespace perfect_family_card_le_l792_792702

-- Define the conditions
def perfect_family (U : Finset α) (ℱ : Finset (Finset α)) : Prop :=
  ∀ (X₁ X₂ X₃ : Finset α), X₁ ∈ ℱ → X₂ ∈ ℱ → X₃ ∈ ℱ →
    ((X₁ \ X₂) ∩ X₃ = ∅) ∨ ((X₂ \ X₁) ∩ X₃ = ∅)

-- Main theorem statement
theorem perfect_family_card_le (U : Finset α) (ℱ : Finset (Finset α))
  (h₁ : ∀ {X}, X ∈ ℱ → X ⊆ U)
  (h₂ : perfect_family U ℱ) :
  ℱ.card ≤ U.card + 1 :=
sorry

end perfect_family_card_le_l792_792702


namespace problem_conjugate_number_l792_792224

noncomputable def complex_conjugate_1_plus_i : Prop :=
  complex.conj (2 * complex.I / (1 + complex.I)) = 1 - complex.I

theorem problem_conjugate_number : complex_conjugate_1_plus_i :=
by
  sorry

end problem_conjugate_number_l792_792224


namespace graph_intersection_sum_l792_792358

theorem graph_intersection_sum 
  (g : ℝ → ℝ)
  (hintersct_1 : g (-1) = 3)
  (hintersct_2 : g 1 = 3) :
  g 1 = g (1 - 2) ∧ (1 + 3) = 4 := 
by 
  -- state the necessary points on the graph
  have h1 : g 1 = 3, from hintersct_2,
  have h2 : g (1 - 2) = g (-1), from hintersct_1,
  -- show the intersection point (1,3) and sum of coordinates
  exact ⟨h1, h2, rfl⟩,
  sorry

end graph_intersection_sum_l792_792358


namespace carla_correct_questions_l792_792937

theorem carla_correct_questions :
  ∀ (Drew_correct Drew_wrong Carla_wrong Total_questions Carla_correct : ℕ), 
    Drew_correct = 20 →
    Drew_wrong = 6 →
    Carla_wrong = 2 * Drew_wrong →
    Total_questions = 52 →
    Carla_correct = Total_questions - Carla_wrong →
    Carla_correct = 40 :=
by
  intros Drew_correct Drew_wrong Carla_wrong Total_questions Carla_correct
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end carla_correct_questions_l792_792937


namespace angle_DAO_plus_AOH_l792_792960

-- Definitions translating conditions
def acute_angled_triangle (A B C : Type*) := sorry
def circumcenter (A B C O : Type*) := sorry
def orthocenter (A B C H : Type*) := sorry
def equilateral_triangle (D O H : Type*) := sorry


theorem angle_DAO_plus_AOH (A B C O H D : Type*)
  (h1 : acute_angled_triangle A B C)
  (h2 : AB > AC)
  (h3 : ∠ACB - ∠ABC = 30)
  (h4 : circumcenter A B C O)
  (h5 : orthocenter A B C H)
  (h6 : equilateral_triangle D O H) :
  ∠DAO + ∠AOH = 60 :=
sorry

end angle_DAO_plus_AOH_l792_792960


namespace largest_angle_of_convex_hexagon_l792_792624

theorem largest_angle_of_convex_hexagon (a d : ℕ) (h_seq : ∀ i, a + i * d < 180 ∧ a + i * d > 0)
  (h_sum : 6 * a + 15 * d = 720)
  (h_seq_arithmetic : ∀ (i j : ℕ), (a + i * d) < (a + j * d) ↔ i < j) :
  ∃ m : ℕ, (m = a + 5 * d ∧ m = 175) :=
by
  sorry

end largest_angle_of_convex_hexagon_l792_792624


namespace complex_expression_identity_l792_792423

noncomputable theory
open Complex

def given_z : ℂ := (1/2) + ((sqrt 3)/2) * I

theorem complex_expression_identity :
  (given_z + 2 * given_z ^ 2 + 3 * given_z ^ 3 + 4 * given_z ^ 4 + 5 * given_z ^ 5 + 6 * given_z ^ 6) = 6 * conj given_z :=
by
  sorry

end complex_expression_identity_l792_792423


namespace rectangle_of_parallel_sides_and_equal_angles_l792_792952

theorem rectangle_of_parallel_sides_and_equal_angles
  (A B C D : Type)
  (AD_parallel_BC : ∀ x : A, x ∈ AD → is_parallel x BC)
  (AB_eq_CD : AB = CD)
  (angle_A_eq_angle_B : ∠ A = ∠ B) :
  is_rectangle (quadrilateral A B C D) :=
by
  sorry

end rectangle_of_parallel_sides_and_equal_angles_l792_792952


namespace lilian_uncertain_mushrooms_l792_792183

theorem lilian_uncertain_mushrooms :
  ∀ (total safe poisonous uncertain : ℕ),
  total = 32 →
  safe = 9 →
  poisonous = 2 * safe →
  uncertain = total - (safe + poisonous) →
  uncertain = 5 := 
by
  intros total safe poisonous uncertain h_total h_safe h_poisonous h_uncertain
  rw [h_total, h_safe, h_poisonous, h_uncertain]
  sorry

end lilian_uncertain_mushrooms_l792_792183


namespace koala_fiber_absorption_l792_792535

theorem koala_fiber_absorption (x : ℝ) (h1 : 0 < x) (h2 : x * 0.30 = 15) : x = 50 :=
sorry

end koala_fiber_absorption_l792_792535


namespace spent_on_new_tires_is_correct_l792_792533

-- Conditions
def amount_spent_on_speakers : ℝ := 136.01
def amount_spent_on_cd_player : ℝ := 139.38
def total_amount_spent : ℝ := 387.85

-- Goal
def amount_spent_on_tires : ℝ := total_amount_spent - (amount_spent_on_speakers + amount_spent_on_cd_player)

theorem spent_on_new_tires_is_correct : 
  amount_spent_on_tires = 112.46 :=
by
  sorry

end spent_on_new_tires_is_correct_l792_792533


namespace tangent_of_11pi_over_4_l792_792780

theorem tangent_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 :=
sorry

end tangent_of_11pi_over_4_l792_792780


namespace total_vacations_and_classes_l792_792054

def kelvin_classes := 90
def grant_vacations := 4 * kelvin_classes
def total := grant_vacations + kelvin_classes

theorem total_vacations_and_classes :
  total = 450 :=
by
  sorry

end total_vacations_and_classes_l792_792054


namespace correct_proposition_l792_792296

-- Definitions based on conditions
def p : Prop := ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0
def not_p : Prop := ∀ x : ℝ, x^2 + 2 * x + 2016 > 0

-- Proof statement
theorem correct_proposition : p → not_p :=
by sorry

end correct_proposition_l792_792296


namespace perfect_number_representation_l792_792336

/-- A perfect number is a positive integer whose sum of all proper divisors equals itself. -/
def is_perfect (n : ℕ) : Prop :=
  n > 0 ∧ ∑ i in (finset.filter (λ d, d < n ∧ n % d = 0) (finset.range n.succ)), i = n

/-- If 2^n - 1 is prime, then 2^(n - 1) * (2^n - 1) is a perfect number. -/
lemma perfect_number_of_2_pow_sub_1_prime {n : ℕ} (hn : nat.prime (2^n - 1)) :
  is_perfect (2^(n - 1) * (2^n - 1)) :=
sorry

/-- 8128 can be represented as a sum of consecutive positive integer powers of 2. -/
theorem perfect_number_representation :
  8128 = ∑ i in finset.range (13 - 6), 2^(6 + i) :=
sorry

end perfect_number_representation_l792_792336


namespace sum_of_two_squares_from_product_of_six_integers_eq_108_l792_792520
open Nat

theorem sum_of_two_squares_from_product_of_six_integers_eq_108 :
  ∃ (p q : ℕ), (∃ (s1 s2 : Finset ℕ), (s1.card = 6 ∧ s2.card = 6 ∧ 
    ∀ (x : ℕ), x ∈ s1 → x ∈ (Finset.range 10)) ∧ 
    ∀ (y : ℕ), y ∈ s2 → y ∈ (Finset.range 10) ∧ s1 ≠ s2 ∧ 
    (∀ (x y : ℕ), x ∈ s1 ∧ y ∈ s2 ∧ (Finset.product s1).val.prod.pow 2 = p^2 ∧ 
    (Finset.product s2).val.prod.pow 2 = q^2) ∧ p + q = 108) := 
sorry

end sum_of_two_squares_from_product_of_six_integers_eq_108_l792_792520


namespace triangle_area_l792_792074

noncomputable def center := (1 : ℝ, 3 : ℝ)
noncomputable def radius := 2 * real.sqrt 2
noncomputable def line (x y : ℝ) := x + y - 2 = 0
noncomputable def circle (x y : ℝ) := x^2 + y^2 - 2*x - 6*y + 2 = 0

theorem triangle_area : 
  ∃ A B : ℝ × ℝ, 
    line A.1 A.2 ∧ line B.1 B.2 ∧ circle A.1 A.2 ∧ circle B.1 B.2 ∧
    let AB := dist A B, 
    let d := real.sqrt 2,
    let area := 1/2 * AB * d in
    area = 2 * real.sqrt 3 :=
sorry

end triangle_area_l792_792074


namespace flowchart_execution_l792_792587

variable a b c : Nat

theorem flowchart_execution :
  (∀ (a b c : Nat), a = 21 → b = 32 → c = 75 →
    (∃ a' b' c' : Nat, a' = c ∧ b' = a ∧ c' = b ∧ a' = 75 ∧ b' = 21 ∧ c' = 32)) :=
by
  intros a b c ha hb hc
  use c, a, b
  simp [ha, hb, hc]
  sorry

end flowchart_execution_l792_792587


namespace max_Bk_l792_792773

theorem max_Bk (k : ℕ) (h0 : 0 ≤ k) (h1 : k ≤ 2000) : k = 181 ↔ 
  ∀ k' : ℕ, (0 ≤ k' ∧ k' ≤ 2000) → B k ≤ B k' :=
sorry

def B (k : ℕ) : ℝ :=
  if (0 ≤ k ∧ k ≤ 2000) then ((nat.choose 2000 k) : ℝ) * (0.1 ^ k) else 0

end max_Bk_l792_792773


namespace fraction_pow_zero_l792_792275

theorem fraction_pow_zero
  (a : ℤ) (b : ℤ)
  (h_a : a = -325123789)
  (h_b : b = 59672384757348)
  (h_nonzero_num : a ≠ 0)
  (h_nonzero_denom : b ≠ 0) :
  (a / b : ℚ) ^ 0 = 1 :=
by {
  sorry
}

end fraction_pow_zero_l792_792275


namespace ratio_shorter_to_longer_l792_792693

theorem ratio_shorter_to_longer (total_length shorter_length longer_length : ℕ) (h1 : total_length = 40) 
(h2 : shorter_length = 16) (h3 : longer_length = total_length - shorter_length) : 
(shorter_length / Nat.gcd shorter_length longer_length) / (longer_length / Nat.gcd shorter_length longer_length) = 2 / 3 :=
by
  sorry

end ratio_shorter_to_longer_l792_792693


namespace find_omega_l792_792032

theorem find_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : (π / ω = π / 2)) : ω = 2 :=
sorry

end find_omega_l792_792032


namespace vertical_asymptotes_count_l792_792487

/-- The function y = (x + 2) / (x^2 - 8x + 15) has exactly 2 vertical asymptotes. -/
theorem vertical_asymptotes_count : 
  let y := λ x : ℝ, (x + 2) / (x^2 - 8x + 15) in
  ∃ a b : ℝ, a ≠ b ∧ (∀ x, x = a ∨ x = b ↔ (x^2 - 8x + 15) = 0) ∧
  (∃ count, count = 2) := 
sorry

end vertical_asymptotes_count_l792_792487


namespace Cara_sitting_between_pairs_l792_792744

theorem Cara_sitting_between_pairs (n : ℕ) (h_n : n = 6) : Nat.choose n 2 = 15 :=
by
  rw h_n
  exact Nat.choose_self_eq_max (6 - 1) 2
-- sorry

end Cara_sitting_between_pairs_l792_792744


namespace not_p_or_not_q_implies_p_and_q_and_p_or_q_l792_792926

variable (p q : Prop)

theorem not_p_or_not_q_implies_p_and_q_and_p_or_q (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
sorry

end not_p_or_not_q_implies_p_and_q_and_p_or_q_l792_792926


namespace common_divisors_l792_792895

theorem common_divisors (a b : ℕ) (ha : a = 9240) (hb : b = 8820) : 
  let g := Nat.gcd a b in 
  g = 420 ∧ Nat.divisors 420 = 24 :=
by
  have gcd_ab := Nat.gcd_n at ha hb
  have fact := Nat.factorize 420
  have divisors_420: ∀ k : Nat, g = 420 ∧ k = 24 := sorry
  exact divisors_420 24

end common_divisors_l792_792895


namespace a4_is_5_l792_792917

-- Define the condition x^5 = a_n + a_1(x-1) + a_2(x-1)^2 + a_3(x-1)^3 + a_4(x-1)^4 + a_5(x-1)^5
noncomputable def polynomial_identity (x a_n a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  x^5 = a_n + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5

-- Define the theorem statement
theorem a4_is_5 (x a_n a_1 a_2 a_3 a_5 : ℝ) (h : polynomial_identity x a_n a_1 a_2 a_3 5 a_5) : a_4 = 5 :=
 by
 sorry

end a4_is_5_l792_792917


namespace min_value_of_quadratic_l792_792281

theorem min_value_of_quadratic :
  ∃ x : ℝ, ∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ 12 :=
by
  let f : ℝ → ℝ := λ x, 4 * x^2 + 8 * x + 16
  use -1
  intros y hy
  have h : f(-1) = 12 := by
    calc
      f(-1) = 4 * (-1)^2 + 8 * (-1) + 16 : by rfl
         ... = 4 * 1 - 8 + 16 : by norm_num
         ... = 4 - 8 + 16 : by rfl
         ... = -4 + 16 : by norm_num
         ... = 12 : by norm_num
  rw [← hy, h]
  have non_neg : 4 * (x + 1)^2 ≥ 0 := by
    apply mul_nonneg
    norm_num
    apply pow_two_nonneg
  exact add_le_add non_neg (by norm_num)

end min_value_of_quadratic_l792_792281


namespace perpendicular_lines_max_OP_AP_l792_792002

theorem perpendicular_lines (α : ℝ) (hα : 0 ≤ α ∧ α < π) (t : ℝ) :
  let l1 := fun t => (t * Real.cos α, t * Real.sin α)
  let l2 := fun ρ θ => ρ * Real.cos (θ - α) = 2 * Real.sin (α + Real.pi / 6)
  ∃ x y, l1 t = (x, y) ∧ l2 (Real.sqrt (x^2 + y^2)) (Real.atan2 y x) := sorry

theorem max_OP_AP (α : ℝ) (hα : 0 ≤ α ∧ α < π) :
  let A := (2, Real.pi / 3)
  ∃ P : ℝ × ℝ, 
    let OP := Real.sqrt (P.1^2 + P.2^2)
    let AP := Real.sqrt ((P.1 - 2 * Real.cos (Real.pi / 3))^2 + (P.2 - 2 * Real.sin (Real.pi / 3))^2)
    OP * AP = 2 := sorry

end perpendicular_lines_max_OP_AP_l792_792002


namespace _l792_792150

noncomputable def problem := 
  ∃ (P : Finset ℕ) (A : Finset ℕ)
  (color : ℕ → ℕ), 
  P.card = 10 ∧
  (∀ p ∈ P, Nat.Prime p) ∧
  (∀ n ∈ A, 1 < n ∧ ∀ p ∈ P, p ∣ n → Nat.Prime p) ∧
  (Function.Injective (λ p, color p) ∧ P ⊆ A) ∧

  -- Conditions:
  (∀ m n ∈ A, color (m * n) = color m ∨ color (m * n) = color n) ∧
  (∀ (R S : ℕ), R ≠ S → 
    ¬ ∃ j k m n ∈ A, 
    color j = R ∧ color k = R ∧ color m = S ∧ color n = S ∧
    j ∣ m ∧ n ∣ k
  ) →

  -- Question:
  ∃ p ∈ P, ∀ n ∈ A, p ∣ n → color n = color p

#eval problem -- This should be 'true' if the theorem statement is valid.

end _l792_792150


namespace problem_arithmetic_sequence_l792_792435

variable {α : Type*} [linear_ordered_field α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + d * (n - 1)

theorem problem_arithmetic_sequence :
  ∃ (a d : α) (a_n : ℕ → α) (S_n : ℕ → α),
    d ≠ 0 ∧
    a = 3 ∧
    (∀ n, a_n n = arithmetic_sequence a d n) ∧
    (∀ n, S_n n = n * (n + 2)) ∧
    (a_n 1 = 3 ∧ 
     a_n 4 = 3 + 3 * d ∧ 
     a_n 13 = 3 + 12 * d ∧ 
     (a_n 4)^2 = (a_n 1) * (a_n 13)) →
    (∀ n, a_n n = 2 * n + 1) ∧
    (∀ n : ℕ, 
      (∑ i in finset.range n, 1 / (S_n (i + 1))) = 
        1 / 2 * ((1 - 1 / 3) + 
                 (1 / 2 - 1 / 4) + 
                 (1 / 3 - 1 / 5) + 
                 (1 / (n - 1) - 1 / (n + 1)) + 
                 (1 / n - 1 / (n + 2))) ∧
      (∑ i in finset.range n, 1 / (S_n (i + 1))) = 
        3 / 4 - (2 * n + 3) / (2 * (n + 1) * (n + 2))) :=
by sorry

end problem_arithmetic_sequence_l792_792435


namespace hyperbola_ratio_l792_792042

noncomputable def point : Type := ℝ × ℝ

def hyperbola (x y : ℝ) := x^2 - (y^2 / 4) = 1

def right_focus : point := (Real.sqrt 5, 0)

def line_through_point_focus_parallel_asymptote (x:ℝ) : point := 
  let y := 2 * (x - Real.sqrt 5)
  (x, y)

def line_OM (x : ℝ) : point := 
  let y := - (1 / 2) * x
  (x, y)

def inner_product (p1 p2 : point) := p1.1 * p2.1 + p1.2 * p2.2

theorem hyperbola_ratio (P M F : point) (h1 : hyperbola P.1 P.2) 
  (h2 : F = right_focus) (h3 : P = line_through_point_focus_parallel_asymptote P.1) 
  (h4 : M = line_OM M.1) (h5 : inner_product (0, 0) (P.1 - F.1, P.2 - F.2) = 0) :
  (|P.1 - M.1| + |P.2 - M.2|) / (|F.1 - P.1| + |F.2 - P.2|) = 1 / 2 :=
sorry

end hyperbola_ratio_l792_792042


namespace sum_divisible_by_each_l792_792796

theorem sum_divisible_by_each : 
  let nums := [1, 2, 3, 6, 12, 24, 48, 96, 192, 384] in
  let s := nums.sum in
  ∀ n ∈ nums, s % n = 0 :=
by
  let nums := [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
  let s := nums.sum
  intros n hn
  sorry

end sum_divisible_by_each_l792_792796


namespace tan_of_11pi_over_4_is_neg1_l792_792794

noncomputable def tan_periodic : Real := 2 * Real.pi

theorem tan_of_11pi_over_4_is_neg1 :
  Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Angle normalization using periodicity of tangent function
  have h1 : Real.tan (11 * Real.pi / 4) = Real.tan (11 * Real.pi / 4 - 2 * Real.pi) := 
    by rw [Real.tan_periodic]
  -- Further normalization
  have h2 : 11 * Real.pi / 4 - 2 * Real.pi = 3 * Real.pi / 4 := sorry
  -- Evaluate tangent at the simplified angle
  have h3 : Real.tan (3 * Real.pi / 4) = -Real.tan (Real.pi / 4) := sorry
  -- Known value of tangent at common angle
  have h4 : Real.tan (Real.pi / 4) = 1 := by simpl tan
  rw [h2, h3, h4]
  norm_num

end tan_of_11pi_over_4_is_neg1_l792_792794


namespace find_valid_pairs_l792_792803

open Nat

def has_zero_digit (n : ℕ) : Prop :=
  ∃ d, 0 ≤ d ∧ d < 10 ∧ (10^d ∣ n) ∧ (n / 10^d) % 10 = 0

def valid_pairs_count (sum : ℕ) : ℕ :=
  (Fin.sum Finset.Ico 1 sum (λ a, if has_zero_digit a ∨ has_zero_digit (sum - a) then 0 else 1))

theorem find_valid_pairs :
  valid_pairs_count 500 = 309 :=
by
  sorry

end find_valid_pairs_l792_792803


namespace find_year_after_2020_l792_792103

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_year_after_2020 :
  ∃ y : ℕ, 2020 < y ∧ sum_of_digits y = 4 ∧ (∀ z : ℕ, 2020 < z ∧ sum_of_digits z = 4 → y ≤ z) := 
begin
  sorry,
end

end find_year_after_2020_l792_792103


namespace fraction_of_liams_stickers_l792_792355

theorem fraction_of_liams_stickers (e : ℕ) :
  (∀ (Liam Noah Emma : ℕ),
    Liam = 4 * Emma ∧
    Noah = 3 * Emma →
    (∃ (fraction : ℚ),
      fraction = 1 / 6 ∧
      Liam - fraction * Liam = (Liam + Noah + Emma) / 3 ∧
      Noah + fraction * Liam = (Liam + Noah + Emma) / 3)
  ) :=
by
  intros Liam Noah Emma h
  use 1/6
  cases' h with hLiam hNoah
  have h_total : Liam + Noah + Emma = 8 * Emma := 
    by rw [hLiam, hNoah]; nat.smul_eq_mul; linarith
  have h_target : (Liam + Noah + Emma) / 3 = 8 * Emma / 3 := 
    by rw h_total
  split
  · refl
  · split
    · rw [h_target, hLiam]; field_simp; ring
    · rw [h_target, hNoah]; ring using hLiam, hNoah; field_simp; norm_num; ring

end fraction_of_liams_stickers_l792_792355


namespace intersection_complement_l792_792479

open Set

def U := ℤ
def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 + 2 * x = 0}

theorem intersection_complement :
  A ∩ (U \ B) = {-1, 1, 2} := by
  sorry

end intersection_complement_l792_792479


namespace minimum_value_f_l792_792799

noncomputable def f (x : ℝ) : ℝ := (Real.tan x)^2 + 3 * (Real.tan x) + 6 * (Real.cot x) + 4 * (Real.cot x)^2 - 1

theorem minimum_value_f : (∃ y, ∀ x, (0 < x) ∧ (x < Real.pi / 2) → f x ≥ y) ∧ 
                         (∃ x, (0 < x) ∧ (x < Real.pi / 2) ∧ f x = 3 + 6 * Real.sqrt 2) :=
sorry

end minimum_value_f_l792_792799


namespace integer_solutions_eq_0_or_2_l792_792291

theorem integer_solutions_eq_0_or_2 (a : ℤ) (x : ℤ) : 
  (a * x^2 + 6 = 0) → (a = -6 ∧ (x = 1 ∨ x = -1)) ∨ (¬ (a = -6) ∧ (x ≠ 1) ∧ (x ≠ -1)) :=
by 
sorry

end integer_solutions_eq_0_or_2_l792_792291


namespace dana_cookies_to_make_l792_792820

-- Definitions for the problem conditions
def art_cookie_area : ℝ := 4 * 3
def art_total_dough : ℝ := 10 * art_cookie_area

def dana_cookie_area : ℝ := 3 * 3
def desired_earnings : ℝ := 18
def price_per_cookie : ℝ := 0.90

-- The formal Lean statement based on the given conditions and calculations
theorem dana_cookies_to_make (X : ℕ) :
  (art_total_dough = 120) →
  (∀ (dough : ℝ), dough = art_total_dough) →
  (desired_earnings / price_per_cookie = 20) →
  (dana_cookie_area * X = art_total_dough) →
  X = 13
  :=
begin
  sorry,
end

end dana_cookies_to_make_l792_792820


namespace intersection_M_N_l792_792088

section

def M (x : ℝ) : Prop := sqrt x < 4
def N (x : ℝ) : Prop := 3 * x >= 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | (1 / 3) ≤ x ∧ x < 16} :=
by
  sorry

end

end intersection_M_N_l792_792088


namespace sum_f_eq_sqrt_3_l792_792829

-- Definition of the function
def f (x : ℝ) : ℝ := sin (π * x + π) / 3 - (sqrt 3) * cos (π * x + π) / 3

-- Main theorem to be proved
theorem sum_f_eq_sqrt_3 : (∑ x in finset.range 2015, f x) = sqrt 3 :=
sorry

end sum_f_eq_sqrt_3_l792_792829


namespace polygon_area_leq_17_point_5_l792_792237

noncomputable def polygonArea (a b c d : ℝ) : ℝ :=
  -- Area calculated with given projections
  let areaSubtract := (1/2) * (a^(2:ℕ)) + (1/2) * ((c - a)^(2:ℕ)) +
                      (1/2) * (b^(2:ℕ)) +  (1/2) * ((d - b)^(2:ℕ))
  20 - areaSubtract

theorem polygon_area_leq_17_point_5 (a b : ℝ) (c d : ℝ):
  -- Given conditions as projections on axes and bisectors
  a = 4 -> b = 5 -> c = 3*Real.sqrt(2) -> d = 4*Real.sqrt(2) ->
  polygonArea a b c d ≤ 17.5 :=
begin
  intros ha hb hc hd,
  -- Here we would input the proof steps, but we use 'sorry' to indicate that we're skipping the proof steps.
  sorry
end

end polygon_area_leq_17_point_5_l792_792237


namespace cos_double_theta_l792_792069

theorem cos_double_theta (theta : ℝ) 
  (h : ∑' n : ℕ, (cos θ)^(2 * n) = 5) : cos (2 * θ) = 3 / 5 := 
sorry

end cos_double_theta_l792_792069


namespace smallest_of_three_numbers_l792_792745

theorem smallest_of_three_numbers : ∀ (a b c : ℕ), (a = 5) → (b = 8) → (c = 4) → min (min a b) c = 4 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  sorry

end smallest_of_three_numbers_l792_792745


namespace area_BCK_right_triangle_l792_792324

-- Define the necessary geometrical elements and hypotheses
variables {α : Type*} [ordered_field α]
variables (a b : α)

noncomputable def area_of_triangle_BCK (a b : α) : α :=
  (a^3 * b) / (2 * (a^2 + b^2))

theorem area_BCK_right_triangle (a b : α) (h : a > 0) (hc_pos : b > 0) :
  ∃ K,
  (right_triangle a b) ∧ 
  (circle_on_BC_as_diameter a) ∧ 
  (intersects_hypotenuse_at_K a b) →
  (area_of_triangle_BCK a b = (a^3 * b) / (2 * (a^2 + b^2))) :=
by
  sorry

end area_BCK_right_triangle_l792_792324


namespace collatz_A_collatz_B_collatz_C_collatz_D_l792_792216

def collatz_step (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def collatz_sequence (a₀ : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate collatz_step n a₀

def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Question A
theorem collatz_A : collatz_sequence 7 11 = 5 := sorry

-- Question B
theorem collatz_B : ¬(∀ n, collatz_sequence 16 n ≤ collatz_sequence 16 (n - 1)) := sorry

-- Question C
theorem collatz_C (a₅ : ℕ) (h₁ : a₅ = 1) (h₂ : ∀ i : ℕ, i ∈ [1, 2, 3, 4] → collatz_sequence 5 i ≠ 1) :
  collatz_sequence 5 0 = 5 := sorry

-- Question D
theorem collatz_D : 
  let a₀ := 10;
  let seq := [collatz_sequence a₀ 1, collatz_sequence a₀ 2, collatz_sequence a₀ 3, collatz_sequence a₀ 4, collatz_sequence a₀ 5, collatz_sequence a₀ 6];
  let count_odds := seq.countp is_odd;
  count_odds * (seq.length - count_odds) + (count_odds * (count_odds - 1) / 2) + (seq.length - count_odds) * ((seq.length - count_odds) - 1) / 2 = 3/5 * (seq.length * (seq.length - 1) / 2) := sorry

end collatz_A_collatz_B_collatz_C_collatz_D_l792_792216


namespace quadratic_function_eq_y_eq_x_squared_plus_2x_l792_792496

-- Definitions for the given conditions
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def has_vertex_at (f : ℝ → ℝ) (v : ℝ × ℝ) : Prop :=
  ∃ a b, a ≠ 0 ∧ f = λ x, a * (x - v.1)^2 + v.2

def intersects_x_axis_with_segment_length (f : ℝ → ℝ) (l : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ abs (x₂ - x₁) = l

-- Problem statement to be proved
theorem quadratic_function_eq_y_eq_x_squared_plus_2x :
  ∃ f : ℝ → ℝ, is_quadratic f ∧ has_vertex_at f (-1, -1) ∧ intersects_x_axis_with_segment_length f 2 ∧ (∀ x, f x = x^2 + 2 * x) :=
sorry

end quadratic_function_eq_y_eq_x_squared_plus_2x_l792_792496


namespace Petya_wins_optimally_l792_792254

-- Defining the game state and rules
inductive GameState
| PetyaWin
| VasyaWin

-- Rules of the game
def game_rule (n : ℕ) : Prop :=
  n > 0 ∧ (n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2)

-- Determine the winner given the initial number of minuses
def determine_winner (n : ℕ) : GameState :=
  if n % 3 = 0 then GameState.PetyaWin else GameState.VasyaWin

-- Theorem: Petya will win the game if both play optimally
theorem Petya_wins_optimally (n : ℕ) (h1 : n = 2021) (h2 : game_rule n) : determine_winner n = GameState.PetyaWin :=
by {
  sorry
}

end Petya_wins_optimally_l792_792254


namespace part_a_possible_part_b_not_possible_l792_792970

-- Define the problem of marking vertices in a regular 14-sided polygon
def k_marked_vertices_rectangles (k : ℕ) (marked_vertices : Finset (Fin 14)) : Prop :=
  k = marked_vertices.card ∧ ∀ (a b c d : Fin 14),
    a ∈ marked_vertices ∧ b ∈ marked_vertices ∧ c ∈ marked_vertices ∧ d ∈ marked_vertices ∧
    parallel (a, b) (c, d) ∧ parallel (a, d) (b, c) →
    rectangle a b c d

-- Prove two main cases
theorem part_a_possible : k_marked_vertices_rectangles 6 (Finset.of_list [0, 1, 3, 7, 8, 10]) :=
by sorry

theorem part_b_not_possible (k : ℕ) (k_ge_7 : k ≥ 7) (marked_vertices : Finset (Fin 14)) :
  k = marked_vertices.card → ¬ k_marked_vertices_rectangles k marked_vertices :=
by sorry

end part_a_possible_part_b_not_possible_l792_792970


namespace remainder_div_30_l792_792560

-- Define the conditions as Lean definitions
variables (x y z p q : ℕ)

-- Hypotheses based on the conditions
def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- assuming the conditions
axiom x_div_by_4 : is_divisible_by x 4
axiom y_div_by_5 : is_divisible_by y 5
axiom z_div_by_6 : is_divisible_by z 6
axiom p_div_by_7 : is_divisible_by p 7
axiom q_div_by_3 : is_divisible_by q 3

-- Statement to be proved
theorem remainder_div_30 : ((x^3) * (y^2) * (z * p * q + (x + y)^3) - 10) % 30 = 20 :=
by {
  sorry -- the proof will go here
}

end remainder_div_30_l792_792560


namespace simplify_sqrt_90000_l792_792207

theorem simplify_sqrt_90000 : Real.sqrt 90000 = 300 :=
by
  /- Proof goes here -/
  sorry

end simplify_sqrt_90000_l792_792207


namespace feathers_per_crown_l792_792413

theorem feathers_per_crown (total_feathers total_crowns feathers_per_crown : ℕ) 
  (h₁ : total_feathers = 6538) 
  (h₂ : total_crowns = 934) 
  (h₃ : feathers_per_crown = total_feathers / total_crowns) : 
  feathers_per_crown = 7 := 
by 
  sorry

end feathers_per_crown_l792_792413


namespace length_of_curve_l792_792009

-- Definitions according to the problem
def Rectangle (A B C D : Type) := (AB BC CD DA: ℝ)

variable (A B C D : ℝ)

-- The given conditions in the problem
variables (line : Type) (AB BC : ℝ) (ℓ : line)
variable (on_line : CD ∈ ℓ)

-- Condition that ABCD is a rectangle
def is_rectangle (AB BC CD DA: ℝ) := AB = 30 ∧ BC = 40

-- Problem statement
theorem length_of_curve (AB BC CD DA: ℝ) :
  is_rectangle AB BC CD DA → 
  AB = 30 → BC = 40 →
  let diagonal := (AB^2 + BC^2).sqrt in
  let length_of_quarter_circle := (1/2:ℝ) * Real.pi * diagonal in 
  2 * length_of_quarter_circle = 50 * Real.pi :=
by
  intros
  sorry

end length_of_curve_l792_792009


namespace tom_apple_slices_left_l792_792260

theorem tom_apple_slices_left
  (initial_apples : ℕ)
  (slices_per_apple : ℕ)
  (fraction_given_to_jerry : ℚ)
  (fraction_eaten_by_tom : ℚ)
  (initial_slices : ℕ := initial_apples * slices_per_apple)
  (slices_given_to_jerry : ℕ := fraction_given_to_jerry * initial_slices)
  (remaining_slices : ℕ := initial_slices - slices_given_to_jerry)
  (slices_eaten_by_tom : ℕ := fraction_eaten_by_tom * remaining_slices)
  (slices_left : ℕ := remaining_slices - slices_eaten_by_tom) :
  initial_apples = 2 →
  slices_per_apple = 8 →
  fraction_given_to_jerry = 3 / 8 →
  fraction_eaten_by_tom = 1 / 2 →
  slices_left = 5 :=
by
  intros h1 h2 h3 h4
  have hs1: initial_slices = 2 * 8, from by rw [h1, h2],
  have hs2: slices_given_to_jerry = (3 / 8) * 16, from by rw [h3, hs1],
  have hs3: remaining_slices = 16 - 6, from by rw [hs2],
  have hs4: slices_eaten_by_tom = (1 / 2) * 10, from by rw [h4, hs3],
  have hs5: slices_left = 10 - 5, from by rw [hs4],
  exact hs5

end tom_apple_slices_left_l792_792260


namespace sum_x_coords_of_f_eq_x_plus_2_l792_792613

def segment1 := (λ x, (x * 2 - 5))
def segment2 := (λ x, (x * -1 + 1))
def segment3 := (λ x, (x * 4 + 3) / 2)
def segment4 := (λ x, (x * -1 + 3))
def segment5 := (λ x, (x * 2 - 3))

def f (x : ℝ) : ℝ :=
  if h : x < -2 then segment1 x
  else if h : x < -1 then segment2 x
  else if h : x < 1 then segment3 x
  else if h : x < 2 then segment4 x
  else segment5 x

theorem sum_x_coords_of_f_eq_x_plus_2 :
  let intersections := [(-2, 0), (0.5, 2.5)]
  in let sum_x := ∑ q in intersections, q.1 
  in sum_x = -1.5 := sorry

end sum_x_coords_of_f_eq_x_plus_2_l792_792613


namespace collinear_points_OO1M1_l792_792991

open EuclideanGeometry

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def orthocenter (A B C : Point) : Point := sorry

theorem collinear_points_OO1M1 (A B C A1 B1 C1 O O1 M1 : Point) 
  (hO : O = circumcenter A B C)
  (hO1 : O1 = incenter A B C)
  (hM1 : M1 = orthocenter A1 B1 C1)
  (h_touch_A : touches_incircle A B C A1)
  (h_touch_B : touches_incircle A B C B1)
  (h_touch_C : touches_incircle A B C C1) : 
  collinear O O1 M1 :=
sorry

end collinear_points_OO1M1_l792_792991


namespace greatest_fifty_supportive_X_l792_792998

def fifty_supportive (X : ℝ) : Prop :=
∀ (a : Fin 50 → ℝ),
  (∑ i, a i).floor = (∑ i, a i) →
  ∃ i, |a i - 0.5| ≥ X

theorem greatest_fifty_supportive_X :
  ∀ X : ℝ, fifty_supportive X ↔ X ≤ 0.01 := sorry

end greatest_fifty_supportive_X_l792_792998


namespace tony_squat_capacity_l792_792650

theorem tony_squat_capacity :
  let curl_weight := 90 in
  let military_press_weight := 2 * curl_weight in
  let total_squat_weight := 5 * military_press_weight in
  total_squat_weight = 900 := by
  sorry

end tony_squat_capacity_l792_792650


namespace ratio_grass_area_weeded_l792_792186

/-- Lucille earns six cents for every weed she pulls. -/
def earnings_per_weed : ℕ := 6

/-- There are eleven weeds in the flower bed. -/
def weeds_flower_bed : ℕ := 11

/-- There are fourteen weeds in the vegetable patch. -/
def weeds_vegetable_patch : ℕ := 14

/-- There are thirty-two weeds in the grass around the fruit trees. -/
def weeds_grass_total : ℕ := 32

/-- Lucille bought a soda for 99 cents on her break. -/
def soda_cost : ℕ := 99

/-- Lucille has 147 cents left after the break. -/
def cents_left : ℕ := 147

/-- Statement to prove: The ratio of the grass area Lucille weeded to the total grass area around the fruit trees is 1:2. -/
theorem ratio_grass_area_weeded :
  (earnings_per_weed * (weeds_flower_bed + weeds_vegetable_patch) + earnings_per_weed * (weeds_flower_bed + (weeds_grass_total - (earnings_per_weed + soda_cost)) / earnings_per_weed) = soda_cost + cents_left)
→ ((earnings_per_weed  * (32 - (147 + 99) / earnings_per_weed)) / weeds_grass_total) = 1 / 2 :=
by
  sorry

end ratio_grass_area_weeded_l792_792186


namespace flag_yellow_area_percentage_l792_792718

theorem flag_yellow_area_percentage (s w : ℝ) (h_flag_area : s > 0)
  (h_width_positive : w > 0) (h_cross_area : 4 * s * w - 3 * w^2 = 0.49 * s^2) :
  (w^2 / s^2) * 100 = 12.25 :=
by
  sorry

end flag_yellow_area_percentage_l792_792718


namespace vkontakte_solution_l792_792307

variables (M I A P : Prop)

theorem vkontakte_solution
  (h1 : M → I ∧ A)
  (h2 : A ⊕ P)
  (h3 : I ∨ M)
  (h4 : P ↔ I) : 
  ¬M ∧ I ∧ A ∧ P :=
begin
  sorry
end

end vkontakte_solution_l792_792307


namespace worm_domino_partition_coprime_equiv_l792_792374

-- Define the notion of a worm and its conditions
structure Worm where
  path : List (ℕ × ℕ)
  start_at_zero : path.head = (0, 0)
  moves_right_or_up : ∀ (p : ℕ × ℕ), p ∈ path → (∃ (next : ℕ × ℕ), next ∈ path ∧ (next.1 = p.1 + 1 ∨ next.2 = p.2 + 1))

-- Define the notion of partitioning a worm into domino pairs
def domino_partitions (w : Worm) : ℕ := -- The number of ways to partition the worm into dominoes
  sorry

-- Define the coprime function
def coprime_count (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ m => Nat.gcd n m = 1).length

-- The main theorem statement
theorem worm_domino_partition_coprime_equiv (n : ℕ) (h : n > 2) :
  (∃ (w : Worm), domino_partitions w = n) ↔ coprime_count n = n :=
sorry

end worm_domino_partition_coprime_equiv_l792_792374


namespace number_of_integers_satisfying_inequality_l792_792484

def polynomial (n : ℤ) : ℤ := (n - 4) * (n + 2) * (n + 6)

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | -15 ≤ n ∧ n ≤ 7 ∧ polynomial n < -1}.finset.card = 12 :=
by
  sorry

end number_of_integers_satisfying_inequality_l792_792484


namespace kai_birth_year_l792_792147

theorem kai_birth_year (h : 2020 - 25 = 1995) : kai_birth_year = 1995 :=
by
  exact h

end kai_birth_year_l792_792147


namespace charlie_work_l792_792350

def ratio_allen_ben_charlie : ℕ × ℕ × ℕ := (3, 5, 2)
def total_work : ℕ := 300

theorem charlie_work : 
  let total_parts := ratio_allen_ben_charlie.1 + ratio_allen_ben_charlie.2 + ratio_allen_ben_charlie.3 in
  let work_per_part := total_work / total_parts in
  let charlie_parts := ratio_allen_ben_charlie.3 in
  charlie_parts * work_per_part = 60 :=
by
  sorry

end charlie_work_l792_792350


namespace arithmetic_sequence_a4_l792_792827

theorem arithmetic_sequence_a4 (S n : ℕ) (a : ℕ → ℕ) (h1 : S = 28) (h2 : S = 7 * a 4) : a 4 = 4 :=
by sorry

end arithmetic_sequence_a4_l792_792827


namespace range_of_odd_function_l792_792379

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, -f x = f (-x)

def specific_odd_function (f : ℝ → ℝ) : Prop :=
  odd_function f ∧ (∀ x : ℝ, x > 0 → f x = 3)

theorem range_of_odd_function (f : ℝ → ℝ) (h : specific_odd_function f) :
  set.range f = {-3, 0, 3} := 
sorry

end range_of_odd_function_l792_792379


namespace floor_eq_solution_l792_792797

theorem floor_eq_solution (x : ℝ) 
  (h1 : ∀ z : ℝ, ∃ n : ℤ, z = n → ∀ z : ℤ, ∃ m : ℤ, z + 1/3 = m) 
  (h2 : ∀ z : ℝ, ∃ k : ℤ, z = k → ∀ z : ℤ, ∃ p : ℤ, z + 3 = p) 
  (hx : ∀ y : ℝ, ∃ q : ℤ, y = q  → ∀ y : ℤ, ∃ r : ℤ, 3*y = r → ∀ y : ℤ, lf (lf y + 1/3) = lf ( y  + 3)) :
  x ∈ Ico (4/3) (5/3) :=
by
  sorry
  
end floor_eq_solution_l792_792797


namespace max_value_f_l792_792450

theorem max_value_f (x y z : ℝ) (hxyz : x * y * z = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (1 - y * z + z) * (1 - x * z + x) * (1 - x * y + y) ≤ 1 :=
sorry

end max_value_f_l792_792450


namespace age_difference_correct_l792_792527

-- Define the ages of John and his parents based on given conditions
def John_age (father_age : ℕ) : ℕ :=
  father_age / 2

def mother_age (father_age : ℕ) : ℕ :=
  father_age - 4

def age_difference (john_age : ℕ) (mother_age : ℕ) : ℕ :=
  abs (john_age - mother_age)

-- Main theorem stating the age difference between John and his mother
theorem age_difference_correct (father_age : ℕ) (h : father_age = 40) :
  age_difference (John_age father_age) (mother_age father_age) = 16 :=
by
  sorry

end age_difference_correct_l792_792527


namespace no_infinite_sequence_of_positive_integers_l792_792317

theorem no_infinite_sequence_of_positive_integers (a : ℕ → ℕ) (H : ∀ n, a n > 0) :
  ¬(∀ n, (a (n+1))^2 ≥ 2 * (a n) * (a (n+2))) :=
sorry

end no_infinite_sequence_of_positive_integers_l792_792317


namespace sum_of_roots_sum_of_values_l792_792289

theorem sum_of_roots (x : ℝ) : x^2 - 5 * x + 1 = 16 → x^2 - 5 * x - 15 = 0 :=
by
  intro h
  rw [←eq_sub_zero, ←sub_eq_add_neg] at h
  simp at h
  exact h

theorem sum_of_values {x : ℝ} :
  (∃ x, x^2 - 5 * x + 1 = 16) → (∃ x, x^2 - 5 * x - 15 = 0) → (∑ x in {x | x^2 - 5 * x - 15 = 0}, x) = 5 :=
by
  intro hx1 hx2
  have h_sum := sum_of_roots
  sorry

end sum_of_roots_sum_of_values_l792_792289


namespace no_positive_integer_solutions_l792_792073

def f (x : ℤ) : ℤ := x^2 + x

theorem no_positive_integer_solutions 
    (a b : ℤ) (ha : 0 < a) (hb : 0 < b) : 4 * f a ≠ f b := by
  sorry

end no_positive_integer_solutions_l792_792073


namespace sum_of_five_integers_l792_792666

-- Definitions of the five integers based on the conditions given in the problem
def a := 12345
def b := 23451
def c := 34512
def d := 45123
def e := 51234

-- Statement of the proof problem
theorem sum_of_five_integers :
  a + b + c + d + e = 166665 :=
by
  -- The proof is omitted
  sorry

end sum_of_five_integers_l792_792666


namespace number_of_positive_area_triangles_l792_792064

theorem number_of_positive_area_triangles :
  let points := ({(i, j) | i j : ℕ, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 }).card in
  ∃ triangles : ℕ, 
  (triangles = points.choose 3 - (6 * points.choose 3 - 6 * points.choose 3) -
              2 * (2 * points.choose 3 + 4 * points.choose 3 + 4 * points.choose 3 + 4 * points.choose 3)) 
  ∧ triangles = 6700 :=
by
  sorry

end number_of_positive_area_triangles_l792_792064


namespace magnitude_of_vector_l792_792480

noncomputable def vector_a (x : ℝ) := (1 : ℝ, x)
noncomputable def vector_b (x : ℝ) := (1 : ℝ, x - 1)
noncomputable def vector_c (x : ℝ) := (vector_a x).1 - 2 * (vector_b x).1, (vector_a x).2 - 2 * (vector_b x).2

theorem magnitude_of_vector :
  ∀ (x : ℝ), ((vector_c x).1 * (vector_a x).1 + (vector_c x).2 * (vector_a x).2 = 0) →
  |(vector_c x)| = real.sqrt 2 :=
by
  intros x h
  -- Proof goes here
  sorry

end magnitude_of_vector_l792_792480


namespace cos_gamma_of_point_l792_792153

noncomputable theory
open Real

def point (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (prime_sum : Nat.Prime (x + y + z)) : Prop :=
    (cos (arccos (x / sqrt (x^2 + y^2 + z^2))) = 1/4) ∧
    (cos (arccos (y / sqrt (x^2 + y^2 + z^2))) = 2/5)

theorem cos_gamma_of_point (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
    (h_prime : Nat.Prime (x + y + z))
    (h_cos_alpha : cos (arccos (x / sqrt (x^2 + y^2 + z^2))) = 1/4)
    (h_cos_beta : cos (arccos (y / sqrt (x^2 + y^2 + z^2))) = 2/5) :
    cos (arccos (z / sqrt (x^2 + y^2 + z^2))) = sqrt 311 / 20 :=
begin
  sorry
end

end cos_gamma_of_point_l792_792153


namespace common_divisors_9240_8820_l792_792892

-- Define the prime factorizations given in the problem.
def pf_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def pf_8820 := [(2, 2), (3, 2), (5, 1), (7, 2)]

-- Define a function to calculate the gcd of two numbers given their prime factorizations.
def gcd_factorizations (pf1 pf2 : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
    List.filterMap (λ ⟨p, e1⟩ =>
      match List.lookup p pf2 with
      | some e2 => some (p, min e1 e2)
      | none => none
      end) pf1 

-- Define a function to compute the number of divisors from the prime factorization.
def num_divisors (pf: List (ℕ × ℕ)) : ℕ :=
    pf.foldl (λ acc ⟨_, e⟩ => acc * (e + 1)) 1

-- The Lean statement for the problem
theorem common_divisors_9240_8820 : 
    num_divisors (gcd_factorizations pf_9240 pf_8820) = 24 :=
by
    -- The proof goes here. We include sorry to indicate that the proof is omitted.
    sorry

end common_divisors_9240_8820_l792_792892


namespace arithmetic_sequence_properties_l792_792132

noncomputable def common_difference (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1) * d

theorem arithmetic_sequence_properties :
  ∃ d : ℚ, d = 5 / 9 ∧ ∃ S : ℚ, S = -29 / 3 ∧
  ∀ n : ℕ, ∃ a₁ a₅ a₈ : ℚ, a₁ = -3 ∧
    a₅ = common_difference a₁ d 5 ∧
    a₈ = common_difference a₁ d 8 ∧ 
    11 * a₅ = 5 * a₈ - 13 ∧
    S = (n / 2) * (2 * a₁ + (n - 1) * d) ∧
    n = 6 := 
sorry

end arithmetic_sequence_properties_l792_792132


namespace polynomial_degree_is_4_l792_792276

noncomputable def f (x : ℝ) : ℝ := 3 + 6*x^2 + 200 + 7*real.pi*x^4 + real.exp(x^3) + 15

theorem polynomial_degree_is_4 : polynomial.degree (3 + 6*x^2 + 200 + 7*real.pi*x^4 + real.exp(x^3) + 15) = 4 := sorry

end polynomial_degree_is_4_l792_792276


namespace petya_wins_optimal_play_l792_792252

theorem petya_wins_optimal_play : 
  ∃ n : Nat, n = 2021 ∧
  (∀ move : Nat → Nat, 
    (move = (λ n, n - 1) ∨ move = (λ n, n - 2)) → 
    (n % 3 ≡ 0) ↔ 
      (∃ optimal_move : Nat → Nat, 
        optimal_move = (λ n, n - 2 + 3) 
          → (optimal_move n % 3 = 0))) -> 
  Petya wins := 
  sorry

end petya_wins_optimal_play_l792_792252


namespace eq_is_quadratic_iff_m_zero_l792_792077

theorem eq_is_quadratic_iff_m_zero (m : ℝ) : (|m| + 2 = 2 ∧ m - 3 ≠ 0) ↔ m = 0 := by
  sorry

end eq_is_quadratic_iff_m_zero_l792_792077


namespace counting_valid_C_values_l792_792409

-- Define the condition that 6 + C is divisible by 3
def divisible_by_3 (n : ℤ) := ∃ k : ℤ, n = 3 * k

-- Define the main theorem statement
theorem counting_valid_C_values :
  {C : ℕ | C ≤ 9 ∧ divisible_by_3 (6 + C)}.finite.card = 4 :=
by
  sorry

end counting_valid_C_values_l792_792409


namespace sum_of_extrema_equal_4_l792_792637

-- Definition of the function
noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log2 (x + 1)

-- The problem statement
theorem sum_of_extrema_equal_4 : 
  let a := interval_integral.min {x : ℝ | 0 <= x ∧ x <= 1} f;
  let b := interval_integral.max {x : ℝ | 0 <= x ∧ x <= 1} f;
  (a + b) = 4 :=
sorry

end sum_of_extrema_equal_4_l792_792637


namespace area_of_triangle_ABC_is_28_l792_792236

-- Define points A, B, and C
def Point := (ℝ × ℝ)
def A : Point := (3, 4)
def B : Point := (A.1, -A.2)
def C : Point := (-B.2, B.1)

-- Define the distance function
def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the area of triangle function using the base and height
def area_of_triangle (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

-- Define the key values
def base : ℝ := dist A B
def height : ℝ := Real.abs (C.1 - A.1)

-- Statement of the problem to prove the area of triangle ABC
theorem area_of_triangle_ABC_is_28 : area_of_triangle base height = 28 := 
  sorry

end area_of_triangle_ABC_is_28_l792_792236


namespace f_neg_three_l792_792026

-- Defining that f is an odd function and specifying its form for positive x
def f (x : ℝ) : ℝ :=
if x > 0 then x * (x - 1) else if x < 0 then -((-x) * (-x - 1)) else 0

theorem f_neg_three : f (-3) = -6 :=
by
  sorry

end f_neg_three_l792_792026


namespace number_of_true_props_l792_792155

variables (α β γ : Plane) (l m n : Line)

-- Proposition definitions
def Prop1 : Prop := (m ⊆ α ∧ n ⊆ β ∧ α ⊥ β) → m ⊥ n
def Prop2 : Prop := (m ⊥ α ∧ n ∥ β ∧ α ∥ β) → m ⊥ n
def Prop3 : Prop := (α ∥ β ∧ l ⊆ α) → l ∥ β
def Prop4 : Prop := (α ∩ β = l ∧ β ∩ γ = m ∧ γ ∩ α = n ∧ l ∥ γ) → m ∥ n

-- Main theorem to prove the number of true propositions
theorem number_of_true_props : Prop :=
  (¬Prop1) ∧ Prop2 ∧ Prop3 ∧ Prop4 → 3

-- Adding sorry to skip the actual proof
example : number_of_true_props α β γ l m n := sorry

end number_of_true_props_l792_792155


namespace find_a_2012_l792_792171

noncomputable def recurrence_a : ℕ → ℝ
| 0       := -2
| (n + 1) := recurrence_a n + recurrence_b n + real.sqrt ((recurrence_a n)^2 + (recurrence_b n)^2)
and recurrence_b : ℕ → ℝ
| 0       := 1
| (n + 1) := recurrence_a n + recurrence_b n - real.sqrt ((recurrence_a n)^2 + (recurrence_b n)^2)

theorem find_a_2012 : recurrence_a 2012 = 2^1006 * real.sqrt (2^2010 + 2) - 2^2011 :=
by
  sorry

end find_a_2012_l792_792171


namespace segment_parametrization_pqrs_l792_792622

theorem segment_parametrization_pqrs :
  ∃ (p q r s : ℤ), 
    q = 1 ∧ 
    s = -3 ∧ 
    p + q = 6 ∧ 
    r + s = 4 ∧ 
    p^2 + q^2 + r^2 + s^2 = 84 :=
by
  use 5, 1, 7, -3
  sorry

end segment_parametrization_pqrs_l792_792622


namespace tan_A_in_right_triangle_l792_792966

theorem tan_A_in_right_triangle 
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (angle_C_right : ∠ C = 90°) 
  (AB_len : dist A B = 13) 
  (BC_len : dist B C = 2 * Real.sqrt 10) : 
  tan ∠ A = (2 * Real.sqrt 1290) / 129 :=
sorry

end tan_A_in_right_triangle_l792_792966


namespace PQ_common_point_l792_792122

-- Define a triangle ABC with angles and vertices
variables {A B C : Point}
variable (ABC : Triangle A B C)

-- Define angle conditions
axiom angle_condition : angle B > angle C

-- Define the points D and E: internal and external bisectors of angle A intersecting BC
variable (D : Point) (E : Point)

axiom D_internal_bisector : is_internal_bisector (angle A) D
axiom E_external_bisector : is_external_bisector (angle A) E

-- Define the variable point P on EA
variable (P : Point)
axiom A_on_EP : is_on_line_segment A E P

-- Define the points M and Q
variable (M : Point) (Q : Point)
axiom DP_intersects_AC_at_M : ∃ M, is_intersection_point (line D P) (line A C) M
axiom ME_intersects_AD_at_Q : ∃ Q, is_intersection_point (line M E) (line A D) Q

-- Prove that all lines PQ have a common point
theorem PQ_common_point : ∃ F, ∀ (P : Point), ∃ Q, is_on_line_segment P Q F :=
sorry  -- proof omitted

end PQ_common_point_l792_792122


namespace sum_of_solutions_eq_zero_l792_792504

theorem sum_of_solutions_eq_zero (x y : ℝ) (h1 : y = 9) (h2 : x^2 + y^2 = 169) : (x = real.sqrt 88 ∨ x = -real.sqrt 88) → x + (-x) = 0 :=
by
  intro h
  cases h
  · rw [h]
    simp,
  · rw [h]
    simp

end sum_of_solutions_eq_zero_l792_792504


namespace parallel_line_slope_l792_792663

theorem parallel_line_slope (x y : ℝ) : 
  (∃ b : ℝ, 3 * x + 6 * y = 15 → y = -1/2 * x + b) → slope_parallel :ℝ :=
sorry

end parallel_line_slope_l792_792663


namespace max_f_in_interval_sinC_over_sinA_l792_792868

def f (x : ℝ) : ℝ := sin (2 * x) + sin x * cos x

theorem max_f_in_interval :
  ∃ x : ℝ, x ∈ Ioo (0 : ℝ) (π / 2) ∧ f x = sqrt (3 / 2) := sorry

theorem sinC_over_sinA (A B C : ℝ) (hA_lt_B : A < B)
  (hA_in_triangle : 0 < A ∧ A < π)
  (hB_in_triangle : 0 < B ∧ B < π)
  (hC_in_triangle : 0 < C ∧ C < π)
  (h_angles : A + B + C = π)
  (h_fA : f A = sqrt (2) / 2)
  (h_fB : f B = sqrt (2) / 2) :
  sin C / sin A = 1 / (sin (π / 3) * cos (π / 4) - cos (π / 3) * sin (π / 4)) := sorry

end max_f_in_interval_sinC_over_sinA_l792_792868


namespace common_divisors_9240_8820_l792_792903

def prime_factors_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def prime_factors_8820 := [(2, 2), (3, 2), (5, 1), (7, 1), (11, 1)]

def gcd_prime_factors := [(2, 2), (3, 1), (5, 1), (7, 1), (11, 1)]

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.snd + 1)) 1

theorem common_divisors_9240_8820 :
  num_divisors gcd_prime_factors = 32 := by
  sorry

end common_divisors_9240_8820_l792_792903


namespace weight_of_fourth_dog_l792_792250

theorem weight_of_fourth_dog (y x : ℝ) : 
  (25 + 31 + 35 + x) / 4 = (25 + 31 + 35 + x + y) / 5 → 
  x = -91 - 5 * y :=
by
  sorry

end weight_of_fourth_dog_l792_792250


namespace bird_wings_l792_792974

theorem bird_wings (birds wings_per_bird : ℕ) (h1 : birds = 13) (h2 : wings_per_bird = 2) : birds * wings_per_bird = 26 := by
  sorry

end bird_wings_l792_792974


namespace sum_f_1_to_2009_l792_792457

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom f_one : f 1 = 1
axiom even_shifted_function (f : ℝ → ℝ) : ∀ x, f (x - 1) = f (1 - x)

theorem sum_f_1_to_2009 : (finset.sum (finset.range 2009) (λ n, f (↑n + 1))) = 1 := 
  sorry

end sum_f_1_to_2009_l792_792457


namespace count_valid_pairs_l792_792804

def contains_zero_digit (n : ℕ) : Prop := 
  ∃ k, 10^k ≤ n ∧ n < 10^(k+1) ∧ (n / 10^k % 10 = 0)

def valid_pair (a b : ℕ) : Prop := 
  a + b = 500 ∧ ¬contains_zero_digit a ∧ ¬contains_zero_digit b

theorem count_valid_pairs : 
  (Finset.card (Finset.filter (λ p : ℕ × ℕ, valid_pair p.1 p.2) 
    (Finset.product (Finset.range 500) (Finset.range 500)))) = 329 := 
sorry

end count_valid_pairs_l792_792804


namespace coin_tosses_l792_792671

theorem coin_tosses (n : ℤ) (h : (1/2 : ℝ)^n = 0.125) : n = 3 :=
by
  sorry

end coin_tosses_l792_792671


namespace triangle_area_proof_l792_792512

noncomputable def area_triangle_ABC_equal_32_div_3 
  (AB BC BD AC BE : ℝ)
  (AB_eq_BC : AB = BC)
  (BD_is_altitude : BD * AC = AB * AC / 2)
  (BE_val : BE = 8)
  (tan_geometric_prog : ∃ (α β : ℝ), tan(α - β) * tan(α + β) = tan(α)^2)
  (cot_arithmetic_prog : ∃ (b a : ℝ), (b + a) / (b - a) = 1 ∧ b / a = 1) 
  : ℝ :=
  let a := (4*sqrt 2) / 3 in
  let b := 4 * sqrt 2 in
  a * b

theorem triangle_area_proof 
  (AB BC BD AC BE : ℝ)
  (AB_eq_BC : AB = BC)
  (BD_is_altitude : BD * AC = AB * AC / 2)
  (BE_val : BE = 8)
  (tan_geometric_prog : ∃ (α β : ℝ), tan(α - β) * tan(α + β) = tan(α)^2)
  (cot_arithmetic_prog : ∃ (b a : ℝ), (b + a) / (b - a) = 1 ∧ b / a = 1) 
  : area_triangle_ABC_equal_32_div_3 AB BC BD AC BE AB_eq_BC BD_is_altitude BE_val tan_geometric_prog cot_arithmetic_prog = 32 / 3 :=
  sorry

end triangle_area_proof_l792_792512


namespace range_of_x_l792_792916

theorem range_of_x (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -2) :
  x > 1/3 ∨ x < -1/2 :=
sorry

end range_of_x_l792_792916


namespace volume_of_R_l792_792149

def box := { x : ℝ × ℝ × ℝ // 0 ≤ x.1 ∧ x.1 ≤ 1 ∧ 0 ≤ x.2 ∧ x.2 ≤ 2 ∧ 0 ≤ x.3 ∧ x.3 ≤ 4 }

def is_within_distance_3 (p q : ℝ × ℝ × ℝ) : Prop :=
  (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2 ≤ 9

def R : set (ℝ × ℝ × ℝ) :=
  { x | ∃ y : ℝ × ℝ × ℝ, y ∈ box ∧ is_within_distance_3 x y }

theorem volume_of_R :
  volume R = 92 + 144 * real.pi :=
sorry

end volume_of_R_l792_792149


namespace smallest_sum_of_exterior_angles_l792_792642

open Real

theorem smallest_sum_of_exterior_angles 
  (p q r : ℕ) 
  (hp : p > 2) 
  (hq : q > 2) 
  (hr : r > 2) 
  (hpq : p ≠ q) 
  (hqr : q ≠ r) 
  (hrp : r ≠ p) 
  : (360 / p + 360 / q + 360 / r) ≥ 282 ∧ 
    (360 / p + 360 / q + 360 / r) = 282 → 
    360 / p = 120 ∧ 360 / q = 90 ∧ 360 / r = 72 := 
sorry

end smallest_sum_of_exterior_angles_l792_792642


namespace sec_minus_tan_l792_792490

theorem sec_minus_tan (x : ℝ) (h : real.sec x + real.tan x = 7) : real.sec x - real.tan x = 1 / 7 :=
by
  sorry

end sec_minus_tan_l792_792490


namespace exist_functions_fg_neq_f1f1_g1g1_l792_792316

-- Part (a)
theorem exist_functions_fg :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, (f ∘ g) x = (g ∘ f) x) ∧ 
    (∀ x, (f ∘ f) x = (g ∘ g) x) ∧ 
    (∀ x, f x ≠ g x) := 
sorry

-- Part (b)
theorem neq_f1f1_g1g1 
  (f1 g1 : ℝ → ℝ)
  (H_comm : ∀ x, (f1 ∘ g1) x = (g1 ∘ f1) x)
  (H_neq: ∀ x, f1 x ≠ g1 x) :
  ∀ x, (f1 ∘ f1) x ≠ (g1 ∘ g1) x :=
sorry

end exist_functions_fg_neq_f1f1_g1g1_l792_792316


namespace original_fraction_eq_2_5_l792_792752

theorem original_fraction_eq_2_5 (a b : ℤ) (h : (a + 4) * b = a * (b + 10)) : (a / b) = (2 / 5) := by
  sorry

end original_fraction_eq_2_5_l792_792752


namespace find_valid_pairs_l792_792801

open Nat

def has_zero_digit (n : ℕ) : Prop :=
  ∃ d, 0 ≤ d ∧ d < 10 ∧ (10^d ∣ n) ∧ (n / 10^d) % 10 = 0

def valid_pairs_count (sum : ℕ) : ℕ :=
  (Fin.sum Finset.Ico 1 sum (λ a, if has_zero_digit a ∨ has_zero_digit (sum - a) then 0 else 1))

theorem find_valid_pairs :
  valid_pairs_count 500 = 309 :=
by
  sorry

end find_valid_pairs_l792_792801


namespace no_similarity_proof_condition_B_l792_792095

/-- Given two right-angled triangles, with one angle in triangle ABC being 30 degrees,
show that the condition ∠C = 60 degrees doesn't necessarily prove the similarity with another
triangle A'B'C' which has a right-angle at B', and given the conditions ∠B = ∠B' = 90 degrees, ∠A = 30 degrees, and other angular constraints.

Definitions:
- ∠B: Angle B in triangle ABC
- ∠B': Angle B' in triangle A'B'C'
- ∠A: Angle A in triangle ABC
- ∠C: Angle C in triangle ABC
- ∠A': Angle A' in triangle A'B'C'
- ∠C': Angle C' in triangle A'B'C'

Problem to prove:
  θC (angle C) cannot alone prove the similarity between triangle ABC and triangle A'B'C'.
-/
theorem no_similarity_proof_condition_B :
  (∠B = 90) → (∠B' = 90) → (∠A = 30) → (∠C = 60) →
  ¬(similar_triangle ABC A'B'C')
:= by
  intros hB hB' hA hC
  sorry

end no_similarity_proof_condition_B_l792_792095


namespace general_term_formula_sum_of_first_n_terms_l792_792876

-- Given conditions
variable (a : ℕ → ℕ)
axiom common_ratio : ∀ n, 2 ^ (a (n + 1) - a n) = 2
axiom a4_a3_condition : a 4 + (a 3) ^ 2 = 21
axiom a1_positive : a 1 > 0

-- Part (1): Prove the general term formula
theorem general_term_formula :
  ∀ n, a n = n + 1 := 
sorry

-- Part (2): Prove the sum of the first n terms
def sequence (n : ℕ) : ℝ := 1 / ((2 * (a n) - 1) * (2 * n - 1))
def S_n (n : ℕ) : ℝ := ∑ i in List.range n, sequence a i

theorem sum_of_first_n_terms (n : ℕ) :
  S_n a n = n / (2 * n + 1) :=
sorry

end general_term_formula_sum_of_first_n_terms_l792_792876


namespace min_value_of_quadratic_l792_792280

theorem min_value_of_quadratic :
  ∃ x : ℝ, ∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ 12 :=
by
  let f : ℝ → ℝ := λ x, 4 * x^2 + 8 * x + 16
  use -1
  intros y hy
  have h : f(-1) = 12 := by
    calc
      f(-1) = 4 * (-1)^2 + 8 * (-1) + 16 : by rfl
         ... = 4 * 1 - 8 + 16 : by norm_num
         ... = 4 - 8 + 16 : by rfl
         ... = -4 + 16 : by norm_num
         ... = 12 : by norm_num
  rw [← hy, h]
  have non_neg : 4 * (x + 1)^2 ≥ 0 := by
    apply mul_nonneg
    norm_num
    apply pow_two_nonneg
  exact add_le_add non_neg (by norm_num)

end min_value_of_quadratic_l792_792280


namespace last_integer_in_division_sequence_l792_792634

theorem last_integer_in_division_sequence : 
  ∃ n ∈ {k : ℕ | 2 * k ≤ 2000000 ∧ 2^(nat.ceiling(log 2 (2000000))) / 2^n ≠ 0 ∧ 2^n / 2^(n+1) ∈ ℕ}, n = 15625 :=
by
  sorry

end last_integer_in_division_sequence_l792_792634


namespace arithmetic_sequences_problem_l792_792885

noncomputable def arithmetic_sequences (a b : ℕ → ℕ) (S T : ℕ → ℕ) : Prop :=
  -- Terms sum definitions and given condition
  (∀ n : ℕ, S n = ∑ i in range n, a i) ∧
  (∀ n : ℕ, T n = ∑ i in range n, b i) ∧
  (∀ n : ℕ, (S n)/(T n) = (3 * n + 1) / (n + 3))

theorem arithmetic_sequences_problem (a b : ℕ → ℕ) (S T : ℕ → ℕ) (h : arithmetic_sequences a b S T) :
  (a 2 + a 20)/(b 7 + b 15) = 8/3 := sorry

end arithmetic_sequences_problem_l792_792885


namespace proposition_p_proposition_q_main_proof_l792_792849

theorem proposition_p (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : ∃ x y, x = -1 ∧ y = 1 ∧ y = Real.log a (a * x + 2 * a) :=
by
  use -1, 1
  constructor
  · rfl
  constructor
  · rfl
  simp [Real.log]
  sorry

theorem proposition_q (f : ℝ → ℝ) (h : ∀ x, f (-x - 3) = -f (x - 3)) : ¬(∀ x, f x = f (x - 3) + 3) :=
by
  intro h'
  have h_f_val : ∀ x, f (-x - 3) = -f (x - 3) := h
  specialize h_f_val 0
  simp at h_f_val
  have : f 0 = f (-3) := by
    sorry

  have h'_0 : f 3 = f 0 + 3 := h' 3
  have h_eq : f (-3) = -3 := by
    sorry

  sorry

theorem main_proof (a : ℝ) (h1 : 0 < a) (h2: a ≠ 1) (f : ℝ → ℝ) (h : ∀ x, f (-x - 3) = -f (x - 3)) 
: (∃ x y, x = -1 ∧ y = 1 ∧ y = Real.log a (a * x + 2 * a)) ∧ ¬ (∀ x, f x = f (x - 3) + 3) :=
by
  constructor
  · exact proposition_p a h1 h2
  · exact proposition_q f h

end proposition_p_proposition_q_main_proof_l792_792849


namespace problem1_problem2_l792_792743

-- Problem 1
theorem problem1 :
  (-6 - 1/2) * (4 / 13) - 8 / abs (-4 + 2) = -6 :=
by
  sorry

-- Problem 2
theorem problem2 :
  (-3)^4 / (1.5)^2 - 6 * (-1 / 6) + abs (-3^2 - 9) = 55 :=
by
  sorry

end problem1_problem2_l792_792743


namespace runners_meet_time_l792_792259

theorem runners_meet_time :
  let time_runner_1 := 2
  let time_runner_2 := 4
  let time_runner_3 := 11 / 2
  Nat.lcm time_runner_1 (Nat.lcm time_runner_2 (Nat.lcm (11) 2)) = 44 := by
  sorry

end runners_meet_time_l792_792259


namespace store_owner_uniforms_l792_792344

theorem store_owner_uniforms (U E : ℕ) (h1 : U + 1 = 2 * E) (h2 : U % 2 = 1) : U = 3 := 
sorry

end store_owner_uniforms_l792_792344


namespace average_hidden_primes_l792_792202

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ (p - q = 9 ∨ q - p = 9) 

theorem average_hidden_primes :
  ∃ p q : ℕ, consecutive_primes p q ∧ (18 + p = 27 + q) → ((p + q) / 2 = 15) :=
begin
  sorry
end

end average_hidden_primes_l792_792202


namespace interesting_arithmetic_progression_l792_792332

def interesting (n : ℕ) : Prop := ∃ d : ℕ, ∃ a : ℕ, d = (a + 1) ∧ 2018 ∣ d *(n div a + 1) 

def infinite_arithmetic_progression (a k : ℕ) : Prop :=
  ∀ n, interesting (a + n*k)

theorem interesting_arithmetic_progression (k : ℕ) :   
  (∃ a, infinite_arithmetic_progression a k) ↔
  ∃ (m : ℕ) (p : ℕ), Nat.Prime p ∧ k = m * p^1009 ∧ ¬(k = 2^2009) := sorry

end interesting_arithmetic_progression_l792_792332


namespace math_problem_l792_792160

noncomputable def a : ℝ := Real.pi / 2010

def sum_expression (n : ℕ) : ℝ :=
  2 * ∑ k in Finset.range (n + 1).filter (· > 0), (Real.cos (k^2 * a) * Real.sin (k * a))

lemma smallest_n (n : ℕ) :
  sum_expression n ∈ Int ↔ (∃ m : ℤ, n * (n + 1) / 2010 = m) := sorry

theorem math_problem :
  ∃ n : ℕ, sum_expression n ∈ Int ∧ (∀ m : ℕ, m < n → ¬(sum_expression m ∈ Int)) :=
begin
  use 67,
  split,
  { sorry }, -- This would contain the proof that the sum_expression 67 is an integer.
  { sorry } -- This would contain the proof that there is no smaller integer m for which sum_expression is an integer.
end

end math_problem_l792_792160


namespace intersection_of_P_and_Q_l792_792478
-- Import the entire math library

-- Define the conditions for sets P and Q
def P := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | (x - 1)^2 ≤ 4}

-- Define the theorem to prove that P ∩ Q = {x | 1 ≤ x ∧ x ≤ 3}
theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 ≤ x ∧ x ≤ 3} :=
by
  -- Placeholder for the proof
  sorry

end intersection_of_P_and_Q_l792_792478


namespace counting_valid_C_values_l792_792408

-- Define the condition that 6 + C is divisible by 3
def divisible_by_3 (n : ℤ) := ∃ k : ℤ, n = 3 * k

-- Define the main theorem statement
theorem counting_valid_C_values :
  {C : ℕ | C ≤ 9 ∧ divisible_by_3 (6 + C)}.finite.card = 4 :=
by
  sorry

end counting_valid_C_values_l792_792408


namespace regression_value_l792_792819

theorem regression_value (x : ℝ) (y : ℝ) (h : y = 4.75 * x + 2.57) (hx : x = 28) : y = 135.57 :=
by
  sorry

end regression_value_l792_792819


namespace find_x_l792_792066

theorem find_x (x : ℝ) (h : (1 / Real.log x / Real.log 5 + 1 / Real.log x / Real.log 7 + 1 / Real.log x / Real.log 11) = 1) : x = 385 := 
sorry

end find_x_l792_792066


namespace weight_of_replaced_student_l792_792608

theorem weight_of_replaced_student :
  ∃ W : ℝ, W = 92 :=
by
  -- Define initial conditions
  let new_student_weight : ℝ := 72
  let avg_decrease : ℝ := 4
  let num_students := 5

  -- Define the equation based on given conditions
  let total_decrease := avg_decrease * num_students
  let W := new_student_weight + total_decrease

  -- Conclude that W = 92
  have hW : W = 92 := by norm_num

  -- Provide the final proof statement
  use W
  exact hW

end weight_of_replaced_student_l792_792608


namespace average_cost_price_correct_l792_792203

theorem average_cost_price_correct :
  let CP_A := 600 / 1.60,
      CP_B := 800 / 1.40,
      CP_C := 900 / 1.50 in
  (CP_A + CP_B + CP_C) / 3 = 515.48 :=
by
  /- Definitions and assumption checks -/
  let CP_A := 600 / 1.60
  let CP_B := 800 / 1.40
  let CP_C := 900 / 1.50
  have h1 : CP_A = 375 := by simp [CP_A]
  have h2 : CP_B = 571.43 := by simp [CP_B] -- with floating point consideration
  have h3 : CP_C = 600 := by simp [CP_C]
  /- Calculation of the average cost price -/
  have h_avg : (CP_A + CP_B + CP_C) / 3 = 515.48 := by
    simp [CP_A, CP_B, CP_C, h1, h2, h3]
  exact h_avg
  sorry

end average_cost_price_correct_l792_792203


namespace smallest_angle_in_right_triangle_l792_792934

-- Given conditions
def angle_α := 90 -- The right-angle in degrees
def angle_β := 55 -- The given angle in degrees

-- Goal: Prove that the smallest angle is 35 degrees.
theorem smallest_angle_in_right_triangle (a b c : ℕ) (h1 : a = angle_α) (h2 : b = angle_β) (h3 : c = 180 - a - b) : c = 35 := 
by {
  -- use sorry to skip the proof steps
  sorry
}

end smallest_angle_in_right_triangle_l792_792934


namespace trajectory_equation_existence_of_m_l792_792958

theorem trajectory_equation (x y : ℝ) (h : (Real.sqrt ((x - 1)^2 + y^2)) / |x - 2| = Real.sqrt 2 / 2) : 
    (x^2 / 2 + y^2 = 1) :=
sorry

theorem existence_of_m (P Q M : ℝ × ℝ) (h₁ : P ≠ Q) 
  (hPQ_E : ∀ x y, (x, y) ∈ { p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1 } → (P = (x, y) ∨ Q = (x, y))) 
  (h₂ : 0 ≤ M.1 ∧ M.1 < 1/2 ∧ M.2 = 0)
  (h₃ : ∀ m, ∃ l : ℝ → ℝ, ∀ t, l t = k * (t - 1) * (k ≠ 0 ∧ m = k^2 / (1 + 2 * k^2))) :
  (\overrightarrow{M P} + \overrightarrow{M Q}) \cdot \overrightarrow{P Q} = 0 :=
sorry

end trajectory_equation_existence_of_m_l792_792958


namespace arithmetic_sequence_example_l792_792633

theorem arithmetic_sequence_example 
    (a : ℕ → ℤ) 
    (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) 
    (h2 : a 1 + a 4 + a 7 = 45) 
    (h3 : a 2 + a 5 + a 8 = 39) :
    a 3 + a 6 + a 9 = 33 :=
sorry

end arithmetic_sequence_example_l792_792633


namespace problem_statement_l792_792689

theorem problem_statement :
  (2 * 3 * 4) * (1/2 + 1/3 + 1/4) = 26 := by
  sorry

end problem_statement_l792_792689


namespace swimming_speed_l792_792330

theorem swimming_speed (s t v : ℝ) (h_stream : s = 1) (h_upstream_downstream_time : (v - s) * (2 * t) = (v + s) * t) : v = 2 :=
by
  assume h_stream : s = 1
  assume h_upstream_downstream_time : (v - s) * (2 * t) = (v + s) * t
  sorry

end swimming_speed_l792_792330


namespace range_of_a_l792_792473

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 - a * x + real.exp (-x) + x * real.log x ≥ 0) →
  a ≤ 1 :=
begin
  sorry
end

end range_of_a_l792_792473


namespace configuration_exists_and_unique_angles_condition_l792_792148

-- Given a square ABCD
variables (A B C D M N P: Point) (x: ℝ)

-- Define points and angles based on given conditions
#check 0 ≤ x ∧ x ≤ 22.5

theorem configuration_exists_and_unique (h: 0 ≤ x ∧ x ≤ 22.5):
  (∃ M: Point, ∃ N: Point, ∃ P: Point,
    ∠(AB, AM) = x ∧ ∠(BC, MN) = 2 * x ∧ ∠(CD, NP) = 3 * x ∧ 
    P ∈ segment DA) :=
sorry
  
theorem angles_condition (h: 0 ≤ x ∧ x ≤ 22.5):
  (angle (DA, PB) = 4 * x) :=
sorry

end configuration_exists_and_unique_angles_condition_l792_792148


namespace bus_travel_time_l792_792321

theorem bus_travel_time :
  let departure_time := Time.mk ⟨9, 30⟩
  let arrival_time := Time.mk ⟨12, 30⟩
  (arrival_time - departure_time).hours = 3 :=
by
  let departure_time := Time.mk ⟨9, 30⟩
  let arrival_time := Time.mk ⟨12, 30⟩
  have h : (arrival_time - departure_time).hours = 3 := sorry
  exact h

end bus_travel_time_l792_792321


namespace max_value_9_l792_792912

noncomputable def max_value_unit_vectors (a b c : ℝ^4) : ℝ :=
  ∥ a - b ∥^2 + ∥ a - c ∥^2 + ∥ b - c ∥^2

theorem max_value_9 (a b c : ℝ^4) (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1) :
  max_value_unit_vectors a b c ≤ 9 :=
sorry

end max_value_9_l792_792912


namespace general_equation_l792_792421

theorem general_equation (n : ℤ) : 
    ∀ (a b : ℤ), 
    (a = 2 ∧ b = 6) ∨ (a = 5 ∧ b = 3) ∨ (a = 7 ∧ b = 1) ∨ (a = 10 ∧ b = -2) → 
    (a / (a - 4) + b / (b - 4) = 2) →
    (n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2) :=
by
  intros a b h_cond h_eq
  sorry

end general_equation_l792_792421


namespace tan_of_11pi_over_4_is_neg1_l792_792793

noncomputable def tan_periodic : Real := 2 * Real.pi

theorem tan_of_11pi_over_4_is_neg1 :
  Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Angle normalization using periodicity of tangent function
  have h1 : Real.tan (11 * Real.pi / 4) = Real.tan (11 * Real.pi / 4 - 2 * Real.pi) := 
    by rw [Real.tan_periodic]
  -- Further normalization
  have h2 : 11 * Real.pi / 4 - 2 * Real.pi = 3 * Real.pi / 4 := sorry
  -- Evaluate tangent at the simplified angle
  have h3 : Real.tan (3 * Real.pi / 4) = -Real.tan (Real.pi / 4) := sorry
  -- Known value of tangent at common angle
  have h4 : Real.tan (Real.pi / 4) = 1 := by simpl tan
  rw [h2, h3, h4]
  norm_num

end tan_of_11pi_over_4_is_neg1_l792_792793


namespace probability_first_grade_probability_one_second_grade_max_defective_units_l792_792323

-- Definition of the problem conditions
def conditions :=
  ∃ (units : List (String × String)), 
  units.length = 6 ∧ 
  units.countp (λ x => x.2 = "first-grade") = 3 ∧ 
  units.countp (λ x => x.2 = "second-grade") = 2 ∧ 
  units.countp (λ x => x.2 = "defective") = 1

-- Define the first part of the problem
theorem probability_first_grade (units : List (String × String)) (h : conditions units) : 
  let selected := (units.choose 2)
  (selected.countp (λ x => x.2 = "first-grade") = 2) / (units.choose 2).length = 1 / 5 :=
sorry

theorem probability_one_second_grade (units : List (String × String)) (h : conditions units) : 
  let selected := (units.choose 2)
  (selected.countp (λ x => x.2 = "second-grade") = 1) / (units.choose 2).length = 8 / 15 :=
sorry

-- Define the second part of the problem
theorem max_defective_units (units : List (String × String)) (h : conditions units) :
  (∃ x, x ≤ 6 ∧ 
         (∃ selected = (units.choose 2), 
          (selected.countp (λ x => x.2 = "defective") ≤ 1) / (units.choose 2).length ≥ 4 / 5)) → 
  x = 3 :=
sorry

end probability_first_grade_probability_one_second_grade_max_defective_units_l792_792323


namespace find_SD_l792_792502

noncomputable def rectangle_PQ_SD : ℚ :=
  let PA := 20 in
  let AQ := 25 in
  let QP := 15 in
  let sim_triangle_ratio := (7 : ℚ) / 9 in
  let TP := 12 in
  sim_triangle_ratio * TP

theorem find_SD :
  ∃ (SD : ℚ), 
    let PA := 20 in
    let AQ := 25 in
    let QP := 15 in
    let sim_triangle_ratio := (7 : ℚ) / 9 in
    let TP := 12 in
    SD = sim_triangle_ratio * TP :=
begin
  use 28 / 3,
  rw div_eq_mul_inv,
  norm_num,
  sorry,
end

end find_SD_l792_792502


namespace evaluate_expression_l792_792768

theorem evaluate_expression 
  (d a b c : ℚ)
  (h1 : d = a + 1)
  (h2 : a = b - 3)
  (h3 : b = c + 5)
  (h4 : c = 6)
  (nz1 : d + 3 ≠ 0)
  (nz2 : a + 2 ≠ 0)
  (nz3 : b - 5 ≠ 0)
  (nz4 : c + 7 ≠ 0) :
  (d + 5) / (d + 3) * (a + 3) / (a + 2) * (b - 3) / (b - 5) * (c + 10) / (c + 7) = 1232 / 585 :=
sorry

end evaluate_expression_l792_792768


namespace duckweed_quarter_covered_l792_792711

theorem duckweed_quarter_covered (N : ℕ) (h1 : N = 64) (h2 : ∀ n : ℕ, n < N → (n + 1 < N) → ∃ k, k = n + 1) :
  N - 2 = 62 :=
by
  sorry

end duckweed_quarter_covered_l792_792711


namespace eq_ellipse_origin_outside_circle_slope_range_l792_792436

variable {a b c : ℝ}
variable (h_passes : ∃ (b : ℝ), b = 1)
variable (h_major_axis_focal_length : 2 * a = 2 * Real.sqrt 2 * c)
variable (h_focal_length : a² - c² = b²)

theorem eq_ellipse : 
  h_passes → h_major_axis_focal_length → h_focal_length → a = Real.sqrt 2 → c = 1 → (∀ x y, (x^2) / 2 + y^2 = 1) := 
  sorry

variable {F : Point}
variable {A B : Point}
variable {O : Point}

theorem origin_outside_circle (h_line_AB_perpendicular: F.x = -1): 
  ∃ r : ℝ, r = Real.sqrt 2 / 2 → (Real.sqrt (1 - (-1)^2) > r) :=
  sorry

theorem slope_range (h_point_inside: True): 
  ∀ k : ℝ, -Real.sqrt 2 < k ∧ k < Real.sqrt 2 :=
  sorry

end eq_ellipse_origin_outside_circle_slope_range_l792_792436


namespace ratio_perimeter_area_equilateral_triangle_l792_792285

theorem ratio_perimeter_area_equilateral_triangle (s : ℝ) (h_s : s = 6):
  let A := (s * s * Real.sqrt 3) / 4 in
  let P := 3 * s in
  P / A = 2 * Real.sqrt 3 / 3 :=
by
  -- The proof is omitted
  sorry

end ratio_perimeter_area_equilateral_triangle_l792_792285


namespace proposition_2_proposition_4_l792_792989

variable {m n : Line}
variable {α β : Plane}

-- Define predicates for perpendicularity, parallelism, and containment
axiom line_parallel_plane (n : Line) (α : Plane) : Prop
axiom line_perp_plane (n : Line) (α : Plane) : Prop
axiom plane_perp_plane (α β : Plane) : Prop
axiom line_in_plane (m : Line) (β : Plane) : Prop

-- State the correct propositions
theorem proposition_2 (m n : Line) (α β : Plane)
  (h1 : line_perp_plane m n)
  (h2 : line_perp_plane n α)
  (h3 : line_perp_plane m β) :
  plane_perp_plane α β := sorry

theorem proposition_4 (n : Line) (α β : Plane)
  (h1 : line_perp_plane n β)
  (h2 : plane_perp_plane α β) :
  line_parallel_plane n α ∨ line_in_plane n α := sorry

end proposition_2_proposition_4_l792_792989


namespace tony_squat_capacity_l792_792651

theorem tony_squat_capacity :
  let curl_weight := 90 in
  let military_press_weight := 2 * curl_weight in
  let total_squat_weight := 5 * military_press_weight in
  total_squat_weight = 900 := by
  sorry

end tony_squat_capacity_l792_792651


namespace evaluate_expression_l792_792387

theorem evaluate_expression (x a b : ℝ) (hx : x ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) :
  (x^3 * a^(-3) - x^3 * b^(-3)) / (x * a^(-1) - x * b^(-1)) = x^5 * (a^(-2) + a^(-1) * b^(-1) + b^(-2)) :=
by
  sorry

end evaluate_expression_l792_792387


namespace disjoint_quadrilaterals_504_l792_792830

theorem disjoint_quadrilaterals_504 (points : Fin 2016 → ℝ × ℝ)
  (no_three_collinear : ∀ (i j k : Fin 2016), 
    i ≠ j → j ≠ k → i ≠ k → 
    let p_i := points i in
    let p_j := points j in
    let p_k := points k in
    ¬ collinear p_i p_j p_k) :
  ∃ (quadrilaterals : Fin 504 → Fin 4 → Fin 2016),
  (∀ i j : Fin 504, i ≠ j → ∀ a b : Fin 4, quadrilaterals i a ≠ quadrilaterals j b) := 
sorry

end disjoint_quadrilaterals_504_l792_792830


namespace find_n_l792_792667

theorem find_n (n : ℝ) : (10:ℝ)^n = 10^(-8) * real.sqrt (10^95 / 0.01) → n = 40.5 :=
by
  intro h
  sorry

end find_n_l792_792667


namespace mix_alcohol_solutions_l792_792210

-- Definitions capturing the conditions from part (a)
def volume_solution_y : ℝ := 600
def percent_alcohol_x : ℝ := 0.1
def percent_alcohol_y : ℝ := 0.3
def desired_percent_alcohol : ℝ := 0.25

-- The resulting Lean statement to prove question == answer given conditions
theorem mix_alcohol_solutions (Vx : ℝ) (h : (percent_alcohol_x * Vx + percent_alcohol_y * volume_solution_y) / (Vx + volume_solution_y) = desired_percent_alcohol) : Vx = 200 :=
sorry

end mix_alcohol_solutions_l792_792210


namespace age_difference_l792_792528

theorem age_difference (john_age father_age mother_age : ℕ) 
    (h1 : john_age * 2 = father_age) 
    (h2 : father_age = mother_age + 4) 
    (h3 : father_age = 40) :
    mother_age - john_age = 16 :=
by
  sorry

end age_difference_l792_792528


namespace total_pieces_of_paper_l792_792193

theorem total_pieces_of_paper (Olivia_paper : ℕ) (Edward_paper : ℕ) (Sam_paper : ℕ) :
  Olivia_paper = 127 → Edward_paper = 345 → Sam_paper = 518 → Olivia_paper + Edward_paper + Sam_paper = 990 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end total_pieces_of_paper_l792_792193


namespace part1_part2_l792_792469

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (b * x) / (Real.log x) - a * x

noncomputable def f_prime (x : ℝ) (a b : ℝ) : ℝ :=
  (b * (Real.log x - 1)) / (Real.log x ^ 2) - a

theorem part1 (a b : ℝ) :
  f_prime (Real.exp 2) a b = -3 / 4 ∧ f (Real.exp 2) a b = -(1 / 2) * (Real.exp 2)
  → a = 1 ∧ b = 1 :=
by
  -- The proof steps would go here.
  sorry

theorem part2 (a : ℝ) :
  ∃ x1 x2 ∈ Icc (Real.exp 1) (Real.exp 2),
    f x1 a 1 ≤ f_prime x2 a 1 + a
    → a >= 1 / 2 - 1 / (4 * Real.exp 2) :=
by
  -- The proof steps would go here.
  sorry

end part1_part2_l792_792469


namespace playoff_requirement_met_l792_792118

/-- In a series of basketball games, the Panthers won 3 out of the first 5 games against the Jaguars.
To qualify for the playoffs, the Jaguars need to win at least 80% of the total games played this season,
including the first 5 games. Assuming the Jaguars won 75% of the additional games played after the
first 5, prove the minimum number of games that need to have been played in total is 45.
-/
theorem playoff_requirement_met : ∃ N : ℕ, N ≥ 45 ∧
  let games_won := 2 + 3/4 * (N - 5) in
  (games_won / N) ≥ 0.8 :=
by sorry

end playoff_requirement_met_l792_792118


namespace find_year_after_2020_l792_792105

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_year_after_2020 :
  ∃ y : ℕ, 2020 < y ∧ sum_of_digits y = 4 ∧ (∀ z : ℕ, 2020 < z ∧ sum_of_digits z = 4 → y ≤ z) := 
begin
  sorry,
end

end find_year_after_2020_l792_792105


namespace max_two_digit_decimals_l792_792657

-- Definitions of the given conditions
def digits : Set ℕ := {2, 0, 5}
def two_digit_decimals : Set ℚ := {x | ∃ d1 d2 ∈ digits, x = d1 / 10 + d2 / 100}

def max_decimals : ℕ := 6

-- The proof problem statement
theorem max_two_digit_decimals : ∀ x ∈ two_digit_decimals, 2 * set.card two_digit_decimals = max_decimals :=
by
  sorry

end max_two_digit_decimals_l792_792657


namespace multiple_of_9_digit_l792_792415

theorem multiple_of_9_digit :
  ∃ d : ℕ, d < 10 ∧ (5 + 6 + 7 + 8 + d) % 9 = 0 ∧ d = 1 :=
by
  sorry

end multiple_of_9_digit_l792_792415


namespace ratio_of_speeds_l792_792607

theorem ratio_of_speeds (d_trac t_trac d_car t_car : ℕ) 
  (speed_ratio : ℕ → ℕ → ℕ)
  (h1 : d_trac = 575) (h2 : t_trac = 23) 
  (h3 : d_car = 360) (h4 : t_car = 4)
  (h5 : ∀ s_trac, speed_ratio 2 s_trac = 2 * s_trac) :
  let speed_trac := d_trac / t_trac in
  let speed_bike := speed_ratio 2 speed_trac in
  let speed_car := d_car / t_car in
  speed_car / speed_bike = 9 / 5 :=
by
  -- Here we will actually provide the proof hypothesis only
  sorry

end ratio_of_speeds_l792_792607


namespace marble_problem_solution_l792_792694

noncomputable def probability_two_marbles (red_marble_initial white_marble_initial total_drawn : ℕ) : ℚ :=
  let total_initial := red_marble_initial + white_marble_initial
  let probability_first_white := (white_marble_initial : ℚ) / total_initial
  let red_marble_after_first_draw := red_marble_initial
  let total_after_first_draw := total_initial - 1
  let probability_second_red := (red_marble_after_first_draw : ℚ) / total_after_first_draw
  probability_first_white * probability_second_red

theorem marble_problem_solution :
  probability_two_marbles 4 6 2 = 4 / 15 := by
  sorry

end marble_problem_solution_l792_792694


namespace range_of_m_l792_792618

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 2 < x ∧ x < 4 ∧ log 2 x + x^2 + m = 0) ↔ m ∈ Ioo (-18 : ℝ) (-5) :=
by
  sorry

end range_of_m_l792_792618


namespace palindrome_divisible_by_11_l792_792709

theorem palindrome_divisible_by_11 :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 
                   (2 * a - 2 * b + c) % 11 = 0 ∧ 
                   ∀ n, n = 900) →
  (𝔹∃ favorable : ℕ, favorable = ( / 11)) :=
begin 
  sorry
end

end palindrome_divisible_by_11_l792_792709


namespace tangent_line_eq_extreme_values_a1_range_of_a_l792_792470

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1)

-- 1. Prove the equation of the tangent line at (1, f(1))
theorem tangent_line_eq (a : ℝ) : 
  let f1 := f 1 a in
  f1 = 0 ∧ (∀ x : ℝ, f x a = Real.log x - a * (x - 1)) → 
  (∀ x : ℝ, (1 - a) * (x - 1) = 0 → y = (1 - a) * (x - 1)) :=
sorry

-- 2. Find extreme points and values when a = 1
theorem extreme_values_a1 :
  (∀ x : ℝ, f x 1 = Real.log x - x + 1) →
  (∃ x0 : ℝ, f x0 1 = 0 ∧ (∀ x : ℝ, x ≠ x0 → f x 1 ≠ 0)) :=
sorry

-- 3. Range of a for which f(x) ≤ ln(x)/(x + 1) holds for x ≥ 1.
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f x a ≤ Real.log x / (x + 1)) ↔ a ∈ (Set.Ici (1 / 2)) :=
sorry

end tangent_line_eq_extreme_values_a1_range_of_a_l792_792470


namespace x_coordinate_equidistant_l792_792756

-- Defining the points
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (2, 6)

-- Distance formula on 2D plane
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Definition of the point on x-axis
def point_on_x_axis (x : ℝ) : ℝ × ℝ :=
  (x, 0)

-- The statement of the theorem
theorem x_coordinate_equidistant :
  ∃ x : ℝ, distance (point_on_x_axis x) A = distance (point_on_x_axis x) B ∧ x = 2 :=
begin
  sorry
end

end x_coordinate_equidistant_l792_792756


namespace solve_for_x_l792_792595

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 8) * x = 12) : x = 168 := by
  sorry

end solve_for_x_l792_792595


namespace correct_transformation_l792_792731

theorem correct_transformation : 
  ∀ y : ℝ, 
    (|y + 1| / 2 = |y| / 3 - |3y - 1| / 6 - y) → 
    (3 * y + 3 = 2 * y - 3 * y + 1 - 6 * y) := 
by 
  intros y h 
  sorry

end correct_transformation_l792_792731


namespace cyclist_C_speed_l792_792266

variables (c d : ℕ) -- Speeds of cyclists C and D in mph
variables (d_eq : d = c + 6) -- Cyclist D travels 6 mph faster than cyclist C
variables (h1 : 80 = 65 + 15) -- Total distance from X to Y and back to the meet point
variables (same_time : 65 / c = 95 / d) -- Equating the travel times of both cyclists

theorem cyclist_C_speed : c = 13 :=
by
  sorry -- Proof is omitted

end cyclist_C_speed_l792_792266


namespace Alyssa_number_of_quarters_l792_792728

def value_penny : ℝ := 0.01
def value_quarter : ℝ := 0.25
def num_pennies : ℕ := 7
def total_money : ℝ := 3.07

def num_quarters (q : ℕ) : Prop :=
  total_money - (num_pennies * value_penny) = q * value_quarter

theorem Alyssa_number_of_quarters : ∃ q : ℕ, num_quarters q ∧ q = 12 :=
by
  sorry

end Alyssa_number_of_quarters_l792_792728


namespace min_distance_circle_parabola_l792_792983

theorem min_distance_circle_parabola :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 - 10 * p.1 + 20 = 0}
  let parabola := {p : ℝ × ℝ | p.2^2 = 8 * p.1}
  ∃ (A B : ℝ × ℝ), A ∈ circle ∧ B ∈ parabola ∧ 
  (∀ (A' B' : ℝ × ℝ), A' ∈ circle → B' ∈ parabola → dist A B ≤ dist A' B') ∧ 
  dist A B = real.sqrt (21 - 8 * real.sqrt 5) := 
by
  -- Insert proof here
  sorry

end min_distance_circle_parabola_l792_792983


namespace average_age_of_boys_l792_792116

theorem average_age_of_boys 
  (total_students : ℕ) 
  (average_age_girls : ℚ) 
  (average_age_school : ℚ) 
  (number_of_girls : ℕ) 
  (number_of_boys := total_students - number_of_girls) : 
  average_age_boys ≈ 11.94 := 
by
  -- Define the known values
  let total_students := 604
  let average_age_girls := 11
  let average_age_school := 11.75
  let number_of_girls := 151
  let number_of_boys := total_students - number_of_girls

  -- Number of Boys
  have total_students_eq : number_of_boys = 453 := rfl

  -- Total Age Calculations
  have total_age_school : ℚ := average_age_school * total_students
  have total_age_girls : ℚ := average_age_girls * number_of_girls
  have total_age_boys := total_age_school - total_age_girls

  -- Conclusion
  have average_age_boys : ℚ := total_age_boys / number_of_boys

  -- Prove the approximation
  suffices (average_age_boys - 11.94).abs < 0.01 by sorry
  sorry

end average_age_of_boys_l792_792116


namespace find_a_value_l792_792562

noncomputable def find_a (A B C : (ℝ × ℝ)) (area_ratio : ℝ) : ℝ := 
  let (Ax, Ay) := A
  let (Bx, By) := B
  let (Cx, Cy) := C
  let total_area := (1 / 2) * (Cx - Bx) * (Ay - By)
  let right_area := total_area / (2 + 1)
  let eq := λ (a : ℝ), (1 / 2) * (Cx - a) * (Ay - (Ay - (Ay / Cx) * a)) = right_area
  a

theorem find_a_value :
  find_a (0, 2) (0, 0) (10, 0) 2 = 5.66 :=
by
  sorry

end find_a_value_l792_792562


namespace points_concyclic_l792_792012

noncomputable def acute_triangle (A B C : Point) : Prop :=
  ∠ A + ∠ B + ∠ C = 180° ∧
  ∠ A < 90° ∧ ∠ B < 90° ∧ ∠ C < 90°

noncomputable def orthocenter (A B C H : Point) : Prop :=
  ∠ AHB = 90° ∧ ∠ BHC = 90° ∧ ∠ CHA = 90°

noncomputable def altitude_intersect (A B C H : Point) (P Q M N : Point) : Prop :=
  -- Altitude from B intersects circle with diameter AC at P and Q
  (∃ H, ∠ BHP = 90° ∧ ∠ BHQ = 90° ∧
      (∃ circle: Circle, circle.diameter = AC ∧ P ∈ circle ∧ Q ∈ circle)) ∧
  -- Altitude from C intersects circle with diameter AB at M and N
  (∃ H, ∠ CHM = 90° ∧ ∠ CHN = 90° ∧
      (∃ circle: Circle, circle.diameter = AB ∧ M ∈ circle ∧ N ∈ circle))

theorem points_concyclic (A B C H P Q M N : Point)
  (h_acute : acute_triangle A B C)
  (h_orthocenter : orthocenter A B C H)
  (h_altitudes : altitude_intersect A B C H P Q M N) :
  concyclic P Q M N :=
by sorry

end points_concyclic_l792_792012


namespace derivative_at_0_eq_6_l792_792397

-- Definition of the function
def f (x : ℝ) : ℝ := (2 * x + 1)^3

-- Theorem statement indicating the derivative at x = 0 is 6
theorem derivative_at_0_eq_6 : (deriv f 0) = 6 := 
by 
  sorry -- The proof is omitted as per the instructions

end derivative_at_0_eq_6_l792_792397


namespace common_divisors_9240_8820_l792_792893

-- Define the prime factorizations given in the problem.
def pf_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def pf_8820 := [(2, 2), (3, 2), (5, 1), (7, 2)]

-- Define a function to calculate the gcd of two numbers given their prime factorizations.
def gcd_factorizations (pf1 pf2 : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
    List.filterMap (λ ⟨p, e1⟩ =>
      match List.lookup p pf2 with
      | some e2 => some (p, min e1 e2)
      | none => none
      end) pf1 

-- Define a function to compute the number of divisors from the prime factorization.
def num_divisors (pf: List (ℕ × ℕ)) : ℕ :=
    pf.foldl (λ acc ⟨_, e⟩ => acc * (e + 1)) 1

-- The Lean statement for the problem
theorem common_divisors_9240_8820 : 
    num_divisors (gcd_factorizations pf_9240 pf_8820) = 24 :=
by
    -- The proof goes here. We include sorry to indicate that the proof is omitted.
    sorry

end common_divisors_9240_8820_l792_792893


namespace compressor_stations_l792_792641

/-- 
Problem: Given three compressor stations connected by straight roads and not on the same line,
with distances satisfying:
1. x + y = 4z
2. x + z + y = x + a
3. z + y + x = 85

Prove:
- The range of values for 'a' such that the described configuration of compressor stations is 
  possible is 60.71 < a < 68.
- The distances between the compressor stations for a = 5 are x = 70, y = 0, z = 15.
--/
theorem compressor_stations (x y z a : ℝ) 
  (h1 : x + y = 4 * z)
  (h2 : x + z + y = x + a)
  (h3 : z + y + x = 85) :
  (60.71 < a ∧ a < 68) ∧ (a = 5 → x = 70 ∧ y = 0 ∧ z = 15) :=
  sorry

end compressor_stations_l792_792641


namespace station_base_volume_l792_792256

theorem station_base_volume :
  let S1 := (1 / (2 * Real.sqrt 2), 1 / (2 * Real.sqrt 2), 1 / (2 * Real.sqrt 2))
  let S2 := (-1 / (2 * Real.sqrt 2), -1 / (2 * Real.sqrt 2), 1 / (2 * Real.sqrt 2))
  let S3 := (1 / (2 * Real.sqrt 2), -1 / (2 * Real.sqrt 2), -1 / (2 * Real.sqrt 2))
  let S4 := (-1 / (2 * Real.sqrt 2), 1 / (2 * Real.sqrt 2), -1 / (2 * Real.sqrt 2))
  ∃ B : (ℝ × ℝ × ℝ) → Prop, 
    (∀ (x y z : ℝ), B (x, y, z) ↔
      (x - 1 / (2 * Real.sqrt 2))^2 + 
      (y - 1 / (2 * Real.sqrt 2))^2 + 
      (z - 1 / (2 * Real.sqrt 2))^2 +
      (x + 1 / (2 * Real.sqrt 2))^2 + 
      (y + 1 / (2 * Real.sqrt 2))^2 + 
      (z + 1 / (2 * Real.sqrt 2))^2 ≤ 15) 
    → volume (B) = 27 * Real.sqrt 6 * Real.pi / 8 := sorry

end station_base_volume_l792_792256


namespace find_p_q_r_sum_l792_792157

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V) (p q r : ℝ)

-- Conditions definition
def mutually_orthogonal (a b c : V) : Prop :=
  inner_product_space.is_ortho a b ∧ inner_product_space.is_ortho b c ∧ inner_product_space.is_ortho c a

def unit_vector (v : V) : Prop :=
  ∥v∥ = 1

def given_eq (a b c : V) (p q r : ℝ) : Prop :=
  a = p • (a × b) + q • (b × c) + r • (c × a)

def dot_product_condition (a b c : V) : Prop :=
  ⟪a, (b × c)⟫ = 1

-- Main proof statement
theorem find_p_q_r_sum :
  mutually_orthogonal a b c ∧ unit_vector a ∧ unit_vector b ∧ unit_vector c ∧
  given_eq a b c p q r ∧ dot_product_condition a b c →
  p + q + r = 1 :=
by 
  sorry

end find_p_q_r_sum_l792_792157


namespace number_of_common_divisors_l792_792909

theorem number_of_common_divisors :
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  let divisors_count := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  gcd_ab = 420 ∧ divisors_count = 24 :=
by
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 420 := sorry
  have h2 : (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 24 := by norm_num
  exact ⟨h1, h2⟩

end number_of_common_divisors_l792_792909


namespace athenas_min_wins_l792_792217

theorem athenas_min_wins (total_games : ℕ) (games_played : ℕ) (wins_so_far : ℕ) (losses_so_far : ℕ) 
                          (win_percentage_threshold : ℝ) (remaining_games : ℕ) (additional_wins_needed : ℕ) :
  total_games = 44 ∧ games_played = wins_so_far + losses_so_far ∧ wins_so_far = 20 ∧ losses_so_far = 15 ∧ 
  win_percentage_threshold = 0.6 ∧ remaining_games = total_games - games_played ∧ additional_wins_needed = 27 - wins_so_far → 
  additional_wins_needed = 7 :=
by
  sorry

end athenas_min_wins_l792_792217


namespace intersection_of_A_and_B_l792_792851

def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {-1, 0, 1}
def intersection : Set ℝ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = intersection := 
by sorry

end intersection_of_A_and_B_l792_792851


namespace probability_different_tens_digit_l792_792602

theorem probability_different_tens_digit :
  let n := 6
  let range := Set.Icc 10 59
  let tens_digit (x : ℕ) := x / 10
  let valid_set (s : Set ℕ) := ∀ x ∈ s, 10 ≤ x ∧ x ≤ 59
  let different_tens_digits (s : Set ℕ) := (∀ (x y : ℕ), x ∈ s → y ∈ s → x ≠ y → tens_digit x ≠ tens_digit y)
  let total_ways := Nat.choose 50 6
  let favorable_ways := 5 * 10 * 9 * 10^4
  let probability := favorable_ways * 1 / total_ways
  valid_set ({ x | x ∈ range } : Set ℕ) →
  different_tens_digits ({ x | x ∈ range } : Set ℕ) →
  probability = (1500000 : ℚ) / 5296900 :=
by
  sorry

end probability_different_tens_digit_l792_792602


namespace B_share_is_552_l792_792333

def initial_amount : ℝ := 1800
def tax_rate : ℝ := 0.05
def interest_rate : ℝ := 0.03
def ratio_A : ℕ := 2
def ratio_B : ℕ := 3
def ratio_C : ℕ := 4

theorem B_share_is_552 :
  let total_deductions := initial_amount * tax_rate + initial_amount * interest_rate,
      remaining_amount := initial_amount - total_deductions,
      total_parts := ratio_A + ratio_B + ratio_C,
      value_per_part := remaining_amount / total_parts in
  ratio_B * value_per_part = 552 := by
  sorry

end B_share_is_552_l792_792333


namespace cost_of_chlorine_l792_792824

/--
Gary has a pool that is 10 feet long, 8 feet wide, and 6 feet deep.
He needs to buy one quart of chlorine for every 120 cubic feet of water.
Chlorine costs $3 per quart.
Prove that the total cost of chlorine Gary spends is $12.
-/
theorem cost_of_chlorine:
  let length := 10
  let width := 8
  let depth := 6
  let volume := length * width * depth
  let chlorine_per_cubic_feet := 1 / 120
  let chlorine_needed := volume * chlorine_per_cubic_feet
  let cost_per_quart := 3
  let total_cost := chlorine_needed * cost_per_quart
  total_cost = 12 :=
by
  sorry

end cost_of_chlorine_l792_792824


namespace probability_even_dice_l792_792672

noncomputable section

def dice_sides := {1, 2, 3, 4, 5, 6}

def is_even (n : ℕ) : Prop := n % 2 = 0

def num_even_sides (d: Set ℕ) : ℕ :=
d.countp is_even

def probability (favorable total : ℕ) : ℚ :=
↑favorable / ↑total

-- Probability calculation for two events being even
theorem probability_even_dice :
  probability ((num_even_sides dice_sides) * (num_even_sides dice_sides)) ((dice_sides.size) * (dice_sides.size)) = 1 / 4 :=
by
  -- detailed step-by-step proof would follow here
  sorry

end probability_even_dice_l792_792672


namespace remainder_modulo_9_l792_792813

noncomputable def power10 := 10^15
noncomputable def power3  := 3^15

theorem remainder_modulo_9 : (7 * power10 + power3) % 9 = 7 := by
  -- Define the conditions given in the problem
  have h1 : (10 % 9 = 1) := by 
    norm_num
  have h2 : (3^2 % 9 = 0) := by 
    norm_num
  
  -- Utilize these conditions to prove the statement
  sorry

end remainder_modulo_9_l792_792813


namespace complex_division_l792_792402

open Complex

theorem complex_division :
  (1 + 2 * I) / (3 - 4 * I) = -1 / 5 + 2 / 5 * I :=
by
  sorry

end complex_division_l792_792402


namespace complex_number_solution_l792_792396

theorem complex_number_solution : ∃ (z : ℂ), (|z - 2| = |z + 4| ∧ |z - 2| = |z - 2 * complex.I| ∧ z = -1 - complex.I) :=
by {
  use -1 - complex.I,
  split,
  { -- part 1: |z - 2| = |z + 4|
    sorry },
  split,
  { -- part 2: |z - 2| = |z - 2i|
    sorry },
  { -- part 3: z = -1 - i
    refl }
}

end complex_number_solution_l792_792396


namespace find_lost_bowls_l792_792264

def bowls_problem (L : ℕ) : Prop :=
  let total_bowls := 638
  let broken_bowls := 15
  let payment := 1825
  let fee := 100
  let safe_bowl_payment := 3
  let lost_broken_bowl_cost := 4
  100 + 3 * (total_bowls - L - broken_bowls) - 4 * (L + broken_bowls) = payment

theorem find_lost_bowls : ∃ L : ℕ, bowls_problem L ∧ L = 26 :=
  by
  sorry

end find_lost_bowls_l792_792264


namespace rachel_homework_difference_l792_792200

def pages_of_math_homework : Nat := 5
def pages_of_reading_homework : Nat := 2

theorem rachel_homework_difference : 
  pages_of_math_homework - pages_of_reading_homework = 3 :=
sorry

end rachel_homework_difference_l792_792200


namespace regular_polygon_interior_angle_l792_792338

theorem regular_polygon_interior_angle (sum_of_interior_angles : ℝ) (h₁ : sum_of_interior_angles = 3420) :
  ∃ (angle : ℝ), angle = 162.857 :=
by
  use 162.857
  sorry

end regular_polygon_interior_angle_l792_792338


namespace relationship_y1_y2_y3_l792_792008

-- Define the quadratic function with the given parameters
def quadratic (a c x : ℝ) : ℝ := a * (x - 3) ^ 2 + c

-- Given conditions
variable (a c : ℝ)
variable (ha : a < 0)

-- Function values at specific x-values
def y1 := quadratic a c (Real.sqrt 5)
def y2 := quadratic a c 0
def y3 := quadratic a c 4

-- The theorem stating the desired relationship
theorem relationship_y1_y2_y3 : y2 < y3 ∧ y3 < y1 :=
by
  -- Proof goes here, using the given conditions
  sorry

end relationship_y1_y2_y3_l792_792008


namespace coeff_x2_in_expansion_l792_792609

theorem coeff_x2_in_expansion : 
  let general_term (r : ℕ) : ℤ := (Nat.choose 5 r) * (2 ^ r) 
  in general_term 2 = 40 :=
by
  -- This is where the proof should go
  sorry

end coeff_x2_in_expansion_l792_792609


namespace sum_prime_factors_of_2_pow_10_minus_1_l792_792248

theorem sum_prime_factors_of_2_pow_10_minus_1 : 
  let n := 2^10 - 1 in 
  ∑ p in {p | p.prime ∧ p ∣ n}, p = 45 := by
  -- Definition and theorem statements all together.
  let n := 2^10 - 1
  have : 2^10 - 1 = (2^5 - 1) * (2^5 + 1) by norm_num
  have pf1 : prime 31 := prime_31
  have pf2 : prime 3 := prime_3
  have pf3 : prime 11 := prime_11
  have factors : multiset.prod {31, 3, 11} = 2^10 - 1 := by sorry
  exact multiset.sum {31, 3, 11} = 45

end sum_prime_factors_of_2_pow_10_minus_1_l792_792248


namespace intersection_of_M_and_N_l792_792090

def M : Set ℝ := {x | x ≥ 0 ∧ x < 16}
def N : Set ℝ := {x | x ≥ 1/3}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l792_792090


namespace A_n_increases_l792_792551

-- Define the conditions given in the problem.
def M : Finset ℕ := {1, 2, ..., 30} -- A placeholder for a set of 30 distinct positive numbers

-- Assume the elements in the set are positive and distinct.
axiom elems_distinct : ∀ x ∈ M, x > 0

-- Define the sequence A_n in Lean.
def A_n (n : ℕ) : ℕ :=
  if h : n ≤ 30 then M.to_list.take n else 0

-- Lean statement for the proof.
theorem A_n_increases (n : ℕ) (h : 1 ≤ n ∧ n ≤ 29) : A_n (n + 1) > A_n n :=
by { sorry }

end A_n_increases_l792_792551


namespace csc_cos_sum_l792_792746

theorem csc_cos_sum :
  (csc (π / 18) + 4 * cos (π / 9) = 2) :=
by
  sorry

end csc_cos_sum_l792_792746


namespace increasing_function_a_l792_792453

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≥ 0 then
    x^2
  else
    x^3 - (a-1)*x + a^2 - 3*a - 4

theorem increasing_function_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f x a ≤ f y a) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end increasing_function_a_l792_792453


namespace determine_plane_l792_792295

def three_points_in_space (A B C : Type) : Prop := 
  ∃ (p1 p2 p3 : A), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

def point_and_line (A B : Type) : Prop :=
  ∃ (p: A) (l : B), true

def two_intersecting_lines (A B : Type) : Prop :=
  ∃ (l1 l2 : B) (p : A), l1 ≠ l2 ∧ p ∈ l1 ∧ p ∈ l2

def countless_points (A : Type) : Prop := 
  ∃ (S : set A), infinite S

theorem determine_plane (A B : Type) [nonempty A]:
  (two_intersecting_lines A B) → 
  ¬(three_points_in_space A) ∧ ¬(point_and_line A B) ∧ ¬(countless_points A) :=
begin
  sorry
end

end determine_plane_l792_792295


namespace complement_intersection_l792_792883

def P := {x : ℝ | x ≥ 2}
def Q := {x : ℝ | 1 < x ∧ x ≤ 2}
def complement_R (S: set ℝ) := {x | ¬(x ∈ S)}

theorem complement_intersection :
  (complement_R P ∩ Q) = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end complement_intersection_l792_792883


namespace surface_area_correct_l792_792434

-- Define the properties of the tetrahedron
structure Tetrahedron :=
(edge_length : ℝ)
(equilateral_faces : ∀ f, True)  -- All faces are equilateral triangles (no actual face enumeration needed for this example)

-- The specific tetrahedron in question
def given_tetrahedron : Tetrahedron :=
{ edge_length := 2,
  equilateral_faces := by simp }

-- Calculation of the surface area
def surface_area (T : Tetrahedron) : ℝ :=
4 * (1 / 2 * T.edge_length * T.edge_length * (Math.sqrt 3 / 2)) 

-- The theorem stating the problem
theorem surface_area_correct :
  surface_area given_tetrahedron = 4 * Math.sqrt 3 :=
by simp [surface_area, given_tetrahedron] sorry

end surface_area_correct_l792_792434


namespace largest_B_181_l792_792770

noncomputable def binom (n k : ℕ) : ℚ := Nat.choose n k
def B (n k : ℕ) (p : ℚ) := binom n k * p^k

theorem largest_B_181 : ∃ k, B 2000 181 (1 / 10) = arg_max k (B 2000 k (1 / 10)) where
  arg_max (k : ℕ) (f : ℕ → ℚ) := k ≤ 2000 ∧ ∀ j, j ≤ 2000 → f j ≤ f k := sorry

end largest_B_181_l792_792770


namespace count_pairs_union_sets_l792_792247

theorem count_pairs_union_sets (a1 a2 a3 : Type) (A B : set Type) :
  (A ∪ B = {a1, a2, a3}) → (A ≠ B) → (∀ A B, (A, B) ≠ (B, A) → true)
  → (∃ pairs : Type, pairs = 27) := sorry

end count_pairs_union_sets_l792_792247


namespace intersection_of_sets_l792_792082

theorem intersection_of_sets :
  let M := { x : ℝ | 0 ≤ x ∧ x < 16 }
  let N := { x : ℝ | x ≥ 1/3 }
  M ∩ N = { x : ℝ | 1/3 ≤ x ∧ x < 16 } :=
by
  sorry

end intersection_of_sets_l792_792082


namespace find_a_l792_792466

-- Defining the problem conditions
def rational_eq (x a : ℝ) :=
  x / (x - 3) - 2 * a / (x - 3) = 2

def extraneous_root (x : ℝ) : Prop :=
  x = 3

-- Theorem: Given the conditions, prove that a = 3 / 2
theorem find_a (a : ℝ) : (∃ x, extraneous_root x ∧ rational_eq x a) → a = 3 / 2 :=
  by
    sorry

end find_a_l792_792466


namespace problem1_problem2_l792_792971

section

variable {n : ℕ}
variable {a : ℕ → ℝ} -- Assume a_n : ℝ to handle general results
variable {b : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {T : ℕ → ℝ}

-- Condition 1: Sum of first n terms of {a_n} is S_n
axiom h₁ (S : ℕ → ℝ) (a : ℕ → ℝ) : ∀ n : ℕ, S n = ∑ i in range (n + 1), a i

-- Condition 2: The sequence 1, a_n, S_n form an arithmetic sequence
axiom h₂ : ∀ n, 1 + (S n - 1) = 2 * a n

-- Problem 1: Verify general formula for the sequence a_n
theorem problem1 : a n = 2^(n-1) := sorry

-- Condition 3: Another sequence {b_n} satisfies a_n b_n = 1 + 2n a_n
axiom h₃ : ∀ n, a n * b n = 1 + 2 * n * a n

-- Problem 2: Verify the sum of the first n terms of {b_n} is T_n
theorem problem2 : T n = n^2 + n + 2 - 1/2^(n-1) := sorry

end

end problem1_problem2_l792_792971


namespace total_vacations_and_classes_l792_792053

def kelvin_classes := 90
def grant_vacations := 4 * kelvin_classes
def total := grant_vacations + kelvin_classes

theorem total_vacations_and_classes :
  total = 450 :=
by
  sorry

end total_vacations_and_classes_l792_792053


namespace marching_band_l792_792623

theorem marching_band (total_members brass woodwind percussion : ℕ)
  (h1 : brass + woodwind + percussion = 110)
  (h2 : woodwind = 2 * brass)
  (h3 : percussion = 4 * woodwind) :
  brass = 10 := by
  sorry

end marching_band_l792_792623


namespace sum_of_squares_eq_l792_792190

theorem sum_of_squares_eq :
  ∀ (M G D : ℝ), 
  (M = G / 3) → 
  (G = 450) → 
  (D = 2 * G) → 
  (M^2 + G^2 + D^2 = 1035000) :=
by
  intros M G D hM hG hD
  sorry

end sum_of_squares_eq_l792_792190


namespace smallest_coefficient_of_expansion_l792_792241

theorem smallest_coefficient_of_expansion (x n : ℕ) : 
  ( (7 / real.sqrt x) - 3 * x )^n 
  → ( (x - 1) ^ n) 
  → (729 = (sum_of_coefficients (7 / real.sqrt x - 3 * x)^n) / (sum_of_binomial_coefficients (7 / real.sqrt x - 3 * x)^n)) 
  → (-20 = coefficient_of_term_with_smallest_coefficient (x - 1)^n ) :=
begin
  -- sorry to skip the proof details
  sorry
end

end smallest_coefficient_of_expansion_l792_792241


namespace find_AB_l792_792957

variables {AB CD AD BC AP PD APD PQ Q: ℝ}

def is_rectangle (ABCD : Prop) := ABCD

variables (P_on_BC : Prop)
variable (BP CP: ℝ)
variable (tan_angle_APD: ℝ)

theorem find_AB (ABCD : Prop) (P_on_BC : Prop) (BP CP: ℝ) (tan_angle_APD: ℝ) : 
  is_rectangle ABCD →
  P_on_BC →
  BP = 24 →
  CP = 12 →
  tan_angle_APD = 2 →
  AB = 27 := 
by
  sorry

end find_AB_l792_792957


namespace find_dividend_l792_792301

theorem find_dividend : ∃ d : ℕ, d = 6 * 40 + 28 := by
  have h1 : 6 * 40 = 240 := by sorry
  have h2 : 240 + 28 = 268 := by sorry
  existsi 268
  rw [h1, h2]
  rfl

end find_dividend_l792_792301


namespace laborers_present_l792_792251

theorem laborers_present (total_laborers : ℕ) (percentage_present : ℝ) (rounded_present : ℕ) 
  (h1 : total_laborers = 156) 
  (h2 : percentage_present = 44.9) 
  (h3 : rounded_present = round (0.449 * total_laborers)) :
  rounded_present = 70 := 
sorry

end laborers_present_l792_792251


namespace domain_log_eq_domain_exp_l792_792856

theorem domain_log_eq_domain_exp (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f (2^x) ∈ set.Icc 0 2) →
  (∀ x, 2 ≤ x ∧ x ≤ 16 → f (real.log2 x) ∈ set.Icc 0 2) :=
begin
  intros h x hx,
  sorry
end

end domain_log_eq_domain_exp_l792_792856


namespace range_of_a_l792_792483

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, a ≤ x ∧ (x : ℝ) < 2 → x = -1 ∨ x = 0 ∨ x = 1) ↔ (-2 < a ∧ a ≤ -1) :=
by
  sorry

end range_of_a_l792_792483


namespace area_of_circle_l792_792757

-- Define the circle equation as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 10 * x + 6 * y = 0

-- Statement to prove the area of the circle
theorem area_of_circle : let r := sqrt 34 in
  let area := π * r^2 in
  ∀ x y : ℝ, circle_eq x y → area = 34 * π :=
by 
  sorry

end area_of_circle_l792_792757


namespace find_quadruples_l792_792394

def is_solution (x y z n : ℕ) : Prop :=
  x^2 + y^2 + z^2 + 1 = 2^n

theorem find_quadruples :
  ∀ x y z n : ℕ, is_solution x y z n ↔ 
  (x, y, z, n) = (1, 1, 1, 2) ∨
  (x, y, z, n) = (0, 0, 1, 1) ∨
  (x, y, z, n) = (0, 1, 0, 1) ∨
  (x, y, z, n) = (1, 0, 0, 1) ∨
  (x, y, z, n) = (0, 0, 0, 0) :=
by
  sorry

end find_quadruples_l792_792394


namespace combined_vacations_and_classes_l792_792055

-- Define the conditions
def Kelvin_classes : ℕ := 90
def Grant_vacations : ℕ := 4 * Kelvin_classes

-- The Lean 4 statement proving the combined total of vacations and classes
theorem combined_vacations_and_classes : Kelvin_classes + Grant_vacations = 450 := by
  sorry

end combined_vacations_and_classes_l792_792055


namespace rook_connected_partition_l792_792179

theorem rook_connected_partition (X : set (ℕ × ℕ)) (hx : rook_connected X) (h100 : |X| = 100) :
  ∃ (pairs : set (ℕ × ℕ) × (ℕ × ℕ)) (H : pairs ⊆ X), ∀ ((a, b) ∈ pairs), (a.1 = b.1 ∨ a.2 = b.2) := 
sorry

-- Definitions used from the conditions:
-- rook_connected : definition of rook-connected should be formally stated in Lean terms.
-- For the sake of this example, we assume it’s already defined adequately.

end rook_connected_partition_l792_792179


namespace area_enclosed_by_cardioid_l792_792741

-- Define the cardioid function in polar coordinates
def cardioid (φ : ℝ) : ℝ := 1 + real.cos φ

-- Define the area function for polar coordinates
def area_polar (ρ : ℝ → ℝ) (a b : ℝ) : ℝ := 
  (1 / 2) * ∫ φ in a..b, (ρ φ)^2

-- State the theorem asserting the area enclosed by the cardioid
theorem area_enclosed_by_cardioid : area_polar cardioid 0 (2 * real.pi) = (3 * real.pi) / 2 := 
sorry

end area_enclosed_by_cardioid_l792_792741


namespace smallest_n_int_expr_l792_792162

noncomputable def a : ℝ := Real.pi / 2010

def expr (n : ℕ) : ℝ :=
  2 * (Finset.sum (Finset.range n) (λ k, cos ((k.succ : ℝ)^2 * a) * sin ((k.succ : ℝ) * a)))

def is_integer (x : ℝ) : Prop :=
  ∃ (m : ℤ), x = m

theorem smallest_n_int_expr : ∃ n : ℕ, (0 < n) ∧ is_integer (expr n) ∧ ∀ m : ℕ, (0 < m ∧ m < n) → ¬ is_integer (expr m) :=
by
  sorry

end smallest_n_int_expr_l792_792162


namespace equilateral_triangle_ratio_l792_792286

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let A := (s^2 * Real.sqrt 3) / 4
      P := 3 * s 
  in P / A = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end equilateral_triangle_ratio_l792_792286


namespace positive_difference_sum_of_squares_l792_792282

-- Given definitions
def sum_of_squares_even (n : ℕ) : ℕ :=
  4 * (n * (n + 1) * (2 * n + 1)) / 6

def sum_of_squares_odd (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- The explicit values for this problem
def sum_of_squares_first_25_even := sum_of_squares_even 25
def sum_of_squares_first_20_odd := sum_of_squares_odd 20

-- The required proof statement
theorem positive_difference_sum_of_squares : 
  (sum_of_squares_first_25_even - sum_of_squares_first_20_odd) = 19230 := by
  sorry

end positive_difference_sum_of_squares_l792_792282


namespace complex_roots_quadratic_l792_792985

theorem complex_roots_quadratic (ω : ℂ)
  (hω : ω^9 = 1)
  (hω_ne : ω ≠ 1) :
  let α := ω + ω^3 + ω^6,
      β := ω^2 + ω^4 + ω^8 in
  (∀ x : ℂ, x^2 + x + 1 = 0 ↔ x = α ∨ x = β) :=
by sorry

end complex_roots_quadratic_l792_792985


namespace count_integers_satisfying_equation_l792_792058

theorem count_integers_satisfying_equation : 
  (finset.filter (λ x : ℤ, ((x^2 - 2*x - 2) ^ (x + 3) = 1)) (finset.range 100)).card = 4 := 
sorry

end count_integers_satisfying_equation_l792_792058


namespace remainder_of_sum_div_7_l792_792990

theorem remainder_of_sum_div_7 (n : ℕ) (h : 0 < n) :
  (∑ k in finset.range (n + 1), 5^k * nat.choose n k) % 7 = 0 ∨
  (∑ k in finset.range (n + 1), 5^k * nat.choose n k) % 7 = 5 :=
by sorry

end remainder_of_sum_div_7_l792_792990


namespace find_angle_B_find_range_of_b_l792_792501

-- Given conditions for part (1) and part (2)
variables {a b c A B C : ℝ}
variables (triangle_acute : B ∈ (0, π / 2) ∧ C ∈ (0, π / 2) ∧ A ∈ (0, π / 2))
variables (side_lengths_positive : a > 0 ∧ b > 0 ∧ c > 0)

-- Part (1): Given condition for cos^2(B) + 2sqrt(3)sin(B)cos(B) - sin^2(B) = 1, prove B = π / 3
theorem find_angle_B (h : cos B ^ 2 + 2 * sqrt 3 * sin B * cos B - sin B ^ 2 = 1) :
  B = π / 3 :=
sorry

-- Part (2): Given condition for 2c cos A + 2a cos C = bc, find the range of b
theorem find_range_of_b (h : 2 * c * cos A + 2 * a * cos C = b * c) :
  b ∈ (sqrt 3, 2 * sqrt 3) :=
sorry

end find_angle_B_find_range_of_b_l792_792501


namespace Julia_played_with_11_kids_on_Monday_l792_792146

theorem Julia_played_with_11_kids_on_Monday
  (kids_on_Tuesday : ℕ)
  (kids_on_Monday : ℕ) 
  (h1 : kids_on_Tuesday = 12)
  (h2 : kids_on_Tuesday = kids_on_Monday + 1) : 
  kids_on_Monday = 11 := 
by
  sorry

end Julia_played_with_11_kids_on_Monday_l792_792146


namespace express_delivery_revenue_growth_l792_792727

variable (x : ℝ)
variables (initial_revenue revenue_2019 : ℝ)
variables (years : ℕ)

def revenue_growth (initial_revenue : ℝ) (x : ℝ) (years : ℕ) : ℝ :=
  initial_revenue * (1 + x) ^ years

theorem express_delivery_revenue_growth :
  initial_revenue = 5000 →
  revenue_2019 = 7500 →
  years = 2 →
  revenue_growth initial_revenue x years = revenue_2019 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  sorry
end

end express_delivery_revenue_growth_l792_792727


namespace lights_remaining_on_l792_792717

theorem lights_remaining_on :
  let total_lights := 150
  let multiples_of_3 := total_lights / 3
  let multiples_of_5 := total_lights / 5
  let multiples_of_15 := total_lights / 15
  let initial_on_lights := 150
  initial_on_lights - ((multiples_of_3) + (multiples_of_5 - multiples_of_15)) = 80 :=
begin
  sorry
end

end lights_remaining_on_l792_792717


namespace sum_of_solutions_eq_zero_l792_792503

theorem sum_of_solutions_eq_zero (x y : ℝ) (h1 : y = 9) (h2 : x^2 + y^2 = 169) : (x = real.sqrt 88 ∨ x = -real.sqrt 88) → x + (-x) = 0 :=
by
  intro h
  cases h
  · rw [h]
    simp,
  · rw [h]
    simp

end sum_of_solutions_eq_zero_l792_792503


namespace part_a_equivalent_part_b_part_c_l792_792732

def is_connected (cells : set (ℤ × ℤ)) : Prop := sorry -- Define connectivity in the grid
def is_n_mino (P : set (ℤ × ℤ)) (n : ℕ) : Prop := (P.card = n) ∧ is_connected P

-- Question (a) Definition for polyminos
def polyminos : set (set (ℤ × ℤ)) := {P | ∃ n : ℕ, is_n_mino P n}

-- Boundary calculation
def boundary_squares (P : set (ℤ × ℤ)) : ℕ := sorry -- Boundary calculation definition

-- |P| is just the size of P 
def size (P : set (ℤ × ℤ)) : ℕ := P.card

-- Question (a)
theorem part_a_equivalent (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  ∑ P in polyminos, (size P) * x^(size P - 1) * (1 - x)^(boundary_squares P) = 1 
:= sorry

-- Question (b)
theorem part_b (P : set (ℤ × ℤ)) (hP : ∃ n : ℕ, is_n_mino P n) : 
  boundary_squares P ≤ 2 * size P + 2 := sorry

-- Question (c)
theorem part_c (n : ℕ) : 
  ∃ upper_bound : ℝ, upper_bound = 6.75 ∧ 
  card ({P : set (ℤ × ℤ) | is_n_mino P n}) < upper_bound^n 
:= sorry

end part_a_equivalent_part_b_part_c_l792_792732


namespace thunderbird_lineups_l792_792372

open Nat
open Function
open Multiset
open List
open FiniteMath
open Classical

variables {α : Type*}

noncomputable theory

def choose (n k : ℕ) : ℕ :=
  if k > n then 0 else (n.choose k : ℕ)

-- Define the team size and specific players.
def team_size := 15
def specific_players := 3
def remaining_players := team_size - specific_players

-- Define binomial coefficients
def binom_12_4 := choose 12 4
def binom_12_5 := choose 12 5

-- Define the total valid lineups
def valid_lineups := 3 * binom_12_4 + binom_12_5

-- The theorem statement
theorem thunderbird_lineups : valid_lineups = 2277 :=
by
  unfold valid_lineups binom_12_4 binom_12_5
  simp [choose]
  sorry

-- Here 'sorry' is a placeholder for the actual proof.

end thunderbird_lineups_l792_792372


namespace magnitude_AB_l792_792447

def point := (ℝ × ℝ)

def magnitude (A B : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

noncomputable def A : point := (-3, 4)
noncomputable def B : point := (5, -2)

theorem magnitude_AB :
  magnitude A B = 10 := by
  sorry

end magnitude_AB_l792_792447


namespace average_distance_per_day_l792_792922

def miles_monday : ℕ := 12
def miles_tuesday : ℕ := 18
def miles_wednesday : ℕ := 21
def total_days : ℕ := 3

def total_distance : ℕ := miles_monday + miles_tuesday + miles_wednesday

theorem average_distance_per_day : total_distance / total_days = 17 := by
  sorry

end average_distance_per_day_l792_792922


namespace intersection_distance_l792_792376

-- Definitions for the problem conditions
def eq1 (x y : ℝ) := x^2 + y^2 = 13
def eq2 (x y : ℝ) := x + y = 4

-- The statement we want to prove
theorem intersection_distance : 
  ∃ (x1 y1 x2 y2 : ℝ), eq1 x1 y1 ∧ eq2 x1 y1 ∧ eq1 x2 y2 ∧ eq2 x2 y2 ∧ 
    (real.dist (x1, y1) (x2, y2) = 4 * real.sqrt 5) := 
begin
  sorry
end

end intersection_distance_l792_792376


namespace number_of_special_subsets_l792_792432

theorem number_of_special_subsets :
  let S := {2, 3, 4, 7}
  ∃ A ⊆ S, (∃ x ∈ A, x % 2 = 1) ∧ (nat.card {A : set ℕ | A ⊆ S ∧ ∃ x ∈ A, x % 2 = 1} = 12) :=
by
  let S := {2, 3, 4, 7}
  sorry

end number_of_special_subsets_l792_792432


namespace max_sum_ab_bc_cd_de_ea_l792_792638

theorem max_sum_ab_bc_cd_de_ea :
  ∃ (a b c d e : ℕ), 
  a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} ∧
  c ∈ {1, 2, 3, 4, 5} ∧ d ∈ {1, 2, 3, 4, 5} ∧
  e ∈ {1, 2, 3, 4, 5} ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (∀ a' b' c' d' e' : ℕ, a' ∈ {1, 2, 3, 4, 5} ∧ b' ∈ {1, 2, 3, 4, 5} ∧
  c' ∈ {1, 2, 3, 4, 5} ∧ d' ∈ {1, 2, 3, 4, 5} ∧
  e' ∈ {1, 2, 3, 4, 5} ∧
  a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ a' ≠ e' ∧
  b' ≠ c' ∧ b' ≠ d' ∧ b' ≠ e' ∧
  c' ≠ d' ∧ c' ≠ e' ∧
  d' ≠ e' → 
  ab + bc + cd + de + ea ≤ 59)
:=
sorry

end max_sum_ab_bc_cd_de_ea_l792_792638


namespace maximum_value_t_2_min_value_exists_in_interval_max_m_value_implies_t_range_l792_792995

noncomputable def f (t x : ℝ) : ℝ := x^3 - (3 * (t + 1) / 2) * x^2 + 3 * t * x + 1

-- (1) Prove that the maximum value of f(x) at t = 2 is 7/2
theorem maximum_value_t_2 :
  ∃ x ∈ set.Icc 0 2, (∀ y ∈ set.Icc 0 2, f 2 y ≤ f 2 x) ∧ f 2 x = 7 / 2 :=
sorry

-- (2) Prove that if there exists x_0 ∈ (0, 2) such that f(x_0) is the minimum 
-- value of f(x) on [0, 2], then the range of t is (0, 1/3]
theorem min_value_exists_in_interval :
  (∃ x_0 ∈ set.Ioo 0 2, ∀ x ∈ [0, 2], f t x_0 ≤ f t x) ↔ 0 < t ∧ t ≤ 1 / 3 :=
sorry

-- (3) Prove that if f(x) ≤ x e^x - m for any x ∈ [0, +∞) with the maximum value 
-- of m being -1, the range of t is (0, 1/3]
theorem max_m_value_implies_t_range :
  (∀ x ∈ set.Ici 0, f t x ≤ x * real.exp x + 1) ↔ 0 < t ∧ t ≤ 1 / 3 :=
sorry

end maximum_value_t_2_min_value_exists_in_interval_max_m_value_implies_t_range_l792_792995


namespace solve_BC_eq_sqrt5_minus1_plus1_l792_792325

noncomputable def sqrt {α : Type*} [linear_ordered_field α] (x : α) : α := sorry

variables (A B C D : Type) 
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]

variable (diam_circle : ℝ)
variable (passes_through_A_and_B : Prop)
variable (tangent_from_C : ℝ)
variable (length_AB : ℝ)
variable (BC : ℝ)

def eq_two_possible_values (x : ℝ) (v1 v2 : ℝ) : Prop :=
  x = v1 ∨ x = v2

theorem solve_BC_eq_sqrt5_minus1_plus1
  (diam_circle = real.sqrt 10)
  (passes_through_A_and_B)
  (tangent_from_C = 3)
  (length_AB = 1) :
  eq_two_possible_values BC (3 / 2 * (real.sqrt 5 - 1)) (3 / 2 * (real.sqrt 5 + 1)) := sorry

end solve_BC_eq_sqrt5_minus1_plus1_l792_792325


namespace even_function_expression_odd_function_expression_odd_function_monotonicity_l792_792468

-- Define the function and conditions
def f (x : ℝ) (a b c : ℝ) := (a * x^2 + 1) / (b * x + c)

-- Condition: f(1) = 2
def cond1 (a b c : ℝ) := f 1 a b c = 2

-- Condition: f(2) = 3
def cond2 (a b c : ℝ) := f 2 a b c = 3

-- Even function scenario
theorem even_function_expression : ∃ (a b c : ℝ), 
  (cond1 a b c) ∧ (cond2 a b c) ∧ (∀ x : ℝ, f x a b c = f (-x) a b c) → 
  f = λ x, (4 * x^2 + 5) / 3 := 
sorry

-- Odd function scenario
theorem odd_function_expression : ∃ (a b c : ℝ), 
  (cond1 a b c) ∧ (cond2 a b c) ∧ (∀ x : ℝ, f (-x) a b c = -f x a b c) → 
  f = λ x, (4 * x^2 + 2) / (3 * x) := 
sorry

-- Monotonicity in (0, 1/2)
theorem odd_function_monotonicity : 
  (∀ x : ℝ, f (-x) 2 (3/2) 0 = -f x 2 (3/2) 0) → 
  ∀ (x1 x2 : ℝ), (0 < x1) ∧ (x1 < x2) ∧ (x2 < (1/2)) → 
  f x1 2 (3/2) 0 > f x2 2 (3/2) 0 :=
sorry

end even_function_expression_odd_function_expression_odd_function_monotonicity_l792_792468


namespace exists_equation_of_line_l_no_real_number_a_l792_792446

-- Definitions
def P : Point := (2, 0)
def C : Circle := ⟨3, -2⟩
def r : ℝ := 3

-- (I)
def line_l1 : Line := 3 * x + 4 * y - 6 = 0
def line_l2 : Line := x = 2

theorem exists_equation_of_line_l (P : Point) (C : Circle) (r : ℝ) : 
  (∃ l : Line, l = line_l1 ∨ l = line_l2) :=
sorry

-- (II)
def chord_a_b (C : Circle) (a : ℝ) (P : Point) : Prop := 
  ∃ A B : Point, A ≠ B ∧ (ax - y + 1 = 0) ∧ (A ∈ C) ∧ (B ∈ C)

theorem no_real_number_a (C : Circle) (P : Point) : 
  ¬ (∃ a : ℝ, chord_a_b C a P ∧ bisect_perpendicularly (P) (A B)) :=
sorry

end exists_equation_of_line_l_no_real_number_a_l792_792446


namespace train_length_l792_792304

theorem train_length (speed_kmhr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h_speed_kmhr : speed_kmhr = 60) 
  (h_time_sec : time_sec = 7) 
  (h_length_m : length_m = speed_kmhr * (5 / 18) * time_sec) : 
  length_m = 116.69 := 
by
  have h_speed_ms : speed_kmhr * (5 / 18) = 16.67 := sorry
  have h_length : 16.67 * time_sec = 116.69 := sorry
  exact h_length

end train_length_l792_792304


namespace proof_problem_l792_792052

noncomputable def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3 * n

noncomputable def sequence_b (n : ℕ) : ℕ :=
  3 ^ n

noncomputable def sequence_c (n : ℕ) : ℕ :=
  sequence_b (sequence_a n)

theorem proof_problem :
  sequence_c 2017 = 27 ^ 2017 :=
by sorry

end proof_problem_l792_792052


namespace incorrect_statement_parallel_vectors_l792_792297

noncomputable def incorrect_statement (v w : ℝ^n) : Prop :=
  ∃ k : ℝ, (k ≠ 0 ∧ k ≠ 1 ∧ k ≠ -1) →
  ¬ (v = k • w) ∧ ¬ (w = k • v)

theorem incorrect_statement_parallel_vectors
  (v w : ℝ^n)
  (h₀ : ∀ u : ℝ^n, u ≠ 0 → (0 = k • u)) -- condition 1
  (h₁ : ‖(0:ℝ^n)‖ ≠ 1) -- condition 2
  (h₂ : ∀ k : ℝ, k ≠ 0 → v = k • w) -- condition 3 (incorrect statement)
  (h₃ : ∀ u v : ℝ^n, (∃ k : ℝ, u = k • v) → ∃ l : ℝ, v = l • u) -- condition 4
  : ∃ k : ℝ, k ≠ 0 ∧ (v = k • w) → (k > 0 → v = w) ∧ (k < 0 → v = -w) :=
by
  sorry

end incorrect_statement_parallel_vectors_l792_792297


namespace students_exceed_guinea_pigs_l792_792737

def number_of_students (students_per_class : ℕ) (number_of_classes : ℕ) : ℕ :=
  students_per_class * number_of_classes

def number_of_guinea_pigs (guinea_pigs_per_class : ℕ) (number_of_classes : ℕ) : ℕ :=
  guinea_pigs_per_class * number_of_classes

theorem students_exceed_guinea_pigs :
  ∀ (students_per_class guinea_pigs_per_class number_of_classes : ℕ),
  students_per_class = 25 →
  guinea_pigs_per_class = 3 →
  number_of_classes = 6 →
  (number_of_students students_per_class number_of_classes) -
  (number_of_guinea_pigs guinea_pigs_per_class number_of_classes) = 132 :=
by
  intros students_per_class guinea_pigs_per_class number_of_classes h_students h_guinea_pigs h_classes
  rw [h_students, h_guinea_pigs, h_classes]
  unfold number_of_students number_of_guinea_pigs
  simp
  sorry

end students_exceed_guinea_pigs_l792_792737


namespace find_year_after_2020_l792_792104

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_year_after_2020 :
  ∃ y : ℕ, 2020 < y ∧ sum_of_digits y = 4 ∧ (∀ z : ℕ, 2020 < z ∧ sum_of_digits z = 4 → y ≤ z) := 
begin
  sorry,
end

end find_year_after_2020_l792_792104


namespace Jane_age_proof_l792_792626

theorem Jane_age_proof (D J : ℕ) (h1 : D + 6 = (J + 6) / 2) (h2 : D + 14 = 25) : J = 28 :=
by
  sorry

end Jane_age_proof_l792_792626


namespace tan_11_pi_over_4_l792_792788

theorem tan_11_pi_over_4 : Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Proof is omitted
  sorry

end tan_11_pi_over_4_l792_792788


namespace ratio_independence_of_B_l792_792653

theorem ratio_independence_of_B (R r : ℝ) (O O1 : Point) (A : Point) (B : Point) (T : Point) 
  (h_circles : circle O R ∧ circle O1 r) 
  (h_external_touch : dist O O1 = R + r ∧ O ≠ O1) 
  (h_B_on_first_circle : dist O B = R) 
  (h_BT_tangent : is_tangent B T O1 r) :
  (dist B A / dist B T = real.sqrt (R / (R + r))) := 
sorry

end ratio_independence_of_B_l792_792653


namespace range_of_x_f_x_le_0_l792_792455

noncomputable def f : ℝ → ℝ := sorry

axiom f_increasing : ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ ≤ f x₂
axiom f_at_4 : f 4 = 0

theorem range_of_x_f_x_le_0 : {x : ℝ | x * f x ≤ 0} = set.Icc 0 4 :=
by
  sorry

end range_of_x_f_x_le_0_l792_792455


namespace rationalize_denominator_l792_792586

theorem rationalize_denominator :
  (Real.sqrt (5 / 12)) = ((Real.sqrt 15) / 6) :=
sorry

end rationalize_denominator_l792_792586


namespace determine_coordinates_of_M_l792_792131

def point_in_fourth_quadrant (M : ℝ × ℝ) : Prop :=
  M.1 > 0 ∧ M.2 < 0

def distance_to_x_axis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  |M.2| = d

def distance_to_y_axis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  |M.1| = d

theorem determine_coordinates_of_M :
  ∃ M : ℝ × ℝ, point_in_fourth_quadrant M ∧ distance_to_x_axis M 3 ∧ distance_to_y_axis M 4 ∧ M = (4, -3) :=
by
  sorry

end determine_coordinates_of_M_l792_792131


namespace seven_rotational_symmetric_hexominoes_l792_792062

def hexomino : Type := sorry -- Hexomino type definition
def is_rotational_symmetric (h : hexomino) : Prop := sorry -- Predicate for rotational symmetry

def count_rotational_symmetric_hexominoes (hexominoes : List hexomino) : Nat :=
  List.length (List.filter is_rotational_symmetric hexominoes)

theorem seven_rotational_symmetric_hexominoes :
  ∀ (hexominoes: List hexomino), hexominoes.length = 15 →
  count_rotational_symmetric_hexominoes(hexominoes) = 7 := sorry

end seven_rotational_symmetric_hexominoes_l792_792062


namespace find_side_c_find_area_l792_792513

theorem find_side_c (a b : ℝ) (angle_C : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : angle_C = 2 * angle_C) :
  c = Real.sqrt 10 :=
sorry

theorem find_area (a b c angle_C : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = Real.sqrt 10) (h4 : angle_C = 2 * angle_C) :
  let sin_C := Real.sqrt(1 - ((a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)) ^ 2) in
  (1 / 2 * a * b * sin_C) = (3 * Real.sqrt 15 / 4) :=
sorry

end find_side_c_find_area_l792_792513


namespace parabola_chord_midpoint_l792_792080

/-- 
If the point (3, 1) is the midpoint of a chord of the parabola y^2 = 2px, 
and the slope of the line containing this chord is 2, then p = 2. 
-/
theorem parabola_chord_midpoint (p : ℝ) :
    (∃ (m : ℝ), (m = 2) ∧ ∀ (x y : ℝ), y = 2 * x - 5 → y^2 = 2 * p * x → 
        ((x1 = 0 ∧ y1 = 0 ∧ x2 = 6 ∧ y2 = 6) → 
            (x1 + x2 = 6) ∧ (y1 + y2 = 2) ∧ (p = 2))) :=
sorry

end parabola_chord_midpoint_l792_792080


namespace age_of_first_person_added_l792_792292

theorem age_of_first_person_added :
  ∀ (T A x : ℕ),
    (T = 7 * A) →
    (T + x = 8 * (A + 2)) →
    (T + 15 = 8 * (A - 1)) →
    x = 39 :=
by
  intros T A x h1 h2 h3
  sorry

end age_of_first_person_added_l792_792292


namespace regions_of_diagonals_formula_l792_792140

def regions_of_diagonals (n : ℕ) : ℕ :=
  ((n - 1) * (n - 2) * (n * n - 3 * n + 12)) / 24

theorem regions_of_diagonals_formula (n : ℕ) (h : 3 ≤ n) :
  ∃ (fn : ℕ), fn = regions_of_diagonals n := by
  sorry

end regions_of_diagonals_formula_l792_792140


namespace circle_sector_radius_l792_792339

theorem circle_sector_radius (r : ℝ) :
  (2 * r + (r * (Real.pi / 3)) = 144) → r = 432 / (6 + Real.pi) := by
  sorry

end circle_sector_radius_l792_792339


namespace imaginary_part_of_complex_number_l792_792621

theorem imaginary_part_of_complex_number : 
  let z := (-3 + complex.i) / (2 + complex.i) in complex.im z = 1 :=
by
  sorry

end imaginary_part_of_complex_number_l792_792621


namespace tables_chairs_legs_l792_792354

theorem tables_chairs_legs (t : ℕ) (c : ℕ) (total_legs : ℕ) 
  (h1 : c = 8 * t) 
  (h2 : total_legs = 4 * c + 6 * t) 
  (h3 : total_legs = 798) : 
  t = 21 :=
by
  sorry

end tables_chairs_legs_l792_792354


namespace angle_A1OB_l792_792342

/-- Given conditions: 
1. The slanted parallelepiped has a base in the shape of a rectangle ABCD.
2. The side edges are AA_1, BB_1, CC_1, and DD_1.
3. A sphere with center at point O touches the edges BC, A_1B_1, and DD_1 at the points B, A_1, and D_1 respectively.
4. AD = 4.
5. The height of the parallelepiped is 1.
We need to prove: angle(A_1OB) = 2 * arcsin(1 / sqrt(5)) 
-/
theorem angle_A1OB 
  (A B C D A_1 B_1 C_1 D_1 O : Point)
  (h_parallelepiped : parallelepiped ABCD AA_1 BB_1 CC_1 DD_1)
  (h_sphere_touches : touches_sphere O B A_1 D_1)
  (h_AD : distance A D = 4)
  (h_height : height h_parallelepiped = 1) : 
  angle A_1 O B = 2 * real.arcsin (1 / real.sqrt 5) := 
sorry

end angle_A1OB_l792_792342


namespace max_value_A_upbound_l792_792442

noncomputable def max_value_A (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i ∧ x i < 1) : ℝ :=
  (Finset.univ.sum (λ i, real.sqrt (1 - x i))) / 
  real.sqrt (Finset.univ.sum (λ i, 1 / x i))

theorem max_value_A_upbound (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i ∧ x i < 1) :
  max_value_A n x hx ≤ real.sqrt (n) / 2 :=
sorry

end max_value_A_upbound_l792_792442


namespace sides_of_equilateral_triangle_l792_792630

theorem sides_of_equilateral_triangle
  (R : ℝ) (α : ℝ) : 
  let s := (R * (Real.sin (α / 2))) / (Real.sin (30 * Real.pi / 180 + α / 2)) in
  ∃ s, s = (R * (Real.sin (α / 2))) / (Real.sin (30 * Real.pi / 180 + α / 2)) :=
by 
  use (R * (Real.sin (α / 2))) / (Real.sin (30 * Real.pi / 180 + α / 2))
  trivial -- This line just serves to satisfy Lean's requirement for a complete proof. Remove when full proof is provided.

end sides_of_equilateral_triangle_l792_792630


namespace clock_hands_night_coincidences_l792_792486

def hours_in_night : ℕ := 12

def position_of_coincidence : ℕ := 12

def clock_hands_coincide_at(hour: ℕ, minute: ℕ, second: ℕ) : Prop := 
  hour = 0 ∧ minute = 0 ∧ second = 0

theorem clock_hands_night_coincidences : 
  ∀ (h1 m1 s1 h2 m2 s2 : ℕ), clock_hands_coincide_at h1 m1 s1 → clock_hands_coincide_at h2 m2 s2 → h1 = 0 ∧ h2 = 12 →
  (∃ n : ℕ, n = 2) :=
sorry

end clock_hands_night_coincidences_l792_792486


namespace tangent_of_11pi_over_4_l792_792782

theorem tangent_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 :=
sorry

end tangent_of_11pi_over_4_l792_792782


namespace primes_with_digit_product_189_l792_792382

/-- A function to return the product of the digits of a given number. -/
def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

/-- Check if a given number is prime.-/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Define the set of three-digit prime numbers whose digits multiply to 189 -/
def three_digit_primes_with_digit_product_189 : set ℕ :=
  {n | 100 ≤ n ∧ n < 1000 ∧ is_prime n ∧ digit_product n = 189}

/-- The set of three-digit prime numbers whose digits multiply to 189 is {379, 397, 739, 937}. -/
theorem primes_with_digit_product_189 :
  three_digit_primes_with_digit_product_189 = {379, 397, 739, 937} :=
by
  sorry

end primes_with_digit_product_189_l792_792382


namespace find_x_value_l792_792214

noncomputable def x_value (x y z : ℝ) := @Root.real 7 144

theorem find_x_value
  (x y z : ℝ)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 3)
  (h3 : z^2 / x = 4)
  (pos_x : 0 < x)
  (pos_y : 0 < y)
  (pos_z : 0 < z) :
  x = x_value x y z :=
sorry

end find_x_value_l792_792214


namespace magnitude_of_2a_minus_b_l792_792482

open EuclideanGeometry

variables (a b : EuclideanSpace ℝ (Fin 2))

theorem magnitude_of_2a_minus_b 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = 2) 
  (hab : ∥a + b∥ = Real.sqrt 5) :
  ∥2 • a - b∥ = 2 * Real.sqrt 2 :=
by
  sorry

end magnitude_of_2a_minus_b_l792_792482


namespace eccentricity_of_hyperbola_l792_792227

theorem eccentricity_of_hyperbola
  (t : ℝ)
  (h : t ≠ 0)
  (asymptote_perpendicular : ∃ (k : ℝ), (t = k^2) ∧ (k * -2 = -1)) :
  let e := (sqrt (4 + 1)) / 2 in
  e = sqrt 5 / 2 :=
by
  sorry

end eccentricity_of_hyperbola_l792_792227


namespace subtraction_result_l792_792192

-- Statement of the problem
theorem subtraction_result : 943 - 87 = 856 :=
begin
  -- The proof is not required, so we use sorry.
  sorry,
end

end subtraction_result_l792_792192


namespace color_lines_blue_l792_792197

-- Definitions needed
def general_position (lines : Set (ℝ × ℝ)) : Prop :=
∀ l1 l2 : (ℝ × ℝ), l1 ≠ l2 → ¬ ∃ p : ℝ × ℝ, p ∈ l1 ∧ p ∈ l2

def no_three_concurrent (lines : Set (ℝ × ℝ)) : Prop :=
∀ l1 l2 l3 : (ℝ × ℝ), ¬ ∃ p : ℝ × ℝ, p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3

axiom sufficiently_large (n : ℕ) : n ≥ 1000000 -- Assume n is a sufficiently large number

-- Main theorem statement
theorem color_lines_blue (n : ℕ) (lines : Set (ℝ × ℝ)) 
  (hn_pos : sufficiently_large n)
  (hg_position : general_position lines)
  (hc_no_three : no_three_concurrent lines) :
  ∃ (blue_lines : Set (ℝ × ℝ)), (blue_lines ⊆ lines) ∧ (size blue_lines ≥ sqrt n) ∧
  (∀ region : Set (ℝ × ℝ), (boundary region ⊆ blue_lines → region.is_infinite)) := 
sorry

end color_lines_blue_l792_792197


namespace find_lambda_l792_792828

theorem find_lambda (λ : ℝ) 
  (a : ℝ × ℝ × ℝ := (1, λ, 2))
  (b : ℝ × ℝ × ℝ := (2, -1, 1))
  (h : real.angle_between a.to_euclidean b.to_euclidean = real.angle.pi_div_three) :
  λ = -17 ∨ λ = 1 := sorry

end find_lambda_l792_792828


namespace common_divisors_9240_8820_l792_792900

def prime_factors_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def prime_factors_8820 := [(2, 2), (3, 2), (5, 1), (7, 1), (11, 1)]

def gcd_prime_factors := [(2, 2), (3, 1), (5, 1), (7, 1), (11, 1)]

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.snd + 1)) 1

theorem common_divisors_9240_8820 :
  num_divisors gcd_prime_factors = 32 := by
  sorry

end common_divisors_9240_8820_l792_792900


namespace product_gcd_lcm_is_correct_l792_792283

-- Define the numbers
def a := 15
def b := 75

-- Definitions related to GCD and LCM
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b
def product_gcd_lcm := gcd_ab * lcm_ab

-- Theorem stating the product of GCD and LCM of a and b is 1125
theorem product_gcd_lcm_is_correct : product_gcd_lcm = 1125 := by
  sorry

end product_gcd_lcm_is_correct_l792_792283


namespace vkontakte_membership_l792_792310

variables (M I A P: Prop)

theorem vkontakte_membership :
  (M → (I ∧ A)) → 
  (A ↔ ¬P) → 
  (I ∨ M) → 
  (P ↔ I) → 
  (¬M ∧ I ∧ ¬A ∧ P) := by {
    intro h1 h2 h3 h4,
    -- Insert the rest of the proof here
    sorry
  }

end vkontakte_membership_l792_792310


namespace infinite_colored_complete_graph_contains_monochromatic_subgraph_l792_792158

theorem infinite_colored_complete_graph_contains_monochromatic_subgraph
  (k : ℕ)
  (K_inf : Type)
  (E : K_inf → K_inf → Fin k) :
  ∃ (H_inf : set K_inf), (∀ (u v : K_inf), u ∈ H_inf → v ∈ H_inf → u ≠ v → E u v = E v u) ∧ 
  (∀ u v w ∈ H_inf, E u v = E u w) ∧ 
  ∃ f : ℕ → K_inf, (∀ n, (f n) ∉ H_inf → false) ∧
  (∀ n m : ℕ, n ≠ m → (f n ≠ f m → E (f n) (f m) = E (f m) (f n))) :=
sorry

end infinite_colored_complete_graph_contains_monochromatic_subgraph_l792_792158


namespace tan_11_pi_over_4_l792_792790

theorem tan_11_pi_over_4 : Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Proof is omitted
  sorry

end tan_11_pi_over_4_l792_792790


namespace unique_even_periodic_increasing_l792_792729

theorem unique_even_periodic_increasing :
  ∃ f : ℝ → ℝ, (∀ x, f x = |sin x|) ∧ (∀ x, f (-x) = f x) ∧ (∃ p > 0, ∀ x, f (x + p) = f x ∧ p = π) ∧ (∀ x, (0 < x ∧ x < π/2) → f (x) < f (x + 1)) :=
sorry

end unique_even_periodic_increasing_l792_792729


namespace alpha_values_l792_792154

noncomputable def α := Complex

theorem alpha_values (α : Complex) :
  (α ≠ 1) ∧ 
  (Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1)) ∧ 
  (Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1)) ∧ 
  (Real.cos α.arg = 1 / 2) →
  α = Complex.mk ((-1 + Real.sqrt 33) / 4) (Real.sqrt (3 * (((-1 + Real.sqrt 33) / 4)^2))) ∨ 
  α = Complex.mk ((-1 - Real.sqrt 33) / 4) (Real.sqrt (3 * (((-1 - Real.sqrt 33) / 4)^2))) :=
sorry

end alpha_values_l792_792154


namespace larger_triangle_height_is_correct_l792_792269

-- Declare the data structures and conditions
structure Triangle :=
  (height : ℚ) -- We use rational numbers for exact fractions.

def similar_triangles (T1 T2 : Triangle) := True -- Similarity relation (not used for proof here)

-- The known conditions
def area_ratio := 9 / 25
def small_triangle_height := 5

-- Define the triangles
def T1 : Triangle := { height := small_triangle_height }

noncomputable def larger_triangle_height : ℚ := 
  let k := (25 / 9).sqrt in
  small_triangle_height * k

-- The statement we want to prove
theorem larger_triangle_height_is_correct : 
  similar_triangles T1 { height := larger_triangle_height } →
  area_ratio = (T1.height^2) / (larger_triangle_height^2) →
  larger_triangle_height = 25 / 3 :=
by
  sorry

end larger_triangle_height_is_correct_l792_792269


namespace triangle_AB_length_l792_792969

-- Definitions of conditions
def medial_perpendicular (A B C M N : EuclideanSpace ℝ (Fin 2)) : Prop :=
  let AM := M - A
  let BN := N - B
  AM ⬝ AM = 15 ^ 2 ∧ BN ⬝ BN = 20 ^ 2 ∧ AM ⬝ BN = 0

-- The proof statement
theorem triangle_AB_length 
  (A B C M N : EuclideanSpace ℝ (Fin 2))
  (h : medial_perpendicular A B C M N)
  : dist A B = 16.67 := 
sorry


end triangle_AB_length_l792_792969


namespace intersection_of_sets_l792_792083

theorem intersection_of_sets :
  let M := { x : ℝ | 0 ≤ x ∧ x < 16 }
  let N := { x : ℝ | x ≥ 1/3 }
  M ∩ N = { x : ℝ | 1/3 ≤ x ∧ x < 16 } :=
by
  sorry

end intersection_of_sets_l792_792083


namespace probability_2x_probability_isosceles_l792_792655

def probability_on_graph_of_2x (a b: ℕ) : Prop :=
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 4)

def probability_isosceles_triangle (a b: ℕ) : Prop :=
  (a = 1 ∧ b = 3) ∨ (a = 1 ∧ b = 4) ∨ (a = 1 ∧ b = 5) ∨
  (a = 2 ∧ b = 2) ∨ (a = 2 ∧ b = 4) ∨ (a = 2 ∧ b = 6) ∨
  (a = 3 ∧ b = 1) ∨ (a = 3 ∧ b = 3) ∨ (a = 3 ∧ b = 5) ∨
  (a = 4 ∧ b = 1) ∨ (a = 4 ∧ b = 2) ∨ (a = 4 ∧ b = 4) ∨
  (a = 5 ∧ b = 1) ∨ (a = 5 ∧ b = 3) 

theorem probability_2x : 
  (∑ a b in finset.range 6 \ finset.singleton 0, if probability_on_graph_of_2x a b then 1 else 0) / 36 = 1 / 18 :=
by sorry

theorem probability_isosceles :
  (∑ a b in finset.range 6 \ finset.singleton 0, if probability_isosceles_triangle a b then 1 else 0) / 36 = 7 / 18 :=
by sorry

end probability_2x_probability_isosceles_l792_792655


namespace sqrt3_a_minus_b_range_l792_792120

theorem sqrt3_a_minus_b_range {A B C a b c : ℝ}
  (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (h_sides : a = 2 * sin A ∧ b = 2 * sin B ∧ c = 1)
  (h_condition : a^2 - sqrt 3 * a * b + b^2 = 1) :
  (1 < sqrt 3 * a - b) ∧ (sqrt 3 * a - b < sqrt 3) :=
by 
  sorry

end sqrt3_a_minus_b_range_l792_792120


namespace license_plates_count_l792_792328

-- Definitions from conditions
def num_digits : ℕ := 4
def num_digits_choices : ℕ := 10
def num_letters : ℕ := 3
def num_letters_choices : ℕ := 26

-- Define the blocks and their possible arrangements
def digits_permutations : ℕ := num_digits_choices^num_digits
def letters_permutations : ℕ := num_letters_choices^num_letters
def block_positions : ℕ := 5

-- We need to show that total possible license plates is 878,800,000.
def total_plates : ℕ := digits_permutations * letters_permutations * block_positions

-- The theorem statement
theorem license_plates_count :
  total_plates = 878800000 := by
  sorry

end license_plates_count_l792_792328


namespace num_of_male_students_l792_792606

variables (M : ℕ) (T : ℕ)

def is_average_students (avg_all avg_male avg_female : ℕ) (num_female : ℕ) (total_score : ℕ) :=
  T = M + num_female ∧
  avg_all * T = avg_male * M + avg_female * num_female

theorem num_of_male_students 
  (M : ℕ) (T : ℕ) (avg_all : ℕ) (avg_male : ℕ) (avg_female : ℕ) (num_female : ℕ) 
  (total_score : ℕ) 
  (h : is_average_students avg_all avg_male avg_female num_female total_score)
  (h_avg_all : avg_all = 90)
  (h_avg_male : avg_male = 85)
  (h_avg_female : avg_female = 92)
  (h_num_female : num_female = 20) :
  M = 8 :=
by
  sorry

end num_of_male_students_l792_792606


namespace amount_paid_correct_l792_792886

def costWithoutDiscount (quantity rate : ℝ) := quantity * rate
def discountedCost (cost discountRate : ℝ) := cost - (cost * discountRate / 100)
def taxedCost (cost taxRate : ℝ) := cost + (cost * taxRate / 100)

def totalCostGrapes : ℝ := 
  let cost := costWithoutDiscount 8 70
  let discounted := discountedCost cost 10
  taxedCost discounted 5

def totalCostMangoes : ℝ := 
  let cost := costWithoutDiscount 9 50
  taxedCost cost 8

def totalCostApples : ℝ := 
  let cost := costWithoutDiscount 5 100
  discountedCost cost 15

def totalCostOranges : ℝ := 
  let cost := costWithoutDiscount 5 40
  taxedCost cost 3

def totalAmountPaid : ℝ := totalCostGrapes + totalCostMangoes + totalCostApples + totalCostOranges

theorem amount_paid_correct : totalAmountPaid = 1646.2 := by
  sorry

end amount_paid_correct_l792_792886


namespace part_one_part_two_l792_792845

section part_one

variables {m n : ℝ}

def z1 (m : ℝ) : ℂ := complex.mk m (-2)
def z2 (n : ℝ) : ℂ := complex.mk 1 (-n)

theorem part_one (h1 : m = 0) (h2 : n = 1) : complex.abs (z1 m + z2 n) = real.sqrt 10 :=
by {
  sorry
}

end part_one

section part_two

variables {m n : ℝ}

def z1 (m : ℝ) : ℂ := complex.mk m (-2)
def z2 (n : ℝ) : ℂ := complex.mk 1 (-n)

theorem part_two (h : z1 m = complex.sqr (complex.conj (z2 n))) : m = 0 ∧ n = -1 :=
by {
  sorry
}

end part_two

end part_one_part_two_l792_792845


namespace smallest_period_of_f_max_min_values_of_f_on_interval_l792_792873
open Real

noncomputable def f (x : ℝ) : ℝ := 4 * sin (x - π / 3) * cos x + sqrt 3

theorem smallest_period_of_f :
  ∀ (T : ℝ), 0 < T ∧ (∀ x, f (x + T) = f x) ↔ T = π := by sorry

theorem max_min_values_of_f_on_interval :
  let interval := Icc (-π / 4) (π / 3)
  ∃ (max_x min_x : ℝ), max_x ∈ interval ∧ min_x ∈ interval ∧
                        (∀ x ∈ interval, f x ≤ f max_x) ∧ f max_x = sqrt 3 ∧
                        (∀ x ∈ interval, f min_x ≤ f x) ∧ f min_x = -2 ∧
                        max_x = π / 3 ∧ min_x = -π / 12 := by sorry

end smallest_period_of_f_max_min_values_of_f_on_interval_l792_792873


namespace incorrect_statements_l792_792298

variable (M : Type) (f : M → ℝ) (x : ℝ) (x1 x2 : M)

theorem incorrect_statements :
  ((¬(∃ x1 x2 ∈ M, x1 ≠ x2 ∧ (f x1 - f x2) * (x2 - x1) > 0) 
      ↔ ∀ x1 x2 ∈ M, x1 ≠ x2 → (f x1 - f x2) * (x2 - x1) ≤ 0) = false) ∨
  ((¬(x ≠ 3 → |x| ≠ 3)) = false) ∨
  ((∀ x, (x³ + 2*x - 3 > 0 ∧ ¬(1/(3-x) > 1) → (x ∈ (-∞,-3) ∪ (1,2] ∪ [3,+∞))) = false))
:= sorry

end incorrect_statements_l792_792298


namespace limit_ratio_S_l792_792406

noncomputable def S (a : ℝ) : ℝ :=
  ∫ x in 0..(1 - a), (exp ((1 + a) / (1 - a) * x) - exp x) +
  ∫ x in (1 - a)..1, (exp (2 - x) - exp x)

theorem limit_ratio_S (h : ∀ a, 0 < a ∧ a < 1) : 
  tendsto (fun a => (S a) / a) (𝓝 0) (𝓝 (-2)) :=
sorry

end limit_ratio_S_l792_792406


namespace different_tens_digit_probability_l792_792600

noncomputable def probability_diff_tens_digit : ℚ :=
  1000000 / 15890700

theorem different_tens_digit_probability :
  let selected : Finset ℕ := Finset.range (59 + 1) \ Finset.range 10 in
  let total_combinations := (selected.card).choose 6 in
  let favorable_combinations := 10^6 in
  (favorable_combinations : ℚ) / (total_combinations : ℚ) = probability_diff_tens_digit :=
by
  sorry

end different_tens_digit_probability_l792_792600


namespace minimum_tangent_slope_l792_792862

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - x^2 - (16 / (x - 1))

def f_derivative (x : ℝ) : ℝ := x^2 - 2 * x + (16 / (x - 1)^2)

theorem minimum_tangent_slope : ∃ x > 1, f_derivative x = 7 :=
sorry

end minimum_tangent_slope_l792_792862


namespace correct_subtraction_l792_792300

theorem correct_subtraction (x : ℕ) (h : x - 42 = 50) : x - 24 = 68 :=
  sorry

end correct_subtraction_l792_792300


namespace find_g_2016_l792_792686

variable {R : Type*} [LinearOrderedField R]

def f (x : R) : R

axiom f_1 : f 1 = 1
axiom f_prop1 : ∀ x : R, f (x + 5) ≥ f x + 5
axiom f_prop2 : ∀ x : R, f (x + 1) ≤ f x + 1

def g (x : R) : R := f x + 1 - x

theorem find_g_2016 : g 2016 = 1 := by
  sorry

end find_g_2016_l792_792686


namespace oblong_perimeter_182_l792_792353

variables (l w : ℕ) (x : ℤ)

def is_oblong (l w : ℕ) : Prop :=
l * w = 4624 ∧ l = 4 * x ∧ w = 3 * x

theorem oblong_perimeter_182 (l w x : ℕ) (hlw : is_oblong l w x) : 
  2 * l + 2 * w = 182 :=
by
  sorry

end oblong_perimeter_182_l792_792353


namespace distance_from_plane_to_center_l792_792635

-- Definitions for tangent and trigonometric functions
noncomputable def tan (x : ℝ) : ℝ := sin x / cos x
noncomputable def cot (x : ℝ) : ℝ := 1 / tan x

-- Definitions for the conditions in the problem
def l : ℝ := sorry
def alpha : ℝ := sorry
def beta : ℝ := sorry

-- Function for the distance as per the provided proof
noncomputable def distance := l * cot(π / 4 + alpha / 2) * sqrt(sin(alpha + beta / 2) * sin(alpha - beta / 2)) / cos(beta / 2)

-- The main theorem stating the required distance
theorem distance_from_plane_to_center :
  distance = l * cot(π / 4 + alpha / 2) * sqrt(sin(alpha + beta / 2) * sin(alpha - beta / 2)) / cos(beta / 2) :=
sorry

end distance_from_plane_to_center_l792_792635


namespace complex_number_imaginary_solution_l792_792424

-- Define the conditions and the proof problem in Lean
theorem complex_number_imaginary_solution (a : ℝ) (h : (a - complex.i) / (1 + complex.i) = 0 - (a + 1) / 2 * complex.i) : a = 1 :=
sorry

end complex_number_imaginary_solution_l792_792424


namespace range_of_a_for_four_distinct_real_roots_l792_792040

def f (x a : ℝ) : ℝ := x^2 + a * x

def g (x a : ℝ) : ℝ :=
  if x >= 0 then f x a else 2 * x + a

theorem range_of_a_for_four_distinct_real_roots :
  ∃ a ∈ {(x : ℝ | x < 0) ∪ {x : ℝ | x > 2}}, 
    ∀ x : ℝ, g (f x a) a = 0 → ∃! x : ℝ, g (f x a) a = x :=
sorry

end range_of_a_for_four_distinct_real_roots_l792_792040


namespace w_squared_solution_l792_792375

theorem w_squared_solution (w : ℝ) (h : (w + 15)^2 = (4 * w + 6) * (3 * w + 9)) : w^2 ≈ 8.94 := 
by 
  sorry

end w_squared_solution_l792_792375


namespace smallest_second_term_l792_792245

theorem smallest_second_term (a d : ℕ) (h1 : 5 * a + 10 * d = 95) (h2 : a > 0) (h3 : d > 0) : 
  a + d = 10 :=
sorry

end smallest_second_term_l792_792245


namespace possible_geometric_configurations_l792_792846

-- Definitions for the distinct points
variables {x1 y1 x2 y2 : ℝ}

-- Definitions for points P, Q, R, S, and O
def P := (x1, y1)
def Q := (x2, y2)
def R := (x1 + x2, y1 + y2)
def S := (x2 - x1, y2 - y1)
def O := (0, 0)

-- Statement: Proving the possible geometric configurations for the points
theorem possible_geometric_configurations :
  (∃ (polygon : set (ℝ × ℝ)), polygon = {O, P, Q, R} ∧ polygon forms a parallelogram) ∧
  ((O = R ∧ O = S) → (are_collinear {O, P, Q}) ∧ are_collinear {P, Q, R, S}) ∧
  (∃ (trapezoid : set (ℝ × ℝ)), trapezoid = {O, P, R, S} ∧ trapezoid forms a trapezoid := by
sorry -- proof goes here

end possible_geometric_configurations_l792_792846


namespace minimum_area_triangl_PQR_l792_792986

theorem minimum_area_triangl_PQR (PQ QR PR : ℕ) (hPQ : PQ = 36) (hQR : QR = 38) (hPR : PR = 40) :
  ∃ Y J₁ J₂,
    (Y ∈ set.Ioo 0 QR) ∧
    (J₁ = incenter (triangle PQ Y) PQ Y) ∧
    (J₂ = incenter (triangle PR Y) PR Y) ∧ 
    (triangle PJ₁J₂).area = 152 := 
by 
  sorry

end minimum_area_triangl_PQR_l792_792986


namespace min_value_of_x_squared_plus_y_squared_l792_792081

theorem min_value_of_x_squared_plus_y_squared (x y : ℝ) 
  (h : (x - 2) ^ 2 + (y - 1) ^ 2 = 1) : 
  ∃ x y, (x - 2) ^ 2 + (y - 1) ^ 2 = 1 ∧ x^2 + y^2 = 6 - 2 * real.sqrt 5 :=
by
  sorry

end min_value_of_x_squared_plus_y_squared_l792_792081


namespace price_second_day_is_81_percent_l792_792345

-- Define the original price P (for the sake of clarity in the proof statement)
variable (P : ℝ)

-- Define the reductions
def first_reduction (P : ℝ) : ℝ := P - 0.1 * P
def second_reduction (P : ℝ) : ℝ := first_reduction P - 0.1 * first_reduction P

-- Question translated to Lean statement
theorem price_second_day_is_81_percent (P : ℝ) : 
  (second_reduction P / P) * 100 = 81 := by
  sorry

end price_second_day_is_81_percent_l792_792345


namespace coeff_x2_in_expansion_l792_792612

theorem coeff_x2_in_expansion : 
  (∃ c : ℕ, c * x^2 ∈ (1 + 2 * x)^5) → c = 40 := 
by
  sorry

end coeff_x2_in_expansion_l792_792612


namespace g_n_plus_1_minus_g_n_minus_1_eq_g_n_l792_792174

def g (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((2 + Real.sqrt 7) / 3) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((2 - Real.sqrt 7) / 3) ^ n

theorem g_n_plus_1_minus_g_n_minus_1_eq_g_n (n : ℕ) : g (n + 1) - g (n - 1) = g n := 
  sorry

end g_n_plus_1_minus_g_n_minus_1_eq_g_n_l792_792174


namespace common_divisors_9240_8820_l792_792894

-- Define the prime factorizations given in the problem.
def pf_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def pf_8820 := [(2, 2), (3, 2), (5, 1), (7, 2)]

-- Define a function to calculate the gcd of two numbers given their prime factorizations.
def gcd_factorizations (pf1 pf2 : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
    List.filterMap (λ ⟨p, e1⟩ =>
      match List.lookup p pf2 with
      | some e2 => some (p, min e1 e2)
      | none => none
      end) pf1 

-- Define a function to compute the number of divisors from the prime factorization.
def num_divisors (pf: List (ℕ × ℕ)) : ℕ :=
    pf.foldl (λ acc ⟨_, e⟩ => acc * (e + 1)) 1

-- The Lean statement for the problem
theorem common_divisors_9240_8820 : 
    num_divisors (gcd_factorizations pf_9240 pf_8820) = 24 :=
by
    -- The proof goes here. We include sorry to indicate that the proof is omitted.
    sorry

end common_divisors_9240_8820_l792_792894


namespace range_of_m_l792_792869

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x + α / x + Real.log x

theorem range_of_m (e l : ℝ) (alpha : ℝ) :
  (∀ (α : ℝ), α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 1 ^ 2) → 
  ∀ (x : ℝ), x ∈ Set.Icc l e → f alpha x < m) →
  m ∈ Set.Ioi (1 + 2 * Real.exp 1 ^ 2) := sorry

end range_of_m_l792_792869


namespace ratio_of_volumes_l792_792494

theorem ratio_of_volumes (R : ℝ) (hR : 0 < R) :
  let V_sphere := (4 / 3) * Real.pi * R^3,
      V_cylinder := 2 * Real.pi * R^3,
      V_cone := (2 / 3) * Real.pi * R^3 in
  V_cylinder / V_cone = 3 ∧ (V_cone / V_cone = 1) ∧ (V_sphere / V_cone = 2) :=
by
  sorry

end ratio_of_volumes_l792_792494


namespace units_digit_square_tens_seven_l792_792244

/-- Proving the possible values for the units digit of a number whose square has the digit 7 in the tens place. -/
theorem units_digit_square_tens_seven (a b : ℕ) (hb : b < 10) :
  (let sqr := (10 * a + b)^2 in (sqr / 10) % 10 = 7) → b = 4 ∨ b = 6 :=
sorry

end units_digit_square_tens_seven_l792_792244


namespace Megan_not_lead_actress_l792_792570

-- Define the conditions: total number of plays and lead actress percentage
def totalPlays : ℕ := 100
def leadActressPercentage : ℕ := 80

-- Define what we need to prove: the number of times Megan was not the lead actress
theorem Megan_not_lead_actress (totalPlays: ℕ) (leadActressPercentage: ℕ) : 
  (totalPlays * (100 - leadActressPercentage)) / 100 = 20 :=
by
  -- proof omitted
  sorry

end Megan_not_lead_actress_l792_792570


namespace mahdi_plays_tennis_on_saturday_l792_792565

-- Define the days of the week
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq, Repr

-- Define the sports
inductive Sport
| Soccer | Cricket | Running | Swimming | Tennis
deriving DecidableEq, Repr

-- Define the conditions as hypotheses in Lean
variables (plays : Day → Sport)

-- Mahdi's schedule conditions
axiom soccer_monday : plays Day.Monday = Sport.Soccer
axiom cricket_thursday : plays Day.Thursday = Sport.Cricket
axiom runs_three_days : ∃ d1 d2 d3, plays d1 = Sport.Running ∧ plays d2 = Sport.Running ∧ plays d3 = Sport.Running ∧ 
                      d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
                      (d1 ≠ Day.Monday ∧ d2 ≠ Day.Monday ∧ d3 ≠ Day.Monday) ∧
                      (d1 ≠ Day.Thursday ∧ d2 ≠ Day.Thursday ∧ d3 ≠ Day.Thursday) ∧
                      (∀ d, plays d = Sport.Running → 
                            (plays (next_day d) ≠ Sport.Running))
axiom no_tennis_after : ∀ d, plays (next_day d) = Sport.Tennis → plays d ≠ Sport.Running ∧ plays d ≠ Sport.Swimming

-- Define the next day function
def next_day : Day → Day
| Day.Monday    => Day.Tuesday
| Day.Tuesday   => Day.Wednesday
| Day.Wednesday => Day.Thursday
| Day.Thursday  => Day.Friday
| Day.Friday    => Day.Saturday
| Day.Saturday  => Day.Sunday
| Day.Sunday    => Day.Monday

-- The proof that Mahdi plays tennis on Saturday
theorem mahdi_plays_tennis_on_saturday : plays Day.Saturday = Sport.Tennis :=
by
  sorry

end mahdi_plays_tennis_on_saturday_l792_792565


namespace inequality_abc_ad_bc_bd_cd_l792_792001

theorem inequality_abc_ad_bc_bd_cd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (a * d) + 1 / (b * c) + 1 / (b * d) + 1 / (c * d)) 
  ≤ (3 / 8) * (1 / a + 1 / b + 1 / c + 1 / d) ^ 2 := sorry

end inequality_abc_ad_bc_bd_cd_l792_792001


namespace max_distance_PQ_l792_792022

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem max_distance_PQ (α β : ℝ) : 
  ∃ γ ∈ set.Icc (-1 : ℝ) 1, distance (real.cos α, real.sin α) (real.cos β, real.sin β) = real.sqrt (2 + 2 * γ) :=
begin
  use -1,
  split,
  { norm_num },
  { norm_num }
end

end max_distance_PQ_l792_792022


namespace sum_of_coefficients_l792_792228

theorem sum_of_coefficients :
  let p := (λ x : ℝ, (x - 2)^6 - (x - 1)^7 + (3 * x - 2)^8) in
  p 1 = 2 :=
by
  -- It's understood that the proof will go here
  sorry

end sum_of_coefficients_l792_792228


namespace number_of_valid_pairs_l792_792240

theorem number_of_valid_pairs : 
  ∃ (pairs : Finset (ℤ × ℤ)), pairs.card = 3 ∧ 
    (∀ (p ∈ pairs), ∃ (r k : ℤ), p = (r, k) ∧ 3 * r - 5 * k = 4 ∧ abs (r - k) ≤ 8) := 
by
  sorry

end number_of_valid_pairs_l792_792240


namespace only_possible_m_l792_792380

theorem only_possible_m (m : ℕ) (h : m > 0) :
  (∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℤ, ¬ (p ∣ (n^m - m))) → m = 2 := 
sorry

end only_possible_m_l792_792380


namespace expansion_identity_l792_792361

theorem expansion_identity : 121 + 2 * 11 * 9 + 81 = 400 := by
  sorry

end expansion_identity_l792_792361


namespace find_ellipse_equation_l792_792127

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem find_ellipse_equation :
  (∃ a b c : ℝ, a = 4 ∧ c = 2 * sqrt 2 ∧ b^2 = a^2 - c^2) →
  (∃ x y : ℝ, ellipse_equation 4 (sqrt 8) x y) :=
by 
  intros h,
  obtain ⟨a, b, c, ha, hc, hb⟩ := h,
  use [4, sqrt 8],
  rw ← ha,
  rw ← hb,
  exact sorry

end find_ellipse_equation_l792_792127


namespace trig_identity_example_l792_792767

theorem trig_identity_example :
  256 * (Real.sin (10 * Real.pi / 180)) * (Real.sin (30 * Real.pi / 180)) *
    (Real.sin (50 * Real.pi / 180)) * (Real.sin (70 * Real.pi / 180)) = 16 := by
  sorry

end trig_identity_example_l792_792767


namespace sin_double_angle_ratio_l792_792913

theorem sin_double_angle_ratio (α : ℝ) (h : Real.sin α = 3 * Real.cos α) : 
  Real.sin (2 * α) / (Real.cos α)^2 = 6 :=
by 
  sorry

end sin_double_angle_ratio_l792_792913


namespace two_circles_with_tangents_l792_792225

theorem two_circles_with_tangents
  (a b : ℝ)                -- radii of the circles
  (length_PQ length_AB : ℝ) -- lengths of the tangents PQ and AB
  (h1 : length_PQ = 14)     -- condition: length of PQ is 14
  (h2 : length_AB = 16)     -- condition: length of AB is 16
  (h3 : length_AB^2 + (a - b)^2 = length_PQ^2 + (a + b)^2) -- from the Pythagorean theorem
  : a * b = 15 := 
sorry

end two_circles_with_tangents_l792_792225


namespace exterior_angle_parallel_lines_l792_792134

theorem exterior_angle_parallel_lines
  (m n : ℝ) -- representing the slopes of parallel lines m and n
  (p q : ℝ) -- representing intersection points
  (angle_m_transversal : ∠(m, transversal) = 45)
  (angle_n_transversal : ∠(n, transversal) = 20)
  (angle_n_additional : ∠(n, additional) = 70) :
  ∠(exterior_angle y) = 65 :=
by
  sorry

end exterior_angle_parallel_lines_l792_792134


namespace number_of_chapters_l792_792418

theorem number_of_chapters
    (total_pages : ℕ)
    (pages_per_chapter : ℕ)
    (total_pages_eq : total_pages = 555)
    (pages_per_chapter_eq : pages_per_chapter = 111)
    : total_pages / pages_per_chapter = 5 :=
by
  rw [total_pages_eq, pages_per_chapter_eq]
  norm_num


end number_of_chapters_l792_792418


namespace geometric_sequence_sum_5_l792_792860

theorem geometric_sequence_sum_5 
  (a : ℕ → ℝ) 
  (h_geom : ∃ q, ∀ n, a (n + 1) = a n * q) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  (a 1 * (1 - (2:ℝ)^5) / (1 - (2:ℝ))) = 31 := 
by
  sorry

end geometric_sequence_sum_5_l792_792860


namespace angle_A_eq_angle_B_implies_rectangle_l792_792946

-- Define the quadrilateral ABCD with given properties
structure Quadrilateral (A B C D : Type) :=
  (AD_parallel_BC : Prop)
  (AB_eq_CD : Prop)

-- The conditions given in the problem
variable (A B C D : Type)

def quadrilateral_ABCD (A B C D : Type) : Quadrilateral A B C D :=
{ AD_parallel_BC := AD_parallel_BC A B C D,
  AB_eq_CD := AB_eq_CD A B C D }

-- Define the angles
variable (angle_A angle_B angle_C angle_D : ℝ)

-- The math proof statement
theorem angle_A_eq_angle_B_implies_rectangle 
  (h1 : quadrilateral_ABCD A B C D)
  (h2 : angle_A = angle_B) : 
  angle_A = 90 ∧ angle_B = 90 ∧ angle_C = 90 ∧ angle_D = 90 :=
sorry

end angle_A_eq_angle_B_implies_rectangle_l792_792946


namespace bed_sheet_cutting_problem_l792_792910

theorem bed_sheet_cutting_problem:
  ∀ (bed_sheet_length : ℝ) (time_per_cut : ℝ) (total_time : ℝ),
    bed_sheet_length = 1 →
    time_per_cut = 5 →
    total_time = 245 →
    let number_of_pieces := total_time / time_per_cut in
    let piece_length_cm := (bed_sheet_length * 100) / number_of_pieces in
    piece_length_cm ≈ 2.04 := 
begin
  intros bed_sheet_length time_per_cut total_time,
  intros h1 h2 h3,
  let number_of_pieces := total_time / time_per_cut,
  let piece_length_cm := (bed_sheet_length * 100) / number_of_pieces,
  have hp1: bed_sheet_length = 1 := h1,
  have hp2: time_per_cut = 5 := h2,
  have hp3: total_time = 245 := h3,
  simp only [hp1, hp2, hp3] at *,
-- Here, we assume that piece_length_cm ≈ 2.04 is true
  sorry -- Proof still needed for the calculation confirmation
end

end bed_sheet_cutting_problem_l792_792910


namespace negation_proposition_l792_792451

variable (a : ℝ)

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) :=
by
  sorry

end negation_proposition_l792_792451


namespace part_one_part_two_part_three_l792_792459

noncomputable def quadratic_has_real_roots (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 4 * k * x1^2 - 4 * k * x1 + k + 1 = 0 ∧ 4 * k * x2^2 - 4 * k * x2 + k + 1 = 0

noncomputable def quadratic_sum_of_squares (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 4 * k * x1^2 - 4 * k * x1 + k + 1 = 0 ∧ 4 * k * x2^2 - 4 * k * x2 + k + 1 = 0 ∧ x1^2 + x2^2 = 4

noncomputable def quadratic_ratio_condition (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 4 * k * x1^2 - 4 * k * x1 + k + 1 = 0 ∧ 4 * k * x2^2 - 4 * k * x2 + k + 1 = 0 ∧ (x1 / x2 + x2 / x1 - 2).den = 1

-- Part (1)
theorem part_one (k : ℝ) (h : quadratic_has_real_roots k) : k < 0 := sorry

-- Part (2)
theorem part_two (k : ℝ) (h : quadratic_sum_of_squares k) : k = -1 / 7 := sorry

-- Part (3)
theorem part_three (k : ℝ) (h : quadratic_ratio_condition k) : k = -2 ∨ k = -3 ∨ k = -5 := sorry

end part_one_part_two_part_three_l792_792459


namespace expectation_fish_l792_792498

noncomputable def fish_distribution : ℕ → ℚ → ℚ → ℚ → ℚ :=
  fun N a b c => (a / b) * (1 - (c / (a + b + c) ^ N))

def x_distribution : ℚ := 0.18
def y_distribution : ℚ := 0.02
def other_distribution : ℚ := 0.80
def total_fish : ℕ := 10

theorem expectation_fish :
  fish_distribution total_fish x_distribution y_distribution other_distribution = 1.6461 :=
  by
    sorry

end expectation_fish_l792_792498


namespace find_f_of_one_half_l792_792000

def g (x : ℝ) : ℝ := 1 - 2 * x

noncomputable def f (x : ℝ) : ℝ := (1 - x ^ 2) / x ^ 2

theorem find_f_of_one_half :
  f (g (1 / 2)) = 15 :=
by
  sorry

end find_f_of_one_half_l792_792000


namespace quadrilateral_properties_l792_792509

-- Define the properties of each type of quadrilateral
def isRectangle (Q : Type) : Prop :=
∀ (a b c d : Q), (angle a b c = 90) ∧ (angle b c d = 90) ∧ (angle c d a = 90) ∧ (angle d a b = 90)

-- Define the proposition
def proposition_b (Q : Type) : Prop :=
∀ (q : Q), (angle_eq q 90 90 90 90) → isRectangle Q

theorem quadrilateral_properties : proposition_b Q :=
sorry

end quadrilateral_properties_l792_792509


namespace find_a_parallel_l792_792020

theorem find_a_parallel (a : ℝ) :
  let l1 := λ x y, a * x - 2 * y - 1 = 0,
      l2 := λ x y, 6 * x - 4 * y + 1 = 0 in
  (∀ x y, l1 x y → l2 x y → (a / 6 = 2 / -2)) → a = 3 :=
by
  sorry

end find_a_parallel_l792_792020


namespace speed_ratio_correct_l792_792320

noncomputable def boat_speed_still_water := 12 -- Boat's speed in still water (in mph)
noncomputable def current_speed := 4 -- Current speed of the river (in mph)

-- Calculate the downstream speed
noncomputable def downstream_speed := boat_speed_still_water + current_speed

-- Calculate the upstream speed
noncomputable def upstream_speed := boat_speed_still_water - current_speed

-- Assume a distance for the trip (1 mile each up and down)
noncomputable def distance := 1

-- Calculate time for downstream
noncomputable def time_downstream := distance / downstream_speed

-- Calculate time for upstream
noncomputable def time_upstream := distance / upstream_speed

-- Calculate total time for the round trip
noncomputable def total_time := time_downstream + time_upstream

-- Calculate total distance for the round trip
noncomputable def total_distance := 2 * distance

-- Calculate the average speed for the round trip
noncomputable def avg_speed_trip := total_distance / total_time

-- Calculate the ratio of average speed to speed in still water
noncomputable def speed_ratio := avg_speed_trip / boat_speed_still_water

theorem speed_ratio_correct : speed_ratio = 8/9 := by
  sorry

end speed_ratio_correct_l792_792320


namespace minimum_value_f_range_of_k_l792_792875

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * (Real.exp x) - x + b
def g (x : ℝ) : ℝ := x - Real.log (x + 1)

theorem minimum_value_f (a b : ℝ) :
  (∃ a b : ℝ, f 0 a b = 0 ∧ f' 0 a b = g' 0) → (∀ x : ℝ, f x a b ≥ 0) :=
sorry

theorem range_of_k (k : ℝ) (a b : ℝ) :
  (x ≥ 0) → (∃ a b : ℝ, f x a b ≥ k * g x) ↔ k ≤ 1 :=
sorry

def f' (x : ℝ) (a b : ℝ) : ℝ := a * (Real.exp x) - 1
def g' (x : ℝ) : ℝ := 1 - 1 / (x + 1)

end minimum_value_f_range_of_k_l792_792875


namespace sufficient_not_necessary_condition_l792_792175
open Real

theorem sufficient_not_necessary_condition (m : ℝ) :
  ((m = 0) → ∃ x y : ℝ, (m + 1) * x + (1 - m) * y - 1 = 0 ∧ (m - 1) * x + (2 * m + 1) * y + 4 = 0 ∧ 
  ((m + 1) * (m - 1) + (1 - m) * (2 * m + 1) = 0 ∨ (m = 1 ∨ m = 0))) :=
by sorry

end sufficient_not_necessary_condition_l792_792175


namespace model_silo_height_l792_792142

/-- Define the actual silo height in meters -/
def actual_height : ℝ := 60

/-- Define the actual silo volume in liters -/
def actual_volume : ℝ := 150000

/-- Define the model silo volume in liters -/
def model_volume : ℝ := 0.15

/-- Prove the height of the model silo is 0.6 meters given the conditions -/
theorem model_silo_height : (model_volume / actual_volume).cbrt * actual_height = 0.6 := by
  sorry

end model_silo_height_l792_792142


namespace number_of_integer_solutions_in_circle_l792_792817

theorem number_of_integer_solutions_in_circle :
  let circle_eq := ∀ (x y : ℝ), (x - 3)^2 + (y - 3)^2 ≤ 225 in
  let point_eq := ∀ (x : ℝ), circle_eq x (2 * x) in
  ∃ (S : Finset ℤ), S.card = 13 ∧ ∀ (x : ℤ), x ∈ S ↔ point_eq x := by
  sorry

end number_of_integer_solutions_in_circle_l792_792817


namespace exists_set_S_l792_792837

theorem exists_set_S (n : ℕ) (h : n ≥ 3) :
  ∃ S : Finset ℕ, S.card = 2 * n ∧
    ∀ m : ℕ, 2 ≤ m ∧ m ≤ n →
      ∃ S1 S2 : Finset ℕ,
        S1 ∪ S2 = S ∧ 
        S1 ∩ S2 = ∅ ∧
        S1.card = S2.card ∧
        S1.card = m ∧
        S1.sum = S2.sum :=
sorry

end exists_set_S_l792_792837


namespace third_team_pies_l792_792523

theorem third_team_pies (total first_team second_team : ℕ) (h_total : total = 750) (h_first : first_team = 235) (h_second : second_team = 275) :
  total - (first_team + second_team) = 240 := by
  sorry

end third_team_pies_l792_792523


namespace sarahs_loan_amount_l792_792592

theorem sarahs_loan_amount 
  (down_payment : ℕ := 10000)
  (monthly_payment : ℕ := 600)
  (repayment_years : ℕ := 5)
  (interest_rate : ℚ := 0) : down_payment + (monthly_payment * (12 * repayment_years)) = 46000 :=
by
  sorry

end sarahs_loan_amount_l792_792592


namespace find_valid_pairs_l792_792802

open Nat

def has_zero_digit (n : ℕ) : Prop :=
  ∃ d, 0 ≤ d ∧ d < 10 ∧ (10^d ∣ n) ∧ (n / 10^d) % 10 = 0

def valid_pairs_count (sum : ℕ) : ℕ :=
  (Fin.sum Finset.Ico 1 sum (λ a, if has_zero_digit a ∨ has_zero_digit (sum - a) then 0 else 1))

theorem find_valid_pairs :
  valid_pairs_count 500 = 309 :=
by
  sorry

end find_valid_pairs_l792_792802


namespace vkontakte_status_l792_792312

variables (M I A P : Prop)

theorem vkontakte_status
  (h1 : M → (I ∧ A))
  (h2 : xor A P)
  (h3 : I ∨ M)
  (h4 : P ↔ I) :
  (¬ M ∧ I ∧ ¬ A ∧ P) :=
by
  sorry

end vkontakte_status_l792_792312


namespace volume_of_pyramid_l792_792462

-- Define the edge lengths of the pyramid base and the side edges
variables (base_edge side_edge : ℝ)
-- Define the height of the pyramid
variables (height : ℝ)
-- Define the volume of the pyramid
variables (volume : ℝ)

-- Given conditions
def base_edge_len : base_edge = 2 := by sorry
def side_edge_len : side_edge = real.sqrt 6 := by sorry

-- Define the problem to prove the volume
theorem volume_of_pyramid (h_base: base_edge = 2) (h_side: side_edge = real.sqrt 6): 
  volume = 2 * 2 * 2 / 3 :=
by 
  sorry

end volume_of_pyramid_l792_792462


namespace seven_digit_number_count_l792_792821

open Finset
open Multiset

-- Define the constraints for the digits
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def even_digits : Finset ℕ := {2, 4, 6}
def odd_digits : Finset ℕ := {1, 3, 5, 7}

-- A utility function to check adjacency of any two elements in a list
def are_adjacent (l : List ℕ) (a b : ℕ) : Prop :=
  ∃ (i : ℕ), (i < l.length) ∧ ((l.nth i = some a ∧ l.nth (i + 1) = some b) ∨ (l.nth i = some b ∧ l.nth (i + 1) = some a))

-- Define the property of having no repeated digits
def no_repeated_digits (l : List ℕ) : Prop :=
  l.nodup

-- Define the property of having two adjacent even digits
def adj_even_digits (l : List ℕ) : Prop :=
  ∃ a ∈ even_digits, ∃ b ∈ even_digits, are_adjacent l a b

-- Define the property that not all odd digits are adjacent
def not_all_adj_odd_digits (l : List ℕ) : Prop :=
  ¬ ∀ a1 a2 a3 a4 ∈ odd_digits, (are_adjacent l a1 a2) ∧ (are_adjacent l a2 a3) ∧ (are_adjacent l a3 a4)

-- Define the function counting valid 7-digit numbers
def count_valid_numbers : ℕ :=
  {l : List ℕ // l.length = 7 ∧ no_repeated_digits l ∧ adj_even_digits l ∧ not_all_adj_odd_digits l}.toList.length

theorem seven_digit_number_count : count_valid_numbers = 576 := 
  by sorry

end seven_digit_number_count_l792_792821


namespace nth_equation_l792_792579

theorem nth_equation (n : ℕ) (h : n > 0) : (1 / n) * ((n^2 + 2 * n) / (n + 1)) - (1 / (n + 1)) = 1 :=
by
  sorry

end nth_equation_l792_792579


namespace eccentricity_range_l792_792843

-- Define the given ellipse and hyperbola parameters
def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 12) = 1

def hyperbola_eq (a b x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the eccentricity of the hyperbola
def eccentricity (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 + b^2)) / a

-- Define the condition that the distance from the focus of the ellipse 
-- to the asymptote of the hyperbola is less than sqrt 3
def distance_constraint (a b : ℝ) : Prop :=
  (|(4:ℝ) * b| / Real.sqrt (b^2 + a^2)) < Real.sqrt 3

-- Define the theorem to prove the range of the eccentricity
theorem eccentricity_range (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
  (h_dist_constraint : distance_constraint a b) : 
  1 < eccentricity a b ∧ eccentricity a b < 4 / Real.sqrt 3 :=
by
  sorry

end eccentricity_range_l792_792843


namespace lambda_range_l792_792841

def arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ) : Prop :=
  (∀ n, a n = 2 * n + λ) ∧
  (∀ n, S n = n^2 + (λ + 1) * n)

theorem lambda_range (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ) 
        (h_seq : ∀ n, a n = 2 * n + λ) 
        (h_sum : ∀ n, S n = n^2 + (λ + 1) * n) 
        (h_increasing : ∀ n, n ≥ 5 → S n > S (n - 1)) :
  λ > -12 :=
by
  sorry

end lambda_range_l792_792841


namespace exist_pairs_with_entropy_condition_l792_792979

-- Definitions for entropy and the discrete random variables
noncomputable def H (X : Type) [ProbabilitySpace X] : ℝ := sorry  -- Assume a definition for Shannon entropy

variables {n : ℕ}
variable (X : Fin n → Type) [∀ i, ProbabilitySpace (X i)]

theorem exist_pairs_with_entropy_condition :
  (∃ (pairs : Finset (Fin n × Fin n)), 
    pairs.card ≥ n^2 / 2 ∧
    ∀ ⟨i, j⟩ ∈ pairs, H (X i + X j) ≥ (1 / 3) * (Finset.fold min (H (X 0)) (λ i, H (X i)) Finset.univ))
    := sorry

end exist_pairs_with_entropy_condition_l792_792979


namespace first_year_after_2020_with_digit_sum_4_l792_792102

theorem first_year_after_2020_with_digit_sum_4 :
  ∃ x : ℕ, x > 2020 ∧ (Nat.digits 10 x).sum = 4 ∧ ∀ y : ℕ, y > 2020 ∧ (Nat.digits 10 y).sum = 4 → x ≤ y :=
sorry

end first_year_after_2020_with_digit_sum_4_l792_792102


namespace math_problem_l792_792023

theorem math_problem (x y z : ℝ) 
  (h1 : 2 * x + y + z = 6)
  (h2 : x + 2 * y + z = 7) : 
  5 * x ^ 2 + 8 * x * y + 5 * y ^ 2 = 41 := 
begin
  sorry
end

end math_problem_l792_792023


namespace compound_interest_rate_l792_792346

variable (P r : ℝ)

-- Given conditions
def A (n : ℕ) := P * (1 + r / 100) ^ n

-- Problem Statement
theorem compound_interest_rate :
  (A 2 = 3650) ∧ (A 3 = 4015) → r = 10 :=
by
  sorry

end compound_interest_rate_l792_792346


namespace tiling_modulo_l792_792692

-- Definitions based on conditions
def board_length : ℕ := 7 -- Length of the board
def colors := {Red, Blue, Green} -- The three colors available

-- Define the tiling problem based on the conditions
def valid_tiling (n : ℕ) (c : Set Color) : Prop :=
  -- A tiling is valid if each color is used at least once
  n = board_length ∧ c = colors

-- Theorem statement to prove the problem
theorem tiling_modulo (N : ℕ) (h : valid_tiling board_length colors) : N % 1000 = 917 := 
by sorry

end tiling_modulo_l792_792692


namespace length_of_PS_l792_792135

theorem length_of_PS {P Q R S : ℝ} (hPQ : PQ = 1) (hQR : QR = 2 * PQ) (hRS : RS = 3 * QR) :
  let PS := PQ + QR + RS in 
  PS = 9 :=
by
  -- Definitions of lengths based on given conditions
  have hQR' : QR = 2 * 1 := by
    rw [hPQ]
    
  have hQR'' : QR = 2 := by
    rw [hQR']

  have hRS' : RS = 3 * 2 := by
    rw [hQR'', hRS]

  have hRS'' : RS = 6 := by
    rw [hRS']

  let PS := 1 + 2 + 6

  -- Required theorem
  show PS = 9
  from sorry

end length_of_PS_l792_792135


namespace age_difference_l792_792529

theorem age_difference (john_age father_age mother_age : ℕ) 
    (h1 : john_age * 2 = father_age) 
    (h2 : father_age = mother_age + 4) 
    (h3 : father_age = 40) :
    mother_age - john_age = 16 :=
by
  sorry

end age_difference_l792_792529


namespace set_complement_union_l792_792996

-- Definitions of the sets
def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

-- The statement to prove
theorem set_complement_union : (U \ A) ∪ (U \ B) = {1, 4, 5} :=
by sorry

end set_complement_union_l792_792996


namespace complex_exp_solution_l792_792747

noncomputable def complex_exp :=
  (((Complex.cos (135 * Real.pi / 180)) + Complex.i * (Complex.sin (135 * Real.pi / 180))) ^ 36)

theorem complex_exp_solution :
  complex_exp = (1 / 2) - (Complex.i * (Real.sqrt 3 / 2)) :=
by
  sorry

end complex_exp_solution_l792_792747


namespace total_cost_correct_percent_decrease_correct_l792_792763

noncomputable def lawn_chair_original_price : ℝ := 79.95
noncomputable def outdoor_umbrella_original_price : ℝ := 125.50
noncomputable def patio_table_original_price : ℝ := 240.65
noncomputable def lawn_chair_first_discount : ℝ := 0.20
noncomputable def lawn_chair_second_discount : ℝ := 0.15
noncomputable def outdoor_umbrella_discount : ℝ := 0.25
noncomputable def sales_tax_rate : ℝ := 0.07

noncomputable def discounted_lawn_chair_price : ℝ :=
  let first_discount := lawn_chair_original_price * lawn_chair_first_discount
  let price_after_first_discount := lawn_chair_original_price - first_discount
  let second_discount := price_after_first_discount * lawn_chair_second_discount
  price_after_first_discount - second_discount

noncomputable def discounted_outdoor_umbrella_price : ℝ :=
  outdoor_umbrella_original_price * (1 - outdoor_umbrella_discount)

noncomputable def patio_table_price : ℝ := patio_table_original_price

noncomputable def sales_tax (price : ℝ) : ℝ := price * sales_tax_rate

noncomputable def total_cost_after_discounts_and_taxes : ℝ :=
  discounted_lawn_chair_price +
  discounted_outdoor_umbrella_price + sales_tax(discounted_outdoor_umbrella_price) +
  patio_table_price + sales_tax(patio_table_price)

noncomputable def original_total_price : ℝ :=
  lawn_chair_original_price + outdoor_umbrella_original_price + patio_table_original_price

noncomputable def percent_decrease : ℝ :=
  ((original_total_price - total_cost_after_discounts_and_taxes) / original_total_price) * 100

theorem total_cost_correct :
  total_cost_after_discounts_and_taxes ≈ 412.58 :=
sorry

theorem percent_decrease_correct :
  percent_decrease ≈ 7.51 :=
sorry

end total_cost_correct_percent_decrease_correct_l792_792763


namespace range_of_ab_c2_l792_792419

theorem range_of_ab_c2
  (a b c : ℝ)
  (h₁: -3 < b)
  (h₂: b < a)
  (h₃: a < -1)
  (h₄: -2 < c)
  (h₅: c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
by 
  sorry

end range_of_ab_c2_l792_792419


namespace sum_p_i_p_i_plus_1_l792_792115

def p_even (k : ℕ) : ℚ := 1 / (2 * k + 1)
def p_odd (k : ℕ) : ℚ := 1 / k

noncomputable def p (n : ℕ) : ℚ :=
  if n % 2 = 0 then p_even (n / 2) else p_odd ((n + 1) / 2)

theorem sum_p_i_p_i_plus_1 : ((∑ i in (finset.range 2018).map nat.succ, p i * p (i + 1)) : ℚ) = 1009 / 1010 := sorry

end sum_p_i_p_i_plus_1_l792_792115


namespace proof_problem_l792_792962

def consistent_system (x y : ℕ) : Prop :=
  x + y = 99 ∧ 3 * x + 1 / 3 * y = 97

theorem proof_problem : ∃ (x y : ℕ), consistent_system x y := sorry

end proof_problem_l792_792962


namespace perimeter_DEF_twice_BC_l792_792632

-- Define the points and structures involved in the problem
def GeometryProblem (A B C D H E F : Point) (BC : Line) :=
  ∃ (circumcircle : Circle), 
    A, B, C, D are on circumcircle ∧
    AD is the diameter of circumcircle ∧
    H is the orthocenter of triangle ABC ∧
    (∃ line_parallel_BC : Line, is_parallel line_parallel_BC BC ∧ passes_through line_parallel_BC H) ∧ 
    (E is_on AB) ∧ (F is_on AC) ∧ 
    intersects line_parallel_BC AB at E ∧ intersects line_parallel_BC AC at F ∧
    DE + EF + FD = 2 * length BC

-- State the perimeter proof goal
theorem perimeter_DEF_twice_BC (A B C D H E F : Point) (BC : Line) : 
  GeometryProblem A B C D H E F BC → 
  perimeter (triangle DEF) = 2 * length BC := 
sorry

end perimeter_DEF_twice_BC_l792_792632


namespace find_C_share_l792_792347

constant A B C D E : ℕ
constant Rs : ℝ

-- Assume the investments and time periods
axiom inv_A : A = 4 * B
axiom inv_A_C : A = (3/4) * C
axiom inv_B_D : B = (1/2) * D
axiom inv_D_E : D = 3 * E

-- Investment durations in months
constant duration_A : ℕ := 6
constant duration_B : ℕ := 9
constant duration_C : ℕ := 12
constant duration_D : ℕ := 8
constant duration_E : ℕ := 10

-- Total profit in Rupees
constant total_profit : ℝ := 220000

-- Calculate profit share
noncomputable def share_A := A * duration_A
noncomputable def share_B := B * duration_B
noncomputable def share_C := C * duration_C
noncomputable def share_D := D * duration_D
noncomputable def share_E := E * duration_E

noncomputable def total_share := share_A + share_B + share_C + share_D + share_E

-- Proof statement
theorem find_C_share : total_share = total_profit → share_C = 49116.32 :=
by
  sorry

end find_C_share_l792_792347


namespace find_b_of_f_f_eq_4_l792_792472

def f (x : ℝ) (b : ℝ): ℝ :=
if x < 1 then 2 * x - b else 2^x

theorem find_b_of_f_f_eq_4 (b : ℝ) :
  (f (f (1 / 2) b) b = 4) → b = -1 :=
sorry

end find_b_of_f_f_eq_4_l792_792472


namespace BE_tangent_iff_D_midpoint_BF_l792_792449

structure Circle (P : Type*) :=
(center : P)
(radius : ℝ)

variables {P : Type*} [euclidean_space P]

-- Given points A, B, C on circle O
variables (O : Circle P) (A B C : P)
(hA : A ∈ O) (hB : B ∈ O) (hC : C ∈ O)
-- Circle C with radius CB is tangent to AB at point B
(CircleC : Circle P) (hCB : CircleC.radius = dist C B)
(hTangent : tangent CircleC A B)
-- Line through point A intersects Circle C at points D and E
variables (D E : P) (hLine : line_through A D E)
(hIntersectionD : D ∈ CircleC) (hIntersectionE : E ∈ CircleC)
-- Line BD intersects Circle O a second time at point F
variables (F : P) (hIntersectsF : second_intersection O B D F)

-- Prove BE is tangent to Circle O if and only if D is the midpoint of BF
theorem BE_tangent_iff_D_midpoint_BF :
  (tangent_to_circle BE O) ↔ (midpoint_of_line_segment D B F) :=
by {
  sorry
}

end BE_tangent_iff_D_midpoint_BF_l792_792449


namespace number_of_common_divisors_l792_792905

theorem number_of_common_divisors :
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  let divisors_count := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  gcd_ab = 420 ∧ divisors_count = 24 :=
by
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 420 := sorry
  have h2 : (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 24 := by norm_num
  exact ⟨h1, h2⟩

end number_of_common_divisors_l792_792905


namespace intersection_of_M_and_N_l792_792091

def M : Set ℝ := {x | x ≥ 0 ∧ x < 16}
def N : Set ℝ := {x | x ≥ 1/3}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l792_792091


namespace emily_small_gardens_l792_792385

theorem emily_small_gardens (total_seeds planted_big_garden seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41) 
  (h2 : planted_big_garden = 29) 
  (h3 : seeds_per_small_garden = 4) : 
  (total_seeds - planted_big_garden) / seeds_per_small_garden = 3 := 
by
  sorry

end emily_small_gardens_l792_792385


namespace abs_ab_cd_leq_one_fourth_l792_792993

theorem abs_ab_cd_leq_one_fourth (a b c d : ℝ) (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) (h3 : 0 ≤ d) (h_sum : a + b + c + d = 1) :
  |a * b - c * d| ≤ 1 / 4 :=
sorry

end abs_ab_cd_leq_one_fourth_l792_792993


namespace range_c_div_b_l792_792121

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (A_acute : A < π / 2)
variable (B_acute : B < π / 2)
variable (C_acute : C < π / 2)
variable (triangle_identity : 2 * (Real.sin A) * (a * (Real.cos C) + c * (Real.cos A)) = sqrt 3 * a)

theorem range_c_div_b : ∃ I : Set ℝ, I = Set.Ioc (sqrt 3 / 3) (2 * sqrt 3 / 3) ∧ ∀ c b, (c / b) ∈ I :=
sorry

end range_c_div_b_l792_792121


namespace min_distance_from_point_on_circle_to_line_l792_792835

theorem min_distance_from_point_on_circle_to_line :
  let C := {P : ℝ × ℝ | P.1^2 + P.2^2 = 1}
  let L := {P : ℝ × ℝ | P.1 + sqrt 3 * P.2 - 4 = 0}
  let distance := λ (P : ℝ × ℝ) (L : ℝ × ℝ → Prop), (abs (P.1 + sqrt 3 * P.2 - 4)) / sqrt 4
  ∃ P ∈ C, ∀ Q ∈ C, distance Q L ≥ 1 :=
begin
  sorry
end

end min_distance_from_point_on_circle_to_line_l792_792835


namespace lunks_needed_for_20_apples_l792_792067

-- Definitions based on given conditions
def lunks_to_kunks (lunks : ℕ) : ℕ := (lunks / 4) * 2
def kunks_to_apples (kunks : ℕ) : ℕ := (kunks / 3) * 5

-- The main statement to be proven
theorem lunks_needed_for_20_apples :
  ∃ l : ℕ, (kunks_to_apples (lunks_to_kunks l)) = 20 ∧ l = 24 :=
by
  sorry

end lunks_needed_for_20_apples_l792_792067


namespace product_fraction_simplification_l792_792360

theorem product_fraction_simplification :
  ∏ k in Finset.range 52, (k + 3)/(k + 7) = (1 / 30030) := by
  sorry

end product_fraction_simplification_l792_792360


namespace yoque_borrowed_amount_l792_792299

theorem yoque_borrowed_amount (X : ℝ) (monthly_payment : ℝ) (months : ℕ) (interest_rate : ℝ) (total_repayment : ℝ) :
  monthly_payment = 15 →
  months = 11 →
  interest_rate = 0.10 →
  total_repayment = monthly_payment * months →
  1.10 * X = total_repayment →
  X = 150 :=
by
  intros h1 h2 h3 h4 h5
  rw h1 at h4
  rw h2 at h4
  simp at h4
  rw h4 at h5
  sorry

end yoque_borrowed_amount_l792_792299


namespace min_moves_to_monochromatic_l792_792274

theorem min_moves_to_monochromatic (n m : ℕ) (h_n : n = 98) (h_m : m = 98) : 
  (∃ min_moves : ℕ, (∀ cb : (ℕ × ℕ) → bool, 
    (∀ k l : ℕ, k < n → l < m → cb (k, l) = ((k + l) % 2 = 0)) → 
    (min_moves = n + m))) :=
begin
  use 98,
  sorry
end

end min_moves_to_monochromatic_l792_792274


namespace complete_grid_number_l792_792629

theorem complete_grid_number (a b c d e f g h i : ℕ) :
  (a, b, c, d, e, f, g, h, i) ∈ 
  [(1, 1, 2), (1, 2, 1), (1, 2, 3), (1, 5, 3), (2, 4, 3), (3, 1, 3), (3, 2, 2)] → 
  ((a, b, c, d, e, f, g, h, i) = (1, 2, 3, 1, 2, 1, 3, 2, 2) →
  (524) ∈ [a, b, c, d, e, f, g, h, i] :=
by sorry

end complete_grid_number_l792_792629


namespace sufficient_condition_for_q_l792_792440

theorem sufficient_condition_for_q :
  (∀ x, (|x - 4| ≤ 6 → x ≤ 1 + m)) → (m ≥ 9) :=
begin
  sorry
end

end sufficient_condition_for_q_l792_792440


namespace find_ellipse_eq_l792_792437

-- Definition of the given ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Given conditions
variables (a b : ℝ) (h : a > b) (hb : b > 0) (he : (sqrt 3) / 2 = (sqrt (a^2 - b^2)) / a)

-- Part (Ⅰ): Equation of the ellipse
theorem find_ellipse_eq (hA : ellipse a b 0 1) : ellipse 2 1 x y := sorry

-- Part (Ⅱ): Slope condition
variables (k : ℝ) (hx : ∃ 
-- Slope-related lemma
lemma sum_of_slopes_is_constant (x1 x2 : ℝ) (hx1 hx2 : ∃ x, ellipse 2 1 x (k*x + 2*k - 1)) :
  let y1 := k*x1 + 2*k - 1,
      y2 := k*x2 + 2*k - 1,
      slope_AP := (y1 - 1) / x1,
      slope_AQ := (y2 - 1) / x2 in
  slope_AP + slope_AQ = 1 := sorry

end find_ellipse_eq_l792_792437


namespace real_and_equal_roots_l792_792404

theorem real_and_equal_roots (k : ℝ) : 
  (∃ x : ℝ, (3 * x^2 - k * x + 2 * x + 10) = 0 ∧ 
  (3 * x^2 - k * x + 2 * x + 10) = 0) → 
  (k = 2 - 2 * Real.sqrt 30 ∨ k = -2 - 2 * Real.sqrt 30) := 
by
  sorry

end real_and_equal_roots_l792_792404


namespace replace_floor_cost_l792_792646

-- Define the conditions
def floor_removal_cost : ℝ := 50
def new_floor_cost_per_sqft : ℝ := 1.25
def room_length : ℝ := 8
def room_width : ℝ := 7

-- Define the area of the room
def room_area : ℝ := room_length * room_width

-- Define the cost of the new floor
def new_floor_cost : ℝ := room_area * new_floor_cost_per_sqft

-- Define the total cost to replace the floor
def total_cost : ℝ := floor_removal_cost + new_floor_cost

-- State the proof problem
theorem replace_floor_cost : total_cost = 120 := by
  sorry

end replace_floor_cost_l792_792646


namespace area_of_triangle_AEB_l792_792956

theorem area_of_triangle_AEB (A B C D F G E : Type)
  [rect : IsRectangle A B C D]
  [AB : segment A B = 8]
  [BC : segment B C = 4]
  [DF : segment D F = 2]
  [GC : segment G C = 1]
  [AF_intersects_B_G_at_E : intersects (line A F) (line B G) (point E)]
  : area (triangle A E B) = 16 :=
sorry

end area_of_triangle_AEB_l792_792956


namespace smallest_period_f_triangle_properties_l792_792481

noncomputable def vector_m (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
noncomputable def vector_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1 / 2)
noncomputable def function_f (x : ℝ) : ℝ :=
(vect.mul (vector_m x) + vect.mul (vector_n x)).dot (vector_m x)

theorem smallest_period_f : ∃ T > 0, ∀ x, function_f (x + T) = function_f x ∧ T = Real.pi := by
  sorry

theorem triangle_properties (A B C : ℝ) (a b c S : ℝ) (hm : A.is_acute) 
(ha : a = 2 * Real.sqrt 3) (hc : c = 4) (hA : function_f A = (function_f 0..pi).sup) :
    A = Real.pi / 3 ∧ b = 2 ∧ S = 2 * Real.sqrt 3 := by
  sorry

end smallest_period_f_triangle_properties_l792_792481


namespace sum_of_even_factors_of_360_l792_792665

theorem sum_of_even_factors_of_360 : 
  let n := 360 in
  let prime_factors := (2^3 * 3^2 * 5) in
  n = prime_factors →
  (∑ d in (finset.filter (λ x, (x % 2 = 0)) (finset.divisors n)), d) = 1092 :=
by
  intro n prime_factors h
  have h1 : n = 360 := rfl
  have h2 : (2^3 * 3^2 * 5) = prime_factors := rfl
  sorry

end sum_of_even_factors_of_360_l792_792665


namespace one_fixed_hat_probability_l792_792318

/-- Define a function to calculate the number of derangements for n elements -/
def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangements (n - 1) + derangements (n - 2))

/-- Define a function for the factorial of n -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

/-- The problem statement: Probability that exactly one person picks their own hat is 1/3 -/
theorem one_fixed_hat_probability (n : ℕ) (hn : n = 4) : 
  ((4 * derangements 3) : ℚ) / (factorial 4) = 1 / 3 := 
sorry

end one_fixed_hat_probability_l792_792318


namespace total_surface_area_cylinder_is_6pi_l792_792426

-- Define the cylinder
def cylinder (r h : ℝ) :=
  2 * π * r * h = 4 * π

-- Define the radius of the external tangent sphere
def sphere_radius (r h : ℝ) :=
  sqrt (r^2 + (h/2)^2)

-- Define the minimum volume condition for the sphere
def minimum_sphere_volume (r h : ℝ) :=
  sqrt (r^2 + (1/r^2)) = sqrt 2

-- Define the total surface area formula
def total_surface_area (r h : ℝ) :=
  2 * π * r^2 + 2 * π * r * h

-- State the theorem to prove
theorem total_surface_area_cylinder_is_6pi (r h : ℝ) 
  (H_cylinder : cylinder r h)
  (H_min_volume : minimum_sphere_volume r h) :
  total_surface_area r h = 6 * π :=
begin
  sorry
end

end total_surface_area_cylinder_is_6pi_l792_792426


namespace simplify_expr_at_sqrt6_l792_792208

noncomputable def simplifyExpression (x : ℝ) : ℝ :=
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) + 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2))) /
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) - 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2)))

theorem simplify_expr_at_sqrt6 : simplifyExpression (Real.sqrt 6) = - (Real.sqrt 6) / 2 :=
by
  sorry

end simplify_expr_at_sqrt6_l792_792208


namespace area_problem_a_area_problem_b_area_problem_c_area_problem_d_l792_792363

-- Problem a
theorem area_problem_a : 
  (let 𝛺 := {p : ℝ × ℝ | 3 * p.1 ^ 2 = 25 * p.2 ∧ 5 * p.2 ^ 2 = 9 * p.1} in 
  measure_theory.measure_space.volume.1 𝛺) = 7 :=
sorry

-- Problem b
theorem area_problem_b : 
  (let 𝛺 := {p : ℝ × ℝ | p.1 * p.2 = 4 ∧ p.1 + p.2 = 5} in 
  measure_theory.measure_space.volume.1 𝛺) = 15 / 2 - 8 * Real.log 2 :=
sorry

-- Problem c
theorem area_problem_c :
  (let 𝛺 := {p : ℝ × ℝ | exp p.1 ≤ p.2 ∧ p.2 ≤ exp (2 * p.1) ∧ p.1 ≤ 1} in 
  measure_theory.measure_space.volume.1 𝛺) = (Real.exp 1 - 1) ^ 2 / 2 :=
sorry

-- Problem d
theorem area_problem_d :
  (let 𝛺 := {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ p.1 + 3 * p.2 = 1 ∧ p.1 = p.2 ∧ p.1 + 2 * p.2 = 2} in
  measure_theory.measure_space.volume.1 𝛺) = 11 / 12 :=
sorry

end area_problem_a_area_problem_b_area_problem_c_area_problem_d_l792_792363


namespace minimum_value_quadratic_function_l792_792278

-- Defining the quadratic function y
def quadratic_function (x : ℝ) : ℝ := 4 * x^2 + 8 * x + 16

-- Statement asserting the minimum value of the quadratic function
theorem minimum_value_quadratic_function : ∃ (y_min : ℝ), (∀ x : ℝ, quadratic_function x ≥ y_min) ∧ y_min = 12 :=
by
  -- Here we would normally insert the proof, but we skip it with sorry
  sorry

end minimum_value_quadratic_function_l792_792278


namespace males_below_50_combined_correct_l792_792111

def Branch := {NumEmployees : ℕ // NumEmployees > 0}
def Males (branch : Branch) (percentage : ℝ) : ℕ := (percentage * branch.1).toNat
def Males_at_least_50_years_old (males : ℕ) (percentage : ℝ) : ℕ := (percentage * males).toNat
def Males_below_50_years_old (males males_at_least_50 : ℕ) : ℕ := males - males_at_least_50

def BranchA := (show Branch, from ⟨4500, by linarith⟩)
def BranchB := (show Branch, from ⟨3500, by linarith⟩)
def BranchC := (show Branch, from ⟨2200, by linarith⟩)

noncomputable def Combined_males_below_50 := 
  let males_A := Males BranchA 0.60
  let males_A_50 := Males_at_least_50_years_old males_A 0.40
  let males_below_50_A := Males_below_50_years_old males_A males_A_50

  let males_B := Males BranchB 0.50
  let males_B_50 := Males_at_least_50_years_old males_B 0.55
  let males_below_50_B := Males_below_50_years_old males_B males_B_50

  let males_C := Males BranchC 0.35
  let males_C_50 := Males_at_least_50_years_old males_C 0.70
  let males_below_50_C := Males_below_50_years_old males_C males_C_50

  males_below_50_A + males_below_50_B + males_below_50_C

theorem males_below_50_combined_correct :
  Combined_males_below_50 = 2639 :=
sorry

end males_below_50_combined_correct_l792_792111


namespace non_allergic_children_l792_792107

theorem non_allergic_children (T : ℕ) (h1 : T / 2 = n) (h2 : ∀ m : ℕ, 10 = m) (h3 : ∀ k : ℕ, 10 = k) :
  10 = 10 :=
by
  sorry

end non_allergic_children_l792_792107


namespace cricketer_new_average_l792_792679

variable (A : ℕ) (runs_19th_inning : ℕ) (avg_increase : ℕ)
variable (total_runs_after_18 : ℕ)

theorem cricketer_new_average
  (h1 : runs_19th_inning = 98)
  (h2 : avg_increase = 4)
  (h3 : total_runs_after_18 = 18 * A)
  (h4 : 18 * A + 98 = 19 * (A + 4)) :
  A + 4 = 26 :=
by sorry

end cricketer_new_average_l792_792679


namespace rectangle_of_parallel_sides_and_equal_angles_l792_792950

theorem rectangle_of_parallel_sides_and_equal_angles
  (A B C D : Type)
  (AD_parallel_BC : ∀ x : A, x ∈ AD → is_parallel x BC)
  (AB_eq_CD : AB = CD)
  (angle_A_eq_angle_B : ∠ A = ∠ B) :
  is_rectangle (quadrilateral A B C D) :=
by
  sorry

end rectangle_of_parallel_sides_and_equal_angles_l792_792950


namespace range_BE_CF_l792_792928

-- Definitions and conditions based on the problem statement
variables {A B C E F : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace E] [MetricSpace F]
variables (a : ℝ)

-- Midpoints
def midpoint (x y : A) : A := sorry   -- Assume a definition is provided

-- Midpoint conditions
def E_midpoint_AC (A C E : A) : Prop := midpoint A C = E
def F_midpoint_AB (A B F : A) : Prop := midpoint A B = F

-- Ratio condition
def AB_by_AC (A B C : A) (a : ℝ) : Prop := 3*(∥A - B∥) = 2*(∥A - C∥)

-- Cosine definition
noncomputable def cos (x : ℝ) : ℝ := sorry

-- Definition of BE and CF based on cosine law
noncomputable def BE (A B E : A) : ℝ := sqrt (a^2 + (9/16)*a^2 - (3/2)*a^2*(cos (∠A)))
noncomputable def CF (A C F : A) : ℝ := sqrt ((9/4)*a^2 + (1/4)*a^2 - (3/2)*a^2*(cos (∠A)))

-- Given cosine range
def cos_range (x : ℝ) : Prop := -1 < x ∧ x < 1

-- Main theorem to be proved
theorem range_BE_CF (A B C E F : A) (a : ℝ) (h1 : E_midpoint_AC A C E) (h2 : F_midpoint_AB A B F) (h3 : AB_by_AC A B C a) :
  ∃ (r : ℝ), r = (BE A B E) / (CF A C F) ∧ (1 / 4 < r ∧ r < 7 / 8) := sorry

end range_BE_CF_l792_792928


namespace problem_statement_l792_792880

-- Define the parametric equation of curve C
def curve_C (α : ℝ) : ℝ × ℝ :=
(x, y) where
  x = 2 * cos α
  y = 2 * sin α

-- Define the polar coordinate equation of line l
def line_l (ρ θ : ℝ) : ℝ :=
ρ * (cos θ + 2 * sin θ) - 4 * sqrt 5

-- Rectangular coordinate equations derived from line l's polar form
def line_l_rectangular (x y : ℝ) : Prop :=
x + 2 * y - 4 * sqrt 5 = 0

-- Definitions of points A and B based on line intersections with axes
def point_A : ℝ × ℝ := (4 * sqrt 5, 0)
def point_B : ℝ × ℝ := (0, 2 * sqrt 5)

-- Defining the distance between points A and B
def distance_AB : ℝ := sqrt ((4 * sqrt 5) ^ 2 + (2 * sqrt 5) ^ 2)

-- Defining the maximum distance from point M on C to line l
def max_distance_M_to_l : ℝ := 6

-- Maximum area of ΔMAB
def max_area_ΔMAB : ℝ := 1 / 2 * distance_AB * max_distance_M_to_l

-- Theorem statement to prove the specifications
theorem problem_statement : 
  (∀ α, (curve_C α).fst ^ 2 + (curve_C α).snd ^ 2 = 4) ∧
  (∀ x y, line_l_rectangular x y ↔ x + 2 * y - 4 * sqrt 5 = 0) ∧
  (point_A = (4 * sqrt 5, 0) ∧ point_B = (0, 2 * sqrt 5)) ∧
  (distance_AB = 10) ∧
  (max_area_ΔMAB = 30) :=
by
  sorry

end problem_statement_l792_792880


namespace multiple_of_9_digit_l792_792414

theorem multiple_of_9_digit :
  ∃ d : ℕ, d < 10 ∧ (5 + 6 + 7 + 8 + d) % 9 = 0 ∧ d = 1 :=
by
  sorry

end multiple_of_9_digit_l792_792414


namespace rectangle_condition_l792_792949

variable (A B C D : Type) [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace D]
variable [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
variable (AB AD BC CD : ℝ) (angleA angleB angleC angleD : ℝ)

noncomputable theory

-- Conditions:
variable (parallelogram : (AD = BC) ∧ (AB = CD))
variable (parallel : ∃ l m : Line A, is_parallel l m)
-- Given:
variable (angle_condition : angleA = angleB)

-- Proof goal:
theorem rectangle_condition : (parallel) ∧ (parallelogram) ∧ (angle_condition) → is_rectangle A B C D :=
by
  sorry

end rectangle_condition_l792_792949


namespace melina_age_l792_792636

theorem melina_age (A M : ℕ) (alma_score : ℕ := 40) 
    (h1 : A + M = 2 * alma_score) 
    (h2 : M = 3 * A) : 
    M = 60 :=
by 
  sorry

end melina_age_l792_792636


namespace find_sum_of_x_and_y_l792_792458

noncomputable def imaginary_unit : ℂ := complex.I

theorem find_sum_of_x_and_y (x y : ℝ) (h : (x : ℂ) + (y - 2) * imaginary_unit = 2 / (1 + imaginary_unit)) : x + y = 2 := 
by
  sorry

end find_sum_of_x_and_y_l792_792458


namespace sequence_correct_l792_792011

def seq_formula (n : ℕ) : ℚ := 3/2 + (-1)^n * 11/2

theorem sequence_correct (n : ℕ) :
  (n % 2 = 0 ∧ seq_formula n = 7) ∨ (n % 2 = 1 ∧ seq_formula n = -4) :=
by
  sorry

end sequence_correct_l792_792011


namespace polynomial_has_rational_root_l792_792994

theorem polynomial_has_rational_root (p : ℕ) (hp : Nat.Prime p) (a : ℚ)
  (h : ∃ (f g : ℚ[X]), f.degree ≥ 1 ∧ g.degree ≥ 1 ∧ (X^p - (C a)) = f * g) :
  ∃ r : ℚ, (X^p - (C a)).eval r = 0 :=
by sorry

end polynomial_has_rational_root_l792_792994


namespace find_a_l792_792036

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then 1 - x else a * x

theorem find_a (a : ℝ) : f (-1) a = f 1 a → a = 2 := by
  intro h
  sorry

end find_a_l792_792036


namespace james_vegetable_consumption_l792_792972

def vegetable_consumption_weekdays (asparagus broccoli cauliflower spinach : ℚ) : ℚ :=
  asparagus + broccoli + cauliflower + spinach

def vegetable_consumption_weekend (asparagus broccoli cauliflower other_veg : ℚ) : ℚ :=
  asparagus + broccoli + cauliflower + other_veg

def total_vegetable_consumption (
  wd_asparagus wd_broccoli wd_cauliflower wd_spinach : ℚ)
  (sat_asparagus sat_broccoli sat_cauliflower sat_other : ℚ)
  (sun_asparagus sun_broccoli sun_cauliflower sun_other : ℚ) : ℚ :=
  5 * vegetable_consumption_weekdays wd_asparagus wd_broccoli wd_cauliflower wd_spinach +
  vegetable_consumption_weekend sat_asparagus sat_broccoli sat_cauliflower sat_other +
  vegetable_consumption_weekend sun_asparagus sun_broccoli sun_cauliflower sun_other

theorem james_vegetable_consumption :
  total_vegetable_consumption 0.5 0.75 0.875 0.5 0.3 0.4 0.6 1 0.3 0.4 0.6 0.5 = 17.225 :=
sorry

end james_vegetable_consumption_l792_792972


namespace exists_graph_with_clique_smaller_than_chromatic_l792_792384

open Classical -- To use classical existence

-- Define a graph C_5
noncomputable def C_5 : SimpleGraph (Fin 5) := {
  adj := λ i j, (i.val + 1) % 5 = j.val ∨ (i.val + 4) % 5 = j.val,
  sym := by { intros i j h, cases h; { simp [h] }},
  loopless := by { intro i, simp }
}

-- Define chromatic number function (assuming such a function exists)
noncomputable def chromaticNumber (G : SimpleGraph V) : ℕ :=
sorry -- The actual implementation is omitted

-- Define clique number function (assuming such a function exists)
noncomputable def cliqueNumber (G : SimpleGraph V) : ℕ :=
sorry -- The actual implementation is omitted

theorem exists_graph_with_clique_smaller_than_chromatic :
  ∃ G : SimpleGraph (Fin 5), chromaticNumber C_5 = 3 ∧ cliqueNumber C_5 = 2 :=
begin
  use C_5,
  split,
  { sorry }, -- Proof that chromaticNumber C_5 = 3
  { sorry }  -- Proof that cliqueNumber C_5 = 2
end

end exists_graph_with_clique_smaller_than_chromatic_l792_792384


namespace exists_common_element_l792_792170

variable (S : Fin 2011 → Set ℤ)
variable (h1 : ∀ i, (S i).Nonempty)
variable (h2 : ∀ i j, (S i ∩ S j).Nonempty)

theorem exists_common_element :
  ∃ a : ℤ, ∀ i, a ∈ S i :=
by {
  sorry
}

end exists_common_element_l792_792170


namespace seth_pounds_lost_l792_792205

-- Definitions
def pounds_lost_by_Seth (S : ℝ) : Prop := 
  let total_loss := S + 3 * S + (S + 1.5)
  total_loss = 89

theorem seth_pounds_lost (S : ℝ) : pounds_lost_by_Seth S → S = 17.5 := by
  sorry

end seth_pounds_lost_l792_792205


namespace ratio_decrease_l792_792334

theorem ratio_decrease (r1 s1 r2 s2 : ℕ)
(H1 : r1 = 6 * 10^6)
(H2 : s1 = 20 * 10^6)
(H3 : r2 = 9 * 10^6)
(H4 : s2 = 108 * 10^6) :
  (r1.toReal / s1.toReal - r2.toReal / s2.toReal) * 100 ≈ 21.67 := 
sorry

end ratio_decrease_l792_792334


namespace curve_symmetric_about_y_eq_x_l792_792035

theorem curve_symmetric_about_y_eq_x (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a^2 * x + (1 - a^2) * y - 4 = 0 ↔ y = x) →
  (a = √2 / 2 ∨ a = - √2 / 2) :=
by {
  sorry
}

end curve_symmetric_about_y_eq_x_l792_792035


namespace rook_connected_partition_l792_792180

theorem rook_connected_partition (X : set (ℕ × ℕ)) (hx : rook_connected X) (h100 : |X| = 100) :
  ∃ (pairs : set (ℕ × ℕ) × (ℕ × ℕ)) (H : pairs ⊆ X), ∀ ((a, b) ∈ pairs), (a.1 = b.1 ∨ a.2 = b.2) := 
sorry

-- Definitions used from the conditions:
-- rook_connected : definition of rook-connected should be formally stated in Lean terms.
-- For the sake of this example, we assume it’s already defined adequately.

end rook_connected_partition_l792_792180


namespace different_tens_digit_probability_l792_792599

noncomputable def probability_diff_tens_digit : ℚ :=
  1000000 / 15890700

theorem different_tens_digit_probability :
  let selected : Finset ℕ := Finset.range (59 + 1) \ Finset.range 10 in
  let total_combinations := (selected.card).choose 6 in
  let favorable_combinations := 10^6 in
  (favorable_combinations : ℚ) / (total_combinations : ℚ) = probability_diff_tens_digit :=
by
  sorry

end different_tens_digit_probability_l792_792599


namespace sum_of_first_70_primes_l792_792288

-- Define the function to check if a number is prime
def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → n % m ≠ 0

-- Sum of the first 70 primes
def sum_first_n_primes (n : Nat) : Nat :=
  (List.filter is_prime (List.range (n * 20))).take n |>.sum

theorem sum_of_first_70_primes :
  sum_first_n_primes 70 = S := sorry

end sum_of_first_70_primes_l792_792288


namespace distinct_values_S_l792_792552

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

def S (n : ℤ) : ℂ := ω^n + ω^(-n)

theorem distinct_values_S : ∃ n_values : ℕ, (∀ n : ℤ, S n ∈ {2, 2 * ω.re, 2 * (ω ^ 2).re}) ∧ n_values = 3 := 
sorry

end distinct_values_S_l792_792552


namespace intersection_of_M_and_N_l792_792092

def M : Set ℝ := {x | x ≥ 0 ∧ x < 16}
def N : Set ℝ := {x | x ≥ 1/3}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l792_792092


namespace Megan_not_lead_plays_l792_792567

-- Define the problem's conditions as variables
def total_plays : ℕ := 100
def lead_play_ratio : ℤ := 80

-- Define the proposition we want to prove
theorem Megan_not_lead_plays : 
  (total_plays - (total_plays * lead_play_ratio / 100)) = 20 := 
by sorry

end Megan_not_lead_plays_l792_792567


namespace constant_function_of_inequality_l792_792428

theorem constant_function_of_inequality
  (f : ℝ → ℝ)
  (a : ℝ)
  (ha : 0 < a)
  (h1 : ∀ x : ℝ, 0 < f x ∧ f x ≤ a)
  (h2 : ∀ x y : ℝ, sqrt (f x * f y) ≥ f ((x + y) / 2)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c := sorry

end constant_function_of_inequality_l792_792428


namespace intersection_on_circumcircle_l792_792538

-- Given definitions
variables {A B C G M N : Point}
variables {circumcircle : Circle}
variables {line : Line}

-- Conditions
def is_centroid (G : Point) (A B C : Point) : Prop :=
  sorry  -- Definition of centroid, G, of a triangle ABC

def midpoint (M : Point) (A B : Point) : Prop :=
  sorry  -- Definition of M being the midpoint of AB

def concyclic (A G M N : Point) : Prop :=
  sorry  -- Definition of points A, G, M, and N being concyclic

def perpendicular (line1 line2 : Line) : Prop :=
  sorry  -- Definition of line1 and line2 being perpendicular

def passes_through (line : Line) (P : Point) : Prop :=
  sorry  -- Definition of line passing through point P

def lies_on (P : Point) (circle : Circle) : Prop :=
  sorry  -- Definition of point P lying on a circle

-- Problem to prove
theorem intersection_on_circumcircle
  (h1 : is_centroid G A B C)
  (h2 : midpoint M A B)
  (h3 : midpoint N A C)
  (h4 : concyclic A G M N)
  (line_AG : Line)
  (perp_AG : perpendicular line_AG (line_through A G))
  (line_GBC : Line)
  (perp_GBC : perpendicular line_GBC (line_through G (line_segment B C))) :
  ∃ P, passes_through line_AG P ∧ passes_through line_GBC P ∧ lies_on P circumcircle :=
begin
  sorry  -- Proof not required
end

end intersection_on_circumcircle_l792_792538


namespace seq_general_term_sum_c_n_l792_792024

variable {n : ℕ}
variable {a_n b_n c_n : ℕ → ℤ}

axiom a1 : (a_n 1) = 1
axiom b1 : (b_n 1) = 1
axiom a_seq : ∀ n, a_n n = 1 + (n - 1) * 2
axiom b_seq : ∀ n, b_n n = 2^(n-1)
axiom cond1 : a_n 2 + b_n 3 = a_n 4
axiom cond2 : a_n 3 + b_n 4 = a_n 7

theorem seq_general_term :
  ∀ n, a_n n = 2 * n - 1 ∧ b_n n = 2^(n-1) := by
  sorry

theorem sum_c_n :
  let c_n := λ n, 1 / n * (Finset.range n).sum (λ k, a_n k + b_n k)
  S_n := (Finset.range n).sum c_n
  in S_n = (n-1) * 2^(n + 1) - n * (n + 1) / 2 + 2 := by
  sorry

end seq_general_term_sum_c_n_l792_792024


namespace three_connected_planar_l792_792388

-- Define the planarity of a graph
def is_planar (G : Type) : Prop := sorry  -- placeholder definition

-- Define a 3-connected graph
def is_3_connected (G : Type) : Prop := sorry  -- placeholder definition

-- Define the non-existence of K5 subgraph
def no_K5_subgraph (G : Type) : Prop := sorry  -- placeholder definition

-- Define the non-existence of K3,3 subgraph
def no_K33_subgraph (G : Type) : Prop := sorry  -- placeholder definition

theorem three_connected_planar
  (G : Type) [is_3_connected G] [no_K5_subgraph G] [no_K33_subgraph G] :
  is_planar G := 
sorry

end three_connected_planar_l792_792388


namespace find_a_maximize_revenue_l792_792331

noncomputable def f (x : ℝ) : ℝ := x + 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 6 then (10 * x + 1) / (x + 1)
else -x^2 + a * x - 45

theorem find_a : (g 10 a = 5) → a = 15 := by
  sorry

def S (x : ℝ) (a : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 6 then (10 - x + 2) + ((10 * x + 1) / (x + 1))
else (10 - x + 2) + (-x^2 + a * x - 45)

theorem maximize_revenue (a : ℝ) (h : a = 15) : ∃ x, 0 ≤ x ∧ x ≤ 10 ∧ S x a = 17 :=
by
  sorry

end find_a_maximize_revenue_l792_792331


namespace train_cross_bridge_time_l792_792680

theorem train_cross_bridge_time (l_train l_bridge v_train : ℕ) (h_train : l_train = 140) (h_bridge : l_bridge = 150) (h_speed : v_train = 36) :
  let v_train_mps := v_train * 1000 / 3600 in
  (l_train + l_bridge) / v_train_mps = 29 :=
by
  sorry

end train_cross_bridge_time_l792_792680


namespace min_omega_l792_792872

theorem min_omega (ω : ℝ) (h_pos : ω > 0) (h_overlap : ∃ n : ℤ, (3 * (n : ℝ)) = ω) : ω = 3 :=
by
  -- We need to prove that the minimum value of ω is 3
  have h_min : ∀ n : ℤ, n > 0 → 3 * (n : ℝ) = 3 → ω = 3 := sorry
  exact h_min

end min_omega_l792_792872


namespace value_of_a_and_period_range_of_f_on_interval_l792_792464

noncomputable def a (f : ℝ → ℝ) (x : ℝ) : ℝ := 
  (2 - Sup (set.image f (set.Icc x x)))

theorem value_of_a_and_period (f : ℝ → ℝ) (hx : ∀ x, f x = 4 * real.cos x * real.sin (x + real.pi / 6) - a f x) :
  a f _ = -1 ∧ (∀ y, f (y + real.pi) = f y) := 
  by 
  sorry
  
theorem range_of_f_on_interval
  (f : ℝ → ℝ)
  (hx : ∀ x, f x = 4 * real.cos x * real.sin (x + real.pi / 6) - a f x)
  (hP : a f _ = -1) :
  set.range (λ x, f x) (set.Icc (-5 * real.pi / 12) 0) = set.Icc (-2 : ℝ) 1 :=
  by 
  sorry

end value_of_a_and_period_range_of_f_on_interval_l792_792464


namespace four_or_one_planes_l792_792936

-- Define the conditions
def points_not_collinear (A B C : Point) : Prop :=
  ¬ collinear A B C

def any_three_not_collinear (P Q R S : Point) : Prop :=
  points_not_collinear P Q R ∧ points_not_collinear P Q S ∧ points_not_collinear P R S ∧ points_not_collinear Q R S

-- Formalization of the problem statement
theorem four_or_one_planes {P Q R S : Point} (h : any_three_not_collinear P Q R S) : 
  ∃ planes : set Plane, (planes = {p1}) ∨ (planes = {p1, p2, p3, p4}) := 
sorry

end four_or_one_planes_l792_792936


namespace sum_of_ages_is_14_l792_792977

/-- Kiana has two older twin brothers and the product of their three ages is 72.
    Prove that the sum of their three ages is 14. -/
theorem sum_of_ages_is_14 (kiana_age twin_age : ℕ) (htwins : twin_age > kiana_age) (h_product : kiana_age * twin_age * twin_age = 72) :
  kiana_age + twin_age + twin_age = 14 :=
sorry

end sum_of_ages_is_14_l792_792977


namespace mayoral_election_proof_l792_792933

variable (P Q R S T U M V : ℕ)
variable (VP VQ VR VS VT VU : ℕ)
variable (total_votes : ℕ := 55000)
variable (valid_percentage_sum : Prop := (P + Q + R + S + T + U ≠ 100))
variable (vp_eq : Prop := VP = (P * total_votes) / 100)
variable (vq_eq : Prop := VQ = (Q * total_votes) / 100)
variable (vr_eq : Prop := VR = (R * total_votes) / 100)
variable (vs_eq : Prop := VS = (S * total_votes) / 100)
variable (vt_eq : Prop := VT = (T * total_votes) / 100)
variable (vu_eq : Prop := VU = (U * total_votes) / 100)
variable (votes_sum : Prop := VP + VQ + VR + VS + VT + VU = total_votes)
variable (win_margin : Prop := VP - max(VQ, VR, VS, VT, VU) = M)
variable (min_votes_validity : Prop := VP ≥ V ∧ VQ ≥ V ∧ VR ≥ V ∧ VS ≥ V ∧ VT ≥ V ∧ VU ≥ V)
variable (winning_condition : Prop := P > max(Q, R, S, T, U))

theorem mayoral_election_proof :
  valid_percentage_sum →
  vp_eq →
  vq_eq →
  vr_eq →
  vs_eq →
  vt_eq →
  vu_eq →
  votes_sum →
  win_margin →
  min_votes_validity →
  winning_condition →
  ∃ P Q R S T U M V, 
    P > max(Q, R, S, T, U) ∧ 
    (P + Q + R + S + T + U ≠ 100) ∧
    (VP, VQ, VR, VS, VT, VU ≥ V) ∧ 
    (VP + VQ + VR + VS + VT + VU = total_votes) ∧
    (VP - max(VQ, VR, VS, VT, VU) = M) :=
sorry

end mayoral_election_proof_l792_792933


namespace doves_eggs_l792_792825

theorem doves_eggs (initial_doves total_doves : ℕ) (fraction_hatched : ℚ) (E : ℕ)
  (h_initial_doves : initial_doves = 20)
  (h_total_doves : total_doves = 65)
  (h_fraction_hatched : fraction_hatched = 3/4)
  (h_after_hatching : total_doves = initial_doves + fraction_hatched * E * initial_doves) :
  E = 3 :=
by
  -- The proof would go here.
  sorry

end doves_eggs_l792_792825


namespace first_year_after_2020_with_digit_sum_4_l792_792101

theorem first_year_after_2020_with_digit_sum_4 :
  ∃ x : ℕ, x > 2020 ∧ (Nat.digits 10 x).sum = 4 ∧ ∀ y : ℕ, y > 2020 ∧ (Nat.digits 10 y).sum = 4 → x ≤ y :=
sorry

end first_year_after_2020_with_digit_sum_4_l792_792101


namespace citizen_income_l792_792753

variable {I : ℝ}

def tax_first_40000 := 0.14 * 40000
def tax_above_40000 (I : ℝ) := 0.20 * (I - 40000)
def total_tax (I : ℝ) := tax_first_40000 + tax_above_40000 I

theorem citizen_income (h : total_tax I = 8000) : I = 52000 :=
sorry

end citizen_income_l792_792753


namespace Telegraph_Road_length_is_162_l792_792604

-- Definitions based on the conditions
def meters_to_kilometers (meters : ℕ) : ℕ := meters / 1000
def Pardee_Road_length_meters : ℕ := 12000
def Telegraph_Road_extra_length_kilometers : ℕ := 150

-- The length of Pardee Road in kilometers
def Pardee_Road_length_kilometers : ℕ := meters_to_kilometers Pardee_Road_length_meters

-- Lean statement to prove the length of Telegraph Road in kilometers
theorem Telegraph_Road_length_is_162 :
  Pardee_Road_length_kilometers + Telegraph_Road_extra_length_kilometers = 162 :=
sorry

end Telegraph_Road_length_is_162_l792_792604


namespace inequality_part1_inequality_part2_l792_792474

theorem inequality_part1 (x y : ℝ) (h : y = 2) : 
  (|1 - 1 * x * y| > |1 * x - y|) ↔ (x ∈ set.Ioo (-∞) (-1) ∪ (1, +∞)) :=
sorry

theorem inequality_part2 (k x y : ℝ) (h1 : |x| < 1) (h2 : |y| < 1) : 
  (|1 - k * x * y| > |k * x - y|) → (k ∈ set.Icc (-1) 1) :=
sorry

end inequality_part1_inequality_part2_l792_792474


namespace real_imag_equal_complex_l792_792027

/-- Given i is the imaginary unit, and a is a real number,
if the real part and the imaginary part of the complex number -3i(a+i) are equal,
then a = -1. -/
theorem real_imag_equal_complex (a : ℝ) (i : ℂ) (h_i : i * i = -1) 
    (h_eq : (3 : ℂ) = -(3 : ℂ) * a * i) : a = -1 :=
sorry

end real_imag_equal_complex_l792_792027


namespace max_value_OM_ON_MN_l792_792540

theorem max_value_OM_ON_MN (X O Y P M N : Point)
  (h1 : angle X O Y = π / 2)
  (h2 : PointInsideAngle P X O Y)
  (h3 : dist O P = 1)
  (h4 : angle X O P = π / 6)
  (h5 : LineThroughPointIntersectsRays P X O Y M N)
  : OM + ON - MN = 1 + sqrt 3 - real.sqrt (real.sqrt 12) := 
sorry

end max_value_OM_ON_MN_l792_792540


namespace find_a_l792_792025

theorem find_a (a : ℝ) (h_pos : a > 0)
  (h_eq : ∀ (f g : ℝ → ℝ), (f = λ x => x^2 + 10) → (g = λ x => x^2 - 6) → f (g a) = 14) :
  a = 2 * Real.sqrt 2 ∨ a = 2 :=
by 
  sorry

end find_a_l792_792025


namespace isosceles_triangle_dot_product_range_l792_792017

theorem isosceles_triangle_dot_product_range (O A B : ℝ³) 
  (hOA : ∥O - A∥ = 2) 
  (hOB : ∥O - B∥ = 2) 
  (hineq : ∥(O - A) + (O - B)∥ ≥ sqrt(3) / 3 * ∥(O - A) - (O - B)∥) :
  -2 ≤ (O - A) ⬝ (O - B) ∧ (O - A) ⬝ (O - B) < 4 :=
by sorry

end isosceles_triangle_dot_product_range_l792_792017


namespace concyclic_points_l792_792836

theorem concyclic_points
  (P A B C D Q : Type)
  (circle : Set P)
  (tangent1 tangent2 : P)
  (secant : P)
  (h1 : P ∉ circle)
  (h2 : A ∈ circle)
  (h3 : B ∈ circle)
  (h4 : C ∈ circle ∧ C ≠ P ∧ C ≠ D)
  (h5 : D ∈ circle)
  (h6 : Q ∈ circle)
  (h7 : between P C D)
  (h8 : ∠ DAQ = ∠ PBC)
  (h9 : ∠ DBQ = ∠ PAC) :
  ∃ (circle' : Set P), A ∈ circle' ∧ P ∈ circle' ∧ B ∈ circle' ∧ Q ∈ circle' :=
sorry

end concyclic_points_l792_792836


namespace dot_product_eq_neg_39_l792_792914

noncomputable def a : ℝ := sorry  -- vector 'a'
noncomputable def b : ℝ := sorry  -- vector 'b'

axiom norm_a : ∥a∥ = 5
axiom norm_b : ∥b∥ = 8
axiom angle_ab : real.angle a b = real.pi / 3  -- 60 degrees in radians

theorem dot_product_eq_neg_39 :
  (a + b) ⬝ (a - b) = -39 :=
by sorry

end dot_product_eq_neg_39_l792_792914


namespace range_m_interval_l792_792619

theorem range_m_interval {f : ℝ → ℝ} (h : ∀ x, f x = x^2 - 2 * x + 3) 
  (m : ℝ) :
  (∀ x ∈ set.Icc 0 m, f x ≤ 3) ∧ (∃ x ∈ set.Icc 0 m, f x = 3) →
  (∀ x ∈ set.Icc 0 m, f x ≥ 2) ∧ (∃ x ∈ set.Icc 0 m, f x = 2) →
  m ∈ set.Icc 1 2 :=
by
  sorry

end range_m_interval_l792_792619


namespace fish_population_estimate_l792_792235

theorem fish_population_estimate 
  (caught_first : ℕ) 
  (caught_first_marked : ℕ) 
  (caught_second : ℕ) 
  (caught_second_marked : ℕ) 
  (proportion_eq : (caught_second_marked : ℚ) / caught_second = (caught_first_marked : ℚ) / caught_first) 
  : caught_first * caught_second / caught_second_marked = 750 := 
by 
  sorry

-- Conditions used as definitions in Lean 4
def pond_fish_total (caught_first : ℕ) (caught_second : ℕ) (caught_second_marked : ℕ) : ℚ :=
  (caught_first : ℚ) * (caught_second : ℚ) / (caught_second_marked : ℚ)

-- Example usage of conditions
example : pond_fish_total 30 50 2 = 750 := 
by
  sorry

end fish_population_estimate_l792_792235


namespace conditions_on_k_and_b_l792_792852

theorem conditions_on_k_and_b (k b : ℝ) :
  (∀ x : ℝ, (1 - k) * x + 2 - b > 0) → (k = 1 ∧ b < 2) :=
begin
  intros h,
  sorry
end

end conditions_on_k_and_b_l792_792852


namespace fraction_equation_solution_l792_792818

theorem fraction_equation_solution (x : ℝ) (h : x ≠ 3) : (2 - x) / (x - 3) + 3 = 2 / (3 - x) ↔ x = 5 / 2 := by
  sorry

end fraction_equation_solution_l792_792818


namespace coin_inequality_l792_792152

theorem coin_inequality (m n k : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) (k_pos : 0 < k)
  (coins : Fin k → Fin m × Fin n)
  (r s : Fin k → ℕ) 
  (hr : ∀ i, r i = (Finset.filter (λ j, (coins i).fst = (coins j).fst) (Finset.univ)).card)
  (hs : ∀ i, s i = (Finset.filter (λ j, (coins i).snd = (coins j).snd) (Finset.univ)).card) :
  ∑ i in Finset.univ, 1 / (r i + s i) ≤ (m + n) / 4 :=
by
  sorry

end coin_inequality_l792_792152


namespace min_detectors_1030_l792_792258

noncomputable def min_detectors (grid_size ship_size num_detectors : ℕ) : Prop :=
  ∃ (k : ℕ), k = num_detectors ∧ 
  (∀ (detector_pos : ℕ → ℕ × ℕ),
    (∀ i, i < k → (let (x, y) := detector_pos i in x < grid_size ∧ y < grid_size)) →
    (∀ (ship_pos : ℕ × ℕ),
      let (sx, sy) := ship_pos in
      sx + ship_size ≤ grid_size ∧ sy + ship_size ≤ grid_size →
      ∃ i, i < k ∧ 
      let (dx, dy) := detector_pos i in
      sx ≤ dx ∧ dx < sx + ship_size ∧ sy ≤ dy ∧ dy < sy + ship_size))

theorem min_detectors_1030 : min_detectors 2015 1500 1030 :=
  sorry

end min_detectors_1030_l792_792258


namespace height_of_water_in_cylinder_l792_792352

-- Definitions of the given values
def r_c : ℝ := 15  -- base radius of the cone in cm
def h_c : ℝ := 20  -- height of the cone in cm
def r_cy : ℝ := 30 -- base radius of the cylinder in cm

-- Volume of the cone calculation based on given definitions
def V_cone : ℝ := (1 / 3) * Real.pi * (r_c ^ 2) * h_c

-- Proof that the height of the water in the cylinder equals 1.67 cm
theorem height_of_water_in_cylinder : let V_cylinder := V_cone
                                       let height := V_cylinder / (Real.pi * (r_cy ^ 2))
                                       height = 1.67 :=
by
  sorry

end height_of_water_in_cylinder_l792_792352


namespace calculate_octagon_properties_l792_792714

noncomputable def inscribed_octagon_area_and_radius (side : ℝ) : ℝ × ℝ :=
let octagon_area := 2 * (1 + Real.sqrt 2) * (side / 2) ^ 2 in
let radius := Real.sqrt ((octagon_area) / (2 * (1 + Real.sqrt 2))) in
(octagon_area, radius)

theorem calculate_octagon_properties : (inscribed_octagon_area_and_radius 40) =
  (1600, 21.65) :=
by
  sorry

end calculate_octagon_properties_l792_792714


namespace find_86th_even_number_l792_792272

-- Define the set of available digits
def available_digits : set ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

-- Predicate to check if a number is four digits, uses available digits, and is even
def is_valid_four_digit_even (n : ℕ) : Prop :=
  (1000 ≤ n) ∧ (n < 10000) ∧ (n % 2 = 0) ∧ (∀ d ∈ {n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10}, d ∈ available_digits) ∧
  (∀ d1 d2 ∈ {n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10}, d1 ≠ d2 → d1 ≠ d2)

-- The main problem statement in Lean
theorem find_86th_even_number :
  let sorted_numbers := list.filter is_valid_four_digit_even (list.Ico 1000 10000),
      ascending_numbers := list.sort (≤) sorted_numbers
    in list.nth ascending_numbers 85 = some 2054 :=
sorry

end find_86th_even_number_l792_792272


namespace number_of_common_divisors_l792_792908

theorem number_of_common_divisors :
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  let divisors_count := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  gcd_ab = 420 ∧ divisors_count = 24 :=
by
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 420 := sorry
  have h2 : (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 24 := by norm_num
  exact ⟨h1, h2⟩

end number_of_common_divisors_l792_792908


namespace surface_area_of_sphere_l792_792433

namespace SphereProof

-- Let O be a sphere with radius R, and let A, B, C be points on the sphere such that
-- the distance from the center of the sphere to the plane containing points A, B, and C is sqrt(3) * R / 2
-- and AB = AC = BC - 2 * sqrt(3). Proof that the surface area of the sphere is 64 * pi.

def radius (R : ℝ) (A : ℝ) (B : ℝ) : Prop := (A = B = (B - 2 * Real.sqrt 3))
def dist_to_plane (R : ℝ) : ℝ := (Real.sqrt 3 * R / 2)
def surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

theorem surface_area_of_sphere (R : ℝ) (A B C : ℝ) 
  (h1 : radius R A B)
  (h2 : dist_to_plane R)
  (h3 : A = C)
  (h4 : B - 2 * Real.sqrt 3 = C) :
  surface_area R = 64 * Real.pi := by
  sorry

end SphereProof

end surface_area_of_sphere_l792_792433


namespace jasmine_max_cards_l792_792973

def total_money : ℝ := 12.00
def cost_per_card : ℝ := 1.25
def max_cards (x y : ℝ) : ℕ := ⌊x / y⌋

theorem jasmine_max_cards : max_cards total_money cost_per_card = 9 := by
  sorry

end jasmine_max_cards_l792_792973


namespace transport_stones_impossible_l792_792518

theorem transport_stones_impossible :
  ∀ (stones : ℕ) (stone_weight : ℕ → ℕ) (trucks : ℕ) (truck_capacity : ℕ),
  stones = 50 →
  (∀ i, i < stones → 370 ≤ stone_weight i ∧ stone_weight i ≤ 468) →
  trucks = 7 →
  truck_capacity = 3000 →
  ¬ ∃ (distribution : ℕ → ℕ), (∀ (i : ℕ), i < stones → distribution i < trucks) ∧ 
  (∀ (j : ℕ), j < trucks → ∑ (i : ℕ) in finset.univ.filter (λ k, distribution k = j), stone_weight i ≤ truck_capacity) 
  :=
by
  intros! stones stone_weight trucks truck_capacity h1 h2 h3 h4
  sorry

end transport_stones_impossible_l792_792518


namespace simplify_and_rationalize_denominator_l792_792594

theorem simplify_and_rationalize_denominator :
  (√2 / √5) * (√3 / √7) * (real.cbrt 4 / √6) = (real.cbrt 4 * √35) / 35 := 
by
  sorry

end simplify_and_rationalize_denominator_l792_792594


namespace replace_floor_cost_l792_792647

-- Define the conditions
def floor_removal_cost : ℝ := 50
def new_floor_cost_per_sqft : ℝ := 1.25
def room_length : ℝ := 8
def room_width : ℝ := 7

-- Define the area of the room
def room_area : ℝ := room_length * room_width

-- Define the cost of the new floor
def new_floor_cost : ℝ := room_area * new_floor_cost_per_sqft

-- Define the total cost to replace the floor
def total_cost : ℝ := floor_removal_cost + new_floor_cost

-- State the proof problem
theorem replace_floor_cost : total_cost = 120 := by
  sorry

end replace_floor_cost_l792_792647


namespace jonas_pairs_of_pants_l792_792144

theorem jonas_pairs_of_pants (socks pairs_of_shoes t_shirts new_socks : Nat) (P : Nat) :
  socks = 20 → pairs_of_shoes = 5 → t_shirts = 10 → new_socks = 35 →
  2 * (2 * socks + 2 * pairs_of_shoes + t_shirts + P) = 2 * (2 * socks + 2 * pairs_of_shoes + t_shirts) + 70 →
  P = 5 :=
by
  intros hs hps ht hr htotal
  sorry

end jonas_pairs_of_pants_l792_792144


namespace count_valid_pairs_l792_792805

def contains_zero_digit (n : ℕ) : Prop := 
  ∃ k, 10^k ≤ n ∧ n < 10^(k+1) ∧ (n / 10^k % 10 = 0)

def valid_pair (a b : ℕ) : Prop := 
  a + b = 500 ∧ ¬contains_zero_digit a ∧ ¬contains_zero_digit b

theorem count_valid_pairs : 
  (Finset.card (Finset.filter (λ p : ℕ × ℕ, valid_pair p.1 p.2) 
    (Finset.product (Finset.range 500) (Finset.range 500)))) = 329 := 
sorry

end count_valid_pairs_l792_792805


namespace problem_statement_l792_792963

-- Definitions based on conditions
def position_of_3_in_8_063 := "thousandths"
def representation_of_3_in_8_063 : ℝ := 3 * 0.001
def unit_in_0_48 : ℝ := 0.01

theorem problem_statement :
  (position_of_3_in_8_063 = "thousandths") ∧
  (representation_of_3_in_8_063 = 3 * 0.001) ∧
  (unit_in_0_48 = 0.01) :=
sorry

end problem_statement_l792_792963


namespace sum_digits_largest_N_l792_792544

-- Define the conditions
def is_multiple_of_six (N : ℕ) : Prop := N % 6 = 0

def P (N : ℕ) : ℚ := 
  let favorable_positions := (N + 1) *
    (⌊(1:ℚ) / 3 * N⌋ + 1 + (N - ⌈(2:ℚ) / 3 * N⌉ + 1))
  favorable_positions / (N + 1)

axiom P_6_equals_1 : P 6 = 1
axiom P_large_N : ∀ ε > 0, ∃ N > 0, is_multiple_of_six N ∧ P N ≥ (5/6) - ε

-- Main theorem statement
theorem sum_digits_largest_N : 
  ∃ N : ℕ, is_multiple_of_six N ∧ P N > 3/4 ∧ (N.digits 10).sum = 6 :=
sorry

end sum_digits_largest_N_l792_792544


namespace common_divisors_9240_8820_l792_792901

def prime_factors_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def prime_factors_8820 := [(2, 2), (3, 2), (5, 1), (7, 1), (11, 1)]

def gcd_prime_factors := [(2, 2), (3, 1), (5, 1), (7, 1), (11, 1)]

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.snd + 1)) 1

theorem common_divisors_9240_8820 :
  num_divisors gcd_prime_factors = 32 := by
  sorry

end common_divisors_9240_8820_l792_792901


namespace find_BC_l792_792940

noncomputable def x : ℝ := (Real.sqrt 17 - 1) / 2

variables {A B C D E F : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables [has_distance A] [has_distance B] [has_distance C] [has_distance D] [has_distance E] [has_distance F]

variables {distance_AB distance_AC : ℝ}
variables {isosceles_triangle : A ≠ B → A ≠ C}
variables {circle_through_ADEF : Set ℝ}

-- Conditions
def AB_eq_AC := distance_AB = 1 ∧ distance_AC = 1
def triangle_is_isosceles := isosceles_triangle
def A_on_circle := A ∈ circle_through_ADEF

-- Goal
theorem find_BC
  (h1 : distance_AB = 1)
  (h2 : distance_AC = 1)
  (h3 : isosceles_triangle B C)
  (h4 : A ∈ circle_through_ADEF) :
  ∃ (x : ℝ), x = (Real.sqrt 17 - 1) / 2 := 
begin
  use x,
  sorry,
end

end find_BC_l792_792940


namespace sarah_loan_amount_l792_792591

-- Define the conditions and question as a Lean theorem
theorem sarah_loan_amount (down_payment : ℕ) (monthly_payment : ℕ) (years : ℕ) (months_in_year : ℕ) :
  (down_payment = 10000) →
  (monthly_payment = 600) →
  (years = 5) →
  (months_in_year = 12) →
  down_payment + (monthly_payment * (years * months_in_year)) = 46000 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end sarah_loan_amount_l792_792591


namespace max_Bk_l792_792772

theorem max_Bk (k : ℕ) (h0 : 0 ≤ k) (h1 : k ≤ 2000) : k = 181 ↔ 
  ∀ k' : ℕ, (0 ≤ k' ∧ k' ≤ 2000) → B k ≤ B k' :=
sorry

def B (k : ℕ) : ℝ :=
  if (0 ≤ k ∧ k ≤ 2000) then ((nat.choose 2000 k) : ℝ) * (0.1 ^ k) else 0

end max_Bk_l792_792772


namespace exists_right_triangle_with_given_area_and_hypotenuse_l792_792215

noncomputable def right_triangle_exists (a b : ℕ) : Prop :=
  (∃ u v : ℕ, (a = (u^2 + v^2) ^ (1 / 2)) ∧ (b = (u * v) / 2))

theorem exists_right_triangle_with_given_area_and_hypotenuse
  (a b : ℕ)
  (h1 : ∀ (x : ℤ), is_root (x^2 + ↑a * x - ↑b))
  (h2 : ∀ (x : ℤ), is_root (x^2 - ↑a * x + ↑b)) :
  right_triangle_exists a b :=
sorry

end exists_right_triangle_with_given_area_and_hypotenuse_l792_792215


namespace tony_squat_weight_l792_792649

-- Definitions from conditions
def curl_weight := 90
def military_press_weight := 2 * curl_weight
def squat_weight := 5 * military_press_weight

-- Theorem statement
theorem tony_squat_weight : squat_weight = 900 := by
  sorry

end tony_squat_weight_l792_792649


namespace num_non_congruent_triangles_l792_792639

theorem num_non_congruent_triangles : 
  let lengths := (Finset.range 1010).image (λ n, 2^n : ℕ) in
  let combinations := Finset.powerset (lengths * ⟨3, λ _, true⟩) in
  (combinations.filter (λ t, t.card = 3)).card = 510555 :=
sorry

end num_non_congruent_triangles_l792_792639


namespace smallest_ducks_l792_792576

theorem smallest_ducks :
  ∃ D : ℕ, 
  ∃ C : ℕ, 
  ∃ H : ℕ, 
  (13 * D = 17 * C) ∧
  (11 * H = (6 / 5) * 13 * D) ∧
  (17 * C = (3 / 8) * 11 * H) ∧ 
  (13 * D = 520) :=
by 
  sorry

end smallest_ducks_l792_792576


namespace length_of_plot_is_56_l792_792233

-- Define the conditions
def breadth (x : ℕ) := x
def length (x : ℕ) := x + 12

-- Define cost related conditions
def cost_per_meter := 26.50
def total_cost := 5300

-- Define the perimeter of the rectangular plot
def perimeter (x : ℕ) := 2 * (breadth x + length x)

-- Define a function to calculate the cost based on the perimeter
def fencing_cost (x : ℕ) := perimeter x * cost_per_meter

-- Statement to prove
theorem length_of_plot_is_56 : ∃ x : ℕ, length x = 56 ∧ fencing_cost x = total_cost :=
sorry

end length_of_plot_is_56_l792_792233


namespace combinedBasketballPercentage_l792_792578

-- Definitions for the conditions
def northHighStudents : ℕ := 1800
def northHighBasketballPercent : ℤ := 30

def southAcademyStudents : ℕ := 3000
def southAcademyBasketballPercent : ℤ := 35

-- Theorem statement for the proof problem
theorem combinedBasketballPercentage : 
  let totalStudents := northHighStudents + southAcademyStudents,
      northHighBasketballStudents := (northHighStudents * northHighBasketballPercent) / 100,
      southAcademyBasketballStudents := (southAcademyStudents * southAcademyBasketballPercent) / 100,
      totalBasketballStudents := northHighBasketballStudents + southAcademyBasketballStudents,
      combinedPercentage := (totalBasketballStudents * 100) / totalStudents
  in combinedPercentage = 33 :=
by
  sorry

end combinedBasketballPercentage_l792_792578


namespace inequality_proof_l792_792585

-- Define the inequality problem in Lean 4
theorem inequality_proof (x y : ℝ) (h1 : x ≠ -1) (h2 : y ≠ -1) (h3 : x * y = 1) : 
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 := 
by
  sorry

end inequality_proof_l792_792585


namespace num_perfect_square_factors_of_18000_l792_792057

noncomputable def num_factors_is_perfect_square : Prop :=
  let factors_18000 := 2^3 * 3^2 * 5^3
  let is_a_perfect_square (n : ℕ) := ∃ k : ℕ, k^2 = n
  let factor_form := ∀ a b c : ℕ, (0 ≤ a ∧ a ≤ 3) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (0 ≤ c ∧ c ≤ 3)
  let perfect_square_factors := ∑ a b c : ℕ, (even a) ∧ (even b) ∧ (even c)
  ∀ a b c : ℕ, factor_form a b c → (perfect_square_factors = 2 * 2 * 2)

theorem num_perfect_square_factors_of_18000 : num_factors_is_perfect_square :=
by sorry

end num_perfect_square_factors_of_18000_l792_792057


namespace correct_sum_l792_792293

theorem correct_sum (incorrect_total : ℤ)
    (tens_digit_mistake : ℤ)
    (hundreds_digit_mistake : ℤ)
    (thousands_digit_mistake : ℤ) :
    incorrect_total = 52000 →
    tens_digit_mistake = (8 * 10 - 3 * 10) →
    hundreds_digit_mistake = (4 * 100 - 7 * 100) →
    thousands_digit_mistake = (9 * 1000 - 2 * 1000) →
    (incorrect_total - (tens_digit_mistake + hundreds_digit_mistake + thousands_digit_mistake)) = 42250 :=
by
  assume h1 h2 h3 h4
  sorry

end correct_sum_l792_792293


namespace solve_for_buttons_l792_792219

def number_of_buttons_on_second_shirt (x : ℕ) : Prop :=
  200 * 3 + 200 * x = 1600

theorem solve_for_buttons : ∃ x : ℕ, number_of_buttons_on_second_shirt x ∧ x = 5 := by
  sorry

end solve_for_buttons_l792_792219


namespace rook_connected_pairing_l792_792181

def rook_connected (cells: set (ℤ × ℤ)) : Prop :=
  ∀x y ∈ cells, ∃seq: list (ℤ × ℤ), seq.head = x ∧ seq.last = y ∧ ∀(a b ∈ seq), 
  (a.1 = b.1 ∨ a.2 = b.2)

theorem rook_connected_pairing (cells: set (ℤ × ℤ)) 
  (h_size: cells.size = 100) 
  (h_conn: rook_connected cells): 
  ∃pairs: list (ℤ × ℤ) × (ℤ × ℤ), 
  ∀ (p: (ℤ × ℤ) × (ℤ × ℤ)) ∈ pairs, 
  (p.1.1 = p.2.1 ∨ p.1.2 = p.2.2) ∧
  ∀ (c : (ℤ × ℤ)), c ∈ cells ↔ ∃ (p : (ℤ × ℤ) × (ℤ × ℤ)), c = p.1 ∨ c = p.2 :=
begin
  sorry
end

end rook_connected_pairing_l792_792181


namespace place_value_ratio_l792_792964

theorem place_value_ratio :
  let n := 56842.7093
  ∃ r : ℕ, r = 10000 ∧
  let digit_8_place := 1000
  let digit_7_place := 0.1
  (digit_8_place / digit_7_place) = r := by
sorry

end place_value_ratio_l792_792964


namespace simplify_expression_l792_792670

theorem simplify_expression : 
  (Real.sqrt 2 * 2^(1/2) * 2) + (18 / 3 * 2) - (8^(1/2) * 4) = 16 - 8 * Real.sqrt 2 :=
by 
  sorry  -- proof omitted

end simplify_expression_l792_792670


namespace integral_value_l792_792855

-- Define the conditions
def binomial_constant_term (a : ℝ) : Prop :=
  let term := -20 * a^3 in
  term = -160

-- State the theorem
theorem integral_value (a : ℝ) (h : binomial_constant_term a) : ∫ x in 0..a, (3 * x^2 - 1) = 6 :=
by
  sorry

end integral_value_l792_792855


namespace range_of_k_l792_792029

-- Definitions for the conditions of the problem
structure TriangleABC where
  A : ℝ -- angle A
  AC : ℝ -- side AC
  BC : ℝ -- side BC

-- Conditions for triangle ABC
def conditions : TriangleABC := { A := 60, AC := 6, BC := ?_ }

-- The theorem: proving the given range for k
theorem range_of_k {k : ℝ} (h : conditions.BC = k) 
  (h1 : conditions.A = 60) 
  (h2 : conditions.AC = 6) : 
  3 * Real.sqrt 3 < k ∧ k < 6 :=
  sorry

end range_of_k_l792_792029


namespace every_end_contains_exactly_one_normal_ray_l792_792068

-- Definitions of a normal spanning tree, graph G, end of G, and normal rays
noncomputable def is_normal_spanning_tree (G : Type) (T : Type) : Prop := sorry
noncomputable def is_end_of_G (G : Type) (e : Type) : Prop := sorry
noncomputable def normal_ray (T : Type) (r : Type) : Prop := sorry

-- The actual theorem
theorem every_end_contains_exactly_one_normal_ray (G T : Type) [Graph G] [Tree T]
  (hT : is_normal_spanning_tree G T) : 
  ∀ e, is_end_of_G G e → ∃! r, normal_ray T r :=
sorry

end every_end_contains_exactly_one_normal_ray_l792_792068


namespace replace_floor_cost_l792_792645

def cost_of_removal := 50
def cost_per_sqft := 1.25
def room_length := 8
def room_width := 7

def total_cost_to_replace_floor : ℝ :=
  cost_of_removal + (cost_per_sqft * (room_length * room_width))

theorem replace_floor_cost :
  total_cost_to_replace_floor = 120 :=
by
  sorry

end replace_floor_cost_l792_792645


namespace larger_number_is_correct_l792_792305

noncomputable def find_larger_number : ℕ :=
  let L := 1611
  in L

theorem larger_number_is_correct (L S : ℕ)
    (h1 : L - S = 1345)
    (h2 : L = 6 * S + 15) : L = 1611 := 
by {
  sorry
}

end larger_number_is_correct_l792_792305


namespace sequence_value_2_l792_792935

/-- 
Given the following sequence:
1 = 6
3 = 18
4 = 24
5 = 30

The sequence follows the pattern that for all n ≠ 6, n is mapped to n * 6.
Prove that the value of the 2nd term in the sequence is 12.
-/

theorem sequence_value_2 (a : ℕ → ℕ) 
  (h1 : a 1 = 6) 
  (h3 : a 3 = 18) 
  (h4 : a 4 = 24) 
  (h5 : a 5 = 30) 
  (h_pattern : ∀ n, n ≠ 6 → a n = n * 6) :
  a 2 = 12 :=
by
  sorry

end sequence_value_2_l792_792935


namespace quadrilateral_area_l792_792425

def circle (a : ℝ) := { p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - 1)^2 = 4 }
def line1 (a : ℝ) := { p : ℝ × ℝ | p.1 + 2 * p.2 = a + 2 }
def line2 (a : ℝ) := { p : ℝ × ℝ | 2 * p.1 - p.2 = 2 * a - 1 }

theorem quadrilateral_area (a : ℝ) :
  let A := (0, 0), B := (0, 0), C := (0, 0), D := (0, 0),  -- Points will be determined by intersections
  ABCD := {A, B, C, D} in
  (∃ A ∈ line1 a ∩ circle a, ∃ B ∈ line2 a ∩ circle a, ∃ C ∈ line1 a ∩ circle a ∧ C ≠ A, ∃ D ∈ line2 a ∩ circle a ∧ D ≠ B) →
  set.is_square ABCD →
  4 := 8 :=
by
  sorry

end quadrilateral_area_l792_792425


namespace probability_three_red_chips_first_l792_792117

theorem probability_three_red_chips_first (red_chips green_chips : ℕ) 
    (draws : list (color)) (draw_without_replacement : ∀ draw_without_replacement : list (color),
    ∃ s : list (color), 
    s.length = draws.length ∧ 
    (∃ n : ℕ, starts_under n red_chips ∨ starts_under n green_chips)) : 
    (draws.count color.red = 3 → probs (draws) = 3 / 7) :=
sorry

end probability_three_red_chips_first_l792_792117


namespace sin_theta_sum_numerator_denominator_l792_792930

theorem sin_theta_sum_numerator_denominator :
  ∀ (θ φ : ℝ), 
  ∀ (sin : ℝ → ℝ),
  (∃ (sin_n q : ℚ), sin θ = sin_n ∧ sin_n.denom ≠ 0 ∧ sin_n.denom < q ∧ (∀ x, sin x = sin_n.toReal) ∧ (7θ +1φ < π ) )  ∧
  sorry -> 
  -- given conditions
  (∃ (sin_theta : ℚ), sin θ = sin_theta ∧ sin_theta.isPositive) ∧ 
  parallel_chords ∧
  parallel_chords_length 5 θ ∧ 
  parallel_chords_length 12 φ ∧ 
  parallel_chords_length 13 (θ + φ) ∧ (θ + φ < π)  →

  -- proving statement
  (let fraction := sin θ in 
   let a := fraction.num in
   let b := fraction.denom in
   a + b = 18) := 
sorry

end sin_theta_sum_numerator_denominator_l792_792930


namespace equation1_solution_equation2_solution_l792_792597

theorem equation1_solution (x : ℝ) : 
  x^2 + 4*x - 1 = 0 → x = sqrt 5 - 2 ∨ x = -sqrt 5 - 2 := 
by sorry

theorem equation2_solution (x : ℝ) :
  (x-1)^2 = 3*(x-1) → x = 1 ∨ x = 4 :=
by sorry

end equation1_solution_equation2_solution_l792_792597


namespace hyperbola_eccentricity_two_l792_792878

noncomputable def semi_focal_distance (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def circle (c x y : ℝ) : Prop := (x - c)^2 + y^2 = c^2
def focal_points (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := semi_focal_distance a b in ((-c, 0), (c, 0))
def eccentricity (a b : ℝ) : ℝ := sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (x y : ℝ)
    (h_hyperbola : hyperbola a b x y) (h_circle : ∃l, (circle (semi_focal_distance a b) x y)) :
    eccentricity a b = 2 :=
sorry

end hyperbola_eccentricity_two_l792_792878


namespace tan_eleven_pi_over_four_eq_neg_one_l792_792784

noncomputable def tan_of_eleven_pi_over_four : Real := 
  let to_degrees (x : Real) : Real := x * 180 / Real.pi
  let angle := to_degrees (11 * Real.pi / 4)
  let simplified := angle - 360 * Real.floor (angle / 360)
  if simplified < 0 then
    simplified := simplified + 360
  if simplified = 135 then -1
  else
    undefined

theorem tan_eleven_pi_over_four_eq_neg_one :
  tan (11 * Real.pi / 4) = -1 := 
by
  sorry

end tan_eleven_pi_over_four_eq_neg_one_l792_792784


namespace num_six_digit_palindromes_l792_792656

-- Define the set of allowed digits
def allowed_digits : Finset ℕ := {6, 7, 8, 9}

-- Define the problem stating the number of six-digit palindromic integers
theorem num_six_digit_palindromes: 
  (∑ a in allowed_digits, ∑ b in allowed_digits, ∑ c in allowed_digits, 1) = 64 :=
by {
  sorry
}

end num_six_digit_palindromes_l792_792656


namespace largest_B_181_l792_792769

noncomputable def binom (n k : ℕ) : ℚ := Nat.choose n k
def B (n k : ℕ) (p : ℚ) := binom n k * p^k

theorem largest_B_181 : ∃ k, B 2000 181 (1 / 10) = arg_max k (B 2000 k (1 / 10)) where
  arg_max (k : ℕ) (f : ℕ → ℚ) := k ≤ 2000 ∧ ∀ j, j ≤ 2000 → f j ≤ f k := sorry

end largest_B_181_l792_792769


namespace problem_solution_l792_792018

variable (f g : ℝ → ℝ)
variable (a b x : ℝ)
variable (h : ∀ x ∈ set.Icc a b, deriv f x < deriv g x)

theorem problem_solution : f x - f b ≥ g x - g b := 
by sorry

end problem_solution_l792_792018


namespace circumradius_ABC_l792_792138

noncomputable def circumradius (a : ℝ) (A : ℝ) : ℝ :=
a / (2 * Real.sin A)

theorem circumradius_ABC :
  ∀ (a : ℝ), a = 4 →  (circumradius a (Real.pi / 12) = 2 * (Real.sqrt 6 + Real.sqrt 2) ∨ circumradius a (Real.pi / 12) = 4 * Real.sqrt (2 + Real.sqrt 3)) :=
by
  intro a h
  rw [h]
  rw [Real.sin_pi_div_12]
  sorry

end circumradius_ABC_l792_792138


namespace evaluate_f_x_l792_792493

def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 2 * x^2 + 4 * x

theorem evaluate_f_x : f 3 - f (-3) = 672 :=
by
  -- Proof omitted
  sorry

end evaluate_f_x_l792_792493


namespace ellipse_and_line_problem_l792_792863

noncomputable def ellipse_equation (a : ℝ) : Prop := 
  (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1) ↔ (x^2 / a^2 + y^2 = 1))

theorem ellipse_and_line_problem :
  ∀ a : ℝ, (a > 1) ∧ (∀ x y : ℝ, (x - a * y - a) / Real.sqrt (a^2 + 1) = ℝ.sqrt 3 / 2) →
  (ellipse_equation (ℝ.sqrt 3) → 
   (∀ k : ℝ, (∃ x1 y1 x2 y2 : ℝ, 
     (x1 = 0) ∧ 
     (x2 = -6 * k / (1 + 3 * k ^ 2)) ∧ 
     (y1 = 1) ∧ 
     (y2 = (1 - 3 * k ^ 2) / (1 + 3 * k ^ 2)) ∧ 
     (x1 + 1) * (x2 + 1) + y1 * y2 = 0 ∧ 
     ∃ M : ℝ × ℝ, M = (-1, 0))
     → k = 1 / 3)))
    := 
  sorry

end ellipse_and_line_problem_l792_792863


namespace area_of_triangle_AEB_is_10_l792_792953

noncomputable def area_of_triangle_AEB : Prop :=
  let AB := 8
  let BC := 4
  let DF := 2
  let GC := 1
  let area_AEB := (1/2 : ℝ) * AB * ((5 * BC) / 8) in
  area_AEB = 10

theorem area_of_triangle_AEB_is_10 :
  area_of_triangle_AEB :=
by
  sorry

end area_of_triangle_AEB_is_10_l792_792953


namespace ellipse_equation_product_of_slopes_l792_792014

noncomputable def ellipse (a b x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_equation (a b : ℝ) (ha : a > b) (hb : b > 0) (h_focus : a^2 - b^2 = 4) (h_point : (sqrt 2)^2 / a^2 + (sqrt 3)^2 / b^2 = 1) :
  a^2 = 8 ∧ b^2 = 4 :=
sorry

theorem product_of_slopes (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0)
  (midpoint_x midpoint_y : ℝ)
  (h_midpoint : midpoint_x = -2 * k * b / (1 + 2 * k^2) ∧ midpoint_y = b / (1 + 2 * k^2)) :
  let slope_OM : ℝ := midpoint_y / midpoint_x,
      slope_l : ℝ := k
  in slope_OM * slope_l = -1 / 2 :=
sorry

end ellipse_equation_product_of_slopes_l792_792014


namespace eccentricity_of_hyperbola_l792_792877

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (b * c) / (Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 * c) / 3) : ℝ :=
  (3 * Real.sqrt 7) / 7

-- Ensure the function returns the correct eccentricity
theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (b * c) / (Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 * c) / 3) : hyperbola_eccentricity a b c ha hb h = (3 * Real.sqrt 7) / 7 :=
sorry

end eccentricity_of_hyperbola_l792_792877


namespace pyramid_dihedral_angle_l792_792199

-- Define the conditions
variable (O A B C D : Type) [has_coe_to_fun A B C D]   -- Points O, A, B, C, D
variable (s a : ℝ)  -- Edge lengths of the pyramid
variable (θ : ℝ)  -- Dihedral angle
variable h_base_square : is_square_base O A B C D
variable h_edges_congruent : congruent_edges O A B C D s
variable angle_AOB : angle A O B = 60

-- Define cosine of the dihedral angle
def cos_dihedral_angle (θ : ℝ) : ℝ := -1 + sqrt 0

-- The final proof statement
theorem pyramid_dihedral_angle (O A B C D : Type) [has_coe_to_fun A B C D] (s a θ : ℝ) 
  (h_base_square : is_square_base O A B C D) 
  (h_edges_congruent : congruent_edges O A B C D s) 
  (angle_AOB : angle A O B = 60) :
  cos_dihedral_angle θ = -1 + sqrt 0 := 
sorry

end pyramid_dihedral_angle_l792_792199


namespace first_year_after_2020_with_digit_sum_4_l792_792100

theorem first_year_after_2020_with_digit_sum_4 :
  ∃ x : ℕ, x > 2020 ∧ (Nat.digits 10 x).sum = 4 ∧ ∀ y : ℕ, y > 2020 ∧ (Nat.digits 10 y).sum = 4 → x ≤ y :=
sorry

end first_year_after_2020_with_digit_sum_4_l792_792100


namespace ellipse_properties_l792_792864

noncomputable def ellipse_eq (a b : ℝ) (h : a > b) : Prop :=
  ∀ x y : ℝ, (y^2 / a^2 + x^2 / b^2 = 1)

noncomputable def circle_tangent_condition (b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 + y^2 = b^2) ∧ (∃ l : ℝ, l > 0 ∧ l*x - l*y + sqrt 2 = 0)

noncomputable def maximize_area (a b : ℝ) (k : ℝ) : Prop :=
  k > 0 ∧ ∀ x y : ℝ, (x^2 + y^2 / 4 = 1) → (∃ x1 x2 : ℝ, (k * x1 = y) ∧ (k * x2 = y) ∧ (x2 = -x1) ∧ (2 * sqrt ((4 + k))) / sqrt (k^2) <= 2 * sqrt 2)

theorem ellipse_properties
  (e : ℝ) (a b : ℝ) (h : a > b) (he : e = sqrt 3 / 2)
  (c_eq : a^2 = 4 * b^2)
  (circle_tangent : circle_tangent_condition b) :
  ellipse_eq a b h ∧
  (maximize_area a b 2 → maximize_area a b k) :=
begin
  sorry
end

end ellipse_properties_l792_792864


namespace max_value_of_expression_l792_792444

noncomputable def max_expression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  (finset.univ.sum (λ i => (1 - x i).sqrt)) / ((finset.univ.sum (λ i => (x i)⁻¹)).sqrt)

theorem max_value_of_expression {n : ℕ} (h : n > 0) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i ∧ x i < 1) : 
  max_expression n x ≤ (n.sqrt / 2) :=
sorry

end max_value_of_expression_l792_792444


namespace part_a_part_a_rev_l792_792556

variable (x y : ℝ)

theorem part_a (hx : x > 0) (hy : y > 0) : x + y > |x - y| :=
sorry

theorem part_a_rev (h : x + y > |x - y|) : x > 0 ∧ y > 0 :=
sorry

end part_a_part_a_rev_l792_792556


namespace merchant_selling_price_l792_792706

def mixedSaltPrice (weight1 weight2 : ℕ) (price1 price2 profit : ℝ) : ℝ :=
  let total_cost := (↑weight1 * price1) + (↑weight2 * price2)
  let total_weight := (weight1 + weight2 : ℝ)
  let total_selling_price := total_cost + (profit * total_cost)
  total_selling_price / total_weight

theorem merchant_selling_price : 
  mixedSaltPrice 20 40 0.50 0.35 0.20 = 0.48 :=
by sorry

end merchant_selling_price_l792_792706


namespace count_solutions_4sin2theta_3cos_theta_eq_2_l792_792065

theorem count_solutions_4sin2theta_3cos_theta_eq_2 :
  ∃ n, (∀ θ, 0 < θ ∧ θ ≤ 2 * Real.pi → 4 * Real.sin (2 * θ) + 3 * Real.cos θ = 2 → θ ∈ set.range (λ i, i)) ∧ n = 8 :=
by
  sorry

end count_solutions_4sin2theta_3cos_theta_eq_2_l792_792065


namespace rectangular_prism_diagonals_l792_792713

structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)
  (length : ℝ)
  (height : ℝ)
  (width : ℝ)
  (length_ne_height : length ≠ height)
  (height_ne_width : height ≠ width)
  (width_ne_length : width ≠ length)

def diagonals (rp : RectangularPrism) : ℕ :=
  let face_diagonals := 12
  let space_diagonals := 4
  face_diagonals + space_diagonals

theorem rectangular_prism_diagonals (rp : RectangularPrism) :
  rp.faces = 6 →
  rp.edges = 12 →
  rp.vertices = 8 →
  diagonals rp = 16 ∧ 4 = 4 :=
by
  intros
  sorry

end rectangular_prism_diagonals_l792_792713


namespace number_of_people_l792_792327

-- Conditions
def cost_oysters : ℤ := 3 * 15
def cost_shrimp : ℤ := 2 * 14
def cost_clams : ℤ := 2 * 135 / 10  -- Using integers for better precision
def total_cost : ℤ := cost_oysters + cost_shrimp + cost_clams
def amount_owed_each_person : ℤ := 25

-- Goal
theorem number_of_people (number_of_people : ℤ) : total_cost = number_of_people * amount_owed_each_person → number_of_people = 4 := by
  -- Proof to be completed here.
  sorry

end number_of_people_l792_792327


namespace largest_prime_factor_l792_792660

lemma largest_prime_factor_1386 : 
  ∀ (p : ℕ), p ∈ ({2, 3, 7, 11} : finset ℕ) → 
  is_prime_factor p 1386 → 
  p ≤ 11 :=
by 
  sorry

def is_prime_factor (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

theorem largest_prime_factor (p : ℕ) (h_1386_factors : 1386 = 2 * 3^2 * 7 * 11) : p ∣ 1386 → p.prime → p ≤ 11 :=
by {
  intro h_div p_prime,
  rw h_1386_factors at h_div,
  cases h_div; sorry
}

end largest_prime_factor_l792_792660


namespace length_of_BC_l792_792515

noncomputable def triangle_side_lengths (A B C H : ℝ) : Prop := 
  A = 2 * real.sqrt 3 ∧ 
  B = 2 ∧ 
  H = real.sqrt 3 →
  C = 4

-- The main theorem stating the problem and proof requirement
theorem length_of_BC (A B C H : ℝ) (h₁ : A = 2 * real.sqrt 3) (h₂ : B = 2) (h₃ : H = real.sqrt 3) : 
  triangle_side_lengths A B C H := 
by 
  split; exact ⟨h₁, h₂, h₃, sorry⟩

end length_of_BC_l792_792515


namespace common_divisors_9240_8820_l792_792902

def prime_factors_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def prime_factors_8820 := [(2, 2), (3, 2), (5, 1), (7, 1), (11, 1)]

def gcd_prime_factors := [(2, 2), (3, 1), (5, 1), (7, 1), (11, 1)]

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.snd + 1)) 1

theorem common_divisors_9240_8820 :
  num_divisors gcd_prime_factors = 32 := by
  sorry

end common_divisors_9240_8820_l792_792902


namespace dinitrogen_monoxide_molecular_weight_l792_792740

def atomic_weight_N : Real := 14.01
def atomic_weight_O : Real := 16.00

def chemical_formula_N2O_weight : Real :=
  (2 * atomic_weight_N) + (1 * atomic_weight_O)

theorem dinitrogen_monoxide_molecular_weight :
  chemical_formula_N2O_weight = 44.02 :=
by
  sorry

end dinitrogen_monoxide_molecular_weight_l792_792740


namespace tom_apples_left_l792_792263

theorem tom_apples_left (slices_per_apple : ℕ) (apples : ℕ) (fraction_given_to_jerry : ℚ) :
  slices_per_apple = 8 →
  apples = 2 →
  fraction_given_to_jerry = 3 / 8 →
  (let total_slices := slices_per_apple * apples in
   let slices_given_to_jerry := total_slices * fraction_given_to_jerry in
   let slices_after_giving := total_slices - slices_given_to_jerry.toNat in
   let slices_left := slices_after_giving / 2 in
   slices_left) = 5 :=
by
  intros
  sorry

end tom_apples_left_l792_792263


namespace range_of_m_condition_l792_792858

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem range_of_m_condition (m : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f(x) = x^3) ∧ f(2) = 8 ∧ (∀ x, f(m * x^2) + 8 * f(4 - 3 * x) ≥ 0)) ↔ m ≥ 9 / 8 :=
by
  sorry

end range_of_m_condition_l792_792858


namespace number_of_integer_solutions_l792_792060

theorem number_of_integer_solutions :
  {x : ℤ | (x^2 - 2*x - 2)^(x + 3) = 1}.finite ∧
  {x : ℤ | (x^2 - 2*x - 2)^(x + 3) = 1}.to_finset.card = 3 :=
by
  sorry

end number_of_integer_solutions_l792_792060


namespace prove_f_0_eq_1_prove_f_even_function_l792_792004

variable {R : Type} [CommRing R]

-- The function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- The conditions
axiom functional_eqn : ∀ x y : ℝ, f(x) * f(y) = (f(x + y) + 2 * f(x - y)) / 3
axiom nonzero_condition : ∀ x : ℝ, f(x) ≠ 0

-- The proof problem in Lean 4
theorem prove_f_0_eq_1 : f 0 = 1 := 
by 
  sorry

theorem prove_f_even_function : ∀ x : ℝ, f(x) = f(-x) := 
by 
  sorry

end prove_f_0_eq_1_prove_f_even_function_l792_792004


namespace petya_wins_optimal_play_l792_792253

theorem petya_wins_optimal_play : 
  ∃ n : Nat, n = 2021 ∧
  (∀ move : Nat → Nat, 
    (move = (λ n, n - 1) ∨ move = (λ n, n - 2)) → 
    (n % 3 ≡ 0) ↔ 
      (∃ optimal_move : Nat → Nat, 
        optimal_move = (λ n, n - 2 + 3) 
          → (optimal_move n % 3 = 0))) -> 
  Petya wins := 
  sorry

end petya_wins_optimal_play_l792_792253


namespace isosceles_triangle_area_l792_792439

theorem isosceles_triangle_area
  (a b : ℝ) -- sides of the triangle
  (inradius : ℝ) (perimeter : ℝ)
  (angle : ℝ) -- angle in degrees
  (h_perimeter : 2 * a + b = perimeter)
  (h_inradius : inradius = 2.5)
  (h_angle : angle = 40)
  (h_perimeter_value : perimeter = 20)
  (h_semiperimeter : (perimeter / 2) = 10) :
  (inradius * (perimeter / 2) = 25) :=
by
  sorry

end isosceles_triangle_area_l792_792439


namespace symmedian_angle_equality_l792_792246

open Real EuclideanGeometry

variables {A B C D E P Q R S : Point} 
variables {tangent_circumcircle_A tangent_circumcircle_B tangent_circumcircle_C : Line}
variables {acute_triangle_ABC : Triangle}
variables {circumcircle : Circle}

-- Define acute triangle and its circumcircle
def acute_triangle (t : Triangle) : Prop := acute t.A t.B t.C

-- Define tangents to the circumcircle
def is_tangent (p : Point) (circ : Circle) (line : Line) : Prop := 
  line ∈ tangents_at p circ

-- Define midpoint
def is_midpoint (p q r : Point) : Prop := 
  2 * vector.to_tuple (p - r) = vector.to_tuple (q - r)

-- Define intersection points
def is_intersection (L : Line) (M : Line) (X : Point) : Prop :=
  collinear X ∧ L = X

-- Proof statement
theorem symmedian_angle_equality
  (h1: acute_triangle acute_triangle_ABC)
  (h2: is_tangent A circumcircle tangent_circumcircle_A)
  (h3: is_tangent B circumcircle tangent_circumcircle_B)
  (h4: is_tangent C circumcircle tangent_circumcircle_C)
  (h5: is_intersection (line_through A E) (line_through B C) P)
  (h6: is_intersection (line_through B D) (line_through A C) R)
  (h7: is_midpoint Q A P)
  (h8: is_midpoint S B R) :
  ∠A B Q = ∠B A S :=
sorry

end symmedian_angle_equality_l792_792246


namespace john_ratio_l792_792530

variable (w : ℕ) (d : ℕ) (e : ℕ) (b : ℕ) (r : ℕ)

def john_problem_data :=
  w = 10 ∧ 
  d = 30 - 4 ∧ 
  e = d * w ∧ 
  b = 50 ∧ 
  r = 160

theorem john_ratio (h : john_problem_data) : 
  (b : ℚ) / (e - b - r) = 1 :=
by
  sorry

end john_ratio_l792_792530


namespace parallel_lines_of_reflections_l792_792169

variables {A1 A2 A3 B12 B13 B21 B23 B31 B32 : Type*}
variables [has_inner B12 B21] [has_inner B13 B31] [has_inner B23 B32]

-- Consider a scalene triangle A1A2A3
def scalene_triangle (A1 A2 A3 : Type*) :=
  A1 ≠ A2 ∧ A2 ≠ A3 ∧ A3 ≠ A1

-- Define the symmetric point B in relation to the angle bisectors
def symmetric_point (Ai Aj Bij : Type*) : Prop :=
  sorry  -- Define the reflection over the angle bisector here

-- Lines B12B21, B13B31, and B23B32 are parallel
theorem parallel_lines_of_reflections (A1 A2 A3 B12 B13 B21 B23 B31 B32 : Type*)
  (h_scalene : scalene_triangle A1 A2 A3)
  (h_sym_B12 : symmetric_point A1 A2 B12)
  (h_sym_B21 : symmetric_point A2 A1 B21)
  (h_sym_B13 : symmetric_point A1 A3 B13)
  (h_sym_B31 : symmetric_point A3 A1 B31)
  (h_sym_B23 : symmetric_point A2 A3 B23)
  (h_sym_B32 : symmetric_point A3 A2 B32) :
  are_parallel (line B12 B21) (line B13 B31) ∧
  are_parallel (line B13 B31) (line B23 B32) ∧
  are_parallel (line B12 B21) (line B23 B32) :=
sorry

end parallel_lines_of_reflections_l792_792169


namespace pi_upper_bound_l792_792584

open Real

theorem pi_upper_bound (n : ℕ) (h_n : n > 55)
    (chebyshev_bound : ∀ n, 0.92129 * n / (ln n) < π n ∧ π n < 1.10555 * n / (ln n)) :
    π n < 3 * ln 2 * (n / ln n) :=
begin
  sorry
end

end pi_upper_bound_l792_792584


namespace common_divisors_l792_792899

theorem common_divisors (a b : ℕ) (ha : a = 9240) (hb : b = 8820) : 
  let g := Nat.gcd a b in 
  g = 420 ∧ Nat.divisors 420 = 24 :=
by
  have gcd_ab := Nat.gcd_n at ha hb
  have fact := Nat.factorize 420
  have divisors_420: ∀ k : Nat, g = 420 ∧ k = 24 := sorry
  exact divisors_420 24

end common_divisors_l792_792899


namespace range_of_c_l792_792476

-- Define proposition p: c^2 < c
def p (c : ℝ) : Prop := c^2 < c

-- Define proposition q: ∀ x ∈ ℝ, x^2 + 4cx + 1 > 0
def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + 4*c*x + 1 > 0

-- The main theorem stating the proof problem
theorem range_of_c (c : ℝ) : (xor (p c) (q c)) → (c ∈ Ioc (-1/2) 0 ∨ c ∈ Ico 1/2 1) :=
sorry

end range_of_c_l792_792476


namespace exists_s_l792_792265

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  P : Point
  Q : Point
  R : Point

def lineEquation (p1 p2 : Point) : ℝ → ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x) * (p1.x) + p1.y

def intersect (line : ℝ → ℝ) (y : ℝ) : Point :=
  let x := (y - line(0)) / ((line(1) - line(0)) / 1)
  { x := x, y := y }

def areaOfTriangle (P Q R : Point) : ℝ :=
  0.5 * (Q.x - P.x) * (P.y - R.y)

theorem exists_s : 
  ∃ s : ℝ, 
    let P : Point := {x := 0, y := 10}
    let Q : Point := {x := 3, y := 0}
    let R : Point := {x := 10, y := 0}
    let linePQ := lineEquation P Q
    let linePR := lineEquation P R
    let V := intersect linePQ s
    let W := intersect linePR s
    let area := areaOfTriangle P V W
    area = 20 ∧ s ≈ 1.92 :=
by
  -- Proof omitted
  sorry

end exists_s_l792_792265


namespace seating_6_around_table_with_reserved_seat_l792_792124

def seating_arrangements (n : ℕ) : ℕ :=
  if n > 0 then (n - 1)! else 1

theorem seating_6_around_table_with_reserved_seat :
  seating_arrangements 6 = 120 :=
by
  unfold seating_arrangements
  simp
  exact dec_trivial

end seating_6_around_table_with_reserved_seat_l792_792124


namespace intersection_M_N_l792_792086

section

def M (x : ℝ) : Prop := sqrt x < 4
def N (x : ℝ) : Prop := 3 * x >= 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | (1 / 3) ≤ x ∧ x < 16} :=
by
  sorry

end

end intersection_M_N_l792_792086


namespace water_left_after_operations_l792_792145

theorem water_left_after_operations :
  let initial_water := (3 : ℚ)
  let water_used := (4 / 3 : ℚ)
  let extra_water := (1 / 2 : ℚ)
  initial_water - water_used + extra_water = (13 / 6 : ℚ) := 
by
  -- Skips the proof, as the focus is on the problem statement
  sorry

end water_left_after_operations_l792_792145


namespace compound_interest_rate_l792_792668

noncomputable def compound_interest_principal := 40000
noncomputable def compound_interest_amount := 16197.12
noncomputable def future_value := compound_interest_principal + compound_interest_amount
noncomputable def time_period := 3
noncomputable def compounding_frequency := 1
noncomputable def interest_rate := 0.117 -- 11.7% in decimal

theorem compound_interest_rate :
  (1 + interest_rate) ^ time_period = future_value / compound_interest_principal :=
  sorry

end compound_interest_rate_l792_792668


namespace nice_set_count_2008_l792_792341

def is_nice_set (X : set ℕ) : Prop :=
∀ a b ∈ X, (a + b ∈ X ∧ |a - b| ∉ X) ∨ (a + b ∉ X ∧ |a - b| ∈ X)

theorem nice_set_count_2008 :
  let X := { x // is_nice_set x ∧ 2008 ∈ x } in
  finset.card { x // is_nice_set x ∧ 2008 ∈ x } = 8 :=
begin
  sorry
end

end nice_set_count_2008_l792_792341


namespace number_of_integer_solutions_l792_792061

theorem number_of_integer_solutions :
  {x : ℤ | (x^2 - 2*x - 2)^(x + 3) = 1}.finite ∧
  {x : ℤ | (x^2 - 2*x - 2)^(x + 3) = 1}.to_finset.card = 3 :=
by
  sorry

end number_of_integer_solutions_l792_792061


namespace each_friend_pays_18_l792_792226

theorem each_friend_pays_18 (total_bill : ℝ) (silas_share : ℝ) (tip_fraction : ℝ) (num_friends : ℕ) (silas : ℕ) (remaining_friends : ℕ) :
  total_bill = 150 →
  silas_share = total_bill / 2 →
  tip_fraction = 0.1 →
  num_friends = 6 →
  remaining_friends = num_friends - 1 →
  silas = 1 →
  (total_bill - silas_share + tip_fraction * total_bill) / remaining_friends = 18 :=
by
  intros
  sorry

end each_friend_pays_18_l792_792226


namespace exists_triangle_with_conditions_l792_792884

/-- Given three distinct, non-collinear points A1, A', and S, there exists a triangle ABC such that:
1. The point A1 is the foot of the altitude from A to BC.
2. The point A' is the intersection of the angle bisector from A with BC.
3. The point S is the centroid of triangle ABC. -/
theorem exists_triangle_with_conditions 
  (A1 A' S : Point)
  (h_distinct : A1 ≠ A' ∧ A' ≠ S ∧ A1 ≠ S)
  (h_non_collinear : ¬ collinear {A1, A', S}) :
  ∃ (A B C : Point), 
    (foot_of_altitude A B C = A1) ∧
    (angle_bisector_intersection A B C = A') ∧
    (centroid A B C = S) :=
sorry

end exists_triangle_with_conditions_l792_792884


namespace angle_BAC_is_right_angle_l792_792536

variable (A B C M N D : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited M] [Inhabited N] [Inhabited D]
variable [HasAngle A]
variable [HasLength A B] [HasLength A C] [HasLength B C]
variable [Midpoint D B C]
variable (AM AN BM CN : ℝ)
variable (angle_BAC angle_MDN : ℝ)

axiom AM_squared_AN_squared_equal_BM_squared_CN_squared : AM^2 + AN^2 = BM^2 + CN^2
axiom angle_MDN_equal_angle_BAC : angle_MDN = angle_BAC

theorem angle_BAC_is_right_angle : angle_BAC = 90 := by
  sorry

end angle_BAC_is_right_angle_l792_792536


namespace merchant_gross_profit_l792_792707

theorem merchant_gross_profit :
  ∃ S : ℝ, (42 + 0.30 * S = S) ∧ ((0.80 * S) - 42 = 6) :=
by
  sorry

end merchant_gross_profit_l792_792707


namespace parametric_to_line_segment_l792_792881

theorem parametric_to_line_segment :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 5 →
  ∃ x y : ℝ, x = 3 * t^2 + 2 ∧ y = t^2 - 1 ∧ (x - 3 * y = 5) ∧ (-1 ≤ y ∧ y ≤ 24) :=
by
  sorry

end parametric_to_line_segment_l792_792881


namespace trajectory_equation_find_m_value_l792_792834

def point (α : Type) := (α × α)
def fixed_points (α : Type) := point α

noncomputable def slopes (x y : ℝ) : ℝ := y / x

theorem trajectory_equation (x y : ℝ) (P : point ℝ) (A B : fixed_points ℝ)
  (k1 k2 : ℝ) (hk : k1 * k2 = -1/4) :
  A = (-2, 0) → B = (2, 0) →
  P = (x, y) → 
  slopes (x + 2) y * slopes (x - 2) y = -1/4 →
  (x^2 / 4) + y^2 = 1 :=
sorry

theorem find_m_value (m x₁ x₂ y₁ y₂ : ℝ) (k : ℝ) (hx : (4 * k^2) + 1 - m^2 > 0)
  (hroots_sum : x₁ + x₂ = -((8 * k * m) / ((4 * k^2) + 1)))
  (hroots_prod : x₁ * x₂ = (4 * m^2 - 4) / ((4 * k^2) + 1))
  (hperp : x₁ * x₂ + y₁ * y₂ = 0) :
  y₁ = k * x₁ + m → y₂ = k * x₂ + m →
  m^2 = 4/5 * (k^2 + 1) →
  m = 2 ∨ m = -2 :=
sorry

end trajectory_equation_find_m_value_l792_792834


namespace mary_investment_l792_792188

variables (M : ℝ) (Mike_investment : ℝ := 300) (total_profit : ℝ := 3000) (extra_amount : ℝ := 800)

-- Conditions setup
def profit_divided_equally : ℝ := (1 / 3) * total_profit
def each_effort_share : ℝ := profit_divided_equally / 2
def remaining_profit : ℝ := total_profit - profit_divided_equally
def mary_share (M : ℝ) : ℝ := (M / (M + Mike_investment)) * remaining_profit + each_effort_share
def mike_share (M : ℝ) : ℝ := (Mike_investment / (M + Mike_investment)) * remaining_profit + each_effort_share

-- The theorem statement to prove Mary's investment
theorem mary_investment : mary_share M = mike_share M + extra_amount ↔ M = 700 :=
sorry

end mary_investment_l792_792188


namespace quadratic_roots_l792_792399

theorem quadratic_roots (z : ℂ) :
  z^2 - z = (3 - 7i) ↔ (z = (1 + 2 * real.sqrt 7 - real.sqrt 7 * I) / 2) ∨ (z = (1 - 2 * real.sqrt 7 + real.sqrt 7 * I) / 2) := 
begin
  sorry
end

end quadratic_roots_l792_792399


namespace rope_cut_into_pieces_l792_792319

theorem rope_cut_into_pieces (length_of_rope_cm : ℕ) (num_equal_pieces : ℕ) (length_equal_piece_mm : ℕ) (length_remaining_piece_mm : ℕ) 
  (h1 : length_of_rope_cm = 1165) (h2 : num_equal_pieces = 150) (h3 : length_equal_piece_mm = 75) (h4 : length_remaining_piece_mm = 100) :
  (num_equal_pieces * length_equal_piece_mm + (11650 - num_equal_pieces * length_equal_piece_mm) / length_remaining_piece_mm = 154) :=
by
  sorry

end rope_cut_into_pieces_l792_792319


namespace polynomial_exists_with_given_properties_l792_792016

noncomputable def exists_polynomial_q (a : ℕ → ℤ) (p : ℝ[X]) : Prop :=
  ∀ n m : ℕ, n ≠ m → (a n - a m) % (n - m) = 0 ∧ ∀ n : ℕ, p.eval (n : ℝ) > |a n|

theorem polynomial_exists_with_given_properties (a : ℕ → ℤ) (p : ℝ[X]) :
  exists_polynomial_q a p → ∃ q : ℤ[X], ∀ n : ℕ, q.eval (n : ℤ) = a n :=
sorry

end polynomial_exists_with_given_properties_l792_792016


namespace prob_of_rolling_2_exactly_4_times_l792_792492

open_locale classical

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

def prob_roll_2_exactly_4_times_in_6_rolls : ℚ :=
let p_roll_2 := (1 : ℚ) / 6 in
let p_not_roll_2 := (5 : ℚ) / 6 in
let specific_order_prob := p_roll_2^4 * p_not_roll_2^2 in
let num_ways := binomial 6 4 in
(num_ways : ℚ) * specific_order_prob

theorem prob_of_rolling_2_exactly_4_times :
  prob_roll_2_exactly_4_times_in_6_rolls = (375 : ℚ) / 46656 := 
by 
  sorry

end prob_of_rolling_2_exactly_4_times_l792_792492


namespace slope_of_tangent_at_neg5_l792_792031

variable (f : ℝ → ℝ)

-- Define conditions
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_differentiable : Differentiable ℝ f
axiom f_prime_at_1 : f' 1 = -2
axiom f_shift_property : ∀ x : ℝ, f (x + 2) = f (x - 2)

-- The theorem to prove
theorem slope_of_tangent_at_neg5 : deriv f (-5) = 2 :=
sorry

end slope_of_tangent_at_neg5_l792_792031


namespace tan_eleven_pi_over_four_eq_neg_one_l792_792786

noncomputable def tan_of_eleven_pi_over_four : Real := 
  let to_degrees (x : Real) : Real := x * 180 / Real.pi
  let angle := to_degrees (11 * Real.pi / 4)
  let simplified := angle - 360 * Real.floor (angle / 360)
  if simplified < 0 then
    simplified := simplified + 360
  if simplified = 135 then -1
  else
    undefined

theorem tan_eleven_pi_over_four_eq_neg_one :
  tan (11 * Real.pi / 4) = -1 := 
by
  sorry

end tan_eleven_pi_over_four_eq_neg_one_l792_792786


namespace fraction_of_complex_number_l792_792422

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := 1 - real.sqrt 3 * i

theorem fraction_of_complex_number :
  4 / z = 1 + real.sqrt 3 * i := 
by
  sorry

end fraction_of_complex_number_l792_792422


namespace min_distance_C_l792_792448

def Point (α : Type) := (x : α) × (y : α)

def line_segment (A B : Point ℝ) : Point ℝ → Prop :=
  λ C, ∃ t ∈ (0 : ℝ) ..1, C.1 = (1 - t) * A.1 + t * B.1 ∧ C.2 = (1 - t) * A.2 + t * B.2

theorem min_distance_C
  (A B : Point ℝ) (C : Point ℝ) (hA : A = (-3, -3)) (hB : B = (6, 3)) (hC : C.1 = 3) :
  C.2 = -1 / 3 :=
by
  sorry

end min_distance_C_l792_792448


namespace asymptote_sum_l792_792749

theorem asymptote_sum (c d : ℝ) 
  (h1 : ∀ x : ℝ, x² + c * x + d = 0 ↔ x = -1 ∨ x = 3) :
  c + d = -5 := 
by 
  sorry

end asymptote_sum_l792_792749


namespace combined_distance_l792_792531

theorem combined_distance (t1 t2 : ℕ) (s1 s2 : ℝ)
  (h1 : t1 = 30) (h2 : s1 = 9.5) (h3 : t2 = 45) (h4 : s2 = 8.3)
  : (s1 * t1 + s2 * t2) = 658.5 := 
by
  sorry

end combined_distance_l792_792531


namespace probability_product_less_than_30_l792_792194

theorem probability_product_less_than_30 :
  let P := ({x : ℕ // 1 ≤ x ∧ x ≤ 5})
  let M := ({y : ℕ // 1 ≤ y ∧ y ≤ 10})
  let event (p : P) (m : M) := (p.val * m.val < 30)
  ∑ x in P, ∑ y in M, (if event x y then 1 else 0) / (finset.card P * finset.card M) = 41 / 50 :=
by
  sorry

end probability_product_less_than_30_l792_792194


namespace factorize_difference_of_squares_l792_792778

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 81 = (a + 9) * (a - 9) :=
by
  sorry

end factorize_difference_of_squares_l792_792778


namespace proof_problem_l792_792699

-- Definitions
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def draws : ℕ := 3

-- Part 1: Probability the first ball drawn is red given the second ball drawn is white
def P(A B : Prop) := sorry
def P_giv (A B : Prop) : ℚ := P(A ∧ B) / P(B)

def first_red_given_second_white (prob : ℚ) : Prop :=
  P_giv (first_draw_red ∧ second_draw_white) = prob

-- Part 2: Distribution and mathematical expectation of the number of red balls drawn
def distribution (X : ℕ → ℚ) : Prop := 
  (X 0 = 27/125) ∧ (X 1 = 549/1000) ∧ (X 2 = 47/200)

def expectation (E : ℚ) : Prop :=
  E = (0 * (27/125) + 1 * (549/1000) + 2 * (47/200))

-- Lean statement
theorem proof_problem : 
  (first_red_given_second_white (5/11)) ∧ 
  (distribution (λ x => [27/125, 549/1000, 47/200].nth x.get_or_else 0)) ∧ 
  (expectation (1019/1000)) :=
by sorry

end proof_problem_l792_792699


namespace maximum_B_k_at_181_l792_792777

open Nat

theorem maximum_B_k_at_181 :
  let B : ℕ → ℝ := λ k, (Nat.choose 2000 k : ℝ) * (0.1 ^ k)
  ∃ k : ℕ, k ≤ 2000 ∧ (∀ m : ℕ, m ≤ 2000 → B m ≤ B 181) :=
by
  let B := λ k : ℕ, (Nat.choose 2000 k : ℝ) * (0.1 ^ k)
  use 181
  split
  · linarith
  · intro m hm
    sorry

end maximum_B_k_at_181_l792_792777


namespace common_divisors_9240_8820_l792_792904

def prime_factors_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def prime_factors_8820 := [(2, 2), (3, 2), (5, 1), (7, 1), (11, 1)]

def gcd_prime_factors := [(2, 2), (3, 1), (5, 1), (7, 1), (11, 1)]

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.snd + 1)) 1

theorem common_divisors_9240_8820 :
  num_divisors gcd_prime_factors = 32 := by
  sorry

end common_divisors_9240_8820_l792_792904


namespace Megan_not_lead_actress_l792_792569

-- Define the conditions: total number of plays and lead actress percentage
def totalPlays : ℕ := 100
def leadActressPercentage : ℕ := 80

-- Define what we need to prove: the number of times Megan was not the lead actress
theorem Megan_not_lead_actress (totalPlays: ℕ) (leadActressPercentage: ℕ) : 
  (totalPlays * (100 - leadActressPercentage)) / 100 = 20 :=
by
  -- proof omitted
  sorry

end Megan_not_lead_actress_l792_792569


namespace lucius_weekly_earnings_l792_792564

/--
Lucius owns a small business and spends $10 every day on ingredients, makes and sells one portion of French Fries and one portion of Poutine each day. He pays ten percent of his weekly income as tax. Given that the price of French Fries is $12 and the price of Poutine is $8, 
we want to show that he earns $56 every week after paying the taxes and buying all the ingredients.
-/
theorem lucius_weekly_earnings
  (daily_cost : ℕ = 10) 
  (french_fries_price : ℕ = 12) 
  (poutine_price : ℕ = 8) 
  (daily_french_fries_sold : ℕ = 1) 
  (daily_poutine_sold : ℕ = 1) 
  (tax_rate : ℤ = 10) : 
  let 
    daily_earnings := daily_french_fries_sold * french_fries_price + daily_poutine_sold * poutine_price,
    weekly_earnings_before_expenses_and_taxes := daily_earnings * 7,
    weekly_expenses := daily_cost * 7,
    weekly_earnings_after_expenses := weekly_earnings_before_expenses_and_taxes - weekly_expenses,
    weekly_tax := weekly_earnings_before_expenses_and_taxes * tax_rate / 100,
    weekly_earnings_after_taxes_and_expenses := weekly_earnings_after_expenses - weekly_tax
  in
  weekly_earnings_after_taxes_and_expenses = 56 := sorry

end lucius_weekly_earnings_l792_792564


namespace tim_initial_books_l792_792589

def books_problem : Prop :=
  ∃ T : ℕ, 10 + T - 24 = 19 ∧ T = 33

theorem tim_initial_books : books_problem :=
  sorry

end tim_initial_books_l792_792589


namespace correct_statements_count_l792_792854

-- Definitions of the conditions as predicates
def statement1 (a b c : Type) [HasOrth a b] [HasOrth a c] [HasPara b c] : Prop :=
  a ⊥ b ∧ a ⊥ c → b ∥ c

def statement2 (a b c : Type) [HasOrth a b] [HasOrth a c] [HasOrth b c] : Prop :=
  a ⊥ b ∧ a ⊥ c → b ⊥ c

def statement3 (a b c : Type) [HasPara a b] [HasOrth b c] [HasOrth a c] : Prop :=
  a ∥ b ∧ b ⊥ c → a ⊥ c

-- Count the number of true statements
def num_of_correct_statements (a b c : Type) [HasOrth a b] [HasOrth a c] [HasPara b c] 
  [HasOrth b c] [HasPara a b] [HasOrth a c] : Nat :=
  (if statement1 a b c then 1 else 0) +
  (if statement2 a b c then 1 else 0) +
  (if statement3 a b c then 1 else 0)

-- The theorem stating the original problem
theorem correct_statements_count {a b c : Type} [HasOrth a b] [HasOrth a c] [HasPara b c] 
  [HasOrth b c] [HasPara a b] [HasOrth a c] : 
  num_of_correct_statements a b c = 1 :=
sorry

end correct_statements_count_l792_792854


namespace cost_of_chlorine_l792_792823

/--
Gary has a pool that is 10 feet long, 8 feet wide, and 6 feet deep.
He needs to buy one quart of chlorine for every 120 cubic feet of water.
Chlorine costs $3 per quart.
Prove that the total cost of chlorine Gary spends is $12.
-/
theorem cost_of_chlorine:
  let length := 10
  let width := 8
  let depth := 6
  let volume := length * width * depth
  let chlorine_per_cubic_feet := 1 / 120
  let chlorine_needed := volume * chlorine_per_cubic_feet
  let cost_per_quart := 3
  let total_cost := chlorine_needed * cost_per_quart
  total_cost = 12 :=
by
  sorry

end cost_of_chlorine_l792_792823


namespace number_of_observations_is_14_l792_792257

theorem number_of_observations_is_14
  (mean_original : ℚ) (mean_new : ℚ) (original_sum : ℚ) 
  (corrected_sum : ℚ) (n : ℚ)
  (h1 : mean_original = 36)
  (h2 : mean_new = 36.5)
  (h3 : corrected_sum = original_sum + 7)
  (h4 : mean_new = corrected_sum / n)
  (h5 : original_sum = mean_original * n) :
  n = 14 :=
by
  -- Here goes the proof
  sorry

end number_of_observations_is_14_l792_792257


namespace gcd_98_63_l792_792232

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l792_792232


namespace combined_vacations_and_classes_l792_792056

-- Define the conditions
def Kelvin_classes : ℕ := 90
def Grant_vacations : ℕ := 4 * Kelvin_classes

-- The Lean 4 statement proving the combined total of vacations and classes
theorem combined_vacations_and_classes : Kelvin_classes + Grant_vacations = 450 := by
  sorry

end combined_vacations_and_classes_l792_792056


namespace negation_of_existential_l792_792627

theorem negation_of_existential :
  (¬ ∃ (x : ℝ), x^2 + x + 1 < 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_l792_792627


namespace base_radius_of_cone_is_4_l792_792340

-- Define the sector with arc length 8 * π
def sector.arc_length : ℝ := 8 * Real.pi

-- Define the relationship between the arc length of the sector and the circumference of the base of the cone
def cone.circumference (arc_length : ℝ) : ℝ := arc_length

-- Define the formula for the radius of the base of the cone in terms of its circumference
def base.radius (circumference : ℝ) : ℝ := circumference / (2 * Real.pi)

-- Prove that the radius of the base of the cone is 4
theorem base_radius_of_cone_is_4 : base.radius (cone.circumference sector.arc_length) = 4 :=
by
-- proof skipped
sorry

end base_radius_of_cone_is_4_l792_792340


namespace measure_angle_B_proof_l792_792184

noncomputable def measure_angle_B (l m : set ℝ) (A C B: ℝ) (angleA angleC: ℝ) (h_parallel : l ∥ m) (h_A_on_l : A ∈ l) (h_C_on_m : C ∈ m) : Prop :=
∃ B ∈ m, measure_angle_B = 70

theorem measure_angle_B_proof
  (l m : set ℝ) (A C : ℝ)
  (h_parallel : l ∥ m)
  (h_A_on_l : A ∈ l)
  (h_C_on_m : C ∈ m)
  (h_angleA : angle l A = 100)
  (h_angleC : angle m C = 70)
: measure_angle_B l m A C 70 :=
sorry

end measure_angle_B_proof_l792_792184


namespace intersection_of_A_and_B_l792_792176

-- Define sets A and B
def A := {1, 2, 3, 4, 5}
def B := {2, 4, 5, 8, 10}

-- Define the intersection of A and B
def A_inter_B := {2, 4, 5}

-- State the theorem to prove that A ∩ B is equal to {2, 4, 5}
theorem intersection_of_A_and_B : A ∩ B = A_inter_B := by
  sorry

end intersection_of_A_and_B_l792_792176


namespace domain_of_f_B_is_R_l792_792351

noncomputable def f_A (x : ℝ) : ℝ := 1 / x
noncomputable def f_B (x : ℝ) : ℝ := (x^2 - 2) / (x^2 + 1)
noncomputable def f_C (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f_D (x : ℝ) : ℝ := x^2

-- The domain of a function f_B is the set of all real numbers:
theorem domain_of_f_B_is_R : ∀ x : ℝ, ∃ y: ℝ, y = f_B x :=
by
  sorry

end domain_of_f_B_is_R_l792_792351


namespace simple_interest_calculation_l792_792720

/-- Define the Principal, Rate, and Time. --/
def principal : ℝ := 7302.272727272727
def rate : ℝ := 11
def time : ℝ := 5

/-- Define the formula for Simple Interest. --/
def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

/-- Problem: Prove that the total simple interest earned is 4016.25 given the conditions --/
theorem simple_interest_calculation : simple_interest principal rate time = 4016.25 :=
by
  -- The proof is omitted to focus on statement translation.
  sorry

end simple_interest_calculation_l792_792720


namespace feral_triples_count_l792_792368

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_feral (a b c : ℕ) : Prop :=
  is_prime (b - a) ∧ is_prime (c - a) ∧ is_prime (c - b)

def num_feral_triples (n : ℕ) : ℕ :=
  ((finset.range (n+1)).product (finset.range (n+1))).product (finset.range (n+1))
  .count (λ ((a, b), c) => 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 20 ∧ is_feral a b c)

theorem feral_triples_count : num_feral_triples 20 = 72 :=
by
  sorry

end feral_triples_count_l792_792368


namespace polynomial_solution_l792_792800

noncomputable def Q (x : ℝ) : ℝ := x^2 + 2*x

theorem polynomial_solution {Q : ℝ → ℝ} 
  (h : ∀ x, Q(Q(x)) = (x^2 + 2*x + 2) * Q(x)) :
  ∃ a b c : ℝ, a ≠ 0 ∧ Q x = a*x^2 + b*x + c ∧ Q x = x^2 + 2*x :=
by 
  existsi [1, 2, 0]
  simp
  sorry

end polynomial_solution_l792_792800


namespace distinct_three_digit_numbers_count_l792_792889

theorem distinct_three_digit_numbers_count : 
  ∃! n : ℕ, n = 5 * 4 * 3 :=
by
  use 60
  sorry

end distinct_three_digit_numbers_count_l792_792889


namespace circle_standard_eq_l792_792814

theorem circle_standard_eq (M N : ℝ × ℝ) (hM : M = (2, 0)) (hN : N = (0, 4)) :
  ∃ x y r, (x - 1) ^ 2 + (y - 2) ^ 2 = r ^ 2 ∧ r ^ 2 = 5 :=
by
  let C := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  have hC : C = (1, 2) := sorry
  let r := Real.sqrt ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2) / 2
  have hr : r ^ 2 = 5 := sorry
  use [1, 2, Real.sqrt 5]
  simp [hC, hr]
  sorry

end circle_standard_eq_l792_792814


namespace Megan_not_lead_plays_l792_792566

-- Define the problem's conditions as variables
def total_plays : ℕ := 100
def lead_play_ratio : ℤ := 80

-- Define the proposition we want to prove
theorem Megan_not_lead_plays : 
  (total_plays - (total_plays * lead_play_ratio / 100)) = 20 := 
by sorry

end Megan_not_lead_plays_l792_792566


namespace star_emilio_sum_difference_l792_792598

theorem star_emilio_sum_difference :
  let star_numbers := List.range' 1 50,
      emilio_numbers := star_numbers.map (fun n => 
        let digits := n.toString.data,
            replaced_digits := digits.map (fun d => if d = '3' then '2' else d),
            replaced_string := replaced_digits.asString
        in replaced_string.toNat) 
  in star_numbers.sum - emilio_numbers.sum = 105 := by
  sorry

end star_emilio_sum_difference_l792_792598


namespace rook_connected_pairing_l792_792182

def rook_connected (cells: set (ℤ × ℤ)) : Prop :=
  ∀x y ∈ cells, ∃seq: list (ℤ × ℤ), seq.head = x ∧ seq.last = y ∧ ∀(a b ∈ seq), 
  (a.1 = b.1 ∨ a.2 = b.2)

theorem rook_connected_pairing (cells: set (ℤ × ℤ)) 
  (h_size: cells.size = 100) 
  (h_conn: rook_connected cells): 
  ∃pairs: list (ℤ × ℤ) × (ℤ × ℤ), 
  ∀ (p: (ℤ × ℤ) × (ℤ × ℤ)) ∈ pairs, 
  (p.1.1 = p.2.1 ∨ p.1.2 = p.2.2) ∧
  ∀ (c : (ℤ × ℤ)), c ∈ cells ↔ ∃ (p : (ℤ × ℤ) × (ℤ × ℤ)), c = p.1 ∨ c = p.2 :=
begin
  sorry
end

end rook_connected_pairing_l792_792182


namespace students_not_like_any_l792_792109

variables (F B P T F_cap_B F_cap_P F_cap_T B_cap_P B_cap_T P_cap_T F_cap_B_cap_P_cap_T : ℕ)

def total_students := 30

def students_like_F := 18
def students_like_B := 12
def students_like_P := 14
def students_like_T := 10

def students_like_F_and_B := 8
def students_like_F_and_P := 6
def students_like_F_and_T := 4
def students_like_B_and_P := 5
def students_like_B_and_T := 3
def students_like_P_and_T := 7

def students_like_all_four := 2

theorem students_not_like_any :
  total_students - ((students_like_F + students_like_B + students_like_P + students_like_T)
                    - (students_like_F_and_B + students_like_F_and_P + students_like_F_and_T
                      + students_like_B_and_P + students_like_B_and_T + students_like_P_and_T)
                    + students_like_all_four) = 11 :=
by sorry

end students_not_like_any_l792_792109


namespace find_a_plus_b_l792_792044

theorem find_a_plus_b (a b x : ℝ) (h1 : x + 2 * a > 4) (h2 : 2 * x < b)
  (h3 : 0 < x) (h4 : x < 2) : a + b = 6 :=
by
  sorry

end find_a_plus_b_l792_792044


namespace third_rectangle_area_l792_792273

-- Definitions for dimensions of the first two rectangles
def rect1_length := 3
def rect1_width := 8

def rect2_length := 2
def rect2_width := 5

-- Total area of the first two rectangles
def total_area := (rect1_length * rect1_width) + (rect2_length * rect2_width)

-- Declaration of the theorem to be proven
theorem third_rectangle_area :
  ∃ a b : ℝ, a * b = 4 ∧ total_area + a * b = total_area + 4 :=
by
  sorry

end third_rectangle_area_l792_792273


namespace limit_derivative_of_differentiable_at_l792_792463

variable {α : Type*} {β : Type*} [NormedField α] [NormedSpace α β] {f : α → β} {x0 : α}

theorem limit_derivative_of_differentiable_at (h : DifferentiableAt α f x0) :
  Filter.Tendsto (fun t => (f (x0 + t) - f (x0 - 3 * t)) / t) (nhds 0) (nhds (4 * (fderiv ℝ f x0 t))) :=
sorry

end limit_derivative_of_differentiable_at_l792_792463


namespace num_values_expression_l792_792628

theorem num_values_expression (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
  ∃ S : set ℝ, S = { a : ℝ | ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ a = 10 - 10 * abs (2 * x - 3) } ∧ S.card = 11 :=
by
  sorry

end num_values_expression_l792_792628


namespace sin_75_is_option_D_l792_792403

noncomputable def sin_75 : ℝ := Real.sin (75 * Real.pi / 180)

noncomputable def option_D : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4

theorem sin_75_is_option_D : sin_75 = option_D :=
by
  sorry

end sin_75_is_option_D_l792_792403


namespace multiplication_with_negative_l792_792366

theorem multiplication_with_negative (a b : Int) (h1 : a = 3) (h2 : b = -2) : a * b = -6 :=
by
  sorry

end multiplication_with_negative_l792_792366


namespace rectangle_condition_l792_792948

variable (A B C D : Type) [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace D]
variable [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
variable (AB AD BC CD : ℝ) (angleA angleB angleC angleD : ℝ)

noncomputable theory

-- Conditions:
variable (parallelogram : (AD = BC) ∧ (AB = CD))
variable (parallel : ∃ l m : Line A, is_parallel l m)
-- Given:
variable (angle_condition : angleA = angleB)

-- Proof goal:
theorem rectangle_condition : (parallel) ∧ (parallelogram) ∧ (angle_condition) → is_rectangle A B C D :=
by
  sorry

end rectangle_condition_l792_792948


namespace market_value_of_stock_l792_792676

def face_value : ℝ := 100
def dividend_per_share : ℝ := 0.10 * face_value
def yield : ℝ := 0.08

theorem market_value_of_stock : (dividend_per_share / yield) = 125 := by
  -- Proof not required
  sorry

end market_value_of_stock_l792_792676


namespace sequence_integer_sum_l792_792243

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
∀ k : ℕ, a (k + 1) = a k + k / a k

theorem sequence_integer_sum
  (a : ℕ → ℝ)
  (ha1 : 1 < a 1 ∧ a 1 < 2)
  (ha_seq : sequence a) :
  ∃ at_most_one_pair : ∀ k j : ℕ, k ≠ j → 
  ¬ (∃ m : ℤ, a k + a j = m) :=
by
  sorry

end sequence_integer_sum_l792_792243


namespace vkontakte_status_l792_792313

variables (M I A P : Prop)

theorem vkontakte_status
  (h1 : M → (I ∧ A))
  (h2 : xor A P)
  (h3 : I ∨ M)
  (h4 : P ↔ I) :
  (¬ M ∧ I ∧ ¬ A ∧ P) :=
by
  sorry

end vkontakte_status_l792_792313


namespace remainder_of_n_squared_minus_n_plus_4_l792_792071

theorem remainder_of_n_squared_minus_n_plus_4 (k : ℤ) : 
  let n := 100 * k - 1 in 
  (n^2 - n + 4) % 100 = 6 := 
by
  sorry

end remainder_of_n_squared_minus_n_plus_4_l792_792071


namespace palindromes_in_seven_steps_and_sum_l792_792412

-- Definitions based on problem conditions
def reverse_number (n : ℕ) : ℕ := 
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  100 * c + 10 * b + a

-- Seven-step addition process
def steps_to_palindrome (n : ℕ) : ℕ × ℕ := -- returns (steps, resulting number)
  let rec aux (m : ℕ) (steps : ℕ) : ℕ × ℕ :=
    let rev := reverse_number m
    if rev = m then (steps, m)
    else if steps = 7 then (steps, m)
    else aux (m + rev) (steps + 1)
  aux n 0

-- Prove that specific numbers 307 and 853 become palindromes in exactly seven steps
theorem palindromes_in_seven_steps_and_sum :
  (steps_to_palindrome 307).fst = 7 ∧ (steps_to_palindrome 853).fst = 7 ∧
  (steps_to_palindrome 307).snd + (steps_to_palindrome 853).snd = 1160 :=
by {
  sorry
}


end palindromes_in_seven_steps_and_sum_l792_792412


namespace square_point_distances_l792_792348

theorem square_point_distances 
  (A B C D : Point)
  (s a b c d k : ℝ)
  (P : Point)
  (hABCD_square : is_square A B C D s)
  (dist_PA : dist P A = a) 
  (dist_PB : dist P B = b) 
  (dist_PC : dist P C = c)
  (dist_PD : dist P D = d) 
  (ha^2 : a ^ 2 = dist P A * dist P A) 
  (hb^2 : b ^ 2 = dist P B * dist P B) 
  (hc^2 : c ^ 2 = dist P C * dist P C) 
  (hd^2 : d ^ 2 = dist P D * dist P D) : 
  (a^2 + c^2 = b^2 + d^2) ∧ 
  (8 * k^2 = 2 * (a^2 + b^2 + c^2 + d^2) - 4 * s^2 - (a^4 + b^4 + c^4 + d^4 - 2 * a^2 * c^2 - 2 * b^2 * d^2) / s^2) := 
sorry

end square_point_distances_l792_792348


namespace hexagon_angles_l792_792980

theorem hexagon_angles
  (AB CD EF BC DE FA : ℝ)
  (F A B C D E : Type*)
  (FAB ABC EFA CDE : ℝ)
  (h1 : AB = CD)
  (h2 : AB = EF)
  (h3 : BC = DE)
  (h4 : BC = FA)
  (h5 : FAB + ABC = 240)
  (h6 : FAB + EFA = 240) :
  FAB + CDE = 240 :=
sorry

end hexagon_angles_l792_792980


namespace possible_measures_of_A_l792_792625

theorem possible_measures_of_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ A = 180 - B ∧ ∃ k : ℕ, k ≥ 1 ∧ A = k * B ∧ finset.count (λ n, (n > 1) ∧ (n ∣ 180)) = 17 :=
sorry

end possible_measures_of_A_l792_792625


namespace probability_at_most_one_success_in_three_attempts_l792_792696

noncomputable def basket_probability := (2 : ℚ) / 3

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_at_most_one_success_in_three_attempts :
  binomial_probability 3 0 basket_probability + binomial_probability 3 1 basket_probability = 7 / 27 :=
by
  sorry

end probability_at_most_one_success_in_three_attempts_l792_792696


namespace find_n_l792_792046

theorem find_n (n : ℕ) (h₁ : 3 * n + 4 = 13) : n = 3 :=
by 
  sorry

end find_n_l792_792046


namespace three_students_same_topic_l792_792764

-- Define the problem's parameters and conditions
def student := ℕ
def topic := ℕ

-- Given conditions
axiom students : set student
axiom topics : set topic
axiom discussed : student → student → topic
axiom student_count : students.card = 17
axiom topics_count : topics.card = 3
axiom discussed_symm : ∀ (a b : student), discussed a b = discussed b a
axiom discussed_diagonal: ∀ (a : student), discussed a a = 0

theorem three_students_same_topic : ∃ (a b c : student), a ∈ students ∧ b ∈ students ∧ c ∈ students ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  discussed a b = discussed a c ∧ discussed a c = discussed b c :=
by
  sorry  -- Proof omitted

end three_students_same_topic_l792_792764


namespace find_f2008_l792_792167

variable (f : ℝ → ℝ) 

-- Conditions from the given problem.
def cond1 := ∀ x : ℝ, f(x + 3) ≤ f(x) + 3
def cond2 := ∀ x : ℝ, f(x + 2) ≥ f(x) + 2
def cond3 := f(4) = 2008

-- Definition of g.
def g (x : ℝ) := f(x) - x

-- The theorem we need to prove.
theorem find_f2008
  (h1 : cond1 f)
  (h2 : cond2 f)
  (h3 : cond3 f) : f 2008 = 4012 := 
sorry

end find_f2008_l792_792167


namespace cucumbers_after_purchase_l792_792294

theorem cucumbers_after_purchase (C U : ℕ) (h1 : C + U = 10) (h2 : C = 4) : U + 2 = 8 := by
  sorry

end cucumbers_after_purchase_l792_792294


namespace sarah_loan_amount_l792_792590

-- Define the conditions and question as a Lean theorem
theorem sarah_loan_amount (down_payment : ℕ) (monthly_payment : ℕ) (years : ℕ) (months_in_year : ℕ) :
  (down_payment = 10000) →
  (monthly_payment = 600) →
  (years = 5) →
  (months_in_year = 12) →
  down_payment + (monthly_payment * (years * months_in_year)) = 46000 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end sarah_loan_amount_l792_792590


namespace eccentricity_of_ellipse_l792_792617

noncomputable def polynomial_roots :=
  { z : ℂ | (z - 1) * (z^2 + 2 * z + 4) * (z^2 + 4 * z + 6) = 0 }

def points : set (ℝ × ℝ) :=
  { (1, 0), (-1, real.sqrt 3), (-1, - real.sqrt 3), (-2, real.sqrt 2), (-2, -real.sqrt 2) }

def is_ellipse (e : ℝ) : Prop :=
  ∃ h a b, (∀ x y, (x, y) ∈ points → ((x - h)^2 / a^2) + (y^2 / b^2) = 1) ∧
           (real.gcd m n = 1) ∧ e = real.sqrt (m / n)

theorem eccentricity_of_ellipse : ∃ m n : ℕ, is_ellipse (real.sqrt (1 / 6)) ∧ (m + n = 7) :=
sorry

end eccentricity_of_ellipse_l792_792617


namespace range_of_x_bound_l792_792456

variable {f : ℝ → ℝ}

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem range_of_x_bound (h_increasing : is_increasing f)
  (hA : f 0 = -1) (hB : f 3 = 1) :
  { x : ℝ | |f (x + 1)| < 1 } = set.Icc (-1 : ℝ) (2 : ℝ) :=
sorry

end range_of_x_bound_l792_792456


namespace predict_tenth_generation_total_grains_l792_792322

def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)

theorem predict_tenth_generation_total_grains :
  -- Given data points
  let generations := [1, 2, 3, 4]
  let grains := [197, 193, 201, 209]
  -- Given regression equation form
  ∃ (a : ℝ), ∀ (x : ℝ), (hat_y x = 4.4 * x + a) →
  -- Predict the number of grains for the 10th generation
  let x := 10 in hat_y x = 233 := 
by
  let x_mean := mean generations
  let y_mean := mean grains
  let a := y_mean - 4.4 * x_mean
  let hat_y := λ x, 4.4 * x + a
  existsi a
  intro x
  intro h
  specialize h 10
  calc
  hat_y 10 = 4.4 * 10 + a : by rw h; refl
  ... = 233 : by 
    have x_mean_comp : x_mean = (1 + 2 + 3 + 4) / 4 := by simp [generations, mean]
    have y_mean_comp : y_mean = (197 + 193 + 201 + 209) / 4 := by simp [grains, mean]
    have a_comp : a = y_mean - 4.4 * x_mean := by simp [a, x_mean_comp, y_mean_comp]
    simp [a_comp, x_mean_comp, y_mean_comp]
    sorry 

end predict_tenth_generation_total_grains_l792_792322


namespace tournament_points_per_win_l792_792938

-- Define the necessary conditions and declare the proof problem
theorem tournament_points_per_win :
  let teams_scores := [12, 10, 9, 8, 7, 6] in
  let total_points := list.sum teams_scores in
  let total_matches := 15 in
  let win_points := 4 in
  (win_points * (total_matches - 2 * (total_matches - total_points / 2)) + 2 * (total_matches - (total_matches - total_points / 2)) = total_points) →
  win_points = 4 :=
by
  intros teams_scores total_points total_matches win_points h
  sorry

end tournament_points_per_win_l792_792938


namespace pure_imaginary_complex_l792_792925

noncomputable def i : ℂ := complex.I

theorem pure_imaginary_complex (a : ℝ) 
  (h : (complex.mk 2 (-a) / (complex.mk 1 1) : ℂ).re = 0) : 
  a = 2 :=
sorry

end pure_imaginary_complex_l792_792925


namespace sum_first_100_terms_eq_neg_200_l792_792231

def general_term (n : ℕ) : ℤ := (-1)^(n-1) * (4 * n - 3)

def sum_first_n_terms (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, general_term (i + 1)

theorem sum_first_100_terms_eq_neg_200 :
  sum_first_n_terms 100 = -200 :=
by
  sorry

end sum_first_100_terms_eq_neg_200_l792_792231


namespace f_of_6_l792_792703

noncomputable def f (u : ℝ) : ℝ := 
  let x := (u + 2) / 4
  x^3 - x + 2

theorem f_of_6 : f 6 = 8 :=
by
  sorry

end f_of_6_l792_792703


namespace snowball_game_l792_792658

theorem snowball_game (x y z : ℕ) (h : 5 * x + 4 * y + 3 * z = 12) : 
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end snowball_game_l792_792658


namespace isosceles_triangle_base_angle_l792_792239

-- Define the problem and the given conditions
theorem isosceles_triangle_base_angle (A B C : ℝ)
(h_triangle : A + B + C = 180)
(h_isosceles : (A = B ∨ B = C ∨ C = A))
(h_ratio : (A = B / 2 ∨ B = C / 2 ∨ C = A / 2)) :
(A = 45 ∨ A = 72) ∨ (B = 45 ∨ B = 72) ∨ (C = 45 ∨ C = 72) :=
sorry

end isosceles_triangle_base_angle_l792_792239


namespace tan_11_pi_over_4_l792_792791

theorem tan_11_pi_over_4 : Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Proof is omitted
  sorry

end tan_11_pi_over_4_l792_792791


namespace sum_of_products_on_ggx_l792_792603

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 6
  | 2 => 4
  | 4 => 2
  | _ => 0  -- We don't care for other values of g(x) as they are not used in the problem

theorem sum_of_products_on_ggx :
  let P1 := (2, g (g 2))
  let P2 := (4, g (g 4))
  (P1.1 * P1.2 + P2.1 * P2.2) = 20 :=
by
  -- We define P1 and P2 based on the definition of g and then prove the sum
  have h1: g (g 2) = 2 := by rfl
  have h2: g (g 4) = 4 := by rfl
  let P1 := (2, 2)
  let P2 := (4, 4)
  calc
    2 * 2 + 4 * 4 = 4 + 16 := by rfl
    ... = 20 := by rfl

end sum_of_products_on_ggx_l792_792603


namespace rectangle_condition_l792_792947

variable (A B C D : Type) [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace D]
variable [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
variable (AB AD BC CD : ℝ) (angleA angleB angleC angleD : ℝ)

noncomputable theory

-- Conditions:
variable (parallelogram : (AD = BC) ∧ (AB = CD))
variable (parallel : ∃ l m : Line A, is_parallel l m)
-- Given:
variable (angle_condition : angleA = angleB)

-- Proof goal:
theorem rectangle_condition : (parallel) ∧ (parallelogram) ∧ (angle_condition) → is_rectangle A B C D :=
by
  sorry

end rectangle_condition_l792_792947


namespace S_is_perfect_square_iff_p_eq_3_l792_792919

-- Define S_n_p as the sum of the p-th powers from 1 to n
def S (n p : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), k ^ p

-- Define the statement to be proved
theorem S_is_perfect_square_iff_p_eq_3 (n p : ℕ) : (∃ m : ℕ, S n p = m^2) ↔ p = 3 := by
  sorry

end S_is_perfect_square_iff_p_eq_3_l792_792919


namespace smallest_positive_multiple_of_37_l792_792664

theorem smallest_positive_multiple_of_37 (a : ℕ) (h1 : 37 * a ≡ 3 [MOD 101]) (h2 : ∀ b : ℕ, 0 < b ∧ (37 * b ≡ 3 [MOD 101]) → a ≤ b) : 37 * a = 1628 :=
sorry

end smallest_positive_multiple_of_37_l792_792664


namespace solution_to_inequality_l792_792798

def g (x : ℝ) : ℝ := ((3 * x - 5) * (x - 2)) / (2 * x)

theorem solution_to_inequality : {x : ℝ | g x ≥ 0} = {x : ℝ | x ∈ (Set.Iio 0) ∪ (Set.Ici (5 / 3))} :=
by
  sorry

end solution_to_inequality_l792_792798


namespace hand_towels_in_set_l792_792887

theorem hand_towels_in_set {h : ℕ}
  (hand_towel_sets : ℕ)
  (bath_towel_sets : ℕ)
  (hand_towel_sold : h * hand_towel_sets = 102)
  (bath_towel_sold : 6 * bath_towel_sets = 102)
  (same_sets_sold : hand_towel_sets = bath_towel_sets) :
  h = 17 := 
sorry

end hand_towels_in_set_l792_792887


namespace perfect_square_sum_modulo_l792_792545

/-- 
  Let T be the sum of all positive integers n such that n^2 + 18n - 1605 is a perfect square.
  The goal is to prove that T modulo 1000 is equal to 478.
--/
theorem perfect_square_sum_modulo :
  let T := (List.sum (List.filter (λ n, ∃ m : ℤ, n^2 + 18 * n - 1605 = m^2) (List.range 1000))) in
  T % 1000 = 478 :=
by
  sorry

end perfect_square_sum_modulo_l792_792545


namespace sequence_term_4_l792_792136

noncomputable def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * sequence n

theorem sequence_term_4 : sequence 3 = 8 := 
by
  sorry

end sequence_term_4_l792_792136


namespace intersection_M_N_l792_792089

section

def M (x : ℝ) : Prop := sqrt x < 4
def N (x : ℝ) : Prop := 3 * x >= 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | (1 / 3) ≤ x ∧ x < 16} :=
by
  sorry

end

end intersection_M_N_l792_792089


namespace digit_d_multiple_of_9_l792_792417

theorem digit_d_multiple_of_9 (d : ℕ) (hd : d = 1) : ∃ k : ℕ, (56780 + d) = 9 * k := by
  have : 56780 + d = 56780 + 1 := by rw [hd]
  rw [this]
  use 6313
  sorry

end digit_d_multiple_of_9_l792_792417


namespace Ivory_more_than_Josh_l792_792521

variable (Josh_riddles : ℕ := 8)
variable (Taso_riddles : ℕ := 24)
variable (Ivory_riddles : ℕ)

axiom Josh_riddles_condition : Josh_riddles = 8
axiom Taso_riddles_condition : Taso_riddles = 24
axiom Taso_twice_Ivory : Taso_riddles = 2 * Ivory_riddles

theorem Ivory_more_than_Josh : Ivory_riddles - Josh_riddles = 4 := by
  have h1 : Taso_riddles = 2 * Ivory_riddles := Taso_twice_Ivory
  have h2 : Taso_riddles = 24 := Taso_riddles_condition
  have t : Ivory_riddles = 24 / 2
  rw [Nat.div_eq_of_eq_mul_right _] at t
  sorry

end Ivory_more_than_Josh_l792_792521


namespace treasure_chest_total_value_l792_792723

def base7_to_base10 (n : Nat) : Nat :=
  let rec convert (n acc base : Nat) : Nat :=
    if n = 0 then acc
    else convert (n / 10) (acc + (n % 10) * base) (base * 7)
  convert n 0 1

theorem treasure_chest_total_value :
  base7_to_base10 5346 + base7_to_base10 6521 + base7_to_base10 320 = 4305 :=
by
  sorry

end treasure_chest_total_value_l792_792723


namespace number_of_valid_pairs_l792_792808

def no_zero_digit (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ (Nat.digits 10 n) → d ≠ 0

theorem number_of_valid_pairs : 
  (∃ n : ℕ,  n = (finset.card { p : ℕ × ℕ | (p.1 + p.2 = 500) ∧ no_zero_digit p.1 ∧ no_zero_digit p.2 })) :=
  sorry

end number_of_valid_pairs_l792_792808


namespace number_of_shelves_l792_792687

-- Define the initial conditions and required values
def initial_bears : ℕ := 6
def shipment_bears : ℕ := 18
def bears_per_shelf : ℕ := 6

-- Define the result we want to prove
theorem number_of_shelves : (initial_bears + shipment_bears) / bears_per_shelf = 4 :=
by
    -- Proof steps go here
    sorry

end number_of_shelves_l792_792687


namespace concyclic_condition_l792_792542

variables {α : Type*} [linear_ordered_field α]

structure Point (α : Type*) := (x y : α)
structure Triangle (α : Type*) := (A B C : Point α)

def is_acute_angle (t : Triangle α) : Prop := sorry  -- Definition for acute-angled triangle

def lies_on_altitude_from_A (H : Point α) (t : Triangle α ) : Prop := sorry -- H lies on the altitude from A

def reflection (P Q: Point α) : Point α := sorry -- Reflection of point P with respect to Q

def midpoint (P Q: Point α) : Point α := sorry -- Midpoint of points P and Q

def are_concyclic (pts : list (Point α)) : Prop := sorry -- Function to check if given list of points are concyclic

theorem concyclic_condition
  (t : Triangle α) (H H_b H_b' H_c H_c' : Point α)
  (h_acute : is_acute_angle t)
  (h_interior : ∃ (u v w : α), u + v + w = 1 ∧ 0 < u ∧ 0 < v ∧ 0 < w ∧ 
                                H = ⟨u * t.A.x + v * t.B.x + w * t.C.x, 
                                     u * t.A.y + v * t.B.y + w * t.C.y⟩)
  (H_b_reflection : H_b = reflection H (midpoint t.A t.C))
  (H_c_reflection : H_c = reflection H (midpoint t.A t.B))
  (H_b'_reflection : H_b' = reflection H (midpoint t.B t.C))
  (H_c'_reflection : H_c' = reflection H (midpoint t.C t.B)) :
  are_concyclic [H_b, H_b', H_c, H_c'] ↔ (∃ (P Q : Point α), P = Q) ∨ lies_on_altitude_from_A H t :=
sorry

end concyclic_condition_l792_792542


namespace greatest_distance_between_A_and_B_l792_792133

def A : Set ℂ := {z : ℂ | z ^ 4 = 16}
def B : Set ℂ := {z : ℂ | z ^ 4 - 16 * z ^ 3 + 64 * z ^ 2 - 16 * z + 16 = 0}

theorem greatest_distance_between_A_and_B (z_A : ℂ) (z_B : ℂ) 
  (hA : z_A ∈ A) (hB : z_B ∈ B) : 
  Real.dist z_A z_B ≤ 3 := 
sorry

end greatest_distance_between_A_and_B_l792_792133


namespace compute_expression_value_l792_792810

theorem compute_expression_value (x y : ℝ) (hxy : x ≠ y) 
  (h : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (xy + 1)) :
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (xy + 1) = 2 :=
by
  sorry

end compute_expression_value_l792_792810


namespace kangaroo_solution_l792_792765

noncomputable section

def perfect_squares : Set Nat := {n | ∃ (k : Nat), k * k = n}
def one_less_than_cubes : Set Nat := {n | ∃ (k : Nat), k * k * k - 1 = n}

variable (x : Nat) (y : Nat)

-- Definitions of conditions based on problem statements
def valid_1_across (n : Nat) : Prop := 
  n ∈ perfect_squares

def valid_2_down (n : Nat) : Prop :=
  n ∈ one_less_than_cubes

def match_last_digit (a b : Nat) : Prop :=
  a % 10 = b / 10

def pair_consistent (a b : Nat) : Prop :=
  valid_1_across a ∧ valid_2_down b ∧ match_last_digit a b

-- The main theorem to state the correctness of the final answer
theorem kangaroo_solution :
  ∃ a b, pair_consistent a b ∧ a = 81 ∧ b = 124 : ℕ → ℕ :=
  sorry

end kangaroo_solution_l792_792765


namespace equilateral_triangle_ratio_l792_792287

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let A := (s^2 * Real.sqrt 3) / 4
      P := 3 * s 
  in P / A = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end equilateral_triangle_ratio_l792_792287


namespace odd_expression_l792_792547

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem odd_expression (k m : ℤ) (o := 2 * k + 3) (n := 2 * m) :
  is_odd (o^2 + n * o) :=
by sorry

end odd_expression_l792_792547


namespace incorrect_minimum_positive_period_of_f_l792_792874

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem incorrect_minimum_positive_period_of_f :
  ¬ ∃ (T : ℝ), T > 0 ∧ ∀ x, f (x + T) = f x ∧ T = Real.pi := by
{
  sorry
}

end incorrect_minimum_positive_period_of_f_l792_792874


namespace proof_sets_l792_792561

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 3, 6}
def complement (s : Set ℕ) : Set ℕ := {x | x ∈ I ∧ x ∉ s}

theorem proof_sets :
  M ∩ (complement N) = {4, 5} ∧ {2, 7, 8} = complement (M ∪ N) :=
by
  sorry

end proof_sets_l792_792561


namespace yellow_marbles_at_least_zero_l792_792187

noncomputable def total_marbles := 30
def blue_marbles (n : ℕ) := n / 3
def red_marbles (n : ℕ) := n / 3
def green_marbles := 10
def yellow_marbles (n : ℕ) := n - ((2 * n) / 3 + 10)

-- Conditions
axiom h1 : total_marbles % 3 = 0
axiom h2 : total_marbles = 30

-- Prove the smallest number of yellow marbles is 0
theorem yellow_marbles_at_least_zero : yellow_marbles total_marbles = 0 := by
  sorry

end yellow_marbles_at_least_zero_l792_792187


namespace part1_part2_l792_792137

-- Definition section based on given conditions in part a)
variables {A B C : ℝ} {a b c : ℝ}
variables (angle_A angle_B angle_C side_a side_b side_c : ℝ)

-- Given conditions for both problems
axiom triangle_condition : 
  (√3 / 3) * side_b * real.sin angle_C + side_c * real.cos angle_B = side_a

-- Problem 1: Given a = 2, b = 1, prove that area of the triangle is √3 /2
noncomputable def area_of_triangle (a b : ℝ) (C : ℝ) := 
  1/2 * a * b * real.sin C

theorem part1 : side_a = 2 → side_b = 1 → 
  area_of_triangle 2 1 (real.pi / 3) = sqrt 3 / 2 :=
by
  intros
  unfold area_of_triangle
  sorry

-- Problem 2: Given c = 2, prove the perimeter of the triangle is in (2√3 + 2, 6]
theorem part2 : side_c = 2 → 
  4 * real.sin (angle_A + real.pi / 6) + 2 ∈ set.Icc (2 * sqrt 3 + 2) 6 :=
by
  intros
  unfold area_of_triangle
  sorry

end part1_part2_l792_792137


namespace symmetric_y_axis_l792_792021

theorem symmetric_y_axis (a b : ℝ) (h₁ : a = -4) (h₂ : b = 3) : a - b = -7 :=
by
  rw [h₁, h₂]
  norm_num

end symmetric_y_axis_l792_792021


namespace range_of_values_for_a_l792_792477

theorem range_of_values_for_a :
  ∀ a : ℝ, (A : set ℝ) = { x | |x - 1| ≤ a ∧ 0 < a } →
    (B : set ℝ) = { x | x^2 - 6 * x - 7 > 0 } →
    A ∩ B = ∅ →
    0 < a ∧ a ≤ 6 :=
by
  sorry

end range_of_values_for_a_l792_792477


namespace total_customers_l792_792724

def initial_customers : ℝ := 29.0    -- 29.0 initial customers
def lunch_rush_customers : ℝ := 20.0 -- Adds 20.0 customers during lunch rush
def additional_customers : ℝ := 34.0 -- Adds 34.0 more customers

theorem total_customers : (initial_customers + lunch_rush_customers + additional_customers) = 83.0 :=
by
  sorry

end total_customers_l792_792724


namespace find_angles_of_triangle_ABC_l792_792516

open Real

noncomputable def triangle_ABC (A B C D E F : ℝ) : Prop :=
  ∃ (ABC AED AFD : ℝ), 
    (D = (B + C) / 2) ∧   -- D is the midpoint of BC
    (AED = 1/14 * ABC) ∧  -- area(AED) is 1/14 of area(ABC)
    (AFD = 7/50 * ABC) ∧  -- area(AFD) is 7/50 of area(ABC)
    (angle A B C = 90) ∧  -- ∠A = 90°
    (angle B A C = arccos (3/5)) ∧  -- ∠B = arccos(3/5)
    (angle C A B = arccos (4/5))  -- ∠C = arccos(4/5)

theorem find_angles_of_triangle_ABC (A B C D E F : ℝ)
  (h : triangle_ABC A B C D E F) : 
  ∃ (a b c : ℝ), 
    angle A B C = 90 ∧
    angle B A C = arccos (3 / 5) ∧
    angle C A B = arccos (4 / 5) :=
sorry

end find_angles_of_triangle_ABC_l792_792516


namespace addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_l792_792761

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem addition_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a + b) :=
sorry

theorem subtraction_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a - b) :=
sorry

end addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_l792_792761


namespace area_of_triangle_AEB_is_10_l792_792954

noncomputable def area_of_triangle_AEB : Prop :=
  let AB := 8
  let BC := 4
  let DF := 2
  let GC := 1
  let area_AEB := (1/2 : ℝ) * AB * ((5 * BC) / 8) in
  area_AEB = 10

theorem area_of_triangle_AEB_is_10 :
  area_of_triangle_AEB :=
by
  sorry

end area_of_triangle_AEB_is_10_l792_792954


namespace compute_sum_of_squares_roots_l792_792748

-- p, q, and r are roots of 3*x^3 - 2*x^2 + 6*x + 15 = 0.
def P (x : ℝ) : Prop := 3*x^3 - 2*x^2 + 6*x + 15 = 0

theorem compute_sum_of_squares_roots :
  ∀ p q r : ℝ, P p ∧ P q ∧ P r → p^2 + q^2 + r^2 = -32 / 9 :=
by
  intros p q r h
  sorry

end compute_sum_of_squares_roots_l792_792748


namespace total_spent_l792_792143

def original_cost_vacuum_cleaner : ℝ := 250
def discount_vacuum_cleaner : ℝ := 0.20
def cost_dishwasher : ℝ := 450
def special_offer_discount : ℝ := 75

theorem total_spent :
  let discounted_vacuum_cleaner := original_cost_vacuum_cleaner * (1 - discount_vacuum_cleaner)
  let total_before_special := discounted_vacuum_cleaner + cost_dishwasher
  total_before_special - special_offer_discount = 575 := by
  sorry

end total_spent_l792_792143


namespace sum_of_digits_eq_4_l792_792097

def sum_digits (n : Nat) : Nat :=
  n.digits 10 |> List.sum

def first_year_after (y : Nat) (p : Nat -> Prop) : Nat :=
  (Nat.iterate (· + 1) (1 + y) (fun n => p n) y)

theorem sum_of_digits_eq_4 : first_year_after 2020 (fun n => sum_digits n = 4) = 2030 :=
  sorry

end sum_of_digits_eq_4_l792_792097


namespace final_weight_is_correct_l792_792343

-- Define the various weights after each week
def initial_weight : ℝ := 180
def first_week_removed : ℝ := 0.28 * initial_weight
def first_week_remaining : ℝ := initial_weight - first_week_removed
def second_week_removed : ℝ := 0.18 * first_week_remaining
def second_week_remaining : ℝ := first_week_remaining - second_week_removed
def third_week_removed : ℝ := 0.20 * second_week_remaining
def final_weight : ℝ := second_week_remaining - third_week_removed

-- State the theorem to prove the final weight equals 85.0176 kg
theorem final_weight_is_correct : final_weight = 85.0176 := 
by 
  sorry

end final_weight_is_correct_l792_792343


namespace intersection_on_incircle_l792_792222

variables (A B C A₁ B₁ A₂ B₂ : Point)
variable (ω : Circle)
variable [incircle : IsIncircle ω (Triangle.mk A B C)]

-- Conditions
variable (touchesA₁ : ω.TouchesSide (Triangle.side BC A B C) A₁)
variable (touchesB₁ : ω.TouchesSide (Triangle.side CA B C A) B₁)
variable (BA_eq_BA₂ : dist B A₁ = dist B A₂)
variable (AB_eq_AB₂ : dist A B₁ = dist A B₂)

-- Theorem to Prove
theorem intersection_on_incircle :
  ∃ P : Point, IsIntersection P (Line.mk A₁ A₂) (Line.mk B₁ B₂) ∧ ω.HasPoint P :=
sorry

end intersection_on_incircle_l792_792222


namespace quadratic_equal_roots_k_value_l792_792078

theorem quadratic_equal_roots_k_value (k : ℝ) :
  (∀ x : ℝ, x^2 - 8 * x - 4 * k = 0 → x^2 - 8 * x - 4 * k = 0 ∧ (0 : ℝ) = 0 ) →
  k = -4 :=
sorry

end quadratic_equal_roots_k_value_l792_792078


namespace vehicles_count_l792_792975

theorem vehicles_count (T : ℕ) : 
    2 * T + 3 * (2 * T) + (T / 2) + T = 180 → 
    T = 19 ∧ 2 * T = 38 ∧ 3 * (2 * T) = 114 ∧ (T / 2) = 9 := 
by 
    intros h
    sorry

end vehicles_count_l792_792975


namespace no_zeros_g_l792_792460

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := (x-1) * f(x) + 1/2

theorem no_zeros_g (f : ℝ → ℝ) (h_cont_diff : ∀ x : ℝ, ContinuousAt f x ∧ DifferentiableAt ℝ f x)
    (h_condition : ∀ x : ℝ, x * (derivative f x) + f x > derivative f x) :
    ∀ x > 1, g f x ≠ 0 := by
  sorry

end no_zeros_g_l792_792460


namespace vkontakte_status_l792_792314

variables (M I A P : Prop)

theorem vkontakte_status
  (h1 : M → (I ∧ A))
  (h2 : xor A P)
  (h3 : I ∨ M)
  (h4 : P ↔ I) :
  (¬ M ∧ I ∧ ¬ A ∧ P) :=
by
  sorry

end vkontakte_status_l792_792314


namespace ratio_of_areas_l792_792961

variables {P : Type} [metric_space P] [normed_group P] [normed_space ℝ P]
variables {A B C D E : P} {alpha : ℝ}

-- Definitions based on given conditions
def is_diameter (A B : P) (circle_center : P) : Prop :=
  dist A B = 2 * dist circle_center A ∧ dist A circle_center = dist B circle_center

def is_chord_parallel_to_diameter (C D A B : P) : Prop :=
  dist C D < dist A B ∧
  ∃ (mid_CD mid_AB : P),
    midpoint ℝ C D = mid_CD ∧ midpoint ℝ A B = mid_AB ∧
    ∀ (x : P), inner (x - mid_AB) (mid_AB - mid_CD) = 0

def intersect_at (AC BD : set P) (E : P) : Prop :=
  E ∈ AC ∧ E ∈ BD

def angle_AED (A E D : P) (alpha : ℝ) : Prop :=
  ∃ (θ : ℝ), θ = alpha ∧ angle A E D = θ

-- The theorem to be proven
theorem ratio_of_areas (circle_center : P)
  (h1 : is_diameter A B circle_center)
  (h2 : is_chord_parallel_to_diameter C D A B)
  (h3 : intersect_at ({A, C} : set P) ({B, D} : set P) E)
  (h4 : angle_AED A E D alpha) :
  (area_hyperplane A B E) / (area_hyperplane C D E) = (cos alpha) ^ 2 :=
sorry

end ratio_of_areas_l792_792961


namespace common_divisors_9240_8820_l792_792891

-- Define the prime factorizations given in the problem.
def pf_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def pf_8820 := [(2, 2), (3, 2), (5, 1), (7, 2)]

-- Define a function to calculate the gcd of two numbers given their prime factorizations.
def gcd_factorizations (pf1 pf2 : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
    List.filterMap (λ ⟨p, e1⟩ =>
      match List.lookup p pf2 with
      | some e2 => some (p, min e1 e2)
      | none => none
      end) pf1 

-- Define a function to compute the number of divisors from the prime factorization.
def num_divisors (pf: List (ℕ × ℕ)) : ℕ :=
    pf.foldl (λ acc ⟨_, e⟩ => acc * (e + 1)) 1

-- The Lean statement for the problem
theorem common_divisors_9240_8820 : 
    num_divisors (gcd_factorizations pf_9240 pf_8820) = 24 :=
by
    -- The proof goes here. We include sorry to indicate that the proof is omitted.
    sorry

end common_divisors_9240_8820_l792_792891


namespace minimum_value_of_expression_l792_792019

noncomputable def line_through_intersection (l1 l2 l3 : ℝ → ℝ) : Prop :=
∀ x y, l2 x y = 0 → l3 x y = 0 → l1 x y = 0

theorem minimum_value_of_expression {m n : ℝ} 
  (h_intersection : line_through_intersection (λ x y, m * x + y + n) (λ x y, x + y - 1) (λ x y, 3 * x - y - 7))
  (h_pos: m * n > 0) : 
  8 ≤ (1 / m) + (2 / n) := 
sorry -- proof omitted for brevity

end minimum_value_of_expression_l792_792019


namespace part1_a7_part1_S4_part2_a3_l792_792932

-- Definitions from the conditions
def a_4 : ℤ := 27
def q : ℤ := -3
def diff_5_1 : ℤ := 15
def diff_4_2 : ℤ := 6

-- Problem statement to translate the question and proof goal
theorem part1_a7 (a_1 : ℤ) (a_7 : ℤ) : (a_4 = a_1 * q ^ 3 → a_1 = -1) → (a_1 = -1 → a_7 = a_1 * q ^ 6) → a_7 = 729 :=
sorry

theorem part1_S4 (a_1 : ℤ) (S_4 : ℤ) : (a_4 = a_1 * q ^ 3 → a_1 = -1) → (a_1 = -1 → S_4 = a_1 * (1 - q ^ 4) / (1 - q)) → S_4 = -20 :=
sorry

theorem part2_a3 (a_1 q : ℤ) (a_3 : ℤ) : 
  (diff_5_1 = a_1 * q ^ 4 - a_1 → (a_1 = -16 ∧ q = 1/2) ∨ (a_1 = 1 ∧ q = 2) ) →
  (a_4 = a_1 * q ^ 3 → a_1 * q - a_1 * q → diff_4_2) →
  ((a_1 = -16 ∧ q = 1/2 → a_3 = a_1 * q ^ 2 → a_3 = -4) → (a_3 = -4 ∨ a_1 = 1 ∧ q = 2 → a_3 = 4)) :=
sorry

end part1_a7_part1_S4_part2_a3_l792_792932


namespace range_of_b_l792_792497

open Real

theorem range_of_b {b x x1 x2 : ℝ} 
  (h1 : ∀ x : ℝ, x^2 - b * x + 1 > 0 ↔ x < x1 ∨ x > x2)
  (h2 : x1 < 1)
  (h3 : x2 > 1) : 
  b > 2 := sorry

end range_of_b_l792_792497


namespace sum_fibonacci_series_is_5_over_19_l792_792543

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

noncomputable def sum_fibonacci_series : ℝ :=
∑' n, (fibonacci n : ℝ) / 5^n

theorem sum_fibonacci_series_is_5_over_19 : sum_fibonacci_series = 5 / 19 :=
sorry

end sum_fibonacci_series_is_5_over_19_l792_792543


namespace ellipse_problem_l792_792465

/-- Given an ellipse C with equation x² / a² + y² / b² = 1, where a > b > 0,
    which passes through the point Q (sqrt 2, 1) and has its right focus at F (sqrt 2, 0):

    (I) Prove the equation of the ellipse C is x² / 4 + y² / 2 = 1.
    (II) Considering the line l: y = k(x - 1) (k > 0), which intersects the x-axis at C (1, 0), 
         the y-axis at D (0, -k), and the ellipse C at points M and N. If the vectors CN and MD are equal,
         prove that k = sqrt 2 / 2 and the chord length |MN| = sqrt 42 / 2. -/
theorem ellipse_problem
    (a b : ℝ) (h : a > b) (ha : a > 0) (hb : b > 0)
    (h_eq : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x = √2 ∧ y = 1) ∨ (x = √2 ∧ y = 0)) :
  (∃ a b : ℝ, a = 2 ∧ b = √2 ∧ ∀ x y, x^2 / 4 + y^2 / 2 = 1 ↔ (x = √2 ∧ y = 1) ∨ (x = √2 ∧ y = 0)) ∧
  (∀ k : ℝ, k > 0 → (∃ MN_len : ℝ, k = √2 / 2 ∧ MN_len = √42 / 2)) :=
sorry

end ellipse_problem_l792_792465


namespace quadratic_intersect_points_distinct_quadratic_expression_chord_cd_length_l792_792430

noncomputable def quadratic_vertex (x m : ℝ) : ℝ := x^2 - 2 * m * x - m^2
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c  -- Helper definition for discriminant

theorem quadratic_intersect_points_distinct (m : ℝ) (hm : m ≠ 0) :
  discriminant 1 (-2 * m) (-m^2) > 0 :=
by
  have : discriminant 1 (-2 * m) (-m^2) = 8 * m^2,
  -- Calculate discriminant.
  { unfold discriminant, simp, ring, },
  -- Establish m ≠ 0 implies 8 * m^2 > 0.
  have H : 8 * m^2 > 0, from mul_pos (by norm_num) (pow_two_pos_of_ne_zero m hm),
  exact H

theorem quadratic_expression (x : ℝ) : 
  ∃ (m : ℝ), m = sqrt (1/2) ∨ m = -sqrt (1/2) ∧
  (quadratic_vertex x (sqrt (1/2)) = x^2 - sqrt 2 * x - 1/2 ∨
    quadratic_vertex x (-sqrt (1/2)) = x^2 + sqrt 2 * x - 1/2) :=
by
  sorry

theorem chord_cd_length :
  |2 * sqrt (1/2)| = sqrt 2 :=
by
  calc 
    |2 * sqrt (1/2)| = 2 * sqrt (1/2) : abs_of_pos (mul_pos two_pos (sqrt_pos (half_pos zero_lt_one))) 
    ... = sqrt 2 : by simp [sqrt_mul, sqrt_sq (by norm_num : (1:ℝ)/2 > 0)]

end quadratic_intersect_points_distinct_quadratic_expression_chord_cd_length_l792_792430


namespace factorial_binomial_theorem_l792_792997

-- Define the generalized power a^(n | h)
def generalized_power (a : ℝ) (n : ℕ) (h : ℝ) : ℝ :=
  (List.range n).foldl (λ acc i => acc * (a - i * h)) 1

-- Define the n choose k function
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem statement
theorem factorial_binomial_theorem (a b : ℝ) (n : ℕ) (h : ℝ) :
  (generalized_power (a + b) n h) = 
  (List.range (n + 1)).sum (λ i => (binomial_coefficient n i) * 
    generalized_power a (n - i) h * (b ^ (generalized_power b i h))) :=
sorry

end factorial_binomial_theorem_l792_792997


namespace domino_covering_4x4_chessboard_l792_792271

-- Define the main statement
theorem domino_covering_4x4_chessboard : 
  ∃ (n : ℕ), (n = 36) ∧ (∃ (coverings : list (list (ℕ × ℕ))), coverings.length = n) :=
begin
  sorry
end

end domino_covering_4x4_chessboard_l792_792271


namespace sin_XZY_l792_792119

noncomputable def sin_of_angle_XZY (α : ℝ) (h_cos_α : cos α = 3/5) : ℝ :=
  sin (π - α)

theorem sin_XZY (α β : ℝ) (h_α: α = ∠(Y, X, Z)) (h_β: β = π - α) (h_cos_α : cos (∠(Y, X, Z)) = 3 / 5) :
  sin β = 4 / 5 :=
by
  sorry

end sin_XZY_l792_792119


namespace point_in_second_quadrant_l792_792076

variable (m : ℝ)

/-- 
If point P(m-1, 3) is in the second quadrant, 
then a possible value of m is -1
--/
theorem point_in_second_quadrant (h1 : (m - 1 < 0)) : m = -1 :=
by sorry

end point_in_second_quadrant_l792_792076


namespace solve_for_x_l792_792596

theorem solve_for_x (x : ℕ) : 8 * 4^x = 2048 → x = 4 := by
  sorry

end solve_for_x_l792_792596


namespace ratio_perimeter_area_equilateral_triangle_l792_792284

theorem ratio_perimeter_area_equilateral_triangle (s : ℝ) (h_s : s = 6):
  let A := (s * s * Real.sqrt 3) / 4 in
  let P := 3 * s in
  P / A = 2 * Real.sqrt 3 / 3 :=
by
  -- The proof is omitted
  sorry

end ratio_perimeter_area_equilateral_triangle_l792_792284


namespace odd_c_perfect_square_no_even_c_infinitely_many_solutions_l792_792848

open Nat

/-- Problem (1): prove that if c is an odd number, then c is a perfect square given 
    c(a c + 1)^2 = (5c + 2b)(2c + b) -/
theorem odd_c_perfect_square (a b c : ℕ) (h_eq : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) (h_odd : Odd c) : ∃ k : ℕ, c = k^2 :=
  sorry

/-- Problem (2): prove that there does not exist an even number c that satisfies 
    c(a c + 1)^2 = (5c + 2b)(2c + b) for some a and b -/
theorem no_even_c (a b : ℕ) : ∀ c : ℕ, Even c → ¬ (c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) :=
  sorry

/-- Problem (3): prove that there are infinitely many solutions of positive integers 
    (a, b, c) that satisfy c(a c + 1)^2 = (5c + 2b)(2c + b) -/
theorem infinitely_many_solutions (n : ℕ) : ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b) :=
  sorry

end odd_c_perfect_square_no_even_c_infinitely_many_solutions_l792_792848


namespace constant_term_in_binomial_expansion_l792_792615

theorem constant_term_in_binomial_expansion (x : ℂ) (h : x ≠ 0) :
  let t := (2 / x - x^2 / 3) ^ 6 in
  (∃ c : ℚ, IsConstantTerm c t) → c = 80 / 3 :=
by
  sorry

end constant_term_in_binomial_expansion_l792_792615


namespace part1_part2_l792_792559

-- Define the complex number z
def z (a : ℝ) : ℂ := (1 - complex.i) * (a ^ 2 : ℂ) - 3 * (a : ℂ) + 2 + complex.i

-- Statement for question 1
theorem part1 (a : ℝ) (h : z a = complex.conj (z a)) : 
  (a = 1 ∨ a = -1) → (abs (z a) = 0 ∨ abs (z a) = 6) :=
sorry

-- Define conditions for question 2
def real_part_positive (a : ℝ) : Prop := a ^ 2 - 3 * a + 2 > 0
def imag_part_positive (a : ℝ) : Prop := 1 - a ^ 2 > 0

-- Statement for question 2
theorem part2 {a : ℝ} : 
  real_part_positive a ∧ imag_part_positive a → -1 < a ∧ a < 1 :=
sorry

end part1_part2_l792_792559


namespace minimal_distance_le_sqrt2_div_20_l792_792981

noncomputable theory

open_locale classical

theorem minimal_distance_le_sqrt2_div_20 (P : fin 2021 → ℝ × ℝ)
  (hP_cond : ∀ i, (0 ≤ (P i).1 ∧ 0 ≤ (P i).2))
  (h_centroid : (1 / 2021) * ∑ i, (P i).1 = 1 ∧ (1 / 2021) * ∑ i, (P i).2 = 1) :
  ∃ (i j : fin 2021), i ≠ j ∧ dist (P i) (P j) ≤ real.sqrt 2 / 20 :=
begin
  sorry
end

end minimal_distance_le_sqrt2_div_20_l792_792981


namespace find_a_l792_792861

noncomputable def geometric_sum_expression (n : ℕ) (a : ℝ) : ℝ :=
  3 * 2^n + a

theorem find_a (a : ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = geometric_sum_expression n a) → a = -3 :=
by
  sorry

end find_a_l792_792861


namespace intersection_A_complement_C_B_range_of_a_l792_792452

def A (x : ℝ) : Prop := x^2 - x - 12 < 0
def B (x : ℝ) : Prop := x^2 + 2x - 8 > 0
def C (a : ℝ) (x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a ≠ 0

-- Part (I) Statement
theorem intersection_A_complement_C_B :
  {x : ℝ | A x} ∩ {x : ℝ | ∀ x, C a x → ¬ B x} = λ x, -3 < x ∧ x ≤ 2 := sorry

-- Part (II) Statement
theorem range_of_a (a : ℝ) :
  (∀ x, (A x ∧ B x) → C a x) → 4/3 ≤ a ∧ a ≤ 2 := sorry

end intersection_A_complement_C_B_range_of_a_l792_792452


namespace geometric_mean_a_b_l792_792853

theorem geometric_mean_a_b : ∀ (a b : ℝ), a > 0 → b > 0 → Real.sqrt 3 = Real.sqrt (3^a * 3^b) → a + b = 1 :=
by
  intros a b ha hb hgeo
  sorry

end geometric_mean_a_b_l792_792853


namespace largest_triangle_angle_l792_792605

theorem largest_triangle_angle (h_ratio : ∃ (a b c : ℕ), a / b = 3 / 4 ∧ b / c = 4 / 9) 
  (h_external_angle : ∃ (θ1 θ2 θ3 θ4 : ℝ), θ1 = 3 * x ∧ θ2 = 4 * x ∧ θ3 = 9 * x ∧ θ4 = 3 * x ∧ θ1 + θ2 + θ3 = 180) :
  ∃ (θ3 : ℝ), θ3 = 101.25 := by
  sorry

end largest_triangle_angle_l792_792605


namespace percentage_girls_is_60_l792_792356

variable (boys girls : ℕ) (total_students : ℕ) (percentage_girls : ℚ)

def school := { boys := 300, girls := 450 }

/- We define the total number of students as the sum of boys and girls -/
def total_students (s : school) := s.boys + s.girls

/- We define the calculation of percentage of girls -/
def percentage_girls (s : school) : ℚ := (s.girls : ℚ) / (total_students s : ℚ) * 100

/- The theorem to be proved: The percentage of girls is 60% -/
theorem percentage_girls_is_60 (s : school) : percentage_girls s = 60 :=
by
  sorry

end percentage_girls_is_60_l792_792356


namespace fraction_girls_at_dance_l792_792736

/-- Definitions for conditions. --/
def dalton := (total_students : Nat) (ratio_boys_girls : Nat × Nat)
def berkeley := (total_students : Nat) (ratio_boys_girls : Nat × Nat)
def kingston := (total_students : Nat) (ratio_boys_girls : Nat × Nat)
def total_students_at_dance := dalton.total_students + berkeley.total_students + kingston.total_students
def total_girls_at_dance := 120 + 120 + 140

/-- Theorem to prove the fraction of students who are girls. --/
theorem fraction_girls_at_dance (dalton berkeley kingston : {{dalton.total_students = 300 ∧ dalton.ratio_boys_girls = (3, 2)} • {berkeley.total_students = 210 ∧ berkeley.ratio_boys_girls = (3, 4)} • {kingston.total_students = 240 ∧ kingston.ratio_boys_girls = (5, 7)}}) :
  (total_girls_at_dance (dalton berkeley kingston)) / (total_students_at_dance (dalton berkeley kingston)) = 38 / 75 :=
by sorry

end fraction_girls_at_dance_l792_792736


namespace smallest_positive_period_max_min_values_on_interval_l792_792471

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * cos x * sin x - (1 / 2) * cos (2 * x)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π := sorry

theorem max_min_values_on_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 →
  (f x ≤ 1 ∧ f x ≥ -1 / 2) ∧
  (f (π / 3) = 1 ∧ f 0 = -1 / 2) := sorry

end smallest_positive_period_max_min_values_on_interval_l792_792471


namespace chandler_weeks_to_save_l792_792370

theorem chandler_weeks_to_save :
  let birthday_money := 50 + 35 + 15 + 20
  let weekly_earnings := 18
  let bike_cost := 650
  ∃ x : ℕ, (birthday_money + x * weekly_earnings) ≥ bike_cost ∧ (birthday_money + (x - 1) * weekly_earnings) < bike_cost := 
by
  sorry

end chandler_weeks_to_save_l792_792370


namespace monotonically_increasing_range_k_l792_792758

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem monotonically_increasing_range_k :
  (∀ x > 1, deriv (f k) x ≥ 0) → k ≥ 1 :=
sorry

end monotonically_increasing_range_k_l792_792758


namespace f_2009_of_17_l792_792079

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

def f (n : ℕ) : ℕ := sum_of_digits ((n ^ 2) + 1)

def f_seq (k n : ℕ) : ℕ :=
  if k = 1 then f n else f (f_seq (k - 1) n)

theorem f_2009_of_17 : f_seq 2009 17 = 5 := sorry

end f_2009_of_17_l792_792079


namespace no_natural_power_small_base_l792_792832

theorem no_natural_power_small_base (n : ℕ) (h_len : n.digits 10 ≠ [] ∧ n.digits 10.length = 1000) (h_no_zero : ∀ d ∈ n.digits 10, d ≠ 0) :
  ∃ m ≤ n, ∀ a < 500, ∀ k ≥ 2, m ≠ a^k := 
by
  sorry

end no_natural_power_small_base_l792_792832


namespace tan_of_11pi_over_4_is_neg1_l792_792792

noncomputable def tan_periodic : Real := 2 * Real.pi

theorem tan_of_11pi_over_4_is_neg1 :
  Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Angle normalization using periodicity of tangent function
  have h1 : Real.tan (11 * Real.pi / 4) = Real.tan (11 * Real.pi / 4 - 2 * Real.pi) := 
    by rw [Real.tan_periodic]
  -- Further normalization
  have h2 : 11 * Real.pi / 4 - 2 * Real.pi = 3 * Real.pi / 4 := sorry
  -- Evaluate tangent at the simplified angle
  have h3 : Real.tan (3 * Real.pi / 4) = -Real.tan (Real.pi / 4) := sorry
  -- Known value of tangent at common angle
  have h4 : Real.tan (Real.pi / 4) = 1 := by simpl tan
  rw [h2, h3, h4]
  norm_num

end tan_of_11pi_over_4_is_neg1_l792_792792


namespace find_m_find_PA_plus_PB_when_m_is_3_l792_792129

/-
Given:
1. The parametric equation of line l:
   x = m - √2 * t
   y = √5 + √2 * t
2. The polar equation of circle C: ρ = 2√5sinθ
3. The length of the chord intercepted by line l on circle C is √2
4. The points A and B where circle C intersects line l
5. Point P has coordinates (m, √5) where m>0
Prove that:
1. The possible values of m are 3 or -3
2. When m = 3, |PA| + |PB| = 3√2
-/

theorem find_m (m t : ℝ) :
  (∀ t, (m - sqrt 2 * t) ^ 2 + (sqrt 5 + sqrt 2 * t) ^ 2 = 5) →
  (m = 3 ∨ m = -3) :=
sorry

theorem find_PA_plus_PB_when_m_is_3 (t1 t2 : ℝ) :
  (m = 3) →
  (m > 0) →
  (t1 + t2 = 3 * sqrt 2 / 2) →
  (t1 * t2 = 1) →
  |PA| + |PB| = 3 * sqrt 2 :=
sorry

end find_m_find_PA_plus_PB_when_m_is_3_l792_792129


namespace find_alpha_l792_792870

noncomputable def is_even_function (f: ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def is_monotonically_decreasing (f: ℝ → ℝ) : Prop := ∀ ⦃x y⦄, 0 < x → x < y → f y < f x

noncomputable def f (α : ℤ) (x : ℝ) : ℝ := x ^ (α.nat_abs ^ 2 - 2 * α.nat_abs - 3)

theorem find_alpha (α : ℤ) : 
  α.nat_abs ^ 2 - 2 * α.nat_abs - 3 < 0 ∧ 
  (α.nat_abs ^ 2 - 2 * α.nat_abs - 3) % 2 = 0 ∧ 
  is_even_function (f α) ∧ 
  is_monotonically_decreasing (f α) → 
  α = 1 := 
sorry

end find_alpha_l792_792870


namespace dentist_fraction_l792_792581

theorem dentist_fraction :
  ∃ F : ℚ, (1 / 6) * (32 - 8) = F * (32 + 8) ∧ F = 1 / 10 :=
by
  -- We state the conditions as definitions for clarity
  let A := 32
  let age_8_years_ago := A - 8
  let age_8_years_hence := A + 8
  let F := (1 / 6) * age_8_years_ago / age_8_years_hence
  -- We need to prove that F = 1 / 10
  use F
  use (1 / 10)
  split
  · sorry
  · sorry

end dentist_fraction_l792_792581


namespace max_distance_l792_792454

-- Definition for point A
def A : ℝ × ℝ := (1, 3 / 2)

-- Definitions matching conditions (1) and (2)
axiom a_gt_b_gt_0 : ∃ a b : ℝ, a > b ∧ b > 0 ∧ (2 * a = 4) 

-- Definition matching condition (3)
def ellipse_eq (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 

noncomputable def b_val := Real.sqrt 3

-- Foci coordinates
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Point on ellipse parameterized
def parametrized_point (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Distance function between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Max distance proof stub
theorem max_distance (θ : ℝ) :
  ∃ maxValue : ℝ, ∀ (θ : ℝ), distance (parametrized_point θ) (0, 1 / 2) ≤ maxValue 
  ∧ maxValue = Real.sqrt 5 :=
sorry

end max_distance_l792_792454


namespace find_side_c_in_triangle_l792_792965

theorem find_side_c_in_triangle 
  (a b c : ℝ) (B : ℝ) (S : ℝ)
  (hB : B = 60)
  (ha : a = 4)
  (hS : S = 20 * real.sqrt 3) :
  S = 1/2 * a * c * (real.sin (B * real.pi / 180)) →
  c = 20 :=
by
  intros
  simp [hB, ha, hS] at *
  sorry

end find_side_c_in_triangle_l792_792965


namespace area_equivalence_l792_792563

variable (A B C M K : Point)
variable (p : Line)
variable (triangle_ABC : Triangle A B C)
variable (intersect_M : p.intersect (A, B) = M)
variable (intersect_K : p.intersect (B, C) = K)

theorem area_equivalence
  (area_triangle_MBK : area (triangle M B K) = area (quadrilateral A M K C)) :
  (|MB| + |BK|) / (|AM| + |CA| + |KC|) ≥ 1 / 3 :=
sorry

end area_equivalence_l792_792563


namespace problem_part1_problem_part2_l792_792013

noncomputable def arithmetic_sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + 2

theorem problem_part1 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) (h2 : S 2 = a 3) (h3 : arithmetic_sequence a) :
  a 2 = 4 := 
sorry

theorem problem_part2 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) (h2 : S 2 = a 3) (h3 : arithmetic_sequence a) 
  (h4 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2) :
  S 10 = 110 :=
sorry

end problem_part1_problem_part2_l792_792013


namespace total_amount_theorem_l792_792978

noncomputable def total_amount (kim_days : ℕ) (david_days : ℕ) (kim_share : ℕ) : ℕ :=
  let kim_rate := (1 : ℚ) / kim_days
  let david_rate := (1 : ℚ) / david_days
  let combined_rate := kim_rate + david_rate
  let kim_work_share := kim_rate / combined_rate
  let T := kim_share * (1 : ℚ) / kim_work_share
  T.natAbs

theorem total_amount_theorem :
  total_amount 3 2 60 = 150 :=
by
  simp [total_amount]
  sorry

end total_amount_theorem_l792_792978


namespace solve_for_x_l792_792212

theorem solve_for_x : (∃ x : ℚ, x = 45 / (8 - 3 / 4) ∧ x = 180 / 29) := 
by
  exists (180 / 29)
  split
  {
    sorry
  }
  {
    rfl
  }

end solve_for_x_l792_792212


namespace common_divisors_l792_792896

theorem common_divisors (a b : ℕ) (ha : a = 9240) (hb : b = 8820) : 
  let g := Nat.gcd a b in 
  g = 420 ∧ Nat.divisors 420 = 24 :=
by
  have gcd_ab := Nat.gcd_n at ha hb
  have fact := Nat.factorize 420
  have divisors_420: ∀ k : Nat, g = 420 ∧ k = 24 := sorry
  exact divisors_420 24

end common_divisors_l792_792896


namespace radius_of_cookie_l792_792218

theorem radius_of_cookie : 
  ∀ x y : ℝ, (x^2 + y^2 - 6.5 = x + 3 * y) → 
  ∃ (c : ℝ × ℝ) (r : ℝ), r = 3 ∧ (x - c.1)^2 + (y - c.2)^2 = r^2 :=
by {
  sorry
}

end radius_of_cookie_l792_792218


namespace ellipse_equation_l792_792857

theorem ellipse_equation (e : ℝ) (c : ℝ) (a b : ℝ) (x y : ℝ) 
  (h_eccentricity : e = 1 / 2)
  (h_foci : ∀ x y: ℝ, (x = -3 ∧ y = 0) ∨ (x = 3 ∧ y = 0))
  (h_c : 2 * c = 6)
  (h_linear_eccentricity : c = a * e)
  (h_b_square : b^2 = a^2 * (1 - e^2)) :
  ∀ (x y : ℝ), (x, y) ∈ set_univ ℝ × set_univ ℝ → (x^2 / (a^2) + y^2 / (b^2) = 1) :=
begin
  intros x y _,
  sorry
end

end ellipse_equation_l792_792857


namespace Megan_not_lead_plays_l792_792573

def total_plays : ℕ := 100
def lead_percentage : ℝ := 0.80
def lead_plays : ℕ := (total_plays : ℝ * lead_percentage).toNat
def not_lead_plays : ℕ := total_plays - lead_plays

theorem Megan_not_lead_plays : not_lead_plays = 20 := by
  sorry

end Megan_not_lead_plays_l792_792573


namespace trajectory_of_C_l792_792508

-- Definitions of points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 3)

-- Definition of point C as a linear combination of points A and B
def C (α β : ℝ) : ℝ × ℝ := (α * A.1 + β * B.1, α * A.2 + β * B.2)

-- The main theorem statement to prove the equation of the trajectory of point C
theorem trajectory_of_C (x y α β : ℝ)
  (h_cond : α + β = 1)
  (h_C : (x, y) = C α β) : 
  x + 2*y = 5 := 
sorry -- Proof to be skipped

end trajectory_of_C_l792_792508


namespace sum_of_areas_of_sixteen_disks_l792_792209

theorem sum_of_areas_of_sixteen_disks :
  let r := 1 - (2:ℝ).sqrt
  let area_one_disk := r^2 * Real.pi
  let total_area := 16 * area_one_disk
  total_area = Real.pi * (48 - 32 * (2:ℝ).sqrt) :=
by
  sorry

end sum_of_areas_of_sixteen_disks_l792_792209


namespace volume_ratio_l792_792712

noncomputable theory
open real

-- Definitions and conditions provided within the problem
def angle_inequality (alpha : ℝ) := alpha ≥ real.pi / 3

-- Main theorem statement about the ratio of volumes
theorem volume_ratio (α : ℝ) (hα : angle_inequality α) :
  let ratio := (π * tan ((π + α) / 4)) / (2 * (cos (α / 2))^3 * sin α) in
  ratio = (π * tan ((π + α) / 4)) / (2 * (cos (α / 2))^3 * sin α) :=
by {
  sorry
}

end volume_ratio_l792_792712


namespace triangle_DEF_is_acute_l792_792684

-- Definitions of the conditions
variables {A B C D E F : Type} [EuclideanGeometry A B C D E F]

/-- Given that a circle is inscribed in triangle ABC and touches side BC at D, side AC at E, and side AB at F,
    show that triangle DEF is acute. -/
theorem triangle_DEF_is_acute
  (h_circle_inscribed : circle.inscribed_in_triangle A B C)
  (h_D_on_BC : circle.touches_side_at h_circle_inscribed BC D)
  (h_E_on_AC : circle.touches_side_at h_circle_inscribed AC E)
  (h_F_on_AB : circle.touches_side_at h_circle_inscribed AB F) :
  acute_triangle D E F :=
sorry

end triangle_DEF_is_acute_l792_792684


namespace replace_floor_cost_l792_792644

def cost_of_removal := 50
def cost_per_sqft := 1.25
def room_length := 8
def room_width := 7

def total_cost_to_replace_floor : ℝ :=
  cost_of_removal + (cost_per_sqft * (room_length * room_width))

theorem replace_floor_cost :
  total_cost_to_replace_floor = 120 :=
by
  sorry

end replace_floor_cost_l792_792644


namespace symm_points_on_circle_exists_l792_792003

theorem symm_points_on_circle_exists (A B C : ℝ) (O : ℝ) (circle : set (ℝ × ℝ)) 
  (h_circle : ∀ (X : ℝ × ℝ), X ∈ circle ↔ (X.1 - O) ^ 2 + (X.2) ^ 2 = (A - O) ^ 2)
  (h_C_on_diameter : C ∈ set.Icc A B) :
  ∃ X Y : ℝ × ℝ, 
    X ∈ circle ∧ Y ∈ circle ∧
    Y.2 = -X.2 ∧ 
    ((C, 0 : ℝ) - Y).snd = (A - X).fst ∧ -- CY ⊥ XA
    (X.1 = X.1 ∧ Y.1 = Y.1) := -- ensuring X and Y are symmetrically placed w.r.t diameter AB
sorry -- Proof not required

end symm_points_on_circle_exists_l792_792003


namespace average_of_rest_students_l792_792931

theorem average_of_rest_students 
    (total_students : Nat)
    (scores_95_count : Nat)
    (scores_0_count : Nat)
    (total_class_average : ℝ) 
    (total_class_count : Nat) 
    (marks_per_95 : Nat) 
    (zero_marks_count : Nat) 
    : total_students = 25 ->
      scores_95_count = 5 -> 
      scores_0_count = 3 ->
      total_class_average = 49.6 ->
      total_class_count = 25 ->
      marks_per_95 = 95 ->
      zero_marks_count = 0 -> 
      let rest_students_count := total_students - scores_95_count - scores_0_count in
      let total_marks_95 := scores_95_count * marks_per_95 in
      let total_marks_0 := scores_0_count * zero_marks_count in
      let total_marks_class := total_class_average * total_class_count in
      let total_marks_rest := total_marks_class - total_marks_95 - total_marks_0 in
      let average_rest_students := total_marks_rest / rest_students_count in
      average_rest_students = 45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  let rest_students_count := total_students - scores_95_count - scores_0_count
  let total_marks_95 := scores_95_count * marks_per_95
  let total_marks_0 := scores_0_count * zero_marks_count
  let total_marks_class := total_class_average * total_class_count
  let total_marks_rest := total_marks_class - total_marks_95 - total_marks_0
  let average_rest_students := total_marks_rest / rest_students_count
  sorry

end average_of_rest_students_l792_792931


namespace ten_men_ten_boys_work_time_l792_792690

theorem ten_men_ten_boys_work_time :
  (∀ (total_work : ℝ) (man_work_rate boy_work_rate : ℝ),
    15 * 10 * man_work_rate = total_work ∧
    20 * 15 * boy_work_rate = total_work →
    (10 * man_work_rate + 10 * boy_work_rate) * 10 = total_work) :=
by
  sorry

end ten_men_ten_boys_work_time_l792_792690


namespace line_through_ellipse_and_midpoint_l792_792329

theorem line_through_ellipse_and_midpoint :
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ (x + y) = 0) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      (x₁ + x₂ = 2 ∧ y₁ + y₂ = 1) ∧
      (x₁^2 / 2 + y₁^2 = 1 ∧ x₂^2 / 2 + y₂^2 = 1) ∧
      l x₁ y₁ ∧ l x₂ y₂ ∧
      ∀ (mx my : ℝ), (mx, my) = (1, 0.5) → (mx = (x₁ + x₂) / 2 ∧ my = (y₁ + y₂) / 2))
  := sorry

end line_through_ellipse_and_midpoint_l792_792329


namespace greatest_fifty_supportive_X_l792_792999

def fifty_supportive (X : ℝ) : Prop :=
∀ (a : Fin 50 → ℝ),
  (∑ i, a i).floor = (∑ i, a i) →
  ∃ i, |a i - 0.5| ≥ X

theorem greatest_fifty_supportive_X :
  ∀ X : ℝ, fifty_supportive X ↔ X ≤ 0.01 := sorry

end greatest_fifty_supportive_X_l792_792999


namespace proof_perpendicular_l792_792941

variable {Point : Type} [Plane Point]
open Point

-- Definitions in the conditions
def is_midpoint (F A B : Point) : Prop := dist A F = dist F B
def is_perpendicular (F D C : Point) : Prop := ∠ F D C = 90
def is_isosceles_triangle (A B C : Point) : Prop := dist A C = dist B C

-- Main theorem
theorem proof_perpendicular {A B C F D G : Point} 
  (h_isosceles : is_isosceles_triangle A B C)
  (h_F_midpoint : is_midpoint F A B)
  (h_D_perpendicular : is_perpendicular F D C)
  (h_G_midpoint : is_midpoint G F D) :
  is_perpendicular A D G :=
sorry

end proof_perpendicular_l792_792941


namespace find_const_func_l792_792392

/-
Define the greatest integer function.
-/
def floor (x : ℝ) : ℤ := Int.floor x

/-
Define the functional equation condition
-/
def func_eq (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x * (floor y)) = ((floor (f x) : ℝ) * f y)

/-
Statement of the theorem
-/
theorem find_const_func (f : ℝ → ℝ) (h : func_eq f) :
  ∃ c : ℝ, (1 ≤ c ∧ c < 2) ∧ (∀ x : ℝ, f x = c) :=
sorry

end find_const_func_l792_792392


namespace BE_length_is_1_point_5_l792_792201

-- Define the properties of the points and rectangle
variables (A B C D E F : Point)
variables (lengthAB lengthAD : ℝ)
variables (AE_length CF_length : ℝ)
variables (BE_length : ℝ)
variables (fold_line_1 fold_line_2 : Line)

-- Define the conditions
axiom rect_properties : lengthAB = 2 ∧ lengthAD = 1
axiom point_positions : AE_length = 0.5 ∧ CF_length = 0.5
axiom fold_conditions : fold_line_1 = Line.mk D E ∧ fold_line_2 = Line.mk D F

-- Define the goal
axiom fold_result : side_AD_coincides_with_CD : fold AD CD fold_line_1 fold_line_2 → BE_length = 1.5

-- The theorem to prove
theorem BE_length_is_1_point_5:
  (lengthAB = 2 ∧ lengthAD = 1) →
  (AE_length = 0.5 ∧ CF_length = 0.5) →
  fold AD CD fold_line_1 fold_line_2 →
  BE_length = 1.5 :=
sorry

end BE_length_is_1_point_5_l792_792201


namespace angle_B1MC1_l792_792139

theorem angle_B1MC1 
  (ABC : Triangle)
  (A₁ B₁ C₁ M : Point)
  (hB : ∠B = 120°)
  (hAA₁ : AngleBisector A A₁ ABC)
  (hBB₁ : AngleBisector B B₁ ABC)
  (hCC₁ : AngleBisector C C₁ ABC)
  (hM_intersect : SegmentIntersection A₁ B₁ (AngleBisector C₁ C ABC) M)
  : ∠B₁ M C₁ = 60° := 
sorry

end angle_B1MC1_l792_792139


namespace ways_to_select_numbers_sum_even_equals_sum_odd_l792_792822

theorem ways_to_select_numbers_sum_even_equals_sum_odd :
  let numbers := {1, 2, 3, 4, 5, 6, 7}
  let evens := {n ∈ numbers | n % 2 = 0}
  let odds := {n ∈ numbers | n % 2 = 1}
  (card {s ∈ powerset numbers | (∑ x in s ∩ evens, x) = (∑ x in s ∩ odds, x)} = 7) :=
by
  sorry

end ways_to_select_numbers_sum_even_equals_sum_odd_l792_792822


namespace num_inhabitants_range_l792_792733

noncomputable def inhabitants_in_range (num_clubs : ℕ) (max_club_members : ℕ)
  (club_pairs_inhabitants : (ℕ × ℕ) → ℕ) : Prop :=
  num_clubs = 50 ∧ max_club_members = 55 ∧
  (∀ (i j : ℕ), i < j ∧ i ≤ num_clubs ∧ j ≤ num_clubs → club_pairs_inhabitants (i, j) = 1) ∧
  (∃ (num_inhabitants : ℕ), (num_inhabitants ≥ 1225) ∧ (num_inhabitants ≤ 1525))

theorem num_inhabitants_range : inhabitants_in_range 50 55 (λ (p : ℕ × ℕ), if p.1 < p.2 ∧ p.1 ≤ 50 ∧ p.2 ≤ 50 then 1 else 0) :=
by { sorry }

end num_inhabitants_range_l792_792733


namespace largest_root_divisible_by_17_l792_792165

theorem largest_root_divisible_by_17 (a : ℝ) (h : Polynomial.eval a (Polynomial.C 1 + Polynomial.C (-3) * Polynomial.X^2 + Polynomial.X^3) = 0) (root_large : ∀ x ∈ {b | Polynomial.eval b (Polynomial.C 1 + Polynomial.C (-3) * Polynomial.X^2 + Polynomial.X^3) = 0}, x ≤ a) :
  a^1788 % 17 = 0 ∧ a^1988 % 17 = 0 :=
by
  sorry

end largest_root_divisible_by_17_l792_792165


namespace solve_for_x_l792_792211

theorem solve_for_x : ∃ x : ℝ, 2 * 2^x + (8 * 8^x)^(1 / 3) = 32 ∧ x = 3 :=
by
  use 3
  sorry

end solve_for_x_l792_792211


namespace number_of_valid_pairs_l792_792807

def no_zero_digit (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ (Nat.digits 10 n) → d ≠ 0

theorem number_of_valid_pairs : 
  (∃ n : ℕ,  n = (finset.card { p : ℕ × ℕ | (p.1 + p.2 = 500) ∧ no_zero_digit p.1 ∧ no_zero_digit p.2 })) :=
  sorry

end number_of_valid_pairs_l792_792807


namespace math_problem_l792_792438

open Real

noncomputable def ellipse_equation : Prop :=
  let a := 4
  let b := sqrt 12
  let c := 2
  a > b ∧ b > 0 ∧ (c / a = 1 / 2) ∧ (a - c = 2) ∧
  equation : eq ((λ x y, x^2 / 16 + y^2 / 12) = 1)

noncomputable def circle_equation : Prop :=
  let A := (-4, 0)
  let F := (2, 0)
  let t := assume_t_value
  let N := (8, t)
  ∃ e:ℝ, ∃ f:ℝ, equation : eq ((λ x y, x^2 + y^2 + 2 * x + e * y - 8) = 0)

noncomputable def triangle_area : Prop :=
  let k := 1 / 2
  let A := (-4, 0)
  let B := (4, 0)
  let M := (12 - 16 * k^2)/(3 + 4 * k^2), 24*k/(3 + 4*k^2)
  let AMB := -sqrt(65) / 65
  equation : eq ((|vector3| AM area A B M) = 12)

theorem math_problem :
  ellipse_equation ∧ circle_equation ∧ triangle_area := sorry

end math_problem_l792_792438


namespace line_through_circle_center_l792_792398

theorem line_through_circle_center
  (C : ℝ × ℝ)
  (hC : C = (-1, 0))
  (hCircle : ∀ (x y : ℝ), x^2 + 2 * x + y^2 = 0 → (x, y) = (-1, 0))
  (hPerpendicular : ∀ (m₁ m₂ : ℝ), (m₁ * m₂ = -1) → m₁ = -1 → m₂ = 1)
  (line_eq : ∀ (x y : ℝ), y = x + 1)
  : ∀ (x y : ℝ), x - y + 1 = 0 :=
sorry

end line_through_circle_center_l792_792398


namespace tom_apples_left_l792_792262

theorem tom_apples_left (slices_per_apple : ℕ) (apples : ℕ) (fraction_given_to_jerry : ℚ) :
  slices_per_apple = 8 →
  apples = 2 →
  fraction_given_to_jerry = 3 / 8 →
  (let total_slices := slices_per_apple * apples in
   let slices_given_to_jerry := total_slices * fraction_given_to_jerry in
   let slices_after_giving := total_slices - slices_given_to_jerry.toNat in
   let slices_left := slices_after_giving / 2 in
   slices_left) = 5 :=
by
  intros
  sorry

end tom_apples_left_l792_792262


namespace probability_not_red_l792_792695

-- Definitions for the conditions
def red_jelly_beans : Nat := 7
def green_jelly_beans : Nat := 8
def yellow_jelly_beans : Nat := 9
def blue_jelly_beans : Nat := 10

-- Define total number of jelly beans and the number that are not red
def total_jelly_beans : Nat := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans
def non_red_jelly_beans : Nat := green_jelly_beans + yellow_jelly_beans + blue_jelly_beans

-- Desired probability
def desired_probability : Rat := non_red_jelly_beans.to_Rat / total_jelly_beans.to_Rat

-- Proof statement
theorem probability_not_red : desired_probability = 27 / 34 := by
  sorry

end probability_not_red_l792_792695


namespace AF_perpendicular_BF_l792_792511

open_locale big_operators

-- Declare the points in the triangle.
noncomputable def point := ℝ × ℝ -- representing points on the plane ℝ²
variables (A B C D E F : point)

-- Declare the given conditions
variables (hAC_gt_AB : dist A C > dist A B)
variables (hD_perp_BC : ∃ D : point, ∀ x : ℝ, line A (D x))
variables (hE_perp_AC : ∃ E : point, ∀ x : ℝ, line D (E x))
variables (hF : ∃ F : point, F ∈ line D E ∧ dist E F * dist A D = dist A B * dist D E)

-- Statement to be proven
theorem AF_perpendicular_BF (h : dist A F * dist B F = dist A B * dist C F) : 
  ∠ A F B = 90 :=
sorry

end AF_perpendicular_BF_l792_792511


namespace part_I_part_II_l792_792420

-- Define the sets A and B for the given conditions
def setA : Set ℝ := {x | -3 ≤ x - 2 ∧ x - 2 ≤ 1}
def setB (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Part (Ⅰ) When a = 1, find A ∩ B
theorem part_I (a : ℝ) (ha : a = 1) :
  (setA ∩ setB a) = {x | 0 ≤ x ∧ x ≤ 3} :=
by
  sorry

-- Part (Ⅱ) If A ∪ B = A, find the range of real number a
theorem part_II : 
  (∀ a : ℝ, setA ∪ setB a = setA → 0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end part_I_part_II_l792_792420


namespace repeating_decimal_subtraction_l792_792290

theorem repeating_decimal_subtraction :
  let x := 0.↦845
  let y := 0.↦267
  let z := 0.↦159
  let xf := 845 / 999
  let yf := 267 / 999
  let zf := 159 / 999
  x = xf →
  y = yf →
  z = zf →
  (xf - yf - zf = 419 / 999) := 
by
  intros x y z xf yf zf hx hy hz
  rw [hx, hy, hz]
  sorry

end repeating_decimal_subtraction_l792_792290


namespace exists_regular_2n_gon_l792_792151

-- Define basic geometric primitives
structure Point :=
  (x : ℝ) (y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define regular polygon inscribed in a circle
structure RegularPolygon :=
  (circumcircle : Circle)
  (vertices : Finset Point)
  (regular : vertices.card ≥ 4 ∧ ∀ v₁ v₂ v₃ ∈ vertices, dist v₁ v₂ = dist v₃ (circumcircle.center))

-- Define the problem and conditions
theorem exists_regular_2n_gon {n : ℕ} (h : n ≥ 4)
  (ω_P ω_Q ω : Circle) 
  (P Q : RegularPolygon)
  (A B C D E F : Point)
  (h1 : ω_P.radius = 1 ∧ ω_Q.radius = 1 ∧ ω.radius = 1)
  (h2 : P.circumcircle = ω_P ∧ Q.circumcircle = ω_Q)
  (h3 : A ∈ P.vertices ∧ B ∈ P.vertices ∧ A ∈ Q.vertices ∧ B ∈ Q.vertices)
  (h4 : ω_P.center ≠ ω_Q.center)
  (h5 : ω.center ≠ ω_P.center ∧ ω.center ≠ ω_Q.center)
  (h6 : C ∈ ω.vertices ∧ D ∈ ω.vertices ∧ E ∈ ω.vertices ∧ F ∈ ω.vertices)
  (h7 : A ∉ ω.vertices ∧ B ∈ ω.vertices)
  (h8 : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A) : 
  ∃ R : RegularPolygon, R.circumcircle.radius = 1 ∧ ∃ (k : Finset ℕ), k.card = n ∧ R.vertices.card = 2 * n ∧ {C, D, E, F} ⊆ R.vertices :=
by
  sorry

end exists_regular_2n_gon_l792_792151


namespace minimum_value_of_function_l792_792831

theorem minimum_value_of_function (x : ℝ) (hx : x > 5 / 4) : 
  ∃ y, y = 4 * x + 1 / (4 * x - 5) ∧ y = 7 :=
sorry

end minimum_value_of_function_l792_792831


namespace sum_a7_a8_a9_l792_792159

variable {a : ℕ → ℤ} -- Let's say the sequence is a function from natural numbers to integers

-- Definitions of the sums S₃ and S₆
def S (n : ℕ) : ℤ := ∑ i in Finset.range n, a i
def S₃ : ℤ := S 3
def S₆ : ℤ := S 6

-- Conditions given
variable (h₁ : S₃ = 7)
variable (h₂ : S₆ = 16)

-- Assertion (what we need to prove)
theorem sum_a7_a8_a9 : a 7 + a 8 + a 9 = 11 :=
by
  sorry

end sum_a7_a8_a9_l792_792159


namespace z_in_first_quadrant_l792_792223

noncomputable def z (z : ℂ) : Prop :=
  (z / (1 - z)) = 2 * complex.I

theorem z_in_first_quadrant (z : ℂ) (hz : z / (1 - z) = 2 * complex.I) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end z_in_first_quadrant_l792_792223


namespace binary_to_decimal_l792_792377

theorem binary_to_decimal : 
  let b := 1 * 2^6 + 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0 
  in b = 108 :=
by
  sorry

end binary_to_decimal_l792_792377


namespace relationship_y1_y2_y3_l792_792007

-- Define the quadratic function with the given parameters
def quadratic (a c x : ℝ) : ℝ := a * (x - 3) ^ 2 + c

-- Given conditions
variable (a c : ℝ)
variable (ha : a < 0)

-- Function values at specific x-values
def y1 := quadratic a c (Real.sqrt 5)
def y2 := quadratic a c 0
def y3 := quadratic a c 4

-- The theorem stating the desired relationship
theorem relationship_y1_y2_y3 : y2 < y3 ∧ y3 < y1 :=
by
  -- Proof goes here, using the given conditions
  sorry

end relationship_y1_y2_y3_l792_792007


namespace infinite_rational_points_no_collinear_l792_792383

-- Definitions and the key theorem statement
theorem infinite_rational_points_no_collinear :
  ∃ (P : ℕ → ℚ × ℚ), 
  (∀ i j : ℕ, i ≠ j → ∃ q : ℚ, dist (P i) (P j) = q) ∧ 
  (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ collinear (P i) (P j) (P k)) := 
sorry

end infinite_rational_points_no_collinear_l792_792383


namespace triangle_ABC_BC_l792_792096

noncomputable def solve_triangle : ℝ :=
  let AC := 1
  let AB := 2
  let A := real.pi / 3  -- 60 degrees in radians
  let BC := real.sqrt (AC^2 + AB^2 - 2 * AC * AB * real.cos A)
  BC

theorem triangle_ABC_BC (AC AB : ℝ) (A : ℝ) (hAC : AC = 1) (hAB : AB = 2) (hA : A = real.pi / 3) : 
  solve_triangle = real.sqrt 3 :=
by
  simp only [solve_triangle, hAC, hAB, hA]
  sorry

end triangle_ABC_BC_l792_792096


namespace school_club_members_l792_792242

theorem school_club_members :
  ∃ n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 3 ∧
  n % 8 = 5 ∧
  n % 9 = 7 ∧
  n = 269 :=
by
  existsi 269
  sorry

end school_club_members_l792_792242


namespace coeff_x2_in_expansion_l792_792610

theorem coeff_x2_in_expansion : 
  let general_term (r : ℕ) : ℤ := (Nat.choose 5 r) * (2 ^ r) 
  in general_term 2 = 40 :=
by
  -- This is where the proof should go
  sorry

end coeff_x2_in_expansion_l792_792610


namespace all_statements_incorrect_l792_792229

-- Defining the statements as conditions
def cond1 (x : ℝ) : Prop := x^(3:ℝ) = x → x = 0 ∨ x = 1
def cond2 (a : ℝ) : Prop := Real.sqrt (a^2) = a
def cond3 : Prop := Real.cbrt (-8) = 2 ∨ Real.cbrt (-8) = -2
def cond4 : Prop := Real.sqrt (Real.sqrt 81) = 9

-- Question is: all conditions are incorrect
theorem all_statements_incorrect : ¬ cond1 0 ∧ ¬ cond1 1 ∧ ¬ (∀ x, cond2 x) ∧ ¬ cond3 ∧ ¬ cond4 := 
by sorry

end all_statements_incorrect_l792_792229


namespace sum_first_6_terms_l792_792429

variable {α : Type*} [LinearOrderedField α]

-- Defining the geometric sequence
def geom_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ (n - 1)

-- Hypotheses derived from the problem
def cond_1 (a q : α) : α := geom_seq a q 3 + geom_seq a q 5 = 20
def cond_2 (a q : α) : α := geom_seq a q 2 * geom_seq a q 6 = 64
def cond_3 (q : α) : Prop := q > 1

-- Sum of the first 6 terms
def geom_seq_sum (a q : α) (n : ℕ) : α := 
  if q = 1 then a * n 
  else a * (1 - q ^ n) / (1 - q)

-- Theorem statement
theorem sum_first_6_terms (a q : α) 
  (h1 : cond_1 a q) (h2 : cond_2 a q) (h3 : cond_3 q) : 
  geom_seq_sum a q 6 = 63 := 
sorry

end sum_first_6_terms_l792_792429


namespace at_least_three_bushes_with_same_number_of_flowers_l792_792113

-- Defining the problem using conditions as definitions.
theorem at_least_three_bushes_with_same_number_of_flowers (n : ℕ) (f : Fin n → ℕ) (h1 : n = 201)
  (h2 : ∀ (i : Fin n), 1 ≤ f i ∧ f i ≤ 100) : 
  ∃ (x : ℕ), (∃ (i1 i2 i3 : Fin n), i1 ≠ i2 ∧ i1 ≠ i3 ∧ i2 ≠ i3 ∧ f i1 = x ∧ f i2 = x ∧ f i3 = x) := 
by
  sorry

end at_least_three_bushes_with_same_number_of_flowers_l792_792113


namespace irwins_family_hike_total_distance_l792_792517

theorem irwins_family_hike_total_distance
    (d1 d2 d3 : ℝ)
    (h1 : d1 = 0.2)
    (h2 : d2 = 0.4)
    (h3 : d3 = 0.1)
    :
    d1 + d2 + d3 = 0.7 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end irwins_family_hike_total_distance_l792_792517


namespace vkontakte_solution_l792_792308

variables (M I A P : Prop)

theorem vkontakte_solution
  (h1 : M → I ∧ A)
  (h2 : A ⊕ P)
  (h3 : I ∨ M)
  (h4 : P ↔ I) : 
  ¬M ∧ I ∧ A ∧ P :=
begin
  sorry
end

end vkontakte_solution_l792_792308


namespace count_integers_satisfying_equation_l792_792059

theorem count_integers_satisfying_equation : 
  (finset.filter (λ x : ℤ, ((x^2 - 2*x - 2) ^ (x + 3) = 1)) (finset.range 100)).card = 4 := 
sorry

end count_integers_satisfying_equation_l792_792059


namespace quadratic_real_roots_prob_l792_792915

theorem quadratic_real_roots_prob :
  let I := set.Icc (0:ℝ) 5
  let P := λ p, p^2 - 4 ≥ 0
  (∫ [I] : Set ℝ, indicator I (λ p, if P p then (1 : Real) else 0) p) = 3/5 :=
by
  sorry

end quadratic_real_roots_prob_l792_792915


namespace certification_cost_l792_792976

theorem certification_cost (adoption_fee : ℕ) (training_weeks : ℕ) (training_cost_per_week : ℕ) 
    (insurance_coverage : ℝ) (total_out_of_pocket : ℕ) :
    adoption_fee = 150 →
    training_weeks = 12 →
    training_cost_per_week = 250 →
    insurance_coverage = 0.90 →
    total_out_of_pocket = 3450 →
    let total_training_cost := training_weeks * training_cost_per_week in
    let total_cost_before_certification := adoption_fee + total_training_cost in
    ∃ (certification_cost : ℕ), 
    total_out_of_pocket = total_cost_before_certification + (1 - insurance_coverage) * certification_cost ∧
    certification_cost = 3000 := 
by
  intros h1 h2 h3 h4 h5
  let total_training_cost := 12 * 250
  let total_cost_before_certification := 150 + total_training_cost
  existsi (3000 : ℕ)
  split
  · simp [total_training_cost, total_cost_before_certification, h1, h2, h3, h4, h5]
    norm_num
  · norm_num
  · sorry

end certification_cost_l792_792976


namespace harmonic_division_reciprocal_l792_792510

variables {A B C D₁ D₂ M N P E : Type*}
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space D₁] [metric_space D₂]
variables (triangle_ABC: triangle A B C)
variables (AB_AC : AB > AC)
variables (D₁_on_BC : is_point_on_side D₁ B C)
variables (circle_AD₁ : is_circle_with_diameter A D₁)
variables (intersect_AB_M : intersects circle_AD₁ AB M)
variables (intersect_AC_N : extends_and_intersects circle_AD₁ AC N)
variables (perpendicular_AP_MN : ⊥_line' AP M N P)
variables (intersect_BC_D₂ : intersects AP BC D₂)
variables (exterior_bisector_AE : is_exterior_angle_bisector E A B C)

theorem harmonic_division_reciprocal :
  (1 / distance B E) + (1 / distance C E) = (1 / distance D₁ E) + (1 / distance D₂ E) :=
by
  sorry

end harmonic_division_reciprocal_l792_792510


namespace math_problem_l792_792161

noncomputable def a : ℝ := Real.pi / 2010

def sum_expression (n : ℕ) : ℝ :=
  2 * ∑ k in Finset.range (n + 1).filter (· > 0), (Real.cos (k^2 * a) * Real.sin (k * a))

lemma smallest_n (n : ℕ) :
  sum_expression n ∈ Int ↔ (∃ m : ℤ, n * (n + 1) / 2010 = m) := sorry

theorem math_problem :
  ∃ n : ℕ, sum_expression n ∈ Int ∧ (∀ m : ℕ, m < n → ¬(sum_expression m ∈ Int)) :=
begin
  use 67,
  split,
  { sorry }, -- This would contain the proof that the sum_expression 67 is an integer.
  { sorry } -- This would contain the proof that there is no smaller integer m for which sum_expression is an integer.
end

end math_problem_l792_792161


namespace mean_transformation_variance_transformation_l792_792840

variables (x1 x2 x3 x4 x5 : ℝ)
variables (data : list ℝ) (transformed_data : list ℝ)
variables (mean variance : ℝ)

def mean_of_data := 8
def variance_of_data := 2

-- Define what is meant by the mean transformation
def transform (x : ℝ) := 4 * x + 1

-- Define the mean of the transformed data
def mean_transformed := mean_of_data * 4 + 1

-- Define the variance of the transformed data
def variance_transformed := variance_of_data * 4^2

theorem mean_transformation :
  (∑ x in [x1, x2, x3, x4, x5], x / 5 = mean_of_data) → (∑ x in [transform x1, transform x2, transform x3, transform x4, transform x5], x / 5 = mean_transformed) :=
sorry

theorem variance_transformation :
  -- For simplicity, assume the variance formula for a finite set is given as a theorem
  (variance_of_data = (∑ x in [x1, x2, x3, x4, x5], (x - mean_of_data)^2) / 5) →
  (variance_transformed = (∑ x in [transform x1, transform x2, transform x3, transform x4, transform x5], (x - mean_transformed)^2) / 5 * 4^2) :=
sorry

end mean_transformation_variance_transformation_l792_792840


namespace Megan_not_lead_plays_l792_792572

def total_plays : ℕ := 100
def lead_percentage : ℝ := 0.80
def lead_plays : ℕ := (total_plays : ℝ * lead_percentage).toNat
def not_lead_plays : ℕ := total_plays - lead_plays

theorem Megan_not_lead_plays : not_lead_plays = 20 := by
  sorry

end Megan_not_lead_plays_l792_792572


namespace evaluate_expression_l792_792386

-- Definition of the ceil function used in the problem
def ceil (x : ℝ) : ℤ := ⌈x⌉₊

theorem evaluate_expression : ceil (4 * (8 - 1 / 3)) = 31 := by 
  sorry

end evaluate_expression_l792_792386


namespace three_digit_numbers_at_least_one_4_one_5_l792_792485

/--
There are 48 three-digit numbers that have at least one digit '4' and at least one digit '5'.
-/
theorem three_digit_numbers_at_least_one_4_one_5 : 
  ∃ (n : ℕ), n = 48 ∧
  ∀ (d : ℕ), 100 ≤ d ∧ d ≤ 999 → 
  let digits := d.digits 10 in
  ((4 ∈ digits) ∧ (5 ∈ digits)) →
  n = 48 :=
by
  sorry

end three_digit_numbers_at_least_one_4_one_5_l792_792485


namespace circumcenter_exists_l792_792050

variables {P : Type*} [euclidean_space P] (A B C : P)

-- Condition: Points A, B, C are non-collinear
def non_collinear (A B C : P) : Prop := ¬ collinear ({A, B, C} : set P)

-- Question: Construct circumcenter of A, B, C
theorem circumcenter_exists :
  non_collinear A B C →
  ∃ O : P, dist O A = dist O B ∧ dist O B = dist O C ∧ is_circumcenter_of_triangle A B C O :=
begin
  intro h_non_collinear,
  sorry -- Construction and proof steps would go here.
end

end circumcenter_exists_l792_792050


namespace vector_dot_product_is_zero_l792_792230

theorem vector_dot_product_is_zero {A B C D : EuclideanSpace ℝ 3}
  (hAB : ∥A - B∥ = 3)
  (hBC : ∥B - C∥ = 7)
  (hCD : ∥C - D∥ = 11)
  (hDA : ∥D - A∥ = 9) :
  (A - C) • (B - D) = 0 :=
sorry

end vector_dot_product_is_zero_l792_792230


namespace tan_11_pi_over_4_l792_792789

theorem tan_11_pi_over_4 : Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Proof is omitted
  sorry

end tan_11_pi_over_4_l792_792789


namespace usual_time_to_school_l792_792682

-- Define the conditions
variables (R T : ℝ) (h1 : 0 < T) (h2 : 0 < R)
noncomputable def boy_reaches_school_early : Prop :=
  (7/6 * R) * (T - 5) = R * T

-- The theorem stating the usual time to reach the school
theorem usual_time_to_school (h : boy_reaches_school_early R T) : T = 35 :=
by
  sorry

end usual_time_to_school_l792_792682


namespace hoseok_days_l792_792888

theorem hoseok_days (pages_solved_total : ℕ) (pages_solved_per_day : ℕ) (h1 : pages_solved_total = 48) (h2 : pages_solved_per_day = 4) :
  pages_solved_total / pages_solved_per_day = 12 :=
begin
  sorry
end

end hoseok_days_l792_792888


namespace total_time_to_complete_work_l792_792303

-- Definitions of conditions
def work_rate_A (W : ℝ) : ℝ := W / 15
def work_rate_B (W : ℝ) : ℝ := W / 10
def combined_work_rate (W : ℝ) : ℝ := (work_rate_A W) + (work_rate_B W)
def work_done_together_in_2_days (W : ℝ) : ℝ := 2 * (combined_work_rate W)
def remaining_work_A_after_2_days (W : ℝ) : ℝ := W - (work_done_together_in_2_days W)
def time_A_to_complete_remaining_work (W : ℝ) : ℝ := (remaining_work_A_after_2_days W) / (work_rate_A W)

-- Theorem to prove
theorem total_time_to_complete_work (W : ℝ) : (W > 0) → (combined_work_rate W) > 0 →
  2 + time_A_to_complete_remaining_work W = 12 :=
by
  intros hW hCWR
  sorry

end total_time_to_complete_work_l792_792303


namespace Megan_not_lead_actress_l792_792571

-- Define the conditions: total number of plays and lead actress percentage
def totalPlays : ℕ := 100
def leadActressPercentage : ℕ := 80

-- Define what we need to prove: the number of times Megan was not the lead actress
theorem Megan_not_lead_actress (totalPlays: ℕ) (leadActressPercentage: ℕ) : 
  (totalPlays * (100 - leadActressPercentage)) / 100 = 20 :=
by
  -- proof omitted
  sorry

end Megan_not_lead_actress_l792_792571


namespace sarahs_loan_amount_l792_792593

theorem sarahs_loan_amount 
  (down_payment : ℕ := 10000)
  (monthly_payment : ℕ := 600)
  (repayment_years : ℕ := 5)
  (interest_rate : ℚ := 0) : down_payment + (monthly_payment * (12 * repayment_years)) = 46000 :=
by
  sorry

end sarahs_loan_amount_l792_792593


namespace part_I_part_II_l792_792431

-- Defining the sequence {a_n}
def seq_a : ℕ → ℕ
| 0     := 0    -- base case for consistency, typically we index from 1
| 1     := 3    -- given a₁ = 3
| (n+2) := 3 * (seq_a (n + 1)) - 4  -- given aₙ₊₁ = 3aₙ - 4

-- Defining the sequence {b_n}
def seq_b (n : ℕ) : ℕ := seq_a (n + 1) - 2  -- given bₙ = aₙ - 2

-- Part (Ⅰ): Prove that {b_n} is a geometric sequence
theorem part_I : ∃ q b₁, ∀ n, seq_b (n+1) = q * seq_b n := by
  -- Provide the proof here
  sorry

-- Part (Ⅱ): Prove general formula for {a_n}
theorem part_II : ∀ n, seq_a (n+1) = 3 ^ n + 2 := by
  -- Provide the proof here
  sorry

end part_I_part_II_l792_792431


namespace number_of_common_divisors_l792_792906

theorem number_of_common_divisors :
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  let divisors_count := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  gcd_ab = 420 ∧ divisors_count = 24 :=
by
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 420 := sorry
  have h2 : (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 24 := by norm_num
  exact ⟨h1, h2⟩

end number_of_common_divisors_l792_792906


namespace max_value_of_expression_l792_792445

noncomputable def max_expression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  (finset.univ.sum (λ i => (1 - x i).sqrt)) / ((finset.univ.sum (λ i => (x i)⁻¹)).sqrt)

theorem max_value_of_expression {n : ℕ} (h : n > 0) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i ∧ x i < 1) : 
  max_expression n x ≤ (n.sqrt / 2) :=
sorry

end max_value_of_expression_l792_792445


namespace parabola_no_intersection_inequality_l792_792177

-- Definitions for the problem
theorem parabola_no_intersection_inequality
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, (a * x^2 + b * x + c ≠ x) ∧ (a * x^2 + b * x + c ≠ -x)) :
  |b^2 - 4 * a * c| > 1 := 
sorry

end parabola_no_intersection_inequality_l792_792177


namespace parallel_lines_iff_l792_792164

theorem parallel_lines_iff (a : ℝ) : (a = 1) ↔ 
  let l1 := fun x y : ℝ => a * x + 2 * y - 1 = 0 in
  let l2 := fun x y : ℝ => 1 * x + 2 * y + 4 = 0 in
  ∃ k : ℝ, ∀ x y : ℝ, (a * x + 2 * y - 1 = 0) = (k * (x + 2 * y + 4 = 0)) :=
by {
  sorry
}

end parallel_lines_iff_l792_792164


namespace tan_diff_eq_rat_l792_792959

theorem tan_diff_eq_rat (A : ℝ × ℝ) (B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (5, 1))
  (α β : ℝ)
  (hα : Real.tan α = 2) (hβ : Real.tan β = 1 / 5) :
  Real.tan (α - β) = 9 / 7 := by
  sorry

end tan_diff_eq_rat_l792_792959


namespace parallelogram_contains_two_points_l792_792550

def L : Set (ℤ × ℤ) := {p | ∃ x y : ℤ, p = (41 * x + 2 * y, 59 * x + 15 * y)}

theorem parallelogram_contains_two_points (parallelogram : Set (ℝ × ℝ)) 
  (centered_at_origin : ∀ p ∈ parallelogram, (-p.1, -p.2) ∈ parallelogram)
  (area_parallelogram : ∀ (a b c d : ℝ × ℝ), 
    parallelogram = {p | ∃ (λ ∈ [0, 1] × [0, 1]), 
                              p = λ.1 • a + λ.2 • b ∨ 
                              p = λ.1 • a + (1 - λ.2) • d ∨ 
                              p = λ.2 • b + (1 - λ.1) • c ∨ 
                              p = λ.2 • d + (1 - λ.1) • c })
  (area_is_1990 : parallelogram_area parallelogram = 1990) :
  ∃ p1 p2 ∈ parallelogram, p1 ∈ L ∧ p2 ∈ L ∧ p1 ≠ p2 :=
by
  sorry

end parallelogram_contains_two_points_l792_792550


namespace no_closed_polygonal_chain_l792_792708

theorem no_closed_polygonal_chain (points : set (ℝ × ℝ)) (h_distinct : ∀ p1 p2 : ℝ × ℝ, p1 ≠ p2 → dist p1 p2 ≠ dist p1 p2) :
  ¬ ∃ (f : (ℝ × ℝ) → (ℝ × ℝ)), (∃ (h : ∀ p1 p2 : ℝ × ℝ, f p1 = p2 → closest_point p1 = p2), closed_polygonal_chain points f) :=
sorry

def closest_point (p : ℝ × ℝ) : ℝ × ℝ := sorry
def closed_polygonal_chain (points : set (ℝ × ℝ)) (f : (ℝ × ℝ) → (ℝ × ℝ)) : Prop := sorry

end no_closed_polygonal_chain_l792_792708


namespace athletes_in_camp_hours_l792_792704

theorem athletes_in_camp_hours (initial_athletes : ℕ) (left_rate : ℕ) (left_hours : ℕ) (arrived_rate : ℕ) 
  (difference : ℕ) (hours : ℕ) 
  (h_initial: initial_athletes = 300) 
  (h_left_rate: left_rate = 28) 
  (h_left_hours: left_hours = 4) 
  (h_arrived_rate: arrived_rate = 15) 
  (h_difference: difference = 7) 
  (h_left: left_rate * left_hours = 112) 
  (h_equation: initial_athletes - (left_rate * left_hours) + (arrived_rate * hours) = initial_athletes - difference) : 
  hours = 7 :=
by
  sorry

end athletes_in_camp_hours_l792_792704


namespace digit_d_multiple_of_9_l792_792416

theorem digit_d_multiple_of_9 (d : ℕ) (hd : d = 1) : ∃ k : ℕ, (56780 + d) = 9 * k := by
  have : 56780 + d = 56780 + 1 := by rw [hd]
  rw [this]
  use 6313
  sorry

end digit_d_multiple_of_9_l792_792416


namespace smallest_four_digit_number_l792_792400

theorem smallest_four_digit_number :
  ∃ x : ℕ, 
    (1000 ≤ x ∧ x ≤ 9999) ∧
    (5 * x ≡ 15 [MOD 20]) ∧ 
    (3 * x + 7 ≡ 10 [MOD 8]) ∧ 
    (-3 * x + 2 ≡ 2 * x [MOD 35]) ∧ 
    ∀ y : ℕ, (1000 ≤ y ∧ y ≤ 9999) → (5 * y ≡ 15 [MOD 20]) → (3 * y + 7 ≡ 10 [MOD 8]) → (-3 * y + 2 ≡ 2 * y [MOD 35]) → x ≤ y :=
  let x := 1009 in
  ⟨x, ⟨by norm_num, 
       by norm_num, ⟩, 
       by norm_num, ⟩, sorry, sorry, sorry⟩

end smallest_four_digit_number_l792_792400


namespace raine_change_l792_792326

theorem raine_change 
  (price_bracelet : ℕ := 15)
  (price_necklace : ℕ := 10)
  (price_coffee_mug : ℕ := 20)
  (quantity_bracelet : ℕ := 3)
  (quantity_necklace : ℕ := 2)
  (quantity_coffee_mug : ℕ := 1)
  (total_money : ℕ := 100) :
  let total_cost := (quantity_bracelet * price_bracelet) + (quantity_necklace * price_necklace) + (quantity_coffee_mug * price_coffee_mug)
  in total_money - total_cost = 15 :=
by
  sorry

end raine_change_l792_792326


namespace find_A_l792_792166

-- Define the given conditions
variables (A B : ℝ)
def f (x : ℝ) := A * x^2 - 2 * B * x - 3 * B^2
def g (x : ℝ) := B * x^2 + x

-- Define the condition that B ≠ 0
hypothesis (hB : B ≠ 0)

-- Define the main theorem statement
theorem find_A : f A B (g B 1) = 0 → A = (5 * B^2 + 2 * B) / (B^2 + 2 * B + 1) := 
by
  sorry

end find_A_l792_792166


namespace f_at_3_l792_792918

def f (x : ℝ) : ℝ := (x^2 + x + 1) / (3 * x^2 - 4)

theorem f_at_3 : f 3 = 13 / 23 :=
by
  sorry

end f_at_3_l792_792918


namespace solve_sin_cos_eqn_l792_792395

theorem solve_sin_cos_eqn (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x + Real.cos x = 1) :
  x = 0 ∨ x = Real.pi / 2 :=
sorry

end solve_sin_cos_eqn_l792_792395


namespace response_rate_is_60_percent_l792_792698

-- Definitions based on conditions
def responses_needed : ℕ := 900
def questionnaires_mailed : ℕ := 1500

-- Derived definition
def response_rate_percentage : ℚ := (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

-- The theorem stating the problem
theorem response_rate_is_60_percent :
  response_rate_percentage = 60 := 
sorry

end response_rate_is_60_percent_l792_792698


namespace find_slope_midpoint_l792_792616

-- Define the curve x^2 * sqrt(2) + y^2 = 1
def curve (x y : ℝ) : Prop := x^2 * sqrt(2) + y^2 = 1

-- Define the line x + y - 1 = 0
def line (x y : ℝ) : Prop := x + y = 1

-- Define the midpoint formula
def midpoint (P Q M : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

-- Define the slope of OM
def slope (O M : ℝ × ℝ) : ℝ := (M.2 - O.2) / (M.1 - O.1)

-- Define intersection points P and Q
def intersects (f g : ℝ → ℝ → Prop) (P Q : ℝ × ℝ) : Prop :=
  f P.1 P.2 ∧ g P.1 P.2 ∧ f Q.1 Q.2 ∧ g Q.1 Q.2 ∧ P ≠ Q

-- The main theorem
theorem find_slope_midpoint (P Q M : ℝ × ℝ) (h_intersect : intersects curve line P Q) (h_midpoint : midpoint P Q M) :
  slope (0, 0) M = sqrt(2) := by
  sorry

end find_slope_midpoint_l792_792616


namespace vkontakte_solution_l792_792306

variables (M I A P : Prop)

theorem vkontakte_solution
  (h1 : M → I ∧ A)
  (h2 : A ⊕ P)
  (h3 : I ∨ M)
  (h4 : P ↔ I) : 
  ¬M ∧ I ∧ A ∧ P :=
begin
  sorry
end

end vkontakte_solution_l792_792306


namespace max_Bk_l792_792774

theorem max_Bk (k : ℕ) (h0 : 0 ≤ k) (h1 : k ≤ 2000) : k = 181 ↔ 
  ∀ k' : ℕ, (0 ≤ k' ∧ k' ≤ 2000) → B k ≤ B k' :=
sorry

def B (k : ℕ) : ℝ :=
  if (0 ≤ k ∧ k ≤ 2000) then ((nat.choose 2000 k) : ℝ) * (0.1 ^ k) else 0

end max_Bk_l792_792774


namespace HNO3_percentage_l792_792700

theorem HNO3_percentage (initial_volume : ℝ) (initial_conc : ℝ) (added_volume : ℝ) 
  (initial_volume = 60) 
  (initial_conc = 0.4) 
  (added_volume = 12) : 
  ((initial_conc * initial_volume + added_volume) / (initial_volume + added_volume)) * 100 = 50 := 
by sorry

end HNO3_percentage_l792_792700


namespace maximum_B_k_at_181_l792_792775

open Nat

theorem maximum_B_k_at_181 :
  let B : ℕ → ℝ := λ k, (Nat.choose 2000 k : ℝ) * (0.1 ^ k)
  ∃ k : ℕ, k ≤ 2000 ∧ (∀ m : ℕ, m ≤ 2000 → B m ≤ B 181) :=
by
  let B := λ k : ℕ, (Nat.choose 2000 k : ℝ) * (0.1 ^ k)
  use 181
  split
  · linarith
  · intro m hm
    sorry

end maximum_B_k_at_181_l792_792775


namespace shopper_total_payment_l792_792716

theorem shopper_total_payment :
  let original_price := 150
  let discount_rate := 0.25
  let coupon_discount := 10
  let sales_tax_rate := 0.10
  let discounted_price := original_price * (1 - discount_rate)
  let price_after_coupon := discounted_price - coupon_discount
  let final_price := price_after_coupon * (1 + sales_tax_rate)
  final_price = 112.75 := by
{
  sorry
}

end shopper_total_payment_l792_792716


namespace tetrahedron_volume_l792_792126

/-- In tetrahedron PQRS:
  - Edge PQ has length 4 cm.
  - The area of face PQR is 18 cm².
  - The area of face PQS is 16 cm².
  - The angle between faces PQR and PQS is 45°.

  Prove the volume of the tetrahedron is 24√2 cm³.
--/
theorem tetrahedron_volume (PQ PQR PQS : ℝ) (angle45 : ℝ) 
  (PQ_len : PQ = 4) 
  (PQR_area : PQR = 18) 
  (PQS_area : PQS = 16) 
  (angle45_val : angle45 = 45) : 
  ∃ V : ℝ, V = 24 * real.sqrt 2 :=
by
  sorry

end tetrahedron_volume_l792_792126


namespace max_value_A_upbound_l792_792443

noncomputable def max_value_A (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i ∧ x i < 1) : ℝ :=
  (Finset.univ.sum (λ i, real.sqrt (1 - x i))) / 
  real.sqrt (Finset.univ.sum (λ i, 1 / x i))

theorem max_value_A_upbound (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i ∧ x i < 1) :
  max_value_A n x hx ≤ real.sqrt (n) / 2 :=
sorry

end max_value_A_upbound_l792_792443


namespace similar_triangles_MNP_XYZ_l792_792652

noncomputable def similar_triangles (M N P X Y Z : Type*) :=
  -- Define the similarity relation between triangles MNP and XYZ
  (triangle M N P ∼ triangle X Y Z)

theorem similar_triangles_MNP_XYZ (M N P X Y Z : Type*)
  (h_similarity : similar_triangles M N P X Y Z)
  (h_MN : MN = 8)
  (h_NP : NP = 10)
  (h_XY : XY = 20) :
  ∃ XZ perimeterXYZ,
  XZ = 25 ∧
  perimeterXYZ = 65 :=
begin
  -- similiar triangle similarity relation
  sorry -- proof is not required as per the instruction
end

end similar_triangles_MNP_XYZ_l792_792652


namespace trigonometric_sign_l792_792675

open Real

theorem trigonometric_sign :
  (0 < 1 ∧ 1 < π / 2) ∧ 
  (∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → sin x ≤ sin y)) ∧ 
  (∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → cos x ≥ cos y)) →
  (cos (cos 1) - cos 1) * (sin (sin 1) - sin 1) < 0 :=
by
  sorry

end trigonometric_sign_l792_792675


namespace ratio_of_areas_l792_792654

-- Define the radii of the concentric circles
def radiusOP (R : ℝ) : ℝ := R
def radiusOX (R : ℝ) : ℝ := (1 / 3) * R

-- Define the areas of the circles
def areaCircle (radius : ℝ) : ℝ := real.pi * radius^2

-- State the theorem
theorem ratio_of_areas {R : ℝ} (hR : R > 0) :
  (areaCircle (radiusOX R)) / (areaCircle (radiusOP R)) = 1 / 9 :=
by
  sorry

end ratio_of_areas_l792_792654


namespace necessary_but_not_sufficient_condition_l792_792614

-- Define the conditions as hypotheses
variables {k : ℝ}

-- Statement to prove in Lean
theorem necessary_but_not_sufficient_condition (h : -2 < k ∧ k < 3) : 
  ∀ x : ℝ, x^2 + k * x + 1 > 0 → h :=
sorry

end necessary_but_not_sufficient_condition_l792_792614


namespace hyperbola_properties_proof_parabola_properties_proof_l792_792866

-- Define the hyperbola equation and associated parameters
def hyperbola_equation (x y : ℝ) : Prop :=
  16 * x^2 - 9 * y^2 = 144

def transverse_axis_length : ℝ := 6
def conjugate_axis_length : ℝ := 8
def hyperbola_eccentricity : ℝ := 5/3

-- Define the conditions for the parabola
def vertex : ℝ × ℝ := (0, 0)
def focus : ℝ × ℝ := (-3, 0)
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = -12 * x

theorem hyperbola_properties_proof :
  (∀ x y : ℝ, hyperbola_equation x y) →
  transverse_axis_length = 6 ∧
  conjugate_axis_length = 8 ∧
  hyperbola_eccentricity = 5 / 3 :=
by
  sorry

theorem parabola_properties_proof :
  vertex = (0, 0) →
  focus = (-3, 0) →
  ∀ x y : ℝ, parabola_equation x y :=
by
  sorry

end hyperbola_properties_proof_parabola_properties_proof_l792_792866


namespace seq_gt_l792_792172

noncomputable def seq (n : ℕ) : ℝ :=
match n with
| 0     => 1
| (n+1) => (sqrt (1 + (seq n)^2) - 1) / (seq n)

theorem seq_gt (n : ℕ) : seq n > (Real.pi / (2 ^ (n+2))) := by
  sorry

end seq_gt_l792_792172


namespace probability_different_tens_digit_l792_792601

theorem probability_different_tens_digit :
  let n := 6
  let range := Set.Icc 10 59
  let tens_digit (x : ℕ) := x / 10
  let valid_set (s : Set ℕ) := ∀ x ∈ s, 10 ≤ x ∧ x ≤ 59
  let different_tens_digits (s : Set ℕ) := (∀ (x y : ℕ), x ∈ s → y ∈ s → x ≠ y → tens_digit x ≠ tens_digit y)
  let total_ways := Nat.choose 50 6
  let favorable_ways := 5 * 10 * 9 * 10^4
  let probability := favorable_ways * 1 / total_ways
  valid_set ({ x | x ∈ range } : Set ℕ) →
  different_tens_digits ({ x | x ∈ range } : Set ℕ) →
  probability = (1500000 : ℚ) / 5296900 :=
by
  sorry

end probability_different_tens_digit_l792_792601


namespace count_valid_pairs_l792_792806

def contains_zero_digit (n : ℕ) : Prop := 
  ∃ k, 10^k ≤ n ∧ n < 10^(k+1) ∧ (n / 10^k % 10 = 0)

def valid_pair (a b : ℕ) : Prop := 
  a + b = 500 ∧ ¬contains_zero_digit a ∧ ¬contains_zero_digit b

theorem count_valid_pairs : 
  (Finset.card (Finset.filter (λ p : ℕ × ℕ, valid_pair p.1 p.2) 
    (Finset.product (Finset.range 500) (Finset.range 500)))) = 329 := 
sorry

end count_valid_pairs_l792_792806


namespace complex_modulus_calc_l792_792391

theorem complex_modulus_calc :
  (complex.abs ((1 / 2 : ℝ) + (complex.I * (real.sqrt 3 / 2 : ℝ))) ^ 12) = 1 :=
begin
  sorry
end

end complex_modulus_calc_l792_792391


namespace median_of_mode_is_4_l792_792839

noncomputable def data_set (x : ℕ) := [3, 8, 5, x, 4]

def is_mode (s : List ℕ) (m : ℕ) : Prop :=
  ∃ (freq_m : ℕ), ∀ n ≠ m, (s.count m > s.count n)

theorem median_of_mode_is_4 (x : ℕ) (h_mode : is_mode (data_set x) 4) : 
  (List.median [3, x, 4, 5, 8]) = 4 :=
sorry

end median_of_mode_is_4_l792_792839


namespace factor_of_P_l792_792992

variable {R : Type*} [CommRing R]

variable (P Q R S : R[X])

theorem factor_of_P (h : P.eval₂ (Polynomial.C) (X^3) + Polynomial.C X * Q.eval₂ (Polynomial.C) (X^3) + Polynomial.C (X^2) * R.eval₂ (Polynomial.C) (X^5) =
  (X^4 + X^3 + X^2 + X + 1) * S) :
  Polynomial.cyclotomic 1 R ∣ P :=
sorry

end factor_of_P_l792_792992


namespace sum_possible_m_continuous_l792_792558

noncomputable def g (x m : ℝ) : ℝ :=
if x < m then x^2 + 4 * x + 3 else 3 * x + 9

theorem sum_possible_m_continuous :
  let m₁ := -3
  let m₂ := 2
  m₁ + m₂ = -1 :=
by
  sorry

end sum_possible_m_continuous_l792_792558


namespace problem_1_problem_2_problem_3_l792_792045

-- Problem 1

theorem problem_1 (k : ℝ) (h : k ≠ 0) : 
  (∀ x : ℝ, k*x^2 - 2*x + 6*k < 0 ↔ x < -3 ∨ x > -2) → k = -2/5 := 
sorry

-- Problem 2

theorem problem_2 (k : ℝ) (h : k ≠ 0) : 
  (∀ x : ℝ, k*x^2 - 2*x + 6*k < 0 ↔ x ≠ 1/k) → k = - Real.sqrt 6 / 6 := 
sorry

-- Problem 3

theorem problem_3 (k : ℝ) : 
  (∀ j : ℝ, k*j^2 - 2*j + 6*k < 0 ↔ False) → k ≥ Real.sqrt 6 / 6 := 
sorry

end problem_1_problem_2_problem_3_l792_792045


namespace minimize_expression_l792_792685

variables {x1 x2 x3 y1 y2 y3 : ℝ}
variables (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3)
variables (h4 : 0 ≤ y1) (h5 : 0 ≤ y2) (h6 : 0 ≤ y3)

theorem minimize_expression :
  (∃ (x1 x2 x3 y1 y2 y3 : ℝ), (0 ≤ x1 ∧ 0 ≤ x2 ∧ 0 ≤ x3 ∧ 0 ≤ y1 ∧ 0 ≤ y2 ∧ 0 ≤ y3) → 
  √((2018 - y1 - y2 - y3)^2 + x3^2) + √(y3^2 + x2^2) + √(y2^2 + x1^2) + 
  √(y1^2 + (x1 + x2 + x3)^2) = 2018) :=
begin
  sorry
end

end minimize_expression_l792_792685


namespace average_waiting_time_l792_792461

theorem average_waiting_time 
  (A B C : ℕ) 
  (intervalA : A = 10) 
  (intervalB : B = 12) 
  (intervalC : C = 15):
  let LCM := Nat.lcm A (Nat.lcm B C),
      average_time := ((20 * 5) + (4 * 1) + (6 * 1.5) + (10 * 2.5) + (8 * 2) + (12 * 3)) / 60 in
  LCM = 60 ∧ average_time = 19 / 6 := 
by
  sorry

end average_waiting_time_l792_792461


namespace boudin_hormel_ratio_l792_792405

noncomputable def ratio_boudin_hormel : Prop :=
  let foster_chickens := 45
  let american_bottles := 2 * foster_chickens
  let hormel_chickens := 3 * foster_chickens
  let del_monte_bottles := american_bottles - 30
  let total_items := 375
  ∃ (boudin_chickens : ℕ), 
    foster_chickens + american_bottles + hormel_chickens + boudin_chickens + del_monte_bottles = total_items ∧
    boudin_chickens / hormel_chickens = 1 / 3

theorem boudin_hormel_ratio : ratio_boudin_hormel :=
sorry

end boudin_hormel_ratio_l792_792405


namespace smallest_angle_l792_792984

open Real EuclideanSpace

variables (a b c d : ℝ^3) -- defining variables as 3-dimensional real vectors.

-- define the norms of the vectors
noncomputable def norm_a : ℝ := ∥a∥ = 1
noncomputable def norm_b : ℝ := ∥b∥ = 2
noncomputable def norm_c : ℝ := ∥c∥ = 3
noncomputable def norm_d : ℝ := ∥d∥ = 1

-- define the vector equation
def vector_equation : Prop := 
  a × (b × c) + d = 0

-- define the angle calculation
noncomputable def angle_between_vecs : ℝ :=
  real.arccos ((2 * (a • b) - 3 * (a • c)) / 6)

-- statement to prove
theorem smallest_angle (h1: norm_a) (h2: norm_b) (h3: norm_c) (h4: norm_d) (h5: vector_equation) :
  angle_between_vecs a b c d = 48.19 :=
sorry

end smallest_angle_l792_792984


namespace circle_eq_l792_792028

theorem circle_eq (A B : ℝ × ℝ) (hA1 : A = (5, 2)) (hA2 : B = (-1, 4)) (hx : ∃ (c : ℝ), (c, 0) = (c, 0)) :
  ∃ (C : ℝ) (D : ℝ) (x y : ℝ), (x + C) ^ 2 + y ^ 2 = D ∧ D = 20 ∧ (x - 1) ^ 2 + y ^ 2 = 20 :=
by
  sorry

end circle_eq_l792_792028


namespace cyclic_points_exist_l792_792553

noncomputable def f (x : ℝ) : ℝ := 
if x < (1 / 3) then 
  2 * x + (1 / 3) 
else 
  (3 / 2) * (1 - x)

theorem cyclic_points_exist :
  ∃ (x0 x1 x2 x3 x4 : ℝ), 
  0 ≤ x0 ∧ x0 ≤ 1 ∧
  0 ≤ x1 ∧ x1 ≤ 1 ∧
  0 ≤ x2 ∧ x2 ≤ 1 ∧
  0 ≤ x3 ∧ x3 ≤ 1 ∧
  0 ≤ x4 ∧ x4 ≤ 1 ∧
  x0 ≠ x1 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ x4 ≠ x0 ∧
  f x0 = x1 ∧ f x1 = x2 ∧ f x2 = x3 ∧ f x3 = x4 ∧ f x4 = x0 :=
sorry

end cyclic_points_exist_l792_792553


namespace JulieCompletesInOneHour_l792_792683

-- Define conditions
def JuliePeelsIn : ℕ := 10
def TedPeelsIn : ℕ := 8
def TimeTogether : ℕ := 4

-- Define their respective rates
def JulieRate : ℚ := 1 / JuliePeelsIn
def TedRate : ℚ := 1 / TedPeelsIn

-- Define the task completion in 4 hours together
def TaskCompletedTogether : ℚ := (JulieRate * TimeTogether) + (TedRate * TimeTogether)

-- Define remaining task after working together
def RemainingTask : ℚ := 1 - TaskCompletedTogether

-- Define time for Julie to complete the remaining task
def TimeForJulieToComplete : ℚ := RemainingTask / JulieRate

-- The theorem statement
theorem JulieCompletesInOneHour :
  TimeForJulieToComplete = 1 := by
  sorry

end JulieCompletesInOneHour_l792_792683


namespace cos_theta_interval_l792_792070

theorem cos_theta_interval (θ : ℝ) (hθ : 0 < θ ∧ θ < π) 
  (h : ∀ x : ℝ, (cos θ) * x^2 - 4 * (sin θ) * x + 6 > 0) : 
  1 / 2 < cos θ ∧ cos θ < 1 := 
by
  sorry

end cos_theta_interval_l792_792070


namespace plum_cost_l792_792582

theorem plum_cost
  (total_fruits : ℕ)
  (total_cost : ℕ)
  (peach_cost : ℕ)
  (plums_bought : ℕ)
  (peaches_bought : ℕ)
  (P : ℕ) :
  total_fruits = 32 →
  total_cost = 52 →
  peach_cost = 1 →
  plums_bought = 20 →
  peaches_bought = total_fruits - plums_bought →
  total_cost = 20 * P + peaches_bought * peach_cost →
  P = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end plum_cost_l792_792582


namespace nearest_integer_to_inverse_cube_root_l792_792541

noncomputable def f (x : ℝ) : ℝ := x^4 - 2009 * x + 1

theorem nearest_integer_to_inverse_cube_root (x : ℝ) (hx : f x = 0) (largest : ∀ y, f y = 0 → y ≤ x) :
  Int.round (1 / (x^3 - 2009)) = -13 := 
sorry

end nearest_integer_to_inverse_cube_root_l792_792541


namespace permutation_cycle_conditions_l792_792982

def cycle_length (σ : perm (fin n)) (a : fin n) : ℕ :=
if ∃ d, (d > 0) ∧ iterate (perm.eval σ) d a = a then
  nat.find (exists_int_of_exists_pos (exists_of_exists_gt (λ d hd, ⟨d, ⟨hd.left, perm.iterate_eq_of_pos_of_eq a hd.right⟩⟩)))
else -1

noncomputable def cycle_sum (σ : perm (fin n)) : ℕ := finset.univ.sum (λ i, cycle_length σ i)

noncomputable def harmonic_sum (σ : perm (fin n)) : ℚ := finset.univ.sum (λ i, 1 / (cycle_length σ i) )

theorem permutation_cycle_conditions :
  ∃ (n : ℕ) (σ : perm (fin n)) (hn : n = 53),
    cycle_sum σ = 2017 ∧ harmonic_sum σ = 2 :=
sorry

end permutation_cycle_conditions_l792_792982


namespace rational_equation_solutions_l792_792381

open Real

theorem rational_equation_solutions :
  (∃ x : ℝ, (x ≠ 1 ∧ x ≠ -1) ∧ ((x^2 - 6*x + 9) / (x - 1) - (3 - x) / (x^2 - 1) = 0)) →
  ∃ S : Finset ℝ, S.card = 2 ∧ ∀ x ∈ S, (x ≠ 1 ∧ x ≠ -1) :=
by
  sorry

end rational_equation_solutions_l792_792381


namespace distance_between_intersections_max_distance_to_line_l792_792879

theorem distance_between_intersections (t θ : ℝ) :
  let l_x := 1 + 1/2 * t,
      l_y := (√3 / 6) * t,
      C1_x := cos θ,
      C1_y := sin θ in
  (l_x = C1_x ∧ l_y = C1_y → l_x = 1 ∧ l_y = 0) ∨
  (l_x = C1_x ∧ l_y = C1_y → l_x = -1/2 ∧ l_y = -(√3)/2) →
  sqrt((1 - (-1/2))^2 + (0 - (-(√3)/2))^2) = √3 :=
sorry

theorem max_distance_to_line (θ : ℝ) :
  let l_x := 1 + 1/2 * t,
      l_y := (√3 / 6) * t,
      C2_x := 1/2 * cos θ,
      C2_y := √3/2 * sin θ,
      α := -π/4,
      cos_α := √2/2,
      sin_α := -√2/2 in
  max (-(cos_α * C2_x + sin_α * C2_y - 1)^2/2) = (√2/4 + 1/2) :=
sorry

end distance_between_intersections_max_distance_to_line_l792_792879


namespace slope_of_line_polar_l792_792859

noncomputable def polar_to_slope (ρ θ : ℝ) : ℝ :=
  ρ * sin θ - 2 * ρ * cos θ + 3

theorem slope_of_line_polar (ρ θ : ℝ) (h : polar_to_slope ρ θ = 0) : ∃ m : ℝ, m = 2 :=
by
  use 2
  sorry

end slope_of_line_polar_l792_792859


namespace not_an_axiom_l792_792315

def PropositionA : Prop :=
  ∀ {P Q R : Plane}, (P ≠ Q ∧ P || R ∧ Q || R) → P || Q

def PropositionB : Prop :=
  ∀ {A B C : Point}, (¬ collinear A B C) → ∃! P : Plane, {A, B, C} ⊆ P

def PropositionC : Prop :=
  ∀ {L : Line} {A B : Point} {P : Plane}, (A ∈ L ∧ B ∈ L ∧ (A ∈ P ∧ B ∈ P)) → L ⊆ P

def PropositionD : Prop :=
  ∀ {P Q : Plane} {A : Point}, (P ≠ Q ∧ A ∈ P ∧ A ∈ Q) → ∃! L : Line, A ∈ L ∧ L ⊆ P ∧ L ⊆ Q

theorem not_an_axiom (B_axiom : PropositionB) (C_axiom : PropositionC) (D_axiom : PropositionD) : ¬ PropositionA :=
  sorry

end not_an_axiom_l792_792315


namespace num_valid_C_l792_792411

open Nat

def is_digit (C : ℕ) : Prop := C ≥ 0 ∧ C ≤ 9

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

theorem num_valid_C : 
  {C : ℕ // is_digit C ∧ is_multiple_of_3 (6 + C)}.card = 4 := 
by 
  sorry

end num_valid_C_l792_792411


namespace distance_to_directrix_l792_792844

theorem distance_to_directrix (x y d : ℝ) (a b c : ℝ) (F1 F2 M : ℝ × ℝ)
  (h_ellipse : x^2 / 25 + y^2 / 9 = 1)
  (h_a : a = 5)
  (h_b : b = 3)
  (h_c : c = 4)
  (h_M_on_ellipse : M.snd^2 / (a^2) + M.fst^2 / (b^2) = 1)
  (h_dist_F1M : dist M F1 = 8) :
  d = 5 / 2 :=
by
  sorry

end distance_to_directrix_l792_792844


namespace prob1_prob2_l792_792037

variable (f g : ℝ → ℝ)
variable (a m : ℝ)
variable (x : ℝ)
variable (x1 x2 : ℝ)
variable (ln : ℝ → ℝ := Real.log)
variable (h : ℝ → ℝ := λ t, t - (1 / t) - 2*ln t)
variable (t : ℝ)

-- Definition of f(x) = ln x + a x
def f_def : Prop := ∀ x, f x = ln x + a * x

-- Definition of g(x) = f(x) + x + 1/(2x) - m
def g_def : Prop := ∀ x, g x = f x + x + (1 / (2 * x)) - m

-- Monotonically decreasing condition
def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 < x → x < y → f y ≤ f x

-- Question 1: if f(x) is monotonically decreasing, then a ≤ -1
theorem prob1 (H1 : monotone_decreasing f) (Hf : f_def f a) : a ≤ -1 := 
sorry

-- g(x) zeros definition
def g_has_two_zeros : Prop :=
  ∃ x1 x2, x1 < x2 ∧ g x1 = 0 ∧ g x2 = 0

-- Question 2: if g(x) has two zeros and a = -1, then x1 + x2 > 1
theorem prob2 (Hg : g_def g a m) (Hf : f_def f a) (Ha : a = -1) (Hz : g_has_two_zeros) : x1 + x2 > 1 := 
sorry

end prob1_prob2_l792_792037


namespace probability_of_sum_perfect_square_or_prime_l792_792268

noncomputable def probability_perfect_square_or_prime : ℚ :=
  let outcomes := finset.product (finset.range 6) (finset.range 6) 
  let is_perfect_square_or_prime (x : ℕ) : Prop :=
    x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7 ∨ x = 11 ∨ x = 4 ∨ x = 9
  let favorable_outcomes := outcomes.filter (λ p, is_perfect_square_or_prime (p.1 + p.2 + 2))
  ((favorable_outcomes.card : ℚ) / (outcomes.card : ℚ))

theorem probability_of_sum_perfect_square_or_prime :
  probability_perfect_square_or_prime = 11 / 18 :=
sorry

end probability_of_sum_perfect_square_or_prime_l792_792268


namespace angle_comparison_in_triangle_l792_792500

open Real

-- Define the problem in Lean
theorem angle_comparison_in_triangle
  (A B C : Type)
  [triangle A B C]
  {medianA : line A}
  {angleBisectorA : line A}
  {altitudeA : line A}
  (hM : median_from A)
  (hL : angle_bisector_from A)
  (hD : altitude_from A)
  (angle_A : ℝ) :
  (angle_A < 90 → angle_between medianA angleBisectorA < angle_between angleBisectorA altitudeA) ∧
  (angle_A > 90 → angle_between medianA angleBisectorA > angle_between angleBisectorA altitudeA) ∧
  (angle_A = 90 ↔ angle_between medianA angleBisectorA = angle_between angleBisectorA altitudeA) :=
by
  sorry

end angle_comparison_in_triangle_l792_792500


namespace chairs_to_exclude_l792_792691

theorem chairs_to_exclude (chairs : ℕ) (h : chairs = 765) : 
  ∃ n, n^2 ≤ chairs ∧ chairs - n^2 = 36 := 
by 
  sorry

end chairs_to_exclude_l792_792691


namespace ellipse_equation_l792_792842

noncomputable def ellipse_constant_a : ℝ := sqrt 2
noncomputable def ellipse_constant_c : ℝ := 1
noncomputable def ellipse_constant_b : ℝ := 1
def eccentricity : ℝ := sqrt 2 / 2
def min_distance_f1 : ℝ := sqrt 2 - 1

theorem ellipse_equation (a b c : ℝ) 
  (h1 : a = ellipse_constant_a) 
  (h2 : b = ellipse_constant_b) 
  (h3 : c = ellipse_constant_c) 
  (h4 : a > b) 
  (h5 : b > 0) 
  (h6 : c/a = eccentricity) 
  (h7 : a - c = min_distance_f1) : 
  (ell_eq : (∀ x y : ℝ, (x^2) / (a^2) + y^2 / (b^2) = 1)
  ∧ angle_B2F1F2 : ∀ F1 F2 B2 P : ℝ, 
    (F1 = x ∧ F2 = ellipse_constant_b ∧ 
     B2 = ellipse_constant_c ∧ O = 0 → 
     angle_B2F1F2 F1 F2 B2 O = 45) :=
begin
  sorry
end

end ellipse_equation_l792_792842


namespace units_digit_same_l792_792548

theorem units_digit_same (a n : ℤ) :
  (a^n % 10) = (a^(n + 4) % 10) := sorry

end units_digit_same_l792_792548


namespace invalid_reasoning_l792_792335

-- Declare our main theorem with the necessary conditions
theorem invalid_reasoning 
  (line_in_plane_parallel_all_lines_in_plane : ∀ (L : Type) (P : set L) 
    (line L : L) (plane_L : P) (parallel_to_plane : L → P → Prop), 
    parallel_to_plane line plane_L → ∀ line_in_plane : P, line_parallel line line_in_plane)
  (b_not_in_alpha : ∀ (L : Type) (b : L) (alpha : set L), ¬ alpha.contains b)
  (a_in_alpha : ∀ (L : Type) (a : L) (alpha : set L), alpha.contains a)
  (b_parallel_to_alpha : ∀ (L : Type) (b : L) (alpha : set L), b.parallel_to alpha) :
  ∃ (b a : L), ¬ (b.parallel_to a) := 
sorry

end invalid_reasoning_l792_792335


namespace decreasing_function_l792_792196

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem decreasing_function (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ ≤ 1) : 
  f x₁ > f x₂ :=
by
  -- Proof goes here
  sorry

end decreasing_function_l792_792196


namespace combination_sum_l792_792365

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Given conditions
axiom combinatorial_identity (n r : ℕ) : combination n r + combination n (r + 1) = combination (n + 1) (r + 1)

-- The theorem we aim to prove
theorem combination_sum : combination 8 2 + combination 8 3 + combination 9 2 = 120 := 
by
  sorry

end combination_sum_l792_792365


namespace vkontakte_membership_l792_792311

variables (M I A P: Prop)

theorem vkontakte_membership :
  (M → (I ∧ A)) → 
  (A ↔ ¬P) → 
  (I ∨ M) → 
  (P ↔ I) → 
  (¬M ∧ I ∧ ¬A ∧ P) := by {
    intro h1 h2 h3 h4,
    -- Insert the rest of the proof here
    sorry
  }

end vkontakte_membership_l792_792311


namespace sum_of_digits_eq_4_l792_792099

def sum_digits (n : Nat) : Nat :=
  n.digits 10 |> List.sum

def first_year_after (y : Nat) (p : Nat -> Prop) : Nat :=
  (Nat.iterate (· + 1) (1 + y) (fun n => p n) y)

theorem sum_of_digits_eq_4 : first_year_after 2020 (fun n => sum_digits n = 4) = 2030 :=
  sorry

end sum_of_digits_eq_4_l792_792099


namespace Megan_not_lead_plays_l792_792574

def total_plays : ℕ := 100
def lead_percentage : ℝ := 0.80
def lead_plays : ℕ := (total_plays : ℝ * lead_percentage).toNat
def not_lead_plays : ℕ := total_plays - lead_plays

theorem Megan_not_lead_plays : not_lead_plays = 20 := by
  sorry

end Megan_not_lead_plays_l792_792574


namespace find_pq_l792_792920

theorem find_pq (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hline : ∀ x y : ℝ, px + qy = 24) 
  (harea : (1 / 2) * (24 / p) * (24 / q) = 48) : p * q = 12 :=
by
  sorry

end find_pq_l792_792920


namespace original_polygon_sides_l792_792710

theorem original_polygon_sides {n : ℕ} 
    (hn : (n - 2) * 180 = 1080) : n = 7 ∨ n = 8 ∨ n = 9 :=
sorry

end original_polygon_sides_l792_792710


namespace intersection_of_M_and_N_l792_792093

def M : Set ℝ := {x | x ≥ 0 ∧ x < 16}
def N : Set ℝ := {x | x ≥ 1/3}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l792_792093


namespace number_of_girls_in_class_l792_792110

theorem number_of_girls_in_class (B G : ℕ) (h1 : G = 4 * B / 10) (h2 : B + G = 35) : G = 10 :=
by
  sorry

end number_of_girls_in_class_l792_792110


namespace repeating_period_eq_prime_period_l792_792555

theorem repeating_period_eq_prime_period (n p : ℕ) (hp : nat.prime p) (hn : n ≤ p - 1) : 
  let d := (minimalPeriod p)
  in minimalPeriod (10^d / p) = d :=
by
  sorry

end repeating_period_eq_prime_period_l792_792555


namespace abs_reciprocal_of_neg_six_l792_792220

theorem abs_reciprocal_of_neg_six : (| 1 / -6 |) = (1 / 6) :=
by
  sorry

end abs_reciprocal_of_neg_six_l792_792220


namespace circumsphere_radius_of_trirectangular_tetrahedron_l792_792125

theorem circumsphere_radius_of_trirectangular_tetrahedron {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ R : ℝ, R = (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) :=
begin
  use (1 / 2) * Real.sqrt (a^2 + b^2 + c^2),
  sorry
end

end circumsphere_radius_of_trirectangular_tetrahedron_l792_792125


namespace triangle_inequality_l792_792557

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_l792_792557


namespace tom_apple_slices_left_l792_792261

theorem tom_apple_slices_left
  (initial_apples : ℕ)
  (slices_per_apple : ℕ)
  (fraction_given_to_jerry : ℚ)
  (fraction_eaten_by_tom : ℚ)
  (initial_slices : ℕ := initial_apples * slices_per_apple)
  (slices_given_to_jerry : ℕ := fraction_given_to_jerry * initial_slices)
  (remaining_slices : ℕ := initial_slices - slices_given_to_jerry)
  (slices_eaten_by_tom : ℕ := fraction_eaten_by_tom * remaining_slices)
  (slices_left : ℕ := remaining_slices - slices_eaten_by_tom) :
  initial_apples = 2 →
  slices_per_apple = 8 →
  fraction_given_to_jerry = 3 / 8 →
  fraction_eaten_by_tom = 1 / 2 →
  slices_left = 5 :=
by
  intros h1 h2 h3 h4
  have hs1: initial_slices = 2 * 8, from by rw [h1, h2],
  have hs2: slices_given_to_jerry = (3 / 8) * 16, from by rw [h3, hs1],
  have hs3: remaining_slices = 16 - 6, from by rw [hs2],
  have hs4: slices_eaten_by_tom = (1 / 2) * 10, from by rw [h4, hs3],
  have hs5: slices_left = 10 - 5, from by rw [hs4],
  exact hs5

end tom_apple_slices_left_l792_792261


namespace rectangle_of_parallel_sides_and_equal_angles_l792_792951

theorem rectangle_of_parallel_sides_and_equal_angles
  (A B C D : Type)
  (AD_parallel_BC : ∀ x : A, x ∈ AD → is_parallel x BC)
  (AB_eq_CD : AB = CD)
  (angle_A_eq_angle_B : ∠ A = ∠ B) :
  is_rectangle (quadrilateral A B C D) :=
by
  sorry

end rectangle_of_parallel_sides_and_equal_angles_l792_792951


namespace find_smaller_root_l792_792751

theorem find_smaller_root :
  ∀ x : ℝ, (x - 2 / 3) ^ 2 + (x - 2 / 3) * (x - 1 / 3) = 0 → x = 1 / 2 :=
by
  sorry

end find_smaller_root_l792_792751


namespace graph_behavior_g_l792_792620

noncomputable def g (x : ℝ) : ℝ := 3*x^4 - 4*x^3 + x - 5

theorem graph_behavior_g :
  (∀ (x : ℝ), x → ∞ → g x → ∞) ∧ (∀ (x : ℝ), x → -∞ → g x → ∞) :=
  sorry

end graph_behavior_g_l792_792620


namespace cotangent_ratio_l792_792549

-- Let x, y, z be the sides of a triangle.
-- Let ξ, η, ζ be the angles opposite these sides respectively.
variables {x y z : ℝ} {ξ η ζ : ℝ}

-- Assume the given condition: x^2 + y^2 = 2023 * z^2
axiom cond : x^2 + y^2 = 2023 * z^2

-- Define the cotangent function
noncomputable def cot (θ : ℝ) : ℝ :=
  real.cos θ / real.sin θ

-- The problem statement to prove the required equation
theorem cotangent_ratio (h : x > 0) (h1 : y > 0) (h2 : z > 0) (h3 : real.sin ξ ≠ 0) (h4 : real.sin η ≠ 0) (h5 : real.sin ζ ≠ 0) :
  (cot ζ) / (cot ξ + cot η) = 1011 :=
begin
  sorry
end

end cotangent_ratio_l792_792549


namespace find_omega_l792_792039

theorem find_omega (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x : ℝ, 2 * sin(ω * x + π / 6) = 2 * sin(ω * (x + 2 * π / 3) + π / 6)) : ω = 3 := by
  sorry

end find_omega_l792_792039


namespace range_of_g_l792_792759

noncomputable def g (x : ℝ) : ℝ := if x ≠ -5 then 3 * (x - 4) else 0

theorem range_of_g : 
  (set.range g) = {y : ℝ | y ≠ -27} :=
by
  sorry

end range_of_g_l792_792759


namespace intersection_M_N_l792_792087

section

def M (x : ℝ) : Prop := sqrt x < 4
def N (x : ℝ) : Prop := 3 * x >= 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | (1 / 3) ≤ x ∧ x < 16} :=
by
  sorry

end

end intersection_M_N_l792_792087


namespace columbian_coffee_price_is_correct_l792_792524

-- Definitions based on the conditions
def total_mix_weight : ℝ := 100
def brazilian_coffee_price_per_pound : ℝ := 3.75
def final_mix_price_per_pound : ℝ := 6.35
def columbian_coffee_weight : ℝ := 52

-- Let C be the price per pound of the Columbian coffee
noncomputable def columbian_coffee_price_per_pound : ℝ := sorry

-- Define the Lean 4 proof problem
theorem columbian_coffee_price_is_correct :
  columbian_coffee_price_per_pound = 8.75 :=
by
  -- Total weight and calculation based on conditions
  let brazilian_coffee_weight := total_mix_weight - columbian_coffee_weight
  let total_value_of_columbian := columbian_coffee_weight * columbian_coffee_price_per_pound
  let total_value_of_brazilian := brazilian_coffee_weight * brazilian_coffee_price_per_pound
  let total_value_of_mix := total_mix_weight * final_mix_price_per_pound
  
  -- Main equation based on the mix
  have main_eq : total_value_of_columbian + total_value_of_brazilian = total_value_of_mix :=
    by sorry

  -- Solve for C (columbian coffee price per pound)
  sorry

end columbian_coffee_price_is_correct_l792_792524


namespace no_such_integers_exist_l792_792198

theorem no_such_integers_exist :
  ¬(∃ (a b c d : ℤ), a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2) :=
by
  sorry

end no_such_integers_exist_l792_792198


namespace product_fraction_eq_l792_792390

theorem product_fraction_eq : 
  ( ∏ i in Finset.range 100, (1 - (1 / (i + 2 : ℝ))) ) = (1 / 101) := 
by
  sorry

end product_fraction_eq_l792_792390


namespace length_first_train_l792_792697

-- Definitions of the conditions
def speed_first_train_kmph : ℝ := 120
def speed_second_train_kmph : ℝ := 80
def length_second_train_m : ℝ := 220.04
def crossing_time_s : ℝ := 9

-- Conversion factor and relative speed calculation
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 5 / 18
def relative_speed_mps : ℝ := kmph_to_mps (speed_first_train_kmph + speed_second_train_kmph)

-- Total distance covered when trains crossed each other
def total_distance_covered_m : ℝ := relative_speed_mps * crossing_time_s

-- Prove the length of the first train
theorem length_first_train : 
  let length_first_train_m := total_distance_covered_m - length_second_train_m in
  length_first_train_m = 280 :=
by
  sorry

end length_first_train_l792_792697


namespace intersection_of_sets_l792_792049

theorem intersection_of_sets :
  let A := {-2, -1, 0, 1, 2}
  let B := {x | -2 < x ∧ x ≤ 2}
  A ∩ B = {-1, 0, 1, 2} :=
by
  sorry

end intersection_of_sets_l792_792049


namespace age_difference_correct_l792_792526

-- Define the ages of John and his parents based on given conditions
def John_age (father_age : ℕ) : ℕ :=
  father_age / 2

def mother_age (father_age : ℕ) : ℕ :=
  father_age - 4

def age_difference (john_age : ℕ) (mother_age : ℕ) : ℕ :=
  abs (john_age - mother_age)

-- Main theorem stating the age difference between John and his mother
theorem age_difference_correct (father_age : ℕ) (h : father_age = 40) :
  age_difference (John_age father_age) (mother_age father_age) = 16 :=
by
  sorry

end age_difference_correct_l792_792526


namespace vkontakte_membership_l792_792309

variables (M I A P: Prop)

theorem vkontakte_membership :
  (M → (I ∧ A)) → 
  (A ↔ ¬P) → 
  (I ∨ M) → 
  (P ↔ I) → 
  (¬M ∧ I ∧ ¬A ∧ P) := by {
    intro h1 h2 h3 h4,
    -- Insert the rest of the proof here
    sorry
  }

end vkontakte_membership_l792_792309


namespace worker_spends_amount_l792_792725

theorem worker_spends_amount :
  ∃ S : ℝ, (0 < S) ∧ S = 16 ∧ (24 - S + 24 - S + 24 - S + 24 - S + 24 - S + 24 - S + 24 - S + 24 - 4S = 48 - S) :=
by sorry

end worker_spends_amount_l792_792725


namespace XY_passes_through_H_l792_792178

noncomputable def points := Type
variables (e f a b c d : points → Prop)
variables (H A B C D X Y : points)

axiom lines_perpendicular : ∀ {p q : points → Prop}, ∃ h : points, p h ∧ q h
axiom H_intersection : lines_perpendicular e f H
axiom A_on_e : e A
axiom B_on_e : e B
axiom C_on_f : f C
axiom D_on_f : f D
axiom a_perpendicular_to_BD : ∀ {q : points → Prop}, q A → ∃ p : points → Prop, lines_perpendicular p q
axiom c_perpendicular_to_BD : ∀ {q : points → Prop}, q C → ∃ p : points → Prop, lines_perpendicular p q
axiom b_perpendicular_to_AC : ∀ {q : points → Prop}, q B → ∃ p : points → Prop, lines_perpendicular p q
axiom d_perpendicular_to_AC : ∀ {q : points → Prop}, q D → ∃ p : points → Prop, lines_perpendicular p q
axiom a_b_intersect_at_X : ∃ X : points, a X ∧ b X
axiom c_d_intersect_at_Y : ∃ Y : points, c Y ∧ d Y
axiom distinct_points : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ H ∧ B ≠ C ∧ B ≠ D ∧ B ≠ H ∧ C ≠ D ∧ C ≠ H ∧ D ≠ H

theorem XY_passes_through_H : (∀ {x y : points}, x ≠ H → y ≠ H → 
  ∃ XY_H : points → Prop, lines_perpendicular XY_H (λ _ : points, True) ∧ XY_H H) ∧
  a X ∧ c Y → 
  (∀ XY_H : points → Prop, XY_H H) :=
by
  sorry

end XY_passes_through_H_l792_792178


namespace maximum_B_k_at_181_l792_792776

open Nat

theorem maximum_B_k_at_181 :
  let B : ℕ → ℝ := λ k, (Nat.choose 2000 k : ℝ) * (0.1 ^ k)
  ∃ k : ℕ, k ≤ 2000 ∧ (∀ m : ℕ, m ≤ 2000 → B m ≤ B 181) :=
by
  let B := λ k : ℕ, (Nat.choose 2000 k : ℝ) * (0.1 ^ k)
  use 181
  split
  · linarith
  · intro m hm
    sorry

end maximum_B_k_at_181_l792_792776


namespace number_of_valid_pairs_l792_792809

def no_zero_digit (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ (Nat.digits 10 n) → d ≠ 0

theorem number_of_valid_pairs : 
  (∃ n : ℕ,  n = (finset.card { p : ℕ × ℕ | (p.1 + p.2 = 500) ∧ no_zero_digit p.1 ∧ no_zero_digit p.2 })) :=
  sorry

end number_of_valid_pairs_l792_792809


namespace right_triangle_cos_angle_l792_792499

theorem right_triangle_cos_angle
  (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (AB AC BC : ℝ)
  (h_adj : AB = 5) 
  (h_hyp : AC = 13) 
  (angle_right : (AB, AC, BC) ∈ metric_space.angle_abc 90) : 
  ∃ (cosA : ℝ), cosA = AB / AC :=
by {
  sorry
}

end right_triangle_cos_angle_l792_792499


namespace solve_expression_l792_792678

theorem solve_expression :
  (sqrt 1.21 / sqrt 0.64 + sqrt 1.44 / sqrt 0.49) ≈ 3.0893 := by
  sorry

end solve_expression_l792_792678


namespace coeff_x2_in_expansion_l792_792611

theorem coeff_x2_in_expansion : 
  (∃ c : ℕ, c * x^2 ∈ (1 + 2 * x)^5) → c = 40 := 
by
  sorry

end coeff_x2_in_expansion_l792_792611


namespace writer_productivity_l792_792726

theorem writer_productivity (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ)
  (h_total_words : total_words = 60000)
  (h_total_hours : total_hours = 120)
  (h_break_hours : break_hours = 20) :
  (total_words / (total_hours - break_hours) = 600) :=
by
  rw [h_total_words, h_total_hours, h_break_hours]
  norm_num

end writer_productivity_l792_792726


namespace ratio_length_breadth_l792_792234

noncomputable def b : ℝ := 18
noncomputable def l : ℝ := 972 / b

theorem ratio_length_breadth
  (A : ℝ)
  (h1 : b = 18)
  (h2 : l * b = 972) :
  (l / b) = 3 :=
by
  sorry

end ratio_length_breadth_l792_792234


namespace connections_required_l792_792112

theorem connections_required (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end connections_required_l792_792112


namespace odd_factor_form_l792_792168

theorem odd_factor_form (x : ℤ) (d : ℕ) (h : ℤ) : 
  d ∣ (x^2 + 1) → (d % 2 = 1) → (∃ h : ℤ, d = 4 * h + 1) :=
by
  sorry

end odd_factor_form_l792_792168


namespace remainder_of_multiple_l792_792072

theorem remainder_of_multiple (m k : ℤ) (h1 : m % 5 = 2) (h2 : (2 * k) % 5 = 1) : 
  (k * m) % 5 = 1 := 
sorry

end remainder_of_multiple_l792_792072


namespace product_of_variables_l792_792302

theorem product_of_variables (a b c d : ℚ)
  (h1 : 4 * a + 5 * b + 7 * c + 9 * d = 56)
  (h2 : 4 * (d + c) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : c - 2 = d) :
  a * b * c * d = 58653 / 10716361 := 
sorry

end product_of_variables_l792_792302


namespace average_brown_mms_l792_792204

def brown_smiley_counts : List Nat := [9, 12, 8, 8, 3]
def brown_star_counts : List Nat := [7, 14, 11, 6, 10]

def average (lst : List Nat) : Float :=
  (lst.foldl (· + ·) 0).toFloat / lst.length.toFloat
  
theorem average_brown_mms :
  average brown_smiley_counts = 8 ∧
  average brown_star_counts = 9.6 :=
by 
  sorry

end average_brown_mms_l792_792204


namespace h_eq_x_solution_l792_792554

noncomputable def h (x : ℝ) : ℝ := (3 * ((x + 3) / 5) + 10)

theorem h_eq_x_solution (x : ℝ) (h_cond : ∀ y, h (5 * y - 3) = 3 * y + 10) : h x = x → x = 29.5 :=
by
  sorry

end h_eq_x_solution_l792_792554


namespace jordans_greatest_average_speed_l792_792739

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s.reverse = s

theorem jordans_greatest_average_speed :
  ∃ (v : ℕ), 
  ∃ (d : ℕ), 
  ∃ (end_reading : ℕ), 
  is_palindrome 72327 ∧ 
  is_palindrome end_reading ∧ 
  72327 < end_reading ∧ 
  end_reading - 72327 = d ∧ 
  d ≤ 240 ∧ 
  end_reading ≤ 72327 + 240 ∧ 
  v = d / 4 ∧ 
  v = 50 :=
sorry

end jordans_greatest_average_speed_l792_792739


namespace range_f_period_f_decreasing_f_l792_792238

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) - cos (2 * x + π / 6)

theorem range_f : ∀ x : ℝ, -sqrt 3 ≤ f x ∧ f x ≤ sqrt 3 :=
sorry

theorem period_f : ∀ x : ℝ, f (x + π) = f x :=
sorry

theorem decreasing_f : ∀ k : ℤ, ∀ x : ℝ, (π / 3 + k * π ≤ x ∧ x ≤ 5 * π / 6 + k * π) → 2 * sqrt 3 * cos (2 * x - π / 6) < 0 :=
sorry

end range_f_period_f_decreasing_f_l792_792238


namespace sum_of_solutions_x_l792_792505

theorem sum_of_solutions_x (y : ℝ) (h1 : y = 9) (x : ℝ) (h2 : x^2 + y^2 = 169) :
  ∑ x in {r : ℝ | r^2 = 88}, id x = 0 :=
by
  sorry

end sum_of_solutions_x_l792_792505


namespace common_divisors_9240_8820_l792_792890

-- Define the prime factorizations given in the problem.
def pf_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def pf_8820 := [(2, 2), (3, 2), (5, 1), (7, 2)]

-- Define a function to calculate the gcd of two numbers given their prime factorizations.
def gcd_factorizations (pf1 pf2 : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
    List.filterMap (λ ⟨p, e1⟩ =>
      match List.lookup p pf2 with
      | some e2 => some (p, min e1 e2)
      | none => none
      end) pf1 

-- Define a function to compute the number of divisors from the prime factorization.
def num_divisors (pf: List (ℕ × ℕ)) : ℕ :=
    pf.foldl (λ acc ⟨_, e⟩ => acc * (e + 1)) 1

-- The Lean statement for the problem
theorem common_divisors_9240_8820 : 
    num_divisors (gcd_factorizations pf_9240 pf_8820) = 24 :=
by
    -- The proof goes here. We include sorry to indicate that the proof is omitted.
    sorry

end common_divisors_9240_8820_l792_792890


namespace olive_needs_two_colours_l792_792580

theorem olive_needs_two_colours (α : Type) [Finite α] (G : SimpleGraph α) (colour : α → Fin 2) :
  (∀ v : α, ∃! w : α, G.Adj v w ∧ colour v = colour w) → ∃ color_map : α → Fin 2, ∀ v, ∃! w, G.Adj v w ∧ color_map v = color_map w :=
sorry

end olive_needs_two_colours_l792_792580


namespace total_length_of_intervals_l792_792401

theorem total_length_of_intervals :
  (∀ (x : ℝ), |x| < 1 → Real.tan (Real.log x / Real.log 5) < 0) →
  ∃ (length : ℝ), length = (2 * (5 ^ (Real.pi / 2))) / (1 + (5 ^ (Real.pi / 2))) :=
sorry

end total_length_of_intervals_l792_792401


namespace mark_points_acute_triangles_l792_792519

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem mark_points_acute_triangles (n : ℕ) (hn1 : is_odd n) (hn2 : 3 < n) :
  ∃ (points : Fin n → EuclideanGeometry.Point), 
    ∀ i : Fin n, 
      (EuclideanGeometry.angle (points i) (points ((i + 1) % n)) (points ((i + 2) % n)) < π / 2) := sorry

end mark_points_acute_triangles_l792_792519


namespace find_x_given_distance_l792_792130

theorem find_x_given_distance (x : ℝ) : abs (x - 4) = 1 → (x = 5 ∨ x = 3) :=
by
  intro h
  sorry

end find_x_given_distance_l792_792130


namespace min_total_cost_of_container_l792_792337

-- Definitions from conditions
def container_volume := 4 -- m^3
def container_height := 1 -- m
def cost_per_square_meter_base : ℝ := 20
def cost_per_square_meter_sides : ℝ := 10

-- Proving the minimum total cost
theorem min_total_cost_of_container :
  ∃ (a b : ℝ), a * b = container_volume ∧
                (20 * (a + b) + 20 * (a * b)) = 160 :=
by
  sorry

end min_total_cost_of_container_l792_792337


namespace keiko_jogging_speed_l792_792532

variable (s : ℝ) -- Keiko's jogging speed
variable (b : ℝ) -- radius of the inner semicircle
variable (L_inner : ℝ := 200 + 2 * Real.pi * b) -- total length of the inner track
variable (L_outer : ℝ := 200 + 2 * Real.pi * (b + 8)) -- total length of the outer track
variable (t_inner : ℝ := L_inner / s) -- time to jog the inside edge
variable (t_outer : ℝ := L_outer / s) -- time to jog the outside edge
variable (time_difference : ℝ := 48) -- time difference between jogging inside and outside edges

theorem keiko_jogging_speed : L_inner = 200 + 2 * Real.pi * b →
                           L_outer = 200 + 2 * Real.pi * (b + 8) →
                           t_outer = t_inner + 48 →
                           s = Real.pi / 3 :=
by
  intro h1 h2 h3
  sorry

end keiko_jogging_speed_l792_792532


namespace principal_amount_l792_792489

theorem principal_amount (P : ℝ) : 
  P * (1.03)^2 * (1.04)^3 * (1.05)^2 = P + 5000 → 
  P ≈ 15714.93 :=
by
  intro h

  -- Variables and definitions
  let compound_factor := (1.03)^2 * (1.04)^3 * (1.05)^2
  have comp_eq : compound_factor = 1.318098
    sorry

  have eqn : P * compound_factor = P + 5000 := h

  -- Start simplifying the equation
  have eqn_simplified : P * 0.318098 = 5000
    sorry

  -- Solve for P and conclude
  have result : P = 5000 / 0.318098 := by sorry
  exact (result ≈ 15714.93)

end principal_amount_l792_792489


namespace bike_ride_distance_l792_792357

theorem bike_ride_distance (x : ℝ) 
  (h_west : 30 - 15 = 15)
  (h_hypotenuse : 15 ^ 2 + (x + 18) ^ 2 = 28.30194339616981 ^ 2) : 
  x ≈ 6.020274 :=
by 
  sorry

end bike_ride_distance_l792_792357


namespace cricket_run_percentage_l792_792677

theorem cricket_run_percentage (total_runs boundaries sixes : ℕ) :
  total_runs = 152 ∧ boundaries = 12 ∧ sixes = 2 → 
  let runs_from_boundaries := boundaries * 4,
      runs_from_sixes := sixes * 6,
      runs_from_boundaries_and_sixes := runs_from_boundaries + runs_from_sixes,
      runs_by_running := total_runs - runs_from_boundaries_and_sixes,
      percentage_by_running := (runs_by_running / total_runs.to_rat) * 100 
  in percentage_by_running ≈ 60.53 := 
by
  intros h
  cases h with Htotal Hbound_six
  cases Hbound_six with Hbound Hsix
  unfold runs_from_boundaries runs_from_sixes runs_from_boundaries_and_sixes runs_by_running percentage_by_running
  -- use rat conditions and float approximation for percentages
  sorry

end cricket_run_percentage_l792_792677


namespace five_digit_numbers_count_l792_792754

theorem five_digit_numbers_count :
  (∃ digits : Finset ℕ, digits = {1, 2, 3, 4, 5} ∧ 
   ∀ d ∈ digits, d ≥ 1 ∧ d ≤ 5 ∧
   ∀ num : List ℕ, num.length = 5 ∧ num.nodup ∧ digits.to_list.perms = [num] ∧
   (num[2] ≠ 5) ∧ (num[0] ≠ 2 ∧ num[0] ≠ 4) ∧ (num[4] ≠ 2 ∧ num[4] ≠ 4) ∧
   (∃ n : ℕ, n = num.foldl (λ acc x, 10 * acc + x) 0) ∧ 
   ∀ num : List ℕ, num.length = 5 ∧ num.nodup → 
   (num[2] ≠ 5 ∧ num[0] ≠ 2 ∧ num[0] ≠ 4 ∧ num[4] ≠ 2 ∧ num[4] ≠ 4)
  → Fintype.card {n // ∃ num : List ℕ, num.length = 5 ∧ num.nodup ∧ (num.foldl (λ acc x, 10 * acc + x) 0) = n} = 32) ∧
   Fintype.card {num : List ℕ // num.length = 5 ∧ num.nodup ∧ (num[2] ≠ 5) ∧ (num[0] ≠ 2 ∧ num[0] ≠ 4) ∧ (num[4] ≠ 2 ∧ num[4] ≠ 4)} = 32 :=
sorry

end five_digit_numbers_count_l792_792754


namespace units_digit_l_squared_plus_2_pow_l_l792_792988

/-- Let l = 15^2 + 2^15. Prove that the units digit of l^2 + 2^l is 7. -/
theorem units_digit_l_squared_plus_2_pow_l :
  let l := 15^2 + 2^15 in
  (l^2 + 2^l) % 10 = 7 :=
by
  sorry

end units_digit_l_squared_plus_2_pow_l_l792_792988


namespace area_relationship_l792_792719

theorem area_relationship (a b c : ℝ) (h : a^2 + b^2 = c^2) : (a + b)^2 = a^2 + 2*a*b + b^2 := 
by sorry

end area_relationship_l792_792719


namespace sin_cos_expression_l792_792249

noncomputable def sin_45 := Real.sin (Real.pi / 4)
noncomputable def cos_15 := Real.cos (Real.pi / 12)
noncomputable def cos_225 := Real.cos (5 * Real.pi / 4)
noncomputable def sin_15 := Real.sin (Real.pi / 12)

theorem sin_cos_expression :
  sin_45 * cos_15 + cos_225 * sin_15 = 1 / 2 :=
by
  sorry

end sin_cos_expression_l792_792249


namespace average_of_remaining_two_l792_792681

-- Definitions (conditions)
def avg_five (a b c d e : ℕ) : ℕ := (a + b + c + d + e) / 5
def avg_three (a b c : ℕ) : ℕ := (a + b + c) / 3

-- Main statement of the proof problem
theorem average_of_remaining_two 
    (a b c d e : ℕ)
    (h1 : avg_five a b c d e = 11)
    (h2 : avg_three a b c = 4) : 
    (d + e) / 2 = 21.5 :=
by
  sorry

end average_of_remaining_two_l792_792681


namespace peter_stamps_l792_792583

theorem peter_stamps (M : ℕ) (h1 : M % 5 = 2) (h2 : M % 11 = 2) (h3 : M % 13 = 2) (h4 : M > 1) : M = 717 :=
by
  -- proof will be filled in
  sorry

end peter_stamps_l792_792583


namespace range_of_lambda_l792_792034

noncomputable def a_n (n : ℕ) (λ : ℝ) : ℝ :=
  2 * n^2 + λ * n

def increasing_sequence (sequence : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, sequence (n + 1) > sequence n

theorem range_of_lambda (λ : ℝ) :
  increasing_sequence (λ n, a_n n λ) → λ > -6 :=
by
  sorry

end range_of_lambda_l792_792034


namespace smallest_n_int_expr_l792_792163

noncomputable def a : ℝ := Real.pi / 2010

def expr (n : ℕ) : ℝ :=
  2 * (Finset.sum (Finset.range n) (λ k, cos ((k.succ : ℝ)^2 * a) * sin ((k.succ : ℝ) * a)))

def is_integer (x : ℝ) : Prop :=
  ∃ (m : ℤ), x = m

theorem smallest_n_int_expr : ∃ n : ℕ, (0 < n) ∧ is_integer (expr n) ∧ ∀ m : ℕ, (0 < m ∧ m < n) → ¬ is_integer (expr m) :=
by
  sorry

end smallest_n_int_expr_l792_792163


namespace eight_percent_is_64_l792_792766

-- Definition of the condition
variable (x : ℝ)

-- The theorem that states the problem to be proven
theorem eight_percent_is_64 (h : (8 / 100) * x = 64) : x = 800 :=
sorry

end eight_percent_is_64_l792_792766


namespace sum_of_solutions_x_l792_792506

theorem sum_of_solutions_x (y : ℝ) (h1 : y = 9) (x : ℝ) (h2 : x^2 + y^2 = 169) :
  ∑ x in {r : ℝ | r^2 = 88}, id x = 0 :=
by
  sorry

end sum_of_solutions_x_l792_792506


namespace problem_l792_792407

open Matrix

-- Define the system of equations as a matrix multiplication equated to zero
def system_matrix (k : ℚ) : Matrix (Fin 4) (Fin 4) ℚ :=
  ![![1, 2*k, 4, -1],
    ![4, k, 2, 1],
    ![3, 5, -3, 2],
    ![2, 3, 1, -4]]

-- The theorem stating the required conclusion
theorem problem (a x y z w : ℚ) (h1 : a ≠ 0) (h2 : x = 3 * a) (h3 : y = a)
  (h4 : z = 5 * a) (h5 : w = 2 * a) (k := 60 / 7) (h : (system_matrix k).mul_vec ![x, y, z, w] = 0) :
  (x * y) / (z * w) = 3 / 10 :=
by
  sorry

end problem_l792_792407


namespace sum_of_digits_eq_4_l792_792098

def sum_digits (n : Nat) : Nat :=
  n.digits 10 |> List.sum

def first_year_after (y : Nat) (p : Nat -> Prop) : Nat :=
  (Nat.iterate (· + 1) (1 + y) (fun n => p n) y)

theorem sum_of_digits_eq_4 : first_year_after 2020 (fun n => sum_digits n = 4) = 2030 :=
  sorry

end sum_of_digits_eq_4_l792_792098


namespace difference_after_iterations_l792_792267

theorem difference_after_iterations
  (a b : ℝ) (h : a > b) :
  ∃ k : ℕ, 2^(-k) * (a - b) < 1 / 2002 := 
sorry

end difference_after_iterations_l792_792267


namespace length_of_platform_correct_l792_792722

noncomputable def length_of_platform (train_length : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) : ℕ :=
  let train_speed_mps := (train_speed_kmph * 1000) / 3600 in
  let total_distance := train_speed_mps * time_seconds in
  total_distance - train_length

theorem length_of_platform_correct :
  length_of_platform 200 72 25 = 300 := by
  sorry

end length_of_platform_correct_l792_792722


namespace range_of_a_l792_792041

def f (x : ℝ) : ℝ := x + 4 / x
def g (x a : ℝ) : ℝ := 2 ^ x + a

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Icc (1/2 : ℝ) 1, ∃ x2 ∈ set.Icc (2 : ℝ) 3, f x1 ≥ g x2 a) → a ≤ 1 := by
  sorry

end range_of_a_l792_792041


namespace find_30_cent_items_l792_792522

-- Define the parameters and their constraints
variables (a d b c : ℕ)

-- Define the conditions
def total_items : Prop := a + d + b + c = 50
def total_cost : Prop := 30 * a + 150 * d + 200 * b + 300 * c = 6000

-- The theorem to prove the number of 30-cent items purchased
theorem find_30_cent_items (h1 : total_items a d b c) (h2 : total_cost a d b c) : 
  ∃ a, a + d + b + c = 50 ∧ 30 * a + 150 * d + 200 * b + 300 * c = 6000 := 
sorry

end find_30_cent_items_l792_792522


namespace power_function_explicit_and_inequality_solution_l792_792475

theorem power_function_explicit_and_inequality_solution:
  ∀ (f : ℝ → ℝ) (m : ℤ),
  (∀ x, f x = (m^3 - m + 1 : ℝ) * x ^ (1/2 * (1 - 8 * m - m^2))) →
  (∀ x, f x ≠ 0) →
  (∀ x, f x = f (-x)) →
  (f = λ x, x ^ (-4)) ∧ { x : ℝ | f (x + 1) > f (x - 2) } = { x : ℝ | x < 1/2 ∧ x ≠ -1 } :=
by
  intro f m h1 h2 h3
  sorry

end power_function_explicit_and_inequality_solution_l792_792475


namespace cos_double_angle_sub_pi_six_l792_792826

variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π / 3)
variable (h2 : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5)

theorem cos_double_angle_sub_pi_six :
  Real.cos (2 * α - π / 6) = 4 / 5 :=
by
  sorry

end cos_double_angle_sub_pi_six_l792_792826


namespace John_total_amount_l792_792525

theorem John_total_amount (x : ℝ)
  (h1 : ∃ x : ℝ, (3 * x * 5 * 3 * x) = 300):
  (x + 3 * x + 15 * x) = 380 := by
  sorry

end John_total_amount_l792_792525


namespace probability_at_least_three_prime_dice_l792_792371

-- Definitions from the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def p := 5 / 12
def q := 7 / 12
def binomial (n k : ℕ) := Nat.choose n k

-- The probability of at least three primes
theorem probability_at_least_three_prime_dice :
  (binomial 5 3 * p ^ 3 * q ^ 2) +
  (binomial 5 4 * p ^ 4 * q ^ 1) +
  (binomial 5 5 * p ^ 5 * q ^ 0) = 40625 / 622080 :=
by
  sorry

end probability_at_least_three_prime_dice_l792_792371


namespace find_angle_l792_792847

variables {V : Type*} [InnerProductSpace ℝ V]

open RealInnerProductSpace

-- Definitions of the conditions
def non_zero_vectors (a b : V) : Prop := (a ≠ 0) ∧ (b ≠ 0)
def norm_relation (a b : V) : Prop := ∥a∥ = sqrt 2 * ∥b∥
def perpendicular_condition (a b : V) : Prop := (a - b) ⬝ (2 • a + 3 • b) = 0

-- The statement of the problem
theorem find_angle (a b : V) (h1 : non_zero_vectors a b) (h2 : norm_relation a b) (h3 : perpendicular_condition a b) :
  real.angle a b = 3 * π / 4 :=
sorry

end find_angle_l792_792847


namespace n_power_d_eq_D_squared_l792_792173

theorem n_power_d_eq_D_squared (n : ℕ) (h : n > 0) (d : ℕ) (D : ℕ) 
  (hd : d = (n.divisors.filter (λ x, x > 0)).length) 
  (hD : D = (n.divisors.filter (λ x, x > 0)).prod id) : 
  n ^ d = D ^ 2 := 
sorry

end n_power_d_eq_D_squared_l792_792173


namespace tan_eleven_pi_over_four_eq_neg_one_l792_792787

noncomputable def tan_of_eleven_pi_over_four : Real := 
  let to_degrees (x : Real) : Real := x * 180 / Real.pi
  let angle := to_degrees (11 * Real.pi / 4)
  let simplified := angle - 360 * Real.floor (angle / 360)
  if simplified < 0 then
    simplified := simplified + 360
  if simplified = 135 then -1
  else
    undefined

theorem tan_eleven_pi_over_four_eq_neg_one :
  tan (11 * Real.pi / 4) = -1 := 
by
  sorry

end tan_eleven_pi_over_four_eq_neg_one_l792_792787


namespace ellipse_properties_lines_symmetric_l792_792865

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b)
  (h2 : b > 0) (h3 : a > 0) (eccentricity : ℝ) 
  (h_eccentricity : eccentricity = 1 / 2) : Prop := 
  ∃ (A : ℝ) (E : ℝ), 
  let c := (a * eccentricity) in
  let area_△OAE := ∥sqrt (3 : ℝ) ∥ in
  c = a / 2 ∧ b = sqrt 3 ∧
  1/2 * a * b = sqrt 3 ∧
  a^2 - b^2 = c^2 ∧
  ( ∀ x y,  (x^2 / (a^2) + y^2 / (b^2) = 1) ) 

noncomputable def symmetric_lines (a b c : ℝ) (x1 y1 x2 y2 t : ℝ) (h1 : x1 + x2 = -t) 
  (h2 : x1 * x2 = t^2 - 3) : Prop :=
  let k_MB := (y1 - 3/2) / (x1 - 1) in
  let k_MC := (y2 - 3/2) / (x2 - 1) in
  k_MB + k_MC = 0

theorem ellipse_properties : ellipse_equation 2 (sqrt 3) 
  (by norm_num) (by norm_num) (by norm_num) 0.5 
  (by norm_num) :=
sorry

theorem lines_symmetric : symmetric_lines 2 (sqrt 3) 1 1 3/2 (-2 : ℝ) 3/2 -2 
  (by norm_num) 
sorry

end ellipse_properties_lines_symmetric_l792_792865


namespace find_number_l792_792927

def x := 77.7
def percent_greater := 0.11
def y := x / (1 + percent_greater)

theorem find_number : y ≈ 70 := by
  sorry

end find_number_l792_792927


namespace max_possible_S_first_2019_integers_l792_792389

noncomputable def mark_sum (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (marks i)

theorem max_possible_S_first_2019_integers :
  ∀ (marks : ℕ → ℕ), (∀ k, marks k ∈ {0, 1, 2}) →
  (∀ k j, marks k = j → marks (k + j) = 0) →
  mark_sum 2019 ≤ 2021 :=
begin
  sorry
end

end max_possible_S_first_2019_integers_l792_792389


namespace count_pairs_satisfying_inequality_l792_792063

theorem count_pairs_satisfying_inequality : 
  (∑ m in finset.range 7, finset.card {n | n ∈ finset.range 20 ∧ m^2 + 2 * n < 40}) = 72 :=
sorry

end count_pairs_satisfying_inequality_l792_792063


namespace average_first_18_even_numbers_l792_792659

theorem average_first_18_even_numbers : 
  let first_even := 2
  let difference := 2
  let n := 18
  let last_even := first_even + (n - 1) * difference
  let sum := (n / 2) * (first_even + last_even)
  let average := sum / n
  average = 19 :=
by
  -- Definitions
  let first_even := 2
  let difference := 2
  let n := 18
  let last_even := first_even + (n - 1) * difference
  let sum := (n / 2) * (first_even + last_even)
  let average := sum / n
  -- The claim
  show average = 19
  sorry

end average_first_18_even_numbers_l792_792659


namespace ellipse_slope_sum_one_l792_792015

theorem ellipse_slope_sum_one (a b k₁ k₂: ℝ) (h₀: a > b) 
  (h₁: ∀ x y : ℝ, (x / a) ^ 2 + (y / b) ^ 2 = 1) 
  (h₂: ∃ P1 P2 Q: ℝ × ℝ, P1.1 = 1 ∧ P2.1 = 1 ∧ P1.2 > P2.2 ∧ 
                       Q = ⟨-2, -3⟩ ∧
                       ∀ M N: ℝ × ℝ, M ≠ N ∧ 
                       (∃ m₁: ℝ, M.2 + 3 = m₁ * (M.1 + 2)) ∧ 
                       (∃ m₂: ℝ, N.2 + 3 = m₂ * (N.1 + 2)) ∧ 
                       P1 ≠ M ∧ P1 ≠ N)
  (h₃: ∀ M N: ℝ × ℝ, M ≠ N → 
                      let k₁ := (M.2 - P1.2) / (M.1 - P1.1) in
                      let k₂ := (N.2 - P1.2) / (N.1 - P1.1) in
                      k₁ + k₂ = 1)
: k₁ + k₂ = 1 := sorry

end ellipse_slope_sum_one_l792_792015


namespace tan_eleven_pi_over_four_eq_neg_one_l792_792785

noncomputable def tan_of_eleven_pi_over_four : Real := 
  let to_degrees (x : Real) : Real := x * 180 / Real.pi
  let angle := to_degrees (11 * Real.pi / 4)
  let simplified := angle - 360 * Real.floor (angle / 360)
  if simplified < 0 then
    simplified := simplified + 360
  if simplified = 135 then -1
  else
    undefined

theorem tan_eleven_pi_over_four_eq_neg_one :
  tan (11 * Real.pi / 4) = -1 := 
by
  sorry

end tan_eleven_pi_over_four_eq_neg_one_l792_792785


namespace solution_set_is_interval_l792_792043

noncomputable def solution_set_of_inequality : set ℝ := { x | -x^2 - x + 6 > 0 }

theorem solution_set_is_interval :
  solution_set_of_inequality = { x | -3 < x ∧ x < 2 } :=
sorry

end solution_set_is_interval_l792_792043


namespace biased_die_odd_sum_probability_l792_792669

noncomputable def biased_die_probability_even_odd (sum_is_odd: ℝ) : Prop :=
  ∃ (p : ℝ), (1 - 3*p = 0) ∧ (2*p = 1) ∧
  let p_odd := p in
  let p_even := 2 * p in
  (2 * p * p_odd) + (p_odd * p_even) = sum_is_odd

theorem biased_die_odd_sum_probability :
  ∃ (p : ℝ), biased_die_probability_even_odd (4 / 9) :=
by
  sorry

end biased_die_odd_sum_probability_l792_792669


namespace proof_problem_l792_792491

variable (f : ℝ → ℝ)

def isPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f(x) = f(x + p)

def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = -f(-x)

theorem proof_problem (h_period : isPeriodic f 4) (h_odd : isOdd f) (h_value : f (-5) = 1) : f 1 = -1 :=
by
  sorry

end proof_problem_l792_792491


namespace segments_of_right_triangle_l792_792715

-- Defining the specific sides of the right triangle
def a : ℝ := 6
def b : ℝ := 8
def c : ℝ := 10 -- hypotenuse
def r : ℝ := 2 -- radius of the inscribed circle

-- Define proofs for the calculated segments
theorem segments_of_right_triangle :
  ∀ (MP EQ DN : ℝ), 
  MP = (3 / 2) ∧
  EQ = (8 / 3) ∧
  DN = (25 / 6) :=
by
  let S := (1 / 2) * a * b
  let p := (a + b + c) / 2
  have h1 : r = S / p := by sorry
  have h2 : MP = (a * r) / b := by sorry
  have h3 : EQ = (b * r) / a := by sorry
  have h4 : DN = (MP + EQ) := by sorry
  exact ⟨h2, h3, h4⟩

end segments_of_right_triangle_l792_792715


namespace one_third_of_product_is_21_l792_792362

def one_third_of_product_of_7_and_9 : ℕ :=
  let product := 7 * 9
  in product / 3

theorem one_third_of_product_is_21 : one_third_of_product_of_7_and_9 = 21 := by
  sorry

end one_third_of_product_is_21_l792_792362


namespace lengthQRIndependentOfP_l792_792967

-- Definitions corresponding to the conditions
variables {A B C P M Q R : Point}
variable {Γ : Circle}

-- Condition 1: M is the midpoint of BC
def isMidpoint (M B C : Point) : Prop :=
  dist B M = dist M C

-- Condition 2: P is a variable interior point of the triangle ⊿ABC such that ∠CPM = ∠PAB
def angleEquality (P A B C M : Point) : Prop :=
  ∠CPM = ∠PAB

-- Condition 3, 4, 5 are implicit in the definitions of the objects involved.

noncomputable def lengthQRIndependent (A B C M P Q R : Point) (Γ : Circle) : Prop :=
  isMidpoint M B C ∧
  angleEquality P A B C M ∧
  (onCircumcircle Γ A B P) ∧
  (lineIntersectCircumcircleTwice M P Q Γ) ∧
  (reflectionOf P B R Γ) → 
  dist Q R = dist B C

-- The main statement to be proved
theorem lengthQRIndependentOfP (A B C M P Q R : Point) (Γ : Circle) :
  lengthQRIndependent A B C M P Q R Γ :=
sorry

end lengthQRIndependentOfP_l792_792967


namespace no_identical_graphs_l792_792673

theorem no_identical_graphs : 
  (∀ x, y₁ = x + 3 → 
  y₂ = (x ≠ 3 → (x^2 - 9) / (x - 3) | 0) →
  (∀ x, (x - 3) * y₃ = x^2 - 9 → y₃ = if x ≠ 3 then (x^2 - 9) / (x - 3) else y) → 
  (∀ x ≠ 3, y₁ = y₂) ∧ (∀ y, x = 3 → y₁ ≠ y₃) ∧ (∀ y, x ≠ 3 → y₃ = y₂)) :=
by
  sorry

end no_identical_graphs_l792_792673


namespace simplify_expression_l792_792760

theorem simplify_expression : 1 + 3 / (2 + 5 / 6) = 35 / 17 := 
  sorry

end simplify_expression_l792_792760


namespace cyclic_quad_eq_l792_792539

theorem cyclic_quad_eq :
  ∀ (A B C D E N I : Point), 
  cyclic_quadrilateral A B C D →
  AD = BD →
  intersects_at AC BD E →
  is_incenter I (triangle B C E) →
  circumcircle_intersects B I E A E N →
  AN * NC = CD * BN :=
by
  intros
  sorry

end cyclic_quad_eq_l792_792539


namespace distribution_plans_l792_792762

-- Defining the parameters
def num_teachers := 4
def num_schools := 3
def min_teachers_per_school := 1

-- The hypothesis that each school gets at least one teacher
theorem distribution_plans (h1 : num_teachers = 4) (h2 : num_schools = 3) (h3 : min_teachers_per_school = 1) :
  (number_of_distribution_plans num_teachers num_schools min_teachers_per_school) = 36 :=
sorry

end distribution_plans_l792_792762


namespace Tim_sweets_are_multiple_of_4_l792_792643

-- Define the conditions
def sweets_are_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Given definitions
def Peter_sweets : ℕ := 44
def largest_possible_number_per_tray : ℕ := 4

-- Define the proposition to be proven
theorem Tim_sweets_are_multiple_of_4 (O : ℕ) (h1 : sweets_are_divisible_by_4 Peter_sweets) (h2 : sweets_are_divisible_by_4 largest_possible_number_per_tray) :
  sweets_are_divisible_by_4 O :=
sorry

end Tim_sweets_are_multiple_of_4_l792_792643


namespace james_out_of_pocket_l792_792141

def consultation_cost := 300
def consultation_coverage := 0.83
def xray_cost := 150
def xray_coverage := 0.74
def prescription_cost := 75
def prescription_coverage := 0.55
def therapy_cost := 120
def therapy_coverage := 0.62
def equipment_cost := 85
def equipment_coverage := 0.49
def followup_cost := 200
def followup_coverage := 0.75

def out_of_pocket (cost : ℕ) (coverage : ℝ) : ℝ :=
  cost - (cost * coverage)

def total_out_of_pocket :=
  out_of_pocket consultation_cost consultation_coverage +
  out_of_pocket xray_cost xray_coverage +
  out_of_pocket prescription_cost prescription_coverage +
  out_of_pocket therapy_cost therapy_coverage +
  out_of_pocket equipment_cost equipment_coverage +
  out_of_pocket followup_cost followup_coverage

theorem james_out_of_pocket : total_out_of_pocket = 262.70 := by
  sorry

end james_out_of_pocket_l792_792141


namespace circles_symmetric_sin_cos_l792_792923

noncomputable def sin_cos_product (θ : Real) : Real := Real.sin θ * Real.cos θ

theorem circles_symmetric_sin_cos (a θ : Real) 
(h1 : ∃ x1 y1, x1 = -a / 2 ∧ y1 = 0 ∧ 2*x1 - y1 - 1 = 0) 
(h2 : ∃ x2 y2, x2 = -a ∧ y2 = -Real.tan θ / 2 ∧ 2*x2 - y2 - 1 = 0) :
sin_cos_product θ = -2 / 5 := 
sorry

end circles_symmetric_sin_cos_l792_792923


namespace num_children_receive_ice_cream_l792_792701

-- Define the volumes of the cylinder and cone+hemisphere based on given conditions.
def V_cylinder : ℝ := Real.pi * (6 : ℝ)^2 * (15 : ℝ)
def V_cone (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h
def V_hemisphere (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

-- The radius and height for the cone
def r_ice_cream_cone : ℝ := 3
def h_conical_portion : ℝ := 4 * r_ice_cream_cone

-- Total volume of one ice cream cone
def V_total_cone := V_cone r_ice_cream_cone h_conical_portion + V_hemisphere r_ice_cream_cone

-- The final proof statement
theorem num_children_receive_ice_cream : V_cylinder / V_total_cone = 10 := by
  sorry

end num_children_receive_ice_cream_l792_792701


namespace exists_permutation_32_exists_permutation_100_l792_792369

def avg_not_between (s : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → (∀ m, m > i ∧ m < j → 2 * s[m] ≠ s[i] + s[j])

theorem exists_permutation_32 :
  ∃ s : List ℕ, s.perm (List.range 32) ∧ avg_not_between s :=
sorry

theorem exists_permutation_100 :
  ∃ s : List ℕ, s.perm (List.range 100) ∧ avg_not_between s :=
sorry

end exists_permutation_32_exists_permutation_100_l792_792369


namespace given_seashells_l792_792378

theorem given_seashells (original left given : ℝ) (h1 : original = 62.5) (h2 : left = 30.75) (h3 : given = original - left) : given = 31.75 :=
by {
  rw [h1, h2] at h3,
  exact h3,
}

end given_seashells_l792_792378


namespace y_relationship_l792_792005

variable (a c : ℝ) (h_a : a < 0)

def f (x : ℝ) : ℝ := a * (x - 3) ^ 2 + c

theorem y_relationship (y1 y2 y3 : ℝ)
  (h1 : y1 = f a c (Real.sqrt 5))
  (h2 : y2 = f a c 0)
  (h3 : y3 = f a c 4) :
  y2 < y3 ∧ y3 < y1 :=
  sorry

end y_relationship_l792_792005


namespace equivalent_conditions_l792_792546

theorem equivalent_conditions (α β : ℝ) (h_pos_α : 0 < α) (h_pos_β : 0 < β) : 
  (∀ n : ℕ, ∃! (k : ℕ × ℕ), n = ⌊(k.1 : ℝ) * α⌋ ∨ n = ⌊(k.2 : ℝ) * β⌋) ↔ 
  (1 / α + 1 / β = 1 ∧ irrational α ∧ irrational β) := 
sorry

end equivalent_conditions_l792_792546


namespace tangent_of_11pi_over_4_l792_792781

theorem tangent_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 :=
sorry

end tangent_of_11pi_over_4_l792_792781


namespace minimum_value_of_a_plus_5b_l792_792030

theorem minimum_value_of_a_plus_5b :
  ∀ (a b : ℝ), a > 0 → b > 0 → (1 / a + 5 / b = 1) → a + 5 * b ≥ 36 :=
by
  sorry

end minimum_value_of_a_plus_5b_l792_792030


namespace largest_B_181_l792_792771

noncomputable def binom (n k : ℕ) : ℚ := Nat.choose n k
def B (n k : ℕ) (p : ℚ) := binom n k * p^k

theorem largest_B_181 : ∃ k, B 2000 181 (1 / 10) = arg_max k (B 2000 k (1 / 10)) where
  arg_max (k : ℕ) (f : ℕ → ℚ) := k ≤ 2000 ∧ ∀ j, j ≤ 2000 → f j ≤ f k := sorry

end largest_B_181_l792_792771


namespace scientific_notation_l792_792734

theorem scientific_notation : (0.000000005 : ℝ) = 5 * 10^(-9 : ℤ) := 
by
  sorry

end scientific_notation_l792_792734


namespace Petya_wins_optimally_l792_792255

-- Defining the game state and rules
inductive GameState
| PetyaWin
| VasyaWin

-- Rules of the game
def game_rule (n : ℕ) : Prop :=
  n > 0 ∧ (n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2)

-- Determine the winner given the initial number of minuses
def determine_winner (n : ℕ) : GameState :=
  if n % 3 = 0 then GameState.PetyaWin else GameState.VasyaWin

-- Theorem: Petya will win the game if both play optimally
theorem Petya_wins_optimally (n : ℕ) (h1 : n = 2021) (h2 : game_rule n) : determine_winner n = GameState.PetyaWin :=
by {
  sorry
}

end Petya_wins_optimally_l792_792255


namespace find_maximize_area_triangle_OAB_l792_792114

open Real

noncomputable def maximize_area_triangle_OAB (θ : ℝ) (A B: ℝ × ℝ) : Prop :=
  θ ∈ Ioo (π/4) (π/2) ∧
  (A.2 = tan θ * A.1) ∧
  (dist (0,0) A = 1 / (sqrt 2 - cos θ)) ∧
  (B.1^2 - B.2^2 = 1) ∧
  (∃ θ_max : ℝ, θ_max = acos (sqrt 2 / 4) ∧ θ = θ_max) ∧
  (∃ max_area : ℝ, max_area = sqrt 6 / 6 ∧ 
                   1 / 2 * dist (0,0) A * (sqrt (tan θ^2 - 1) / sqrt (tan θ^2 + 1)) = max_area)
                   
theorem find_maximize_area_triangle_OAB : ∃ θ, ∃ A B, maximize_area_triangle_OAB θ A B :=
sorry

end find_maximize_area_triangle_OAB_l792_792114


namespace y_relationship_l792_792006

variable (a c : ℝ) (h_a : a < 0)

def f (x : ℝ) : ℝ := a * (x - 3) ^ 2 + c

theorem y_relationship (y1 y2 y3 : ℝ)
  (h1 : y1 = f a c (Real.sqrt 5))
  (h2 : y2 = f a c 0)
  (h3 : y3 = f a c 4) :
  y2 < y3 ∧ y3 < y1 :=
  sorry

end y_relationship_l792_792006


namespace trailing_zeros_l792_792488

theorem trailing_zeros (a b : ℕ) 
  (h1 : a = 4^(5^6))
  (h2 : b = 6^(5^4)) :
  Nat.trailingZeros (a + b) = 5 := 
  sorry

end trailing_zeros_l792_792488


namespace total_surface_area_of_cone_l792_792705

theorem total_surface_area_of_cone (r l : ℝ) (sector_fraction : ℝ)
  (h : ℝ) (base_radius : ℝ) (cone_radius : ℝ) (lateral_area : ℝ) (base_area : ℝ) :
  r = 6 → sector_fraction = 0.5 → l = r → 
  base_radius = sector_fraction * r * 2 * real.pi → 
  cone_radius = base_radius / (2 * real.pi) → 
  base_radius = 6 * real.pi → 
  cone_radius = 3 → 
  h^2 = r^2 - cone_radius^2 → 
  h = 3 * real.sqrt 3 → 
  lateral_area = real.pi * cone_radius * l → 
  lateral_area = 18 * real.pi → 
  base_area = real.pi * cone_radius^2 → 
  base_area = 9 * real.pi → 
  lateral_area + base_area = 27 * real.pi → 
  lateral_area + base_area = 27 * real.pi := begin
  sorry
end

end total_surface_area_of_cone_l792_792705


namespace bread_piece_weights_l792_792575

def slice_weight : ℕ → ℚ := λ n, 50 / n

def first_slice_half_weight := slice_weight 2

def first_slice_half_third_weight := first_slice_half_weight / 3

def first_slice_half_quarter_weight := first_slice_half_weight / 4

def second_slice_third_weight := slice_weight 3

def second_slice_third_half_weight := second_slice_third_weight / 2

def second_slice_remaining_twothirds_weight := second_slice_third_weight * 2 

def second_slice_remaining_twothirds_fifth_weight := second_slice_remaining_twothirds_weight / 5

theorem bread_piece_weights :
  let weight_833 := 50 / 6
  let weight_625 := 50 / 8
  let weight_667 := 50 / 7.5
  (5 * weight_833 + 4 * weight_625 + 10 * weight_667) ≈ 100 :=
by sorry

end bread_piece_weights_l792_792575


namespace percentage_of_males_l792_792123

noncomputable def total_employees : ℝ := 1800
noncomputable def males_below_50_years_old : ℝ := 756
noncomputable def percentage_below_50 : ℝ := 0.70

theorem percentage_of_males : (males_below_50_years_old / percentage_below_50 / total_employees) * 100 = 60 :=
by
  sorry

end percentage_of_males_l792_792123


namespace rob_total_amount_is_correct_l792_792588

/-- 
Rob has seven quarters, three dimes, five nickels, twelve pennies, and three half-dollars.
He loses one coin from each type and also decides to exchange three nickels with two dimes.
After that, he takes out a half-dollar and exchanges it for a quarter and two dimes.
Prove that the total amount of money Rob has now is $3.01.
-/

theorem rob_total_amount_is_correct :
  let initial_quarters := 7 * 0.25,
      initial_dimes := 3 * 0.10,
      initial_nickels := 5 * 0.05,
      initial_pennies := 12 * 0.01,
      initial_half_dollars := 3 * 0.50,
      after_losing_coins_quarters := (7 - 1) * 0.25,
      after_losing_coins_dimes := (3 - 1) * 0.10,
      after_losing_coins_nickels := (5 - 1) * 0.05,
      after_losing_coins_pennies := (12 - 1) * 0.01,
      after_losing_coins_half_dollars := (3 - 1) * 0.50,
      exchanged_nickels := 1 * 0.05,
      exchanged_dimes := 4 * 0.10,
      final_half_dollars := 1 * 0.50,
      final_quarters := (6 + 1) * 0.25,
      final_dimes := (2 + 4) * 0.10
  in final_quarters + final_dimes + exchanged_nickels + after_losing_coins_pennies + final_half_dollars = 3.01 := 
by
  sorry

end rob_total_amount_is_correct_l792_792588


namespace part1_part2_l792_792838

noncomputable theory

open_locale big_operators

-- Definitions for the sequence {a_n}
def seq_a : ℕ → ℕ
| 1       := 4
| (n + 1) := n * (seq_a n + 2 * n + 1) / (n + 1)

-- Question 1: Proving {a_n / n} forms an arithmetic progression
theorem part1 : ∀ n ≥ 2, (n - 1) * (seq_a n) = n * (seq_a (n - 1) + 2 * n - 2) := sorry

-- Now we define seq_b in terms of seq_a
def seq_b (n : ℕ) : ℝ := (2 * n + 1) / (seq_a n)^2

-- Finding the sum of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℝ := ∑ i in finset.range n, seq_b (i + 1)

theorem part2 : ∀ n, S n = (n^2 + 2 * n) / (4 * (n + 1)^2) := sorry

end part1_part2_l792_792838


namespace intersection_of_sets_l792_792084

theorem intersection_of_sets :
  let M := { x : ℝ | 0 ≤ x ∧ x < 16 }
  let N := { x : ℝ | x ≥ 1/3 }
  M ∩ N = { x : ℝ | 1/3 ≤ x ∧ x < 16 } :=
by
  sorry

end intersection_of_sets_l792_792084


namespace minimum_value_quadratic_function_l792_792279

-- Defining the quadratic function y
def quadratic_function (x : ℝ) : ℝ := 4 * x^2 + 8 * x + 16

-- Statement asserting the minimum value of the quadratic function
theorem minimum_value_quadratic_function : ∃ (y_min : ℝ), (∀ x : ℝ, quadratic_function x ≥ y_min) ∧ y_min = 12 :=
by
  -- Here we would normally insert the proof, but we skip it with sorry
  sorry

end minimum_value_quadratic_function_l792_792279


namespace smallest_class_size_l792_792939

theorem smallest_class_size
(FiveScores : ∀ i ∈ (Finset.range 5), (λ i => 100)) -- Condition 1
(AtLeastSixtyFive : ∀ i ∈ (Finset.range n), (λ i => i >= 65)) -- Condition 2
(MeanScore : (∑ i in (Finset.range 5), 100 + ∑ j in (Finset.range (n - 5)), j ) / n = 80) -- Condition 3
: n = 12 := by
  sorry

end smallest_class_size_l792_792939


namespace intersect_at_one_point_l792_792750

def y1 (x : ℝ) := 3 * log x
def y2 (x : ℝ) := log (x^2) + 1

theorem intersect_at_one_point : ∃! x : ℝ, y1 x = y2 x :=
by
  sorry

end intersect_at_one_point_l792_792750


namespace area_of_triangle_AEB_l792_792955

theorem area_of_triangle_AEB (A B C D F G E : Type)
  [rect : IsRectangle A B C D]
  [AB : segment A B = 8]
  [BC : segment B C = 4]
  [DF : segment D F = 2]
  [GC : segment G C = 1]
  [AF_intersects_B_G_at_E : intersects (line A F) (line B G) (point E)]
  : area (triangle A E B) = 16 :=
sorry

end area_of_triangle_AEB_l792_792955


namespace Alex_hula_hoop_duration_l792_792191

-- Definitions based on conditions
def Nancy_duration := 10
def Casey_duration := Nancy_duration - 3
def Morgan_duration := Casey_duration * 3
def Alex_duration := Casey_duration + Morgan_duration - 2

-- The theorem we need to prove
theorem Alex_hula_hoop_duration : Alex_duration = 26 := by
  -- proof to be provided
  sorry

end Alex_hula_hoop_duration_l792_792191


namespace number_of_routes_to_spell_contest_l792_792735

theorem number_of_routes_to_spell_contest :
  -- Given the conditions:
  -- * Letters must be connected by horizontal or vertical segments.
  -- * The word to be spelled is "CONTEST."
  (let n := 7 in (2^n - 1) = 127) :=
by
  sorry 

end number_of_routes_to_spell_contest_l792_792735


namespace runner_a_beats_runner_b_at_each_hurdle_l792_792106

theorem runner_a_beats_runner_b_at_each_hurdle : 
  let race_distance := 120 
  let hurdle_distance := 20
  let time_a := 36
  let time_b := 45
  let number_of_hurdles := race_distance / hurdle_distance
  let total_time_difference := time_b - time_a
  let time_difference_per_hurdle := total_time_difference / number_of_hurdles
  time_difference_per_hurdle = 1.5 :=
by 
  -- Define the conditions
  let race_distance := 120
  let hurdle_distance := 20
  let time_a := 36
  let time_b := 45
  let number_of_hurdles := race_distance / hurdle_distance
  let total_time_difference := time_b - time_a
  let time_difference_per_hurdle := total_time_difference / number_of_hurdles
  -- Assert the result
  show time_difference_per_hurdle = 1.5, by sorry

end runner_a_beats_runner_b_at_each_hurdle_l792_792106


namespace hexagon_area_l792_792968

open Complex EuclideanGeometry

-- Definitions for sides of triangle ABC
def AB := 13
def BC := 14
def CA := 15

-- Main theorem statement
theorem hexagon_area : 
  ∃ (A B C : ℂ), 
    abs (A - B) = AB ∧ 
    abs (B - C) = BC ∧ 
    abs (C - A) = CA ∧ 
    ∃ (A_5 A_6 B_5 B_6 C_5 C_6 : ℂ), 
      hexagon_constructed_correctly A B C A_5 A_6 B_5 B_6 C_5 C_6 →
      area (hexagon A_5 A_6 B_5 B_6 C_5 C_6) = 19444 := 
sorry

end hexagon_area_l792_792968


namespace remainder_7325_mod_11_l792_792662

theorem remainder_7325_mod_11 : 7325 % 11 = 6 := sorry

end remainder_7325_mod_11_l792_792662


namespace find_power_y_l792_792815

theorem find_power_y 
  (y : ℕ) 
  (h : (12 : ℝ)^y * (6 : ℝ)^3 / (432 : ℝ) = 72) : 
  y = 2 :=
by
  sorry

end find_power_y_l792_792815


namespace find_x_squared_plus_y_squared_l792_792010

-- Defining the sample set and the conditions
def sample_set : Set ℝ := {8, 9, 10, x, y}

-- Condition 1: Mean of the sample is 9
def mean_condition (x y : ℝ) : Prop :=
  (1 / 5) * (8 + 9 + 10 + x + y) = 9

-- Condition 2: Variance of the sample is 2
def variance_condition (x y : ℝ) : Prop :=
  (1 / 5) * ((8 - 9)^2 + (9 - 9)^2 + (10 - 9)^2 + (x - 9)^2 + (y - 9)^2) = 2

-- The theorem to prove
theorem find_x_squared_plus_y_squared (x y : ℝ) (mean_cond : mean_condition x y) (var_cond : variance_condition x y) : x^2 + y^2 = 170 :=
by
  sorry

end find_x_squared_plus_y_squared_l792_792010


namespace dandelion_puffs_fraction_l792_792367

theorem dandelion_puffs_fraction :
  let total_picked := 85
  let given_away := 7 + 9 + 11 + 5 + 4
  let remaining := total_picked - given_away
  let friends := 5
  let fraction_per_friend := remaining / friends
  let simplified_fraction := (fraction_per_friend : ℝ) / remaining
  simplified_fraction = 1 / 5 :=
by {
  let total_picked := 85
  let given_away := 7 + 9 + 11 + 5 + 4
  let remaining := total_picked - given_away
  let friends := 5
  let fraction_per_friend := remaining / friends
  let simplified_fraction := (fraction_per_friend : ℝ) / remaining
  have h : given_away = 36 := by rfl
  have h_remaining : remaining = 49 := by rfl
  have h_fraction_per_friend : fraction_per_friend = 9.8 := by rfl
  have h_simplified_fraction : simplified_fraction = 1 / 5 := by {
    rw [fraction_per_friend, h_remaining, ←div_div],
    norm_num,
  }
  exact h_simplified_fraction
}

end dandelion_puffs_fraction_l792_792367


namespace problem1_problem2_problem3_problem4_l792_792688

-- Problem 1
theorem problem1 : -9 + 5 - 11 + 16 = 1 :=
by
  sorry

-- Problem 2
theorem problem2 : -9 + 5 - (-6) - 18 / (-3) = 8 :=
by
  sorry

-- Problem 3
theorem problem3 : -2^2 - ((-3) * (-4 / 3) - (-2)^3) = -16 :=
by
  sorry

-- Problem 4
theorem problem4 : (59 - (7 / 9 - 11 / 12 + 1 / 6) * (-6)^2) / (-7)^2 = 58 / 49 :=
by
  sorry

end problem1_problem2_problem3_problem4_l792_792688


namespace parking_spots_full_iff_num_sequences_l792_792359

noncomputable def num_parking_sequences (n : ℕ) : ℕ :=
  (n + 1) ^ (n - 1)

-- Statement of the theorem
theorem parking_spots_full_iff_num_sequences (n : ℕ) :
  ∀ (a : ℕ → ℕ), (∀ (i : ℕ), i < n → a i ≤ n) → 
  (∀ (j : ℕ), j ≤ n → (∃ i, i < n ∧ a i = j)) ↔ 
  num_parking_sequences n = (n + 1) ^ (n - 1) :=
sorry

end parking_spots_full_iff_num_sequences_l792_792359


namespace greening_problem_proof_l792_792507

noncomputable theory

open Real -- Use real numbers from the Lean library

def team_B_daily_area (x : ℝ) : Prop :=
  x = 50

def team_A_daily_area (xA xB: ℝ) : Prop :=
  xA = 1.8 * xB

def time_difference_condition (xB : ℝ) : Prop :=
  450 / xB - 4 = 450 / (1.8 * xB)

def total_area_condition (a b : ℝ) : Prop :=
  90 * a + 50 * b = 3600

def construction_period_condition (a b : ℝ) : Prop :=
  a + b ≤ 48

def total_cost (a b : ℝ) : ℝ :=
  1.05 * a + 0.5 * b

theorem greening_problem_proof :
  ∃ a b : ℝ,
    team_B_daily_area 50 ∧
    team_A_daily_area 90 50 ∧
    time_difference_condition 50 ∧ 
    total_area_condition a b ∧
    construction_period_condition a b ∧
    total_cost a b = 40.5 :=
begin
  sorry
end

end greening_problem_proof_l792_792507


namespace range_m_false_proposition_l792_792882

theorem range_m_false_proposition (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) → (m > 1) :=
by
  intro h
  have : ∀ x : ℝ, x^2 + 2*x + m > 0, from 
    λ x, not_le.mp (λ h₁, h ⟨x, h₁⟩)
  sorry

end range_m_false_proposition_l792_792882


namespace angle_c_in_triangle_l792_792514

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A/B = 1/3) (h3 : A/C = 1/5) : C = 100 :=
by
  sorry

end angle_c_in_triangle_l792_792514


namespace marks_physics_eq_l792_792755

open Real

-- Definitions from conditions
def marks_english := 90
def marks_math := 92
def marks_chemistry := 87
def marks_biology := 85
def average_marks := 87.8
def num_subjects := 5

-- Goal: prove the marks in Physics
theorem marks_physics_eq :
  let total_marks := average_marks * (num_subjects:ℝ)
  let known_marks := (marks_english + marks_math + marks_chemistry + marks_biology:ℝ)
  total_marks - known_marks = 85 := by
  sorry

end marks_physics_eq_l792_792755


namespace angle_between_a_b_is_pi_over_2_l792_792051

open Real EuclideanSpace

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 2)) (h₁ : b ∈ ((EuclideanSpace ℝ (Fin 2))).basis) : ℝ :=
if h : b ≠ 0 then
  real.arcsin ((a ⬝ b) / (∥a∥ * ∥b∥))
else
  0

theorem angle_between_a_b_is_pi_over_2
  (a b : EuclideanSpace ℝ (Fin 2))
  (h1 : b ⬝ (a + b) = 1)
  (h2 : ∥b∥ = 1) : 
  angle_between_vectors a b = π / 2 :=
by
  unfold angle_between_vectors
  sorry

end angle_between_a_b_is_pi_over_2_l792_792051


namespace distance_between_home_and_shangri_la_l792_792185

-- Defining the main variables and conditions
variables (D : ℝ)

-- Condition 1: Distance traveled in the first hour
def distance_first_hour := (1 / 3) * D

-- Condition 2: Distance traveled in the second hour
def remaining_after_first_hour := D - distance_first_hour
def distance_second_hour := (1 / 2) * remaining_after_first_hour

-- Condition 3: Distance traveled in the third hour
def distance_third_hour := distance_first_hour - (1 / 10) * D

-- Condition 4: Total remaining distance after the third hour
def remaining_distance := D - (distance_first_hour + distance_second_hour + distance_third_hour)

-- Given condition: Remaining distance is 9 kilometers
axiom remaining_distance_is_9 : remaining_distance D = 9

-- The theorem to prove
theorem distance_between_home_and_shangri_la : D = 90 :=
by
  -- Proof would go here
  sorry

end distance_between_home_and_shangri_la_l792_792185


namespace factor_is_given_sum_l792_792921

theorem factor_is_given_sum (P Q : ℤ)
  (h1 : ∀ x : ℝ, (x^2 + 3 * x + 7) * (x^2 + (-3) * x + 7) = x^4 + P * x^2 + Q) :
  P + Q = 54 := 
sorry

end factor_is_given_sum_l792_792921


namespace tony_squat_weight_l792_792648

-- Definitions from conditions
def curl_weight := 90
def military_press_weight := 2 * curl_weight
def squat_weight := 5 * military_press_weight

-- Theorem statement
theorem tony_squat_weight : squat_weight = 900 := by
  sorry

end tony_squat_weight_l792_792648


namespace product_w_z_l792_792943

variables {EF FG GH HE : ℝ} {w z : ℝ}

def parallelogram_problem :=
  EF = 50 ∧ FG = 4 * z^2 ∧ GH = 3 * w + 6 ∧ HE = 32 ∧ EF = GH ∧ FG = HE

theorem product_w_z (h : parallelogram_problem) : w * z = 88 * Real.sqrt 2 / 3 := 
by
  cases h with hef hfg
  cases hfg with hgh heh
  cases heh with he hfgh
  sorry

end product_w_z_l792_792943


namespace equation_roots_properties_l792_792640

theorem equation_roots_properties (n : ℕ) 
  (a : fin n → ℝ) 
  (b : fin n → ℝ) 
  (c : ℝ)
  (h1 : ∀ i, 0 < b i)
  (h2 : strict_mono a) 
  :
  ∃ (x : fin (n-1) → ℝ),
  ( ∀ i : fin (n-1), a i ≤ x i ∧ x i < a (i.succ) ) ∧ 
  x (n-2) ≤ a (n-1) := 
sorry

end equation_roots_properties_l792_792640


namespace irrational_numbers_condition_l792_792816

noncomputable def irrational_solutions (x : ℝ) : Prop :=
  let cond1 := x^3 - 6 * x ∈ ℚ
  let cond2 := x^4 - 8 * x^2 ∈ ℚ
  let solutions := (x = Real.sqrt 6 ∨ x = -Real.sqrt 6 ∨
                    x = 1 + Real.sqrt 3 ∨ x = - (1 + Real.sqrt 3) ∨
                    x = 1 - Real.sqrt 3 ∨ x = - (1 - Real.sqrt 3))
  cond1 ∧ cond2 → solutions

theorem irrational_numbers_condition :
  ∀ x : ℝ, irrational_solutions x :=
begin
  sorry
end

end irrational_numbers_condition_l792_792816


namespace tangent_of_11pi_over_4_l792_792783

theorem tangent_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 :=
sorry

end tangent_of_11pi_over_4_l792_792783


namespace polynomial_div_remainder_l792_792661

open Polynomial

noncomputable def p : Polynomial ℤ := 2 * X^4 + 10 * X^3 - 45 * X^2 - 52 * X + 63
noncomputable def d : Polynomial ℤ := X^2 + 6 * X - 7
noncomputable def r : Polynomial ℤ := 48 * X - 70

theorem polynomial_div_remainder : p % d = r :=
sorry

end polynomial_div_remainder_l792_792661


namespace common_tangents_l792_792495

theorem common_tangents {C₁ C₂ : ℝ → ℝ → Prop}
  (hC₁ : ∀ x y, C₁ x y ↔ (x - 2)^2 + (y - 2)^2 = 1)
  (hC₂ : ∀ x y, C₂ x y ↔ (x - 2)^2 + (y - 5)^2 = 16) :
  ∃! t, tangents C₁ C₂ t :=
sorry

def tangents (C₁ C₂ : ℝ → ℝ → Prop) : ℕ → Prop :=
λ t, t = 1

end common_tangents_l792_792495


namespace range_of_m_l792_792467

theorem range_of_m (m : ℝ) :
  (¬(∀ x y : ℝ, x^2 / (25 - m) + y^2 / (m - 7) = 1 → 25 - m > 0 ∧ m - 7 > 0 ∧ 25 - m > m - 7) ∨ 
   ¬(∀ x y : ℝ, y^2 / 5 - x^2 / m = 1 → 1 < (5 + m) / 5 ∧ (5 + m) / 5 < 4)) 
  → 7 < m ∧ m < 15 :=
by
  sorry

end range_of_m_l792_792467


namespace mod_remainder_l792_792812

theorem mod_remainder (a b c : ℕ) : 
  (7 * 10 ^ 20 + 1 ^ 20) % 11 = 8 := by
  -- Lean proof will be written here
  sorry

end mod_remainder_l792_792812


namespace ratio_of_spinsters_to_cats_l792_792631

theorem ratio_of_spinsters_to_cats :
  (∀ S C : ℕ, (S : ℚ) / (C : ℚ) = 2 / 9) ↔
  (∃ S C : ℕ, S = 18 ∧ C = S + 63 ∧ (S : ℚ) / (C : ℚ) = 2 / 9) :=
sorry

end ratio_of_spinsters_to_cats_l792_792631


namespace vector_b_condition_l792_792156

def vector3 := ℝ × ℝ × ℝ

def a : vector3 := (5, -3, 6)
def c : vector3 := (-3, 2, -1)
def b : vector3 := (1, 1/2, 2.5)

def collinear (u v w : vector3) : Prop :=
  ∃ k: ℝ, u = k • v ∧ w = k • v

def norm (v : vector3) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def dot (u v : vector3) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def angle_bisector_condition (a b c : vector3) : Prop :=
  (dot a b) / (norm a * norm b) = (dot b c) / (norm b * norm c)

theorem vector_b_condition : collinear a b c ∧ angle_bisector_condition a b c :=
by
  sorry

end vector_b_condition_l792_792156


namespace polynomial_coeff_sum_eq_four_l792_792911

theorem polynomial_coeff_sum_eq_four (a a1 a2 a3 a4 a5 a6 a7 a8 : ℤ) :
  (∀ x : ℤ, (2 * x - 1)^6 * (x + 1)^2 = a * x ^ 8 + a1 * x ^ 7 + a2 * x ^ 6 + a3 * x ^ 5 + 
                      a4 * x ^ 4 + a5 * x ^ 3 + a6 * x ^ 2 + a7 * x + a8) →
  a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 4 := by
  sorry

end polynomial_coeff_sum_eq_four_l792_792911


namespace sum_base_10_to_base_4_l792_792742

theorem sum_base_10_to_base_4 : 
  let n1 := 45
  let n2 := 52
  let sum_base_10 := n1 + n2
  let sum_base_4 := Nat.digits 4 sum_base_10
  in sum_base_4 = [1, 2, 0, 1] :=
by
  let n1 := 45
  let n2 := 52
  let sum_base_10 := n1 + n2
  let sum_base_4 := Nat.digits 4 sum_base_10
  sorry

end sum_base_10_to_base_4_l792_792742


namespace proof_unit_prices_proof_profit_l792_792108

noncomputable def unit_price_A := 600
noncomputable def unit_price_B := unit_price_A + 40

def total_cost_A := 60000
def total_cost_B := 128000

def quantity_A := total_cost_A / unit_price_A
def quantity_B := total_cost_B / unit_price_B

def sell_price := 700
def discount_rate := 0.8
def discounted_price_B := sell_price * discount_rate

def remaining_B := 50
def sold_at_full_price := (quantity_A + quantity_B) - remaining_B

def revenue := sold_at_full_price * sell_price + remaining_B * discounted_price_B
def total_cost := total_cost_A + total_cost_B

def profit := revenue - total_cost

theorem proof_unit_prices : unit_price_A = 600 ∧ unit_price_B = 640 := by {
  sorry
}

theorem proof_profit : profit = 15000 := by {
  sorry
}

end proof_unit_prices_proof_profit_l792_792108


namespace Megan_not_lead_plays_l792_792568

-- Define the problem's conditions as variables
def total_plays : ℕ := 100
def lead_play_ratio : ℤ := 80

-- Define the proposition we want to prove
theorem Megan_not_lead_plays : 
  (total_plays - (total_plays * lead_play_ratio / 100)) = 20 := 
by sorry

end Megan_not_lead_plays_l792_792568


namespace common_divisors_l792_792898

theorem common_divisors (a b : ℕ) (ha : a = 9240) (hb : b = 8820) : 
  let g := Nat.gcd a b in 
  g = 420 ∧ Nat.divisors 420 = 24 :=
by
  have gcd_ab := Nat.gcd_n at ha hb
  have fact := Nat.factorize 420
  have divisors_420: ∀ k : Nat, g = 420 ∧ k = 24 := sorry
  exact divisors_420 24

end common_divisors_l792_792898


namespace tan_of_11pi_over_4_is_neg1_l792_792795

noncomputable def tan_periodic : Real := 2 * Real.pi

theorem tan_of_11pi_over_4_is_neg1 :
  Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Angle normalization using periodicity of tangent function
  have h1 : Real.tan (11 * Real.pi / 4) = Real.tan (11 * Real.pi / 4 - 2 * Real.pi) := 
    by rw [Real.tan_periodic]
  -- Further normalization
  have h2 : 11 * Real.pi / 4 - 2 * Real.pi = 3 * Real.pi / 4 := sorry
  -- Evaluate tangent at the simplified angle
  have h3 : Real.tan (3 * Real.pi / 4) = -Real.tan (Real.pi / 4) := sorry
  -- Known value of tangent at common angle
  have h4 : Real.tan (Real.pi / 4) = 1 := by simpl tan
  rw [h2, h3, h4]
  norm_num

end tan_of_11pi_over_4_is_neg1_l792_792795


namespace prime_number_divisible_by_3_plus_1_l792_792393

theorem prime_number_divisible_by_3_plus_1 (p : ℕ) (hp : p.prime) :
  (∃ (x y : ℚ) (hx : x > 0) (hy : y > 0) (n : ℕ), x + y + p / x + p / y = 3 * n) ↔ 3 ∣ (p + 1) :=
by {
  sorry
}

end prime_number_divisible_by_3_plus_1_l792_792393


namespace compound_interest_initial_sum_l792_792738

theorem compound_interest_initial_sum :
  let r := 0.06
  let A := 1348.32
  let n := 1
  let t := 2
  let factor := (1 + r / n) ^ (n * t)
  A / factor = 1200 := 
by
  let r := 0.06
  let A := 1348.32
  let n := 1
  let t := 2
  let factor := (1 + r / n) ^ (n * t)
  have h1 : factor = 1.1236 := by sorry
  have h2 : A / factor = 1348.32 / 1.1236 := by rw h1
  have h3 : 1348.32 / 1.1236 = 1200 := by sorry
  rw [h2, h3]
  rfl

end compound_interest_initial_sum_l792_792738


namespace uncle_bruce_dough_weight_l792_792270

-- Definitions based on the conditions
variable {TotalChocolate : ℕ} (h1 : TotalChocolate = 13)
variable {ChocolateLeftOver : ℕ} (h2 : ChocolateLeftOver = 4)
variable {ChocolatePercentage : ℝ} (h3 : ChocolatePercentage = 0.2) 
variable {WeightOfDough : ℝ}

-- Target statement expressing the final question and answer
theorem uncle_bruce_dough_weight 
  (h1 : TotalChocolate = 13) 
  (h2 : ChocolateLeftOver = 4) 
  (h3 : ChocolatePercentage = 0.2) : 
  WeightOfDough = 36 := by
  sorry

end uncle_bruce_dough_weight_l792_792270


namespace min_value_sym_center_l792_792427

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (π * x + π / 6) + 2

theorem min_value_sym_center :
  (∀ ω > 0, ∀ φ, |φ| < π / 2 → f 0 = 3 → ω = π ∧ φ = π / 6 ∧ ∃ k ∈ Int, (a, b) := (k - 1 / 6, 2) ∧ |a + b| = (k - 1 / 6 + 2)) →
  ∃ k ∈ Int, k = -2 → min (|k - 1 / 6 + 2|) = 1 / 6 :=
sorry

end min_value_sym_center_l792_792427


namespace angle_A_eq_angle_B_implies_rectangle_l792_792945

-- Define the quadrilateral ABCD with given properties
structure Quadrilateral (A B C D : Type) :=
  (AD_parallel_BC : Prop)
  (AB_eq_CD : Prop)

-- The conditions given in the problem
variable (A B C D : Type)

def quadrilateral_ABCD (A B C D : Type) : Quadrilateral A B C D :=
{ AD_parallel_BC := AD_parallel_BC A B C D,
  AB_eq_CD := AB_eq_CD A B C D }

-- Define the angles
variable (angle_A angle_B angle_C angle_D : ℝ)

-- The math proof statement
theorem angle_A_eq_angle_B_implies_rectangle 
  (h1 : quadrilateral_ABCD A B C D)
  (h2 : angle_A = angle_B) : 
  angle_A = 90 ∧ angle_B = 90 ∧ angle_C = 90 ∧ angle_D = 90 :=
sorry

end angle_A_eq_angle_B_implies_rectangle_l792_792945


namespace sticks_problem_solution_l792_792577

theorem sticks_problem_solution :
  ∃ n : ℕ, n > 0 ∧ 1012 = 2 * n * (n + 1) ∧ 1012 > 1000 ∧ 
           1012 % 3 = 1 ∧ 1012 % 5 = 2 :=
by
  sorry

end sticks_problem_solution_l792_792577


namespace amplitude_and_period_range_of_f_decreasing_intervals_l792_792871

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem amplitude_and_period : 
  (∀ x, f(x) ≤ 2 ∧ f(x) ≥ -2) ∧ 
  (∀ x, f(x + Real.pi) = f(x)) := by
  sorry

theorem range_of_f : {y : ℝ | ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), y = f x} = Set.Icc (-1 : ℝ) 2 :=
  by sorry

theorem decreasing_intervals : 
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
    ∀ y ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), x < y → f(y) < f(x)) ∧ 
  (∀ x ∈ Set.Icc (-5 * Real.pi / 6) (-Real.pi / 3), 
    ∀ y ∈ Set.Icc (-5 * Real.pi / 6) (-Real.pi / 3), x < y → f(y) < f(x)) := by
  sorry

end amplitude_and_period_range_of_f_decreasing_intervals_l792_792871


namespace compute_expression_l792_792373

theorem compute_expression :
  (-(-2) + (1 + real.pi)^0 - abs (1 - real.sqrt 2) + real.sqrt 8 - real.cos (real.pi / 4)) = 2 + 5 / real.sqrt 2 :=
by sorry

end compute_expression_l792_792373


namespace vector_projection_example_l792_792811

open Real

noncomputable def vector_projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let scale (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (k * v.1, k * v.2, k * v.3)
  scale (dot_product a b / dot_product b b) b

theorem vector_projection_example :
  vector_projection (4, -1, 3) (3, -2, 1) = (51/14, -17/7, 17/14) := by
  sorry

end vector_projection_example_l792_792811


namespace prod_eq_diff_squares_l792_792730

variable (a b : ℝ)

theorem prod_eq_diff_squares :
  ( (1 / 4 * a + b) * (b - 1 / 4 * a) = b^2 - (1 / 16 * a^2) ) :=
by
  sorry

end prod_eq_diff_squares_l792_792730


namespace unit_area_triangle_has_largest_square_l792_792674

noncomputable def unit_area_triangle_with_largest_square (T : Type) [is_triangle T] : Prop :=
  ∃ (base height : ℝ), triangle_area base height = 1 ∧ ∀ (s : ℝ), s ≤ largest_inscribed_square base height → (base = sqrt 2 ∧ height = sqrt 2 ∧ is_not_obtuse base height)

theorem unit_area_triangle_has_largest_square :
  ∀ (T : Type) [is_triangle T], unit_area_triangle_with_largest_square T := by
  sorry

end unit_area_triangle_has_largest_square_l792_792674


namespace closest_integer_sum_l792_792364

noncomputable def targetSum : ℚ := 500 * ∑ n in (Finset.range 9996).filter (λ n, n ≥ 5), (1 / (n + 5 : ℚ)^2 - 16)

theorem closest_integer_sum :
  Int.closestInteger (targetSum.toReal) = 130 :=
sorry

end closest_integer_sum_l792_792364


namespace solve_inequality_l792_792213

theorem solve_inequality (x : ℝ) :
  2 * sqrt((4 * x - 9)^2) + 
  (sqrt (sqrt((3 * x^2) + 6 * x + 7) + sqrt((5 * x^2) + 10 * x + 14) + x^2 + 2 * x - 4))^(1 / 4) 
  ≤ 18 - 8 * x ↔ x = -1 :=
by
  -- Mathematical proof is omitted here
  sorry

end solve_inequality_l792_792213


namespace part1_part2_l792_792048

def P (x : ℝ) : Prop := |x - 1| > 2
def S (x : ℝ) (a : ℝ) : Prop := x^2 - (a + 1) * x + a > 0

theorem part1 (a : ℝ) (h : a = 2) : ∀ x, S x a ↔ x < 1 ∨ x > 2 :=
by
  sorry

theorem part2 (a : ℝ) (h : a ≠ 1) : ∀ x, (P x → S x a) → (-1 ≤ a ∧ a < 1) ∨ (1 < a ∧ a ≤ 3) :=
by
  sorry

end part1_part2_l792_792048


namespace num_elements_T_l792_792987

noncomputable def g (x : ℝ) : ℝ := (x + 5) / x

def g_seq : ℕ → (ℝ → ℝ)
| 0       := g
| (n + 1) := g ∘ g_seq n

def T : Set ℝ := { x | ∃ n > 0, g_seq n x = x }

theorem num_elements_T : Set.card T = 2 := sorry

end num_elements_T_l792_792987


namespace paint_coverage_area_l792_792924

variable (paint_cost_per_quart : ℝ) (paint_total_cost : ℝ) (cube_edge : ℝ)

theorem paint_coverage_area :
  paint_cost_per_quart = 3.20 →
  paint_total_cost = 1.60 →
  cube_edge = 10 →
  (6 * (cube_edge^2) / (paint_total_cost / paint_cost_per_quart)) = 1200 :=
by
  intros h1 h2 h3
  have h4 : 6 * (cube_edge ^ 2) = 6 * (10 ^ 2) := by { rw h3, norm_num }
  have h5 : 10 ^ 2 = 100 := by norm_num
  rw [h5] at h4
  have h6 : 6 * 100 = 600 := by norm_num
  rw [h6] at h4
  have h7 : paint_total_cost / paint_cost_per_quart = 1.60 / 3.20 := by { rw [h2, h1] }
  have h8 : 1.60 / 3.20 = 0.5 := by norm_num
  rw [h8] at h7
  have h9 : 600 / 0.5 = 1200 := by norm_num
  rw [h9]
  assumption

end paint_coverage_area_l792_792924


namespace num_valid_C_l792_792410

open Nat

def is_digit (C : ℕ) : Prop := C ≥ 0 ∧ C ≤ 9

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

theorem num_valid_C : 
  {C : ℕ // is_digit C ∧ is_multiple_of_3 (6 + C)}.card = 4 := 
by 
  sorry

end num_valid_C_l792_792410


namespace problem_1_problem_2_l792_792047

def A (x : ℝ) : Prop := abs (x - 1) < 2
def B (x : ℝ) (a : ℝ) : Prop := x^2 + a * x - 6 < 0
def C (x : ℝ) : Prop := x^2 - 2 * x - 15 < 0

-- 1. Prove that \( A \subseteq B \) implies \( a \in [-5, -1] \)
theorem problem_1 (a : ℝ) : (∀ x, A x → B x a) ↔ a ∈ set.Icc (-5 : ℝ) (-1 : ℝ) := sorry

-- 2. Prove the existence of \( a \) such that \( A \cup B = B \cap C \)
theorem problem_2 : ∃ a : ℝ, (∀ x, (A x ∨ B x a) ↔ (B x a ∧ C x)) ∧ a ∈ set.Icc (-19 / 5 : ℝ) (-1 : ℝ) := sorry

end problem_1_problem_2_l792_792047


namespace domain_and_value_of_f_f_is_even_f_is_increasing_on_positive_l792_792867

noncomputable def f (x : ℝ) := real.log ( |x| ) / real.log 2

-- 1. Prove the domain and value of f(-sqrt(2))
theorem domain_and_value_of_f : 
  (∀ x, f x = real.log ( |x| ) / real.log 2) ∧ 
  (∀ x, x ≠ 0 → (∃ y, y = f x)) ∧ 
  (f (-real.sqrt 2) = 1/2) := sorry

-- 2. Prove that f(x) is an even function
theorem f_is_even : 
  ∀ x, f (-x) = f x := sorry

-- 3. Prove that f(x) is increasing on (0, +∞)
theorem f_is_increasing_on_positive :
  ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2 := sorry

end domain_and_value_of_f_f_is_even_f_is_increasing_on_positive_l792_792867


namespace ratio_of_square_areas_l792_792094

theorem ratio_of_square_areas (s2 : ℝ) (h : ∃ s1, s1 = 4 * s2) : 
  (∃ s1, (s1^2) / (s2^2) = 16) :=
by
  obtain ⟨s1, hs⟩ := h
  use s1
  rw hs
  have A1 := (4 * s2)^2
  have A2 := s2^2
  calc
    (A1) / (A2) = (16 * s2^2) / (s2^2) : by rw [A1, A2]
             ... = 16 : by ring 

end ratio_of_square_areas_l792_792094


namespace number_of_common_divisors_l792_792907

theorem number_of_common_divisors :
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  let divisors_count := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  gcd_ab = 420 ∧ divisors_count = 24 :=
by
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 420 := sorry
  have h2 : (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 24 := by norm_num
  exact ⟨h1, h2⟩

end number_of_common_divisors_l792_792907


namespace no_such_metric_exists_l792_792206

theorem no_such_metric_exists (C : Set (α → ℝ)) [∀ f, Continuous (f : α → ℝ)] :
  ¬ ∃ (rho : (α → ℝ) → (α → ℝ) → ℝ) [MetricSpace (α → ℝ)] [∀ f g, rho f g = 0 → f = g],
    (∀ {f seq}, (∀ x, seq x ⟶ f x) ↔ (∀ ε > 0, ∃ N, ∀ n ≥ N, rho (seq n) f < ε)) :=
sorry

end no_such_metric_exists_l792_792206


namespace monotonic_intervals_f_intersection_range_k_l792_792441

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x
def g (k : ℝ) (x : ℝ) : ℝ := k * x - 1

theorem monotonic_intervals_f 
  (a : ℝ) (h₀ : a ≠ 0) : 
  ((a > 0 ∧ (∀ x : ℝ, x > (1 / Real.exp 1) -> 0 < f' a x) ∧ (∀ x : ℝ, 0 < x -> x < (1 / Real.exp 1) -> f' a x < 0)) 
  ∨ (a < 0 ∧ (∀ x : ℝ, x > (1 / Real.exp 1) -> f' a x < 0) ∧ (∀ x : ℝ, 0 < x -> x < (1 / Real.exp 1) -> 0 < f' a x))) := sorry

theorem intersection_range_k 
  (k : ℝ) 
  (hn : ∀ x : ℝ, (x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1) → f 1 x = g k x) 
  (distinct : ∃ x1 x2, x1 ∈ Set.Icc (1 / Real.exp 1) Real.exp 1 ∧ x2 ∈ Set.Icc (1 / Real.exp 1) Real.exp 1 ∧ x1 ≠ x2) : 
  1 < k ∧ k ≤ 1 + 1 / Real.exp 1 := sorry

end monotonic_intervals_f_intersection_range_k_l792_792441


namespace kendra_sunday_shirts_l792_792534

def total_shirts := 22
def shirts_weekdays := 5 * 1
def shirts_after_school := 3
def shirts_saturday := 1

theorem kendra_sunday_shirts : 
  (total_shirts - 2 * (shirts_weekdays + shirts_after_school + shirts_saturday)) = 4 :=
by
  sorry

end kendra_sunday_shirts_l792_792534


namespace common_points_sum_l792_792833

noncomputable def f (x : ℝ) : ℝ := sorry -- Define f with given property
def g (x : ℝ) : ℝ := (4 * x + 3) / (x - 2)

theorem common_points_sum :
  (∀ x : ℝ, f (-x) = 8 - f (4 + x)) → (∃ P : Fin 168 → ℝ × ℝ, ∀ i, f (P i).fst = g (P i).fst) →
  ∑ i in finset.range 168, (P ⟨i, by linarith⟩).1 + (P ⟨i, by linarith⟩).2 = 1008 :=
begin
  intros hf hP,
  sorry
end

end common_points_sum_l792_792833


namespace find_eq_C1_find_eq_l_l792_792128

noncomputable def eq_C1 : Prop :=
  ∃ (a b x y : ℝ) (M : ℝ × ℝ),
    a > b ∧ b > 0 ∧
    (x, y) ∈ {(x, y) | y^2 = 4*x} ∧
    M = (x, y) ∧
    sqrt ((x - 1)^2 + y^2) = 5 / 3 ∧
    1 = x^2 / a^2 + y^2 / b^2 ∧
    b^2 = a^2 - 1 ∧
    a = 2

theorem find_eq_C1 : eq_C1 :=
sorry

noncomputable def eq_l : Prop :=
  ∃ (a b x y m : ℝ) (A B M F1 F2 : ℝ × ℝ),
    a > b ∧ b > 0 ∧
    (F2.1 = 1 ∧ F2.2 = 0) ∧
    M = (x, y) ∧
    x = 2 / 3 ∧ y = 2 * sqrt 6 / 3 ∧
    (x, y) ∈ {(x, y) | y^2 = 4*x} ∧
    1 = x^2 / a^2 + y^2 / b^2 ∧
    b^2 = a^2 - 1 ∧
    a = 2 ∧
    x = 2 / 3 ∧
    y = 2 * sqrt 6 / 3 ∧
    M = (2 / 3, 2 * sqrt 6 / 3) ∧
    ∃ N : ℝ × ℝ,
      N = (F1.1 + F2.1, F1.2 + F2.2) ∧
      (O : ℝ × ℝ = (0, 0)) ∧
      (9 * x^2 - 16 * m * x + 8 * m^2 - 4 = 0) ∧
      (A.1 + B.1 = 16 * m / 9 ∧ A.1 * B.1 = (8 * m^2 - 4) / 9) ∧
      (A.2 = sqrt 6 * (A.1 - m)) ∧
      (B.2 = sqrt 6 * (B.1 - m)) ∧
      (A.1 * B.1 + A.2 * B.2 = 0) ∧
      m = sqrt 2 ∨ m = - sqrt 2 ∧
      (l : ℝ → ℝ) = λ x, sqrt 6 * x - 2 * sqrt 3 ∨ (l : ℝ → ℝ) = λ x, sqrt 6 * x + 2 * sqrt 3

theorem find_eq_l : eq_l :=
sorry

end find_eq_C1_find_eq_l_l792_792128


namespace light_ray_distance_l792_792929

open Real

def point (x y : ℝ) := (x, y)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((fst p2 - fst p1) ^ 2 + (snd p2 - snd p1) ^ 2)

theorem light_ray_distance :
  let A := point (-3) 5
  let B := point 2 7
  let A' := point (-3) (-5)
  distance A' B = 13 :=
by
  let A := point (-3) 5
  let B := point 2 7
  let A' := point (-3) (-5)
  have h : distance A' B = 13
  { unfold distance
    simp [Real.sqrt_eq_rpow]
    norm_num, }
  exact h

end light_ray_distance_l792_792929


namespace common_divisors_l792_792897

theorem common_divisors (a b : ℕ) (ha : a = 9240) (hb : b = 8820) : 
  let g := Nat.gcd a b in 
  g = 420 ∧ Nat.divisors 420 = 24 :=
by
  have gcd_ab := Nat.gcd_n at ha hb
  have fact := Nat.factorize 420
  have divisors_420: ∀ k : Nat, g = 420 ∧ k = 24 := sorry
  exact divisors_420 24

end common_divisors_l792_792897


namespace ways_to_express_528_as_sum_of_consecutive_integers_l792_792942

theorem ways_to_express_528_as_sum_of_consecutive_integers :
  ∃! (s : Finset (Finset ℕ)), 2 ≤ s.card ∧ (s.sum id = 528) ∧ 
  (∀ (t : Finset ℕ), t ∈ s → (∀ a ∈ t, ∃ k, (t = Finset.range k ∧ k > 1))) :=
sorry

end ways_to_express_528_as_sum_of_consecutive_integers_l792_792942


namespace difference_between_max_and_min_coins_l792_792195

theorem difference_between_max_and_min_coins (n : ℕ) : 
  (∃ x y : ℕ, x * 10 + y * 25 = 45 ∧ x + y = n) →
  (∃ p q : ℕ, p * 10 + q * 25 = 45 ∧ p + q = n) →
  (n = 2) :=
by
  sorry

end difference_between_max_and_min_coins_l792_792195


namespace intersection_of_sets_l792_792085

theorem intersection_of_sets :
  let M := { x : ℝ | 0 ≤ x ∧ x < 16 }
  let N := { x : ℝ | x ≥ 1/3 }
  M ∩ N = { x : ℝ | 1/3 ≤ x ∧ x < 16 } :=
by
  sorry

end intersection_of_sets_l792_792085


namespace find_a_b_find_k_range_l792_792038

-- Problem 1: Find the values of a and b
theorem find_a_b (a b : ℝ) (hf : ∀ x, f x = (1/3) * x^3 + a * x^2 + 3 * x + b)
(hmax : f 1 = 2) (hderiv : f' 1 = 0) : 
  a = -2 ∧ b = 2 / 3 :=
  sorry

-- Problem 2: Find the range of k
theorem find_k_range (k : ℝ) 
(hf : ∀ x, f x = (1/3) * x^3 - 2 * x^2 + 3 * x + 2/3)
(hroots : ∀ x, f x = -x^2 + 6 * x + k → 
            ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  k ∈ set.Ioo (-(25/3):ℝ) (7/3:ℝ) :=
  sorry

end find_a_b_find_k_range_l792_792038


namespace alice_has_ball_after_three_turns_l792_792349

open ProbabilityMassFunction

/-- Alice and Bob's ball game probability problem -/
def alice_bob_game (P : Population (Fin 4)) : ProbabilityMassFunction (Fin 2) :=
  let first_turn := if P = 0 then PMF.ofMultiset [(0, 1 / 3), (1, 2 / 3)] else PMF.ofMultiset [(1, 2 / 3), (0, 1 / 3)]
  let second_turn := if first_turn = 0 then PMF.ofMultiset [(0, 1 / 3), (1, 2 / 3)] else PMF.ofMultiset [(1, 2 / 3), (0, 1 / 3)]
  let third_turn := if second_turn = 0 then PMF.ofMultiset [(0, 1 / 3), (1, 2 / 3)] else PMF.ofMultiset [(1, 2 / 3), (0, 1 / 3)]
  third_turn

/-- Probability of Alice having the ball after three turns is 5/9 given the game rules -/
theorem alice_has_ball_after_three_turns:
  alice_bob_game 0.V.ProbabilityTree.stashing 3 0 = (5 / 9) := sorry

end alice_has_ball_after_three_turns_l792_792349


namespace biquadratic_roots_formula_l792_792277

noncomputable def biquadratic_roots (p q : ℂ) (h : p^2 - 4 * q < 0) : set ℂ :=
  {x | x^4 + p * x^2 + q = 0}

theorem biquadratic_roots_formula (p q : ℂ) (h : p^2 - 4 * q < 0) :
  biquadratic_roots p q h = 
  {x | x = complex.sqrt ((2 * complex.sqrt q - p) / 4) + complex.I * complex.sqrt ((2 * complex.sqrt q + p) / 4) ∨ 
        x = complex.sqrt ((2 * complex.sqrt q - p) / 4) - complex.I * complex.sqrt ((2 * complex.sqrt q + p) / 4) ∨ 
        x = -complex.sqrt ((2 * complex.sqrt q - p) / 4) + complex.I * complex.sqrt ((2 * complex.sqrt q + p) / 4) ∨ 
        x = -complex.sqrt ((2 * complex.sqrt q - p) / 4) - complex.I * complex.sqrt ((2 * complex.sqrt q + p) / 4)} :=
by
  sorry

end biquadratic_roots_formula_l792_792277


namespace permutation_order_24_l792_792221

-- Definitions for the problem conditions
def T0 : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
def T1 : List Char := ['J', 'Q', 'O', 'W', 'I', 'P', 'A', 'N', 'T', 'Z', 'R', 'C', 'V', 'M', 'Y', 'E', 'G', 'S', 'H', 'U', 'F', 'D', 'K', 'B', 'L', 'X']

-- Function to apply the permutation T1 to a given List of Char
def apply_permutation (perm T : List Char) : List Char :=
  perm.map (λ c => T.get (perm.indexOf c))

-- Prove that applying the permutation 24 times results in T0
theorem permutation_order_24 : (apply_permutation T1^[24] T0) = T0 := 
sorry

end permutation_order_24_l792_792221


namespace probability_of_joining_between_1890_and_1969_l792_792075

theorem probability_of_joining_between_1890_and_1969 :
  let total_provinces_and_territories := 13
  let joined_1890_to_1929 := 3
  let joined_1930_to_1969 := 1
  let total_joined_between_1890_and_1969 := joined_1890_to_1929 + joined_1930_to_1969
  total_joined_between_1890_and_1969 / total_provinces_and_territories = 4 / 13 :=
by
  sorry

end probability_of_joining_between_1890_and_1969_l792_792075


namespace cos_neg_three_pi_over_two_eq_zero_l792_792779

noncomputable def cos_neg_three_pi_over_two : ℝ :=
  Real.cos (-3 * Real.pi / 2)

theorem cos_neg_three_pi_over_two_eq_zero :
  cos_neg_three_pi_over_two = 0 :=
by
  -- Using trigonometric identities and periodicity of cosine function
  sorry

end cos_neg_three_pi_over_two_eq_zero_l792_792779


namespace subset_inequality_l792_792537

open Finset

variable {A : Type*} [Fintype A]

def is_pretty (P : Finset A) : Prop := sorry  -- Define the pretty condition here

def is_small (S P : Finset A) : Prop := S ⊆ P ∧ is_pretty P

def is_big (B P : Finset A) : Prop := P ⊆ B ∧ is_pretty P

def num_pretty : ℕ := (Fintype.card {P // is_pretty P})

def num_small : ℕ := (Fintype.card {S // ∃ P, is_pretty P ∧ S ⊆ P})

def num_big : ℕ := (Fintype.card {B // ∃ P, is_pretty P ∧ P ⊆ B})

theorem subset_inequality :
  2 ^ Fintype.card A * num_pretty ≤ num_small * num_big :=
begin
  sorry  -- Proof goes here
end

end subset_inequality_l792_792537
