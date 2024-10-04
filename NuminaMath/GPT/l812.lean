import Mathlib

namespace principal_calculation_l812_812991

-- Define the conditions
def amount : ℝ := 1568
def rate : ℝ := 0.05
def time : ℝ := 12/5

-- Define the principal amount to be proven
def principal : ℝ := 1400

-- State the main theorem
theorem principal_calculation :
  let P := principal in
  amount = P + P * rate * time :=
by
  -- Here we would provide the proof, but we skip it with sorry
  sorry

end principal_calculation_l812_812991


namespace problem_1_problem_2_l812_812170

-- Define the sets M and N as conditions and include a > 0 condition.
def M (a : ℝ) : Set ℝ := {x : ℝ | (x + a) * (x - 1) ≤ 0}
def N : Set ℝ := {x : ℝ | 4 * x ^ 2 - 4 * x - 3 < 0}

-- Problem 1: Prove that a = 2 given the set conditions.
theorem problem_1 (a : ℝ) (h_pos : a > 0) :
  M a ∪ N = {x : ℝ | -2 ≤ x ∧ x < 3 / 2} → a = 2 :=
sorry

-- Problem 2: Prove the range of a is 0 < a ≤ 1 / 2 given the set conditions.
theorem problem_2 (a : ℝ) (h_pos : a > 0) :
  N ∪ (compl (M a)) = Set.univ → 0 < a ∧ a ≤ 1 / 2 :=
sorry

end problem_1_problem_2_l812_812170


namespace parabola_circle_problem_l812_812830

noncomputable def parabola_eq (p : ℝ) : Prop :=
  sorry   -- The complete definition would involve the constraints y^2 = 2px.

noncomputable def circle_eq (x y r : ℝ) : Prop :=
  sorry   -- The complete definition would involve the equation (x - h)^2 + (y - k)^2 = r^2.

theorem parabola_circle_problem :
  (∀ (l C : ℝ → ℝ), 
    (∀ y, y = x - 1) → 
    (∃ p > 0, C = λ x, y^2 = 2px) → 
    l ((1, 0) ∈ l) →
    parabola_eq (1/2) C = 
      λ y, y^2 = 4x ) ∧
  (∀ (l C Q : ℝ → ℝ), 
    (∀ y, y = x - 1) → 
    C = λ x, y^2 = 4x →
    Q (AB as its diameter) 
    → circle_eq (3, 2, 4) Q = 
      (x - 3)^2 + (y - 2)^2 = 16) :=
begin
  sorry  -- Proof omitted
end

end parabola_circle_problem_l812_812830


namespace find_imaginary_and_modulus_l812_812648

def ω := (3 * Complex.I - 1) / Complex.I

theorem find_imaginary_and_modulus :
  (ω.im, Complex.abs ω) = (1, Real.sqrt 10) :=
by
  sorry

end find_imaginary_and_modulus_l812_812648


namespace sum_series_l812_812017

noncomputable def H (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), (1 : ℝ) / (k + 1)

theorem sum_series :
  ∑' n : ℕ, (1 : ℝ) / ((n + 2) * H n * H (n + 1)) = 1 / 3 := by
  sorry

end sum_series_l812_812017


namespace arccos_one_eq_zero_l812_812535

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812535


namespace percentage_bug_population_remaining_l812_812946

-- Define the initial conditions
def initial_bug_population : ℕ := 400
def spiders_introduced : ℕ := 12
def bugs_eaten_per_spider : ℕ := 7
def bugs_left_after_spraying : ℕ := 236

-- Define the proof statement
theorem percentage_bug_population_remaining :
  let bugs_eaten_by_spiders := spiders_introduced * bugs_eaten_per_spider in
  let bugs_before_spraying := initial_bug_population - bugs_eaten_by_spiders in
  (bugs_left_after_spraying * 100 / bugs_before_spraying : ℚ) = 74.68 := by
    sorry

end percentage_bug_population_remaining_l812_812946


namespace segments_two_points_condition_l812_812042

theorem segments_two_points_condition {α : Type*} [LinearOrderedField α] (segments : list (α × α)) :
  (∀ (s1 s2 s3 : α × α), s1 ∈ segments → s2 ∈ segments → s3 ∈ segments → 
   ∃ (p1 p2 : α), (p1 ∈ s1 ∨ p2 ∈ s1) ∧ (p1 ∈ s2 ∨ p2 ∈ s2) ∧ (p1 ∈ s3 ∨ p2 ∈ s3)) →
  ∃ (p1 p2 : α), ∀ (s : α × α), s ∈ segments → (p1 ∈ s ∨ p2 ∈ s) :=
by
  sorry

end segments_two_points_condition_l812_812042


namespace arccos_one_eq_zero_l812_812473

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812473


namespace find_a_and_b_min_value_expression_l812_812049

universe u

-- Part (1): Prove the values of a and b
theorem find_a_and_b :
    (∀ x : ℝ, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) →
    a = 1 ∧ b = 2 :=
sorry

-- Part (2): Given a = 1 and b = 2 prove the minimum value of 2x + y + 3
theorem min_value_expression :
    (1 / (x + 1) + 2 / (y + 1) = 1) →
    (x > 0) →
    (y > 0) →
    ∀ x y : ℝ, 2 * x + y + 3 ≥ 8 :=
sorry

end find_a_and_b_min_value_expression_l812_812049


namespace number_of_ways_to_choose_tribe_leadership_l812_812382

noncomputable def num_ways_to_choose_leadership (n : ℕ) : ℕ :=
  let choices_chief := n in
  let choices_sup_A := n - 1 in
  let choices_sup_B := n - 2 in
  let choices_assist_A := n - 3 in
  let choices_assist_B := n - 4 in
  let choices_inferior_officers_A := Nat.choose (n - 5) 2 in
  let choices_inferior_officers_B := Nat.choose (n - 7) 2 in
  choices_chief * choices_sup_A * choices_sup_B * choices_assist_A * choices_assist_B * choices_inferior_officers_A * choices_inferior_officers_B

theorem number_of_ways_to_choose_tribe_leadership (n : ℕ) (h : n = 15) : num_ways_to_choose_leadership n = 216216000 :=
by
  subst h
  have choices_chief : 15 = 15 := rfl
  have choices_sup_A : 14 = 14 := rfl
  have choices_sup_B : 13 = 13 := rfl
  have choices_assist_A : 12 = 12 := rfl
  have choices_assist_B : 11 = 11 := rfl
  have choices_inferior_officers_A : Nat.choose 10 2 = 45 := by norm_num
  have choices_inferior_officers_B : Nat.choose 8 2 = 28 := by norm_num
  calc
    num_ways_to_choose_leadership 15
        = 15 * 14 * 13 * 12 * 11 * Nat.choose 10 2 * Nat.choose 8 2 : rfl
    ... = 15 * 14 * 13 * 12 * 11 * 45 * 28 : by rw [choices_inferior_officers_A, choices_inferior_officers_B]
    ... = 216216000 : by norm_num

end number_of_ways_to_choose_tribe_leadership_l812_812382


namespace arccos_1_eq_0_l812_812413

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812413


namespace increasing_function_iff_l812_812063

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≤ 1 then (-x^2 + 2 * a * x - 3) else (4 - a) * x + 1

theorem increasing_function_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 1 ≤ a ∧ a ≤ 3 :=
by 
  -- The proof would go here. For now, we use sorry to skip it.
  sorry

end increasing_function_iff_l812_812063


namespace value_of_g_f_neg_2_l812_812652

def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - x else g x

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

axiom odd_f : is_odd f 

-- Define g(x) in accordance with the conditions derived
def g (x : ℝ) : ℝ := -x^2 - x

-- Prove the statement
theorem value_of_g_f_neg_2 : g (f (-2)) = -2 :=
by
  have h1 : f (-2) = 6 :=
    by
      sorry  -- Here, f(-2) calculation can be added
  have h2 : g 6 = -2 :=
    by
      sorry  -- Here, g(6) calculation can be added
  sorry

end value_of_g_f_neg_2_l812_812652


namespace arccos_one_eq_zero_l812_812541

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812541


namespace not_directly_nor_inversely_proportional_A_not_directly_nor_inversely_proportional_D_l812_812584

def equationA (x y : ℝ) : Prop := 2 * x + 3 * y = 5
def equationD (x y : ℝ) : Prop := 4 * x + 2 * y = 8

def directlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, y = k * x
def inverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

theorem not_directly_nor_inversely_proportional_A (x y : ℝ) :
  equationA x y → ¬ (directlyProportional x y ∨ inverselyProportional x y) := 
sorry

theorem not_directly_nor_inversely_proportional_D (x y : ℝ) :
  equationD x y → ¬ (directlyProportional x y ∨ inverselyProportional x y) := 
sorry

end not_directly_nor_inversely_proportional_A_not_directly_nor_inversely_proportional_D_l812_812584


namespace angle_C_pi_div_2_l812_812097

theorem angle_C_pi_div_2 (A B C a b c : ℝ) (h1 : sin A ≠ 0) (h2 : sin B ≠ 0) 
(h3 : a = c * sin A / sin C) (h4 : b = c * sin B / sin C) 
(h5 : a / sin B + b / sin A = 2 * c) : 
  C = π / 2 :=
by
  -- Detailed proof from given conditions will go here
  sorry

end angle_C_pi_div_2_l812_812097


namespace solution_set_l812_812167

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ (-∞, 0) ∪ (0, ∞), f (-(x + 1)) = f (x + 1)

def decreasing_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y ∈ (-∞, 0), x < y → f (x + 1) > f (y + 1)

def symmetric_around_one (f : ℝ → ℝ) : Prop :=
  f (1) = 0

noncomputable def proof_problem (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  even_function f ∧ 
  decreasing_interval f ∧ 
  symmetric_around_one f ∧ 
  (∀ x, x < 1 → f (x + 1) > 0) → 
  (∀ x, x > 1 → f (x + 1) < 0) → 
  s = { x | (x - 1) * f x ≤ 0}

theorem solution_set (f : ℝ → ℝ) :
  proof_problem f ((-∞) ∪ [2, ∞) ∪ {1, 2}) :=
sorry

end solution_set_l812_812167


namespace polynomial_identity_l812_812882

theorem polynomial_identity (x : ℝ) : 
  (2 * x^2 + 5 * x + 8) * (x + 1) - (x + 1) * (x^2 - 2 * x + 50) 
  + (3 * x - 7) * (x + 1) * (x - 2) = 4 * x^3 - 2 * x^2 - 34 * x - 28 := 
by 
  sorry

end polynomial_identity_l812_812882


namespace cos_triple_angle_l812_812088

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l812_812088


namespace find_equation_of_W_prove_perimeter_ABC_greater_than_3sqrt3_l812_812725

def locus_condition (P : ℝ × ℝ) : Prop := 
  abs P.snd = real.sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2)

def equation_W (x y : ℝ) : Prop := 
  y = x^2 + 1/4

theorem find_equation_of_W :
  ∀ (P : ℝ × ℝ), locus_condition P → equation_W P.fst P.snd :=
by 
  sorry

def three_vertices_on_W (A B C : ℝ × ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), A = (x₁, x₁^2 + 1/4) ∧ B = (x₂, x₂^2 + 1/4) ∧ C = (x₃, x₃^2 + 1/4)

def is_rectangle (A B C D : ℝ × ℝ) : Prop := 
  three_vertices_on_W A B C ∧ 
  ∃ Dx Dy, D = (Dx, Dy) ∧ 
  -- some conditions that ensure ABCD forms a rectangle

def perimeter (A B C D : ℝ × ℝ) : ℝ := 
  dist A B + dist B C + dist C D + dist D A

theorem prove_perimeter_ABC_greater_than_3sqrt3 :
  ∀ (A B C D : ℝ × ℝ), is_rectangle A B C D → three_vertices_on_W A B C → perimeter A B C D > 3 * real.sqrt 3 :=
by 
  sorry

end find_equation_of_W_prove_perimeter_ABC_greater_than_3sqrt3_l812_812725


namespace arccos_one_eq_zero_l812_812524

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812524


namespace largest_multiple_of_9_less_than_100_l812_812291

theorem largest_multiple_of_9_less_than_100 : ∃ x : ℕ, 9 * x < 100 ∧ ∀ y : ℕ, 9 * y < 100 → y ≤ x :=
by
  exists 11
  split
  · linarith
  · intro y h
    have : 9 * y < 100 := h
    calc
      y ≤ floor (11.11) : by linarith

end largest_multiple_of_9_less_than_100_l812_812291


namespace angle_bisectors_product_theorem_l812_812158

theorem angle_bisectors_product_theorem
  (a b c f_a f_b f_c T : ℝ)
  (h_fa : f_a = sqrt((4 * b * c * (a + b + c) / 2) * ((a + b + c) / 2 - a) / (b + c)^2))
  (h_fb : f_b = sqrt((4 * a * c * (a + b + c) / 2) * ((a + b + c) / 2 - b) / (a + c)^2))
  (h_fc : f_c = sqrt((4 * a * b * (a + b + c) / 2) * ((a + b + c) / 2 - c) / (a + b)^2))
  (area_T : 4 * T = sqrt((a + b + c) / 2 - a) * sqrt((a + b + c) / 2 - b) * sqrt((a + b + c) / 2 - c)) :
  (f_a * f_b * f_c) / (a * b * c) = 4 * T * (a + b + c) / ((a + b) * (b + c) * (a + c)) := 
sorry

end angle_bisectors_product_theorem_l812_812158


namespace diagonal_length_QS_is_18_closest_l812_812797

-- Definitions and conditions
def square_diagonal_length : ℝ := 2

def side_length_square : ℝ := Math.sqrt 2 

def horizontal_squares : ℝ := 5
def vertical_squares : ℝ := 12

def length_horizontal : ℝ := horizontal_squares * side_length_square
def length_vertical : ℝ := vertical_squares * side_length_square

-- Statement to prove
theorem diagonal_length_QS_is_18_closest : 
    (length_horizontal = 5 * Math.sqrt 2) →
    (length_vertical = 12 * Math.sqrt 2) →
    Real.sqrt ((length_horizontal)^2 + (length_vertical)^2) ≈ 18 := 
by 
    sorry

end diagonal_length_QS_is_18_closest_l812_812797


namespace real_number_roots_of_eqn_is_well_defined_set_l812_812935

-- Definition of conditions
def very_large_numbers := {x : ℝ | x > 10^6}  -- Arbitrary threshold for "very large"
def infinitely_close_to_zero := {x : ℝ | ∀ ε > 0, x ≠ 0 → |x| < ε}
def smart_people := {p : Type | true}  -- Smartness is subjective, so we use an arbitrary type
def real_number_roots_of_eqn := {x : ℝ | x^2 = 2}

-- Statement of the problem
theorem real_number_roots_of_eqn_is_well_defined_set : 
  real_number_roots_of_eqn = {√2, -√2} :=
sorry

end real_number_roots_of_eqn_is_well_defined_set_l812_812935


namespace largest_multiple_of_9_less_than_100_l812_812306

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ (∀ k, k < 100 → k % 9 = 0 → k ≤ n) :=
by {
  use 99,
  split,
  { exact lt_of_le_of_ne (nat.le_of_dvd (by norm_num) (by norm_num)) (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by use 11) },
  { intros k hk hkm, apply nat.mul_le_mul_left 9 (le_of_lt $ nat.div_lt_self hk $ by norm_num), exact le_of_lt_succ (nat.succ_le_of_lt (nat.div_lt_self hk $ by norm_num)) }
  sorry
}

end largest_multiple_of_9_less_than_100_l812_812306


namespace arccos_one_eq_zero_l812_812453

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812453


namespace describe_all_possible_painting_ways_l812_812962

def coloring_type := ℕ → ℕ

-- Define the predicate that checks the coloring condition.
def satisfies_condition (f : coloring_type) : Prop :=
∀ a b c : ℕ, 2000 * (a + b) = c → (f a = f b ∧ f b = f c) ∨ (f a ≠ f b ∧ f b ≠ f c ∧ f a ≠ f c)

-- A definition for cyclic coloring with period 3
def cyclic_coloring (c1 c2 c3 : ℕ) (n : ℕ) : ℕ :=
if n % 3 = 0 then c3 else if n % 3 = 1 then c1 else c2

-- A definition for a single color
def single_color (c : ℕ) : coloring_type := λ _, c

theorem describe_all_possible_painting_ways :
  ∀ f : coloring_type,
    satisfies_condition f →
    (∃ c : ℕ, f = single_color c) ∨
    (∃ (c1 c2 c3 : ℕ), f = cyclic_coloring c1 c2 c3) :=
begin
  sorry
end

end describe_all_possible_painting_ways_l812_812962


namespace largest_multiple_of_9_less_than_100_l812_812295

theorem largest_multiple_of_9_less_than_100 : ∃ n, n < 100 ∧ n % 9 = 0 ∧ (∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n) :=
begin
  use 99,
  split,
  -- Prove the number is less than 100
  { linarith, },
  split,
  -- Prove the number is a multiple of 9
  { norm_num, },
  -- Prove it is the largest
  { intros m hlt hdiv,
    linarith [hlt, hdiv],
    norm_num at hdiv,
    cases hdiv,
    exact hdiv_left, },
end

end largest_multiple_of_9_less_than_100_l812_812295


namespace largest_multiple_of_9_less_than_100_l812_812270

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l812_812270


namespace probability_A_and_B_same_car_l812_812616

theorem probability_A_and_B_same_car :
  let scenarios := {("AB", "CD"), ("AC", "BD"), ("AD", "BC"), ("BC", "AD"), ("BD", "AC"), ("CD", "AB")} in
  let favorable := {("AB", "CD"), ("CD", "AB")} in
  let probability := (favorable.to_finset.card : ℝ) / (scenarios.to_finset.card : ℝ) in
  probability = (1 : ℝ) / 3 := 
by
  sorry

end probability_A_and_B_same_car_l812_812616


namespace jason_cousins_l812_812745

theorem jason_cousins (x y : Nat) (dozen_to_cupcakes : Nat) (number_of_cousins : Nat)
    (hx : x = 4) (hy : y = 3) (hdozen : dozen_to_cupcakes = 12)
    (h : number_of_cousins = (x * dozen_to_cupcakes) / y) : number_of_cousins = 16 := 
by
  rw [hx, hy, hdozen] at h
  exact h
  sorry

end jason_cousins_l812_812745


namespace balance_of_diamondsuits_and_bullets_l812_812022

variable (a b c : ℕ)

theorem balance_of_diamondsuits_and_bullets 
  (h1 : 4 * a + 2 * b = 12 * c)
  (h2 : a = b + 3 * c) :
  3 * b = 6 * c := 
sorry

end balance_of_diamondsuits_and_bullets_l812_812022


namespace maximum_value_l812_812151

theorem maximum_value (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  3 * a * b * Real.sqrt 2 + 6 * b * c ≤ 4.5 :=
sorry

end maximum_value_l812_812151


namespace find_percentage_l812_812904

noncomputable def percentage (P : ℝ) : Prop :=
  (P / 100) * 1265 / 6 = 354.2

theorem find_percentage : ∃ (P : ℝ), percentage P ∧ P = 168 :=
by
  sorry

end find_percentage_l812_812904


namespace distance_points_parabola_line_y_eq_2_l812_812222

theorem distance_points_parabola_line_y_eq_2 (p q : ℕ) (hpq_coprime : Nat.gcd p q = 1) (h_dist : (sqrt p : ℝ) / q = (2 * sqrt 22) / 3) : p - q = 85 := by
  sorry

end distance_points_parabola_line_y_eq_2_l812_812222


namespace find_constants_PQR_l812_812988

theorem find_constants_PQR :
  ∃ P Q R : ℝ, 
    (6 * x + 2) / ((x - 4) * (x - 2) ^ 3) = P / (x - 4) + Q / (x - 2) + R / (x - 2) ^ 3 :=
by
  use 13 / 4
  use -6.5
  use -7
  sorry

end find_constants_PQR_l812_812988


namespace solve_for_t_l812_812810

theorem solve_for_t (t : ℝ) (h : sqrt (3 * sqrt (t - 3)) = real.nth_root 4 (10 - t)) : t = 3.7 :=
sorry

end solve_for_t_l812_812810


namespace find_a_plus_b_l812_812656

theorem find_a_plus_b 
  (a b : ℝ)
  (f : ℝ → ℝ) 
  (f_def : ∀ x, f x = x^3 + 3 * x^2 + 6 * x + 14)
  (cond_a : f a = 1) 
  (cond_b : f b = 19) :
  a + b = -2 :=
sorry

end find_a_plus_b_l812_812656


namespace beautiful_interval_x2_no_beautiful_interval_3_minus_4_over_x_max_n_minus_m_l812_812612

-- Definitions for Part 1
def isBeautifulInterval (f : ℝ → ℝ) (I : Set ℝ) (m n : ℝ) [Preorder ℝ] : Prop :=
  I ⊆ Set.Icc m n ∧
  MonotoneOn f (Set.Icc m n) ∧
  (Set.image f (Set.Icc m n)) = Set.Icc m n

-- Part 1, Statement 1: For function x^2
theorem beautiful_interval_x2 : isBeautifulInterval (fun x => x^2) Set.univ 0 1 := sorry

-- Part 1, Statement 2: For function 3 - 4/x
theorem no_beautiful_interval_3_minus_4_over_x : ¬∃ (m n : ℝ), isBeautifulInterval (fun x => 3 - 4 / x) (Set.Ioi 0) m n := sorry

-- Definitions for Part 2
def f_a (a : ℝ) (x : ℝ) := ((a^2 + a) * x - 1) / (a^2 * x)

-- Part 2: Maximum value of n - m
theorem max_n_minus_m (a : ℝ) (h : a ≠ 0) (m n : ℝ) (beautiful : isBeautifulInterval (f_a a) Set.univ m n) :
  n - m ≤ 2 * Real.sqrt 3 / 3 := sorry

end beautiful_interval_x2_no_beautiful_interval_3_minus_4_over_x_max_n_minus_m_l812_812612


namespace bisect_MX_l812_812712

noncomputable def acute_triangle (ABC : Type) [MetricSpace ABC] 
  (a b c : ABC) : Prop := 
  ∃ H : ABC, 
  ∃ BF CE : Set (ABC × ABC), 
  ∃ M X : ABC, 
  ∃ EF : Set (ABC × ABC), 
  is_orthocenter H a b c ∧  
  is_altitude H b F ∧ 
  is_altitude H c E ∧ 
  is_midpoint M b c ∧ 
  is_on_EF X E F ∧ 
  ∠HAM = ∠HMX ∧ 
  opposite_sides X a (line MH)

theorem bisect_MX (ABC : Type) [MetricSpace ABC]
  (a b c : ABC) (H M X : ABC) 
  (EF : Set (ABC × ABC)) 
  (h : acute_triangle ABC a b c) :
  bisects (line AH) (segment MX) := 
sorry

end bisect_MX_l812_812712


namespace arccos_one_eq_zero_l812_812512

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812512


namespace cube_faces_sum_l812_812808

theorem cube_faces_sum (a b c d e f : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
                        (hd : d > 0) (he : e > 0) (hf : f > 0)
                        (h : (a + d) * (b + e) * (c + f) = 357) :
  a + b + c + d + e + f = 27 :=
begin
  sorry,
end

end cube_faces_sum_l812_812808


namespace KolyaMaxSameLength_l812_812749

-- Define the base structure related to the triangular grid
inductive Direction : Type
| North | East | SouthWest

def left_turn : Direction → Direction
| Direction.North => Direction.West
| Direction.West => Direction.SouthWest
| Direction.SouthWest => Direction.North

structure Point where
  x : Int
  y : Int
  deriving Repr, BEq

structure Path where
  steps : List Point
  length : Nat
  turns : Nat

-- Define Kolya's and Max's paths
def KolyaPath : Path :=
  { steps := [ ⟨0, 0⟩, ⟨2, 0⟩, ⟨2, 1⟩, ⟨1, 2⟩, ⟨0, 2⟩, ⟨0, 1⟩, ⟨-1, 1⟩, ⟨-1, 0⟩ ]
    length := 8
    turns := 4 }

def MaxPath : Path :=
  { steps := [ ⟨0, 0⟩, ⟨3, 0⟩, ⟨3, 1⟩, ⟨2, 1⟩, ⟨1, 1⟩, ⟨1, 2⟩, ⟨2, 2⟩, ⟨0, 2⟩ ]
    length := 8
    turns := 1 }

-- Proving the equivalence of the lengths of the paths despite different turns
theorem KolyaMaxSameLength (kp : Path) (mp : Path) : 
  kp.length = mp.length ∧ kp.turns ≠ mp.turns → True := 
by
  intro h
  sorry

end KolyaMaxSameLength_l812_812749


namespace arccos_one_eq_zero_l812_812525

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812525


namespace ratio_closest_to_10_l812_812589

theorem ratio_closest_to_10 :
  (⌊(10^3000 + 10^3004 : ℝ) / (10^3001 + 10^3003) + 0.5⌋ : ℝ) = 10 :=
sorry

end ratio_closest_to_10_l812_812589


namespace E_radius_calc_l812_812588

noncomputable def find_radius_E (radius_A radius_B radius_C radius_D : ℚ) (m n: ℤ) 
  (T_is_equilateral : ∀ P1 P2 P3 : ℝ, equilateral P1 P2 P3)
  (T_inscribed_in_A : ∀ P1 P2 P3 : ℝ, is_inscribed_in T P1 P2 P3 A)
  (B_tangent_to_A : ∀ P : ℝ, internally_tangent B A P)
  (C_tangent_to_A : ∀ P : ℝ, internally_tangent C A P)
  (D_tangent_to_A : ∀ P : ℝ, internally_tangent D A P)
  (B_tangent_to_C : ∀ P : ℝ, externally_tangent B C P)
  (B_tangent_to_D : ∀ P : ℝ, externally_tangent B D P)
  (B_tangent_to_E : ∀ P : ℝ, externally_tangent B E P)
  (C_tangent_to_E : ∀ P : ℝ, externally_tangent C E P)
  (D_tangent_to_E : ∀ P : ℝ, externally_tangent D E P) : ℚ :=
if relatively_prime m n then radius_B + radius_E else 0

theorem E_radius_calc (radius_A : ℚ := 12) (radius_B: ℚ := 4) (radius_C: ℚ := 3)
  : find_radius_E radius_A radius_B radius_C D E T_is_equilateral T_inscribed_in_A 
    B_tangent_to_A C_tangent_to_A D_tangent_to_A B_tangent_to_C 
    C_tangent_to_D B_tangent_to_E C_tangent_to_E D_tangent_to_E = 111 :=
by
  sorry

end E_radius_calc_l812_812588


namespace calculate_two_squared_l812_812392

theorem calculate_two_squared : 2^2 = 4 :=
by
  sorry

end calculate_two_squared_l812_812392


namespace intersection_A_B_l812_812754

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℝ := {x | (x - 1) * (x - 5) < 0}

theorem intersection_A_B :
  A ∩ B = {2, 3, 4} :=
sorry

end intersection_A_B_l812_812754


namespace oxen_grazing_months_l812_812324

theorem oxen_grazing_months (a_oxen : ℕ) (a_months : ℕ) (b_oxen : ℕ) (c_oxen : ℕ) (c_months : ℕ) (total_rent : ℝ) (c_share_rent : ℝ) (x : ℕ) :
  a_oxen = 10 →
  a_months = 7 →
  b_oxen = 12 →
  c_oxen = 15 →
  c_months = 3 →
  total_rent = 245 →
  c_share_rent = 63 →
  (c_oxen * c_months) / ((a_oxen * a_months) + (b_oxen * x) + (c_oxen * c_months)) = c_share_rent / total_rent →
  x = 5 :=
sorry

end oxen_grazing_months_l812_812324


namespace milk_intake_l812_812994

theorem milk_intake (gallons_flora : ℕ) (extra_gallons_brother : ℕ) (weeks : ℕ) :
  gallons_flora = 3 → extra_gallons_brother = 2 → weeks = 3 →
  let daily_intake := gallons_flora + extra_gallons_brother in
  let total_days := weeks * 7 in
  let total_milk := daily_intake * total_days in
  total_milk = 105 :=
by
  intros h1 h2 h3
  unfold daily_intake total_days total_milk
  rw [h1, h2, h3]
  sorry

end milk_intake_l812_812994


namespace find_a_l812_812056

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else 2^x

theorem find_a (a : ℝ) : f a = 1 / 2 ↔ (a = -1 ∨ a = Real.sqrt 2) :=
by
  sorry

end find_a_l812_812056


namespace arccos_one_eq_zero_l812_812499

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812499


namespace maximum_deviation_l812_812846

/-- 
Given a set of stones each with a mass not exceeding 2 kg and a total mass of 100 kg,
proves that the maximum value of the minimum deviation \( d \), such that \( d \) is
the minimum difference from 10 kg when selecting a subset of stones, 
is \( \frac{10}{11} \).
-/
theorem maximum_deviation (stones : list ℝ) (h_mass : ∀ m ∈ stones, m ≤ 2) (htotal : stones.sum = 100) :
  ∃ (d : ℝ), d = 10 / 11 ∧ (∀ s ⊆ stones, |10 - s.sum| ≥ d) :=
sorry

end maximum_deviation_l812_812846


namespace julie_read_pages_tomorrow_l812_812138

-- Definitions based on conditions
def total_pages : ℕ := 120
def pages_read_yesterday : ℕ := 12
def pages_read_today : ℕ := 2 * pages_read_yesterday

-- Problem statement to prove Julie should read 42 pages tomorrow
theorem julie_read_pages_tomorrow :
  let total_read := pages_read_yesterday + pages_read_today in
  let remaining_pages := total_pages - total_read in
  let pages_tomorrow := remaining_pages / 2 in
  pages_tomorrow = 42 :=
by
  sorry

end julie_read_pages_tomorrow_l812_812138


namespace square_inscribed_area_eq_l812_812815

theorem square_inscribed_area_eq :
  ∀ (s : ℝ), 
  let x := s in
  let vertex_R := (3 + s, -2 * s) in
  let parabola_eq := (vertex_R.2 = (3+s)^2 - 6*(3+s) + 5) in
  parabola_eq → (2 * s)^2 = 64 - 16 * real.sqrt 5 := 
by 
  intro s,
  let x := s,
  let vertex_R := (3 + s, -2 * s),
  let parabola_eq := (vertex_R.2 = (3 + s)^2 - 6 * (3 + s) + 5),
  intro h,
  sorry

end square_inscribed_area_eq_l812_812815


namespace product_divisibility_l812_812144

theorem product_divisibility (k m n : ℕ) (hk : nat.prime (m + k + 1)) (hmk : m + k + 1 > n + 1) :
  let c (s : ℕ) := s * (s + 1) in
  (∏ i in finset.range n, (c (m + i + 1) - c k)) % (∏ i in finset.range n, c (i + 1)) = 0 :=
by
  sorry

end product_divisibility_l812_812144


namespace mertens_first_theorem_l812_812160

noncomputable def N : ℕ := sorry

-- Definition of the sum involving the logarithms of primes
noncomputable def log_sum (N : ℕ) : ℝ :=
(∑ p in Finset.filter Nat.Prime (Finset.range (N + 1)), real.log p / p)

theorem mertens_first_theorem (N : ℕ) : 
  let log_N := real.log (N : ℝ) in
  abs (log_sum N - log_N) < 4 :=
sorry

end mertens_first_theorem_l812_812160


namespace arccos_one_eq_zero_l812_812487

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812487


namespace out_of_79_consecutive_integers_exists_one_sum_divisible_by_13_not_true_for_78_consecutive_integers_l812_812796

def sum_of_digits (n : ℕ) : ℕ :=
  n.to_string.to_list.sum (λ c, c.to_nat - '0'.to_nat)

theorem out_of_79_consecutive_integers_exists_one_sum_divisible_by_13 :
  ∀ (n : ℕ), ∃ (x : ℕ), (n ≤ x) ∧ (x ≤ n + 78) ∧ (sum_of_digits x % 13 = 0) := 
by {
  sorry
}

theorem not_true_for_78_consecutive_integers :
  ∃ (n : ℕ), ∀ (x : ℕ), (n ≤ x) ∧ (x ≤ n + 77) → (sum_of_digits x % 13 ≠ 0) := 
by {
  sorry
}

end out_of_79_consecutive_integers_exists_one_sum_divisible_by_13_not_true_for_78_consecutive_integers_l812_812796


namespace factorize_expression_l812_812987

variable {R : Type} [Ring R]
variables (a b x y : R)

theorem factorize_expression :
  8 * a * x - b * y + 4 * a * y - 2 * b * x = (4 * a - b) * (2 * x + y) :=
sorry

end factorize_expression_l812_812987


namespace sec_prod_eq_pq_sum_eq_l812_812881

-- Define the product defined in the problem
def sec_prod : ℤ := ∏ k in (Finset.range 30).map ((λ n ↦ 3 * (n + 1) : ℕ → ℤ)), (Real.sec (k : ℝ)).pow 2

-- Define the integers p and q
def p : ℤ := 2
def q : ℤ := 30

-- State the theorem to be proved
theorem sec_prod_eq :
  sec_prod = p ^ q := by
sorry

-- State the final sum p+q
theorem pq_sum_eq :
  p + q = 32 := by
sorry

end sec_prod_eq_pq_sum_eq_l812_812881


namespace factorize_expression_l812_812591

theorem factorize_expression (m n : ℤ) : 
  4 * m^2 * n - 4 * n^3 = 4 * n * (m + n) * (m - n) :=
by
  sorry

end factorize_expression_l812_812591


namespace abscissa_of_Q_after_rotation_l812_812715

theorem abscissa_of_Q_after_rotation (P : ℝ × ℝ) (θ : ℝ) (x : ℝ) (y : ℝ) : 
  P = (1, 2) → θ = 5 * Real.pi / 6 → (x, y) = ((1 * Real.cos θ) - (2 * Real.sin θ), (1 * Real.sin θ) + (2 * Real.cos θ)) → 
  (x = -Real.sqrt 3 / 2 - 2 * Real.sqrt 5) ∧ (x^2 + y^2 = 5) := 
by 
  intros hP hθ hQ
  sorry

#eval abscissa_of_Q_after_rotation (1, 2) (5 * Real.pi / 6) ?x ?y

end abscissa_of_Q_after_rotation_l812_812715


namespace range_of_p_l812_812081

-- Conditions: p is a prime number and the roots of the quadratic equation are integers 
def p_is_prime (p : ℕ) : Prop := Nat.Prime p

def roots_are_integers (p : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ x * y = -204 * p ∧ (x + y) = p

-- Main statement: Prove the range of p
theorem range_of_p (p : ℕ) (hp : p_is_prime p) (hr : roots_are_integers p) : 11 < p ∧ p ≤ 21 :=
  sorry

end range_of_p_l812_812081


namespace arccos_one_eq_zero_l812_812520

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812520


namespace angle_AQH_is_90_l812_812806

noncomputable def angle_AQH (A B G H Q : Point) : Prop :=
  -- Preconditions describing the regular octagon and the extension
  RegularOctagon A B G H ∧ ExtendsToMeet A B G H Q →
  Angle A Q H = 90

theorem angle_AQH_is_90 (A B G H Q : Point) :
  angle_AQH A B G H Q :=
sorry

end angle_AQH_is_90_l812_812806


namespace max_percentage_l812_812343

def total_students : ℕ := 100
def group_size : ℕ := 66
def min_percentage (scores : Fin 100 → ℝ) : Prop :=
  ∀ (S : Finset (Fin 100)), S.card = 66 → (S.sum scores) / (Finset.univ.sum scores) ≥ 0.5

theorem max_percentage (scores : Fin 100 → ℝ) (h : min_percentage scores) :
  ∃ (x : ℝ), ∀ i : Fin 100, scores i <= x ∧ x <= 0.25 * (Finset.univ.sum scores) := sorry

end max_percentage_l812_812343


namespace problem_statement_l812_812792

theorem problem_statement (n : ℕ) : 2 ^ n ∣ (1 + ⌊(3 + Real.sqrt 5) ^ n⌋) :=
by
  sorry

end problem_statement_l812_812792


namespace arccos_one_eq_zero_l812_812518

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812518


namespace find_equation_of_W_prove_perimeter_ABC_greater_than_3sqrt3_l812_812721

def locus_condition (P : ℝ × ℝ) : Prop := 
  abs P.snd = real.sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2)

def equation_W (x y : ℝ) : Prop := 
  y = x^2 + 1/4

theorem find_equation_of_W :
  ∀ (P : ℝ × ℝ), locus_condition P → equation_W P.fst P.snd :=
by 
  sorry

def three_vertices_on_W (A B C : ℝ × ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), A = (x₁, x₁^2 + 1/4) ∧ B = (x₂, x₂^2 + 1/4) ∧ C = (x₃, x₃^2 + 1/4)

def is_rectangle (A B C D : ℝ × ℝ) : Prop := 
  three_vertices_on_W A B C ∧ 
  ∃ Dx Dy, D = (Dx, Dy) ∧ 
  -- some conditions that ensure ABCD forms a rectangle

def perimeter (A B C D : ℝ × ℝ) : ℝ := 
  dist A B + dist B C + dist C D + dist D A

theorem prove_perimeter_ABC_greater_than_3sqrt3 :
  ∀ (A B C D : ℝ × ℝ), is_rectangle A B C D → three_vertices_on_W A B C → perimeter A B C D > 3 * real.sqrt 3 :=
by 
  sorry

end find_equation_of_W_prove_perimeter_ABC_greater_than_3sqrt3_l812_812721


namespace simple_interest_is_correct_l812_812231

-- Define the principal amount, rate of interest, and time
def P : ℕ := 400
def R : ℚ := 22.5
def T : ℕ := 2

-- Define the formula for simple interest
def simple_interest (P : ℕ) (R : ℚ) (T : ℕ) : ℚ :=
  (P * R * T) / 100

-- The statement we need to prove
theorem simple_interest_is_correct : simple_interest P R T = 90 :=
by
  sorry

end simple_interest_is_correct_l812_812231


namespace inverse_function_value_l812_812685

theorem inverse_function_value : 
  (∃ x : ℝ, (λ x, (x^3 - 3) / 2) x = 1 / 4) → ∃ y : ℝ, y = (7 / 2)^(1 / 3) := 
sorry

end inverse_function_value_l812_812685


namespace total_shirts_produced_l812_812942

def group1_shirt_rate := 24 / 12  -- Shirts per minute per machine for group 1
def group2_shirt_rate := 45 / 20  -- Shirts per minute per machine for group 2

def group1_total_time := 18  -- Minutes of operation for group 1
def group2_total_time := 22  -- Minutes of operation for group 2

def group1_production := 24 * group1_total_time  -- Total production for group 1
def group2_production := 45 * group2_total_time  -- Total production for group 2

def total_production := group1_production + group2_production

theorem total_shirts_produced : total_production = 1422 := by
  calc 
    total_production = group1_production + group2_production := by rfl
    ... = 24 * group1_total_time + 45 * group2_total_time := by rfl
    ... = 24 * 18 + 45 * 22 := by rfl
    ... = 432 + 990 := by rfl
    ... = 1422 := by rfl

end total_shirts_produced_l812_812942


namespace inclination_angle_range_l812_812837

theorem inclination_angle_range (α θ : ℝ) (h_line : ∃ (sin_α : ℝ), x * sin_α - y + 1 = 0)
   (h_theta_range : 0 ≤ θ ∧ θ < π)
   (h_tan_theta : ∃ (tan_θ : ℝ), tan_θ = -1 ∨ tan_θ = 1) :
   θ ∈ set.Icc 0 (π / 4) ∪ set.Ico (3 * π / 4) π :=
sorry

end inclination_angle_range_l812_812837


namespace neg_prop_p_l812_812892

-- Define the function f as a real-valued function
variable (f : ℝ → ℝ)

-- Definitions for the conditions in the problem
def prop_p := ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

-- Theorem stating the negation of proposition p
theorem neg_prop_p : ¬prop_p f ↔ ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
by 
  sorry

end neg_prop_p_l812_812892


namespace slope_of_line_l812_812912

open Real

-- Definitions for the given problem conditions
def point (x y : ℝ) := (x, y)
def circle_center : point 0 0 := (0, 0)

def line_through_point (P : point (-(sqrt 3)) 0) (k : ℝ) : Prop :=
  ∃ A B : point.1 = x ∧ point.2 = y, 
  let line_eq : x = k * (y + sqrt 3) := sorry

def triangle_area (A B O : point) : ℝ :=
  1/2 * Real.abs ((A.1 * (B.2 - O.2)) + (B.1 * (O.2 - A.2)) + (O.1 * (A.2 - B.2)))

-- Given conditions
def passes_through_P : line_through_point (-(sqrt 3)) 0 k := sorry
def intersects_circle (x y : ℝ) : x^2 + y^2 = 1 := sorry
def angle_AOB (A B : point) : ℝ := ∠(0,0) (A.1, A.2) (B.1, B.2)
def area_of_triangle_is_sqrt_three_over_four (A B : point) (O : circle_center) :=
  triangle_area A B O = (sqrt 3) / 4

-- Main problem statement
theorem slope_of_line (P : point (-(sqrt 3)) 0) (A B : point) (O : circle_center) 
  (k : ℝ) (theta : ℝ) (h1 : passes_through_P)
  (h2 : intersects_circle A.1 A.2)
  (h3 : intersects_circle B.1 B.2)
  (h4 : angle_AOB A B = theta) 
  (h5 : theta ∈ (0, π/2))
  (h6 : area_of_triangle_is_sqrt_three_over_four A B O) :
  k = -(sqrt 3)/3 ∨ k = (sqrt 3)/3 := sorry

end slope_of_line_l812_812912


namespace find_equation_of_W_prove_perimeter_ABC_greater_than_3sqrt3_l812_812724

def locus_condition (P : ℝ × ℝ) : Prop := 
  abs P.snd = real.sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2)

def equation_W (x y : ℝ) : Prop := 
  y = x^2 + 1/4

theorem find_equation_of_W :
  ∀ (P : ℝ × ℝ), locus_condition P → equation_W P.fst P.snd :=
by 
  sorry

def three_vertices_on_W (A B C : ℝ × ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), A = (x₁, x₁^2 + 1/4) ∧ B = (x₂, x₂^2 + 1/4) ∧ C = (x₃, x₃^2 + 1/4)

def is_rectangle (A B C D : ℝ × ℝ) : Prop := 
  three_vertices_on_W A B C ∧ 
  ∃ Dx Dy, D = (Dx, Dy) ∧ 
  -- some conditions that ensure ABCD forms a rectangle

def perimeter (A B C D : ℝ × ℝ) : ℝ := 
  dist A B + dist B C + dist C D + dist D A

theorem prove_perimeter_ABC_greater_than_3sqrt3 :
  ∀ (A B C D : ℝ × ℝ), is_rectangle A B C D → three_vertices_on_W A B C → perimeter A B C D > 3 * real.sqrt 3 :=
by 
  sorry

end find_equation_of_W_prove_perimeter_ABC_greater_than_3sqrt3_l812_812724


namespace solved_problem_l812_812055

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_deriv : ∀ x : ℝ, f' x < 1/2

theorem solved_problem (x : ℝ) : (f (x^2) < x^2 / 2 + 1 / 2) ↔ (x < -1 ∨ x > 1) := 
by
  sorry

end solved_problem_l812_812055


namespace find_f_7_l812_812636

theorem find_f_7 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f(-x) = -f(x)) (h2 : ∀ x : ℝ, f(x + 4) = f(x)) 
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f(x) = 2 * x^2) : 
  f 7 = -2 := 
sorry

end find_f_7_l812_812636


namespace number_of_boys_in_school_l812_812366

variable (x : ℕ) (y : ℕ)

theorem number_of_boys_in_school 
    (h1 : 1200 = x + (1200 - x))
    (h2 : 200 = y + (y + 10))
    (h3 : 105 / 200 = (x : ℝ) / 1200) 
    : x = 630 := 
  by 
  sorry

end number_of_boys_in_school_l812_812366


namespace juan_stamp_cost_l812_812894

-- Defining the prices of the stamps
def price_brazil : ℝ := 0.07
def price_peru : ℝ := 0.05

-- Defining the number of stamps from the 70s and 80s
def stamps_brazil_70s : ℕ := 12
def stamps_brazil_80s : ℕ := 15
def stamps_peru_70s : ℕ := 6
def stamps_peru_80s : ℕ := 12

-- Calculating total number of stamps from the 70s and 80s
def total_stamps_brazil : ℕ := stamps_brazil_70s + stamps_brazil_80s
def total_stamps_peru : ℕ := stamps_peru_70s + stamps_peru_80s

-- Calculating total cost
def total_cost_brazil : ℝ := total_stamps_brazil * price_brazil
def total_cost_peru : ℝ := total_stamps_peru * price_peru

def total_cost : ℝ := total_cost_brazil + total_cost_peru

-- Proof statement
theorem juan_stamp_cost : total_cost = 2.79 :=
by
  sorry

end juan_stamp_cost_l812_812894


namespace julie_reads_tomorrow_l812_812139

theorem julie_reads_tomorrow :
  let total_pages := 120
  let pages_read_yesterday := 12
  let pages_read_today := 2 * pages_read_yesterday
  let pages_read_so_far := pages_read_yesterday + pages_read_today
  let remaining_pages := total_pages - pages_read_so_far
  remaining_pages / 2 = 42 :=
by
  sorry

end julie_reads_tomorrow_l812_812139


namespace cubic_equation_with_real_roots_l812_812352

theorem cubic_equation_with_real_roots :
  ∃ a b c : ℤ, (∀ x1 x2 x3 : ℝ,
                  x1 * x2 * x3 = x1 + x2 + x3 + 2 ∧
                  x1^2 + x2^2 + x3^2 = 10 ∧
                  x1^3 + x2^3 + x3^3 = 6 →
                  ∃ (coeffs : ℤ × ℤ × ℤ), coeffs = (a, b, c)) ∧
                a = 0 ∧ b = -5 ∧ c = -2 :=
begin
  use [0, -5, -2],
  intro x1 x2 x3,
  intro h,
  have prod_eq_sum_plus_2 : x1 * x2 * x3 = x1 + x2 + x3 + 2 := h.1,
  have sum_of_squares : x1^2 + x2^2 + x3^2 = 10 := h.2.1,
  have sum_of_cubes : x1^3 + x2^3 + x3^3 = 6 := h.2.2,
  sorry
end

end cubic_equation_with_real_roots_l812_812352


namespace omega_range_l812_812653

def f (ω x : ℝ) : ℝ := Real.cos (2 * ω * x + Real.pi / 3)

theorem omega_range 
  (ω : ℝ) 
  (hω : 0 < ω) 
  (hzeros : ∃ l : List ℝ, l.length = 10 ∧ ∀ x ∈ l, f ω x = 0 ∧ 0 < x ∧ x < 2 * Real.pi) :
  (55 / 24) < ω ∧ ω ≤ (61 / 24) := by
  sorry

end omega_range_l812_812653


namespace range_of_a_l812_812697

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) := 
sorry

end range_of_a_l812_812697


namespace quad_function_one_zero_l812_812094

theorem quad_function_one_zero (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 6 * x + 1 = 0 ∧ (∀ x1 x2 : ℝ, m * x1^2 - 6 * x1 + 1 = 0 ∧ m * x2^2 - 6 * x2 + 1 = 0 → x1 = x2)) ↔ (m = 0 ∨ m = 9) :=
by
  sorry

end quad_function_one_zero_l812_812094


namespace sequence_remainder_204_l812_812760

/-- Let T be the increasing sequence of positive integers whose binary representation has exactly 6 ones.
    Let M be the 500th number in T.
    Prove that the remainder when M is divided by 500 is 204. -/
theorem sequence_remainder_204 :
    let T := {n : ℕ | (nat.popcount n = 6)}.to_list.sorted (≤)
    let M := T.get 499
    M % 500 = 204 :=
by
    sorry

end sequence_remainder_204_l812_812760


namespace arccos_1_eq_0_l812_812409

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812409


namespace line_parallel_plane_l812_812226

theorem line_parallel_plane (l : Line) (P : Plane) :
  (∀ L : Line, L ∈ P → ¬(l ∩ L ≠ ∅)) ↔ l ∥ P := 
sorry

end line_parallel_plane_l812_812226


namespace probability_sarah_thomas_in_picture_l812_812801

/-- Sarah and Thomas run on a circular track. Sarah runs counterclockwise, completing a lap every 
100 seconds, and Thomas runs clockwise, completing a lap every 75 seconds. They both start from 
the same line at the same time. At a random time between 12 minutes and 14 minutes after they begin 
to run, a photographer standing inside the track takes a picture covering one-third of the track, 
centered on the starting line. The probability that both Sarah and Thomas are in the picture is 
111/200. -/
theorem probability_sarah_thomas_in_picture : 
  let time_start := 12 * 60  -- starting time in seconds (12 minutes)
      time_end := 14 * 60   -- ending time in seconds (14 minutes)
      lap_sarah := 100  -- time for Sarah to complete one lap
      lap_thomas := 75  -- time for Thomas to complete one lap
      photo_coverage := 1 / 3  -- proportion of the track covered by the picture
  in ∀ t : ℕ, (time_start ≤ t ∧ t ≤ time_end) →
      let pos_sarah := (t % lap_sarah)  -- Sarah's position within her current lap
          pos_thomas := (t % lap_thomas)  -- Thomas's position within his current lap
          start_line := 0  -- starting line position reference
          coverage_time_sarah := (photo_coverage * lap_sarah).to_nat  -- time Sarah is within the picture coverage
          coverage_time_thomas := (photo_coverage * lap_thomas).to_nat  -- time Thomas is within the picture coverage
          start_coverage_sarah := pos_sarah  -- when Sarah enters the picture
          end_coverage_sarah := pos_sarah + coverage_time_sarah  -- when Sarah exits the picture
          start_coverage_thomas := pos_thomas  -- when Thomas enters the picture
          end_coverage_thomas := pos_thomas + coverage_time_thomas  -- when Thomas exits the picture
          common_start := max start_coverage_sarah start_coverage_thomas  -- start of common coverage
          common_end := min end_coverage_sarah end_coverage_thomas -- end of common coverage
      in (common_start ≤ common_end) →
         let common_coverage := common_end - common_start  -- duration of common coverage
             total_time_window := 60 -- total time window considered (in seconds)
         in common_coverage / total_time_window = 111 / 200 :=
by sorry

end probability_sarah_thomas_in_picture_l812_812801


namespace wolf_cannot_catch_any_sheep_l812_812217

-- Definitions representing the conditions
def plane := ℝ × ℝ
def move (pos: plane) (direction: plane) (distance: ℝ) : plane :=
  (pos.1 + direction.1 * distance, pos.2 + direction.2 * distance)

noncomputable def w_moves : ℕ → plane
def s_moves : ℕ → plane → ℕ → plane

-- The theorem to be proven
theorem wolf_cannot_catch_any_sheep
  (wolf_moves : ℕ → plane) (sheep_moves : ℕ → plane → ℕ → plane) 
  (max_move: ∀ (n : ℕ), ∥wolf_moves (n+1) - wolf_moves n∥ ≤ 1 ∨ ∀ m < 50, ∥sheep_moves (n/2) (wolf_moves n) m - sheep_moves (n/2-1) (wolf_moves (n-1)) m∥ ≤ 1):
  ∀ (i : ℕ), ∀ t : ℕ, ¬ (wolf_moves t = sheep_moves (t/2) (wolf_moves t) i) :=
sorry

end wolf_cannot_catch_any_sheep_l812_812217


namespace tj_second_half_time_l812_812821

theorem tj_second_half_time :
  ∀ (total_distance half_distance : ℝ) (time_first_half avg_time_per_km : ℝ),
    total_distance = 10 ∧
    half_distance = total_distance / 2 ∧
    time_first_half = 20 ∧
    avg_time_per_km = 5 →
    (total_distance * avg_time_per_km - time_first_half) = 30 :=
by
  intros total_distance half_distance time_first_half avg_time_per_km
  intro h
  cases h with h1 hrest
  cases hrest with h2 hrest
  cases hrest with h3 h4
  rw [h1, h2, h3, h4]
  calc
    10 * 5 - 20 = 50 - 20 := by ring
             ... = 30 := by ring

end tj_second_half_time_l812_812821


namespace g_has_three_zeros_in_interval_l812_812695

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * cos (2 / 3 * x)

-- Define the function g(x), which is a shift of f(x) to the right by π/2 units
def g (x : ℝ) : ℝ := 2 * cos (2 / 3 * x - π / 3)

-- The theorem stating the desired property about g(x)
theorem g_has_three_zeros_in_interval :
  ∃ a b c : ℝ, 0 < a ∧ a < b ∧ b < c ∧ c < 5 * π ∧ g a = 0 ∧ g b = 0 ∧ g c = 0 :=
sorry

end g_has_three_zeros_in_interval_l812_812695


namespace equation_of_locus_perimeter_greater_than_3sqrt3_l812_812730

theorem equation_of_locus 
  (P : ℝ × ℝ)
  (hx : ∀ y : ℝ, P.snd = |P.snd| → |P.snd| = sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2))
  :
  (P.snd = P.fst ^ 2 + 1/4) := 
sorry

theorem perimeter_greater_than_3sqrt3
  (A B C D : ℝ × ℝ)
  (h_parabola : ∀ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4)
  (h_vert_A : A ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_vert_B : B ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_vert_C : C ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_perpendicular: ((B.fst - A.fst) * (C.fst - B.fst) + (B.snd - A.snd) * (C.snd - B.snd)) = 0) 
  :
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
sorry

end equation_of_locus_perimeter_greater_than_3sqrt3_l812_812730


namespace arccos_one_eq_zero_l812_812495

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812495


namespace area_ratio_l812_812370

-- Defining the given problem conditions and the final proof statement.
section SquareFold

variables (s : ℝ) (A B : ℝ)

-- Condition 1: A square piece of paper with side length s
def original_area : ℝ := s^2

-- Condition 2: The square is divided into three equal sections along one side
def section_length : ℝ := s / 3

-- Condition 4: The area of the new shape after folding along the dotted line
def triangle_area : ℝ := (1 / 2) * section_length * s
def new_area : ℝ := original_area - triangle_area

-- The final proof problem
theorem area_ratio (hA : A = s^2) (hB : B = s^2 - (1 / 6) * s^2) :
  B / A = 5 / 6 :=
sorry

end SquareFold

end area_ratio_l812_812370


namespace symmetric_curve_polar_equation_l812_812070

theorem symmetric_curve_polar_equation (ρ θ : ℝ) :
  (∃ ρ θ, ρ^2 - 4*ρ*cos θ - 4 = 0) ∧ 
  C₂_symmetric_to_C₁ ↔ ρ^2 - 4*ρ*sin θ - 4 = 0) := 
begin
  sorry
end

end symmetric_curve_polar_equation_l812_812070


namespace equation_of_locus_perimeter_greater_than_3sqrt3_l812_812728

theorem equation_of_locus 
  (P : ℝ × ℝ)
  (hx : ∀ y : ℝ, P.snd = |P.snd| → |P.snd| = sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2))
  :
  (P.snd = P.fst ^ 2 + 1/4) := 
sorry

theorem perimeter_greater_than_3sqrt3
  (A B C D : ℝ × ℝ)
  (h_parabola : ∀ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4)
  (h_vert_A : A ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_vert_B : B ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_vert_C : C ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_perpendicular: ((B.fst - A.fst) * (C.fst - B.fst) + (B.snd - A.snd) * (C.snd - B.snd)) = 0) 
  :
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
sorry

end equation_of_locus_perimeter_greater_than_3sqrt3_l812_812728


namespace coefficient_x_term_l812_812118

noncomputable def binomial_expansion_coefficient (x : ℝ) (n : ℕ) : ℝ :=
  let sqrt_x := real.cbrt x
  let term := ∑ r in finset.range (n+1), ((-2) ^ r) * (nat.choose n r) * (sqrt_x ^ (n - 3 * r))
  term

theorem coefficient_x_term :
  (∑ r in finset.range 8, ((-2) ^ r) * (nat.choose 7 r) * (real.cbrt x ^ (7 - 3 * r))) = -14 :=
by
  admit

end coefficient_x_term_l812_812118


namespace intersection_is_correct_l812_812632

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}

theorem intersection_is_correct : A ∩ B = {-1, 2} := 
by 
  -- proof goes here 
  sorry

end intersection_is_correct_l812_812632


namespace cos_two_pi_over_three_plus_two_alpha_l812_812082

theorem cos_two_pi_over_three_plus_two_alpha 
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := 
by
  sorry

end cos_two_pi_over_three_plus_two_alpha_l812_812082


namespace arccos_one_eq_zero_l812_812536

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812536


namespace quadratic_matching_for_roots_l812_812605

theorem quadratic_matching_for_roots:
  (∀ x : ℝ, 3 * sqrt x + 5 * x ^ (-1 / 2) = 8 ↔ (x = 25/9 ∨ x = 1)) →
  (∀ a b c : ℝ, 3 * sqrt a + 5 * a ^ (-1 / 2) = 8 → (9 * a^2 - 46 * a + 25 = 0)) :=
by
  intros h x a b c h' h'' 
  sorry

end quadratic_matching_for_roots_l812_812605


namespace arccos_one_eq_zero_l812_812502

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812502


namespace smallest_angle_of_isosceles_obtuse_triangle_l812_812383

theorem smallest_angle_of_isosceles_obtuse_triangle :
  ∃ (α : ℝ), α = 36.0 ∧
  ∃ (A : ℝ), A = 108 ∧
  (∀ (B C : ℝ), B = α ∧ C = α → A + B + C = 180) :=
by
  -- condition: one angle is 20% larger than 90 degrees
  let A := 1.2 * 90
  -- to simplify the problem, let's introduce the two equal smallest angles as α
  let α := (180 - A) / 2
  use α
  split
  sorry

end smallest_angle_of_isosceles_obtuse_triangle_l812_812383


namespace largest_multiple_of_9_less_than_100_l812_812283

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n < 100 ∧ n % 9 = 0 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 0 → m ≤ n :=
by
  use 99
  split
  · exact dec_trivial
  split 
  · exact dec_trivial
  intro m hm
  cases hm 
  cases hm_right 
  have h : m ≤ 99 / 1 := by norm_cast; simp only [Nat.le_div_iff_mul_le dec_trivial, mul_one, div_one]
  exact h
  sorry -- Complete the proof

end largest_multiple_of_9_less_than_100_l812_812283


namespace construct_quadrilateral_l812_812649

variables (α β ε : Real) (e f : ℝ)
constants (A B C D M : Point)
noncomputable def isQuadrilateralConstructible (A B C D M : Point) (α β ε : ℝ) (e f : ℝ) : Prop :=
 ∃ (ABCD : Quad), 
   adjacent_angles ABCD α β ∧
   diagonals ABCD e f ∧
   intersection_angle ABCD M ε

theorem construct_quadrilateral (α β ε : ℝ) (e f : ℝ) :
  isQuadrilateralConstructible A B C D M α β ε e f :=
sorry

end construct_quadrilateral_l812_812649


namespace sqrt_9_fact_over_126_eq_12_sqrt_10_l812_812011

theorem sqrt_9_fact_over_126_eq_12_sqrt_10 :
  real.sqrt (9! / 126) = 12 * real.sqrt 10 := 
sorry

end sqrt_9_fact_over_126_eq_12_sqrt_10_l812_812011


namespace arccos_1_eq_0_l812_812410

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812410


namespace arccos_one_eq_zero_l812_812425

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812425


namespace arccos_one_eq_zero_l812_812426

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812426


namespace arccos_1_eq_0_l812_812415

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812415


namespace apples_total_l812_812174

theorem apples_total (lexie_apples : ℕ) (tom_apples : ℕ) (h1 : lexie_apples = 12) (h2 : tom_apples = 2 * lexie_apples) : lexie_apples + tom_apples = 36 :=
by
  sorry

end apples_total_l812_812174


namespace problem_1_2_3_l812_812046

open Real

variables (a b : ℝ^3) (x : ℝ)

-- Given conditions
constants (mag_a mag_b : ℝ) (theta : ℝ)
axiom mag_a_is_2 : mag_a = 2
axiom mag_b_is_3 : mag_b = 3
axiom angle_is_120_deg : theta = 120

-- Questions stated as hypotheses and correct answers to be proved:
theorem problem_1_2_3
  (h₀ : ‖a‖ = mag_a)
  (h₁ : ‖b‖ = mag_b)
  (h₂ : cos (θ : ℝ) = -1/2) :
  
  -- Question 1
  (a • b = -3) ∧
  (‖3 • a + 2 • b‖ = 6) ∧

  -- Question 2
  (inner (x • a - b) (a + 3 • b) = 0 → x = -24 / 5) ∧

  -- Question 3
  (cos (θ : ℝ) = 1/2 → angle (a, 3 • a + 2 • b) = 60) :=
by { sorry }

end problem_1_2_3_l812_812046


namespace max_ellipse_area_in_common_region_l812_812335

-- Define the conditions
def condition1 (p x y : ℝ) : Prop := 2 * p * y = x^2 - p^2
def condition2 (p x y : ℝ) : Prop := 2 * p * y = p^2 - x^2

-- Define the statement for the maximum ellipse area
theorem max_ellipse_area_in_common_region (p : ℝ) :
  (∃ a b : ℝ, 
    a = sqrt(2 / 3) * p ∧ 
    b = sqrt(2) / 3 * p ∧
    (∀ x y, condition1 p x y ∧ condition2 p x y → 2 * p * y ≤ p^2 - x^2 ∧ x^2 - p^2 ≤ 2 * p * y) →
    π * a * b = (2 * π / 3 / sqrt 3) * p^2) :=
sorry

end max_ellipse_area_in_common_region_l812_812335


namespace eq1_ellipse_eq2_hyperbola_n_ge_abs_m_l812_812159

open Complex

variables {m n : ℝ} (z : ℂ)

def eq1 (m n : ℝ) (z : ℂ) := abs (z + n * I) + abs (z - m * I) = n
def eq2 (m n : ℝ) (z : ℂ) := abs (z + n * I) - abs (z - m * I) = -m

theorem eq1_ellipse (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) : n > 0 → ∃ E : set ℂ, ∀ z, z ∈ E ↔ eq1 m n z := sorry

theorem eq2_hyperbola (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) : m < 0 → ∃ H : set ℂ, ∀ z, z ∈ H ↔ eq2 m n z := sorry

theorem n_ge_abs_m (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (h_eq1 : eq1 m n z) :
  n ≥ abs m := sorry

end eq1_ellipse_eq2_hyperbola_n_ge_abs_m_l812_812159


namespace min_sum_xy_l812_812034

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y + x * y = 3) : x + y ≥ 2 :=
by
  sorry

end min_sum_xy_l812_812034


namespace _l812_812794

noncomputable def tetrahedron_sine_theorem 
  (a b : ℝ)
  (α β : ℝ)
  (S1 S2 S3 S4 V : ℝ) 
  (h₁ : V = (2 * S1 * S2 * sin α) / (3 * a))
  (h₂ : V = (2 * S3 * S4 * sin β) / (3 * b)) 
  : ℝ :=
  have h3 := V * V,
  let lhs := (a * b) / (sin α * sin β),
  let rhs := (4 * S1 * S2 * S3 * S4) / (9 * h3),
  lhs = rhs

end _l812_812794


namespace appropriate_sampling_method_l812_812848

-- Defining the sizes of the boxes
def size_large : ℕ := 120
def size_medium : ℕ := 60
def size_small : ℕ := 20

-- Define a sample size
def sample_size : ℕ := 25

-- Define the concept of appropriate sampling method as being equivalent to stratified sampling in this context
theorem appropriate_sampling_method : 3 > 0 → sample_size > 0 → size_large = 120 ∧ size_medium = 60 ∧ size_small = 20 → 
("stratified sampling" = "stratified sampling") :=
by 
  sorry

end appropriate_sampling_method_l812_812848


namespace quadratic_one_root_greater_than_two_other_less_than_two_l812_812580

theorem quadratic_one_root_greater_than_two_other_less_than_two (m : ℝ) :
  (∀ x y : ℝ, x^2 + (2 * m - 3) * x + m - 150 = 0 ∧ x > 2 ∧ y < 2) →
  m > 5 :=
by
  sorry

end quadratic_one_root_greater_than_two_other_less_than_two_l812_812580


namespace arccos_one_eq_zero_l812_812534

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812534


namespace three_digit_number_with_units5_and_hundreds3_divisible_by_9_l812_812869

theorem three_digit_number_with_units5_and_hundreds3_divisible_by_9 :
  ∃ n : ℕ, ∃ x : ℕ, n = 305 + 10 * x ∧ (n % 9) = 0 ∧ n = 315 := by
sorry

end three_digit_number_with_units5_and_hundreds3_divisible_by_9_l812_812869


namespace arccos_one_eq_zero_l812_812528

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812528


namespace complex_number_twelfth_root_of_unity_l812_812985

noncomputable def tan_pi_six : ℂ := complex.tan (real.pi / 6)

theorem complex_number_twelfth_root_of_unity :
  (tan_pi_six + complex.I) / (tan_pi_six - complex.I) = complex.cos (4 * real.pi / 6) + complex.I * complex.sin (4 * real.pi / 6) :=
by
  sorry

end complex_number_twelfth_root_of_unity_l812_812985


namespace solution_set_contains_four_positive_integers_l812_812019

theorem solution_set_contains_four_positive_integers (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*(m+1)*x + 4*m ≤ 0 → x ∈ ℕ ∧ x > 0 ∧ x ≤ 5) ↔ (5/2 ≤ m ∧ m < 3) := 
sorry

end solution_set_contains_four_positive_integers_l812_812019


namespace total_students_l812_812234

-- Definitions from the conditions
def ratio_boys_to_girls (B G : ℕ) : Prop := B / G = 1 / 2
def girls_count := 60

-- The main statement to prove
theorem total_students (B G : ℕ) (h1 : ratio_boys_to_girls B G) (h2 : G = girls_count) : B + G = 90 := sorry

end total_students_l812_812234


namespace arccos_1_eq_0_l812_812411

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812411


namespace multiple_of_32_only_l812_812752

theorem multiple_of_32_only :
  let y = 64 + 96 + 192 + 256 + 352 + 480 + 4096 + 8192 in
  (y % 32 = 0) ∧ (y % 64 ≠ 0) ∧ (y % 128 ≠ 0) ∧ (y % 256 ≠ 0) :=
by
  let y := 64 + 96 + 192 + 256 + 352 + 480 + 4096 + 8192
  have h₁ : y % 32 = 0 := by sorry
  have h₂ : y % 64 ≠ 0 := by sorry
  have h₃ : y % 128 ≠ 0 := by sorry
  have h₄ : y % 256 ≠ 0 := by sorry
  exact ⟨h₁, h₂, h₃, h₄⟩

end multiple_of_32_only_l812_812752


namespace arccos_one_eq_zero_l812_812523

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812523


namespace arccos_one_eq_zero_l812_812457

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812457


namespace largest_multiple_of_9_less_than_100_l812_812280

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n < 100 ∧ n % 9 = 0 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 0 → m ≤ n :=
by
  use 99
  split
  · exact dec_trivial
  split 
  · exact dec_trivial
  intro m hm
  cases hm 
  cases hm_right 
  have h : m ≤ 99 / 1 := by norm_cast; simp only [Nat.le_div_iff_mul_le dec_trivial, mul_one, div_one]
  exact h
  sorry -- Complete the proof

end largest_multiple_of_9_less_than_100_l812_812280


namespace number_of_y_axis_returns_l812_812201

-- Definitions based on conditions
noncomputable def unit_length : ℝ := 0.5
noncomputable def diagonal_length : ℝ := Real.sqrt 2 * unit_length
noncomputable def pen_length_cm : ℝ := 8000 * 100 -- converting meters to cm
noncomputable def circle_length (n : ℕ) : ℝ := ((3 + Real.sqrt 2) * n ^ 2 + 2 * n) * unit_length

-- The main theorem
theorem number_of_y_axis_returns : ∃ n : ℕ, circle_length n ≤ pen_length_cm ∧ circle_length (n+1) > pen_length_cm :=
sorry

end number_of_y_axis_returns_l812_812201


namespace total_GDP_l812_812706

noncomputable def GDP_first_quarter : ℝ := 232
noncomputable def GDP_fourth_quarter : ℝ := 241

theorem total_GDP (x y : ℝ) (h1 : GDP_first_quarter < x)
                  (h2 : x < y) (h3 : y < GDP_fourth_quarter)
                  (h4 : (x + y) / 2 = (GDP_first_quarter + x + y + GDP_fourth_quarter) / 4) :
  GDP_first_quarter + x + y + GDP_fourth_quarter = 946 :=
by
  sorry

end total_GDP_l812_812706


namespace largest_multiple_of_9_less_than_100_l812_812282

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n < 100 ∧ n % 9 = 0 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 0 → m ≤ n :=
by
  use 99
  split
  · exact dec_trivial
  split 
  · exact dec_trivial
  intro m hm
  cases hm 
  cases hm_right 
  have h : m ≤ 99 / 1 := by norm_cast; simp only [Nat.le_div_iff_mul_le dec_trivial, mul_one, div_one]
  exact h
  sorry -- Complete the proof

end largest_multiple_of_9_less_than_100_l812_812282


namespace transformed_roots_equation_l812_812071

theorem transformed_roots_equation (α β : ℂ) (h1 : 3 * α^2 + 2 * α + 1 = 0) (h2 : 3 * β^2 + 2 * β + 1 = 0) :
  ∃ (y : ℂ), (y - (3 * α + 2)) * (y - (3 * β + 2)) = y^2 + 4 := 
sorry

end transformed_roots_equation_l812_812071


namespace altitudes_of_acute_triangle_l812_812711

variables {A B C D E F : Type} [is_acute_triangle : ∀ (A B C : Type) := sorry]

/-- Representation of a triangle. -/
structure Triangle (A B C : Type) :=
  (acute : A <| acute_angle B C)
  (D : B × C)
  (E : C × A)
  (F : A × B)
  (ratio1 : D.1 / E.1 = A / B)
  (ratio2 : E.1 / F.1 = B / C)
  (ratio3 : F.1 / D.1 = C / A)

/-- Statement in Lean 4: Given an acute triangle ABC with points D, E, F on sides BC, CA, AB
in certain ratios, prove that AD, BE, CF are altitudes. -/
theorem altitudes_of_acute_triangle 
  (triangle : Triangle A B C) 
  (h1 : triangle.D.1 / triangle.E.1 = triangle.D.2 / triangle.E.2)
  (h2 : triangle.E.1 / triangle.F.1 = triangle.E.2 / triangle.F.2)
  (h3 : triangle.F.1 / triangle.D.1 = triangle.F.2 / triangle.D.2)
  : (AD ⊥ BC) ∧ (BE ⊥ CA) ∧ (CF ⊥ AB) :=
sorry

end altitudes_of_acute_triangle_l812_812711


namespace sqrt_9_fact_over_126_eq_12_sqrt_10_l812_812008

theorem sqrt_9_fact_over_126_eq_12_sqrt_10 :
  real.sqrt (9! / 126) = 12 * real.sqrt 10 := 
sorry

end sqrt_9_fact_over_126_eq_12_sqrt_10_l812_812008


namespace Xiaoming_coins_l812_812316

theorem Xiaoming_coins (x : ℕ) :
  (x ≥ 2) ∧ (0.5 * (15 - x) + x < 10) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by
  -- Sorry is used to skip the detailed proof steps
  sorry

end Xiaoming_coins_l812_812316


namespace total_cookies_l812_812317

theorem total_cookies (bags : ℕ) (cookies_per_bag : ℕ) (candies : ℕ) (total_cookies : ℕ) :
  bags = 26 → cookies_per_bag = 2 → candies = 15 → total_cookies = bags * cookies_per_bag → total_cookies = 52 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2]
  simp at h4
  exact h4

end total_cookies_l812_812317


namespace midpoint_of_tangent_l812_812927

variables (a b : ℝ)
noncomputable def midpoint (x₁ y₁ x_A y_A x_B y_B : ℝ) : Prop :=
(x₁ = (x_A + x_B) / 2) ∧ (y₁ = (y_A + y_B) / 2)

noncomputable def hyperbola (x₁ y₁ : ℝ) : Prop :=
(x₁^2)/(a^2) - (y₁^2)/(b^2) = 1 

noncomputable def tangent (x₁ y₁ x y : ℝ) : Prop :=
(x₁ * x)/(a^2) - (y₁ * y)/(b^2) = 1 

noncomputable def asymptote_pos (x y : ℝ) : Prop :=
y = (b / a) * x

noncomputable def asymptote_neg (x y : ℝ) : Prop :=
y = -(b / a) * x

theorem midpoint_of_tangent
  (x₁ y₁ : ℝ)
  (H_hyperbola : hyperbola a b x₁ y₁) :
  ∃ x_A y_A x_B y_B,
    (tangent a b x₁ y₁ x_A y_A ∧ asymptote_pos a b x_A y_A) ∧
    (tangent a b x₁ y₁ x_B y_B ∧ asymptote_neg a b x_B y_B) ∧
    midpoint x₁ y₁ x_A y_A x_B y_B :=
sorry

end midpoint_of_tangent_l812_812927


namespace platform_length_l812_812900
-- Importing the math library for necessary support

-- The theorem statement
theorem platform_length 
  (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ)
  (h1 : train_length = 250)
  (h2 : train_speed_kmph = 100)
  (h3 : crossing_time = 30) : 
  ∃ P : ℝ, P = 583.4 :=
by 
  -- Conversion from kmph to m/s
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  -- Total distance covered by the train while crossing the platform
  let distance_covered := train_speed_mps * crossing_time
  -- Equation relating train length, platform length, and distance covered
  let P := distance_covered - train_length
  -- Prove the platform length is 583.4 meters
  have result : P = 583.4 := sorry
  exact ⟨P, result⟩

-- Adding noncomputable before def only when necessary
noncomputable def main := theorem platform_length 

end platform_length_l812_812900


namespace largest_multiple_of_9_less_than_100_l812_812294

theorem largest_multiple_of_9_less_than_100 : ∃ n, n < 100 ∧ n % 9 = 0 ∧ (∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n) :=
begin
  use 99,
  split,
  -- Prove the number is less than 100
  { linarith, },
  split,
  -- Prove the number is a multiple of 9
  { norm_num, },
  -- Prove it is the largest
  { intros m hlt hdiv,
    linarith [hlt, hdiv],
    norm_num at hdiv,
    cases hdiv,
    exact hdiv_left, },
end

end largest_multiple_of_9_less_than_100_l812_812294


namespace arccos_one_eq_zero_l812_812552

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812552


namespace sugar_at_home_l812_812175

-- Definitions based on conditions
def bags_of_sugar := 2
def cups_per_bag := 6
def cups_for_batter_per_12_cupcakes := 1
def cups_for_frosting_per_12_cupcakes := 2
def dozens_of_cupcakes := 5

-- Calculation of total sugar needed and bought, in terms of definitions
def total_cupcakes := dozens_of_cupcakes * 12
def total_sugar_needed_for_batter := (total_cupcakes / 12) * cups_for_batter_per_12_cupcakes
def total_sugar_needed_for_frosting := dozens_of_cupcakes * cups_for_frosting_per_12_cupcakes
def total_sugar_needed := total_sugar_needed_for_batter + total_sugar_needed_for_frosting
def total_sugar_bought := bags_of_sugar * cups_per_bag

-- The statement to be proven in Lean
theorem sugar_at_home : total_sugar_needed - total_sugar_bought = 3 := by
  sorry

end sugar_at_home_l812_812175


namespace Carla_more_miles_than_Daniel_after_5_hours_l812_812214

theorem Carla_more_miles_than_Daniel_after_5_hours (Carla_distance : ℝ) (Daniel_distance : ℝ) (h_Carla : Carla_distance = 100) (h_Daniel : Daniel_distance = 75) : 
  Carla_distance - Daniel_distance = 25 := 
by
  sorry

end Carla_more_miles_than_Daniel_after_5_hours_l812_812214


namespace alpha_sq_gt_beta_sq_l812_812687

theorem alpha_sq_gt_beta_sq (α β : ℝ) (h1 : α ∈ Icc (-π/2) (π/2)) (h2 : β ∈ Icc (-π/2) (π/2))
    (h3 : α * sin α - β * sin β > 0) : α^2 > β^2 := 
sorry

end alpha_sq_gt_beta_sq_l812_812687


namespace range_of_a_l812_812692

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x + 3

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Ioo (a-1) (a+1), 4*x - 1/x = 0) ↔ 1 ≤ a ∧ a < 3/2 :=
sorry

end range_of_a_l812_812692


namespace July_husband_age_l812_812871

namespace AgeProof

variable (HannahAge JulyAge HusbandAge : ℕ)

def double_age_condition (hannah_age : ℕ) (july_age : ℕ) : Prop :=
  hannah_age = 2 * july_age

def twenty_years_later (current_age : ℕ) : ℕ :=
  current_age + 20

def two_years_older (age : ℕ) : ℕ :=
  age + 2

theorem July_husband_age :
  ∃ (hannah_age july_age : ℕ), double_age_condition hannah_age july_age ∧
    twenty_years_later hannah_age = 26 ∧
    twenty_years_later july_age = 23 ∧
    two_years_older (twenty_years_later july_age) = 25 :=
by
  sorry
end AgeProof

end July_husband_age_l812_812871


namespace equation_of_W_rectangle_perimeter_gt_3sqrt3_l812_812734

namespace MathProof

/- Defining the conditions and the questions -/
def point (P : ℝ × ℝ) := P.fst
def P (x y : ℝ) := (x, y)

def dist_x_axis (y : ℝ) := |y|
def dist_to_point (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + (y - 1/2)^2)

-- Condition: The distance from point P to the x-axis equals distance to (0, 1/2)
def condition_P (x y : ℝ) : Prop :=
  dist_x_axis y = dist_to_point x y

-- Locus W
def W_eq (x : ℝ) : ℝ := x^2 + 1/4

/- Proving the statements -/
theorem equation_of_W (x y : ℝ) (h : condition_P x y): 
  y = W_eq x :=
sorry

-- Perimeter of rectangle ABCD
structure rectangle :=
(A : ℝ × ℝ)
(B : ℝ × ℝ)
(C : ℝ × ℝ)
(D : ℝ × ℝ)

def on_W (P : ℝ × ℝ) : Prop :=
  P.snd = W_eq P.fst

def perimeter (rect : rectangle) : ℝ :=
  let d1 := Real.sqrt ((rect.A.fst - rect.B.fst)^2 + (rect.A.snd - rect.B.snd)^2)
  let d2 := Real.sqrt ((rect.B.fst - rect.C.fst)^2 + (rect.B.snd - rect.C.snd)^2)
  let d3 := Real.sqrt ((rect.C.fst - rect.D.fst)^2 + (rect.C.snd - rect.D.snd)^2)
  let d4 := Real.sqrt ((rect.D.fst - rect.A.fst)^2 + (rect.D.snd - rect.A.snd)^2)
  d1 + d2 + d3 + d4

theorem rectangle_perimeter_gt_3sqrt3 
(rect : rectangle) 
(hA : on_W rect.A) 
(hB : on_W rect.B) 
(hC : on_W rect.C) :
  perimeter rect > 3 * Real.sqrt 3 :=
sorry

end MathProof

end equation_of_W_rectangle_perimeter_gt_3sqrt3_l812_812734


namespace square_root_of_factorial_division_l812_812004

theorem square_root_of_factorial_division :
  (sqrt (9! / 126) = 24 * sqrt 5) :=
sorry

end square_root_of_factorial_division_l812_812004


namespace arccos_one_eq_zero_l812_812442

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812442


namespace arccos_one_eq_zero_l812_812511

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812511


namespace m_minus_n_eq_2_l812_812230

theorem m_minus_n_eq_2 (m n : ℕ) (h1 : ∃ x : ℕ, m = 101 * x) (h2 : ∃ y : ℕ, n = 63 * y) (h3 : m + n = 2018) : m - n = 2 :=
sorry

end m_minus_n_eq_2_l812_812230


namespace math_problem_l812_812885

-- Defining the intermediate values for clarity
def a : ℝ := 32400 * (4^3)
def b : ℝ := 3 * real.sqrt 343
def c : ℝ := ((49^2 * 11)^4 : ℝ)
def d : ℝ := 2 * real.sqrt c / (25^3)

theorem math_problem :
  (a / b / 18 / (7^3 * 10)) / d = 0.00005366 := 
by
  sorry

end math_problem_l812_812885


namespace cos_sufficient_not_necessary_sin_l812_812638

theorem cos_sufficient_not_necessary_sin
  (A : ℝ)
  (h1 : 0 < A ∧ A < real.pi) :
  (cos A = 1 / 2 → sin A = real.sqrt 3 / 2) ∧ 
  ((sin A = real.sqrt 3 / 2 → (cos A = 1 / 2 ∨ cos A = -1 / 2)) ∧ 
   ¬ (sin A = real.sqrt 3 / 2 → cos A = 1 / 2)) :=
by
  sorry

end cos_sufficient_not_necessary_sin_l812_812638


namespace cylinder_surface_area_l812_812353

theorem cylinder_surface_area (h : ℝ) (c : ℝ) (r : ℝ) (S : ℝ) (h_eq : h = 2) (c_eq : c = 2 * Real.pi) (r_eq : r = c / (2 * Real.pi)) : S = 6 * Real.pi :=
by
  rw [h_eq, c_eq] at r_eq
  simp [r_eq] at r_eq
  sorry

end cylinder_surface_area_l812_812353


namespace arccos_one_eq_zero_l812_812503

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812503


namespace arccos_one_eq_zero_l812_812545

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812545


namespace total_athletes_l812_812944

theorem total_athletes (g : ℕ) (p : ℕ)
  (h₁ : g = 7)
  (h₂ : p = 5)
  (h₃ : 3 * (g + p - 1) = 33) : 
  3 * (g + p - 1) = 33 :=
sorry

end total_athletes_l812_812944


namespace largest_multiple_of_9_less_than_100_l812_812288

theorem largest_multiple_of_9_less_than_100 : ∃ x : ℕ, 9 * x < 100 ∧ ∀ y : ℕ, 9 * y < 100 → y ≤ x :=
by
  exists 11
  split
  · linarith
  · intro y h
    have : 9 * y < 100 := h
    calc
      y ≤ floor (11.11) : by linarith

end largest_multiple_of_9_less_than_100_l812_812288


namespace equation_of_W_rectangle_perimeter_gt_3sqrt3_l812_812735

namespace MathProof

/- Defining the conditions and the questions -/
def point (P : ℝ × ℝ) := P.fst
def P (x y : ℝ) := (x, y)

def dist_x_axis (y : ℝ) := |y|
def dist_to_point (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + (y - 1/2)^2)

-- Condition: The distance from point P to the x-axis equals distance to (0, 1/2)
def condition_P (x y : ℝ) : Prop :=
  dist_x_axis y = dist_to_point x y

-- Locus W
def W_eq (x : ℝ) : ℝ := x^2 + 1/4

/- Proving the statements -/
theorem equation_of_W (x y : ℝ) (h : condition_P x y): 
  y = W_eq x :=
sorry

-- Perimeter of rectangle ABCD
structure rectangle :=
(A : ℝ × ℝ)
(B : ℝ × ℝ)
(C : ℝ × ℝ)
(D : ℝ × ℝ)

def on_W (P : ℝ × ℝ) : Prop :=
  P.snd = W_eq P.fst

def perimeter (rect : rectangle) : ℝ :=
  let d1 := Real.sqrt ((rect.A.fst - rect.B.fst)^2 + (rect.A.snd - rect.B.snd)^2)
  let d2 := Real.sqrt ((rect.B.fst - rect.C.fst)^2 + (rect.B.snd - rect.C.snd)^2)
  let d3 := Real.sqrt ((rect.C.fst - rect.D.fst)^2 + (rect.C.snd - rect.D.snd)^2)
  let d4 := Real.sqrt ((rect.D.fst - rect.A.fst)^2 + (rect.D.snd - rect.A.snd)^2)
  d1 + d2 + d3 + d4

theorem rectangle_perimeter_gt_3sqrt3 
(rect : rectangle) 
(hA : on_W rect.A) 
(hB : on_W rect.B) 
(hC : on_W rect.C) :
  perimeter rect > 3 * Real.sqrt 3 :=
sorry

end MathProof

end equation_of_W_rectangle_perimeter_gt_3sqrt3_l812_812735


namespace f_minimum_positive_period_and_max_value_l812_812225

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.cos x) + (1 + (Real.tan x)^2) * (Real.cos x)^2

theorem f_minimum_positive_period_and_max_value :
  (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ π) ∧ (∃ M, ∀ x : ℝ, f x ≤ M ∧ M = 3 / 2) := by
  sorry

end f_minimum_positive_period_and_max_value_l812_812225


namespace largest_multiple_of_9_less_than_100_l812_812286

theorem largest_multiple_of_9_less_than_100 : ∃ x : ℕ, 9 * x < 100 ∧ ∀ y : ℕ, 9 * y < 100 → y ≤ x :=
by
  exists 11
  split
  · linarith
  · intro y h
    have : 9 * y < 100 := h
    calc
      y ≤ floor (11.11) : by linarith

end largest_multiple_of_9_less_than_100_l812_812286


namespace find_function_value_l812_812166

noncomputable def f : ℝ → ℝ
| x := if h : 0 < x ∧ x ≤ 9 then Real.log x / Real.log 3 else f (x - 4)

theorem find_function_value :
  f 13 + 2 * f (1 / 3) = 0 :=
sorry

end find_function_value_l812_812166


namespace prove_k_range_l812_812670

open Real

def vector_dot (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def vector_norm (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def vector_cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  vector_dot v1 v2 / (vector_norm v1 * vector_norm v2)

def obtuse_angle_condition (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  vector_dot v1 v2 < 0

def problem_conditions (m k : ℝ) : Prop :=
  let a := (1, 1, 0) in
  let b := (m, 0, 2) in
  vector_cos_angle a b = -sqrt 10 / 10 ∧
  obtuse_angle_condition (1 + k * -1, 1 + k * 0, 0 + k * 2) (2 * 1 + -1, 2 * 1 + 0, 2 * 0 + 2)

theorem prove_k_range (m k : ℝ) (h : problem_conditions m k) : k < -1 := sorry

end prove_k_range_l812_812670


namespace equation_of_W_perimeter_of_rectangle_ABCD_l812_812718

-- Definition for the locus W
def locusW (x y : ℝ) : Prop :=
  y = x^2 + 1/4

-- Proof for the equation of W
theorem equation_of_W (x y : ℝ) :
  (abs y = sqrt (x^2 + (y - 1/2)^2)) ↔ locusW x y := 
sorry

-- Prove the perimeter of rectangle ABCD
theorem perimeter_of_rectangle_ABCD (A B C D : ℝ × ℝ) :
  (locusW A.1 A.2 ∧ locusW B.1 B.2 ∧ locusW C.1 C.2 ∧ (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) ∧
  (B.1 - C.1) * (D.2 - C.2) = (D.1 - B.1) * (C.2 - B.2) ∧ (D.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (D.2 - A.2)) →
  (4 < 3 * sqrt 3) := 
sorry

end equation_of_W_perimeter_of_rectangle_ABCD_l812_812718


namespace perpendicular_iff_collinear_l812_812190

-- Definitions used in the conditions
variables {S N T O M : Type}
variables {OM MN : Line}
variables {collinear : S → N → T → Prop}
variables {perpendicular : OM → MN → Prop}

-- Statement of the problem
theorem perpendicular_iff_collinear :
  perpendicular OM MN ↔ collinear S N T :=
begin
  sorry
end

end perpendicular_iff_collinear_l812_812190


namespace apples_fraction_of_pears_l812_812322

variables (A O P : ℕ)

-- Conditions
def oranges_condition := O = 3 * A
def pears_condition := P = 4 * O

-- Statement we need to prove
theorem apples_fraction_of_pears (A O P : ℕ) (h1 : O = 3 * A) (h2 : P = 4 * O) : (A : ℚ) / P = 1 / 12 :=
by
  sorry

end apples_fraction_of_pears_l812_812322


namespace find_tank_capacity_l812_812313

-- Define the overall problem
def tank_capacity (capacity : ℝ) : Prop := 
  capacity * 0.4 + 36 = capacity * 0.9

-- State the theorem
theorem find_tank_capacity : 
  ∃ capacity : ℝ, tank_capacity capacity ∧ capacity = 72 := 
by 
  use 72
  simp [tank_capacity]
  norm_num
  sorry -- This is to indicate a skipped proof as specified

end find_tank_capacity_l812_812313


namespace apples_total_l812_812173

theorem apples_total (lexie_apples : ℕ) (tom_apples : ℕ) (h1 : lexie_apples = 12) (h2 : tom_apples = 2 * lexie_apples) : lexie_apples + tom_apples = 36 :=
by
  sorry

end apples_total_l812_812173


namespace ratio_of_radii_l812_812886

open Real

variables {R1 R2 : ℝ} {O1 O2 B E C : Point}

noncomputable def condition1 := (parallel ell1 ell2) ∧ (tangent ell1 omega1 A) ∧ (tangent ell2 omega1 B)
noncomputable def condition2 := (tangent ell1 omega2 D) ∧ (intersects omega2 ell2 B E) ∧ (intersects omega2 omega1 C)
noncomputable def condition3 := (between O2 ell1 ell2)
noncomputable def condition4 := (area_BO1CO2 / area_O2BE = 5)
noncomputable def condition5 := (BD = 5)

theorem ratio_of_radii (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) : 
  (R2 / R1 = 9 / 5) :=
sorry

end ratio_of_radii_l812_812886


namespace expected_value_sum_marble_numbers_l812_812246

theorem expected_value_sum_marble_numbers : 
  let marbles := [1, 2, 3, 4, 5, 6, 7]
  let all_combinations := (marbles.combinations 3).toList
  let total_combinations := all_combinations.length
  let total_sum := all_combinations.sum
  total_sum / total_combinations = 12 :=
by
  sorry

end expected_value_sum_marble_numbers_l812_812246


namespace cannot_fold_to_polyhedron_l812_812956

theorem cannot_fold_to_polyhedron
  (squares : Type) [fintype squares] [decidable_eq squares]
  (seven_squares : finset squares)
  (missing_line : Prop) :
  seven_squares.card = 7 →
  missing_line →
  ¬ (∃ (polyhedron : Type) [simple_polyhedron polyhedron], 
    foldable_into seven_squares polyhedron) := 
by sorry

end cannot_fold_to_polyhedron_l812_812956


namespace rain_on_tuesdays_l812_812180

theorem rain_on_tuesdays (x : ℝ) (monday_rain total_monday_rain tuesday_rain total_tuesday_rain : ℝ) 
  (hm : monday_rain = 1.5) 
  (h_monday: total_monday_rain = 7 * monday_rain) 
  (ht : total_tuesday_rain = total_monday_rain + 12) 
  (h_tuesday: total_tuesday_rain = 9 * x) : 
  x = 2.5 := 
by 
{
  -- Convert Monday rain into a total
  rw hm at h_monday,
  have h1 : total_monday_rain = 7 * 1.5 := by exact h_monday,
  -- Calculate Monday total rain
  have h2 : total_monday_rain = 10.5 := by linarith,
  -- Calculate Tuesday total rain
  rw h2 at ht,
  have h3 : total_tuesday_rain = 10.5 + 12 := by exact ht,
  -- Calculate Tuesday total in numerical form
  have h4 : total_tuesday_rain = 22.5 := by linarith,
  -- Calculate each Tuesday rain amount
  rw h4 at h_tuesday,
  have h5 : 22.5 = 9 * x := by exact h_tuesday,
  -- Solve for x
  have h6 : x = 22.5 / 9 := by linarith,
  -- x should be 2.5
  have h7 : x = 2.5 := by linarith,
  exact h7,
}

end rain_on_tuesdays_l812_812180


namespace hyperbola_foci_coordinates_l812_812212

theorem hyperbola_foci_coordinates :
  ∀ (x y : ℝ), 
  (x^2 / 7 - y^2 / 9 = 1) →
  ∃_(c : ℝ), c = 4 ∧ ((x = c ∧ y = 0) ∨ (x = -c ∧ y = 0)) :=
by
  sorry

end hyperbola_foci_coordinates_l812_812212


namespace intersection_complement_A_B_eq_l812_812618

noncomputable def real_univ_set : Set ℝ := Set.univ

noncomputable def set_A : Set ℝ := { x | (x + 2) / (3 - x) ≥ 0 }

noncomputable def set_B : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }

noncomputable def complement_A : Set ℝ := real_univ_set \ set_A

noncomputable def intersection_complement_A_B : Set ℝ := complement_A ∩ set_B

theorem intersection_complement_A_B_eq :
  intersection_complement_A_B = (Icc 3 4) := by
  sorry

end intersection_complement_A_B_eq_l812_812618


namespace min_PA_PC_n_equals_zero_l812_812776

def is_on_parabola (a b c x y : ℝ) : Prop := y = a * x^2 + b * x + c

def axis_of_symmetry (a b : ℝ) : ℝ := -b / (2 * a)

def point_on_axis_of_symmetry (x axis : ℝ) : Prop := x = axis

theorem min_PA_PC_n_equals_zero
  (a b c : ℝ)
  (A B C : (ℝ × ℝ))
  (P : (ℝ × ℝ))
  (h_axis : axis_of_symmetry a b = 2)
  (hA : is_on_parabola a b c (A.fst) (A.snd))
  (hB : is_on_parabola a b c (B.fst) (B.snd))
  (hC : is_on_parabola a b c (C.fst) (C.snd))
  (h_axis_points : A = (-1, -3) ∧ B = (4, 2) ∧ C = (0, 2))
  (hP_on_axis : point_on_axis_of_symmetry (P.fst) 2) :
  P.snd = 0 :=
by
  sorry

end min_PA_PC_n_equals_zero_l812_812776


namespace problem_statement_l812_812993

noncomputable def sum_of_solutions : ℂ :=
  let f (x : ℂ) := 2 ^ (x ^ 2 - 3 * x - 2) - 8 ^ (x - 5)
  let solutions := {x : ℂ | f x = 0}
  solutions.sum

theorem problem_statement : sum_of_solutions = 6 := 
  sorry

end problem_statement_l812_812993


namespace john_spent_half_on_fruits_and_vegetables_l812_812943

theorem john_spent_half_on_fruits_and_vegetables (M : ℝ) (F : ℝ) 
  (spent_on_meat : ℝ) (spent_on_bakery : ℝ) (spent_on_candy : ℝ) :
  (M = 120) → 
  (spent_on_meat = (1 / 3) * M) → 
  (spent_on_bakery = (1 / 10) * M) → 
  (spent_on_candy = 8) → 
  (F * M + spent_on_meat + spent_on_bakery + spent_on_candy = M) → 
  (F = 1 / 2) := 
  by 
    sorry

end john_spent_half_on_fruits_and_vegetables_l812_812943


namespace coefficient_sum_of_polynomial_l812_812025

theorem coefficient_sum_of_polynomial 
  (a : Fin 8 → ℤ)
  (h : (3 * x - 1)^7 = ∑ i in Finset.range 8, a i * x^(7 - i)) :
  a 0 + a 2 + a 4 + a 6 = 8256 :=
sorry

end coefficient_sum_of_polynomial_l812_812025


namespace find_z_coordinate_l812_812358

-- Define points A and B
def A : ℝ × ℝ × ℝ := (3, 3, 2)
def B : ℝ × ℝ × ℝ := (6, 2, -1)

-- Define the direction vector
def direction_vector (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

-- Parameterize the line
def line (A d : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (A.1 + d.1 * t, A.2 + d.2 * t, A.3 + d.3 * t)

-- Given t where y (coordinate 2) is 1, find the z (coordinate 3) value
theorem find_z_coordinate : ∃ (z : ℝ), 
  ∃ (t : ℝ), 3 - t = 1 ∧ z = 2 - 3 * t := by
  sorry

end find_z_coordinate_l812_812358


namespace bob_walking_speed_l812_812378

theorem bob_walking_speed :
  ∀ (d_a_b d_alice_walk d_total : ℕ) (v_alice : ℕ) (t_gap t_total t_bob : ℕ) (v_bob : ℕ),
  d_a_b = 41 →
  d_alice_walk = 25 →
  d_total = 41 - 25 →
  v_alice = 5 →
  t_gap = 1 →
  t_total = d_alice_walk / v_alice →
  t_bob = t_total - t_gap →
  v_bob = d_total / t_bob →
  v_bob = 4 :=
by
  intros d_a_b d_alice_walk d_total v_alice t_gap t_total t_bob v_bob
  assume h_dab : d_a_b = 41,
  assume h_alice_walk : d_alice_walk = 25,
  assume h_total : d_total = 41 - 25,
  assume h_valice : v_alice = 5,
  assume h_gap : t_gap = 1,
  assume h_total_time : t_total = d_alice_walk / v_alice,
  assume h_bob_time : t_bob = t_total - t_gap,
  assume h_bob_speed : v_bob = d_total / t_bob,
  sorry

end bob_walking_speed_l812_812378


namespace least_k_for_divisibility_l812_812091

theorem least_k_for_divisibility (k : ℕ) : (k ^ 4) % 1260 = 0 ↔ k ≥ 210 :=
sorry

end least_k_for_divisibility_l812_812091


namespace largest_multiple_of_9_less_than_100_l812_812261

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l812_812261


namespace julie_read_pages_tomorrow_l812_812137

-- Definitions based on conditions
def total_pages : ℕ := 120
def pages_read_yesterday : ℕ := 12
def pages_read_today : ℕ := 2 * pages_read_yesterday

-- Problem statement to prove Julie should read 42 pages tomorrow
theorem julie_read_pages_tomorrow :
  let total_read := pages_read_yesterday + pages_read_today in
  let remaining_pages := total_pages - total_read in
  let pages_tomorrow := remaining_pages / 2 in
  pages_tomorrow = 42 :=
by
  sorry

end julie_read_pages_tomorrow_l812_812137


namespace largest_multiple_of_9_less_than_100_l812_812275

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ ∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n :=
begin
  use 99,
  split,
  { norm_num }, -- 99 < 100
  split,
  { norm_num }, -- 99 % 9 = 0
  { intros m hm,
    obtain ⟨k, rfl⟩ := nat.exists_eq_mul_right_of_dvd (dvd_of_mod_eq_zero hm.2),
    rw [mul_comm, nat.mul_lt_mul_iff_left (by norm_num : 0 < 9)],
    norm_num,
    exact nat.succ_le_iff.mpr (le_of_lt hm.1), }
end

end largest_multiple_of_9_less_than_100_l812_812275


namespace largest_multiple_of_9_less_than_100_l812_812269

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l812_812269


namespace isosceles_triangle_legs_l812_812206

theorem isosceles_triangle_legs (A B C E : ℝ) (h1 : BE_angle_bisects_base ∠B)
  (h2 : divides_perimeter BE 168 112) :
  (B = 80) ∨ (B = 105) :=
sorry

end isosceles_triangle_legs_l812_812206


namespace length_chord_GH_26_l812_812955

-- Define the circle with center O and radius OB = 15 units
definition circle_center_O_radius_15 {O B : Type} [MetricSpace O] [HasDist O B] (r : ℝ) (hr : r = 15) : Prop :=
  dist O B = r

-- Define a chord EF, which is the perpendicular bisector of OB
definition chord_EF_perpendicular_bisector_OB {O B M E F : Type} [MetricSpace O] [HasDist O B] (hEF_perp_OB : Prop) : Prop :=
  dist O M = 7.5 ∧ (∀ (x : O), x = E ∨ x = F → dist O x = 15) ∧ perp (line_through E F) (line_through O B)

-- Define another chord GH parallel and equidistant from EF by 5 units
definition chord_GH_parallel_equidistant_EF {G H E F : Type} [MetricSpace G] [HasDist G H] (hGH_par_EF : Prop) : Prop :=
  parallel (line_through G H) (line_through E F) ∧ dist G H = dist E F

-- Define the final proof statement: Chord GH has length 26 units
theorem length_chord_GH_26 (O B M E F G H : Type) [MetricSpace O] [HasDist O B] [HasDist O M] [HasDist E F] [HasDist G H]
  (r : ℝ) (hr : r = 15) (hEF_perp_OB : Prop) (hGH_par_EF : Prop) :
  circle_center_O_radius_15 r hr →
  chord_EF_perpendicular_bisector_OB hEF_perp_OB →
  chord_GH_parallel_equidistant_EF hGH_par_EF →
  dist G H = 26 :=
by
  intros,
  sorry

end length_chord_GH_26_l812_812955


namespace boundary_of_T_is_polygon_with_4_sides_l812_812774

open Set

def point_in_T (b x y : ℝ) : Prop :=
  b > 0 ∧ b ≤ x ∧ x ≤ 3 * b ∧ b ≤ y ∧ y ≤ 3 * b ∧ 
  x + y ≥ 2 * b ∧ x + 2 * b ≥ y ∧ y + 2 * b ≥ x

theorem boundary_of_T_is_polygon_with_4_sides (b : ℝ) :
  (∀ x y : ℝ, point_in_T b x y → True) → 
  {points : Set (ℝ × ℝ) | point_in_T b points.1 points.2}.finite :=
sorry

end boundary_of_T_is_polygon_with_4_sides_l812_812774


namespace at_least_one_black_ball_l812_812112

open Finset

def black_balls : Finset ℕ := {1, 2, 3, 4}
def white_balls : Finset ℕ := {5, 6}
def all_balls : Finset ℕ := black_balls ∪ white_balls

theorem at_least_one_black_ball (drawn: Finset ℕ) (h₁: drawn ⊆ all_balls) (h₂: drawn.card = 3) :
  ∃ x ∈ drawn, x ∈ black_balls := 
sorry

end at_least_one_black_ball_l812_812112


namespace altitudes_of_extriangle_l812_812771

variables {Point : Type} [EuclideanGeometry Point]

-- Assume points O_a, O_b, O_c as centers of excircles of triangle ABC
variables (A B C O_a O_b O_c : Point)

-- Assume these points form triangles within Euclidean Geometry
noncomputable def CentersOfExcircles (A B C O_a O_b O_c : Point) : Prop :=
  -- Define excircle centers for the purpose of this theorem statement
  exntrCenter A B C O_a ∧ exntrCenter B C A O_b ∧ exntrCenter C A B O_c

-- Definition stating A, B, C are the feet of the altitudes of triangle O_a O_b O_c
noncomputable def FeetOfAltitudes (A B C O_a O_b O_c : Point) : Prop :=
  footOfAltitude O_a O_b O_c A ∧ footOfAltitude O_b O_c O_a B ∧ footOfAltitude O_c O_a O_b C

-- Theorem statement
theorem altitudes_of_extriangle (h : CentersOfExcircles A B C O_a O_b O_c) :
  FeetOfAltitudes A B C O_a O_b O_c :=
sorry

end altitudes_of_extriangle_l812_812771


namespace temp_fri_l812_812210

-- Define the temperatures on Monday, Tuesday, Wednesday, Thursday, and Friday
variables (M T W Th F : ℝ)

-- Define the conditions as given in the problem
axiom avg_mon_thurs : (M + T + W + Th) / 4 = 48
axiom avg_tues_fri : (T + W + Th + F) / 4 = 46
axiom temp_mon : M = 39

-- The theorem to prove that the temperature on Friday is 31 degrees
theorem temp_fri : F = 31 :=
by
  -- placeholder for proof
  sorry

end temp_fri_l812_812210


namespace domain_of_f_range_of_f_graph_of_f_bacteria_count_expression_l812_812394

namespace Bacteria

def split_time := 2 -- bacteria split every 2 hours

def initial_bacteria := 2 -- initial number of bacteria

def bacteria_count (t : ℝ) : ℕ :=
  if t < split_time then initial_bacteria
  else initial_bacteria * 2^(⌊t / split_time⌋ : ℕ)

theorem domain_of_f : ∀ t, 0 ≤ t :=
by { intro t, exact le_of_lt (lt_add_one t) }

theorem range_of_f : ∀ y, ∃ n : ℕ, y = 2^(n + 1) :=
by { sorry }

theorem graph_of_f : ∀ t, 0 ≤ t → t < 6 →
  bacteria_count t = if t < 2 then 2
                     else if t < 4 then 4
                     else 8 :=
by { sorry }

theorem bacteria_count_expression (n : ℕ) : 
  bacteria_count n = if n % 2 = 0 
                     then 2^(n / 2 + 1) 
                     else 2^((n - 1) / 2 + 1) :=
by { sorry }

end Bacteria

end domain_of_f_range_of_f_graph_of_f_bacteria_count_expression_l812_812394


namespace small_bottles_needed_l812_812367

theorem small_bottles_needed 
  (small_bottle_capacity : ℕ := 45)
  (large_bottle1_capacity : ℕ := 630)
  (large_bottle2_capacity : ℕ := 850)
  (total_large_capacity : ℕ := large_bottle1_capacity + large_bottle2_capacity) :
  nat.ceil (total_large_capacity / small_bottle_capacity) = 33 :=
  sorry

end small_bottles_needed_l812_812367


namespace real_inequality_l812_812187

theorem real_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + a * c + b * c := by
  sorry

end real_inequality_l812_812187


namespace arccos_one_eq_zero_l812_812489

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812489


namespace circumscribed_circle_radius_l812_812838

theorem circumscribed_circle_radius (h8 h15 h17 : ℝ) (h_triangle : h8 = 8 ∧ h15 = 15 ∧ h17 = 17) : 
  ∃ R : ℝ, R = 17 := 
sorry

end circumscribed_circle_radius_l812_812838


namespace derivative_at_two_l812_812691

def f (x : ℝ) : ℝ := x^3 + 4 * x - 5

noncomputable def derivative_f (x : ℝ) : ℝ := 3 * x^2 + 4

theorem derivative_at_two : derivative_f 2 = 16 :=
by
  sorry

end derivative_at_two_l812_812691


namespace volume_ratio_tetrahedron_pyramid_l812_812714

theorem volume_ratio_tetrahedron_pyramid
  (pyramid : Type)
  [HasVertices pyramid (A B C D : ℝ^3)]
  (regular_tetrahedron : Type)
  [HasVertices regular_tetrahedron (M N P Q : ℝ^3)]
  (h_angles : ∀ {a b c}, right_angle a b c)
  (h_A_on_AC : M ∈ line_segment A C)
  (h_B_on_BC : N ∈ line_segment B C)
  (h_C_on_AB : P ∈ line_segment A B)
  (h_D_on_BD : Q ∈ line_segment B D)
  (h_MN_parallel_AB : parallel (join M N) (join A B)) :
  (volume regular_tetrahedron) / (volume pyramid) = 24 / 343 :=
sorry

noncomputable def volume (T : Type) [HasVertices T] := sorry
noncomputable def right_angle (a b c : ℝ^3) := sorry
noncomputable def line_segment (a b : ℝ^3) := sorry
noncomputable def join (a b : ℝ^3) := sorry
noncomputable def parallel (l m : set ℝ^3) := sorry
noncomputable def HasVertices (T : Type) := sorry

end volume_ratio_tetrahedron_pyramid_l812_812714


namespace arccos_one_eq_zero_l812_812402

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812402


namespace decimal_to_binary_15_l812_812571

theorem decimal_to_binary_15 :
  nat.div_rem 15 2 = (7, 1) ∧
  nat.div_rem 7 2 = (3, 1) ∧
  nat.div_rem 3 2 = (1, 1) ∧
  nat.div_rem 1 2 = (0, 1) →
  nat.to_binary 15 = 1111 :=
by
  intros
  sorry

end decimal_to_binary_15_l812_812571


namespace area_of_circumcircle_l812_812570

noncomputable def circumcircle_area_of_equilateral_triangle (s : ℝ) : ℝ :=
  let R := s / Real.sqrt 3 in π * R^2

theorem area_of_circumcircle {DE EF DF : ℝ} (h1 : DE = EF) (h2 : EF = DF) (h3 : DE = 5 * Real.sqrt 3) :
  circumcircle_area_of_equilateral_triangle DE = 25 * π :=
by
  sorry

end area_of_circumcircle_l812_812570


namespace intersection_of_M_and_N_l812_812169

-- Definition of the solution set M.
def M : set ℝ := {x | 2 * |x - 1| + x - 1 ≤ 1}

-- Definition of the solution set N.
def N : set ℝ := {x | 16 * x^2 - 8 * x + 1 ≤ 4}

-- The proof problem statement to show the intersection of M and N.
theorem intersection_of_M_and_N : M ∩ N = {x | 0 ≤ x ∧ x ≤ 3 / 4} := by
  sorry

end intersection_of_M_and_N_l812_812169


namespace russian_writer_surname_l812_812841

def is_valid_surname (x y z w v u : ℕ) : Prop :=
  x = z ∧
  y = w ∧
  v = x + 9 ∧
  u = y + w - 2 ∧
  3 * x = y - 4 ∧
  x + y + z + w + v + u = 83

def position_to_letter (n : ℕ) : String :=
  if n = 4 then "Г"
  else if n = 16 then "О"
  else if n = 13 then "Л"
  else if n = 30 then "Ь"
  else "?"

theorem russian_writer_surname : ∃ x y z w v u : ℕ, 
  is_valid_surname x y z w v u ∧
  position_to_letter x ++ position_to_letter y ++ position_to_letter z ++ position_to_letter w ++ position_to_letter v ++ position_to_letter u = "Гоголь" :=
by
  sorry

end russian_writer_surname_l812_812841


namespace pipe_A_time_to_fill_l812_812254

theorem pipe_A_time_to_fill (T_B : ℝ) (T_combined : ℝ) (T_A : ℝ): 
  T_B = 75 → T_combined = 30 → 
  (1 / T_B + 1 / T_A = 1 / T_combined) → T_A = 50 :=
by
  -- Placeholder proof
  intro h1 h2 h3
  have h4 : T_B = 75 := h1
  have h5 : T_combined = 30 := h2
  have h6 : 1 / T_B + 1 / T_A = 1 / T_combined := h3
  sorry

end pipe_A_time_to_fill_l812_812254


namespace largest_percent_error_l812_812136

theorem largest_percent_error
  (s : ℝ)
  (actual_s : s = 30)
  (error_bound : ∀ (measured_s : ℝ), measured_s ∈ set.Icc (30 * (1 - 0.2)) (30 * (1 + 0.2))) :
  ∃ (percent_error : ℝ), percent_error = 44 :=
by
  sorry

end largest_percent_error_l812_812136


namespace solve_equation_l812_812198

theorem solve_equation : 
  ∀ x : ℤ, (x - 7) / 3 - (1 + x) / 2 = 1 → x = -23 := 
by 
  intros x h,
  sorry

end solve_equation_l812_812198


namespace dot_product_equivalence_l812_812665

variable (a : ℝ × ℝ) 
variable (b : ℝ × ℝ)

-- Given conditions
def condition_1 : Prop := a = (2, 1)
def condition_2 : Prop := a - b = (-1, 2)

-- Goal
theorem dot_product_equivalence (h1 : condition_1 a) (h2 : condition_2 a b) : a.1 * b.1 + a.2 * b.2 = 5 :=
  sorry

end dot_product_equivalence_l812_812665


namespace roots_negative_condition_l812_812578

theorem roots_negative_condition (a b c r s : ℝ) (h_eqn : a ≠ 0) (h_root : a * r^2 + b * r + c = 0) (h_neg : r = -s) : b = 0 := sorry

end roots_negative_condition_l812_812578


namespace triangle_angles_l812_812929

theorem triangle_angles (R a b c : ℝ) (A B C : ℝ) (h : R * (b + c) = a * real.sqrt (b * c)) :
  2 * R * real.sin A = a ∧ 2 * R * real.sin B = b ∧ 2 * R * real.sin C = c →
  (A = real.pi / 2) ∧ (B = real.pi / 4) ∧ (C = real.pi / 4) :=
by
  sorry

end triangle_angles_l812_812929


namespace smallest_n_mod_equality_l812_812972

theorem smallest_n_mod_equality :
  ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 5]) ∧ (∀ m : ℕ, 0 < m ∧ m < n → (7^m ≡ m^7 [MOD 5]) = false) :=
  sorry

end smallest_n_mod_equality_l812_812972


namespace find_smallest_n_l812_812582

theorem find_smallest_n 
  (n : ℕ) 
  (hn : 23 * n ≡ 789 [MOD 8]) : 
  ∃ n : ℕ, n > 0 ∧ n ≡ 3 [MOD 8] :=
sorry

end find_smallest_n_l812_812582


namespace temp_pot_C_to_F_l812_812255

-- Definitions
def boiling_point_C : ℕ := 100
def boiling_point_F : ℕ := 212
def melting_point_C : ℕ := 0
def melting_point_F : ℕ := 32
def temp_pot_C : ℕ := 55
def celsius_to_fahrenheit (c : ℕ) : ℕ := (c * 9 / 5) + 32

-- Theorem to be proved
theorem temp_pot_C_to_F : celsius_to_fahrenheit temp_pot_C = 131 := by
  sorry

end temp_pot_C_to_F_l812_812255


namespace points_on_sphere_l812_812202

theorem points_on_sphere 
    (n : ℕ) (S : finset (ℝ × ℝ × ℝ)) 
    (h_n : n ≥ 4) 
    (hS_finite : S.finite) 
    (h_planar : ∀ p1 p2 p3 p4 ∈ S, coplanar p1 p2 p3 p4 → false)
    (h_coloring : ∃ (color : S → ℤ), 
                 (∀ s, s ∈ S → (color s = 1 ∨ color s = -1)) ∧
                 (∀ sphere, ∃ (intersection : finset (ℝ × ℝ × ℝ)) (h_inter : finset.card intersection ≥ 4), 
                             ∀ p ∈ intersection, p ∈ sphere ∧ 
                             (finset.card { p ∈ intersection | color p = -1 } = finset.card intersection / 2))) : 
  ∃ sphere, ∀ p, p ∈ S → p ∈ sphere := 
sorry

end points_on_sphere_l812_812202


namespace sum_abs_diff_eq_n_squared_l812_812936

theorem sum_abs_diff_eq_n_squared (n : ℕ) (hn : n > 0)
  (points : Fin (2 * n) → ℕ)
  (is_blue : Fin (2 * n) → Bool)
  (a : Fin n → Fin (2 * n))
  (b : Fin n → Fin (2 * n))
  (ha : ∀ i : Fin n, a i < 2 * n ∧ is_blue (a i) = true)
  (hb : ∀ i : Fin n, b i < 2 * n ∧ is_blue (b i) = false)
  (distinct_ab : ∀ (i j : Fin n), a i = a j → i = j ∧ b i = b j → i = j) :
  ∑ i in Finset.finRange n, abs (points (a i) - points (b i)) = n * n := sorry

end sum_abs_diff_eq_n_squared_l812_812936


namespace equation_of_W_perimeter_of_rectangle_ABCD_l812_812719

-- Definition for the locus W
def locusW (x y : ℝ) : Prop :=
  y = x^2 + 1/4

-- Proof for the equation of W
theorem equation_of_W (x y : ℝ) :
  (abs y = sqrt (x^2 + (y - 1/2)^2)) ↔ locusW x y := 
sorry

-- Prove the perimeter of rectangle ABCD
theorem perimeter_of_rectangle_ABCD (A B C D : ℝ × ℝ) :
  (locusW A.1 A.2 ∧ locusW B.1 B.2 ∧ locusW C.1 C.2 ∧ (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) ∧
  (B.1 - C.1) * (D.2 - C.2) = (D.1 - B.1) * (C.2 - B.2) ∧ (D.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (D.2 - A.2)) →
  (4 < 3 * sqrt 3) := 
sorry

end equation_of_W_perimeter_of_rectangle_ABCD_l812_812719


namespace right_triangle_identity_l812_812188

theorem right_triangle_identity (A B C : ℝ) 
  (h1 : A + B + C = π / 2) 
  (h2 : A = π / 2 ∨ B = π / 2 ∨ C = π / 2) : 
  sin A * sin B * sin (A - B) + 
  sin B * sin C * sin (B - C) + 
  sin C * sin A * sin (C - A) + 
  sin (A - B) * sin (B - C) * sin (C - A) = 0 := 
sorry

end right_triangle_identity_l812_812188


namespace probability_even_distinct_in_range_l812_812937

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0
noncomputable def all_digits_distinct (n : ℕ) : Prop := 
  let digits : List ℕ := List.digits 10 n
  List.nodup digits

def within_range (n : ℕ) : Prop := 2000 ≤ n ∧ n ≤ 9999

def favorable (n : ℕ) : Prop :=
  within_range n ∧ is_even n ∧ all_digits_distinct n

def total_count : ℕ := 8000

def favorable_count : ℕ := 2016

theorem probability_even_distinct_in_range :
  (↑favorable_count : ℚ) / ↑total_count = 63 / 250 :=
  sorry

end probability_even_distinct_in_range_l812_812937


namespace calculate_expression_l812_812950

theorem calculate_expression : 
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 :=
by
  sorry

end calculate_expression_l812_812950


namespace shaded_region_area_l812_812905

noncomputable def area_shaded_region (r_small r_large : ℝ) (A B : ℝ × ℝ) : ℝ := 
  let pi := Real.pi
  let sqrt_5 := Real.sqrt 5
  (5 * pi / 2) - (4 * sqrt_5)

theorem shaded_region_area : 
  ∀ (r_small r_large : ℝ) (A B : ℝ × ℝ), 
  r_small = 2 → 
  r_large = 3 → 
  (A = (0, 0)) → 
  (B = (4, 0)) → 
  area_shaded_region r_small r_large A B = (5 * Real.pi / 2) - (4 * Real.sqrt 5) := 
by
  intros r_small r_large A B h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end shaded_region_area_l812_812905


namespace binomial_square_correct_k_l812_812977

theorem binomial_square_correct_k (k : ℚ) : (∃ t u : ℚ, k = t^2 ∧ 28 = 2 * t * u ∧ 9 = u^2) → k = 196 / 9 :=
by
  sorry

end binomial_square_correct_k_l812_812977


namespace largest_multiple_of_9_less_than_100_l812_812271

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l812_812271


namespace set_A_set_B_sufficient_condition_necessary_condition_l812_812041

-- Define the sets A and B.
def A : set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 6}
def B (m : ℝ) : set ℝ := {x : ℝ | -2 * m ≤ x ∧ x ≤ 3 * m ∧ m > 0}

-- Prove that the set A is [-2, 6]
theorem set_A : A = {x : ℝ | -2 ≤ x ∧ x ≤ 6} := sorry

-- Prove that the set B is defined as described.
theorem set_B (m : ℝ) (h : m > 0) : B m = {x : ℝ | -2 * m ≤ x ∧ x ≤ 3 * m ∧ m > 0} := sorry

-- Prove that if x ∈ A is a sufficient condition for x ∈ B, then m ≥ 2
theorem sufficient_condition (m : ℝ) (h : m > 0) : (∀ x, (x ∈ A → x ∈ (B m))) → (m ≥ 2) := sorry

-- Prove that if x ∈ A is a necessary condition for x ∈ B, then m ≤ 1
theorem necessary_condition (m : ℝ) (h : m > 0) : (∀ x, (x ∈ (B m) → x ∈ A)) → (m ≤ 1) := sorry

end set_A_set_B_sufficient_condition_necessary_condition_l812_812041


namespace rose_incorrect_answers_l812_812100

-- Define the total number of items in the exam.
def total_items : ℕ := 60

-- Define the percentage of correct answers Liza got.
def liza_percentage_correct : ℝ := 0.90

-- Calculate the number of correct answers Liza got.
def liza_correct_answers : ℕ := (liza_percentage_correct * total_items.to_real).to_nat

-- Define the number of more correct answers Rose got compared to Liza.
def rose_more_correct : ℕ := 2

-- Calculate the number of correct answers Rose got.
def rose_correct_answers : ℕ := liza_correct_answers + rose_more_correct

-- Define the problem as a theorem to prove Rose's incorrect answers.
theorem rose_incorrect_answers (total_items liza_correct_answers rose_correct_answers : ℕ) 
  (liza_percentage_correct : ℝ) (rose_more_correct: ℕ)
  (h1 : total_items = 60)
  (h2 : liza_percentage_correct = 0.90)
  (h3 : liza_correct_answers = (liza_percentage_correct * total_items.to_real).to_nat)
  (h4 : rose_more_correct = 2)
  (h5 : rose_correct_answers = liza_correct_answers + rose_more_correct) :
  rose_incorrect_answers = total_items - rose_correct_answers := by
  sorry

end rose_incorrect_answers_l812_812100


namespace jelly_beans_correct_l812_812193

-- Define the constants and conditions
def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def total_amount : ℕ := sandra_savings + mother_gift + father_gift

def candy_cost : ℕ := 5 / 10 -- == 0.5
def jelly_bean_cost : ℕ := 2 / 10 -- == 0.2

def candies_bought : ℕ := 14
def money_spent_on_candies : ℕ := candies_bought * candy_cost

def remaining_money : ℕ := total_amount - money_spent_on_candies
def money_left : ℕ := 11

-- Prove the number of jelly beans bought is 20
def number_of_jelly_beans : ℕ :=
  (remaining_money - money_left) / jelly_bean_cost

theorem jelly_beans_correct : number_of_jelly_beans = 20 :=
sorry

end jelly_beans_correct_l812_812193


namespace min_value_expression_l812_812934

theorem min_value_expression (x y : ℝ) : 
  (∃ x, x - 2 * real.sqrt x + 3 = 2) ∧ 
  (∀ x > 0, x + 1/x ≠ 2) ∧ 
  (∀ x > 0, ∀ y > 0, (sqrt (x^2 + 2) + 4 / sqrt (x^2 + 2)) ≠ 2) ∧ 
  (∀ x > 0, ∀ y > 0, x / y + y / x ≠ 2) :=
by 
  sorry

end min_value_expression_l812_812934


namespace side_length_of_square_l812_812600

theorem side_length_of_square (length_rect width_rect : ℝ) (h_length : length_rect = 7) (h_width : width_rect = 5) :
  (∃ side_length : ℝ, 4 * side_length = 2 * (length_rect + width_rect) ∧ side_length = 6) :=
by
  use 6
  simp [h_length, h_width]
  sorry

end side_length_of_square_l812_812600


namespace parabola_equation_correct_l812_812917

noncomputable def parabola_equation (p : ℕ) (q : ℕ) (a b c d e f : ℤ) : Prop :=
  -- Points (p, q) must satisfy the parabola equation
  ∀ x y : ℝ, (x, y) = (p, q) → 
    (25 * x^2 - 20 * x * y + 4 * y^2 - 152 * x + 84 * y - 180 = 0) ∧
    (2 * x + 5 * y = 20)

theorem parabola_equation_correct :
  parabola_equation 4 2 25 (-20) 4 (-152) 84 (-180) :=
by
  intros x y hpq hpx hyq
  sorry

end parabola_equation_correct_l812_812917


namespace sum_sin_binomial_eq_l812_812986

theorem sum_sin_binomial_eq (n : ℕ) (a : ℝ) :
  ∑ k in finset.range(n+1).filter (λ k, k > 0), nat.choose n k * real.sin (k * a) =
  2^n * real.cos (a / 2)^n * real.sin (n * a / 2) :=
by sorry

end sum_sin_binomial_eq_l812_812986


namespace shortest_distance_from_circle_to_line_l812_812238

-- Define the circle equation
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the line equation
def line_eqn (x y : ℝ) : Prop := x - y - 3 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 0)

-- Define the radius of the circle
def circle_radius : ℝ := 1

-- Define the distance from a point to a line given by Ax + By + C = 0
def point_to_line_distance (A B C x y : ℝ) : ℝ := abs (A * x + B * y + C) / real.sqrt (A^2 + B^2)

-- Calculate the perpendicular distance from the center to the line
def perpendicular_distance := point_to_line_distance 1 (-1) (-3) 1 0

-- The shortest distance from a point on the circle to the line
def shortest_distance : ℝ := real.sqrt 2 - 1

-- Theorem statement
theorem shortest_distance_from_circle_to_line :
  ∀ p : ℝ × ℝ, circle_eqn p.1 p.2 → ∃ d : ℝ, d = shortest_distance :=
by
  sorry

end shortest_distance_from_circle_to_line_l812_812238


namespace min_value_reciprocal_sum_l812_812031

theorem min_value_reciprocal_sum 
  (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
  (tangent_condition : ∀ x : ℝ, y = (1 / e) * x + m + 1 ↔ y = log x - n + 2): 
  ∃ r : ℝ, r = 4 ∧ (∀ (k l : ℝ), k > 0 ∧ l > 0 ∧ ((1 / e) * k + m + 1 = log k - n + 2) ∧ 
  ((1 / e) * l + m + 1 = log l - n + 2)) → (1 / k + 1 / l = r) :=
sorry

end min_value_reciprocal_sum_l812_812031


namespace part1_part2_l812_812631

open Real

noncomputable def condition1 (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a^2 + 3 * b^2 = 3

theorem part1 {a b : ℝ} (h : condition1 a b) : sqrt 5 * a + b ≤ 4 := 
sorry

theorem part2 {x a b : ℝ} (h₁ : condition1 a b) (h₂ : 2 * abs (x - 1) + abs x ≥ 4) : 
x ≤ -2/3 ∨ x ≥ 2 := 
sorry

end part1_part2_l812_812631


namespace prob_tangent_line_obtuse_l812_812624

noncomputable def probability_tangent_obtuse_angle (b : ℝ) (h : -1 ≤ b ∧ b ≤ 5) : ℝ :=
  if b < 1 then (1 - (-1)) / (5 - (-1)) else 0

theorem prob_tangent_line_obtuse :
  (∀ b : ℝ, (-1 ≤ b ∧ b ≤ 5) → probability_tangent_obtuse_angle b (and.intro (le_refl _) (le_refl _)) = (if b < 1 then 2/6 else 0)) →
  probability_tangent_obtuse_angle 0 (and.intro (le_refl (-1)) (le_refl 5)) = 1 / 3 :=
by
  intro h_condition
  have h_prob : (∀ b : ℝ, b ∈ Set.Icc (-1:ℝ) (5:ℝ) → probability_tangent_obtuse_angle b (by simp; rintro ⟨-, -⟩; assumption) = (if b < 1 then 1/3 else 0)) :=
    by
    intros
    simp [probability_tangent_obtuse_angle]
    split
    all_goals sorry
  exact h_condition 

end prob_tangent_line_obtuse_l812_812624


namespace quantities_with_opposite_meanings_l812_812879

-- Definitions for the conditions

def Clockwise := "Clockwise"
def Counterclockwise := "Counterclockwise"
def WinGames (n : ℕ) := n  -- Number of games won
def LoseGames (n : ℕ) := -n  -- Number of games lost (negative for opposition)
def Profit (amount : ℤ) := amount
def Spending (amount : ℤ) := -amount  -- Spending is negative of profit
def MoveEast (dist : ℕ) := (dist, 0)  -- Movement vector
def MoveNorth (dist : ℕ) := (0, dist)  -- Movement vector

theorem quantities_with_opposite_meanings :
  ∃ (A B : Prop), 
    A = (Clockwise ≠ Counterclockwise) ∧ 
    B = (WinGames 2 ≠ LoseGames 3) ∧
    ¬ C = (Profit 3000 ≠ Spending 3000) ∧
    ¬ D = (MoveEast 30 ≠ MoveNorth 30) →
    B :=
by
  sorry

end quantities_with_opposite_meanings_l812_812879


namespace A_investment_is_100_l812_812356

-- Definitions directly from the conditions in a)
def A_investment (X : ℝ) := X * 12
def B_investment : ℝ := 200 * 6
def total_profit : ℝ := 100
def A_share_of_profit : ℝ := 50

-- Prove that given these conditions, A's initial investment X is 100
theorem A_investment_is_100 (X : ℝ) (h : A_share_of_profit / total_profit = A_investment X / B_investment) : X = 100 :=
by
  sorry

end A_investment_is_100_l812_812356


namespace largest_multiple_of_9_less_than_100_l812_812302

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ (∀ k, k < 100 → k % 9 = 0 → k ≤ n) :=
by {
  use 99,
  split,
  { exact lt_of_le_of_ne (nat.le_of_dvd (by norm_num) (by norm_num)) (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by use 11) },
  { intros k hk hkm, apply nat.mul_le_mul_left 9 (le_of_lt $ nat.div_lt_self hk $ by norm_num), exact le_of_lt_succ (nat.succ_le_of_lt (nat.div_lt_self hk $ by norm_num)) }
  sorry
}

end largest_multiple_of_9_less_than_100_l812_812302


namespace arccos_one_eq_zero_l812_812542

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812542


namespace angle_AQG_in_regular_octagon_is_90_l812_812804

theorem angle_AQG_in_regular_octagon_is_90 
  (ABCDEFGH : tuple (fin 8 → fin 2 → ℝ)) 
  (reg_oct : ∀ i : fin 8, ∃ j : fin 8, ∠(ABCDEFGH i) = 135) 
  (A B G H Q : fin 2)
  (extension_AB_Q : line (ABCDEFGH A) (ABCDEFGH B) = extension_line (ABCDEFGH Q))
  (extension_GH_Q : line (ABCDEFGH G) (ABCDEFGH H) = extension_line (ABCDEFGH Q)) :
∠(ABCDEFGH Q) = 90 :=
sorry

end angle_AQG_in_regular_octagon_is_90_l812_812804


namespace ratio_of_sines_l812_812126

-- Definitions from conditions
noncomputable def A : Type := sorry
noncomputable def B : A := sorry
noncomputable def C : A := sorry
noncomputable def D : A := sorry

def angle_B : ℝ := 45
def angle_C : ℝ := 60

def BD := 2 * sorry -- length is proportional to 2
def CD := 1 * sorry -- length is proportional to 1

noncomputable def sin_angle_BAD := sorry
noncomputable def sin_angle_CAD := sorry

-- Statement to prove
theorem ratio_of_sines : 
  ∀ (A B C D : A), 
  ∃ (BD CD AD : ℝ), 
  (BD / CD = 2 / 1) → 
  (angle_B = 45) → 
  (angle_C = 60) → 
  (sin_angle_BAD / sin_angle_CAD) = Real.sqrt 6 :=
begin
  intros,
  sorry,
end

end ratio_of_sines_l812_812126


namespace sqrt_9_fact_over_126_eq_12_sqrt_10_l812_812009

theorem sqrt_9_fact_over_126_eq_12_sqrt_10 :
  real.sqrt (9! / 126) = 12 * real.sqrt 10 := 
sorry

end sqrt_9_fact_over_126_eq_12_sqrt_10_l812_812009


namespace sequence_sum_l812_812921

open Stream

noncomputable def sequence_a : Stream ℝ :=
  cons 2 (cons 3 ((λs, ¼ * s.head + (1/3) * s.tail.head) <$> (tail² sequence_a)))

theorem sequence_sum :
  let S := sum₁ sequence_a
  S = 10.8 := by
  sorry

end sequence_sum_l812_812921


namespace arccos_one_eq_zero_l812_812437

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812437


namespace necessary_but_not_sufficient_l812_812337

-- Variables for the conditions
variables (x y : ℝ)

-- Conditions
def cond1 : Prop := x ≠ 1 ∨ y ≠ 4
def cond2 : Prop := x + y ≠ 5

-- Statement to prove the type of condition
theorem necessary_but_not_sufficient :
  cond2 x y → cond1 x y ∧ ¬(cond1 x y → cond2 x y) :=
sorry

end necessary_but_not_sufficient_l812_812337


namespace cos_alpha_minus_beta_l812_812027

theorem cos_alpha_minus_beta (α β : ℝ) (h1 : cos α = 1/3) (h2 : cos (α + β) = -1/3) (h3 : 0 < α ∧ α < π/2) (h4 : 0 < β ∧ β < π/2) :
  cos (α - β) = 23/27 :=
by
  sorry

end cos_alpha_minus_beta_l812_812027


namespace smallest_M_satisfying_conditions_l812_812606

theorem smallest_M_satisfying_conditions :
  ∃ M : ℕ, M > 0 ∧ M = 250 ∧
    ( (M % 125 = 0 ∧ ((M + 1) % 8 = 0 ∧ (M + 2) % 9 = 0) ∨ ((M + 1) % 9 = 0 ∧ (M + 2) % 8 = 0)) ∨
      (M % 8 = 0 ∧ ((M + 1) % 125 = 0 ∧ (M + 2) % 9 = 0) ∨ ((M + 1) % 9 = 0 ∧ (M + 2) % 125 = 0)) ∨
      (M % 9 = 0 ∧ ((M + 1) % 8 = 0 ∧ (M + 2) % 125 = 0) ∨ ((M + 1) % 125 = 0 ∧ (M + 2) % 8 = 0)) ) :=
by
  sorry

end smallest_M_satisfying_conditions_l812_812606


namespace largest_second_largest_product_l812_812609

theorem largest_second_largest_product :
  let numbers := [2.1, 3.2, 32, 12, 3.4, 2.5]
  let max := numbers.maximum
  let second_max := (numbers.filter (≠ max)).maximum
  max * second_max = 384 :=
by
  sorry

end largest_second_largest_product_l812_812609


namespace arccos_one_eq_zero_l812_812504

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812504


namespace max_subsets_l812_812141

def mySet : Set ℕ := {x | x > 0 ∧ x ≤ 100}

theorem max_subsets : ∃ k, (∀ (A B : Set ℕ), A ∈ mySet → B ∈ mySet → A ≠ B → A ∩ B ≠ ∅ → 
  (min (A ∩ B) ≠ max A ∧ min (A ∩ B) ≠ max B)) → k = 2^99 - 1 :=
sorry

end max_subsets_l812_812141


namespace top8_requirement_l812_812338

theorem top8_requirement (scores : list ℝ) (h : scores.length = 15) (h_distinct : list.nodup scores) :
  ∀ score ∈ scores, (rank_le 7 score scores ↔ score_in_median (median scores)) :=
sorry

end top8_requirement_l812_812338


namespace bonnie_roark_wire_ratio_l812_812390

theorem bonnie_roark_wire_ratio :
  (∀ (bonnie_segments : ℕ) (segment_length : ℕ) (unit_cube_volume : ℕ) 
   (cube_edge_length : ℕ) (unit_cube_num : ℕ),
    bonnie_segments = 12 ∧
    segment_length = 8 ∧
    unit_cube_volume = 1 ∧
    cube_edge_length = 8 ∧
    unit_cube_num = 512 →
    let bonnie_total_length := bonnie_segments * segment_length;
        bonnie_cube_volume := cube_edge_length ^ 3;
        roark_total_volume := bonnie_cube_volume;
        unit_cube_total_length := unit_cube_num * (unit_cube_volume ^ (1/3) * 12);
        ratio := bonnie_total_length / unit_cube_total_length
    in ratio = 1 / 64) :=
begin
  intros bonnie_segments segment_length unit_cube_volume cube_edge_length unit_cube_num,
  sorry
end

end bonnie_roark_wire_ratio_l812_812390


namespace min_blue_eyes_and_lunch_box_l812_812252

theorem min_blue_eyes_and_lunch_box (n B L : ℕ) (h1 : n = 35) (h2 : B = 20) (h3 : L = 22) : 
  min (B - (n - L)) (L - (n - B)) = 7 :=
by 
  have h4 : n - L = 13 := by rw [h1, h3]; exact rfl
  have h5 : B - (n - L) = 7 := by rw [h2, h4]; exact rfl
  exact h5

end min_blue_eyes_and_lunch_box_l812_812252


namespace arccos_one_eq_zero_l812_812517

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812517


namespace arccos_one_eq_zero_l812_812479

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812479


namespace smallest_n_modulo_5_l812_812968

theorem smallest_n_modulo_5 :
  ∃ n : ℕ, n > 0 ∧ 2^n % 5 = n^7 % 5 ∧
    ∀ m : ℕ, (m > 0 ∧ 2^m % 5 = m^7 % 5) → n ≤ m := 
begin
  use 7,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m h,
    cases h with hpos heq,
    sorry, -- Here would be proof that 7 is the smallest n satisfying 2^n ≡ n^7 mod 5.
  }
end

end smallest_n_modulo_5_l812_812968


namespace area_ratio_triangle_gfc_abc_l812_812127

/-- In triangle ABC, AD and BE are medians intersecting at centroid G.
    F is the midpoint of segment AB.
    The area of triangle GFC is l times the area of triangle ABC.
    Prove that l = 1/3. --/
theorem area_ratio_triangle_gfc_abc
  (A B C D E G F : Type)
  [is_triangle A B C]
  (AD BE : A → B → C)
  [median AD]
  [median BE]
  [centroid G AD BE]
  [midpoint F A B]
  (l : ℝ) :
  area (triangle G F C) = (1 / 3) * area (triangle A B C) :=
sorry

end area_ratio_triangle_gfc_abc_l812_812127


namespace primes_have_property_P_infinitely_many_composites_have_property_P_l812_812915

/-- Property P: for any natural number n and integer a, if n divides a^n - 1, then n^2 divides a^n - 1. -/
def property_P (n : ℕ) : Prop :=
  ∀ (a : ℤ), (↑n ∣ a^n - 1) → (↑(n^2) ∣ a^n - 1)

theorem primes_have_property_P (p : ℕ) (hp : Nat.Prime p) : property_P p :=
  sorry

theorem infinitely_many_composites_have_property_P : ∃ᶠ (n : ℕ) in atTop, nat.is_composite n ∧ property_P n :=
  sorry

end primes_have_property_P_infinitely_many_composites_have_property_P_l812_812915


namespace sum_of_constants_l812_812828

theorem sum_of_constants :
  ∃ (a b c d e : ℤ), 1000 * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e) ∧ a + b + c + d + e = 92 :=
by
  sorry

end sum_of_constants_l812_812828


namespace parabola_tangent_area_eq_l812_812572

noncomputable def parabola_tangent_area (s t a : ℝ) (h_t : t < 0) (h_a : a > 0) : Prop :=
  let α := Real.sqrt (s^2 + 1 - t) in
  4 * α * (s^2 - s + α^2 / 6) = a

theorem parabola_tangent_area_eq (s t a : ℝ) (h_t : t < 0) (h_a : a > 0) :
  parabola_tangent_area s t a h_t h_a := by
  sorry

end parabola_tangent_area_eq_l812_812572


namespace rose_incorrect_answers_is_4_l812_812101

-- Define the total number of items in the exam
def exam_total_items : ℕ := 60

-- Define the percentage of items Liza got correctly
def liza_correct_percentage : ℝ := 0.9

-- Compute the number of items Liza got correctly
def liza_correct_items : ℕ := (exam_total_items : ℝ) * liza_correct_percentage |> int.to_nat

-- Define the additional correct answers that Rose got compared to Liza
def rose_additional_correct : ℕ := 2

-- Compute the number of correct items Rose got
def rose_correct_items : ℕ := liza_correct_items + rose_additional_correct

-- Define the expected number of incorrect answers Rose had
def rose_incorrect_items : ℕ := exam_total_items - rose_correct_items

-- The statement to be proven
theorem rose_incorrect_answers_is_4 : rose_incorrect_items = 4 := 
by 
-- Provide the initial structure for the proof, actual steps are omitted
sorry

end rose_incorrect_answers_is_4_l812_812101


namespace largest_multiple_of_9_less_than_100_l812_812268

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l812_812268


namespace arccos_1_eq_0_l812_812420

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812420


namespace relationship_between_a_b_c_l812_812581

theorem relationship_between_a_b_c (a b c : ℝ) (h : b^2 - a * c = 1 / 4) :
  ¬ (∃ k : ℝ, a = b - k ∧ b = c - k) ∧ -- not an arithmetic progression
  ¬ (∃ r : ℝ, a = b / r ∧ b = c / r) ∧ -- not a geometric progression
  ¬ (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ -- not all unequal
  ¬ (a < 0 ∧ b < 0 ∧ c < 0) ∧ -- not all negative
  ¬ (b < 0 ∧ a > 0 ∧ c > 0) := -- not b negative with a and c positive
sorry

end relationship_between_a_b_c_l812_812581


namespace jason_cousins_l812_812742

theorem jason_cousins :
  let dozen := 12
  let cupcakes_bought := 4 * dozen
  let cupcakes_per_cousin := 3
  let number_of_cousins := cupcakes_bought / cupcakes_per_cousin
  number_of_cousins = 16 :=
by
  sorry

end jason_cousins_l812_812742


namespace arccos_one_eq_zero_l812_812557

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812557


namespace arccos_one_eq_zero_l812_812522

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812522


namespace value_of_sum_l812_812775

-- Define the function f
def f (x : ℝ) : ℝ := 4^x / (4^x + 2)

-- Define the main theorem
theorem value_of_sum : 
  f (1 / 11) + f (2 / 11) + f (3 / 11) + f (4 / 11) + 
  f (5 / 11) + f (6 / 11) + f (7 / 11) + f (8 / 11) + 
  f (9 / 11) + f (10 / 11) = 5 :=
by 
  sorry

end value_of_sum_l812_812775


namespace milk_mixture_l812_812077

variable (x : ℝ)
variable (butterfat_45 : ℝ)
variable (butterfat_10 : ℝ)
variable (final_butterfat : ℝ)

theorem milk_mixture :
  let tot_butterfat_45 := 0.45 * x in
  let tot_butterfat_10 := 0.10 * 20 in
  let final_volume := x + 20 in
  let total_butterfat := 0.20 * final_volume in
  tot_butterfat_45 + tot_butterfat_10 = total_butterfat → x = 8 :=
by {
  intros,
  sorry
}

end milk_mixture_l812_812077


namespace unknown_diagonal_length_l812_812207

noncomputable def rhombus_diagonal_length
  (area : ℝ) (d2 : ℝ) : ℝ :=
  (2 * area) / d2

theorem unknown_diagonal_length
  (area : ℝ) (d2 : ℝ) (h_area : area = 150)
  (h_d2 : d2 = 30) :
  rhombus_diagonal_length area d2 = 10 :=
  by
  rw [h_area, h_d2]
  -- Here, the essential proof would go
  -- Since solving would require computation,
  -- which we are omitting, we use:
  sorry

end unknown_diagonal_length_l812_812207


namespace probability_at_least_8_flyers_l812_812702

open Nat

noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))
  else 0

theorem probability_at_least_8_flyers : 
  (binom 10 5) ≠ 0 → 
  let total_outcomes := (binom 10 5) * (binom 10 5) in
  let favorable_outcomes := (binom 10 5) * 126 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 1 / 2 :=
by 
  intro h1
  let total_outcomes := (binom 10 5) * (binom 10 5)
  let favorable_outcomes := (binom 10 5) * 126
  let probability := favorable_outcomes / total_outcomes
  have h2 : (binom 10 5) = 252 := sorry -- use a known fact or algebraic calculation
  rw h2 at ⊢
  have h3 : favorable_outcomes = 126 * 252 := sorry -- simplify favorable_outcomes using h2
  have h4 : total_outcomes = 252 * 252 := sorry -- simplify total_outcomes using h2
  have h5 : probability = 126 / 252 := sorry -- simplify probability using h3 and h4
  have h6 : 126 / 252 = 1 / 2 := sorry -- reduce fraction
  exact h6


end probability_at_least_8_flyers_l812_812702


namespace trains_clear_time_l812_812328

/-
Two trains of length 120 meters and 300 meters are running towards each other
on parallel lines at speeds of 42 km/h and 30 km/h respectively.
Prove that the time it will take for them to be clear of each other from the moment they meet is 21 seconds.
-/

noncomputable def length_train1 : ℕ := 120
noncomputable def length_train2 : ℕ := 300
noncomputable def speed_train1_kmph : ℝ := 42
noncomputable def speed_train2_kmph : ℝ := 30

noncomputable def speed_train1_mps : ℝ := speed_train1_kmph * (5 / 18)
noncomputable def speed_train2_mps : ℝ := speed_train2_kmph * (5 / 18)

noncomputable def total_length : ℕ := length_train1 + length_train2
noncomputable def relative_speed_mps : ℝ := speed_train1_mps + speed_train2_mps

noncomputable def time_to_be_clear : ℝ := total_length / relative_speed_mps

theorem trains_clear_time : time_to_be_clear ≈ 21 := 
by
  sorry

end trains_clear_time_l812_812328


namespace find_number_l812_812898

theorem find_number (x : ℝ) : 50 + (x * 12) / (180 / 3) = 51 ↔ x = 5 := by
  sorry

end find_number_l812_812898


namespace range_of_m_l812_812630

theorem range_of_m (m : ℝ) :
  (1 < m ∧ m < 3 ∧ m ≠ 2 ∧ (m > sqrt 5 / 2 ∨ m < -sqrt 5 / 2)) →
  m ∈ set.Ioo (sqrt 5 / 2) 2 ∨ m ∈ set.Ioo 2 3 :=
sorry

end range_of_m_l812_812630


namespace Mary_has_4_times_more_balloons_than_Nancy_l812_812785

theorem Mary_has_4_times_more_balloons_than_Nancy :
  ∃ t, t = 28 / 7 ∧ t = 4 :=
by
  sorry

end Mary_has_4_times_more_balloons_than_Nancy_l812_812785


namespace smallest_degree_l812_812205

-- Define the terms
def root1 := 3 - Real.sqrt 8
def root2 := 3 + Real.sqrt 8
def root3 := 5 + Real.sqrt 13
def root4 := 5 - Real.sqrt 13
def root5 := 17 - 3 * Real.sqrt 6
def root6 := 17 + 3 * Real.sqrt 6
def root7 := -2 * Real.sqrt 3
def root8 := 2 * Real.sqrt 3

-- Define the polynomials formed by pairs of these roots
def poly1 (x : ℝ) := (x - root1) * (x - root2)
def poly2 (x : ℝ) := (x - root3) * (x - root4)
def poly3 (x : ℝ) := (x - root5) * (x - root6)
def poly4 (x : ℝ) := (x - root7) * (x - root8)

-- The statement to be proved
theorem smallest_degree (p : ℝ → ℝ) (p_degree : ℕ) 
  (h1 : p root1 = 0)
  (h2 : p root2 = 0)
  (h3 : p root3 = 0)
  (h4 : p root4 = 0)
  (h5 : p root5 = 0)
  (h6 : p root6 = 0)
  (h7 : p root7 = 0)
  (h8 : p root8 = 0) 
  (hrational : ∀ (x : ℝ), x ∈ p.roots → x ∈ ℚ) : 
  p_degree ≥ 8 := sorry

end smallest_degree_l812_812205


namespace all_are_truth_tellers_l812_812981

-- Define the possible states for Alice, Bob, and Carol
inductive State
| true_teller
| liar

-- Define the predicates for each person's statements
def alice_statement (B C : State) : Prop :=
  B = State.true_teller ∨ C = State.true_teller

def bob_statement (A C : State) : Prop :=
  A = State.true_teller ∧ C = State.true_teller

def carol_statement (A B : State) : Prop :=
  A = State.true_teller → B = State.true_teller

-- The theorem to be proved
theorem all_are_truth_tellers
    (A B C : State)
    (alice: A = State.true_teller → alice_statement B C)
    (bob: B = State.true_teller → bob_statement A C)
    (carol: C = State.true_teller → carol_statement A B)
    : A = State.true_teller ∧ B = State.true_teller ∧ C = State.true_teller :=
by
  sorry

end all_are_truth_tellers_l812_812981


namespace perfect_square_trinomial_l812_812682

theorem perfect_square_trinomial (m x : ℝ) : 
  ∃ a b : ℝ, (4 * x^2 + (m - 3) * x + 1 = (a + b)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end perfect_square_trinomial_l812_812682


namespace equation_of_locus_perimeter_greater_than_3sqrt3_l812_812729

theorem equation_of_locus 
  (P : ℝ × ℝ)
  (hx : ∀ y : ℝ, P.snd = |P.snd| → |P.snd| = sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2))
  :
  (P.snd = P.fst ^ 2 + 1/4) := 
sorry

theorem perimeter_greater_than_3sqrt3
  (A B C D : ℝ × ℝ)
  (h_parabola : ∀ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4)
  (h_vert_A : A ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_vert_B : B ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_vert_C : C ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_perpendicular: ((B.fst - A.fst) * (C.fst - B.fst) + (B.snd - A.snd) * (C.snd - B.snd)) = 0) 
  :
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
sorry

end equation_of_locus_perimeter_greater_than_3sqrt3_l812_812729


namespace arccos_one_eq_zero_l812_812515

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812515


namespace hyperbola_equation_l812_812066

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x, y) = (4, 3) → x^2 + y^2 = (sqrt (4^2 + 3^2))^2) 
  (h4 : 3 = b / a * 4) :
  (∀ x y : ℝ, (x^2 / 16 - y^2 / 9 = 1) ↔ (x^2 / a^2 - y^2 / b^2 = 1)) :=
by
  sorry

end hyperbola_equation_l812_812066


namespace tangency_point_exists_l812_812603

theorem tangency_point_exists :
  ∃ (x y : ℝ), y = x^2 + 18 * x + 47 ∧ x = y^2 + 36 * y + 323 ∧ x = -17 / 2 ∧ y = -35 / 2 :=
by
  sorry

end tangency_point_exists_l812_812603


namespace concurrency_and_length_relation_l812_812844

-- Define the vertices of the convex hexagon inscribed in a circle
variables {A1 A2 A3 A4 A5 A6 M : Point}
variable h_convex : ConvexHexagon A1 A2 A3 A4 A5 A6
variable h_cyclic : CyclicHexagon A1 A2 A3 A4 A5 A6

-- Define concurrencies and segment lengths
variable h_concurrent : Concurrent A1 A4 A2 A5 A3 A6

noncomputable def segment_length (A B : Point) : ℝ := sorry

-- Theorem statement
theorem concurrency_and_length_relation :
  (Concurrent A1 A4 A2 A5 A3 A6) ↔ 
  (segment_length A1 A2 * segment_length A3 A4 * segment_length A5 A6 = 
   segment_length A2 A3 * segment_length A4 A5 * segment_length A6 A1) := 
sorry

end concurrency_and_length_relation_l812_812844


namespace sum_first_20_terms_l812_812657

noncomputable def a_n (n : ℕ) : ℝ :=
  2 * n - 3 * ((1 / 5) ^ n)

noncomputable def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a_n (i + 1)

theorem sum_first_20_terms :
  S 20 = 420 - (3 / 4) * (1 - (1 / 5) ^ 20) :=
by
  sorry

end sum_first_20_terms_l812_812657


namespace length_XY_correct_l812_812753

noncomputable def length_segment_XY (A B C D : Point) (P : Segment AB) (Q : Segment CD) (X Y : CircleIntersection)
  (h1 : distance A B = 15) 
  (h2 : distance C D = 25) 
  (h3 : distance A P = 10) 
  (h4 : distance C Q = 11) 
  (h5 : distance P Q = 30) 
  (h6 : LineThroughPoints P Q X Y) : ℝ :=
  sorry

theorem length_XY_correct (A B C D : Point) (P : Segment AB) (Q : Segment CD) (X Y : CircleIntersection)
  (h1 : distance A B = 15) 
  (h2 : distance C D = 25) 
  (h3 : distance A P = 10) 
  (h4 : distance C Q = 11) 
  (h5 : distance P Q = 30) 
  (h6 : LineThroughPoints P Q X Y) : length_segment_XY A B C D P Q X Y h1 h2 h3 h4 h5 h6 = 31.4 :=
  sorry

end length_XY_correct_l812_812753


namespace arccos_one_eq_zero_l812_812454

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812454


namespace largest_multiple_of_9_less_than_100_l812_812279

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n < 100 ∧ n % 9 = 0 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 0 → m ≤ n :=
by
  use 99
  split
  · exact dec_trivial
  split 
  · exact dec_trivial
  intro m hm
  cases hm 
  cases hm_right 
  have h : m ≤ 99 / 1 := by norm_cast; simp only [Nat.le_div_iff_mul_le dec_trivial, mul_one, div_one]
  exact h
  sorry -- Complete the proof

end largest_multiple_of_9_less_than_100_l812_812279


namespace arccos_one_eq_zero_l812_812449

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812449


namespace g_has_exactly_three_zeros_in_0_5π_l812_812694

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos ((2 / 3) * x)

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.cos (((2 / 3) * x) - (π / 3))

theorem g_has_exactly_three_zeros_in_0_5π :
  (∃ x1 x2 x3 ∈ set.Ioo 0 (5 * π), g x1 = 0 ∧ g x2 = 0 ∧ g x3 = 0 ∧
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
    (∀ x ∈ set.Ioo 0 (5 * π), g x = 0 → x = x1 ∨ x = x2 ∨ x = x3)) := sorry

end g_has_exactly_three_zeros_in_0_5π_l812_812694


namespace seating_arrangements_l812_812901

/--
  A and B want to sit in a row of 8 empty seats. If it is required that there are empty seats on both sides of A and B, 
  then the number of different seating arrangements is 20.
-/
theorem seating_arrangements : 
  ∃ (seats : Fin 8 → Prop), 
      (forall x, 0 < x ∧ x < 7 → seats x = false) → 
      (∃ (A B : ℕ), A ≠ B ∧ ∀ x ∈ {A, B}, 1 ≤ x ∧ x ≤ 6) ∧ 
      (∃ (x y : ℕ), (x < y) ∧ ∀ z, z = x + 1 ∨ z = y - 1 → seats z = true) -> 
  ∃ n, n = 20 := 
sorry

end seating_arrangements_l812_812901


namespace positive_difference_l812_812209

-- State the conditions
variables (x y : ℝ)
def condition1 : Prop := (35 + x) / 2 = 45
def condition2 : Prop := (35 + x + y) / 3 = 40

-- State the theorem
theorem positive_difference (h1 : condition1 x) (h2 : condition2 x y) : | y - 35 | = 5 :=
sorry

end positive_difference_l812_812209


namespace Mary_has_4_times_more_balloons_than_Nancy_l812_812784

theorem Mary_has_4_times_more_balloons_than_Nancy :
  ∃ t, t = 28 / 7 ∧ t = 4 :=
by
  sorry

end Mary_has_4_times_more_balloons_than_Nancy_l812_812784


namespace largest_multiple_of_9_less_than_100_l812_812272

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ ∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n :=
begin
  use 99,
  split,
  { norm_num }, -- 99 < 100
  split,
  { norm_num }, -- 99 % 9 = 0
  { intros m hm,
    obtain ⟨k, rfl⟩ := nat.exists_eq_mul_right_of_dvd (dvd_of_mod_eq_zero hm.2),
    rw [mul_comm, nat.mul_lt_mul_iff_left (by norm_num : 0 < 9)],
    norm_num,
    exact nat.succ_le_iff.mpr (le_of_lt hm.1), }
end

end largest_multiple_of_9_less_than_100_l812_812272


namespace number_of_swaps_l812_812883

theorem number_of_swaps :
  let n := 2017,
      initial_seq := list.range n,
      target_seq := list.tail initial_seq ++ [initial_seq.head]
  in  (n : ℕ) ^ (n - 2) = 2017 ^ 2015 := sorry

end number_of_swaps_l812_812883


namespace number_of_solutions_to_equation_l812_812227

theorem number_of_solutions_to_equation : 
  (∃ (count : ℕ), count = 5 ∧ ∀ (x y : ℤ), (x ≠ 0 ∧ y ≠ 0) → (1 / (x:ℚ) + 1 / y = 1 / 1987 → (x, y) ∈ 
  {(1988, 1987 * 1988), (1987 * 1987, 1988), (1986, 1986 * -1987), (-1987 * 1986, 1986), (2 * 1987, 2 * 1987)})) :=
  sorry

end number_of_solutions_to_equation_l812_812227


namespace handshake_problem_l812_812387

theorem handshake_problem (n total_handshakes : ℕ) (h_n : n = 10) (h_total : total_handshakes = 45) :
  ∀ boy_shakes_hand : ℕ, boy_shakes_hand = n - 1 :=
by
  -- Definitions
  let number_of_boys := n
  let total_number_of_handshakes := (number_of_boys * (number_of_boys - 1)) / 2

  -- Proof problem
  have h : total_number_of_handshakes = total_handshakes := by
    rw [h_n]
    norm_num

  -- Conclusion
  intro boy_shakes_hand
  assumption sorry

end handshake_problem_l812_812387


namespace quadratic_polynomial_l812_812241

-- Define the third difference function for sequences
def third_diff {α : Type*} [AddGroup α] [HasSub α] (f : ℕ → α) : ℕ → α :=
  λ n, f (n + 3) - 3 * f (n + 2) + 3 * f (n + 1) - f n

-- Define the main proposition
theorem quadratic_polynomial (f : ℕ → ℝ) (h : ∀ n : ℕ, third_diff f n = 0) : 
  ∃ a b c : ℝ, ∀ n : ℕ, f n = (a / 2) * n^2 + b * n + c :=
sorry

end quadratic_polynomial_l812_812241


namespace remainder_when_multiplied_l812_812308

theorem remainder_when_multiplied (a b m : ℕ) (ha : a ≡ 202 [MOD m]) (hb : b ≡ 293 [MOD m]) : 
  (a * b) % m = 86 :=
by
  have h1 : a * b ≡ 202 * 293 [MOD m] := sorry
  have h2 : 202 * 293 ≡ 86 [MOD m] := sorry
  exact (Nat.mod_eq_of_lt sorry).symm

end remainder_when_multiplied_l812_812308


namespace angle_ADH_l812_812787

theorem angle_ADH (ABCD : Type) (A B C D K H : ABCD)
  (is_rectangle : rectangle ABCD)
  (K_on_BC : Point_on_side K B C)
  (H_on_AK : Point_on_segment H A K)
  (angle_AHD : ∠HAD = 90)
  (AK_eq_BC : AK = BC)
  (CKD_eq_71 : ∠CKD = 71) :
  ∠ADH = 52 := sorry

end angle_ADH_l812_812787


namespace unique_ellipse_eccentricity_l812_812576

theorem unique_ellipse_eccentricity :
  let z_pts := [1, 
                (-3 / 2 + Complex.I * Real.sqrt 11 / 2), 
                (-3 / 2 - Complex.I * Real.sqrt 11 / 2),
                (-5 / 2 + Complex.I * Real.sqrt 15 / 2), 
                (-5 / 2 - Complex.I * Real.sqrt 15 / 2)] in
  let cartesian_pts := [(1,0), 
                       (-3 / 2, Real.sqrt 11 / 2), 
                       (-3 / 2, -Real.sqrt 11 / 2),
                       (-5 / 2, Real.sqrt 15 / 2),
                       (-5 / 2, -Real.sqrt 15 / 2)] in
  let h := -3/4 in
  let a := 5/4 in
  let b := 11/16 in
  let c := Real.sqrt (a^2 - b^2) in
  let e := c / a in
  e = Real.sqrt (3 / 16) :=
sorry

end unique_ellipse_eccentricity_l812_812576


namespace arccos_one_eq_zero_l812_812438

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812438


namespace base4_arithmetic_l812_812595

theorem base4_arithmetic : 
  let a := 231 in let b := 21 in let c := 3 in let d := 2 in 
  (a * b / c * d : ℕ) = 10232 :=
sorry

end base4_arithmetic_l812_812595


namespace exists_infinite_k_no_prime_sequence_l812_812795

theorem exists_infinite_k_no_prime_sequence :
  ∃ᵢ (k : ℕ), ∀ n, let x : ℕ → ℕ := λ n, if n = 0 then 1 else if n = 1 then k + 2 else x (n + 2) - (k + 1) * x (n + 1) + x n in ¬ ∃ (m : ℕ), Prime (x m) :=
sorry

end exists_infinite_k_no_prime_sequence_l812_812795


namespace measure_of_angle_B_length_of_b_l812_812740

variable (a b c B : ℝ)
variable (triangle_ABC : Triangle)
variable (sin_2B_eq_sqrt2_sin_B : sin (2 * B) = sqrt 2 * sin B)
variable (area_triangle_ABC : Area triangle_ABC = 6)
variable (a_eq_4 : a = 4)
variable (cos_B_eq_sqrt2_div_2 : cos B = sqrt 2 / 2)
variable (c_eq_3_sqrt2 : c = 3 * sqrt 2)

theorem measure_of_angle_B (triangle_ABC : Triangle) 
(sin_2B_eq_sqrt2_sin_B : sin (2 * B) = sqrt 2 * sin B) :
B = π / 4 := sorry

theorem length_of_b (triangle_ABC : Triangle) 
(area_triangle_ABC : Area triangle_ABC = 6) 
(a_eq_4 : a = 4)
(cos_B_eq_sqrt2_div_2 : cos B = sqrt 2 / 2)
(c_eq_3_sqrt2 : c = 3 * sqrt 2) :
b = sqrt 10 := sorry

end measure_of_angle_B_length_of_b_l812_812740


namespace interest_rate_lending_l812_812084

variable (P : ℕ) (R_A : ℚ) (T : ℕ) (G_B : ℕ)

theorem interest_rate_lending (hP : P = 2000) (hR_A : R_A = 15) (hT : T = 4) (hG_B : G_B = 160) :
  let interest_paid_to_A := P * R_A * T / 100,
      interest_received_from_C := interest_paid_to_A + G_B,
      R_C := interest_received_from_C * 100 / (P * T)
  in R_C = 17 :=
by
  -- Definitions for interest_paid_to_A and interest_received_from_C
  have interest_paid_to_A := 2000 * 15 * 4 / 100
  have interest_received_from_C := interest_paid_to_A + 160
  have R_C := interest_received_from_C * 100 / (2000 * 4)
  -- We want to show R_C = 17
  sorry

end interest_rate_lending_l812_812084


namespace arccos_one_eq_zero_l812_812456

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812456


namespace sum_of_interior_angles_n_plus_4_l812_812827

    noncomputable def sum_of_interior_angles (sides : ℕ) : ℝ :=
      180 * (sides - 2)

    theorem sum_of_interior_angles_n_plus_4 (n : ℕ) (h : sum_of_interior_angles n = 2340) :
      sum_of_interior_angles (n + 4) = 3060 :=
    by
      sorry
    
end sum_of_interior_angles_n_plus_4_l812_812827


namespace arccos_one_eq_zero_l812_812498

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812498


namespace arccos_one_eq_zero_l812_812406

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812406


namespace problem1_solution_set_problem2_range_of_m_l812_812654

open Set

-- Definitions for the problem
def f (x : ℝ) (m : ℝ) : ℝ := |x + 1| + |x - 2| - m

-- Proof Problem (1): Solution set of f(x) > 0 for m = 5
theorem problem1_solution_set (x : ℝ) : f(x, 5) > 0 ↔ x ∈ ((-∞, -2) ∪ (3, ∞)) :=
by sorry

-- Proof Problem (2): Range of m for f(x) >= 0 to hold for all x in ℝ
theorem problem2_range_of_m (m : ℝ) :
  (∀ (x : ℝ), f x m ≥ 0) → m ≤ 3 :=
by sorry

end problem1_solution_set_problem2_range_of_m_l812_812654


namespace minimum_value_sine_shift_l812_812216

theorem minimum_value_sine_shift :
  ∀ (f : ℝ → ℝ) (φ : ℝ), (∀ x, f x = Real.sin (2 * x + φ)) → |φ| < Real.pi / 2 →
  (∀ x, f (x + Real.pi / 6) = f (-x)) →
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = - Real.sqrt 3 / 2 :=
by
  sorry

end minimum_value_sine_shift_l812_812216


namespace Seth_gave_to_his_mother_l812_812195

variable (x : ℕ)

-- Define the conditions as per the problem statement
def initial_boxes := 9
def remaining_boxes_after_giving_to_mother := initial_boxes - x
def remaining_boxes_after_giving_half := remaining_boxes_after_giving_to_mother / 2

-- Specify the final condition
def final_boxes := 4

-- Form the main theorem
theorem Seth_gave_to_his_mother :
  final_boxes = remaining_boxes_after_giving_to_mother / 2 →
  initial_boxes - x = 8 :=
by sorry

end Seth_gave_to_his_mother_l812_812195


namespace optionA_is_incorrect_l812_812880

-- Define the conditions in Lean
inductive ProgramStatement
| Input : String → ProgramStatement
| Print : String → Int → ProgramStatement
| Assign : Int → Int → ProgramStatement
| AssignWithMath : String → Int → Int → ProgramStatement

-- Define the options as per the given problem statement
def optionA : ProgramStatement := ProgramStatement.Input "MATH=; a+b+c"
def optionB : ProgramStatement := ProgramStatement.Print "MATH=" (a + b + c)
def optionC : ProgramStatement := ProgramStatement.Assign a (b + c)
def optionD : ProgramStatement := ProgramStatement.AssignWithMath "$a_1" b (-c)

-- The proof problem translated to proving that option A is incorrect
theorem optionA_is_incorrect : ∀ (option : ProgramStatement),
  option = optionA →
  ¬ (∃ vars : List String, optionA = ProgramStatement.Input vars.head) :=
by
  intro option h_option
  rw [h_option]
  intro h_exists
  sorry

end optionA_is_incorrect_l812_812880


namespace arccos_one_eq_zero_l812_812556

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812556


namespace lexie_and_tom_apple_picking_l812_812171

theorem lexie_and_tom_apple_picking :
    ∀ (lexie_apples : ℕ),
    lexie_apples = 12 →
    (let tom_apples := 2 * lexie_apples in
     let total_apples := lexie_apples + tom_apples in
     total_apples = 36) :=
by
  intros lexie_apples h_lexie_apples
  let tom_apples := 2 * lexie_apples
  let total_apples := lexie_apples + tom_apples
  rw h_lexie_apples at *
  simp at *
  exact sorry

end lexie_and_tom_apple_picking_l812_812171


namespace largest_multiple_of_9_less_than_100_l812_812267

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l812_812267


namespace min_value_expr_l812_812635

theorem min_value_expr (a b : ℝ) (h : a - 2 * b + 8 = 0) : ∃ x : ℝ, x = 2^a + 1 / 4^b ∧ x = 1 / 8 :=
by
  sorry

end min_value_expr_l812_812635


namespace butterfat_milk_mixing_l812_812677

theorem butterfat_milk_mixing :
  ∀ (x : ℝ), 
  (0.35 * x + 0.10 * 12 = 0.20 * (x + 12)) → x = 8 :=
by
  intro x
  intro h
  sorry

end butterfat_milk_mixing_l812_812677


namespace isoscelesTriangleDistanceFromAB_l812_812585

-- Given definitions
def isoscelesTriangleAreaInsideEquilateral (t m c x : ℝ) : Prop :=
  let halfEquilateralAltitude := m / 2
  let equilateralTriangleArea := (c^2 * (Real.sqrt 3)) / 4
  let equalsAltitudeCondition := x = m / 2
  let distanceFormula := x = (m + Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2 ∨ x = (m - Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2
  (2 * t = halfEquilateralAltitude * c / 2) ∧ 
  equalsAltitudeCondition ∧ distanceFormula

-- The theorem to prove given the above definition
theorem isoscelesTriangleDistanceFromAB (t m c x : ℝ) :
  isoscelesTriangleAreaInsideEquilateral t m c x →
  x = (m + Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2 ∨ x = (m - Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2 :=
sorry

end isoscelesTriangleDistanceFromAB_l812_812585


namespace largest_multiple_of_9_less_than_100_l812_812285

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n < 100 ∧ n % 9 = 0 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 0 → m ≤ n :=
by
  use 99
  split
  · exact dec_trivial
  split 
  · exact dec_trivial
  intro m hm
  cases hm 
  cases hm_right 
  have h : m ≤ 99 / 1 := by norm_cast; simp only [Nat.le_div_iff_mul_le dec_trivial, mul_one, div_one]
  exact h
  sorry -- Complete the proof

end largest_multiple_of_9_less_than_100_l812_812285


namespace tetrahedron_edge_length_l812_812611

theorem tetrahedron_edge_length (r : ℝ) (s : ℝ) 
  (h₁ : r = 2) 
  (h₂ : ∃ (c₁ c₂ c₃ c₄ : ℝ × ℝ), c₁ = (0, 0) ∧ c₂ = (4, 0) ∧ c₃ = (0, 4) ∧ c₄ = (4, 4))
  (h₃ : ∃ (c₅ : ℝ × ℝ × ℝ), c₅ = (2, 2, 2 * 2 * sqrt 2)) :
  s = 4 :=
sorry

end tetrahedron_edge_length_l812_812611


namespace t_squared_minus_one_range_l812_812162

theorem t_squared_minus_one_range (x y : ℝ) (hx : x ≠ 0) :
  ∃ t : ℝ, t = y / x ∧ (t^2 - 1) = real :=
begin
  sorry
end

end t_squared_minus_one_range_l812_812162


namespace remove_two_then_average_8_5_l812_812867

noncomputable theory

def list := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
def remaining_numbers := [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
def average (l : List ℕ) : ℕ := (l.sum : ℕ) / l.length

theorem remove_two_then_average_8_5 :
  average remaining_numbers = 8.5 :=
by sorry

end remove_two_then_average_8_5_l812_812867


namespace arccos_one_eq_zero_l812_812529

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812529


namespace train_crossing_time_l812_812325

-- Define the given conditions
def length_of_train : ℤ := 100
def length_of_bridge : ℤ := 140
def speed_of_train_kmph : ℤ := 36

-- Convert speed to meters per second
def speed_conversion_factor : ℝ := 1000 / 3600
def speed_of_train_mps : ℝ := speed_of_train_kmph * speed_conversion_factor

-- Calculate total distance
def total_distance : ℤ := length_of_train + length_of_bridge

-- Calculate expected time in seconds
def expected_time : ℝ := total_distance / speed_of_train_mps

-- Now we state the theorem
theorem train_crossing_time :
  expected_time = 24 := sorry

end train_crossing_time_l812_812325


namespace problem1_problem2_problem3_l812_812062

def f (x a : ℝ) : ℝ := x * abs (x - a)
def g (x a : ℝ) : ℝ := f x a + x

theorem problem1 (h: ∀ x: ℝ, g x a ≥ g x 0) : -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem problem2 (h: ∀ x: ℝ, 1 ≤ x ∧ x ≤ 2 → f x a < 1) : (3/2) < a ∧ a < 2 :=
sorry

theorem problem3 (h: a ≥ 2) :
  ((a > 8) → (∀ x: ℝ, 2 ≤ x ∧ x ≤ 4 → 2a - 4 ≤ f x a ∧ f x a ≤ 4a - 16)) ∧
  (4 ≤ a ∧ a ≤ 8 → (∀ x: ℝ, 2 ≤ x ∧ x ≤ 4 → 4a - 16 ≤ f x a ∧ f x a ≤ a^2 / 4 ∨ 2a - 4 ≤ f x a ∧ f x a ≤ a^2 / 4)) ∧
  ((2 ≤ a ∧ a < 4) → (∀ x: ℝ, 2 ≤ x ∧ x ≤ 4 → 0 ≤ f x a ∧ f x a ≤ 16 - 4a ∨ 0 ≤ f x a ∧ f x a ≤ 2a - 4)) :=
sorry

end problem1_problem2_problem3_l812_812062


namespace surname_assignment_l812_812861

variables (Person : Type) [Fintype Person] [DecidableEq Person]

-- The three friends
variables (Vasya Petya Grisha : Person)
-- Their surnames
variables (Lepeshkin Vatrushkin Bublikov : Person)

-- Conditions
def at_house (x y : Person) := ∃ (z : Person), z = y
def letter (x y : Person) := x = y
def chess_move (x y : Person) := x ≠ y

axiom a1 : at_house Vasya Petya
axiom a2 : at_house Grisha Petya
axiom a3 : letter Vatrushkin Petya
axiom a4 : chess_move Lepeshkin Vasya
axiom a5 : chess_move Vasya Petya

theorem surname_assignment : 
  (Vasya = Bublikov) ∧ (Petya = Vatrushkin) ∧ (Grisha = Lepeshkin) :=
begin
  sorry
end

end surname_assignment_l812_812861


namespace bacteria_initial_population_l812_812229

theorem bacteria_initial_population
  (P : ℝ → ℝ)
  (P_t : ℝ := 500000)
  (t : ℝ := 53.794705707972525)
  (T : ℝ := 6)
  (hP : ∀ t, P t = P 0 * 2^(t / T)) :
  P 0 ≈ 1010 := by
  -- Given the conditions that the population doubles every 6 minutes and reaches 500,000 in approximately 53.794705707972525 minutes
  -- We need to determine the initial population
  let P0 := 500000 / 2^(53.794705707972525 / 6)
  let approx_result := Float.round (Float.ofRat P0)
  have : approx_result ≈ 1010 := by norm_num
  exact this

end bacteria_initial_population_l812_812229


namespace trig_inequality_sin_cos_le_sqrt2_trig_inequality_sin_cos_eq_sqrt2_iff_l812_812700

theorem trig_inequality_sin_cos_le_sqrt2 (α : ℝ) (hα : 0 < α ∧ α < π / 2) : 
  sin α + cos α ≤ sqrt 2 :=
by
  sorry

theorem trig_inequality_sin_cos_eq_sqrt2_iff (α : ℝ) (hα : 0 < α ∧ α < π / 2) : 
  sin α + cos α = sqrt 2 ↔ α = π / 4 :=
by
  sorry

end trig_inequality_sin_cos_le_sqrt2_trig_inequality_sin_cos_eq_sqrt2_iff_l812_812700


namespace output_value_is_3_l812_812876

-- Define the variables and the program logic
def program (a b : ℕ) : ℕ :=
  if a > b then a else b

-- The theorem statement
theorem output_value_is_3 (a b : ℕ) (ha : a = 2) (hb : b = 3) : program a b = 3 :=
by
  -- Automatically assume the given conditions and conclude the proof. The actual proof is skipped.
  sorry

end output_value_is_3_l812_812876


namespace arccos_one_eq_zero_l812_812494

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812494


namespace quadratic_completing_square_l812_812836

theorem quadratic_completing_square (b c : ℝ) (h : ∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) :
    b + c = -106 :=
sorry

end quadratic_completing_square_l812_812836


namespace max_sections_with_five_lines_l812_812122

def sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  n * (n + 1) / 2 + 1

theorem max_sections_with_five_lines : sections 5 = 16 := by
  sorry

end max_sections_with_five_lines_l812_812122


namespace largest_multiple_of_9_less_than_100_l812_812296

theorem largest_multiple_of_9_less_than_100 : ∃ n, n < 100 ∧ n % 9 = 0 ∧ (∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n) :=
begin
  use 99,
  split,
  -- Prove the number is less than 100
  { linarith, },
  split,
  -- Prove the number is a multiple of 9
  { norm_num, },
  -- Prove it is the largest
  { intros m hlt hdiv,
    linarith [hlt, hdiv],
    norm_num at hdiv,
    cases hdiv,
    exact hdiv_left, },
end

end largest_multiple_of_9_less_than_100_l812_812296


namespace pascal_triangle_sum_diff_l812_812147

open BigOperators

noncomputable def binom (n k : ℕ) : ℕ := nat.choose n k

theorem pascal_triangle_sum_diff :
  let a_i := λ i, binom 1004 i,
      b_i := λ i, binom 1005 i,
      c_i := λ i, binom 1006 i in
  (∑ i in Finset.range 1005, b_i i / c_i i) - (∑ i in Finset.range 1004, a_i i / b_i i) = 0.5 :=
by sorry

end pascal_triangle_sum_diff_l812_812147


namespace arccos_one_eq_zero_l812_812460

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812460


namespace number_in_interval_l812_812834

def number := 0.2012
def lower_bound := 0.2
def upper_bound := 0.25

theorem number_in_interval : lower_bound < number ∧ number < upper_bound :=
by
  sorry

end number_in_interval_l812_812834


namespace min_a_monotone_f_l812_812064

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * (Real.log x + a)

def is_monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem min_a_monotone_f :
  ∃ (a : ℝ), (∀ x : ℝ, 1 < x → (f x a).monotone_on (Set.Ioi 1)) ∧ a = -1 :=
sorry

end min_a_monotone_f_l812_812064


namespace arccos_one_eq_zero_l812_812455

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812455


namespace seating_arrangements_l812_812251

def numChairs : ℕ := 12
def numCouples : ℕ := 6

-- Representing the men and women as sets of natural numbers
def men_chairs (n : ℕ) : Set ℕ := {i | i % 2 = 1}
def women_chairs (n : ℕ) : Set ℕ := {i | i % 2 = 0}

-- Defining restrictions
def not_next_to (a b : ℕ) : Prop := (a + 1) % numChairs ≠ b % numChairs ∧ (a - 1 + numChairs) % numChairs ≠ b % numChairs
def not_across (a b : ℕ) : Prop := (a + numChairs / 2) % numChairs ≠ b % numChairs

-- Combining restrictions for no one sitting next to or across from their spouse
def valid_arrangement (men women : List ℕ) : Prop :=
  ∀ (i : ℕ), i < numCouples → not_next_to (men.nth_le i (by linarith)) (women.nth_le i (by linarith)) ∧ 
  not_across (men.nth_le i (by linarith)) (women.nth_le i (by linarith))

theorem seating_arrangements : 
  ∃ (men women : List ℕ), List.length men = numCouples ∧ List.length women = numCouples ∧ 
                          (∀ (i : ℕ), i < numCouples → men_chairs numChairs (men.nth_le i (by linarith))) ∧
                          (∀ (i : ℕ), i < numCouples → women_chairs numChairs (women.nth_le i (by linarith))) ∧
                          valid_arrangement men women ∧ 
                          List.permutations men.cardinality * List.permutations women.cardinality = 24 :=
sorry

end seating_arrangements_l812_812251


namespace arccos_one_eq_zero_l812_812399

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812399


namespace arccos_one_eq_zero_l812_812508

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812508


namespace nina_total_amount_l812_812875

theorem nina_total_amount:
  ∃ (x y z w : ℕ), 
  x + y + z + w = 27 ∧
  y = 2 * z ∧
  z = 2 * x ∧
  7 < w ∧ w < 20 ∧
  10 * x + 5 * y + 2 * z + 3 * w = 107 :=
by 
  sorry

end nina_total_amount_l812_812875


namespace total_items_l812_812948

theorem total_items (slices_of_bread bottles_of_milk cookies : ℕ) (h1 : slices_of_bread = 58)
  (h2 : bottles_of_milk = slices_of_bread - 18) (h3 : cookies = slices_of_bread + 27) :
  slices_of_bread + bottles_of_milk + cookies = 183 :=
by
  sorry

end total_items_l812_812948


namespace polynomial_irreducible_l812_812793

/-- A polynomial defined as P(x) = (x^2 - 8x + 25)(x^2 - 16x + 100) ... (x^2 - 8nx + 25n^2) - 1, 
  for n ∈ ℕ*, cannot be factored into two polynomials with integer coefficients of degree 
  greater or equal to 1. -/
theorem polynomial_irreducible (n : ℕ) (h : 0 < n) :
  let P : Polynomial ℤ := (List.range n).foldl (fun acc k => acc * (Polynomial.C 1 - Polynomial.X ^ 2 + 8 * Polynomial.X - 25 * Polynomial.C k ^ 2)) 1 - 1
  in ∀ A B : Polynomial ℤ, (A * B = P) → (A.degree ≥ 1) → (B.degree ≥ 1) → false :=
sorry

end polynomial_irreducible_l812_812793


namespace find_a_if_eigenvector_l812_812633

open Matrix

variables (a : ℝ) (v : Vector (Fin 2) ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ)

theorem find_a_if_eigenvector (h : v = ![2, 3]) 
  (hM : M = ![![a, 2], ![3, 2]]) 
  (hMv : M.vecMul v = 4 • v) : a = 1 :=
by sorry

end find_a_if_eigenvector_l812_812633


namespace current_revenue_calculation_l812_812930

def previous_revenue : ℝ := 72.0
def fall_percentage : ℝ := 33.33333333333333 / 100
def current_revenue : ℝ := previous_revenue - (previous_revenue * fall_percentage)

theorem current_revenue_calculation : current_revenue = 48.0 := by
  sorry

end current_revenue_calculation_l812_812930


namespace pencils_selected_l812_812701

theorem pencils_selected (n : ℕ) (h₁ : 8 > 0) (h₂ : 2 > 0) (h_prob : 0.35714285714285715 = (Nat.choose 6 n) / (Nat.choose 8 n)) : n = 3 := by
  sorry

end pencils_selected_l812_812701


namespace train_pass_jogger_in_35_seconds_l812_812357

noncomputable def kmph_to_mps (v : ℕ) : ℕ := v * 1000 / 3600

noncomputable def time_to_pass_jogger
(speed_jogger_kmph : ℕ) -- given in kmph
(speed_train_kmph : ℕ) -- given in kmph
(head_start_m : ℕ) -- head start in meters
(train_length_m : ℕ) -- length of the train in meters
: ℕ :=
let speed_jogger := kmph_to_mps speed_jogger_kmph in
let speed_train := kmph_to_mps speed_train_kmph in
let relative_speed := speed_train - speed_jogger in
let total_distance := head_start_m + train_length_m in
total_distance / relative_speed

theorem train_pass_jogger_in_35_seconds :
  time_to_pass_jogger 9 45 230 120 = 35 :=
by
  sorry

end train_pass_jogger_in_35_seconds_l812_812357


namespace total_tennis_balls_used_l812_812372

theorem total_tennis_balls_used 
  (round1_games : Nat := 8) 
  (round2_games : Nat := 4) 
  (round3_games : Nat := 2) 
  (finals_games : Nat := 1)
  (cans_per_game : Nat := 5) 
  (balls_per_can : Nat := 3) : 

  3 * (5 * (8 + 4 + 2 + 1)) = 225 := 
by
  sorry

end total_tennis_balls_used_l812_812372


namespace geo_seq_product_l812_812120

variable {α : Type*} [OrderedRing α] [Nontrivial α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a(n + 1) = r * a(n)

theorem geo_seq_product {a : ℕ → α} (h : is_geometric_sequence a) (h5 : a 5 = 4) :
  a 2 * a 8 = 16 := 
  sorry

end geo_seq_product_l812_812120


namespace ACM_eq_BCD_l812_812736

structure Point :=
(x : ℝ)
(y : ℝ)

def isAcuteTriangle (A B C : Point) : Prop := sorry
def tangentsIntersectAt (A B D : Point) (circumcircle : Circle) : Prop := sorry
def midpoint (M A B : Point) : Prop := sorry
def angle (A B C : Point) : ℝ := sorry  -- This represents the angle ∠ABC

theorem ACM_eq_BCD
  (A B C D M : Point)
  (h_acute : isAcuteTriangle A B C)
  (h_tangents : tangentsIntersectAt A B D (circumcircle A B C))
  (h_midpoint : midpoint M A B) :
  angle A C M = angle B C D := 
sorry

end ACM_eq_BCD_l812_812736


namespace muffins_flour_and_sugar_l812_812362

theorem muffins_flour_and_sugar:
  ( ∀ (flour_per_48 sugar_per_48 muffins_wanted : ℕ) (flour_needed_sugar_needed : ℚ × ℚ),
    flour_per_48 = 3 ∧ sugar_per_48 = 2 ∧ muffins_wanted = 72 →
    (flour_needed_sugar_needed = (4.5, 3)))
: sorry

end muffins_flour_and_sugar_l812_812362


namespace solve_exact_diff_eq_l812_812812

theorem solve_exact_diff_eq (x y C : ℝ) : 
  ∃ C, (∂ (λ x y, x * Real.sin(x * y)) / ∂ y = (λ x y, x^2 * Real.cos(x * y)) ∧ 
        ∂ (λ x y, x * Real.sin(x * y)) / ∂ x = (λ x y, Real.sin(x * y) + x * y * Real.cos(x * y))) :=
sorry

end solve_exact_diff_eq_l812_812812


namespace volume_of_A_union_B_union_C_l812_812016

def A (x y z : ℝ) : Prop := |x| ≤ 1 ∧ y^2 + z^2 ≤ 1

def B (x y z : ℝ) : Prop := |y| ≤ 1 ∧ z^2 + x^2 ≤ 1

def C (x y z : ℝ) : Prop := |z| ≤ 1 ∧ x^2 + y^2 ≤ 1

theorem volume_of_A_union_B_union_C :
  let volume := 6 * π - 4 * π + 8 * (2 - √2)
  volume = 2 * π + 16 - 8 * √2 := by
  sorry

end volume_of_A_union_B_union_C_l812_812016


namespace arccos_one_eq_zero_l812_812551

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812551


namespace students_count_l812_812913

theorem students_count :
  ∃ S : ℕ, (S + 4) % 9 = 0 ∧ S = 23 :=
by
  sorry

end students_count_l812_812913


namespace arccos_one_eq_zero_l812_812475

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812475


namespace perimeter_rectangle_OAPB_range_l812_812115

noncomputable def point_on_curve_C (x : ℝ) (y : ℝ) : Prop :=
  y = 3 + real.sqrt (-x^2 + 8 * x - 15)

theorem perimeter_rectangle_OAPB_range (x y : ℝ) (hx : 4 ≤ x ∧ x ≤ 5) (hy : 3 ≤ y) :
  point_on_curve_C x y →
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi ∧
  let P := (4 + Real.cos θ, 3 + Real.sin θ) in
  let PA := y in
  let PB := x in
  let perimeter := 2 * (PA + PB) in
  12 ≤ perimeter ∧ perimeter ≤ 14 + 2 * Real.sqrt 2 :=
sorry

end perimeter_rectangle_OAPB_range_l812_812115


namespace arccos_1_eq_0_l812_812421

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812421


namespace students_taking_history_but_not_statistics_l812_812105

theorem students_taking_history_but_not_statistics (H S U : ℕ) (total_students : ℕ) 
  (H_val : H = 36) (S_val : S = 30) (U_val : U = 59) (total_students_val : total_students = 90) :
  H - (H + S - U) = 29 := 
by
  sorry

end students_taking_history_but_not_statistics_l812_812105


namespace find_k_l812_812673

variables (a b : EuclideanSpace ℝ (Fin 3))
variable k : ℝ

-- Definitions based on conditions
def magnitude_a := ‖a‖ = 4
def magnitude_b := ‖b‖ = 8
def angle_ab := real.angle a b = real.pi * (2 / 3)
def perpendicular := ⟪a + 2 • b, k • a - b⟫ = 0

-- The theorem statement
theorem find_k 
  (h_a : magnitude_a a)
  (h_b : magnitude_b b)
  (h_angle : angle_ab a b)
  (h_perpendicular : perpendicular a b k) : 
  k = -7 :=
by {
  -- Proof would be provided here
  sorry
}

end find_k_l812_812673


namespace candy_B_cost_correct_l812_812347

def candy_store_example : Prop :=
  let x := 1.70 in
  let weight_total := 5 in
  let cost_total := 2 * weight_total in
  let weight_A := 1 in
  let cost_A := 3.20 in
  let weight_B := weight_total - weight_A in
  let cost_B := x in
  cost_A + (weight_B * cost_B) = cost_total

theorem candy_B_cost_correct : candy_store_example :=
by
  sorry

end candy_B_cost_correct_l812_812347


namespace digit_seven_count_in_range_l812_812379

def count_digit_seven (n : ℕ) : ℕ :=
  n.digits 10 |>.count (λ d, d = 7)

def count_sevens_upto (m : ℕ) : ℕ :=
  (1 to m).foldl (λ count n, count + count_digit_seven n) 0

theorem digit_seven_count_in_range : count_sevens_upto 2017 = 602 :=
by sorry

end digit_seven_count_in_range_l812_812379


namespace fraction_irreducible_l812_812196

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := 
by 
  sorry

end fraction_irreducible_l812_812196


namespace jack_average_speed_l812_812132

theorem jack_average_speed
  (segment1_speed : ℝ) (segment1_time : ℝ)
  (segment2_speed : ℝ) (segment2_time : ℝ)
  (segment3_speed : ℝ) (segment3_time : ℝ)
  (h1 : segment1_speed = 3) (h2 : segment1_time = 1.5)
  (h3 : segment2_speed = 2) (h4 : segment2_time = 1.75)
  (h5 : segment3_speed = 4) (h6 : segment3_time = 2.25) :
  let total_distance := (segment1_speed * segment1_time) + (segment2_speed * segment2_time) + (segment3_speed * segment3_time)
  let total_time := segment1_time + segment2_time + segment3_time
  let average_speed := total_distance / total_time
  average_speed ≈ 3.09 := 
by sorry

end jack_average_speed_l812_812132


namespace largest_multiple_of_9_less_than_100_l812_812281

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n < 100 ∧ n % 9 = 0 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 0 → m ≤ n :=
by
  use 99
  split
  · exact dec_trivial
  split 
  · exact dec_trivial
  intro m hm
  cases hm 
  cases hm_right 
  have h : m ≤ 99 / 1 := by norm_cast; simp only [Nat.le_div_iff_mul_le dec_trivial, mul_one, div_one]
  exact h
  sorry -- Complete the proof

end largest_multiple_of_9_less_than_100_l812_812281


namespace set_difference_A_cap_B_l812_812165

open Set

noncomputable def A : Set ℝ := {x | abs x ≤ 2}
noncomputable def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

theorem set_difference_A_cap_B :
  (ℝ \ (A ∩ B)) = (Iio (-2) ∪ Ioi 0) :=
  sorry

end set_difference_A_cap_B_l812_812165


namespace largest_multiple_of_9_less_than_100_l812_812277

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ ∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n :=
begin
  use 99,
  split,
  { norm_num }, -- 99 < 100
  split,
  { norm_num }, -- 99 % 9 = 0
  { intros m hm,
    obtain ⟨k, rfl⟩ := nat.exists_eq_mul_right_of_dvd (dvd_of_mod_eq_zero hm.2),
    rw [mul_comm, nat.mul_lt_mul_iff_left (by norm_num : 0 < 9)],
    norm_num,
    exact nat.succ_le_iff.mpr (le_of_lt hm.1), }
end

end largest_multiple_of_9_less_than_100_l812_812277


namespace choir_selection_remainder_l812_812351

theorem choir_selection_remainder :
  (finset.card (((finset.range 9).filter (λ t, t % 2 = 1) ×ˢ (finset.range 11).filter (λ b, (t + b) = 6 ∧ (t - b) % 4 = 0)).filter)
    # finset.univ.1.prod_subset (λ t b, (t + b = 6 ∧ t - b = 0 ∧ t ∈ finset.range 9 ∧ b ∈ finset.range 11)))).card % 100 = 96 := sorry

end choir_selection_remainder_l812_812351


namespace bobby_pancakes_left_l812_812947

def total_pancakes : ℕ := 21
def pancakes_eaten_by_bobby : ℕ := 5
def pancakes_eaten_by_dog : ℕ := 7

theorem bobby_pancakes_left : total_pancakes - (pancakes_eaten_by_bobby + pancakes_eaten_by_dog) = 9 :=
  by
  sorry

end bobby_pancakes_left_l812_812947


namespace identify_incorrect_statement_l812_812381

-- Definitions of the conditions
def regression_line_passes_through_mean (x̄ ȳ : ℝ) : Prop := 
∀ (x y : ℝ), (x = x̄) → (y = ȳ)

def residual_points_even_distribution : Prop :=
∀ (residuals : list ℝ), (∀ r ∈ residuals, -ε ≤ r ∧ r ≤ ε) → appropriate_model residuals

def smaller_sum_of_squared_residuals_better_fit (s1 s2 : ℝ) : Prop :=
(s1 < s2) → (model_with_residuals s1 better_than model_with_residuals s2)

def r_squared_values (rA rB : ℝ) : Prop :=
(rA = 0.98) ∧ (rB = 0.80)

-- The incorrect statement (D)
def incorrect_statement_D (rA rB : ℝ) : Prop :=
r_squared_values rA rB → ¬ (rB has_better_fit_than rA)

-- Main theorem statement
theorem identify_incorrect_statement (x̄ ȳ : ℝ) (s1 s2 : ℝ) (rA rB : ℝ):
  regression_line_passes_through_mean x̄ ȳ →
  residual_points_even_distribution →
  smaller_sum_of_squared_residuals_better_fit s1 s2 →
  r_squared_values rA rB →
  incorrect_statement_D rA rB :=
by
  sorry

end identify_incorrect_statement_l812_812381


namespace sum_of_abc_eq_11_l812_812818

theorem sum_of_abc_eq_11 (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_order : a < b ∧ b < c)
  (h_inv_sum : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1) : a + b + c = 11 :=
  sorry

end sum_of_abc_eq_11_l812_812818


namespace arccos_one_eq_zero_l812_812505

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812505


namespace arccos_one_eq_zero_l812_812543

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812543


namespace geometric_sequence_log_l812_812707

theorem geometric_sequence_log : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a n = (1 / 2) * (sqrt 2) ^ (n - 1)) → 
  (a 2 * a 12 = 16) → 
  (log 2 (a 15) = 6) :=
by
  sorry

end geometric_sequence_log_l812_812707


namespace solve_for_x_l812_812811

theorem solve_for_x (q r x : ℚ)
  (h1 : 5 / 6 = q / 90)
  (h2 : 5 / 6 = (q + r) / 102)
  (h3 : 5 / 6 = (x - r) / 150) :
  x = 135 :=
by sorry

end solve_for_x_l812_812811


namespace g_has_three_zeros_in_interval_l812_812696

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * cos (2 / 3 * x)

-- Define the function g(x), which is a shift of f(x) to the right by π/2 units
def g (x : ℝ) : ℝ := 2 * cos (2 / 3 * x - π / 3)

-- The theorem stating the desired property about g(x)
theorem g_has_three_zeros_in_interval :
  ∃ a b c : ℝ, 0 < a ∧ a < b ∧ b < c ∧ c < 5 * π ∧ g a = 0 ∧ g b = 0 ∧ g c = 0 :=
sorry

end g_has_three_zeros_in_interval_l812_812696


namespace retirement_total_correct_l812_812350

-- Definitions of the conditions
def hire_year : Nat := 1986
def hire_age : Nat := 30
def retirement_year : Nat := 2006

-- Calculation of age and years of employment at retirement
def employment_duration : Nat := retirement_year - hire_year
def age_at_retirement : Nat := hire_age + employment_duration

-- The required total of age and years of employment for retirement
def total_required_for_retirement : Nat := age_at_retirement + employment_duration

-- The theorem to be proven
theorem retirement_total_correct :
  total_required_for_retirement = 70 :=
  by 
  sorry

end retirement_total_correct_l812_812350


namespace rose_incorrect_answers_is_4_l812_812102

-- Define the total number of items in the exam
def exam_total_items : ℕ := 60

-- Define the percentage of items Liza got correctly
def liza_correct_percentage : ℝ := 0.9

-- Compute the number of items Liza got correctly
def liza_correct_items : ℕ := (exam_total_items : ℝ) * liza_correct_percentage |> int.to_nat

-- Define the additional correct answers that Rose got compared to Liza
def rose_additional_correct : ℕ := 2

-- Compute the number of correct items Rose got
def rose_correct_items : ℕ := liza_correct_items + rose_additional_correct

-- Define the expected number of incorrect answers Rose had
def rose_incorrect_items : ℕ := exam_total_items - rose_correct_items

-- The statement to be proven
theorem rose_incorrect_answers_is_4 : rose_incorrect_items = 4 := 
by 
-- Provide the initial structure for the proof, actual steps are omitted
sorry

end rose_incorrect_answers_is_4_l812_812102


namespace arrangement_count_l812_812708

-- Definitions coming from the conditions
def workers : List char := ['A', 'B', 'C']
def steps : Nat := 4
def options_step_1 : List char := ['A', 'B']
def options_step_4 : List char := ['A', 'C']

-- Theorem stating the number of arrangements
theorem arrangement_count : Σ n : Nat, n = 36 :=
  sorry

end arrangement_count_l812_812708


namespace sphere_surface_area_l812_812341

theorem sphere_surface_area (V : ℝ) (π : ℝ) (volume_def : V = 8) : 
  let a := (8 : ℝ)^(1/3)
  let r := a / 2
  in  4 * π * r^2 = 4 * π := 
by
  sorry

end sphere_surface_area_l812_812341


namespace find_equation_of_W_prove_perimeter_ABC_greater_than_3sqrt3_l812_812723

def locus_condition (P : ℝ × ℝ) : Prop := 
  abs P.snd = real.sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2)

def equation_W (x y : ℝ) : Prop := 
  y = x^2 + 1/4

theorem find_equation_of_W :
  ∀ (P : ℝ × ℝ), locus_condition P → equation_W P.fst P.snd :=
by 
  sorry

def three_vertices_on_W (A B C : ℝ × ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), A = (x₁, x₁^2 + 1/4) ∧ B = (x₂, x₂^2 + 1/4) ∧ C = (x₃, x₃^2 + 1/4)

def is_rectangle (A B C D : ℝ × ℝ) : Prop := 
  three_vertices_on_W A B C ∧ 
  ∃ Dx Dy, D = (Dx, Dy) ∧ 
  -- some conditions that ensure ABCD forms a rectangle

def perimeter (A B C D : ℝ × ℝ) : ℝ := 
  dist A B + dist B C + dist C D + dist D A

theorem prove_perimeter_ABC_greater_than_3sqrt3 :
  ∀ (A B C D : ℝ × ℝ), is_rectangle A B C D → three_vertices_on_W A B C → perimeter A B C D > 3 * real.sqrt 3 :=
by 
  sorry

end find_equation_of_W_prove_perimeter_ABC_greater_than_3sqrt3_l812_812723


namespace at_least_one_not_less_than_third_l812_812164

theorem at_least_one_not_less_than_third (a b c : ℝ) (h : a + b + c = 1) : 
  ∃ x ∈ ({a, b, c} : set ℝ), x ≥ (1 / 3) :=
by
  sorry

end at_least_one_not_less_than_third_l812_812164


namespace overall_percent_increase_l812_812182

theorem overall_percent_increase (d1 d2 d3 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) (h3 : d3 = 15) :
  (π * (d3 / 2)^2 - π * (d1 / 2)^2) / (π * (d1 / 2)^2) * 100 = 125 :=
by
  -- definitions for areas
  let A1 := π * (d1 / 2)^2
  let A2 := π * (d2 / 2)^2
  let A3 := π * (d3 / 2)^2
  
  -- calculations for proof steps
  have hA1 : A1 = 25 * π := by sorry
  have hA2 : A2 = 36 * π := by sorry
  have hA3 : A3 = 56.25 * π := by sorry

  -- use the given values
  rw [h1, h2, h3] at hA1 hA2 hA3

  -- calculate the percent increase
  let overall_increase := (A3 - A1) / A1 * 100
  have : A3 - A1 = 31.25 * π := by sorry
  have : (31.25 * π) / (25 * π) = 1.25 := by sorry
  have : 1.25 * 100 = 125 := by sorry
  
  -- conclude the theorem
  exact this

end overall_percent_increase_l812_812182


namespace arccos_one_eq_zero_l812_812510

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812510


namespace orchard_tree_growth_problem_l812_812354

theorem orchard_tree_growth_problem
  (T0 : ℕ) (Tn : ℕ) (n : ℕ)
  (h1 : T0 = 1280)
  (h2 : Tn = 3125)
  (h3 : Tn = (5/4 : ℚ) ^ n * T0) :
  n = 4 :=
by
  sorry

end orchard_tree_growth_problem_l812_812354


namespace polynomials_constant_sum_or_diff_l812_812857

theorem polynomials_constant_sum_or_diff
  (F G : Polynomial ℝ)
  (hF_int_at_pts : ∀ x : ℤ, F.eval x ∈ ℤ)
  (hG_int_at_pts : ∀ x : ℤ, G.eval x ∈ ℤ) :
  ∃ c : ℝ, F + G = c ∨ F - G = c :=
  sorry

end polynomials_constant_sum_or_diff_l812_812857


namespace arccos_1_eq_0_l812_812412

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812412


namespace range_of_a_l812_812045

theorem range_of_a (a : ℝ) : 
  (∀ x y, (x-a)^2 + (y-a)^2 = 2*a^2 → x^2 + y^2 - 2*a*x - 2*a*y = 0) ∧
  (∀ x y, (x, y) = (0, 2) → (x-a)^2 + (y-a)^2 > 2*a^2) ∧
  (∃ T : (ℝ × ℝ), ∃ (x y : ℝ), (x - a) ^ 2 + (y - a) ^ 2 = 2 * a ^ 2) ∧ 
  (∀ T, T ∈ circle M → ∠MAT = 45°)
  → (sqrt 3 - 1) ≤ a ∧ a < 1 := 
sorry

end range_of_a_l812_812045


namespace base_is_twelve_l812_812705

-- Define the conditions as per the problem statement
def conditions (b : ℕ) : Prop :=
  b > 5 ∧ 
  let n := 2 * b + 4 in 
  n * n = 5 * b * b + 5 * b + 4

-- The theorem to be proved, given the conditions
theorem base_is_twelve : conditions 12 := by
  sorry

end base_is_twelve_l812_812705


namespace regression_analysis_correct_l812_812877

-- Definitions based on problem conditions
inductive AnalysisMethod
| residual_analysis
| regression_analysis
| isopleth_bar_chart
| independence_test

-- Assume height and weight have a correlation (Condition taken from the solution explanation)
axiom height_weight_correlation : Prop

-- The proof problem statement
theorem regression_analysis_correct : height_weight_correlation → (AnalysisMethod.regression_analysis = AnalysisMethod.regression_analysis) :=
by
  intro h
  refl

end regression_analysis_correct_l812_812877


namespace surface_area_of_ball_l812_812902

theorem surface_area_of_ball : 
  let r := 13 in
  4 * Real.pi * r ^ 2 = 676 * Real.pi :=
by
  sorry

end surface_area_of_ball_l812_812902


namespace cindy_distance_l812_812954

noncomputable def running_speed : ℝ := 3
noncomputable def walking_speed : ℝ := 1
noncomputable def total_time_hours : ℝ := 2 / 3
variable (d : ℝ)

theorem cindy_distance (h : d / running_speed + d / walking_speed = total_time_hours) : d = 1 / 2 := by
  sorry

end cindy_distance_l812_812954


namespace square_root_of_factorial_division_l812_812007

theorem square_root_of_factorial_division :
  (sqrt (9! / 126) = 24 * sqrt 5) :=
sorry

end square_root_of_factorial_division_l812_812007


namespace arccos_one_eq_zero_l812_812563

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812563


namespace arccos_one_eq_zero_l812_812427

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812427


namespace july_husband_current_age_l812_812873

-- Define the initial ages and the relationship between Hannah and July's age
def hannah_initial_age : ℕ := 6
def hannah_july_age_relation (hannah_age july_age : ℕ) : Prop := hannah_age = 2 * july_age

-- Define the time that has passed and the age difference between July and her husband
def time_passed : ℕ := 20
def july_husband_age_relation (july_age husband_age : ℕ) : Prop := husband_age = july_age + 2

-- Lean statement to prove July's husband's current age
theorem july_husband_current_age : ∃ (july_age husband_age : ℕ),
  hannah_july_age_relation hannah_initial_age july_age ∧
  july_husband_age_relation (july_age + time_passed) husband_age ∧
  husband_age = 25 :=
by
  sorry

end july_husband_current_age_l812_812873


namespace find_apples_l812_812359

theorem find_apples :
  ∃ n : ℕ, 50 < n ∧ n < 70 ∧
    (∀ d, d ∈ {2, 3, 5} → d ∣ n) ∧
    (∀ d, d ∈ {1, 2, 4, 6, 10, 12, 15, 20, 30, 60} → d ∣ n) :=
by
  use 60
  split
  repeat { sorry }

end find_apples_l812_812359


namespace compute_sum_t_l812_812565

noncomputable def t (x : ℝ) : ℝ :=
  (/* unknown coefficients and terms */) -- Placeholder, as coefficients are unspecified

theorem compute_sum_t :
  (∑ k in 0..10, (-1)^k * t(k * Real.pi / 5)) = 10 := by
  sorry

end compute_sum_t_l812_812565


namespace find_percentage_l812_812689

-- Given conditions
def seventy_percent_of_number (number : ℝ) : ℝ := 0.7 * number
def P_percent_of_ninety (P : ℝ) : ℝ := P * 90

-- The number given
def number := 60.00000000000001
def seventy_percent_of_number_value := seventy_percent_of_number number
def value_greater_by_30 := seventy_percent_of_number_value + 30

-- The condition we need to prove
theorem find_percentage (P : ℝ) : 
  seventy_percent_of_number_value = 42.000000000000007 ∧ 
  P_percent_of_ninety P = value_greater_by_30 → 
  P = 0.8000000000000001 :=
by
  -- prove it here
  sorry

end find_percentage_l812_812689


namespace democrat_republican_arrangement_l812_812345

-- Define the numbers of Democrats and Republicans
def num_democrats : ℕ := 7
def num_republicans : ℕ := 5

-- Define the total number of ways to arrange them around a circular table 
-- so that no two Democrats sit next to each other.
def ways_to_arrange : ℕ := 60480

-- State and assume the conditions
theorem democrat_republican_arrangement :
  ∃ (arrangements : ℕ),
    arrangements = ways_to_arrange :=
by {
  -- Sorry is used to skip the actual proof
  use ways_to_arrange,
  sorry,
}

end democrat_republican_arrangement_l812_812345


namespace time_to_cross_man_l812_812859

-- Definitions based on the conditions
def speed_faster_train_kmph := 72 -- km per hour
def speed_slower_train_kmph := 36 -- km per hour
def length_faster_train_m := 200 -- meters

-- Convert speeds from km/h to m/s
def speed_faster_train_mps := speed_faster_train_kmph * 1000 / 3600 -- meters per second
def speed_slower_train_mps := speed_slower_train_kmph * 1000 / 3600 -- meters per second

-- Relative speed calculation
def relative_speed_mps := speed_faster_train_mps - speed_slower_train_mps -- meters per second

-- Prove the time to cross the man in the slower train
theorem time_to_cross_man : length_faster_train_m / relative_speed_mps = 20 := by
  -- Placeholder for the actual proof
  sorry

end time_to_cross_man_l812_812859


namespace shares_distribution_correct_l812_812887

def shares_distributed (a b c d e : ℕ) : Prop :=
  a = 50 ∧ b = 100 ∧ c = 300 ∧ d = 150 ∧ e = 600

theorem shares_distribution_correct (a b c d e : ℕ) :
  (a = (1/2 : ℚ) * b)
  ∧ (b = (1/3 : ℚ) * c)
  ∧ (c = 2 * d)
  ∧ (d = (1/4 : ℚ) * e)
  ∧ (a + b + c + d + e = 1200) → shares_distributed a b c d e :=
sorry

end shares_distribution_correct_l812_812887


namespace percentage_of_l812_812331

theorem percentage_of (part whole : ℕ) (h_part : part = 120) (h_whole : whole = 80) : 
  ((part : ℚ) / (whole : ℚ)) * 100 = 150 := 
by
  sorry

end percentage_of_l812_812331


namespace value_of_y_l812_812614

theorem value_of_y (x y : ℝ) (h1 : x + y = 5) (h2 : x = 3) : y = 2 :=
by
  sorry

end value_of_y_l812_812614


namespace f_2014_l812_812053

noncomputable def f(x : ℝ) : ℝ := sin x + exp x + x ^ 2013

def f_deriv : ℕ → (ℝ → ℝ)
| 0       := f
| (n + 1) := deriv (f_deriv n)

theorem f_2014 (x : ℝ) : f_deriv 2014 x = -sin x + exp x :=
sorry

end f_2014_l812_812053


namespace triangle_circumcircle_problem_l812_812037

noncomputable def rightTriangle (A B C : Type*) [point A] [point B] [point C] : Prop :=
(isRightTriangle A B C) ∧ (isIsosceles A B C B C)

noncomputable def midpoint (A B M : Type*) [point A] [point B] [point M] : Prop :=
(isMidpoint A B M)

noncomputable def circumcircleArc (A B C K : Type*) [point A] [point B] 
 [point C] [point K] : Prop :=
(onCircumcircle A B C K) ∧ (withinArc A C K)

noncomputable def perpendicularFoot (A B H K : Type*) [point A] [point B] 
 [point H] [point K] : Prop :=
(perpendicularFrom K H A B)

noncomputable def segmentEquality (K H B M : Type*) [point K] [point H] 
 [point B] [point M] : Prop :=
(lengthEqual K H B M)

noncomputable def parallelLines (M H C K : Type*) [point M] [point H] 
 [point C] [point K] : Prop :=
(parallel M H C K)

theorem triangle_circumcircle_problem (A B C M K H : Type*) [point A] [point B] 
 [point C] [point M] [point K] [point H]
(h0 : rightTriangle A B C) 
(h1 : midpoint B C M)
(h2 : circumcircleArc A B C K)
(h3 : perpendicularFoot A B H K)
(h4 : segmentEquality K H B M)
(h5 : parallelLines M H C K) 
: (angle A C K = 22.5) := 
sorry

end triangle_circumcircle_problem_l812_812037


namespace gcd_of_8a_plus_3_and_5a_plus_2_l812_812598

theorem gcd_of_8a_plus_3_and_5a_plus_2 (a : ℕ) : Nat.gcd (8 * a + 3) (5 * a + 2) = 1 :=
by
  sorry

end gcd_of_8a_plus_3_and_5a_plus_2_l812_812598


namespace arccos_one_eq_zero_l812_812540

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812540


namespace hyperbola_equation_l812_812621

theorem hyperbola_equation {a b: ℝ} (h₀: ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1 → true)
  (h_asym: (b = (sqrt 6 / 3) * a))
  (h_chord: ∀ x, y = 2 * x + (sqrt 210 / 3) ∧  (λ h : ¬((10 * x)^2 + (4 * x * sqrt 210) + (35 + 6 * m) = 0), false) ∧ (resulting_chord_length = 4)) :
  (a^2 = 3 ∧ b^2 = 2) :=
by
  sorry

end hyperbola_equation_l812_812621


namespace arccos_one_eq_zero_l812_812547

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812547


namespace actual_speed_of_car_l812_812349

noncomputable def actual_speed (t : ℝ) (d : ℝ) (reduced_speed_factor : ℝ) : ℝ := 
  (d / t) * (1 / reduced_speed_factor)

noncomputable def time_in_hours : ℝ := 1 + (40 / 60) + (48 / 3600)

theorem actual_speed_of_car : 
  actual_speed time_in_hours 42 (5 / 7) = 35 :=
by
  sorry

end actual_speed_of_car_l812_812349


namespace arccos_one_eq_zero_l812_812537

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812537


namespace number_of_students_in_third_event_l812_812911

theorem number_of_students_in_third_event
    (total_students : ℕ := 960)
    (selected_students : ℕ := 32)
    (first_selected_number : ℕ := 30)
    (range_first_event : ℕ × ℕ := (1, 350))
    (range_second_event : ℕ × ℕ := (351, 700))
    (range_third_event : ℕ × ℕ := (701, 960)) :
    (260 / 30).to_nat + 1 = 9 :=
  sorry

end number_of_students_in_third_event_l812_812911


namespace perimeter_triangle_l812_812829

-- Definitions and conditions
def side1 : ℕ := 2
def side2 : ℕ := 5
def is_odd (n : ℕ) : Prop := n % 2 = 1
def valid_third_side (x : ℕ) : Prop := 3 < x ∧ x < 7 ∧ is_odd x

-- Theorem statement
theorem perimeter_triangle : ∃ (x : ℕ), valid_third_side x ∧ (side1 + side2 + x = 12) :=
by 
  sorry

end perimeter_triangle_l812_812829


namespace largest_multiple_of_9_less_than_100_l812_812299

theorem largest_multiple_of_9_less_than_100 : ∃ n, n < 100 ∧ n % 9 = 0 ∧ (∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n) :=
begin
  use 99,
  split,
  -- Prove the number is less than 100
  { linarith, },
  split,
  -- Prove the number is a multiple of 9
  { norm_num, },
  -- Prove it is the largest
  { intros m hlt hdiv,
    linarith [hlt, hdiv],
    norm_num at hdiv,
    cases hdiv,
    exact hdiv_left, },
end

end largest_multiple_of_9_less_than_100_l812_812299


namespace focus_magnitude_eq_two_sqrt_thirteen_l812_812757

def hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 4 = 1

theorem focus_magnitude_eq_two_sqrt_thirteen
  {x y : ℝ}
  (hx : hyperbola x y)
  (F1 F2 P : ℝ × ℝ)
  (hfoci : F1 = (c, 0) ∧ F2 = (-c, 0))
  (horthog : ∃ (p q : ℝ), p > 0 ∧ q > 0 ∧ 
                     (euclidean_distance P F1 = p) ∧ 
                     (euclidean_distance P F2 = q) ∧ 
                     (p * q = 8) ∧ 
                     (p^2 + q^2 = 52))
  (hF1F2 : euclidean_distance F1 F2 = 2 * sqrt 13) : 
  euclidean_norm (F1 + F2) = 2 * sqrt 13 :=
sorry

end focus_magnitude_eq_two_sqrt_thirteen_l812_812757


namespace car_speed_first_hour_l812_812239

theorem car_speed_first_hour (x : ℝ) :
  (let total_distance := x + 60 in
   let average_speed := total_distance / 2 in
   average_speed = 70) →
  x = 80 :=
by
  intros h
  have : (x + 60) / 2 = 70 := h
  sorry

end car_speed_first_hour_l812_812239


namespace count_lines_through_four_points_l812_812078

def is_valid_point (i j k : ℕ) : Prop :=
  1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5 ∧ 1 ≤ k ∧ k ≤ 5

def line_through_four_points (p1 p2 p3 p4 : ℕ × ℕ × ℕ) : Prop :=
  -- placeholder definition, flesh out as needed for actual validation
  True

def count_lines : ℕ :=
  -- placeholder calculation representing the logic to count valid lines
  120

theorem count_lines_through_four_points (M : ℕ := 5) :
  (∑ p1 p2 p3 p4 in finset.univ, if is_valid_point p1.1 p1.2.1 p1.2.2 ∧
    is_valid_point p2.1 p2.2.1 p2.2.2 ∧
    is_valid_point p3.1 p3.2.1 p3.2.2 ∧
    is_valid_point p4.1 p4.2.1 p4.2.2 ∧
    line_through_four_points p1 p2 p3 p4 then 1 else 0) = 120 :=
sorry

end count_lines_through_four_points_l812_812078


namespace largest_multiple_of_9_less_than_100_l812_812260

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l812_812260


namespace max_shapes_cut_from_grid_l812_812023

theorem max_shapes_cut_from_grid : 
  ∀ (n s r : ℕ), s = 2 * 2 ∧ r = 1 * 4 ∧ (n = 12) → 
  ∃ (x y : ℕ), x + y = n ∧ 4 * x + 4 * y ≤ 49 ∧ x = y :=
begin
  sorry
end

end max_shapes_cut_from_grid_l812_812023


namespace solution_set_inequality_l812_812047

noncomputable def f : ℝ → ℝ := sorry

variable {f : ℝ → ℝ}
variable (hf_diff : Differentiable ℝ f)
variable (hf_ineq : ∀ x, f x > deriv f x)
variable (hf_zero : f 0 = 2)

theorem solution_set_inequality : {x : ℝ | f x < 2 * Real.exp x} = {x | 0 < x} :=
by
  sorry

end solution_set_inequality_l812_812047


namespace arccos_one_eq_zero_l812_812431

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812431


namespace max_S_value_l812_812079

theorem max_S_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hline : 2 * a + b = 1) : 
  S = 2 * sqrt (a * b) - 4 * a^2 - b^2 → 
  S ≤ (sqrt 2 - 1) / 2 :=
sorry

end max_S_value_l812_812079


namespace total_wage_is_75_l812_812896

noncomputable def wages_total (man_wage : ℕ) : ℕ :=
  let men := 5
  let women := (5 : ℕ)
  let boys := 8
  (man_wage * men) + (man_wage * men) + (man_wage * men)

theorem total_wage_is_75
  (W : ℕ)
  (man_wage : ℕ := 5)
  (h1 : 5 = W) 
  (h2 : W = 8) 
  : wages_total man_wage = 75 := by
  sorry

end total_wage_is_75_l812_812896


namespace timothy_avg_speed_l812_812249

-- Definitions from conditions
variables (T : ℝ) (x : ℝ)
assume h1 : Real
assume h2 : 0 < T (h3 := x > 0)
-- Timothy drove at 50 mph for 0.75 of the total time.
def speed_portion_2 := 50 * (0.75 * T)
def time_part_1 := 0.25 * T
def time_part_2 := 0.75 * T

-- Total distance formulas
noncomputable def total_distance := (x * time_part_1) + (50 * time_part_2 : ℝ)

-- Given average speed for the entire journey is 40 mph, we have:
noncomputable def average_speed := 40

-- Equivalent to total distance = average speed * T
theorem timothy_avg_speed : total_distance = average_speed * T := by
  sorry

end timothy_avg_speed_l812_812249


namespace six_sin6_cos6_l812_812087

theorem six_sin6_cos6 (A : ℝ) (h : Real.cos (2 * A) = - Real.sqrt 5 / 3) : 
  6 * Real.sin (A) ^ 6 + 6 * Real.cos (A) ^ 6 = 4 := 
sorry

end six_sin6_cos6_l812_812087


namespace arccos_one_eq_zero_l812_812462

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812462


namespace greatest_four_digit_divisible_by_6_l812_812864

-- Define a variable to represent a four-digit number
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a variable to represent divisibility by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define a variable to represent divisibility by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- State the theorem to prove that 9996 is the greatest four-digit number divisible by 6
theorem greatest_four_digit_divisible_by_6 : 
  (∀ n : ℕ, is_four_digit_number n → divisible_by_6 n → n ≤ 9996) ∧ (is_four_digit_number 9996 ∧ divisible_by_6 9996) :=
by
  -- Insert the proof here
  sorry

end greatest_four_digit_divisible_by_6_l812_812864


namespace sum_of_squares_bounds_l812_812360

-- Given quadrilateral vertices' distances from the nearest vertices of the square
variable (w x y z : ℝ)
-- The side length of the square
def side_length_square : ℝ := 1

-- Expression for the square of each side of the quadrilateral
def square_AB : ℝ := w^2 + x^2
def square_BC : ℝ := (side_length_square - x)^2 + y^2
def square_CD : ℝ := (side_length_square - y)^2 + z^2
def square_DA : ℝ := (side_length_square - z)^2 + (side_length_square - w)^2

-- Sum of the squares of the sides
def sum_of_squares := square_AB w x + square_BC x y + square_CD y z + square_DA z w

-- Proof that the sum of the squares is within the bounds [2, 4]
theorem sum_of_squares_bounds (hw : 0 ≤ w ∧ w ≤ side_length_square)
                              (hx : 0 ≤ x ∧ x ≤ side_length_square)
                              (hy : 0 ≤ y ∧ y ≤ side_length_square)
                              (hz : 0 ≤ z ∧ z ≤ side_length_square)
                              : 2 ≤ sum_of_squares w x y z ∧ sum_of_squares w x y z ≤ 4 := sorry

end sum_of_squares_bounds_l812_812360


namespace angle_between_vectors_magnitude_vector_expression_l812_812666

open Real

variables (a b : ℝ → ℝ) (theta : ℝ) (ha : |a| = 1) (hb : |b| = 6) (h : a • (b - a) = 2)

-- Question 1: Prove the angle between vectors a and b is π/3
theorem angle_between_vectors (hθ : a • b = 6 * cos theta) : theta = π / 3 := sorry

-- Question 2: Prove |2a - b| is 2√7
theorem magnitude_vector_expression : |2 * a - b| = 2 * sqrt 7 := sorry

end angle_between_vectors_magnitude_vector_expression_l812_812666


namespace square_side_subsegment_property_l812_812814

-- Given definitions based on conditions:
variables (A B C D E F : Point)
variables (ABCD_square : square ABCD)
variables (side_length_of_ABCD : length(ABCD_square.side) = 15)
variables (BE : line_segment B E)
variables (DF : line_segment D F)
variables (AE : line_segment A E)
variables (CF : line_segment C F)
variables (length_BE : length(BE) = 7)
variables (length_DF : length(DF) = 7)
variables (length_AE : length(AE) = 17)
variables (length_CF : length(CF) = 17)

theorem square_side_subsegment_property :
  ∀ (A B C D E F : Point), square ABCD →
  length(ABCD_square.side) = 15 →
  length(BE) = 7 →
  length(DF) = 7 →
  length(AE) = 17 →
  length(CF) = 17 →
  length(EF) ^ 2 = 1216 :=
sorry

end square_side_subsegment_property_l812_812814


namespace arccos_one_eq_zero_l812_812483

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812483


namespace base3_base8_digit_difference_l812_812963

theorem base3_base8_digit_difference (n : ℕ) (h : n = 729) :
  (nat.digits 3 n).length - (nat.digits 8 n).length = 4 :=
by sorry

end base3_base8_digit_difference_l812_812963


namespace general_formula_sequence_exist_lambda_no_exist_lambda_l812_812626

-- Define the sequence {a_n} and sequence of sums S_n
variable {a : ℕ → ℝ} {S : ℕ → ℝ} {T : ℕ → ℝ}
variables {t s : ℝ}

-- Condition: a_n - 2 * S_n = 1
axiom a_n_condition (n : ℕ) : a n - 2 * S n = 1 

-- Proof of the general formula for the sequence {a_n}
theorem general_formula_sequence (n : ℕ) : a n = (-1) ^ n :=
sorry

-- Proof of existence and range of λ
theorem exist_lambda (n : ℕ) (h1 : 0 < t) (h2 : t < s) :
   ∃ λ : ℝ, (∀ (n : ℕ+), λ * (a n) + s / (T n) < 3 * t) ↔ (0 < t ∧ s < 6 * t ∧ -3 * t ≤ λ ∧ λ < 3 * t - s) :=
sorry

theorem no_exist_lambda (n : ℕ) (h1 : 0 < t) (h2 : t < s) (h3 : s ≥ 6 * t) :
   ¬ ∃ λ : ℝ, ∀ (n : ℕ+), λ * (a n) + s / (T n) < 3 * t :=
sorry

end general_formula_sequence_exist_lambda_no_exist_lambda_l812_812626


namespace wage_increase_for_productivity_increase_l812_812236

open Real

theorem wage_increase_for_productivity_increase (x : ℝ) : ∀ Δx : ℝ, Δx = 1 → 
  let y := 10 + 70 * x in 
  let y' := 10 + 70 * (x + Δx) in 
  y' - y = 70 :=
by
  intro Δx hΔx
  simp [hΔx]
  sorry

end wage_increase_for_productivity_increase_l812_812236


namespace arccos_one_eq_zero_l812_812440

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812440


namespace range_of_x_in_tight_sequence_arithmetic_tight_sequence_geometric_tight_sequence_l812_812625

-- Problem (1)
theorem range_of_x_in_tight_sequence (a : ℕ → ℝ) (x : ℝ) (h : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = x ∧ a 4 = 4 → 2 ≤ x ∧ x ≤ 3 :=
sorry

-- Problem (2)
theorem arithmetic_tight_sequence (a : ℕ → ℝ) (a1 d : ℝ) (h : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2) :
  ∀ n : ℕ, a n = a1 + ↑n * d → 0 < d ∧ d ≤ a1 →
  ∀ n : ℕ, 1 / 2 ≤ (a (n + 1) / a n) ∧ (a (n + 1) / a n) ≤ 2 :=
sorry

-- Problem (3)
theorem geometric_tight_sequence (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) (h_seq : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2)
(S : ℕ → ℝ) (h_sum_seq : ∀ n : ℕ, 1 / 2 ≤ S (n + 1) / S n ∧ S (n + 1) / S n ≤ 2) :
  (∀ n : ℕ, a n = a1 * q ^ n ∧ S n = a1 * (1 - q ^ (n + 1)) / (1 - q)) → 
  1 / 2 ≤ q ∧ q ≤ 1 :=
sorry

end range_of_x_in_tight_sequence_arithmetic_tight_sequence_geometric_tight_sequence_l812_812625


namespace arccos_1_eq_0_l812_812416

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812416


namespace functional_expression_y_l812_812044

theorem functional_expression_y (x y : ℝ) (k : ℝ) 
  (h1 : ∀ x, y + 2 = k * x) 
  (h2 : y = 7) 
  (h3 : x = 3) : 
  y = 3 * x - 2 := 
by 
  sorry

end functional_expression_y_l812_812044


namespace arccos_one_eq_zero_l812_812546

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812546


namespace correct_propositions_count_l812_812847

theorem correct_propositions_count (a b : ℝ) :
  (∀ a b, a > b → a + 1 > b + 1) ∧
  (∀ a b, a > b → a - 1 > b - 1) ∧
  (∀ a b, a > b → -2 * a < -2 * b) ∧
  (¬ ∀ a b, a > b → 2 * a < 2 * b) → 
  3 = 3 :=
by
  intro h
  sorry

end correct_propositions_count_l812_812847


namespace matrix_vector_addition_l812_812566

theorem matrix_vector_addition :
  let A := ![![4, -2], ![-5, 6]]
  let v := ![5, -3]
  let w := ![1, -1]
  (A.mul_vec v + w) = ![27, -44] :=
by
  sorry

end matrix_vector_addition_l812_812566


namespace sum_powers_divisible_by_10_l812_812686

theorem sum_powers_divisible_by_10 (n : ℕ) (hn : n % 4 ≠ 0) : 
  ∃ k : ℕ, 1^n + 2^n + 3^n + 4^n = 10 * k :=
  sorry

end sum_powers_divisible_by_10_l812_812686


namespace sara_initial_black_marbles_l812_812194

-- Define the given conditions
def red_marbles (sara_has : Nat) : Prop := sara_has = 122
def black_marbles_taken_by_fred (fred_took : Nat) : Prop := fred_took = 233
def black_marbles_now (sara_has_now : Nat) : Prop := sara_has_now = 559

-- The proof problem statement
theorem sara_initial_black_marbles
  (sara_has_red : ∀ n : Nat, red_marbles n)
  (fred_took_marbles : ∀ f : Nat, black_marbles_taken_by_fred f)
  (sara_has_now_black : ∀ b : Nat, black_marbles_now b) :
  ∃ b, b = 559 + 233 :=
by
  sorry

end sara_initial_black_marbles_l812_812194


namespace arccos_one_eq_zero_l812_812422

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812422


namespace find_x_floor_mul_eq_100_l812_812604

theorem find_x_floor_mul_eq_100 (x : ℝ) (h1 : 0 < x) (h2 : (⌊x⌋ : ℝ) * x = 100) : x = 10 :=
by
  sorry

end find_x_floor_mul_eq_100_l812_812604


namespace wedge_volume_is_125_pi_l812_812909

open Real

noncomputable def volume_of_wedge (diameter : ℝ) (angle : ℝ) : ℝ :=
  let r := diameter / 2
  let h := r * cos (π / 3) -- 60 degrees in radians is π/3
  π * (r^2) * h

theorem wedge_volume_is_125_pi :
  volume_of_wedge 10 60 = 125 * π :=
by
  unfold volume_of_wedge
  norm_num
  rw [cos_pi_div_three, mul_one_div]
  norm_num
  ring

end wedge_volume_is_125_pi_l812_812909


namespace tan_alpha_fraction_eq_five_sevenths_l812_812340

theorem tan_alpha_fraction_eq_five_sevenths (α : ℝ) (h : Real.tan α = 3) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 :=
sorry

end tan_alpha_fraction_eq_five_sevenths_l812_812340


namespace sqrt_sum_eq_sqrt_N_l812_812085

theorem sqrt_sum_eq_sqrt_N:
  (√12 + √108 = √192) :=
by
  sorry

end sqrt_sum_eq_sqrt_N_l812_812085


namespace largest_integer_in_box_l812_812334

theorem largest_integer_in_box (k : ℤ) (h : (k:ℚ) / 11 < 2 / 3) : k ≤ 7 :=
begin
  -- Proof goes here
  sorry
end

end largest_integer_in_box_l812_812334


namespace actual_number_is_65_l812_812824

theorem actual_number_is_65 (avg_incorrect : ℕ) (num_elements : ℕ) (incorrect_number : ℕ) (avg_correct : ℕ) (sum_incorrect : ℕ) (sum_correct : ℕ) :
  avg_incorrect = 46 ∧ num_elements = 10 ∧ incorrect_number = 25 ∧ avg_correct = 50 
  ∧ sum_incorrect = num_elements * avg_incorrect 
  ∧ sum_correct = num_elements * avg_correct
  ∧ (sum_correct - sum_incorrect) = (avg_correct - avg_incorrect) * num_elements + incorrect_number
  → (correct_number : ℕ), correct_number = 65 :=
by
  intro h
  sorry

end actual_number_is_65_l812_812824


namespace final_position_after_swaps_l812_812344

def jia_final_position (initial_position : ℕ) (swaps : ℕ) : ℕ :=
  if swaps % 2 = 0 then initial_position
  else if initial_position = 1 then 2
  else if initial_position = 2 then 1
  else if initial_position = 3 then 2
  else initial_position

theorem final_position_after_swaps : 
  ∀ (jia_initial_position : ℕ) (number_of_swaps : ℕ), 
  jia_initial_position = 3 → number_of_swaps = 9 → jia_final_position jia_initial_position number_of_swaps = 2 := 
by 
  intros jia_initial_position number_of_swaps h_initial_pos h_swaps
  rw [h_initial_pos, h_swaps]
  simp [jia_final_position]
  sorry

end final_position_after_swaps_l812_812344


namespace ellipse_equation_correct_l812_812642

noncomputable def ellipse_equation_proof : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ), (x - 2 * y + 4 = 0) ∧ (∃ (f : ℝ × ℝ), f = (-4, 0)) ∧ (∃ (v : ℝ × ℝ), v = (0, 2)) → 
    (x^2 / (a^2) + y^2 / (b^2) = 1 → x^2 / 20 + y^2 / 4 = 1))

theorem ellipse_equation_correct : ellipse_equation_proof :=
  sorry

end ellipse_equation_correct_l812_812642


namespace surface_area_ratio_l812_812235

theorem surface_area_ratio (a : ℝ) (h : a > 0) :
  let r_inscribed := a / 2,
      r_circumscribed := (a * Real.sqrt 3) / 2,
      sa_inscribed := 4 * Real.pi * r_inscribed ^ 2,
      sa_circumscribed := 4 * Real.pi * r_circumscribed ^ 2 in
  sa_inscribed / sa_circumscribed = 1 / 3 := by
  sorry

end surface_area_ratio_l812_812235


namespace sqrt_of_factorial_div_l812_812002

theorem sqrt_of_factorial_div : 
  √(9! / 126) = 8 * √5 :=
by
  sorry

end sqrt_of_factorial_div_l812_812002


namespace arccos_one_eq_zero_l812_812480

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812480


namespace arccos_one_eq_zero_l812_812436

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812436


namespace arccos_one_eq_zero_l812_812555

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812555


namespace arccos_one_eq_zero_l812_812397

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812397


namespace complex_circle_l812_812659

def det (a b c d : ℂ) : ℂ := a * d - b * c

def i : ℂ := complex.I

def z1 : ℂ := det 1 2 i (i^2018)

def is_circle_with_center_and_radius (z center : ℂ) (radius : ℝ) : Prop :=
  complex.abs (z - center) = radius

theorem complex_circle :
  is_circle_with_center_and_radius z1 (-1 - 2 * i) 4 := by
  sorry

end complex_circle_l812_812659


namespace largest_multiple_of_9_less_than_100_l812_812289

theorem largest_multiple_of_9_less_than_100 : ∃ x : ℕ, 9 * x < 100 ∧ ∀ y : ℕ, 9 * y < 100 → y ≤ x :=
by
  exists 11
  split
  · linarith
  · intro y h
    have : 9 * y < 100 := h
    calc
      y ≤ floor (11.11) : by linarith

end largest_multiple_of_9_less_than_100_l812_812289


namespace count_common_divisors_45_75_l812_812680

def is_divisor (a b : ℕ) : Prop := b % a = 0

def common_divisors_count (m n : ℕ) : ℕ :=
  (finset.filter (λ k, is_divisor k m ∧ is_divisor k n) (finset.range (min m n + 1))).card

theorem count_common_divisors_45_75 : common_divisors_count 45 75 = 4 := by
  sorry

end count_common_divisors_45_75_l812_812680


namespace vectors_parallel_x_eq_pm_2_l812_812664

theorem vectors_parallel_x_eq_pm_2 (x : ℝ) (k : ℝ) (a b : ℝ × ℝ) :
  a = (-1, x) → b = (x, -4) → a = k • b → x = 2 ∨ x = -2 :=
by { intros h₁ h₂ h₃, rw [h₁, h₂] at h₃, sorry }

end vectors_parallel_x_eq_pm_2_l812_812664


namespace tg_gamma_half_eq_2_div_5_l812_812690

theorem tg_gamma_half_eq_2_div_5
  (α β γ : ℝ)
  (a b c : ℝ)
  (triangle_angles : α + β + γ = π)
  (tg_half_alpha : Real.tan (α / 2) = 5/6)
  (tg_half_beta : Real.tan (β / 2) = 10/9)
  (ac_eq_2b : a + c = 2 * b):
  Real.tan (γ / 2) = 2 / 5 :=
sorry

end tg_gamma_half_eq_2_div_5_l812_812690


namespace decreasing_interval_of_even_function_l812_812080

noncomputable def f (m : ℝ) (x : ℝ) := (m - 2) * x^2 + m * x + 4

theorem decreasing_interval_of_even_function (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  ∃ (a b : ℝ), is_decreasing (f 0) [a, b] :=
by
  sorry

end decreasing_interval_of_even_function_l812_812080


namespace sum_u1_u2_u3_l812_812573

def v0 : ℝ × ℝ := (2, 1)
def u0 : ℝ × ℝ := (5, 2)

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let v_norm_sq := v.1 * v.1 + v.2 * v.2
  (dot_product / v_norm_sq * v.1, dot_product / v_norm_sq * v.2)

theorem sum_u1_u2_u3 :
  let u_n (n : ℕ) : ℝ × ℝ :=
    match n with
    | 0 => u0
    | _ => projection (projection (u_n (n - 1)) v0) u0
  u_n 1 + u_n 2 + u_n 3 = (5, 2) * (14 * 12 / (29 * 5) + (14 * 12 / (29 * 5))^2 + (14 * 12 / (29 * 5))^3) := 
sorry

end sum_u1_u2_u3_l812_812573


namespace isosceles_triangle_angle_l812_812938

theorem isosceles_triangle_angle (x : ℕ) (h1 : 2 * x + x + x = 180) :
  x = 45 ∧ 2 * x = 90 :=
by
  have h2 : 4 * x = 180 := by linarith
  have h3 : x = 45 := by linarith
  have h4 : 2 * x = 90 := by linarith
  exact ⟨h3, h4⟩

end isosceles_triangle_angle_l812_812938


namespace general_term_a_sum_term_T_l812_812644

variable (λ : ℝ) (S : ℕ → ℝ) (a b T : ℕ → ℝ)

-- Conditions
axiom geo_seq (n : ℕ) : S n = λ * 4^n - 3 * λ + 1
axiom a_geom (n : ℕ) : ∀ n, a 1 = λ + 1 ∧ (a (n + 1) = 4 * a n)
axiom b_def (n : ℕ) : b n = Real.log2 (S n + 1/2) + 1

-- Prove the general term of {a_n}
theorem general_term_a (n : ℕ) : a n = 3/2 * 4^(n - 1) :=
by sorry

-- Prove the sum T_n of the sequence {3b_n / 4a_n}
theorem sum_term_T (n : ℕ) : T n = 4/9 * (4^(n+1) - 3*n - 4) / 4^n :=
by sorry

end general_term_a_sum_term_T_l812_812644


namespace water_amount_in_sport_formulation_l812_812737

/-
The standard formulation has the ratios:
F : CS : W = 1 : 12 : 30
Where F is flavoring, CS is corn syrup, and W is water.
-/

def standard_flavoring_ratio : ℚ := 1
def standard_corn_syrup_ratio : ℚ := 12
def standard_water_ratio : ℚ := 30

/-
In the sport formulation:
1) The ratio of flavoring to corn syrup is three times as great as in the standard formulation.
2) The ratio of flavoring to water is half that of the standard formulation.
-/
def sport_flavor_to_corn_ratio : ℚ := 3 * (standard_flavoring_ratio / standard_corn_syrup_ratio)
def sport_flavor_to_water_ratio : ℚ := 1 / 2 * (standard_flavoring_ratio / standard_water_ratio)

/-
The sport formulation contains 6 ounces of corn syrup.
The target is to find the amount of water in the sport formulation.
-/
def corn_syrup_in_sport_formulation : ℚ := 6
def flavoring_in_sport_formulation : ℚ := sport_flavor_to_corn_ratio * corn_syrup_in_sport_formulation

def water_in_sport_formulation : ℚ := 
  (flavoring_in_sport_formulation / sport_flavor_to_water_ratio)

theorem water_amount_in_sport_formulation : water_in_sport_formulation = 90 := by
  sorry

end water_amount_in_sport_formulation_l812_812737


namespace heptagon_diagonals_l812_812908

theorem heptagon_diagonals : 
  let n := 7 in
  (n * (n - 3)) / 2 = 14 :=
by
  sorry

end heptagon_diagonals_l812_812908


namespace largest_multiple_of_9_less_than_100_l812_812266

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l812_812266


namespace no_inscribed_sphere_l812_812567

theorem no_inscribed_sphere (V F : ℕ) (hV : V ≥ F) (original_polyhedron : Type) 
  (truncate : original_polyhedron → Prop) 
  (new_polyhedron : original_polyhedron → Type) :
  (∃ s : Sphere, s.inscribed (new_polyhedron (truncate original_polyhedron))) → false :=
by
  sorry

end no_inscribed_sphere_l812_812567


namespace salted_duck_eggs_min_cost_l812_812336

-- Define the system of equations and their solutions
def salted_duck_eggs_pricing (a b : ℕ) : Prop :=
  (9 * a + 6 * b = 390) ∧ (5 * a + 8 * b = 310)

-- Total number of boxes and constraints
def total_boxes_conditions (x y : ℕ) : Prop :=
  (x + y = 30) ∧ (x ≥ y + 5) ∧ (x ≤ 2 * y)

-- Minimize cost function given prices and constraints
def minimum_cost (x y a b : ℕ) : Prop :=
  (salted_duck_eggs_pricing a b) ∧
  (total_boxes_conditions x y) ∧
  (a = 30) ∧ (b = 20) ∧
  (10 * x + 600 = 780)

-- Statement to prove
theorem salted_duck_eggs_min_cost : ∃ x y : ℕ, minimum_cost x y 30 20 :=
by
  sorry

end salted_duck_eggs_min_cost_l812_812336


namespace ratio_of_bens_cards_l812_812388

variables (ben_initial : ℕ) (tim_initial : ℕ) (ben_bought : ℕ)

def ben_final := ben_initial + ben_bought
def tim_final := tim_initial

theorem ratio_of_bens_cards (h1 : ben_initial = 37)
                            (h2 : tim_initial = 20)
                            (h3 : ben_bought = 3) :
  ben_final ben_initial ben_bought / tim_final tim_initial = 2 :=
by {
  simp [ben_final, tim_final, h1, h2, h3],
  norm_num,
  sorry
}

end ratio_of_bens_cards_l812_812388


namespace happiness_estimate_l812_812823

theorem happiness_estimate (n k : ℕ) (A : Fin n → Fin k → bool) :
  (∀ i j : Fin k, i ≠ j → (∑ l : Fin n, if A l i = A l j then 1 else 0) = n / 2) →
  (∃ m : ℕ, m ≤ n - n / k ∧ ∀ (j : Fin n), 
    (∑ i : Fin k, if A j i then 1 else -1) = 0 → j < m) :=
sorry

end happiness_estimate_l812_812823


namespace part_one_union_part_one_complement_intersection_part_two_l812_812999

open Set Real

def A : Set ℝ := { x : ℝ | 2^x > 1 }
def B : Set ℝ := { x : ℝ | log 3 (x + 1) < 1 }
def C (a : ℝ) : Set ℝ := { x : ℝ | x < a }

theorem part_one_union : A ∪ B = { x : ℝ | x > -1 } :=
sorry

theorem part_one_complement_intersection : (Aᶜ) ∩ B = { x : ℝ | -1 < x ∧ x ≤ 0 } :=
sorry

theorem part_two (a : ℝ) (h : B ∪ C a = C a) : 2 ≤ a :=
sorry

end part_one_union_part_one_complement_intersection_part_two_l812_812999


namespace jason_cousins_l812_812743

theorem jason_cousins :
  let dozen := 12
  let cupcakes_bought := 4 * dozen
  let cupcakes_per_cousin := 3
  let number_of_cousins := cupcakes_bought / cupcakes_per_cousin
  number_of_cousins = 16 :=
by
  sorry

end jason_cousins_l812_812743


namespace max_non_managers_with_9_managers_l812_812703

-- Definitions based on conditions
def ratio_greater_than_seven_thirtyseven (m n : ℕ) : Prop := m > 0 ∧ n > 0 ∧ (m.to_rat / n.to_rat > 7 / 37)
def num_must_be_12_percent (m e : ℕ) : Prop := m = (12 * e) / 100
def valid_employee_count (m e : ℕ) : Prop := m >= 5 ∧ e <= 300

-- The main theorem statement
theorem max_non_managers_with_9_managers :
  ∃ (e : ℕ), num_must_be_12_percent 9 e ∧ valid_employee_count 9 e ∧ e - 9 = 66 :=
begin
  -- We write the proof here in Lean, but for now, we'll use sorry to bypass the actual proof steps.
  sorry
end

end max_non_managers_with_9_managers_l812_812703


namespace nonincreasing_7_digit_integers_l812_812602

theorem nonincreasing_7_digit_integers : 
  ∃ n : ℕ, n = 11439 ∧ (∀ x : ℕ, (10^6 ≤ x ∧ x < 10^7) → 
    (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 7 → (x / 10^(7 - i) % 10) ≥ (x / 10^(7 - j) % 10))) :=
by
  sorry

end nonincreasing_7_digit_integers_l812_812602


namespace sum_of_a_b_l812_812074

theorem sum_of_a_b (a b : ℝ) (h₁ : a^3 - 3 * a^2 + 5 * a = 1) (h₂ : b^3 - 3 * b^2 + 5 * b = 5) : a + b = 2 :=
sorry

end sum_of_a_b_l812_812074


namespace original_denominator_value_l812_812922

theorem original_denominator_value (d : ℕ) (h1 : 3 + 3 = 6) (h2 : ((6 : ℕ) / (d + 3 : ℕ) = (1 / 3 : ℚ))) : d = 15 :=
sorry

end original_denominator_value_l812_812922


namespace sum_triang_areas_half_l812_812181

variable {α : Type*} [EuclideanGeometry α]

open EuclideanGeometry

-- Definitions and assumptions
def N (α : Type*): Point α := sorry
def B (α : Type*): Point α := sorry
def A (α : Type*): Point α := sorry
def Q (α : Type*) (N A : Point α) : Point α := point_between N A (1/4)
def F (α : Type*) (N A : Point α) : Point α := point_between N A (3/4)
def L (α : Type*) (Q F : Point α) : Point α := point_between Q F (1/2)
def BL (α : Type*) (B L : Point α) : LineSegment α := line_segment B L
def D (α : Type*) (N B Q : Point α) : Point α := intersection_point N B (parallel_line BL Q)
def K (α : Type*) (A B F : Point α) : Point α := intersection_point A B (parallel_line BL F)
def triangle_area (α : Type*) : ∀ (p1 p2 p3 : Point α), ℝ := sorry

-- The theorem statement
theorem sum_triang_areas_half :
  let NBA_area := triangle_area N B A in
  let NDL_area := triangle_area N D L in
  let AKL_area := triangle_area A K L in
  NDL_area + AKL_area = NBA_area / 2 := sorry

end sum_triang_areas_half_l812_812181


namespace arccos_one_eq_zero_l812_812550

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812550


namespace max_books_with_23_dollars_l812_812932

theorem max_books_with_23_dollars :
  ∃ (single_book_sets four_book_sets seven_book_sets : ℕ),
    (2 * single_book_sets + 7 * four_book_sets + 12 * seven_book_sets = 23) ∧
    (single_book_sets + 4 * four_book_sets + 7 * seven_book_sets = 13) :=
begin
  use [2, 1, 1],
  split,
  { calc
    2 * 2 + 7 * 1 + 12 * 1 = 4 + 7 + 12 : by ring
    ... = 23 : by ring },
  { calc
    2 + 4 * 1 + 7 * 1 = 2 + 4 + 7 : by ring
    ... = 13 : by ring },
end

end max_books_with_23_dollars_l812_812932


namespace arccos_one_eq_zero_l812_812474

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812474


namespace circles_tangent_internally_l812_812189

theorem circles_tangent_internally 
  (x y : ℝ) 
  (h : x^4 - 16 * x^2 + 2 * x^2 * y^2 - 16 * y^2 + y^4 = 4 * x^3 + 4 * x * y^2 - 64 * x) :
  ∃ c₁ c₂ : ℝ × ℝ, 
    (c₁ = (0, 0)) ∧ (c₂ = (2, 0)) ∧ 
    ((x - c₁.1)^2 + (y - c₁.2)^2 = 16) ∧ 
    ((x - c₂.1)^2 + (y - c₂.2)^2 = 4) ∧
    dist c₁ c₂ = 2 := 
sorry

end circles_tangent_internally_l812_812189


namespace gerald_tasks_needed_per_month_l812_812997

-- Conditions given in the problem
def cost_of_supplies_per_month := 100
def number_of_months_for_supplies := 4
def cost_of_glove := 80
def number_of_months_to_save := 8
def charge_for_raking := 8
def charge_for_shoveling := 12
def charge_for_mowing := 15

-- Definition to calculate the total cost for the season
def total_cost_for_season := cost_of_supplies_per_month * number_of_months_for_supplies + cost_of_glove

-- Definition to calculate the amount Gerald needs to save each month
def amount_to_save_each_month := total_cost_for_season / number_of_months_to_save

-- Definition for the average charge per task
def average_charge_per_task := (charge_for_raking + charge_for_shoveling + charge_for_mowing) / 3

-- Definition to calculate the number of tasks needed per month
def number_of_tasks_per_month := (amount_to_save_each_month / average_charge_per_task).ceil

-- Theorem to state and prove
theorem gerald_tasks_needed_per_month :
  number_of_tasks_per_month = 6 :=
sorry

end gerald_tasks_needed_per_month_l812_812997


namespace solve_system_of_equations_l812_812200

theorem solve_system_of_equations :
  ∀ x y : ℚ, (4 * x - 7 * y = -14) ∧ (5 * x + 3 * y = -13) →
  x = -133 / 47 ∧ y = 18 / 47 :=
by
  intros x y h
  cases h with h1 h2
  sorry

end solve_system_of_equations_l812_812200


namespace part_a_part_b_l812_812146

def A (n : ℕ) : Set ℕ := { p | ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ p ∣ a + b ∧ p^2 ∣ a^n + b^n ∧ Nat.gcd p (a + b) = 1 }

noncomputable def f (n : ℕ) : ℕ := @Set.finite_toFinset ℕ _ (A n) sorry Finset.card

theorem part_a (n : ℕ) : Set.Finite (A n) ↔ n ≠ 2 := sorry

theorem part_b (m k : ℕ) (hmo : m % 2 = 1) (hko : k % 2 = 1) (d : ℕ) (hd : Nat.gcd m k = d) :
  f d ≤ f k + f m - f (k * m) ∧ f k + f m - f (k * m) ≤ 2 * f d := sorry

end part_a_part_b_l812_812146


namespace determine_ab_l812_812068

noncomputable def hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b < 0) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let sqrt_ab := Real.sqrt (a / -b)
  ((1 / (1 + sqrt_ab), sqrt_ab / (1 + sqrt_ab)),
   (1 / (1 - sqrt_ab), -sqrt_ab / (1 - sqrt_ab)))

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  let (xA, yA) := A
  let (xB, yB) := B
  ((xA + xB) / 2, (yA + yB) / 2)

noncomputable def slope (origin m : ℝ × ℝ) : ℝ :=
  let (xO, yO) := origin
  let (xM, yM) := m
  yM / xM

theorem determine_ab (a b : ℝ) (ha : a > 0) (hb : b < 0)
  (slope_condition : slope (0, 0) (midpoint (hyperbola_asymptotes a b ha hb).1 (hyperbola_asymptotes a b ha hb).2) = -√3/2) :
  a / b = -√3 / 2 :=
sorry

end determine_ab_l812_812068


namespace mary_black_balloons_l812_812782

theorem mary_black_balloons (nancyBalloons maryBalloons : ℕ) 
  (H_nancy : nancyBalloons = 7) 
  (H_mary : maryBalloons = 28) : maryBalloons / nancyBalloons = 4 := 
by
  -- Provided conditions
  rw [H_nancy, H_mary]
  -- perform division
  exact (28 / 7).nat_cast().of_nat_proved_eq
  exact (28 / 7 = 4)

-- Proof is omitted
-- sorry

end mary_black_balloons_l812_812782


namespace equation_of_W_perimeter_of_rectangle_ABCD_l812_812720

-- Definition for the locus W
def locusW (x y : ℝ) : Prop :=
  y = x^2 + 1/4

-- Proof for the equation of W
theorem equation_of_W (x y : ℝ) :
  (abs y = sqrt (x^2 + (y - 1/2)^2)) ↔ locusW x y := 
sorry

-- Prove the perimeter of rectangle ABCD
theorem perimeter_of_rectangle_ABCD (A B C D : ℝ × ℝ) :
  (locusW A.1 A.2 ∧ locusW B.1 B.2 ∧ locusW C.1 C.2 ∧ (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) ∧
  (B.1 - C.1) * (D.2 - C.2) = (D.1 - B.1) * (C.2 - B.2) ∧ (D.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (D.2 - A.2)) →
  (4 < 3 * sqrt 3) := 
sorry

end equation_of_W_perimeter_of_rectangle_ABCD_l812_812720


namespace total_tennis_balls_used_l812_812373

theorem total_tennis_balls_used 
  (round1_games : Nat := 8) 
  (round2_games : Nat := 4) 
  (round3_games : Nat := 2) 
  (finals_games : Nat := 1)
  (cans_per_game : Nat := 5) 
  (balls_per_can : Nat := 3) : 

  3 * (5 * (8 + 4 + 2 + 1)) = 225 := 
by
  sorry

end total_tennis_balls_used_l812_812373


namespace arccos_one_eq_zero_l812_812538

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812538


namespace no_solution_xy_in_nat_star_l812_812802

theorem no_solution_xy_in_nat_star (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : ¬ (x * (x + 1) = 4 * y * (y + 1)) :=
by
  -- The proof would go here, but we'll leave it out for now.
  sorry

end no_solution_xy_in_nat_star_l812_812802


namespace smallest_n_7_pow_n_eq_n_7_mod_5_l812_812976

theorem smallest_n_7_pow_n_eq_n_7_mod_5 :
  ∃ n : ℕ, 0 < n ∧ (7^n ≡ n^7 [MOD 5]) ∧ ∀ m : ℕ, 0 < m ∧ (7^m ≡ m^7 [MOD 5]) → n ≤ m :=
begin
  sorry -- proof omitted as requested
end

end smallest_n_7_pow_n_eq_n_7_mod_5_l812_812976


namespace solve_system_l812_812813

theorem solve_system :
  ∃ (x : Fin 100 → ℤ),
    (∀ i : Fin 100, ((x 0) + (i : ℤ + 2) * (x 1) + (i : ℤ + 2) * (x 2) + ... + (i : ℤ + 2) * (x 99) = i + 1)) ∧
    (x = λ i, if even i then -1 else 1) :=
by
  -- Placeholder for the proof
  sorry

end solve_system_l812_812813


namespace largest_multiple_of_9_less_than_100_l812_812298

theorem largest_multiple_of_9_less_than_100 : ∃ n, n < 100 ∧ n % 9 = 0 ∧ (∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n) :=
begin
  use 99,
  split,
  -- Prove the number is less than 100
  { linarith, },
  split,
  -- Prove the number is a multiple of 9
  { norm_num, },
  -- Prove it is the largest
  { intros m hlt hdiv,
    linarith [hlt, hdiv],
    norm_num at hdiv,
    cases hdiv,
    exact hdiv_left, },
end

end largest_multiple_of_9_less_than_100_l812_812298


namespace total_fuel_l812_812779

theorem total_fuel (fuel_this_week : ℝ) (reduction_percent : ℝ) :
  fuel_this_week = 15 → reduction_percent = 0.20 → 
  (fuel_this_week + (fuel_this_week * (1 - reduction_percent))) = 27 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end total_fuel_l812_812779


namespace largest_multiple_of_9_less_than_100_l812_812258

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l812_812258


namespace expected_rainfall_week_l812_812376

theorem expected_rainfall_week :
  let P_sun := 0.35
  let P_2 := 0.40
  let P_8 := 0.25
  let rainfall_2 := 2
  let rainfall_8 := 8
  let daily_expected := P_sun * 0 + P_2 * rainfall_2 + P_8 * rainfall_8
  let total_expected := 7 * daily_expected
  total_expected = 19.6 :=
by
  sorry

end expected_rainfall_week_l812_812376


namespace point_inequality_l812_812791

variable (A B C M : Point)
variable [inside_triangle A B C M]

theorem point_inequality (N : Point) (hN : inside_triangle A B C N) :
  (distance A N > distance A M) ∨ (distance B N > distance B M) ∨ (distance C N > distance C M) :=
sorry

end point_inequality_l812_812791


namespace stanley_ran_farther_l812_812816

-- Definitions based on given conditions
def distance_ran := 4.8
def distance_walked_meters := 950
def meter_to_kilometer_conversion := 1000

-- The main theorem to be proven
theorem stanley_ran_farther : distance_ran - (distance_walked_meters / meter_to_kilometer_conversion) = 3.85 := by
  sorry

end stanley_ran_farther_l812_812816


namespace arccos_one_eq_zero_l812_812553

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812553


namespace conjugate_of_z_l812_812989

noncomputable def z : ℂ := 2 - 2 * (√3 : ℝ) * complex.I

theorem conjugate_of_z :
  ((√3 : ℝ) * z + complex.I * z = 4 * ((√3 : ℝ) - complex.I)) →
  conj z = 2 + 2 * (√3 : ℝ) * complex.I :=
by
  intro h
  sorry

end conjugate_of_z_l812_812989


namespace find_sin_value_l812_812028

variables (α : ℝ)

theorem find_sin_value (h : cos (α - π / 6) + sin α = (4 / 5) * sqrt 3) : 
  sin (α + 7 * π / 6) = -4 / 5 :=
sorry

end find_sin_value_l812_812028


namespace floor_ineq_solution_l812_812150

def floor (x : ℝ) : ℤ := int.floor x

theorem floor_ineq_solution (x : ℝ) (y : ℤ) (h1: y = floor x) :
  y^2 - 5*y + 6 ≤ 0 ↔ 2 ≤ x ∧ x < 4 :=
sorry

end floor_ineq_solution_l812_812150


namespace prism_surface_area_equals_three_times_volume_l812_812365

noncomputable def log_base (a x : ℝ) := Real.log x / Real.log a

theorem prism_surface_area_equals_three_times_volume (x : ℝ) 
  (h : 2 * (log_base 5 x * log_base 6 x + log_base 5 x * log_base 10 x + log_base 6 x * log_base 10 x) 
        = 3 * (log_base 5 x * log_base 6 x * log_base 10 x)) :
  x = Real.exp ((2 / 3) * Real.log 300) :=
sorry

end prism_surface_area_equals_three_times_volume_l812_812365


namespace acd_over_b_eq_neg_210_l812_812577

theorem acd_over_b_eq_neg_210 
  (a b c d x : ℤ) 
  (h1 : x = (a + b*Real.sqrt c)/d) 
  (h2 : (7*x)/8 + 1 = 4/x) 
  : (a * c * d) / b = -210 := 
by 
  sorry

end acd_over_b_eq_neg_210_l812_812577


namespace cauchy_schwarz_variant_l812_812773

theorem cauchy_schwarz_variant
  {n : ℕ} 
  (a A : Fin n → ℝ) 
  (ha : ∀ i, 0 < a i) 
  (hA : ∀ i, 0 < A i) :
  (∑ i, a i * A i) * (∑ i, a i / A i) ≥ (∑ i, a i)^2 :=
by
  sorry

end cauchy_schwarz_variant_l812_812773


namespace yuan_conversion_l812_812895
-- Defining the conversion rates
def jiao_to_yuan := 0.1
def fen_to_yuan := 0.01

-- Definition of the main problem
def currency_conversion (yuan : ℝ) (jiao : ℝ) (fen : ℝ) : ℝ :=
  yuan + (jiao * jiao_to_yuan) + (fen * fen_to_yuan)

-- Proving the problem statement
theorem yuan_conversion : currency_conversion 2 3 4 = 2.34 := by
  sorry

end yuan_conversion_l812_812895


namespace contrapositive_proposition_l812_812232

theorem contrapositive_proposition (x : ℝ) : (x^2 = 1 → x = 1) → (x ≠ 1 → x ≠ 1 ∧ x ≠ -1) :=
by
  intro h hx
  split
  exact hx
  intro hx_neg
  apply hx
  have h1 : x^2 = 1 := by
    rw [←mul_self_eq_one_iff, h]
    exact hx_neg
  exact (h h1).symm

end contrapositive_proposition_l812_812232


namespace interest_rate_second_part_l812_812926

theorem interest_rate_second_part (P1 P2: ℝ) (total_sum : ℝ) (rate1 : ℝ) (time1 : ℝ) (time2 : ℝ) (interest_second_part: ℝ ) : 
  total_sum = 2717 → P2 = 1672 → time1 = 8 → rate1 = 3 → time2 = 3 →
  P1 + P2 = total_sum →
  P1 * rate1 * time1 / 100 = P2 * interest_second_part * time2 / 100 →
  interest_second_part = 5 :=
by
  sorry

end interest_rate_second_part_l812_812926


namespace divisibility_of_product_l812_812185

theorem divisibility_of_product (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a * b) % 5 = 0) :
  a % 5 = 0 ∨ b % 5 = 0 :=
sorry

end divisibility_of_product_l812_812185


namespace smallest_n_for_polygon_cutting_l812_812240

theorem smallest_n_for_polygon_cutting : 
  ∃ n : ℕ, (∃ k : ℕ, n - 2 = k * 31) ∧ (∃ k' : ℕ, n - 2 = k' * 65) ∧ n = 2017 :=
sorry

end smallest_n_for_polygon_cutting_l812_812240


namespace nth_equation_l812_812786

theorem nth_equation (n: ℕ) (h: 1 ≤ n):
  ∏ k in (finset.range n).image (λ k, n + 1 + k) = 2^n * ∏ k in finset.range n, (2*k + 1) :=
by 
  sorry

end nth_equation_l812_812786


namespace test_approximate_average_price_per_bottle_l812_812747

noncomputable def approximate_average_price_per_bottle
(total_large_bottles : ℕ) (price_per_large_bottle : ℚ) 
(total_small_bottles : ℕ) (price_per_small_bottle : ℚ) 
(discount_large : ℚ) (discount_small : ℚ) (sales_tax : ℚ) : ℚ :=
let total_cost_before_discounts := (total_large_bottles * price_per_large_bottle) + (total_small_bottles * price_per_small_bottle) in
let total_discount := (discount_large * (total_large_bottles * price_per_large_bottle)) + (discount_small * (total_small_bottles * price_per_small_bottle)) in
let total_cost_after_discounts := total_cost_before_discounts - total_discount in
let final_total_cost := total_cost_after_discounts * (1 + sales_tax) in
final_total_cost / (total_large_bottles + total_small_bottles)

theorem test_approximate_average_price_per_bottle :
  approximate_average_price_per_bottle 1375 (175 / 100) 690 (135 / 100) (12 / 100) (9 / 100) (8 / 100) ≈ 155 / 100 :=
by sorry

end test_approximate_average_price_per_bottle_l812_812747


namespace actual_average_height_is_correct_l812_812326

noncomputable def incorrect_average_height : ℝ := 181
noncomputable def number_of_boys : ℕ := 35
noncomputable def incorrect_height : ℝ := 166
noncomputable def actual_height : ℝ := 106
noncomputable def rounded_actual_average_height : ℝ := 179.29

theorem actual_average_height_is_correct :
  ((incorrect_average_height * number_of_boys - (incorrect_height - actual_height)) / number_of_boys).round = rounded_actual_average_height.round :=
by
  sorry

end actual_average_height_is_correct_l812_812326


namespace perpendicular_lines_slope_l812_812663

theorem perpendicular_lines_slope (a : ℝ) : 
  let line1 := λ x, a * x - 2
  let line2 := λ x, (a + 2) * x + 1 
  (a * (a + 2) = -1) → a = -1 := 
by 
  intro h
  sorry

end perpendicular_lines_slope_l812_812663


namespace lexie_and_tom_apple_picking_l812_812172

theorem lexie_and_tom_apple_picking :
    ∀ (lexie_apples : ℕ),
    lexie_apples = 12 →
    (let tom_apples := 2 * lexie_apples in
     let total_apples := lexie_apples + tom_apples in
     total_apples = 36) :=
by
  intros lexie_apples h_lexie_apples
  let tom_apples := 2 * lexie_apples
  let total_apples := lexie_apples + tom_apples
  rw h_lexie_apples at *
  simp at *
  exact sorry

end lexie_and_tom_apple_picking_l812_812172


namespace necessary_and_sufficient_condition_l812_812040

universe u

variables {Point : Type u} 
variables (Plane : Type u) (Line : Type u)
variables (α β : Plane) (l : Line)
variables (P Q : Point)
variables (is_perpendicular : Plane → Plane → Prop)
variables (is_on_plane : Point → Plane → Prop)
variables (is_on_line : Point → Line → Prop)
variables (PQ_perpendicular_to_l : Prop) 
variables (PQ_perpendicular_to_β : Prop)
variables (line_in_plane : Line → Plane → Prop)

-- Given conditions
axiom plane_perpendicular : is_perpendicular α β
axiom plane_intersection : ∀ (α β : Plane), is_perpendicular α β → ∃ l : Line, line_in_plane l β
axiom point_on_plane_alpha : is_on_plane P α
axiom point_on_line : is_on_line Q l

-- Problem statement
theorem necessary_and_sufficient_condition :
  (PQ_perpendicular_to_l ↔ PQ_perpendicular_to_β) :=
sorry

end necessary_and_sufficient_condition_l812_812040


namespace arccos_one_eq_zero_l812_812452

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812452


namespace prob_A_plus_B_l812_812765

variable {R : Type*} [Field R]

noncomputable def f (A B : R) (x : R) : R := A * x + B
noncomputable def g (A B : R) (x : R) : R := B * x + A

theorem prob_A_plus_B (A B : R) (h1 : A ≠ B) (h2 : B - A = 2) (h3 : ∀ x, f(A, B) (g(A, B) x) - g(A, B) (f(A, B) x) = B^2 - A^2) : 
  A + B = -1 / 2 := 
  sorry

end prob_A_plus_B_l812_812765


namespace minor_arc_length_unit_circle_l812_812116

theorem minor_arc_length_unit_circle (A P : ℝ × ℝ) 
  (hA : A = (1, 0)) 
  (hP : P = (1/2, (sqrt 3)/2)) 
  (h_on_circle : A.1^2 + A.2^2 = 1 ∧ P.1^2 + P.2^2 = 1) :
  arc_length A P = π / 3 :=
by
  sorry

end minor_arc_length_unit_circle_l812_812116


namespace arccos_one_eq_zero_l812_812430

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812430


namespace sally_forgot_poems_l812_812799

theorem sally_forgot_poems (initially_memorized : ℕ) (recited_correctly : ℕ) (mixed_up : ℕ) :
  initially_memorized = 15 → recited_correctly = 5 → mixed_up = 3 → (initially_memorized - (recited_correctly + mixed_up) = 7) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end sally_forgot_poems_l812_812799


namespace largest_multiple_of_9_less_than_100_l812_812301

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ (∀ k, k < 100 → k % 9 = 0 → k ≤ n) :=
by {
  use 99,
  split,
  { exact lt_of_le_of_ne (nat.le_of_dvd (by norm_num) (by norm_num)) (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by use 11) },
  { intros k hk hkm, apply nat.mul_le_mul_left 9 (le_of_lt $ nat.div_lt_self hk $ by norm_num), exact le_of_lt_succ (nat.succ_le_of_lt (nat.div_lt_self hk $ by norm_num)) }
  sorry
}

end largest_multiple_of_9_less_than_100_l812_812301


namespace min_ratio_value_l812_812086

noncomputable def min_ratio (C D : ℝ) := D - 2 / D

theorem min_ratio_value (x : ℝ) (C D : ℝ) (hC : x^2 + 1 / x^2 = C) 
    (hD : x + 1 / x = D) (hPosC : 0 < C) (hPosD : 0 < D) : 
    C / D ≥ 1 :=
by
  have h1 : D^2 = C + 2, from sorry, -- D^2 = (x + 1/x)^2 = x^2 + 2 + 1/x^2
  have h2 : C = D^2 - 2, from sorry, -- C = D^2 - 2
  have h3 : C / D = D - 2 / D, from sorry, -- (D^2 - 2) / D = D - 2 / D
  have hD_ge_2 : D ≥ 2, from sorry, -- Since x + 1/x ≥ 2 for x > 0
  have h4 : D - 2 / D ≥ 1, from sorry, -- D - 2/D ≥ 1 with D ≥ 2
  exact h4

end min_ratio_value_l812_812086


namespace sin_double_angle_plus_pi_over_3_l812_812684

variables (α : ℝ)
hypothesis (α_acute : 0 < α ∧ α < π / 2)
hypothesis (cos_condition : Real.cos (α + π / 6) = 3 / 5)

theorem sin_double_angle_plus_pi_over_3 :
  Real.sin (2 * α + π / 3) = 24 / 25 :=
by
  sorry

end sin_double_angle_plus_pi_over_3_l812_812684


namespace total_items_bought_l812_812952

theorem total_items_bought (total_money pies_price coffee_price service_fee : ℝ) 
  (total_money = 50) 
  (pies_price = 7) 
  (coffee_price = 2.5) 
  (service_fee = 6) : 
  let available_money := total_money - service_fee,
      max_pies := ⌊available_money / pies_price⌋.to_nat in
  max_pies + ⌊(available_money - (max_pies * pies_price)) / coffee_price⌋.to_nat = 6 :=
by
  sorry

end total_items_bought_l812_812952


namespace sum_of_possible_sums_eq_nine_l812_812992

theorem sum_of_possible_sums_eq_nine :
  ∑ (a b : ℕ) in (finset.filter (λ ab : ℕ × ℕ, ∃ x : ℕ, 4^ab.1 + 2^ab.2 + 5 = x^2) 
             finset.univ) (ab : ℕ × ℕ), ab.1 + ab.2 = 9 :=
by
  sorry

end sum_of_possible_sums_eq_nine_l812_812992


namespace range_of_a_l812_812651

def f (x : ℝ) : ℝ :=
if 0 ≤ x then (1 / 2 : ℝ) ^ x else (1 / Real.exp 1) ^ x

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ set.Icc (1 - 2 * a) (1 + 2 * a), f (2 * x + a) ≥ (f x) ^ 3) ↔ (0 < a ∧ a ≤ 1 / 3) :=
by
  sorry

end range_of_a_l812_812651


namespace max_true_statements_l812_812154

theorem max_true_statements (x : ℝ) :
  (0 < x^2 ∧ x^2 < 1 ∧ (¬ (x^2 > 1)) ∧ (¬ (-1 < x < 0)) ∧ 0 < x < 1 ∧ (¬ (0 < x^3 - x < 1))) ∨
  (0 < x^2 ∧ x^2 < 1 ∧ (¬ (x^2 > 1)) ∧ (¬ (0 < x < 1)) ∧ (¬ (¬ (-1 < x < 0))) ∧ (¬ (0 < x^3 - x < 1))) →
  true → sorry :=
begin
  sorry
end

end max_true_statements_l812_812154


namespace total_fuel_l812_812780

theorem total_fuel (fuel_this_week : ℝ) (reduction_percent : ℝ) :
  fuel_this_week = 15 → reduction_percent = 0.20 → 
  (fuel_this_week + (fuel_this_week * (1 - reduction_percent))) = 27 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end total_fuel_l812_812780


namespace trains_pass_each_other_in_8_64_seconds_l812_812327

noncomputable def relative_speed_km_h (speed_1 speed_2 : ℕ) : ℝ :=
  (speed_1 + speed_2 : ℕ)

noncomputable def relative_speed_m_s (relative_speed_km_h : ℝ) : ℝ :=
  relative_speed_km_h * 1000 / 3600

noncomputable def total_distance (length_1 length_2 : ℕ) : ℝ :=
  (length_1 + length_2 : ℕ)

noncomputable def time_to_pass_each_other (total_distance relative_speed_m_s : ℝ) : ℝ :=
  total_distance / relative_speed_m_s

theorem trains_pass_each_other_in_8_64_seconds :
  ∀ (len_1 len_2 : ℕ) (speed_1 speed_2 : ℕ),
  len_1 = 210 → len_2 = 210 → speed_1 = 90 → speed_2 = 85 →
  time_to_pass_each_other
    (total_distance len_1 len_2)
    (relative_speed_m_s (relative_speed_km_h speed_1 speed_2)) ≈ 8.64 :=
by
  intros len_1 len_2 speed_1 speed_2 h_len1 h_len2 h_speed1 h_speed2
  rw [h_len1, h_len2, h_speed1, h_speed2]
  sorry

end trains_pass_each_other_in_8_64_seconds_l812_812327


namespace number_of_students_l812_812237

theorem number_of_students (buses : ℕ) (seats_per_bus : ℕ) 
  (hbuses : buses = 37) (hseats_per_bus : seats_per_bus = 3) :
  let n := buses * seats_per_bus in
  n = 111 := 
by
  rw [hbuses, hseats_per_bus]
  show 37 * 3 = 111 from calc
    37 * 3 = 111 : by norm_num
  rfl

end number_of_students_l812_812237


namespace range_j_l812_812153

def h (x : ℝ) : ℝ := 2 * x + 3

def j (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_j : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 61 ≤ j x ∧ j x ≤ 93) := 
by 
  sorry

end range_j_l812_812153


namespace complement_of_angle_l812_812026

variable (α : ℝ)

axiom given_angle : α = 63 + 21 / 60

theorem complement_of_angle :
  90 - α = 26 + 39 / 60 :=
by
  sorry

end complement_of_angle_l812_812026


namespace odd_function_decreasing_function_solution_set_l812_812057

/-- Given the function f(x) = (-2^x + 1) / (2^(x+1) + 2), prove:
1. f(x) is an odd function.
2. f(x) is a decreasing function on ℝ.
3. The solution set for the inequality f(x) > -1/6 is {x | x < 1} --/
namespace Proof

def f (x : ℝ) : ℝ :=
  (-2^x + 1) / (2^(x + 1) + 2)

theorem odd_function : ∀ x : ℝ, f (-x) = -f(x) := by
  sorry

theorem decreasing_function : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) > f(x2) := by
  sorry

theorem solution_set : {x : ℝ | f(x) > -1/6} = {x : ℝ | x < 1} := by
  sorry

end Proof

end odd_function_decreasing_function_solution_set_l812_812057


namespace arccos_one_eq_zero_l812_812484

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812484


namespace arccos_one_eq_zero_l812_812559

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812559


namespace students_called_back_l812_812850

theorem students_called_back (girls boys not_called_back called_back : ℕ) 
  (h1 : girls = 17)
  (h2 : boys = 32)
  (h3 : not_called_back = 39)
  (h4 : called_back = (girls + boys) - not_called_back):
  called_back = 10 := by
  sorry

end students_called_back_l812_812850


namespace largest_multiple_of_9_less_than_100_l812_812287

theorem largest_multiple_of_9_less_than_100 : ∃ x : ℕ, 9 * x < 100 ∧ ∀ y : ℕ, 9 * y < 100 → y ≤ x :=
by
  exists 11
  split
  · linarith
  · intro y h
    have : 9 * y < 100 := h
    calc
      y ≤ floor (11.11) : by linarith

end largest_multiple_of_9_less_than_100_l812_812287


namespace equation_of_W_perimeter_of_rectangle_ABCD_l812_812717

-- Definition for the locus W
def locusW (x y : ℝ) : Prop :=
  y = x^2 + 1/4

-- Proof for the equation of W
theorem equation_of_W (x y : ℝ) :
  (abs y = sqrt (x^2 + (y - 1/2)^2)) ↔ locusW x y := 
sorry

-- Prove the perimeter of rectangle ABCD
theorem perimeter_of_rectangle_ABCD (A B C D : ℝ × ℝ) :
  (locusW A.1 A.2 ∧ locusW B.1 B.2 ∧ locusW C.1 C.2 ∧ (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) ∧
  (B.1 - C.1) * (D.2 - C.2) = (D.1 - B.1) * (C.2 - B.2) ∧ (D.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (D.2 - A.2)) →
  (4 < 3 * sqrt 3) := 
sorry

end equation_of_W_perimeter_of_rectangle_ABCD_l812_812717


namespace area_of_grey_part_l812_812858

theorem area_of_grey_part :
  let area1 := 8 * 10
  let area2 := 12 * 9
  let area_black := 37
  let area_white := 43
  area2 - area_white = 65 :=
by
  let area1 := 8 * 10
  let area2 := 12 * 9
  let area_black := 37
  let area_white := 43
  have : area2 - area_white = 65 := by sorry
  exact this

end area_of_grey_part_l812_812858


namespace projection_vector_magnitude_l812_812672

def a : ℝ × ℝ := (16, -2)
def b : ℝ × ℝ := (1, 3)
def lambda := 1 -- derived from the condition that should be solved first, using lambda value directly

noncomputable def magnitude_of_projection : ℝ :=
(abs ((a.1 + lambda * b.1) * b.1 + (a.2 + lambda * b.2) * b.2) / real.sqrt ((b.1)^2 + (b.2)^2))

theorem projection_vector_magnitude :
  magnitude_of_projection = 2 * real.sqrt 10 :=
sorry

end projection_vector_magnitude_l812_812672


namespace simplify_expression_l812_812807

theorem simplify_expression (x : ℝ) (hx_nonzero : x ≠ 0) (hx_range : 0 < x ∧ x ≤ 1) :
  1.37 * (∛(2 * x^2 / (9 + 18 * x + 9 * x^2))) * (sqrt((1 + x) * ∛(1 - x) / x)) * (∛(3 * sqrt(1 - x^2) / (2 * x * sqrt(x)))) = ∛((1 - x) / (3 * x)) :=
by
  sorry

end simplify_expression_l812_812807


namespace grains_in_gray_areas_l812_812113

theorem grains_in_gray_areas (total_grains_circle1 : ℕ) (total_grains_circle2 : ℕ) (common_grains_in_overlap : ℕ)
  (h1 : total_grains_circle1 = 110) (h2 : total_grains_circle2 = 87) (h3 : common_grains_in_overlap = 68) :
  total_grains_circle1 - common_grains_in_overlap + (total_grains_circle2 - common_grains_in_overlap) = 61 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end grains_in_gray_areas_l812_812113


namespace tangent_curve_line_l812_812069

/-- Given the line y = x + 1 and the curve y = ln(x + a) are tangent, prove that the value of a is 2. -/
theorem tangent_curve_line (a : ℝ) :
  (∃ x₀ y₀, y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧ (1 / (x₀ + a) = 1)) → a = 2 :=
by
  sorry

end tangent_curve_line_l812_812069


namespace arithmetic_mean_inequality_l812_812767

theorem arithmetic_mean_inequality (n : ℕ) (hn : n > 0) :
  (1 / n) * (∑ k in Finset.range n, Real.sqrt (k + 1)) > (2 / 3) * Real.sqrt n := 
sorry

end arithmetic_mean_inequality_l812_812767


namespace sum_of_slope_and_x_intercept_is_infinity_l812_812184

noncomputable def line_through_points (C D : ℝ × ℝ) : Prop :=
  C.1 = 8 ∧ D.1 = 8 ∧ C.2 ≠ D.2

theorem sum_of_slope_and_x_intercept_is_infinity (C D : ℝ × ℝ) 
  (h : line_through_points C D) : 
  (⊤ : ℝ) + 8 = ⊤ :=
by 
  sorry

end sum_of_slope_and_x_intercept_is_infinity_l812_812184


namespace car_traveling_speed_l812_812348

-- Definition of the two speeds and the extra time
def speed_100 : ℝ := 100 -- Speed in km/h
def distance : ℝ := 1 -- Distance in km
def extra_time : ℝ := 2 -- Extra time in seconds

-- Conversion factor from km/h to km/s
def hours_to_seconds : ℝ := 3600

-- Time to travel 1 km at 100 km/h in seconds
def time_100 := distance / speed_100 * hours_to_seconds

-- Time to travel 1 km at the car's speed in seconds
def time_car := time_100 + extra_time

-- Car's speed in km/h
noncomputable def car_speed := distance / (time_car / hours_to_seconds)

-- The theorem to prove
theorem car_traveling_speed : abs (car_speed - 94.74) < 0.01 := by
  sorry

end car_traveling_speed_l812_812348


namespace largest_multiple_of_9_less_than_100_l812_812273

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ ∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n :=
begin
  use 99,
  split,
  { norm_num }, -- 99 < 100
  split,
  { norm_num }, -- 99 % 9 = 0
  { intros m hm,
    obtain ⟨k, rfl⟩ := nat.exists_eq_mul_right_of_dvd (dvd_of_mod_eq_zero hm.2),
    rw [mul_comm, nat.mul_lt_mul_iff_left (by norm_num : 0 < 9)],
    norm_num,
    exact nat.succ_le_iff.mpr (le_of_lt hm.1), }
end

end largest_multiple_of_9_less_than_100_l812_812273


namespace trajectory_of_B_is_hyperbola_l812_812640

-- Definitions based on given conditions
variables (l : Line) (alpha : Plane) (P : Point) (B : Point)

-- Assumptions
axiom line_parallel_to_plane : l.is_parallel_to alpha
axiom point_on_line : P ∈ l
axiom point_in_plane : B ∈ alpha
axiom angle_cone_condition : ∀ B, ∠PB l = 30

-- Theorem to prove
theorem trajectory_of_B_is_hyperbola :
  trajectory B = hyperbola :=
sorry

end trajectory_of_B_is_hyperbola_l812_812640


namespace rectangle_horizontal_length_l812_812228

theorem rectangle_horizontal_length (s v : ℕ) (h : ℕ) 
  (hs : s = 80) (hv : v = 100) 
  (eq_perimeters : 4 * s = 2 * (v + h)) : h = 60 :=
by
  sorry

end rectangle_horizontal_length_l812_812228


namespace arccos_one_eq_zero_l812_812561

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812561


namespace xyz_value_l812_812043

variable {x y z : ℝ}

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
                  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
                x * y * z = 6 := by
  sorry

end xyz_value_l812_812043


namespace ana_guarantees_painted_board_l812_812751

noncomputable def min_m_value : ℕ :=
  88

theorem ana_guarantees_painted_board (m : ℕ) (h : m ≤ 2024) 
  (game : ∀ k : ℕ, (k ≤ m → bool) → bool) 
  (ana_move : ∀ (k : ℕ) (h_k : k ≤ m), Prop)
  (banana_move : ∀ k : ℕ, Prop) :
  m = min_m_value :=
by
  sorry

end ana_guarantees_painted_board_l812_812751


namespace exists_arith_seq_distinct_l812_812889

/-- For every integer k ≥ 1, there exists an arithmetic sequence x_i in ℚ₊ˣ such that
  when written in irreducible form x_i = a_i / b_i with a_i, b_i ≥ 1 and gcd(a_i, b_i) = 1,
  the integers a_1, ..., a_k, b_1, ..., b_k are distinct. -/
theorem exists_arith_seq_distinct (k : ℕ) (hk : k ≥ 1) :
  ∃ (x : ℕ → ℚ) (a b : ℕ → ℕ),
    (∀ i < k, x i = (a i : ℚ) / (b i : ℚ)) ∧
    (∀ i < k, a i ≥ 1 ∧ b i ≥ 1 ∧ Nat.gcd (a i) (b i) = 1) ∧
    (∀ i j < k, i ≠ j → a i ≠ a j ∧ b i ≠ b j) := by
  sorry

end exists_arith_seq_distinct_l812_812889


namespace lisa_heavier_than_sam_l812_812843

def jack_weight : ℝ := 52
def total_weight : ℝ := 210
def sam_weight : ℝ := 65       -- Based on the condition that Jack's weight is 20% less than Sam's weight
def lisa_weight : ℝ := 52 * 1.4 -- Based on the condition that Lisa weighs 40% more than Jack

theorem lisa_heavier_than_sam : lisa_weight - sam_weight = 7.8 := 
by 
  unfold lisa_weight
  unfold sam_weight
  norm_num
  sorry

end lisa_heavier_than_sam_l812_812843


namespace equation_of_W_rectangle_perimeter_gt_3sqrt3_l812_812731

namespace MathProof

/- Defining the conditions and the questions -/
def point (P : ℝ × ℝ) := P.fst
def P (x y : ℝ) := (x, y)

def dist_x_axis (y : ℝ) := |y|
def dist_to_point (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + (y - 1/2)^2)

-- Condition: The distance from point P to the x-axis equals distance to (0, 1/2)
def condition_P (x y : ℝ) : Prop :=
  dist_x_axis y = dist_to_point x y

-- Locus W
def W_eq (x : ℝ) : ℝ := x^2 + 1/4

/- Proving the statements -/
theorem equation_of_W (x y : ℝ) (h : condition_P x y): 
  y = W_eq x :=
sorry

-- Perimeter of rectangle ABCD
structure rectangle :=
(A : ℝ × ℝ)
(B : ℝ × ℝ)
(C : ℝ × ℝ)
(D : ℝ × ℝ)

def on_W (P : ℝ × ℝ) : Prop :=
  P.snd = W_eq P.fst

def perimeter (rect : rectangle) : ℝ :=
  let d1 := Real.sqrt ((rect.A.fst - rect.B.fst)^2 + (rect.A.snd - rect.B.snd)^2)
  let d2 := Real.sqrt ((rect.B.fst - rect.C.fst)^2 + (rect.B.snd - rect.C.snd)^2)
  let d3 := Real.sqrt ((rect.C.fst - rect.D.fst)^2 + (rect.C.snd - rect.D.snd)^2)
  let d4 := Real.sqrt ((rect.D.fst - rect.A.fst)^2 + (rect.D.snd - rect.A.snd)^2)
  d1 + d2 + d3 + d4

theorem rectangle_perimeter_gt_3sqrt3 
(rect : rectangle) 
(hA : on_W rect.A) 
(hB : on_W rect.B) 
(hC : on_W rect.C) :
  perimeter rect > 3 * Real.sqrt 3 :=
sorry

end MathProof

end equation_of_W_rectangle_perimeter_gt_3sqrt3_l812_812731


namespace resulting_ratio_correct_l812_812746

-- Define initial conditions
def initial_coffee : ℕ := 20
def joe_drank : ℕ := 3
def joe_added_cream : ℕ := 4
def joAnn_added_cream : ℕ := 3
def joAnn_drank : ℕ := 4

-- Define the resulting amounts of cream
def joe_cream : ℕ := joe_added_cream
def joAnn_initial_cream_frac : ℚ := joAnn_added_cream / (initial_coffee + joAnn_added_cream)
def joAnn_cream_drank : ℚ := (joAnn_drank : ℚ) * joAnn_initial_cream_frac
def joAnn_cream_left : ℚ := joAnn_added_cream - joAnn_cream_drank

-- Define the resulting ratio of cream in Joe's coffee to JoAnn's coffee
def resulting_ratio : ℚ := joe_cream / joAnn_cream_left

-- Theorem stating the resulting ratio is 92/45
theorem resulting_ratio_correct : resulting_ratio = 92 / 45 :=
by
  unfold resulting_ratio joe_cream joAnn_cream_left joAnn_cream_drank joAnn_initial_cream_frac
  norm_num
  sorry

end resulting_ratio_correct_l812_812746


namespace value_of_2a_plus_b_l812_812215

theorem value_of_2a_plus_b : ∀ (a b : ℝ), (∀ x : ℝ, x^2 - 4*x + 7 = 19 → (x = a ∨ x = b)) → a ≥ b → 2 * a + b = 10 :=
by
  intros a b h_sol h_order
  sorry

end value_of_2a_plus_b_l812_812215


namespace triangle_parallel_lines_perimeter_l812_812855

theorem triangle_parallel_lines_perimeter 
  (AB BC AC : ℕ) 
  (AB_eq : AB = 150) 
  (BC_eq : BC = 270) 
  (AC_eq : AC = 210) 
  (l_A_len : ℕ) 
  (l_B_len : ℕ) 
  (l_C_len : ℕ) 
  (l_A_eq : l_A_len = 60) 
  (l_B_eq : l_B_len = 50) 
  (l_C_eq : l_C_len = 20) 
: (perimeter_triangle_parallel_lines AB BC AC l_A_len l_B_len l_C_len) = 270 :=
sorry

end triangle_parallel_lines_perimeter_l812_812855


namespace inverse_exists_A_inverse_exists_D_inverse_exists_F_inverse_exists_G_inverse_exists_H_l812_812978

noncomputable def p (x : ℝ) : ℝ := real.sqrt (3 - x)
def q (x : ℝ) : ℝ := x^4 - x^2
def r (x : ℝ) : ℝ := x + 2 / x
def s (x : ℝ) : ℝ := 3*x^2 + 6*x + 8
def t (x : ℝ) : ℝ := abs (x - 3) + abs (x + 4)
def u (x : ℝ) : ℝ := 2^x + 6^x
def v (x : ℝ) : ℝ := x - 2 / x
def w (x : ℝ) : ℝ := x / 3

theorem inverse_exists_A : ∃ f : ℝ → ℝ, function.right_inverse f p := sorry
theorem inverse_exists_D : ∃ f : ℝ → ℝ, function.right_inverse f s := sorry
theorem inverse_exists_F : ∃ f : ℝ → ℝ, function.right_inverse f u := sorry
theorem inverse_exists_G : ∃ f : ℝ → ℝ, function.right_inverse f v := sorry
theorem inverse_exists_H : ∃ f : ℝ → ℝ, function.right_inverse f w := sorry

end inverse_exists_A_inverse_exists_D_inverse_exists_F_inverse_exists_G_inverse_exists_H_l812_812978


namespace distinct_roots_condition_l812_812599

theorem distinct_roots_condition (a : ℝ) : 
  (∃ (x1 x2 x3 x4 : ℝ), (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧ 
  (|x1^2 - 4| = a * x1 + 6) ∧ (|x2^2 - 4| = a * x2 + 6) ∧ (|x3^2 - 4| = a * x3 + 6) ∧ (|x4^2 - 4| = a * x4 + 6)) ↔ 
  ((-3 < a ∧ a < -2 * Real.sqrt 2) ∨ (2 * Real.sqrt 2 < a ∧ a < 3)) := sorry

end distinct_roots_condition_l812_812599


namespace cos_magnitude_a_l812_812998

theorem cos_magnitude_a (x₁ x₂ : ℝ) (h₀ : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 * π)
  (h₁ : sin x₁ = 1 / 3) (h₂ : sin x₂ = 1 / 3)
  (OA OB : ℝ × ℝ) (hOA : OA = (x₁, sin x₁)) (hOB : OB = (x₂, sin x₂)) :
  ∃ a : ℝ × ℝ, a = (OA.1 - OB.1, OA.2 - OB.2) ∧ cos (|a|) = -7 / 9 := 
sorry

end cos_magnitude_a_l812_812998


namespace ratio_of_three_numbers_l812_812248

noncomputable def A (x : ℕ) := 40 * x
noncomputable def B (y : ℕ) := 40 * y
noncomputable def C (z : ℕ) := 40 * z

theorem ratio_of_three_numbers (x y z : ℕ) :
  LCM (A x) (LCM (B y) (C z)) = 2400 → gcd (A x) (gcd (B y) (C z)) = 40 → x * y * z = 60 ∧ x = 3 ∧ y = 4 ∧ z = 5 :=
by
  sorry

end ratio_of_three_numbers_l812_812248


namespace product_of_integers_l812_812615

theorem product_of_integers (A B C D : ℚ)
  (h1 : A + B + C + D = 100)
  (h2 : A + 5 = B - 5)
  (h3 : A + 5 = 2 * C)
  (h4 : A + 5 = D / 2) :
  A * B * C * D = 1517000000 / 6561 := by
  sorry

end product_of_integers_l812_812615


namespace log2_lt_one_iff_l812_812033

theorem log2_lt_one_iff (x : ℝ) : log 2 x < 1 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end log2_lt_one_iff_l812_812033


namespace equation_of_W_perimeter_of_rectangle_ABCD_l812_812716

-- Definition for the locus W
def locusW (x y : ℝ) : Prop :=
  y = x^2 + 1/4

-- Proof for the equation of W
theorem equation_of_W (x y : ℝ) :
  (abs y = sqrt (x^2 + (y - 1/2)^2)) ↔ locusW x y := 
sorry

-- Prove the perimeter of rectangle ABCD
theorem perimeter_of_rectangle_ABCD (A B C D : ℝ × ℝ) :
  (locusW A.1 A.2 ∧ locusW B.1 B.2 ∧ locusW C.1 C.2 ∧ (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) ∧
  (B.1 - C.1) * (D.2 - C.2) = (D.1 - B.1) * (C.2 - B.2) ∧ (D.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (D.2 - A.2)) →
  (4 < 3 * sqrt 3) := 
sorry

end equation_of_W_perimeter_of_rectangle_ABCD_l812_812716


namespace base8_perfect_square_b_zero_l812_812093

-- Define the base 8 representation and the perfect square condition
def base8_to_decimal (a b : ℕ) : ℕ := 512 * a + 64 + 8 * b + 4

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating that if the number in base 8 is a perfect square, then b = 0
theorem base8_perfect_square_b_zero (a b : ℕ) (h₀ : a ≠ 0) 
  (h₁ : is_perfect_square (base8_to_decimal a b)) : b = 0 :=
sorry

end base8_perfect_square_b_zero_l812_812093


namespace altitude_triangle_l812_812363

variables (a b : ℝ)
def rectangle_area := a * b
def diagonal_of_rectangle := Real.sqrt (a^2 + b^2)
def triangle_area_on_diagonal (h : ℝ) := (1/2) * diagonal_of_rectangle a b * h

theorem altitude_triangle (a b : ℝ) (h : ℝ) (h_area : triangle_area_on_diagonal a b h = rectangle_area a b) :
  h = (2 * a * b) / (Real.sqrt (a^2 + b^2)) :=
by sorry

end altitude_triangle_l812_812363


namespace tournament_log2_n_l812_812021

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem tournament_log2_n :
  let teams := 40
  let games_total := teams * (teams - 1) / 2
  let outcomes_total := 2 ^ games_total
  let win_counts_unique := factorial teams
  let factors_of_two_in_factorial :=
       (Array.map (λ k, teams / 2 ^ k) (List.range (Nat.log2 (teams * 2))) |>.map Nat.floor).sum,
  let probability_simplified_denominator := 2 ^ (games_total - factors_of_two_in_factorial)
  ∃ (n : ℕ), 
    probability_simplified_denominator = n ∧ 
    Nat.log2 n = 742 :=
by {
  sorry
}

end tournament_log2_n_l812_812021


namespace sum_squares_of_roots_l812_812835

def a := 8
def b := 12
def c := -14

theorem sum_squares_of_roots : (b^2 - 2 * a * c)/(a^2) = 23/4 := by
  sorry

end sum_squares_of_roots_l812_812835


namespace range_of_g_l812_812643

noncomputable def f (a x : ℝ) := a * x^2 + x + a
noncomputable def g (a x : ℝ) := 2^x + a

theorem range_of_g (a : ℝ) (h₀ : ∀ y : ℝ, ∃ x : ℝ, f a x = y) 
  (h₁ : ∀ x : ℝ, f a x ≤ 3 / 4) : (∃ x : ℝ, g a x ≥ -1 / 4) ∧ (∀ y : ℝ, g a y < ∞) :=
by 
  sorry

end range_of_g_l812_812643


namespace ratio_correct_l812_812307

-- Define the side length of the equilateral triangle
def side_length := 15

-- Define the perimeter of the equilateral triangle
def perimeter := 3 * side_length

-- Define the altitude of the equilateral triangle using a known geometric property
def altitude := (side_length * Real.sqrt 3) / 2

-- Define the area of the equilateral triangle using the base and height
def area := (1 / 2) * side_length * altitude

-- Define the ratio of the area of the equilateral triangle to the square of its perimeter
def ratio : ℝ := area / (perimeter^2)

-- Prove that the ratio is equal to sqrt(3) / 36
theorem ratio_correct : ratio = Real.sqrt 3 / 36 := by
  sorry

end ratio_correct_l812_812307


namespace general_formula_inequality_l812_812050

noncomputable def sequence_sum (n : ℕ) : ℕ → ℕ := sorry

noncomputable def a (n : ℕ) : ℕ := sorry

theorem general_formula (n : ℕ) (S : ℕ → ℕ) 
    (h1 : ∀ n, S n = ∑ i in range n, a i)
    (h2 : ∀ n, (∑ i in range n, S i) = 4 * a n - 2 * n - 4)
    : ∀ n, a n = 2 ^ n := sorry

noncomputable def b (n : ℕ) : ℝ := real.log2 (a n)

theorem inequality (n : ℕ) 
    (S : ℕ → ℕ) 
    (h1 : ∀ n, S n = ∑ i in range n, a i)
    (h2 : ∀ n, (∑ i in range n, S i) = 4 * a n - 2 * n - 4) : 
    ∑ i in range n, (1 / (b i)^2) < 5 / 3 := sorry

end general_formula_inequality_l812_812050


namespace sqrt_of_factorial_div_l812_812000

theorem sqrt_of_factorial_div : 
  √(9! / 126) = 8 * √5 :=
by
  sorry

end sqrt_of_factorial_div_l812_812000


namespace arccos_one_eq_zero_l812_812465

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812465


namespace existence_of_x2_with_sum_ge_2_l812_812769

variables (a b c x1 x2 : ℝ) (h_root1 : a * x1^2 + b * x1 + c = 0) (h_x1_pos : x1 > 0)

theorem existence_of_x2_with_sum_ge_2 :
  ∃ x2, (c * x2^2 + b * x2 + a = 0) ∧ (x1 + x2 ≥ 2) :=
sorry

end existence_of_x2_with_sum_ge_2_l812_812769


namespace converse_of_implication_l812_812825

-- Given propositions p and q
variables (p q : Prop)

-- Proving the converse of "if p then q" is "if q then p"

theorem converse_of_implication (h : p → q) : q → p :=
sorry

end converse_of_implication_l812_812825


namespace polynomial_rational_coeffs_l812_812130

theorem polynomial_rational_coeffs (P : Polynomial ℚ) :
  (∀ x : ℚ, P.eval x ∈ ℚ) → ∀ i : ℕ, P.coeff i ∈ ℚ := by
  sorry

end polynomial_rational_coeffs_l812_812130


namespace arccos_one_eq_zero_l812_812405

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812405


namespace part_I_part_II_l812_812048

def f (a x : ℝ) : ℝ := a^x

def S (a : ℝ) (n : ℕ) : ℝ := f a n - 1

def a_n (a : ℝ) (n : ℕ) : ℝ := S a n - S a (n-1)

def b_n (a : ℝ) (n : ℕ) : ℝ := n / a_n a (n + 1)

def T (a : ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b_n a i -- sum from 0 to n-1

theorem part_I (a : ℝ) (n : ℕ) (h_a : a = 2)
  (h_cond1: ∀ n, S a n = 2^n - 1) :
  a_n a n = 2^(n-1) :=
sorry

theorem part_II (a : ℝ) (n : ℕ)
  (h_a : a = 2)
  (h_cond1: ∀ n, S a n = 2^n - 1)
  (h_an: ∀ n, a_n a n = 2^(n-1)) :
  T a n = 2 - (n+2)/(2^n) :=
sorry

end part_I_part_II_l812_812048


namespace arccos_one_eq_zero_l812_812471

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812471


namespace quadrilateral_is_parallelogram_l812_812840

theorem quadrilateral_is_parallelogram
    (A B C D M N P Q : Type)
    [inner_product_space ℝ (Type)]
    [metric_space A] [metric_space B] [metric_space C] [metric_space D]
    [metric_space M] [metric_space N] [metric_space P] [metric_space Q]
    (hM : midpoint A B = M) (hN : midpoint B C = N) 
    (hP : midpoint C D = P) (hQ : midpoint D A = Q) 
    (h1 : dist M P + dist N Q = (dist A B + dist B C + dist C D + dist D A) / 2) : 
  parallelogram A B C D := sorry

end quadrilateral_is_parallelogram_l812_812840


namespace arccos_one_eq_zero_l812_812560

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812560


namespace arccos_one_eq_zero_l812_812527

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812527


namespace largest_multiple_of_9_less_than_100_l812_812264

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l812_812264


namespace length_of_BI_in_triangle_l812_812125

theorem length_of_BI_in_triangle
  (A B C I : Type)
  (AB AC BC : ℝ)
  (hab : AB = 19)
  (hac : AC = 17)
  (hbc : BC = 20)
  (hI : incenter I A B C) :
  ∃ (BI : ℝ), BI ≈ 12.2 :=
sorry

end length_of_BI_in_triangle_l812_812125


namespace window_area_l812_812931

def meter_to_feet : ℝ := 3.28084
def length_in_meters : ℝ := 2
def width_in_feet : ℝ := 15

def length_in_feet := length_in_meters * meter_to_feet
def area_in_square_feet := length_in_feet * width_in_feet

theorem window_area : area_in_square_feet = 98.4252 := 
by
  sorry

end window_area_l812_812931


namespace thalassa_population_2050_l812_812984

def population_in_2000 : ℕ := 250

def population_doubling_interval : ℕ := 20

def population_linear_increase_interval : ℕ := 10

def linear_increase_amount : ℕ := 500

noncomputable def population_in_2050 : ℕ :=
  let double1 := population_in_2000 * 2
  let double2 := double1 * 2
  double2 + linear_increase_amount

theorem thalassa_population_2050 : population_in_2050 = 1500 := by
  sorry

end thalassa_population_2050_l812_812984


namespace abc_equivalence_l812_812145

theorem abc_equivalence (n : ℕ) (k : ℤ) (a b c : ℤ)
  (hn : 0 < n) (hk : k % 2 = 1)
  (h : a^n + k * b = b^n + k * c ∧ b^n + k * c = c^n + k * a) :
  a = b ∧ b = c := 
sorry

end abc_equivalence_l812_812145


namespace smallest_angle_in_right_triangle_l812_812709

noncomputable def is_consecutive_primes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ ∀ r, Nat.Prime r → p < r → r < q → False

theorem smallest_angle_in_right_triangle : ∃ p : ℕ, ∃ q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ p + q = 90 ∧ is_consecutive_primes p q ∧ p = 43 :=
by
  sorry

end smallest_angle_in_right_triangle_l812_812709


namespace find_f_100_l812_812386

theorem find_f_100 (f : ℤ → ℤ)
  (h₁ : ∀ a b, f(a + b) = f(a) + f(b) + a * b)
  (h₂ : f(75) - f(51) = 1230) : f(100) = 3825 := by
  sorry

end find_f_100_l812_812386


namespace part1_part2_l812_812054

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + x * Real.sin x + Real.cos x

-- Prove the first statement: For the curve y = f(x) to be tangent to y = b at (a, f(a))
theorem part1 (a : ℝ) (b : ℝ) (h : ∀ x : ℝ, f x = b → x = a) : a = 0 ∧ b = 1 :=
by sorry

-- Prove the second statement: For the curve y = f(x) to intersect y = b at two distinct points
theorem part2 (b : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ f x = b ∧ f y = b) : 1 < b :=
by sorry

end part1_part2_l812_812054


namespace rodney_commission_per_system_l812_812798

theorem rodney_commission_per_system :
  ∃ commission_per_system : ℕ,
  let street1_sales := 2,
      street2_sales := 4,
      street3_sales := 0,
      street4_sales := 1,
      total_sales := street1_sales + street2_sales + street3_sales + street4_sales,
      total_commission := 175
  in
  total_commission = commission_per_system * total_sales ∧ commission_per_system = 25 :=
by
  sorry

end rodney_commission_per_system_l812_812798


namespace sequence_remainder_204_l812_812761

/-- Let T be the increasing sequence of positive integers whose binary representation has exactly 6 ones.
    Let M be the 500th number in T.
    Prove that the remainder when M is divided by 500 is 204. -/
theorem sequence_remainder_204 :
    let T := {n : ℕ | (nat.popcount n = 6)}.to_list.sorted (≤)
    let M := T.get 499
    M % 500 = 204 :=
by
    sorry

end sequence_remainder_204_l812_812761


namespace arccos_one_eq_zero_l812_812403

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812403


namespace arccos_one_eq_zero_l812_812434

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812434


namespace opposite_number_statements_correct_l812_812380

theorem opposite_number_statements_correct :
  (∀ a : ℝ, ¬ (a = -a) ∨ a = 0) ∧
  (∀ a b : ℝ, (a * b < 0 → ¬ (a = -b))) ∧
  (∀ a : ℝ, Real.dist 0 a = Real.dist 0 (-a)) ∧
  (∀ a : ℝ, a = -a) ∧
  (∀ a b : ℚ, (a = -b → a ≠ 0 → b ≠ 0 → ¬ (a * b > 0))) →
  ( (∀ a : ℝ, Real.dist 0 a = Real.dist 0 (-a)) ∧
    (∀ a : ℝ, a = -a) ) :=
by
  sorry

end opposite_number_statements_correct_l812_812380


namespace largest_multiple_of_9_less_than_100_l812_812284

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n < 100 ∧ n % 9 = 0 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 0 → m ≤ n :=
by
  use 99
  split
  · exact dec_trivial
  split 
  · exact dec_trivial
  intro m hm
  cases hm 
  cases hm_right 
  have h : m ≤ 99 / 1 := by norm_cast; simp only [Nat.le_div_iff_mul_le dec_trivial, mul_one, div_one]
  exact h
  sorry -- Complete the proof

end largest_multiple_of_9_less_than_100_l812_812284


namespace total_money_received_l812_812874

theorem total_money_received (zachary_games: ℕ) (price_mainstream: ℕ) 
  (perc_jason_zachary: ℕ) (extra_ryan: ℕ) (perc_emily_ryan: ℕ) (extra_lily: ℕ) 
  (total: ℕ) 
  (h_zachary: zachary_games = 40)
  (h_price_mainstream: price_mainstream = 5)
  (h_perc_jason_zachary: perc_jason_zachary = 30)
  (h_extra_ryan: extra_ryan = 50)
  (h_perc_emily_ryan: perc_emily_ryan = 20)
  (h_extra_lily: extra_lily = 70)
  (h_total: total = 1336) :
  
  let zachary := zachary_games * price_mainstream in
  let jason := zachary + (perc_jason_zachary * zachary / 100) in
  let ryan := jason + extra_ryan in
  let emily := ryan - (perc_emily_ryan * ryan / 100) in
  let lily := emily + extra_lily in
  zachary + jason + ryan + emily + lily = total :=
  
  by
    -- Variables declarations
    let zachary := zachary_games * price_mainstream
    let jason := zachary + (perc_jason_zachary * zachary / 100)
    let ryan := jason + extra_ryan
    let emily := ryan - (perc_emily_ryan * ryan / 100)
    let lily := emily + extra_lily
    show zachary + jason + ryan + emily + lily = total
    sorry

end total_money_received_l812_812874


namespace arccos_one_eq_zero_l812_812463

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812463


namespace min_moves_is_correct_l812_812257
-- Adding necessary imports

-- Defining the problem variables
def N := 2019
def l := 100

-- The smallest number of moves that makes all boxes contain the same number of stones
def min_moves : ℕ := ⌈(N * N) / l⌉

-- Proving that the minimum number of moves is 40762
theorem min_moves_is_correct : min_moves = 40762 := by
  -- Assuming the facts about N and l from the problem
  let d := Nat.gcd N l
  have d_eq_one : d = 1 := by sorry
  -- Calculation that ⌈2019^2 / 100⌉ = 40762
  have calculation : ⌈(N * N) / l⌉ = 40762 := by sorry

  -- Combining these facts
  exact calculation

end min_moves_is_correct_l812_812257


namespace cone_volume_ratio_l812_812253

theorem cone_volume_ratio (h_C r_C h_D r_D : ℝ)
  (hC_eq : h_C = 16) (rC_eq : r_C = 8)
  (hD_eq : h_D = 8) (rD_eq : r_D = 16) :
  (1/3 * π * r_C^2 * h_C) / (1/3 * π * r_D^2 * h_D) = 1/2 :=
by {
  rw [hC_eq, rC_eq, hD_eq, rD_eq],
  -- calculation steps would go here
  sorry
}

end cone_volume_ratio_l812_812253


namespace minimum_positive_period_of_cos_func_l812_812831

-- Define the function cosine with a phase shift and an angular frequency
def cos_func (x : ℝ) : ℝ := Real.cos (4 * x + Real.pi / 3)

-- Define the expected period
def expected_period : ℝ := Real.pi / 2

-- The theorem stating that the minimum positive period of the function cos_func is expected_period
theorem minimum_positive_period_of_cos_func : ∃ (T : ℝ), T > 0 ∧ ∀ x : ℝ, cos_func(x + T) = cos_func x ∧ T = expected_period :=
by
  sorry

end minimum_positive_period_of_cos_func_l812_812831


namespace samantha_routes_l812_812800

theorem samantha_routes : 
  let routes_from_house_to_park := Nat.choose 5 2,
      routes_through_park := 2,
      routes_from_park_to_school := Nat.choose 6 3
  in
  routes_from_house_to_park * routes_through_park * routes_from_park_to_school = 400 := 
by
  sorry

end samantha_routes_l812_812800


namespace area_codes_count_l812_812928

theorem area_codes_count : 
  let digits := {2, 4, 5}
  let codes := {x // (∀ n ∈ x, n ∈ digits) ∧ x.length = 3 ∧ (∃ (a b c : ℕ), x = [a, b, c] ∧ a * b * c % 2 = 0 ∧ a + b + c = 12)}
  in codes.card = 4 :=
by
  sorry

end area_codes_count_l812_812928


namespace neg_of_if_pos_then_real_roots_l812_812766

variable (m : ℝ)

def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b * x + c = 0

theorem neg_of_if_pos_then_real_roots :
  (∀ m : ℝ, m > 0 → has_real_roots 1 1 (-m) )
  → ( ∀ m : ℝ, m ≤ 0 → ¬ has_real_roots 1 1 (-m) ) := 
sorry

end neg_of_if_pos_then_real_roots_l812_812766


namespace dog_moves_away_from_cat_l812_812903

theorem dog_moves_away_from_cat :
  let cat := (15, 12)
  let dog_start := (3, -3)
  let dog_path : ℝ → ℝ := λ x, -4 * x + 15
  let c := 27 / 17
  let d := 135 / 68
  let c_d_sum := 243 / 68 in
  ∃ (c d : ℝ), 
    let dog_move_away_point := (c, d)
    in (y - cat.2 = (1 / 4) * (x - cat.1))
    ∧ (dog_path c = d)
    ∧ (c + d = c_d_sum) :=
begin
  sorry
end

end dog_moves_away_from_cat_l812_812903


namespace prove_composite_k_l812_812039

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := ∃ p q, p > 1 ∧ q > 1 ∧ n = p * q

def problem_statement (a b c d : ℕ) (h : a * b = c * d) : Prop :=
  is_composite (a^1984 + b^1984 + c^1984 + d^1984)

-- The theorem to prove
theorem prove_composite_k (a b c d : ℕ) (h : a * b = c * d) : 
  problem_statement a b c d h := sorry

end prove_composite_k_l812_812039


namespace plane_distance_l812_812918

theorem plane_distance :
  let distance_AB := 100
  let distance_BC := distance_AB + 50
  let distance_CD := 2 * distance_BC
  let total_distance_AD := distance_AB + distance_BC + distance_CD
  total_distance_AD = 550 :=
by
  intros
  let distance_AB := 100
  let distance_BC := distance_AB + 50
  let distance_CD := 2 * distance_BC
  let total_distance_AD := distance_AB + distance_BC + distance_CD
  sorry

end plane_distance_l812_812918


namespace age_difference_is_24_l812_812961

theorem age_difference_is_24 (d f : ℕ) (h1 : d = f / 9) (h2 : f + 1 = 7 * (d + 1)) : f - d = 24 := sorry

end age_difference_is_24_l812_812961


namespace time_between_rings_is_288_minutes_l812_812089

def intervals_between_rings (total_rings : ℕ) (total_minutes : ℕ) : ℕ := 
  let intervals := total_rings - 1
  total_minutes / intervals

theorem time_between_rings_is_288_minutes (total_minutes_in_day total_rings : ℕ) 
  (h1 : total_minutes_in_day = 1440) (h2 : total_rings = 6) : 
  intervals_between_rings total_rings total_minutes_in_day = 288 := 
by 
  sorry

end time_between_rings_is_288_minutes_l812_812089


namespace arccos_one_eq_zero_l812_812509

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812509


namespace largest_multiple_of_9_less_than_100_l812_812259

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l812_812259


namespace reach_all_cities_in_one_intermediate_l812_812245

-- Define the cities
inductive City
| C1 | C2 | C3 | C4 | C5 | C6 | C7

open City

-- Define the road connections as a directed graph
def road (from to : City) : Prop :=
  match from, to with
  | C1, C2 => true
  | C1, C3 => true
  | C2, C3 => true
  | C2, C4 => true
  | C3, C4 => true
  | C3, C5 => true
  | C4, C5 => true
  | C4, C6 => true
  | C5, C6 => true
  | C5, C7 => true
  | C6, C7 => true
  | C6, C1 => true
  | C7, C1 => true
  | C7, C2 => true
  | _, _ => false

-- Define reachability within one intermediate city
def reachable_in_one (from to : City) : Prop :=
  road from to ∨ ∃ (mid : City), road from mid ∧ road mid to

-- The proof statement
theorem reach_all_cities_in_one_intermediate :
  ∀ (u v : City), reachable_in_one u v :=
by
  intros
  -- Proof omitted
  sorry

end reach_all_cities_in_one_intermediate_l812_812245


namespace arccos_one_eq_zero_l812_812433

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812433


namespace measure_angle_CAG_l812_812983

-- Define the vertices of the figures and the angles.
variables (A B C G : Point)
variables (m_ANGLE_ACB m_ANGLE_BCG : ℝ) (is_equilateral : is_equilateral_triangle A B C) (is_rectangle : is_rectangle B C F G)
variable (AC_eq_CG : length A C = length C G)

-- Define degrees in the required calculations.
variable degree : ℝ := 180 / 𝜋

-- The statement to prove:
theorem measure_angle_CAG (h_eq1 : m_ANGLE_ACB = 60 / degree) (h_eq2 : m_ANGLE_BCG = 90 / degree):
  angle A C G = 15 / degree :=
by sorry

end measure_angle_CAG_l812_812983


namespace alice_bob_not_next_to_each_other_l812_812713

open Nat

theorem alice_bob_not_next_to_each_other (A B C D E : Type) :
  let arrangements := 5!
  let together := 4! * 2
  arrangements - together = 72 :=
by
  let arrangements := 5!
  let together := 4! * 2
  sorry

end alice_bob_not_next_to_each_other_l812_812713


namespace new_average_income_l812_812208

/-!
# Average Monthly Income Problem

## Problem Statement
Given:
1. The average monthly income of a family of 4 earning members was Rs. 735.
2. One of the earning members died, and the average income changed.
3. The income of the deceased member was Rs. 1170.

Prove that the new average monthly income of the family is Rs. 590.
-/

theorem new_average_income (avg_income : ℝ) (num_members : ℕ) (income_deceased : ℝ) (new_num_members : ℕ) 
  (h1 : avg_income = 735) 
  (h2 : num_members = 4) 
  (h3 : income_deceased = 1170) 
  (h4 : new_num_members = 3) : 
  (num_members * avg_income - income_deceased) / new_num_members = 590 := 
by 
  sorry

end new_average_income_l812_812208


namespace arccos_one_eq_zero_l812_812496

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812496


namespace two_numbers_sum_l812_812856

theorem two_numbers_sum (N1 N2 : ℕ) (h1 : N1 % 10^5 = 0) (h2 : N2 % 10^5 = 0) 
  (h3 : N1 ≠ N2) (h4 : (Nat.divisors N1).card = 42) (h5 : (Nat.divisors N2).card = 42) : 
  N1 + N2 = 700000 := 
by
  sorry

end two_numbers_sum_l812_812856


namespace sqrt_of_factorial_div_l812_812001

theorem sqrt_of_factorial_div : 
  √(9! / 126) = 8 * √5 :=
by
  sorry

end sqrt_of_factorial_div_l812_812001


namespace paula_twice_as_old_as_karl_6_years_later_l812_812897

theorem paula_twice_as_old_as_karl_6_years_later
  (P K : ℕ)
  (h1 : P - 5 = 3 * (K - 5))
  (h2 : P + K = 54) :
  P + 6 = 2 * (K + 6) :=
sorry

end paula_twice_as_old_as_karl_6_years_later_l812_812897


namespace number_of_odd_integers_between_l812_812678

theorem number_of_odd_integers_between : 
  ∃ n : ℕ, 4 = n ∧ ∀ k ∈ {9, 11, 13, 15}, n = 4 :=
sorry

end number_of_odd_integers_between_l812_812678


namespace tank_capacity_l812_812219

theorem tank_capacity (one_third_full : ℚ) (added_water : ℚ) (capacity : ℚ) 
  (h1 : one_third_full = 1 / 3) 
  (h2 : 2 * one_third_full * capacity = 16) 
  (h3 : added_water = 16) 
  : capacity = 24 := 
by
  sorry

end tank_capacity_l812_812219


namespace arccos_one_eq_zero_l812_812404

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812404


namespace square_root_of_factorial_division_l812_812006

theorem square_root_of_factorial_division :
  (sqrt (9! / 126) = 24 * sqrt 5) :=
sorry

end square_root_of_factorial_division_l812_812006


namespace angle_AQH_is_90_l812_812805

noncomputable def angle_AQH (A B G H Q : Point) : Prop :=
  -- Preconditions describing the regular octagon and the extension
  RegularOctagon A B G H ∧ ExtendsToMeet A B G H Q →
  Angle A Q H = 90

theorem angle_AQH_is_90 (A B G H Q : Point) :
  angle_AQH A B G H Q :=
sorry

end angle_AQH_is_90_l812_812805


namespace max_min_values_l812_812224

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + 2 * cos x

theorem max_min_values :
  ∃ y_max y_min,
    y_max = (7 / 4) ∧
    y_min = -2 ∧
    (∀ x, (π / 3) ≤ x ∧ x ≤ (4 * π / 3) → f x ≤ y_max) ∧
    (∀ x, (π / 3) ≤ x ∧ x ≤ (4 * π / 3) → y_min ≤ f x) :=
sorry

end max_min_values_l812_812224


namespace arccos_one_eq_zero_l812_812521

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812521


namespace prove_k_range_l812_812671

open Real

def vector_dot (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def vector_norm (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def vector_cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  vector_dot v1 v2 / (vector_norm v1 * vector_norm v2)

def obtuse_angle_condition (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  vector_dot v1 v2 < 0

def problem_conditions (m k : ℝ) : Prop :=
  let a := (1, 1, 0) in
  let b := (m, 0, 2) in
  vector_cos_angle a b = -sqrt 10 / 10 ∧
  obtuse_angle_condition (1 + k * -1, 1 + k * 0, 0 + k * 2) (2 * 1 + -1, 2 * 1 + 0, 2 * 0 + 2)

theorem prove_k_range (m k : ℝ) (h : problem_conditions m k) : k < -1 := sorry

end prove_k_range_l812_812671


namespace arccos_one_eq_zero_l812_812401

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812401


namespace determine_other_number_l812_812982

theorem determine_other_number (a b : ℤ) (h₁ : 3 * a + 4 * b = 161) (h₂ : a = 17 ∨ b = 17) : 
(a = 31 ∨ b = 31) :=
by
  sorry

end determine_other_number_l812_812982


namespace july_husband_current_age_l812_812872

-- Define the initial ages and the relationship between Hannah and July's age
def hannah_initial_age : ℕ := 6
def hannah_july_age_relation (hannah_age july_age : ℕ) : Prop := hannah_age = 2 * july_age

-- Define the time that has passed and the age difference between July and her husband
def time_passed : ℕ := 20
def july_husband_age_relation (july_age husband_age : ℕ) : Prop := husband_age = july_age + 2

-- Lean statement to prove July's husband's current age
theorem july_husband_current_age : ∃ (july_age husband_age : ℕ),
  hannah_july_age_relation hannah_initial_age july_age ∧
  july_husband_age_relation (july_age + time_passed) husband_age ∧
  husband_age = 25 :=
by
  sorry

end july_husband_current_age_l812_812872


namespace limit_f_ratio_eq_c_l812_812574

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

def p (S : List Bool) : ℝ :=
  let indices := S.reverse.enum.filterMap (fun ⟨i, b⟩ => if b then some 𝜙^i else none)
  indices.sum

def f (n : ℕ) : ℕ :=
  (List.filter (fun S => p S = (phi^48*n - 1)/(phi^48 - 1)) (List.range 2^n)).length

theorem limit_f_ratio_eq_c :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs ((f (n+1) : ℝ) / (f n) - ((25 + 3 * Real.sqrt 69) / 2)) < ε :=
sorry

end limit_f_ratio_eq_c_l812_812574


namespace sum_of_squares_lt_one_l812_812627

noncomputable def sequence (x : ℕ → ℝ) : Prop :=
  x 0 > 0 ∧ x 0 < 1 ∧ ∀ k, x (k + 1) = x k - x k ^ 2

theorem sum_of_squares_lt_one (x : ℕ → ℝ) (hx : sequence x) (n : ℕ) :
  (∑ i in Finset.range (n + 1), (x i) ^ 2) < 1 := 
sorry

end sum_of_squares_lt_one_l812_812627


namespace equation_is_true_l812_812594

theorem equation_is_true :
  10 * 6 - (9 - 3) * 2 = 48 :=
by
  sorry

end equation_is_true_l812_812594


namespace product_approximation_l812_812949

-- Define the infinite product for 1 - 1/10^k for k from 1 to n
def product_P (n : ℕ) : ℝ :=
  ∏ k in Finset.range n, (1 - (1 / 10^k))

-- Statement to prove
theorem product_approximation : 
  ∃ ε : ℝ, ε = 10^(-5) ∧ (|product_P 99 - 0.89001| < ε) :=
by
  sorry

end product_approximation_l812_812949


namespace arccos_one_eq_zero_l812_812554

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812554


namespace num_modified_monotonous_pos_integers_l812_812951

/-- A number is modified-monotonous if it is a one-digit number or its digits
    form a non-decreasing or non-increasing sequence and the digit '0' can be repeated
    at most once if it appears. -/
def modified_monotonous (n : ℕ) : Prop :=
  (n < 10) ∨ (list.sorted (≤) (nat.digits 10 n)) ∨
  (list.sorted (≥) (nat.digits 10 n) ∧ (nat.digits 10 n).count 0 ≤ 1)

theorem num_modified_monotonous_pos_integers : 
  (finset.filter modified_monotonous (finset.range 1000000000)).card = 2426 := 
sorry

end num_modified_monotonous_pos_integers_l812_812951


namespace remainder_correct_l812_812967

noncomputable def remainder_division : ℕ → ℕ → Polynomial ℚ → Polynomial ℚ → Polynomial ℚ
| n1, n2, poly_y50, poly_divisor := 
  let c := n1 - n2 in
  let d := n2 - 2 * n1 + 2 * n2 in
  c * X - d * X + d - c

theorem remainder_correct (y : ℚ) : 
  let P := Polynomial.C 1
  let Q := Polynomial.X^2 - Polynomial.C 5 * Polynomial.X + Polynomial.C 6
  let R := 2^50 * (Polynomial.X - 3) - 3^50 * (Polynomial.X - 2)
  Polynomial.eval y (Polynomial.modByMonic (Polynomial.X^50) Q) = Polynomial.eval y R :=
by
  sorry

end remainder_correct_l812_812967


namespace total_amount_l812_812323

def share_ratio (x y z a b c : ℝ) := y = x * a ∧ z = x * b ∧ c = a + b + 1

theorem total_amount 
  (x y z : ℝ) 
  (a b c m : ℝ)
  (h₁ : share_ratio x y z a b c)
  (h₂ : y = 45)
  (h₃ : a = 0.45)
  (h₄ : b = 0.3)
  (h₅ : m = y / a) 
  : m * c = 175 :=
by
  unfold share_ratio at h₁
  have hx : y = x * 0.45 := by
    rw [←h₃, ←and.left h₁]
  have hz : z = x * 0.3 := by
    rw [←h₄, ←and.right (and.right h₁)]
  have hc : c = 1.75 := by
    rw [←and.right (and.right h₁), h₃, h₄]
    norm_num
  rw [h₂, h₃, h₅]
  field_simp
  norm_num
  rw hc
  norm_num

-- sorry

end total_amount_l812_812323


namespace sufficient_condition_for_A_l812_812072

variables {A B C : Prop}

theorem sufficient_condition_for_A (h1 : A ↔ B) (h2 : C → B) : C → A :=
sorry

end sufficient_condition_for_A_l812_812072


namespace arccos_1_eq_0_l812_812419

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812419


namespace radical_axis_theorem_l812_812617

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

def power_of_point (p : Point) (c : Circle) : ℝ :=
  ((p.x - c.center.x)^2 + (p.y - c.center.y)^2 - c.radius^2)

theorem radical_axis_theorem (O1 O2 : Circle) :
  ∃ L : ℝ → Point, 
  (∀ p : Point, (power_of_point p O1 = power_of_point p O2) → (L p.x = p)) ∧ 
  (O1.center.y = O2.center.y) ∧ 
  (∃ k : ℝ, ∀ x, L x = Point.mk x k) :=
sorry

end radical_axis_theorem_l812_812617


namespace harry_has_19_apples_l812_812852

def apples_problem := 
  let A_M := 68  -- Martha's apples
  let A_T := A_M - 30  -- Tim's apples (68 - 30)
  let A_H := A_T / 2  -- Harry's apples (38 / 2)
  A_H = 19

theorem harry_has_19_apples : apples_problem :=
by
  -- prove A_H = 19 given the conditions
  sorry

end harry_has_19_apples_l812_812852


namespace max_value_geometric_product_l812_812035

open Nat

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)

theorem max_value_geometric_product
  (h1 : a 1 + a 3 = 10)
  (h2 : a 2 + a 4 = 5)
  (hgeom : ∀ n, a (n+1) = a n * q) :
  ∃ n, ∏ i in range n, a (i + 1) ≤ 64 :=
sorry

end max_value_geometric_product_l812_812035


namespace find_t_value_l812_812013

noncomputable def t_value : ℚ :=
  let v : ℚ × ℚ × ℚ → ℚ → ℚ × ℚ × ℚ := λ p t, (p.1 + 3 * t, p.2 - 4 * t, p.3 + t)
  let dot_prod : ℚ × ℚ × ℚ → ℚ × ℚ × ℚ → ℚ :=
    λ p q, p.1 * q.1 + p.2 * q.2 + p.3 * q.3
  (λ t, dot_prod (v (1, 2, 3) t) (3, -4, 1)) 1 / 13

theorem find_t_value : t_value = 1 / 13 := 
by
  sorry

end find_t_value_l812_812013


namespace percentage_difference_l812_812916

theorem percentage_difference (x y : ℝ) (h : x = 12 * y) : (1 - y / x) * 100 = 91.67 :=
by {
  sorry
}

end percentage_difference_l812_812916


namespace determine_coefficients_l812_812964

-- Define the polynomial function f
def f (a b x : ℝ) : ℝ := a * x^4 - 7 * x^3 + b * x^2 - 12 * x - 8

-- Define the main theorem statement
theorem determine_coefficients (a b : ℝ) :
  (f a b 2 = -7) ∧ (f a b (-3) = -80) → a = -9/4 ∧ b = 29.25 :=
by {
  intros h,
  cases h with h1 h2,
  sorry
}

end determine_coefficients_l812_812964


namespace age_difference_l812_812395

theorem age_difference (C D m : ℕ) 
  (h1 : C = D + m)
  (h2 : C - 1 = 3 * (D - 1)) 
  (h3 : C * D = 72) : 
  m = 9 :=
sorry

end age_difference_l812_812395


namespace infinite_series_closed_form_l812_812820

noncomputable def series (a : ℝ) : ℝ :=
  ∑' (k : ℕ), (2 * (k + 1) - 1) / a^k

theorem infinite_series_closed_form (a : ℝ) (ha : 1 < a) : 
  series a = (a^2 + a) / (a - 1)^2 :=
sorry

end infinite_series_closed_form_l812_812820


namespace perpendicular_lines_l812_812762

variables {α β : Type*} [plane α] [plane β]
variables {l m : Type*} [line l] [line m]

theorem perpendicular_lines
  (distinct_planes : α ≠ β)
  (distinct_lines : l ≠ m)
  (plane_perpendicular : α ⊥ β)
  (line_perpendicular_to_plane1 : l ⊥ α)
  (line_perpendicular_to_plane2 : m ⊥ β) :
  l ⊥ m :=
sorry

end perpendicular_lines_l812_812762


namespace western_novels_shelved_l812_812176

-- Definitions based on conditions in the problem
def total_books_initial := 46
def history_books := 12
def romance_books := 8
def poetry_books := 4
def biographies := 6

-- The problem statement in Lean 4
theorem western_novels_shelved (total_books_initial history_books romance_books poetry_books biographies : ℕ) 
  (h1 : total_books_initial = 46)
  (h2 : history_books = 12)
  (h3 : romance_books = 8)
  (h4 : poetry_books = 4)
  (h5 : biographies = 6) :
  let top_books := history_books + romance_books + poetry_books,
      bottom_books := total_books_initial - top_books,
      mystery_books := bottom_books / 2,
      remaining_books := bottom_books - mystery_books,
      western_novels := remaining_books - biographies
  in western_novels = 5 :=
by {
  sorry
}

end western_novels_shelved_l812_812176


namespace increasing_interval_sin_l812_812832

theorem increasing_interval_sin (k : ℤ) : 
  let f : ℝ → ℝ := λ x, sin(π / 4 - 2 * x) in 
  ∀ x, (k * π + 3 * π / 8) ≤ x ∧ x ≤ (k * π + 7 * π / 8) ↔ increasing [f x | k ∈ ℤ] :=
sorry

end increasing_interval_sin_l812_812832


namespace not_in_range_of_f_x_l812_812020

open Real

theorem not_in_range_of_f_x (b : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + 3 ≠ -3) ↔ b ∈ Ioo (-2 * sqrt 6) (2 * sqrt 6) :=
by
  sorry

end not_in_range_of_f_x_l812_812020


namespace num_integer_multiples_of_6006_l812_812679

theorem num_integer_multiples_of_6006 : 
  (∃ n : ℕ, n = 6006 ∧ 
   ∀ i j : ℕ, 0 ≤ i → i < j → j ≤ 150 → ∃ k : ℕ, 12^j - 12^i = 6006 * k)
   → (finset.range 151).sum 
          (λ i, (finset.range 151).filter (λ j, j > i ∧ (12 ^ (j - i) - 1) % 1001 = 0)).card = 1825 :=
by { sorry }


end num_integer_multiples_of_6006_l812_812679


namespace length_of_AC_l812_812123

theorem length_of_AC {A B C H X Y K : Type*}
  [metric_space A] [metric_space B] [metric_space C]
  (hABC : angle A = 90) 
  (hAH : altitude A H = distance A H)
  (circle_through_AH : metric_space.line_circle_intersect A H)
  (hX : metric_space.line_circle_intersect A X)
  (hY : metric_space.line_circle_intersect A Y)
  (AX : distance A X = 5)
  (AY : distance A Y = 6)
  (AB : distance A B = 9) :
  distance A C = 13.5 :=
begin
  sorry
end

end length_of_AC_l812_812123


namespace find_solutions_l812_812597

open Real

def solution_exists (x y u v : ℝ) : Prop :=
  (x ^ 2 + y ^ 2 + u ^ 2 + v ^ 2 = 4) ∧
  (xu + yv + xv + yu = 0) ∧
  (xyu + yuv + uvx + vxy = -2) ∧
  (xyuv = -1)

theorem find_solutions :
  (∃ x y u v : ℝ, solution_exists x y u v ∧ ((x, y, u, v) = (1, 1, 1, -1) ∨ (x, y, u, v) = (-1 + sqrt 2, -1 - sqrt 2, 1, -1))) :=
sorry

end find_solutions_l812_812597


namespace equation_of_locus_perimeter_greater_than_3sqrt3_l812_812727

theorem equation_of_locus 
  (P : ℝ × ℝ)
  (hx : ∀ y : ℝ, P.snd = |P.snd| → |P.snd| = sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2))
  :
  (P.snd = P.fst ^ 2 + 1/4) := 
sorry

theorem perimeter_greater_than_3sqrt3
  (A B C D : ℝ × ℝ)
  (h_parabola : ∀ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4)
  (h_vert_A : A ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_vert_B : B ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_vert_C : C ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_perpendicular: ((B.fst - A.fst) * (C.fst - B.fst) + (B.snd - A.snd) * (C.snd - B.snd)) = 0) 
  :
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
sorry

end equation_of_locus_perimeter_greater_than_3sqrt3_l812_812727


namespace trees_spacing_l812_812319

theorem trees_spacing (total_trees : ℕ) (length_of_road : ℝ) (num_spaces : ℕ) (distance_between_trees : ℝ) :
  total_trees = 24 →
  length_of_road = 239.66 →
  num_spaces = total_trees - 1 →
  distance_between_trees = length_of_road / num_spaces →
  distance_between_trees ≈ 10.42 :=
by
  intros h1 h2 h3 h4
  rw [h1] at h3
  rw [nat.cast_sub] at h3
  rw [← h3] at h4
  exact h4
  exact dec_trivial -- ensures a clean rw
  sorry

end trees_spacing_l812_812319


namespace tangent_line_at_perpendicular_l812_812065

noncomputable theory

-- Define the given function
def f (x a : ℝ) : ℝ := (x + a) * Real.exp x

-- Define the slope of the line perpendicular to x + y + 1 = 0
def perp_slope : ℝ := 1

-- Define tangent point coordinates
def tangent_point : ℝ × ℝ := (0, 0)

-- Define the tangent line equation
def tangent_line_eq (x : ℝ) : ℝ := x

theorem tangent_line_at_perpendicular {a : ℝ} 
  (h_deriv : ∀ x, deriv (λ x, f x a) x = (x + a + 1) * Real.exp x)
  (h_perpendicular : perp_slope = 1)
  (h_tangent : tangent_point = (0, 0)) :
  tangent_line_eq 0 = 0 :=
begin
  sorry
end

end tangent_line_at_perpendicular_l812_812065


namespace find_N_l812_812310

theorem find_N (N : ℕ) : (4^5)^2 * (2^5)^4 = 2^N → N = 30 :=
by
  intros h
  -- Sorry to skip the proof.
  sorry

end find_N_l812_812310


namespace remainder_div_1442_l812_812220

theorem remainder_div_1442 (x k l r : ℤ) (h1 : 1816 = k * x + 6) (h2 : 1442 = l * x + r) (h3 : x = Int.gcd 1810 374) : r = 0 := by
  sorry

end remainder_div_1442_l812_812220


namespace min_workers_to_complete_on_time_l812_812960

theorem min_workers_to_complete_on_time
  (total_project_length : ℕ)
  (days_already_worked : ℕ)
  (number_of_workers_initial : ℕ)
  (fraction_completed : ℚ)
  (days_remaining : ℕ)
  (fraction_needed : ℚ)
  (rate_per_day_initial : ℚ)
  (required_rate_per_day : ℚ) :
  total_project_length = 40 →
  days_already_worked = 10 →
  number_of_workers_initial = 12 →
  fraction_completed = 2/5 →
  days_remaining = 30 →
  fraction_needed = 3/5 →
  rate_per_day_initial = fraction_completed / days_already_worked →
  required_rate_per_day = fraction_needed / days_remaining →
  rate_per_day_initial = 2 * required_rate_per_day →
  number_of_workers_initial / 2 = 6 :=
begin
  sorry
end

end min_workers_to_complete_on_time_l812_812960


namespace triangle_side_difference_l812_812108

theorem triangle_side_difference :
  ∀ (x : ℕ), 8 + 13 > x ∧ x + 8 > 13 → 20 - 6 = 14 :=
by
  -- Given conditions
  assume x,
  intro h,
  -- Use triangle inequalities:
  -- 1. x + 8 > 13 (x > 5)
  -- 2. 8 + 13 > x (x < 21)
  sorry

end triangle_side_difference_l812_812108


namespace arccos_1_eq_0_l812_812417

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812417


namespace equation_of_W_rectangle_perimeter_gt_3sqrt3_l812_812732

namespace MathProof

/- Defining the conditions and the questions -/
def point (P : ℝ × ℝ) := P.fst
def P (x y : ℝ) := (x, y)

def dist_x_axis (y : ℝ) := |y|
def dist_to_point (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + (y - 1/2)^2)

-- Condition: The distance from point P to the x-axis equals distance to (0, 1/2)
def condition_P (x y : ℝ) : Prop :=
  dist_x_axis y = dist_to_point x y

-- Locus W
def W_eq (x : ℝ) : ℝ := x^2 + 1/4

/- Proving the statements -/
theorem equation_of_W (x y : ℝ) (h : condition_P x y): 
  y = W_eq x :=
sorry

-- Perimeter of rectangle ABCD
structure rectangle :=
(A : ℝ × ℝ)
(B : ℝ × ℝ)
(C : ℝ × ℝ)
(D : ℝ × ℝ)

def on_W (P : ℝ × ℝ) : Prop :=
  P.snd = W_eq P.fst

def perimeter (rect : rectangle) : ℝ :=
  let d1 := Real.sqrt ((rect.A.fst - rect.B.fst)^2 + (rect.A.snd - rect.B.snd)^2)
  let d2 := Real.sqrt ((rect.B.fst - rect.C.fst)^2 + (rect.B.snd - rect.C.snd)^2)
  let d3 := Real.sqrt ((rect.C.fst - rect.D.fst)^2 + (rect.C.snd - rect.D.snd)^2)
  let d4 := Real.sqrt ((rect.D.fst - rect.A.fst)^2 + (rect.D.snd - rect.A.snd)^2)
  d1 + d2 + d3 + d4

theorem rectangle_perimeter_gt_3sqrt3 
(rect : rectangle) 
(hA : on_W rect.A) 
(hB : on_W rect.B) 
(hC : on_W rect.C) :
  perimeter rect > 3 * Real.sqrt 3 :=
sorry

end MathProof

end equation_of_W_rectangle_perimeter_gt_3sqrt3_l812_812732


namespace arccos_one_eq_zero_l812_812532

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812532


namespace dig_same_canal_in_days_l812_812789

variables (m₁ d₁ h₁ m₂ h₂ : ℕ) (d₂ : ℝ)

-- Initial conditions
def initial_conditions := 
  (m₁ = 20) ∧ (d₁ = 3) ∧ (h₁ = 6)

-- Workload constant
def workload_constant := m₁ * d₁ * h₁

-- New conditions
def new_conditions := 
  (m₂ = 30) ∧ (h₂ = 8) ∧ (m₂ * d₂ * h₂ = workload_constant)

theorem dig_same_canal_in_days :
  initial_conditions ∧ new_conditions → d₂ = 1.5 :=
begin
  sorry
end

end dig_same_canal_in_days_l812_812789


namespace outfits_count_l812_812884

theorem outfits_count :
  let red_shirts := 7
  let green_shirts := 6
  let blue_shirts := 5
  let pants := 6
  let green_hats := 6
  let red_hats := 7
  let blue_hats := 5
  calc
    let case1 := red_shirts * pants * (green_hats + blue_hats) := 7 * 6 * (6 + 5)
    let case2 := green_shirts * pants * (red_hats + blue_hats) := 6 * 6 * (7 + 5)
    let case3 := blue_shirts * pants * (green_hats + red_hats) := 5 * 6 * (6 + 7)
    let total_outfits := case1 + case2 + case3 := 7*6*(6+5) + 6*6*(7+5) + 5*6*(6+7)
  in total_outfits = 1284
:=
by
  sorry

end outfits_count_l812_812884


namespace chris_first_day_breath_l812_812953

theorem chris_first_day_breath (x : ℕ) (h1 : x + 10 = 20) : x = 10 :=
by
  sorry

end chris_first_day_breath_l812_812953


namespace sum_of_divisors_18_l812_812607

theorem sum_of_divisors_18 : ∑ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 39 := sorry

end sum_of_divisors_18_l812_812607


namespace arccos_one_eq_zero_l812_812466

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812466


namespace smallest_positive_period_of_f_f_nonnegative_on_interval_l812_812058

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 - Real.cos (2 * x)

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
sorry

theorem f_nonnegative_on_interval : ∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f x ≥ 0 :=
sorry

end smallest_positive_period_of_f_f_nonnegative_on_interval_l812_812058


namespace unattainable_for_n_eq_12_possible_values_only_3_or_4_l812_812107

noncomputable def configuration_unattainable (n : ℕ) : Prop :=
  ∀ (a : ℕ → ℕ), 
    n > 2 → 
    (∀ i, 1 ≤ i ∧ i ≤ n → a i = 3 * (i - 1) + 1) → 
    (∑ i in finset.range n, a (i+1) = (n^2 + n) / 2) → 
    n ≠ 12

noncomputable def possible_values_exists (n : ℕ) : Prop :=
  ∀ (a : ℕ → ℕ),
    n ≥ 3 → 
    (∀ i, 1 ≤ i ∧ i ≤ n → a i = 3 * (i - 1) + 1) → 
    (∑ i in finset.range n, a (i+1) = (n^2 + n) / 2) →
    n = 3 ∨ n = 4

-- Lean statements to prove the given conditions
theorem unattainable_for_n_eq_12 : configuration_unattainable 12 := sorry

theorem possible_values_only_3_or_4 : ∀ n, possible_values_exists n := sorry

end unattainable_for_n_eq_12_possible_values_only_3_or_4_l812_812107


namespace algebraic_cofactor_neg3_l812_812014

def determinant_3x3 (a b c d e f g h i : ℤ) : ℤ :=
  a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g

def minor_2x2 (a b c d : ℤ) : ℤ :=
  a * d - b * c

theorem algebraic_cofactor_neg3 : 
  let M_12 := minor_2x2 4 3 (-1) 1 in
  (-1) ^ (1 + 2) * M_12 = -7 :=
  by
    let M_12 := minor_2x2 4 3 (-1) 1
    have h1 : M_12 = 7 := by
      unfold minor_2x2 
      calc 4 * 1 - 3 * (-1) = 4 + 3 := by ring
      ... = 7 := rfl
    show (-1) ^ (1 + 2) * M_12 = -7
    -- proof continuation for algebraic cofactor omitted
    sorry

end algebraic_cofactor_neg3_l812_812014


namespace distance_CD_l812_812756

open Real

theorem distance_CD
  (α β : ℝ)
  (h1 : α - β = π / 3) :
  (let Cx := 5 * cos α,
       Cy := 5 * sin α,
       Dx := 12 * cos β,
       Dy := 12 * sin β in
   sqrt ((Dx - Cx)^2 + (Dy - Cy)^2) = 13) := by
  sorry

end distance_CD_l812_812756


namespace integral_value_l812_812119

noncomputable def binomial_term_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def general_term (a x : ℝ) (r k : ℕ) : ℝ :=
  (binomial_term_coefficient 12 r) * (a^(12 - r)) * ((-2017)^k) * (x^((r - 4037 * k) / 2))

theorem integral_value (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ k : ℕ, k = 0 ∧ ∃ r : ℕ, r = 10 ∧ binomial_term_coefficient 12 r * a^2 = 264) : 
  ∫ x in 0..a, (Real.exp x + 2 * x) = Real.exp 2 + 3 := 
by
  sorry

end integral_value_l812_812119


namespace largest_multiple_of_9_less_than_100_l812_812292

theorem largest_multiple_of_9_less_than_100 : ∃ x : ℕ, 9 * x < 100 ∧ ∀ y : ℕ, 9 * y < 100 → y ≤ x :=
by
  exists 11
  split
  · linarith
  · intro y h
    have : 9 * y < 100 := h
    calc
      y ≤ floor (11.11) : by linarith

end largest_multiple_of_9_less_than_100_l812_812292


namespace arithmetic_sequence_sum_l812_812391

theorem arithmetic_sequence_sum :
  3 * (75 + 77 + 79 + 81 + 83) = 1185 := by
  sorry

end arithmetic_sequence_sum_l812_812391


namespace arithmetic_sequence_fourth_term_l812_812646

-- Define the arithmetic sequence and conditions
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
def a₂ := 606
def S₄ := 3834

-- Problem statement
theorem arithmetic_sequence_fourth_term :
  (a 1 + a 2 + a 3 = 1818) →
  (a 4 = 2016) :=
sorry

end arithmetic_sequence_fourth_term_l812_812646


namespace arccos_one_eq_zero_l812_812472

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812472


namespace interest_percent_l812_812177

theorem interest_percent (purchase_price down_payment monthly_payment : ℝ)
(monthly_payments_count : ℕ) :=
(purchase_price = 118 ∧ down_payment = 18 ∧ monthly_payment = 10 ∧ monthly_payments_count = 12) →
let total_paid := down_payment + monthly_payment * monthly_payments_count in
let interest_paid := total_paid - purchase_price in
let interest_percent := (interest_paid / purchase_price) * 100 in
interest_percent = 16.9 :=
begin
  intro h,
  cases h with h1 h,
  cases h with h2 h,
  cases h with h3 h4,
  all_goals sorry,
end

end interest_percent_l812_812177


namespace arccos_one_eq_zero_l812_812500

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812500


namespace pdxy_cyclic_l812_812155

theorem pdxy_cyclic
  (A B C D P Q X Y : Point)
  (h_acute : acute_triangle A B C)
  (h_isosceles : AB = AC)
  (h_D_on_BC : D ∈ line B C)
  (h_circle_D : ∃ (circle_D : Circle), center circle_D = D ∧ C ∈ circle circle_D)
  (h_circle_intersect : ∃ (circumcircle_ABD : Circle), is_circumcircle circumcircle_ABD A B D ∧ 
                        { P, Q } = circle_intersection (circle_D.circle) circumcircle_ABD ∧ closer Q B)
  (h_BQ_intersection : (line B Q).intersect (line A D) = X)
  (h_BQ_intersection_AC : (line B Q).intersect (line A C) = Y) :
  cyclic_quad P D X Y :=
sorry

end pdxy_cyclic_l812_812155


namespace arccos_one_eq_zero_l812_812501

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812501


namespace jenna_total_bill_l812_812134

-- Definitions from the conditions
def original_bill := 500
def first_late_charge_rate := 0.01
def second_late_charge_rate := 0.02
def additional_late_fee := 5

-- Statement of the problem
theorem jenna_total_bill : 
  let first_late_bill := original_bill * (1 + first_late_charge_rate),
      second_late_bill := first_late_bill * (1 + second_late_charge_rate),
      total_amount := second_late_bill + additional_late_fee
  in total_amount = 520.1 := 
by
  unfold first_late_bill second_late_bill total_amount,
  norm_num,
  sorry

end jenna_total_bill_l812_812134


namespace area_of_circle_above_line_l812_812575

noncomputable def circle_area_above_line (x y : ℝ) : ℝ :=
if x^2 + y^2 - 8*x - 16*y + 48 = 0 ∧ y > x then sqrt 32 else 0

theorem area_of_circle_above_line : ∫ above (x y : ℝ), circle_area_above_line x y = 24 * π := 
sorry

end area_of_circle_above_line_l812_812575


namespace arccos_one_eq_zero_l812_812507

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812507


namespace postage_formula_l812_812996

def floor (x : ℝ) : ℤ := Int.floor x

def postage (W : ℝ) : ℤ := -6 * floor (-W)

theorem postage_formula (W : ℝ) : 
  postage W = -6 * (floor (-W)) :=
sorry

end postage_formula_l812_812996


namespace real_roots_of_equation_l812_812199

theorem real_roots_of_equation (a b : ℝ) : 
  (∀ x : ℝ, (1 - a * x) / (1 + a * x) * real.sqrt((1 + b * x) / (1 - b * x)) = 1 → x ∈ ℝ) ↔ (1 / 2 < a / b ∧ a / b < 1) :=
by sorry

end real_roots_of_equation_l812_812199


namespace arccos_one_eq_zero_l812_812429

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812429


namespace max_value_f_in_interval_l812_812641

-- Definitions based on the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4 * x else -(x^2 - 4 * -x)

-- Lean 4 statement for the problem
theorem max_value_f_in_interval :
  is_odd_function f →
  (∀ x, x ≥ 0 → f x = x^2 - 4 * x) →
  ∃ M, M = 4 ∧ ∀ x ∈ set.Icc (-4 : ℝ) (1 : ℝ), f x ≤ M :=
by
  sorry

end max_value_f_in_interval_l812_812641


namespace total_fuel_two_weeks_l812_812778

def fuel_used_this_week : ℝ := 15
def percentage_less_last_week : ℝ := 0.2
def fuel_used_last_week : ℝ := fuel_used_this_week * (1 - percentage_less_last_week)
def total_fuel_used : ℝ := fuel_used_this_week + fuel_used_last_week

theorem total_fuel_two_weeks : total_fuel_used = 27 := 
by
  -- Placeholder for the proof
  sorry

end total_fuel_two_weeks_l812_812778


namespace father_l812_812914

theorem father's_age : 
  ∀ (M F : ℕ), 
  (M = (2 : ℚ) / 5 * F) → 
  (M + 10 = (1 : ℚ) / 2 * (F + 10)) → 
  F = 50 :=
by
  intros M F h1 h2
  sorry

end father_l812_812914


namespace remainder_500th_T_div_500_l812_812759

-- Define the sequence T
def T : ℕ → ℕ :=
λ n, sorry -- definition for translating n-th element with 6 ones in binary

-- The 500th element in the sequence T
def M : ℕ := T 500

-- The statement to prove
theorem remainder_500th_T_div_500 :
  M % 500 = 24 :=
sorry

end remainder_500th_T_div_500_l812_812759


namespace matrix_vector_combination_l812_812763

variables (N : Matrix (Fin 2) (Fin 2) ℝ) 
variables (a b : Vector (Fin 2) ℝ)
variables hN_a : N.mulVec a = ![3, -7]
variables hN_b : N.mulVec b = ![-4, 3]

theorem matrix_vector_combination :
  N.mulVec (3 • a - 2 • b) = ![17, -27] :=
by
  sorry

end matrix_vector_combination_l812_812763


namespace total_fuel_two_weeks_l812_812777

def fuel_used_this_week : ℝ := 15
def percentage_less_last_week : ℝ := 0.2
def fuel_used_last_week : ℝ := fuel_used_this_week * (1 - percentage_less_last_week)
def total_fuel_used : ℝ := fuel_used_this_week + fuel_used_last_week

theorem total_fuel_two_weeks : total_fuel_used = 27 := 
by
  -- Placeholder for the proof
  sorry

end total_fuel_two_weeks_l812_812777


namespace decimal_representation_and_100th_digit_l812_812590

-- Define the repeating decimal structure and the digit in a specific position
def repeating_decimal (n : ℕ) (d : ℕ) (r : ℕ) : ℕ :=
  ((n - 1) % r) + 1

-- Provide the conditions as definitions in the Lean statement
def repeating_cycle_length : ℕ := 3

def repeating_decimal_value : ℕ -> ℕ
| 1 := 1
| 2 := 7
| 3 := 5
| _ := 0

-- Define the given condition
def decimal_representation := "2.1\overset{\cdot }{7}5\overset{\cdot }{6}"

-- Prove the equivalent math problem using Lean's statement
theorem decimal_representation_and_100th_digit:
  decimal_representation = "2.1\overset{\cdot }{7}5\overset{\cdot }{6}" ∧
  repeating_decimal 100 repeating_cycle_length = 6 :=
by sorry

end decimal_representation_and_100th_digit_l812_812590


namespace complement_U_A_l812_812662

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 5, 7}

theorem complement_U_A : (U \ A) = {3, 9} :=
by
  sorry

end complement_U_A_l812_812662


namespace area_of_circle_outside_triangle_l812_812148

-- Definition of the right triangle ABC with BC = 12 and ∠BAC = 90°
def right_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB AC BC : ℝ) (BAC_right_angle : angle A B C = 90) (BC_eq_twelve : BC = 12) :=
true

-- Definition of the circle with radius r = 6
def circle (r : ℝ) (tangent_AB AC : Prop) (diameter_BC : r = 6) :=
true

-- Definition of the area calculation
def area_outside_triangle (π : ℝ) (area_quarter_circle area_triangle : ℝ)
  (area_ext : area_ext = (9 * π - 18)) :=
true

-- Statement of the theorem to be proved
theorem area_of_circle_outside_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (BC : ℝ) (π : ℝ) :
  (∃ (AB AC : ℝ), right_triangle A B C AB AC BC 90 12) ∧
  (∃ (r : ℝ), circle r (tangent_AB AC) (diameter_BC 6)) →
  area_outside_triangle π (9 * π) 18 := 
by sorry

end area_of_circle_outside_triangle_l812_812148


namespace vector_angle_obtuse_l812_812668

theorem vector_angle_obtuse (a b : ℝ × ℝ × ℝ) (k : ℝ) (m : ℝ)
    (h_a : a = (1, 1, 0))
    (h_b : b = (m, 0, 2))
    (h_cos : (1: ℝ) / (Real.sqrt 2 * Real.sqrt (m^2 + 4)) = -Real.sqrt 10 / 10) :
  ((1 - k, 1, 2 * k) • (1, 2, 2) < 0) ↔ k < -1 := 
sorry

end vector_angle_obtuse_l812_812668


namespace count_special_6_digit_integers_l812_812114

theorem count_special_6_digit_integers : 
  ∃ n : ℕ, n = 11754 ∧ 
  ∀ (x : ℕ), 99999 < x ∧ x < 1000000 → 
    (∀ d : ℕ, d ∈ [0,1,2,3,4,5,6,7,8,9] → 
      (x.digit_count d) ≥ 2 → (x.digit_count = 6)) :=
begin
  sorry
end

end count_special_6_digit_integers_l812_812114


namespace arccos_one_eq_zero_l812_812467

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812467


namespace largest_multiple_of_9_less_than_100_l812_812278

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ ∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n :=
begin
  use 99,
  split,
  { norm_num }, -- 99 < 100
  split,
  { norm_num }, -- 99 % 9 = 0
  { intros m hm,
    obtain ⟨k, rfl⟩ := nat.exists_eq_mul_right_of_dvd (dvd_of_mod_eq_zero hm.2),
    rw [mul_comm, nat.mul_lt_mul_iff_left (by norm_num : 0 < 9)],
    norm_num,
    exact nat.succ_le_iff.mpr (le_of_lt hm.1), }
end

end largest_multiple_of_9_less_than_100_l812_812278


namespace max_number_of_rook_jumps_over_fence_l812_812256

noncomputable def max_rook_jumps_on_fence (chessboard : ℕ × ℕ := (8, 8))
                                           (fence : list (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)])
                                           (rook_position : ℕ × ℕ)
                                           (rook_moves : list (ℕ × ℕ)) : Prop :=
∀ (rook_position : ℕ × ℕ) (rook_moves : list (ℕ × ℕ)),
(chessboard = (8, 8) ∧ fence = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)])
→ (∀ move ∈ rook_moves,
    (1 ≤ move.1 ∧ move.1 ≤ 8) ∧ (1 ≤ move.2 ∧ move.2 ≤ 8) ∧
    ¬ (move ∈ fence))
→ ∑ move in rook_moves, (move ∈ fence) ≤ 47

theorem max_number_of_rook_jumps_over_fence : max_rook_jumps_on_fence :=
begin
  sorry
end

end max_number_of_rook_jumps_over_fence_l812_812256


namespace find_P_x_l812_812764

noncomputable def P (x : ℝ) : ℝ :=
  (-17 / 3) * x^3 + (68 / 3) * x^2 - (31 / 3) * x - 18

variable (a b c : ℝ)

axiom h1 : a^3 - 4 * a^2 + 2 * a + 3 = 0
axiom h2 : b^3 - 4 * b^2 + 2 * b + 3 = 0
axiom h3 : c^3 - 4 * c^2 + 2 * c + 3 = 0

axiom h4 : P a = b + c
axiom h5 : P b = a + c
axiom h6 : P c = a + b
axiom h7 : a + b + c = 4
axiom h8 : P 4 = -20

theorem find_P_x :
  P x = (-17 / 3) * x^3 + (68 / 3) * x^2 - (31 / 3) * x - 18 := sorry

end find_P_x_l812_812764


namespace arccos_one_eq_zero_l812_812444

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812444


namespace arccos_one_eq_zero_l812_812488

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812488


namespace function_properties_l812_812315

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (π * x / 4)

theorem function_properties :
  (∀ x : ℝ, f x = f (-x)) ∧  -- f(x) is even
  (∀ x Δ : ℝ, f (x - 2) = f (2 - x)) ∧  -- Symmetric about (-2, 0)
  (∀ x : ℝ, f x ≤ 3) :=  -- Maximum value is 3
by
  sorry

end function_properties_l812_812315


namespace odd_and_increasing_l812_812878

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f(x) < f(y)

def y1 (x : ℝ) : ℝ := x^2 + 1
def y2 (x : ℝ) : ℝ := x + 1
def y3 (x : ℝ) : ℝ := x^(1/2)
def y4 (x : ℝ) : ℝ := x^3

theorem odd_and_increasing :
  is_odd y4 ∧ is_increasing y4 ∧
  ¬ (is_odd y1 ∧ is_increasing y1) ∧
  ¬ (is_odd y2 ∧ is_increasing y2) ∧
  ¬ (is_odd y3 ∧ is_increasing y3) :=
by {
  -- Proof will go here
  sorry
}

end odd_and_increasing_l812_812878


namespace arccos_one_eq_zero_l812_812464

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812464


namespace find_m_l812_812593

noncomputable def crate_height := {2, 5, 7}

def stacks_count : ℕ := 15
def stack_height : ℕ := 60

def valid_orientations : ℕ := 
  let sol1 := nat.choose 15 5 * nat.choose 10 10
  let sol2 := nat.choose 15 7 * nat.choose 8 5 * nat.choose 3 3
  let sol3 := nat.choose 15 9 * nat.choose 6 6
  sol1 + sol2 + sol3

def total_orientations : ℕ := nat.pow 3 stacks_count

def m_n_gcd (m n : ℕ) := nat.gcd m n

theorem find_m : ∃ m n : ℕ, m = 158158 ∧ (valid_orientations = m) ∧  m_n_gcd m total_orientations = 1 := by
  sorry

end find_m_l812_812593


namespace range_of_a_l812_812163

def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2 * x > a
def proposition_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0
def proposition_r (a : ℝ) : Prop := (proposition_p a ∨ proposition_q a) ∧ ¬ (proposition_p a ∧ proposition_q a)

theorem range_of_a (a : ℝ) : proposition_r a → ((a ∈ set.Ioo (-2 : ℝ) (-1)) ∨ (a ∈ set.Ici (1 : ℝ))) :=
sorry

end range_of_a_l812_812163


namespace running_speed_ratio_l812_812979

variables (S_w S_r : ℝ)
variables (x y T : ℝ)
variables (half_way bell_before run_bell_after walk_bell_late : ℝ)

-- Conditions as Lean definitions
def half_way_condition : Prop :=
  x / 2 = T - 3

def run_bell_after_condition : Prop :=
  (y / 2) + y = T + 3

def walk_bell_late_condition : Prop :=
  (y / 2) + x = T + 15

-- The theorem we want to prove
theorem running_speed_ratio (hw : half_way_condition)
    (rb : run_bell_after_condition) (wl : walk_bell_late_condition) :
    S_r = 2 * S_w :=
sorry

end running_speed_ratio_l812_812979


namespace arccos_one_eq_zero_l812_812446

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812446


namespace green_peaches_are_six_l812_812845

/-- There are 5 red peaches in the basket. -/
def red_peaches : ℕ := 5

/-- There are 14 yellow peaches in the basket. -/
def yellow_peaches : ℕ := 14

/-- There are total of 20 green and yellow peaches in the basket. -/
def green_and_yellow_peaches : ℕ := 20

/-- The number of green peaches is calculated as the difference between the total number of green and yellow peaches and the number of yellow peaches. -/
theorem green_peaches_are_six :
  (green_and_yellow_peaches - yellow_peaches) = 6 :=
by
  sorry

end green_peaches_are_six_l812_812845


namespace propositions_correct_l812_812650

def f (x : ℝ) : ℝ := cos x * sin x

-- Lean statement for the problem conditions
theorem propositions_correct :
  (¬ ∀ x1 x2 : ℝ, f x1 = -f x2 → x1 = -x2) ∧
  (¬ ∀ p : ℝ, 0 < p → ∀ x : ℝ, f (x + p) = f x → p = 2 * Real.pi) ∧
  (∀ x1 x2 : ℝ, -Real.pi / 4 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ Real.pi / 4 → f x1 < f x2) ∧
  (∀ x : ℝ, f (3 * Real.pi / 4 - x) = f (3 * Real.pi / 4 + x)) :=
by sorry

end propositions_correct_l812_812650


namespace find_k_values_l812_812168

open Set

def A : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def B (k : ℝ) : Set ℝ := {x | x^2 - (k + 1) * x + k = 0}

theorem find_k_values (k : ℝ) : (A ∩ B k = B k) ↔ k ∈ ({1, -3} : Set ℝ) := by
  sorry

end find_k_values_l812_812168


namespace largest_multiple_of_9_less_than_100_l812_812274

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ ∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n :=
begin
  use 99,
  split,
  { norm_num }, -- 99 < 100
  split,
  { norm_num }, -- 99 % 9 = 0
  { intros m hm,
    obtain ⟨k, rfl⟩ := nat.exists_eq_mul_right_of_dvd (dvd_of_mod_eq_zero hm.2),
    rw [mul_comm, nat.mul_lt_mul_iff_left (by norm_num : 0 < 9)],
    norm_num,
    exact nat.succ_le_iff.mpr (le_of_lt hm.1), }
end

end largest_multiple_of_9_less_than_100_l812_812274


namespace extreme_values_sum_alpha_beta_l812_812059

noncomputable def f (x : Real) : Real := 2 * sin x * cos x + 2 * sqrt 3 * cos x ^ 2 - sqrt 3

variables (α β : Real)

-- Defining the initial conditions
axiom alpha_in_range : α ∈ Set.Icc (π / 4) π
axiom beta_in_range : β ∈ Set.Icc π (3 * π / 2)
axiom f_value : f (α - π / 6) = 2 * sqrt 5 / 5
axiom sin_value : sin (β - α) = sqrt 10 / 10

-- Statements to be proven
theorem extreme_values :
  ∀ x ∈ Set.Icc 0 (2 * π / 3),
  f x ≥ -2 ∧ f x ≤ 2 := sorry

theorem sum_alpha_beta :
  α + β = 7 * π / 4 := sorry

end extreme_values_sum_alpha_beta_l812_812059


namespace find_some_number_l812_812610

theorem find_some_number :
  let calc := 0.47 * 1442 - 0.36 * 1412
  ∃ some_number : ℝ, calc + some_number = 3 ∧ some_number = -166.42 := 
by
  let calc := 0.47 * 1442 - 0.36 * 1412
  use 3 - calc
  split
  · exact eq_sub_of_add_eq rfl
  · sorry

end find_some_number_l812_812610


namespace isosceles_triangle_altitude_angle_l812_812891

theorem isosceles_triangle_altitude_angle 
  (A B C D : Type) 
  (AC BC : ℕ) 
  (h_isosceles : AC = BC) 
  (h_angle_C : ∠C = 50) 
  (h_altitude : AD ⟂ BC) 
  : ∠ADB = 25 :=
sorry

end isosceles_triangle_altitude_angle_l812_812891


namespace vector_dot_product_zero_if_equal_magnitudes_l812_812191

variables (a b : EuclideanSpace ℝ (Fin 2))

theorem vector_dot_product_zero_if_equal_magnitudes (h : ‖a + b‖ = ‖a - b‖) : dot_product a b = 0 := 
sorry

end vector_dot_product_zero_if_equal_magnitudes_l812_812191


namespace trapezoid_DC_length_l812_812941

theorem trapezoid_DC_length 
  (A B C D F E : Type)
  (AB DC : ℝ) 
  (BC DE : ℝ) 
  (angle_BCD angle_CDA : ℝ) 
  (H1 : AB = 5)
  (H2 : BC = 3 * real.sqrt 2)
  (H3 : angle_BCD = real.pi / 4) -- 45 degrees in radians
  (H4 : angle_CDA = real.pi / 3) -- 60 degrees in radians
  (H5 : AB ∥ DC) :
  DC = 8 + real.sqrt 3 :=
by
  sorry

end trapezoid_DC_length_l812_812941


namespace arccos_one_eq_zero_l812_812506

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l812_812506


namespace number_of_cans_on_third_day_l812_812320

-- Definition of an arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

theorem number_of_cans_on_third_day :
  (arithmetic_sequence 4 5 2 = 9) →   -- on the second day, he found 9 cans
  (arithmetic_sequence 4 5 7 = 34) →  -- on the seventh day, he found 34 cans
  (arithmetic_sequence 4 5 3 = 14) :=  -- therefore, on the third day, he found 14 cans
by
  intros h1 h2
  sorry

end number_of_cans_on_third_day_l812_812320


namespace tangent_circle_construction_l812_812860

-- Given conditions
variables {O : Type*} [metric_space O] [normed_group O] [normed_space ℝ O]
variables (R1 R2 : ℝ) (L : set O) (hR : R1 < R2)

-- Definitions of the key parameters
def r : ℝ := (R1 + R2) / 2
def radius : ℝ := (R2 - R1) / 2

-- Circle T1 and T2 are concentric circles around the origin O with radii R1 and R2 respectively
noncomputable def concentric_circles (O : O) (R : ℝ) : set O := {P | ∥P - O∥ = R}

-- Given a line L and the distance d
variables (d : ℝ)

-- Geometric locus for tangency to the given line
def parallel_to_L (dist : ℝ) (L : set O) : set O := {P | ∃ Q ∈ L, ∥P - Q∥ = dist}

-- Proof statement
theorem tangent_circle_construction {O : O} (T1 T2 : set O)
  (hT1 : T1 = concentric_circles O R1) (hT2 : T2 = concentric_circles O R2) :
  ∃ (C : O) (rC : ℝ), 
    (C ∈ parallel_to_L r L) ∧
    (∥C - O∥ = r) ∧
    (∀ (P ∈ T1), (∥P - C∥ = radius)) ∧
    (∀ (P ∈ T2), (∥P - C∥ = radius)) ∧
    (∀ (Q ∈ L), (∥C - Q∥ = radius)) := sorry

end tangent_circle_construction_l812_812860


namespace weekend_weekday_reading_ratio_l812_812076

theorem weekend_weekday_reading_ratio
    (newspaper_minutes_per_day : ℕ := 30)
    (novel_minutes_per_day : ℕ := 60)
    (days_per_week : ℕ := 5)
    (total_minutes_per_week : ℕ := 810) :
  let weekday_reading := (newspaper_minutes_per_day + novel_minutes_per_day) * days_per_week in
  let weekend_reading := total_minutes_per_week - weekday_reading in
  weekend_reading / weekday_reading = 4 / 5 :=
by
  sorry

end weekend_weekday_reading_ratio_l812_812076


namespace largest_multiple_of_9_less_than_100_l812_812263

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l812_812263


namespace convince_jury_l812_812129

-- Define predicates for being a criminal, normal man, guilty, or a knight
def Criminal : Prop := sorry
def NormalMan : Prop := sorry
def Guilty : Prop := sorry
def Knight : Prop := sorry

-- Define your status
variable (you : Prop)

-- Assumptions as per given conditions
axiom criminal_not_normal_man : Criminal → ¬NormalMan
axiom you_not_guilty : ¬Guilty
axiom you_not_knight : ¬Knight

-- The statement to prove
theorem convince_jury : ¬Guilty ∧ ¬Knight := by
  exact And.intro you_not_guilty you_not_knight

end convince_jury_l812_812129


namespace tournament_total_players_l812_812710

theorem tournament_total_players {n m : ℕ} (h1 : m = n + 8)
  (h2 : ∀ {i : ℕ}, i < n + 8 → (2 / 3) * points i = points_against_lowest 8 i) :
  m = 20 :=
by
  sorry

end tournament_total_players_l812_812710


namespace dot_product_of_a_b_l812_812637

theorem dot_product_of_a_b 
  (a b : ℝ)
  (θ : ℝ)
  (ha : a = 2 * Real.sin (15 * Real.pi / 180))
  (hb : b = 4 * Real.cos (15 * Real.pi / 180))
  (hθ : θ = 30 * Real.pi / 180) :
  (a * b * Real.cos θ) = Real.sqrt 3 := by
  sorry

end dot_product_of_a_b_l812_812637


namespace arccos_one_eq_zero_l812_812461

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812461


namespace smallest_number_l812_812247

def numbers : set ℝ := {0.8, 1 / 2, 0.5}

theorem smallest_number (h1: ∀ n ∈ numbers, n > 0.1) : ∃ (m : ℝ), m ∈ numbers ∧ m > 0.1 ∧ ∀ n ∈ numbers, n > 0.1 → m ≤ n := 
by 
  use 0.5
  split
  { -- Prove 0.5 belongs to the set of numbers
    exact set.mem_insert_of_mem 0.8 (set.mem_insert_of_mem (1 / 2) (set.mem_singleton 0.5)) 
  }
  split
  { -- Prove 0.5 is greater than 0.1
    exact by norm_num 
  }
  { -- Prove that 0.5 is the smallest number greater than 0.1 in the set
    intros n hn hn1
    norm_num
    cases hn
    { -- Case: n = 0.8
      linarith 
    }
    cases hn
    { -- Case: n = 1 / 2
      norm_num
    }
    { -- Case: n = 0.5
      linarith 
    }
  }

end smallest_number_l812_812247


namespace arccos_one_eq_zero_l812_812398

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812398


namespace simplify_complex_fraction_l812_812197

noncomputable def complex_example : Prop :=
  let i := Complex.I in
  (3 - 2 * i) / (1 + 4 * i) = -5 / 17 - (14 / 17) * i

theorem simplify_complex_fraction : complex_example := 
by
  sorry

end simplify_complex_fraction_l812_812197


namespace KL_parallel_BB1_l812_812221

noncomputable theory
open_locale classical

variables {A B C I L K: Point}
variables {C1 A1 B1: Point}
variables {incircle: Circle}

-- Given conditions
axiom incircle_touches_sides :
  incircle.center = I ∧
  incircle.tangent_point A B = C1 ∧
  incircle.tangent_point B C = A1 ∧
  incircle.tangent_point C A = B1

axiom L_is_bisector_foot : L ∈ Line.from_points B I

axiom K_intersection :
  ∃ K, K = Line.from_points B1 I ∩ Line.from_points A1 C1

-- Statement to prove
theorem KL_parallel_BB1 (h1 : incircle_touches_sides) (h2 : L_is_bisector_foot) (h3 : K_intersection):
  parallel (Line.from_points K L) (Line.from_points B B1) := 
sorry

end KL_parallel_BB1_l812_812221


namespace max_saturdays_l812_812128

theorem max_saturdays (days_in_month : ℕ) (month : string) (is_leap_year : Prop) (start_day : ℕ) : 
  (days_in_month = 29 → is_leap_year → start_day = 6 → true) ∧ -- February in a leap year starts on Saturday
  (days_in_month = 30 → (start_day = 5 ∨ start_day = 6) → true) ∧ -- 30-day months start on Friday or Saturday
  (days_in_month = 31 → (start_day = 4 ∨ start_day = 5 ∨ start_day = 6) → true) ∧ -- 31-day months start on Thursday, Friday, or Saturday
  (31 ≤ days_in_month ∧ days_in_month ≤ 28 → false) → -- Other case should be false
  ∃ n : ℕ, n = 5 := -- Maximum number of Saturdays is 5
sorry

end max_saturdays_l812_812128


namespace solve_for_x_l812_812029

theorem solve_for_x (x y : ℕ) (h1 : x / y = 10 / 4) (h2 : y = 18) : x = 45 :=
sorry

end solve_for_x_l812_812029


namespace arccos_one_eq_zero_l812_812482

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812482


namespace arccos_one_eq_zero_l812_812423

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812423


namespace sum_80_consecutive_l812_812329

-- Define the sequence
def sequence (n : ℕ) (start : ℤ) : ℕ → ℤ
| 0     => start
| (m+1) => sequence start m + 1

-- Define the sum of an arithmetic sequence
def arith_sum (n : ℕ) (start : ℤ) : ℤ :=
(n * (start + (start + n - 1))) / 2

-- Given conditions
def start : ℤ := -39
def n : ℕ := 80

-- Assertion to prove
theorem sum_80_consecutive :
  arith_sum n start = 40 := by
  sorry

end sum_80_consecutive_l812_812329


namespace arccos_one_eq_zero_l812_812469

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812469


namespace num_values_f100_eq_0_l812_812143

def f0 (x : ℝ) : ℝ := x + |x - 100| - |x + 100|

def fn : ℕ → ℝ → ℝ
| 0, x   => f0 x
| (n+1), x => |fn n x| - 1

theorem num_values_f100_eq_0 : ∃ (xs : Finset ℝ), ∀ x ∈ xs, fn 100 x = 0 ∧ xs.card = 301 :=
by
  sorry

end num_values_f100_eq_0_l812_812143


namespace arccos_one_eq_zero_l812_812485

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812485


namespace erased_number_is_100_l812_812244

theorem erased_number_is_100! :
  ∃ k : ℕ, k = 100 ∧
    ∀ (f : ℕ → ℕ), f = Nat.factorial →
    let N := ∏ i in Finset.range 200, f (i + 1) in
    let remaining_product := N / f k in
    Nat.is_square (remaining_product) :=
begin
  sorry
end

end erased_number_is_100_l812_812244


namespace find_p_q_sum_l812_812156

def point := (ℕ × ℕ × ℕ)
def T : set point := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 4 ∧ 0 ≤ p.3 ∧ p.3 ≤ 5 }

def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def is_in_T (p : point) : Prop :=
  p ∈ T

theorem find_p_q_sum : ∃ (p q : ℕ), gcd p q = 1 ∧ (p + q = 65) ∧
  ∃ (a b : point), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ midpoint a b ∈ T ∧ 
  (∃ (numerator denominator : ℕ), numerator / denominator = p / q) :=
sorry

end find_p_q_sum_l812_812156


namespace original_denominator_value_l812_812923

theorem original_denominator_value (d : ℕ) (h1 : 3 + 3 = 6) (h2 : ((6 : ℕ) / (d + 3 : ℕ) = (1 / 3 : ℚ))) : d = 15 :=
sorry

end original_denominator_value_l812_812923


namespace arccos_one_eq_zero_l812_812447

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812447


namespace arccos_one_eq_zero_l812_812513

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812513


namespace mass_percentage_of_S_in_Al2S3_l812_812601

theorem mass_percentage_of_S_in_Al2S3 :
  let molar_mass_Al : ℝ := 26.98
  let molar_mass_S : ℝ := 32.06
  let formula_of_Al2S3: (ℕ × ℕ) := (2, 3)
  let molar_mass_Al2S3 : ℝ := (2 * molar_mass_Al) + (3 * molar_mass_S)
  let total_mass_S_in_Al2S3 : ℝ := 3 * molar_mass_S
  (total_mass_S_in_Al2S3 / molar_mass_Al2S3) * 100 = 64.07 :=
by
  sorry

end mass_percentage_of_S_in_Al2S3_l812_812601


namespace largest_multiple_of_9_less_than_100_l812_812265

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l812_812265


namespace f_m1_expr_f_mn_expr_f_nn_not_2013_l812_812907

noncomputable def f : ℕ → ℕ → ℕ
| 1, 1 => 1
| m, n+1 => f m n + 3
| m+1, 1 => 3 * f m 1

theorem f_m1_expr (m : ℕ) : f m 1 = 3^(m-1) :=
by sorry

theorem f_mn_expr (m n : ℕ) : f m n = 3^(m-1) + 3*(n-1) :=
by sorry

theorem f_nn_not_2013 (n : ℕ) : f n n ≠ 2013 :=
by sorry

end f_m1_expr_f_mn_expr_f_nn_not_2013_l812_812907


namespace arccos_one_eq_zero_l812_812516

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812516


namespace range_of_k_l812_812698

theorem range_of_k (k : ℝ) (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < 2)
  (h_roots : ∀ x, x^2 + (k-2)*x + (2*k-1) = 0 → (x = x₁ ∨ x = x₂)) : 1/2 < k ∧ k < 2/3 :=
begin
  -- sorry is used to skip the actual proof steps
  sorry,
end

end range_of_k_l812_812698


namespace jesse_gave_pencils_l812_812135

theorem jesse_gave_pencils (initial_pencils : ℕ) (final_pencils : ℕ) (pencils_given : ℕ) :
  initial_pencils = 78 → final_pencils = 34 → pencils_given = initial_pencils - final_pencils → pencils_given = 44 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jesse_gave_pencils_l812_812135


namespace arccos_1_eq_0_l812_812418

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812418


namespace D_is_painting_l812_812098

def A_activity (act : String) : Prop := 
  act ≠ "walking" ∧ act ≠ "playing basketball"

def B_activity (act : String) : Prop :=
  act ≠ "dancing" ∧ act ≠ "running"

def C_activity_implies_A_activity (C_act A_act : String) : Prop :=
  C_act = "walking" → A_act = "dancing"

def D_activity (act : String) : Prop :=
  act ≠ "playing basketball" ∧ act ≠ "running"

def C_activity (act : String) : Prop :=
  act ≠ "dancing" ∧ act ≠ "playing basketball"

theorem D_is_painting :
  (∃ a b c d : String,
    A_activity a ∧
    B_activity b ∧
    C_activity_implies_A_activity c a ∧
    D_activity d ∧
    C_activity c) →
  ∃ d : String, d = "painting" :=
by
  intros h
  sorry

end D_is_painting_l812_812098


namespace shelves_of_picture_books_l812_812674

-- Define the conditions
def n_mystery : ℕ := 5
def b_per_shelf : ℕ := 4
def b_total : ℕ := 32

-- State the main theorem to be proven
theorem shelves_of_picture_books :
  (b_total - n_mystery * b_per_shelf) / b_per_shelf = 3 :=
by
  -- The proof is omitted
  sorry

end shelves_of_picture_books_l812_812674


namespace area_of_triangle_DCM_l812_812038

-- Define the isosceles trapezoid ABCD with the specific properties
structure IsoscelesTrapezoid (A B C D : Type) :=
(lateral_side_AB_perpendicular_to_bases : ∀ (A B C D M : Point), 
  is_perpendicular (line A B) (line A D) ∧ is_perpendicular (line A B) (line B C))
(inscribed_circle_radius : ∀ (A B C D : Point), ℝ)
  
-- Define the point M as the intersection of diagonals AC and BD
def Intersection_point {A B C D M : Point} (h : is_intersection_point (line A C) (line B D))

-- The main theorem to prove that the area of triangle DCM equals r^2
theorem area_of_triangle_DCM {A B C D M : Point} (ABCD : IsoscelesTrapezoid A B C D)
  (h1 : ∀ (A B C D M : Point), M = intersection (line A C) (line B D))
  (h2 : ∀ (A B C D : Point), inscribed_circle_radius A B C D = r)
  : area (triangle D C M) = r^2 :=
sorry

end area_of_triangle_DCM_l812_812038


namespace solve_fraction_zero_l812_812242

theorem solve_fraction_zero (x : ℕ) (h : x ≠ 0) (h_eq : (x - 1) / x = 0) : x = 1 := by 
  sorry

end solve_fraction_zero_l812_812242


namespace find_ordered_triple_l812_812772

theorem find_ordered_triple (a b c : ℝ) (h₁ : 2 < a) (h₂ : 2 < b) (h₃ : 2 < c)
    (h_eq : (a + 1)^2 / (b + c - 1) + (b + 2)^2 / (c + a - 3) + (c + 3)^2 / (a + b - 5) = 32) :
    (a = 8 ∧ b = 6 ∧ c = 5) :=
sorry

end find_ordered_triple_l812_812772


namespace arccos_one_eq_zero_l812_812424

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812424


namespace team_E_has_not_played_against_team_B_l812_812822

-- We begin by defining the teams as an enumeration
inductive Team
| A | B | C | D | E | F

open Team

-- Define the total number of matches each team has played
def matches_played (t : Team) : Nat :=
  match t with
  | A => 5
  | B => 4
  | C => 3
  | D => 2
  | E => 1
  | F => 0 -- Note: we assume F's matches are not provided; this can be adjusted if needed

-- Prove that team E has not played against team B
theorem team_E_has_not_played_against_team_B :
  ∃ t : Team, matches_played B = 4 ∧ matches_played E < matches_played B ∧
  (t = E) :=
by
  sorry

end team_E_has_not_played_against_team_B_l812_812822


namespace mary_black_balloons_l812_812783

theorem mary_black_balloons (nancyBalloons maryBalloons : ℕ) 
  (H_nancy : nancyBalloons = 7) 
  (H_mary : maryBalloons = 28) : maryBalloons / nancyBalloons = 4 := 
by
  -- Provided conditions
  rw [H_nancy, H_mary]
  -- perform division
  exact (28 / 7).nat_cast().of_nat_proved_eq
  exact (28 / 7 = 4)

-- Proof is omitted
-- sorry

end mary_black_balloons_l812_812783


namespace inradius_of_triangle_with_sides_3_4_5_l812_812628

theorem inradius_of_triangle_with_sides_3_4_5 : 
  ∀ (a b c : ℕ),
    a = 3 → b = 4 → c = 5 → (∀ r : ℝ, let s := (a + b + c) / 2 in let area := 6 in r * s = area → r = 1) := 
by
  intros a b c ha hb hc r s_area_eq
  sorry

end inradius_of_triangle_with_sides_3_4_5_l812_812628


namespace total_area_rectABCD_l812_812919

theorem total_area_rectABCD (BF CF : ℝ) (X Y : ℝ)
  (h1 : BF = 3 * CF)
  (h2 : 3 * X - Y - (X - Y) = 96)
  (h3 : X + 3 * X = 192) :
  X + 3 * X = 192 :=
by
  sorry

end total_area_rectABCD_l812_812919


namespace probability_closer_to_C_l812_812738

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  0.5 * abs (p1.1*(p2.2 - p3.2) + p2.1*(p3.2 - p1.2) + p3.1*(p1.2 - p2.2))

theorem probability_closer_to_C
  (A B C P : ℝ × ℝ)
  (hA : A = (0, 6))
  (hB : B = (8, 0))
  (hC : C = (0, 0))
  (h_right_angle : distance A C ^ 2 + distance C B ^ 2 = distance A B ^ 2)
  (h_triangle_area : triangle_area A B C = 24) :
  (set_of (λ P, distance P C < distance P A ∧ distance P C < distance P B)).measure / (set_of (λ P, P ∈ {P | ∃ (x y : ℝ), 0 ≤ x ∧ 0 ≤ y})).measure = 0.5 := 
sorry

end probability_closer_to_C_l812_812738


namespace part1_part2_l812_812957

def sequence_property (x : ℕ → ℝ) : Prop :=
  x 0 = 1 ∧ (∀ i, x (i + 1) ≤ x i) ∧ (∀ i, x i > 0)

theorem part1 (x : ℕ → ℝ) (h : sequence_property x) :
  ∃ n ≥ 1, ∑ i in finset.range n, (x i)^2 / (x (i + 1)) ≥ 3.999 :=
sorry

theorem part2 :
  let x := λ n, (1 / 2)^n in
  (∀ n ≥ 1, ∑ i in finset.range n, (x i)^2 / (x (i + 1)) < 4) :=
sorry

end part1_part2_l812_812957


namespace katrina_fraction_money_left_l812_812748

theorem katrina_fraction_money_left : 
  ∀ (m : ℕ) (b : ℕ) (n : ℕ), 
  m = 200 → 
  (1 / 4 : ℚ) * m = (1 / 2 : ℚ) * n * b →
  (m - n * b) / m = 1 / 2 :=
by
  intros m b n hm hc
  rw [hm, mul_assoc (1 / 4 : ℚ) m (2 : ℚ), ← hc, add_sub_cancel, div_self]
  sorry

end katrina_fraction_money_left_l812_812748


namespace arccos_one_eq_zero_l812_812564

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812564


namespace smallest_n_7_pow_n_eq_n_7_mod_5_l812_812974

theorem smallest_n_7_pow_n_eq_n_7_mod_5 :
  ∃ n : ℕ, 0 < n ∧ (7^n ≡ n^7 [MOD 5]) ∧ ∀ m : ℕ, 0 < m ∧ (7^m ≡ m^7 [MOD 5]) → n ≤ m :=
begin
  sorry -- proof omitted as requested
end

end smallest_n_7_pow_n_eq_n_7_mod_5_l812_812974


namespace parking_cost_l812_812213

theorem parking_cost (C : ℝ) (h2 : C + 1.75 * 7 = 2.361111111111111 * 9) : 
  C = 9 := 
begin
  sorry
end

end parking_cost_l812_812213


namespace original_denominator_is_15_l812_812925

theorem original_denominator_is_15
  (d : ℕ)
  (h1 : (6 : ℚ) / (d + 3) = 1 / 3)
  (h2 : 3 = 3) : d = 15 := -- h2 is trivial but included according to the problem condition
by
  sorry

end original_denominator_is_15_l812_812925


namespace count_multiples_2_or_3_but_not_4_l812_812681

theorem count_multiples_2_or_3_but_not_4 : 
  (∃ n : ℕ, n = 42 ∧ ∀ (k : ℕ), 1 ≤ k ∧ k ≤ 100 → (k % 2 = 0 ∨ k % 3 = 0) ∧ k % 4 ≠ 0 → set_of k).card = 42 :=
sorry

end count_multiples_2_or_3_but_not_4_l812_812681


namespace vector_angle_obtuse_l812_812669

theorem vector_angle_obtuse (a b : ℝ × ℝ × ℝ) (k : ℝ) (m : ℝ)
    (h_a : a = (1, 1, 0))
    (h_b : b = (m, 0, 2))
    (h_cos : (1: ℝ) / (Real.sqrt 2 * Real.sqrt (m^2 + 4)) = -Real.sqrt 10 / 10) :
  ((1 - k, 1, 2 * k) • (1, 2, 2) < 0) ↔ k < -1 := 
sorry

end vector_angle_obtuse_l812_812669


namespace inscribed_sphere_to_cube_volume_ratio_l812_812369

theorem inscribed_sphere_to_cube_volume_ratio :
  let s := 8
  let r := s / 2
  let V_sphere := (4/3) * Real.pi * r^3
  let V_cube := s^3
  (V_sphere / V_cube) = Real.pi / 6 :=
by
  let s := 8
  let r := s / 2
  let V_sphere := (4/3) * Real.pi * r^3
  let V_cube := s^3
  sorry

end inscribed_sphere_to_cube_volume_ratio_l812_812369


namespace smallest_n_modulo_5_l812_812969

theorem smallest_n_modulo_5 :
  ∃ n : ℕ, n > 0 ∧ 2^n % 5 = n^7 % 5 ∧
    ∀ m : ℕ, (m > 0 ∧ 2^m % 5 = m^7 % 5) → n ≤ m := 
begin
  use 7,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m h,
    cases h with hpos heq,
    sorry, -- Here would be proof that 7 is the smallest n satisfying 2^n ≡ n^7 mod 5.
  }
end

end smallest_n_modulo_5_l812_812969


namespace locus_C2_angle_measure_90_l812_812623

variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a)

-- Conditions for Question 1
def ellipse_C1 (x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

variable (x0 y0 x1 y1 : ℝ)
variable (hA : ellipse_C1 a b x0 y0)
variable (hE : ellipse_C1 a b x1 y1)
variable (h_perpendicular : x1 * x0 + y1 * y0 = 0)

theorem locus_C2 :
  ∀ (x y : ℝ), ellipse_C1 a b x y → 
  x ≠ 0 → y ≠ 0 → 
  (x^2 / a^2 + y^2 / b^2 = (a^2 - b^2)^2 / (a^2 + b^2)^2) := 
sorry

-- Conditions for Question 2
def circle_C3 (x y : ℝ) : Prop := 
  x^2 + y^2 = 1

theorem angle_measure_90 :
  (a^2 + b^2)^3 = a^2 * b^2 * (a^2 - b^2)^2 → 
  ∀ (x y : ℝ), ellipse_C1 a b x y → 
  circle_C3 x y → 
  (∃ (theta : ℝ), θ = 90) := 
sorry

end locus_C2_angle_measure_90_l812_812623


namespace cosine_identity_l812_812339

theorem cosine_identity (x : Real) : x = 2017 → cos x = -cos 37 :=
by
  intro hx
  have : cos 2017 = cos (5 * 360 + 217) := by sorry
  have : cos (5 * 360 + 217) = cos (180 + 37) := by sorry
  have : cos (180 + 37) = -cos 37 := by sorry
  rw hx
  rw this
  sorry

end cosine_identity_l812_812339


namespace painted_surface_area_ratio_thm_l812_812368

noncomputable def painted_surface_area_ratio : ℕ :=
  let h : ℝ := 8
  let r : ℝ := 5
  let h_C : ℝ := 3
  let r_C : ℝ := (h_C / h) * r
  let l_C := Real.sqrt(h_C^2 + r_C^2)
  let A_C := π * r_C * l_C
  let h_F : ℝ := h - h_C
  let r_F : ℝ := r - r_C
  let l_F := Real.sqrt(h_F^2 + r_F^2)
  let A_F := π * r_F * l_F
  let k := (A_C / A_F)
  -- simplify k to get m, n where k = m/n and gcd(m,n) = 1
  let m : ℕ := 2
  let n : ℕ := 11
  m + n

theorem painted_surface_area_ratio_thm : painted_surface_area_ratio = 13 := 
by
  sorry

end painted_surface_area_ratio_thm_l812_812368


namespace minimize_expression_l812_812109

noncomputable def P_minimizes_expression (A B C P : Point) (L M N : Point)
  (h1 : Triangle A B C) (h2 : AcuteTriangle h1)
  (hL : PerpendicularFromPointToLine P B C L)
  (hM : PerpendicularFromPointToLine P C A M)
  (hN : PerpendicularFromPointToLine P A B N)
  (hCircumcenter : IsCircumcenter P A B C) : Prop :=
  (BL : ℝ) (CM : ℝ) (AN : ℝ)
  (hPerpendicularBL : BL = Distance B L)
  (hPerpendicularCM : CM = Distance C M)
  (hPerpendicularAN : AN = Distance A N) →
  (BL^2 + CM^2 + AN^2 = minimum_value)

theorem minimize_expression (A B C : Point) (h : Triangle A B C) (h_acute : AcuteTriangle h)
  : ∃ P, IsCircumcenter P A B C ∧ ∀ (L M N : Point) (hL : PerpendicularFromPointToLine P B C L)
  (hM : PerpendicularFromPointToLine P C A M) (hN : PerpendicularFromPointToLine P A B N)
  (BL : ℝ) (CM : ℝ) (AN : ℝ)
  (hPerpendicularBL : BL = Distance B L)
  (hPerpendicularCM : CM = Distance C M)
  (hPerpendicularAN : AN = Distance A N),
  BL^2 + CM^2 + AN^2 = minimum_value := 
begin
  assume P : Point,
  assume hCircumcenter : IsCircumcenter P A B C,
  use P,
  split,
  { exact hCircumcenter, },
  { intros L M N hL hM hN BL CM AN hPerpendicularBL hPerpendicularCM hPerpendicularAN,
    sorry,  -- The proof is beyond the scope of this exercise
  }
end

end minimize_expression_l812_812109


namespace arccos_one_eq_zero_l812_812451

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812451


namespace arithmetic_sequence_values_l812_812647

noncomputable def common_difference (a₁ a₂ : ℕ) : ℕ := (a₂ - a₁) / 2

theorem arithmetic_sequence_values (x y z d: ℕ) 
    (h₁: d = common_difference 7 11) 
    (h₂: x = 7 + d) 
    (h₃: y = 11 + d) 
    (h₄: z = y + d): 
    x = 9 ∧ y = 13 ∧ z = 15 :=
by {
  sorry
}

end arithmetic_sequence_values_l812_812647


namespace range_of_g_l812_812966

-- Define trigonometric functions and ranges
noncomputable def g (A : ℝ) : ℝ :=
  (cos A) * (3 * (sin A)^2 + (sin A)^4 + 3 * (cos A)^2 + (sin A)^2 * (cos A)^2) / 
  (cot A * (csc A - (cos A) * (cot A)))

-- State the condition A ≠ n * π for any integer n
def valid_A (A : ℝ) : Prop :=
  ∀ (n : ℤ), A ≠ n * Real.pi

-- The statement to prove the range of g(A)
theorem range_of_g (A : ℝ) (hA : valid_A A) :
  3 < g A ∧ g A < 4 :=
sorry

end range_of_g_l812_812966


namespace distance_AB_l812_812121

def pointA : ℝ × ℝ := (3, Real.pi / 4)
def pointB : ℝ × ℝ := (Real.sqrt 2, Real.pi / 2)

theorem distance_AB : dist pointA pointB = Real.sqrt 5 := 
sorry

end distance_AB_l812_812121


namespace min_sum_first_n_terms_l812_812211

variable {a₁ d c : ℝ} (n : ℕ)

noncomputable def sum_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem min_sum_first_n_terms (h₁ : ∀ x, 1/3 ≤ x ∧ x ≤ 4/5 → a₁ * x^2 + (d/2 - a₁) * x + c ≥ 0)
                              (h₂ : a₁ = -15/4 * d)
                              (h₃ : d > 0) :
                              ∃ n : ℕ, n > 0 ∧ sum_first_n_terms a₁ d n ≤ sum_first_n_terms a₁ d 4 :=
by
  use 4
  sorry

end min_sum_first_n_terms_l812_812211


namespace arccos_one_eq_zero_l812_812441

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812441


namespace smallest_n_modulo_5_l812_812970

theorem smallest_n_modulo_5 :
  ∃ n : ℕ, n > 0 ∧ 2^n % 5 = n^7 % 5 ∧
    ∀ m : ℕ, (m > 0 ∧ 2^m % 5 = m^7 % 5) → n ≤ m := 
begin
  use 7,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m h,
    cases h with hpos heq,
    sorry, -- Here would be proof that 7 is the smallest n satisfying 2^n ≡ n^7 mod 5.
  }
end

end smallest_n_modulo_5_l812_812970


namespace full_house_probability_correct_l812_812568

def modified_deck_ranks := [10, 11, 12, 13, 14]
def cards_per_rank := 4
def total_cards := 20
def hand_size := 5

def total_combinations (n k : ℕ) : ℕ := Nat.choose n k
def successful_outcomes : ℕ := 5 * Nat.choose 4 3 * 4 * Nat.choose 4 2
def total_outcomes : ℕ := total_combinations total_cards hand_size

noncomputable def probability_full_house : ℚ := successful_outcomes / total_outcomes

theorem full_house_probability_correct :
  probability_full_house = 40 / 1292 := by
  calc
    probability_full_house = successful_outcomes / total_outcomes := rfl
    ... = 40 / 1292 := rfl
sorry

end full_house_probability_correct_l812_812568


namespace symmetric_circle_l812_812839

theorem symmetric_circle
    (x y : ℝ)
    (circle_eq : x^2 + y^2 + 4 * x - 1 = 0) :
    (x - 2)^2 + y^2 = 5 :=
sorry

end symmetric_circle_l812_812839


namespace original_ratio_of_flour_to_baking_soda_l812_812124

-- Define the conditions
def sugar_to_flour_ratio_5_to_5 (sugar flour : ℕ) : Prop :=
  sugar = 2400 ∧ sugar = flour

def baking_soda_mass_condition (flour : ℕ) (baking_soda : ℕ) : Prop :=
  flour = 2400 ∧ (∃ b : ℕ, baking_soda = b ∧ flour / (b + 60) = 8)

-- The theorem statement we need to prove
theorem original_ratio_of_flour_to_baking_soda :
  ∃ flour baking_soda : ℕ,
  sugar_to_flour_ratio_5_to_5 2400 flour ∧
  baking_soda_mass_condition flour baking_soda →
  flour / baking_soda = 10 :=
by
  sorry

end original_ratio_of_flour_to_baking_soda_l812_812124


namespace smallest_integer_in_set_A_l812_812660

def set_A : Set ℝ := {x | |x - 2| ≤ 5}

theorem smallest_integer_in_set_A : ∃ m ∈ set_A, ∀ n ∈ set_A, m ≤ n := 
  sorry

end smallest_integer_in_set_A_l812_812660


namespace geom_proof_l812_812024

noncomputable def geom_problem (P A B C D E F O : Point) (circle_O : Circle) : Prop :=
  is_tangent P A circle_O ∧ 
  is_secant P C D circle_O ∧ 
  is_diameter A B circle_O ∧
  parallel (line_through C E) (line_through P O) ∧ 
  intersects (line_through C E) (line_through A B) E ∧ 
  intersects (line_through C E) (line_through B D) F 

theorem geom_proof (P A B C D E F O : Point) (circle_O : Circle) :
  geom_problem P A B C D E F O circle_O →
  dist C E = dist E F := by
sorry

end geom_proof_l812_812024


namespace piravena_total_distance_l812_812790

-- Define the cities and their distances in a right-angled triangle
def A_to_C := 3500 -- Distance from A to C in km
def A_to_B := 4000 -- Distance from A to B in km

-- Calculate the distance from B to C using Pythagorean theorem
def B_to_C : ℕ := Int.sqrt (A_to_B^2 - A_to_C^2).to_nat

-- Define the total distance
def total_distance := A_to_B + B_to_C + A_to_C

-- The theorem stating the total distance Piravena travels 
theorem piravena_total_distance : total_distance = 9437 := by
  have h1 : A_to_C = 3500 := rfl
  have h2 : A_to_B = 4000 := rfl
  have h3 : B_to_C = 1937 := 
    calc
      Int.sqrt (A_to_B^2 - A_to_C^2).to_nat
      = Int.sqrt ((4000:ℕ)^2 - (3500:ℕ)^2) : by simp [A_to_B, A_to_C]
      = 1937 : by 
        have h : (4000^2 - 3500^2) = 3750000 := by norm_num
        simp [h]
  show (A_to_B + B_to_C + A_to_C) = 9437
  calc
    A_to_B + B_to_C + A_to_C
    = 4000 + 1937 + 3500 : by simp [A_to_B, B_to_C, A_to_C]
    = 9437 : by norm_num

end piravena_total_distance_l812_812790


namespace arccos_one_eq_zero_l812_812445

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812445


namespace a_minus_b_perpendicular_to_b_l812_812667

open Real

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (1/2, 1/2)
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem a_minus_b_perpendicular_to_b :
  ∀ (x y : ℝ), ((a.1 - b.1, a.2 - b.2) = a_minus_b)
  → (a_minus_b.1 * b.1 + a_minus_b.2 * b.2 = 0) := by
  intros x y h_eq
  sorry

end a_minus_b_perpendicular_to_b_l812_812667


namespace find_n_to_sum_2010_l812_812959

-- Define the sequence blocks
def block_sum (k : ℕ) : ℕ := 1 + 2 * k
def block_length (k : ℕ) : ℕ := 1 + k

-- Define the sum of the first m blocks
def sum_blocks (m : ℕ) : ℕ := ∑ k in finset.range(m + 1), block_sum k

-- Define the sequence that constructs the sum
def sequence_sum (n : ℕ) : ℕ :=
  let m := n / (n + 1) in
  let remaining := n % (n + 1) in
  sum_blocks m + remaining * block_sum (m + 1)

theorem find_n_to_sum_2010 : ∃ n, sequence_sum n = 2010 ∧ n = 1027 :=
by {
  use 1027,
  sorry
}

end find_n_to_sum_2010_l812_812959


namespace positive_minimum_at_minus_one_eighth_l812_812579

/-- Given the quadratic function 
    y = f(x) (2^(3u - 5v - 3) - 1) * x^2 + (2^(7u - 8v - 18) - 2^(4u - 3v - 15)) * x + (11t^2 - 5u) / (154v - 10t^2) 
    has a positive minimum at x = -1/8, prove that t = 3, u = 4, v = 1 for positive integer parameters t, u, v. -/
theorem positive_minimum_at_minus_one_eighth (f : ℝ → ℝ) :
  ∃ (t u v : ℕ), 1 < t ∧ 1 < u ∧ 1 < v ∧
    t = 3 ∧ u = 4 ∧ v = 1 ∧ 
    let a := 2^(3 * u - 5 * v - 3) - 1 in
    let b := 2^(7 * u - 8 * v - 18) - 2^(4 * u - 3 * v - 15) in
    let c := (11 * t^2 - 5 * u) / (154 * v - 10 * t^2) in
    y = f x * a * x^2 + b * x + c ∧
    (b / (2 * a)) = -1 / 8 ∧ a > 0 ∧ f(-1 / 8) > 0 :=
sorry

end positive_minimum_at_minus_one_eighth_l812_812579


namespace f_monotonically_decreasing_intervals_exists_constant_k_l812_812060
-- Necessary import

open Real

noncomputable def f (x : ℝ) := 2 * x / (log x)

theorem f_monotonically_decreasing_intervals :
  (∀ x ∈ Ioo 0 1, deriv f x < 0) ∧ (∀ x ∈ Ioo 1 exp 1, deriv f x < 0) :=
by
  sorry

theorem exists_constant_k : ∃ k, (∀ x > 0, f x > k / log x + 2 * sqrt x) ∧ k = 2 :=
by
  use 2
  sorry

end f_monotonically_decreasing_intervals_exists_constant_k_l812_812060


namespace find_fx_neg_l812_812075

noncomputable def f (x : ℝ) : ℝ :=
  if h : x > 0 then x^2 - 2*x else x^2 + 2*x

theorem find_fx_neg (x : ℝ) (hx : x < 0) : f x = x^2 + 2*x := by
  have h_neg : -x > 0 := by linarith
  have f_neg_eq_f_pos : f (-x) = (-x)^2 - 2*(-x) := by
    unfold f
    simp [h_neg]
  rw [f_neg_eq_f_pos]
  simp
  sorry

end find_fx_neg_l812_812075


namespace rose_incorrect_answers_l812_812099

-- Define the total number of items in the exam.
def total_items : ℕ := 60

-- Define the percentage of correct answers Liza got.
def liza_percentage_correct : ℝ := 0.90

-- Calculate the number of correct answers Liza got.
def liza_correct_answers : ℕ := (liza_percentage_correct * total_items.to_real).to_nat

-- Define the number of more correct answers Rose got compared to Liza.
def rose_more_correct : ℕ := 2

-- Calculate the number of correct answers Rose got.
def rose_correct_answers : ℕ := liza_correct_answers + rose_more_correct

-- Define the problem as a theorem to prove Rose's incorrect answers.
theorem rose_incorrect_answers (total_items liza_correct_answers rose_correct_answers : ℕ) 
  (liza_percentage_correct : ℝ) (rose_more_correct: ℕ)
  (h1 : total_items = 60)
  (h2 : liza_percentage_correct = 0.90)
  (h3 : liza_correct_answers = (liza_percentage_correct * total_items.to_real).to_nat)
  (h4 : rose_more_correct = 2)
  (h5 : rose_correct_answers = liza_correct_answers + rose_more_correct) :
  rose_incorrect_answers = total_items - rose_correct_answers := by
  sorry

end rose_incorrect_answers_l812_812099


namespace arccos_one_eq_zero_l812_812459

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812459


namespace alternating_sum_squares_l812_812186

theorem alternating_sum_squares (n : ℕ) : 
  (∑ i in Finset.range (n + 1), (-1) ^ (i - 1) * i ^ 2) = (-1) ^ (n - 1) * (n * (n + 1) / 2) :=
sorry

end alternating_sum_squares_l812_812186


namespace initial_volume_of_solution_l812_812849

theorem initial_volume_of_solution (V : ℝ) :
  (∀ (init_vol : ℝ), 0.84 * init_vol / (init_vol + 26.9) = 0.58) →
  V = 60 :=
by
  intro h
  sorry

end initial_volume_of_solution_l812_812849


namespace insured_fraction_l812_812371

theorem insured_fraction (premium : ℝ) (rate : ℝ) (insured_value : ℝ) (original_value : ℝ)
  (h₁ : premium = 910)
  (h₂ : rate = 0.013)
  (h₃ : insured_value = premium / rate)
  (h₄ : original_value = 87500) :
  insured_value / original_value = 4 / 5 :=
by
  sorry

end insured_fraction_l812_812371


namespace unfolded_paper_holes_placement_l812_812318

noncomputable theory

-- Define the conditions
def rectangular_paper (P : Type) := True
def fold_left_to_right (P : Type) := True
def fold_diagonally (P : Type) := True
def punch_hole_near_center (P : Type) := True

-- State the theorem based on the conditions and conclusion
theorem unfolded_paper_holes_placement (P : Type) 
  [rectangular_paper P] 
  [fold_left_to_right P] 
  [fold_diagonally P] 
  [punch_hole_near_center P] : 
  ∃ holes : Set (P → Prop), 
    (∀ hole ∈ holes, Symmetric_placement hole).
Proof := sorry

end unfolded_paper_holes_placement_l812_812318


namespace sqrt_9_fact_over_126_eq_12_sqrt_10_l812_812012

theorem sqrt_9_fact_over_126_eq_12_sqrt_10 :
  real.sqrt (9! / 126) = 12 * real.sqrt 10 := 
sorry

end sqrt_9_fact_over_126_eq_12_sqrt_10_l812_812012


namespace arccos_one_eq_zero_l812_812531

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812531


namespace area_of_quad_EFHG_is_one_third_proof_area_of_quad_EFHG_is_one_third_l812_812384

-- Definition: Quadrilateral ABCD with area 1
def quadrilateral_ABCD_with_area_1 : Prop :=
  ∃ (A B C D : ℝ × ℝ), (some function to calculate area of quadrilateral ABCD) = 1

-- Definition: Points E and F are trisection points on AB
def points_trisection_AB (A B C D E F : ℝ × ℝ) : Prop :=
  (E = (2/3 * A.1 + 1/3 * B.1, 2/3 * A.2 + 1/3 * B.2)) ∧
  (F = (1/3 * A.1 + 2/3 * B.1, 1/3 * A.2 + 2/3 * B.2))

-- Definition: Points G and H are trisection points on CD
def points_trisection_CD (C D G H : ℝ × ℝ) : Prop :=
  (G = (2/3 * C.1 + 1/3 * D.1, 2/3 * C.2 + 1/3 * D.2)) ∧
  (H = (1/3 * C.1 + 2/3 * D.1, 1/3 * C.2 + 2/3 * D.2))

-- Theorem: Area of quadrilateral EFHG is 1/3
theorem area_of_quad_EFHG_is_one_third : Prop :=
  ∀ (A B C D E F G H : ℝ × ℝ),
  quadrilateral_ABCD_with_area_1 → points_trisection_AB A B C D E F → points_trisection_CD C D G H →
  (some function to calculate area of quadrilateral EFHG) = (1 / 3)

-- Placeholder for the actual proof
theorem proof_area_of_quad_EFHG_is_one_third (A B C D E F G H : ℝ × ℝ)
  (habcd : quadrilateral_ABCD_with_area_1)
  (trisectAB : points_trisection_AB A B C D E F)
  (trisectCD : points_trisection_CD C D G H) : area_of_quad_EFHG_is_one_third :=
by sorry

end area_of_quad_EFHG_is_one_third_proof_area_of_quad_EFHG_is_one_third_l812_812384


namespace total_balls_in_bag_l812_812111

theorem total_balls_in_bag (x : ℕ) (H : 3/(4 + x) = x/(4 + x)) : 3 + 1 + x = 7 :=
by
  -- We would provide the proof here, but it's not required as per the instructions.
  sorry

end total_balls_in_bag_l812_812111


namespace QM_bisects_RL_l812_812385

noncomputable theory

variables {α : Type*} [EuclideanGeometry α]
variables {A B C H D E M P Q R L : α}

-- Conditions
variables (h_triangle : acute_triangle A B C)
variables (h_orthocenter_H : is_orthocenter H A B C)
variables (h_altitude_D : is_altitude B D A C)
variables (h_altitude_E : is_altitude C E A B)
variables (h_midpoint_M : midpoint M B C)
variables (h_on_BM : ∀ P, on_line_seg P B M)
variables (h_on_DE : ∀ Q, on_line_seg Q D E)
variables (h_on_PQ : ∀ R, on_line_seg R P Q)
variables (h_ratio_1 : ∀ {P Q}, BP / EQ = CP / DQ)
variables (h_ratio_2 : ∀ {P Q R}, BP / EQ = PR / QR)
variables (h_orthocenter_L : is_orthocenter L A H R)

-- Prove that
theorem QM_bisects_RL :
  is_bisector Q M R L :=
sorry

end QM_bisects_RL_l812_812385


namespace inequality_holds_l812_812312

theorem inequality_holds (k : ℝ) (x : ℝ) :
  (kx^2 - kx + 1 > 0) ↔ k ∈ set.Ico 0 4 :=
sorry

end inequality_holds_l812_812312


namespace julie_reads_tomorrow_l812_812140

theorem julie_reads_tomorrow :
  let total_pages := 120
  let pages_read_yesterday := 12
  let pages_read_today := 2 * pages_read_yesterday
  let pages_read_so_far := pages_read_yesterday + pages_read_today
  let remaining_pages := total_pages - pages_read_so_far
  remaining_pages / 2 = 42 :=
by
  sorry

end julie_reads_tomorrow_l812_812140


namespace angle_KMT_l812_812788

-- Define the problem conditions in Lean
structure triangle (A B C : Type) :=
(right_angle : A = B ∧ B = C)

def is_right_triangle (ABC : triangle A B C) : Prop :=
  ABC.right_angle

structure right_triangle (X Y Z : Type) :=
(base_triangle : triangle X Y Z)
(angle_XYZ: X = Y)

-- Main problem
theorem angle_KMT (A B C T K M : Type)
  [right_triangle ABC (angle B C A = 90)]
  (ABT : right_triangle AB T)
  (ACK : right_triangle AC K)
  (M_on_BC : M = BC / 2) : 
  angle KMT = 120 :=
sorry

end angle_KMT_l812_812788


namespace jason_cousins_l812_812744

theorem jason_cousins (x y : Nat) (dozen_to_cupcakes : Nat) (number_of_cousins : Nat)
    (hx : x = 4) (hy : y = 3) (hdozen : dozen_to_cupcakes = 12)
    (h : number_of_cousins = (x * dozen_to_cupcakes) / y) : number_of_cousins = 16 := 
by
  rw [hx, hy, hdozen] at h
  exact h
  sorry

end jason_cousins_l812_812744


namespace find_q_extend_arithmetic_sequence_l812_812030

-- Definitions related to geometric sequence and arithmetic sequence

def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n
def sum_first_n_terms (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

def is_arithmetic_sequence {α : Type*} [Add α] [Mul α] (x y z : α) : Prop :=
  2 * y = x + z

-- Part (I): Prove value of q
theorem find_q (a : ℝ) (q Sn : ℕ → ℝ) (h1 : is_arithmetic_sequence (Sn 1) (Sn 3) (Sn 4)) : 
  q = (1 + Real.sqrt 5) / 2 ∨ q = (1 - Real.sqrt 5) / 2 := 
sorry

-- Part (II): Prove the arithmetic sequence property for any natural number k
theorem extend_arithmetic_sequence (a : ℝ) (q : ℝ) (an : ℕ → ℝ) (m n l k : ℕ) 
  (h2 : is_arithmetic_sequence (sum_first_n_terms a q m) (sum_first_n_terms a q n) (sum_first_n_terms a q l)) :
  is_arithmetic_sequence (an (m + k)) (an (n + k)) (an (l + k)) := 
sorry

end find_q_extend_arithmetic_sequence_l812_812030


namespace min_value_frac_sqrt_l812_812768

theorem min_value_frac_sqrt (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (∃ c, (∀ x y, 0 < x → 0 < y → c ≤ (sqrt ((2*x^2 + y^2)*(4*x^2 + y^2)) / (x*y))) ∧
  (∀ x y, 0 < x → 0 < y → (c = (sqrt ((2*x^2 + y^2)*(4*x^2 + y^2)) / (x*y)) → x = y * real.sqrt (real.sqrt 8)))) :=
sorry

end min_value_frac_sqrt_l812_812768


namespace evaluate_series_l812_812321

theorem evaluate_series : 1 + (1 / 2) + (1 / 4) + (1 / 8) = 15 / 8 := by
  sorry

end evaluate_series_l812_812321


namespace sum_of_x_coords_is_zero_l812_812958

def point := (ℕ × ℕ)

def above_line (p : point) : Prop := 
  let (x, y) := p
  y > 3 * x + 5

def sum_of_x_coords_above_line (points : list point) : ℕ :=
  points.filter above_line |>.map prod.fst |>.sum

theorem sum_of_x_coords_is_zero :
  sum_of_x_coords_above_line [(4, 12), (7, 23), (13, 38), (19, 43), (21, 55)] = 0 :=
sorry

end sum_of_x_coords_is_zero_l812_812958


namespace square_root_of_factorial_division_l812_812003

theorem square_root_of_factorial_division :
  (sqrt (9! / 126) = 24 * sqrt 5) :=
sorry

end square_root_of_factorial_division_l812_812003


namespace how_many_n_leq_1000_l812_812018

noncomputable def floor_sum_not_divisible_by_5 (n : ℕ) : Prop :=
  (Nat.floor (500 / n) + Nat.floor (501 / n) + Nat.floor (502 / n)) % 5 ≠ 0

theorem how_many_n_leq_1000 : 
  (finset.range 1000).filter (λ n, n > 0 ∧ floor_sum_not_divisible_by_5 (n + 1)).card = 24 := 
by {
  sorry
}

end how_many_n_leq_1000_l812_812018


namespace original_denominator_is_15_l812_812924

theorem original_denominator_is_15
  (d : ℕ)
  (h1 : (6 : ℚ) / (d + 3) = 1 / 3)
  (h2 : 3 = 3) : d = 15 := -- h2 is trivial but included according to the problem condition
by
  sorry

end original_denominator_is_15_l812_812924


namespace square_root_of_factorial_division_l812_812005

theorem square_root_of_factorial_division :
  (sqrt (9! / 126) = 24 * sqrt 5) :=
sorry

end square_root_of_factorial_division_l812_812005


namespace hou_yi_score_l812_812675

theorem hou_yi_score (a b c : ℕ) (h1 : 2 * b + c = 29) (h2 : 2 * a + c = 43) : a + b + c = 36 := 
by 
  sorry

end hou_yi_score_l812_812675


namespace factorization_l812_812592

theorem factorization (a : ℝ) : a^3 + 2 * a^2 + a = a * (a + 1)^2 := 
sorry

end factorization_l812_812592


namespace root_of_equation_imp_expression_eq_one_l812_812032

variable (m : ℝ)

theorem root_of_equation_imp_expression_eq_one
  (h : m^2 - m - 1 = 0) : m^2 - m = 1 :=
  sorry

end root_of_equation_imp_expression_eq_one_l812_812032


namespace inequality_solution_l812_812067

-- Problem statement: Prove that for the inequality k x^2 - 2 x + 6 k < 0 to hold for all real x, k must be less than -sqrt(6)/6.
theorem inequality_solution (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) → k < - real.sqrt 6 / 6 :=
sorry

end inequality_solution_l812_812067


namespace magnitude_of_z_l812_812819

theorem magnitude_of_z (w z : ℂ) (h1 : w * z = 15 - 20 * complex.I) (h2 : complex.abs w = 5) : complex.abs z = 5 :=
by
  sorry

end magnitude_of_z_l812_812819


namespace net_profit_correct_l812_812233

noncomputable def purchase_price : ℝ := 48
noncomputable def overhead_rate : ℝ := 0.05
noncomputable def markup : ℝ := 30

def overhead := overhead_rate * purchase_price
def net_profit := markup - overhead

theorem net_profit_correct :
  net_profit = 27.60 :=
by
  unfold net_profit overhead
  sorry

end net_profit_correct_l812_812233


namespace long_furred_and_brown_dogs_l812_812704

theorem long_furred_and_brown_dogs (total_dogs long_furred_dogs brown_dogs neither_dogs : ℕ)
  (h_total : total_dogs = 45) (h_long_furred : long_furred_dogs = 26)
  (h_brown : brown_dogs = 30) (h_neither : neither_dogs = 8) :
  let both := long_furred_dogs + brown_dogs - (total_dogs - neither_dogs) in
  both = 19 := by
  sorry

end long_furred_and_brown_dogs_l812_812704


namespace proportion_false_if_x_is_0_75_correct_value_of_x_in_proportion_l812_812092

theorem proportion_false_if_x_is_0_75 (x : ℚ) (h1 : x = 0.75) : ¬ (x / 2 = 2 / 6) :=
by sorry

theorem correct_value_of_x_in_proportion (x : ℚ) (h1 : x / 2 = 2 / 6) : x = 2 / 3 :=
by sorry

end proportion_false_if_x_is_0_75_correct_value_of_x_in_proportion_l812_812092


namespace expected_vertices_convex_hull_approx_22_l812_812906

noncomputable def expected_vertices_convex_hull (n : ℕ) : ℝ :=
  O(n ^ (1 / 3 : ℝ))

theorem expected_vertices_convex_hull_approx_22 : 
  expected_vertices_convex_hull 10000 ≈ 22 :=
sorry

end expected_vertices_convex_hull_approx_22_l812_812906


namespace arccos_one_eq_zero_l812_812493

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812493


namespace largest_multiple_of_9_less_than_100_l812_812305

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ (∀ k, k < 100 → k % 9 = 0 → k ≤ n) :=
by {
  use 99,
  split,
  { exact lt_of_le_of_ne (nat.le_of_dvd (by norm_num) (by norm_num)) (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by use 11) },
  { intros k hk hkm, apply nat.mul_le_mul_left 9 (le_of_lt $ nat.div_lt_self hk $ by norm_num), exact le_of_lt_succ (nat.succ_le_of_lt (nat.div_lt_self hk $ by norm_num)) }
  sorry
}

end largest_multiple_of_9_less_than_100_l812_812305


namespace smallest_n_mod_equality_l812_812971

theorem smallest_n_mod_equality :
  ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 5]) ∧ (∀ m : ℕ, 0 < m ∧ m < n → (7^m ≡ m^7 [MOD 5]) = false) :=
  sorry

end smallest_n_mod_equality_l812_812971


namespace sum_fractional_zeta2k_zero_l812_812613

open Real

noncomputable
def zeta (x : ℝ) : ℝ := ∑' n, 1 / (n ^ x)

noncomputable
def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

noncomputable
def sum_fractional_zeta2k : ℝ := ∑' k, fractional_part (zeta (2 * k))

theorem sum_fractional_zeta2k_zero (h : ∀ x, x > 1 → zeta x = ∑' n, 1 / (n ^ x)) :
  sum_fractional_zeta2k = 0 := by
  sorry

end sum_fractional_zeta2k_zero_l812_812613


namespace volume_solid_rotated_l812_812015

-- Define the curve y = x^3 - x
def curve (x : ℝ) : ℝ := x^3 - x

-- Define the line y = x
def line (x : ℝ) : ℝ := x

-- Define the integral representing the volume of the solid of revolution
noncomputable def volume_of_solid : ℝ :=
  let r (u : ℝ) := (4 * u)^(1/3) - u in
  π * ∫ u in 0..2, r u ^ 2

-- The main theorem stating the volume
theorem volume_solid_rotated :
  volume_of_solid = 64 * π / 105 :=
begin
  sorry
end

end volume_solid_rotated_l812_812015


namespace arccos_one_eq_zero_l812_812435

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812435


namespace no_infinite_sequence_divisible_by_succ_l812_812980

theorem no_infinite_sequence_divisible_by_succ :
  ¬ ∃ (a : ℕ → ℕ), ∀ k : ℕ, ∀ m : ℕ, m < k →
    ∑ i in range k, a(m + i) % (k + 1) = 0 := sorry

end no_infinite_sequence_divisible_by_succ_l812_812980


namespace arithmetic_sequence_general_term_and_sum_l812_812036

theorem arithmetic_sequence_general_term_and_sum :
  (∃ a : ℕ → ℝ, (a 3 = 4) ∧ ((\sum i in finset.range 6, a (i + 1)) = 27) 
  ∧ (∀ n, a n = n + 1) 
  ∧ (∃ m, (4 * (2 ^ m - 1) = 124) → m = 5) := 
by sorry

end arithmetic_sequence_general_term_and_sum_l812_812036


namespace common_difference_arithmetic_sequence_l812_812629

theorem common_difference_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 1) (h2 : a 3 + a 5 = 14) :
  d = 2 :=
by {
  -- Define an arithmetic sequence
  have a_def : ∀ n, a n = a 1 + (n - 1) * d := sorry,
  
  -- Evaluate a 3 and a 5 in terms of d
  have a3 : a 3 = a 1 + 2 * d,
  { rw a_def, linarith },
  
  have a5 : a 5 = a 1 + 4 * d,
  { rw a_def, linarith },
  
  -- Use given conditions
  have cond : a 1 + 2 * d + (a 1 + 4 * d) = 14,
  { rw [a3, a5, h1], linarith },
  
  -- Prove that d = 2
  linarith,
}

end common_difference_arithmetic_sequence_l812_812629


namespace arccos_one_eq_zero_l812_812519

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812519


namespace g_has_exactly_three_zeros_in_0_5π_l812_812693

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos ((2 / 3) * x)

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.cos (((2 / 3) * x) - (π / 3))

theorem g_has_exactly_three_zeros_in_0_5π :
  (∃ x1 x2 x3 ∈ set.Ioo 0 (5 * π), g x1 = 0 ∧ g x2 = 0 ∧ g x3 = 0 ∧
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
    (∀ x ∈ set.Ioo 0 (5 * π), g x = 0 → x = x1 ∨ x = x2 ∨ x = x3)) := sorry

end g_has_exactly_three_zeros_in_0_5π_l812_812693


namespace system_solutions_l812_812332

-- Define the four variables a, b, c, and d and their conditions
variables (a b c d : ℝ)

-- The four equations given in the problem
def system_of_equations :=
  (a^2 = (1 / b) + (1 / c) + (1 / d)) ∧
  (b^2 = (1 / a) + (1 / c) + (1 / d)) ∧
  (c^2 = (1 / a) + (1 / b) + (1 / d)) ∧
  (d^2 = (1 / a) + (1 / b) + (1 / c))

-- The proposed solutions
def solution_1 := (a = Real.cbrt 3) ∧ (b = Real.cbrt 3) ∧ (c = Real.cbrt 3) ∧ (d = Real.cbrt 3)
def solution_2 := (a = Real.cbrt (4 / 3)) ∧ (b = Real.cbrt (4 / 3)) ∧ (c = Real.cbrt (4 / 3)) ∧ (d = - Real.cbrt (9 / 2))
def solution_3 := (a = (-1 + Real.sqrt 5) / 2) ∧ (b = (-1 + Real.sqrt 5) / 2) ∧ (c = (-1 - Real.sqrt 5) / 2) ∧ (d = (-1 - Real.sqrt 5) / 2)

-- The theorem to prove that the set of proposed solutions satisfies the system of equations
theorem system_solutions :
  system_of_equations a b c d ↔
  (solution_1 a b c d ∨ solution_2 a b c d ∨ solution_3 a b c d) :=
by sorry

end system_solutions_l812_812332


namespace number_of_female_students_in_pool_l812_812854

variable {x : ℕ}

-- Given conditions
def total_students : ℕ := 200
def selected_students : ℕ := 30
def selected_male_students : ℕ := 12
def selected_female_students : ℕ := selected_students - selected_male_students

-- The goal is to show:
theorem number_of_female_students_in_pool :
  let total_students := 200
  let selected_students := 30
  let remaining_students := selected_students - selected_male_students
  let ratio_selected_female := (remaining_students.toRat) / selected_students.toRat
  let ratio_total_female := (x.toRat) / total_students.toRat
  ratio_total_female = ratio_selected_female
  ∧ x = 120 :=
by
  sorry

end number_of_female_students_in_pool_l812_812854


namespace area_of_square_PQRS_l812_812192

-- Definitions based on conditions
variables (X Y Z P Q R S : Type)
variables (XP YQ : ℝ)
variables (XY XZ YZ : ℝ)
variable (s : ℝ)

-- Assumptions based on the conditions
axiom right_triangle_XYZ : ∀ (X Y Z : Type), is_right_triangle XYZ
axiom point_positions : ∀ (P Q R S : Type), (P ∈ XY) ∧ (Q ∈ XY) ∧ (R ∈ YZ) ∧ (S ∈ XZ)
axiom XP_value : XP = 30
axiom YQ_value : YQ = 70
axiom XY_position : XP • XY = XP
axiom s_def : ∀ (s : ℝ), (30 / s) = (s / 70)

-- Theorem stating the final goal
theorem area_of_square_PQRS : s * s = 2100 :=
by
  sorry

end area_of_square_PQRS_l812_812192


namespace hyperbola_m_value_l812_812658

noncomputable def m_value : ℝ := 2 * (Real.sqrt 2 - 1)

theorem hyperbola_m_value (a : ℝ) (m : ℝ) (AF_2 AF_1 BF_2 BF_1 : ℝ)
  (h1 : a = 1)
  (h2 : AF_2 = m)
  (h3 : AF_1 = 2 + AF_2)
  (h4 : AF_1 = m + BF_2)
  (h5 : BF_2 = 2)
  (h6 : BF_1 = 4)
  (h7 : BF_1 = Real.sqrt 2 * AF_1) :
  m = m_value :=
by
  sorry

end hyperbola_m_value_l812_812658


namespace more_girls_than_boys_l812_812103

theorem more_girls_than_boys :
  ∀ (boys girls : ℕ),
  (5 * girls = 13 * boys) ∧ (boys = 80) →
  (girls - boys = 128) :=
by
  intros boys girls h
  cases h with ratio boys_count
  have ratio := ratio
  sorry

end more_girls_than_boys_l812_812103


namespace min_value_expr_l812_812083

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 ≥ 4 :=
sorry

end min_value_expr_l812_812083


namespace find_r_l812_812161

-- Declaring the roots of the first polynomial
variables (a b m : ℝ)
-- Declaring the roots of the second polynomial
variables (p r : ℝ)

-- Assumptions based on the given conditions
def roots_of_first_eq : Prop :=
  a + b = m ∧ a * b = 3

def roots_of_second_eq : Prop :=
  ∃ (p : ℝ), (a^2 + 1/b) * (b^2 + 1/a) = r

-- The desired theorem
theorem find_r 
  (h1 : roots_of_first_eq a b m)
  (h2 : (a^2 + 1/b) * (b^2 + 1/a) = r) :
  r = 46/3 := by sorry

end find_r_l812_812161


namespace part_a_l812_812314

theorem part_a (z : Complex) : Complex.abs (z - Complex.I*2) ≥ 4 ↔ (Complex.abs -z.re^2 + (z.im - 2)^2 ≥ 16) := sorry

end part_a_l812_812314


namespace frosting_sugar_l812_812346

-- Define the conditions as constants
def total_sugar : ℝ := 0.8
def cake_sugar : ℝ := 0.2

-- The theorem stating that the sugar required for the frosting is 0.6 cups
theorem frosting_sugar : total_sugar - cake_sugar = 0.6 := by
  sorry

end frosting_sugar_l812_812346


namespace sum_of_coordinates_reflection_l812_812183

theorem sum_of_coordinates_reflection (y : ℝ) : 
  let C := (3, y)
  let D := (3, -y)
  (C.1 + C.2 + D.1 + D.2) = 6 :=
by
  let C := (3, y)
  let D := (3, -y)
  have h : C.1 + C.2 + D.1 + D.2 = 6 := sorry
  exact h

end sum_of_coordinates_reflection_l812_812183


namespace min_triangle_perimeter_l812_812699

def triangle (A B C : Type) : Prop := sorry

variable {ABC : Type} [triangle ABC]
variables (AB AC BC : ℕ)
variable (ω : Type) -- ω is the circle centered at the incenter

-- Conditions
axiom AB_eq_AC : AB = AC
axiom sides_are_integers : AB ∈ ℕ ∧ AC ∈ ℕ ∧ BC ∈ ℕ
axiom internal_tangency (ω : Type) : 
  (∃ r_a r_b r_c : ℕ, r_a = r_b ∧ r_a = r_c ∧ 
   ∀ (a b : ℕ) (h: a ∈ ℕ ∧ b ∈ ℕ), 
   (a = BC → (s - b) = r_a) ∧ 
   (b = AB → (s - a) = r_b))

-- Theorem statement
theorem min_triangle_perimeter : 
  ∃(P : ℕ), (P = BC + 2*AB) ∧ 
  minimal (λ P, P) P :=
  sorry

end min_triangle_perimeter_l812_812699


namespace coloring_possible_l812_812178

-- Define what it means for a graph to be planar and bipartite
def planar_graph (G : Type) : Prop := sorry
def bipartite_graph (G : Type) : Prop := sorry

-- The planar graph G results after subdivision without introducing new intersections
def subdivided_graph (G : Type) : Type := sorry

-- Main theorem to prove
theorem coloring_possible (G : Type) (h1 : planar_graph G) : 
  bipartite_graph (subdivided_graph G) :=
sorry

end coloring_possible_l812_812178


namespace machines_needed_second_scenario_l812_812090

noncomputable def second_scenario_machines (machines1 : ℕ) (portion1 : ℚ) (time1 : ℕ) 
  (portion2 : ℚ) (time2 : ℕ) : ℕ :=
  let rate1 := portion1 / (machines1 * time1)
  let rate_one_machine := rate1 / machines1
  let portion_per_machine_in_time2 := rate_one_machine * time2
in  (portion2 / portion_per_machine_in_time2).toNat

theorem machines_needed_second_scenario :
  second_scenario_machines 5 (3/4 : ℚ) 30 (3/5 : ℚ) 60 = 2 :=
by
  sorry

end machines_needed_second_scenario_l812_812090


namespace triangle_area_l812_812110

/-- In an isosceles triangle ABC with base AC and side AB equal to 2, the segment AD is the angle bisector. 
    A tangent DH to the circumcircle of triangle ADB is drawn through point D, with point H lying on side AC.
    Given that CD = sqrt(2) * CH, the area of triangle ABC is sqrt(4 * sqrt(2) - 5). --/
theorem triangle_area (A B C D H : Point)
  (isosceles : is_isosceles ⟨A, B, C⟩ AC AB 2)
  (angle_bisector : is_angle_bisector ⟨A, D, B⟩)
  (tangent : is_tangent ⟨D, H⟩ (circumcircle ⟨A, D, B⟩))
  (H_on_AC : lies_on H AC)
  (CD_eq : length (segment D C) = sqrt(2) * length (segment C H)) :
  area (triangle A B C) = sqrt(4 * sqrt(2) - 5) :=
sorry

end triangle_area_l812_812110


namespace intersection_of_sets_l812_812661

open set

theorem intersection_of_sets :
  let A := { x : ℝ | x^2 - 4*x + 3 < 0 } in
  let B := { x : ℝ | 2 < x ∧ x < 4 } in
  A ∩ B = { x : ℝ | 2 < x ∧ x < 3 } := 
by
  let A := { x : ℝ | x^2 - 4*x + 3 < 0 }
  let B := { x : ℝ | 2 < x ∧ x < 4 }
  have h1 : A = { x | 1 < x ∧ x < 3 } := by sorry
  show A ∩ B = { x | 2 < x ∧ x < 3 } from by sorry

end intersection_of_sets_l812_812661


namespace profit_is_29_percent_l812_812868

variables (P C : ℝ)
variables (h : 2/3 * P = 0.86 * C)

def profit_percent(P C : ℝ) (h : 2/3 * P = 0.86 * C) : ℝ :=
  ((P - C) / C) * 100

theorem profit_is_29_percent (P C : ℝ) (h : 2/3 * P = 0.86 * C) :
  profit_percent P C h = 29 := by
  sorry

end profit_is_29_percent_l812_812868


namespace arccos_one_eq_zero_l812_812548

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812548


namespace find_lambda_l812_812073

variable (λ : ℝ)

def vector_a : ℝ × ℝ := (2, 4)
def vector_b : ℝ × ℝ := (2, λ)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_lambda (h_par : parallel (vector_add vector_a (scalar_mul 2 vector_b)) (vector_add (scalar_mul 2 vector_a) vector_b)) :
  λ = 4 :=
sorry

end find_lambda_l812_812073


namespace smallest_n_mod_equality_l812_812973

theorem smallest_n_mod_equality :
  ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 5]) ∧ (∀ m : ℕ, 0 < m ∧ m < n → (7^m ≡ m^7 [MOD 5]) = false) :=
  sorry

end smallest_n_mod_equality_l812_812973


namespace suff_but_not_nec_condition_suff_but_not_nec_condition_rev_l812_812890

theorem suff_but_not_nec_condition (x : ℝ) : x > 1 → x^2 > x :=
by
  assume h : x > 1
  sorry

theorem suff_but_not_nec_condition_rev (x : ℝ) : x^2 > x → (x > 1 ∨ x < 0) :=
by
  assume h : x^2 > x
  sorry

end suff_but_not_nec_condition_suff_but_not_nec_condition_rev_l812_812890


namespace trigonometric_identity_l812_812634

theorem trigonometric_identity (a θ : ℝ) (h : a > 1) (h1 : tan θ = a) :
  (sin (π / 4 + θ) / sin (π / 2 - θ) * tan (2 * θ)) = (sqrt 2 * a / (1 - a)) := by
sorry

end trigonometric_identity_l812_812634


namespace arccos_one_eq_zero_l812_812530

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812530


namespace arccos_one_eq_zero_l812_812428

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812428


namespace bobby_initial_candies_l812_812389

theorem bobby_initial_candies (a b c x : ℕ) : 
  a = 5 → b = 9 → c = 7 → (x - a - b = c) → x = 21 :=
by
  intros ha hb hc H
  rw [ha, hb, hc] at H
  linarith

end bobby_initial_candies_l812_812389


namespace inverse_function_inequality_l812_812204

noncomputable def f_inv (x : ℝ) : ℝ := sorry

theorem inverse_function_inequality
  (f : ℝ → ℝ)
  (h_inv : ∀ y, f (f_inv y) = y ∧ f_inv (f y) = y)
  (h_property : ∀ x1 x2 > 0, f (x1 * x2) = f x1 + f x2)
  (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  f_inv (x1) + f_inv (x2) ≥ 2 * f_inv (x1 / 2) * f_inv (x2 / 2) :=
sorry

end inverse_function_inequality_l812_812204


namespace minimum_distinct_terms_of_scalene_6tuple_l812_812250

def scalene_triangle (a b c : ℝ) (α β γ : ℝ) :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ α ≠ β ∧ β ≠ γ ∧ γ ≠ α

def law_of_sines (a b c α β γ : ℝ) :=
  a / (Real.sin α) = b / (Real.sin β) ∧ b / (Real.sin β) = c / (Real.sin γ)

theorem minimum_distinct_terms_of_scalene_6tuple {a b c α β γ : ℝ} 
  (h_scalene_triangle : scalene_triangle a b c α β γ)
  (h_law_of_sines : law_of_sines a b c α β γ) :
  ∃ n, n = 4 ∧ n = set.card (set.of_list [a, b, c, α, β, γ]) := 
sorry

end minimum_distinct_terms_of_scalene_6tuple_l812_812250


namespace arccos_one_eq_zero_l812_812533

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812533


namespace angle_AQG_in_regular_octagon_is_90_l812_812803

theorem angle_AQG_in_regular_octagon_is_90 
  (ABCDEFGH : tuple (fin 8 → fin 2 → ℝ)) 
  (reg_oct : ∀ i : fin 8, ∃ j : fin 8, ∠(ABCDEFGH i) = 135) 
  (A B G H Q : fin 2)
  (extension_AB_Q : line (ABCDEFGH A) (ABCDEFGH B) = extension_line (ABCDEFGH Q))
  (extension_GH_Q : line (ABCDEFGH G) (ABCDEFGH H) = extension_line (ABCDEFGH Q)) :
∠(ABCDEFGH Q) = 90 :=
sorry

end angle_AQG_in_regular_octagon_is_90_l812_812803


namespace trigonometric_identity_l812_812583

theorem trigonometric_identity :
  sin 50 * sin 70 - cos 50 * sin 20 = 1 / 2 := by
  sorry

end trigonometric_identity_l812_812583


namespace polynomial_solution_l812_812596

theorem polynomial_solution (P : ℝ[X]) (hP : ∀ x : ℝ, abs (P.eval x - x) ≤ x^2 + 1) :
  ∃ (a b c : ℤ), (P = X^2 * a + X * (b + 1) + c) ∧ (-1 ≤ c ∧ c ≤ 1) ∧
  (abs a = 1 ∧ b = 0 ∨ abs a < 1 ∧ abs b ≤ 2 * sqrt (1 + a.toReal * c.toReal - abs (a.toReal + c.toReal))) :=
sorry

end polynomial_solution_l812_812596


namespace median_of_data_set_l812_812619

theorem median_of_data_set :
  ∀ (x : ℝ), (mode {8, 8, x, 10, 10} = mean {8, 8, x, 10, 10}) → (median {8, 8, x, 10, 10} = 8 ∨ median {8, 8, x, 10, 10} = 10) := 
by 
  sorry

end median_of_data_set_l812_812619


namespace arccos_one_eq_zero_l812_812486

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812486


namespace arccos_one_eq_zero_l812_812476

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812476


namespace mean_proportional_49_64_l812_812990

theorem mean_proportional_49_64 : Real.sqrt (49 * 64) = 56 :=
by
  sorry

end mean_proportional_49_64_l812_812990


namespace distinct_values_l812_812052

-- Define the expressions as terms in Lean
def expr1 : ℕ := 3 ^ (3 ^ 3)
def expr2 : ℕ := (3 ^ 3) ^ 3

-- State the theorem that these terms yield exactly two distinct values
theorem distinct_values : (expr1 ≠ expr2) ∧ ((expr1 = 3^27) ∨ (expr1 = 19683)) ∧ ((expr2 = 3^27) ∨ (expr2 = 19683)) := 
  sorry

end distinct_values_l812_812052


namespace garden_length_80_l812_812364

-- Let the width of the garden be denoted by w and the length by l
-- Given conditions
def is_rectangular_garden (l w : ℝ) := l = 2 * w ∧ 2 * l + 2 * w = 240

-- We want to prove that the length of the garden is 80 yards
theorem garden_length_80 (w : ℝ) (h : is_rectangular_garden (2 * w) w) : 2 * w = 80 :=
by
  sorry

end garden_length_80_l812_812364


namespace convex_polygon_formation_l812_812888

section geometric_problem

variable (Γ : Type) [metric_space Γ] [has_angles Γ]
variables (A A' O : Γ) (n : ℕ) (A_i : fin n → Γ) (A_i' : fin n → Γ) (X : fin n → Γ)

-- Conditions
-- 1. Γ are two identical shapes.
def identical_shape : Prop := (∀ i, dist A (A_i i) = dist A' (A_i' i)) ∧ (∃ θ : ℝ, ∀ i, ∠ (A, O, A_i i) = ∠ (A', O, A_i' i) + θ)

-- 2. A and A' are the ends of their short strokes.
-- 3. The long strokes are divided into n equal parts creating points A_1, ..., A_n-1 on one and A_1', ..., A_n-1' on the other.
def divided_into_equal_parts : Prop := (∀ i, dist A (A_i i) = dist A' (A_i' i))

-- 4. The lines AA_i and A'A_i' intersect at the point X_i for i = 1, ..., n-1.
def lines_intersect (i : fin n) : Prop := ∃ X_i : Γ, ∃ k : ℝ, ∃ l : ℝ, X_i = k • A + (1 - k) • (A_i i) ∧ X_i = l • A' + (1 - l) • (A_i' i)

-- Mathematical equivalent proof problem
def convex_polygon_points (n : ℕ) : Prop :=
  (∀ i, lines_intersect A A' i) →
  identical_shape A A' Γ →
  divided_into_equal_parts A A' Γ →
  convex_list (filter (λ j, j < n) X)

theorem convex_polygon_formation (h1 : identical_shape A A' Γ)
  (h2 : divided_into_equal_parts A A' Γ)
  (h3 : ∀ i, lines_intersect A A' i):
  convex_polygon_points n :=
sorry

end geometric_problem

end convex_polygon_formation_l812_812888


namespace tangent_line_ln_curve_l812_812095

theorem tangent_line_ln_curve (b : ℝ) : 
  (∃ (m : ℝ) (Hpos : m > 0), (1 / m) = (1 / 3) ∧ (m, real.log m) ∈ {p : ℝ × ℝ | p.snd = (1 / 3) * p.fst + b}) → 
  b = real.log 3 - 1 :=
begin
  sorry
end

end tangent_line_ln_curve_l812_812095


namespace arccos_one_eq_zero_l812_812400

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812400


namespace appetizer_cost_is_9_60_l812_812940

noncomputable def meal_without_appetizer_and_discounted_entree : ℝ := 20 + 2*3 + 6
noncomputable def discounted_entree : ℝ := 20 / 2
noncomputable def meal_with_discounted_entree : ℝ := discounted_entree + 2*3 + 6
noncomputable def full_cost_without_discount : ℝ := 20 + 2*3 + 6
noncomputable def tip : ℝ := 0.20 * full_cost_without_discount
noncomputable def total_meal_cost_with_discount_and_tip : ℝ := meal_with_discounted_entree + tip

axiom total_spent_by_arthur : ℝ := 38

noncomputable def appetizer_cost : ℝ := total_spent_by_arthur - total_meal_cost_with_discount_and_tip

theorem appetizer_cost_is_9_60 :
  appetizer_cost = 9.60 :=
by
  sorry

end appetizer_cost_is_9_60_l812_812940


namespace arccos_one_eq_zero_l812_812562

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812562


namespace arccos_one_eq_zero_l812_812526

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812526


namespace arccos_one_eq_zero_l812_812549

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812549


namespace arccos_one_eq_zero_l812_812470

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812470


namespace arccos_one_eq_zero_l812_812544

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812544


namespace arccos_one_eq_zero_l812_812478

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812478


namespace max_value_01_max_value_R_l812_812142

def f (x y : ℝ) : ℝ :=
  (x + y) / ((x^2 + 1)*(y^2 + 1))

theorem max_value_01 (x y : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) : 
  f x y ≤ 1/2 :=
sorry

theorem max_value_R (x y : ℝ) : 
  f x y ≤ (3 * Real.sqrt 3) / 8 :=
sorry

end max_value_01_max_value_R_l812_812142


namespace smallest_prime_factor_of_2537_l812_812309

-- Definition of smallest prime factor with conditions provided
def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then
    let ps := Nat.factors n in
    List.head ps
  else 0

theorem smallest_prime_factor_of_2537 : smallest_prime_factor 2537 = 7 := by
  sorry

end smallest_prime_factor_of_2537_l812_812309


namespace calculate_expression_l812_812683

open Complex

def B : Complex := 5 - 2 * I
def N : Complex := -3 + 2 * I
def T : Complex := 2 * I
def Q : ℂ := 3

theorem calculate_expression : B - N + T - 2 * Q = 2 - 2 * I := by
  sorry

end calculate_expression_l812_812683


namespace tank_capacity_l812_812218

theorem tank_capacity (one_third_full : ℚ) (added_water : ℚ) (capacity : ℚ) 
  (h1 : one_third_full = 1 / 3) 
  (h2 : 2 * one_third_full * capacity = 16) 
  (h3 : added_water = 16) 
  : capacity = 24 := 
by
  sorry

end tank_capacity_l812_812218


namespace negation_of_exists_l812_812833

theorem negation_of_exists {P : Type} (x0 y0 : P)
  (h : ∃ (P : P) (x0 y0 : P), x0^2 + y0^2 - 1 ≤ 0): 
  (∀ (x y : P), x^2 + y^2 - 1 > 0) :=
  sorry

end negation_of_exists_l812_812833


namespace min_x_fifth_iteration_positive_integer_l812_812655

-- Given function f
def f (x : ℝ) : ℝ := (4 / 5) * (x - 1)

-- Fifth iteration of f
def f_iter (f : ℝ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  nat.iterate f n x

-- Proof statement for minimum x such that f^(5)(x) is a positive integer
theorem min_x_fifth_iteration_positive_integer :
  ∃ x : ℤ, f_iter f 5 (x : ℝ) > 0 ∧ x = 3121 := 
sorry

end min_x_fifth_iteration_positive_integer_l812_812655


namespace transformations_return_T_l812_812149

noncomputable theory

-- Define the type for points in the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define triangle T' with specific vertices
def T' : Point × Point × Point :=
  ({x := 0, y := 0}, {x := 6, y := 0}, {x := 0, y := 4})

-- Define the transformations as functions
def rotate90 (p : Point) : Point := {x := -p.y, y := p.x}
def rotate180 (p : Point) : Point := {x := -p.x, y := -p.y}
def rotate270 (p : Point) : Point := {x := p.y, y := -p.x}
def reflect_y_equals_x (p : Point) : Point := {x := p.y, y := p.x}
def reflect_y_axis (p : Point) : Point := {x := -p.x, y := p.y}

-- Consider a sequence of transformations applied to a triangle
def apply_sequence (seq : List (Point → Point)) (T : Point × Point × Point) : Point × Point × Point := 
  match T with
  | (p1, p2, p3) => (seq.foldl (λ p f => f p) p1, seq.foldl (λ p f => f p) p2, seq.foldl (λ p f => f p) p3)

-- Prove there are 12 sequences of transformations returning T' to its original position
theorem transformations_return_T' : 
  ∃ (seqs : List (List (Point → Point))), seqs.length = 125 ∧ (seqs.count (λ seq => apply_sequence seq T' = T') = 12) :=
  sorry

end transformations_return_T_l812_812149


namespace arccos_one_eq_zero_l812_812558

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812558


namespace internal_diagonal_passes_through_600_cubes_face_diagonal_passes_through_360_cubes_l812_812899

theorem internal_diagonal_passes_through_600_cubes (a b c : ℕ) (h₁ : a = 120) (h₂ : b = 270) (h₃ : c = 300) :
  let g_ab := Nat.gcd a b in
  let g_bc := Nat.gcd b c in
  let g_ac := Nat.gcd a c in
  let g_abc := Nat.gcd (Nat.gcd a b) c in
  (a + b + c - (g_ab + g_bc + g_ac) + g_abc) = 600 := by
  sorry

theorem face_diagonal_passes_through_360_cubes (a b : ℕ) (h₁ : a = 120) (h₂ : b = 270) :
  let g_ab := Nat.gcd a b in
  (a + b - g_ab) = 360 := by
  sorry

end internal_diagonal_passes_through_600_cubes_face_diagonal_passes_through_360_cubes_l812_812899


namespace cube_roof_ratio_proof_l812_812862

noncomputable def cube_roof_edge_ratio : Prop :=
  ∃ (a b : ℝ), (∃ isosceles_triangles symmetrical_trapezoids : ℝ, isosceles_triangles = 2 ∧ symmetrical_trapezoids = 2)
  ∧ (∀ edge : ℝ, edge = a)
  ∧ (∀ face1 face2 : ℝ, face1 = face2)
  ∧ b = (Real.sqrt 5 - 1) / 2 * a

theorem cube_roof_ratio_proof : cube_roof_edge_ratio :=
sorry

end cube_roof_ratio_proof_l812_812862


namespace balloons_kept_by_Andrew_l812_812939

theorem balloons_kept_by_Andrew :
  let blue := 303
  let purple := 453
  let red := 165
  let yellow := 324
  let blue_kept := (2/3 : ℚ) * blue
  let purple_kept := (3/5 : ℚ) * purple
  let red_kept := (4/7 : ℚ) * red
  let yellow_kept := (1/3 : ℚ) * yellow
  let total_kept := blue_kept.floor + purple_kept.floor + red_kept.floor + yellow_kept
  total_kept = 675 := by
  sorry

end balloons_kept_by_Andrew_l812_812939


namespace f_sum_eq_l812_812152

def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem f_sum_eq (x1 x2 : ℝ) (h : x1 + x2 = 1) : 
  f x1 + f x2 = Real.sqrt 3 / 3 :=
by
  -- skipping the proof, here you can add proof steps later
  sorry

end f_sum_eq_l812_812152


namespace solution_set_l812_812995

-- Define the odd function f
variable (f : ℝ → ℝ)

-- Conditions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def slope_lt_one (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, (0 < x1) → (x1 < x2) → (f(x2) - f(x1)) / (x2 - x1) < 1

def f_one (f : ℝ → ℝ) : Prop :=
  f 1 = 1

-- The target statement
theorem solution_set (Hodd : is_odd f) (Hlt : slope_lt_one f) (Hone : f_one f) : 
  {x : ℝ | f(x) - x > 0} = set.Iio (-1) ∪ set.Ioo 0 1 :=
sorry

end solution_set_l812_812995


namespace sum_of_diameters_of_subsets_l812_812333

theorem sum_of_diameters_of_subsets (S : Finset ℕ) (S_def : S = Finset.range 100) :
  ∑ (i : ℕ) in (Finset.Icc 92 100), 91 * 2^90 = 9 * 91 * 2^90 :=
by
  sorry

end sum_of_diameters_of_subsets_l812_812333


namespace arccos_one_eq_zero_l812_812468

theorem arccos_one_eq_zero : ∃ θ ∈ set.Icc 0 π, real.cos θ = 1 ∧ real.arccos 1 = θ :=
by 
  use 0
  split
  {
    exact ⟨le_refl 0, le_of_lt real.pi_pos⟩,
  }
  split
  {
    exact real.cos_zero,
  }
  {
    exact real.arccos_eq_zero,
  }

end arccos_one_eq_zero_l812_812468


namespace smallest_n_7_pow_n_eq_n_7_mod_5_l812_812975

theorem smallest_n_7_pow_n_eq_n_7_mod_5 :
  ∃ n : ℕ, 0 < n ∧ (7^n ≡ n^7 [MOD 5]) ∧ ∀ m : ℕ, 0 < m ∧ (7^m ≡ m^7 [MOD 5]) → n ≤ m :=
begin
  sorry -- proof omitted as requested
end

end smallest_n_7_pow_n_eq_n_7_mod_5_l812_812975


namespace find_equation_of_W_prove_perimeter_ABC_greater_than_3sqrt3_l812_812722

def locus_condition (P : ℝ × ℝ) : Prop := 
  abs P.snd = real.sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2)

def equation_W (x y : ℝ) : Prop := 
  y = x^2 + 1/4

theorem find_equation_of_W :
  ∀ (P : ℝ × ℝ), locus_condition P → equation_W P.fst P.snd :=
by 
  sorry

def three_vertices_on_W (A B C : ℝ × ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), A = (x₁, x₁^2 + 1/4) ∧ B = (x₂, x₂^2 + 1/4) ∧ C = (x₃, x₃^2 + 1/4)

def is_rectangle (A B C D : ℝ × ℝ) : Prop := 
  three_vertices_on_W A B C ∧ 
  ∃ Dx Dy, D = (Dx, Dy) ∧ 
  -- some conditions that ensure ABCD forms a rectangle

def perimeter (A B C D : ℝ × ℝ) : ℝ := 
  dist A B + dist B C + dist C D + dist D A

theorem prove_perimeter_ABC_greater_than_3sqrt3 :
  ∀ (A B C D : ℝ × ℝ), is_rectangle A B C D → three_vertices_on_W A B C → perimeter A B C D > 3 * real.sqrt 3 :=
by 
  sorry

end find_equation_of_W_prove_perimeter_ABC_greater_than_3sqrt3_l812_812722


namespace largest_multiple_of_9_less_than_100_l812_812276

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ ∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n :=
begin
  use 99,
  split,
  { norm_num }, -- 99 < 100
  split,
  { norm_num }, -- 99 % 9 = 0
  { intros m hm,
    obtain ⟨k, rfl⟩ := nat.exists_eq_mul_right_of_dvd (dvd_of_mod_eq_zero hm.2),
    rw [mul_comm, nat.mul_lt_mul_iff_left (by norm_num : 0 < 9)],
    norm_num,
    exact nat.succ_le_iff.mpr (le_of_lt hm.1), }
end

end largest_multiple_of_9_less_than_100_l812_812276


namespace arccos_one_eq_zero_l812_812450

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812450


namespace arccos_one_eq_zero_l812_812497

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812497


namespace arccos_one_eq_zero_l812_812490

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812490


namespace arccos_one_eq_zero_l812_812448

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812448


namespace mode_of_data_set_is_60_l812_812853

theorem mode_of_data_set_is_60
  (data : List ℕ := [65, 60, 75, 60, 80])
  (mode : ℕ := 60) :
  mode = 60 ∧ (∀ x ∈ data, data.count x ≤ data.count 60) :=
by {
  sorry
}

end mode_of_data_set_is_60_l812_812853


namespace pencils_left_proof_l812_812842

-- Given conditions
constant total_pencils : ℕ := 42
constant number_of_students : ℕ := 12

-- Calculation to find out the number of pencils left
noncomputable def pencils_per_student := (total_pencils / number_of_students : ℝ).floor
noncomputable def pencils_given_out := pencils_per_student * number_of_students
noncomputable def pencils_left := total_pencils - pencils_given_out

-- Problem statement to prove
theorem pencils_left_proof : pencils_left = 6 := 
sorry

end pencils_left_proof_l812_812842


namespace find_angle_B_l812_812750

-- Definitions for points and triangles
section Geometry
variables (P Q A B C : Type) [plane_geometry P Q A B C]
variables [midpoint P (A, B)] [midpoint Q (B, C)]
variables [angle A_deg 30] [angle PQC_deg 110]

noncomputable def angle_B : ℝ := 80

theorem find_angle_B :
  angle B = 80 :=
sorry
end Geometry

end find_angle_B_l812_812750


namespace minimize_f_l812_812608

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∃ x, f x = f 3 ∧ ∀ y, f x ≤ f y :=
by
  let x := 3
  use x
  split
  · sorry -- Proof that f(3) is the value of the function at x=3.
  · intro y
    sorry -- Proof that f(3) ≤ f(y) for all y.

end minimize_f_l812_812608


namespace greatest_four_digit_divisible_by_6_l812_812865

-- Define a variable to represent a four-digit number
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a variable to represent divisibility by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define a variable to represent divisibility by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- State the theorem to prove that 9996 is the greatest four-digit number divisible by 6
theorem greatest_four_digit_divisible_by_6 : 
  (∀ n : ℕ, is_four_digit_number n → divisible_by_6 n → n ≤ 9996) ∧ (is_four_digit_number 9996 ∧ divisible_by_6 9996) :=
by
  -- Insert the proof here
  sorry

end greatest_four_digit_divisible_by_6_l812_812865


namespace convex_polygon_area_achievable_l812_812920

def sweet_hexagon_area : ℝ := 1

def total_hexagons : ℕ := 2000000

def target_area : ℝ := 1900000

theorem convex_polygon_area_achievable :
  ∃ (hexagons : ℕ), hexagons ≤ total_hexagons ∧ 
                   (∃ (arrangement : ℕ), arrangement * sweet_hexagon_area ≥ target_area) ∧
                   is_convex (hexagon_union hexagons) :=
sorry

end convex_polygon_area_achievable_l812_812920


namespace largest_multiple_of_9_less_than_100_l812_812290

theorem largest_multiple_of_9_less_than_100 : ∃ x : ℕ, 9 * x < 100 ∧ ∀ y : ℕ, 9 * y < 100 → y ≤ x :=
by
  exists 11
  split
  · linarith
  · intro y h
    have : 9 * y < 100 := h
    calc
      y ≤ floor (11.11) : by linarith

end largest_multiple_of_9_less_than_100_l812_812290


namespace lydia_current_age_l812_812131

def years_for_apple_tree_to_bear_fruit : ℕ := 7
def lydia_age_when_planted_tree : ℕ := 4
def lydia_age_when_eats_apple : ℕ := 11

theorem lydia_current_age 
  (h : lydia_age_when_eats_apple - lydia_age_when_planted_tree = years_for_apple_tree_to_bear_fruit) :
  lydia_age_when_eats_apple = 11 := 
by
  sorry

end lydia_current_age_l812_812131


namespace unique_weights_count_l812_812676

-- Define the weights
def weight1 : ℕ := 2
def weight2 : ℕ := 5
def weight3 : ℕ := 18

-- The maximum weight possible
def max_weight : ℕ := weight1 + weight2 + weight3

-- The main statement of the problem in Lean
theorem unique_weights_count : 
    ∃ (measurable_weights : set ℕ), measurable_weights = { w | w ≤ max_weight } ∧ measurable_weights.size = max_weight :=
by
  sorry

end unique_weights_count_l812_812676


namespace sqrt_9_fact_over_126_eq_12_sqrt_10_l812_812010

theorem sqrt_9_fact_over_126_eq_12_sqrt_10 :
  real.sqrt (9! / 126) = 12 * real.sqrt 10 := 
sorry

end sqrt_9_fact_over_126_eq_12_sqrt_10_l812_812010


namespace calvin_overall_score_l812_812393

theorem calvin_overall_score :
  let test1_pct := 0.6
  let test1_total := 15
  let test2_pct := 0.85
  let test2_total := 20
  let test3_pct := 0.75
  let test3_total := 40
  let total_problems := 75

  let correct_test1 := test1_pct * test1_total
  let correct_test2 := test2_pct * test2_total
  let correct_test3 := test3_pct * test3_total
  let total_correct := correct_test1 + correct_test2 + correct_test3

  let overall_percentage := (total_correct / total_problems) * 100
  overall_percentage.round = 75 :=
sorry

end calvin_overall_score_l812_812393


namespace rung_spacing_l812_812741

-- Definitions based on conditions
def rung_length_in_inches : ℕ := 18
def climb_height_in_inches : ℕ := 50 * 12
def total_wood_in_inches : ℕ := 150 * 12

-- The problem statement in Lean 4
theorem rung_spacing :
  (total_wood_in_inches / rung_length_in_inches) = 100 → 
  (climb_height_in_inches / (100 - 1)) = 6.06 :=
by
  intros h
  -- this is the hint for the proof, skipping the actual steps
  sorry

end rung_spacing_l812_812741


namespace largest_multiple_of_9_less_than_100_l812_812262

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l812_812262


namespace cyclic_quadrilateral_from_conditions_l812_812104

open EuclideanGeometry

-- Definitions of given conditions
def circle (O : Point) (r : ℝ) := ∃ k : ℝ, Circle O k = r
def is_chord (O : Point) (A P : Point) := OnCircle O A P
def is_tangent (O : Point) (P : Point) (l : Line) := TangentToCircle O P l
def parallel (l m : Line) := Parallel l m
def intersect (l : Line) (A B : Point) := LineContainsPoints l A ∧ LineContainsPoints l B
def cyclic_quad (A B C D : Point) := CyclicQuadrilateral A B C D
  
-- Main theorem
theorem cyclic_quadrilateral_from_conditions {O P A B C D : Point} {l m : Line} :
  (circle O r) →
  (is_chord O A P) →
  (is_chord O B P) →
  (is_tangent O P l) →
  (parallel l m) →
  (intersect m A C) →
  (intersect m B D) →
  cyclic_quad A B C D :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof to be filled in
  sorry

end cyclic_quadrilateral_from_conditions_l812_812104


namespace volume_range_of_rectangular_solid_l812_812051

theorem volume_range_of_rectangular_solid
  (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + c * a) = 48)
  (h2 : 4 * (a + b + c) = 36) :
  (16 : ℝ) ≤ a * b * c ∧ a * b * c ≤ 20 :=
by sorry

end volume_range_of_rectangular_solid_l812_812051


namespace inscribed_circle_exists_l812_812893

-- Define the structure of the proof problem in Lean 4
variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables {P Q R S : A}

-- Define convex quadrilateral with perpendicular diagonals and equal opposite angles
def convex_quadrilateral (quad : set (A × A)) (AC_perp_BD : A × A → Prop) (equal_opposite_angles : A → Prop) : Prop :=
  quad = {(P, Q), (Q, R), (R, S), (S, P)} ∧
  AC_perp_BD (P, R) ∧ AC_perp_BD (Q, S) ∧
  equal_opposite_angles P ∧ equal_opposite_angles R

-- Define the problem statement
theorem inscribed_circle_exists {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (P Q R S : A) (quad : set (A × A)) (AC_perp_BD : A × A → Prop) (equal_opposite_angles : A → Prop) :
  convex_quadrilateral quad AC_perp_BD equal_opposite_angles → ∃ (circle : set A), ∀ (pt ∈ quad), pt ∈ circle :=
by
  -- Hypotheses (conditions)
  assume h1 : convex_quadrilateral quad AC_perp_BD equal_opposite_angles,
  sorry

end inscribed_circle_exists_l812_812893


namespace bertha_daughters_and_granddaughters_have_no_daughters_l812_812945

def total_daughters_and_granddaughters (daughters granddaughters : Nat) : Nat :=
daughters + granddaughters

def no_daughters (bertha_daughters bertha_granddaughters : Nat) : Nat :=
bertha_daughters + bertha_granddaughters

theorem bertha_daughters_and_granddaughters_have_no_daughters :
  (bertha_daughters : Nat) →
  (daughters_with_6_daughters : Nat) →
  (granddaughters : Nat) →
  (total_daughters_and_granddaughters bertha_daughters granddaughters = 30) →
  bertha_daughters = 6 →
  granddaughters = 6 * daughters_with_6_daughters →
  no_daughters (bertha_daughters - daughters_with_6_daughters) granddaughters = 26 :=
by
  intros bertha_daughters daughters_with_6_daughters granddaughters h_total h_bertha h_granddaughters
  sorry

end bertha_daughters_and_granddaughters_have_no_daughters_l812_812945


namespace arithmetic_seq_common_diff_l812_812117

theorem arithmetic_seq_common_diff (a b : ℕ) (d : ℕ) (a1 a2 a8 a9 : ℕ) 
  (h1 : a1 + a8 = 10)
  (h2 : a2 + a9 = 18)
  (h3 : a2 = a1 + d)
  (h4 : a8 = a1 + 7 * d)
  (h5 : a9 = a1 + 8 * d)
  : d = 4 :=
by
  sorry

end arithmetic_seq_common_diff_l812_812117


namespace arccos_one_eq_zero_l812_812481

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812481


namespace arccos_one_eq_zero_l812_812458

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812458


namespace arccos_one_eq_zero_l812_812408

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812408


namespace largest_multiple_of_9_less_than_100_l812_812303

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ (∀ k, k < 100 → k % 9 = 0 → k ≤ n) :=
by {
  use 99,
  split,
  { exact lt_of_le_of_ne (nat.le_of_dvd (by norm_num) (by norm_num)) (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by use 11) },
  { intros k hk hkm, apply nat.mul_le_mul_left 9 (le_of_lt $ nat.div_lt_self hk $ by norm_num), exact le_of_lt_succ (nat.succ_le_of_lt (nat.div_lt_self hk $ by norm_num)) }
  sorry
}

end largest_multiple_of_9_less_than_100_l812_812303


namespace family_trip_eggs_l812_812910

theorem family_trip_eggs (adults girls boys : ℕ)
  (eggs_per_adult : ℕ) (eggs_per_girl : ℕ) (extra_eggs_for_boy : ℕ) :
  adults = 3 →
  eggs_per_adult = 3 →
  girls = 7 →
  eggs_per_girl = 1 →
  boys = 10 →
  extra_eggs_for_boy = 1 →
  (adults * eggs_per_adult + girls * eggs_per_girl + boys * (eggs_per_girl + extra_eggs_for_boy)) = 36 :=
by
  intros
  sorry

end family_trip_eggs_l812_812910


namespace largest_multiple_of_9_less_than_100_l812_812293

theorem largest_multiple_of_9_less_than_100 : ∃ n, n < 100 ∧ n % 9 = 0 ∧ (∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n) :=
begin
  use 99,
  split,
  -- Prove the number is less than 100
  { linarith, },
  split,
  -- Prove the number is a multiple of 9
  { norm_num, },
  -- Prove it is the largest
  { intros m hlt hdiv,
    linarith [hlt, hdiv],
    norm_num at hdiv,
    cases hdiv,
    exact hdiv_left, },
end

end largest_multiple_of_9_less_than_100_l812_812293


namespace arccos_one_eq_zero_l812_812492

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812492


namespace min_soda_packs_90_l812_812809

def soda_packs (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), 6 * x + 12 * y + 24 * z = n

theorem min_soda_packs_90 : (x y z : ℕ) → soda_packs 90 → x + y + z = 5 := by
  sorry

end min_soda_packs_90_l812_812809


namespace max_distance_ratio_l812_812622

-- Define the given parabola y^2 = 4x, focus F, and points A, B on the parabola, such that ∠AFB = 90°
variables {F A B M M1 : ℝ → ℝ} -- F, A, B, M, M1 are points on the curve
noncomputable def parabola (x y : ℝ) := y^2 = 4 * x
def is_focus (F : ℝ → ℝ) := F 0 0
def is_on_parabola (A B : ℝ → ℝ) := parabola (A 0) (A 1) ∧ parabola (B 0) (B 1)
def right_angle (F A B : ℝ → ℝ) := ∠ A F B = 90°
def midpoint (A B M : ℝ → ℝ) := M 0 = (A 0 + B 0) / 2 ∧ M 1 = (A 1 + B 1) / 2
def projection (M M1 : ℝ → ℝ) := M1 0 = (M 0 + F 0) / 2 ∧ M1 1 = F 1

-- Proving the maximum value of |MM1|/|AB| given the conditions above
theorem max_distance_ratio : 
  let a := |F A| 
  let b := |F B|
  let AQ := |A Q|
  let BP := |B P|
  |M M1|/|A B| ≤ (sqrt 2)/2 := 
    begin
      sorry -- proof to be filled in
    end

end max_distance_ratio_l812_812622


namespace num_solutions_l812_812157

def f (n : ℕ) (x : ℝ) : ℝ := sin x ^ n + cos x ^ n

theorem num_solutions (h : (8 * f 4 x - 6 * f 6 x + f 8 x = 3 * f 2 x)) :
  ∃ a : ℝ, x ∈ set.Icc 0 (2 * real.pi) → a = 9 := 
by
  sorry

end num_solutions_l812_812157


namespace average_parking_cost_l812_812826

def parking_cost_per_hour (first_two_hours_cost : ℝ) (additional_hour_cost : ℝ) (total_hours : ℕ) : ℝ :=
  let additional_hours := total_hours - 2
  let total_cost := first_two_hours_cost + additional_hours * additional_hour_cost
  total_cost / total_hours

theorem average_parking_cost :
  parking_cost_per_hour 20 1.75 9 = 3.58 :=
by
  sorry

end average_parking_cost_l812_812826


namespace forty_third_digit_of_one_thirteen_l812_812863

theorem forty_third_digit_of_one_thirteen :
  let repeating_pattern := [0, 7, 6, 9, 2, 3] in
  (1 / 13).decimal_expansion.part 43 = repeating_pattern.head :=
by
  sorry

end forty_third_digit_of_one_thirteen_l812_812863


namespace arccos_one_eq_zero_l812_812514

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l812_812514


namespace find_g_degree_l812_812203

noncomputable def f (g : Polynomial ℤ) : Polynomial ℤ := sorry  -- Degree 3 polynomial

def h (f : Polynomial ℤ → Polynomial ℤ) (g : Polynomial ℤ) : Polynomial ℤ := f(g) + g^2

theorem find_g_degree (g : Polynomial ℤ) :
  (∃ f : Polynomial ℤ → Polynomial ℤ, 
    (∀ x : Polynomial ℤ, degree (f x) = 3 * degree x) ∧ 
    degree (h f g) = 8) → 
  degree g = 4 :=
by
  intros
  sorry

end find_g_degree_l812_812203


namespace arccos_one_eq_zero_l812_812407

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812407


namespace money_equation_l812_812587

variables (a b: ℝ)

theorem money_equation (h1: 8 * a + b > 160) (h2: 4 * a + b = 120) : a > 10 ∧ ∀ (a1 a2 : ℝ), a1 > a2 → b = 120 - 4 * a → b = 120 - 4 * a1 ∧ 120 - 4 * a1 < 120 - 4 * a2 :=
by 
  sorry

end money_equation_l812_812587


namespace average_speed_over_ride_l812_812179

theorem average_speed_over_ride :
  let speed1 := 12 -- speed in km/h
  let time1 := 5 / 60 -- time in hours
  
  let speed2 := 15 -- speed in km/h
  let time2 := 10 / 60 -- time in hours
  
  let speed3 := 18 -- speed in km/h
  let time3 := 15 / 60 -- time in hours
  
  let distance1 := speed1 * time1 -- distance for the first segment
  let distance2 := speed2 * time2 -- distance for the second segment
  let distance3 := speed3 * time3 -- distance for the third segment
  
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1 + time2 + time3
  let avg_speed := total_distance / total_time
  
  avg_speed = 16 :=
by
  sorry

end average_speed_over_ride_l812_812179


namespace arccos_one_eq_zero_l812_812491

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812491


namespace largest_multiple_of_9_less_than_100_l812_812300

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ (∀ k, k < 100 → k % 9 = 0 → k ≤ n) :=
by {
  use 99,
  split,
  { exact lt_of_le_of_ne (nat.le_of_dvd (by norm_num) (by norm_num)) (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by use 11) },
  { intros k hk hkm, apply nat.mul_le_mul_left 9 (le_of_lt $ nat.div_lt_self hk $ by norm_num), exact le_of_lt_succ (nat.succ_le_of_lt (nat.div_lt_self hk $ by norm_num)) }
  sorry
}

end largest_multiple_of_9_less_than_100_l812_812300


namespace volume_of_max_area_rect_prism_l812_812311

noncomputable def side_length_of_square_base (P: ℕ) : ℕ := P / 4

noncomputable def area_of_square_base (side: ℕ) : ℕ := side * side

noncomputable def volume_of_rectangular_prism (base_area: ℕ) (height: ℕ) : ℕ := base_area * height

theorem volume_of_max_area_rect_prism
  (P : ℕ) (hP : P = 32) 
  (H : ℕ) (hH : H = 9) 
  : volume_of_rectangular_prism (area_of_square_base (side_length_of_square_base P)) H = 576 := 
by
  sorry

end volume_of_max_area_rect_prism_l812_812311


namespace angle_T_is_120_l812_812933

-- Define the conditions related to the convex pentagon
variables (P Q R S T : Type)

-- Define equal sides and specific angles
variables (PQ QR RS ST TP : ℝ)
variables (angle_P angle_Q angle_T: ℝ)

-- Let the conditions be exactly as given in the mathematical problem
def conditions : Prop :=
  PQ = QR ∧ QR = RS ∧ RS = ST ∧ ST = TP ∧ TP = PQ ∧
  angle_P = 120 ∧ angle_Q = 120

-- The theorem to prove
theorem angle_T_is_120 (h : conditions P Q R S T PQ QR RS ST TP angle_P angle_Q angle_T)
: angle_T = 120 :=
sorry

end angle_T_is_120_l812_812933


namespace largest_multiple_of_9_less_than_100_l812_812304

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ (∀ k, k < 100 → k % 9 = 0 → k ≤ n) :=
by {
  use 99,
  split,
  { exact lt_of_le_of_ne (nat.le_of_dvd (by norm_num) (by norm_num)) (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by use 11) },
  { intros k hk hkm, apply nat.mul_le_mul_left 9 (le_of_lt $ nat.div_lt_self hk $ by norm_num), exact le_of_lt_succ (nat.succ_le_of_lt (nat.div_lt_self hk $ by norm_num)) }
  sorry
}

end largest_multiple_of_9_less_than_100_l812_812304


namespace equation_of_locus_perimeter_greater_than_3sqrt3_l812_812726

theorem equation_of_locus 
  (P : ℝ × ℝ)
  (hx : ∀ y : ℝ, P.snd = |P.snd| → |P.snd| = sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2))
  :
  (P.snd = P.fst ^ 2 + 1/4) := 
sorry

theorem perimeter_greater_than_3sqrt3
  (A B C D : ℝ × ℝ)
  (h_parabola : ∀ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4)
  (h_vert_A : A ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_vert_B : B ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_vert_C : C ∈ set_of (λ P : ℝ × ℝ, P.snd = P.fst ^ 2 + 1/4))
  (h_perpendicular: ((B.fst - A.fst) * (C.fst - B.fst) + (B.snd - A.snd) * (C.snd - B.snd)) = 0) 
  :
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
sorry

end equation_of_locus_perimeter_greater_than_3sqrt3_l812_812726


namespace siblings_count_l812_812133

noncomputable def Masud_siblings (M : ℕ) : Prop :=
  (4 * M - 60 = (3 * M) / 4 + 135) → M = 60

theorem siblings_count (M : ℕ) : Masud_siblings M :=
  by
  sorry

end siblings_count_l812_812133


namespace find_n_l812_812688

theorem find_n :
  ∀ (n : ℕ),
    2^200 * 2^203 + 2^163 * 2^241 + 2^126 * 2^277 = 32^n →
    n = 81 :=
by
  intros n h
  sorry

end find_n_l812_812688


namespace square_vertex_possible_pairs_l812_812770

noncomputable def gcd (a b : ℕ) : ℕ := a.gcd b
noncomputable def lcm (a b : ℕ) : ℕ := a.lcm b

theorem square_vertex_possible_pairs (x y : ℕ) (h1 : gcd x y = 2) 
    (h2 : 2 * (x^2 + y^2) = 10 * lcm x y) : (x = 4 ∧ y = 2) ∨ (x = 2 ∧ y = 4) :=
by
  sorry

#eval square_vertex_possible_pairs 4 2 (by norm_num) (by norm_num)
#eval square_vertex_possible_pairs 2 4 (by norm_num) (by norm_num)

end square_vertex_possible_pairs_l812_812770


namespace total_tennis_balls_used_l812_812374

-- Definitions based on given conditions
def number_of_games_in_round1 := 8
def number_of_games_in_round2 := 4
def number_of_games_in_round3 := 2
def number_of_games_in_round4 := 1

def number_of_cans_per_game := 5
def number_of_balls_per_can := 3

-- Lean statement asserting the total number of tennis balls used
theorem total_tennis_balls_used :
  number_of_games_in_round1 + number_of_games_in_round2 + number_of_games_in_round3 + number_of_games_in_round4 =
  15 → 
  (15 * number_of_cans_per_game * number_of_balls_per_can) = 225 :=
by
  intro h
  rw [h]
  norm_num
  sorry

end total_tennis_balls_used_l812_812374


namespace arccos_one_eq_zero_l812_812439

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812439


namespace arccos_one_eq_zero_l812_812477

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l812_812477


namespace ram_shyam_weight_ratio_l812_812243

theorem ram_shyam_weight_ratio
    (R S : ℝ)
    (h1 : 1.10 * R + 1.22 * S = 82.8)
    (h2 : R + S = 72) :
    R / S = 7 / 5 :=
by sorry

end ram_shyam_weight_ratio_l812_812243


namespace arccos_1_eq_0_l812_812414

theorem arccos_1_eq_0 : real.arccos 1 = 0 := 
by 
  sorry

end arccos_1_eq_0_l812_812414


namespace value_of_a_10_l812_812645

variables {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}
variable (a_1 : ℕ)
variable {n : ℕ}

noncomputable def arithmetic_seq (n : ℕ) : ℕ := a_1 + (n - 1) * d

axiom h1 : (arithmetic_seq 3 + arithmetic_seq 5 = 26)
axiom h2 : (S_n 4 = 28)
axiom sum_formula : S_n n = n * a_1 + n * (n - 1) / 2 * d

theorem value_of_a_10 : (arithmetic_seq 10 = 37) :=
by {
  sorry
}

end value_of_a_10_l812_812645


namespace July_husband_age_l812_812870

namespace AgeProof

variable (HannahAge JulyAge HusbandAge : ℕ)

def double_age_condition (hannah_age : ℕ) (july_age : ℕ) : Prop :=
  hannah_age = 2 * july_age

def twenty_years_later (current_age : ℕ) : ℕ :=
  current_age + 20

def two_years_older (age : ℕ) : ℕ :=
  age + 2

theorem July_husband_age :
  ∃ (hannah_age july_age : ℕ), double_age_condition hannah_age july_age ∧
    twenty_years_later hannah_age = 26 ∧
    twenty_years_later july_age = 23 ∧
    two_years_older (twenty_years_later july_age) = 25 :=
by
  sorry
end AgeProof

end July_husband_age_l812_812870


namespace first_component_worst_quality_l812_812361

theorem first_component_worst_quality :
  let deviation1 := 0.16
  let deviation2 := -0.12
  let deviation3 := -0.15
  let deviation4 := 0.11
  abs deviation1 > abs deviation2 ∧ abs deviation1 > abs deviation3 ∧ abs deviation1 > abs deviation4 :=
by {
  let deviation1 := 0.16
  let deviation2 := -0.12
  let deviation3 := -0.15
  let deviation4 := 0.11
  show abs deviation1 > abs deviation2 ∧ abs deviation1 > abs deviation3 ∧ abs deviation1 > abs deviation4,
  sorry
}

end first_component_worst_quality_l812_812361


namespace non_crossing_segments_exists_l812_812106

theorem non_crossing_segments_exists (points: Finset (ℝ × ℝ)) (blue_pts red_pts: Finset (ℝ × ℝ))
  (h_points_nocollinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points → (p1.1 - p2.1) * (p2.2 - p3.2) ≠ (p2.1 - p3.1) * (p1.2 - p2.2))
  (h_card_points : points.card = 2022)
  (h_card_blue : blue_pts.card = 1011)
  (h_card_red : red_pts.card = 1011)
  (h_disjoint : Disjoint blue_pts red_pts)
  (h_union : blue_pts ∪ red_pts = points) :
  ∃ segments : Finset (ℝ × ℝ) × (ℝ × ℝ), (Finset.card segments = 1011 ∧ ∀ seg ∈ segments, seg.1 ∈ blue_pts ∧ seg.2 ∈ red_pts ∧ ¬∃ seg', seg' ∈ segments ∧ seg ≠ seg' ∧ seg.1 = seg'.1 ∧ seg.2 = seg'.2 ∧ intersect seg seg') := sorry

end non_crossing_segments_exists_l812_812106


namespace equation_of_W_rectangle_perimeter_gt_3sqrt3_l812_812733

namespace MathProof

/- Defining the conditions and the questions -/
def point (P : ℝ × ℝ) := P.fst
def P (x y : ℝ) := (x, y)

def dist_x_axis (y : ℝ) := |y|
def dist_to_point (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + (y - 1/2)^2)

-- Condition: The distance from point P to the x-axis equals distance to (0, 1/2)
def condition_P (x y : ℝ) : Prop :=
  dist_x_axis y = dist_to_point x y

-- Locus W
def W_eq (x : ℝ) : ℝ := x^2 + 1/4

/- Proving the statements -/
theorem equation_of_W (x y : ℝ) (h : condition_P x y): 
  y = W_eq x :=
sorry

-- Perimeter of rectangle ABCD
structure rectangle :=
(A : ℝ × ℝ)
(B : ℝ × ℝ)
(C : ℝ × ℝ)
(D : ℝ × ℝ)

def on_W (P : ℝ × ℝ) : Prop :=
  P.snd = W_eq P.fst

def perimeter (rect : rectangle) : ℝ :=
  let d1 := Real.sqrt ((rect.A.fst - rect.B.fst)^2 + (rect.A.snd - rect.B.snd)^2)
  let d2 := Real.sqrt ((rect.B.fst - rect.C.fst)^2 + (rect.B.snd - rect.C.snd)^2)
  let d3 := Real.sqrt ((rect.C.fst - rect.D.fst)^2 + (rect.C.snd - rect.D.snd)^2)
  let d4 := Real.sqrt ((rect.D.fst - rect.A.fst)^2 + (rect.D.snd - rect.A.snd)^2)
  d1 + d2 + d3 + d4

theorem rectangle_perimeter_gt_3sqrt3 
(rect : rectangle) 
(hA : on_W rect.A) 
(hB : on_W rect.B) 
(hC : on_W rect.C) :
  perimeter rect > 3 * Real.sqrt 3 :=
sorry

end MathProof

end equation_of_W_rectangle_perimeter_gt_3sqrt3_l812_812733


namespace period_of_y_l812_812866

noncomputable def y (x : ℝ) : ℝ := (Real.tan x + Real.cot x) ^ 2

theorem period_of_y : ∀ x, y x = y (x + Real.pi) :=
by
  intro x
  sorry

end period_of_y_l812_812866


namespace smallest_integer_1323_divisors_l812_812965

theorem smallest_integer_1323_divisors (m k : ℕ) (h1 : ¬ 30 ∣ m) (h2 : (m * 30^k).num_divisors = 1323) :
  m + k = 83 :=
sorry

end smallest_integer_1323_divisors_l812_812965


namespace area_triang_CDM_l812_812739

-- Given definitions:
variables (A B C M D : Type)
variables (AC BC AB AM AD BD DM MN : ℝ)
variables (m n p : ℕ)

-- Conditions
def condition1 : AC = 8 := by sorry
def condition2 : BC = 15 := by sorry
def condition3 : ∠C = 90 := by sorry
def condition4 : AM = 1 / 2 * AB := by sorry
def condition5 : AD = 17 := by sorry
def condition6 : BD = 17 := by sorry
def condition7 : D on same side of AB as C := by sorry

-- Question
theorem area_triang_CDM 
  (AC BC AB AM AD BD DM MN : ℝ)
  (AC_eq : AC = 8)
  (BC_eq : BC = 15)
  (C_angle : ∠C = 90)
  (midpoint_AM : AM = 1 / 2 * AB)
  (AD_eq : AD = 17)
  (BD_eq : BD = 17)
  (D_same_side : D on same side of AB as C)
  : ∃ (m n p : ℕ), 
    (m * p) = 9 ∧ 
    (n = 29) ∧ 
    (m.gcd p = 1) ∧ 
    (areaCDM : ℝ)
    (areaCDM = (m * real.sqrt n) / p :=
begin 
  -- Proof to be inserted
  sorry
end

end area_triang_CDM_l812_812739


namespace intersect_circle_line_l812_812096

theorem intersect_circle_line (k m : ℝ) : 
  (∃ (x y : ℝ), y = k * x + 2 * k ∧ x^2 + y^2 + m * x + 4 = 0) → m > 4 :=
by
  -- This statement follows from the conditions given in the problem
  -- You can use implicit for pure documentation
  -- We include a sorry here to skip the proof
  sorry

end intersect_circle_line_l812_812096


namespace sum_of_first_n_odd_integers_l812_812330

theorem sum_of_first_n_odd_integers (n : ℕ) (h : (∑ i in finset.range n, 2 * i + 1) = 169) : n = 13 :=
sorry

end sum_of_first_n_odd_integers_l812_812330


namespace sacks_per_day_l812_812851

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (h1 : total_sacks = 56) (h2 : days = 14) : (total_sacks / days) = 4 := 
by
  rw [h1, h2]
  norm_num

end sacks_per_day_l812_812851


namespace relationship_between_k_and_a_l812_812223

theorem relationship_between_k_and_a (a k : ℝ) (h_a : 0 < a ∧ a < 1) :
  (k^2 + 1) * a^2 ≥ 1 :=
sorry

end relationship_between_k_and_a_l812_812223


namespace arccos_one_eq_zero_l812_812396

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812396


namespace infinite_set_exists_l812_812620

noncomputable def f (x : ℕ) : ℕ := sorry

theorem infinite_set_exists :
  (∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) →
  ∃ (M : set ℕ), set.infinite M ∧ ∀ i j k ∈ M, (i - j) * f k + (j - k) * f i + (k - i) * f j = 0 :=
begin
  sorry
end

end infinite_set_exists_l812_812620


namespace smallest_to_large_area_ratio_l812_812586

noncomputable def ratio_of_areas (L : ℝ) : ℝ :=
  let large_square_area := L^2
  let mid_square_side := L / 2
  let smallest_square_side := mid_square_side / 2
  let smallest_square_area := smallest_square_side^2
  smallest_square_area / large_square_area

theorem smallest_to_large_area_ratio (L : ℝ) (hL : 0 < L) : ratio_of_areas L = 1 / 16 :=
by
  have large_square_area := L^2
  have mid_square_side := L / 2
  have smallest_square_side := mid_square_side / 2
  have smallest_square_area := smallest_square_side^2
  have ratio := smallest_square_area / large_square_area
  calc
    ratio_of_areas L
        = smallest_square_area / large_square_area : rfl
    ... = (mid_square_side / 2)^2 / L^2 : by congr; field_simp; ring
    ... = ( (L / 2) / 2 )^2 / L^2 : rfl
    ... = ( (L / 4)^2 ) / L^2 : rfl
    ... = ( L^2 / 16 ) / L^2 : by congr
    ... = 1 / 16 : by field_simp; ring

end smallest_to_large_area_ratio_l812_812586


namespace probability_of_different_suits_l812_812781

-- Let’s define the parameters of the problem
def total_cards : ℕ := 104
def first_card_remaining : ℕ := 103
def same_suit_cards : ℕ := 26
def different_suit_cards : ℕ := first_card_remaining - same_suit_cards

-- The probability that the two cards drawn are of different suits
def probability_different_suits : ℚ := different_suit_cards / first_card_remaining

-- The main statement to prove
theorem probability_of_different_suits :
  probability_different_suits = 78 / 103 :=
by {
  -- The proof would go here
  sorry
}

end probability_of_different_suits_l812_812781


namespace exists_mutual_acquaintances_l812_812639

structure School :=
  (students : Set ℕ)
  (knows : ℕ → Set ℕ)
  (knows_condition : ∀ s, knows s ⊆ students)
  (knows_count : ∀ s, (knows s).card = n + 1)

variable (n : ℕ)
variable (school1 school2 school3 : School)
variable (disjoint_schools : (school1.students ∩ school2.students = ∅) ∧ 
                             (school1.students ∩ school3.students = ∅) ∧ 
                             (school2.students ∩ school3.students = ∅))

theorem exists_mutual_acquaintances :
  ∃ (a b c : ℕ), a ∈ school1.students ∧ 
                 b ∈ school2.students ∧ 
                 c ∈ school3.students ∧ 
                 b ∈ school1.knows a ∧ 
                 c ∈ school1.knows a ∧
                 a ∈ school2.knows b ∧ 
                 c ∈ school2.knows b ∧
                 a ∈ school3.knows c ∧ 
                 b ∈ school3.knows c :=
sorry

end exists_mutual_acquaintances_l812_812639


namespace math_problem_l812_812342

theorem math_problem : 1012^2 - 992^2 - 1009^2 + 995^2 = 12024 := sorry

end math_problem_l812_812342


namespace largest_multiple_of_9_less_than_100_l812_812297

theorem largest_multiple_of_9_less_than_100 : ∃ n, n < 100 ∧ n % 9 = 0 ∧ (∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n) :=
begin
  use 99,
  split,
  -- Prove the number is less than 100
  { linarith, },
  split,
  -- Prove the number is a multiple of 9
  { norm_num, },
  -- Prove it is the largest
  { intros m hlt hdiv,
    linarith [hlt, hdiv],
    norm_num at hdiv,
    cases hdiv,
    exact hdiv_left, },
end

end largest_multiple_of_9_less_than_100_l812_812297


namespace probability_reaches_3_3_in_8_or_fewer_steps_l812_812817

theorem probability_reaches_3_3_in_8_or_fewer_steps :
  let q := (75 : ℕ) / 1024 in
  ∃ a b : ℕ, Nat.gcd a b = 1 ∧ q = a / b ∧ a + b = 1099 :=
by
  -- Introduce the definitions from the conditions
  let steps := 8
  let destination := (3, 3)

  -- Mathematical properties and probability calculations 
  let q := 75 / 1024
  
  -- Correct answer based on solution steps
  have q_is_correct : q = 75 / 1024 := rfl
  
  -- Define a and b based on simplified probability fraction
  let a := 75
  let b := 1024
  
  -- Check that they are relatively prime
  have a_b_rel_prime : Nat.gcd a b = 1 := by sorry
  
  -- Summation check
  have sum_a_b : a + b = 1099 := by sorry
  
  -- Existential statement
  exists a, b
  exact ⟨a_b_rel_prime, q_is_correct, sum_a_b⟩


end probability_reaches_3_3_in_8_or_fewer_steps_l812_812817


namespace arccos_one_eq_zero_l812_812443

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l812_812443


namespace set_inter_union_l812_812755

variable {α : Type*} [DecidableEq α]

def A : set α := {3, 5, 6, 8}
def B : set α := {4, 5, 7, 8}

theorem set_inter_union :
  A ∩ B = {5, 8} ∧ A ∪ B = {3, 4, 5, 6, 7, 8} :=
by
  -- proof goes here
  sorry

end set_inter_union_l812_812755


namespace max_phi_l812_812061

noncomputable def f (ω x φ : ℝ) : ℝ := Real.tan (ω * x - φ)

open Real

theorem max_phi (ω φ : ℝ) (a : ℝ) :
  (0 < ω) →
  (0 < φ ∧ φ < π) →
  (f ω (π / (3 * ω)) φ = a) →
  ((f 3 (π / 3 - φ) φ = f 3 (π / 3 + φ) φ) →
  (∀ k : ℤ, φ = (π / 4) - (k * (π / 2))) →
  (φ ≤ 3 * (π / 4))) ∧ (φ = 3 * (π / 4)) :=
begin
  intros hω hφ hmin hsymm hsum,
  sorry
end

end max_phi_l812_812061


namespace remainder_500th_T_div_500_l812_812758

-- Define the sequence T
def T : ℕ → ℕ :=
λ n, sorry -- definition for translating n-th element with 6 ones in binary

-- The 500th element in the sequence T
def M : ℕ := T 500

-- The statement to prove
theorem remainder_500th_T_div_500 :
  M % 500 = 24 :=
sorry

end remainder_500th_T_div_500_l812_812758


namespace arccos_one_eq_zero_l812_812432

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l812_812432


namespace fruit_seller_profit_percentage_l812_812355

/-- Suppose a fruit seller sells mangoes at the rate of Rs. 12 per kg and incurs a loss of 15%. 
    The mangoes should have been sold at Rs. 14.823529411764707 per kg to make a specific profit percentage. 
    This statement proves that the profit percentage is 5%. 
-/
theorem fruit_seller_profit_percentage :
  ∃ P : ℝ, 
    (∀ (CP SP : ℝ), 
        SP = 14.823529411764707 ∧ CP = 12 / 0.85 → 
        SP = CP * (1 + P / 100)) → 
    P = 5 := 
sorry

end fruit_seller_profit_percentage_l812_812355


namespace ara_height_l812_812377

/-
Conditions:
1. Shea's height increased by 25%.
2. Shea is now 65 inches tall.
3. Ara grew by three-quarters as many inches as Shea did.

Prove Ara's height is 61.75 inches.
-/

def shea_original_height (x : ℝ) : Prop := 1.25 * x = 65

def ara_growth (growth : ℝ) (shea_growth : ℝ) : Prop := growth = (3 / 4) * shea_growth

def shea_growth (original_height : ℝ) : ℝ := 0.25 * original_height

theorem ara_height (shea_orig_height : ℝ) (shea_now_height : ℝ) (ara_growth_inches : ℝ) :
  shea_original_height shea_orig_height → 
  shea_now_height = 65 →
  ara_growth ara_growth_inches (shea_now_height - shea_orig_height) →
  shea_orig_height + ara_growth_inches = 61.75 :=
by
  sorry

end ara_height_l812_812377


namespace arccos_one_eq_zero_l812_812539

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  have hcos : Real.cos 0 = 1 := by sorry   -- Known fact about cosine
  have hrange : 0 ∈ set.Icc (0:ℝ) (π:ℝ) := by sorry  -- Principal value range for arccos
  sorry

end arccos_one_eq_zero_l812_812539


namespace number_of_valid_n_l812_812569

theorem number_of_valid_n : 
  (∃ n : ℕ, 0 < n ∧ n < 42 ∧ (∃ k : ℕ, k > 0 ∧ n = 42 * k / (k + 1))) →
  ((finset.univ.filter (λ n : ℕ, 0 < n ∧ n < 42 ∧ (∃ k : ℕ, k > 0 ∧ n = 42 * k / (k + 1)))).card = 7) :=
by
  sorry

end number_of_valid_n_l812_812569


namespace total_tennis_balls_used_l812_812375

-- Definitions based on given conditions
def number_of_games_in_round1 := 8
def number_of_games_in_round2 := 4
def number_of_games_in_round3 := 2
def number_of_games_in_round4 := 1

def number_of_cans_per_game := 5
def number_of_balls_per_can := 3

-- Lean statement asserting the total number of tennis balls used
theorem total_tennis_balls_used :
  number_of_games_in_round1 + number_of_games_in_round2 + number_of_games_in_round3 + number_of_games_in_round4 =
  15 → 
  (15 * number_of_cans_per_game * number_of_balls_per_can) = 225 :=
by
  intro h
  rw [h]
  norm_num
  sorry

end total_tennis_balls_used_l812_812375
