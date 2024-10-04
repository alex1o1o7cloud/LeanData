import Mathlib

namespace count_correct_expressions_l378_378478

theorem count_correct_expressions :
  (¬ (0 = ∅) ∧
   ∅ ∈ {∅} ∧
   0 ∈ {0} ∧
   ¬ (∅ ∈ {a})) → 
  (2 = 2) :=
by
  intro h
  sorry

end count_correct_expressions_l378_378478


namespace relationship_among_abc_l378_378659

noncomputable def f (x : ℝ) : ℝ := sorry

variables (a b c : ℝ)
variables (hx1 : f(1 - x) = f(1 + x) ∀ x : ℝ)
variables (hx2 : ∀ x y : ℝ, x ≤ y → y ≤ 1 → f(y) ≤ f(x)) -- Monotonically decreasing on (-∞, 1]

-- Definitions of a, b, c
def a := f (Real.log 0.5 / Real.log 4)
def b := f (Real.log 3 / Real.log (1/3))
def c := f (Real.log 9 / Real.log 3)

theorem relationship_among_abc (h : c < a ∧ a < b) : c < a ∧ a < b :=
  by
    -- Proof steps would go here
    sorry

end relationship_among_abc_l378_378659


namespace problem_l378_378681

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else 3 ^ x

theorem problem (h : f (f (1 / 9)) = 1 / 9) : True :=
begin
  rw [function.comp],
  unfold f,
  sorry,  -- proof to be added
end

end problem_l378_378681


namespace ten_gon_sum_l378_378557

theorem ten_gon_sum (a b c d : ℕ) (r : ℝ) (h_radius : r = 10) 
  (h_sum : ∀ (A : ℝ), A = 
    (200 * Real.sin (Real.pi * 18 / 180) + 
     200 * Real.sin (Real.pi * 36 / 180) + 
     200 * Real.cos (Real.pi * 36 / 180) + 
     200 * Real.cos (Real.pi * 18 / 180) + 100) 
    = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5) : 
  a + b + c + d = 225 :=
by 
  sorry

end ten_gon_sum_l378_378557


namespace derivative_is_even_then_b_eq_zero_l378_378201

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- The statement that the derivative is an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Our main theorem
theorem derivative_is_even_then_b_eq_zero : is_even (f' a b c) → b = 0 :=
by
  intro h
  have h1 := h 1
  have h2 := h (-1)
  sorry

end derivative_is_even_then_b_eq_zero_l378_378201


namespace fib_mod_5_last_digit_4_l378_378199

noncomputable def fib_mod_5 : ℕ → ℕ 
| 0       := 0
| 1       := 1
| n       := (fib_mod_5 (n - 1) + fib_mod_5 (n - 2)) % 5

theorem fib_mod_5_last_digit_4 :
  ∃ n m, n ≠ m ∧ fib_mod_5 m = 4 ∧ ∀ k, k < m → fib_mod_5 k ≠ 4 :=
begin
  sorry
end

end fib_mod_5_last_digit_4_l378_378199


namespace circumscribed_circle_diameter_eq_l378_378056

theorem circumscribed_circle_diameter_eq (r : ℝ) (h1 : (3 * (r * sqrt 3 / 2)) = (π * r^2)) :
  (2 * r) = (3 * sqrt 3 / π) :=
sorry

end circumscribed_circle_diameter_eq_l378_378056


namespace smallest_base_10_integer_exists_l378_378876

theorem smallest_base_10_integer_exists :
  ∃ (x a b : ℕ), (a > 2) ∧ (b > 2) ∧ (x = 2 * a + 1) ∧ (x = b + 2) ∧ (x = 7) :=
by
  sorry

end smallest_base_10_integer_exists_l378_378876


namespace speed_of_man_in_still_water_l378_378151

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : v_m + v_s = 6.2) (h2 : v_m - v_s = 6) : v_m = 6.1 :=
by
  sorry

end speed_of_man_in_still_water_l378_378151


namespace lottery_prize_l378_378775

theorem lottery_prize (n : ℕ) (first_ticket_price : ℕ) (profit : ℕ) : 
  n = 5 ∧ first_ticket_price = 1 ∧ profit = 4 
  → let tickets_sold := (List.range n).map (λ i, first_ticket_price + i) in
    let total_amount := tickets_sold.sum in
    let prize_money := total_amount - profit in
    prize_money = 11 :=
by
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h3,
  let tickets_sold := (List.range 5).map (λ i, 1 + i),
  let total_amount := tickets_sold.sum,
  let prize_money := total_amount - 4,
  sorry

end lottery_prize_l378_378775


namespace find_desired_ellipse_l378_378276

noncomputable def isEllipseThroughPoint (E : ℝ → ℝ → Prop) (A : ℝ × ℝ) : Prop := 
  E A.1 A.2

def ellipseStandardForm (a b : ℝ) : ℝ → ℝ → Prop := 
  λ x y, (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def commonFocusEllipse (c : ℝ) (a b : ℝ) : Prop :=
  (a^2 - b^2 = c^2)

theorem find_desired_ellipse : 
  ∃ a b : ℝ, (a^2 = 6) ∧ (b^2 = 3) ∧ 
             commonFocusEllipse (√3) a b ∧ 
             isEllipseThroughPoint (ellipseStandardForm a b) (2, 1) :=
by
  sorry

end find_desired_ellipse_l378_378276


namespace largest_possible_n_l378_378116

theorem largest_possible_n (n : ℕ) : 
  (∃ n, ∀ m, n ≥ m → (170! / 10^n : ℚ) ∈ ℤ) ∧ 
  (∀ n', (∀ m, n' ≥ m → (170! / 10^n' : ℚ) ∈ ℤ) → n' ≤ 41) :=
begin
  sorry
end

end largest_possible_n_l378_378116


namespace tiling_ways_l378_378088

-- Define the recurrence relation for the number of ways to tile the rectangle
def a : ℕ → ℕ
| 0     => 1  -- Normally not used, but we need a default start
| 1     => 3  -- Initial condition
| 2     => 5  -- Initial condition
| (n+3) => a (n+2) + 2 * a (n+1)

-- Prove the given formula for n > 3
theorem tiling_ways (n : ℕ) (h : n > 3) : 
  a n = (2^(n+2) + (-1)^(n+1)) / 3 := 
by
  sorry

end tiling_ways_l378_378088


namespace cost_of_one_dozen_pens_l378_378120

theorem cost_of_one_dozen_pens
  (x : ℝ)
  (hx : 20 * x = 150) :
  12 * 5 * (150 / 20) = 450 :=
by
  sorry

end cost_of_one_dozen_pens_l378_378120


namespace aluminium_atoms_in_compound_l378_378143

/-- Definition of given conditions -/
def molecular_weight : ℝ := 132
def chlorine_atoms : ℕ := 3
def chlorine_atomic_weight : ℝ := 35.45
def aluminium_atomic_weight : ℝ := 26.98

/-- Proof that the number of Aluminium atoms in the compound is 1 -/
theorem aluminium_atoms_in_compound : 
  let weight_of_chlorine := (chlorine_atoms:ℝ) * chlorine_atomic_weight,
      weight_of_aluminium := molecular_weight - weight_of_chlorine,
      aluminium_atoms := weight_of_aluminium / aluminium_atomic_weight in
  aluminium_atoms ≈ 1 :=
by {
  sorry
}

end aluminium_atoms_in_compound_l378_378143


namespace cube_sum_identity_l378_378897

theorem cube_sum_identity (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end cube_sum_identity_l378_378897


namespace graph_passes_through_P_l378_378052

variable (a : ℝ) (x : ℝ) (f : ℝ → ℝ)
-- Given conditions
noncomputable def f (x : ℝ) := log a (x - 4) + 2
def a_pos : Prop := a > 0
def a_ne_one : Prop := a ≠ 1
def P : ℝ × ℝ := (5, 2)

-- Statement to prove
theorem graph_passes_through_P (h1 : a_pos a) (h2 : a_ne_one a) : 
  f a 5 = 2 := sorry

end graph_passes_through_P_l378_378052


namespace sum_a_b_c_fifth_l378_378425

def a_n (n : ℕ) : ℕ := 2 * n - 1
def b_n (n : ℕ) : ℕ := 2^(n-1)
def c_n (n : ℕ) : ℕ := a_n n * b_n n

theorem sum_a_b_c_fifth : a_n 5 + b_n 5 + c_n 5 = 169 := by
  -- Local definitions
  let a := a_n 5
  let b := b_n 5
  let c := c_n 5
  -- Use exact values to show the addition satisfies the equation
  have h_a : a = 9 := by
    simp only [a_n]
    exact rfl
  have h_b : b = 16 := by
    simp only [b_n]
    exact rfl
  have h_c : c = 144 := by
    simp only [c_n, h_a, h_b]
    exact rfl
  -- Prove the final sum
  calc
    a + b + c = 9 + 16 + 144 := by simp [h_a, h_b, h_c]
    _ = 169 := by norm_num

end sum_a_b_c_fifth_l378_378425


namespace pentagon_cross_section_l378_378545

noncomputable def pentagon_angles : Prop :=
  let angles := [120, 120, 120, 120, 60]
  ∃ (a1 a2 a3 a4 a5 : ℕ), 
    [a1, a2, a3, a4, a5] = angles ∧ 
    a1 + a2 + a3 + a4 + a5 = 540

theorem pentagon_cross_section :
  ∃ (a1 a2 a3 a4 a5 : ℕ),
    [a1, a2, a3, a4, a5] = [120, 120, 120, 120, 60] ∧
    a1 + a2 + a3 + a4 + a5 = 540 :=
by {
  use [120, 120, 120, 120, 60],
  split,
  exact rfl,
  norm_num,
}

#print pentagon_cross_section

end pentagon_cross_section_l378_378545


namespace right_triangle_l378_378799

theorem right_triangle (A B C : ℝ) 
  (h1 : A + B + C = π) 
  (h2 : (sin A)^2 + (sin B)^2 + (sin C)^2 = 2 * (cos A * cos B * cos C + 1)) 
  (h3 : (sin A)^2 + (sin B)^2 + (sin C)^2 > 0)
  (h4 : (sin A)^2 + (sin B)^2 + (sin C)^2 < 3): 
  (cos A * cos B * cos C = 0) := 
by sorry

end right_triangle_l378_378799


namespace coefficient_x2_expansion_l378_378622

theorem coefficient_x2_expansion :
  let x := (1 - (1/2) * X) * (1 + 2 * X.sqrt) ^ 5 in
  (coeff x 2 = 60) :=
sorry

end coefficient_x2_expansion_l378_378622


namespace log_base_one_fourth_of_sixteen_l378_378264

theorem log_base_one_fourth_of_sixteen : log (1/4) 16 = -2 :=  sorry

end log_base_one_fourth_of_sixteen_l378_378264


namespace trigonometric_expression_l378_378311

theorem trigonometric_expression (θ : ℝ) (h : Real.tan θ = -3) :
    2 / (3 * (Real.sin θ) ^ 2 - (Real.cos θ) ^ 2) = 10 / 13 :=
by
  -- sorry to skip the proof
  sorry

end trigonometric_expression_l378_378311


namespace growingPathProduct_l378_378467

-- Define the grid and condition constraints
def isValidPoint (p : ℤ × ℤ) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2

def distance (p q : ℤ × ℤ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def isGrowingPath (path : List (ℤ × ℤ)) : Prop :=
  (∀ p ∈ path, isValidPoint p) ∧
  (∀ i j, i < j → i < path.length ∧ j < path.length → distance (path.nth! i) (path.nth! j) < distance (path.nth! i.succ) (path.nth! i.succ.succ))

-- Main problem statement
theorem growingPathProduct (m r : ℕ) (h₁ : m = 6) (h₂ : r = 4) :
    m * r = 24 := by
  rw [h₁, h₂]
  norm_num

end growingPathProduct_l378_378467


namespace operation_example_l378_378211

def operation (a b : ℤ) : ℤ := 2 * a * b - b^2

theorem operation_example : operation 1 (-3) = -15 := by
  sorry

end operation_example_l378_378211


namespace perpendicular_vectors_solve_l378_378332

noncomputable def vector_a (x : ℝ) := (x, 1 : ℝ)
noncomputable def vector_b := (0, 1 : ℝ)
noncomputable def vector_2a_b (x : ℝ) := (2 * x, 1 : ℝ)
noncomputable def vector_a_2b (x : ℝ) := (x, -1 : ℝ)

-- Define the dot product of two 2D vectors
noncomputable def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

-- The main statement
theorem perpendicular_vectors_solve (x : ℝ) (hx : 0 < x) :
  dot_product (vector_2a_b x) (vector_a_2b x) = 0 → x = Real.sqrt 2 / 2 :=
by
  sorry

end perpendicular_vectors_solve_l378_378332


namespace zeros_in_interval_l378_378324

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 - a * Real.log(x)

theorem zeros_in_interval (a : ℝ) (h: a > 0) :
  (∀ x : ℝ, 1 < x ∧ x < Real.exp 1 → f a x = 0) ↔ (a ∈ Set.Ioo (Real.exp 1) ((1 / 2) * Real.exp 2)) :=
sorry

end zeros_in_interval_l378_378324


namespace additional_men_joined_l378_378130

noncomputable def solve_problem := 
  let M := 1000
  let days_initial := 17
  let days_new := 11.333333333333334
  let total_provisions := M * days_initial
  let additional_men := (total_provisions / days_new) - M
  additional_men

theorem additional_men_joined : solve_problem = 500 := by
  sorry

end additional_men_joined_l378_378130


namespace kimberly_bought_skittles_l378_378392

-- Conditions
def initial_skittles : ℕ := 5
def total_skittles : ℕ := 12

-- Prove
theorem kimberly_bought_skittles : ∃ bought_skittles : ℕ, (total_skittles = initial_skittles + bought_skittles) ∧ bought_skittles = 7 :=
by
  sorry

end kimberly_bought_skittles_l378_378392


namespace pentagonal_pyramid_edges_faces_l378_378153

-- Define a structure for a pentagonal pyramid
structure PentagonalPyramid :=
(base_edges : ℕ)  -- Number of edges on the base
(lateral_edges : ℕ)  -- Number of edges connecting the base to the apex
(triangular_faces : ℕ)  -- Number of triangular faces
(base_faces : ℕ)  -- Number of faces at the base

-- Assuming the characteristics of a pentagonal pyramid
def pentagonal_pyramid_characteristics : PentagonalPyramid :=
{ base_edges := 5,
  lateral_edges := 5,
  triangular_faces := 5,
  base_faces := 1 }

-- The theorem stating that a pentagonal pyramid has 10 edges and 6 faces
theorem pentagonal_pyramid_edges_faces (P : PentagonalPyramid) :
  P.base_edges + P.lateral_edges = 10 ∧ P.triangular_faces + P.base_faces = 6 := 
by
  -- Using the given characteristics
  have hP : P = pentagonal_pyramid_characteristics := sorry,
  -- Proving the statement using the characteristics
  rw hP,
  constructor; -- Prove both parts of the conjunction
  norm_num

end pentagonal_pyramid_edges_faces_l378_378153


namespace sin_pi14_sin_3pi14_sin_5pi14_eq_1div8_l378_378443

theorem sin_pi14_sin_3pi14_sin_5pi14_eq_1div8 : 
  sin (Real.pi / 14) * sin (3 * Real.pi / 14) * sin (5 * Real.pi / 14) = 1 / 8 :=
by
  sorry

end sin_pi14_sin_3pi14_sin_5pi14_eq_1div8_l378_378443


namespace domain_of_function_l378_378599

variable (x : ℝ)

theorem domain_of_function :
  {x : ℝ | x + 3 ≥ 0 ∧ 2 - x ≠ 0} =
  {x : ℝ | x ≥ -3 ∧ x ≠ 2} :=
by
  sorry

end domain_of_function_l378_378599


namespace num_ways_to_choose_president_and_vice_president_l378_378340

theorem num_ways_to_choose_president_and_vice_president (n : ℕ) (h_n : n = 5) : 
  let possible_ways := n * (n - 1) in
  possible_ways = 20 := 
by
  sorry

end num_ways_to_choose_president_and_vice_president_l378_378340


namespace room_width_is_12_l378_378048

variable (w : ℕ)

-- Definitions of given conditions
def room_length := 19
def veranda_width := 2
def veranda_area := 140

-- Statement that needs to be proven
theorem room_width_is_12
  (h1 : veranda_width = 2)
  (h2 : veranda_area = 140)
  (h3 : room_length = 19) :
  w = 12 :=
by
  sorry

end room_width_is_12_l378_378048


namespace sum_first_120_terms_l378_378647

noncomputable def sequence (a : ℝ) : ℕ → ℝ
| 0 => a
| n + 1 => (2 * |Real.sin (n * Real.pi / 2)| - 1) * sequence a n + 2 * (n + 1)

theorem sum_first_120_terms (a : ℝ) : 
  (Finset.range 120).sum (λ n => sequence a (n + 1)) = 10860 :=
by
  sorry

end sum_first_120_terms_l378_378647


namespace intersection_A_B_eq_123_l378_378401

/-- A is the set of natural numbers such that x^2 - 3x - 4 < 0 -/
def A : Set ℕ := {x | x^2 - 3 * x - 4 < 0}

/-- B is the set of numbers such that -1/3 ≤ 2x - 1 ≤ 9 -/
def B : Set ℝ := {x | -1 / 3 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 9}

theorem intersection_A_B_eq_123 : (A ∩ (B ∩ Set.univ)) = {1, 2, 3} :=
by
  sorry

end intersection_A_B_eq_123_l378_378401


namespace periodic_sequences_bounded_below_l378_378615

/-
  Prove that the set of all \( C \in \mathbb{R} \) such that every sequence of integers \(\{a_n\}_{n=1}^{\infty}\)
  which is bounded from below and for all \( n \geq 2 \) satisfies \( 0 \leq a_{n-1} + C a_n + a_{n+1} < 1 \)
  is periodic, is \([-2, \infty)\).
-/
theorem periodic_sequences_bounded_below (C : ℝ) :
  (∀ (a : ℕ → ℤ), (∃ N : ℤ, ∀ n, a n ≥ N) →
    (∀ n ≥ 2, 0 ≤ a (n - 1) + ⌊C * a n⌋ + a (n + 1) ∧ a (n - 1) + ⌊C * a n⌋ + a (n + 1) < 1) →
    (∃ T : ℕ, ∀ n, a n = a (n + T))) ↔ C ∈ set.Ici (-2) :=
begin
  sorry
end

end periodic_sequences_bounded_below_l378_378615


namespace smallest_base_10_integer_l378_378885

noncomputable def smallest_integer (a b: ℕ) (h₁: a > 2) (h₂: b > 2) (h₃: n = 2 * a + 1) (h₄: n = b + 2) : ℕ :=
  n

theorem smallest_base_10_integer : smallest_integer 3 5 (by decide) (by decide) (by decide) (by decide) = 7 :=
sorry

end smallest_base_10_integer_l378_378885


namespace factorization_l378_378613

theorem factorization (a b : ℝ) : a^2 - a * b = a * (a - b) := by
  sorry

end factorization_l378_378613


namespace probability_of_4_stable_l378_378950

-- Define the probabilities for positive, negative, and neutral states
def positive_prob : ℚ := 1 / 5
def negative_prob : ℚ := 1 / 10
def neutral_prob : ℚ := 17 / 20

-- Define the number of ecosystems
def num_ecosystems : ℕ := 7

-- Define the probability of exactly 4 stable ecosystems without disturbance
def probability_exactly_4_stable : ℚ :=
  (nat.choose num_ecosystems 4) * (positive_prob ^ 4) * (neutral_prob ^ 3)

-- The theorem to prove
theorem probability_of_4_stable : probability_exactly_4_stable = 34391 / 1000000 := by
  sorry

end probability_of_4_stable_l378_378950


namespace hexagon_colorings_correct_l378_378989

def valid_hexagon_colorings : Prop :=
  ∃ (colors : Fin 6 → Fin 7),
    (colors 0 ≠ colors 1) ∧
    (colors 1 ≠ colors 2) ∧
    (colors 2 ≠ colors 3) ∧
    (colors 3 ≠ colors 4) ∧
    (colors 4 ≠ colors 5) ∧
    (colors 5 ≠ colors 0) ∧
    (colors 0 ≠ colors 2) ∧
    (colors 1 ≠ colors 3) ∧
    (colors 2 ≠ colors 4) ∧
    (colors 3 ≠ colors 5) ∧
    ∃! (n : Nat), n = 12600

theorem hexagon_colorings_correct : valid_hexagon_colorings :=
sorry

end hexagon_colorings_correct_l378_378989


namespace count_of_desired_polynomials_l378_378602

def is_desired_polynomial (a : list ℤ) (n : ℕ) : Prop :=
  list.sum (a.map Int.natAbs) + 2 * n = 5

def count_desired_polynomials : ℕ :=
  -- This skips the actual implementation for now.
  sorry

theorem count_of_desired_polynomials : count_desired_polynomials = 16 := 
  sorry

end count_of_desired_polynomials_l378_378602


namespace inheritance_problem_l378_378845

theorem inheritance_problem
    (A B C : ℕ)
    (h1 : A + B + C = 30000)
    (h2 : A - B = B - C)
    (h3 : A = B + C) :
    A = 15000 ∧ B = 10000 ∧ C = 5000 := by
  sorry

end inheritance_problem_l378_378845


namespace parametric_eq_line_l378_378626

def point_M : ℝ := 1
def point_N : ℝ := 5
def angle : ℝ := real.pi / 3
def parametric_x (t : ℝ) : ℝ := 1 + (1 / 2) * t
def parametric_y (t : ℝ) : ℝ := 5 + (real.sqrt 3 / 2) * t

theorem parametric_eq_line : ∀ t : ℝ, 
  (parametric_x t) = (1 + (1 / 2) * t) ∧ 
  (parametric_y t) = (5 + (real.sqrt 3 / 2) * t) :=
by
  intro t
  sorry

end parametric_eq_line_l378_378626


namespace investment_duration_l378_378419

-- Definitions of the given conditions
def initial_investment : ℝ := 3500
def final_investment : ℝ := 31500
def interest_rate : ℝ := 8
def compounding_period (x : ℝ) : ℝ := 112 / x

-- The main statement
theorem investment_duration :
  let tripling_period := compounding_period interest_rate in
  let number_of_triplings := (log (final_investment / initial_investment)) / (log 3) in
  let total_years := number_of_triplings * tripling_period in
  total_years = 28 := 
sorry

end investment_duration_l378_378419


namespace ratio_DK_AB_l378_378948

variables (A B C D C1 K : Point)
variables [rect : Rect A B C D]
variables (midpointA : Midpoint C1 A D)
variables (segmentDK : Segment D K)
variables (segmentAB : Segment A B)

theorem ratio_DK_AB : ∃ CD, DK = (1 / 3) * CD → CD = AB → (DK / AB) = (1 / 3) :=
by {
  sorry
}

end ratio_DK_AB_l378_378948


namespace homework_points_40_l378_378788

-- Define the variables and hypotheses in Lean
variables (x quiz_points test_points : ℕ)

-- Define the conditions as hypotheses
def total_points_eq : Prop := x + quiz_points + test_points = 265
def quiz_points_eq : Prop := quiz_points = x + 5
def test_points_eq : Prop := test_points = 4 * quiz_points

-- State the theorem we wish to prove
theorem homework_points_40 (h1 : total_points_eq) (h2 : quiz_points_eq) (h3 : test_points_eq) : x = 40 :=
by
  sorry

end homework_points_40_l378_378788


namespace find_positive_number_l378_378913

theorem find_positive_number (x : ℝ) (hx : 0 < x) (h : (2 / 3) * x = (144 / 216) * (1 / x)) : x = 1 := by
  sorry

end find_positive_number_l378_378913


namespace hens_that_do_not_lay_eggs_l378_378003

theorem hens_that_do_not_lay_eggs (total_chickens : ℕ) (roosters : ℕ) (egg_laying_hens : ℕ) 
  (h1 : total_chickens = 325) (h2 : roosters = 28) (h3 : egg_laying_hens = 277) : 
  total_chickens - roosters - egg_laying_hens = 20 :=
by
  rw [h1, h2, h3]
  norm_num

end hens_that_do_not_lay_eggs_l378_378003


namespace unique_binomial_representation_l378_378439

theorem unique_binomial_representation 
  (t l : ℕ) : 
  ∃ (a : ℕ → ℕ), a l > a (l-1) ∧ a (l-1) > a (l-2) ∧ ... ∧ a (m) ≥ m ∧
  t = (nat.choose (a l) l) + (nat.choose (a (l-1)) (l-1)) + ... + (nat.choose (a m) m) ∧ 
  (∀ b : ℕ → ℕ, b l > b (l-1) ∧ b (l-1) > b (l-2) ∧ ... ∧ b (m) ≥ m ∧ 
  t = (nat.choose (b l) l) + (nat.choose (b (l-1)) (l-1)) + ... + (nat.choose (b m) m) → a = b) := sorry

end unique_binomial_representation_l378_378439


namespace log_base_fraction_l378_378226

theorem log_base_fraction : ∀ (a b : ℝ) (x : ℝ), 16 = (4:ℝ)^2 ∧ (1 / 4:ℝ) = 4^(-1) → log (1 / 4) 16 = -2 :=
begin
  intros a b x h,
  -- Skipping the proof by adding sorry
  sorry,
end

end log_base_fraction_l378_378226


namespace log_base_fraction_l378_378228

theorem log_base_fraction : ∀ (a b : ℝ) (x : ℝ), 16 = (4:ℝ)^2 ∧ (1 / 4:ℝ) = 4^(-1) → log (1 / 4) 16 = -2 :=
begin
  intros a b x h,
  -- Skipping the proof by adding sorry
  sorry,
end

end log_base_fraction_l378_378228


namespace polynomial_remainder_eq_one_l378_378766

/-
We are given the conditions:
1. \( z^{2023} - z + 1 = (z^3 + z^2 + 1)P(z) + S(z) \)
2. The degree of \( S(z) \) is less than 3

We need to show that \( S(z) = 1 \).
-/

theorem polynomial_remainder_eq_one :
  ∃ S P : Polynomial ℂ, 
  (∀ z : ℂ, z^{2023} - z + 1 = (z^3 + z^2 + 1) * P + S) ∧ 
  (degree S < 3) ∧ 
  S = 1 :=
by
sorry

end polynomial_remainder_eq_one_l378_378766


namespace new_median_after_adding_ten_l378_378542

def initial_collection := [4, 4, 5, 5, 6, 9] -- derived from the solution steps

theorem new_median_after_adding_ten : 
  let new_collection := initial_collection ++ [10] in
  List.median new_collection = 5 :=
by
  sorry

end new_median_after_adding_ten_l378_378542


namespace sum_of_coefficients_l378_378694

theorem sum_of_coefficients:
  (∀ x : ℝ, (2*x - 1)^6 = a_0*x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6) →
  a_1 + a_3 + a_5 = -364 :=
by
  sorry

end sum_of_coefficients_l378_378694


namespace find_a_l378_378679

def f (a x : ℝ) : ℝ := (a - 1) * x^2 + a * sin x

theorem find_a (a : ℝ) : (∀ x : ℝ, f a x = f a (-x)) ↔ a = 0 :=
by
  intros
  sorry

end find_a_l378_378679


namespace cot_alpha_le_neg_e_equality_when_y_l378_378300

variables (a b c e x y : ℝ) (alpha : ℝ)
variables (e1 e2 : ℝ × ℝ)
variable P : ℝ × ℝ
variable (e2_pos : e^2 > (1 / 2) * (Real.sqrt 5 - 1))
variable (h1 : b^2 * x^2 + a^2 * y^2 = a^2 * b^2)
variable (h2 : P.1 = x ∧ P.2 = y)
variable (h3 : alpha = ArcTan ((y - e1.2) / (x - e1.1)) - ArcTan ((y - e2.2) / (x - e2.1)))
variable (h4 : e = Real.sqrt (1 - (b^2 / a^2)))
variable (h5 : c = Real.sqrt (a^2 - b^2))
variable (alpha_obtuse : α > π / 2 ∧ α < π)
variable (directrices : e1 = (-a^2 / c, 0) ∧ e2 = (a^2 / c, 0))
variable (y_eq : y = (a * b^2) / (c^2))

theorem cot_alpha_le_neg_e : 
  cot (angle_alpha P e1 e2) <= -e := sorry

theorem equality_when_y :
  cot (angle_alpha P e1 e2) = -e ↔ |P.2| = (a * b^2) / (c^2) := sorry

end cot_alpha_le_neg_e_equality_when_y_l378_378300


namespace find_point_and_line_l378_378669

-- Define the conditions
def curve (x : ℝ) : ℝ := x^3 + x - 2

-- Derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

-- Given line l1 is parallel to 4x - y - 1 = 0, so slope of l1 is 4
def tangent_line_slope : ℝ := 4

-- Point P₀ is in the third quadrant
def third_quadrant_point (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ P.2 < 0

-- The equation of a line passing through a point with a given slope
def line_through_point (P : ℝ × ℝ) (m : ℝ) : ℝ → ℝ := 
  λ x, m * (x - P.1) + P.2

-- The perpendicular condition and the equations derived from the conditions.
theorem find_point_and_line :
  ∃ P₀ : ℝ × ℝ, 
    (curve P₀.1 = P₀.2) ∧ 
    (curve_derivative P₀.1 = tangent_line_slope) ∧ 
    third_quadrant_point P₀ ∧ 
    ∃ l : ℝ → ℝ, 
      (∀ x, l x = -x / 4 - 17 / 4) ∧ 
      l P₀.1 = P₀.2 :=
by
  sorry

end find_point_and_line_l378_378669


namespace ferris_wheel_height_expression_best_visual_effect_time_l378_378035

noncomputable def ferris_wheel_height (t : ℝ) : ℝ :=
  -50 * Real.cos ((2 * Real.pi / 3) * t) + 50

theorem ferris_wheel_height_expression :
  ∀ t : ℝ, ferris_wheel_height t = -50 * Real.cos ((2 * Real.pi / 3) * t) + 50 :=
by intro t; rfl

theorem best_visual_effect_time :
  t = 3 - (3 / Real.pi) * Real.arccos (-7/10) :=
sorry

end ferris_wheel_height_expression_best_visual_effect_time_l378_378035


namespace sequence_sum_eq_126_implies_n_eq_6_l378_378381

theorem sequence_sum_eq_126_implies_n_eq_6 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 2 →
  (∀ n, a (n + 1) = 2 * a n) →
  (∀ n, S n = ∑ k in finset.range n, a (k + 1)) →
  S 6 = 126 →
  ∃ n, S n = 126 ∧ n = 6 :=
by
  intros ha1 hrec hsum hs6
  use 6
  exact ⟨hs6, rfl⟩

end sequence_sum_eq_126_implies_n_eq_6_l378_378381


namespace smallest_base_10_integer_exists_l378_378874

theorem smallest_base_10_integer_exists :
  ∃ (x a b : ℕ), (a > 2) ∧ (b > 2) ∧ (x = 2 * a + 1) ∧ (x = b + 2) ∧ (x = 7) :=
by
  sorry

end smallest_base_10_integer_exists_l378_378874


namespace sum_of_money_l378_378140

theorem sum_of_money (A B C : ℝ) (hB : B = 0.65 * A) (hC : C = 0.40 * A) (hC_value : C = 32) :
  A + B + C = 164 :=
by
  sorry

end sum_of_money_l378_378140


namespace geom_series_common_ratio_l378_378066

theorem geom_series_common_ratio (a r S : ℝ) (h1 : S = a / (1 - r)) 
  (h2 : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
sorry

end geom_series_common_ratio_l378_378066


namespace obtain_n_from_a_cannot_obtain_n_from_b_cannot_obtain_n_from_c_l378_378289

def seq (f : ℕ → ℝ) : ℕ → ℝ := f

def drop_initial_terms {α : Type*} (s : ℕ → α) (n : ℕ) : ℕ → α := λ m, s (m + n)

def add_seqs (s1 s2 : ℕ → ℝ) : ℕ → ℝ := λ n, s1 n + s2 n
def subtract_seqs (s1 s2 : ℕ → ℝ) : ℕ → ℝ := λ n, s1 n - s2 n
def multiply_seqs (s1 s2 : ℕ → ℝ) : ℕ → ℝ := λ n, s1 n * s2 n
def divide_seqs (s1 s2 : ℕ → ℝ) (h : ∀ n, s2 n ≠ 0) : ℕ → ℝ := λ n, s1 n / s2 n

noncomputable def an_a (n : ℕ) : ℝ := n^2
noncomputable def an_b (n : ℕ) : ℝ := (n : ℝ) + real.sqrt 2
noncomputable def an_c (n : ℕ) : ℝ := (n^2000 + 1) / n

theorem obtain_n_from_a (f : ℕ → ℝ) : 
  (∀ n, f n = an_a n) → (∃ g : ℕ → ℝ, ∀ n, g n = n) :=
sorry

theorem cannot_obtain_n_from_b (f : ℕ → ℝ) : 
  (∀ n, f n = an_b n) → (¬ ∃ g : ℕ → ℝ, ∀ n, g n = n) :=
sorry

theorem cannot_obtain_n_from_c (f : ℕ → ℝ) : 
  (∀ n, f n = an_c n) → (¬ ∃ g : ℕ → ℝ, ∀ n, g n = n) :=
sorry

end obtain_n_from_a_cannot_obtain_n_from_b_cannot_obtain_n_from_c_l378_378289


namespace small_bottles_count_l378_378746

theorem small_bottles_count : 
  ∀ (x : ℕ), (1365 * 1.89 + x * 1.42) / (1365 + x) = 1.73 → x = 701 :=
by
  sorry

end small_bottles_count_l378_378746


namespace intersect_or_parallel_l378_378852

def line1 (t : ℝ) (k : ℝ) : ℝ × ℝ := (3 + 2 * t, -1 - k * t)
def line2 (u : ℝ) (k : ℝ) : ℝ × ℝ := (2 + k * u, 5 + 3 * u)

theorem intersect_or_parallel (k : ℝ) (t u : ℝ) :
    (line1 t k = line2 u k) ↔ (k = real.sqrt 6 ∨ k = -real.sqrt 6) :=
    sorry

end intersect_or_parallel_l378_378852


namespace total_winter_clothing_l378_378113

theorem total_winter_clothing (boxes : ℕ) (scarves_per_box mittens_per_box : ℕ) (h_boxes : boxes = 8) (h_scarves : scarves_per_box = 4) (h_mittens : mittens_per_box = 6) : 
  boxes * (scarves_per_box + mittens_per_box) = 80 := 
by
  sorry

end total_winter_clothing_l378_378113


namespace time_for_B_to_complete_job_l378_378136

-- Define the conditions
def a_rate := 1 / 10
def b_days (x : ℝ) := x
def b_rate (x : ℝ) := 1 / x
def combined_work (x : ℝ) := 4 * (a_rate + b_rate x)
def work_left := 0.4

-- Prove the main theorem
theorem time_for_B_to_complete_job (x : ℝ) (h1 : combined_work x = 0.6) : b_days x = 20 :=
by {
  -- The proof steps go here
  sorry
}

end time_for_B_to_complete_job_l378_378136


namespace semicircles_cover_quadrilateral_l378_378429

theorem semicircles_cover_quadrilateral (A B C D : Point) (h_convex : ConvexQuadrilateral A B C D) (h_semicircles : ∀ (P : Point), P.insideQuadrilateral A B C D → ∃ (side : LineSegment), P.onOrInsideSemicircle diameter side) : 
  ∀ (P : Point), P.insideQuadrilateral A B C D → ∃ (side : LineSegment), P.onOrInsideSemicircle diameter side := 
by
  sorry

end semicircles_cover_quadrilateral_l378_378429


namespace square_area_l378_378954

theorem square_area (y1 y2 y3 y4 : ℤ) 
  (h1 : y1 = 0) (h2 : y2 = 3) (h3 : y3 = 0) (h4 : y4 = -3) : 
  ∃ (area : ℤ), area = 36 :=
by
  sorry

end square_area_l378_378954


namespace Jack_gave_Mike_six_notebooks_l378_378387

theorem Jack_gave_Mike_six_notebooks :
  ∀ (Gerald_notebooks Jack_notebooks_left notebooks_given_to_Paula total_notebooks_initial jack_notebooks_after_Paula notebooks_given_to_Mike : ℕ),
  Gerald_notebooks = 8 →
  Jack_notebooks_left = 10 →
  notebooks_given_to_Paula = 5 →
  total_notebooks_initial = Gerald_notebooks + 13 →
  jack_notebooks_after_Paula = total_notebooks_initial - notebooks_given_to_Paula →
  notebooks_given_to_Mike = jack_notebooks_after_Paula - Jack_notebooks_left →
  notebooks_given_to_Mike = 6 :=
by
  intros Gerald_notebooks Jack_notebooks_left notebooks_given_to_Paula total_notebooks_initial jack_notebooks_after_Paula notebooks_given_to_Mike
  intros Gerald_notebooks_eq Jack_notebooks_left_eq notebooks_given_to_Paula_eq total_notebooks_initial_eq jack_notebooks_after_Paula_eq notebooks_given_to_Mike_eq
  sorry

end Jack_gave_Mike_six_notebooks_l378_378387


namespace rightmost_four_digits_of_5_pow_2023_l378_378091

theorem rightmost_four_digits_of_5_pow_2023 :
  (5 ^ 2023) % 10000 = 8125 :=
sorry

end rightmost_four_digits_of_5_pow_2023_l378_378091


namespace area_cross_section_correct_l378_378064

noncomputable def area_cross_section (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt 2) / 4

theorem area_cross_section_correct (a b : ℝ) :
  let SABCD := (regular_pyramid (base := square (side := a)) (lateral_edge := b))
  let BD := diagonal SABCD.base
  let plane := plane_through_line_parallel_to_edge BD SABCD.lateral_edge
  cross_sectional_area SABCD plane = area_cross_section a b :=
sorry

end area_cross_section_correct_l378_378064


namespace smallest_sum_of_digits_l378_378218

noncomputable def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem smallest_sum_of_digits (n : ℕ) (h : sum_of_digits n = 2017) : sum_of_digits (n + 1) = 2 := 
sorry

end smallest_sum_of_digits_l378_378218


namespace vertex_y_coordinate_l378_378945

-- Define the parabola
def parabola (x : ℝ) : ℝ := -5 * x^2 - 40 * x - 100

-- Statement to be proven: The y-coordinate of the vertex is -20
theorem vertex_y_coordinate : ∃ k, k = -20 ∧ ∀ x, parabola(x) = -5 * (x + 4)^2 + k := 
by {
  use -20,
  sorry
}

end vertex_y_coordinate_l378_378945


namespace solution_for_n_l378_378760

def σ (n : ℕ) := (n.prime_divisors).sum

theorem solution_for_n (n : ℕ) :
  (σ (2^n + 1) = σ n) ↔ (n = 3) :=
by
  sorry

end solution_for_n_l378_378760


namespace donny_total_spending_l378_378604

noncomputable def total_saving_mon : ℕ := 15
noncomputable def total_saving_tue : ℕ := 28
noncomputable def total_saving_wed : ℕ := 13
noncomputable def total_saving_fri : ℕ := 22

noncomputable def total_savings_mon_to_wed : ℕ := total_saving_mon + total_saving_tue + total_saving_wed
noncomputable def thursday_spending : ℕ := total_savings_mon_to_wed / 2
noncomputable def remaining_savings_after_thursday : ℕ := total_savings_mon_to_wed - thursday_spending
noncomputable def total_savings_before_sat : ℕ := remaining_savings_after_thursday + total_saving_fri
noncomputable def saturday_spending : ℕ := total_savings_before_sat * 40 / 100

theorem donny_total_spending : thursday_spending + saturday_spending = 48 := by sorry

end donny_total_spending_l378_378604


namespace trapezoid_area_regular_octagon_l378_378196

theorem trapezoid_area_regular_octagon (s : ℝ) (h_s : s = 6) :
  let octagon_angle := 135
  let trig_value := Real.sin ( 22.5 / 180 * Real.pi ) 
  let base_1 := s
  let base_2 := s
  let diagonal := s * Real.sqrt ( 4 + 2 * Real.sqrt 2 )
  let height := diagonal * trig_value in
  (base_1 + base_2) / 2 * height = 18 * Real.sqrt 12 :=
by
  sorry

end trapezoid_area_regular_octagon_l378_378196


namespace number_of_people_l378_378178

-- Definitions based on the conditions
def total_cookies : ℕ := 420
def cookies_per_person : ℕ := 30

-- The goal is to prove the number of people is 14
theorem number_of_people : total_cookies / cookies_per_person = 14 :=
by
  sorry

end number_of_people_l378_378178


namespace joggers_difference_l378_378567

theorem joggers_difference (Tyson_joggers Alexander_joggers Christopher_joggers : ℕ) 
  (h1 : Alexander_joggers = Tyson_joggers + 22) 
  (h2 : Christopher_joggers = 20 * Tyson_joggers)
  (h3 : Christopher_joggers = 80) : 
  Christopher_joggers - Alexander_joggers = 54 :=
by 
  sorry

end joggers_difference_l378_378567


namespace fourth_term_of_sequence_l378_378327

theorem fourth_term_of_sequence (x : ℤ) (h : x^2 - 2 * x - 3 < 0) (hx : x ∈ {n : ℤ | x^2 - 2 * x - 3 < 0}) :
  ∃ a_1 a_2 a_3 a_4 : ℤ, 
  (a_1 = x) ∧ (a_2 = x + 1) ∧ (a_3 = x + 2) ∧ (a_4 = x + 3) ∧ 
  (a_4 = 3 ∨ a_4 = -1) :=
by { sorry }

end fourth_term_of_sequence_l378_378327


namespace wise_men_minimize_hair_loss_l378_378546

theorem wise_men_minimize_hair_loss (a : Fin 100 → ℕ) (N : ℕ) (mod_sum : ℕ) :
  (N = 101) →
  (mod_sum = ∑ i, a i % N) →
  ∃ (decoded_numbers : Fin 100 → ℕ), 
    (∀ i, decoded_numbers i = a i) ∧ 
    ∀ (hair_loss : ℕ), (hair_loss = 1) :=
by
  sorry

end wise_men_minimize_hair_loss_l378_378546


namespace find_angle_A_find_area_S_l378_378357

open Real

variables {a b c : ℝ} {A B C : ℝ}

/-- Given that in a triangle ABC where the sides opposite to angles A, B, and C are a, b, 
    and c respectively and a * cos C + c * cos A = 2 * b * cos A, 
    the value of angle A is π / 3. -/
theorem find_angle_A 
    (h1 : a * cos C + c * cos A = 2 * b * cos A)
    (A_pos : 0 < A) (A_lt_pi : A < π) : 
    A = π / 3 :=
sorry

/-- Given that in a triangle ABC where sides a, b, and c satisfy a = 2, 
    b + c = sqrt 10, and angle A = π / 3, the area of the triangle is sqrt 3 / 2. -/
theorem find_area_S 
    (b c : ℝ) 
    (h2 : b + c = sqrt 10) 
    (ha : a = 2) 
    (hA : A = π / 3) : 
    let S := (1 / 2) * b * c * sin A in
    S = sqrt 3 / 2 :=
sorry

end find_angle_A_find_area_S_l378_378357


namespace log_base_frac_l378_378235

theorem log_base_frac (x : ℝ) : log (1/4) 16 = x → x = -2 := by
  sorry

end log_base_frac_l378_378235


namespace find_x_such_that_fff_eq_15_l378_378318

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 1 else 3 * x

theorem find_x_such_that_fff_eq_15 (x : ℝ) : f (f x) = 15 ↔ x = -4 ∨ x = 5 :=
by
  sorry

end find_x_such_that_fff_eq_15_l378_378318


namespace cut_7x7_square_l378_378529

theorem cut_7x7_square :
  ∃ a b : ℕ, (7 * 7 = 49) ∧ (3 * a + 4 * b = 49) ∧ (b = 1) :=
by
  existsi 15
  existsi 1
  split
  { exact rfl }
  split
  { exact rfl.trivial }
  { exact rfl }

end cut_7x7_square_l378_378529


namespace sqrt_and_exponent_l378_378586

theorem sqrt_and_exponent : sqrt 16 - (Real.pi - 3)^0 = 3 := by
  have h1 : sqrt 16 = 4 := by
    sorry
  have h2 : (Real.pi - 3)^0 = 1 := by
    sorry
  rw [h1, h2]
  norm_num

end sqrt_and_exponent_l378_378586


namespace barbi_monthly_loss_l378_378173

variable (x : Real)

theorem barbi_monthly_loss : 
  (∃ x : Real, 12 * x = 99 - 81) → x = 1.5 :=
by
  sorry

end barbi_monthly_loss_l378_378173


namespace scientific_notation_113700_l378_378832

theorem scientific_notation_113700 : (113700 : ℝ) = 1.137 * 10^5 :=
by
  sorry

end scientific_notation_113700_l378_378832


namespace find_integer_pairs_l378_378280

theorem find_integer_pairs :
  ∀ (k ℓ : ℤ), (∃ x : ℤ, k = -32 + 3 * x ∧ ℓ = 64 - 5 * x) ↔ 5 * k + 3 * ℓ = 32 :=
by
  intro k ℓ
  constructor
  · intro ⟨x, hk, hℓ⟩
    rw [hk, hℓ]
    linarith
  · intro h
    obtain ⟨x, hx⟩ : ∃ x, k = -32 + 3 * x := sorry
    obtain ⟨y, hy⟩ : ∃ y, ℓ = 64 - 5 * y := sorry
    exact ⟨x, hx, hy⟩

end find_integer_pairs_l378_378280


namespace sequence_sum_formula_l378_378062

noncomputable def sequence_sum (n : ℕ) : ℕ :=
∑ k in Finset.range (n + 1), (∑ s in Finset.powersetLen k (Finset.range (n + 1)), (s.prod (λ x, 2^(x + 1) - 1)))

theorem sequence_sum_formula (n : ℕ) : sequence_sum n = (2 ^ (n * (n + 1) / 2)) - 1 := 
sorry

end sequence_sum_formula_l378_378062


namespace probability_of_green_light_l378_378902

theorem probability_of_green_light (red_time green_time yellow_time : ℕ) (h_red : red_time = 30) (h_green : green_time = 25) (h_yellow : yellow_time = 5) :
  (green_time.toRat / (red_time + green_time + yellow_time).toRat) = (5 / 12 : ℚ) :=
by
  sorry

end probability_of_green_light_l378_378902


namespace average_speed_of_car_l378_378486

theorem average_speed_of_car (dist1 dist2 time1 time2 : ℕ) (h1 : dist1 = 20) (h2 : dist2 = 60) (h3 : time1 = 1) (h4 : time2 = 1) : 
  let total_distance := dist1 + dist2,
      total_time := time1 + time2,
      average_speed := total_distance / total_time
  in average_speed = 40 := 
by
  sorry 

end average_speed_of_car_l378_378486


namespace find_a_b_sum_specific_find_a_b_sum_l378_378663

-- Define the sets A and B based on the given inequalities
def set_A : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
def set_B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Intersect the sets A and B
def set_A_int_B : Set ℝ := set_A ∩ set_B

-- Define the inequality with parameters a and b
def quad_ineq (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 2 > 0}

-- Define the parameters a and b based on the given condition
noncomputable def a : ℝ := -1
noncomputable def b : ℝ := -1

-- The statement to be proved
theorem find_a_b_sum : ∀ a b : ℝ, set_A ∩ set_B = {x | a * x^2 + b * x + 2 > 0} → a + b = -2 :=
by
  sorry

-- Fixing the parameters a and b for our specific proof condition
theorem specific_find_a_b_sum : a + b = -2 :=
by
  sorry

end find_a_b_sum_specific_find_a_b_sum_l378_378663


namespace attic_useful_items_l378_378001

-- Define the problem's conditions and the theorem we want to prove.
theorem attic_useful_items (T : ℝ) (junk_items : ℝ) (useful_percentage : ℝ) (junk_percentage : ℝ)
  (h_junk_total : junk_percentage * T = junk_items) (h_junk_count : junk_items = 28) 
  (h_junk_rate : junk_percentage = 0.70) (h_useful_rate : useful_percentage = 0.20) :
  useful_percentage * T = 8 := 
by 
  -- Use the given conditions
  have h_T : T = 28 / 0.70, from
    eq_div_of_mul_eq (ne_of_gt (by norm_num : 0.70 > 0)) (eq.trans (mul_comm _ _) h_junk_total.symm),
  -- Now substitute T
  have h_useful : useful_percentage * T = 0.20 * (28 / 0.70), from congr_arg (fun x => useful_percentage * x) h_T,
  rw [h_useful_rate] at h_useful,
  -- Simplify the right hand side
  norm_num at h_useful,
  -- Finally, simplify the equation
  exact h_useful,
  sorry

end attic_useful_items_l378_378001


namespace correct_function_C_l378_378168

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

lemma function_C_odd : ∀ x : ℝ, f (-x) = -f x := by
  intros x
  have h1 : f (-x) = Real.exp (-x) - Real.exp x := by
    unfold f
    rw [neg_neg x]
  rw [h1, neg_sub]
  sorry

lemma function_C_monotonically_increasing : ∀ x y : ℝ, 0 < x → x < 1 → 0 < y → y < 1 → x < y → f x < f y := by
  sorry

theorem correct_function_C :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x < y → f x < f y) := by
  apply And.intro
  exact function_C_odd
  exact function_C_monotonically_increasing

end correct_function_C_l378_378168


namespace sum_midpoint_x_coords_l378_378067

theorem sum_midpoint_x_coords (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a - b = 3) :
    (a + (a - 3)) / 2 + (a + c) / 2 + ((a - 3) + c) / 2 = 15 := 
by 
  sorry

end sum_midpoint_x_coords_l378_378067


namespace log_base_one_fourth_of_sixteen_l378_378253

theorem log_base_one_fourth_of_sixteen :
  log (1 / 4 : ℝ) (16 : ℝ) = -2 :=
sorry

end log_base_one_fourth_of_sixteen_l378_378253


namespace exponent_form_l378_378865

theorem exponent_form (x : ℕ) (k : ℕ) : (3^x) % 10 = 7 ↔ x = 4 * k + 3 :=
by
  sorry

end exponent_form_l378_378865


namespace f_periodicity_and_symmetry_l378_378644

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ set.Icc 0 2 then -x^2 + 1 else sorry

theorem f_periodicity_and_symmetry (x : ℝ) :
  (∀ x, f (-x) = f x) ∧ (∀ x, f (4 + x) = f x) ∧ 
  (∀ x, x ∈ set.Icc 0 2 → f x = -x^2 + 1) →
  ∀ x, x ∈ set.Icc (-6) (-4) → f x = -(x + 4)^2 + 1 :=
by
  sorry

end f_periodicity_and_symmetry_l378_378644


namespace total_volume_of_five_cubes_l378_378217

-- Define the edge length and number of cubes as constants.
def edge_length : ℕ := 6
def number_of_cubes : ℕ := 5

-- Define the volume of a cube.
def cube_volume (s : ℕ) : ℕ := s * s * s

-- Define the total volume for a number of identical cubes.
def total_volume (n : ℕ) (v : ℕ) : ℕ := n * v

-- The proof statement.
theorem total_volume_of_five_cubes : total_volume number_of_cubes (cube_volume edge_length) = 1080 := by
  let v := cube_volume edge_length
  let total := total_volume number_of_cubes v
  have volume_one_cube : v = 216 := by
    sorry
  have total_volume := 5 * 216
  show total = 1080

end total_volume_of_five_cubes_l378_378217


namespace reggie_games_lost_l378_378447

-- Given conditions:
def initial_marbles : ℕ := 100
def marbles_per_game : ℕ := 10
def games_played : ℕ := 9
def marbles_after_games : ℕ := 90

-- The statement to prove:
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / marbles_per_game = 1 := 
sorry

end reggie_games_lost_l378_378447


namespace max_N_impassable_roads_l378_378373

def number_of_cities : ℕ := 1000
def number_of_roads : ℕ := 2017
def initial_connected_components : ℕ := 1
def target_connected_components : ℕ := 7

theorem max_N_impassable_roads 
    (h : ∀ (G : SimpleGraph (Fin 1000)), G.edgeCount = 2017 ∧ G.isConnected ∧ G.connectedComponents.card = target_connected_components) :
    ∃ N : ℕ, N = 993 :=
begin
  use 993,
  sorry
end

end max_N_impassable_roads_l378_378373


namespace problem1_problem2_l378_378326

-- Problem 1: Prove the solution set of the given inequality
theorem problem1 (x : ℝ) : (|x - 2| + 2 * |x - 1| > 5) ↔ (x < -1/3 ∨ x > 3) := 
sorry

-- Problem 2: Prove the range of values for 'a' such that the inequality holds
theorem problem2 (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ |a - 2|) ↔ (a ≤ 3/2) :=
sorry

end problem1_problem2_l378_378326


namespace playerA_wins_6th_game_l378_378791

-- Define the conditions of the game
def fair_coin := 0.5

-- Define probability calculation of losing n games and winning the n+1 game.
def prob_loses_then_wins (n : Nat) (p : ℝ) : ℝ :=
  (p ^ n) * p

-- Statement to prove
theorem playerA_wins_6th_game :
  prob_loses_then_wins 5 fair_coin = (1 / 64) := 
  by
    sorry

end playerA_wins_6th_game_l378_378791


namespace unique_integers_exist_l378_378630

theorem unique_integers_exist :
  ∃ b2 b3 b4 b5 b6 : ℤ,
  (11 / 15 : ℚ) = (b2 / 2.factorial : ℚ) + (b3 / 3.factorial) + (b4 / 4.factorial) + (b5 / 5.factorial) + (b6 / 6.factorial) ∧
  (0 ≤ b2 ∧ b2 < 2) ∧
  (0 ≤ b3 ∧ b3 < 3) ∧
  (0 ≤ b4 ∧ b4 < 4) ∧
  (0 ≤ b5 ∧ b5 < 5) ∧
  (0 ≤ b6 ∧ b6 < 6) ∧
  (b2 + b3 + b4 + b5 + b6 = 3) :=
by
  sorry

end unique_integers_exist_l378_378630


namespace avg_annual_growth_rate_l378_378966

variable (x : ℝ)

/-- Initial GDP in 2020 is 43903.89 billion yuan and GDP in 2022 is 53109.85 billion yuan. 
    Prove that the average annual growth rate x satisfies the equation 43903.89 * (1 + x)^2 = 53109.85 -/
theorem avg_annual_growth_rate (x : ℝ) :
  43903.89 * (1 + x)^2 = 53109.85 :=
sorry

end avg_annual_growth_rate_l378_378966


namespace count_rationals_in_list_l378_378818

def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem count_rationals_in_list :
  let l := [0.5, (2 : ℝ) * Real.pi, 1.264850349, 0, (22 : ℝ) / 7, 
            let a := λ n, (List.replicate (n + 1) 1).intercalate [2] in
            (0.2121121112 + a)]
  in l.countp is_rational = 4 :=
by
  sorry

end count_rationals_in_list_l378_378818


namespace find_x_for_perpendicular_vectors_l378_378331

theorem find_x_for_perpendicular_vectors :
  ∀ (x : ℝ),
  let a := (-2 : ℝ, 1 : ℝ)
  let b := (x, x^2 + 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 1 :=
by
  intros x a b h
  sorry

end find_x_for_perpendicular_vectors_l378_378331


namespace limit_seq_l378_378590

open Filter Real

-- Define the sequence as a function from natural numbers to real numbers
def seq (n : ℕ) : ℝ := (1 - 2 * (n:ℝ)) / ((n:ℝ) + 2)

-- State the theorem as a limit problem with the given sequence
theorem limit_seq : Tendsto seq atTop (𝓝 (-2)) :=
sorry

end limit_seq_l378_378590


namespace probability_at_least_one_white_ball_l378_378534

-- Define the probability function
def probability {α : Type} (s : Finset α) (event : Set α) : ℚ :=
  (Finset.card (s.filter event)) / (Finset.card s)

-- Define the box and the events
def box : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def red_balls : Set ℕ := {1, 2, 3, 4, 5}
def white_balls : Set ℕ := {6, 7, 8, 9}

-- Define the event of picking two balls
def event (a b : ℕ) : Set (ℕ × ℕ) := { (x, y) | x ≠ y ∧ (x ∈ red_balls ∨ x ∈ white_balls) ∧ (y ∈ red_balls ∨ y ∈ white_balls)}

-- Define the event of picking two red balls
def event_red_red (a b : ℕ) : Set (ℕ × ℕ) := { (x, y) | x ≠ y ∧ x ∈ red_balls ∧ y ∈ red_balls}

-- Define the set of pairs of any two balls
def pairs_of_balls : Finset (ℕ × ℕ) := { (x, y) | x ∈ box ∧ y ∈ box ∧ x ≠ y}.to_finset

-- Prove the required probability 
theorem probability_at_least_one_white_ball 
  : probability pairs_of_balls (λ (p : ℕ × ℕ), p.1 ∈ white_balls ∨ p.2 ∈ white_balls) = 13 / 18 :=
by 
  sorry

end probability_at_least_one_white_ball_l378_378534


namespace new_median_after_adding_ten_l378_378543

def initial_collection := [4, 4, 5, 5, 6, 9] -- derived from the solution steps

theorem new_median_after_adding_ten : 
  let new_collection := initial_collection ++ [10] in
  List.median new_collection = 5 :=
by
  sorry

end new_median_after_adding_ten_l378_378543


namespace quadratic_inequality_solution_l378_378485

theorem quadratic_inequality_solution:
  ∀ x : ℝ, (x^2 + 2 * x < 3) ↔ (-3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l378_378485


namespace length_of_BD_l378_378376

-- Definitions of the points and lengths
variable (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)

-- Conditions
def right_angled_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  (dist B C)^2 = (dist A B)^2 + (dist A C)^2

def lengths : Prop :=
  (dist A B = 50) ∧ (dist A C = 120)

def perpendicular (D : Type) [MetricSpace D] : Prop :=
  ∃ D ∈ line BC, dist A D = d ∧ is_perpendicular AD BC

-- The proof goal
theorem length_of_BD (h : right_angled_triangle A B C) (h1 : lengths A B C) (h2 : perpendicular D A B C) :
  dist B D ≈ 110.77 :=
sorry

end length_of_BD_l378_378376


namespace find_additional_speed_l378_378925

noncomputable def speed_initial : ℝ := 55
noncomputable def t_initial : ℝ := 4
noncomputable def speed_total : ℝ := 60
noncomputable def t_total : ℝ := 6

theorem find_additional_speed :
  let distance_initial := speed_initial * t_initial
  let distance_total := speed_total * t_total
  let t_additional := t_total - t_initial
  let distance_additional := distance_total - distance_initial
  let speed_additional := distance_additional / t_additional
  speed_additional = 70 :=
by
  sorry

end find_additional_speed_l378_378925


namespace smallest_N_satisfying_conditions_l378_378157

def is_divisible (n m : ℕ) : Prop :=
  m ∣ n

def satisfies_conditions (N : ℕ) : Prop :=
  (is_divisible N 10) ∧
  (is_divisible N 5) ∧
  (N > 15)

theorem smallest_N_satisfying_conditions : ∃ N, satisfies_conditions N ∧ N = 20 := 
  sorry

end smallest_N_satisfying_conditions_l378_378157


namespace Manu_takes_12_more_seconds_l378_378434

theorem Manu_takes_12_more_seconds (P M A : ℕ) 
  (hP : P = 60) 
  (hA1 : A = 36) 
  (hA2 : A = M / 2) : 
  M - P = 12 :=
by
  sorry

end Manu_takes_12_more_seconds_l378_378434


namespace rightmost_four_digits_of_5_pow_2023_l378_378094

theorem rightmost_four_digits_of_5_pow_2023 :
  5 ^ 2023 % 5000 = 3125 :=
  sorry

end rightmost_four_digits_of_5_pow_2023_l378_378094


namespace barry_shirt_discount_l378_378175

theorem barry_shirt_discount 
  (original_price : ℤ) 
  (discount_percent : ℤ) 
  (discounted_price : ℤ) 
  (h1 : original_price = 80) 
  (h2 : discount_percent = 15)
  (h3 : discounted_price = original_price - (discount_percent * original_price / 100)) : 
  discounted_price = 68 :=
sorry

end barry_shirt_discount_l378_378175


namespace sum_of_areas_l378_378796

-- Let A, B, and C be the areas of the squares.
def A : ℝ := 6 * 6
def B : ℝ := 8 * 8
def C : ℝ := 10 * 10

-- The theorem statement
theorem sum_of_areas (A B C : ℝ) (hA : A = 36) (hB : B = 64) (hC : C = 100) : A + B = C :=
by {
  rw [hA, hB, hC],
  norm_num,
}

end sum_of_areas_l378_378796


namespace convert_cost_to_usd_l378_378967

def sandwich_cost_gbp : Float := 15.0
def conversion_rate : Float := 1.3

theorem convert_cost_to_usd :
  (Float.round ((sandwich_cost_gbp * conversion_rate) * 100) / 100) = 19.50 :=
by
  sorry

end convert_cost_to_usd_l378_378967


namespace circles_intersect_l378_378315

-- Definition of circles and line segment properties
noncomputable def circleM (a : ℝ) := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 2 * a * p.2 = 0}
noncomputable def circleN := {p : ℝ × ℝ | (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 = 1}
def lineL := {p : ℝ × ℝ | p.1 + p.2 = 0}

-- Condition: The circle M intersects the line at segment of length 2√2
def intersects_at_segment (a : ℝ) := 
  segment_length (circleM a) lineL = 2 * Real.sqrt 2

-- Proving the positional relationship
theorem circles_intersect (a : ℝ) (h_a : a > 0) (h_segment : intersects_at_segment a) :
  let R := a,
      centerM := (0, a),
      centerN := (1, 1),
      r := 1,
      d := dist centerM centerN in
  (dist centerM centerN) < (R + r) ∧ (dist centerM centerN) > (R - r) :=
by sorry

end circles_intersect_l378_378315


namespace smallest_base_10_integer_l378_378870

theorem smallest_base_10_integer (a b : ℕ) (ha : a > 2) (hb : b > 2) 
  (h1: 21_a = 2 * a + 1) (h2: 12_b = b + 2) : 2 * a + 1 = 7 :=
by 
  sorry

end smallest_base_10_integer_l378_378870


namespace tangent_line_at_point_is_correct_l378_378046

theorem tangent_line_at_point_is_correct :
  ∀ (x : ℝ), (x • exp x + 2 * x + 1) for given x = 0 := 
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ),
  f x = x * exp x + 2 * x + 1 ∧
  (∀ (p : ℝ × ℝ), p = (0, 1) →
    (∃ (m : ℝ), m = (derivative f) 0 ∧
    (∃ (b : ℝ), ∀ (y : ℝ), y = m * x + b ∧
    b = 1 ∧ m = 3))))

sorry

end tangent_line_at_point_is_correct_l378_378046


namespace modulus_complex_number_l378_378668

theorem modulus_complex_number : 
  ∀ (i : ℂ), i^2 = -1 → ∥(i - 2) * (2 * i + 1)∥ = 5 := 
by
  intros i h
  sorry

end modulus_complex_number_l378_378668


namespace union_of_sets_l378_378329

open Set

theorem union_of_sets (x : Nat) (hx : {0, x} ∩ {1, 2} = {1}) : {0, x} ∪ {1, 2} = {0, 1, 2} := 
by
  sorry

end union_of_sets_l378_378329


namespace tangent_line_eq_min_value_f_l378_378319

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x
noncomputable def f' (x : ℝ) : ℝ := ((x * Real.exp x) - Real.exp x) / (x^2)

theorem tangent_line_eq (x : ℝ) (h : x = 1) : deriv f x = 0 ∧ f x = Real.exp 1 := by
  sorry

theorem min_value_f (t x : ℝ) (ht : 0 < t) (h_inter : t ≤ x ∧ x ≤ t + 1) :
  (t ≥ 1 → ∃ y, y ∈ set.Icc t (t+1) ∧ f y = f t) ∧
  (t < 1 → ∃ y, y ∈ set.Icc t (t+1) ∧ f y = Real.exp 1) := by
  sorry

end tangent_line_eq_min_value_f_l378_378319


namespace time_to_cross_second_platform_l378_378163

-- Definition of the conditions
variables (l_train l_platform1 l_platform2 t1 : ℕ)
variable (v : ℕ)

-- The conditions given in the problem
def conditions : Prop :=
  l_train = 190 ∧
  l_platform1 = 140 ∧
  l_platform2 = 250 ∧
  t1 = 15 ∧
  v = (l_train + l_platform1) / t1

-- The statement to prove
theorem time_to_cross_second_platform
    (l_train l_platform1 l_platform2 t1 : ℕ)
    (v : ℕ)
    (h : conditions l_train l_platform1 l_platform2 t1 v) :
    (l_train + l_platform2) / v = 20 :=
  sorry

end time_to_cross_second_platform_l378_378163


namespace area_increase_is_35_42_percent_l378_378823

variables (L B : ℝ)
def original_area (L B : ℝ) : ℝ := L * B
def new_length (L : ℝ) : ℝ := 1.11 * L
def new_breadth (B : ℝ) : ℝ := 1.22 * B
def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B
def area_increase_percentage (L B : ℝ) : ℝ := (new_area L B - original_area L B) / original_area L B * 100

theorem area_increase_is_35_42_percent (L B : ℝ) : area_increase_percentage L B = 35.42 :=
by
  sorry

end area_increase_is_35_42_percent_l378_378823


namespace conic_sections_from_equation_l378_378999

-- Define the given equation
def original_eq := ∀ (x y : ℝ), y^4 - 8 * x^4 = 8 * y^2 - 4

-- Prove that the given equation represents the union of an ellipse and a hyperbola
theorem conic_sections_from_equation : 
  (original_eq) → 
  (∀ (x y : ℝ), (y^2 = 2 * real.sqrt 2 * x^2 + 4) ∨ (y^2 = -2 * real.sqrt 2 * x^2 + 4)) :=
by
  sorry

end conic_sections_from_equation_l378_378999


namespace tickets_not_went_to_concert_l378_378009

theorem tickets_not_went_to_concert :
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  remaining_after_start - (after_first_song + during_middle) = 20 := 
by
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  show remaining_after_start - (after_first_song + during_middle) = 20
  sorry

end tickets_not_went_to_concert_l378_378009


namespace q_at_4_l378_378202

def q (x : ℝ) : ℝ := |x - 3|^(1/3) + 3 * |x - 3|^(1/5) + 2 

theorem q_at_4 : q 4 = 6 := by
  sorry

end q_at_4_l378_378202


namespace length_of_c_proof_range_of_2c_minus_a_proof_l378_378761

noncomputable def triangleABC (A B C a b c : ℝ) :=
  ∃ (A B C : ℝ) (a b c : ℝ), 
    b = sqrt 3 ∧
    C = 5 * Real.pi / 6 ∧
    1 / 2 * a * sqrt 3 * Real.sin (5 * Real.pi / 6) = sqrt 3 / 2 ∧
    c = sqrt 13

theorem length_of_c_proof :
  ∀ (A B C a b c : ℝ), 
  triangleABC A B C a b c → 
  c = sqrt 13 := 
by
  intros
  sorry

noncomputable def triangleABCrange (A B C a b c : ℝ) :=
    ∃ (A B C : ℝ) (a b c : ℝ), 
    b = sqrt 3 ∧
    B = Real.pi / 3 ∧
    C ∈ set.Ioo 0 (2 * Real.pi / 3)

theorem range_of_2c_minus_a_proof :
  ∀ (A B C a b c : ℝ), 
  triangleABCrange A B C a b c → 
  2 * c - a ∈ set.Ioo (- sqrt 3) (2 * sqrt 3) :=
by
  intros
  sorry


end length_of_c_proof_range_of_2c_minus_a_proof_l378_378761


namespace sara_makes_cakes_on_20_weekdays_l378_378019

theorem sara_makes_cakes_on_20_weekdays (cakes_per_day : ℕ) (price_per_cake : ℕ) (total_amount_collected : ℕ) (weeks : ℕ) :
  (cakes_per_day = 4) →
  (price_per_cake = 8) →
  (total_amount_collected = 640) →
  (weeks = 4) →
  (total_amount_collected / price_per_cake / cakes_per_day) = 20 :=
by
  intros hp1 hp2 hp3 hp4
  rw [hp1, hp2, hp3, hp4]
  norm_num
  sorry

end sara_makes_cakes_on_20_weekdays_l378_378019


namespace barry_should_pay_l378_378177

def original_price : ℝ := 80
def discount_rate : ℝ := 0.15

theorem barry_should_pay:
  original_price * (1 - discount_rate) = 68 := 
by 
  -- Original price: 80
  -- Discount rate: 0.15
  -- Question: Final price after discount
  sorry

end barry_should_pay_l378_378177


namespace part_a_part_b_l378_378817

def shape_sequence := [ "triangle", "leftrightarrow", "diamond", "uparrow", "odot", "square" ]

def shape_at_position (n : Nat) : String :=
  shape_sequence.get! (n % shape_sequence.length)

theorem part_a (n : Nat) (h : n = 1000) : shape_at_position n = "uparrow" :=
by {
  dsimp [shape_at_position, shape_sequence];
  rw [h, Nat.mod_eq_of_lt];
  norm_num;
  exact by norm_num,
}

def diamond_position (k : Nat) := 6 * k + 3

theorem part_b (n k : Nat) (h1 : n = 1000) (h2 : k = n - 1) : diamond_position k = 5997 :=
by {
  dsimp [diamond_position];
  rw [h1, h2, Nat.mul_sub_left_distrib, Nat.sub_mul, Nat.mul_one];
  norm_num;
}

end part_a_part_b_l378_378817


namespace actual_distance_traveled_l378_378117

theorem actual_distance_traveled (T : ℝ) :
  ∀ D : ℝ, (D = 4 * T) → (D + 6 = 5 * T) → D = 24 :=
by
  intro D h1 h2
  sorry

end actual_distance_traveled_l378_378117


namespace product_odd_primes_mod_32_l378_378180

open Nat

theorem product_odd_primes_mod_32 : 
  let primes := [3, 5, 7, 11, 13] 
  let product := primes.foldl (· * ·) 1 
  product % 32 = 7 := 
by
  sorry

end product_odd_primes_mod_32_l378_378180


namespace arithmetic_sequence_max_sum_l378_378772

noncomputable def max_sum_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  (|a 3| = |a 11| ∧ 
   (∃ d : ℤ, d < 0 ∧ 
   (∀ n, a (n + 1) = a n + d) ∧ 
   (∀ m, S m = (m * (2 * a 1 + (m - 1) * d)) / 2)) →
   ((n = 6) ∨ (n = 7)))

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) :
  max_sum_n a S 6 ∨ max_sum_n a S 7 := sorry

end arithmetic_sequence_max_sum_l378_378772


namespace fraction_spent_on_fruits_l378_378973

theorem fraction_spent_on_fruits (M : ℕ) (hM : M = 24) :
  (M - (M / 3 + M / 6) - 6) / M = 1 / 4 :=
by
  sorry

end fraction_spent_on_fruits_l378_378973


namespace minimum_n_l378_378144

-- Define the conditions
variable (n d : ℕ)
variable (d_positive : d > 0)
variable (total_cost : ℕ)
variable (donation_cost : ℕ)
variable (remaining_radios_sold_cost : ℕ)
variable (profit : ℕ)

-- Condition that the dealer's total cost is d dollars
def total_cost_def : Prop := total_cost = d

-- Condition that the donation cost for two radios is half their total cost, which is equals to d / n
def donation_cost_def : Prop := donation_cost = d / n

-- Condition that the remaining radios are sold for a profit of $10 each,
-- where n - 2 radios are sold for (d / n) + 10
def remaining_radios_sold_cost_def : Prop :=
  remaining_radios_sold_cost = (n - 2) * (d / n + 10)

-- Condition that the total profit is 100 dollars
def profit_def : Prop := profit = remaining_radios_sold_cost + donation_cost - total_cost

theorem minimum_n (h1 : total_cost_def)
                  (h2 : donation_cost_def)
                  (h3 : remaining_radios_sold_cost_def)
                  (h4 : profit_def)
                  (h5 : profit = 100) : n = 13 := by
  sorry

end minimum_n_l378_378144


namespace construct_triangle_ABC_l378_378667

variables (O_b O_c : Point) (varrho_b : ℝ) (beta gamma : ℝ)

structure Triangle :=
  (A B C : Point)
  (a b c : ℝ)
  (parallel_to_plane : ∀ P Q : Point, parallel (segment A B) (plane P Q))
  (touches_spheres : ∀ X Y : Point, tangent (segment B C) (sphere O_b varrho_b) ∧ tangent (segment C A) (sphere O_b varrho_b))
  (extensions_touch_other_sphere : ∀ X Y : Point, tangent (extension (segment A B)) (sphere O_c (some_radius))) -- some_radius should be defined as per the geometric problem context
  (extensions_touch_both_spheres : ∀ X Y : Point, tangent (extension (segment A B)) (sphere O_b varrho_b) ∧ tangent (extension (segment A B)) (sphere O_c (some_radius))) -- some_radius should be defined as per the geometric problem context

theorem construct_triangle_ABC (O_b O_c : Point) (varrho_b : ℝ) (beta gamma : ℝ) : 
  ∃ (ABC : Triangle), 
  abc.parallel_to_plane ABC = ∀ P Q : Point, parallel (segment ABC.A ABC.B) (plane P Q) ∧ 
  abc.touches_spheres ABC = ∀ X Y : Point, tangent (segment ABC.B ABC.C) (sphere O_b varrho_b) ∧ tangent (segment ABC.C ABC.A) (sphere O_b varrho_b) ∧
  abc.extensions_touch_other_sphere ABC = ∀ X Y : Point, tangent (extension (segment ABC.A ABC.B)) (sphere O_c (some_radius)) ∧
  abc.extensions_touch_both_spheres ABC = ∀ X Y : Point, tangent (extension (segment ABC.A ABC.B)) (sphere O_b varrho_b) ∧ tangent (extension (segment ABC.A ABC.B)) (sphere O_c (some_radius))
  sorry

end construct_triangle_ABC_l378_378667


namespace sqrt_sum_trig_l378_378600

theorem sqrt_sum_trig (θ : ℝ) (hθ : θ = π / 8 ∨ θ = 3 * π / 8 ∨ θ = 5 * π / 8)
  (h_poly : 128 * sin(θ)^8 - 256 * sin(θ)^6 + 160 * sin(θ)^4 - 32 * sin(θ)^2 = 0) :
  sqrt ((2 - sin (π / 8) ^ 2) * (2 - sin (3 * π / 8) ^ 2) * (2 - sin (5 * π / 8) ^ 2)) = 1 :=
sorry

end sqrt_sum_trig_l378_378600


namespace area_enclosed_by_graph_l378_378862

theorem area_enclosed_by_graph (x y : ℝ) (h : abs (5 * x) + abs (3 * y) = 15) : 
  ∃ (area : ℝ), area = 30 :=
sorry

end area_enclosed_by_graph_l378_378862


namespace probability_A_wins_black_white_pair_game_l378_378370

theorem probability_A_wins_black_white_pair_game :
  let outcomes : List (Bool × Bool × Bool) := [
    (false, false, false), (false, false, true), (false, true, false),
    (false, true, true), (true, false, false), (true, false, true),
    (true, true, false), (true, true, true)
  ],
  let winning_outcomes_for_A : List (Bool × Bool × Bool) := [
    (true, false, false), (false, true, true)
  ],
  (winning_outcomes_for_A.length : ℝ) / (outcomes.length : ℝ) = 0.25 :=
by
  sorry

end probability_A_wins_black_white_pair_game_l378_378370


namespace Sn_result_l378_378658

noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then -2 * x^2 + 4 * x else 2 * f (x + 2)

def a_n (n : ℕ) (hp : n > 0) : ℝ :=
  if n = 1 then f 1 else 2 * a_n (n - 1) (nat.sub_pos_of_lt hp)

def S_n (n : ℕ) (hp : n > 0) : ℝ :=
  (finset.range n).sum (λ i, a_n (i + 1) (nat.succ_pos _))

theorem Sn_result (n : ℕ) (hp : n > 0) : S_n n hp = 4 - 1 / 2 ^ (n - 2) :=
by
  sorry

end Sn_result_l378_378658


namespace coffee_price_increase_l378_378839

theorem coffee_price_increase (price_first_quarter price_fourth_quarter : ℕ) 
  (h_first : price_first_quarter = 40) (h_fourth : price_fourth_quarter = 60) : 
  ((price_fourth_quarter - price_first_quarter) * 100) / price_first_quarter = 50 := 
by
  -- proof would proceed here
  sorry

end coffee_price_increase_l378_378839


namespace probability_event_A_l378_378032

noncomputable theory

open MeasureTheory

theorem probability_event_A : 
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (1 / 2 * x + y ≥ 1) → 
  (@measure_space volume).measure_set {p : ℝ × ℝ | 1 / 2 * p.1 + p.2 ≥ 1}
    .to_outer_measure
    (set.univ) = 1 / 4 :=
by
  sorry

end probability_event_A_l378_378032


namespace original_children_count_l378_378026

theorem original_children_count (x : ℕ) (h1 : 46800 / x + 1950 = 46800 / (x - 2))
    : x = 8 :=
sorry

end original_children_count_l378_378026


namespace fantasy_gala_handshakes_l378_378079

theorem fantasy_gala_handshakes
    (gremlins imps : ℕ)
    (gremlin_handshakes : ℕ)
    (imp_handshakes : ℕ)
    (imp_gremlin_handshakes : ℕ)
    (total_handshakes : ℕ)
    (h1 : gremlins = 30)
    (h2 : imps = 20)
    (h3 : gremlin_handshakes = (30 * 29) / 2)
    (h4 : imp_handshakes = (20 * 5) / 2)
    (h5 : imp_gremlin_handshakes = 20 * 30)
    (h6 : total_handshakes = gremlin_handshakes + imp_handshakes + imp_gremlin_handshakes) :
    total_handshakes = 1085 := by
    sorry

end fantasy_gala_handshakes_l378_378079


namespace coefficient_x7_expansion_l378_378813

theorem coefficient_x7_expansion : 
  (coeff (expand (x^2 - 1/x)^8) 7 = -56) :=
  sorry

end coefficient_x7_expansion_l378_378813


namespace three_roads_different_colors_l378_378360

theorem three_roads_different_colors {n : ℕ}
  (c₁ c₂ c₃ : ℕ)
  (h1 : ∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (h2 : ∀ i, i ∈ {1, 2, 3} → (i = 1 ∨ i = 2 ∨ i = 3))
  (h3 : n ≠ 0)
  (h4 : ∃ n, (n + c₁ + c₂ + c₃) % 2 = 0)
  : c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₃ ≠ c₁ :=
sorry

end three_roads_different_colors_l378_378360


namespace find_standard_equation_of_ellipse_min_area_of_quadrilateral_l378_378650

-- Definitions and conditions
def A : ℝ×ℝ := (-real.sqrt 2 / 2, real.sqrt 3 / 2)
def e : ℝ := real.sqrt 2 / 2
def parabola_y_squared := λ x : ℝ, (real.sqrt (4 * x)) 

def ellipse (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def on_parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Question 1
theorem find_standard_equation_of_ellipse (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b < a)
  (h4 : e = c / a) (h5 : c = real.sqrt (a^2 - b^2)) (h6 : ellipse a b (A.1) (A.2)) :
  ∃ a b, ellipse a b = λ x y, x^2 / 2 + y^2 = 1 := 
sorry

-- Question 2
theorem min_area_of_quadrilateral (a b : ℝ) 
  (ellipse_eq : ellipse a b = λ x y, x^2 / 2 + y^2 = 1) 
  (collinear_MNF2 : ∀ M N F2 : ℝ×ℝ, on_parabola M.1 M.2 → on_parabola N.1 N.2 → collinear F2 M N)
  (collinear_PQF2 : ∀ P Q F2 : ℝ×ℝ, ellipse a b P.1 P.2 → ellipse a b Q.1 Q.2 → collinear F2 P Q)
  (perpendicular_PQ_MN : ∀ M N P Q : ℝ×ℝ, collinear_MNF2 M N → collinear_PQF2 P Q → perpendicular PQ MN) :
  ∃ S : ℝ, S = 4 * real.sqrt 2 :=
sorry

end find_standard_equation_of_ellipse_min_area_of_quadrilateral_l378_378650


namespace greatest_possible_x_l378_378098

theorem greatest_possible_x (x : ℕ) (h : x^4 / x^2 < 18) : x ≤ 4 :=
sorry

end greatest_possible_x_l378_378098


namespace greatest_x_value_l378_378100

theorem greatest_x_value : 
  ∀ x : ℝ, (∀ y : ℝ, (y = 4*x - 16) → (∀ z : ℝ, (z = 3*x - 4) → (y/z)^2 + y/z = 6)) → 
  x ≤ 28/13 :=
begin
  sorry
end

end greatest_x_value_l378_378100


namespace sum_of_n_and_k_l378_378047

open Nat

theorem sum_of_n_and_k (n k : ℕ)
  (h1 : 2 = n - 3 * k)
  (h2 : 8 = 2 * n - 5 * k) :
  n + k = 18 :=
sorry

end sum_of_n_and_k_l378_378047


namespace smoothie_mix_packet_size_l378_378964

theorem smoothie_mix_packet_size :
  (∀ (total_smoothie_ounces packets_needed : ℕ),
     total_smoothie_ounces = 150 * 12 →
     packets_needed = 180 →
     total_smoothie_ounces / packets_needed = 10) :=
begin
  intros total_smoothie_ounces packets_needed h1 h2,
  rw [h1, h2],
  norm_num,
end

end smoothie_mix_packet_size_l378_378964


namespace log_base_frac_l378_378231

theorem log_base_frac (x : ℝ) : log (1/4) 16 = x → x = -2 := by
  sorry

end log_base_frac_l378_378231


namespace vertex_coordinates_l378_378040

-- Definition of the quadratic function and relevant coefficients
def a : ℝ := 3
def b : ℝ := -6
def c : ℝ := 5
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Definition of the vertex coordinates
def vertex_x : ℝ := -b / (2 * a)
def vertex_y : ℝ := (4 * a * c - b^2) / (4 * a)

-- Theorem stating that the vertex of the quadratic function has coordinates (1, 2)
theorem vertex_coordinates :
  (vertex_x, vertex_y) = (1, 2) := by
sorry

end vertex_coordinates_l378_378040


namespace log_base_fraction_l378_378221

theorem log_base_fraction : ∀ (a b : ℝ) (x : ℝ), 16 = (4:ℝ)^2 ∧ (1 / 4:ℝ) = 4^(-1) → log (1 / 4) 16 = -2 :=
begin
  intros a b x h,
  -- Skipping the proof by adding sorry
  sorry,
end

end log_base_fraction_l378_378221


namespace log_base_one_four_of_sixteen_l378_378241

theorem log_base_one_four_of_sixteen : log (1 / 4) 16 = -2 := by
  sorry

end log_base_one_four_of_sixteen_l378_378241


namespace at_least_two_of_three_equations_have_solutions_l378_378653

theorem at_least_two_of_three_equations_have_solutions
  (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ x : ℝ, (x - a) * (x - b) = x - c ∨ (x - b) * (x - c) = x - a ∨ (x - c) * (x - a) = x - b := 
sorry

end at_least_two_of_three_equations_have_solutions_l378_378653


namespace problem_1_problem_2_l378_378328

-- Definition of set A with m as a parameter
def setA (m : ℕ) : Set ℝ := {0, 1, Real.logBase 3 (m^2 + 2), m^2 - 3 * m}

-- Definition of set C with n as a parameter
def setC (n : ℕ) : Set ℝ := {-1, 1, n}

-- The mapping f: x ↦ 2x - 3
def f (x : ℝ) : ℝ := 2 * x - 3

-- Given conditions for the first question
def A_m_5 : Set ℝ := setA 5
def C_n_3 : Set ℝ := setC 3

-- Lean statement for the first question
theorem problem_1 : A_m_5 ∩ C_n_3 = {1, 3} :=
sorry

-- Given condition for the second question
axiom A_contains_neg2 (m : ℕ) : -2 ∈ setA m

-- Lean statement for the second question
theorem problem_2 : ∀ m : ℕ, A_contains_neg2 m → m = 2 :=
sorry

end problem_1_problem_2_l378_378328


namespace clermontville_residents_l378_378382

/-- Math problem: Determining the number of residents who watch all five shows. --/

theorem clermontville_residents (
  (total_residents : ℕ) 
  (watch_is : ℕ) (watch_ll : ℕ) (watch_me : ℕ) (watch_mm : ℕ) (watch_ssa : ℕ)
  (exactly_one : ℕ) (exactly_two : ℕ) (exactly_three : ℕ) (exactly_four : ℕ)
  (total_residents = 1000)
  (watch_is = 250)
  (watch_ll = 300)
  (watch_me = 400)
  (watch_mm = 200)
  (watch_ssa = 150)
  (exactly_one = 300)
  (exactly_two = 280)
  (exactly_three = 120)
  (exactly_four = 50) :
  ∃ (watch_all_five : ℕ), watch_all_five = 250 := 
sorry

end clermontville_residents_l378_378382


namespace justin_and_tim_play_same_game_210_times_l378_378787

def number_of_games_with_justin_and_tim : ℕ :=
  have num_players : ℕ := 12
  have game_size : ℕ := 6
  have justin_and_tim_fixed : ℕ := 2
  have remaining_players : ℕ := num_players - justin_and_tim_fixed
  have players_to_choose : ℕ := game_size - justin_and_tim_fixed
  Nat.choose remaining_players players_to_choose

theorem justin_and_tim_play_same_game_210_times :
  number_of_games_with_justin_and_tim = 210 :=
by sorry

end justin_and_tim_play_same_game_210_times_l378_378787


namespace find_projection_matrix_pair_l378_378583

open Matrix

def is_projection_matrix {α : Type*} [CommRing α] (P : Matrix (Fin 2) (Fin 2) α) : Prop :=
  P ⬝ P = P

theorem find_projection_matrix_pair :
  ∃ (a c : ℚ), is_projection_matrix (λ i j, ![![a, 3/7], ![c, 4/7]]) ∧ (a, c) = (1, 3 / 7) :=
by
  let a := (1 : ℚ)
  let c := (3 / 7 : ℚ)
  use a, c
  rw [is_projection_matrix]
  simp only [mul_comm, Fin.val_zero, Fin.val_one, Matrix.mul_eq_mul, Matrix.dot_product, Matrix.mul_apply, Matrix.cons_val, Matrix.head_cons, Matrix.tail_cons, Matrix.empty_eq]
  sorry

end find_projection_matrix_pair_l378_378583


namespace max_planes_determined_l378_378148

-- Definitions for conditions
variables (Point Line Plane : Type)
variables (l : Line) (A B C : Point)
variables (contains : Point → Line → Prop)
variables (plane_contains_points : Plane → Point → Point → Point → Prop)
variables (plane_contains_line_and_point : Plane → Line → Point → Prop)
variables (non_collinear : Point → Point → Point → Prop)
variables (not_on_line : Point → Line → Prop)

-- Hypotheses based on the conditions
axiom three_non_collinear_points : non_collinear A B C
axiom point_not_on_line (P : Point) : not_on_line P l

-- Goal: Prove that the number of planes is 4
theorem max_planes_determined : 
  ∃ total_planes : ℕ, total_planes = 4 :=
sorry

end max_planes_determined_l378_378148


namespace scientific_notation_113700_l378_378831

theorem scientific_notation_113700 : (113700 : ℝ) = 1.137 * 10^5 :=
by
  sorry

end scientific_notation_113700_l378_378831


namespace phone_calls_to_reach_Davina_l378_378029

theorem phone_calls_to_reach_Davina : 
  (∀ (a b : ℕ), (0 ≤ a ∧ a < 10) ∧ (0 ≤ b ∧ b < 10)) → (least_num_calls : ℕ) = 100 :=
by
  sorry

end phone_calls_to_reach_Davina_l378_378029


namespace reggie_games_lost_l378_378448

-- Define the necessary conditions
def initial_marbles : ℕ := 100
def bet_per_game : ℕ := 10
def marbles_after_games : ℕ := 90
def total_games : ℕ := 9

-- Define the proof problem statement
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / bet_per_game = 1 := by
  sorry

end reggie_games_lost_l378_378448


namespace certain_number_condition_l378_378354

theorem certain_number_condition (x y z : ℤ) (N : ℤ)
  (hx : Even x) (hy : Odd y) (hz : Odd z)
  (hxy : x < y) (hyz : y < z)
  (h1 : y - x > N)
  (h2 : z - x = 7) :
  N < 3 := by
  sorry

end certain_number_condition_l378_378354


namespace minimum_questions_to_find_two_white_balls_l378_378493

theorem minimum_questions_to_find_two_white_balls (n : ℕ) (boxes : Fin n → Prop) (wballs_even : Even (Finset.filter boxes (Finset.univ : Finset (Fin n)).card)) (questions : (Fin n × Fin n) → Prop) :
  n = 2004 → (∃ q : ℕ, q = 4005 ∧ (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 2004 → questions (⟨i, _⟩, ⟨j, _⟩))) :=
sorry

end minimum_questions_to_find_two_white_balls_l378_378493


namespace rho_squared_leq_four_thirds_l378_378403

-- Define the mathematical framework and constants
variable (a b : ℝ) (x y : ℝ)

-- Assume the conditions from the problem
theorem rho_squared_leq_four_thirds
  (h_pos_a: a > 0)
  (h_pos_b: b > 0)
  (h_geq: a ≥ b)
  (h_system: a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b - y)^2)
  (h_tri: a^2 = x^2 + b^2 ∧ y^2 = b^2 - (x - b)^2)
  (h_bounds: 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b)
  : (a / b)^2 ≤ 4 / 3 :=
begin
  sorry
end

end rho_squared_leq_four_thirds_l378_378403


namespace positive_sine_function_range_l378_378730

theorem positive_sine_function_range (x r x0 y0 : ℝ) (h1 : r > 0) 
  (h2 : x0 = r * cos x) (h3 : y0 = r * sin x) : 
  let sos := (y0 + x0) / r in 
  ∀ x : ℝ, sos = sin x + cos x → 
    ∀ y : sos, y ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
by
  intros _ _ _
  sorry

end positive_sine_function_range_l378_378730


namespace kelly_needs_more_apples_l378_378391

theorem kelly_needs_more_apples (initial_apples : ℕ) (total_apples : ℕ) (needed_apples : ℕ) :
  initial_apples = 128 → total_apples = 250 → needed_apples = total_apples - initial_apples → needed_apples = 122 :=
by
  intros h_initial h_total h_needed
  rw [h_initial, h_total] at h_needed
  exact h_needed

end kelly_needs_more_apples_l378_378391


namespace part1_part2_l378_378209

noncomputable def determinant (a b c d : ℤ) : ℤ :=
  a * d - b * c

-- Lean statement for Question (1)
theorem part1 :
  determinant 2022 2023 2021 2022 = 1 :=
by sorry

-- Lean statement for Question (2)
theorem part2 (m : ℤ) :
  determinant (m + 2) (m - 2) (m - 2) (m + 2) = 32 → m = 4 :=
by sorry

end part1_part2_l378_378209


namespace good_apples_count_l378_378714

theorem good_apples_count (total_apples : ℕ) (rotten_percentage : ℝ) (good_apples : ℕ) (h1 : total_apples = 75) (h2 : rotten_percentage = 0.12) :
  good_apples = (1 - rotten_percentage) * total_apples := by
  sorry

end good_apples_count_l378_378714


namespace average_water_consumption_l378_378508

theorem average_water_consumption :
  let january := 12.5
  let february := 13.8
  let march := 13.7
  let april := 11.4
  let may := 12.1
  (january + february + march + april + may) / 5 = 12.7 := 
by {
  let january := 12.5
  let february := 13.8
  let march := 13.7
  let april := 11.4
  let may := 12.1
  have sum_eq : january + february + march + april + may = 63.5,
  {
    calc
      january + february + march + april + may
        = 12.5 + 13.8 + 13.7 + 11.4 + 12.1 : by simp [january, february, march, april, may]
    ... = 26.3 + 13.7 : by norm_num
    ... = 40.0 + 11.4 : by norm_num
    ... = 51.4 + 12.1 : by norm_num
    ... = 63.5 : by norm_num
  },
  show (january + february + march + april + may) / 5 = 12.7, from
    calc
      (january + february + march + april + may) / 5
        = 63.5 / 5 : by rw sum_eq
    ... = 12.7 : by norm_num
}

end average_water_consumption_l378_378508


namespace hyperbola_problem_l378_378646

-- Define the given conditions
def is_hyperbola_with_focus_directrix (C : ℝ → ℝ → Prop)
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (F : ℝ × ℝ) (directrix : ℝ → Prop) : Prop :=
  (∀ x y, C x y ↔ (x^2 / a^2 - y^2 / b^2 = 1)) ∧
  (F = (2, 0)) ∧
  (directrix = λ x, x = 3/2)

-- Define the first question's solution
def standard_eqn_asymptotes (C : ℝ → ℝ → Prop) : Prop :=
  (∀ x y, C x y ↔ (x^2 / 3 - y^2 = 1)) ∧
  (∀ x, (y = (1 / real.sqrt 3) * x ∨ y = -(1 / real.sqrt 3) * x))

-- Define the second question's solution
def shared_asymptotes_eqn (C : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  (P = (real.sqrt 3, 2)) → (∀ x y, (C x y ↔ (y^2 / 3 - x^2 / 9 = 1)))

-- The main theorem statement
theorem hyperbola_problem (C : ℝ → ℝ → Prop)
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (directrix : ℝ → Prop) :
  is_hyperbola_with_focus_directrix C a b ha hb (2, 0) directrix →
  standard_eqn_asymptotes C ∧ shared_asymptotes_eqn (λ x y, C x y ∧ P x y) (real.sqrt 3, 2) :=
by
  intros
  -- sorry is a placeholder for the proof
  sorry

end hyperbola_problem_l378_378646


namespace log_base_one_four_of_sixteen_l378_378239

theorem log_base_one_four_of_sixteen : log (1 / 4) 16 = -2 := by
  sorry

end log_base_one_four_of_sixteen_l378_378239


namespace exp1_eval_exp2_eval_l378_378179

-- Define the conditions for Expression (1)
def exp1_c1 : 2^0 = 1 := by sorry
def exp1_c2 : 4^{-1} = 1 / 4 := by sorry
def exp1_c3 : (-1)^(2009) = -1 := by sorry
def exp1_c4 : (-1 / 2)^{-2} = 1 / 4 := by sorry

-- Proof problem for Expression (1)
theorem exp1_eval (h1 : 2^0 = 1) (h2 : 4^{-1} = 1 / 4) (h3 : (-1)^(2009) = -1) (h4 : (-1 / 2)^{-2} = 1 / 4) : 
  -2^0 + 4^{-1} * (-1)^(2009) * (-1 / 2)^{-2} = -17 / 16 :=
by sorry

-- Define the conditions for Expression (2)
def exp2_c1 (x : ℝ) : (x + 1)^2 = x^2 + 2*x + 1 := by sorry
def exp2_c2 (x : ℝ) : (x - 1) * (x + 2) = x^2 - x + 2*x - 2 := by sorry

-- Proof problem for Expression (2)
theorem exp2_eval (x : ℝ) (h1 : (x + 1)^2 = x^2 + 2*x + 1) (h2 : (x - 1) * (x + 2) = x^2 - x + 2*x - 2) : 
  (x + 1)^2 - (x - 1) * (x + 2) = 3*x - 1 :=
by sorry

end exp1_eval_exp2_eval_l378_378179


namespace find_pointC_l378_378660

-- Define points A and B
def pointA : (ℝ × ℝ) := (-4, 2)
def pointB : (ℝ × ℝ) := (3, 1)

-- Define the point C
def pointC : (ℝ × ℝ) := (2, 4)

-- Define the line y = 2x which is the angle bisector of ∠C
def line_y_2x : ℝ → ℝ := λ x, 2 * x

/- The coordinates of point C are (2, 4) given that y = 2x is the angle bisector
   of ∠C in ΔABC with A = (-4, 2) and B = (3, 1) -/
theorem find_pointC 
  (A B C : (ℝ × ℝ))
  (hA : A = pointA)
  (hB : B = pointB)
  (hC : C = pointC) 
  (bisector : (y: ℝ) -> ∃ x, y = line_y_2x x) :
  C = (2, 4) :=
sorry

end find_pointC_l378_378660


namespace max_n_exists_convex_ngon_with_integer_diagonals_l378_378287

theorem max_n_exists_convex_ngon_with_integer_diagonals :
  ∀ n : ℕ, (∃ (polygon : list ℕ) (hconvex : convex polygon) (hside1 : 1 ∈ polygon) 
    (hdiagonals_integer : ∀ (i j : ℕ), i ≠ j → i ≠ j + 1 → j ≠ i + 1 → is_integer_diagonal (polygon !! i) (polygon !! j)),
    ∀ m : ℕ, m ≥ n → m ≤ 5 := 
begin
    sorry
end

end max_n_exists_convex_ngon_with_integer_diagonals_l378_378287


namespace green_light_probability_l378_378900

def red_duration : ℕ := 30
def green_duration : ℕ := 25
def yellow_duration : ℕ := 5

def total_cycle : ℕ := red_duration + green_duration + yellow_duration
def green_probability : ℚ := green_duration / total_cycle

theorem green_light_probability :
  green_probability = 5 / 12 := by
  sorry

end green_light_probability_l378_378900


namespace dig_site_date_l378_378169

theorem dig_site_date (S1 S2 S3 S4 : ℕ) (S2_bc : S2 = 852) 
  (h1 : S1 = S2 - 352) 
  (h2 : S3 = S1 + 3700) 
  (h3 : S4 = 2 * S3) : 
  S4 = 6400 :=
by sorry

end dig_site_date_l378_378169


namespace find_x_plus_y_l378_378757

-- Define the points A, B, and C with given conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 1, y := 1}
def C : Point := {x := 2, y := 4}

-- Define what it means for C to divide AB in the ratio 2:1
open Point

def divides_in_ratio (A B C : Point) (r₁ r₂ : ℝ) :=
  (C.x = (r₁ * A.x + r₂ * B.x) / (r₁ + r₂))
  ∧ (C.y = (r₁ * A.y + r₂ * B.y) / (r₁ + r₂))

-- Prove that x + y = 8 given the conditions
theorem find_x_plus_y {x y : ℝ} (B : Point) (H_B : B = {x := x, y := y}) :
  divides_in_ratio A B C 2 1 →
  x + y = 8 :=
by
  intro h
  sorry

end find_x_plus_y_l378_378757


namespace _l378_378916

noncomputable def is_not_negative (x : ℝ) : Prop :=
  x >= 0

noncomputable def square_is_not_positive (x : ℝ) : Prop :=
  x * x <= 0

noncomputable def square_is_positive (x : ℝ) : Prop :=
  x * x > 0

noncomputable theorem equivalence_statement : 
  (∀ x : ℝ, is_not_negative x → ¬ square_is_not_positive x) ↔ (∀ x : ℝ, ¬ square_is_positive x → ¬ is_not_negative x) := 
by
  sorry

end _l378_378916


namespace avg_annual_growth_rate_l378_378965

variable (x : ℝ)

/-- Initial GDP in 2020 is 43903.89 billion yuan and GDP in 2022 is 53109.85 billion yuan. 
    Prove that the average annual growth rate x satisfies the equation 43903.89 * (1 + x)^2 = 53109.85 -/
theorem avg_annual_growth_rate (x : ℝ) :
  43903.89 * (1 + x)^2 = 53109.85 :=
sorry

end avg_annual_growth_rate_l378_378965


namespace log_base_fraction_l378_378222

theorem log_base_fraction : ∀ (a b : ℝ) (x : ℝ), 16 = (4:ℝ)^2 ∧ (1 / 4:ℝ) = 4^(-1) → log (1 / 4) 16 = -2 :=
begin
  intros a b x h,
  -- Skipping the proof by adding sorry
  sorry,
end

end log_base_fraction_l378_378222


namespace functional_equation_solution_l378_378998

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1 ∨ f x = -x - 1) :=
by
  sorry

end functional_equation_solution_l378_378998


namespace rearrange_six_digit_number_divisible_by_37_l378_378158

theorem rearrange_six_digit_number_divisible_by_37
  (N : ℕ)
  (h1 : 100000 ≤ N ∧ N < 1000000)
  (h2 : N % 37 = 0)
  (h3 : ∃ d1 d2, d1 ≠ d2 ∧ List.mem d1 (digits 10 N) ∧ List.mem d2 (digits 10 N))
  (h4 : ∃ d, List.head (digits 10 N) = some d ∧ d ≠ 0)
  (h5 : ∃ d, List.nth (digits 10 N) 3 = some d ∧ d ≠ 0)
  : ∃ N' : ℕ, N' ≠ N ∧ (List.mem N' (permutations N)) ∧ N' % 37 = 0 ∧ ∃ d, List.head (digits 10 N') = some d ∧ d ≠ 0 :=
sorry

end rearrange_six_digit_number_divisible_by_37_l378_378158


namespace length_increase_100_l378_378013

theorem length_increase_100 (n : ℕ) (h : (n + 2) / 2 = 100) : n = 198 :=
sorry

end length_increase_100_l378_378013


namespace circumradius_of_triangle_ABC_is_7_sqrt_3_l378_378037

variables {A B C D : Type} [EuclideanGeometry ℝ]

-- Define the properties of the triangle ABC and the conditions.
variables (triangle_ABC : Triangle A B C) 
          (angle_bisector_bisects : Bisector A D BC ∧ D ∈ BC) 
          (circle_radius : ∀ (r : ℝ), r = 35 ∧ Circle r (center B) A D)
          (AB_sq_minus_AC_sq : AB^2 - AC^2 = 216)
          (area_ABC : Area (Triangle A B C) = 90 * sqrt 3)

theorem circumradius_of_triangle_ABC_is_7_sqrt_3 
    (R : ℝ) : 
    Circumradius (Triangle A B C) = R → 
    R = 7 * sqrt 3 := 
sorry

end circumradius_of_triangle_ABC_is_7_sqrt_3_l378_378037


namespace log_base_one_fourth_of_sixteen_l378_378258

theorem log_base_one_fourth_of_sixteen :
  log (1 / 4 : ℝ) (16 : ℝ) = -2 :=
sorry

end log_base_one_fourth_of_sixteen_l378_378258


namespace card_statements_true_l378_378780

def statement1 (statements : Fin 5 → Prop) : Prop :=
  ∃! i, i < 5 ∧ statements i

def statement2 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j, i < 5 ∧ j < 5 ∧ i ≠ j ∧ statements i ∧ statements j) ∧ ¬(∃ h k l, h < 5 ∧ k < 5 ∧ l < 5 ∧ h ≠ k ∧ h ≠ l ∧ k ≠ l ∧ statements h ∧ statements k ∧ statements l)

def statement3 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j k, i < 5 ∧ j < 5 ∧ k < 5 ∧ i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ statements i ∧ statements j ∧ statements k) ∧ ¬(∃ a b c d, a < 5 ∧ b < 5 ∧ c < 5 ∧ d < 5 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ statements a ∧ statements b ∧ statements c ∧ statements d)

def statement4 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j k l, i < 5 ∧ j < 5 ∧ k < 5 ∧ l < 5 ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ statements i ∧ statements j ∧ statements k ∧ statements l) ∧ ¬(∃ a b c d e, a < 5 ∧ b < 5 ∧ c < 5 ∧ d < 5 ∧ e < 5 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ statements a ∧ statements b ∧ statements c ∧ statements d ∧ statements e)

def statement5 (statements : Fin 5 → Prop) : Prop :=
  ∀ i, i < 5 → statements i

theorem card_statements_true : ∃ (statements : Fin 5 → Prop), 
  statement1 statements ∨ statement2 statements ∨ statement3 statements ∨ statement4 statements ∨ statement5 statements 
  ∧ statement3 statements := 
sorry

end card_statements_true_l378_378780


namespace neg_p_necessary_but_not_sufficient_for_neg_q_l378_378302

variable (x : ℝ)

def p : Prop := x^2 < 1
def q : Prop := x < 1

theorem neg_p_necessary_but_not_sufficient_for_neg_q : (¬ p → ¬ q) ∧ ¬ (¬ q → ¬ p) :=
by
  sorry

end neg_p_necessary_but_not_sufficient_for_neg_q_l378_378302


namespace simplify_fraction_l378_378453

theorem simplify_fraction (a b c : ℕ) (h1 : 222 = 2 * 111) (h2 : 999 = 3 * 333) (h3 : 111 = 3 * 37) :
  (222 / 999 * 111) = 74 :=
by
  sorry

end simplify_fraction_l378_378453


namespace sequence_limit_sum_l378_378061

noncomputable def x_seq : ℕ → ℝ
| 0       := 150
| (k + 1) := x_seq k ^ 2 - x_seq k

theorem sequence_limit_sum :
  tendsto (λ n, ∑ k in finset.range n, 1 / (x_seq k + 1)) at_top (𝓝 (1 / 150)) :=
sorry

end sequence_limit_sum_l378_378061


namespace max_candies_eaten_l378_378984

def initial_state := (10, 10, 10)
def labels := (4, 7, 10)

def operation (state : ℕ × ℕ × ℕ) (label : ℕ) : ℕ × ℕ × ℕ :=
  let taken := if label = 4 then (state.1 - 4, state.2, state.3)
               else if label = 7 then (state.1, state.2 - 7, state.3)
               else (state.1, state.2, state.3 - 10) in
  let after_eating := if label = 4 then (taken.1 + 1, taken.2, taken.3)
                      else if label = 7 then (taken.1, taken.2 + 4, taken.3)
                      else (taken.1, taken.2, taken.3 + 7) in
  after_eating

theorem max_candies_eaten : 
  ∀ (state : ℕ × ℕ × ℕ), state = initial_state → (∃ n, ∀ label ∈ [4, 7, 10], operation state label → n * 3) ∧ n = 9 → 
  max_candies_eaten = 27 := by 
  sorry

end max_candies_eaten_l378_378984


namespace max_jackets_purchaseable_l378_378743

def original_price_shirt := 12
def discount_10 := 0.10
def discount_15 := 0.15
def discount_20 := 0.20
def shirts_10 := 5
def shirts_15 := 7
def shirts_20 := 3

def original_price_pant := 20
def pants := 4
def bogo_discount := 0.50

def original_price_jacket := 30
def sales_tax := 0.10
def max_jackets := 5

def budget := 350

noncomputable def total_cost_shirts :=
  (shirts_10 * original_price_shirt * (1 - discount_10)) +
  (shirts_15 * original_price_shirt * (1 - discount_15)) +
  (shirts_20 * original_price_shirt * (1 - discount_20))

noncomputable def total_cost_pants :=
  (pants // 2 * original_price_pant) +
  (pants % 2 * original_price_pant * (1 + bogo_discount))

noncomputable def total_cost_jacket :=
  original_price_jacket * (1 + sales_tax)

noncomputable def remaining_budget :=
  budget - total_cost_shirts - total_cost_pants

noncomputable def max_jackets_affordable :=
  remaining_budget // total_cost_jacket

theorem max_jackets_purchaseable :
  (remaining_budget // total_cost_jacket) = 4 :=
sorry

end max_jackets_purchaseable_l378_378743


namespace ant_reaches_end_at_seventh_minute_l378_378558

/--
Given:
1. The length of the rubber band is initially 4 inches.
2. The ant starts at the left end and walks 1 inch every minute.
3. After every minute, the rubber band is uniformly stretched by 1 inch.

Prove:
The ant reaches the right end of the rubber band during the 7th minute.
-/
theorem ant_reaches_end_at_seventh_minute :
  let length_initial := 4
  let ant_position_minute (n : ℕ) : ℕ := n -- Ant walks 1 inch every minute
  let band_length_minute (n : ℕ) := length_initial + n -- Band increases by 1 inch every minute
  let fractional_distance_minute (n : ℕ) := ant_position_minute n / (band_length_minute n : ℚ)
  ∑ k in Finset.range 7, fractional_distance_minute (k + 4) > 1 :=
by
  sorry

end ant_reaches_end_at_seventh_minute_l378_378558


namespace unclaimed_candy_fraction_l378_378167

theorem unclaimed_candy_fraction:
  (ratio4 ratio3 ratio2 : ℕ) (total_candy : ℚ) 
  (candy_for_charlie candy_for_alice candy_for_bob : ℚ) :
  ratio4 = 4 → ratio3 = 3 → ratio2 = 2 →
  candy_for_charlie = (2/9) * total_candy →
  candy_for_alice = (4/9) * (1 - (2/9)) * total_candy →
  candy_for_bob = (1/3) * (1 - (2/9) - (4/9) * (7/9)) * total_candy →
  (total_candy - (candy_for_charlie + candy_for_alice + candy_for_bob)) = (2/9) * total_candy :=
by
  intros
  sorry

end unclaimed_candy_fraction_l378_378167


namespace simplify_and_evaluate_l378_378456

theorem simplify_and_evaluate :
  let x := (-1 : ℚ) / 2
  3 * x^2 - (5 * x - 3 * (2 * x - 1) + 7 * x^2) = -9 / 2 :=
by
  let x : ℚ := (-1 : ℚ) / 2
  sorry

end simplify_and_evaluate_l378_378456


namespace product_of_first_2017_terms_l378_378648

noncomputable def a_sequence : ℕ → ℝ
| 0       := 2
| (n + 1) := (1 + a_sequence n) / (1 - a_sequence n)

theorem product_of_first_2017_terms : 
  ∏ i in finset.range 2017, a_sequence i = 2 :=
sorry

end product_of_first_2017_terms_l378_378648


namespace sin2alpha_cos2alpha_eq_l378_378074

open Real

theorem sin2alpha_cos2alpha_eq (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 2 = log a (4 - 3) + 2) :
  let α := Real.atan (2 / 1) in
  sin (2 * α) + cos (2 * α) = 7 / 5 :=
by
  have α := Real.atan (2 / 1)
  -- Given conditions
  have hsin : sin α = √5 / 5 := 
    sorry -- Proven from given conditions
    
  have hcos : cos α = 2 * √5 / 5 :=
    sorry -- Proven from given conditions

  -- Proof of the main result (to be completed)
  sorry

end sin2alpha_cos2alpha_eq_l378_378074


namespace PY_length_l378_378740

open Real

variables {X Y Z P Q : Type} -- Triangle points
variables (XY YZ PQ PY : ℝ) -- side lengths

-- Conditions
def triangle_XYZ (X Y Z : Π {T: Type}, set T) (X ≠ Y) (Y ≠ Z) (X ≠ Z) : Prop :=
  ∃ XY YZ XZ, XY ≂ YZ ∧ XY ≂ XZ ∧ XY = 8 ∧ YZ = 6

def right_angle_Y (X Y Z : Π {T: Type}, set T) : Prop := angle X Y Z = 90°

def point_PQ (P Q : Π {T: Type}, set T) : Prop := 
  P ∈ XZ ∧ Q ∈ YZ ∧ angle P Q Y = 90°

def PQ_length_3 (PQ : Π {T: Type}, set T) : Prop := length PQ = 3

-- Proof statement to prove PY == 15/4
theorem PY_length 
  (h1 : triangle_XYZ X Y Z)
  (h2 : right_angle_Y X Y Z)
  (h3 : XY = 8)
  (h4 : YZ = 6)
  (h5 : point_PQ P Q)
  (h6 : PQ_length_3 PQ) : 
  PY = 15 / 4 := 
by 
  sorry

end PY_length_l378_378740


namespace simplify_expression_l378_378671

-- Given the condition
def cond (a b : ℝ) : Prop := (3 * a) / 2 + b = 1

-- State the main theorem
theorem simplify_expression (a b : ℝ) (h : cond a b) : (9^a * 3^b) / (sqrt (3^a)) = 3 :=
by
  sorry

end simplify_expression_l378_378671


namespace uncovered_area_l378_378146

theorem uncovered_area {s₁ s₂ : ℝ} (hs₁ : s₁ = 10) (hs₂ : s₂ = 4) : 
  (s₁^2 - 2 * s₂^2) = 68 := by
  sorry

end uncovered_area_l378_378146


namespace tangent_line_at_point_is_correct_l378_378045

theorem tangent_line_at_point_is_correct :
  ∀ (x : ℝ), (x • exp x + 2 * x + 1) for given x = 0 := 
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ),
  f x = x * exp x + 2 * x + 1 ∧
  (∀ (p : ℝ × ℝ), p = (0, 1) →
    (∃ (m : ℝ), m = (derivative f) 0 ∧
    (∃ (b : ℝ), ∀ (y : ℝ), y = m * x + b ∧
    b = 1 ∧ m = 3))))

sorry

end tangent_line_at_point_is_correct_l378_378045


namespace pages_in_first_chapter_l378_378797

/--
Rita is reading a five-chapter book with 95 pages. Each chapter has three pages more than the previous one. 
Prove the number of pages in the first chapter.
-/
theorem pages_in_first_chapter (h : ∃ p1 p2 p3 p4 p5 : ℕ, p1 + p2 + p3 + p4 + p5 = 95 ∧ p2 = p1 + 3 ∧ p3 = p1 + 6 ∧ p4 = p1 + 9 ∧ p5 = p1 + 12) : 
  ∃ x : ℕ, x = 13 := 
by
  sorry

end pages_in_first_chapter_l378_378797


namespace tension_in_cord_l378_378164

variables (M m : ℝ) (R g: ℝ)
variable T : ℝ

-- Conditions given in the problem
def moment_of_inertia := (1/2) * M * R^2
def mass_disk := 8.0
def mass_block := 6.0
def gravity := 9.8

-- The final statement we need to prove
theorem tension_in_cord : 
  let I := moment_of_inertia M R in
  let a := (m * g) / (m + (1/2) * M) in
  T = (1/2) * M * a :=
sorry

end tension_in_cord_l378_378164


namespace range_of_a_l378_378682

noncomputable def f (x : ℝ) : ℝ := - 4 ^ x + 2 ^ (x + 1) - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := g(a * x ^ 2 - 4 * x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 = g a x2) ↔ a ∈ set.Iic (4 : ℝ) := -- Iic means interval (-∞, 4]
sorry

end range_of_a_l378_378682


namespace remainder_of_sum_div_500_l378_378759

theorem remainder_of_sum_div_500 :
  let T := ∑ n in Finset.range 1005, (-1)^n * Nat.choose 3006 (3 * n)
  T % 500 = 18 :=
by
  sorry

end remainder_of_sum_div_500_l378_378759


namespace expected_number_of_digits_l378_378790

-- Define a noncomputable expected_digits function for an icosahedral die
noncomputable def expected_digits : ℝ :=
  let p1 := 9 / 20
  let p2 := 11 / 20
  (p1 * 1) + (p2 * 2)

theorem expected_number_of_digits :
  expected_digits = 1.55 :=
by
  -- The proof will be filled in here
  sorry

end expected_number_of_digits_l378_378790


namespace find_f_2000_l378_378050

noncomputable def f : ℕ → ℕ := sorry

axiom f_double (n : ℕ) (hn : 0 < n) : f(f(n)) = 2 * n
axiom f_special (n : ℕ) (hn : 0 < n) : f(4 * n + 1) = 4 * n + 5

theorem find_f_2000 : f 2000 = 2064 := sorry

end find_f_2000_l378_378050


namespace solutions_to_z_cubed_eq_27i_l378_378282

noncomputable def verify_solutions (z : ℂ) : Prop :=
  z^3 = 27 * complex.I

theorem solutions_to_z_cubed_eq_27i :
  ({3 * complex.I, -3 / 2 * complex.I + 3 * real.sqrt 3 / 2, -3 / 2 * complex.I - 3 * real.sqrt 3 / 2} : set ℂ) =
  {z : ℂ | verify_solutions z} :=
by
  sorry

end solutions_to_z_cubed_eq_27i_l378_378282


namespace inequality_proof_l378_378399
noncomputable def inequality (x y z : ℝ) (h : x > 0) (h1 : y > 0) (h2 : z > 0) (h3 : x + y + z = 1) : Prop :=
  x / (y^2 + z) + y / (z^2 + x) + z / (x^2 + y) ≥ 9 / 4

theorem inequality_proof {x y z : ℝ} (h : x > 0) (h1 : y > 0) (h2 : z > 0) (h3 : x + y + z = 1) : 
  x / (y^2 + z) + y / (z^2 + x) + z / (x^2 + y) ≥ 9 / 4 :=
begin
  sorry -- Proof goes here
end

end inequality_proof_l378_378399


namespace height_of_rotated_square_l378_378846

theorem height_of_rotated_square {s₁ s₂ s₃ : ℝ} (h₁ : s₁ = 1) (h₂ : s₂ = 2) (h₃ : s₃ = 1) 
(rotation_angle : ℝ) (ha : rotation_angle = 30) :
  let height_projection := s₂ * Real.sqrt 2 / 2 in
  let center_height := s₂ / 2 in
  height_projection + center_height = 1 + Real.sqrt 2 :=
by sorry

end height_of_rotated_square_l378_378846


namespace choose_5_from_30_l378_378422

theorem choose_5_from_30 :
  (nat.choose 30 5) = 54810 :=
by
  sorry

end choose_5_from_30_l378_378422


namespace average_age_of_first_7_students_l378_378807

theorem average_age_of_first_7_students (avg_15_students : ℕ) 
 (total_students : ℕ) 
 (avg_second_7_students : ℕ) 
 (total_second_7_students : ℕ) 
 (age_15th_student : ℕ) : 
 avg_15_students = 15 ∧ total_students = 15 ∧ avg_second_7_students = 16 ∧ total_second_7_students = 7 ∧ age_15th_student = 15 → 
 (let total_age_15_students := avg_15_students * total_students in 
  let total_age_second_7_students := avg_second_7_students * total_second_7_students in 
  let total_age_first_7_students := total_age_15_students - total_age_second_7_students - age_15th_student in 
  total_age_first_7_students / total_second_7_students = 14) :=
begin
  intros h,
  rcases h with ⟨h1, h2, h3, h4, h5⟩, 
  have total_age_15_students := h1 * h2, 
  have total_age_second_7_students := h3 * h4,
  have total_age_first_7_students := total_age_15_students - total_age_second_7_students - h5,
  have avg_first_7_students := total_age_first_7_students / h4,
  linarith,
  sorry
end

end average_age_of_first_7_students_l378_378807


namespace cubic_product_of_roots_l378_378274

theorem cubic_product_of_roots (k : ℝ) :
  (∃ a b c : ℝ, a + b + c = 2 ∧ ab + bc + ca = 1 ∧ abc = -k ∧ -k = (max (max a b) c - min (min a b) c)^2) ↔ k = -2 :=
by
  sorry

end cubic_product_of_roots_l378_378274


namespace pucks_cannot_return_to_original_after_25_hits_l378_378012

/-- 
Given three pucks A, B, and C, initially arranged in a specific manner (either "correct" or "incorrect"),
where a "correct" arrangement means a clockwise traversal of the triangle ABC and "incorrect" means counterclockwise.
A hockey player hits one puck 25 times, with each hit toggling the arrangement between "correct" and "incorrect".
Prove that the pucks cannot return to their original positions after 25 hits.
-/
theorem pucks_cannot_return_to_original_after_25_hits
  (initial_arrangement : Bool) -- true for "correct", false for "incorrect"
  (hits : ℕ) : (hits = 25) → (hits % 2 ≠ 0) → (initial_arrangement ≠ ¬initial_arrangement) :=
by
  assume hits_eq : hits = 25
  assume hits_odd : hits % 2 ≠ 0
  sorry

end pucks_cannot_return_to_original_after_25_hits_l378_378012


namespace train_passes_man_in_time_l378_378908

def length_of_train : ℝ := 240 -- in meters
def speed_of_train : ℝ := 60  -- in km/h
def speed_of_man : ℝ := 6     -- in km/h

def kmph_to_mps (kmph : ℝ) : ℝ :=
  kmph * 1000 / 3600

def relative_speed (speed_train : ℝ) (speed_man : ℝ) : ℝ :=
  speed_train + speed_man

def time_to_pass (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_passes_man_in_time :
  time_to_pass length_of_train (kmph_to_mps (relative_speed speed_of_train speed_of_man)) ≈ 13.09 :=
by
  sorry

end train_passes_man_in_time_l378_378908


namespace boys_in_other_communities_correct_l378_378367

def total_boys : ℕ := 700
def percentage_muslims : ℕ := 44
def percentage_hindus : ℕ := 28
def percentage_sikhs : ℕ := 10
def percentage_other_communities : ℕ := 100 - (percentage_muslims + percentage_hindus + percentage_sikhs)
def boys_other_communities : ℝ := (percentage_other_communities / 100) * total_boys

theorem boys_in_other_communities_correct :
  boys_other_communities = 126 :=
by sorry

end boys_in_other_communities_correct_l378_378367


namespace ratio_equivalence_l378_378525

theorem ratio_equivalence (x : ℚ) (h : x / 360 = 18 / 12) : x = 540 :=
by
  -- Proof goes here, to be filled in
  sorry

end ratio_equivalence_l378_378525


namespace ratio_of_socks_l378_378172

-- Conditions:
variable (B : ℕ) (W : ℕ) (L : ℕ)
-- B = number of black socks
-- W = initial number of white socks
-- L = number of white socks lost

-- Setting given conditions:
axiom hB : B = 6
axiom hL : L = W / 2
axiom hCond : W / 2 = B + 6

-- Prove the ratio of white socks to black socks is 4:1
theorem ratio_of_socks : B = 6 → W / 2 = B + 6 → (W / 2) + (W / 2) = 24 → (B : ℚ) / (W : ℚ) = 1 / 4 :=
by intros hB hCond hW
   sorry

end ratio_of_socks_l378_378172


namespace minimize_sum_AC_CB_l378_378303

noncomputable def point_min_distance : ℝ × ℝ := (-2, -3)
def pointA : ℝ × ℝ := (-2, -3)
def pointB : ℝ × ℝ := (5, 3)
def x_coordinate : ℝ := 2

def y_coordinate_min (n : ℝ) : Prop :=
  let (xA, yA) := pointA;
  let (xB, yB) := pointB;
  let C := (x_coordinate, n);
  ∃ n, (x_coordinate, n) ∈ C ∧ (AC + CB is minimized)

theorem minimize_sum_AC_CB : y_coordinate_min (13/2) :=
sorry

end minimize_sum_AC_CB_l378_378303


namespace sqrt_inequality_abc_inequality_l378_378588

-- Problem 1
theorem sqrt_inequality : (sqrt 7) + (sqrt 13) < 3 + (sqrt 11) :=
sorry

-- Problem 2
theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * (b^2 + c^2) + b * (c^2 + a^2) + c * (a^2 + b^2) ≥ 6 * a * b * c :=
sorry

end sqrt_inequality_abc_inequality_l378_378588


namespace hyperbola_asymptotes_eccentricity_hyperbola_asymptotes_eccentricity_alternative_hyperbola_eccentricity_correct_l378_378666

noncomputable def hyperbola_eccentricity (a b c : ℝ) (x : ℝ) : ℝ :=
c / a

theorem hyperbola_asymptotes_eccentricity (a b : ℝ) (h₀ : b = 2 * a) : 
  hyperbola_eccentricity a b (sqrt (5 * a^2)) = sqrt 5 :=
sorry

theorem hyperbola_asymptotes_eccentricity_alternative (a b : ℝ) (h₀ : a = 2 * b) : 
  hyperbola_eccentricity a b ((sqrt 5) / 2 * a) = sqrt 5 / 2 :=
sorry

theorem hyperbola_eccentricity_correct (a b : ℝ) (x : ℝ) (h₀ : b = 2 * a ∨ a = 2 * b) : 
  hyperbola_eccentricity a b (sqrt (5 * a^2)) = sqrt 5 ∨ 
  hyperbola_eccentricity a b ((sqrt 5) / 2 * a) = sqrt 5 / 2 :=
by 
  cases h₀ 
  · left
    exact (hyperbola_asymptotes_eccentricity a b h₀)
  · right
    exact (hyperbola_asymptotes_eccentricity_alternative a b h₀)

end hyperbola_asymptotes_eccentricity_hyperbola_asymptotes_eccentricity_alternative_hyperbola_eccentricity_correct_l378_378666


namespace find_k_l378_378283

theorem find_k (k : ℚ) :
  (5 + ∑' n : ℕ, (5 + 2*k*(n+1)) / 4^n) = 10 → k = 15/4 :=
by
  sorry

end find_k_l378_378283


namespace math_evening_problem_l378_378972

theorem math_evening_problem
  (S : ℕ)
  (r : ℕ)
  (fifth_graders_per_row : ℕ := 3)
  (sixth_graders_per_row : ℕ := r - fifth_graders_per_row)
  (total_number_of_students : ℕ := r * r) :
  70 < total_number_of_students ∧ total_number_of_students < 90 → 
  r = 9 ∧ 
  6 * r = 54 ∧
  3 * r = 27 :=
sorry

end math_evening_problem_l378_378972


namespace presidency_meeting_ways_l378_378142

theorem presidency_meeting_ways :
  let total_schools := 4
  let members_per_school := 4
  let host_school_choices := total_schools
  let choose_3_from_4 := Nat.choose 4 3
  let choose_1_from_4 := Nat.choose 4 1
  let ways_per_host := choose_3_from_4 * choose_1_from_4 ^ 3
  let total_ways := host_school_choices * ways_per_host
  total_ways = 1024 := by
  sorry

end presidency_meeting_ways_l378_378142


namespace m_range_l378_378675

noncomputable def f (x a : ℝ) := log x + x^2 - 2 * a * x + 1

noncomputable def h (a m : ℝ) := 2 * m * exp a * (a + 1) - a^2 - 4 * a - 2

theorem m_range (x_0 a m : ℝ) (hx_0 : 0 < x_0 ∧ x_0 ≤ 1) (ha : -2 < a ∧ a ≤ 0) 
(h_ineq : ∀ a ∈ Icc (-2 : ℝ) 0, h a m > 0 ) : 
1 < m ∧ m ≤ exp 2 :=
sorry

end m_range_l378_378675


namespace exponent_problem_1_log_problem_1_l378_378521

-- Problem 1
theorem exponent_problem_1 :
  (0.064 : ℝ) ^ (-1 / 3) - (- (1 / 8) : ℝ) ^ 0 + (16 : ℝ) ^ (3 / 4) + (0.25 : ℝ) ^ (1 / 2) = 10 :=
by
  sorry

-- Problem 2
theorem log_problem_1 :
  (2 * real.log 2 + real.log 3) / (1 + (1 / 2) * real.log 0.36 + (1 / 3) * real.log 8) = 1 :=
by
  sorry

end exponent_problem_1_log_problem_1_l378_378521


namespace sum_of_non_solutions_l378_378415

theorem sum_of_non_solutions (A B C x : ℝ) 
  (h : ∀ x, ((x + B) * (A * x + 32)) = 4 * ((x + C) * (x + 8))) :
  (x = -B ∨ x = -8) → x ≠ -B → -B ≠ -8 → x ≠ -8 → x + 8 + B = 0 := 
sorry

end sum_of_non_solutions_l378_378415


namespace square_in_hexagon_side_length_l378_378027

theorem square_in_hexagon_side_length :
  ∀ (A B C D E F X Y Z : Point) (AXYZ : Square)
  (hexagon : EquiangularHexagon A B C D E F)
  (hX : on_segment X B C)
  (hY : on_segment Y D E)
  (hZ : on_segment Z E F),
  dist A B = 40 ∧ dist E F = 41 * (Real.sqrt 3 - 1) →
  dist A X = 29 * Real.sqrt 3 := 
by
  sorry

end square_in_hexagon_side_length_l378_378027


namespace simplification_problem_l378_378769

theorem simplification_problem (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h_sum : p + q + r = 1) :
  (1 / (q^2 + r^2 - p^2) + 1 / (p^2 + r^2 - q^2) + 1 / (p^2 + q^2 - r^2) = 3 / (1 - 2 * q * r)) :=
by
  sorry

end simplification_problem_l378_378769


namespace alpha_in_third_quadrant_l378_378306

noncomputable def quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, (4 * k + 1) * 180 < α ∧ α < (4 * k + 1) * 180 + 60

theorem alpha_in_third_quadrant (k : ℤ) (α : ℝ) (h : (4 * k + 1) * 180 < α ∧ α < (4 * k + 1) * 180 + 60) :
  180 < α ∧ α < 270 :=
begin
  sorry
end

end alpha_in_third_quadrant_l378_378306


namespace boys_play_basketball_is_15_l378_378494

-- Definitions based on conditions
def total_students : ℕ := 30
def fraction_of_girls : ℝ := 1 / 3
def fraction_of_boys_playing_basketball : ℝ := 3 / 4

-- Given conditions as definitions
def number_of_girls (total_students : ℕ) (fraction_of_girls : ℝ) : ℕ :=
  (fraction_of_girls * total_students).to_nat

def number_of_boys (total_students number_of_girls : ℕ) : ℕ :=
  total_students - number_of_girls

def boys_playing_basketball (number_of_boys : ℕ) (fraction_of_boys_playing_basketball : ℝ) : ℕ :=
  (fraction_of_boys_playing_basketball * number_of_boys).to_nat

-- Theorem to prove
theorem boys_play_basketball_is_15 :
  boys_playing_basketball (number_of_boys total_students (number_of_girls total_students fraction_of_girls)) fraction_of_boys_playing_basketball = 15 := 
sorry

end boys_play_basketball_is_15_l378_378494


namespace average_age_of_first_7_students_l378_378810

variables {x : ℕ}

theorem average_age_of_first_7_students (h1 : 15 = 15) 
                                       (h2 : 15 * 15 = 225)
                                       (h3 : 16 * 7 = 112)
                                       (h4 : 225 - 112 - 15 = 98)
                                       (h5 : 98 / 7 = 14) :
                                       x = 14 := 
begin
  sorry
end

end average_age_of_first_7_students_l378_378810


namespace water_level_rise_by_cube_l378_378977

-- Define the given conditions
def edge_length : ℝ := 1
def tank_cross_sectional_area (A : ℝ) := A > 0

-- Define the volume of the cube based on its edge length
def cube_volume (a : ℝ) : ℝ := a^3

-- Define the volume of water displaced when the cube is placed in the tank
def water_displaced_volume (v_cube : ℝ) : ℝ := v_cube

-- Define the rise in water level calculated from the displaced water and the cross-sectional area
def water_level_rise (A : ℝ) (v_displaced : ℝ) : ℝ := v_displaced / A

-- Prove that the rise in water level is 1 / A
theorem water_level_rise_by_cube (A : ℝ) (h : ℝ) (H : tank_cross_sectional_area A)
: h = water_level_rise A (water_displaced_volume (cube_volume edge_length)) := by
  sorry

end water_level_rise_by_cube_l378_378977


namespace log_one_fourth_sixteen_l378_378249

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 := 
by
  let x := log (1 / 4) 16
  have h₁ : (1 / 4) ^ x = 16 := by simp [log_eq_iff]
  have h₂ : (4 ^ (-1)) ^ x = 16 := by rw [one_div, inv_pow]
  have h₃ : 4 ^ (-x) = 16 := by simp [pow_mul]
  have h₄ : 16 = 4 ^ 2 := by norm_num
  rw [h₄] at h₃
  have h₅ : -x = 2 := by exact pow_inj (lt_trans zero_lt_one (by norm_num)) zero_lt_four h₃
  have h₆ : x = -2 := by linarith
  exact h₆

end log_one_fourth_sixteen_l378_378249


namespace daily_evaporation_l378_378134

theorem daily_evaporation (initial_water: ℝ) (days: ℝ) (evap_percentage: ℝ) : 
  initial_water = 10 → days = 50 → evap_percentage = 2 →
  (initial_water * evap_percentage / 100) / days = 0.04 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end daily_evaporation_l378_378134


namespace smaller_interior_angle_of_parallelogram_l378_378473

theorem smaller_interior_angle_of_parallelogram (x : ℝ) 
  (h1 : ∃ l, l = x + 90 ∧ x + l = 180) :
  x = 45 :=
by
  obtain ⟨l, hl1, hl2⟩ := h1
  simp only [hl1] at hl2
  linarith

end smaller_interior_angle_of_parallelogram_l378_378473


namespace folded_rectangle_perimeter_l378_378949

theorem folded_rectangle_perimeter (l : ℝ) (w : ℝ) (h_diag : ℝ)
  (h_l : l = 20) (h_w : w = 12)
  (h_diag : h_diag = Real.sqrt (l^2 + w^2)) :
  2 * (l + w) = 64 :=
by
  rw [h_l, h_w]
  simp only [mul_add, mul_two, add_mul] at *
  norm_num


end folded_rectangle_perimeter_l378_378949


namespace median_in_isosceles_triangle_is_altitude_median_in_isosceles_triangle_is_angle_bisector_l378_378368

-- Define an isosceles triangle and its properties
variables {A B C M : Type}
variables [IsoscelesTriangle : IsoscelesTriangle A B C]
variables [is_midpoint : MidPoint M B C]
variables [is_median : Median A M]

-- Declare properties to be proven
theorem median_in_isosceles_triangle_is_altitude :
  Perpendicular A M B C :=
sorry

theorem median_in_isosceles_triangle_is_angle_bisector :
  AngleBisector A M B C :=
sorry

end median_in_isosceles_triangle_is_altitude_median_in_isosceles_triangle_is_angle_bisector_l378_378368


namespace log_base_one_four_of_sixteen_l378_378242

theorem log_base_one_four_of_sixteen : log (1 / 4) 16 = -2 := by
  sorry

end log_base_one_four_of_sixteen_l378_378242


namespace area_enclosed_by_absolute_value_linear_eq_l378_378858

theorem area_enclosed_by_absolute_value_linear_eq (x y : ℝ) :
  (|5 * x| + |3 * y| = 15) → ∃ (A : ℝ), A = 30 :=
by
  sorry

end area_enclosed_by_absolute_value_linear_eq_l378_378858


namespace barry_should_pay_l378_378176

def original_price : ℝ := 80
def discount_rate : ℝ := 0.15

theorem barry_should_pay:
  original_price * (1 - discount_rate) = 68 := 
by 
  -- Original price: 80
  -- Discount rate: 0.15
  -- Question: Final price after discount
  sorry

end barry_should_pay_l378_378176


namespace probability_at_least_one_white_ball_l378_378535

-- Define the probability function
def probability {α : Type} (s : Finset α) (event : Set α) : ℚ :=
  (Finset.card (s.filter event)) / (Finset.card s)

-- Define the box and the events
def box : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def red_balls : Set ℕ := {1, 2, 3, 4, 5}
def white_balls : Set ℕ := {6, 7, 8, 9}

-- Define the event of picking two balls
def event (a b : ℕ) : Set (ℕ × ℕ) := { (x, y) | x ≠ y ∧ (x ∈ red_balls ∨ x ∈ white_balls) ∧ (y ∈ red_balls ∨ y ∈ white_balls)}

-- Define the event of picking two red balls
def event_red_red (a b : ℕ) : Set (ℕ × ℕ) := { (x, y) | x ≠ y ∧ x ∈ red_balls ∧ y ∈ red_balls}

-- Define the set of pairs of any two balls
def pairs_of_balls : Finset (ℕ × ℕ) := { (x, y) | x ∈ box ∧ y ∈ box ∧ x ≠ y}.to_finset

-- Prove the required probability 
theorem probability_at_least_one_white_ball 
  : probability pairs_of_balls (λ (p : ℕ × ℕ), p.1 ∈ white_balls ∨ p.2 ∈ white_balls) = 13 / 18 :=
by 
  sorry

end probability_at_least_one_white_ball_l378_378535


namespace max_N_impassable_roads_l378_378374

def number_of_cities : ℕ := 1000
def number_of_roads : ℕ := 2017
def initial_connected_components : ℕ := 1
def target_connected_components : ℕ := 7

theorem max_N_impassable_roads 
    (h : ∀ (G : SimpleGraph (Fin 1000)), G.edgeCount = 2017 ∧ G.isConnected ∧ G.connectedComponents.card = target_connected_components) :
    ∃ N : ℕ, N = 993 :=
begin
  use 993,
  sorry
end

end max_N_impassable_roads_l378_378374


namespace area_enclosed_by_graph_l378_378864

theorem area_enclosed_by_graph (x y : ℝ) (h : abs (5 * x) + abs (3 * y) = 15) : 
  ∃ (area : ℝ), area = 30 :=
sorry

end area_enclosed_by_graph_l378_378864


namespace shaded_area_l378_378063

theorem shaded_area (PQ : ℝ) (n_squares : ℕ) (d_intersect : ℝ)
  (h1 : PQ = 8) (h2 : n_squares = 20) (h3 : d_intersect = 8) : ∃ (A : ℝ), A = 160 := 
by {
  sorry
}

end shaded_area_l378_378063


namespace area_proportions_and_point_on_line_l378_378054

theorem area_proportions_and_point_on_line (T : ℝ × ℝ) :
  (∃ r s : ℝ, T = (r, s) ∧ s = -(5 / 3) * r + 10 ∧ 1 / 2 * 6 * s = 7.5) 
  ↔ T.1 + T.2 = 7 :=
by { sorry }

end area_proportions_and_point_on_line_l378_378054


namespace parallel_midpoints_angle_bisector_l378_378022

theorem parallel_midpoints_angle_bisector
  {X O Y A A' B B' M M' : Point}
  (hAA'_eq_BB' : dist A A' = dist B B')
  (hM_midpoint : midpoint A B M)
  (hM'_midpoint : midpoint A' B' M') :
  parallel (line_through O (bisector_angle X O Y)) (line_through M M') :=
sorry

end parallel_midpoints_angle_bisector_l378_378022


namespace tangent_line_ellipse_l378_378516

variable {a b x x0 y y0 : ℝ}

theorem tangent_line_ellipse (h : a * x0^2 + b * y0^2 = 1) :
  a * x0 * x + b * y0 * y = 1 :=
sorry

end tangent_line_ellipse_l378_378516


namespace mixed_repeating_decimal_divisibility_l378_378155

theorem mixed_repeating_decimal_divisibility
  (m k : ℕ) (b a : ℕ → ℕ)
  (h1 : m ≥ 1) (h2 : b m ≠ a k)
  (p q : ℕ) 
  (h3 : irreducible_fraction (0.b[1] * 10^(m-1) + ... + 0.b[m] * 10^0 + 0.a[1] * 10^(k-1) + ... + 0.a[k] / (10^m * (10^k - 1)) = p / q)) :
  q % 2 = 0 ∨ q % 5 = 0 :=
sorry

end mixed_repeating_decimal_divisibility_l378_378155


namespace find_a_b_find_max_profit_find_max_m_l378_378926

section fruit_distribution

-- Definitions of constants and conditions
variables (a b m x : ℝ)

-- Condition 1: cost price relation between A and B
def cost_price_condition : Prop := b = a + 2.5

-- Condition 2: cost ratio relation
def cost_ratio_condition : Prop := (400 * b) / (200 * a) = 24 / 7

-- Condition 3: total daily purchase
def daily_purchase_condition : Prop := 300 = x + (300 - x)

-- Condition 4: amount of fruit A sold each day
def amount_sold_condition : Prop := 80 ≤ x ∧ x ≤ 120

-- Condition 5: minimum profit requirement on weekends
def minimum_profit_condition : Prop := (0.5 - m) * 80 + 300 ≥ 312

-- Prove the values of a and b
theorem find_a_b :
  cost_price_condition a b ∧ cost_ratio_condition a b → a = 3.5 ∧ b = 6 :=
sorry

-- Prove the maximum profit from selling the two fruits on that day
theorem find_max_profit (x : ℝ) :
  daily_purchase_condition x ∧ amount_sold_condition x →
  let W := 0.5 * x + 300 in W = 360 :=
sorry

-- Prove the maximum value of m during weekends to ensure minimum profit
theorem find_max_m :
  amount_sold_condition 80 ∧ minimum_profit_condition m →
  m = 0.35 :=
sorry

end fruit_distribution

end find_a_b_find_max_profit_find_max_m_l378_378926


namespace angle_is_right_l378_378849

noncomputable def is_second_intersection (T : Point) (A B C : Triangle) : Prop := sorry
noncomputable def is_symmedian (A B C : Triangle) (T : Point) : Prop := sorry
noncomputable def lies_on (D : Point) (l : Line) : Prop := sorry
noncomputable def is_equal_length (BA BD : LineSegment) : Prop := sorry
noncomputable def is_tangent_at (D : Point) (O : Circle) : Prop := sorry
noncomputable def intersects_second_time_at (tangent_line : Line) (O1 O2 : Circle) (K : Point) : Prop := sorry
noncomputable def is_circumcircle (O : Circle) (T : Triangle) : Prop := sorry
noncomputable def right_angle (angle : Angle) : Prop := sorry

theorem angle_is_right {A B C T D K : Point} {ABC T ADT DCT : Triangle} 
                       {O : Circle} (h1 : is_second_intersection T A B C) 
                       (h2 : is_symmedian A B C T) 
                       (h3 : lies_on D (Line.mk A C)) 
                       (h4 : is_equal_length (LineSegment.mk B A) (LineSegment.mk B D)) 
                       (h5 : is_tangent_at D (circumcircle ADT)) 
                       (h6 : intersects_second_time_at (tangent_line_through D (circumcircle ADT)) (circumcircle ADT) (circumcircle DCT) K) : 
  right_angle (Angle.mk B K C) :=
sorry

end angle_is_right_l378_378849


namespace max_height_l378_378532

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 10

theorem max_height : ∃ t : ℝ, (∀ x : ℝ, h(x) ≤ h(t)) ∧ h(t) = 135 :=
by
  sorry

end max_height_l378_378532


namespace correct_derivative_of_sin_x_squared_l378_378904

theorem correct_derivative_of_sin_x_squared :
  ∀ (x : ℝ), 
  let y := sin (x^2)
  in (deriv (λ x, deriv (λ x, sin (x^2)) x) x = 2 * x * cos (x^2)) :=
by
  sorry

end correct_derivative_of_sin_x_squared_l378_378904


namespace arithmetic_sequence_num_terms_l378_378216

theorem arithmetic_sequence_num_terms (a d l : ℕ) (h1 : a = 15) (h2 : d = 4) (h3 : l = 159) :
  ∃ n : ℕ, l = a + (n-1) * d ∧ n = 37 :=
by {
  sorry
}

end arithmetic_sequence_num_terms_l378_378216


namespace regular_n_gon_coloring_l378_378841

theorem regular_n_gon_coloring (n : ℕ) (colors : V → Color) (is_regular_polygon : ∀ (c : Color), is_regular_polygon (colors⁻¹' {c})) : 
  ∃ (A B : Color), A ≠ B ∧ is_congruent (colors⁻¹' {A}) (colors⁻¹' {B}) := 
sorry

end regular_n_gon_coloring_l378_378841


namespace find_xyz_l378_378273

theorem find_xyz (x y z : ℝ) (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : z ≥ 1) :
  (min (sqrt (x + x * y * z)) (min (sqrt (y + x * y * z)) (sqrt (z + x * y * z))) = 
   sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1)) ↔
  ∃ t : ℝ, t > 0 ∧
  (x = 1 + (t / (t^2 + 1))^2 ∧ y = 1 + (1 / t^2) ∧ z = 1 + t^2) :=
by sorry

end find_xyz_l378_378273


namespace last_remaining_number_cannot_be_zero_l378_378504

theorem last_remaining_number_cannot_be_zero :
  (∃ final_int : Int, 
    ∀ steps : List (Int × Int), 
    (∀ step in steps, step.snd = step.fst - step.snd) →
    List.range 2014 ≠ steps.foldl (fun acc step => acc.erase step.fst.erase step.snd ++ [step.fst - step.snd]) (List.range 2014) :=
    List.repeat final_int 2013) →
  final_int ≠ 0 :=
sorry

end last_remaining_number_cannot_be_zero_l378_378504


namespace empty_set_is_d_l378_378919

open Set

theorem empty_set_is_d : {x : ℝ | x^2 - x + 1 = 0} = ∅ :=
by
  sorry

end empty_set_is_d_l378_378919


namespace cherry_pies_count_correct_l378_378452

def total_pies : ℕ := 36

def ratio_ap_bb_ch : (ℕ × ℕ × ℕ) := (2, 3, 4)

def total_ratio_parts : ℕ := 2 + 3 + 4

def pies_per_part (total_pies : ℕ) (total_ratio_parts : ℕ) : ℕ := total_pies / total_ratio_parts

def num_parts_ch : ℕ := 4

def num_cherry_pies (total_pies : ℕ) (total_ratio_parts : ℕ) (num_parts_ch : ℕ) : ℕ :=
  pies_per_part total_pies total_ratio_parts * num_parts_ch

theorem cherry_pies_count_correct : num_cherry_pies total_pies total_ratio_parts num_parts_ch = 16 := by
  sorry

end cherry_pies_count_correct_l378_378452


namespace factorization_correct_l378_378271

theorem factorization_correct (x : ℤ) :
  (3 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 2 * x^2) =
  ((3 * x^2 + 35 * x + 72) * (x + 3) * (x + 6)) :=
by sorry

end factorization_correct_l378_378271


namespace curve_line_intersection_ratio_l378_378347

theorem curve_line_intersection_ratio (p q : ℕ) (hpq : Nat.Coprime p q) (hpq_rel : p < q) :
  (∀ x : ℝ, (real.sin (x + real.pi / 6) = 1 / 2)) →
  p = 1 ∧ q = 2 :=
begin
  sorry,
end

end curve_line_intersection_ratio_l378_378347


namespace original_number_of_men_l378_378114

-- Define the conditions
def work_days_by_men (M : ℕ) (days : ℕ) : ℕ := M * days
def additional_men (M : ℕ) : ℕ := M + 10
def completed_days : ℕ := 9

-- The main theorem
theorem original_number_of_men : ∀ (M : ℕ), 
  work_days_by_men M 12 = work_days_by_men (additional_men M) completed_days → 
  M = 30 :=
by
  intros M h
  sorry

end original_number_of_men_l378_378114


namespace leesburg_population_l378_378451

theorem leesburg_population (salem_population leesburg_population half_salem_population number_moved_out : ℕ)
  (h1 : half_salem_population * 2 = salem_population)
  (h2 : salem_population - number_moved_out = 754100)
  (h3 : salem_population = 15 * leesburg_population)
  (h4 : half_salem_population = 377050)
  (h5 : number_moved_out = 130000) :
  leesburg_population = 58940 :=
by
  sorry

end leesburg_population_l378_378451


namespace length_DC_l378_378375

theorem length_DC (AB BD BC : ℝ) (h_AB : AB = 30) (h_angle_ADB : angle A D B = 90)
  (h_sin_A : sin A = 4 / 5) (h_sin_C : sin C = 1 / 4)
  (h_BD : BD = (4 / 5) * AB) (h_BC : BC = 4 * BD) :
  ∃ DC : ℝ, DC = 24 * sqrt 15 :=
by sorry

end length_DC_l378_378375


namespace pedestrian_average_speed_not_5_l378_378946

open Real

-- Define constants and conditions
def time_walked : ℝ := 3.5
def distance_per_hour : ℝ := 5

-- Define total distance covered
def total_distance := distance_per_hour * time_walked

-- Define the average speed calculation
def average_speed := total_distance / time_walked

-- Define the statement to prove that the average speed is not 5 km/h
theorem pedestrian_average_speed_not_5 :
  ¬ (average_speed = 5) :=
by 
  sorry

end pedestrian_average_speed_not_5_l378_378946


namespace f_10_equals_169_l378_378935

def f : ℕ → ℤ
| 1 := 2
| 2 := 3
| n := f(n-1) + f(n-2) - n

theorem f_10_equals_169 : f 10 = 169 := 
by {
  sorry
}

end f_10_equals_169_l378_378935


namespace count_squares_in_region_l378_378213

def region_bounded_by (x y : ℕ) : Prop :=
  y <= 2 * x ∧ y >= 0 ∧ x <= 6

def is_square (x1 y1 x2 y2 : ℕ) : Prop :=
  x2 = x1 + 1 ∧ y2 = y1 + 1 ∧ region_bounded_by x1 y1 ∧ region_bounded_by x2 y2

theorem count_squares_in_region : 
  (finset.univ.filter (λ x1 : ℕ, x1 <= 6)).sum (λ x1, 
    (finset.univ.filter (λ y1 : ℕ, y1 <= 12)).sum (λ y1, 
      if is_square x1 y1 (x1+1) (y1+1) then 1 else 0)) = 77 :=
sorry

end count_squares_in_region_l378_378213


namespace new_median_is_five_l378_378541

/-- Given a collection of six positive integers has a mean of 5.5, a unique mode of 4, and a median of 5,
    and adding a 10 to the collection, prove that the new median is equal to 5. --/
theorem new_median_is_five (a b c d e f : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0)
  (h7 : (a + b + c + d + e + f) / 6 = 5.5)
  (h8 : ∃ x, (∃ y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ ∀ p, (p = x ∨ p = y ∨ p = z) → p = 4) 
    ∧ ∀ q, (q ≠ 4 → (a = q ∨ b = q ∨ c = q ∨ d = q ∨ e = q ∨ f = q) → q ≠ 4)) 
  (h9 : {a, b, c, d, e, f}.toFinset.sort (≤).get! 2 = 5 ∧ {a, b, c, d, e, f}.toFinset.sort (≤).get! 3 = 5) :
  ∃ (g : ℕ), g = 10 ∧ ({a, b, c, d, e, f, g}.toFinset.sort (≤).get! 3 = 5) :=
by
  sorry

end new_median_is_five_l378_378541


namespace log_base_one_fourth_of_sixteen_l378_378265

theorem log_base_one_fourth_of_sixteen : log (1/4) 16 = -2 :=  sorry

end log_base_one_fourth_of_sixteen_l378_378265


namespace smallest_base_10_integer_l378_378873

theorem smallest_base_10_integer (a b : ℕ) (ha : a > 2) (hb : b > 2) 
  (h1: 21_a = 2 * a + 1) (h2: 12_b = b + 2) : 2 * a + 1 = 7 :=
by 
  sorry

end smallest_base_10_integer_l378_378873


namespace probability_triangle_or_hexagon_l378_378785

theorem probability_triangle_or_hexagon
  (num_triangles : ℕ) (num_squares : ℕ) (num_circles : ℕ) (num_hexagons : ℕ)
  (total_figures : ℕ) :
  (num_triangles = 3) → (num_squares = 4) → (num_circles = 3) → (num_hexagons = 2) →
  (total_figures = 12) →
  (num_triangles + num_hexagons) / total_figures = 5 / 12 :=
by
  intros h_triangles h_squares h_circles h_hexagons h_total
  rw [h_triangles, h_hexagons, h_total]
  norm_num
  sorry

end probability_triangle_or_hexagon_l378_378785


namespace problem_statement_l378_378978

theorem problem_statement : 2456 + 144 / 12 * 5 - 256 = 2260 := 
by
  -- statements and proof steps would go here
  sorry

end problem_statement_l378_378978


namespace angle_equality_l378_378574

variable {P Q R X Y Z : Type*}
variable {A B C D G E F : P}

-- Linear Algebra Basics and Geometric configurations typically found in Mathlib
open_locale real
open affine_geometry

-- Definitions of points and conditions
variable [MetricSpace P] [normed_add_torsor (EuclideanSpace ℝ) P]
variable [MetricSpace Q] [normed_add_torsor (EuclideanSpace ℝ) Q]
variable [MetricSpace R] [normed_add_torsor (EuclideanSpace ℝ) R]
noncomputable theory

-- Variables to represent conditions
variable (h₁ : ∡(B, A, C) = 90°)
variable (h₂ : collinear ℝ ({C, A, D}) ∧ collinear ℝ ({C, A, G}))
variable (h₃ : dist B E = dist E F)

-- Main proof that we need to show
theorem angle_equality 
  (h₄ : ∡(A, B, G) = ∡(D, F, C)) : 
  ∡(A, B, G) = ∡(D, F, C) :=
sorry

end angle_equality_l378_378574


namespace perfect_square_x4_x3_x2_x1_1_eq_x0_l378_378619

theorem perfect_square_x4_x3_x2_x1_1_eq_x0 :
  ∀ x : ℤ, ∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2 ↔ x = 0 :=
by sorry

end perfect_square_x4_x3_x2_x1_1_eq_x0_l378_378619


namespace intervals_of_monotonicity_l378_378293

noncomputable def f (x a : ℝ) : ℝ := real.sqrt x - real.log (x + a)

theorem intervals_of_monotonicity (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 < x → f x a = real.sqrt x - real.log (x + a)) ∧
  ((1 ≤ a → (∀ x : ℝ, 0 < x → (f x a) = (real.sqrt x - real.log (x + a)) → strict_mono_on (f x a) (set.Ioi 0))) ∧
  (0 < a ∧ a < 1 →
    (∀ x : ℝ, 0 < x → x < (1 - real.sqrt(1 - a))^2 → 
      (f x a) = (real.sqrt x - real.log (x + a)) → strict_mono_on (f x a) (set.Ioo 0 (1 - real.sqrt(1 - a))^2)) ∧ 
    (∀ x : ℝ, (1 + real.sqrt (1 - a))^2 < x → 
      (f x a) = (real.sqrt x - real.log (x + a)) → strict_mono_on (f x a) (set.Ioi ((1 + real.sqrt(1 - a))^2))) ∧ 
    (∀ x : ℝ, (1 - real.sqrt(1 - a))^2 < x → x < (1 + real.sqrt (1 - a))^2 → 
      (f x a) = (real.sqrt x - real.log (x + a)) → strict_anti_on (f x a) (set.Ioo (1 - real.sqrt(1 - a))^2 (1 + real.sqrt(1 - a))^2)))) :=
  sorry

end intervals_of_monotonicity_l378_378293


namespace choose_committee_l378_378420

theorem choose_committee (n r : ℕ) (h_n : n = 30) (h_r : r = 5) : nat.choose n r = 118755 := by {
  sorry
}

end choose_committee_l378_378420


namespace probability_same_color_l378_378936

theorem probability_same_color (black_chairs brown_chairs : ℕ)
    (h_black : black_chairs = 15) (h_brown : brown_chairs = 18) :
    let total_chairs := 33 in
    let p_black_black := (15 / 33) * (14 / 32) in
    let p_brown_brown := (18 / 33) * (17 / 32) in
    let p_same_color := p_black_black + p_brown_brown in
    p_same_color = 43 / 88 := 
by
  sorry

end probability_same_color_l378_378936


namespace only_one_prime_in_sequence_l378_378990

def sequence (n : ℕ) : ℕ := Nat.repeat 47 n

theorem only_one_prime_in_sequence :
  ∃! (n : ℕ), Nat.Prime (sequence n) :=
by 
  sorry

end only_one_prime_in_sequence_l378_378990


namespace length_PR_l378_378792

-- Define the basic setup of the circle and points
variables {P Q R O : Point}
variable {r : Real}
variable {chord_length : Real}

-- Main theorem statement
theorem length_PR (hP_circle: dist O P = 8)
                  (hQ_circle: dist O Q = 8)
                  (hPQ_chord: dist P Q = 10)
                  (hR_midpoint : is_midpoint_of_minor_arc R P Q O) :
                  dist P R = 8 * sin (1/2 * acos (25/32)) :=
sorry

end length_PR_l378_378792


namespace pq_sufficient_not_necessary_l378_378917

theorem pq_sufficient_not_necessary (p q : Prop) :
  (¬ (p ∨ q)) → (¬ p ∧ ¬ q) ∧ ¬ ((¬ p ∧ ¬ q) → (¬ (p ∨ q))) :=
sorry

end pq_sufficient_not_necessary_l378_378917


namespace choose_committee_l378_378421

theorem choose_committee (n r : ℕ) (h_n : n = 30) (h_r : r = 5) : nat.choose n r = 118755 := by {
  sorry
}

end choose_committee_l378_378421


namespace Ted_age_48_l378_378576

/-- Given ages problem:
 - t is Ted's age
 - s is Sally's age
 - a is Alex's age 
 - The following conditions hold:
   1. t = 2s + 17 
   2. a = s / 2
   3. t + s + a = 72
 - Prove that Ted's age (t) is 48.
-/ 
theorem Ted_age_48 {t s a : ℕ} (h1 : t = 2 * s + 17) (h2 : a = s / 2) (h3 : t + s + a = 72) : t = 48 := by
  sorry

end Ted_age_48_l378_378576


namespace angle_of_incline_is_approximately_36_degrees_58_minutes_l378_378564
open Real

noncomputable def angle_between_incline_and_horizontal (q : ℝ) :=
  let α := acos 0.8 in 
  α

-- The theorem statement
theorem angle_of_incline_is_approximately_36_degrees_58_minutes (q : ℝ) :
  angle_between_incline_and_horizontal q ≈ 36.87 :=
sorry

end angle_of_incline_is_approximately_36_degrees_58_minutes_l378_378564


namespace find_number_l378_378835

theorem find_number :
  ∃ (x : ℤ), (∃ (y : ℤ), y = 30) ∧ 30 - x = 7 ↔ x = 23 :=
by
  existsi (23 : ℤ)
  split
  sorry

end find_number_l378_378835


namespace area_enclosed_by_graph_l378_378863

theorem area_enclosed_by_graph (x y : ℝ) (h : abs (5 * x) + abs (3 * y) = 15) : 
  ∃ (area : ℝ), area = 30 :=
sorry

end area_enclosed_by_graph_l378_378863


namespace man_work_alone_days_l378_378550

theorem man_work_alone_days:
  ∃ M : ℚ, (M : ℚ) = 1/5 :=
begin
  -- Definitions similar to conditions
  let combined_work_rate := 1/4,
  let son_work_rate := 1/20,

  -- Statement to prove
  have man_work_rate := combined_work_rate - son_work_rate,
  use man_work_rate,
  norm_num at man_work_rate,
  exact man_work_rate,
end

end man_work_alone_days_l378_378550


namespace solve_system1_l378_378458

structure SystemOfEquations :=
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)

def system1 : SystemOfEquations :=
  { a₁ := 1, b₁ := -3, c₁ := 4,
    a₂ := 2, b₂ := -1, c₂ := 3 }

theorem solve_system1 :
  ∃ x y : ℝ, x - 3 * y = 4 ∧ 2 * x - y = 3 ∧ x = 1 ∧ y = -1 :=
by
  sorry

end solve_system1_l378_378458


namespace factorize_expression_l378_378614

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 := 
  sorry

end factorize_expression_l378_378614


namespace log_base_frac_l378_378234

theorem log_base_frac (x : ℝ) : log (1/4) 16 = x → x = -2 := by
  sorry

end log_base_frac_l378_378234


namespace find_a_l378_378680

def f (a x : ℝ) : ℝ := (a - 1) * x^2 + a * sin x

theorem find_a (a : ℝ) : (∀ x : ℝ, f a x = f a (-x)) ↔ a = 0 :=
by
  intros
  sorry

end find_a_l378_378680


namespace perfect_square_x4_x3_x2_x1_1_eq_x0_l378_378618

theorem perfect_square_x4_x3_x2_x1_1_eq_x0 :
  ∀ x : ℤ, ∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2 ↔ x = 0 :=
by sorry

end perfect_square_x4_x3_x2_x1_1_eq_x0_l378_378618


namespace cookies_needed_for_goal_l378_378386

theorem cookies_needed_for_goal
  (brownies_sold : ℕ)
  (brownie_price : ℕ)
  (lemon_squares_sold : ℕ)
  (lemon_square_price : ℕ)
  (goal : ℕ)
  (cookie_price : ℕ)
  (discount : ℕ)
  (three_cookies : ℕ)
  (brownies_sold = 4)
  (brownie_price = 3)
  (lemon_squares_sold = 5)
  (lemon_square_price = 2)
  (goal = 50)
  (cookie_price = 4)
  (discount = 2)
  (three_cookies = 3) :
  ∃ (cookies_needed : ℕ), cookies_needed = 9 :=
by 
  sorry

end cookies_needed_for_goal_l378_378386


namespace cross_section_area_SABC_l378_378295

theorem cross_section_area_SABC {SO : ℝ} {a : ℝ} (SO_eq : SO = 3) (a_eq : a = 6)
  (AP_POr_ratio : ∀ (AP PO' : ℝ), AP / PO' = 8) :
  let base_area := (sqrt 3 / 4) * a ^ 2 in
  let height_ratio := (1 / 9) in
  let section_area := height_ratio * base_area in
  section_area = sqrt 3 :=
by
  sorry

end cross_section_area_SABC_l378_378295


namespace sugar_needed_proof_l378_378156

-- Define the given mixed number as an improper fraction
def mixed_to_improper (a b c : ℕ) : ℚ := a + b / c

-- Define the operation to compute one-third of a given quantity
def one_third (x : ℚ) : ℚ := x / 3

-- Define conversion from improper fraction back to mixed number
def improper_to_mixed (x : ℚ) : ℕ × ℚ :=
  let n := x.natAbs
  let denom := x.denom
  let num := x.num
  (n / denom, num % denom)

-- Given: recipe requires 5 3/4 cups of sugar
def sugar_amount : ℚ := mixed_to_improper 5 3 4

-- Question: if you make one-third of the recipe, how much sugar is needed?
def sugar_needed : ℚ := one_third sugar_amount

-- Proof that the sugar needed for one-third of the recipe is 1 11/12 cups of sugar
theorem sugar_needed_proof : improper_to_mixed sugar_needed = (1, 11 / 12) := by
  sorry

end sugar_needed_proof_l378_378156


namespace prove_perpendicular_l378_378396

open EuclideanGeometry

noncomputable def problem_statement (A B C D M S X Y O : Point) (w₁ w₂ : Circle) : Prop := 
  is_trapezoid A B C D ∧
  parallel BC AD ∧
  AD > BC ∧
  intersection (line AC) (line BD) M ∧
  circle_tangent_on_line w₁ AD A ∧
  circle_passes_through_point w₁ M ∧
  circle_tangent_on_line w₂ AD D ∧
  circle_passes_through_point w₂ M ∧
  intersection (line AB) (line DC) S ∧
  intersection (line AS) w₁ X ∧
  intersection (line DS) w₂ Y ∧
  is_circumcenter O A S D

theorem prove_perpendicular (A B C D M S X Y O : Point) (w₁ w₂ : Circle) :
  problem_statement A B C D M S X Y O w₁ w₂ → perpendicular (line SO) (line XY) :=
by sorry

end prove_perpendicular_l378_378396


namespace max_possible_N_l378_378372

theorem max_possible_N (cities roads N : ℕ) (h1 : cities = 1000) (h2 : roads = 2017) (h3 : N = roads - (cities - 1 + 7 - 1)) :
  N = 1009 :=
by {
  sorry
}

end max_possible_N_l378_378372


namespace sqrt_expression_pattern_l378_378334

theorem sqrt_expression_pattern (n : ℕ) :
  (sqrt (1 + 1/(n^2 : ℝ) + 1/((n+1)^2 : ℝ)) = 1 + 1/(n*(n+1) : ℝ)) :=
by
  sorry

end sqrt_expression_pattern_l378_378334


namespace num_positive_integers_k_lt_1995_with_a_n_zero_l378_378598

def seq_a (a_init k: ℕ) (step: ℕ → ℕ → ℕ) : ℕ → ℕ
| 0     := k
| (n+1) := step (seq_a a_init k n) (seq_b 4 n)

def seq_b (b_init: ℕ) : ℕ → ℕ
| 0     := b_init
| (n+1) := if (seq_a a_init k n) % 2 = 0 then 2 * (seq_b b_init n) else seq_b b_init n

def seq_c (c_init: ℕ) : ℕ → ℕ
| 0     := c_init
| (n+1) := if (seq_a a_init k n) % 2 = 0 then seq_c c_init n else (seq_b 4 n) + (seq_c c_init n)

def step (a_n b_n: ℕ) : ℕ :=
if a_n % 2 = 0 then a_n / 2 else a_n - (b_n / 2) - c_n

def has_zero_in_seq_a_range (k: ℕ) : Prop :=
∃ n, seq_a k (step seq_a k) n = 0

theorem num_positive_integers_k_lt_1995_with_a_n_zero : 
  {k // k < 1995 ∧ has_zero_in_seq_a_range k}.to_finset.card = 31 := by 
sorry

end num_positive_integers_k_lt_1995_with_a_n_zero_l378_378598


namespace monkey_permutations_l378_378338

theorem monkey_permutations : 
  let word := "monkey" in
  (word.length = 6) → 
  nat.factorial word.length = 720 :=
by
  sorry

end monkey_permutations_l378_378338


namespace first_number_removed_l378_378811

theorem first_number_removed (X : ℤ) (S : ℤ) :
  let n := 50
  let removed1 := 55
  let avg1 := 56
  let avg2 := 56.25
  S = avg1 * n →
  ((S - X - removed1) / (n - 2) = avg2) →
  X = 45 :=
by
  sorry

end first_number_removed_l378_378811


namespace range_of_a_minus_b_l378_378672

theorem range_of_a_minus_b (a b : ℝ) (h₁ : a > 0) (h₂ : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (ax^2 + bx - 1 = 0) ∧ r1 ∈ (1, 2)) : 
  (a - b) ∈ set.Ioi (-1 : ℝ) :=
sorry

end range_of_a_minus_b_l378_378672


namespace gcd_of_ratios_l378_378350

noncomputable def gcd_of_two_ratios (A B : ℕ) : ℕ :=
  if h : A % B = 0 then B else gcd B (A % B)

theorem gcd_of_ratios (A B : ℕ) (k : ℕ) (h1 : Nat.lcm A B = 180) (h2 : A = 2 * k) (h3 : B = 3 * k) : gcd_of_two_ratios A B = 30 :=
  by
    sorry

end gcd_of_ratios_l378_378350


namespace find_product_l378_378907

theorem find_product
  (a b c d : ℝ) :
  3 * a + 2 * b + 4 * c + 6 * d = 60 →
  4 * (d + c) = b^2 →
  4 * b + 2 * c = a →
  c - 2 = d →
  a * b * c * d = 0 :=
by
  sorry

end find_product_l378_378907


namespace decrease_in_demand_is_833_percent_l378_378544

variable (P Q : ℝ) -- original price and quantity
variable (income_increase target_income : ℝ)
variable (Q' : ℝ) -- new quantity sold
variable (price_increase : ℝ) := 0.20
variable (goal_income_increase : ℝ) := 0.10
variable (new_price : ℝ) := 1.20 * P

-- The goal is to achieve a 10% increase in income
def income_condition : Prop :=
  new_price * Q' = (1 + goal_income_increase) * P * Q

-- Calculate the new quantity Q'
def new_quantity : ℝ :=
  (1 + goal_income_increase) * P * Q / new_price

-- Calculate the percentage decrease in demand
def percentage_decrease_in_demand : ℝ :=
  1 - new_quantity / Q

-- Prove that the percentage decrease is approximately 8.33%
theorem decrease_in_demand_is_833_percent :
  income_condition →
  percentage_decrease_in_demand = 1 / 12 :=
by
  sorry

end decrease_in_demand_is_833_percent_l378_378544


namespace tony_walks_miles_each_morning_l378_378848

-- Define the conditions

variable (W : ℕ) -- The number of miles Tony walks with the backpack each morning
variable (walk_speed run_distance run_speed total_exercise_hours_per_week days_per_week : ℕ)

-- Assume the given conditions:
-- 1. Speed of walking = 3 miles per hour
def walk_speed : ℕ := 3
-- 2. Distance of running without backpack = 10 miles
def run_distance : ℕ := 10
-- 3. Speed of running = 5 miles per hour
def run_speed : ℕ := 5
-- 4. Total hours spent exercising per week = 21 hours
def total_exercise_hours_per_week : ℕ := 21
-- 5. Number of days per week Tony exercises = 7 days
def days_per_week : ℕ := 7

-- The equation derived from the given problem
def exercise_time_per_day (W : ℕ) : ℕ := W / walk_speed + run_distance / run_speed

theorem tony_walks_miles_each_morning : 
  days_per_week * exercise_time_per_day W = total_exercise_hours_per_week → W = 3 := 
by {
  sorry
}

end tony_walks_miles_each_morning_l378_378848


namespace max_possible_N_l378_378371

theorem max_possible_N (cities roads N : ℕ) (h1 : cities = 1000) (h2 : roads = 2017) (h3 : N = roads - (cities - 1 + 7 - 1)) :
  N = 1009 :=
by {
  sorry
}

end max_possible_N_l378_378371


namespace sqrt_a_mul_cbrt_a_l378_378637

theorem sqrt_a_mul_cbrt_a (a : ℝ) (h : a > 0) : sqrt (a * cbrt a) = a ^ (2 / 3) :=
sorry

end sqrt_a_mul_cbrt_a_l378_378637


namespace lowest_possible_prices_l378_378947

noncomputable def maxDiscountedPrice (price : ℝ) (discount : ℝ) : ℝ :=
  price - (price * discount)

noncomputable def finalDiscountedPrice (price : ℝ) (regularDiscount : ℝ) (saleDiscount : ℝ) : ℝ :=
  let discountedPrice := maxDiscountedPrice(price, regularDiscount)
  discountedPrice - (discountedPrice * saleDiscount)

def manufacturerPrice_A : ℝ := 35.00
def manufacturerPrice_B : ℝ := 28.00
def manufacturerPrice_C : ℝ := 45.00

def regularDiscount : ℝ := 0.30
def saleDiscount : ℝ := 0.20

theorem lowest_possible_prices :
  finalDiscountedPrice manufacturerPrice_A regularDiscount saleDiscount = 19.60 ∧
  finalDiscountedPrice manufacturerPrice_B regularDiscount saleDiscount = 15.68 ∧
  finalDiscountedPrice manufacturerPrice_C regularDiscount saleDiscount = 25.20 := by
  sorry

end lowest_possible_prices_l378_378947


namespace find_m_l378_378565

noncomputable def f (x : ℝ) (c : ℝ) (m : ℝ) : ℝ :=
if x < m then c / real.sqrt x else c / real.sqrt m

theorem find_m
  (m c : ℝ)
  (h1 : f m c m = 15)
  (h2 : f 4 c m = 30) :
  m = 16 :=
by
  sorry

end find_m_l378_378565


namespace rhombus_area_l378_378275

-- Definitions
def side_length := 25 -- cm
def diagonal1 := 30 -- cm

-- Statement to prove
theorem rhombus_area (s : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_s : s = 25) 
  (h_d1 : d1 = 30)
  (h_side : s^2 = (d1/2)^2 + (d2/2)^2) :
  (d1 * d2) / 2 = 600 :=
by sorry

end rhombus_area_l378_378275


namespace third_group_selected_number_l378_378605

theorem third_group_selected_number (total_students : ℕ) (selected_students : ℕ) (initial_number : ℕ) (step : ℕ) 
  (h1 : total_students = 100) (h2 : selected_students = 10) (h3 : initial_number = 3) (h4 : step = total_students / selected_students) :
  initial_number + (3 - 1) * step = 23 := by
  simp [h1, h2, h3, h4]
  norm_num
  sorry

end third_group_selected_number_l378_378605


namespace hexagon_extension_length_l378_378023

theorem hexagon_extension_length 
  (CD : ℝ) 
  (hexagon_side : ℝ)
  (H1 : hexagon_side = 3)
  (H2 : CD = hexagon_side)
  (CY : ℝ)
  (H3 : CY = 4 * CD) : 
  AY = 15 * Real.sqrt 3 / 2 :=
by
  apologize -- to replace with actual proof

end hexagon_extension_length_l378_378023


namespace maximum_area_l378_378649

variable {A B C : Type}
variable [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]

-- Given conditions
variables (a b c : A)
variables (R : B)
variables (triangle : Type)
variables (circumradius : triangle → B)
variables (area : triangle → C)

-- Condition: Given the equation a^2 + b^2 = c^2 + 2/3 * a * b
axiom h1 : a^2 + b^2 = c^2 + (2/3) * a * b

-- Condition: The circumradius R of triangle is 3 * sqrt(2) / 2
axiom h2 : R = (3 * sqrt(2) / 2)

-- Objective: Prove that the maximum possible area is 4 * sqrt(2)
theorem maximum_area (t : triangle) 
  (ht1 : circumradius t = R) 
  (ht2 : area t = (1/2) * a * b * (2 * sqrt(2) / 3))
  (ht3 : ∃ a b c : A, h1 ∧ circumradius t = h2) :
  area t ≤ 4 * sqrt(2) := 
sorry

end maximum_area_l378_378649


namespace solve_cubic_trig_equation_l378_378457

theorem solve_cubic_trig_equation (θ : ℝ) (x : ℝ)
  (h : (cos θ)^2 * x^3 - (1 + 3 * (sin θ)^2) * x + sin (2 * θ) = 0) :
  (cos θ = 0 → x = 0) ∧ (cos θ ≠ 0 → (x = 2 * tan θ ∨ x = -tan θ + sec θ ∨ x = -tan θ - sec θ)) :=
by
  sorry

end solve_cubic_trig_equation_l378_378457


namespace find_percentage_l378_378139

theorem find_percentage (P : ℝ) :
  (P / 100) * 1280 = ((0.20 * 650) + 190) ↔ P = 25 :=
by
  sorry

end find_percentage_l378_378139


namespace angle_comparison_l378_378294

open_locale classical

noncomputable def midpoint_arc (A B : Point) (C : Circle) : Point :=
sorry -- Assume this function gives the midpoint of arc AB.

theorem angle_comparison (C : Circle) (O : Point) (r : ℝ) (P A B : Point)
  (hP_inside : inside_circle P O r)
  (hA_on : on_circle A O r)
  (hB_on : on_circle B O r)
  (hPA_lt_PB : dist P A < dist P B) :
  let F := midpoint_arc A B C in 
  angle A P F > angle F P B :=
sorry

end angle_comparison_l378_378294


namespace length_of_segment_XY_l378_378497

noncomputable def rectangle_length (A B C D : ℝ) (BX DY : ℝ) : ℝ :=
  2 * BX + DY

theorem length_of_segment_XY (A B C D : ℝ) (BX DY : ℝ) (h1 : C = 2 * B) (h2 : BX = 4) (h3 : DY = 10) :
  rectangle_length A B C D BX DY = 13 :=
by
  rw [rectangle_length, h2, h3]
  sorry

end length_of_segment_XY_l378_378497


namespace length_of_MN_l378_378038

-- Define the circles with a common intersection point A
variables {K1 K2 : Circle} (A : Point) (B C M N : Point)

-- Conditions
-- Two circles K1 and K2 intersect at point A
axiom intersect_at_A : A ∈ K1 ∧ A ∈ K2
-- Through point A, two lines: AB and AC intersect circles at B and C respectively
axiom lines_through_centers : centered_at K1 A B ∧ centered_at K2 A C
-- Length of segment BC is given as a
variable (a : ℝ)
axiom length_of_BC : dist B C = a
-- The third line through A is parallel to BC and intersects the circles at M and N
axiom line_parallel_to_BC : parallel (line_through A M) (line_through B C) ∧ parallel (line_through A N) (line_through B C)
axiom intersects_at_MN : M ∈ K1 ∧ N ∈ K2

-- Proof problem: show that the length of segment MN is equal to a
theorem length_of_MN : dist M N = a :=
sorry

end length_of_MN_l378_378038


namespace solve_problem_l378_378071

open Matrix

variables {R : Type*} [Field R]
variables {S : Matrix (Fin 3) (Fin 1) R → Matrix (Fin 3) (Fin 1) R}
variables {a b : R} {u v : Matrix (Fin 3) (Fin 1) R}
variables {w x : Matrix (Fin 3) (Fin 1) Int}

-- conditions
def cond1 : Prop := ∀ (a b : R) (u v : Matrix (Fin 3) (Fin 1) R), 
  S (a • u + b • v) = a • (S u) + b • (S v)
def cond2 : Prop := ∀ (u v : Matrix (Fin 3) (Fin 1) R),
  S (u × v) = S u × S v
def cond3 : S (λ i, ![5, 2, 7]) = (λ i, ![1, 3, 4])
def cond4 : S (λ i, ![3, 7, 2]) = (λ i, ![4, 6, 5])

-- desired outcome
def target : Prop := S (λ i, ![4, 11, 9]) = (λ i, ![5, 10, 9])

theorem solve_problem
  (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : target := by
  sorry

end solve_problem_l378_378071


namespace distance_between_points_l378_378688

theorem distance_between_points (A B : ℝ) (dA : |A| = 2) (dB : |B| = 7) : |A - B| = 5 ∨ |A - B| = 9 := 
by
  sorry

end distance_between_points_l378_378688


namespace proof_acd_over_b_eq_neg_52_5_l378_378277

noncomputable def largest_x (a b c d : ℤ) :=
  (a + b * Real.sqrt c) / d

theorem proof_acd_over_b_eq_neg_52_5
  (a b c d : ℤ)
  (ha : a = -4)
  (hb : b = 8)
  (hc : c = 15)
  (hd : d = 7)
  (h : 7 * (largest_x a b c d)^2 + 8 * (largest_x a b c d) - 32 = 0)
  : (a * c * d) / b = -52.5 :=
by sorry

end proof_acd_over_b_eq_neg_52_5_l378_378277


namespace cover_one_mile_stretch_l378_378957

noncomputable def time_to_cover_stretch (highway_length miles radius feet speed mph : ℝ) : ℝ :=
  (106 * (π * radius) / 5280) / speed

theorem cover_one_mile_stretch :
  time_to_cover_stretch 1 0.5 25 6 = π / 12 := 
by
  sorry

end cover_one_mile_stretch_l378_378957


namespace real_solutions_121_l378_378624

noncomputable def f (x : ℝ) : ℝ :=
  ∑ i in Finset.range 120, (i + 1) / (x - (i + 1))

theorem real_solutions_121 : ∃ (s : Finset ℝ), s.card = 121 ∧ ∀ x ∈ s, f x = 2 * x :=
sorry

end real_solutions_121_l378_378624


namespace smallest_integer_value_l378_378490

theorem smallest_integer_value (a : ℕ → ℝ) (h_sum : ∑ i in finset.range 8, a i = 4 / 3)
    (h_pos_sum7 : ∀ i < 8, ∑ j in finset.range 8 \ {i}, a j > 0) : 
    ∃ (a_min : ℤ), (∀ i < 8, a i ≥ a_min) ∧ (∀ m < a_min, ∃ i < 8, a i < m) ∧ a_min = -7 :=
by
  sorry

end smallest_integer_value_l378_378490


namespace num_sets_A_l378_378479

-- Define the set U = {1, 2, 3}
def U : set ℕ := {1, 2, 3}

-- Define the condition that {1} is a subset of A and A is a subset of U
def satisfies_conditions (A : set ℕ) : Prop :=
  {1} ⊆ A ∧ A ⊆ U

-- Define and state the theorem
theorem num_sets_A : 
  ∃ n : ℕ, n = 3 ∧ ∀ (A : set ℕ), satisfies_conditions A → (A = {1, 2} ∨ A = {1, 3} ∨ A = {1, 2, 3}):=
begin
  use 3,
  split,
  { refl },
  { intros A h,
    rw satisfies_conditions at h,
    cases h with h1 h2,
    sorry -- proof skipped
  }
end

end num_sets_A_l378_378479


namespace standard_ellipse_eq_l378_378837

def ellipse_standard_eq (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem standard_ellipse_eq (P: ℝ × ℝ) (Q: ℝ × ℝ) (a b : ℝ) (h1 : P = (-3, 0)) (h2 : Q = (0, -2)) :
  ellipse_standard_eq 3 2 x y :=
by
  sorry

end standard_ellipse_eq_l378_378837


namespace greatest_x_l378_378097

theorem greatest_x (x : ℕ) (h : x > 0 ∧ (x^4 / x^2 : ℚ) < 18) : x ≤ 4 :=
by
  sorry

end greatest_x_l378_378097


namespace hyperbola_asymptotes_l378_378475

theorem hyperbola_asymptotes 
  (m : ℝ)
  (h : 2 * real.sqrt (-1 / m) = 4) : (∀ x y : ℝ, x^2 + m * y^2 = 1 → y = 2 * x ∨ y = -2 * x) :=
begin
  sorry
end

end hyperbola_asymptotes_l378_378475


namespace julie_school_hours_per_week_l378_378748

def summer_hours_per_week := 48
def summer_weeks := 8
def total_summer_earnings := 7000
def school_weeks := 48

theorem julie_school_hours_per_week : 
  let hourly_pay := (total_summer_earnings : ℝ) / (summer_hours_per_week * summer_weeks)
  let required_school_income_weeks := (total_summer_earnings : ℝ) / school_weeks
  let required_school_hours_per_week := required_school_income_weeks / hourly_pay
  required_school_hours_per_week ≈ 8 :=
sorry

end julie_school_hours_per_week_l378_378748


namespace angle_BAD_is_20_l378_378383

-- Defining the conditions given in the problem
variable {A B C D : Type}
-- Points A, B, C, and D are in plane
variable [inhabited A] [inhabited B] [inhabited C] [inhabited D]
-- Point D is on line segment BC
variable (on_BC : ∃ (t : ℝ), D = t • B + (1 - t) • C)
-- Given angles
variable (angle_DAC : ℝ := 20) (angle_DBC : ℝ := 60)

-- The theorem to be proved
theorem angle_BAD_is_20 :
  ∃ (angle_BAD : ℝ), angle_DAC = 20 ∧ angle_DBC = 60 → angle_BAD = 20 :=
by
  sorry

end angle_BAD_is_20_l378_378383


namespace perpendicular_vector_solution_l378_378689

theorem perpendicular_vector_solution 
    (a b : ℝ × ℝ) (m : ℝ) 
    (h_a : a = (1, -1)) 
    (h_b : b = (-2, 3)) 
    (h_perp : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) 
    : m = 2 / 5 := 
sorry

end perpendicular_vector_solution_l378_378689


namespace coloring_hexagonal_pyramids_l378_378986

def color_ways : ℕ := 405

theorem coloring_hexagonal_pyramids :
  ∃ n : ℕ, n = color_ways :=
by {
  use 405,
  sorry
}

end coloring_hexagonal_pyramids_l378_378986


namespace xy_addition_l378_378854

theorem xy_addition (x y : ℕ) (h1 : x * y = 24) (h2 : x - y = 5) (hx_pos : 0 < x) (hy_pos : 0 < y) : x + y = 11 := 
sorry

end xy_addition_l378_378854


namespace petrol_price_increase_l378_378058

theorem petrol_price_increase
  (P P_new : ℝ)
  (C : ℝ)
  (h1 : P * C = P_new * (C * 0.7692307692307693))
  (h2 : C * (1 - 0.23076923076923073) = C * 0.7692307692307693) :
  ((P_new - P) / P) * 100 = 30 := 
  sorry

end petrol_price_increase_l378_378058


namespace min_elements_in_set_A_l378_378484
open Set

theorem min_elements_in_set_A :
  ∃ (A : Set ℤ) (h0 : A ⊆ ℤ) 
  (h1 : 1 ∈ A) 
  (h2 : 100 ∈ A) 
  (h3 : ∀ (a ∈ A), a ≠ 1 → (∃ b c ∈ A, a = b + c)), 
  ∀ (B : Set ℤ), (B ⊆ ℤ) → 
  (1 ∈ B) → 
  (100 ∈ B) → 
  (∀ (a ∈ B), a ≠ 1 → (∃ b c ∈ B, a = b + c)) → 
  9 ≤ Set.card A ∧ Set.card B = 9 :=
sorry

end min_elements_in_set_A_l378_378484


namespace cannot_be_sum_of_even_integers_l378_378822

theorem cannot_be_sum_of_even_integers (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 2000) 
  (h₂ : ∃ k : ℕ, 2 * k ≤ 2000 ∧ n = (list.range (2 * k)).sum) : 
  n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 200098 ∧ n ≠ 200099 :=
sorry

end cannot_be_sum_of_even_integers_l378_378822


namespace b_completion_days_l378_378115

theorem b_completion_days (x : ℝ) :
  (7 * (1 / 24 + 1 / x + 1 / 40) + 4 * (1 / 24 + 1 / x) = 1) → x = 26.25 := 
by 
  sorry

end b_completion_days_l378_378115


namespace pyramid_cross_section_quadrilateral_l378_378438

noncomputable def cross_section_of_pyramid (A B C D K M : Type)
  [Plane ABCD] [Edge CD K] [Edge AB M] [Line AD] : Type :=
  exists (quadrilateral : Type), 
    is_cross_section_pyramid_quadrilateral quadrilateral ABCD K M AD

-- Further Lean definitions to fully define the context and assumptions
variables (A B C D K M : Type)
variables [Plane ABCD] [Edge CD K] [Edge AB M] [Line AD]

theorem pyramid_cross_section_quadrilateral :
  cross_section_of_pyramid A B C D K M :=
begin
  -- the proof would go here, but per the instructions, we are leaving it as a sorry.
  sorry,
end

end pyramid_cross_section_quadrilateral_l378_378438


namespace sqrt_product_consecutive_integers_l378_378589

theorem sqrt_product_consecutive_integers :
  sqrt (43 * 42 * 41 * 40 + 1) = 1721 := 
sorry

end sqrt_product_consecutive_integers_l378_378589


namespace greatest_integer_value_l378_378866

theorem greatest_integer_value (x : ℤ) : 7 - 3 * x > 20 → x ≤ -5 :=
by
  intros h
  sorry

end greatest_integer_value_l378_378866


namespace part_I_solution_set_part_II_range_of_a_l378_378678

-- Define the absolute value function
def abs (x : ℝ) : ℝ := if x < 0 then -x else x

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a)

-- Proof statement for part (I)
theorem part_I_solution_set (x : ℝ) :
  ∀ (a : ℝ), a = 1 → f x a ≥ abs(x + 1) + 1 ↔ x > 0.5 :=
sorry

-- Proof statement for part (II)
theorem part_II_range_of_a (a : ℝ) :
  (∀ x, f x a + 3 * x ≤ 0 → x ≤ -1) → -4 ≤ a ∧ a ≤ 2 :=
sorry

end part_I_solution_set_part_II_range_of_a_l378_378678


namespace sum_even_integers_l378_378068

theorem sum_even_integers (sum_first_50_even : Nat) (sum_from_100_to_200 : Nat) : 
  sum_first_50_even = 2550 → sum_from_100_to_200 = 7550 :=
by
  sorry

end sum_even_integers_l378_378068


namespace sum_b_lt_n_plus_one_l378_378762

noncomputable def a : ℕ → ℝ
| n := 4 * n - 2

def b (n : ℕ) : ℝ := 
  if h : n > 0 then 
    let n' := n.succ in
    (a n' / a n + a n / a n') / 2 
  else 0

theorem sum_b_lt_n_plus_one (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range n.succ, b i) < n + 1 :=
sorry

end sum_b_lt_n_plus_one_l378_378762


namespace x_intercepts_count_l378_378215

theorem x_intercepts_count (y : ℝ → ℝ) (h : ∀ x, y x = Real.sin (1 / x))
  (a b : ℝ) (h1 : 0.0002 < a) (h2 : b < 0.002) :
  {x : ℝ | a < x ∧ x < b ∧ y x = 0}.to_finset.card = 1433 :=
by 
  sorry

end x_intercepts_count_l378_378215


namespace rhombus_inscribed_circle_tangent_independence_l378_378958

theorem rhombus_inscribed_circle_tangent_independence
  (rhombus: Type)
  (A B C D : rhombus)
  (I : rhombus) -- I denotes the inscribed circle
  (tangent : Π (l : rhombus), Prop)
  (l : rhombus) -- tangent line
  (E F : rhombus) -- points where tangent meets the sides
  (h1 : tangent l)
  (h2 : meets l A B E)
  (h3 : meets l B C F) :
  ∃ k, AE * CF = k :=
sorry

end rhombus_inscribed_circle_tangent_independence_l378_378958


namespace range_of_m_l378_378352

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + 6 < 4x - 3 ∧ x > m → x > 3) → m ≤ 3 :=
by
  intros h
  sorry

end range_of_m_l378_378352


namespace probability_of_multiple_135_l378_378705

noncomputable def single_digit_multiples_of_3 := {3, 6, 9} : Finset ℕ
noncomputable def primes_less_than_50 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47} : Finset ℕ
def product_is_multiple_of_135 (a b : ℕ) : Prop := a * b ∣ 135

theorem probability_of_multiple_135 :
  (1 / 45 : ℚ) = 
  (single_digit_multiples_of_3.card * primes_less_than_50.card).toRat / 
  (single_digit_multiples_of_3.filter (λ a, primes_less_than_50.card * a ∣ 135)).card :=
by 
  sorry

end probability_of_multiple_135_l378_378705


namespace exists_f_l378_378397

open Nat

noncomputable def iterated (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, n := n
| (k + 1), n := f (iterated f k n)

theorem exists_f : ∃ f : ℕ → ℕ, ∀ n, iterated f 2003 n = 5 * n :=
sorry

end exists_f_l378_378397


namespace max_sum_between_pairs_l378_378782

open Nat

theorem max_sum_between_pairs : 
  let cards := [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
  ∃ (perm : List ℕ) (card_indices : List (Nat × ℕ)), 
  perm ~ cards ∧ -- perm is a permutation of cards
  (∀ n ∈ [1, 2, 3, 4, 5], ∃ i j, i < j ∧ (perm.get? i = some n) ∧ (perm.get? j = some n) ∧ 
  card_indices n = ((j - i - 1) : ℕ)) ∧ 
  (∑ k in card_indices, k) = 20 := 
sorry

end max_sum_between_pairs_l378_378782


namespace no_real_roots_l378_378325

noncomputable def f (x : ℝ) : ℝ := 2^(x+1)
noncomputable def g (x : ℝ) : ℝ := 2^x + 2^(-x)  -- Given g(x) is even
noncomputable def h (x : ℝ) : ℝ := 2^x - 2^(-x)  -- Given h(x) is odd

def t (x : ℝ) : ℝ := h(x)

noncomputable def p (t : ℝ) (m : ℝ) : ℝ :=
  (t^2 + 2*m*t + m^2 - m + 1)

theorem no_real_roots (m : ℝ) : p (p (t 0)) m ≠ 0 :=
begin
  sorry
end

end no_real_roots_l378_378325


namespace product_representation_count_l378_378369

theorem product_representation_count :
  let n := 1000000
  let distinct_ways := 139
  (∃ (a b c d e f : ℕ), 2^(a+b+c) * 5^(d+e+f) = n ∧ 
    a + b + c = 6 ∧ d + e + f = 6 ) → 
    139 = distinct_ways := 
by {
  sorry
}

end product_representation_count_l378_378369


namespace carl_typing_hours_per_day_l378_378183

theorem carl_typing_hours_per_day (words_per_minute : ℕ) (total_words : ℕ) (days : ℕ) (hours_per_day : ℕ) :
  words_per_minute = 50 →
  total_words = 84000 →
  days = 7 →
  hours_per_day = (total_words / days) / (words_per_minute * 60) →
  hours_per_day = 4 :=
by
  intros h_word_rate h_total_words h_days h_hrs_formula
  rewrite [h_word_rate, h_total_words, h_days] at h_hrs_formula
  exact h_hrs_formula

end carl_typing_hours_per_day_l378_378183


namespace total_distance_travelled_l378_378575

noncomputable def initial_speed : ℝ := 30 -- The initial speed in km/h
noncomputable def speed_increment : ℝ := 6 -- The speed increment every 5 minutes in km/h
noncomputable def time_interval_minutes : ℕ := 5 -- The time interval in minutes 
noncomputable def total_hours : ℝ := 3 -- The total time to consider in hours
noncomputable def total_intervals : ℕ := 36 -- The total number of 5-minute intervals in 3 hours

theorem total_distance_travelled : 
  let distance := (1 / 12 : ℝ) * (∑ i in finset.range (total_intervals + 1), initial_speed + speed_increment * i) 
  in distance = 425.5 :=
by
  sorry

end total_distance_travelled_l378_378575


namespace existence_of_special_triangle_l378_378755

structure Point where
  x : ℝ
  y : ℝ

-- Definition for a set of points such that no three points are collinear
def no_three_collinear (S : Finset Point) : Prop :=
  ∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → ¬ (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y) = 0)

-- Definition for the convex hull being a polygon
def is_convex_hull_polygon (S : Finset Point) (n : ℕ) : Prop :=
  ∃ (A : Fin n → Point), (∀ i, A i ∈ S) ∧ (ConvexHull ℝ (range A) = S)

-- Assuming the labels for points
def labels (S : Finset Point) : Point → ℤ
| p := if p ∈ S then (ε : ℤ) else 0

def labeling_condition (S : Finset Point) (labels : Point → ℤ) : Prop :=
  ∀ i ∈ Finset.range 1008, labels (∃ (A_i A_{i+1008} : Point), A_i ∈ S ∧ A_{i+1008} ∈ S ∧ labels A_i = -labels A_{i+1008})

theorem existence_of_special_triangle (S : Finset Point)
  (h_no_three_collinear : no_three_collinear S)
  (h_convex_hull : is_convex_hull_polygon S 2016)
  (h_labeling : labeling_condition S labels) : 
  ∃ (A B C : Point) (T : Triangle), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ {A, B, C} ∈ T.simplices ∧ (labels A = -labels B ∨ labels A = -labels C ∨ labels B = -labels C) :=
sorry

end existence_of_special_triangle_l378_378755


namespace dodecahedron_triangle_probability_l378_378496

def regular_dodecahedron_vertices : ℕ := 20

def connected_vertices_per_vertex : ℕ := 3

theorem dodecahedron_triangle_probability :
  (∃ dodecahedron : Type,
    ∃ V E : set (set dodecahedron),
    V.card = regular_dodecahedron_vertices ∧
    (∀ v ∈ V, ∃ connected : finset (set dodecahedron), connected.card = connected_vertices_per_vertex ∧ connected ⊆ (E ∩ {e | v ∈ e})) ∧
    (∃ triangles, triangles.card = (regular_dodecahedron_vertices * connected_vertices_per_vertex) / connected_vertices_per_vertex ∧
    (∀ t ∈ triangles, ∃ v1 v2 v3 ∈ V, t = {v1, v2, v3} ∧ (v1, v2) ∈ E ∧ (v2, v3) ∈ E ∧ (v3, v1) ∈ E)) ∧
    let total_triangles := choose regular_dodecahedron_vertices 3,
        unique_triangles := regular_dodecahedron_vertices in
    unique_triangles / total_triangles = 1 / 57) :=
begin
  sorry
end

end dodecahedron_triangle_probability_l378_378496


namespace price_reduction_for_profit_l378_378563

theorem price_reduction_for_profit (x : ℝ) : 
  (∃ x, -400 * x ^ 2 + 200 * x - 24 = 0) :=
begin
  sorry
end

end price_reduction_for_profit_l378_378563


namespace floor_S_4901_l378_378204

noncomputable def S_n (n : ℕ) :=
  ∑ k in Finset.range n, (1 / Real.sqrt (2 * k + 1 : ℝ))

theorem floor_S_4901 : (⌊S_n 4901⌋ : ℤ) = 98 := sorry

end floor_S_4901_l378_378204


namespace parabola_trajectory_l378_378657

theorem parabola_trajectory :
  ∀ P : ℝ × ℝ, (dist P (0, -1) + 1 = dist P (0, 3)) ↔ (P.1 ^ 2 = -8 * P.2) := by
  sorry

end parabola_trajectory_l378_378657


namespace eccentricity_of_parametric_curve_l378_378214

theorem eccentricity_of_parametric_curve :
  (∀ φ : ℝ, let x := 3 * Real.cos φ in 
             let y := Real.sqrt 5 * Real.sin φ in 
             True) →
  (∃ e : ℝ, e = 2 / 3) :=
by
  intros _
  use 2 / 3
  sorry

end eccentricity_of_parametric_curve_l378_378214


namespace mathborough_total_rainfall_2004_l378_378358

-- Given conditions as variables
def avg_rainfall_2003 : ℕ := 45
def extra_rainfall_2004 : ℕ := 3
def additional_rain_high_month : ℕ := 5
def high_months : ℕ := 8

-- Calculate average monthly rainfall in 2004
def avg_rainfall_2004 : ℕ := avg_rainfall_2003 + extra_rainfall_2004

-- Calculate rainfall in high months
def rainfall_high_months : ℕ := (avg_rainfall_2004 + additional_rain_high_month) * high_months

-- Calculate remaining months in a year
def low_months : ℕ := 12 - high_months

-- Calculate rainfall in low months
def rainfall_low_months : ℕ := avg_rainfall_2004 * low_months

-- Total rainfall in 2004
def total_rainfall_2004 : ℕ := rainfall_high_months + rainfall_low_months

-- Prove the total rainfall
theorem mathborough_total_rainfall_2004 : total_rainfall_2004 = 616 := 
by {
  unfold total_rainfall_2004 rainfall_high_months rainfall_low_months,
  unfold low_months high_months additional_rain_high_month avg_rainfall_2003 extra_rainfall_2004 avg_rainfall_2004,
  norm_num,
}

end mathborough_total_rainfall_2004_l378_378358


namespace smallest_base10_integer_l378_378889

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end smallest_base10_integer_l378_378889


namespace distance_from_foci_to_asymptotes_l378_378043

def hyperbola_foci_distance_to_asymptote (a b c : ℝ) (h₁ : a^2 = 4) (h₂ : b^2 = 12) (h₃ : c^2 = a^2 + b^2) : ℝ :=
  let eq_hyperbola := (a^2 / 4) - (b^2 / 12) = 1
  let eq_asymptotes := (b/a = Real.sqrt3)
  (c / Real.sqrt3)

theorem distance_from_foci_to_asymptotes : hyperbola_foci_distance_to_asymptote 2 (Real.sqrt12) 4 (by norm_num) (by norm_num) (by norm_num) = 2 * Real.sqrt3 := 
by 
  sorry

end distance_from_foci_to_asymptotes_l378_378043


namespace proof_acd_over_b_eq_neg_52_5_l378_378278

noncomputable def largest_x (a b c d : ℤ) :=
  (a + b * Real.sqrt c) / d

theorem proof_acd_over_b_eq_neg_52_5
  (a b c d : ℤ)
  (ha : a = -4)
  (hb : b = 8)
  (hc : c = 15)
  (hd : d = 7)
  (h : 7 * (largest_x a b c d)^2 + 8 * (largest_x a b c d) - 32 = 0)
  : (a * c * d) / b = -52.5 :=
by sorry

end proof_acd_over_b_eq_neg_52_5_l378_378278


namespace smallest_base_10_integer_exists_l378_378877

theorem smallest_base_10_integer_exists :
  ∃ (x a b : ℕ), (a > 2) ∧ (b > 2) ∧ (x = 2 * a + 1) ∧ (x = b + 2) ∧ (x = 7) :=
by
  sorry

end smallest_base_10_integer_exists_l378_378877


namespace union_when_a_eq_2_condition_1_condition_2_condition_3_l378_378587

open Set

def setA (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def setB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_when_a_eq_2 : setA 2 ∪ setB = {x | -1 ≤ x ∧ x ≤ 3} :=
sorry

theorem condition_1 (a : ℝ) : 
  (setA a ∪ setB = setB) → (0 ≤ a ∧ a ≤ 2) :=
sorry

theorem condition_2 (a : ℝ) :
  (∀ x, (x ∈ setA a ↔ x ∈ setB)) → (0 ≤ a ∧ a ≤ 2) :=
sorry

theorem condition_3 (a : ℝ) :
  (setA a ∩ setB = ∅) → (a < -2 ∨ 4 < a) :=
sorry

end union_when_a_eq_2_condition_1_condition_2_condition_3_l378_378587


namespace greek_cross_decomposition_exists_l378_378337

-- Define what it means for a shape to be a Greek cross
structure GreekCross (shape : Type) :=
(center : shape)
(arms : set shape)
(valid : ∀ s ∈ arms, s is_square) -- Assume each arm is a valid square

-- Hypothesis: One larger Greek cross can be cut into parts to form three smaller Greek crosses
def large_to_small_greek_cross_decomposition (larger smaller : GreekCross ℝ) : Prop :=
∃ parts : set (GreekCross ℝ),
  parts.card = 12 ∧
  (∀ s ∈ parts, s.valid) ∧
  disjoint_union parts = larger ∧
  (∀ (A B C : GreekCross ℝ), A ∈ parts ∧ B ∈ parts ∧ C ∈ parts → size A = size B ∧ size B = size C)

-- Statement that needs proof
theorem greek_cross_decomposition_exists :
  ∃ (larger smaller : GreekCross ℝ), large_to_small_greek_cross_decomposition larger smaller :=
sorry

end greek_cross_decomposition_exists_l378_378337


namespace range_of_a_l378_378685

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l378_378685


namespace neg_half_to_fourth_power_eq_sixteenth_l378_378181

theorem neg_half_to_fourth_power_eq_sixteenth :
  (- (1 / 2 : ℚ)) ^ 4 = (1 / 16 : ℚ) :=
sorry

end neg_half_to_fourth_power_eq_sixteenth_l378_378181


namespace max_weight_of_crates_l378_378515

variable (max_crates : ℕ)
variable (weight_per_crate : ℕ)

theorem max_weight_of_crates
  (h_max_crates : max_crates = 5)
  (h_min_weight_per_crate : weight_per_crate ≥ 120) :
  max_crates * 120 = 600 :=
by
  rw [h_max_crates]
  sorry

end max_weight_of_crates_l378_378515


namespace square_is_special_rectangle_l378_378212

-- Definitions based on conditions
def is_rectangle (Q : Type) [Quadrilateral Q] := 
  ∀(a b : side_length Q), opposite_sides Q a b → equal_length a b

def opposite_sides (Q : Type) [Quadrilateral Q] (a b : side_length Q) := -- some condition
sorry

def equal_length (a b : side_length Q) := -- some condition
sorry

def perpendicular_diagonals (Q : Type) [Quadrilateral Q] := 
  ∀(d1 d2 : diagonal_length Q), intersect Q d1 d2 → perpendicular d1 d2

def bisect_angles (Q : Type) [Quadrilateral Q] := 
  ∀(d : diagonal_length Q), bisect_angle Q d

def equal_sides (Q : Type) [Quadrilateral Q] := 
  ∀(s1 s2 : side_length Q), same_side Q s1 s2 → equal_length s1 s2

def is_square (Q : Type) [Quadrilateral Q] := 
  equal_sides Q ∧ perpendicular_diagonals Q ∧ bisect_angles Q

-- Theorem to prove
theorem square_is_special_rectangle (Q : Type) [Quadrilateral Q] 
  (perpendicular_diagonals_Q : perpendicular_diagonals Q)
  (bisect_angles_Q : bisect_angles Q) :
  is_square Q → is_rectangle Q :=
sorry

end square_is_special_rectangle_l378_378212


namespace time_to_cook_rest_of_potatoes_l378_378539

-- Definitions of the conditions
def total_potatoes : ℕ := 12
def already_cooked : ℕ := 6
def minutes_per_potato : ℕ := 6

-- Proof statement
theorem time_to_cook_rest_of_potatoes : (total_potatoes - already_cooked) * minutes_per_potato = 36 :=
by
  sorry

end time_to_cook_rest_of_potatoes_l378_378539


namespace denis_chameleons_total_l378_378997

theorem denis_chameleons_total (t : ℕ) (h1 : ∃ t, 5 * t > 0) (h2 : 5 * t + 2 = 8 * (t - 2)) :
  t + 5 * t = 36 :=
by
  obtain ⟨t, t_pos⟩ := h1
  have eq1 : 5 * t + 2 = 8 * (t - 2) := h2
  sorry

end denis_chameleons_total_l378_378997


namespace kiwi_count_l378_378145

theorem kiwi_count (o a b k : ℕ) (h1 : o + a + b + k = 540) (h2 : a = 3 * o) (h3 : b = 4 * a) (h4 : k = 5 * b) : k = 420 :=
sorry

end kiwi_count_l378_378145


namespace always_true_statements_l378_378802

variable (a b c : ℝ)

theorem always_true_statements (h1 : a < 0) (h2 : a < b ∧ b ≤ 0) (h3 : b < c) : 
  (a + b < b + c) ∧ (c / a < 1) :=
by 
  sorry

end always_true_statements_l378_378802


namespace choose_5_from_30_l378_378423

theorem choose_5_from_30 :
  (nat.choose 30 5) = 54810 :=
by
  sorry

end choose_5_from_30_l378_378423


namespace similar_triangles_side_length_l378_378083

theorem similar_triangles_side_length
  (P Q R X Y Z : Type)
  [metric_space P] [metric_space Q] [metric_space R]
  [metric_space X] [metric_space Y] [metric_space Z]
  (h_sim : similar PQR XYZ)
  (hPQ : dist P Q = 9)
  (hQR : dist Q R = 15)
  (hYZ : dist Y Z = 30) :
  dist X Y = 18 :=
by sorry

end similar_triangles_side_length_l378_378083


namespace pyramid_volume_distance_from_centroid_to_vertex_l378_378194

noncomputable def triangle_vertices : list (ℝ × ℝ) := [(0, 0), (30, 0), (10, 20)]

def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def midpoints : list (ℝ × ℝ) :=
  [midpoint (0, 0) (30, 0), midpoint (0, 0) (10, 20), midpoint (30, 0) (10, 20)]

def centroid (vertices : list (ℝ × ℝ)) : (ℝ × ℝ) :=
  let (xs, ys) := (vertices.map Prod.fst, vertices.map Prod.snd) in
  (xs.sum / xs.length, ys.sum / ys.length)

noncomputable def base_centroid : (ℝ × ℝ) := centroid triangle_vertices

theorem pyramid_volume :
  let base_area := 75
  let height := (20 / 3)
  volume = (1 / 3) * base_area * height :=
sorry

theorem distance_from_centroid_to_vertex (G : ℝ × ℝ) (V : ℝ × ℝ) :
  dist G V = (Real.sqrt 1700) / 3 :=
sorry

end pyramid_volume_distance_from_centroid_to_vertex_l378_378194


namespace smallest_n_for_g_eq_4_l378_378404

def g (n : ℕ) : ℕ :=
  (List.filter (λ ⟨a, b⟩, a^2 + b^2 = n) (List.product (List.range (n + 1)) (List.range (n + 1)))).length

def isOrderedPair (a b : ℕ) : Prop := a^2 + b^2 = n ∧ a ≠ b

theorem smallest_n_for_g_eq_4 : ∃ n : ℕ, g n = 4 ∧ ∀ (m : ℕ), m < n → g m ≠ 4 :=
  exists.intro 65 (and.intro (by sorry) (by sorry))

end smallest_n_for_g_eq_4_l378_378404


namespace log_base_fraction_l378_378224

theorem log_base_fraction : ∀ (a b : ℝ) (x : ℝ), 16 = (4:ℝ)^2 ∧ (1 / 4:ℝ) = 4^(-1) → log (1 / 4) 16 = -2 :=
begin
  intros a b x h,
  -- Skipping the proof by adding sorry
  sorry,
end

end log_base_fraction_l378_378224


namespace debby_soda_bottles_last_l378_378996

theorem debby_soda_bottles_last :
  ∀ (soda water : ℕ), soda = 360 → water = 162 →
  (∀ (r : ℕ), r = (3 : ℕ) → 
  (∀ (w : ℕ), w = (2 : ℕ) → 
  water / w <= soda / r)) →
  water / 2 = 81 :=
by
  intro soda water hsoda hwater r hr w hw
  rw [hwater, hw]
  exact sorry

end debby_soda_bottles_last_l378_378996


namespace parabola_properties_l378_378016

theorem parabola_properties (x : ℝ) :
  (∀ x, y = -3 * x^2 → (parabola_opens_downwards : true) ∧ vertex_at_origin : true) ∧ 
  (∀ x > 0, y = -3 * x^2 → y_decreases_as_x_increases : true) :=
by
  -- proof
  sorry

end parabola_properties_l378_378016


namespace geom_seq_308th_term_l378_378734

def geom_seq (a : ℤ) (r : ℤ) (n : ℕ) : ℤ :=
  a * r ^ n

-- Given conditions
def a := 10
def r := -1

theorem geom_seq_308th_term : geom_seq a r 307 = -10 := by
  sorry

end geom_seq_308th_term_l378_378734


namespace measure_angle_DPO_l378_378739

-- Definitions and conditions
variables (DOG : Type) [triangle DOG]
variables (O D G P : DOG)
variables (bisects_angle_DOG : bisects O P D G)
variables (angle_DOG angle_DGO : ℝ)
variables (angle_GOD : ℝ)
variables [h1 : angle_DGO = angle_DOG]
variables [h2 : angle_GOD = 24]
variables [h3 : bisects OP DOG]

-- The theorem to prove
theorem measure_angle_DPO : 
  ∀ (DOG : Type) [triangle DOG] (O D G P : DOG)
  (h1 : ∠DGO = ∠DOG) (h2 : ∠GOD = 24) (h3 : bisects OP ∠DOG),
  ∠DPO = 117 := 
begin
  sorry,
end

end measure_angle_DPO_l378_378739


namespace critics_voting_l378_378929

theorem critics_voting {n : ℕ} (n_pos : 1 ≤ n ∧ n ≤ 100)
    (critics : Type) [fintype critics] (hcritics : fintype.card critics = 3366)
    (actors : Type) [fintype actors] (act_votes : actors → ℕ)
    (act_votes_range : ∀ i, act_votes i ∈ finset.Icc 1 100)
    (act_votes_occurrence : ∀ n ∈ finset.Icc 1 100, ∃ i, act_votes i = n)
    (actresses : Type) [fintype actresses] (actress_votes : actresses → ℕ)
    (actress_votes_range : ∀ j, actress_votes j ∈ finset.Icc 1 100)
    (actress_votes_occurrence : ∀ n ∈ finset.Icc 1 100, ∃ j, actress_votes j = n)
    (votes : critics → actors × actresses) :
    ∃ (c₁ c₂ : critics), c₁ ≠ c₂ ∧ votes c₁ = votes c₂ :=
by
  sorry

end critics_voting_l378_378929


namespace length_of_train_is_approx_l378_378909

-- Define the speed of the train in km/hr
def speed_kmh : ℝ := 60

-- Define the time taken to cross the pole in seconds
def time_s : ℝ := 3

-- Define the conversion factor from km/hr to m/s
def kmh_to_ms : ℝ := 1000 / 3600

-- Define the speed of the train in m/s
def speed_ms : ℝ := speed_kmh * kmh_to_ms

-- Define the length of the train
def length_of_train : ℝ := speed_ms * time_s

-- Prove that the length of the train is approximately 50.01 meters
theorem length_of_train_is_approx : length_of_train ≈ 50.01 := by
  sorry

end length_of_train_is_approx_l378_378909


namespace isosceles_triangle_area_l378_378132

theorem isosceles_triangle_area 
  (A B C M N G : Type) 
  [IsMpoint M A C] [IsMpoint N B C]
  (h1 : isosceles ABC)
  (h2 : length AB = 10)
  (h3 : intersection G (line BM) (line AN))
  (h4 : is_right_angle (angle AGB)) :
  area ABC = 75 :=
by
  sorry

end isosceles_triangle_area_l378_378132


namespace rightmost_four_digits_of_5_pow_2023_l378_378092

theorem rightmost_four_digits_of_5_pow_2023 :
  (5 ^ 2023) % 10000 = 8125 :=
sorry

end rightmost_four_digits_of_5_pow_2023_l378_378092


namespace smallest_base10_integer_l378_378881

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end smallest_base10_integer_l378_378881


namespace trigonometric_functions_of_angle_l378_378629

theorem trigonometric_functions_of_angle (α : ℝ) (h1 : cos α = 5 / 13) (h2 : sin α = 12 / 13) :
  sin α = 12 / 13 ∧ 
  cos α = 5 / 13 ∧ 
  tan α = 12 / 5 ∧ 
  cot α = 5 / 12 ∧ 
  sec α = 13 / 5 ∧ 
  csc α = 13 / 12 :=
by
  sorry

end trigonometric_functions_of_angle_l378_378629


namespace rational_terms_and_largest_term_l378_378313

theorem rational_terms_and_largest_term (n : ℕ) (h : ∃ (r : ℕ), r = 6 ∧ 
  (choose n r) * (2^r) * (sqrt x * (1 / x^2)^r) = max (binomial_expansion n (sqrt x + 2 / x^2))) :
  (n = 10) ∧ (num_rational_terms n = 6) ∧ (largest_term_coefficient n = 15360 * x^(-25 / 2)) :=
by
  sorry

end rational_terms_and_largest_term_l378_378313


namespace field_trip_fraction_l378_378691

theorem field_trip_fraction (b g : ℕ) (hb : g = b)
  (girls_trip_fraction : ℚ := 4/5)
  (boys_trip_fraction : ℚ := 3/4) :
  girls_trip_fraction * g / (girls_trip_fraction * g + boys_trip_fraction * b) = 16 / 31 :=
by {
  sorry
}

end field_trip_fraction_l378_378691


namespace factorize_expression_l378_378610

theorem factorize_expression (a b : ℝ) : a^2 - a * b = a * (a - b) :=
by sorry

end factorize_expression_l378_378610


namespace find_alpha_l378_378665

theorem find_alpha (α : ℝ) (k : ℤ) 
  (h : ∃ (k : ℤ), α + 30 = k * 360 + 180) : 
  α = k * 360 + 150 :=
by 
  sorry

end find_alpha_l378_378665


namespace sum_x_minus_sum_xx_le_one_l378_378285

theorem sum_x_minus_sum_xx_le_one (n : ℕ) (h : n ≥ 2) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  (Finset.sum (Finset.univ) (λ i, x i)) - 
  (Finset.sum (Finset.filter (λ p, (p.1 < p.2)) (Finset.product Finset.univ Finset.univ)) (λ p, x p.1 * x p.2)) ≤ 1 :=
sorry

end sum_x_minus_sum_xx_le_one_l378_378285


namespace money_given_by_mom_l378_378184

theorem money_given_by_mom (cost_sweater cost_tshirt cost_shoes remaining_money total_money : ℕ) 
  (h1 : cost_sweater = 24) 
  (h2 : cost_tshirt = 6)
  (h3 : cost_shoes = 11)
  (h4 : remaining_money = 50)
  (h_total : total_money = cost_sweater + cost_tshirt + cost_shoes + remaining_money) : 
  total_money = 91 :=
by
  -- Assuming the provided conditions are correct
  have h_calculated := calc
    cost_sweater + cost_tshirt + cost_shoes + remaining_money
    = 24 + 6 + 11 + 50 : by rw [h1, h2, h3, h4]
    = 91 : by linarith
  exact h_total.symm.trans h_calculated

end money_given_by_mom_l378_378184


namespace problem_statement_l378_378461

variable (a b c : ℝ)

-- Conditions given in the problem
axiom h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -24
axiom h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8

-- The Lean statement for the proof problem
theorem problem_statement (a b c : ℝ) (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -24)
    (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8) :
    (b / (a + b) + c / (b + c) + a / (c + a)) = 19 / 2 :=
sorry

end problem_statement_l378_378461


namespace remainder_of_expression_mod_9_l378_378342

theorem remainder_of_expression_mod_9 (n : ℕ) (h : 0 < n) : 
  (7^n + ∑ i in finset.range n, 7^(n-1-i) * nat.choose n (i+1)) % 9 =
    if even n then 0 else 7 :=
by sorry

end remainder_of_expression_mod_9_l378_378342


namespace number_of_good_games_l378_378779

def total_games : ℕ := 11
def bad_games : ℕ := 5
def good_games : ℕ := total_games - bad_games

theorem number_of_good_games : good_games = 6 := by
  sorry

end number_of_good_games_l378_378779


namespace negation_of_ellipse_l378_378477

def is_ellipse (x y m : ℝ) : Prop :=
  ∃ m ∈ set.univ, (x^2 / m) + y^2 = 1

theorem negation_of_ellipse :
  (¬ ∃ m : ℕ, ∃ (x y : ℝ), is_ellipse x y (m : ℝ)) ↔ (∀ m : ℕ, ∀ (x y : ℝ), ¬ is_ellipse x y (m : ℝ)) :=
by
  sorry

end negation_of_ellipse_l378_378477


namespace sprint_championship_races_needed_l378_378940

-- Define the conditions
def total_sprinters : ℕ := 300
def lanes_per_race : ℕ := 8
def top_finishers_advance : ℕ := 2

-- Theorem statement
theorem sprint_championship_races_needed : 
  (let races_needed := calc_races_needed total_sprinters lanes_per_race top_finishers_advance in
   races_needed = 53) :=
by
  sorry

-- Assuming a helper function is defined to calculate the races needed
def calc_races_needed (total_sprinters : ℕ) (lanes_per_race : ℕ) (top_finishers_advance : ℕ) : ℕ := 
  sorry

end sprint_championship_races_needed_l378_378940


namespace evaluate_definite_integral_l378_378269

noncomputable def antiderivative (x : ℝ) : ℝ := Real.log x + 1 / 2 * x^2

theorem evaluate_definite_integral :
  ∫ x in 1..2, (1 / x + x) = Real.log 2 + 3 / 2 :=
by
  sorry

end evaluate_definite_integral_l378_378269


namespace min_entries_to_unique_sums_l378_378159

theorem min_entries_to_unique_sums :
  let M : Matrix (Fin 3) (Fin 3) ℕ := ![![5, 10, 3], ![9, 2, 8], ![4, 6, 9]],
      row_sums : Fin 3 → ℕ := fun i => (M i 0) + (M i 1) + (M i 2),
      col_sums : Fin 3 → ℕ := fun j => (M 0 j) + (M 1 j) + (M 2 j) in
  row_sums 0 = 18 ∧ row_sums 1 = 19 ∧ row_sums 2 = 19 ∧
  col_sums 0 = 18 ∧ col_sums 1 = 18 ∧ col_sums 2 = 20 →
  ∀ M' : Matrix (Fin 3) (Fin 3) ℕ,
    (∃ x y, 1 ≤ x ∧ x < 5 ∧  matrix_diff M M' = y ∧ y = x) →
    ∀ r_sums' c_sums' : Fin 3 → ℕ,
      r_sums' = fun i => (M' i 0) + (M' i 1) + (M' i 2) ∧
      c_sums' = fun j => (M' 0 j) + (M' 1 j) + (M' 2 j) →
      ¬ (r_sums' 0 = r_sums' 1 ∨ r_sums' 0 = r_sums' 2 ∨ r_sums' 1 = r_sums' 2 ∨
        c_sums' 0 = c_sums' 1 ∨ c_sums' 0 = c_sums' 2 ∨ c_sums' 1 = c_sums' 2) :=
by {
  sorry
}

def matrix_diff (M1 M2 : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  Finset.card (Finset.filter (λ (i, j), M1 i j ≠ M2 i j) (Finset.univ.product Finset.univ))

end min_entries_to_unique_sums_l378_378159


namespace geometric_sequence_common_ratio_l378_378483

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a(n + 1) = a(n) * q

theorem geometric_sequence_common_ratio (a b : ℕ → ℝ) (q : ℝ) :
  (∀ n, a(n + 1) = a(n) * q) ∧
  (|q| > 1) ∧
  (∀ n, b(n) = a(n) + 1) ∧
  (∃ i j k l ∈ {0,1,2,3}, {b i, b j, b k, b l} = {-53, -23, 19, 37, 82}) →
  q = - (3 / 2) := 
sorry

end geometric_sequence_common_ratio_l378_378483


namespace smallest_base10_integer_l378_378880

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end smallest_base10_integer_l378_378880


namespace find_other_number_l378_378805

variable (a b : ℕ)

theorem find_other_number (HCF LCM : ℕ) (num : ℕ) (HCF_eq : HCF = 12) (LCM_eq : LCM = 396) (num_eq : num = 24) :
  (HCF * LCM) / num = 198 :=
by 
  rw [HCF_eq, LCM_eq, num_eq]
  calc (12 * 396) / 24 = 4752 / 24 : by rfl
                      ... = 198     : by rfl
                      ... = 198     : by rfl


end find_other_number_l378_378805


namespace prove_a3_l378_378379

variable (a1 a2 a3 a4 : ℕ)
variable (q : ℕ)

-- Definition of the geometric sequence
def geom_seq (n : ℕ) : ℕ :=
  a1 * q^(n-1)

-- Given conditions
def cond1 := geom_seq 4 = 8
def cond2 := (geom_seq 2 + geom_seq 3) / (geom_seq 1 + geom_seq 2) = 2

-- Proving the required condition
theorem prove_a3 : cond1 ∧ cond2 → geom_seq 3 = 4 :=
by
sorry

end prove_a3_l378_378379


namespace problem_statement_l378_378407

/-- Let x, y, z be nonzero real numbers such that x + y + z = 0.
    Prove that ∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → x + y + z = 0 → (x^3 + y^3 + z^3) / (x * y * z) = 3. -/
theorem problem_statement (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h₁ : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := 
by 
  sorry

end problem_statement_l378_378407


namespace inverse_value_l378_378971

variables {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
variables (f : X → Y)

-- Define the symmetry condition about the point (1,2)
def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) :=
  ∀ x, f (x + p.1) + f (p.1 - x) = 2 * p.2

-- Given conditions
variable (h_symm : symmetric_about f (1, 2))
variable (h_inv : Function.LeftInverse f f⁻¹)
variable (h_f4 : f 4 = 0)

-- The theorem to prove
theorem inverse_value : f⁻¹ 4 = -2 :=
by
  sorry

end inverse_value_l378_378971


namespace remainder_is_70_l378_378944

theorem remainder_is_70 : ∃ r, 220070 = 1000 * 220 + r ∧ r = 70 := by
  -- Definitions from conditions
  let n := 220070
  let sum := 555 + 445
  let difference := 555 - 445
  let quotient := 2 * difference
  
  -- Some basic operations
  have h_sum : sum = 1000 := by rfl
  have h_difference : difference = 110 := by rfl
  have h_quotient : quotient = 220 := by rfl
  
  -- Now, we need to state and prove the final step
  existsi (n % sum)
  split
  calc
    n = sum * quotient + (n % sum) : by rfl
  show n % sum = 70
  sorry

end remainder_is_70_l378_378944


namespace license_plate_combinations_l378_378579

def num_choices_two_repeat_letters : ℕ :=
  (Nat.choose 26 2) * (Nat.choose 4 2) * (5 * 4)

theorem license_plate_combinations : num_choices_two_repeat_letters = 39000 := by
  sorry

end license_plate_combinations_l378_378579


namespace intersection_nonempty_l378_378462

open Nat

theorem intersection_nonempty (a : ℕ) (ha : a ≥ 2) :
  ∃ (b : ℕ), b = 1 ∨ b = a ∧
  ∃ y, (∃ x, y = a^x ∧ x ≥ 1) ∧
       (∃ x, y = (a + 1)^x + b ∧ x ≥ 1) :=
by sorry

end intersection_nonempty_l378_378462


namespace areas_equal_l378_378298

-- Definitions of geometric entities and properties
variables {A B C L K M N : Type*} [metric_space A] [metric_space B] [metric_space C] 
variable {triangle_ABC : ∀ x : A, x ∈ triangle A B C}

-- Conditions of the problem
def acute_triangle : Prop := ∀ cl : triangle_ABC, acute_triangle cl

def angle_bisector (a : A) (t : triangle_ABC) : Prop := 
  let angle_bisector_point := L in 
  ∃ (L : Type*) (N : Type*), angle A B L = angle A C L ∧ angle B A N = angle C A N

def perpendicular_lines (L : Type*) (AB : A) (AC : A) : Prop := 
  let K := L ∩ AB in
  let M := L ∩ AC in 
  is_perpendicular_to K AB ∧ is_perpendicular_to M AC

-- Statement of the problem to be proven
theorem areas_equal (triangle ABC : acute_triangle) 
  (bisector : angle_bisector A triangle_ABC)
  (perpendiculars : perpendicular_lines L AB AC) : 
  area (triangle ABC) = area (quadrilateral A K N M) :=
sorry

end areas_equal_l378_378298


namespace part1_part2_l378_378702

variable (a b : ℝ)

theorem part1 (h : |a - 3| + |b + 6| = 0) : a + b - 2 = -5 := sorry

theorem part2 (h : |a - 3| + |b + 6| = 0) : a - b - 2 = 7 := sorry

end part1_part2_l378_378702


namespace factorization_l378_378612

theorem factorization (a b : ℝ) : a^2 - a * b = a * (a - b) := by
  sorry

end factorization_l378_378612


namespace male_female_ratio_l378_378141

-- Definitions and constants
variable (M F : ℕ) -- Number of male and female members respectively
variable (h_avg_members : 66 * (M + F) = 58 * M + 70 * F) -- Average ticket sales condition

-- Statement of the theorem
theorem male_female_ratio (M F : ℕ) (h_avg_members : 66 * (M + F) = 58 * M + 70 * F) : M / F = 1 / 2 :=
sorry

end male_female_ratio_l378_378141


namespace polar_to_rectangular_rectangular_to_polar_l378_378206

theorem polar_to_rectangular (ρ θ : ℝ) (h1 : ρ * cos (θ + π / 4) + 8 = 0) : 
  ∃ x y : ℝ, x - y + 8 * Real.sqrt 2 = 0 ∧ x = ρ * cos θ ∧ y = ρ * sin θ := 
by 
  sorry

theorem rectangular_to_polar (x y : ℝ) (h2 : (x - 2)^2 + y^2 = 4) :
  ∃ ρ θ : ℝ, ρ = y / (sin θ) ∧ ρ = 4 * cos θ :=
by 
  sorry

end polar_to_rectangular_rectangular_to_polar_l378_378206


namespace range_of_f_is_interval_l378_378348

-- Define the function and properties
def f (x : ℝ) : ℝ := x^2 - 4*x - 2

-- State the main Theorem which encapsulates the proof problem
theorem range_of_f_is_interval {m : ℝ} :
  (∀ x ∈ set.Icc 0 m, f x ∈ set.Icc (-6) (-2)) ↔ m ∈ set.Icc 2 4 :=
by sorry

end range_of_f_is_interval_l378_378348


namespace total_revenue_l378_378910

theorem total_revenue (package_price : ℕ) (discount_factor : ℚ) (initial_packages : ℕ) (total_packages : ℕ) 
    (percentage_x : ℚ) (percentage_y : ℚ) : 
  package_price = 25 → 
  discount_factor = 4/5 → 
  initial_packages = 10 → 
  total_packages = 60 → 
  percentage_x = 0.15 → 
  percentage_y = 0.15 → 
  total_revenue = 1250 :=
by
  intro package_price discount_factor initial_packages total_packages percentage_x percentage_y
  sorry

end total_revenue_l378_378910


namespace perimeter_overlap_region_l378_378591

-- Define unit circle and conditions
noncomputable def unit_circle (x y : ℝ) : set (ℝ × ℝ) :=
{ p : ℝ × ℝ | (p.fst - x)^2 + (p.snd - y)^2 = 1 }

-- Define the specific intersecting circles
def circle1 := unit_circle 0 0
def circle2 := unit_circle 1 0

-- Intersection definition
def overlap_region := circle1 ∩ circle2

-- The main theorem statement
theorem perimeter_overlap_region : 
  let R := overlap_region in 
  (perimeter_of R = 2 * (π / 3)) :=
sorry

end perimeter_overlap_region_l378_378591


namespace smallest_integer_value_of_eight_nums_l378_378488

theorem smallest_integer_value_of_eight_nums (a : Fin 8 → ℝ)
    (h_sum : (Finset.univ.sum (λ i, a i) = (4 : ℝ) / 3))
    (h_sum_pos : ∀ i : Fin 8, (Finset.univ.erase i).sum (λ j, a j) > 0)
    : ∃ (k : ℤ), (k = -7) ∧ (∀ i : Fin 8, (a i) ≥ (↑k : ℝ)) :=
by
  sorry

end smallest_integer_value_of_eight_nums_l378_378488


namespace log_base_one_fourth_of_sixteen_l378_378256

theorem log_base_one_fourth_of_sixteen :
  log (1 / 4 : ℝ) (16 : ℝ) = -2 :=
sorry

end log_base_one_fourth_of_sixteen_l378_378256


namespace four_fold_application_of_f_l378_378416

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 then
    x / 3
  else
    5 * x + 2

theorem four_fold_application_of_f : f (f (f (f 3))) = 187 := 
  by
    sorry

end four_fold_application_of_f_l378_378416


namespace cos_seven_pi_over_four_proof_l378_378979

def cos_seven_pi_over_four : Prop := (Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2)

theorem cos_seven_pi_over_four_proof : cos_seven_pi_over_four :=
by
  sorry

end cos_seven_pi_over_four_proof_l378_378979


namespace carnival_total_cost_l378_378418

def morning_costs (under18_cost over18_cost : ℕ) : ℕ :=
  under18_cost + over18_cost

def afternoon_costs (under18_cost over18_cost : ℕ) : ℕ :=
  under18_cost + 1 + over18_cost + 1

noncomputable def mara_cost : ℕ :=
  let bumper_car_cost := morning_costs 2 0 + afternoon_costs 2 0
  let ferris_wheel_cost := morning_costs 5 5 + 5
  bumper_car_cost + ferris_wheel_cost

noncomputable def riley_cost : ℕ :=
  let space_shuttle_cost := morning_costs 0 5 + afternoon_costs 0 5
  let ferris_wheel_cost := morning_costs 0 6 + (6 + 1)
  space_shuttle_cost + ferris_wheel_cost

theorem carnival_total_cost :
  mara_cost + riley_cost = 61 := by
  sorry

end carnival_total_cost_l378_378418


namespace function_range_l378_378482

noncomputable def f : ℝ → ℝ := λ x, -x^2 + 3 * x + 1

theorem function_range :
  ∃ range_set : Set ℝ, range_set = Set.Icc (-3 : ℝ) (13 / 4) ∧ ∀ y, (∃ x ∈ Set.Ico (-1 : ℝ) 2, f x = y) ↔ y ∈ range_set :=
by
  sorry

end function_range_l378_378482


namespace smallest_base10_integer_l378_378893

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end smallest_base10_integer_l378_378893


namespace length_n_possible_values_l378_378413

-- Define the function f(n)
def f (n : ℕ) : ℕ :=
  Nat.find (λ k, k > 0 ∧ ¬ (n % k = 0))

-- Define the length of n
def l_n (n : ℕ) [h : Fact (n ≥ 3)] : ℕ :=
  Nat.find (λ k, (f^[k] n = 2))

-- The main theorem
theorem length_n_possible_values (n : ℕ) (h : n ≥ 3) :
  l_n n = 1 ∨ l_n n = 2 ∨ l_n n = 3 :=
sorry

end length_n_possible_values_l378_378413


namespace range_of_sum_of_products_l378_378640

theorem range_of_sum_of_products (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) (hxyz : x + y + z = 2) :
  1 < x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ 4 / 3 := 
begin
  sorry
end

end range_of_sum_of_products_l378_378640


namespace smallest_base_10_integer_l378_378888

noncomputable def smallest_integer (a b: ℕ) (h₁: a > 2) (h₂: b > 2) (h₃: n = 2 * a + 1) (h₄: n = b + 2) : ℕ :=
  n

theorem smallest_base_10_integer : smallest_integer 3 5 (by decide) (by decide) (by decide) (by decide) = 7 :=
sorry

end smallest_base_10_integer_l378_378888


namespace square_of_neg_three_l378_378073

theorem square_of_neg_three : (-3 : ℤ)^2 = 9 := by
  sorry

end square_of_neg_three_l378_378073


namespace triangle_midline_angle_bisector_l378_378385

theorem triangle_midline_angle_bisector (A B C A1 C1 K : Point) (a b c : ℝ)
  (hA1 : midpoint A C A1) (hC1 : midpoint C B C1)
  (hMidline : midline A1 C1)
  (hAngleBisector : angle_bisector A B C AD)
  (hIntersection : intersects AD A1 C1 K) :
  2 * distance A1 K = |b - c| :=
sorry

end triangle_midline_angle_bisector_l378_378385


namespace complex_fraction_identity_l378_378639

noncomputable def z1 : ℂ := 1 + complex.i
noncomputable def z2 : ℂ := 1 - complex.i

theorem complex_fraction_identity : (z1 / z2) + (z2 / z1) = 0 :=
by
  sorry

end complex_fraction_identity_l378_378639


namespace jill_shopping_trip_tax_l378_378778

theorem jill_shopping_trip_tax :
  let total_amount_spent_excluding_taxes := 100
  let percent_clothing := 0.50
  let percent_food := 0.10
  let percent_other_items := 0.40
  let tax_clothing := 0.04
  let tax_food := 0
  let tax_other_items := 0.08 in
  
  let total_tax_paid := (percent_clothing * total_amount_spent_excluding_taxes * tax_clothing) +
                        (percent_food * total_amount_spent_excluding_taxes * tax_food) +
                        (percent_other_items * total_amount_spent_excluding_taxes * tax_other_items) in
                         
  (total_tax_paid / total_amount_spent_excluding_taxes) * 100 = 5.20 := sorry

end jill_shopping_trip_tax_l378_378778


namespace fraction_female_basketball_l378_378361

variable (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
variable (male_basketball : ℕ) (female_basketball : ℕ)

axiom ratio_students : male_students = 3 * female_students / 2
axiom total_population : total_students = 1000
axiom division_population : male_students + female_students = total_students
axiom male_like_basketball : male_basketball = 2 * male_students / 3
axiom total_basketball : (male_basketball + female_basketball) = (48 * total_students / 100)

theorem fraction_female_basketball :
  (female_basketball = (total_basketball - male_basketball)) →
  (female_students = 2 * total_students / 5) →
  (female_basketball / female_students = 1 / 5) :=
by
  sorry

end fraction_female_basketball_l378_378361


namespace hexagon_diagonal_angles_l378_378719

theorem hexagon_diagonal_angles(
  A B C D E F G : Type*
) (h1 : is_circumscribed_hexagon A B C D E F)
  (h2 : intersect_at G A D B E C F) 
  (h3 : ∀ d1 d2 ∈ {A D, B E, C F}, angle_at_60_deg d1 d2)
  : 
  dist A G + dist C G + dist E G = dist B G + dist D G + dist F G :=
sorry

end hexagon_diagonal_angles_l378_378719


namespace angle_difference_ninety_l378_378437

-- Given conditions
variables (A B C C1 O : Type*)
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited C1] [Inhabited O]

-- Representing angles of the triangle ABC using the points
variables (α β γ : ℝ)
variables (angleA angleB angleC : α == β + γ)

-- Midpoint condition
axiom midpoint_C1 : C1 = (A + B) / 2

-- Circumcenter O and the right angle condition
axiom circumcircle_center : C ∈ O
axiom right_angle_COC1 : ∠COC1 = 90

-- Prove the final statement
theorem angle_difference_ninety :
  |angleB - angleA| = 90 :=
sorry

end angle_difference_ninety_l378_378437


namespace tangent_range_of_values_for_k_l378_378939

def circle (k : ℝ) : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), (p.1)^2 + (p.2)^2 + 2*k*p.1 + 4*p.2 + 3*k + 8 = 0

def line_through (p : ℝ × ℝ) (m : ℝ) : ℝ × ℝ → Prop := λ (l : ℝ × ℝ), l.2 = m * (l.1 + 1)

theorem tangent_range_of_values_for_k :
  {k : ℝ | ∃ m : ℝ, ∀ (x y : ℝ), line_through (-1, 0) m (x, y) → circle k (x, y)} = {k : ℝ | (-9 < k ∧ k < -1) ∨ (4 < k)} :=
sorry

end tangent_range_of_values_for_k_l378_378939


namespace calculate_seedlings_l378_378476

-- Define conditions
def condition_1 (x n : ℕ) : Prop :=
  x = 5 * n + 6

def condition_2 (x m : ℕ) : Prop :=
  x = 6 * m - 9

-- Define the main theorem based on these conditions
theorem calculate_seedlings (x : ℕ) : (∃ n, condition_1 x n) ∧ (∃ m, condition_2 x m) → x = 81 :=
by {
  sorry
}

end calculate_seedlings_l378_378476


namespace ratio_K_T₁_T₂_l378_378380

noncomputable def isosceles_trapezoid (A B C D : Point) : Prop := 
  is_trapezoid A B C D ∧ length (A B) = length (C D)

noncomputable def inscribed_circle_touches_points (A B C D : Point) (T₁ T₂ T₃ T₄ : Point) : Prop := 
  isosceles_trapezoid A B C D ∧
  circle_inside (A B C D) (circle) ∧
  touches (circle) (A B) T₁ ∧
  touches (circle) (B C) T₂ ∧
  touches (circle) (C D) T₃ ∧
  touches (circle) (D A) T₄

noncomputable def intersection_point (C T₃ T₁ T₂ : Point) : Point := 
  intersection (line_through C T₃) (line_through T₁ T₂)

theorem ratio_K_T₁_T₂ (A B C D T₁ T₂ T₃ T₄ : Point) (circle : Circle) (K : Point) :
  isosceles_trapezoid A B C D →
  inscribed_circle_touches_points A B C D T₁ T₂ T₃ T₄ →
  K = intersection_point C T₃ T₁ T₂ →
  divides_ratio K T₁ T₂ 3 1 :=
by
  intros h_trapezoid h_circle_touch_points h_K_intersection
  sorry

end ratio_K_T₁_T₂_l378_378380


namespace correct_option_D_l378_378502

open Classical

variable (countsD : List Nat) (count_bounds : Nat × Nat := (30, 300)) (dilution_factor : Nat := 10^6)

-- Define the conditions
def valid_count (count : Nat) : Prop :=
  count ∈ Icc (count_bounds.1) (count_bounds.2)

def spread_plate_condition (plates : List Nat) : Prop :=
  plates.length ≥ 3 ∧ ∀ val ∈ plates, valid_count val

-- Define the options
def option_A := [230]
def option_B := [251, 260]
def option_C := [21, 212, 256]
def option_D := [210, 240, 250]

-- Define the main statement to be proved
theorem correct_option_D :
  spread_plate_condition option_D :=
by
  have condition_A : spread_plate_condition option_A := by sorry
  have condition_B : spread_plate_condition option_B := by sorry
  have condition_C : spread_plate_condition option_C := by sorry
  have condition_D : spread_plate_condition option_D := by sorry
  exact condition_D
  sorry

end correct_option_D_l378_378502


namespace nearest_multiple_of_21_l378_378509

theorem nearest_multiple_of_21 (n : ℕ) (h : n = 2304) : 
  ∃ m, m % 21 = 0 ∧ abs (m - n) = min (abs (21 * (n / 21) - n)) (abs (21 * ((n / 21) + 1) - n)) ∧ m = 2310 :=
by
  sorry

end nearest_multiple_of_21_l378_378509


namespace trajectory_of_midpoint_l378_378643

theorem trajectory_of_midpoint (A B P : ℝ × ℝ)
  (hA : A = (2, 4))
  (hB : ∃ m n : ℝ, B = (m, n) ∧ n^2 = 2 * m)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.2 - 2)^2 = P.1 - 1 :=
sorry

end trajectory_of_midpoint_l378_378643


namespace probability_after_first_new_draw_is_five_ninths_l378_378715

-- Defining the conditions in Lean
def total_balls : ℕ := 10
def new_balls : ℕ := 6
def old_balls : ℕ := 4

def balls_remaining_after_first_draw : ℕ := total_balls - 1
def new_balls_after_first_draw : ℕ := new_balls - 1

-- Using the classic probability definition
def probability_of_drawing_second_new_ball := (new_balls_after_first_draw : ℚ) / (balls_remaining_after_first_draw : ℚ)

-- Stating the theorem to be proved
theorem probability_after_first_new_draw_is_five_ninths :
  probability_of_drawing_second_new_ball = 5/9 := sorry

end probability_after_first_new_draw_is_five_ninths_l378_378715


namespace circle_center_radius_l378_378812

theorem circle_center_radius :
  ∀ (x y : ℝ), (x + 1) ^ 2 + (y - 2) ^ 2 = 9 ↔ (x = -1 ∧ y = 2 ∧ ∃ r : ℝ, r = 3) :=
by
  sorry

end circle_center_radius_l378_378812


namespace Lavinia_son_older_than_daughter_l378_378750

theorem Lavinia_son_older_than_daughter :
  ∀ (K : ℕ),  -- Katie's daughter's age
  K = 12 →  -- Katie's daughter's age is 12
  let L_d := K / 3 in  -- Lavinia's daughter's age
  let L_s := 2 * K in  -- Lavinia's son's age
  let Children_sum := L_d + L_s in  -- Sum of children's ages
  Children_sum = 2 * K + 5 →  -- Given sum condition
  L_s - L_d = 20 :=  -- Prove this
by 
  intros K hK
  let L_d := K / 3
  let L_s := 2 * K
  let Children_sum := L_d + L_s
  intro hsum
  rw hK at *
  sorry

end Lavinia_son_older_than_daughter_l378_378750


namespace expr_max_value_l378_378634

theorem expr_max_value (x : ℝ) (h1 : -4 < x) (h2 : x < 1) : 
  let expr := (x^2 - 2 * x + 2) / (2 * x - 2) in
  expr ≤ -1 :=
sorry

end expr_max_value_l378_378634


namespace molly_current_age_l378_378002

-- Define Molly's initial and additional candles as constants
constant initial_candles : ℕ := 14
constant additional_candles : ℕ := 6

-- Problem statement: Proving Molly's current age due to these candles
theorem molly_current_age : initial_candles + additional_candles = 20 := by
  sorry

end molly_current_age_l378_378002


namespace smallest_base_10_integer_l378_378871

theorem smallest_base_10_integer (a b : ℕ) (ha : a > 2) (hb : b > 2) 
  (h1: 21_a = 2 * a + 1) (h2: 12_b = b + 2) : 2 * a + 1 = 7 :=
by 
  sorry

end smallest_base_10_integer_l378_378871


namespace clerical_percentage_after_reduction_l378_378011

theorem clerical_percentage_after_reduction (total_employees : ℕ)
  (clerical_fraction managers_supervisors_fraction : ℚ)
  (clerical_reduction_fraction : ℚ)
  (h1 : total_employees = 5000)
  (h2 : clerical_fraction = 3/7)
  (h3 : managers_supervisors_fraction = 1/3)
  (h4 : clerical_reduction_fraction = 3/8) :
  let clerical_initial := (clerical_fraction * total_employees).to_int
      clerical_reduction := (clerical_initial * clerical_reduction_fraction).to_int
      clerical_remaining := clerical_initial - clerical_reduction
      total_remaining := total_employees - clerical_reduction in
  (clerical_remaining.to_rat / total_remaining.to_rat) * 100 ≈ 31.94 := sorry

end clerical_percentage_after_reduction_l378_378011


namespace cylinder_surface_square_l378_378899

theorem cylinder_surface_square (C h : ℝ) (hC_pos : 0 < C) (hh_pos : 0 < h) :
  (lateral_surface_unfolds_into_rectangle C h) → 
  (lateral_surface_unfolds_into_square C h ↔ C = h) :=
sorry

def lateral_surface_unfolds_into_rectangle (C h : ℝ) : Prop :=
-- The lateral surface is a rectangle with length C and width h.
sorry

def lateral_surface_unfolds_into_square (C h : ℝ) : Prop :=
-- The lateral surface is a square (which means it has equal length and width).
C = h

end cylinder_surface_square_l378_378899


namespace sum_of_fractions_l378_378609

theorem sum_of_fractions : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) = 3 / 8) :=
by sorry

end sum_of_fractions_l378_378609


namespace coefficient_x3_expansion_l378_378466

theorem coefficient_x3_expansion : 
  let expansion := (1 - 2 * x)^6
  in (coeff 3 (expand expansion)) = -160 :=
by
  sorry

end coefficient_x3_expansion_l378_378466


namespace log_base_fraction_l378_378225

theorem log_base_fraction : ∀ (a b : ℝ) (x : ℝ), 16 = (4:ℝ)^2 ∧ (1 / 4:ℝ) = 4^(-1) → log (1 / 4) 16 = -2 :=
begin
  intros a b x h,
  -- Skipping the proof by adding sorry
  sorry,
end

end log_base_fraction_l378_378225


namespace find_BD_l378_378642

open Real EuclideanGeometry

variables (A B C D : Point)
variables (r : ℝ)
variables (h : Circle)
variables (diam : diameter h A B)
variables (on_circle_C : on h C)
variables (on_circle_A : on h A)
variables (on_circle_B : on h B)
variables (dist_A_B : dist A B = 13)
variables (perp_CD_AB : is_perpendicular CD AB)
variables (dist_CD : dist C D = 6)

theorem find_BD : dist B D = 4 ∨ dist B D = 9 := sorry

end find_BD_l378_378642


namespace log_base_one_four_of_sixteen_l378_378238

theorem log_base_one_four_of_sixteen : log (1 / 4) 16 = -2 := by
  sorry

end log_base_one_four_of_sixteen_l378_378238


namespace find_n_and_m_l378_378696

noncomputable def permutation := 
  ∃ n m : ℕ, A_n^m = 17 * 16 * 15 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 ∧
  A_n^m = n * (n - 1) * (n - 2) * ... * (n - m + 1)

theorem find_n_and_m : ∃ n m : ℕ, n = 17 ∧ m = 14 ∧ permutation :=
sorry

end find_n_and_m_l378_378696


namespace overlapping_triangles_are_congruent_l378_378108

-- Define the concept of triangle congruence
def triangles_congruent (Δ₁ Δ₂ : Triangle) : Prop :=
  Δ₁.side₁ = Δ₂.side₁ ∧ Δ₁.side2 = Δ₂.side2 ∧ Δ₁.side3 = Δ₂.side3 ∧ 
  Δ₁.angle₁ = Δ₂.angle₁ ∧ Δ₁.angle2 = Δ₂.angle2 ∧ Δ₁.angle3 = Δ₂.angle3

-- Given the condition that two triangles are completely overlapping
theorem overlapping_triangles_are_congruent 
  (Δ₁ Δ₂ : Triangle)
  (h : Δ₁ = Δ₂)
  : triangles_congruent Δ₁ Δ₂ :=
sorry

end overlapping_triangles_are_congruent_l378_378108


namespace perfect_square_polynomial_l378_378616

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2) ↔ (x = -1 ∨ x = 0 ∨ x = 3) :=
sorry

end perfect_square_polynomial_l378_378616


namespace agms_seating_arrangements_l378_378034

noncomputable def count_possible_seating_arrangements : ℕ := 28

theorem agms_seating_arrangements : 
  let M := 4
  let V := 4
  let E := 4
  let positions := list.range 12
  (positions.head = M) ∧ 
  (positions.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.head = E) ∧ 
  (∀ i, i < 11 → positions.nth i = M → positions.nth (i+1) ≠ E) ∧
  (∀ i, i < 11 → positions.nth i = M → positions.nth (i+1) ≠ V) ∧
  (∀ i, i < 11 → positions.nth i = V → positions.nth (i+1) ≠ E) →
  count_possible_seating_arrangements  = 28 :=
sorry

end agms_seating_arrangements_l378_378034


namespace stacking_100_unit_cubes_surface_areas_l378_378460

theorem stacking_100_unit_cubes_surface_areas :
  ∃ (surface_areas : List ℕ),
    surface_areas = [130, 160, 208, 240, 250, 258]
    ∧ surface_areas.nth 0 = some 130
    ∧ surface_areas.nth 1 = some 160
    ∧ surface_areas.nth 2 = some 208
    ∧ surface_areas.nth 3 = some 240
    ∧ surface_areas.nth 4 = some 250
    ∧ surface_areas.nth 5 = some 258 := 
sorry

end stacking_100_unit_cubes_surface_areas_l378_378460


namespace find_minimum_magnitude_l378_378765

open Complex

noncomputable def minimum_magnitude_z (z : ℂ) : ℝ :=
  sqrt (85) / 3

theorem find_minimum_magnitude (z : ℂ)
  (h : complex.abs (z - 5 * I) + complex.abs (z - 6) + complex.abs (z - 2 * I) = 12) :
  ∃ w : ℂ, complex.abs z = minimum_magnitude_z w ∧
            (complex.abs (w - 5 * I) + complex.abs (w - 6) + complex.abs (w - 2 * I) = 12) := sorry

end find_minimum_magnitude_l378_378765


namespace animals_in_field_l378_378932

def dog := 1
def cats := 4
def rabbits_per_cat := 2
def hares_per_rabbit := 3

def rabbits := cats * rabbits_per_cat
def hares := rabbits * hares_per_rabbit

def total_animals := dog + cats + rabbits + hares

theorem animals_in_field : total_animals = 37 := by
  sorry

end animals_in_field_l378_378932


namespace place_value_ratio_l378_378735

theorem place_value_ratio :
  let d8_place := 0.1
  let d7_place := 10
  d8_place / d7_place = 0.01 :=
by
  -- proof skipped
  sorry

end place_value_ratio_l378_378735


namespace number_of_students_l378_378777

theorem number_of_students (T : ℕ)
  (hA : T / 5)
  (hB : T / 4)
  (hC : T / 2)
  (hD : T - (T / 5 + T / 4 + T / 2) = 30) : 
  T = 600 :=
by
  sorry

end number_of_students_l378_378777


namespace mnPQ_is_parallelogram_locus_of_w_is_uv_l378_378754

variables (A B C D A' B' C' D' M N P Q : Type*)
variables (A B C D A' B' C' D' M N P Q u : ℝ)
variables (vector_space : ℝ → ℝ → ℝ)
variables (is_parallelogram : Type* → Type* → Type* → Type* → Prop)
variables (ratio : ℝ)

-- Assume ABCD and A'B'C'D' are parallelograms
variable h_ABCD : is_parallelogram A B C D
variable h_A'B'C'D' : is_parallelogram A' B' C' D'

-- Points M, N, P, Q divide the segments AA', BB', CC', DD' in equal ratios
variable h_ratios : u = ratio

-- Part (a) Proving MNPQ is a parallelogram
theorem mnPQ_is_parallelogram 
  (h_M : M = vector_space (1 - u) A + vector_space u A')
  (h_N : N = vector_space (1 - u) B + vector_space u B')
  (h_P : P = vector_space (1 - u) C + vector_space u C')
  (h_Q : Q = vector_space (1 - u) D + vector_space u D') :
  is_parallelogram M N P Q := 
sorry

-- Part (b) Locus of the center W of parallelogram MNPQ as M varies on AA'
theorem locus_of_w_is_uv 
  (U V W : Type*)
  (h_U : U = vector_space 0.5 (A + C))
  (h_V : V = vector_space 0.5 (A' + C'))
  (h_W : W = vector_space 0.5 (u * (A + C) + (1 - u) * (A' + C'))) :
  (h_locus : ∀ M, W ∈ [U, V]) :=
sorry

end mnPQ_is_parallelogram_locus_of_w_is_uv_l378_378754


namespace radius_increase_l378_378512

theorem radius_increase (C₁ C₂ : ℝ) (C₁_eq : C₁ = 30) (C₂_eq : C₂ = 40) :
  let r₁ := C₁ / (2 * Real.pi)
  let r₂ := C₂ / (2 * Real.pi)
  r₂ - r₁ = 5 / Real.pi :=
by
  simp [C₁_eq, C₂_eq]
  sorry

end radius_increase_l378_378512


namespace parrot_seeds_consumed_l378_378433

theorem parrot_seeds_consumed (H1 : ∃ T : ℝ, 0.40 * T = 8) : 
  (∃ T : ℝ, 0.40 * T = 8 ∧ 2 * T = 40) :=
sorry

end parrot_seeds_consumed_l378_378433


namespace number_of_zeros_l378_378828

noncomputable def f (x : ℝ) : ℝ := log x - x + 1

theorem number_of_zeros {f : ℝ → ℝ} (h : ∀ x, f x = log x - x + 1) : 
  ∃! x, f x = 0 :=
sorry

end number_of_zeros_l378_378828


namespace max_smaller_rectangles_l378_378592

theorem max_smaller_rectangles (a : ℕ) (d : ℕ) (n : ℕ) 
    (ha : a = 100) (hd : d = 2) (hn : n = 50) : 
    n + 1 * (n + 1) = 2601 :=
by
  rw [hn]
  norm_num
  sorry

end max_smaller_rectangles_l378_378592


namespace problem_statement_l378_378963

variables {α : Type*} [metric_space α] {A B C D : α}

def convex_quadrilateral (A B C D : α) : Prop :=
  ∃ (P : α), P ∈ convex_hull ℝ ({A, B, C, D} : set α)

def circumradius (A B C : α) : ℝ :=
  classical.some (circumradius_existence.1 ⟨A, B, C⟩)

noncomputable 
def R_A : ℝ := circumradius B C D

noncomputable 
def R_B : ℝ := circumradius A C D

noncomputable 
def R_C : ℝ := circumradius A B D

noncomputable 
def R_D : ℝ := circumradius A B C

def angle_sum_gt (A B C D : α) : Prop :=
  measure_theory.angle_sum A + measure_theory.angle_sum C > measure_theory.angle_sum B + measure_theory.angle_sum D

theorem problem_statement
  (h : convex_quadrilateral A B C D) :
  R_A + R_C > R_B + R_D ↔ angle_sum_gt A B C D :=
sorry

end problem_statement_l378_378963


namespace chess_tournament_participants_l378_378717

theorem chess_tournament_participants (n : ℕ) :
  (∀ i j : fin n, i ≠ j → i ≠ j) →  -- Each player played against every other player exactly once
  (∃ w : fin n, 
    (w.val ≤ (n - 1) / 2) ∧                  -- The winner won half of the games
    ((n - 1) - w.val) ≤ (n - 1) / 2) ∧      -- The winner drew the other half
  (let total_points := (w.val * 1 + (n - w.val) * 0.5) in
    9 * total_points = ((n * (n - 1) / 2) - total_points))  -- Winner's points were 9 times less than the others
  → n = 15 :=
sorry

end chess_tournament_participants_l378_378717


namespace problem_statement_l378_378764

open Real

noncomputable def f (x : ℝ) : ℝ := 2023 * x^2 + 2024 * x * sin x

theorem problem_statement (x1 x2 : ℝ) (h1 : x1 ∈ Ioo (-π) π) (h2 : x2 ∈ Ioo (-π) π) (h3 : f x1 > f x2) : (x1^2) > (x2^2) :=
sorry

end problem_statement_l378_378764


namespace chocolate_doughnut_cost_l378_378581

theorem chocolate_doughnut_cost
  (total_students : ℕ) 
  (students_chocolate : ℕ) 
  (students_glazed : ℕ)
  (cost_glazed : ℤ)
  (total_cost : ℤ)
  (total_students_eq : total_students = 25)
  (students_chocolate_eq : students_chocolate = 10)
  (students_glazed_eq : students_glazed = 15)
  (cost_glazed_eq : cost_glazed = 1)
  (total_cost_eq : total_cost = 35) : 
  let x := 2 in
  10 * x + 15 * cost_glazed = total_cost :=
by
  sorry

end chocolate_doughnut_cost_l378_378581


namespace rahim_books_l378_378444

/-- 
Rahim bought some books for Rs. 6500 from one shop and 35 books for Rs. 2000 from another. 
The average price he paid per book is Rs. 85. 
Prove that Rahim bought 65 books from the first shop. 
-/
theorem rahim_books (x : ℕ) 
  (h1 : 6500 + 2000 = 8500) 
  (h2 : 85 * (x + 35) = 8500) : 
  x = 65 := 
sorry

end rahim_books_l378_378444


namespace phi_value_proof_l378_378709

noncomputable def shifted_symmetric_phi (f : ℝ → ℝ) (φ : ℝ) : Prop :=
  ∀ x, f(x - φ) = f(-x - φ)

noncomputable def func (x : ℝ) : ℝ :=
  2 * (Real.sin x * Real.cos x) - 2 * (Real.sin x)^2 + 1

def smallest_positive_phi : ℝ := π * 3 / 8

theorem phi_value_proof :
  ∃ φ, (φ > 0) ∧ shifted_symmetric_phi func φ ∧ φ = smallest_positive_phi :=
by
  sorry

end phi_value_proof_l378_378709


namespace average_multiples_of_10_l378_378507

theorem average_multiples_of_10 (a b : ℕ) (h₁ : a = 10) (h₂ : b = 400) :
  (∑ k in finset.range ((b - a) / 10 + 1), (a + k * 10)) / ((b - a) / 10 + 1) = 205 :=
by
  -- Using ∑ to denote the sum of all multiples of 10 in the given range
  sorry

end average_multiples_of_10_l378_378507


namespace inductive_reasoning_correct_l378_378107

theorem inductive_reasoning_correct :
  (is_analogical_reasoning (inferring_properties ball circle)) ∧
  (is_inductive_reasoning (sum_internal_angles triangles 180)) ∧
  (is_deductive_reasoning (odd_function_sin f_x_eq_sin x_neg_eq_neg_f_x x_in_R)) ∧
  (is_inductive_reasoning (sum_internal_angles_convex_polygon (n_minus_2_times_180 triangle quadrilateral pentagon))) →
  ([2, 4].forall (λ x, is_inductive_reasoning (nth_method x)) = tt) ∧  
  ([1,3].forall (λ x, is_analogical_reasoning (nth_method x) = tt ∨ is_deductive_reasoning (nth_method x) = tt)) →
  C = ① ② ④ :=
by
  intros h1 h2 h3
  sorry

end inductive_reasoning_correct_l378_378107


namespace scientists_comm_same_language_l378_378524

theorem scientists_comm_same_language :
  ∀ (S : Finset ℕ), S.card = 17 →
  (∀ i ∈ S, ∀ j ∈ S, i ≠ j → ∃ L : string, L ∈ {"English", "French", "Russian"} ∧ communicates_in_language i j L) →
  ∃ (A B C : ℕ), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ ∃ L : string, L ∈ {"English", "French", "Russian"} ∧ communicates_in_language A B L ∧ communicates_in_language A C L ∧ communicates_in_language B C L :=
by
  sorry

end scientists_comm_same_language_l378_378524


namespace train_cross_pole_time_l378_378960

-- Definitions based on given conditions
def train_speed_km_per_hr : ℝ := 50
def train_length_m : ℝ := 250
def speed_conversion_factor : ℝ := 1000 / 3600  -- Conversion factor from km/hr to m/s
def train_speed_m_per_s : ℝ := train_speed_km_per_hr * speed_conversion_factor
def expected_time_to_cross_pole : ℝ := 18  -- in seconds

-- Statement we need to prove
theorem train_cross_pole_time :
  (train_length_m / train_speed_m_per_s) ≈ expected_time_to_cross_pole :=
by
  -- Here would be the actual proof
  sorry

end train_cross_pole_time_l378_378960


namespace curvilinear_triangle_area_l378_378815

-- Defining the mathematical entities and the problem statement in Lean 4
variables (x y d α : ℝ)

axiom ext_tangent_circles : d = x + y
axiom angle_tangent : α > 0 ∧ α < 2 * Real.pi

theorem curvilinear_triangle_area
  (hx₀ : 0 < x)
  (hy₀ : 0 < y)
  (d_eq : d = x + y)
  (α_bound : 0 < α ∧ α < 2 * Real.pi) :
  let area := (d^2 / 8) * (4 * Real.cos (α / 2) - Real.pi * (1 + (Real.sin (α / 2))^2) + 2 * α * Real.sin (α / 2)) in
  True := sorry

end curvilinear_triangle_area_l378_378815


namespace solution_l378_378272

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

-- Formalizing the problem in Lean
theorem solution (n : Nat) (beta : ℚ) (h1 : 2 ≤ n) (h2 : beta > 0) (h3 : beta < 1) :
  (∃ (a : Fin n → Nat), (∀ (I : Finset (Fin n)), 2 ≤ I.card → sum_of_digits (∑ i in I, a i) = beta * (∑ i in I, sum_of_digits (a i)))) →
  n ≤ 10 :=
sorry

end solution_l378_378272


namespace matrix_mult_correct_l378_378585

def matrix_mult_example : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2], ![2, 1]]

def matrix_mult_example2 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[-1, 2], ![3, -4]]

theorem matrix_mult_correct :
  matrix_mult_example ⬝ matrix_mult_example2 = ![![5, -6], ![1, 0]] :=
by
  sorry

end matrix_mult_correct_l378_378585


namespace area_of_IJKL_equals_answer_l378_378028

noncomputable def area_of_inner_square
  (WXYZ IJKL : Type) 
  [square WXYZ 10] 
  [square_side_extends_vertex IJKL WXYZ] 
  (W I : point) 
  (W_distance_I : distance W I = 3) : ℝ :=
  (sqrt 91 - 3) ^ 2

theorem area_of_IJKL_equals_answer
  (WXYZ IJKL : Type) 
  [square WXYZ 10] 
  [square_side_extends_vertex IJKL WXYZ]
  (W I : point) 
  (W_distance_I : distance W I = 3) : 
  area_of_inner_square WXYZ IJKL W I W_distance_I = (sqrt 91 - 3) ^ 2 :=
sorry

end area_of_IJKL_equals_answer_l378_378028


namespace quadrilateral_is_cyclic_l378_378961

theorem quadrilateral_is_cyclic 
    (A B' C' C B : Type) 
    (triangle_acc : triangle A C C) 
    (cut_line : line_segment B' C')
    (similar_triangles : similar (triangle A C C) (triangle A B' C')) : 
    cyclic_quadrilateral B B' C' C :=
sorry

end quadrilateral_is_cyclic_l378_378961


namespace perimeter_of_ABFCDE_is_correct_l378_378198

-- Definition of the problem's conditions
def side_length_square : ℕ := 20
def leg_length_triangle : ℕ := 12

-- Hypotenuse calculation using Pythagorean theorem
noncomputable def hypotenuse_triangle : ℝ := Real.sqrt ((leg_length_triangle:ℝ)^2 + (leg_length_triangle:ℝ)^2)

-- Axiom stating that the shape is formed by square and translated triangle.
def perimeter_ABFCDE : ℝ := (side_length_square - leg_length_triangle) + 2 * side_length_square + 2 * leg_length_triangle + hypotenuse_triangle

-- The theorem to prove
theorem perimeter_of_ABFCDE_is_correct :
  hypotenuse_triangle = 12 * Real.sqrt 2 → perimeter_ABFCDE = 72 + 12 * Real.sqrt 2 :=
by
  sorry

end perimeter_of_ABFCDE_is_correct_l378_378198


namespace midpoint_AB_on_Γ_l378_378526

-- To avoid computational issues for geometric constructs.
noncomputable section

open Classical

-- Definitions for the problem
structure Circle (α : Type) := 
(center : α) (radius : ℝ)

variables {α : Type} 
variables [MetricSpace α]

def midpoint (a b : α) : α := sorry  -- Define midpoint (usual geometric midpoint)

-- Problem conditions
variables (Γ Σ : Circle α)
variables (O P A M B : α)

-- Given conditions
def P_on_Γ : Prop := (dist P Γ.center) = Γ.radius
def A_on_Σ : Prop := (dist A Σ.center) = Σ.radius
def M_is_midpoint_AP : Prop := M = midpoint A P
def B_on_Σ : Prop := (dist B Σ.center) = Σ.radius
def AB_parallel_OM : Prop := sorry -- Define parallel between vectors AB and OM

-- The proof problem statement
theorem midpoint_AB_on_Γ 
  (hP_on_Γ : P_on_Γ) 
  (hA_on_Σ : A_on_Σ) 
  (hM_is_midpoint_AP : M_is_midpoint_AP) 
  (hB_on_Σ : B_on_Σ) 
  (hAB_parallel_OM : AB_parallel_OM) : 
  ∃ S : α, S = midpoint A B ∧ (dist S Γ.center) = Γ.radius := 
begin
  sorry -- Skip proof
end

end midpoint_AB_on_Γ_l378_378526


namespace speed_ratio_l378_378559

theorem speed_ratio (v_A v_B : ℝ) (t : ℝ) (h1 : v_A = 200 / t) (h2 : v_B = 120 / t) : 
  v_A / v_B = 5 / 3 :=
by
  sorry

end speed_ratio_l378_378559


namespace MarysScore_l378_378030

theorem MarysScore :
  ∀ (s c w : ℕ), 
    (s = 35 + 5 * c - 2 * w) ∧ 
    (s > 90) ∧ 
    (∀ s', s' < s → s' > 90 → ∃ (c' w' : ℕ), s' = 35 + 5 * c' - 2 * w' ∧ c ≠ c') →
    s = 91 := 
by
  intros s c w h,
  sorry

end MarysScore_l378_378030


namespace parallel_lines_slope_l378_378351

theorem parallel_lines_slope (m : ℝ) :
  ((m + 2) * (2 * m - 1) = 3 * 1) →
  m = - (5 / 2) :=
by
  sorry

end parallel_lines_slope_l378_378351


namespace chessboard_cover_l378_378631

open Nat

/-- 
  For an m × n chessboard, after removing any one small square, it can always be completely covered
  with L-shaped tiles if and only if 3 divides (mn - 1) and min(m,n) is not equal to 1, 2, 5 or m=n=2.
-/
theorem chessboard_cover (m n : ℕ) :
  (∃ k : ℕ, 3 * k = m * n - 1) ∧ (min m n ≠ 1 ∧ min m n ≠ 2 ∧ min m n ≠ 5 ∨ m = 2 ∧ n = 2) :=
sorry

end chessboard_cover_l378_378631


namespace percentage_of_consumer_credit_by_auto_credit_l378_378974

noncomputable def total_automobile_credit : ℝ := 150
noncomputable def total_consumer_credit : ℝ := 416.6666666666667

theorem percentage_of_consumer_credit_by_auto_credit :
  ((total_automobile_credit / total_consumer_credit) * 100) ≈ 36 := by
  sorry

end percentage_of_consumer_credit_by_auto_credit_l378_378974


namespace range_of_independent_variable_l378_378833

theorem range_of_independent_variable (x : ℝ) :
  (x + 2 >= 0) → (x - 1 ≠ 0) → (x ≥ -2 ∧ x ≠ 1) :=
by
  intros h₁ h₂
  sorry

end range_of_independent_variable_l378_378833


namespace slower_train_speed_l378_378085

noncomputable def speed_of_slower_train : ℝ :=
  let V_faster := 72 -- speed of faster train in km/h
  let time := 10 -- time in seconds
  let length := 100 -- length of the faster train in meters
  (V_faster - (length / time * 18 / 5))

theorem slower_train_speed (V_faster : ℝ) (time : ℝ) (length : ℝ) (V_slower : ℝ) :
  V_faster = 72 → time = 10 → length = 100 → V_slower = 36 :=
by
  intros h1 h2 h3
  have h4 : V_faster - (length / time * 18 / 5) = 36, by sorry
  rw [h1, h2, h3] at h4
  exact h4

end slower_train_speed_l378_378085


namespace exponentiation_multiplication_identity_l378_378983

theorem exponentiation_multiplication_identity :
  (-4)^(2010) * (-0.25)^(2011) = -0.25 :=
by
  sorry

end exponentiation_multiplication_identity_l378_378983


namespace machine_X_takes_2_days_longer_l378_378450

-- Definitions for the rates of the machines and their production
variables {W : ℕ}

def rate_X := W / 6
def rate_Y := W / 4
def combined_rate := 5 * W / 12
def time_X_to_produce_W := W / rate_X
def time_Y_to_produce_W := W / rate_Y

theorem machine_X_takes_2_days_longer :
  time_X_to_produce_W - time_Y_to_produce_W = 2 :=
by
  sorry

end machine_X_takes_2_days_longer_l378_378450


namespace soda_cost_l378_378751

theorem soda_cost (x : ℝ) : 
  (let Lee_money := 10 in 
   let Friend_money := 8 in 
   let Chicken_wings := 6 in 
   let Chicken_salad := 4 in 
   let Tax := 3 in 
   let Initial_money := Lee_money + Friend_money in 
   let Total_change := 3 in 
   let Total_food_cost := Chicken_wings + Chicken_salad in 
   let Total_spent := Initial_money - Total_change in 
   let Total_cost := Total_food_cost + 2 * x + Tax in 
   Total_cost = Total_spent → x = 1) := by
   intros h1
   sorry

end soda_cost_l378_378751


namespace elliptic_formation_l378_378844

noncomputable def geometric_sequence_property
  (n : ℕ) (h : ℝ) (complex_seq : Fin n → ℂ)
  (common_ratio : ℂ) (condition_z1 : ∀ i : Fin n, (complex_seq i).norm ≠ 1) 
  (condition_q : common_ratio ≠ 1 ∧ common_ratio ≠ -1)  
  (w_k : Fin n → ℂ) : Prop :=
  ∀ k : Fin n, w_k k = complex_seq k + (1 / complex_seq k) + h

theorem elliptic_formation
  {n : ℕ} (h : ℝ) (complex_seq : Fin n → ℂ) 
  (common_ratio : ℂ) 
  (condition_z1 : ∀ i : Fin n, (complex_seq i).norm ≠ 1)
  (condition_q : common_ratio ≠ 1 ∧ common_ratio ≠ -1) 
  (w_k : Fin n → ℂ) 
  (w_k_def : geometric_sequence_property n h complex_seq common_ratio condition_z1 condition_q w_k) :
  (∃ (a b : ℝ), a = 2 ∧ b = √4 ∧ 
      (∀ k : Fin n, ((complex.re (w_k k)) - h)^2 / a^2 + (complex.im (w_k k))^2 / b^2 = 1)) := 
sorry

end elliptic_formation_l378_378844


namespace total_amount_payment_l378_378498

-- Definitions based on conditions from the problem
def payment_first_member : ℕ := 1
def payment (a : ℕ → ℕ) (n : ℕ) := 
  if n = 1 then
    payment_first_member
  else
    (∑ i in Finset.range (n - 1), a (i + 1)) + (n - 1)

-- Define the total payment function
def total_payment (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, a (i + 1)

-- Define the recurrence relation based on the problem conditions
def recurrence_relation (s : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 1 then
    s 1 = payment_first_member
  else
    s n = 2 * s (n - 1) + n - 1

-- Prove that the total amount follows the pattern a_n = 3 * 2^(n-2) - 1
theorem total_amount_payment (n : ℕ) : (payment (λ k, 3 * 2 ^ (k - 2) - 1) n = 3 * 2 ^ (n - 2) - 1) := 
  sorry

end total_amount_payment_l378_378498


namespace solve_for_x_l378_378992

theorem solve_for_x (x z : ℝ) (h : z = 3 * x) :
  (4 * z^2 + z + 5 = 3 * (8 * x^2 + z + 3)) ↔ 
  (x = (1 + Real.sqrt 19) / 4 ∨ x = (1 - Real.sqrt 19) / 4) := by
  sorry

end solve_for_x_l378_378992


namespace abe_age_sum_is_31_l378_378838

-- Define the present age of Abe
def abe_present_age : ℕ := 19

-- Define Abe's age 7 years ago
def abe_age_7_years_ago : ℕ := abe_present_age - 7

-- Define the sum of Abe's present age and his age 7 years ago
def abe_age_sum : ℕ := abe_present_age + abe_age_7_years_ago

-- Prove that the sum is 31
theorem abe_age_sum_is_31 : abe_age_sum = 31 := 
by 
  sorry

end abe_age_sum_is_31_l378_378838


namespace group_total_l378_378720

-- The given data from the problem
def I : ℕ := 35  -- People who have visited Iceland
def N : ℕ := 23  -- People who have visited Norway
def B : ℕ := 31  -- People who have visited both Iceland and Norway
def Ne : ℕ := 33 -- People who have visited neither country

-- Using these definitions, our objective is to prove T = 60
theorem group_total : let E := I + N - B in E + Ne = 60 := 
by
  sorry

end group_total_l378_378720


namespace original_hexagon_is_centrally_symmetric_l378_378089

theorem original_hexagon_is_centrally_symmetric
  (original_hexagon : Type)
  (is_hexagon : isRegularHexagon original_hexagon)
  (equilateral_triangles : ∀ (side : original_hexagon.edge), EquilateralTriangle side)
  (midpoints_form_regular_hexagon : isRegularHexagon (midpoints original_hexagon equilateral_triangles)) :
  centrally_symmetric original_hexagon :=
sorry

end original_hexagon_is_centrally_symmetric_l378_378089


namespace log_base_one_fourth_of_sixteen_l378_378263

theorem log_base_one_fourth_of_sixteen : log (1/4) 16 = -2 :=  sorry

end log_base_one_fourth_of_sixteen_l378_378263


namespace log_base_frac_l378_378233

theorem log_base_frac (x : ℝ) : log (1/4) 16 = x → x = -2 := by
  sorry

end log_base_frac_l378_378233


namespace sum_of_cubes_l378_378894

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
sorry

end sum_of_cubes_l378_378894


namespace percent_area_covered_by_hexagons_l378_378057

theorem percent_area_covered_by_hexagons (a : ℝ) (h1 : 0 < a) :
  let large_square_area := 4 * a^2
  let hexagon_contribution := a^2 / 4
  (hexagon_contribution / large_square_area) * 100 = 25 := 
by
  sorry

end percent_area_covered_by_hexagons_l378_378057


namespace three_non_zero_digit_1995_numbers_exist_l378_378442

theorem three_non_zero_digit_1995_numbers_exist :
  ∃ (num1 num2 num3 : ℕ), 
    (∀ d ∈ digits 10 num1, d ≠ 0) ∧
    (∀ d ∈ digits 10 num2, d ≠ 0) ∧
    (∀ d ∈ digits 10 num3, d ≠ 0) ∧
    (length (digits 10 num1) = 1995) ∧
    (length (digits 10 num2) = 1995) ∧
    (length (digits 10 num3) = 1995) ∧
    (num1 + num2 = num3) := by
  sorry

end three_non_zero_digit_1995_numbers_exist_l378_378442


namespace find_length_from_center_to_side_l378_378956

theorem find_length_from_center_to_side 
    (side : ℝ) (area_triangle : ℝ) (area_hexagon : ℝ) (total_area : ℝ) : 
    side = 2 → total_area = 4 → area_triangle = (total_area / 5) → 
    (∃ x : ℝ, area_triangle = (1/2 * side * x) ∧ x = (4/5)) :=
begin
  intros hside htotal_area harea_triangle,
  use (4 / 5),
  split,
  {
    rw [hside, htotal_area, harea_triangle],
    norm_num,
  },
  {
    norm_num,
  }
end

end find_length_from_center_to_side_l378_378956


namespace seating_arrangements_l378_378729

theorem seating_arrangements (n m: ℕ) (h₁ : n = 4) (h₂ : m = 5):
  (m - 1).factorial = 24 :=
by {
  rw [h₁, h₂, Nat.factorial],
  norm_num,
  sorry
}

end seating_arrangements_l378_378729


namespace simplify_fraction_l378_378800

theorem simplify_fraction : (75 : ℚ) / (100 : ℚ) = (3 : ℚ) / (4 : ℚ) :=
by
  sorry

end simplify_fraction_l378_378800


namespace atLeastOneNotLessThanTwo_l378_378309

open Real

theorem atLeastOneNotLessThanTwo (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) → False := 
by
  sorry

end atLeastOneNotLessThanTwo_l378_378309


namespace smallest_integer_value_l378_378489

theorem smallest_integer_value (a : ℕ → ℝ) (h_sum : ∑ i in finset.range 8, a i = 4 / 3)
    (h_pos_sum7 : ∀ i < 8, ∑ j in finset.range 8 \ {i}, a j > 0) : 
    ∃ (a_min : ℤ), (∀ i < 8, a i ≥ a_min) ∧ (∀ m < a_min, ∃ i < 8, a i < m) ∧ a_min = -7 :=
by
  sorry

end smallest_integer_value_l378_378489


namespace marks_in_mathematics_l378_378208

-- Definitions for the given conditions in the problem
def marks_in_english : ℝ := 86
def marks_in_physics : ℝ := 82
def marks_in_chemistry : ℝ := 87
def marks_in_biology : ℝ := 81
def average_marks : ℝ := 85
def number_of_subjects : ℕ := 5

-- Defining the total marks based on the provided conditions
def total_marks : ℝ := average_marks * number_of_subjects

-- Proving that the marks in mathematics are 89
theorem marks_in_mathematics : total_marks - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) = 89 :=
by
  sorry

end marks_in_mathematics_l378_378208


namespace total_dogwood_trees_l378_378722

def worker_plant_rate (i : ℕ) : ℕ → ℝ :=
  match i with
  | 1 => λ t, 2 * (if t = 1 then 3 - 0.5 else 1.5 - 0.25)
  | 2 => λ t, 3 * (if t = 1 then 3 - 0.5 else 1.5 - 0.25)
  | 3 => λ t, 1 * (if t = 1 then 3 - 0.75 else 1.5)
  | 4 => λ t, 4 * (if t = 1 then 3 - 0.75 else 1.5)
  | 5 => λ t, 2.5 * (if t = 1 then 3 else 1.5 - 0.5)
  | 6 => λ t, 3.5 * (if t = 1 then 3 else 1.5 - 0.5)
  | 7 => λ t, 1.5 * (if t = 1 then 3 - 0.25 else 1.5 - 0.75)
  | 8 => λ t, 2 * (if t = 1 then 3 - 0.25 else 1.5 - 0.75)
  | _ => 0

theorem total_dogwood_trees :
  let initial_trees := 7
  let planted_trees := ∑ i in finset.range 8, worker_plant_rate (i + 1) 1 + worker_plant_rate (i + 1) 2
  initial_trees + planted_trees = 81 :=
sorry

end total_dogwood_trees_l378_378722


namespace petya_not_lose_in_simple_state_vasya_guarantee_win_in_complex_state_l378_378726

-- Definitions of cities, roads, states, and game conditions
variables {City : Type} [fintype City]

-- State definition: cities connected such that there's a path between any two cities
structure State :=
  (connected : ∀ (c1 c2 : City), c1 ≠ c2 → (∃ path : List (City × City), path.head = c1 ∧ path.last = c2 ∧ ∀ (e : City × City) (i : Fin path.length), e = path[i]))

-- Simple state definition
def is_simple (s : State) : Prop :=
  ∀ (c : City), ∀ (road : List (City × City)), road.head = c ∧ road.last = c ∧ (∀ (i : Fin road.length), road[i] ≠ road[j]) → False

-- Complex state definition
def is_complex (s : State) : Prop :=
  ∃ (c : City), ∃ (road : List (City × City)), road.head = c ∧ road.last = c ∧ (∀ (i : Fin road.length), road[i] ≠ road[j])

-- Game conditions and player strategies
structure Game :=
  (state : State)
  (petya_strategy : ∀ (current_city : City), List (City × City))
  (vasya_strategy : ∀ (current_city : City), City × City)

-- Theorems to prove
theorem petya_not_lose_in_simple_state (s : State) (hf : is_simple s) : 
  ∀ (game : Game), ∃ (moves : ℕ), (game.petya_strategy game.state.connected) ≠ ∅ → False :=
sorry

theorem vasya_guarantee_win_in_complex_state (s : State) (hf : is_complex s) :
  ∀ (game : Game), ∃ (moves : ℕ), (game.vasya_strategy game.state.connected) = ∅ :=
sorry

end petya_not_lose_in_simple_state_vasya_guarantee_win_in_complex_state_l378_378726


namespace exists_positive_integer_n_l378_378632

def num_divisors (n : ℕ) : ℕ := if n = 0 then 0 else (n.divisors.card)

theorem exists_positive_integer_n (m : ℕ) (h_odd : odd m) :
  ∃ n : ℕ, n > 0 ∧ (num_divisors (n^2) / num_divisors n) = m :=
sorry

end exists_positive_integer_n_l378_378632


namespace log_base_one_fourth_of_sixteen_l378_378261

theorem log_base_one_fourth_of_sixteen : log (1/4) 16 = -2 :=  sorry

end log_base_one_fourth_of_sixteen_l378_378261


namespace log_one_fourth_sixteen_l378_378252

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 := 
by
  let x := log (1 / 4) 16
  have h₁ : (1 / 4) ^ x = 16 := by simp [log_eq_iff]
  have h₂ : (4 ^ (-1)) ^ x = 16 := by rw [one_div, inv_pow]
  have h₃ : 4 ^ (-x) = 16 := by simp [pow_mul]
  have h₄ : 16 = 4 ^ 2 := by norm_num
  rw [h₄] at h₃
  have h₅ : -x = 2 := by exact pow_inj (lt_trans zero_lt_one (by norm_num)) zero_lt_four h₃
  have h₆ : x = -2 := by linarith
  exact h₆

end log_one_fourth_sixteen_l378_378252


namespace garage_sale_items_count_l378_378577

theorem garage_sale_items_count (h_highest : 15 = n_highest) (h_lowest : 22 = n_lowest) : 
  n_highest + n_lowest - 1 = 36 := 
by 
  have h1 : n_highest = 14 := sorry
  have h2 : n_lowest = 22 := sorry
  rw [h1, h2]
  norm_num

end garage_sale_items_count_l378_378577


namespace ant_climb_ways_l378_378969

theorem ant_climb_ways : 
  let climbing_sequences (n : ℕ) : ℕ := 
    if n = 12 then 12 else 0
  in climbing_sequences 12 = 12 :=
sorry

end ant_climb_ways_l378_378969


namespace triangle_count_relationship_l378_378741

theorem triangle_count_relationship :
  let n_0 : ℕ := 20
  let n_1 : ℕ := 19
  let n_2 : ℕ := 18
  n_0 > n_1 ∧ n_1 > n_2 :=
by
  let n_0 := 20
  let n_1 := 19
  let n_2 := 18
  have h0 : n_0 > n_1 := by sorry
  have h1 : n_1 > n_2 := by sorry
  exact ⟨h0, h1⟩

end triangle_count_relationship_l378_378741


namespace hannahs_brothers_l378_378336

theorem hannahs_brothers (B : ℕ) (h1 : ∀ (b : ℕ), b = 8) (h2 : 48 = 2 * (8 * B)) : B = 3 :=
by
  sorry

end hannahs_brothers_l378_378336


namespace base12_addition_C97_26A_l378_378566

theorem base12_addition_C97_26A :
  let C := 12 in 
  ∀ (A B : ℕ), 
  A = 10 → 
  B = 11 → 
  let C97_12 := C * C^2 + 9 * C + 7 in
  let 26A_12 := 2 * C^2 + 6 * C + A in
  let 341B_12 := 3 * C^2 + 4 * C + B in
  C97_12 + 26A_12 = 341B_12 := 
by {
  intros,
  sorry
}

end base12_addition_C97_26A_l378_378566


namespace sum_of_eight_terms_l378_378182

theorem sum_of_eight_terms :
  (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) = 3125000 :=
by
  sorry

end sum_of_eight_terms_l378_378182


namespace percentage_female_on_duty_l378_378784

-- Definitions as per conditions in the problem:
def total_on_duty : ℕ := 240
def female_on_duty := total_on_duty / 2 -- Half of those on duty are female
def total_female_officers : ℕ := 300
def percentage_of_something (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Statement of the problem to prove
theorem percentage_female_on_duty : percentage_of_something female_on_duty total_female_officers = 40 :=
by
  sorry

end percentage_female_on_duty_l378_378784


namespace A_can_complete_work_in_4_days_l378_378137

-- Definitions based on conditions
def work_done_in_one_day (days : ℕ) : ℚ := 1 / days

def combined_work_done_in_two_days (a b c : ℕ) : ℚ :=
  work_done_in_one_day a + work_done_in_one_day b + work_done_in_one_day c

-- Theorem statement based on the problem
theorem A_can_complete_work_in_4_days (A B C : ℕ) 
  (hB : B = 8) (hC : C = 8) 
  (h_combined : combined_work_done_in_two_days A B C = work_done_in_one_day 2) :
  A = 4 :=
sorry

end A_can_complete_work_in_4_days_l378_378137


namespace bisectors_incircle_intersect_l378_378465

universe u
variables {Point : Type u} [EuclideanGeometry Point]

noncomputable def intersects_line (I B' C' : Point) (incircle : Circle Point) : Prop := 
  ∃ Q : Point, Q ∈ incircle ∧ LiesOn Q (Line.mk B' C')

theorem bisectors_incircle_intersect {A B C B' C' I D E F : Point} (incircle : Circle Point)
  (hB : AngleBisector B C A B') 
  (hC : AngleBisector C B A C') 
  (h_incenter : Incenter I \triangle ABC) 
  (hD : TangentPoint incircle B C D)
  (hE : TangentPoint incircle C A E)
  (hF : TangentPoint incircle A B F) :
  intersects_line I B' C' incircle := 
sorry

end bisectors_incircle_intersect_l378_378465


namespace nail_squares_with_2n_minus_2_nails_l378_378781
-- import the necessary libraries

-- define the problem in Lean 4
theorem nail_squares_with_2n_minus_2_nails {Square : Type} (colors : Finset (Set Square)) (n : ℕ) (h_color_num : colors.card = n) 
  (h_parallel : ∀ {s1 s2 : Square}, s1 ∈ ⋃₀ colors → s2 ∈ ⋃₀ colors → sides_parallel s1 s2)
  (h_nailable : ∀ (c : Set Square) (hc : c ∈ colors), ∀ t (ht : t.card = n) (htc : t ⊆ colors), n_nailable t)
  : ∃ c ∈ colors, n_nails c ≤ 2 * n - 2 := 
sorry

end nail_squares_with_2n_minus_2_nails_l378_378781


namespace smallest_base_10_integer_exists_l378_378875

theorem smallest_base_10_integer_exists :
  ∃ (x a b : ℕ), (a > 2) ∧ (b > 2) ∧ (x = 2 * a + 1) ∧ (x = b + 2) ∧ (x = 7) :=
by
  sorry

end smallest_base_10_integer_exists_l378_378875


namespace log_base_frac_l378_378229

theorem log_base_frac (x : ℝ) : log (1/4) 16 = x → x = -2 := by
  sorry

end log_base_frac_l378_378229


namespace sum_series_l378_378987

theorem sum_series :
  ∑' n : ℕ, (0 < n) → 
  (sqrt (4 * n + 2) / ((4 * n) ^ 2 * (4 * n + 4) ^ 2)) = 1 / 512 := 
by
  sorry

end sum_series_l378_378987


namespace tray_height_is_5_sqrt_2_l378_378160

noncomputable def height_of_tray (side_length cut_length : ℝ) : ℝ :=
  if side_length = 120 ∧ cut_length = 10 then 5 * real.sqrt 2 else 0

-- Prove that the height of the tray is 5√2 based on the above conditions.
theorem tray_height_is_5_sqrt_2:
  height_of_tray 120 10 = 5 * real.sqrt 2 :=
by
  sorry

end tray_height_is_5_sqrt_2_l378_378160


namespace simplify_expression_l378_378290

theorem simplify_expression (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  log (cos x * tan x + 1 - 2 * sin (x / 2) ^ 2) + log (sqrt 2 * cos (x - π / 4)) - log (1 + sin (2 * x)) = 0 := 
sorry

end simplify_expression_l378_378290


namespace number_of_odd_coefficients_l378_378441

theorem number_of_odd_coefficients (n : ℕ) : 
  let d := (nat.binary_to_nat n).sum_digits in
  number_of_odd_coeffs ((1 + X)^n) = 2^d :=
sorry

end number_of_odd_coefficients_l378_378441


namespace abs_add_eq_abs_sub_implies_mul_eq_zero_l378_378701

variable {a b : ℝ}

theorem abs_add_eq_abs_sub_implies_mul_eq_zero (h : |a + b| = |a - b|) : a * b = 0 :=
sorry

end abs_add_eq_abs_sub_implies_mul_eq_zero_l378_378701


namespace inequality_4th_power_l378_378440

theorem inequality_4th_power (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_ineq : a ≥ b) :
  (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 :=
sorry

end inequality_4th_power_l378_378440


namespace ellipse_proof_l378_378317

theorem ellipse_proof (a b : ℝ) (h : a > b ∧ b > 0)
  (heq : ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → True)
  (hfoci : ∀ (F1 F2 P : ℝ × ℝ), F1.1 < 0 ∧ F2.1 > 0 ∧ (P.1 = F2.1 ∧ P.2 ≠ 0))
  (hsin : sin (angle P F1 F2) = 1 / 3) : a = sqrt 2 * b :=
by
  sorry

end ellipse_proof_l378_378317


namespace smallest_base10_integer_l378_378879

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end smallest_base10_integer_l378_378879


namespace winner_N_10_winner_N_12_winner_N_15_winner_N_30_l378_378518

open Nat

/-- Determines the winner when N = 10 -/
theorem winner_N_10: "first player wins" := sorry

/-- Determines the winner when N = 12 -/
theorem winner_N_12: "first player wins" := sorry

/-- Determines the winner when N = 15 -/
theorem winner_N_15: "second player wins" := sorry

/-- Determines the winner when N = 30 -/
theorem winner_N_30: "first player wins" := sorry

end winner_N_10_winner_N_12_winner_N_15_winner_N_30_l378_378518


namespace angle_halved_first_quadrant_l378_378698

theorem angle_halved_first_quadrant
  (θ : ℝ)
  (hθ1 : π / 2 < θ)
  (hθ2 : θ < π)
  (h : cos (θ / 2) - sin (θ / 2) = real.sqrt (1 - sin θ)) :
  (0 < θ / 2 ∧ θ / 2 < π / 2) :=
by { sorry }

end angle_halved_first_quadrant_l378_378698


namespace pen_price_l378_378080

theorem pen_price :
  ∃ x y : ℕ, 3 * x + 4 * y = 316 ∧ 5 * x + 2 * y = 348 ∧ x = 54 :=
by {
  use 54,
  use 32, -- assumed value to satisfy the system, subject to calculation
  split,
  {
    calc
      3 * 54 + 4 * 32 = 3 * 54 + 4 * 32 : by rw [nat.mul_comm 3 54, nat.mul_comm 4 32]
      ... = 162 + 128 : by refl
      ... = 290 : by norm_num
    sorry,
  },
  split,
  {
    calc
      5 * 54 + 2 * 32 = 5 * 54 + 2 * 32 : by rw [nat.mul_comm 5 54, nat.mul_comm 2 32]
      ... = 270 + 64 : by refl
      ... = 334 : by norm_num
    sorry,
  },
  refl,
}

end pen_price_l378_378080


namespace z_coordinate_of_point_on_line_l378_378938

theorem z_coordinate_of_point_on_line (t : ℝ)
  (h₁ : (1 + 3 * t, 3 + 2 * t, 2 + 4 * t) = (x, 7, z))
  (h₂ : x = 1 + 3 * t) :
  z = 10 :=
sorry

end z_coordinate_of_point_on_line_l378_378938


namespace area_of_quadrilateral_l378_378435

noncomputable def shoelace_area (v : List (ℕ × ℕ)) : ℝ :=
  let n := v.length - 1
  0.5 * abs (∑ i in List.range n, (v[i].1 : ℝ) * (v[(i + 1) % n].2 : ℝ) - (v[i].2 : ℝ) * (v[(i + 1) % n].1 : ℝ))

theorem area_of_quadrilateral :
  let vertices := [(1,1), (2,3), (4,2), (3,0)]
  shoelace_area vertices = 5 := by
  sorry

end area_of_quadrilateral_l378_378435


namespace number_of_sequences_l378_378768

theorem number_of_sequences (n k : ℕ) (h₁ : 1 ≤ k) (h₂ : k ≤ n) :
  ∃ C : ℕ, C = Nat.choose (Nat.floor ((n + 2 - k) / 2) + k - 1) k :=
sorry

end number_of_sequences_l378_378768


namespace smallest_base10_integer_l378_378883

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end smallest_base10_integer_l378_378883


namespace cricketer_runs_l378_378930

theorem cricketer_runs (R x : ℝ) : 
  (R / 85 = 12.4) →
  ((R + x) / 90 = 12.0) →
  x = 26 := 
by
  sorry

end cricketer_runs_l378_378930


namespace product_of_third_side_in_right_triangle_l378_378296

theorem product_of_third_side_in_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 7) :
    let c1 := real.sqrt (a^2 + b^2)
    let c2 := real.sqrt (b^2 - a^2)
    let product := c1 * c2
    real.approx product 33.2 0.1 :=
by
  sorry

end product_of_third_side_in_right_triangle_l378_378296


namespace log_base_one_four_of_sixteen_l378_378243

theorem log_base_one_four_of_sixteen : log (1 / 4) 16 = -2 := by
  sorry

end log_base_one_four_of_sixteen_l378_378243


namespace find_x_l378_378411

-- Define the diamond operator
def diamond (a b : ℕ) : ℕ := a + Real.sqrt (b + Real.sqrt b + Real.sqrt b + Real.sqrt b)

-- State the main theorem
theorem find_x (x : ℕ) : diamond 5 x = 12 → x = 42 := by
  sorry

end find_x_l378_378411


namespace pair_not_proportional_l378_378513

theorem pair_not_proportional (a1 a2 b1 b2 : ℕ) (proportion1 proportion2 : ℚ)
  (ha1 : a1 = 6) (ha2 : a2 = 9) (hb1 : b1 = 9) (hb2 : b2 = 12) :
  proportion1 = a1 / a2 → proportion2 = b1 / b2 → proportion1 ≠ proportion2 :=
by 
  intros 
  have h1: proportion1 = 6 / 9, from ha1.symm ▸ ha2.symm ▸ rfl,
  have h2: proportion2 = 9 / 12, from hb1.symm ▸ hb2.symm ▸ rfl,
  have h12: 6 / 9 ≠ 9 / 12, from by norm_num,
  exact h12.symm

end pair_not_proportional_l378_378513


namespace minimum_number_of_guests_l378_378121

theorem minimum_number_of_guests (total_food : ℝ) (max_food_per_guest : ℝ) (H₁ : total_food = 406) (H₂ : max_food_per_guest = 2.5) : 
  ∃ n : ℕ, (n : ℝ) ≥ 163 ∧ total_food / max_food_per_guest ≤ (n : ℝ) := 
by
  sorry

end minimum_number_of_guests_l378_378121


namespace matches_left_l378_378006

-- Define the initial number of matches
def initial_matches : ℕ := 100

-- Define the percentage of matches dropped in the creek
def percentage_dropped : ℝ := 0.15

-- Calculate the number of matches dropped in the creek
def dropped_matches := (percentage_dropped * initial_matches).toNat

-- Define the factor by which the dog ate matches
def dog_ate_factor : ℕ := 3

-- Calculate the number of matches the dog ate
def dog_ate_matches := dog_ate_factor * dropped_matches

-- Calculate the total number of matches lost
def total_matches_lost := dropped_matches + dog_ate_matches

-- Calculate the remaining matches
def remaining_matches := initial_matches - total_matches_lost

-- The statement that needs to be proved
theorem matches_left : remaining_matches = 40 :=
by
  sorry

end matches_left_l378_378006


namespace log_base_one_four_of_sixteen_l378_378237

theorem log_base_one_four_of_sixteen : log (1 / 4) 16 = -2 := by
  sorry

end log_base_one_four_of_sixteen_l378_378237


namespace standard_equation_of_ellipse_length_PQ_m_eq_1_no_value_of_m_for_area_delta_OPQ_4_3_l378_378651

namespace MathProofs

open Real

-- Conditions for the ellipse and the triangle
variable {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3: a > b)
(ha : a = sqrt 2) (hypotenuse_length : 2 * sqrt 2)
(var c : ℝ) (hc : c = sqrt 2)

-- Condition for the line
variable {m : ℝ}

-- Defining the ellipse C
def ellipse_C (x y : ℝ) :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Finding the standard equation of ellipse C
theorem standard_equation_of_ellipse :
  (a = sqrt 2) ∧ (b = sqrt 2) → 
  ∀ x y, ellipse_C a b ha hypotenuse_length x y
  = ((x^2) / 4 + (y^2) / 2 = 1) := sorry

-- Finding the length of segment PQ when m = 1
theorem length_PQ_m_eq_1 :
  ∀ x y (m : ℝ), (m = 1) →
  length_PQ (ellipse_C a b ha hypotenuse_length x y) m
  = (4 / 3 * sqrt 5) := sorry

-- Determining the value of m for the area of ΔOPQ to be 4/3
theorem no_value_of_m_for_area_delta_OPQ_4_3 :
  ∀ x y,
  ¬ (∃ (m : ℝ), area_OPQ (ellipse_C a b ha hypotenuse_length x y) m = 4 / 3) := sorry
end MathProofs

end standard_equation_of_ellipse_length_PQ_m_eq_1_no_value_of_m_for_area_delta_OPQ_4_3_l378_378651


namespace smallest_base_10_integer_l378_378869

theorem smallest_base_10_integer (a b : ℕ) (ha : a > 2) (hb : b > 2) 
  (h1: 21_a = 2 * a + 1) (h2: 12_b = b + 2) : 2 * a + 1 = 7 :=
by 
  sorry

end smallest_base_10_integer_l378_378869


namespace adjugate_power_null_l378_378753

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ)

def adjugate (A : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ := sorry

theorem adjugate_power_null (A : Matrix (Fin n) (Fin n) ℂ) (m : ℕ) (hm : 0 < m) (h : (adjugate A) ^ m = 0) : 
  (adjugate A) ^ 2 = 0 := 
sorry

end adjugate_power_null_l378_378753


namespace scientific_notation_113700_l378_378830

theorem scientific_notation_113700 :
  ∃ (a : ℝ) (b : ℤ), 113700 = a * 10 ^ b ∧ a = 1.137 ∧ b = 5 :=
by
  sorry

end scientific_notation_113700_l378_378830


namespace product_of_two_numbers_l378_378084

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x + y = 8 * (x - y)) 
  (h3 : x * y = 40 * (x - y)) 
  : x * y = 63 := 
by 
  sorry

end product_of_two_numbers_l378_378084


namespace max_area_triangle_PQR_l378_378082

-- Define the conditions of the problem
def PR_QR_ratio_30_31 (PR QR : ℝ) : Prop := PR / QR = 30 / 31

-- The maximum area problem for triangle PQR
theorem max_area_triangle_PQR :
  ∀ (P Q R : Type) (PQ PR QR : ℝ),
  PQ = 10 →
  PR_QR_ratio_30_31 PR QR →
  (∃ (area : ℝ), area ≤ 1250) :=
begin
  intros P Q R PQ PR QR hPQ hPRQR,
  sorry
end

end max_area_triangle_PQR_l378_378082


namespace solution_correct_l378_378025

noncomputable def solve_system (a b c : ℝ) : ℝ × ℝ × ℝ :=
  let x := (3 * c - a - b) / 4
  let y := (3 * b - a - c) / 4
  let z := (3 * a - b - c) / 4
  (x, y, z)

theorem solution_correct (a b c : ℝ) (x y z : ℝ) :
  (x + y + 2 * z = a) →
  (x + 2 * y + z = b) →
  (2 * x + y + z = c) →
  (x, y, z) = solve_system a b c :=
by sorry

end solution_correct_l378_378025


namespace percentage_increase_B_over_C_l378_378825

noncomputable def A_annual_income : ℝ := 571200
noncomputable def C_monthly_income : ℝ := 17000

def A_monthly_income := A_annual_income / 12
def B_monthly_income := 2 * A_monthly_income / 5

def percentage_increase (B_m C_m : ℝ) := ((B_m - C_m) / C_m) * 100

theorem percentage_increase_B_over_C :
  percentage_increase B_monthly_income C_monthly_income = 12 :=
by
  sorry

end percentage_increase_B_over_C_l378_378825


namespace log_base_one_fourth_of_sixteen_l378_378257

theorem log_base_one_fourth_of_sixteen :
  log (1 / 4 : ℝ) (16 : ℝ) = -2 :=
sorry

end log_base_one_fourth_of_sixteen_l378_378257


namespace DL_is_sqrt_2_div_2_l378_378499

noncomputable def DL_length_equivalence : Type :=
  let DE := 8
  let EF := 10
  let DF := 6
  let angle_EFD := Real.arccos ((DF^2 + EF^2 - DE^2) / (2 * DF * EF))
  let s := (DF + EF + DE) / 2
  let Area_DEF := Real.sqrt (s * (s - DF) * (s - EF) * (s - DE))
  let DL := Real.sqrt (Area_DEF / (DE * DF)) in
  DL = Real.sqrt 2 / 2

theorem DL_is_sqrt_2_div_2 (DE EF DF : ℝ) (h1 : DE = 8) (h2 : EF = 10) (h3 : DF = 6) :
  DL_length_equivalence :=
by
  -- This is to avoid calculation steps and directly assume the proof concludes here.
  sorry

end DL_is_sqrt_2_div_2_l378_378499


namespace find_smaller_page_l378_378109

theorem find_smaller_page (sum_pages : ℕ) (h : sum_pages = 185) : ∃ n : ℕ, n + (n + 1) = sum_pages ∧ n = 92 :=
by
  existsi (92 : ℕ)
  split
  . rw h
    norm_num
  . refl

end find_smaller_page_l378_378109


namespace bowling_tournament_outcomes_l378_378721
open nat

def game_outcomes : ℕ := 2

theorem bowling_tournament_outcomes :
  ∃ (orders : ℕ), orders = game_outcomes ^ 5 ∧ orders = 32 :=
begin
  use game_outcomes ^ 5,
  split,
  { refl }, -- game_outcomes ^ 5
  { norm_num }, -- 32
end

end bowling_tournament_outcomes_l378_378721


namespace quotient_of_distinct_cubes_mod_13_l378_378207

theorem quotient_of_distinct_cubes_mod_13 :
  let remainders := (Finset.range 11).image (λ n, (n^3 % 13))
  let m := remainders.sum
  m / 13 = 2 :=
by
  let remainders := (Finset.range 11).image (λ n, (n^3 % 13))
  let m := remainders.sum
  have h_remainders : remainders = {1, 8, 12, 3, 2} := sorry
  have sum_distinct : m = 26 := by
    -- We will prove the sum of the set {1, 8, 12, 3, 2}
    rw h_remainders
    exact Finset.sum_of_eq_add (Finset.sum_of_finset [1, 8, 12, 3, 2]) sorry
  rw [sum_distinct, Nat.div_eq m 13]
  exact Nat.div_self 13

  sorry

end quotient_of_distinct_cubes_mod_13_l378_378207


namespace student_number_choice_l378_378561

theorem student_number_choice (x : ℤ) (h : 2 * x - 138 = 104) : x = 121 :=
sorry

end student_number_choice_l378_378561


namespace log_base_frac_l378_378232

theorem log_base_frac (x : ℝ) : log (1/4) 16 = x → x = -2 := by
  sorry

end log_base_frac_l378_378232


namespace solve_system_equations_l378_378459

theorem solve_system_equations (x y : ℝ) :
  x + y = 0 ∧ 2 * x + 3 * y = 3 → x = -3 ∧ y = 3 :=
by {
  -- Leave the proof as a placeholder with "sorry".
  sorry
}

end solve_system_equations_l378_378459


namespace max_value_of_f_l378_378320

noncomputable def f (f'_1 : ℝ) (x : ℝ) : ℝ := 2 * f'_1 * Math.log x - 1 / x
noncomputable def f_deriv (f'_1 : ℝ) (x : ℝ) : ℝ := -2 / x + 1 / (x * x)
noncomputable def f'_at_1 : ℝ := -1

theorem max_value_of_f :
  ∃ x : ℝ, x = 1 / 2 ∧ f f'_at_1 x = 2 * Math.log 2 - 2 :=
by 
  use 1 / 2
  split
  · rfl
  · sorry

end max_value_of_f_l378_378320


namespace ratio_a_f_l378_378704

theorem ratio_a_f (a b c d e f : ℕ)
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13)
  (h4 : d / e = 2 / 3)
  (h5 : e / f = 7 / 5) :
  a / f = 7 / 6 := by
  sorry

end ratio_a_f_l378_378704


namespace ordered_pair_exists_l378_378625

noncomputable theory

theorem ordered_pair_exists :
  ∃ (a b : ℤ), 0 < a ∧ 0 < b ∧ a < b ∧ (sqrt (1 + sqrt (24 + 15 * sqrt 3)) = sqrt a + sqrt b) :=
begin
  use [2, 3],
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  sorry
end

end ordered_pair_exists_l378_378625


namespace log_one_fourth_sixteen_l378_378246

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 := 
by
  let x := log (1 / 4) 16
  have h₁ : (1 / 4) ^ x = 16 := by simp [log_eq_iff]
  have h₂ : (4 ^ (-1)) ^ x = 16 := by rw [one_div, inv_pow]
  have h₃ : 4 ^ (-x) = 16 := by simp [pow_mul]
  have h₄ : 16 = 4 ^ 2 := by norm_num
  rw [h₄] at h₃
  have h₅ : -x = 2 := by exact pow_inj (lt_trans zero_lt_one (by norm_num)) zero_lt_four h₃
  have h₆ : x = -2 := by linarith
  exact h₆

end log_one_fourth_sixteen_l378_378246


namespace coordinates_of_A_l378_378814

-- Defining the point A
def point_A : ℤ × ℤ := (1, -4)

-- Statement that needs to be proved
theorem coordinates_of_A :
  point_A = (1, -4) :=
by
  sorry

end coordinates_of_A_l378_378814


namespace molecular_weight_2N_5O_l378_378867

def molecular_weight (num_N num_O : ℕ) (atomic_weight_N atomic_weight_O : ℝ) : ℝ :=
  (num_N * atomic_weight_N) + (num_O * atomic_weight_O)

theorem molecular_weight_2N_5O :
  molecular_weight 2 5 14.01 16.00 = 108.02 :=
by
  -- proof goes here
  sorry

end molecular_weight_2N_5O_l378_378867


namespace volume_of_inscribed_sphere_of_cube_l378_378492

theorem volume_of_inscribed_sphere_of_cube (a : ℝ) (h : a = 2) : 
  ∀ (V : ℝ), V = (4 * Real.pi) / 3 ↔ V = volume_of_sphere (a / 2) :=
by
  sorry

def volume_of_sphere (r : ℝ) := (4 * Real.pi * r^3) / 3

end volume_of_inscribed_sphere_of_cube_l378_378492


namespace chosen_number_l378_378162

theorem chosen_number (x : ℕ) (h : 5 * x - 138 = 102) : x = 48 :=
sorry

end chosen_number_l378_378162


namespace voting_cases_count_l378_378569

theorem voting_cases_count : 
  let P := {'Jungkook, 'Jimin, 'Yoongi}
  number_of_unordered_pairs_with_repetition(P) = 6 := sorry

end voting_cases_count_l378_378569


namespace square_area_is_16_l378_378713

noncomputable def area_of_square
  (area_APQ : ℝ) (area_PBS : ℝ) (area_QRC : ℝ) : ℝ :=
  if h : area_APQ = 4 ∧ area_PBS = 4 ∧ area_QRC = 12 then 16 else 0

theorem square_area_is_16
  (area_APQ : ℝ) (area_PBS : ℝ) (area_QRC : ℝ) :
  area_APQ = 4 → area_PBS = 4 → area_QRC = 12 → area_of_square area_APQ area_PBS area_QRC = 16 :=
by
  intros h1 h2 h3
  unfold area_of_square
  rw [if_pos]
  · refl
  · exact ⟨h1, h2, h3⟩

end square_area_is_16_l378_378713


namespace weight_of_each_bag_l378_378941

theorem weight_of_each_bag (empty_weight loaded_weight : ℕ) (number_of_bags : ℕ) (weight_per_bag : ℕ)
    (h1 : empty_weight = 500)
    (h2 : loaded_weight = 1700)
    (h3 : number_of_bags = 20)
    (h4 : loaded_weight - empty_weight = number_of_bags * weight_per_bag) :
    weight_per_bag = 60 :=
by
  sorry

end weight_of_each_bag_l378_378941


namespace total_gold_is_100_l378_378335

-- Definitions based on conditions
def GregsGold : ℕ := 20
def KatiesGold : ℕ := GregsGold * 4
def TotalGold : ℕ := GregsGold + KatiesGold

-- Theorem to prove
theorem total_gold_is_100 : TotalGold = 100 := by
  sorry

end total_gold_is_100_l378_378335


namespace max_abs_diff_ge_half_d_exists_seq_achieves_equality_l378_378654

variable {n : ℕ} (a : Fin n → ℝ)

def d_i (i : Fin n) : ℝ :=
  Finset.sup (Finset.univ.filter (λ j => j.val < i.val + 1)) (λ j => a j) -
  Finset.inf (Finset.univ.filter (λ j => i.val ≤ j.val)) (λ j => a j)

def d : ℝ := Finset.sup Finset.univ (λ i => d_i a i)

theorem max_abs_diff_ge_half_d (x : Fin n → ℝ)
    (h : ∀ i j, i.val < j.val → x i ≤ x j) :
    Finset.sup Finset.univ (λ i => |x i - a i|) ≥ d a / 2 :=
sorry

theorem exists_seq_achieves_equality :
    ∃ x : Fin n → ℝ, (∀ i j, i.val < j.val → x i ≤ x j) ∧
    Finset.sup Finset.univ (λ i => |x i - a i|) = d a / 2 :=
sorry

end max_abs_diff_ge_half_d_exists_seq_achieves_equality_l378_378654


namespace peanut_butter_cost_l378_378004

noncomputable def cost_of_peanut_butter (B : ℕ) (A : ℕ) : Prop :=
  A = 3 * B ∧ 0.5 * A - 0.5 * B = 3 → B = 3

axiom B : ℕ
axiom A : ℕ

theorem peanut_butter_cost : cost_of_peanut_butter B A :=
by
  sorry

end peanut_butter_cost_l378_378004


namespace fred_basketball_games_l378_378288

theorem fred_basketball_games (games_this_year games_last_year : ℕ) (h1 : games_this_year = 36) (h2 : games_last_year = 11) :
    games_this_year + games_last_year = 47 :=
by
  rw [h1, h2]
  exact rfl

end fred_basketball_games_l378_378288


namespace pairwise_coprime_f_sequence_l378_378398

def f (x : ℤ) : ℤ := x^2002 - x^2001 + 1

theorem pairwise_coprime_f_sequence (m : ℕ) (hm : m > 0) :
  ∀ i j : ℕ, i ≠ j → Nat.coprime (natAbs (f^[i] (m : ℤ))) (natAbs (f^[j] (m : ℤ))) :=
sorry

end pairwise_coprime_f_sequence_l378_378398


namespace num_mappings_l378_378402

-- Define the sets M and N
def M := {a, b, c}
def N := {-2, 0, 2}

-- Define the function type from M to N
def f : M → N

-- Define the conditions
def condition_1 : f a > f b := sorry
def condition_2 : f b ≥ f c := sorry

-- State the theorem
theorem num_mappings : set.count (f : M → N | condition_1 ∧ condition_2) = 4 := sorry

end num_mappings_l378_378402


namespace system_of_equations_solution_l378_378836

theorem system_of_equations_solution :
  ∃ x y : ℝ, (x + y = 3) ∧ (2 * x - 3 * y = 1) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end system_of_equations_solution_l378_378836


namespace grinder_purchase_price_l378_378747

-- Define conditions
variables (G : ℝ) -- Purchase price of the grinder
variables (mobile_purchase : ℝ) (mobile_sell : ℝ) (profit : ℝ)
variables (grinder_loss : ℝ)

-- Given conditions
def mobile_purchase_price := 8000
def mobile_sell_price := mobile_purchase + 0.10 * mobile_purchase
def overall_profit := profit = 500
def grinder_loss_condition := grinder_loss = 0.02
def grinder_sell_price := G - grinder_loss * G
def total_sell_price := grinder_sell_price + mobile_sell
def total_purchase_price := G + mobile_purchase

-- Main statement to prove
theorem grinder_purchase_price (h1 : mobile_purchase = mobile_purchase_price)
                              (h2 : overall_profit)
                              (h3 : grinder_loss_condition)
                              (h4 : mobile_sell = mobile_sell_price) :
                              total_sell_price = total_purchase_price + 500 → G = 15000 :=
by
  sorry

end grinder_purchase_price_l378_378747


namespace min_value_x_plus_y_l378_378310

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 / y + 1 / x = 4) :
  x + y ≥ 9 / 4 :=
sorry

end min_value_x_plus_y_l378_378310


namespace calculate_new_shipment_bears_l378_378959

theorem calculate_new_shipment_bears 
  (initial_bears : ℕ)
  (shelves : ℕ)
  (bears_per_shelf : ℕ)
  (total_bears_on_shelves : ℕ) 
  (h_total_bears_on_shelves : total_bears_on_shelves = shelves * bears_per_shelf)
  : initial_bears = 6 → shelves = 4 → bears_per_shelf = 6 → total_bears_on_shelves - initial_bears = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at *
  simp at *
  sorry

end calculate_new_shipment_bears_l378_378959


namespace log_base_one_fourth_of_sixteen_l378_378260

theorem log_base_one_fourth_of_sixteen :
  log (1 / 4 : ℝ) (16 : ℝ) = -2 :=
sorry

end log_base_one_fourth_of_sixteen_l378_378260


namespace brother_grade_l378_378270

def tells_truth (brother_claim : ℕ) : Prop := sorry -- This will need further formalization based on truth and sneeze conditions

theorem brother_grade
  (truth : ∀ (claim : ℕ), tells_truth claim → ∃ (sneezes : bool), sneezes = true)
  (no_sneeze5 : ¬ tells_truth 5)
  (sneeze4 : tells_truth 4)
  (no_sneeze3 : ¬ tells_truth 3) :
  ∃ (grade : ℕ), grade = 2 :=
by
  sorry

end brother_grade_l378_378270


namespace maximum_unique_planes_l378_378291

-- Define the number of points and the condition that no four points are coplanar.
axiom points_in_space (n : ℕ) : n = 15
axiom no_four_points_coplanar : Prop

-- The main theorem stating the maximum number of unique planes.
theorem maximum_unique_planes (h1 : points_in_space 15) (h2 : no_four_points_coplanar) : 
  ∃ max_planes : ℕ, max_planes = 455 :=
sorry

end maximum_unique_planes_l378_378291


namespace min_rows_required_l378_378077

-- Condition definitions
def number_of_students : ℕ := 2016
def seats_per_row : ℕ := 168
def max_students_per_school : ℕ := 45

-- Theorem statement to prove the minimum number of rows
theorem min_rows_required : (∀ students : ℕ, students ≤ number_of_students → 
  ∀ max_per_school : ℕ, max_per_school = max_students_per_school → 
  ∀ seats : ℕ, seats = seats_per_row → 
  ∃ rows : ℕ, rows = 16) :=
begin
  sorry
end

end min_rows_required_l378_378077


namespace most_likely_sitting_people_l378_378430

theorem most_likely_sitting_people :
  let num_people := 100
  let seats := 100
  let favorite_seats : Fin num_people → Fin seats := sorry
  -- Conditions related to people sitting behavior
  let sits_in_row (i : Fin num_people) : Prop :=
    ∀ j : Fin num_people, j < i → favorite_seats j ≠ favorite_seats i
  let num_sitting_in_row := Finset.card (Finset.filter sits_in_row (Finset.univ : Finset (Fin num_people)))
  -- Prove
  num_sitting_in_row = 10 := 
sorry

end most_likely_sitting_people_l378_378430


namespace min_value_f_when_a_0_extreme_value_points_of_f_l378_378321

noncomputable def f (x a : ℝ) : ℝ := ln x + x^2 - 2 * a * x + a^2

theorem min_value_f_when_a_0 :
  (∀ x ∈ set.Icc 1 real.exp 1, f x 0 ≥ 1) ∧ ∃ x ∈ set.Icc 1 real.exp 1, f x 0 = 1 :=
sorry

theorem extreme_value_points_of_f (a : ℝ) :
  (a ≤ 0 ∨ (0 < a ∧ a ≤ real.sqrt 2)) →
  (∀ x > 0, deriv (λ x, ln x + x^2 - 2 * a * x + a^2) x ≥ 0) ∧
  (∃ x < 0, deriv (λ x, ln x + x^2 - 2 * a * x + a^2) x = 0) ∧
  (a > real.sqrt 2 →
    (∃ x, x = (a - real.sqrt (a^2 - 2)) / 2 ∧
      (∃ y, y = (a + real.sqrt (a^2 - 2)) / 2 ∧
        (∀ z, z = (a - real.sqrt (a^2 - 2)) / 2 ∨ z = (a + real.sqrt (a^2 - 2)) / 2)))) :=
sorry

end min_value_f_when_a_0_extreme_value_points_of_f_l378_378321


namespace not_applicable_Rolles_theorem_l378_378323

noncomputable def f (x : ℝ) : ℝ := 1 - x^(2/3)

theorem not_applicable_Rolles_theorem : 
  ¬(continuous_on f (set.Icc (-1) 1) ∧ 
    (f (-1)) = (f 1) ∧ 
    differentiable_on ℝ f (set.Ioo (-1) 1)) := 
by 
  -- Verify that the function meets continuity criteria
  have h_cont : continuous_on f (set.Icc (-1) 1) := sorry,

  -- Verify that the function has equal values at the endpoints
  have h_endpoints : f (-1) = f (1) := by
    calc
      f (-1) = 0 : sorry
      f (1) = 0 : sorry,

  -- Check differentiability on open interval
  have h_diff : ¬ differentiable_on ℝ f (set.Ioo (-1) 1) := 
  by
    intros h_diff_at_0,
    -- f'(x) is undefined at x = 0
    calc 
      f' 0 = _ : sorry -- need definition of derivative,

  show 
    ¬(continuous_on f (set.Icc (-1) 1) ∧ (f (-1)) = (f 1) ∧ differentiable_on ℝ f (set.Ioo (-1) 1)) from
    by 
      intros H,
      cases H with hc Hf,
      cases Hf with he hd,
      -- contradicting differentiability assertion
      have h_contradiction : false := h_diff,
      contradiction,

end not_applicable_Rolles_theorem_l378_378323


namespace unique_prime_triple_l378_378922

/-- A prime is an integer greater than 1 whose only positive integer divisors are itself and 1. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

/-- Prove that the only triple of primes (p, q, r), such that p = q + 2 and q = r + 2 is (7, 5, 3). -/
theorem unique_prime_triple (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  (p = q + 2) ∧ (q = r + 2) → (p = 7 ∧ q = 5 ∧ r = 3) := by
  sorry

end unique_prime_triple_l378_378922


namespace min_rows_required_l378_378078

-- Condition definitions
def number_of_students : ℕ := 2016
def seats_per_row : ℕ := 168
def max_students_per_school : ℕ := 45

-- Theorem statement to prove the minimum number of rows
theorem min_rows_required : (∀ students : ℕ, students ≤ number_of_students → 
  ∀ max_per_school : ℕ, max_per_school = max_students_per_school → 
  ∀ seats : ℕ, seats = seats_per_row → 
  ∃ rows : ℕ, rows = 16) :=
begin
  sorry
end

end min_rows_required_l378_378078


namespace arithmetic_mean_eq_one_l378_378980

theorem arithmetic_mean_eq_one (x a : ℝ) (hx : x ≠ 0) : 
  (1 / 2) * ((x^2 + a^2) / x^2 + (x^2 - a^2) / x^2) = 1 :=
by
  have h1 : (x^2 + a^2) / x^2 = 1 + (a^2 / x^2), sorry
  have h2 : (x^2 - a^2) / x^2 = 1 - (a^2 / x^2), sorry
  calc
    (1 / 2) * ((x^2 + a^2) / x^2 + (x^2 - a^2) / x^2)
        = (1 / 2) * ((1 + (a^2 / x^2)) + (1 - (a^2 / x^2))) : by rw [h1, h2]
    ... = (1 / 2) * (2 : ℝ) : by linarith
    ... = 1 : by norm_num

end arithmetic_mean_eq_one_l378_378980


namespace constant_term_in_expansion_l378_378664

theorem constant_term_in_expansion (n : ℕ) 
  (h : (2 : ℕ)^n = 64) :
  n = 6 ∧ (∃ c : ℝ, c = 60 ∧ (∀ x : ℝ, x ≠ 0 → 
  ∃ r : ℕ, c = (Polynomial.C (Real.ofRat (Nat.choose 6 r) * 2^(6-r) * (-1 : ℝ)^r / (x^(3/2 * r - 6)))))) := 
by
  sorry

end constant_term_in_expansion_l378_378664


namespace sufficient_not_necessary_condition_l378_378039

-- Define the condition on a
def condition (a : ℝ) : Prop := a > 0

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) : Prop := a^2 + a ≥ 0

-- The proof statement that "a > 0" is a sufficient but not necessary condition for "a^2 + a ≥ 0"
theorem sufficient_not_necessary_condition (a : ℝ) : condition a → quadratic_inequality a :=
by
    intro ha
    -- [The remaining part of the proof is skipped.]
    sorry

end sufficient_not_necessary_condition_l378_378039


namespace log_base_one_four_of_sixteen_l378_378244

theorem log_base_one_four_of_sixteen : log (1 / 4) 16 = -2 := by
  sorry

end log_base_one_four_of_sixteen_l378_378244


namespace log_base_one_four_of_sixteen_l378_378240

theorem log_base_one_four_of_sixteen : log (1 / 4) 16 = -2 := by
  sorry

end log_base_one_four_of_sixteen_l378_378240


namespace reggie_games_lost_l378_378446

-- Given conditions:
def initial_marbles : ℕ := 100
def marbles_per_game : ℕ := 10
def games_played : ℕ := 9
def marbles_after_games : ℕ := 90

-- The statement to prove:
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / marbles_per_game = 1 := 
sorry

end reggie_games_lost_l378_378446


namespace jane_apples_l378_378745

theorem jane_apples :
  ∃ J : ℕ, 
  let average := (20 + J + 40) / 3 in
  20 = 2 * average ∧ J = 30 :=
by
  sorry

end jane_apples_l378_378745


namespace ellipse_problem_l378_378670

noncomputable def ellipse_equation (E : ℝ × ℝ) (e a b : ℝ) (h_ab : a > b) (h_b0 : b > 0) (h_e : e = (Real.sqrt (a^2 - b^2)) / a) := 
  ∀ x y : ℝ, (x, y) = E → ((x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def max_dist_AB (l : ℝ × ℝ → Prop) (circ : ℝ × ℝ → Prop) (ellipse : ℝ × ℝ → Prop) :=
  ∀ A B : ℝ × ℝ, (l B) ∧ circ B ∧ ellipse B → 
  ∃ k, (l B) ∧ (B.2 = k * B.1 + (Circ[ℝ, B, 1])) → 
  |A - B| ≤ 2

theorem ellipse_problem :
  (∃ a b : ℝ, 
    (a > b ∧ b > 0) ∧ 
    (E = (Real.sqrt 3, 1 / 2)) ∧ 
    (a = 2 ∧ b = 1) ∧ 
    (∀ x y : ℝ, ((x^2 / 4 + y^2 = 1)) ∧ 
    (max_dist_AB (λ p, p.1 = 1) (λ p, p.1^2 + p.2^2 = 1) (λ p, p.1^2 / 4 + p.2^2 = 1) = 2))
  sorry

end ellipse_problem_l378_378670


namespace log_base_one_fourth_of_sixteen_l378_378267

theorem log_base_one_fourth_of_sixteen : log (1/4) 16 = -2 :=  sorry

end log_base_one_fourth_of_sixteen_l378_378267


namespace side_length_equilateral_triangle_l378_378928

theorem side_length_equilateral_triangle (O A B C : Point) (r : ℝ) (s : ℝ)
  (h_circle_area : π * r^2 = 169 * π)
  (h_eq_triangle : equilateral_triangle A B C)
  (h_chord : chord O A C s)
  (h_OB : dist O B = 5 * real.sqrt 3)
  (h_O_outside : O ∉ triangle A B C) :
  s = 5 * real.sqrt 3 := 
sorry

end side_length_equilateral_triangle_l378_378928


namespace max_distinct_tangent_counts_l378_378427

-- Define the types and conditions for our circles and tangents
structure Circle where
  radius : ℝ

def circle1 : Circle := { radius := 3 }
def circle2 : Circle := { radius := 4 }

-- Define the statement to be proved
theorem max_distinct_tangent_counts :
  ∃ (k : ℕ), k = 5 :=
sorry

end max_distinct_tangent_counts_l378_378427


namespace range_of_m_intersection_l378_378710

noncomputable def f (x m : ℝ) : ℝ := (1/x) - (m/(x^2)) - (x/3)

theorem range_of_m_intersection (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ m ∈ (Set.Iic 0 ∪ {2/3}) :=
sorry

end range_of_m_intersection_l378_378710


namespace AI_in_radius_l378_378072

theorem AI_in_radius :
  ∀ (A B C I : Point) (AB AC BC : ℝ),
    is_isosceles_right_triangle A B C ∧
    side_length AB = 6 * Real.sqrt 2 ∧
    angle_A_is_right ∧
    is_incenter I A B C →
    distance A I = 6 - 3 * Real.sqrt 2 := 
by
  intros A B C I AB AC BC H,
  sorry

end AI_in_radius_l378_378072


namespace correct_option_l378_378314

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg_real (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ > f x₂

theorem correct_option (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_decr : is_decreasing_on_nonneg_real f) :
  f 2 < f (-1) ∧ f (-1) < f 0 :=
by
  sorry

end correct_option_l378_378314


namespace quadratic_expression_l378_378349

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def has_roots_and_max (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (f (-2) = 0) ∧ (f 4 = 0) ∧ (∃ v_max, (∀ x, f x ≤ v_max) ∧ v_max = 9)

theorem quadratic_expression :
  ∃ a b c : ℝ, has_roots_and_max (quadratic_function a b c) a ∧
  (quadratic_function a b c = λ x, -x^2 + 2*x + 8) :=
sorry

end quadratic_expression_l378_378349


namespace area_enclosed_by_graph_l378_378861

noncomputable def enclosed_area (x y : ℝ) : ℝ := 
  if h : (|5 * x| + |3 * y| = 15) then
    30 -- The area enclosed by the graph
  else
    0 -- Default case for definition completeness

theorem area_enclosed_by_graph : ∀ (x y : ℝ), (|5 * x| + |3 * y| = 15) → enclosed_area x y = 30 :=
by
  sorry

end area_enclosed_by_graph_l378_378861


namespace smallest_base_10_integer_l378_378887

noncomputable def smallest_integer (a b: ℕ) (h₁: a > 2) (h₂: b > 2) (h₃: n = 2 * a + 1) (h₄: n = b + 2) : ℕ :=
  n

theorem smallest_base_10_integer : smallest_integer 3 5 (by decide) (by decide) (by decide) (by decide) = 7 :=
sorry

end smallest_base_10_integer_l378_378887


namespace original_prices_sum_l378_378911

def candy_new_price := 25 -- New price of a candy box after a 25% increase
def candy_increase := 0.25 -- Percentage increase in the price of candy box

def soda_new_price := 9 -- New price of a can of soda after a 50% increase
def soda_increase := 0.50 -- Percentage increase in the price of soda can

theorem original_prices_sum :
  let original_candy_price := candy_new_price / (1 + candy_increase),
      original_soda_price := soda_new_price / (1 + soda_increase)
  in original_candy_price + original_soda_price = 26 :=
by
  -- Proof omitted
  sorry

end original_prices_sum_l378_378911


namespace base6_sum_eq_10_l378_378400

theorem base6_sum_eq_10 
  (A B C : ℕ) 
  (hA : 0 < A ∧ A < 6) 
  (hB : 0 < B ∧ B < 6) 
  (hC : 0 < C ∧ C < 6)
  (distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (h_add : A*36 + B*6 + C + B*6 + C = A*36 + C*6 + A) :
  A + B + C = 10 := 
by
  sorry

end base6_sum_eq_10_l378_378400


namespace exponential_to_rectangular_l378_378205

theorem exponential_to_rectangular (z : ℂ) (h : z = 2 * complex.exp (15 * real.pi * complex.I / 4)) : 
  z = complex.sqrt 2 - complex.I * complex.sqrt 2 :=
by
  sorry

end exponential_to_rectangular_l378_378205


namespace complement_of_intersection_l378_378774

-- Declare the universal set U
def U : Set ℤ := {-1, 1, 2, 3}

-- Declare the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the given quadratic equation
def is_solution (x : ℤ) : Prop := x^2 - 2 * x - 3 = 0
def B : Set ℤ := {x : ℤ | is_solution x}

-- The main theorem to prove
theorem complement_of_intersection (A_inter_B_complement : Set ℤ) :
  A_inter_B_complement = {1, 2, 3} :=
by
  sorry

end complement_of_intersection_l378_378774


namespace tickets_not_went_to_concert_l378_378010

theorem tickets_not_went_to_concert :
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  remaining_after_start - (after_first_song + during_middle) = 20 := 
by
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  show remaining_after_start - (after_first_song + during_middle) = 20
  sorry

end tickets_not_went_to_concert_l378_378010


namespace average_couples_together_l378_378570

theorem average_couples_together {n : ℕ} (h : n > 0) :
  let total_arrangements := (2 * n - 1)!,
      couples_together_arrangements := 4 * n * (2 * n - 2)!,
      average_together := couples_together_arrangements / total_arrangements
  in average_together = (2 * n) / (2 * n - 1) :=
by
  sorry

end average_couples_together_l378_378570


namespace find_a2_in_geometric_sequence_l378_378733

theorem find_a2_in_geometric_sequence (q : ℕ) (S_3 : ℕ) (a_2 : ℕ) :
  q = 2 → S_3 = 34685 → a_2 = 9910 := 
by
  assume hq : q = 2
  assume hS3 : S_3 = 34685
  -- The proof will go here
  sorry

end find_a2_in_geometric_sequence_l378_378733


namespace range_of_theta_l378_378806

-- Define necessary vectors and their properties
variables (t t0 θ : Real)
          (OA OB : EuclideanSpace R^n)
          (OA_dot_OB : Real)
          [NormedAddTorsor2 EuclideanSpace Point R]
          (f : Real → Real)

-- Given conditions as assumptions
axiom h1 : ∥OA∥ = 2
axiom h2 : ∥OB∥ = 1
axiom h3 : OA • OB = 2 * cos θ
axiom h4 : ∃ t0, (0 < t0) ∧ (t0 < 1/5) ∧ (t0 = (1 + 2 * cos θ) / (5 + 4 * cos θ))
axiom h5 : ∀ t, f t = sqrt ((5 + 4 * cos θ) * t^2 - 2 * (1 + 2 * cos θ) * t + 1)
axiom h6 : ∀ t, has_min_on f {t0} {t in Ioi 0 ∩ Iio (1/5)}

-- The theorem to prove the range of θ
theorem range_of_theta : θ ∈ Ioo (π / 2) (2 * π / 3) :=
sorry

end range_of_theta_l378_378806


namespace circle_radius_l378_378481

theorem circle_radius (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y = 0) : ∃ r : ℝ, r = Real.sqrt 13 :=
by
  sorry

end circle_radius_l378_378481


namespace solution_l378_378538

-- Given conditions and definitions
def point_rectangular : ℝ × ℝ := (10, 3)
def point_polar : ℝ × ℝ := let r := (point_rectangular.1^2 + point_rectangular.2^2).sqrt in
                           let θ := real.arctan (point_rectangular.2 / point_rectangular.1) in
                           (r, θ)

-- Proof problem to show rectangular coordinates of (r^2, 2θ)
def proof_problem : Prop :=
  let (r, θ) := point_polar in
  let x := (r^2) * (real.cos (2 * θ)) in
  let y := (r^2) * (real.sin (2 * θ)) in
  (x = 91 ∧ y = 60)

theorem solution : proof_problem :=
  by
    sorry -- Proof should be done here

end solution_l378_378538


namespace minimum_rows_required_l378_378075

theorem minimum_rows_required (total_students : ℕ) (max_students_per_school : ℕ) (seats_per_row : ℕ) (num_schools : ℕ) 
    (h_total_students : total_students = 2016) 
    (h_max_students_per_school : max_students_per_school = 45) 
    (h_seats_per_row : seats_per_row = 168) 
    (h_num_schools : num_schools = 46) : 
    ∃ (min_rows : ℕ), min_rows = 16 := 
by 
  -- Proof omitted
  sorry

end minimum_rows_required_l378_378075


namespace geom_seq_sum_l378_378522

theorem geom_seq_sum (q : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 3)
  (h2 : a 1 + a 3 + a 5 = 21)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
  a 3 + a 5 + a 7 = 42 :=
sorry

end geom_seq_sum_l378_378522


namespace negation_proposition_l378_378827

theorem negation_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) := by
suffices : (∀ x : ℝ, P x) ↔ (¬ ∃ x : ℝ, ¬ P x), from
  ⟨λ h1 => by 
    rw [← this] at h1
    assumption,
   λ h2 => by 
     rw [← this]
     assumption⟩
sorry

example : (¬ ∀ x : ℝ, x ^ 2 - x + 1 < 0) ↔ (∃ x : ℝ, x ^ 2 - x + 1 ≥ 0) := by
  have P : ℝ → Prop := λ x => x ^ 2 - x + 1 < 0
  have h := negation_proposition P
  simp_rw [not_forall, not_lt, exists_prop]
  exact h

end negation_proposition_l378_378827


namespace smallest_base10_integer_l378_378891

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end smallest_base10_integer_l378_378891


namespace log_base_fraction_l378_378227

theorem log_base_fraction : ∀ (a b : ℝ) (x : ℝ), 16 = (4:ℝ)^2 ∧ (1 / 4:ℝ) = 4^(-1) → log (1 / 4) 16 = -2 :=
begin
  intros a b x h,
  -- Skipping the proof by adding sorry
  sorry,
end

end log_base_fraction_l378_378227


namespace solve_system_l378_378801

-- Definitions for the system of equations.
def system_valid (y : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

-- Main theorem to prove.
theorem solve_system (y : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) : 
  system_valid y x₁ x₂ x₃ x₄ x₅ →
  ((y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨ 
  (y = 2 → ∃ (t : ℝ), x₁ = t ∧ x₂ = t ∧ x₃ = t ∧ x₄ = t ∧ x₅ = t) ∨ 
  (y^2 + y - 1 = 0 → ∃ (u v : ℝ), 
    x₁ = u ∧ 
    x₅ = v ∧ 
    x₂ = y * u - v ∧ 
    x₃ = -y * (u + v) ∧ 
    x₄ = y * v - u ∧ 
    (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2))) :=
by
  intro h
  sorry

end solve_system_l378_378801


namespace speed_of_man_l378_378530

-- Conditions
def train_length : ℝ := 100 -- length of the train in meters
def crossing_time : ℝ := 5.999520038396929 -- time taken to cross the man in seconds
def speed_of_train_kmh : ℝ := 63 -- speed of the train in km/hr

-- Conversion of speed from km/hr to m/s
def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600)

-- Theorem stating the speed of the man
theorem speed_of_man (train_length crossing_time speed_of_train_ms : ℝ) : ℝ :=
  train_length = crossing_time * (speed_of_train_ms - speed_of_man) →
  speed_of_man = 0.831946

-- Proof step for the speed of the man
example : speed_of_man train_length crossing_time speed_of_train_ms :=
by {
  sorry
}

end speed_of_man_l378_378530


namespace min_value_frac_ineq_l378_378316

theorem min_value_frac_ineq (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) : 
  (9/m + 1/n) ≥ 16 :=
sorry

end min_value_frac_ineq_l378_378316


namespace tan_x_eq_1_max_f_x_l378_378690

-- Problem 1
theorem tan_x_eq_1 (x : ℝ) (h1 : (1/2) * Real.cos x * sqrt 3 + (sqrt 3 / 2) * Real.sin x * -1 = 0) : 
  Real.tan x = 1 :=
sorry

-- Problem 2
theorem max_f_x (x : ℝ) (h2 : x ∈ Set.Icc (0: ℝ) (Real.pi / 2))
  (f : ℝ → ℝ := λ x, 
    (1/2 * Real.cos x + sqrt 3) * sqrt 3 + (sqrt 3 / 2 * Real.sin x - 1) * -1) :
  ∃ x_max, x = 0 ∧ f x_max = 4 + sqrt 6 / 2 :=
sorry

end tan_x_eq_1_max_f_x_l378_378690


namespace smallest_base10_integer_l378_378890

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end smallest_base10_integer_l378_378890


namespace log_one_fourth_sixteen_l378_378245

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 := 
by
  let x := log (1 / 4) 16
  have h₁ : (1 / 4) ^ x = 16 := by simp [log_eq_iff]
  have h₂ : (4 ^ (-1)) ^ x = 16 := by rw [one_div, inv_pow]
  have h₃ : 4 ^ (-x) = 16 := by simp [pow_mul]
  have h₄ : 16 = 4 ^ 2 := by norm_num
  rw [h₄] at h₃
  have h₅ : -x = 2 := by exact pow_inj (lt_trans zero_lt_one (by norm_num)) zero_lt_four h₃
  have h₆ : x = -2 := by linarith
  exact h₆

end log_one_fourth_sixteen_l378_378245


namespace ellipse_standard_eq_line_eq_l378_378607

noncomputable section

-- Define the conditions
variables {a b c : ℝ}
variables (A : ℝ × ℝ) (e : ℝ)
variables (l : ℝ × ℝ → ℝ) (P Q N : ℝ × ℝ)

-- The given conditions
def ellipse (a b : ℝ) := λ (x y : ℝ), (x^2) / (a^2) + (y^2) / (b^2) = 1
def passes_through_A := A = (0, real.sqrt 2)
def eccentricity := e = real.sqrt 3 / 2
def line_passes_through := λ l, l (1, 0) = 0
def equilateral_triangle := λ (P Q N : ℝ × ℝ), (P = Q → False)
  ∧ (N.1 = 1) ∧ (dist P N = dist Q N) ∧ (dist P Q = dist P N) 

-- The proof statements
theorem ellipse_standard_eq 
  (h₁ : ellipse a b 0 (real.sqrt 2))
  (h₂ : eccentricity)
  (h₃ : real.lt b a)
  (h₄ : passes_through_A) :
  ellipse 2 real.sqrt 2 = λ x y, x^2 / 8 + y^2 / 2 = 1 :=
sorry

theorem line_eq 
  (h₅ : line_passes_through l)
  (h₆ : equilateral_triangle P Q N)
  (h₇ : ∃ t : ℝ, P = (t * y + 1, y))
  (h₈ : Q = (t * y + 1, y))
  (h₉ : N = (1, t + 3 * t / (t^2 + 4))) :
  (l = λ x y, x + real.sqrt 10 * y - 1 = 0) ∨
  (l = λ x y, x - real.sqrt 10 * y - 1 = 0) :=
sorry

end ellipse_standard_eq_line_eq_l378_378607


namespace product_of_solutions_product_of_all_solutions_main_theorem_l378_378281

theorem product_of_solutions (t : ℝ) (h : t^2 - 64 = 0) : t = 8 ∨ t = -8 :=
begin
  -- Proof will go here
  sorry
end

theorem product_of_all_solutions : (8 : ℝ) * (-8) = -64 :=
begin
  -- Proof will go here
  sorry
end

theorem main_theorem : 
  (∀ t : ℝ, t^2 - 64 = 0 → t = 8 ∨ t = -8) → (∏ t in ({8, -8} : finset ℝ), t) = -64 :=
begin
  intros h,
  -- Proof will go here
  sorry
end

end product_of_solutions_product_of_all_solutions_main_theorem_l378_378281


namespace italian_dressing_solution_l378_378927

theorem italian_dressing_solution :
  ∃ (x y : ℝ), x + y = 320 ∧ 0.08 * x + 0.13 * y = 35.2 ∧ x = 128 ∧ y = 192 :=
by
  use 128
  use 192
  simp
  split
  · norm_num
  split
  · norm_num
  split
  · refl
  · refl

end italian_dressing_solution_l378_378927


namespace mary_score_unique_l378_378364

theorem mary_score_unique (c w : ℕ) (s : ℕ) (h1 : s = 30 + 4 * c - 2 * w) (h2 : s > 100)
    (h_unique : ∀ (c1 w1 : ℕ), (s = 30 + 4 * c1 - 2 * w1) → c1 = c ∧ w1 = w) :
    s = 116 :=
begin
    sorry
end

end mary_score_unique_l378_378364


namespace regular_2015gon_with_64_marked_vertices_has_trapezoid_l378_378366

-- Assume a regular 2015-sided polygon with vertices A1, A2, ..., A2015
-- and that 64 vertices are marked.

-- Defining the property that for any 4 distinct marked vertices, they can form a trapezoid.
def exists_trapezoid (polygon : Fin (2015 + 1) → Prop) (marked_vertices : Finset (Fin (2015 + 1))) :=
  ∃ (v w x y : Fin (2015 + 1)), v ≠ w ∧ v ≠ x ∧ v ≠ y ∧ w ≠ x ∧ w ≠ y ∧ x ≠ y ∧
  (v ∈ marked_vertices ∧ w ∈ marked_vertices ∧ x ∈ marked_vertices ∧ y ∈ marked_vertices) ∧
  (are_collinear v w x y → False) ∧ -- Ensuring they form a trapezoid
  (¬are_parallelogram v w x y)       -- Eliminating the parallelogram possibility

theorem regular_2015gon_with_64_marked_vertices_has_trapezoid 
  (polygon : Fin (2015 + 1) → Prop) (marked_vertices : Finset (Fin (2015 + 1))) 
  (h_regular : ∀ i, polygon i) (h_marked : marked_vertices.card = 64) :
  exists_trapezoid polygon marked_vertices :=
sorry

end regular_2015gon_with_64_marked_vertices_has_trapezoid_l378_378366


namespace no_pairing_possible_l378_378742

theorem no_pairing_possible (n : ℕ) (h : n = 800) :
  ¬ ∃ (pairs : Finset (Finset ℕ)), 
    (∀ x ∈ pairs, ∃ a b, a + b = 6 * d ∧ d ∈ ℕ) ∧ 
    (pairs.card = 400) ∧ 
    (∀ x ∈ Finset.range (n+1), ∃ y ∈ pairs, x ∈ y) :=
sorry

end no_pairing_possible_l378_378742


namespace volume_of_inscribed_sphere_l378_378661

theorem volume_of_inscribed_sphere
  (a h r R : ℝ)
  (tetrahedron_volume : ℝ)
  (inscription : ℝ)
  (angle_APO : ℝ)
  (sphere_volume : ℝ) :
  (tetrahedron_volume = (9 * sqrt 3) / 4) →
  (angle_APO = 30) →
  (R ^ 2 = (h - R) ^ 2 + r ^ 2) →
  (sphere_volume = (4 / 3) * π * R ^ 3) →
  sphere_volume = (32 / 3) * π :=
by
  sorry

end volume_of_inscribed_sphere_l378_378661


namespace log_one_fourth_sixteen_l378_378250

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 := 
by
  let x := log (1 / 4) 16
  have h₁ : (1 / 4) ^ x = 16 := by simp [log_eq_iff]
  have h₂ : (4 ^ (-1)) ^ x = 16 := by rw [one_div, inv_pow]
  have h₃ : 4 ^ (-x) = 16 := by simp [pow_mul]
  have h₄ : 16 = 4 ^ 2 := by norm_num
  rw [h₄] at h₃
  have h₅ : -x = 2 := by exact pow_inj (lt_trans zero_lt_one (by norm_num)) zero_lt_four h₃
  have h₆ : x = -2 := by linarith
  exact h₆

end log_one_fourth_sixteen_l378_378250


namespace periodic_sequence_a_2008_l378_378738

def sequence : ℕ → ℕ
| 0       := 2
| 1       := 7
| (n + 2) := (sequence n * sequence (n + 1)) % 10

theorem periodic_sequence_a_2008 :
  sequence 2008 = 8 :=
sorry

end periodic_sequence_a_2008_l378_378738


namespace conversion_problems_l378_378128

def decimal_to_binary (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 2 + 10 * decimal_to_binary (n / 2)

def largest_two_digit_octal : ℕ := 77

theorem conversion_problems :
  decimal_to_binary 111 = 1101111 ∧ (7 * 8 + 7) = 63 :=
by
  sorry

end conversion_problems_l378_378128


namespace problem_statement_l378_378783

-- Define rational number representations for points A, B, and C
def a : ℚ := (-4)^2 - 8

-- Define that B and C are opposites
def are_opposites (b c : ℚ) : Prop := b = -c

-- Define the distance condition
def distance_is_three (a c : ℚ) : Prop := |c - a| = 3

-- Main theorem statement
theorem problem_statement :
  (∃ b c : ℚ, are_opposites b c ∧ distance_is_three a c ∧ -a^2 + b - c = -74) ∨
  (∃ b c : ℚ, are_opposites b c ∧ distance_is_three a c ∧ -a^2 + b - c = -86) :=
sorry

end problem_statement_l378_378783


namespace sum_of_intercepts_of_line_l378_378069

theorem sum_of_intercepts_of_line (x y : ℝ) (h : 2 * x - y + 4 = 0) : 
  x = -2 ∧ y = 4 → x + y = 2 :=
by
  intro hxy
  cases hxy with hx hy
  rw [hx, hy]
  norm_num

end sum_of_intercepts_of_line_l378_378069


namespace smallest_base10_integer_l378_378892

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end smallest_base10_integer_l378_378892


namespace actual_amount_is_17_l378_378138

noncomputable def actual_amount (x : ℕ) : Prop :=
∃ (a : ℕ), (a = 9 * x ∨ a = 0.9 * x) ∧ a = 153

theorem actual_amount_is_17 : ∃ x, actual_amount x ∧ x = 17 :=
by
  sorry

end actual_amount_is_17_l378_378138


namespace smallest_n_exists_l378_378627

theorem smallest_n_exists (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : 8 / 15 < n / (n + k)) (h4 : n / (n + k) < 7 / 13) : 
  n = 15 :=
  sorry

end smallest_n_exists_l378_378627


namespace weight_of_6m_rod_l378_378962

theorem weight_of_6m_rod (r ρ : ℝ) (h₁ : 11.25 > 0) (h₂ : 6 > 0) (h₃ : 0 < r) (h₄ : 42.75 = π * r^2 * 11.25 * ρ) : 
  (π * r^2 * 6 * (42.75 / (π * r^2 * 11.25))) = 22.8 :=
by
  sorry

end weight_of_6m_rod_l378_378962


namespace basketball_player_probability_l378_378533

theorem basketball_player_probability :
  (let p_success := 3 / 5
   let p_failure := 1 - p_success
   let p_all_failure := p_failure ^ 3
   let p_at_least_one_success := 1 - p_all_failure
   in p_at_least_one_success = 0.936) :=
by
  sorry

end basketball_player_probability_l378_378533


namespace smallest_base_10_integer_l378_378886

noncomputable def smallest_integer (a b: ℕ) (h₁: a > 2) (h₂: b > 2) (h₃: n = 2 * a + 1) (h₄: n = b + 2) : ℕ :=
  n

theorem smallest_base_10_integer : smallest_integer 3 5 (by decide) (by decide) (by decide) (by decide) = 7 :=
sorry

end smallest_base_10_integer_l378_378886


namespace five_points_max_min_ratio_l378_378301

-- Define the problem in Lean
theorem five_points_max_min_ratio (pts : Fin 5 → ℝ × ℝ) (λ : ℝ) :
  (λ = (max (dist pts) / min (dist pts))) → λ ≥ 2 * sin 54 :=
sorry

end five_points_max_min_ratio_l378_378301


namespace even_iff_zero_mod_two_odd_iff_one_mod_two_even_iff_2018_mod_two_l378_378505

theorem even_iff_zero_mod_two (n : ℤ) : n % 2 = 0 ↔ ∃ k : ℤ, n = 2 * k := sorry

theorem odd_iff_one_mod_two (n : ℤ) : n % 2 = 1 ↔ ∃ k : ℤ, n = 2 * k + 1 := sorry

theorem even_iff_2018_mod_two (n : ℤ) : n % 2 = 2018 % 2 ↔ ∃ k : ℤ, n = 2 * k := 
by {
  have h1 : 2018 % 2 = 0 := by norm_num,
  rw h1,
  exact even_iff_zero_mod_two n,
}

end even_iff_zero_mod_two_odd_iff_one_mod_two_even_iff_2018_mod_two_l378_378505


namespace ellipse_equation_is_correct_l378_378662

theorem ellipse_equation_is_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let F := (3 / 2, 0)
  let midpoint_AB := (1, -1 / 2)
  let angle_AB := (Real.pi / 4)
  (a^2 = 2 * b^2) ∧ (a^2 - b^2 = 9 / 4) →
  (\frac{x^{2}}{a^{2}} + \frac{y^{2}}{b^{2}} = 1) = \frac{2x^{2}}{9} + \frac{4y^{2}}{9} = 1 :=
by
  sorry

end ellipse_equation_is_correct_l378_378662


namespace prod2025_min_sum_l378_378695

theorem prod2025_min_sum : ∃ (a b : ℕ), a * b = 2025 ∧ a > 0 ∧ b > 0 ∧ (∀ (x y : ℕ), x * y = 2025 → x > 0 → y > 0 → x + y ≥ a + b) ∧ a + b = 90 :=
sorry

end prod2025_min_sum_l378_378695


namespace ratio_of_border_to_tile_l378_378147

theorem ratio_of_border_to_tile (n p b : ℝ) (h₁ : n = 15) 
                               (h₂ : 225 = n^2) 
                               (h₃ : ∃ b, (225 * p ^ 2) = 0.49 * ((15 * p + 30 * b) ^ 2)) :
  (b / p = 4 / 7) :=
by 
  let np := 15 * p
  let nb := 2 * 15 * b
  have A := (np + nb)^2
  simp at A
  let left_side := 225 * p^2
  let right_side := (0.49 * A)
  simp at right_side
  let fraction := left_side / right_side
  simp at fraction
  let answer := fraction
  simp [answer]
  sorry

end ratio_of_border_to_tile_l378_378147


namespace job_completion_time_l378_378537

theorem job_completion_time (h1 : ∀ {a d : ℝ}, 4 * (1/a + 1/d) = 1)
                             (h2 : ∀ d : ℝ, d = 11.999999999999998) :
                             (∀ a : ℝ, a = 6) :=
by
  sorry

end job_completion_time_l378_378537


namespace cos_equation_proof_l378_378125

theorem cos_equation_proof :
  4.74 * ((cos 64 * cos 4 - cos 86 * cos 26) / (cos 71 * cos 41 - cos 49 * cos 19)) = -1 :=
by
  -- Leaving proof as placeholder
  sorry

end cos_equation_proof_l378_378125


namespace lemonade_total_l378_378219

-- Define the conditions
def ed_lemonade (x : ℝ) := x
def ann_lemonade (x : ℝ) := (3 / 2) * x

def ed_consumed (x : ℝ) := ed_lemonade x * (3 / 4)
def ann_consumed (x : ℝ) := ann_lemonade x * (3 / 4)

def ed_remaining (x : ℝ) := ed_lemonade x / 4
def ann_remaining (x : ℝ) := ann_lemonade x / 4

def ann_to_ed (x : ℝ) := ann_remaining x / 3 + 2

-- Calculate total consumption
def ed_total_consumed (x : ℝ) := ed_consumed x + ann_to_ed x
def ann_total_consumed (x : ℝ) := ann_consumed x - ann_to_ed x

-- State the theorem
theorem lemonade_total (x : ℝ) (h : ed_total_consumed x = ann_total_consumed x) :
  ed_lemonade x + ann_lemonade x = 40 :=
by
  sorry

end lemonade_total_l378_378219


namespace find_height_of_tank_A_l378_378463

noncomputable def height_of_tank_A (h : ℝ) : Prop :=
  ∃ (r_A r_B : ℝ), 
    let V_A := π * r_A^2 * h in
    let V_B := π * r_B^2 * 7 in
    2 * π * r_A = 7 ∧
    2 * π * r_B = 10 ∧
    V_A = 0.7 * V_B

theorem find_height_of_tank_A : height_of_tank_A 10 :=
sorry

end find_height_of_tank_A_l378_378463


namespace inequality_l378_378412

variables {R : Type*} [Real R]

open scoped BigOperators

def a_n (n : ℕ) (a : ℕ → ℝ) : ℝ := (1 / n) * ∑ i in finset.range n, a i
def b_n (n : ℕ) (b : ℕ → ℝ) : ℝ := (1 / n) * ∑ i in finset.range n, b i

def c_i (n : ℕ) (a b : ℕ → ℝ) (i : ℕ) : ℝ :=
  |a_n n a * b i + a i * b_n n b - a i * b i|

theorem inequality (n : ℕ) (a b : ℕ → ℝ) :
  (∑ i in finset.range n, c_i n a b i) ^ 2 ≤ (∑ i in finset.range n, (a i)^2) * (∑ i in finset.range n, (b i)^2) :=
by
  sorry

end inequality_l378_378412


namespace eliminate_denominators_l378_378104

theorem eliminate_denominators (x : ℝ) :
  (6 : ℝ) * ((x - 1) / 3) = (6 : ℝ) * (4 - (2 * x + 1) / 2) ↔ 2 * (x - 1) = 24 - 3 * (2 * x + 1) :=
by
  intros
  sorry

end eliminate_denominators_l378_378104


namespace point_D_not_on_graph_l378_378514

-- Define the function y = 2x / (x + 2)
def f (x : ℝ) : ℝ := 2 * x / (x + 2)

-- Define points
def p_A := (0, 0) : ℝ × ℝ
def p_B := (-1, -2) : ℝ × ℝ
def p_C := (1, 2 / 3) : ℝ × ℝ
def p_D := (-2, 1) : ℝ × ℝ
def p_E := (-3, 3) : ℝ × ℝ

-- Prove that point D is not on the graph
theorem point_D_not_on_graph : p_D.2 ≠ f p_D.1 :=
by {
  -- Proof steps will go here
  sorry
}


end point_D_not_on_graph_l378_378514


namespace sum_of_possible_values_of_y_l378_378203

open Real

def satisfies_conditions (l : List ℝ) (y : ℝ) : Prop :=
  l = [14, 3, 6, 3, 7, 3, y, 10] ∧
  let mean := (14 + 3 + 6 + 3 + 7 + 3 + y + 10) / 8 in
  let mode := 3 in
  let median := if y <= 6 then 6 else if y <= 10 then 7 else y in
  let ap := [3, median, mean] in
  (median - 3 = mean - median ∧ median - 3 = 2)

theorem sum_of_possible_values_of_y : ∃ y : ℝ, satisfies_conditions [14,3,6,3,7,3,y,10] y ∧ y = 70 / 15 :=
by
  sorry

end sum_of_possible_values_of_y_l378_378203


namespace radius_of_circle_is_seven_l378_378711

def diameter := 14

def radius (d : ℝ) := d / 2

theorem radius_of_circle_is_seven : radius diameter = 7 := by
  sorry

end radius_of_circle_is_seven_l378_378711


namespace geometric_sequence_relation_l378_378773

theorem geometric_sequence_relation (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (n : ℕ) 
  (h_geo : ∀ k, a (k+1) = q * a k)
  (h_S : ∀ k, S k = (finset.range k).sum (λ i, a (i+1))) :
  let x := (S n) ^ 2 + (S (2 * n)) ^ 2,
      y := (S n) * (S (2 * n) + S (3 * n)) in
  x = y :=
by
  sorry

end geometric_sequence_relation_l378_378773


namespace intersection_unique_l378_378200

theorem intersection_unique :
  ∀ x : ℝ, x > 0 → 3 * Real.log x = Real.log (4 * x) → x = 2 :=
by
  assume x hx h
  sorry 

end intersection_unique_l378_378200


namespace log_base_fraction_l378_378223

theorem log_base_fraction : ∀ (a b : ℝ) (x : ℝ), 16 = (4:ℝ)^2 ∧ (1 / 4:ℝ) = 4^(-1) → log (1 / 4) 16 = -2 :=
begin
  intros a b x h,
  -- Skipping the proof by adding sorry
  sorry,
end

end log_base_fraction_l378_378223


namespace relative_error_comparison_l378_378970

theorem relative_error_comparison :
  let error1 := 0.05
  let length1 := 25
  let error2 := 0.25
  let length2 := 125
  (error1 / length1) = (error2 / length2) :=
by
  sorry

end relative_error_comparison_l378_378970


namespace mr_deane_total_expense_l378_378055

-- Given conditions
def oil_price_today : ℝ := 1.4
def oil_price_rollback : ℝ := 0.4
def price_decrease_per_day : ℝ := 0.1
def liters_today : ℝ := 10
def liters_friday : ℝ := 25
def distance_driven_before_friday : ℝ := 200
def car_fuel_efficiency : ℝ := 8
def total_trip_distance : ℝ := 320

-- Definitions derived from conditions
def friday_price := oil_price_today - (price_decrease_per_day * 4)
def monday_cost := liters_today * oil_price_today
def friday_cost := liters_friday * friday_price
def total_cost_of_35_liters := monday_cost + friday_cost

def remaining_distance := total_trip_distance - distance_driven_before_friday
def additional_liters_needed := remaining_distance / car_fuel_efficiency
def cost_for_additional_liters := additional_liters_needed * friday_price
def total_expense := total_cost_of_35_liters + cost_for_additional_liters

theorem mr_deane_total_expense : total_expense = 54 := by
  sorry

end mr_deane_total_expense_l378_378055


namespace maximize_expression_l378_378674

-- Define the function expression
def expression (a b c d : ℕ) : ℕ := d * b ^ c - a

-- Define the set of the variables available
def vars := {0, 1, 3, 4}

-- Define the condition that the variables are distinct and within the set 
def valid_assignments (a b c d : ℕ) : Prop :=
  a ∈ vars ∧ b ∈ vars ∧ c ∈ vars ∧ d ∈ vars ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Formulate the theorem
theorem maximize_expression :
  ∃ a b c d, valid_assignments a b c d ∧ expression a b c d = 12 :=
by
  sorry

end maximize_expression_l378_378674


namespace find_b_l378_378712

theorem find_b (A B C : ℝ) (a b c : ℝ)
  (h1 : Real.tan A = 1 / 3)
  (h2 : Real.tan B = 1 / 2)
  (h3 : a = 1)
  (h4 : A + B + C = π) -- This condition is added because angles in a triangle sum up to π.
  : b = Real.sqrt 2 :=
by
  sorry

end find_b_l378_378712


namespace scout_earnings_weekend_l378_378021

-- Define the conditions
def base_pay_per_hour : ℝ := 10.00
def saturday_hours : ℝ := 6
def saturday_customers : ℝ := 5
def saturday_tip_per_customer : ℝ := 5.00
def sunday_hours : ℝ := 8
def sunday_customers_with_3_tip : ℝ := 5
def sunday_customers_with_7_tip : ℝ := 5
def sunday_tip_3_per_customer : ℝ := 3.00
def sunday_tip_7_per_customer : ℝ := 7.00
def overtime_multiplier : ℝ := 1.5

-- Statement to prove earnings for the weekend is $255.00
theorem scout_earnings_weekend : 
  (base_pay_per_hour * saturday_hours + saturday_customers * saturday_tip_per_customer) +
  (base_pay_per_hour * overtime_multiplier * sunday_hours + 
   sunday_customers_with_3_tip * sunday_tip_3_per_customer +
   sunday_customers_with_7_tip * sunday_tip_7_per_customer) = 255 :=
by
  sorry

end scout_earnings_weekend_l378_378021


namespace green_light_probability_l378_378901

def red_duration : ℕ := 30
def green_duration : ℕ := 25
def yellow_duration : ℕ := 5

def total_cycle : ℕ := red_duration + green_duration + yellow_duration
def green_probability : ℚ := green_duration / total_cycle

theorem green_light_probability :
  green_probability = 5 / 12 := by
  sorry

end green_light_probability_l378_378901


namespace original_chipmunk_families_l378_378608

theorem original_chipmunk_families (families_left : ℕ) (families_went_away : ℕ) : 
  families_left = 21 ∧ families_went_away = 65 → (families_left + families_went_away = 86) :=
begin
  sorry
end

end original_chipmunk_families_l378_378608


namespace trigonometric_identity_l378_378636

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by 
  sorry

end trigonometric_identity_l378_378636


namespace remaining_surface_area_correct_l378_378606

noncomputable def remaining_surface_area (a : ℕ) (c : ℕ) : ℕ :=
  let original_surface_area := 6 * a^2
  let corner_cube_area := 3 * c^2
  let net_change := corner_cube_area - corner_cube_area
  original_surface_area + 8 * net_change 

theorem remaining_surface_area_correct :
  remaining_surface_area 4 1 = 96 := by
  sorry

end remaining_surface_area_correct_l378_378606


namespace smallest_base_10_integer_exists_l378_378878

theorem smallest_base_10_integer_exists :
  ∃ (x a b : ℕ), (a > 2) ∧ (b > 2) ∧ (x = 2 * a + 1) ∧ (x = b + 2) ∧ (x = 7) :=
by
  sorry

end smallest_base_10_integer_exists_l378_378878


namespace total_selling_price_correct_l378_378150

def cost_price_1 := 750
def cost_price_2 := 1200
def cost_price_3 := 500

def loss_percent_1 := 10
def loss_percent_2 := 15
def loss_percent_3 := 5

noncomputable def selling_price_1 := cost_price_1 - ((loss_percent_1 / 100) * cost_price_1)
noncomputable def selling_price_2 := cost_price_2 - ((loss_percent_2 / 100) * cost_price_2)
noncomputable def selling_price_3 := cost_price_3 - ((loss_percent_3 / 100) * cost_price_3)

noncomputable def total_selling_price := selling_price_1 + selling_price_2 + selling_price_3

theorem total_selling_price_correct : total_selling_price = 2170 := by
  sorry

end total_selling_price_correct_l378_378150


namespace carousel_ticket_cost_l378_378017

theorem carousel_ticket_cost :
  ∃ (x : ℕ), 
  (2 * 5) + (3 * x) = 19 ∧ x = 3 :=
by
  sorry

end carousel_ticket_cost_l378_378017


namespace problem_statement_l378_378344

variable {x y z : ℝ}

-- Lean 4 statement of the problem
theorem problem_statement (h₀ : 0 ≤ x) (h₁ : x ≤ 1) (h₂ : 0 ≤ y) (h₃ : y ≤ 1) (h₄ : 0 ≤ z) (h₅ : z ≤ 1) :
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end problem_statement_l378_378344


namespace at_least_two_of_three_equations_have_solutions_l378_378652

theorem at_least_two_of_three_equations_have_solutions
  (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ x : ℝ, (x - a) * (x - b) = x - c ∨ (x - b) * (x - c) = x - a ∨ (x - c) * (x - a) = x - b := 
sorry

end at_least_two_of_three_equations_have_solutions_l378_378652


namespace log_one_fourth_sixteen_l378_378248

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 := 
by
  let x := log (1 / 4) 16
  have h₁ : (1 / 4) ^ x = 16 := by simp [log_eq_iff]
  have h₂ : (4 ^ (-1)) ^ x = 16 := by rw [one_div, inv_pow]
  have h₃ : 4 ^ (-x) = 16 := by simp [pow_mul]
  have h₄ : 16 = 4 ^ 2 := by norm_num
  rw [h₄] at h₃
  have h₅ : -x = 2 := by exact pow_inj (lt_trans zero_lt_one (by norm_num)) zero_lt_four h₃
  have h₆ : x = -2 := by linarith
  exact h₆

end log_one_fourth_sixteen_l378_378248


namespace find_q_l378_378193

-- Definitions based on the conditions provided in the problem.
def geom_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def geom_progression (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_seq (S1 S2 S3 : ℝ) : Prop :=
  2 * S2 = S1 + S3

-- The theorem based on the information provided and the solution derived.
theorem find_q (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h_geom : geom_progression a q)
  (h_arith : arithmetic_seq (geom_sum a (n+1)) (geom_sum a n) (geom_sum a (n+2))) :
  q = -2 :=
sorry

end find_q_l378_378193


namespace find_fourth_day_temp_l378_378840

def temp1 : ℤ := -36
def temp2 : ℤ := -15
def temp3 : ℤ := -10
def average_temp : ℚ := -12

theorem find_fourth_day_temp (temp1 temp2 temp3 : ℤ) (average_temp : ℚ) :
  temp1 = -36 → 
  temp2 = -15 →
  temp3 = -10 →
  average_temp = -12 →
  (4 * average_temp : ℚ) = (temp1 + temp2 + temp3 + (13 : ℤ)) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_cast
  ring
  done

end find_fourth_day_temp_l378_378840


namespace length_of_MN_l378_378731

noncomputable def curve_eq (α : ℝ) : ℝ × ℝ := (2 * Real.cos α + 1, 2 * Real.sin α)

noncomputable def line_eq (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem length_of_MN : ∀ (M N : ℝ × ℝ), 
  M ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2)^2 = 4} ∧
  N ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2)^2 = 4} ∧
  M ∈ {p : ℝ × ℝ | p.1 + p.2 = 2} ∧
  N ∈ {p : ℝ × ℝ | p.1 + p.2 = 2} →
  dist M N = Real.sqrt 14 :=
by
  sorry

end length_of_MN_l378_378731


namespace natural_numbers_divisors_l378_378620

theorem natural_numbers_divisors (n : ℕ) : 
  n + 1 ∣ n^2 + 1 → n = 0 ∨ n = 1 :=
by
  intro h
  sorry

end natural_numbers_divisors_l378_378620


namespace probability_of_green_light_l378_378903

theorem probability_of_green_light (red_time green_time yellow_time : ℕ) (h_red : red_time = 30) (h_green : green_time = 25) (h_yellow : yellow_time = 5) :
  (green_time.toRat / (red_time + green_time + yellow_time).toRat) = (5 / 12 : ℚ) :=
by
  sorry

end probability_of_green_light_l378_378903


namespace prime_integer_root_p_existence_l378_378635

def is_integer_root (p : ℕ) (Q : ℕ → ℤ) (x : ℕ) : Prop :=
  1 + p + ∏ i in finset.range (2 * p - 2), Q (x ^ (i + 1)) = 0

def is_polynomial (Q : ℕ → ℤ) : Prop :=
  ∀ x ∈ ℤ, Q x ∈ ℤ

theorem prime_integer_root_p_existence (p : ℕ) (Q : ℕ → ℤ):
  prime p → is_polynomial Q → (∃ x : ℕ, is_integer_root p Q x) ↔ p = 2 :=
sorry

end prime_integer_root_p_existence_l378_378635


namespace part_a_2001_good_part_a_3001_not_good_part_b_product_of_good_is_good_part_c_product_good_then_one_good_l378_378154

-- Definition of a good number
def is_good (n : ℕ) : Prop := (n % 6 = 3)

-- Lean 4 statements

-- 1. 2001 is good
theorem part_a_2001_good : is_good 2001 :=
by sorry

-- 2. 3001 isn't good
theorem part_a_3001_not_good : ¬ is_good 3001 :=
by sorry

-- 3. The product of two good numbers is a good number
theorem part_b_product_of_good_is_good (x y : ℕ) (hx : is_good x) (hy : is_good y) : is_good (x * y) :=
by sorry

-- 4. If the product of two numbers is good, then at least one of the numbers is good
theorem part_c_product_good_then_one_good (x y : ℕ) (hxy : is_good (x * y)) : is_good x ∨ is_good y :=
by sorry

end part_a_2001_good_part_a_3001_not_good_part_b_product_of_good_is_good_part_c_product_good_then_one_good_l378_378154


namespace geometric_sequence_find_T_l378_378297

variable {a : ℕ → ℝ} -- Sequence {a_n}
variable {S : ℕ → ℝ} -- Sum of first n terms of the sequence {a_n}
variable {T : ℕ → ℝ} -- Sum of first n terms of the sequence {S_n}

-- Given condition
axiom cond : ∀ (n : ℕ), n > 0 → 3 * a n = 2 * S n + n

-- Statement (I)
theorem geometric_sequence (h₀ : ∀ n : ℕ, n > 0 → 3 * a (n + 1) = 2 * (S (n + 1)) + (n + 1)) :
  ∃ r : ℝ, ∃ c : ℝ, ∀ n : ℕ, n >= 1 → (a n + 1 / 2) = c * r^n := sorry

-- Statement (II)
theorem find_T (h₀ : ∀ n : ℕ, n > 0 → 3 * a (n + 1) = 2 * (S (n + 1)) + (n + 1)) :
  ∀ n : ℕ, n > 0 → T n = (3^(n+2) - 9) / 8 - (n^2 + 4 * n) / 4 := sorry

end geometric_sequence_find_T_l378_378297


namespace find_f_one_inequality_solution_l378_378699

noncomputable def f : ℝ → ℝ := sorry

axiom f_increasing : ∀ x y : ℝ, (0 < x → 0 < y → x < y → f(x) < f(y))

axiom f_condition : ∀ x y : ℝ, (0 < x → 0 < y → f(x / y) = f(x) - f(y))

axiom f_6 : f 6 = 1

-- Prove f(1) = 0
theorem find_f_one : f 1 = 0 := sorry

-- Prove the solution set of the inequality is (-3,9)
theorem inequality_solution (x : ℝ) : f(x + 3) - f(1 / 3) < 2 ↔ -3 < x ∧ x < 9 := sorry

end find_f_one_inequality_solution_l378_378699


namespace sum_a_k_l378_378982

-- Define the sequence a_k
def a_k (k : Nat) : ℚ :=
  (k + 2 : ℚ) / (Nat.factorial k + Nat.factorial (k + 1) + Nat.factorial (k + 2))

-- Define the theorem to prove the sum of the sequence
theorem sum_a_k : ∑ k in Finset.range 1999, a_k (k + 1) = (1 / 2 : ℚ) - 1 / Nat.factorial 2001 :=
by
  sorry

end sum_a_k_l378_378982


namespace coordinates_of_P_with_respect_to_y_axis_l378_378736

-- Define the coordinates of point P
def P_x : ℝ := 5
def P_y : ℝ := -1

-- Define the point P
def P : Prod ℝ ℝ := (P_x, P_y)

-- State the theorem
theorem coordinates_of_P_with_respect_to_y_axis :
  (P.1, P.2) = (-P_x, P_y) :=
sorry

end coordinates_of_P_with_respect_to_y_axis_l378_378736


namespace wire_cutting_l378_378994

theorem wire_cutting : 
  ∃ (n : ℕ), n = 33 ∧ (∀ (x y : ℕ), 3 * x + y = 100 → x > 0 ∧ y > 0 → ∃ m : ℕ, m = n) :=
by {
  sorry
}

end wire_cutting_l378_378994


namespace calculate_number_of_sides_l378_378868

theorem calculate_number_of_sides (n : ℕ) (h : n ≥ 6) :
  ((6 : ℚ) / n^2) * ((6 : ℚ) / n^2) = 0.027777777777777776 →
  n = 6 :=
by
  sorry

end calculate_number_of_sides_l378_378868


namespace log_base_one_fourth_of_sixteen_l378_378254

theorem log_base_one_fourth_of_sixteen :
  log (1 / 4 : ℝ) (16 : ℝ) = -2 :=
sorry

end log_base_one_fourth_of_sixteen_l378_378254


namespace average_speed_correct_l378_378053

-- Define the initial and final time
def initial_time := 5 -- 5 a.m.
def final_time := 11 -- 11 a.m.

-- Define the initial and final distance
def initial_distance := 0 -- miles
def final_distance := 250 -- miles

-- Calculate total time in hours
def total_time := final_time - initial_time -- 6 hours

-- Calculate total distance driven
def total_distance := final_distance - initial_distance -- 250 miles

-- Define average speed function
def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Prove that the average speed is 41.67 miles per hour
theorem average_speed_correct : average_speed total_distance total_time = 41.67 := by
  -- calculation of the average speed based on given data
  sorry

end average_speed_correct_l378_378053


namespace min_shift_for_even_function_l378_378165

theorem min_shift_for_even_function :
  ∃ (m : ℝ), (m > 0) ∧ (∀ x : ℝ, (Real.sin (x + m) + Real.cos (x + m)) = (Real.sin (-x + m) + Real.cos (-x + m))) ∧ m = π / 4 :=
by
  sorry

end min_shift_for_even_function_l378_378165


namespace triangle_construction_l378_378330

theorem triangle_construction 
  (A D : ℝ × ℝ)
  (ρ ρ_a : ℝ)
  (vertex_A : A ≠ D) 
  (equal_angles : ∀ plane1 plane2, plane1 ∈ {x : ℕ | x ∈ {1,2}} → plane2 ∈ {x : ℕ | x ∈ {1,2}} → A.1 * A.2 = D.1 * D.2) -- this is a basic approximation for the plane condition, adjust as necessary
  (inscribed_circle_radius : circle A.1 ∈ {ρ})
  (excircle_radius_opposite_BC : circle D.1 ∈ {ρ_a})
: (ρ_a + ρ) / (ρ_a - ρ) = ( (A.1 * D.2) + (A.2 * D.1) / 2 ) / ( (A.1 * D.2) - (A.2 * D.1) / 2 )  :=
sorry

end triangle_construction_l378_378330


namespace percentage_difference_l378_378355

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.60 * x) (h2 : z = 0.60 * y) :
  abs ((z - x) / z * 100) = 4.17 :=
by
  sorry

end percentage_difference_l378_378355


namespace series_induction_change_l378_378501

theorem series_induction_change (k : ℕ) (h : k ≥ 2) :
  (∑ i in range (k+1), (1 / (k + 1 + i : ℝ))) + (1 / (2*(k+1) : ℝ)) - (1 / ((k+1) : ℝ) : ℝ))
  =
  (∑ i in range k, (1 / (k+1 + i : ℝ))) + (1 / (2*k + 1 : ℝ) : ℝ) :=
sorry

end series_induction_change_l378_378501


namespace range_of_g_l378_378393

noncomputable def g (x : ℝ) : ℝ := (Real.arctan x)^2 + (Real.arctan (1/x))^2

theorem range_of_g (x : ℝ) (h : x > 0) :
  ∃ y, y = g x ∧ (∃ a b, a = Real.arctan x ∧ b = Real.arctan (1/x) ∧ a + b = π / 2 ∧ y ∈ set.Icc (π^2 / 8) (π^2 / 4)) :=
sorry

end range_of_g_l378_378393


namespace average_points_per_player_l378_378752

variable (Lefty Righty Center Big : ℕ)

-- Conditions
axiom Lefty_scores : Lefty = 20
axiom Righty_scores : Righty = Lefty / 2
axiom Center_scores : Center = 6 * Righty
axiom Big_scores : Big = 3 * Center

-- Theorem to prove
theorem average_points_per_player (Lefty Righty Center Big : ℕ) : (Lefty + Righty + Center + Big) / 4 = 67.5 :=
by
  rw [Lefty_scores, Righty_scores, Center_scores, Big_scores]
  sorry

end average_points_per_player_l378_378752


namespace geometric_configuration_l378_378915

open EuclideanGeometry

theorem geometric_configuration 
  (A B C P Q R S : Point)
  (AC : Line) (PQ : Line)
  (K1 K2 : Circle)
  (B_between_A_C : between A B C)
  (K1_diameter_AB : K1.diameter = dist A B)
  (K2_diameter_BC : K2.diameter = dist B C)
  (touches_AC_at_B : touches AC B)
  (intersects_K1_at_P : K1 ∩ Circle P = dist A B)
  (intersects_K2_at_Q : K2 ∩ Circle Q = dist B C)
  (PQ_intersect_K1_at_R : PQ ∩ Circle R = K1)
  (PQ_intersect_K2_at_S : PQ ∩ Circle S = K2) :
  let perpendicular_AC_B := Line.perpendicular AC B in
  meet (Line.through A R) (Line.through C S) ∈ perpendicular_AC_B :=
sorry

end geometric_configuration_l378_378915


namespace sum_of_roots_l378_378414

variable {h b : ℝ}
variable {x₁ x₂ : ℝ}

-- Definition of the distinct property
def distinct (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂

-- Definition of the original equations given the conditions
def satisfies_equation (x : ℝ) (h b : ℝ) : Prop := 3 * x^2 - h * x = b

-- Main theorem statement translating the given mathematical problem
theorem sum_of_roots (h b : ℝ) (x₁ x₂ : ℝ) (h₁ : satisfies_equation x₁ h b) 
  (h₂ : satisfies_equation x₂ h b) (h₃ : distinct x₁ x₂) : x₁ + x₂ = h / 3 :=
sorry

end sum_of_roots_l378_378414


namespace determinant_modified_l378_378697

variable (a b c d : ℝ)

theorem determinant_modified (h : a * d - b * c = 10) :
  (a + 2 * c) * d - (b + 3 * d) * c = 10 - c * d := by
  sorry

end determinant_modified_l378_378697


namespace train_crossing_time_l378_378500

theorem train_crossing_time :
  ∀ (sf ss : ℝ) (lf : ℕ), sf = 72 → ss = 36 → lf = 100 →
  (lf / ((sf - ss) * (5 / 18)) = 10) :=
begin
  intros sf ss lf hsf hss hlf,
  have h1 : (sf - ss) * (5 / 18) = 10,
  { rw [hsf, hss],
    norm_num },
  rw [hlf],
  exact (div_eq_iff_mul_eq _ _).2 (by rw [h1] ; norm_num),
  { norm_num }
  { rw [h1] ; norm_num }
end

end train_crossing_time_l378_378500


namespace num_ways_product_72_l378_378758

def num_ways_product (n : ℕ) : ℕ := sorry  -- Definition for D(n), the number of ways to write n as a product of integers greater than 1

def example_integer := 72  -- Given integer n

theorem num_ways_product_72 : num_ways_product example_integer = 67 := by 
  sorry

end num_ways_product_72_l378_378758


namespace log_base_one_fourth_of_sixteen_l378_378259

theorem log_base_one_fourth_of_sixteen :
  log (1 / 4 : ℝ) (16 : ℝ) = -2 :=
sorry

end log_base_one_fourth_of_sixteen_l378_378259


namespace maximum_volume_box_l378_378090

noncomputable def maximum_volume (side_length : ℝ) :=
  let V := λ x : ℝ, x * (side_length - 2*x)^2
  real_Sup (set.range V)

theorem maximum_volume_box :
  maximum_volume 20 = 16000 / 27 :=
begin
  sorry
end

end maximum_volume_box_l378_378090


namespace area_enclosed_by_graph_l378_378859

noncomputable def enclosed_area (x y : ℝ) : ℝ := 
  if h : (|5 * x| + |3 * y| = 15) then
    30 -- The area enclosed by the graph
  else
    0 -- Default case for definition completeness

theorem area_enclosed_by_graph : ∀ (x y : ℝ), (|5 * x| + |3 * y| = 15) → enclosed_area x y = 30 :=
by
  sorry

end area_enclosed_by_graph_l378_378859


namespace intersection_at_x_is_minus_2_8475_l378_378510

noncomputable def intersection_x : ℝ :=
  let y (x : ℝ) := 10 / (x ^ 2 + 4)
  Xu (x + y x = 3) := x

theorem intersection_at_x_is_minus_2_8475 :
  intersection_x = -2.8475 :=
by
  sorry

end intersection_at_x_is_minus_2_8475_l378_378510


namespace profit_percentage_first_fund_is_correct_l378_378572

-- Define the given values and conditions
def total_investment : ℝ := 1900
def profit_percentage_second_fund : ℝ := 2
def total_profit : ℝ := 52
def investment_first_fund : ℝ := 1700
def investment_second_fund := total_investment - investment_first_fund
def profit_second_fund := (profit_percentage_second_fund / 100) * investment_second_fund

-- Define the unknown profit percentage for the first mutual fund
noncomputable def profit_percentage_first_fund (P : ℝ) :=
  (P / 100) * investment_first_fund + profit_second_fund = total_profit

-- The theorem to prove the profit percentage of the first mutual fund
theorem profit_percentage_first_fund_is_correct :
  ∃ P, profit_percentage_first_fund P ∧ P = 2.82 := by
  use 2.82
  sorry

end profit_percentage_first_fund_is_correct_l378_378572


namespace centroid_position_count_correct_l378_378033

noncomputable def centroid_position_count : ℕ :=
  let points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 33, 34, 35, 28, 20, 12] in
  let centroids := finset.image (λ (p q r : ℤ × ℤ), 
    ((p.1 + q.1 + r.1) / 3, (p.2 + q.2 + r.2) / 3)) 
    ((finset.univ.product finset.univ).product finset.univ).filter (λ ((p, q), r), 
      ¬collinear p q r) in
  finset.card (centroids)

theorem centroid_position_count_correct :
  centroid_position_count = 529 :=
by {
  sorry,
}

end centroid_position_count_correct_l378_378033


namespace animals_in_field_l378_378934

theorem animals_in_field :
  let dog := 1 in
  let cats := 4 in
  let rabbits_per_cat := 2 in
  let hares_per_rabbit := 3 in
  let total_animals := 
    dog + 
    cats + 
    cats * rabbits_per_cat + 
    (cats * rabbits_per_cat) * hares_per_rabbit 
  in total_animals = 37 := by
sorry

end animals_in_field_l378_378934


namespace sequence_extremes_l378_378638

noncomputable def a_n (n : ℕ) (h : n > 0) : ℝ := (n - Real.sqrt 2015) / (n - Real.sqrt 2016)

theorem sequence_extremes :
  ∀ (n : ℕ) (h : 0 < n ≤ 50),
    let seq := λ (k : ℕ) (hk : k > 0), a_n k hk in
    seq 44 (by linarith) = Inf (set.image (λ k, seq k (by linarith)) {k | 0 < k ∧ k ≤ 50}) ∧
    seq 45 (by linarith) = Sup (set.image (λ k, seq k (by linarith)) {k | 0 < k ∧ k ≤ 50}) :=
sorry

end sequence_extremes_l378_378638


namespace cost_to_paint_floor_l378_378474

-- Define the conditions
def length_more_than_breadth_by_200_percent (L B : ℝ) : Prop :=
L = 3 * B

def length_of_floor := 23
def cost_per_sq_meter := 3

-- Prove the cost to paint the floor
theorem cost_to_paint_floor (B : ℝ) (L : ℝ) 
    (h1: length_more_than_breadth_by_200_percent L B) (h2: L = length_of_floor) 
    (rate: ℝ) (h3: rate = cost_per_sq_meter) :
    rate * (L * B) = 529.23 :=
by
  -- intermediate steps would go here
  sorry

end cost_to_paint_floor_l378_378474


namespace sum_of_squares_of_roots_l378_378059

theorem sum_of_squares_of_roots :
  let a := (5 : ℤ)
  let b := (6 : ℤ)
  let c := (-7 : ℤ)
  let x1_plus_x2 := -(b : ℚ) / a
  let x1_times_x2 := (c : ℚ) / a
  (x1_plus_x2)^2 - 2 * x1_times_x2 = 106 / 25 := by
sory

end sum_of_squares_of_roots_l378_378059


namespace zeros_of_f_l378_378341

noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem zeros_of_f (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (∃ x, a < x ∧ x < b ∧ f a b c x = 0) ∧ (∃ y, b < y ∧ y < c ∧ f a b c y = 0) :=
by
  sorry

end zeros_of_f_l378_378341


namespace probability_two_boxes_with_four_same_color_blocks_l378_378573

-- Definitions based on the conditions
structure Person :=
  (blocks : Finset String) -- assuming blocks are identified by color strings

structure Box :=
  (contents : Finset (Person × String)) -- assumes contents are pairs of person and block color

-- Definition of the problem
noncomputable def probability_at_least_two_boxes_with_same_color_blocks : ℚ :=
  -- To represent the correct answer here
  2 / 18645

-- Top-level theorem statement
theorem probability_two_boxes_with_four_same_color_blocks :
  let Ang : Person := ⟨{"red", "blue", "yellow", "white", "green"}⟩,
      Ben : Person := ⟨{"red", "blue", "yellow", "white", "green"}⟩,
      Jasmin : Person := ⟨{"red", "blue", "yellow", "white", "green"}⟩,
      Kyle : Person := ⟨{"red", "blue", "yellow", "white", "green"}⟩,
      boxes : Fin 5 → Box := λ _, ⟨∅⟩ in
  -- Theorem statement to check the calculated probability
  probability_at_least_two_boxes_with_same_color_blocks = 2 / 18645 :=
sorry

end probability_two_boxes_with_four_same_color_blocks_l378_378573


namespace vector_perpendicular_solution_l378_378333

/-- Prove that for given vectors a and b, if a + x * b is perpendicular to b then x = -2/5. -/
theorem vector_perpendicular_solution (x : ℝ) 
  (ha : (3 : ℝ, 4 : ℝ)) 
  (hb : (2 : ℝ, -1 : ℝ)) 
  (h_perpendicular : (3 + 2 * x, 4 - x) ∙ hb = 0) : 
  x = -2 / 5 :=
sorry

end vector_perpendicular_solution_l378_378333


namespace sum_of_elements_eq_sum_of_counts_l378_378520

theorem sum_of_elements_eq_sum_of_counts (n : ℕ) (a : Fin n → ℕ) (b : ℕ → ℕ) 
  (h_b : ∀ k, b k = (Finset.univ.filter (λ i, a i ≥ k)).card) :
  (Finset.univ.sum (λ i, a i)) = (Finset.range (Finset.univ.sup a + 1)).sum (λ k, b k) := 
sorry

end sum_of_elements_eq_sum_of_counts_l378_378520


namespace solve_rings_l378_378000

variable (B : ℝ) (S : ℝ)

def conditions := (S = (5/8) * (Real.sqrt B)) ∧ (S + B = 52)

theorem solve_rings : conditions B S → (S + B = 52) := by
  intros h
  sorry

end solve_rings_l378_378000


namespace price_increase_2012_2013_l378_378912

variables (P : ℝ) (p : ℝ)
def price_2012 (P : ℝ) (p : ℝ): ℝ := P + (p / 100) * P
def price_2013 (P : ℝ) (p : ℝ): ℝ := (price_2012 P p) - 0.12 * (price_2012 P p)

theorem price_increase_2012_2013 (h : price_2013 P p = 1.1 * P) :
  p = 22.27 :=
sorry

end price_increase_2012_2013_l378_378912


namespace mean_of_medians_is_45_l378_378195

-- Define the conditions
def isValidDigitArray (d : Fin 10 × Fin 7 → ℕ) : Prop :=
  ∀ i : Fin 9, ∀ j : Fin 7, d ⟨i.1 + 1, j⟩ - d ⟨i, j⟩ = 1 ∨ d ⟨i.1 + 1, j⟩ - d ⟨i , j⟩ = -9

def median (l : List ℕ) : ℕ :=
  l.nth ((l.length - 1) / 2)

def m (d : Fin 10 × Fin 7 → ℕ) (i : Fin 10) : ℕ :=
  median (List.ofFn (λ j : Fin 7 => d (⟨i, j⟩)))

def mean (l : List ℕ) : ℚ := l.sum / l.length

-- The theorem to be proved
theorem mean_of_medians_is_45 (d : Fin 10 × Fin 7 → ℕ) (h_valid : isValidDigitArray d) : mean (List.ofFn (m d)) = 4.5 := 
  sorry -- Proof goes here

end mean_of_medians_is_45_l378_378195


namespace proof_problem_l378_378603

theorem proof_problem (x y : ℝ) (h1 : 3 * x ^ 2 - 5 * x + 4 * y + 6 = 0) 
                      (h2 : 3 * x - 2 * y + 1 = 0) : 
                      4 * y ^ 2 - 2 * y + 24 = 0 := 
by 
  sorry

end proof_problem_l378_378603


namespace line_quadrant_relationship_l378_378937

theorem line_quadrant_relationship
  (a b c : ℝ)
  (passes_first_second_fourth : ∀ x y : ℝ, (a * x + b * y + c = 0) → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) :
  (a * b > 0) ∧ (b * c < 0) :=
sorry

end line_quadrant_relationship_l378_378937


namespace log_base_frac_l378_378236

theorem log_base_frac (x : ℝ) : log (1/4) 16 = x → x = -2 := by
  sorry

end log_base_frac_l378_378236


namespace area_enclosed_by_graph_l378_378860

noncomputable def enclosed_area (x y : ℝ) : ℝ := 
  if h : (|5 * x| + |3 * y| = 15) then
    30 -- The area enclosed by the graph
  else
    0 -- Default case for definition completeness

theorem area_enclosed_by_graph : ∀ (x y : ℝ), (|5 * x| + |3 * y| = 15) → enclosed_area x y = 30 :=
by
  sorry

end area_enclosed_by_graph_l378_378860


namespace xiao_lin_wednesday_xiao_lin_least_and_most_difference_xiao_lin_total_distance_l378_378110

-- Definitions based on conditions
def standard_distance : ℕ := 1000
def deviations : List ℤ := [420, 460, -100, -210, -330, 200, 0]

-- Lean statements for the questions with their answers
theorem xiao_lin_wednesday : (standard_distance : ℤ) + deviations.nth_le 2 (by simp) = 900 := by sorry

theorem xiao_lin_least_and_most_difference : 
  let least_run := standard_distance + deviations.argmin (by simp)
  let most_run := (standard_distance + deviations.argmax (by simp))
  least_run = 670 ∧ (most_run - least_run) = 790 :=
by sorry

theorem xiao_lin_total_distance : 
  list.sum (List.map (λ x => (standard_distance : ℤ) + x) deviations) = 7440 := 
by sorry

end xiao_lin_wednesday_xiao_lin_least_and_most_difference_xiao_lin_total_distance_l378_378110


namespace marbles_per_group_l378_378445

theorem marbles_per_group (marbles groups : ℕ) (h_marble_count : marbles = 20) (h_group_count : groups = 5) : marbles / groups = 4 := by
  rw [h_marble_count, h_group_count]
  norm_num
  sorry

end marbles_per_group_l378_378445


namespace curve_equations_and_triangle_area_l378_378737

theorem curve_equations_and_triangle_area :
  (∀ (α : ℝ), ∃ (θ : ℝ), (2 + 2 * Real.cos α = ρ := 4 * Real.cos θ) ∧ (2 * Real.sin α = 2 * Real.sin θ)) ∧
  (∀ (θ : ℝ), ∃ ρ, (ρ = 2 * Real.sin θ) ↔ ((x := ρ * cos θ, y := ρ * sin θ) → x^2 + y^2 - 2 * y = 0)) ∧
  (P Q : ℝ × ℝ) (θ : ℝ), 
  (P ∈ curve(4 * Real.cos θ)) ∧ (Q ∈ curve (2 * Real.sin θ)) ∧ (angle P O Q = π / 3) →
  area_of_triangle P O Q ≤ sqrt 3 :=
sorry

end curve_equations_and_triangle_area_l378_378737


namespace cone_height_from_sphere_l378_378560

theorem cone_height_from_sphere (d_sphere d_base : ℝ) (h : ℝ) (V_sphere : ℝ) (V_cone : ℝ) 
  (h₁ : d_sphere = 6) 
  (h₂ : d_base = 12)
  (h₃ : V_sphere = 36 * Real.pi)
  (h₄ : V_cone = (1/3) * Real.pi * (d_base / 2)^2 * h) 
  (h₅ : V_sphere = V_cone) :
  h = 3 := by
  sorry

end cone_height_from_sphere_l378_378560


namespace each_goldfish_eats_l378_378843

variable {G : Type} -- G for Goldfish type

-- Conditions
def total_goldfish : ℕ := 50
def percentage_needing_special_food : ℝ := 0.20
def cost_per_ounce : ℝ := 3
def total_cost : ℝ := 45

-- Definitions derived from the conditions
def num_goldfish_needing_special_food : ℕ :=
  (percentage_needing_special_food * total_goldfish).to_nat

def total_ounces_special_food : ℝ :=
  total_cost / cost_per_ounce

def ounces_per_goldfish : ℝ :=
  total_ounces_special_food / num_goldfish_needing_special_food

-- The theorem to be proved
theorem each_goldfish_eats :
  ounces_per_goldfish = 1.5 := by
  sorry

end each_goldfish_eats_l378_378843


namespace polynomial_simplification_l378_378051

theorem polynomial_simplification 
  (x : ℝ)
  (H : x ≠ -2) :
  let A : ℝ := 1
  let B : ℝ := 8
  let C : ℝ := 15
  let D : ℝ := -2
  let f := λ x, (x^3 + 10 * x^2 + 31 * x + 30) / (x + 2)
  let g := λ x, A * x^2 + B * x + C
  f x = g x ∧ x ≠ -2 →
  A + B + C + D = 22 := 
by 
  sorry

end polynomial_simplification_l378_378051


namespace problem1_l378_378918

theorem problem1 : (real.sqrt 12 - 4 * abs (real.sin (real.pi / 3)) + real.inv (1/3) - real.pow (2023 - real.pi) 0) = 2 := 
sorry

end problem1_l378_378918


namespace truck_distance_on_7_liters_l378_378562

-- Define the conditions
def truck_300_km_per_5_liters := 300
def liters_5 := 5
def liters_7 := 7
def expected_distance_7_liters := 420

-- The rate of distance (km per liter)
def rate := truck_300_km_per_5_liters / liters_5

-- Proof statement
theorem truck_distance_on_7_liters :
  rate * liters_7 = expected_distance_7_liters :=
  by
  sorry

end truck_distance_on_7_liters_l378_378562


namespace simplify_and_evaluate_expression_l378_378455

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 2 - 3) : 
  (1 - (3 / (m + 3))) / (m / (m^2 + 6 * m + 9)) = Real.sqrt 2 := 
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l378_378455


namespace exotic_odd_implies_square_infinitely_many_exotic_numbers_l378_378503

def is_exotic (n : ℕ) : Prop := ∃ d, d > 0 ∧ d ∣ n ∧ nat.totient d = d

theorem exotic_odd_implies_square (n : ℕ) (h₁ : is_exotic n) (h₂ : n % 2 = 1) : ∃ k, n = k * k :=
sorry

theorem infinitely_many_exotic_numbers : ∀ m, ∃ n > m, is_exotic n :=
sorry

end exotic_odd_implies_square_infinitely_many_exotic_numbers_l378_378503


namespace absolute_difference_tangency_lengths_l378_378556

theorem absolute_difference_tangency_lengths :
  ∀ (x y : ℝ), let a := 80 in
  let b := 120 in
  let c := 140 in
  let d := 100 in
  let s := (a + b + c + d) / 2 in
  let K := Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) in
  let r := K / s in
  x + y = c → 
  x = y - r → 
  |x - y| = 166.36 :=
  sorry

end absolute_difference_tangency_lengths_l378_378556


namespace triangle_AOB_is_right_l378_378683

noncomputable def parabola_line_intersection (k : ℝ) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ),
    (y1^2 + x1 = 0) ∧ (y2^2 + x2 = 0) ∧         -- Points lie on the parabola y^2 = -x
    (y1 = k * (x1 + 1)) ∧ (y2 = k * (x2 + 1)) ∧ -- Points lie on the line y = k(x + 1)
    (x1 * x2 + y1 * y2 = 0)                     -- The dot product is zero

theorem triangle_AOB_is_right (k : ℝ) : parabola_line_intersection k → ∃ A B O : ℝ × ℝ, is_right_angle A B O :=
sorry

end triangle_AOB_is_right_l378_378683


namespace melissa_total_points_l378_378776

-- Definition of the points scored per game and the number of games played.
def points_per_game : ℕ := 7
def number_of_games : ℕ := 3

-- The total points scored by Melissa is defined as the product of points per game and number of games.
def total_points_scored : ℕ := points_per_game * number_of_games

-- The theorem stating the verification of the total points scored by Melissa.
theorem melissa_total_points : total_points_scored = 21 := by
  -- The proof will be given here.
  sorry

end melissa_total_points_l378_378776


namespace angle_PMN_eq_40_l378_378732

variables {P Q R M N : Type} 
variables (angle : P → P → ℝ)

-- Conditions
axiom angle_PQR_eq_60 : angle P Q R = 60
axiom angle_MPR_eq_20 : angle M P R = 20
axiom PR_eq_RQ : ∀ {a b : P}, a = b → (angle a P R) = (angle b P Q)
axiom PM_eq_PN : ∀ {a b : P}, a = b

-- Proof goal (we do not provide the actual proof here)
theorem angle_PMN_eq_40 : angle P M N = 40 := 
sorry

end angle_PMN_eq_40_l378_378732


namespace trigonometric_product_identity_l378_378192

theorem trigonometric_product_identity : 
  let cos_40 : Real := Real.cos (Real.pi * 40 / 180)
  let sin_40 : Real := Real.sin (Real.pi * 40 / 180)
  let cos_50 : Real := Real.cos (Real.pi * 50 / 180)
  let sin_50 : Real := Real.sin (Real.pi * 50 / 180)
  (sin_50 = cos_40) → (cos_50 = sin_40) →
  (1 - cos_40⁻¹) * (1 + sin_50⁻¹) * (1 - sin_40⁻¹) * (1 + cos_50⁻¹) = 1 := by
  sorry

end trigonometric_product_identity_l378_378192


namespace closest_integer_to_square_of_largest_root_l378_378767

noncomputable def f (x : ℝ) : ℝ := x^3 - 8*x^2 - 2*x + 3

theorem closest_integer_to_square_of_largest_root :
  let a := Real.lub { x : ℝ | f x = 0 }
  (Int.round (a^2) = 67) :=
sorry

end closest_integer_to_square_of_largest_root_l378_378767


namespace simplify_expression_l378_378703

theorem simplify_expression (u : ℝ) (cbrt_3 : ℝ) (h : u = 1 / (2 - cbrt_3)) (h_cbrt_3 : cbrt_3 = real.cbrt 3) :
  u = 2 + real.cbrt 3 :=
by
  -- This is the statement part, the proof is not included as per the instructions
  sorry

end simplify_expression_l378_378703


namespace find_divisor_l378_378118

theorem find_divisor (D Q R d: ℕ) (hD: D = 16698) (hQ: Q = 89) (hR: R = 14) (hDiv: D = d * Q + R): d = 187 := 
by 
  sorry

end find_divisor_l378_378118


namespace odd_divisibility_l378_378087

theorem odd_divisibility (n : ℕ) (k : ℕ) (x y : ℤ) (h : n = 2 * k + 1) : (x^n + y^n) % (x + y) = 0 :=
by sorry

end odd_divisibility_l378_378087


namespace range_of_a_plus_c_l378_378727

-- Definitions as per the given conditions
variable (A B C a b c : ℝ)
variable (acute_triangle_ABC : (0 < A) ∧ (A < π / 2) ∧ (0 < B) ∧ (B < π / 2) ∧ (0 < C) ∧ (C < π / 2) ∧ (A + B + C = π))
variable (sides_relation1 : (cos B / b + cos C / c = 2 * sqrt 3 * sin A / (3 * sin C)))
variable (sides_relation2 : (cos B + sqrt 3 * sin B = 2))
variable (cosine_law_bc : (b^2 = a^2 + c^2 - 2 * a * c * cos B))

-- The theorem to be proved
theorem range_of_a_plus_c :
  acute_triangle_ABC →
  sides_relation1 →
  sides_relation2 →
  cosine_law_bc →
  (sqrt 3 / 2 < a + c ∧ a + c ≤ sqrt 3) :=
by
  sorry

end range_of_a_plus_c_l378_378727


namespace integral_limit_l378_378794

noncomputable theory

variables {f g : ℝ → ℝ}

-- Assumptions on f and g
axiom f_continuous : continuous f
axiom g_continuous : continuous g
axiom f_periodic : ∀ x, f (x + 1) = f x
axiom g_periodic : ∀ x, g (x + 1) = g x

-- The theorem to be proved
theorem integral_limit :
  (lim (λ n : ℕ, ∫ x in 0..1, f x * g (n * x))) = 
  (∫ x in 0..1, f x) * (∫ x in 0..1, g x) := sorry

end integral_limit_l378_378794


namespace units_digit_M_M10_l378_378597

def M : ℕ → ℕ 
| 0 := 3
| 1 := 2
| (n + 2) := M (n + 1) + M n

def units_digit (x : ℕ) : ℕ :=
  x % 10

theorem units_digit_M_M10 :
  units_digit (M (M 10)) = 1 := by
  sorry

end units_digit_M_M10_l378_378597


namespace card_stack_sum_l378_378018

theorem card_stack_sum :
  ∃ (stack : List ℚ), (stack.length = 10) ∧
  (∀ i, i < 9 → ((stack.nth i).getD 0 < 10 → ((i % 2 = 0 → stack.nth i).getD 0 = 1 ∨ 2 ∨ 3 ∨ 4 ∨ 5) ∧ (i % 2 = 1 → stack.nth i).getD 0 = 4 ∨ 5.5 ∨ 6 ∨ 7 ∨ 8)
  ∧ (∀ j, 1 ≤ j ∧ j ≤ 8, (stack.nth (j - 1)).getD 0 / (stack.nth j).getD 0 ∈ int )) ∧
  ((stack.nth 4).getD 0 + (stack.nth 5).getD 0 + (stack.nth 6).getD 0 = 16)) :=
by
  sorry

end card_stack_sum_l378_378018


namespace garbage_decomposition_time_l378_378312

theorem garbage_decomposition_time 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h1 : a * b^12 = 0.1) 
  (h2 : a * b^24 = 0.2) :
  ∃ t : ℝ, (0.05 * (2^(1 / 12))^t = 1) ∧ (t ≈ 52) := 
sorry

end garbage_decomposition_time_l378_378312


namespace john_third_task_completion_time_l378_378389

theorem john_third_task_completion_time:
  let start_time := 510    -- 8:30 AM in minutes after midnight
  let end_time := 670      -- 11:10 AM in minutes after midnight
  let task_count := 2
  let total_time := end_time - start_time
  let time_per_task := total_time / task_count
  let third_task_completion_time := end_time + time_per_task
  in third_task_completion_time = 750 :=   -- 12:30 PM in minutes after midnight
  sorry

end john_third_task_completion_time_l378_378389


namespace find_a_solve_inequality_l378_378322

-- Define the given conditions
def g (a : ℝ) (x : ℝ) := (a + 1)^(x - 2) + 1
def f (a : ℝ) (x : ℝ) := Real.log (x + a) / Real.log (Real.sqrt 3)

-- Problem 1: prove the value of a is 1
theorem find_a (h1 : a > 0) (hx : g a 2 = 2) (hy : f a 2 = 2) : a = 1 :=
by
  -- We skip the proof here
  sorry

-- Problem 2: solve the inequality g(x) > 3
theorem solve_inequality (a : ℝ) (x : ℝ) (h : a = 1) : g a x > 3 ↔ x > 3 :=
by
  -- We skip the proof here
  sorry

end find_a_solve_inequality_l378_378322


namespace probability_x_squared_lt_y_correct_l378_378555

open interval_integral measure_theory set filter
open_locale big_operators

noncomputable def probability_x_squared_lt_y : ℝ :=
let total_area : ℝ := 12 in
let lower_bound : ℝ := 0 in
let upper_bound : ℝ := real.sqrt 2 in
let area_under_parabola : ℝ := ∫ x in lower_bound..upper_bound, x^2 in
let probability : ℝ := area_under_parabola / total_area in
probability

theorem probability_x_squared_lt_y_correct :
  probability_x_squared_lt_y = real.sqrt 2 / 18 :=
by 
  have total_area : ℝ := 12,
  have lower_bound : ℝ := 0,
  have upper_bound : ℝ := real.sqrt 2,
  have area_under_parabola : ℝ := ∫ x in lower_bound..upper_bound, x^2,
  have area_value : ℝ := (2 * real.sqrt 2) / 3,
  have probability : ℝ := area_value / total_area,
  have correct_probability : ℝ := real.sqrt 2 / 18,
  sorry

end probability_x_squared_lt_y_correct_l378_378555


namespace smallest_integer_value_of_eight_nums_l378_378487

theorem smallest_integer_value_of_eight_nums (a : Fin 8 → ℝ)
    (h_sum : (Finset.univ.sum (λ i, a i) = (4 : ℝ) / 3))
    (h_sum_pos : ∀ i : Fin 8, (Finset.univ.erase i).sum (λ j, a j) > 0)
    : ∃ (k : ℤ), (k = -7) ∧ (∀ i : Fin 8, (a i) ≥ (↑k : ℝ)) :=
by
  sorry

end smallest_integer_value_of_eight_nums_l378_378487


namespace log_one_fourth_sixteen_l378_378251

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 := 
by
  let x := log (1 / 4) 16
  have h₁ : (1 / 4) ^ x = 16 := by simp [log_eq_iff]
  have h₂ : (4 ^ (-1)) ^ x = 16 := by rw [one_div, inv_pow]
  have h₃ : 4 ^ (-x) = 16 := by simp [pow_mul]
  have h₄ : 16 = 4 ^ 2 := by norm_num
  rw [h₄] at h₃
  have h₅ : -x = 2 := by exact pow_inj (lt_trans zero_lt_one (by norm_num)) zero_lt_four h₃
  have h₆ : x = -2 := by linarith
  exact h₆

end log_one_fourth_sixteen_l378_378251


namespace smallest_base_10_integer_l378_378872

theorem smallest_base_10_integer (a b : ℕ) (ha : a > 2) (hb : b > 2) 
  (h1: 21_a = 2 * a + 1) (h2: 12_b = b + 2) : 2 * a + 1 = 7 :=
by 
  sorry

end smallest_base_10_integer_l378_378872


namespace sum_of_nonneg_numbers_ineq_l378_378491

theorem sum_of_nonneg_numbers_ineq
  (a b c d : ℝ)
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0)
  (h_sum : a + b + c + d = 4) :
  (a * b + c * d) * (a * c + b * d) * (a * d + b * c) ≤ 8 := sorry

end sum_of_nonneg_numbers_ineq_l378_378491


namespace general_terms_sum_c_n_l378_378299

noncomputable def a_n (n : ℕ) : ℝ := 2 * n - 1

def b_seq (T : ℕ → ℝ) : ℕ → ℝ
| 0     := 3
| (n+1) := 2 * T n + 3

noncomputable def b_n (n : ℕ) : ℝ := 3 ^ n

noncomputable def c_n (n : ℕ) : ℝ := (a_n n) / (b_n n)

noncomputable def M_n (n : ℕ) : ℝ := 1 - (n + 1) / (3 ^ n)

theorem general_terms :
  (∀ n, a_n n = 2 * n - 1) ∧
  (∀ n, b_n n = 3 ^ n) := sorry

theorem sum_c_n (n : ℕ) :
  (∑ i in finset.range n, c_n (i + 1)) = M_n n := sorry

end general_terms_sum_c_n_l378_378299


namespace animals_in_field_l378_378931

def dog := 1
def cats := 4
def rabbits_per_cat := 2
def hares_per_rabbit := 3

def rabbits := cats * rabbits_per_cat
def hares := rabbits * hares_per_rabbit

def total_animals := dog + cats + rabbits + hares

theorem animals_in_field : total_animals = 37 := by
  sorry

end animals_in_field_l378_378931


namespace full_time_employees_l378_378152

theorem full_time_employees (total_employees part_time_employees number_full_time_employees : ℕ)
  (h1 : total_employees = 65134)
  (h2 : part_time_employees = 2041)
  (h3 : number_full_time_employees = total_employees - part_time_employees)
  : number_full_time_employees = 63093 :=
by {
  sorry
}

end full_time_employees_l378_378152


namespace problem_statement_l378_378676

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then Real.log x / Real.log 3 else 2 ^ x

theorem problem_statement : f (f (1/9)) = 1/4 := by
  sorry

end problem_statement_l378_378676


namespace abs_neg_one_div_three_l378_378036

open Real

theorem abs_neg_one_div_three : abs (-1 / 3) = 1 / 3 :=
by
  sorry

end abs_neg_one_div_three_l378_378036


namespace valid_schedules_week5_l378_378119

-- Definitions for experiments and their weeks
def Week := ℕ
def Group := {A, B, C, D, E}
def Experiment := {1, 2, 3, 4, 5}

def week1 := (D, 4)
def week2 := (C, 5)
def week3 := (E, 5)
def week4 := ((A, 4), (D, 2))

-- Valid schedules for Week 5
def schedule1 : List (Group × Experiment) := [(A, 1), (B, 2), (C, 3), (D, 5), (E, 4)]
def schedule2 : List (Group × Experiment) := [(A, 3), (B, 1), (C, 2), (D, 5), (E, 4)]

theorem valid_schedules_week5 :
  ∃ (s1 s2 : List (Group × Experiment)), s1 = schedule1 ∧ s2 = schedule2 ∧
  (∀ g1 e1 g2 e2, (g1, e1) ∈ s1 ∧ (g2, e2) ∈ s1 → g1 ≠ g2 ∧ e1 ≠ e2) ∧
  (∀ g1 e1 g2 e2, (g1, e1) ∈ s2 ∧ (g2, e2) ∈ s2 → g1 ≠ g2 ∧ e1 ≠ e2) ∧
  (D, 4) ∈ [(D, 4)] ∧
  (C, 5) ∈ [(C, 5)] ∧
  (E, 5) ∈ [(E, 5)] ∧
  ((A, 4), (D, 2)) ∈ [((A, 4), (D, 2))] := sorry

end valid_schedules_week5_l378_378119


namespace daily_shirt_ironing_time_l378_378976

theorem daily_shirt_ironing_time (total_ironing_time : ℕ) (daily_pants_time : ℕ) (days_per_week : ℕ) (weeks : ℕ) 
  (H1 : total_ironing_time = 160)
  (H2 : daily_pants_time = 3)
  (H3 : days_per_week = 5)
  (H4 : weeks = 4) : 
  let daily_shirt_time := (total_ironing_time - daily_pants_time * days_per_week * weeks) / (days_per_week * weeks) in
  daily_shirt_time = 5 := 
by
  sorry

end daily_shirt_ironing_time_l378_378976


namespace distance_sum_leq_perimeter_l378_378395

open EuclideanGeometry

variables {A B C I D E F X Y Z : Point}

-- Definitions and assumptions
def Incenter (I A B C : Point) : Prop := isIncenter I A B C
def Intersection (P Q R S : Point) (M : Point) : Prop := isIntersection P Q R S M
def OnSegment (P Q R : Point) : Prop := onSegment P Q R
def distance (P : Point) (ℓ : Line) : ℝ := distPointLine P ℓ -- assuming a function that gives this distance

axiom A1 : Triangle A B C
axiom A2 : Incenter I A B C
axiom A3 : Intersection A I B C D
axiom A4 : Intersection B I C A E
axiom A5 : Intersection C I A B F
axiom A6 : OnSegment X E F
axiom A7 : OnSegment Y F D
axiom A8 : OnSegment Z D E

-- Problem statement
theorem distance_sum_leq_perimeter :
  distance X (Line.mk A B) + distance Y (Line.mk B C) + distance Z (Line.mk C A) ≤ dist X Y + dist Y Z + dist Z X :=
sorry

end distance_sum_leq_perimeter_l378_378395


namespace greatest_x_l378_378096

theorem greatest_x (x : ℕ) (h : x > 0 ∧ (x^4 / x^2 : ℚ) < 18) : x ≤ 4 :=
by
  sorry

end greatest_x_l378_378096


namespace count_seniors_l378_378723

theorem count_seniors (total_students juniors_percentage not_sophomores_percentage sophomores_to_freshmen_difference : ℕ) 
  (h1 : total_students = 800)
  (h2 : juniors_percentage = 27)
  (h3 : not_sophomores_percentage = 75)
  (h4 : sophomores_to_freshmen_difference = 24) :
  let juniors := juniors_percentage * total_students / 100 in
  let sophomores := (100 - not_sophomores_percentage) * total_students / 100 in
  let freshmen := sophomores + sophomores_to_freshmen_difference in
  let seniors := total_students - (juniors + sophomores + freshmen) in
  seniors = 160 :=
by {
  sorry
}

end count_seniors_l378_378723


namespace pedal_triangles_of_H_similar_no_pedal_triangle_of_G_similar_l378_378914

-- Definitions for the conditions
def is_similar (Δ1 Δ2 : Triangle) : Prop := sorry -- Assume definition for similarity
def is_pedal_triangle (parent child : Triangle) : Prop := sorry -- Assume definition for pedal triangle

-- Main problem statement
theorem pedal_triangles_of_H_similar {H T1 T2 : Triangle} 
  (h_ratio : angles_ratio H 1 2 4)
  (h_T1_is_pedal : is_pedal_triangle H T1)
  (h_T2_is_pedal : is_pedal_triangle T1 T2): 
  is_similar T2 H :=
sorry

theorem no_pedal_triangle_of_G_similar {G T1 : Triangle}
  (g_ratio : angles_ratio G 1 1 8)
  (h_T1_is_pedal : is_pedal_triangle G T1) :
  ¬ is_similar T1 G :=
sorry

end pedal_triangles_of_H_similar_no_pedal_triangle_of_G_similar_l378_378914


namespace coeff_x3_in_sum_l378_378582

theorem coeff_x3_in_sum (sum_coeffs : ℕ → ℕ → ℕ)
  (coeff_5_3 : sum_coeffs 5 3 = 10)
  (coeff_6_3 : sum_coeffs 6 3 = 20)
  (coeff_7_3 : sum_coeffs 7 3 = 35)
  -- Add similar conditions for all required binomial coefficients...
  (coeff_18_3 : sum_coeffs 18 3 = 816)
  (coeff_19_3 : sum_coeffs 19 3 = 969) :

  -- Calculate the binomial coefficients' sum from 5 to 19
  let sum := -(sum_coeffs 5 3 + sum_coeffs 6 3 +
               sum_coeffs 7 3 + sum_coeffs 8 3 +
               sum_coeffs 9 3 + sum_coeffs 10 3 +
               sum_coeffs 11 3 + sum_coeffs 12 3 +
               sum_coeffs 13 3 + sum_coeffs 14 3 +
               sum_coeffs 15 3 + sum_coeffs 16 3 +
               sum_coeffs 17 3 + sum_coeffs 18 3 +
               sum_coeffs 19 3) 

  in sum = -4840 := 
  by sorry

end coeff_x3_in_sum_l378_378582


namespace evaluate_sum_of_ceilings_l378_378220

theorem evaluate_sum_of_ceilings : 
  ⌈Real.sqrt (16 / 9)⌉ + ⌈ 16 / 9 ⌉ + ⌈(16 / 9) ^ 2⌉ + ⌈Real.cbrt (16 / 9)⌉ = 10 :=
by
  sorry

end evaluate_sum_of_ceilings_l378_378220


namespace sum_2501_terms_l378_378951

variable (b : ℕ → ℤ)

noncomputable def sum_seq (n : ℕ) : ℤ :=
  ∑ i in finset.range n, b (i + 1)

axiom seq_rule : ∀ n ≥ 3, b n = b (n - 1) - b (n - 2)
axiom sum_2007 : sum_seq b 2007 = 1522
axiom sum_1522 : sum_seq b 1522 = 2007

theorem sum_2501_terms : sum_seq b 2501 = 1522 :=
by sorry

end sum_2501_terms_l378_378951


namespace interest_rate_B_to_C_l378_378548

theorem interest_rate_B_to_C
  (P : ℕ)                -- Principal amount
  (r_A : ℚ)              -- Interest rate A charges B per annum
  (t : ℕ)                -- Time period in years
  (gain_B : ℚ)           -- Gain of B in 3 years
  (H_P : P = 3500)
  (H_r_A : r_A = 0.10)
  (H_t : t = 3)
  (H_gain_B : gain_B = 315) :
  ∃ R : ℚ, R = 0.13 := 
by
  sorry

end interest_rate_B_to_C_l378_378548


namespace digit_9_occurrences_1_to_800_l378_378339

def count_digit_9 (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (λ acc x => acc + (x.toString.toList.count (λ d => d = '9'))) 0

theorem digit_9_occurrences_1_to_800 : count_digit_9 800 = 160 :=
by
  sorry

end digit_9_occurrences_1_to_800_l378_378339


namespace difference_approx_l378_378042

-- Let L be the larger number and S be the smaller number
variables (L S : ℝ)

-- Conditions given:
-- 1. L is approximately 1542.857
def approx_L : Prop := abs (L - 1542.857) < 1

-- 2. When L is divided by S, quotient is 8 and remainder is 15
def division_condition : Prop := L = 8 * S + 15

-- The theorem stating the difference L - S is approximately 1351.874
theorem difference_approx (hL : approx_L L) (hdiv : division_condition L S) :
  abs ((L - S) - 1351.874) < 1 :=
sorry

#check difference_approx

end difference_approx_l378_378042


namespace possible_values_S3_general_formula_possible_values_general_l378_378286

-- Definitions for sequences and sums
def is_derived_sequence (A a : ℕ → ℝ) := ∀ n > 1, a n = A n ∨ a n = -A n

noncomputable def A (n : ℕ) := 1 / 2^n

noncomputable def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, a (i + 1)

-- Statements to be proven
theorem possible_values_S3 (a : ℕ → ℝ) (h : is_derived_sequence A a) :
  set_of (λ x, x = S a 3) = { 1 / 8, 3 / 8, 5 / 8, 7 / 8 } := sorry

theorem general_formula (a : ℕ → ℝ) (h : is_derived_sequence A a) 
  (hS : ∀ n, S a (3 * n) = 1 / 7 * (1 - 1 / 8^n)) :
  ∀ n, a n = if (∃ k, n = 3 * k - 2) then 1 / 2^n else (-1)^(n + 1) / 2^n := sorry

theorem possible_values_general (a : ℕ → ℝ) (h : is_derived_sequence A a) :
  ∀ (n : ℕ), n > 0 →
  set_of (λ x, x = S a n) = { x | ∃ (m : ℕ), x = (2 * m - 1) / 2^n ∧ m ≤ 2^(n-1) } := sorry

end possible_values_S3_general_formula_possible_values_general_l378_378286


namespace correct_result_l378_378111

-- Define the conditions
variables (x : ℤ)
axiom condition1 : (x - 27 + 19 = 84)

-- Define the goal
theorem correct_result : x - 19 + 27 = 100 :=
  sorry

end correct_result_l378_378111


namespace final_milk_concentration_l378_378531

theorem final_milk_concentration
  (initial_mixture_volume : ℝ)
  (initial_milk_volume : ℝ)
  (replacement_volume : ℝ)
  (replacements_count : ℕ)
  (final_milk_volume : ℝ) :
  initial_mixture_volume = 100 → 
  initial_milk_volume = 36 → 
  replacement_volume = 50 →
  replacements_count = 2 →
  final_milk_volume = 9 →
  (final_milk_volume / initial_mixture_volume * 100) = 9 :=
by
  sorry

end final_milk_concentration_l378_378531


namespace find_f_3_l378_378819

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) : f x + 3 * f (1 - x) = 4 * x ^ 2

theorem find_f_3 : f 3 = 3 / 2 := 
by
  sorry

end find_f_3_l378_378819


namespace inequality1_inequality2_l378_378628

-- Problem (1):
theorem inequality1 (x : ℝ) : 
    3^(2*x - 1) > (1/3)^(x - 2) ↔ x > 1 := sorry

-- Problem (2):
theorem inequality2 (x : ℝ) : 
    3 + log 2 (x - 1) < 2 * log 4 (x + 1) ↔ 1 < x ∧ x < 9/7 := sorry

end inequality1_inequality2_l378_378628


namespace geometric_sequence_a6_l378_378363

-- Definitions and conditions
variable (a : ℕ → ℝ)
variable (a_3 a_9 : ℝ)
variable (geometric_sequence : ∀ n, a (n + 1) = a n * (a 2 / a 1))

-- Given conditions
axiom a_3_root : 3 * a_3^2 - 11 * a_3 + 9 = 0
axiom a_9_root : 3 * a_9^2 - 11 * a_9 + 9 = 0
axiom geo_condition : a 3 = a_3 ∧ a 9 = a_9

-- The problem to prove
theorem geometric_sequence_a6 :
  (a 6 = sqrt 3) ∨ (a 6 = -sqrt 3) :=
sorry

end geometric_sequence_a6_l378_378363


namespace first_10_digits_of_expr_l378_378601

theorem first_10_digits_of_expr :
  let expr := (5 + Real.sqrt 26) ^ 100
  (Real.decimal_digits expr 10) = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9] :=
sorry

end first_10_digits_of_expr_l378_378601


namespace Ali_total_flowers_sold_l378_378568

theorem Ali_total_flowers_sold :
  ∀ (monday tuesday : ℕ), monday = 4 → tuesday = 8 →
  ∀ (friday : ℕ), friday = 2 * monday →
  (monday + tuesday + friday = 20) :=
by
  intros monday tuesday h_monday h_tuesday friday h_friday
  rw [h_monday, h_tuesday, h_friday]
  norm_num

end Ali_total_flowers_sold_l378_378568


namespace correct_similar_statements_one_l378_378842

-- Define the statements on similarity for shapes
def right_triangles_are_similar : Prop :=
  ∀ (a b c d e f : ℕ), right a b c ∧ right d e f → ¬(similar a b c d e f)

def squares_are_similar : Prop :=
  ∀ (a b c d : ℕ), square a b c d → similar a b c d a b c d

def isosceles_triangles_are_similar : Prop :=
  ∀ (a b c d e : ℕ), isosceles a b c ∧ isosceles d e → ¬(similar a b c d e)

def rhombi_are_similar : Prop :=
  ∀ (a b c d e f : ℕ), rhombus a b c d e f → ¬(similar a b c d e f)

-- Count the number of true propositions
def correct_statements_count : ℕ :=
  (if right_triangles_are_similar then 1 else 0) +
  (if squares_are_similar then 1 else 0) +
  (if isosceles_triangles_are_similar then 1 else 0) +
  (if rhombi_are_similar then 1 else 0)

-- The goal statement in Lean
theorem correct_similar_statements_one : correct_statements_count = 1 :=
sorry

end correct_similar_statements_one_l378_378842


namespace max_length_CD_l378_378384

theorem max_length_CD (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 4)
  (h2 : c * cos C * cos (A - B) + c = c * sin^2 C + b * sin A * sin C) :
  ∃ D : ℝ, CD = (2 * sqrt 3) :=
sorry

end max_length_CD_l378_378384


namespace not_possible_when_alpha_70_possible_when_alpha_80_l378_378553

-- Define the primary predicate for angles in a triangle
def angles_in_triangle_sum_to_180 (A B C : ℝ) : Prop :=
  A + B + C = 180

-- Define the condition that all angles are less than a given α
def all_angles_less_than (Ts : List (ℝ × ℝ × ℝ)) (α : ℝ) : Prop :=
  ∀ (T : ℝ × ℝ × ℝ), T ∈ Ts → T.1 < α ∧ T.2 < α ∧ T.3 < α

-- Part (a): Prove that it is impossible when α = 70°
theorem not_possible_when_alpha_70 (Ts : List (ℝ × ℝ × ℝ)) (α : ℝ) :
  α = 70 → (∀ (T : ℝ × ℝ × ℝ), T ∈ Ts → angles_in_triangle_sum_to_180 T.1 T.2 T.3) →
  ¬ all_angles_less_than Ts α :=
by
  intros hα hsum
  sorry

-- Part (b): Prove that it is possible when α = 80°
theorem possible_when_alpha_80 (Ts : List (ℝ × ℝ × ℝ)) (α : ℝ) :
  α = 80 → (∀ (T : ℝ × ℝ × ℝ), T ∈ Ts → angles_in_triangle_sum_to_180 T.1 T.2 T.3) →
  all_angles_less_than Ts α :=
by
  intros hα hsum
  sorry

end not_possible_when_alpha_70_possible_when_alpha_80_l378_378553


namespace smallest_integer_x_l378_378102

-- Conditions
def condition1 (x : ℤ) : Prop := 7 - 5 * x < 25
def condition2 (x : ℤ) : Prop := ∃ y : ℤ, y = 10 ∧ y - 3 * x > 6

-- Statement
theorem smallest_integer_x : ∃ x : ℤ, condition1 x ∧ condition2 x ∧ ∀ z : ℤ, condition1 z ∧ condition2 z → x ≤ z :=
  sorry

end smallest_integer_x_l378_378102


namespace employee_b_payment_l378_378123

-- Definitions based on conditions
def total_payment := 580
def a_to_b_ratio := 1.5

-- The proof problem statement
theorem employee_b_payment (b : ℝ) (a : ℝ) (h1 : a = a_to_b_ratio * b) (h2 : a + b = total_payment) : b = 232 := by
  sorry

end employee_b_payment_l378_378123


namespace Josh_pencils_left_l378_378390

theorem Josh_pencils_left (initial_pencils : ℕ) (given_pencils : ℕ) (remaining_pencils : ℕ) 
  (h_initial : initial_pencils = 142) 
  (h_given : given_pencils = 31) 
  (h_remaining : remaining_pencils = 111) : 
  initial_pencils - given_pencils = remaining_pencils :=
by
  sorry

end Josh_pencils_left_l378_378390


namespace determine_a_b_l378_378305

def M (x : ℝ) : Prop := x^2 - 2012 * x - 2013 > 0

def N (x a b : ℝ) : Prop := x^2 + a * x + b ≤ 0

theorem determine_a_b (a b : ℝ) (h1 : ∀ x, M x ∨ N x a b) (h2 : ∀ x, x ∈ (2013,2014] → M x ∧ N x a b) :
  a = -2013 ∧ b = -2014 := 
sorry

end determine_a_b_l378_378305


namespace number_of_samples_with_score_less_than_48_l378_378716

noncomputable def find_number_of_samples (mu : ℝ) (σ : ℝ) (n : ℕ) (X : ℝ → ℝ) : ℕ :=
let p := 0.04 in
n * p

theorem number_of_samples_with_score_less_than_48 :
  ∀ (μ : ℝ) (σ : ℝ) (n : ℕ) (p : ℝ),
  (μ = 85) →
  (p = 0.04) →
  (100 * p = 4) →
  find_number_of_samples μ σ n (λ x, x) = 4 :=
by
  intros μ σ n p hμ hp hcalc
  rw hμ
  rw hp
  rw hcalc
  exact rfl

end number_of_samples_with_score_less_than_48_l378_378716


namespace line_passes_through_fixed_point_l378_378824

theorem line_passes_through_fixed_point (k : ℝ) :
  ∃ x y : ℝ, k * x + 3 * y + k - 9 = 0 ∧ x = -1 ∧ y = 3 :=
by 
  use (-1 : ℝ), (3 : ℝ)
  split
  · ring
  split; trivial

end line_passes_through_fixed_point_l378_378824


namespace xiaoming_statement_incorrect_l378_378906

theorem xiaoming_statement_incorrect (s : ℕ) : 
    let x_h := 3
    let x_m := 6
    let steps_xh := (x_h - 1) * s
    let steps_xm := (x_m - 1) * s
    (steps_xm ≠ 2 * steps_xh) :=
by
  let x_h := 3
  let x_m := 6
  let steps_xh := (x_h - 1) * s
  let steps_xm := (x_m - 1) * s
  sorry

end xiaoming_statement_incorrect_l378_378906


namespace log_base_frac_l378_378230

theorem log_base_frac (x : ℝ) : log (1/4) 16 = x → x = -2 := by
  sorry

end log_base_frac_l378_378230


namespace example_function_not_power_function_l378_378471

-- Definition of a power function
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the function y = 2x^(1/2)
def example_function (x : ℝ) : ℝ :=
  2 * x ^ (1 / 2)

-- The statement we want to prove
theorem example_function_not_power_function : ¬ is_power_function example_function := by
  sorry

end example_function_not_power_function_l378_378471


namespace decompose_zero_l378_378596

theorem decompose_zero (a : ℤ) : 0 = 0 * a := by
  sorry

end decompose_zero_l378_378596


namespace find_lambda_l378_378031

theorem find_lambda (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ)
  (h_geom : ∀ n, a n = a 0 * (3 ^ n)) 
  (h_sum : ∀ n, S n = 3 ^ (n - 1) + λ) 
  (h_a1 : a 0 = 2 / 3): 
  λ = -1 / 3 :=
by 
  sorry

end find_lambda_l378_378031


namespace spherical_to_rectangular_l378_378595

theorem spherical_to_rectangular :
  ∀ (ρ θ φ: ℝ), ρ = 15 ∧ θ = 5 * Real.pi / 4 ∧ φ = Real.pi / 6 →
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) =
  (-15 * Real.sqrt 2 / 4, -15 * Real.sqrt 2 / 4, 15 * Real.sqrt 3 / 2) :=
by
  intros ρ θ φ h
  cases h
  cases h_left
  rw [h_left, h_right, h_left]
  sorry

end spherical_to_rectangular_l378_378595


namespace gumball_sharing_l378_378388

theorem gumball_sharing (init_j : ℕ) (init_jq : ℕ) (mult_j : ℕ) (mult_jq : ℕ) :
  init_j = 40 → init_jq = 60 → mult_j = 5 → mult_jq = 3 →
  (init_j + mult_j * init_j + init_jq + mult_jq * init_jq) / 2 = 240 :=
by
  intros h1 h2 h3 h4
  sorry

end gumball_sharing_l378_378388


namespace solve_for_x_l378_378621

theorem solve_for_x (x : ℝ) (h : x > 6) : 
  (sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) ↔ x ≥ 18 :=
by
  sorry

end solve_for_x_l378_378621


namespace ratatouille_cost_per_quart_l378_378020

theorem ratatouille_cost_per_quart:
  let eggplants_pounds := 5
  let eggplants_cost_per_pound := 2.00
  let zucchini_pounds := 4
  let zucchini_cost_per_pound := 2.00
  let tomatoes_pounds := 4
  let tomatoes_cost_per_pound := 3.50
  let onions_pounds := 3
  let onions_cost_per_pound := 1.00
  let basil_pounds := 1
  let basil_cost_per_half_pound := 2.50
  let total_quarts := 4
  let eggplants_cost := eggplants_pounds * eggplants_cost_per_pound
  let zucchini_cost := zucchini_pounds * zucchini_cost_per_pound
  let tomatoes_cost := tomatoes_pounds * tomatoes_cost_per_pound
  let onions_cost := onions_pounds * onions_cost_per_pound
  let basil_cost := basil_pounds * (basil_cost_per_half_pound / 0.5)
  let total_cost := eggplants_cost + zucchini_cost + tomatoes_cost + onions_cost + basil_cost
  let cost_per_quart := total_cost / total_quarts
  cost_per_quart = 10.00 :=
  by
    sorry

end ratatouille_cost_per_quart_l378_378020


namespace probability_two_roads_at_least_5_miles_long_l378_378197

-- Probabilities of roads being at least 5 miles long
def prob_A_B := 3 / 4
def prob_B_C := 2 / 3
def prob_C_D := 1 / 2

-- Theorem: Probability of at least two roads being at least 5 miles long
theorem probability_two_roads_at_least_5_miles_long :
  prob_A_B * prob_B_C * (1 - prob_C_D) +
  prob_A_B * prob_C_D * (1 - prob_B_C) +
  (1 - prob_A_B) * prob_B_C * prob_C_D +
  prob_A_B * prob_B_C * prob_C_D = 11 / 24 := 
by
  sorry -- Proof goes here

end probability_two_roads_at_least_5_miles_long_l378_378197


namespace inclination_angle_of_line_l378_378821

-- Define the equation of the line
def line_eq (x : ℝ) : Prop := x = sqrt 3

-- Define the inclination angle in degrees
def inclination_angle (θ : ℝ) := θ

-- Problem statement: The inclination angle θ of the line x - sqrt(3) = 0 is 90 degrees.
theorem inclination_angle_of_line : inclination_angle 90 = 90 := 
by sorry

end inclination_angle_of_line_l378_378821


namespace fourth_person_height_l378_378495

theorem fourth_person_height 
  (h : ℝ)
  (height_average : (h + (h + 2) + (h + 4) + (h + 10)) / 4 = 79)
  : h + 10 = 85 := 
by
  sorry

end fourth_person_height_l378_378495


namespace find_larger_number_l378_378014

theorem find_larger_number
  (x y : ℝ)
  (h1 : y = 2 * x + 3)
  (h2 : x + y = 27)
  : y = 19 :=
by
  sorry

end find_larger_number_l378_378014


namespace sqrt_ineq_l378_378771

variable (a b : ℝ)
variable (h_a_pos : 0 < a)
variable (h_b_pos : 0 < b)

theorem sqrt_ineq (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
    sqrt (a^2 - a * b + b^2) ≥ (a + b) / 2 :=
sorry

end sqrt_ineq_l378_378771


namespace total_households_l378_378365

def households_no_car_bike := 11
def households_both_car_bike := 16
def households_car := 44
def households_bike_only := 35

theorem total_households : 
  let households_car_only := households_car - households_both_car_bike in
  let H := households_no_car_bike + households_car_only + households_bike_only + households_both_car_bike in
  H = 90 :=
by
  let households_car_only := households_car - households_both_car_bike
  have H := households_no_car_bike + households_car_only + households_bike_only + households_both_car_bike
  show H = 90
  sorry

end total_households_l378_378365


namespace cube_root_27_minus_2_l378_378127

theorem cube_root_27_minus_2 : (∛ 27) - 2 = 1 := by
  sorry

end cube_root_27_minus_2_l378_378127


namespace syrup_per_cone_is_six_l378_378549

-- Definitions based on given conditions
def shake_syrup := 4 -- ounces of syrup per shake
def num_shakes := 2 -- number of shakes sold
def total_syrup := 14 -- total ounces of syrup used
def num_cones := 1 -- number of cones sold

-- Question: Prove that the chocolate syrup used per cone is 6 ounces
theorem syrup_per_cone_is_six
    (h1 : shake_syrup = 4)
    (h2 : num_shakes = 2)
    (h3 : total_syrup = 14)
    (h4 : num_cones = 1):
    let x := total_syrup - (num_shakes * shake_syrup) in
    x = 6 :=
by
    -- proof is omitted here
    sorry

end syrup_per_cone_is_six_l378_378549


namespace concentric_circles_lattice_points_l378_378793

theorem concentric_circles_lattice_points :
  ∃ P : ℝ × ℝ, ∀ (A B : ℤ × ℤ), A ≠ B →
    let dist (C D : ℝ × ℝ) := (C.1 - D.1)^2 + (C.2 - D.2)^2 in
    dist (P.1, P.2) (A.1, A.2) ≠ dist (P.1, P.2) (B.1, B.2) :=
begin
  sorry
end

end concentric_circles_lattice_points_l378_378793


namespace compare_real_numbers_l378_378686

open Real

theorem compare_real_numbers (a b c : ℝ) (h1 : a = 2^(ln 2))
                               (h2 : b = 2 + 2*(ln 2))
                               (h3 : c = (ln 2)^2) :
  c < a ∧ a < b :=
by
  sorry

end compare_real_numbers_l378_378686


namespace compare_negatives_l378_378191

theorem compare_negatives : -3 < -2 := 
by { sorry }

end compare_negatives_l378_378191


namespace exists_fraction_for_every_n_no_fraction_for_infinitely_many_n_l378_378129

-- Part (a)
theorem exists_fraction_for_every_n :
  ∀ n : ℕ, ∃ a b : ℤ, 0 < b ∧ b ≤ ⌊real.sqrt (n : ℝ)⌋ + 1 ∧ real.sqrt (n : ℝ) ≤ (a / b) ∧ (a / b) ≤ real.sqrt (n + 1) := sorry

-- Part (b)
theorem no_fraction_for_infinitely_many_n :
  ∃ᶠ n in at_top, ∀ a b : ℤ, ¬ (0 < b ∧ b ≤ ⌊real.sqrt (n : ℝ)⌋ ∧ real.sqrt (n : ℝ) ≤ (a / b) ∧ (a / b) ≤ real.sqrt (n + 1)) := sorry

end exists_fraction_for_every_n_no_fraction_for_infinitely_many_n_l378_378129


namespace operation_example_l378_378210

def operation (a b : ℤ) : ℤ := 2 * a * b - b^2

theorem operation_example : operation 1 (-3) = -15 := by
  sorry

end operation_example_l378_378210


namespace difference_of_two_numbers_l378_378041

-- Definitions as per conditions
def L : ℕ := 1656
def S : ℕ := 273
def quotient : ℕ := 6
def remainder : ℕ := 15

-- Statement of the proof problem
theorem difference_of_two_numbers (h1 : L = 6 * S + 15) : L - S = 1383 :=
by sorry

end difference_of_two_numbers_l378_378041


namespace second_pay_cut_percentage_l378_378749

theorem second_pay_cut_percentage (x : ℝ)
  (h1 : (S : ℝ) > 0)
  (h2 : 0.92 * (1 - x / 100) * 0.82 = 0.648784) :
  x ≈ 13.98 :=
begin
  sorry
end

end second_pay_cut_percentage_l378_378749


namespace smallest_base_10_integer_l378_378884

noncomputable def smallest_integer (a b: ℕ) (h₁: a > 2) (h₂: b > 2) (h₃: n = 2 * a + 1) (h₄: n = b + 2) : ℕ :=
  n

theorem smallest_base_10_integer : smallest_integer 3 5 (by decide) (by decide) (by decide) (by decide) = 7 :=
sorry

end smallest_base_10_integer_l378_378884


namespace geometric_figures_are_axisymmetric_l378_378170

/-- Definitions for the geometric figures and axisymmetry --/
def is_equilateral_triangle (T : Type) := ∃ (a : ℝ), a > 0 ∧ ∀ (P Q : T), dist P Q = a → (Δ P Q) is_equilateral.
def is_square (S : Type) := ∃ (side : ℝ), side > 0 ∧ ∀ (P Q : S), dist P Q = side → (□ P Q) is_square.
def is_parallelogram (P : Type) := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (P Q R S : P), dist P Q = a ∧ dist Q R = b ∧ dist R S = a ∧ dist S P = b → (▀ P Q R S) is_parallelogram.
def is_axisymmetric (F : Type) := ∃ (l : ℝ → ℝ), ∀ (x : F), l x = l (-x).

theorem geometric_figures_are_axisymmetric 
  (T : Type) (S : Type) (P : Type) 
  (hT : is_equilateral_triangle T) 
  (hS : is_square S) 
  (hP : is_parallelogram P) :
  is_axisymmetric T ∧ is_axisymmetric S ∧ is_axisymmetric P := 
by
  sorry

end geometric_figures_are_axisymmetric_l378_378170


namespace certain_amount_is_19_l378_378065

theorem certain_amount_is_19 (x y certain_amount : ℤ) 
  (h1 : x + y = 15)
  (h2 : 3 * x = 5 * y - certain_amount)
  (h3 : x = 7) : 
  certain_amount = 19 :=
by
  sorry

end certain_amount_is_19_l378_378065


namespace triangle_side_sum_is_27_point4_l378_378724

noncomputable def AB : ℝ := 6 * Real.sqrt 3
noncomputable def BC : ℝ := 6 * Real.sqrt 2
noncomputable def AC : ℝ := 6 * Real.sqrt 2
noncomputable def triangle_sum : ℝ := AB + BC

theorem triangle_side_sum_is_27_point4 :
  let A := 60
  let C := 45
  let BC := 6 * Real.sqrt 2
  triangle_sum ≈ 27.4 := 
by
  sorry

end triangle_side_sum_is_27_point4_l378_378724


namespace last_two_digits_sum_of_factorials_l378_378981

theorem last_two_digits_sum_of_factorials :
  let S := finset.sum (finset.range 15) (λ k, if k = 0 then 7! else (7 * k)!)
  in (S % 100) = 40 :=
by
  let S := finset.sum (finset.range 15) (λ k, if k = 0 then 7! else (7 * k)!)
  have h1: ∀ k ≥ 2, (7 * k)! % 100 = 0, 
  { intro k,
    have : 7 * k ≥ 10,
    { linarith },
    sorry },
  have h2: S = 7! + finset.sum (finset.range 14) (λ k, if k = 0 then 0 else (7 * k)!),
  { unfold S,
    sorry },
  have h3: finset.sum (finset.range 14) (λ k, if k = 0 then 0 else (7 * k)!) % 100 = 0,
  { apply finset.sum_eq_zero,
    intros x hx,
    simp only [finset.mem_range] at hx,
    cases x,
    { refl },
    apply h1,
    linarith },
  rw [h2, h3],
  simp,
  sorry

end last_two_digits_sum_of_factorials_l378_378981


namespace one_ton_in_pounds_l378_378431

-- Definitions for the conditions in the problem
def one_pound : ℕ := 16  -- number of ounces 
def packets_weight_pounds : ℕ := 16
def packets_weight_ounces : ℕ := 4
def total_packets : ℕ := 2000
def gunny_bag_capacity : ℕ := 13   -- tons

-- Converting ounces to pounds to check the total weight
noncomputable def oz_to_lb (oz: ℕ) : ℕ :=
  oz / one_pound + if oz % one_pound != 0 then 1 else 0

noncomputable def packet_weight_total_lb (p: ℕ) (oz: ℕ) : ℕ :=
  p + oz_to_lb oz

noncomputable def gunny_bag_total_lb : ℕ :=
  gunny_bag_capacity * 2000

-- Main theorem statement
theorem one_ton_in_pounds : ∃ ton_lb: ℕ, ton_lb = 2000 :=
by
  have packet_lb : ℕ := packet_weight_total_lb packets_weight_pounds packets_weight_ounces
  have total_packet_weight_lb : ℕ := total_packets * packet_lb
  have gunny_bag_capacity_lb : ℕ := gunny_bag_total_lb
  suffices ton_lb_eq : total_packet_weight_lb = gunny_bag_capacity_lb,
  {
    use (gunny_bag_capacity * 2000),
    sorry, -- Proof omitted
  }
  sorry -- Proof omitted

end one_ton_in_pounds_l378_378431


namespace people_left_in_village_l378_378725

-- Defining the initial population
def initial_population : ℕ := 4599

-- Defining the percentage of people who died by bombardment
def died_percentage : ℝ := 0.10

-- Defining the percentage of people who left due to fear
def left_percentage : ℝ := 0.20

-- Defining the function to calculate remaining population
noncomputable def remaining_population (population : ℕ) : ℕ :=
  let died := (died_percentage * population).toInt
  let remained_after_deaths := population - died
  let left := (left_percentage * remained_after_deaths).toInt
  remained_after_deaths - left

-- Proof statement that exactly 3312 people are left in the village
theorem people_left_in_village : remaining_population initial_population = 3312 :=
by
  -- Insert proof here
  sorry

end people_left_in_village_l378_378725


namespace probability_at_least_two_students_succeeding_l378_378923

-- The probabilities of each student succeeding
def p1 : ℚ := 1 / 2
def p2 : ℚ := 1 / 4
def p3 : ℚ := 1 / 5

/-- Calculation of the total probability that at least two out of the three students succeed -/
theorem probability_at_least_two_students_succeeding : 
  (p1 * p2 * (1 - p3)) + (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3) + (p1 * p2 * p3) = 9 / 40 :=
  sorry

end probability_at_least_two_students_succeeding_l378_378923


namespace three_layer_carpet_area_l378_378834

-- Define the dimensions of the carpets and the hall
structure Carpet := (width : ℕ) (height : ℕ)

def principal_carpet : Carpet := ⟨6, 8⟩
def caretaker_carpet : Carpet := ⟨6, 6⟩
def parent_committee_carpet : Carpet := ⟨5, 7⟩
def hall : Carpet := ⟨10, 10⟩

-- Define the area function
def area (c : Carpet) : ℕ := c.width * c.height

-- Prove the area of the part of the hall covered by all three carpets
theorem three_layer_carpet_area : area ⟨3, 2⟩ = 6 :=
by
  sorry

end three_layer_carpet_area_l378_378834


namespace find_a_l378_378641

noncomputable def is_real (z : ℂ) : Prop :=
  z.im = 0

theorem find_a (a : ℝ) (i : ℂ) (h : i = complex.I) :
  is_real ((a - i) / (2 + i)) → a = -2 :=
by
  sorry

end find_a_l378_378641


namespace line_parallel_plane_no_intersection_l378_378706

/-- Let l be a line, α be a plane, and m be another line. 
Given: 
1. l is parallel to α.
2. m is contained within α.
To prove: l and m have no common points. -/
theorem line_parallel_plane_no_intersection (l m : ℝ^3 → Prop) (α : ℝ^3 → Prop) 
  (h1 : ∀ x, l x → ¬ α x) (h2 : ∀ x, m x ↔ α x) :
  ∀ x, ¬ (l x ∧ m x) :=
by 
  sorry

end line_parallel_plane_no_intersection_l378_378706


namespace four_integer_roots_three_integer_roots_l378_378673

theorem four_integer_roots (k : ℤ) :
  (∃ (x1 x2 x3 x4 : ℤ), x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
    (x1^2 - 4 * |x1| + k = 0) ∧ (x2^2 - 4 * |x2| + k = 0) ∧ (x3^2 - 4 * |x3| + k = 0) ∧ 
    (x4^2 - 4 * |x4| + k = 0)) ↔ 
  (k = 3 ∧ ∃ (x1 x2 x3 x4 : ℤ), {x1, x2, x3, x4} = {1, -1, 3, -3}) :=
sorry

theorem three_integer_roots (k : ℤ) :
  (∃ (x1 x2 x3 : ℤ), x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ 
    (x1^2 - 4 * |x1| + k = 0) ∧ (x2^2 - 4 * |x2| + k = 0) ∧ 
    (x3^2 - 4 * |x3| + k = 0)) ↔ 
  (k = 0 ∧ ∃ (x1 x2 x3 : ℤ), {x1, x2, x3} = {0, 4, -4}) :=
sorry

end four_integer_roots_three_integer_roots_l378_378673


namespace sin_angle_EAC_eq_sqrt2_div_2_l378_378756

noncomputable def points := (A : EuclideanSpace ℝ (Fin 3)) (C : EuclideanSpace ℝ (Fin 3)) (E : EuclideanSpace ℝ (Fin 3)) 
  (hA : A = ![0,0,0])
  (hC : C = ![2,2,0])
  (hE : E = ![0,0,2])

theorem sin_angle_EAC_eq_sqrt2_div_2 :
  ∃ (A C E : EuclideanSpace ℝ (Fin 3)), A = ![0, 0, 0] ∧ C = ![2, 2, 0] ∧ E = ![0, 0, 2] ∧
  Real.sin (Real.angle (E - A) (C - A)) = (Real.sqrt 2) / 2 := 
by
  use ![0, 0, 0]
  use ![2, 2, 0]
  use ![0, 0, 2]
  split
  simp
  split
  simp
  split
  simp
  sorry

end sin_angle_EAC_eq_sqrt2_div_2_l378_378756


namespace simple_interest_rate_l378_378975

-- Define the entities and conditions
variables (P A T : ℝ) (R : ℝ)

-- Conditions given in the problem
def principal := P = 12500
def amount := A = 16750
def time := T = 8

-- Result that needs to be proved
def correct_rate := R = 4.25

-- Main statement to be proven: Given the conditions, the rate is 4.25%
theorem simple_interest_rate :
  principal P → amount A → time T → (A - P = (P * R * T) / 100) → correct_rate R :=
by
  intros hP hA hT hSI
  sorry

end simple_interest_rate_l378_378975


namespace sqrt_defined_iff_l378_378353

theorem sqrt_defined_iff (a : ℝ) : (∃ x : ℝ, x = sqrt (a + 1)) ↔ (a ≥ -1) :=
by
  sorry

end sqrt_defined_iff_l378_378353


namespace christine_needs_min_bottles_l378_378189

noncomputable def fluidOuncesToLiters (fl_oz : ℝ) : ℝ := fl_oz / 33.8

noncomputable def litersToMilliliters (liters : ℝ) : ℝ := liters * 1000

noncomputable def bottlesRequired (total_ml : ℝ) (bottle_size_ml : ℝ) : ℕ := 
  Nat.ceil (total_ml / bottle_size_ml)

theorem christine_needs_min_bottles (required_fl_oz : ℝ) (bottle_size_ml : ℝ) (fl_oz_per_l : ℝ) :
  required_fl_oz = 60 →
  bottle_size_ml = 250 →
  fl_oz_per_l = 33.8 →
  bottlesRequired (litersToMilliliters (fluidOuncesToLiters required_fl_oz)) bottle_size_ml = 8 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- We can leave the proof as sorry which signals it's not yet completed.
  sorry

end christine_needs_min_bottles_l378_378189


namespace max_Sm_l378_378687

-- Define recurrence sequence a_n
noncomputable def a : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 3
| 4 := 4
| 5 := 5
| (n + 1) := (a 1) * (a 2) * (a 3) * (a 4) * (a 5) + (∑ i in (finset.range n), a (i + 6)) - 1

-- Define sequence S_m
def S (m : ℕ) : ℕ :=
(a 1 * a 2 * a 3 * a 4 * a 5) - ((∑ i in (finset.range m), (a (i+1))^2))

-- The main theorem to prove
theorem max_Sm : 
  ∃ (m : ℕ), S m = 65 :=
sorry

end max_Sm_l378_378687


namespace eq_positive_root_a_value_l378_378708

theorem eq_positive_root_a_value (x a : ℝ) (hx : x > 0) :
  ((x + a) / (x + 3) - 2 / (x + 3) = 0) → a = 5 :=
by
  sorry

end eq_positive_root_a_value_l378_378708


namespace original_faculty_members_260_l378_378953

-- Definitions for the conditions given in the problem
def original_faculty (x : ℕ) := 0.75 * x
def reduced_faculty := 195
def reduction_percent := 0.25
def remaining_percent := 0.75

-- Proof statement
theorem original_faculty_members_260 (x : ℕ) (h : original_faculty x = reduced_faculty) : x = 260 :=
by sorry

end original_faculty_members_260_l378_378953


namespace greatest_possible_x_l378_378099

theorem greatest_possible_x (x : ℕ) (h : x^4 / x^2 < 18) : x ≤ 4 :=
sorry

end greatest_possible_x_l378_378099


namespace factorize_expression_l378_378611

theorem factorize_expression (a b : ℝ) : a^2 - a * b = a * (a - b) :=
by sorry

end factorize_expression_l378_378611


namespace concert_attendance_problem_l378_378008

theorem concert_attendance_problem:
  (total_tickets sold before_start minutes_after first_song during_middle_part remaining not_go: ℕ) 
  (H1: total_tickets = 900)
  (H2: sold_before_start = (3 * total_tickets) / 4)
  (H3: remaining = total_tickets - sold_before_start)
  (H4: minutes_after_first_song = (5 * remaining) / 9)
  (H5: during_middle_part = 80)
  (H5_remaining: remaining - minutes_after_first_song - during_middle_part = not_go) :
  not_go = 20 :=
sorry

end concert_attendance_problem_l378_378008


namespace new_trailer_homes_added_l378_378081

theorem new_trailer_homes_added
  (n : ℕ) (avg_age_3_years_ago avg_age_today age_increase new_home_age : ℕ) (k : ℕ) :
  n = 30 → avg_age_3_years_ago = 15 → avg_age_today = 12 → age_increase = 3 → new_home_age = 3 →
  (n * (avg_age_3_years_ago + age_increase) + k * new_home_age) / (n + k) = avg_age_today →
  k = 20 :=
by
  intros h_n h_avg_age_3y h_avg_age_today h_age_increase h_new_home_age h_eq
  sorry

end new_trailer_homes_added_l378_378081


namespace car_game_combinations_l378_378106

theorem car_game_combinations : ∃ n, n = 3 * 3 := 
by {
  use 9,
  norm_num -- This is just to show derivation if needed; you can also skip and directly conclude 9 = 9.
}

end car_game_combinations_l378_378106


namespace three_circles_tangent_lie_on_sphere_or_plane_l378_378417

theorem three_circles_tangent_lie_on_sphere_or_plane 
  (k1 k2 k3 : Circle) 
  (tangent_k1_k2 : Tangent k1 k2) 
  (tangent_k2_k3 : Tangent k2 k3) 
  (tangent_k3_k1 : Tangent k3 k1) 
  (distinct_tangencies : DistinctTangencies k1 k2 k3) : 
  (ExistsPlaneContainingCircles k1 k2 k3) ∨ (ExistsSphereContainingCircles k1 k2 k3) := 
sorry

end three_circles_tangent_lie_on_sphere_or_plane_l378_378417


namespace find_k_l378_378428

theorem find_k (k : ℕ) (a : ℕ) (h1 : a = 100) (h2 : ∀ (r : ℕ) (b : ℕ) (hr : r < a) (hw : b < k), sum_water r > 0 ∧ sum_water b > 0) (h3 : ∀ (r1 r2 : ℕ) (w1 w2 : ℕ), (r1 < a) → (w1 < k) → (selected_together r1 w1) → (selected_together r2 w2) → water r1 = water w1 ∧ water r2 = water w2) : k = 100 :=
sorry

end find_k_l378_378428


namespace log_base_one_fourth_of_sixteen_l378_378268

theorem log_base_one_fourth_of_sixteen : log (1/4) 16 = -2 :=  sorry

end log_base_one_fourth_of_sixteen_l378_378268


namespace notebook_cost_l378_378359

open Nat

theorem notebook_cost
  (s : ℕ) (c : ℕ) (n : ℕ)
  (h_majority : s > 21)
  (h_notebooks : n > 2)
  (h_cost : c > n)
  (h_total : s * c * n = 2773) : c = 103 := 
sorry

end notebook_cost_l378_378359


namespace negation_of_proposition_l378_378826

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0)) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by sorry

end negation_of_proposition_l378_378826


namespace min_fraction_value_l378_378343

noncomputable def min_value_fraction (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h₁ : x^2 + y^2 + z^2 = 1) : ℝ :=
  (z+1)^2 / (2 * x * y * z)

theorem min_fraction_value (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h₁ : x^2 + y^2 + z^2 = 1) :
  min_value_fraction x y z h h₁ = 3 + 2 * Real.sqrt 2 :=
  sorry

end min_fraction_value_l378_378343


namespace linear_system_solution_l378_378594

theorem linear_system_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 7) (h2 : 6 * x - 5 * y = 4) :
  x = 43 / 27 ∧ y = 10 / 9 :=
sorry

end linear_system_solution_l378_378594


namespace C1_general_equation_C2_rectangular_equation_intersection_F_value_l378_378684

-- Definitions based on conditions
def C1_parametric_equations (t : ℝ) : ℝ × ℝ :=
  (1 + 1/2 * t, (sqrt 3) / 2 * t)

def C2_polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 + (sin θ)^2)

def F : ℝ × ℝ := (1, 0)

-- Theorem statements based on the problem
theorem C1_general_equation : ∀ (t x y : ℝ), 
  (x = 1 + 1/2 * t) ∧ (y = (sqrt 3) / 2 * t) → 
  (y = sqrt 3 * (x - 1)) :=
sorry

theorem C2_rectangular_equation : ∀ (ρ θ x y : ℝ), 
  (ρ^2 = 12 / (3 + (sin θ)^2)) ∧ (x = ρ * cos θ) ∧ (y = ρ * sin θ) → 
  (3 * x^2 + 4 * y^2 = 12) :=
sorry

theorem intersection_F_value : ∀ (t1 t2 x1 y1 x2 y2 : ℝ), 
  (x1 = 1 + 1/2 * t1) ∧ (y1 = (sqrt 3) / 2 * t1) ∧ 
  (x2 = 1 + 1/2 * t2) ∧ (y2 = (sqrt 3) / 2 * t2) ∧ 
  (5 * t1^2 + 4 * t1 - 12 = 0) ∧ (5 * t2^2 + 4 * t2 - 12 = 0) ∧ 
  ( (t1 + t2) = -4/5) ∧ (t1 * t2 = -12/5) →
  ( (1 / real.sqrt ((1 - x1)^2 + (0 - y1)^2)) + 
    (1 / real.sqrt ((1 - x2)^2 + (0 - y2)^2)) = 4 / 3) :=
sorry

end C1_general_equation_C2_rectangular_equation_intersection_F_value_l378_378684


namespace simplify_expr_l378_378593

theorem simplify_expr (x y : ℝ) (P Q : ℝ) (hP : P = x^2 + y^2) (hQ : Q = x^2 - y^2) : 
  (P * Q / (P + Q)) + ((P + Q) / (P * Q)) = ((x^4 + y^4) ^ 2) / (2 * x^2 * (x^4 - y^4)) :=
by sorry

end simplify_expr_l378_378593


namespace prove_4x_plus_4_neg_x_prove_expression_simplification_l378_378523

-- Proof Problem I
theorem prove_4x_plus_4_neg_x (x : ℝ) (h : 2^x + 2^(-x) = 5) : 4^x + 4^(-x) = 23 :=
by sorry

-- Proof Problem II
theorem prove_expression_simplification :
  2 * (32 * real.sqrt 3)^6 + (real.sqrt (2 * real.sqrt 2))^(4/3) - 
  4 * (16 / 49)^(-1/2) - 42 * 8^(0.25) + (-2005)^0 = 210 :=
by sorry

end prove_4x_plus_4_neg_x_prove_expression_simplification_l378_378523


namespace perfect_square_polynomial_l378_378617

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2) ↔ (x = -1 ∨ x = 0 ∨ x = 3) :=
sorry

end perfect_square_polynomial_l378_378617


namespace width_of_wall_is_6_l378_378122

-- Definitions of the conditions given in the problem
def height_of_wall (w : ℝ) := 4 * w
def length_of_wall (h : ℝ) := 3 * h
def volume_of_wall (w h l : ℝ) := w * h * l

-- Proof statement that the width of the wall is 6 meters given the conditions
theorem width_of_wall_is_6 :
  ∃ w : ℝ, 
  (height_of_wall w = 4 * w) ∧ 
  (length_of_wall (height_of_wall w) = 3 * (height_of_wall w)) ∧ 
  (volume_of_wall w (height_of_wall w) (length_of_wall (height_of_wall w)) = 10368) ∧ 
  (w = 6) :=
sorry

end width_of_wall_is_6_l378_378122


namespace seats_empty_l378_378924

def number_of_people : ℕ := 532
def total_seats : ℕ := 750

theorem seats_empty (n : ℕ) (m : ℕ) : m - n = 218 := by
  have number_of_people : ℕ := 532
  have total_seats : ℕ := 750
  sorry

end seats_empty_l378_378924


namespace find_line_equation_l378_378920

-- We are given a fixed point A (-3,4)
def A : ℝ × ℝ := (-3, 4)

-- The area of the triangle formed by line l and the coordinate axes is 3
def area_of_triangle (a b : ℝ) : ℝ :=
  (1 / 2) * abs (a * b)

-- The line l passes through point A
def passes_through (m : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a * m.1 + b * m.2 + c = 0 ∧ a * p.1 + b * p.2 + c = 0

-- The line equation should be either 2x + 3y - 6 = 0 or 8x + 3y + 12 = 0
theorem find_line_equation (l : ℝ × ℝ → Prop) 
  (h1 : passes_through (-3, 4) (A)) 
  (h2 : area_of_triangle 6 4 = 3 ∨ area_of_triangle (-8) 4 = 3) :
  l = (λ p, 2 * p.1 + 3 * p.2 - 6 = 0) ∨ l = (λ p, 8 * p.1 + 3 * p.2 + 12 = 0) :=
sorry

end find_line_equation_l378_378920


namespace problem_one_problem_two_problem_three_l378_378103

-- Problem (1)
theorem problem_one (a : ℝ) (h : a < 0) : (a / (a - 1)) > 0 := sorry

-- Problem (2)
theorem problem_two (x : ℝ) (h : x < -1) : 
  (2 / (x^2 - 1)) > (1 / ((x^2 - 2*x + 1) / (x - 1))) := sorry

-- Problem (3)①
def xiao_wang_avg (x y : ℝ) := (2 * x * y) / (x + y)
def xiao_zhang_avg (x y : ℝ) := (x + y) / 2

-- Problem (3)②
theorem problem_three (x y : ℝ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) : 
  xiao_wang_avg x y < xiao_zhang_avg x y := sorry

end problem_one_problem_two_problem_three_l378_378103


namespace list_price_eq_34_l378_378166

theorem list_price_eq_34 (x : ℝ) : x = 34 :=
  assume (Alice_selling_price : x - 15) 
         (Alice_commission : 0.12 * Alice_selling_price) 
         (Bob_selling_price : x - 25) 
         (Bob_commission : 0.25 * Bob_selling_price)
         (commissions_equal : 0.12 * Alice_selling_price = 0.25 * Bob_selling_price),
  by sorry

end list_price_eq_34_l378_378166


namespace range_of_lambda_obtuse_angle_l378_378346

noncomputable def range_of_lambda : Set ℝ := { x | x < -2 } ∪ { x | x > -2 ∧ x < 2 }

theorem range_of_lambda_obtuse_angle (λ : ℝ) :
  let a := (1, 1)
      b := (λ, -2)
  in (a.1 * b.1 + a.2 * b.2 < 0) ∧ (λ ≠ -2) ↔ (λ < 2 ∧ λ ≠ -2) :=
by
  sorry

end range_of_lambda_obtuse_angle_l378_378346


namespace fraction_decomposition_sum_of_series_l378_378656

-- First problem statement
theorem fraction_decomposition (a b c d : ℤ) (x : ℂ) (h : a * d ≠ b * c) :
  ∃ r s : ℂ, (1 / ((a : ℂ) * x + (b : ℂ)) * ((c : ℂ) * x + (d : ℂ)) = (r / ((a: ℂ) * x + (b : ℂ)) + s / ((c : ℂ) * x + (d : ℂ)))) := 
sorry

-- Second problem statement
theorem sum_of_series :
  ((∑ k in Finset.range 1000, (1 : ℚ) / ((3 * k + 1) * (3 * k + 4))) = 1000 / 3001) :=
sorry

end fraction_decomposition_sum_of_series_l378_378656


namespace seven_digit_55_divisibles_l378_378855

theorem seven_digit_55_divisibles (digits : Finset ℕ) (h_digits : digits = {0, 1, 2, 3, 4, 5, 6}) :
  ∃ Nmin Nmax : ℕ, 
  (∀ n ∈ digits.powerset.filter (λ s, s.card = 7),  
    let N := (s.to_list.permutations.map (λ l, l.foldl (λ a i, 10 * a + i) 0)).filter (λ x, x % 55 = 0) in
    N.min' (N.to_finset) = Nmin ∧
    N.max' (N.to_finset) = Nmax)
  ∧ Nmin = 1042635 ∧ Nmax = 6431205 :=
by
  sorry

end seven_digit_55_divisibles_l378_378855


namespace distance_between_foci_of_ellipse_l378_378623

theorem distance_between_foci_of_ellipse :
  ∀ (x y : ℝ),
  x^2 + 9 * y^2 = 144 →  
  ∃ c : ℝ, 2 * c = 16 * Real.sqrt 2 :=
by
  intros x y h,
  sorry

end distance_between_foci_of_ellipse_l378_378623


namespace three_times_value_intervals_correctness_l378_378044

open Function

theorem three_times_value_intervals_correctness :
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 1/3 ∧ (∀ x, a ≤ x → x ≤ b → IsDecreasing (λ x, x⁻¹))) ∨
  (∀ a b : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ a^2 = 3 * a ∧ b^2 = 3 * b ∧ (∀ x, a ≤ x → x ≤ b → (λ x, x^2)')) :=
sorry

end three_times_value_intervals_correctness_l378_378044


namespace angle_CNB_66_l378_378850

open EuclideanGeometry

/-- The Lean statement for the problem given the conditions. -/
theorem angle_CNB_66 
  (A B C N : Point) 
  (hABC : isosceles_triangle A B C) 
  (hACB : ∠ A C B = 108) 
  (hN_in_interior : interior_angle ABC N)
  (hNAC : ∠ N A C = 9) 
  (hNCA : ∠ N C A = 21) : 
  ∠ C N B = 66 :=
sorry

end angle_CNB_66_l378_378850


namespace find_missing_student_arithmetic_sequence_l378_378718

theorem find_missing_student_arithmetic_sequence :
  ∃ x d : ℕ, d > 0 ∧ 6 + d < 52 ∧ 6 + 2 * d < 52 ∧ 6 + 3 * d < 52 ∧ x = 6 + k * d ∧ x ∉ {32, 45} → 
  (6, 32, 45, x).perm [a, b, c, d] ∧ a + d = 2b ∧ b + d = 2c :=
sorry

end find_missing_student_arithmetic_sequence_l378_378718


namespace hexagon_perimeter_invariant_l378_378394

-- Definitions for points and segments
variable {P : Type} [inner_product_space ℝ P]

-- Define the quadrilateral and positions
variables (A B C D M N P Q : P)
variables (d : ℝ)
variables (h_quad_eq : dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A)
variables (h_perp_MN_BD : orthogonal (M - (M + N) / 2) (D - B))
variables (h_perp_PQ_BD : orthogonal (P - (P + Q) / 2) (D - B))
variables (h_dist_d : dist ((M + N) / 2) ((P + Q) / 2) = d)

-- The statement of the proof problem
theorem hexagon_perimeter_invariant :
  d > (dist B D / 2) → dist A M + dist M N + dist N C + dist C Q + dist Q P + dist P A = 
    dist A B + dist B C + 2 * d :=
sorry

end hexagon_perimeter_invariant_l378_378394


namespace min_length_of_AB_l378_378304

variable {a b : ℝ} 
variable (x1 y1 x2 y2 : ℝ)

noncomputable def ellipse (x y a b : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def perp (x1 y1 x2 y2 : ℝ) : Prop := 
  x1 * x2 + y1 * y2 = 0

theorem min_length_of_AB (ha : 0 < a) (hb : 0 < b) (hab : a > b) 
                         (on_ellipse_A : ellipse x1 y1 a b) 
                         (on_ellipse_B : ellipse x2 y2 a b)
                         (perpendicular : perp x1 y1 x2 y2) : 
  let AB := sqrt ((x1 - x2)^2 + (y1 - y2)^2) in
  AB = (2 * a * b * sqrt (a^2 + b^2)) / (a^2 + b^2) := sorry

end min_length_of_AB_l378_378304


namespace average_score_last_3_matches_l378_378464

theorem average_score_last_3_matches 
  (a b : ℕ) 
  (avg_first_2 : ℕ) 
  (avg_all_5 : ℕ) 
  (h_avg_first_2 : avg_first_2 = 20) 
  (h_avg_all_5 : avg_all_5 = 26) 
  (h_total_matches : a = 2) 
  (h_last_matches : b = 3) 
  (total_matches : a + b = 5) :
  let total_score_first_2 := a * avg_first_2 in
  let total_score_all_5 := (a + b) * avg_all_5 in
  let total_score_last_3 := total_score_all_5 - total_score_first_2 in
  let avg_last_3 := total_score_last_3 / b in
  avg_last_3 = 30 :=
by
  sorry

end average_score_last_3_matches_l378_378464


namespace simplify_expression_l378_378024

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := 
by 
  sorry

end simplify_expression_l378_378024


namespace min_cost_l378_378898

-- Definitions based on the problem conditions
def V_box : ℝ := 20 * 20 * 15 -- volume of one box in cubic inches
def V_total : ℝ := 3_060_000 -- total volume required in cubic inches
def P_box : ℝ := 0.70 -- price per box in dollars

-- Compute number of boxes needed
def N_box : ℝ := (V_total / V_box).ceil

-- Compute total cost
def Cost : ℝ := N_box * P_box

-- The theorem statement proving the minimum cost
theorem min_cost : Cost = 357 := sorry

end min_cost_l378_378898


namespace right_handed_players_total_l378_378517

theorem right_handed_players_total (total throwers : ℕ) (one_third_left_handed : ℕ) (right_handed_players : ℕ) : 
  total = 70 → 
  throwers = 31 → 
  nonThrowers = total - throwers → 
  leftHandedNonThrowers = nonThrowers / 3 → 
  right_handed_players = throwers + (nonThrowers - leftHandedNonThrowers) → 
  right_handed_players = 57 :=
by 
  intros total_eq seventy_eq thirtyone_eq nonThrowers_eq leftHandedNonThrowers_eq rightHandedPlayers_eq
  sorry

end right_handed_players_total_l378_378517


namespace min_socks_for_15_pairs_l378_378161

theorem min_socks_for_15_pairs
  (P O Y G : ℕ)
  (total_socks : P + O + Y + G)
  (H1 : P = 150)
  (H2 : O = 120)
  (H3 : Y = 90)
  (H4 : G = 70) :
  ∃ n : ℕ, (n ≥ 33) ∧ (∀ selection : ℕ → ℕ, 
  (∀ i : ℕ, selection i ∈ {1, 2, 3, 4}) → 
  (∃ count : ℕ, count ≥ 15 ∧ count = number_of_pairs_selected selection n) := sorry


end min_socks_for_15_pairs_l378_378161


namespace christine_needs_min_bottles_l378_378188

noncomputable def fluidOuncesToLiters (fl_oz : ℝ) : ℝ := fl_oz / 33.8

noncomputable def litersToMilliliters (liters : ℝ) : ℝ := liters * 1000

noncomputable def bottlesRequired (total_ml : ℝ) (bottle_size_ml : ℝ) : ℕ := 
  Nat.ceil (total_ml / bottle_size_ml)

theorem christine_needs_min_bottles (required_fl_oz : ℝ) (bottle_size_ml : ℝ) (fl_oz_per_l : ℝ) :
  required_fl_oz = 60 →
  bottle_size_ml = 250 →
  fl_oz_per_l = 33.8 →
  bottlesRequired (litersToMilliliters (fluidOuncesToLiters required_fl_oz)) bottle_size_ml = 8 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- We can leave the proof as sorry which signals it's not yet completed.
  sorry

end christine_needs_min_bottles_l378_378188


namespace point_A_not_on_transformed_plane_l378_378405

-- Define the original plane equation as a function
def plane (x y z : ℝ) : ℝ := 2 * x - y + 3 * z - 1

-- Define the transformed plane equation as a function
def transformed_plane (x y z : ℝ) (k : ℝ) : ℝ := 2 * x - y + 3 * z - k

-- Given point A and scale factor k
def point_A := (0 : ℝ, 3 : ℝ, -1 : ℝ)
def k := 2

-- Proof statement: Point A does not lie on the transformed plane with k = 2
theorem point_A_not_on_transformed_plane : 
  transformed_plane point_A.1 point_A.2 point_A.3 k ≠ 0 := by
  sorry

end point_A_not_on_transformed_plane_l378_378405


namespace remainder_of_12345678_mod_9_l378_378101

theorem remainder_of_12345678_mod_9 : 
  (12345678 % 9) = 0 := 
begin
  sorry
end

end remainder_of_12345678_mod_9_l378_378101


namespace find_z_value_l378_378804

noncomputable def y_varies_inversely_with_z (y z : ℝ) (k : ℝ) : Prop :=
  (y^4 * z^(1/4) = k)

theorem find_z_value (y z : ℝ) (k : ℝ) : 
  y_varies_inversely_with_z y z k → 
  y_varies_inversely_with_z 3 16 162 → 
  k = 162 →
  y = 6 → 
  z = 1 / 4096 := 
by 
  sorry

end find_z_value_l378_378804


namespace average_speed_BC_is_40_l378_378551

-- all conditions translated into Lean 4 definitions
def distance_AB : ℝ := 120
def total_distance : ℝ := 180
def total_average_speed : ℝ := 30
def distance_BC : ℝ := distance_AB / 2
def time_AB (t : ℝ) : ℝ := t
def time_BC (t : ℝ) : ℝ := t / 3
def total_time (t : ℝ) : ℝ := t + time_BC t
def total_time_corrected (t : ℝ) : ℝ := (4 / 3) * t

-- time taken between A and B is t hours, and corresponding total time corrected
axiom t_value : ℝ
axiom t_is_derived : total_average_speed = total_distance / total_time_corrected t

-- final statement to be proven
theorem average_speed_BC_is_40 :
  ∃ t_value, 
  let average_speed_BC := distance_BC / time_BC t_value in
  average_speed_BC = 40 := 
by
  sorry

end average_speed_BC_is_40_l378_378551


namespace count_integers_satisfying_abs_inequality_l378_378692

theorem count_integers_satisfying_abs_inequality :
  {x : ℤ | |3 * x + 1| ≤ 10}.to_finset.card = 6 :=
by
  sorry

end count_integers_satisfying_abs_inequality_l378_378692


namespace number_of_correct_propositions_l378_378795

variables (a b : Type) (M N : a → Prop)

axiom parallel : ∀ (x y : Type) (P : x → Prop), (P x ∧ P y) → (x = y)
axiom perpendicular : ∀ (x y : Type) (P : x → Prop), (P x ∧ P y) → (¬ (x = y))

theorem number_of_correct_propositions :
  (¬ (forall (a b M : Type), (parallel a M ∧ parallel b M) → (parallel a b))) ∧
  (forall (a b M : Type), (parallel a M ∧ perpendicular b M) → (perpendicular a b)) ∧
  (¬ (forall (a b M : Type), (parallel a b ∧ parallel b M) → (parallel a M))) ∧
  (forall (a M N : Type), (perpendicular a M ∧ parallel a N) → (perpendicular M N)) →
  2 = (2 : ℕ) :=
by sorry

end number_of_correct_propositions_l378_378795


namespace fractions_ordering_l378_378095

theorem fractions_ordering :
  (8 / 25 : ℚ) < (6 / 17) ∧ (6 / 17 : ℚ) < (11 / 29) :=
by {
  have h1 : (8 / 25 : ℚ) < (6 / 17),
  {
    calc (8 / 25 : ℚ) = (8 * 17) / (25 * 17) : by rw [div_eq_div_one_div, one_div_mul_cancel, one_div_mul_cancel]
    ... = (8 * 17 / (25 * 17)) : by norm_num
    ... = (8 * 17 < 6 * 25) / (6 * 25) : by norm_num,
    exact_mod_cast (by norm_num : 232 < 275)
  },
  have h2 : (6 / 17 : ℚ) < (11 / 29),
  {
    calc (6 / 17) = (6 * 29) / (17 * 29) : by rw [div_eq_div_one_div, one_div_mul_cancel, one_div_mul_cancel]
    ... = (6 * 29 / (17 * 29)) : by norm_num
    ... = (6 * 29 < 11 * 17) / (11 * 17) : by norm_num,
    exact_mod_cast (by norm_num : 174 < 187)
  },
  exact ⟨h1, h2⟩,
}

end fractions_ordering_l378_378095


namespace new_median_is_five_l378_378540

/-- Given a collection of six positive integers has a mean of 5.5, a unique mode of 4, and a median of 5,
    and adding a 10 to the collection, prove that the new median is equal to 5. --/
theorem new_median_is_five (a b c d e f : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0)
  (h7 : (a + b + c + d + e + f) / 6 = 5.5)
  (h8 : ∃ x, (∃ y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ ∀ p, (p = x ∨ p = y ∨ p = z) → p = 4) 
    ∧ ∀ q, (q ≠ 4 → (a = q ∨ b = q ∨ c = q ∨ d = q ∨ e = q ∨ f = q) → q ≠ 4)) 
  (h9 : {a, b, c, d, e, f}.toFinset.sort (≤).get! 2 = 5 ∧ {a, b, c, d, e, f}.toFinset.sort (≤).get! 3 = 5) :
  ∃ (g : ℕ), g = 10 ∧ ({a, b, c, d, e, f, g}.toFinset.sort (≤).get! 3 = 5) :=
by
  sorry

end new_median_is_five_l378_378540


namespace difference_of_coordinates_l378_378015

-- Define point and its properties in Lean.
structure Point where
  x : ℝ
  y : ℝ

-- Define the midpoint property.
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Given points A and M
def A : Point := {x := 8, y := 0}
def M : Point := {x := 4, y := 1}

-- Assume B is a point with coordinates x and y
variable (B : Point)

-- The theorem to prove.
theorem difference_of_coordinates :
  is_midpoint M A B → B.x - B.y = -2 :=
by
  sorry

end difference_of_coordinates_l378_378015


namespace nina_running_distance_l378_378424

theorem nina_running_distance (x : ℝ) (hx : 2 * x + 0.67 = 0.83) : x = 0.08 := by
  sorry

end nina_running_distance_l378_378424


namespace tangent_line_circumcircle_l378_378410

-- Let ABC be a triangle with angle bisector of ∠BAC intersecting BC at D
-- and the circumcircle of triangle ABC at S. Prove that line SB is tangent
-- to the circumcircle of triangle ABD.

theorem tangent_line_circumcircle
    (A B C D S : Point)
    (h_triangle : Triangle A B C)
    (h_angle_bisector_1 : AngleBisector (∠ BAC) A (LineSegment B C) D)
    (h_angle_bisector_2 : AngleBisector (∠ BAC) A (Circumcircle A B C) S) :
    Tangent (Line S B) (Circumcircle A B D) := 
sorry

end tangent_line_circumcircle_l378_378410


namespace find_two_numbers_l378_378292

/-
Given \( n \) distinct natural numbers where \( n \geq 4 \), prove that there exist two numbers such that 
neither their sum nor the absolute value of their difference appears among the remaining numbers.
-/

theorem find_two_numbers 
  (n : ℕ) (h : n ≥ 4) (nums : Finset ℕ) (hn : nums.card = n) 
  (h_distinct : ∀ a b : ℕ, a ∈ nums → b ∈ nums → a ≠ b → a ≠ b) :
  ∃ (a b : ℕ), a ∈ nums ∧ b ∈ nums ∧ a ≠ b ∧ 
    (¬((a + b) ∈ nums) ∧ ¬((a - b).nat_abs ∈ nums)) :=
begin
  -- skip the proof as per the instruction
  sorry
end

end find_two_numbers_l378_378292


namespace full_size_mustang_length_l378_378798

theorem full_size_mustang_length 
  (smallest_model_length : ℕ)
  (mid_size_factor : ℕ)
  (full_size_factor : ℕ)
  (h1 : smallest_model_length = 12)
  (h2 : mid_size_factor = 2)
  (h3 : full_size_factor = 10) :
  (smallest_model_length * mid_size_factor) * full_size_factor = 240 := 
sorry

end full_size_mustang_length_l378_378798


namespace minimum_rows_required_l378_378076

theorem minimum_rows_required (total_students : ℕ) (max_students_per_school : ℕ) (seats_per_row : ℕ) (num_schools : ℕ) 
    (h_total_students : total_students = 2016) 
    (h_max_students_per_school : max_students_per_school = 45) 
    (h_seats_per_row : seats_per_row = 168) 
    (h_num_schools : num_schools = 46) : 
    ∃ (min_rows : ℕ), min_rows = 16 := 
by 
  -- Proof omitted
  sorry

end minimum_rows_required_l378_378076


namespace least_number_subtracted_divisible_by_15_l378_378511

-- Define the problem
def n : ℕ := 9679
def remainder (a b : ℕ) : ℕ := a % b

-- The main statement to prove
theorem least_number_subtracted_divisible_by_15 : ∃ x : ℕ, x = remainder n 15 ∧ (n - x) % 15 = 0 := by
  have h : remainder n 15 = 4 := by sorry
  use 4
  constructor
  · exact h
  · rw [h]
    exact Nat.mod_sub_self_right n 15 (Nat.le_of_lt (Nat.lt_of_le_of_ne (Nat.mod_lt n (Nat.zero_lt_bit1 3)) (fun h => by cases h)))

end least_number_subtracted_divisible_by_15_l378_378511


namespace barry_shirt_discount_l378_378174

theorem barry_shirt_discount 
  (original_price : ℤ) 
  (discount_percent : ℤ) 
  (discounted_price : ℤ) 
  (h1 : original_price = 80) 
  (h2 : discount_percent = 15)
  (h3 : discounted_price = original_price - (discount_percent * original_price / 100)) : 
  discounted_price = 68 :=
sorry

end barry_shirt_discount_l378_378174


namespace pythagorean_theorem_depends_on_parallel_postulate_l378_378527

theorem pythagorean_theorem_depends_on_parallel_postulate
  (similarity_theorems : Prop)
  (area_equivalence_theorems : Prop)
  (euclidean_geometry : Prop)
  (parallel_postulate : Prop)
  (H1 : euclidean_geometry -> parallel_postulate)
  (H2 : similarity_theorems -> euclidean_geometry)
  (H3 : area_equivalence_theorems -> euclidean_geometry) :
  (direct_proof_of_pythagorean : Prop)
  (pythagorean_theorem : similarity_theorems ∨ area_equivalence_theorems)
  (H4 : pythagorean_theorem -> direct_proof_of_pythagorean) :
  direct_proof_of_pythagorean -> parallel_postulate :=
sorry

end pythagorean_theorem_depends_on_parallel_postulate_l378_378527


namespace incenters_and_excenters_concyclic_l378_378124

-- Definitions of points and cycles
variables {A B C D O I₁ I₂ I₃ I₄ J₁ J₂ J₃ J₄ : Type*}

-- Define incenters and excenters
variables [IsConvexQuad A B C D] [IntersectsAt AC BD O]
          [Incenter A O B I₁] [Incenter B O C I₂]
          [Incenter C O D I₃] [Incenter D O A I₄]
          [Excenter A O B J₁] [Excenter B O C J₂]
          [Excenter C O D J₃] [Excenter D O A J₄]

-- Statement of the theorem to be proved
theorem incenters_and_excenters_concyclic :
  Concyclic I₁ I₂ I₃ I₄ ↔ Concyclic J₁ J₂ J₃ J₄ :=
sorry

end incenters_and_excenters_concyclic_l378_378124


namespace max_sum_on_chessboard_l378_378943

theorem max_sum_on_chessboard (x y : Finset ℕ) (x_sum y_sum : ∀ i : ℕ, i ∈ (Finset.range 10) → ℕ)
  (h1 : x.card = 33) (h2 : y.card = 33)
  (h3 : Finset.sum (Finset.image x_sum (Finset.range 10)) = 33)
  (h4 : Finset.sum (Finset.image y_sum (Finset.range 10)) = 33) :
  660 - ∑ i in Finset.range 10, x_sum i ^ 2 - ∑ i in Finset.range 10, y_sum i ^ 2 = 438 :=
by
  sorry

end max_sum_on_chessboard_l378_378943


namespace problem1_problem2_l378_378308

-- Define the conditions
variables {α β : ℝ}
variable h1 : cos (α + β) = 2 * sqrt 5 / 5
variable h2 : tan β = 1 / 7
variable h3 : 0 < α ∧ α < π / 2
variable h4 : 0 < β ∧ β < π / 2

-- Problem (1): Prove the value of cos 2β + sin 2β - sin β cos β
theorem problem1 : cos (2 * β) + sin (2 * β) - sin β * cos β = 11 / 10 :=
by
  sorry

-- Problem (2): Prove the value of 2α + β
theorem problem2 : 2 * α + β = π / 4 :=
by
  sorry

end problem1_problem2_l378_378308


namespace trajectory_circle_equation_l378_378816

theorem trajectory_circle_equation :
  (∀ (x y : ℝ), dist (x, y) (0, 0) = 4 ↔ x^2 + y^2 = 16) :=  
sorry

end trajectory_circle_equation_l378_378816


namespace sum_of_cubes_l378_378895

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
sorry

end sum_of_cubes_l378_378895


namespace law_of_large_numbers_l378_378105

-- We declare a noncomputable section because we are dealing with real numbers and probabilities.
noncomputable section

-- Define the probability of getting heads for a fair coin.
def prob_heads := 0.5

-- The statement we need to prove:
theorem law_of_large_numbers :
  ∀ (n : ℕ), n > 0 → (tosses : Fin n → Bool) → 
  (frequency_heads : ℝ) → 
  (frequency_heads = (Finset.univ.filter (λ i, tosses i = true)).card / n) → 
  abs (frequency_heads - prob_heads) < ε :=
by
  sorry

end law_of_large_numbers_l378_378105


namespace g_of_negative_8_l378_378763

def f (x : ℝ) : ℝ := 4 * x - 9
def g (y : ℝ) : ℝ := y^2 + 6 * y - 7

theorem g_of_negative_8 : g (-8) = -87 / 16 :=
by
  -- Proof goes here
  sorry

end g_of_negative_8_l378_378763


namespace range_of_a_l378_378677

def f (a x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (deriv (f a) x) ≤ 0) ↔ -real.sqrt 3 ≤ a ∧ a ≤ real.sqrt 3 :=
by {
  sorry
}

end range_of_a_l378_378677


namespace smallest_sum_l378_378853

theorem smallest_sum (x y : ℕ) (h : (2010 / 2011 : ℚ) < x / y ∧ x / y < (2011 / 2012 : ℚ)) : x + y = 8044 :=
sorry

end smallest_sum_l378_378853


namespace reggie_games_lost_l378_378449

-- Define the necessary conditions
def initial_marbles : ℕ := 100
def bet_per_game : ℕ := 10
def marbles_after_games : ℕ := 90
def total_games : ℕ := 9

-- Define the proof problem statement
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / bet_per_game = 1 := by
  sorry

end reggie_games_lost_l378_378449


namespace general_formula_geo_seq_sum_first_n_terms_geo_seq_l378_378378

-- Definition of the geometric sequence with conditions
def geo_seq (a : ℕ → ℝ) (q : ℝ) :=
  a 4 = 2 / 3 ∧ a 3 + a 5 = 20 / 9

-- First part: Proving the general formula for a_n
theorem general_formula_geo_seq :
  ∃ a : ℕ → ℝ, ∃ q : ℝ, q ≠ 0 ∧ geo_seq a q ∧
  ((q = 1 / 3 → ∀ n, a n = 2 * 3^(3 - n)) ∧
   (q = 3 → ∀ n, a n = 2 * 3^(n - 5))) :=
by sorry

-- Second part: Proving the sum of the first n terms of the sequence b_n
theorem sum_first_n_terms_geo_seq (a : ℕ → ℝ) (q : ℝ) (b : ℕ → ℝ) (n : ℕ) :
  q > 1 → geo_seq a q → (∀ n, a n = 2 * 3^(n - 5)) →
  (∀ n, b n = log 3 (a n / 2)) → 
  (∑ i in range n, b i) = n^2 / 2 - 9 * n / 2 :=
by sorry

end general_formula_geo_seq_sum_first_n_terms_geo_seq_l378_378378


namespace animals_in_field_l378_378933

theorem animals_in_field :
  let dog := 1 in
  let cats := 4 in
  let rabbits_per_cat := 2 in
  let hares_per_rabbit := 3 in
  let total_animals := 
    dog + 
    cats + 
    cats * rabbits_per_cat + 
    (cats * rabbits_per_cat) * hares_per_rabbit 
  in total_animals = 37 := by
sorry

end animals_in_field_l378_378933


namespace total_cost_of_paving_floor_l378_378049

theorem total_cost_of_paving_floor (b1 b2 h : ℝ) (CostA_per_sq : ℝ) (CostB_per_sq : ℝ)
  (AreaA_fraction AreaB_fraction : ℝ)
  (hb1 : b1 = 5.5) (hb2 : b2 = 6.5) (hh : h = 3.75) 
  (hCostA : CostA_per_sq = 350) (hCostB : CostB_per_sq = 450)
  (hAreaA_fraction : AreaA_fraction = 0.6) (hAreaB_fraction : AreaB_fraction = 0.4) :
  let A := 0.5 * (b1 + b2) * h
  let AreaA := AreaA_fraction * A
  let AreaB := AreaB_fraction * A
  let CostA := AreaA * CostA_per_sq
  let CostB := AreaB * CostB_per_sq
  Total :=
  CostA + CostB
  in Total = 8775 := 
by {
  sorry
}

end total_cost_of_paving_floor_l378_378049


namespace perfume_purchase_l378_378985

theorem perfume_purchase :
  let Christian_saved := 7
  let Sue_saved := 9
  let Bob_saved := 3
  let Christian_charge := 7
  let Christian_yards := 7
  let Sue_charge := 4
  let Sue_dogs := 10
  let Bob_charge := 2
  let Bob_jobs := 5
  let perfume_price := 100
  let discount_rate := 20
  let Christian_earned := Christian_charge * Christian_yards
  let Sue_earned := Sue_charge * Sue_dogs
  let Bob_earned := Bob_charge * Bob_jobs
  let total_saved := Christian_saved + Sue_saved + Bob_saved
  let total_earned := Christian_earned + Sue_earned + Bob_earned
  let total_money := total_saved + total_earned
  let discount := perfume_price * discount_rate / 100
  let discounted_price := perfume_price - discount
  let total_cost := discounted_price
  let remaining_money := total_money - total_cost
  in remaining_money = 38 :=
by {
  sorry
}

end perfume_purchase_l378_378985


namespace A_inter_complement_RB_eq_l378_378345

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x^2)}

def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

def complement_RB : Set ℝ := {x | x ≥ 1}

theorem A_inter_complement_RB_eq : A ∩ complement_RB = {x | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end A_inter_complement_RB_eq_l378_378345


namespace age_difference_l378_378468

theorem age_difference (O Y : ℕ) (h₀ : O = 38) (h₁ : Y + O = 74) : O - Y = 2 := by
  sorry

end age_difference_l378_378468


namespace not_equivalent_l378_378905

-- Define the variables
def a := 0.0000375
def b := 3.75 * (10 : ℝ)^(-5)
def c := 37.5 * (10 : ℝ)^(-6)
def d := 375 * (10 : ℝ)^(-7)
def e := (37 / 1000) * (10 : ℝ)^(-5)
def f := (3 / 80000 : ℝ)

-- Prove that e ≠ a
theorem not_equivalent : e ≠ a :=
by
  sorry

end not_equivalent_l378_378905


namespace no_integer_root_of_P_l378_378408

noncomputable def P (x : ℤ) : ℤ := sorry
noncomputable def Q (x : ℤ) : ℤ := P x + 12

theorem no_integer_root_of_P
  (hQ_roots : ∃ a1 a2 a3 a4 a5 a6 : ℤ, ∀ a ∈ [a1, a2, a3, a4, a5, a6], Q a = 0 ∧ list.nodup [a1, a2, a3, a4, a5, a6]) :
  ∀ r : ℤ, P r ≠ 0 := 
by
  sorry

end no_integer_root_of_P_l378_378408


namespace pascal_triangle_sum_l378_378803

theorem pascal_triangle_sum :
  ∑ i in Finset.range (2024), i * (Nat.choose 2023 i / Nat.choose 2024 i) = 1025703 :=
by
  sorry

end pascal_triangle_sum_l378_378803


namespace constant_term_in_expansion_l378_378377

theorem constant_term_in_expansion (x : ℝ) :
  (∑ i in Finset.range (9 + 1), Nat.choose 9 i) = 512 →
  ∑ i in Finset.range (9 + 1), (Nat.choose 9 i) * ((sqrt x)^(9 - i) * ((-1 / x)^i) : ℝ) =
  -84 :=
by
  -- Proof of the theorem can be added here.
  sorry

end constant_term_in_expansion_l378_378377


namespace cheese_pops_count_l378_378171

-- Define the number of hotdogs, chicken nuggets, and total portions
def hotdogs : ℕ := 30
def chicken_nuggets : ℕ := 40
def total_portions : ℕ := 90

-- Define the number of bite-sized cheese pops
def cheese_pops : ℕ := total_portions - hotdogs - chicken_nuggets

-- Theorem to prove that the number of bite-sized cheese pops Andrew brought is 20
theorem cheese_pops_count :
  cheese_pops = 20 :=
by
  -- The following proof is omitted
  sorry

end cheese_pops_count_l378_378171


namespace cube_sum_identity_l378_378896

theorem cube_sum_identity (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end cube_sum_identity_l378_378896


namespace initial_distance_between_trains_l378_378086

theorem initial_distance_between_trains (speedA speedB remaining_distance time_hour : ℝ) (h1 : speedA = 30) (h2 : speedB = 40) (h3 : remaining_distance = 70) (h4 : time_hour = 1) :
  let relative_speed := speedA + speedB in
  let distance_covered := relative_speed * time_hour in
  let initial_distance := distance_covered + remaining_distance in
  initial_distance = 140 :=
by
  sorry

end initial_distance_between_trains_l378_378086


namespace concert_attendance_problem_l378_378007

theorem concert_attendance_problem:
  (total_tickets sold before_start minutes_after first_song during_middle_part remaining not_go: ℕ) 
  (H1: total_tickets = 900)
  (H2: sold_before_start = (3 * total_tickets) / 4)
  (H3: remaining = total_tickets - sold_before_start)
  (H4: minutes_after_first_song = (5 * remaining) / 9)
  (H5: during_middle_part = 80)
  (H5_remaining: remaining - minutes_after_first_song - during_middle_part = not_go) :
  not_go = 20 :=
sorry

end concert_attendance_problem_l378_378007


namespace car_tank_capacity_l378_378578

theorem car_tank_capacity
  (speed : ℝ) (usage_rate : ℝ) (time : ℝ) (used_fraction : ℝ) (distance : ℝ := speed * time) (gallons_used : ℝ := distance / usage_rate) 
  (fuel_used : ℝ := 10) (tank_capacity : ℝ := fuel_used / used_fraction)
  (h1 : speed = 60) (h2 : usage_rate = 30) (h3 : time = 5) (h4 : used_fraction = 0.8333333333333334) : 
  tank_capacity = 12 :=
by
  sorry

end car_tank_capacity_l378_378578


namespace basketball_team_lineup_l378_378432

theorem basketball_team_lineup :
  let 
    total_players : ℕ := 16,
    twins : set ℕ := {1, 2},   -- Assume twins are players 1 and 2
    triplets : set ℕ := {3, 4, 5},  -- Assume triplets are players 3, 4, and 5
    lineup_size : ℕ := 5,
    remaining_players : set ℕ := {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
  in
  ∃ (num_lineups : ℕ), num_lineups = 198 := 
  by
    sorry

end basketball_team_lineup_l378_378432


namespace area_enclosed_by_absolute_value_linear_eq_l378_378856

theorem area_enclosed_by_absolute_value_linear_eq (x y : ℝ) :
  (|5 * x| + |3 * y| = 15) → ∃ (A : ℝ), A = 30 :=
by
  sorry

end area_enclosed_by_absolute_value_linear_eq_l378_378856


namespace visit_each_seat_once_l378_378135

theorem visit_each_seat_once (n : ℕ) (h_n_gt_1 : n > 0) :
  (∃ (f : fin n → fin n), 
    (∀ i, ∃! k, k < n ∧ (f i = i + k.succ) ∧ (f i ≠ i)) ∧ 
    (∀ i j, i ≠ j → f i ≠ f j)) ↔ even n := 
sorry

end visit_each_seat_once_l378_378135


namespace average_age_of_first_7_students_l378_378808

theorem average_age_of_first_7_students (avg_15_students : ℕ) 
 (total_students : ℕ) 
 (avg_second_7_students : ℕ) 
 (total_second_7_students : ℕ) 
 (age_15th_student : ℕ) : 
 avg_15_students = 15 ∧ total_students = 15 ∧ avg_second_7_students = 16 ∧ total_second_7_students = 7 ∧ age_15th_student = 15 → 
 (let total_age_15_students := avg_15_students * total_students in 
  let total_age_second_7_students := avg_second_7_students * total_second_7_students in 
  let total_age_first_7_students := total_age_15_students - total_age_second_7_students - age_15th_student in 
  total_age_first_7_students / total_second_7_students = 14) :=
begin
  intros h,
  rcases h with ⟨h1, h2, h3, h4, h5⟩, 
  have total_age_15_students := h1 * h2, 
  have total_age_second_7_students := h3 * h4,
  have total_age_first_7_students := total_age_15_students - total_age_second_7_students - h5,
  have avg_first_7_students := total_age_first_7_students / h4,
  linarith,
  sorry
end

end average_age_of_first_7_students_l378_378808


namespace arithmetic_sequence_common_difference_l378_378307

theorem arithmetic_sequence_common_difference
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (a₁ d : ℤ)
  (h1 : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2)
  (h2 : ∀ n, a n = a₁ + (n - 1) * d)
  (h3 : S 5 = 5 * (a 4) - 10) :
  d = 2 := sorry

end arithmetic_sequence_common_difference_l378_378307


namespace unique_lines_through_point_l378_378847

-- Define the mathematical objects and relationships
variables {ℝ : Type} [LinearOrderedField ℝ]

-- Definitions for the problem conditions
structure Point (ℝ) :=
  (x : ℝ) (y : ℝ) (z : ℝ)

structure Line (ℝ) :=
  (point : Point ℝ) (direction : Point ℝ)

structure Plane (ℝ) :=
  (point : Point ℝ) (normal : Point ℝ)

-- Distance between two lines in 3D space
def distance_between_lines (l1 l2 : Line ℝ) : ℝ := sorry

-- Definition to check if a line lies in a plane
def line_in_plane (l : Line ℝ) (P : Plane ℝ) : Prop := sorry

-- Predicate to verify all conditions
def satisfies_conditions (A : Point ℝ) (l : Line ℝ) (g : Line ℝ) (d : ℝ) : Prop :=
  ∃ (P : Plane ℝ), line_in_plane g P ∧ P.normal = l.direction ∧ distance_between_lines l g = d ∧ g.point = A

-- The final theorem statement for the proof problem
theorem unique_lines_through_point (A : Point ℝ) (l : Line ℝ) (d : ℝ) :
  ∃! (g : Line ℝ), satisfies_conditions A l g d := sorry

end unique_lines_through_point_l378_378847


namespace log_base_one_fourth_of_sixteen_l378_378255

theorem log_base_one_fourth_of_sixteen :
  log (1 / 4 : ℝ) (16 : ℝ) = -2 :=
sorry

end log_base_one_fourth_of_sixteen_l378_378255


namespace divide_square_5x5_l378_378955

noncomputable def can_divide_square : Prop :=
  ∃ (segments : list segment), 
    (all segments along grid lines of a 5x5 square) ∧
    (total_length segments ≤ 16) ∧
    (segments divide the square into 5 parts of equal area)

theorem divide_square_5x5 : can_divide_square :=
  sorry

end divide_square_5x5_l378_378955


namespace num_correct_props_l378_378968

/-- Conditions for the problem -/
def prop1 : Prop := ∀ (P Q R: Type) [Fintype P] [Fintype Q] [Fintype R], P × Q × R -> Set (P × Q × R)
def prop2 : Prop := ∃ (P Q R: Type) [Fintype P] [Fintype Q] [Fintype R], P × Q × R ∈ Set (P × Q × R)
def prop3 : Prop := ∃ (P Q R: Type) [Fintype P] [Fintype Q] [Fintype R], P × Q × R -> Set (P × Q × R) ∧ Set (P × Q × R) -> Set (P × Q × R)
def prop4 : Prop := ∀ (P Q R: Type) [Fintype P] [Fintype Q] [Fintype R], P × Q × R -> Set (P × Q × R) ∧ Set (P × Q × R) -> Set (P × Q × R)

/-- Main proof statement -/
theorem num_correct_props :
  let cond1 := prop1
  let cond2 := prop2
  let cond3 := prop3
  let cond4 := prop4
  (∀ cond1 cond2 cond3 cond4, count_correct [cond1, cond2, cond3, cond4] = 2) :=
by {
  sorry
}

end num_correct_props_l378_378968


namespace scientific_notation_113700_l378_378829

theorem scientific_notation_113700 :
  ∃ (a : ℝ) (b : ℤ), 113700 = a * 10 ^ b ∧ a = 1.137 ∧ b = 5 :=
by
  sorry

end scientific_notation_113700_l378_378829


namespace min_value_y_min_value_9a_b_l378_378519

-- Proof problem for Part 1
theorem min_value_y (x : ℝ) (hx : x > 1) : 
  (∀ x > 1, (x + 4 / (x - 1))) ≥ 5 :=
sorry

-- Proof problem for Part 2
theorem min_value_9a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) : 
  (∀ a b > 0, a + b = a * b → 9 * a + b) ≥ 16 :=
sorry

end min_value_y_min_value_9a_b_l378_378519


namespace probability_all_white_balls_l378_378536

theorem probability_all_white_balls 
  (white_balls : ℕ) (black_balls : ℕ) (total_balls : ℕ) (drawn_balls : ℕ) 
  (h1 : white_balls = 8) (h2 : black_balls = 7) (h3 : total_balls = white_balls + black_balls)
  (h4 : drawn_balls = 3) : 
  (∃ p : ℚ, p = (comb white_balls drawn_balls) / (comb total_balls drawn_balls) ∧ 
  p = 8 / 65) := 
sorry

noncomputable def comb (n k : ℕ) : ℕ :=
  n.choose k

end probability_all_white_balls_l378_378536


namespace min_a2_plus_b2_quartic_eq_l378_378060

theorem min_a2_plus_b2_quartic_eq (a b : ℝ) (x : ℝ) 
  (h : x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  a^2 + b^2 ≥ 4/5 := 
sorry

end min_a2_plus_b2_quartic_eq_l378_378060


namespace log_base_one_fourth_of_sixteen_l378_378266

theorem log_base_one_fourth_of_sixteen : log (1/4) 16 = -2 :=  sorry

end log_base_one_fourth_of_sixteen_l378_378266


namespace area_of_triangle_GEF_l378_378789

def triangle (A B C G : Type) : Prop := 
  -- Define a triangle with a centroid 
  ∃ (a b c : ℝ), -- Real numbers representing the sides of the triangle
  ∃ (ABC : ℝ), -- The area of the triangle
    A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ -- Distinct points
    G = centroid A B C ∧ -- G is the centroid of triangle ABC
    ∃ (D E F : Type), -- The points where perpendiculars from G intersect sides of the triangle
      perpendicular G D (line B C) ∧ -- Perpendicular from G to BC at D
      perpendicular G E (line C A) ∧ -- Perpendicular from G to CA at E
      perpendicular G F (line A B) -- Perpendicular from G to AB at F

theorem area_of_triangle_GEF {A B C G D E F : Type}
  (abc : ∀ (x : Type), ℝ)  -- Sides of the triangle
  (ABC : ℝ)  -- Area of the triangle
  (h : triangle A B C G)
  : 
  area (triangle D E F) = 
  (4 / 9) * (abc A ^ 2 + abc B ^ 2 + abc C ^ 2) / (abc A ^ 2 * abc B ^ 2 * abc C ^ 2) * ABC ^ 3
:=
sorry

end area_of_triangle_GEF_l378_378789


namespace min_seq_sum_l378_378993

def sequence (n : ℕ) : ℕ → ℤ
| 0     := 0
| (n+1) := sequence n + (-1)^(n+1) * 1

theorem min_seq_sum : 
  let seq := sequence in
  (∀ n, n ≤ 2002 → | seq (n+1) - seq n | = 1) →
  seq 0 = 0 →
  finset.sum (finset.range 2003) seq = -1001 := 
begin
  intros,
  sorry
end

end min_seq_sum_l378_378993


namespace condition_for_n_total_sum_n_2002_l378_378112

-- Necessary and sufficient condition for n to reduce to one number
theorem condition_for_n (n : ℕ) : 
  (∃ p : ℕ, n = 3 * p + 1) ↔ n % 3 = 1 :=
sorry

-- Total sum of all numbers when n = 2002
theorem total_sum_n_2002 (n := 2002) : 
  (∑ i in (Finset.range n).map (λ i => i+1), i : ℕ) + 
  6 * ((∑ i in (Finset.range (2002)).map (λ i => i+1), i) : ℕ) = 12881478 :=
sorry

end condition_for_n_total_sum_n_2002_l378_378112


namespace average_age_of_first_7_students_l378_378809

variables {x : ℕ}

theorem average_age_of_first_7_students (h1 : 15 = 15) 
                                       (h2 : 15 * 15 = 225)
                                       (h3 : 16 * 7 = 112)
                                       (h4 : 225 - 112 - 15 = 98)
                                       (h5 : 98 / 7 = 14) :
                                       x = 14 := 
begin
  sorry
end

end average_age_of_first_7_students_l378_378809


namespace jenna_jonathan_in_picture_probability_l378_378744

/-- 
  Jenna and Jonathan are running on a circular track. 
  Jenna runs counterclockwise and completes a lap every 75 seconds. 
  Jonathan runs clockwise and completes a lap every 60 seconds. 
  They start from the same line at the same time. 
  Between 15 and 16 minutes after they start running, 
  a photographer situated in the infield captures a third of the track, centered on the starting line. 
  The probability that both Jenna and Jonathan are captured in the picture is 2/3. 
-/ 
theorem jenna_jonathan_in_picture_probability : 
  ∀ t ∈ set.Icc (900 : ℝ) (960 : ℝ), 
  let jenna_position := (t / 75 - ⌊t / 75⌋ : ℝ) in 
  let jonathan_position := (1 - t / 60 + ⌊t / 60⌋ : ℝ) in 
  (jenna_position < 1/3 ∨ jenna_position > 2/3) ∧ (jonathan_position < 1/3 ∨ jonathan_position > 2/3)
  -> ∀t ∈ (set.Icc (900 : ℝ) (960 : ℝ)), P t = (2 / 3 : ℝ) sorry

end jenna_jonathan_in_picture_probability_l378_378744


namespace minimum_cells_to_remove_not_rook_connected_l378_378952

/-- A set of cells on a grid plane is called rook-connected if you can reach every cell in 
the set starting from any cell of the set by moving horizontally or vertically along the grid lines. 
The minimum number of cells that must be removed from a 3 × 4 rectangle of cells so that the remaining 
set of cells is not rook-connected is 2. -/
theorem minimum_cells_to_remove_not_rook_connected (R : Type) [decidable_eq R] [fintype R] :
  (∃ (cells : finset (ℕ × ℕ)), cells.card = 2 ∧ 
  ∃ (remaining : finset (ℕ × ℕ)), remaining = (finset.univ.filter (λ x, x ∉ cells)) ∧ 
  ∀ a b : ℕ × ℕ, a ∈ remaining → b ∈ remaining → rook_connected remaining a b = false) :=
sorry

/-- A helper theorem to define rook-connectedness. -/
def rook_connected (remaining : finset (ℕ × ℕ)) (a b : ℕ × ℕ) := 
  -- Definition of rook-connectedness should be provided here
  sorry


end minimum_cells_to_remove_not_rook_connected_l378_378952


namespace cos_A_eq_perimeter_eq_l378_378728

noncomputable def triangleABC (a b c : ℝ) (S : ℝ) :=
  (∀ A B C: ℝ,
    S = 1/2 * b * c * Real.sin A ∧
    2 * S = a^2 - (b - c)^2 ∧
    S = 1/2 * b * c * Real.sin A ∧
    0 < a ∧ 0 < b ∧ 0 < c ∧
    a < b + c ∧ b < a + c ∧ c < a + b ∧
    0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  )

theorem cos_A_eq (a b c S : ℝ) (h : triangleABC a b c S) : (∀ A : ℝ, Real.cos A = 3 / 5) := by
  sorry

theorem perimeter_eq (S : ℝ) :
  (∀ a b c : ℝ,
    a = sqrt 41 / 2 ∧ S = 4 ∧
    triangleABC a b c S → 
    a + b + c = (sqrt 41 + 13) / 2
  ) := by
  sorry

end cos_A_eq_perimeter_eq_l378_378728


namespace probability_two_sprouts_out_of_three_l378_378133

theorem probability_two_sprouts_out_of_three (p : ℚ) (h : p = 3/5) : 
  (∃ q, q = (3.choose 2 * (p ^ 2) * (1 - p)^1) ∧ q = 54/125) :=
by
  use 3.choose 2 * (p ^ 2) * (1 - p)
  have h1 : 3.choose 2 = 3,
  sorry
  have h2 : (p ^ 2) = (3/5) ^ 2,
  sorry
  have h3 : (1 - p) = (2/5),
  sorry
  have h4 : 3 * (3/5)^2 * (2/5) = 54/125,
  sorry
  rw [←h1, ← h2, ← h3]
  exact h4


end probability_two_sprouts_out_of_three_l378_378133


namespace monotonicity_solution_set_l378_378645

noncomputable def f : ℝ → ℝ := sorry

variable (x y a : ℝ)

-- Condition 1: ∀ x, y ∈ ℝ, f(x) + f(y) = 2 + f(x + y)
axiom ax1 : ∀ x y : ℝ, f(x) + f(y) = 2 + f(x + y)

-- Condition 2: f(3) = 5
axiom ax2 : f 3 = 5

-- Condition 3: ∀ x > 0, f(x) > 2
axiom ax3 : ∀ x : ℝ, x > 0 → f x > 2

-- Problem 1: Prove monotonicity
theorem monotonicity : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := sorry

-- Problem 2: Find the solution set of f(a^2 - 2a - 2) < 3
theorem solution_set : f (a^2 - 2 * a - 2) < 3 ↔ -1 < a ∧ a < 3 := sorry

end monotonicity_solution_set_l378_378645


namespace num_subsets_union_count_l378_378409

def num_ways_choose_subsets_union_eq (n : ℕ) : ℕ := (3^n + 1) / 2

theorem num_subsets_union_count (S : Finset α) (h : S.card = n) :
  ∃ (num_ways : ℕ), num_ways = num_ways_choose_subsets_union_eq n :=
by
  use num_ways_choose_subsets_union_eq n
  sorry

end num_subsets_union_count_l378_378409


namespace bart_total_pages_l378_378580

theorem bart_total_pages (total_spent : ℝ) (cost_per_notepad : ℝ) (pages_per_notepad : ℕ)
  (h1 : total_spent = 10) (h2 : cost_per_notepad = 1.25) (h3 : pages_per_notepad = 60) :
  total_spent / cost_per_notepad * pages_per_notepad = 480 :=
by
  sorry

end bart_total_pages_l378_378580


namespace goldfish_feeding_l378_378786

theorem goldfish_feeding (g : ℕ) (h : g = 8) : 4 * g = 32 :=
by
  sorry

end goldfish_feeding_l378_378786


namespace log_one_fourth_sixteen_l378_378247

theorem log_one_fourth_sixteen : log (1 / 4) 16 = -2 := 
by
  let x := log (1 / 4) 16
  have h₁ : (1 / 4) ^ x = 16 := by simp [log_eq_iff]
  have h₂ : (4 ^ (-1)) ^ x = 16 := by rw [one_div, inv_pow]
  have h₃ : 4 ^ (-x) = 16 := by simp [pow_mul]
  have h₄ : 16 = 4 ^ 2 := by norm_num
  rw [h₄] at h₃
  have h₅ : -x = 2 := by exact pow_inj (lt_trans zero_lt_one (by norm_num)) zero_lt_four h₃
  have h₆ : x = -2 := by linarith
  exact h₆

end log_one_fourth_sixteen_l378_378247


namespace max_value_of_quadratic_l378_378700

theorem max_value_of_quadratic:
  ∀ (x : ℝ), (∃ y : ℝ, y = -3 * x ^ 2 + 9) → (∃ max_y : ℝ, max_y = 9 ∧ ∀ x : ℝ, -3 * x ^ 2 + 9 ≤ max_y) :=
by
  sorry

end max_value_of_quadratic_l378_378700


namespace find_f_max_min_values_l378_378655

noncomputable def f : ℝ → ℝ := sorry     -- To be proven to be -2x + 1

theorem find_f :
  (∀ x, f(f x) = 4 * x - 1) ∧ (∀ x1 x2, x1 < x2 → f x1 > f x2) → f = λ x, -2 * x + 1 := sorry

theorem max_min_values :
  (∀ x, f(f x) = 4 * x - 1) ∧ (∀ x1 x2, x1 < x2 → f x1 > f x2) → 
  (f = λ x, -2 * x + 1) →
  (∀ x ∈ set.Icc (-1 : ℝ) 2, f x + x^2 - x ≤ 6) ∧ 
  (∀ x ∈ set.Icc (-1 : ℝ) 2, f x + x^2 - x ≥ -(5 / 4)) := sorry

end find_f_max_min_values_l378_378655


namespace locus_of_points_x_coordinate_l378_378279

theorem locus_of_points_x_coordinate (a c : ℝ) (M : ℝ × ℝ) :
  let A := (-c, 0 : ℝ)
  let B := (c, 0 : ℝ)
  (dist M A)^2 - (dist M B)^2 = a → M.fst = a / (4 * c) := 
by
  sorry

end locus_of_points_x_coordinate_l378_378279


namespace number_of_rhombuses_is_84_l378_378571

def total_rhombuses (side_length_large_triangle : Nat) (side_length_small_triangle : Nat) (num_small_triangles : Nat) : Nat :=
  if side_length_large_triangle = 10 ∧ 
     side_length_small_triangle = 1 ∧ 
     num_small_triangles = 100 then 84 else 0

theorem number_of_rhombuses_is_84 :
  total_rhombuses 10 1 100 = 84 := by
  sorry

end number_of_rhombuses_is_84_l378_378571


namespace log_base_one_fourth_of_sixteen_l378_378262

theorem log_base_one_fourth_of_sixteen : log (1/4) 16 = -2 :=  sorry

end log_base_one_fourth_of_sixteen_l378_378262


namespace sin_shifted_decreasing_on_interval_l378_378472

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2)

theorem sin_shifted_decreasing_on_interval : 
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → y ≤ Real.pi → f y ≤ f x := 
begin
  sorry
end

end sin_shifted_decreasing_on_interval_l378_378472


namespace midpoint_intersection_theorem_centroid_coincidence_theorem_nine_point_circle_theorem_hyperbola_with_perpendicular_asymptotes_l378_378126

variables (A B C D : Point)
variable (Γ : ConicSection)

-- Definitions of midpoints and intersections, orthocenter, centroid, etc. should be present in the library
-- Placeholders used as Point, ConicSection might need actual definition or imported from library

theorem midpoint_intersection_theorem :
  (∃ (M1 M2 M3 M4 M5 M6 : Point), 
      (Γ passes_through M1) ∧
      (Γ passes_through M2) ∧
      (Γ passes_through M3) ∧
      (Γ passes_through M4) ∧
      (Γ passes_through M5) ∧
      (Γ passes_through M6) ∧
      is_midpoint M1 A B ∧
      is_midpoint M2 A C ∧
      is_midpoint M3 A D ∧
      is_midpoint M4 B C ∧
      is_midpoint M5 B D ∧
      is_midpoint M6 C D ∧
  ∃ (O1 O2 O3 : Point),
      (Γ passes_through O1) ∧
      (Γ passes_through O2) ∧
      (Γ passes_through O3) ∧
      is_intersection O1 (line A B) (line C D) ∧
      is_intersection O2 (line A C) (line B D) ∧
      is_intersection O3 (line A D) (line B C)) :=
sorry

theorem centroid_coincidence_theorem :
  (center Γ = centroid A B C D) :=
sorry

theorem nine_point_circle_theorem :
  (is_orthocenter D A B C → Γ = nine_point_circle A B C) :=
sorry

theorem hyperbola_with_perpendicular_asymptotes :
  (is_cyclic_quadrilateral A B C D → is_hyperbola_with_perpendicular_asymptotes Γ) :=
sorry

end midpoint_intersection_theorem_centroid_coincidence_theorem_nine_point_circle_theorem_hyperbola_with_perpendicular_asymptotes_l378_378126


namespace num_multiples_of_4_between_101_and_350_l378_378921

theorem num_multiples_of_4_between_101_and_350 : 
  ∃ n : ℕ, n = 62 ∧
  n = (nat.div (348 - 104) 4 + 1) := 
begin
  use (nat.div (348 - 104) 4 + 1),
  split,
  { refl },
  { rw nat.div_add_mod, sorry }
end

end num_multiples_of_4_between_101_and_350_l378_378921


namespace find_a_and_solve_inequality_l378_378470

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / (2^x - 2)) + a

theorem find_a_and_solve_inequality
  (a : ℝ)
  (H_symm : ∀ x : ℝ, f(x) = \frac{1}{2^{x}-2} + a, symmetric about (1,0))
  : a = \frac{1}{4} 
    /\ (∀ x : ℝ, f x < 5 / 4 → (x > real.log 3 / real.log 2 ∨ x < 1)) := 
by 
  sorry

end find_a_and_solve_inequality_l378_378470


namespace cost_of_cherries_l378_378185

-- Definitions and conditions
def cost_of_crust : ℝ := 2 + 1 + 1.5
def blueberry_cost_per_pound : ℝ := (2.25 / 8) * 16
def total_blueberry_cost (pounds : ℝ) : ℝ := pounds * blueberry_cost_per_pound
def total_cost_of_cheapest_pie : ℝ := 18

-- Proof statement
theorem cost_of_cherries : 
  (total_cost_of_cheapest_pie = cost_of_crust + total_blueberry_cost 3) →
  (∀ (cherry_cost : ℝ), total_cost_of_cheapest_pie = cost_of_crust + cherry_cost → cherry_cost = 13.5) :=
by
  intros h1 h2
  have h_crust : cost_of_crust = 4.5 := by norm_num
  have h_blueberry : blueberry_cost_per_pound = 4.5 := by norm_num
  have h_total_blueberry : total_blueberry_cost 3 = 13.5 := by simp [total_blueberry_cost, h_blueberry]
  rw [h_total_blueberry, h_crust] at h1
  have h_cheapest : total_cost_of_cheapest_pie = 18 := h1
  sorry

end cost_of_cherries_l378_378185


namespace figure_at_214_l378_378480

def sequence : List String := ["△", "○", "□", "△", "O"]

def position_in_cycle (n : Nat) : Nat :=
  (n - 1) % sequence.length + 1

theorem figure_at_214 (h : position_in_cycle 214 = 4) : sequence[3] = "△" :=
by 
  unfold position_in_cycle at h
  sorry

end figure_at_214_l378_378480


namespace triangle_angle_magnitude_triangle_bc_value_l378_378356

theorem triangle_angle_magnitude {A B C : Real} (a b c : Real) 
  (h1 : 2 * b * cos A = c * cos A + a * cos C) : 
  A = π / 3 :=
by
  sorry

theorem triangle_bc_value {A B C : Real} (a b c : Real) 
  (h2 : a = sqrt 7) (h3 : b + c = 4)
  (h1 : 2 * b * cos A = c * cos A + a * cos C):
  a^2 = (b + c)^2 - 3 * (b * c) ∧ b * c = 3 :=
by
  sorry

end triangle_angle_magnitude_triangle_bc_value_l378_378356


namespace time_to_finish_all_problems_l378_378633

def mathProblems : ℝ := 17.0
def spellingProblems : ℝ := 15.0
def problemsPerHour : ℝ := 8.0
def totalProblems : ℝ := mathProblems + spellingProblems

theorem time_to_finish_all_problems : totalProblems / problemsPerHour = 4.0 :=
by
  sorry

end time_to_finish_all_problems_l378_378633


namespace triangles_with_perimeter_20_l378_378693

theorem triangles_with_perimeter_20 (sides : Finset (Finset ℕ)) : 
  (∀ {a b c : ℕ}, (a + b + c = 20) → (a > 0) → (b > 0) → (c > 0) 
  → (a + b > c) → (a + c > b) → (b + c > a) → ({a, b, c} ∈ sides)) 
  → sides.card = 8 := 
by
  sorry

end triangles_with_perimeter_20_l378_378693


namespace find_number_l378_378552

theorem find_number (x : ℝ) : 
  220050 = (555 + x) * (2 * (x - 555)) + 50 ↔ x = 425.875 ∨ x = -980.875 := 
by 
  sorry

end find_number_l378_378552


namespace minimum_bottles_needed_l378_378187

theorem minimum_bottles_needed (fl_oz_needed : ℝ) (bottle_size_ml : ℝ) (fl_oz_per_liter : ℝ) (ml_per_liter : ℝ)
  (h1 : fl_oz_needed = 60)
  (h2 : bottle_size_ml = 250)
  (h3 : fl_oz_per_liter = 33.8)
  (h4 : ml_per_liter = 1000) :
  ∃ n : ℕ, n = 8 ∧ fl_oz_needed * ml_per_liter / fl_oz_per_liter / bottle_size_ml ≤ n :=
by
  sorry

end minimum_bottles_needed_l378_378187


namespace rightmost_four_digits_of_5_pow_2023_l378_378093

theorem rightmost_four_digits_of_5_pow_2023 :
  5 ^ 2023 % 5000 = 3125 :=
  sorry

end rightmost_four_digits_of_5_pow_2023_l378_378093


namespace initial_money_equals_26_l378_378005

def cost_jumper : ℕ := 9
def cost_tshirt : ℕ := 4
def cost_heels : ℕ := 5
def money_left : ℕ := 8

def total_cost_items : ℕ := cost_jumper + cost_tshirt + cost_heels

theorem initial_money_equals_26 : total_cost_items + money_left = 26 := by
  sorry

end initial_money_equals_26_l378_378005


namespace exists_function_intersect_axes_l378_378436

theorem exists_function_intersect_axes :
  ∃ (f : ℝ → ℝ), (f = λ x, x + 1) ∧ (∀ x : ℝ, x < 0 → f x > 0) ∧ (f 0 > 0) :=
by
  let f := λ x : ℝ, x + 1
  use f
  split
  . rfl
  . split
    . intros x hx
      simp
      assumption
    . simp
      linarith

end exists_function_intersect_axes_l378_378436


namespace intersection_M_N_l378_378707

-- Definitions for sets M and N
def set_M : Set ℝ := {x | abs x < 1}
def set_N : Set ℝ := {x | x^2 <= x}

-- The theorem stating the intersection of M and N
theorem intersection_M_N : {x : ℝ | x ∈ set_M ∧ x ∈ set_N} = {x : ℝ | 0 <= x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l378_378707


namespace max_possible_integer_l378_378149

noncomputable def largest_integer_in_list : ℕ :=
  12

theorem max_possible_integer (L : List ℕ) (h_pos : ∀ n ∈ L, 0 < n) (h_len : L.length = 6)
  (h_median : (L.sorted 10 ≤ 10)) (h_mean : L.sum / 6 = 11) (h_9_multiple : ∃ k ∈ L, k = 9 ∧ ∃ m ∈ L, m = 9 ∧ ¬(∃ n ∈ L,  n = n ∧ n ≠ 9)):
  ∃ x ∈ L, x = largest_integer_in_list :=
by
  sorry

end max_possible_integer_l378_378149


namespace right_triangle_leg_square_l378_378820

theorem right_triangle_leg_square (a c b : ℕ) (h1 : c = a + 1) (h2 : a^2 + b^2 = c^2) : b^2 = c + a :=
by
  sorry

end right_triangle_leg_square_l378_378820


namespace part1_part2_l378_378584

-- Part (1)
theorem part1 : 0.064^(-1/3) - (-1/8)^0 + 16^(3/4) + 0.25^(1/2) = 10 :=
by
  sorry

-- Part (2)
theorem part2 : 3^(Math.log 3 4) - 27^(2/3) - Math.log10 0.01 + Real.log (Real.exp 3) = 0 :=
by
  sorry

end part1_part2_l378_378584


namespace vector_dot_cross_product_zero_l378_378770

variable {ℝ} [normedAddCommGroup ℝ]
variable {u v w : ℝ}

noncomputable def norm (u : ℝ) := ∥u∥
noncomputable def crossProduct (u v : ℝ) := u × v
noncomputable def dotProduct (u v : ℝ) := u • v

theorem vector_dot_cross_product_zero
  (hu : ∥u∥ = 2)
  (hv : ∥v∥ = 3)
  (hw : crossProduct u (v + u) = w)
  (hwu : crossProduct w u = 2 * v) :
  dotProduct u (crossProduct v w) = 0 := sorry

end vector_dot_cross_product_zero_l378_378770


namespace minimum_bottles_needed_l378_378186

theorem minimum_bottles_needed (fl_oz_needed : ℝ) (bottle_size_ml : ℝ) (fl_oz_per_liter : ℝ) (ml_per_liter : ℝ)
  (h1 : fl_oz_needed = 60)
  (h2 : bottle_size_ml = 250)
  (h3 : fl_oz_per_liter = 33.8)
  (h4 : ml_per_liter = 1000) :
  ∃ n : ℕ, n = 8 ∧ fl_oz_needed * ml_per_liter / fl_oz_per_liter / bottle_size_ml ≤ n :=
by
  sorry

end minimum_bottles_needed_l378_378186


namespace money_after_purchase_l378_378995

def initial_money : ℕ := 4
def cost_of_candy_bar : ℕ := 1
def money_left : ℕ := 3

theorem money_after_purchase :
  initial_money - cost_of_candy_bar = money_left := by
  sorry

end money_after_purchase_l378_378995


namespace probability_king_and_heart_correct_l378_378851

open Probability

noncomputable def probability_king_then_heart : ℚ :=
  let total_cards : ℚ := 52
  let kings : ℚ := 4
  let hearts : ℚ := 13
  let king_of_hearts : ℚ := 1
  let remaining_cards_after_first_draw : ℚ := total_cards - 1
  
  -- Probability calculations
  let pr_case1 : ℚ := (king_of_hearts / total_cards) * ((hearts - 1) / remaining_cards_after_first_draw)
  let pr_case2 : ℚ := ((kings - king_of_hearts) / total_cards) * (hearts / remaining_cards_after_first_draw)
  pr_case1 + pr_case2

-- The target theorem
theorem probability_king_and_heart_correct :
  probability_king_then_heart = 1 / 52 :=
sorry

end probability_king_and_heart_correct_l378_378851


namespace cindy_envelopes_l378_378190

theorem cindy_envelopes (h₁ : ℕ := 4) (h₂ : ℕ := 7) (h₃ : ℕ := 5) (h₄ : ℕ := 10) (h₅ : ℕ := 3) (initial : ℕ := 137) :
  initial - (h₁ + h₂ + h₃ + h₄ + h₅) = 108 :=
by
  sorry

end cindy_envelopes_l378_378190


namespace area_of_rectangle_l378_378469

-- Given conditions
def shadedSquareArea : ℝ := 4
def nonShadedSquareArea : ℝ := shadedSquareArea
def largerSquareArea : ℝ := 4 * 4  -- Since the side length is twice the previous squares

-- Problem statement
theorem area_of_rectangle (shadedSquareArea nonShadedSquareArea largerSquareArea : ℝ) :
  shadedSquareArea + nonShadedSquareArea + largerSquareArea = 24 :=
sorry

end area_of_rectangle_l378_378469


namespace parallel_and_perpendicular_implies_perpendicular_l378_378406

variables (l : Line) (α β : Plane)

axiom line_parallel_plane (l : Line) (π : Plane) : Prop
axiom line_perpendicular_plane (l : Line) (π : Plane) : Prop
axiom planes_are_perpendicular (π₁ π₂ : Plane) : Prop

theorem parallel_and_perpendicular_implies_perpendicular
  (h1 : line_parallel_plane l α)
  (h2 : line_perpendicular_plane l β) 
  : planes_are_perpendicular α β :=
sorry

end parallel_and_perpendicular_implies_perpendicular_l378_378406


namespace find_monthly_salary_l378_378942

def man_saves_25_percent (S : ℕ) : ℕ := 0.25 * S

def expenses_increase_30_percent (E : ℕ) : ℕ := 1.30 * E

def monthly_savings_after_increase (S E : ℕ) : ℕ :=
  S - expenses_increase_30_percent E

def monthly_savings_after_donation (S E : ℕ) : ℕ :=
  monthly_savings_after_increase S E - 0.05 * S

theorem find_monthly_salary (S : ℕ) (E : ℕ)
    (h1 : monthly_savings_after_increase S E = 350)
    (h2 : monthly_savings_after_donation S E = 250) :
    S = 2000 :=
by sorry

end find_monthly_salary_l378_378942


namespace geometric_sequence_a3a5_l378_378362

theorem geometric_sequence_a3a5 :
  ∀ (a : ℕ → ℝ) (r : ℝ), (a 4 = 4) → (a 3 = a 0 * r ^ 3) → (a 5 = a 0 * r ^ 5) →
  a 3 * a 5 = 16 :=
by
  intros a r h1 h2 h3
  sorry

end geometric_sequence_a3a5_l378_378362


namespace area_enclosed_by_absolute_value_linear_eq_l378_378857

theorem area_enclosed_by_absolute_value_linear_eq (x y : ℝ) :
  (|5 * x| + |3 * y| = 15) → ∃ (A : ℝ), A = 30 :=
by
  sorry

end area_enclosed_by_absolute_value_linear_eq_l378_378857


namespace lowest_score_l378_378988

theorem lowest_score (score1 score2 : ℕ) (max_score : ℕ) (desired_mean : ℕ) (lowest_possible_score : ℕ) 
  (h_score1 : score1 = 82) (h_score2 : score2 = 75) (h_max_score : max_score = 100) (h_desired_mean : desired_mean = 85)
  (h_lowest_possible_score : lowest_possible_score = 83) : 
  ∃ x1 x2 : ℕ, x1 = max_score ∧ x2 = lowest_possible_score ∧ (score1 + score2 + x1 + x2) / 4 = desired_mean := by
  sorry

end lowest_score_l378_378988


namespace binom_2n_2_eq_n_2n_minus_1_l378_378506

theorem binom_2n_2_eq_n_2n_minus_1 (n : ℕ) (h : n > 0) : 
  (Nat.choose (2 * n) 2) = n * (2 * n - 1) := 
sorry

end binom_2n_2_eq_n_2n_minus_1_l378_378506


namespace median_of_trapezoid_l378_378547

noncomputable def large_triangle_side : ℝ := 4
noncomputable def large_triangle_area := (sqrt 3 / 4) * large_triangle_side ^ 2
noncomputable def small_triangle_area := large_triangle_area / 3
noncomputable def small_triangle_side := sqrt (4 * small_triangle_area / sqrt 3)

theorem median_of_trapezoid :
  (4 + sqrt 3 * 4 / 3) / 2 = 2 + 2 * sqrt 3 / 3 :=
by sorry

end median_of_trapezoid_l378_378547


namespace smallest_base10_integer_l378_378882

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end smallest_base10_integer_l378_378882


namespace q3_is_150_l378_378528

namespace Geometry

variable (x : ℕ → ℝ) (n : ℕ)

-- Definition: Condition that the sum of the x-coordinates of Q1's vertices equals 150
def sum_of_x_Q1_eq_150 : Prop := (Finset.range 50).sum x = 150

-- Definition: Midpoint being used to form the x-coordinates of Q2 and Q3 accordingly
def midpoints (f : ℕ → ℝ) (i : ℕ) : ℝ := (f i + f ((i + 1) % 50)) / 2

-- Predicate: Sum of the x-coordinates for Q2
def sum_of_x_Q2_eq_150 : Prop := (Finset.range 50).sum (midpoints x) = 150

-- Predicate: Sum of the x-coordinates for Q3
def sum_of_x_Q3_eq_150 : Prop := (Finset.range 50).sum (midpoints (midpoints x)) = 150

theorem q3_is_150 (x : ℕ → ℝ) (h : sum_of_x_Q1_eq_150 x) : sum_of_x_Q3_eq_150 x := 
sorry

end Geometry

end q3_is_150_l378_378528


namespace fraction_crop_CD_l378_378991

variable (length_AB length_AD length_BC : ℝ)
variable (angle_45 angle_135 : ℝ)

-- Conditions
def is_trapezoid (ABCD : Type) : Prop :=
  true -- assuming all given conditions define a trapezoid

def trapezoid_ADJ_angles (angle_45 angle_135 : ℝ) : Prop :=
  angle_45 = (45 : ℝ) ∧ angle_135 = (135 : ℝ)

def perpendicular_sides (AD BC : ℝ) : Prop :=
  AD = 100 ∧ BC = 100

def parallel_sides_length (AB CD : ℝ) : Prop :=
  AB = 80 ∧ CD > 100

-- Define the fraction of the crop transported to the longest side CD
theorem fraction_crop_CD :
  ∀ (ABCD : Type) (length_AB length_AD length_BC : ℝ) (angle_45 angle_135 : ℝ)
  (H1 : is_trapezoid ABCD) (H2 : trapezoid_ADJ_angles angle_45 angle_135) 
  (H3 : perpendicular_sides length_AD length_BC) (H4 : parallel_sides_length length_AB length_AD),
  (3 / 4 : ℝ) := 
by
  sorry

end fraction_crop_CD_l378_378991


namespace sum_of_squares_consecutive_nat_l378_378070

theorem sum_of_squares_consecutive_nat (n : ℕ) (h : n = 26) : (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2 = 2030 :=
by
  sorry

end sum_of_squares_consecutive_nat_l378_378070


namespace bacteria_original_count_l378_378554

theorem bacteria_original_count (current: ℕ) (increase: ℕ) (hc: current = 8917) (hi: increase = 8317) : current - increase = 600 :=
by
  sorry

end bacteria_original_count_l378_378554


namespace F_2457_find_Q_l378_378284

-- Define the properties of a "rising number"
def is_rising_number (m : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    m = 1000 * a + 100 * b + 10 * c + d ∧
    a < b ∧ b < c ∧ c < d ∧
    a + d = b + c

-- Define F(m) as specified
def F (m : ℕ) : ℤ :=
  let a := m / 1000
  let b := (m / 100) % 10
  let c := (m / 10) % 10
  let d := m % 10
  let m' := 1000 * c + 100 * b + 10 * a + d
  (m' - m) / 99

-- Problem statement for F(2457)
theorem F_2457 : F 2457 = 30 := sorry

-- Properties given in the problem statement for P and Q
def is_specific_rising_number (P Q : ℕ) : Prop :=
  ∃ (x y z t : ℕ),
    P = 1000 + 100 * x + 10 * y + z ∧
    Q = 1000 * x + 100 * t + 60 + z ∧
    1 < x ∧ x < t ∧ t < 6 ∧ 6 < z ∧
    1 + z = x + y ∧
    x + z = t + 6 ∧
    F P + F Q % 7 = 0

-- Problem statement to find the value of Q
theorem find_Q (Q : ℕ) : 
  ∃ (P : ℕ), is_specific_rising_number P Q ∧ Q = 3467 := sorry

end F_2457_find_Q_l378_378284


namespace simplify_and_evaluate_expr_l378_378454

-- Define the expression as a function of x
def expr (x : ℝ) : ℝ :=
  ((x ^ 2 / (x - 1) - x + 1) / ((4 * x ^ 2 - 4 * x + 1) / (1 - x)))

noncomputable def x_value : ℝ := (Real.sqrt 5) + 1 / 2

-- State the theorem we want to prove
theorem simplify_and_evaluate_expr :
  expr x_value = -Real.sqrt 5 / 10 :=
by
  sorry

end simplify_and_evaluate_expr_l378_378454


namespace total_displacement_zero_farthest_point_twenty_two_total_fuel_five_point_four_l378_378426

/-
Problem:
Given a taxi's itinerary and its travel directions, prove the following:
1. The total displacement of the taxi after the journey.
2. The farthest point from the starting point the taxi reaches.
3. The total fuel consumption of the taxi after the journey.
-/

def itinerary : List Int := [10, -3, 4, -2, 13, -8, -7, -5, -2]
def fuel_per_km : Float := 0.1

theorem total_displacement_zero : 
  itinerary.sum = 0 := sorry

theorem farthest_point_twenty_two :
  itinerary.scanl (+) 0 |>.map (λ x => Int.abs x) |>.max = some 22 := sorry

theorem total_fuel_five_point_four :
  itinerary.map Int.abs |>.sum * fuel_per_km = 5.4 := sorry

end total_displacement_zero_farthest_point_twenty_two_total_fuel_five_point_four_l378_378426


namespace total_population_l378_378131

theorem total_population (x T : ℝ) (h : 128 = (x / 100) * (50 / 100) * T) : T = 25600 / x :=
by
  sorry

end total_population_l378_378131
