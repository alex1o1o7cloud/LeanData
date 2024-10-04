import Mathlib

namespace negate_even_condition_l428_428776

theorem negate_even_condition (a b c : ℤ) :
  (¬(∀ a b c : ℤ, ∃ x : ℚ, a * x^2 + b * x + c = 0 → Even a ∧ Even b ∧ Even c)) →
  (¬Even a ∨ ¬Even b ∨ ¬Even c) :=
by
  sorry

end negate_even_condition_l428_428776


namespace domain_of_f_monotonicity_of_f_inequality_solution_l428_428209

open Real

noncomputable def f (x : ℝ) : ℝ := log ((1 - x) / (1 + x))

theorem domain_of_f :
  ∀ x, -1 < x ∧ x < 1 → ∃ y, y = f x :=
by
  intro x h
  use log ((1 - x) / (1 + x))
  simp [f]

theorem monotonicity_of_f :
  ∀ x y, -1 < x ∧ x < 1 → -1 < y ∧ y < 1 → x < y → f x > f y :=
sorry

theorem inequality_solution :
  ∀ x, f (2 * x - 1) < 0 ↔ (1 / 2 < x ∧ x < 1) :=
sorry

end domain_of_f_monotonicity_of_f_inequality_solution_l428_428209


namespace two_d1_eq_d2_l428_428048

variables {a b c : ℝ}

/-- 
Definition for arithmetic sequence and definitions of d1 and d2 as
mentioned in the problem.
-/
def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def d1 (a b c : ℝ) : ℝ := (2*b - a - c) / 2

def d2 (a b c : ℝ) : ℝ := |2 * b - a - c|

/-- 
Proof that for any arithmetic sequence (a, b, c),
the relationship 2*d1 = d2 holds.
-/
theorem two_d1_eq_d2 (a b c : ℝ) (h : is_arithmetic_sequence a b c) : 
  2 * d1 a b c = d2 a b c :=
begin
  sorry
end

end two_d1_eq_d2_l428_428048


namespace max_chain_length_in_subdivided_triangle_l428_428959

-- Define an equilateral triangle subdivision
structure EquilateralTriangleSubdivided (n : ℕ) :=
(n_squares : ℕ)
(n_squares_eq : n_squares = n^2)

-- Define the problem's chain concept
def maximum_chain_length (n : ℕ) : ℕ :=
n^2 - n + 1

-- Main statement
theorem max_chain_length_in_subdivided_triangle
  (n : ℕ) (triangle : EquilateralTriangleSubdivided n) :
  maximum_chain_length n = n^2 - n + 1 :=
by sorry

end max_chain_length_in_subdivided_triangle_l428_428959


namespace radius_of_sphere_l428_428003

theorem radius_of_sphere (r_c : ℝ) (d_c : ℝ) (r_s : ℝ) : 
  r_c = real.sqrt 2 → d_c = 1 → r_s = real.sqrt (r_c^2 + d_c^2) → r_s = real.sqrt 3 :=
by
  intros hrc hdc hrs
  rw [hrc, hdc] at hrs
  exact hrs

end radius_of_sphere_l428_428003


namespace problem_proof_l428_428547

variable {a b n : ℕ}

theorem problem_proof (h₀ : 0 < a ∧ 0 < b ∧ 0 < n)
  (h₁ : ∀ k : ℕ, 0 < k ∧ k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n :=
begin
  sorry
end

end problem_proof_l428_428547


namespace median_parallel_to_BC_general_form_median_parallel_to_BC_intercept_form_median_on_BC_general_form_median_on_BC_intercept_form_l428_428151

-- Definitions based on the conditions
def point_A := (1, -4)
def point_B := (6, 6)
def point_C := (-2, 0)

-- Proof problem statements
theorem median_parallel_to_BC_general_form :
  ∃ l : ℝ → ℝ → Prop,
    l = (λ x y, 6 * x - 8 * y - 13 = 0) :=
sorry

theorem median_parallel_to_BC_intercept_form :
  ∃ l : ℝ → ℝ → Prop,
    l = (λ x y, x / (13 / 6) - y / (13 / 8) = 1) :=
sorry

theorem median_on_BC_general_form :
  ∃ l : ℝ → ℝ → Prop,
    l = (λ x y, 7 * x - y - 11 = 0) :=
sorry

theorem median_on_BC_intercept_form :
  ∃ l : ℝ → ℝ → Prop,
    l = (λ x y, x / (11 / 7) - y / 11 = 1) :=
sorry

end median_parallel_to_BC_general_form_median_parallel_to_BC_intercept_form_median_on_BC_general_form_median_on_BC_intercept_form_l428_428151


namespace max_area_convex_quadrilateral_l428_428287

structure Point (α : Type) := (x y : α)
def Vec2 := Point ℝ

structure ConvexQuadrilateral :=
(A B C D : Vec2)
(BC_len : dist B C = 3)
(CD_len : dist C D = 8)
(is_equilateral_centroids : let g1 := (1 / 3 * (A.x + B.x + C.x), 1 / 3 * (A.y + B.y + C.y)),
                                 g2 := (1 / 3 * (B.x + C.x + D.x), 1 / 3 * (B.y + C.y + D.y)),
                                 g3 := (1 / 3 * (A.x + C.x + D.x), 1 / 3 * (A.y + C.y + D.y))
                             in dist g1 g2 = dist g2 g3 ∧ dist g2 g3 = dist g3 g1)
(right_angle_at_B : let u := (B.x - A.x, B.y - A.y), v := (C.x - B.x, C.y - B.y)
                    in u.1 * v.1 + u.2 * v.2 = 0)  -- dot product is zero, indicating a right angle

noncomputable def maximum_area (q : ConvexQuadrilateral) : ℝ :=
15.75

theorem max_area_convex_quadrilateral : ∀ q : ConvexQuadrilateral, maximum_area q = 15.75 :=
sorry

end max_area_convex_quadrilateral_l428_428287


namespace minimum_value_frac_l428_428558

variable (x y : ℝ)

def conditions : Prop :=
  (0 < x) ∧ (0 < y) ∧ (2^(x-3) = (1/2)^y)

theorem minimum_value_frac (h : conditions x y) : 
  ∀ x y, (2^(x-3) = (1/2)^y) → x + y = 3 → (1/x + 4/y) ≥ 9 :=
by
  sorry

end minimum_value_frac_l428_428558


namespace largest_n_eq_777_l428_428802

open Nat

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_prime_digit (p : ℕ) : Prop := p < 10 ∧ prime p

def large_possible_n := 
  exists (p q : ℕ), is_prime_digit p ∧ q < 10 ∧ prime (10 * p + q) ∧ 
  (100 ≤ p * q * (10 * p + q) ∧ p * q * (10 * p + q) < 1000)

theorem largest_n_eq_777 : ∃ n : ℕ, large_possible_n ∧ n = 777 :=
by
  let p := 3
  let q := 7
  have hp : is_prime_digit p := by {
    exact ⟨by decide, by norm_num⟩
  }
  have hq : q < 10 := by decide
  have h_prime : prime (10 * p + q) := by norm_num
  have hn : p * q * (10 * p + q) = 777 := by decide
  exact ⟨777, ⟨p, q, hp, hq, h_prime, ⟨by decide, by decide⟩, hn⟩,
         rfl⟩

end largest_n_eq_777_l428_428802


namespace polynomial_g_l428_428784

theorem polynomial_g (g : ℝ → ℝ) (f : ℝ → ℝ) 
  (hf : ∀ x, f(x) = x^2) 
  (hg : ∀ x, f(g(x)) = 9*x^2 - 6*x + 1) :
  (∀ x, g(x) = 3*x - 1) ∨ (∀ x, g(x) = -3*x + 1) :=
by {
  sorry
}

end polynomial_g_l428_428784


namespace rectangles_in_grid_l428_428670

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428670


namespace find_BC_l428_428045

theorem find_BC (R r a : ℝ) (hRr : R > r) (h_tangent : ∃ A B C, 
  (B ∈ sphere R A) ∧ line_through_point_tangent_to_circle C B r A) :
  ∃ O Q, B_dist : ℝ, dist BC (O, Q, a) = a * sqrt(1 + r / R) :=
begin
  sorry
end

end find_BC_l428_428045


namespace points_on_single_line_l428_428847

noncomputable def point_on_single_line (A B C : Point) : Prop :=
∀ (X : Point) (par_bX : Segment) (par_cX : Segment),
  parallel par_bX.toLine AC.toLine ∧ parallel par_cX.toLine AB.toLine ∧
  equal_length par_bX par_cX →
  ∃ (l : Line), X ∈ l

theorem points_on_single_line (A B C : Point) : point_on_single_line A B C :=
sorry

end points_on_single_line_l428_428847


namespace number_of_rectangles_in_grid_l428_428592

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428592


namespace percentage_gain_for_products_l428_428060

-- Definitions based on conditions
def product_A_actual_weight : ℚ := 900 / 1000
def product_A_advertised_weight : ℚ := 1
def product_A_cost_per_kg : ℚ := 10

def product_B_actual_weight : ℚ := 475 / 500
def product_B_advertised_weight : ℚ := 1
def product_B_cost_per_500g : ℚ := 5

def product_C_actual_weight : ℚ := 195 / 200
def product_C_advertised_weight : ℚ := 1
def product_C_cost_per_200g : ℚ := 15

-- Statement of the theorem
theorem percentage_gain_for_products :
  let percentage_gain (actual_weight advertised_weight cost : ℚ) : ℚ :=
      ((advertised_weight - actual_weight) / advertised_weight) * cost * 100 in
  percentage_gain product_A_actual_weight product_A_advertised_weight product_A_cost_per_kg = 10 ∧
  percentage_gain (product_B_actual_weight * 500 / 475) product_B_advertised_weight product_B_cost_per_500g = 5 ∧
  percentage_gain (product_C_actual_weight * 200 / 195) product_C_advertised_weight product_C_cost_per_200g = 2.5 := 
by
  sorry

end percentage_gain_for_products_l428_428060


namespace sum_alternating_binom_eq_zero_l428_428047

theorem sum_alternating_binom_eq_zero {m n : ℕ} (h : m < n) : 
  ∑ k in Finset.range (n + 1), (-1) ^ k * k ^ m * Nat.choose n k = 0 := 
sorry

end sum_alternating_binom_eq_zero_l428_428047


namespace kitten_weight_l428_428451

theorem kitten_weight (k r p : ℚ) (h1 : k + r + p = 38) (h2 : k + r = 3 * p) (h3 : k + p = r) : k = 9.5 :=
begin
  sorry
end

end kitten_weight_l428_428451


namespace rectangles_in_grid_l428_428665

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428665


namespace mul_101_101_l428_428124

theorem mul_101_101 : 101 * 101 = 10201 := 
by
  sorry

end mul_101_101_l428_428124


namespace min_swap_cost_to_original_l428_428382

noncomputable def swap_cost (x y : ℕ) : ℕ :=
2 * (abs (x - y))

theorem min_swap_cost_to_original (n : ℕ) (a : Fin n → ℕ) :
  ∃F : Fin n → Fin n, (∀i : Fin n, a (F i) = (i + 1)) ∧
  (∑ i, swap_cost (a i) (i + 1)) ≤ ∑ i, abs ((a i) - (i + 1)) := by {
  sorry
}

end min_swap_cost_to_original_l428_428382


namespace carl_tax_deduction_l428_428936

theorem carl_tax_deduction (carl_wage_dollars : ℕ) (tax_rate : ℝ) (to_cents : ℕ → ℝ) :
  carl_wage_dollars = 25 →
  tax_rate = 0.02 →
  to_cents carl_wage_dollars = 25 * 100 →
  0.02 * (to_cents carl_wage_dollars) = 50 :=
by
  intros h_wage h_rate h_cents
  rw [h_wage, h_rate, h_cents]
  norm_num
  sorry

end carl_tax_deduction_l428_428936


namespace count_rectangles_5x5_l428_428684

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428684


namespace rectangle_count_5x5_l428_428652

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428652


namespace length_PQ_l428_428820

-- Define the triangle with given side lengths
structure Triangle (α : Type) [MetricSpace α] :=
  (A B C : α)
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)
  (hAB : dist A B = AB)
  (hAC : dist A C = AC)
  (hBC : dist B C = BC)

-- Define the specific triangle XYZ with given side lengths
def triangleXYZ : Triangle ℝ :=
{ A := (0 : ℝ),
  B := 15,
  C := (17 : ℝ),
  AB := 15,
  AC := 17,
  BC := 8,
  hAB := rfl,
  hAC := rfl,
  hBC := rfl }

-- Define the points P and Q such that PQ is parallel to YZ and passes through the midpoint of YZ
def P := (7.5 : ℝ)
def Q := (8.5 : ℝ)
def Y := 15
def Z := 17
def M := midpoint ℝ (Y : ℝ) (Z : ℝ)

-- Claim that length of PQ is 4
theorem length_PQ : dist P Q = 4 := sorry

end length_PQ_l428_428820


namespace rectangle_count_5x5_l428_428643

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428643


namespace number_of_rectangles_in_grid_l428_428591

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428591


namespace sqrt_value_eq_l428_428099

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def thirteen_fact : ℕ := 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def divisor := 2 * 3 * 5 * 7 * 11

def value := thirteen_fact / divisor

theorem sqrt_value_eq : real.sqrt value = 72 * real.sqrt 2 :=
by
  have fact11 : factorial 11 = 13 * 22 * 90, by rfl
  have div_def : divisor = 2 * 3 * 5 * 7 * 11, by ring
  have val_def : value = 13 * 22 * 90 / divisor, from div_def,
  sorry

end sqrt_value_eq_l428_428099


namespace part1_part2_l428_428533

-- Define sets A and B
def set_A : set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }
def set_B (m : ℝ) : set ℝ := { x | x^2 - 2 * m * x + m^2 - 4 ≤ 0 }

-- Part 1: Prove m = 2 given A ∩ B = [0, 3]
theorem part1 (m : ℝ) : (set_A ∩ set_B m = { x | 0 ≤ x ∧ x ≤ 3 }) → m = 2 := by
  sorry

-- Part 2: Prove m > 5 or m < -3 given A ⊆ complement of B
theorem part2 (m : ℝ) : (set_A ⊆ set_B mᶜ) → (m > 5 ∨ m < -3) := by
  sorry

end part1_part2_l428_428533


namespace problem_I_problem_II_l428_428178

-- Problem Setup
def seq_a (n : ℕ) : ℝ :=
  if n = 1 then 1
  else 2^(n - 1)

def seq_b (n : ℕ) : ℝ :=
  (-1)^n * Real.logb 2 (seq_a n) * Real.logb 2 (seq_a (n + 1))

noncomputable def sum_b (n : ℕ) : ℝ :=
  (Finset.range (2 * n)).sum (λ i, seq_b (i + 1))

-- Questions to Prove
theorem problem_I (n : ℕ) : seq_a n = 2^(n - 1) :=
begin
  sorry
end

theorem problem_II (n : ℕ) : sum_b n = 2 * n^2 :=
begin
  sorry
end

end problem_I_problem_II_l428_428178


namespace xi_distribution_l428_428772

-- Defining the problem conditions
def balls := 6
def red_balls := 2
def yellow_balls := 2
def blue_balls := 2
def boxes := 3
def balls_per_box := 2

-- The random placement of balls and the possible values of xi
inductive xi_val
| zero | one | three

-- The distribution probabilities we're aiming to prove
def P_xi_3 := 1 / 12
def P_xi_1 := 2 / 5
def P_xi_0 := 31 / 60

-- The main theorem to prove the distribution of ξ
theorem xi_distribution : 
  let P : xi_val → ℚ := 
    λ x, match x with
    | xi_val.three := P_xi_3
    | xi_val.one := P_xi_1
    | xi_val.zero := P_xi_0
    end in 
  P xi_val.three = 1 / 12 ∧ P xi_val.one = 2 / 5 ∧ P xi_val.zero = 31 / 60 := by
  sorry

end xi_distribution_l428_428772


namespace fraction_videocassette_recorders_l428_428267

variable (H : ℝ) (F : ℝ)

-- Conditions
variable (cable_TV_frac : ℝ := 1 / 5)
variable (both_frac : ℝ := 1 / 20)
variable (neither_frac : ℝ := 0.75)

-- Main theorem statement
theorem fraction_videocassette_recorders (H_pos : 0 < H) 
  (cable_tv : cable_TV_frac * H > 0)
  (both : both_frac * H > 0) 
  (neither : neither_frac * H > 0) :
  F = 1 / 10 :=
by
  sorry

end fraction_videocassette_recorders_l428_428267


namespace min_value_x_plus_y_l428_428197

open Real

noncomputable def xy_plus_x_minus_y_minus_10_eq_zero (x y: ℝ) := x * y + x - y - 10 = 0

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : xy_plus_x_minus_y_minus_10_eq_zero x y) : 
  x + y ≥ 6 :=
by
  sorry

end min_value_x_plus_y_l428_428197


namespace common_chord_eq_line_through_point_eq_l428_428582

noncomputable def eq_C1 : (x y : ℝ) → Prop := 
  λ x y, x^2 + y^2 - 4 * x - 2 * y - 5 = 0

noncomputable def eq_C2 : (x y : ℝ) → Prop := 
  λ x y, x^2 + y^2 - 6 * x - y - 9 = 0

theorem common_chord_eq (x y : ℝ) : 
  (eq_C1 x y → eq_C2 x y → (2 * x - y + 4 = 0)) :=
sorry

theorem line_through_point_eq (x y : ℝ) : 
  (eq_C1 x y ∧ eq_C1 4 (-4) ∧ (x ≠ 4 → 4 ≤ (21 * x + 20 * y + 4)) ) →
  (x = 4 ∨ 21 * x + 20 * y + 4 = 0 ) :=
sorry

end common_chord_eq_line_through_point_eq_l428_428582


namespace simplify_and_evaluate_l428_428780

noncomputable def simplified_expression (x : ℝ) : ℝ :=
  ((1 / (x - 1)) + (1 / (x + 1))) / (x^2 / (3 * x^2 - 3))

theorem simplify_and_evaluate : simplified_expression (Real.sqrt 2) = 3 * Real.sqrt 2 :=
by 
  sorry

end simplify_and_evaluate_l428_428780


namespace pyramid_top_positive_count_l428_428723

def sign_propagation (a b : ℤ) : ℤ := if a = b then 1 else -1

def top_value (a b c d e : ℤ) : ℤ :=
  sign_propagation
    (sign_propagation (sign_propagation a b * sign_propagation b c) (sign_propagation b c * sign_propagation c d))
    (sign_propagation (sign_propagation b c * sign_propagation c d) (sign_propagation c d * sign_propagation d e))

theorem pyramid_top_positive_count :
  (∑ a b c d e in ({-1, 1} : Finset ℤ),
    if top_value a b c d e = 1 then 1 else 0) = 17 := sorry

end pyramid_top_positive_count_l428_428723


namespace balloon_minimum_volume_l428_428431

theorem balloon_minimum_volume 
  (p V : ℝ)
  (h1 : p * V = 24000)
  (h2 : p ≤ 40000) : 
  V ≥ 0.6 :=
  sorry

end balloon_minimum_volume_l428_428431


namespace bisect_inaccessible_angle_with_ruler_l428_428231

-- Define the points and the existence of an inaccessible angle
variables (A B C : Type) [point A] [point B] [point C]

-- Assume the vertex A is inaccessible
axiom inaccessible_vertex : ∀ (A : point), inaccessible A

-- Definition of able to use a two-sided ruler
def two_sided_ruler : Type := sorry  -- Definition to be provided

-- Define the angle bisector construction using a two-sided ruler
theorem bisect_inaccessible_angle_with_ruler (A B C : point) (R : two_sided_ruler) :
  ∃ (L : line), bisects (∠BAC) L :=
sorry

end bisect_inaccessible_angle_with_ruler_l428_428231


namespace direct_proportional_function_point_l428_428261

theorem direct_proportional_function_point 
    (h₁ : ∃ k : ℝ, ∀ x : ℝ, (2, -3).snd = k * (2, -3).fst)
    (h₂ : ∃ k : ℝ, ∀ x : ℝ, (4, -6).snd = k * (4, -6).fst)
    : (∃ k : ℝ, k = -(3 / 2)) :=
by
  sorry

end direct_proportional_function_point_l428_428261


namespace length_of_first_train_l428_428823

noncomputable def first_train_length 
  (speed_first_train_km_h : ℕ) 
  (speed_second_train_km_h : ℕ) 
  (length_second_train_m : ℕ) 
  (time_seconds : ℝ) 
  (relative_speed_m_s : ℝ) : ℝ :=
  let relative_speed_mps := (speed_first_train_km_h + speed_second_train_km_h) * (5 / 18)
  let distance_covered := relative_speed_mps * time_seconds
  let length_first_train := distance_covered - length_second_train_m
  length_first_train

theorem length_of_first_train : 
  first_train_length 40 50 165 11.039116870650348 25 = 110.9779217662587 :=
by 
  rw [first_train_length]
  sorry

end length_of_first_train_l428_428823


namespace average_monthly_balance_l428_428770

def balances : list ℕ := [120, 150, 180, 150, 210, 180]

def number_of_months : ℕ := 6

noncomputable def total_balance : ℕ := balances.sum

theorem average_monthly_balance : (total_balance / number_of_months) = 165 := by
  sorry

end average_monthly_balance_l428_428770


namespace num_rectangles_in_5x5_grid_l428_428628

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428628


namespace similar_triangles_XY_length_l428_428388

-- Defining necessary variables.
variables (PQ QR YZ XY : ℝ) (area_XYZ : ℝ)

-- Given conditions to be used in the proof.
def condition1 : PQ = 8 := sorry
def condition2 : QR = 16 := sorry
def condition3 : YZ = 24 := sorry
def condition4 : area_XYZ = 144 := sorry

-- Statement of the mathematical proof problem to show XY = 12
theorem similar_triangles_XY_length :
  PQ = 8 → QR = 16 → YZ = 24 → area_XYZ = 144 → XY = 12 :=
by
  intros hPQ hQR hYZ hArea
  sorry

end similar_triangles_XY_length_l428_428388


namespace total_payment_difference_l428_428948

noncomputable def accumulated_amount 
(P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
P * (1 + r / n) ^ (n * t)

theorem total_payment_difference
  (P : ℝ := 15000)
  (r₁ r₂ : ℝ := 0.1, 0.12)
  (n₁ n₂ : ℕ := 2, 1)
  (t₁ t₂ : ℕ := 10, 10)
  (payment1_year5 : ℝ := 0.5)
  (payment1_year7 : ℝ := 1/3) :
  | (accumulated_amount P r₂ n₂ t₂) - 
    (accumulated_amount P r₁ n₁ 5 * payment1_year5 +
     accumulated_amount (P * (1 - payment1_year5)) r₁ n₁ 2 * payment1_year7 +
     accumulated_amount (P * (1 - payment1_year5) * (1 - payment1_year7)) r₁ n₁ 3) | 
  = 16155 :=
  sorry

end total_payment_difference_l428_428948


namespace semicircle_radius_of_triangle_l428_428462

-- Definitions of the variables and conditions
variables {a b r : ℝ} (α : ℝ)

-- The theorem stating the given problem and its corresponding proof
theorem semicircle_radius_of_triangle
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (alpha_in_deg : α ∈ Icc 0 (2 * π)) :
  r = (a * b * Real.sin α) / (a + b) := sorry

end semicircle_radius_of_triangle_l428_428462


namespace plane_passing_through_points_l428_428038

def point := ℝ × ℝ × ℝ

def A : point := (-1, 2, -3)
def B : point := (3, 4, 1)
def F : point := (-4, 8, -3)

def plane_eq (x y z : ℝ) : ℝ := 5 * z - 2 * y - 4 * x + 15

theorem plane_passing_through_points : 
  ∃ plane_eq : point → ℝ,
    plane_eq A.1 A.2 A.3 = 0 ∧
    plane_eq B.1 B.2 B.3 = 0 ∧
    plane_eq F.1 F.2 F.3 = 0 :=
sorry

end plane_passing_through_points_l428_428038


namespace range_of_a_l428_428548

theorem range_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  -real.sqrt 6 / 3 ≤ a ∧ a ≤ real.sqrt 6 / 3 :=
  sorry

end range_of_a_l428_428548


namespace grid_sign_flipping_l428_428725

theorem grid_sign_flipping (m n : ℕ) 
  (grid : Fin m.succ × Fin n.succ → Bool) -- +1 represented as true, -1 represented as false
  (can_flip : ∀ i j : ℕ, i < m.succ ∧ j < n.succ → (grid ⟨i, _⟩ ⟨j, _⟩ ↔ ¬ grid ⟨m - i, _⟩ ⟨n - j, _⟩)) :
  (∃ (k l : ℕ), m = 4 * k ∧ n = 4 * l) ↔ 
  (∃ flip_operations : ℕ, ∀ i j : ℕ, i < m.succ ∧ j < n.succ → grid ⟨i, _⟩ ⟨j, _⟩ = ¬ (grid ⟨i, _⟩ ⟨j, _⟩)) :=
by
  sorry

end grid_sign_flipping_l428_428725


namespace wire_length_before_cut_l428_428082

-- Defining the conditions
def wire_cut (L S : ℕ) : Prop :=
  S = 20 ∧ S = (2 / 5 : ℚ) * L

-- The statement we need to prove
theorem wire_length_before_cut (L S : ℕ) (h : wire_cut L S) : (L + S) = 70 := 
by 
  sorry

end wire_length_before_cut_l428_428082


namespace num_rectangles_grid_l428_428609

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428609


namespace num_rectangles_grid_l428_428610

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428610


namespace train_cross_time_l428_428413

-- Definitions based on conditions
def train_length : ℝ := 110 -- in meters
def bridge_length : ℝ := 390 -- in meters
def speed_kmph : ℝ := 60 -- in kilometers per hour

-- Convert speed from kmph to m/s
def speed_mps (kmph : ℝ) : ℝ := kmph * 1000 / 3600

-- Calculate total distance
def total_distance (train_len bridge_len : ℝ) : ℝ := train_len + bridge_len

-- Calculate time to cross the bridge
def time_to_cross (distance speed : ℝ) : ℝ := distance / speed

-- Prove the time to cross is approximately 30 seconds
theorem train_cross_time :
  let speed := speed_mps speed_kmph in
  let distance := total_distance train_length bridge_length in
  time_to_cross distance speed ≈ 30 :=
by
  sorry

end train_cross_time_l428_428413


namespace p_p_eq_twenty_l428_428322

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else if x ≥ 0 ∧ y < 0 then 4 * x + 2 * y
  else 3 * x + 2 * y

theorem p_p_eq_twenty : p (p 2 (-3)) (p (-3) (-4)) = 20 :=
by
  sorry

end p_p_eq_twenty_l428_428322


namespace correct_calculation_l428_428049

theorem correct_calculation :
  ∃ x : ℤ, (x - 23 = 4) ∧ (x * 23 = 621) :=
begin
  sorry
end

end correct_calculation_l428_428049


namespace independent_events_probability_l428_428200

variables {Ω : Type*} {P : MeasureTheory.ProbabilityMeasure Ω}
variables {A B : Set Ω}

theorem independent_events_probability (hA : P A = 1/3) 
                                       (hB : P B = 3/4)
                                       (h_indep : MeasureTheory.IndepEvents P A B) :
  P (A ∩ Bᶜ) = 1/12 := by
  sorry

end independent_events_probability_l428_428200


namespace rectangles_not_all_similar_l428_428409

theorem rectangles_not_all_similar : ¬ ∀ (r1 r2 : rectangle), similar r1 r2 := by
  -- Define a rectangle as a quadrilateral with opposite sides equal
  -- Define similarity of rectangles
  -- Use the properties of similarity to show that there exist rectangles not similar
  sorry

end rectangles_not_all_similar_l428_428409


namespace solve_system_of_equations_l428_428129

-- Definition of the system of equations as conditions
def eq1 (x y : ℤ) : Prop := 3 * x + y = 2
def eq2 (x y : ℤ) : Prop := 2 * x - 3 * y = 27

-- The theorem claiming the solution set is { (3, -7) }
theorem solve_system_of_equations :
  ∀ x y : ℤ, eq1 x y ∧ eq2 x y ↔ (x, y) = (3, -7) :=
by
  sorry

end solve_system_of_equations_l428_428129


namespace water_pouring_problem_l428_428449

theorem water_pouring_problem : 
  (∀ n : ℕ, (n > 0) → (∏ i in Finset.range n, (1 - 1 / (i + 3))) = (5 / 2)) → 
  ∃ n : ℕ, (n > 0) ∧ (2 / (n + 2) = 1 / 5) :=
by
  sorry

end water_pouring_problem_l428_428449


namespace approx_time_for_600_parts_l428_428218

theorem approx_time_for_600_parts :
  ∀ (x y : ℝ), (x = 600) → ( ∀ (x' : ℝ), (y = 0.01 * x' + 0.5) ) → (y = 6.5) :=
by
  intros x y hx hxy
  rw hx at hxy
  specialize hxy 600
  rw hx
  exact hxy

end approx_time_for_600_parts_l428_428218


namespace alice_savings_l428_428084

def sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.10

theorem alice_savings :
  (basic_salary + (sales * commission_rate)) * savings_rate = 29 :=
by
  sorry

end alice_savings_l428_428084


namespace six_valid_rectangular_formations_l428_428878

theorem six_valid_rectangular_formations : 
  {t : ℕ // 12 ≤ t ∧ t ≤ 50 ∧ 360 % t = 0 ∧ 12 ≤ 360 / t}.to_finset.card = 6 :=
by
  sorry

end six_valid_rectangular_formations_l428_428878


namespace trig_identity_l428_428103

theorem trig_identity :
  sin (65: ℝ) * cos (35: ℝ) - sin (25: ℝ) * sin (35: ℝ) = 1/2 :=
by
  sorry

end trig_identity_l428_428103


namespace f_g_2_equals_169_l428_428700

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x^2 + x + 3

-- The theorem statement
theorem f_g_2_equals_169 : f (g 2) = 169 :=
by
  sorry

end f_g_2_equals_169_l428_428700


namespace Y_lies_on_median_BM_l428_428292

variable {Ω1 Ω2 : Type}
variable {A B C M : Ω2}
variable [EuclideanGeometry Ω2]

-- Definitions coming from conditions
variable (Y : Ω2)
variable (hY1 : Y ∈ circle_omega1) (hY2 : Y ∈ circle_omega2)
variable (hSameSide : SameSide Y B (Line AC))

-- The theorem we want to prove
theorem Y_lies_on_median_BM :
  LiesOnMedian Y B M := 
  sorry

end Y_lies_on_median_BM_l428_428292


namespace minimum_value_fraction_l428_428557

theorem minimum_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : log 2 ^ x + log 8 ^ y = log 2) : 
  (1 / x + 1 / (3 * y) >= 4) := 
sorry

end minimum_value_fraction_l428_428557


namespace Y_pdf_from_X_pdf_l428_428368

/-- Given random variable X with PDF p(x), prove PDF of Y = X^3 -/
noncomputable def X_pdf (σ : ℝ) (x : ℝ) : ℝ := 
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x ^ 2) / (2 * σ ^ 2))

noncomputable def Y_pdf (σ : ℝ) (y : ℝ) : ℝ :=
  (1 / (3 * σ * y^(2/3) * Real.sqrt (2 * Real.pi))) * Real.exp (-(y^(2/3)) / (2 * σ ^ 2))

theorem Y_pdf_from_X_pdf (σ : ℝ) (y : ℝ) :
  ∀ x : ℝ, X_pdf σ x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x ^ 2) / (2 * σ ^ 2)) →
  Y_pdf σ y = (1 / (3 * σ * y^(2/3) * Real.sqrt (2 * Real.pi))) * Real.exp (-(y^(2/3)) / (2 * σ ^ 2)) :=
sorry

end Y_pdf_from_X_pdf_l428_428368


namespace number_of_rectangles_in_grid_l428_428595

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428595


namespace longer_side_of_rectangle_l428_428447

theorem longer_side_of_rectangle 
  (radius : ℝ) (A_rectangle : ℝ) (shorter_side : ℝ) 
  (h1 : radius = 6)
  (h2 : A_rectangle = 3 * (π * radius^2))
  (h3 : shorter_side = 2 * 2 * radius) :
  (A_rectangle / shorter_side) = 4.5 * π :=
by
  sorry

end longer_side_of_rectangle_l428_428447


namespace solution_l428_428403

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def sum_primes_in_range (a b : ℕ) : ℕ :=
  (list.range' a (b - a + 1)).filter is_prime |>.sum

def three_times_sum_primes_15_to_30 : ℕ :=
  3 * sum_primes_in_range 15 30

theorem solution : three_times_sum_primes_15_to_30 = 264 := by
  sorry

end solution_l428_428403


namespace ball_drawing_cases_l428_428034

theorem ball_drawing_cases :
  let balls := finset.range 12 \ (finset.range 1) in
  let favorable_first_draw := { b | b < 3 } in
  let favorable_second_draw := { b | b >= 10 } in
  (favorable_first_draw.card) * (favorable_second_draw.card) = 6 := by
  let balls := finset.range 12
  let favorable_first_draw := balls.filter (λ b, b < 3)
  let favorable_second_draw := balls.filter (λ b, b >= 10)
  have h1 : favorable_first_draw.card = 2 := by
    -- proof step skipped
    sorry
  have h2 : favorable_second_draw.card = 3 := by
    -- proof step skipped
    sorry
  calc
    favorable_first_draw.card * favorable_second_draw.card
      = 2 * 3 : by rw [h1, h2]
      = 6 : by norm_num

end ball_drawing_cases_l428_428034


namespace number_of_rectangles_in_5x5_grid_l428_428655

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428655


namespace average_disk_space_per_hour_eq_56_l428_428873

noncomputable def musicLibraryDays : ℕ := 15
noncomputable def totalDiskSpaceMB : ℕ := 20000

theorem average_disk_space_per_hour_eq_56 :
  (totalDiskSpaceMB / (musicLibraryDays * 24) : ℝ).round = 56 :=
by
  sorry

end average_disk_space_per_hour_eq_56_l428_428873


namespace probability_of_odd_divisor_of_factorial_l428_428495

theorem probability_of_odd_divisor_of_factorial :
  let total_divisors := (11 + 1) * (6 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1),
      odd_divisors := (6 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1),
      probability := odd_divisors / total_divisors
  in probability = 1 / 6 :=
by
  sorry

end probability_of_odd_divisor_of_factorial_l428_428495


namespace remainder_calculation_l428_428970

theorem remainder_calculation :
  (7 * 10^23 + 3^25) % 11 = 5 :=
by
  sorry

end remainder_calculation_l428_428970


namespace mod_equivalence_of_neg2357_l428_428399

theorem mod_equivalence_of_neg2357 : ∃ n : ℤ, (0 ≤ n ∧ n < 9) ∧ (-2357 ≡ n [ZMOD 9]) :=
by
  use 6
  split
  { split
    { norm_num }
    { norm_num } }
  { norm_num }

end mod_equivalence_of_neg2357_l428_428399


namespace problem_intersection_l428_428549

def setA : Set ℕ := { x | -1 ≤ x ∧ x < 4 }
def setB : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def intersectionA_B : Set ℕ := { x | -1 < x ∧ x < 3 }

theorem problem_intersection :
  (setA ∩ setB : Set ℕ) = intersectionA_B := 
sorry

end problem_intersection_l428_428549


namespace compute_product_sum_l428_428927

theorem compute_product_sum (a b c : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  (a * b * c) * ((1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c) = 47 :=
by
  sorry

end compute_product_sum_l428_428927


namespace number_of_terms_in_arithmetic_sequence_l428_428686

theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, list.length (list.filter (λ k, ∃ m : ℕ, k = 1 + 3 * m) (list.range (2008 + 1))) = n ∧ n = 670 :=
begin
  sorry
end

end number_of_terms_in_arithmetic_sequence_l428_428686


namespace tape_shortfall_l428_428273

theorem tape_shortfall (total_tape : ℝ) (length width : ℝ) (perimeter : ℝ) (shortfall : ℝ) :
  total_tape = 180 → 
  length = 80 → 
  width = 35 →
  perimeter = 2 * (length + width) →
  perimeter > total_tape →
  shortfall = perimeter - total_tape →
  shortfall = 50 :=
by
  intros h_tot h_len h_wid h_perim h_comp h_short
  rw [h_tot, h_len, h_wid, ←h_perim, ←h_short]
  norm_num
  sorry

end tape_shortfall_l428_428273


namespace num_rectangles_in_5x5_grid_l428_428634

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428634


namespace brother_birth_year_1990_l428_428274

variable (current_year : ℕ) -- Assuming the current year is implicit for the problem, it should be 2010 if Karina is 40 years old.
variable (karina_birth_year : ℕ)
variable (karina_current_age : ℕ)
variable (brother_current_age : ℕ)
variable (karina_twice_of_brother : Prop)

def karinas_brother_birth_year (karina_birth_year karina_current_age brother_current_age : ℕ) : ℕ :=
  karina_birth_year + brother_current_age

theorem brother_birth_year_1990 
  (h1 : karina_birth_year = 1970) 
  (h2 : karina_current_age = 40) 
  (h3 : karina_twice_of_brother) : 
  karinas_brother_birth_year 1970 40 20 = 1990 := 
by
  sorry

end brother_birth_year_1990_l428_428274


namespace sum_within_bounds_l428_428007

theorem sum_within_bounds (n : ℕ) (a : Fin n → ℝ) (h_sum : (∑ i, a i) = 99)
    (h_bound : ∀ i, |a i| ≤ 3) :
    ∃ S : Finset (Fin n), 32 < ∑ i in S, a i ∧ ∑ i in S, a i < 34 := 
sorry

end sum_within_bounds_l428_428007


namespace m_range_l428_428559

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^3 - 2 * m * x^2 - m * x

def Omega_1 (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, (∀ x > 0, g x = f x / x ∧ ∀ x y > 0, x < y → g x ≤ g y)

def Omega_2 (f : ℝ → ℝ) : Prop := ∃ h : ℝ → ℝ, (∀ x > 0, h x = f x / (x^2) ∧ ∀ x y > 0, x < y → h x ≤ h y)

theorem m_range (m : ℝ) : Omega_1 (f x m) ∧ ¬ Omega_2 (f x m) → m ∈ set.Iio 0 :=
sorry

end m_range_l428_428559


namespace general_solution_of_differential_equation_l428_428967

theorem general_solution_of_differential_equation (a₀ : ℝ) (x : ℝ) :
  ∃ y : ℝ → ℝ, (∀ x, deriv y x = (y x)^2) ∧ y x = a₀ / (1 - a₀ * x) :=
sorry

end general_solution_of_differential_equation_l428_428967


namespace angle_BFG_is_60_l428_428256
-- Importing the necessary library

-- Definitions according to the conditions
axiom equilateral_triangle (A B C : Type) (ABC : triangle A B C) : Prop :=
∀ (a b c : A), angle a b c = 60

axiom inscribed_equilateral (A B C D E F : Type) (ABC : triangle A B C) (DEF : triangle D E F) : Prop :=
(D ∈ line_segment A B) ∧ (E ∈ line_segment B C) ∧ (F ∈ line_segment C A) ∧ (equilateral_triangle D E F)

axiom triangle_on_segment (A B G : Type) (D F G : Type) (ABG : triangle A B G) (DFG : triangle D F G) : Prop :=
(G ∈ line_segment A B) ∧ (equilateral_triangle D F G)

-- Angle at BFG condition
def angle_BFG_eq (A B C D E F G : Type) (ABC : triangle A B C) (DEF : triangle D E F) (DFG : triangle D F G): Prop :=
angle B F G = 60

-- Prove statement
theorem angle_BFG_is_60 (A B C D E F G : Type) (ABC : triangle A B C) (DEF : triangle D E F) (DFG : triangle D F G)
(insc_eq : inscribed_equilateral A B C D E F ABC DEF)
(tri_seg : triangle_on_segment A B G D F G ABC DFG) :
angle_BFG_eq A B C D E F G ABC DEF DFG :=
by sorry

end angle_BFG_is_60_l428_428256


namespace rectangles_in_grid_l428_428667

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428667


namespace find_equation_of_perpendicular_line_l428_428516

noncomputable theory

open_locale classical

def line1 (x y : ℝ) : Prop := 7 * x - 8 * y - 1 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 17 * y + 9 = 0
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 7 = 0

def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y
def desired_line (x y : ℝ) : Prop := 27 * x + 54 * y + 37 = 0

theorem find_equation_of_perpendicular_line :
  ∃ (x y : ℝ), intersection_point x y ∧ desired_line x y :=
sorry

end find_equation_of_perpendicular_line_l428_428516


namespace find_original_price_l428_428056

theorem find_original_price (SP GP : ℝ) (h_SP : SP = 1150) (h_GP : GP = 27.77777777777778) :
  ∃ CP : ℝ, CP = 900 :=
by
  sorry

end find_original_price_l428_428056


namespace circle_radius_tangent_l428_428870

theorem circle_radius_tangent (a : ℝ) (R : ℝ) (h1 : a = 25)
  (h2 : ∀ BP DE CP CE, BP = 2 ∧ DE = 2 ∧ CP = 23 ∧ CE = 23 ∧ BP + CP = a ∧ DE + CE = a)
  : R = 17 :=
sorry

end circle_radius_tangent_l428_428870


namespace domain_of_composite_function_l428_428213

theorem domain_of_composite_function
    (f : ℝ → ℝ)
    (h : ∀ x, -2 ≤ x ∧ x ≤ 3 → f (x + 1) ∈ (Set.Icc (-2:ℝ) (3:ℝ))):
    ∃ s : Set ℝ, s = Set.Icc 0 (5/2) ∧ (∀ x, x ∈ s ↔ f (2 * x - 1) ∈ Set.Icc (-1) 4) :=
by
  sorry

end domain_of_composite_function_l428_428213


namespace arithmetic_progression_value_l428_428204

variable (a : ℕ → ℝ)
variable (d : ℝ)

def is_arithmetic_progression (a: ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

def is_geometric_progression (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_progression_value
  (h_arith: is_arithmetic_progression a)
  (h_initial: a 1 = 1)
  (h_common_diff: d ≠ 0)
  (h_geometric: is_geometric_progression (a 1) (a 2) (a 5)) :
  a 2015 = 4029 := sorry

end arithmetic_progression_value_l428_428204


namespace general_formula_magnitude_comparison_l428_428179

open Classical

variable (t : ℝ) (h_t : t ≠ 1) (h_tNeqNeg1 : t ≠ -1) (h_tPos : 0 < t) (n : ℕ)

def a (n : ℕ) : ℝ :=
  if n = 1 then 2 * t - 3 else
    have ih : ∀ k, 1 ≤ k ∧ k ≤ n → k ≠ 1 → a k = (2 * (t ^ k - 1)) / k - 1 from sorry,
    (2 * t ^ (n + 1) - 3) * (a n) + 2 * (t - 1) * t ^ n - 1) / (a n + 2 * t ^ n - 1)

theorem general_formula (t : ℝ) (h_t : t ≠ 1) (n : ℕ) :
  a t n = (2 * (t ^ n - 1)) / n - 1 := sorry

theorem magnitude_comparison (t : ℝ) (h_tPos : 0 < t) (h_tNeqNeg1 : t ≠ -1) (n : ℕ) :
  a t (n + 1) > a t n := sorry

end general_formula_magnitude_comparison_l428_428179


namespace number_of_blue_candles_l428_428934

-- Conditions
def grandfather_age : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def total_candles : ℕ := grandfather_age
def yellow_red_candles : ℕ := yellow_candles + red_candles
def blue_candles : ℕ := total_candles - yellow_red_candles

-- Proof statement
theorem number_of_blue_candles : blue_candles = 38 :=
by
  -- sorry indicates the proof is omitted
  sorry

end number_of_blue_candles_l428_428934


namespace central_value_is_mean_l428_428433

theorem central_value_is_mean {α : Type*} [normed_group α] [probability_space α] [symm_dist : symmetric_distribution α μ] 
  (P : probability_distribution α) (d : ℝ) (h68 : P.measure (Icc (μ - d) (μ + d)) = 0.68) 
  (h84 : P.measure (Iic (μ + d)) = 0.84) : 
  P.mean = μ :=
sorry

end central_value_is_mean_l428_428433


namespace hyperbola_eccentricity_l428_428703

theorem hyperbola_eccentricity (a b c : ℝ) (h : (c^2 - a^2 = 5 * a^2)) (hb : a / b = 2) :
  (c / a = Real.sqrt 5) :=
by
  sorry

end hyperbola_eccentricity_l428_428703


namespace BF_bisects_angle_PBC_tan_angle_PCB_l428_428245

variable {P B C A D E F : Type*}

/-- Problem (a) -/
theorem BF_bisects_angle_PBC (h1 : ∠ PBC = 60) 
  (h2 : tangent P (circumcircle ⟨P, B, C⟩) ∩ CB = A)
  (h3 : D ∈ PA ∧ E ∈ circumcircle ⟨P, B, C⟩ ∧ ∠DBE = 90 ∧ PD = PE)
  (h4 : BE ∩ PC = F)
  (h5 : are_concurrent [AF, BP, CD]) :
  is_angle_bisector BF (∠ PBC) := sorry

/-- Problem (b) -/
theorem tan_angle_PCB (h1 : ∠ PBC = 60) 
  (h2 : tangent P (circumcircle ⟨P, B, C⟩) ∩ CB = A)
  (h3 : D ∈ PA ∧ E ∈ circumcircle ⟨P, B, C⟩ ∧ ∠DBE = 90 ∧ PD = PE)
  (h4 : BE ∩ PC = F)
  (h5 : are_concurrent [AF, BP, CD]) :
  tan (∠ PCB) = (6 + sqrt 3) / 11 := sorry

end BF_bisects_angle_PBC_tan_angle_PCB_l428_428245


namespace range_of_m_l428_428324

noncomputable def p (x : ℝ) : Prop := (x^3 - 4*x) / (2*x) ≤ 0
noncomputable def q (x m : ℝ) : Prop := (x^2 - (2*m + 1)*x + m^2 + m) ≤ 0

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, p x → q x m) ∧ ¬ (∀ x : ℝ, p x → q x m) ↔ m ∈ Set.Ico (-2 : ℝ) (-1) ∪ Set.Ioc 0 (1 : ℝ) :=
by
  sorry

end range_of_m_l428_428324


namespace count_rectangles_5x5_l428_428680

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428680


namespace card_rearrangement_l428_428380

def rearrange_cost (n : ℕ) (a : Fin n → ℕ) : ℕ :=
  ∑ i in Finset.range n, abs (a i - (i + 1))

theorem card_rearrangement (n : ℕ) (a : Fin n → ℕ) :
  ∃ (f : Fin n → Fin n), ∀ i, f (f i) = i ∧ 
  rearrange_cost n (λ i, (f i).val + 1) ≤ rearrange_cost n a :=
sorry

end card_rearrangement_l428_428380


namespace vec_b_satisfies_conditions_l428_428752

open Matrix

def a : Vector3 ℝ := ![3, 2, 4]
def b : Vector3 ℝ := ![-30/58, 9/58, 128/58]

theorem vec_b_satisfies_conditions : 
  (a ⬝ b = 20) ∧ (a ⨯ b = ![10, -18, 14]) :=
by 
  sorry

end vec_b_satisfies_conditions_l428_428752


namespace intersection_of_A_and_B_l428_428993

def A : Set ℚ := { x | x^2 - 4*x + 3 < 0 }
def B : Set ℚ := { x | 2 < x ∧ x < 4 }

theorem intersection_of_A_and_B : A ∩ B = { x | 2 < x ∧ x < 3 } := by
  sorry

end intersection_of_A_and_B_l428_428993


namespace quadrilateral_comparison_l428_428115

noncomputable def distance (p q : ℝ × ℝ) : ℝ := real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def area_I : ℝ := 2 * 1
noncomputable def area_II : ℝ := 0.5 * 1 * 1 + 0.5 * 1 * 1

noncomputable def perimeter_I : ℝ := 2 + 2 + 1 + 1
noncomputable def perimeter_II : ℝ := 1 + distance (1,0) (1,1) + 1 + distance (1,1) (0,2)

theorem quadrilateral_comparison :
  area_I > area_II ∧ perimeter_I > perimeter_II :=
by
  -- Placeholder for the proof
  sorry

end quadrilateral_comparison_l428_428115


namespace alice_savings_l428_428088

-- Define the constants and conditions
def gadget_sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.1

-- State the theorem to be proved
theorem alice_savings : 
  let commission := gadget_sales * commission_rate in
  let total_earnings := basic_salary + commission in
  let savings := total_earnings * savings_rate in
  savings = 29 :=
by
  sorry

end alice_savings_l428_428088


namespace tangent_line_at_point_l428_428137

noncomputable def parabola (x : ℝ) : ℝ := 4 * x^2

theorem tangent_line_at_point (x : ℝ) (y : ℝ) (m : ℝ) :
  parabola x = y → P = (1/2, 1) →
  ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧ 
  a = 4 ∧ b = -1 ∧ c = -1 :=
by
  intro h_eq h_point
  use [4, -1, -1]
  sorry

end tangent_line_at_point_l428_428137


namespace binomial_expansion_properties_l428_428525

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| n, k := if k = 0 ∨ k = n then 1 else binomial_coefficient (n - 1) (k - 1) + binomial_coefficient (n - 1) k

noncomputable def term (n r : ℕ) (x : ℝ) : ℝ :=
  binomial_coefficient n r * (-x) ^ r

theorem binomial_expansion_properties (x : ℝ):
  -- Conditions
  let T_1000 := term 1999 999 x in
  let prop_1 := T_1000 = -binomial_coefficient 1999 999 * x ^ 999 in
  let sum_of_non_constant_terms := -1 in
  let prop_2 := sum_of_non_constant_terms = 1 in
  let largest_terms := term 1999 999 x ∧ term 1999 1000 x in
  let prop_3 := (largest_terms.term_1000 = largest_terms.term_1001) in
  let remainder_when_x_2000 := 1 in
  let prop_4 := remainder_when_x_2000 = (1-x)^1999 % 2000 = 1 in
  
  -- To be proved
  (prop_1 ∧ prop_4) ∧ ¬prop_2 ∧ ¬prop_3 :=
begin
  sorry
end

end binomial_expansion_properties_l428_428525


namespace hyperbola_quadratic_curve_l428_428168

def quadratic_curve (x y : ℝ) : Prop :=
  3 * y^2 - 4 * (x + 1) * y + 12 * (x - 2) = 0

theorem hyperbola_quadratic_curve :
  (∀ (x y : ℝ), quadratic_curve x y → ∃ a b c : ℝ, a*x^2 + b*y^2 + c = 1 ∧ a > 0 ∧ b < 0) :=
by
  intro x y h
  use [16, -16, -48] -- Example constants coherent with discriminant analysis; normally more complex derivation.
  split
  { 
    -- rewrite the quadratic equation in the hyperbola form, assumptive usage for illustration
    calc
    16 * x^2 - 16 * y^2 - 48 = 1 : sorry -- Omitted detailed symbolic transformation.
  }
  split
  { linarith }
  { linarith }

end hyperbola_quadratic_curve_l428_428168


namespace correct_answer_l428_428035

theorem correct_answer (a b c : ℝ) : a - (b + c) = a - b - c :=
by sorry

end correct_answer_l428_428035


namespace equal_if_sum_of_fourth_powers_equals_l428_428977

theorem equal_if_sum_of_fourth_powers_equals {
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a^4 + b^4 + c^4 + d^4 = 4 * a * b * c * d) :
  a = b ∧ b = c ∧ c = d :=
by
  sorry

end equal_if_sum_of_fourth_powers_equals_l428_428977


namespace jenicek_decorated_cookies_total_time_for_work_jenicek_decorating_time_l428_428327

/-- Conditions:
1. The grandmother decorates five gingerbread cookies for every cycle.
2. Little Mary decorates three gingerbread cookies for every cycle.
3. Little John decorates two gingerbread cookies for every cycle.
4. All three together decorated five trays, with each tray holding twelve gingerbread cookies.
5. Little John also sorted the gingerbread cookies onto trays twelve at a time and carried them to the pantry.
6. The grandmother decorates one gingerbread cookie in four minutes.
-/

def decorated_cookies_per_cycle := 10
def total_trays := 5
def cookies_per_tray := 12
def total_cookies := total_trays * cookies_per_tray
def babicka_cookies_per_cycle := 5
def marenka_cookies_per_cycle := 3
def jenicek_cookies_per_cycle := 2
def babicka_time_per_cookie := 4

theorem jenicek_decorated_cookies :
  (total_cookies - (total_cookies / decorated_cookies_per_cycle * marenka_cookies_per_cycle + total_cookies / decorated_cookies_per_cycle * babicka_cookies_per_cycle)) = 4 :=
sorry

theorem total_time_for_work :
  (total_cookies / decorated_cookies_per_cycle * babicka_time_per_cookie * babicka_cookies_per_cycle) = 140 :=
sorry

theorem jenicek_decorating_time :
  (4 / jenicek_cookies_per_cycle * babicka_time_per_cookie * babicka_cookies_per_cycle) = 40 :=
sorry

end jenicek_decorated_cookies_total_time_for_work_jenicek_decorating_time_l428_428327


namespace second_machine_time_l428_428074

theorem second_machine_time (x : ℝ) : 
  (600 / 10) + (1000 / x) = 1000 / 4 ↔ 
  1 / 10 + 1 / x = 1 / 4 :=
by
  sorry

end second_machine_time_l428_428074


namespace area_of_large_rectangle_excluding_hole_l428_428452

-- Definitions for the side lengths of the large rectangle and the hole.
def large_rectangle_length (x : ℝ) : ℝ := x + 7
def large_rectangle_width (x : ℝ) : ℝ := x + 5
def hole_length (x : ℝ) : ℝ := 2x - 3
def hole_width (x : ℝ) : ℝ := x - 2

-- Definitions for the area calculations.
def area_large_rectangle (x : ℝ) : ℝ := (x + 7) * (x + 5)
def area_hole (x : ℝ) : ℝ := (2x - 3) * (x - 2)
def remaining_area (x : ℝ) : ℝ := area_large_rectangle x - area_hole x

-- The statement to be proved.
theorem area_of_large_rectangle_excluding_hole (x : ℝ) : 
  remaining_area x = -x^2 + 19*x + 29 := 
by
  sorry

end area_of_large_rectangle_excluding_hole_l428_428452


namespace total_students_is_45_l428_428716

def num_students_in_class 
  (excellent_chinese : ℕ) 
  (excellent_math : ℕ) 
  (excellent_both : ℕ) 
  (no_excellent : ℕ) : ℕ :=
  excellent_chinese + excellent_math - excellent_both + no_excellent

theorem total_students_is_45 
  (h1 : excellent_chinese = 15)
  (h2 : excellent_math = 18)
  (h3 : excellent_both = 8)
  (h4 : no_excellent = 20) : 
  num_students_in_class excellent_chinese excellent_math excellent_both no_excellent = 45 := 
  by 
    sorry

end total_students_is_45_l428_428716


namespace number_of_rectangles_in_5x5_grid_l428_428653

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428653


namespace evaluate_expression_l428_428507

-- Define the base and the exponent
def base : ℝ := -64
def exponent : ℝ := -1/3

-- Define the target expression and its evaluated result
def target_expression := base ^ exponent
def evaluated_result := -1/4

-- The statement to prove: the target expression equals the evaluated result
theorem evaluate_expression : target_expression = evaluated_result :=
by sorry

end evaluate_expression_l428_428507


namespace num_rectangles_grid_l428_428618

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428618


namespace tommy_initial_balloons_l428_428017

theorem tommy_initial_balloons :
  ∃ x : ℝ, x + 78.5 = 132.25 ∧ x = 53.75 := by
  sorry

end tommy_initial_balloons_l428_428017


namespace solve_for_b_l428_428796

theorem solve_for_b (b : ℝ) (h_slope : (fun (y : ℝ) => y = (-2 / 3) * (y: ℝ) + 2))
  (h_perpendicular : (fun (y : ℝ) => y = (-b / 4) * (y: ℝ) + 5 / 4)) :
  b = -6 :=
by
  sorry

end solve_for_b_l428_428796


namespace problem_a4_inv_a4_eq_seven_l428_428699

theorem problem_a4_inv_a4_eq_seven (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^4 + (1/a)^4 = 7 :=
sorry

end problem_a4_inv_a4_eq_seven_l428_428699


namespace line_shift_upwards_l428_428241

theorem line_shift_upwards (x y : ℝ) (h : y = -2 * x) : y + 3 = -2 * x + 3 :=
by sorry

end line_shift_upwards_l428_428241


namespace reflected_graph_passes_through_point_l428_428214

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the given condition that y = f(x + 1) passes through (3, 2)
axiom h : f (3 + 1) = 2

-- Define the theorem to prove
theorem reflected_graph_passes_through_point (f : ℝ → ℝ) (h : f 4 = 2) : f 4 = -2 :=
begin
  sorry
end

end reflected_graph_passes_through_point_l428_428214


namespace interrogate_jones_first_l428_428837

def SmithAccusesBrownLying : Prop := ¬BrownTruth
def BrownAccusesJonesLying : Prop := ¬JonesTruth
def JonesClaimsSmithBrownLying : Prop := ¬SmithTruth ∧ ¬BrownTruth

theorem interrogate_jones_first (SmithTruth BrownTruth JonesTruth : Prop) 
  (H1 : SmithAccusesBrownLying)
  (H2 : BrownAccusesJonesLying)
  (H3 : JonesClaimsSmithBrownLying)
  (H4 : ¬SmithTruth)
  (H5 : ¬BrownTruth) :
  JonesTruth :=
by {
  sorry
}

end interrogate_jones_first_l428_428837


namespace sum_possible_students_l428_428461

theorem sum_possible_students :
  ∑ k in Finset.filter (λ s, s ≡ 1 [MOD 8]) (Finset.Icc 160 210), k = 1295 :=
by
  sorry

end sum_possible_students_l428_428461


namespace num_rectangles_in_5x5_grid_l428_428598

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428598


namespace num_rectangles_in_5x5_grid_l428_428607

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428607


namespace donation_value_l428_428423

def donation_in_yuan (usd: ℝ) (exchange_rate: ℝ): ℝ :=
  usd * exchange_rate

theorem donation_value :
  donation_in_yuan 1.2 6.25 = 7.5 :=
by
  -- Proof to be filled in
  sorry

end donation_value_l428_428423


namespace polynomial_no_real_roots_l428_428284

open Real

theorem polynomial_no_real_roots (n : ℕ) (c : Fin n → ℝ) :
  Even n →
  (∑ i in Finset.range n, |c i - 1|) < 1 →
  ∀ x : ℝ, 2 * x^n - ∑ i in Finset.range n, c i * x^(n - i) + 2 ≠ 0 :=
by
  intro h_even h_sum x
  sorry

end polynomial_no_real_roots_l428_428284


namespace bn_is_geometric_Sn_sum_lambda_mu_exist_l428_428734

noncomputable theory

-- Conditions
def a (n : ℕ) : ℝ 
-- a sequence with a condition
-- given a2 = 8
  | 2 := 8
  | n := sorry -- specify the sequence based on the given conditions

-- additional condition
def b (n : ℕ) : ℝ :=
  Real.log10 (a n + 1)

-- another condition
def c (n : ℕ) : ℝ :=
  2 * ((1 / a n) - (1 / a (n+1)))

-- Sequences
def S (n : ℕ) : ℝ := 
  (2^n - 1) * Real.log10 3

def T (n : ℕ) : ℝ :=
  2 * (1/2 - 1 / (3^(2^n) - 1))

-- Proofs required
theorem bn_is_geometric : ∀ n : ℕ, b (n+1) = 2 * b n :=
sorry

theorem Sn_sum : ∀ n : ℕ, S n = (2^n - 1) * Real.log10 3 :=
sorry

theorem lambda_mu_exist : ∃ λ μ : ℝ, λ ≠ 0 ∧ μ ≠ 0 ∧ λ = 3/2 ∧ μ = -1/2 ∧ (∀n : ℕ, T n + 1 / (λ * 10^(S n) + μ) = 1) :=
sorry

end bn_is_geometric_Sn_sum_lambda_mu_exist_l428_428734


namespace Y_lies_on_median_BM_l428_428293

variable {Ω1 Ω2 : Type}
variable {A B C M : Ω2}
variable [EuclideanGeometry Ω2]

-- Definitions coming from conditions
variable (Y : Ω2)
variable (hY1 : Y ∈ circle_omega1) (hY2 : Y ∈ circle_omega2)
variable (hSameSide : SameSide Y B (Line AC))

-- The theorem we want to prove
theorem Y_lies_on_median_BM :
  LiesOnMedian Y B M := 
  sorry

end Y_lies_on_median_BM_l428_428293


namespace cage_cost_proof_l428_428741

def toy_cost : ℝ := 8.77
def change : ℝ := 0.26
def bill : ℝ := 20.00

theorem cage_cost_proof : ∃ cage_cost : ℝ, cage_cost = bill - change ∧ cage_cost = 19.74 := by
  use bill - change
  simp [bill, change]
  norm_num
  sorry

end cage_cost_proof_l428_428741


namespace train_pass_time_l428_428467

noncomputable def time_to_pass (L: ℝ) (v_t: ℝ) (v_m: ℝ) : ℝ := 
  L / ((v_t * 1000 / 3600) + (v_m * 1000 / 3600))

theorem train_pass_time (L: ℝ) (v_t: ℝ) (v_m: ℝ) (hL: L = 250) (hv_t: v_t = 80) (hv_m: v_m = 12) :
  time_to_pass L v_t v_m ≈ 9.79 :=
by
  rw [time_to_pass, hL, hv_t, hv_m]
  norm_num
  sorry

end train_pass_time_l428_428467


namespace possible_areas_of_triangle_AMN_l428_428978

theorem possible_areas_of_triangle_AMN {M N A : Type} [LinearOrderedField M]
  (h_angle : A = π / 4)
  (h_moving_points : ∃ M ∈ {M' | M' moves on one side of ∠ A} ∃ N ∈ {N' | N' moves on the other side of ∠ A})
  (h_MN : ∀ M N, dist M N = 2) : 
  (∃ S, S = 1 ∨ S = 2) :=
begin
  sorry
end

end possible_areas_of_triangle_AMN_l428_428978


namespace count_rectangles_5x5_l428_428675

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428675


namespace smallest_prime_divisor_of_expression_l428_428027

theorem smallest_prime_divisor_of_expression :
  let N := 5 ^ (7 ^ (10 ^ (7 ^ 10))) + 1 in
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ N ∧ (∀ q : ℕ, Nat.Prime q ∧ q ∣ N → p ≤ q) :=
by
  let N := 5 ^ (7 ^ (10 ^ (7 ^ 10))) + 1
  use 2
  split
  · exact Nat.prime_two
  · split
  · sorry -- 2 divides N
  · intros q hq hdiv
    sorry -- finding smallest prime divisor

end smallest_prime_divisor_of_expression_l428_428027


namespace gcd_of_1230_and_920_is_10_l428_428799

theorem gcd_of_1230_and_920_is_10 : Int.gcd 1230 920 = 10 :=
sorry

end gcd_of_1230_and_920_is_10_l428_428799


namespace find_correct_fraction_l428_428253

theorem find_correct_fraction
  (mistake_frac : ℚ) (n : ℕ) (delta : ℚ)
  (correct_frac : ℚ) (number : ℕ)
  (h1 : mistake_frac = 5 / 6)
  (h2 : number = 288)
  (h3 : mistake_frac * number = correct_frac * number + delta)
  (h4 : delta = 150) :
  correct_frac = 5 / 32 :=
by
  sorry

end find_correct_fraction_l428_428253


namespace basketball_opponents_total_score_l428_428426

theorem basketball_opponents_total_score :
  ∃ (scores : Finset ℕ), scores = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} ∧
  (∀ n ∈ scores, 1 ≤ n ∧ n ≤ 12) ∧
  ∃ (lost_scores : Finset ℕ) (won_scores : Finset ℕ),
    lost_scores = {1, 2, 3, 4, 5, 6} ∧
    won_scores = {7, 8, 9, 10, 11, 12} ∧
    (∀ n ∈ lost_scores, ∃ m ∈ scores, m = n + 2) ∧
    (∀ n ∈ won_scores, ∃ m ∈ scores, m = n / 3) ∧
    ∑ x in lost_scores, (x + 2) + ∑ x in won_scores, (x / 3) = 52 := by
  sorry

end basketball_opponents_total_score_l428_428426


namespace sum_of_integers_satisfying_inequality_l428_428971

theorem sum_of_integers_satisfying_inequality : 
  (∑ n in (finset.filter (λ n : ℕ, 1.5 * n - 5 > 4.5) (finset.range 11)), n) = 34 :=
by
  sorry

end sum_of_integers_satisfying_inequality_l428_428971


namespace solution_set_system_l428_428374

theorem solution_set_system :
  { (x, y) | x + y = 1 ∧ x^2 - y^2 = 9 } = {(5, -4)} :=
by
  sorry

end solution_set_system_l428_428374


namespace polygon_area_inequality_l428_428458

-- Declare the areas of circles and polygon
variables {A B C : ℝ}

-- State the theorem we need to prove
theorem polygon_area_inequality (hA : ∃ r₁, A = π * r₁ ^ 2)
                                (hB : ∃ P, polygon P ∧ area P = B)
                                (hC : ∃ r₂, C = π * r₂ ^ 2) :
                                2 * B ≤ A + C :=
sorry

end polygon_area_inequality_l428_428458


namespace simplify_tan_cot_expr_l428_428781

theorem simplify_tan_cot_expr :
  ∀ (θ : ℝ), θ = 45 → (Real.tan θ)^4 + (Real.cot θ)^4 = ((Real.tan θ) + (Real.cot θ)) * 1 :=
by
  intro θ hθ
  have ht : Real.tan θ = 1 := by sorry
  have hc : Real.cot θ = 1 := by sorry
  calc 
  (Real.tan θ)^4 + (Real.cot θ)^4 = ((Real.tan θ)^4 + (Real.cot θ)^4) : by sorry
  ... = (1^4 + 1^4) : by sorry
  ... = (1 + 1) : by sorry
  ... = 1 * 2 : by sorry
  ... =  ((Real.tan θ) + (Real.cot θ)) := by sorry

end simplify_tan_cot_expr_l428_428781


namespace prove_primes_powers_of_two_l428_428133

open Function

-- Definition of the condition for the polynomial
def condition (n : ℕ) (P : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, m ≥ 1 → 
    let Pm := (Nat.iterate P m) in
       multiset.card (multiset.to_finset ((multiset.map Pm (multiset.range n)).map (λ x, x % n))) 
        = Nat.ceil (n / 2^m)

-- The statement to be proven
theorem prove_primes_powers_of_two (n : ℕ) : 
  (∃ P : ℕ → ℕ, condition n P) ↔ 
    (Nat.prime n ∨ ∃ k : ℕ, n = 2^k) :=
by
  sorry

end prove_primes_powers_of_two_l428_428133


namespace problem_solution_l428_428102

variable (x y : ℝ)

theorem problem_solution :
  (x - y + 1) * (x - y - 1) = x^2 - 2 * x * y + y^2 - 1 :=
by
  sorry

end problem_solution_l428_428102


namespace sequence_properties_l428_428372

-- Define the sequence a_n with given initial conditions and recurrence relation
def a : ℕ → ℕ
| 1     := 1
| 2     := 2
| (n+3) := (2 + (-1)^n) * a (n+1) + 2

-- Prove the required properties
theorem sequence_properties :
  (a 3 = 3) ∧
  (a 4 = 8) ∧
  (∀ n : ℕ, (a (2*n) + 1) / (a (2*(n-1)) + 1) = 3) ∧
  (∀ n : ℕ, ∑ i in finset.range (2*n), a (i+1) = (1/2) * (3^(n+1) - 3) + n^2 - n) :=
begin
  sorry
end

end sequence_properties_l428_428372


namespace probability_sum_not_less_than_6_is_half_l428_428717

def tetrahedral_faces : List ℕ := [1, 2, 3, 5]

def all_possible_sums : List ℕ := do
  x ← tetrahedral_faces
  y ← tetrahedral_faces
  pure (x + y)

def favorable_sums : List ℕ :=
  all_possible_sums.filter (λ x => x ≥ 6)

def probability_of_sum_not_less_than_6 : ℚ :=
  favorable_sums.length / all_possible_sums.length

theorem probability_sum_not_less_than_6_is_half : probability_of_sum_not_less_than_6 = 1 / 2 := by
  sorry

end probability_sum_not_less_than_6_is_half_l428_428717


namespace product_abc_l428_428317

variable (a b c : ℝ)

axiom h1 : a * b = 24 * Real.root 4 3
axiom h2 : a * c = 50 * Real.root 4 3
axiom h3 : b * c = 18 * Real.root 4 3

theorem product_abc : a * b * c = 120 * Real.root 4 3 := sorry

end product_abc_l428_428317


namespace area_of_inscribed_triangle_l428_428895

noncomputable def calculate_triangle_area_inscribed_in_circle 
  (arc1 : ℝ) (arc2 : ℝ) (arc3 : ℝ) (total_circumference := arc1 + arc2 + arc3)
  (radius := total_circumference / (2 * Real.pi))
  (theta := (2 * Real.pi) / total_circumference)
  (angle1 := 5 * theta) (angle2 := 7 * theta) (angle3 := 8 * theta) : ℝ :=
  0.5 * (radius ^ 2) * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem area_of_inscribed_triangle : 
  calculate_triangle_area_inscribed_in_circle 5 7 8 = 119.85 / (Real.pi ^ 2) :=
by
  sorry

end area_of_inscribed_triangle_l428_428895


namespace mod_exp_sub_6_l428_428109

theorem mod_exp_sub_6 :
  (47^2045 - 18^2045) % 6 = 5 := by
  have h1 : 47 % 6 = 5 := by norm_num
  have h2 : 18 % 6 = 0 := by norm_num
  have h3 : (-1)^2045 % 6 = 5 := by
    have h : 2045 % 2 = 1 := by norm_num
    have h_neg1_odd : (-1 : ℤ)^2045 = -1 := by
      rw [pow_odd, neg_one_pow_eq_pow_mod_two]
      exact h
    rw [h_neg1_odd, neg_mod]
    norm_num
  have h4 : 18^2045 % 6 = 0 := by
    rw [pow_zero', zero_mod] -- 18^2045 % 6 is 0 as any number ^ n % the number is 0
  rw [←h3, ←h4]
  conv_lhs
  {
    congr
    congr
    rw ←h1
    congr
    rw ←h2
    skip
    rw zero_mod
  }
  exact h3 -- Applying the earlier fact

  exact sorry

end mod_exp_sub_6_l428_428109


namespace increasing_interval_of_g_l428_428571

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (Real.pi / 3 - 2 * x)) -
  2 * (Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 - x))

noncomputable def g (x : ℝ) : ℝ :=
  f (x + Real.pi / 12)

theorem increasing_interval_of_g :
  ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2),
  ∃ a b, a = -Real.pi / 12 ∧ b = Real.pi / 4 ∧
      (∀ x y, (a ≤ x ∧ x ≤ y ∧ y ≤ b) → g x ≤ g y) :=
sorry

end increasing_interval_of_g_l428_428571


namespace counterexample_9918_l428_428940

-- Checking whether a given number is a counterexample.
def isCounterexample (n : ℕ) : Prop :=
  let sumDigits := n.digits.sum
  sumDigits % 27 = 0 ∧ n % 27 ≠ 0

-- The specific number to be tested
def num : ℕ := 9918

-- Prove that 9918 is a counterexample
theorem counterexample_9918 : isCounterexample num :=
  by
    -- Proof is omitted
    sorry

end counterexample_9918_l428_428940


namespace problem_solution_l428_428541

-- Definitions from conditions
def ellipse (a b : ℝ) : set (ℝ × ℝ) := {p | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1}
def eccentricity (a c : ℝ) : ℝ := c / a
def perimeter (a c : ℝ) : ℝ := 2*a + 2*c
def foci_distance (a : ℝ) : ℝ := 2*sqrt(a^2 - 1)

-- Given conditions
def ellipse_definition : Prop := 
  ∃ a b c : ℝ, 
  a > b ∧ b > 0 ∧ 
  eccentricity a c = 2*sqrt(2)/3 ∧ 
  perimeter a c = 6 + 4*sqrt(2) 

-- Questions:
-- 1. Find the equation of ellipse C
noncomputable def equation_of_ellipse_C : Prop := 
  ∀ a b c : ℝ, ellipse_definition → ellipse a b = {p : ℝ × ℝ | (p.1^2) / 9 + (p.2^2) / 1 = 1}

-- 2. Find the maximum area of ΔABD
def maximum_area_triangle_ABD : Prop := 
  ∀ k m : ℝ, 
  ∃ y1 y2 x1 x2 : ℝ, 
  ellipse_definition ∧ 
  (y1 + y2 = -2*k*m / (k^2 + 9)) ∧ 
  (y1*y2 = (m^2 - 9) / (k^2 + 9)) ∧  
  ((k^2 + 1)*y1*y2 + k*(m - 3)*(y1 + y2) + (m - 3)^2 = 0) → 
  ellipse 3 1 = {(p.1^2) / 9 + (p.2^2) / 1 = 1} → 
  (1/2 * (3/5) * sqrt((y1 + y2)^2 - 4*y1*y2) = 3/8) 

-- Combining both questions to form the theorem
theorem problem_solution : ellipse_definition → 
equation_of_ellipse_C ∧ maximum_area_triangle_ABD :=
by
  intro h,
  sorry

end problem_solution_l428_428541


namespace books_arrangement_l428_428726

-- Define the problem constants
def num_math_books : ℕ := 5
def num_history_books : ℕ := 4

-- Define the function to calculate the number of valid arrangements
def valid_arrangements : ℕ :=
  num_history_books * (num_history_books - 1) * (num_math_books + (num_history_books - 2))!

-- The goal is to prove that the number of valid arrangements equals 60,480
theorem books_arrangement : valid_arrangements = 60480 :=
by
  dsimp [valid_arrangements]
  norm_num
  sorry

end books_arrangement_l428_428726


namespace seating_arrangements_l428_428148

theorem seating_arrangements (n m k : Nat) (couples : Fin n -> Fin m -> Prop):
  let pairs : Nat := k
  let adjusted_pairs : Nat := pairs / 24
  adjusted_pairs = 5760 := by
  sorry

end seating_arrangements_l428_428148


namespace picnic_adults_children_difference_l428_428456

theorem picnic_adults_children_difference :
  ∃ (M W A C : ℕ),
    (M = 65) ∧
    (M = W + 20) ∧
    (A = M + W) ∧
    (C = 200 - A) ∧
    ((A - C) = 20) :=
by
  sorry

end picnic_adults_children_difference_l428_428456


namespace radius_of_tangent_circles_l428_428851

-- Definitions of the circles and their properties
variables 
  (O O1 O2 : Type)  -- Centers of the circles
  (r R : ℝ)         -- Radii of the circles
  (A B : O)         -- Points of tangency

-- Conditions
def properties : Prop :=
  -- The third circle has radius R
  R = 8 ∧
  -- The circles with centers at O1 and O2 both have radius r
  (∃ (r : ℝ), r = dist O1 O2 / 2) ∧
  -- The circles with centers at O1 and O2 both touch externally
  (dist O1 O2 = 2 * r) ∧
  -- The distance AB between the points of tangency A and B is 12
  (dist A B = 12)

-- Main theorem stating that r = 24 based on the conditions
theorem radius_of_tangent_circles (h : properties O O1 O2 r R A B) : r = 24 :=
  by
  sorry

end radius_of_tangent_circles_l428_428851


namespace maximum_profit_selling_price_l428_428857

noncomputable def purchase_price : ℝ := 80
noncomputable def initial_selling_price : ℝ := 90
noncomputable def initial_units_sold : ℕ := 400
noncomputable def price_increase_effect : ℝ := 1
noncomputable def sales_decrease_effect : ℕ := 10

-- Define the selling price function
noncomputable def selling_price (x : ℝ) : ℝ := initial_selling_price + x

-- Define the sales volume function
noncomputable def sales_volume (x : ℝ) : ℕ := initial_units_sold - (sales_decrease_effect * int.to_nat x)

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := (selling_price x - purchase_price) * (sales_volume x)

-- Proof statement
theorem maximum_profit_selling_price : ∀ x : ℝ, (profit x <= profit 15) -> (selling_price 15 = 105) :=
sorry

end maximum_profit_selling_price_l428_428857


namespace num_rectangles_in_5x5_grid_l428_428633

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428633


namespace length_SQ_eq_3_l428_428263

-- Definitions based on conditions
variables {P Q R S T M : Point}
variable angle_RPQ bisecting_PQ : ∀ (a b c : Angle), (a ≺ b) → (b ≺ c) → a ≺ c
variable length_PQ : ℝ
variable length_MT : ℝ
variable angle_SQT : ℝ

-- Given conditions
axiom midpoint_M (h : midpoint M P Q)
axiom bisects_PS (h : bisecting_PQ (angle R P Q) (angle_P Q S))
axiom ST_parallel_PR (h : parallel ST PR)
axiom lengthPQ_eq_12 : length_PQ = 12
axiom lengthMT_eq_1 : length_MT = 1
axiom angle_SQT_is_120 : angle_SQT = 120

-- To Prove
theorem length_SQ_eq_3 : length SQ = 3 :=
sorry

end length_SQ_eq_3_l428_428263


namespace taehyung_collected_most_points_l428_428039

def largest_collector : Prop :=
  let yoongi_points := 7
  let jungkook_points := 6
  let yuna_points := 9
  let yoojung_points := 8
  let taehyung_points := 10
  taehyung_points > yoongi_points ∧ 
  taehyung_points > jungkook_points ∧ 
  taehyung_points > yuna_points ∧ 
  taehyung_points > yoojung_points

theorem taehyung_collected_most_points : largest_collector :=
by
  let yoongi_points := 7
  let jungkook_points := 6
  let yuna_points := 9
  let yoojung_points := 8
  let taehyung_points := 10
  sorry

end taehyung_collected_most_points_l428_428039


namespace rectangles_in_grid_l428_428673

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428673


namespace increasing_interval_shift_l428_428240

variable {α : Type*} [Preorder α] (f : α → α)

theorem increasing_interval_shift (h : ∀ x y : α, (-2 : α) < x → x < 3 → x ≤ y → y < 3 → f x ≤ f y) :
  ∀ x y : α, (-7 : α) < x → x < -2 → x ≤ y → y < -2 → f (x + 5) ≤ f (y + 5) :=
by
  sorry

end increasing_interval_shift_l428_428240


namespace area_of_inscribed_triangle_l428_428900

noncomputable def area_inscribed_triangle (r : ℝ) (a b c : ℝ) : ℝ :=
  1 / 2 * r^2 * (Real.sin a + Real.sin b + Real.sin c)

theorem area_of_inscribed_triangle : 
  ∃ (r a b c : ℝ),
    a + b + c = 2 * π ∧
    r = 10 / π ∧
    a = 5 * (18 * π / 180) ∧
    b = 7 * (18 * π / 180) ∧
    c = 8 * (18 * π / 180) ∧
    area_inscribed_triangle r a b c = 119.84 / π^2 :=
begin
  sorry
end

end area_of_inscribed_triangle_l428_428900


namespace number_of_ordered_pairs_xy_2007_l428_428001

theorem number_of_ordered_pairs_xy_2007 : 
  ∃ n, n = 6 ∧ (∀ x y : ℕ, x * y = 2007 → x > 0 ∧ y > 0) :=
sorry

end number_of_ordered_pairs_xy_2007_l428_428001


namespace sound_speed_new_rod_l428_428719

theorem sound_speed_new_rod (a b t1 t2 t3 t4 l : ℝ)
  (h1 : t1 + t2 + t3 = a)
  (h2 : t1 = 2 * (t2 + t3))
  (h3 : t1 + t4 + t3 = b)
  (h4 : t1 + t4 = 2 * t3)
  (ha : a ≠ b) : 
  let v := 3 * l / (2 * (b - a)) in 
  true := sorry

end sound_speed_new_rod_l428_428719


namespace probability_of_drawing_red_ball_l428_428259

def totalBalls : Nat := 3 + 5 + 2
def redBalls : Nat := 3
def probabilityOfRedBall : ℚ := redBalls / totalBalls

theorem probability_of_drawing_red_ball :
  probabilityOfRedBall = 3 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l428_428259


namespace expr_value_l428_428101

-- Define the given expression
def expr : ℕ := 11 - 10 / 2 + (8 * 3) - 7 / 1 + 9 - 6 * 2 + 4 - 3

-- Assert the proof goal
theorem expr_value : expr = 21 := by
  sorry

end expr_value_l428_428101


namespace solve_for_y_l428_428824

theorem solve_for_y : ∃ y : ℝ, (2 / 3) * y = 40 ∧ y = 60 :=
by
  use 60
  split
  · norm_num
  sorry

end solve_for_y_l428_428824


namespace cost_price_of_book_l428_428043

theorem cost_price_of_book
  (SP : Real)
  (profit_percentage : Real)
  (h1 : SP = 300)
  (h2 : profit_percentage = 0.20) :
  ∃ CP : Real, CP = 250 :=
by
  -- Proof of the statement
  sorry

end cost_price_of_book_l428_428043


namespace num_rectangles_in_5x5_grid_l428_428632

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428632


namespace rectangles_in_grid_l428_428674

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428674


namespace sandwiches_count_l428_428356

theorem sandwiches_count : 
  let total_sandwiches := 5 * 7 * 6,
      unwanted_turkey_swiss := 5,
      unwanted_rye_roast_beef := 6
  in total_sandwiches - unwanted_turkey_swiss - unwanted_rye_roast_beef = 199 := 
by 
  let total_sandwiches := 5 * 7 * 6
  let unwanted_turkey_swiss := 5
  let unwanted_rye_roast_beef := 6
  have h : total_sandwiches - unwanted_turkey_swiss - unwanted_rye_roast_beef = 199 := by sorry
  exact h

end sandwiches_count_l428_428356


namespace rectangle_perimeter_greater_than_16_l428_428169

theorem rectangle_perimeter_greater_than_16 (a b : ℝ) (h : a * b > 2 * a + 2 * b) (ha : a > 0) (hb : b > 0) : 
  2 * (a + b) > 16 :=
sorry

end rectangle_perimeter_greater_than_16_l428_428169


namespace group_ways_group_ways_calc_l428_428422

theorem group_ways (n : ℕ) : ℕ :=
  let num_ways := (2 * n)!
  (num_ways / (2^n * (n)!))

-- goal: number of ways to group 2n people into n teams of 2 is (2n)! / (2^n * n!)
theorem group_ways_calc (n : ℕ) : group_ways n = (2 * n)! / (2^n * (n)!) :=
sorry

end group_ways_group_ways_calc_l428_428422


namespace smallest_fraction_divides_exactly_l428_428026

theorem smallest_fraction_divides_exactly (a b c p q r m n : ℕ)
    (h1: a = 6) (h2: b = 5) (h3: c = 10) (h4: p = 7) (h5: q = 14) (h6: r = 21)
    (h1_frac: 6/7 = a/p) (h2_frac: 5/14 = b/q) (h3_frac: 10/21 = c/r)
    (h_lcm: m = Nat.lcm p (Nat.lcm q r)) (h_gcd: n = Nat.gcd a (Nat.gcd b c)) :
  (n/m) = 1/42 :=
by 
  sorry

end smallest_fraction_divides_exactly_l428_428026


namespace red_other_side_probability_is_one_l428_428862

/-- Definitions from the problem conditions --/
def total_cards : ℕ := 10
def green_both_sides : ℕ := 5
def green_red_sides : ℕ := 2
def red_both_sides : ℕ := 3
def red_faces : ℕ := 6 -- 3 cards × 2 sides each

/-- The theorem proves the probability is 1 that the other side is red given that one side seen is red --/
theorem red_other_side_probability_is_one
  (h_total_cards : total_cards = 10)
  (h_green_both : green_both_sides = 5)
  (h_green_red : green_red_sides = 2)
  (h_red_both : red_both_sides = 3)
  (h_red_faces : red_faces = 6) :
  1 = (red_faces / red_faces) :=
by
  -- Write the proof steps here
  sorry

end red_other_side_probability_is_one_l428_428862


namespace fifth_selected_is_01_l428_428080

/-
  Given:
  1. 20 individuals numbered from 01 to 20.
  2. Random number table: 
     - Row 1: 7816, 6572, 0802, 6314, 0702, 4369, 9728, 0198
     - Row 2: 3204, 9234, 4935, 8200, 3623, 4869, 6938, 7481
  3. Start selection from the 5th and 6th numbers in the 1st row.
  4. Move left to right, selecting two numbers at a time.
  5. Only numbers less than 20 are selected.
  6. Avoid repeating numbers.

  Prove:
  The number of the 5th individual selected is 01.
-/
theorem fifth_selected_is_01 :
  ∃ (selected : List ℕ),
    selected = [08, 02, 14, 07, 01] ∧
    nth_le selected 4 sorry = 01 :=
begin
  sorry
end

end fifth_selected_is_01_l428_428080


namespace sum_squares_difference_l428_428483

open BigOperators

theorem sum_squares_difference :
  let sumOfFirstNSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6
  let sumSquaresEvenNumbers : ℕ := 4 * sumOfFirstNSquares 100
  let sumSquaresOddNumbers : ℕ := ∑ k in Finset.range 100, (2 * k + 1) ^ 2
  sumSquaresEvenNumbers - sumSquaresOddNumbers = 20100 := by
  let n := 100
  have sumOfFirstNSquares := fun n => n * (n + 1) * (2 * n + 1) / 6
  have sumSquaresEven := 4 * sumOfFirstNSquares n
  have sumSquaresOdd := ∑ k in Finset.range n, (2 * k + 1) ^ 2
  show (sumSquaresEven - sumSquaresOdd) = 20100
  sorry

end sum_squares_difference_l428_428483


namespace car_return_speed_l428_428863

theorem car_return_speed (d : ℝ) (v_cd : ℝ) (v_avg : ℝ) (h1 : d = 150)
  (h2 : v_cd = 50) (h3 : v_avg = 40) :
  let r := 33.333333333333336 in (2 * d) / (d / v_cd + d / r) = v_avg :=
by {
  sorry
}

end car_return_speed_l428_428863


namespace processing_decision_l428_428438

-- Definitions of given conditions
def processing_fee (grade: Char) : ℤ :=
  match grade with
  | 'A' => 90
  | 'B' => 50
  | 'C' => 20
  | 'D' => -50
  | _   => 0

def processing_cost (branch: Char) : ℤ :=
  match branch with
  | 'A' => 25
  | 'B' => 20
  | _   => 0

structure FrequencyDistribution :=
  (gradeA : ℕ)
  (gradeB : ℕ)
  (gradeC : ℕ)
  (gradeD : ℕ)

def branchA_distribution : FrequencyDistribution :=
  { gradeA := 40, gradeB := 20, gradeC := 20, gradeD := 20 }

def branchB_distribution : FrequencyDistribution :=
  { gradeA := 28, gradeB := 17, gradeC := 34, gradeD := 21 }

-- Lean 4 statement for proof of questions
theorem processing_decision : 
  let profit (grade: Char) (branch: Char) := processing_fee grade - processing_cost branch
  let avg_profit (dist: FrequencyDistribution) (branch: Char) : ℤ :=
    (profit 'A' branch) * dist.gradeA / 100 +
    (profit 'B' branch) * dist.gradeB / 100 +
    (profit 'C' branch) * dist.gradeC / 100 +
    (profit 'D' branch) * dist.gradeD / 100
  (pA_branchA : Float := branchA_distribution.gradeA / 100.0) = 0.4 ∧
  (pA_branchB : Float := branchB_distribution.gradeA / 100.0) = 0.28 ∧
  avg_profit branchA_distribution 'A' = 15 ∧
  avg_profit branchB_distribution 'B' = 10 →
  avg_profit branchA_distribution 'A' > avg_profit branchB_distribution 'B'
:= by 
  sorry

end processing_decision_l428_428438


namespace treasure_count_base_10_l428_428881

theorem treasure_count_base_10 
  (sapphires_base_7 : ℕ := 6532) 
  (silverware_base_7 : ℕ := 1650) 
  (spices_base_7 : ℕ := 250) :
  let sapphires_base_10 := 2 + 3*7 + 5*7^2 + 6*7^3,
      silverware_base_10 := 0*7^0 + 5*7^1 + 6*7^2 + 1*7^3,
      spices_base_10 := 0*7^0 + 5*7^1 + 2*7^2 in
  sapphires_base_10 + silverware_base_10 + spices_base_10 = 3131 := by
    sorry

end treasure_count_base_10_l428_428881


namespace central_moments_second_central_moments_third_central_moments_fourth_l428_428128

variable (X : Type) [Probability X]  -- Type of random variable X

def mathExpectation (X : Type) [Probability X] : X → ℝ := sorry

variables (a : ℝ) (M : X → ℝ) (v1 v2 v3 v4 : ℝ)

-- Conditions
axiom mean_def : M(X) = a
axiom v1_def : v1 = M(X)
axiom v2_def : v2 = M(X^2)
axiom v3_def : v3 = M(X^3)
axiom v4_def : v4 = M(X^4)

-- Proof statements
theorem central_moments_second : μ2 = v2 - v1^2 :=
sorry

theorem central_moments_third : μ3 = v3 - 3*v1*v2 + 2*v1^3 :=
sorry

theorem central_moments_fourth : μ4 = v4 - 4*v1*v3 + 6*v1^2*v2 - 3*v1^4 :=
sorry

end central_moments_second_central_moments_third_central_moments_fourth_l428_428128


namespace magnitude_sub_eq_sqrt_3_l428_428195

variables {a b : EuclideanSpace ℝ (Fin 3)}

noncomputable def magnitude (v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  real.sqrt (norm_sq v)

noncomputable def angle (v w : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  real.acos (inner v w / (magnitude v * magnitude w))

-- conditions
axiom non_collinear_non_zero (h : a ≠ 0) (h' : b ≠ 0) : ¬collinear {a, b}
axiom magnitudes_one : magnitude a = 1 ∧ magnitude b = 1
axiom angle_120 : angle a b = real.pi / 3  -- 120 degrees is pi/3 radians

-- expression
noncomputable def magnitude_sub (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  magnitude (a - b)

-- theorem to prove
theorem magnitude_sub_eq_sqrt_3
  (h : a ≠ 0) (h' : b ≠ 0) 
  (h₁ : magnitude a = 1) (h₂ : magnitude b = 1) 
  (h₃ : angle a b = real.pi / 3) :
  magnitude_sub a b = real.sqrt 3 :=
sorry

end magnitude_sub_eq_sqrt_3_l428_428195


namespace sum_max_min_b_minus_a_l428_428992

open Real

theorem sum_max_min_b_minus_a : 
  ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ 8 ∧ 0 ≤ b ∧ b ≤ 8 ∧ b^2 = 16 + a^2 → 
  let max_val := 4 in
  let min_val := (8 - 4*sqrt 3) in
  max_val + min_val = 12 - 4*sqrt 3 :=
begin
  intros a b h,
  let max_val := 4,
  let min_val := 8 - 4*sqrt 3,
  exact 12 - 4*sqrt 3,
end

end sum_max_min_b_minus_a_l428_428992


namespace pounds_lost_per_month_l428_428104

variable (starting_weight : ℕ) (ending_weight : ℕ) (months_in_year : ℕ) 

theorem pounds_lost_per_month
    (h_start : starting_weight = 250)
    (h_end : ending_weight = 154)
    (h_months : months_in_year = 12) :
    (starting_weight - ending_weight) / months_in_year = 8 := 
sorry

end pounds_lost_per_month_l428_428104


namespace integral_f_l428_428155

def f (x : ℝ) : ℝ :=
if h : -1 ≤ x ∧ x ≤ 0 then x^2
else if h : 0 < x ∧ x < 1 then 1
else 0

theorem integral_f : ∫ x in -1..1, f x = 4 / 3 := 
sorry

end integral_f_l428_428155


namespace qudrilateral_diagonal_length_l428_428514

theorem qudrilateral_diagonal_length (A h1 h2 d : ℝ) 
  (h_area : A = 140) (h_offsets : h1 = 8) (h_offsets2 : h2 = 2) 
  (h_formula : A = 1 / 2 * d * (h1 + h2)) : 
  d = 28 :=
by
  sorry

end qudrilateral_diagonal_length_l428_428514


namespace tangential_circumcircle_tangent_l428_428736

variables {P : Type} [euclidean_geometry P] 

-- Given conditions for the tangential convex quadrilateral ABCD
variables (A B C D I K L M N E F X Y Z T : P)
variables (Γ : circle P I)

-- Hypotheses
hypothesis h1 : incircle_triangle I K L M N A B C D
hypothesis h2 : intersects A D B C E
hypothesis h3 : intersects A B C D F
hypothesis h4 : line_intersects X (line_through A B) (line_through K M)
hypothesis h5 : line_intersects Y (line_through C D) (line_through K M)
hypothesis h6 : line_intersects Z (line_through A D) (line_through L N)
hypothesis h7 : line_intersects T (line_through B C) (line_through L N)

-- The target statement
theorem tangential_circumcircle_tangent 
  (Γ1 Γ2 : circle P) (circle_EI : circle P) (circle_FI : circle P) :
    circumscribed_triangle (triangle X F Y) Γ1 →
    circumscribed_triangle (circle_EI) (diameter_circle E I) →
    circumscribed_triangle (triangle T E Z) Γ2 →
    circumscribed_triangle (circle_FI) (diameter_circle F I) →
    tangency (Γ1) (circle_EI) ↔ tangency (Γ2) (circle_FI) :=
sorry

end tangential_circumcircle_tangent_l428_428736


namespace find_lighter_elephant_l428_428010

theorem find_lighter_elephant (m1 m2 m3 m4 m5 m6 m7 m8 : ℕ) :
  (∀ n ≥ 3, m n = m (n - 1) + m (n - 2)) →
  ∃ e : {e // e ∈ {1, 2, 3, 4, 5, 6, 7, 8}},
  lighter_eph_finder (m1, m2, m3, m4, m5, m6, m7, m8) = e
sorry

end find_lighter_elephant_l428_428010


namespace alice_savings_l428_428087

-- Define the constants and conditions
def gadget_sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.1

-- State the theorem to be proved
theorem alice_savings : 
  let commission := gadget_sales * commission_rate in
  let total_earnings := basic_salary + commission in
  let savings := total_earnings * savings_rate in
  savings = 29 :=
by
  sorry

end alice_savings_l428_428087


namespace find_x1_l428_428194

theorem find_x1 (x1 x2 x3 x4 : ℝ) (h1 : 0 ≤ x4) (h2 : x4 ≤ x3) (h3 : x3 ≤ x2) (h4 : x2 ≤ x1) (h5 : x1 ≤ 1)
  (h6 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) : x1 = 4 / 5 := 
  sorry

end find_x1_l428_428194


namespace equal_chords_through_P_exists_l428_428420

variables {K L P : Point} {r R : ℝ} (h_r_lt_R : r < R) (h_P_not_on_K : ¬P ∈ K.circle r) (h_P_not_on_L : ¬P ∈ L.circle R)

theorem equal_chords_through_P_exists (P : Point) (r R : ℝ) (K L : Circle) (h_r_lt_R : r < R) (h_P_not_on_K : ¬P ∈ K) (h_P_not_on_L : ¬P ∈ L) :
  ∃ e : Line, (∀ x y : ℝ, e.through P → e.cuts_chords_of_equal_length K L r R x y) :=
sorry

end equal_chords_through_P_exists_l428_428420


namespace processing_decision_l428_428440

-- Definitions of given conditions
def processing_fee (grade: Char) : ℤ :=
  match grade with
  | 'A' => 90
  | 'B' => 50
  | 'C' => 20
  | 'D' => -50
  | _   => 0

def processing_cost (branch: Char) : ℤ :=
  match branch with
  | 'A' => 25
  | 'B' => 20
  | _   => 0

structure FrequencyDistribution :=
  (gradeA : ℕ)
  (gradeB : ℕ)
  (gradeC : ℕ)
  (gradeD : ℕ)

def branchA_distribution : FrequencyDistribution :=
  { gradeA := 40, gradeB := 20, gradeC := 20, gradeD := 20 }

def branchB_distribution : FrequencyDistribution :=
  { gradeA := 28, gradeB := 17, gradeC := 34, gradeD := 21 }

-- Lean 4 statement for proof of questions
theorem processing_decision : 
  let profit (grade: Char) (branch: Char) := processing_fee grade - processing_cost branch
  let avg_profit (dist: FrequencyDistribution) (branch: Char) : ℤ :=
    (profit 'A' branch) * dist.gradeA / 100 +
    (profit 'B' branch) * dist.gradeB / 100 +
    (profit 'C' branch) * dist.gradeC / 100 +
    (profit 'D' branch) * dist.gradeD / 100
  (pA_branchA : Float := branchA_distribution.gradeA / 100.0) = 0.4 ∧
  (pA_branchB : Float := branchB_distribution.gradeA / 100.0) = 0.28 ∧
  avg_profit branchA_distribution 'A' = 15 ∧
  avg_profit branchB_distribution 'B' = 10 →
  avg_profit branchA_distribution 'A' > avg_profit branchB_distribution 'B'
:= by 
  sorry

end processing_decision_l428_428440


namespace axis_of_symmetry_l428_428352

noncomputable def quadratic_function : ℝ → ℝ := λ x, x^2 - 4 * x + 3

theorem axis_of_symmetry : ∃ h : ℝ, ∀ x : ℝ, quadratic_function x = quadratic_function x ↔ x = h :=
by sorry

end axis_of_symmetry_l428_428352


namespace add_least_number_l428_428797

theorem add_least_number (n : ℕ) (h1 : n = 1789) (h2 : ∃ k : ℕ, 5 * k = n + 11) (h3 : ∃ j : ℕ, 6 * j = n + 11) (h4 : ∃ m : ℕ, 4 * m = n + 11) (h5 : ∃ l : ℕ, 11 * l = n + 11) : 11 = 11 :=
by
  sorry

end add_least_number_l428_428797


namespace four_digit_numbers_with_two_repeating_digits_l428_428915

theorem four_digit_numbers_with_two_repeating_digits :
  let num := 2736 in
  (λ n : ℕ, n = num) 2736 :=
by
  sorry

end four_digit_numbers_with_two_repeating_digits_l428_428915


namespace triangle_sides_determine_l428_428220

theorem triangle_sides_determine (r a λ : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < r) 
  (h3 : 0 < λ) :
  ∃ b c : ℝ, 
    c = 2 * a * r / (real.sqrt (4 * r ^ 2 * (λ ^ 2 + 1) - 4 * r * λ * real.sqrt (4 * r ^ 2 - a ^ 2)))
    ∧ b = 2 * λ * a * r / (real.sqrt (4 * r ^ 2 * (λ ^ 2 + 1) - 4 * r * λ * real.sqrt (4 * r ^ 2 - a ^ 2))) :=
sorry

end triangle_sides_determine_l428_428220


namespace count_three_digit_numbers_l428_428688

theorem count_three_digit_numbers :
  let numbers := {n : ℕ | ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b < a ∧ 0 ≤ c ∧ c < a ∧ n = 100 * a + 10 * b + c} in
  numbers.card = 285 :=
by
  sorry

end count_three_digit_numbers_l428_428688


namespace time_after_seconds_l428_428738

def initial_time : Nat := 8 * 60 * 60 -- 8:00:00 a.m. in seconds
def seconds_passed : Nat := 8035
def target_time : Nat := (10 * 60 * 60 + 13 * 60 + 35) -- 10:13:35 in seconds

theorem time_after_seconds : initial_time + seconds_passed = target_time := by
  -- proof skipped
  sorry

end time_after_seconds_l428_428738


namespace least_possible_b_l428_428258

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem least_possible_b (a b : Nat) (h1 : is_prime a) (h2 : is_prime b) (h3 : a + 2 * b = 180) (h4 : a > b) : b = 19 :=
by 
  sorry

end least_possible_b_l428_428258


namespace max_divisions_circle_and_lines_l428_428055

theorem max_divisions_circle_and_lines (n : ℕ) (h₁ : n = 5) : 
  let R_lines := n * (n + 1) / 2 + 1 -- Maximum regions formed by n lines
  let R_circle_lines := 2 * n       -- Additional regions formed by a circle intersecting n lines
  R_lines + R_circle_lines = 26 := by
  sorry

end max_divisions_circle_and_lines_l428_428055


namespace ellipse_equation_l428_428565

theorem ellipse_equation (F1 F2 : ℝ × ℝ) (d : ℝ) 
  (hF1 : F1 = (0, -4)) (hF2 : F2 = (0, 4)) (hd : d = 2) : 
  ∃ a b c : ℝ, a = 6 ∧ c = 4 ∧ b^2 = a^2 - c^2 ∧ 
  (∀ x y : ℝ, ((y^2) / a^2 + (x^2) / b^2 = 1) ↔ 
   (y^2 / 36 + x^2 / 20 = 1)) := 
begin
  -- Inside this block, we would provide the steps of the solution.
  sorry
end

end ellipse_equation_l428_428565


namespace gcf_lcm_calculation_l428_428400

def prime_factors (n : ℕ) : ℕ → ℕ := sorry

def lcm (a b : ℕ) : ℕ :=
  Nat.div (a * b) (Nat.gcd a b)

def gcd (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem gcf_lcm_calculation :
  gcd (lcm 18 30) (lcm 21 28) = 6 :=
by
  sorry

end gcf_lcm_calculation_l428_428400


namespace sam_needs_change_l428_428818

-- Definitions based on conditions
def cost_structure (toys : Fin 8 → ℚ) : Prop :=
  (∀ i j, i < j → toys i > toys j) ∧ (∀ i, toys i - toys (i + 1) = 0.25) ∧ 
  toys 7 = 1.75

def enough_quarters : ℚ := 8 * 0.25

def favorite_toy (toys : Fin 8 → ℚ) : ℚ := toys 7

def can_buy_directly (toys : Fin 8 → ℚ) : Fin 8 → Prop 
| 0 => favorite_toy toys ≤ enough_quarters
| i => (favorite_toy toys ≤ enough_quarters + i * 0.25 ∧ 
        ∀ j, (j < i → toys j > 0.25) ∨ j = 7)

-- The lean statement
theorem sam_needs_change (toys : Fin 8 → ℚ) (h_cost_structure : cost_structure toys) :
  (∑ i in Finset.range 2, if can_buy_directly toys i then 7! / 8! else 0) = 1/7 → 
  (1 - (1/7) = 6/7) :=
by
  sorry

end sam_needs_change_l428_428818


namespace correct_calculation_l428_428835

theorem correct_calculation : sqrt 12 - sqrt 3 = sqrt 3 :=
by
  sorry

end correct_calculation_l428_428835


namespace odd_function_domain_symmetric_l428_428708

theorem odd_function_domain_symmetric {α : Type*} [LinearOrder α] {f : α → ℝ} {p q : α} 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_domain : ∀ x, x ∈ set.Icc p q) :
  p + q = 0 := 
sorry

end odd_function_domain_symmetric_l428_428708


namespace stuffed_animal_cost_l428_428477

theorem stuffed_animal_cost
  (M S A C : ℝ)
  (h1 : M = 3 * S)
  (h2 : M = (1/2) * A)
  (h3 : C = (1/2) * A)
  (h4 : C = 2 * S)
  (h5 : M = 6) :
  A = 8 :=
by
  sorry

end stuffed_animal_cost_l428_428477


namespace no_bounded_sequences_at_least_one_gt_20_l428_428145

variable (x y z : ℕ → ℝ)
variable (x1 y1 z1 : ℝ)
variable (h0 : x1 > 0) (h1 : y1 > 0) (h2 : z1 > 0)
variable (h3 : ∀ n, x (n + 1) = y n + (1 / z n))
variable (h4 : ∀ n, y (n + 1) = z n + (1 / x n))
variable (h5 : ∀ n, z (n + 1) = x n + (1 / y n))

-- Part (a)
theorem no_bounded_sequences : (∀ n, x n > 0) ∧ (∀ n, y n > 0) ∧ (∀ n, z n > 0) → ¬ (∃ M, ∀ n, x n < M ∧ y n < M ∧ z n < M) :=
sorry

-- Part (b)
theorem at_least_one_gt_20 : x 1 = x1 ∧ y 1 = y1 ∧ z 1 = z1 → x 200 > 20 ∨ y 200 > 20 ∨ z 200 > 20 :=
sorry

end no_bounded_sequences_at_least_one_gt_20_l428_428145


namespace count_rectangles_5x5_l428_428678

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428678


namespace false_proposition_b_l428_428583

variable {A B : Prop} (p : Prop) (q : Prop)

-- Conditions
def condition_1 := (¬A ∨ ¬B) ↔ ¬(A ↔ B) -- Event A opposite to event B implies mutually exclusive, but not necessarily vice versa.
def condition_2 := ∀ (f : ℝ → ℝ), (∀ x, f (-x) = f x) → SymmetricAboutYAxis f  -- Even function symmetry condition

axiom p_false : ¬condition_1
axiom q_true : condition_2

-- The main statement to be proved
theorem false_proposition_b : ¬(p_false ∧ q_true) :=
by { sorry }

end false_proposition_b_l428_428583


namespace inequality_range_of_a_l428_428576

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |2 * x - a| > x - 1) ↔ a < 3 ∨ a > 5 :=
by
  sorry

end inequality_range_of_a_l428_428576


namespace num_rectangles_in_5x5_grid_l428_428603

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428603


namespace point_Y_on_median_BM_l428_428301

variables {A B C M Y : Type} -- Points in geometry
variables (ω1 ω2 : set Type) -- Circles defined as sets of points

-- Definitions for intersection and symmetry conditions
def intersects (ω1 ω2 : set Type) (y : Type) : Prop := y ∈ ω1 ∧ y ∈ ω2

def same_side (A B C : Type) (Y : Type) : Prop := -- geometric definition that Y and B are on the same side of line AC
  sorry

def median (B M : Type) : set Type := -- geometric construction of median BM
  sorry 

def lies_on_median (Y : Type) (B M : Type) : Prop :=
  Y ∈ median B M

theorem point_Y_on_median_BM
  (h1 : intersects ω1 ω2 Y)
  (h2 : same_side A B C Y) :
  lies_on_median Y B M :=
sorry

end point_Y_on_median_BM_l428_428301


namespace rectangle_perimeter_greater_than_16_l428_428170

theorem rectangle_perimeter_greater_than_16 (a b : ℝ) (h : a * b > 2 * a + 2 * b) (ha : a > 0) (hb : b > 0) : 
  2 * (a + b) > 16 :=
sorry

end rectangle_perimeter_greater_than_16_l428_428170


namespace number_of_rectangles_in_grid_l428_428596

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428596


namespace max_number_of_rectangles_in_square_l428_428079

-- Definitions and conditions
def area_square (n : ℕ) : ℕ := 4 * n^2
def area_rectangle (n : ℕ) : ℕ := n + 1
def max_rectangles (n : ℕ) : ℕ := area_square n / area_rectangle n

-- Lean theorem statement for the proof problem
theorem max_number_of_rectangles_in_square (n : ℕ) (h : n ≥ 4) :
  max_rectangles n = 4 * (n - 1) :=
sorry

end max_number_of_rectangles_in_square_l428_428079


namespace number_of_multiples_of_53_l428_428081

def triangular_array := {b : ℕ × ℕ → ℕ // 
  (∀ (n k : ℕ), b (n, k) = 2^(n-1) * (n + 2 * k - 1)) ∧
  (∀ (n : ℕ), b (n, 1) = 2^(n-1) * (n + 1 - 1))
}

theorem number_of_multiples_of_53 : 
  ∃ S : Finset (ℕ × ℕ),
  (∀ x ∈ S, ∃ n k : ℕ, x = (n, k) ∧  (b (n, k) % 53 = 0)) ∧
  S.card = 24 :=
begin
  sorry
end

end number_of_multiples_of_53_l428_428081


namespace conjugate_z_quadrant_l428_428701

noncomputable def z (z : ℂ) := (1 - I) * z = complex.abs (-3 + I)

theorem conjugate_z_quadrant (z : ℂ) (hz : z (z)) : 
  (complex.re (conj z) > 0 ∧ complex.im (conj z) < 0) :=
sorry

end conjugate_z_quadrant_l428_428701


namespace product_gcf_lcm_30_75_l428_428832

noncomputable def gcf (a b : ℕ) : ℕ := Nat.gcd a b
noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem product_gcf_lcm_30_75 : gcf 30 75 * lcm 30 75 = 2250 := by
  sorry

end product_gcf_lcm_30_75_l428_428832


namespace xy_range_l428_428243

theorem xy_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 1/x + y + 1/y = 5) :
  1/4 ≤ x * y ∧ x * y ≤ 4 :=
sorry

end xy_range_l428_428243


namespace area_of_inscribed_triangle_l428_428899

noncomputable def area_inscribed_triangle (r : ℝ) (a b c : ℝ) : ℝ :=
  1 / 2 * r^2 * (Real.sin a + Real.sin b + Real.sin c)

theorem area_of_inscribed_triangle : 
  ∃ (r a b c : ℝ),
    a + b + c = 2 * π ∧
    r = 10 / π ∧
    a = 5 * (18 * π / 180) ∧
    b = 7 * (18 * π / 180) ∧
    c = 8 * (18 * π / 180) ∧
    area_inscribed_triangle r a b c = 119.84 / π^2 :=
begin
  sorry
end

end area_of_inscribed_triangle_l428_428899


namespace sujis_age_l428_428000

theorem sujis_age (x : ℕ) (Abi Suji : ℕ)
  (h1 : Abi = 5 * x)
  (h2 : Suji = 4 * x)
  (h3 : (Abi + 3) / (Suji + 3) = 11 / 9) : 
  Suji = 24 := 
by 
  sorry

end sujis_age_l428_428000


namespace intersection_necessary_but_not_sufficient_l428_428754

variables {M N P : Set α}

theorem intersection_necessary_but_not_sufficient : 
  (M ∩ P = N ∩ P) → (M ≠ N) :=
sorry

end intersection_necessary_but_not_sufficient_l428_428754


namespace num_rectangles_in_5x5_grid_l428_428599

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428599


namespace part1_intersection_when_a_is_zero_part2_range_of_a_l428_428186

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x < a + 5}

-- Part (1): When a = 0, find A ∩ B
theorem part1_intersection_when_a_is_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

-- Part (2): If A ∪ B = A, find the range of values for a
theorem part2_range_of_a (a : ℝ) :
  (A ∪ B a = A) → (0 < a ∧ a ≤ 1) ∨ (6 ≤ a) :=
sorry

end part1_intersection_when_a_is_zero_part2_range_of_a_l428_428186


namespace point_Y_on_median_BM_l428_428303

variables {A B C M Y : Type} -- Points in geometry
variables (ω1 ω2 : set Type) -- Circles defined as sets of points

-- Definitions for intersection and symmetry conditions
def intersects (ω1 ω2 : set Type) (y : Type) : Prop := y ∈ ω1 ∧ y ∈ ω2

def same_side (A B C : Type) (Y : Type) : Prop := -- geometric definition that Y and B are on the same side of line AC
  sorry

def median (B M : Type) : set Type := -- geometric construction of median BM
  sorry 

def lies_on_median (Y : Type) (B M : Type) : Prop :=
  Y ∈ median B M

theorem point_Y_on_median_BM
  (h1 : intersects ω1 ω2 Y)
  (h2 : same_side A B C Y) :
  lies_on_median Y B M :=
sorry

end point_Y_on_median_BM_l428_428303


namespace number_of_mappings_l428_428755

noncomputable theory

-- Define the set A
def A := Finset.range 10

-- Define the mappings f (as a function from A to A)
variable (f : A → A)

-- Define f_k iteratively from f
def f_k (k : ℕ) (x : A) : A :=
  (nat.iterate f k) x

-- Define the conditions
def condition1 : Prop :=
  ∀ x ∈ A, f_k 30 x = x

def condition2 : Prop :=
  ∀ k : ℕ, 1 ≤ k → k ≤ 29 → ∃ a ∈ A, f_k k a ≠ a

-- Lean statement proving the number of such functions f is 120960
theorem number_of_mappings : condition1 f → condition2 f → ∃ n : ℕ, n = 120960 :=
by
  sorry

end number_of_mappings_l428_428755


namespace calc_A_n_calc_F1_calc_F2_calc_F3_calc_F4_F_recursive_F_sum_l428_428828

-- Definitions for the conditions
def A (n : ℕ) : ℕ := 2^n
def F : ℕ → ℕ 
| 0 => 0  -- Not defined for n = 0 as per the problem description
| 1 => 2
| 2 => 3
| 3 => 5
| 4 => 8
| n => F (n-1) + F (n-2)

-- Proofs of the statements:

-- Statement for Part 1
theorem calc_A_n (n : ℕ) : A n = 2^n := sorry

-- Statements for Part 2
theorem calc_F1 : F 1 = 2 := sorry
theorem calc_F2 : F 2 = 3 := sorry
theorem calc_F3 : F 3 = 5 := sorry
theorem calc_F4 : F 4 = 8 := sorry

-- Statement for Part 3
theorem F_recursive (n : ℕ) (hn : n ≥ 3) : F n = F (n-1) + F (n-2) := sorry

-- Statement for Part 4
theorem F_sum (n p : ℕ) (hn : n ≥ 1) (hp : p ≥ 1) : F (n+p+1) = F n * F p + F (n-1) * F (p-1) := sorry

end calc_A_n_calc_F1_calc_F2_calc_F3_calc_F4_F_recursive_F_sum_l428_428828


namespace math_problem_solution_l428_428973

theorem math_problem_solution (x y z : ℚ) (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  let k := 6 in
  (x + k * y - z = 0) ∧
  (4 * x + 2 * k * y + 3 * z = 0) ∧
  (3 * x + 6 * y + 2 * z = 0) →
  (x * z) / (y ^ 2) = 1368 / 25 :=
by
  intros h
  sorry

end math_problem_solution_l428_428973


namespace rectangles_in_grid_l428_428671

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428671


namespace count_rectangles_5x5_l428_428682

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428682


namespace smallest_circle_area_l428_428840

/-- The smallest possible area of a circle passing through two given points in the coordinate plane. -/
theorem smallest_circle_area (P Q : ℝ × ℝ) (hP : P = (-3, -2)) (hQ : Q = (2, 4)) : 
  ∃ (A : ℝ), A = (61 * Real.pi) / 4 :=
by
  sorry

end smallest_circle_area_l428_428840


namespace sin_ordering_l428_428492

theorem sin_ordering : sin 3 < sin 1 < sin 2 :=
by
  have h1 : sin 2 = sin (π - 2),
  { sorry },
  
  have h2 : sin 3 = sin (π - 3),
  { sorry },
  
  have h3 : 0 < π - 3 ∧ π - 3 < 1 ∧ 1 < π - 2 ∧ π - 2 < π / 2,
  { sorry },
  
  have h4 : ∀ x : ℝ, 0 < x ∧ x < π / 2 → strict_mono_on sin (Icc 0 (π / 2)),
  { sorry },
  
  -- use the above to prove sin 3 < sin 1 < sin 2
  sorry

end sin_ordering_l428_428492


namespace find_number_l428_428070

-- Define the condition
def exceeds_by_30 (x : ℝ) : Prop :=
  x = (3/8) * x + 30

-- Prove the main statement
theorem find_number : ∃ x : ℝ, exceeds_by_30 x ∧ x = 48 := by
  sorry

end find_number_l428_428070


namespace students_like_all_three_l428_428251

noncomputable theory

def total_students := 50
def likes_apple := 22
def likes_chocolate := 25
def likes_brownies := 10
def likes_none := 8

def likes_at_least_one : ℕ := total_students - likes_none

def sum_likes := likes_apple + likes_chocolate + likes_brownies

theorem students_like_all_three : 
  ∃ (n : ℕ), 
    likes_at_least_one = sum_likes - (likes_apple + likes_chocolate + likes_brownies) + n := 
by {
  existsi 9,
  unfold likes_at_least_one sum_likes,
  exact calc
    42 = 22 + 25 + 10 - 9 - 7 - 8 + 9 : by { sorry } -- this line simulates the computation steps, to be filled out in actual proof.
}

end students_like_all_three_l428_428251


namespace restruct_possible_l428_428787

variable {City : Type} (A B C D : City)
variable (initial new : City → set City)
variable (roads : set (City × City))

noncomputable def restructuring {City : Type} (initial new : City → set City) : Prop :=
  ∃ (F : set (City × City → set (City × City))), 
    (∀ r ∈ F, ∃ a b c d : City, r = {p | p = (a, b) ∨ p = (c, d)}) ∧
    ∀ p ∈ F, (new = (initial \ p)) unon (set.map (λ (x : City × City), (if x.1 = p.1 ∧ x.2 = p.2 then (p.1, p.3) else if x.1 = p.3 ∧ x.2 = p.4 then (p.4, p.2) else x)) (initial))

def road_problem : Prop :=
  ∀ (initial new : City → set City),
    (∀ c : City, initial c \ new c = initial c) ∧ 
    (∀ c : City, new c = initial c.unon 1) ∧ 
    ∀ a b c d : City, restructuring initial new

theorem restruct_possible :
  road_problem :=
sorry

end restruct_possible_l428_428787


namespace angle_AE_A1ED1_60_degrees_l428_428534

-- Define the properties of the cuboid
structure Cuboid :=
  (A B C D A1 B1 C1 D1 : Type)
  (AB BC : ℝ) (AA1 : ℝ)
  (midpoint : (B, B1) -> E)

constant cuboid : Cuboid
axiom h1 : cuboid.AB = 1
axiom h2 : cuboid.BC = 1
axiom h3 : cuboid.AA1 = 2
axiom h4 : ∀ B B1, cuboid.midpoint (B, B1) = cuboid.E

-- Define the angle between line and plane
def angle_between_line_and_plane : ℝ :=
  60  -- The angle in degrees

-- The proof theorem
theorem angle_AE_A1ED1_60_degrees :
  angle_between_line_and_plane = 60 :=
sorry

end angle_AE_A1ED1_60_degrees_l428_428534


namespace count_rectangles_5x5_l428_428676

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428676


namespace max_tan_angle_POI_l428_428560

-- Define the ellipse and its properties
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Define the foci of the ellipse
def F1 : ℝ × ℝ := (- sqrt 1, 0)  -- since c^2 = a^2 - b^2 and a = 2, b = sqrt(3)
def F2 : ℝ × ℝ := (sqrt 1, 0)

-- Variables representing points in the first quadrant of the ellipse
def P (x y : ℝ) : Prop := ellipse x y ∧ 0 < x ∧ 0 < y

-- Define the incenter I of the triangle PF1F2
def incenter (P F1 F2 : ℝ × ℝ) : ℝ × ℝ := sorry  -- We can define it as needed

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- The maximum value of tan(angle POI)
theorem max_tan_angle_POI :
  ∀ (x y : ℝ), P x y → ∃ (I : ℝ × ℝ), incenter (x, y) F1 F2 = I ∧ 
  tan (angle (0, 0) (x, y) I) ≤ (sqrt 6 / 12) :=
sorry

end max_tan_angle_POI_l428_428560


namespace probability_AC_lt_12_l428_428475

noncomputable def distance (p1 p2 : Point) : ℝ := Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

theorem probability_AC_lt_12 :
  ∀ (A B : Point) (α : ℝ), 
  A = ⟨0, -10⟩ → 
  B = ⟨0, 0⟩ → 
  0 < α ∧ α < π →
  (∃ C : Point, distance B C = 6 ∧ ∃ β : ℝ, β = α ∧ distance B A = 10) → 
  (realMeasure setting) (setOf (λ C, distance A C < 12)) = 1 / 3 := 
by 
  sorry

end probability_AC_lt_12_l428_428475


namespace largest_number_of_stores_visited_l428_428011

-- Define the conditions from the problem statement
variables (P : Type) [fintype P] (S : set P)
variables (total_visits unique_visitors visitors_two_stores stores_count : ℕ)

-- Define the relevant quantities
def conditions :=
  total_visits = 23 ∧
  unique_visitors = 12 ∧
  visitors_two_stores = 8 ∧
  stores_count = 8 

-- Define the specific facts derived from the conditions
def solution (S : Type) [fintype S] [decidable_pred S] :=
  ∀ (p ∈ P), #P ≥ 1 ∧ #P ∈ P → 
  ∃ largest_visits possible, 
    largest_visits = 4  

-- The final statement
theorem largest_number_of_stores_visited :
  conditions → solution := 
by
  intros,
  sorry

end largest_number_of_stores_visited_l428_428011


namespace divide_circle_into_three_equal_parts_l428_428022

noncomputable def angle_between_chords : ℝ :=
  30 + 43 / 60 + 33 / 3600

theorem divide_circle_into_three_equal_parts:
  ∃ (x : ℝ), (x + Math.sin x = Real.pi / 3) ∧ x = angle_between_chords :=
sorry

end divide_circle_into_three_equal_parts_l428_428022


namespace trig_identity_l428_428501

theorem trig_identity : 
  cos (20 * real.pi / 180) * cos (25 * real.pi / 180) - sin (20 * real.pi / 180) * sin (25 * real.pi / 180) = real.sqrt 2 / 2 :=
by
  sorry

end trig_identity_l428_428501


namespace average_value_of_integers_in_T_l428_428288

variable {T : Finset ℤ}
variable [fintype T]
variable {b_1 b_m : ℤ}
variable {m : ℕ}

open Finset

theorem average_value_of_integers_in_T 
  (h1 : ∑ x in (T.erase b_m : Finset ℤ), x = 45 * (T.card - 1))
  (h2 : ∑ x in ((T.erase b_1).erase b_m : Finset ℤ), x = 50 * (T.card - 2))
  (h3 : ∑ x in (insert b_m (T.erase b_1) : Finset ℤ), x = 65 * (T.card - 1))
  (h4 : b_m = b_1 + 110)
  (h5 : b_1 ∈ T)
  (h6 : b_m ∈ T)
  (h7 : ∀ x ∈ T, (0 : ℤ) < x) :
  (T.sum id) / T.card = 47.9 :=
by
  sorry

end average_value_of_integers_in_T_l428_428288


namespace trapezoid_plane_figure_l428_428036

-- Define a Trapezoid
structure Trapezoid (ℝ : Type*) [LinearOrderedField ℝ] :=
(a b c d : ℝ × ℝ) -- Four vertices
(parallel : (a.2 = b.2 ∧ c.2 = d.2) ∨ (a.1 = d.1 ∧ b.1 = c.1)) -- One pair of sides are parallel.

-- Define a Plane
def is_plane_figure {ℝ : Type*} [LinearOrderedField ℝ] (t : Trapezoid ℝ) : Prop :=
∃ plane : Set (ℝ × ℝ), ∀ p ∈ {t.a, t.b, t.c, t.d}, p ∈ plane

-- Statement: A trapezoid is definitely a plane figure.
theorem trapezoid_plane_figure {ℝ : Type*} [LinearOrderedField ℝ] (t : Trapezoid ℝ) :
  is_plane_figure t :=
sorry

end trapezoid_plane_figure_l428_428036


namespace train_pass_time_l428_428465

noncomputable def relative_speed_kmh (train_speed man_speed : ℕ) : ℕ :=
  train_speed + man_speed

noncomputable def relative_speed_ms (speed_kmh : ℕ) : ℚ :=
  (speed_kmh * 1000) / 3600

noncomputable def time_to_pass (distance : ℕ) (speed_ms : ℚ) : ℚ :=
  distance / speed_ms

theorem train_pass_time :
  let train_length := 250
  let train_speed := 80
  let man_speed := 12
  let rel_speed_kmh := relative_speed_kmh train_speed man_speed
  let rel_speed_ms := relative_speed_ms rel_speed_kmh
  let pass_time := time_to_pass train_length rel_speed_ms
  pass_time ≈ 9.78
:=
by
  let train_length := 250
  let train_speed := 80
  let man_speed := 12
  let rel_speed_kmh := relative_speed_kmh train_speed man_speed
  let rel_speed_ms := relative_speed_ms rel_speed_kmh
  let pass_time := time_to_pass train_length rel_speed_ms
  have h: pass_time ≈ 9.78 := sorry
  exact h

end train_pass_time_l428_428465


namespace intersection_a_zero_range_of_a_l428_428188

variable (x a : ℝ)

def setA : Set ℝ := { x | - 1 < x ∧ x < 6 }
def setB (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x < a + 5 }

theorem intersection_a_zero :
  setA x ∧ setB 0 x ↔ - 1 < x ∧ x < 5 := by
  sorry

theorem range_of_a (h : ∀ x, setA x ∨ setB a x → setA x) :
  (0 < a ∧ a ≤ 1) ∨ 6 ≤ a :=
  sorry

end intersection_a_zero_range_of_a_l428_428188


namespace correct_option_is_B_l428_428408

-- Definitions based on given conditions
def empty_set_is_empty : ∀ x, x ∉ (∅ : Set α) := by
  intro x
  intro hx
  cases hx

def empty_set_true_subset_nonempty (s : Set α) : s ≠ ∅ → ∅ ⊂ s :=
  λ h, by simp [Set.ssubset_def]; exact ⟨λ x hx, by cases hx, λ h0, h rfl⟩

-- Theorem statement based on problem and solution
theorem correct_option_is_B : ∅ ⊂ ({0} : Set ℕ) :=
  empty_set_true_subset_nonempty {0} (by simp)

end correct_option_is_B_l428_428408


namespace max_distance_P_to_D_l428_428112

open Real

theorem max_distance_P_to_D : 
  ∀ (P : ℝ × ℝ) (u v w : ℝ), 
    let A := (0, 0 : ℝ)
    let B := (2, 0 : ℝ)
    let C := (3, 1 : ℝ)
    let D := (1, 1 : ℝ)
    u = dist P A ∧ v = dist P B ∧ w = dist P C ∧
    u^2 + w^2 = 2 * v^2
    → dist P D ≤ 1 / sqrt 2 := by 
  sorry

end max_distance_P_to_D_l428_428112


namespace fiona_reaches_14_without_predators_l428_428378

def LilyPads := {n : ℕ // n ≤ 15}

def predators : LilyPads → Prop
| ⟨4, _⟩ := true
| ⟨9, _⟩ := true
| _ := false

def food : LilyPads → Prop
| ⟨14, _⟩ := true
| _ := false

def start_pad : LilyPads := ⟨0, by norm_num⟩

def hop_probability (from to : LilyPads) : ℚ :=
  if from.val + 1 = to.val then 1/2 else 0

def jump_probability (from to : LilyPads) : ℚ :=
  if from.val + 2 = to.val then 1/2 else 0

def return_probability (from to : LilyPads) : ℚ :=
  if from.val = 1 ∧ to.val = 0 then 1/4 else 0

def next_pad_probability (from to : LilyPads) : ℚ :=
  hop_probability from to + jump_probability from to + return_probability from to

theorem fiona_reaches_14_without_predators :
    let target_pad : LilyPads := ⟨14, by norm_num⟩ in
    let intermediate_pads : set LilyPads := {n : LilyPads | ¬ (predators n)} in
    let paths : list (LilyPads) := [start_pad, ⟨2, by norm_num⟩, ⟨3, by norm_num⟩, ⟨5, by norm_num⟩, ⟨6, by norm_num⟩, ⟨8, by norm_num⟩, ⟨10, by norm_num⟩, target_pad] in
    let valid_paths := all_paths intermediate_pads target_pad in
    (calculate_probability valid_paths = 3 / 128) := 
begin
  sorry
end

end fiona_reaches_14_without_predators_l428_428378


namespace isosceles_triangle_length_eq_l428_428727

variables {Point : Type} [EuclideanGeometry Point]

/--
In isosceles triangle ABC (AB = BC), CD is the angle bisector of ∠ACB.
O is the center of the circumcircle of triangle ABC.
E is the intersection of the perpendicular from O to CD with BC.
F is the intersection of the line parallel to CD through E with AB.
Show that BE = FD.
-/
theorem isosceles_triangle_length_eq
  {A B C D E F O : Point}
  (h_iso : AB = BC)
  (h_angle_bis : is_angle_bisector C D B)
  (h_center : is_circumcenter O A B C)
  (h_perp : is_perpendicular_from O E D)
  (h_intersect_E : E = intersect (perpendicular_from O D) BC)
  (h_parallel_F : F = intersect (parallel_through E D) AB):
  distance B E = distance F D := 
sorry

end isosceles_triangle_length_eq_l428_428727


namespace point_Y_lies_on_median_l428_428309

-- Define the geometric points and circles
variable (A B C M Y : Point)
variable (ω1 ω2 : Circle)

-- Definitions of the given conditions
variable (P : Point) (hP : P ∈ (ω1)) (hInt : ω1 ∩ ω2 = {Y})

-- Express conditions in terms of Lean definitions
variable (hSameSide : same_side Y B (line_through A C))
variable (hMedian : M = (midpoint A C))
variable (hBM : is_median B M)

-- The theorem that we need to prove
theorem point_Y_lies_on_median :
  Y ∈ line_through B M :=
sorry

end point_Y_lies_on_median_l428_428309


namespace alice_savings_l428_428089

noncomputable def commission (sales : ℝ) : ℝ := 0.02 * sales
noncomputable def totalEarnings (basic_salary commission : ℝ) : ℝ := basic_salary + commission
noncomputable def savings (total_earnings : ℝ) : ℝ := 0.10 * total_earnings

theorem alice_savings (sales basic_salary : ℝ) (commission_rate savings_rate : ℝ) :
  commission_rate = 0.02 →
  savings_rate = 0.10 →
  sales = 2500 →
  basic_salary = 240 →
  savings (totalEarnings basic_salary (commission_rate * sales)) = 29 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_savings_l428_428089


namespace area_of_inscribed_triangle_l428_428893

noncomputable def calculate_triangle_area_inscribed_in_circle 
  (arc1 : ℝ) (arc2 : ℝ) (arc3 : ℝ) (total_circumference := arc1 + arc2 + arc3)
  (radius := total_circumference / (2 * Real.pi))
  (theta := (2 * Real.pi) / total_circumference)
  (angle1 := 5 * theta) (angle2 := 7 * theta) (angle3 := 8 * theta) : ℝ :=
  0.5 * (radius ^ 2) * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem area_of_inscribed_triangle : 
  calculate_triangle_area_inscribed_in_circle 5 7 8 = 119.85 / (Real.pi ^ 2) :=
by
  sorry

end area_of_inscribed_triangle_l428_428893


namespace angle_a_c_pi_over_6_l428_428224

noncomputable def magnitude {V : Type*} [inner_product_space ℝ V] (v : V) : ℝ :=
real.sqrt (inner_product.inner v v)

noncomputable def angle_between {V : Type*} [inner_product_space ℝ V] (u v : V) : ℝ :=
real.arccos (inner_product.inner u v / (magnitude u * magnitude v))

variables
  {V : Type*} [inner_product_space ℝ V]
  (a b c : V)
  (h1 : magnitude a = magnitude b)
  (h2 : magnitude b = magnitude c)
  (h3 : magnitude c ≠ 0)
  (h4 : a + b = (real.sqrt 3) • c)
  (m : ℝ := magnitude a)

theorem angle_a_c_pi_over_6 :
  angle_between a c = real.pi / 6 :=
sorry

end angle_a_c_pi_over_6_l428_428224


namespace represent_259BC_as_neg259_l428_428765

def year_AD (n: ℤ) : ℤ := n

def year_BC (n: ℕ) : ℤ := -(n : ℤ)

theorem represent_259BC_as_neg259 : year_BC 259 = -259 := 
by 
  rw [year_BC]
  norm_num

end represent_259BC_as_neg259_l428_428765


namespace complex_subtraction_l428_428161

theorem complex_subtraction (x y : ℝ) (h : (x : ℂ) * complex.I + 2 = y - complex.I) : x - y = -3 :=
by
  sorry

end complex_subtraction_l428_428161


namespace alternating_squares_sum_l428_428929

theorem alternating_squares_sum :
  (∑ k in finset.range 2008, (-1)^k * (k + 1)^2) = -2017036 :=
sorry

end alternating_squares_sum_l428_428929


namespace area_S_l428_428315

noncomputable def omega : ℂ := -1/2 + (1/2) * complex.I * real.sqrt 3

def S' := {z : ℂ | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧
                             0 ≤ b ∧ b ≤ 1 ∧
                             0 ≤ c ∧ c ≤ 2 ∧ 
                             z = a + b * omega + c * omega^2}

theorem area_S'_is_6sqrt3 : complex.area S' = 6 * real.sqrt 3 := 
sorry

end area_S_l428_428315


namespace complex_operation_correct_l428_428960

noncomputable def complex_operation_result : ℂ :=
  let z1 := 3 + 5 * complex.I
  let z2 := 4 - 7 * complex.I
  let result1 := z1 * (2 * complex.I)
  let result2 := z2 * (2 * complex.I)
  result1 + result2

theorem complex_operation_correct : complex_operation_result = 4 + 14 * complex.I :=
by
  -- The details of the proof would go here.
  sorry

end complex_operation_correct_l428_428960


namespace dot_product_range_l428_428167

def point (α : Type) := (α × α)

def vector (α : Type) := (α × α)

def dot_product (a b : vector ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def P : point ℝ := (3, 4)
noncomputable def O : point ℝ := (0, 0)

noncomputable def A (θ : ℝ) : point ℝ := (2 + 2 * Real.cos θ, 2 * Real.sin θ)
noncomputable def B (θ : ℝ) : point ℝ := (2 + 2 * Real.cos (θ + Real.arccos (1/2)), 2 * Real.sin (θ + Real.arccos (1/2)))

def AB_length : ℝ := 2 * Real.sqrt 3

theorem dot_product_range :
  let OP : vector ℝ := (3, 4)
      OA (θ : ℝ) : vector ℝ := (A θ).val - O.val
      OB (θ : ℝ) : vector ℝ := (B θ).val - O.val in
  2 ≤ dot_product OP (OA 0 + OB 0) ∧ dot_product OP (OA 0 + OB 0) ≤ 22 := 
by
  sorry

end dot_product_range_l428_428167


namespace det_nonzero_of_diagonally_dominant_l428_428120

theorem det_nonzero_of_diagonally_dominant
  (n : ℕ) 
  (a : Matrix (Fin n) (Fin n) ℝ) 
  (h : ∀ i : Fin n, 2 * |a i i| > ∑ j, |a i j|) :
  Matrix.det a ≠ 0 := 
sorry

end det_nonzero_of_diagonally_dominant_l428_428120


namespace triangle_area_inscribed_in_circle_l428_428907

noncomputable def circle_inscribed_triangle_area : ℝ :=
  let r := 10 / Real.pi
  let angle_A := Real.pi / 2
  let angle_B := 7 * Real.pi / 10
  let angle_C := 4 * Real.pi / 5
  let sin_sum := Real.sin(angle_A) + Real.sin(angle_B) + Real.sin(angle_C)
  1 / 2 * r^2 * sin_sum

theorem triangle_area_inscribed_in_circle (h_circumference : 5 + 7 + 8 = 20)
  (h_radius : 10 / Real.pi * 2 * Real.pi = 20) :
  circle_inscribed_triangle_area = 138.005 / Real.pi^2 :=
by
  sorry

end triangle_area_inscribed_in_circle_l428_428907


namespace value_of_c_l428_428361

theorem value_of_c (c : ℝ) :
  (∀ x y : ℝ, (x, y) = ((2 + 8) / 2, (6 + 10) / 2) → x + y = c) → c = 13 :=
by
  -- Placeholder for proof
  sorry

end value_of_c_l428_428361


namespace number_of_mathematics_books_l428_428338

-- Define a function that converts a three-letter code to an integer index.
def code_to_index (code : string) : ℕ :=
  let letters := "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  in letters.indexOf code[0] * 26^2 + letters.indexOf code[1] * 26 + letters.indexOf code[2]

-- Define the range of valid codes for mathematics books.
def from_code := "HZP"
def to_code := "LAC"

-- Calculate the total number of mathematics books.
def total_books : ℕ :=
  (code_to_index to_code - code_to_index from_code + 1 : ℕ)

theorem number_of_mathematics_books :
  total_books = 2042 :=
by
  -- Conversion of codes to their respective indices.
  let from_index := code_to_index from_code
  let to_index := code_to_index to_code

  -- Calculate number of books based on ranged intervals.
  let HZP_to_HZZ_books := 26 - (code_to_index "HZP" % 26)
  let IAA_to_KZZ_books := 3 * 26 * 26
  let LAA_to_LAC_books := (code_to_index "LAC" % 26) + 1

  -- Sum of all books in all intervals.
  let total := HZP_to_HZZ_books + IAA_to_KZZ_books + LAA_to_LAC_books

  -- Assert the total matches the expected number of books.
  exact calc
    total_books = HZP_to_HZZ_books + IAA_to_KZZ_books + LAA_to_LAC_books : rfl
              ... = 11 + 2028 + 3 : by rfl
              ... = 2042 : by rfl

end number_of_mathematics_books_l428_428338


namespace matrix_power_difference_l428_428314

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 4;
     0, 1]

theorem matrix_power_difference :
  B^30 - 3 * B^29 = !![-2, 0;
                       0,  2] := 
by
  sorry

end matrix_power_difference_l428_428314


namespace number_of_triangles_l428_428226

theorem number_of_triangles (a b c : ℕ) (h1 : a ≤ b) (h2 : b < c) (h3 : b = 5) :
  (∃ t : finset (ℕ × ℕ × ℕ), t.card = 10 ∧
  ∀ (t ∈ t), ∃ (a b c : ℕ), t = (a, b, c) ∧ a ≤ b ∧ b < c ∧ b = 5) :=
sorry

end number_of_triangles_l428_428226


namespace books_from_first_shop_l428_428777

theorem books_from_first_shop (x : ℕ) (h : (2080 : ℚ) / (x + 50) = 18.08695652173913) : x = 65 :=
by
  -- proof steps
  sorry

end books_from_first_shop_l428_428777


namespace num_rectangles_in_5x5_grid_l428_428602

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428602


namespace rectangle_count_5x5_l428_428651

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428651


namespace Y_on_median_BM_l428_428297

-- Let circle ω1 and ω2 be defined. Point Y is an intersection of ω1 and ω2
variable (ω1 ω2 : Set (ℝ × ℝ))
variable (Y B A C M : ℝ × ℝ)

-- Assume that point Y and point B lie on the same side of line AC
variable (same_side : (Y.1 - A.1) * (C.2 - A.2) = (Y.2 - A.2) * (C.1 - A.1)
  ∧ (B.1 - A.1) * (C.2 - A.2) = (B.2 - A.2) * (C.1 - A.1))

-- Intersection of circles ω1 and ω2 at point Y
variable (intersect_Y : ω1 ∩ ω2 = {Y})

-- Definition of the median BM from point B through the midpoint M of AC
variable (BM : Set (ℝ × ℝ))
variable (midpoint_M : M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
variable (median_BM : BM = {p | ∃ t : ℝ, p = (B.1 + t * (midpoint_M.1 - B.1), B.2 + t * (midpoint_M.2 - B.2))})

-- The statement to prove is that point Y lies on the median BM
theorem Y_on_median_BM : Y ∈ BM :=
  sorry

end Y_on_median_BM_l428_428297


namespace number_of_rectangles_in_grid_l428_428590

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428590


namespace probability_at_least_one_l428_428386

variable (Ω : Type)
variable [ProbabilitySpace Ω]

-- Independent events representing the successful decryption by A, B, and C
variable (A B C : Event Ω)
variable (PA : ℝ) (PB : ℝ) (PC : ℝ)

-- Given conditions
axiom independent_events : IndependentEvents [A, B, C]
axiom prob_A : ℙ[A] = 1 / 2
axiom prob_B : ℙ[B] = 1 / 3
axiom prob_C : ℙ[C] = 1 / 4

-- Define the event that at least one person decrypts the code
def at_least_one_decrypts : Event Ω := A ∪ B ∪ C

-- Statement to prove
theorem probability_at_least_one : ℙ[at_least_one_decrypts Ω A B C] = 3 / 4 := by
  sorry

end probability_at_least_one_l428_428386


namespace mass_relation_l428_428094

-- Define the conditions as Lean definitions
def rate_of_flow : ℝ := 4 -- kilograms per minute
def max_capacity : ℝ := 10 -- kilograms
def observation_period : ℝ := 5 -- minutes

-- Define piecewise function m(t)
def m (t : ℝ) : ℝ :=
  if 0 ≤ t ∧ t < max_capacity / rate_of_flow then
    rate_of_flow * t
  else if max_capacity / rate_of_flow ≤ t ∧ t ≤ observation_period then
    max_capacity
  else
    0 -- for completeness

-- The theorem to prove the relationship
theorem mass_relation (t : ℝ) (ht : 0 ≤ t ∧ t ≤ observation_period) : 
∃ (m : ℝ), m = 
  if 0 ≤ t ∧ t < 2.5 then 4 * t
  else if 2.5 ≤ t ∧ t ≤ 5 then 10
  else 0 :=
sorry

end mass_relation_l428_428094


namespace ab_equals_three_l428_428704

theorem ab_equals_three (a b : ℝ) (h : sqrt (a - 3) + abs (1 - b) = 0) : a * b = 3 :=
by
  -- The proof part is omitted and replaced with sorry
  sorry

end ab_equals_three_l428_428704


namespace find_sin_2α_find_tan_expr_l428_428551

/-
Define the conditions for α, the given equation, and prove the expected values.
-/
variables {α : ℝ}

def condition1 := (cos (π / 6 + α) * cos (π / 3 - α) = -1 / 4)
def condition2 := (π / 3 < α) ∧ (α < π / 2)

theorem find_sin_2α (h1 : condition1) (h2 : condition2) : sin (2 * α) = sqrt 3 / 2 :=
by
  sorry

theorem find_tan_expr (h1 : condition1) (h2 : condition2) : tan α - 1 / tan α = 2 * sqrt 3 / 3 :=
by
  sorry

end find_sin_2α_find_tan_expr_l428_428551


namespace centroid_distance_relation_l428_428747

variable {A B C G : Point}
variable (GA GB GC AB BC CA : ℝ)

-- Define distances
def d1 := GA + GB + GC
def d2 := AB + BC + CA

-- Given that G is the centroid, prove the relationship
theorem centroid_distance_relation {G : Point} 
    (h1 : G = centroid A B C) :
    d1 = (1 / 3) * d2 :=
sorry

end centroid_distance_relation_l428_428747


namespace virus_diameter_scientific_notation_l428_428801

theorem virus_diameter_scientific_notation :
  (0.0000001 : ℝ) = 10 ^ (-7 : ℕ) :=
by
sorry

end virus_diameter_scientific_notation_l428_428801


namespace fred_balloons_count_l428_428976

-- Conditions from the problem
def Sam_balloons : ℕ := 46
def destroyed_balloons : ℕ := 16
def total_balloons : ℕ := 40

-- Proof that Fred's balloons count is 10
theorem fred_balloons_count : 
  ∃ (F : ℕ), F + Sam_balloons - destroyed_balloons = total_balloons ∧ F = 10 :=
by
  use 10
  split
  · simp [Sam_balloons, destroyed_balloons, total_balloons]
    sorry


end fred_balloons_count_l428_428976


namespace find_multiplier_value_l428_428880

def number : ℤ := 18
def increase : ℤ := 198

theorem find_multiplier_value (x : ℤ) (h : number * x = number + increase) : x = 12 :=
by
  sorry

end find_multiplier_value_l428_428880


namespace exponential_inequality_l428_428318

theorem exponential_inequality (a b : ℝ) (h : a > b) : 2^a > 2^b :=
sorry

end exponential_inequality_l428_428318


namespace line_intersection_x_value_l428_428136

theorem line_intersection_x_value :
  let line1 (x : ℝ) := 3 * x + 14
  let line2 (x : ℝ) (y : ℝ) := 5 * x - 2 * y = 40
  ∃ x : ℝ, ∃ y : ℝ, (line1 x = y) ∧ (line2 x y) ∧ (x = -68) :=
by
  sorry

end line_intersection_x_value_l428_428136


namespace determine_relationship_l428_428546

noncomputable def relationship_between_m_and_n (m n b : ℝ) : Prop :=
  m < n ↔ (∃ (m n b : ℝ), ∃ k : ℝ, k = 3 ∧ 
  (-1/2, m) ∈ (λ x, k * x + b) ∧ 
  (2, n) ∈ (λ x, k * x + b))

theorem determine_relationship (m n b : ℝ) : relationship_between_m_and_n m n b :=
by 
  sorry

end determine_relationship_l428_428546


namespace num_rectangles_in_5x5_grid_l428_428636

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428636


namespace num_rectangles_in_5x5_grid_l428_428608

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428608


namespace processing_decision_l428_428441

-- Definitions of given conditions
def processing_fee (grade: Char) : ℤ :=
  match grade with
  | 'A' => 90
  | 'B' => 50
  | 'C' => 20
  | 'D' => -50
  | _   => 0

def processing_cost (branch: Char) : ℤ :=
  match branch with
  | 'A' => 25
  | 'B' => 20
  | _   => 0

structure FrequencyDistribution :=
  (gradeA : ℕ)
  (gradeB : ℕ)
  (gradeC : ℕ)
  (gradeD : ℕ)

def branchA_distribution : FrequencyDistribution :=
  { gradeA := 40, gradeB := 20, gradeC := 20, gradeD := 20 }

def branchB_distribution : FrequencyDistribution :=
  { gradeA := 28, gradeB := 17, gradeC := 34, gradeD := 21 }

-- Lean 4 statement for proof of questions
theorem processing_decision : 
  let profit (grade: Char) (branch: Char) := processing_fee grade - processing_cost branch
  let avg_profit (dist: FrequencyDistribution) (branch: Char) : ℤ :=
    (profit 'A' branch) * dist.gradeA / 100 +
    (profit 'B' branch) * dist.gradeB / 100 +
    (profit 'C' branch) * dist.gradeC / 100 +
    (profit 'D' branch) * dist.gradeD / 100
  (pA_branchA : Float := branchA_distribution.gradeA / 100.0) = 0.4 ∧
  (pA_branchB : Float := branchB_distribution.gradeA / 100.0) = 0.28 ∧
  avg_profit branchA_distribution 'A' = 15 ∧
  avg_profit branchB_distribution 'B' = 10 →
  avg_profit branchA_distribution 'A' > avg_profit branchB_distribution 'B'
:= by 
  sorry

end processing_decision_l428_428441


namespace num_rectangles_in_5x5_grid_l428_428631

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428631


namespace cubes_and_quartics_sum_l428_428236

theorem cubes_and_quartics_sum (a b : ℝ) (h1 : a + b = 2) (h2 : a^2 + b^2 = 2) : 
  a^3 + b^3 = 2 ∧ a^4 + b^4 = 2 :=
by 
  sorry

end cubes_and_quartics_sum_l428_428236


namespace num_rectangles_in_5x5_grid_l428_428635

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428635


namespace right_triangle_incenter_circumcenter_distance_l428_428775

theorem right_triangle_incenter_circumcenter_distance (A B C : Point)
  (h_right_angle : is_right_angle ∠ A C B)
  (O1 : Point) (h1 : is_incenter O1 A B C)
  (r : ℝ) (h_r : inradius O1 r)
  (O2 : Point) (h2 : is_circumcenter O2 A B C)
  (R : ℝ) (h_R : circumradius O2 R) :
  dist O1 O2 ≥ R * (Real.sqrt 2 - 1) :=
by
  sorry

end right_triangle_incenter_circumcenter_distance_l428_428775


namespace exists_special_triangle_l428_428494

noncomputable def distance (p1 p2 : Point) : ℝ := sorry

structure Triangle := (a b c : Point)

noncomputable def side_lengths (t : Triangle) : set ℝ := 
{distance t.a t.b, distance t.b t.c, distance t.c t.a}

def smallest_side (t : Triangle) : ℝ :=
side_lengths t.min

def largest_side (t : Triangle) : ℝ :=
side_lengths t.max

def six_distinct_points (s : set Point) : Prop := 
s.card = 6 ∧ ∀ p1 p2 ∈ s, p1 ≠ p2 → distance p1 p2 ≠ distance p2 p1

def no_three_collinear (s : set Point) : Prop := sorry

theorem exists_special_triangle
(s : set Point)
(h1 : six_distinct_points s)
(h2 : no_three_collinear s) : 
∃ t₁ t₂ : Triangle, t₁ ∈ s.triangles ∧ t₂ ∈ s.triangles ∧ 
smallest_side t₁ = largest_side t₂ := sorry

end exists_special_triangle_l428_428494


namespace triangle_XA_XB_XC_sum_l428_428018

theorem triangle_XA_XB_XC_sum :
  ∀ (A B C : Type) [EuclideanGeometry A B C] (AB BC AC : ℝ),
  AB = 15 → BC = 13 → AC = 14 →
  ∃ D E F G X : A B C, 
  (perpendicular_from A D BC) →
  (perpendicular_from B E CA) →
  (perpendicular_from C F AB) →
  (centroid G A B C) →
  (circumcircle_intersection D G X BEF) →
  XA + XB + XC = (4095 * real.sqrt 2) / 336 :=
by sorry

end triangle_XA_XB_XC_sum_l428_428018


namespace min_tangent_length_l428_428891

-- Define the equation of the line y = x
def line_eq (x y : ℝ) : Prop := y = x

-- Define the equation of the circle centered at (4, -2) with radius 1
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y + 2)^2 = 1

-- Prove that the minimum length of the tangent line is √17
theorem min_tangent_length : 
  ∀ (x y : ℝ), line_eq x y → circle_eq x y → 
  (∃ p : ℝ, p = (√17)) :=
by 
  sorry

end min_tangent_length_l428_428891


namespace cos_A_cos_B_identity_area_of_triangle_l428_428268

structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (h : A + B + C = π)

theorem cos_A_cos_B_identity (a b c : ℝ) (A B C : ℝ) (h : A + B + C = π)
  (h1 : C = π / 4)
  (h2 : cos A / cos B = (sqrt 5 * c - a) / b) :
  cos A = sqrt 10 / 10 :=
by sorry

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h : A + B + C = π)
  (h1 : C = π / 4)
  (h2 : cos A / cos B = (sqrt 5 * c - a) / b)
  (h3 : b = sqrt 5) :
  (1/2) * b * c * sin A = 15 / 8 :=
by sorry

end cos_A_cos_B_identity_area_of_triangle_l428_428268


namespace moles_HCl_needed_l428_428227

theorem moles_HCl_needed (c1 : ℕ) (c2 : ℕ) (HCl_needed : ℕ)
  (reaction : HCl_needed = c1) :
  (2 : ℕ) = HCl_needed :=
by
  have equation := λ (H2O CO2 NaCl : ℕ), c1 * H2O = c2 * CO2 ∧ c1 * NaCl = 2,
  -- Given the balanced chemical reaction:
  -- HCl + NaHCO₃ → H₂O + CO₂ + NaCl
  -- and the condition: c1 = 1
  have balanced_equation : c1 = 1 := eq.refl 1,
  -- With 2 moles of NaHCO₃:
  have input_moles_NaHCO3 : c2 = 2 := eq.refl 2,
  -- We need to check the number of moles of HCl (HCl_needed) for the reaction:
  have result : (2 : ℕ) = HCl_needed := by rw [balanced_equation, HCl_needed],
  exact result

end moles_HCl_needed_l428_428227


namespace rectangle_count_5x5_l428_428646

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428646


namespace fill_pool_time_l428_428092

theorem fill_pool_time (pool_volume : ℕ := 32000) 
                       (num_hoses : ℕ := 5) 
                       (flow_rate_per_hose : ℕ := 4) 
                       (operation_minutes : ℕ := 45) 
                       (maintenance_minutes : ℕ := 15) 
                       : ℕ :=
by
  -- Calculation steps will go here in the actual proof
  sorry

example : fill_pool_time = 47 := by
  -- Proof of the theorem fill_pool_time here
  sorry

end fill_pool_time_l428_428092


namespace ellipse_eq_hyperbola_eq_l428_428856

-- Ellipse conditions and proof
def ellipse_standard_eq (x y : ℝ) : Prop := 
  ∃ (a b : ℝ), a = 2 ∧ specific_conditions a b x y ∧
  (b * b = a * a - 1) ∧
  (a > b) ∧ 
  (a > 0) ∧ 
  (b > 0) ∧
  (x^2 / 4 + y^2 / 3 = 1)

def specific_conditions (a b x y : ℝ) : Prop :=
  (a = 2) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_eq : ∀ x y : ℝ, ellipse_standard_eq x y

-- Hyperbola conditions and proof
def hyperbola_standard_eq (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), (b / a = 3 / 4) ∧
             (a^2 / 16 / 5 = a^2 / 16 / 5) ∧
             (x^2 / 16 - y^2 / 9 = 1) ∧
             (c^2 = a^2 + b^2) ∧ (a = 4 ∧ b = 3)

theorem hyperbola_eq : ∀ x y : ℝ, hyperbola_standard_eq x y


end ellipse_eq_hyperbola_eq_l428_428856


namespace sequence_formula_l428_428987

noncomputable def a : ℕ → ℤ
| 0     := 0
| (n+1) := a n + 3

theorem sequence_formula (n : ℕ) : a n = 3 * n + 1 :=
by
  induction n with d hd
  . simp [a]
  . simp [a, hd, add_assoc]
  sorry

end sequence_formula_l428_428987


namespace g_g_neg3_equals_71449800_div_3051421_l428_428196

noncomputable def g (x : ℚ) : ℚ := x⁻² + x⁻² / (1 + x⁻²)

theorem g_g_neg3_equals_71449800_div_3051421 : g (g (-3)) = 71449800 / 3051421 := by
  sorry

end g_g_neg3_equals_71449800_div_3051421_l428_428196


namespace delivery_driver_stops_l428_428059

theorem delivery_driver_stops (total_boxes : ℕ) (boxes_per_stop : ℕ) (stops : ℕ) :
  total_boxes = 27 → boxes_per_stop = 9 → stops = total_boxes / boxes_per_stop → stops = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end delivery_driver_stops_l428_428059


namespace ap_geq_ai_l428_428737

theorem ap_geq_ai (ABC : Triangle) (I : Point) (P : Point)
  (h1 : Incenter I ABC)
  (h2 : InsideTriangle P ABC)
  (h3 : ∠PBA + ∠PCA = ∠PBC + ∠PCB) :
  AP P A ≥ AP I A ∧ (AP P A = AP I A ↔ P = I) :=
by sorry

end ap_geq_ai_l428_428737


namespace f_neg9_eq_neg2_l428_428555

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then
    2 * x
  else if 0 ≤ -x ∧ -x ≤ 2 then
    -2 * x
  else
    sorry  -- f is otherwise undefined in this statement

-- Assume f(x) is periodic with period 8
lemma periodic_f (x : ℝ) : f (x + 8) = f x := sorry

-- Prove that f(-9) = -2
theorem f_neg9_eq_neg2 : f (-9) = -2 := by
  have h1 : f (-9) = f (-9 + 8) := periodic_f (-9)
  rw [h1]
  have h2 : f (-1) = -f 1 := sorry  -- Using the odd function property
  rw [h2]
  have h3 : f 1 = 2 := sorry  -- Given that f(x) = 2x for x ∈ [0,2]
  rw [h3]
  norm_num

end f_neg9_eq_neg2_l428_428555


namespace will_earnings_correct_l428_428838

-- Definitions of hourly wages and hours worked for each day
def wage_mon := 8
def hours_mon := 8

def wage_tue := 10
def hours_tue := 2

def wage_wed := 9
def hours_wed := 6

def wage_thu := 7
def hours_thu := 4

def wage_fri := 7
def hours_fri := 4

-- Definition of the tax rate
def tax_rate := 0.12

-- Calculate the total earnings before tax
def earnings_before_tax :=
  wage_mon * hours_mon + wage_tue * hours_tue + wage_wed * hours_wed + wage_thu * hours_thu + wage_fri * hours_fri

-- Calculate the tax deduction
def tax_deduction := tax_rate * earnings_before_tax

-- Calculate the total earnings after tax
def earnings_after_tax := earnings_before_tax - tax_deduction

-- Theorem statement
theorem will_earnings_correct :
  earnings_after_tax = 170.72 :=
by
  sorry

end will_earnings_correct_l428_428838


namespace solve_quadratic_eqn_l428_428955

theorem solve_quadratic_eqn:
  (∃ x: ℝ, (x + 10)^2 = (4 * x + 6) * (x + 8)) ↔ 
  (∀ x: ℝ, x = 2.131 ∨ x = -8.131) := 
by
  sorry

end solve_quadratic_eqn_l428_428955


namespace even_and_monotonic_increasing_f_A_not_monotonic_increasing_f_B_not_even_f_C_even_and_monotonic_increasing_f_D_correct_answers_l428_428406

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x ≤ f y
def is_monotonic_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x ≥ f y

-- The four given functions
def f_A := abs ∘ (λ x : ℝ, 2 * x)
def f_B := λ x : ℝ, 1 - x^2
def f_C := λ x : ℝ, -1 / x
def f_D := λ x : ℝ, 2 * x^2 + 3

theorem even_and_monotonic_increasing_f_A : is_even f_A ∧ is_monotonic_increasing f_A 0 1 := sorry

theorem not_monotonic_increasing_f_B : ¬ is_monotonic_increasing f_B 0 1 := sorry

theorem not_even_f_C : ¬ is_even f_C := sorry

theorem even_and_monotonic_increasing_f_D : is_even f_D ∧ is_monotonic_increasing f_D 0 1 := sorry

-- Final proof that the answers are exact functions f_A and f_D
theorem correct_answers : (is_even f_A ∧ is_monotonic_increasing f_A 0 1) ∧ 
                         (is_even f_D ∧ is_monotonic_increasing f_D 0 1) ∧ 
                         (¬ is_monotonic_increasing f_B 0 1) ∧ 
                         (¬ is_even f_C) := 
  by exact ⟨even_and_monotonic_increasing_f_A, even_and_monotonic_increasing_f_D, not_monotonic_increasing_f_B, not_even_f_C⟩

end even_and_monotonic_increasing_f_A_not_monotonic_increasing_f_B_not_even_f_C_even_and_monotonic_increasing_f_D_correct_answers_l428_428406


namespace light_intensity_panels_l428_428910

theorem light_intensity_panels (a : ℝ) (h_a : 0 < a) : ∃ x : ℕ, (0.9 : ℝ)^x < 1 / 11 ∧ x = 12 :=
by {
  use 12,
  norm_num,
  sorry
}

end light_intensity_panels_l428_428910


namespace true_proposition_is_D_l428_428917

open Real

theorem true_proposition_is_D :
  (∃ x_0 : ℝ, exp x_0 ≤ 0) = False ∧
  (∀ x : ℝ, 2 ^ x > x ^ 2) = False ∧
  (∀ a b : ℝ, a + b = 0 ↔ a / b = -1) = False ∧
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) = True :=
by
    sorry

end true_proposition_is_D_l428_428917


namespace inscribed_triangle_area_l428_428903

theorem inscribed_triangle_area 
  (r : ℝ) (theta : ℝ) 
  (A B C : ℝ) (arc1 arc2 arc3 : ℝ)
  (h_arc1 : arc1 = 5)
  (h_arc2 : arc2 = 7)
  (h_arc3 : arc3 = 8)
  (h_sum_arcs : arc1 + arc2 + arc3 = 2 * π * r)
  (h_theta : theta = 20)
  -- in radians: h_theta_rad : θ = (20 * π / 180)
  (h_A : A = 100)
  (h_B : B = 140)
  (h_C : C = 120) :
  let sin_A := sin (A * π / 180)
    sin_B := sin (B * π / 180)
    sin_C := sin (C * π / 180) in
  1 / 2 * (10 / π) ^ 2 * (sin_A + sin_B + sin_C) = 249.36 / π^2 := 
sorry

end inscribed_triangle_area_l428_428903


namespace pentagon_area_l428_428984

theorem pentagon_area (A B C D E : Point)
  (hAB_BC : distance A B = distance B C)
  (hCD_DE : distance C D = distance D E)
  (hABC : angle A B C = 150)
  (hCDE : angle C D E = 30)
  (hBD : distance B D = 2) :
  area_of_pentagon A B C D E = 1 :=
  sorry

end pentagon_area_l428_428984


namespace part_a_part_b_part_c_l428_428849

-- Definitions of the conditions directly used in the problem
variable {n : ℕ}
variable (strength : Fin (2^n) → Fin (n+1) → ℕ)

-- Definitions:
def possible_winners (athletes : Fin (2^n)) : Set (Fin (2^n)) :=
  { a | ∃ (arrangement : Fin (n+1) → Fin (n+1)), ∀ i j, (i < j → strength (arrangement a) i > strength (arrangement a) j) }

-- a) Prove that it is possible for at least half of the athletes to be possible winners.
theorem part_a : (∃ (S : Set (Fin (2^n))), S ⊆ possible_winners strength ∧ S.card ≥ 2^(n-1)) :=
by
  sorry

-- b) Prove that the number of possible winners does not exceed 2^n - n.
theorem part_b : (possible_winners strength).card ≤ 2^n - n :=
by
  sorry

-- c) Prove that it is possible to have exactly 2^n - n possible winners.
theorem part_c : (∃ (S : Set (Fin (2^n))), S ⊆ possible_winners strength ∧ S.card = 2^n - n) :=
by
  sorry

end part_a_part_b_part_c_l428_428849


namespace diagonal_squared_eq_l428_428357

theorem diagonal_squared_eq (a b c d : ℝ) (h1 : ∃ θ : ℝ, diag_bisects_angle θ a b c d) :
  d^2 = ab + (ac^2 - bd^2) / (a - b) := by
  sorry

-- Additional definitions to specify the given conditions more formally would be added here.
-- E.g., define what it means for the diagonal to bisect the angle at the vertex.

def diag_bisects_angle (θ : ℝ) (a b c d : ℝ) : Prop := 
  -- This would encapsulate the angle bisector property for the quadrilateral 
  sorry

end diagonal_squared_eq_l428_428357


namespace f_n_no_roots_in_unit_disk_l428_428285

noncomputable def f_n (n : ℕ) (z : ℂ) : ℂ :=
  ∑ k in finset.range n, (n - k) * z ^ k

theorem f_n_no_roots_in_unit_disk (n : ℕ) (hn : 0 < n) :
  ∀ z : ℂ, abs z ≤ 1 → f_n n z ≠ 0 :=
sorry

end f_n_no_roots_in_unit_disk_l428_428285


namespace range_of_m_l428_428160

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 4 / y = 1) (H : x + y > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l428_428160


namespace inscribed_triangle_area_l428_428902

theorem inscribed_triangle_area 
  (r : ℝ) (theta : ℝ) 
  (A B C : ℝ) (arc1 arc2 arc3 : ℝ)
  (h_arc1 : arc1 = 5)
  (h_arc2 : arc2 = 7)
  (h_arc3 : arc3 = 8)
  (h_sum_arcs : arc1 + arc2 + arc3 = 2 * π * r)
  (h_theta : theta = 20)
  -- in radians: h_theta_rad : θ = (20 * π / 180)
  (h_A : A = 100)
  (h_B : B = 140)
  (h_C : C = 120) :
  let sin_A := sin (A * π / 180)
    sin_B := sin (B * π / 180)
    sin_C := sin (C * π / 180) in
  1 / 2 * (10 / π) ^ 2 * (sin_A + sin_B + sin_C) = 249.36 / π^2 := 
sorry

end inscribed_triangle_area_l428_428902


namespace rectangle_count_5x5_l428_428649

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428649


namespace triangle_area_inscribed_in_circle_l428_428908

noncomputable def circle_inscribed_triangle_area : ℝ :=
  let r := 10 / Real.pi
  let angle_A := Real.pi / 2
  let angle_B := 7 * Real.pi / 10
  let angle_C := 4 * Real.pi / 5
  let sin_sum := Real.sin(angle_A) + Real.sin(angle_B) + Real.sin(angle_C)
  1 / 2 * r^2 * sin_sum

theorem triangle_area_inscribed_in_circle (h_circumference : 5 + 7 + 8 = 20)
  (h_radius : 10 / Real.pi * 2 * Real.pi = 20) :
  circle_inscribed_triangle_area = 138.005 / Real.pi^2 :=
by
  sorry

end triangle_area_inscribed_in_circle_l428_428908


namespace pyarelal_loss_l428_428842

theorem pyarelal_loss (total_loss : ℝ) (P : ℝ) (Ashok_capital : ℝ) (ratio_Ashok_Pyarelal : ℝ) :
  total_loss = 670 →
  Ashok_capital = P / 9 →
  ratio_Ashok_Pyarelal = 1 / 9 →
  Pyarelal_loss = 603 :=
by
  intro total_loss_eq Ashok_capital_eq ratio_eq
  sorry

end pyarelal_loss_l428_428842


namespace Y_lies_on_median_BM_l428_428290

variable {Ω1 Ω2 : Type}
variable {A B C M : Ω2}
variable [EuclideanGeometry Ω2]

-- Definitions coming from conditions
variable (Y : Ω2)
variable (hY1 : Y ∈ circle_omega1) (hY2 : Y ∈ circle_omega2)
variable (hSameSide : SameSide Y B (Line AC))

-- The theorem we want to prove
theorem Y_lies_on_median_BM :
  LiesOnMedian Y B M := 
  sorry

end Y_lies_on_median_BM_l428_428290


namespace find_locus_of_T1_T2_l428_428051

noncomputable def point (α : Type*) := { p : α × α // true }
structure TriangleLocusProblem :=
  (A B X O : point ℝ)
  (A_eq : A.val = (0, 0))
  (B_eq : B.val = (b, 0))
  (X_eq : X.val = (t, 0))
  (O_eq : O.val = (t, λ))
  (X_on_AB : t < b)
  (O_on_r : True) -- This is a placeholder as the line constraint is redundant with O_eq
  (omega : Type)
  (circle_def : ∀ p : point ℝ, p ∈ omega ↔ (p.val.1 - t)^2 + (p.val.2 - λ)^2 = t^2 + λ^2)

theorem find_locus_of_T1_T2
  {A B X O : point ℝ}
  (A_eq : A.val = (0, 0))
  (B_eq : B.val = (b, 0))
  (X_eq : X.val = (t, 0))
  (O_eq : O.val = (t, λ))
  (X_on_AB : t < b)
  (omega : Type)
  (circle_def : ∀ p : point ℝ, p ∈ omega ↔ (p.val.1 - t)^2 + (p.val.2 - λ)^2 = t^2 + λ^2) :
  (∃ (T1 T2 : point ℝ), (T1.val.1 - b)^2 + T1.val.2^2 = (b * t)^2 ∧ (T2.val.1 - b)^2 + T2.val.2^2 = (b * t)^2) :=
sorry

end find_locus_of_T1_T2_l428_428051


namespace calculate_second_half_speed_l428_428453

noncomputable def speed_second_half (total_distance : ℕ) (first_half_speed : ℕ) (total_time : ℕ) : ℕ :=
  let first_half_time := total_distance / 2 / first_half_speed in
  let second_half_distance := total_distance / 2 in
  let second_half_time := total_time - first_half_time in
  second_half_distance / second_half_time

theorem calculate_second_half_speed :
  speed_second_half 960 20 40 = 30 := by
  -- Here we would normally include the proof.
  sorry

end calculate_second_half_speed_l428_428453


namespace percentage_of_girl_scouts_with_slips_l428_428841

-- Define the proposition that captures the problem
theorem percentage_of_girl_scouts_with_slips 
    (total_scouts : ℕ)
    (scouts_with_slips : ℕ := total_scouts * 60 / 100)
    (boy_scouts : ℕ := total_scouts * 45 / 100)
    (boy_scouts_with_slips : ℕ := boy_scouts * 50 / 100)
    (girl_scouts : ℕ := total_scouts - boy_scouts)
    (girl_scouts_with_slips : ℕ := scouts_with_slips - boy_scouts_with_slips) :
  (girl_scouts_with_slips * 100 / girl_scouts) = 68 :=
by 
  -- The proof goes here
  sorry

end percentage_of_girl_scouts_with_slips_l428_428841


namespace sum_geq_20_for_m_eq_3_min_n_for_m_eq_2018_l428_428885

variable (A : List ℕ) (n : ℕ) (m : ℕ)

-- Condition for sequences A_n
def valid_sequence (A : List ℕ) (n : ℕ) (m : ℕ) : Prop :=
  A.length = n ∧ 
  A.head = some 1 ∧ 
  A.getLast (by simp) = m ∧ 
  (∀ k, k < n - 1 → (A[k+1] - A[k] = 0 ∨ A[k+1] - A[k] = 1)) ∧
  (∀ i j s t, i < n ∧ j < n ∧ s < n ∧ t < n ∧ list.pairwise (≠) [i, j, s, t] →
    A[i] + A[j] = A[s] + A[t])

-- Part I
def sequences_meeting_conditions_for_m_eq_2 : List (List ℕ) :=
  [[1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 1, 2, 2, 2, 2]]

-- Part II
theorem sum_geq_20_for_m_eq_3 (A : List ℕ) (n : ℕ) (h_valid : valid_sequence A n 3) : 
  list.sum A ≥ 20 :=
sorry

-- Part III
theorem min_n_for_m_eq_2018 (A : List ℕ) : 
  valid_sequence A 2026 2018 :=
sorry

end sum_geq_20_for_m_eq_3_min_n_for_m_eq_2018_l428_428885


namespace batting_average_46_innings_l428_428427

-- Given conditions
variables {A H L : ℕ}
def total_in_46_innings (A : ℕ) : ℕ := 46 * A
def total_in_44_innings (avg : ℕ) : ℕ := 44 * avg
def highest_score : ℕ := 156
def lowest_score (H L : ℕ) : ℕ := H - 150

-- Proof statement
theorem batting_average_46_innings :
  let A := 59 in
  (H = highest_score) → (H - L = 150) → (total_in_44_innings 58 + highest_score + lowest_score H L = total_in_46_innings A) :=
sorry

end batting_average_46_innings_l428_428427


namespace inequality_C_incorrect_l428_428916

theorem inequality_C_incorrect (x : ℝ) (h : x ≠ 0) : ¬(e^x < 1 + x) → (e^1 ≥ 1 + 1) :=
by {
  sorry
}

end inequality_C_incorrect_l428_428916


namespace logarithm_expression_evaluation_l428_428123

theorem logarithm_expression_evaluation :
  (3 / (log 3 (1000 ^ 4))) + (2 / (log 5 (1000 ^ 4))) = 1 / 12 :=
by
  sorry

end logarithm_expression_evaluation_l428_428123


namespace advanced_harmonious_group_validity_l428_428076

/-- Shooter A's hit rate --/
def P1 : ℚ := 2 / 3

/-- Shooter B's hit rate --/
variable (P2 : ℚ)

/-- Probability of being recognized as an "Advanced Harmonious Group" when P2 = 1/2 --/
def probability_advanced_harmonious_group : ℚ :=
  (2 * P1 * (1 - P1)) * (2 * (1/2) * (1/2)) + P1^2 * (1/2)^2

noncomputable def condition_1 : Prop :=
  probability_advanced_harmonious_group = 2 / 9

/-- Determine the valid range of P2 given Eξ ≥ 5 --/
noncomputable def valid_range_P2 : Prop :=
  let P := (8 / 9) * P2 - (4 / 9) * P2^2 in
  12 * P >= 5 → 3 / 4 ≤ P2 ∧ P2 ≤ 1

/-- Statement to be proved in Lean --/
theorem advanced_harmonious_group_validity : condition_1 ∧ valid_range_P2 sorry

end advanced_harmonious_group_validity_l428_428076


namespace total_birds_caught_l428_428867

theorem total_birds_caught 
  (day_birds : ℕ) 
  (night_birds : ℕ)
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) 
  : day_birds + night_birds = 24 := 
by 
  sorry

end total_birds_caught_l428_428867


namespace part1_part2_l428_428531

variables (x y : ℝ)
def x_value := sqrt 5 + sqrt 3
def y_value := sqrt 5 - sqrt 3

theorem part1 : 
  x = x_value → 
  y = y_value → 
  (x^2 + 2*x*y + y^2) / (x^2 - y^2) = (sqrt 15) / 3 :=
by
  intro h₁ h₂
  sorry

theorem part2 : 
  x = x_value → 
  y = y_value → 
  sqrt (x^2 + y^2 - 3) = sqrt 13 ∨ sqrt (x^2 + y^2 - 3) = - sqrt 13 :=
by
  intro h₁ h₂
  sorry

end part1_part2_l428_428531


namespace rectangles_in_grid_l428_428668

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428668


namespace positive_difference_between_numbers_l428_428008

theorem positive_difference_between_numbers:
  ∃ x y : ℤ, x + y = 40 ∧ 3 * y - 4 * x = 7 ∧ |y - x| = 6 := by
  sorry

end positive_difference_between_numbers_l428_428008


namespace find_f_one_fourth_l428_428694

variable (x : ℝ)

def g (x : ℝ) : ℝ := 1 - x^2
noncomputable def f : ℝ → ℝ := λ x, if x ≠ 0 then (1 - x^2) / x^2 else 0

theorem find_f_one_fourth : f (g (sqrt (3 / 4))) = 1 / 3 :=
by 
  have h : g (sqrt (3 / 4)) = 1 / 4,
  { sorry },
  rw h,
  have h2 : f (1 / 4) = 1 / 3,
  { sorry },
  exact h2

end find_f_one_fourth_l428_428694


namespace balloon_minimum_volume_l428_428432

theorem balloon_minimum_volume 
  (p V : ℝ)
  (h1 : p * V = 24000)
  (h2 : p ≤ 40000) : 
  V ≥ 0.6 :=
  sorry

end balloon_minimum_volume_l428_428432


namespace probability_grade_A_branch_a_probability_grade_A_branch_b_average_profit_branch_a_average_profit_branch_b_select_branch_l428_428443

def frequencies_branch_a := (40, 20, 20, 20) -- (A, B, C, D)
def frequencies_branch_b := (28, 17, 34, 21) -- (A, B, C, D)

def fees := (90, 50, 20, -50)  -- (A, B, C, D respectively)
def processing_cost_branch_a := 25
def processing_cost_branch_b := 20

theorem probability_grade_A_branch_a :
  let (fa, fb, fc, fd) := frequencies_branch_a in
  (fa : ℝ) / 100 = 0.4 := by
  sorry

theorem probability_grade_A_branch_b :
  let (fa, fb, fc, fd) := frequencies_branch_b in
  (fa : ℝ) / 100 = 0.28 := by
  sorry

theorem average_profit_branch_a :
  let (fa, fb, fc, fd) := frequencies_branch_a in
  let (qa, qb, qc, qd) := fees in
  ((qa - processing_cost_branch_a) * (fa / 100) + 
   (qb - processing_cost_branch_a) * (fb / 100) +
   (qc - processing_cost_branch_a) * (fc / 100) +
   (qd - processing_cost_branch_a) * (fd / 100) : ℝ) = 15 := by
  sorry

theorem average_profit_branch_b :
  let (fa, fb, fc, fd) := frequencies_branch_b in
  let (qa, qb, qc, qd) := fees in
  ((qa - processing_cost_branch_b) * (fa / 100) + 
   (qb - processing_cost_branch_b) * (fb / 100) +
   (qc - processing_cost_branch_b) * (fc / 100) +
   (qd - processing_cost_branch_b) * (fd / 100) : ℝ) = 10 := by
  sorry

theorem select_branch :
  let profit_a := 15 in
  let profit_b := 10 in
  profit_a > profit_b → 
  "Branch A" = "Branch A" := by
  sorry

end probability_grade_A_branch_a_probability_grade_A_branch_b_average_profit_branch_a_average_profit_branch_b_select_branch_l428_428443


namespace num_rectangles_in_5x5_grid_l428_428629

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428629


namespace math_proof_problem_l428_428135

noncomputable def problem_statement : Prop :=
  let a_bound := 14
  let b_bound := 7
  let c_bound := 14
  let num_square_divisors := (a_bound / 2 + 1) * (b_bound / 2 + 1) * (c_bound / 2 + 1)
  let num_cube_divisors := (a_bound / 3 + 1) * (b_bound / 3 + 1) * (c_bound / 3 + 1)
  let num_sixth_power_divisors := (a_bound / 6 + 1) * (b_bound / 6 + 1) * (c_bound / 6 + 1)
  
  num_square_divisors + num_cube_divisors - num_sixth_power_divisors = 313

theorem math_proof_problem : problem_statement := by sorry

end math_proof_problem_l428_428135


namespace circumcircle_BQY_tangent_XY_l428_428581

variables {A B C X Q Y : Type} 

-- Define the basic geometric entities and relationships
variables [Geometry Point Circles Tangents]

/-- Given circle Γ passes through points A and B and is tangent to AC. -/
axiom Circle_Gamma : Circle
axiom Passes_A : PassesThrough Circle_Gamma A
axiom Passes_B : PassesThrough Circle_Gamma B
axiom Tangent_AC : TangentTo Circle_Gamma AC

/-- Tangent at point B on circle Γ meets AC at point X (distinct from C). -/
axiom Tangent_At_B : TangentAt Circle_Gamma B
axiom Intersection_X : Intersection Point Circle_Gamma.Tangent AC = X
axiom Distinct_X_C : X ≠ C

/-- Circumcircle of triangle BXC (Γ1) intersects Γ at point Q (distinct from B). -/
axiom Circle_Gamma1 : Circumcircle Triangle_BXC
axiom Intersects_Q : Intersects Circle_Gamma1 Circle_Gamma = Q 
axiom Distinct_Q_B : Q ≠ B

/-- Tangent at point X to circle Γ1 intersects AB at point Y. -/
axiom Tangent_At_X : TangentAt Circle_Gamma1 X
axiom Intersection_Y : Intersection Point Circle_Gamma1.Tangent AB = Y

/-- Prove that the circumcircle of triangle BQY is tangent to XY. -/
theorem circumcircle_BQY_tangent_XY :
  TangentTo (Circumcircle Triangle_BQY) XY :=
sorry

end circumcircle_BQY_tangent_XY_l428_428581


namespace min_N_such_that_next_person_sits_next_to_someone_l428_428448

def circular_table_has_80_chairs : Prop := ∃ chairs : ℕ, chairs = 80
def N_people_seated (N : ℕ) : Prop := N > 0
def next_person_sits_next_to_someone (N : ℕ) : Prop :=
  ∀ additional_person_seated : ℕ, additional_person_seated ≤ N → additional_person_seated > 0 
  → ∃ adjacent_person : ℕ, adjacent_person ≤ N ∧ adjacent_person > 0
def smallest_value_for_N (N : ℕ) : Prop :=
  (∀ k : ℕ, k < N → ¬next_person_sits_next_to_someone k)

theorem min_N_such_that_next_person_sits_next_to_someone :
  circular_table_has_80_chairs →
  smallest_value_for_N 20 :=
by
  intro h
  sorry

end min_N_such_that_next_person_sits_next_to_someone_l428_428448


namespace basketball_player_probability_l428_428425

-- Define the binomial coefficient function
noncomputable def binom (n k : ℕ) : ℝ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the probability function for the binomial distribution
noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  binom n k * p^k * (1 - p)^(n - k)

-- Declare the main theorem
theorem basketball_player_probability (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  binomial_prob 10 3 p = binom 10 3 * p^3 * (1 - p)^7 :=
by
  -- We state the answer directly as per the solution
  sorry

end basketball_player_probability_l428_428425


namespace simplify_expression_l428_428346

theorem simplify_expression : (0.25 ^ (-2) + 8 ^ (2 / 3) - log 25 / log 10 - 2 * (log 2 / log 10)) = 18 := by
  sorry

end simplify_expression_l428_428346


namespace number_of_rectangles_in_5x5_grid_l428_428661

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428661


namespace total_journey_time_eq_5_l428_428040

-- Define constants for speed and times
def speed1 : ℕ := 40
def speed2 : ℕ := 60
def total_distance : ℕ := 240
def time1 : ℕ := 3

-- Noncomputable definition to avoid computation issues
noncomputable def journey_time : ℕ :=
  let distance1 := speed1 * time1
  let distance2 := total_distance - distance1
  let time2 := distance2 / speed2
  time1 + time2

-- Theorem to state the total journey time
theorem total_journey_time_eq_5 : journey_time = 5 := by
  sorry

end total_journey_time_eq_5_l428_428040


namespace total_apples_l428_428854

theorem total_apples (baskets apples_per_basket : ℕ) (h1 : baskets = 37) (h2 : apples_per_basket = 17) : baskets * apples_per_basket = 629 := by
  sorry

end total_apples_l428_428854


namespace find_a_l428_428140

theorem find_a (a : ℝ)
    (h1 : (set_of (λ p : ℝ × ℝ, (p.1^2 + p.2^2 - 6 * p.1 - 2 * p.2 + 3 = 0))) = { p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 7 })
    (h2 : (set_of (λ p : ℝ × ℝ, (p.1 + a * p.2 - 1 = 0))) = { p : ℝ × ℝ | p.1 + a * p.2 - 1 = 0 }) :
    ∀ (d : ℝ), d = 1 → d = real.dist (3, 1) (set_of (λ p : ℝ × ℝ, p.1 + a * p.2 - 1 = 0)) → a = -3 / 4 :=
by
  intros d h_d_eq_1 h_d_correct
  sorry

end find_a_l428_428140


namespace target_hit_probability_l428_428858

theorem target_hit_probability (prob_A_hits : ℝ) (prob_B_hits : ℝ) (hA : prob_A_hits = 0.5) (hB : prob_B_hits = 0.6) :
  (1 - (1 - prob_A_hits) * (1 - prob_B_hits)) = 0.8 := 
by 
  sorry

end target_hit_probability_l428_428858


namespace num_rectangles_grid_l428_428615

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428615


namespace tan_cos_squared_l428_428336

-- Definitions for sin, cos, and tan from the conditions:
def P : ℝ × ℝ := (-1, 2)
def r : ℝ := real.sqrt ((-1)^2 + 2^2)
def α : ℝ := real.arctan (2 / -1)
def sin_α := P.2 / r
def cos_α := P.1 / r

-- The theorem to be proved:
theorem tan_cos_squared (P : ℝ × ℝ) (hP1 : P.1 = -1) (hP2 : P.2 = 2) :
  let r := real.sqrt (P.1^2 + P.2^2) in
  let sin_α := P.2 / r in
  let cos_α := P.1 / r in
  (sin_α / cos_α) / cos_α^2 = -10 :=
by
  sorry

end tan_cos_squared_l428_428336


namespace number_of_rectangles_in_5x5_grid_l428_428654

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428654


namespace number_of_pencils_l428_428363

theorem number_of_pencils (P : ℕ) (h : ∃ (n : ℕ), n * 4 = P) : ∃ k, 4 * k = P :=
  by
  sorry

end number_of_pencils_l428_428363


namespace reinforcement_size_l428_428063

theorem reinforcement_size :
  ∃ R, 
    let men_initial := 2000 in
    let daily_consumption_per_man := 1.5 in
    let initial_days := 40 in
    let days_before_reinforcement := 20 in
    let daily_consumption_per_reinforcement_man := 2 in
    let additional_days := 10 in
    let total_provisions := men_initial * daily_consumption_per_man * initial_days in
    let provisions_used_before_reinforcement := men_initial * daily_consumption_per_man * days_before_reinforcement in
    let remaining_provisions := total_provisions - provisions_used_before_reinforcement in
    remaining_provisions = (men_initial * daily_consumption_per_man + R * daily_consumption_per_reinforcement_man) * additional_days → 
    R = 1500 :=
by {
  sorry
}

end reinforcement_size_l428_428063


namespace rectangle_count_5x5_l428_428645

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428645


namespace Xiaogang_Mathematics_score_l428_428410

theorem Xiaogang_Mathematics_score :
  ∀ (x : ℝ), (x + 88 + 91) / 3 = 90 → x = 91 := 
by {
  intros x h,
  sorry
}

end Xiaogang_Mathematics_score_l428_428410


namespace intersection_a_zero_range_of_a_l428_428187

variable (x a : ℝ)

def setA : Set ℝ := { x | - 1 < x ∧ x < 6 }
def setB (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x < a + 5 }

theorem intersection_a_zero :
  setA x ∧ setB 0 x ↔ - 1 < x ∧ x < 5 := by
  sorry

theorem range_of_a (h : ∀ x, setA x ∨ setB a x → setA x) :
  (0 < a ∧ a ≤ 1) ∨ 6 ≤ a :=
  sorry

end intersection_a_zero_range_of_a_l428_428187


namespace find_y_l428_428790

theorem find_y (n x y : ℕ) 
    (h1 : (n + 200 + 300 + x) / 4 = 250)
    (h2 : (300 + 150 + n + x + y) / 5 = 200) :
    y = 50 := 
by
  -- Placeholder for the proof
  sorry

end find_y_l428_428790


namespace largest_k_in_checkerboard_l428_428105

theorem largest_k_in_checkerboard (k : ℕ) (grid : ℕ → ℕ → ℕ) :
  (∀ x y, grid x y < k) →
  (∀ x y, ∃ a b, a ≠ b ∧ 
    ∀ i j, ((i: ℕ) < 3) → ((j: ℕ) < 4) → (grid (x + i) (y + j) = a) ∨ (grid (x + i) (y + j) = b)
  ) →
  k ≤ 10 :=
begin
  sorry,
end

end largest_k_in_checkerboard_l428_428105


namespace number_of_rectangles_in_grid_l428_428587

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428587


namespace airlines_round_trip_l428_428424

theorem airlines_round_trip :
  ∃ airline : ℕ → Prop, (∀ cities : ℕ, cities = 1983 → ∀ airlines : ℕ, airlines = 10 → 
  ∀ (services : (ℕ × ℕ) → ℕ → Prop), 
  (∀ (c₁ c₂ : ℕ), services (c₁, c₂)) → 
  ∃ t, (∀ c : ℕ, services (t c, t (c + 1))) ∧ 
  (t 0 = t (c - 1)) ∧ 
  odd (t.bind_length)) :=
begin
  sorry
end

end airlines_round_trip_l428_428424


namespace probability_of_condition1_before_condition2_l428_428428

-- Definitions for conditions
def condition1 (draw_counts : List ℕ) : Prop :=
  ∃ count ∈ draw_counts, count ≥ 3

def condition2 (draw_counts : List ℕ) : Prop :=
  ∀ count ∈ draw_counts, count ≥ 1

-- Probability function
def probability_condition1_before_condition2 : ℚ :=
  13 / 27

-- The proof statement
theorem probability_of_condition1_before_condition2 :
  (∃ draw_counts : List ℕ, (condition1 draw_counts) ∧  ¬(condition2 draw_counts)) →
  probability_condition1_before_condition2 = 13 / 27 :=
sorry

end probability_of_condition1_before_condition2_l428_428428


namespace range_of_quadratic_function_l428_428579

theorem range_of_quadratic_function :
  ∀ x ∈ Icc (0 : ℝ) 3, (1 : ℝ) ≤ (-x^2 + 4*x + 1) ∧ (-x^2 + 4*x + 1) ≤ 5 :=
by
  sorry

end range_of_quadratic_function_l428_428579


namespace cone_height_ratio_l428_428389

theorem cone_height_ratio (l : ℝ) (h_A h_B S_A S_B : ℝ) (α β : ℝ) 
  (h_equal_slant : l > 0) 
  (h_sum_angles : α + β = 2 * real.pi) 
  (h_area_ratio : S_A / S_B = 2) 
  (h_lateral_A : S_A = (1 / 2) * α * l^2) 
  (h_lateral_B : S_B = (1 / 2) * β * l^2) 
  (h_height_A : h_A = sqrt (l^2 - ((α * l) / (2 * real.pi))^2)) 
  (h_height_B : h_B = sqrt (l^2 - ((β * l) / (2 * real.pi))^2)) 
: h_A / h_B = sqrt(10) / 4 := 
sorry

end cone_height_ratio_l428_428389


namespace prob_relations_l428_428249

-- Define the probabilities for each method
def P1 : ℚ := 1 / 3
def P2 : ℚ := 1 / 2
def P3 : ℚ := 2 / 3

-- Prove the relations between the probabilities
theorem prob_relations : (P1 < P2) ∧ (P1 < P3) ∧ (2 * P1 = P3) :=
by {
  -- Individual proofs for each part (not needed, just 'sorry' them) 
  have h1 : P1 < P2, sorry,
  have h2 : P1 < P3, sorry,
  have h3 : 2 * P1 = P3, sorry,
  exact ⟨h1, h2, h3⟩
}

end prob_relations_l428_428249


namespace impossible_fractions_l428_428282

theorem impossible_fractions (a b c r s t : ℕ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_pos_t : 0 < t)
  (h1 : a * b + 1 = r ^ 2) (h2 : a * c + 1 = s ^ 2) (h3 : b * c + 1 = t ^ 2) :
  ¬ (∃ (k1 k2 k3 : ℕ), rt / s = k1 ∧ rs / t = k2 ∧ st / r = k3) :=
by
  sorry

end impossible_fractions_l428_428282


namespace minimize_sum_of_sequence_l428_428540

theorem minimize_sum_of_sequence (a : ℕ → ℤ) (a_n_def : ∀ n, a n = 3 * n - 28) : 
  ∃ n, n = 9 ∧ ∀ m, m ≥ n → (3 * m - 28) ≤ 0 := 
begin
  sorry
end

end minimize_sum_of_sequence_l428_428540


namespace problem_l428_428890

noncomputable def parabola (x : ℝ) : ℝ := (1 / 4) * x^2

def distance (P F : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2)

theorem problem (P : ℝ × ℝ) (hP : P.1 ^ 2 = 4 * P.2) (h_area : (1 / 2) * |(1 / 2) * P.1| * |-(1 / 4) * P.1 ^ 2| = 1 / 2):
  distance P (0, 1) = 2 :=
by
  sorry

end problem_l428_428890


namespace sum_of_k_for_distinct_roots_eq_zero_l428_428031

def has_distinct_integer_roots (a b c : ℤ) : Prop :=
  ∃ p q : ℤ, p ≠ q ∧ a * p * p + b * p + c = 0 ∧ a * q * q + b * q + c = 0

theorem sum_of_k_for_distinct_roots_eq_zero :
  (∑ k in { k : ℤ | has_distinct_integer_roots 3 (-k) 9 }, k) = 0 :=
sorry

end sum_of_k_for_distinct_roots_eq_zero_l428_428031


namespace parallel_vectors_m_value_l428_428709

theorem parallel_vectors_m_value :
  ∃ (m : ℝ), let a := (3, 1), b := (m, m + 1) in (3 * (m + 1) - 1 * m = 0) ∧ m = -3/2 :=
by
  sorry

end parallel_vectors_m_value_l428_428709


namespace ratio_second_largest_to_largest_coefficient_l428_428518

theorem ratio_second_largest_to_largest_coefficient :
  let a := Nat.choose 8 4 in
  let b := (Nat.choose 8 5) * (2^5) in
  (b : ℚ) / (a : ℚ) = 128 / 5 :=
by
  sorry

end ratio_second_largest_to_largest_coefficient_l428_428518


namespace median_length_l428_428564

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def length_of_median (a b c : ℕ) (h : is_right_triangle a b c) : ℝ :=
  c / 2

theorem median_length (a b c : ℕ) (h : is_right_triangle a b c) : length_of_median a b c h = 5 :=
by {
  have h1: c = 10 := sorry,
  have h2: a = 6 := sorry,
  have h3: b = 8 := sorry,
  unfold length_of_median,
  rw h1,
  norm_num,
}

end median_length_l428_428564


namespace evaluate_expression_l428_428412

theorem evaluate_expression :
  11 + sqrt (-4 + 6 * 4 / 3) = 13 :=
sorry

end evaluate_expression_l428_428412


namespace b_value_for_continuity_at_2_l428_428144

def f (x : ℝ) (b : ℝ) : ℝ := 
  if x > 2 then 
    x + 4 
  else 
    3 * x^2 + b

theorem b_value_for_continuity_at_2 (b : ℝ) : (∀ x, x = 2 → f x b = 6) ↔ b = -6 :=
by
  sorry

end b_value_for_continuity_at_2_l428_428144


namespace point_Y_lies_on_median_l428_428311

-- Define the geometric points and circles
variable (A B C M Y : Point)
variable (ω1 ω2 : Circle)

-- Definitions of the given conditions
variable (P : Point) (hP : P ∈ (ω1)) (hInt : ω1 ∩ ω2 = {Y})

-- Express conditions in terms of Lean definitions
variable (hSameSide : same_side Y B (line_through A C))
variable (hMedian : M = (midpoint A C))
variable (hBM : is_median B M)

-- The theorem that we need to prove
theorem point_Y_lies_on_median :
  Y ∈ line_through B M :=
sorry

end point_Y_lies_on_median_l428_428311


namespace num_rectangles_grid_l428_428619

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428619


namespace rectangle_count_5x5_l428_428647

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428647


namespace triangle_is_equilateral_l428_428154

theorem triangle_is_equilateral (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + ac + bc) : a = b ∧ b = c :=
by
  sorry

end triangle_is_equilateral_l428_428154


namespace minimum_volume_for_safety_l428_428429

noncomputable def pressure_is_inversely_proportional_to_volume (k V : ℝ) : ℝ :=
  k / V

-- Given conditions
def k := 8000 * 3
def p (V : ℝ) := pressure_is_inversely_proportional_to_volume k V
def balloon_will_explode (V : ℝ) : Prop := p V > 40000

-- Goal: To ensure the balloon does not explode, the volume V must be at least 0.6 m^3
theorem minimum_volume_for_safety : ∀ V : ℝ, (¬ balloon_will_explode V) → V ≥ 0.6 :=
by
  intro V
  unfold balloon_will_explode p pressure_is_inversely_proportional_to_volume
  intro h
  sorry

end minimum_volume_for_safety_l428_428429


namespace probability_both_hardcover_l428_428722

theorem probability_both_hardcover (total_books hardcover_books chosen_books: ℕ) (h1: total_books = 6) (h2: hardcover_books = 3) (h3: chosen_books = 2) : 
  (hardcover_books / total_books) * ((hardcover_books - 1) / (total_books - 1)) = 0.2 :=
by
  sorry

end probability_both_hardcover_l428_428722


namespace heartsuit_ratio_l428_428695

-- Define the operation \heartsuit
def heartsuit (n m : ℕ) : ℕ := n^3 * m^2

-- The proposition we want to prove
theorem heartsuit_ratio :
  heartsuit 2 4 / heartsuit 4 2 = 1 / 2 := by
  sorry

end heartsuit_ratio_l428_428695


namespace area_ADE_l428_428269

-- Definitions of the geometric setup and given values:
variables (A B C D E : Point)
variables (ABC : Triangle A B C)
variable (area_ABC : ℝ)
variable (midpoint_D : isMidpoint D A B)
variable (midpoint_E : isMidpoint E A C)

-- Given conditions:
axiom h_area_ABC : area_ABC = 80
axiom h_mid_D : midpoint_D
axiom h_mid_E : midpoint_E

theorem area_ADE (h_area_ABC : area ABC = 80) 
  (h_mid_D : isMidpoint D A B) 
  (h_mid_E : isMidpoint E A C) : 
  area (Triangle A D E) = 20 := 
sorry

end area_ADE_l428_428269


namespace find_avg_speed_l428_428874

variables (v t : ℝ)

noncomputable def avg_speed_cond := 
  (v + Real.sqrt 15) * (t - Real.pi / 4) = v * t

theorem find_avg_speed (h : avg_speed_cond v t) : v = Real.sqrt 15 :=
by
  sorry

end find_avg_speed_l428_428874


namespace inequality_I_l428_428111

theorem inequality_I (a b x y : ℝ) (hx : x < a) (hy : y < b) : x * y < a * b :=
sorry

end inequality_I_l428_428111


namespace sum_squares_even_odd_difference_l428_428484

theorem sum_squares_even_odd_difference :
  let sum_squares_natural := ∑ k in finset.range 101, k^2
  let sum_squares_100_even := 2^2 * sum_squares_natural
  let sum_squares_100_odd := ∑ n in finset.range 100, (2*n + 1)^2
  sum_squares_100_even - sum_squares_100_odd = 20100 :=
by
  sorry

end sum_squares_even_odd_difference_l428_428484


namespace find_v_l428_428751

open Matrix

def a : Vector ℝ 3 := ![3, 2, 1]
def b : Vector ℝ 3 := ![4, -1, 2]

def vector_cross_product (u v : Vector ℝ 3) : Vector ℝ 3 :=
  ![
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0]
  ]

theorem find_v (v : Vector ℝ 3) :
  let va := vector_cross_product v a
  let vb := vector_cross_product v b
  let ab := vector_cross_product a b
  2 • va = vector_cross_product b a ∧ vb = 2 • ab → 
    v = ![1, 4.5, 0] :=
by
  intros va vb ab h
  sorry

end find_v_l428_428751


namespace problem1_problem2_l428_428550

-- Define Set A
def setA (x : ℝ) : Prop := (x - 7) / (x + 2) > 0

-- Define Set B
def setB (x : ℝ) : Prop := -x^2 + 3 * x + 28 > 0

-- Define Set C based on parameter m
def setC (m x : ℝ) : Prop := (m + 1 ≤ x) ∧ (x ≤ 2 * m - 1)

-- Define complements and intersections
def complementR (s : ℝ → Prop) : ℝ → Prop := λ x, ¬ s x

def intersect (s1 s2 : ℝ → Prop) : ℝ → Prop := λ x, s1 x ∧ s2 x

-- Statement 1: Prove (complementR setA) ∩ setB = { x | -2 ≤ x < 7 }
theorem problem1 : ∀ x : ℝ, ((complementR setA) x ∧ setB x) ↔ ((-2 ≤ x) ∧ (x < 7)) := 
by
  sorry

-- Statement 2: If B ∪ C = B, then m < 4
theorem problem2 (m : ℝ) : (∀ x : ℝ, (setB x ∨ setC m x) ↔ setB x) → m < 4 :=
by
  sorry

end problem1_problem2_l428_428550


namespace card_rearrangement_l428_428381

def rearrange_cost (n : ℕ) (a : Fin n → ℕ) : ℕ :=
  ∑ i in Finset.range n, abs (a i - (i + 1))

theorem card_rearrangement (n : ℕ) (a : Fin n → ℕ) :
  ∃ (f : Fin n → Fin n), ∀ i, f (f i) = i ∧ 
  rearrange_cost n (λ i, (f i).val + 1) ≤ rearrange_cost n a :=
sorry

end card_rearrangement_l428_428381


namespace quadratic_completion_l428_428016

theorem quadratic_completion (x : ℝ) :
  (x^2 + 6 * x - 2) = ((x + 3)^2 - 11) := sorry

end quadratic_completion_l428_428016


namespace part_I_part_II_l428_428193

-- First, we need to state the conditions
variables {x : Real} (h1 : -π / 2 < x) (h2 : x < 0) (h3 : sin x + cos x = 1 / 5)

-- Prove the first part: sin x - cos x = -7/5
theorem part_I : (sin x - cos x = -7 / 5) :=
by
  sorry

-- Prove the second part: the value of the given complex expression
theorem part_II : ( (3 * sin^2 (x / 2) - 2 * sin (x / 2) * cos (x / 2) + cos^2 (x / 2)) / (tan x + cot x) = -108 / 125 ) :=
by
  sorry

end part_I_part_II_l428_428193


namespace largest_prime_factor_1755_l428_428830

/--
The largest prime factor of 1755 is 13.
-/
theorem largest_prime_factor_1755 : ∃ p : ℕ, nat.prime p ∧ p = 13 ∧ ∀ q : ℕ, nat.prime q → q ∣ 1755 → q ≤ 13 := by
  sorry

end largest_prime_factor_1755_l428_428830


namespace least_faces_combined_l428_428390

theorem least_faces_combined (a b : ℕ) (h1 : a ≥ 6) (h2 : b ≥ 6)
  (h3 : (∃ k : ℕ, k * a * b = 20) → (∃ m : ℕ, 2 * m = 10 * (k + 10))) 
  (h4 : (∃ n : ℕ, n = (a * b) / 10)) (h5 : ∃ l : ℕ, l = 5) : a + b = 20 :=
by
  sorry

end least_faces_combined_l428_428390


namespace omega_range_for_monotonicity_l428_428552

theorem omega_range_for_monotonicity
  (ω : ℝ)
  (hω : ω > 0)
  (monotonic_decreasing : ∀ x y : ℝ, (π/2 < x ∧ x < π) ∧ (π/2 < y ∧ y < π) ∧ x < y → f(ω, x) ≥ f(ω, y)) :
  ω ∈ set.Icc (1/2 : ℝ) (5/4 : ℝ) := 
sorry
where
  f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

end omega_range_for_monotonicity_l428_428552


namespace rectangle_area_solution_l428_428023

theorem rectangle_area_solution (x : ℝ) 
  (h1 : (x - 3) * (3 * x + 4) = 12 * x - 9) :
  x = (17 + 5 * real.sqrt 13) / 6 :=
by
  sorry

end rectangle_area_solution_l428_428023


namespace find_multiplier_l428_428697

theorem find_multiplier (x y n : ℤ) (h1 : 3 * x + y = 40) (h2 : 2 * x - y = 20) (h3 : y^2 = 16) :
  n * y^2 = 48 :=
by 
  -- proof goes here
  sorry

end find_multiplier_l428_428697


namespace trains_clear_time_l428_428020

noncomputable def time_to_clear_each_other
  (length_train1 length_train2 : ℕ)
  (speed_train1_kmh speed_train2_kmh : ℕ) : ℝ :=
let total_distance := length_train1 + length_train2 in
let relative_speed_kmh := speed_train1_kmh + speed_train2_kmh in
let relative_speed_ms := relative_speed_kmh * 1000 / 3600 in
total_distance / relative_speed_ms

theorem trains_clear_time
  (length_train1 length_train2 : ℕ)
  (speed_train1_kmh speed_train2_kmh : ℕ)
  (h1 : length_train1 = 110)
  (h2 : length_train2 = 200)
  (h3 : speed_train1_kmh = 80)
  (h4 : speed_train2_kmh = 65) :
  time_to_clear_each_other length_train1 length_train2 speed_train1_kmh speed_train2_kmh ≈ 7.695 :=
by
  have h_total_distance : length_train1 + length_train2 = 310 := by
    rw [h1, h2]
  have h_relative_speed_kmh : speed_train1_kmh + speed_train2_kmh = 145 := by
    rw [h3, h4]
  have h_relative_speed_ms : 145 * 1000 / 3600 ≈ 40.2778 := by
    sorry
  have h_time := (310 : ℝ) / (40.2778 : ℝ)
  have h_result : h_time ≈ 7.695 := by
    sorry
  exact h_result

end trains_clear_time_l428_428020


namespace number_of_rectangles_in_5x5_grid_l428_428662

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428662


namespace ant_growth_rate_correct_l428_428252

noncomputable def ant_growth_rate (A0 : ℕ) (A5 : ℕ) : ℝ :=
  let r := (A5.to_real / A0.to_real)^(1/5.0)
  r

theorem ant_growth_rate_correct :
  ant_growth_rate 50 1600 = 2 :=
by
  sorry

end ant_growth_rate_correct_l428_428252


namespace Carly_first_week_miles_l428_428937

theorem Carly_first_week_miles:
  ∃ (x : ℝ), 
  let second_week := 2 * x + 3,
      third_week := (9/7) * second_week,
      fourth_week := third_week - 5
  in 
    fourth_week = 4 ∧ x = 2 :=
by
  sorry

end Carly_first_week_miles_l428_428937


namespace average_cookies_per_package_l428_428481

def cookie_counts : List ℕ := [9, 11, 13, 15, 15, 17, 19, 21, 5]

theorem average_cookies_per_package :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 125 / 9 :=
by
  sorry

end average_cookies_per_package_l428_428481


namespace abby_bridget_adjacent_probability_l428_428471

theorem abby_bridget_adjacent_probability :
  let n := 9
  let total_arrangements := n!
  let row_pairs := 3 * 2 -- 3 rows, 2 adjacent pairs per row
  let column_pairs := 3 * 2 -- 3 columns, 2 adjacent pairs per column
  let ab_permutations := 2 -- 2 ways to arrange Abby and Bridget
  let remaining_arrangements := (n - 2)!
  let favorable_arrangements := (row_pairs + column_pairs) * ab_permutations * remaining_arrangements
  let probability := favorable_arrangements / total_arrangements 
  probability = 1 / 3 :=
by
  sorry

end abby_bridget_adjacent_probability_l428_428471


namespace max_value_of_f_l428_428138

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f (Real.exp 1) = 1 / Real.exp 1 := by
  sorry

end max_value_of_f_l428_428138


namespace area_of_inscribed_triangle_l428_428898

noncomputable def area_inscribed_triangle (r : ℝ) (a b c : ℝ) : ℝ :=
  1 / 2 * r^2 * (Real.sin a + Real.sin b + Real.sin c)

theorem area_of_inscribed_triangle : 
  ∃ (r a b c : ℝ),
    a + b + c = 2 * π ∧
    r = 10 / π ∧
    a = 5 * (18 * π / 180) ∧
    b = 7 * (18 * π / 180) ∧
    c = 8 * (18 * π / 180) ∧
    area_inscribed_triangle r a b c = 119.84 / π^2 :=
begin
  sorry
end

end area_of_inscribed_triangle_l428_428898


namespace f_2016_eq_2_l428_428208

def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * (1 - x)
else if 1 < x ∧ x ≤ 2 then x - 1
else 0

noncomputable def f_n : ℕ → ℝ → ℝ
| 0, x := x
| (n + 1), x := f (f_n n x)

theorem f_2016_eq_2 : f_n 2016 2 = 2 :=
sorry

end f_2016_eq_2_l428_428208


namespace sum_of_k_for_distinct_roots_eq_zero_l428_428030

def has_distinct_integer_roots (a b c : ℤ) : Prop :=
  ∃ p q : ℤ, p ≠ q ∧ a * p * p + b * p + c = 0 ∧ a * q * q + b * q + c = 0

theorem sum_of_k_for_distinct_roots_eq_zero :
  (∑ k in { k : ℤ | has_distinct_integer_roots 3 (-k) 9 }, k) = 0 :=
sorry

end sum_of_k_for_distinct_roots_eq_zero_l428_428030


namespace cory_fruit_orders_l428_428117

theorem cory_fruit_orders :
  let apples := 4
  let oranges := 2
  let bananas := 2
  let grapes := 1
  ∃ n : ℕ, n = 9! / (apples! * oranges! * bananas! * grapes!) ∧ n = 3780 :=
by
  let apples := 4
  let oranges := 2
  let bananas := 2
  let grapes := 1
  use 9! / (4! * 2! * 2! * 1!)
  split
  . reflexivity
  . norm_num
  . exact sorry

end cory_fruit_orders_l428_428117


namespace functional_eq_satisfied_prime_condition_exists_l428_428283

noncomputable def f : ℕ → ℕ := sorry

theorem functional_eq_satisfied (a b : ℕ) (h : f 0 ≠ 0):
  2 * f (a * b) = (b + 1) * f a + (a + 1) * f b :=
sorry

theorem prime_condition_exists (p : ℕ) (hp : Nat.Prime p) :
  ∃ q x : ℕ, ∃ xs : List ℕ, xs.All Nat.Prime ∧ ∃ m : ℕ,
  let product := List.foldr (λ xi acc, (p * xi + 1) * acc) (p^m) xs
  in f (q^p) / f q = product :=
sorry

end functional_eq_satisfied_prime_condition_exists_l428_428283


namespace dot_product_value_perpendicular_value_l428_428198

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (norm_a : ∥a∥ = 2) (norm_b : ∥b∥ = 3) 
variables (angle_ab : real.angleOf a b = real.pi / 3 * 2)

-- First proof problem
theorem dot_product_value :
  (2 • a - b) ⬝ (a + 3 • b) = -34 := sorry

-- Second proof problem
theorem perpendicular_value (x : ℝ) :
  (x • a - b) ⬝ (a + 3 • b) = 0 ↔ x = (-24 / 5 : ℝ) := sorry

end dot_product_value_perpendicular_value_l428_428198


namespace num_rectangles_grid_l428_428613

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428613


namespace solve_b_values_l428_428134

open Int

theorem solve_b_values :
  {b : ℤ | ∃ x1 x2 x3 : ℤ, x1^2 + b * x1 - 2 ≤ 0 ∧ x2^2 + b * x2 - 2 ≤ 0 ∧ x3^2 + b * x3 - 2 ≤ 0 ∧
  ∀ x : ℤ, x ≠ x1 ∧ x ≠ x2 ∧ x ≠ x3 → x^2 + b * x - 2 > 0} = { -4, -3 } :=
by sorry

end solve_b_values_l428_428134


namespace next_term_seq1_next_term_seq2_next_term_seq3_next_term_seq4_l428_428773

-- Sequence ①
def seq1 := ["A", "D", "G", "J"]
def next1 (s : List String) := s ++ ["M"]

theorem next_term_seq1 : next1 seq1 = ["A", "D", "G", "J", "M] :=
by
  sorry

-- Sequence ②
def seq2 := [21, 20, 18, 15, 11]
def next2 (s : List Nat) := s ++ [6]

theorem next_term_seq2 : next2 seq2 = [21, 20, 18, 15, 11, 6] :=
by
  sorry

-- Sequence ③
def seq3 := [8, 6, 7, 5, 6, 4]
def next3 (s : List Nat) := s ++ [5]

theorem next_term_seq3 : next3 seq3 = [8, 6, 7, 5, 6, 4, 5] :=
by
  sorry

-- Sequence ④
def seq4 := [18, 10, 6, 4]
def next4 (s : List Nat) := s ++ [3]

theorem next_term_seq4 : next4 seq4 = [18, 10, 6, 4, 3] :=
by
  sorry

end next_term_seq1_next_term_seq2_next_term_seq3_next_term_seq4_l428_428773


namespace compute_expression_l428_428237

theorem compute_expression (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 16 / (x - 3)^2 = 23 := 
  sorry

end compute_expression_l428_428237


namespace chandler_total_rolls_l428_428974

-- Definitions based on given conditions
def rolls_sold_grandmother : ℕ := 3
def rolls_sold_uncle : ℕ := 4
def rolls_sold_neighbor : ℕ := 3
def rolls_needed_more : ℕ := 2

-- Total rolls sold so far and needed
def total_rolls_to_sell : ℕ :=
  rolls_sold_grandmother + rolls_sold_uncle + rolls_sold_neighbor + rolls_needed_more

theorem chandler_total_rolls : total_rolls_to_sell = 12 :=
by
  sorry

end chandler_total_rolls_l428_428974


namespace smallest_integer_C_l428_428953

-- Define the function f(n) = 6^n / n!
def f (n : ℕ) : ℚ := (6 ^ n) / (Nat.factorial n)

theorem smallest_integer_C (C : ℕ) (h : ∀ n : ℕ, n > 0 → f n ≤ C) : C = 65 :=
by
  sorry

end smallest_integer_C_l428_428953


namespace find_pairs_l428_428964

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ n : ℕ, (n > 0) ∧ (a = n ∧ b = n) ∨ (a = n ∧ b = 1)) ↔ 
  (a^3 ∣ b^2) ∧ ((b - 1) ∣ (a - 1)) :=
by {
  sorry
}

end find_pairs_l428_428964


namespace conjugate_of_z_l428_428355

-- Define the complex number z as given in conditions
def z : ℂ := 5 * complex.I / (1 - 2 * complex.I)

-- State the theorem to prove
theorem conjugate_of_z : complex.conj z = -2 - complex.I :=
by
  -- Proof skipped (to be filled in)
  sorry

end conjugate_of_z_l428_428355


namespace expression_not_prime_l428_428405

open Nat

theorem expression_not_prime (p : Nat) (hp : Prime p) : ¬ Prime (p^2 + 18) :=
sorry

end expression_not_prime_l428_428405


namespace slower_train_speed_l428_428822

noncomputable def speed_of_slower_train (v_f : ℕ) (l1 l2 : ℚ) (t : ℚ) : ℚ :=
  let total_distance := l1 + l2
  let time_in_hours := t / 3600
  let relative_speed := total_distance / time_in_hours
  relative_speed - v_f

theorem slower_train_speed :
  speed_of_slower_train 210 (11 / 10) (9 / 10) 24 = 90 := by
  sorry

end slower_train_speed_l428_428822


namespace brother_birth_year_1990_l428_428275

variable (current_year : ℕ) -- Assuming the current year is implicit for the problem, it should be 2010 if Karina is 40 years old.
variable (karina_birth_year : ℕ)
variable (karina_current_age : ℕ)
variable (brother_current_age : ℕ)
variable (karina_twice_of_brother : Prop)

def karinas_brother_birth_year (karina_birth_year karina_current_age brother_current_age : ℕ) : ℕ :=
  karina_birth_year + brother_current_age

theorem brother_birth_year_1990 
  (h1 : karina_birth_year = 1970) 
  (h2 : karina_current_age = 40) 
  (h3 : karina_twice_of_brother) : 
  karinas_brother_birth_year 1970 40 20 = 1990 := 
by
  sorry

end brother_birth_year_1990_l428_428275


namespace rectangle_vertex_x_coordinate_l428_428813

theorem rectangle_vertex_x_coordinate
  (x : ℝ)
  (y1 y2 : ℝ)
  (slope : ℝ)
  (h1 : x = 1)
  (h2 : 9 = 9)
  (h3 : slope = 0.2)
  (h4 : y1 = 0)
  (h5 : y2 = 2)
  (h6 : ∀ (x : ℝ), (0.2 * x : ℝ) = 1 → x = 1) :
  x = 1 := 
by sorry

end rectangle_vertex_x_coordinate_l428_428813


namespace num_rectangles_in_5x5_grid_l428_428622

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428622


namespace exists_m_and_P_l428_428510

noncomputable def is_tight (n : ℕ) : Prop :=
  ∃ k : ℕ, (∀ p : ℕ, Prime p → p ∣ n → ∃ i : ℕ, i < k ∧ p = finset.nth (finset.primes) i)

noncomputable def is_loose (n : ℕ) : Prop := ¬ is_tight n

theorem exists_m_and_P (n : ℕ) (hn : n ≥ 2) : 
  (∃ m : ℕ, ∃ P : ℕ → ℕ, (m > 1) ∧ (gcd m n = 1) ∧ 
    (∀ k < m, ¬ n ∣ (nat.iterate P k) 0) ∧ (n ∣ (nat.iterate P m) 0)) ↔ 
  is_loose n :=
sorry

end exists_m_and_P_l428_428510


namespace last_two_nonzero_digits_95_factorial_l428_428098

def last_two_nonzero_digits_factorial (n : ℕ) : ℕ :=
  let f := λ x, if x % 10 == 0 then last_two_nonzero_digits_factorial (x / 10) else x % 100
  in f n

theorem last_two_nonzero_digits_95_factorial : last_two_nonzero_digits_factorial (nat.factorial 95) = 0 := sorry

end last_two_nonzero_digits_95_factorial_l428_428098


namespace trajectory_segments_l428_428573

noncomputable def function_range : set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

theorem trajectory_segments (a b : ℝ) (h : ∀ x ∈ set.Icc a b, (x^2 - 2*x) ∈ function_range) : 
  trajectory_represents_segments_AB_CD :=
sorry

end trajectory_segments_l428_428573


namespace rectangles_in_grid_l428_428666

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428666


namespace count_distribution_methods_l428_428504

-- Defining the conditions
def numStudents : ℕ := 5
def numUniversities : ℕ := 3
def universities := {Peking, ShanghaiJiaoTong, Tsinghua : String}

-- Define the distribution function
def distributionMethods (students universities : ℕ) :=
  if students <= 0 ∨ universities <= 0 then 0 else
    let distribute_group_221 := (Nat.C (students) 2) * (Nat.C (students - 2) 2) * (Nat.fact universities)
    let distribute_group_311 := (Nat.C (students) 3) * (Nat.C (students - 3) 1) * (Nat.fact universities)
    distribute_group_221 / 2 + distribute_group_311 / 2

-- The main theorem
theorem count_distribution_methods : distributionMethods numStudents numUniversities = 150 := by
  sorry

end count_distribution_methods_l428_428504


namespace square_adjacent_corners_area_l428_428768

-- Define the points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (5, 6)

-- Distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Calculate the area of the square given distance between adjacent corners
def square_area (d : ℝ) : ℝ :=
  d^2

-- Prove that the area of the square when A and B are adjacent corners is 32
theorem square_adjacent_corners_area :
  A = (1, 2) →
  B = (5, 6) →
  (∃ d, distance A B = d ∧ square_area d = 32) :=
  by
    sorry

end square_adjacent_corners_area_l428_428768


namespace locus_of_perpendicular_foot_l428_428181

theorem locus_of_perpendicular_foot (A B C : Point) (hA : acute_angle A B C) (hB : acute_angle B C A) (hC : acute_angle C A B) :
  ∃ hexagon ℋ, ∀ P : Point, (lateral_faces_acute P A B C) → (projection_on_plane P A B C) ∈ ℋ :=
sorry

def acute_angle (x y z : Point) : Prop := sorry
def lateral_faces_acute (P A B C : Point) : Prop := ∀ (Q R S : Point), has_lateral_face (Q R S) ∧ acute_angle (Q R S)
def projection_on_plane (P A B C : Point) : Point := sorry
def has_lateral_face (P A B : Point) : Prop := sorry

end locus_of_perpendicular_foot_l428_428181


namespace arithmetic_sequence_sum_l428_428350

variable (a d : ℕ)

theorem arithmetic_sequence_sum (n : ℕ) (n_eq : n = 19)
  (odd_terms_sum : ∑ i in finset.range ((n+1)/2), (a + 2*i*d) = 320)
  (every_third_term_sum : ∑ i in finset.range ((n+2)/3), (a + 3*i*d) = 224) :
  ∑ i in finset.range n, (a + i*d) = 608 := sorry

end arithmetic_sequence_sum_l428_428350


namespace watermelons_left_to_be_sold_tomorrow_l428_428913

def initial_watermelons : ℕ := 10 * 12
def sold_yesterday : ℕ := initial_watermelons * 40 / 100
def remaining_after_yesterday : ℕ := initial_watermelons - sold_yesterday
def sold_today : ℕ := remaining_after_yesterday / 4
def remaining_after_today : ℕ := remaining_after_yesterday - sold_today

theorem watermelons_left_to_be_sold_tomorrow : remaining_after_today = 54 := 
by
  sorry

end watermelons_left_to_be_sold_tomorrow_l428_428913


namespace cos_angle_FAC_l428_428945

variable (a b c : ℝ)

def vector (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 * u.1 + u.2 * u.2 + u.3 * u.3)

def cos_angle (u v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cos_angle_FAC :
  cos_angle (vector (-a) 0 (-c)) (vector 0 b (-c)) = c^2 / (Real.sqrt (a^2 + c^2) * Real.sqrt (b^2 + c^2)) :=
by
  sorry

end cos_angle_FAC_l428_428945


namespace comparison_of_negatives_l428_428131

theorem comparison_of_negatives : -(- (1 / 9)) > -| - (1 / 9) | := 
by {
  -- Solution goes here.
}

end comparison_of_negatives_l428_428131


namespace partition_quaternary_even_l428_428965

theorem partition_quaternary_even (n : ℕ) (h : 0 < n) :
  (∃ M : finset (finset ℕ), 
    M.card = n ∧
    (∀ k ∈ M, ∃ a b c d : ℕ, a ∈ k ∧ b ∈ k ∧ c ∈ k ∧ d ∈ k ∧ 
      a = (b + c + d) / 3 ∧ k = {a, b, c, d}) ∧ 
    (∪ k ∈ M, k) = finset.range (4 * n + 1)) → 
  ∃ m : ℕ, n = 2 * m :=
by
  sorry

end partition_quaternary_even_l428_428965


namespace point_Y_lies_on_median_l428_428313

-- Define the geometric points and circles
variable (A B C M Y : Point)
variable (ω1 ω2 : Circle)

-- Definitions of the given conditions
variable (P : Point) (hP : P ∈ (ω1)) (hInt : ω1 ∩ ω2 = {Y})

-- Express conditions in terms of Lean definitions
variable (hSameSide : same_side Y B (line_through A C))
variable (hMedian : M = (midpoint A C))
variable (hBM : is_median B M)

-- The theorem that we need to prove
theorem point_Y_lies_on_median :
  Y ∈ line_through B M :=
sorry

end point_Y_lies_on_median_l428_428313


namespace counterexample_9918_l428_428939

-- Checking whether a given number is a counterexample.
def isCounterexample (n : ℕ) : Prop :=
  let sumDigits := n.digits.sum
  sumDigits % 27 = 0 ∧ n % 27 ≠ 0

-- The specific number to be tested
def num : ℕ := 9918

-- Prove that 9918 is a counterexample
theorem counterexample_9918 : isCounterexample num :=
  by
    -- Proof is omitted
    sorry

end counterexample_9918_l428_428939


namespace number_of_rectangles_in_5x5_grid_l428_428656

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428656


namespace range_of_m_l428_428182

-- Definitions from conditions
def p (m : ℝ) : Prop := (∃ x y : ℝ, 2 * x^2 / m + y^2 / (m - 1) = 1)
def q (m : ℝ) : Prop := ∃ x1 : ℝ, 8 * x1^2 - 8 * m * x1 + 7 * m - 6 = 0
def proposition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬ (p m ∧ q m)

-- Proof statement
theorem range_of_m (m : ℝ) (h : proposition m) : (m ≤ 1 ∨ (3 / 2 < m ∧ m < 2)) :=
by
  sorry

end range_of_m_l428_428182


namespace fraction_in_range_l428_428037

theorem fraction_in_range : 
  (2:ℝ) / 5 < (4:ℝ) / 7 ∧ (4:ℝ) / 7 < 3 / 4 := by
  sorry

end fraction_in_range_l428_428037


namespace least_faces_combined_l428_428019

noncomputable def least_number_of_faces (c d : ℕ) : ℕ :=
c + d

theorem least_faces_combined (c d : ℕ) (h_cge8 : c ≥ 8) (h_dge8 : d ≥ 8)
  (h_sum9_prob : 8 / (c * d) = 1 / 2 * 16 / (c * d))
  (h_sum15_prob : ∃ m : ℕ, m / (c * d) = 1 / 15) :
  least_number_of_faces c d = 28 := sorry

end least_faces_combined_l428_428019


namespace count_rectangles_5x5_l428_428679

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428679


namespace calculate_blue_candles_l428_428932

-- Definitions based on identified conditions
def total_candles : Nat := 79
def yellow_candles : Nat := 27
def red_candles : Nat := 14
def blue_candles : Nat := total_candles - (yellow_candles + red_candles)

-- The proof statement
theorem calculate_blue_candles : blue_candles = 38 :=
by
  sorry

end calculate_blue_candles_l428_428932


namespace karina_brother_birth_year_l428_428276

theorem karina_brother_birth_year :
  ∀ (karina_birth_year karina_age_brother_ratio karina_current_age current_year brother_age birth_year: ℤ),
    karina_birth_year = 1970 →
    karina_age_brother_ratio = 2 →
    karina_current_age = 40 →
    current_year = karina_birth_year + karina_current_age →
    brother_age = karina_current_age / karina_age_brother_ratio →
    birth_year = current_year - brother_age →
    birth_year = 1990 :=
by
  intros karina_birth_year karina_age_brother_ratio karina_current_age current_year brother_age birth_year
  assume h1 : karina_birth_year = 1970
  assume h2 : karina_age_brother_ratio = 2
  assume h3 : karina_current_age = 40
  assume h4 : current_year = karina_birth_year + karina_current_age
  assume h5 : brother_age = karina_current_age / karina_age_brother_ratio
  assume h6 : birth_year = current_year - brother_age
  sorry

end karina_brother_birth_year_l428_428276


namespace alice_savings_l428_428083

def sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.10

theorem alice_savings :
  (basic_salary + (sales * commission_rate)) * savings_rate = 29 :=
by
  sorry

end alice_savings_l428_428083


namespace question_1_question_2_l428_428470

-- Definitions and assumptions
variables {α : Type* } [plane_geometry α]

noncomputable def triangle (A B C : α) : Prop :=
angle A B C = 90 ∧ A ≠ B

noncomputable def angle_bisector (A B C : α) (X : α) : Prop :=
line_on X A ∧ line_on X C ∧ ∀ X', X' ∈ line (A, C) → X' ≠ X → angle X' A C ≠ angle X' B C

noncomputable def altitude (A B C : α) (Y : α) : Prop :=
line_on Y A ∧ line_on Y B ∧ ∀ Y', Y' ∈ line (A, B) → Y' ≠ Y → angle Y' A C ≠ angle Y' B C

-- Questions to prove
theorem question_1 (A B C X : α) (h_triangle : triangle A B C) 
(h_bisector : angle_bisector A B C X) : 
X ≠ C → ¬(angle X A C = angle X B C) :=
sorry

theorem question_2 (A B C Y : α) (h_triangle : triangle A B C) 
(h_altitude : altitude A B C Y) :
Y ≠ C → ¬(angle Y A C = angle Y B C) := 
sorry

end question_1_question_2_l428_428470


namespace initial_chocolate_bars_l428_428869

theorem initial_chocolate_bars (B : ℕ) 
  (H1 : Thomas_and_friends_take = B / 4)
  (H2 : One_friend_returns_5 = Thomas_and_friends_take - 5)
  (H3 : Piper_takes = Thomas_and_friends_take - 5 - 5)
  (H4 : Remaining_bars = B - Thomas_and_friends_take - Piper_takes)
  (H5 : Remaining_bars = 110) :
  B = 190 := 
sorry

end initial_chocolate_bars_l428_428869


namespace BorisIsLiar_l428_428918

-- Define the predicates or types for knights and liars
inductive Tribe
| Knight : Tribe
| Liar : Tribe

-- Declare the four individuals
variable Anton Boris Vasya Grisha : Tribe

-- Conditions
axiom AntonGrishaSameTribe : Anton = Grisha
axiom BorisVasyaBothKnights : (Boris = Tribe.Knight) ∧ (Vasya = Tribe.Knight)
axiom GrishaAtMostTwoKnights : (Grisha = Tribe.Knight) → (Anton = Tribe.Knight → ∀ b v : Tribe, (b = Tribe.Knight → v = Tribe.Knight → False))

-- The goal is to prove that Boris is a liar
theorem BorisIsLiar : Boris = Tribe.Liar :=
by
  sorry

end BorisIsLiar_l428_428918


namespace distance_from_point_to_line_l428_428264

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def cartesian_distance_to_line (point : ℝ × ℝ) (y_line : ℝ) : ℝ :=
  abs (point.snd - y_line)

theorem distance_from_point_to_line
  (ρ θ : ℝ)
  (h_point : ρ = 2 ∧ θ = Real.pi / 6)
  (h_line : ∀ θ, (3 : ℝ) = ρ * Real.sin θ) :
  cartesian_distance_to_line (polar_to_cartesian ρ θ) 3 = 2 :=
  sorry

end distance_from_point_to_line_l428_428264


namespace g_42_value_l428_428753

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing (n : ℕ) (hn : n > 0) : g (n + 1) > g n
axiom g_multiplicative (m n : ℕ) (hm : m > 0) (hn : n > 0) : g (m * n) = g m * g n
axiom g_property_iii (m n : ℕ) (hm : m > 0) (hn : n > 0) : (m ≠ n ∧ m^n = n^m) → (g m = n ∨ g n = m)

theorem g_42_value : g 42 = 4410 :=
by
  sorry

end g_42_value_l428_428753


namespace equal_lengths_AK_AL_l428_428392

-- Define points and their relationships
variables (A B C K L D : Type) [EuclideanGeometry A B C K L D]
variables {triangleABC : triangle A B C}
variables {inside_triangle_K : inside_triangle A B C K}
variables {inside_triangle_L : inside_triangle A B C L}
variables {on_sideAB_D : on_side A B D}

-- Define angles based on given conditions
variables (angleAKD : angle A K D)
variables (angleBCK : angle B C K)
variables (angleALD : angle A L D)

-- Definitions of concyclic points and angle conditions
variables {concyclic_BKLC : concyclic_points B K L C}
variables {angle_condition1 : angleAKD = angleBCK}
variables {angle_condition2 : angleALD = angleBCL}

-- Theorem stating that AK = AL under given conditions
theorem equal_lengths_AK_AL 
  (h1 : inside_triangle_K)
  (h2 : inside_triangle_L)
  (h3 : on_sideAB_D)
  (h4 : concyclic_BKLC)
  (h5 : angle_condition1)
  (h6 : angle_condition2) :
  length A K = length A L := 
sorry

end equal_lengths_AK_AL_l428_428392


namespace largest_value_among_l428_428995

theorem largest_value_among (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hneq : a ≠ b) :
  max (a + b) (max (2 * Real.sqrt (a * b)) ((a^2 + b^2) / (2 * a * b))) = a + b :=
sorry

end largest_value_among_l428_428995


namespace vector_dot_product_l428_428735

variables (a b : ℝ × ℝ)
variables (h1 : a = (1, 2))
variables (h2 : a - (1 / 5) • b = (-2, 1))

theorem vector_dot_product : (a.1 * b.1 + a.2 * b.2) = 25 :=
by
  sorry

end vector_dot_product_l428_428735


namespace PB_equals_2_PD_l428_428947

theorem PB_equals_2_PD {A B C D P : Point} (h1 : D ∈ AB) 
  (h2 : AB = 4 * AD) (h3 : P ∈ circumcircle ABC) 
  (h4 : ∠ADP = ∠C) : PB = 2 * PD :=
by {
  sorry
}

end PB_equals_2_PD_l428_428947


namespace calculate_area_of_pentagon_l428_428925

noncomputable def area_of_pentagon (a b c d e : ℕ) : ℝ :=
  let triangle_area := (1/2 : ℝ) * b * a
  let trapezoid_area := (1/2 : ℝ) * (c + e) * d
  triangle_area + trapezoid_area

theorem calculate_area_of_pentagon : area_of_pentagon 18 25 28 30 25 = 1020 :=
sorry

end calculate_area_of_pentagon_l428_428925


namespace range_of_k_l428_428223

variables (k : ℝ)

def vector_a (k : ℝ) : ℝ × ℝ := (-k, 4)
def vector_b (k : ℝ) : ℝ × ℝ := (k, k + 3)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem range_of_k (h : 0 < dot_product (vector_a k) (vector_b k)) : 
  -2 < k ∧ k < 0 ∨ 0 < k ∧ k < 6 :=
sorry

end range_of_k_l428_428223


namespace num_rectangles_in_5x5_grid_l428_428605

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428605


namespace solve_for_x_l428_428347

-- Define the variables and conditions
variable (x : ℚ)

-- Define the given condition
def condition : Prop := (x + 4)/(x - 3) = (x - 2)/(x + 2)

-- State the theorem that x = -2/11 is a solution to the condition
theorem solve_for_x (h : condition x) : x = -2 / 11 := by
  sorry

end solve_for_x_l428_428347


namespace interesting_functions_count_l428_428062
open BigOperators

def interesting_function (f : ℤ → ℕ) : Prop :=
  ∀ x y : ℤ, gcd (f x) (f y) = gcd (f x) (x - y)

def count_interesting_functions : ℕ :=
  ∑ n in Finset.range (1000 + 1), n

theorem interesting_functions_count :
  ∃ (f : ℤ → ℕ), (∀ x, 1 ≤ f(x) ∧ f(x) ≤ 1000) ∧ interesting_function f ∧ count_interesting_functions = 500500 := 
by
  sorry

end interesting_functions_count_l428_428062


namespace donut_combinations_l428_428480

-- Define the problem statement where Bill needs to purchase 10 donuts,
-- with at least one of each of the 5 kinds, and calculate the combinations.

def count_donut_combinations : ℕ :=
  Nat.choose 9 4

theorem donut_combinations :
  count_donut_combinations = 126 :=
by
  -- Proof can be filled in here
  sorry

end donut_combinations_l428_428480


namespace number_of_blue_candles_l428_428935

-- Conditions
def grandfather_age : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def total_candles : ℕ := grandfather_age
def yellow_red_candles : ℕ := yellow_candles + red_candles
def blue_candles : ℕ := total_candles - yellow_red_candles

-- Proof statement
theorem number_of_blue_candles : blue_candles = 38 :=
by
  -- sorry indicates the proof is omitted
  sorry

end number_of_blue_candles_l428_428935


namespace range_of_m_l428_428157

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (hxy : (1/x) + (4/y) = 1) :
  (x + y > m^2 + 8 * m) → (-9 < m ∧ m < 1) :=
by 
  sorry

end range_of_m_l428_428157


namespace problem_solution_l428_428489

noncomputable def expression : ℝ :=
  (Real.sqrt 2 - 1)^0 - (1/3)^(-1) - Real.sqrt 8 - Real.sqrt ((-2)^2)

theorem problem_solution : expression = -4 - 2 * Real.sqrt 2 := by
  sorry

end problem_solution_l428_428489


namespace dave_probability_l428_428497

theorem dave_probability :
  let gates := 15
  let dist_between_gates := 100
  let initial_gate := (0 : Fin gates)
  let new_gate := (0 : Fin gates)
  
  let total_positions := gates * (gates - 1)
  let favorable_positions :=
    2 * (5 + 6 + 7 + 8 + 9) + 5 * 10
  
  let probability := favorable_positions / total_positions in
  let p := 4
  let q := 7 in
  probability = (p / q) := by
  sorry

end dave_probability_l428_428497


namespace set_difference_example_l428_428748

-- Define P and Q based on the given conditions
def P : Set ℝ := {x | 0 < x ∧ x < 2}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the theorem: P - Q equals to the set {x | 0 < x ≤ 1}
theorem set_difference_example : P \ Q = {x | 0 < x ∧ x ≤ 1} := 
  by
  sorry

end set_difference_example_l428_428748


namespace probability_grade_A_branch_a_probability_grade_A_branch_b_average_profit_branch_a_average_profit_branch_b_select_branch_l428_428444

def frequencies_branch_a := (40, 20, 20, 20) -- (A, B, C, D)
def frequencies_branch_b := (28, 17, 34, 21) -- (A, B, C, D)

def fees := (90, 50, 20, -50)  -- (A, B, C, D respectively)
def processing_cost_branch_a := 25
def processing_cost_branch_b := 20

theorem probability_grade_A_branch_a :
  let (fa, fb, fc, fd) := frequencies_branch_a in
  (fa : ℝ) / 100 = 0.4 := by
  sorry

theorem probability_grade_A_branch_b :
  let (fa, fb, fc, fd) := frequencies_branch_b in
  (fa : ℝ) / 100 = 0.28 := by
  sorry

theorem average_profit_branch_a :
  let (fa, fb, fc, fd) := frequencies_branch_a in
  let (qa, qb, qc, qd) := fees in
  ((qa - processing_cost_branch_a) * (fa / 100) + 
   (qb - processing_cost_branch_a) * (fb / 100) +
   (qc - processing_cost_branch_a) * (fc / 100) +
   (qd - processing_cost_branch_a) * (fd / 100) : ℝ) = 15 := by
  sorry

theorem average_profit_branch_b :
  let (fa, fb, fc, fd) := frequencies_branch_b in
  let (qa, qb, qc, qd) := fees in
  ((qa - processing_cost_branch_b) * (fa / 100) + 
   (qb - processing_cost_branch_b) * (fb / 100) +
   (qc - processing_cost_branch_b) * (fc / 100) +
   (qd - processing_cost_branch_b) * (fd / 100) : ℝ) = 10 := by
  sorry

theorem select_branch :
  let profit_a := 15 in
  let profit_b := 10 in
  profit_a > profit_b → 
  "Branch A" = "Branch A" := by
  sorry

end probability_grade_A_branch_a_probability_grade_A_branch_b_average_profit_branch_a_average_profit_branch_b_select_branch_l428_428444


namespace rectangles_in_grid_l428_428669

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428669


namespace num_rectangles_in_5x5_grid_l428_428637

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428637


namespace minimize_bribe_expenses_max_information_cost_l428_428846

-- Function that describes the offer of votes for any candidate
def vote_function (units: ℕ) : ℕ :=
  units * 1  -- Each 1 unit of money buys 1 vote

-- Conditions of the problem
constants (total_voters : ℕ) (initial_voters_ratibor : ℕ) (initial_voters_nikifor : ℕ)
          (initial_voters_neutral : ℕ) (max_units_ratibor : ℕ)
          (vote_win_threshold : ℕ) (correct_min_expenses : ℕ) (F_max : ℕ)

-- Problem conditions settings:
axiom total_voters_ax : total_voters = 25
axiom initial_voters_neutral_ax : initial_voters_neutral = 15  -- 60% of total 
axiom vote_win_threshold_ax : vote_win_threshold = 13           -- 50% of total +1
axiom max_units_ratibor_ax : max_units_ratibor = 13             -- Enough to win if opponent is honest

-- Defining that Nikifor's minimum expenses will ensure a win
def minimum_expenses_nikifor (bribed_votes: ℕ) : ℕ :=
  bribed_votes  -- Direct minimum expenses required to win

-- Defining the maximum F cost Nikifor is willing to pay for information
def max_info_cost (bribed_votes: ℕ) (F: ℕ) : Prop :=
  bribed_votes * F ≤ correct_min_expenses

-- Assertion that correct_min_expenses ensures the required win votes
theorem minimize_bribe_expenses : minimum_expenses_nikifor vote_win_threshold = correct_min_expenses := sorry

-- Assertion that F_max is correctly calculated for information win
theorem max_information_cost : max_info_cost vote_win_threshold F_max := sorry

end minimize_bribe_expenses_max_information_cost_l428_428846


namespace range_g_area_triangle_ABC_l428_428211

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * sin x * cos ((x / 2) + (Real.pi / 4))^2 - cos (2 * x)

-- Define the function g(x)
def g (x : ℝ) : ℝ := 2 * sin (2 * x - Real.pi / 3) - 1

-- Prove the range of g(x) when x ∈ [π/12, π/2]
theorem range_g : ∀ x ∈ Icc (Real.pi / 12) (Real.pi / 2), 
  -2 ≤ g x ∧ g x ≤ 1 :=
sorry

-- Define sides and angles in triangle ABC
variables {a b c : ℝ} {A B C : ℝ}

-- Given conditions for the triangle
axiom h_b : b = 2
axiom h_fA : f A = sqrt 2 - 1
axiom h_a : a = sqrt 3⁻¹ * 2 * b * sin A
axiom h_B : 0 < B ∧ B < Real.pi / 2

-- Prove the area of triangle ABC
theorem area_triangle_ABC : 
  let area := 1 / 2 * a * b * sin C
  in area = (3 + sqrt 3) / 3 :=
sorry

end range_g_area_triangle_ABC_l428_428211


namespace sum_first_10_b_eq_45_l428_428538

open Nat Log

def a (n : ℕ) : ℕ := if n > 0 then 2^(n-1) else 0
def b (n : ℕ) : ℕ := if n > 0 then (n - 1) else 0

theorem sum_first_10_b_eq_45 : (Finset.range 10).sum b = 45 :=
by sorry

end sum_first_10_b_eq_45_l428_428538


namespace part1_intersection_when_a_is_zero_part2_range_of_a_l428_428184

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x < a + 5}

-- Part (1): When a = 0, find A ∩ B
theorem part1_intersection_when_a_is_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

-- Part (2): If A ∪ B = A, find the range of values for a
theorem part2_range_of_a (a : ℝ) :
  (A ∪ B a = A) → (0 < a ∧ a ≤ 1) ∨ (6 ≤ a) :=
sorry

end part1_intersection_when_a_is_zero_part2_range_of_a_l428_428184


namespace part_I_part_II_l428_428569

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + 4 * a * x - 3

-- Part (I)
theorem part_I (a : ℝ) (h_a : a > 0) (h_roots: ∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : 
  0 < a ∧ a < 2 / 5 :=
sorry

-- Part (II)
theorem part_II (a : ℝ) (h_max : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f a x ≤ f a 2) : 
  a ≥ -1 / 3 :=
sorry

end part_I_part_II_l428_428569


namespace complete_triangles_l428_428793

noncomputable def possible_placements_count : Nat :=
  sorry

theorem complete_triangles {a b c : Nat} :
  (1 + 2 + 4 + 10 + a + b + c) = 23 →
  ∃ (count : Nat), count = 4 := 
by
  sorry

end complete_triangles_l428_428793


namespace branch_A_grade_A_probability_branch_B_grade_A_probability_branch_A_average_profit_branch_B_average_profit_choose_branch_l428_428435

theorem branch_A_grade_A_probability : 
  let total_A := 100
  let grade_A_A := 40
  (grade_A_A / total_A) = 0.4 := by
  sorry

theorem branch_B_grade_A_probability : 
  let total_B := 100
  let grade_A_B := 28
  (grade_A_B / total_B) = 0.28 := by
  sorry

theorem branch_A_average_profit :
  let freq_A_A := 0.4
  let freq_A_B := 0.2
  let freq_A_C := 0.2
  let freq_A_D := 0.2
  let process_cost_A := 25
  let profit_A := (90 - process_cost_A) * freq_A_A + (50 - process_cost_A) * freq_A_B + (20 - process_cost_A) * freq_A_C + (-50 - process_cost_A) * freq_A_D
  profit_A = 15 := by
  sorry

theorem branch_B_average_profit :
  let freq_B_A := 0.28
  let freq_B_B := 0.17
  let freq_B_C := 0.34
  let freq_B_D := 0.21
  let process_cost_B := 20
  let profit_B := (90 - process_cost_B) * freq_B_A + (50 - process_cost_B) * freq_B_B + (20 - process_cost_B) * freq_B_C + (-50 - process_cost_B) * freq_B_D
  profit_B = 10 := by
  sorry

theorem choose_branch :
  let profit_A := 15
  let profit_B := 10
  profit_A > profit_B -> "Branch A"

end branch_A_grade_A_probability_branch_B_grade_A_probability_branch_A_average_profit_branch_B_average_profit_choose_branch_l428_428435


namespace italian_clock_hand_coincidence_l428_428714

theorem italian_clock_hand_coincidence :
  let hour_hand_rotation := 1 / 24
  let minute_hand_rotation := 1
  ∃ (t : ℕ), 0 ≤ t ∧ t < 24 ∧ (t * hour_hand_rotation) % 1 = (t * minute_hand_rotation) % 1
:= sorry

end italian_clock_hand_coincidence_l428_428714


namespace veranda_width_l428_428795

theorem veranda_width (w : ℝ) (h_room : 18 * 12 = 216) (h_veranda : 136 = 136) : 
  (18 + 2*w) * (12 + 2*w) = 352 → w = 2 :=
by
  sorry

end veranda_width_l428_428795


namespace position_of_largest_rational_number_l428_428476

theorem position_of_largest_rational_number : 
  let numbers := λ n: ℕ, (sqrt (3 * n))
  let position (n : ℕ) : ℕ × ℕ :=
    if h : 1 ≤ n ∧ n <= 30 then
      let row := (n - 1) / 5 + 1
      let col := (n - 1) % 5 + 1
      (row, col)
    else
      (0, 0)
  in
  (position 27) = (6, 2) :=
by
  -- Definitions of sqrt and other necessary math operations are implied.
  sorry

end position_of_largest_rational_number_l428_428476


namespace power_mean_inequality_l428_428164

open Real

theorem power_mean_inequality 
  {n : ℕ} {a m : ℝ} {x : ℕ → ℝ} {s : ℝ}
  (hx : ∀ i, 0 < x i)
  (ha : 0 < a)
  (hm : 0 < m)
  (hs : s = ∑ i in Finset.range n, x i) :
  (∑ i in Finset.range n, (a + (1 / x i ^ m)) ^ n) 
  ≥ n * (( n / s ) ^ m + a) ^ n :=
by
  sorry

end power_mean_inequality_l428_428164


namespace minimum_5_fribees_l428_428887

theorem minimum_5_fribees (x y z : ℕ) 
  (h1 : x + y + z = 115) 
  (h2 : 3 * x + 4 * y + 5 * z = 450) : 
  z ≥ 1 :=
begin
  -- Proof goes here
  sorry
end

end minimum_5_fribees_l428_428887


namespace rectangle_count_5x5_l428_428650

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428650


namespace shorten_commercial_l428_428394

theorem shorten_commercial :
  ∀ (original_length : ℝ) (percentage_to_cut : ℝ),
  original_length = 30 →
  percentage_to_cut = 0.30 →
  original_length * percentage_to_cut = 9 :=
by
  intros original_length percentage_to_cut
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end shorten_commercial_l428_428394


namespace part1_intersection_when_a_is_zero_part2_range_of_a_l428_428185

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x < a + 5}

-- Part (1): When a = 0, find A ∩ B
theorem part1_intersection_when_a_is_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

-- Part (2): If A ∪ B = A, find the range of values for a
theorem part2_range_of_a (a : ℝ) :
  (A ∪ B a = A) → (0 < a ∧ a ≤ 1) ∨ (6 ≤ a) :=
sorry

end part1_intersection_when_a_is_zero_part2_range_of_a_l428_428185


namespace geometric_sequence_b_l428_428808

theorem geometric_sequence_b (b : ℝ) (h1 : b > 0) (h2 : 30 * (b / 30) = b) (h3 : b * (b / 30) = 9 / 4) :
  b = 3 * Real.sqrt 30 / 2 :=
by
  sorry

end geometric_sequence_b_l428_428808


namespace simplify_f_find_f_l428_428999

noncomputable def f (α : ℝ) : ℝ :=
  (sin (α - π / 2) * cos (3 * π / 2 + α) * tan (π - α)) /
  (tan (-α - π) * sin (-α - π))

theorem simplify_f (α : ℝ) (h1 : π < α ∧ α < 3 * π) : 
  f(α) = -cos α :=
by sorry

theorem find_f (α : ℝ) (h1 : π < α ∧ α < 3 * π) (h2 : cos (α - 3 * π / 2) = 1 / 5) :
  f(α) = 2 * sqrt 6 / 5 :=
by sorry

end simplify_f_find_f_l428_428999


namespace sum_of_all_possible_k_values_l428_428028

noncomputable def sum_k_with_distinct_integer_solutions : Int :=
  let is_distinct_integer_solution (p q : Int) : Bool :=
    (p ≠ q) && (3 == p * q)
  let k_values := {k | ∃ (p q : Int), is_distinct_integer_solution p q ∧ k = 3 * (p + q)}
  k_values.sum

theorem sum_of_all_possible_k_values : sum_k_with_distinct_integer_solutions = 0 := 
  sorry

end sum_of_all_possible_k_values_l428_428028


namespace distance_from_p_to_center_is_2_sqrt_10_l428_428446

-- Define the conditions
def r : ℝ := 4
def PA : ℝ := 4
def PB : ℝ := 6

-- The conjecture to prove
theorem distance_from_p_to_center_is_2_sqrt_10
  (r : ℝ) (PA : ℝ) (PB : ℝ) 
  (PA_mul_PB : PA * PB = 24) 
  (r_squared : r = 4)  : 
  ∃ d : ℝ, d = 2 * Real.sqrt 10 := 
by sorry

end distance_from_p_to_center_is_2_sqrt_10_l428_428446


namespace volunteer_scheduling_l428_428015

theorem volunteer_scheduling (days volunteers : Finset ℕ) (A B C : ℕ) :
  days = {1, 2, 3, 4, 5} ∧ 
  volunteers = {A, B, C} ∧
  (∀ a ∈ volunteers, a ∈ days) ∧ 
  (∀ p q r, p ≠ q ∧ p ≠ r ∧ q ≠ r) ∧ 
  A < B ∧ A < C
  → 20 := 
sorry

end volunteer_scheduling_l428_428015


namespace part1_part2_l428_428191

def setA (x : ℝ) : Prop := x^2 - 5 * x - 6 < 0

def setB (a x : ℝ) : Prop := 2 * a - 1 ≤ x ∧ x < a + 5

open Set

theorem part1 : 
  let a := 0
  A = {x : ℝ | -1 < x ∧ x < 6} →
  B a = {x : ℝ | -1 ≤ x ∧ x < 5} →
  {x | (setA x) ∧ (setB a x)} = {x | -1 < x ∧ x < 5} :=
by
  sorry

theorem part2 : 
  A = {x : ℝ | -1 < x ∧ x < 6} →
  (B : ℝ → Set real) →
  (∀ x, (setA x ∨ setB a x) → setA x) →
  { a : ℝ | (0 < a ∧ a ≤ 1) ∨ a ≥ 6 } :=
by
  sorry

end part1_part2_l428_428191


namespace solution_MEO_2013_P1_l428_428014

open Int

noncomputable def a : ℕ → ℕ 
| n => n^2

noncomputable def b : ℕ → ℕ 
| n => (n+1)^2 - 2

noncomputable def c : ℕ → ℕ
| n => if (∃ i, n = i^2) then
          let ⟨i, _⟩ := Nat.sqrt_eq_first_pn n
          (i+1)^2 - 1
       else
          let ⟨i, j, _⟩ := Nat.sqrt_le_cond n
          (i+1)^2 + j

theorem solution_MEO_2013_P1 :
  a 2010 = 4040100 ∧ b 2010 = 4044119 ∧ c 2010 = 2099 :=
by
  split
  . simp [a]
  . split
    . simp [b]
    . simp [c, Nat.sqrt_eq_first_pn 2010, Nat.sqrt_le_cond 2010]
  sorry

end solution_MEO_2013_P1_l428_428014


namespace find_b_value_l428_428988

theorem find_b_value
    (k1 k2 b : ℝ)
    (y1 y2 : ℝ → ℝ)
    (a n : ℝ)
    (h1 : ∀ x, y1 x = k1 / x)
    (h2 : ∀ x, y2 x = k2 * x + b)
    (intersection_A : y1 1 = 4)
    (intersection_B : y2 a = 1 ∧ y1 a = 1)
    (translated_C_y1 : y1 (-1) = n + 6)
    (translated_C_y2 : y2 1 = n)
    (k1k2_nonzero : k1 ≠ 0 ∧ k2 ≠ 0)
    (sum_k1_k2 : k1 + k2 = 0) :
  b = -6 :=
sorry

end find_b_value_l428_428988


namespace present_age_of_B_l428_428415

theorem present_age_of_B
  (A B : ℕ)
  (h1 : A = B + 5)
  (h2 : A + 30 = 2 * (B - 30)) :
  B = 95 :=
by { sorry }

end present_age_of_B_l428_428415


namespace rectangle_integral_side_l428_428743

theorem rectangle_integral_side
  (R : Type) [rectangle R]
  (R_i : ℕ → rectangle R)
  (n : ℕ)
  (h_union : ∀ r ∈ (finset.range n).image R_i, r ⊆ R)
  (h_parallel : ∀ i, 1 ≤ i → i ≤ n → is_parallel R (R_i i))
  (h_disjoint : ∀ i j, 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → i ≠ j → disjoint (interior (R_i i)) (interior (R_i j)))
  (h_integral_side : ∀ i, 1 ≤ i → i ≤ n → has_integral_side (R_i i)) :
  has_integral_side R := by
  sorry

end rectangle_integral_side_l428_428743


namespace part1_part2_l428_428530

noncomputable def z (m : ℝ) : ℂ := (m * (m - 1) / (m + 1)) + (m^2 + 2 * m - 3) * complex.I

theorem part1 (m : ℝ) : z m.im = 0 → m = 0 := by
  sorry

theorem part2 (m : ℝ) : (↑(m * (m - 1) / (m + 1)) + (m^2 + 2 * m - 3 : ℝ) + 3 : ℝ) = 0 → 
  m = 0 ∨ m = -2 + √3 ∨ m = -2 - √3 := by
  sorry

end part1_part2_l428_428530


namespace distribute_balls_into_boxes_l428_428230

theorem distribute_balls_into_boxes : 
  let n := 5
  let k := 4
  (n.choose (k - 1) + k - 1).choose (k - 1) = 56 :=
by
  sorry

end distribute_balls_into_boxes_l428_428230


namespace cat_birds_total_l428_428864

def day_birds : ℕ := 8
def night_birds : ℕ := 2 * day_birds
def total_birds : ℕ := day_birds + night_birds

theorem cat_birds_total : total_birds = 24 :=
by
  -- proof goes here
  sorry

end cat_birds_total_l428_428864


namespace num_rectangles_in_5x5_grid_l428_428638

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428638


namespace max_a_value_l428_428577

theorem max_a_value : 
  (∀ (x : ℝ), (x - 1) * x - (a - 2) * (a + 1) ≥ 1) → a ≤ 3 / 2 :=
sorry

end max_a_value_l428_428577


namespace mixed_number_product_correct_l428_428931

-- Define the mixed numbers as improper fraction
def mixed_number1 := 39 + 18 / 19
def mixed_number2 := 18 + 19 / 20

-- Define their product
def mixed_product := mixed_number1 * mixed_number2

-- Define the expected result in improper fraction form
def expected_result := 757 + 1 / 380

-- The theorem to prove
theorem mixed_number_product_correct : 
  mixed_product = expected_result := 
begin
  sorry
end

end mixed_number_product_correct_l428_428931


namespace alice_savings_l428_428086

-- Define the constants and conditions
def gadget_sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.1

-- State the theorem to be proved
theorem alice_savings : 
  let commission := gadget_sales * commission_rate in
  let total_earnings := basic_salary + commission in
  let savings := total_earnings * savings_rate in
  savings = 29 :=
by
  sorry

end alice_savings_l428_428086


namespace ellipse_line_intersection_unique_l428_428216

theorem ellipse_line_intersection_unique {m n : ℝ} (h₁ : n > m) (h₂ : m > 0) :
  (∃ (x y : ℝ), x + sqrt 2 * y = 4 * sqrt 2 ∧ m * x^2 + n * y^2 = 1) ∧ 
  ∀ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ + sqrt 2 * x₂ = 4 * sqrt 2 ∧ m * x₁^2 + n * x₂^2 = 1 ∧
     x₃ + sqrt 2 * x₄ = 4 * sqrt 2 ∧ m * x₃^2 + n * x₄^2 = 1 →
     (x₁, x₂) = (x₃, x₄)) →
  (∃ (a b : ℝ), a = 1/16 ∧ b = 1/8 ∧ m = a ∧ n = b ∧ 
    ∀ (Q : ℝ × ℝ), 
      let A := (-4, 0), B := (4, 0), O := (0, 0) in
      (snd Q = 0 → fst Q = 4) →
      (let x₁ := 4 - 8 * (snd Q)^2 / (32 + (snd Q)^2),
           y₁ := (snd Q) / 8 * (x₁ + 4),
           OQ := (4, snd Q),
           OP := (x₁, y₁) in
        OQ.1 * OP.1 + OQ.2 * OP.2 = 16)).

end ellipse_line_intersection_unique_l428_428216


namespace rectangles_in_grid_l428_428664

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428664


namespace minimum_value_of_function_on_interval_l428_428997

open set

theorem minimum_value_of_function_on_interval :
  let f (x : ℝ) := 2 * x ^ 3 - 6 * x ^ 2 + 3 in
  ∃ x ∈ (Icc (-2 : ℝ) 2), (∀ y ∈ (Icc (-2 : ℝ) 2), f x ≤ f y) ∧ f x = -37 :=
by
  let f (x : ℝ) := 2 * x ^ 3 - 6 * x ^ 2 + 3
  have h_max : (∀ x ∈ (Icc (-2 : ℝ) 2), f x ≤ 3) ∧ ∃ x ∈ (Icc (-2 : ℝ) 2), f x = 3 := sorry
  sorry

end minimum_value_of_function_on_interval_l428_428997


namespace problem_probability_six_is_largest_l428_428861

noncomputable def probability (n : ℕ) (s : finset ℕ) : ℚ :=
(s.card.choose n : ℚ)⁻¹ -- Definition of uniform probability over combinations

theorem problem_probability_six_is_largest (s : finset ℕ) (h : s = {1, 2, 3, 4, 5, 6, 7}) :
  probability 4 ({1, 2, 3, 4, 5, 6}) = 2/7 := by
  sorry

end problem_probability_six_is_largest_l428_428861


namespace sum_of_values_of_x_l428_428520

theorem sum_of_values_of_x (x : ℝ) (hx : 0 < x ∧ x < 180) :
  (sin (3 * x))^3 + (sin (4 * x))^3 = 12 * (sin (2 * x))^3 * (sin x)^3 →
  (finset.sum {45, 60, 90, 120, 135} id) = 450 :=
by
  sorry

end sum_of_values_of_x_l428_428520


namespace extreme_points_interval_l428_428004

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem extreme_points_interval (a : ℝ) :
  (∃ x ∈ (a, a + 1), deriv f x = 0) ↔ (a ∈ set.Ioo (-3 : ℝ) (-2) ∨ a ∈ set.Ioo (-1) 0) :=
by
  sorry

end extreme_points_interval_l428_428004


namespace cat_birds_total_l428_428865

def day_birds : ℕ := 8
def night_birds : ℕ := 2 * day_birds
def total_birds : ℕ := day_birds + night_birds

theorem cat_birds_total : total_birds = 24 :=
by
  -- proof goes here
  sorry

end cat_birds_total_l428_428865


namespace minimum_a_l428_428702

def f (x a : ℝ) : ℝ := x^2 - 2*x - abs (x-1-a) - abs (x-2) + 4

theorem minimum_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a = -2 :=
sorry

end minimum_a_l428_428702


namespace a_b_c_at_least_one_not_less_than_one_third_l428_428826

theorem a_b_c_at_least_one_not_less_than_one_third (a b c : ℝ) (h : a + b + c = 1) :
  ¬ (a < 1/3 ∧ b < 1/3 ∧ c < 1/3) :=
by
  sorry

end a_b_c_at_least_one_not_less_than_one_third_l428_428826


namespace cars_same_order_at_two_flags_l428_428788

theorem cars_same_order_at_two_flags :
  ∃ (F G : Fin 2011), ∃ (p q : List (Fin 10)),
  (∀ (i : Fin 10), p.head! i ∈ [F, G] ∧ q.head! i ∈ [F, G]) ∧
  (∀ (i j : Fin 10), i < j → ((∀ (k : Fin 2011), p.nth k = i) ↔ (∀ (k : Fin 2011), q.nth k = i))) :=
sorry

end cars_same_order_at_two_flags_l428_428788


namespace evaluation_expression_l428_428122

theorem evaluation_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 8 * y^x - 2 * x * y = 893 :=
by
  rw [h1, h2]
  -- Here we would perform the arithmetic steps to show the equality
  sorry

end evaluation_expression_l428_428122


namespace magnitude_of_vector_l428_428584

open Real

variables {m n : ℝ}

-- a and b are vectors
def vec_a : ℝ × ℝ := (m, 2)
def vec_b : ℝ × ℝ := (-1, n)

-- dot product condition
def dot_product_zero : Prop := vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 0

-- point on circle
def point_on_circle : Prop := m^2 + n^2 = 5

-- magnitude of the vector 2a + b
def vector_magnitude : ℝ := 
  let sum_vec := (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2)
  sqrt (sum_vec.1^2 + sum_vec.2^2)

theorem magnitude_of_vector 
  (h1 : dot_product_zero) 
  (h2 : point_on_circle) 
  (h3 : 0 < n) : 
  vector_magnitude = sqrt 34 := sorry

end magnitude_of_vector_l428_428584


namespace rotated_line_equation_l428_428217

theorem rotated_line_equation :
  ∀ (x y : ℝ), y = 3 * x + 1 → ∃ x' y' : ℝ, x' + 3 * y' - 3 = 0 :=
begin
  sorry
end

end rotated_line_equation_l428_428217


namespace water_added_correct_l428_428868

noncomputable def amount_of_water_added (V_original : ℝ) (c_original : ℝ) (c_new : ℝ) : ℝ :=
let W := (c_original * V_original - c_new * V_original) / c_new in W

theorem water_added_correct :
  let V_original := 3
  let c_original := 0.33
  let c_new := 0.2475 in
  amount_of_water_added V_original c_original c_new = 1 :=
by
  sorry

end water_added_correct_l428_428868


namespace proper_subsets_cardinality_of_intersection_l428_428994

def M : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ x + y = 2}
def N : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ x - y = 4}

theorem proper_subsets_cardinality_of_intersection :
  ∀ A B : Set (ℝ × ℝ), (M = A) → (N = B) → 
  ((A ∩ B).toFinset.card = 1 → Finset.card ((A ∩ B).toFinset.powerset.filter fun s => s ≠ (A ∩ B).toFinset) = 1)
  := by
    intros A B hM hN h_card
    sorry

end proper_subsets_cardinality_of_intersection_l428_428994


namespace correct_calculation_l428_428690

theorem correct_calculation (x : ℕ) (h1 : 21 * x = 63) : x + 40 = 43 :=
by
  -- proof steps would go here, but we skip them with 'sorry'
  sorry

end correct_calculation_l428_428690


namespace central_angle_measures_l428_428561

-- Definitions for the conditions
def perimeter_eq (r l : ℝ) : Prop := l + 2 * r = 6
def area_eq (r l : ℝ) : Prop := (1 / 2) * l * r = 2
def central_angle (r l α : ℝ) : Prop := α = l / r

-- The final proof statement
theorem central_angle_measures (r l α : ℝ) (h1 : perimeter_eq r l) (h2 : area_eq r l) :
  central_angle r l α → (α = 1 ∨ α = 4) :=
sorry

end central_angle_measures_l428_428561


namespace average_monthly_growth_rate_l428_428962

theorem average_monthly_growth_rate (p_sep p_nov : ℝ) (h1 : p_sep = 5000) (h2 : p_nov = 11250) :
  ∃ x : ℝ, (0 < x ∧ (1 + x)^2 = 2.25) :=
by
  use 0.5
  split
  -- Proof of the first part: 0 < x
  { intros
    linarith },
  -- Proof of the second part: (1 + x)^2 = 2.25
  { calc (1 + 0.5) ^ 2 = 1.5 ^ 2 : by norm_num
                    ... = 2.25    : by norm_num }
  sorry

end average_monthly_growth_rate_l428_428962


namespace BethsHighSchoolStudents_l428_428478

-- Define the variables
variables (B P : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := B = 4 * P
def condition2 : Prop := B + P = 5000

-- The theorem to be proved
theorem BethsHighSchoolStudents (h1 : condition1 B P) (h2 : condition2 B P) : B = 4000 :=
by
  -- Proof will be here
  sorry

end BethsHighSchoolStudents_l428_428478


namespace number_of_extreme_points_l428_428803

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + 3 * x^2 + 4 * x - a

theorem number_of_extreme_points (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 6 * x + 4) > 0) →
  0 = 0 :=
by
  intro h
  sorry

end number_of_extreme_points_l428_428803


namespace surface_area_circumscribed_sphere_l428_428963

theorem surface_area_circumscribed_sphere :
  let r := 3 / 4 in
  4 * π * r^2 = 9 * π / 4 :=
by
  -- We define the given radius r
  let r := 3 / 4
  -- The proof follows directly from the calculations in the given solution,
  -- but here we only state the conclusion
  sorry

end surface_area_circumscribed_sphere_l428_428963


namespace num_possible_integer_values_of_x_l428_428693

noncomputable def count_x_values : ℕ :=
  let lb := 19^2	
  let ub := 20^2 - 1	
  let result := (ub - lb + 1)
  result

theorem num_possible_integer_values_of_x : count_x_values = 39 := by
  sorry

end num_possible_integer_values_of_x_l428_428693


namespace no_such_tetrahedron_exists_l428_428957

noncomputable theory

-- Define the structure of a tetrahedron
structure Tetrahedron :=
(A B C D : Type)

-- Define what it means for a triangle to be isosceles
def is_isosceles_triangle {α : Type} [linear_ordered_field α] [metric_space α] {a b c : α} : Prop :=
dist a b = dist a c ∨ dist b a = dist b c ∨ dist c a = dist c b

-- Define the property that no two triangles are congruent
def no_two_tris_congruent (triangles : set (triangle α)) : Prop :=
∀ ⦃t1 t2 : triangle α⦄, t1 ≠ t2 → t1 ≠≅ t2

-- Lean proof problem statement: we will add sorry to skip the proof
theorem no_such_tetrahedron_exists : 
  ¬ ∃ (t : Tetrahedron), 
  (is_isosceles_triangle t.A t.B t.C ∧ is_isosceles_triangle t.A t.B t.D ∧ is_isosceles_triangle t.A t.C t.D ∧ is_isosceles_triangle t.B t.C t.D) ∧
  no_two_tris_congruent {⟨t.A, t.B, t.C⟩, ⟨t.A, t.B, t.D⟩, ⟨t.A, t.C, t.D⟩, ⟨t.B, t.C, t.D⟩} :=
sorry

end no_such_tetrahedron_exists_l428_428957


namespace radius_of_inscribed_circle_l428_428365

theorem radius_of_inscribed_circle (p : ℝ) (α : ℝ) : 
  ∃ (r : ℝ), r = (p * real.sin(α)) / 8 :=
sorry

end radius_of_inscribed_circle_l428_428365


namespace angle_of_inclination_l428_428789

theorem angle_of_inclination (θ : ℝ) : 
  (∃ m : ℝ, m = 1 ∧ tan θ = m) → θ = 45 :=
by 
  sorry

end angle_of_inclination_l428_428789


namespace non_existence_of_nonzero_complex_numbers_l428_428958

open Complex

theorem non_existence_of_nonzero_complex_numbers :
  ∀ (a b c : ℂ) (h: ℕ), 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → 
  (∀ (k l m : ℤ), |k| + |l| + |m| ≥ 1996 → |k * a + l * b + m * c| > 1 / h) → 
  false :=
by
  sorry

end non_existence_of_nonzero_complex_numbers_l428_428958


namespace projection_correct_l428_428002

open Real

noncomputable def v := (12:ℝ, -4:ℝ)
noncomputable def u1 := (0:ℝ, 4:ℝ)
noncomputable def u2 := (3:ℝ, 2:ℝ)
noncomputable def proj_v_u1 := (-12/13:ℝ, 4/13:ℝ)
noncomputable def proj_v_u2 := (21/10:ℝ, -7/10:ℝ)

axiom proj_condition : (u1.1 * v.1 + u1.2 * v.2) / (v.1 * v.1 + v.2 * v.2) * v = proj_v_u1

theorem projection_correct :
  (u2.1 * v.1 + u2.2 * v.2) / (v.1 * v.1 + v.2 * v.2) * v = proj_v_u2 := by
  sorry

end projection_correct_l428_428002


namespace is_isosceles_triangle_l428_428805

theorem is_isosceles_triangle :
  let L1 := λ x : ℝ, 3 * x + 2
  let L2 := λ x : ℝ, -3 * x + 2
  let L3 := λ x : ℝ, -2
  ∃ A B C : ℝ × ℝ, 
    A = (0, L1 0) ∧ 
    B = (-4 / 3, L3 (-4 / 3)) ∧ 
    C = (4 / 3, L3 (4 / 3)) ∧ 
    (dist A B = dist A C ∧ 
      dist A B ≠ dist B C ∧ 
      dist A C ≠ dist B C) :=
sorry

end is_isosceles_triangle_l428_428805


namespace divisor_of_1058_l428_428404

theorem divisor_of_1058 :
  ∃ (d : ℕ), (∃ (k : ℕ), 1058 = d * k) ∧ (¬ ∃ (d : ℕ), (∃ (l : ℕ), 1 < d ∧ d < 1058 ∧ 1058 = d * l)) :=
by {
  sorry
}

end divisor_of_1058_l428_428404


namespace find_d_l428_428065

noncomputable def line_intersects (d: ℝ) : Prop :=
  1 < d ∧ d < 8 ∧ 
  (let P : ℝ × ℝ := (0, d),
       S : ℝ × ℝ := (4, d - 8),
       Q : ℝ × ℝ := (d / 2, 0) in 
   d ≠ 0 ∧
   let area_ratio := (1 / 2 * (Q.1 - S.1) * (S.2 - Q.2)) / (1 / 2 * Q.1 * d) in
   area_ratio = 1 / 4)

theorem find_d : ∃ d: ℝ, line_intersects d ∧ d = 4 :=
sorry

end find_d_l428_428065


namespace sum_of_coordinates_l428_428562

theorem sum_of_coordinates (f : ℝ → ℝ) (h₁ : f 9 = 7) : 
  (let y := (1/3) * (f (3 * 3) / 3 + 3) in 3 + y = 43 / 9) := 
by
  have h₂ : f 9 = 7 := h₁
  let y := (1/3) * (7 / 3 + 3)
  show 3 + y = 43 / 9
  sorry

end sum_of_coordinates_l428_428562


namespace minimum_volume_for_safety_l428_428430

noncomputable def pressure_is_inversely_proportional_to_volume (k V : ℝ) : ℝ :=
  k / V

-- Given conditions
def k := 8000 * 3
def p (V : ℝ) := pressure_is_inversely_proportional_to_volume k V
def balloon_will_explode (V : ℝ) : Prop := p V > 40000

-- Goal: To ensure the balloon does not explode, the volume V must be at least 0.6 m^3
theorem minimum_volume_for_safety : ∀ V : ℝ, (¬ balloon_will_explode V) → V ≥ 0.6 :=
by
  intro V
  unfold balloon_will_explode p pressure_is_inversely_proportional_to_volume
  intro h
  sorry

end minimum_volume_for_safety_l428_428430


namespace lamps_at_top_is_three_l428_428728

theorem lamps_at_top_is_three
  (n : ℕ)
  (r : ℕ)
  (total_lamps : ℕ)
  (geometric_sum : ∀ (a₁ r n : ℕ), a₁ * (1 - r ^ n) / (1 - r)) :
  n = 7 → r = 2 → total_lamps = 381 → geometric_sum a₁ r 7 = 381 → a₁ = 3 :=
by
  intros hn hr h_total h_sum
  subst hn
  subst hr
  subst h_total
  have h : a₁ * (1 - 2^7) / (1 - 2) = 381, from h_sum
  -- Skipping proof with sorry
  sorry

end lamps_at_top_is_three_l428_428728


namespace eval_expr_l428_428928

theorem eval_expr : 8^3 + 8^3 + 8^3 + 8^3 - 2^6 * 2^3 = 1536 := by
  -- Proof will go here
  sorry

end eval_expr_l428_428928


namespace triple_satisfies_lcm_and_square_condition_l428_428687

theorem triple_satisfies_lcm_and_square_condition :
  ∃! (x y z : ℕ), (Nat.lcm x y = 90) ∧ (Nat.lcm x z = 720) ∧ (Nat.lcm y z = 1000)
                   ∧ (x < y) ∧ (y < z) ∧ (Nat.isSquare x ∨ Nat.isSquare y ∨ Nat.isSquare z)
                   ∧ (x = 9) ∧ (y = 100) ∧ (z = 625) :=
sorry

end triple_satisfies_lcm_and_square_condition_l428_428687


namespace baby_mice_ratio_l428_428923

theorem baby_mice_ratio (total_baby_mice : ℕ) (mice_to_robbie : ℕ) (x : ℕ) 
  (h1 : total_baby_mice = 24)
  (h2 : mice_to_robbie = 24 / 6) 
  (h3 : 24 - mice_to_robbie - 4 * x = 8) :
  (4 * x) / mice_to_robbie = 3 :=
by
  have h4 : mice_to_robbie = 4, from h2
  have h5 : 24 - 4 - 4 * x = 8, from h3
  have h6 : 20 - 4 * x = 8, from h5
  have h7 : 12 = 4 * x, from h6
  have h8 : x = 3, from eq_of_mul_eq_mul_left (by norm_num) h7
  rw [h8, nat.mul_div_cancel_left 4] 
  symmetry
  exact ratio_self 4

end baby_mice_ratio_l428_428923


namespace equal_intercepts_lines_area_two_lines_l428_428207

-- Defining the general equation of the line l with parameter a
def line_eq (a : ℝ) (x y : ℝ) : Prop := y = -(a + 1) * x + 2 - a

-- Problem statement for equal intercepts condition
theorem equal_intercepts_lines (a : ℝ) : 
  (∃ x y : ℝ, line_eq a x y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (x = y ∨ x + y = 2*a + 2)) →
  (a = 2 ∨ a = 0) → 
  (line_eq a 1 (-3) ∨ line_eq a 1 1) :=
sorry

-- Problem statement for triangle area condition
theorem area_two_lines (a : ℝ) : 
  (∃ x y : ℝ, line_eq a x y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (1 / 2 * |x| * |y| = 2)) →
  (a = 8 ∨ a = 0) → 
  (line_eq a 1 (-9) ∨ line_eq a 1 1) :=
sorry

end equal_intercepts_lines_area_two_lines_l428_428207


namespace number_of_rectangles_in_grid_l428_428594

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428594


namespace elevator_problem_l428_428384

theorem elevator_problem :
  let floors := {1, 2, 3, 4, 5, 6}
  let people := {A, B, C}
  ∃ (choices : people → floors), 
    choices A ≠ 2 ∧ 
    (∃! p, choices p = 6) ∧ 
    ∃ (n : ℕ), n = 65 :=
  sorry

end elevator_problem_l428_428384


namespace calculate_blue_candles_l428_428933

-- Definitions based on identified conditions
def total_candles : Nat := 79
def yellow_candles : Nat := 27
def red_candles : Nat := 14
def blue_candles : Nat := total_candles - (yellow_candles + red_candles)

-- The proof statement
theorem calculate_blue_candles : blue_candles = 38 :=
by
  sorry

end calculate_blue_candles_l428_428933


namespace root_quadratic_eq_l428_428234

theorem root_quadratic_eq (n m : ℝ) (h : n ≠ 0) (root_condition : n^2 + m * n + 3 * n = 0) : m + n = -3 :=
  sorry

end root_quadratic_eq_l428_428234


namespace absolute_difference_55_l428_428975

def tau (n: ℕ) : ℕ := DivisorFinset.card (Finset.filter (λ k, k ∣ n) (Finset.range (n + 1)))

noncomputable def S (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), tau k

def oddIntegersCount (n : ℕ) : ℕ := (Finset.range n).filter (λ k, S k % 2 = 1).card

def evenIntegersCount (n : ℕ) : ℕ := (Finset.range n).filter (λ k, S k % 2 = 0).card

theorem absolute_difference_55 : |oddIntegersCount 3000 - evenIntegersCount 3000| = 55 :=
sorry

end absolute_difference_55_l428_428975


namespace karina_brother_birth_year_l428_428277

theorem karina_brother_birth_year :
  ∀ (karina_birth_year karina_age_brother_ratio karina_current_age current_year brother_age birth_year: ℤ),
    karina_birth_year = 1970 →
    karina_age_brother_ratio = 2 →
    karina_current_age = 40 →
    current_year = karina_birth_year + karina_current_age →
    brother_age = karina_current_age / karina_age_brother_ratio →
    birth_year = current_year - brother_age →
    birth_year = 1990 :=
by
  intros karina_birth_year karina_age_brother_ratio karina_current_age current_year brother_age birth_year
  assume h1 : karina_birth_year = 1970
  assume h2 : karina_age_brother_ratio = 2
  assume h3 : karina_current_age = 40
  assume h4 : current_year = karina_birth_year + karina_current_age
  assume h5 : brother_age = karina_current_age / karina_age_brother_ratio
  assume h6 : birth_year = current_year - brother_age
  sorry

end karina_brother_birth_year_l428_428277


namespace angle_between_a_and_c_is_90_l428_428989

variables {G : Type*} [inner_product_space ℝ G]

-- Define non-zero vectors a, b, and c
variables (a b c : G) 

-- Add the given conditions as hypotheses
hypothesis (h1 : a ≠ 0)
hypothesis (h2 : b ≠ 0)
hypothesis (h3 : c ≠ 0)
hypothesis (h_sum : a + b + c = 0)
hypothesis (h_angle_ab : ⟪a, b⟫ = -∥a∥^2 / 2) -- cos 120° = -1/2
hypothesis (h_norm : ∥b∥ = 2 * ∥a∥)

-- State the goal is to prove that the angle between a and c is 90°
theorem angle_between_a_and_c_is_90 :
  ⟪a, c⟫ = 0 :=
sorry

end angle_between_a_and_c_is_90_l428_428989


namespace minCircles_correct_l428_428756
noncomputable def minCircles (n : ℕ) (a : Fin n → ℕ) : ℕ :=
  max (Finset.max' (Finset.image (λ i => a i) Finset.univ) (by simp)) 
      ((∑ i : Fin n, abs (a i - a ((i + 1) % n))) / 2)

theorem minCircles_correct (n : ℕ) (a : Fin n → ℕ) : 
  ∃ m : ℕ, 
    (∀ i : Fin n, ∃ S : Finset ℕ, (∀ j ∈ S, contains_circle (P i)) ∧ S.card = a i) 
    → 
    m = minCircles n a := 
sorry

end minCircles_correct_l428_428756


namespace rectangle_perimeter_gt_16_l428_428174

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
  sorry

end rectangle_perimeter_gt_16_l428_428174


namespace tax_on_clothing_is_4_percent_l428_428769

-- Define the total amount spent before taxes
def total_amount : ℝ := 100

-- Define the amount spent on clothing, food, and other items
def clothing_amount : ℝ := 0.4 * total_amount
def food_amount : ℝ := 0.3 * total_amount
def other_items_amount : ℝ := 0.3 * total_amount

-- Define the tax on other items and the total tax paid
def other_items_tax : ℝ := 0.08 * other_items_amount
def total_tax_paid : ℝ := 0.04 * total_amount

-- Define the tax paid on clothing and the percentage tax on clothing
def clothing_tax : ℝ := total_tax_paid - other_items_tax
def clothing_tax_percentage : ℝ := (clothing_tax / clothing_amount) * 100

-- Theorem stating the percentage tax on clothing is 4%
theorem tax_on_clothing_is_4_percent : clothing_tax_percentage = 4 := by
  sorry

end tax_on_clothing_is_4_percent_l428_428769


namespace compute_fraction_pow_mul_l428_428944

theorem compute_fraction_pow_mul :
  8 * (2 / 3)^4 = 128 / 81 :=
by 
  sorry

end compute_fraction_pow_mul_l428_428944


namespace roots_of_unity_in_quadratic_l428_428883

-- Definitions based on problem conditions
def root_of_unity (n : ℕ) (z : ℂ) : Prop :=
  z ^ n = 1

def is_root_of_quadratic (z : ℂ) (a b : ℤ) : Prop :=
  z ^ 2 + (a : ℂ) * z + (b : ℂ) = 0

-- Main theorem statement
theorem roots_of_unity_in_quadratic : ∃ (z : ℂ), 
  (∃ n : ℕ, root_of_unity n z) ∧ 
  (∃ (a b : ℤ), is_root_of_quadratic z a b) ∧
  (finset.card ⟨z, _⟩ = 8) := 
sorry

end roots_of_unity_in_quadratic_l428_428883


namespace triangular_array_of_coins_l428_428469

theorem triangular_array_of_coins (N : ℤ) (h : N * (N + 1) / 2 = 3003) : N = 77 :=
by
  sorry

end triangular_array_of_coins_l428_428469


namespace tan_ratio_alpha_beta_l428_428554

theorem tan_ratio_alpha_beta 
  (α β : ℝ) 
  (h1 : Real.sin (α + β) = 1 / 5) 
  (h2 : Real.sin (α - β) = 3 / 5) : 
  Real.tan α / Real.tan β = -1 :=
sorry

end tan_ratio_alpha_beta_l428_428554


namespace vec_angle_acute_l428_428585

def vec_a : ℝ × ℝ := (1, 3)
def vec_b (λ : ℝ) : ℝ × ℝ := (2 + λ, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem vec_angle_acute (λ : ℝ) (h : dot_product vec_a (vec_b λ) > 0) : λ > -5 ∧ λ ≠ -5 / 3 :=
begin
  sorry
end

end vec_angle_acute_l428_428585


namespace complex_number_implies_a_neg_one_l428_428529

theorem complex_number_implies_a_neg_one (a : ℝ) (i : ℂ) (hi : i = complex.I) :
  ((1 + a * i) * (1 + i)).im = 0 → a = -1 := 
by
  sorry

end complex_number_implies_a_neg_one_l428_428529


namespace eqn_solution_set_l428_428005

theorem eqn_solution_set :
  {x : ℝ | x ^ 2 - 1 = 0} = {-1, 1} := 
sorry

end eqn_solution_set_l428_428005


namespace Y_on_median_BM_l428_428295

-- Let circle ω1 and ω2 be defined. Point Y is an intersection of ω1 and ω2
variable (ω1 ω2 : Set (ℝ × ℝ))
variable (Y B A C M : ℝ × ℝ)

-- Assume that point Y and point B lie on the same side of line AC
variable (same_side : (Y.1 - A.1) * (C.2 - A.2) = (Y.2 - A.2) * (C.1 - A.1)
  ∧ (B.1 - A.1) * (C.2 - A.2) = (B.2 - A.2) * (C.1 - A.1))

-- Intersection of circles ω1 and ω2 at point Y
variable (intersect_Y : ω1 ∩ ω2 = {Y})

-- Definition of the median BM from point B through the midpoint M of AC
variable (BM : Set (ℝ × ℝ))
variable (midpoint_M : M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
variable (median_BM : BM = {p | ∃ t : ℝ, p = (B.1 + t * (midpoint_M.1 - B.1), B.2 + t * (midpoint_M.2 - B.2))})

-- The statement to prove is that point Y lies on the median BM
theorem Y_on_median_BM : Y ∈ BM :=
  sorry

end Y_on_median_BM_l428_428295


namespace find_sides_l428_428246

noncomputable def triangle := Type
axiom sides_opposite (A B C : triangle) (a b c : ℝ)

def given_conditions (A B C : triangle) (a b c : ℝ) (B_angle : ℝ) (S : ℝ) : Prop :=
  (c = 4 * Real.sqrt 2) ∧ 
  (B_angle = Real.pi / 4) ∧ -- 45 degrees in radians
  (S = 2)

theorem find_sides (A B C : triangle) (a b c : ℝ) (B_angle : ℝ) (S : ℝ) : 
  given_conditions A B C a b c B_angle S → a = 1 ∧ b = 5 :=
  by
    intros h
    sorry

end find_sides_l428_428246


namespace triangular_weight_60_l428_428809

/-- The weight of the triangular weight equals 60 grams given the provided conditions. --/
theorem triangular_weight_60 (C T : ℕ) (rectangular_weight : ℕ) 
  (h1 : rectangular_weight = 90)
  (h2 : 3 * C = rectangular_weight) 
  (h3 : T = 2 * C) : T = 60 :=
by {
  rw [h1, h2] at *,
  rw h3,
  simp,
}

end triangular_weight_60_l428_428809


namespace triangle_construction_l428_428116

-- Definitions of the given conditions
variables (a m_b s_c : ℝ)
-- The conditions for the construction
def condition1 : Prop := a > m_b
def condition2 : Prop := s_c > m_b / 2

-- Statement of the theorem
theorem triangle_construction (h1 : condition1 a m_b) (h2 : condition2 m_b s_c) : 
  (if h1 ∧ h2 then 2 else 1) = 2 :=
sorry

end triangle_construction_l428_428116


namespace men_in_first_group_l428_428705

variable (M : ℕ) (daily_wage : ℝ)
variable (h1 : M * 10 * daily_wage = 1200)
variable (h2 : 9 * 6 * daily_wage = 1620)
variable (dw_eq : daily_wage = 30)

theorem men_in_first_group : M = 4 :=
by sorry

end men_in_first_group_l428_428705


namespace max_partition_sum_correct_max_value_S_l428_428811

noncomputable def max_partition_sum (numbers : List ℝ) : ℝ :=
  if H : ∀ n ∈ numbers, 0 ≤ n ∧ n ≤ 1 ∧ (List.sum numbers ≤ 11.2) then
    let (A, B) := numbers.partition (fun x => List.sum A + x ≤ 8)
    if List.sum A ≤ 8 ∧ List.sum B ≤ 4 then 11.2 else 0
  else 0

theorem max_partition_sum_correct (numbers : List ℝ) (H1 : ∀ n ∈ numbers, 0 ≤ n ∧ n ≤ 1) (H2 : List.sum numbers ≤ 11.2) :
  ∃ (A B : List ℝ), List.sum A ≤ 8 ∧ List.sum B ≤ 4 ∧ A ++ B = numbers :=
begin
  sorry,
end

theorem max_value_S : ∀ (numbers : List ℝ), (∀ n ∈ numbers, 0 ≤ n ∧ n ≤ 1) ∧ (List.sum numbers ≤ 11.2) →
  ∃ (A B : List ℝ), List.sum A ≤ 8 ∧ List.sum B ≤ 4 ∧ (A ++ B = numbers) :=
  max_partition_sum_correct

end max_partition_sum_correct_max_value_S_l428_428811


namespace number_of_true_propositions_l428_428225

-- Defining types for planes and lines
axiom Plane : Type
axiom Line : Type

-- Definitions of basic predicates
axiom is_different (p₁ p₂ : Plane) : Prop
axiom is_perpendicular (l : Line) (p : Plane) : Prop
axiom is_contained (l : Line) (p : Plane) : Prop
axiom is_parallel (l : Line) (p : Plane) : Prop
axiom is_intersection (p₁ p₂ : Plane) (l : Line) : Prop

-- Conditions
axiom α β : Plane
axiom m n : Line
axiom diff_planes : is_different α β
axiom diff_lines : m ≠ n

-- Proposition definitions
def proposition_1 : Prop := is_perpendicular m α ∧ is_contained m β → α ⊥ β
def proposition_2 : Prop := is_perpendicular m n ∧ is_perpendicular m α → n ‖ α
def proposition_3 : Prop := is_intersection α β m ∧ is_parallel n m ∧ ¬ (is_contained n α ∨ is_contained n β) → (is_parallel n α ∧ is_parallel n β)
def proposition_4 : Prop := is_parallel m α ∧ α ⊥ β → m ⊥ β

-- Statement asserting the number of true propositions
theorem number_of_true_propositions : (proposition_1 ↔ True) ∧
                                      (proposition_2 ↔ False) ∧
                                      (proposition_3 ↔ True) ∧
                                      (proposition_4 ↔ False) →
                                      (1 + 1 = 2) :=
by sorry

end number_of_true_propositions_l428_428225


namespace alice_savings_l428_428090

noncomputable def commission (sales : ℝ) : ℝ := 0.02 * sales
noncomputable def totalEarnings (basic_salary commission : ℝ) : ℝ := basic_salary + commission
noncomputable def savings (total_earnings : ℝ) : ℝ := 0.10 * total_earnings

theorem alice_savings (sales basic_salary : ℝ) (commission_rate savings_rate : ℝ) :
  commission_rate = 0.02 →
  savings_rate = 0.10 →
  sales = 2500 →
  basic_salary = 240 →
  savings (totalEarnings basic_salary (commission_rate * sales)) = 29 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_savings_l428_428090


namespace find_value_of_k_l428_428119

theorem find_value_of_k (k : ℤ) : 
  (2 + 3 * k * -1/3 = -7 * 4) → k = 30 := 
by
  sorry

end find_value_of_k_l428_428119


namespace angle_B1K_B2_75_degrees_l428_428262

theorem angle_B1K_B2_75_degrees
  (ABC : Triangle)
  (acute_ABC : Triangle.Acute ABC)
  (angle_A : ABC.AngleC = 35)
  (B1 : Point)
  (C1 : Point)
  (B2 : Midpoint ABC.sideAC)
  (C2 : Midpoint ABC.sideAB)
  (alt_B1 : Altitude ABC B1)
  (alt_C1 : Altitude ABC C1)
  (intersect_at_K : ∃ K, LineThrough B1 C2 ∩ LineThrough C1 B2 = K) :
  ∠B1 K B2 = 75 := 
sorry

end angle_B1K_B2_75_degrees_l428_428262


namespace qp_square_proof_l428_428732

noncomputable def qp_square {r1 r2 d x : ℝ} (h_r1 : r1 = 10) (h_r2 : r2 = 7) (h_d : d = 15)
    (h_eq_len : ∀ Q P R, QP = PR → ∃ P, P ∈ intersection points) :=
  x^2 = 154

theorem qp_square_proof :
  qp_square (r1 := 10) (r2 := 7) (d := 15) (x := x) sorry := 
sorry

end qp_square_proof_l428_428732


namespace point_Y_on_median_BM_l428_428299

variables {A B C M Y : Type} -- Points in geometry
variables (ω1 ω2 : set Type) -- Circles defined as sets of points

-- Definitions for intersection and symmetry conditions
def intersects (ω1 ω2 : set Type) (y : Type) : Prop := y ∈ ω1 ∧ y ∈ ω2

def same_side (A B C : Type) (Y : Type) : Prop := -- geometric definition that Y and B are on the same side of line AC
  sorry

def median (B M : Type) : set Type := -- geometric construction of median BM
  sorry 

def lies_on_median (Y : Type) (B M : Type) : Prop :=
  Y ∈ median B M

theorem point_Y_on_median_BM
  (h1 : intersects ω1 ω2 Y)
  (h2 : same_side A B C Y) :
  lies_on_median Y B M :=
sorry

end point_Y_on_median_BM_l428_428299


namespace speed_conversion_l428_428077

theorem speed_conversion
  (speed_kmh : ℝ)
  (conversion_factor : ℝ)
  (approx_speed_m_s : ℝ)
  (h1 : speed_kmh = 0.8666666666666666)
  (h2 : conversion_factor = 0.27777777777778)
  (h3 : approx_speed_m_s = 0.241) :
  (speed_kmh * conversion_factor) ≈ approx_speed_m_s :=
by
  sorry

end speed_conversion_l428_428077


namespace _l428_428248

noncomputable def triangle_XYZ := {
  X Y Z : Type,
  angle_XYZ : ℝ,
  length_XY : ℝ,
  length_XZ : ℝ,
  P Q : Y X Z
}

noncomputable theorem minimum_path_Y_P_Q_Z 
  (XYZ : triangle_XYZ)
  (h_angle_XYZ : XYZ.angle_XYZ = 50) 
  (h_length_XY : XYZ.length_XY = 8)
  (h_length_XZ : XYZ.length_XZ = 14)
  (hP : XYZ.P ∈ line XYZ.Y XYZ.X)
  (hQ : XYZ.Q ∈ line XYZ.X XYZ.Z) :
  ∃ P Q : XYZ.P XYZ.Q, YP PQ QZ = 20 :=
sorry

end _l428_428248


namespace Y_lies_on_median_BM_l428_428289

variable {Ω1 Ω2 : Type}
variable {A B C M : Ω2}
variable [EuclideanGeometry Ω2]

-- Definitions coming from conditions
variable (Y : Ω2)
variable (hY1 : Y ∈ circle_omega1) (hY2 : Y ∈ circle_omega2)
variable (hSameSide : SameSide Y B (Line AC))

-- The theorem we want to prove
theorem Y_lies_on_median_BM :
  LiesOnMedian Y B M := 
  sorry

end Y_lies_on_median_BM_l428_428289


namespace identify_b_l428_428163

theorem identify_b (a b c N : ℤ) (ha : a > 1) (hb : b > 1) (hc : c > 1) (hN : N > 1)
  (h :  N ^ ((1 : ℝ) / (a : ℝ) + 1 / ((a : ℝ) * (b : ℝ)) + 1 / ((a : ℝ) * (b : ℝ) * (c : ℝ)))
    = N ^ ((25 : ℝ) / 36)) 
  : b = 3 :=
sorry

end identify_b_l428_428163


namespace Sequence_a_correct_Sum_Tn_correct_l428_428375

noncomputable def S (a : ℕ → ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := S(n) + a(n + 1)

noncomputable def a (n : ℕ) : ℝ :=
if n = 0 then 0 else 2 * 3 ^ (n - 1)

noncomputable def b (n : ℕ) : ℝ :=
if n = 0 then 0 else Real.log_base 3 (a n / 2)

noncomputable def T (n : ℕ) : ℝ :=
∑ i in Finset.range n, a i.succ * b i.succ

theorem Sequence_a_correct :
  a 1 = 2 ∧ ∀ n : ℕ, 2 * S a n = a (n + 1) - 2 ∧ a (n + 1) = 3 * a n :=
begin
  -- proof here
  sorry
end

theorem Sum_Tn_correct :
  ∀ n : ℕ, T n = (3 / 2) + (n - 3 / 2) * 3^n :=
begin
  -- proof here
  sorry
end

end Sequence_a_correct_Sum_Tn_correct_l428_428375


namespace sqrt_arithmetic_seq_and_general_formula_a_sum_of_first_n_terms_l428_428536

-- First problem
open BigOperators

/-- The sequence {a_n} satisfies the conditions: -/
def seq_a (n : ℕ) : ℕ :=
if n = 1 then 0 else seq_a (n - 1) + 2 * nat.sqrt (seq_a (n - 1) + 1) + 1

/-- Prove that {sqrt(a_n + 1)} is an arithmetic sequence and find general formula for {a_n} -/
theorem sqrt_arithmetic_seq_and_general_formula_a :
  ∀ n, seq_a n = n^2 - 1 := sorry

-- Second problem

/-- b_n definition and finding the sum of its first n terms -/
def seq_b (n : ℕ) : ℕ :=
(seq_a n * 2^n) / (n - 1)

def T (n : ℕ) : ℕ :=
∑ k in finset.range n, seq_b k

theorem sum_of_first_n_terms :
  ∀ n, T n = n * 2^(n + 1) := sorry

end sqrt_arithmetic_seq_and_general_formula_a_sum_of_first_n_terms_l428_428536


namespace intersecting_points_convex_polygon_l428_428815

theorem intersecting_points_convex_polygon {A A' : Point} 
  {A₁ A₂ ... Aₙ₋₁ A₁' A₂' ... Aₙ₋₁' : Point} 
  (i : Finₓ (n - 1))
  (X : Point) (h₁ : Line A Aₙ ∋ Aₙ) 
  (h₂ : Line A' Aₙ' ∋ Aₙ') 
  (h₃ : X ∈ (Line A Aₙ) ∩ (Line A' Aₙ')) :
  ConvexPolygon X₁ X₂ ... Xₙ₋₁ :=
sorry

end intersecting_points_convex_polygon_l428_428815


namespace initial_amount_milk_is_60_l428_428783

-- Conditions
def cost_per_litre_milk : ℝ := 20 / 1.5
def final_mixture_value_per_litre : ℝ := 32 / 3
def water_added : ℝ := 15

-- Theorem: initial amount of milk was 60 litres
theorem initial_amount_milk_is_60 (M : ℝ) 
  (h1 : 20 / 1.5 = 40 / 3)
  (h2 : 32 / 3 = (40 / 3 * M) / (M + water_added)) :
  M = 60 := by
  sorry

end initial_amount_milk_is_60_l428_428783


namespace ratio_woodwind_to_brass_l428_428720

theorem ratio_woodwind_to_brass (num_members : ℕ) (woodwind : ℕ) (brass : ℕ) (percussion : ℕ) 
  (h1 : num_members = 110) 
  (h2 : brass = 10) 
  (h3 : percussion = 4 * woodwind) 
  (h4 : num_members = percussion + woodwind + brass) : 
  (woodwind : ℤ) / (brass : ℤ) = 2 / 1 :=
by
  have h5 : percussion + woodwind + brass = 110, by rw [←h1, h4]
  have h6 : 4 * woodwind + woodwind + 10 = 110, by rw [h3, h4, h2, h1]
  have h7 : 5 * woodwind = 100, by linarith
  have h8 : woodwind = 20, by linarith
  have h9 : (woodwind : ℤ) = 20, by exact_mod_cast h8
  have h10 : (brass : ℤ) = 10, by exact_mod_cast h2
  have h11 : (woodwind : ℤ) / (brass : ℤ) = 20 / 10, by rw [h9, h10]
  norm_num at h11
  exact h11

end ratio_woodwind_to_brass_l428_428720


namespace Y_on_median_BM_l428_428308

variables {A B C M Y : Point}
variables {omega1 omega2 : Circle}

-- Definitions of points and circles intersection
def points_on_same_side (Y B : Point) (lineAC : Line) : Prop :=
-- Here should be the appropriate condition that defines when Y and B are on the same side of the line.
sorry

-- The main theorem
theorem Y_on_median_BM (h1 : Y ∈ (omega1 ∩ omega2)) 
(h2 : points_on_same_side Y B (line_through A C)) : 
lies_on Y (line_through B M) :=
sorry

end Y_on_median_BM_l428_428308


namespace count_pairs_divisible_by_five_l428_428139

theorem count_pairs_divisible_by_five 
: (∑ a in (Finset.range 81).filter (λ a, 1 ≤ a), ∑ b in (Finset.range 31).filter (λ b, 1 ≤ b), if (a * b) % 5 = 0 then 1 else 0) = 864 := 
  sorry

end count_pairs_divisible_by_five_l428_428139


namespace rectangle_count_5x5_l428_428642

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428642


namespace alex_average_speed_l428_428912

def total_distance : ℕ := 48
def biking_time : ℕ := 6

theorem alex_average_speed : (total_distance / biking_time) = 8 := 
by
  sorry

end alex_average_speed_l428_428912


namespace general_term_formula_l428_428580

theorem general_term_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a (n + 1) - a n = n) : ∀ n, a n = (n * (n + 1)) / 2 :=
by
  intro n
  induction n with d hd
  . rw [Nat.mul_zero, zero_add, Nat.div_zero, zero_add]
    exact h₁
  . sorry

end general_term_formula_l428_428580


namespace speed_of_third_part_l428_428072

theorem speed_of_third_part (d : ℝ) (v : ℝ)
  (h1 : 3 * d = 3.000000000000001)
  (h2 : d / 3 + d / 4 + d / v = 47/60) :
  v = 5 := by
  sorry

end speed_of_third_part_l428_428072


namespace geometric_sequence_value_l428_428718

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_value
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geo : geometric_sequence a r)
  (h_pos : ∀ n, a n > 0)
  (h_roots : ∀ (a1 a19 : ℝ), a1 = a 1 → a19 = a 19 → a1 * a19 = 16 ∧ a1 + a19 = 10) :
  a 8 * a 10 * a 12 = 64 := 
sorry

end geometric_sequence_value_l428_428718


namespace area_difference_l428_428871

/-
  Definitions of the given problem conditions.
-/
def r := 3
def s := 6

/-
  Definitions of the areas based on the given conditions.
-/
def area_circle : ℝ := π * r ^ 2
def area_triangle : ℝ := (sqrt 3 / 4) * s ^ 2

/-
  The statement we need to prove.
-/
theorem area_difference :
  (area_triangle - area_circle) = 9 * sqrt 3 - 9 * π :=
by
  sorry

end area_difference_l428_428871


namespace marbles_exchange_l428_428505

-- Define the initial number of marbles for Drew and Marcus
variables {D M x : ℕ}

-- Conditions
axiom Drew_initial (D M : ℕ) : D = M + 24
axiom Drew_after_give (D x : ℕ) : D - x = 25
axiom Marcus_after_receive (M x : ℕ) : M + x = 25

-- The goal is to prove: x = 12
theorem marbles_exchange : ∀ {D M x : ℕ}, D = M + 24 ∧ D - x = 25 ∧ M + x = 25 → x = 12 :=
by 
    sorry

end marbles_exchange_l428_428505


namespace exists_n_l428_428143

def F_n (a n : ℕ) : ℕ :=
  let q := a ^ (1 / n)
  let r := a % n
  q + r

noncomputable def largest_A : ℕ :=
  53590

theorem exists_n (a : ℕ) (h : a ≤ largest_A) :
  ∃ n1 n2 n3 n4 n5 n6 : ℕ, 
    F_n (F_n (F_n (F_n (F_n (F_n a n1) n2) n3) n4) n5) n6 = 1 := 
sorry

end exists_n_l428_428143


namespace num_rectangles_in_5x5_grid_l428_428606

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428606


namespace angle_MBN_is_60_l428_428280

-- Let A, B, C, D be points on a line such that AB = BC = CD
variables {A B C D P Q M N : Type} [is_point A] [is_point B] [is_point C] [is_point D]
variable (h1 : A ≠ B)
variable (h2 : B ≠ C)
variable (h3 : C ≠ D)
variable (h4 : AB = BC)
variable (h5 : BC = CD)

-- Points P and Q are chosen such that △CPQ is equilateral, vertices named clockwise
@[is_equilateral]
variable (h_cpq : triangle C P Q)

-- Points M and N are such that △MAP and △NQD are equilateral, vertices named clockwise
@[is_equilateral]
variable (h_map : triangle M A P)
@[is_equilateral]
variable (h_nqd : triangle N Q D)

-- Find the angle ∠MBN
noncomputable def angle_MBN : Type := sorry

theorem angle_MBN_is_60 (A B C D P Q M N : Type) [is_point A] [is_point B] [is_point C] [is_point D] 
    [is_point P] [is_point Q] [is_point M] [is_point N]
    (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ D) (h4 : AB = BC) (h5 : BC = CD)
    (h_cpq : triangle C P Q) (h_map : triangle M A P) (h_nqd : triangle N Q D) :
    angle_MBN = 60 := sorry

end angle_MBN_is_60_l428_428280


namespace apple_moves_preserves_initial_state_l428_428377

-- Define initial state
def initial_A : ℕ := 6
def initial_B : ℕ := 6
def initial_C : ℕ := 6

-- Moves 
def move1 (A B C : ℕ) := (A, B - 1, C + 1)
def move2 (A B C : ℕ) := (A - 2, B, C + 2)
def move3 (A B C : ℕ) := (A - 3, B + 3, C)
def move4 (A B C : ℕ) := (A, B - 4, C + 4)
def move5 (A B C : ℕ) := (A + 5, B, C - 5)

-- Final state after all moves
def final_state (A B C : ℕ) :=
  let (A, B, C) := move1 A B C in
  let (A, B, C) := move2 A B C in
  let (A, B, C) := move3 A B C in
  let (A, B, C) := move4 A B C in
  let (A, B, C) := move5 A B C in
  (A, B, C)

-- Prove that the final state returns to the initial state
theorem apple_moves_preserves_initial_state : 
  final_state initial_A initial_B initial_C = (6, 6, 6) :=
by {
  -- Skipping the proof
  sorry
}

end apple_moves_preserves_initial_state_l428_428377


namespace solution_count_reduction_l428_428370

theorem solution_count_reduction (a : ℝ) :
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x-a)^2 + y^2 = 1) →
  (a = -1 ∨ a = 1 ∨ a = -Real.sqrt 2 ∨ a = Real.sqrt 2) ↔
  (count_solution (x^2 - y^2 = 0 ∧ (x-a)^2 + y^2 = 1) = 2 ∨ count_solution (x^2 - y^2 = 0 ∧ (x-a)^2 + y^2 = 1) = 3) :=
begin
  sorry, -- Proof goes here
end

end solution_count_reduction_l428_428370


namespace numerator_divisible_by_p3_l428_428286

section
variable {p : ℕ} (hp : Nat.Prime p) (hpg : 5 < p)
def f_p (x : ℕ) : ℚ := ∑ k in Finset.range (p - 1) |>(fun k => (1 : ℚ) / ((p * x + (k + 1))* (p * x + (k + 1))))

theorem numerator_divisible_by_p3 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ((f_p p x hp hpg) - (f_p p y hp hpg)).num % (p^3) = 0 :=
sorry
end

end numerator_divisible_by_p3_l428_428286


namespace light_bulbs_not_broken_l428_428013

-- Define the conditions as hypotheses
theorem light_bulbs_not_broken :
  let kitchen_total := 35 in
  let kitchen_broken := (3 / 5) * kitchen_total in
  let foyer_broken := 10 in
  let foyer_total := 3 * foyer_broken in
  let living_total := 24 in
  let living_broken := (1 / 2) * living_total in
  let kitchen_not_broken := kitchen_total - kitchen_broken in
  let foyer_not_broken := foyer_total - foyer_broken in
  let living_not_broken := living_total - living_broken in
  let not_broken_total := kitchen_not_broken + foyer_not_broken + living_not_broken in
  not_broken_total = 46 :=
by
  sorry

end light_bulbs_not_broken_l428_428013


namespace cherie_sparklers_count_l428_428278

-- Conditions
def koby_boxes : ℕ := 2
def koby_sparklers_per_box : ℕ := 3
def koby_whistlers_per_box : ℕ := 5
def cherie_boxes : ℕ := 1
def cherie_whistlers : ℕ := 9
def total_fireworks : ℕ := 33

-- Total number of fireworks Koby has
def koby_total_fireworks : ℕ :=
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box)

-- Total number of fireworks Cherie has
def cherie_total_fireworks : ℕ :=
  total_fireworks - koby_total_fireworks

-- Number of sparklers in Cherie's box
def cherie_sparklers : ℕ :=
  cherie_total_fireworks - cherie_whistlers

-- Proof statement
theorem cherie_sparklers_count : cherie_sparklers = 8 := by
  sorry

end cherie_sparklers_count_l428_428278


namespace right_triangle_integers_solutions_l428_428362

theorem right_triangle_integers_solutions :
  ∃ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a^2 + b^2 = c^2 ∧ (a + b + c : ℕ) = (1 / 2 * a * b : ℚ) ∧
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 6 ∧ b = 8 ∧ c = 10)) :=
sorry

end right_triangle_integers_solutions_l428_428362


namespace initial_loss_is_11_11_percent_l428_428066

noncomputable def percentage_loss_initial (C S1 : ℝ) : ℝ :=
  ((C - S1) / C) * 100

def C : ℝ := 1 / (12 * 1.20)
def S1 : ℝ := 1 / 16

theorem initial_loss_is_11_11_percent :
  abs (percentage_loss_initial C S1 - 11.11) < 0.01 := by
  sorry

end initial_loss_is_11_11_percent_l428_428066


namespace shaded_area_fraction_is_correct_l428_428335

noncomputable def fraction_of_shaded_area (O Y A B C D E F G H : Point) 
  (is_center : is_center O A B C D E F G H) 
  (is_midpoint : is_midpoint Y C D) 
  (is_regular_octagon : is_regular_octagon A B C D E F G H): ℚ :=
  let shaded_area_fraction := (3 / 8) + (1 / 16) in
  shaded_area_fraction

theorem shaded_area_fraction_is_correct 
  (O Y A B C D E F G H : Point)
  (is_center : is_center O A B C D E F G H) 
  (is_midpoint : is_midpoint Y C D) 
  (is_regular_octagon : is_regular_octagon A B C D E F G H): 
  fraction_of_shaded_area O Y A B C D E F G H is_center is_midpoint is_regular_octagon = 7 / 16 :=
by 
  sorry

end shaded_area_fraction_is_correct_l428_428335


namespace min_S_value_l428_428180

noncomputable def S (A : Finset ℕ) : ℕ :=
  A.sum * A.card * (A.card - 1) / 2

theorem min_S_value (A : Finset ℕ) (h_card : A.card = 200)
  (h_triangle : ∀ {a b c : ℕ}, a ∈ A → b ∈ A → c ∈ A → a^2 + b^2 ≥ c^2) :
  S(A) ≥ 2279405700 :=
sorry

end min_S_value_l428_428180


namespace max_f_g_l428_428212

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)
def g (x : ℝ) : ℝ := f (x - Real.pi / 6)

theorem max_f_g (x : ℝ) : (f x + g x) ≤ Real.sqrt 3 :=
by
  sorry

end max_f_g_l428_428212


namespace num_rectangles_in_5x5_grid_l428_428641

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428641


namespace john_back_squat_increase_l428_428272

-- Definitions based on conditions
def back_squat_initial : ℝ := 200
def k : ℝ := 0.8
def j : ℝ := 0.9
def total_weight_moved : ℝ := 540

-- The variable representing the increase in back squat
variable (x : ℝ)

-- The Lean statement to prove
theorem john_back_squat_increase :
  3 * (j * k * (back_squat_initial + x)) = total_weight_moved → x = 50 := by
  sorry

end john_back_squat_increase_l428_428272


namespace number_of_rectangles_in_grid_l428_428593

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428593


namespace sphere_surface_area_l428_428990

theorem sphere_surface_area
  (S A B C O : Type)
  (dist_SA : ℝ) (dist_AB : ℝ) (dist_BC : ℝ) 
  (SA_perp_ABC : ∀ (P Q R : Type), S ≠ A ∧ A ≠ B ∧ B ≠ C → (dist_SA^2 + dist_AB^2 + dist_BC^2 = 4))
  (AB_perp_BC : ∀ (P Q R : Type), A ≠ B ∧ B ≠ C ∧ A ≠ C → (dist_SA^2 + dist_AB^2 + dist_BC^2 = 4)) :
  dist_SA = 1 →
  dist_AB = 1 →
  dist_BC = √2 →
  4 * Real.pi = 4 * Real.pi :=
by
  intros
  sorry

end sphere_surface_area_l428_428990


namespace odd_product_square_l428_428337

open Int

theorem odd_product_square (a b : ℤ) (ha : Odd a) (hb : Odd b) (h : ∃ k : ℤ, a^b * b^a = k^2) : ∃ m : ℤ, a * b = m^2 :=
by
  sorry

end odd_product_square_l428_428337


namespace coeffs_of_j_l428_428806

-- The polynomial h(x)
def h (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

-- Definition for j(x)
def j (b c d x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- The conditions on the polynomial h(x)
axiom roots_of_h_distinct : ∀ x : ℝ, h x = 0 → ∀ y : ℝ, h y = 0 → x = y

-- The target values of the coefficients
def b := 10
def c := 31
def d := 30

-- We claim that j(x) with (b, c, d) = (10, 31, 30) has roots that are the cubes of the roots of h(x)
theorem coeffs_of_j :
  ∃ (b c d : ℝ), roots_of_h_distinct →
  (∀ s : ℝ, h s = 0 → j (b) (c) (d) (s^3) = 0) →
  b = 10 ∧ c = 31 ∧ d = 30 :=
sorry

end coeffs_of_j_l428_428806


namespace max_average_speed_palindromic_journey_l428_428107

theorem max_average_speed_palindromic_journey
  (initial_odometer : ℕ)
  (final_odometer : ℕ)
  (trip_duration : ℕ)
  (max_speed : ℕ)
  (palindromic : ℕ → Prop)
  (initial_palindrome : palindromic initial_odometer)
  (final_palindrome : palindromic final_odometer)
  (max_speed_constraint : ∀ t, t ≤ trip_duration → t * max_speed ≤ final_odometer - initial_odometer)
  (trip_duration_eq : trip_duration = 5)
  (max_speed_eq : max_speed = 85)
  (initial_odometer_eq : initial_odometer = 69696)
  (final_odometer_max : final_odometer ≤ initial_odometer + max_speed * trip_duration) :
  (max_speed * (final_odometer - initial_odometer) / trip_duration : ℚ) = 82.2 :=
by sorry

end max_average_speed_palindromic_journey_l428_428107


namespace sum_of_a_values_for_integer_zeroes_l428_428972

theorem sum_of_a_values_for_integer_zeroes :
  (∑ (a : ℤ) in {a | ∃ x y : ℤ, x ≠ y ∧ g(x) = 0 ∧ g(y) = 0 ∧ a = x + y}, a) = 53 := by
  let g (x : ℤ) (a : ℤ) := x^2 - a * x + 3 * a
  sorry

end sum_of_a_values_for_integer_zeroes_l428_428972


namespace count_valid_five_digit_numbers_l428_428396

theorem count_valid_five_digit_numbers : 
  let digits := {0, 1, 2, 3, 4}
  in ∃ n : ℕ, 
    (∀ (d1 d2 d3 d4 d5 : ℕ), 
      ({d1, d2, d3, d4, d5} ⊆ digits ∧ 
      d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ 
      d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ 
      d3 ≠ d4 ∧ d3 ≠ d5 ∧
      d4 ≠ d5 ∧ 
      ((∃ i, (d1 = odd i ∧ d2 = even i ∧ d3 = odd i) ∨ 
      (d2 = odd i ∧ d3 = even i ∧ d4 = odd i) ∨ 
      (d3 = odd i ∧ d4 = even i ∧ d5 = odd i))) 
    ) = 28 :=
sorry

end count_valid_five_digit_numbers_l428_428396


namespace range_of_m_l428_428158

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (hxy : (1/x) + (4/y) = 1) :
  (x + y > m^2 + 8 * m) → (-9 < m ∧ m < 1) :=
by 
  sorry

end range_of_m_l428_428158


namespace volume_ratio_l428_428969

theorem volume_ratio (b : ℝ) : let a := b * Real.sqrt 2 in
  let V_cube := b^3 in
  let V_tetra := (a^3 * Real.sqrt 2) / 12 in
  V_cube / V_tetra = 3 :=
by
  let a := b * Real.sqrt 2
  let V_cube := b^3
  let V_tetra := (a^3 * Real.sqrt 2) / 12
  sorry

end volume_ratio_l428_428969


namespace Y_on_median_BM_l428_428294

-- Let circle ω1 and ω2 be defined. Point Y is an intersection of ω1 and ω2
variable (ω1 ω2 : Set (ℝ × ℝ))
variable (Y B A C M : ℝ × ℝ)

-- Assume that point Y and point B lie on the same side of line AC
variable (same_side : (Y.1 - A.1) * (C.2 - A.2) = (Y.2 - A.2) * (C.1 - A.1)
  ∧ (B.1 - A.1) * (C.2 - A.2) = (B.2 - A.2) * (C.1 - A.1))

-- Intersection of circles ω1 and ω2 at point Y
variable (intersect_Y : ω1 ∩ ω2 = {Y})

-- Definition of the median BM from point B through the midpoint M of AC
variable (BM : Set (ℝ × ℝ))
variable (midpoint_M : M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
variable (median_BM : BM = {p | ∃ t : ℝ, p = (B.1 + t * (midpoint_M.1 - B.1), B.2 + t * (midpoint_M.2 - B.2))})

-- The statement to prove is that point Y lies on the median BM
theorem Y_on_median_BM : Y ∈ BM :=
  sorry

end Y_on_median_BM_l428_428294


namespace apples_in_basket_l428_428710

-- Define the conditions in Lean
def four_times_as_many_apples (O A : ℕ) : Prop :=
  A = 4 * O

def emiliano_consumes (O A : ℕ) : Prop :=
  (2/3 : ℚ) * O + (2/3 : ℚ) * A = 50

-- Formulate the main proposition to prove there are 60 apples
theorem apples_in_basket (O A : ℕ) (h1 : four_times_as_many_apples O A) (h2 : emiliano_consumes O A) : A = 60 := 
by
  sorry

end apples_in_basket_l428_428710


namespace cos_theta_neg_three_fifths_l428_428528

theorem cos_theta_neg_three_fifths 
  (θ : ℝ)
  (h1 : Real.sin θ = -4 / 5)
  (h2 : Real.tan θ > 0) : 
  Real.cos θ = -3 / 5 := 
sorry

end cos_theta_neg_three_fifths_l428_428528


namespace daily_coaching_charge_l428_428344

theorem daily_coaching_charge (total_fee : ℝ) (total_days : ℕ) (daily_charge : ℝ)
  (h1 : total_fee = 11895)
  (h2 : total_days = 307)
  : daily_charge = total_fee / total_days :=
begin
  sorry,
end

end daily_coaching_charge_l428_428344


namespace processing_decision_l428_428439

-- Definitions of given conditions
def processing_fee (grade: Char) : ℤ :=
  match grade with
  | 'A' => 90
  | 'B' => 50
  | 'C' => 20
  | 'D' => -50
  | _   => 0

def processing_cost (branch: Char) : ℤ :=
  match branch with
  | 'A' => 25
  | 'B' => 20
  | _   => 0

structure FrequencyDistribution :=
  (gradeA : ℕ)
  (gradeB : ℕ)
  (gradeC : ℕ)
  (gradeD : ℕ)

def branchA_distribution : FrequencyDistribution :=
  { gradeA := 40, gradeB := 20, gradeC := 20, gradeD := 20 }

def branchB_distribution : FrequencyDistribution :=
  { gradeA := 28, gradeB := 17, gradeC := 34, gradeD := 21 }

-- Lean 4 statement for proof of questions
theorem processing_decision : 
  let profit (grade: Char) (branch: Char) := processing_fee grade - processing_cost branch
  let avg_profit (dist: FrequencyDistribution) (branch: Char) : ℤ :=
    (profit 'A' branch) * dist.gradeA / 100 +
    (profit 'B' branch) * dist.gradeB / 100 +
    (profit 'C' branch) * dist.gradeC / 100 +
    (profit 'D' branch) * dist.gradeD / 100
  (pA_branchA : Float := branchA_distribution.gradeA / 100.0) = 0.4 ∧
  (pA_branchB : Float := branchB_distribution.gradeA / 100.0) = 0.28 ∧
  avg_profit branchA_distribution 'A' = 15 ∧
  avg_profit branchB_distribution 'B' = 10 →
  avg_profit branchA_distribution 'A' > avg_profit branchB_distribution 'B'
:= by 
  sorry

end processing_decision_l428_428439


namespace part1_part2_l428_428491

-- Define the proposition p
def p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Iio 0, Log (3 * a) (a * x / (a - 2)) < 0

-- Define the proposition q
def q (a : ℝ) : Prop :=
  ∀ x, sqrt(x ^ 2 + 4 * x + 5 * a) ∈ Set.Ici 0

-- Equivalent proof problems as Lean 4 statements

-- Part (I)
theorem part1 (a : ℝ) : p(a) → (1 / 3) < a ∧ a < 2 :=
by sorry

-- Part (II)
theorem part2 (a : ℝ) : (p(a) ∨ q(a)) ∧ ¬ (p(a) ∧ q(a)) → ((a ≤ 1 / 3) ∨ (4 / 5 < a ∧ a < 2)) :=
by sorry

end part1_part2_l428_428491


namespace epidemic_control_indicator_l428_428506

-- Definitions for conditions
def avg_le_n (seq : Fin 7 → ℕ) (n : ℕ) : Prop := (∑ i, seq i) / 7 ≤ n
def stdev_le_n (seq : Fin 7 → ℕ) (n : ℕ) : Prop := 
  let mean := (∑ i, seq i) / 7 in
  let variance := (∑ i, (seq i - mean) ^ 2) / 7 in
  (variance ^ (1/2 : ℝ)) ≤ n
def range_le_n (seq : Fin 7 → ℕ) (n : ℕ) : Prop := 
  (∑ i, max (seq i) - min (seq i)) ≤ n
def mode_1 (seq : Fin 7 → ℕ) : Prop := 
  (1 ≤ ∑ i, ite (seq i = 1) 1 0)

-- Theorem statement based on the problem
theorem epidemic_control_indicator :
  ∀ seq : Fin 7 → ℕ,
    (range_le_n seq 2 ∧ avg_le_n seq 3) ∨ 
    (mode_1 seq ∧ range_le_n seq 4) ↔
    (∀ i, seq i ≤ 5) := 
by
-- providing a statement with sorry to indicate that proof is required
sorry

end epidemic_control_indicator_l428_428506


namespace train_speed_in_kmph_l428_428892

def distance := 225 -- distance in meters
def time := 9 -- time in seconds
def speed_in_mps : ℝ := distance / time -- speed in meters per second
def conversion_factor := 3.6 -- 1 meter/second = 3.6 km/hr
def speed_in_kmph := speed_in_mps * conversion_factor -- speed in km/hr

theorem train_speed_in_kmph : speed_in_kmph = 90 :=
by
  -- We declare the expected result directly as the theorem statement.
  sorry

end train_speed_in_kmph_l428_428892


namespace num_rectangles_in_5x5_grid_l428_428621

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428621


namespace permissible_alpha_alpha_range_l428_428353

noncomputable def prism_volume (a b : ℝ) (α : ℝ) : ℝ :=
  (a^2 * b / 2) * real.sqrt (real.sin (α + real.pi / 6) * real.sin (α - real.pi / 6))

theorem permissible_alpha (a b : ℝ) (α : ℝ) :
  0 < a ∧ 0 < b ∧ 30 * real.pi / 180 < α ∧ α < 150 * real.pi / 180 →
  prism_volume a b α = (a^2 * b / 2) * real.sqrt (real.sin (α + real.pi / 6) * real.sin (α - real.pi / 6)) :=
by sorry

theorem alpha_range (α : ℝ) :
  30 * real.pi / 180 < α ∧ α < 150 * real.pi / 180 ↔ (real.cos (2 * α) < 1 / 2 ∧ 2 * α > real.pi / 3 ∧ 2 * α < 5 * real.pi / 3) :=
by sorry

end permissible_alpha_alpha_range_l428_428353


namespace initial_gummy_worms_l428_428527

variable (G : ℕ)

theorem initial_gummy_worms (h : (G : ℚ) / 16 = 4) : G = 64 :=
by
  sorry

end initial_gummy_worms_l428_428527


namespace train_pass_time_l428_428468

noncomputable def time_to_pass (L: ℝ) (v_t: ℝ) (v_m: ℝ) : ℝ := 
  L / ((v_t * 1000 / 3600) + (v_m * 1000 / 3600))

theorem train_pass_time (L: ℝ) (v_t: ℝ) (v_m: ℝ) (hL: L = 250) (hv_t: v_t = 80) (hv_m: v_m = 12) :
  time_to_pass L v_t v_m ≈ 9.79 :=
by
  rw [time_to_pass, hL, hv_t, hv_m]
  norm_num
  sorry

end train_pass_time_l428_428468


namespace number_of_rectangles_in_grid_l428_428597

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428597


namespace profit_increase_l428_428825

theorem profit_increase (x y : ℝ) (a : ℝ)
  (h1 : x = (57 / 20) * y)
  (h2 : (x - y) / y = a / 100)
  (h3 : (x - 0.95 * y) / (0.95 * y) = (a + 15) / 100) :
  a = 185 := sorry

end profit_increase_l428_428825


namespace count_rectangles_5x5_l428_428683

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428683


namespace number_of_rectangles_in_5x5_grid_l428_428663

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428663


namespace quadratic_real_roots_k_leq_one_l428_428239

theorem quadratic_real_roots_k_leq_one (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ k ≤ 1 :=
by
  sorry

end quadratic_real_roots_k_leq_one_l428_428239


namespace floor_equation_solution_l428_428951

theorem floor_equation_solution (x : ℝ) :
  (⌊⌊3 * x⌋ + 1/3⌋ = ⌊x + 5⌋) ↔ (7/3 ≤ x ∧ x < 3) := 
sorry

end floor_equation_solution_l428_428951


namespace count_rectangles_5x5_l428_428681

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428681


namespace cylinder_volume_400pi_l428_428519

def rectangle := { width : ℝ := 10, height : ℝ := 16 }

noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem cylinder_volume_400pi (R : rectangle) :
  volume_cylinder (R.width / 2) R.height = 400 * π := by
  sorry

end cylinder_volume_400pi_l428_428519


namespace round_trip_time_is_72_hours_l428_428333

noncomputable def round_trip_time : ℝ :=
  let time_first_leg := 18 / 9 in
  let time_second_leg := 12 / 10 in
  let time_to_destination := time_first_leg + time_second_leg in
  let distance_to_destination := 18 + 12 in
  let return_speed := 7.5 in
  let time_return_trip := distance_to_destination / return_speed in
  time_to_destination + time_return_trip

theorem round_trip_time_is_72_hours :
  round_trip_time = 7.2 :=
sorry

end round_trip_time_is_72_hours_l428_428333


namespace sqrt_eq_sum_iff_zero_l428_428393

theorem sqrt_eq_sum_iff_zero (a b : ℝ) (h : 0 ≤ a ∧ 0 ≤ b) : 
  (sqrt (a * b) = a + b) ↔ (a = 0 ∧ b = 0) :=
by
  sorry

end sqrt_eq_sum_iff_zero_l428_428393


namespace simplify_and_evaluate_problem_l428_428779

noncomputable def problem_expression (a : ℤ) : ℚ :=
  (1 - (3 : ℚ) / (a + 1)) / ((a^2 - 4 * a + 4 : ℚ) / (a + 1))

theorem simplify_and_evaluate_problem :
  ∀ (a : ℤ), -2 ≤ a ∧ a ≤ 2 → a ≠ -1 → a ≠ 2 →
  (problem_expression a = 1 / (a - 2 : ℚ)) ∧
  (a = 0 → problem_expression a = -1 / 2) ∧
  (a = 1 → problem_expression a = -1) :=
sorry

end simplify_and_evaluate_problem_l428_428779


namespace num_rectangles_in_5x5_grid_l428_428624

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428624


namespace kilometers_driven_equal_l428_428938

theorem kilometers_driven_equal (x : ℝ) :
  (20 + 0.25 * x = 24 + 0.16 * x) → x = 44 := by
  sorry

end kilometers_driven_equal_l428_428938


namespace Isoland_license_plates_count_l428_428359

open Function

def IsolandAlphabet := {'A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'T', 'U', 'V'}

def Vowels := {'A', 'E', 'I', 'O', 'U'}
def Consonants := {'G', 'K', 'P', 'R', 'T', 'V'}

def LicensePlate (s : String) : Prop :=
  s.length = 6 ∧
  s.toList.all (λ c, c ∈ IsolandAlphabet) ∧
  (s.toList.headD ' ' ∈ Vowels) ∧
  (s.toList.getLast ' ' ∈ Consonants) ∧
  (s.toList.eraseDuplicates = s.toList) ∧
  ('S' ∉ s.toList)

theorem Isoland_license_plates_count :
  ∃ n : ℕ, n = 151200 ∧
  ∀ (s : String), LicensePlate s ↔
  (first : s.toList.headD ' ' ∈ Vowels) ∧
  (last : s.toList.getLast ' ' ∈ Consonants) ∧
  (no_repeats : s.length = s.toList.eraseDuplicates.length) ∧
  (no_s : 'S' ∉ s.toList) :=
by
  sorry

end Isoland_license_plates_count_l428_428359


namespace percentage_brand_A_l428_428411

theorem percentage_brand_A
  (A B : ℝ)
  (h1 : 0.6 * A + 0.65 * B = 0.5 * (A + B))
  : (A / (A + B)) * 100 = 60 :=
by
  sorry

end percentage_brand_A_l428_428411


namespace h_of_j_of_3_l428_428233

def h (x : ℝ) : ℝ := 4 * x + 3
def j (x : ℝ) : ℝ := (x + 2) ^ 2

theorem h_of_j_of_3 : h (j 3) = 103 := by
  sorry

end h_of_j_of_3_l428_428233


namespace original_price_of_sarees_l428_428369

theorem original_price_of_sarees (P : ℝ) (h : 0.95 * 0.80 * P = 456) : P = 600 :=
by
  sorry

end original_price_of_sarees_l428_428369


namespace monic_poly_with_root_sqrt3_sqrt5_l428_428509

theorem monic_poly_with_root_sqrt3_sqrt5 :
  ∃ P : Polynomial ℚ, P.monic ∧ P.degree = 4 ∧ P.eval (ℚ.ofReal (Real.sqrt 3 + Real.sqrt 5)) = 0 :=
sorry

end monic_poly_with_root_sqrt3_sqrt5_l428_428509


namespace graphs_intersect_once_l428_428142

variable {a b c d : ℝ}

theorem graphs_intersect_once 
(h1: ∃ x, (2 * a + 1 / (x - b)) = (2 * c + 1 / (x - d)) ∧ 
∃ y₁ y₂: ℝ, ∀ x, (2 * a + 1 / (x - b)) ≠ 2 * c + 1 / (x - d)) : 
∃ x, ((2 * b + 1 / (x - a)) = (2 * d + 1 / (x - c))) ∧ 
∃ y₁ y₂: ℝ, ∀ x, 2 * b + 1 / (x - a) ≠ 2 * d + 1 / (x - c) := 
sorry

end graphs_intersect_once_l428_428142


namespace apples_total_l428_428503

theorem apples_total
    (cecile_apples : ℕ := 15)
    (diane_apples_more : ℕ := 20) :
    (cecile_apples + (cecile_apples + diane_apples_more)) = 50 :=
by
  sorry

end apples_total_l428_428503


namespace movie_final_length_l428_428052

theorem movie_final_length (original_length : ℕ) (cut_length : ℕ) (final_length : ℕ) 
  (h1 : original_length = 60) (h2 : cut_length = 8) : 
  final_length = 52 :=
by
  sorry

end movie_final_length_l428_428052


namespace magnitude_complex_number_l428_428961

theorem magnitude_complex_number : complex.abs (1 - (5/4 : ℝ) * complex.I) = real.sqrt 41 / 4 :=
by
  sorry

end magnitude_complex_number_l428_428961


namespace inscribed_triangle_area_l428_428904

theorem inscribed_triangle_area 
  (r : ℝ) (theta : ℝ) 
  (A B C : ℝ) (arc1 arc2 arc3 : ℝ)
  (h_arc1 : arc1 = 5)
  (h_arc2 : arc2 = 7)
  (h_arc3 : arc3 = 8)
  (h_sum_arcs : arc1 + arc2 + arc3 = 2 * π * r)
  (h_theta : theta = 20)
  -- in radians: h_theta_rad : θ = (20 * π / 180)
  (h_A : A = 100)
  (h_B : B = 140)
  (h_C : C = 120) :
  let sin_A := sin (A * π / 180)
    sin_B := sin (B * π / 180)
    sin_C := sin (C * π / 180) in
  1 / 2 * (10 / π) ^ 2 * (sin_A + sin_B + sin_C) = 249.36 / π^2 := 
sorry

end inscribed_triangle_area_l428_428904


namespace lagrange_mean_value_problem_l428_428713

noncomputable def g (x : ℝ) : ℝ := (Real.log x) + x

theorem lagrange_mean_value_problem :
  ∃ c : ℝ, c ∈ set.Ioo 1 2 ∧ 
           deriv g c = Real.log 2 + 1 :=
by
  have h1 : continuous_on g (set.Icc 1 2),
  { sorry }
  have h2 : ∀ x ∈ set.Ioo 1 2, differentiable_at ℝ g x,
  { sorry }
  have h3 : ∀ x ∈ set.Ioo 1 2, deriv g x = (1 / x) + 1,
  { sorry }
  use 1 / Real.log 2
  split
  { sorry }
  { sorry }
  sorry


end lagrange_mean_value_problem_l428_428713


namespace largest_valid_seven_digit_number_l428_428255

def is_divisible_by_11_or_13 (n : ℕ) : Prop :=
  (n % 11 = 0) ∨ (n % 13 = 0)

def valid_seven_digit_number (num : ℕ) : Prop :=
  let d1 := num / 1000000 % 10 in
  let d2 := num / 100000 % 10 in
  let d3 := num / 10000 % 10 in
  let d4 := num / 1000 % 10 in
  let d5 := num / 100 % 10 in
  let d6 := num / 10 % 10 in
  let d7 := num % 10 in
  (is_divisible_by_11_or_13 (d1 * 100 + d2 * 10 + d3)) ∧
  (is_divisible_by_11_or_13 (d2 * 100 + d3 * 10 + d4)) ∧
  (is_divisible_by_11_or_13 (d3 * 100 + d4 * 10 + d5)) ∧
  (is_divisible_by_11_or_13 (d4 * 100 + d5 * 10 + d6)) ∧
  (is_divisible_by_11_or_13 (d5 * 100 + d6 * 10 + d7))

noncomputable def largest_seven_digit_number : ℕ :=
  9884737

theorem largest_valid_seven_digit_number : valid_seven_digit_number largest_seven_digit_number :=
  sorry

end largest_valid_seven_digit_number_l428_428255


namespace number_of_gardens_l428_428376

theorem number_of_gardens (pots_per_garden : ℕ) (flowers_per_pot : ℕ) (total_flowers : ℕ) :
  pots_per_garden = 544 →
  flowers_per_pot = 32 →
  total_flowers = 174080 →
  (total_flowers / (pots_per_garden * flowers_per_pot)) = 10 :=
by
  intros h_pots h_flowers h_total
  rw [h_pots, h_flowers, h_total]
  norm_num
  sorry

end number_of_gardens_l428_428376


namespace sum_gcd_lcm_60_429_l428_428032

theorem sum_gcd_lcm_60_429 : 
  let a := 60
  let b := 429
  gcd a b + lcm a b = 8583 :=
by
  -- Definitions of a and b
  let a := 60
  let b := 429
  
  -- The GCD and LCM calculations would go here
  
  -- Proof body (skipped with 'sorry')
  sorry

end sum_gcd_lcm_60_429_l428_428032


namespace interval_of_increase_f_a_half_minimum_integer_value_a_l428_428215

open Real

def f (a : ℝ) (x : ℝ) := a * x^2 - log x
def g (a : ℝ) (x : ℝ) := (1/2) * a * x^2 + x
def F (a : ℝ) (x : ℝ) := f a x - g a x

theorem interval_of_increase_f_a_half :
  {x : ℝ | 1 < x} = {x : ℝ | 1 < x ∧ ∀ y, f (1/2) y > f (1/2) x} :=
sorry

theorem minimum_integer_value_a :
  (∀ x : ℝ, F a x ≥ 1 - a * x) → a ≥ 2 :=
sorry

end interval_of_increase_f_a_half_minimum_integer_value_a_l428_428215


namespace jar_filling_fraction_l428_428385

theorem jar_filling_fraction (C1 C2 C3 W : ℝ)
  (h1 : W = (1/7) * C1)
  (h2 : W = (2/9) * C2)
  (h3 : W = (3/11) * C3)
  (h4 : C3 > C1 ∧ C3 > C2) :
  (3 * W) = (9 / 11) * C3 :=
by sorry

end jar_filling_fraction_l428_428385


namespace positive_integers_sum_digits_less_than_9000_l428_428414

theorem positive_integers_sum_digits_less_than_9000 : 
  ∃ n : ℕ, n = 47 ∧ ∀ x : ℕ, (1 ≤ x ∧ x < 9000 ∧ (Nat.digits 10 x).sum = 5) → (Nat.digits 10 x).length = n :=
sorry

end positive_integers_sum_digits_less_than_9000_l428_428414


namespace quadratic_expression_value_l428_428794

variable {x₁ x₂ : ℝ}

-- Defining the conditions
def condition1 := x₁ + x₂ = -5
def condition2 := x₁ * x₂ = 1
def quadraticEquation := ∀ x : ℝ, x^2 + 5 * x + 1 = 0

-- Stating the theorem
theorem quadratic_expression_value :
  condition1 ∧ condition2 ∧ quadraticEquation x₁ ∧ quadraticEquation x₂ →
  ( ( (x₁ * Real.sqrt 6) / (1 + x₂) ) ^ 2 + ( (x₂ * Real.sqrt 6) / (1 + x₁) ) ^ 2 = 220 ) :=
by
  intro h
  sorry

end quadratic_expression_value_l428_428794


namespace good_subsets_count_l428_428417

open Finset

def is_good_subset (S : Finset ℕ) (A : Finset ℕ) : Prop :=
  (A.filter (λ x, x % 2 = 0)).card ≥ (A.filter (λ x, x % 2 ≠ 0)).card

def count_good_subsets (S : Finset ℕ) : ℕ :=
  (powerset S).filter (is_good_subset S).card

theorem good_subsets_count :
  let S := range 11 \ {0} in
  count_good_subsets S = 637 := 
by
  sorry

end good_subsets_count_l428_428417


namespace problem_solution_l428_428526

def f (x : ℝ) : ℝ :=
if x ∈ set.Ico 0 2 then -((1/2) ^ |x - (3/2)|) else sorry

theorem problem_solution :
(∀ x : ℝ, f (x + 2) = 2 * f x) →
(∀ x : ℝ, x ∈ set.Ico 0 2 → f x = -((1/2) ^ |x - (3/2)|)) →
f (- (5/2)) = - (1/4) :=
sorry

end problem_solution_l428_428526


namespace quadrilateral_is_parallelogram_l428_428774

-- Define the quadrilateral \(ABCD\) and its sides
structure Quadrilateral :=
(A B C D : Point)
(segment_AD : Segment A D)
(segment_BC : Segment B C)

-- Define conditions: sides AD and BC are equal and parallel
def sides_equal_parallel (quad : Quadrilateral) : Prop :=
  quad.segment_AD.length = quad.segment_BC.length ∧ quad.segment_AD.parallel quad.segment_BC

-- State the theorem to prove that if AD = BC and AD || BC, then ABCD is a parallelogram
theorem quadrilateral_is_parallelogram (quad : Quadrilateral) (h : sides_equal_parallel quad) : is_parallelogram quad :=
sorry

end quadrilateral_is_parallelogram_l428_428774


namespace difference_max_min_t_l428_428316

noncomputable def difference_max_min_value (a b : ℝ) : ℝ := 7 - (-3)

theorem difference_max_min_t (a b : ℝ) (h : a^2 + b^2 - 2a - 4 = 0) : difference_max_min_value a b = 10 :=
sorry

end difference_max_min_t_l428_428316


namespace alex_ahead_in_second_race_l428_428721

variables (a b : ℝ) -- 'a' is the speed of Alex, 'b' is the speed of Blake
variables (d1 d2 : ℝ) -- 'd1' is the distance Alex runs in the first race, 'd2' is the distance Blake runs in the first race
variables (h : ℝ) -- head start for Blake in the second race

-- Conditions
def alex_speed_ratio (a b : ℝ) : Prop := a / b = 5 / 4
def blake_head_start (d2 : ℝ) : ℝ := 200 + 20 - d2

-- Define the distance Blake runs in the second race given head start
def blake_distance_second_race (a b d1 h : ℝ) : ℝ := 
  let time_alex := (200 + h) / a in 
  b * time_alex

-- Prove that Alex finishes 24 meters ahead of Blake in the second race
theorem alex_ahead_in_second_race 
  (a b : ℝ)
  (h : ℝ) 
  (ha : alex_speed_ratio a b)
  (d1 d2 : ℝ)
  (hb : blake_head_start d2 = 20) : 
  blake_distance_second_race a b d1 h = 176 :=
sorry

end alex_ahead_in_second_race_l428_428721


namespace max_total_weight_of_chocolates_l428_428345

theorem max_total_weight_of_chocolates 
  (A B C : ℕ)
  (hA : A ≤ 100)
  (hBC : B - C ≤ 100)
  (hC : C ≤ 100)
  (h_distribute : A ≤ 100 ∧ (B - C) ≤ 100)
  : (A + B = 300) :=
by 
  sorry

end max_total_weight_of_chocolates_l428_428345


namespace gift_cost_calculation_l428_428279

theorem gift_cost_calculation :
  let davemoney := 46
  let kylemoney := 3 * davemoney - 12
  let kyleexpenses := kylemoney / 3
  let kylerevised := kylemoney - kyleexpenses
  let lisamoney := kylerevised + 20
  let totalmoney := kylerevised + lisamoney
  let giftcost := totalmoney / 2
  giftcost = 94 :=
by
  unfold davemoney kylemoney kyleexpenses kylerevised lisamoney totalmoney giftcost
  sorry

end gift_cost_calculation_l428_428279


namespace sum_of_digits_5_pow_eq_2_pow_l428_428132

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_5_pow_eq_2_pow (n : ℕ) (h : sum_of_digits (5^n) = 2^n) : n = 3 :=
by
  sorry

end sum_of_digits_5_pow_eq_2_pow_l428_428132


namespace constant_sum_l428_428762

variables {E : Type*} [Ellipse E]
variables {F1 F2 M A B : E}
variables {a b : ℝ}

-- Given conditions
def is_foci (F1 F2 : E) : Prop := ...
def intersects (M F1 A : E) : Prop := ...
def intersects (M F2 B : E) : Prop := ...

-- Main theorem
theorem constant_sum :
  is_foci F1 F2 →
  intersects M F1 A →
  intersects M F2 B →
  ((|M F1| / |F1 A|) + (|M F2| / |F2 B|)) = (4 * a^2 / b^2) - 2 :=
begin
  sorry,
end

end constant_sum_l428_428762


namespace trajectory_of_Q_lines_are_perpendicular_l428_428201

universe u

variables {P Q : Type u} [EuclideanSpace ℝ 2 O] -- O is origin
def onCircle (P : EuclideanSpace ℝ 2 O) : Prop := P.x^2 + P.y^2 = 4

def projections (P : EuclideanSpace ℝ 2 O) : EuclideanSpace ℝ 2 O × EuclideanSpace ℝ 2 O :=
  let M := (P.x, 0)
  let N := (0, P.y)
  (M, N)

def OQ_relation (M N Q : EuclideanSpace ℝ 2 O) : Prop :=
  Q = ( (sqrt 3 / 2) * M.x, (1 / 2) * N.y )

def isOnEllipse (Q : EuclideanSpace ℝ 2 O) : Prop :=
  (Q.x ^ 2) / 3 + Q.y ^ 2 = 1

def line_through_P (P : EuclideanSpace ℝ 2 O) (k : ℝ) : EuclideanSpace ℝ 2 O → EuclideanSpace ℝ 2 O :=
  λ Q, Q.y = k * (Q.x - P.x) + P.y

def intersect_once (P Q : EuclideanSpace ℝ 2 O) (k : ℝ) : Prop :=
  let lineP k := λ Q : EuclideanSpace ℝ 2 O, Q.y = k * (Q.x - P.x) + P.y
  ∃ Q, (isOnEllipse Q) ∧ (lineP k Q) 

theorem trajectory_of_Q {P : EuclideanSpace ℝ 2 O}
  (hP : onCircle P)
  (M N : EuclideanSpace ℝ 2 O)
  (hproj : projections P = (M, N))
  (Q : EuclideanSpace ℝ 2 O)
  (hOQ : OQ_relation M N Q) :
  isOnEllipse Q := sorry

theorem lines_are_perpendicular {P : EuclideanSpace ℝ 2 O}
  (hP : onCircle P)
  (k1 k2 : ℝ) :
  (intersect_once P k1) ∧ (intersect_once P k2) → (k1 ∗ k2 = -1) :=
  sorry

end trajectory_of_Q_lines_are_perpendicular_l428_428201


namespace find_pairs_solution_l428_428511

theorem find_pairs_solution (x y : ℝ) :
  (x^3 + x^2 * y + x * y^2 + y^3 = 8 * (x^2 + x * y + y^2 + 1)) ↔ 
  (x, y) = (8, -2) ∨ (x, y) = (-2, 8) ∨ 
  (x, y) = (4 + Real.sqrt 15, 4 - Real.sqrt 15) ∨ 
  (x, y) = (4 - Real.sqrt 15, 4 + Real.sqrt 15) :=
by 
  sorry

end find_pairs_solution_l428_428511


namespace perfect_square_trinomial_k_l428_428692

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ a b : ℤ, (λ x : ℤ, (a * x + b) * (a * x + b) = 4 * x * x + k * x + 9)) → (k = 12 ∨ k = -12) :=
by
  intro ⟨a, b, h⟩
  sorry

end perfect_square_trinomial_k_l428_428692


namespace cos_alpha_eq_sqrt_1_minus_m_squared_l428_428553

noncomputable theory

variables {α β m : ℝ}

-- Conditions
axiom h1 : sin (α - β) * cos β + cos (α - β) * sin β = -m
axiom h2 : 0 < cos α -- Since α is in the 4th quadrant, 0 < cos α < 1

-- Statement of our desired proof
theorem cos_alpha_eq_sqrt_1_minus_m_squared :
  cos α = sqrt (1 - m^2) :=
sorry

end cos_alpha_eq_sqrt_1_minus_m_squared_l428_428553


namespace num_rectangles_in_5x5_grid_l428_428620

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428620


namespace final_value_of_A_l428_428126

theorem final_value_of_A : 
  ∀ (A : Int), 
    (A = 20) → 
    (A = -A + 10) → 
    A = -10 :=
by
  intros A h1 h2
  sorry

end final_value_of_A_l428_428126


namespace total_birds_caught_l428_428866

theorem total_birds_caught 
  (day_birds : ℕ) 
  (night_birds : ℕ)
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) 
  : day_birds + night_birds = 24 := 
by 
  sorry

end total_birds_caught_l428_428866


namespace exists_smallest_m_l428_428986

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ :=
  if n = 1 then 10 else 10 - 2 * n

-- Define the sequence {b_n}
def b (n : ℕ) : ℝ :=
  if n = 1 then 1 / 2 else 1 / 2 * (1 / n - 1 / (n + 1))

-- Define the partial sum T_n for the sequence {b_n}
def T (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, b (k + 1))

-- Define the condition on m
def condition (m : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → T n < m / 32

-- Prove that there exists a smallest positive integer m such that the condition holds
theorem exists_smallest_m :
  ∃ m : ℕ, m > 32 ∧ condition m :=
by
  use 32
  split
  · exact lt_add_one 32
  · sorry

end exists_smallest_m_l428_428986


namespace area_of_quadrilateral_l428_428730

theorem area_of_quadrilateral
    (outer_area : ℝ)
    (inner_area : ℝ)
    (num_quadrilaterals : ℕ)
    (h1 : outer_area = 36)
    (h2 : inner_area = 1)
    (h3 : num_quadrilaterals = 3) :
    (outer_area - inner_area) / num_quadrilaterals = 35 / 3 := by
  sorry

end area_of_quadrilateral_l428_428730


namespace number_of_elements_l428_428542

theorem number_of_elements (p : ℕ) (a : ℕ) [fact (nat.prime p)] (h_odd : p % 2 = 1) :
  let S := {xy : ℕ × ℕ | (xy.1 ^ 2 + xy.2 ^ 2) % p = a % p ∧ xy.1 < p ∧ xy.2 < p} in
  (a % p = 0 ∧ p % 4 = 1 → fintype.card S = 2 * p - 1) ∧
  (a % p = 0 ∧ p % 4 = 3 → fintype.card S = 1) ∧
  (a % p ≠ 0 ∧ p % 4 = 1 → fintype.card S = p - 1) ∧
  (a % p ≠ 0 ∧ p % 4 = 3 → fintype.card S = p + 1) := by
  sorry

end number_of_elements_l428_428542


namespace problem_statement_l428_428524

-- Define the logarithm function
def f (n : ℕ) : ℝ :=
  real.logb 1001 (n:nat)^2

theorem problem_statement : f 7 + f 11 + f 13 = 2 := by
  -- Proof goes here
  sorry

end problem_statement_l428_428524


namespace no_solution_for_floor_eq_l428_428512

theorem no_solution_for_floor_eq :
  ∀ s : ℝ, ¬ (⌊s⌋ + s = 15.6) :=
by sorry

end no_solution_for_floor_eq_l428_428512


namespace num_rectangles_in_5x5_grid_l428_428625

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428625


namespace compute_fraction_pow_mul_l428_428943

theorem compute_fraction_pow_mul :
  8 * (2 / 3)^4 = 128 / 81 :=
by 
  sorry

end compute_fraction_pow_mul_l428_428943


namespace final_elevation_proof_l428_428742

def initial_elevation : ℕ := 400
def rate_descent1 : ℕ := 10
def time_descent1 : ℕ := 5
def rate_descent2 : ℕ := 15
def time_descent2 : ℕ := 3
def rate_descent3 : ℕ := 12
def time_descent3 : ℕ := 6
def rate_ascent : ℕ := 8
def time_ascent : ℕ := 4
def rate_descent4 : ℕ := 5
def time_descent4 : ℕ := 2

theorem final_elevation_proof :
  let descent1 := rate_descent1 * time_descent1,
      descent2 := rate_descent2 * time_descent2,
      descent3 := rate_descent3 * time_descent3,
      ascent := rate_ascent * time_ascent,
      descent4 := rate_descent4 * time_descent4,
      total_descent := descent1 + descent2 + descent3 + descent4,
      net_elevation_change := total_descent - ascent,
      final_elevation := initial_elevation - net_elevation_change
  in final_elevation = 255 := by
  sorry

end final_elevation_proof_l428_428742


namespace calc_difference_l428_428946

def count_zeros (n : ℕ) : ℕ :=
  (n.digits 2).count (0 =ᶠ id)

def count_ones (n : ℕ) : ℕ :=
  (n.digits 2).count (1 =ᶠ id)

theorem calc_difference : count_ones 313 - count_zeros 313 = 3 := by
  sorry

end calc_difference_l428_428946


namespace number_of_rectangles_in_grid_l428_428589

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428589


namespace binom_identity1_binom_identity2_l428_428778

variable (n k : ℕ)

theorem binom_identity1 (hn : n > 0) (hk : k > 0) :
  (Nat.choose n k) + (Nat.choose n (k + 1)) = (Nat.choose (n + 1) (k + 1)) :=
sorry

theorem binom_identity2 (hn : n > 0) (hk : k > 0) :
  (Nat.choose n k) = (n * Nat.choose (n - 1) (k - 1)) / k :=
sorry

end binom_identity1_binom_identity2_l428_428778


namespace parabola_problem_l428_428578

theorem parabola_problem:
  ∀ (A B : ℝ × ℝ) (F : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ)
  (distance : ℝ × ℝ → ℝ × ℝ → ℝ)
  (directrix : ℝ → Prop),
  directrix A.1 →
  directrix A.2 →
  distance F (1, 0) = 0 →
  (A.1, A.2) = (-1, ±2 * √3) →
  (B.1, B.2) = (1 / 3, ±(2 √3/3)) →
  3 * (distance F A) = (distance F B) →
  distance A F = 4 :=
sorry

end parabola_problem_l428_428578


namespace num_rectangles_grid_l428_428612

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428612


namespace point_Y_lies_on_median_l428_428310

-- Define the geometric points and circles
variable (A B C M Y : Point)
variable (ω1 ω2 : Circle)

-- Definitions of the given conditions
variable (P : Point) (hP : P ∈ (ω1)) (hInt : ω1 ∩ ω2 = {Y})

-- Express conditions in terms of Lean definitions
variable (hSameSide : same_side Y B (line_through A C))
variable (hMedian : M = (midpoint A C))
variable (hBM : is_median B M)

-- The theorem that we need to prove
theorem point_Y_lies_on_median :
  Y ∈ line_through B M :=
sorry

end point_Y_lies_on_median_l428_428310


namespace point_Y_on_median_BM_l428_428302

variables {A B C M Y : Type} -- Points in geometry
variables (ω1 ω2 : set Type) -- Circles defined as sets of points

-- Definitions for intersection and symmetry conditions
def intersects (ω1 ω2 : set Type) (y : Type) : Prop := y ∈ ω1 ∧ y ∈ ω2

def same_side (A B C : Type) (Y : Type) : Prop := -- geometric definition that Y and B are on the same side of line AC
  sorry

def median (B M : Type) : set Type := -- geometric construction of median BM
  sorry 

def lies_on_median (Y : Type) (B M : Type) : Prop :=
  Y ∈ median B M

theorem point_Y_on_median_BM
  (h1 : intersects ω1 ω2 Y)
  (h2 : same_side A B C Y) :
  lies_on_median Y B M :=
sorry

end point_Y_on_median_BM_l428_428302


namespace min_small_containers_needed_l428_428068

def medium_container_capacity : ℕ := 450
def small_container_capacity : ℕ := 28

theorem min_small_containers_needed : ⌈(medium_container_capacity : ℝ) / small_container_capacity⌉ = 17 :=
by
  sorry

end min_small_containers_needed_l428_428068


namespace theater_total_seats_l428_428050

theorem theater_total_seats
  (occupied_seats : ℕ) (empty_seats : ℕ) 
  (h1 : occupied_seats = 532) (h2 : empty_seats = 218) :
  occupied_seats + empty_seats = 750 := 
by
  -- This is the placeholder for the proof
  sorry

end theater_total_seats_l428_428050


namespace roots_of_Q_l428_428750

theorem roots_of_Q (p q r : ℝ) (u : ℝ) 
  (h1 : (u + 4 * complex.I) * (u - 5 * complex.I) * (3 * u + 2) = 0) :
  p + q + r = 0 :=
sorry

end roots_of_Q_l428_428750


namespace problem_statement_l428_428539

open Classical

variable (a_n : ℕ → ℝ) (a1 d : ℝ)

-- Condition: Arithmetic sequence with first term a1 and common difference d
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ (n : ℕ), a_n (n + 1) = a1 + n * d 

-- Condition: Geometric relationship between a1, a3, and a9
def geometric_relation (a1 a3 a9 : ℝ) : Prop :=
  a3 / a1 = a9 / a3

-- Given conditions for the arithmetic sequence and geometric relation
axiom arith : arithmetic_sequence a_n a1 d
axiom geom : geometric_relation a1 (a1 + 2 * d) (a1 + 8 * d)

theorem problem_statement : d ≠ 0 → (∃ (a1 d : ℝ), d ≠ 0 ∧ arithmetic_sequence a_n a1 d ∧ geometric_relation a1 (a1 + 2 * d) (a1 + 8 * d)) → (a1 + 2 * d) / a1 = 3 := by
  sorry

end problem_statement_l428_428539


namespace value_of_a_l428_428983

def f (a x : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 + 2

def f_prime (a x : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x

theorem value_of_a (a : ℝ) (h : f_prime a (-1) = 4) : a = 10 / 3 :=
by
  -- Proof goes here
  sorry

end value_of_a_l428_428983


namespace navigationAffected_l428_428012

def parabolicArch (x y : ℝ) := y = - (1 / 25) * x ^ 2

def widthUnderBridge (h : ℝ) : ℝ := 10 * real.sqrt (4 - h)

def safePassageCondition (d : ℝ) := d ≥ 18

theorem navigationAffected (h : ℝ) :
  parabolicArch 10 (-4) ∧ parabolicArch 0 (-4) ∧ widthUnderBridge h ≥ 18 ∧ 2 + h ≥ 2.76 :=
  sorry

end navigationAffected_l428_428012


namespace probability_sin_cos_ge_one_eq_two_thirds_l428_428069

noncomputable def probability_sin_cos_ge_one (x : ℝ) : ℝ :=
  if h : 0 <= x ∧ x <= (3 * Real.pi) / 4 then
    (Real.pi / 2) / ((3 * Real.pi) / 4)
  else
    0

theorem probability_sin_cos_ge_one_eq_two_thirds :
  ∀ x : ℝ, 0 <= x ∧ x <= (3 * Real.pi) / 4 → probability_sin_cos_ge_one x = 2 / 3 :=
by
  intro x hx
  rw [probability_sin_cos_ge_one, if_pos hx]
  have h1 : (Real.pi / 2) = Real.pi * 2 / 4 := by linarith
  have h2 : (3 * Real.pi / 4) = Real.pi * 3 / 4 := rfl
  rw [h1, h2]
  ring
  sorry

end probability_sin_cos_ge_one_eq_two_thirds_l428_428069


namespace overall_average_marks_l428_428257

theorem overall_average_marks 
  (num_candidates : ℕ) 
  (num_passed : ℕ) 
  (avg_passed : ℕ) 
  (avg_failed : ℕ)
  (h1 : num_candidates = 120) 
  (h2 : num_passed = 100)
  (h3 : avg_passed = 39)
  (h4 : avg_failed = 15) :
  (num_passed * avg_passed + (num_candidates - num_passed) * avg_failed) / num_candidates = 35 := 
by
  sorry

end overall_average_marks_l428_428257


namespace usual_time_to_catch_bus_l428_428397

theorem usual_time_to_catch_bus (S T : ℝ) (h1 : S / ((5/4) * S) = (T + 5) / T) : T = 25 :=
by sorry

end usual_time_to_catch_bus_l428_428397


namespace diagonals_in_150_sided_polygon_l428_428952

/-- Define a function to calculate the number of diagonals in a polygon given the number of sides -/
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

/-- Prove that the number of diagonals in a polygon with 150 sides is 11025 -/
theorem diagonals_in_150_sided_polygon : number_of_diagonals 150 = 11025 :=
by {
  -- We substitute in 150 in the formula
  show 150 * (150 - 3) / 2 = 11025,
  -- Simplify the left-hand side
  calc
    150 * (150 - 3) / 2
    -- Simplifying step by step
    = 150 * 147 / 2 : by sorry
    = 22050 / 2 : by sorry
    = 11025 : by sorry
}

end diagonals_in_150_sided_polygon_l428_428952


namespace solve_for_t_l428_428532

noncomputable def z1 (t : ℝ) : ℂ := 2 * t + complex.I
noncomputable def z2 : ℂ := 1 - 2 * complex.I

theorem solve_for_t (t : ℝ) (h : (z1 t) / z2 ∈ set.real) : t = -1/4 :=
sorry

end solve_for_t_l428_428532


namespace area_of_rhombus_l428_428202

theorem area_of_rhombus (x y : ℝ) (d1 d2 : ℝ) (hx : x^2 + y^2 = 130) (hy : d1 = 2 * x) (hz : d2 = 2 * y) (h_diff : abs (d1 - d2) = 4) : 
  4 * 0.5 * x * y = 126 :=
by
  sorry

end area_of_rhombus_l428_428202


namespace triangle_to_isosceles_l428_428339

theorem triangle_to_isosceles (A B C : Point) (ABC : Triangle A B C) :
  ∃ (P Q R : Point) (PQR : Triangle P Q R),
    (is_partition ABC [triangle_PQ, triangle_PR, triangle_RQ]) ∧
    (isosceles_triangle PQR) :=
sorry

end triangle_to_isosceles_l428_428339


namespace proof_equivalence_l428_428320

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - (1 / 2) * x^2 - 2 * x + 5

-- Define the proposition that f(x) is increasing on certain intervals and decreasing on others
def isIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

-- Define the specific intervals of increase and decrease for our function f(x)
def intervals_of_increase_decrease (f : ℝ → ℝ) : Prop :=
  (isIncreasingOn f (-∞) (-2/3)) ∧ (isDecreasingOn f (-2/3) 1) ∧ (isIncreasingOn f 1 (∞))

-- Define the proposition that f(x) < m for x in [-1, 2] if and only if m > 7
def function_max_on_interval (f : ℝ → ℝ) (a b m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x < m) ↔ m > 7

-- Main theorem to prove both propositions
theorem proof_equivalence : intervals_of_increase_decrease f ∧ function_max_on_interval f (-1) 2 :=
by
  sorry

end proof_equivalence_l428_428320


namespace incr_circumference_area_l428_428749

theorem incr_circumference_area (d : ℝ) :
  let Δd := 2 * Real.pi in
  let ΔC := 2 * Real.pi^2 in
  let ΔA := Real.pi^2 * d + Real.pi^3 in
  (let C_orig := Real.pi * d,
       C_new := Real.pi * (d + Δd),
       A_orig := Real.pi * (d / 2)^2,
       A_new := Real.pi * ((d + Δd) / 2)^2 in
  (C_new - C_orig = ΔC) ∧ (A_new - A_orig = ΔA)) := 
by
  sorry

end incr_circumference_area_l428_428749


namespace smallest_cube_condition_l428_428954

noncomputable def smallest_perfect_cube_divisor (p q r : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] [hr : Fact (Nat.Prime r)] : ℕ :=
(p * q * r^2)^3

theorem smallest_cube_condition (p q r : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] [hr : Fact (Nat.Prime r)] :
  ∃ k, p^2 * q^3 * r^5 ∣ k ∧ (∃ t, k = t^3) ∧ (∀ m, p^2 * q^3 * r^5 ∣ m ∧ (∃ t, m = t^3) → k ≤ m) :=
begin
  use smallest_perfect_cube_divisor p q r,
  split,
  {
    sorry -- Divisibility proof
  },
  split,
  {
    use p * q * r^2,
    rw pow_succ,
    rw pow_succ,
    rw pow_succ,
    ring,
  },
  {
    intros m hm,
    sorry -- Minimality proof
  }
end

end smallest_cube_condition_l428_428954


namespace k_value_of_polynomial_square_l428_428235

theorem k_value_of_polynomial_square (k : ℤ) :
  (∃ (f : ℤ → ℤ), ∀ x, f x = x^2 + 6 * x + k^2) → (k = 3 ∨ k = -3) :=
by
  sorry

end k_value_of_polynomial_square_l428_428235


namespace blueberry_pancakes_count_l428_428499

-- Definitions of the conditions
def total_pancakes : ℕ := 67
def banana_pancakes : ℕ := 24
def plain_pancakes : ℕ := 23

-- Statement of the problem
theorem blueberry_pancakes_count :
  total_pancakes - banana_pancakes - plain_pancakes = 20 := by
  sorry

end blueberry_pancakes_count_l428_428499


namespace square_area_four_circles_l428_428464

theorem square_area_four_circles : 
  ∀ (r : ℝ), r = 5 → (let side_length := 2 * (2 * r) in side_length ^ 2) = 400 :=
by 
  intros r h
  simp [h]
  sorry

end square_area_four_circles_l428_428464


namespace domain_when_a_is_7_range_of_a_when_f_geq_3_l428_428572

noncomputable def f (x : ℝ) (a : ℝ) := Real.log (|x - 1| + |x + 2| - a) / Real.log 2

theorem domain_when_a_is_7 : 
  ∀ x : ℝ, (|x - 1| + |x + 2| - 7) > 0 ↔ (x ∈ Set.Ioo (-∞) (-4) ∪ Set.Ioo 3 ∞) := 
sorry

theorem range_of_a_when_f_geq_3 (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 3) → a ≤ -5 := 
sorry

end domain_when_a_is_7_range_of_a_when_f_geq_3_l428_428572


namespace range_of_m_l428_428556

-- Definitions based on given conditions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * x + m ≠ 0
def q (m : ℝ) : Prop := m > 1 ∧ m - 1 > 1

-- The mathematically equivalent proof problem
theorem range_of_m (m : ℝ) (hnp : ¬p m) (hapq : ¬ (p m ∧ q m)) : 1 < m ∧ m ≤ 2 :=
  by sorry

end range_of_m_l428_428556


namespace perimeter_gt_sixteen_l428_428177

theorem perimeter_gt_sixteen (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 :=
by
  sorry

end perimeter_gt_sixteen_l428_428177


namespace prime_K_n_power_of_3_l428_428764

def seqK : ℕ → ℤ
| 0        := 2 -- K1 = 2
| 1        := 8 -- K2 = 8
| (n + 2)  := 3 * seqK (n + 1) - seqK n + 5 * (-1) ^ n

theorem prime_K_n_power_of_3 (n : ℕ) (h_prime : Nat.Prime (seqK n)) : ∃ k : ℕ, n = 3 ^ k := 
sorry

end prime_K_n_power_of_3_l428_428764


namespace possible_knight_configuration_l428_428914

inductive Relationship
| friend
| enemy

def knight (n : Nat) := Fin n → Fin n → Relationship

def satisfies_conditions (n : Nat) (rel : knight n) : Prop :=
  ∃ k : knight n,
    (∀ i : Fin n, (∃ count : Nat, count = 3 ∧ (count = (Finset.filter (λ j, k i j = Relationship.enemy) (Finset.univ : Finset (Fin n))).card))) ∧
    (∀ i j : Fin n, i ≠ j ∧ k i j = Relationship.friend → 
                    ∀ e : Fin n, k i e = Relationship.enemy ∧ k j e = Relationship.enemy)

theorem possible_knight_configuration : 
  ∀ (n : Nat), satisfies_conditions n → n = 4 ∨ n = 6 :=
begin
  sorry
end

end possible_knight_configuration_l428_428914


namespace pq_sufficient_necessary_l428_428150

variable {α β : ℝ}

def p := α > β
def q := α + (Real.sin α) * (Real.cos β) > β + (Real.sin β) * (Real.cos α)

theorem pq_sufficient_necessary : p ↔ q := 
sorry

end pq_sufficient_necessary_l428_428150


namespace new_sequence_arithmetic_l428_428473

variables (a : ℕ → ℝ) (d : ℝ)

-- Define the condition of the problem that the sequence a is arithmetic with common difference d
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the new sequence b based on a
def new_sequence (a : ℕ → ℝ) : ℕ → ℝ :=
  λ n, a n + a (n + 3)

-- State the theorem that the new sequence b is arithmetic with common difference 2d
theorem new_sequence_arithmetic (a : ℕ → ℝ) (d : ℝ) (h : is_arithmetic_sequence a d) :
  is_arithmetic_sequence (new_sequence a) (2 * d) :=
sorry

end new_sequence_arithmetic_l428_428473


namespace coefficient_x3y7_in_expansion_l428_428024

theorem coefficient_x3y7_in_expansion :
  let expression := (2/3 : ℚ) * x - (1/3 : ℚ) * y,
      expansion := (expression ^ 10 : ℚ),
      term := finset.sum finset.univ (λ (k : ℕ), (nat.choose 10 k) * ((2/3 * x)^k) * ((-1/3 * y)^(10 - k))),
      target_term := (x^3 * y^7) : ℚ
  in term.coeff (x^3 * y^7) = (x^3 * y^7) * (-960/59049 : ℚ) :=
sorry

end coefficient_x3y7_in_expansion_l428_428024


namespace decision_box_two_exits_l428_428472

def termination_box := Type
def input_output_box := Type
def processing_box := Type
def decision_box := Type

def can_have_two_exits (P : Type) : Prop := -- Property indicating if a box type can have two exits.

theorem decision_box_two_exits :
  can_have_two_exits decision_box ∧ 
  ¬ can_have_two_exits termination_box ∧ 
  ¬ can_have_two_exits input_output_box ∧ 
  ¬ can_have_two_exits processing_box :=
sorry

end decision_box_two_exits_l428_428472


namespace counterexample_9918_l428_428942

theorem counterexample_9918 : 
  (sum_of_digits 9918 % 27 = 0) ∧ (9918 % 27 ≠ 0) := 
by
  sorry

end counterexample_9918_l428_428942


namespace hexagon_area_in_square_l428_428888

noncomputable def side_length : ℝ := real.sqrt 12

noncomputable def area_of_hexagon (s : ℝ) : ℝ :=
  (3 * real.sqrt 3 / 2) * (s / 2)^2

theorem hexagon_area_in_square : 
  ∃ h : ℝ, ∃ s : ℝ, s = real.sqrt 12 ∧
  h = area_of_hexagon s ∧ h = 9 * real.sqrt 3 / 2 :=
by
  use 9 * real.sqrt 3 / 2
  use real.sqrt 12
  split
  . refl
  split
  . unfold area_of_hexagon
    norm_num
  . refl
  sorry

end hexagon_area_in_square_l428_428888


namespace Y_on_median_BM_l428_428298

-- Let circle ω1 and ω2 be defined. Point Y is an intersection of ω1 and ω2
variable (ω1 ω2 : Set (ℝ × ℝ))
variable (Y B A C M : ℝ × ℝ)

-- Assume that point Y and point B lie on the same side of line AC
variable (same_side : (Y.1 - A.1) * (C.2 - A.2) = (Y.2 - A.2) * (C.1 - A.1)
  ∧ (B.1 - A.1) * (C.2 - A.2) = (B.2 - A.2) * (C.1 - A.1))

-- Intersection of circles ω1 and ω2 at point Y
variable (intersect_Y : ω1 ∩ ω2 = {Y})

-- Definition of the median BM from point B through the midpoint M of AC
variable (BM : Set (ℝ × ℝ))
variable (midpoint_M : M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
variable (median_BM : BM = {p | ∃ t : ℝ, p = (B.1 + t * (midpoint_M.1 - B.1), B.2 + t * (midpoint_M.2 - B.2))})

-- The statement to prove is that point Y lies on the median BM
theorem Y_on_median_BM : Y ∈ BM :=
  sorry

end Y_on_median_BM_l428_428298


namespace largest_angle_in_isosceles_right_triangle_is_90_l428_428819

-- Definitions for the given conditions
def is_right_triangle (ABC : Type) [Inhabited ABC] :=
  ∃ (A B C : ABC), ∠ABC = 90 ∧ is_triangle ABC

def is_isosceles_right_triangle (ABC : Type) [Inhabited ABC] :=
  ∃ (A B C : ABC), ∠ABC = 90 ∧ is_isosceles ABC ∧ is_triangle ABC

-- Problem statement as a Lean theorem
theorem largest_angle_in_isosceles_right_triangle_is_90
  (ABC : Type) [Inhabited ABC] (h1 : is_isosceles_right_triangle ABC) (h2 : ∠ABC = 45) :
  ∃ (angle : ℝ), angle = 90 :=
by
  sorry

end largest_angle_in_isosceles_right_triangle_is_90_l428_428819


namespace proof_l428_428247

noncomputable def problem : Prop :=
  let a := 1
  let b := 2
  let angleC := 60 * Real.pi / 180 -- convert degrees to radians
  let cosC := Real.cos angleC
  let sinC := Real.sin angleC
  let c_squared := a^2 + b^2 - 2 * a * b * cosC
  let c := Real.sqrt c_squared
  let area := 0.5 * a * b * sinC
  c = Real.sqrt 3 ∧ area = Real.sqrt 3 / 2

theorem proof : problem :=
by
  sorry

end proof_l428_428247


namespace hyperbola_asymptotes_and_ellipse_l428_428575

theorem hyperbola_asymptotes_and_ellipse (a b c : ℝ) (h : 0 < a) (k : 0 < b)
  (h_imag : 2 * b = 2) (h_focal : 2 * c = 2 * Real.sqrt 3) :
  2 * (a^2 + b^2 = c^2) ∧
  ((∀ x y : ℝ, x * (Real.sqrt 2 / 2) = y) ∨ (∀ x y : ℝ, x * (-Real.sqrt 2 / 2) = y)) ∧
  (∀ x y : ℝ, x^2 / 3 + y^2 = 1) :=
by
  sorry

end hyperbola_asymptotes_and_ellipse_l428_428575


namespace passing_percentage_is_correct_l428_428889

theorem passing_percentage_is_correct :
  ∀ (marks_obtained : ℕ) (marks_failed_by : ℕ) (max_marks : ℕ),
    marks_obtained = 59 →
    marks_failed_by = 40 →
    max_marks = 300 →
    (marks_obtained + marks_failed_by) / max_marks * 100 = 33 :=
by
  intros marks_obtained marks_failed_by max_marks h1 h2 h3
  sorry

end passing_percentage_is_correct_l428_428889


namespace four_letter_combination_count_l428_428421

/-- Number of different four-letter arrangements with conditions. -/
theorem four_letter_combination_count : 
  let_letters : set char := {'A', 'B', 'C', 'D', 'E', 'F', 'G'},
  ∃ arrangements : finset (list char), combinations_of_letters:
    (arrangements.card = 60) ∧
    (∀ arr ∈ arrangements, arr.length = 4) ∧ 
    (∀ arr ∈ arrangements, arr.head = 'A') ∧ 
    (∀ arr ∈ arrangements, 'B' ∈ arr) ∧ 
    (∀ arr ∈ arrangements, ∀ c ∈ arr, c ∈ let_letters) := sorry

end four_letter_combination_count_l428_428421


namespace monotonic_intervals_f_zero_range_of_a_for_two_extreme_points_l428_428570

section
variables {a x : ℝ}

-- Definition of the function f when a = 0
def f_zero (x : ℝ) : ℝ := 2 * x^3 - 6 * x

-- Definition of the function g used in the second part
def g (x : ℝ) : ℝ := 2 * real.log x - x^2 + 1

-- Function f for any a
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^3 - 6 * x - 3 * a * abs (2 * real.log x - x^2 + 1)

-- Part (1): Proving monotonicity of f_zero on specified intervals
theorem monotonic_intervals_f_zero : 
    (∀ x, 0 < x ∧ x < 1 → deriv f_zero x < 0) ∧
    (∀ x, 1 < x → deriv f_zero x > 0) :=
begin
  -- Proof goes here.
  sorry,
end

-- Part (2): Proving the range of a such that f has two extreme points
theorem range_of_a_for_two_extreme_points : 
    (0 < a ∧ a < 1) ∨ (1 < a) → 
    (∃ x1 x2, x1 ≠ x2 ∧ deriv (λ x, f x a) x1 = 0 ∧ deriv (λ x, f x a) x2 = 0) :=
begin
  -- Proof goes here.
  sorry,
end

end

end monotonic_intervals_f_zero_range_of_a_for_two_extreme_points_l428_428570


namespace num_rectangles_grid_l428_428611

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428611


namespace num_rectangles_in_5x5_grid_l428_428639

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428639


namespace Y_on_median_BM_l428_428296

-- Let circle ω1 and ω2 be defined. Point Y is an intersection of ω1 and ω2
variable (ω1 ω2 : Set (ℝ × ℝ))
variable (Y B A C M : ℝ × ℝ)

-- Assume that point Y and point B lie on the same side of line AC
variable (same_side : (Y.1 - A.1) * (C.2 - A.2) = (Y.2 - A.2) * (C.1 - A.1)
  ∧ (B.1 - A.1) * (C.2 - A.2) = (B.2 - A.2) * (C.1 - A.1))

-- Intersection of circles ω1 and ω2 at point Y
variable (intersect_Y : ω1 ∩ ω2 = {Y})

-- Definition of the median BM from point B through the midpoint M of AC
variable (BM : Set (ℝ × ℝ))
variable (midpoint_M : M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
variable (median_BM : BM = {p | ∃ t : ℝ, p = (B.1 + t * (midpoint_M.1 - B.1), B.2 + t * (midpoint_M.2 - B.2))})

-- The statement to prove is that point Y lies on the median BM
theorem Y_on_median_BM : Y ∈ BM :=
  sorry

end Y_on_median_BM_l428_428296


namespace problem_statement_l428_428203

namespace ProofProblem

def A := {x : ℝ | 0 < x ∧ x ≤ 8}

def B := {x : ℝ | x < 0 ∨ 4 < x}

theorem problem_statement :
  (A ∪ B = {x : ℝ | x ≠ 0}) ∧ (∀ x : ℝ, 4 < x → (∀ y : ℝ, x ≤ y → log (y^2 - 4 * y) ≥ log (x^2 - 4 * x))) :=
by
  -- union part
  have Hu : A ∪ B = {x : ℝ | x ≠ 0} := sorry,
  -- increasing interval part
  have Hi : ∀ x : ℝ, 4 < x → (∀ y : ℝ, x ≤ y → log (y^2 - 4 * y) ≥ log (x^2 - 4 * x)) := sorry,
  exact ⟨Hu, Hi⟩

end ProofProblem

end problem_statement_l428_428203


namespace no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1_l428_428745

theorem no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1 :
  ∀ (a b n : ℕ), (a > 1) → (b > 1) → (a ∣ 2^n - 1) → (b ∣ 2^n + 1) → ∀ (k : ℕ), ¬ (a ∣ 2^k + 1 ∧ b ∣ 2^k - 1) :=
by
  intros a b n a_gt_1 b_gt_1 a_div_2n_minus_1 b_div_2n_plus_1 k
  sorry

end no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1_l428_428745


namespace min_period_pi_and_max_value_4_l428_428567

def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - (Real.sin x)^2 + 2

theorem min_period_pi_and_max_value_4 : 
  (∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x) ∧ 
  (∃ M : ℝ, ∀ x, f x ≤ M ∧ (∃ x₀, f x₀ = M)) :=
by 
  sorry

end min_period_pi_and_max_value_4_l428_428567


namespace find_k_values_l428_428574

variable {k : ℝ}

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + k * x + 1

theorem find_k_values 
  (h_max : ∀ x ∈ set.Icc (-2 : ℝ) 2, f k x ≤ 4) 
  (h_exists : ∃ x ∈ set.Icc (-2 : ℝ) 2, f k x = 4) : 
  k = 1 / 2 ∨ k = -12 := 
sorry

end find_k_values_l428_428574


namespace Y_on_median_BM_l428_428307

variables {A B C M Y : Point}
variables {omega1 omega2 : Circle}

-- Definitions of points and circles intersection
def points_on_same_side (Y B : Point) (lineAC : Line) : Prop :=
-- Here should be the appropriate condition that defines when Y and B are on the same side of the line.
sorry

-- The main theorem
theorem Y_on_median_BM (h1 : Y ∈ (omega1 ∩ omega2)) 
(h2 : points_on_same_side Y B (line_through A C)) : 
lies_on Y (line_through B M) :=
sorry

end Y_on_median_BM_l428_428307


namespace cistern_water_breadth_l428_428872

theorem cistern_water_breadth (l w h A : ℝ) (hl : l = 7) (hw : w = 5) (hA : A = 68.6) :
  35 + 14 * h + 10 * h = A → h = 1.4 := 
by 
  intros h_eq
  rw [hl, hw, hA] at h_eq
  linarith

end cistern_water_breadth_l428_428872


namespace coin_arrangement_l428_428850

/-- Given a large square box and 12 identical coins, prove the arrangement of coins along each wall is possible or not -/
theorem coin_arrangement (n : ℕ) : 
  (n = 2 → false) ∧ 
  (n = 3 → true) ∧ 
  (n = 4 → false) ∧ 
  (n = 5 → false) ∧ 
  (n = 6 → false) ∧ 
  (n = 7 → false) :=
by {
  split,
  { intro h, linarith },
  split,
  { intro h, exact true.intro },
  split,
  { intro h, linarith },
  split,
  { intro h, linarith },
  split,
  { intro h, linarith },
  { intro h, linarith }
}

end coin_arrangement_l428_428850


namespace convex_polygon_triangulation_l428_428419

theorem convex_polygon_triangulation (n : ℕ) (a : Fin n → ℝ) (hn : ∀ i, 0 < a i) :
  ∃ (P : Fin (n+3) → ℝ × ℝ), is_convex_polygon P ∧ has_triangulation P a :=
  sorry

end convex_polygon_triangulation_l428_428419


namespace total_balls_calculation_l428_428859

theorem total_balls_calculation :
  ∃ T : ℕ, T = 100 ∧ 0.8 * T = (50 + 20 + 10) := 
by
  sorry

end total_balls_calculation_l428_428859


namespace min_ab_l428_428391

theorem min_ab {a b : ℝ} (h1 : (a^2) * (-b) + (a^2 + 1) = 0) : |a * b| = 2 :=
sorry

end min_ab_l428_428391


namespace number_of_recipes_l428_428919

theorem number_of_recipes (students : ℕ) (cookies_per_student : ℕ) (att_decrease_percent : ℝ) (cookies_per_recipe : ℕ) 
  (h1 : students = 150)
  (h2 : cookies_per_student = 3)
  (h3 : att_decrease_percent = 0.70)
  (h4 : cookies_per_recipe = 20) :
  let expected_attendance := (students : ℝ) * att_decrease_percent,
      total_cookies := expected_attendance * (cookies_per_student : ℝ),
      recipes := total_cookies / cookies_per_recipe in
  recipes.ceil = 16 :=
by
  sorry

end number_of_recipes_l428_428919


namespace circle_count_l428_428153

theorem circle_count : 
  let A := {3, 4, 6}
  let B := {1, 2, 7, 8}
  let R := {5, 9}
  (∃ a ∈ A, ∃ b ∈ B, ∃ r ∈ R, (x - a)^2 + (y - b)^2 = r^2) → 
    (finset.card (A.product (B.product R)) = 24) :=
by
  sorry

end circle_count_l428_428153


namespace vector_c_equals_a_sub_b_l428_428244

variables (a b c :  ℝ × ℝ)

def vector_a := (1, 1)
def vector_b := (2, -1)
def vector_c := (-1, 2)

theorem vector_c_equals_a_sub_b : c = (a.1 - b.1, a.2 - b.2) :=
by
  have h_a := vector_a
  have h_b := vector_b
  have h_c := vector_c
  dsimp at h_a h_b h_c
  rw [h_a, h_b, h_c]
  -- Continue proof
  sorry

end vector_c_equals_a_sub_b_l428_428244


namespace area_of_inscribed_triangle_l428_428894

noncomputable def calculate_triangle_area_inscribed_in_circle 
  (arc1 : ℝ) (arc2 : ℝ) (arc3 : ℝ) (total_circumference := arc1 + arc2 + arc3)
  (radius := total_circumference / (2 * Real.pi))
  (theta := (2 * Real.pi) / total_circumference)
  (angle1 := 5 * theta) (angle2 := 7 * theta) (angle3 := 8 * theta) : ℝ :=
  0.5 * (radius ^ 2) * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem area_of_inscribed_triangle : 
  calculate_triangle_area_inscribed_in_circle 5 7 8 = 119.85 / (Real.pi ^ 2) :=
by
  sorry

end area_of_inscribed_triangle_l428_428894


namespace problem_part1_problem_part2_problem_part3_l428_428319

noncomputable def f (x a b c : ℝ) : ℝ := (x - a) * (x - b) * (x - c)

theorem problem_part1 (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : b = 0) (h3 : c = 1) :
  ∃ m y_sol, (m, y_sol) = (2, 0) ∧ (λ x y, 2 * x - y - 2 = 0) 1 y_sol := by sorry

theorem problem_part2 (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : b - a = 3) :
  ((6 * real.sqrt 3), (-6 * real.sqrt 3)) ∈ (λ x, f x a b c) '' (set_of (λ x, f' x = 0)) := by sorry

theorem problem_part3 (a b c : ℝ) (h1 : 2 * b = a + c) :
  (λ x, f x a b c) intersects (λ x, -(x - b) - 2) in 3 distinct points → 
  ∃ d, d ∈ set.union (set.Iio (-4)) (set.Ioi 4) := by sorry

end problem_part1_problem_part2_problem_part3_l428_428319


namespace infinite_jump_impossible_l428_428817

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
if n < 2 then 1 else
  let primes := List.filter (λ p, p.prime) (List.range n).reverse in
  List.find (λ p, n % p = 0) primes

noncomputable def smallest_prime_factor (n : ℕ) : ℕ :=
if n < 2 then 1 else
  let primes := List.filter (λ p, p.prime) (List.range n).tail in
  List.find (λ p, n % p = 0) primes

theorem infinite_jump_impossible (k : ℕ) (h : k > 1) :
  ¬ ∃ S : Set ℕ, k ∈ S ∧ (∀ n ∈ S, largest_prime_factor n + smallest_prime_factor n ∈ S) ∧ S.infinite := by
  sorry

end infinite_jump_impossible_l428_428817


namespace tv_height_l428_428766

theorem tv_height (H : ℝ) : 
  672 / (24 * H) = (1152 / (48 * 32)) + 1 → 
  H = 16 := 
by
  have h_area_first_TV : 24 * H ≠ 0 := sorry
  have h_new_condition: 1152 / (48 * 32) + 1 = 1.75 := sorry
  have h_cost_condition: 672 / (24 * H) = 1.75 := sorry
  sorry

end tv_height_l428_428766


namespace intersection_A_B_eq_complement_union_eq_subset_condition_l428_428222

open Set

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | x > 3 / 2}
noncomputable def C (a : ℝ) : Set ℝ := {x : ℝ | 1 < x ∧ x < a}

theorem intersection_A_B_eq : A ∩ B = {x : ℝ | 3 / 2 < x ∧ x ≤ 3} :=
by sorry

theorem complement_union_eq : (univ \ B) ∪ A = {x : ℝ | x ≤ 3} :=
by sorry

theorem subset_condition (a : ℝ) : (C a ⊆ A) → (a ≤ 3) :=
by sorry

end intersection_A_B_eq_complement_union_eq_subset_condition_l428_428222


namespace least_number_to_make_divisible_by_3_l428_428025

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem least_number_to_make_divisible_by_3 : ∃ k : ℕ, (∃ n : ℕ, 
  sum_of_digits 625573 ≡ 28 [MOD 3] ∧ 
  (625573 + k) % 3 = 0 ∧ 
  k = 2) :=
by
  sorry

end least_number_to_make_divisible_by_3_l428_428025


namespace complex_imaginary_condition_l428_428706

theorem complex_imaginary_condition (m : ℝ) : (∀ m : ℝ, (m^2 - 3*m - 4 = 0) → (m^2 - 5*m - 6) ≠ 0) ↔ (m ≠ -1 ∧ m ≠ 6) :=
by
  sorry

end complex_imaginary_condition_l428_428706


namespace polynomial_strictly_monotonic_l428_428807

variable {P : ℝ → ℝ}

/-- The polynomial P(x) is such that the polynomials P(P(x)) and P(P(P(x))) are strictly monotonic 
on the entire real axis. Prove that P(x) is also strictly monotonic on the entire real axis. -/
theorem polynomial_strictly_monotonic
  (h1 : StrictMono (P ∘ P))
  (h2 : StrictMono (P ∘ P ∘ P)) :
  StrictMono P :=
sorry

end polynomial_strictly_monotonic_l428_428807


namespace branch_A_grade_A_probability_branch_B_grade_A_probability_branch_A_average_profit_branch_B_average_profit_choose_branch_l428_428437

theorem branch_A_grade_A_probability : 
  let total_A := 100
  let grade_A_A := 40
  (grade_A_A / total_A) = 0.4 := by
  sorry

theorem branch_B_grade_A_probability : 
  let total_B := 100
  let grade_A_B := 28
  (grade_A_B / total_B) = 0.28 := by
  sorry

theorem branch_A_average_profit :
  let freq_A_A := 0.4
  let freq_A_B := 0.2
  let freq_A_C := 0.2
  let freq_A_D := 0.2
  let process_cost_A := 25
  let profit_A := (90 - process_cost_A) * freq_A_A + (50 - process_cost_A) * freq_A_B + (20 - process_cost_A) * freq_A_C + (-50 - process_cost_A) * freq_A_D
  profit_A = 15 := by
  sorry

theorem branch_B_average_profit :
  let freq_B_A := 0.28
  let freq_B_B := 0.17
  let freq_B_C := 0.34
  let freq_B_D := 0.21
  let process_cost_B := 20
  let profit_B := (90 - process_cost_B) * freq_B_A + (50 - process_cost_B) * freq_B_B + (20 - process_cost_B) * freq_B_C + (-50 - process_cost_B) * freq_B_D
  profit_B = 10 := by
  sorry

theorem choose_branch :
  let profit_A := 15
  let profit_B := 10
  profit_A > profit_B -> "Branch A"

end branch_A_grade_A_probability_branch_B_grade_A_probability_branch_A_average_profit_branch_B_average_profit_choose_branch_l428_428437


namespace sum_combinatorial_identity_l428_428340

theorem sum_combinatorial_identity (n : ℕ) (hn : n > 0) :
  ∑ k in finset.range (n / 2 + 1), (-1 : ℤ)^k * nat.choose (n+1) k * nat.choose (2*n - 2*k - 1) n = (n * (n + 1)) / 2 :=
begin
  sorry
end

end sum_combinatorial_identity_l428_428340


namespace integral_I1_approx_integral_I2_approx_integral_I3_approx_l428_428046

noncomputable def I1_approx : ℝ :=
  (((1 / 3) : ℝ) - (1 / 2) * ((1 / 3) ^ 3) / 3 + (1 * 3) * ((1 / 3) ^ 5) / ((2 * 4) * 5) - (1 * 3 * 5) * ((1 / 3) ^ 7) / ((2 * 4 * 6) * 7)) -- truncated to required terms

noncomputable def I2_approx : ℝ :=
  1 - (1 / 4) + (1 / 72) - (1 / 2880) -- truncated to required terms

noncomputable def I3_approx : ℝ :=
  ((1 / 4) * real.log(1.5) - ((1.5)^2 - 1) / (32 * 3) + ((1.5)^4 - 1) / (1024 * 5) - ((1.5)^6 - 1) / (16384 * 7)) -- truncated to required terms

theorem integral_I1_approx :
  ∫ t in (0 : ℝ)..(1 / 3), (1 / (real.sqrt (1 + t ^ 2))) ≈ 0.32716 ∂t :=
begin
  -- Skipping actual proof, it is assumed to hold based on steps outlined.
  sorry
end

theorem integral_I2_approx :
  ∫ x in (0 : ℝ)..1, real.cos (real.pi * x) ≈ 0.76354 ∂x :=
begin
  -- Skipping actual proof, it is assumed to hold based on steps outlined.
  sorry
end

theorem integral_I3_approx :
  ∫ v in (1 : ℝ)..(1.5), (1 / v) * real.arctan (v / 4) ≈ 0.1211 ∂v :=
begin
  -- Skipping actual proof, it is assumed to hold based on steps outlined.
  sorry
end

end integral_I1_approx_integral_I2_approx_integral_I3_approx_l428_428046


namespace roses_multiple_of_four_l428_428271

theorem roses_multiple_of_four (R : ℕ) (H_daisies : 12 % 4 = 0)
                               (H_marigolds : 48 % 4 = 0)
                               (H_arrangements : ∀ r d m, d = 12 / 4 ∧ m = 48 / 4 → R % 4 = 0): 
  R % 4 = 0 := 
by 
  sorry

end roses_multiple_of_four_l428_428271


namespace largest_three_digit_hidden_smallest_hiding_both_multiple_hides_units_digit_three_l428_428459

-- Define the predicate "hides" for numbers.
def hides (x y : ℕ) : Prop :=
  ∃ (s : List (Fin n)), List.isSubsequenceOf s.toList (x.toString.toList ⊗ y.toString.toList)

-- (a) Prove that the largest three-digit number hidden by 47239 is 739.
theorem largest_three_digit_hidden {n : ℕ} : hides 47239 739 ∧ ∀ x, hides 47239 x → x < 1000 → x <= 739 :=
  sorry

-- (b) Prove that the smallest number that simultaneously hides 2009 and 9002 is 290029.
theorem smallest_hiding_both : hides 290029 2009 ∧ hides 290029 9002 ∧ ∀ x, (hides x 2009 ∧ hides x 9002) → x >= 290029 :=
  sorry

-- (c) Prove that there exists a multiple of 2009 that hides 2009 and whose units digit is 3.
theorem multiple_hides_units_digit_three : hides 200914063 2009 ∧ 200914063 % 10 = 3 ∧ ∃ k, 200914063 = 2009 * k :=
  sorry

end largest_three_digit_hidden_smallest_hiding_both_multiple_hides_units_digit_three_l428_428459


namespace part_1_part_2_l428_428152

variable {a b : ℝ}

theorem part_1 (ha : a > 0) (hb : b > 0) : a^2 + 3 * b^2 ≥ 2 * b * (a + b) :=
sorry

theorem part_2 (ha : a > 0) (hb : b > 0) : a^3 + b^3 ≥ a * b^2 + a^2 * b :=
sorry

end part_1_part_2_l428_428152


namespace parabola_directrix_distance_l428_428707

theorem parabola_directrix_distance (a : ℝ) : 
  (abs (a / 4 + 1) = 2) → (a = -12 ∨ a = 4) := 
by
  sorry

end parabola_directrix_distance_l428_428707


namespace find_common_difference_find_max_sum_find_max_n_l428_428371

-- Condition for the sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Problem statement (1): Find the common difference
theorem find_common_difference (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 23)
  (h2 : is_arithmetic_sequence a d)
  (h6 : a 6 > 0)
  (h7 : a 7 < 0) : d = -4 :=
sorry

-- Problem statement (2): Find the maximum value of the sum S₆
theorem find_max_sum (d : ℤ) (h : d = -4) : 6 * 23 + (6 * 5 / 2) * d = 78 :=
sorry

-- Problem statement (3): Find the maximum value of n when S_n > 0
theorem find_max_n (d : ℤ) (h : d = -4) : ∀ n : ℕ, (n > 0 ∧ (23 * n + (n * (n - 1) / 2) * d > 0)) → n ≤ 12 :=
sorry

end find_common_difference_find_max_sum_find_max_n_l428_428371


namespace new_parabola_through_point_l428_428242

def original_parabola (x : ℝ) : ℝ := x ^ 2 + 2 * x - 1

theorem new_parabola_through_point : 
  (∃ b : ℝ, ∀ x : ℝ, (x ^ 2 + 2 * x - 1 + b) = (x ^ 2 + 2 * x + 3)) :=
by
  sorry

end new_parabola_through_point_l428_428242


namespace five_minus_three_to_the_negative_three_eq_134_div_27_l428_428834

theorem five_minus_three_to_the_negative_three_eq_134_div_27 : 5 - 3⁻³ = 134 / 27 := 
by
  sorry

end five_minus_three_to_the_negative_three_eq_134_div_27_l428_428834


namespace lucy_headmaster_duration_l428_428328

-- Define the months as an inductive type
@[derive DecidableEq]
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August
| September
| October
| November
| December

-- Function to count the number of months from the start month to the end month, inclusive
def monthsBetween : Month → Month → ℕ
| Month.March, Month.March   => 1
| Month.March, Month.April   => 2
| Month.March, Month.May     => 3
| Month.March, Month.June    => 4
| _, _ => 0  -- Other cases not needed for this problem

-- Given conditions
def startMonth := Month.March
def endMonth := Month.June

-- Statement to prove
theorem lucy_headmaster_duration : monthsBetween startMonth endMonth = 4 :=
by
  -- Proof goes here
  sorry

end lucy_headmaster_duration_l428_428328


namespace excircle_identity_l428_428712

variables (a b c r_a r_b r_c : ℝ)

-- Conditions: r_a, r_b, r_c are the radii of the excircles opposite vertices A, B, and C respectively.
-- In the triangle ABC, a, b, c are the sides opposite vertices A, B, and C respectively.

theorem excircle_identity:
  (a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b))) = 2 :=
by
  sorry

end excircle_identity_l428_428712


namespace triangle_area_inscribed_in_circle_l428_428905

noncomputable def circle_inscribed_triangle_area : ℝ :=
  let r := 10 / Real.pi
  let angle_A := Real.pi / 2
  let angle_B := 7 * Real.pi / 10
  let angle_C := 4 * Real.pi / 5
  let sin_sum := Real.sin(angle_A) + Real.sin(angle_B) + Real.sin(angle_C)
  1 / 2 * r^2 * sin_sum

theorem triangle_area_inscribed_in_circle (h_circumference : 5 + 7 + 8 = 20)
  (h_radius : 10 / Real.pi * 2 * Real.pi = 20) :
  circle_inscribed_triangle_area = 138.005 / Real.pi^2 :=
by
  sorry

end triangle_area_inscribed_in_circle_l428_428905


namespace problem_statement_l428_428568

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 4

theorem problem_statement (a x₁ x₂: ℝ) (ha : a > 0) (hx : x₁ < x₂) (hxsum : x₁ + x₂ = 0) :
  f a x₁ < f a x₂ := by
  sorry

end problem_statement_l428_428568


namespace number_of_rectangles_in_5x5_grid_l428_428657

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428657


namespace num_rectangles_in_5x5_grid_l428_428623

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428623


namespace pancake_fundraiser_l428_428354

-- Define the constants and conditions
def cost_per_stack_of_pancakes : ℕ := 4
def cost_per_slice_of_bacon : ℕ := 2
def stacks_sold : ℕ := 60
def slices_sold : ℕ := 90
def total_raised : ℕ := 420

-- Define a theorem that states what we want to prove
theorem pancake_fundraiser : 
  (stacks_sold * cost_per_stack_of_pancakes + slices_sold * cost_per_slice_of_bacon) = total_raised :=
by
  sorry -- We place a sorry here to skip the proof, as instructed.

end pancake_fundraiser_l428_428354


namespace sum_squares_even_odd_difference_l428_428485

theorem sum_squares_even_odd_difference :
  let sum_squares_natural := ∑ k in finset.range 101, k^2
  let sum_squares_100_even := 2^2 * sum_squares_natural
  let sum_squares_100_odd := ∑ n in finset.range 100, (2*n + 1)^2
  sum_squares_100_even - sum_squares_100_odd = 20100 :=
by
  sorry

end sum_squares_even_odd_difference_l428_428485


namespace proof_w3_u2_y2_l428_428325

variable (x y z w u d : ℤ)

def arithmetic_sequence := x = 1370 ∧ z = 1070 ∧ w = -180 ∧ u = -6430 ∧ (z = x + 2 * d) ∧ (y = x + d)

theorem proof_w3_u2_y2 (h : arithmetic_sequence x y z w u d) : w^3 - u^2 + y^2 = -44200100 :=
  by
    sorry

end proof_w3_u2_y2_l428_428325


namespace frac_m_q_eq_one_l428_428232

theorem frac_m_q_eq_one (m n p q : ℕ) 
  (h1 : m = 40 * n)
  (h2 : p = 5 * n)
  (h3 : p = q / 8) : (m / q = 1) :=
by
  sorry

end frac_m_q_eq_one_l428_428232


namespace sum_of_k_for_distinct_integer_roots_l428_428833

theorem sum_of_k_for_distinct_integer_roots : 
  (∃ k : ℤ, ∀ x, 2 * x^2 - k * x + 8 = 0 → (∃ p q : ℤ, p ≠ q ∧ 2 * (p + q) = k ∧ p * q = 4)) →
  (0 = ∑ k in {k ∈ ℤ | ∀ x, 2 * x^2 - k * x + 8 = 0 → (∃ p q : ℤ, p ≠ q ∧ 2 * (p + q) = k ∧ p * q = 4)}, k) := 
by
  sorry

end sum_of_k_for_distinct_integer_roots_l428_428833


namespace work_rate_c_l428_428041

variables (rate_a rate_b rate_c : ℚ)

-- Given conditions
axiom h1 : rate_a + rate_b = 1 / 15
axiom h2 : rate_a + rate_b + rate_c = 1 / 6

theorem work_rate_c : rate_c = 1 / 10 :=
by sorry

end work_rate_c_l428_428041


namespace sum_of_roots_eq_sum_of_squares_l428_428513

theorem sum_of_roots_eq_sum_of_squares (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 + x2 = x1^2 + x2^2 ∧ x^2 - 2a * (x - 1) - 1 = 0) ↔ (a = 1 ∨ a = 1 / 2) :=
sorry

end sum_of_roots_eq_sum_of_squares_l428_428513


namespace scenario_1_independence_l428_428093

-- Define the scenarios as predicates:
def first_toss_heads (ω : Ω) : Prop := ω = "first heads"
def second_toss_tails (ω : Ω) : Prop := ω = "second tails"
def first_draw_white (ω : Ω) : Prop := ω = "first white"
def second_draw_white (ω : Ω) : Prop := ω = "second white"
def outcome_odd (ω : Ω) : Prop := ω = "odd number"
def outcome_even (ω : Ω) : Prop := ω = "even number"
def live_to_20 (ω : Ω) : Prop := ω = "lives to 20 years"
def live_to_50 (ω : Ω) : Prop := ω = "lives to 50 years"

-- Define mutual independence:
def mutually_independent (A B : Ω → Prop) (P : Ω → ℝ) :=
  P { ω | A ω ∧ B ω } = P { ω | A ω } * P { ω | B ω }

-- Prove that the first scenario has mutually independent events.
theorem scenario_1_independence (Ω : Type) (P : Ω → ℝ) :
  mutually_independent first_toss_heads second_toss_tails P :=
by
  sorry

end scenario_1_independence_l428_428093


namespace number_of_rectangles_in_5x5_grid_l428_428660

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428660


namespace distances_from_vertices_l428_428810

structure Triangle :=
(a b c : ℝ)
(h1 : a > 0)
(h2 : b > 0)
(h3 : c > 0)
(h_triangle_inequality1 : a + b > c)
(h_triangle_inequality2 : a + c > b)
(h_triangle_inequality3 : b + c > a)

noncomputable def findPoint_P (T : Triangle) : (ℝ × ℝ × ℝ) :=
let a := T.a,
    b := T.b,
    c := T.c in
(17.24, 35.94, 54.61) -- The distances from the vertices to the point P

theorem distances_from_vertices (T : Triangle) :
  let (PA, PB, PC) := findPoint_P T in
  PA ≈ 17.24 ∧ PB ≈ 35.94 ∧ PC ≈ 54.61 :=
by sorry

end distances_from_vertices_l428_428810


namespace distance_between_points_l428_428515

theorem distance_between_points:
  dist (0, 4) (3, 0) = 5 :=
by
  sorry

end distance_between_points_l428_428515


namespace area_of_moving_point_l428_428848

theorem area_of_moving_point (a b : ℝ) :
  (∀ (x y : ℝ), abs x ≤ 1 ∧ abs y ≤ 1 → a * x - 2 * b * y ≤ 2) →
  ∃ (A : ℝ), A = 8 := sorry

end area_of_moving_point_l428_428848


namespace closest_estimate_l428_428121

theorem closest_estimate : 
  let options := {500, 1500, 5000, 15000, 50000}
  let number := 504 / 0.102 * 3
  (∀ x ∈ options, abs (number - 15000) ≤ abs (number - x)) :=
by
  let options := {500, 1500, 5000, 15000, 50000}
  let number := 504 / 0.102 * 3
  have closest := 15000
  sorry

end closest_estimate_l428_428121


namespace car_speed_40_kmph_l428_428053

theorem car_speed_40_kmph (v : ℝ) (h : 1 / v = 1 / 48 + 15 / 3600) : v = 40 := 
sorry

end car_speed_40_kmph_l428_428053


namespace impracticable_metro_l428_428250

theorem impracticable_metro (G : SimpleGraph (Fin 2019)) (hconn : G.Connected) :
  ¬ ∃ (k : ℕ), k ≤ 1008 ∧ 
  ∀ v : Fin 2019, ∃ (path : FinSet (Fin 2019)), path.val ⊆ G.NeighborSet v ∧ path.card ≥ 2 :=
sorry

end impracticable_metro_l428_428250


namespace inscribed_triangle_area_l428_428901

theorem inscribed_triangle_area 
  (r : ℝ) (theta : ℝ) 
  (A B C : ℝ) (arc1 arc2 arc3 : ℝ)
  (h_arc1 : arc1 = 5)
  (h_arc2 : arc2 = 7)
  (h_arc3 : arc3 = 8)
  (h_sum_arcs : arc1 + arc2 + arc3 = 2 * π * r)
  (h_theta : theta = 20)
  -- in radians: h_theta_rad : θ = (20 * π / 180)
  (h_A : A = 100)
  (h_B : B = 140)
  (h_C : C = 120) :
  let sin_A := sin (A * π / 180)
    sin_B := sin (B * π / 180)
    sin_C := sin (C * π / 180) in
  1 / 2 * (10 / π) ^ 2 * (sin_A + sin_B + sin_C) = 249.36 / π^2 := 
sorry

end inscribed_triangle_area_l428_428901


namespace escalator_walk_rate_l428_428474

theorem escalator_walk_rate (v : ℝ) : (v + 15) * 10 = 200 → v = 5 := by
  sorry

end escalator_walk_rate_l428_428474


namespace WXYZ_is_parallelogram_l428_428321

variable (A B C D W X Y Z : Type) 
variable [Parallelogram A B C D] 
variable [OnSide W A B] 
variable [OnSide X B C] 
variable [OnSide Y C D] 
variable [OnSide Z D A] 

variable (I₁ I₂ I₃ I₄ : Type) 
variable [Incenter I₁ A W Z] 
variable [Incenter I₂ B X W] 
variable [Incenter I₃ C Y X] 
variable [Incenter I₄ D Z Y] 
variable [Parallelogram I₁ I₂ I₃ I₄] 

theorem WXYZ_is_parallelogram : Parallelogram W X Y Z := 
sorry

end WXYZ_is_parallelogram_l428_428321


namespace find_b_l428_428141

noncomputable def polynomial_roots (a b c d : ℝ) : Prop := 
  ∃ p q r s : ℂ, 
    (p + q = 5 + 2 * complex.I) ∧ 
    (r * s = 10 - complex.I) ∧ 
    (p * q = conj p * conj q) ∧ 
    (r * s = conj r * conj s) ∧
    (p + conj p + q + conj q + r + conj r + s + conj s = -a) ∧ 
    (p * conj p + q * conj q + r * conj r + s * conj s + (p * q + r * s + p * conj q + q * conj p + r * conj s + s * conj r) = b) ∧
    (p * conj p * q * conj q * r * conj r * s * conj s = d)
    
theorem find_b (a d c: ℝ) : 
  polynomial_roots a 49 c d ↔ 
  (∃ p q r s : ℂ, 
    (p + q = 5 + 2 * complex.I) ∧ 
    (conj p + conj q = 5 - 2 * complex.I) ∧
    (r * s = 10 - complex.I) ∧
    (conj r * conj s = 10 + complex.I) ∧
    (p * conj p + q * conj q + r * conj r + s * conj s + 
    (p * q + r * s + p * conj q + q * conj p + r * conj s + s * conj r) = 49)) :=
by
  sorry

end find_b_l428_428141


namespace cosine_theorem_l428_428950

theorem cosine_theorem 
  (A B C : Point)
  (a b c : ℝ)
  (hC : ∠ B C A = C)
  (ha : a = dist A B)
  (hb : b = dist B C)
  (hc : c = dist C A) :
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C := 
sorry

end cosine_theorem_l428_428950


namespace number_of_correct_propositions_is_one_l428_428545

variables (a b : ℝ → ℝ → Prop) (β : ℝ → ℝ → ℝ → Prop)

def parallel (x y : ℝ → ℝ → Prop) : Prop :=
  ∀ z w, (x z w ↔ y z w)

def perpendicular (x y : ℝ → ℝ → Prop) : Prop :=
  ∀ z w, (x z w → ¬ y z w)

noncomputable def subset (x : ℝ → ℝ → Prop) (ℯ : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ z w, (x z w → ∃ u, ℯ z w u)

noncomputable def intersect (x : ℝ → ℝ → Prop) (ℯ : ℝ → ℝ → ℝ → Prop) (B : Prop) : Prop :=
  ∃ z w u, (x z w ∧ ℯ z w u ∧ B)

def skew_lines (x y : ℝ → ℝ → Prop) : Prop :=
  ∀ z w u, (¬ (x z w ∧ y z w ∧ z = w))

theorem number_of_correct_propositions_is_one :
  ¬ (parallel a β ∧ parallel a b → parallel b β) ∧
  ¬ (subset a β ∧ intersect b β (true) → skew_lines a b) ∧
  ¬ (perpendicular a b ∧ perpendicular a β → parallel b β) ∧
  (parallel a b ∧ perpendicular b β → perpendicular a β) :=
sorry

end number_of_correct_propositions_is_one_l428_428545


namespace circumscribed_quadrilateral_inequality_l428_428009

theorem circumscribed_quadrilateral_inequality 
    (A B C D : ℤ × ℤ) 
    (grid_vertices : ∀ P ∈ {A, B, C, D}, ∃ m n : ℤ, P = (m, n)) 
    (not_trapezoid : ¬ (is_trapezoid A B C D))
    (circumscribed : is_circumscribed_quadrilateral A B C D) :
    |(euclidean_distance_sq A C) * (euclidean_distance_sq A D) - (euclidean_distance_sq B C) * (euclidean_distance_sq B D)| ≥ 1 :=
by
  sorry

end circumscribed_quadrilateral_inequality_l428_428009


namespace closed_form_F_l428_428982

noncomputable def α : ℝ := (3 - Real.sqrt 5) / 2
def f (n : ℕ) : ℕ := int.floor (α * n)

-- Recursive function definition for F
def F : ℕ → ℕ 
| 0 := 1
| 1 := 3
| (k+2) := 3 * F (k+1) - F k

theorem closed_form_F (k : ℕ) : 
  F k = (1 / Real.sqrt 5) * (( (3 + Real.sqrt 5) / 2)^(k+1) - ((3 - Real.sqrt 5) / 2)^(k+1)) :=
sorry

end closed_form_F_l428_428982


namespace meadow_grazing_days_l428_428879

theorem meadow_grazing_days 
    (a b x : ℝ) 
    (h1 : a + 6 * b = 27 * 6 * x)
    (h2 : a + 9 * b = 23 * 9 * x)
    : ∃ y : ℝ, (a + y * b = 21 * y * x) ∧ y = 12 := 
by
    sorry

end meadow_grazing_days_l428_428879


namespace sequence_general_term_l428_428221

theorem sequence_general_term (a : ℕ → ℕ) 
  (h₀ : a 1 = 4) 
  (h₁ : ∀ n : ℕ, a (n + 1) = 2 * a n + n^2) : 
  ∀ n : ℕ, a n = 5 * 2^n - n^2 - 2*n - 3 :=
by
  sorry

end sequence_general_term_l428_428221


namespace cos_sin_sum_eq_neg_one_l428_428488

-- Define the necessary trigonometric and complex number properties.
noncomputable def complex_trigonometric_sum : ℂ :=
  ∑ n in finset.range 31, (λ k =>
    if k % 4 = 0 then complex.ofReal (real.cos ((45 + 180 * k : ℤ).toReal))
    else if k % 4 = 1 then complex.i * complex.ofReal (real.sin ((135 + 180 * k : ℤ).toReal))
    else if k % 4 = 2 then -complex.ofReal (real.cos ((225 + 180 * k : ℤ).toReal))
    else -complex.i * complex.ofReal (real.sin ((315 + 180 * k : ℤ).toReal)))
  n

-- Establish the given conditions using local definitions or assumptions.
theorem cos_sin_sum_eq_neg_one :
  complex_trigonometric_sum = -1 := by
  -- The following proof is omitted.
  sorry

end cos_sin_sum_eq_neg_one_l428_428488


namespace eating_time_l428_428740

-- Define conditions
def jayes_eating_rate_per_hour : ℝ := 75  -- Jayes can eat 75 marshmallows per hour
def dylan_eating_rate_per_hour : ℝ := 25  -- Dylan can eat 25 marshmallows per hour
def combined_rate : ℝ := 150              -- Combined rate is 150 marshmallows per hour

-- Statement of the theorem
theorem eating_time (M : ℝ) (hM : M ≥ 0) : T where
  T = M / combined_rate :=
begin
  -- Calculate the time T
  sorry
end

end eating_time_l428_428740


namespace perimeter_gt_sixteen_l428_428175

theorem perimeter_gt_sixteen (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 :=
by
  sorry

end perimeter_gt_sixteen_l428_428175


namespace range_of_m_l428_428159

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 4 / y = 1) (H : x + y > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l428_428159


namespace intersection_a_zero_range_of_a_l428_428189

variable (x a : ℝ)

def setA : Set ℝ := { x | - 1 < x ∧ x < 6 }
def setB (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x < a + 5 }

theorem intersection_a_zero :
  setA x ∧ setB 0 x ↔ - 1 < x ∧ x < 5 := by
  sorry

theorem range_of_a (h : ∀ x, setA x ∨ setB a x → setA x) :
  (0 < a ∧ a ≤ 1) ∨ 6 ≤ a :=
  sorry

end intersection_a_zero_range_of_a_l428_428189


namespace lines_PQ_intersect_at_single_point_l428_428266

open_locale classical
noncomputable theory

variables {A B C P A1 B1 C1 Q: Type} [Inhabited A] [Inhabited B] [Inhabited C]
  [Inhabited P] [Inhabited A1] [Inhabited B1] [Inhabited C1] [Inhabited Q]

-- Given that the points are in the plane
variables (triangle : Type) [inhabited triangle]
variable circumcircle : set triangle

-- Assumptions
variables
  (ABC : triangle)
  (P : triangle) (hP_BC : P ∈ segment BC)
  (A1 B1 C1 : triangle) (hA1_circ : A1 ∈ circumcircle) (hB1_circ : B1 ∈ circumcircle) (hC1_circ : C1 ∈ circumcircle)
  (hAA1_Q : line_through A A1 ∩ line_through B B1 ∩ line_through C C1 = {Q})

-- Conclusion to prove
theorem lines_PQ_intersect_at_single_point :
  ∃ X : triangle, ∀ P : line_through P Q, line_through P Q ∋ X :=
sorry

end lines_PQ_intersect_at_single_point_l428_428266


namespace square_side_length_l428_428075

-- Define the conditions
def d1 : ℝ := 16
def d2 : ℝ := 8
def square_area : ℝ := (d1 * d2) / 2

-- State the theorem
theorem square_side_length :
  ∃ s : ℝ, square_area = s^2 ∧ s = 8 :=
by
  exists 8
  split
  · sorry
  · sorry

end square_side_length_l428_428075


namespace car_initial_wait_time_l428_428057

theorem car_initial_wait_time
  (v_cyclist : ℝ) (v_car : ℝ) (T_hours : ℝ) (T_minutes : ℝ)
  (cyclist_speed : v_cyclist = 15) (car_speed : v_car = 60)
  (wait_time_in_hours : T_hours = 0.3) (wait_time_in_minutes : T_minutes = 18) :
  let relative_speed := v_car - v_cyclist in
  let distance_covered_by_cyclist := v_cyclist * T_hours in
  let time_car_traveled := distance_covered_by_cyclist / v_car in
  (time_car_traveled * 60) = 4.5 :=
by
  sorry

end car_initial_wait_time_l428_428057


namespace bianca_carrots_l428_428479

theorem bianca_carrots : 
  ∀ (initial_picked : ℕ) (thrown_out : ℕ) (total_carrots : ℕ),
  initial_picked = 23 →
  thrown_out = 10 →
  total_carrots = 60 →
  (total_carrots = initial_picked - thrown_out + carrots_picked_next_day) →
  carrots_picked_next_day = 47 :=
  let initial_picked := 23
  let thrown_out := 10
  let total_carrots := 60
  let remaining_carrots := initial_picked - thrown_out
  let carrots_picked_next_day := total_carrots - remaining_carrots
  (remaining_carrots_eq : remaining_carrots = 13) →
  sorry
  sorry

end bianca_carrots_l428_428479


namespace length_of_AB_l428_428729

noncomputable def calculate_AB (a b : ℝ) : ℝ :=
  let AD := 2 * a
  let CD := 3 * b
  4 * a / 5

theorem length_of_AB (a b : ℝ) :
  let AD := 2 * a in
  let CD := 3 * b in
  let AB := calculate_AB a b in
  AD = 2 * a ∧ CD = 3 * b → AB = 4 * a / 5 :=
by
  intros AD_eq CD_eq
  simp [calculate_AB, AD_eq, CD_eq]
  sorry

end length_of_AB_l428_428729


namespace num_rectangles_in_5x5_grid_l428_428630

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428630


namespace minimum_value_of_x_is_correct_l428_428360

def f (x : ℝ) : ℝ := 2 * real.sqrt 3 * real.cos x ^ 2 - 2 * real.sin x * real.cos x - real.sqrt 3

def t (k : ℤ) : ℝ := k * real.pi + real.pi / 6

noncomputable def x : ℤ → ℝ
| k => k * real.pi + 11 * real.pi / 12

theorem minimum_value_of_x_is_correct (k : ℤ) :
  (2 * (k * real.pi + real.pi / 6) - real.pi / 3 = k * real.pi) ∧ 
  (-2 * real.sin (2 * (k * real.pi + real.pi / 6) - real.pi / 3) = -2 * real.sin (2 * (k * real.pi + real.pi / 6) - real.pi / 3)) →
  ∃ m : ℤ, k = 0 → x k = 59 * real.pi / 12 :=
begin
  intros h,
  use 0,
  sorry
end

end minimum_value_of_x_is_correct_l428_428360


namespace exponent_simplification_l428_428114

theorem exponent_simplification : (7^3 * (2^5)^3) / (7^2 * 2^(3*3)) = 448 := by
  sorry

end exponent_simplification_l428_428114


namespace arithmetic_mean_difference_l428_428844

theorem arithmetic_mean_difference (p q r : ℝ)
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 22) :
  r - p = 24 :=
by
  sorry

end arithmetic_mean_difference_l428_428844


namespace project_completion_time_l428_428367

def process_duration (a b c d e f : Nat) : Nat :=
  let duration_c := max a b + c
  let duration_d := duration_c + d
  let duration_e := duration_c + e
  let duration_f := max duration_d duration_e + f
  duration_f

theorem project_completion_time :
  ∀ (a b c d e f : Nat), a = 2 → b = 3 → c = 2 → d = 5 → e = 4 → f = 1 →
  process_duration a b c d e f = 11 := by
  intros
  subst_vars
  sorry

end project_completion_time_l428_428367


namespace jaewoong_ran_the_most_l428_428739

def distance_jaewoong : ℕ := 20000 -- Jaewoong's distance in meters
def distance_seongmin : ℕ := 2600  -- Seongmin's distance in meters
def distance_eunseong : ℕ := 5000  -- Eunseong's distance in meters

theorem jaewoong_ran_the_most : distance_jaewoong > distance_seongmin ∧ distance_jaewoong > distance_eunseong := by
  sorry

end jaewoong_ran_the_most_l428_428739


namespace problem_l428_428814

noncomputable def complete_graph_inequality (n k : ℕ) (N_k N_k1 : ℕ) : Prop :=
  (k ∈ {2, 3, ..., n-1}) → (N_k ≠ 0) →
  (N_k1 / N_k) ≥ (1 / (k^2 - 1)) * ((k^2 * N_k) / N_k1 - n)

theorem problem (n k : ℕ) (N_k N_k1 : ℕ)
  (h1 : n > 0) (h2 : k ∈ {2, 3, ..., n-1}) (h3 : N_k ≠ 0) : 
  complete_graph_inequality n k N_k N_k1 :=
by sorry

end problem_l428_428814


namespace pentagon_ratio_l428_428985

theorem pentagon_ratio (ABCDE : list Point)
                       (h_reg : is_regular_pentagon ABCDE)
                       (K L : Point)
                       (h_K_on_AE : on_segment K (nth 0 ABCDE) (nth 4 ABCDE))
                       (h_L_on_CD : on_segment L (nth 2 ABCDE) (nth 3 ABCDE))
                       (angle_cond : ∠LAE + ∠KCD = 108)
                       (ratio_cond : side_ratio (AK : KE) = 3 : 7) :
    (CL / AB) = 0.7 :=
by
  sorry

end pentagon_ratio_l428_428985


namespace minimum_value_l428_428183

variables (a b c d : ℝ)
-- Conditions
def condition1 := (b - 2 * a^2 + 3 * Real.log a)^2 = 0
def condition2 := (c - d - 3)^2 = 0

-- Theorem stating the goal
theorem minimum_value (h1 : condition1 a b) (h2 : condition2 c d) : 
  (a - c)^2 + (b - d)^2 = 8 :=
sorry

end minimum_value_l428_428183


namespace binom_difference_30_3_2_l428_428100

-- Define the binomial coefficient function.
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement: binom(30, 3) - binom(30, 2) = 3625
theorem binom_difference_30_3_2 : binom 30 3 - binom 30 2 = 3625 := by
  sorry

end binom_difference_30_3_2_l428_428100


namespace yellow_fraction_after_tripling_l428_428715

variable {x : ℕ} -- general variable for total number of marbles

-- Definitions based on conditions
def green_fraction := 2 / 3
def initial_yellow_fraction := 1 / 3
def initial_green_marbles := (2 / 3 : ℚ) * x
def initial_yellow_marbles := (1 / 3 : ℚ) * x
def tripled_yellow_marbles := 3 * initial_yellow_marbles
def new_total_marbles := initial_green_marbles + tripled_yellow_marbles

-- Proof statement
theorem yellow_fraction_after_tripling (total_marbles : ℕ) :
    new_total_marbles = (2 / 3 : ℚ) * total_marbles + 3 * ((1 / 3 : ℚ) * total_marbles) →
    (tripled_yellow_marbles / new_total_marbles) = (3 / 5 : ℚ) :=
by
  sorry

end yellow_fraction_after_tripling_l428_428715


namespace smallest_multiple_3_4_5_l428_428886

theorem smallest_multiple_3_4_5 : ∃ (n : ℕ), (∀ (m : ℕ), (m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 0) → n ≤ m) ∧ n = 60 := 
sorry

end smallest_multiple_3_4_5_l428_428886


namespace num_rectangles_in_5x5_grid_l428_428626

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428626


namespace rectangle_perimeter_gt_16_l428_428173

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
  sorry

end rectangle_perimeter_gt_16_l428_428173


namespace pirate_overtakes_merchant_at_8pm_l428_428457

noncomputable def initial_time := 14 -- 2:00 p.m. in 24-hour notation
noncomputable def initial_merchant_advance := 15 -- miles
noncomputable def pirate_speed_initial := 14 -- mph
noncomputable def merchant_speed_initial := 10 -- mph
noncomputable def time_of_speed_change := 3 -- hours
noncomputable def pirate_speed_after_change := 12 -- mph
noncomputable def merchant_speed_after_change := 11 -- mph

theorem pirate_overtakes_merchant_at_8pm :
  let time_of_overtake := initial_time + time_of_speed_change + (3: ℕ), -- 8:00 p.m. in 24-hour notation
    time_of_overtake = 20 := -- 20 in 24-hour notation corresponds to 8:00 p.m.
by
  sorry

end pirate_overtakes_merchant_at_8pm_l428_428457


namespace c_values_for_one_vertical_asymptote_l428_428146

noncomputable def has_one_vertical_asymptote (c : ℝ) : Prop :=
  let numerator := (X^2 + 3 * X + C : Polynomial ℝ)
  let denominator := (X^2 - 3 * X - 10 : Polynomial ℝ)
  (numerator.eval 5 = 0 ∧ numerator.eval (-2) ≠ 0) ∨ 
  (numerator.eval (-2) = 0 ∧ numerator.eval 5 ≠ 0)

theorem c_values_for_one_vertical_asymptote :
  ∀ c : ℝ, has_one_vertical_asymptote c ↔ c = -40 ∨ c = 2 :=
by
  -- Proof goes here (sorry leaves the proof incomplete intentionally)
  sorry

end c_values_for_one_vertical_asymptote_l428_428146


namespace count_rectangles_5x5_l428_428685

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428685


namespace fraction_to_decimal_l428_428058

theorem fraction_to_decimal : (17 : ℝ) / 50 = 0.34 := 
by 
  sorry

end fraction_to_decimal_l428_428058


namespace common_chord_equation_l428_428358

-- Definitions of the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 12 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 14*y + 15 = 0

-- Definition of the common chord line
def common_chord_line (x y : ℝ) : Prop := 6*x + 8*y - 3 = 0

-- The theorem to be proved
theorem common_chord_equation :
  (∀ x y, circle1 x y → circle2 x y → common_chord_line x y) :=
by sorry

end common_chord_equation_l428_428358


namespace calculate_expression_l428_428487

-- Define the conditions
def a := log 3 2
def b := log 10 (1 / 2)
def c := log 10 5

-- Statement of the problem
theorem calculate_expression :
  3 ^ a + b - c = 1 := sorry

end calculate_expression_l428_428487


namespace inscribed_square_area_l428_428078

theorem inscribed_square_area : 
  ∃ (s : ℝ), y = x^2 - 10 * x + 21 ∧ (vertex_of_parabola R lies_on parabola) → 
  (2 * s)^2 = 24 - 8 * √5 := 
by
  sorry

end inscribed_square_area_l428_428078


namespace problem_statement_l428_428998

theorem problem_statement (p q : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) (r s : ℕ)
  (consecutive_primes : Nat.Prime r ∧ Nat.Prime s ∧ (r + 1 = s ∨ s + 1 = r))
  (roots_condition : r + s = p ∧ r * s = 2 * q) :
  (r * s = 2 * q) ∧ (Nat.Prime (p^2 - 2 * q)) ∧ (Nat.Prime (p + 2 * q)) :=
by
  sorry

end problem_statement_l428_428998


namespace num_rectangles_in_5x5_grid_l428_428627

def count_rectangles (n : ℕ) : ℕ :=
  let choose2 := n * (n - 1) / 2
  choose2 * choose2

theorem num_rectangles_in_5x5_grid : count_rectangles 5 = 100 :=
  sorry

end num_rectangles_in_5x5_grid_l428_428627


namespace problem1_problem2_l428_428486

theorem problem1 : 5^(2 * log 5 3) + log 4 32 - log 3 (log 2 8) = 21 / 2 := sorry

theorem problem2 : 0.027^(-1 / 3) - (-1 / 6)^(-2) + 256^(0.75) + (1 / (sqrt 3 - 1))^0 = 97 / 3 := sorry

end problem1_problem2_l428_428486


namespace susan_reading_hours_l428_428416

/--
Given:
- The ratio of time spent on swimming, reading, and hanging out with friends is 1:4:10,
- Susan hung out with her friends for 20 hours,

Prove:
- Susan spent 8 hours reading.
-/
theorem susan_reading_hours 
  (ratio_swim_reading_friends: ℕ × ℕ × ℕ) 
  (hours_friends: ℕ) 
  (hours_reading: ℕ) :
  ratio_swim_reading_friends = (1, 4, 10) → hours_friends = 20 → (hours_reading / hours_friends = 4 / 10) → hours_reading = 8 :=
by
  intros h_ratio h_hours_friends h_proportion
  have : hours_reading = (4 / 10) * hours_friends := 
    by sorry
  rw [h_hours_friends] at this
  have : (4 / 10) * 20 = 8 := 
    by sorry
  rw [this] at this
  exact this

end susan_reading_hours_l428_428416


namespace Y_on_median_BM_l428_428305

variables {A B C M Y : Point}
variables {omega1 omega2 : Circle}

-- Definitions of points and circles intersection
def points_on_same_side (Y B : Point) (lineAC : Line) : Prop :=
-- Here should be the appropriate condition that defines when Y and B are on the same side of the line.
sorry

-- The main theorem
theorem Y_on_median_BM (h1 : Y ∈ (omega1 ∩ omega2)) 
(h2 : points_on_same_side Y B (line_through A C)) : 
lies_on Y (line_through B M) :=
sorry

end Y_on_median_BM_l428_428305


namespace length_CM_sqrt_31_25_l428_428348

theorem length_CM_sqrt_31_25 :
  ∀ (A B C D M N : ℝ) (side : ℝ) (area : ℝ) (CM CN : ℝ),
  side = 5 ∧
  area = 25 ∧
  CMCN_dvs_square_to_four := (CMCN_dvs_square_to_four side CM CN) ∧
  M_lies_on_AB := (M_lies_on_AB side M) ∧
  N_lies_on_AD := (N_lies_on_AD side N) →
  CM = Real.sqrt 31.25 := 
begin
  sorry
end

end length_CM_sqrt_31_25_l428_428348


namespace probability_at_least_one_equals_four_l428_428875

theorem probability_at_least_one_equals_four :
  ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 ∧
    (a = 4 ∨ a + b = 4 ∨ a + b + c = 4 ∨ a + b + c + d = 4) →
      (∑ (i ∈ {a, a + b, a + b + c, a + b + c + d}) (h : i = 4), 1 / 6^4) = 343 / 1296 := 
sorry

end probability_at_least_one_equals_four_l428_428875


namespace quotient_in_base5_correct_l428_428127

-- Define the conversion from base 5 to base 10
def base5_to_base10 (d : ℕ → ℕ) : ℕ :=
  d 0 * 5^0 + d 1 * 5^1 + d 2 * 5^2 + d 3 * 5^3

-- Specific values for 1302_5 and 23_5 in base 5
def d1 : ℕ → ℕ
| 0 := 2
| 1 := 0
| 2 := 3
| 3 := 1
| _ := 0

def d2 : ℕ → ℕ
| 0 := 3
| 1 := 2
| _ := 0

-- Convert to base 10
def n1 := base5_to_base10 d1 -- 1302_5 in base 10
def n2 := base5_to_base10 d2 -- 23_5 in base 10

-- Perform the division in base 10 and truncate to integer
def quotient_base10 := n1 / n2

-- Convert the quotient 15 (base 10) to base 5
def quotient_base5 (q : ℕ) : ℕ → ℕ
| 0 := (q % 5)
| 1 := (q / 5)
| _ := 0

-- Specific expected base 5 quotient
def expected_quotient_base5 : ℕ → ℕ
| 0 := 0
| 1 := 3
| _ := 0

-- Prove the quotient in base 5
theorem quotient_in_base5_correct : quotient_base5 quotient_base10 = expected_quotient_base5 := 
by
  sorry

end quotient_in_base5_correct_l428_428127


namespace parabola_addition_l428_428454

def f (a b c x : ℝ) : ℝ := a * x^2 - b * (x + 3) + c
def g (a b c x : ℝ) : ℝ := a * x^2 + b * (x - 4) + c

theorem parabola_addition (a b c x : ℝ) : 
  (f a b c x + g a b c x) = (2 * a * x^2 + 2 * c - 7 * b) :=
by
  sorry

end parabola_addition_l428_428454


namespace count_rectangles_5x5_l428_428677

/-- Number of rectangles in a 5x5 grid with sides parallel to the grid -/
theorem count_rectangles_5x5 : 
  let n := 5 
  in (nat.choose n 2) * (nat.choose n 2) = 100 :=
by
  sorry

end count_rectangles_5x5_l428_428677


namespace basketball_team_games_l428_428860

def total_games_played (G : ℕ) : Prop :=
  ∃ R : ℕ, 
    (0.50 * 40 + 0.60 * 30 + 0.85 * R = 0.62 * (70 + R)) ∧
    (G = 70 + R)

theorem basketball_team_games : 
  total_games_played 93 :=
sorry

end basketball_team_games_l428_428860


namespace investment_schemes_l428_428061

theorem investment_schemes :
  let n_projects := 3
  let n_cities := 4
  (∃ schemes : ℕ,
    -- Case 1: 1 project in one city and 2 projects in another city
    let case1 := (comb n_projects 1) * (perm n_cities 2) in
    -- Case 2: 1 project in each of 3 cities
    let case2 := (perm n_cities 3) in
    -- Total number of schemes
    schemes = case1 + case2) →
  schemes = 60 :=
by
  sorry

end investment_schemes_l428_428061


namespace third_side_length_is_six_l428_428798

-- Defining the lengths of the sides of the triangle
def side1 : ℕ := 2
def side2 : ℕ := 6

-- Defining that the third side is an even number between 4 and 8
def is_even (x : ℕ) : Prop := x % 2 = 0
def valid_range (x : ℕ) : Prop := 4 < x ∧ x < 8

-- Stating the theorem
theorem third_side_length_is_six (x : ℕ) (h1 : is_even x) (h2 : valid_range x) : x = 6 :=
by
  sorry

end third_side_length_is_six_l428_428798


namespace sixteen_sided_polygon_ABDM_area_l428_428113

theorem sixteen_sided_polygon_ABDM_area
  (polygon : Fin 16 → ℝ × ℝ)
  (h_side_length : ∀ i, dist (polygon i) (polygon ((i + 1) % 16)) = 5)
  (h_right_angle : ∀ i, angle (polygon i) (polygon ((i + 1) % 16)) (polygon ((i + 2) % 16)) = π / 2)
  (A G D J M : ℝ × ℝ)
  (h_AG : A = polygon 0 ∧ G = polygon 6)
  (h_DJ : D = polygon 3 ∧ J = polygon 9)
  (h_intersection : M = intersection (line_from_to A G) (line_from_to D J)) :
  area_of_quadrilateral A (polygon 1) D M = 40.625 :=
sorry

end sixteen_sided_polygon_ABDM_area_l428_428113


namespace probability_increasing_function_correct_l428_428996

def is_increasing (a b : ℤ) (f : ℝ → ℝ) : Prop :=
  ∀ x > 1, 2 * a * x - 2 * b > 0

noncomputable def probability_increasing_function : ℚ :=
  sorry

theorem probability_increasing_function_correct :
  (∀ a ∈ {0, 1, 2}, ∀ b ∈ {-1, 1, 3, 5}, is_increasing a b (λ x, a * x^2 - 2 * b * x)) →
  probability_increasing_function = 1 / 3 :=
sorry

end probability_increasing_function_correct_l428_428996


namespace at_most_six_acute_angle_vectors_l428_428342

theorem at_most_six_acute_angle_vectors (V : Finset (EuclideanSpace ℝ (Fin 3))) : 
  (∀ (v1 v2 ∈ V), v1 ≠ v2 → inner v1 v2 > 0) → V.card ≤ 6 := 
by sorry

end at_most_six_acute_angle_vectors_l428_428342


namespace num_rectangles_grid_l428_428614

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428614


namespace weight_of_4_moles_CaBr2_l428_428401

theorem weight_of_4_moles_CaBr2 :
  let atomic_weight_Ca := 40.08
  let atomic_weight_Br := 79.904
  let moles_of_CaBr2 := 4
  let molecular_weight_CaBr2 := atomic_weight_Ca + 2 * atomic_weight_Br
  weight := moles_of_CaBr2 * molecular_weight_CaBr2 
in weight = 799.552 := by
   let atomic_weight_Ca := 40.08
   let atomic_weight_Br := 79.904
   let moles_of_CaBr2 := 4
   let molecular_weight_CaBr2 := atomic_weight_Ca + 2 * atomic_weight_Br
   let weight := moles_of_CaBr2 * molecular_weight_CaBr2 
   have : weight = 799.552 := by sorry
   exact this

end weight_of_4_moles_CaBr2_l428_428401


namespace find_slope_l428_428877

noncomputable def slope_of_first_line
    (m : ℝ)
    (intersect_point : ℝ × ℝ)
    (slope_second_line : ℝ)
    (x_intercept_distance : ℝ) 
    : Prop :=
  let (x₀, y₀) := intersect_point
  let x_intercept_first := (40 * m - 30) / m
  let x_intercept_second := 35
  abs (x_intercept_first - x_intercept_second) = x_intercept_distance

theorem find_slope : ∃ m : ℝ, slope_of_first_line m (40, 30) 6 10 :=
by
  use 2
  sorry

end find_slope_l428_428877


namespace quadratic_negative_root_condition_l428_428792

theorem quadratic_negative_root_condition (a : ℝ) (h : a < 0) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) → (some_condition a) :=
sorry

end quadratic_negative_root_condition_l428_428792


namespace largest_tile_size_l428_428882

def length_cm : ℕ := 378
def width_cm : ℕ := 525

theorem largest_tile_size :
  Nat.gcd length_cm width_cm = 21 := by
  sorry

end largest_tile_size_l428_428882


namespace cosine_angle_st_qr_l428_428821

variables {P Q R S T : Type*}
variables [inner_product_space ℝ (P)] [inner_product_space ℝ (Q)] [inner_product_space ℝ (R)]
variables [inner_product_space ℝ (S)] [inner_product_space ℝ (T)]
variables (PQ PR QS QT ST QR : ℝ)

-- Given Conditions
def midpoint_Q (ST : ℝ) := 
  Q = ½(S + T)

axiom PQ_condition : PQ = 1
axiom ST_condition : ST = 1
axiom QR_condition : QR = 8
axiom PR_condition : PR = √65
axiom dot_product_condition : ⟪PQ, PS⟫ + ⟪PR, PT⟫ = 4

-- Theorem Statement (Proof not included)
theorem cosine_angle_st_qr (PQ PR QS QT ST QR : ℝ) :
  ∃ φ, cos φ = ¾ :=
sorry

end cosine_angle_st_qr_l428_428821


namespace find_line_eqn_l428_428563

-- Definitions of given points and condition
def A : ℝ × ℝ := (-Real.sqrt 2, 0)
def B : ℝ × ℝ := (Real.sqrt 2, 0)

def slope (P Q : ℝ × ℝ) : ℝ := (Q.snd - P.snd) / (Q.fst - P.fst)

-- Condition: product of slopes is -1/2
def condition (P : ℝ × ℝ) : Prop :=
  slope P A * slope P B = -1 / 2

-- Derived trajectory equation
def trajectory_eqn (P : ℝ × ℝ) : Prop :=
  P.fst ^ 2 / 2 + P.snd ^ 2 = 1

-- Line l definition
def line_l (k : ℝ) (x y : ℝ) : ℝ :=
  y = k * x + 1

-- Deriving line equation using length MN and intersecting curve C
def curve_C (x y : ℝ) : Prop :=
  x ^ 2 / 2 + y ^ 2 = 1

theorem find_line_eqn (k x1 x2 y1 y2: ℝ) (hk : k ≠ 0)
  (hM : line_l k x1 y1) (hN : line_l k x2 y2)
  (hC1 : curve_C x1 y1) (hC2 : curve_C x2 y2)
  (hMN : Real.sqrt (1 + k ^ 2) * Real.sqrt ((x1 + x2)^2 - 4 * (x1 * x2)) = 4 * Real.sqrt 2 / 3) :
  x1 = 0 ∧ y1 = 1 ∨ x2 = 0 ∧ y2 = 1 → k = 1 ∨ k = -1 :=
begin
  sorry
end

end find_line_eqn_l428_428563


namespace point_Y_on_median_BM_l428_428300

variables {A B C M Y : Type} -- Points in geometry
variables (ω1 ω2 : set Type) -- Circles defined as sets of points

-- Definitions for intersection and symmetry conditions
def intersects (ω1 ω2 : set Type) (y : Type) : Prop := y ∈ ω1 ∧ y ∈ ω2

def same_side (A B C : Type) (Y : Type) : Prop := -- geometric definition that Y and B are on the same side of line AC
  sorry

def median (B M : Type) : set Type := -- geometric construction of median BM
  sorry 

def lies_on_median (Y : Type) (B M : Type) : Prop :=
  Y ∈ median B M

theorem point_Y_on_median_BM
  (h1 : intersects ω1 ω2 Y)
  (h2 : same_side A B C Y) :
  lies_on_median Y B M :=
sorry

end point_Y_on_median_BM_l428_428300


namespace wire_goes_around_field_l428_428517

theorem wire_goes_around_field :
  (7348 / (4 * Real.sqrt 27889)) = 11 :=
by
  sorry

end wire_goes_around_field_l428_428517


namespace exists_sum_of_ten_distinct_numbers_l428_428334

theorem exists_sum_of_ten_distinct_numbers :
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (x ∈ s) (y ∈ s), x ≠ y) ∧
      (∃ (t₅ : Finset ℕ), t₅.card = 3 ∧ (∀ x ∈ t₅, x ∣ 5) ∧ t₅ ⊆ s) ∧
      (∃ (t₄ : Finset ℕ), t₄.card = 4 ∧ (∀ x ∈ t₄, x ∣ 4) ∧ t₄ ⊆ s) ∧
      (s.sum id < 75) :=
by
  -- The proof goes here.
  sorry

end exists_sum_of_ten_distinct_numbers_l428_428334


namespace chord_bisected_eq_line_l428_428199

-- Definitions for the given ellipse and the bisected point
def ellipse (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 4 = 1
def midpoint (x y : ℝ) : Prop := x = 2 ∧ y = 1

-- The line equation we aim to prove
def line_eq (x y : ℝ) : Prop := x + 2 * y - 4 = 0

-- The main theorem to be proved
theorem chord_bisected_eq_line :
  ∃ (A B : ℝ × ℝ), (ellipse A.1 A.2) ∧ (ellipse B.1 B.2) ∧ midpoint ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧ 
  ∀ (x y : ℝ), (line_eq x y) :=
  sorry

end chord_bisected_eq_line_l428_428199


namespace composite_n_factorable_l428_428758

noncomputable def is_n_factorable (n : ℕ) (a : ℕ) : Prop :=
  a > 2 ∧ ∀ (d : ℕ), d ∣ n ∧ d ≠ n → (a^n - 2^n) % (a^d + 2^d) = 0

theorem composite_n_factorable:
  ∀ (n : ℕ), (∃ (a : ℕ), is_n_factorable n a) ↔ (∃ m : ℕ, m > 1 ∧ n = 2^m) :=
begin
  sorry
end

end composite_n_factorable_l428_428758


namespace first_month_sale_l428_428450

theorem first_month_sale 
  (sales4 : ℕ) (sale1 sale2 sale3 sale4 : ℕ) (average_sale : ℕ) (sale6 : ℕ) 
  (h_sales4 : sales4 = sale1 + sale2 + sale3 + sale4)
  (h_avg : average_sale = 6500)
  (h_sale1 : sale1 = 6927)
  (h_sale2 : sale2 = 6855)
  (h_sale3 : sale3 = 7230)
  (h_sale4 : sale4 = 6562)
  (h_sale6 : sale6 = 4691) :
  let total_sales_required := average_sale * 6 in
  let total_sales_4months := sale1 + sale2 + sale3 + sale4 in
  let total_sales_5months := total_sales_4months + sale6 in
  let first_month := total_sales_required - total_sales_5months in
  first_month = 6735 := 
by
  sorry

end first_month_sale_l428_428450


namespace jog_to_walk_ratio_l428_428949

-- Definitions from conditions
def total_time : ℝ := 21 / 60  -- in hours
def walk_time : ℝ := 9 / 60   -- in hours
def walk_speed : ℝ := 4       -- in km/h
def jog_speed : ℝ := 8        -- in km/h
def jog_time : ℝ := total_time - walk_time

-- Definitions to calculate distances
def walk_distance : ℝ := walk_speed * walk_time
def jog_distance : ℝ := jog_speed * jog_time

-- Statement of the problem to prove
theorem jog_to_walk_ratio : jog_distance / walk_distance = 8 / 3 :=
by
  -- This proof is omitted here; just the statement is required.
  sorry

end jog_to_walk_ratio_l428_428949


namespace point_Y_lies_on_median_l428_428312

-- Define the geometric points and circles
variable (A B C M Y : Point)
variable (ω1 ω2 : Circle)

-- Definitions of the given conditions
variable (P : Point) (hP : P ∈ (ω1)) (hInt : ω1 ∩ ω2 = {Y})

-- Express conditions in terms of Lean definitions
variable (hSameSide : same_side Y B (line_through A C))
variable (hMedian : M = (midpoint A C))
variable (hBM : is_median B M)

-- The theorem that we need to prove
theorem point_Y_lies_on_median :
  Y ∈ line_through B M :=
sorry

end point_Y_lies_on_median_l428_428312


namespace distribution_of_X_maximize_expected_score_l428_428054
open ProbabilityTheory

noncomputable def prob_X_eq_0 (p_A_correct : ℝ) : ℝ := 1 - p_A_correct
noncomputable def prob_X_eq_20 (p_A_correct p_B_correct : ℝ) : ℝ := p_A_correct * (1 - p_B_correct)
noncomputable def prob_X_eq_100 (p_A_correct p_B_correct : ℝ) : ℝ := p_A_correct * p_B_correct

theorem distribution_of_X (p_A_correct p_B_correct : ℝ) 
    (hA : p_A_correct = 0.8) (hB : p_B_correct = 0.6) :
    ∃ p_X0 p_X20 p_X100 : ℝ, p_X0 = 0.2 ∧ p_X20 = 0.32 ∧ p_X100 = 0.48 :=
by 
    let X0 := prob_X_eq_0 0.8
    let X20 := prob_X_eq_20 0.8 0.6
    let X100 := prob_X_eq_100 0.8 0.6
    use [X0, X20, X100]
    simp [X0, X20, X100]
    sorry

noncomputable def expected_score_A (p_A_correct p_B_correct : ℝ) : ℝ :=
    (prob_X_eq_0 p_A_correct) * 0 + (prob_X_eq_20 p_A_correct p_B_correct) * 20 + (prob_X_eq_100 p_A_correct p_B_correct) * 100

noncomputable def prob_Y_eq_0 (p_B_correct : ℝ) : ℝ := 1 - p_B_correct
noncomputable def prob_Y_eq_80 (p_B_correct p_A_correct : ℝ) : ℝ := p_B_correct * (1 - p_A_correct)
noncomputable def prob_Y_eq_100 (p_B_correct p_A_correct : ℝ) : ℝ := p_B_correct * p_A_correct

noncomputable def expected_score_B (p_A_correct p_B_correct : ℝ) : ℝ :=
    (prob_Y_eq_0 p_B_correct) * 0 + (prob_Y_eq_80 p_B_correct p_A_correct) * 80 + (prob_Y_eq_100 p_B_correct p_A_correct) * 100

theorem maximize_expected_score (p_A_correct p_B_correct : ℝ) 
    (hA : p_A_correct = 0.8) (hB : p_B_correct = 0.6) :
    expected_score_B p_A_correct p_B_correct > expected_score_A p_A_correct p_B_correct :=
by 
    let E_X := expected_score_A 0.8 0.6
    let E_Y := expected_score_B 0.8 0.6
    simp [E_X, E_Y]
    sorry

end distribution_of_X_maximize_expected_score_l428_428054


namespace quadratic_roots_real_equal_l428_428522

theorem quadratic_roots_real_equal (m : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ a = 3 ∧ b = 2 - m ∧ c = 6 ∧
    (b^2 - 4 * a * c = 0)) ↔ (m = 2 - 6 * Real.sqrt 2 ∨ m = 2 + 6 * Real.sqrt 2) :=
by
  sorry

end quadratic_roots_real_equal_l428_428522


namespace probability_first_year_student_selected_l428_428387

-- Define the conditions as constants
def first_year_students : ℕ := 800
def second_year_students : ℕ := 600
def third_year_students : ℕ := 500
def sampled_third_year_students : ℕ := 25

-- Formulate the theorem to prove the probability of each first-year student being selected
theorem probability_first_year_student_selected :
  (sampled_third_year_students:ℚ / third_year_students:ℚ) = (1:ℚ / 20:ℚ) →
  (sampled_third_year_students:ℚ / third_year_students:ℚ) = (1:ℚ / 20:ℚ) :=
by
  sorry

end probability_first_year_student_selected_l428_428387


namespace perimeter_gt_sixteen_l428_428176

theorem perimeter_gt_sixteen (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 :=
by
  sorry

end perimeter_gt_sixteen_l428_428176


namespace volume_of_one_piece_l428_428884

open Real

noncomputable def pizza_thickness : ℝ := 1 / 2
noncomputable def pizza_diameter : ℝ := 18
noncomputable def pizza_radius : ℝ := pizza_diameter / 2
noncomputable def number_of_pieces : ℕ := 16
noncomputable def total_volume : ℝ := π * (pizza_radius ^ 2) * pizza_thickness

theorem volume_of_one_piece : 
  (total_volume / number_of_pieces) = (2.53125 * π) := by
  -- proof omitted
  sorry

end volume_of_one_piece_l428_428884


namespace probability_grade_A_branch_a_probability_grade_A_branch_b_average_profit_branch_a_average_profit_branch_b_select_branch_l428_428442

def frequencies_branch_a := (40, 20, 20, 20) -- (A, B, C, D)
def frequencies_branch_b := (28, 17, 34, 21) -- (A, B, C, D)

def fees := (90, 50, 20, -50)  -- (A, B, C, D respectively)
def processing_cost_branch_a := 25
def processing_cost_branch_b := 20

theorem probability_grade_A_branch_a :
  let (fa, fb, fc, fd) := frequencies_branch_a in
  (fa : ℝ) / 100 = 0.4 := by
  sorry

theorem probability_grade_A_branch_b :
  let (fa, fb, fc, fd) := frequencies_branch_b in
  (fa : ℝ) / 100 = 0.28 := by
  sorry

theorem average_profit_branch_a :
  let (fa, fb, fc, fd) := frequencies_branch_a in
  let (qa, qb, qc, qd) := fees in
  ((qa - processing_cost_branch_a) * (fa / 100) + 
   (qb - processing_cost_branch_a) * (fb / 100) +
   (qc - processing_cost_branch_a) * (fc / 100) +
   (qd - processing_cost_branch_a) * (fd / 100) : ℝ) = 15 := by
  sorry

theorem average_profit_branch_b :
  let (fa, fb, fc, fd) := frequencies_branch_b in
  let (qa, qb, qc, qd) := fees in
  ((qa - processing_cost_branch_b) * (fa / 100) + 
   (qb - processing_cost_branch_b) * (fb / 100) +
   (qc - processing_cost_branch_b) * (fc / 100) +
   (qd - processing_cost_branch_b) * (fd / 100) : ℝ) = 10 := by
  sorry

theorem select_branch :
  let profit_a := 15 in
  let profit_b := 10 in
  profit_a > profit_b → 
  "Branch A" = "Branch A" := by
  sorry

end probability_grade_A_branch_a_probability_grade_A_branch_b_average_profit_branch_a_average_profit_branch_b_select_branch_l428_428442


namespace chess_match_schedule_l428_428106

-- Define the problem conditions
def num_players_school := 4
def games_per_player := 2
def total_games := num_players_school * num_players_school * games_per_player
def games_per_round := 4
def num_rounds := total_games / games_per_round

-- Define the claim
theorem chess_match_schedule : num_rounds = 8 ∧ factorial 8 / factorial 2 = 20160 :=
by
  sorry

end chess_match_schedule_l428_428106


namespace probability_grade_A_branch_a_probability_grade_A_branch_b_average_profit_branch_a_average_profit_branch_b_select_branch_l428_428445

def frequencies_branch_a := (40, 20, 20, 20) -- (A, B, C, D)
def frequencies_branch_b := (28, 17, 34, 21) -- (A, B, C, D)

def fees := (90, 50, 20, -50)  -- (A, B, C, D respectively)
def processing_cost_branch_a := 25
def processing_cost_branch_b := 20

theorem probability_grade_A_branch_a :
  let (fa, fb, fc, fd) := frequencies_branch_a in
  (fa : ℝ) / 100 = 0.4 := by
  sorry

theorem probability_grade_A_branch_b :
  let (fa, fb, fc, fd) := frequencies_branch_b in
  (fa : ℝ) / 100 = 0.28 := by
  sorry

theorem average_profit_branch_a :
  let (fa, fb, fc, fd) := frequencies_branch_a in
  let (qa, qb, qc, qd) := fees in
  ((qa - processing_cost_branch_a) * (fa / 100) + 
   (qb - processing_cost_branch_a) * (fb / 100) +
   (qc - processing_cost_branch_a) * (fc / 100) +
   (qd - processing_cost_branch_a) * (fd / 100) : ℝ) = 15 := by
  sorry

theorem average_profit_branch_b :
  let (fa, fb, fc, fd) := frequencies_branch_b in
  let (qa, qb, qc, qd) := fees in
  ((qa - processing_cost_branch_b) * (fa / 100) + 
   (qb - processing_cost_branch_b) * (fb / 100) +
   (qc - processing_cost_branch_b) * (fc / 100) +
   (qd - processing_cost_branch_b) * (fd / 100) : ℝ) = 10 := by
  sorry

theorem select_branch :
  let profit_a := 15 in
  let profit_b := 10 in
  profit_a > profit_b → 
  "Branch A" = "Branch A" := by
  sorry

end probability_grade_A_branch_a_probability_grade_A_branch_b_average_profit_branch_a_average_profit_branch_b_select_branch_l428_428445


namespace convex_pentagon_possible_l428_428044

noncomputable def square : Type := {
  A1 A2 A3 A4 : Point,
  square A1 A2 A3 A4
}

noncomputable def convex_quad (S : square) : Type := {
  A5 A6 A7 A8 : Point,
  quadrilateral A5 A6 A7 A8,
  inside_quadrilateral A5 A6 A7 A8 S
}

noncomputable def point_inside_convex_quad (Q : convex_quad) : Type := {
  A9 : Point,
  inside_quadrilateral A9 Q
}

axiom no_three_collinear (P : list Point) : Prop :=
  ¬(∃ (A B C : Point), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ collinear A B C)

theorem convex_pentagon_possible
  (S : square)
  (Q : convex_quad S)
  (P : point_inside_convex_quad Q)
  (H : no_three_collinear [S.A1, S.A2, S.A3, S.A4, Q.A5, Q.A6, Q.A7, Q.A8, P.A9]) :
  ∃ (P1 P2 P3 P4 P5 : Point), convex_pentagon [P1, P2, P3, P4, P5] :=
sorry

end convex_pentagon_possible_l428_428044


namespace number_of_correct_statements_l428_428156

def line : Type := sorry
def plane : Type := sorry
def parallel (x y : line) : Prop := sorry
def perpendicular (x : line) (y : plane) : Prop := sorry
def subset (x : line) (y : plane) : Prop := sorry
def skew (x y : line) : Prop := sorry

variable (m n : line) -- two different lines
variable (alpha beta : plane) -- two different planes

theorem number_of_correct_statements :
  (¬parallel m alpha ∨ subset n alpha ∧ parallel m n) ∧
  (parallel m alpha ∧ perpendicular alpha beta ∧ perpendicular m n ∧ perpendicular n beta) ∧
  (subset m alpha ∧ subset n beta ∧ perpendicular m n) ∧
  (skew m n ∧ subset m alpha ∧ subset n beta ∧ parallel m beta ∧ parallel n alpha) :=
sorry

end number_of_correct_statements_l428_428156


namespace area_of_inscribed_triangle_l428_428896

noncomputable def calculate_triangle_area_inscribed_in_circle 
  (arc1 : ℝ) (arc2 : ℝ) (arc3 : ℝ) (total_circumference := arc1 + arc2 + arc3)
  (radius := total_circumference / (2 * Real.pi))
  (theta := (2 * Real.pi) / total_circumference)
  (angle1 := 5 * theta) (angle2 := 7 * theta) (angle3 := 8 * theta) : ℝ :=
  0.5 * (radius ^ 2) * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem area_of_inscribed_triangle : 
  calculate_triangle_area_inscribed_in_circle 5 7 8 = 119.85 / (Real.pi ^ 2) :=
by
  sorry

end area_of_inscribed_triangle_l428_428896


namespace wayne_shrimp_plan_l428_428827

theorem wayne_shrimp_plan (n_total_guests : ℕ) (cost_per_pound : ℝ) (shrimp_per_pound : ℕ) (total_spent : ℝ) (n_total_shrimp : ℕ) (shrimp_per_guest : ℕ) :
  n_total_guests = 40 →
  cost_per_pound = 17.0 →
  shrimp_per_pound = 20 →
  total_spent = 170.0 →
  n_total_shrimp = (total_spent / cost_per_pound) * shrimp_per_pound →
  shrimp_per_guest = n_total_shrimp / n_total_guests →
  shrimp_per_guest = 5 :=
begin
  sorry
end

end wayne_shrimp_plan_l428_428827


namespace no_repeat_color_in_1x1201_rectangles_l428_428804

theorem no_repeat_color_in_1x1201_rectangles 
  (coloring : ℤ × ℤ → fin 1201)
  (h_no_repeat : ∀ (x1 y1 x2 y2 : ℤ), (abs (x2 - x1) + abs (y2 - y1) = 50) → (coloring (x1, y1) ≠ coloring (x2, y2))) :
  ∀ (x y : ℤ), 
    ∀ (dx : ℤ), (dx = 1 ∨ dx = 1201) → 
    ∀ (dy : ℤ), (dy = 1201 ∨ dy = 1) → 
    ∀ (i j : ℤ) (hi : 0 ≤ i ∧ i < dx) (hj : 0 ≤ j ∧ j < dy),
      coloring (x + i, y + j) ≠ coloring (x + i + dx - 1, y + j + dy - 1) := 
sorry

end no_repeat_color_in_1x1201_rectangles_l428_428804


namespace proof_expression_value_l428_428698

theorem proof_expression_value (x y : ℝ) (h : x + 2 * y = 30) : 
  (x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3) = 16 := 
by 
  sorry

end proof_expression_value_l428_428698


namespace part_a_part_b_part_c_l428_428281

variable {G : Type*} [Group G] [Fintype G]
variable (H : Set G) (e : G)
variable [DecidableEq H]
variable [Fintype H]

-- Definitions from conditions
def isFiniteGroupOrderN (n : ℕ) : Prop := Fintype.card G = n

def setH := { x : G | x ^ 2 = e }

def cardinalityH (p : ℕ) := Fintype.card (setH)

def xH (x : G) := { y : G | ∃ h ∈ setH, y = x * h }

-- Questions rephrased:
-- Part a
theorem part_a (n p : ℕ) (x : G) [isFiniteGroupOrderN n] [cardinalityH p] :
  Fintype.card (setH ∩ (xH x)) ≥ 2 * p - n := sorry

-- Part b
theorem part_b (n p : ℕ) [isFiniteGroupOrderN n] [cardinalityH p] :
  p > 3 * n / 4 → isCommutative (G) := sorry

-- Part c
theorem part_c (n p : ℕ) [isFiniteGroupOrderN n] [cardinalityH p] :
  n / 2 < p ∧ p ≤ 3 * n / 4 → ¬ isCommutative (G) := sorry

end part_a_part_b_part_c_l428_428281


namespace solve_system_simplify_expression_l428_428855

section Problem1

variable (x : ℝ)

def Inequality1 := 2 * x - 1 < 5
def Inequality2 := (4 - x) / 2 ≥ 1

theorem solve_system : Inequality1 x → Inequality2 x → x ≤ 2 :=
sorry

end Problem1


section Problem2

variable (a : ℝ)
def expr := (1 / (a - 1)) / ((a^2 + a) / (a^2 - 1) + 1 / (a - 1))
def a_value := (-2023)^0 + (1 / 2) ^ (-1)
def simplified_expr := 1 / (a + 1)

theorem simplify_expression : a = a_value → expr a = 1 / 4 :=
sorry

end Problem2

end solve_system_simplify_expression_l428_428855


namespace quadratic_inequality_solution_l428_428205

-- Definitions based on the conditions
def roots_eq (a b : ℝ) : Prop := 
  ∀ x : ℝ, x^2 - a * x - b = 0 ↔ (x = 2 ∨ x = 3)

def solution_set_eq (a b : ℝ) : Set ℝ :=
  {x : ℝ | bx^2 - ax - 1 > 0}

-- The proof statement
theorem quadratic_inequality_solution (a b : ℝ) (h : roots_eq a b) : 
  solution_set_eq a b = {x : ℝ | - (1/2:ℝ) < x ∧ x < - (1/3:ℝ)} :=
by
  sorry

end quadratic_inequality_solution_l428_428205


namespace number_of_rectangles_in_5x5_grid_l428_428659

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428659


namespace S_2011_l428_428265

open Nat

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, 0 < n → a (n + 1) + a n = 1

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n + 1, a i.succ

theorem S_2011 (a : ℕ → ℝ) (h : seq a) : S a 2011 = 1007 := 
by
  sorry

end S_2011_l428_428265


namespace smallest_positive_x_for_maximum_sine_sum_l428_428110

theorem smallest_positive_x_for_maximum_sine_sum :
  ∃ x : ℝ, (0 < x) ∧ (∃ k m : ℕ, x = 450 + 1800 * k ∧ x = 630 + 2520 * m ∧ x = 12690) := by
  sorry

end smallest_positive_x_for_maximum_sine_sum_l428_428110


namespace find_m_probability_l428_428786

theorem find_m_probability (m : ℝ) (ξ : ℕ → ℝ) :
  (ξ 1 = m * (2/3)) ∧ (ξ 2 = m * (2/3)^2) ∧ (ξ 3 = m * (2/3)^3) ∧ 
  (ξ 1 + ξ 2 + ξ 3 = 1) → 
  m = 27 / 38 := 
sorry

end find_m_probability_l428_428786


namespace distance_PQ_l428_428326

noncomputable theory
open set

def point (x y : ℝ) : ℝ × ℝ := (x, y)

def line1 (P : ℝ × ℝ) (θ : ℝ) : set (ℝ × ℝ) :=
  { Q | ∃ t : ℝ, Q = (P.1 + t * cos θ, P.2 + t * sin θ) }

def line2 (x y : ℝ) : set (ℝ × ℝ) :=
  { Q | Q.1 - 2 * Q.2 + 11 = 0 }

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem distance_PQ : 
  ∀ P Q : ℝ × ℝ,
  P = (3, 2) →
  ∃ θ θ' : ℝ, θ = real.arctan (3 / 4) ∧ θ' = π/2 - θ ∧
  Q ∈ line1 P θ ∧ Q ∈ line2 x y →
  distance P Q = 25 :=
begin
  intros P Q hP hQl1 hQl2,
  sorry
end

end distance_PQ_l428_428326


namespace number_of_new_students_l428_428260

theorem number_of_new_students (initial_students end_students students_left : ℕ) 
  (h_initial: initial_students = 33) 
  (h_left: students_left = 18) 
  (h_end: end_students = 29) : 
  initial_students - students_left + (end_students - (initial_students - students_left)) = 14 :=
by
  sorry

end number_of_new_students_l428_428260


namespace midpoint_correct_l428_428219

def A := (-3, 1, 5)
def B := (4, 3, 1)
def midpoint (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2, (P.3 + Q.3) / 2)

theorem midpoint_correct :
  midpoint A B = (1 / 2, 2, 3) :=
by
  sorry

end midpoint_correct_l428_428219


namespace solution_set_inequality_l428_428981

theorem solution_set_inequality (t : ℝ) (ht : 0 < t ∧ t < 1) :
  {x : ℝ | x^2 - (t + t⁻¹) * x + 1 < 0} = {x : ℝ | t < x ∧ x < t⁻¹} :=
sorry

end solution_set_inequality_l428_428981


namespace Y_lies_on_median_BM_l428_428291

variable {Ω1 Ω2 : Type}
variable {A B C M : Ω2}
variable [EuclideanGeometry Ω2]

-- Definitions coming from conditions
variable (Y : Ω2)
variable (hY1 : Y ∈ circle_omega1) (hY2 : Y ∈ circle_omega2)
variable (hSameSide : SameSide Y B (Line AC))

-- The theorem we want to prove
theorem Y_lies_on_median_BM :
  LiesOnMedian Y B M := 
  sorry

end Y_lies_on_median_BM_l428_428291


namespace sum_of_values_g_eq_zero_l428_428323

def g (x : ℝ) : ℝ :=
if x ≤ 2 then -x - 5 else x / 3 + 2

theorem sum_of_values_g_eq_zero : ∑ x in (set_of (λ x, g x = 0)), x = -5 :=
sorry

end sum_of_values_g_eq_zero_l428_428323


namespace manufacturers_price_l428_428455

theorem manufacturers_price (M : ℝ) 
  (h1 : 0.1 ≤ 0.3) 
  (h2 : 0.2 = 0.2) 
  (h3 : 0.56 * M = 25.2) : 
  M = 45 := 
sorry

end manufacturers_price_l428_428455


namespace calculate_expression_l428_428097

theorem calculate_expression :
  12 * 11 + 7 * 8 - 5 * 6 + 10 * 4 = 198 :=
by
  sorry

end calculate_expression_l428_428097


namespace greatest_two_digit_prime_saturated_l428_428071

def prime_factors (n : ℕ) : List ℕ :=
  -- A stub for a function that returns the list of prime factors of n.
  sorry

def product_of_primes (l : List ℕ) : ℕ :=
  -- A stub for a function that returns the product of elements in the list l.
  sorry

theorem greatest_two_digit_prime_saturated :
  ∃ (g : ℕ), g = 98 ∧ 10 ≤ g ∧ g < 100 ∧
  product_of_primes (prime_factors g) < Real.sqrt g ∧
  ∀ (k : ℕ), 10 ≤ k ∧ k < 100 ∧
  product_of_primes (prime_factors k) < Real.sqrt k → k ≤ g :=
begin
  sorry
end

end greatest_two_digit_prime_saturated_l428_428071


namespace find_m_l428_428760

-- Definition of the constraints and the values of x and y that satisfy them
def constraint1 (x y : ℝ) : Prop := x + y - 2 ≥ 0
def constraint2 (x y : ℝ) : Prop := x - y + 1 ≥ 0
def constraint3 (x : ℝ) : Prop := x ≤ 3

-- Given conditions
def satisfies_constraints (x y : ℝ) : Prop := 
  constraint1 x y ∧ constraint2 x y ∧ constraint3 x

-- The objective to prove
theorem find_m (x y m : ℝ) (h : satisfies_constraints x y) : 
  (∀ x y, satisfies_constraints x y → (- 3 = m * x + y)) → m = -2 / 3 :=
by
  sorry

end find_m_l428_428760


namespace largest_reciprocal_l428_428407

def given_numbers := [5 / 6, 1 / 2, 3, 8 / 3, 240]
def reciprocal (x : ℝ) : ℝ := 1 / x

theorem largest_reciprocal :
  let smallest := given_numbers.min (by apply_instance)
  smallest = 1 / 2 ∧ reciprocal smallest = 2 :=
by
  sorry

end largest_reciprocal_l428_428407


namespace trees_died_more_than_survived_l428_428909

theorem trees_died_more_than_survived (initial_trees died_trees : ℕ) 
  (h1 : initial_trees = 3) 
  (h2 : died_trees = 13) : 
  died_trees - initial_trees = 10 :=
by
  rw [h1, h2]
  simp
  sorry

end trees_died_more_than_survived_l428_428909


namespace problem_proof_l428_428535

noncomputable def f : ℕ → ℝ
| n := sorry

theorem problem_proof :
  (∀ p q : ℕ, f(p + q) = f(p) * f(q)) ∧ f(1) = 3 →
  (∑ i in [1, 2, 3, 4, 5], (f(2 * i) + f(i * 2)) / f(i)) = 30 :=
by
  intro h
  sorry

end problem_proof_l428_428535


namespace probability_valid_pairings_l428_428130

theorem probability_valid_pairings (S : Finset (Nat × Nat)) (hS : S.card = 15) :
  let m := 209
  let n := 3120
  ∃ p : ℚ, p = (m : ℚ) / n ∧ m + n = 3329 :=
by
  -- We need to prove that the probability of valid pairings is 209/3120
  -- according to the described conditions.
  sorry

end probability_valid_pairings_l428_428130


namespace part1_part2_l428_428544

theorem part1 (m n : ℕ) (h1 : m > n) (h2 : Nat.gcd m n + Nat.lcm m n = m + n) : n ∣ m := 
sorry

theorem part2 (m n : ℕ) (h1 : m > n) (h2 : Nat.gcd m n + Nat.lcm m n = m + n)
(h3 : m - n = 10) : (m, n) = (11, 1) ∨ (m, n) = (12, 2) ∨ (m, n) = (15, 5) ∨ (m, n) = (20, 10) := 
sorry

end part1_part2_l428_428544


namespace num_rectangles_in_5x5_grid_l428_428601

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428601


namespace altitudes_intersect_at_one_point_l428_428523

-- Define Points and properties
variables (circle : Type) [MetricSpace circle] [ProperSpace circle]

variables (M N C A B P : circle)
variable (MN : set circle)
variables (MN' : MetricSpace.closed_ball M (dist M N))
variables (ABC : Tri A B P)
variables (Ceq : MetricSpace.dist A B = 1)
variables (MN_diam : MetricSpace.dist M N < MetricSpace.dist A B)
variables (MN_int : set.subset MN' MN)
variable (C_ceq : MetricSpace.dist A C = MetricSpace.dist B C)

-- Theorem about the intersection of altitudes
theorem altitudes_intersect_at_one_point :
  ∃ P : circle,
  ∀ A B C : circle, 
  (MetricSpace.dist A B = 1) → 
  (MetricSpace.dist A M ≠ MetricSpace.dist B N) → 
  Π (H : circle), 
  is_orthocenter H A B C → 
  MetricSpace.dist P H = 0 := sorry

end altitudes_intersect_at_one_point_l428_428523


namespace greatest_n_divisible_by_10_power_l428_428843

theorem greatest_n_divisible_by_10_power :
  ∃ n, (10! - 2 * (5! ^ 2)) % (10 ^ n) = 0 ∧ (10! - 2 * (5! ^ 2)) % (10 ^ (n + 1)) ≠ 0 :=
sorry

end greatest_n_divisible_by_10_power_l428_428843


namespace three_digit_integers_count_l428_428228

theorem three_digit_integers_count : 
  let digits := {0, 1, 2, 3, 4, 5}
  let combinations := (finset.powersetLen 3 digits)
  let valid_count :=
    (combinations.filter (λ c, 0 ∈ c)).card * 4 +
    (combinations.filter (λ c, 0 ∉ c)).card * 6
  (valid_count = 100) := 
by 
  let digits := {0, 1, 2, 3, 4, 5}
  let combinations := (finset.powersetLen 3 digits)
  let count_combinations_with_zero := 
    (combinations.filter (λ c, 0 ∈ c)).card
  
  have comb_with_zero : count_combinations_with_zero = 10 := sorry
  have comb_without_zero : (combinations.card - count_combinations_with_zero) = 10 := sorry
  
  have valid_with_zero := comb_with_zero * 4
  have valid_without_zero := comb_without_zero * 6
  
  have total_valid_count : valid_with_zero + valid_without_zero = 100 := sorry
  
  exact total_valid_count

end three_digit_integers_count_l428_428228


namespace problem_statement_l428_428343

def sine_of_angle_between_lines : Real :=
  let A := (0, 0)
  let B := (0, 2)
  let C := (4, 2)
  let D := (4, 0)
  let M := (2, 2) -- Midpoint of BC
  let N := (4, 1.5) -- Point on CD such that CN = 1/4 * CD
  let AM := Real.sqrt ((2 - 0) ^ 2 + (2 - 0) ^ 2)
  let AN := Real.sqrt ((4 - 0) ^ 2 + (1.5 - 0) ^ 2)
  let MN := Real.sqrt ((4 - 2) ^ 2 + (1.5 - 2) ^ 2)
  let cosθ := (AM ^ 2 + AN ^ 2 - MN ^ 2) / (2 * AM * AN)
  let sin2θ := 1 - cosθ ^ 2
  let sinθ := Real.sqrt sin2θ
  sinθ

theorem problem_statement : sine_of_angle_between_lines = Real.sqrt (103 / 544) := by
  sorry

end problem_statement_l428_428343


namespace small_pizza_slices_l428_428911

-- Definitions based on conditions
def large_pizza_slices : ℕ := 16
def num_large_pizzas : ℕ := 2
def num_small_pizzas : ℕ := 2
def total_slices_eaten : ℕ := 48

-- Statement to prove
theorem small_pizza_slices (S : ℕ) (H : num_large_pizzas * large_pizza_slices + num_small_pizzas * S = total_slices_eaten) : S = 8 :=
by
  sorry

end small_pizza_slices_l428_428911


namespace num_rectangles_grid_l428_428616

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428616


namespace min_swap_cost_to_original_l428_428383

noncomputable def swap_cost (x y : ℕ) : ℕ :=
2 * (abs (x - y))

theorem min_swap_cost_to_original (n : ℕ) (a : Fin n → ℕ) :
  ∃F : Fin n → Fin n, (∀i : Fin n, a (F i) = (i + 1)) ∧
  (∑ i, swap_cost (a i) (i + 1)) ≤ ∑ i, abs ((a i) - (i + 1)) := by {
  sorry
}

end min_swap_cost_to_original_l428_428383


namespace sum_series_eq_l428_428493

theorem sum_series_eq :
  (∑ a : ℕ in finset.range 1000, ∑ b : ℕ in finset.range 1000, ∑ c : ℕ in finset.range 1000, 
    if (1 ≤ a ∧ a < b ∧ b < c) then 1 / (3^a * 5^b * 7^c) else 0) = 1 / 21216 :=
by
  sorry

end sum_series_eq_l428_428493


namespace no_solution_eqn_l428_428238

theorem no_solution_eqn (m : ℝ) : (∀ x : ℝ, (m * (x + 1) - 5) / (2 * x + 1) ≠ m - 3) ↔ m = 6 := 
by
  sorry

end no_solution_eqn_l428_428238


namespace area_of_inscribed_triangle_l428_428897

noncomputable def area_inscribed_triangle (r : ℝ) (a b c : ℝ) : ℝ :=
  1 / 2 * r^2 * (Real.sin a + Real.sin b + Real.sin c)

theorem area_of_inscribed_triangle : 
  ∃ (r a b c : ℝ),
    a + b + c = 2 * π ∧
    r = 10 / π ∧
    a = 5 * (18 * π / 180) ∧
    b = 7 * (18 * π / 180) ∧
    c = 8 * (18 * π / 180) ∧
    area_inscribed_triangle r a b c = 119.84 / π^2 :=
begin
  sorry
end

end area_of_inscribed_triangle_l428_428897


namespace Rosy_l428_428498

/-- Prove that Rosy's current age is 8 years old, given the conditions. -/
theorem Rosy's_age (R : ℕ) -- Let R be Rosy's current age
  (h1 : ∀ R, David's current age = R + 12)
  (h2 : ∀ R, David's age in 4 years = 2 * (Rosy's age in 4 years)) :
  R = 8 :=
by
  sorry

end Rosy_l428_428498


namespace branch_A_grade_A_probability_branch_B_grade_A_probability_branch_A_average_profit_branch_B_average_profit_choose_branch_l428_428434

theorem branch_A_grade_A_probability : 
  let total_A := 100
  let grade_A_A := 40
  (grade_A_A / total_A) = 0.4 := by
  sorry

theorem branch_B_grade_A_probability : 
  let total_B := 100
  let grade_A_B := 28
  (grade_A_B / total_B) = 0.28 := by
  sorry

theorem branch_A_average_profit :
  let freq_A_A := 0.4
  let freq_A_B := 0.2
  let freq_A_C := 0.2
  let freq_A_D := 0.2
  let process_cost_A := 25
  let profit_A := (90 - process_cost_A) * freq_A_A + (50 - process_cost_A) * freq_A_B + (20 - process_cost_A) * freq_A_C + (-50 - process_cost_A) * freq_A_D
  profit_A = 15 := by
  sorry

theorem branch_B_average_profit :
  let freq_B_A := 0.28
  let freq_B_B := 0.17
  let freq_B_C := 0.34
  let freq_B_D := 0.21
  let process_cost_B := 20
  let profit_B := (90 - process_cost_B) * freq_B_A + (50 - process_cost_B) * freq_B_B + (20 - process_cost_B) * freq_B_C + (-50 - process_cost_B) * freq_B_D
  profit_B = 10 := by
  sorry

theorem choose_branch :
  let profit_A := 15
  let profit_B := 10
  profit_A > profit_B -> "Branch A"

end branch_A_grade_A_probability_branch_B_grade_A_probability_branch_A_average_profit_branch_B_average_profit_choose_branch_l428_428434


namespace verify_salary_problem_l428_428351

def salary_problem (W : ℕ) (S_old : ℕ) (S_new : ℕ := 780) (n : ℕ := 9) : Prop :=
  (W + S_old) / n = 430 ∧ (W + S_new) / n = 420 → S_old = 870

theorem verify_salary_problem (W S_old : ℕ) (h1 : (W + S_old) / 9 = 430) (h2 : (W + 780) / 9 = 420) : S_old = 870 :=
by {
  sorry
}

end verify_salary_problem_l428_428351


namespace marvin_birthday_next_thursday_l428_428330

-- Define the basic properties of leap years and weekday progression.
def is_leap (year : ℕ) : Prop := (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

-- Given conditions
def may27_weekday (year : ℕ) : ℕ :=
  let start_weekday : ℕ := 5 -- Friday in the year 2007
  let number_of_days := List.range (year - 2007)
  number_of_days.foldl (λ acc y =>
    if is_leap (y + 2007) then (acc + 2) % 7 else (acc + 1) % 7
  ) start_weekday

-- Theorem to be proven: The first occurrence after 2007 where May 27 is a Thursday.
theorem marvin_birthday_next_thursday : ∃ (year : ℕ), year > 2007 ∧ may27_weekday year = 4 ∧ year = 2017 := by
  have : ∀ y ∈ List.range (2017 - 2007), may27_weekday (y + 2007) ≠ 4 := by sorry
  have : may27_weekday 2017 = 4 := by sorry
  exact ⟨2017, Nat.lt_succ_self 2007, by rwa [←Nat.succ_pred_eq_of_pos, not_ne_iff, ←List.foldr_ext (· + ·)]⟩

end marvin_birthday_next_thursday_l428_428330


namespace red_triangles_l428_428332

theorem red_triangles (n : ℕ) (h : n > 1) :
  ∀ (points : Fin 2n → ℝ × ℝ), 
  (∀ i j k : Fin 2n, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬collinear points i points j points k) →
  ∃ red_segments : Fin (n^2+1) → (Fin 2n × Fin 2n),
  ∃ triangles (t : Fin n → (Fin 2n × Fin 2n × Fin 2n)),
  ∀ i : Fin n, 
    ∃ (a b c : Fin 2n), 
    (a, b) ∈ red_segments ∧ (b, c) ∈ red_segments ∧ (c, a) ∈ red_segments ∧
    t i = (a, b, c)∧
    ¬collinear points a points b points c := 
sorry

end red_triangles_l428_428332


namespace greg_total_cost_l428_428586

def total_purchase (p1 p2 p3 : ℝ) : ℝ :=
  p1 + p2 + p3

def apply_discount (total : ℝ) (discount_rate : ℝ) : ℝ :=
  total - (total * discount_rate)

def round_to_nearest_dollar (amount : ℝ) : ℤ :=
  Int.round amount

theorem greg_total_cost : 
  let p1 := 2.45
  let p2 := 7.60
  let p3 := 3.15
  let discount_rate := 0.10
  let total := round_to_nearest_dollar (apply_discount (total_purchase p1 p2 p3) discount_rate)
  total = 12 :=
by
  sorry

end greg_total_cost_l428_428586


namespace triangle_area_example_l428_428926

def point := (ℝ × ℝ)

def area_of_triangle (P Q R : point) : ℝ :=
  let base := abs (Q.1 - P.1)
  let height := abs (R.2 - P.2)
  (1 / 2) * base * height

theorem triangle_area_example : 
  let P : point := (-2, 2)
  let Q : point := (8, 2)
  let R : point := (6, -4)
  area_of_triangle P Q R = 30 :=
by
  -- proof goes here
  sorry

end triangle_area_example_l428_428926


namespace graph_shape_matches_diagram_4_l428_428366

noncomputable def func : ℝ → ℝ := λ x, 1 - abs (x - x^2)

theorem graph_shape_matches_diagram_4 :
  let diagram_4_shape := (λ x, if 0 ≤ x ∧ x ≤ 1 then 1 - x + x^2 else 1 + x - x^2)
  (∀ x : ℝ, func x = diagram_4_shape x) :=
sorry

end graph_shape_matches_diagram_4_l428_428366


namespace knights_and_liars_l428_428853

theorem knights_and_liars (n : ℕ) (hn : n > 1000) 
  (knights_liars_statement : ∀ i : ℤ, 0 ≤ i ∧ i < n → 
    (∃ K L : ℕ, K + L = n ∧ 
      (∀ i : ℤ, 0 ≤ i ∧ i < n → 
        let cw := (list.range 20).map (λ j, (i + j + 1) % n)
        let ccw := (list.range 20).map (λ j, (i - j - 1 + n) % n)
        (∀ j in cw, (j < K) ↔ (j < L)) ∨ (∀ j in ccw, (j < K) ↔ (j < L))
     ))) : n = 1020 :=
begin
  sorry
end

end knights_and_liars_l428_428853


namespace number_of_rectangles_in_5x5_grid_l428_428658

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l428_428658


namespace area_excluding_hole_l428_428064

open Polynomial

theorem area_excluding_hole (x : ℝ) : 
  ((x^2 + 7) * (x^2 + 5)) - ((2 * x^2 - 3) * (x^2 - 2)) = -x^4 + 19 * x^2 + 29 :=
by
  sorry

end area_excluding_hole_l428_428064


namespace num_rectangles_grid_l428_428617

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l428_428617


namespace count_two_digit_numbers_l428_428229

-- Define the set of available digits
def digits := {2, 3, 4}

-- Define a predicate for two-digit numbers with given conditions
def two_digit_numbers (a b : ℕ) : Prop := 
  a ∈ digits ∧ b ∈ digits ∧ a ≠ b

-- State the main theorem
theorem count_two_digit_numbers : 
  ∃ count : ℕ, count = 6 ∧ 
  (∀ a b, two_digit_numbers a b → ∃ x, x / 10 = a ∧ x % 10 = b) :=
sorry

end count_two_digit_numbers_l428_428229


namespace find_m_l428_428206

-- Define the predicate for the quadratic equation having exactly one positive integer solution
def quadratic_has_one_pos_int_solution (a b c : ℤ) : Prop :=
  let discriminant := b^2 - 4 * a * c in
  (discriminant ≥ 0) ∧ ∃ x : ℤ, (x > 0) ∧ (a * x^2 + b * x + c = 0)

-- State the problem for the specific equation given in the conditions
theorem find_m (m : ℤ) :
  quadratic_has_one_pos_int_solution 
    6 
    (2 * (m - 13)) 
    (12 - m) 
  → m = 8 :=
sorry

end find_m_l428_428206


namespace tetrahedron_is_regular_if_spheres_coincide_l428_428724

-- Defining the concept of a tetrahedron
structure Tetrahedron :=
(vertices : Fin 4 → ℝ × ℝ × ℝ)

-- Definition for a sphere
structure Sphere :=
(center : ℝ × ℝ × ℝ)
(radius : ℝ)

-- Conditions for the problem
-- A circumscribed sphere and a semiscribed sphere for a tetrahedron
def has_circumscribed_sphere (T : Tetrahedron) (S : Sphere) : Prop := sorry -- Definition omitted for simplicity
def has_semiscribed_sphere (T : Tetrahedron) (S : Sphere) : Prop := sorry -- Definition omitted for simplicity

-- Conditional statement and conclusion in Lean 4
theorem tetrahedron_is_regular_if_spheres_coincide (T : Tetrahedron) (S : Sphere) 
  (H1 : has_circumscribed_sphere(T, S)) 
  (H2 : has_semiscribed_sphere(T, S)) 
  (H3 : S.center = S.center):
  ∀ (a b c d : Fin 4), 
  (T.vertices a = T.vertices b → T.vertices a = T.vertices c → T.vertices a = T.vertices d → a = b ∧ b = c ∧ c = d) :=
sorry -- Proof omitted

end tetrahedron_is_regular_if_spheres_coincide_l428_428724


namespace geometric_log_sum_l428_428165

-- Define the geometric sequence and positive terms condition
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the condition a_5 * a_6 = 8
def geometric_sequence_condition (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ 0 < a 1 ∧ 0 < a 2 ∧ 0 < a 3 ∧ 0 < a 4 ∧ 
  0 < a 5 ∧ 0 < a 6 ∧ 0 < a 7 ∧ 0 < a 8 ∧ 0 < a 9 ∧ 0 < a 10 ∧ 
  a 5 * a 6 = 8

-- The theorem statement
theorem geometric_log_sum (a : ℕ → ℝ) (h : geometric_sequence_condition a) :
  ∑ i in finset.range 10, real.log (a (i + 1)) / real.log 2 = 15 :=
sorry

end geometric_log_sum_l428_428165


namespace regular_18gon_lines_rotational_symmetry_sum_l428_428460

def L : ℕ := 18
def R : ℕ := 20

theorem regular_18gon_lines_rotational_symmetry_sum : L + R = 38 :=
by 
  sorry

end regular_18gon_lines_rotational_symmetry_sum_l428_428460


namespace Y_on_median_BM_l428_428304

variables {A B C M Y : Point}
variables {omega1 omega2 : Circle}

-- Definitions of points and circles intersection
def points_on_same_side (Y B : Point) (lineAC : Line) : Prop :=
-- Here should be the appropriate condition that defines when Y and B are on the same side of the line.
sorry

-- The main theorem
theorem Y_on_median_BM (h1 : Y ∈ (omega1 ∩ omega2)) 
(h2 : points_on_same_side Y B (line_through A C)) : 
lies_on Y (line_through B M) :=
sorry

end Y_on_median_BM_l428_428304


namespace S_2016_eq_0_l428_428537

-- Define the sequence with given initial conditions and recurrence relation
def a : ℕ → ℤ
| 1       := 2
| 2       := 3
| (n + 2) := a (n + 1) - a n

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) : ℤ :=
  (Finset.range n).sum (λ i, a (i + 1))

-- The proof problem statement
theorem S_2016_eq_0 : S 2016 = 0 := 
sorry

end S_2016_eq_0_l428_428537


namespace num_divisors_not_divisors_l428_428118

theorem num_divisors_not_divisors : 
  let a_max := 49999
  let prime1 := 2
  let prime2 := 499
  let n1 := prime1 ^ a_max
  let n2 := prime2 ^ a_max
  let N := n1 * n2 
  let M := N / prime1 / prime2 -- 998^(49999-1)
  ((N : ℕ) → divisor N) \  (divisor M) = 99999 :=
sorry

end num_divisors_not_divisors_l428_428118


namespace problem_statement_l428_428696

theorem problem_statement (x y : ℝ) (h : |x + 1| + |y + 2 * x| = 0) : (x + y) ^ 2004 = 1 := by
  sorry

end problem_statement_l428_428696


namespace average_riding_speed_l428_428921

theorem average_riding_speed
  (initial_reading : ℕ) (final_reading : ℕ) (time_day1 : ℕ) (time_day2 : ℕ)
  (h_initial : initial_reading = 2332)
  (h_final : final_reading = 2552)
  (h_time_day1 : time_day1 = 5)
  (h_time_day2 : time_day2 = 4) :
  (final_reading - initial_reading) / (time_day1 + time_day2) = 220 / 9 :=
by
  sorry

end average_riding_speed_l428_428921


namespace divide_45_to_get_900_l428_428956

theorem divide_45_to_get_900 (x : ℝ) (h : 45 / x = 900) : x = 0.05 :=
by
  sorry

end divide_45_to_get_900_l428_428956


namespace misha_second_round_score_l428_428331

def misha_score_first_round (darts : ℕ) (score_per_dart_min : ℕ) : ℕ := 
  darts * score_per_dart_min

def misha_score_second_round (score_first : ℕ) (multiplier : ℕ) : ℕ := 
  score_first * multiplier

def misha_score_third_round (score_second : ℕ) (multiplier : ℚ) : ℚ := 
  score_second * multiplier

theorem misha_second_round_score (darts : ℕ) (score_per_dart_min : ℕ) (multiplier_second : ℕ) (multiplier_third : ℚ) 
  (h_darts : darts = 8) (h_score_per_dart_min : score_per_dart_min = 3) (h_multiplier_second : multiplier_second = 2) (h_multiplier_third : multiplier_third = 1.5) :
  misha_score_second_round (misha_score_first_round darts score_per_dart_min) multiplier_second = 48 :=
by sorry

end misha_second_round_score_l428_428331


namespace parallelogram_sides_l428_428373

variable (AB BC CD AD M N : ℝ)
variable (a : ℝ)

-- Conditions
def condition1 := BC = 2 * AB
def condition2 := BC = 2 * a
def condition3 := MN = 12

-- Proof Problem Statement
theorem parallelogram_sides 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  : AB = 4 ∧ BC = 8 ∧ CD = 4 ∧ AD = 8 := 
sorry

end parallelogram_sides_l428_428373


namespace ivan_spent_fraction_l428_428270

theorem ivan_spent_fraction (f : ℝ) (h1 : 10 - 10 * f - 5 = 3) : f = 1 / 5 :=
by
  sorry

end ivan_spent_fraction_l428_428270


namespace part1_sol_set_part2_range_a_l428_428761

def f (x a : ℝ) := 5 - |x + a| - |x - 2|

theorem part1_sol_set (a : ℝ) : 
  a = 1 → {x : ℝ | f x a ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem part2_range_a (H : ∀ x : ℝ, f x a ≤ 1) : 
  a ∈ (-∞, -6] ∪ [2, ∞) :=
by
  sorry

end part1_sol_set_part2_range_a_l428_428761


namespace tickets_savings_percentage_l428_428920

theorem tickets_savings_percentage (P S : ℚ) (h : 8 * S = 5 * P) :
  (12 * P - 12 * S) / (12 * P) * 100 = 37.5 :=
by 
  sorry

end tickets_savings_percentage_l428_428920


namespace AE_eq_AB_plus_AC_l428_428418

variables {A B C D E : Type} [EuclideanGeometry A B C D E]
variables (AB AC AD BD CD DE CE : ℝ)

-- Conditions
axiom cond1 : AD = (BD^2) / (AB + AD)
axiom cond2 : AD = (CD^2) / (AC + AD)
axiom cond3 : CD = (DE^2) / (CD + CE)

-- To Prove
theorem AE_eq_AB_plus_AC  (h1 : cond1) (h2 : cond2) (h3 : cond3) : AE = AB + AC := 
sorry

end AE_eq_AB_plus_AC_l428_428418


namespace sum_of_xi_is_1_l428_428763

-- Definition of the conditions given in the problem
def conditions (x y z : ℂ) : Prop :=
  x + y * z = 9 ∧ y + x * z = 13 ∧ z + x * y = 12 ∧ x - y + z = 5

-- Defining the main proposition to prove
theorem sum_of_xi_is_1 :
  (∀ (solutions : List (ℂ × ℂ × ℂ)), 
    (∀ sol ∈ solutions, conditions sol.1 sol.2 sol.3) → 
    solutions.map (λ sol, sol.1)).sum = 1 := 
  sorry

end sum_of_xi_is_1_l428_428763


namespace calculate_expression_l428_428930

theorem calculate_expression : |real.sqrt 3 - 2| + 2 * real.sin (real.pi / 3) - 2023^0 = 1 := 
by
  -- detailed proof steps will go here
  sorry

end calculate_expression_l428_428930


namespace fraction_is_percent_l428_428845

theorem fraction_is_percent (y : ℝ) (hy : y > 0) : (6 * y / 20 + 3 * y / 10) = (60 / 100) * y :=
by
  sorry

end fraction_is_percent_l428_428845


namespace rectangle_perimeter_greater_than_16_l428_428171

theorem rectangle_perimeter_greater_than_16 (a b : ℝ) (h : a * b > 2 * a + 2 * b) (ha : a > 0) (hb : b > 0) : 
  2 * (a + b) > 16 :=
sorry

end rectangle_perimeter_greater_than_16_l428_428171


namespace not_p_is_sufficient_but_not_necessary_for_q_l428_428543

-- Definitions for the conditions
def p (x : ℝ) : Prop := x^2 > 4
def q (x : ℝ) : Prop := x ≤ 2

-- Definition of ¬p based on the solution derived
def not_p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- The theorem statement
theorem not_p_is_sufficient_but_not_necessary_for_q :
  ∀ x : ℝ, (not_p x → q x) ∧ ¬(q x → not_p x) := sorry

end not_p_is_sufficient_but_not_necessary_for_q_l428_428543


namespace find_c_l428_428349

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x, g_inv (g x c) = x) ↔ c = 3 / 2 := by
  sorry

end find_c_l428_428349


namespace numbers_eq_sum_of_others_l428_428341

theorem numbers_eq_sum_of_others
    (a b c : ℝ)
    (h1 : abs(a - b) ≥ abs(c))
    (h2 : abs(b - c) ≥ abs(a))
    (h3 : abs(c - a) ≥ abs(b)) :
    a = b + c ∨ b = a + c ∨ c = a + b := by
  sorry

end numbers_eq_sum_of_others_l428_428341


namespace num_rectangles_in_5x5_grid_l428_428640

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l428_428640


namespace evaluate_expression_l428_428125

theorem evaluate_expression (a b : ℕ) :
  a = 3 ^ 1006 →
  b = 7 ^ 1007 →
  (a + b)^2 - (a - b)^2 = 42 * 10^x :=
by
  intro h1 h2
  sorry

end evaluate_expression_l428_428125


namespace sum_of_solutions_l428_428402

theorem sum_of_solutions : (∑ n in (Finset.filter (λ n, abs n < abs (n - 3)) (Finset.Ico (-7) 13)), n) = -20 :=
by
  sorry

end sum_of_solutions_l428_428402


namespace initial_fraction_of_cylinder_l428_428095

theorem initial_fraction_of_cylinder (F : ℚ) (bottles_added total_capacity : ℚ) :
  -- Conditions
  bottles_added = 4 →
  total_capacity = 80 →
  F = (total_capacity * 4 / 5 - bottles_added) / total_capacity →
  -- Conclusion
  F = 3 / 4 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

-- The requirement is to write the statement only, hence the complete proof is not needed.

end initial_fraction_of_cylinder_l428_428095


namespace sum_of_all_possible_k_values_l428_428029

noncomputable def sum_k_with_distinct_integer_solutions : Int :=
  let is_distinct_integer_solution (p q : Int) : Bool :=
    (p ≠ q) && (3 == p * q)
  let k_values := {k | ∃ (p q : Int), is_distinct_integer_solution p q ∧ k = 3 * (p + q)}
  k_values.sum

theorem sum_of_all_possible_k_values : sum_k_with_distinct_integer_solutions = 0 := 
  sorry

end sum_of_all_possible_k_values_l428_428029


namespace train_pass_time_l428_428466

noncomputable def relative_speed_kmh (train_speed man_speed : ℕ) : ℕ :=
  train_speed + man_speed

noncomputable def relative_speed_ms (speed_kmh : ℕ) : ℚ :=
  (speed_kmh * 1000) / 3600

noncomputable def time_to_pass (distance : ℕ) (speed_ms : ℚ) : ℚ :=
  distance / speed_ms

theorem train_pass_time :
  let train_length := 250
  let train_speed := 80
  let man_speed := 12
  let rel_speed_kmh := relative_speed_kmh train_speed man_speed
  let rel_speed_ms := relative_speed_ms rel_speed_kmh
  let pass_time := time_to_pass train_length rel_speed_ms
  pass_time ≈ 9.78
:=
by
  let train_length := 250
  let train_speed := 80
  let man_speed := 12
  let rel_speed_kmh := relative_speed_kmh train_speed man_speed
  let rel_speed_ms := relative_speed_ms rel_speed_kmh
  let pass_time := time_to_pass train_length rel_speed_ms
  have h: pass_time ≈ 9.78 := sorry
  exact h

end train_pass_time_l428_428466


namespace ads_ratio_proof_correct_l428_428021

noncomputable def ads_ratio_proof : Prop :=
  let ads_first := 12 in
  let ads_second := 2 * ads_first in
  let ads_third := ads_second + 24 in
  let ads_fourth := (3 / 4) * ads_second in
  let total_ads := ads_first + ads_second + ads_third + ads_fourth in
  let clicked_ads := 68 in
  (clicked_ads / total_ads) = (2 / 3)

theorem ads_ratio_proof_correct : ads_ratio_proof := 
by 
  sorry

end ads_ratio_proof_correct_l428_428021


namespace sum_squares_difference_l428_428482

open BigOperators

theorem sum_squares_difference :
  let sumOfFirstNSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6
  let sumSquaresEvenNumbers : ℕ := 4 * sumOfFirstNSquares 100
  let sumSquaresOddNumbers : ℕ := ∑ k in Finset.range 100, (2 * k + 1) ^ 2
  sumSquaresEvenNumbers - sumSquaresOddNumbers = 20100 := by
  let n := 100
  have sumOfFirstNSquares := fun n => n * (n + 1) * (2 * n + 1) / 6
  have sumSquaresEven := 4 * sumOfFirstNSquares n
  have sumSquaresOdd := ∑ k in Finset.range n, (2 * k + 1) ^ 2
  show (sumSquaresEven - sumSquaresOdd) = 20100
  sorry

end sum_squares_difference_l428_428482


namespace sqrt_sum_eq_seven_l428_428785

theorem sqrt_sum_eq_seven (y : ℝ) (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) : 
  sqrt (64 - y^2) + sqrt (36 - y^2) = 7 := 
by
  sorry

end sqrt_sum_eq_seven_l428_428785


namespace range_of_a_over_b_proof_l428_428979

noncomputable def range_of_a_over_b {a b : ℝ} (h1 : a^2 + b^2 = 1) (h2 : b ≠ 0) (h3 : ∃ x y : ℝ, ax + by = 2 ∧ (x^2 / 6 + y^2 / 2 = 1)) : Set ℝ :=
{r : ℝ | r = a / b ∧ (r ∈ Set.Icc (-1) 1 ∨ r ∈ Set.Icc 1 (1/r))}

theorem range_of_a_over_b_proof {a b : ℝ} (h1 : a^2 + b^2 = 1) (h2 : b ≠ 0) (h3 : ∃ x y : ℝ, ax + by = 2 ∧ (x^2 / 6 + y^2 / 2 = 1)) :
  range_of_a_over_b h1 h2 h3 = {r : ℝ | r ≤ -1 ∨ 1 ≤ r} :=
sorry

end range_of_a_over_b_proof_l428_428979


namespace rectangle_count_5x5_l428_428648

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428648


namespace max_f_when_a_eq_1_m_range_when_a_lt_0_l428_428210

def f (x a : ℝ) : ℝ := x / (1 + x) - a * Real.log (1 + x)
def g (x m : ℝ) : ℝ := x^2 * Real.exp (m * x)

theorem max_f_when_a_eq_1 :
  (∀ x : ℝ, f x 1 ≤ 0) :=
sorry

theorem m_range_when_a_lt_0 (a : ℝ) (h1 : a < 0) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 → f x1 a + 1 ≥ g x2 m) ↔ m ∈ set.Iic (-Real.log 2) :=
sorry

end max_f_when_a_eq_1_m_range_when_a_lt_0_l428_428210


namespace branch_A_grade_A_probability_branch_B_grade_A_probability_branch_A_average_profit_branch_B_average_profit_choose_branch_l428_428436

theorem branch_A_grade_A_probability : 
  let total_A := 100
  let grade_A_A := 40
  (grade_A_A / total_A) = 0.4 := by
  sorry

theorem branch_B_grade_A_probability : 
  let total_B := 100
  let grade_A_B := 28
  (grade_A_B / total_B) = 0.28 := by
  sorry

theorem branch_A_average_profit :
  let freq_A_A := 0.4
  let freq_A_B := 0.2
  let freq_A_C := 0.2
  let freq_A_D := 0.2
  let process_cost_A := 25
  let profit_A := (90 - process_cost_A) * freq_A_A + (50 - process_cost_A) * freq_A_B + (20 - process_cost_A) * freq_A_C + (-50 - process_cost_A) * freq_A_D
  profit_A = 15 := by
  sorry

theorem branch_B_average_profit :
  let freq_B_A := 0.28
  let freq_B_B := 0.17
  let freq_B_C := 0.34
  let freq_B_D := 0.21
  let process_cost_B := 20
  let profit_B := (90 - process_cost_B) * freq_B_A + (50 - process_cost_B) * freq_B_B + (20 - process_cost_B) * freq_B_C + (-50 - process_cost_B) * freq_B_D
  profit_B = 10 := by
  sorry

theorem choose_branch :
  let profit_A := 15
  let profit_B := 10
  profit_A > profit_B -> "Branch A"

end branch_A_grade_A_probability_branch_B_grade_A_probability_branch_A_average_profit_branch_B_average_profit_choose_branch_l428_428436


namespace positive_integers_eq_negative_integers_eq_positive_fractions_eq_negative_fractions_eq_integers_eq_positive_numbers_eq_l428_428508

-- Conditions: Given numbers list
def given_numbers : List ℝ :=
  [-10, 6, -7.33, 0, 3.25, -2.25, 0.3, 67, -2/7, 0.1, -18, Real.pi]

-- Proof statements
theorem positive_integers_eq : {6, 67} = {x | x ∈ given_numbers ∧ x > 0 ∧ (∃ n : ℤ, x = n)} := by
  sorry

theorem negative_integers_eq : {-10, -18} = {x | x ∈ given_numbers ∧ x < 0 ∧ (∃ n : ℤ, x = n)} := by
  sorry

theorem positive_fractions_eq : {3.25, 0.3, 0.1} = {x | x ∈ given_numbers ∧ x > 0 ∧ ∀ n : ℤ, x ≠ n} := by
  sorry

theorem negative_fractions_eq : {-7.33, -2.25, -2/7} = {x | x ∈ given_numbers ∧ x < 0 ∧ ∀ n : ℤ, x ≠ n} := by
  sorry

theorem integers_eq : {-10, 6, 0, 67, -18} = {x | x ∈ given_numbers ∧ ∃ n : ℤ, x = n} := by
  sorry

theorem positive_numbers_eq : {6, 3.25, 0.3, 67, 0.1, Real.pi} = {x | x ∈ given_numbers ∧ x > 0} := by
  sorry

end positive_integers_eq_negative_integers_eq_positive_fractions_eq_negative_fractions_eq_integers_eq_positive_numbers_eq_l428_428508


namespace total_money_made_l428_428839

-- Definitions for costs
def bracelet_cost : ℕ := 5
def discounted_bracelet_cost : ℕ := 8
def necklace_cost : ℕ := 10
def discounted_necklace_cost : ℕ := 25

-- Definitions for starting quantities
def initial_bracelets : ℕ := 30
def initial_necklaces : ℕ := 20

-- Definitions for bracelets sold
def regular_price_bracelets_sold : ℕ := 12
def discounted_price_bracelets_sold : ℕ := 12

-- Definitions for necklaces sold
def regular_price_necklaces_sold : ℕ := 8
def discounted_sets_necklaces_sold : ℕ := 2

-- Theorem stating the total amount of money Zayne made
theorem total_money_made : 
  let total_bracelets_money := (regular_price_bracelets_sold * bracelet_cost) + ((discounted_price_bracelets_sold / 2) * discounted_bracelet_cost) in
  let total_necklaces_money := (regular_price_necklaces_sold * necklace_cost) + (discounted_sets_necklaces_sold * discounted_necklace_cost) in
  total_bracelets_money + total_necklaces_money = 238 :=
  by
  sorry

end total_money_made_l428_428839


namespace extra_time_A_to_reach_destination_l428_428254

theorem extra_time_A_to_reach_destination (speed_ratio : ℕ -> ℕ -> Prop) (t_A t_B : ℝ)
  (h_ratio : speed_ratio 3 4)
  (time_A : t_A = 2)
  (distance_constant : ∀ a b : ℝ, a / b = (3 / 4)) :
  (t_A - t_B) * 60 = 30 :=
by
  sorry

end extra_time_A_to_reach_destination_l428_428254


namespace base_eight_addition_l428_428521

theorem base_eight_addition :
  ∃ h : ℕ, 
    (h > 6) ∧ (6453_h + 7512_h = 16165_h) ∧ (h = 8) := 
begin
  use 8,
  split,
  { linarith, },
  split,
  { 
    unfold has_add.add,
    simp, 
  },
  { refl }
end

end base_eight_addition_l428_428521


namespace product_gcf_lcm_30_75_l428_428831

noncomputable def gcf (a b : ℕ) : ℕ := Nat.gcd a b
noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem product_gcf_lcm_30_75 : gcf 30 75 * lcm 30 75 = 2250 := by
  sorry

end product_gcf_lcm_30_75_l428_428831


namespace pipes_fill_time_unique_l428_428711

variable {t₁ t₂ : ℝ}

noncomputable def time_to_fill_pipes := 
  (1 / t₁) + (1 / t₂) = 1 / 2.4 ∧ (t₂ / (4 * t₁)) + (t₁ / (4 * t₂)) = 13 / 24

theorem pipes_fill_time_unique (h : time_to_fill_pipes) :
  t₁ = 4 ∧ t₂ = 6 :=
sorry

end pipes_fill_time_unique_l428_428711


namespace interval_of_p_l428_428398

theorem interval_of_p (p : ℝ) :
  (∀ (q : ℝ), q > 0 → (4 * (p * q^2 + p^3 * q + 4 * q^2 + 4 * p * q) / (p + q) > 3 * p^2 * q)) ↔ 
  (0 ≤ p ∧ p < (2 + sqrt 13) / 3) :=
by
  sorry

end interval_of_p_l428_428398


namespace solution_set_for_product_l428_428757

def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def derivative_cond (f g : ℝ → ℝ) (x : ℝ) : Prop := f' x * g x + f x * g' x > 0

theorem solution_set_for_product
  (f g : ℝ → ℝ)
  (Hodd : odd f)
  (Heven : even g)
  (Hderiv : ∀ x < 0, derivative_cond f g x)
  (Hg : g 3 = 0) :
  {x | f x * g x < 0} = Iio (-3) ∪ Ioo 0 3 :=
sorry

end solution_set_for_product_l428_428757


namespace num_rectangles_in_5x5_grid_l428_428604

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428604


namespace problem_statement_l428_428836

def has_two_zeros (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ f a = 0 ∧ f b = 0

def option_A : ℝ → ℝ := λ x, x^2 - 4 * x + 3
def option_B : ℝ → ℝ := λ x, 3 * x + 10
def option_C : ℝ → ℝ := λ x, x^2 - 3 * x + 5
def option_D : ℝ → ℝ := λ x, Real.log x / Real.log 2

theorem problem_statement : has_two_zeros option_A ∧ 
  ¬ (has_two_zeros option_B ∨ has_two_zeros option_C ∨ has_two_zeros option_D) :=
by 
  sorry

end problem_statement_l428_428836


namespace cole_total_round_trip_time_l428_428108

noncomputable def cole_round_trip_time (speed_to_work speed_to_home : ℝ) (time_to_work_minutes : ℝ) : ℝ :=
  let time_to_work_hours := time_to_work_minutes / 60
  let distance_to_work := speed_to_work * time_to_work_hours
  let time_return_home_hours := distance_to_work / speed_to_home
  time_to_work_hours + time_return_home_hours

theorem cole_total_round_trip_time :
  cole_round_trip_time 75 105 140 ≈ 4 :=
by
  sorry

end cole_total_round_trip_time_l428_428108


namespace trig_ineq_l428_428980

theorem trig_ineq (x y z : ℝ) (hx : 0 < z ∧ z < y ∧ y < x ∧ x < π / 2) :
  (π / 2) + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
by
  sorry

end trig_ineq_l428_428980


namespace floor_abs_expression_l428_428924

theorem floor_abs_expression : 
  (Int.floor (Real.abs (-5.7)))^2 + Real.abs (Int.floor (-5.7)) - 1 / 2 = 61 / 2 :=
by 
  sorry

end floor_abs_expression_l428_428924


namespace sports_meeting_duration_and_medals_l428_428006

-- Define the recursive formula for the number of medals awarded on the k-th day
def medals_awarded_on_kth_day (k : ℕ) (m : ℕ) (a : ℕ → ℕ) : ℕ :=
  k + (1 / 7 : ℝ) * (m - ∑ i in Finset.range(k), a i - k)

-- Define the total number of medals
def total_medals (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  ∑ i in Finset.range(n), a i

-- Final theorem statement
theorem sports_meeting_duration_and_medals :
  ∃ n m : ℕ, n > 1 ∧ n = 6 ∧ m = 36 ∧ ∀ k : ℕ, 
    (1 ≤ k ∧ k ≤ n) → (medals_awarded_on_kth_day k m (λ i, medals_awarded_on_kth_day i m (λ _, 0))) = (if k < n then 1 else 0)) :=
begin
    sorry
end

end sports_meeting_duration_and_medals_l428_428006


namespace rectangle_perimeter_gt_16_l428_428172

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
  sorry

end rectangle_perimeter_gt_16_l428_428172


namespace modulus_is_sqrt5_l428_428800

noncomputable def modulus_of_complex_number : ℂ :=
  (4 - 3 * Complex.i) / (2 - Complex.i)

theorem modulus_is_sqrt5 : Complex.abs modulus_of_complex_number = Real.sqrt 5 := by
  sorry

end modulus_is_sqrt5_l428_428800


namespace sum_p_q_r_s_l428_428379

theorem sum_p_q_r_s :
  ∃ (p q r s : ℕ), 
    (∀ (x y : ℝ), 
      x + y = 5 ∧ 2 * x * y = 5 →
      x = (p + q * Real.sqrt r) / s ∨ x = (p - q * Real.sqrt r) / s
    ) ∧ (p + q + r + s = 23) :=
begin
  sorry
end

end sum_p_q_r_s_l428_428379


namespace trig_identity_neg_cos_half_alpha_l428_428746

theorem trig_identity_neg_cos_half_alpha 
  (α : ℝ) 
  (h1 : -3 * Real.pi < α) 
  (h2 : α < -5 * Real.pi / 2) :
  sqrt ((1 + Real.cos (α - 2018 * 2 * Real.pi)) / 2) = -Real.cos (α / 2) :=
by sorry

end trig_identity_neg_cos_half_alpha_l428_428746


namespace num_rectangles_in_5x5_grid_l428_428600

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l428_428600


namespace alice_savings_l428_428091

noncomputable def commission (sales : ℝ) : ℝ := 0.02 * sales
noncomputable def totalEarnings (basic_salary commission : ℝ) : ℝ := basic_salary + commission
noncomputable def savings (total_earnings : ℝ) : ℝ := 0.10 * total_earnings

theorem alice_savings (sales basic_salary : ℝ) (commission_rate savings_rate : ℝ) :
  commission_rate = 0.02 →
  savings_rate = 0.10 →
  sales = 2500 →
  basic_salary = 240 →
  savings (totalEarnings basic_salary (commission_rate * sales)) = 29 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_savings_l428_428091


namespace min_expression_for_right_triangle_l428_428744

theorem min_expression_for_right_triangle 
  (A B C D : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  (angle_ACB : angle A C B = π / 2)
  (perpendicular_CD_AB : ∀ x, inner_product (CD) x = 0)
  (area_ABC : 1/2 * ∥A - B∥ * ∥B - C∥ = 84) :
  ∃ AC BC CD : ℝ, AC^2 + (3 * CD)^2 + BC^2 = 1008 :=
sorry

end min_expression_for_right_triangle_l428_428744


namespace number_of_solutions_l428_428968

def sign (a : ℝ) : ℝ :=
  if a > 0 then 1 else if a = 0 then 0 else -1

-- Given conditions as Lean functions
noncomputable def cond_x (y z : ℝ) : ℝ := 2023 - 2024 * sign (y + z + 1)
noncomputable def cond_y (x z : ℝ) : ℝ := 2023 - 2024 * sign (x + z - 1)
noncomputable def cond_z (x y : ℝ) : ℝ := 2023 - 2024 * sign (x + y + 1)

-- Number of solutions
theorem number_of_solutions : 
  {t : ℝ × ℝ × ℝ | 
    let (x, y, z) := t in 
    x = cond_x y z ∧ y = cond_y x z ∧ z = cond_z x y
  }.toFinset.card = 3 := 
sorry

end number_of_solutions_l428_428968


namespace number_of_divisible_sums_l428_428816

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def count_divisible_sums (A B : Finset ℕ) : ℕ :=
  (A.product B).filter (λ (ab : ℕ × ℕ), is_divisible_by_3 (ab.1 + ab.2)).card

theorem number_of_divisible_sums :
  count_divisible_sums (Finset.range 21) (Finset.range 21) = 134 :=
sorry

end number_of_divisible_sums_l428_428816


namespace counterexample_9918_l428_428941

theorem counterexample_9918 : 
  (sum_of_digits 9918 % 27 = 0) ∧ (9918 % 27 ≠ 0) := 
by
  sorry

end counterexample_9918_l428_428941


namespace pasture_total_rent_l428_428042

/-- Given the number of oxen and the duration in months for three people (a, b, and c), and
the share of rent paid by c, prove that the total rent of the pasture is Rs. 105.00. -/
theorem pasture_total_rent 
  (a_oxen : ℕ := 10) (a_months : ℕ := 7)
  (b_oxen : ℕ := 12) (b_months : ℕ := 5)
  (c_oxen : ℕ := 15) (c_months : ℕ := 3)
  (c_rent : ℝ := 26.999999999999996) :
  let a_oxen_months := a_oxen * a_months,
      b_oxen_months := b_oxen * b_months,
      c_oxen_months := c_oxen * c_months,
      total_oxen_months := a_oxen_months + b_oxen_months + c_oxen_months,
      rent_per_oxen_month := c_rent / c_oxen_months,
      total_rent := rent_per_oxen_month * total_oxen_months in
  Real.Round (total_rent) = 105.00 := by
    sorry

end pasture_total_rent_l428_428042


namespace number_of_true_propositions_is_four_l428_428364

-- Definitions for each condition
def prop₁ : Prop := -- The length of the perpendicular segment from a point outside a line to the line is called the distance from the point to the line
  ∀ (P L: Type) [isPoint P] [isLine L] (p : P) (l : L), distance_from_point_to_line p l = length_of_perpendicular_segment p l 

def prop₂ : Prop := -- There is one and only one line perpendicular to a given line passing through a point
  ∀ (P L: Type) [isPoint P] [isLine L] (p : P) (l : L), perpendicular_line_through_point p l

def prop₃ : Prop := -- There is one and only one line parallel to a given line passing through a point
  ∀ (P L: Type) [isPoint P] [isLine L] (p : P) (l : L), parallel_line_through_point p l

def prop₄ : Prop := -- Rational numbers correspond one-to-one with points on the number line
  ∀ (r: ℚ), position_on_number_line r 

def prop₅ : Prop := -- Pi is an irrational number
  irrational_number π 

def numberOfTrueProps: ℕ := countTrue [prop₁, prop₂, prop₃, prop₄, prop₅]

theorem number_of_true_propositions_is_four : numberOfTrueProps = 4 :=
by
  sorry -- Proof goes here

end number_of_true_propositions_is_four_l428_428364


namespace no_solution_exists_l428_428767

theorem no_solution_exists :
  ¬ ∃ (x1 x2 x3 x4 : ℝ), 
    (x1 + x2 = 1) ∧
    (x2 + x3 - x4 = 1) ∧
    (0 ≤ x1) ∧
    (0 ≤ x2) ∧
    (0 ≤ x3) ∧
    (0 ≤ x4) ∧
    ∀ (F : ℝ), F = x1 - x2 + 2 * x3 - x4 → 
    ∀ (b : ℝ), F ≤ b :=
by sorry

end no_solution_exists_l428_428767


namespace billy_apples_ratio_l428_428922

theorem billy_apples_ratio :
  let monday := 2
  let tuesday := 2 * monday
  let wednesday := 9
  let friday := monday / 2
  let total_apples := 20
  let thursday := total_apples - (monday + tuesday + wednesday + friday)
  thursday / friday = 4 := 
by
  let monday := 2
  let tuesday := 2 * monday
  let wednesday := 9
  let friday := monday / 2
  let total_apples := 20
  let thursday := total_apples - (monday + tuesday + wednesday + friday)
  sorry

end billy_apples_ratio_l428_428922


namespace triangle_ABC_is_isosceles_right_l428_428149

theorem triangle_ABC_is_isosceles_right (a b c : ℝ) (A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * a * b * c)
  (h2 : sin A = 2 * sin B * cos C) : 
  -- Statement of the question translated to Lean
  ∃ k : ℝ, (a = b ∧ b = k * sqrt 2 * c) :=
sorry

end triangle_ABC_is_isosceles_right_l428_428149


namespace rectangles_in_grid_l428_428672

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem rectangles_in_grid :
  let n := 5 in 
  binomial n 2 * binomial n 2 = 100 :=
by
  sorry

end rectangles_in_grid_l428_428672


namespace slope_angle_of_AB_is_135_degrees_l428_428166

-- Define the points A and B
def A : ℝ × ℝ := (3, -1)
def B : ℝ × ℝ := (0, 2)

-- Define the slope angle problem
theorem slope_angle_of_AB_is_135_degrees (A B : ℝ × ℝ) (hA : A = (3, -1)) (hB : B = (0, 2)) :
  ∃ α : ℝ, α = 135 ∧ tan α = -(slope A B) :=
by
  -- Proof goes here
  sorry

-- Auxiliary function to calculate the slope between two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

end slope_angle_of_AB_is_135_degrees_l428_428166


namespace find_number_l428_428033

theorem find_number (x : ℝ) (h : sqrt x / 6 = 2) : x = 144 :=
by
  sorry

end find_number_l428_428033


namespace part1_part2_l428_428192

def setA (x : ℝ) : Prop := x^2 - 5 * x - 6 < 0

def setB (a x : ℝ) : Prop := 2 * a - 1 ≤ x ∧ x < a + 5

open Set

theorem part1 : 
  let a := 0
  A = {x : ℝ | -1 < x ∧ x < 6} →
  B a = {x : ℝ | -1 ≤ x ∧ x < 5} →
  {x | (setA x) ∧ (setB a x)} = {x | -1 < x ∧ x < 5} :=
by
  sorry

theorem part2 : 
  A = {x : ℝ | -1 < x ∧ x < 6} →
  (B : ℝ → Set real) →
  (∀ x, (setA x ∨ setB a x) → setA x) →
  { a : ℝ | (0 < a ∧ a ≤ 1) ∨ a ≥ 6 } :=
by
  sorry

end part1_part2_l428_428192


namespace triangle_not_uniquely_constructible_l428_428791

open EuclideanGeometry

/-- Given points O, F, and K and Angle α, the triangle ABC is not uniquely constructible -/
theorem triangle_not_uniquely_constructible
  (A B C O F K : Point)
  (α : Angle) 
  (hO : Circumcenter O A B C)
  (hF : Midpoint F B C)
  (hK : OnLine K A C)
  (hα : ∠ BAC = α) : 
  ¬ (Constructible A B C)
  :=
sorry

end triangle_not_uniquely_constructible_l428_428791


namespace alice_savings_l428_428085

def sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.10

theorem alice_savings :
  (basic_salary + (sales * commission_rate)) * savings_rate = 29 :=
by
  sorry

end alice_savings_l428_428085


namespace clothing_probability_l428_428691

theorem clothing_probability :
  let total_clothing := 6 + 8 + 9 + 4,
      ways_to_choose_four := Nat.choose total_clothing 4,
      specific_ways := 6 * 8 * 9 * 4 in
  (6 + 8 + 9 + 4 = 27)
  -> (ways_to_choose_four = 17550)
  -> (specific_ways = 1728)
  -> (specific_ways / ways_to_choose_four = 96 / 975) :=
by sorry

end clothing_probability_l428_428691


namespace simplify_expression_l428_428782

-- Define the expressions involved
def expr1 (x : ℝ) := (x^2 - 4*x + 3) / (x^2 - 6*x + 9)
def expr2 (x : ℝ) := (x^2 - 6*x + 8) / (x^2 - 8*x + 15)
def simplified_expr (x : ℝ) := ((x - 1) * (x - 5)) / ((x - 4) * (x - 2))

-- The theorem to be proved
theorem simplify_expression (x : ℝ) :
  expr1 x / expr2 x = simplified_expr x :=
sorry

end simplify_expression_l428_428782


namespace minimum_bottles_needed_l428_428067

theorem minimum_bottles_needed (medium_volume jumbo_volume : ℕ) (h_medium : medium_volume = 120) (h_jumbo : jumbo_volume = 2000) : 
  let minimum_bottles := (jumbo_volume + medium_volume - 1) / medium_volume
  minimum_bottles = 17 :=
by
  sorry

end minimum_bottles_needed_l428_428067


namespace number_of_rectangles_in_grid_l428_428588

theorem number_of_rectangles_in_grid : 
  let num_lines := 5 in
  let ways_to_choose_2_lines := Nat.choose num_lines 2 in
  ways_to_choose_2_lines * ways_to_choose_2_lines = 100 :=
by
  let num_lines := 5
  let ways_to_choose_2_lines := Nat.choose num_lines 2
  show ways_to_choose_2_lines * ways_to_choose_2_lines = 100 from sorry

end number_of_rectangles_in_grid_l428_428588


namespace count_correct_statements_l428_428991

theorem count_correct_statements
  (M N : ℝ → ℝ) 
  (h₁ : ∀ x, M x = 2 - 4 * x)
  (h₂ : ∀ x, N x = 4 * x + 1) :
  (¬∃ x, M x + N x = 0) ∧ (¬∀ x, 0 < M x ∧ 0 < N x) ∧ (∀ a, ∀ x, (M x + a) * N x = 1 - 16 * x^2 → a = -1) ∧ (∀ x, M x * N x = -3 → M x ^ 2 + N x ^ 2 = 11) ↔ false :=
begin
  sorry
end

end count_correct_statements_l428_428991


namespace grumpy_not_orange_l428_428490

variable (Liz : Type) [Fintype Liz]

variable (is_lizard : Liz → Prop)
variable (is_orange : Liz → Prop)
variable (is_grumpy : Liz → Prop)
variable (can_swim : Liz → Prop)
variable (can_jump : Liz → Prop)

variable (cathy_lizards : Fintype.card Liz = 15)
variable (orange_lizards : Fintype.card {l : Liz // is_orange l} = 6)
variable (grumpy_lizards : Fintype.card {l : Liz // is_grumpy l} = 7)

variable (grumpy_can_swim : ∀ l : Liz, is_grumpy l → can_swim l)
variable (orange_cannot_jump : ∀ l : Liz, is_orange l → ¬ can_jump l)
variable (cannot_jump_cannot_swim : ∀ l : Liz, ¬ (can_jump l) → ¬ (can_swim l))

theorem grumpy_not_orange (l : Liz) : is_grumpy l → ¬ is_orange l :=
sorry

end grumpy_not_orange_l428_428490


namespace group_total_time_180_minutes_l428_428689

-- Define conditions and parameters
def total_students := 30
def number_of_groups := 5
def students_per_group := total_students / number_of_groups

def time_per_student (group: ℕ) : ℕ :=
  match group with
  | 1 => 4
  | 2 => 5
  | 3 => 6
  | 4 => 7
  | 5 => 8
  | _ => 0 -- although groups are limited to 1-5, we include this case to complete the match expression

-- Calculate the total time taken for all groups sequentially
def time_for_group (group: ℕ) : ℕ :=
  students_per_group * time_per_student(group)

def total_time : ℕ :=
  time_for_group(1) + time_for_group(2) + time_for_group(3) +
  time_for_group(4) + time_for_group(5)

-- The main theorem to be proved
theorem group_total_time_180_minutes : total_time = 180 := by sorry

end group_total_time_180_minutes_l428_428689


namespace odd_heads_prob_2015_l428_428829

def coin_heads_prob (n : ℕ) : ℚ :=
  if h : n = 1 then 0
  else
    let rec P (k : ℕ) : ℚ :=
    if k = 1 then 0
    else (2 * ((k : ℚ)^2 - 1) * P (k - 1) + 1) / (2 * (k : ℚ)^2)
    P n

theorem odd_heads_prob_2015 :
  coin_heads_prob 2015 = 1007 / 4030 :=
sorry

end odd_heads_prob_2015_l428_428829


namespace evaluate_polynomial_at_three_l428_428395

def polynomial (x : ℕ) : ℕ :=
  x^6 + 2 * x^5 + 4 * x^3 + 5 * x^2 + 6 * x + 12

theorem evaluate_polynomial_at_three :
  polynomial 3 = 588 :=
by
  sorry

end evaluate_polynomial_at_three_l428_428395


namespace min_distance_Omega1_Omega2_l428_428496

noncomputable def Omega1 (x y : ℝ) : Prop := (y ≤ x) ∧ (3*y ≥ x) ∧ (x + y ≤ 4)

noncomputable def Omega2 (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 ≤ 2

theorem min_distance_Omega1_Omega2 :
    ∃ M N : ℝ × ℝ, Omega1 M.1 M.2 ∧ Omega2 N.1 N.2 ∧ dist M N = sqrt 2 :=
sorry

end min_distance_Omega1_Omega2_l428_428496


namespace bill_due_in_nine_month_l428_428812

-- Define the conditions
def true_discount := 150
def face_value := 1400
def interest_rate := 0.16

-- Define the formula for true discount
def true_discount_formula (FV : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  (FV * r * t) / (1 + r * t)

-- Define the number of months as a function of the number of years
def months (t : ℝ) : ℝ := t * 12

-- Theorem statement
theorem bill_due_in_nine_month (t : ℝ) (n : ℝ):
  true_discount = true_discount_formula face_value interest_rate t →
  n = months t →
  n = 9 :=
by
  intros h1 h2
  sorry

end bill_due_in_nine_month_l428_428812


namespace probability_is_pi_over_12_l428_428073

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let radius := 2
  let circle_area := Real.pi * radius^2
  let rectangle_area := 6 * 8
  circle_area / rectangle_area

theorem probability_is_pi_over_12 :
  probability_within_two_units_of_origin = Real.pi / 12 :=
by
  sorry

end probability_is_pi_over_12_l428_428073


namespace greatest_possible_x_plus_y_l428_428463

theorem greatest_possible_x_plus_y
  (a b c d : ℕ)
  (h : multiset {x | ∃ (a b : ℕ), x = a + b ∧ a ≠ b} = {220, 360, 310, 250, x, y}) :
  x + y ≤ 870 := sorry

end greatest_possible_x_plus_y_l428_428463


namespace page_final_shoe_count_l428_428771

-- Definitions based on given conditions
def initial_pairs : ℕ := 120
def first_donation_percentage : ℝ := 0.45
def pairs_bought_after_first_donation : ℕ := 15
def second_donation_percentage : ℝ := 0.20
def pairs_bought_after_second_donation : ℕ := 10

noncomputable def pairs_after_first_donation : ℕ := initial_pairs - (initial_pairs * first_donation_percentage).toNat
noncomputable def pairs_after_first_purchase : ℕ := pairs_after_first_donation + pairs_bought_after_first_donation
noncomputable def pairs_after_second_donation : ℕ := pairs_after_first_purchase - (pairs_after_first_purchase * second_donation_percentage).toNat
noncomputable def final_pairs : ℕ := pairs_after_second_donation + pairs_bought_after_second_donation

-- The main theorem to prove
theorem page_final_shoe_count : final_pairs = 75 := by
  sorry

end page_final_shoe_count_l428_428771


namespace Y_on_median_BM_l428_428306

variables {A B C M Y : Point}
variables {omega1 omega2 : Circle}

-- Definitions of points and circles intersection
def points_on_same_side (Y B : Point) (lineAC : Line) : Prop :=
-- Here should be the appropriate condition that defines when Y and B are on the same side of the line.
sorry

-- The main theorem
theorem Y_on_median_BM (h1 : Y ∈ (omega1 ∩ omega2)) 
(h2 : points_on_same_side Y B (line_through A C)) : 
lies_on Y (line_through B M) :=
sorry

end Y_on_median_BM_l428_428306


namespace no_solution_eq_iff_l428_428147

theorem no_solution_eq_iff (m : ℤ) : (∀ x : ℤ, x ≠ 4 → x ≠ 8 → (x - 3) * (x - 8) = (x - m) * (x - 4) → false) ↔ (m = 7) :=
begin
  sorry
end

end no_solution_eq_iff_l428_428147


namespace very_good_year_in_21st_century_l428_428876

def four_digit_very_good (YEAR: ℕ) : Prop :=
  let Y := YEAR / 1000
  let E := (YEAR / 100) % 10
  let A := (YEAR / 10) % 10
  let R := YEAR % 10 in
  let coeff_matrix := ![
    ![Y, E, A, R],
    ![R, Y, E, A],
    ![A, R, Y, E],
    ![E, A, R, Y]
  ] in
  Det (coeff_matrix) = 0

theorem very_good_year_in_21st_century: ∀ (YEAR: ℕ), 
  (2001 ≤ YEAR ∧ YEAR ≤ 2100) → (four_digit_very_good YEAR ↔ YEAR = 2020) :=
  sorry

end very_good_year_in_21st_century_l428_428876


namespace angle_MLD_eq_45_l428_428852

-- Definitions for the conditions
variable {radius : ℝ} (h_radius : radius = 1)
variable {chord_length : ℝ} (h_chord_length : chord_length = real.sqrt 2)
variable (M D L : ℝ → ℝ → Prop) -- Points M, D, L on the circle
variable (max_area : ∀ x y z, (x = M ∧ y = L ∧ z = D) → True) -- Condition for maximizing area of triangle

-- The theorem to prove
theorem angle_MLD_eq_45 (h_radius : radius = 1) (h_chord_length : chord_length = real.sqrt 2) (M D L : ℝ → ℝ → Prop) (max_area : ∀ x y z, (x = M ∧ y = L ∧ z = D) → True) : ∃ (angle_MLD : ℝ), angle_MLD = real.pi / 4 :=
by
  sorry

end angle_MLD_eq_45_l428_428852


namespace max_leap_years_l428_428096

theorem max_leap_years (years : ℕ) (leap_frequency : ℕ) (period : ℕ) 
  (h1 : leap_frequency = 4) (h2 : period = 150) 
  (h3 : ∀ n : ℕ, n > 0 → period % leap_frequency = n): 
  ∃ max_leap_years, max_leap_years = 38 := 
begin 
  -- The proof is omitted 
  sorry
end

end max_leap_years_l428_428096


namespace radii_equal_of_parallelogram_and_common_chord_l428_428759

variables {Point : Type} [EuclideanGeometry Point]

def Parallelogram (A B C D : Point) : Prop :=
  Parallelogram ABCD ∧ ¬Rectangle ABCD

variables (A B C D P : Point)

theorem radii_equal_of_parallelogram_and_common_chord
  (h_para : Parallelogram A B C D)
  (hP_inside : Inside P (Parallelogram A B C D))
  (h_common_chord_perpendicular_AD : ∃ M : Point, CommonChordPerpendicularPAB_PCD P A B C D M ∧ M ⟂ AD)
  : Radius (CircumscribedCircle P A B) = Radius (CircumscribedCircle P C D) :=
begin
  sorry
end

end radii_equal_of_parallelogram_and_common_chord_l428_428759


namespace prove_y_value_l428_428502

theorem prove_y_value (y : ℝ) (h : 8^(y + 3) = 512 + 8^y) : y = 0.000685 :=
sorry

end prove_y_value_l428_428502


namespace proposition_verification_l428_428500

def odd_numbers : Set ℕ := { n | n % 2 = 1 }
def even_numbers : Set ℕ := { n | n % 2 = 0 }

def same_cardinality (A B : Set ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), (∀ a₁ a₂ ∈ A, f a₁ = f a₂ → a₁ = a₂) ∧ (∀ b ∈ B, ∃ a ∈ A, f a = b)

variables {C : Set ℕ} (smaller_circle_points larger_circle_points : Set ℝ) 

theorem proposition_verification :
  (same_cardinality odd_numbers even_numbers) ∧
  ¬ (same_cardinality smaller_circle_points larger_circle_points) ∧
  ¬ (∀ (A B : Set ℕ), A ⊂ B → ¬ (same_cardinality A B)) ∧
  (∀ (A B C : Set ℕ), same_cardinality A B → same_cardinality B C → same_cardinality A C) :=
by
  sorry

end proposition_verification_l428_428500


namespace mahdi_plays_tennis_on_friday_l428_428329

theorem mahdi_plays_tennis_on_friday :
  (∃ (sports : ℕ → ℕ), -- A function from days (1 to 7) to sports
    -- Conditions
    (forall n, sports n ∈ {1, 2, 3, 4, 5}) ∧ -- There are 5 sports in total
    (∃ four_days : fin 7 → fin 7, -- A function from the four running days
      (∀ i, 1 ≤ four_days i + 1 ∧ four_days i < 7 ∧ sports (four_days i) = 1) ∧  -- Runs on four days
      (∀ i j, i ≠ j → four_days i ≠ four_days j ∧ abs (four_days i - four_days j) > 1)) ∧ -- No two consecutive running days
    sports 0 = 2 ∧ -- Basketball on Monday
    sports 3 = 3 ∧ -- Golf on Thursday (three days after Monday)
    (∃ swim_day tennis_day : fin 7, -- Swim and Tennis days
      sports swim_day = 4 ∧
      sports tennis_day = 5 ∧
      abs (swim_day - tennis_day) > 1) -- Tennis is not the day before or after swimming
  ) :=
  do
    -- Prove the tennis day is Friday (day 4)
    sorry

end mahdi_plays_tennis_on_friday_l428_428329


namespace ABCD_concyclic_l428_428731

theorem ABCD_concyclic {A B C O K D N M : Type}
  [triangle : Triangle A B C]
  [circumcenter : Circumcenter O A B C]
  [is_acute_angled_triangle : AcuteAngledTriangle A B C]
  (hK_on_BC : OnSegment K B C)
  (hK_not_midpoint : ¬ Midpoint K B C)
  (hD_on_AK_ext : OnExtendedSegment D A K)
  (hBD_intersects_AC_at_N : IntersectsAt BD AC N)
  (hCD_intersects_AB_at_M : IntersectsAt CD AB M)
  (hOK_perpendicular_MN : Perpendicular OK MN) :
  Concyclic A B D C :=
sorry

end ABCD_concyclic_l428_428731


namespace solution_lies_in_interval_l428_428566

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem solution_lies_in_interval (k : ℕ) (h : 0 < k) :
  (∃ x : ℝ, x ∈ set.Ioo (k : ℝ) (k + 1 : ℝ) ∧ log_base_10 x = 3 - x) ↔ k = 2 :=
by
  sorry

end solution_lies_in_interval_l428_428566


namespace part1_part2_l428_428190

def setA (x : ℝ) : Prop := x^2 - 5 * x - 6 < 0

def setB (a x : ℝ) : Prop := 2 * a - 1 ≤ x ∧ x < a + 5

open Set

theorem part1 : 
  let a := 0
  A = {x : ℝ | -1 < x ∧ x < 6} →
  B a = {x : ℝ | -1 ≤ x ∧ x < 5} →
  {x | (setA x) ∧ (setB a x)} = {x | -1 < x ∧ x < 5} :=
by
  sorry

theorem part2 : 
  A = {x : ℝ | -1 < x ∧ x < 6} →
  (B : ℝ → Set real) →
  (∀ x, (setA x ∨ setB a x) → setA x) →
  { a : ℝ | (0 < a ∧ a ≤ 1) ∨ a ≥ 6 } :=
by
  sorry

end part1_part2_l428_428190


namespace rectangle_count_5x5_l428_428644

theorem rectangle_count_5x5 : (Nat.choose 5 2) * (Nat.choose 5 2) = 100 := by
  sorry

end rectangle_count_5x5_l428_428644


namespace all_valid_quadruples_l428_428966

theorem all_valid_quadruples (a b c d : ℝ) :
  (a = b * c ∧ a = b * d ∧ a = c * d) ∧ 
  (b = a * c ∧ b = a * d ∧ b = c * d) ∧
  (c = a * b ∧ c = a * d ∧ c = b * d) ∧
  (d = a * b ∧ d = a * c ∧ d = b * c) →
  (a = b ∧ b = c ∧ c = d ∧ (a = 0 ∨ a = 1 ∨ a = -1)) :=
begin
  sorry
end

end all_valid_quadruples_l428_428966


namespace triangle_area_inscribed_in_circle_l428_428906

noncomputable def circle_inscribed_triangle_area : ℝ :=
  let r := 10 / Real.pi
  let angle_A := Real.pi / 2
  let angle_B := 7 * Real.pi / 10
  let angle_C := 4 * Real.pi / 5
  let sin_sum := Real.sin(angle_A) + Real.sin(angle_B) + Real.sin(angle_C)
  1 / 2 * r^2 * sin_sum

theorem triangle_area_inscribed_in_circle (h_circumference : 5 + 7 + 8 = 20)
  (h_radius : 10 / Real.pi * 2 * Real.pi = 20) :
  circle_inscribed_triangle_area = 138.005 / Real.pi^2 :=
by
  sorry

end triangle_area_inscribed_in_circle_l428_428906


namespace min_tetrahedron_volume_l428_428162

theorem min_tetrahedron_volume :
  ∃ (a b c : ℝ), (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (1 / a + 4 / b + 5 / c = 1) ∧ 
  (∀ a b c : ℝ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (1 / a + 4 / b + 5 / c = 1) → (1 / 6 * a * b * c ≥ 90)) :=
begin
  sorry
end

end min_tetrahedron_volume_l428_428162


namespace number_of_false_statements_l428_428733

theorem number_of_false_statements (a b : ℝ) :
  let original := a > b → a^2 > b^2,
      converse := a^2 > b^2 → a > b,
      inverse := a ≤ b → a^2 ≤ b^2,
      contrapositive := a^2 ≤ b^2 → a ≤ b in
  ¬(original) ∧ ¬(converse) ∧ ¬(contrapositive) ↔ true :=
by
  let original : Prop := a > b → a^2 > b^2
  let converse : Prop := a^2 > b^2 → a > b
  let inverse : Prop := a ≤ b → a^2 ≤ b^2
  let contrapositive : Prop := a^2 ≤ b^2 → a ≤ b
  sorry

end number_of_false_statements_l428_428733
