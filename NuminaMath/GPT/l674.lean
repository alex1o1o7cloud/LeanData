import Mathlib

namespace calculate_expression_l674_674083

def f (x : ℕ) : ℕ := x^2 - 3*x + 4
def g (x : ℕ) : ℕ := 2*x + 1

theorem calculate_expression : f (g 3) - g (f 3) = 23 := by
  sorry

end calculate_expression_l674_674083


namespace number_of_quadrilaterals_with_circumcenter_is_3_l674_674740

inductive QuadrilateralType
| square
| rectangle_not_square
| rhombus_not_square
| parallelogram_not_rectangle_nor_rhombus
| kite_not_rhombus
| isosceles_trapezoid_not_parallelogram

/-- Define the property of having a point equidistant from all four vertices -/
def has_circumcenter : QuadrilateralType → Prop
| QuadrilateralType.square := true
| QuadrilateralType.rectangle_not_square := true
| QuadrilateralType.rhombus_not_square := false
| QuadrilateralType.parallelogram_not_rectangle_nor_rhombus := false
| QuadrilateralType.kite_not_rhombus := false
| QuadrilateralType.isosceles_trapezoid_not_parallelogram := true

/-- Theorem statement -/
theorem number_of_quadrilaterals_with_circumcenter_is_3 :
  (Finset.filter has_circumcenter
    (Finset.univ : Finset QuadrilateralType)).card = 3 := by
  sorry

end number_of_quadrilaterals_with_circumcenter_is_3_l674_674740


namespace necessarily_negative_l674_674110

theorem necessarily_negative (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : -2 < z ∧ z < -1) : 
  y + z < 0 := 
sorry

end necessarily_negative_l674_674110


namespace interval_length_l674_674706

theorem interval_length (c d : ℝ) (h : ∃ x : ℝ, c ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ d)
  (length : (d - 4) / 3 - (c - 4) / 3 = 15) : d - c = 45 :=
by
  sorry

end interval_length_l674_674706


namespace find_derivative_at_x0_l674_674346

-- Definitions
noncomputable def is_differentiable_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
∃ (f' : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs ((f x - f x₀) / (x - x₀) - f') < ε

-- The problem statement: 
theorem find_derivative_at_x0 (f : ℝ → ℝ) (x₀ : ℝ)
  (h₁ : is_differentiable_at f x₀)
  (h₂ : ∀ (Δx : ℝ), tendsto (λ Δx, (f (x₀ + 3 * Δx) - f x₀) / Δx) (𝓝 0) (𝓝 1)) :
  deriv f x₀ = 1 / 3 :=
sorry

end find_derivative_at_x0_l674_674346


namespace circle_division_l674_674720

theorem circle_division (n : ℕ) (x : ℕ → ℤ)
  (h1 : ∀ i, 0 < i ∧ i ≤ n → x i = 1 ∨ x i = -1)
  (h2 : ∀ k, 1 ≤ k ∧ k < n → ∑ i in finset.range n, (x i) * (x ((i + k) % n)) = 0) :
  ∃ m : ℕ, n = m * m := 
sorry

end circle_division_l674_674720


namespace sum_roots_l674_674770

variable (α β : ℝ)

def is_root_log (x : ℝ) : Prop := log 3 x + x - 3 = 0
def is_root_exp (x : ℝ) : Prop := 3 ^ x + x - 3 = 0

theorem sum_roots (hα : is_root_log α) (hβ : is_root_exp β) : α + β = 3 := 
sorry

end sum_roots_l674_674770


namespace greatest_distance_A_B_is_10_l674_674438

noncomputable def complex_roots_A : set ℂ :=
{z | z^3 = 8}

noncomputable def complex_roots_B : set ℂ :=
{z | z^4 - 8*z^3 + 8*z^2 - 64*z + 128 = 0}

noncomputable def greatest_distance_between_sets (A B : set ℂ) : ℝ :=
  real.sqrt (sup ((λ a b, complex.abs (a - b)) '' (A ×ˢ B)))

theorem greatest_distance_A_B_is_10 :
  greatest_distance_between_sets complex_roots_A complex_roots_B = 10 :=
sorry

end greatest_distance_A_B_is_10_l674_674438


namespace similar_triangles_AB_proportional_l674_674748

theorem similar_triangles_AB_proportional 
  (ABC A'B'C' : Type) 
  [is_triangle ABC] 
  [is_triangle A'B'C']
  (h_sim : similar ABC A'B'C' (3/2)) 
  (h_AB' : length A'B' = 10) 
  : length AB = 15 :=
sorry

end similar_triangles_AB_proportional_l674_674748


namespace interior_lattice_points_of_triangle_l674_674816

-- Define the vertices of the triangle
def A : (ℤ × ℤ) := (0, 99)
def B : (ℤ × ℤ) := (5, 100)
def C : (ℤ × ℤ) := (2003, 500)

-- The problem is to find the number of interior lattice points
-- according to Pick's Theorem (excluding boundary points).

theorem interior_lattice_points_of_triangle :
  let I : ℤ := 0 -- number of interior lattice points
  I = 0 :=
by
  sorry

end interior_lattice_points_of_triangle_l674_674816


namespace pow_add_div_eq_l674_674284

   theorem pow_add_div_eq (a b c d e : ℕ) (h1 : b = 2) (h2 : c = 345) (h3 : d = 9) (h4 : e = 8 - 5) : 
     a = b^c + d^e -> a = 2^345 + 729 := 
   by 
     intros 
     sorry
   
end pow_add_div_eq_l674_674284


namespace find_a_l674_674479

variable a : ℝ

def A := {a^2, a+1, -3}
def B := {a-3, 2a-1, a^2+1}

theorem find_a (h : A ∩ B = {-3}) : a = -1 :=
by
  sorry

end find_a_l674_674479


namespace smallest_solution_x_abs_x_eq_3x_plus_2_l674_674301

theorem smallest_solution_x_abs_x_eq_3x_plus_2 : ∃ x : ℝ, x|x| = 3x + 2 ∧ ∀ y : ℝ, y|y| = 3y + 2 → x ≤ y :=
by
  sorry

end smallest_solution_x_abs_x_eq_3x_plus_2_l674_674301


namespace t_range_inequality_l674_674741

theorem t_range_inequality (t : ℝ) :
  (1/8) * (2 * t - t^2) ≤ -1/4 ∧ 3 - t^2 ≥ 2 ↔ -1 ≤ t ∧ t ≤ 1 - Real.sqrt 3 :=
by
  sorry

end t_range_inequality_l674_674741


namespace knight_returns_to_start_l674_674440

theorem knight_returns_to_start (castles : Type) [fintype castles] (roads : castles → fin 3 → castles)
  (start : castles) (turn_direction : castles → fin 3 → bool)
  (no_turn_same_twice : ∀ (c : castles) (r₁ r₂ : fin 3), turn_direction c r₁ ≠ turn_direction c r₂) :
  ∃ (n : ℕ), (iterate (λ c, roads c (if turn_direction c (c.2) then (c.2+1) % 3 else (c.2+2) % 3)) n (start, 0)).1 = start :=
by
  sorry

end knight_returns_to_start_l674_674440


namespace smallest_prime_factor_1729_l674_674581

theorem smallest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → p ≤ q) :=
begin
  sorry
end

end smallest_prime_factor_1729_l674_674581


namespace dot_product_calculation_l674_674018

open Real EuclideanGeometry

variable {A B C D : Point}
variable (AB AD BD BC : Vector)

theorem dot_product_calculation 
  (hABC : angle A B C = 90) 
  (hAB : ∥A -ᵥ B∥ = 4) 
  (hD_on_BC : D ∈ line_through B C) :
  (A -ᵥ B) • (A -ᵥ D) = 16 := 
by
  sorry

end dot_product_calculation_l674_674018


namespace num_divisors_180_cubed_eq_9_l674_674711

/-
  Define the prime factorization of 180. 
  From this, deduce the prime factorization of 180^3.
  Then prove that the number of positive integer divisors of 180^3 
  that have exactly 18 divisors is 9.
-/

def prime_factors_180 : ℕ × ℕ × ℕ := (2, 2, 5) -- this represents the exponent triplet (2, 2, 1)
def power_three (p : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  (p.1 * 3, p.2 * 3, p.3 * 3)

def prime_factors_180_cubed : ℕ × ℕ × ℕ :=
  power_three prime_factors_180 -- resulting in (6, 6, 3)

def num_divisors (exponents : ℕ × ℕ × ℕ) :=
  (exponents.1 + 1) * (exponents.2 + 1) * (exponents.3 + 1)

theorem num_divisors_180_cubed_eq_9 :
  ∃ count : ℕ, count = 9 ∧
    (∀ exponents (a b c : ℕ), 
      exponents = (a, b, c) ∧
      (2^a * 3^b * 5^c ∣ 2^6 * 3^6 * 5^3) ∧
      num_divisors (a, b, c) = 18) → count = 9 :=
by {
  sorry
}

end num_divisors_180_cubed_eq_9_l674_674711


namespace intersection_parallelogram_l674_674144

open_locale classical

variables {P : Type*} [add_comm_group P] [module ℝ P]

structure Parallelogram (A B C D : P) : Prop :=
(linear_comb1 : ∃ α : ℝ, α • (B - A) = C - D)
(linear_comb2 : ∃ β : ℝ, β • (C - B) = D - A)

def midpoint (A B : P) : P := (A + B) / 2

def form_parallelogram (A B C D : P) : Prop :=
∃ (α β : ℝ), α • (B - A) = D - C ∧ β • (C - B) = A - D

variables {A B C D K L M N : P}

theorem intersection_parallelogram (hA : Parallelogram A B C D)
  (hK : K = midpoint A B) (hL : L = midpoint B C)
  (hM : M = midpoint C D) (hN : N = midpoint D A) :
  form_parallelogram (intersection_point (line A L) (line C N))
                     (intersection_point (line A L) (line B M))
                     (intersection_point (line C N) (line D K))
                     (intersection_point (line B M) (line D K)) :=
sorry

end intersection_parallelogram_l674_674144


namespace goose_eggs_count_l674_674978

theorem goose_eggs_count (E : ℕ) (h1 : E % 3 = 0) 
(h2 : ((4 / 5) * (1 / 3) * E) * (2 / 5) = 120) : E = 1125 := 
sorry

end goose_eggs_count_l674_674978


namespace probability_sum_formula_l674_674466

open ProbabilityTheory

-- Definitions for the problem
variable (n k : ℕ)
variable (p q r : ℝ)
variable (ξ : ℤ → ℝ)

def distribution_ξ : ℝ → ℝ :=
  λ x, if x = 2 then p else if x = 1 then q else if x = 0 then r else 0

noncomputable def P_n (n k : ℕ) (p q r : ℝ) : ℝ :=
  ∑ j in finset.range (n + 1), 
    if |k - n| ≤ j ∧ j ≤ (n + |k - n|) / 2 then 
      (nat.choose n j) * (nat.choose (n - j) (j - |k - n|)) * p ^ j * r ^ (j - |k - n|) * q ^ (n + |k - n| - 2 * j)
    else 
      0

-- Theorem to be proven
theorem probability_sum_formula (n k : ℕ) (p q r : ℝ) (h₀ : 0 ≤ p) (h₁ : 0 ≤ q) (h₂ : 0 ≤ r) (h₃ : p + q + r = 1) :
  P_n n k p q r = ∑ j in finset.range (n + 1), 
    if |k - n| ≤ j ∧ j ≤ (n + |k - n|) / 2 then 
      (nat.choose n j) * (nat.choose (n - j) (j - |k - n|)) * p ^ j * r ^ (j - |k - n|) * q ^ (n + |k - n| - 2 * j)
    else 
      0 :=
by sorry

end probability_sum_formula_l674_674466


namespace tangent_line_at_P_l674_674090

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2

def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x

theorem tangent_line_at_P 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (h1 : P.1 + P.2 = 0)
  (h2 : f' P.1 a = -1) 
  (h3 : P.2 = f P.1 a) 
  : P = (1, -1) ∨ P = (-1, 1) := 
  sorry

end tangent_line_at_P_l674_674090


namespace decreasing_intervals_tangent_line_eq_l674_674359

-- Define the function f and its derivative.
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + 1
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Part 1: Prove intervals of monotonic decreasing.
theorem decreasing_intervals :
  (∀ x, f' x < 0 → x < -1 ∨ x > 3) := 
sorry

-- Part 2: Prove the tangent line equation.
theorem tangent_line_eq :
  15 * (-2) + (-13) + 27 = 0 :=
sorry

end decreasing_intervals_tangent_line_eq_l674_674359


namespace question1_question2_l674_674460

noncomputable def middle_term_expansion (n : ℕ) (x : ℕ) : ℕ := binomial n (n / 2) * (2 ^ (n / 2))

theorem question1 (n : ℕ) (h₁ : n = 8) : middle_term_expansion n 2 = 1120 :=
by sorry

theorem question2 (n : ℕ) (h₁ : n = 8) : 
  let a := (x + 2) ^ n, 
      sum_coeff_odd := a.coeff 1 + a.coeff 3 + a.coeff 5 + a.coeff 7 
  in sum_coeff_odd = 3280 :=
by sorry

end question1_question2_l674_674460


namespace volume_3_sphere_volume_4_ball_l674_674837

noncomputable def radius (r : ℝ) := r
noncomputable def three_sphere (w x y z r : ℝ) := w^2 + x^2 + y^2 + z^2 = r^2

theorem volume_3_sphere (r : ℝ) (h : ∀ w x y z : ℝ, three_sphere w x y z r) : 
  (2 * real.pi^2 * r^3) = 2 * real.pi^2 * r^3 := 
sorry

theorem volume_4_ball (r : ℝ) (h : ∀ w x y z : ℝ, three_sphere w x y z r) : 
  (real.pi^2 * r^4 / 2) = real.pi^2 * r^4 / 2 := 
sorry

end volume_3_sphere_volume_4_ball_l674_674837


namespace focus_of_parabola_l674_674128

theorem focus_of_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 16 * x^2) : 
    ∃ p, p = (0, 1/64) := 
by
    existsi (0, 1/64)
    -- The proof would go here, but we are adding sorry to skip it 
    sorry

end focus_of_parabola_l674_674128


namespace relationship_between_p_and_q_l674_674341

theorem relationship_between_p_and_q (p q : ℝ) (h1 : q ≠ 1) (h2 : log p + log q = log (p + q + q^2)) : 
  p = (q + q^2) / (q - 1) := 
by 
  sorry

end relationship_between_p_and_q_l674_674341


namespace tan_sixty_eq_sqrt_three_l674_674619

theorem tan_sixty_eq_sqrt_three : Real.tan (Real.pi / 3) = Real.sqrt 3 := 
by
  sorry

end tan_sixty_eq_sqrt_three_l674_674619


namespace probability_of_non_adjacent_zeros_l674_674390

-- Define the total number of arrangements of 3 ones and 2 zeros
def totalArrangements : ℕ := Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

-- Define the number of arrangements where the 2 zeros are together
def adjacentZerosArrangements : ℕ := 2 * Nat.factorial 4 / (Nat.factorial 3 * Nat.factorial 1)

-- Calculate the desired probability
def nonAdjacentZerosProbability : ℚ := 
  1 - (adjacentZerosArrangements.toRat / totalArrangements.toRat)

theorem probability_of_non_adjacent_zeros :
  nonAdjacentZerosProbability = 3/5 :=
sorry

end probability_of_non_adjacent_zeros_l674_674390


namespace first_storm_rainfall_rate_l674_674165

theorem first_storm_rainfall_rate:
  ∀ (x : ℝ), (20 * x + 375 = 975) → (x = 30) :=
begin
  intros x h,
  sorry
end

end first_storm_rainfall_rate_l674_674165


namespace circles_intersect_l674_674926

def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x + 8 * y - 8 = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 5 = 0

theorem circles_intersect :
  let c1_center := (-1, -4)
    let c1_radius := 5
    let c2_center := (2, 0)
    let c2_radius := 3
    let distance := (real.sqrt ((-1 - 2)^2 + (-4 - 0)^2))
  in distance < c1_radius + c2_radius ∧ distance > |c1_radius - c2_radius| := by
  sorry

end circles_intersect_l674_674926


namespace intervals_of_monotonicity_min_max_of_g_l674_674002

noncomputable def f (x : ℝ) : ℝ :=
  1 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * Real.sin x ^ 2

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x - (Real.pi / 6))

theorem intervals_of_monotonicity (k : ℤ) :
  (∃ a b : ℝ, k * Real.pi - (Real.pi / 3) ≤ a ∧ a ≤ k * Real.pi + (Real.pi / 6) ∧ 
              k * Real.pi + (Real.pi / 6) ≤ b ∧ b ≤ k * Real.pi + (Real.pi / 3) ∧ 
              ∀ x, a ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 6) → increasing_on f [a, k * Real.pi + (Real.pi / 6)] ∧ 
                      k * Real.pi + (Real.pi / 6) ≤ x ∧ x ≤ b → decreasing_on f [k * Real.pi + (Real.pi / 6), b]) :=
sorry

theorem min_max_of_g :
  ∀ x ∈ Icc (-Real.pi / 2) 0, g x ∈ Icc (-2 : ℝ) 1 :=
sorry

end intervals_of_monotonicity_min_max_of_g_l674_674002


namespace unique_peg_placement_l674_674099

theorem unique_peg_placement :
  ∃! f : Fin 6 → Fin 6 → Option (Fin 6), ∀ i j k, 
    (∃ c, f i k = some c) →
    (∃ c, f j k = some c) →
    i = j ∧ match f i j with
    | some c => f j k ≠ some c
    | none => True :=
  sorry

end unique_peg_placement_l674_674099


namespace ellipse_properties_and_min_AB_distance_l674_674791

-- Lean definitions for the given conditions and problem

noncomputable def a : ℝ := 3
noncomputable def b : ℝ := Real.sqrt 5
noncomputable def c : ℝ := 2

def ellipse_eq (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def point_on_ellipse_P : Prop := ellipse_eq 0 (Real.sqrt 5)

def eccentricity : ℝ := 2 / 3
def elliptical_properties : Prop := a^2 =  b^2 + c^2 ∧ (c / a) = eccentricity
def ellipse_C_eq : Prop := ∀ x y : ℝ, ellipse_eq x y → (x^2 / 9) + (y^2 / 5) = 1

def point_A : ℝ × ℝ := (4, 0) -- A moves on the line x = 4
def OA_perp_OB (a_x a_y b_x b_y : ℝ) : Prop := a_x * b_x + a_y * b_y = 0
def B_on_ellipse (m n : ℝ) : Prop := ellipse_eq m n

def AB_distance_squared (m n y : ℝ) : ℝ := (m - 4)^2 + (n - y)^2

def min_AB_distance : Prop := ∃ m n y : ℝ, 
  B_on_ellipse m n ∧ OA_perp_OB 4 y m n ∧ AB_distance_squared m n y = 21

-- The main problem statement in Lean
theorem ellipse_properties_and_min_AB_distance :
  point_on_ellipse_P ∧ elliptical_properties ∧ ellipse_C_eq ∧ min_AB_distance :=
sorry

end ellipse_properties_and_min_AB_distance_l674_674791


namespace inscribed_circle_radius_l674_674508

theorem inscribed_circle_radius (ABCD : CyclicQuadrilateral)
  (P : Point)
  (K L M N : Point)
  (R d : ℝ)
  (perpendiculars_dropped : feet_of_perpendiculars ABCD P = (K, L, M, N))
  (diagonals_perpendicular : perpendicular_diagonals ABCD P)
  (radius_circumscribed : radius_circumscribed_circle ABCD = R)
  (distance_center_intersection : distance_center_intersection_point ABCD P = d) :
  is_cyclic_quadrilateral K L M N ∧
  inscribed_circle_radius K L M N = (R^2 - d^2) / (2 * R) := sorry

end inscribed_circle_radius_l674_674508


namespace can_cut_triangle_l674_674850

theorem can_cut_triangle (ABC : Triangle) : 
  ∃ D E F : Point, 
    is_on_edge D ABC.BC ∧ 
    is_on_edge E ABC.AC ∧ 
    (is_triangle ABC.A D E) ∧
    (is_quadrilateral D E C A) ∧
    (is_pentagon A B C D E) :=
sorry

end can_cut_triangle_l674_674850


namespace carpenter_needs_80_woodblocks_l674_674996

-- Define the number of logs the carpenter currently has
def existing_logs : ℕ := 8

-- Define the number of woodblocks each log can produce
def woodblocks_per_log : ℕ := 5

-- Define the number of additional logs needed
def additional_logs : ℕ := 8

-- Calculate the total number of woodblocks needed
def total_woodblocks_needed : ℕ := 
  (existing_logs * woodblocks_per_log) + (additional_logs * woodblocks_per_log)

-- Prove that the total number of woodblocks needed is 80
theorem carpenter_needs_80_woodblocks : total_woodblocks_needed = 80 := by
  sorry

end carpenter_needs_80_woodblocks_l674_674996


namespace fraction_of_planted_area_l674_674048

-- Definitions of the conditions
def right_triangle (a b : ℕ) : Prop :=
  a * a + b * b = (Int.sqrt (a ^ 2 + b ^ 2))^2

def unplanted_square_distance (dist : ℕ) : Prop :=
  dist = 3

-- The main theorem to be proved
theorem fraction_of_planted_area (a b : ℕ) (dist : ℕ) (h_triangle : right_triangle a b) (h_square_dist : unplanted_square_distance dist) :
  (a = 5) → (b = 12) → ((a * b - dist ^ 2) / (a * b) = 412 / 1000) :=
by
  sorry

end fraction_of_planted_area_l674_674048


namespace max_actors_chess_tournament_l674_674055

-- Definitions based on conditions
variable {α : Type} [Fintype α] [DecidableEq α]

-- Each actor played with every other actor exactly once.
def played_with_everyone (R : α → α → ℝ) : Prop :=
  ∀ a b, a ≠ b → (R a b = 1 ∨ R a b = 0.5 ∨ R a b = 0)

-- Among every three participants, one earned exactly 1.5 solidus in matches against the other two.
def condition_1_5_solidi (R : α → α → ℝ) : Prop :=
  ∀ a b c, a ≠ b → b ≠ c → a ≠ c → 
   (R a b + R a c = 1.5 ∨ R b a + R b c = 1.5 ∨ R c a + R c b = 1.5)

-- Prove the maximum number of such participants is 5
theorem max_actors_chess_tournament (actors : Finset α) (R : α → α → ℝ) 
  (h_played : played_with_everyone R) (h_condition : condition_1_5_solidi R) :
  actors.card ≤ 5 :=
  sorry

end max_actors_chess_tournament_l674_674055


namespace equation_of_line_l_l674_674417

noncomputable def point := (ℝ × ℝ)
noncomputable def line := { l : ℝ × ℝ × ℝ // l ≠ (0, 0, 0) }

def line_through_point (A : point) (k : ℝ) : line :=
  ⟨(k, -1, A.2 - k * A.1), by simp only [ne.def, prod.mk.inj_iff]; linarith⟩

def distance_to_line (B : point) (l : line) : ℝ :=
  let d := l.1.1 * B.1 + l.1.2 * B.2 + l.1.3 in
  let norm := real.sqrt (l.1.1^2 + l.1.2^2) in
  |d| / norm

def perpendicular_slope (m : ℝ) : ℝ := -1 / m

theorem equation_of_line_l :
  ∃ l : line, (3, 4) ∈ {(x, y) : point | l.1.1 * x + l.1.2 * y + l.1.3 = 0} ∧
   (∀ l' : line, distance_to_line (-3, 2) l ≤ distance_to_line (-3, 2) l') ∧
   l.1.1 = 3 ∧ l.1.2 = 1 ∧ l.1.3 = -13 :=
sorry

end equation_of_line_l_l674_674417


namespace number_of_distinct_arrangements_SEES_l674_674716

theorem number_of_distinct_arrangements_SEES : 
  let SEES := "SEES" in 
  let n := 4 in 
  let k1 := 2 in  -- Number of E's
  let k2 := 2 in  -- Number of S's
  (nat.factorial n / (nat.factorial k1 * nat.factorial k2)) = 6 :=
by
  sorry

end number_of_distinct_arrangements_SEES_l674_674716


namespace moe_mowing_time_correct_l674_674881

def moe_mowing_time (length width swath_width overlap walking_rate: ℝ) : ℝ :=
  let effective_swath_width := (swath_width - overlap) / 12
  let number_of_strips := width / effective_swath_width
  let total_distance := number_of_strips * length
  let time_required := total_distance / walking_rate
  time_required

theorem moe_mowing_time_correct :
  moe_mowing_time 120 180 30 6 4500 = 2.4 :=
sorry

end moe_mowing_time_correct_l674_674881


namespace rectangle_area_l674_674606

theorem rectangle_area
  (b : ℝ)
  (l : ℝ)
  (P : ℝ)
  (h1 : l = 3 * b)
  (h2 : P = 2 * (l + b))
  (h3 : P = 112) :
  l * b = 588 := by
  sorry

end rectangle_area_l674_674606


namespace system_of_equations_solution_l674_674116

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (4 * x - 7 * y = -14) ∧ (5 * x + 3 * y = -7) ∧ (x = -91/47) ∧ (y = -42/47) :=
by
  use -91/47, -42/47
  split
  · norm_num
  split
  · norm_num
  split
  · refl
  · refl

end system_of_equations_solution_l674_674116


namespace sum_of_ages_l674_674856

theorem sum_of_ages (a b c : ℕ) (twin : a = b) (product : a * b * c = 256) : a + b + c = 20 := by
  sorry

end sum_of_ages_l674_674856


namespace at_most_eight_roots_of_abs_sum_eq_abs_l674_674808

theorem at_most_eight_roots_of_abs_sum_eq_abs 
  (P1 P2 P3 : ℝ → ℝ) 
  (hP1 : ∃ (b1 c1 : ℝ), ∀ x, P1 x = x^2 + b1 * x + c1) 
  (hP2 : ∃ (b2 c2 : ℝ), ∀ x, P2 x = x^2 + b2 * x + c2) 
  (hP3 : ∃ (b3 c3 : ℝ), ∀ x, P3 x = x^2 + b3 * x + c3) : 
  ∀ x : ℝ, (|P1 x| + |P2 x| = |P3 x|) → Fintype.card { x : ℝ // |P1 x| + |P2 x| = |P3 x| } ≤ 8 :=
sorry

end at_most_eight_roots_of_abs_sum_eq_abs_l674_674808


namespace jelly_cost_l674_674725

theorem jelly_cost (N B J : ℕ) (hN_gt_1 : N > 1) (h_cost_eq : N * (3 * B + 7 * J) = 252) : 7 * N * J = 168 := by
  sorry

end jelly_cost_l674_674725


namespace jill_second_bus_ride_time_l674_674948

theorem jill_second_bus_ride_time :
  let wait_time := 12
  let ride_time := 30
  let combined_time := wait_time + ride_time
  let second_bus_ride_time := combined_time / 2
  second_bus_ride_time = 21 :=
by
  -- Introduce the definitions
  let wait_time := 12
  let ride_time := 30
  let combined_time := wait_time + ride_time
  let second_bus_ride_time := combined_time / 2
  -- State and confirm the goal
  have : second_bus_ride_time = 21 := by rfl
  exact this
  sorry

end jill_second_bus_ride_time_l674_674948


namespace negation_equivalence_l674_674554

noncomputable def negation_proposition (N : Type) [Nat N] :=
  ∃ m0 : N, N.sqrt (m0 ^ 2 + 1)

theorem negation_equivalence (N : Type) [Nat N] :
  ¬ (negation_proposition N) ↔ ∀ m : N, ¬ (N.sqrt (m ^ 2 + 1)) := 
by
  sorry

end negation_equivalence_l674_674554


namespace log_identity_l674_674705

theorem log_identity :
  (Real.log 25 / Real.log 10) - 2 * (Real.log (1 / 2) / Real.log 10) = 2 :=
by
  sorry

end log_identity_l674_674705


namespace sum_of_roots_l674_674772

theorem sum_of_roots : ∀ (α β : ℝ), (log 3 α + α - 3 = 0) → (3^β + β - 3 = 0) → α + β = 3 :=
by
  assume α β hα hβ
  sorry

end sum_of_roots_l674_674772


namespace find_matrix_A_l674_674732

-- Define the condition that A v = 3 v for all v in R^3
def satisfiesCondition (A : Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  ∀ (v : Fin 3 → ℝ), A.mulVec v = 3 • v

theorem find_matrix_A (A : Matrix (Fin 3) (Fin 3) ℝ) :
  satisfiesCondition A → A = 3 • 1 :=
by
  intro h
  sorry

end find_matrix_A_l674_674732


namespace find_ellipse_l674_674544

noncomputable def ellipse := {a b : ℝ // a > 0 ∧ b > 0 ∧ a ≠ b}
def line (a b : ℝ) := ∀ x y : ℝ, a*x^2 + b*y^2 = 1 ∧ x + y = 1
def distance_AB := 2 * Real.sqrt 2
def slope_OC := Real.sqrt 2 / 2

theorem find_ellipse (a b : ℝ) (h1 : {a b // a > 0 ∧ b > 0 ∧ a ≠ b}) :
  line a b (a, b) →
  distance_AB = 2 * Real.sqrt 2 →
  slope_OC = Real.sqrt 2 / 2 →
  a = 1/3 ∧ b = Real.sqrt 2 / 3 := sorry

end find_ellipse_l674_674544


namespace sufficient_but_not_necessary_condition_l674_674470

variables {R : Type} [CommRing R]

def even (f : R → R) : Prop :=
∀ x, f(-x) = f(x)

def even_condition (f g : R → R) : Prop :=
f (-x) = f x ∧ g (-x) = g x

-- Define h function
def h (f g : R → R) : R → R := λ x, f x + g x

theorem sufficient_but_not_necessary_condition (f g : R → R) (x : R) :
  even (f) → even (g) → even (h f g) ∧ ∃ f g, ¬ (even (f)) ∧ ¬ (even (g)) ∧ even (h f g) :=
by
  sorry

end sufficient_but_not_necessary_condition_l674_674470


namespace sum_of_all_possible_values_of_g10_l674_674710

noncomputable def g : ℕ → ℝ := sorry

axiom h1 : g 1 = 2
axiom h2 : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = 3 * (g m + g n)
axiom h3 : g 0 = 0

theorem sum_of_all_possible_values_of_g10 : g 10 = 59028 :=
by
  sorry

end sum_of_all_possible_values_of_g10_l674_674710


namespace fastest_route_time_l674_674905

theorem fastest_route_time (d1 d2 : ℕ) (s1 s2 : ℕ) (h1 : d1 = 1500) (h2 : d2 = 750) (h3 : s1 = 75) (h4 : s2 = 25) :
  min (d1 / s1) (d2 / s2) = 20 := by
  sorry

end fastest_route_time_l674_674905


namespace evaluate_expression_l674_674171

noncomputable def a : ℕ := 3^2 + 5^2 + 7^2
noncomputable def b : ℕ := 2^2 + 4^2 + 6^2

theorem evaluate_expression : (a / b : ℚ) - (b / a : ℚ) = 3753 / 4656 :=
by
  sorry

end evaluate_expression_l674_674171


namespace minimum_s_value_l674_674894

-- Define the conditions
def point_P := (Real.pi / 4, Real.sin (Real.pi / 4 - Real.pi / 12))
def shifted_point (s : Real) := (Real.pi / 4 - s, Real.sin (Real.pi / 4 - s - Real.pi / 12))

-- The proof statement
theorem minimum_s_value (s : Real) (t : Real) 
  (h1 : shifted_point s = (Real.pi / 2 - 2 * s, Real.sin (Real.pi / 2 - 2 * s)))
  (h2 : s > 0) : t = 1/2 ∧ s = Real.pi / 6 :=
sorry

end minimum_s_value_l674_674894


namespace length_BD_fraction_of_AC_l674_674505

noncomputable def line_segment_AD_fraction_BDAC (BD AC : ℝ) : Prop :=
  BD / AC = 2 / 7

theorem length_BD_fraction_of_AC :
  ∀ (AB BD AC CD AD : ℝ),
  AB = 3 * BD →
  AC = 7 * CD →
  AD = 40 →
  AD = AB + BD →
  AD = AC + CD →
  line_segment_AD_fraction_BDAC BD AC :=
by
  intros AB BD AC CD AD hAB hAC hAD1 hAD2 hAD3
  rw [hAB, hAC, hAD2, hAD3, hAD1] at hAD2 hAD3
  sorry

end length_BD_fraction_of_AC_l674_674505


namespace monthly_salary_is_6000_l674_674596

-- Definitions from conditions
variables (S : ℝ) (H : 0.20 * S = 240 ⇔ S = 6000)

-- Statement to prove
theorem monthly_salary_is_6000 :
  (20% of salary as savings) ->
  (savings reduced to 240 due to 20% expense increase) ->
  S = 6000 :=
by
  intros H1 H2
  sorry

end monthly_salary_is_6000_l674_674596


namespace find_constants_l674_674878

variable (A B Q : ℝ → ℝ)
variable (x y : ℝ)
variable (h1 : ∀ A B Q : ℝ → ℝ, AQ:QB = 7:2)
variable (h2 : ∃ (x y : ℝ), Q = x * A + y * B)

noncomputable def solution : Prop :=
∃ (x y : ℝ), (Q = x * A + y * B) ∧ x = -2/5 ∧ y = 7/5

theorem find_constants (A B Q : ℝ → ℝ) (h1 : AQ:QB = 7:2) (h2: ∃ (x y: ℝ), Q = x * A + y * B): solution A B Q x y :=
sorry

end find_constants_l674_674878


namespace inequality_am_gm_l674_674895

theorem inequality_am_gm (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b)^2 / 2 + (a + b) / 4 ≥ a * real.sqrt b + b * real.sqrt a :=
by {
  sorry -- The proof is omitted as per the instructions
}

end inequality_am_gm_l674_674895


namespace relationship_among_abc_l674_674312

noncomputable def a : ℝ := (0.7 : ℝ)^(1 / 2)
noncomputable def b : ℝ := (0.2 : ℝ)^(-2)
noncomputable def c : ℝ := (Real.log 0.7) / (Real.log 3)

theorem relationship_among_abc : c < a ∧ a < b := by
  sorry

end relationship_among_abc_l674_674312


namespace tangent_line_at_2_l674_674287

def f (x : ℝ) : ℝ := x - (2 / x)

theorem tangent_line_at_2 : ∃ (a b c : ℝ), a = 3 ∧ b = -2 ∧ c = -4 ∧ ∀ (x y : ℝ), y = f(2) + f'(2) * (x - 2) → a * x + b * y + c = 0 :=
by
  let f' := sorry -- The derivative of f(x)
  have h : f'(2) = 3 / 2 := sorry -- The value of the derivative at x = 2
  have hy : f 2 = 1 := sorry -- The value of the function at x = 2
  use [3, -2, -4]
  split
  rfl
  split
  rfl
  split
  rfl
  intro x y
  intro hy_tangent
  sorry

end tangent_line_at_2_l674_674287


namespace find_second_number_in_second_set_l674_674534

theorem find_second_number_in_second_set :
    (14 + 32 + 53) / 3 = 3 + (21 + x + 22) / 3 → x = 47 :=
by intro h
   sorry

end find_second_number_in_second_set_l674_674534


namespace count_five_digit_progressive_numbers_find_110th_five_digit_progressive_number_l674_674624

def is_progressive_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 : ℕ), 1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 ≤ 9 ∧
                          n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5

theorem count_five_digit_progressive_numbers : ∃ n, n = 126 :=
by
  sorry

theorem find_110th_five_digit_progressive_number : ∃ n, n = 34579 :=
by
  sorry

end count_five_digit_progressive_numbers_find_110th_five_digit_progressive_number_l674_674624


namespace negation_of_forall_2_pow_x_pos_l674_674555

theorem negation_of_forall_2_pow_x_pos :
  ¬ (∀ x : ℝ, 2^x > 0) ↔ ∃ x : ℝ, 2^x ≤ 0 :=
sorry

end negation_of_forall_2_pow_x_pos_l674_674555


namespace union_M_N_l674_674805

def M : Set ℝ := { x | |x - 1| ≤ 1 }

def N : Set ℝ := { x | x^2 - 1 > 0 }

theorem union_M_N : M ∪ N = (Set.univ \ (-1,1)) := by
  have hM : M = Set.Icc 0 2 := sorry
  have hN : N = (Set.Iio (-1) ∪ Set.Ioi 1) := sorry
  sorry

end union_M_N_l674_674805


namespace price_of_water_margin_comic_books_l674_674920

/-
The excellent traditional Chinese culture is the "root" and "soul" of the Chinese nation, 
which is the cultural root and gene that must be passed down from generation to generation. 
In order to inherit the excellent traditional culture, 
a certain school purchased several sets of "Romance of the Three Kingdoms" 
and "Water Margin" comic books for each class.
The price of each set of "Romance of the Three Kingdoms" comic books 
is $60 more expensive than the price of each set of "Water Margin" comic books. 
It is known that the school spent $3600 on "Romance of the Three Kingdoms" comic books 
and $4800 on "Water Margin" comic books. 
The number of sets of "Romance of the Three Kingdoms" comic books purchased 
is half the number of sets of "Water Margin" comic books purchased.
-/

variable (x : ℕ)
variable (c : ℕ)
variable (price_water_margin : ℕ)
variable (price_romance : ℕ)
variable (n_water_margin : ℕ)
variable (n_romance : ℕ)

def conditions := 
  (price_water_margin = x) ∧
  (price_romance = x + 60) ∧
  (n_romance = 3600 / price_romance) ∧
  (n_water_margin = 4800 / price_water_margin) ∧
  (n_romance = 1 / 2 * n_water_margin)

theorem price_of_water_margin_comic_books : 
  ∀ (x : ℕ), conditions x → price_water_margin = 120 := 
by
  sorry

end price_of_water_margin_comic_books_l674_674920


namespace probability_non_adjacent_zeros_l674_674387

theorem probability_non_adjacent_zeros (total_ones total_zeros : ℕ) (h₁ : total_ones = 3) (h₂ : total_zeros = 2) : 
  (total_zeros != 0 ∧ total_ones != 0 ∧ total_zeros + total_ones = 5) → 
  (prob_non_adjacent (total_ones + total_zeros) total_zeros = 0.6) :=
by
  sorry

def prob_non_adjacent (total num_zeros: ℕ) : ℚ :=
  let total_arrangements := (Nat.factorial total) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros)))
  let adjacent_arrangements := (Nat.factorial (total - num_zeros + 1)) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros - 1)))
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements / total_arrangements

end probability_non_adjacent_zeros_l674_674387


namespace initial_percentage_increase_l674_674234

theorem initial_percentage_increase (W R : ℝ) (P : ℝ) 
  (h1 : R = W * (1 + P / 100)) 
  (h2 : R * 0.75 = W * 1.3500000000000001) : P = 80 := 
by
  sorry

end initial_percentage_increase_l674_674234


namespace vertex_of_parabola_l674_674541

def parabola_vertex (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - (b^2) / (4 * a)
  (h, k)

theorem vertex_of_parabola : parabola_vertex 1 0 (-9) = (0, -9) := by
  sorry

end vertex_of_parabola_l674_674541


namespace reduced_price_after_discount_l674_674976

theorem reduced_price_after_discount (P R : ℝ) (h1 : R = 0.8 * P) (h2 : 1500 / R - 1500 / P = 10) :
  R = 30 := 
by
  sorry

end reduced_price_after_discount_l674_674976


namespace original_number_is_l674_674839

theorem original_number_is (a b c : ℕ) (h : 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10) (N : ℕ) :
  N = 3194 → 
  N = (100 * a + 10 * b + c) + 
      (100 * a + 10 * c + b) + 
      (100 * b + 10 * a + c) + 
      (100 * b + 10 * c + a) + 
      (100 * c + 10 * a + b) + 
      (100 * c + 10 * b + a) 
 \rightarrow  (a = 3 ∧ b = 5 ∧ c = 8) :=
begin
  intros hN perm_sum,
  sorry
end

end original_number_is_l674_674839


namespace coeff_x2_in_expansion_l674_674713

-- Defining the problem conditions and the main theorem
theorem coeff_x2_in_expansion :
  coeff (expand ((x + 1)^5 * (x - 2))) 2 = -15 := by
sorry

end coeff_x2_in_expansion_l674_674713


namespace polynomial_properties_l674_674802

noncomputable def P (x : ℝ) : ℝ := 3 * x^4 + 0 * x^3 + 12 * x^2 + 0 * x - 15
noncomputable def Q (x : ℝ) : ℝ := 12 * x^3 + 24 * x

noncomputable def p : ℝ := 3
noncomputable def q : ℝ := 0
noncomputable def r : ℝ := 12
noncomputable def s : ℝ := 0
noncomputable def t : ℝ := -15

theorem polynomial_properties :
  P (real.sqrt (-5)) = 0 ∧ 
  Q (real.sqrt (-2)) = 0 ∧
  ∫ x in 0..1, P x = -52 / 5 :=
by
  sorry

end polynomial_properties_l674_674802


namespace equal_ivan_petrovich_and_peter_ivanovich_l674_674452

theorem equal_ivan_petrovich_and_peter_ivanovich :
  (∀ n : ℕ, n % 10 = 0 → (n % 20 = 0) = (n % 200 = 0)) :=
by
  sorry

end equal_ivan_petrovich_and_peter_ivanovich_l674_674452


namespace eduardo_frankie_classes_total_l674_674723

theorem eduardo_frankie_classes_total (eduardo_classes : ℕ) (h₁ : eduardo_classes = 3) 
                                       (h₂ : ∀ frankie_classes, frankie_classes = 2 * eduardo_classes) :
  ∃ total_classes : ℕ, total_classes = eduardo_classes + 2 * eduardo_classes := 
by
  use 3 + 2 * 3
  sorry

end eduardo_frankie_classes_total_l674_674723


namespace Paige_recycled_pounds_l674_674104

-- Definitions based on conditions from step a)
def points_per_pound := 1 / 4
def friends_pounds_recycled := 2
def total_points := 4

-- The proof statement (no proof required)
theorem Paige_recycled_pounds :
  let total_pounds_recycled := total_points * 4
  let paige_pounds_recycled := total_pounds_recycled - friends_pounds_recycled
  paige_pounds_recycled = 14 :=
by
  sorry

end Paige_recycled_pounds_l674_674104


namespace arrangement_problem_l674_674939
   
   def numberOfArrangements (n : Nat) : Nat :=
     n.factorial

   def exclusiveArrangements (total people : Nat) (positions : Nat) : Nat :=
     (positions.choose 2) * (total - 2).factorial

   theorem arrangement_problem : 
     (numberOfArrangements 5) - (exclusiveArrangements 5 3) = 84 := 
   by
     sorry
   
end arrangement_problem_l674_674939


namespace olaf_and_dad_total_score_l674_674493

variable (dad_score : ℕ)
variable (olaf_score : ℕ)
variable (total_score : ℕ)

-- Define conditions based on the problem
def condition1 : Prop := dad_score = 7
def condition2 : Prop := olaf_score = 3 * dad_score
def condition3 : Prop := total_score = olaf_score + dad_score

-- Define the theorem to be proven
theorem olaf_and_dad_total_score : condition1 ∧ condition2 ∧ condition3 → total_score = 28 := by
  intro h
  cases h with
    | intro h1 h'
    | intro h2 h3 =>
        -- Using the conditions given
        unfold condition1 at h1
        unfold condition2 at h2
        unfold condition3 at h3
        sorry

end olaf_and_dad_total_score_l674_674493


namespace sum_of_digits_28561_base_ne_37_l674_674971

theorem sum_of_digits_28561_base_ne_37 (b : ℕ) (hb : b > 29) : 
  let sum_digits (n b : ℕ) := 
    nat.digits b n |>.sum 
  in sum_digits 28561 b ≠ 37 :=
sorry

end sum_of_digits_28561_base_ne_37_l674_674971


namespace income_of_deceased_member_l674_674913

theorem income_of_deceased_member 
    (n1 n2 i1 i2: ℕ)
    (h1 : n1 = 4) 
    (h2 : i1 = 840)
    (h3 : n2 = 3)
    (h4 : i2 = 650)
    (h : n1 * i1 - n2 * i2 = 1410) :
    1410 = n1 * i1 - n2 * i2 :=
by
    rw [←h]
    refl

end income_of_deceased_member_l674_674913


namespace parabola_focus_distance_l674_674419

theorem parabola_focus_distance (x_0 p : ℝ) (h1 : (sqrt 2)^2 = 2 * p * x_0)
  (h2 : 3 * x_0 = x_0 + p / 2) (h3 : p > 0) : p = 2 := sorry

end parabola_focus_distance_l674_674419


namespace not_all_odd_l674_674176

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1
def divides (a b c d : ℕ) : Prop := a = b * c + d ∧ 0 ≤ d ∧ d < b

theorem not_all_odd (a b c d : ℕ) 
  (h_div : divides a b c d)
  (h_odd_a : is_odd a)
  (h_odd_b : is_odd b)
  (h_odd_c : is_odd c)
  (h_odd_d : is_odd d) :
  False :=
sorry

end not_all_odd_l674_674176


namespace radius_of_fourth_circle_is_12_l674_674444

theorem radius_of_fourth_circle_is_12 (r : ℝ) (radii : Fin 7 → ℝ) 
  (h_geometric : ∀ i, radii (Fin.succ i) = r * radii i) 
  (h_smallest : radii 0 = 6)
  (h_largest : radii 6 = 24) :
  radii 3 = 12 :=
by
  sorry

end radius_of_fourth_circle_is_12_l674_674444


namespace olaf_and_dad_total_score_l674_674494

variable (dad_score : ℕ)
variable (olaf_score : ℕ)
variable (total_score : ℕ)

-- Define conditions based on the problem
def condition1 : Prop := dad_score = 7
def condition2 : Prop := olaf_score = 3 * dad_score
def condition3 : Prop := total_score = olaf_score + dad_score

-- Define the theorem to be proven
theorem olaf_and_dad_total_score : condition1 ∧ condition2 ∧ condition3 → total_score = 28 := by
  intro h
  cases h with
    | intro h1 h'
    | intro h2 h3 =>
        -- Using the conditions given
        unfold condition1 at h1
        unfold condition2 at h2
        unfold condition3 at h3
        sorry

end olaf_and_dad_total_score_l674_674494


namespace parallel_line_correct_perpendicular_line_correct_parallel_line_through_point0_perpendicular_line_through_point0_l674_674551

-- Definitions for the conditions
def line1 (x : ℝ) : ℝ := 3*x - 4
def point0 : ℝ × ℝ := (1, 2)

-- The parallel line equation passing through point0
def parallel_line_equation (x : ℝ) : ℝ := 3*x - 1

-- The perpendicular line equation passing through point0
def perpendicular_line_equation (x : ℝ) : ℝ := -(1/3)*x + 7/3

-- The Lean statement for the proof requirements
theorem parallel_line_correct :
  ∀ x, parallel_line_equation x = 3*x - 1 :=
by
  intros x
  simp[parallel_line_equation]

theorem perpendicular_line_correct :
  ∀ x, perpendicular_line_equation x = -(1/3)*x + 7/3 :=
by
  intros x
  simp[perpendicular_line_equation]

-- Verifying the lines pass through point0
theorem parallel_line_through_point0 :
  parallel_line_equation 1 = 2 :=
by
  simp[parallel_line_equation]
  norm_num

theorem perpendicular_line_through_point0 :
  perpendicular_line_equation 1 = 2 :=
by
  simp[perpendicular_line_equation]
  norm_num

end parallel_line_correct_perpendicular_line_correct_parallel_line_through_point0_perpendicular_line_through_point0_l674_674551


namespace combine_sqrt_with_2sqrt3_l674_674179

theorem combine_sqrt_with_2sqrt3 :
  (∃ k, sqrt (8 : ℝ) = k * sqrt (3 : ℝ)) ∨
  (∃ k, sqrt (18 : ℝ) = k * sqrt (3 : ℝ)) ∨
  (∃ k, sqrt (9 : ℝ) = k * sqrt (3 : ℝ)) ∨
  (∃ k, sqrt (1/3 : ℝ) = k * sqrt (3 : ℝ)) :=
by
  right
  right
  right
  use 1/3
  sorry

end combine_sqrt_with_2sqrt3_l674_674179


namespace roof_length_width_difference_l674_674607

theorem roof_length_width_difference :
  ∀ (w l : ℝ), l = 5 * w ∧ l * w = 676 → l - w ≈ 46.52 := by
  intros w l h
  cases h with hw hl
  sorry

end roof_length_width_difference_l674_674607


namespace problem1_problem2_problem3_problem4_l674_674264

theorem problem1 : 25 - 9 + (-12) - (-7) = 4 := by
  sorry

theorem problem2 : (1 / 9) * (-2)^3 / ((2 / 3)^2) = -2 := by
  sorry

theorem problem3 : ((5 / 12) + (2 / 3) - (3 / 4)) * (-12) = -4 := by
  sorry

theorem problem4 : -(1^4) + (-2) / (-1/3) - |(-9)| = -4 := by
  sorry

end problem1_problem2_problem3_problem4_l674_674264


namespace pow_nat_continuous_everywhere_l674_674517

theorem pow_nat_continuous_everywhere (n : ℕ) : ∀ a : ℝ, ContinuousAt (λ x : ℝ, x^n) a :=
by
  sorry

end pow_nat_continuous_everywhere_l674_674517


namespace probability_non_adjacent_zeros_l674_674384

theorem probability_non_adjacent_zeros (total_ones total_zeros : ℕ) (h₁ : total_ones = 3) (h₂ : total_zeros = 2) : 
  (total_zeros != 0 ∧ total_ones != 0 ∧ total_zeros + total_ones = 5) → 
  (prob_non_adjacent (total_ones + total_zeros) total_zeros = 0.6) :=
by
  sorry

def prob_non_adjacent (total num_zeros: ℕ) : ℚ :=
  let total_arrangements := (Nat.factorial total) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros)))
  let adjacent_arrangements := (Nat.factorial (total - num_zeros + 1)) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros - 1)))
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements / total_arrangements

end probability_non_adjacent_zeros_l674_674384


namespace dodecagon_primes_not_possible_l674_674446

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def adjacent_pairs (a : Fin 12 → ℕ) : Prop :=
  ∀ i : Fin 12, is_prime (a i + a (Fin.mod (i + 1) 12))
  
def pairs_with_two_between (a : Fin 12 → ℕ) : Prop :=
  ∀ i : Fin 12, is_prime (a i + a (Fin.mod (i + 3) 12))

theorem dodecagon_primes_not_possible (a : Fin 12 → ℕ) (h_unique : Function.Injective a) 
  (h_range : ∀ i, a i ∈ Finset.range 1 13) : 
  ¬ (adjacent_pairs a ∧ pairs_with_two_between a) :=
sorry

end dodecagon_primes_not_possible_l674_674446


namespace determine_truck_loading_l674_674514

noncomputable def loading_time_and_truck_count (v : ℝ) (D : ℝ) (t : ℝ) (n : ℕ) : Prop :=
  let speed_of_loaded := (6/7) * v in
  let total_trip_time := (D / v) + (7 * D / (6 * v)) in
  let petrov_return_time := 6 in
  let second_meeting_time := 40 in
  let ivanov_meeting_time_lower_bound := 16 in
  let ivanov_meeting_time_upper_bound := 19 in
  t = 13 ∧ n = 5 ∧
  (petrov_return_time <= (D / v + 7 * D / (6 * v))) ∧
  (second_meeting_time - petrov_return_time) <= (ivanov_meeting_time_upper_bound - ivanov_meeting_time_lower_bound)

theorem determine_truck_loading (v D : ℝ) :
  ∃ t n, loading_time_and_truck_count v D t n :=
by {
  let t := 13,
  let n := 5,
  use t,
  use n,
  unfold loading_time_and_truck_count,
  sorry
}

end determine_truck_loading_l674_674514


namespace ratio_of_perimeter_to_length_XY_l674_674051

noncomputable def XY : ℝ := 17
noncomputable def XZ : ℝ := 8
noncomputable def YZ : ℝ := 15
noncomputable def ZD : ℝ := 240 / 17

-- Defining the perimeter P
noncomputable def P : ℝ := 17 + 2 * (240 / 17)

-- Finally, the statement with the ratio in the desired form
theorem ratio_of_perimeter_to_length_XY : 
  (P / XY) = (654 / 289) :=
by
  sorry

end ratio_of_perimeter_to_length_XY_l674_674051


namespace max_value_of_function_y_corresponding_x_value_l674_674295

noncomputable def function_y (x : ℝ) : ℝ :=
sin x ^ 2 + 3 * sin x * cos x + 4 * cos x ^ 2

theorem max_value_of_function_y :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ π / 2 ∧ function_y x = (5 + 3 * real.sqrt 2) / 2 :=
begin
  sorry
end

theorem corresponding_x_value :
  function_y (π / 8) = (5 + 3 * real.sqrt 2) / 2 :=
begin
  sorry
end

end max_value_of_function_y_corresponding_x_value_l674_674295


namespace greatest_integer_value_l674_674578

theorem greatest_integer_value (x : ℤ) : (7 - 6 * x > 23) → x = -3 :=
begin
  sorry
end

end greatest_integer_value_l674_674578


namespace remainder_55_57_div_8_l674_674979

def remainder (a b n : ℕ) := (a * b) % n

theorem remainder_55_57_div_8 : remainder 55 57 8 = 7 := by
  -- proof omitted
  sorry

end remainder_55_57_div_8_l674_674979


namespace numerator_harmonic_prime_divisible_p_l674_674600

theorem numerator_harmonic_prime_divisible_p (p : ℕ) (hp : Nat.Prime p) (hp_gt_2 : p > 2) :
  ∃ m n : ℕ, (1 + ∑ i in Finset.range (p - 1), (1 : ℚ) / (i.succ)) = (m : ℚ) / (n : ℚ) ∧ p ∣ m :=
sorry

end numerator_harmonic_prime_divisible_p_l674_674600


namespace valid_3_word_sentences_in_gnollish_l674_674122

-- Definitions of the words in the Gnollish language
inductive Gnollish : Type
| splargh
| glumph
| amr

-- Define a sentence as a list of three Gnollish words
def sentence : Type := list Gnollish

-- Conditions for invalid sentences
def invalid_sentence1 (s : sentence) : Prop :=
  match s with
  | [_, Gnollish.splargh, Gnollish.glumph] => true
  | [Gnollish.splargh, Gnollish.glumph, _] => true
  | _ => false

def invalid_sentence2 (s : sentence) : Prop :=
  match s with
  | [_, Gnollish.amr, Gnollish.glumph] => true
  | [Gnollish.amr, Gnollish.glumph, _] => true
  | _ => false

-- Define the main theorem
theorem valid_3_word_sentences_in_gnollish :
  let total_sentences := list.length (list.zip (list.zip Gnollish.values Gnollish.values) Gnollish.values) in
  let invalid_sentences :=
    finset.filter (λ s, invalid_sentence1 s ∨ invalid_sentence2 s) (finset.univ : finset sentence) in
  total_sentences - finset.card invalid_sentences = 16 :=
sorry  -- proof goes here

end valid_3_word_sentences_in_gnollish_l674_674122


namespace cookie_circle_radius_area_l674_674532

theorem cookie_circle_radius_area :
  (∀ (x y : ℝ), x^2 + y^2 - 10 = 2x + 6y) →
  ∃ (r : ℝ), r = 2 * Real.sqrt 5 ∧ (∃ (A : ℝ), A = Real.pi * r^2 ∧ A = 20 * Real.pi) :=
by
  sorry

end cookie_circle_radius_area_l674_674532


namespace soldiers_line_l674_674227

theorem soldiers_line (n x y z : ℕ) (h₁ : y = 6 * x) (h₂ : y = 7 * z)
                      (h₃ : n = x + y) (h₄ : n = 7 * x) (h₅ : n = 8 * z) : n = 98 :=
by 
  sorry

end soldiers_line_l674_674227


namespace inverse_shifted_point_l674_674547

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def inverse_function (f g : ℝ → ℝ) : Prop := ∀ y, f (g y) = y ∧ ∀ x, g (f x) = x

theorem inverse_shifted_point
  (f : ℝ → ℝ)
  (hf_odd : odd_function f)
  (hf_point : f (-1) = 3)
  (g : ℝ → ℝ)
  (hg_inverse : inverse_function f g) :
  g (2 - 5) = 1 :=
by
  sorry

end inverse_shifted_point_l674_674547


namespace spider_total_distance_l674_674652

-- Define points where spider starts and moves
def start_position : ℤ := 3
def first_move : ℤ := -4
def second_move : ℤ := 8
def final_move : ℤ := 2

-- Define the total distance the spider crawls
def total_distance : ℤ :=
  |first_move - start_position| +
  |second_move - first_move| +
  |final_move - second_move|

-- Theorem statement
theorem spider_total_distance : total_distance = 25 :=
sorry

end spider_total_distance_l674_674652


namespace train_length_1200_l674_674241

def length_of_train (t_tree t_plat : ℕ) (d_plat : ℕ) (L : ℕ) : Prop :=
  (L / t_tree) = ((L + d_plat) / t_plat)

theorem train_length_1200:
  ∃ L, (length_of_train 120 170 500 L) ∧ L = 1200 :=
begin
  sorry
end

end train_length_1200_l674_674241


namespace houses_with_garage_l674_674042

theorem houses_with_garage (P GP N : ℕ) (hP : P = 40) (hGP : GP = 35) (hN : N = 10) 
    (total_houses : P + GP - GP + N = 65) : 
    P + 65 - P - GP + GP - N = 50 :=
by
  sorry

end houses_with_garage_l674_674042


namespace player_A_wins_l674_674642

def canPlayerAWin (x : ℝ) : Prop :=
  ∃ (n k : ℕ), n % 2 = 1 ∧ x = n / (2 ^ k : ℝ) ∧ 0 ≤ x ∧ x ≤ 1

theorem player_A_wins (x : ℝ) :
  (x > 0 ∧ x < 1) → canPlayerAWin x :=
begin
  sorry
end

end player_A_wins_l674_674642


namespace find_xy_pairs_l674_674729

theorem find_xy_pairs (x y: ℝ) (h₀: 0 < x ∧ x < π / 2)  (h₁: 0 < y ∧ y < π / 2)
  (h2: cos x / cos y = 2 * cos y * cos y)
  (h3: sin x / sin y = 2 * sin y * sin y):
  x = π / 4 ∧ y = π / 4 := 
by
  sorry

end find_xy_pairs_l674_674729


namespace projection_problems_l674_674168

noncomputable def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := λ u v : ℝ × ℝ, (u.1 * v.1 + u.2 * v.2)
  let scalar := dot_product a b / dot_product b b
  (scalar * b.1, scalar * b.2)

theorem projection_problems :
  let v := (5, -5)
  proj (-3, 2) v = (-2.5, 2.5) ∧ proj (1, -4) v = (1.5, -1.5) :=
by
  sorry

end projection_problems_l674_674168


namespace grant_percentage_increase_l674_674256

theorem grant_percentage_increase
  (Parker_throw : ℝ)
  (Kyle_throw_more : ℝ)
  (Kyle_factor : ℝ)
  (h_Parker : Parker_throw = 16)
  (h_Kyle_throw_more : Kyle_throw_more = 24)
  (h_Kyle_factor : Kyle_factor = 2) :
  (Kyle_factor * (Kyle_throw_more + Parker_throw) / Kyle_factor - Parker_throw) / Parker_throw * 100 = 25 :=
by
  have h1 : Kyle_throw_more + Parker_throw = 40, by { rw [h_Parker, h_Kyle_throw_more], norm_num }
  have h2 : Kyle_factor * (Kyle_throw_more + Parker_throw) = 2 * 40, by { rw h_Kyle_factor, simp [h1] }
  have h3 : (Kyle_factor * (Kyle_throw_more + Parker_throw)) / Kyle_factor = 40, by { rw [h2, h_Kyle_factor], norm_num }
  have h_Grant_throw: (Kyle_factor * (Kyle_throw_more + Parker_throw) / Kyle_factor) - Parker_throw = 24, by { simp [h3, h_Parker] }
  have h4 : 24 / Parker_throw = 1.5, by { rw h_Parker, norm_num }
  have h5 : 1.5 * 100 = 150, by norm_num
  rw h5, norm_num
  sorry

end grant_percentage_increase_l674_674256


namespace proof_problem_l674_674005

   -- Define the function and conditions
   noncomputable def f (x : ℝ) : ℝ := A * Real.sin (ω * x + Real.pi / 6)
   variable {A ω : ℝ}
   variable (A_pos : A > 0) (omega_pos : ω > 0)

   -- Define the two specific conditions
   def condition_1 := ∀ x, f x ≤ 2
   def condition_2 := ∃ d : ℝ, (d > 0 ∧ d = π / (2 * ω))

   -- Define the equivalent proof problem
   theorem proof_problem :
     (A = 2) →
     (ω = 2) →
     (f = λ x, 2 * Real.sin (2 * x + Real.pi / 6)) →
     (∀ x ∈ Set.Icc 0 (Real.pi / 4), 1 ≤ f x ∧ f x ≤ Real.sqrt 3) →
     (∀ x ∈ Set.Icc (-Real.pi) Real.pi,
       f x + 1 = 0 → x = -(Real.pi / 6) ∨ x = 5 * Real.pi / 6 ∨ x = - Real.pi / 2 ∨ x = Real.pi / 2) →
     ∑ (sol : Finset ℝ) in
      {x ∈ Set ((λ x, f x + 1 = 0) (Set.Icc (-Real.pi) Real.pi))},
      sol.val = 2 * Real.pi / 3 :=
   sorry
   
end proof_problem_l674_674005


namespace standing_arrangement_count_l674_674425

-- Definitions corresponding to the problem conditions
variable (students : Finset String)
variable (A B C D : String)

-- Condition: A and B must be adjacent
def adjacent (x y : String) : Prop := 
  ∃ (xs ys zs : List String), students.val = xs ++ [x, y] ++ ys ∨ students.val = xs ++ [y, x] ++ ys

-- Condition: C and D must not be adjacent
def not_adjacent (x y : String) : Prop := 
  ¬ (∃ (xs ys zs : List String), students.val = xs ++ [x, y] ++ ys ∨ students.val = xs ++ [y, x] ++ ys)

theorem standing_arrangement_count :
  (students.card = 7) →
  adjacent A B →
  not_adjacent C D →
  count_arrangements students = (A_2^2 * A_4^4 * A_5^2) :=
by
  sorry

end standing_arrangement_count_l674_674425


namespace factory_can_pack_all_apples_l674_674943

-- Definitions based on conditions
def crates := 35
def apples_per_crate := 400
def rotten_percentage := 0.11
def apples_per_box := 30
def available_boxes := 1000

-- Derived calculations
def total_apples : ℕ := crates * apples_per_crate
def rotten_apples : ℕ := (rotten_percentage * total_apples).toNat
def good_apples : ℕ := total_apples - rotten_apples
def needed_boxes : ℕ := (good_apples + apples_per_box - 1) / apples_per_box  -- ceil(good_apples / apples_per_box)

-- Theorem statement
theorem factory_can_pack_all_apples : needed_boxes ≤ available_boxes ∧ needed_boxes = 416 := by
  sorry

end factory_can_pack_all_apples_l674_674943


namespace solve_chord_length_through_focus_l674_674160

noncomputable def chord_length_through_focus (x y : ℝ) : Prop :=
  let ellipse := (x^2 / 2) + y^2 = 1
  let focus_2 := (1, 0)
  let chord_line := y = x - 1
  let chord_A := (0, -1)
  let chord_B := (4 / 3, 1 / 3)
  let length_AB := real.sqrt ((chord_A.1 - chord_B.1)^2 + (chord_A.2 - chord_B.2)^2) = 4 * real.sqrt 2 / 3
  length_AB

theorem solve_chord_length_through_focus :
  ∀ (x y : ℝ), chord_length_through_focus x y :=
by
  intros x y
  sorry

end solve_chord_length_through_focus_l674_674160


namespace sum_of_ages_in_5_years_l674_674719

-- Definitions of conditions
def Will_age_3_years_ago := 4
def Diane_age_now (Will_age_now : ℕ) := 2 * Will_age_now
def Janet_age_now (Diane_age_now : ℕ) := Diane_age_now + 3

-- Theorem to prove the sum of their ages in 5 years
theorem sum_of_ages_in_5_years:
  let Will_age_now := Will_age_3_years_ago + 3,
      Diane_age_now := Diane_age_now Will_age_now,
      Janet_age_now := Janet_age_now Diane_age_now,
      Will_age_in_5_years := Will_age_now + 5,
      Diane_age_in_5_years := Diane_age_now + 5,
      Janet_age_in_5_years := Janet_age_now + 5
  in Will_age_in_5_years + Diane_age_in_5_years + Janet_age_in_5_years = 53 := 
by
  sorry

end sum_of_ages_in_5_years_l674_674719


namespace problem_l674_674776

open Real

-- Definitions based on the conditions
variables {f : ℝ → ℝ}
variable h_nonneg : ∀ x, 0 < x → 0 ≤ f x
variable h_diff : ∀ x, 0 < x → ∃ f' : ℝ, differentiable f x
variable h_ineq : ∀ x, 0 < x → x * deriv f x - f x ≤ 0

-- The theorem to prove
theorem problem 
  (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m < n) : m * f n ≤ n * f m :=
sorry

end problem_l674_674776


namespace no_real_a_l674_674336

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}

theorem no_real_a (a : ℝ) : ¬ ((A a ≠ B) ∧ (A a ∪ B = B) ∧ (∅ ⊂ (A a ∩ B))) :=
by
  intro h
  sorry

end no_real_a_l674_674336


namespace number_of_regular_soda_bottles_l674_674221

-- Define the total number of bottles and the number of diet soda bottles
def total_bottles : ℕ := 30
def diet_soda_bottles : ℕ := 2

-- Define the number of regular soda bottles
def regular_soda_bottles : ℕ := total_bottles - diet_soda_bottles

-- Statement of the main proof problem
theorem number_of_regular_soda_bottles : regular_soda_bottles = 28 := by
  -- Proof goes here
  sorry

end number_of_regular_soda_bottles_l674_674221


namespace find_eccentricity_l674_674363

-- Define the parabola and hyperbola
def parabola (x y : ℝ) := y^2 = 8 * x
def hyperbola (x y a : ℝ) := (x^2 / a^2) - y^2 = 1

-- Define the condition that the curves are tangent to each other at a common tangent line
def tangent_condition (x1 y1 a : ℝ) :=
  ∃ t, (parabola x1 y1 ∧ hyperbola x1 y1 a ∧ 
       y1 = 4 * t ∧ x1 = a^2 / (4 * t))

-- Eccentricity of the hyperbola
noncomputable def eccentricity (a : ℝ) := Real.sqrt (1 + (4 * a - 4) / a^2)

-- Main theorem
theorem find_eccentricity {a : ℝ} (ha : a > 0) :
  (∀ x1 y1, tangent_condition x1 y1 a) →
  eccentricity a = Real.sqrt 5 / 2 :=
by
  intro h
  have tangent_exists : ∃ x y, tangent_condition x y a :=
    sorry -- Prove existence
  obtain ⟨x1, y1, ht⟩ := tangent_exists
  have m : y1 / 4 = (x1 / (a^2 * y1)) :=
    sorry -- Use condition and slopes
  have p : x1 = a^2 / 4 :=
    sorry -- Solve for x1
  have ecc : eccentricity a = Real.sqrt (1 + (4 * a - 4) / a^2) :=
    sorry -- Find eccentricity
  apply ecc
  sorry -- Conclude proof

end find_eccentricity_l674_674363


namespace martin_improved_lap_time_l674_674094

def initial_laps := 15
def initial_time := 45 -- in minutes
def final_laps := 18
def final_time := 42 -- in minutes

noncomputable def initial_lap_time := initial_time / initial_laps
noncomputable def final_lap_time := final_time / final_laps
noncomputable def improvement := initial_lap_time - final_lap_time

theorem martin_improved_lap_time : improvement = 2 / 3 := by 
  sorry

end martin_improved_lap_time_l674_674094


namespace t_shirt_jersey_price_difference_l674_674909

theorem t_shirt_jersey_price_difference :
  ∀ (T J : ℝ), (0.9 * T = 192) → (0.9 * J = 34) → (T - J = 175.55) :=
by
  intros T J hT hJ
  sorry

end t_shirt_jersey_price_difference_l674_674909


namespace ratio_of_abc_l674_674314

theorem ratio_of_abc (a b c : ℝ) (h1 : a ≠ 0) (h2 : 14 * (a^2 + b^2 + c^2) = (a + 2 * b + 3 * c)^2) : a / b = 1 / 2 ∧ a / c = 1 / 3 := 
sorry

end ratio_of_abc_l674_674314


namespace sequence_term_product_l674_674670

theorem sequence_term_product (a : ℕ → ℝ) (h : ∀ n, (∏ i in finset.range(n + 1), a i) = (n + 1) ^ 2) : ∀ n ≥ 2, a n = (n + 1) ^ 2 / (n ^ 2) :=
by
  sorry

end sequence_term_product_l674_674670


namespace graph_shift_sin_l674_674573

theorem graph_shift_sin (x : ℝ) : 
  (∀ x, sin (2 * (x - π/12)) = sin (2 * x - π / 6)) :=
by
  intros x
  rw [mul_sub, mul_div_cancel _ two_ne_zero]
  -- This would typically contain the proof steps, but we leave it as sorry.
  sorry

end graph_shift_sin_l674_674573


namespace intersection_point_l674_674278

theorem intersection_point (a b d x y : ℝ) (h1 : a = b + d) (h2 : a * x + b * y = b + 2 * d) :
    (x, y) = (-1, 1) :=
by
  sorry

end intersection_point_l674_674278


namespace sum_of_six_angles_l674_674371

theorem sum_of_six_angles (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 + a3 + a5 = 180)
                           (h2 : a2 + a4 + a6 = 180) : 
                           a1 + a2 + a3 + a4 + a5 + a6 = 360 := 
by
  -- omitted proof
  sorry

end sum_of_six_angles_l674_674371


namespace squirting_students_l674_674617

theorem squirting_students (n : ℕ) (students : Fin (2 * n + 1) → ℕ) 
  (distinct_distances : ∀ i j : Fin (2 * n + 1), i ≠ j → students i ≠ students j)
  (closest_shooting : ∀ i : Fin (2 * n + 1), ∃ j : Fin (2 * n + 1), i ≠ j ∧ 
    (∀ k : Fin (2 * n + 1), k ≠ i → students i - students j = abs (students i - students k))) :
  (∃ i j : Fin (2 * n + 1), i ≠ j ∧ 
    (∀ k : Fin (2 * n + 1), (k = i ∨ k = j) ↔ students.i - students.j = abs (students.i - students.k))) ∧ 
  (∃ i : Fin (2 * n + 1), ∀ j : Fin (2 * n + 1), j ≠ i → students j ≠ students i) :=
by
  sorry

end squirting_students_l674_674617


namespace correct_statements_count_l674_674618

-- Define the three statements as separate propositions

def statement1 : Prop := 
  (∑ i in finset.range 5, 1/(i + 1)) = 137 / 60

def statement2 (x : ℝ) : Prop :=
  (∑ i in finset.range 3, (↑i + 1) * x^2) = 54 → x = 3 ∨ x = -3

def statement3 (n : ℕ) (xs : ℕ → ℝ) : Prop :=
  (∑ i in finset.range n, xs i + 1/xs i) = 2022 ∧
  (∑ i in finset.range n, xs i - 1/xs i) = 2024 →
  (∑ i in finset.range n, xs i) * (∑ i in finset.range n, 1/xs i) = -2023

-- Prove the number of correct statements is exactly 3
theorem correct_statements_count :
  (statement1 ∧ ∀ x, statement2 x ∧ ∀ n xs, statement3 n xs) :=
by {
  -- proof goes here
  sorry
}

end correct_statements_count_l674_674618


namespace symmetric_line_eq_l674_674008

-- Given lines
def line₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def mirror_line (x y : ℝ) : Prop := y = -x

-- Definition of symmetry about the line y = -x
def symmetric_about (l₁ l₂: ℝ → ℝ → Prop) : Prop :=
∀ x y, l₁ x y ↔ l₂ y (-x)

-- Definition of line l₂ that is symmetric to line₁ about the mirror_line
def line₂ (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem stating that the symmetric line to line₁ about y = -x is line₂
theorem symmetric_line_eq :
  symmetric_about line₁ line₂ :=
sorry

end symmetric_line_eq_l674_674008


namespace sector_area_l674_674832

theorem sector_area (s θ r : ℝ) (hs : s = 4) (hθ : θ = 2) (hr : r = s / θ) : (1/2) * r^2 * θ = 4 := by
  sorry

end sector_area_l674_674832


namespace white_balls_in_bag_l674_674049

theorem white_balls_in_bag :
  ∀ (x : ℕ), 
  15 / (x + 15) = 0.75 → 
  x = 5 := 
by
  intro x
  intro h
  sorry

end white_balls_in_bag_l674_674049


namespace perpendicular_planes_l674_674464

noncomputable theory

open_locale classical

variables (α β : Plane) (l m : Line)
variables (h1 : l ∈ α) (h2 : m ∈ β)
variables (h3 : α ≠ β) (h4 : l ⊥ β)

theorem perpendicular_planes (α β : Plane) (l : Line) (h1 : l ∈ α) (h3 : α ≠ β) (h4 : l ⊥ β) : α ⊥ β :=
sorry

end perpendicular_planes_l674_674464


namespace trig_identity_l674_674897

theorem trig_identity (x y : ℝ) :
  sin x ^ 2 + sin (x + y + π / 4) ^ 2 - 2 * sin x * sin (y + π / 4) * sin (x + y + π / 4) =
  1 - 1 / 2 * sin y ^ 2 :=
by
  sorry

end trig_identity_l674_674897


namespace pyramid_volume_l674_674982

-- Define the conditions as variables and type structure
variables {A B C D S: Point}
variables (areaSAB: ℝ) (areaSBC: ℝ) (areaSCD: ℝ) (areaSDA: ℝ)
variables (dihedral_angle_eq: ∀ (P Q: Point), dihedral_angle S P Q = dihedral_angle S B C)
variables (area_ABCD: ℝ)

-- Formalization of the statement to prove the volume of the pyramid
theorem pyramid_volume 
  (h1 : areaSAB = 9)
  (h2 : areaSBC = 9)
  (h3 : areaSCD = 27)
  (h4 : areaSDA = 27)
  (h5 : dihedral_angle_eq A B)
  (h6 : area_ABCD = 36) : 
  volume_pyramid S A B C D = 54 :=
begin
  sorry -- Proof will go here
end

end pyramid_volume_l674_674982


namespace count_solutions_l674_674818

theorem count_solutions :
  (Set.toFinset {p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ (6 / p.1 + 3 / p.2 = 1)}).card = 6 := 
by
  sorry

end count_solutions_l674_674818


namespace first_tribe_term_is_longer_l674_674842

def years_to_days_first_tribe (years : ℕ) : ℕ := 
  years * 12 * 30

def months_to_days_first_tribe (months : ℕ) : ℕ :=
  months * 30

def total_days_first_tribe (years months days : ℕ) : ℕ :=
  (years_to_days_first_tribe years) + (months_to_days_first_tribe months) + days

def years_to_days_second_tribe (years : ℕ) : ℕ := 
  years * 13 * 4 * 7

def moons_to_days_second_tribe (moons : ℕ) : ℕ :=
  moons * 4 * 7

def weeks_to_days_second_tribe (weeks : ℕ) : ℕ :=
  weeks * 7

def total_days_second_tribe (years moons weeks days : ℕ) : ℕ :=
  (years_to_days_second_tribe years) + (moons_to_days_second_tribe moons) + (weeks_to_days_second_tribe weeks) + days

theorem first_tribe_term_is_longer :
  total_days_first_tribe 7 1 18 > total_days_second_tribe 6 12 1 3 :=
by
  sorry

end first_tribe_term_is_longer_l674_674842


namespace B_alone_work_time_l674_674225

-- Define the predicates for the problem
variable (A_rate B_rate : ℕ) (work_total : ℕ)

-- Conditions provided in the problem
-- Condition 1: A is twice as good a workman as B
def good_workman_condition : Prop := A_rate = 2 * B_rate

-- Condition 2: A and B took 9 days together to do the work
def together_work_time_condition (days : ℕ) : Prop := 
  (A_rate + B_rate) * days = work_total

-- Goal: Prove that B alone can complete the work in 27 days
theorem B_alone_work_time (days : ℕ)
  (h1 : good_workman_condition A_rate B_rate)
  (h2 : together_work_time_condition A_rate B_rate 9)
  (h3 : work_total = 1) :
  B_rate * 27 = work_total :=
by
  sorry

end B_alone_work_time_l674_674225


namespace students_suggested_pasta_l674_674899

-- Define the conditions as variables in Lean
variable (total_students : ℕ := 470)
variable (suggested_mashed_potatoes : ℕ := 230)
variable (suggested_bacon : ℕ := 140)

-- The problem statement to prove
theorem students_suggested_pasta : 
  total_students - (suggested_mashed_potatoes + suggested_bacon) = 100 := by
  sorry

end students_suggested_pasta_l674_674899


namespace value_two_std_dev_less_l674_674911

theorem value_two_std_dev_less (mean : ℝ) (std_dev : ℝ) (h_mean : mean = 15) (h_std_dev : std_dev = 1.5) :
  mean - 2 * std_dev = 12 :=
by
  rw [h_mean, h_std_dev]
  norm_num
  sorry

end value_two_std_dev_less_l674_674911


namespace seq_a_4_eq_7_over_4_l674_674803

noncomputable def seq_a : ℕ → ℚ
| 1       := 1
| (n + 1) := seq_a n + 1 / (n * (n + 1))

theorem seq_a_4_eq_7_over_4 : seq_a 4 = 7 / 4 :=
sorry

end seq_a_4_eq_7_over_4_l674_674803


namespace range_of_m_l674_674006

def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m
  (m : ℝ)
  (hθ : ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2)
  (h : ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → f (m * Real.sin θ) + f (1 - m) > 0) :
  m < 1 :=
by
  sorry

end range_of_m_l674_674006


namespace EF_length_in_circle_inscribed_quadrilateral_l674_674109

theorem EF_length_in_circle_inscribed_quadrilateral :
  ∀ (E F G H : ℝ) (circle_inscribed : Bool)
    (angle_EFG angle_EHG : ℝ) (EH FG : ℝ),
    circle_inscribed = true →
    angle_EFG = 60 →
    angle_EHG = 50 →
    EH = 3 →
    FG = 5 →
    let EF := (5 * Real.sin (70 * Real.pi / 180)) / (Real.sin (60 * Real.pi / 180))
  in EF = (5 * Real.sin (70 * Real.pi / 180)) / (Real.sin (60 * Real.pi / 180)) :=
begin
  intros,
  let EF := (5 * Real.sin (70 * Real.pi / 180)) / (Real.sin (60 * Real.pi / 180)),
  exact sorry
end

end EF_length_in_circle_inscribed_quadrilateral_l674_674109


namespace farmer_total_cows_l674_674634

theorem farmer_total_cows :
  let n : ℕ := 24 in
  (1 / 3 + 1 / 6 + 1 / 8 : ℚ) = 5 / 8 ∧
  (1 - (5 / 8) : ℚ) = 3 / 8 ∧
  (9 : ℚ) / (3 / 8) = n
  → n = 24 := by
  sorry

end farmer_total_cows_l674_674634


namespace average_student_headcount_is_correct_l674_674262

noncomputable def average_student_headcount : ℕ :=
  let a := 11000
  let b := 10200
  let c := 10800
  let d := 11300
  (a + b + c + d) / 4

theorem average_student_headcount_is_correct :
  average_student_headcount = 10825 :=
by
  -- Proof will go here
  sorry

end average_student_headcount_is_correct_l674_674262


namespace inverse_contrapositive_l674_674136

theorem inverse_contrapositive (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : a^2 + b^2 ≠ 0 :=
sorry

end inverse_contrapositive_l674_674136


namespace number_of_mappings_l674_674151

theorem number_of_mappings {A : Type} (h : fintype.card A = 2) : fintype.card (A → A) = 4 :=
sorry

end number_of_mappings_l674_674151


namespace double_rooms_percentage_l674_674254

theorem double_rooms_percentage (S : ℝ) (h1 : 0 < S)
  (h2 : ∃ Sd : ℝ, Sd = 0.75 * S)
  (h3 : ∃ Ss : ℝ, Ss = 0.25 * S):
  (0.375 * S) / (0.625 * S) * 100 = 60 := 
by 
  sorry

end double_rooms_percentage_l674_674254


namespace tan_x_proof_l674_674316

noncomputable def tan_problem (x : ℝ) : Prop :=
  x ∈ Ioo (-π/2) 0 ∧ cos x = 4/5 → tan x = -3/4

-- Proof omitted
theorem tan_x_proof (x : ℝ) : tan_problem x :=
sorry

end tan_x_proof_l674_674316


namespace DT_passes_through_circumcenter_l674_674857

open EuclideanGeometry

noncomputable def triangle_ABC (a b c: Point) : Prop :=
  is_acute_triangle a b c

noncomputable def orthocenter_H (a b c: Point) (h: Point) : Prop :=
  is_orthocenter a b c h

noncomputable def midpoint (p1 p2 m: Point) : Prop :=
  is_midpoint p1 p2 m

noncomputable def concyclic (p1 p2 p3 p4: Point) : Prop :=
  is_cyclic_quadrilateral p1 p2 p3 p4

theorem DT_passes_through_circumcenter
  (A B C H E F D M N T O: Point)
  (h_triangle_ABC : triangle_ABC A B C)
  (h_orthocenter_H: orthocenter_H A B C H)
  (hE: intersection_point (line B H) (line A C) E)
  (hF: intersection_point (line C H) (line A B) F)
  (hD: midpoint A H D)
  (hM: midpoint B D M)
  (hN: midpoint C D N)
  (hT: intersection_point (line F M) (line E N) T)
  (h_concyclic: concyclic D E T F)
  (h_circumcenter_O: is_circumcenter A B C O) :
  passes_through (line D T) O :=
sorry

end DT_passes_through_circumcenter_l674_674857


namespace club_member_enemies_friendly_pairs_l674_674426

theorem club_member_enemies_friendly_pairs (n q : ℕ) 
  (h1 : ∀ (x y z : Fin n), x ≠ y ∧ y ≠ z ∧ z ≠ x → 
    (x ≠ y ∧ y ≠ z ∧ z ≠ x → ∃ (p : Fin n), p = x ∨ p = y ∨ p = z ∧ p ≠ y ∧ p ≠ z)) :
  ∃ (i : Fin n), (∃ (enemy : Fin n), set_Of_enemy (i) ≤ q * (1 - 4 * q / (n * n))) :=
by
  sorry

end club_member_enemies_friendly_pairs_l674_674426


namespace volume_of_given_tetrahedron_l674_674783

noncomputable def volume_of_tetrahedron (radius : ℝ) (total_length : ℝ) : ℝ := 
  let R := radius
  let L := total_length
  let a := (2 * Real.sqrt 33) / 3
  let V := (a^3 * Real.sqrt 2) / 12
  V

theorem volume_of_given_tetrahedron :
  volume_of_tetrahedron (Real.sqrt 22 / 2) (8 * Real.pi) = 48 := 
  sorry

end volume_of_given_tetrahedron_l674_674783


namespace problem1_problem2_l674_674796

-- Definition of the function f
def f (a b x : ℝ) : ℝ := (a * x) / (x + b)

-- Conditions for the problem
def conditions (a b : ℝ) : Prop := 
  f a b 1 = 1 ∧ f a b (-2) = 4

-- Problem 1: Values of a and b, existence of constant c
theorem problem1 (a b : ℝ) (h : conditions a b) : a = 2 ∧ b = 1 ∧ ∃ c, ∀ x, f a b x + f a b (c-x) = 4 :=
  sorry

-- Problem 2: Range of m ensuring inequality holds
theorem problem2 (m : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 2 → f 2 1 x ≤ (2 * m) / ((x + 1) * abs (x - m))) ↔ (2 < m ∧ m ≤ 4) :=
  sorry

end problem1_problem2_l674_674796


namespace problem_solution_l674_674826

theorem problem_solution
  (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end problem_solution_l674_674826


namespace hyperbola_equation_l674_674319

theorem hyperbola_equation
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (c : ℝ) (hc : c = 6)
  (h_asymptote : (√3) * a = b) :
  (c^2 = a^2 + b^2) -> (1 / a^2 = 1 / 9) ∧ (1 / b^2 = 1 / 27) :=
by
  sorry

end hyperbola_equation_l674_674319


namespace dodecagon_area_l674_674117

theorem dodecagon_area (a : ℝ) (h : a > 0) :
  let hexagon_area := (3 * a^2 * Real.sqrt 3) / 2 in
  let square_area := 6 * a^2 in
  let dodecagon_area := 3 * a^2 * (Real.sqrt 3 + 2) in 
  let total_area := hexagon_area + square_area in
  total_area = dodecagon_area :=
begin
  let hexagon_area := (3 * a^2 * Real.sqrt 3) / 2,
  let square_area := 6 * a^2,
  let dodecagon_area := 3 * a^2 * (Real.sqrt 3 + 2),
  let total_area := hexagon_area + square_area,
  exact (by linarith),
end

end dodecagon_area_l674_674117


namespace student_selection_scheme_l674_674311

theorem student_selection_scheme (students : Finset ℕ) (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) (P : ℕ) (H_cardinal : students.card = 5)
  (H_subjects : {A, B, C, D}.to_finset ⊆ students) (H_no_biology : P ∉ {A, B, C, D}.to_finset) : 
  96 = (comb 3 1 * fact 3 + fact 4) :=
by {
  -- Proof goes here
  sorry
}

end student_selection_scheme_l674_674311


namespace royWeight_l674_674072

-- Define the problem conditions
def johnWeight : ℕ := 81
def johnHeavierBy : ℕ := 77

-- Define the main proof problem
theorem royWeight : (johnWeight - johnHeavierBy) = 4 := by
  sorry

end royWeight_l674_674072


namespace compute_N_l674_674080

noncomputable theory

open Matrix

theorem compute_N:
  ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  (N ⬝ (col_vec (3:: -2:: [])) = col_vec (4:: 1:: [])) ∧
  (N ⬝ (col_vec (-2:: 4:: [])) = col_vec (0:: 2:: [])) →
  (N ⬝ (col_vec (7:: 0:: [])) = col_vec (14:: 7:: [])) :=
by
  sorry

end compute_N_l674_674080


namespace juice_problem_l674_674989

theorem juice_problem 
  (p_a p_y : ℚ)
  (v_a v_y : ℚ)
  (p_total v_total : ℚ)
  (ratio_a : p_a / v_a = 4)
  (ratio_y : p_y / v_y = 1 / 5)
  (p_a_val : p_a = 20)
  (p_total_val : p_total = 24)
  (v_total_eq : v_total = v_a + v_y)
  (p_y_def : p_y = p_total - p_a) :
  v_total = 25 :=
by
  sorry

end juice_problem_l674_674989


namespace find_c_l674_674399

theorem find_c (a b c : ℝ) (h : 1/a + 1/b = 1/c) : c = (a * b) / (a + b) := 
by
  sorry

end find_c_l674_674399


namespace equation_of_ellipse_l674_674763

-- Define the conditions and required properties
variables {a b c : ℝ}

def isEllipseCenteredAtOrigin (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ c = 1

def eccentricity (c a : ℝ) : Prop :=
  c / a = 1 / 2

def parabolaFocus : Prop :=
  (1 = 1 : ℝ)

-- The main theorem to prove the equation of the ellipse
theorem equation_of_ellipse (h1 : isEllipseCenteredAtOrigin a b)
                            (h2 : eccentricity c a)
                            (h3 : parabolaFocus) :
  a = 2 ∧ b^2 = a^2 - c^2 → 
  (∀ x y : ℝ, (x^2 / (2 : ℝ)^2) + (y^2 / (3 : ℝ)) = 1) :=
by
  sorry

end equation_of_ellipse_l674_674763


namespace telescoping_product_l674_674700

theorem telescoping_product :
  (∏ x in {3, 4, 5, 6, 7}, (x^3 - 1) / (x^3 + 1)) = 57 / 168 := by
  sorry

end telescoping_product_l674_674700


namespace math_problem_l674_674687

theorem math_problem :
  ( ∏ i in [3, 4, 5, 6, 7], (i^3 - 1) / (i^3 + 1) ) = 57 / 84 := sorry

end math_problem_l674_674687


namespace problem_proof_l674_674921

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the function f

def g (x : ℝ) : ℝ := f x + f (-x)

axiom condition1 (x : ℝ) (hx : 0 < x) : deriv g x ≤ 0
axiom condition2 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : f a - f b > f (-b) - f (-a)

theorem problem_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 < b^2 :=
sorry

end problem_proof_l674_674921


namespace math_problem_l674_674688

theorem math_problem :
  ( ∏ i in [3, 4, 5, 6, 7], (i^3 - 1) / (i^3 + 1) ) = 57 / 84 := sorry

end math_problem_l674_674688


namespace quadratic_inequality_solution_l674_674924

theorem quadratic_inequality_solution
  (a : ℝ) :
  (∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 ≤ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end quadratic_inequality_solution_l674_674924


namespace viable_combinations_l674_674656

-- Given conditions
def totalHerbs : Nat := 4
def totalCrystals : Nat := 6
def incompatibleComb1 : Nat := 2
def incompatibleComb2 : Nat := 1

-- Theorem statement proving the number of viable combinations
theorem viable_combinations : totalHerbs * totalCrystals - (incompatibleComb1 + incompatibleComb2) = 21 := by
  sorry

end viable_combinations_l674_674656


namespace vector_collinearity_implies_lambda_l674_674810

variable (a b c : Vector ℝ)
variable (λ : ℝ)

def given_vectors : Prop := a = (1, 2) ∧ b = (2, 0) ∧ c = (1, -2)

def collinearity (u v : Vector ℝ) : Prop :=
  ∃ k : ℝ, u = k • v

theorem vector_collinearity_implies_lambda (h : given_vectors a b c) :
  collinearity (λ • a + b) c → λ = -1 :=
begin
  sorry
end

end vector_collinearity_implies_lambda_l674_674810


namespace smallest_value_of_f4_l674_674859

def f (x : ℝ) : ℝ := (x + 3) ^ 2 - 2

theorem smallest_value_of_f4 : ∀ x : ℝ, f (f (f (f x))) ≥ 23 :=
by 
  sorry -- Proof goes here.

end smallest_value_of_f4_l674_674859


namespace convex_quad_area_inscribed_quad_area_circumscribed_quad_area_l674_674601

noncomputable def semiperimeter (a b c d : ℝ) : ℝ :=
  (a + b + c + d) / 2

def area_formula (a b c d p : ℝ) (B D : ℝ) : ℝ :=
  (p - a) * (p - b) * (p - c) * (p - d) - a * b * c * d * Real.cos (B + D / 2) ^ 2

def area_inscribed_formula (a b c d p : ℝ) : ℝ :=
  (p - a) * (p - b) * (p - c) * (p - d)

def area_circumscribed_formula (a b c d B D : ℝ) : ℝ :=
  a * b * c * d * Real.sin (B + D / 2) ^ 2

theorem convex_quad_area (a b c d : ℝ) (p : ℝ) (B D : ℝ) :
  p = semiperimeter a b c d →
  S^2 = area_formula a b c d p B D := sorry

theorem inscribed_quad_area (a b c d : ℝ) (p : ℝ) (B D : ℝ) :
  p = semiperimeter a b c d →
  S^2 = area_inscribed_formula a b c d p := sorry

theorem circumscribed_quad_area (a b c d : ℝ) (p : ℝ) (B D : ℝ) :
  p = semiperimeter a b c d →
  S^2 = area_circumscribed_formula a b c d B D := sorry

end convex_quad_area_inscribed_quad_area_circumscribed_quad_area_l674_674601


namespace sqrt_inequality_solution_l674_674521

variable (x : ℝ)

theorem sqrt_inequality_solution (h : (∃ y : ℝ, y = real.sqrt x ∧ (y + 3 / (y - 2) ≤ 0))) : x < 16 :=
begin
  sorry
end

end sqrt_inequality_solution_l674_674521


namespace solution_to_inequality_system_l674_674522

theorem solution_to_inequality_system :
  (∀ x : ℝ, 2 * (x - 1) < x + 2 → (x + 1) / 2 < x → 1 < x ∧ x < 4) :=
by
  intros x h1 h2
  sorry

end solution_to_inequality_system_l674_674522


namespace find_ellipse_equation_find_range_l674_674331

section ellipse_problem

variables {a b : ℝ} (a_gt_b : a > b) (b_pos : b > 0)

-- Condition: Ellipse equation and eccentricity
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (e : ℝ) : Prop := e = real.sqrt 2 / 2

-- Given point P(x_0, y_0) and its symmetric point P_1(x_1, y_1) with respect to y = 2x
variables (x0 y0 x1 y1 : ℝ)

-- Define foci sum condition
def foci_sum_condition : Prop := 2 * a = 4

-- Define symmetry with respect to y = 2x
def symmetry_y_2x : Prop :=
  (y0 - y1) / (x0 - x1) * 2 = -1 ∧
  (y0 + y1) / 2 = 2 * (x0 + x1) / 2

-- Equation of ellipse C to prove
theorem find_ellipse_equation
  (h1 : eccentricity (real.sqrt 2 / 2))
  (h2 : foci_sum_condition)
  (h3 : a = 2)
  (h4 : b = real.sqrt (a^2 - (real.sqrt 2)^2)) :
  ellipse 2 2 := sorry

-- Range for 3x1 - 4y1
theorem find_range
  (P_on_ellipse : ellipse x0 y0)
  (symmetry : symmetry_y_2x)
  (h5 : x1 = (4 * y0 - 3 * x0) / 5)
  (h6 : y1 = (3 * y0 + 4 * x0) / 5) :
  -10 ≤ 3 * x1 - 4 * y1 ∧ 3 * x1 - 4 * y1 ≤ 10 := sorry

end ellipse_problem

end find_ellipse_equation_find_range_l674_674331


namespace exists_disjoint_translations_l674_674462

variable (S : Set ℕ) (A : Set ℕ) (t : Fin 100 → ℕ)

noncomputable def problem_statement : Prop :=
  ∀ j k : Fin 100, j ≠ k → (A.map (· + t j) ∩ A.map (· + t k)).IsEmpty

axiom A_condition : A ⊆ S ∧ A.card = 101

theorem exists_disjoint_translations (A : Set ℕ) (S : Set ℕ)
  (hS : S = {1, 2, ..., 1000000}) (hA : A ⊆ S ∧ A.card = 101) :
  ∃ t : Fin 100 → ℕ, problem_statement S A t :=
sorry

end exists_disjoint_translations_l674_674462


namespace arithmetic_sequence_general_formula_l674_674862

def arithmeticSequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
13 - 2 * (n - 1)

def sumArithmeticSequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_general_formula (a₂ : ℤ) (S₁₀ : ℤ) :
  a₂ = 11 → S₁₀ = 40 → 
  (∀ n : ℕ, n > 0 → arithmeticSequence n 13 (-2) = -2 * n + 15) ∧
  (∀ n : ℕ, n > 0 →
    if n ≤ 7 then 
      sumArithmeticSequence n 13 (-2) = -n^2 + 14 * n 
    else 
      let basic_sum := sumArithmeticSequence n 13 (-2)
      let fixed_term_sum := 2 * (13 + 1) * 7 / 2
      fixed_term_sum - basic_sum = n^2 - 14 * n + 98)
:=
begin
  intros h₁ h₂,
  split,
  { intros n hn,
    sorry
  },
  { intros n hn,
    sorry
  }

end arithmetic_sequence_general_formula_l674_674862


namespace probability_red_tile_l674_674205

theorem probability_red_tile :
  let n := 80
  let congruent_count := finite { x ∈ finset.range n | x % 7 = 3 }.card
  let total_count := n
  let prob := congruent_count / total_count
  prob = 3 / 20 :=
by
  -- Sorry placeholder to skip direct proof.
  sorry

end probability_red_tile_l674_674205


namespace even_integers_count_l674_674021

theorem even_integers_count (n : ℤ) (m : ℤ) (total_even : ℤ) 
  (h1 : m = 45) (h2 : total_even = 10) (h3 : m % 2 = 1) :
  (∃ k : ℤ, ∀ x : ℤ, 0 ≤ x ∧ x < total_even → k = n + 2 * x) ∧ (n = 26) :=
by
  sorry

end even_integers_count_l674_674021


namespace second_car_speed_l674_674956

theorem second_car_speed
  (d_AB : ℕ)
  (d_total : ℕ)
  (s1 : ℕ)
  (t : ℕ)
  (remaining_distance : ℕ)
  (h1 : d_AB = 360)
  (h2 : s1 = 50)
  (h3 : t = 3)
  (h4 : remaining_distance = 48)
  (h_total_distance : d_total = d_AB - remaining_distance)
  (h_total_speed : d_total / t = 104) :
  (h_second_car_speed : 104 - s1 = 54) :=
begin
  rw [h1, h2, h3, h4] at *,
  have : d_total = 360 - 48, by rw [h4],
  rw [h_total_distance] at this,
  have : d_total = 312, by norm_num at the goal,
  rw [this] at h_total_speed,
  have : 312 / 3 = 104, by norm_num at the goal,
  rw [this] at h_total_speed,
  have : 104 - 50 = 54, by norm_num at the goal,
  exact this,
end

end second_car_speed_l674_674956


namespace equilateral_triangle_side_length_l674_674666

variable (R : ℝ)

theorem equilateral_triangle_side_length (R : ℝ) :
  (∃ (s : ℝ), s = R * Real.sqrt 3) :=
sorry

end equilateral_triangle_side_length_l674_674666


namespace no_correlation_pair_D_l674_674662

-- Define the pairs of variables and their relationships
def pair_A : Prop := ∃ (fertilizer_applied grain_yield : ℝ), (fertilizer_applied ≠ 0 → grain_yield ≠ 0)
def pair_B : Prop := ∃ (review_time scores : ℝ), (review_time ≠ 0 → scores ≠ 0)
def pair_C : Prop := ∃ (advertising_expenses sales : ℝ), (advertising_expenses ≠ 0 → sales ≠ 0)
def pair_D : Prop := ∃ (books_sold revenue : ℕ), (revenue = books_sold * 5)

/-- Prove that pair D does not have a correlation in the context of the problem. --/
theorem no_correlation_pair_D : ¬pair_D :=
by
  sorry

end no_correlation_pair_D_l674_674662


namespace monotonic_increasing_iff_l674_674421

open Real

noncomputable def f (x a : ℝ) : ℝ := abs ((exp x) / 2 - a / (exp x))

theorem monotonic_increasing_iff (a : ℝ) : 
  (∀ x ∈ set.Icc 1 2, ∀ y ∈ set.Icc 1 2, x ≤ y → f x a ≤ f y a) ↔ (- (exp 2)^2 / 2 ≤ a ∧ a ≤ (exp 2) / 2) := 
by 
  sorry

end monotonic_increasing_iff_l674_674421


namespace AM_perpendicular_BC_l674_674447

variables (A B C D E F G M : Type) [has_coe_line A B] [has_coe_line A C] [has_coe_line B C]
variables (h_triangle : triangle ABC)
variables (h_diameter : is_diameter B C)
variables (h_semicircle : semicircle B C intersects A B D ∧ semicircle B C intersects A C E)
variables (h_perpendicular_D : perpendicular D F to B C)
variables (h_perpendicular_E : perpendicular E G to B C)
variables (h_intersection : DE intersects EF at M)

theorem AM_perpendicular_BC : perpendicular A M to B C :=
sorry

end AM_perpendicular_BC_l674_674447


namespace sequence_sum_S2018_l674_674760

theorem sequence_sum_S2018 :
  ∀ (a : ℕ → ℕ), (S : ℕ → ℕ),
  (a 1 = 1) →
  (∀ n, n ≥ 2 → a n + 2 * S (n - 1) = n) →
  S 2018 = 1009 :=
by
  intros a S ha1 ha_prop
  sorry

end sequence_sum_S2018_l674_674760


namespace last_child_loses_l674_674612

-- Definitions corresponding to conditions
def num_children := 11
def child_sequence := List.range' 1 num_children
def valid_two_digit_numbers := 90
def invalid_digit_sum_6 := 6
def invalid_digit_sum_9 := 9
def valid_numbers := valid_two_digit_numbers - invalid_digit_sum_6 - invalid_digit_sum_9
def complete_cycles := valid_numbers / num_children
def remaining_numbers := valid_numbers % num_children

-- Statement to be proven
theorem last_child_loses (h1 : num_children = 11)
                         (h2 : valid_two_digit_numbers = 90)
                         (h3 : invalid_digit_sum_6 = 6)
                         (h4 : invalid_digit_sum_9 = 9)
                         (h5 : valid_numbers = valid_two_digit_numbers - invalid_digit_sum_6 - invalid_digit_sum_9)
                         (h6 : remaining_numbers = valid_numbers % num_children) :
  (remaining_numbers = 9) ∧ (num_children - remaining_numbers = 2) :=
by
  sorry

end last_child_loses_l674_674612


namespace Harold_spending_l674_674814

theorem Harold_spending
  (num_shirt_boxes : ℕ)
  (num_xl_boxes : ℕ)
  (wraps_shirt_boxes : ℕ)
  (wraps_xl_boxes : ℕ)
  (cost_per_roll : ℕ)
  (h1 : num_shirt_boxes = 20)
  (h2 : num_xl_boxes = 12)
  (h3 : wraps_shirt_boxes = 5)
  (h4 : wraps_xl_boxes = 3)
  (h5 : cost_per_roll = 4) :
  num_shirt_boxes / wraps_shirt_boxes + num_xl_boxes / wraps_xl_boxes * cost_per_roll = 32 :=
by
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end Harold_spending_l674_674814


namespace profit_percentage_correct_l674_674992

def cost_price : ℝ := 32
def selling_price : ℝ := 56
def trading_tax_rate : ℝ := 0.07

def trading_tax : ℝ := trading_tax_rate * selling_price
def actual_selling_price_after_tax : ℝ := selling_price - trading_tax
def profit : ℝ := actual_selling_price_after_tax - cost_price
def profit_percentage : ℝ := (profit / cost_price) * 100

theorem profit_percentage_correct :
  profit_percentage = 62.75 := by
  sorry

end profit_percentage_correct_l674_674992


namespace no_real_roots_of_quadratic_l674_674835

theorem no_real_roots_of_quadratic (m : ℝ) (h1 : ∀ x : ℤ, 3 ≤ x ∧ x < m → x ∈ {3, 4, 5, 6}) (h2 : 6 < m ∧ m ≤ 7) :
  ∀ x : ℝ, 8 * x^2 - 8 * x + m = 0 → false :=
begin
  intros x hx,
  have h : 64 - 32 * m < 0,
  { linarith, },
  apply (ne_of_lt h).symm,
  rw [← sub_nonpos, ← real.sqrt_sq (8 * x^2 - 8 * x + m)],
  simp,
  sorry
end

end no_real_roots_of_quadratic_l674_674835


namespace responses_needed_l674_674032

noncomputable def Q : ℝ := 461.54
noncomputable def percentage : ℝ := 0.65
noncomputable def required_responses : ℝ := percentage * Q

theorem responses_needed : required_responses = 300 := by
  sorry

end responses_needed_l674_674032


namespace team_matches_per_season_l674_674039

theorem team_matches_per_season (teams_count total_games : ℕ) (h1 : teams_count = 50) (h2 : total_games = 4900) : 
  ∃ n : ℕ, n * (teams_count - 1) * teams_count / 2 = total_games ∧ n = 2 :=
by
  sorry

end team_matches_per_season_l674_674039


namespace quadrilateral_areas_product_l674_674214

noncomputable def areas_product_property (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) : Prop :=
  (S_ADP * S_BCP * S_ABP * S_CDP) % 10000 ≠ 1988
  
theorem quadrilateral_areas_product (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) :
  areas_product_property S_ADP S_ABP S_CDP S_BCP h1 :=
by
  sorry

end quadrilateral_areas_product_l674_674214


namespace problem_statement_l674_674305

noncomputable def lcm_upto (n : ℕ) : ℕ :=
  Nat.lcm_list (List.range (n + 1))

noncomputable def hat_l (n : ℕ) : ℕ :=
  n * Nat.choose (2 * n) n

theorem problem_statement
  (n : ℕ) (h_pos : 1 ≤ n) :
  (hat_l n | lcm_upto (2 * n)) ∧ (4 ≤ n → hat_l n ≥ 2 ^ (2 * n)) ∧ (7 ≤ n → lcm_upto n ≥ 2 ^ n) :=
begin
  split,
  { -- Proof of hat_l(n) | lcm_upto(2n)
    sorry },
  split,
  { -- Proof of hat_l(n) ≥ 2^(2n) for all n ≥ 4
    intro h_ge_4,
    sorry },
  { -- Proof of lcm_upto(n) ≥ 2^n for all n ≥ 7
    intro h_ge_7,
    sorry }
end

end problem_statement_l674_674305


namespace distance_from_circle_center_to_hyperbola_center_l674_674629

-- Define the hyperbola equation as given in conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1

-- Define the center of the circle, note that (4, y) lies on the hyperbola
def circle_center_on_hyperbola (y : ℝ) : Prop := hyperbola 4 y

-- Define the coordinates of the hyperbola center
def hyperbola_center : ℝ × ℝ := (0, 0)

-- Define the distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the problem, proving the distance from the center of the circle to the hyperbola center
theorem distance_from_circle_center_to_hyperbola_center :
  ∃ y : ℝ, circle_center_on_hyperbola y ∧ distance (4, y) hyperbola_center = 16 / 3 :=
by
  sorry

end distance_from_circle_center_to_hyperbola_center_l674_674629


namespace sequence_parity_l674_674804

def sequence (n : ℕ) : ℤ := Int.floor ((3 + Real.sqrt 17) / 2) ^ n

theorem sequence_parity (n : ℕ) :
  (n % 2 = 1 ∧ sequence n % 2 = 1) ∨ (n % 2 = 0 ∧ sequence n % 2 = 0) := 
sorry

end sequence_parity_l674_674804


namespace arithmetic_sequence_general_formula_l674_674351

def f (x: ℝ) : ℝ := x^2 - 2*x + 4

theorem arithmetic_sequence_general_formula (d : ℝ) (n : ℕ) : 
  let a1 := f (d - 1)
  let a3 := f (d + 1)
  let a := λ n, a1 + (n - 1) * (a3 - a1) / 2
  (a n) = 2*n + 1 :=
by
  sorry

end arithmetic_sequence_general_formula_l674_674351


namespace cricket_players_count_l674_674841

theorem cricket_players_count (hockey: ℕ) (football: ℕ) (softball: ℕ) (total: ℕ) : 
  hockey = 15 ∧ football = 21 ∧ softball = 19 ∧ total = 77 → ∃ cricket, cricket = 22 := by
  sorry

end cricket_players_count_l674_674841


namespace find_C_plus_D_l674_674001

theorem find_C_plus_D (C D : ℝ) (h : ∀ x : ℝ, (Cx - 20) / (x^2 - 3 * x - 10) = D / (x + 2) + 4 / (x - 5)) :
  C + D = 4.7 :=
sorry

end find_C_plus_D_l674_674001


namespace probability_non_adjacent_zeros_l674_674386

theorem probability_non_adjacent_zeros (total_ones total_zeros : ℕ) (h₁ : total_ones = 3) (h₂ : total_zeros = 2) : 
  (total_zeros != 0 ∧ total_ones != 0 ∧ total_zeros + total_ones = 5) → 
  (prob_non_adjacent (total_ones + total_zeros) total_zeros = 0.6) :=
by
  sorry

def prob_non_adjacent (total num_zeros: ℕ) : ℚ :=
  let total_arrangements := (Nat.factorial total) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros)))
  let adjacent_arrangements := (Nat.factorial (total - num_zeros + 1)) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros - 1)))
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements / total_arrangements

end probability_non_adjacent_zeros_l674_674386


namespace correct_translation_of_tradition_l674_674574

def is_adjective (s : String) : Prop :=
  s = "传统的"

def is_correct_translation (s : String) (translation : String) : Prop :=
  s = "传统的" → translation = "traditional"

theorem correct_translation_of_tradition : 
  is_adjective "传统的" ∧ is_correct_translation "传统的" "traditional" :=
by
  sorry

end correct_translation_of_tradition_l674_674574


namespace arithmetic_mean_increase_by_50_l674_674831

theorem arithmetic_mean_increase_by_50 (b : Fin 15 → ℝ) :
  let T := ∑ i, b i,
      original_mean := T / 15,
      new_set := (fun i => b i + 50)
  in 
  let new_sum := ∑ i, new_set i,
      new_mean := new_sum / 15
  in 
  new_mean = original_mean + 50 :=
by
  sorry

end arithmetic_mean_increase_by_50_l674_674831


namespace composite_ac_plus_bd_l674_674873

theorem composite_ac_plus_bd
    (a b c d e : ℕ) 
    (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
    (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e)
    (h8 : c ≠ d) (h9 : c ≠ e)
    (h10 : d ≠ e)
    (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
    (h : a^4 + b^4 = c^4 + d^4 ∧ c^4 + d^4 = e^5) :
    ∃ k, k > 1 ∧ k < ac + bd ∧ (ac + bd) % k = 0 := 
begin
    sorry
end

end composite_ac_plus_bd_l674_674873


namespace smaller_octagon_area_fraction_l674_674139

noncomputable theory

-- Define a regular octagon and its properties in Lean
def regular_octagon (A B C D E F G H : Point) : Prop := 
-- Properties of a regular octagon (e.g., equal side lengths and angles) to be defined here

-- Define the property of the smaller octagon formed by midpoints
def smaller_octagon (A B C D E F G H M1 M2 M3 M4 M5 M6 M7 M8 : Point) : Prop :=
-- Properties of a smaller octagon formed by midpoints to be defined here

theorem smaller_octagon_area_fraction (A B C D E F G H M1 M2 M3 M4 M5 M6 M7 M8 : Point)
  (hreg_octagon : regular_octagon A B C D E F G H)
  (hsmall_octagon : smaller_octagon A B C D E F G H M1 M2 M3 M4 M5 M6 M7 M8) :
  (area (smaller_octagon A B C D E F G H M1 M2 M3 M4 M5 M6 M7 M8)) = 
  (1 / 2) * (area (regular_octagon A B C D E F G H)) :=
sorry

end smaller_octagon_area_fraction_l674_674139


namespace polynomial_degree_is_seven_l674_674263

noncomputable def E1 := (λ x : ℝ, x^5)
noncomputable def E2 := (λ x : ℝ, x^2 + (1 / x^2))
noncomputable def E3 := (λ x : ℝ, 1 + (2 / x^3) + (3 / x^2))

theorem polynomial_degree_is_seven (x : ℝ) :
  polynomial.degree ((E1 x) * (E2 x) * (E3 x)) = 7 :=
sorry

end polynomial_degree_is_seven_l674_674263


namespace min_distance_between_curves_l674_674757

theorem min_distance_between_curves :
  let P := (x : ℝ) × ℝ := (x, (1 / 2) * Real.exp x)
  let Q := (x : ℝ) × ℝ := (x, Real.log (2 * x))
  ∃ x₀ : ℝ, ∃ dist_min : ℝ, 
  dist_min = Real.sqrt 2 * (1 - Real.log 2) ∧
  dist_min = Real.sqrt 2 * Real.abs ((((1 : ℝ) / 2) * Real.exp x₀ - x₀) / Real.sqrt 2) :=
sorry

end min_distance_between_curves_l674_674757


namespace find_quotient_l674_674542

theorem find_quotient :
  ∃ quotient : ℕ,
  let divisor := 21,
      remainder := 7,
      dividend := 301 in
  (dividend = (divisor * quotient) + remainder) ∧ quotient = 14 :=
by
  sorry

end find_quotient_l674_674542


namespace cos_probability_ge_one_half_in_range_l674_674443

theorem cos_probability_ge_one_half_in_range :
  let interval_length := (Real.pi / 2) - (- (Real.pi / 2))
  let favorable_length := (Real.pi / 3) - (- (Real.pi / 3))
  (favorable_length / interval_length) = (2 / 3) := by
  sorry

end cos_probability_ge_one_half_in_range_l674_674443


namespace positive_diff_of_squares_l674_674567

theorem positive_diff_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 10) : a^2 - b^2 = 400 := by
  sorry

end positive_diff_of_squares_l674_674567


namespace probability_not_grey_l674_674628

/--
A box contains 1 grey ball, 2 white balls, and 3 black balls. John chooses one ball at random. 
Prove that the probability of choosing a non-grey ball is 5/6.
-/
theorem probability_not_grey :
  let total_balls := 1 + 2 + 3 in
  let non_grey_balls := 2 + 3 in
  non_grey_balls / total_balls = 5 / 6 :=
by
  let total_balls := 1 + 2 + 3
  let non_grey_balls := 2 + 3
  have h1 : total_balls = 6 := by rfl
  have h2 : non_grey_balls = 5 := by rfl
  rw [h1, h2]
  norm_num

end probability_not_grey_l674_674628


namespace total_dress_designs_l674_674216

def num_colors := 5
def num_patterns := 6
def num_sizes := 3

theorem total_dress_designs : num_colors * num_patterns * num_sizes = 90 :=
by
  sorry

end total_dress_designs_l674_674216


namespace three_digits_of_sequence_equal_294_l674_674095

/-- Define the sequence of numbers with the first digit 2 and 
    prove the 1498th, 1499th, and 1500th digits form the number 294.
-/
theorem three_digits_of_sequence_equal_294 :
  let sequence := List.filter (λ n, Nat.digits 10 n |> List.head! == 2) (List.range 10000) in
  let digits := List.bind Nat.digits (List.drop 1 sequence) in
  let relevant_digits := digits.drop 1497 |>.take 3 in
  Nat.ofDigits 10 relevant_digits = 294 := by
  sorry

end three_digits_of_sequence_equal_294_l674_674095


namespace calculate_value_l674_674680

theorem calculate_value :
  ( (3^3 - 1) / (3^3 + 1) ) * ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * ( (7^3 - 1) / (7^3 + 1) )
  = 57 / 84 := by
  sorry

end calculate_value_l674_674680


namespace smallest_k_for_a2005_zero_l674_674304

noncomputable def sequence (K : ℕ) : ℕ → ℕ
| 1       := K
| (n + 1) := if sequence n % 2 = 0 then sequence n - 1 else (sequence n - 1) / 2

theorem smallest_k_for_a2005_zero :
  ∃ K : ℕ, ∀ n : ℕ,
    (∀ i : ℕ, i < n → sequence K (i + 1) ≠ 0) ∧ sequence K 2005 = 0 →
    K = 2^1003 - 2 :=
begin
  sorry
end

end smallest_k_for_a2005_zero_l674_674304


namespace point_on_or_outside_circle_l674_674306

theorem point_on_or_outside_circle (a : ℝ) : 
  let P := (a, 2 - a)
  let r := 2
  let center := (0, 0)
  let distance_square := (P.1 - center.1)^2 + (P.2 - center.2)^2
  distance_square >= r := 
by
  sorry

end point_on_or_outside_circle_l674_674306


namespace count_valid_integers_between_1_and_200_l674_674296

def prime_factors_sum (n : ℕ) : ℕ := (unique_list_of_factors_of n).filter (λ p, prime p).sum

def is_valid_integer (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 200 ∧ prime_factors_sum n = 16

theorem count_valid_integers_between_1_and_200 : 
  {n : ℕ // is_valid_integer n}.to_finset.card = 6 :=
by
  sorry

end count_valid_integers_between_1_and_200_l674_674296


namespace store_paid_price_l674_674563

theorem store_paid_price (selling_price : ℕ) (less_amount : ℕ) 
(h1 : selling_price = 34) (h2 : less_amount = 8) : ∃ p : ℕ, p = selling_price - less_amount ∧ p = 26 := 
by
  sorry

end store_paid_price_l674_674563


namespace polynomial_example_l674_674890

theorem polynomial_example :
  ∃ P : ℝ → ℝ, (∀ x : ℝ, P x = (x - 1/2)^2001 + 1/2) ∧ (∀ x : ℝ, P x + P (1 - x) = 1) :=
begin
  let P := λ x : ℝ, (x - 1/2)^2001 + 1/2,
  use P,
  split,
  { intro x,
    reflexivity },
  { intro x,
    calc P x + P (1 - x)
          = ((x - 1/2)^2001 + 1/2) + ((1 - x - 1/2)^2001 + 1/2) : by reflexivity
      ... = ((x - 1/2)^2001 + 1/2) + (-(x - 1/2)^2001 + 1/2) : by {
        have : (1/2 - x)^2001 = -(x - 1/2)^2001,
        { -- should check that 2001 is an odd number
          sorry },
        simp only [sub_self, zero_add, add_assoc, add_left_comm, add_comm],
        exact this,
      },
    simp only [add_right_neg, add_zero, add_comm] },
end

end polynomial_example_l674_674890


namespace count_non_integer_interior_angles_l674_674475

theorem count_non_integer_interior_angles : 
  (∑ n in (Finset.Ico 3 15), if (180 * (n - 2)) % n ≠ 0 then 1 else 0) = 4 := 
by 
  sorry

end count_non_integer_interior_angles_l674_674475


namespace proof_b_c_value_l674_674355

def f (x b c : ℝ) := real.sqrt (-x^2 + b * x + c)

theorem proof_b_c_value :
  ∃ (b c : ℝ), (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f (-1) b c ≤ f x b c ∧ f x b c ≤ f (1) b c) ∧
                b * c + f 3 b c = 6 :=
by sorry

end proof_b_c_value_l674_674355


namespace bananas_to_oranges_l674_674906

theorem bananas_to_oranges :
  (3 / 4) * 16 * (1 / 1 : ℝ) = 10 * (1 / 1 : ℝ) → 
  (3 / 5) * 15 * (1 / 1 : ℝ) = 7.5 * (1 / 1 : ℝ) := 
by
  intros h
  sorry

end bananas_to_oranges_l674_674906


namespace complex_number_z0_exists_l674_674315

open Complex

noncomputable def polynomial_exists_z0 (n : ℕ) (C : Fin (n + 1) → ℂ) : Prop :=
  ∃ (z0 : ℂ), |z0| ≤ 1 ∧ |(C 0) * (z0 ^ n) + (C 1) * (z0 ^ (n - 1)) + (C 2) * (z0 ^ (n - 2)) + 
               ... + (C (n-1)) * z0 + (C n)| ≥ |C 0| + |C n|

theorem complex_number_z0_exists {n : ℕ} (C : Fin (n + 1) → ℂ)
  (h : ∀ z, z = (C 0) * (z ^ n) + (C 1) * (z ^ (n - 1)) + (C 2) * (z ^ (n - 2)) + 
               ... + (C (n-1)) * z + (C n)) :
  polynomial_exists_z0 n C :=
sorry

end complex_number_z0_exists_l674_674315


namespace radio_game_show_probability_l674_674232

def probability_of_winning : ℚ :=
  let p := 1 / 4 in
  (p ^ 4) + 4 * (p ^ 3 * (1 - p))

theorem radio_game_show_probability :
  probability_of_winning = 13 / 256 :=
by
  sorry

end radio_game_show_probability_l674_674232


namespace area_of_triangle_AME_l674_674510

noncomputable def area_triangle_AME (AB BC : ℝ) :=
let AC := real.sqrt (AB^2 + BC^2) in
let AM := AC / 2 in
let ME := BC / 2 in
1/2 * AM * ME

theorem area_of_triangle_AME :
  area_triangle_AME 12 10 = 5 * real.sqrt 244 / 4 :=
by
  sorry

end area_of_triangle_AME_l674_674510


namespace ball_redistribution_l674_674569

theorem ball_redistribution (n : ℕ) (h : n ≥ 1005) :
  ∃ (f : fin 2010 → ℕ), (∀ i, i < 2010 → f i = n) :=
begin
  sorry
end

end ball_redistribution_l674_674569


namespace logarithm_problem_l674_674482

noncomputable def f (a x : ℝ) : ℝ := Math.log x / Math.log a

theorem logarithm_problem 
  (a : ℝ) 
  (x : Fin 2010 → ℝ) 
  (h_a1 : 0 < a) 
  (h_a2 : a ≠ 1) 
  (h_f_product : f a (x 0 * x 1 * ... * x 2009) = 8) :
  (Finset.univ.sum (λ i : Fin 2010, f a ((x i) ^ 2))) = 16 := 
by
  -- add the proof here
  sorry

end logarithm_problem_l674_674482


namespace smallest_possible_n_l674_674868

theorem smallest_possible_n {n : ℕ} (x : ℕ → ℝ) (h1 : ∀ i, i < n → |x i| < 1) 
  (h2 : ∑ i in finset.range n, |x i| = 17 + |∑ i in finset.range n, x i|) : n = 18 :=
sorry

end smallest_possible_n_l674_674868


namespace john_initial_plays_l674_674070

noncomputable def initial_plays
  (acts_per_play : ℕ)
  (wigs_per_act : ℕ)
  (cost_per_wig : ℚ)
  (number_of_wigs_dropped : ℕ)
  (resale_value_per_wig : ℚ)
  (total_cost : ℚ) : ℕ :=
  (total_cost + number_of_wigs_dropped * resale_value_per_wig) / (acts_per_play * wigs_per_act * cost_per_wig)

theorem john_initial_plays :
  initial_plays 5 2 5 10 4 110 = 3 :=
by
  dsimp [initial_plays]
  norm_num
  sorry

end john_initial_plays_l674_674070


namespace smallest_positive_z_l674_674527

theorem smallest_positive_z (x z : ℝ) (n m : ℤ) :
  (sin x = 0 ∧ sin (x + z) = sqrt 3 / 2) →
  z = π / 3 :=
by
  sorry

end smallest_positive_z_l674_674527


namespace sum_exterior_angles_regular_pentagon_l674_674934

theorem sum_exterior_angles_regular_pentagon : 
  ∀ (pentagon : Type) [is_regular_polygon pentagon 5],  -- a regular pentagon with 5 sides
    sum_exterior_angles pentagon = 360 :=
by
  sorry

end sum_exterior_angles_regular_pentagon_l674_674934


namespace telescoping_product_l674_674704

theorem telescoping_product :
  (∏ x in {3, 4, 5, 6, 7}, (x^3 - 1) / (x^3 + 1)) = 57 / 168 := by
  sorry

end telescoping_product_l674_674704


namespace number_of_valid_As_l674_674203

def is_single_digit (n : ℕ) : Prop := n < 10

def divisible_by_8 (n : ℕ) : Prop := n % 8 = 0
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def valid_last_three_digits (b : ℕ) : Prop := 
  exists (k : ℕ), divisible_by_8 (75 + b * 1 + k * 1000)

def valid_digit_sum (a : ℕ) (b : ℕ) : Prop := 
  divisible_by_3 (a + 3 + 7 + 5 + b)

def valid_a (a : ℕ) : Prop := 
  is_single_digit a ∧ valid_digit_sum a 2

theorem number_of_valid_As : 
  (∑' (a : ℕ) (h : valid_a a), 1) = 3 :=
sorry

end number_of_valid_As_l674_674203


namespace function_is_periodic_l674_674922

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the function definition

axiom even_function (x : ℝ) : f x = f (-x)

axiom given_condition (x : ℝ) : f x + f (10 - x) = 4

theorem function_is_periodic : ∃ T > 0, ∀ x, f (x + T) = f x :=
by
  use 20
  intros x
  sorry

end function_is_periodic_l674_674922


namespace transform_polynomial_l674_674028

theorem transform_polynomial (x y : ℝ) (h1 : y = x + 1 / x) (h2 : x^4 - x^3 - 6 * x^2 - x + 1 = 0) :
  x^2 * (y^2 - y - 6) = 0 := 
  sorry

end transform_polynomial_l674_674028


namespace inequality_proof_l674_674467

noncomputable def a : ℝ := Real.log 7 / Real.log 3
noncomputable def b : ℝ := 2 ^ 1.1
noncomputable def c : ℝ := 0.8 ^ 3.1

theorem inequality_proof : c < a ∧ a < b := 
by
  have h1 : 1 < a,
    exact Real.log_pos (by norm_num) (by norm_num : 3 > 1)
  have h2 : a < 2,
    exact Real.exp_le_self (by tautology)
  have h3 : b > 2,
    calc
      b = 2 ^ 1.1  : rfl
      ... > 2 ^ 1  : by apply Real.pow_lt_pow_of_lt_left; norm_num
      ... = 2      : by norm_num
  have h4 : c < 0.8,
    exact Real.pow_lt_of_lt_one (by norm_num : 0.8 < 1) (by norm_num) (by norm_num)
  exact ⟨lt_trans h4 h1, h2.subst h3⟩

end inequality_proof_l674_674467


namespace nylon_cord_length_l674_674594

theorem nylon_cord_length
  (arc_length : ℝ)
  (π_approx : ℝ)
  (h_arc_length : arc_length = 30)
  (h_π_approx : π_approx = 3.14159) :
  let r := arc_length / π_approx
  in r ≈ 9.55 :=
by
  sorry

end nylon_cord_length_l674_674594


namespace commutative_or_associative_eq_l674_674078

variable {R : Type} [Mul R]

theorem commutative_or_associative_eq (h_comm : ∀ (a b : R), a * b = b * a ∨
                                           h_assoc : ∀ (a b c : R), a * (b * c) = (a * b) * c) :
    ∀ (x : R), x * (x * x) = (x * x) * x :=
by
  intro x
  cases h_comm x (x * x) with comm assoc
  {
    rw comm
  }
  {
    rw [←assoc, ←h_assoc x x x, h_assoc x x x]
  }
  sorry

end commutative_or_associative_eq_l674_674078


namespace relationship_xy_l674_674481

def f (t : ℝ) (ht : t > 1) : ℝ := t ^ (2 / (t - 1))
def g (t : ℝ) (ht : t > 1) : ℝ := t ^ ((t + 1) / (t - 1))

theorem relationship_xy (t : ℝ) (ht : t > 1) :
  (g t ht) ^ (2 * (f t ht)) = (f t ht) ^ (g t ht) :=
sorry

end relationship_xy_l674_674481


namespace star_k_l674_674273

def star (x y : ℤ) : ℤ := x^2 - 2 * y + 1

theorem star_k (k : ℤ) : star k (star k k) = -k^2 + 4 * k - 1 :=
by 
  sorry

end star_k_l674_674273


namespace number_of_valid_m_values_l674_674564

noncomputable def polynomial (m : ℤ) (x : ℤ) : ℤ := 
  2 * (m - 1) * x ^ 2 - (m ^ 2 - m + 12) * x + 6 * m

noncomputable def discriminant (m : ℤ) : ℤ :=
  (m ^ 2 - m + 12) ^ 2 - 4 * 2 * (m - 1) * 6 * m

def is_perfect_square (n : ℤ) : Prop :=
  ∃ (k : ℤ), k * k = n

def has_integral_roots (m : ℤ) : Prop :=
  ∃ (r1 r2 : ℤ), polynomial m r1 = 0 ∧ polynomial m r2 = 0

def valid_m_values (m : ℤ) : Prop :=
  (discriminant m) > 0 ∧ is_perfect_square (discriminant m) ∧ has_integral_roots m

theorem number_of_valid_m_values : 
  (∃ M : List ℤ, (∀ m ∈ M, valid_m_values m) ∧ M.length = 4) :=
  sorry

end number_of_valid_m_values_l674_674564


namespace range_of_a_l674_674345

-- Definition of point A
def A (a : ℝ) : ℝ × ℝ := (a, 3)

-- Definition of the circle equation
def circle (a : ℝ) : ℝ → ℝ → ℝ :=
λ x y, x^2 + y^2 - 2*a*x - 3*y + a^2 + a

-- The proof statement
theorem range_of_a (a : ℝ) (h_outside : ¬ (∃ x y : ℝ, circle a x y = 0 ∧ A a = (x, y))) : 0 < a ∧ a < 9 / 4 := 
sorry

end range_of_a_l674_674345


namespace eduardo_frankie_classes_total_l674_674724

theorem eduardo_frankie_classes_total (eduardo_classes : ℕ) (h₁ : eduardo_classes = 3) 
                                       (h₂ : ∀ frankie_classes, frankie_classes = 2 * eduardo_classes) :
  ∃ total_classes : ℕ, total_classes = eduardo_classes + 2 * eduardo_classes := 
by
  use 3 + 2 * 3
  sorry

end eduardo_frankie_classes_total_l674_674724


namespace function_machine_output_is_38_l674_674847

def function_machine (input : ℕ) : ℕ :=
  let multiplied := input * 3
  if multiplied > 40 then
    multiplied - 7
  else
    multiplied + 10

theorem function_machine_output_is_38 :
  function_machine 15 = 38 :=
by
   sorry

end function_machine_output_is_38_l674_674847


namespace mutually_exclusive_any_two_l674_674746

variables (A B C : Prop)
axiom all_not_defective : A
axiom all_defective : B
axiom not_all_defective : C

theorem mutually_exclusive_any_two :
  (¬(A ∧ B)) ∧ (¬(A ∧ C)) ∧ (¬(B ∧ C)) :=
sorry

end mutually_exclusive_any_two_l674_674746


namespace f_one_value_l674_674353

noncomputable def f : ℝ → ℝ
| x => if x ≥ 3 then (1 / 2) ^ x else f (x + 1)

theorem f_one_value : f 1 = 1 / 8 :=
by
  sorry

end f_one_value_l674_674353


namespace number_of_ways_l674_674112

-- Definitions based on the given conditions
def people : Type := {a, b, c, d, e, f}
def cities : Type := {paris, london, sydney, moscow}
def visit (p : people) (c : cities) : Prop := sorry

-- Conditions
def constraints (p : people) (c : cities) : Prop :=
  (p ≠ a ∨ c ≠ paris) ∧
  (p ≠ b ∨ c ≠ paris)

-- Statement of the problem
theorem number_of_ways :
  (finset.card {f : people → cities | 
      function.injective f ∧
      (∀ p c, visit p c → constraints p c)} = 240) :=
begin
  sorry
end

end number_of_ways_l674_674112


namespace num_perfect_square_factors_of_8820_l674_674559

theorem num_perfect_square_factors_of_8820 : 
  let factors := \{m : ℕ | ∃ a b c d, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧ m = 2^a * 3^b * 5^c * 7^d\}
  in (∃ perfect_squares : finset ℕ, 
        perfect_squares = factors.filter (λ n, ∀ k : ℕ, n = k^2 → true)) →
     8 :=
by
  sorry

end num_perfect_square_factors_of_8820_l674_674559


namespace fastest_route_is_B_l674_674118

-- Definitions for the distances of each route
def distance_A := 1500
def distance_B := 1300
def distance_C := 1800
def distance_D := 750

-- Definitions for the average speeds of each route
def speed_A := 75
def speed_B := 70
def speed_C := 80
def speed_D := 25

-- Definitions for the respective delays and breaks
def traffic_delay_A := 2
def rest_stops_A := 3 * 30 / 60
def meal_break_A := 1
def fuel_stops_A := 2 * 10 / 60
def road_closures_A := 1.5

def traffic_delay_B := 0
def rest_stops_B := 2 * 45 / 60
def meal_break_B := 45 / 60
def fuel_stops_B := 3 * 15 / 60
def construction_delay_B := 1

def traffic_delay_C := 2.5
def rest_stops_C := 4 * 20 / 60
def meal_breaks_C := 2 * 1
def fuel_stops_C := 2 * 10 / 60
def road_closures_C := 2

def traffic_delay_D := 0
def rest_stop_D := 1
def meal_break_D := 30 / 60
def fuel_stop_D := 20 / 60
def construction_delay_D := 3

-- Calculate the total time for each route
noncomputable def total_time_A := (distance_A / speed_A) + traffic_delay_A + rest_stops_A + meal_break_A + fuel_stops_A + road_closures_A
noncomputable def total_time_B := (distance_B / speed_B) + traffic_delay_B + rest_stops_B + meal_break_B + fuel_stops_B + construction_delay_B
noncomputable def total_time_C := (distance_C / speed_C) + traffic_delay_C + rest_stops_C + meal_breaks_C + fuel_stops_C + road_closures_C
noncomputable def total_time_D := (distance_D / speed_D) + traffic_delay_D + rest_stop_D + meal_break_D + fuel_stop_D + construction_delay_D

-- Prove that Route B is the fastest
theorem fastest_route_is_B : total_time_B < total_time_A ∧ total_time_B < total_time_C ∧ total_time_B < total_time_D := by
  sorry

end fastest_route_is_B_l674_674118


namespace no_convex_function_exists_l674_674030

theorem no_convex_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), (f(x) + f(y)) / 2 ≥ f((x + y) / 2) + |x - y| := by
  sorry

end no_convex_function_exists_l674_674030


namespace quadratic_other_root_l674_674322

theorem quadratic_other_root (m x2 : ℝ) (h₁ : 1^2 - 4*1 + m = 0) (h₂ : x2^2 - 4*x2 + m = 0) : x2 = 3 :=
sorry

end quadratic_other_root_l674_674322


namespace problem_order_relations_l674_674081

noncomputable def a : ℝ := Real.logBase (1/3) 2
noncomputable def b : ℝ := (1/4)^0.1
noncomputable def c : ℝ := (1/2)^0.3

theorem problem_order_relations : a < c ∧ c < b := by
  sorry

end problem_order_relations_l674_674081


namespace abc_inequality_l674_674108

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b + b * c + a * c)^2 ≥ 3 * a * b * c * (a + b + c) :=
by sorry

end abc_inequality_l674_674108


namespace minimize_cost_l674_674645

-- Define the unit prices of the soccer balls.
def price_A := 50
def price_B := 80

-- Define the condition for the total number of balls and cost function.
def total_balls := 80
def cost (a : ℕ) : ℕ := price_A * a + price_B * (total_balls - a)
def valid_a (a : ℕ) : Prop := 30 ≤ a ∧ a ≤ (3 * (total_balls - a))

-- Prove the number of brand A soccer balls to minimize the total cost.
theorem minimize_cost : ∃ a : ℕ, valid_a a ∧ ∀ b : ℕ, valid_a b → cost a ≤ cost b :=
sorry

end minimize_cost_l674_674645


namespace equilateral_triangle_side_length_l674_674665

theorem equilateral_triangle_side_length (a : ℝ) :
  (∃ (a : ℝ), let perimeters_sum := ∑' n : ℕ, (3 * a / 2^n) in perimeters_sum = 240) →
  a = 40 :=
by
  intros h
  have series_sum : ∑' n : ℕ, (3 * a / 2^n) = 3 * a * (∑' n : ℕ, (1 / 2)^n),
    from sorry,
  have geometric_series_sum : ∑' n : ℕ, (1 / 2)^n = 2,
    from sorry,
  rw series_sum at h,
  rw geometric_series_sum at h,
  simp at h,
  exact h

-- sorry statement is included as we are not proving the theorem, just stating it.

end equilateral_triangle_side_length_l674_674665


namespace actual_distance_travelled_l674_674412

theorem actual_distance_travelled :
  ∃ (D : ℝ), (D / 10 = (D + 20) / 14) ∧ D = 50 :=
by
  sorry

end actual_distance_travelled_l674_674412


namespace rainfall_third_day_is_18_l674_674954

-- Define the conditions including the rainfall for each day
def rainfall_first_day : ℕ := 4
def rainfall_second_day : ℕ := 5 * rainfall_first_day
def rainfall_third_day : ℕ := (rainfall_first_day + rainfall_second_day) - 6

-- Prove that the rainfall on the third day is 18 inches
theorem rainfall_third_day_is_18 : rainfall_third_day = 18 :=
by
  -- Use the definitions and directly state that the proof follows
  sorry

end rainfall_third_day_is_18_l674_674954


namespace z_in_first_quadrant_l674_674439

-- Definition of the complex number z
def z : ℂ := 1 / (2 - I)

-- Statement of the proof problem
theorem z_in_first_quadrant : z.re > 0 ∧ z.im > 0 := 
by
  -- placeholder for the actual proof
  sorry

end z_in_first_quadrant_l674_674439


namespace telescoping_product_l674_674703

theorem telescoping_product :
  (∏ x in {3, 4, 5, 6, 7}, (x^3 - 1) / (x^3 + 1)) = 57 / 168 := by
  sorry

end telescoping_product_l674_674703


namespace number_of_allocation_schemes_l674_674562

-- Define the total number of spots and classes
def total_spots : ℕ := 5
def total_classes : ℕ := 4
def min_spots_for_class_A : ℕ := 2

-- Define a predicate that an allocation is valid
def valid_allocation (allocation : Vector ℕ total_classes) : Prop :=
  allocation.head ≥ min_spots_for_class_A ∧ allocation.sum = total_spots

-- State the theorem
theorem number_of_allocation_schemes : 
  (Finset.card {allocation : Vector ℕ total_classes | valid_allocation allocation}) = 20 := 
sorry

end number_of_allocation_schemes_l674_674562


namespace compute_fraction_product_l674_674695

theorem compute_fraction_product :
  (∏ i in (finset.range 5).map (λ n, n + 3), (i ^ 3 - 1) / (i ^ 3 + 1)) = (57 / 168) := by
  sorry

end compute_fraction_product_l674_674695


namespace time_addition_correct_l674_674851

def start_time := (3, 0, 0) -- Representing 3:00:00 PM as (hours, minutes, seconds)
def additional_time := (315, 78, 30) -- Representing additional time as (hours, minutes, seconds)

noncomputable def resulting_time (start add : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (sh, sm, ss) := start -- start hours, minutes, seconds
  let (ah, am, as) := add -- additional hours, minutes, seconds
  let total_seconds := ss + as
  let extra_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_minutes := sm + am + extra_minutes
  let extra_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let total_hours := sh + ah + extra_hours
  let resulting_hours := (total_hours % 12) -- Modulo 12 for wrap-around
  (resulting_hours, remaining_minutes, remaining_seconds)

theorem time_addition_correct :
  let (A, B, C) := resulting_time start_time additional_time
  A + B + C = 55 := by
  sorry

end time_addition_correct_l674_674851


namespace solve_equation_l674_674114

theorem solve_equation (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 2)
  (h₃ : (3 * x + 6)/(x^2 + 5 * x + 6) = (3 - x)/(x - 2)) :
  x = 3 ∨ x = -3 :=
sorry

end solve_equation_l674_674114


namespace num_terms_in_expansion_eq_3_pow_20_l674_674059

-- Define the expression 
def expr (x y : ℝ) := (1 + x + y) ^ 20

-- Statement of the problem
theorem num_terms_in_expansion_eq_3_pow_20 (x y : ℝ) : (3 : ℝ)^20 = (1 + x + y) ^ 20 :=
by sorry

end num_terms_in_expansion_eq_3_pow_20_l674_674059


namespace zeros_not_adjacent_probability_l674_674373

def total_arrangements : ℕ := Nat.factorial 5

def adjacent_arrangements : ℕ := 2 * Nat.factorial 4

def probability_not_adjacent : ℚ := 
  1 - (adjacent_arrangements / total_arrangements)

theorem zeros_not_adjacent_probability :
  probability_not_adjacent = 0.6 := 
by 
  sorry

end zeros_not_adjacent_probability_l674_674373


namespace A_greater_than_B_l674_674584

noncomputable def A : ℝ := (1 : ℝ) / 2015 * (∑ k in Finset.range (2015 + 1), (1 : ℝ) / k)
noncomputable def B : ℝ := (1 : ℝ) / 2016 * (∑ k in Finset.range (2016 + 1), (1 : ℝ) / k)

theorem A_greater_than_B : A > B :=
by sorry

end A_greater_than_B_l674_674584


namespace complex_calculation_l674_674962

noncomputable def a : ℂ := 5 - 3 * complex.I
noncomputable def b : ℂ := 2 + 4 * complex.I

theorem complex_calculation :
  3 * a - 4 * b = 7 - 25 * complex.I := 
by sorry

end complex_calculation_l674_674962


namespace max_f_on_interval_l674_674733

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + (Real.sqrt 3) * Real.sin x * Real.cos x

theorem max_f_on_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ ∀ y ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f y ≤ f x ∧ f x = 3 / 2 :=
  sorry

end max_f_on_interval_l674_674733


namespace possible_num_of_males_surveyed_l674_674163

-- Definitions based on conditions
def num_male_students (n : ℕ) : ℕ := 5 * n
def num_female_students (n : ℕ) : ℕ := 5 * n
def likes_male (n : ℕ) : ℕ := 2 * n
def likes_female (n : ℕ) : ℕ := 4 * n
def cereal_male (n : ℕ) : ℕ := 3 * n
def cereal_female (n : ℕ) : ℕ := n
def total_students (n : ℕ) : ℕ := 10 * n

-- Given formula for K^2
def K_squared (n : ℕ) :=
  let a := likes_male n
  let b := likes_female n
  let c := cereal_male n
  let d := cereal_female n
  let total := total_students n
  (total * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

def condition (n : ℕ) : Prop :=
  3.841 ≤ K_squared n ∧ K_squared n < 6.635

-- Lean 4 mathematical equivalent proof problem statement
theorem possible_num_of_males_surveyed (n : ℕ) (h1: condition n) : num_male_students n = 15 :=
sorry

end possible_num_of_males_surveyed_l674_674163


namespace system_solution_l674_674524

theorem system_solution :
  ∀ x y : ℝ, 
    ((3 / 2)^(x - y) - (2 / 3)^(x - y) = 65 / 36 ∧ x * y - x + y = 118) ↔ 
      (x = 12 ∧ y = 10) ∨ (x = -10 ∧ y = -12) :=
by
  sorry

end system_solution_l674_674524


namespace sum_of_possible_values_of_g_l674_674933

-- Conditions extracted from the problem statement
def is_magic_square (matrix : List (List ℕ)) : Prop :=
  (matrix.length = 3 ∧
   matrix.all (λ row, row.length = 3) ∧
   matrix.all (λ row, row.all (λ x, x > 0)) ∧
   let ⟨row1, row2, row3⟩ := (matrix.nth! 0, matrix.nth! 1, matrix.nth! 2) in
   let ⟨a, b, c⟩ := (row1.nth! 0, row1.nth! 1, row1.nth! 2) in
   let ⟨d, e, f⟩ := (row2.nth! 0, row2.nth! 1, row2.nth! 2) in
   let ⟨g, h, i⟩ := (row3.nth! 0, row3.nth! 1, row3.nth! 2) in
   let P := a * b * c in
   P = (a * b * c) ∧ P = (d * e * f) ∧ P = (g * h * i) ∧
   P = (a * e * i) ∧ P = (c * e * g) ∧
   P = (a * e * i) ∧ P = (a * e * g) ∧
   a = 25 ∧ i = 3 ∧ b = 75 ∧ h = e
  )

-- Lean 4 statement for the problem
theorem sum_of_possible_values_of_g :
  ∀ (matrix : List (List ℕ)),
  is_magic_square matrix →
  let ⟨_, row2, row3⟩ := (matrix.nth! 0, matrix.nth! 1, matrix.nth! 2) in
  let ⟨g, _, _⟩ := (row3.nth! 0, row3.nth! 1, row3.nth! 2) in
  g = 25 ∨ g = 15 → (25 + 15) = 40 :=
by
  intros matrix h mg
  cases mg with cond1 cond2
  simp only [] at cond1 cond2
  sorry

end sum_of_possible_values_of_g_l674_674933


namespace initial_shells_l674_674455

theorem initial_shells (added_shells total_shells initial_shells: ℕ) 
  (h_add: added_shells = 12) (h_total: total_shells = 17) : initial_shells = 5 :=
by 
  have h := total_shells - added_shells
  rw [h_add, h_total] at h
  exact congr_arg Nat.pred h

end initial_shells_l674_674455


namespace math_problem_proof_l674_674338

section
variable {a : ℕ → ℤ}
variable (d : ℤ)
variable (n : ℕ)

-- (I) Given conditions
def arithmetic_sequence_cond1 : Prop :=
  a 2 = 5 ∧ a 5 = 14

-- (II) General term formula
def general_term_formula : Prop :=
  ∀ n, a n = 3 * n - 1

-- (III) Sum of first n terms
def sum_of_first_n_terms : ℤ :=
  (n * (a 1 + a n)) / 2

def sum_cond : Prop :=
  sum_of_first_n_terms 10 = 155

-- Proof problem
theorem math_problem_proof :
  arithmetic_sequence_cond1 →
  general_term_formula →
  sum_cond →
  sum_of_first_n_terms n = 155 →
  n = 10 := sorry
end

end math_problem_proof_l674_674338


namespace major_premise_error_in_argument_l674_674509

theorem major_premise_error_in_argument :
  ¬ (∀ (f : ℝ → ℝ) (x : ℝ), deriv f x = 0 → ∀ y, f x = y → ∀ y' ≠ y, false) →
  deriv (λ x : ℝ, x^3) 0 = 0 →
  ¬ ∃ (y : ℝ), (λ x : ℝ, x^3) 0 = y ∧ (∀ y' ≠ y, false) :=
by
  intro major_premise incorrect_premise x0_zero
  have derivative_zero_at_0 : deriv (λ x : ℝ, x^3) 0 = 0 := by sorry
  have is_extreme : ¬ ∃ y, (λ x : ℝ, x^3) 0 = y ∧ ¬ (λ x : ℝ, x^3) x is not extreme at this point := by sorry
  exact is_extreme

end major_premise_error_in_argument_l674_674509


namespace find_other_root_of_quadratic_l674_674325

theorem find_other_root_of_quadratic (m x_1 x_2 : ℝ) 
  (h_root1 : x_1 = 1) (h_eqn : ∀ x, x^2 - 4 * x + m = 0) : x_2 = 3 :=
by
  sorry

end find_other_root_of_quadratic_l674_674325


namespace arithmetic_seq_sum_l674_674846

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : a 1 + a 4 + a 7 = 39) (h2 : a 3 + a 6 + a 9 = 27) :
  (∑ i in Finset.range 9, a (i + 1)) = 99 :=
sorry

end arithmetic_seq_sum_l674_674846


namespace jill_second_bus_time_l674_674950

-- Define constants representing the times
def wait_time_first_bus : ℕ := 12
def ride_time_first_bus : ℕ := 30

-- Define a function to calculate the total time for the first bus
def total_time_first_bus (wait : ℕ) (ride : ℕ) : ℕ :=
  wait + ride

-- Define a function to calculate the time for the second bus
def time_second_bus (total_first_bus_time : ℕ) : ℕ :=
  total_first_bus_time / 2

-- The theorem to prove
theorem jill_second_bus_time : 
  time_second_bus (total_time_first_bus wait_time_first_bus ride_time_first_bus) = 21 := by
  sorry

end jill_second_bus_time_l674_674950


namespace distance_from_house_to_work_l674_674129

-- Definitions for the conditions
variables (D : ℝ) (speed_to_work speed_back_work : ℝ) (time_to_work time_back_work total_time : ℝ)

-- Specific conditions in the problem
noncomputable def conditions : Prop :=
  (speed_back_work = 20) ∧
  (speed_to_work = speed_back_work / 2) ∧
  (time_to_work = D / speed_to_work) ∧
  (time_back_work = D / speed_back_work) ∧
  (total_time = 6) ∧
  (time_to_work + time_back_work = total_time)

-- The statement to prove the distance D is 40 km given the conditions
theorem distance_from_house_to_work (h : conditions D speed_to_work speed_back_work time_to_work time_back_work total_time) : D = 40 :=
sorry

end distance_from_house_to_work_l674_674129


namespace min_shift_for_monotonic_decrease_l674_674515

noncomputable def shifted_sine_is_monotonically_decreasing (m : ℝ) : Prop :=
  ∀ x : ℝ, -π / 12 ≤ x ∧ x ≤ 5π / 12 → 
    (real.sin (2 * (x + m) + π / 6) - real.sin (2 * (x + m + ε) + π / 6) < 0)

theorem min_shift_for_monotonic_decrease :
  ∃ m > 0, shifted_sine_is_monotonically_decreasing (m) ∧ m = π / 4 :=
sorry

end min_shift_for_monotonic_decrease_l674_674515


namespace determine_a_if_fx_odd_l674_674823

theorem determine_a_if_fx_odd (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = 2^x + a * 2^(-x)) (h2 : ∀ x, f (-x) = -f x) : a = -1 :=
by
  sorry

end determine_a_if_fx_odd_l674_674823


namespace stddev_transformed_data_l674_674013

noncomputable def stddev (X : List ℝ) : ℝ := sorry

theorem stddev_transformed_data (X : List ℝ) (h : stddev X = 8) :
  stddev (X.map (λ x, 2 * x - 1)) = 16 := by
  sorry

end stddev_transformed_data_l674_674013


namespace can_derive_card_1_50_cannot_derive_card_1_100_derive_card_1_n_l674_674568

-- Defining automata operations as functions
def first_automaton (a b : ℕ) : ℕ × ℕ := (a + 1, b + 1)
def second_automaton (a b : ℕ) : option (ℕ × ℕ) := if a % 2 = 0 ∧ b % 2 = 0 then some (a / 2, b / 2) else none
def third_automaton (a b c : ℕ) : ℕ × ℕ := (a, c)

-- Problem 1: Can we derive card (1, 50) from (5, 19)?
theorem can_derive_card_1_50 : ∃ (seq : list (ℕ × ℕ)) (initial := (5, 19)), 
  list.last seq = some (1, 50) ∧ 
  (seq.ilast ∈ [{p : ℕ × ℕ | ∃ q, first_automaton q.1 q.2 = p} ∪
                {p : ℕ × ℕ | ∃ q, second_automaton q.1 q.2 = some p} ∪
                {p : ℕ × ℕ | ∃ q r, third_automaton q.1 q.2 p.2 = p}]) := sorry

-- Problem 2: Can we derive card (1, 100) from (5, 19)? 
theorem cannot_derive_card_1_100 : ¬ ∃ (seq : list (ℕ × ℕ)) (initial := (5, 19)), 
  list.last seq = some (1, 100) ∧ 
  (seq.ilast ∈ [{p : ℕ × ℕ | ∃ q, first_automaton q.1 q.2 = p} ∪
                {p : ℕ × ℕ | ∃ q, second_automaton q.1 q.2 = some p} ∪
                {p : ℕ × ℕ | ∃ q r, third_automaton q.1 q.2 p.2 = p}]) := sorry

-- Problem 3: Given (a, b) where a < b, derive (1, n=1+(b-a)*k).
theorem derive_card_1_n (a b k : ℕ) (h : a < b) : 
  ∃ (seq : list (ℕ × ℕ)), (initial := (a, b)) ∧ 
  list.last seq = some (1, 1 + (b - a) * k) ∧ 
  (seq.ilast ∈ [{p : ℕ × ℕ | ∃ q, first_automaton q.1 q.2 = p} ∪
                {p : ℕ × ℕ | ∃ q, second_automaton q.1 q.2 = some p} ∪
                {p : ℕ × ℕ | ∃ q r, third_automaton q.1 q.2 p.2 = p}]) := sorry

end can_derive_card_1_50_cannot_derive_card_1_100_derive_card_1_n_l674_674568


namespace f_inv_sum_l674_674074

def f (x : ℝ) : ℝ :=
  if x < 15 then x + 2 else 2 * x + 1

def f_inv_10 : ℝ :=
  if 10 - 2 < 15 then 10 - 2 else (10 - 1) / 2

def f_inv_37 : ℝ :=
  if (37 - 1) / 2 < 15 then 37 - 2 else (37 - 1) / 2

theorem f_inv_sum : f_inv_10 + f_inv_37 = 26 := by
  -- Function definitions
  have h1 : f_inv_10 = 8 := by
    unfold f_inv_10
    simp [if_pos (show 10 - 2 < 15 from by norm_num)]
  have h2 : f_inv_37 = 18 := by
    unfold f_inv_37
    simp [if_neg (show ¬((37 - 1) / 2 < 15) from by norm_num)]
  -- Result
  rw [h1, h2]
  norm_num

end f_inv_sum_l674_674074


namespace product_of_sum_positive_and_quotient_negative_l674_674834

-- Definitions based on conditions in the problem
def sum_positive (a b : ℝ) : Prop := a + b > 0
def quotient_negative (a b : ℝ) : Prop := a / b < 0

-- Problem statement as a theorem
theorem product_of_sum_positive_and_quotient_negative (a b : ℝ)
  (h1 : sum_positive a b)
  (h2 : quotient_negative a b) :
  a * b < 0 := by
  sorry

end product_of_sum_positive_and_quotient_negative_l674_674834


namespace line_does_not_pass_through_point_l674_674000

theorem line_does_not_pass_through_point 
  (m : ℝ) (h : (2*m + 1)^2 - 4*(m^2 + 4) > 0) : 
  ¬((2*m - 3)*(-2) - 4*m + 7 = 1) :=
by
  sorry

end line_does_not_pass_through_point_l674_674000


namespace calculate_value_l674_674681

theorem calculate_value :
  ( (3^3 - 1) / (3^3 + 1) ) * ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * ( (7^3 - 1) / (7^3 + 1) )
  = 57 / 84 := by
  sorry

end calculate_value_l674_674681


namespace back_wheel_revolutions_l674_674101

-- Defining relevant distances and conditions
def front_wheel_radius : ℝ := 3 -- radius in feet
def back_wheel_radius : ℝ := 0.5 -- radius in feet
def front_wheel_revolutions : ℕ := 120

-- The target theorem
theorem back_wheel_revolutions :
  let front_wheel_circumference := 2 * Real.pi * front_wheel_radius
  let total_distance := front_wheel_circumference * (front_wheel_revolutions : ℝ)
  let back_wheel_circumference := 2 * Real.pi * back_wheel_radius
  let back_wheel_revs := total_distance / back_wheel_circumference
  back_wheel_revs = 720 :=
by
  sorry

end back_wheel_revolutions_l674_674101


namespace sum_real_parts_correct_l674_674047

noncomputable def center : ℂ := (1 + complex.i)
noncomputable def vertex : ℂ := (1 + complex.i, -4 * complex.i)
noncomputable def focus : ℂ := (1 + complex.i, 8 * complex.i)

noncomputable def h := 1 + complex.i
noncomputable def k := 0
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := 8
noncomputable def b : ℝ := real.sqrt (c^2 - a^2)

noncomputable def sum_real_parts : ℝ :=
  complex.re h + complex.re k + complex.re a + complex.re b

theorem sum_real_parts_correct : 
  sum_real_parts = 5 + 4 * real.sqrt (3) := 
sorry

end sum_real_parts_correct_l674_674047


namespace smallest_n_for_rotation_matrix_l674_674737

open Real
open Matrix

-- Define the rotation matrix for 60 degrees
def rotationMatrix60 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![cos (π / 3), -sin (π / 3)],
    ![sin (π / 3), cos (π / 3)]
  ]

-- Identity matrix of size 2
def identityMatrix2 : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem smallest_n_for_rotation_matrix :
  ∃ n : ℕ, n > 0 ∧ (rotationMatrix60 ^ n = identityMatrix2) ∧ ∀ m : ℕ, m > 0 → m < n → rotationMatrix60 ^ m ≠ identityMatrix2 :=
by 
  existsi 6
  split
  · exact Nat.succ_pos'
  · sorry
  · sorry

end smallest_n_for_rotation_matrix_l674_674737


namespace graph_of_log2_function_l674_674134

theorem graph_of_log2_function (x y : ℝ) (H : (y = log 2 (x + 1))) :
  (0,0) ∈ {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ y = log 2 (x + 1)} :=
by sorry

end graph_of_log2_function_l674_674134


namespace venerable_power_of_two_l674_674960

def is_venerable (n : ℕ) : Prop :=
  ∑ d in (Finset.range n).filter (λ d => n % d = 0), d = n - 1

theorem venerable_power_of_two (n : ℕ) (m : ℕ) (h : m > 1) :
  is_venerable n →
  is_venerable (n^m) →
  ∃ k > 0, n = 2^k :=
by
  sorry

end venerable_power_of_two_l674_674960


namespace cos_squared_plus_cos_fourth_l674_674400

theorem cos_squared_plus_cos_fourth (α : ℝ) (h : sin α + (sin α) ^ 2 = 1) : cos α ^ 2 + (cos α) ^ 4 = 1 :=
by
  sorry

end cos_squared_plus_cos_fourth_l674_674400


namespace eval_expression_l674_674980

-- Define the absolute value conditions
def abs (x : Int) : Int :=
  if x < 0 then -x else x

-- The problem states that | -25 | = 25, | 5 | = 5, and we need to prove the following:
theorem eval_expression : abs (- 2) * (abs (- 25) - abs 5) = -40 :=
by
  -- we skip the proof details
  sorry

end eval_expression_l674_674980


namespace sum_of_elements_l674_674272

def A : Set ℝ := {0, 2}
def B : Set ℝ := {1, 2}
def C : Set ℝ := {1}

def nabla (X Y : Set ℝ) : Set ℝ := {z | ∃ x ∈ X, ∃ y ∈ Y, z = x * y + x / y}

def A_nabla_B : Set ℝ := nabla A B
def result_set : Set ℝ := nabla A_nabla_B C

theorem sum_of_elements : (∑ z in result_set, id z) = 18 :=
by
  sorry

end sum_of_elements_l674_674272


namespace number_of_distinct_real_solutions_l674_674469

def f (x: ℝ) : ℝ :=
  x^2 + 2 * x

theorem number_of_distinct_real_solutions :
  {c : ℝ | f (f (f (f c))) = 0 }.toFinset.card = 2 :=
by
  sorry

end number_of_distinct_real_solutions_l674_674469


namespace log_range_conditions_l674_674035

theorem log_range_conditions (a : ℝ) : 
  (log (2 * a - 1) (a^2 - 2 * a + 1) > 0) → 
  (a ∈ set.Ioo 0.5 1 ∪ set.Ioi 2) := by
  sorry

end log_range_conditions_l674_674035


namespace sum_exterior_angles_regular_pentagon_l674_674935

theorem sum_exterior_angles_regular_pentagon : 
  ∀ (pentagon : Type) [is_regular_polygon pentagon 5],  -- a regular pentagon with 5 sides
    sum_exterior_angles pentagon = 360 :=
by
  sorry

end sum_exterior_angles_regular_pentagon_l674_674935


namespace zeros_not_adjacent_probability_l674_674376

def total_arrangements : ℕ := Nat.factorial 5

def adjacent_arrangements : ℕ := 2 * Nat.factorial 4

def probability_not_adjacent : ℚ := 
  1 - (adjacent_arrangements / total_arrangements)

theorem zeros_not_adjacent_probability :
  probability_not_adjacent = 0.6 := 
by 
  sorry

end zeros_not_adjacent_probability_l674_674376


namespace polar_form_equivalence_l674_674540

-- Definitions based on the problem conditions
def cis (θ : ℝ) : ℂ := complex.exp (complex.I * θ * real.pi / 180)

def polar_form (r : ℝ) (θ : ℝ) : ℂ := r * cis θ

-- Given problem conditions
def complex_mul (a b : ℂ) (r1 θ1 r2 θ2 : ℝ) : Prop :=
  a = polar_form r1 θ1 ∧ b = polar_form r2 θ2

-- Correct answer
def result : Prop :=
  polar_form (4 * -3) (25 + 48) = polar_form 12 253

-- Lean statement to prove the problem
theorem polar_form_equivalence (a b c : ℂ) (r1 θ1 r2 θ2 r3 θ3 : ℝ)
  (h1 : 0 ≤ θ1 ∧ θ1 < 360)
  (h2 : 0 ≤ θ2 ∧ θ2 < 360)
  (h3 : r1 > 0)
  (h4 : r2 < 0)
  (h5 : r3 > 0):
  complex_mul a b r1 θ1 r2 θ2 →
  c = polar_form r3 θ3 →
  r1 * -r2 = r3 ∧ (θ1 + θ2 + 180) % 360 = θ3 :=
sorry

end polar_form_equivalence_l674_674540


namespace value_of_g_neg2_l674_674131

def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem value_of_g_neg2 : g (-2) = -1 := by
  sorry

end value_of_g_neg2_l674_674131


namespace shortest_distance_between_circles_l674_674172

noncomputable def circle1_eq : ℝ → ℝ → Prop :=
  λ x y, x^2 - 6*x + y^2 - 8*y + 9 = 0

noncomputable def circle2_eq : ℝ → ℝ → Prop :=
  λ x y, x^2 + 8*x + y^2 + 2*y + 16 = 0

theorem shortest_distance_between_circles :
  (∀ x y, circle1_eq x y → (x-3)^2 + (y-4)^2 = 49) →
  (∀ x' y', circle2_eq x' y' → (x'+4)^2 + (y'+1)^2 = 1) →
  sqrt((3 - (-4))^2 + (4 - (-1))^2) - (7 + 1) = sqrt(74) - 8 :=
by
  intro h1 h2
  sorry

end shortest_distance_between_circles_l674_674172


namespace find_a_l674_674202

noncomputable def f (x : ℝ) : ℝ := 5^(abs x)

noncomputable def g (a x : ℝ) : ℝ := a*x^2 - x

theorem find_a (a : ℝ) (h : f (g a 1) = 1) : a = 1 := 
by
  sorry

end find_a_l674_674202


namespace intersection_of_M_and_N_l674_674012

def set_M (x : ℝ) : Prop := 1 - 2 / x < 0
def set_N (x : ℝ) : Prop := -1 ≤ x
def set_Intersection (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem intersection_of_M_and_N :
  ∀ x, (set_M x ∧ set_N x) ↔ set_Intersection x :=
by sorry

end intersection_of_M_and_N_l674_674012


namespace problem_specified_l674_674789

-- We will define our conditions as Lean propositions.
variables (a b c : ℕ)

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def angle_condition (a b c : ℕ) : Prop :=
  ∃ x, times_angle a b x ∧ half_angle x c b

noncomputable def minimal_b : ℕ := 1991^2

-- The Lean proposition encoding the problem and proving b == minimal_b under given conditions.
theorem problem_specified (h1 : is_valid_triangle a b c) (h2 : angle_condition a b c)
  (1990_triangles : exactly_1990_triangles a b c) : b = minimal_b := 
sorry

-- Auxiliary definitions, if needed for the predicates times_angle, half_angle, and exactly_1990_triangles.
def times_angle (a b : ℕ) (x : ℕ) : Prop := 
-- Define based on the angle condition related to sides a and b

def half_angle (x : ℕ) (c b : ℕ) : Prop :=
-- Define based on the condition ∠ABC = 1/2 * ∠BAC
  
def exactly_1990_triangles (a b c : ℕ) : Prop :=
-- Define based on the condition there are exactly 1990 triangles satisfying the properties

end problem_specified_l674_674789


namespace max_min_diff_l674_674778

def quadratic_function (x : ℝ) : ℝ := -x^2 + 10 * x + 9

theorem max_min_diff (a : ℝ) (h : 8 ≤ a / 9) :
  let f := quadratic_function in
  let interval := { x : ℝ | 2 ≤ x ∧ x ≤ a / 9 } in
  let max_val := 34 in
  let min_val := 25 in
  (∃ x ∈ interval, f x = max_val) ∧ 
  (∀ x ∈ interval, f x ≥ min_val) → 
  max_val - min_val = 9 := 
sorry

end max_min_diff_l674_674778


namespace time_for_train_to_pass_man_l674_674189

-- Define constants
def length_of_train : ℝ := 110
def speed_of_train_kmh : ℝ := 56
def speed_of_man_kmh : ℝ := 6

-- Convert speeds from km/hr to m/s
def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600)
def speed_of_man_ms : ℝ := speed_of_man_kmh * (1000 / 3600)

-- Define the relative speed since they are moving in opposite directions
def relative_speed_ms : ℝ := speed_of_train_ms + speed_of_man_ms

-- Calculation of time
def time_to_pass : ℝ := length_of_train / relative_speed_ms

-- The main theorem statement
theorem time_for_train_to_pass_man : time_to_pass ≈ 6.39 :=
by
  sorry

end time_for_train_to_pass_man_l674_674189


namespace time_taken_by_B_l674_674592

theorem time_taken_by_B :
  let A := 1 / 4 in
  let C := (1 / 2) - A in
  let B := (1 / 3) - C in
  (1 / B) = 12 :=
by
  sorry

end time_taken_by_B_l674_674592


namespace distribution_of_books_l674_674940

theorem distribution_of_books (students books : ℕ) (h_students : students = 7) (h_books : books = 3) :
  ∃ (ways : ℕ), ways = students.perm books ∧ ways = 210 :=
by
  sorry

end distribution_of_books_l674_674940


namespace unique_maximized_distance_line_l674_674100

noncomputable theory
open Classical

structure Point (α : Type) :=
(x : α)
(y : α)

def distance (p₁ p₂ : Point ℝ) : ℝ :=
real.sqrt ((p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2)

def line_through (p₁ p₂ : Point ℝ) (m : ℝ) :=
{ l | ∃ (b : ℝ), l = λ p: Point ℝ, p.x * m + b = p.y}

def product_distances_maximized (A B C : Point ℝ) (m : ℝ) : Prop :=
 ∀ l ∈ line_through C, 
  let dA := (l A).abst 
  ∧ let dB := (l B).abst 
in max (dA * dB)

theorem unique_maximized_distance_line (A B C : Point ℝ) (h_distinct: A ≠ B ∧ B ≠ C ∧ C ≠ A):
  ∃! l ∈ line_through C, product_distances_maximized A B C (l.m ab) :=
begin
  sorry
end

end unique_maximized_distance_line_l674_674100


namespace change_received_l674_674650

def laptop_price : ℕ := 600
def smartphone_price : ℕ := 400
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def initial_amount : ℕ := 3000

theorem change_received : (initial_amount -
  ((laptop_price * num_laptops) + (smartphone_price * num_smartphones))) = 200 :=
by
  calc
    initial_amount - ((laptop_price * num_laptops) + (smartphone_price * num_smartphones))
        = 3000 - ((600 * 2) + (400 * 4)) : by simp [initial_amount, laptop_price, num_laptops, smartphone_price, num_smartphones]
    ... = 3000 - (1200 + 1600) : by norm_num
    ... = 3000 - 2800 : by norm_num
    ... = 200 : by norm_num

end change_received_l674_674650


namespace words_per_page_l674_674627

theorem words_per_page (p : ℕ) (h1 : p ≤ 150) (h2 : 120 * p ≡ 172 [MOD 221]) : p = 114 := by
  sorry

end words_per_page_l674_674627


namespace product_of_triangle_areas_not_end_in_1988_l674_674213

theorem product_of_triangle_areas_not_end_in_1988
  (a b c d : ℕ)
  (h1 : a * c = b * d)
  (hp : (a * b * c * d) = (a * c)^2)
  : ¬(∃ k : ℕ, (a * b * c * d) = 10000 * k + 1988) :=
sorry

end product_of_triangle_areas_not_end_in_1988_l674_674213


namespace total_saltwater_animals_l674_674169

theorem total_saltwater_animals 
    (a₁ : Nat) (n₁ : Nat) (n₁_eq : a₁ = 8) (a₁_each : Nat) (a₁_each_eq : a₁_each = 128)
    (a₂ : Nat) (n₂ : Nat) (n₂_eq : a₂ = 5) (a₂_each : Nat) (a₂_each_eq : a₂_each = 85)
    (a₃ : Nat) (n₃ : Nat) (n₃_eq : a₃ = 2) (a₃_each : Nat) (a₃_each_eq : a₃_each = 155) : 
    n₁ * a₁_each + n₂ * a₂_each + n₃ * a₃_each = 1759 :=
by 
    have h1 : n₁ = 8 := n₁_eq
    have h2 : a₁_each = 128 := a₁_each_eq
    have h3 : n₂ = 5 := n₂_eq
    have h4 : a₂_each = 85 := a₂_each_eq
    have h5 : n₃ = 2 := n₃_eq
    have h6 : a₃_each = 155 := a₃_each_eq
    calc
        8 * 128 + 5 * 85 + 2 * 155
        _ = 1024 + 425 + 310 : by rw [h1, h2, h3, h4, h5, h6]
        _ = 1759 : by norm_num


end total_saltwater_animals_l674_674169


namespace distance_travelled_l674_674413

variables (D : ℝ) (h_eq : D / 10 = (D + 20) / 14)

theorem distance_travelled : D = 50 :=
by sorry

end distance_travelled_l674_674413


namespace cyclic_quadrilateral_of_incircle_touch_l674_674065

open EuclideanGeometry

theorem cyclic_quadrilateral_of_incircle_touch
  {A B C D E F X Y Z : Point} 
  (h1 : InCircleTouches \( \triangle A B C \) BC D) 
  (h2 : InCircleTouches \( \triangle A B C \) CA E) 
  (h3 : InCircleTouches \( \triangle A B C \) AB F) 
  (h4 : Interior \( \triangle A B C \) X)
  (h5 : InCircleTouches \( \triangle X B C \) BC D)
  (h6 : InCircleTouches \( \triangle X B C \) CX Y)
  (h7 : InCircleTouches \( \triangle X B C \) XB Z) : CyclicQuadrilateral E F Z Y :=
by
  sorry

end cyclic_quadrilateral_of_incircle_touch_l674_674065


namespace nth_equation_l674_674490

theorem nth_equation (n : ℕ) :
  1 - (∑ k in finset.range (n) (k.odd) (k.even + 1)) = (∑ k in finset.range (n+1) (k)) :=
sorry

end nth_equation_l674_674490


namespace compound_not_determined_by_oxygen_percentage_l674_674038

theorem compound_not_determined_by_oxygen_percentage (percentage_oxygen: ℝ) (h: percentage_oxygen = 57.14) : 
  ¬ ∃ compound, mass_percentage_of_element compound "O" = percentage_oxygen :=
by sorry

end compound_not_determined_by_oxygen_percentage_l674_674038


namespace find_inradius_l674_674558

-- Define variables and constants
variables (P A : ℝ)
variables (s r : ℝ)

-- Given conditions as definitions
def perimeter_triangle : Prop := P = 36
def area_triangle : Prop := A = 45

-- Semi-perimeter definition
def semi_perimeter : Prop := s = P / 2

-- Inradius and area relationship
def inradius_area_relation : Prop := A = r * s

-- Theorem statement
theorem find_inradius (hP : perimeter_triangle P) (hA : area_triangle A) (hs : semi_perimeter P s) (har : inradius_area_relation A r s) :
  r = 2.5 :=
by
  sorry

end find_inradius_l674_674558


namespace total_score_is_correct_l674_674492

def dad_points : ℕ := 7
def olaf_points : ℕ := 3 * dad_points
def total_points : ℕ := dad_points + olaf_points

theorem total_score_is_correct : total_points = 28 := by
  sorry

end total_score_is_correct_l674_674492


namespace probability_of_non_adjacent_zeros_l674_674388

-- Define the total number of arrangements of 3 ones and 2 zeros
def totalArrangements : ℕ := Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

-- Define the number of arrangements where the 2 zeros are together
def adjacentZerosArrangements : ℕ := 2 * Nat.factorial 4 / (Nat.factorial 3 * Nat.factorial 1)

-- Calculate the desired probability
def nonAdjacentZerosProbability : ℚ := 
  1 - (adjacentZerosArrangements.toRat / totalArrangements.toRat)

theorem probability_of_non_adjacent_zeros :
  nonAdjacentZerosProbability = 3/5 :=
sorry

end probability_of_non_adjacent_zeros_l674_674388


namespace find_other_leg_l674_674432

structure RightTriangle where
  a : ℝ 
  c : ℝ 
  hypotenuse_eq : c^2 = a^2 + b^2
  leg_eq_1 : a = 1
  hypotenuse_eq_sqrt5 : c = Real.sqrt 5

theorem find_other_leg (rt : RightTriangle) : ∃ b : ℝ, b = 2 :=
by
  have h1 : rt.a = 1 := rt.leg_eq_1
  have h2 : rt.c = Real.sqrt 5 := rt.hypotenuse_eq_sqrt5
  have hyp_eq : rt.c^2 = rt.a^2 + b^2 := rt.hypotenuse_eq
  -- Substitute values and prove b = 2
  sorry

end find_other_leg_l674_674432


namespace find_phase_shift_l674_674298

def phase_shift_of_sine_function (b c : ℝ) : ℝ := - c / b

theorem find_phase_shift :
  phase_shift_of_sine_function 3 (Real.pi / 4) = - Real.pi / 12 :=
by
  sorry

end find_phase_shift_l674_674298


namespace number_of_sides_of_polygon_l674_674029

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)
noncomputable def sum_known_angles : ℝ := 3780

theorem number_of_sides_of_polygon
  (n : ℕ)
  (h1 : sum_known_angles + missing_angle = sum_of_interior_angles n)
  (h2 : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a = 3 * c ∧ b = 3 * c ∧ a + b + c ≤ sum_known_angles) :
  n = 23 :=
sorry

end number_of_sides_of_polygon_l674_674029


namespace max_value_a_n_div_n_l674_674759

def a : ℕ → ℕ
| 1 := 1
| 2 := 3
| (n+1) => sorry -- the recurrence relation is complex to define explicitly here

theorem max_value_a_n_div_n :
  (∀ n : ℕ, n ≥ 2 → 2 * n * a n = (n - 1) * a (n - 1) + (n + 1) * a (n + 1)) →
  (∀ n : ℕ, n > 0 → a n / n ≤ a 2 / 2) :=
sorry

end max_value_a_n_div_n_l674_674759


namespace hoot_difference_l674_674103

def owl_hoot_rate : ℕ := 5
def heard_hoots_per_min : ℕ := 20
def owls_count : ℕ := 3

theorem hoot_difference :
  heard_hoots_per_min - (owls_count * owl_hoot_rate) = 5 := by
  sorry

end hoot_difference_l674_674103


namespace acute_angles_l674_674102

noncomputable theory

variables {A B C K L P : Type*} [normed_group A] [normed_group B] [normed_group C] 
variables [normed_group K] [normed_group L] [normed_group P]
variables {α : Type*} [normed_group α] [normed_group (Type* → α)] 

def right_triangle (A B C : A) : Prop :=
  angle A C B = 90

def point_on_hypotenuse (B C K : A) : Prop :=
  K ∈ line_segment B C

def equal_segments (A B K : A) : Prop :=
  dist A B = dist A K

def angle_bisector (C : A) (L : A → A → B) : Prop := sorry  -- Definition placeholder

def midpoint_segment (L P : A → A → B) : Prop := sorry  -- Definition placeholder

def acute_angle (α : angle α) : Prop := α < 90

theorem acute_angles {A B C K L P : Type*} [right_triangle A B C] 
  [point_on_hypotenuse B C K] [equal_segments A B K] [angle_bisector C L] 
  [midpoint_segment L P] : acute_angle (angle A B C) ∧ 
  acute_angle (angle A C B) := by
  sorry

end acute_angles_l674_674102


namespace area_of_region_l674_674576

theorem area_of_region : ∀ (x y : ℝ), x^2 + y^2 - 8*x + 10*y = -25 → 16 * Real.pi :=
begin
  sorry
end

end area_of_region_l674_674576


namespace seashells_left_l674_674487

theorem seashells_left (initial_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) :
  initial_seashells = 75 → given_seashells = 18 → remaining_seashells = initial_seashells - given_seashells → remaining_seashells = 57 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end seashells_left_l674_674487


namespace initial_walking_speed_l674_674640

theorem initial_walking_speed
  (t : ℝ) -- Time in minutes for bus to reach the bus stand from when the person starts walking
  (h₁ : 5 = 5 * ((t - 5) / 60)) -- When walking at 5 km/h, person reaches 5 minutes early
  (h₂ : 5 = v * ((t + 10) / 60)) -- At initial speed v, person misses the bus by 10 minutes
  : v = 4 := 
by
  sorry

end initial_walking_speed_l674_674640


namespace change_received_l674_674649

def laptop_price : ℕ := 600
def smartphone_price : ℕ := 400
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def initial_amount : ℕ := 3000

theorem change_received : (initial_amount -
  ((laptop_price * num_laptops) + (smartphone_price * num_smartphones))) = 200 :=
by
  calc
    initial_amount - ((laptop_price * num_laptops) + (smartphone_price * num_smartphones))
        = 3000 - ((600 * 2) + (400 * 4)) : by simp [initial_amount, laptop_price, num_laptops, smartphone_price, num_smartphones]
    ... = 3000 - (1200 + 1600) : by norm_num
    ... = 3000 - 2800 : by norm_num
    ... = 200 : by norm_num

end change_received_l674_674649


namespace inequality_transformation_l674_674401

theorem inequality_transformation (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end inequality_transformation_l674_674401


namespace sum_roots_l674_674769

variable (α β : ℝ)

def is_root_log (x : ℝ) : Prop := log 3 x + x - 3 = 0
def is_root_exp (x : ℝ) : Prop := 3 ^ x + x - 3 = 0

theorem sum_roots (hα : is_root_log α) (hβ : is_root_exp β) : α + β = 3 := 
sorry

end sum_roots_l674_674769


namespace find_λ_l674_674334

noncomputable def λ_value 
  (A : ℝ × ℝ := (1, 0)) 
  (B : ℝ × ℝ := (1, Real.sqrt 3)) 
  (C : ℝ × ℝ) 
  (angle_AOC : ℝ := 150)
  (OC_eq : C = λ' => (-4 : ℝ) * A + λ' • B) : ℝ :=
  λ'  

theorem find_λ (
  A B : ℝ × ℝ
  C : ℝ × ℝ 
  angle_AOC : ℝ
  OC_eq' : ∃ λ', C = λ_value A B C angle_AOC OC_eq 
  ) (hA: A = (1, 0)) 
   (hB: B = (1, Real.sqrt 3)) 
   (hangle: angle_AOC = 150) 
   (hOC: ∃ λ', C = (-4 : ℝ) * A + λ' • B) :
  λ' = 1 :=
sorry

end find_λ_l674_674334


namespace distance_travelled_l674_674414

variables (D : ℝ) (h_eq : D / 10 = (D + 20) / 14)

theorem distance_travelled : D = 50 :=
by sorry

end distance_travelled_l674_674414


namespace probability_zeros_not_adjacent_is_0_6_l674_674393

-- Define the total number of arrangements of 5 elements where we have 3 ones and 2 zeros
def total_arrangements : Nat := 5.choose 2

-- Define the number of arrangements where 2 zeros are adjacent
def adjacent_zeros_arrangements : Nat := 4.choose 1 * 2

-- Define the probability that the 2 zeros are not adjacent
def probability_not_adjacent : Rat := (total_arrangements - adjacent_zeros_arrangements) / total_arrangements

-- Prove the desired probability is 0.6
theorem probability_zeros_not_adjacent_is_0_6 : probability_not_adjacent = 3 / 5 := by
  sorry

end probability_zeros_not_adjacent_is_0_6_l674_674393


namespace largest_subset_no_diff_5_or_9_l674_674079
open Nat

/-- 
Let T be a subset of {1, 2, ..., 2146} such that no two members of T differ by 5 or 9. 
We need to show that the largest number of elements T can have is 660.
-/
theorem largest_subset_no_diff_5_or_9 :
  ∃ T : Finset ℕ, T ⊆ Finset.range 2147 ∧ (∀ (x ∈ T) (y ∈ T), x ≠ y → (x - y).nat_abs ≠ 5 ∧ (x - y).nat_abs ≠ 9) ∧ T.card = 660 := by
  sorry

end largest_subset_no_diff_5_or_9_l674_674079


namespace max_obtuse_angles_in_quadrilateral_l674_674817

theorem max_obtuse_angles_in_quadrilateral (a b c d : ℝ) 
  (h1 : a + b + c + d = 360) 
  (h2 : a > 90 → b > 90 → c > 90 → d > 90 → False) :
  (a > 90) + (b > 90) + (c > 90) + (d > 90) ≤ 3 :=
by {
  sorry
}

end max_obtuse_angles_in_quadrilateral_l674_674817


namespace problem1_problem2_l674_674806

-- Problem 1
theorem problem1 :
  A = {x : ℝ | -1 ≤ x ∧ x ≤ 1} ∪ {x : ℝ | 4 ≤ x ∧ x ≤ 5} :=
by
  let A := {x : ℝ | -1 ≤ x ∧ x ≤ 5}
  let B := {x : ℝ | x ≤ 1 ∨ x ≥ 4}
  have h : A ∩ B = {x : ℝ | -1 ≤ x ∧ x ≤ 1} ∪ {x : ℝ | 4 ≤ x ∧ x ≤ 5} := sorry
  exact h

-- Problem 2
theorem problem2 :
  (A = {x : ℝ | 2 - a ≤ x ∧ x ≤ 2 + a}) →
  (B = {x : ℝ | x ≤ 1 ∨ x ≥ 4}) →
  (a > 0) →
  (A ∩ B = ∅) →
  (0 < a ∧ a < 1) :=
by
  assume hA : A = {x : ℝ | 2 - a ≤ x ∧ x ≤ 2 + a}
  assume hB : B = {x : ℝ | x ≤ 1 ∨ x ≥ 4}
  assume ha : a > 0
  assume hIntersect : A ∩ B = ∅
  have hRange : 0 < a ∧ a < 1 := sorry
  exact hRange

end problem1_problem2_l674_674806


namespace no_integer_solutions_19x2_minus_76y2_eq_1976_l674_674507

theorem no_integer_solutions_19x2_minus_76y2_eq_1976 :
  ∀ x y : ℤ, 19 * x^2 - 76 * y^2 ≠ 1976 :=
by sorry

end no_integer_solutions_19x2_minus_76y2_eq_1976_l674_674507


namespace probability_yellow_face_l674_674530

-- Define the total number of faces and the number of yellow faces on the die
def total_faces := 12
def yellow_faces := 4

-- Define the probability calculation
def probability_of_yellow := yellow_faces / total_faces

-- State the theorem
theorem probability_yellow_face : probability_of_yellow = 1 / 3 := by
  sorry

end probability_yellow_face_l674_674530


namespace weight_of_replaced_person_l674_674126

/-- The weight of the person who was replaced is calculated given the average weight increase for 8 persons and the weight of the new person. --/
theorem weight_of_replaced_person
  (avg_weight_increase : ℝ)
  (num_persons : ℕ)
  (weight_new_person : ℝ) :
  avg_weight_increase = 3 → 
  num_persons = 8 →
  weight_new_person = 89 →
  weight_new_person - avg_weight_increase * num_persons = 65 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end weight_of_replaced_person_l674_674126


namespace find_equation_of_line_l674_674766

noncomputable def equation_of_line (P : ℝ × ℝ) (PA PB : ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧
  let A := (2 - 1/k, 0) in
  let B := (0, 1 - 2*k) in
  ((2 - (2 - 1/k))^2 + 1^2)^0.5 * ((2 - 0)^2 + ((1 - 2*k) - 1)^2)^0.5 = 4 ∧
  (A.1 - 2, A.2 - 1) × (B.1 - 2, B.2 - 1) = l

theorem find_equation_of_line : equation_of_line (2, 1) 4 :=
  sorry

end find_equation_of_line_l674_674766


namespace inequality_holds_l674_674735

theorem inequality_holds (a : ℝ) (h : a ≤ -2) : 
  ∀ x : ℝ, sin x ^ 2 + a * cos x + a ^ 2 ≥ 1 + cos x :=
sorry

end inequality_holds_l674_674735


namespace find_omega_phi_l674_674781

-- Definitions based on the conditions given in step a)
def P : ℝ × ℝ := (π / 12, 1)
def Q : ℝ × ℝ := (5 * π / 12, -1)
def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
def omega_pos (ω : ℝ) : Prop := ω > 0
def phi_bound (φ : ℝ) : Prop := |φ| < π / 2

noncomputable def omega_phi (ω φ : ℝ) : ℝ := ω * φ

-- Proof statement
theorem find_omega_phi (ω φ : ℝ)
  (h1 : P.2 = f ω φ P.1)
  (h2 : Q.2 = f ω φ Q.1)
  (hω : omega_pos ω)
  (hφ : phi_bound φ) :
  omega_phi ω φ = 3 * π / 4 :=
sorry

end find_omega_phi_l674_674781


namespace min_phi_symmetry_l674_674516

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

theorem min_phi_symmetry (phi : ℝ) (h_phi : phi > 0)
  (h_sym : ∀ x, Real.sin (x + Real.pi / 6 + phi) = -Real.sin (x + Real.pi / 6 + 2 * Real.pi - phi)) :
  phi = Real.pi / 2 :=
by 
  sorry

end min_phi_symmetry_l674_674516


namespace trigonometric_identity_solution_l674_674183

theorem trigonometric_identity_solution (x : ℝ) : 
  (sin (π/4 + 5 * x) * cos (π/4 + 2 * x) = sin (π/4 + x) * sin (π/4 - 6 * x)) ↔ 
  (∃ n : ℤ, x = (n * π) / 4) :=
by
  sorry

end trigonometric_identity_solution_l674_674183


namespace equation_has_unique_integer_solution_l674_674309

theorem equation_has_unique_integer_solution:
  ∀ m n : ℤ, (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n → m = 0 ∧ n = 0 := by
  intro m n
  -- The proof is omitted
  sorry

end equation_has_unique_integer_solution_l674_674309


namespace chemical_x_percentage_l674_674407

variables (x y : Type) [division_ring x] [has_div x] (mixtureL totalL : x)
variables (initialPercent chemicalXAdded newPercent : x)

def initial_liters_of_chemical_x (mixtureL : x) (initialPercent : x) := initialPercent * mixtureL
def new_liters_of_chemical_x (initialLiters chemicalXAdded : x) := initialLiters + chemicalXAdded
def new_total_mixture (mixtureL chemicalXAdded : x) := mixtureL + chemicalXAdded
def new_percent_chemical_x (newLiters newTotal : x) := (newLiters / newTotal) * 100

theorem chemical_x_percentage (h1 : mixtureL = 80) (h2 : initialPercent = 0.20) 
  (h3 : chemicalXAdded = 20) (h4 : newPercent = 36) :
  (new_percent_chemical_x (new_liters_of_chemical_x 
  (initial_liters_of_chemical_x mixtureL initialPercent) chemicalXAdded) 
  (new_total_mixture mixtureL chemicalXAdded)) = newPercent :=
by sorry

end chemical_x_percentage_l674_674407


namespace positive_integer_M_l674_674965

theorem positive_integer_M (M : ℕ) (h : 14^2 * 35^2 = 70^2 * M^2) : M = 7 :=
sorry

end positive_integer_M_l674_674965


namespace number_of_valid_three_digit_numbers_l674_674022

def valid_three_digit_numbers : Nat :=
  -- Proving this will be the task: showing that there are precisely 24 such numbers
  24

theorem number_of_valid_three_digit_numbers : valid_three_digit_numbers = 24 :=
by
  -- Proof would go here.
  sorry

end number_of_valid_three_digit_numbers_l674_674022


namespace find_number_l674_674944

theorem find_number (x : ℝ) (h : x / 5 + 10 = 21) : x = 55 :=
by
  sorry

end find_number_l674_674944


namespace cos_shift_l674_674572

theorem cos_shift (x : ℝ) : 
  cos (x / 2 - π / 3) = cos ((x - 2 * π / 3) / 2 - π / 3 + π / 3) :=
by
  sorry

end cos_shift_l674_674572


namespace number_drawn_from_third_group_l674_674239

theorem number_drawn_from_third_group :
  ∀ (total_bags groups_per_bag first_sampled_gap first_group_number: ℕ),
    total_bags = 300 →
    groups_per_bag = 20 →
    first_sampled_gap = 15 →
    first_group_number = 6 →
    let second_group = first_group_number + first_sampled_gap in
    let third_group = second_group + first_sampled_gap in
    third_group = 36 :=
begin
  intros,
  sorry
end

end number_drawn_from_third_group_l674_674239


namespace part1_optimal_strategy_part2_optimal_strategy_l674_674987

noncomputable def R (x1 x2 : ℝ) : ℝ := -2 * x1^2 - x2^2 + 13 * x1 + 11 * x2 - 28

theorem part1_optimal_strategy :
  ∃ x1 x2 : ℝ, x1 + x2 = 5 ∧ x1 = 2 ∧ x2 = 3 ∧
    ∀ y1 y2, y1 + y2 = 5 → (R y1 y2 - (y1 + y2) ≤ R x1 x2 - (x1 + x2)) := 
by
  sorry

theorem part2_optimal_strategy :
  ∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = 5 ∧
    ∀ y1 y2, (R y1 y2 - (y1 + y2) ≤ R x1 x2 - (x1 + x2)) := 
by
  sorry

end part1_optimal_strategy_part2_optimal_strategy_l674_674987


namespace rectangle_ratio_l674_674744

-- Definitions for the problem
def inner_square_side : ℝ := s
def shorter_side (y : ℝ) : Prop := 2 * y = s
def longer_side (x : ℝ) : Prop := x = s
def area_condition (inner_side outer_area : ℝ) : Prop := 
  outer_area = 9 * inner_side^2

-- The main theorem
theorem rectangle_ratio (s x y : ℝ) 
  (h_shorter : shorter_side y) 
  (h_longer : longer_side x) 
  (h_area : area_condition s (3 * s)) : 
  x / y = 2 := by
  sorry

end rectangle_ratio_l674_674744


namespace fill_tank_time_l674_674503

theorem fill_tank_time :
  ∀ (capacity rateA rateB rateC fill_timeA fill_timeB drain_time : ℕ),
    capacity = 2000 →
    rateA = 200 →
    rateB = 50 →
    rateC = 25 →
    fill_timeA = 1 →
    fill_timeB = 2 →
    drain_time = 2 →
    let net_fill_per_cycle := (rateA * fill_timeA) + (rateB * fill_timeB) - (rateC * drain_time),
        cycles := capacity / net_fill_per_cycle,
        time_per_cycle := fill_timeA + fill_timeB + drain_time,
        total_time := cycles * time_per_cycle
    in total_time = 40 := 
by
  intros capacity rateA rateB rateC fill_timeA fill_timeB drain_time h_capacity h_rateA h_rateB h_rateC h_fill_timeA h_fill_timeB h_drain_time,
  let net_fill_per_cycle := (rateA * fill_timeA) + (rateB * fill_timeB) - (rateC * drain_time),
  let cycles := capacity / net_fill_per_cycle,
  let time_per_cycle := fill_timeA + fill_timeB + drain_time,
  let total_time := cycles * time_per_cycle,
  sorry

end fill_tank_time_l674_674503


namespace factorial_inequality_l674_674887

theorem factorial_inequality (n : ℕ) :
  (fact (1998 * n)) ≤ (∏ k in Ico 1 3996, if k % 2 = 1 then (k * n + 1) / 2 else 1) ^ n :=
by sorry

end factorial_inequality_l674_674887


namespace correct_judgment_l674_674364

-- Define the propositions p and q
def p : Prop := ∀ x : ℝ, 2 * x^2 + 2 * x + 1 / 2 < 0
def q : Prop := ∃ x_0 : ℝ, sin x_0 - cos x_0 = √2

-- The theorem to be proved
theorem correct_judgment : ¬q := sorry

end correct_judgment_l674_674364


namespace volume_pyramid_ABGF_l674_674985

noncomputable def volume_of_pyramid (side_length : ℝ) : ℝ :=
  let base_area := (1 / 2) * side_length * side_length
  let height := side_length
  (1 / 3) * base_area * height

theorem volume_pyramid_ABGF
  (ABCDEFGH_is_cube : Π (A B C D E F G H : ℝ × ℝ × ℝ), 
    (∃ s, s = 2 ∧ 
     (B = (A.1 + s, A.2, A.3)) ∧ 
     (G = (C.1 + s, C.2, C.3 + s)) ∧ 
     .. -- other conditions to describe the cube structure
    )) :
  volume_of_pyramid 2 = 4 / 3 :=
by sorry

end volume_pyramid_ABGF_l674_674985


namespace width_of_wall_is_two_l674_674993

noncomputable def volume_of_brick : ℝ := 20 * 10 * 7.5 / 10^6 -- Volume in cubic meters
def number_of_bricks : ℕ := 27000
noncomputable def volume_of_wall (width : ℝ) : ℝ := 27 * width * 0.75

theorem width_of_wall_is_two :
  ∃ (W : ℝ), volume_of_wall W = number_of_bricks * volume_of_brick ∧ W = 2 :=
by
  sorry

end width_of_wall_is_two_l674_674993


namespace _l674_674726

-- Define the encryption function
def encrypt_digit (d : ℕ) : ℕ := (d^3 + 1) % 10

-- Define a function for concatenating digits into a number
def digits_to_number (digits : List ℕ) : ℕ := digits.foldl (λ acc d, acc * 10 + d) 0

-- Define the function to encrypt a multi-digit number
def encrypt_number (n : ℕ) : ℕ :=
  let digits := n.digits
  let encrypted_digits := digits.map encrypt_digit
  digits_to_number encrypted_digits

-- The main theorem statement with given condition and correct answer
example : encrypt_number 2568 = 9673 := 
by
  sorry

end _l674_674726


namespace smallest_initial_value_of_positive_sequence_l674_674082

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
∀ n > 1, a n = 5 * a (n - 1) - 2 * n

theorem smallest_initial_value_of_positive_sequence :
  ∃ a1 : ℝ, a1 = 7 / 12 ∧ ∀ (a : ℕ → ℝ), sequence a → (a 1 = a1 → ∀ n, 0 < a n) :=
begin
  sorry
end

end smallest_initial_value_of_positive_sequence_l674_674082


namespace value_of_sine_neg_10pi_over_3_l674_674936

theorem value_of_sine_neg_10pi_over_3 : Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end value_of_sine_neg_10pi_over_3_l674_674936


namespace remainder_base15_div7_is_zero_l674_674141

theorem remainder_base15_div7_is_zero :
  let num := 14 + 13 * 15 + 12 * 15^2 + 11 * 15^3 + 10 * 15^4 + 9 * 15^5 + 8 * 15^6 + 7 * 15^7 + 6 * 15^8 +
             5 * 15^9 + 4 * 15^10 + 3 * 15^11 + 2 * 15^12 + 1 * 15^13 in
  num % 7 = 0 :=
by
  sorry

end remainder_base15_div7_is_zero_l674_674141


namespace reliable_deviation_and_speed_l674_674252

theorem reliable_deviation_and_speed
  (allowable_error_probability : ℝ)
  (allowable_error_probability_pos : 0 < allowable_error_probability)
  (allowable_error_probability_lt_one : allowable_error_probability < 1)
  (n : ℕ)
  (n_pos : 0 < n)
  :
  allowable_error_probability = 0.05 → n = 1000 →
  let deviation := 5.429 * Real.sqrt n in
  let largest_even_deviation := 170 in
  let reliable_speed := largest_even_deviation / n in
  deviation ≤ largest_even_deviation ∧ reliable_speed = 0.17 :=
by 
  sorry

end reliable_deviation_and_speed_l674_674252


namespace project_completion_time_l674_674207

theorem project_completion_time 
    (x : ℕ) 
    (h : 8 * (1 / 20 + 1 / x) + 10 * (1 / x) = 1) : 
    x = 30 :=
by {
   have h1 : 8 / 20 + 8 / x + 10 / x = 1 := by rw [mul_div, mul_div]; exact h,
   have h2 : 18 / x + 2 / 5 = 1 := by ring_nf,
   have h3 : 18 / x = 3 / 5 := by linarith,
   have h4 : 90 = 3 * x := by { field_simp at h3, exact_mod_cast h3, },
   exact_mod_cast eq_div_iff_mul (by norm_num : 3 ≠ 0).mpr h4
}

end project_completion_time_l674_674207


namespace find_second_to_last_value_l674_674613

-- Definitions for the problem
def a (n : ℕ) : ℕ := sorry -- Will be defined as appropriate in the proof

-- Problem conditions
axiom a1 : a 1 = 19999
axiom a2 : a 201 = 19999
axiom seq_relation : ∀ n, 2 ≤ n ∧ n ≤ 200 → a n = (a (n-1) + a (n+1)) / 2 - t

-- The target we need to prove
theorem find_second_to_last_value : a 200 = 19800 :=
sorry

end find_second_to_last_value_l674_674613


namespace average_rst_l674_674024

variable (r s t : ℝ)

theorem average_rst
  (h : (4 / 3) * (r + s + t) = 12) :
  (r + s + t) / 3 = 3 :=
sorry

end average_rst_l674_674024


namespace minimum_questionnaires_l674_674977

theorem minimum_questionnaires (responses_needed : ℕ) (response_rate : ℝ)
  (h1 : responses_needed = 300) (h2 : response_rate = 0.70) :
  ∃ (n : ℕ), n = Nat.ceil (responses_needed / response_rate) ∧ n = 429 :=
by
  sorry

end minimum_questionnaires_l674_674977


namespace angle_BE_A1_and_AE_B1_l674_674848

-- Definitions for the problem conditions
variables (A B C A1 B1 C1 D E J : Point)
variable (ABC : Triangle A B C)
variable (on_semicircle : CenterCircle J (Semicircle (BC A1 ABC)))
variable (extensions : Touches J (Extensions A B1 C) (Extensions A C1 B))
variable (right_angle_intersection : IntersectsRightAngle(Elline A1 B1) (Segment AB D))
variable (projection_E_C1 : Projection (Line DJ) C1 E)

-- Statement to be proved
theorem angle_BE_A1_and_AE_B1 :
  ∠BEA_1 = 90 ∧ ∠AEB_1 = 90 :=
sorry

end angle_BE_A1_and_AE_B1_l674_674848


namespace seq_proof_l674_674777
noncomputable def seq1_arithmetic (a1 a2 : ℝ) : Prop :=
  ∃ d : ℝ, a1 = -2 + d ∧ a2 = a1 + d ∧ -8 = a2 + d

noncomputable def seq2_geometric (b1 b2 b3 : ℝ) : Prop :=
  ∃ r : ℝ, b1 = -2 * r ∧ b2 = b1 * r ∧ b3 = b2 * r ∧ -8 = b3 * r

theorem seq_proof (a1 a2 b1 b2 b3: ℝ) (h1 : seq1_arithmetic a1 a2) (h2 : seq2_geometric b1 b2 b3) :
  (a2 - a1) / b2 = 1 / 2 :=
sorry

end seq_proof_l674_674777


namespace books_borrowed_by_lunchtime_l674_674812

theorem books_borrowed_by_lunchtime (x : ℕ) :
  (∀ x : ℕ, 100 - x + 40 - 30 = 60) → (x = 50) :=
by
  intro h
  have eqn := h x
  sorry

end books_borrowed_by_lunchtime_l674_674812


namespace bisection_algorithm_structures_l674_674972

theorem bisection_algorithm_structures :
  (∀ x : ℝ, x^2 - 10 = 0 → ∃ y : ℝ, y = approximate_root x (bisection_method x))
  ∧ (∀ algorithm, involves_sequential_structure algorithm)
  ∧ (∀ loop, includes_conditional_structure loop)
  → (uses_sequential_structure (bisection_method (λ x, x^2 - 10 = 0))
     ∧ uses_conditional_structure (bisection_method (λ x, x^2 - 10 = 0))
     ∧ uses_loop_structure (bisection_method (λ x, x^2 - 10 = 0))) :=
  sorry

end bisection_algorithm_structures_l674_674972


namespace even_monotonic_implies_greater_abs_val_l674_674764

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def is_monotonic_increasing_on_neg_reals (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → y < 0 → f x ≤ f y

theorem even_monotonic_implies_greater_abs_val
  (f : ℝ → ℝ)
  (hf_even : is_even_function f)
  (hf_mono_inc_neg : is_monotonic_increasing_on_neg_reals f)
  (x1 x2 : ℝ)
  (hx1 : x1 < 0)
  (hx2 : x2 > 0)
  (habs : |x1| < |x2|) :
  f(-x1) > f(-x2) :=
sorry

end even_monotonic_implies_greater_abs_val_l674_674764


namespace cubics_product_l674_674694

theorem cubics_product :
  (∏ n in [3, 4, 5, 6, 7], (n^3 - 1) / (n^3 + 1)) = (57 / 168) := by
  sorry

end cubics_product_l674_674694


namespace general_formula_for_a_n_sum_of_first_n_terms_of_b_l674_674091

section

variables {a b : ℕ → ℚ} {n : ℕ}

-- Conditions
def S (n : ℕ) : ℚ := n * (n + 1) / 2
def c (n : ℕ) : ℚ := S n / n
def a_seq (n : ℕ) : ℕ := n

-- Condition and given data
axiom a_1 : a_seq 1 = 1
axiom sum_c : c 2 + c 3 + c 4 = 6

-- Definition of b_n
def b (n : ℕ) : ℚ := (a_seq (n + 1) / a_seq (n + 2)) + (a_seq (n + 2) / a_seq (n + 1)) - 2

-- Result to prove for (1)
theorem general_formula_for_a_n : ∀ n, a_seq n = n :=
sorry

-- Result to prove for (2)
theorem sum_of_first_n_terms_of_b (n : ℕ) : (∑ k in finset.range n, b k) = (1/2) - (1 / (n + 2)) :=
sorry

end

end general_formula_for_a_n_sum_of_first_n_terms_of_b_l674_674091


namespace probability_non_adjacent_zeros_l674_674385

theorem probability_non_adjacent_zeros (total_ones total_zeros : ℕ) (h₁ : total_ones = 3) (h₂ : total_zeros = 2) : 
  (total_zeros != 0 ∧ total_ones != 0 ∧ total_zeros + total_ones = 5) → 
  (prob_non_adjacent (total_ones + total_zeros) total_zeros = 0.6) :=
by
  sorry

def prob_non_adjacent (total num_zeros: ℕ) : ℚ :=
  let total_arrangements := (Nat.factorial total) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros)))
  let adjacent_arrangements := (Nat.factorial (total - num_zeros + 1)) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros - 1)))
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements / total_arrangements

end probability_non_adjacent_zeros_l674_674385


namespace no_unique_y_exists_l674_674830

theorem no_unique_y_exists (x y : ℕ) (k m : ℤ) 
  (h1 : x % 82 = 5)
  (h2 : (x + 7) % y = 12) :
  ¬ ∃! y, (∃ k m : ℤ, x = 82 * k + 5 ∧ (x + 7) = y * m + 12) :=
by
  sorry

end no_unique_y_exists_l674_674830


namespace roller_coaster_cars_l674_674941

theorem roller_coaster_cars
  (people : ℕ)
  (runs : ℕ)
  (seats_per_car : ℕ)
  (people_per_run : ℕ)
  (h1 : people = 84)
  (h2 : runs = 6)
  (h3 : seats_per_car = 2)
  (h4 : people_per_run = people / runs) :
  (people_per_run / seats_per_car) = 7 :=
by
  sorry

end roller_coaster_cars_l674_674941


namespace part1_part2_l674_674007

-- Definitions for parts (a) and (b)
def f (x : ℝ) : ℝ := Real.log x
def g (a b x : ℝ) : ℝ := (1/2) * a * x + b
def P := (1 : ℝ, f 1)

-- Part 1: Common tangent line at point P(1, f(1))
theorem part1 (a b : ℝ) (h_tangent : (differentiable ℝ f at 1 'and' differentiable ℝ (λ x, g a b x) at 1) ∧ deriv f 1 = deriv (λ x, g a b x) 1 ∧ f 1 = g a b 1): g a b = λ x, x - 1 :=
sorry

-- Definitions for part (2)
def ϕ (m x : ℝ) : ℝ := (m * (x - 1)) / (x + 1) - f x

-- Part 2: Proving ϕ(x) is a decreasing function on [1, +∞) implies range of m
theorem part2 (m : ℝ) (h_decreasing : ∀ x : ℝ, x ≥ 1 → derivative (ϕ m) x ≤ 0) : m ≤ 2 :=
sorry

end part1_part2_l674_674007


namespace zeros_not_adjacent_probability_l674_674374

def total_arrangements : ℕ := Nat.factorial 5

def adjacent_arrangements : ℕ := 2 * Nat.factorial 4

def probability_not_adjacent : ℚ := 
  1 - (adjacent_arrangements / total_arrangements)

theorem zeros_not_adjacent_probability :
  probability_not_adjacent = 0.6 := 
by 
  sorry

end zeros_not_adjacent_probability_l674_674374


namespace one_of_a_b_c_is_one_l674_674342

theorem one_of_a_b_c_is_one (a b c : ℝ) (h1 : a * b * c = 1) (h2 : a + b + c = (1 / a) + (1 / b) + (1 / c)) :
  a = 1 ∨ b = 1 ∨ c = 1 :=
by
  sorry -- proof to be filled in

end one_of_a_b_c_is_one_l674_674342


namespace train_pass_bridge_in_50_seconds_l674_674190

def length_of_train : ℕ := 360
def length_of_bridge : ℕ := 140
def speed_of_train_kmh : ℕ := 36
def total_distance : ℕ := length_of_train + length_of_bridge
def speed_of_train_ms : ℚ := (speed_of_train_kmh * 1000 : ℚ) / 3600 -- we use ℚ to avoid integer division issues
def time_to_pass_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_pass_bridge_in_50_seconds :
  time_to_pass_bridge = 50 := by
  sorry

end train_pass_bridge_in_50_seconds_l674_674190


namespace balls_in_boxes_l674_674819

theorem balls_in_boxes : ∃ (n : ℕ), n = 4 ∧ (number_of_ways 4 3 = 8) :=
by
  -- Define conditions
  have h_balls : 4 = 4 := rfl
  have h_boxes : 3 = 3 := rfl
  -- State the problem
  use 4
  split
  . exact h_balls
  . sorry -- proof that number_of_ways 4 3 = 8

end balls_in_boxes_l674_674819


namespace rate_of_change_of_area_l674_674561

theorem rate_of_change_of_area (dr_dt : ℝ) (r : ℝ) (dA_dt : ℝ) :
    r = 6 → dr_dt = 2 → dA_dt = 4 * 3.14159 * r → dA_dt = 24 * 3.14159 :=
by
  intros hr hrate hda_dt
  rw [hr, hrate] at hda_dt
  have : 4 * 3.14159 * 6 = 24 * 3.14159 := by norm_num
  rw [this] at hda_dt
  exact hda_dt

end rate_of_change_of_area_l674_674561


namespace polyhedron_vertex_bound_l674_674435

-- Define integer points in 3D space
structure IntPoint3D where
  x : ℤ
  y : ℤ
  z : ℤ

-- Define properties of a convex polyhedron in 3D space
structure ConvexPolyhedron (Ω : Type) where
  vertices : Set Ω
  is_vertex : Ω → Prop
  is_convex : Prop

-- Define that the given point set are on integer coordinates
def vertices_are_integer (P : ConvexPolyhedron IntPoint3D) : Prop :=
  ∀ v, P.is_vertex v → ∃ (x y z : ℤ), v = ⟨x, y, z⟩

-- Define there are no other integer points inside, on faces, or on edges
def no_other_integer_points (P : ConvexPolyhedron IntPoint3D) : Prop :=
  ∀ (x y z : ℤ), (∀ v ∈ P.vertices, (v.x, v.y, v.z) ≠ (x, y, z)) →
    -- Additional conditions for inside, on faces, or on edges
    (by sorry : ∀ edge ∈ P.vertices → P.vertices, (v.x, v.y, v.z) ∉ segment(edge)) →

-- Main theorem statement
theorem polyhedron_vertex_bound (P : ConvexPolyhedron IntPoint3D) 
  (h1 : vertices_are_integer P)
  (h2 : no_other_integer_points P) :
  P.vertices.size ≤ 8 :=
  sorry

end polyhedron_vertex_bound_l674_674435


namespace number_of_correct_statements_l674_674057

-- Definitions based on conditions
def independence_test_certainty (p : ℝ) : Prop := p = 0.9
def chi_squared_error_probability (p : ℝ) : Prop := p = 0.1
def inference_validity (certainty : ℝ) (error_probability : ℝ) : Prop :=
  certainty = 0.9 ∧ error_probability ≤ 0.1

-- The number of correct statements given the conditions
theorem number_of_correct_statements :
  (independence_test_certainty 0.9) →
  (chi_squared_error_probability 0.1) →
  (inference_validity 0.9 0.1) →
  2 = 2 :=
by {
  intros,
  sorry
}

end number_of_correct_statements_l674_674057


namespace probability_zeros_not_adjacent_is_0_6_l674_674396

-- Define the total number of arrangements of 5 elements where we have 3 ones and 2 zeros
def total_arrangements : Nat := 5.choose 2

-- Define the number of arrangements where 2 zeros are adjacent
def adjacent_zeros_arrangements : Nat := 4.choose 1 * 2

-- Define the probability that the 2 zeros are not adjacent
def probability_not_adjacent : Rat := (total_arrangements - adjacent_zeros_arrangements) / total_arrangements

-- Prove the desired probability is 0.6
theorem probability_zeros_not_adjacent_is_0_6 : probability_not_adjacent = 3 / 5 := by
  sorry

end probability_zeros_not_adjacent_is_0_6_l674_674396


namespace locus_of_midpoints_l674_674294

-- Define points and the circle
variables {M O : Point} {r : ℝ}
-- Assume M, O are distinct and r > 0
axiom distinct_MO : M ≠ O
axiom positive_radius : 0 < r

-- Define the midpoint function for chord passing through M
noncomputable def midpoint_of_chord (A B : Point) : Point :=
  Point.midpoint A B

-- Define the function to calculate the diameter and center of the circle
noncomputable def diameter (A B : Point) : ℝ :=
  Point.dist A B

noncomputable def circle_center (A B : Point) : Point :=
  Point.midpoint A B

-- Theorem stating the locus of midpoints of chords passing through M is a circle
theorem locus_of_midpoints (X : Point) :
  (∃ A B : Point, X = midpoint_of_chord A B ∧ A ≠ B ∧ A ≠ X ∧ B ≠ X ∧ lies_on_circle A O r ∧ lies_on_circle B O r ∧ lies_on_chord M A B) ↔
  Point.dist X (circle_center M O) = diameter M O / 2 := 
sorry

end locus_of_midpoints_l674_674294


namespace time_crossing_bridge_approx_24_minutes_l674_674638

def walking_speed (km_per_hr : ℝ) : ℝ := km_per_hr * 1000 / 60

def time_to_cross_bridge (bridge_length : ℝ) (speed_m_per_min : ℝ) : ℝ := bridge_length / speed_m_per_min

theorem time_crossing_bridge_approx_24_minutes :
  let speed := walking_speed 10
  let bridge_length := 4000
  (time_to_cross_bridge bridge_length speed) ≈ 24 := by
  -- Here we can assume the necessary steps to prove it's approximately 24
  sorry

end time_crossing_bridge_approx_24_minutes_l674_674638


namespace number_exceeds_part_l674_674639

theorem number_exceeds_part (x : ℝ) (h : x = (5 / 9) * x + 150) : x = 337.5 := sorry

end number_exceeds_part_l674_674639


namespace probability_of_non_adjacent_zeros_l674_674392

-- Define the total number of arrangements of 3 ones and 2 zeros
def totalArrangements : ℕ := Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

-- Define the number of arrangements where the 2 zeros are together
def adjacentZerosArrangements : ℕ := 2 * Nat.factorial 4 / (Nat.factorial 3 * Nat.factorial 1)

-- Calculate the desired probability
def nonAdjacentZerosProbability : ℚ := 
  1 - (adjacentZerosArrangements.toRat / totalArrangements.toRat)

theorem probability_of_non_adjacent_zeros :
  nonAdjacentZerosProbability = 3/5 :=
sorry

end probability_of_non_adjacent_zeros_l674_674392


namespace is_random_variable_l674_674089

open MeasureTheory

variables {Ω : Type*} {n : ℕ}
variables {ξ : Ω → vector ℝ n} {ϕ : vector ℝ n → ℝ}

-- Assume {ξ₁, ξ₂, ..., ξₙ} are random variables
variables (ξ₁ ξ₂ ... ξₙ : Ω → ℝ)
-- Assume ϕ is a Borel measurable function
variable [measurable_space (vector ℝ n)] [borel_space (vector ℝ n)] 
          (ϕ_borel : measurable ϕ)

-- Show that ϕ(ξ₁(ω), ..., ξₙ(ω)) is a random variable
theorem is_random_variable :
  measurable (λ ω : Ω, ϕ (vector.of_fn (λ i, (ξ₁ ω, ξ₂ ω, ..., ξₙ ω)))) :=
sorry -- proof to be provided

end is_random_variable_l674_674089


namespace cubics_product_l674_674690

theorem cubics_product :
  (∏ n in [3, 4, 5, 6, 7], (n^3 - 1) / (n^3 + 1)) = (57 / 168) := by
  sorry

end cubics_product_l674_674690


namespace kendy_transfer_amount_l674_674073

theorem kendy_transfer_amount
  (X : ℝ)
  (initial_balance remaining_balance : ℝ)
  (h_initial : initial_balance = 190)
  (h_remaining : remaining_balance = 100)
  (h_total_transferred : initial_balance - remaining_balance = 90)
  (h_transfer_sister : X / 2)
  (h_total_eq : X + X / 2 = 90) :
  X = 60 :=
by
  sorry

end kendy_transfer_amount_l674_674073


namespace pirates_share_l674_674981

def initial_coins (N : ℕ) := N ≥ 3000 ∧ N ≤ 4000

def first_pirate (N : ℕ) := N - (2 + (N - 2) / 4)
def second_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)
def third_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)
def fourth_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)

def final_remaining (N : ℕ) :=
  let step1 := first_pirate N
  let step2 := second_pirate step1
  let step3 := third_pirate step2
  let step4 := fourth_pirate step3
  step4

theorem pirates_share (N : ℕ) (h : initial_coins N) :
  final_remaining N / 4 = 660 :=
by
  sorry

end pirates_share_l674_674981


namespace petya_guaranteed_win_l674_674885

-- Define the initial sequence as a list of digits
def initial_sequence : List ℕ := [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,
                                  6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7]

-- Define the rules of the game
def erase_move (seq : List ℕ) (digit : ℕ) : List ℕ :=
  seq.filter (λ x => x ≠ digit)

def is_winner (seq : List ℕ) : Bool :=
  seq = []

-- Define the proposition that Petya can guarantee a win
theorem petya_guaranteed_win : Prop :=
  ∃ moves : List (List ℕ), -- sequence of moves
    moves.headD initial_sequence = erase_move initial_sequence 7 ∧ -- Petya first move
    ∀ (n : ℕ → List ℕ), -- any move Vasya makes
      erase_move (List.foldr erase_move initial_sequence moves) n = is_winner (List.foldr erase_move initial_sequence moves) → true -- Petya can counter and win

end petya_guaranteed_win_l674_674885


namespace smallest_n_value_l674_674869

def conditions_met (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (∀ i, 0 ≤ i ∧ i < n → |x i| < 1) ∧
  (∑ i in Finset.range n, |x i| = 17 + |∑ i in Finset.range n, x i|)

theorem smallest_n_value :
  ∃ (x : ℕ → ℝ), conditions_met 18 x :=
sorry

end smallest_n_value_l674_674869


namespace actual_distance_travelled_l674_674411

theorem actual_distance_travelled :
  ∃ (D : ℝ), (D / 10 = (D + 20) / 14) ∧ D = 50 :=
by
  sorry

end actual_distance_travelled_l674_674411


namespace max_acutes_convex_polygon_l674_674579

theorem max_acutes_convex_polygon (n : ℕ) (hn : n ≥ 3) : 
  ∃ k : ℕ, k ≤ 3 ∧ ∀ (θ : ℕ → ℝ), 
  (∀ i, i < n → 0 < θ i ∧ θ i < 180) → (Σ i in finset.range k, θ i < 90) → (Σ i in finset.range n, θ i = (n-2) * 180) :=
by
  sorry

end max_acutes_convex_polygon_l674_674579


namespace digit_100th_place_of_8_over_11_l674_674025

/-- Decimal expansion of 8/11 -/
def decimal_expansion_8_over_11 : ℕ → ℕ
| n := if n % 2 = 0 then 7 else 2

/-- The 100th digit of the decimal expansion of 8/11 is 2 -/
theorem digit_100th_place_of_8_over_11 : decimal_expansion_8_over_11 99 = 2 :=
sorry

end digit_100th_place_of_8_over_11_l674_674025


namespace percentage_of_married_women_l674_674255

theorem percentage_of_married_women
  (total_employees : ℕ)
  (percent_women : ℝ)
  (percent_married : ℝ)
  (fraction_single_men : ℝ)
  (h1 : percent_women = 0.76)
  (h2 : percent_married = 0.60)
  (h3 : fraction_single_men = 2/3) :
  let women := total_employees * percent_women in
  let men := total_employees - women in
  let married_men := men * (1/3) in
  let married_women := (total_employees * percent_married) - married_men in
  (married_women / women) * 100 ≈ 68.42 := sorry

end percentage_of_married_women_l674_674255


namespace arithmetic_sequence_properties_l674_674773

-- Definitions from the conditions
def is_increasing_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n * (a 1 + a n)) / 2

def forms_geometric_sequence (a : ℕ → ℝ) (n₁ n₂ n₃ : ℕ) : Prop :=
a n₂ ^ 2 = a n₁ * a n₃

-- Problem Statement
theorem arithmetic_sequence_properties:
  (∃ a₁ d : ℝ, 
      is_increasing_arithmetic_sequence (λ n, a₁ + n * d) d ∧
      sum_first_n_terms (λ n, a₁ + n * d) 5 = 5 ∧ 
      forms_geometric_sequence (λ n, a₁ + n * d) 3 4 7) →
  (∀ n, (λ n, 2 * n - 5) n) ∧
  ( ∑ k in finset.range 100, abs ((λ n, 2 * n - 5) (k + 1)) = 9608 ) ∧
  (∃ λ : ℝ, ∀ n, (∀ x ∈ {n : ℕ | (-1)^n * ((2*n - 5) / 2^n) > λ}, x < 2) ∧
    (λ ≥ 7 / 64) ∧ (λ < 3 / 16)) :=
sorry

end arithmetic_sequence_properties_l674_674773


namespace rectangle_width_is_nine_l674_674143

theorem rectangle_width_is_nine (w l : ℝ) (h1 : l = 2 * w)
  (h2 : l * w = 3 * 2 * (l + w)) : 
  w = 9 :=
by
  sorry

end rectangle_width_is_nine_l674_674143


namespace probability_no_two_adjacent_stand_l674_674908

-- Definitions and Conditions
def fair_coin := { outcome : Bool // outcome = true ∨ outcome = false }
def flip_coin := (flip : fin 10 → fair_coin)

/-- Predicate to check if two adjacent people stand given a table arrangement -/
def no_two_adjacent (arr : flip_coin) : Prop :=
  ∀ i : fin 10, ¬ (arr.flip i).outcome = true ∧ (arr.flip (i + 1) % 10).outcome = true

-- The problem statement
theorem probability_no_two_adjacent_stand :
  (∑ (arr : flip_coin) in finset.univ, if no_two_adjacent arr then 1 else 0) / 1024 = (123 : ℚ) / 1024 :=
begin
  sorry
end

end probability_no_two_adjacent_stand_l674_674908


namespace divide_grid_l674_674280

def is_dotted (grid : ℕ × ℕ → Prop) (cells : list (ℕ × ℕ)) : Prop :=
  ∀i, cells[i] -> grid (cells[i].fst, cells[i].snd)

def equal_parts (regions : list (set (ℕ × ℕ))) : Prop :=
  regions.length = 4 ∧ ∀i j, i ≠ j → regions[i].card = regions[j].card

def correct_partition (grid : ℕ × ℕ → Prop)
    (regions : list (set (ℕ × ℕ))) (cells : list (ℕ × ℕ)) : Prop :=
  equal_parts(regions) ∧ ∀i, ∃cell, cell ∈ regions[i] ∧ grid(cell.fst, cell.snd)

theorem divide_grid (grid : ℕ × ℕ → Prop)
    (cells_dots : list (ℕ × ℕ))
    (regions : list (set (ℕ × ℕ))) :
  is_dotted(grid, cells_dots) ∧ correct_partition(grid, regions, cells_dots) :=
sorry

end divide_grid_l674_674280


namespace rain_third_day_l674_674953

theorem rain_third_day (rain_day1 rain_day2 rain_day3 : ℕ)
  (h1 : rain_day1 = 4)
  (h2 : rain_day2 = 5 * rain_day1)
  (h3 : rain_day3 = (rain_day1 + rain_day2) - 6) : 
  rain_day3 = 18 := 
by
  -- Proof omitted
  sorry

end rain_third_day_l674_674953


namespace vacation_days_proof_l674_674633

-- Define the conditions
def family_vacation (total_days rain_days clear_afternoons : ℕ) : Prop :=
  total_days = 18 ∧ rain_days = 13 ∧ clear_afternoons = 12

-- State the theorem to be proved
theorem vacation_days_proof : family_vacation 18 13 12 → 18 = 18 :=
by
  -- Skip the proof
  intro h
  sorry

end vacation_days_proof_l674_674633


namespace price_of_horse_and_cattle_l674_674056

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := 4 * x + 6 * y = 48
def equation2 (x y : ℝ) : Prop := 3 * x + 5 * y = 38

-- Define the values that are claimed to be solutions
def solution_x : ℝ := 6
def solution_y : ℝ := 4

-- Prove that the given values satisfy both equations
theorem price_of_horse_and_cattle : equation1 solution_x solution_y ∧ equation2 solution_x solution_y := 
by 
  show equation1 solution_x solution_y,
  { rw [solution_x, solution_y],
    norm_num },
  show equation2 solution_x solution_y,
  { rw [solution_x, solution_y],
    norm_num }

end price_of_horse_and_cattle_l674_674056


namespace determine_P_n_plus_1_l674_674872

noncomputable def P (x : ℕ) (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1),
    (k.factorial * (n - k).factorial / n.factorial) *
    ((Finset.range (n + 1)).erase k).prod (λ i => (x - i) / (k - i))

theorem determine_P_n_plus_1 {n : ℕ} (P : ℕ → ℚ) :
  (∀ k, k ≤ n → P k = 1 / (Nat.choose n k)) →
  P (n+1) = if n % 2 = 0 then 1 else 0 :=
by
  intro h
  sorry

end determine_P_n_plus_1_l674_674872


namespace angle_B_is_pi_over_4_l674_674782

theorem angle_B_is_pi_over_4 
  (A B C : Point) 
  (S : ℝ)
  (h_area : S = (1 / 2) * |A - C| * |B - C| * Real.sin (angle A C B))
  (h_condition : |B - C|^2 = (A - C) • (B - C) + 2 * S) :
  angle A B C = π / 4 :=
by
  sorry

end angle_B_is_pi_over_4_l674_674782


namespace angle_TSM_l674_674445

theorem angle_TSM {K L M T S : Type} [EuclideanGeometry K]
  (KL KM KT KS LKS TSM: Ω)
  (hKLKM: KL = KM)
  (hKTKS: KT = KS)
  (hLKS: LKS = 30)
  (hKT_KS: ∠KT = ∠KS)
  (htriangle1: IsIsoscelesTriangle K L M)
  (htriangle2: IsIsoscelesTriangle K S T) :
  ∠TSM = 15 :=
  sorry

end angle_TSM_l674_674445


namespace hyperbola_properties_l674_674318

noncomputable def hyperbola_eq : Prop := 
  ∃ (x y a b : ℝ), 
    (a > 0) ∧ (b > 0) ∧ 
    (a = 2 * sqrt 3) ∧ 
    (b^2 = 3) ∧
    (f : ℝ) (hf : f^2 = a^2 + b^2)
    (abs (b * f / sqrt (b^2 + 12)) = sqrt 3)
    (right_branch : x^2 / a^2 - y^2 / b^2 = 1)
    (right_branch_result : x^2 / 12 - y^2 / 3 = 1)

noncomputable def points_eq : Prop := 
  ∃ (x₁ y₁ x₂ y₂ x₀ y₀ m : ℝ), 
    (intersect_eq : y₁ = sqrt 3 / 3 * x₁ - 2) ∧ 
    (intersect_eq : y₂ = sqrt 3 / 3 * x₂ - 2) ∧ 
    (x₁ + x₂ = 16 * sqrt 3) ∧
    (y₁ + y₂ = 12) ∧
    (x₀ = 4 * sqrt 3) ∧
    (y₀ = 3) ∧
    (m = 4)

theorem hyperbola_properties : hyperbola_eq ∧ points_eq := 
  sorry

end hyperbola_properties_l674_674318


namespace dig_second_hole_l674_674199

theorem dig_second_hole (w1 h1 d1 w2 d2 : ℕ) (extra_workers : ℕ) (h2 : ℕ) :
  w1 = 45 ∧ h1 = 8 ∧ d1 = 30 ∧ extra_workers = 65 ∧
  w2 = w1 + extra_workers ∧ d2 = 55 →
  360 * d2 / d1 = w2 * h2 →
  h2 = 6 :=
by
  intros h cond
  sorry

end dig_second_hole_l674_674199


namespace bridge_length_is_245_l674_674549

-- Define the conditions
def train_length : ℝ := 130
def train_speed_kmh : ℝ := 45
def crossing_time : ℝ := 30

-- Define the conversion from km/hr to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := (speed_kmh * 1000) / 3600

-- Define the speed of the train in m/s
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

-- Define the total distance traveled by the train in 30 seconds
def total_distance : ℝ := train_speed_ms * crossing_time

-- Define the length of the bridge
def bridge_length : ℝ := total_distance - train_length

-- State the theorem to be proved
theorem bridge_length_is_245 : bridge_length = 245 :=
by
  sorry

end bridge_length_is_245_l674_674549


namespace tangent_length_AR_l674_674208

-- Define the setup of the geometry problem with the given constants
variables {A P Q R S : Point}
variables {O1 O2 : Point}
variables (ω1 ω2 : Circle)
variables (ha1 : ω1.radius = 15)
variables (ha2 : ω2.radius = 13)
variables (hPQ : dist P Q = 24)
variables (hO1P : on_circle P ω1)
variables (hO1Q : on_circle Q ω1)
variables (hO2P : on_circle P ω2)
variables (hO2Q : on_circle Q ω2)
variables (hPAQ : on_line P Q A)
variables (h_tang : tangent_point R ω1 A ∧ tangent_point S ω2 A)
variables (h_no_intersect_s : ¬ (line_through A S).intersects ω1)
variables (h_no_intersect_r : ¬ (line_through A R).intersects ω2)
variables (h_angle : angle R A S = 90)

-- Define the theorem to prove the length of AR
theorem tangent_length_AR : dist A R = 14 + real.sqrt 97 := sorry

end tangent_length_AR_l674_674208


namespace triangle_area_DBC_l674_674441

structure Point where
  x : ℝ
  y : ℝ

def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2
    y := (p1.y + p2.y) / 2 }

def area_of_triangle (A B C : Point) : ℝ :=
  abs (A.x*(B.y - C.y) + B.x*(C.y - A.y) + C.x*(A.y - B.y)) / 2

theorem triangle_area_DBC :
  let A := Point.mk 0 8
  let B := Point.mk 0 0
  let C := Point.mk 10 0
  let D := midpoint A B
  (area_of_triangle D B C = 20) :=
by
  sorry

end triangle_area_DBC_l674_674441


namespace quadratic_trinomial_positive_c_l674_674146

theorem quadratic_trinomial_positive_c
  (a b c : ℝ)
  (h1 : b^2 < 4 * a * c)
  (h2 : a + b + c > 0) :
  c > 0 :=
sorry

end quadratic_trinomial_positive_c_l674_674146


namespace a_n_formula_T_n_formula_l674_674328

noncomputable def a (n : ℕ) : ℕ := 2 ^ n

noncomputable def S (n : ℕ) : ℕ := (finset.range n).sum a

noncomputable def b (n : ℕ) : ℤ := (-1) ^ n * (2 * n + 1)

noncomputable def T (n : ℕ) : ℤ := (finset.range n).sum b

theorem a_n_formula (n : ℕ) : a n = 2 ^ n := 
by 
have a_def : a 1 = 2 := sorry
have arithmetic_mean : ∀ n, a n = (S n + 2) / 2 := sorry 
exact sorry

theorem T_n_formula (n : ℕ) : T n = if n % 2 = 1 then -n - 2 else n := 
by 
have b_def : ∀ n, b n = (-1) ^ n * (2 * n + 1) := sorry 
exact sorry

end a_n_formula_T_n_formula_l674_674328


namespace length_segment_AC_l674_674266

theorem length_segment_AC
  (Q : Type)
  (circumference_Q : ℝ)
  (h_circumference_Q : circumference_Q = 16 * real.pi)
  (AB : ℝ)
  (is_diameter : AB = 2 * (8 : ℝ)) -- Since the radius is 8 meters derived from the circumference
  (angle_QAC : ℝ)
  (h_angle_QAC : angle_QAC = 30) :
  ∃ AC : ℝ, AC = 8 :=
by
  -- Define the given conditions
  have r : ℝ := 8, -- since 2 * π * r = 16 * π => r = 8
  -- Use the fact that sin(30) is 1/2
  use 8,
  -- Geometry based on given angle and radius
  sorry

end length_segment_AC_l674_674266


namespace linear_function_decreases_l674_674756

theorem linear_function_decreases (m : ℝ) : (∀ x : ℝ, (m-3) < 0 → (m-3)*x + (6 + 2*m) = (m-3)*x + 6 + 2*m → x → y = (m-3)*x + 6 + 2*m) → (m < 3) :=
by
  sorry

end linear_function_decreases_l674_674756


namespace same_foci_condition_l674_674543

-- Defining the equations of the ellipse and hyperbola with the given conditions
def ellipse (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def hyperbola (a k : ℝ) (x y : ℝ) := (x^2 / a^2) - (y^2 / k^2) = 1

-- Given that both the ellipse and the hyperbola have the same foci, we need to prove the condition on k
theorem same_foci_condition (a b k : ℝ) :
  (∀ x y : ℝ, ellipse a b x y) →
  (∀ x y : ℝ, hyperbola a k x y) →
  (sqrt (a^2 - b^2) = sqrt (a^2 + k^2)) →
  k = 2 :=
by
  sorry

end same_foci_condition_l674_674543


namespace sum_of_divisors_30_l674_674173

theorem sum_of_divisors_30 : (∑ d in (Finset.filter (λ d, 30 % d = 0) (Finset.range 31)), d) = 72 := 
by
  sorry

end sum_of_divisors_30_l674_674173


namespace range_g_l674_674471

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2

noncomputable def g (x : ℝ) : ℝ := f (f x)

theorem range_g : ∀ x ∈ Icc (0 : ℝ) 1, 1 ≤ g x ∧ g x ≤ 12 := by
  sorry

end range_g_l674_674471


namespace math_problem_l674_674686

theorem math_problem :
  ( ∏ i in [3, 4, 5, 6, 7], (i^3 - 1) / (i^3 + 1) ) = 57 / 84 := sorry

end math_problem_l674_674686


namespace find_f_value_l674_674798

def f (x a b : ℝ) := x^3 + a / x + b * x - 3

theorem find_f_value (a b : ℝ) (h : f (-2023) a b = 2023) : f 2023 a b = -2029 := by
  sorry

end find_f_value_l674_674798


namespace prob_zeros_not_adjacent_l674_674382

theorem prob_zeros_not_adjacent :
  let total_arrangements := (5.factorial : ℝ)
  let zeros_together_arrangements := (4.factorial : ℝ)
  let prob_zeros_together := (zeros_together_arrangements / total_arrangements)
  let prob_zeros_not_adjacent := 1 - prob_zeros_together
  prob_zeros_not_adjacent = 0.6 :=
by
  sorry

end prob_zeros_not_adjacent_l674_674382


namespace fraction_stamp_collection_l674_674019

theorem fraction_stamp_collection (sold_amount total_value : ℝ) (sold_for : sold_amount = 28) (total : total_value = 49) : sold_amount / total_value = 4 / 7 :=
by
  sorry

end fraction_stamp_collection_l674_674019


namespace acute_triangle_has_grid_vertex_inside_or_on_sides_l674_674500

theorem acute_triangle_has_grid_vertex_inside_or_on_sides
  {A B C : ℤ × ℤ} -- Points A, B, and C are vertices at grid points
  (h_acute : ∀ (α β γ : ℝ), 
    α + β + γ = π ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2) :
  ∃ D : ℤ × ℤ, 
  (D ≠ A ∧ D ≠ B ∧ D ≠ C) ∧
  (D ∈ (triangle_interior A B C ∪ triangle_border A B C)) := 
sorry

end acute_triangle_has_grid_vertex_inside_or_on_sides_l674_674500


namespace count_sets_l674_674142

variable {M A B X : Finset ℕ}

theorem count_sets (hM : M.card = 10) (hA : A ⊆ M) (hB : B ⊆ M)
    (h_disjoint : A ∩ B = ∅) (hA_card : A.card = 2) (hB_card : B.card = 3) :
    (Finset.filter (λ X, X ⊆ M ∧ ¬ (A ⊆ X) ∧ ¬ (B ⊆ X)) (Finset.powerset M)).card = 672 :=
sorry

end count_sets_l674_674142


namespace six_lines_six_intersections_l674_674067

/-- It is possible to draw six lines on a plane such that there are exactly six points of intersection. -/
theorem six_lines_six_intersections : ∃ (lines : set (line ℝ)), lines.card = 6 ∧ (∃ (intersections : set (point ℝ)), intersections.card = 6 ∧ ∀ p ∈ intersections, ∃ l1 l2 ∈ lines, l1 ≠ l2 ∧ p ∈ l1 ∧ p ∈ l2) :=
sorry

end six_lines_six_intersections_l674_674067


namespace probability_is_ten_over_twenty_one_l674_674595

noncomputable def probability_black_and_white : ℚ :=
  let total_ways := nat.choose 7 2 in
  let favorable_ways := 5 * 2 in
  (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_is_ten_over_twenty_one : probability_black_and_white = 10 / 21 :=
by
  sorry

end probability_is_ten_over_twenty_one_l674_674595


namespace intervals_of_monotonicity_range_on_interval_zero_one_l674_674795

noncomputable def f (x : ℝ) : ℝ := x * Real.exp(x) + 5

theorem intervals_of_monotonicity :
  (∀ x, x > -1 -> f.deriv x > 0) ∧ (∀ x, x < -1 -> f.deriv x < 0) := sorry

theorem range_on_interval_zero_one :
  set.image f (set.Icc 0 1) = set.Icc 5 (Real.exp 1 + 5) := sorry

end intervals_of_monotonicity_range_on_interval_zero_one_l674_674795


namespace quadratic_no_real_roots_l674_674886

theorem quadratic_no_real_roots (c : ℝ) (h : c > 1) : ∀ x : ℝ, x^2 + 2 * x + c ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l674_674886


namespace board_numbers_l674_674448

theorem board_numbers (a b c : ℕ) (h1 : a = 3) (h2 : b = 9) (h3 : c = 15)
    (op : ∀ x y z : ℕ, (x = y + z - t) → true)  -- simplifying the operation representation
    (min_number : ∃ x, x = 2013) : ∃ n m, n = 2019 ∧ m = 2025 := 
sorry

end board_numbers_l674_674448


namespace sum_of_squares_l674_674937

theorem sum_of_squares :
  (2^2 + 1^2 + 0^2 + (-1)^2 + (-2)^2 = 10) :=
by
  sorry

end sum_of_squares_l674_674937


namespace maximum_elements_in_A_l674_674779

theorem maximum_elements_in_A (n : ℕ) (h : n > 0)
  (A : Finset (Finset (Fin n))) 
  (hA : ∀ a ∈ A, ∀ b ∈ A, a ≠ b → ¬ a ⊆ b) :  
  A.card ≤ Nat.choose n (n / 2) :=
sorry

end maximum_elements_in_A_l674_674779


namespace sum_of_solutions_l674_674461

theorem sum_of_solutions :
  let solutions := [(x, y) | x y : ℤ, |x - 4| = |y - 8| ∧ |x - 8| = 3 * |y - 4|] in
  List.sum (solutions.map (λ p => p.1 + p.2)) = 4 :=
by
  sorry

end sum_of_solutions_l674_674461


namespace volume_of_sphere_l674_674343

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def tetrahedron_on_sphere (A B C D : EuclideanSpace ℝ 3) (O : EuclideanSpace ℝ 3) : Prop :=
  dist O A = dist O B ∧
  dist O B = dist O C ∧
  dist O C = dist O D

theorem volume_of_sphere
  (A B C D : EuclideanSpace ℝ 3)
  (O : EuclideanSpace ℝ 3)
  (h1 : tetrahedron_on_sphere A B C D O)
  (h2 : (dist D C) ≠ 0)
  (h3 : (dist D (dist C Plane(ABC)) = 0))
  (h4 : dist A C = 2 * Real.sqrt 3)
  (h5 : ∀ (X Y Z : EuclideanSpace ℝ 3), (X = A ∧ Y = B ∧ Z = C) → equilateral_triangle X Y Z)
  (h6 : ∀ (X Y Z : EuclideanSpace ℝ 3), (X = A ∧ Y = C ∧ Z = D) → isosceles_triangle X Y Z)
  : sphere_volume (Real.sqrt 7) = (28 * Real.sqrt 7 / 3) * Real.pi :=
by
sory

end volume_of_sphere_l674_674343


namespace lateral_surface_area_of_cone_l674_674754

open Real

theorem lateral_surface_area_of_cone
  (SA : ℝ) (SB : ℝ)
  (cos_angle_SA_SB : ℝ) (angle_SA_base : ℝ)
  (area_SAB : ℝ) :
  cos_angle_SA_SB = 7 / 8 →
  angle_SA_base = π / 4 →
  area_SAB = 5 * sqrt 15 →
  SA = 4 * sqrt 5 →
  SB = SA →
  (1/2) * (sqrt 2 / 2 * SA) * (2 * π * SA) = 40 * sqrt 2 * π :=
sorry

end lateral_surface_area_of_cone_l674_674754


namespace minnie_takes_longer_l674_674488

def minnie_speed_flat := 25 -- kph
def minnie_speed_downhill := 35 -- kph
def minnie_speed_uphill := 10 -- kph

def penny_speed_flat := 35 -- kph
def penny_speed_downhill := 45 -- kph
def penny_speed_uphill := 15 -- kph

def distance_flat := 25 -- km
def distance_downhill := 20 -- km
def distance_uphill := 15 -- km

noncomputable def minnie_time := 
  (distance_uphill / minnie_speed_uphill) + 
  (distance_downhill / minnie_speed_downhill) + 
  (distance_flat / minnie_speed_flat) -- hours

noncomputable def penny_time := 
  (distance_uphill / penny_speed_uphill) + 
  (distance_downhill / penny_speed_downhill) + 
  (distance_flat / penny_speed_flat) -- hours

noncomputable def minnie_time_minutes := minnie_time * 60 -- minutes
noncomputable def penny_time_minutes := penny_time * 60 -- minutes

noncomputable def time_difference := minnie_time_minutes - penny_time_minutes -- minutes

theorem minnie_takes_longer : time_difference = 130 :=
  sorry

end minnie_takes_longer_l674_674488


namespace rem_fib_2011_l674_674366

/-- Define the Fibonacci sequence -/
def Fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n+2) := Fibonacci (n+1) + Fibonacci n

/-- Define the remainder sequence r_n when Fibonacci is divided by 3 -/
def r (n : ℕ) : ℕ :=
(Fibonacci n) % 3

/-- The main theorem giving the value of r_2011 -/
theorem rem_fib_2011 : r 2011 = 2 :=
by sorry

end rem_fib_2011_l674_674366


namespace diagonal_AC_length_l674_674434

theorem diagonal_AC_length (AB BC CD DA : ℝ) (angle_ADC : ℝ) (h_AB : AB = 12) (h_BC : BC = 12) 
(h_CD : CD = 13) (h_DA : DA = 13) (h_angle_ADC : angle_ADC = 60) : 
  AC = 13 := 
sorry

end diagonal_AC_length_l674_674434


namespace ratio_of_wall_to_pool_l674_674998

-- Definitions for the problem based on the given conditions
def radius_pool : ℝ := 20
def width_wall : ℝ := 4
def r_outer : ℝ := radius_pool + width_wall

def area_circle (r : ℝ) : ℝ := Real.pi * r^2

-- Define the areas based on the given definitions
def area_pool : ℝ := area_circle radius_pool
def area_total : ℝ := area_circle r_outer
def area_wall : ℝ := area_total - area_pool

-- The statement to be proved: ratio_area_wall_to_pool = 11 / 25
theorem ratio_of_wall_to_pool : (area_wall / area_pool) = (11 / 25) := by
  sorry

end ratio_of_wall_to_pool_l674_674998


namespace sum_max_marks_is_1300_l674_674560

-- Define the conditions for the Math test
def total_marks_math (marks_obtained : ℕ) (failed_by : ℕ) (percentage : ℚ) :=
  let passing_marks := marks_obtained + failed_by in
  passing_marks / percentage

-- Define the conditions for the Science test
def total_marks_science (marks_obtained : ℕ) (failed_by : ℕ) (percentage : ℚ) :=
  let passing_marks := marks_obtained + failed_by in
  passing_marks / percentage

-- Define the conditions for the English test
def total_marks_english (marks_obtained : ℕ) (failed_by : ℕ) (percentage : ℚ) :=
  let passing_marks := marks_obtained + failed_by in
  passing_marks / percentage

-- Define the total sum of maximum marks.
def sum_of_max_marks (math_marks science_marks english_marks : ℕ) :=
  math_marks + science_marks + english_marks

-- Problem statement to prove that the sum of maximum marks is 1300
theorem sum_max_marks_is_1300 (m1_failed_by s1_failed_by e1_failed_by : ℕ)
  (m1_marks_obtained s1_marks_obtained e1_marks_obtained : ℕ)
  (m1_percentage s1_percentage e1_percentage : ℚ) :
  m1_failed_by = 100 → m1_marks_obtained = 80 → m1_percentage = 0.30 →
  s1_failed_by = 80 → s1_marks_obtained = 120 → s1_percentage = 0.50 →
  e1_failed_by = 60 → e1_marks_obtained = 60 → e1_percentage = 0.40 →
  sum_of_max_marks (total_marks_math m1_marks_obtained m1_failed_by m1_percentage).to_nat
                   (total_marks_science s1_marks_obtained s1_failed_by s1_percentage).to_nat
                   (total_marks_english e1_marks_obtained e1_failed_by e1_percentage).to_nat = 1300 :=
by
  intros
  sorry

end sum_max_marks_is_1300_l674_674560


namespace solve_for_y_l674_674198

variable (x y z : ℝ)

theorem solve_for_y (h : 3 * x + 3 * y + 3 * z + 11 = 143) : y = 44 - x - z :=
by 
  sorry

end solve_for_y_l674_674198


namespace angle_bisector_area_ratio_l674_674849

theorem angle_bisector_area_ratio (a b c : ℝ) (h₁ : a = b) (triangle_area : ℝ) :
  let area_1 := triangle_area / (1 + ℝ.sqrt 2),
      area_2 := triangle_area - area_1 in
  area_1 / area_2 = 1 / (ℝ.sqrt 2) :=
by
  sorry

end angle_bisector_area_ratio_l674_674849


namespace correctPattern_l674_674106

-- Define the grid as a 2D list of cells
def grid : List (List (String)) := 
  List.repeat (List.repeat "A" 10) 1 ++ -- Top row
  List.repeat (List.repeat "B" 10) 1    -- Bottom row

-- Define the target pattern
def targetPattern (r c : Nat) : String :=
  if r = 0 then "A" else "B"

-- Define a function to create the pattern on the grid
def createPattern (rows cols : Nat) : List (List (String)) :=
  List.init rows (λ r =>
    List.init cols (λ c => targetPattern r c))

-- The proof statement
theorem correctPattern : 
  createPattern 2 10 = grid :=
by sorry

end correctPattern_l674_674106


namespace find_ellipse_eqn_find_abscissa_G_l674_674344

-- Step 1: Define the conditions under which the problem is set
def is_ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def has_eccentricity (a b : ℝ) : Prop := 
  (b^2 / a^2 = 3 / 4)

def point_on_ellipse (a b : ℝ) : Prop :=
  ((2^2 / a^2) + (3 / b^2)) = 1

def vertices (a b : ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (-a, 0) ∧ B = (a, 0)

def abscissa_G (x : ℝ) : Prop := 
  x = 8

-- Step 2: State the problem in Lean as theorem declarations
theorem find_ellipse_eqn (a b : ℝ) (h : a > b ∧ b > 0)
  (he : has_eccentricity a b) (hp : point_on_ellipse a b) : 
    ∀ x y : ℝ, is_ellipse 4 2 (by norm_num) x y :=
sorry

theorem find_abscissa_G (a b : ℝ) (A B : ℝ × ℝ) 
  (hAB : vertices a b A B) (HQ : (2, 0) ∈ ellipse) 
  (hMN : intersects (line_through Q) (ellipse) at M N) 
  (hG : intersection (line_AN A N) (line_BM B M) = G) : 
    abscissa_G G.fst :=
sorry

end find_ellipse_eqn_find_abscissa_G_l674_674344


namespace new_average_score_l674_674489

theorem new_average_score (num_students : ℕ)
    (scores_100 scores_90 scores_80 scores_70 scores_60 scores_50 scores_40 scores_30 : ℕ) 
    (H1 : num_students = 120) 
    (H2 : scores_100 = 10) 
    (H3 : scores_90 = 20) 
    (H4 : scores_80 = 40) 
    (H5 : scores_70 = 30) 
    (H6 : scores_60 = 10) 
    (H7 : scores_50 = 5) 
    (H8 : scores_40 = 3) 
    (H9 : scores_30 = 2) :
  (100 * scores_100 + 90 * scores_90 + 80 * scores_80 + 70 * scores_70 + 60 * scores_60 + 
   55 * scores_50 + 45 * scores_40 + 35 * scores_30) / num_students = 76.5 :=
by
  sorry

end new_average_score_l674_674489


namespace solve_for_x_l674_674518

theorem solve_for_x (x : ℂ) (h : 5 - 3 * complex.I * x = -2 + 6 * complex.I * x) : x = -7 * complex.I / 9 := 
sorry

end solve_for_x_l674_674518


namespace men_wages_l674_674200

theorem men_wages (W : ℕ) (wage : ℕ) :
  (5 + W + 8) * wage = 75 ∧ 5 * wage = W * wage ∧ W * wage = 8 * wage → 
  wage = 5 := 
by
  sorry

end men_wages_l674_674200


namespace sequence_cubes_mod_6_l674_674468

theorem sequence_cubes_mod_6 (a : ℕ → ℕ) (h : ∀ i j, i < j → a i < a j)
  (sum_cond : (Finset.range 2021).sum (λ i, a i) = 2021 ^ 2021) :
  ((Finset.range 2021).sum (λ i, (a i)^3)) % 6 = 5 := 
sorry

end sequence_cubes_mod_6_l674_674468


namespace remainder_9_minus_n_plus_n_plus_5_mod_8_l674_674402

theorem remainder_9_minus_n_plus_n_plus_5_mod_8 (n : ℤ) : 
  ((9 - n) + (n + 5)) % 8 = 6 := by
  sorry

end remainder_9_minus_n_plus_n_plus_5_mod_8_l674_674402


namespace inequality_of_positives_l674_674767

theorem inequality_of_positives (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
begin
    sorry
end

end inequality_of_positives_l674_674767


namespace find_s_l674_674286

theorem find_s (s : ℝ) : (s, 7) ∈ line_through (0, 4) (-6, 1) → s = 6 :=
sorry

end find_s_l674_674286


namespace number_never_appeared_l674_674486

theorem number_never_appeared (dice : Fin 6 → ℕ) (prime : ℕ → Prop) : 
  (∀ i, prime (dice i)) → 
  (∀ i, ∃ p : ℕ, prime p ∧ dice i = p) →
  (∑ i in Finset.univ, dice i = 87) →
  let rolled1 := finset.sum ({0, 1, 2} : Finset (Fin 3)) (fun i => dice i) in
  let rolled2 := finset.sum ({0, 1, 2} : Finset (Fin 3)) (fun i => dice i) in
  (rolled1 = 10) →
  (rolled2 = 62) →
  ∃ p : ℕ, prime p ∧ (∀ i, dice i ≠ p) ∧ ∑ (i : Fin 6 => dice i = 87) :=
by
  sorry

end number_never_appeared_l674_674486


namespace count_of_king_checkers_l674_674496

-- Define the concept of checkers flipped by students and what constitutes a "king" checker
def is_flipped_odd_number_of_times (n : ℕ) : Prop :=
  n % 2 = 1

-- Number of students/checkers and their ordering
def number_of_checkers : ℕ :=
  64

-- The checker numbers where checkers are flipped an odd number of times
def odd_flipped_checkers : ℕ :=
  List.length (List.filter is_flipped_odd_number_of_times (List.range (number_of_checkers + 1)))

-- The theorem stating the number of "king" checkers
theorem count_of_king_checkers : odd_flipped_checkers = 8 :=
  sorry

end count_of_king_checkers_l674_674496


namespace number_of_sequences_l674_674750

noncomputable section

open Set

theorem number_of_sequences :
  (∀ b1 b2 b3 b4 : ℝ, 
    { x | ∃ i j, (1 ≤ i < j ≤ 4) ∧ x = ([2, 4, 8, 16].nth_le i sorry + [2, 4, 8, 16].nth_le j sorry)} = 
    { y | ∃ i j, (1 ≤ i < j ≤ 4) ∧ y = ([b1, b2, b3, b4].nth_le i sorry + [b1, b2, b3, b4].nth_le j sorry)} → 
    ∃ (l : Finset (Fin 4 → ℝ)), 
     l.card = 48 ∧ 
     ∀ f ∈ l, { x | ∃ i j, (1 ≤ i < j ≤ 4) ∧ x = (f i + f j)} = 
     { y | ∃ i j, (1 ≤ i < j ≤ 4) ∧ y = ([b1, b2, b3, b4].nth_le i sorry + [b1, b2, b3, b4].nth_le j sorry)}) :=
begin
  sorry
end

end number_of_sequences_l674_674750


namespace share_difference_l674_674664

theorem share_difference 
  (ratio_f : ℕ) (ratio_v : ℕ) (ratio_r : ℕ)
  (share_v : ℕ)
  (h_ratio : ratio_f = 3 ∧ ratio_v = 5 ∧ ratio_r = 8)
  (h_share_v : share_v = 1500) :
  let part := share_v / ratio_v,
      share_f := ratio_f * part,
      share_r := ratio_r * part
  in share_r - share_f = 1500 :=
by {
  sorry
}

end share_difference_l674_674664


namespace celine_change_l674_674647

theorem celine_change
  (price_laptop : ℕ)
  (price_smartphone : ℕ)
  (num_laptops : ℕ)
  (num_smartphones : ℕ)
  (total_money : ℕ)
  (h1 : price_laptop = 600)
  (h2 : price_smartphone = 400)
  (h3 : num_laptops = 2)
  (h4 : num_smartphones = 4)
  (h5 : total_money = 3000) :
  total_money - (num_laptops * price_laptop + num_smartphones * price_smartphone) = 200 :=
by
  sorry

end celine_change_l674_674647


namespace find_ab_l674_674774

theorem find_ab (a b : ℝ) :
  let M := matrix.vec_cons (vector.cons (-1 : ℝ) (vector.cons a vector.nil))
                           (matrix.vec_cons (vector.cons b (vector.cons (3 : ℝ) vector.nil))) in
  (∀ x y : ℝ, let x₀ := (-x + a * y) in
             let y₀ := (b * x + 3 * y) in
             (2 * x₀ - y₀ = 3) ↔ (2 * x - y = 3)) →
  a = 1 ∧ b = -4 :=
by
  intros
  sorry

end find_ab_l674_674774


namespace monotonic_increasing_range_of_a_l674_674003

theorem monotonic_increasing_range_of_a :
  (∀ (x : ℝ), monotone (λ x, x^3 - a * x)) ↔ a ≤ 0 :=
by
  sorry

end monotonic_increasing_range_of_a_l674_674003


namespace angle_BDC_is_30_l674_674428

theorem angle_BDC_is_30 
    (A E C B D : ℝ) 
    (hA : A = 50) 
    (hE : E = 60) 
    (hC : C = 40) : 
    BDC = 30 :=
by
  sorry

end angle_BDC_is_30_l674_674428


namespace triangle_condition_to_find_m_not_equal_neg_one_right_triangle_condition_to_find_m_for_right_angle_A_l674_674811

def vector (α : Type*) := (x y : α) -- A vector in 2D

def OA : vector ℝ := ⟨3, -4⟩
def OB : vector ℝ := ⟨6, -3⟩
def OC (m : ℝ) : vector ℝ := ⟨5 - m, - (4 + m)⟩

def AB := λ (OA OB : vector ℝ), ⟨OB.1 - OA.1, OB.2 - OA.2⟩
def AC := λ (OA OC : vector ℝ), ⟨OC.1 - OA.1, OC.2 - OA.2⟩

def dot_product (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem triangle_condition_to_find_m_not_equal_neg_one (OA OB OC : vector ℝ) : 
  real := λ m, 
  m ≠ -1 := sorry

theorem right_triangle_condition_to_find_m_for_right_angle_A (OA OB OC : vector ℝ) :
  real := λ m, 
  dot_product (AB OA OB) (AC OA (OC m)) = 0 → m = 3 / 2 := sorry

end triangle_condition_to_find_m_not_equal_neg_one_right_triangle_condition_to_find_m_for_right_angle_A_l674_674811


namespace g_of_1001_l674_674529

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x * y) + x = x * g y + g x
axiom g_of_1 : g 1 = -3

theorem g_of_1001 : g 1001 = -2001 := 
by sorry

end g_of_1001_l674_674529


namespace triangle_area_example_l674_674603

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
let s := (a + b + c) / 2 in
Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_example :
    triangle_area 26 24 15 ≈ 175.8 := by
  sorry

end triangle_area_example_l674_674603


namespace sin_double_angle_is_one_l674_674833

variable (θ : ℝ)

def a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def b : ℝ × ℝ := (1, -1)

def perp (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem sin_double_angle_is_one 
  (h : perp a b) : 
  Real.sin (2 * θ) = 1 := by
  sorry

end sin_double_angle_is_one_l674_674833


namespace diagonals_diff_heptagon_octagon_l674_674408

-- Define the function to calculate the number of diagonals in a polygon with n sides
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_diff_heptagon_octagon : 
  let A := num_diagonals 7
  let B := num_diagonals 8
  B - A = 6 :=
by
  sorry

end diagonals_diff_heptagon_octagon_l674_674408


namespace minimum_rows_required_l674_674999

-- Define the conditions
variables (n : ℕ) (C : ℕ → ℕ)
hypothesis h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 1 ≤ C i ∧ C i ≤ 39
hypothesis h2 : (∑ i in finset.range n, C i) = 1990
def row_seats : ℕ := 199
def total_students : ℕ := 1990

-- Define the problem statement
theorem minimum_rows_required : 
  ∃ (r : ℕ), (∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → C i ≤ row_seats) ∧
  (∑ i in finset.range n, C i) = total_students → r = 12 :=
begin
  sorry
end

end minimum_rows_required_l674_674999


namespace binomial_coeff_linear_term_l674_674418

theorem binomial_coeff_linear_term (a : ℝ) :
    let expr := (2 * x + a / x) ^ 7
    let linear_term_coeff := binomial_coeff 7 3 * 2 ^ 4 * a ^ 3
    linear_term_coeff = -70 → a = -1/2 := by
  sorry

-- Definitions for binomial coefficient and expressions
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

variable (x : ℝ)

end binomial_coeff_linear_term_l674_674418


namespace new_average_rent_l674_674536

theorem new_average_rent 
  (n : ℕ) (h_n : n = 4) 
  (avg_old : ℝ) (h_avg_old : avg_old = 800) 
  (inc_rate : ℝ) (h_inc_rate : inc_rate = 0.16) 
  (old_rent : ℝ) (h_old_rent : old_rent = 1250) 
  (new_rent : ℝ) (h_new_rent : new_rent = old_rent * (1 + inc_rate)) 
  (total_rent_old : ℝ) (h_total_rent_old : total_rent_old = n * avg_old)
  (total_rent_new : ℝ) (h_total_rent_new : total_rent_new = total_rent_old - old_rent + new_rent)
  (avg_new : ℝ) (h_avg_new : avg_new = total_rent_new / n) : 
  avg_new = 850 := 
sorry

end new_average_rent_l674_674536


namespace rhombus_area_proof_l674_674891

variables (EFGH : Type u) [rhombus EFGH] 
variable (perimeter : ℝ)
variable (diag_EG : ℝ)
variable (area : ℝ)

def rhombus_area (EFGH : Type u) [rhombus EFGH] (perimeter : ℝ) (diag_EG : ℝ) : ℝ :=
  sorry

theorem rhombus_area_proof (EFGH : Type u) [rhombus EFGH] (h1 : perimeter = 40)
  (h2 : diag_EG = 16) : rhombus_area EFGH perimeter diag_EG = 96 :=
sorry

end rhombus_area_proof_l674_674891


namespace exists_integers_not_all_zero_l674_674076

-- Given conditions
variables (a b c : ℝ)
variables (ab bc ca : ℚ)
variables (ha : a * b = ab) (hb : b * c = bc) (hc : c * a = ca)
variables (x y z : ℤ)

-- The theorem to prove
theorem exists_integers_not_all_zero (ha : a * b = ab) (hb : b * c = bc) (hc : c * a = ca):
  ∃ (x y z : ℤ), (¬ (x = 0 ∧ y = 0 ∧ z = 0)) ∧ (a * x + b * y + c * z = 0) :=
sorry

end exists_integers_not_all_zero_l674_674076


namespace find_s_l674_674285

theorem find_s (s : ℝ) : (s, 7) ∈ line_through (0, 4) (-6, 1) → s = 6 :=
sorry

end find_s_l674_674285


namespace BQ_equals_BP_l674_674614

variables {P Q R S A B O : Point}
variables {circle : Circle}

-- Conditions
axiom distinct_points_on_circle (hP : P ∈ circle) (hQ : Q ∈ circle) (hR : R ∈ circle) (hS : S ∈ circle) :
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S

axiom diameter (hPS : P ≠ S) (h_diameter : ∃ (d : Diameter), d.pts = (P, S))

axiom parallel_diameter (hQR : Q ≠ R) (h_parallel : IsParallel QR PS)

axiom intersection_PR_QS (h_intersect : Intersect PR QS = A)

axiom center_of_circle (hcenter : center circle = O)

axiom parallelogram_POAB (h_parallelogram : IsParallelogram POAB)

-- Statement to prove
theorem BQ_equals_BP :
  BQ = BP :=
sorry

end BQ_equals_BP_l674_674614


namespace num_terms_simplified_expression_l674_674546

theorem num_terms_simplified_expression (x y z : ℕ) :
  let expr := ((x + y + z) ^ 2006 + (x - y - z) ^ 2006)
  term_count expr = 1008016 :=
begin
  -- The proof would go here
  sorry
end

end num_terms_simplified_expression_l674_674546


namespace smallest_value_of_expression_l674_674195

theorem smallest_value_of_expression :
  ∃ (k l : ℕ), 36^k - 5^l = 11 := 
sorry

end smallest_value_of_expression_l674_674195


namespace solution_count_eq_three_l674_674410

theorem solution_count_eq_three (x y : ℕ) (h : ⌊2.018 * x⌋ + ⌊5.13 * y⌋ = 24) :
  {p : ℕ × ℕ | ⌊2.018 * (p.1 : ℝ)⌋ + ⌊5.13 * (p.2 : ℝ)⌋ = 24}.to_finset.card = 3 := sorry

end solution_count_eq_three_l674_674410


namespace cosine_shift_half_unit_left_l674_674928

-- Main Theorem stating the graph transformation
theorem cosine_shift_half_unit_left (x : ℝ) : cos (2 * (x + 1/2)) = cos (2 * x + 1) :=
by sorry

end cosine_shift_half_unit_left_l674_674928


namespace correct_option_B_l674_674586

-- Definitions for the operations in the problem
def abs_val (x : ℤ) : ℤ := if x < 0 then -x else x
def neg_abs_val (x : ℤ) : ℤ := -abs_val x
def sqrt (x : ℕ) : ℕ := Nat.sqrt x
def power (x y : ℕ) : ℕ := x ^ y

-- Problem statement in Lean 4
theorem correct_option_B :
  neg_abs_val (-2) ≠ 2 ∧
  sqrt 16 = 4 ∧
  sqrt 9 ≠ ±3 ∧
  power 2 3 ≠ 6 →
  sqrt 16 = 4 :=
by
  sorry

end correct_option_B_l674_674586


namespace reciprocal_neg_7_l674_674147

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem reciprocal_neg_7 : reciprocal (-7) = -1 / 7 := 
by
  sorry

end reciprocal_neg_7_l674_674147


namespace ten_m_plus_n_eq_49_l674_674951

def centroid_m (A B C : ℝ × ℝ) : ℝ :=
  (A.1 + B.1 + C.1) / 3

def centroid_n (A B C : ℝ × ℝ) : ℝ :=
  (A.2 + B.2 + C.2) / 3

theorem ten_m_plus_n_eq_49 (A B C : ℝ × ℝ) (m n : ℝ)
  (hA : A = (5, 8))
  (hB : B = (3, -2))
  (hC : C = (6, 1))
  (hm : m = centroid_m A B C)
  (hn : n = centroid_n A B C) :
  10 * m + n = 49 :=
by
  simp [centroid_m, centroid_n, hA, hB, hC, hm, hn]
  sorry

end ten_m_plus_n_eq_49_l674_674951


namespace frac_sum_equals_seven_eights_l674_674824

theorem frac_sum_equals_seven_eights (p q r u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 := 
  sorry

end frac_sum_equals_seven_eights_l674_674824


namespace complement_of_intersection_l674_674807

-- Definitions of the sets M and N
def M : Set ℝ := { x | x ≥ 2 }
def N : Set ℝ := { x | x < 3 }

-- Definition of the intersection of M and N
def M_inter_N : Set ℝ := { x | 2 ≤ x ∧ x < 3 }

-- Definition of the complement of M ∩ N in ℝ
def complement_M_inter_N : Set ℝ := { x | x < 2 ∨ x ≥ 3 }

-- The theorem to be proved
theorem complement_of_intersection :
  (M_inter_Nᶜ) = complement_M_inter_N :=
by sorry

end complement_of_intersection_l674_674807


namespace factorization_t2_minus_97_l674_674727

theorem factorization_t2_minus_97 (t : ℝ) : t^2 - 97 = (t - real.sqrt 97) * (t + real.sqrt 97) :=
by sorry

end factorization_t2_minus_97_l674_674727


namespace sequence_periodic_from_some_term_l674_674313

def is_bounded (s : ℕ → ℤ) (M : ℤ) : Prop :=
  ∀ n, |s n| ≤ M

def is_periodic_from (s : ℕ → ℤ) (N : ℕ) (p : ℕ) : Prop :=
  ∀ n, s (N + n) = s (N + n + p)

theorem sequence_periodic_from_some_term (s : ℕ → ℤ) (M : ℤ) (h_bounded : is_bounded s M)
    (h_recurrence : ∀ n, s (n + 5) = (5 * s (n + 4) ^ 3 + s (n + 3) - 3 * s (n + 2) + s n) / (2 * s (n + 2) + s (n + 1) ^ 2 + s (n + 1) * s n)) :
    ∃ N p, is_periodic_from s N p := by
  sorry

end sequence_periodic_from_some_term_l674_674313


namespace acute_triangle_has_grid_vertex_inside_or_on_sides_l674_674499

theorem acute_triangle_has_grid_vertex_inside_or_on_sides
  {A B C : ℤ × ℤ} -- Points A, B, and C are vertices at grid points
  (h_acute : ∀ (α β γ : ℝ), 
    α + β + γ = π ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2) :
  ∃ D : ℤ × ℤ, 
  (D ≠ A ∧ D ≠ B ∧ D ≠ C) ∧
  (D ∈ (triangle_interior A B C ∪ triangle_border A B C)) := 
sorry

end acute_triangle_has_grid_vertex_inside_or_on_sides_l674_674499


namespace coeff_x6_expansion_l674_674963

theorem coeff_x6_expansion : 
  (∃ c : ℤ, (1 - 3 * (x : ℝ) ^ 2) ^ 5 = ∑ n in finset.range 6, c * x ^ 6) → c = -270 := 
sorry

end coeff_x6_expansion_l674_674963


namespace length_of_BC_l674_674243

noncomputable theory

open Real

def parabola (x : ℝ) : ℝ := x^2 + 4

theorem length_of_BC :
  ∃ (b : ℝ), 
    (b ≠ 0) ∧
    parabola b = parabola (-b) ∧
    (2 * b) ^ 2 - (4 * (4 - (b ^ 2))) = 0 ∧
    let BC_length := 2 * abs b in 
    BC_length = 10 :=
sorry

end length_of_BC_l674_674243


namespace sector_area_l674_674127

theorem sector_area (θ r : ℝ) (hθ : θ = 2) (hr : r = 1) :
  (1 / 2) * r^2 * θ = 1 :=
by
  -- Conditions are instantiated
  rw [hθ, hr]
  -- Simplification is left to the proof
  sorry

end sector_area_l674_674127


namespace phirme_sequence_form_l674_674121
noncomputable def fibonacci : ℕ → ℕ
| 0     => 1
| 1     => 1
| (n+2) => fibonacci (n+1) + fibonacci n

def is_phirme (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n, n ≥ 1 → a n + a (n + 1) = fibonacci (n + k)

theorem phirme_sequence_form (a : ℕ → ℕ) (k : ℕ) (d : ℕ) :
  is_phirme a k ↔ ∀ n, a n = fibonacci (n + k - 2) + (-1)^(n-1) * d :=
sorry

end phirme_sequence_form_l674_674121


namespace prove_2x_l674_674788

noncomputable def problem (x y z : ℕ) : Prop :=
  x > y ∧ y > z ∧ z = 3 ∧ 2 * x + 3 * y + 3 * z = 5 * y + 11

theorem prove_2x (x y z : ℕ) (h : problem x y z) : 2 * x = 10 :=
by
  cases h with h₁ h_rest
  cases h_rest with h₂ h_rest
  cases h_rest with h₃ h_eq
  rw [h₃] at *
  sorry

end prove_2x_l674_674788


namespace angle_AGH_is_90_l674_674436

variables {A B C H G : Type} [euclidean_space ℝ (fin 3)]
variables {triangle_ABC : Triangle ℝ} (acute_triangle : triangle_ABC.is_acute)
variables (AB_neq_AC : triangle_ABC.AB ≠ triangle_ABC.AC)
variables (H_orthocenter : H = triangle_ABC.orthocenter)
variables (G_centroid : G = triangle_ABC.centroid)
variables (area_condition : 1 / area (triangle_ABC.subtriangle H A B) +
                            1 / area (triangle_ABC.subtriangle H A C) = 
                            2 / area (triangle_ABC.subtriangle H B C))

theorem angle_AGH_is_90 (acute_triangle : triangle_ABC.is_acute)
    (AB_neq_AC : triangle_ABC.AB ≠ triangle_ABC.AC)
    (H_orthocenter : H = triangle_ABC.orthocenter)
    (G_centroid : G = triangle_ABC.centroid)
    (area_condition : 1 / area (triangle_ABC.subtriangle H A B) +
                      1 / area (triangle_ABC.subtriangle H A C) = 
                      2 / area (triangle_ABC.subtriangle H B C)) :
    inner_three_vectors.angle A G H = 90 := by sorry

end angle_AGH_is_90_l674_674436


namespace range_of_a_l674_674861

-- Sets A and B
def A : set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : set ℝ := {x : ℝ | x < a}

-- The proof problem
theorem range_of_a (a : ℝ) (h : A ∩ B a ≠ ∅) : -1 < a :=
by {
  sorry
}

end range_of_a_l674_674861


namespace find_a_for_inequality_l674_674152

theorem find_a_for_inequality (a : ℚ) :
  (∀ x : ℚ, (ax / (x - 1)) < 1 ↔ (x < 1 ∨ x > 2)) → a = 1/2 :=
by
  sorry

end find_a_for_inequality_l674_674152


namespace zeros_at_end_of_product_l674_674712

theorem zeros_at_end_of_product : 
  (35 = 5 * 7) → 
  (4900 = 2^2 * 5^2 * 7^2) → 
  (nat_trailing_zeros (35 * 4900) = 2) :=
by
  intro h1 h2 
  sorry

end zeros_at_end_of_product_l674_674712


namespace probability_top_face_odd_correct_l674_674248

noncomputable def octahedral_dice_probability_top_face_odd : ℚ :=
  let faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
  let dot_prob (n : ℕ) : ℚ := n / 36
  let odd_prob (n : ℕ) : ℚ := if n % 2 = 1 then 1 - dot_prob n else dot_prob n
  let total_prob : ℚ := (∑ n in faces, (1 / 8) * odd_prob n)
  total_prob

theorem probability_top_face_odd_correct :
  octahedral_dice_probability_top_face_odd = 77 / 144 :=
by
  -- Placeholder for the proof
  sorry

end probability_top_face_odd_correct_l674_674248


namespace convert_base_7_to_base_10_l674_674663

theorem convert_base_7_to_base_10 : 
  ∀ n : ℕ, (n = 3 * 7^2 + 2 * 7^1 + 1 * 7^0) → n = 162 :=
by
  intros n h
  rw [pow_zero, pow_one, pow_two] at h
  norm_num at h
  exact h

end convert_base_7_to_base_10_l674_674663


namespace hazel_eyed_brunettes_count_l674_674433

theorem hazel_eyed_brunettes_count :
  ∀ (num_students blondes green_eyed_blondes brunettes hazel_eyed : ℕ),
    num_students = 60 →
    brunettes = 35 →
    green_eyed_blondes = 20 →
    hazel_eyed = 25 →
    blondes = num_students - brunettes →
    hazel_eyed_brunettes = hazel_eyed - (blondes - green_eyed_blondes) →
    hazel_eyed_brunettes = 20 :=
by
  intros num_students blondes green_eyed_blondes brunettes hazel_eyed
         h_num_students h_brunettes h_green_eyed_blondes h_hazel_eyed
         h_blondes h_hazel_eyed_brunettes
  rw [h_num_students, h_brunettes, h_green_eyed_blondes, h_hazel_eyed,
      h_blondes, h_hazel_eyed_brunettes]
  sorry

end hazel_eyed_brunettes_count_l674_674433


namespace angle_between_vectors_l674_674406

variables {a b : ℝ}

-- Condition 1: |a| = sqrt(2)
def magnitude_a : ℝ := real.sqrt 2

-- Condition 2: |b| = 2
def magnitude_b : ℝ := 2

-- Condition 3: (a - b) ⊥ a
def perp_condition (a b : ℝ) : Prop :=
  (a - b) * a = 0

-- The proof statement
theorem angle_between_vectors (a b : ℝ) 
  (h₁ : |a| = magnitude_a) 
  (h₂ : |b| = magnitude_b) 
  (h₃ : perp_condition a b) : 
  real.angle a b = real.pi / 4 := 
sorry

end angle_between_vectors_l674_674406


namespace distance_from_center_to_line_l674_674714

-- Conditions
def circle_center : ℝ × ℝ := (0, 0)
def circle (x y : ℝ) : Prop := x^2 + y^2 = 2
def line (x y : ℝ) : Prop := y = x + Real.sqrt 2

-- The problem as a Lean 4 statement
theorem distance_from_center_to_line : 
  let d := |(1 : ℝ) * 0 + (-1 : ℝ) * 0 + Real.sqrt 2| / Real.sqrt (1^2 + (-1 : ℝ)^2) in
  d = 1 := sorry

end distance_from_center_to_line_l674_674714


namespace cube_root_expression_l674_674398

theorem cube_root_expression (N : ℝ) (h : N > 1) : 
  real.cbrt (N^2 * real.cbrt (N^3 * real.cbrt (N^2))) = N^(29 / 27) :=
sorry

end cube_root_expression_l674_674398


namespace problem_statement_l674_674786

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

def g (x : ℝ) (b : ℝ) : ℝ := (f x)^2 - 2 * b * (f x) + 3

theorem problem_statement :
  (∀ x, f x = Real.log x / Real.log 2) ∧ 
  (∀ b, (∃ (min_y : ℝ), min_y = 
            if b < 1 / 2 
            then (13 / 4) - b 
            else if 1 / 2 ≤ b ∧ b ≤ 4 
            then 3 - (b ^ 2) 
            else 19 - 8 * b) 
             ∧ 
           (∀ x, x ∈ Set.Icc (Real.sqrt 2) 16 → 
            g x b ≥ min_y)) :=
sorry

end problem_statement_l674_674786


namespace expected_worth_of_coin_flip_l674_674249

-- Define the probabilities and gains/losses
def prob_heads := 2 / 3
def prob_tails := 1 / 3
def gain_heads := 5 
def loss_tails := -9

-- Define the expected value calculation for a coin flip
def expected_value := prob_heads * gain_heads + prob_tails * loss_tails

-- The theorem to be proven
theorem expected_worth_of_coin_flip : expected_value = 1 / 3 :=
by sorry

end expected_worth_of_coin_flip_l674_674249


namespace line_equation_l674_674636

theorem line_equation (x y : ℝ) (h : (3 * (x + 2) - 4 * (y - 8)) = 0) :
    ∃ (m b : ℝ), y = m * x + b ∧ m = 3/4 ∧ b = 9.5 :=
by
  use 3/4, 9.5
  constructor
  · sorry  -- We skip the proof since it's not required
  · constructor
    · refl
    · refl

end line_equation_l674_674636


namespace paint_house_time_l674_674303

-- Define the condition that five people with 80% efficiency can paint in 8 hours.
def condition1 (efficiency : ℝ) (peopleCount : ℕ) (hours : ℝ) : Prop :=
  let n_eff := peopleCount * efficiency in
  n_eff * hours = 4 * 8

-- Define the condition that four people with the same efficiency are considered.
def condition2 (efficiency : ℝ) (peopleCount : ℕ) (hours : ℝ) : Prop :=
  let n_eff := peopleCount * efficiency in
  n_eff * hours = 32

-- Define the theorem to prove the time t for four less experienced people to paint the house.
theorem paint_house_time 
  (efficiency : ℝ := 0.8) 
  (peopleCount1 peopleCount2 : ℕ := 5 := 4) 
  (hours1 : ℝ := 8) 
  (t : ℝ) 
  (h1 : condition1 efficiency peopleCount1 hours1)
  (h2 : condition2 efficiency peopleCount2 t) :
  t = 10 :=
sorry

end paint_house_time_l674_674303


namespace contains_variable_in_denominator_l674_674660

theorem contains_variable_in_denominator (a x y : ℚ) (b : ℚ) : 
  (∃ z, 2 + 2 / z = 2 + 2 / a) ∧ (¬ ∃ z, (x - 2 * y) / 3 = 2 + 2 / a) ∧ (¬ ∃ z, 1 / 2 = 2 + 2 / a) ∧ (¬ ∃ z, 1 / 2 * (a + b) = 2 + 2 / a) := 
by 
  sorry

end contains_variable_in_denominator_l674_674660


namespace factorial_expression_simplification_l674_674258

theorem factorial_expression_simplification : (4 * 6! + 32 * 5!) / 7! = 4 / 3 := by
  sorry

end factorial_expression_simplification_l674_674258


namespace jill_second_bus_time_l674_674949

-- Define constants representing the times
def wait_time_first_bus : ℕ := 12
def ride_time_first_bus : ℕ := 30

-- Define a function to calculate the total time for the first bus
def total_time_first_bus (wait : ℕ) (ride : ℕ) : ℕ :=
  wait + ride

-- Define a function to calculate the time for the second bus
def time_second_bus (total_first_bus_time : ℕ) : ℕ :=
  total_first_bus_time / 2

-- The theorem to prove
theorem jill_second_bus_time : 
  time_second_bus (total_time_first_bus wait_time_first_bus ride_time_first_bus) = 21 := by
  sorry

end jill_second_bus_time_l674_674949


namespace maximum_perimeter_remaining_l674_674233

def rectangle := { length : ℝ, width : ℝ }

def original_paper : rectangle := { length := 20, width := 16 }
def smaller_paper : rectangle := { length := 8, width := 4 }

theorem maximum_perimeter_remaining 
  (h_align : ∃ (x y : rectangle), x = original_paper ∧ y = smaller_paper ∧
    (y.length = x.length ∨ y.width = x.width)) : 
  ∃ (p : ℝ), p = 88 := 
sorry

end maximum_perimeter_remaining_l674_674233


namespace domain_of_h_is_all_reals_except_5_l674_674577

def h : ℝ → ℝ := λ x, (4 * x - 2) / (x - 5)

noncomputable def domain_h (x : ℝ) : Prop := x ≠ 5

theorem domain_of_h_is_all_reals_except_5 :
  ∀ x : ℝ, domain_h x ↔ x ∈ (Set.Ioo Float.negInf 5).union (Set.Ioo 5 Float.posInf) :=
by
  intro x
  sorry

end domain_of_h_is_all_reals_except_5_l674_674577


namespace ratio_seven_l674_674015

theorem ratio_seven (a b : ℕ → ℝ) (S T : ℕ → ℝ) 
  (h1 : ∀n, S n = ∑ i in finset.range (n + 1), a i)
  (h2: ∀n, T n = ∑ i in finset.range (n + 1), b i)
  (h3: ∀n, S n / T n = (4 * n + 1:ℝ) / (3 * n - 1)) : 
  a 7 / b 7 = 53 / 38 :=
by
  sorry

end ratio_seven_l674_674015


namespace find_x_from_sequence_l674_674061

noncomputable def sequenceProblem (x y z k : ℤ) : Prop :=
  k = 1 ∧ z + k = -2 ∧ y + z = -1 ∧ x + y = 1 

theorem find_x_from_sequence : 
  ∃ x y z k : ℤ, sequenceProblem x y z k ∧ x = -1 :=
by
  exists (-1)
  exists 2
  exists (-3)
  exists 1
  simp [sequenceProblem]
  exact ⟨rfl, by simp⟩

end find_x_from_sequence_l674_674061


namespace find_time_l674_674495

def container := {r : ℝ} -- A cylindrical container is represented by its base radius

noncomputable def A : container := ⟨1⟩ -- Container A has a base radius of 1
noncomputable def B : container := ⟨2⟩ -- Container B has a base radius of 2
noncomputable def C : container := ⟨1⟩ -- Container C has a base radius of 1

def initial_height_A : ℝ := 1 -- Initial water height in container A is 1 cm

def pipe_height : ℝ := 5 -- Height of the pipes from the base is 5 cm

def initial_height_B : ℝ := 0 -- Initially, height in B is 0 cm

def rate_of_increase_B : ℝ := 5 / 12 / 0.5 -- B's height increases by 5/12 cm in 0.5 minutes

noncomputable def volumes_added (t : ℝ) : ℝ := rate_of_increase_B * t -- Volume added to B in t minutes

def water_height_A (t : ℝ) : ℝ := initial_height_A -- Height in A remains fixed initially since pipes equalize to the added volumes.

def water_height_B (t : ℝ) : ℝ := rate_of_increase_B * t -- Height in B at time t

def height_difference (t : ℝ) : ℝ := water_height_A(t) - water_height_B(t) -- Difference in height between A and B

theorem find_time 
    (t : ℝ) 
    (h_diff : height_difference(t) = 1 / 2) 
    : t = 3/5 ∨ t = 33/20 ∨ t = 171/40 :=
sorry

end find_time_l674_674495


namespace find_angle_l674_674784

theorem find_angle (a : ℝ) (h : (4 * sin 3, -4 * cos 3) = (4 * sin a, -4 * cos a)) :
  a = 3 - (Real.pi / 2) :=
sorry

end find_angle_l674_674784


namespace prob_zeros_not_adjacent_l674_674381

theorem prob_zeros_not_adjacent :
  let total_arrangements := (5.factorial : ℝ)
  let zeros_together_arrangements := (4.factorial : ℝ)
  let prob_zeros_together := (zeros_together_arrangements / total_arrangements)
  let prob_zeros_not_adjacent := 1 - prob_zeros_together
  prob_zeros_not_adjacent = 0.6 :=
by
  sorry

end prob_zeros_not_adjacent_l674_674381


namespace parallelogram_base_length_l674_674605

theorem parallelogram_base_length (b : ℝ) (h : ℝ) (area : ℝ) 
  (h_eq_2b : h = 2 * b) (area_eq_98 : area = 98) 
  (area_def : area = b * h) : b = 7 :=
by
  sorry

end parallelogram_base_length_l674_674605


namespace average_student_headcount_is_correct_l674_674261

noncomputable def average_student_headcount : ℕ :=
  let a := 11000
  let b := 10200
  let c := 10800
  let d := 11300
  (a + b + c + d) / 4

theorem average_student_headcount_is_correct :
  average_student_headcount = 10825 :=
by
  -- Proof will go here
  sorry

end average_student_headcount_is_correct_l674_674261


namespace isosceles_triangle_count_l674_674062

-- Definitions based on the given conditions
variable (A B C D E F G : Type)
variable [triangleABC : triangle A B C]
variable (ABcongruentAC : congruent AB AC)
variable (angle_ABC_eq_60 : ∠ ABC = 60)
variable (BD_bisects_ABC : angle_bisector BD ABC)
variable (D_on_AC : on D AC)
variable (DE_parallel_AB : parallel DE AB)
variable (E_on_BC : on E BC)
variable (EF_parallel_BD : parallel EF BD)
variable (F_on_AC : on F AC)
variable (GF_parallel_BC : parallel GF BC)
variable (G_on_AC : on G AC)

-- The main theorem statement (lean statement) to prove the count of isosceles triangles
theorem isosceles_triangle_count : number_of_isosceles_triangles A B C D E F G = 6 :=
by
  -- proof to be filled in
  sorry

end isosceles_triangle_count_l674_674062


namespace centrally_symmetric_convex_polygon_division_l674_674599

open_locale classical

def convex_polygon (A : finset (ℝ × ℝ)) : Prop :=
  ∀ x y ∈ A, Ioo (x + (y - x) / 2) A ⊆ Icc x y

def centrally_symmetric (A : finset (ℝ × ℝ)) (c : ℝ × ℝ) : Prop :=
  ∀ x ∈ A, (2 * c - x) ∈ A

noncomputable def number_of_parallelograms (k : ℕ) : ℕ :=
  k * (k - 1) / 2

theorem centrally_symmetric_convex_polygon_division
  (A : finset (ℝ × ℝ)) (k : ℕ) (c : ℝ × ℝ)
  (h_convex : convex_polygon A) (h_symmetric : centrally_symmetric A c) :
  ∃ P : finset (finset (ℝ × ℝ)), (∀ p ∈ P, is_parallelogram p) ∧ P.card = number_of_parallelograms k :=
sorry

end centrally_symmetric_convex_polygon_division_l674_674599


namespace probability_non_adjacent_zeros_l674_674383

theorem probability_non_adjacent_zeros (total_ones total_zeros : ℕ) (h₁ : total_ones = 3) (h₂ : total_zeros = 2) : 
  (total_zeros != 0 ∧ total_ones != 0 ∧ total_zeros + total_ones = 5) → 
  (prob_non_adjacent (total_ones + total_zeros) total_zeros = 0.6) :=
by
  sorry

def prob_non_adjacent (total num_zeros: ℕ) : ℚ :=
  let total_arrangements := (Nat.factorial total) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros)))
  let adjacent_arrangements := (Nat.factorial (total - num_zeros + 1)) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros - 1)))
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements / total_arrangements

end probability_non_adjacent_zeros_l674_674383


namespace stratified_sampling_males_l674_674655

theorem stratified_sampling_males (
  total_male : ℕ := 48,
  total_female : ℕ := 36,
  sample_size : ℕ := 21
  ) : 
  let total_athletes := total_male + total_female,
      proportion_male := total_male / total_athletes in
  (proportion_male * sample_size = 12) :=
by
  skip_proof_step -- skip is used to illustrate the notion of skipping problem steps in this context
  sorry

end stratified_sampling_males_l674_674655


namespace smallest_n_value_l674_674870

def conditions_met (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (∀ i, 0 ≤ i ∧ i < n → |x i| < 1) ∧
  (∑ i in Finset.range n, |x i| = 17 + |∑ i in Finset.range n, x i|)

theorem smallest_n_value :
  ∃ (x : ℕ → ℝ), conditions_met 18 x :=
sorry

end smallest_n_value_l674_674870


namespace boat_speed_in_still_water_l674_674185

theorem boat_speed_in_still_water:
  ∀ (V_b : ℝ) (V_s : ℝ) (D : ℝ),
    V_s = 3 → 
    (D = (V_b + V_s) * 1) → 
    (D = (V_b - V_s) * 1.5) → 
    V_b = 15 :=
by
  intros V_b V_s D V_s_eq H_downstream H_upstream
  sorry

end boat_speed_in_still_water_l674_674185


namespace number_of_even_products_l674_674892

-- Setting up the range for rows and columns
def rows : List ℕ := List.range' 12 49  -- 49 elements starting from 12
def columns : List ℕ := List.range' 15 26  -- 26 elements starting from 15

-- Function to determine if a number is even
def is_even (n : ℕ) : Bool := n % 2 = 0

-- Function to count how many products are even in a given table
def count_even_products (rows columns : List ℕ) : ℕ :=
  List.foldl (λ acc r, acc + (List.foldl (λ sub_acc c, sub_acc + (if is_even (r * c) then 1 else 0)) 0 columns)) 0 rows

-- Total even products
theorem number_of_even_products : count_even_products rows columns = 962 :=
by sorry

end number_of_even_products_l674_674892


namespace probability_zeros_not_adjacent_is_0_6_l674_674397

-- Define the total number of arrangements of 5 elements where we have 3 ones and 2 zeros
def total_arrangements : Nat := 5.choose 2

-- Define the number of arrangements where 2 zeros are adjacent
def adjacent_zeros_arrangements : Nat := 4.choose 1 * 2

-- Define the probability that the 2 zeros are not adjacent
def probability_not_adjacent : Rat := (total_arrangements - adjacent_zeros_arrangements) / total_arrangements

-- Prove the desired probability is 0.6
theorem probability_zeros_not_adjacent_is_0_6 : probability_not_adjacent = 3 / 5 := by
  sorry

end probability_zeros_not_adjacent_is_0_6_l674_674397


namespace solve_system1_solve_system2_l674_674115

theorem solve_system1 (x y : ℝ) (h1 : 2 * x + 3 * y = 9) (h2 : x = 2 * y + 1) : x = 3 ∧ y = 1 := 
by sorry

theorem solve_system2 (x y : ℝ) (h1 : 2 * x - y = 6) (h2 : 3 * x + 2 * y = 2) : x = 2 ∧ y = -2 := 
by sorry

end solve_system1_solve_system2_l674_674115


namespace problem_1_problem_2_l674_674622

noncomputable theory

-- Problem (Ⅰ)
variables (a b : ℝ → ℝ) (u v : ℝ → ℝ) (AB BC : ℝ) (angle_B : ℝ)
variables (AB_eq : AB = 1) (BC_eq : BC = 2) (angle_B_eq : angle_B = real.pi / 3) 
          (AB_vec : ∀ x, AB * a x = AB * a x) (BC_vec : ∀ x, BC * b x = BC * b x)

def dot_product (a b : ℝ → ℝ) : ℝ :=
2 * forall fin.vec cons (0,0) ∧ 1 − 2 * forall value real

theorem problem_1 : (2 • a - 3 • b) • (4 • a + b) = 6 :=
sorry

-- Problem (Ⅱ)
variables (a b : ℝ × ℝ) (t: ℝ)
variables (a_def : a = (2, 1)) (b_def : b = (-1, 3))

theorem problem_2 : ∃ t: ℝ, (t • a + b) = ({1, 3}, -2) := -1 :=
sorry

end problem_1_problem_2_l674_674622


namespace count_solutions_eq_946_l674_674297

theorem count_solutions_eq_946 :
  ∃ (x y z w : ℕ), x + y + z + w = 25 ∧ x < y ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 → (∑ x in finset.range 12, nat.choose (24 - 2 * x) 2) = 946 :=
sorry

end count_solutions_eq_946_l674_674297


namespace garden_perimeter_l674_674608

theorem garden_perimeter 
  (garden_width : ℕ)
  (playground_length : ℕ)
  (playground_width : ℕ)
  (playground_area : ℕ)
  (garden_length : ℕ)
  (garden_area : ℕ) 
  (area_eq : garden_area = playground_area)
  (area_playground : playground_area = playground_length * playground_width)
  (area_garden : garden_area = garden_length * garden_width)
  : garden_width = 24 ∧ playground_length = 16 ∧ playground_width = 12 ∧ garden_length = 8 ∧ perimeter = 64 :=
by
  -- Use the conditions given
  have garden_width_def : garden_width = 24 := rfl
  have playground_length_def : playground_length = 16 := rfl
  have playground_width_def : playground_width = 12 := rfl
  have playground_area_def : playground_area = playground_length * playground_width := area_playground
  have garden_area_def : garden_area = garden_length * garden_width := area_garden
  have garden_area_eq : garden_area = playground_area := area_eq
  
  -- Prove garden length using given conditions
  have garden_length_def : garden_length = 192 / 24 := sorry  -- This step to be filled in proof
  
  -- Prove perimeter
  have perimeter_def : 2 * (garden_length + garden_width) = 64 := sorry  -- This step to be filled in proof
  
  -- Combining all definitions and results
  exact ⟨garden_width_def, playground_length_def, playground_width_def, garden_length_def, perimeter_def⟩

end garden_perimeter_l674_674608


namespace jenna_weight_lift_l674_674853

theorem jenna_weight_lift:
  ∀ (n : Nat), (2 * 10 * 25 = 500) ∧ (15 * n >= 500) ∧ (n = Nat.ceil (500 / 15 : ℝ))
  → n = 34 := 
by
  intros n h
  have h₀ : 2 * 10 * 25 = 500 := h.1
  have h₁ : 15 * n >= 500 := h.2.1
  have h₂ : n = Nat.ceil (500 / 15 : ℝ) := h.2.2
  sorry

end jenna_weight_lift_l674_674853


namespace find_lambda_l674_674017

noncomputable def vector_a: ℝ × ℝ := (12, -5)

def vector_b (λ: ℝ): ℝ × ℝ := (12 * λ, -5 * λ)

def magnitude (v: ℝ × ℝ): ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_lambda (λ: ℝ) (h1: λ < 0) (h2: magnitude (vector_b λ) = 13):
  λ = -1 :=
sorry

end find_lambda_l674_674017


namespace charlie_older_than_bobby_by_three_l674_674453

variable (J C B x : ℕ)

def jenny_older_charlie_by_five (J C : ℕ) := J = C + 5
def charlie_age_when_jenny_twice_bobby_age (C x : ℕ) := C + x = 11
def jenny_twice_bobby (J B x : ℕ) := J + x = 2 * (B + x)

theorem charlie_older_than_bobby_by_three
  (h1 : jenny_older_charlie_by_five J C)
  (h2 : charlie_age_when_jenny_twice_bobby_age C x)
  (h3 : jenny_twice_bobby J B x) :
  (C = B + 3) :=
by
  sorry

end charlie_older_than_bobby_by_three_l674_674453


namespace book_selection_l674_674821

def num_books_in_genre (mystery fantasy biography : ℕ) : ℕ :=
  mystery + fantasy + biography

def num_combinations_two_diff_genres (mystery fantasy biography : ℕ) : ℕ :=
  if mystery = 4 ∧ fantasy = 4 ∧ biography = 4 then 48 else 0

theorem book_selection : 
  ∀ (mystery fantasy biography : ℕ),
  num_books_in_genre mystery fantasy biography = 12 →
  num_combinations_two_diff_genres mystery fantasy biography = 48 :=
by
  intros mystery fantasy biography h
  sorry

end book_selection_l674_674821


namespace reciprocal_neg_7_l674_674148

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem reciprocal_neg_7 : reciprocal (-7) = -1 / 7 := 
by
  sorry

end reciprocal_neg_7_l674_674148


namespace part_a_part_b_l674_674598

namespace ProofProblems

-- Part (a)
theorem part_a (p : ℕ) (hp : p.prime) (hp_not_3 : p ≠ 3) :
  ¬ p ∣ (10^p - 1) / 9 := by
  sorry

-- Part (b)
theorem part_b (p : ℕ) (hp : p.prime) (hp_gt_5 : p > 5) :
  p ∣ (10^(p-1) - 1) / 9 := by
  sorry

end ProofProblems

end part_a_part_b_l674_674598


namespace find_n_l674_674961

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 101) (h3 : 100 * n % 101 = 72) : n = 29 := 
by
  sorry

end find_n_l674_674961


namespace find_ratio_l674_674463

-- Define geometric sequence and its sums
def geom_seq (a₁ q : ℚ) (n : ℕ) : ℚ :=
  a₁ * q ^ (n - 1)

def sum_geom_seq (a₁ q : ℚ) (n : ℕ) : ℚ :=
  if q = 1 then n * a₁ else a₁ * (1 - q ^ n) / (1 - q)

-- Define the conditions provided in the problem
def condition (a₁ q : ℚ) : Prop :=
  geom_seq a₁ q 3 + 2 * geom_seq a₁ q 6 = 0

-- Define the target statement to prove
theorem find_ratio (a₁ q : ℚ) (h : condition a₁ q) :
  sum_geom_seq a₁ q 3 / sum_geom_seq a₁ q 6 = 2 :=
begin
  -- This is where you'd normally prove the theorem, but we'll skip it as requested.
  sorry,
end

end find_ratio_l674_674463


namespace perpendicular_vectors_k_value_parallel_vector_coordinates_l674_674060

-- Problem (1)
theorem perpendicular_vectors_k_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (-1, 2)
  let c : ℝ × ℝ := (4, 1)
  let lhs : ℝ × ℝ := ⟨3 + 4 * k, 2 + k⟩
  let rhs : ℝ × ℝ := ⟨-5, 2⟩
  (lhs.fst * rhs.fst + lhs.snd * rhs.snd = 0) → k = -(11 / 18) :=
sorry

-- Problem (2)
theorem parallel_vector_coordinates :
  let c : ℝ × ℝ := (4, 1)
  let d : ℝ × ℝ := if (x^2 + y^2 = 34 ∧ x = 4 * y) then ⟨4 * Real.sqrt(2), Real.sqrt(2)⟩ else ⟨-4 * Real.sqrt(2), -Real.sqrt(2)⟩
  (d.fst^2 + d.snd^2 = 34 ∧ d.fst = 4 * d.snd ∨ d.fst = -4 * d.snd) :=
sorry

end perpendicular_vectors_k_value_parallel_vector_coordinates_l674_674060


namespace solve_equation_l674_674902

theorem solve_equation (x : ℝ) (h : (x - 1) / 2 = 1 - (x + 2) / 3) : x = 1 :=
sorry

end solve_equation_l674_674902


namespace sum_factors_30_less_15_l674_674582

theorem sum_factors_30_less_15 : (1 + 2 + 3 + 5 + 6 + 10) = 27 := by
  sorry

end sum_factors_30_less_15_l674_674582


namespace calc_expression_l674_674677

theorem calc_expression : (-1 : ℝ)^2 + |1 - real.sqrt 2| + (real.pi - 3.14)^0 - (1 / 2)^(-1 : ℝ) = real.sqrt 2 - 1 :=
by
  sorry

end calc_expression_l674_674677


namespace probability_margo_pairing_l674_674424

-- Definition of the problem
def num_students : ℕ := 32
def num_pairings (n : ℕ) : ℕ := n - 1
def favorable_pairings : ℕ := 2

-- Theorem statement
theorem probability_margo_pairing :
  num_students = 32 →
  ∃ (p : ℚ), p = favorable_pairings / num_pairings num_students ∧ p = 2/31 :=
by
  intros h
  -- The proofs are omitted for brevity.
  sorry

end probability_margo_pairing_l674_674424


namespace non_similar_regular_500_pointed_stars_l674_674988

noncomputable def phi (n : ℕ) : ℕ :=
  n * (1 - (1 / 2)) * (1 - (1 / 5))

theorem non_similar_regular_500_pointed_stars :
  let n := 500
  let count := (phi n) - 2
  let non_similar_stars := count / 2
  non_similar_stars = 99 :=
by
  let n := 500
  let phi_n := 200 -- Given by \(\phi(500)\)
  let count := phi_n - 2
  let non_similar := count / 2
  have : non_similar = 99 := sorry
  assumption

end non_similar_regular_500_pointed_stars_l674_674988


namespace cube_diagonal_side_angle_60_l674_674925

/-- In a cube, the lines forming a 60-degree angle are the diagonals of the faces of the cube. -/
def angle_between_diagonal_and_side_in_cube : Real := 60

/-- Define what it means for lines in a cube to make a given angle. -/
def lines_forming_given_angle (a b c : ℝ) (angle : ℝ) : Prop :=
  ∃ (x y : ℝ), x * x + y * y = a * a + b * b + c * c - 2 * a * b * (Real.cos angle)

/-- Prove the angle between a diagonal of a face of the cube and a side of the cube is 60 degrees. -/
theorem cube_diagonal_side_angle_60 (a : ℝ) (ha : a > 0) :
  ∃ x : ℝ, lines_forming_given_angle (a * √2) a a angle_between_diagonal_and_side_in_cube :=
begin
  sorry -- Proof steps to be filled later
end

end cube_diagonal_side_angle_60_l674_674925


namespace allocation_schemes_count_l674_674368

theorem allocation_schemes_count (schools total_people : ℕ) (at_least_one_per_school : ℕ) :
  schools = 7 → total_people = 10 → at_least_one_per_school = 1 → 
  (∑ i in Finset.range schools, at_least_one_per_school) + 3 = total_people → 
  Nat.choose (total_people - 1) (schools - 1) = 84 :=
by
  intros h_schools h_people h_min selection_eq
  replace selection_eq : total_people - schools = 3 := by linarith [h_people, selection_eq]
  have : Nat.choose 9 6 = 84 := by
    sorry
  exact this

end allocation_schemes_count_l674_674368


namespace cos_eq_cos_is_necessary_but_not_sufficient_l674_674027

theorem cos_eq_cos_is_necessary_but_not_sufficient (x y : ℝ) : 
  (cos x = cos y) → ¬ (x = y) := 
by {
  sorry
}

end cos_eq_cos_is_necessary_but_not_sufficient_l674_674027


namespace find_p_plus_q_l674_674929

noncomputable def f (k p : ℚ) : ℚ := 5 * k^2 - 2 * k + p
noncomputable def g (k q : ℚ) : ℚ := 4 * k^2 + q * k - 6

theorem find_p_plus_q (p q : ℚ) (h : ∀ k : ℚ, f k p * g k q = 20 * k^4 - 18 * k^3 - 31 * k^2 + 12 * k + 18) :
  p + q = -3 :=
sorry

end find_p_plus_q_l674_674929


namespace repeating_decimal_fraction_l674_674583

theorem repeating_decimal_fraction :
  (\exists y : ℚ, y = 0.73 + 264/999900) ↔ (\exists z : ℚ, z = 732635316) :=
by
  sorry

end repeating_decimal_fraction_l674_674583


namespace sqrt3_ineq_l674_674736

theorem sqrt3_ineq (x : ℝ) (h : x > 0) : (∛x < 3 * x) ↔ (x > 1 / (3 * sqrt 3)) :=
sorry

end sqrt3_ineq_l674_674736


namespace probability_sum_7_is_1_over_3_l674_674164

def odd_die : Set ℕ := {1, 3, 5}
def even_die : Set ℕ := {2, 4, 6}

noncomputable def total_outcomes : ℕ := 6 * 6

noncomputable def favorable_outcomes : ℕ := 4 + 4 + 4

noncomputable def probability_sum_7 : ℚ := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_sum_7_is_1_over_3 :
  probability_sum_7 = 1 / 3 :=
by
  sorry

end probability_sum_7_is_1_over_3_l674_674164


namespace find_denomination_l674_674068

variable (initial_amount now_amount : ℕ) (num_bills : ℕ) (denomination : ℕ)

-- Define the given conditions
noncomputable def initial_amount := 75
noncomputable def now_amount := 135
noncomputable def num_bills := 3

-- Formulate the problem to find the specific denomination of the bills
theorem find_denomination : denomination = (now_amount - initial_amount) / num_bills :=
sorry

end find_denomination_l674_674068


namespace can_spend_all_money_l674_674177

theorem can_spend_all_money (n : Nat) (h : n > 7) : 
  ∃ (x y : Nat), 3 * x + 5 * y = n :=
by
  sorry

end can_spend_all_money_l674_674177


namespace minimize_on_interval_l674_674734

def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 2

theorem minimize_on_interval (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x a ≥ if a < 0 then -2 else if 0 ≤ a ∧ a ≤ 2 then -a^2 - 2 else 2 - 4*a) :=
by 
  sorry

end minimize_on_interval_l674_674734


namespace binary_110_eq_6_l674_674616

-- Define the binary number
def binary_110 := [1, 1, 0]  -- binary representations as list of digits

-- Function to convert binary list to decimal
def binary_to_decimal (b : List ℕ) : ℕ :=
  b.foldl (λ acc d, acc * 2 + d) 0

-- Prove that the decimal conversion of binary 110 is 6
theorem binary_110_eq_6 : binary_to_decimal binary_110 = 6 :=
by simp [binary_to_decimal, binary_110]; sorry

end binary_110_eq_6_l674_674616


namespace distance_between_foci_of_ellipse_l674_674247

theorem distance_between_foci_of_ellipse : 
  ∃ (a b: ℝ), 
    a = 6 ∧ 
    b = 2 ∧ 
    (2 * real.sqrt (a^2 - b^2)) = 8 * real.sqrt 2 :=
by
  exists 6, 2
  constructor; 
  { refl } 
  constructor;
  { refl } 
  { sorry }

end distance_between_foci_of_ellipse_l674_674247


namespace prob_all_three_not_win_prob_at_least_two_not_win_l674_674045

noncomputable def prob_not_win (p_win : ℝ) : ℝ := 1 - p_win

theorem prob_all_three_not_win
    (p_win : ℝ)
    (p_win = 1 / 6) :
    ((prob_not_win p_win) ^ 3) = 125 / 216 := by
sorry

theorem prob_at_least_two_not_win
    (p_win : ℝ)
    (p_win = 1 / 6) :
    1 - (3 * (p_win ^ 2) * (prob_not_win p_win) + (p_win ^ 3)) = 25 / 27 := by
sorry

end prob_all_three_not_win_prob_at_least_two_not_win_l674_674045


namespace outdoor_tables_count_l674_674537

variable (numIndoorTables : ℕ) (chairsPerIndoorTable : ℕ) (totalChairs : ℕ)
variable (chairsPerOutdoorTable : ℕ)

theorem outdoor_tables_count 
  (h1 : numIndoorTables = 8) 
  (h2 : chairsPerIndoorTable = 3) 
  (h3 : totalChairs = 60) 
  (h4 : chairsPerOutdoorTable = 3) :
  ∃ (numOutdoorTables : ℕ), numOutdoorTables = 12 := by
  admit

end outdoor_tables_count_l674_674537


namespace shakes_indeterminable_l674_674672

variable {B S C x : ℝ}

theorem shakes_indeterminable (h1 : 3 * B + x * S + C = 130) (h2 : 4 * B + 10 * S + C = 164.5) : 
  ¬ (∃ x, 3 * B + x * S + C = 130 ∧ 4 * B + 10 * S + C = 164.5) :=
by
  sorry

end shakes_indeterminable_l674_674672


namespace sqrt_sum_of_powers_of_2_l674_674968

theorem sqrt_sum_of_powers_of_2 : Real.sqrt (4 * (2^4)) = 8 := 
by 
  rw [pow_succ, mul_assoc] 
  norm_num 
  rw [Real.sqrt_eq_rpow, ← Real.rpow_nat_cast] 
  norm_num 
  sorry

end sqrt_sum_of_powers_of_2_l674_674968


namespace card_number_factors_l674_674157

theorem card_number_factors {A B : ℕ} 
  (hB : B = 3 * A)
  (cards_digits_sum_eq : 10 * (1 + 2 + ... + 9) = 450)
  (sum_of_digits_div_by_3 : (∃ k, cards_digits_sum_eq = 3 * k)) :
  ∃ a b c d : ℕ, a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ B = a * b * c * d :=
by
  sorry

end card_number_factors_l674_674157


namespace greatest_n_factor_3_in_16_factorial_greatest_n_factor_3_in_16_factorial_exact_l674_674602

theorem greatest_n_factor_3_in_16_factorial (n : ℕ) (m := 3 ^ n) : (3 ^ n ∣ 16.factorial) → n ≤ 6 := 
sorry

theorem greatest_n_factor_3_in_16_factorial_exact (n : ℕ) (m := 3 ^ n) : (3 ^ 6 ∣ 16.factorial) ∧ ¬ (3 ^ (6 + 1) ∣ 16.factorial) := 
sorry

end greatest_n_factor_3_in_16_factorial_greatest_n_factor_3_in_16_factorial_exact_l674_674602


namespace smallest_positive_solution_exists_l674_674300

open Nat

theorem smallest_positive_solution_exists :
  ∃ x : ℕ, 
    (x % 12 = 9 % 12) ∧
    (x * 5 + 4) % 7 = 14 % 7) ∧
    (4 * x - 3) % 17 = (2 * x + 5) % 17) ∧
    (x % 11 = 4 % 11) ∧
    x = 1309 :=
by sorry

end smallest_positive_solution_exists_l674_674300


namespace find_value_of_a_l674_674828

-- Definitions based on the conditions
def x (k : ℕ) : ℕ := 3 * k
def y (k : ℕ) : ℕ := 4 * k
def z (k : ℕ) : ℕ := 6 * k

-- Setting up the sum equation
def sum_eq_52 (k : ℕ) : Prop := x k + y k + z k = 52

-- Defining the y equation
def y_eq (a : ℚ) (k : ℕ) : Prop := y k = 15 * a + 5

-- Stating the main problem
theorem find_value_of_a (a : ℚ) (k : ℕ) : sum_eq_52 k → y_eq a k → a = 11 / 15 := by
  sorry

end find_value_of_a_l674_674828


namespace pythagorean_triple_l674_674983

theorem pythagorean_triple (k p q : ℕ) (hk : k > 0)
  (hpq_coprime : Nat.coprime p q) (h_parity : ¬(Nat.even p ∧ Nat.even q) ∧ ¬(Nat.odd p ∧ Nat.odd q)) :
  let x := 2 * k * p * q
  let y := k * (p^2 - q^2)
  let z := k * (p^2 + q^2)
  x^2 + y^2 = z^2 := by
  sorry

end pythagorean_triple_l674_674983


namespace manufacturing_department_degrees_l674_674123

def percentage_of_circle (percentage : ℕ) (total_degrees : ℕ) : ℕ :=
  (percentage * total_degrees) / 100

theorem manufacturing_department_degrees :
  percentage_of_circle 30 360 = 108 :=
by
  sorry

end manufacturing_department_degrees_l674_674123


namespace smallest_q_exists_l674_674085

theorem smallest_q_exists (p q : ℕ) (h : 0 < q) (h_eq : (p : ℚ) / q = 123456789 / 100000000000) :
  q = 10989019 :=
sorry

end smallest_q_exists_l674_674085


namespace parabola_equation_l674_674291

theorem parabola_equation :
  ∃ a b c : ℝ, (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y = -3 * x^2 + 18 * x - 22) ∧
    (∃ a : ℝ, ∀ x : ℝ, y = a * (x - 3) ^ 2 + 5 ↔ ∀ x y : ℝ, y = a * (x - 3)^2 + 5) ∧
    ∃ x y : ℝ, (x = 2 ∧ y = 2) → y = a * (x - 3)^2 + 5 :=
begin
  sorry
end

end parabola_equation_l674_674291


namespace sqrt2_mul_sqrt12_minus_2_in_range_l674_674282

theorem sqrt2_mul_sqrt12_minus_2_in_range : 2 < sqrt 2 * sqrt 12 - 2 ∧ sqrt 2 * sqrt 12 - 2 < 3 := 
by 
  sorry

end sqrt2_mul_sqrt12_minus_2_in_range_l674_674282


namespace product_of_real_roots_eq_neg_two_l674_674588

theorem product_of_real_roots_eq_neg_two :
  let f := (λ x : ℝ, x^4 + 3 * x^3 + 5 * x^2 + 21 * x - 14)
  (∀ r : ℝ, polynomial.root (polynomial.C (1 : ℝ) * polynomial.X^4 + 
                                 polynomial.C (3 : ℝ) * polynomial.X^3 + 
                                 polynomial.C (5 : ℝ) * polynomial.X^2 + 
                                 polynomial.C (21 : ℝ) * polynomial.X + polynomial.C (-14))) →
    polynomial.eval r f = 0 →
  let real_roots := {r : ℝ | polynomial.eval r f = 0} in
  real_roots.prod id = -2 :=
sorry

end product_of_real_roots_eq_neg_two_l674_674588


namespace line_through_point_parallel_to_given_line_l674_674621

theorem line_through_point_parallel_to_given_line :
  ∃ m : ℝ, ∀ x y : ℝ, (x - 2 * y + 7 = 0) ↔ (m = 7) :=
by
  let p := (-1 : ℝ, 3 : ℝ)
  let line := λ (x y : ℝ) => x - 2 * y + 3 = 0
  have parallel_condition := λ m => 
    ∃ (x y : ℝ), (x = -1 ∧ y = 3 ∧ (x - 2 * y + m = 0))
  use 7
  sorry

end line_through_point_parallel_to_given_line_l674_674621


namespace expected_value_correct_l674_674626

-- Define the parameters for the problem
def number_of_teams : ℕ := 6
def total_pairs : ℕ := (number_of_teams * (number_of_teams - 1)) / 2

-- Define the expected value calculation
noncomputable def expected_undefeated_teams : ℚ :=
  2^(-total_pairs) * (∑ i in (finset.range (number_of_teams + 1)).filter (1 <= ·),
    (nat.choose number_of_teams i : ℚ) * (i^(i-2) : ℚ) * 2^((total_pairs - (i * (i - 1)) / 2)))

-- The correct answer
def correct_answer : ℚ := 5055 / (2^total_pairs)

-- The final proof statement
theorem expected_value_correct :
  expected_undefeated_teams = correct_answer := by
    sorry -- Proof skipped

end expected_value_correct_l674_674626


namespace cylinder_volume_ratio_l674_674236

theorem cylinder_volume_ratio (s : ℝ) (h1 : s > 0):
  let V_cube := s^3,
      r := s / 2,
      V_cylinder := π * r^2 * s
  in (V_cylinder / V_cube) = π / 4 :=
by
  let V_cube := s^3
  let r := s / 2
  let V_cylinder := π * (r^2) * s
  have h2 : V_cylinder = π * (s / 2)^2 * s := by rfl
  have h3 : (s / 2)^2 = s^2 / 4 := by norm_num
  have h4 : V_cylinder = π * (s^2 / 4) * s := by rw [h2, h3]
  have h5 : V_cylinder = π * s^3 / 4 := by ring
  have h6 : (V_cylinder / V_cube) = (π * s^3 / 4) / s^3 := by rw [h5]
  have h7 : (π * s^3 / 4) / s^3 = π / 4 := by field_simp
  exact h7

end cylinder_volume_ratio_l674_674236


namespace find_y_value_l674_674302

theorem find_y_value (y : ℝ) (h : 12^2 * y^3 / 432 = 72) : y = 6 :=
by
  sorry

end find_y_value_l674_674302


namespace class_size_l674_674840

theorem class_size :
  ∃ n : ℕ, 
    let total_score := 4 * 95 + 3 * 0 + (n - 7) * 45,
        avg_score := total_score / n in
    avg_score = 47.32142857142857 ∧ n = 28 :=
by
  sorry

end class_size_l674_674840


namespace circles_common_tangents_l674_674915

noncomputable def distance (p q : Real × Real) : Real :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def c1_center : Real × Real := (2, 1)
def c1_radius : Real := 2
def c2_center : Real × Real := (-1, 2)
def c2_radius : Real := 3

def number_of_common_tangents (center1 center2 : Real × Real) (r1 r2 : Real) : Nat :=
  let dist := distance center1 center2
  if dist > r1 + r2 ∨ dist < Real.abs (r1 - r2) then 0
  else if dist == r1 + r2 ∨ dist == Real.abs (r1 - r2) then 1
  else 2

theorem circles_common_tangents : number_of_common_tangents c1_center c2_center c1_radius c2_radius = 2 :=
by
  unfold number_of_common_tangents
  have distance_centers : distance c1_center c2_center = Real.sqrt 10 := by sorry
  rw [distance_centers]
  -- The actual steps to simplify and show the result will go here
  have h1 : c1_radius > 0 := by sorry
  have h2 : c2_radius > 0 := by sorry
  have abs_diff : Real.abs (c1_radius - c2_radius) = 1 := by sorry
  have sum_rad : c1_radius + c2_radius = 5 := by sorry
  have h3 : Real.sqrt 10 < 5 := by sorry
  have h4 : 1 < Real.sqrt 10 := by sorry
  simp only [h3, h4, if_false]
  exact rfl

end circles_common_tangents_l674_674915


namespace smallest_possible_sum_l674_674927

theorem smallest_possible_sum (A B C D : ℤ) 
  (h1 : A + B = 2 * C)
  (h2 : B * D = C * C)
  (h3 : 3 * C = 7 * B)
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D) : 
  A + B + C + D = 76 :=
sorry

end smallest_possible_sum_l674_674927


namespace ball_hits_ground_at_t_l674_674919

noncomputable def ball_height (t : ℝ) : ℝ := -6 * t^2 - 10 * t + 56

theorem ball_hits_ground_at_t :
  ∃ t : ℝ, ball_height t = 0 ∧ t = 7 / 3 := by
  sorry

end ball_hits_ground_at_t_l674_674919


namespace parabola_equation_l674_674290

theorem parabola_equation :
  ∃ a b c : ℝ, (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y = -3 * x^2 + 18 * x - 22) ∧
    (∃ a : ℝ, ∀ x : ℝ, y = a * (x - 3) ^ 2 + 5 ↔ ∀ x y : ℝ, y = a * (x - 3)^2 + 5) ∧
    ∃ x y : ℝ, (x = 2 ∧ y = 2) → y = a * (x - 3)^2 + 5 :=
begin
  sorry
end

end parabola_equation_l674_674290


namespace value_range_of_function_l674_674938

noncomputable def f : ℝ → ℝ := λ x, x^2 - 2 * x + 3

theorem value_range_of_function : 
  ∀ x, -1 ≤ x ∧ x ≤ 2 → 2 ≤ f x ∧ f x ≤ 6 :=
sorry

end value_range_of_function_l674_674938


namespace find_rate_of_interest_l674_674597

-- Define the initial sum lent out
def principal : ℝ := 468.75

-- Define the total amount received after 1 year and 8 months
def amount_received : ℝ := 500

-- Define the interest earned
def interest : ℝ := amount_received - principal

-- Convert 1 year and 8 months into years
def time_in_years : ℝ := 1 + (8 / 12)

-- Define the rate of interest (which we need to prove is 4%)
def rate : ℝ := (interest / (principal * time_in_years)) * 100

theorem find_rate_of_interest :
  rate = 4 := by
  sorry

end find_rate_of_interest_l674_674597


namespace initial_deposit_l674_674097

theorem initial_deposit (A r : ℝ) (n t : ℕ) (hA : A = 169.40) 
  (hr : r = 0.20) (hn : n = 2) (ht : t = 1) :
  ∃ P : ℝ, P = 140 ∧ A = P * (1 + r / n)^(n * t) :=
by
  sorry

end initial_deposit_l674_674097


namespace length_of_first_train_l674_674957

/-- 
Two trains of different lengths are running towards each other on parallel lines at 42 kmph and 30 kmph respectively. 
The second train is 280 m long. 
From the moment they meet, they will be clear of each other in 24.998 seconds. 
What is the length of the first train?
-/
theorem length_of_first_train
  (speed_train1_kmph : ℕ := 42) -- Speed of the first train in kmph
  (speed_train2_kmph : ℕ := 30) -- Speed of the second train in kmph
  (length_train2_m : ℕ := 280) -- Length of the second train in meters
  (time_clear_s : real := 24.998) -- Time to clear each other in seconds
  : real :=
  -- Length of the first train
  219.96

#eval length_of_first_train

end length_of_first_train_l674_674957


namespace minimum_hexagon_perimeter_l674_674156

-- Define the conditions given in the problem
def small_equilateral_triangle (side_length : ℝ) (triangle_count : ℕ) :=
  triangle_count = 57 ∧ side_length = 1

def hexagon_with_conditions (angle_condition : ℝ → Prop) :=
  ∀ θ, angle_condition θ → θ ≤ 180 ∧ θ > 0

-- State the main problem as a theorem
theorem minimum_hexagon_perimeter : ∀ n : ℕ, ∃ p : ℕ,
  (small_equilateral_triangle 1 57) → 
  (∃ angle_condition, hexagon_with_conditions angle_condition) →
  (n = 57) →
  p = 19 :=
by
  sorry

end minimum_hexagon_perimeter_l674_674156


namespace balloon_total_and_cost_l674_674310

theorem balloon_total_and_cost
  (Fred_red_balloons : ℕ := 10)
  (Fred_blue_balloons : ℕ := 5)
  (Sam_red_balloons : ℕ := 46)
  (Sam_blue_balloons : ℕ := 20)
  (Dan_red_balloons : ℕ := 16)
  (Dan_blue_balloons : ℕ := 12)
  (cost_red_balloon : ℕ := 10)
  (cost_blue_balloon : ℕ := 5)
: 
  let total_red_balloons := Fred_red_balloons + Sam_red_balloons + Dan_red_balloons,
      total_blue_balloons := Fred_blue_balloons + Sam_blue_balloons + Dan_blue_balloons,
      total_cost := total_red_balloons * cost_red_balloon + total_blue_balloons * cost_blue_balloon in
  total_red_balloons = 72 ∧
  total_blue_balloons = 37 ∧
  total_cost = 905 :=
by
  sorry

end balloon_total_and_cost_l674_674310


namespace lcm_of_36_48_75_l674_674293

-- Definitions of the numbers and their factorizations
def num1 := 36
def num2 := 48
def num3 := 75

def factor_36 := (2^2, 3^2)
def factor_48 := (2^4, 3^1)
def factor_75 := (3^1, 5^2)

def highest_power_2 := 2^4
def highest_power_3 := 3^2
def highest_power_5 := 5^2

def lcm_36_48_75 := highest_power_2 * highest_power_3 * highest_power_5

-- The theorem statement
theorem lcm_of_36_48_75 : lcm_36_48_75 = 3600 := by
  sorry

end lcm_of_36_48_75_l674_674293


namespace jenny_total_distance_seven_hops_l674_674854

noncomputable def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

theorem jenny_total_distance_seven_hops :
  let a := (1 / 4 : ℚ)
  let r := (3 / 4 : ℚ)
  let n := 7
  sum_geometric_series a r n = (14197 / 16384 : ℚ) :=
by
  sorry

end jenny_total_distance_seven_hops_l674_674854


namespace profit_percentage_is_correct_l674_674204

noncomputable def cost_price : ℝ := 47.50
noncomputable def selling_price : ℝ := 65.97
noncomputable def list_price := selling_price / 0.90
noncomputable def profit := selling_price - cost_price
noncomputable def profit_percentage := (profit / cost_price) * 100

theorem profit_percentage_is_correct : profit_percentage = 38.88 := by
  sorry

end profit_percentage_is_correct_l674_674204


namespace daniel_gpa_probability_l674_674838

theorem daniel_gpa_probability :
  let A_points := 4
  let B_points := 3
  let C_points := 2
  let D_points := 1
  let total_classes := 4
  let gpa (points: ℚ) := points / total_classes
  let min_gpa := (13 : ℚ) / total_classes
  let prob_A_eng := (1 : ℚ) / 5
  let prob_B_eng := (1 : ℚ) / 3
  let prob_C_eng := 1 - prob_A_eng - prob_B_eng
  let prob_A_sci := (1 : ℚ) / 3
  let prob_B_sci := (1 : ℚ) / 2
  let prob_C_sci := (1 : ℚ) / 6
  let required_points := 13
  let points_math := A_points
  let points_hist := A_points
  let points_achieved := points_math + points_hist
  let needed_points := required_points - points_achieved
  let success_prob := prob_A_eng * prob_A_sci + 
                      prob_A_eng * prob_B_sci + 
                      prob_B_eng * prob_A_sci + 
                      prob_B_eng * prob_B_sci 
  in success_prob = (4 : ℚ) / 9 := sorry

end daniel_gpa_probability_l674_674838


namespace vector_addition_l674_674747

-- Let vectors a and b be defined as
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, -3)

-- Theorem statement to prove
theorem vector_addition : a + 2 • b = (4, -5) :=
by
  sorry

end vector_addition_l674_674747


namespace cyclic_quad_angles_l674_674630

theorem cyclic_quad_angles (A B C D : ℝ) (angle_sum : A + C = 180 ∧ B + D = 180)
  (BD_bisector_B : ∠ABD = ∠CBD)
  (angle_BDA : ∠BDA = 75)
  (angle_BDC : ∠BDC = 70) :
  (A = 75 ∧ B = 70 ∧ C = 105 ∧ D = 110) ∨ (A = 105 ∧ B = 10 ∧ C = 75 ∧ D = 170) :=
sorry

end cyclic_quad_angles_l674_674630


namespace calculate_value_l674_674683

theorem calculate_value :
  ( (3^3 - 1) / (3^3 + 1) ) * ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * ( (7^3 - 1) / (7^3 + 1) )
  = 57 / 84 := by
  sorry

end calculate_value_l674_674683


namespace probability_of_non_adjacent_zeros_l674_674391

-- Define the total number of arrangements of 3 ones and 2 zeros
def totalArrangements : ℕ := Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

-- Define the number of arrangements where the 2 zeros are together
def adjacentZerosArrangements : ℕ := 2 * Nat.factorial 4 / (Nat.factorial 3 * Nat.factorial 1)

-- Calculate the desired probability
def nonAdjacentZerosProbability : ℚ := 
  1 - (adjacentZerosArrangements.toRat / totalArrangements.toRat)

theorem probability_of_non_adjacent_zeros :
  nonAdjacentZerosProbability = 3/5 :=
sorry

end probability_of_non_adjacent_zeros_l674_674391


namespace length_of_second_dimension_l674_674229

def volume_of_box (w : ℝ) : ℝ :=
  (w - 16) * (46 - 16) * 8

theorem length_of_second_dimension (w : ℝ) (h_volume : volume_of_box w = 4800) : w = 36 :=
by
  sorry

end length_of_second_dimension_l674_674229


namespace sum_of_first_n_odd_numbers_l674_674526

theorem sum_of_first_n_odd_numbers (n : ℕ) : 
  ∑ i in finset.range (n + 1), (2 * i + 1) = n^2 :=
begin
  sorry
end

example : ∑ i in finset.range 10, (2 * i + 1) = 100 :=
begin
  convert sum_of_first_n_odd_numbers 10,
  norm_num,
end

end sum_of_first_n_odd_numbers_l674_674526


namespace history_students_count_l674_674223

theorem history_students_count
  (total_students : ℕ)
  (sample_students : ℕ)
  (physics_students_sampled : ℕ)
  (history_students_sampled : ℕ)
  (x : ℕ)
  (H1 : total_students = 1500)
  (H2 : sample_students = 120)
  (H3 : physics_students_sampled = 80)
  (H4 : history_students_sampled = sample_students - physics_students_sampled)
  (H5 : x = 1500 * history_students_sampled / sample_students) :
  x = 500 :=
by
  sorry

end history_students_count_l674_674223


namespace quadrilateral_areas_product_l674_674215

noncomputable def areas_product_property (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) : Prop :=
  (S_ADP * S_BCP * S_ABP * S_CDP) % 10000 ≠ 1988
  
theorem quadrilateral_areas_product (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) :
  areas_product_property S_ADP S_ABP S_CDP S_BCP h1 :=
by
  sorry

end quadrilateral_areas_product_l674_674215


namespace find_parabola_eq_l674_674289

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = -3 * x ^ 2 + 18 * x - 22 ↔ (x - 3) ^ 2 + 5 = y

theorem find_parabola_eq :
  ∃ a b c : ℝ, (vertex = (3, 5) ∧ axis_of_symmetry ∧ point_on_parabola = (2, 2)) →
  parabola_equation a b c :=
sorry

end find_parabola_eq_l674_674289


namespace find_a_in_subset_l674_674875

theorem find_a_in_subset 
  (A : Set ℝ)
  (B : Set ℝ)
  (hA : A = { x | x^2 ≠ 1 })
  (hB : ∃ a : ℝ, B = { x | a * x = 1 })
  (h_subset : B ⊆ A) : 
  ∃ a : ℝ, a = 0 ∨ a = 1 ∨ a = -1 := 
by
  sorry

end find_a_in_subset_l674_674875


namespace ant_travel_distance_l674_674538

/-- 
  There is a bamboo tree on the southern mountain with 30 nodes. 
  The first node has a height of 0.5 feet and each subsequent node is 0.03 feet taller than the previous one.
  The first circle at the top of the first node has a circumference of 1.3 feet and each subsequent circle decreases by 0.013 feet in circumference from the previous one.
  We need to prove that the total distance an ant travels when it reaches the top of the bamboo, including the length of each node and the circumference of each circle, is 61.395 feet.
--/
theorem ant_travel_distance :
  let n := 30 in
  let a1 := 0.5 in
  let d' := 0.03 in
  let b1 := 1.3 in
  let d := -0.013 in
  let S := (n * a1 + (n * (n - 1) / 2) * d') + (n * b1 + (n * (n - 1) / 2) * d) in
  S = 61.395 := 
by
  sorry

end ant_travel_distance_l674_674538


namespace organizing_ball_l674_674883

theorem organizing_ball (n : ℕ) : 
  let guests := 3 * n  -- number of guests
  let cylinders := 3 * n  -- number of cylinders
  (∃ groups : list (list ℕ), 
     (∀ group ∈ groups, list.length group = 3 ∧ 
     (group.get_nth 0).is_A ∧
     (group.get_nth 1).is_B ∧
     (group.get_nth 2).is_C) ∧ 
     (3 * n)! = (3 * n)!) :=
sorry

end organizing_ball_l674_674883


namespace total_rooms_in_hotel_l674_674485

def first_wing_floors : ℕ := 9
def first_wing_halls_per_floor : ℕ := 6
def first_wing_rooms_per_hall : ℕ := 32

def second_wing_floors : ℕ := 7
def second_wing_halls_per_floor : ℕ := 9
def second_wing_rooms_per_hall : ℕ := 40

def third_wing_floors : ℕ := 12
def third_wing_halls_per_floor : ℕ := 4
def third_wing_rooms_per_hall : ℕ := 50

def first_wing_total_rooms : ℕ := 
  first_wing_floors * first_wing_halls_per_floor * first_wing_rooms_per_hall

def second_wing_total_rooms : ℕ := 
  second_wing_floors * second_wing_halls_per_floor * second_wing_rooms_per_hall

def third_wing_total_rooms : ℕ := 
  third_wing_floors * third_wing_halls_per_floor * third_wing_rooms_per_hall

theorem total_rooms_in_hotel : 
  first_wing_total_rooms + second_wing_total_rooms + third_wing_total_rooms = 6648 := 
by 
  sorry

end total_rooms_in_hotel_l674_674485


namespace find_line_AB_l674_674016

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 16

-- Define the line AB
def lineAB (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Proof statement: Line AB is the correct line through the intersection points of the two circles
theorem find_line_AB :
  (∃ x y, circle1 x y ∧ circle2 x y) →
  (∀ x y, (circle1 x y ∧ circle2 x y) ↔ lineAB x y) :=
by
  sorry

end find_line_AB_l674_674016


namespace min_value_xy_l674_674274

-- Defining the operation ⊗
def otimes (a b : ℝ) : ℝ := a * b - a - b

theorem min_value_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : otimes x y = 3) : 9 ≤ x * y := by
  sorry

end min_value_xy_l674_674274


namespace express_in_scientific_notation_l674_674910

def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| < 10

theorem express_in_scientific_notation :
  is_scientific_notation 2.5 7 25000000 :=
  sorry

end express_in_scientific_notation_l674_674910


namespace grace_pennies_l674_674217

theorem grace_pennies (dime_value nickel_value : ℕ) (dimes nickels : ℕ) 
  (h₁ : dime_value = 10) (h₂ : nickel_value = 5) (h₃ : dimes = 10) (h₄ : nickels = 10) : 
  dimes * dime_value + nickels * nickel_value = 150 := 
by 
  sorry

end grace_pennies_l674_674217


namespace limit_seqs_equal_l674_674863

noncomputable def seq_a (a b : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a else (2 * seq_a a b (n - 1) * seq_b a b (n - 1)) / (seq_a a b (n - 1) + seq_b a b (n - 1))

noncomputable def seq_b (a b : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then b else (seq_a a b (n - 1) + seq_b a b (n - 1)) / 2

theorem limit_seqs_equal (a b : ℝ) (h : 0 < a ∧ a < b) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |seq_a a b n - sqrt (a * b)| < ε) ∧ 
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |seq_b a b n - sqrt (a * b)| < ε) := by
  sorry

end limit_seqs_equal_l674_674863


namespace angle_range_l674_674809

variable {α β : ℝ²}

def vector_magnitude (v : ℝ²) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def vector_cross_product_magnitude (a b : ℝ²) : ℝ := 
  abs (a.1 * b.2 - a.2 * b.1)

theorem angle_range (h1 : vector_magnitude α ≤ 1) 
                    (h2 : vector_magnitude β ≤ 1)
                    (h3 : vector_cross_product_magnitude α β = 1 / 2) :
  ∃ θ, Real.sin θ = vector_cross_product_magnitude α β / (vector_magnitude α * vector_magnitude β) ∧ 
  θ ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6) := 
sorry

end angle_range_l674_674809


namespace no_possible_black_white_division_of_2017gon_l674_674066

/-- 
  Prove that it is not possible to divide a convex 2017-gon into black and white triangles 
  such that:
  1. Any two triangles of different colors share a side.
  2. Any two triangles of the same color share only a vertex or do not touch at all.
  3. Each side of the 2017-gon is a side of one of the black triangles.
-/
theorem no_possible_black_white_division_of_2017gon :
  ¬ ∃ (black white : set (set point)),
    (∃ bw_intersect, 
      (∀ b ∈ black, ∀ w ∈ white, (∃ s, s ∈ b ∧ s ∈ w))    ∧ -- condition 1
      (∀ b₁ b₂ ∈ black, ¬ ∃ s, s ∈ b₁ ∧ s ∈ b₂)            ∧ -- condition 2 (black)
      (∀ w₁ w₂ ∈ white, ¬ ∃ s, s ∈ w₁ ∧ s ∈ w₂)            ∧ -- condition 2 (white)
      (∀ s _s, (_s ∈ 2017gon_sides) → ∃ b ∈ black, s ∈ b))   -- condition 3
  := 
  sorry

end no_possible_black_white_division_of_2017gon_l674_674066


namespace rain_third_day_l674_674952

theorem rain_third_day (rain_day1 rain_day2 rain_day3 : ℕ)
  (h1 : rain_day1 = 4)
  (h2 : rain_day2 = 5 * rain_day1)
  (h3 : rain_day3 = (rain_day1 + rain_day2) - 6) : 
  rain_day3 = 18 := 
by
  -- Proof omitted
  sorry

end rain_third_day_l674_674952


namespace largest_consecutive_sum_30_l674_674556

-- Define the problem statement in Lean 4
theorem largest_consecutive_sum_30 :
  ∃ n : ℕ, (∃ a : ℕ, (finset.range n).sum (λ i, a + i) = 30) ∧
           (∀ m : ℕ, (∃ a' : ℕ, (finset.range m).sum (λ i, a' + i) = 30) → m ≤ n) ∧
           n = 5 :=
by
  sorry

end largest_consecutive_sum_30_l674_674556


namespace intersection_M_N_l674_674986

def M : Set ℝ := { x | x > 1 }
def N : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem intersection_M_N :
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } := 
sorry

end intersection_M_N_l674_674986


namespace find_dimensions_l674_674130

theorem find_dimensions (x y : ℝ) 
  (h1 : 90 = (2 * x + y) * (2 * y))
  (h2 : x * y = 10) : x = 2 ∧ y = 5 :=
by
  sorry

end find_dimensions_l674_674130


namespace power_equation_l674_674119

theorem power_equation (a : ℂ) (h : 5 = a + a⁻¹) : a^4 + a⁻⁴ = 527 :=
by
  sorry

end power_equation_l674_674119


namespace train_speed_correct_l674_674240

def train_length : ℝ := 100
def bridge_length : ℝ := 150
def time_to_cross : ℝ := 16.665333439991468
def speed_in_kmph : ℝ := 54

theorem train_speed_correct :
  let distance := train_length + bridge_length in
  let speed_in_mps := distance / time_to_cross in
  let speed_in_kmph_calc := speed_in_mps * 3.6 in
  speed_in_kmph_calc = speed_in_kmph :=
by
  -- Placeholder for the actual proof, which is not required in this task
  sorry

end train_speed_correct_l674_674240


namespace general_identity_l674_674348

theorem general_identity (a b : ℝ) (n : ℕ) (h : n > 0) :
  (a - b) * (finset.range (n + 1)).sum (λ k, a^(n - k) * b^k) = a^(n + 1) - b^(n + 1) :=
by
  sorry

end general_identity_l674_674348


namespace zeros_not_adjacent_probability_l674_674375

def total_arrangements : ℕ := Nat.factorial 5

def adjacent_arrangements : ℕ := 2 * Nat.factorial 4

def probability_not_adjacent : ℚ := 
  1 - (adjacent_arrangements / total_arrangements)

theorem zeros_not_adjacent_probability :
  probability_not_adjacent = 0.6 := 
by 
  sorry

end zeros_not_adjacent_probability_l674_674375


namespace probability_green_then_blue_l674_674991

theorem probability_green_then_blue :
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := green_marbles / total_marbles
  let prob_second_blue := blue_marbles / (total_marbles - 1)
  prob_first_green * prob_second_blue = 4 / 15 :=
sorry

end probability_green_then_blue_l674_674991


namespace perpendicular_centers_of_equilateral_triangles_l674_674077

open Classical

noncomputable def quadrilateral_with_equilateral_triangles : Prop :=
  ∃ (A B C D O1 O2 O3 O4 : ℝ × ℝ), 
  convex_quadrilateral A B C D ∧
  dist A C = dist B D ∧
  equilateral_triangle A B O1 ∧
  equilateral_triangle B C O2 ∧
  equilateral_triangle C D O3 ∧
  equilateral_triangle D A O4 ∧
  is_perpendicular (O1, O3) (O2, O4)

theorem perpendicular_centers_of_equilateral_triangles (A B C D O1 O2 O3 O4 : ℝ × ℝ)
  (h_quad : convex_quadrilateral A B C D)
  (h_diag : dist A C = dist B D)
  (h_eq_tri1 : equilateral_triangle A B O1)
  (h_eq_tri2 : equilateral_triangle B C O2)
  (h_eq_tri3 : equilateral_triangle C D O3)
  (h_eq_tri4 : equilateral_triangle D A O4) :
  is_perpendicular (O1, O3) (O2, O4) :=
sorry

end perpendicular_centers_of_equilateral_triangles_l674_674077


namespace man_l674_674228

-- Define the relevant variables and conditions.
variables (downstream_rate still_water_rate current_rate upstream_rate : ℝ)
variables (h_downstream : downstream_rate = 24)
variables (h_still_water : still_water_rate = 15.5)
variables (h_current : current_rate = 8.5)

-- Theorem stating the man's rate when rowing upstream.
theorem man's_upstream_rate : upstream_rate = still_water_rate - current_rate :=
begin
  -- Use the given conditions to state the upstream rate.
  rw [h_still_water, h_current],
  -- Calculate the upstream rate given the conditions.
  exact (begin
    calc 
      upstream_rate = 15.5 - 8.5 : by sorry
  end),
end

end man_l674_674228


namespace triangle_inscribed_circle_ratio_l674_674209

noncomputable section

open Real

def inscribed_circle_segment_ratio (a b c : ℝ) (r s : ℝ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13 ∧ r < s ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (r + s = a) ∧ (r / s = 2 / 3)

theorem triangle_inscribed_circle_ratio :
  ∃ r s, inscribed_circle_segment_ratio 5 12 13 r s :=
begin
  use [2, 3],
  split,
  { exact eq.refl 5 },
  split,
  { exact eq.refl 12 },
  split,
  { exact eq.refl 13 },
  split,
  { linarith },
  split,
  { linarith },
  split,
  { linarith },
  split,
  { linarith },
  split,
  { linarith },
  { linarith }
end

end triangle_inscribed_circle_ratio_l674_674209


namespace find_m_inequality_l674_674350

open Real

noncomputable def f (x m : ℝ) : ℝ := m - |x - 2|

theorem find_m (m : ℝ) (h : ∀ x : ℝ, f (x + 2) m ≥ 0 ↔ x ∈ Icc (-1) 1) : m = 1 := 
sorry

theorem inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) : a + 2 * b + 3 * c ≥ 9 := 
sorry

end find_m_inequality_l674_674350


namespace smallest_integer_with_remainders_l674_674966

theorem smallest_integer_with_remainders :
  ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 10 = 9 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) := 
sorry

end smallest_integer_with_remainders_l674_674966


namespace pyramidal_stack_of_logs_sum_l674_674046

theorem pyramidal_stack_of_logs_sum :
  ∃ n : ℕ, ∃ a l d : ℤ, a = 15 ∧ l = 5 ∧ d = -2 ∧
  (n = (l - a) / d + 1) ∧ (∑ i in finset.range n, a + d * i = 60) :=
sorry

end pyramidal_stack_of_logs_sum_l674_674046


namespace trig_sum_identity_l674_674889

theorem trig_sum_identity :
    (∑ n in Finset.range 89, 1 / (Real.cos (n * Real.pi / 180) * Real.cos ((n + 1) * Real.pi / 180)))
    = (Real.cos (Real.pi / 180)) / (Real.sin (Real.pi / 180))^2 := by
  sorry

end trig_sum_identity_l674_674889


namespace count_not_integer_values_l674_674473

def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

def not_integer_angle (n : ℕ) : Prop := ¬ (interior_angle n).denom = 1

def count_not_integer_angles : ℕ :=
  (Finset.range 15).filter (λ n, 3 ≤ n ∧ not_integer_angle n).card

theorem count_not_integer_values :
  count_not_integer_angles = 4 :=
by
  sorry

end count_not_integer_values_l674_674473


namespace product_of_triangle_areas_not_end_in_1988_l674_674212

theorem product_of_triangle_areas_not_end_in_1988
  (a b c d : ℕ)
  (h1 : a * c = b * d)
  (hp : (a * b * c * d) = (a * c)^2)
  : ¬(∃ k : ℕ, (a * b * c * d) = 10000 * k + 1988) :=
sorry

end product_of_triangle_areas_not_end_in_1988_l674_674212


namespace quadratic_other_root_l674_674323

theorem quadratic_other_root (m x2 : ℝ) (h₁ : 1^2 - 4*1 + m = 0) (h₂ : x2^2 - 4*x2 + m = 0) : x2 = 3 :=
sorry

end quadratic_other_root_l674_674323


namespace area_of_quadrilateral_PQRS_l674_674271

-- Definitions for the four sides and one angle
def PQ : ℝ := 6
def QR : ℝ := 8
def RS : ℝ := 15
def PS : ℝ := 17
def angle_PQR : ℝ := 90 

-- The final statement to prove
theorem area_of_quadrilateral_PQRS : 
  let PR := Real.sqrt (PQ^2 + QR^2), 
      s := (PR + RS + PS) / 2,
      area_triangle_PQR := (1 / 2) * PQ * QR,
      area_triangle_PRS := Real.sqrt (s * (s - PR) * (s - RS) * (s - PS)) in
  area_triangle_PQR + area_triangle_PRS = 98.5 := 
sorry

end area_of_quadrilateral_PQRS_l674_674271


namespace ratio_of_tory_to_combined_cookies_l674_674267

theorem ratio_of_tory_to_combined_cookies (Clementine_cookies Jake_cookies Tory_cookies : ℕ)
(sales_total : ℕ)
(price_per_cookie : ℕ)
(h1 : Clementine_cookies = 72)
(h2 : Jake_cookies = 2 * Clementine_cookies)
(h3 : sales_total = 648)
(h4 : price_per_cookie = 2)
(h5 : sales_total / price_per_cookie = Clementine_cookies + Jake_cookies + Tory_cookies) :
Tory_cookies : (Clementine_cookies + Jake_cookies) = 1 : 2 :=
by
  sorry

end ratio_of_tory_to_combined_cookies_l674_674267


namespace solve_for_x_l674_674519

theorem solve_for_x : ∃ x : ℝ, 49^(x + 2) = 7^(4 * x - 3) ∧ x = 7 / 2 :=
by
  sorry

end solve_for_x_l674_674519


namespace simplify_frac_and_find_cd_l674_674898

theorem simplify_frac_and_find_cd :
  ∀ (m : ℤ), ∃ (c d : ℤ), 
    (c * m + d = (6 * m + 12) / 3) ∧ (c = 2) ∧ (d = 4) ∧ (c / d = 1 / 2) :=
by
  sorry

end simplify_frac_and_find_cd_l674_674898


namespace length_of_pc_l674_674457

theorem length_of_pc (a x : ℝ) (h₁ : PQ = sqrt 3 * PS) 
  (h₂ : a^2 = sqrt 3 * x^2) : length_of_PC = 2 * sqrt 7 := by 
sorry

end length_of_pc_l674_674457


namespace find_angle_A_max_area_triangle_ABC_l674_674064

variables (A B C a b c : ℝ)
variables (triangle_ABC : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
variables (a_opposite_A : a > 0)
variables (b_opposite_B : b > 0)
variables (c_opposite_C : c > 0)
variables (condition1 : sqrt 3 * a * sin B - b * cos A = 0)
variables (condition2 : a = 2)

theorem find_angle_A : A = π / 6 :=
by sorry

theorem max_area_triangle_ABC : (1/2 * b * c * sin A) ≤ 2 + sqrt 3 :=
by sorry

end find_angle_A_max_area_triangle_ABC_l674_674064


namespace number_of_candies_l674_674182

-- Define the conditions of the problem
def totalCookies : Nat := 28
def totalBags : Nat := 14
def bagsInPossession : Nat := 2

-- Compute the number of cookies per bag
def cookiesPerBag : Nat := totalCookies / totalBags

-- Compute the total number of cookies in the bags in possession
def cookiesInPossession : Nat := cookiesPerBag * bagsInPossession

-- Prove that the number of candies is as specified
theorem number_of_candies : Nat := totalCookies + totalCandies - cookiesInPossession = totalCookies - cookiesInPossession :=
by
  let totalCandies := 24 -- Correct answer based on the calculation
  sorry -- Proof to be completed

end number_of_candies_l674_674182


namespace monotonicity_intervals_exists_a_for_positive_f_l674_674357

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x - a / x + 2 * a

theorem monotonicity_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, f a x ∂ x < 0) ∧
  (a > 0 → ∀ x > 0, 
    (x < (a + Real.sqrt(a^2 + 4*a)) / 2 → f a x ∂ x > 0) ∧
    (x > (a + Real.sqrt(a^2 + 4*a)) / 2 → f a x ∂ x < 0)) := sorry

theorem exists_a_for_positive_f (h : ℝ) (e : ℝ) (h0 : 0 < a) (h1 : a ∈ (Real.exp 2 / (3 * Real.exp 1 - 1), +∞)) :
  ∀ x ∈ Icc 1 e, f a x > 0 ↔ 
  (∀ x ∈ Icc 1 e, f a x > 0 → a > 0 ∧ 
   ∃ a > 0, (a ∈ (Real.exp 2 / (3 * e - 1), +∞))) := sorry


end monotonicity_intervals_exists_a_for_positive_f_l674_674357


namespace left_handed_only_jazz_l674_674040

def club_members : ℕ := 30
def left_handed_members : ℕ := 12
def jazz_likers : ℕ := 22
def both_jazz_and_rock_likers : ℕ := 9
def left_handed_dislikes_both : ℕ := 1

theorem left_handed_only_jazz :
  ∀ (total_club_members left_handed_members jazz_likers both_jazz_and_rock_likers left_handed_dislikes_both : ℕ),
    total_club_members = 30 →
    left_handed_members = 12 →
    jazz_likers = 22 →
    both_jazz_and_rock_likers = 9 →
    left_handed_dislikes_both = 1 →
    (left_handed_members - 1 = 12) :=
by
  intros total_club_members left_handed_members jazz_likers both_jazz_and_rock_likers left_handed_dislikes_both
  intros h_total_club_members h_left_handed_members h_jazz_likers h_both_jazz_and_rock_likers h_left_handed_dislikes_both
  have h1 : total_club_members = 30 := h_total_club_members
  have h2 : left_handed_members = 12 := h_left_handed_members
  have h3 : jazz_likers = 22 := h_jazz_likers
  have h4 : both_jazz_and_rock_likers = 9 := h_both_jazz_and_rock_likers
  have h5 : left_handed_dislikes_both = 1 := h_left_handed_dislikes_both
  have h6 : left_handed_members = 12 - 1 := by rw [h2]; exact dec_trivial
  exact h6

end left_handed_only_jazz_l674_674040


namespace smallest_value_of_c_l674_674145

/-- The polynomial x^3 - cx^2 + dx - 2550 has three positive integer roots,
    and the product of the roots is 2550. Prove that the smallest possible value of c is 42. -/
theorem smallest_value_of_c :
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 2550 ∧ c = a + b + c) → c = 42 :=
sorry

end smallest_value_of_c_l674_674145


namespace compute_fraction_product_l674_674699

theorem compute_fraction_product :
  (∏ i in (finset.range 5).map (λ n, n + 3), (i ^ 3 - 1) / (i ^ 3 + 1)) = (57 / 168) := by
  sorry

end compute_fraction_product_l674_674699


namespace p_implies_q_q_not_implies_p_l674_674477

variable (a b : ℝ)
def p : Prop := a * b ≠ 0
def q : Prop := a ≠ 0

theorem p_implies_q : p a b → q a :=
begin
  intro h,
  -- Proof omitted
  sorry,
end

theorem q_not_implies_p : q a → ¬ (p a b) :=
begin
  intro hq,
  -- Proof omitted
  sorry,
end

example : (p a b → q a) ∧ ¬ (q a → p a b) :=
begin
  split,
  { apply p_implies_q, },
  { apply q_not_implies_p, }
end

end p_implies_q_q_not_implies_p_l674_674477


namespace modulus_of_product_l674_674416

namespace ComplexModule

open Complex

-- Definition of the complex numbers z1 and z2
def z1 : ℂ := 1 + I
def z2 : ℂ := 2 - I

-- Definition of their product z1z2
def z1z2 : ℂ := z1 * z2

-- Statement we need to prove (the modulus of z1z2 is √10)
theorem modulus_of_product : abs z1z2 = Real.sqrt 10 := by
  sorry

end ComplexModule

end modulus_of_product_l674_674416


namespace line_parallel_to_parallel_lines_in_plane_l674_674033

-- Definitions based on the problem conditions
variables {α : Type*} [plane α]
variables {a : Type*} [line a]

-- Definition of parallel relation
def line_parallel_to_plane (a : Type*) (α : Type*) := ∀ (l : line_in_plane α), parallel a l

-- Given condition that line a is parallel to plane α
axiom line_parallel_to_plane_a_alpha : line_parallel_to_plane a α

-- Theorem to prove: line a is parallel to a group of parallel lines in plane α
theorem line_parallel_to_parallel_lines_in_plane :
  ∃ (group : set (line_in_plane α)), (∀ l ∈ group, parallel a l) ∧ is_group_of_parallel_lines group :=
sorry

end line_parallel_to_parallel_lines_in_plane_l674_674033


namespace quadratic_no_real_roots_l674_674923

theorem quadratic_no_real_roots 
  (a b c m : ℝ) 
  (h1 : c > 0) 
  (h2 : c = a * m^2) 
  (h3 : c = b * m)
  : ∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0) :=
by 
  sorry

end quadratic_no_real_roots_l674_674923


namespace images_per_memory_card_l674_674454

-- Define the constants based on the conditions given in the problem
def daily_pictures : ℕ := 10
def years : ℕ := 3
def days_per_year : ℕ := 365
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140

-- Define the properties to be proved
theorem images_per_memory_card :
  (years * days_per_year * daily_pictures) / (total_spent / cost_per_card) = 50 :=
by
  sorry

end images_per_memory_card_l674_674454


namespace maximum_n_gons_in_triangle_no_overlap_l674_674036

noncomputable def maximum_n_gons_no_overlap (AB BC CA : ℕ) (triangle_has_sides : AB = 2019 ∧ BC = 2020 ∧ CA = 2021) : ℕ :=
  have sides : AB = 2019 ∧ BC = 2020 ∧ CA = 2021 from triangle_has_sides
  sorry

/-- In a triangle ABC with sides AB = 2019, BC = 2020, and CA = 2021, show that the maximum n
such that three regular n-gons can be drawn sharing sides with the triangle without overlapping is 11 -/
theorem maximum_n_gons_in_triangle_no_overlap (AB BC CA : ℕ) (triangle_has_sides : AB = 2019 ∧ BC = 2020 ∧ CA = 2021) :
  maximum_n_gons_no_overlap AB BC CA triangle_has_sides = 11 :=
sorry

end maximum_n_gons_in_triangle_no_overlap_l674_674036


namespace abs_lt_one_iff_sq_lt_one_l674_674829

variable {x : ℝ}

theorem abs_lt_one_iff_sq_lt_one : |x| < 1 ↔ x^2 < 1 := sorry

end abs_lt_one_iff_sq_lt_one_l674_674829


namespace power_of_2_implies_power_of_2_l674_674113

theorem power_of_2_implies_power_of_2 {n : ℕ} (h1 : n ≥ 4) (h2 : ∃ k : ℕ, 2^k = (2^n / n).toNat) : ∃ m : ℕ, n = 2^m :=
by
  sorry

end power_of_2_implies_power_of_2_l674_674113


namespace number_of_integer_values_of_x_such_that_12_diamond_x_is_positive_integer_l674_674557

def diamond (a b : ℤ) := a^2 / b

theorem number_of_integer_values_of_x_such_that_12_diamond_x_is_positive_integer :
  {x : ℤ // 12 ∧ x ∧ diamond 12 x ∈ ℕ}.to_finset.card = 15 :=
by
  sorry

end number_of_integer_values_of_x_such_that_12_diamond_x_is_positive_integer_l674_674557


namespace hyperbola_focal_length_l674_674731

theorem hyperbola_focal_length (m : ℝ) (h : -5 < m ∧ m < 20) : 
   hyperbola_focal_length (m + 5) (20 - m) = 10 :=
sorry

end hyperbola_focal_length_l674_674731


namespace count_not_integer_values_l674_674474

def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

def not_integer_angle (n : ℕ) : Prop := ¬ (interior_angle n).denom = 1

def count_not_integer_angles : ℕ :=
  (Finset.range 15).filter (λ n, 3 ≤ n ∧ not_integer_angle n).card

theorem count_not_integer_values :
  count_not_integer_angles = 4 :=
by
  sorry

end count_not_integer_values_l674_674474


namespace telescoping_product_l674_674701

theorem telescoping_product :
  (∏ x in {3, 4, 5, 6, 7}, (x^3 - 1) / (x^3 + 1)) = 57 / 168 := by
  sorry

end telescoping_product_l674_674701


namespace interval_length_condition_l674_674137

theorem interval_length_condition (c : ℝ) (x : ℝ) (H1 : 3 ≤ 5 * x - 4) (H2 : 5 * x - 4 ≤ c) 
                                  (H3 : (c + 4) / 5 - 7 / 5 = 15) : c - 3 = 75 := 
sorry

end interval_length_condition_l674_674137


namespace volume_of_inscribed_sphere_is_correct_l674_674651

noncomputable def volume_of_inscribed_sphere (length width height : ℝ) : ℝ :=
  if (length <= width ∧ length <= height) then
    let r := length / 2 in
    4 / 3 * Real.pi * r ^ 3
  else if (width <= length ∧ width <= height) then
    let r := width / 2 in
    4 / 3 * Real.pi * r ^ 3
  else
    let r := height / 2 in
    4 / 3 * Real.pi * r ^ 3

theorem volume_of_inscribed_sphere_is_correct :
  volume_of_inscribed_sphere 8 10 10 = (256 / 3) * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_is_correct_l674_674651


namespace remainder_g_x12_div_g_x_l674_674801

-- Define the polynomial g
noncomputable def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Proving the remainder when g(x^12) is divided by g(x) is 6
theorem remainder_g_x12_div_g_x : 
  (g (x^12) % g x) = 6 :=
sorry

end remainder_g_x12_div_g_x_l674_674801


namespace arithmetic_sqrt_of_49_l674_674125

theorem arithmetic_sqrt_of_49 : ∃ x : ℕ, x^2 = 49 ∧ x = 7 :=
by
  sorry

end arithmetic_sqrt_of_49_l674_674125


namespace ratio_50kg_to_05tons_not_100_to_1_l674_674449

theorem ratio_50kg_to_05tons_not_100_to_1 (weight1 : ℕ) (weight2 : ℕ) (r : ℕ) 
  (h1 : weight1 = 50) (h2 : weight2 = 500) (h3 : r = 100) : ¬ (weight1 * r = weight2) := 
by
  sorry

end ratio_50kg_to_05tons_not_100_to_1_l674_674449


namespace binom_20_4_plus_10_l674_674268

open Nat

noncomputable def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem binom_20_4_plus_10 :
  binom 20 4 + 10 = 4855 := by
  sorry

end binom_20_4_plus_10_l674_674268


namespace Joann_lollipop_theorem_l674_674855

noncomputable def Joann_lollipops (a : ℝ) : ℝ := a + 9

theorem Joann_lollipop_theorem (a : ℝ) (total_lollipops : ℝ) 
  (h1 : a + (a + 3) + (a + 6) + (a + 9) + (a + 12) + (a + 15) = 150) 
  (h2 : total_lollipops = 150) : 
  Joann_lollipops a = 26.5 :=
by
  sorry

end Joann_lollipop_theorem_l674_674855


namespace average_student_headcount_l674_674259

variable (headcount_02_03 headcount_03_04 headcount_04_05 headcount_05_06 : ℕ)
variable {h_02_03 : headcount_02_03 = 10900}
variable {h_03_04 : headcount_03_04 = 10500}
variable {h_04_05 : headcount_04_05 = 10700}
variable {h_05_06 : headcount_05_06 = 11300}

theorem average_student_headcount : 
  (headcount_02_03 + headcount_03_04 + headcount_04_05 + headcount_05_06) / 4 = 10850 := 
by 
  sorry

end average_student_headcount_l674_674259


namespace xy_ratio_l674_674335

variables (x y z t : ℝ)
variables (hx : x > y) (hz : z = (x + y) / 2) (ht : t = Real.sqrt (x * y)) (h : x - y = 3 * (z - t))

theorem xy_ratio (x y : ℝ) (hx : x > y) (hz : z = (x + y) / 2) (ht : t = Real.sqrt (x * y)) (h : x - y = 3 * (z - t)) :
  x / y = 25 :=
sorry

end xy_ratio_l674_674335


namespace triangle_angle_and_area_l674_674762

theorem triangle_angle_and_area (a b c A B C : ℝ)
  (h₁ : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A))
  (h₂ : 0 < C ∧ C < Real.pi)
  (h₃ : c = 2 * Real.sqrt 3) :
  C = Real.pi / 3 ∧ 0 ≤ (1 / 2) * a * b * Real.sin C ∧ (1 / 2) * a * b * Real.sin C ≤ 3 * Real.sqrt 3 :=
by
  sorry

end triangle_angle_and_area_l674_674762


namespace roger_garden_partition_count_l674_674087

variable (n : ℕ) (h : 0 < n)

-- representing the garden's dimensions and the partition conditions
def gardenDimension := (2 * n + 1, 2 * n + 1)

-- proof statement
theorem roger_garden_partition_count : 
  exists (partitions : finset (finset (ℕ × ℕ))), 
  (partitions.card = 2 ^ n) ∧
  ∀ p ∈ partitions, 
    (p = {(x, y) | ∃ k, k ∈ (2 * n + 1) ∧ x = k ∧ (even k)}) ∨ 
    (p = {(x, y) | ∃ k, k ∈ (1 * 2 * n + 1) ∧ y = k ∧ (even k)}) 
  sorry

end roger_garden_partition_count_l674_674087


namespace isosceles_triangle_perimeter_l674_674332

theorem isosceles_triangle_perimeter (a b : ℝ) (h_iso : a = 4 ∨ b = 4) (h_iso2 : a = 8 ∨ b = 8) : 
  (a = 4 ∧ b = 8 ∧ 4 + a + b = 16 ∨ 
  a = 4 ∧ b = 8 ∧ b + a + a = 20 ∨ 
  a = 8 ∧ b = 4 ∧ a + a + b = 20) :=
by sorry

end isosceles_triangle_perimeter_l674_674332


namespace smallest_possible_n_l674_674867

theorem smallest_possible_n {n : ℕ} (x : ℕ → ℝ) (h1 : ∀ i, i < n → |x i| < 1) 
  (h2 : ∑ i in finset.range n, |x i| = 17 + |∑ i in finset.range n, x i|) : n = 18 :=
sorry

end smallest_possible_n_l674_674867


namespace correct_operation_l674_674615

theorem correct_operation :
  (∃ op : ℕ → ℕ → ℕ, 
    ((op 6 3) + 4 - (2 - 1) = 5) ∧ 
    (op = (λ x y, x / y))) :=
begin
  use (λ x y, x / y),
  split,
  { simp,
    norm_num,
  },
  { refl },
end

end correct_operation_l674_674615


namespace sum_of_digits_of_largest_number_with_13_matchsticks_l674_674958

def matchsticks_required : Nat → Nat
| 0 => 6
| 1 => 2
| 2 => 5
| 3 => 5
| 4 => 4
| 5 => 5
| 6 => 6
| 7 => 3
| 8 => 7
| 9 => 6
| _ => 0

def largest_number_with_matchsticks (total_matchsticks : Nat) : Nat := 51111 -- corresponding to 5 + 1 + 1 + 1 + 1

theorem sum_of_digits_of_largest_number_with_13_matchsticks : 
  let n := largest_number_with_matchsticks 13 in
  (n.digits.sum = 9) :=
by 
  let n := largest_number_with_matchsticks 13
  have h1: n = 51111 := rfl
  rw [h1, Nat.digits, List.sum]
  exact rfl

end sum_of_digits_of_largest_number_with_13_matchsticks_l674_674958


namespace find_a_2018_l674_674877

noncomputable def sequence_a : ℕ → ℚ
| 0       := 1
| (n + 1) := (9 - 4 * sequence_a n) / (4 - sequence_a n)

theorem find_a_2018 : sequence_a 2017 = 5 / 3 :=
by sorry

end find_a_2018_l674_674877


namespace iso_trapezoid_proof_l674_674730

-- Definition of the isosceles trapezoid problem
def iso_trapezoid (x y z u t r s : ℕ) :=
  x = 2 * t^2 ∧ 
  y = 2 * r * s ∧ 
  z = (r - s) * t ∧ 
  u = (r + s) * t

-- Statement to be proved
theorem iso_trapezoid_proof (x y z u t r s : ℕ) 
  (h : iso_trapezoid x y z u t r s) : 
  x * y + z^2 = u^2 :=
by
  -- Extracting assumptions from definition
  cases h with hx hy hz hu,
  -- Simplifying the goal using the extracted assumptions
  rw [hx, hy, hz, hu],
  -- Simplifying both sides of the equation
  sorry

end iso_trapezoid_proof_l674_674730


namespace number_of_technicians_l674_674914

-- Definitions of the conditions
def average_salary_all_workers := 10000
def average_salary_technicians := 12000
def average_salary_rest := 8000
def total_workers := 14

-- Variables for the number of technicians and the rest of the workers
variable (T R : ℕ)

-- Problem statement in Lean
theorem number_of_technicians :
  (T + R = total_workers) →
  (T * average_salary_technicians + R * average_salary_rest = total_workers * average_salary_all_workers) →
  T = 7 :=
by
  -- leaving the proof as sorry
  sorry

end number_of_technicians_l674_674914


namespace no_lattice_points_y_eq_mx_plus_5_l674_674635

theorem no_lattice_points_y_eq_mx_plus_5
  (m : ℚ)
  (h₁ : 1 / 3 < m)
  (h₂ : m < 52 / 151) :
  ∀ x : ℤ, 0 < x ∧ x ≤ 150 → ∀ y : ℤ, y ≠ m * x + 5 := 
by
  sorry

end no_lattice_points_y_eq_mx_plus_5_l674_674635


namespace compute_fraction_product_l674_674698

theorem compute_fraction_product :
  (∏ i in (finset.range 5).map (λ n, n + 3), (i ^ 3 - 1) / (i ^ 3 + 1)) = (57 / 168) := by
  sorry

end compute_fraction_product_l674_674698


namespace verify_solution_set_l674_674362

noncomputable def value_of_a : ℝ :=
  let a := -2
  show a = -2 from rfl

noncomputable def solution_set_of_inequality (x : ℝ) : Prop :=
  ∃ a : ℝ, a = -2 ∧ (-2 * x^2 - 5 * x + 3 > 0)

theorem verify_solution_set :
  ∀ x : ℝ, solution_set_of_inequality x ↔ -3 < x ∧ x < 1 / 2 := by
  intro x
  apply Iff.intro
  · intro ⟨a, ha, h⟩
    rw [ha] at h
    sorry
  · intro h
    use -2
    constructor
    · rfl
    · sorry

end verify_solution_set_l674_674362


namespace part1_cos_pi_over_3_plus_alpha_part2_tan_beta_given_tan_alpha_plus_beta_l674_674347

variables (α β : ℝ)

-- Condition 1: given
axiom condition1 : sin (Real.pi + α) = (2 * Real.sqrt 5) / 5

-- Additional conditions based on problem statements:
axiom alpha_quadrant4 : α ∈ Set.Icc (-Real.pi) 0

-- Translate the proof problems

theorem part1_cos_pi_over_3_plus_alpha :
  cos (Real.pi / 3 + α) = (Real.sqrt 5 + 2 * Real.sqrt 15) / 10 := 
  sorry

theorem part2_tan_beta_given_tan_alpha_plus_beta :
  tan (α + β) = 1 / 7 → tan β = 3 := 
  sorry

end part1_cos_pi_over_3_plus_alpha_part2_tan_beta_given_tan_alpha_plus_beta_l674_674347


namespace people_relaxing_at_beach_l674_674155

theorem people_relaxing_at_beach :
  let row1 := 24 - 3 - 4 in
  let row2 := 20 - 5 - 2 in
  let row3 := 18 - 2 in
  let row4 := 16 - 8 - 3 in
  let row5 := 30 - 10 - 5 in
  row1 + row2 + row3 + row4 + row5 = 66 :=
by
  let row1 := 24 - 3 - 4
  let row2 := 20 - 5 - 2
  let row3 := 18 - 2
  let row4 := 16 - 8 - 3
  let row5 := 30 - 10 - 5
  have h1 : row1 = 17 := by sorry
  have h2 : row2 = 13 := by sorry
  have h3 : row3 = 16 := by sorry
  have h4 : row4 = 5 := by sorry
  have h5 : row5 = 15 := by sorry
  have total : row1 + row2 + row3 + row4 + row5 = 66 := by 
    calc
      row1 + row2 + row3 + row4 + row5
        = 17 + 13 + 16 + 5 + 15 : by rw [h1, h2, h3, h4, h5]
        ... = 66 : by norm_num
  exact total

end people_relaxing_at_beach_l674_674155


namespace order_of_f_l674_674864

variable (f : ℝ → ℝ)

/-- Conditions:
1. f is an even function for all x ∈ ℝ
2. f is increasing on [0, +∞)
Question:
Prove that the order of f(-2), f(-π), f(3) is f(-2) < f(3) < f(-π) -/
theorem order_of_f (h_even : ∀ x : ℝ, f (-x) = f x)
                   (h_incr : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y) : 
                   f (-2) < f 3 ∧ f 3 < f (-π) :=
by
  sorry

end order_of_f_l674_674864


namespace prob_zeros_not_adjacent_l674_674378

theorem prob_zeros_not_adjacent :
  let total_arrangements := (5.factorial : ℝ)
  let zeros_together_arrangements := (4.factorial : ℝ)
  let prob_zeros_together := (zeros_together_arrangements / total_arrangements)
  let prob_zeros_not_adjacent := 1 - prob_zeros_together
  prob_zeros_not_adjacent = 0.6 :=
by
  sorry

end prob_zeros_not_adjacent_l674_674378


namespace range_s2_minus_c2_l674_674866

variable {x y z : ℝ}
def r : ℝ := Real.sqrt (x^2 + y^2 + z^2)
def s : ℝ := z / r
def c : ℝ := Real.sqrt (x^2 + y^2) / r

theorem range_s2_minus_c2 (x y z : ℝ) : 
  let r := Real.sqrt (x^2 + y^2 + z^2) in
  let s := z / r in
  let c := Real.sqrt (x^2 + y^2) / r in
  -1 ≤ s^2 - c^2 ∧ s^2 - c^2 ≤ 1 :=
sorry

end range_s2_minus_c2_l674_674866


namespace int_n_satisfying_conditions_l674_674308

theorem int_n_satisfying_conditions : 
  (∃! (n : ℤ), ∃ (k : ℤ), (n + 3 = k^2 * (23 - n)) ∧ n ≠ 23) :=
by
  use 2
  -- Provide a proof for this statement here
  sorry

end int_n_satisfying_conditions_l674_674308


namespace monotonic_intervals_of_f_unique_zero_F_when_m_is_2_l674_674483

-- Conditions definitions
def f (x : ℝ) (m : ℝ) : ℝ := (1/2) * x^2 - m * Real.log x
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - (m + 1) * x

-- Monotonic intervals of f
theorem monotonic_intervals_of_f (m : ℝ) (h : 0 < m) :
  (∀ x : ℝ, 0 < x ∧ x < Real.sqrt m → ∀ y : ℝ, 0 < y ∧ y < Real.sqrt m → f x m > f y m) ∧ 
  (∀ x : ℝ, x > Real.sqrt m → ∀ y : ℝ, y > Real.sqrt m → f x m > f y m) := by
  sorry

-- Extreme value when m = 2
theorem unique_zero_F_when_m_is_2 :
  ∃ x : ℝ, F x 2 = 0 ∧ ∀ y : ℝ, F y 2 = 0 → x = y := by
  sorry

end monotonic_intervals_of_f_unique_zero_F_when_m_is_2_l674_674483


namespace find_m_l674_674752

noncomputable def a_seq (n : ℕ) : ℝ :=
if n = 0 then 0 else
  have h : ℝ := a_seq (n - 1),
  if n = 1 then π/6 else
  atan (1 / cos h)

theorem find_m : (∃ m : ℕ, (a_seq 1 = π/6) ∧ (∀ n ≥ 1, a_seq n ∈ (0, π/2)) ∧ (∀ n ≥ 1, tan (a_seq (n + 1)) * cos (a_seq n) = 1) ∧ (∏ k in finset.range m, sin (a_seq (k + 1)) = 1/100) ∧ m = 3333) :=
begin
  sorry
end

end find_m_l674_674752


namespace count_non_integer_interior_angles_l674_674476

theorem count_non_integer_interior_angles : 
  (∑ n in (Finset.Ico 3 15), if (180 * (n - 2)) % n ≠ 0 then 1 else 0) = 4 := 
by 
  sorry

end count_non_integer_interior_angles_l674_674476


namespace sin_n_theta_eq_product_l674_674888

noncomputable
def sin_product_eq (n : ℕ) (θ : ℝ) : Prop :=
  sin (n * θ) = 2 ^ (n - 1) * ∏ k in finset.range n, sin (θ + (k * real.pi / n))

theorem sin_n_theta_eq_product (n : ℕ) (θ : ℝ) : sin_product_eq n θ :=
by sorry

end sin_n_theta_eq_product_l674_674888


namespace find_dimension_l674_674917

-- Define constants and areas based on the problem conditions
def room_width : ℕ := 25
def room_height : ℕ := 12
def door_area : ℕ := 6 * 3
def window_area : ℕ := 3 * (4 * 3)
def painted_area (x : ℕ) : ℕ := 2 * (room_width * room_height) + 2 * (x * room_height) - door_area - window_area
def cost_per_sqft : ℕ := 2
def total_cost (x : ℕ) : ℕ := painted_area x * cost_per_sqft

-- Theorem to prove given the conditions
theorem find_dimension (x : ℕ) (h : total_cost x = 1812) : x = 15 :=
begin
  sorry
end

end find_dimension_l674_674917


namespace find_f_cos_10_l674_674751

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1

theorem find_f_cos_10 : f (cos 10) = 21 - 7 * Real.pi :=
by
  sorry

end find_f_cos_10_l674_674751


namespace decompose_fraction_correct_l674_674088

def decompose_fraction_ways (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (hpq : p ≠ q) : ℕ :=
  8  -- This is the final answer, representing the number of decompositions

theorem decompose_fraction_correct (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (hpq : p ≠ q) :
  let ways := decompose_fraction_ways p q
  in ways = 8 :=
by
  -- Proof would go here
  sorry

end decompose_fraction_correct_l674_674088


namespace guest_bedroom_ratio_l674_674224

theorem guest_bedroom_ratio 
  (lr_dr_kitchen : ℝ) (total_house : ℝ) (master_bedroom : ℝ) (guest_bedroom : ℝ) 
  (h1 : lr_dr_kitchen = 1000) 
  (h2 : total_house = 2300)
  (h3 : master_bedroom = 1040)
  (h4 : guest_bedroom = total_house - (lr_dr_kitchen + master_bedroom)) :
  guest_bedroom / master_bedroom = 1 / 4 := 
by
  sorry

end guest_bedroom_ratio_l674_674224


namespace complex_numbers_omega_expression_l674_674874

noncomputable theory

open Complex

theorem complex_numbers_omega_expression (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + x * y + y^2 = 0) :
  (x / (x + y)) ^ 1990 + (y / (x + y)) ^ 1990 = -1 :=
by
  sorry

end complex_numbers_omega_expression_l674_674874


namespace dot_product_zero_not_implies_zero_vector_l674_674749

def vectors_perpendicular (a b : Vector) : Prop := a.dot b = 0

theorem dot_product_zero_not_implies_zero_vector (a b : Vector) (h : vectors_perpendicular a b) : ¬(a = 0 ∨ b = 0) :=
sorry

end dot_product_zero_not_implies_zero_vector_l674_674749


namespace volume_intersection_octahedra_l674_674969

open Set Real

/-- 
  The region in three-dimensional space defined by the inequalities 
  |x| + |y| + |z| ≤ 1 and |x| + |y| + |z - 1| ≤ 1 has a volume of 1/6.
-/
theorem volume_intersection_octahedra : 
  let region := { p : ℝ × ℝ × ℝ | abs p.1 + abs p.2 + abs p.3 ≤ 1 ∧ abs p.1 + abs p.2 + abs (p.3 - 1) ≤ 1 } in
  volume region = 1 / 6 :=
sorry

end volume_intersection_octahedra_l674_674969


namespace tetrahedron_distinguishable_colorings_l674_674722

theorem tetrahedron_distinguishable_colorings : 
  let colors := {red, white, blue, green} in
  count_distinguishable_colorings_of_tetrahedron(colors) = 35 :=
by sorry

end tetrahedron_distinguishable_colorings_l674_674722


namespace sum_powers_eq_123_l674_674882

-- Define the conditions
variables a b : ℝ 

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

-- Define the theorem we want to prove
theorem sum_powers_eq_123 : a^10 + b^10 = 123 :=
by
  sorry

end sum_powers_eq_123_l674_674882


namespace frac_sum_equals_seven_eights_l674_674825

theorem frac_sum_equals_seven_eights (p q r u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 := 
  sorry

end frac_sum_equals_seven_eights_l674_674825


namespace accurate_value_is_357_l674_674430

-- Define the measured value and the accuracy
def measured_k : ℝ := 3.56897
def accuracy : ℝ := 0.00145

-- Define the upper and lower bounds for k
def k_upper : ℝ := measured_k + accuracy
def k_lower : ℝ := measured_k - accuracy

-- Define the round function for rounding to specific decimal places
def round_to (n : ℤ) (x : ℝ) : ℝ := 
  (Float.ofInt ((Float.toNat (x * (10 ^ n)) + 5) / 10) / (10 ^ n))

-- Define the rounded values
def rounded_k_upper := round_to 2 k_upper
def rounded_k_lower := round_to 2 k_lower

-- State the theorem
theorem accurate_value_is_357 : rounded_k_upper = 3.57 ∧ rounded_k_lower = 3.57 :=
by
  -- The proof is omitted
  sorry

end accurate_value_is_357_l674_674430


namespace originally_planned_days_l674_674211

def man_days (men : ℕ) (days : ℕ) : ℕ := men * days

theorem originally_planned_days (D : ℕ) (h : man_days 5 10 = man_days 10 D) : D = 5 :=
by 
  sorry

end originally_planned_days_l674_674211


namespace sum_of_products_of_roots_l674_674478

-- Define the given polynomial
def poly (x : ℝ) : ℝ := 6 * x^3 - 5 * x^2 + 20 * x - 10

-- Define the roots of the polynomial
def p : ℝ := sorry
def q : ℝ := sorry
def r : ℝ := sorry

-- Assume that p, q, r are the roots of the polynomial
axiom roots_of_poly : poly p = 0 ∧ poly q = 0 ∧ poly r = 0

-- Prove that pq + pr + qr = 10/3
theorem sum_of_products_of_roots : p * q + p * r + q * r = 10 / 3 :=
by
  have : p * q + p * r + q * r = (20 / 6), { sorry },
  exact this

end sum_of_products_of_roots_l674_674478


namespace triangle_area_approximate_l674_674192

theorem triangle_area_approximate :
  let a := 30
      b := 28
      c := 12
      s := (a + b + c) / 2
  in abs ((sqrt (s * (s - a) * (s - b) * (s - c))) - 110.84) < 0.01 :=
by
  let a := 30
  let b := 28
  let c := 12
  let s := (a + b + c) / 2
  sorry

end triangle_area_approximate_l674_674192


namespace series_convergence_l674_674480

def B_n (n : ℕ) (x : ℝ) : ℝ := ∑ k in Finset.range n, 1 / k^x

def f (n : ℕ) : ℝ := B_n n (Real.log (n^2)) / (n * Real.log (2 * n))^2

theorem series_convergence : Summable (λ n : ℕ, if h : n ≥ 2 then f n else 0) :=
by
  sorry

end series_convergence_l674_674480


namespace log_expr_eval_trig_expr_simplify_l674_674197

theorem log_expr_eval : (log 2 9) * (log 3 4) - (2 * real.sqrt 2) ^ (2 / 3) - real.exp (real.log 2) = 0 := 
by 
  sorry

theorem trig_expr_simplify : real.sqrt (1 - real.sin (real.pi / 9)) / (real.cos (real.pi / 18) - real.sin ((170 * real.pi) / 180)) = 1 :=
by 
  sorry

end log_expr_eval_trig_expr_simplify_l674_674197


namespace areas_equal_of_parallel_l674_674501

variable (V : Type) [add_comm_group V] [module ℝ V]

def is_parallelogram (A B C D : V) : Prop :=
  (B - A) + (D - A) = (C - A) + (A - A)

def area (u v : V) := abs (u.1 * v.2 - u.2 * v.1) / 2

theorem areas_equal_of_parallel (A B C D E F : V) 
  (h_parallelogram : is_parallelogram A B C D)
  (hE : E ∈ line_segment A B)
  (hF : F ∈ line_segment A D)
  (h_parallel : ∃ t : ℝ, t • (D - B) = F - E) :
  area (B - C) (E - C) = area (D - C) (F - C) := sorry

end areas_equal_of_parallel_l674_674501


namespace find_k1k2_l674_674226

noncomputable def line_through_point (M : Point) (k1 : ℝ) : Line :=
{ slope := k1,
  point := M }

noncomputable def ellipse : Ellipse :=
{ a := √3,
  b := 1,
  center := (0, 0) }

structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(slope : ℝ)
(point : Point)

structure Ellipse :=
(a : ℝ)
(b : ℝ)
(center : Point)

def intersects_at (l : Line) (e : Ellipse) : (Point × Point) :=
sorry -- Implementation of intersection finding

def midpoint (P1 P2 : Point) : Point :=
{ x := (P1.x + P2.x) / 2,
  y := (P1.y + P2.y) / 2 }

def slope (P1 P2 : Point) : ℝ :=
(P2.y - P1.y) / (P2.x - P1.x)

constant O : Point := { x := 0, y := 0 }

theorem find_k1k2
  (M : Point) (k1 k2 : ℝ)
  (h_M : M = ⟨-2, 0⟩)
  (h_k1 : k1 ≠ 0)
  (h_line_M : Line = line_through_point M k1)
  (P1 P2 : Point)
  (h_intersect : intersects_at (line_through_point M k1) ellipse = (P1, P2))
  (P : Point := midpoint P1 P2)
  (h_slope_OP : slope O P = k2) :
  k1 * k2 = -1 / 3 :=
sorry

end find_k1k2_l674_674226


namespace triangular_faces_at_least_eight_l674_674220

theorem triangular_faces_at_least_eight 
  (V E F : ℕ) 
  (convex : true)
  (degree_condition : ∀ v, vertex v → 4 ≤ degree v)
  (euler_formula : V - E + F = 2) :
  ∃ FΔ : ℕ, FΔ ≥ 8 :=
by
  sorry

end triangular_faces_at_least_eight_l674_674220


namespace n_eq_19_of_15_solutions_l674_674472

theorem n_eq_19_of_15_solutions (n : ℕ) (h_pos : 0 < n) (h : ∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 3 * x + 3 * y + z = n) (num_solutions : (nat.card {p : ℕ × ℕ × ℕ // 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.3 ∧ 3 * p.1 + 3 * p.2 + p.3 = n} = 15)) : n = 19 :=
sorry

end n_eq_19_of_15_solutions_l674_674472


namespace min_pq_length_l674_674409

-- Define the conditions
variables {A B C P Q : Type}
variables [Triangle ABC] -- Assuming ABC forms a triangle
variables [PointOnPerimeter ABC P] [PointOnPerimeter ABC Q] -- P and Q are on the perimeter
variable {Δ : ℝ} -- Δ represents the area of the triangle ABC
variable {λ : ℝ} -- λ represents the constant ratio

-- Define the target theorem
theorem min_pq_length (h_λ : ∃ (Δ : ℝ), (area_cut_off_ratio PQ ABC = λ) (P Q : PerimeterPoints ABC)):
  PQ = 2 * sqrt(λ * Δ * tan(A / 2)) :=
sorry

end min_pq_length_l674_674409


namespace evaluate_propositions_l674_674718

-- Definitions for lines and planes
variables {Line Plane : Type} (l m n : Line) (α β : Plane)

-- Conditions translated to Lean
variable [Perpendicular : ∀ (l : Line) (α : Plane), Prop]
variable [Subset : ∀ (m : Line) (α : Plane), Prop]
variable [ParallelPlanes : ∀ (α β : Plane), Prop]
variable [SkewLines : ∀ (m n : Line), Prop]
variable {p : Point}

-- Proposition 1
axiom perp_two_lines_in_plane (h : ∃ l1 l2 : Line, l1 ≠ l2 ∧ Perpendicular l1 α ∧ Perpendicular l2 α) : Perpendicular l α

-- Proposition 2
axiom skew_lines_counterexample (h1 : SkewLines m n) (h2 : SkewLines n l) : ¬SkewLines m l

-- Proposition 3
axiom parallel_planes_line (h1 : m ∈ α) (h2 : ParallelPlanes α β) : ParallelPlanes m β

-- The main theorem
theorem evaluate_propositions :
  (perp_two_lines_in_plane true) ∧ (¬(skew_lines_counterexample true true)) ∧ (parallel_planes_line true true) :=
by
  sorry

end evaluate_propositions_l674_674718


namespace meaningful_sqrt_l674_674026

theorem meaningful_sqrt (a : ℝ) (h : a ≥ 4) : a = 6 ↔ ∃ x ∈ ({-1, 0, 2, 6} : Set ℝ), x = 6 := 
by
  sorry

end meaningful_sqrt_l674_674026


namespace complex_number_equality_l674_674792

theorem complex_number_equality (a b : ℂ) : a - b = 0 ↔ a = b := sorry

end complex_number_equality_l674_674792


namespace speed_of_stream_l674_674932

theorem speed_of_stream (v : ℝ) 
    (h1 : ∀ (v : ℝ), v ≠ 0 → (80 / (36 + v) = 40 / (36 - v))) : 
    v = 12 := 
by 
    sorry

end speed_of_stream_l674_674932


namespace equiangular_hexagon_side_lengths_l674_674222

theorem equiangular_hexagon_side_lengths (x y : ℕ) : 
  (A B C D E F : ℕ) 
  (hA : A = 3) 
  (hB : B = y) 
  (hC : C = 5) 
  (hD : D = 4) 
  (hE : E = 1) 
  (hF : F = x) 
  (h_equiangular : true) -- Implicitly stating that hexagon is equiangular
  
  (h1 : 3 + y + 5 = 10) 
  (h2 : 1 + x + 3 = 10) 
  : x = 6 ∧ y = 2 := 
begin
  sorry  -- Proof should be added here
end

end equiangular_hexagon_side_lengths_l674_674222


namespace dessert_menus_count_l674_674632

def dessert := {cake, pie, ice_cream, pudding}

def valid_dessert_menus (menus : list dessert) : Prop :=
  menus.length = 7 ∧
  menus.nth 4 = some pie ∧
  menus.nth 2 = some ice_cream ∧
  ∀ i, i < 6 → menus.nth i ≠ menus.nth (i + 1)

theorem dessert_menus_count :
  ∃ menus : list dessert, valid_dessert_menus menus ∧ list.length (filter valid_dessert_menus (permutations dessert)) = 324 :=
sorry

end dessert_menus_count_l674_674632


namespace painting_methods_count_l674_674625

/-- A $2 × 2$ grid has 4 squares which can be red or green, with constraints:
   - A green square cannot have a red square immediately above or to the right of it.
   - All squares being green or none being green is allowed.
   We want to show that the number of valid painting configurations is 5.
-/
def count_valid_paintings : Nat :=
  5

theorem painting_methods_count :
  ∃ count : Nat, count = count_valid_paintings ∧ count = 5 := by
  use 5
  split
  . exact rfl
  . exact rfl
  sorry -- proof steps go here

end painting_methods_count_l674_674625


namespace marco_cards_l674_674879

theorem marco_cards (C : ℕ) 
  (h1 : 1/4 * C = ⌊1/4 * C⌋) 
  (h2 : 1/5 * (1/4 * C) = 25) : 
  C = 500 :=
sorry

end marco_cards_l674_674879


namespace bill_profit_difference_l674_674257

theorem bill_profit_difference :
  (∀ P SP NP NSP : ℝ,
    SP = 1.10 * P →
    P = 879.9999999999993 / 1.10 →
    NP = 0.90 * P →
    NSP = 1.30 * NP →
    NSP - SP = 56) :=
by
  assume P SP NP NSP : ℝ,
  assume h1 : SP = 1.10 * P,
  assume h2 : P = 879.9999999999993 / 1.10,
  assume h3 : NP = 0.90 * P,
  assume h4 : NSP = 1.30 * NP,
  sorry

end bill_profit_difference_l674_674257


namespace polynomial_f_property_l674_674321

open Polynomial

-- Define the polynomial f with integer coefficients and the conditions on the roots
variables {R : Type*} [CommRing R] {f : R[X]}
variables {r1 r2 r3 r4 : R} [IsDomain R]

noncomputable def example (r1 r2 r3 r4 : ℤ) : Prop :=
  ∀ x : ℤ, (poly.eval x f ≠ 1) ∧ (poly.eval x f ≠ 3) ∧ (poly.eval x f ≠ 6) ∧ (poly.eval x f ≠ 9)

theorem polynomial_f_property (f : ℤ[X])
  (h_coeffs : ∀ i, i < f.nat_degree → (f.coeff i : ℤ))
  (h_roots : f.eval r1 = 4 ∧ f.eval r2 = 4 ∧ f.eval r3 = 4 ∧ f.eval r4 = 4)
  (distinct_roots : r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4) :
  example r1 r2 r3 r4 := sorry

end polynomial_f_property_l674_674321


namespace closest_point_on_plane_l674_674299

theorem closest_point_on_plane (Q : ℝ × ℝ × ℝ)
  (Q_in_plane : Q.1 = 102 / 61 ∧ Q.2 = 76 / 61 ∧ Q.3 = 214 / 61)
  (point_in_plane : 4 * Q.1 - 3 * Q.2 + 6 * Q.3 = 24) : 
  Q = (⟨102 / 61, 76 / 61, 214 / 61⟩) := 
sorry

end closest_point_on_plane_l674_674299


namespace probability_of_non_adjacent_zeros_l674_674389

-- Define the total number of arrangements of 3 ones and 2 zeros
def totalArrangements : ℕ := Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

-- Define the number of arrangements where the 2 zeros are together
def adjacentZerosArrangements : ℕ := 2 * Nat.factorial 4 / (Nat.factorial 3 * Nat.factorial 1)

-- Calculate the desired probability
def nonAdjacentZerosProbability : ℚ := 
  1 - (adjacentZerosArrangements.toRat / totalArrangements.toRat)

theorem probability_of_non_adjacent_zeros :
  nonAdjacentZerosProbability = 3/5 :=
sorry

end probability_of_non_adjacent_zeros_l674_674389


namespace complementary_angle_beta_l674_674337

theorem complementary_angle_beta (α β : ℝ) (h_compl : α + β = 90) (h_alpha : α = 40) : β = 50 :=
by
  -- Skipping the proof, which initial assumption should be defined.
  sorry

end complementary_angle_beta_l674_674337


namespace solution_l674_674231

-- Define the conditions based on the given problem
variables {A B C D : Type}
variables {AB BC CD DA : ℝ} (h1 : AB = 65) (h2 : BC = 105) (h3 : CD = 125) (h4 : DA = 95)
variables (cy_in_circle : CyclicQuadrilateral A B C D)
variables (circ_inscribed : TangentialQuadrilateral A B C D)

-- Function that computes the absolute difference between segments x and y on side of length CD
noncomputable def find_absolute_difference (x y : ℝ) (h5 : x + y = 125) : ℝ := |x - y|

-- The proof statement
theorem solution :
  ∃ (x y : ℝ), x + y = 125 ∧
  (find_absolute_difference x y (by sorry) = 14) := sorry

end solution_l674_674231


namespace sum_of_roots_g_equals_2010_l674_674218

noncomputable def g : ℝ → ℝ := sorry

theorem sum_of_roots_g_equals_2010 :
  (∀ (x : ℝ), x ≠ 0 → 3 * g x + g (1 / x) = 7 * x + 6) →
  (finset.univ.filter (λ x, g x = 2010)).sum id ≈ 765 :=
by
  sorry

end sum_of_roots_g_equals_2010_l674_674218


namespace length_XY_l674_674450

-- Definitions for the given conditions
variables {P Q R X Y : Type}
variable [fact (is_isosceles_triangle PQR)]
variable (area_PQR : ℝ := 180)
variable (area_trapezoid : ℝ := 135)
variable (altitude_PQR : ℝ := 30)
variable (area_PXY : ℝ := 45)
variable (base_QR : ℝ := 12)

-- Theorem statement corresponding to the mathematical proof problem
theorem length_XY 
  (h1 : is_isosceles_triangle PQR)
  (h2 : area PQR = 180)
  (h3 : altitude PQR = 30)
  (h4 : area_trapezoid = 135)
  (h5 : divides_into (PQR, PXY, isosceles_trapezoid))
  (h6 : similar PXY PQR)
: length XY = 6 :=
by
sorry

end length_XY_l674_674450


namespace factor_expression_l674_674728

variable (y : ℝ)

theorem factor_expression : 64 - 16 * y ^ 3 = 16 * (2 - y) * (4 + 2 * y + y ^ 2) := by
  sorry

end factor_expression_l674_674728


namespace correct_statement_l674_674661

-- Definitions
def certain_event (P : ℝ → Prop) : Prop := P 1
def impossible_event (P : ℝ → Prop) : Prop := P 0
def uncertain_event (P : ℝ → Prop) : Prop := ∀ p, 0 < p ∧ p < 1 → P p

-- Theorem to prove
theorem correct_statement (P : ℝ → Prop) :
  (certain_event P ∧ impossible_event P ∧ uncertain_event P) →
  (∀ p, P p → p = 1)
:= by
  sorry

end correct_statement_l674_674661


namespace solution_to_inequality_system_l674_674523

theorem solution_to_inequality_system :
  (∀ x : ℝ, 2 * (x - 1) < x + 2 → (x + 1) / 2 < x → 1 < x ∧ x < 4) :=
by
  intros x h1 h2
  sorry

end solution_to_inequality_system_l674_674523


namespace system_solution_y_greater_than_five_l674_674279

theorem system_solution_y_greater_than_five (m x y : ℝ) :
  (y = (m + 1) * x + 2) → 
  (y = (3 * m - 2) * x + 5) → 
  y > 5 ↔ 
  m ≠ 3 / 2 := 
sorry

end system_solution_y_greater_than_five_l674_674279


namespace team_formation_problem_l674_674743

def num_team_formation_schemes : Nat :=
  let comb (n k : Nat) : Nat := Nat.choose n k
  (comb 5 1 * comb 4 2) + (comb 5 2 * comb 4 1)

theorem team_formation_problem :
  num_team_formation_schemes = 70 :=
sorry

end team_formation_problem_l674_674743


namespace alyssa_bought_224_new_cards_l674_674852

theorem alyssa_bought_224_new_cards
  (initial_cards : ℕ)
  (after_purchase_cards : ℕ)
  (h1 : initial_cards = 676)
  (h2 : after_purchase_cards = 900) :
  after_purchase_cards - initial_cards = 224 :=
by
  -- Placeholder to avoid proof since it's explicitly not required 
  sorry

end alyssa_bought_224_new_cards_l674_674852


namespace tan_theta_parallel_vectors_l674_674333

theorem tan_theta_parallel_vectors (θ : ℝ) (h : (sin θ, 1) ∥ (cos θ, -2)) : Real.tan θ = -2 :=
by
  -- Proof goes here
  sorry

end tan_theta_parallel_vectors_l674_674333


namespace log_expression_eval_log_inequality_range_l674_674623

-- Problem 1: Prove the logarithm expression evaluates to -1
theorem log_expression_eval :
  2 * log 3 2 - log 3 (32 / 9) + log 3 8 - 5 ^ (log 5 3) = -1 := by sorry

-- Problem 2: Determine the range of x based on the given inequality
theorem log_inequality_range (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (a > 1 → ∀ x : ℝ, log a (2*x + 1) < log a (4*x - 3) → x > 5/2) ∧
  (0 < a ∧ a < 1 → ∀ x : ℝ, log a (2*x + 1) < log a (4*x - 3) → 3/4 < x ∧ x < 2) := by sorry

end log_expression_eval_log_inequality_range_l674_674623


namespace number_of_pies_is_correct_l674_674513

def weight_of_apples : ℕ := 120
def weight_for_applesauce (w : ℕ) : ℕ := w / 2
def weight_for_pies (w wholly_app : ℕ) : ℕ := w - wholly_app
def pies (weight_per_pie total_weight : ℕ) : ℕ := total_weight / weight_per_pie

theorem number_of_pies_is_correct :
  pies 4 (weight_for_pies weight_of_apples (weight_for_applesauce weight_of_apples)) = 15 :=
by
  sorry

end number_of_pies_is_correct_l674_674513


namespace no_same_last_four_digits_pow_l674_674250

theorem no_same_last_four_digits_pow (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  (5^n % 10000) ≠ (6^m % 10000) :=
by sorry

end no_same_last_four_digits_pow_l674_674250


namespace no_tetrahedron_example_l674_674721

def no_tetrahedron_with_given_edges (a b c d e f : ℝ) : Prop :=
  ∀ (T : Tetrahedron), 
    (T.opposite_edges = [(a, a), (b, b), (c, d), (e, f)]) → false

theorem no_tetrahedron_example : no_tetrahedron_with_given_edges 12 12.5 5.13 13 12.5 13 :=
by
  sorry

end no_tetrahedron_example_l674_674721


namespace greatest_integer_solution_l674_674715

theorem greatest_integer_solution :
  (∃ x : ℤ, ∃ x_real : ℝ, (|7 * x_real - 3| - 2 * x_real < 5 - 3 * x_real) ∧ (x_real = x) ∧ 
            (∀ y : ℤ, ∃ y_real : ℝ, (|7 * y_real - 3| - 2 * y_real < 5 - 3 * y_real) ∧ 
            (y_real = y) → y ≤ x)) :=
begin
  sorry
end

end greatest_integer_solution_l674_674715


namespace limit_proof_l674_674675

noncomputable def limit_problem : ℝ :=
  limit (fun x => (exp (3 * x) - exp (-2 * x)) / (2 * arcsin x - sin x)) 0

theorem limit_proof : limit_problem = 5 := by
  sorry

end limit_proof_l674_674675


namespace AB_eq_CD_l674_674506

-- Define the geometric context
variables {X Y Z A B C D : Point} (ℓ : Line)

-- Conditions
-- X, Y, Z lie on a straight line in that order
axiom collinear_XYZ : Collinear X Y Z

-- Circle definitions with specified diameters
def ω1 : Circle := Circle.mk X Z -- circle with diameter XZ
def ω2 : Circle := Circle.mk X Y -- circle with diameter XY
def ω3 : Circle := Circle.mk Y Z -- circle with diameter YZ

-- Line ℓ intersects ω1 at A
axiom A_on_ω1 : A ∈ ω1
axiom ℓ_intersects_A : A ∈ ℓ

-- Line ℓ intersects ω2 at B
axiom B_on_ω2 : B ∈ ω2
axiom ℓ_intersects_B : B ∈ ℓ

-- Line ℓ passes through Y
axiom ℓ_through_Y : Y ∈ ℓ

-- Line ℓ intersects ω3 at C
axiom C_on_ω3 : C ∈ ω3
axiom ℓ_intersects_C : C ∈ ℓ

-- Line ℓ intersects ω1 again at D
axiom D_on_ω1 : D ∈ ω1
axiom ℓ_intersects_D_again : D ∈ ℓ

-- The proof goal
theorem AB_eq_CD : dist A B = dist C D :=
by sorry

end AB_eq_CD_l674_674506


namespace total_score_is_correct_l674_674491

def dad_points : ℕ := 7
def olaf_points : ℕ := 3 * dad_points
def total_points : ℕ := dad_points + olaf_points

theorem total_score_is_correct : total_points = 28 := by
  sorry

end total_score_is_correct_l674_674491


namespace gain_percent_is_30_l674_674975

-- Given conditions
def CostPrice : ℕ := 100
def SellingPrice : ℕ := 130
def Gain : ℕ := SellingPrice - CostPrice
def GainPercent : ℕ := (Gain * 100) / CostPrice

-- The theorem to be proven
theorem gain_percent_is_30 :
  GainPercent = 30 := sorry

end gain_percent_is_30_l674_674975


namespace sum_of_roots_l674_674771

theorem sum_of_roots : ∀ (α β : ℝ), (log 3 α + α - 3 = 0) → (3^β + β - 3 = 0) → α + β = 3 :=
by
  assume α β hα hβ
  sorry

end sum_of_roots_l674_674771


namespace max_min_values_theta_range_for_monotonicity_l674_674358

-- Define the function f(x) given theta
def f (x : ℝ) (θ : ℝ) : ℝ := x^2 + 2 * x * Real.tan θ - 1

-- Define the conditions
def theta_cond (θ : ℝ) : Prop := θ > -π/2 ∧ θ < π/2

-- Prove the problem statements
theorem max_min_values (θ : ℝ) (x : ℝ) (hθ : θ = -π/4) (hx : -1 ≤ x ∧ x ≤ Real.sqrt 3) :
  f x θ = x^2 - 2*x - 1 ∧
  (∀ x, -1 ≤ x ∧ x ≤ Real.sqrt 3 → f x θ ≤ 2) ∧
  (∃ x, -1 ≤ x ∧ x ≤ Real.sqrt 3 ∧ f x θ = 2) ∧
  (∀ x, -1 ≤ x ∧ x ≤ Real.sqrt 3 → f x θ ≥ -2) ∧
  (∃ x, -1 ≤ x ∧ x ≤ Real.sqrt 3 ∧ f x θ = -2) :=
by sorry

theorem theta_range_for_monotonicity :
  (∀ (θ : ℝ), theta_cond θ → (-θ ≤ -Real.sqrt 3 ∨ -θ ≥ 1) ↔ (θ ∈ (-π / 2 : Set ℝ) ∪ (-π / 4 : Set ℝ) ∪ (π / 3 : Set ℝ) ∪ (π/ 2 : Set ℝ))) :=
by sorry

end max_min_values_theta_range_for_monotonicity_l674_674358


namespace find_f_prime_at_1_l674_674794

variable (f : ℝ → ℝ)

-- Initial condition
variable (h : ∀ x, f x = x^2 + deriv f 2 * (Real.log x - x))

-- The goal is to prove that f'(1) = 2
theorem find_f_prime_at_1 : deriv f 1 = 2 :=
by
  sorry

end find_f_prime_at_1_l674_674794


namespace jessie_weight_before_jogging_l674_674658

-- Definitions: conditions from the problem statement
variables (lost_weight current_weight : ℤ)
-- Conditions
def condition_lost_weight : Prop := lost_weight = 126
def condition_current_weight : Prop := current_weight = 66

-- Proposition to be proved
theorem jessie_weight_before_jogging (W_before_jogging : ℤ) :
  condition_lost_weight lost_weight → condition_current_weight current_weight →
  W_before_jogging = current_weight + lost_weight → W_before_jogging = 192 :=
by
  intros
  sorry

end jessie_weight_before_jogging_l674_674658


namespace containers_needed_l674_674511

-- Define the conditions: 
def weight_in_pounds : ℚ := 25 / 2
def ounces_per_pound : ℚ := 16
def ounces_per_container : ℚ := 50

-- Define the total weight in ounces
def total_weight_in_ounces := weight_in_pounds * ounces_per_pound

-- Theorem statement: Number of containers.
theorem containers_needed : total_weight_in_ounces / ounces_per_container = 4 := 
by
  -- Write the proof here
  sorry

end containers_needed_l674_674511


namespace maria_score_l674_674044

theorem maria_score (total_questions correct_answers incorrect_answers unanswered_questions : ℕ)
  (h1 : total_questions = 20)
  (h2 : correct_answers = 15)
  (h3 : incorrect_answers = 3)
  (h4 : unanswered_questions = 2)
  (scoring_system : ℕ → ℕ → ℕ → ℕ)
  (h5 : scoring_system correct_answers incorrect_answers unanswered_questions = correct_answers * 1) :
  scoring_system correct_answers incorrect_answers unanswered_questions = 15 := by
  rw [h5, h2]
  sorry

end maria_score_l674_674044


namespace preferred_sequence_bound_l674_674458

theorem preferred_sequence_bound (k n : ℕ) (h_pos_k : 0 < k) (h_pos_n : 0 < n)
  (A : Fin k → Matrix (Fin n) (Fin n) ℝ)
  (h1 : ∀ i : Fin k, A i ⬝ A i ≠ 0)
  (h2 : ∀ i j : Fin k, i ≠ j → A i ⬝ A j = 0) :
  k ≤ n := 
sorry

end preferred_sequence_bound_l674_674458


namespace product_negative_probability_l674_674945

theorem product_negative_probability:
  let S := ({-5, -8, 7, 4, -2, 0, 6} : Finset ℤ),
      neg_ints := ({-5, -8, -2} : Finset ℤ),
      pos_ints := ({7, 4, 6} : Finset ℤ) in
  (S.card = 7) ∧
  (neg_ints.card = 3) ∧
  (pos_ints.card = 3) ∧
  (0 ∈ S) →
  let total_ways := (S.card.choose 3),
      neg_product_ways := ((neg_ints.card.choose 2) * (pos_ints.card.choose 1) +
                            neg_ints.card.choose 3) in
  (neg_product_ways : ℚ) / total_ways = 2 / 7 :=
by
  sorry

end product_negative_probability_l674_674945


namespace min_value_of_x_l674_674403

open Real

-- Defining the conditions
def condition1 (x : ℝ) : Prop := x > 0
def condition2 (x : ℝ) : Prop := log x ≥ 2 * log 3 + (1/3) * log x

-- Statement of the theorem
theorem min_value_of_x (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x ≥ 27 :=
sorry

end min_value_of_x_l674_674403


namespace average_words_per_hour_l674_674244

/-- Prove that given a total of 50,000 words written in 100 hours with the 
writing output increasing by 10% each subsequent hour, the average number 
of words written per hour is 500. -/
theorem average_words_per_hour 
(words_total : ℕ) 
(hours_total : ℕ) 
(increase : ℝ) :
  words_total = 50000 ∧ hours_total = 100 ∧ increase = 0.1 →
  (words_total / hours_total : ℝ) = 500 :=
by 
  intros h
  sorry

end average_words_per_hour_l674_674244


namespace shaded_area_ratio_correct_l674_674974

noncomputable def shaded_area_ratio (r : ℝ) : ℝ :=
  let area_largest_semicircle := (8 * π * r^2)
  let area_semicircle_AC := (9 / 2 * π * r^2)
  let area_semicircle_CB := (1 / 2 * π * r^2)
  let shaded_area := area_largest_semicircle - (area_semicircle_AC + area_semicircle_CB)
  let area_circle_with_CD := (4 * π * r^2)
  shaded_area / area_circle_with_CD

theorem shaded_area_ratio_correct (r : ℝ) : shaded_area_ratio r = 3 / 4 :=
  sorry

end shaded_area_ratio_correct_l674_674974


namespace part_I_part_II_l674_674329

noncomputable section

open Real
open Classical

variables (a b c A B C : ℝ)

-- Assuming conditions
axiom tri_condition_1 : ∀ (A B : ℝ), 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ A + B < π 
axiom tri_condition_2 : sin (A + B) / (sin A + sin B) = (a - b) / (a - c)
axiom sides_condition : ∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0

-- Proof Part (I): to show B = π / 3
theorem part_I : ∀ (a b c A C : ℝ) (H1 : sin (A + π/3) = sin C)
                (H2: (sin (A + π/3) / (sin A + sin (π/3))) = (a - b) / (a - c)), 
                B = π / 3 :=
by
  sorry

-- Proof Part (II): to show max area S = 3 * sqrt 3 / 4 when b = sqrt 3
theorem part_II : ∀ (a c : ℝ), b = sqrt 3 → 
                  (∀ (A B C : ℝ) (H3 : area (a b c A B C)) = (1/2) * a * c * sin (π/3) → 
                  S ≤ 3 * sqrt 3 / 4 :=
by
  sorry

end part_I_part_II_l674_674329


namespace smallest_d_proof_l674_674075

noncomputable def smallest_possible_d : ℝ := 17

theorem smallest_d_proof (S : set (ℝ × ℝ)) (hS : S.card = 100) 
  (h_diff : ∀ p1 p2 ∈ S, p1 ≠ p2 → dist p1 p2 ≠ dist p2 p3) 
  (h_largest : ∀ p1 p2 ∈ S, dist p1 p2 ≤ 30)
  (A : ℝ × ℝ) (hA : A ∈ S) :
  ∃ B C ∈ S, (B ≠ A ∧ C ≠ B) ∧
  (dist A B = 30) ∧
  (d = round (dist B C)) ∧
  d = smallest_possible_d :=
begin
  sorry
end

end smallest_d_proof_l674_674075


namespace largest_percentage_increase_is_2013_to_2014_l674_674043

-- Defining the number of students in each year as constants
def students_2010 : ℕ := 50
def students_2011 : ℕ := 56
def students_2012 : ℕ := 62
def students_2013 : ℕ := 68
def students_2014 : ℕ := 77
def students_2015 : ℕ := 81

-- Defining the percentage increase between consecutive years
def percentage_increase (a b : ℕ) : ℚ := ((b - a) : ℚ) / (a : ℚ)

-- Calculating all the percentage increases
def pi_2010_2011 := percentage_increase students_2010 students_2011
def pi_2011_2012 := percentage_increase students_2011 students_2012
def pi_2012_2013 := percentage_increase students_2012 students_2013
def pi_2013_2014 := percentage_increase students_2013 students_2014
def pi_2014_2015 := percentage_increase students_2014 students_2015

-- The theorem stating the largest percentage increase is between 2013 and 2014
theorem largest_percentage_increase_is_2013_to_2014 :
  max (pi_2010_2011) (max (pi_2011_2012) (max (pi_2012_2013) (max (pi_2013_2014) (pi_2014_2015)))) = pi_2013_2014 :=
sorry

end largest_percentage_increase_is_2013_to_2014_l674_674043


namespace prob1_prob2_l674_674876

-- Problem (1)
theorem prob1 (x : ℝ) : |x - 2| ≥ 7 - |x - 1| → x ∈ set.Iic (-2) ∪ set.Ici 5 :=
by
  sorry

-- Problem (2)
theorem prob2 (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1 / m + 1 / (2 * n) = 1) : 
  m + 4 * n ≥ 2 * real.sqrt 2 + 3 :=
by
  sorry

end prob1_prob2_l674_674876


namespace feet_per_mile_l674_674415

theorem feet_per_mile 
  (travel_distance : ℝ)
  (travel_time : ℝ)
  (object_speed_mph : ℝ)
  (one_hour_in_seconds : ℝ)
  (x_feet_per_mile : ℝ)
  (travel_distance = 200)
  (travel_time = 2)
  (object_speed_mph = 68.18181818181819)
  (one_hour_in_seconds = 3600) :
  x_feet_per_mile = 5280 :=
sorry

end feet_per_mile_l674_674415


namespace speed_of_first_train_l674_674167

theorem speed_of_first_train
  (v : ℝ)
  (d : ℝ)
  (distance_between_stations : ℝ := 450)
  (speed_of_second_train : ℝ := 25)
  (additional_distance_first_train : ℝ := 50)
  (meet_time_condition : d / v = (d - additional_distance_first_train) / speed_of_second_train)
  (total_distance_condition : d + (d - additional_distance_first_train) = distance_between_stations) :
  v = 31.25 :=
by {
  sorry
}

end speed_of_first_train_l674_674167


namespace find_black_cells_l674_674678

-- Define the board type and the input configuration for Lara's board
def Board := Array (Array Nat)

def LarasBoard : Board :=
#[ #[1, 2, 1, 1], #[0, 2, 1, 2], #[2, 3, 3, 1], #[1, 0, 2, 1] ]

-- Define the problem: Finding the number of black cells on Camila's board
theorem find_black_cells (b : Board) : (number_of_black_cells b) = 4 :=
by
  sorry

-- Assume the definition of the function counting black cells based on the problem conditions
def number_of_black_cells (b : Board) : Nat := sorry

end find_black_cells_l674_674678


namespace least_five_digit_palindrome_divisible_by_5_l674_674964

theorem least_five_digit_palindrome_divisible_by_5 : ∃ n : ℕ, 
  (∃ (a b c : ℕ), a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ n = 10001 * a + 1010 * b + 100 * c) ∧ 
  n % 5 = 0 ∧ 
  (∀ m : ℕ, (∃ (a' b' c' : ℕ), a' ≠ 0 ∧ a' ≤ 9 ∧ b' ≤ 9 ∧ c' ≤ 9 ∧ m = 10001 * a' + 1010 * b' + 100 * c' ∧ m % 5 = 0) → n ≤ m) :=
begin
  use 50005,
  split,
  { use [5, 0, 0],
    simp,
    norm_num },
  split,
  { norm_num },
  { intros m hm,
    rcases hm with ⟨a', b', c', ha', hle_a', hle_b', hle_c', rfl, hm5⟩,
    interval_cases a' with ha'c,
    { exfalso,
      norm_num at *,
      contradiction },
    rw [mul_comm, nat.mul_add_left] at hm5,
    linarith },
end

end least_five_digit_palindrome_divisible_by_5_l674_674964


namespace olivia_henry_matchups_l674_674502

-- Definitions of conditions
def num_players : ℕ := 12
def num_players_each_game : ℕ := 6
def player_olivia : ℕ := 1
def player_henry : ℕ := 2

-- Statement of the problem
theorem olivia_henry_matchups : 
  (number_of_times_olivia_plays_with_henry num_players num_players_each_game player_olivia player_henry) = 210 :=
sorry

end olivia_henry_matchups_l674_674502


namespace company_picnic_attendance_l674_674037

theorem company_picnic_attendance :
  ∀ (employees men women men_attending women_attending : ℕ)
  (h_employees : employees = 100)
  (h_men : men = 55)
  (h_women : women = 45)
  (h_men_attending: men_attending = 11)
  (h_women_attending: women_attending = 18),
  (100 * (men_attending + women_attending) / employees) = 29 := 
by
  intros employees men women men_attending women_attending 
         h_employees h_men h_women h_men_attending h_women_attending
  sorry

end company_picnic_attendance_l674_674037


namespace k_range_proof_l674_674054

/- Define points in the Cartesian plane as ordered pairs. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/- Define two points P and Q. -/
def P : Point := { x := -1, y := 1 }
def Q : Point := { x := 2, y := 2 }

/- Define the line equation. -/
def line_equation (k : ℝ) (x : ℝ) : ℝ :=
  k * x - 1

/- Define the range of k. -/
def k_range (k : ℝ) : Prop :=
  1 / 3 < k ∧ k < 3 / 2

/- Theorem statement. -/
theorem k_range_proof (k : ℝ) (intersects_PQ_extension : ∀ k : ℝ, ∀ x : ℝ, ((P.y ≤ line_equation k x ∧ line_equation k x ≤ Q.y) ∧ line_equation k x ≠ Q.y) → k_range k) :
  ∀ k, k_range k :=
by
  sorry

end k_range_proof_l674_674054


namespace incorrect_conclusion_l674_674356

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (2 * x)

theorem incorrect_conclusion :
  ¬ (∀ x : ℝ, f ( (3 * Real.pi) / 4 - x ) + f x = 0) :=
by
  sorry

end incorrect_conclusion_l674_674356


namespace rainfall_third_day_is_18_l674_674955

-- Define the conditions including the rainfall for each day
def rainfall_first_day : ℕ := 4
def rainfall_second_day : ℕ := 5 * rainfall_first_day
def rainfall_third_day : ℕ := (rainfall_first_day + rainfall_second_day) - 6

-- Prove that the rainfall on the third day is 18 inches
theorem rainfall_third_day_is_18 : rainfall_third_day = 18 :=
by
  -- Use the definitions and directly state that the proof follows
  sorry

end rainfall_third_day_is_18_l674_674955


namespace books_borrowed_by_lunchtime_l674_674813

theorem books_borrowed_by_lunchtime (x : ℕ) :
  (∀ x : ℕ, 100 - x + 40 - 30 = 60) → (x = 50) :=
by
  intro h
  have eqn := h x
  sorry

end books_borrowed_by_lunchtime_l674_674813


namespace dart_game_solution_l674_674181

theorem dart_game_solution (x y z : ℕ) (h_x : 8 * x + 9 * y + 10 * z = 100) (h_y : x + y + z > 11) :
  (x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0) :=
by
  sorry

end dart_game_solution_l674_674181


namespace solve_equation_l674_674901

theorem solve_equation (x : ℝ) (h : (x - 1) / 2 = 1 - (x + 2) / 3) : x = 1 :=
sorry

end solve_equation_l674_674901


namespace first_20_kg_cost_100_l674_674191

variables (l q : ℝ) (cost_33 cost_36 : ℝ)

def cost (x : ℝ) := if x ≤ 30 then x * l else (30 * l) + (x - 30) * q

theorem first_20_kg_cost_100 (h1 : cost 33 = 168)
                           (h2 : cost 36 = 186) :
  20 * l = 100 :=
by
  -- Definitions directly from conditions
  have : cost (30 : ℝ) = 30 * l := by
    unfold cost
    simp only [if_pos]
  have C30_plus_3Cadd : 30 * l + 3 * q = 168 := by
    rw [cost] at h1
    split_ifs at h1 with h3
    contradiction
    assumption
  have C30_plus_6Cadd : 30 * l + 6 * q = 186 := by
    rw [cost] at h2
    split_ifs at h2 with h4
    contradiction
    assumption
  have q_value : q = 6 := sorry
  have l_value : l = 5 := sorry
  exact sorry

end first_20_kg_cost_100_l674_674191


namespace volume_proof_l674_674646

-- Define a structure representing the problem states
structure Shape :=
  (units : ℕ := 14) -- 14 unit squares
  (rect5x2 : ℕ × ℕ := (5, 2)) -- 5 x 2 rectangle
  (rect1x2_adj : ℕ × ℕ := (1, 2)) -- 1 x 2 rectangle adjacent to the 5 x 2 rectangle
  (rect2x3_above : ℕ × ℕ := (2, 3)) -- 2 x 3 rectangle positioned above the 5 x 2 rectangle
  (axis : char := 'y') -- rotation about the y-axis

-- Define the total volume calculation based on given shape
def volume_of_rotated_shape (shape : Shape) : ℝ :=
  let v1 := Real.pi * (2:ℝ)^2 * (5:ℝ) -- Volume of first cylinder from 5 x 2 rectangle
  let v2 := Real.pi * (2:ℝ)^2 * (1:ℝ) -- Volume of second cylinder from 1 x 2 rectangle
  let v3 := Real.pi * (3:ℝ)^2 * (2:ℝ) -- Volume of third cylinder from 2 x 3 rectangle
  v1 + v2 + v3

-- Prove the volume is 42π cubic units
theorem volume_proof (shape : Shape) : volume_of_rotated_shape shape = 42 * Real.pi := by
  sorry

end volume_proof_l674_674646


namespace zeros_at_end_of_quotient_factorial_l674_674820

def count_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625

theorem zeros_at_end_of_quotient_factorial :
  count_factors_of_five 2018 - count_factors_of_five 30 - count_factors_of_five 11 = 493 :=
by
  sorry

end zeros_at_end_of_quotient_factorial_l674_674820


namespace problem_part1_problem_part2_l674_674004

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + 2 * Real.pi / 3) + sqrt 3 * Real.sin (2 * x)

theorem problem_part1 : 
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f(x + T) = f(x) ∧ T = Real.pi) ∧ 
  (∀ x : ℝ, -1 <= f(x) ∧ f(x) <= 1) ∧ 
  (∃ x : ℝ, f(x) = 1) :=
sorry

theorem problem_part2 (A B C : ℝ) (hC : 0 < C ∧ C < Real.pi) (hAC : AC = 1) (hBC : BC = 3)
  (h_f : f (C / 2) = -1 / 2) :
  ∃ s : ℝ, s = (3 * sqrt 21) / 14 :=
sorry

end problem_part1_problem_part2_l674_674004


namespace income_is_108000_l674_674187

theorem income_is_108000 (S I : ℝ) (h1 : S / I = 5 / 9) (h2 : 48000 = I - S) : I = 108000 :=
by
  sorry

end income_is_108000_l674_674187


namespace simplify_expression_l674_674184

theorem simplify_expression :
  (4.625 - 13/18 * 9/26) / (9/4) + 2.5 / 1.25 / 6.75 / 1 + 53/68 / ((1/2 - 0.375) / 0.125 + (5/6 - 7/12) / (0.358 - 1.4796 / 13.7)) = 17/27 :=
by sorry

end simplify_expression_l674_674184


namespace tissues_used_l674_674265

theorem tissues_used (initial_tissues : ℕ) (final_tissues : ℕ) (total_used : ℕ) :
  initial_tissues = 97 →
  final_tissues = 58 →
  total_used = initial_tissues - final_tissues →
  total_used = 39 :=
by
  intros h_initial h_final h_total_used
  rw [h_initial, h_final] at h_total_used
  rw h_total_used
  norm_num
  sorry

end tissues_used_l674_674265


namespace triangle_area_equals_2r_l674_674107

theorem triangle_area_equals_2r (ABC : Triangle) (BC : Length) (r_a r : Length) (h1 : BC = 1) (h2 : r_a = 2 * r) : 
  area ABC = 2 * r := sorry

end triangle_area_equals_2r_l674_674107


namespace example_problem_l674_674484

variable {𝕜 : Type*} [LinearOrderedField 𝕜]
variable {f : 𝕜 → 𝕜}
variable {x : 𝕜}

theorem example_problem (hf : ∀ x : 𝕜, (x-2) * (deriv f x) ≥ 0 ) :
  f 1 + f 3 ≥ 2 * f 2 :=
sorry

end example_problem_l674_674484


namespace Eulerian_Cycle_exists_l674_674133

-- Definitions of conditions
structure CastleConfig where
  central_hall : bool
  halls_divisions : ℕ → ℕ
  room_setup : (ℕ × ℕ) → ℕ → ℕ

def is_even_degree (v : ℕ) : Prop :=
  v % 2 = 0

noncomputable def validate_castle_config : CastleConfig := 
  { central_hall := true,
    halls_divisions := λ i, 9,
    room_setup := λ (x, y), 9 }

-- Main theorem: The baron can visit all residential rooms exactly once and return.
theorem Eulerian_Cycle_exists : 
  (∀ p, p ∈ (residential_rooms validate_castle_config) → is_even_degree (degree_of p validate_castle_config)) →
  True :=
by
  -- Each livable room has even degree 4, no need to check specific structure due to problem constraints.
  intros
  exact True.intro

def residential_rooms (config : CastleConfig) : List (ℕ × ℕ) :=
  sorry -- implementation of valid residential rooms configuration

def degree_of (room : (ℕ × ℕ)) (config : CastleConfig) : ℕ :=
  sorry -- implementation determining the degree of each room


end Eulerian_Cycle_exists_l674_674133


namespace min_value_collinear_l674_674465

variables {α : Type*} [linear_ordered_field α] 
variables (e1 e2 : α →₀ ℝ)

noncomputable def vec_ab (a : α) : α →₀ ℝ := (a - 1) • e1 + e2
noncomputable def vec_ac (b : α) : α →₀ ℝ := b • e1 - 2 • e2

theorem min_value_collinear (a b : α) (h1 : 0 < a) (h2 : 0 < b) 
  (h_collinear : ∃ (λ : α), vec_ab e1 e2 a = λ • vec_ac e1 e2 b) :
  ∃ (a_min : α), a_min = 1 / 2 ∧ (1 / a_min + 2 / (2 - 2 * a_min)) = 4 :=
by
  sorry

end min_value_collinear_l674_674465


namespace snow_leopards_arrangement_l674_674880

theorem snow_leopards_arrangement : 
  ∃ (perm : Fin 9 → Fin 9), 
    (∀ i, perm i ≠ perm j → i ≠ j) ∧ 
    (perm 0 < perm 1 ∧ perm 8 < perm 1 ∧ perm 0 < perm 8) ∧ 
    (∃ count_ways, count_ways = 4320) :=
sorry

end snow_leopards_arrangement_l674_674880


namespace John_leftover_money_l674_674071

variables (q : ℝ)

def drinks_price (q : ℝ) : ℝ := 4 * q
def small_pizza_price (q : ℝ) : ℝ := q
def large_pizza_price (q : ℝ) : ℝ := 4 * q
def total_cost (q : ℝ) : ℝ := drinks_price q + small_pizza_price q + 2 * large_pizza_price q
def John_initial_money : ℝ := 50
def John_money_left (q : ℝ) : ℝ := John_initial_money - total_cost q

theorem John_leftover_money : John_money_left q = 50 - 13 * q :=
by
  sorry

end John_leftover_money_l674_674071


namespace interest_earned_is_correct_l674_674096

noncomputable def calculate_interest : ℝ :=
let P : ℝ := 1000
let r : ℝ := 0.06
let n : ℝ := 10
let A := P * (1 + r)^n in
A - P

theorem interest_earned_is_correct :
  calculate_interest = 790.848 :=
by
  sorry

end interest_earned_is_correct_l674_674096


namespace cricket_team_captain_age_l674_674041

theorem cricket_team_captain_age
  (C : ℕ)
  (total_team_members : ℕ := 11)
  (total_age_of_team : ℕ := 23 * total_team_members)
  (average_age_excluding_captain_wicketkeeper : ℕ := 22)
  (total_age_excluding_captain_wicketkeeper : ℕ := average_age_excluding_captain_wicketkeeper * 9)
  (total_age_with_captain_wicketkeeper : ℕ := total_age_of_team - total_age_excluding_captain_wicketkeeper) :
  2 * C + 1 = total_age_with_captain_wicketkeeper →
  C = 27 :=
begin
  intros h1,
  calc
  2 * C + 1 = 54 : by linarith [h1, total_age_with_captain_wicketkeeper]
  ...  = 54 := rfl,
  exact nat.div_eq C (by linarith : 2 * C = C + C)
end

end cricket_team_captain_age_l674_674041


namespace probability_zeros_not_adjacent_is_0_6_l674_674394

-- Define the total number of arrangements of 5 elements where we have 3 ones and 2 zeros
def total_arrangements : Nat := 5.choose 2

-- Define the number of arrangements where 2 zeros are adjacent
def adjacent_zeros_arrangements : Nat := 4.choose 1 * 2

-- Define the probability that the 2 zeros are not adjacent
def probability_not_adjacent : Rat := (total_arrangements - adjacent_zeros_arrangements) / total_arrangements

-- Prove the desired probability is 0.6
theorem probability_zeros_not_adjacent_is_0_6 : probability_not_adjacent = 3 / 5 := by
  sorry

end probability_zeros_not_adjacent_is_0_6_l674_674394


namespace percent_boys_not_in_science_club_is_correct_l674_674423

variables (total_students boys_ratio girls_ratio boys_in_science_club boys_not_in_science_club : ℕ)
variable (percent_boys_not_in_science_club : ℚ)

def ratio_boys_to_girls := boys_ratio / girls_ratio
def total_boys := (boys_ratio * total_students) / (boys_ratio + girls_ratio)
def boys_not_in_science := total_boys - boys_in_science_club
def percent_boys_not_in_science := (boys_not_in_science * 100) / total_students

theorem percent_boys_not_in_science_club_is_correct :
  boys_ratio = 3 →
  girls_ratio = 4 →
  total_students = 42 →
  boys_in_science_club = 5 →
  percent_boys_not_in_science (total_students:=total_students) (boys_ratio:=boys_ratio) (girls_ratio:=girls_ratio) (boys_in_science_club:=boys_in_science_club) (boys_not_in_science_club:=boys_not_in_science_club) ≈ 30.95 :=
by {
  sorry
}

end percent_boys_not_in_science_club_is_correct_l674_674423


namespace sandcastles_on_marks_beach_l674_674942

theorem sandcastles_on_marks_beach (M : ℕ)
  (h1 : ∀ (n : ℕ), (10 * n) = (number of towers on Mark's beach))
  (h2 : ∀ (n : ℕ), (3 * n) = (number of sandcastles on Jeff's beach))
  (h3 : ∀ (n : ℕ), (5 * (3 * n)) = (number of towers on Jeff's beach))
  (h4 : ∀ (n : ℕ), M + (10 * M) + (3 * M) + (15 * M) = 580) :
  M = 20 :=
by
  sorry

end sandcastles_on_marks_beach_l674_674942


namespace general_term_formula_l674_674548

-- Define the sequence
def sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 3 / 2
  | n => (-1)^(n+1) * (2*n + 1) / 2^n

theorem general_term_formula :
  ∀ n, sequence n = (-1)^(n+1) * (2*n + 1) / 2^n :=
by
  intro n
  sorry

end general_term_formula_l674_674548


namespace functional_equation_unique_zero_function_l674_674276

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_unique_zero_function :
  (∀ (x y : ℝ), f (x + y) = x * f x + y * f y) → (∀ x : ℝ, f x = 0) :=
begin
  assume h : ∀ (x y : ℝ), f (x + y) = x * f x + y * f y,
  sorry
end

end functional_equation_unique_zero_function_l674_674276


namespace roots_modulus_constraint_l674_674275

theorem roots_modulus_constraint (p q : ℝ) :
  (∀ (r : ℝ), r ∈ (λ x, x^3 + p * x + q = 0) → |r| ≤ 1) ↔ p > |q| - 1 :=
sorry

end roots_modulus_constraint_l674_674275


namespace hyperbola_eccentricity_no_common_points_when_r_is_sqrt6_two_common_points_when_r_is_2sqrt2_l674_674361

-- Lean 4 equivalent proof problem

-- Hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

-- Circle with radius r
def circle (x y r : ℝ) : Prop := x^2 + (y-3)^2 = r^2 ∧ r > 0

-- Eccentricity of Hyperbola C is √6 / 2
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 2

-- 1. Prove the eccentricity condition
theorem hyperbola_eccentricity : 
  let a := Real.sqrt 2
  let b := 1
  let e := Real.sqrt (1 + b^2 / a^2)
  e = Real.sqrt 6 / 2 :=
by
  let a := Real.sqrt 2
  let b := 1
  have e := Real.sqrt (1 + b^2 / a^2)
  sorry

-- 2. Prove no common points when r = √6
theorem no_common_points_when_r_is_sqrt6 (r : ℝ) : 
  r = Real.sqrt 6 → ∀ x y : ℝ, ¬ (hyperbola x y ∧ circle x y r) :=
by
  assume h_r_eq : r = Real.sqrt 6
  assume x y : ℝ
  unfold hyperbola circle, sorry

-- 3. Prove exactly two common points when r = 2√2
theorem two_common_points_when_r_is_2sqrt2 (r : ℝ) : 
  r = 2 * Real.sqrt 2 → ∃! (x y : ℝ), (hyperbola x y ∧ circle x y r) :=
by
  assume h_r_eq : r = 2 * Real.sqrt 2
  unfold hyperbola circle, sorry

end hyperbola_eccentricity_no_common_points_when_r_is_sqrt6_two_common_points_when_r_is_2sqrt2_l674_674361


namespace k_value_l674_674193

theorem k_value (m n k : ℤ) (h₁ : m + 2 * n + 5 = 0) (h₂ : (m + 2) + 2 * (n + k) + 5 = 0) : k = -1 :=
by sorry

end k_value_l674_674193


namespace smallest_cube_volume_l674_674893

noncomputable def sculpture_height : ℝ := 15
noncomputable def sculpture_base_radius : ℝ := 8
noncomputable def cube_side_length : ℝ := 16

theorem smallest_cube_volume :
  ∀ (h r s : ℝ), 
    h = sculpture_height ∧
    r = sculpture_base_radius ∧
    s = cube_side_length →
    s ^ 3 = 4096 :=
by
  intros h r s 
  intro h_def
  sorry

end smallest_cube_volume_l674_674893


namespace sum_of_n_conditions_l674_674327

open Nat

-- Define the sequence {a_n}
def a : ℕ+ → ℝ
| ⟨1, _⟩ := 3 / 2
| ⟨n + 1, hp⟩ := (3 - (∑ i in range n, a ⟨i + 1, nat.succ_pos i⟩)) / 2

-- Define the sum of the first n terms S_n
def S : ℕ+ → ℝ
| ⟨n + 1, hp⟩ := ∑ i in Finset.range (n + 1), a ⟨i + 1, nat.succ_pos i⟩

-- The main theorem to prove
theorem sum_of_n_conditions : 
  ∑ n in Finset.filter (λ n, (18 / 17 < (S ⟨2 * n, by sorry⟩ / S ⟨n, by sorry⟩) ∧ (S ⟨2 * n, by sorry⟩ / S ⟨n, by sorry⟩ < 8 / 7)))
                         (Finset.range 10), 
  ↑n = 7 :=
by
  sorry -- The proof is omitted

end sum_of_n_conditions_l674_674327


namespace city_grid_sinks_l674_674539

-- Define the main conditions of the grid city
def cell_side_meter : Int := 500
def max_travel_km : Int := 1

-- Total number of intersections in a 100x100 grid
def total_intersections : Int := (100 + 1) * (100 + 1)

-- Number of sinks that need to be proven
def required_sinks : Int := 1300

-- Lean theorem statement to prove that given the conditions,
-- there are at least 1300 sinks (intersections that act as sinks)
theorem city_grid_sinks :
  ∀ (city_grid : Matrix (Fin 101) (Fin 101) IntersectionType),
  (∀ i j, i < 100 → j < 100 → cell_side_meter ≤ max_travel_km * 1000) →
  ∃ (sinks : Finset (Fin 101 × Fin 101)), 
  (sinks.card ≥ required_sinks) := sorry

end city_grid_sinks_l674_674539


namespace allison_uploads_480_hours_in_june_l674_674246

noncomputable def allison_upload_total_hours : Nat :=
  let before_june_16 := 10 * 15
  let from_june_16_to_23 := 15 * 8
  let from_june_24_to_end := 30 * 7
  before_june_16 + from_june_16_to_23 + from_june_24_to_end

theorem allison_uploads_480_hours_in_june :
  allison_upload_total_hours = 480 := by
  sorry

end allison_uploads_480_hours_in_june_l674_674246


namespace problem_solution_l674_674827

theorem problem_solution
  (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end problem_solution_l674_674827


namespace max_tangent_lines_to_cubic_l674_674790

theorem max_tangent_lines_to_cubic (t : ℝ) :
  let f := λ x : ℝ, x^3 - x in
  ∀ x₀ : ℝ, (∃ (k : ℝ), k = (f x₀ - 0) / (x₀ - t) ∧
    (∀ x : ℝ, x ≠ x₀ → y = k * (x - x₀) + f x₀ → y = (3 * x₀^2 - 1) * x - 2 * x₀^3) ∧
    t * (3 * x₀^2 - 1) = 2 * x₀^3) →
  (∃ (roots : finset ℝ), roots.card ≤ 3 ∧
    (∀ x₀ ∈ roots, has_deriv_at f (3 * x₀^2 - 1) x₀)) :=
by
  sorry

end max_tangent_lines_to_cubic_l674_674790


namespace range_of_g_l674_674787

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x * Real.cos x + (Real.cos x)^2 - 1 / 2

noncomputable def g (x : ℝ) : ℝ := f (x + (5 * Real.pi / 12))

def g_range := Set.Icc (-1 : ℝ) (1/2 : ℝ)

theorem range_of_g :
  Set.range (λ x, g x) ⊆ g_range :=
sorry

end range_of_g_l674_674787


namespace calculate_f_at_pi_div_6_l674_674317

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem calculate_f_at_pi_div_6 (ω φ : ℝ) 
  (h : ∀ x : ℝ, f (π / 3 + x) ω φ = f (-x) ω φ) :
  f (π / 6) ω φ = 2 ∨ f (π / 6) ω φ = -2 :=
sorry

end calculate_f_at_pi_div_6_l674_674317


namespace crayons_problem_l674_674180

noncomputable def crayons_difference (initial_willy initial_lucy initial_jake : ℕ) (willy_loss lucy_loss jake_loss : ℝ) : ℤ :=
  let willy_left := initial_willy - (willy_loss * initial_willy).to_nat
  let lucy_left := initial_lucy - (lucy_loss * initial_lucy).to_nat
  let jake_left := initial_jake - (jake_loss * initial_jake).to_nat
  lucy_left + jake_left - willy_left

theorem crayons_problem : crayons_difference 5092 3971 2435 0.15 0.10 0.05 = 1559 :=
by
  sorry

end crayons_problem_l674_674180


namespace volume_of_tetrahedron_l674_674610

structure Tetrahedron :=
  (A B C D : ℝ)
  (AB AC BC BD AD CD : ℝ)

noncomputable def tetrahedronVolume (T : Tetrahedron) : ℝ :=
  -- define the volume calculation here (we'll leave it as sorry, since the proof is not required)
  sorry

def T : Tetrahedron :=
{
  A := 0, B := 0, C := 0, D := 0,  -- Points are placeholder values since they are not required
  AB := 5,
  AC := 3,
  BC := 4,
  BD := 4,
  AD := 3,
  CD := 12 * Real.sqrt 2 / 5
}

theorem volume_of_tetrahedron : tetrahedronVolume T = 24 / 5 := by
  sorry

end volume_of_tetrahedron_l674_674610


namespace selling_price_per_copy_l674_674210

-- The conditions
def cost_per_program : ℝ := 0.70
def advertisement_revenue : ℝ := 15000
def copies_sold : ℝ := 35000
def desired_profit : ℝ := 8000

-- The assertion to prove
theorem selling_price_per_copy : 
  let P := (advertisement_revenue + desired_profit) / copies_sold
  in P = 0.50 :=
by
  let total_cost := copies_sold * cost_per_program
  let total_revenue_needed := advertisement_revenue + desired_profit
  let P := total_revenue_needed / copies_sold
  have : total_cost = copies_sold * cost_per_program := rfl
  have : total_revenue_needed - total_cost = desired_profit := by linarith
  show P = 0.50, by simp [P]; linarith

-- To define constant values
def cost_to_produce_total : ℝ := copies_sold * cost_per_program
def total_revenue_from_sales (P : ℝ) : ℝ := copies_sold * P
def total_profit (P : ℝ) : ℝ := (total_revenue_from_sales P) + advertisement_revenue - cost_to_produce_total

end selling_price_per_copy_l674_674210


namespace numbers_product_l674_674566

theorem numbers_product (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 8) : x * y = 128 := by
  sorry

end numbers_product_l674_674566


namespace quadrilateral_identity_l674_674843

theorem quadrilateral_identity 
  {A B C D : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (AC : ℝ) (BD : ℝ)
  (angle_A : ℝ) (angle_C : ℝ) 
  (h_angle_sum : angle_A + angle_C = 120)
  : (AC * BD)^2 = (AB * CD)^2 + (BC * AD)^2 + AB * BC * CD * DA := 
by {
  sorry
}

end quadrilateral_identity_l674_674843


namespace quadratic_to_binomial_square_l674_674970

theorem quadratic_to_binomial_square (m : ℝ) : 
  (∃ c : ℝ, (x : ℝ) → x^2 - 12 * x + m = (x + c)^2) ↔ m = 36 := 
sorry

end quadratic_to_binomial_square_l674_674970


namespace min_races_required_to_determine_top_3_horses_l674_674671

def maxHorsesPerRace := 6
def totalHorses := 30
def possibleConditions := "track conditions and layouts change for each race"

noncomputable def minRacesToDetermineTop3 : Nat :=
  7

-- Problem Statement: Prove that given the conditions on track and race layout changes,
-- the minimum number of races needed to confidently determine the top 3 fastest horses is 7.
theorem min_races_required_to_determine_top_3_horses 
  (maxHorsesPerRace : Nat := 6) 
  (totalHorses : Nat := 30)
  (possibleConditions : String := "track conditions and layouts change for each race") :
  minRacesToDetermineTop3 = 7 :=
  sorry

end min_races_required_to_determine_top_3_horses_l674_674671


namespace triangle_base_length_l674_674533

theorem triangle_base_length (h : 3 = (b * 3) / 2) : b = 2 :=
by
  sorry

end triangle_base_length_l674_674533


namespace reciprocal_of_neg_seven_l674_674150

theorem reciprocal_of_neg_seven : (1 : ℚ) / (-7 : ℚ) = -1 / 7 :=
by
  sorry

end reciprocal_of_neg_seven_l674_674150


namespace subtract_500_from_sum_of_calculations_l674_674372

theorem subtract_500_from_sum_of_calculations (x : ℕ) (h : 423 - x = 421) : 
  (421 + 423 * x) - 500 = 767 := 
by
  sorry

end subtract_500_from_sum_of_calculations_l674_674372


namespace expected_length_of_first_group_l674_674931

theorem expected_length_of_first_group (n_ones n_zeros : ℕ) : 
  n_ones = 19 → n_zeros = 49 → 
  let X := ∑ k in finset.range n_ones, (1 / (n_ones + n_zeros)) + ∑ m in finset.range n_zeros, (1 / (n_zeros + n_ones)) in
  (∑ k in finset.range n_ones, (1 / (n_ones + n_zeros)) + ∑ m in finset.range n_zeros, (1 / (n_zeros + n_ones))) = 2.83 :=
begin
  sorry /- Proof not required /-
end

end expected_length_of_first_group_l674_674931


namespace rational_x_sqrt2_eq_zero_l674_674120

theorem rational_x_sqrt2_eq_zero (x : ℚ) (h : ∃ (y : ℚ), x * real.sqrt 2 = y) : x = 0 :=
sorry

end rational_x_sqrt2_eq_zero_l674_674120


namespace point_in_polar_coordinates_l674_674709

theorem point_in_polar_coordinates (x y : ℝ) (h_x : x = 8) (h_y : y = 2 * real.sqrt 3) :
    ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * real.pi ∧ r = 2 * real.sqrt 19 ∧ 
      θ = real.arctan (real.sqrt 3 / 4) :=
by
  use 2 * real.sqrt 19, real.arctan (real.sqrt 3 / 4)
  sorry

end point_in_polar_coordinates_l674_674709


namespace point_on_terminal_side_l674_674780

theorem point_on_terminal_side (x : ℝ) (θ : ℝ) :
  (cos θ = (sqrt 2 / 2) * x) → (x = 1 ∨ x = -1 ∨ x = 0) :=
by
  sorry

end point_on_terminal_side_l674_674780


namespace sum_of_angles_is_110_l674_674997

def inscribed_angle {α : Type} [LinearOrder α] {A B C : α} : Prop :=
  sorry -- Define an inscribed angle property

def central_angle {α : Type} [LinearOrder α] {A B C : α} : Prop :=
  sorry -- Define a central angle property

variables {α : Type} [LinearOrder α] {A B C D : α}

theorem sum_of_angles_is_110 
  (h1 : inscribed_angle A C B 40)
  (h2 : inscribed_angle C A D 30) : 
  central_angle CAB + central_angle ACD = 110 := 
sorry

end sum_of_angles_is_110_l674_674997


namespace farmer_pays_per_row_l674_674637

def ears_per_row : ℕ := 70
def seeds_per_bag : ℕ := 48
def seeds_per_ear : ℕ := 2
def dinner_cost_per_kid : ℕ := 36
def bags_per_kid : ℕ := 140

def total_seeds_used_per_kid := bags_per_kid * seeds_per_bag
def ears_of_corn_planted_per_kid := total_seeds_used_per_kid / seeds_per_ear
def rows_planted_per_kid := ears_of_corn_planted_per_kid / ears_per_row
def total_earnings_per_kid := dinner_cost_per_kid * 2
def pay_per_row := total_earnings_per_kid / rows_planted_per_kid

theorem farmer_pays_per_row : pay_per_row = 1.50 := by
  sorry

end farmer_pays_per_row_l674_674637


namespace locus_of_point_A_l674_674668

noncomputable def ellipse_point (θ : ℝ) : ℝ × ℝ := 
  (2 * Real.cos θ, Real.sin θ)

structure Point (α : Type) := (x : α) (y : α)

def distance (p1 p2 : Point ℝ) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

variables (r : ℝ) (h : 0 < r ∧ r < 1)
variables (M : Point ℝ) 
variables (F1 F2 : Point ℝ) 
variables (A B : Point ℝ)
variables (k1 k2 : ℝ)

theorem locus_of_point_A :
  distance A F1 - distance B F2 = 2 * r → 
  ∃ θ : ℝ, A = ellipse_point θ :=
sorry

end locus_of_point_A_l674_674668


namespace translate_left_to_find_f_l674_674575

-- Define the condition
def condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 2) = 2^(2 * x - 1)

-- Prove the main statement
theorem translate_left_to_find_f (f : ℝ → ℝ) (h : condition f) : 
  ∀ x : ℝ, f(x) = 2^(2 * x - 5) :=
by
  sorry

end translate_left_to_find_f_l674_674575


namespace recurrence_f_l674_674194

open Finset

def valid_permutation (n : ℕ) (perm : Perm (Fin n)) : Prop :=
  ∀ i (hi : i < n - 1), (|perm i - perm (i + 1)| : ℕ) ≠ 1

noncomputable def f (n : ℕ) : ℕ :=
  (univ.filter (valid_permutation n) (perms (Fin n))).card

theorem recurrence_f :
  ∀ (n : ℕ),
  f n = 
         if n = 2 then 0 
         else if n = 3 then 0 
         else if n = 4 then 2 
         else if n = 5 then 14 
         else if n = 6 then 90 
         else sorry :=
by sorry

end recurrence_f_l674_674194


namespace total_segment_length_l674_674009

noncomputable def quadratic_function (n x : ℝ) : ℝ :=
  n * (n + 1) * x^2 - (2 * n + 1) * x + 1

lemma segment_length (n : ℕ) : n ≥ 1 → ℝ := 
  have h: n > 1 := sorry -- This will be the actual proof.
  (1 / n) - (1 / (n + 1))

theorem total_segment_length : 
  (finset.range 10).sum (λ n, if n = 0 then 0 else segment_length (n+1) n.succ_pos) = 10 / 11 :=
by
  sorry

end total_segment_length_l674_674009


namespace find_list_price_l674_674553

theorem find_list_price (P : ℝ) (h1 : 0.873 * P = 61.11) : P = 61.11 / 0.873 :=
by
  sorry

end find_list_price_l674_674553


namespace verify_sub_by_add_verify_sub_by_sub_verify_mul_by_div1_verify_mul_by_div2_verify_mul_by_mul_l674_674959

variable (A B C P M N : ℝ)

-- Verification of Subtraction by Addition
theorem verify_sub_by_add (h : A - B = C) : C + B = A :=
sorry

-- Verification of Subtraction by Subtraction
theorem verify_sub_by_sub (h : A - B = C) : A - C = B :=
sorry

-- Verification of Multiplication by Division (1)
theorem verify_mul_by_div1 (h : M * N = P) : P / N = M :=
sorry

-- Verification of Multiplication by Division (2)
theorem verify_mul_by_div2 (h : M * N = P) : P / M = N :=
sorry

-- Verification of Multiplication by Multiplication
theorem verify_mul_by_mul (h : M * N = P) : M * N = P :=
sorry

end verify_sub_by_add_verify_sub_by_sub_verify_mul_by_div1_verify_mul_by_div2_verify_mul_by_mul_l674_674959


namespace math_problem_l674_674086

variable {R : Type} [LinearOrderedField R]

theorem math_problem
  (a b : R) (ha : 0 < a) (hb : 0 < b)
  (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
by
  sorry

end math_problem_l674_674086


namespace triangle_vertex_coordinates_l674_674140

theorem triangle_vertex_coordinates (
    M1 : ℚ × ℚ,
    M2 : ℚ × ℚ,
    M3 : ℚ × ℚ
) : M1 = ((1/4:ℚ), (13/4:ℚ)) → M2 = ((-1/2:ℚ), (1:ℚ)) → M3 = ((-5/4:ℚ), (5/4:ℚ)) →
∃ A B C : ℚ × ℚ, 
( (A.1 + B.1) / 2 = M3.1 ∧ (A.1 + C.1) / 2 = M2.1 ∧ (B.1 + C.1) / 2 = M1.1 ) ∧
( (A.2 + B.2) / 2 = M3.2 ∧ (A.2 + C.2) / 2 = M2.2 ∧ (B.2 + C.2) / 2 = M1.2 ) ∧
A = ((-2:ℚ), (-1:ℚ))
    ∧ B = ((-1/2:ℚ), (13/4:ℚ))
    ∧ C = ((1:ℚ), (5/4:ℚ)) :=
sorry

end triangle_vertex_coordinates_l674_674140


namespace Elina_donuts_iced_simultaneously_l674_674673

theorem Elina_donuts_iced_simultaneously :
  let Elina_radius := 5
  let Mark_radius := 9
  let Lila_radius := 12
  let Elina_area := 4 * Real.pi * Elina_radius^2
  let Mark_area := 4 * Real.pi * Mark_radius^2
  let Lila_area := 4 * Real.pi * Lila_radius^2
  let lcm_Elina_Mark_Lila_area := Real.lcm (Real.lcm Elina_area Mark_area) Lila_area
  let number_of_Elina_donuts := lcm_Elina_Mark_Lila_area / Elina_area
  number_of_Elina_donuts = 3240 :=
by
  let Elina_radius := 5
  let Mark_radius := 9
  let Lila_radius := 12
  
  let Elina_area := 4 * Real.pi * Elina_radius^2
  let Mark_area := 4 * Real.pi * Mark_radius^2
  let Lila_area := 4 * Real.pi * Lila_radius^2
  
  let lcm_Elina_Mark_Lila_area := Real.lcm (Real.lcm Elina_area Mark_area) Lila_area
  let number_of_Elina_donuts := lcm_Elina_Mark_Lila_area / Elina_area
  
  have : number_of_Elina_donuts = 3240 := sorry
  exact this

end Elina_donuts_iced_simultaneously_l674_674673


namespace sum_of_special_two_digit_integers_l674_674967

-- Define a function to identify two-digit integers satisfying the conditions
def is_desired_number (num : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
  num = 10 * a + b ∧ 
  (a + b ∣ 10 * a + b) ∧ 
  (a - b ∣ 10 * a + b) ∧ 
  (a * b ∣ 10 * a + b)

-- Define the sum of all two-digit integers meeting the conditions
def sum_of_desired_numbers : ℕ :=
  Finset.sum (Finset.filter is_desired_number (Finset.range 100)) id

-- The theorem to prove is that the sum is 72
theorem sum_of_special_two_digit_integers : sum_of_desired_numbers = 72 := 
  sorry

end sum_of_special_two_digit_integers_l674_674967


namespace vector_subtraction_l674_674370

-- Define the vectors a and b
def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-3, -4)

-- Statement we want to prove: 2a - b = (-1, 6)
theorem vector_subtraction : 2 • a - b = (-1, 6) := by
  sorry

end vector_subtraction_l674_674370


namespace A_sub_B_multiple_p_l674_674860

variable (X : Type*) [Fintype X] (p : ℕ) [Fact (Nat.Prime p)] [NeZero (Fintype.card X)]

def p_family (X : Type*) [Fintype X] (p : ℕ) : Set (Finset (Finset X)) :=
  {A : Finset (Finset X) | ∀ s t ∈ A, s ≠ t → s ∩ t = ∅ ∧ ∀ s ∈ A, s.card = p }

noncomputable def A (X : Type*) [Fintype X] (p : ℕ) [Fact (Nat.Prime p)] [NeZero (Fintype.card X)] : ℕ :=
  ((p_family X p).filter (λ A, A.card % 2 = 0)).toFinset.card

noncomputable def B (X : Type*) [Fintype X] (p : ℕ) [Fact (Nat.Prime p)] [NeZero (Fintype.card X)] : ℕ :=
  ((p_family X p).filter (λ A, A.card % 2 = 1)).toFinset.card

theorem A_sub_B_multiple_p (X : Type*) [Fintype X] (p : ℕ) [Fact (Nat.Prime p)] [NeZero (Fintype.card X)] :
  (A X p - B X p) % p = 0 := sorry

end A_sub_B_multiple_p_l674_674860


namespace maximum_value_of_T_4_l674_674768

noncomputable def maximum_value_of_T (a : Fin 5 → ℝ) : ℝ :=
  ∑ i, |a i - a ((i.1 + 1) % 5)⟩

theorem maximum_value_of_T_4 {a : Fin 5 → ℝ} (h : ∑ i, (a i)^2 = 1) : 
  maximum_value_of_T a ≤ 4 :=
sorry

end maximum_value_of_T_4_l674_674768


namespace inscribed_circle_radius_in_quadrilateral_pyramid_l674_674431

theorem inscribed_circle_radius_in_quadrilateral_pyramid
  (a : ℝ) (α : ℝ)
  (h_pos : 0 < a) (h_α : 0 < α ∧ α < π / 2) :
  ∃ r : ℝ, r = a * Real.sqrt 2 / (1 + 2 * Real.cos α + Real.sqrt (4 * Real.cos α ^ 2 + 1)) :=
by
  sorry

end inscribed_circle_radius_in_quadrilateral_pyramid_l674_674431


namespace hours_buses_leave_each_day_l674_674253

theorem hours_buses_leave_each_day
  (num_buses : ℕ)
  (num_days : ℕ)
  (buses_per_half_hour : ℕ)
  (h1 : num_buses = 120)
  (h2 : num_days = 5)
  (h3 : buses_per_half_hour = 2) :
  (num_buses / num_days) / buses_per_half_hour = 12 :=
by
  sorry

end hours_buses_leave_each_day_l674_674253


namespace unique_sums_count_l674_674020

def set := {2, 6, 10, 14, 18, 22, 26}

theorem unique_sums_count :
  let sums := {a + b + c | a b c : ℕ, a ∈ set ∧ b ∈ set ∧ c ∈ set ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c} in
  sums.card = 25 :=
by sorry

end unique_sums_count_l674_674020


namespace value_of_a_l674_674360

def f (x : ℝ) : ℝ := 
if x < 1 then x^2 + 1 else 2 * x

theorem value_of_a (a : ℝ) (h : f(a) = 10) : a = -3 ∨ a = 5 := 
sorry

end value_of_a_l674_674360


namespace width_of_alley_l674_674990

-- Define the conditions
def ladder_length : ℝ := 10
def height_wall_1 : ℝ := 6
def angle_wall_1 : ℝ := 60
def height_wall_2 : ℝ := 8
def angle_wall_2 : ℝ := 45

-- Define the problem statement as a theorem
theorem width_of_alley (l : ℝ) (h1 : ℝ) (a1 : ℝ) (h2 : ℝ) (a2 : ℝ) :
  l = ladder_length → 
  h1 = height_wall_1 → 
  a1 = angle_wall_1 →
  h2 = height_wall_2 → 
  a2 = angle_wall_2 → 
  (l * real.cos (a1 * real.pi / 180) + l * real.cos (a2 * real.pi / 180) = 12.07) :=
by
  intro hl hh1 ha1 hh2 ha2
  rw [hl, hh1, ha1, hh2, ha2]
  sorry

end width_of_alley_l674_674990


namespace infinitely_many_primes_in_sequence_l674_674238

def seq : ℕ → ℕ
| 0       := 1
| (n + 1) := (seq n)^2 + 1

theorem infinitely_many_primes_in_sequence :
  ∃ᶠ p : ℕ in Filter.at_top, Prime p ∧ ∃ n : ℕ, p ∣ seq n := 
sorry

end infinitely_many_primes_in_sequence_l674_674238


namespace cone_sphere_ratio_l674_674235

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cone (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (2 * r)^2 * h

theorem cone_sphere_ratio (r h : ℝ) (V_cone V_sphere : ℝ) (h_sphere : V_sphere = volume_of_sphere r)
  (h_cone : V_cone = volume_of_cone r h) (h_relation : V_cone = (1/3) * V_sphere) :
  (h / (2 * r) = 1 / 6) :=
by
  sorry

end cone_sphere_ratio_l674_674235


namespace b_plus_c_max_not_min_l674_674340

/--
Given f(x) = x^3 + bx^2 + cx + d is a decreasing function over the interval [-1, 2].
Show that b + c has a maximum value but not a minimum value.
-/
theorem b_plus_c_max_not_min (b c d : ℝ) (h : ∀ x ∈ set.Icc (-1 : ℝ) (2 : ℝ), deriv (λ x : ℝ, x^3 + b * x^2 + c * x + d) x ≤ 0) :
    ∃ max_val : ℝ, ∃ c_min : ℝ, c_min < max_val ∧ ∀ b c, b^2 - 3 * c < 0 → b + c ≤ max_val ∧ (¬ ∀ c, b + c ≥ c_min) :=
sorry

end b_plus_c_max_not_min_l674_674340


namespace find_city_mpg_l674_674593

noncomputable def car_mpg_city (H C : ℕ) (T : ℕ) : Prop :=
  (448 = H * T) ∧
  (336 = C * T) ∧
  (C = H - 6)

theorem find_city_mpg :
  ∃ H C : ℕ, ∀ T : ℕ, car_mpg_city H C T → C = 18 :=
begin
  sorry
end

end find_city_mpg_l674_674593


namespace probability_r20_to_r30_after_one_operation_l674_674320

noncomputable theory
open Classical

def sequence := list ℝ

def operation (seq : sequence) : sequence :=
(list.scanl (fun (acc : list ℝ) (x : ℝ) => if x < acc.head' then x :: acc.tail else x :: acc) seq).head'.reverse

def second_largest (l : list ℝ) : ℝ :=
(list.sort (≤) l).nth_le (l.length - 2) sorry

theorem probability_r20_to_r30_after_one_operation (seq : sequence) (h : seq.length = 40) (h_distinct : seq.nodup)
    : let r20 := seq.nth_le 19 sorry,
          r31 := seq.nth_le 30 sorry,
          sorted_seq := operation seq,
          p := sorted_seq.nth_le 29 sorry,
          prob := (1 : ℚ) / 31 * (1 : ℚ) / 30
      in p + q = 931 :=
sorry

end probability_r20_to_r30_after_one_operation_l674_674320


namespace angle_between_vectors_l674_674405

variables {a b : ℝ}

-- Condition 1: |a| = sqrt(2)
def magnitude_a : ℝ := real.sqrt 2

-- Condition 2: |b| = 2
def magnitude_b : ℝ := 2

-- Condition 3: (a - b) ⊥ a
def perp_condition (a b : ℝ) : Prop :=
  (a - b) * a = 0

-- The proof statement
theorem angle_between_vectors (a b : ℝ) 
  (h₁ : |a| = magnitude_a) 
  (h₂ : |b| = magnitude_b) 
  (h₃ : perp_condition a b) : 
  real.angle a b = real.pi / 4 := 
sorry

end angle_between_vectors_l674_674405


namespace distance_in_scientific_notation_l674_674093

-- Definition for the number to be expressed in scientific notation
def distance : ℝ := 55000000

-- Expressing the number in scientific notation
def scientific_notation : ℝ := 5.5 * (10 ^ 7)

-- Theorem statement asserting the equality
theorem distance_in_scientific_notation : distance = scientific_notation :=
  by
  -- Proof not required here, so we leave it as sorry
  sorry

end distance_in_scientific_notation_l674_674093


namespace solve_triangle_l674_674525

noncomputable def triangle_side_lengths (a b c : ℝ) : Prop :=
  a = 10 ∧ b = 9 ∧ c = 17

theorem solve_triangle (a b c : ℝ) :
  (a ^ 2 - b ^ 2 = 19) ∧ 
  (126 + 52 / 60 + 12 / 3600 = 126.87) ∧ -- Converting the angle into degrees for simplicity
  (21.25 = 21.25)  -- Diameter given directly
  → triangle_side_lengths a b c :=
sorry

end solve_triangle_l674_674525


namespace woman_weaving_second_day_output_l674_674844

def geometric_sequence_second_term_correct (a₁ : ℝ) (q : ℝ) (S : ℝ) : Prop :=
  S = (a₁ * (q ^ 5 - 1)) / (q - 1) → q = 2 → S = 5 → a₁ * 2 = a₁ * 2 → (a₁ * q)

theorem woman_weaving_second_day_output
  (a₁ : ℝ) (H : geometric_sequence_second_term_correct a₁ 2 5) :
  a₁ * 2 = 10 / 31 := 
by
  sorry

end woman_weaving_second_day_output_l674_674844


namespace tangent_lines_through_point_l674_674793

noncomputable def f (x : ℝ) : ℝ :=
  (x - 1) * (x^2 + 1) + 1

def f' (x : ℝ) : ℝ :=
  deriv f x

def tangent_line_eqn_1 (x y : ℝ) : Prop :=
  2 * x - y - 1 = 0

def tangent_line_eqn_2 (x y : ℝ) : Prop :=
  y = x

theorem tangent_lines_through_point :
  ∃ (line_eqn : ℝ → ℝ → Prop), line_eqn = tangent_line_eqn_1 ∨ line_eqn = tangent_line_eqn_2 := by
  sorry

end tangent_lines_through_point_l674_674793


namespace A_inter_B_empty_iff_A_inter_B_eq_A_iff_l674_674010

variable {x a : ℝ}

noncomputable def A : set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
noncomputable def B : set ℝ := {x | x ≤ 0 ∨ x ≥ 4}

theorem A_inter_B_empty_iff : (A ∩ B = ∅) ↔ 0 < a ∧ a < 2 := 
by
  sorry

theorem A_inter_B_eq_A_iff : (A ∩ B = A) ↔ (a ≤ -2 ∨ a ≥ 4) := 
by
  sorry

end A_inter_B_empty_iff_A_inter_B_eq_A_iff_l674_674010


namespace part1_part2_part3_l674_674799

-- Definition of the functions f and g
def f (m x : ℝ) : ℝ := (m / x) + x * log x
def g (x : ℝ) : ℝ := log x - 2

-- 1. Prove that for m = 1, f(x) is increasing on (1, +∞)
theorem part1 (x : ℝ) (hx : 1 < x) : 
  ∀ (m = 1), ∃ (I : Set ℝ), I = { x | x ∈ I ∧ x > 1 } → 
    increasingOn I (f 1) :=
sorry

-- 2. Given minimum value of y = h(h(x)) is 3√2/2, prove that m = 1 when h(x) = f(x) − xg(x) − √2
def h (m x : ℝ) : ℝ := f m x - x * g x - sqrt 2

theorem part2 (x : ℝ) (hx : 0 < x) (y : ℝ) (hy : y = h m (h m x)) : 
  min y = (3 * sqrt 2) / 2 → m = 1 :=
sorry

-- 3. For m > 0, with f and g having domain [1, e], prove that the range of m is [1/2, e]
theorem part3 (m : ℝ) (hm : 0 < m) : 
  ∀ (x ∈ Set.Icc (1:ℝ) real.exp), 
    ∃ I : Set ℝ, 
    I = { m | 1 / 2 ≤ m ∧ m ≤ real.exp } → 
    ∀ (A : ℝ) (B : ℝ) (HA : A ∈ image (λ x, (x, f m x)) Set.univ) (HB : B ∈ image (λ x, (x, g x)) Set.univ), 
    (0, A) ⬝ (0, B) = 0 :=
sorry

end part1_part2_part3_l674_674799


namespace simplify_and_evaluate_l674_674896

theorem simplify_and_evaluate (a : ℕ) (h : a = 2) :
  (3 * a - 3) / a / ((a * a - 2 * a + 1) / (a * a)) - a / (a - 1) = 4 :=
by
  rw [h],
  sorry


end simplify_and_evaluate_l674_674896


namespace amount_each_person_needs_to_raise_l674_674590

theorem amount_each_person_needs_to_raise (Total_goal Already_collected Number_of_people : ℝ) 
(h1 : Total_goal = 2400) (h2 : Already_collected = 300) (h3 : Number_of_people = 8) : 
    (Total_goal - Already_collected) / Number_of_people = 262.5 := 
by
  sorry

end amount_each_person_needs_to_raise_l674_674590


namespace number_of_tiles_in_each_row_l674_674124

theorem number_of_tiles_in_each_row:
  ∀ (area_sq_ft: ℕ) (tile_side_in_inch: ℕ), 
  area_sq_ft = 225 ∧ tile_side_in_inch = 6 
  → let side_length_in_ft := Real.sqrt area_sq_ft
      let side_length_in_inch := side_length_in_ft * 12
      let num_tiles := side_length_in_inch / tile_side_in_inch
    in num_tiles = 30 :=
by
  sorry

end number_of_tiles_in_each_row_l674_674124


namespace min_value_when_a_is_2_min_value_when_0_lt_a_lt_1_l674_674797

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x + a / (x + 1)

-- First proof: When a = 2, the minimum value of f(x) on [0, ∞) is 2√2 - 1
theorem min_value_when_a_is_2 :
  (∀ x : ℝ, 0 ≤ x → f x 2 ≥ 2 * Real.sqrt 2 - 1) ∧ 
  (∃ x : ℝ, 0 ≤ x ∧ f x 2 = 2 * Real.sqrt 2 - 1) :=
sorry

-- Second proof: When 0 < a < 1, the minimum value of f(x) on [0, ∞) is a
theorem min_value_when_0_lt_a_lt_1 (a : ℝ) (h : 0 < a ∧ a < 1) :
  (∀ x : ℝ, 0 ≤ x → f x a ≥ a) ∧ 
  (∃ x : ℝ, 0 ≤ x ∧ f x a = a) :=
sorry

end min_value_when_a_is_2_min_value_when_0_lt_a_lt_1_l674_674797


namespace history_but_not_statistics_l674_674604

theorem history_but_not_statistics (H S H_union_S : ℕ)
  (hH : H = 36) 
  (hS : S = 32) 
  (hH_union_S : H_union_S = 59) :
  H - (H + S - H_union_S) = 27 :=
by
  rw [hH, hS, hH_union_S]
  sorry

end history_but_not_statistics_l674_674604


namespace union_sets_eq_real_l674_674367

def A : Set ℝ := {x | x ≥ 0}
def B : Set ℝ := {x | x < 1}

theorem union_sets_eq_real : A ∪ B = Set.univ :=
by
  sorry

end union_sets_eq_real_l674_674367


namespace arithmetic_series_sum_l674_674738

theorem arithmetic_series_sum :
  let a1 := 5
  let an := 105
  let d := 1
  let n := (an - a1) / d + 1
  (n * (a1 + an) / 2) = 5555 := by
  sorry

end arithmetic_series_sum_l674_674738


namespace english_score_is_96_l674_674069

variable (Science_score : ℕ) (Social_studies_score : ℕ) (English_score : ℕ)

/-- Jimin's social studies score is 6 points higher than his science score -/
def social_studies_score_condition := Social_studies_score = Science_score + 6

/-- The science score is 87 -/
def science_score_condition := Science_score = 87

/-- The average score for science, social studies, and English is 92 -/
def average_score_condition := (Science_score + Social_studies_score + English_score) / 3 = 92

theorem english_score_is_96
  (h1 : social_studies_score_condition Science_score Social_studies_score)
  (h2 : science_score_condition Science_score)
  (h3 : average_score_condition Science_score Social_studies_score English_score) :
  English_score = 96 :=
  by
    sorry

end english_score_is_96_l674_674069


namespace min_value_of_E_l674_674580

noncomputable def E : ℝ := sorry

theorem min_value_of_E :
  (∀ x : ℝ, |E| + |x + 7| + |x - 5| ≥ 12) →
  (∃ x : ℝ, |x + 7| + |x - 5| = 12 → |E| = 0) :=
sorry

end min_value_of_E_l674_674580


namespace math_problem_l674_674689

theorem math_problem :
  ( ∏ i in [3, 4, 5, 6, 7], (i^3 - 1) / (i^3 + 1) ) = 57 / 84 := sorry

end math_problem_l674_674689


namespace ones_digit_of_prime_in_arithmetic_sequence_l674_674159

theorem ones_digit_of_prime_in_arithmetic_sequence (p q r : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) 
  (arithmetic_sequence : q = p + 4 ∧ r = q + 4)
  (h : p > 5) : 
    (p % 10 = 3 ∨ p % 10 = 9) :=
sorry

end ones_digit_of_prime_in_arithmetic_sequence_l674_674159


namespace sales_last_year_l674_674654

theorem sales_last_year (x : ℝ) (h1 : 416 = (1 + 0.30) * x) : x = 320 :=
by
  sorry

end sales_last_year_l674_674654


namespace factorize_polynomial_value_of_x_cubed_l674_674620

-- Problem 1: Factorization
theorem factorize_polynomial (x : ℝ) : 42 * x^2 - 33 * x + 6 = 3 * (2 * x - 1) * (7 * x - 2) :=
sorry

-- Problem 2: Given condition and proof of x^3 + 1/x^3
theorem value_of_x_cubed (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^3 + 1 / x^3 = 18 :=
sorry

end factorize_polynomial_value_of_x_cubed_l674_674620


namespace celine_change_l674_674648

theorem celine_change
  (price_laptop : ℕ)
  (price_smartphone : ℕ)
  (num_laptops : ℕ)
  (num_smartphones : ℕ)
  (total_money : ℕ)
  (h1 : price_laptop = 600)
  (h2 : price_smartphone = 400)
  (h3 : num_laptops = 2)
  (h4 : num_smartphones = 4)
  (h5 : total_money = 3000) :
  total_money - (num_laptops * price_laptop + num_smartphones * price_smartphone) = 200 :=
by
  sorry

end celine_change_l674_674648


namespace cubics_product_l674_674693

theorem cubics_product :
  (∏ n in [3, 4, 5, 6, 7], (n^3 - 1) / (n^3 + 1)) = (57 / 168) := by
  sorry

end cubics_product_l674_674693


namespace set_of_positive_reals_l674_674785

theorem set_of_positive_reals (S : Set ℝ) (h1 : ∀ x, x ∈ S → 0 < x)
  (h2 : ∀ a b, a ∈ S → b ∈ S → a + b ∈ S)
  (h3 : ∀ (a b : ℝ), 0 < a → a ≤ b → ∃ c d, a ≤ c ∧ c ≤ d ∧ d ≤ b ∧ ∀ x, c ≤ x ∧ x ≤ d → x ∈ S) :
  S = {x : ℝ | 0 < x} :=
sorry

end set_of_positive_reals_l674_674785


namespace equal_student_distribution_l674_674153

theorem equal_student_distribution
  (students_bus1_initial : ℕ)
  (students_bus2_initial : ℕ)
  (students_to_move : ℕ)
  (students_bus1_final : ℕ)
  (students_bus2_final : ℕ)
  (total_students : ℕ) :
  students_bus1_initial = 57 →
  students_bus2_initial = 31 →
  total_students = students_bus1_initial + students_bus2_initial →
  students_to_move = 13 →
  students_bus1_final = students_bus1_initial - students_to_move →
  students_bus2_final = students_bus2_initial + students_to_move →
  students_bus1_final = 44 ∧ students_bus2_final = 44 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end equal_student_distribution_l674_674153


namespace sum_of_solutions_eq_6_l674_674277

theorem sum_of_solutions_eq_6 : 
  (∑ x in {x : ℝ | (x ≠ 1 ∧ x ≠ -1 ∧ (-12 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1)))}.to_finset, x) = 6 :=
by
  sorry

end sum_of_solutions_eq_6_l674_674277


namespace exists_triangle_with_angle_bisectors_exists_triangle_with_altitudes_exists_triangle_with_special_points_l674_674270

open Triangle

section

/-- Prove that triangle ABC exists such that the points P, Q, R are on its circumcircle and intersect the angle bisectors of ABC. -/
theorem exists_triangle_with_angle_bisectors : 
  ∃ (A B C P Q R : Point), 
    is_vertex_of_circumcircle ABC P ∧ 
    is_vertex_of_circumcircle ABC Q ∧ 
    is_vertex_of_circumcircle ABC R ∧
    is_angle_bisector P A B C ∧ 
    is_angle_bisector Q B C A ∧ 
    is_angle_bisector R C A B := 
sorry

/-- Prove that triangle ABC exists such that the points P, Q, R are on its circumcircle and intersect the altitudes of ABC. -/
theorem exists_triangle_with_altitudes : 
  ∃ (A B C P Q R : Point), 
    is_vertex_of_circumcircle ABC P ∧ 
    is_vertex_of_circumcircle ABC Q ∧ 
    is_vertex_of_circumcircle ABC R ∧
    is_altitude P A B C ∧ 
    is_altitude Q B C A ∧ 
    is_altitude R C A B := 
sorry

/-- Prove that triangle ABC exists such that the points P, Q, R are on its circumcircle where the 
    altitude, angle bisector, and median from one vertex intersect. -/
theorem exists_triangle_with_special_points : 
  ∃ (A B C P Q R : Point), 
    is_vertex_of_circumcircle ABC P ∧ 
    is_vertex_of_circumcircle ABC Q ∧ 
    is_vertex_of_circumcircle ABC R ∧
    is_altitude P A B C ∧ 
    is_angle_bisector Q B C A ∧ 
    is_median R C A B := 
sorry

end

end exists_triangle_with_angle_bisectors_exists_triangle_with_altitudes_exists_triangle_with_special_points_l674_674270


namespace problem_solution_l674_674283

noncomputable def floorSum : ℕ :=
  ∑ k in (Finset.range (1415)), (⌊(1 + real.sqrt (2000000 / 4^k)) / 2⌋ : ℕ)

theorem problem_solution : floorSum = 1414 :=
by
  sorry

end problem_solution_l674_674283


namespace shortest_distance_opposite_edges_l674_674031

-- Definitions
def Tetrahedron := Type u -- Define Tetrahedron as a Type

-- Altitude in a Tetrahedron
def isAltitude (T : Tetrahedron) (h : ℝ) : Prop := h ≥ 1

-- Distance between edges of Tetrahedron is more than 2
def shortestDistanceGreater (dist : ℝ) : Prop := dist > 2

-- Main statement 
theorem shortest_distance_opposite_edges (T : Tetrahedron)
  (altitudes_at_least_one : ∀ h, isAltitude T h) :
  ∀ d, shortestDistanceGreater d :=
by
  sorry

end shortest_distance_opposite_edges_l674_674031


namespace jill_second_bus_ride_time_l674_674947

theorem jill_second_bus_ride_time :
  let wait_time := 12
  let ride_time := 30
  let combined_time := wait_time + ride_time
  let second_bus_ride_time := combined_time / 2
  second_bus_ride_time = 21 :=
by
  -- Introduce the definitions
  let wait_time := 12
  let ride_time := 30
  let combined_time := wait_time + ride_time
  let second_bus_ride_time := combined_time / 2
  -- State and confirm the goal
  have : second_bus_ride_time = 21 := by rfl
  exact this
  sorry

end jill_second_bus_ride_time_l674_674947


namespace correct_proposition_l674_674349

-- Definitions from conditions
def prop1 (m : ℝ) : Prop :=
  m ∈ set.Ioo (-1 : ℝ) 2 → 
  (∀ (x y : ℝ), (x^2 / (m + 1)) - (y^2 / (m - 2)) = 1 → false)

def prop2 : Prop :=
  ∀ (P : ℝ × ℝ), 
    abs ((P.1 + 4)^2 + P.2^2) - abs ((P.1 - 4)^2 + P.2^2) = 8 → 
    false

def prop3 (p q : Prop) : Prop :=
  ¬(p ∧ q) → ¬p ∧ ¬q

def prop4 (a x : ℝ) : Prop :=
  ((x < -3) ∨ (x > 1)) → x > a → a ≥ 1

-- Resulting proposition that needs proof
theorem correct_proposition : 
  ∃ a, ∀ (m : ℝ),
    (¬(prop1 m) ∧  
    ¬prop2 ∧ 
    ¬(∀ p q, prop3 p q)) ∧ 
    (∃ x, prop4 a x) := 
sorry

end correct_proposition_l674_674349


namespace single_cube_edge_length_l674_674369

noncomputable def cube_edge_length (V : ℝ) : ℝ := real.cbrt V

theorem single_cube_edge_length (e : ℝ) (V : ℝ) (n : ℕ) (two_cubes_volume : V = n * e ^ 3) 
  (n_eq_two : n = 2) (e_eq_one : e = 1) : cube_edge_length V = real.cbrt 2 :=
by
  rw [n_eq_two, e_eq_one] at two_cubes_volume
  simp only [one_pow, mul_one] at two_cubes_volume
  exact two_cubes_volume
  sorry

end single_cube_edge_length_l674_674369


namespace choose_company_A_for_time_choose_company_B_for_expenses_l674_674092

-- Given conditions
def working_together_time := 6
def working_together_cost := 52000
def company_A_alone_time := 4
def company_A_and_B_time := 13 -- Company A (4 weeks) + Company B (9 weeks)
def company_A_and_B_cost := 48000

-- Statement for saving time:
theorem choose_company_A_for_time:
  company_A_and_B_time > working_together_time → 
  company_A < company_B 
:= sorry

-- Statement for saving expenses:
theorem choose_company_B_for_expenses:
  working_together_cost > company_A_and_B_cost →
  company_B < company_A 
:= sorry

end choose_company_A_for_time_choose_company_B_for_expenses_l674_674092


namespace trigonometric_identity_l674_674591

theorem trigonometric_identity
  (α : ℝ)
  (h_tg : ∀ x, tg x = sin x / cos x)
  (h_ctg : ∀ x, ctg x = cos x / sin x)
  (h_sin_cos : ∀ x, sin x ^ 2 + cos x ^ 2 = 1)
  (h_double_angle : ∀ x, sin (2 * x) = 2 * sin x * cos x)
  (h_triple_angle : ∀ x, sin (6 * x) = 2 * sin (3 * x) * cos (3 * x))
  (h_sum_to_product : ∀ x, sin (6 * x) + sin (2 * x) = 2 * sin (4 * x) * cos (2 * x))
  (h_sin_4x : ∀ x, sin (4 * x) = 2 * sin (2 * x) * cos (2 * x)) :
  tg α + ctg α + tg (3 * α) + ctg (3 * α) = 8 * (cos (2 * α)) ^ 2 / sin (6 * α) :=
  sorry

end trigonometric_identity_l674_674591


namespace builder_total_amount_l674_674994

noncomputable def drill_bits_count := 5
noncomputable def hammer_count := 3
noncomputable def toolbox_count := 1
noncomputable def nail_count := 50

noncomputable def drill_bit_cost := 6.0
noncomputable def hammer_cost := 8.0
noncomputable def toolbox_cost := 25.0
noncomputable def nail_cost := 0.10

noncomputable def drill_bit_tax_rate := 0.10
noncomputable def toolbox_tax_rate := 0.15
noncomputable def hammer_discount_rate := 0.05
noncomputable def overall_discount_rate := 0.05
noncomputable def cost_threshold_for_discount := 60.0

noncomputable def total_cost_before_taxes_and_discounts :=
  (drill_bits_count * drill_bit_cost) + 
  (hammer_count * hammer_cost) + 
  (toolbox_count * toolbox_cost) + 
  (nail_count/2 * nail_cost)

noncomputable def total_tax :=
  (drill_bits_count * drill_bit_cost * drill_bit_tax_rate) + 
  (toolbox_count * toolbox_cost * toolbox_tax_rate)

noncomputable def total_discount :=
  (hammer_count * hammer_cost * hammer_discount_rate)

noncomputable def total_cost_before_overall_discount :=
  total_cost_before_taxes_and_discounts + total_tax - total_discount

noncomputable def overall_discount :=
  if total_cost_before_overall_discount > cost_threshold_for_discount then
    total_cost_before_overall_discount * overall_discount_rate
  else
    0

noncomputable def final_total_amount :=
  total_cost_before_overall_discount - overall_discount

theorem builder_total_amount : final_total_amount = 82.70 :=
by
  sorry

end builder_total_amount_l674_674994


namespace calculate_value_l674_674684

theorem calculate_value :
  ( (3^3 - 1) / (3^3 + 1) ) * ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * ( (7^3 - 1) / (7^3 + 1) )
  = 57 / 84 := by
  sorry

end calculate_value_l674_674684


namespace sine_shift_equivalence_l674_674162

theorem sine_shift_equivalence :
  ∀ x : ℝ,
    2 * sin (2 * (x - (π / 3))) = 2 * sin (2 * (x + (π / 3))) - 2π / 3 →
    (∃ C, 2 * sin (2 * (x + C)) = 2 * sin (2 * x + 2 * π / 3) := by
      intro x
      sorry

end sine_shift_equivalence_l674_674162


namespace probability_two_teachers_in_A_l674_674946

/-- 
Statement of the problem:
Three teachers are randomly assigned to support teaching in two places, A and B,
with each teacher being assigned to only one of the two locations. 

The probability that exactly two teachers are assigned to place A
is 3/8.
-/
theorem probability_two_teachers_in_A : 
  let total_outcomes := (2 : ℝ)^3 in
  let favorable_outcomes := (3 : ℝ) in
  favorable_outcomes / total_outcomes = 3 / 8 :=
by 
  let total_outcomes : ℝ := 8
  let favorable_outcomes : ℝ := 3
  sorry

end probability_two_teachers_in_A_l674_674946


namespace find_a_b_monotonic_decreasing_find_k_range_l674_674755

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := (b - 2^x) / (2^(x + 1) + a)

-- 1. Prove that a = 2 and b = 1 given f(x) is an odd function on ℝ
theorem find_a_b (h_odd : ∀ x, f a b (-x) = -f a b x) : a = 2 ∧ b = 1 :=
sorry

-- 2. Prove that f(x) is monotonically decreasing on ℝ
theorem monotonic_decreasing (a := 2) (b := 1) : ∀ x₁ x₂, x₁ < x₂ → f a b x₁ > f a b x₂ :=
sorry

-- 3. Prove the range of values for k given f(k ⋅ 3^x) + f(3^x - 9^x + 2) > 0 ∀ x ≥ 1
theorem find_k_range (a := 2) (b := 1) (h_ineq : ∀ x ≥ 1, f a b (k * 3^x) + f a b (3^x - 9^x + 2) > 0) : k < 4 / 3 :=
sorry

end find_a_b_monotonic_decreasing_find_k_range_l674_674755


namespace triangle_area_ratio_l674_674063

-- Let us consider the triangle and points with given ratios
variable (X Y Z G H I : Type) 
variables [linear_ordered_field R]

-- Assume point relations
variables (is_point : X → Y → Z → Prop)
variables (XM YN ZO : Type)

-- Given conditions
variable (ratioYM_MZ_eq_2_3 : ∀ (M Y Z : Type), YM ∈ (2:3))
variable (ratioZN_NX_eq_2_3 : ∀ (N Z X : Type), YN ∈ (2:3))
variable (ratioXO_OY_eq_2_3 : ∀ (O X Y : Type), ZO ∈ (2:3))

-- Prove the required area ratio
theorem triangle_area_ratio {X Y Z G H I : Type} :
  (GHI : Type) / (XYZ : Type) = 36 / 125 :=
sorry

end triangle_area_ratio_l674_674063


namespace find_fraction_l674_674201

noncomputable def calculate_fraction (x : ℚ) : Prop :=
  22 = x * 25 + 2

theorem find_fraction : ∃ x : ℚ, calculate_fraction x ∧ x = 4 / 5 :=
by
  use (4 / 5)
  split
  · unfold calculate_fraction
    norm_num
  · norm_num

end find_fraction_l674_674201


namespace difference_sum_average_consecutive_even_l674_674154

theorem difference_sum_average_consecutive_even (n : ℤ)
  (h1 : even n) (h2 : even (n+2)) (h3 : even (n+4)) (h4 : n + 4 = 24) :
  (n + (n + 2) + (n + 4)) - ((n + (n + 2) + (n + 4)) / 3) = 44 :=
by
  sorry

end difference_sum_average_consecutive_even_l674_674154


namespace calculate_value_l674_674682

theorem calculate_value :
  ( (3^3 - 1) / (3^3 + 1) ) * ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * ( (7^3 - 1) / (7^3 + 1) )
  = 57 / 84 := by
  sorry

end calculate_value_l674_674682


namespace distance_M_to_midpoint_KL_equals_R_sqrt2_div2_l674_674326

-- Define the structure of a right triangle and its elements
structure RightTriangle :=
  (A B C : ℝ × ℝ)
  (hypotenuse : line_segment A C)
  (leg1 : line_segment A B)
  (leg2 : line_segment B C)
  (right_angle : line_segment.angle (leg1) (leg2) = π / 2)

-- Define the incircle centers and properties
structure IncircleCenters :=
  (K L : ℝ × ℝ)
  (center_ABM : incenter (RightTriangle.leg1) (RightTriangle.hypotenuse))
  (center_CBM : incenter (RightTriangle.leg2) (RightTriangle.hypotenuse))

-- Define the distance formula
def dist (p1 p2 : ℝ × ℝ) := (sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

-- Lean imports noncomputable if we deal with non-constructive elements
noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def circ_radius (B K L : ℝ × ℝ) : ℝ := sorry  -- placeholder for the radius definition

noncomputable def distance_from_M_to_midpoint_KL (M K L : ℝ × ℝ) :=
  dist M (midpoint K L)

noncomputable def check_distance_from_M_to_midpoint_KL (M K L : ℝ × ℝ) (R : ℝ) : Prop :=
  distance_from_M_to_midpoint_KL M K L = R * sqrt 2 / 2

theorem distance_M_to_midpoint_KL_equals_R_sqrt2_div2
  (rt : RightTriangle)
  (inc : IncircleCenters)
  (M : ℝ × ℝ)
  (R : ℝ)
  (H_radius : circ_radius rt.B inc.K inc.L = R)
  : check_distance_from_M_to_midpoint_KL M inc.K inc.L R :=
  sorry

end distance_M_to_midpoint_KL_equals_R_sqrt2_div2_l674_674326


namespace expansion_coefficients_eq_l674_674442

theorem expansion_coefficients_eq (a b : ℕ) (h_coprime : Nat.coprime a b) (h_coeff_eq : ∀ γ α β : ℕ, 
  (α = 2012 - γ ∧ β = 2012 - γ - 1) → 
  (Nat.choose 2012 γ) * (a ^ γ) * (b ^ (2012 - γ)) = 
  (Nat.choose 2012 (γ + 1)) * (a ^ (γ + 1)) * (b ^ (2012 - γ - 1))) : 
  δ = a + b → δ = 671 := 
by
  sorry

end expansion_coefficients_eq_l674_674442


namespace distance_home_to_school_l674_674186

-- Define the variables and conditions
variables (D T : ℝ)
def boy_travel_5km_hr_late := 5 * (T + 5 / 60) = D
def boy_travel_10km_hr_early := 10 * (T - 10 / 60) = D

-- State the theorem to prove
theorem distance_home_to_school 
    (H1 : boy_travel_5km_hr_late D T) 
    (H2 : boy_travel_10km_hr_early D T) : 
  D = 2.5 :=
by
  sorry

end distance_home_to_school_l674_674186


namespace length_EQ_proof_l674_674058

def EFGH (E F G H : Point) : Prop :=
  E.distance_to(F) = 8 ∧ F.distance_to(G) = 8 ∧ G.distance_to(H) = 8 ∧ H.distance_to(E) = 8 ∧
  is_perpendicular(EF, FG) ∧ is_perpendicular(FG, GH) ∧ is_perpendicular(GH, HE)

def KLMN (K L M N : Point) : Prop :=
  K.distance_to(L) = 14 ∧ K.distance_to(M) = 8 ∧
  is_perpendicular(KL, KM)

def shaded_area_is_one_third (K L M N : Point) (area_shaded : ℚ) : Prop :=
  area(KLMN(K, L, M, N)) / 3 = area_shaded

def desires_length_EQ (E Q : Point) : ℚ :=
  EQ.length = 3 + 1/3

theorem length_EQ_proof (E F G H K L M N Q : Point)
  (h1 : EFGH(E, F, G, H))
  (h2 : KLMN(K, L, M, N))
  (h3 : is_perpendicular(EH, KL))
  (h4 : shaded_area_is_one_third(K, L, M, N, 112 / 3)) : 
  desires_length_EQ(E, Q) :=
sorry

end length_EQ_proof_l674_674058


namespace car_average_speed_l674_674995

def time_to_travel (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

def total_distance : ℝ := 100 + 50

def total_time : ℝ :=
  time_to_travel 100 30 + time_to_travel 50 40

def average_speed : ℝ :=
  total_distance / total_time

theorem car_average_speed :
  |average_speed - 32.73| < 0.01 :=
by
  sorry

end car_average_speed_l674_674995


namespace optimal_station_placement_l674_674845

def distance_between_buildings : ℕ := 50
def workers_in_building (n : ℕ) : ℕ := n

def total_walking_distance (x : ℝ) : ℝ :=
  |x| + 2 * |x - 50| + 3 * |x - 100| + 4 * |x - 150| + 5 * |x - 200|

theorem optimal_station_placement : ∃ x : ℝ, x = 150 ∧ (∀ y : ℝ, total_walking_distance x ≤ total_walking_distance y) :=
  sorry

end optimal_station_placement_l674_674845


namespace prob_zeros_not_adjacent_l674_674380

theorem prob_zeros_not_adjacent :
  let total_arrangements := (5.factorial : ℝ)
  let zeros_together_arrangements := (4.factorial : ℝ)
  let prob_zeros_together := (zeros_together_arrangements / total_arrangements)
  let prob_zeros_not_adjacent := 1 - prob_zeros_together
  prob_zeros_not_adjacent = 0.6 :=
by
  sorry

end prob_zeros_not_adjacent_l674_674380


namespace parabola_p_equals_24_l674_674800

noncomputable def parabola_p : ℝ :=
  let C := { p : ℝ | 0 < p ∧ ∃ (y_0 : ℝ), y_0^2 = 8 * 4 * p }
  in if h : ∃ p ∈ C, (2 * 24 / 3 : ℝ) = 4 + 24 / 2
     then 24
     else 0

theorem parabola_p_equals_24 : parabola_p = 24 :=
  sorry

end parabola_p_equals_24_l674_674800


namespace telescoping_product_l674_674702

theorem telescoping_product :
  (∏ x in {3, 4, 5, 6, 7}, (x^3 - 1) / (x^3 + 1)) = 57 / 168 := by
  sorry

end telescoping_product_l674_674702


namespace solve_for_y_l674_674900

theorem solve_for_y (y : ℝ) : y^2 - 6 * y + 5 = 0 ↔ y = 1 ∨ y = 5 :=
by
  sorry

end solve_for_y_l674_674900


namespace rectangle_area_increase_l674_674178

-- Definitions to match the conditions
variables {l w : ℝ}

-- The statement 
theorem rectangle_area_increase (h1 : l > 0) (h2 : w > 0) :
  (((1.15 * l) * (1.2 * w) - (l * w)) / (l * w)) * 100 = 38 :=
by
  sorry

end rectangle_area_increase_l674_674178


namespace triangle_side_cube_l674_674611

theorem triangle_side_cube 
  (a b c : ℕ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_gcd : Nat.gcd a (Nat.gcd b c) = 1)
  (angle_condition : ∃ A B : ℝ, A = 3 * B) 
  : ∃ n m : ℕ, (a = n ^ 3 ∨ b = n ^ 3 ∨ c = n ^ 3) :=
sorry

end triangle_side_cube_l674_674611


namespace total_trip_cost_l674_674105

theorem total_trip_cost
  (distance_XZ : ℝ) (h1 : distance_XZ = 4000)
  (distance_XY : ℝ) (h2 : distance_XY = 4500)
  (bus_cost_per_km : ℝ) (h3 : bus_cost_per_km = 0.20)
  (airplane_cost_per_km : ℝ) (h4 : airplane_cost_per_km = 0.12)
  (airplane_booking_fee : ℝ) (h5 : airplane_booking_fee = 120) :
  ∃ total_cost : ℝ, total_cost = 1627.39 :=
by
  let distance_YZ := Real.sqrt (distance_XY^2 - distance_XZ^2) -- by Pythagorean theorem
  have h_distance_YZ : distance_YZ = 2061.55 := sorry -- computation step skipped
  let cost_X_to_Y_bus := distance_XY * bus_cost_per_km
  let cost_X_to_Y_airplane := distance_XY * airplane_cost_per_km + airplane_booking_fee
  let cheaper_X_to_Y := min cost_X_to_Y_bus cost_X_to_Y_airplane
  let cost_Y_to_Z_bus := distance_YZ * bus_cost_per_km
  let cost_Y_to_Z_airplane := distance_YZ * airplane_cost_per_km + airplane_booking_fee
  let cheaper_Y_to_Z := min cost_Y_to_Z_bus cost_Y_to_Z_airplane
  let cost_Z_to_X_bus := distance_XZ * bus_cost_per_km
  let cost_Z_to_X_airplane := distance_XZ * airplane_cost_per_km + airplane_booking_fee
  let cheaper_Z_to_X := min cost_Z_to_X_bus cost_Z_to_X_airplane
  let total_cost := cheaper_X_to_Y + cheaper_Y_to_Z + cheaper_Z_to_X
  have h_total_cost : total_cost = 1627.39 := sorry -- summation step skipped
  use total_cost
  exact h_total_cost

end total_trip_cost_l674_674105


namespace remainder_when_divided_by_15_l674_674188

theorem remainder_when_divided_by_15 (N : ℤ) (k : ℤ) 
  (h : N = 45 * k + 31) : (N % 15) = 1 := by
  sorry

end remainder_when_divided_by_15_l674_674188


namespace acute_triangle_has_extra_vertex_l674_674498

-- Definitions of points and conditions
structure Point :=
  (x : ℤ)
  (y : ℤ)

structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (acute : ∀ (u v w : Point), u ≠ v ∧ v ≠ w ∧ w ≠ u → 
            ((u.x - v.x) * (u.x - w.x) + (u.y - v.y) * (u.y - w.y)) > 0)

theorem acute_triangle_has_extra_vertex (A B C : Point) (h_acute : Triangle A B C) : 
  ∃ P : Point, P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ 
               ((P.x > min (min A.x B.x) C.x ∧ P.x < max (max A.x B.x) C.x) ∧
                (P.y > min (min A.y B.y) C.y ∧ P.y < max (max A.y B.y) C.y) ∨
                (∃ (u v : Point), u ≠ v ∧ Triangle u v P)) :=
sorry

end acute_triangle_has_extra_vertex_l674_674498


namespace drug_effective_with_high_certainty_l674_674161

noncomputable def chi_square_value : ℝ := 10.921
noncomputable def critical_value_001 : ℝ := 10.828

theorem drug_effective_with_high_certainty :
  chi_square_value > critical_value_001 →
  "Consider the drug effective with a probability of making an error not exceeding 0.1%" :=
by
  intro h
  sorry

end drug_effective_with_high_certainty_l674_674161


namespace gas_to_grandmas_house_l674_674984

variables (fuel_efficiency : ℝ) (distance : ℝ)

def gallons_needed (fuel_efficiency distance : ℝ) : ℝ := distance / fuel_efficiency

theorem gas_to_grandmas_house :
  gallons_needed 20 100 = 5 :=
by
  unfold gallons_needed
  sorry

end gas_to_grandmas_house_l674_674984


namespace number_of_terms_l674_674132

open Classical -- Let's assume classical logic for convenience.

-- Define the sequence terms
def a (n : ℕ) (h : 0 < n) : ℚ := 1 / (n * (n + 1))

-- Define the sum of the first n terms of the sequence
noncomputable def S (n : ℕ) : ℚ :=
  if h : 0 < n then 
    (Finset.range n).sum (λ i, a (i + 1) (nat.succ_pos i))
  else 0

theorem number_of_terms :
  ∃ n : ℕ, S n = 10 / 11 ∧ 0 < n :=
begin
  sorry -- Proof steps not required
end

end number_of_terms_l674_674132


namespace intersecting_lines_find_m_l674_674552

theorem intersecting_lines_find_m : ∃ m : ℚ, 
  (∃ x y : ℚ, y = 4*x + 2 ∧ y = -3*x - 18 ∧ y = 2*x + m) ↔ m = -26/7 :=
by
  sorry

end intersecting_lines_find_m_l674_674552


namespace general_formula_a_n_sum_b_n_l674_674761

noncomputable def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

def a_n (n : ℕ) : ℝ :=
if n = 0 then 0 else if n = 1 then 1 else 2^(n-1)

def b_n (n : ℕ) : ℝ := 
(2 * n - 1) / a_n n

theorem general_formula_a_n : ∀ n : ℕ, n > 0 → a_n n = 2^(n-1) :=
begin
  sorry
end

theorem sum_b_n (n : ℕ) (h : n > 0) : 
  let T_n := ∑ i in Finset.range n, b_n (i + 1)
  in T_n = 6 - (2 * n + 3) * (1 / 2)^(n - 1) :=
begin
  sorry
end

end general_formula_a_n_sum_b_n_l674_674761


namespace nursing_home_received_boxes_l674_674098

-- Each condition will be a definition in Lean 4.
def vitamins := 472
def supplements := 288
def total_boxes := 760

-- Statement of the proof problem in Lean
theorem nursing_home_received_boxes : vitamins + supplements = total_boxes := by
  sorry

end nursing_home_received_boxes_l674_674098


namespace find_largest_s_l674_674292

theorem find_largest_s (s : ℚ) :
  (∃ s : ℚ, (15 * s^2 - 40 * s + 18) / (4 * s - 3) + 6 * s = 7 * s - 1) →
  s = 3 :=
begin
  sorry
end

end find_largest_s_l674_674292


namespace exist_complex_root_r_l674_674708

def polynomial : ℂ → ℂ := λ x, x^3 + x^2 - x + 2 

theorem exist_complex_root_r (r : ℝ) :
  (∃ z : ℂ, z.im ≠ 0 ∧ polynomial z = (r : ℂ)) → (3 < r ∧ r < (49 / 27 : ℝ)) :=
by
  sorry

end exist_complex_root_r_l674_674708


namespace cos_F_l674_674052

theorem cos_F (DE EF : ℝ) (h1 : DE = 21) (h2 : EF = 28) (h3 : ∠D = 90): cos (F : ℝ) = 4 / 5 := by
  sorry

end cos_F_l674_674052


namespace weekly_income_sum_5_weeks_l674_674643

variable (weekly_income : Fin 5 → ℝ) -- Represents incomes over the past 5 weeks

-- Base salary
def base_salary : ℝ := 400

-- Average commission for the next two weeks
def avg_commission_next_two_weeks : ℝ := 315

-- Average weekly income target over 7 weeks
def avg_weekly_income_target : ℝ := 500

-- Total income for the next two weeks
def total_income_next_two_weeks : ℝ := 2 * (base_salary + avg_commission_next_two_weeks)

-- Total desired income over 7 weeks
def total_income_7_weeks : ℝ := 7 * avg_weekly_income_target

-- Correct answer we aim to prove
def total_income_past_5_weeks : ℝ := total_income_7_weeks - total_income_next_two_weeks

theorem weekly_income_sum_5_weeks : (∑ i, weekly_income i) = total_income_past_5_weeks := sorry

end weekly_income_sum_5_weeks_l674_674643


namespace Barry_head_standing_turns_l674_674674

theorem Barry_head_standing_turns :
  ∀ (stand_duration sit_duration total_period : ℕ), 
    stand_duration = 10 → 
    sit_duration = 5 → 
    total_period = 2 * 60 → 
    (total_period / (stand_duration + sit_duration)) = 8 :=
by
  intros stand_duration sit_duration total_period h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Barry_head_standing_turns_l674_674674


namespace surface_area_of_union_of_cones_l674_674166

-- Definition of base radius and height
def base_radius := 4
def height := 3

-- Given two right cones with the apex of each cone at the center of the base of the other cone
def cone_surface_area_union := 2 * (π * base_radius^2 + π * base_radius * (Real.sqrt (base_radius^2 + height^2)) - π * (base_radius / 2) * (Real.sqrt ((base_radius / 2)^2 + (height / 2)^2)))

-- The theorem to be proven
theorem surface_area_of_union_of_cones : cone_surface_area_union = 62 * π :=
  by simpa using sorry

end surface_area_of_union_of_cones_l674_674166


namespace balls_in_boxes_l674_674745

theorem balls_in_boxes :
  let balls := {1, 2, 3, 4}
      boxes := {A, B, C}
      condition := (∀ b ∈ boxes, ∃ (S : Set balls), S.nonempty ∧ S.card = 1 ∨ S.card = 2)
  in ∃ (arrangements : ℕ), arrangements = 36 :=
sorry

end balls_in_boxes_l674_674745


namespace determine_positions_l674_674907

-- Define team labels
inductive Team : Type
| A | B | C | D | E
deriving Repr, DecidableEq

open Team

-- Define team positions
def position : Team → ℕ
| D := 1
| B := 2
| C := 3
| A := 4
| E := 5

-- Predictions and their conditions
def prediction1 := (position D = 1 ∨ position C = 2) ∧ ¬(position D = 1 ∧ position C = 2)
def prediction2 := position A = 2
def prediction3 := position C = 3
def prediction4 := (position C = 1 ∨ position D = 4) ∧ ¬(position C = 1 ∧ position D = 4)
def prediction5 := (position A = 2 ∨ position C = 3) ∧ ¬(position A = 2 ∧ position C = 3)

-- The main theorem to prove
theorem determine_positions : 
  position D = 1 ∧
  position B = 2 ∧
  position C = 3 ∧
  position A = 4 ∧
  position E = 5 := by
  sorry

end determine_positions_l674_674907


namespace cost_book_sold_at_loss_l674_674023

-- Definitions for the given conditions
def total_cost := 360
def loss_percentage := 0.85
def gain_percentage := 1.19

theorem cost_book_sold_at_loss :
  ∃ (C1 C2 : ℝ), 
    C1 + C2 = total_cost ∧ 
    C1 * loss_percentage = C2 * gain_percentage ∧ 
    C1 = 210 :=
by
  sorry

end cost_book_sold_at_loss_l674_674023


namespace sam_morning_run_distance_l674_674512

variable (n : ℕ) (x : ℝ)

theorem sam_morning_run_distance (h : x + 2 * n * x + 12 = 18) : x = 6 / (1 + 2 * n) :=
by
  sorry

end sam_morning_run_distance_l674_674512


namespace area_of_shaded_region_l674_674669

theorem area_of_shaded_region:
  let π := 3 in
  let diameter := 2 in
  let leg_length := 2 in
  let radius := diameter / 2 in
  let area_large_triangle := (1 / 2) * leg_length * leg_length in
  let area_semicircle := (1 / 2) * π * radius^2 in
  let area_small_triangle := (1 / 2) * (radius) * (radius) in
  let area_lune := area_semicircle - area_small_triangle in
  let area_shaded := area_large_triangle + area_lune in
  area_shaded = 4.5 :=
by
  -- Proof steps go here
  sorry

end area_of_shaded_region_l674_674669


namespace course_selection_ways_l674_674644

-- Definitions
def numCoursesA : ℕ := 3
def numCoursesB : ℕ := 4
def totalCourses : ℕ := 3

-- Main statement
theorem course_selection_ways : 
  ∃ (ways : ℕ), ways = 30 ∧ 
  (ways = (choose numCoursesA 1 * choose numCoursesB 2 + choose numCoursesA 2 * choose numCoursesB 1)) :=
by
  -- Proof will go here
  sorry

end course_selection_ways_l674_674644


namespace energy_increase_l674_674531

theorem energy_increase (E : ℝ) (s : ℝ) (q : ℝ)
  (hE : 0 < E)
  (hs : 0 < s)
  (hq : 0 < q)
  (h_initial_energy : 4 * (E * q * q / s) = 20) :
  let new_energy := 4 * (2 * (E * q * q / (s / Real.sqrt 2))) + 20 in
  new_energy - 20 = 40 :=
by
  -- The proof would be provided here
  sorry

end energy_increase_l674_674531


namespace intersection_of_M_and_N_l674_674011

def set_M (x : ℝ) : Prop := 1 - 2 / x < 0
def set_N (x : ℝ) : Prop := -1 ≤ x
def set_Intersection (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem intersection_of_M_and_N :
  ∀ x, (set_M x ∧ set_N x) ↔ set_Intersection x :=
by sorry

end intersection_of_M_and_N_l674_674011


namespace count_4digit_odd_numbers_l674_674815

-- Define the conditions
def digits : List ℕ := [1, 2, 3, 4]
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Theorem: Number of 4-digit odd numbers composed of 1, 2, 3, 4 without repetition is 12
theorem count_4digit_odd_numbers : 
  (∃ (nums : List (List ℕ)), 
    (∀ n ∈ nums, ∀ d ∈ n, d ∈ digits) ∧
    (∀ n ∈ nums, n.length = 4) ∧
    (∀ n ∈ nums, n.nodup) ∧
    (∀ n ∈ nums, is_odd (List.last n sorry)) ∧ 
    nums.length = 12). 
sorry

end count_4digit_odd_numbers_l674_674815


namespace problem_inequality_l674_674609

theorem problem_inequality
  (n : ℕ)
  (a b : ℕ → ℝ)
  (h₁ : ∀ i : ℕ, i > 0 → i ≤ n → 0 ≤ a i)
  (h₂ : a 1 ≤ b 1)
  (h₃ : ∀ k : ℕ, k > 0 → k ≤ n → (∑ i in Finset.range k, a (i + 1)) ≤ (∑ i in Finset.range k, b (i + 1)))
  (h₄ : ∀ i : ℕ, i > 0 → i < n → a (i + 1) ≤ a i) :
  (∑ i in Finset.range n, (a (i + 1))^2) ≤ (∑ i in Finset.range n, (b (i + 1))^2) :=
by
  sorry

end problem_inequality_l674_674609


namespace alfred_scooter_purchase_price_l674_674245

theorem alfred_scooter_purchase_price
  (repair_cost : ℝ := 600) 
  (selling_price : ℝ := 5800) 
  (gain_percent : ℝ := 9.433962264150944 / 100) :
  ∃ P : ℝ, P = 4700 := 
by
  let total_cost := P + repair_cost
  let profit := gain_percent * total_cost
  let equation := selling_price - total_cost = profit
  use 4700
  sorry

end alfred_scooter_purchase_price_l674_674245


namespace triangle_stick_problem_l674_674653

theorem triangle_stick_problem (n : ℕ) : 
  (7 + 11 > n ∧ 7 + n > 11 ∧ 11 + n > 7 ∧ 7 + 11 + n > 35) → n = 18 := 
by
  intros h
  cases h
  cases h_left
  cases h_right
  sorry

end triangle_stick_problem_l674_674653


namespace total_weight_full_bucket_l674_674175

theorem total_weight_full_bucket (x y p q : ℝ)
  (h1 : x + (3 / 4) * y = p)
  (h2 : x + (1 / 3) * y = q) :
  x + y = (8 * p - 11 * q) / 5 :=
by
  sorry

end total_weight_full_bucket_l674_674175


namespace calculate_expression_l674_674676

theorem calculate_expression : 
  (12^0 - (3^2 * 6⁻¹ * 2^2)) / (-3⁻²) * 5⁻¹ = 9 :=
by sorry

end calculate_expression_l674_674676


namespace sawing_time_determination_l674_674111

variable (totalLength pieceLength sawTime : Nat)

theorem sawing_time_determination
  (h1 : totalLength = 10)
  (h2 : pieceLength = 2)
  (h3 : sawTime = 10) :
  (totalLength / pieceLength - 1) * sawTime = 40 := by
  sorry

end sawing_time_determination_l674_674111


namespace andrew_brought_40_nuggets_l674_674667

theorem andrew_brought_40_nuggets (h_hotdogs : ℕ) (h_cheese : ℕ) (h_total : ℕ) (h_appetizers : ℕ) :
  h_hotdogs = 30 →
  h_cheese = 20 →
  h_appetizers = 90 →
  h_total = 90 - (30 + 20) →
  h_total = 40 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end andrew_brought_40_nuggets_l674_674667


namespace find_angle_B_and_max_area_l674_674330

-- Definitions for the conditions
variables (A B C a b c : ℝ)
variable (b_eq_2 : b = 2)
variable (B_is_acute : B < π / 2)
variable (m_parallel_n : (2 * sin B, -√3) = (cos (2 * B), 2 * cos^2 (B / 2) - 1))

-- Proof for question and answer equivalence
theorem find_angle_B_and_max_area (h1 : m_parallel_n) (h2 : b_eq_2) :
  (B = π / 3) ∧ (let a_c := a * c in
                let area := (1 / 2) * a_c * (sin B) in
                area ≤ √3) :=
by
  sorry

end find_angle_B_and_max_area_l674_674330


namespace circle_has_greatest_lines_of_symmetry_l674_674585

def num_lines_of_symmetry (figure : String) : Nat :=
  match figure with
  | "circle" => Nat.Infinity.toNat
  | "regular_pentagon" => 5
  | "non_square_rectangle" => 2
  | "isosceles_trapezoid" => 1
  | "square" => 4
  | _ => 0

theorem circle_has_greatest_lines_of_symmetry :
  ∀ (fig : String), fig ∈ ["regular_pentagon", "non_square_rectangle", "isosceles_trapezoid", "square"] → 
  num_lines_of_symmetry "circle" > num_lines_of_symmetry fig :=
by 
  intros
  sorry

end circle_has_greatest_lines_of_symmetry_l674_674585


namespace find_position_20096_l674_674707

theorem find_position_20096 :
  ∃ (k : ℕ) (row col : ℕ), (m : ℕ) = 20096 ∧
  k = Nat.ceil (sqrt (8 * m + 1).toReal - 1) / 2 ∧
  col = m - (k * (k - 1)) / 2 ∧
  row = (k * (k + 1)) / 2 + 1 - m ∧
  row = 5 ∧
  col = 196 :=
by
  sorry

end find_position_20096_l674_674707


namespace max_expression_value_l674_674251

open Real

theorem max_expression_value (a b d x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : (x₁^4 - a * x₁^3 + b * x₁^2 - a * x₁ + d = 0))
  (h2 : (x₂^4 - a * x₂^3 + b * x₂^2 - a * x₂ + d = 0))
  (h3 : (x₃^4 - a * x₃^3 + b * x₃^2 - a * x₃ + d = 0))
  (h4 : (x₄^4 - a * x₄^3 + b * x₄^2 - a * x₄ + d = 0))
  (h5 : (1 / 2 ≤ x₁ ∧ x₁ ≤ 2))
  (h6 : (1 / 2 ≤ x₂ ∧ x₂ ≤ 2))
  (h7 : (1 / 2 ≤ x₃ ∧ x₃ ≤ 2))
  (h8 : (1 / 2 ≤ x₄ ∧ x₄ ≤ 2)) :
  ∃ (M : ℝ), M = 5 / 4 ∧
  (∀ (y₁ y₂ y₃ y₄ : ℝ),
    (y₁^4 - a * y₁^3 + b * y₁^2 - a * y₁ + d = 0) →
    (y₂^4 - a * y₂^3 + b * y₂^2 - a * y₂ + d = 0) →
    (y₃^4 - a * y₃^3 + b * y₃^2 - a * y₃ + d = 0) →
    (y₄^4 - a * y₄^3 + b * y₄^2 - a * y₄ + d = 0) →
    (1 / 2 ≤ y₁ ∧ y₁ ≤ 2) →
    (1 / 2 ≤ y₂ ∧ y₂ ≤ 2) →
    (1 / 2 ≤ y₃ ∧ y₃ ≤ 2) →
    (1 / 2 ≤ y₄ ∧ y₄ ≤ 2) →
    (y = (y₁ + y₂) * (y₁ + y₃) * y₄ / ((y₄ + y₂) * (y₄ + y₃) * y₁)) →
    y ≤ M) := 
sorry

end max_expression_value_l674_674251


namespace solve_equation1_solve_equation2_l674_674903

-- Define the two equations
def equation1 (x : ℝ) := 3 * x - 4 = -2 * (x - 1)
def equation2 (x : ℝ) := 1 + (2 * x + 1) / 3 = (3 * x - 2) / 2

-- The statements to prove
theorem solve_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1.2 :=
by
  sorry

theorem solve_equation2 : ∃ x : ℝ, equation2 x ∧ x = 2.8 :=
by
  sorry

end solve_equation1_solve_equation2_l674_674903


namespace journalist_selection_l674_674281

theorem journalist_selection :
  let n_dom := 5 in
  let n_for := 4 in
  let total_selected := 3 in
  let no_consecutive_domestic (arrangement : list ℕ) := ∀ i, i < arrangement.length - 1 → arrangement.nth i ≠ arrangement.nth (i + 1) in
  let select_and_arrange (D F : ℕ) := ((nat.choose n_dom D) * (nat.choose n_for F) * nat.factorial total_selected) in
  let ways := select_and_arrange 2 1 + select_and_arrange 1 2 in
  (ways = 260) :=
begin
  -- Define the specific combinations and permutations as per the problem's requirements
  sorry
end

end journalist_selection_l674_674281


namespace units_digit_odd_product_l674_674174

def odd_integers_between (a b : ℕ) : list ℕ :=
  (list.filter (λ n, n % 2 = 1) (list.range' a (b - a + 1)))

def not_multiples_of_5 (lst : list ℕ) : list ℕ :=
  list.filter (λ n, n % 5 ≠ 0) lst

def units_digit_of_product_mod_10 (lst : list ℕ) : ℕ :=
  (list.foldl (*) 1 lst) % 10

theorem units_digit_odd_product :
  units_digit_of_product_mod_10 (not_multiples_of_5 (odd_integers_between 11 50)) = 1 :=
sorry

end units_digit_odd_product_l674_674174


namespace reciprocal_of_neg_seven_l674_674149

theorem reciprocal_of_neg_seven : (1 : ℚ) / (-7 : ℚ) = -1 / 7 :=
by
  sorry

end reciprocal_of_neg_seven_l674_674149


namespace cardinal_transitivity_l674_674014

variable {α β γ : Cardinal}

theorem cardinal_transitivity (h1 : α < β) (h2 : β < γ) : α < γ :=
  sorry

end cardinal_transitivity_l674_674014


namespace constant_term_l674_674420

theorem constant_term (n r : ℕ) (h₁ : n = 9) (h₂ : r = 3) : 
  nat.choose n r = 84 :=
by
  rw [h₁, h₂]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)
  sorry

end constant_term_l674_674420


namespace distance_between_skew_lines_in_cube_l674_674631

theorem distance_between_skew_lines_in_cube :
  let A := (0, 0, 0)
  let B := (1, 0, 0)
  let C := (1, 1, 0)
  let D := (0, 1, 0)
  let A₁ := (0, 0, 1)
  let B₁ := (1, 0, 1)
  let C₁ := (1, 1, 1)
  let D₁ := (0, 1, 1)
  let line_A₁C₁ : set (ℝ × ℝ × ℝ) := {p | ∃ t, p = (1 - t) • A₁ + t • C₁}
  let line_BD₁ : set (ℝ × ℝ × ℝ) := {p | ∃ t, p = (1 - t) • B + t • D₁}
  dist(line_A₁C₁, line_BD₁) = (Real.sqrt 6) / 6 :=
by
  sorry

end distance_between_skew_lines_in_cube_l674_674631


namespace acute_triangle_has_extra_vertex_l674_674497

-- Definitions of points and conditions
structure Point :=
  (x : ℤ)
  (y : ℤ)

structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (acute : ∀ (u v w : Point), u ≠ v ∧ v ≠ w ∧ w ≠ u → 
            ((u.x - v.x) * (u.x - w.x) + (u.y - v.y) * (u.y - w.y)) > 0)

theorem acute_triangle_has_extra_vertex (A B C : Point) (h_acute : Triangle A B C) : 
  ∃ P : Point, P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ 
               ((P.x > min (min A.x B.x) C.x ∧ P.x < max (max A.x B.x) C.x) ∧
                (P.y > min (min A.y B.y) C.y ∧ P.y < max (max A.y B.y) C.y) ∨
                (∃ (u v : Point), u ≠ v ∧ Triangle u v P)) :=
sorry

end acute_triangle_has_extra_vertex_l674_674497


namespace intervals_of_monotonicity_max_min_on_interval_l674_674352

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem intervals_of_monotonicity :
  (∀ x y : ℝ, x ∈ (Set.Iio (-1) ∪ Set.Ioi (1)) → y ∈ (Set.Iio (-1) ∪ Set.Ioi (1)) → x < y → f x < f y) ∧
  (∀ x y : ℝ, x ∈ (Set.Ioo (-1) 1) → y ∈ (Set.Ioo (-1) 1) → x < y → f x > f y) :=
by
  sorry

theorem max_min_on_interval :
  (∀ x : ℝ, x ∈ Set.Icc (-3) 2 → f x ≤ 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-3) 2 → -18 ≤ f x) ∧
  ((∃ x₁ : ℝ, x₁ ∈ Set.Icc (-3) 2 ∧ f x₁ = 2) ∧ (∃ x₂ : ℝ, x₂ ∈ Set.Icc (-3) 2 ∧ f x₂ = -18)) :=
by
  sorry

end intervals_of_monotonicity_max_min_on_interval_l674_674352


namespace angle_equality_l674_674053

noncomputable def problem := sorry

theorem angle_equality 
  (A B C D H P : Point)
  (h1 : is_square A B C D)
  (h2 : ∠ D A C = 90)
  (h3 : is_perpendicular A H D C)
  (h4: P ∈ AC)
  (h5: is_tangent P D (circumcircle A B D)) :
  ∠ P B A = ∠ D B H :=
sorry

end angle_equality_l674_674053


namespace value_of_D_l674_674437

theorem value_of_D (E F D : ℕ) (cond1 : E + F + D = 15) (cond2 : F + E = 11) : D = 4 := 
by
  sorry

end value_of_D_l674_674437


namespace evaluate_expression_l674_674170

theorem evaluate_expression : 150 * (150 - 5) - (150 * 150 - 7) = -743 :=
by
  sorry

end evaluate_expression_l674_674170


namespace minimization_part1_l674_674765

variable (n : ℕ) (n_ge_2 : 2 ≤ n)
variable (x : ℕ → ℝ)
variable (sum_condition : ∑ i in Finset.range (n + 1), i * x i = 1)

theorem minimization_part1 :
  (∑ k in Finset.range (n + 1), x k ^ 2 + ∑ i in Finset.range (n + 1), ∑ j in Finset.range (i + 1), x i * x j) 
  ≥ 3 / (n * (n + 1) * (n + 2)) :=
by sorry

end minimization_part1_l674_674765


namespace cubics_product_l674_674692

theorem cubics_product :
  (∏ n in [3, 4, 5, 6, 7], (n^3 - 1) / (n^3 + 1)) = (57 / 168) := by
  sorry

end cubics_product_l674_674692


namespace numeral_system_base_6_l674_674717

theorem numeral_system_base_6 :
  ∃ x : ℕ, x = 6 ∧ (3 * x ^ 2 + 5 * x + 2) * (3 * x + 1) = 2 * x ^ 4 + x ^ 2 + 5 * x + 2 :=
by 
  use 6
  split
  exact rfl
  calc (3 * 6 ^ 2 + 5 * 6 + 2) * (3 * 6 + 1) 
      = (3 * 36 + 30 + 2) * 19 : by {norm_num} 
  ... = 110 * 19 : by {norm_num} 
  ... = 2090 : by {norm_num}
  ... = 2 * 6 ^ 4 + 6 ^ 2 + 5 * 6 + 2 : by norm_num

end numeral_system_base_6_l674_674717


namespace correct_stability_estimator_l674_674587

-- Define the statistics as an enumerated type
inductive Statistic
| sample_mean
| sample_median
| sample_variance
| sample_maximum

-- Stability estimator function
def is_stability_estimator (stat: Statistic) : Prop :=
  stat = Statistic.sample_variance

-- The theorem stating that the correct statistic to estimate population stability is sample variance
theorem correct_stability_estimator : is_stability_estimator Statistic.sample_variance :=
begin
  unfold is_stability_estimator,
  exact rfl,
end

end correct_stability_estimator_l674_674587


namespace integral_zero_l674_674196

-- Define the function f(x) as given in the conditions
def f (x : ℝ) : ℝ := log (sqrt (x^2 + 1) + x) + sin x

-- State the theorem for the integral of the function over the symmetric interval
theorem integral_zero :
  ∫ x in -2023..2023, f x = 0 :=
  sorry

end integral_zero_l674_674196


namespace parabola_focus_l674_674916

theorem parabola_focus (x y p : ℝ) (h_eq : y = 2 * x^2) (h_standard_form : x^2 = (1 / 2) * y) (h_p : p = 1 / 4) : 
    (0, p / 2) = (0, 1 / 8) := by
    sorry

end parabola_focus_l674_674916


namespace find_other_root_of_quadratic_l674_674324

theorem find_other_root_of_quadratic (m x_1 x_2 : ℝ) 
  (h_root1 : x_1 = 1) (h_eqn : ∀ x, x^2 - 4 * x + m = 0) : x_2 = 3 :=
by
  sorry

end find_other_root_of_quadratic_l674_674324


namespace product_of_integers_abs_less_than_2023_l674_674930

theorem product_of_integers_abs_less_than_2023 : 
  ∏ i in (Finset.range 4045).map (λ i, i - 2022), i = 0 :=
by
  sorry

end product_of_integers_abs_less_than_2023_l674_674930


namespace compute_fraction_product_l674_674697

theorem compute_fraction_product :
  (∏ i in (finset.range 5).map (λ n, n + 3), (i ^ 3 - 1) / (i ^ 3 + 1)) = (57 / 168) := by
  sorry

end compute_fraction_product_l674_674697


namespace find_multiple_l674_674904

-- Given conditions
variables (P W m : ℕ)
variables (h1 : P * 24 = W) (h2 : m * P * 6 = W / 2)

-- The statement to prove
theorem find_multiple (P W m : ℕ) (h1 : P * 24 = W) (h2 : m * P * 6 = W / 2) : m = 4 :=
by
  sorry

end find_multiple_l674_674904


namespace non_zero_bk_l674_674858

noncomputable def omega (n : ℕ) [Fact (n > 0)] : ℂ :=
Complex.exp (2 * Real.pi * Complex.I / n)

def b_k (n : ℕ) (a : Fin n → ℂ) (k : ℕ) : ℂ :=
∑ i in Finset.range n, a ⟨i, Nat.lt_of_lt_of_le (Finset.mem_range.mpr (Nat.lt_succ_self i)) (Nat.le_of_lt (Nat.lt_succ_self n))⟩ * (omega n)^(k * i)

theorem non_zero_bk (n : ℕ) (a : Fin n → ℂ) (p : ℕ) (hp : 0 < p) (ha : (Finset.univ.filter (λ i, a i ≠ 0)).card = p) : 
  (Finset.univ.filter (λ k, b_k n a k ≠ 0)).card ≥ n / p :=
sorry

end non_zero_bk_l674_674858


namespace fixed_point_proof_l674_674871

-- Define the structure for the inscribed trapezium
structure trapezium_inscribed_circle (A B C D : Point) (O: Circle) :=
  (inscribed : O.inscribed A B C D)
  (is_trapezium : is_trapezium A B C D)
  (BC_larger_base : larger_base B C A D)

-- Define the points for the problem
variables (A B C D O P N E M : Point)

-- Define the given conditions as hypostheses
def given_conditions : Prop :=
  let t := trapezium_inscribed_circle A B C D O in
  t.BC_larger_base ∧
  P ∉ segment B C ∧
  N ∈ circle_intersection O (line PA) ∧
  E ∈ circle_intersection (circle_with_diameter P D) O ∧
  M ∈ line_intersection (line DE) (line BC)

-- Define the problem to prove
theorem fixed_point_proof
  (A B C D : Point) (O: Circle)
  (P N E M : Point)
  (hp: given_conditions A B C D O P N E M) :
  ∃ F : Point, ∀ t : trapezium_inscribed_circle A B C D O,
    line MN = line MF :=
sorry

end fixed_point_proof_l674_674871


namespace revenue_from_premium_tickets_l674_674219

def number_of_premium_tickets (p s x : ℕ) : Prop :=
  p + s = 200 ∧ 1.5 * x * p + x * s = 3500

theorem revenue_from_premium_tickets (p s x : ℕ) 
  (h1 : p + s = 200) 
  (h2 : 1.5 * x * p + x * s = 3500) :
  1.5 * x * p = 1200 := 
sorry

end revenue_from_premium_tickets_l674_674219


namespace cubics_product_l674_674691

theorem cubics_product :
  (∏ n in [3, 4, 5, 6, 7], (n^3 - 1) / (n^3 + 1)) = (57 / 168) := by
  sorry

end cubics_product_l674_674691


namespace roots_inequality_l674_674545

noncomputable def t (a b t : ℝ) := t^4 + a * t^3 + b * t^2 - (a + b) * (2 * t - 1)

theorem roots_inequality (a b : ℝ) (t_1 t_2 t_3 t_4 : ℝ)
  (h1 : t a b t_1 = 0)
  (h2 : t a b t_2 = 0)
  (h3 : t a b t_3 = 0)
  (h4 : t a b t_4 = 0)
  (h5 : 0 < t_1)
  (h6 : t_1 < t_2)
  (h7 : t_2 < t_3)
  (h8 : t_3 < t_4) :
  t_1 * t_4 > t_2 * t_3 := 
begin
  sorry
end

end roots_inequality_l674_674545


namespace arithmetic_mean_q_r_l674_674912

theorem arithmetic_mean_q_r (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 22) 
  (h3 : r - p = 24) : 
  (q + r) / 2 = 22 := 
by
  sorry

end arithmetic_mean_q_r_l674_674912


namespace petya_vasya_same_result_l674_674158

theorem petya_vasya_same_result (a b : ℤ) 
  (h1 : b = a + 1)
  (h2 : (a - 1) / (b - 2) = (a + 1) / b) :
  (a / b) = 1 := 
by
  sorry

end petya_vasya_same_result_l674_674158


namespace quadratic_monotonic_range_l674_674365

theorem quadratic_monotonic_range {t : ℝ} (h : ∀ x1 x2 : ℝ, (1 < x1 ∧ x1 < 3) → (1 < x2 ∧ x2 < 3) → x1 < x2 → (x1^2 - 2 * t * x1 + 1 ≤ x2^2 - 2 * t * x2 + 1)) : 
  t ≤ 1 ∨ t ≥ 3 :=
by
  sorry

end quadratic_monotonic_range_l674_674365


namespace solve_nat_eq_l674_674520

theorem solve_nat_eq (x y z : ℕ) : x + 1 / (y + 1 / z : ℝ) = 10 / 7 ↔ x = 1 ∧ y = 2 ∧ z = 3 :=
begin
  -- No proof is required.
  sorry
end

end solve_nat_eq_l674_674520


namespace imaginary_part_of_z_l674_674404

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + I) = 2 - 2 * I) : z.im = -2 :=
sorry

end imaginary_part_of_z_l674_674404


namespace final_price_after_discounts_l674_674659

noncomputable def initial_price : ℝ := 9795.3216374269
noncomputable def discount_20 (p : ℝ) : ℝ := p * 0.80
noncomputable def discount_10 (p : ℝ) : ℝ := p * 0.90
noncomputable def discount_5 (p : ℝ) : ℝ := p * 0.95

theorem final_price_after_discounts : discount_5 (discount_10 (discount_20 initial_price)) = 6700 := 
by
  sorry

end final_price_after_discounts_l674_674659


namespace slope_of_line_l674_674565

theorem slope_of_line : ∃ m, (m = -1/2) ∧ (∀ x y : ℝ, x + 2 * y - 3 = 0 → y = m * x + 3/2) :=
begin
  sorry
end

end slope_of_line_l674_674565


namespace root_interval_f_l674_674135

noncomputable def f (x : ℝ) : ℝ := Real.log x - 1 / x

theorem root_interval_f :
  (∃ c ∈ Ioo (1 : ℝ) (Real.exp 1), f c = 0) :=
by
  -- Let f(x) be continuous
  have h_cont: ContinuousOn (fun x => Real.log x - 1 / x) (Set.Ioo 1 (Real.exp 1)),
  { sorry },
  
  -- Evaluate f(1) and f(e)
  have h1: f 1 = -1,
  { calc f 1 = Real.log 1 - 1 / 1 : rfl
         ... = 0 - 1               : by simp
         ... = -1                  : rfl },

  have he: 0 < Real.exp 1  ∧ f (Real.exp 1) = 1 - 1 / Real.exp 1,
  { apply And.intro,
    { exact Real.exp_pos 1 },
    { calc f (Real.exp 1) = Real.log (Real.exp 1) - 1 / Real.exp 1 : rfl
                    ... = 1 - 1 / Real.exp 1                       : by rw Real.log_exp } },

  -- Use Intermediate Value Theorem
  exact IVT (h_cont) h1 he

end root_interval_f_l674_674135


namespace math_problem_l674_674685

theorem math_problem :
  ( ∏ i in [3, 4, 5, 6, 7], (i^3 - 1) / (i^3 + 1) ) = 57 / 84 := sorry

end math_problem_l674_674685


namespace count_valid_n_8000_pow_l674_674307

def count_valid_n : Prop :=
  let e := 8000
  let exp := fun n : ℤ => e * (2 ^ n) * (5 ^ (-n))
  (∀ n : ℤ, (exp n).denom = 1 → (exp n).numerator ∈ {8000}) → {n : ℤ | (exp n).denom = 1}.card = 10

# The statement checks the cardinality of the set of integer n values for which the expression is an integer.
-- Lean code for the statement
theorem count_valid_n_8000_pow : count_valid_n :=
sorry

end count_valid_n_8000_pow_l674_674307


namespace triangle_area_l674_674836

noncomputable def triangle_proof (a b c A B C : ℝ) : Prop :=
  let S := (1/2) * b * c * Real.sin A
  in 
  (A = π / 6) ∧ (S = sqrt 3 / 4)

theorem triangle_area :
  ∀ (A B C a b c : ℝ),
    a = 1 →
    (∀ (x : ℝ), Real.sin (x + π / 3) = 4 * Real.sin (x / 2) * Real.cos (x / 2) → A = x) →
    Real.sin B = sqrt 3 * Real.sin C →
    triangle_proof a b c A B C :=
by
  intros A B C a b c ha hA hBC
  rw ha at hA
  have hA' : A = π / 6, from sorry
  have hc : c = 1, from sorry
  have hb : b = sqrt 3, from sorry
  have area : (1/2) * b * c * Real.sin A = sqrt 3 / 4, from sorry
  exact ⟨hA', area⟩

end triangle_area_l674_674836


namespace parallelogram_area_l674_674269

variables 
  (a b p q : ℝ^3) -- Define vectors as elements of 3-dimensional real space
  (angle : ℝ) -- Define the angle variable 

-- Conditions
axiom norm_p : ‖p‖ = 4 
axiom norm_q : ‖q‖ = 3 
axiom angle_pq : angle = 3 * π / 4 
axiom a_def : a = 3 • p + 2 • q 
axiom b_def : b = 2 • p - q 

-- Statement to prove the area of parallelogram
theorem parallelogram_area : ‖a × b‖ = 42 * sqrt 2 := by
  sorry

end parallelogram_area_l674_674269


namespace sampling_problem_is_systematic_sampling_l674_674429

-- Definitions as provided in the problem
def classes : ℕ := 35
def students_per_class : ℕ := 56
def chosen_student (class_num : ℕ) (student_num : ℕ) : Bool :=
  student_num = 14

-- The Problem: Prove that the sampling method is Systematic Sampling
theorem sampling_problem_is_systematic_sampling :
  (∀ (class_num : ℕ), class_num ≤ classes → chosen_student class_num 14) →
  -- Therefore, the sampling method used in the problem is Systematic Sampling
  "Systematic Sampling" = "Systematic Sampling" :=
begin
  intros h,
  -- Proof skipped
  sorry,
end

end sampling_problem_is_systematic_sampling_l674_674429


namespace compute_fraction_product_l674_674696

theorem compute_fraction_product :
  (∏ i in (finset.range 5).map (λ n, n + 3), (i ^ 3 - 1) / (i ^ 3 + 1)) = (57 / 168) := by
  sorry

end compute_fraction_product_l674_674696


namespace lines_intersect_or_parallel_l674_674884

universe u

noncomputable theory

variables {P : Type u} [add_comm_group P] [vec_space ℝ P]

-- Points on the lines
variables (A1 A2 A3 B1 B2 B3 C1 C2 C3 : P)
-- Conditions given
variables (l1 l2 l3 : set P)
variables 
  (A1_eq_B2 : A1 = B2) 
  (B2_eq_C3 : B2 = C3)
  (A2_eq_B3 : A2 = B3)
  (B3_eq_C1 : B3 = C1)
  (A3_eq_B1 : A3 = B1)
  (B1_eq_C2 : B1 = C2)

-- Function defining the intersection of two lines
def intersection (l1 l2 : set P) [is_subspace l1] [is_subspace l2] (non_parallel : ∃ p, p ∈ l1 ∧ p ∈ l2) : P :=
  classical.some non_parallel

-- Definitions of lines m, n, p
def line_m : set P := line_through (intersection l2 l3 sorry) (intersection l3 l1 sorry)
def line_n : set P := line_through (intersection l3 l1 sorry) (intersection l1 l2 sorry)
def line_p : set P := line_through (intersection l1 l2 sorry) (intersection l2 l3 sorry)

-- The theorem we want to prove
theorem lines_intersect_or_parallel : ∀ (m n p : set P),
  line_m = m → line_n = n → line_p = p → 
  (∃ p, p ∈ m ∧ p ∈ n ∧ p ∈ p) ∨ 
  (parallel m n ∧ parallel n p ∧ parallel m p) :=
begin
  sorry
end

end lines_intersect_or_parallel_l674_674884


namespace number_of_points_P_l674_674865

-- Definitions for the problem
def is_point_inside_circle (P : ℝ × ℝ) (radius : ℝ) : Prop :=
  P.1^2 + P.2^2 < radius^2

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def satisfies_conditions (P A B : ℝ × ℝ) : Prop :=
  (distance P A + distance P B) ≥ real.sqrt 5 ∧ 
  (distance P A)^2 + (distance P B)^2 = 4

-- Points A and B are the endpoints of the given diameter of a circle with radius √2.
def A : ℝ × ℝ := (-real.sqrt 2, 0)
def B : ℝ × ℝ := (real.sqrt 2, 0)

-- The main theorem statement
theorem number_of_points_P (n : ℕ) : 
  ∃ (n : ℕ), (∀ P : ℝ × ℝ, is_point_inside_circle P (real.sqrt 2) → satisfies_conditions P A B) ∧ n = ℕ∞ :=
sorry

end number_of_points_P_l674_674865


namespace complex_polygon_area_l674_674571

theorem complex_polygon_area:
  let side_length := 8
  let bottom_sheet_area := side_length * side_length
  let middle_sheet_rotation := 45
  let top_sheet_rotation := 90
  let top_sheet_shift := side_length / 2
  true → bottom_sheet_area = 64 → (complex_polygon_formed_by_sheets_with_transformations side_length middle_sheet_rotation top_sheet_rotation top_sheet_shift) = 144 :=
by
  intros,
  sorry

end complex_polygon_area_l674_674571


namespace area_probability_l674_674230

/-- Let’s define the segment and set up the conditions. -/
structure Segment :=
(length : ℝ)

def A : Segment := {length := 10}
def B : Segment := {length := 10}

/-- Define the point C arbitrarily chosen on segment AB -/
variable (x : ℝ)
variable (hx : 0 ≤ x ∧ x ≤ 10)

/-- Define the lengths of segments AC and CB -/
def AC := x
def CB := 10 - x

/-- Define the area of the rectangle formed by AC and CB -/
def area := AC * CB

/-- The probability condition that the area is greater than 16 -/
def probability_area_gt_16 := (2 < x ∧ x < 8)

/-- The actual probability calculation -/
def probability := 
  let favorable_outcome_length := 8 - 2 in
  let total_possible_outcome_length := 10 in
  favorable_outcome_length / total_possible_outcome_length

-- The final theorem statement
theorem area_probability :
  probability_area_gt_16 x hx →
  probability = 3 / 5 :=
by
  sorry

end area_probability_l674_674230


namespace parallel_lines_m_l674_674550

theorem parallel_lines_m (m : ℝ) :
  (∀ x y : ℝ, x + 2 * m * y - 1 = 0 → (3 * m - 1) * x - m * y - 1 = 0)
  → m = 0 ∨ m = 1 / 6 := 
sorry

end parallel_lines_m_l674_674550


namespace min_checks_to_find_honey_l674_674589

theorem min_checks_to_find_honey :
  ∃ i, i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} ∧ 
       (∀ j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 
        (j ≤ 8 → i = 4) ∨ (j ≥ 4 → i = 8)) :=
begin
  sorry
end

end min_checks_to_find_honey_l674_674589


namespace prime_divisor_sequence_l674_674758

theorem prime_divisor_sequence 
  (a_d : ℕ) (a_0 : ℕ) (a : fin (d.succ) → ℕ) 
  (P : polynomial ℕ) 
  (d : ℕ) 
  (hP : P = ∑ i in finset.range (d+1), a i * X^i)
  (h_deg : d ≥ 2)
  (h_coeffs : ∀ i, 0 ≤ a i) 
  (h : ∀ i, i < d + 1 → finite a i) :
  (∀ n ≥ 2, ∃ p : ℕ, prime p ∧ p ∣ b n ∧ (∀ m < n, ¬ p ∣ b m)) := sorry

end prime_divisor_sequence_l674_674758


namespace polynomial_sum_of_squares_l674_674459

theorem polynomial_sum_of_squares (p : Polynomial ℝ) (h : ∀ x : ℝ, p.eval x ≥ 0) :
  ∃ k : ℕ, ∃ f : Fin k → Polynomial ℝ, p = ∑ i in Finset.finRange k, (f i)^2 := 
by
  sorry

end polynomial_sum_of_squares_l674_674459


namespace smallest_positive_period_l674_674753

def sequence (x : ℕ → ℤ) : Prop :=
  x 1 = 1 ∧
  x 2 = 2 ∧
  ∀ n : ℕ, 1 ≤ n →
  x (n + 2) = (x (n + 1))^2 - 7 / x n

theorem smallest_positive_period (x : ℕ → ℤ) (h : sequence x) : 
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, x (n + p) = x n ∧ (∀ q : ℕ, q > 0 → (∀ n : ℕ, x (n + q) = x n → p ≤ q)) :=
sorry

end smallest_positive_period_l674_674753


namespace cyclic_product_intersections_l674_674050

open Set

variable {α : Type*} [MetricSpace α]

variables (A B C D E F G H I J A' B' C' D' E' : α)

axiom AD_BE_F : ∃ p : α, LineThrough A D ∧ LineThrough B E ∧ F = p
axiom BE_CA_G : ∃ p : α, LineThrough B E ∧ LineThrough C A ∧ G = p
axiom CA_DB_H : ∃ p : α, LineThrough C A ∧ LineThrough D B ∧ H = p
axiom DB_EC_I : ∃ p : α, LineThrough D B ∧ LineThrough E C ∧ I = p
axiom EC_AD_J : ∃ p : α, LineThrough E C ∧ LineThrough A D ∧ J = p

axiom AI_BE_A' : ∃ p : α, LineThrough A I ∧ LineThrough B E ∧ A' = p
axiom BJ_CA_B' : ∃ p : α, LineThrough B J ∧ LineThrough C A ∧ B' = p
axiom CF_DB_C' : ∃ p : α, LineThrough C F ∧ LineThrough D B ∧ C' = p
axiom DG_EC_D' : ∃ p : α, LineThrough D G ∧ LineThrough E C ∧ D' = p
axiom EH_AD_E' : ∃ p : α, LineThrough E H ∧ LineThrough A D ∧ E' = p

theorem cyclic_product_intersections :
  (dist A B' / dist B' C) * (dist C D' / dist D' E) * (dist E A' / dist A' B) * (dist B C' / dist C' D) * (dist D E' / dist E' A) = 1 := sorry

end cyclic_product_intersections_l674_674050


namespace pipe_a_fill_time_l674_674504

-- Definitions based on the conditions
def time_pipe_a : ℝ  -- time for Pipe A to fill the tank alone
def time_pipe_b := time_pipe_a / 2  -- time for Pipe B to fill the tank alone
def combined_rate := 1 / time_pipe_a + 1 / time_pipe_b
def combined_time := 2  -- both pipes together fill the tank in 2 minutes

-- The statement to be proven
theorem pipe_a_fill_time : combined_rate = 1 / combined_time → time_pipe_a = 6 :=
by {
  sorry
}

end pipe_a_fill_time_l674_674504


namespace average_student_headcount_l674_674260

variable (headcount_02_03 headcount_03_04 headcount_04_05 headcount_05_06 : ℕ)
variable {h_02_03 : headcount_02_03 = 10900}
variable {h_03_04 : headcount_03_04 = 10500}
variable {h_04_05 : headcount_04_05 = 10700}
variable {h_05_06 : headcount_05_06 = 11300}

theorem average_student_headcount : 
  (headcount_02_03 + headcount_03_04 + headcount_04_05 + headcount_05_06) / 4 = 10850 := 
by 
  sorry

end average_student_headcount_l674_674260


namespace average_age_combined_l674_674535

theorem average_age_combined (n1 n2 : ℕ) (avg1 avg2 : ℕ) 
  (h1 : n1 = 45) (h2 : n2 = 60) (h3 : avg1 = 12) (h4 : avg2 = 40) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 28 :=
by
  sorry

end average_age_combined_l674_674535


namespace relationship_among_a_b_c_l674_674775

noncomputable def a := (1/2)^(2/3)
noncomputable def b := (1/5)^(2/3)
noncomputable def c := (1/2)^(1/3)

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  sorry

end relationship_among_a_b_c_l674_674775


namespace part1_solution_part2_solution_part3_solution_l674_674354

section

def f (α : ℝ) (x : ℝ) : ℝ := α * x / (1 + x ^ α)

def a_seq (α : ℝ) (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 1 / 2 else f α (a (n - 1))

theorem part1_solution (n : ℕ) : a_seq 1 (λ n, 1 / (n + 1)) n = 1 / (n + 1) :=
sorry

theorem part2_solution (n : ℕ) (hn : ∀ k, a_seq 1 (λ n, 1 / (n + 1)) k = 1 / (k + 1)) :
  ∑ k in finset.range n, a_seq 1 (λ n, 1 / (n + 1)) k * a_seq 1 (λ n, 1 / (n + 1)) (k + 1) * a_seq 1 (λ n, 1 / (n + 1)) (k + 2)
  = n * (n + 5) / (12 * (n + 2) * (n + 3)) :=
sorry

theorem part3_solution (n : ℕ) (hn : ∀ m, 0 < a_seq 2 (λ n, (n: ℝ)/ (n+1)) m ∧ a_seq 2 (λ n, (n: ℝ)/ (n+1)) m < 1) :
  a_seq 2 (λ n, (n: ℝ)/ (n+1)) (n + 1) - a_seq 2 (λ n, (n: ℝ)/ (n+1)) n < (Real.sqrt 2 + 1) / 8 :=
sorry

end

end part1_solution_part2_solution_part3_solution_l674_674354


namespace problem_l674_674739

def max (x y : ℝ) : ℝ := if x > y then x else y
def min (x y : ℝ) : ℝ := if x < y then x else y

theorem problem (p q r s t : ℝ) (hpq : p < q) (hqr : q < r) (hrs : r < s) (hst : s < t) :
  max (max p (min q r)) (min s (min p t)) = q :=
sorry

end problem_l674_674739


namespace problem_l674_674084

theorem problem (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, f (x + 2) = -f x) :
  f 4 = 0 ∧ (∀ x, f (x + 4) = f x) ∧ (∀ x, f (2 - x) = f (2 + x)) :=
sorry

end problem_l674_674084


namespace probability_two_students_same_school_l674_674570

/-- Definition of the problem conditions -/
def total_students : ℕ := 3
def total_schools : ℕ := 4
def total_basic_events : ℕ := total_schools ^ total_students
def favorable_events : ℕ := 36

/-- Theorem stating the probability of exactly two students choosing the same school -/
theorem probability_two_students_same_school : 
  favorable_events / (total_schools ^ total_students) = 9 / 16 := 
  sorry

end probability_two_students_same_school_l674_674570


namespace time_difference_l674_674237

variables (v : ℝ) (t1 t2 : ℝ)

-- Define the conditions
def first_half_time (v : ℝ) := 20 / v
def second_half_speed (v : ℝ) := v / 2
def second_half_time (v : ℝ) := 40 / v

-- Given conditions
axiom injures_foot_halfway : second_half_time v = 16
axiom total_distance : 40 = 20 * 2

-- Prove the time difference
theorem time_difference (h : first_half_time v = 20 / v ∧ second_half_time v = 16): second_half_time v - first_half_time v = 8 :=
by
  sorry

end time_difference_l674_674237


namespace probability_zeros_not_adjacent_is_0_6_l674_674395

-- Define the total number of arrangements of 5 elements where we have 3 ones and 2 zeros
def total_arrangements : Nat := 5.choose 2

-- Define the number of arrangements where 2 zeros are adjacent
def adjacent_zeros_arrangements : Nat := 4.choose 1 * 2

-- Define the probability that the 2 zeros are not adjacent
def probability_not_adjacent : Rat := (total_arrangements - adjacent_zeros_arrangements) / total_arrangements

-- Prove the desired probability is 0.6
theorem probability_zeros_not_adjacent_is_0_6 : probability_not_adjacent = 3 / 5 := by
  sorry

end probability_zeros_not_adjacent_is_0_6_l674_674395


namespace discriminant_quadratic_eqn_l674_674918

def a := 1
def b := 1
def c := -2
def Δ : ℤ := b^2 - 4 * a * c

theorem discriminant_quadratic_eqn : Δ = 9 := by
  sorry

end discriminant_quadratic_eqn_l674_674918


namespace calculate_divisor_sum_l674_674742

-- Define the sum of divisors excluding the number itself
def sum_proper_divisors (n : ℕ) : ℕ :=
  (Nat.divisors n).filter (λ d => d ≠ n).sum

-- Define the sum of all divisors of the number
def sum_divisors (n : ℕ) : ℕ :=
  (Nat.divisors n).sum

-- Define the given problem to prove that \(\langle \sigma(\langle 15 \rangle) \rangle = 1\)
theorem calculate_divisor_sum : sum_proper_divisors (sum_divisors (sum_proper_divisors 15)) = 1 :=
by
  sorry

end calculate_divisor_sum_l674_674742


namespace find_a_given_conditions_l674_674528

variables (a b c k : ℚ)

theorem find_a_given_conditions 
  (h1 : ∀ b c, a = k / (c * b))
  (h2 : a = 16)
  (h3 : b = 4)
  (h4 : c = 1) :
  let k := 64 in a = 32 / 5 :=
by 
  sorry

end find_a_given_conditions_l674_674528


namespace percent_of_dollar_in_purse_l674_674456

noncomputable def value_of_coins : ℕ :=
  1 + 2 * 5 + 10 + 25 + 2 * 50

theorem percent_of_dollar_in_purse (h : value_of_coins = 146) : (value_of_coins * 100) / 100 = 146 :=
by {
  rw h,
  norm_num
}

end percent_of_dollar_in_purse_l674_674456


namespace probability_not_pulling_prize_twice_l674_674422

theorem probability_not_pulling_prize_twice
  (favorable : ℕ)
  (unfavorable : ℕ)
  (total : ℕ := favorable + unfavorable)
  (P_prize : ℚ := favorable / total)
  (P_not_prize : ℚ := 1 - P_prize)
  (P_not_prize_twice : ℚ := P_not_prize * P_not_prize) :
  P_not_prize_twice = 36 / 121 :=
by
  have favorable : ℕ := 5
  have unfavorable : ℕ := 6
  have total : ℕ := favorable + unfavorable
  have P_prize : ℚ := favorable / total
  have P_not_prize : ℚ := 1 - P_prize
  have P_not_prize_twice : ℚ := P_not_prize * P_not_prize
  sorry

end probability_not_pulling_prize_twice_l674_674422


namespace zeros_not_adjacent_probability_l674_674377

def total_arrangements : ℕ := Nat.factorial 5

def adjacent_arrangements : ℕ := 2 * Nat.factorial 4

def probability_not_adjacent : ℚ := 
  1 - (adjacent_arrangements / total_arrangements)

theorem zeros_not_adjacent_probability :
  probability_not_adjacent = 0.6 := 
by 
  sorry

end zeros_not_adjacent_probability_l674_674377


namespace translation_correct_l674_674242

noncomputable def translation (z w : ℂ) : ℂ := z + w

theorem translation_correct : 
  ∀ (z1 z2 z3 : ℂ), 
  translation z1 z2 = z3 → 
  (translation (2 - complex.I) (z3 - z1) = 6 + 3 * complex.I) :=
by
  intros z1 z2 z3 h
  let z := z3 - z1
  rw [translation, h]
  exact eq.refl _

-- Formalize the specific case
example : translation_correct (1 + 3 * complex.I) (5 + 7 * complex.I) (5 + 7 * complex.I) :=
by
  apply translation_correct
  sorry

end translation_correct_l674_674242


namespace poly_coeff_sum_l674_674822

theorem poly_coeff_sum (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) (x : ℝ) :
  (2 * x + 3)^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + 
                 a_3 * (x + 1)^3 + a_4 * (x + 1)^4 + 
                 a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
                 a_7 * (x + 1)^7 + a_8 * (x + 1)^8 →
  a_0 + a_2 + a_4 + a_6 + a_8 = 3281 :=
by
  sorry

end poly_coeff_sum_l674_674822


namespace area_of_common_region_l674_674641

noncomputable def common_area (length : ℝ) (width : ℝ) (radius : ℝ) : ℝ :=
  let pi := Real.pi
  let sector_area := (pi * radius^2 / 4) * 4
  let triangle_area := (1 / 2) * (width / 2) * (length / 2) * 4
  sector_area - triangle_area

theorem area_of_common_region :
  common_area 10 (Real.sqrt 18) 3 = 9 * (Real.pi) - 9 :=
by
  sorry

end area_of_common_region_l674_674641


namespace cos_angle_POQ_l674_674451

theorem cos_angle_POQ (P Q O : ℝ × ℝ)
  (hP_circle : P.1 ^ 2 + P.2 ^ 2 = 1)
  (hQ_circle : Q.1 ^ 2 + Q.2 ^ 2 = 1)
  (hP_quad1 : 0 < P.1 ∧ 0 < P.2)
  (hQ_quad4 : 0 < Q.1 ∧ Q.2 < 0)
  (hP_x : P.1 = 4 / 5)
  (hQ_x : Q.1 = 5 / 13) :
  Real.cos (Real.arctan2 P.2 P.1 + Real.arctan2 Q.2 Q.1) = 56 / 65 := 
sorry

end cos_angle_POQ_l674_674451


namespace prob_zeros_not_adjacent_l674_674379

theorem prob_zeros_not_adjacent :
  let total_arrangements := (5.factorial : ℝ)
  let zeros_together_arrangements := (4.factorial : ℝ)
  let prob_zeros_together := (zeros_together_arrangements / total_arrangements)
  let prob_zeros_not_adjacent := 1 - prob_zeros_together
  prob_zeros_not_adjacent = 0.6 :=
by
  sorry

end prob_zeros_not_adjacent_l674_674379


namespace ellipse_major_axis_length_l674_674138

theorem ellipse_major_axis_length : 
  (∃ (x y : ℝ), (x^2) / 9 + (y^2) / 4 = 1) → 2 * sqrt 9 = 6 :=
by
  intro h
  have a : sqrt 9 = 3 := by apply Real.sqrt_eq_iff.2; split; norm_num
  rw [← two_mul, a]
  norm_num

example : 2 * sqrt 9 = 6 :=
by
  exact ellipse_major_axis_length ⟨3, 0, by norm_num⟩

end ellipse_major_axis_length_l674_674138


namespace A_annual_share_l674_674657

def annual_gain : ℝ := 24000

def A_share_of_gain (A_investment B_investment C_investment : ℝ) (gain : ℝ) : ℝ :=
  let A_ratio := A_investment * 12
  let B_ratio := B_investment * 6
  let C_ratio := C_investment * 4
  (A_ratio / (A_ratio + B_ratio + C_ratio)) * gain

theorem A_annual_share (A B C : ℝ) (annual_gain : ℝ) :
  A_share_of_gain A (2 * A) (3 * A) annual_gain = 8000 :=
by sorry

end A_annual_share_l674_674657


namespace largest_inscribed_equilateral_triangle_area_l674_674679

noncomputable def inscribed_triangle_area (r : ℝ) : ℝ :=
  let s := r * (3 / Real.sqrt 3)
  let h := (Real.sqrt 3 / 2) * s
  (1 / 2) * s * h

theorem largest_inscribed_equilateral_triangle_area :
  inscribed_triangle_area 10 = 75 * Real.sqrt 3 :=
by
  simp [inscribed_triangle_area]
  sorry

end largest_inscribed_equilateral_triangle_area_l674_674679


namespace not_rectangle_determined_by_angle_and_side_l674_674973

axiom parallelogram_determined_by_two_sides_and_angle : Prop
axiom equilateral_triangle_determined_by_area : Prop
axiom square_determined_by_perimeter_and_side : Prop
axiom rectangle_determined_by_two_diagonals : Prop
axiom rectangle_determined_by_angle_and_side : Prop

theorem not_rectangle_determined_by_angle_and_side : ¬rectangle_determined_by_angle_and_side := 
sorry

end not_rectangle_determined_by_angle_and_side_l674_674973


namespace geometric_sequence_a5_eq_neg1_l674_674339

-- Definitions for the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def roots_of_quadratic (a3 a7 : ℝ) : Prop :=
  a3 + a7 = -4 ∧ a3 * a7 = 1

-- The statement to prove
theorem geometric_sequence_a5_eq_neg1 {a : ℕ → ℝ}
  (h_geo : is_geometric_sequence a)
  (h_roots : roots_of_quadratic (a 3) (a 7)) :
  a 5 = -1 :=
sorry

end geometric_sequence_a5_eq_neg1_l674_674339


namespace find_parabola_eq_l674_674288

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = -3 * x ^ 2 + 18 * x - 22 ↔ (x - 3) ^ 2 + 5 = y

theorem find_parabola_eq :
  ∃ a b c : ℝ, (vertex = (3, 5) ∧ axis_of_symmetry ∧ point_on_parabola = (2, 2)) →
  parabola_equation a b c :=
sorry

end find_parabola_eq_l674_674288


namespace cards_sum_divisible_l674_674427

theorem cards_sum_divisible (k n : ℕ) (h_gcd : Nat.gcd k n = 1)
  (h_cards : ∀ i : ℕ, ∃ (A_i : Finset ℕ), (∑ j in A_i, j) % n = 0) :
  ∃ (cards : Vector ℕ n), (∑ i, cards.get i) % n = 0 := 
sorry

end cards_sum_divisible_l674_674427


namespace side_length_square_field_l674_674206

-- Definitions based on the conditions.
def time_taken := 56 -- in seconds
def speed := 9 * 1000 / 3600 -- in meters per second, converting 9 km/hr to m/s
def distance_covered := speed * time_taken -- calculating the distance covered in meters
def perimeter := 4 * 35 -- defining the perimeter given the side length is 35

-- Problem statement for proof: We need to prove that the calculated distance covered matches the perimeter.
theorem side_length_square_field : distance_covered = perimeter :=
by
  sorry

end side_length_square_field_l674_674206


namespace coordinates_of_point_l674_674034

theorem coordinates_of_point (a : ℝ) (P : ℝ × ℝ) (hy : P = (a^2 - 1, a + 1)) (hx : (a^2 - 1) = 0) :
  P = (0, 2) ∨ P = (0, 0) :=
sorry

end coordinates_of_point_l674_674034
