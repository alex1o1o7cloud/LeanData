import Mathbin.Data.Finset
import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.TangentCone
import Mathlib.Analysis.Optimization.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.FieldTheory.Vieta
import Mathlib.Init.Data.Complex.Basic
import Mathlib.Init.Function
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Divisors
import Mathlib.Probability.Basic
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Real

namespace floor_sum_eq_log_sum_l751_751616

theorem floor_sum_eq_log_sum (n : ℕ) (hn : n > 1) :
  (Finset.sum (Finset.range n) (λ m, ⌊n^(1/(m+2))⌋)) = 
  (Finset.sum (Finset.range n) (λ k, ⌊log k (n:ℝ)⌋)) :=
sorry

end floor_sum_eq_log_sum_l751_751616


namespace geometric_sequence_a6_l751_751542

theorem geometric_sequence_a6 (a : ℕ → ℕ) (r : ℕ)
  (h₁ : a 1 = 1)
  (h₄ : a 4 = 8)
  (h_geometric : ∀ n, a n = a 1 * r^(n-1)) : 
  a 6 = 32 :=
by
  sorry

end geometric_sequence_a6_l751_751542


namespace sum_of_solutions_l751_751266

theorem sum_of_solutions (s : Finset ℝ) :
  (∀ x ∈ s, |x^2 - 16 * x + 60| = 4) →
  s.sum id = 24 := 
by
  sorry

end sum_of_solutions_l751_751266


namespace sin_450_eq_1_l751_751386

theorem sin_450_eq_1 : sin (450 * π / 180) = 1 := by
  have angle_eq : 450 = 360 + 90 := by norm_num
  -- Simplify 450 degrees to radians
  rw [angle_eq, Nat.cast_add, add_mul, Nat.cast_mul]
  -- Convert degrees to radians
  rw [sin_add, sin_mul_pi_div, cos_mul_pi_div, sin_mul_pi_div, (show (90 : ℝ) = π / 2 from by norm_num)]

  sorry -- Omitting proof details

end sin_450_eq_1_l751_751386


namespace sum_x_coords_P3_l751_751337

theorem sum_x_coords_P3 (x_coords : Fin 100 → ℝ) 
  (sum_x_coords_P1 : ∑ i : Fin 100, x_coords i = 2009) : 
  let P2_coords := λ i : Fin 100, (x_coords i + x_coords ((i + 1) % 100)) / 2
  let P3_coords := λ i : Fin 100, (P2_coords i + P2_coords ((i + 1) % 100)) / 2
  (∑ i : Fin 100, P3_coords i) = 2009 := 
by
  sorry

end sum_x_coords_P3_l751_751337


namespace smallest_n_l751_751262

theorem smallest_n (n : ℕ) (h : ↑n > 0 ∧ (Real.sqrt (↑n) - Real.sqrt (↑n - 1)) < 0.02) : n = 626 := 
by
  sorry

end smallest_n_l751_751262


namespace A_inter_B_l751_751893

variable (U : Set ℕ) (A : Set ℕ) (CUA : Set ℕ) (B : Set ℕ)

def is_universal_set : Prop :=
  U = {0, 1, 3, 7, 9}

def complement_A : Prop :=
  CUA = {0, 5, 9}

def B_set : Prop :=
  B = {3, 5, 7}

theorem A_inter_B :
  (is_universal_set U) →
  (complement_A U CUA) →
  (B_set B) →
  A = {1, 3, 7} →
  A ∩ B = {3, 7} :=
by
  intros hU hCUA hB hA
  sorry

end A_inter_B_l751_751893


namespace girls_in_class_l751_751379

theorem girls_in_class : ∀ (boys girls : ℕ), boys = 16 ∧ 4 * girls = 5 * boys → girls = 20 :=
by
  intros boys girls h
  cases h with hb hg
  rw [hb] at hg
  have : girls = 20, by linarith,
  exact this
  sorry

end girls_in_class_l751_751379


namespace romance_movie_tickets_l751_751216

-- Define the given conditions.
def horror_movie_tickets := 93
def relationship (R : ℕ) := 3 * R + 18 = horror_movie_tickets

-- The theorem we need to prove
theorem romance_movie_tickets (R : ℕ) (h : relationship R) : R = 25 :=
by sorry

end romance_movie_tickets_l751_751216


namespace circles_intersect_l751_751798

open Real

-- Define the first circle
def circle1_center := (-2 : ℝ, 0 : ℝ)
def circle1_radius := 2 : ℝ

-- Define the second circle
def circle2_center := (2 : ℝ, 1 : ℝ)
def circle2_radius := 3 : ℝ

-- Calculate the distance between the centers
def distance_between_centers :=
  sqrt ((circle2_center.1 - circle1_center.1)^2 + (circle2_center.2 - circle1_center.2)^2)

-- The condition that the circles intersect
theorem circles_intersect :
  |circle1_radius - circle2_radius| < distance_between_centers ∧
  distance_between_centers < circle1_radius + circle2_radius :=
by {
  -- Plug in the centers and radii, and use the conditions to prove the theorem
  let r1 := circle1_radius,
  let r2 := circle2_radius,
  let d := distance_between_centers,
  have h_diff : |r1 - r2| = 1 := by norm_num,
  have h_sum : r1 + r2 = 5 := by norm_num,
  have h_d : d = sqrt 17 := by {
    simp [distance_between_centers, circle1_center, circle2_center],
  },
  rw [h_diff, h_sum, h_d],
  split;
  norm_num; linarith,
  sorry
}

end circles_intersect_l751_751798


namespace intersection_eq_l751_751158

theorem intersection_eq {A B : Set ℤ} (hA : A = Set.ofList [-1, 1, 2]) (hB : B = Set.ofList [2, 3]) :
  A ∩ B = Set.ofList [2] :=
by
  rw [hA, hB]
  exact sorry

end intersection_eq_l751_751158


namespace domain_of_function_l751_751644

theorem domain_of_function:
  (∀ x, (x - 1 ≥ 0 → 2 - x > 0) ↔ 1 ≤ x ∧ x < 2) → (∀ x, (1 ≤ x ∧ x < 2) → (∃ y, y = sqrt (x - 1) + log (2 - x))) :=
by sorry

end domain_of_function_l751_751644


namespace deck_length_is_30_l751_751978

theorem deck_length_is_30
  (x : ℕ)
  (h1 : ∀ a : ℕ, a = 40 * x)
  (h2 : ∀ b : ℕ, b = 3 * a + 1 * a ∧ b = 4800) :
  x = 30 := by
  sorry

end deck_length_is_30_l751_751978


namespace modulus_ge_one_of_poly_eq_zero_l751_751604

theorem modulus_ge_one_of_poly_eq_zero {n : ℕ} {a : ℕ → ℝ} {z : ℂ}
  (h_non_neg : ∀ i, 0 ≤ a i)
  (h_non_decreasing : ∀ i, i < n → a i ≤ a (i + 1))
  (h_poly_zero : (∑ i in finset.range (n + 1), a i * z^(n - i) : ℂ) = 0) :
  |z| ≥ 1 :=
sorry

end modulus_ge_one_of_poly_eq_zero_l751_751604


namespace complex_exponent_identity_l751_751863

theorem complex_exponent_identity (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (5 * Real.pi / 180)) : 
  z^1500 + z^(-1500) = 1 :=
sorry

end complex_exponent_identity_l751_751863


namespace jail_time_calculation_l751_751246

def total_arrests (arrests_per_day : ℕ) (cities : ℕ) (days : ℕ) : ℕ := 
  arrests_per_day * cities * days

def jail_time_before_trial (arrests : ℕ) (days_before_trial : ℕ) : ℕ := 
  days_before_trial * arrests

def jail_time_after_trial (arrests : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_after_trial * arrests

def combined_jail_time (weeks_before_trial : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_before_trial + weeks_after_trial

noncomputable def total_jail_time_in_weeks : ℕ := 
  let arrests := total_arrests 10 21 30
  let weeks_before_trial := jail_time_before_trial arrests 4 / 7
  let weeks_after_trial := jail_time_after_trial arrests 1
  combined_jail_time weeks_before_trial weeks_after_trial

theorem jail_time_calculation : 
  total_jail_time_in_weeks = 9900 :=
sorry

end jail_time_calculation_l751_751246


namespace find_intersections_l751_751594

-- Define f(x) under the given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 1 then x
  else if h : 1 < x ∧ x ≤ 2 then 2 - x
  else if h : -1 ≤ x ∧ x < 0 then -x
  else 0 -- Placeholder for the actual definition on the rest of the domain

-- Define the given exponential function
def g (x : ℝ) : ℝ := (1 / 10) ^ x

theorem find_intersections : ∃ n, n = 4 ∧ (finset.univ.filter (λ x:ℝ, x ∈ Icc (0:ℝ) (4:ℝ) ∧ f x = g x)).card = n :=
by 
  -- Proof to be provided
  sorry

end find_intersections_l751_751594


namespace inequality_holds_l751_751300

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l751_751300


namespace parabola_directrix_tangent_circle_l751_751885

noncomputable def parabola_has_tangent_circle_directrix (p : ℝ) (h : p > 0) : Prop :=
  let directrix := -p / 2 in
  ∃ (x y : ℝ), ((x - 3)^2 + y^2 = 16) ∧ ((y^2 = 2 * p * x) ∨ (y^2 = 2 * p * (2 * directrix - x)))

theorem parabola_directrix_tangent_circle (p : ℝ) (h : p > 0) :
  parabola_has_tangent_circle_directrix p h → p = 2 :=
sorry

end parabola_directrix_tangent_circle_l751_751885


namespace sum_of_digits_in_rectangle_l751_751993

theorem sum_of_digits_in_rectangle :
  ∃ A B C : ℕ,
    (4 + A + 1 + B = 12) ∧
    (4 + A + 1 + B = 6 + 6) ∧
    (C + 1 + 6 + C = 11) ∧
    (1 + B + 2 + C = 9) ∧
    (A + 8 + 8 = 8) ∧
    (A + 8 + B + 2 + C = 13) ∧
    (9 = 4 + A + 1 + B) ∧
    (B + 2 + C = 9) ∧    
    B = 5 ∧ A = 2 ∧ C = 6 :=
sorry

end sum_of_digits_in_rectangle_l751_751993


namespace area_of_square_A_l751_751202

noncomputable def square_areas (a b : ℕ) : Prop :=
  (b ^ 2 = 81) ∧ (a = b + 4)

theorem area_of_square_A : ∃ a b : ℕ, square_areas a b → a ^ 2 = 169 :=
by
  sorry

end area_of_square_A_l751_751202


namespace EFN_collinear_l751_751951

open geometry

/- Definitions -/
variables {A B C D E F N : Point} {M : Point}

/- Hypotheses -/
def is_cyclic_quadrilateral (A B C D : Point) : Prop := 
  -- Placeholder for defining a cyclic quadrilateral
  sorry

def is_intersection (P Q : Point) (l₁ l₂ : Line) : Prop := 
  -- Placeholder for defining intersection of two lines
  sorry

def is_midpoint (M : Point) (A B : Point) : Prop := 
  -- Placeholder for defining that a point is the midpoint of a segment
  sorry

def is_circumcircle_point (N : Point) (triangle : Triangle) (M : Point) : Prop := 
  -- Placeholder for defining N on the circumcircle of a triangle distinct from M 
  sorry

def collinear (P Q R : Point) : Prop := 
  -- Placeholder for defining collinearity of three points
  sorry

/- Theorem Statement -/
theorem EFN_collinear (h_cyclic : is_cyclic_quadrilateral A B C D)
  (hE : is_intersection E AD BC) 
  (hF : is_intersection F AC BD)
  (hM : is_midpoint M C D) 
  (hN : is_circumcircle_point N (triangle A M B) M)
  (h_ratio : (AM/BM) = (AN/BN)) :
  collinear E F N := 
begin
  sorry
end

end EFN_collinear_l751_751951


namespace solve_for_x_l751_751917

theorem solve_for_x : 
  ∃ x, (sqrt 0.85 * (2 / (3 * Real.pi)) * x^(Real.log x / Real.log 2) = 36) → x ≈ 4.094 :=
by
  sorry

end solve_for_x_l751_751917


namespace minimum_employees_l751_751736

theorem minimum_employees : 
  ∀ (customer_service technical_support both : ℕ), 
    customer_service = 150 → 
    technical_support = 125 → 
    both = 50 → 
    (customer_service + technical_support - both) = 225 :=
by
  intros customer_service technical_support both h1 h2 h3
  rw [h1, h2, h3]
  sorry

end minimum_employees_l751_751736


namespace find_k_l751_751105

/-
    Given a right triangle △ACB with ∠C = 90°, points D and E lie on BC and CA respectively, and 
    BD / AC = AE / CD = K. Let BE intersect AD at O. If ∠BOD = 75°, then K = 2 + √3.
-/
theorem find_k 
    (A B C D E O : Type)
    [is_right_triangle : is_right_triangle (A, C, B)]
    (hC : ∠ C = 90)
    (D_on_BC : lies_on_line D B C)
    (E_on_CA : lies_on_line E C A)
    (k_eq_1 : BD / AC = K)
    (k_eq_2 : AE / CD = K)
    (O_intersection : intersects BE AD O)
    (angle_BOD : ∠ B O D = 75)
    :
    K = 2 + sqrt 3 := 
sorry

end find_k_l751_751105


namespace sequence_bn_geometric_and_max_n_st_l751_751961

-- Define the sequence a_n with the given conditions
def seq_an (n : ℕ) : ℕ := 
  if n = 1 then 1
  else 2 * (seq_an (n-1))^2

-- Define T_n as the product of first n terms of a_n
noncomputable def T (n : ℕ) : ℕ := 
  (List.prod (List.map seq_an (List.range (n+1))))

-- Define b_n = 1 + log_2 a_n
def b (n : ℕ) : ℕ := 1 + (Nat.log 2 (seq_an n))

-- Define S_n as the sum of the first n terms of the sequence log_2 b_n / b_n
noncomputable def S (n : ℕ) : ℕ := 
  ∑ i in Finset.range n, (Nat.log 2 (b i)) / (b i)

-- The problem statement:
theorem sequence_bn_geometric_and_max_n_st (b (n) = 2^(n-1)) (∃ n : ℕ, (S n * b n) < 2023 ∧ ∀ k > n, ¬(S k * b k < 2023)) :=
begin
  -- Proof not needed, so we add sorry to skip the proof
  sorry
end

end sequence_bn_geometric_and_max_n_st_l751_751961


namespace cylinder_surface_area_base_diameter_height_4_l751_751204

theorem cylinder_surface_area_base_diameter_height_4 (d h : ℝ) (hd : d = 4) (hh : h = 4) :
  let r := d / 2 in
  let S := 2 * Real.pi * r * (r + h) in
  S = 24 * Real.pi :=
by
  rw [hd, hh]
  let r := 4 / 2
  let S := 2 * Real.pi * r * (r + 4)
  have hr : r = 2 := by norm_num
  rw [hr]
  let S' := 2 * Real.pi * 2 * (2 + 4)
  have hs : S' = 24 * Real.pi := by norm_num
  rw [hs]
  sorry

end cylinder_surface_area_base_diameter_height_4_l751_751204


namespace sequence_sum_l751_751447

theorem sequence_sum (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (b : ℕ → ℕ)
  (c : ℕ → ℕ)
  (T : ℕ → ℕ) 
  (hS : ∀ n, S n = n ^ 2)
  (ha : ∀ n, a n = 2 * n - 1)
  (hb : ∀ n, b n = 2 ^ (n - 1))
  (hc : ∀ n, c n = a (b n))
  (hT : ∀ n, T n = (Finset.range n).sum (λ i, c (i + 1))) :
  ∀ n, T n = 2 ^ (n + 1) - 2 - n :=
begin
  sorry
end

end sequence_sum_l751_751447


namespace natural_number_sum_of_coprimes_l751_751617

theorem natural_number_sum_of_coprimes (n : ℕ) (h : n ≥ 2) : ∃ a b : ℕ, n = a + b ∧ Nat.gcd a b = 1 :=
by
  use (n - 1), 1
  sorry

end natural_number_sum_of_coprimes_l751_751617


namespace sphere_radius_of_melted_cone_l751_751672

noncomputable def coneVolume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * π * r^3

theorem sphere_radius_of_melted_cone :
  ∀ (r_cone s_cone l_cone : ℝ) (r_sphere : ℝ),
  r_cone = 3 →
  s_cone = 5 →
  l_cone = real.sqrt (s_cone^2 - r_cone^2) →
  coneVolume r_cone l_cone = 12 * π →
  sphereVolume r_sphere = 12 * π →
  r_sphere = real.cbrt 9 :=
by
  intros r_cone s_cone l_cone r_sphere h_rcone h_scone h_lcone h_coneVolume h_sphereVolume   
  sorry

end sphere_radius_of_melted_cone_l751_751672


namespace incircle_radius_is_correct_l751_751683

-- Define the given conditions
structure Triangle :=
  (D E F : ℝ)
  (angleD : ℝ)
  (DF : ℝ)

def right_angle_at_F (T : Triangle) : Prop := T.angleD = 45 ∧ T.DF = 8

-- Define the radius of the incircle function based on conditions
noncomputable def radius_of_incircle (T : Triangle) : ℝ :=
  let DF := T.DF in
  let DE := DF in
  let EF := DF * Real.sqrt 2 in
  let area := (DF * DE) / 2 in
  let s := (DF + DE + EF) / 2 in
  let r := area / s in
  r

-- The proof goal
theorem incircle_radius_is_correct (T : Triangle) (h : right_angle_at_F T) :
  radius_of_incircle T = 4 - 2 * Real.sqrt 2 := by
  sorry

end incircle_radius_is_correct_l751_751683


namespace proof_problem_l751_751976

open Set Real

noncomputable def f (x : ℝ) : ℝ := sin x
noncomputable def g (x : ℝ) : ℝ := cos x
def U : Set ℝ := univ
def M : Set ℝ := {x | f x ≠ 0}
def N : Set ℝ := {x | g x ≠ 0}
def C_U (s : Set ℝ) : Set ℝ := U \ s

theorem proof_problem :
  {x : ℝ | f x * g x = 0} = (C_U M) ∪ (C_U N) :=
by
  sorry

end proof_problem_l751_751976


namespace evaluate_f_x_plus_3_l751_751197

def f (x : ℝ) : ℝ := x^2

theorem evaluate_f_x_plus_3 (x : ℝ) : f (x + 3) = x^2 + 6 * x + 9 := by
  sorry

end evaluate_f_x_plus_3_l751_751197


namespace common_sales_in_july_eq_two_l751_751341

def isSaleDayBookstore (d : ℕ) : Prop :=
  ∃ k : ℕ, d = 4 * k ∧ d ≤ 31

def isSaleDayClothingStore (d : ℕ) : Prop :=
  ∃ k : ℕ, d = 1 + 7 * k ∧ d ≤ 31

def common_sale_days (n : ℕ) : Prop := n ∈ {d : ℕ | isSaleDayBookstore d ∧ isSaleDayClothingStore d}

theorem common_sales_in_july_eq_two : 
  ∃ S : Finset ℕ, (∀ s ∈ S, common_sale_days s) ∧ S.card = 2 :=
sorry

end common_sales_in_july_eq_two_l751_751341


namespace charlyn_visible_area_l751_751378

noncomputable def visible_area {s : ℝ} (side : s = 10) (radius : ℝ) (r : radius = 2) : ℝ :=
  let square_area := s * s
  let inner_square_side := s - 2 * r
  let inner_square_area := inner_square_side * inner_square_side
  let viewable_region_inside := square_area - inner_square_area
  let viewable_region_outside := (s * radius * 4) + (real.pi * radius * radius)
  viewable_region_inside + viewable_region_outside

theorem charlyn_visible_area : visible_area (side := 10) (radius := 2) ≈ 157 := by
  sorry

end charlyn_visible_area_l751_751378


namespace points_in_fourth_quarter_l751_751777

theorem points_in_fourth_quarter 
  (L_1 : ℕ := 10)
  (W_1 : ℕ := 2 * L_1)
  (W_2 : ℕ := W_1 + 10)
  (W_3 : ℕ := W_2 + 20)
  (W_total : ℕ := 80) :
  W_total - W_3 = 30 :=
by
  calc
    W_total - W_3 = 80 - (W_2 + 20) : by rfl
    ... = 80 - ((W_1 + 10) + 20) : by rfl
    ... = 80 - ((2 * 10 + 10) + 20) : by rfl
    ... = 80 - (20 + 10 + 20) : by rfl
    ... = 80 - 50 : by rfl
    ... = 30 : by rfl

end points_in_fourth_quarter_l751_751777


namespace coeff_x4_l751_751819

-- Define the two polynomials
def P1 := 2 * (λ x: ℕ, x^3) + 5 * (λ x: ℕ, x^2) - 3 * (λ x: ℕ, x)
def P2 := 3 * (λ x: ℕ, x^3) - 8 * (λ x: ℕ, x^2) + 6 * (λ x: ℕ, x) - 9

-- Define the product of the two polynomials
def product := (λ x : ℕ, (2 * x^3 + 5 * x^2 - 3 * x) * (3 * x^3 - 8 * x^2 + 6 * x - 9))

-- Statement that we want to prove
theorem coeff_x4: coeff (product) 4 = -37 :=
sorry

end coeff_x4_l751_751819


namespace keaton_annual_profit_l751_751580

theorem keaton_annual_profit :
  let orange_harvests_per_year := 12 / 2
  let apple_harvests_per_year := 12 / 3
  let peach_harvests_per_year := 12 / 4
  let blackberry_harvests_per_year := 12 / 6

  let orange_profit_per_harvest := 50 - 20
  let apple_profit_per_harvest := 30 - 15
  let peach_profit_per_harvest := 45 - 25
  let blackberry_profit_per_harvest := 70 - 30

  let total_orange_profit := orange_harvests_per_year * orange_profit_per_harvest
  let total_apple_profit := apple_harvests_per_year * apple_profit_per_harvest
  let total_peach_profit := peach_harvests_per_year * peach_profit_per_harvest
  let total_blackberry_profit := blackberry_harvests_per_year * blackberry_profit_per_harvest

  let total_annual_profit := total_orange_profit + total_apple_profit + total_peach_profit + total_blackberry_profit

  total_annual_profit = 380
:= by
  sorry

end keaton_annual_profit_l751_751580


namespace prob_C_correct_l751_751339

noncomputable def prob_A : ℚ := 2/9
noncomputable def prob_B : ℚ := 1/6
noncomputable def prob_C : ℚ := 11/90 -- This derives from our conditions

theorem prob_C_correct :
  let prob_D : ℚ := prob_C,
      prob_E : ℚ := prob_C,
      prob_F : ℚ := 2 * prob_C,
      sum_prob := prob_A + prob_B + prob_C + prob_D + prob_E + prob_F
  in sum_prob = 1 → prob_C = 11 / 90 :=
by
  intros prob_D prob_E prob_F sum_prob h,
  sorry

end prob_C_correct_l751_751339


namespace no_integer_a_exists_l751_751415

theorem no_integer_a_exists (a x : ℤ)
  (h : x^3 - a * x^2 - 6 * a * x + a^2 - 3 = 0)
  (unique_sol : ∀ y : ℤ, (y^3 - a * y^2 - 6 * a * y + a^2 - 3 = 0 → y = x)) :
  false :=
by 
  sorry

end no_integer_a_exists_l751_751415


namespace isogonal_conjugation_unit_circle_l751_751665

noncomputable def isogonally_conjugate (z w : ℂ) : Prop := sorry

theorem isogonal_conjugation_unit_circle
  (a b c z w : ℂ)
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = 1)
  (h3 : ∥c∥ = 1)
  (h4 : isogonally_conjugate z w) :
  z + w + a * b * c * conj z * conj w = a + b + c := 
sorry

end isogonal_conjugation_unit_circle_l751_751665


namespace gcd_polynomials_l751_751458

theorem gcd_polynomials (b : ℤ) (h : b % 8213 = 0 ∧ b % 2 = 1) :
  Int.gcd (8 * b^2 + 63 * b + 144) (2 * b + 15) = 9 :=
sorry

end gcd_polynomials_l751_751458


namespace ellipse_equation_l751_751866

def major_axis_length (a : ℝ) := 2 * a = 8
def eccentricity (c a : ℝ) := c / a = 3 / 4

theorem ellipse_equation (a b c x y : ℝ) (h1 : major_axis_length a)
    (h2 : eccentricity c a) (h3 : b^2 = a^2 - c^2) :
    (x^2 / 16 + y^2 / 7 = 1 ∨ x^2 / 7 + y^2 / 16 = 1) :=
by
  sorry

end ellipse_equation_l751_751866


namespace part1_part2_l751_751475

def f (x : ℝ) (a : ℝ) : ℝ := log2 ((2 / (x - 1)) + a)

theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f x a = -f (-x) a) ↔ a = 1 :=
by sorry

theorem part2 (b : ℝ) : 
  (∀ x : ℝ, 2^(f (2^x) 1) + 3*2^x - b ≥ 0) ↔ b ≤ 2*sqrt 6 + 4 :=
by sorry

end part1_part2_l751_751475


namespace find_principal_amount_l751_751763

variables (P R : ℝ)

theorem find_principal_amount (h : (4 * P * (R + 2) / 100) - (4 * P * R / 100) = 56) : P = 700 :=
sorry

end find_principal_amount_l751_751763


namespace circle_line_intersection_length_l751_751471

theorem circle_line_intersection_length :
  let C := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}
  let L := {p : ℝ × ℝ | p.1 - sqrt 3 * p.2 - 5 = 0}
  ∃ P Q : ℝ × ℝ, P ∈ C ∧ P ∈ L ∧ Q ∈ C ∧ Q ∈ L ∧ dist P Q = sqrt 7 :=
by
  sorry

end circle_line_intersection_length_l751_751471


namespace inequality_proof_l751_751280

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751280


namespace three_term_inequality_l751_751292

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l751_751292


namespace inequality_proof_l751_751288

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751288


namespace probability_xavier_yvonne_not_zelda_wendell_l751_751164

theorem probability_xavier_yvonne_not_zelda_wendell
  (P_Xavier_solves : ℚ)
  (P_Yvonne_solves : ℚ)
  (P_Zelda_solves : ℚ)
  (P_Wendell_solves : ℚ) :
  P_Xavier_solves = 1/4 →
  P_Yvonne_solves = 1/3 →
  P_Zelda_solves = 5/8 →
  P_Wendell_solves = 1/2 →
  (P_Xavier_solves * P_Yvonne_solves * (1 - P_Zelda_solves) * (1 - P_Wendell_solves)) = 1/64 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end probability_xavier_yvonne_not_zelda_wendell_l751_751164


namespace min_tokens_in_grid_l751_751700

/-- Minimum number of tokens in a 99x99 grid so that each 4x4 subgrid contains at least 8 tokens -/
theorem min_tokens_in_grid : ∀ (grid : Matrix (Fin 99) (Fin 99) ℕ), (∀ i j : Fin 96, 8 ≤ ∑ r in FinSet.range 4, ∑ c in FinSet.range 4, grid (i + r) (j + c)) → ∑ i j, grid i j ≥ 4801 :=
by
  -- The proof goes here
  sorry

end min_tokens_in_grid_l751_751700


namespace domain_of_f_l751_751208

open Real
open Set

noncomputable def f (x : ℝ) : ℝ := log (sqrt (x - 1))

theorem domain_of_f :
  {x : ℝ | f x = log (sqrt (x - 1))}.dom = Ioi 1 := sorry

end domain_of_f_l751_751208


namespace cans_per_bag_l751_751438

theorem cans_per_bag (bags_saturday bags_sunday total_cans : ℕ) (h_sat : bags_saturday = 5) (h_sun : bags_sunday = 3) (h_total : total_cans = 40) :
  (total_cans / (bags_saturday + bags_sunday)) = 5 :=
by 
  rw [h_sat, h_sun, h_total]
  sorry

end cans_per_bag_l751_751438


namespace points_lie_on_parabola_l751_751840

-- Define the conditions for x and y in terms of t
def x (t : ℝ) : ℝ := 3^t - 5
def y (t : ℝ) : ℝ := 6^t - 4 * 3^t - 2

-- Define the type of the curve
def curve_type (f : ℝ → ℝ) : Type := f = λ x, x^2 + 6 * x + 17

-- State that the points (x(t), y(t)) lie on a parabola
theorem points_lie_on_parabola : 
  ∀ t : ℝ, ∃ x y : ℝ, (x = 3^t - 5) ∧ (y = 6^t - 4 * 3^t - 2) ∧ (curve_type (λ x, y) = curve_type (λ x, x^2 + 6 * x + 17)) :=
sorry

end points_lie_on_parabola_l751_751840


namespace half_angle_quadrant_l751_751903

-- Define the condition
def isSecondQuadrant (α : ℝ) (k : ℤ) : Prop := 
  (π / 2) + 2 * k * π < α ∧ α < π + 2 * k * π

-- Define the result we want to prove
def isFirstOrThirdQuadrant (β : ℝ) (k : ℤ) : Prop := 
  (π / 4) + k * π < β ∧ β < (π / 2) + k * π

-- The statement of the problem
theorem half_angle_quadrant (α : ℝ) (k : ℤ) (h : isSecondQuadrant α k) : 
  isFirstOrThirdQuadrant (α / 2) k :=
sorry

end half_angle_quadrant_l751_751903


namespace students_liking_both_l751_751093

theorem students_liking_both (total_students sports_enthusiasts music_enthusiasts neither : ℕ)
  (h1 : total_students = 55)
  (h2: sports_enthusiasts = 43)
  (h3: music_enthusiasts = 34)
  (h4: neither = 4) : 
  ∃ x, ((sports_enthusiasts - x) + x + (music_enthusiasts - x) = total_students - neither) ∧ (x = 22) :=
by
  sorry -- Proof omitted

end students_liking_both_l751_751093


namespace density_and_expectation_of_Z_l751_751969

noncomputable def uniform_density (a b : ℝ) (x : ℝ) : ℝ :=
  if a ≤ x ∧ x ≤ b then 1 / (b - a) else 0

def is_uniform_distribution (X : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, X x = uniform_density a b x

def is_independent (X Y : ℝ → ℝ) : Prop :=
  ∀ x y, (X x) * (Y y) = (X x) * (Y y)

def Z_density (X Y : ℝ → ℝ) :=
  λ z, if 0 ≤ z ∧ z ≤ 1 then 2 * (1 - z) else 0

def expectation (Z : ℝ → ℝ) : ℝ :=
  ∫ z in 0..1, z * (Z z)

theorem density_and_expectation_of_Z :
  ∀ (X Y : ℝ → ℝ), is_uniform_distribution X 0 1 → is_uniform_distribution Y 0 1 → 
  is_independent X Y →
  ∀ (Z : ℝ → ℝ), 
  (Z = λ z, if 0 ≤ z ∧ z ≤ 1 then 2 * (1 - z) else 0) →
  expectation Z = 1 / 3 :=
by
  sorry

end density_and_expectation_of_Z_l751_751969


namespace rectangle_area_increase_l751_751373

theorem rectangle_area_increase (x y : ℝ) : 
  let original_area := x * y,
      new_length := 1.2 * x,
      new_width := 1.1 * y,
      new_area := new_length * new_width,
      delta_area := new_area - original_area
  in (delta_area / original_area) * 100 = 32 := 
sorry

end rectangle_area_increase_l751_751373


namespace max_valid_sum_l751_751859

-- Let's define the problem in Lean
def is_valid_partition (numbers : list ℕ) (a b : ℕ) : Prop :=
  a ≤ 70 ∧ b ≤ 70 ∧ (a + b = list.sum numbers)

theorem max_valid_sum (numbers : list ℕ) :
  (∀ n ∈ numbers, 0 < n ∧ n ≤ 10) →
  (∃ a b, is_valid_partition numbers a b) →
  list.sum numbers ≤ 133 :=
begin
  intros h1 h2,
  sorry
end

end max_valid_sum_l751_751859


namespace smallest_number_from_1_to_40_largest_number_from_1_to_40_l751_751362

theorem smallest_number_from_1_to_40 (digits : List Nat) (n : Nat) :
  (digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) →
  (n = 11) →
  nat_to_largest_number digits n = 12333330 := sorry

theorem largest_number_from_1_to_40 (digits : List Nat) (n : Nat) :
  (digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) →
  (n = 11) →
  nat_to_largest_number digits n = 99967383940 := sorry

end smallest_number_from_1_to_40_largest_number_from_1_to_40_l751_751362


namespace wendy_distance_difference_l751_751251

-- Defining the distances ran and walked by Wendy
def distance_ran : ℝ := 19.83
def distance_walked : ℝ := 9.17

-- The theorem to prove the difference in distance
theorem wendy_distance_difference : distance_ran - distance_walked = 10.66 := by
  -- Proof goes here
  sorry

end wendy_distance_difference_l751_751251


namespace inequality_inequality_holds_l751_751315

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l751_751315


namespace perpendicular_lines_slope_eq_l751_751106

theorem perpendicular_lines_slope_eq (m : ℝ) :
  (∀ x y : ℝ, x - 2 * y + 5 = 0 → 
               2 * x + m * y - 6 = 0 → 
               (1 / 2) * (-2 / m) = -1) →
  m = 1 := 
by sorry

end perpendicular_lines_slope_eq_l751_751106


namespace charlie_pennies_l751_751076

variable (a c : ℕ)

theorem charlie_pennies (h1 : c + 1 = 4 * (a - 1)) (h2 : c - 1 = 3 * (a + 1)) : c = 31 := 
by
  sorry

end charlie_pennies_l751_751076


namespace sum_of_solutions_l751_751265

theorem sum_of_solutions :
  ∑ x in (finset.Icc 1 30).filter (λ x, 13 * (3 * x - 2) % 7 = 39 % 7),
  x = 66 := sorry

end sum_of_solutions_l751_751265


namespace find_angle_A_l751_751935

namespace Geometry

-- Definitions of the angles and lengths
variables {α : Type*} [LinearOrder α] (trapezoid ABCD : Type*) [is_trapezoid ABCD] (A B C D : Angle α) 
           (AB CD : Length α) 

-- Definition of the conditions
def trapezoid_conditions (AB CD : Length α) (A D C B : Angle α) :=
  parallel AB CD ∧ 
  A = 3 • D ∧ 
  C = 2 • B ∧ 
  AB = 2 ∧ 
  CD = 1

-- Prove that angle A is 135 degrees given the conditions
theorem find_angle_A {α : Type*} [LinearOrder α] 
  (trapezoid : Type*) [is_trapezoid trapezoid] (A B C D : Angle α) (AB CD : Length α)
  (H : trapezoid_conditions AB CD A D C B) : A = 135 := 
  sorry -- proof is not required

end Geometry

end find_angle_A_l751_751935


namespace problem_1_problem_2_problem_3_l751_751489

variables {x y m : ℝ}

-- Problem (1)
theorem problem_1 (h1 : 2 * x + y = 4 * m) (h2 : x + 2 * y = 2 * m + 1) (h3 : x + y = 1) :
  m = 1 / 3 :=
by {
  sorry
}

-- Problem (2)
theorem problem_2 (h1 : 2 * x + y = 4 * m) (h2 : x + 2 * y = 2 * m + 1) (h4 : -1 ≤ x - y ∧ x - y ≤ 5) :
  0 ≤ m ∧ m ≤ 3 :=
by {
  sorry
}

-- Problem (3)
theorem problem_3 (h1 : 2 * x + y = 4 * m) (h2 : x + 2 * y = 2 * m + 1) (h4 : -1 ≤ x - y ∧ x - y ≤ 5) :
  ∀ m, (0 ≤ m ∧ m ≤ 3) → 
    (0 ≤ m ∧ m ≤ 3/2 → |m + 2| + |2 * m - 3| = 5 - m) ∧
    (3/2 < m ∧ m ≤ 3 → |m + 2| + |2 * m - 3| = 3 * m - 1) :=
by {
  sorry
}

end problem_1_problem_2_problem_3_l751_751489


namespace smallest_nineteen_multiple_l751_751264

theorem smallest_nineteen_multiple (n : ℕ) 
  (h₁ : 19 * n ≡ 5678 [MOD 11]) : n = 8 :=
by sorry

end smallest_nineteen_multiple_l751_751264


namespace min_colors_tessellation_l751_751696

/-- 
Given a tessellation consisting of alternating squares and equilateral triangles,
where no two adjacent tiles (tiles sharing a side) are the same color,
prove that the minimum number of colors needed to shade the tessellation is 3.
--/
theorem min_colors_tessellation (T : set (set ℝ)) (H1 : ∀ t1 t2 ∈ T, (∃ s ∈ T, (s ≠ t1 ∧ s ≠ t2 ∧ s.adjacent t1 ∧ s.adjacent t2))) :
  ∃ (num_colors : ℕ), num_colors = 3 := sorry

end min_colors_tessellation_l751_751696


namespace geometric_sequence_sum_l751_751661

theorem geometric_sequence_sum :
  let a_1 := 1
  let r := 3
  let a (n : ℕ) := a_1 * r^(n-1)
  (a 4 + a 5 + a 6 + a 7) = 1080 :=
by
  let a_1 := 1
  let r := 3
  let a (n : ℕ) := a_1 * r^(n-1)
  calc
    a 4 + a 5 + a 6 + a 7 = 1 * 3^3 + 1 * 3^4 + 1 * 3^5 + 1 * 3^6 : by simp [a]
                      ... = 27 + 81 + 243 + 729 : by norm_num
                      ... = 1080 : by norm_num

end geometric_sequence_sum_l751_751661


namespace chord_length_l751_751524

theorem chord_length (r : ℝ) (h : r = 15) :
  ∃ (cd : ℝ), cd = 13 * Real.sqrt 3 :=
by
  sorry

end chord_length_l751_751524


namespace hyperbola_eccentricity_l751_751446

-- Define the hyperbola and its conditions
-- Given a hyperbola \(\Gamma : \frac{y^2}{a^2} - \frac{x^2}{b^2} = 1 (a, b > 0)\)
-- with foci \(F_1\) and \(F_2\).
variable {a b : ℝ} (ha : 0 < a) (hb : 0 < b)

def hyperbola_eq (x y : ℝ) := (y^2 / a^2) - (x^2 / b^2) = 1

-- Definition of foci \(F_1\) and \(F_2\)
def F1 := (-sqrt (a^2 + b^2), 0)
def F2 := (sqrt (a^2 + b^2), 0)

-- Line passes through \(F_1\) perpendicular to the real axis intersects the lower branch at points \(M\) and \(N\)
-- \(S\) and \(T\) are the intersection points with the imaginary axis of lines \(M{F_2}\) and \(N{F_2}\) respectively
-- The perimeter of \(\Delta S{F_2}T\) is 20
variable (perimeter : ℝ) (hperimeter : perimeter = 20)

-- Prove the eccentricity \(e = \frac{2\sqrt{3}}{3}\) when \(ab\) is at its maximum value
theorem hyperbola_eccentricity (h_max_ab : ab = max_ab_value) : 
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = 2 * sqrt 3 / 3 :=
sorry

end hyperbola_eccentricity_l751_751446


namespace different_diagonal_lengths_l751_751250

theorem different_diagonal_lengths (m n : ℕ) : 
  m > 3 * 2^n → 
  ∃ (diagonals : finset ℕ), 
  (∀ d ∈ diagonals, d > 0 ∧ d < m / 2) ∧ 
  diagonals.card ≥ n + 1 :=
by
  sorry

end different_diagonal_lengths_l751_751250


namespace derivative_poly_derivative_exp_poly_l751_751821

-- Problem 1: Derivative of y = (2x^2 + 3)(3x - 1)
theorem derivative_poly:
  ∀ (x : ℝ), deriv (λ x, (2*x^2 + 3)*(3*x - 1)) x = 18*x^2 - 4*x + 9 :=
by
  intro x
  sorry

-- Problem 2: Derivative of y = xe^x + 2x + 1
theorem derivative_exp_poly:
  ∀ (x : ℝ), deriv (λ x, x*exp(x) + 2*x + 1) x = exp(x) + x*exp(x) + 2 :=
by
  intro x
  sorry

end derivative_poly_derivative_exp_poly_l751_751821


namespace f_is_odd_for_n_gt_1_l751_751649

def f : ℕ → ℤ
| 1 := 2
| 2 := 7
| (n + 1) := 
  let f_n : ℤ := f n
  let f_n_1 : ℤ := f (n-1)
  f_n * f_n / f_n_1 + sorry -- placeholder to satisfy the conditions

theorem f_is_odd_for_n_gt_1 
    (h1 : f 1 = 2) 
    (h2 : f 2 = 7) 
    (h3 : ∀ n ≥ 2, -1/2 < f (n + 1) - f n * f n / f (n - 1) ∧ f (n + 1) - f n * f n / f (n - 1) ≤ 1/2) : 
    ∀ n > 1, ∃ k : ℤ, f n = 2 * k + 1 :=
by
-- Proof by induction or other methods would go here
sorry

end f_is_odd_for_n_gt_1_l751_751649


namespace definite_integral_cos_pi_div_2_l751_751809

theorem definite_integral_cos_pi_div_2 : (∫ x in 0..(Real.pi / 2), Real.cos x) = 1 := 
by 
  -- proof goes here
  sorry

end definite_integral_cos_pi_div_2_l751_751809


namespace hot_dog_cost_l751_751366

variable {Real : Type} [LinearOrderedField Real]

-- Define the cost of a hamburger and a hot dog
variables (h d : Real)

-- Arthur's buying conditions
def condition1 := 3 * h + 4 * d = 10
def condition2 := 2 * h + 3 * d = 7

-- Problem statement: Proving that the cost of a hot dog is 1 dollar
theorem hot_dog_cost
    (h d : Real)
    (hc1 : condition1 h d)
    (hc2 : condition2 h d) : 
    d = 1 :=
sorry

end hot_dog_cost_l751_751366


namespace tan_alpha_mul_tan_beta_l751_751070

-- We state our variables
variables (α β : ℝ)

-- We state our conditions as assumptions
axiom cos_sum : cos (α + β) = 3 / 5
axiom cos_diff : cos (α - β) = 4 / 5

-- We state the theorem which needs to be proven
theorem tan_alpha_mul_tan_beta : (tan α) * (tan β) = 1 / 7 :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end tan_alpha_mul_tan_beta_l751_751070


namespace sin_arithmetic_sequence_l751_751411

theorem sin_arithmetic_sequence (a : ℝ) (h1 : 0 < a) (h2 : a < 360) 
  (h3 : sin a + sin (3 * a) = 2 * sin (2 * a)) : 
  a = 45 ∨ a = 180 ∨ a = 315 :=
by
  sorry

end sin_arithmetic_sequence_l751_751411


namespace equation_of_ellipse_equation_of_line_l751_751030

-- Define the given conditions
def pointA : ℝ × ℝ := (0, -2)

def ellipseE (a b : ℝ) (a_gt_b : a > b) : Prop :=
  ∃ c : ℝ, c = sqrt 3 ∧ ∃ e : ℝ, e = sqrt 3 / 2 ∧ 
    a = 2 ∧ b = 1 ∧ 
    ((x y : ℝ) -> (x^2)/(a^2) + (y^2)/(b^2) = 1)

def slope_AF (AF_slope : ℝ) : Prop := 
  AF_slope = 2*sqrt 3 / 3

axiom origin : (ℝ × ℝ) := (0, 0)

-- Proving the equation of the ellipse
theorem equation_of_ellipse : 
  ∀ (A : ℝ × ℝ) (a b : ℝ) (a_gt_b : a > b) (c e : ℝ),
    A = pointA →
    ellipseE a b a_gt_b →
    e = sqrt 3 / 2 →
    slope_AF (2*sqrt 3 / 3) →
    ∃ (x y : ℝ), (x^2)/4 + y^2 = 1 :=
  sorry

-- Proving the condition for the maximal area of triangle POQ
theorem equation_of_line : 
  ∀ (A : ℝ × ℝ) (a b : ℝ) (a_gt_b : a > b) (c e : ℝ),
    A = pointA →
    ellipseE a b a_gt_b →
    e = sqrt 3 / 2 →
    slope_AF (2*sqrt 3 / 3) →
    ∃ k : ℝ, k = sqrt 7 / 2 ∨ k = -sqrt 7 / 2 →
    ∃ (y x : ℝ), y = (sqrt 7 / 2) * x - 2 :=
  sorry

end equation_of_ellipse_equation_of_line_l751_751030


namespace solve_for_x_l751_751190

theorem solve_for_x : 
  ∀ (x : ℝ), (1 / 9) ^ (3 * x + 10) = 27 ^ (x + 4) → x = -32 / 9 :=
by
  intro x
  assume h
  -- The detailed proof goes here
  sorry

end solve_for_x_l751_751190


namespace students_speaking_both_languages_l751_751344

theorem students_speaking_both_languages
  (total_students : ℕ)
  (percentage_not_speaking_french : ℕ)
  (students_speaking_french_not_english : ℕ)
  (h1 : total_students = 200)
  (h2 : percentage_not_speaking_french = 55)
  (h3 : students_speaking_french_not_english = 65) :
  ∃ students_speaking_both_french_and_english : ℕ,
    students_speaking_both_french_and_english = total_students * (100 - percentage_not_speaking_french) / 100 - students_speaking_french_not_english := 
by
  have h4 : total_students * (100 - percentage_not_speaking_french) / 100 = 90 := by sorry
  have h5 : 90 - students_speaking_french_not_english = 25 := by sorry
  use 25
  exact h5

end students_speaking_both_languages_l751_751344


namespace min_value_sum_pos_int_l751_751464

theorem min_value_sum_pos_int 
  (a b c : ℕ)
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots: ∃ (A B : ℝ), A < 0 ∧ A > -1 ∧ B > 0 ∧ B < 1 ∧ (∀ x : ℝ, x^2*x*a + x*b + c = 0 → x = A ∨ x = B))
  : a + b + c = 11 :=
sorry

end min_value_sum_pos_int_l751_751464


namespace intersection_A_complement_B_l751_751729

def set_A : Set ℝ := {x | 1 < x ∧ x < 4}
def set_B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_Complement_B : Set ℝ := {x | x < -1 ∨ x > 3}
def set_Intersection : Set ℝ := {x | set_A x ∧ set_Complement_B x}

theorem intersection_A_complement_B : set_Intersection = {x | 3 < x ∧ x < 4} := by
  sorry

end intersection_A_complement_B_l751_751729


namespace eval_line_integral_triangle_l751_751688

noncomputable def P (x y : ℝ) : ℝ := 2 * (x^2 + y^2)
noncomputable def Q (x y : ℝ) : ℝ := (x + y)^2

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (1, 3)

theorem eval_line_integral_triangle :
  ∮ (λ x y => P x y) dx + (λ x y => Q x y) dy = 4 / 3 :=
by
  sorry

end eval_line_integral_triangle_l751_751688


namespace sum_abc_eq_five_l751_751118

variables {R : Type*} [OrderedField R] (a b c : R)

theorem sum_abc_eq_five (h₁ : a^2 * b + a^2 * c + b^2 * a + b^2 * c + c^2 * a + c^2 * b + 3 * a * b * c = 30)
                        (h₂ : a^2 + b^2 + c^2 = 13) :
    a + b + c = 5 :=
by
  sorry

end sum_abc_eq_five_l751_751118


namespace find_f_11_l751_751964

noncomputable def f (u : ℝ) : ℝ := by
  let x := (u + 1) -- Placeholder to define using quadratic solution, sorry to be used

theorem find_f_11 (h : ∀ x : ℝ, f (2 * x^2 + 3 * x - 1) = x^2 + x + 1) : f 11 = 129 / 16 := by
  sorry

end find_f_11_l751_751964


namespace greatest_distance_between_centers_l751_751238

def rectangle_width := 15
def rectangle_height := 16
def circle_diameter := 7
def circle_radius := circle_diameter / 2
def distance_between_centres :=
  let x_distance := rectangle_width - 2 * circle_radius
  let y_distance := rectangle_height - 2 * circle_radius
  Real.sqrt (x_distance^2 + y_distance^2)

theorem greatest_distance_between_centers :
  distance_between_centres = Real.sqrt 145 :=
by
  unfold distance_between_centres
  have x_dist : (rectangle_width - 2 * circle_radius) = 8 := by sorry
  have y_dist : (rectangle_height - 2 * circle_radius) = 9 := by sorry
  rw [x_dist, y_dist]
  norm_num
  sorry

end greatest_distance_between_centers_l751_751238


namespace reflection_matrix_values_l751_751400

theorem reflection_matrix_values (a b : ℝ) (I : Matrix (Fin 2) (Fin 2) ℝ) :
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![![a, 9/26], ![b, 17/26]]
  (R * R = I) → a = -17/26 ∧ b = 0 :=
by
  sorry

end reflection_matrix_values_l751_751400


namespace angle_A_is_equilateral_l751_751919

namespace TriangleProof

variables {A B C : ℝ} {a b c : ℝ}

-- Given condition (a+b+c)(a-b-c) + 3bc = 0
def condition1 (a b c : ℝ) : Prop := (a + b + c) * (a - b - c) + 3 * b * c = 0

-- Given condition a = 2c * cos B
def condition2 (a c B : ℝ) : Prop := a = 2 * c * Real.cos B

-- Prove that if (a+b+c)(a-b-c) + 3bc = 0, then A = π / 3
theorem angle_A (h1 : condition1 a b c) : A = Real.pi / 3 :=
sorry

-- Prove that if a = 2c * cos B and A = π / 3, then ∆ ABC is an equilateral triangle
theorem is_equilateral (h2 : condition2 a c B) (hA : A = Real.pi / 3) : 
  b = c ∧ a = b ∧ B = C :=
sorry

end TriangleProof

end angle_A_is_equilateral_l751_751919


namespace largest_possible_degree_p_for_horizontal_asymptote_l751_751790

   theorem largest_possible_degree_p_for_horizontal_asymptote :
     ∀ {p q : Polynomial ℝ},
       q = 3 * Polynomial.monomial 6 1 - 2 * Polynomial.monomial 3 1 + Polynomial.monomial 1 1 - Polynomial.C 4 →
       ∃ n : ℕ, Polynomial.degree p ≤ Polynomial.degree q ∧ Polynomial.degree q = 6 ∧ n = 6 :=
   by
     intros p q hq
     have hq_deg : Polynomial.degree q = 6 := by
       simp [hq]
     exact ⟨6, le_rfl, hq_deg, rfl⟩
   
end largest_possible_degree_p_for_horizontal_asymptote_l751_751790


namespace sum_of_coefficients_is_11_l751_751956

noncomputable def polynomial_has_roots (u p q r : ℝ) : Prop :=
  ∃ (f : ℂ → ℂ), f = λ z, z ^ 3 + p * z ^ 2 + q * z + r ∧
  f(u + 2 * complex.I) = 0 ∧ f(u - 2 * complex.I) = 0 ∧ f(2 * u + 5) = 0

theorem sum_of_coefficients_is_11 (u p q r : ℝ) (h_roots : polynomial_has_roots u p q r) : p + q + r = 11 :=
sorry

end sum_of_coefficients_is_11_l751_751956


namespace center_of_mass_equal_l751_751618

variables {n m : ℕ}
variables (X : Fin n → EuclideanSpace ℝ (n + m)) (Y : Fin m → EuclideanSpace ℝ (n + m))
variables (a : Fin n → ℝ) (b : Fin m → ℝ)

noncomputable def center_mass (points : Fin n → EuclideanSpace ℝ (n + m)) (masses : Fin n → ℝ) : EuclideanSpace ℝ (n + m) :=
  (∑ i, masses i • points i) / (∑ i, masses i)

noncomputable def system_center_mass : EuclideanSpace ℝ (n + m) :=
  let mass_X := ∑ i, a i
  let mass_Y := ∑ j, b j
  let Xc := center_mass X a
  let Yc := center_mass Y b 
  (mass_X • Xc + mass_Y • Yc) / (mass_X + mass_Y)

theorem center_of_mass_equal (X : Fin n → EuclideanSpace ℝ (n + m)) (Y : Fin m → EuclideanSpace ℝ (n + m))
                            (a : Fin n → ℝ) (b : Fin m → ℝ) :
  system_center_mass X Y a b = (∑ i, a i • X i + ∑ j, b j • Y j) / (∑ i, a i + ∑ j, b j) :=
sorry

end center_of_mass_equal_l751_751618


namespace Julie_downstream_distance_l751_751127

/-- Julie rows 32 km upstream and a certain distance downstream taking 4 hours each.
    The speed of the stream is 0.5 km/h.
    We need to prove that the distance Julie rowed downstream is 36 km.
-/
theorem Julie_downstream_distance :
  ∃ (V : ℝ), 
  (32 = (V - 0.5) * 4) ∧ 
  (∀ D, D = (V + 0.5) * 4 → D = 36) :=
begin
  sorry
end

end Julie_downstream_distance_l751_751127


namespace distance_from_F_to_midpoint_l751_751099

-- Definitions of the given problem
variable (DE DF EF : ℝ)
variable (F : {x : ℝ × ℝ // x.1 ^ 2 + x.2 ^ 2 = DF ^ 2})
def is_right_triangle (DE DF EF : ℝ) : Prop := (DE ^ 2 + DF ^ 2 = EF ^ 2)

theorem distance_from_F_to_midpoint (h : is_right_triangle DE DF EF) (hDE : DE = 15) (hDF : DF = 9) (hEF : EF = 12) :
  (DE / 2 - sqrt (((DF ^ 2) / 2)^2 - (DF / 2) ^ 2) = 6) :=
by
  sorry

end distance_from_F_to_midpoint_l751_751099


namespace unique_ordered_pairs_l751_751187

theorem unique_ordered_pairs {n : ℕ} (h_n : n = 6) (f m : ℕ) :
  ∃ (pairs : finset (ℕ × ℕ)), pairs = {(0, 6), (2, 6), (4, 6), (6, 6), (6, 0)} ∧ pairs.card = 5 :=
by
  use {(0, 6), (2, 6), (4, 6), (6, 6), (6, 0)}
  split
  . refl
  . exact finset.card_finset_of_eq (dec_trivial : 5 = 5)
  done

end unique_ordered_pairs_l751_751187


namespace mary_shirts_left_l751_751167

theorem mary_shirts_left :
  ∀ (initial_blue : ℕ) (initial_brown : ℕ),
  initial_blue = 26 →
  initial_brown = 36 →
  let given_away_blue := initial_blue / 2
  let given_away_brown := initial_brown / 3
  let left_blue := initial_blue - given_away_blue
  let left_brown := initial_brown - given_away_brown
  left_blue + left_brown = 37 :=
by
  intros initial_blue initial_brown h_initial_blue h_initial_brown
  dsimp
  rw [h_initial_blue, h_initial_brown]
  sorry

end mary_shirts_left_l751_751167


namespace sine_of_central_angle_EQ_l751_751092

-- Definitions
def circle_radius : Real := 8
def PX : Real := 9
def RX : Real := 5

-- Question (The problem to prove)
theorem sine_of_central_angle_EQ {
  P Q R S X : Type -- Points on Circle 
  (circle : ∀ x, dist circle_radius x = 8) -- Radius constraint
  (chord_PQ_RS : intersection_chord P Q R S X) -- Intersecting chords at X
  (PX_equals_9 : dist P X = 9)
  (RX_equals_5 : dist R X = 5)
  (QS_bisects_RS : bisects QS RS X) -- QS bisects RS at X
} : 
  sine (angle P O R) = 16 / 25 :=
sorry

end sine_of_central_angle_EQ_l751_751092


namespace find_S_l751_751662

noncomputable def S : ℕ+ → ℝ := sorry
noncomputable def a : ℕ+ → ℝ := sorry

axiom h : ∀ n : ℕ+, 2 * S n = 3 * a n + 4

theorem find_S : ∀ n : ℕ+, S n = 2 - 2 * 3 ^ (n : ℕ) :=
  sorry

end find_S_l751_751662


namespace find_other_root_l751_751845

theorem find_other_root (m : ℝ) (α : ℝ) :
  (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C m * Polynomial.X - Polynomial.C 10 = 0) →
  (α = -5) →
  ∃ β : ℝ, (α + β = -m) ∧ (α * β = -10) :=
by 
  sorry

end find_other_root_l751_751845


namespace volume_of_frustum_l751_751760

theorem volume_of_frustum (base_edge : ℝ) (altitude : ℝ) 
    (altitude_ratio : ℝ) (volume : ℝ) (h_base_edge : base_edge = 36) 
    (h_altitude : altitude = 2 * 12) -- altitude is given in feet but converted to inches since other dimensions may be in inches
    (h_altitude_ratio : altitude_ratio = 1 / 3)
    (h_volume_ori : volume = (1/27) * volume * (1 / altitude_ratio)^3 )
    (vol_frustum : ℝ) :
    vol_frustum = volume * (1 - (1 / 27)) :=
begin
  sorry
end

end volume_of_frustum_l751_751760


namespace range_of_f_l751_751701

def f (x : ℝ) : ℝ := (x^2 + 4 * x + 3) / (x + 2)

theorem range_of_f : set.range (λ x : {x : ℝ // x ≠ -2}, f x.val) = {y : ℝ | y ≠ -1} :=
by sorry

end range_of_f_l751_751701


namespace collinear_CAD_l751_751175

variable {Point : Type} [PlaneGeometry Point] 
variables (A B C D : Point)

-- Definitions for the conditions
def triangle_ABC (A B C : Point) : Prop := 
  triangle A B C

def point_D_on_BC (B C D : Point) : Prop :=
  collinear B C D ∧ D ≠ B ∧ D ≠ C

def bisectors_intersect_on_AB (A B C D : Point) : Prop :=
  ∃ E, (angle_bisector A C B intersects on (line_segment A B) E) ∧ 
       (angle_bisector A D B intersects on (line_segment A B) E)

def reflection_across_AB (A B D : Point) : Point :=
  reflect D (line_segment A B)

-- Statement of the proof problem
theorem collinear_CAD' (A B C D : Point) (h1 : triangle_ABC A B C)
  (h2 : point_D_on_BC B C D) (h3 : bisectors_intersect_on_AB A B C D) :
  let D' := reflection_across_AB A B D in 
  collinear C A D' :=
sorry

end collinear_CAD_l751_751175


namespace common_difference_gt_30000_l751_751333

open Nat

/-- 
Define the problem: if 15 prime numbers form an increasing arithmetic progression, 
prove that the common difference is greater than 30000.
-/
theorem common_difference_gt_30000 
  (a : ℕ → ℕ) 
  (h_arith : ∀ n m : ℕ, n < m → a m = a n + (m - n) * d) 
  (h_prime_seq : ∀ i : ℕ, 1 ≤ i → i ≤ 15 → Prime (a i)) 
  (h_increasing : ∀ n m : ℕ, n < m → a n < a m) 
  (h_natural : ∀ i : ℕ, 1 ≤ i → i ≤ 15 → 0 < a i) 
  (d : ℕ) :
  d > 30000 :=
sorry

end common_difference_gt_30000_l751_751333


namespace find_value_of_expression_l751_751864

noncomputable def log2 (x : ℝ) : ℝ := real.log x / real.log 2

theorem find_value_of_expression :
  (∀ (x : ℝ), 6 * 2^x = 256 → (x + 2) * (x - 2) = 60 - 16 * log2 6 + (log2 6)^2) :=
begin
  sorry,
end

end find_value_of_expression_l751_751864


namespace obtuse_triangle_range_a_l751_751467

noncomputable def is_obtuse_triangle (a b c : ℝ) : Prop :=
  ∃ (θ : ℝ), θ > 90 ∧ θ ≤ 120 ∧ c^2 > a^2 + b^2

theorem obtuse_triangle_range_a (a : ℝ) :
  (a + (a + 1) > a + 2) →
  is_obtuse_triangle a (a + 1) (a + 2) →
  (1.5 ≤ a ∧ a < 3) :=
by
  sorry

end obtuse_triangle_range_a_l751_751467


namespace line_equation_through_point_l751_751422

theorem line_equation_through_point (A : ℝ × ℝ) (m : ℝ) (y_intercept : ℝ) :
  A = (real.sqrt 3, 1) → 
  m = real.tan (real.pi / 3) → 
  (∀ x y, y - A.2 = m * (x - A.1) ↔ √3 * x - y - 2 = 0) := 
begin 
  intros hA hm,
  simp [hA, hm],
  sorry
end

end line_equation_through_point_l751_751422


namespace perimeter_square_D_l751_751195

-- Declare constants for the conditions
constant perimeter_C : ℝ
constant area_D : ℝ

-- Set the given conditions
axiom perimeter_C_eq : perimeter_C = 16
axiom area_D_eq : area_D = (1 / 3) * (perimeter_C / 4)^2

-- The goal is to prove the perimeter of square D
theorem perimeter_square_D : 4 * (sqrt area_D) = 4 * (4 * sqrt 3 / 3) :=
by
  -- Notice that the Lean environment has already been set up to prove the specific result, according to equivalency check.
  rw [perimeter_C_eq, area_D_eq]
  sorry

end perimeter_square_D_l751_751195


namespace circumcircle_eq_l751_751857

structure Point2D (α : Type _) :=
  (x y : α)

def O : Point2D ℝ := { x := 0, y := 0 }
def A : Point2D ℝ := { x := 2, y := 4 }
def B : Point2D ℝ := { x := 6, y := 2 }

noncomputable def equation_of_circumcircle (D E F : ℝ) : Prop :=
  ∀ (P : Point2D ℝ), (P = O ∨ P = A ∨ P = B) → (P.x^2 + P.y^2 + D * P.x + E * P.y + F = 0)

theorem circumcircle_eq :
  equation_of_circumcircle (-6) (-2) 0 :=
by
  intros P hP
  cases hP
  { -- Case P = O
    rw hP
    norm_num }
  { cases hP
    { -- Case P = A
      rw hP
      norm_num }
    { -- Case P = B
      rw hP
      norm_num } }

end circumcircle_eq_l751_751857


namespace greatest_integer_m_l751_751429

-- Definition for h, greatest power of 3 that divides y
def h (y : ℕ) : ℕ :=
  if y % 3 = 0 then 3 * h (y / 3) else 1

-- Definition for T_m
def T_m (m : ℕ) : ℕ :=
  ∑ k in finset.range (3 ^ (m - 1)), h (3 * (k + 1) - 1)

-- Lean theorem statement (without proof)
theorem greatest_integer_m (m : ℕ) (h_m : m < 500) :
  ∃ m, m < 500 ∧ ∃ k, T_m m = k ^ 3 ∧ m = 440 :=
begin
  sorry
end

end greatest_integer_m_l751_751429


namespace inequality_holds_l751_751303

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l751_751303


namespace selection_condition_1_selection_condition_2_selection_condition_3_selection_condition_4_l751_751229

theorem selection_condition_1 (boys girls : ℕ) (conditions : boys = 5 ∧ girls = 3) : 
  (choose girls 1 * choose boys 4 * fact 5) + (choose girls 2 * choose boys 3 * fact 5) = (binom 3 1 * binom 5 4 * 5! + binom 3 2 * binom 5 3 * 5!) :=
by sorry

theorem selection_condition_2 (remaining_people : ℕ) (conditions : remaining_people = 7) : 
  choose remaining_people 4 * fact 4 = (binom 7 4 * 4!) :=
by sorry

theorem selection_condition_3 (remaining_people : ℕ) (conditions : remaining_people = 7) : 
  choose remaining_people 4 * fact 4 * 4 = (binom 7 4 * 4! * 4) :=
by sorry

theorem selection_condition_4 (remaining_people : ℕ) (conditions : remaining_people = 6) : 
  choose remaining_people 3 * fact 3 * 4 = (binom 6 3 * 3! * 4) :=
by sorry

end selection_condition_1_selection_condition_2_selection_condition_3_selection_condition_4_l751_751229


namespace sum_of_digits_in_rectangle_l751_751994

theorem sum_of_digits_in_rectangle :
  ∃ A B C : ℕ,
    (4 + A + 1 + B = 12) ∧
    (4 + A + 1 + B = 6 + 6) ∧
    (C + 1 + 6 + C = 11) ∧
    (1 + B + 2 + C = 9) ∧
    (A + 8 + 8 = 8) ∧
    (A + 8 + B + 2 + C = 13) ∧
    (9 = 4 + A + 1 + B) ∧
    (B + 2 + C = 9) ∧    
    B = 5 ∧ A = 2 ∧ C = 6 :=
sorry

end sum_of_digits_in_rectangle_l751_751994


namespace even_number_of_convenient_numbers_l751_751163

def is_convenient (n : ℕ) : Prop :=
  (n ^ 2 + 1) % 1000001 = 0

theorem even_number_of_convenient_numbers :
  (Finset.filter is_convenient (Finset.range 1000000.succ)).card % 2 = 0 :=
by
  sorry

end even_number_of_convenient_numbers_l751_751163


namespace more_not_rep_sum_square_cube_l751_751775

theorem more_not_rep_sum_square_cube (N : ℕ) (hN : N = 1000000) :
  ∃ (M : ℕ), M > (N - M) ∧ (∀ n, n ≤ N → (∃ k m : ℕ, n = k^2 + m^3) → false) :=
begin
  sorry

end more_not_rep_sum_square_cube_l751_751775


namespace F_of_1_6_and_2_99_l751_751987

-- Define the sequence function F(m)
def F (m : ℕ) : ℚ :=
  -- sequence logic (to be defined based on the description)

-- Now specify the theorem to be proved
theorem F_of_1_6_and_2_99 :
  F 16 = 1 / 6 ∧ F 4952 = 2 / 99 :=
by
  sorry

end F_of_1_6_and_2_99_l751_751987


namespace shared_property_diagonals_bisect_l751_751223

-- Define the properties as propositions
def DiagonalsBisect (Q : Type) := Q → Prop
def DiagonalsPerpendicular (Q : Type) := Q → Prop
def DiagonalsEqualLength (Q : Type) := Q → Prop

-- Define the four types of quadrilaterals
def Parallelogram (Q : Type) := Q → Prop
def Rectangle (Q : Type) := Q → Prop
def Rhombus (Q : Type) := Q → Prop
def Square (Q : Type) := Q → Prop

-- Assume the properties for each type of quadrilateral
axiom parallelogram_diagonals_bisect (Q : Type) [Parallelogram Q] :
  DiagonalsBisect Q

axiom rectangle_diagonals_bisect (Q : Type) [Rectangle Q] :
  DiagonalsBisect Q

axiom rhombus_diagonals_bisect (Q : Type) [Rhombus Q] :
  DiagonalsBisect Q

axiom square_diagonals_bisect (Q : Type) [Square Q] :
  DiagonalsBisect Q

-- Main theorem to prove
theorem shared_property_diagonals_bisect (Q : Type) [Parallelogram Q] [Rectangle Q] [Rhombus Q] [Square Q] :
  DiagonalsBisect Q := 
by {
  apply parallelogram_diagonals_bisect,
  sorry
}

end shared_property_diagonals_bisect_l751_751223


namespace infinite_n_dividing_a_pow_n_plus_1_l751_751838

theorem infinite_n_dividing_a_pow_n_plus_1 (a : ℕ) (h1 : 1 < a) (h2 : a % 2 = 0) :
  ∃ (S : Set ℕ), S.Infinite ∧ ∀ n ∈ S, n ∣ a^n + 1 := 
sorry

end infinite_n_dividing_a_pow_n_plus_1_l751_751838


namespace aaron_and_carson_scoops_l751_751768

def initial_savings (a c : ℕ) : Prop :=
  a = 150 ∧ c = 150

def total_savings (t a c : ℕ) : Prop :=
  t = a + c

def restaurant_expense (r t : ℕ) : Prop :=
  r = 3 * t / 4

def service_charge_inclusive (r sc : ℕ) : Prop :=
  r = sc * 115 / 100

def remaining_money (t r rm : ℕ) : Prop :=
  rm = t - r

def money_left (al cl : ℕ) : Prop :=
  al = 4 ∧ cl = 4

def ice_cream_scoop_cost (s : ℕ) : Prop :=
  s = 4

def total_scoops (rm ml s scoop_total : ℕ) : Prop :=
  scoop_total = (rm - (ml - 4 - 4)) / s

theorem aaron_and_carson_scoops :
  ∃ a c t r sc rm al cl s scoop_total, initial_savings a c ∧
  total_savings t a c ∧
  restaurant_expense r t ∧
  service_charge_inclusive r sc ∧
  remaining_money t r rm ∧
  money_left al cl ∧
  ice_cream_scoop_cost s ∧
  total_scoops rm (al + cl) s scoop_total ∧
  scoop_total = 16 :=
sorry

end aaron_and_carson_scoops_l751_751768


namespace marcus_total_earnings_l751_751934

-- Defining the conditions given in the problem
variables (h₃ h₂ : ℝ) (e₃ e₂ : ℝ) (w : ℝ)
variable (earn_mul : ℝ → ℝ → ℝ )

-- Given conditions specific to problem
def conditions := 
  h₃ = 18 ∧
  e₃ = e₂ + 36 ∧
  h₂ = 12 ∧
  ∀ (h₃ h₂ : ℕ), w = w

-- Translate the problem to a Lean theorem
theorem marcus_total_earnings : 
  conditions h₃ h₂ e₃ e₂ w earn_mul → 
  earn_mul (h₃ + h₂) w = 180 :=
by
  sorry

end marcus_total_earnings_l751_751934


namespace opposite_of_neg3_squared_l751_751219

theorem opposite_of_neg3_squared : -(-3^2) = 9 :=
by
  sorry

end opposite_of_neg3_squared_l751_751219


namespace number_of_ways_2020_l751_751134

-- We are defining b_i explicitly restricted by the conditions in the problem.
def b (i : ℕ) : ℕ :=
  sorry

-- Given conditions
axiom h_bounds : ∀ i, 0 ≤ b i ∧ b i ≤ 99
axiom h_indices : ∀ (i : ℕ), i < 4

-- Main theorem statement
theorem number_of_ways_2020 (M : ℕ) 
  (h : 2020 = b 3 * 1000 + b 2 * 100 + b 1 * 10 + b 0) 
  (htotal : M = 203) : 
  M = 203 :=
  by 
    sorry

end number_of_ways_2020_l751_751134


namespace area_of_triangle_l751_751937

namespace TriangleArea

def triangle : Type := sorry  -- Define a triangle type

variables {XYZ : triangle}
variables (XY XZ XM : ℝ)
variables (area : ℝ)

-- Defining conditions
def sideXY : XY = 9 := sorry
def sideXZ : XZ = 17 := sorry
def medianXM : XM = 12 := sorry

-- Theorem statement: Given the conditions, the area of triangle XYZ is 40√2.
theorem area_of_triangle (h1 : sideXY) (h2 : sideXZ) (h3 : medianXM) : 
  area = 40 * real.sqrt 2 := 
sorry

end TriangleArea

end area_of_triangle_l751_751937


namespace A_work_days_l751_751733

theorem A_work_days (x : ℕ) : (∀ (A B : ℕ), (B = 28) → (∀ t : ℕ, t = 3 → ∀ t' : ℕ, t' = 21 → 
  (3 * (1/x + 1/28) + 21 * (1/28) = 1) → x = 21)) :=
begin
  sorry
end

end A_work_days_l751_751733


namespace terminating_decimal_count_l751_751009

theorem terminating_decimal_count : ∃ n, n = 23 ∧ (∀ k, 1 ≤ k ∧ k ≤ 499 → (∃ m, k = 21 * m)) :=
by
  sorry

end terminating_decimal_count_l751_751009


namespace exists_real_solution_for_any_y_l751_751414

noncomputable def quadraticDiscriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem exists_real_solution_for_any_y {x : ℝ} :
  (∀ (y : ℝ), ∃ (z : ℝ), x^2 + y^2 + z^2 + 2 * x * y * z = 1) ↔ (x = 1 ∨ x = -1) := 
begin
  sorry
end

end exists_real_solution_for_any_y_l751_751414


namespace opposite_directions_l751_751491

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem opposite_directions (a b : V) (h : a + 4 • b = 0) : a = -4 • b := sorry

end opposite_directions_l751_751491


namespace lambda_values_l751_751072

noncomputable def vector_a (λ : ℝ) : EuclideanSpace ℝ (Fin 3) := ![1, λ, 2]
noncomputable def vector_b : EuclideanSpace ℝ (Fin 3) := ![2, -1, 2]

theorem lambda_values (λ : ℝ) (h_cos : real.cos_angle (vector_a λ) vector_b = 8 / 9) :
  λ = -2 ∨ λ = 2 / 55 :=
sorry

end lambda_values_l751_751072


namespace sum_of_possible_values_of_A_l751_751745

theorem sum_of_possible_values_of_A:
  let possible_values := {A | A ∈ Finset.range 10 ∧ (41 + A) % 9 = 0} in
  possible_values.sum (λ x, x) = 4 :=
by
  sorry

end sum_of_possible_values_of_A_l751_751745


namespace product_of_roots_eq_neg7_l751_751390

open Polynomial

theorem product_of_roots_eq_neg7 :
  let p := (2 : ℝ) * X^3 - (3 : ℝ) * X^2 - 10 * X + 14 in
  (p.roots.map ((id : ℝ → ℝ) ^ (-1)).prod = -7) :=
by
  sorry

end product_of_roots_eq_neg7_l751_751390


namespace quad_root_l751_751842

theorem quad_root (m : ℝ) (β : ℝ) (root_condition : ∃ α : ℝ, α = -5 ∧ (α + β) * (α * β) = x^2 + m * x - 10) : β = 2 :=
by
  sorry

end quad_root_l751_751842


namespace taxi_total_charge_l751_751326

theorem taxi_total_charge :
  let initial_fee := 2.25
      additional_charge_per_increment := 0.3
      distance := 3.6
      increment := 2.0 / 5.0
      n_increments := distance / increment
      total_charge := initial_fee + n_increments * additional_charge_per_increment
  in total_charge = 4.95 :=
by
  -- Step-by-step definitions are written to match the solution provided
  -- Proof/validation steps are omitted as per the instructions.
  sorry

end taxi_total_charge_l751_751326


namespace tennis_club_matches_l751_751355

theorem tennis_club_matches :
  ∃ (M : Finset (Fin 14)), M.card = 6 ∧ ∀ i j ∈ M, i ≠ j →
    (∃ P₁ P₂ : Finset (Fin 20), P₁.card = 2 ∧ P₂.card = 2 ∧ P₁ ∩ P₂ = ∅) :=
sorry

end tennis_club_matches_l751_751355


namespace AC_length_l751_751182

variable (A B C : Point)

-- Conditions
def segment_AB : ℝ := 5
def seg_BC : ℝ := 3

noncomputable def point_C_between (A B C : Point) (seg_AB seg_BC : ℝ) : Prop :=
  A ≠ B ∧ colinear A B C ∧ (dist A B = seg_AB) ∧ 
  (dist B C = seg_BC) ∧ (dist A C = dist A B - dist B C)

noncomputable def point_C_extension (A B C : Point) (seg_AB seg_BC : ℝ) : Prop :=
  A ≠ B ∧ colinear A B C ∧ (dist A B = seg_AB) ∧ 
  (dist B C = seg_BC) ∧ (dist A C = dist A B + dist B C)

theorem AC_length (A B C : Point) (seg_AB seg_BC : ℝ) (h1 : segment_AB = 5) 
  (h2 : seg_BC = 3) :
  point_C_between A B C segment_AB seg_BC ∨ point_C_extension A B C segment_AB seg_BC → 
  (dist A C = 2 ∨ dist A C = 8) :=
by
  sorry

end AC_length_l751_751182


namespace inequality_holds_l751_751302

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l751_751302


namespace probability_of_cold_given_rhinitis_l751_751803

/-- Define the events A and B as propositions --/
def A : Prop := sorry -- A represents having rhinitis
def B : Prop := sorry -- B represents having a cold

/-- Define the given probabilities as assumptions --/
axiom P_A : ℝ -- P(A) = 0.8
axiom P_A_and_B : ℝ -- P(A ∩ B) = 0.6

/-- Adding the conditions --/
axiom P_A_val : P_A = 0.8
axiom P_A_and_B_val : P_A_and_B = 0.6

/-- Define the conditional probability --/
noncomputable def P_B_given_A : ℝ := P_A_and_B / P_A

/-- The main theorem which states the problem --/
theorem probability_of_cold_given_rhinitis : P_B_given_A = 0.75 :=
by 
  sorry

end probability_of_cold_given_rhinitis_l751_751803


namespace closest_fraction_l751_751371

theorem closest_fraction (a b : ℕ) (ha : a = 23) (hb : b = 120) :
  ∃ x ∈ {1/4, 1/5, 1/6, 1/7, 1/8}, abs (a / b - x) = abs (a / b - 1/5) :=
by
  sorry

end closest_fraction_l751_751371


namespace angle_of_inclination_l751_751926

theorem angle_of_inclination (α : ℝ) (h_slope : ∀ x : ℝ, y : ℝ, y = -√3 * x + 1 → y = -√3 * x + 1) : 
  α = 2 * π / 3 ↔ α ∈ Set.Ico 0 π ∧ tan α = -√3 := 
by
  sorry

end angle_of_inclination_l751_751926


namespace slope_intersects_circle_l751_751020

theorem slope_intersects_circle (k : ℝ) :
  (∃ x y : ℝ, y = k * (x + 1) ∧ x^2 + y^2 = 2 * x) ↔ 
  -real.sqrt 3 / 3 < k ∧ k < real.sqrt 3 / 3 :=
by 
  sorry

end slope_intersects_circle_l751_751020


namespace proportionality_problem_l751_751801

theorem proportionality_problem 
  (A : ∀ x y : ℝ, x^2 + y^2 = 16)
  (B : ∀ x y : ℝ, 2 * x * y = 5)
  (C : ∀ x y : ℝ, x = 3 * y)
  (D : ∀ x y : ℝ, x^2 = 4 * y)
  (E : ∀ x y : ℝ, 5 * x + 2 * y = 20) :
  (∀ x y : ℝ, (y is neither directly nor inversely proportional to x <-> 
                  (x^2 + y^2 = 16 ∨ x^2 = 4 * y ∨ 5 * x + 2 * y = 20))) :=
sorry

end proportionality_problem_l751_751801


namespace intersection_complement_l751_751608

-- Define the universal set U
def U := Set ℝ

-- Define the set M as described
def M := { x : ℝ | ∃ y : ℝ, y = Real.log (x^2 - 1)}

-- Define the set N
def N := { x : ℝ | 0 < x ∧ x < 2}

-- Prove that the intersection of N and the complement of M in U equals { x | 0 < x ≤ 1 }
theorem intersection_complement :
  N ∩ (U \ M) = { x : ℝ | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_complement_l751_751608


namespace area_of_triangle_BCD_l751_751111

-- Define the points A, B, C, D
variables {A B C D : Type} 

-- Define the lengths of segments AC and CD
variables (AC CD : ℝ)
-- Define the area of triangle ABC
variables (area_ABC : ℝ)

-- Define height h
variables (h : ℝ)

-- Initial conditions
axiom length_AC : AC = 9
axiom length_CD : CD = 39
axiom area_ABC_is_36 : area_ABC = 36
axiom height_is_8 : h = (2 * area_ABC) / AC

-- Define the area of triangle BCD
def area_BCD (CD h : ℝ) : ℝ := 0.5 * CD * h

-- The theorem that we want to prove
theorem area_of_triangle_BCD : area_BCD 39 8 = 156 :=
by
  sorry

end area_of_triangle_BCD_l751_751111


namespace three_term_inequality_l751_751291

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l751_751291


namespace all_songs_same_genre_l751_751347

-- Define the number of genres and the sequence of songs
def genres : ℕ := 10
def first_chosen_songs : ℕ := 17

-- Define a predicate for the system eventually playing songs of the same genre from a certain point onwards
def same_genre_eventually (songs : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, ∀ n : ℕ, n ≥ k → songs n = songs k

-- Define the main theorem
theorem all_songs_same_genre (songs : ℕ → ℕ) (h_initial : ∀ n < first_chosen_songs, songs n < genres)
  (h_majority : ∀ n ≥ first_chosen_songs, 
    let lst_17 := (λ m, songs (n - m)) in 
    let counts := (λ g, finset.card (finset.filter (λ m : ℕ, songs m = g) (finset.range 17))) in
    ∃ majority_genre, counts majority_genre >= counts i ∀ i < genres) :
  same_genre_eventually songs :=
sorry

end all_songs_same_genre_l751_751347


namespace solve_for_n_l751_751631

theorem solve_for_n (n : ℚ) (h : 5^(2 * n + 1) = 1 / 25) : n = -3 / 2 :=
sorry

end solve_for_n_l751_751631


namespace min_tokens_in_grid_l751_751699

/-- Minimum number of tokens in a 99x99 grid so that each 4x4 subgrid contains at least 8 tokens -/
theorem min_tokens_in_grid : ∀ (grid : Matrix (Fin 99) (Fin 99) ℕ), (∀ i j : Fin 96, 8 ≤ ∑ r in FinSet.range 4, ∑ c in FinSet.range 4, grid (i + r) (j + c)) → ∑ i j, grid i j ≥ 4801 :=
by
  -- The proof goes here
  sorry

end min_tokens_in_grid_l751_751699


namespace HA_JA_on_Euler_circle_l751_751184

variable (A B C H L_A H_A J_A : Type)
variable [is_orthocenter H A B C]
variable [is_foot_of_altitude L_A A B C]
variable [is_midpoint H_A H L_A]
variable [is_midpoint J_A H A]
variable [belongs_to_circumcircle L_A A B C]
variable [is_euler_circle E A B C]

theorem HA_JA_on_Euler_circle :
  H_A ∈ E ∧ J_A ∈ E := sorry

end HA_JA_on_Euler_circle_l751_751184


namespace find_p_l751_751220

-- Define the parabola C1 as x^2 = 2py where p > 0
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the hyperbola C2 as x^2/3 - y^2 = 1
def hyperbola (x y : ℝ) : Prop := (x^2) / 3 - y^2 = 1

-- Define the focus of the parabola C1
def focus_parabola (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the right focus of the hyperbola C2
def right_focus_hyperbola : ℝ × ℝ := (2, 0)

-- Define the equation of the line connecting the two foci
def line_through_foci (p x y : ℝ) : Prop := p * x + 4 * y - 2 * p = 0

-- Define the point M on the parabola such that the tangent at M is parallel to an asymptote of the hyperbola
def point_M (p x₀ : ℝ) : Prop := x₀ / p = (Real.sqrt 3) / 3

-- Define the tangent condition
def tangent_condition (p x₀ : ℝ) : Prop := parabola p x₀ (x₀^2 / (2 * p)) ∧ point_M p x₀

-- The proof problem: Prove that given the conditions, p = 4 * sqrt 3 / 3
theorem find_p (p : ℝ) (x₀ : ℝ) : 
  (0 < p) → 
  hyperbola (2 : ℝ) (0 : ℝ) → 
  focus_parabola p = (0, p / 2) → 
  right_focus_hyperbola = (2, 0) → 
  line_through_foci p x₀ (x₀^2 / (2*p)) → 
  tangent_condition p x₀ → 
  p = 4 * Real.sqrt 3 / 3 := 
sorry

end find_p_l751_751220


namespace three_term_inequality_l751_751296

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l751_751296


namespace sum_of_divisors_of_10_is_18_l751_751831

theorem sum_of_divisors_of_10_is_18 :
  ∑ n in { n : ℕ | n > 0 ∧ 10 % n = 0 }, n = 18 :=
by
  sorry

end sum_of_divisors_of_10_is_18_l751_751831


namespace binary_to_decimal_is_1023_l751_751641

-- Define the binary number 1111111111 in terms of its decimal representation
def binary_to_decimal : ℕ :=
  (1 * 2^9 + 1 * 2^8 + 1 * 2^7 + 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0)

-- The theorem statement
theorem binary_to_decimal_is_1023 : binary_to_decimal = 1023 :=
by
  sorry

end binary_to_decimal_is_1023_l751_751641


namespace ultra_prime_looking_count_eq_380_l751_751793

def is_composite (n : ℕ) : Prop :=
  ∃ d, d > 1 ∧ d < n ∧ n % d = 0

def ultra_prime_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬ (n % 2 = 0 ∨ n % 3 = 0 ∨ n % 5 = 0 ∨ n % 7 = 0)

def count_ultra_prime_looking_numbers (limit : ℕ) : ℕ :=
  (List.range' 2 limit).countp ultra_prime_looking

theorem ultra_prime_looking_count_eq_380 : count_ultra_prime_looking_numbers 2000 = 380 :=
  sorry

end ultra_prime_looking_count_eq_380_l751_751793


namespace area_of_quadrilateral_l751_751179

theorem area_of_quadrilateral 
  (ABCD : Type) 
  (A B C D E : ABCD)
  (angle_ABC : ℝ) (angle_ACD : ℝ)
  (AC CD AE : ℝ)
  (h_angle_ABC : angle_ABC = 90) 
  (h_angle_ACD : angle_ACD = 90)
  (h_AC : AC = 25) 
  (h_CD : CD = 40)
  (h_AE : AE = 10) :
  ∃ (x : ℝ), 500 + 12.5 * real.sqrt (625 - x^2) = 
  (let y := real.sqrt (625 - x^2) in 
    1/2 * AC * CD + 12.5 * y) :=
by sorry

end area_of_quadrilateral_l751_751179


namespace equation_of_ellipse_l751_751865

-- Conditions given
variables (a b c : ℝ)
axiom (h_symmetry : a > 0 ∧ b > 0 ∧ c > 0 ∧ 2 * b = 8 * sqrt 5)
axiom (h_eccentricity : c / a = 2 / 3)
axiom (h_pythagorean : a^2 = b^2 + c^2)

-- Question: Proving one of the possible forms of the ellipse equation
theorem equation_of_ellipse : 
  ( (x y: ℝ), 
    ((x^2 / 144) + (y^2 / 80) = 1 ∨ (y^2 / 144) + (x^2 / 80) = 1) ) :=
by
  sorry -- Solution steps proving the correct form go here

end equation_of_ellipse_l751_751865


namespace relations_of_sets_l751_751913

open Set

theorem relations_of_sets {A B : Set ℝ} (h : ∃ x ∈ A, x ∉ B) : 
  ¬(A ⊆ B) ∧ ((A ∩ B ≠ ∅) ∨ (B ⊆ A) ∨ (A ∩ B = ∅)) := sorry

end relations_of_sets_l751_751913


namespace range_of_t_l751_751479

-- Definitions based on the conditions provided
def a (n : ℕ) (t : ℝ) := -n + t
def b (n : ℕ) := 3 ^ (n - 3)

def c (n : ℕ) (t : ℝ) := (a n t + b n) / 2 + abs (a n t - b n) / 2

theorem range_of_t (t : ℝ) : 
  (∀ n : ℕ, n > 0 → c n t ≥ c 3 t) → (3 ≤ t ∧ t ≤ 6) :=
  sorry

end range_of_t_l751_751479


namespace no_valid_point_C_l751_751530

open Real EuclideanGeometry

noncomputable def distance (p q : Point ℝ) : ℝ := euclidean_distance p q

def perimeter (A B C : Point ℝ) : ℝ := distance A B + distance B C + distance C A

def area (A B C : Point ℝ) : ℝ := (1 / 2) * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)).abs

theorem no_valid_point_C : ∀ (A B : Point ℝ), distance A B = 10 → ¬ (∃ C : Point ℝ, perimeter A B C = 50 ∧ area A B C = 100) :=
by
  intros A B h_dist
  sorry

end no_valid_point_C_l751_751530


namespace projection_bisects_segment_l751_751585

open EuclideanGeometry

variables (A B C H_a H_b P Q : Point) (ABC : Triangle A B C)
variables (H_a_proj_AB : P = projection H_a AB)
variables (H_a_proj_AC : Q = projection H_a AC)
variables (altitude_AH_a : AltitudeABC ABC A H_a)
variables (altitude_BH_b : AltitudeABC ABC B H_b)

theorem projection_bisects_segment :
  bisects (lineThrough P Q) (segment H_a H_b) :=
sorry

end projection_bisects_segment_l751_751585


namespace vector_parallel_l751_751789

theorem vector_parallel (t : ℝ) (k : ℝ) : 
  (3 * t + 5 = 3 * k) → (2 * t + 1 = 2 * k) → k = 7 / 2 → 
  ∃ (a b : ℝ), (a = 3 * k) ∧ (b = 2 * k) ∧ a = 21 / 2 ∧ b = 7 :=
begin
  intros h1 h2 hk,
  use 3 * k,
  use 2 * k,
  split,
  { refl, },
  split,
  { refl, },
  split,
  { calc
    3 * k = 3 * (7 / 2) : by { rw hk }
         ... = 21 / 2 : by ring, },
  { calc
    2 * k = 2 * (7 / 2) : by { rw hk }
         ... = 7 : by ring, },
end

end vector_parallel_l751_751789


namespace t_f_3_equals_sqrt_44_l751_751599

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 4)
noncomputable def f (x : ℝ) : ℝ := 6 + t x

theorem t_f_3_equals_sqrt_44 : t (f 3) = Real.sqrt 44 := by
  sorry

end t_f_3_equals_sqrt_44_l751_751599


namespace arithmetic_sequence_common_difference_l751_751027

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ) 
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_variance : (1/5) * ((a 1 - (a 3)) ^ 2 + (a 2 - (a 3)) ^ 2 + (a 3 - (a 3)) ^ 2 + (a 4 - (a 3)) ^ 2 + (a 5 - (a 3)) ^ 2) = 8) :
  d = 2 ∨ d = -2 := 
sorry

end arithmetic_sequence_common_difference_l751_751027


namespace speed_of_faster_train_l751_751236

noncomputable def length_train1 : ℝ := 140
noncomputable def length_train2 : ℝ := 210
noncomputable def time_to_cross : ℝ := 12.59899208063355
noncomputable def slower_train_speed_km_per_hr : ℝ := 40

noncomputable def slower_train_speed_m_per_s : ℝ := slower_train_speed_km_per_hr / 3.6
noncomputable def total_distance : ℝ := length_train1 + length_train2
noncomputable def relative_speed : ℝ := total_distance / time_to_cross
noncomputable def faster_train_speed_m_per_s : ℝ := relative_speed - slower_train_speed_m_per_s
noncomputable def faster_train_speed_km_per_hr : ℝ := faster_train_speed_m_per_s * 3.6

theorem speed_of_faster_train : faster_train_speed_km_per_hr ≈ 60.0044 := 
sorry

end speed_of_faster_train_l751_751236


namespace sin_450_eq_1_l751_751382

theorem sin_450_eq_1 :
  sin (450 * Real.pi / 180) = 1 :=
by
  sorry

end sin_450_eq_1_l751_751382


namespace factorization_of_x4_plus_16_l751_751398

theorem factorization_of_x4_plus_16 :
  (x : ℝ) → x^4 + 16 = (x^2 + 2 * x + 2) * (x^2 - 2 * x + 2) :=
by
  intro x
  -- Placeholder for the proof
  sorry

end factorization_of_x4_plus_16_l751_751398


namespace solve_system_of_equations_l751_751258

theorem solve_system_of_equations (x y : ℚ)
  (h1 : 15 * x + 24 * y = 18)
  (h2 : 24 * x + 15 * y = 63) :
  x = 46 / 13 ∧ y = -19 / 13 := 
sorry

end solve_system_of_equations_l751_751258


namespace quadratic_perfect_square_l751_751972

theorem quadratic_perfect_square (a b : ℤ) :
  (∀ᶠ x in filter.at_top, ∃ y, x^2 + a * x + b = y^2) ↔ (a^2 - 4 * b = 0) :=
sorry

end quadratic_perfect_square_l751_751972


namespace product_469160_9999_l751_751782

theorem product_469160_9999 :
  469160 * 9999 = 4690696840 :=
by
  sorry

end product_469160_9999_l751_751782


namespace part1_part2_l751_751047

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := log x + a * x^2 - 3 * x

theorem part1 (a : ℝ) (h_tangent : deriv (λ x, g x a) 1 = 0) : a = 1 :=
by 
  have h_deriv : ∀ x, deriv (λ x, log x + a * x^2 - 3 * x) x = (1 / x) + 2 * a * x - 3,
  { intro x, exact deriv_add (deriv_add (deriv_log x) (deriv_const_mul x (a * x) 2)) (deriv_const_mul x (-3) 1) },
  have h1 := h_deriv 1, rw [h_tangent, add_eq_zero_iff] at h1, linarith

theorem part2 (a : ℝ) (h_a : a = 1) :
  (∀ x > 0, deriv (λ x, g x 1) x = (1 / x) + 2 * x - 3) ∧
  (fderiv ℝ (λ x, g x 1) 1).toLinearMap 1 = -2 ∧
  (fderiv ℝ (λ x, g x 1) (1 / 2)).toLinearMap (1 / 2) = -log 2 - 5 / 4 :=
by sorry

end part1_part2_l751_751047


namespace find_z_l751_751904

-- Define z as a complex number
variable (z : ℂ)

-- Define the conjugate of z
noncomputable def conj_z := conj z

-- Define the condition given in the problem
axiom condition : conj_z * (1 - I)^2 = 4 + 2 * I

-- The theorem to prove that z = -1 - 2i
theorem find_z : z = -1 - 2 * I :=
sorry

end find_z_l751_751904


namespace find_k_range_of_k_l751_751974

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.exp (2 * x)
def g (x : ℝ) (k : ℝ) : ℝ := k * x + 1

-- The condition that the line y = g(x) is tangent to the curve y = f(x)
def tangent_condition (k : ℝ) : Prop :=
  ∃ t : ℝ, f t = g t k ∧ (deriv f t) = k

-- Condition for |f(x) - g(x)| > 2x for all x in (0, m)
def condition_greater_than_2x (k m : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < m → |f x - g x k| > 2 * x

-- Prove k = 2 when the tangent condition holds
theorem find_k (k : ℝ) : tangent_condition k → k = 2 := sorry

-- Prove the range of k is (4, +∞) when the condition |f(x) - g(x)| > 2x holds for any x in (0, m)
theorem range_of_k (k m : ℝ) (h : 0 < m) : 
  (∀ k > 0, condition_greater_than_2x k m → 4 < k) := sorry

end find_k_range_of_k_l751_751974


namespace length_CB_l751_751087

/--
In a triangle ABC, line segment DE is parallel to line segment AB.
Given:
  1. CD = 4 cm
  2. DA = 10 cm
  3. CE = 6 cm
Prove that the length of CB equals 21 cm.
-/
theorem length_CB (C D E A B : Type) [AddGroup A] [AddGroup B]  
  (hParallel : ∃ DE AB : Set Line [ Parallel DE AB ])
  (hCD : CD = 4)
  (hDA : DA = 10)
  (hCE : CE = 6) :
  CB = 21 := 
sorry

end length_CB_l751_751087


namespace sin_A_value_triangle_area_l751_751550

-- Definition of the given conditions
variables {a b c : ℝ}
variables (A B C : ℝ)
variables (triangle_ABC : Prop) [triangle_ABC : IsTriangle a b c]

-- The conditions
axiom condition1 : 4 * a = Real.sqrt 5 * c
axiom condition2 : Real.cos C = 3 / 5
axiom condition3 : b = 11
axiom opposite_sides : SidesOppositeToAngles a b c A B C

-- Proving the required results
theorem sin_A_value : sin A = Real.sqrt 5 / 5 :=
by
  sorry

theorem triangle_area : area_triangle a b c = 22 :=
by
  sorry

end sin_A_value_triangle_area_l751_751550


namespace inequality_proof_l751_751275

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751275


namespace distribute_40_candies_l751_751117

theorem distribute_40_candies :
  ∃ (a : Fin 6 → ℕ), (∑ i, a i = 40) ∧ (∀ i j, i ≠ j → a i ≠ a j) ∧ (∀ i j, i ≠ j → a i + a j < 20) :=
by
  sorry

end distribute_40_candies_l751_751117


namespace angles_of_triangle_l751_751642

theorem angles_of_triangle (a b c R : ℝ) (h : R = a * real.sqrt (b * c) / (b + c)) :
  ∃ A B C : ℝ, A = π / 2 ∧ B = π / 4 ∧ C = π / 4 :=
by
  sorry

end angles_of_triangle_l751_751642


namespace x_in_interval_l751_751744

theorem x_in_interval (x : ℝ) (h : x = (1 / x) * (-x) + 2) : 0 < x ∧ x ≤ 2 :=
by
  -- Place the proof here
  sorry

end x_in_interval_l751_751744


namespace sum_of_eight_numbers_l751_751515

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l751_751515


namespace sum_of_eight_numbers_l751_751511

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l751_751511


namespace number_of_ways_to_draw_balls_l751_751332

theorem number_of_ways_to_draw_balls :
  let balls := (finset.range 15).map (λ n, n + 1),
      odd_balls := {x ∈ balls | x % 2 = 1},
      even_balls := {x ∈ balls | x % 2 = 0} in
  finset.card balls = 15 ∧
  finset.card odd_balls = 8 ∧
  finset.card even_balls = 7 →
  (8 * 7 * 6 * 7 + 7 * 8 * 7 * 6) = 32736 := 
by 
  intros balls odd_balls even_balls h,
  have h_card_balls : finset.card balls = 15 := h.1,
  have h_card_odd : finset.card odd_balls = 8 := h.2.1,
  have h_card_even : finset.card even_balls = 7 := h.2.2,
  sorry

end number_of_ways_to_draw_balls_l751_751332


namespace triangle_side_relationship_l751_751483

theorem triangle_side_relationship (a b c : ℝ) (h1 : a > c) (h2 : a + c > b) : 
  |c - a| - real.sqrt ((a + c - b)^2) = b - 2c :=
by sorry

end triangle_side_relationship_l751_751483


namespace num_of_nickels_l751_751350

theorem num_of_nickels (n : ℕ) (h1 : n = 17) (h2 : (17 * n) - 1 = 18 * (n - 1)) : n = 17 → 17 * n = 289 → ∃ k, k = 2 :=
by 
  intros hn hv
  sorry

end num_of_nickels_l751_751350


namespace slope_tangent_at_one_l751_751876

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x

theorem slope_tangent_at_one : 
    let f' := (fun x => deriv f x)
    f' 1 = -1 := 
by
  let f' := (fun x => deriv f x)
  have : f' 1 = (1 / 1) - 2 := by sorry
  rw [←this]
  exact rfl

end slope_tangent_at_one_l751_751876


namespace necessary_and_sufficient_condition_l751_751363

theorem necessary_and_sufficient_condition (a b : ℝ) : a > b ↔ a^3 > b^3 :=
by {
  sorry
}

end necessary_and_sufficient_condition_l751_751363


namespace magnitude_z_l751_751443

noncomputable def z : ℂ := 1 / (1 + complex.I) + complex.I

theorem magnitude_z : complex.abs z = real.sqrt 2 / 2 := by
  sorry

end magnitude_z_l751_751443


namespace find_a5_l751_751110

variable {a_n : ℕ → ℤ} -- Type of the arithmetic sequence
variable (d : ℤ)       -- Common difference of the sequence

-- Assuming the sequence is defined as an arithmetic progression
axiom arithmetic_seq (a d : ℤ) : ∀ n : ℕ, a_n n = a + n * d

theorem find_a5
  (h : a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 45):
  a_n 5 = 9 :=
by 
  sorry

end find_a5_l751_751110


namespace f_one_value_l751_751036

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h_f_defined : ∀ x, x > 0 → ∃ y, f x = y
axiom h_f_strict_increasing : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom h_f_eq : ∀ x, x > 0 → f x * f (f x + 1/x) = 1

theorem f_one_value : f 1 = (1 + Real.sqrt 5) / 2 := 
by
  sorry

end f_one_value_l751_751036


namespace difference_of_cats_l751_751170

-- Definitions based on given conditions
def number_of_cats_sheridan : ℕ := 11
def number_of_cats_garrett : ℕ := 24

-- Theorem statement (proof problem) based on the question and correct answer
theorem difference_of_cats : (number_of_cats_garrett - number_of_cats_sheridan) = 13 := by
  sorry

end difference_of_cats_l751_751170


namespace arithmetic_sequence_properties_l751_751960

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : a 2 = 11)
  (h₂ : S 10 = 40)
  (h₃ : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) :
  (∀ n, a n = -2 * n + 15) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 7 → ∑ i in finset.range n, abs (a (i + 1)) = -n^2 + 14*n) ∧
  (∀ n, n ≥ 8 → ∑ i in finset.range n, abs (a (i + 1)) = n^2 - 14*n + 98) :=
by
  sorry  -- sorry to skip the proof

end arithmetic_sequence_properties_l751_751960


namespace tatyana_correct_l751_751705

/-- Define a condition where Svetlana has 25 copper coins. --/
def num_coins := 25

/-- Define the number of denominations to be 4. --/
def num_denominations := 4

/-- The pigeonhole principle states that if n items are distributed among m containers,
     with n > m, then at least one container must contain more than ⌈n / m⌉ items. --/

/-- Prove that among 25 copper coins distributed among 4 denominations,
     there is at least one denomination with at least 7 coins. --/
theorem tatyana_correct (n : ℕ) (m : ℕ) (h1 : n = 25) (h2 : m = 4) :
  ∃ d : ℕ, d >= 7 ∧ d <= ⌈n / m⌉ :=
by
  sorry

end tatyana_correct_l751_751705


namespace semicircle_segment_sum_eq_diameter_l751_751493

variables {r a : ℝ} (M N A B : ℝ)

theorem semicircle_segment_sum_eq_diameter
  (h_AB : ∥A - B∥ = 2 * r)
  (h_AT : ∥A - T∥ = 2 * a)
  (h_a_cond : 2 * a < r / 2)
  (h_MP_AM : ∀ P, ∥M - P∥ / ∥A - M∥ = 1)
  (h_NQ_AN : ∀ Q, ∥N - Q∥ / ∥A - N∥ = 1)
  : ∥A - M∥ + ∥A - N∥ = ∥A - B∥ :=
sorry

end semicircle_segment_sum_eq_diameter_l751_751493


namespace mike_total_tires_changed_l751_751168

theorem mike_total_tires_changed :
  let tires_motorcycles := 12 * 2 in
  let tires_cars := 10 * 4 in
  let tires_bicycles := 8 * 2 in
  let tires_trucks := 5 * 18 in
  let tires_atvs := 7 * 4 in
  tires_motorcycles + tires_cars + tires_bicycles + tires_trucks + tires_atvs = 198 := by
  sorry

end mike_total_tires_changed_l751_751168


namespace sum_of_digits_in_rectangle_l751_751996

theorem sum_of_digits_in_rectangle :
  ∃ A B C : ℕ,
    (4 + A + 1 + B = 12) ∧
    (4 + A + 1 + B = 6 + 6) ∧
    (C + 1 + 6 + C = 11) ∧
    (1 + B + 2 + C = 9) ∧
    (A + 8 + 8 = 8) ∧
    (A + 8 + B + 2 + C = 13) ∧
    (9 = 4 + A + 1 + B) ∧
    (B + 2 + C = 9) ∧    
    B = 5 ∧ A = 2 ∧ C = 6 :=
sorry

end sum_of_digits_in_rectangle_l751_751996


namespace polyhedron_volume_is_8_l751_751932

noncomputable def volume_of_polyhedron : ℝ :=
  let A := (equilateral_triangle (real.sqrt 3))
  let B := (equilateral_triangle (real.sqrt 3))
  let C := (equilateral_triangle (real.sqrt 3))
  let D := (square 2)
  let E := (square 2)
  let F := (square 2)
  let G := (right_triangle 2 (2 * real.sqrt 2))
  let polyhedron := fold_into_polyhedron [A, B, C, D, E, F, G] 
  volume polyhedron

theorem polyhedron_volume_is_8 : volume_of_polyhedron = 8 := by
  sorry

end polyhedron_volume_is_8_l751_751932


namespace inequality_ge_one_l751_751308

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l751_751308


namespace tetrahedron_edge_length_l751_751846

-- Define the problem specifications
def mutuallyTangent (r : ℝ) (a b c d : ℝ → ℝ → ℝ → Prop) :=
  a = b ∧ a = c ∧ a = d ∧ b = c ∧ b = d ∧ c = d

noncomputable def tetrahedronEdgeLength (r : ℝ) : ℝ :=
  2 + 2 * Real.sqrt 6

-- Proof goal: edge length of tetrahedron containing four mutually tangent balls each of radius 1
theorem tetrahedron_edge_length (r : ℝ) (a b c d : ℝ → ℝ → ℝ → Prop)
  (h1 : r = 1)
  (h2 : mutuallyTangent r a b c d)
  : tetrahedronEdgeLength r = 2 + 2 * Real.sqrt 6 :=
sorry

end tetrahedron_edge_length_l751_751846


namespace ratio_of_marbles_l751_751806

noncomputable def marble_ratio : ℕ :=
  let initial_marbles := 40
  let marbles_after_breakfast := initial_marbles - 3
  let marbles_after_lunch := marbles_after_breakfast - 5
  let marbles_after_moms_gift := marbles_after_lunch + 12
  let final_marbles := 54
  let marbles_given_back_by_Susie := final_marbles - marbles_after_moms_gift
  marbles_given_back_by_Susie / 5

theorem ratio_of_marbles : marble_ratio = 2 := by
  -- proof steps would go here
  sorry

end ratio_of_marbles_l751_751806


namespace find_x0_l751_751080

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def df (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 : ∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ + df x₀ = 1 ∧ x₀ = 1 :=
by
  exists 1
  split
  · norm_num
  · split
    · sorry
    · refl

end find_x0_l751_751080


namespace area_smallest_region_enclosed_l751_751797

theorem area_smallest_region_enclosed {x y : ℝ} (circle_eq : x^2 + y^2 = 9) (abs_line_eq : y = |x|) :
  ∃ area, area = (9 * Real.pi) / 4 :=
by
  sorry

end area_smallest_region_enclosed_l751_751797


namespace sin_450_eq_1_l751_751384

theorem sin_450_eq_1 : sin (450 * π / 180) = 1 := by
  have angle_eq : 450 = 360 + 90 := by norm_num
  -- Simplify 450 degrees to radians
  rw [angle_eq, Nat.cast_add, add_mul, Nat.cast_mul]
  -- Convert degrees to radians
  rw [sin_add, sin_mul_pi_div, cos_mul_pi_div, sin_mul_pi_div, (show (90 : ℝ) = π / 2 from by norm_num)]

  sorry -- Omitting proof details

end sin_450_eq_1_l751_751384


namespace store_customers_l751_751401

theorem store_customers (customers : ℕ) (h : customers = 2016) : 
  ∃ k : ℕ, k = 44 ∧ (∀ (n : ℕ), n ≤ k → 
    (∃ (intervals : list (ℕ × ℕ)), intervals.length = n ∧ 
    ((∃ p : ℕ, ∀ i : ℕ, i < n → p ∈ intervals.to_list[i]) 
    ∨ (∀ i j : ℕ, i < j ∧ j < n → disjoint intervals.to_list[i] intervals.to_list[j]))))) :=
by
  sorry

end store_customers_l751_751401


namespace cos_8_identity_l751_751071

theorem cos_8_identity (m : ℝ) (h : Real.sin 74 = m) : 
  Real.cos 8 = Real.sqrt ((1 + m) / 2) :=
sorry

end cos_8_identity_l751_751071


namespace count_mixed_4_digit_numbers_l751_751623

-- Defining what constitutes a mixed number according to conditions stated
def is_mixed (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  n ≥ 1000 ∧ n < 10000 ∧
  (List.length digits = 4) ∧
  (List.nodup digits) ∧
  (digits.head! ≠ List.maximum digits) ∧
  (digits.head! ≠ List.minimum digits) ∧
  (digits.last! ≠ List.minimum digits)

-- Statement: Prove that the number of 4-digit mixed integers is 1680
theorem count_mixed_4_digit_numbers :
  let mixed_count := (Nat.choose 10 4) * 8 in
  mixed_count = 1680 :=
by
  sorry

end count_mixed_4_digit_numbers_l751_751623


namespace cube_edge_numbers_possible_l751_751567

theorem cube_edge_numbers_possible :
  ∃ (top bottom : Finset ℕ), top.card = 4 ∧ bottom.card = 4 ∧ 
  (top ∪ bottom).card = 8 ∧ 
  ∀ (n : ℕ), n ∈ top ∪ bottom → n ∈ (Finset.range 12).map (+1) ∧ 
  (∏ x in top, x) = (∏ y in bottom, y) :=
by {
  sorry,
}

end cube_edge_numbers_possible_l751_751567


namespace total_peaches_l751_751666

/-- There are 150 baskets of peaches.
    Each odd numbered basket has 8 red peaches and 6 green peaches.
    Each even numbered basket has 5 red peaches and 7 green peaches.
    Prove that the total number of peaches is 1950.
-/
theorem total_peaches (n_baskets : ℕ) (odd_red : ℕ) (odd_green : ℕ) (even_red : ℕ) (even_green : ℕ) :
  n_baskets = 150 →
  odd_red = 8 →
  odd_green = 6 →
  even_red = 5 →
  even_green = 7 →
  let odd_baskets := n_baskets / 2,
      even_baskets := n_baskets / 2,
      odd_peaches_per_basket := odd_red + odd_green,
      even_peaches_per_basket := even_red + even_green,
      total_odd_peaches := odd_baskets * odd_peaches_per_basket,
      total_even_peaches := even_baskets * even_peaches_per_basket,
      total_peaches := total_odd_peaches + total_even_peaches in
  total_peaches = 1950 :=
 by simp; sorry

end total_peaches_l751_751666


namespace sin_arithmetic_sequence_l751_751410

theorem sin_arithmetic_sequence (a : ℝ) (h1 : 0 < a) (h2 : a < 360) 
  (h3 : sin a + sin (3 * a) = 2 * sin (2 * a)) : 
  a = 45 ∨ a = 180 ∨ a = 315 :=
by
  sorry

end sin_arithmetic_sequence_l751_751410


namespace inequality_proof_necessary_and_sufficient_conditions_for_equality_l751_751963

theorem inequality_proof (n : ℕ) (a b : Fin n → ℝ) 
  (h1 : ∀ i, 1 ≤ a i ∧ a i ≤ 2)
  (h2 : ∀ i, 1 ≤ b i ∧ b i ≤ 2)
  (h3 : ∑ i, (a i)^2 = ∑ i, (b i)^2)
  : ∑ i, (a i)^3 / (b i) ≤ (17 / 10) * ∑ i, (a i)^2 :=
sorry

-- Statements for equality conditions
theorem necessary_and_sufficient_conditions_for_equality (n : ℕ) (a b : Fin n → ℝ)
  (h1 : ∑ i, (a i)^2 = ∑ i, (b i)^2)
  (h2 : ∀ i, 1 ≤ a i ∧ a i ≤ 2)
  (h3 : ∀ i, 1 ≤ b i ∧ b i ≤ 2)
  : (∀ i, (a i = 1 ∧ b i = 2) ∨ (a i = 2 ∧ b i = 1)) ↔ (n % 2 = 0) :=
sorry

end inequality_proof_necessary_and_sufficient_conditions_for_equality_l751_751963


namespace no_straight_line_can_intersect_all_sides_except_vertices_of_1989_gon_l751_751802

theorem no_straight_line_can_intersect_all_sides_except_vertices_of_1989_gon :
  ¬ ∃ (polygon : Type) [is_polygon polygon 1989], ∃ (line : Type),
    intersects_each_side_except_vertices polygon line :=
by
  sorry

-- Definitions that might be needed
class is_polygon (P : Type) (n : ℕ) : Prop :=
(number_of_sides : ℕ)
(number_of_sides_eq : number_of_sides = n)

class intersects_each_side_except_vertices (polygon : Type) (line : Type) : Prop :=
(intersects_side : ∀ (side : polygon), intersects_side_in_other_than_vertices line side)

end no_straight_line_can_intersect_all_sides_except_vertices_of_1989_gon_l751_751802


namespace find_n_l751_751000

theorem find_n :
  ∃ (n : ℤ), -180 ≤ n ∧ n ≤ 180 ∧ sin (n * Real.pi / 180) = cos (510 * Real.pi / 180) → n = -60 :=
by
  sorry

end find_n_l751_751000


namespace ratio_of_areas_l751_751543

theorem ratio_of_areas (A B C P : ℝ^2) 
  (h₁ : (P - A) + (P - B) + (P - C) = B - A) 
  (h₂ : True) : 
  let area := λ X Y Z : ℝ^2, 0.5 * abs ((X - Y) ⬝ (Z - Y)) in 
  area P B C / area A B C = 2 / 3 := 
sorry

end ratio_of_areas_l751_751543


namespace hyperbola_passing_through_M_l751_751786

noncomputable def hyperbola_equation : ℝ → ℝ → ℝ := λ x y, x^2 - y^2

theorem hyperbola_passing_through_M :
  ∃ a^2, (a^2 = 12) ∧ ∀ x y, ((x, y) = (4, -2) → hyperbola_equation x y = a^2) :=
by
  use 12
  split
  { sorry }
  { intros x y h
    rw [←h]
    calc
      hyperbola_equation 4 (-2)
          = 4^2 - (-2)^2 : rfl
      ... = 16 - 4 : by norm_num
      ... = 12 : by norm_num }

end hyperbola_passing_through_M_l751_751786


namespace asia_fraction_correct_l751_751906

-- Define the problem conditions
def fraction_NA (P : ℕ) : ℚ := 1/3 * P
def fraction_Europe (P : ℕ) : ℚ := 1/8 * P
def fraction_Africa (P : ℕ) : ℚ := 1/5 * P
def others : ℕ := 42
def total_passengers : ℕ := 240

-- Define the target fraction for Asia
def fraction_Asia (P: ℕ) : ℚ := 17 / 120

-- Theorem: the fraction of the passengers from Asia equals 17/120
theorem asia_fraction_correct : ∀ (P : ℕ), 
  P = total_passengers →
  fraction_NA P + fraction_Europe P + fraction_Africa P + fraction_Asia P * P + others = P →
  fraction_Asia P = 17 / 120 := 
by sorry

end asia_fraction_correct_l751_751906


namespace complex_sum_l751_751069

def B := 3 + 2 * Complex.i
def Q := -6 + Complex.i
def R := 3 * Complex.i
def T := 4 + 3 * Complex.i
def U := -2 - 2 * Complex.i

theorem complex_sum :
  B + Q + R + T + U = -1 + 7 * Complex.i :=
by
  -- Here, we would provide the proof steps, but we use sorry to skip the proof itself
  sorry

end complex_sum_l751_751069


namespace shaded_area_ratio_l751_751788

noncomputable def ratio_of_shaded_area_to_area_of_circle (r : ℝ) : ℝ :=
  let area_large_semicircle := (9 * π * r^2) / 2
  let area_small_semicircle_AC := (π * r^2) / 2
  let area_small_semicircle_CB := (2 * π * r^2)
  let shaded_area := area_large_semicircle - (area_small_semicircle_AC + area_small_semicircle_CB)
  let area_circle_CD := π * r^2
  shaded_area / area_circle_CD

theorem shaded_area_ratio (r : ℝ) (h : 0 < r) :
  ratio_of_shaded_area_to_area_of_circle r = 2 :=
by
  -- Proof can be filled here
  sorry

end shaded_area_ratio_l751_751788


namespace james_blue_yarn_l751_751120

theorem james_blue_yarn (berets_needed : ℕ) (spools_per_beret : ℕ) (red_yarn : ℕ) (black_yarn : ℕ) (berets_can_make : ℕ) (blue_yarn : ℕ) : 
  berets_needed = 11 → 
  spools_per_beret = 3 → 
  red_yarn = 12 → 
  black_yarn = 15 → 
  berets_can_make = 11 → 
  (berets_needed * spools_per_beret - (red_yarn + black_yarn)) = blue_yarn → 
  blue_yarn = 6 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  exact h6

end james_blue_yarn_l751_751120


namespace angle_BAD_l751_751537

-- Definition of the problem parameters
variables {A B C D : Type} 
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (P Q R S: A)

-- Hypothesis regarding the quadrilateral ABCD
def is_quadrilateral_abcd (h₁ : dist P Q = dist P S)
                          (h₂ : dist Q R = 13)
                          (h₃ : dist R S = 25)
                          (h₄ : angle P Q R = 85)
                          (h₅ : angle Q R S = 160) : Prop :=
  ∃ (x : ℝ), angle S P Q = x ∧ x = 71.25

-- The statement of the problem
theorem angle_BAD (P Q R S : A) 
  (h₁ : dist P Q = dist P S)
  (h₂ : dist Q R = 13)
  (h₃ : dist R S = 25)
  (h₄ : angle P Q R = 85)
  (h₅ : angle Q R S = 160) : angle S P Q = 71.25 :=
begin
  sorry
end

end angle_BAD_l751_751537


namespace min_value_on_transformed_curve_l751_751486

theorem min_value_on_transformed_curve :
  let C := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1},
      transform := λ (p : ℝ × ℝ), (2 * p.1, p.2),
      C' := {p : ℝ × ℝ | (p.1 / 2) ^ 2 + p.2 ^ 2 = 1} in
  ∀ p ∈ C, transform p ∈ C' ∧
    (∀ (x y : ℝ), (x, y) ∈ C' → x + 2 * real.sqrt 3 * y ≥ -4) ∧
    ∃ (x y : ℝ), (x, y) ∈ C' ∧ x + 2 * real.sqrt 3 * y = -4 :=
by
  sorry

end min_value_on_transformed_curve_l751_751486


namespace union_P_Q_eq_l751_751055

noncomputable def a : ℝ := 1
def P : Set ℝ := {3, Real.log 3 a}
def f (x : ℝ) : ℝ := a * Real.cos (2 * x) + 1 + a * Real.sin (2 * x)
def Q : Set ℝ := {f x | x : ℝ}
def intersection_zero : Set ℝ := {0}

-- Given conditions
def condition1 : P = {3, Real.log 3 a} := by sorry
def condition2 : Q = {f x | x : ℝ} := by sorry
def condition3 : P ∩ Q = intersection_zero := by sorry
def condition4 : a = 1 := by sorry

-- The equivalent proof problem statement
theorem union_P_Q_eq : P ∪ Q = {f x | x : ℝ} ∪ {3} := by
  rw [condition1, condition2, condition3, condition4]
  sorry

end union_P_Q_eq_l751_751055


namespace integer_point_intersection_l751_751108

theorem integer_point_intersection :
  ∀ k : ℝ, (4 * k * x - 1 / k = 1 / k * x + 2) → ((∃ x y : ℤ, x = (2 * k + 1) / (4 * k ^ 2 - 1) ∧ y = (8 * k ^ 2 + 4 * k - 1) / (4 * k ^ 2 - 1)) ↔ (k ≠ 0 ∧ k ≠ -1/2)) →
  ∃! (k : ℝ), true := 
begin
  sorry
end

end integer_point_intersection_l751_751108


namespace inequality_proof_l751_751284

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751284


namespace angle_is_2pi_over_3_magnitude_of_combination_l751_751492

variables {α : Type*} [inner_product_space ℝ α]

variables (a b : α)
variables (ha : ∥a∥ = 2) (hb : ∥b∥ = 4) (hab : ⟪a - b, b⟫ = -20)

noncomputable def angle_between_vectors : ℝ := real.arccos ((⟪a, b⟫) / (∥a∥ * ∥b∥))

theorem angle_is_2pi_over_3 : angle_between_vectors a b ha hb hab = 2 * real.pi / 3 :=
sorry

theorem magnitude_of_combination : ∥3 • a + b∥ = 2 * real.sqrt 7 :=
sorry

end angle_is_2pi_over_3_magnitude_of_combination_l751_751492


namespace maximal_value_of_cubed_sum_l751_751150

theorem maximal_value_of_cubed_sum (n : ℕ) (x : Fin n → ℤ) :
  (∀ i, -2 ≤ x i ∧ x i ≤ 3) →
  (∑ i, x i = 25) →
  (∑ i, (x i) ^ 2 = 145) →
  (∑ i, (x i) ^ 3 ≤ 217) := 
sorry

end maximal_value_of_cubed_sum_l751_751150


namespace pencils_in_all_l751_751807

/-- Eugene's initial number of pencils -/
def initial_pencils : ℕ := 51

/-- Pencils Eugene gets from Joyce -/
def additional_pencils : ℕ := 6

/-- Total number of pencils Eugene has in all -/
def total_pencils : ℕ :=
  initial_pencils + additional_pencils

/-- Proof that Eugene has 57 pencils in all -/
theorem pencils_in_all : total_pencils = 57 := by
  sorry

end pencils_in_all_l751_751807


namespace birthday_gift_l751_751427

-- Define the conditions
def friends : Nat := 8
def dollars_per_friend : Nat := 15

-- Formulate the statement to prove
theorem birthday_gift : friends * dollars_per_friend = 120 := by
  -- Proof is skipped using 'sorry'
  sorry

end birthday_gift_l751_751427


namespace hectors_sibling_product_l751_751898

theorem hectors_sibling_product (sisters : Nat) (brothers : Nat) (helen : Nat -> Prop): 
  (helen 4) → (helen 7) → (helen 5) → (helen 6) →
  (sisters + 1 = 5) → (brothers + 1 = 7) → ((sisters * brothers) = 30) :=
by
  sorry

end hectors_sibling_product_l751_751898


namespace copy_pages_count_l751_751643

-- Definitions and conditions
def cost_per_page : ℕ := 5  -- Cost per page in cents
def total_money : ℕ := 50 * 100  -- Total money in cents

-- Proof goal
theorem copy_pages_count : total_money / cost_per_page = 1000 := 
by sorry

end copy_pages_count_l751_751643


namespace problem_statement_l751_751329

open Nat

theorem problem_statement 
  (x y : ℕ)
  (h1 : (x + 2) ^ 3 = x ^ 3 + 6 * x ^ 2 + 12 * x + 8)
  (h2 : (x + 4) ^ 3 = x ^ 3 + 12 * x ^ 2 + 48 * x + 64)
  (h3 : x ^ 3 + 6 * x ^ 2 + 12 * x + 8 < x ^ 3 + 8 * x ^ 2 + 42 * x + 27)
  (h4 : x ^ 3 + 8 * x ^ 2 + 42 * x + 27 < x ^ 3 + 12 * x ^ 2 + 48 * x + 64)
  (h5 : y = x + 3)
  (h6 : (x + 3) ^ 3 = x ^ 3 + 9 * x ^ 2 + 27 * x + 27)
  (h7 : x ^ 2 = 15 * x):
  x = 15 ∧ y = 18 :=
begin
  sorry
end

end problem_statement_l751_751329


namespace smallest_n_leq_l751_751702

theorem smallest_n_leq (n : ℤ) : (n ^ 2 - 13 * n + 40 ≤ 0) → (n = 5) :=
sorry

end smallest_n_leq_l751_751702


namespace cuboid_edge_length_l751_751210

-- This is the main statement we want to prove
theorem cuboid_edge_length (L : ℝ) (w : ℝ) (h : ℝ) (V : ℝ) (w_eq : w = 5) (h_eq : h = 3) (V_eq : V = 30) :
  V = L * w * h → L = 2 :=
by
  -- Adding the sorry allows us to compile and acknowledge the current placeholder for the proof.
  sorry

end cuboid_edge_length_l751_751210


namespace sufficient_but_not_necessary_l751_751205

-- Define the point in the complex plane
def point (a : ℝ) : ℂ := (a - 2 * I) * I

-- Define what it means for a point to be in the fourth quadrant
def inFourthQuadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Prove the equivalence
theorem sufficient_but_not_necessary (a : ℝ) : inFourthQuadrant (point (-1)) ↔ (a = -1) := 
  by
    -- Establish the equivalence
    sorry

end sufficient_but_not_necessary_l751_751205


namespace prove_f_8_eq_2sqrt2_l751_751466

-- Assumptions and definitions
variable {a : ℝ}
variable (f : ℝ → ℝ)

-- Mathematical conditions from the problem
def condition1 := a > 0
def condition2 := a ≠ 1
def condition3 (x : ℝ) := log a (2 * x - 3) + sqrt 2 = sqrt 2
def power_function := ∀ x, f x = x^a

-- The goal to prove
theorem prove_f_8_eq_2sqrt2 :
  (condition1 ∧ condition2 ∧ condition3 2 ∧ power_function) → f 8 = 2 * sqrt 2 :=
by
  intros h,
  sorry

end prove_f_8_eq_2sqrt2_l751_751466


namespace final_sequence_has_large_number_l751_751096

noncomputable def transform (a : Fin 25 → ℤ) : Fin 25 → ℤ :=
fun i => a i + a ((i + 1) % 25)

-- Initial sequence definition
def initial_seq : Fin 25 → ℤ
| i => if i < 13 then 1 else -1

-- Sequence after 100 iterations
def final_seq : Fin 25 → ℤ :=
(λ n => (iterate transform 100 initial_seq) n)

theorem final_sequence_has_large_number :
  ∃ n : Fin 25, final_seq n > 10^20 :=
sorry

end final_sequence_has_large_number_l751_751096


namespace prime_sums_count_eq_6_l751_751224

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function that computes the sum of the first k primes
noncomputable def sum_of_first_n_primes (n : ℕ) : ℕ :=
  (List.range n).map (fun i => Nat.prime (i + 1)).sum

theorem prime_sums_count_eq_6 : 
  (List.range 15).filter (fun i => is_prime (sum_of_first_n_primes (i+1))).length = 6 :=
by
  sorry

end prime_sums_count_eq_6_l751_751224


namespace solution_inequality_l751_751075

theorem solution_inequality (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0)
    (h : -q / p > -q' / p') : q / p < q' / p' :=
by
  sorry

end solution_inequality_l751_751075


namespace total_biked_distance_in_a_week_l751_751613

noncomputable def onur_speed : ℝ := 35
noncomputable def onur_duration : ℝ := 6
noncomputable def hanil_speed : ℝ := 45
noncomputable def additional_distance : ℝ := 40

def onur_daily_distance : ℝ := onur_speed * onur_duration
def onur_weekly_distance : ℝ := onur_daily_distance * 5
def hanil_daily_distance : ℝ := onur_daily_distance + additional_distance
def hanil_weekly_distance : ℝ := hanil_daily_distance * 3 

def total_weekly_distance : ℝ := onur_weekly_distance + hanil_weekly_distance

theorem total_biked_distance_in_a_week : total_weekly_distance = 1800 := by
  sorry

end total_biked_distance_in_a_week_l751_751613


namespace J_of_given_values_l751_751433

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_of_given_values : J 3 (-15) 10 = 49 / 30 := 
by 
  sorry

end J_of_given_values_l751_751433


namespace find_g2_l751_751648

variable {R : Type*} [Nonempty R] [Field R]

-- Define the function g
def g (x : R) : R := sorry

-- Given conditions
axiom condition1 : ∀ x y : R, x * g y = 2 * y * g x
axiom condition2 : g 10 = 5

-- The statement to be proved
theorem find_g2 : g 2 = 2 :=
by
  sorry

end find_g2_l751_751648


namespace part1_part2_l751_751607

-- Define the solution set M for the inequality
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- Define the problem conditions
variables {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M)

-- First part: Prove that |(1/3)a + (1/6)b| < 1/4
theorem part1 : |(1/3 : ℝ) * a + (1/6 : ℝ) * b| < 1/4 :=
sorry

-- Second part: Prove that |1 - 4 * a * b| > 2 * |a - b|
theorem part2 : |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end part1_part2_l751_751607


namespace find_annual_rate_l751_751748

noncomputable def compoundInterest (P A r : ℝ) (n t : ℕ) : ℝ := P * (1 + r / n)^(n * t)

theorem find_annual_rate
  (P A : ℝ) (r : ℝ) (t : ℕ) :
  P = 8000 →
  A = 9261.000000000002 →
  t = 3 →
  (compoundInterest P A r 1 t) = A →
  r = 0.05 :=
by
  intros P_eq A_eq t_eq compound_eq
  sorry

end find_annual_rate_l751_751748


namespace inequality_solution_set_inequality_solution_set_discussion_l751_751050

theorem inequality_solution_set (a : ℝ) :
  (∃ (x : ℝ), ax^2 + (1 - 2*a)*x - 2 > 0 ∧ (x < -1 ∨ x > 2)) ↔ a = 1 :=
sorry

theorem inequality_solution_set_discussion (a : ℝ) :
  (ax^2 + (1 - 2*a)*x - 2 > 0 → 
    ((a = 0 ∧ ∀ x, x > 2 → ax^2 + (1 - 2*a)*x - 2 > 0) ∨
     (a > 0 ∧ ∀ x, (x < -1/a ∨ x > 2) → ax^2 + (1 - 2*a)*x - 2 > 0) ∨
     (-1/2 < a ∧ a < 0 ∧ ∀ x, (2 < x ∧ x < -1/a) → ax^2 + (1 - 2*a)*x - 2 > 0) ∨
     (a = -1/2 ∧ ∀ x, false) ∨
     (a < -1/2 ∧ ∀ x, (-1/a < x ∧ x < 2) → ax^2 + (1 - 2*a)*x - 2 > 0)))
:= sorry

end inequality_solution_set_inequality_solution_set_discussion_l751_751050


namespace remainder_of_poly_div_l751_751423

-- Definitions based on conditions
def poly (r : ℕ) : ℕ := r^14 + r

-- Statement of the proof problem
theorem remainder_of_poly_div : 
  ∃ remainder, (poly 1 = remainder) ∧ (remainder = 2) := 
by 
  use 2
  split
  { rw [poly, pow_succ, pow_zero], simp }
  { refl }

end remainder_of_poly_div_l751_751423


namespace length_of_AB_l751_751770

theorem length_of_AB (b : ℝ) (h₁ : ∀ (x : ℝ), (x, -x^2) ∈ ({A : ℝ × ℝ | A.2 = -A.1^2}) → (x = -b ∨ x = b))
  (h₂ : ∀ (x : ℝ), (0, 0) = A)
  (h₃ : 0 = -b^2 → (b = 0))
  (h₄ : 2 * b * b^2 / 2 = 28) :
  b = real.cbrt 28 :=
by 
  sorry

end length_of_AB_l751_751770


namespace inequality_proof_l751_751287

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751287


namespace Gabriel_boxes_correct_l751_751947

noncomputable theory -- Lean may need this for real number calculations

variable (Stan_boxes: ℕ) (Joseph_boxes Jules_boxes John_boxes Martin_boxes Alice_boxes Gabriel_boxes: ℕ)

-- Conditions
variable (H_Stan : Stan_boxes = 120)
variable (H_Joseph : Joseph_boxes = (120 * 0.20))
variable (H_Jules : Jules_boxes = Joseph_boxes + 5)
variable (H_John : John_boxes = Nat.round (Jules_boxes * 1.20))
variable (H_Martin : Martin_boxes = Nat.round (Jules_boxes * 1.50))
variable (H_Alice : Alice_boxes = Nat.round (John_boxes * 0.75))
variable (H_Gabriel : Gabriel_boxes = Nat.round ((Martin_boxes + Alice_boxes) / 2))

-- Theorem to prove
theorem Gabriel_boxes_correct : Gabriel_boxes = 35 :=
by
  rw [H_Stan, H_Joseph, H_Jules, H_John, H_Martin, H_Alice, H_Gabriel]
  sorry -- Proof omitted

end Gabriel_boxes_correct_l751_751947


namespace num_paths_A_to_B_l751_751899

/--
There are exactly 12 continuous paths from A to B, along the segments of the figure, 
without revisiting any of the seven labeled points.
-/
theorem num_paths_A_to_B : 
  ∃ paths : Finset (List Char), 
  ∀ p ∈ paths, p.head = 'A' ∧ p.getLast sorry = 'B' ∧ p.Nodup ∧
  #paths = 12 := sorry

end num_paths_A_to_B_l751_751899


namespace probability_A_70_out_of_243_is_approx_0_0231_l751_751828

noncomputable def probability_event_A_exactly_k_times_in_n_trials (n k : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - p
  let np := n * p
  let nq := n * q
  let sqrt_npq := Real.sqrt (n * p * q)
  let x := (k - np) / sqrt_npq
  let φ := (Real.exp (-0.5 * x ^ 2)) / (Real.sqrt (2 * Real.pi))
  in (1 / sqrt_npq) * φ

theorem probability_A_70_out_of_243_is_approx_0_0231 :
  probability_event_A_exactly_k_times_in_n_trials 243 70 0.25 ≈ 0.0231 := 
by {
  -- Here we'd write the full proof, but we'll leave it as a sorry for now
  sorry
}

end probability_A_70_out_of_243_is_approx_0_0231_l751_751828


namespace problem_B_problem_D_l751_751445

noncomputable def z : ℂ := (2 * complex.I) / (real.sqrt 3 + complex.I)
noncomputable def z_conjugate : ℂ := conj z

theorem problem_B :
  complex.abs z_conjugate = 1 :=
sorry

theorem problem_D : 
  z_conjugate.re > 0 ∧ z_conjugate.im < 0 :=
sorry

end problem_B_problem_D_l751_751445


namespace solve_inequalities_l751_751721

-- Conditions from the problem
def condition1 (x y : ℝ) : Prop := 13 * x ^ 2 - 4 * x * y + 4 * y ^ 2 ≤ 2
def condition2 (x y : ℝ) : Prop := 2 * x - 4 * y ≤ -3

-- Given answers from the solution
def solution_x : ℝ := -1/3
def solution_y : ℝ := 2/3

-- Translate the proof problem in Lean
theorem solve_inequalities : condition1 solution_x solution_y ∧ condition2 solution_x solution_y :=
by
  -- Here you will provide the proof.
  sorry

end solve_inequalities_l751_751721


namespace inequality_proof_l751_751285

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751285


namespace sin_B_value_in_triangle_l751_751552

variable {A B C : ℝ}

theorem sin_B_value_in_triangle
    (h1 : sin C - cos C + sin B = sqrt 3)
    (h2 : A + B + C = π)
    (h3 : A ≥ π / 4) :
    sin B = sqrt 6 / 3 :=
sorry

end sin_B_value_in_triangle_l751_751552


namespace seeds_first_plot_germination_percentage_l751_751428

-- Definitions of conditions
def total_seeds : ℕ := 300 + 200

def seeds_second_plot_germinated : ℕ := 0.30 * 200

def total_germinated : ℕ := 0.27 * total_seeds

def seeds_first_plot_germinated : ℕ := total_germinated - seeds_second_plot_germinated

-- Statement to prove
theorem seeds_first_plot_germination_percentage :
  (seeds_first_plot_germinated / 300) * 100 = 25 := by
sorry

end seeds_first_plot_germination_percentage_l751_751428


namespace inequality_ge_one_l751_751313

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l751_751313


namespace possible_to_place_12_numbers_on_cube_edges_l751_751555

-- Define a list of numbers from 1 to 12
def nums := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the faces of the cube in terms of the indices of nums list
def top_face := [1, 2, 9, 10]
def bottom_face := [3, 5, 6, 8]

-- Define the product of the numbers on the faces of the cube
def product_face (face : List Nat) : Nat := face.foldr (*) 1

-- The lean statement proving the problem
theorem possible_to_place_12_numbers_on_cube_edges :
  product_face (top_face.map (λ i => nums.get! (i - 1))) =
  product_face (bottom_face.map (λ i => nums.get! (i - 1))) :=
by
  sorry

end possible_to_place_12_numbers_on_cube_edges_l751_751555


namespace arithmetic_sequence_properties_l751_751959

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : a 2 = 11)
  (h₂ : S 10 = 40)
  (h₃ : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) :
  (∀ n, a n = -2 * n + 15) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 7 → ∑ i in finset.range n, abs (a (i + 1)) = -n^2 + 14*n) ∧
  (∀ n, n ≥ 8 → ∑ i in finset.range n, abs (a (i + 1)) = n^2 - 14*n + 98) :=
by
  sorry  -- sorry to skip the proof

end arithmetic_sequence_properties_l751_751959


namespace balanced_mass_equivalence_l751_751663

theorem balanced_mass_equivalence (bigcirc nablah square : ℝ) 
  (h1 : 3 * bigcirc = 2 * nablah) 
  (h2 : square + bigcirc + nablah = 2 * square) : 
  bigcirc + 3 * nablah = 11 * nablah / 3 :=
by 
  have h3 : bigcirc = 2 * nablah / 3 := by { linarith }
  have square_eq : square = 5 * nablah / 3 := by { rw [h3] at h2, linarith }
  rw [h3]
  linarith
  sorry

end balanced_mass_equivalence_l751_751663


namespace three_term_inequality_l751_751294

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l751_751294


namespace replacement_problem_l751_751485

-- Define the given number string with 100 zeros
def given_number : ℕ := 530000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000035

-- Conditions for divisibility
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0
def divisible_by_9 (n : ℕ) : Prop := n.digits.sum % 9 = 0
def divisible_by_11 (n : ℕ) : Prop := (n.digits.enum.filter (λ x, x.snd % 2 = 0)).sum - (n.digits.enum.filter (λ x, x.snd % 2 = 1)).sum % 11 = 0

-- Final statement
theorem replacement_problem : 
  ∃ (a b : ℕ), 
    (1 ≤ a ∧ a ≤ 9) ∧ 
    (1 ≤ b ∧ b ≤ 9) ∧ 
    (divisible_by_5 (replace_two_zeros given_number a b)) ∧ 
    (divisible_by_9 (replace_two_zeros given_number a b)) ∧ 
    (divisible_by_11 (replace_two_zeros given_number a b)) ∧ 
    (22100 ways to replace zeros to satisfy these conditions) := 
sorry

end replacement_problem_l751_751485


namespace problem_1_problem_2_l751_751468

variables {Plane Line : Type*} [LinearAlgebra.Plane Line]

-- Definition of parallel lines
def parallel (a b : Line) : Prop := sorry

-- Definition of line subset of plane
def subset (c : Line) (α : Plane) : Prop := sorry

-- Given problem (1)
theorem problem_1 (a b c : Line) (α : Plane) 
  (h1 : parallel a b) 
  (h2 : parallel b c) 
  (h3 : subset c α) 
  (h4 : ¬ subset a α) : 
  parallel a α := 
sorry

-- Given problem (2)
theorem problem_2 (a b : Line) (α β : Plane) 
  (h1 : α ∩ β = b) 
  (h2 : parallel a b) 
  (h3 : subset a β) : 
  parallel a α := 
sorry

end problem_1_problem_2_l751_751468


namespace find_a9_l751_751929

variable (a : ℕ → ℝ)

theorem find_a9 (h1 : a 4 - a 2 = -2) (h2 : a 7 = -3) : a 9 = -5 :=
sorry

end find_a9_l751_751929


namespace range_of_f3_l751_751478

def f (a c x : ℝ) : ℝ := a * x^2 - c

theorem range_of_f3 (a c : ℝ)
  (h1 : -4 ≤ f a c 1 ∧ f a c 1 ≤ -1)
  (h2 : -1 ≤ f a c 2 ∧ f a c 2 ≤ 5) :
  -1 ≤ f a c 3 ∧ f a c 3 ≤ 20 := 
sorry

end range_of_f3_l751_751478


namespace domain_of_function_l751_751209

theorem domain_of_function :
  {x : ℝ | (sqrt ((x - 1) / (2 * x)) - log 2 (4 - x^2)).is_defined} =
  (set.Ioo (-2) 0) ∪ (set.Ico 1 2) :=
by
  sorry

end domain_of_function_l751_751209


namespace smallest_prime_factor_set_C_l751_751625

def smallest_prime_factor (n : ℕ) : ℕ :=
  Nat.find (λ p, p.Prime ∧ p ∣ n)

def number_with_smallest_prime_factor (s : Set ℕ) : ℕ :=
  s.toFinset.min' sorry -- We assume the set is non-empty for now.

def C : Set ℕ := {67, 71, 73, 76, 85}

theorem smallest_prime_factor_set_C : number_with_smallest_prime_factor C = 76 :=
by 
  sorry

end smallest_prime_factor_set_C_l751_751625


namespace solve_cryptarithm_l751_751633

-- Definitions for the cryptarithm problem
def valid_digits (T O K : ℕ) : Prop :=
  T ≠ O ∧ O ≠ K ∧ T ≠ K ∧ 
  1 ≤ T ∧ T ≤ 9 ∧
  0 ≤ O ∧ O ≤ 9 ∧
  1 ≤ K ∧ K ≤ 9

def cryptarithm_solution (T O K : ℕ) (TOK KOT KTO : ℕ) : Prop :=
  TOK = 100 * T + 10 * O + K ∧
  KOT = 100 * K + 10 * O + T ∧
  KTO = 100 * K + 10 * T + O ∧
  TOK = KOT + KTO

theorem solve_cryptarithm : 
  ∃ (T O K : ℕ), valid_digits T O K ∧ cryptarithm_solution T O K 954 459 495 :=
begin
  use [9, 5, 4],    -- Substituting the solution values
  split,            -- Splitting the conjunction into parts, first proving valid_digits
  {
    -- Prove the digits are valid as per conditions
    repeat { split }, -- Split all conditions in valid_digits
    -- Disjointness of digits
    exact dec_trivial,
    exact dec_trivial,
    exact dec_trivial,
    -- T is a non-zero digit
    exact dec_trivial,
    exact dec_trivial,
    -- O is a digit
    exact dec_trivial,
    exact dec_trivial,
    -- K is a non-zero digit
    exact dec_trivial,
    exact dec_trivial,
  },
  {
    -- Now, proving the cryptarithm equation for given TOK, KOT, KTO
    unfold cryptarithm_solution,
    split,
    exact dec_trivial,
    split,
    exact dec_trivial,
    split,
    exact dec_trivial,
    exact dec_trivial,
  },
  sorry
end

end solve_cryptarithm_l751_751633


namespace rectangle_solution_l751_751033

theorem rectangle_solution (x : ℝ) :
  let l := 4 * x,
      w := x + 4 in
  l * w = 2 * l + 2 * w →
  x = (-3 + Real.sqrt 41) / 4 :=
by
  intros l w h
  sorry

end rectangle_solution_l751_751033


namespace edge_length_of_cube_l751_751943

theorem edge_length_of_cube (a : ℝ) (h1 : ∃ r : ℝ, 4 * π * r^2 = 16 * π ∧ r = (√3 / 2) * a) : a = 4 * √3 / 3 := 
      sorry

end edge_length_of_cube_l751_751943


namespace count_divisibles_l751_751066

open Nat

theorem count_divisibles (n : ℕ) (h : n < 1000) :
  ∃ m : ℕ, m = 4 ∧ (∀ k ∈ (Finset.range 1000).filter (λ x, x % 2 = 0 ∧ x % 3 = 0 ∧ x % 5 = 0 ∧ x % 7 = 0), k = m) := sorry

end count_divisibles_l751_751066


namespace inequality_proof_l751_751279

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751279


namespace cube_edge_numbers_equal_top_bottom_l751_751563

theorem cube_edge_numbers_equal_top_bottom (
  numbers : List ℕ,
  h : numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) :
  ∃ (top bottom : List ℕ),
    (∀ x, x ∈ top → x ∈ numbers) ∧
    (∀ x, x ∈ bottom → x ∈ numbers) ∧
    (top ≠ bottom) ∧
    (top.length = 4) ∧ 
    (bottom.length = 4) ∧ 
    (top.product = bottom.product) :=
begin
  sorry
end

end cube_edge_numbers_equal_top_bottom_l751_751563


namespace smallest_value_e_l751_751222

noncomputable def polynomial_min_positive_e (a b c d e : ℤ) : Prop :=
  ∃ (roots : List ℚ), 
  roots = [-2, 5, 9, -1/3] ∧
  (∀ x ∈ roots, is_int x) ∧
  (ax^4 + bx^3 + cx^2 + dx + e = 0) ∧
  a ≠ 0 ∧  
  e > 0 

theorem smallest_value_e (a b c d e : ℤ): polynomial_min_positive_e a b c d e → e = 90 :=
by sorry

end smallest_value_e_l751_751222


namespace inequality_preservation_l751_751073

theorem inequality_preservation (a b x : ℝ) (h : a > b) : a * 2^x > b * 2^x :=
sorry

end inequality_preservation_l751_751073


namespace parity_difference_l751_751596

-- Define f(n) as the sum of numerator of reduced fractions 1/n, 2/n, ..., (n-1)/n
def f (n : ℕ) : ℕ := (∑ k in finset.range n, (nat.gcd k n = 1))

-- Prove the statement:
theorem parity_difference (n : ℕ) (h : 1 < n) : (f n % 2) ≠ (f (2015 * n) % 2) :=
begin
  sorry
end

end parity_difference_l751_751596


namespace limit_one_sided_0_pos_limit_one_sided_0_neg_limit_two_sided_0_does_not_exist_l751_751375

noncomputable theory

open Filter
open Topology
open Real

theorem limit_one_sided_0_pos :
  tendsto (λ x : ℝ, 1 / (1 + 3^(2 / x))) (nhds_within 0 (set.Ioi 0)) (nhds 0) :=
begin
  -- proof goes here
  sorry
end

theorem limit_one_sided_0_neg :
  tendsto (λ x : ℝ, 1 / (1 + 3^(2 / x))) (nhds_within 0 (set.Iio 0)) (nhds 1) :=
begin
  -- proof goes here
  sorry
end

theorem limit_two_sided_0_does_not_exist :
  ¬ tendsto (λ x : ℝ, 1 / (1 + 3^(2 / x))) (nhds 0) (nhds 0) :=
begin
  -- proof goes here
  sorry
end

end limit_one_sided_0_pos_limit_one_sided_0_neg_limit_two_sided_0_does_not_exist_l751_751375


namespace percentage_discount_on_pencils_l751_751942

-- Establish the given conditions
variable (cucumbers pencils price_per_cucumber price_per_pencil total_spent : ℕ)
variable (h1 : cucumbers = 100)
variable (h2 : price_per_cucumber = 20)
variable (h3 : price_per_pencil = 20)
variable (h4 : total_spent = 2800)
variable (h5 : cucumbers = 2 * pencils)

-- Propose the statement to be proved
theorem percentage_discount_on_pencils : 20 * pencils * price_per_pencil = 20 * (total_spent - cucumbers * price_per_cucumber) ∧ pencils = 50 ∧ ((total_spent - cucumbers * price_per_cucumber) * 100 = 80 * pencils * price_per_pencil) :=
by
  sorry

end percentage_discount_on_pencils_l751_751942


namespace sum_of_divisors_of_10_l751_751833

theorem sum_of_divisors_of_10 :
  ∑ k in {1, 2, 5, 10}, k = 18 :=
by
  sorry

end sum_of_divisors_of_10_l751_751833


namespace prime_divisor_congruent_mod_p_l751_751971

theorem prime_divisor_congruent_mod_p (p : ℕ) (n : ℤ) (hp : Nat.Prime p) :
  ∀ q : ℕ, Nat.Prime q → q ∣ ((n + 1)^p - n^p) → q ≡ 1 [MOD p] :=
by
  intro q hq hdiv
  sorry

end prime_divisor_congruent_mod_p_l751_751971


namespace circle_properties_l751_751042

-- Given curve C and line l
def curve (x y m : ℝ) := x^2 + y^2 - 2 * x - 4 * y + m = 0
def line (x y : ℝ) := x + 2 * y - 4 = 0

-- Distance from point (a, b) to line ax + by + c = 0
def distance_to_line (a b c : ℝ) (x y : ℝ) : ℝ :=
  (abs (a * x + b * y + c)) / sqrt (a^2 + b^2)

-- Center and radius of the circle
def center_of_circle (x y m : ℝ) : bool :=
  x = 1 ∧ y = 2 ∧ m = -6

def radius_of_circle (r : ℝ) : bool :=
  r = sqrt 11

-- Intersect condition to solve for m
def intersect_condition (m : ℝ) : bool :=
  let r := sqrt (5 - m) in
  let d := 1 / sqrt 5 in
  r^2 + d^2 = 5 - m ∧ (4 / sqrt 5)^2 = (2 / sqrt 5)^2 + (1 / sqrt 5)^2 ∧ m = 4

-- Main theorem statement
theorem circle_properties :
  (∀ x y, curve x y (-6) →
    center_of_circle x y (-6) ∧ radius_of_circle (sqrt 11)) ∧
  (∀ m, intersect_condition m → m = 4) :=
by {
  sorry
}

end circle_properties_l751_751042


namespace min_value_l751_751601

-- Given: Polynomial definition
def f (x : ℝ) := x^4 + 14 * x^3 + 52 * x^2 + 56 * x + 16

-- Given: Roots of the polynomial
variables (z1 z2 z3 z4 : ℝ)
-- Assuming z1, z2, z3, and z4 are roots of f.
axiom h_roots : f z1 = 0 ∧ f z2 = 0 ∧ f z3 = 0 ∧ f z4 = 0

-- Problem: Find the minimum value of |za * zb + zc * zd| where {a, b, c, d} = {1, 2, 3, 4}
-- Prove: the smallest possible value is 8.
theorem min_value : ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  {|za | zb + |zc | zd|}.{a, b, c, d} = {1, 2, 3, 4} ∧ |za | zb + |zc | zd| = 8 :=
sorry

end min_value_l751_751601


namespace cube_face_product_l751_751572

open Finset

theorem cube_face_product (numbers : Finset ℕ) (hs : numbers = range (12 + 1)) :
  ∃ top_face bottom_face : Finset ℕ,
    top_face.card = 4 ∧
    bottom_face.card = 4 ∧
    (numbers \ (top_face ∪ bottom_face)).card = 4 ∧
    (∏ x in top_face, x) = (∏ x in bottom_face, x) :=
by
  use {2, 4, 9, 10}
  use {3, 5, 6, 8}
  repeat { split };
  -- Check cardinality conditions
  sorry

end cube_face_product_l751_751572


namespace sum_f_k_l751_751867

def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom condition1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom condition2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_at_2 : g 2 = 4

theorem sum_f_k : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
by sorry

end sum_f_k_l751_751867


namespace moment_of_inertia_unit_masses_l751_751715

variable (n : ℕ) (a : Fin n → Fin n → ℝ)

theorem moment_of_inertia_unit_masses :
  let I_O := 1 / n * ∑ (i : Fin n) (j : Fin n) (h : i.1 < j.1), a i j ^ 2
  in I_O = 1 / n * ∑ (i : Fin n) (j : Fin n), a i j ^ 2 := sorry

end moment_of_inertia_unit_masses_l751_751715


namespace distance_from_F_to_midpoint_l751_751100

-- Definitions of the given problem
variable (DE DF EF : ℝ)
variable (F : {x : ℝ × ℝ // x.1 ^ 2 + x.2 ^ 2 = DF ^ 2})
def is_right_triangle (DE DF EF : ℝ) : Prop := (DE ^ 2 + DF ^ 2 = EF ^ 2)

theorem distance_from_F_to_midpoint (h : is_right_triangle DE DF EF) (hDE : DE = 15) (hDF : DF = 9) (hEF : EF = 12) :
  (DE / 2 - sqrt (((DF ^ 2) / 2)^2 - (DF / 2) ^ 2) = 6) :=
by
  sorry

end distance_from_F_to_midpoint_l751_751100


namespace total_unique_plants_l751_751437

noncomputable def bed_A : ℕ := 600
noncomputable def bed_B : ℕ := 550
noncomputable def bed_C : ℕ := 400
noncomputable def bed_D : ℕ := 300

noncomputable def intersection_A_B : ℕ := 75
noncomputable def intersection_A_C : ℕ := 125
noncomputable def intersection_B_D : ℕ := 50
noncomputable def intersection_A_B_C : ℕ := 25

theorem total_unique_plants : 
  bed_A + bed_B + bed_C + bed_D - intersection_A_B - intersection_A_C - intersection_B_D + intersection_A_B_C = 1625 := 
by
  sorry

end total_unique_plants_l751_751437


namespace gloria_turtle_time_l751_751895

theorem gloria_turtle_time (g_time : ℕ) (george_time : ℕ) (gloria_time : ℕ) 
  (h1 : g_time = 6) 
  (h2 : george_time = g_time - 2)
  (h3 : gloria_time = 2 * george_time) : 
  gloria_time = 8 :=
sorry

end gloria_turtle_time_l751_751895


namespace total_marbles_l751_751525

theorem total_marbles (b : ℕ) : 
  let r := 1.30 * b
  let g := 1.70 * r
  b + r + g = 4.51 * b :=
by
  sorry

end total_marbles_l751_751525


namespace problem_statements_l751_751393

open Classical
noncomputable theory

-- Definitions for the geometric series
def a : ℝ := 3
def r : ℝ := 1 / 4
def S : ℝ := a / (1 - r)

-- Lean statements corresponding to each assertion in the problem
theorem problem_statements :
  (S ≤ 10) ∧                            -- statement 1
  (S < 5) ∧                             -- statement 2
  (∀ ε > 0, ∃ n : ℕ, abs (a * r ^ n) < ε) ∧  -- statement 3
  (¬(∀ ε > 0, abs (S - 4) < ε)) ∧            -- statement 4
  (∃ L, L = S)                            -- statement 5
  :=
by
  -- proof here
  sorry

end problem_statements_l751_751393


namespace incenter_to_circumcenter_same_circumcircle_l751_751953

variable {P : Type} [euclidean_geometry P]
variable (A B C I A' B' C' : P) (circumcenter Of ℝ : P → P → P → P)

theorem incenter_to_circumcenter_same_circumcircle
  (h_incenter : incenter I A B C)
  (h_circum1 : circumcenter A' I B C)
  (h_circum2 : circumcenter B' I A C)
  (h_circum3 : circumcenter C' I A B) :
  circle I A B C = circle A' B' C' :=
sorry

end incenter_to_circumcenter_same_circumcircle_l751_751953


namespace mixed_solution_salt_percentage_l751_751338

variable (V1 V2 Vm : ℕ) (C1 C2 Cm : ℝ)

theorem mixed_solution_salt_percentage :
  V1 = 600 → C1 = 0.03 →
  V2 = 400 → C2 = 0.12 →
  Vm = 1000 → 
  Cm = ((C1 * V1) + (C2 * V2)) / Vm →
  Cm * 100 = 6.6 :=
by
  intros hV1 hC1 hV2 hC2 hVm hCm
  rw [hV1, hC1, hV2, hC2, hVm] at hCm
  have hSalt : (C1 * V1 + C2 * V2) = 66 := by
    rw [←hV1, ←hC1, ←hV2, ←hC2]
    norm_num
  rw hSalt at hCm
  norm_num at hCm
  sorry

end mixed_solution_salt_percentage_l751_751338


namespace complex_solution_l751_751409

theorem complex_solution (a b : ℂ) (ha : a ≠ 0) (hab : a + b ≠ 0) (h : (a + b) / a = 3 * b / (a + b)) : 
  ¬ (a ∈ ℝ) ∨ ¬ (b ∈ ℝ) := 
by
  sorry

end complex_solution_l751_751409


namespace min_value_of_sum_l751_751850

theorem min_value_of_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2 * a + b) : a + b ≥ 2 * Real.sqrt 2 + 3 :=
sorry

end min_value_of_sum_l751_751850


namespace find_angle_CLA1_l751_751097

variables {A B C A1 B1 C1 T O K L : Type*}
variables {triangle_ABC : Type*}

-- Assume the existence of points and properties as per the conditions:
axiom triangle_ABC_is_acute : is_acute_triangle A B C
axiom altitudes_AA1_BB1_CC1 : are_altitudes A A1 B B1 C C1
axiom tangents_TA_TB : is_tangent TA T (circumcircle A B C) ∧ is_tangent TB T (circumcircle A B C)
axiom center_O : is_circumcenter O A B C
axiom perpendicular_T_to_A1B1_at_K : ∃ K, is_perpendicular (line T K) (line A1 B1) ∧ K ∈ (line C C1)
axiom parallel_through_C1_parallel_to_OK : ∃ L, L ∈ (segment C O) ∧ is_parallel (line C1 L) (line O K)

-- State the goal angle:
theorem find_angle_CLA1 : angle C L A1 = 90 :=
sorry

end find_angle_CLA1_l751_751097


namespace perp_lines_solution_l751_751060

theorem perp_lines_solution (a : ℝ) (l₁ : ℝ → ℝ → ℝ := λ x y, a * x + 2 * y + 1) (l₂ : ℝ → ℝ → ℝ := λ x y, (3 - a) * x - y + a) :
  (∀ x y : ℝ, l₁ x y = 0 ∧ l₂ x y = 0 → ( (3 - a) * (-a / 2) = -1 )) → (a = 1 ∨ a = 2) :=
by
  sorry

end perp_lines_solution_l751_751060


namespace sum_of_numbers_l751_751507

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l751_751507


namespace proof_problem_l751_751454

open Real

noncomputable def p : Prop := ∃ x : ℝ, x - 2 > log x / log 10
noncomputable def q : Prop := ∀ x : ℝ, x^2 > 0

theorem proof_problem :
  (p ∧ ¬q) := by
  sorry

end proof_problem_l751_751454


namespace jail_time_weeks_l751_751240

theorem jail_time_weeks (days_protest : ℕ) (cities : ℕ) (arrests_per_day : ℕ)
  (days_pre_trial : ℕ) (half_week_sentence_days : ℕ) :
  days_protest = 30 →
  cities = 21 →
  arrests_per_day = 10 →
  days_pre_trial = 4 →
  half_week_sentence_days = 7 →
  (21 * 30 * 10 * (4 + 7)) / 7 = 9900 :=
by
  intros h_days_protest h_cities h_arrests_per_day h_days_pre_trial h_half_week_sentence_days
  rw [h_days_protest, h_cities, h_arrests_per_day, h_days_pre_trial, h_half_week_sentence_days]
  exact sorry

end jail_time_weeks_l751_751240


namespace find_k_squared_l751_751051

noncomputable def line_eq (k x : ℝ) : ℝ := k * x + 4

def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

def point_A (x1 y1 : ℝ) : Prop := hyperbola_eq x1 y1 ∧ y1 = line_eq k x1

def point_B (x2 y2 : ℝ) : Prop := hyperbola_eq x2 y2 ∧ y2 = line_eq k x2

def point_Q (k : ℝ) : ℝ × ℝ := (- 4 / k, 0)

def point_P : ℝ × ℝ := (0, 4)

def condition_lambdas (k x1 x2 : ℝ) (λ1 λ2 : ℝ) : Prop :=
(λ1 = -4 / (4 + k * x1)) ∧ (λ2 = -4 / (4 + k * x2)) ∧ (λ1 + λ2 = -8 / 3)

theorem find_k_squared (k : ℝ) (h : k ≠ 4 ∧ k ≠ -4) (x1 x2 y1 y2 λ1 λ2 : ℝ) :
  (point_A x1 y1) ∧ (point_B x2 y2) ∧ (condition_lambdas k x1 x2 λ1 λ2) → k^2 = 4 :=
  sorry

end find_k_squared_l751_751051


namespace solve_inequality_l751_751191

theorem solve_inequality 
  (x : ℝ) :
  (\{x \mid \frac{(x - 3) * (x - 5) * (x - 7)}{(x - 2) * (x - 6) * (x - 8)} > 0\} =
  {x | x < 2} ∪ {x | 3 < x ∧ x < 5} ∪ {x | 6 < x ∧ x < 7}  ∪ {x | 8 < x}) :=
sorry

end solve_inequality_l751_751191


namespace integer_solutions_to_cube_sum_eq_2_pow_30_l751_751417

theorem integer_solutions_to_cube_sum_eq_2_pow_30 (x y : ℤ) :
  x^3 + y^3 = 2^30 → (x = 0 ∧ y = 2^10) ∨ (x = 2^10 ∧ y = 0) :=
by
  sorry

end integer_solutions_to_cube_sum_eq_2_pow_30_l751_751417


namespace average_of_remaining_numbers_l751_751203

theorem average_of_remaining_numbers (n : ℕ) (avg_original : ℚ) (discard1 discard2 : ℚ) 
  (h_n : n = 50) (h_avg_original : avg_original = 62) (h_discard1 : discard1 = 45)
  (h_discard2 : discard2 = 55) : 
let sum_original := avg_original * n in
let sum_discarded := discard1 + discard2 in
let sum_remaining := sum_original - sum_discarded in
let new_n := n - 2 in
let new_avg := sum_remaining / new_n in
new_avg = 62.5 := 
by
  sorry

end average_of_remaining_numbers_l751_751203


namespace tangency_condition_and_point_l751_751255

variable (a b p q : ℝ)

/-- Condition for the line y = px + q to be tangent to the ellipse b^2 x^2 + a^2 y^2 = a^2 b^2. -/
theorem tangency_condition_and_point
  (h_cond : a^2 * p^2 + b^2 - q^2 = 0)
  : 
  ∃ (x₀ y₀ : ℝ), 
  x₀ = - (a^2 * p) / q ∧
  y₀ = b^2 / q ∧ 
  (b^2 * x₀^2 + a^2 * y₀^2 = a^2 * b^2 ∧ y₀ = p * x₀ + q) :=
sorry

end tangency_condition_and_point_l751_751255


namespace problem_statement_l751_751858

theorem problem_statement {a b c d : ℝ} (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (4 / (b - c)) + (9 / (c - d)) ≥ (36 / (a - d)) :=
by
  sorry -- proof is omitted according to the instructions

end problem_statement_l751_751858


namespace no_suitable_rectangle_exists_l751_751977

theorem no_suitable_rectangle_exists :
  ∀ (length width : ℝ), length * width = 30 ∧ length = 2 * width ∧ length <= 6 → false :=
begin
  intros length width h,
  sorry,
end

end no_suitable_rectangle_exists_l751_751977


namespace circumcircle_nine_point_circle_intersect_at_right_angle_l751_751201

-- Define the main problem condition
variables {A B C : ℝ} -- Angles in a triangle

-- The condition given in the problem
axiom sin_squared_sum : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 1

-- Main theorem to be proved
theorem circumcircle_nine_point_circle_intersect_at_right_angle
  (h : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 1) : 
  -- Prove that the circumcircle and nine-point circle intersect at a right angle
  ⊢ "The circumcircle and the nine-point circle intersect at a right angle" sorry

end circumcircle_nine_point_circle_intersect_at_right_angle_l751_751201


namespace custom_op_3_7_l751_751909

-- Define the custom operation (a # b)
def custom_op (a b : ℕ) : ℕ := a * b - b + b^2

-- State the theorem that proves the result
theorem custom_op_3_7 : custom_op 3 7 = 63 := by
  sorry

end custom_op_3_7_l751_751909


namespace sum_of_squares_l751_751499

theorem sum_of_squares (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 512 * x ^ 3 + 125 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 6410 := 
sorry

end sum_of_squares_l751_751499


namespace min_dist_point_outside_convex_hull_l751_751453

theorem min_dist_point_outside_convex_hull (n : ℕ) (h : n > 4) :
  ∃ (A : fin n → euclidean_space ℝ (fin 3)) (P : euclidean_space ℝ (fin 3)),
    (P = ⟨0, 0, 0⟩) ∧
    (∀ Q : euclidean_space ℝ (fin 3), (∑ i, (| Q - A i |)) ≥ (∑ i, (| P - A i |))) ∧ 
    (P ∉ convex_hull (set.range A)) :=
begin
  sorry
end

end min_dist_point_outside_convex_hull_l751_751453


namespace range_of_m2_plus_n2_l751_751593

theorem range_of_m2_plus_n2 
  (f : ℝ → ℝ)
  (monotone_f : ∀ {x y : ℝ}, x ≤ y → f(x) ≤ f(y))
  (antisymmetric_f : ∀ x : ℝ, f(-x) + f(x) = 0)
  (m n : ℝ)
  (h : f(m^2 - 6 * m + 21) + f(n^2 - 8 * n) < 0) : 
  9 < m^2 + n^2 ∧ m^2 + n^2 < 49 :=
sorry

end range_of_m2_plus_n2_l751_751593


namespace probability_at_least_one_expired_l751_751771

theorem probability_at_least_one_expired (total_bottles : ℕ) (expired_bottles : ℕ)
  (selection_size : ℕ) (prob_both_unexpired : ℚ) :
  total_bottles = 30 →
  expired_bottles = 3 →
  selection_size = 2 →
  prob_both_unexpired = 351 / 435 →
  (1 - prob_both_unexpired) = 28 / 145 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end probability_at_least_one_expired_l751_751771


namespace sum_of_eight_numbers_l751_751517

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l751_751517


namespace line_L_equal_angles_with_AB_and_CD_l751_751449

-- Define the triangle ABC and its circumcircle center O.
variable {ABC : Triangle}
variable {O : Point} (hO : O = Circumcenter ABC)

-- Define the points D and X with their respective projections.
variable {D X : Point}
variable {l L : Line}

-- Conditions: lines l parallel to XO.
variable (hl : LineParallel l (LineThrough X O))

-- Define the goal: L forms equal angles with lines AB and CD.
theorem line_L_equal_angles_with_AB_and_CD (hABC : IsTriangle ABC)
  (hXD : LineProj D ABC = l)
  (hX_L : LineProj X ABC = L)
  (heq : LineParallel l (LineThrough X O)) :
  (AngleBetweenLines L (LineThrough (A : Point) (B : Point))) = 
  (AngleBetweenLines L (LineThrough (C : Point) (D : Point))) :=
sorry

end line_L_equal_angles_with_AB_and_CD_l751_751449


namespace points_coplanar_l751_751907

theorem points_coplanar
  {O A B C P : Type*}
  [AddCommGroup O] [Module ℝ O]
  (h1 : ¬ collinear ℝ ({A, B, C} : Set O))
  (h2 : ∀ O : O, (P : O) = (3 / 4 : ℝ) • (A : O) + (1 / 8 : ℝ) • (B : O) + (1 / 8 : ℝ) • (C : O)) :
  coplanar ℝ ({P, A, B, C} : Set O) :=
sorry

end points_coplanar_l751_751907


namespace modulus_condition_l751_751067

noncomputable def a : ℝ := -1/2
noncomputable def b : ℝ := -1
def i : ℂ := complex.I -- Define the imaginary unit

-- Condition: (1 + 2a*i)*i = 1 - b*i
lemma condition : (1 + 2*a*i)*i = 1 - b*i := by
  -- Expansion and simplification involved (given in the solution)
  have h1 : (1 : ℂ) + 2*a*i = 1 + 2*(-1/2)*i := by rw [show a = -1/2 by sorry]
  simp [h1, i, complex.I_re, complex.I_im]

-- The goal to be proved
theorem modulus_condition : complex.abs (a + b*i) = real.sqrt (5) / 2 := by
  rw [complex.abs_def]
  have h_a : a = -1/2 := by sorry
  have h_b : b = -1 := by sorry
  rw [h_a, h_b]
  simp [complex.abs, complex.norm_sq, complex.I_re, complex.I_im]
  norm_num

end modulus_condition_l751_751067


namespace starfish_cannot_be_determined_l751_751126

-- Conditions
def initial_seashells : Nat := 49
def seashells_given : Nat := 13
def final_seashells : Nat := 36
def starfish_found : Nat -- This is what we need to determine.

-- Question: Prove that the number of starfish Jason found cannot be determined from the given conditions.
theorem starfish_cannot_be_determined (a b c : Nat) (H1 : a = 49) (H2 : b = 13) (H3 : c = 36) : ∃ n : Nat, (initial_seashells - seashells_given = final_seashells) ∧ (n ≠ starfish_found) :=
by {
  sorry
}

end starfish_cannot_be_determined_l751_751126


namespace inequality_proof_l751_751281

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751281


namespace possible_to_place_12_numbers_on_cube_edges_l751_751557

-- Define a list of numbers from 1 to 12
def nums := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the faces of the cube in terms of the indices of nums list
def top_face := [1, 2, 9, 10]
def bottom_face := [3, 5, 6, 8]

-- Define the product of the numbers on the faces of the cube
def product_face (face : List Nat) : Nat := face.foldr (*) 1

-- The lean statement proving the problem
theorem possible_to_place_12_numbers_on_cube_edges :
  product_face (top_face.map (λ i => nums.get! (i - 1))) =
  product_face (bottom_face.map (λ i => nums.get! (i - 1))) :=
by
  sorry

end possible_to_place_12_numbers_on_cube_edges_l751_751557


namespace car_highway_mileage_l751_751377

theorem car_highway_mileage
  (city_mpg : ℕ)
  (city_distance : ℕ)
  (highway_distance : ℕ)
  (gas_cost_per_gallon : ℕ)
  (total_spent : ℕ) :
  city_mpg = 30 → 
  city_distance = 60 → 
  highway_distance = 200 → 
  gas_cost_per_gallon = 3 → 
  total_spent = 42 → 
  (highway_distance / ((total_spent - (city_distance / city_mpg) * gas_cost_per_gallon) / gas_cost_per_gallon)) = 16.67 := by
  sorry

end car_highway_mileage_l751_751377


namespace find_constant_g_l751_751724

open Real

theorem find_constant_g
  (g : ℝ → ℝ)
  (hg_diff : ∀ x ∈ Icc 0 π, differentiable_at ℝ g x)
  (hg_cont_diff : continuous_on (deriv g) (Icc 0 π))
  (f := λ x => g x * sin x)
  (h_eq : ∫ x in 0..π, (f x)^2 = ∫ x in 0..π, (deriv f x)^2) :
  ∃ c : ℝ, ∀ x ∈ Icc 0 π, g x = c :=
begin
  -- proof placeholder
  sorry
end

end find_constant_g_l751_751724


namespace find_shapes_both_symmetric_l751_751114

def shape := Type -- Defining a type for shapes

-- Definitions for specific shapes
def line_segment : shape := sorry
def circle : shape := sorry
def equilateral_triangle : shape := sorry

-- Conditions for central symmetry
def is_centrally_symmetric (s : shape) : Prop := 
  sorry -- Placeholder for the central symmetry definition

-- Conditions for axial symmetry
def is_axially_symmetric (s : shape) : Prop := 
  sorry -- Placeholder for the axial symmetry definition

-- The set of shapes we're analyzing
def shapes : set shape := {line_segment, circle, equilateral_triangle}

-- Define which shapes are both centrally and axially symmetric
def is_both_centrally_and_axially_symmetric (s : shape) : Prop :=
  is_centrally_symmetric s ∧ is_axially_symmetric s

-- Expected result
def expected_shapes_set : set shape := {line_segment, circle}

-- Proof statement (we are stating we need to prove this, 'sorry' to skip the proof)
theorem find_shapes_both_symmetric :
  {s ∈ shapes | is_both_centrally_and_axially_symmetric s} = expected_shapes_set :=
sorry

end find_shapes_both_symmetric_l751_751114


namespace solve_inequality_l751_751192

theorem solve_inequality 
  (x : ℝ) :
  (\{x \mid \frac{(x - 3) * (x - 5) * (x - 7)}{(x - 2) * (x - 6) * (x - 8)} > 0\} =
  {x | x < 2} ∪ {x | 3 < x ∧ x < 5} ∪ {x | 6 < x ∧ x < 7}  ∪ {x | 8 < x}) :=
sorry

end solve_inequality_l751_751192


namespace students_remaining_after_fifth_stop_l751_751504

theorem students_remaining_after_fifth_stop (initial_students : ℕ) (stops : ℕ) :
  initial_students = 60 →
  stops = 5 →
  (∀ n, (n < stops → ∃ k, n = 3 * k + 1) → ∀ x, x = initial_students * ((2 : ℚ) / 3)^stops) →
  initial_students * ((2 : ℚ) / 3)^stops = (640 / 81 : ℚ) :=
by
  intros h_initial h_stops h_formula
  sorry

end students_remaining_after_fifth_stop_l751_751504


namespace sin_450_eq_1_l751_751385

theorem sin_450_eq_1 : sin (450 * π / 180) = 1 := by
  have angle_eq : 450 = 360 + 90 := by norm_num
  -- Simplify 450 degrees to radians
  rw [angle_eq, Nat.cast_add, add_mul, Nat.cast_mul]
  -- Convert degrees to radians
  rw [sin_add, sin_mul_pi_div, cos_mul_pi_div, sin_mul_pi_div, (show (90 : ℝ) = π / 2 from by norm_num)]

  sorry -- Omitting proof details

end sin_450_eq_1_l751_751385


namespace minimum_jellybeans_l751_751769

theorem minimum_jellybeans (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n = 164 :=
by sorry

end minimum_jellybeans_l751_751769


namespace eq_x2_inv_x2_and_x8_inv_x8_l751_751068

theorem eq_x2_inv_x2_and_x8_inv_x8 (x : ℝ) 
  (h : 47 = x^4 + 1 / x^4) : 
  (x^2 + 1 / x^2 = 7) ∧ (x^8 + 1 / x^8 = -433) :=
by
  sorry

end eq_x2_inv_x2_and_x8_inv_x8_l751_751068


namespace volume_tetrahedron_PQRS_l751_751200

noncomputable def volume_tetrahedron (PQ PR QR PS QS RS : ℝ) (PQR PRS PQS QRS : ℝ → ℝ → ℝ → Prop) (hPQ : PQ = 6) (hPR : PR = 4) (hQR : QR = 5) (hPS : PS = 5) (hQS : QS = 4) (hRS : RS = (10/3) * Real.sqrt 3) : ℝ :=
  let V := (20 * Real.sqrt 3 / 9) in
  V

theorem volume_tetrahedron_PQRS : volume_tetrahedron 6 4 5 5 4 ((10/3) * Real.sqrt 3) (λ _ _ _, true) (λ _ _ _, true) (λ _ _ _, true) (λ _ _ _, true) 
  (λ _ _ _, true) rfl rfl rfl rfl rfl rfl = (20 * Real.sqrt 3 / 9) := 
  by sorry

end volume_tetrahedron_PQRS_l751_751200


namespace function_fixed_point_l751_751432

def f (a x : ℝ) : ℝ := a^(x^2 + x - 2) + Real.sqrt x

theorem function_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : f a 1 = 2 := by
  sorry

end function_fixed_point_l751_751432


namespace sector_area_l751_751039

theorem sector_area (r α : ℝ) (h_r : r = 3) (h_α : α = 2) : (1/2 * r^2 * α) = 9 := by
  sorry

end sector_area_l751_751039


namespace max_k_value_l751_751854

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem max_k_value
    (h : ∀ (x : ℝ), 1 < x → f x > k * (x - 1)) :
    k = 3 := sorry

end max_k_value_l751_751854


namespace point_slope_intersection_lines_l751_751431

theorem point_slope_intersection_lines : 
  ∀ s : ℝ, ∃ x y : ℝ, 2*x - 3*y = 8*s + 6 ∧ x + 2*y = 3*s - 1 ∧ y = -((2*x)/25 + 182/175) := 
sorry

end point_slope_intersection_lines_l751_751431


namespace max_gcd_of_sequence_l751_751010

/-- Define the sequence as a function. -/
def a (n : ℕ) : ℕ := 100 + n^2

/-- Define the greatest common divisor of the sequence terms. -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- State the theorem of the maximum value of d. -/
theorem max_gcd_of_sequence : ∃ n : ℕ, d n = 401 := sorry

end max_gcd_of_sequence_l751_751010


namespace frog_jump_probability_l751_751742

open scoped ProbabilityTheory

/-
  Given:
  - The frog makes 4 jumps in a pond.
  - The first three jumps are 1 meter each.
  - The last jump can be either 1 meter or 2 meters long, each with equal probability.
  - The directions of all jumps are chosen independently at random.

  Prove that:
  - The probability that the frog’s final position is no more than 1.5 meters from its starting position is 1/6.
-/
theorem frog_jump_probability :
  let u v w : ℝ^2 := sorry -- random unit vectors representing the first three jumps
  let z1 z2 : ℝ^2 := sorry -- random unit vectors representing the last jump of length 1 or 2 meters
  let final_position_with_z1 := u + v + w + z1
  let final_position_with_z2 := u + v + w + z2
  let within_1_5_meters (p : ℝ^2) := ∥p∥ ≤ 1.5
  let probability_within_1_5_meters := 
  (1/2) * P (within_1_5_meters final_position_with_z1) +
  (1/2) * P (within_1_5_meters final_position_with_z2)
  in
  probability_within_1_5_meters = 1/6 :=
sorry

end frog_jump_probability_l751_751742


namespace sufficient_and_necessary_condition_for_perpendicular_l751_751196

variables {a b : ℝ^3}

def is_unit_vector (v : ℝ^3) : Prop := ∥v∥ = 1

def is_perpendicular (x y : ℝ^3) : Prop := x ⬝ y = 0

theorem sufficient_and_necessary_condition_for_perpendicular
  (ha : is_unit_vector a)
  (hb : is_unit_vector b) :
  ∥a - 3 • b∥ = ∥3 • a + b∥ ↔ is_perpendicular a b :=
sorry

end sufficient_and_necessary_condition_for_perpendicular_l751_751196


namespace range_of_a_l751_751133

theorem range_of_a (a : ℝ) :
  let A := {x | x^2 + 4 * x = 0}
  let B := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}
  A ∩ B = B → (a = 1 ∨ a ≤ -1) := 
by
  sorry

end range_of_a_l751_751133


namespace nested_composition_l751_751146

def g (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem nested_composition : g (g (g (g (g (g 2))))) = 2 := by
  sorry

end nested_composition_l751_751146


namespace find_sin_alpha_plus_pi_div_2_l751_751882

noncomputable def f : ℝ → ℝ := λ x, sqrt 3 * Real.sin (2 * x - π / 6)

theorem find_sin_alpha_plus_pi_div_2 (α : ℝ) (h1 : π / 6 < α) (h2 : α < 2 * π / 3) (h3 : f (α / 2) = sqrt 3 / 4) :
  Real.sin (α + π / 2) = (3 * sqrt 5 - 1) / 8 :=
by
  sorry

end find_sin_alpha_plus_pi_div_2_l751_751882


namespace inequality_proof_l751_751023

variable (n : ℕ) (a : Fin n → ℝ)
variable (h₀ : 2 ≤ n)
variable (h₁ : ∀ i : Fin n, 0 < a i)
variable (h₂ : (∑ i, a i) = 1)
variable (b : ℝ) (hb : b = ∑ i, (i+1) * a i)

theorem inequality_proof :
  ∑ i j in Fin.range n ×ˢ Fin.range n, if i < j then ((i + 1) - (j + 1))^2 * a i * a j else 0
  ≤ (n - b) * (b - 1) :=
sorry

end inequality_proof_l751_751023


namespace xy_extrema_l751_751652

noncomputable def xy_product (a : ℝ) : ℝ := a^2 - 1

theorem xy_extrema (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^2 + y^2 = -a^2 + 2) : 
  -1 ≤ xy_product a ∧ xy_product a ≤ 1/3 :=
by
  sorry

end xy_extrema_l751_751652


namespace sum_of_numbers_l751_751509

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l751_751509


namespace area_of_WXYZ_l751_751538

-- Defining the basic setup of the rectangle and its properties
def WXYZ : Type := rectangle
def Z : WXYZ.interior := angle_trisection_point WXYZ Z
def M : WXYZ.side WY := side_point WXYZ WY 3
def N : WXYZ.side WX := side_point WXYZ WX 8

-- Proving the area of rectangle WXYZ
theorem area_of_WXYZ : 
  let WZ := 8 * sqrt 3,
      MY := 8 * sqrt 3 - 3,
      ZY := (8 * sqrt 3 - 3) * sqrt 3,
      Area := WZ * ZY in
  Area = 192 * sqrt 3 - 72 := sorry

end area_of_WXYZ_l751_751538


namespace symmetry_of_geometry_word_l751_751711

-- Definitions based on conditions
def word : String := "ГЕОМЕТРИЯ"

def is_symmetrical_with_respect_to (O : Point) (figure : Point → Point) : Prop :=
  ∀ A, let A' := rotate_180 A O in figure A = A'

-- Lean 4 Statement
theorem symmetry_of_geometry_word (O : Point) : 
  is_symmetrical_with_respect_to O (rotate_180_all_points word) :=
by
  sorry

end symmetry_of_geometry_word_l751_751711


namespace find_k_l751_751877

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  real.logb 4 (4^x + 1) + k * x

theorem find_k (k : ℝ) (h: ∀ x : ℝ, f k x = f k (-x)) : k = -1/2 := by
  sorry

end find_k_l751_751877


namespace sum_of_divisors_of_10_is_18_l751_751832

theorem sum_of_divisors_of_10_is_18 :
  ∑ n in { n : ℕ | n > 0 ∧ 10 % n = 0 }, n = 18 :=
by
  sorry

end sum_of_divisors_of_10_is_18_l751_751832


namespace cosine_angle_between_vectors_l751_751820

open Real

def point (x y z : ℝ) := (x, y, z)

def vector_sub (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem cosine_angle_between_vectors (A B C : ℝ × ℝ × ℝ)
  (hA : A = (5, 3, -1)) (hB : B = (5, 2, 0)) (hC : C = (6, 4, -1)) :
  let AB := vector_sub B A
  let AC := vector_sub C A
  let dot := dot_product AB AC
  let magAB := magnitude AB
  let magAC := magnitude AC
  (dot / (magAB * magAC)) = -1 / 2 := by
  sorry

end cosine_angle_between_vectors_l751_751820


namespace even_number_of_convenient_numbers_l751_751162

def is_convenient (n : ℕ) : Prop :=
  (n ^ 2 + 1) % 1000001 = 0

theorem even_number_of_convenient_numbers :
  (Finset.filter is_convenient (Finset.range 1000000.succ)).card % 2 = 0 :=
by
  sorry

end even_number_of_convenient_numbers_l751_751162


namespace arithmetic_sequence_length_l751_751065

theorem arithmetic_sequence_length :
  ∀ (a d an : ℕ), an = a + (n - 1) * d → a = 3 → d = 3 → an = 144 → ∃ n, n = 48 :=
by {
  intros,
  sorry
}

end arithmetic_sequence_length_l751_751065


namespace f_expression_pos_l751_751871

axiom odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = -f (-x)

axiom f_expression_neg (f : ℝ → ℝ) : ∀ x, x < 0 → f x = 1 - 2 * x

theorem f_expression_pos (f : ℝ → ℝ) (h_odd : odd_function f) (h_neg : f_expression_neg f) :
  ∀ x, x > 0 → f x = -1 - 2 * x :=
by
  sorry

end f_expression_pos_l751_751871


namespace base2_to_base4_l751_751691

theorem base2_to_base4 (n : ℕ) (h : n = 0b10111100) : nat.to_digits 4 n = [2, 3, 3, 0] :=
by sorry

end base2_to_base4_l751_751691


namespace area_of_octagon_l751_751457

theorem area_of_octagon (PQRS_is_square : ∀ (P Q R S : ℝ × ℝ), square P Q R S)
  (AP_length : ∀ (A P C : ℝ × ℝ), dist A P = 1 ∧ dist P C = 1) :
  ∃ (area : ℝ), area = 4 + 4 * Real.sqrt 2 :=
  sorry

end area_of_octagon_l751_751457


namespace nested_composition_l751_751147

def g (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem nested_composition : g (g (g (g (g (g 2))))) = 2 := by
  sorry

end nested_composition_l751_751147


namespace problem_l751_751997

-- Define the values in the grid
def grid : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ := (4, 3, 1, 1, 6, 2, 3)

-- Define the variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def condition_1 := (A = 3) ∧ (B = 2) ∧ (C = 4)
def condition_2 := (4 + A + 1 + B + C + 3 = 9)
def condition_3 := (A + 1 + 6 = 9)
def condition_4 := (1 + A + 6 = 9)
def condition_5 := (B + 2 + C + 5 = 9)

-- Define that the sum of the red cells is equal to any row
def sum_of_red_cells := (A + B + C = 9)

-- The final goal to prove
theorem problem : condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5 ∧ sum_of_red_cells := 
by {
  refine ⟨_, _, _, _, _, _⟩;
  sorry   -- proofs for each condition
}

end problem_l751_751997


namespace P_xi_lt_2mu_plus_1_l751_751159

noncomputable theory
open ProbabilityTheory

variables (μ σ : ℝ) (ξ : ℝ → MeasureTheory.MeasurableSpace.ennreal) 

def normal_dist (μ σ : ℝ) : MeasureTheory.ProbabilityMeasure ℝ := sorry

axiom xi_normal_dist : ξ = normal_dist μ σ

-- Given conditions
axiom P_xi_lt_neg1 : MeasureTheory.Measure.probability_measure (MeasureTheory.event {ω | ξ(ω) < -1}) = 0.3
axiom P_xi_gt_2 : MeasureTheory.Measure.probability_measure (MeasureTheory.event {ω | ξ(ω) > 2}) = 0.3

-- The target to prove
theorem P_xi_lt_2mu_plus_1 : MeasureTheory.Measure.probability_measure (MeasureTheory.event {ω | ξ(ω) < 2 * μ + 1}) = 0.7 :=
sorry

end P_xi_lt_2mu_plus_1_l751_751159


namespace incircle_radius_l751_751682

-- Definitions based on conditions
variables {D E F : Type} [EuclideanGeometry] [Triangle DEF]
variables (right_angle_F : ∠D F E = 90°) (angle_D : ∠D = 45°) (side_DF : length D F = 8)

-- Theorem statement
theorem incircle_radius (h : right_angle_F) (h' : angle_D) (h'' : side_DF) : 
  incircle_radius DEF = 4 - 2 * Real.sqrt 2 := 
sorry

end incircle_radius_l751_751682


namespace circumference_of_shaded_region_of_square_with_quarter_circles_l751_751534

theorem circumference_of_shaded_region_of_square_with_quarter_circles 
  (A B C D E F G H : Point)
  (ABCD_square : is_square A B C D 1)
  (quarter_circle_A : is_quarter_circle A 1)
  (quarter_circle_B : is_quarter_circle B 1)
  (quarter_circle_C : is_quarter_circle C 1)
  (quarter_circle_D : is_quarter_circle D 1)
  (pi_value : real.pi = 3.141) :
  circumference_of_shaded_region A B C D E F G H = 3.141 :=
sorry

end circumference_of_shaded_region_of_square_with_quarter_circles_l751_751534


namespace johns_weekly_allowance_l751_751495

theorem johns_weekly_allowance (A : ℝ) (h1: A - (3/5) * A = (2/5) * A)
  (h2: (2/5) * A - (1/3) * (2/5) * A = (4/15) * A)
  (h3: (4/15) * A = 0.92) : A = 3.45 :=
by {
  sorry
}

end johns_weekly_allowance_l751_751495


namespace two_b_leq_a_plus_c_l751_751025

variable (t a b c : ℝ)

theorem two_b_leq_a_plus_c (ht : t > 1)
  (h : 2 / Real.log t / Real.log b = 1 / Real.log t / Real.log a + 1 / Real.log t / Real.log c) :
  2 * b ≤ a + c := by sorry

end two_b_leq_a_plus_c_l751_751025


namespace interest_calculation_years_l751_751206

theorem interest_calculation_years (P r : ℝ) (diff : ℝ) (n : ℕ) 
  (hP : P = 3600) (hr : r = 0.10) (hdiff : diff = 36) 
  (h_eq : P * (1 + r)^n - P - (P * r * n) = diff) : n = 2 :=
sorry

end interest_calculation_years_l751_751206


namespace average_speed_of_train_l751_751358

theorem average_speed_of_train (x : ℝ) (h1 : 0 < x) : 
  let Time1 := x / 40
  let Time2 := x / 10
  let TotalDistance := 3 * x
  let TotalTime := x / 8
  (TotalDistance / TotalTime = 24) :=
by
  sorry

end average_speed_of_train_l751_751358


namespace swap_numbers_l751_751636

-- Define the initial state
variables (a b c : ℕ)
axiom initial_state : a = 8 ∧ b = 17

-- Define the assignment sequence
axiom swap_statement1 : c = b 
axiom swap_statement2 : b = a
axiom swap_statement3 : a = c

-- Define the theorem to be proved
theorem swap_numbers (a b c : ℕ) (initial_state : a = 8 ∧ b = 17)
  (swap_statement1 : c = b) (swap_statement2 : b = a) (swap_statement3 : a = c) :
  (a = 17 ∧ b = 8) :=
sorry

end swap_numbers_l751_751636


namespace even_number_of_convenient_numbers_l751_751161

def is_convenient (n : ℕ) : Prop :=
  (n^2 + 1) % 1000001 = 0

theorem even_number_of_convenient_numbers :
  (1 ≤ 1000000) →
  let convenient_count := (Finset.range 1000001).filter is_convenient in
  even (Finset.card convenient_count) :=
sorry

end even_number_of_convenient_numbers_l751_751161


namespace bipartite_edge_coloring_l751_751586

variable (G : Type _) [Graph G] [Bipartite G]

theorem bipartite_edge_coloring (Δ : ℕ) (hΔ : Δ = 2019)
  (hG : ∀ v ∈ G, degree v ≤ Δ) :
  ∃ m : ℕ, (m = Δ) ∧ (∀ e1 e2 ∈ edges G, 
    common_vertex e1 e2 → e1.color ≠ e2.color) :=
by
  existsi Δ
  split
  · exact hΔ
  · sorry

end bipartite_edge_coloring_l751_751586


namespace result_is_approx_2983_95_l751_751335

noncomputable def calculateResult : Float :=
  3034 - (1002 / 20.04)

theorem result_is_approx_2983_95 : abs (calculateResult - 2983.95) < 0.01 :=
by
  -- Proof omitted
  sorry

end result_is_approx_2983_95_l751_751335


namespace sum_of_numbers_l751_751510

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l751_751510


namespace sum_of_divisors_of_10_l751_751834

theorem sum_of_divisors_of_10 :
  ∑ k in {1, 2, 5, 10}, k = 18 :=
by
  sorry

end sum_of_divisors_of_10_l751_751834


namespace max_sum_of_arithmetic_sequence_l751_751040

-- Define the problem in Lean
noncomputable theory
open_locale big_operators

def arithmetic_sequence_sum (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1) / 2) * d

theorem max_sum_of_arithmetic_sequence :
  ∀ {a : ℤ} {d : ℤ},
    (a + 2 * d = 8) → 
    (a + 3 * d = 4) →
    ∃ n : ℕ, (arithmetic_sequence_sum a d 4 = arithmetic_sequence_sum a d 5) ∧
             (arithmetic_sequence_sum a d n = arithmetic_sequence_sum a d 4) :=
by sorry

end max_sum_of_arithmetic_sequence_l751_751040


namespace number_of_toys_purchased_reduced_selling_price_for_desired_profit_l751_751804

section Double11

variables (x y : ℕ) (m : ℕ)

-- Conditions
def condition_1 : Prop := x + y = 50
def condition_2 : Prop := 28 * x + 24 * y = 1320
def profit_condition : Prop := (m - 28) * (48 - m) = 96

-- Prove quantity of toys purchased
theorem number_of_toys_purchased (h1 : condition_1) (h2 : condition_2) :
  x = 30 ∧ y = 20 :=
sorry

-- Prove reduced selling price for desired profit
theorem reduced_selling_price_for_desired_profit (h : profit_condition) :
  m = 36 :=
sorry

end Double11

end number_of_toys_purchased_reduced_selling_price_for_desired_profit_l751_751804


namespace inequality_holds_l751_751299

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l751_751299


namespace ticket_cost_is_correct_l751_751981

/-
Define the prices of different types of tickets and the ages of the members of Mrs. Lopez's group.
-/
def adult_ticket_price : ℕ := 11
def child_ticket_price : ℕ := 8
def senior_ticket_price : ℕ := 9

def number_of_adults : ℕ := 2
def number_of_seniors : ℕ := 2
def number_of_children : ℕ := 3

def child_ages : list ℕ := [7, 10, 14]

noncomputable def is_adult (age : ℕ) : bool := age ≥ 13
noncomputable def is_child (age : ℕ) : bool := age ≥ 3 ∧ age ≤ 12
noncomputable def is_senior (age : ℕ) : bool := age ≥ 60

/-
Here we compute the total cost of the tickets based on the provided conditions.
-/
noncomputable def total_ticket_cost : ℕ :=
let adults_cost : ℕ := number_of_adults * adult_ticket_price in
let seniors_cost : ℕ := number_of_seniors * senior_ticket_price in
let children_cost : ℕ := child_ages.filter is_child.length * child_ticket_price in
let additional_adult_cost : ℕ := child_ages.filter is_adult.length * adult_ticket_price in
adults_cost + seniors_cost + children_cost + additional_adult_cost

/-
The goal is to prove that the total cost of tickets is $67.
-/
theorem ticket_cost_is_correct :
  total_ticket_cost = 67 :=
  sorry

end ticket_cost_is_correct_l751_751981


namespace power_identity_l751_751902

theorem power_identity (y : ℝ) (h : 128^3 = 16^y) : 2^(-y) = 1 / 2^(5.25) :=
by {
  sorry
}

end power_identity_l751_751902


namespace glorias_turtle_time_l751_751897

theorem glorias_turtle_time :
  ∀ (t_G t_{Ge} t_{Gl} : ℕ), 
    t_G = 6 →
    t_{Ge} = t_G - 2 →
    t_{Gl} = 2 * t_{Ge} →
    t_{Gl} = 8 := by
  intros t_G t_{Ge} t_{Gl} hG hGe hGl
  rw [hG] at hGe
  rw [hGe, hG] at hGl
  rw [hGe]
  exact hGl
  sorry -- this is a placeholder indicating that there's no need to complete the proof steps

end glorias_turtle_time_l751_751897


namespace soccer_scenarios_l751_751372

theorem soccer_scenarios (x1 x2 x3 x4 x5 x6 x7 : ℕ) 
  (h1 : x1 % 7 = 0) (h2 : x2 % 7 = 0) (h3 : x3 % 7 = 0) (h4 : x4 % 7 = 0) 
  (h5 : x5 % 13 = 0) (h6 : x6 % 13 = 0) (h7 : x7 % 13 = 0) 
  (h_sum : x1 + x2 + x3 + x4 + x5 + x6 + x7 = 270) : 
  ∃ (m n : ℕ), 7 * m + 13 * n = 270 ∧ m ∈ {6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33} 
  ∧ n ∈ {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18} 
  ∧ 24244 scenarios for these values := sorry


end soccer_scenarios_l751_751372


namespace incircle_radius_is_correct_l751_751685

-- Define the given conditions
structure Triangle :=
  (D E F : ℝ)
  (angleD : ℝ)
  (DF : ℝ)

def right_angle_at_F (T : Triangle) : Prop := T.angleD = 45 ∧ T.DF = 8

-- Define the radius of the incircle function based on conditions
noncomputable def radius_of_incircle (T : Triangle) : ℝ :=
  let DF := T.DF in
  let DE := DF in
  let EF := DF * Real.sqrt 2 in
  let area := (DF * DE) / 2 in
  let s := (DF + DE + EF) / 2 in
  let r := area / s in
  r

-- The proof goal
theorem incircle_radius_is_correct (T : Triangle) (h : right_angle_at_F T) :
  radius_of_incircle T = 4 - 2 * Real.sqrt 2 := by
  sorry

end incircle_radius_is_correct_l751_751685


namespace parallel_vectors_x_eq_2_l751_751891

/-- Given vectors a and b, where a = (1, 2) and b = (x, 4), 
    if the vectors a and b are parallel, then x = 2. -/
theorem parallel_vectors_x_eq_2 (x : ℝ)
  (h : (1:ℝ) * (4:ℝ) - (2:ℝ) * x = 0) : x = 2 := 
begin
  sorry
end

end parallel_vectors_x_eq_2_l751_751891


namespace inequality_ge_one_l751_751312

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l751_751312


namespace remaining_food_after_days_l751_751754

theorem remaining_food_after_days (initial_food : ℕ) (used_after_one_day : ℚ) (used_after_two_days : ℚ) :
  initial_food = 400 →
  used_after_one_day = (2/5 : ℚ) →
  used_after_two_days = (3/5 : ℚ) →
  let remaining_after_one_day := initial_food - (initial_food * used_after_one_day).to_nat in
  let remaining_after_two_days := remaining_after_one_day - (remaining_after_one_day * used_after_two_days).to_nat in
  remaining_after_two_days = 96 :=
by
  intros h_initial_food h_used_after_one_day h_used_after_two_days
  simp [h_initial_food, h_used_after_one_day, h_used_after_two_days]
  sorry

end remaining_food_after_days_l751_751754


namespace amplitude_period_phase_shift_increasing_interval_l751_751883

def func (x : ℝ) : ℝ := sin x ^ 2 + 2 * sin x * cos x + 3 * cos x ^ 2

theorem amplitude (x : ℝ) : amplitude (func x) = sqrt 2 := sorry

theorem period (x : ℝ) : period (func x) = π := sorry

theorem phase_shift (x : ℝ) : phase_shift (func x) = π / 4 := sorry

theorem increasing_interval (k : ℤ) : ∀ x ∈ set.Icc (k * π - π / 8) (k * π + π / 8), derivative (func) x > 0 := sorry

end amplitude_period_phase_shift_increasing_interval_l751_751883


namespace tangent_line_at_2_number_of_zeros_l751_751880

noncomputable def f (x : ℝ) := 3 * Real.log x + (1/2) * x^2 - 4 * x + 1

theorem tangent_line_at_2 :
  let x := 2
  ∃ k b : ℝ, (∀ y : ℝ, y = k * x + b) ∧ (k = -1/2) ∧ (b = 3 * Real.log 2 - 5) ∧ (∀ x y : ℝ, (y - (3 * Real.log 2 - 5) = -1/2 * (x - 2)) ↔ (x + 2 * y - 6 * Real.log 2 + 8 = 0)) :=
by
  sorry

noncomputable def g (x : ℝ) (m : ℝ) := f x - m

theorem number_of_zeros (m : ℝ) :
  let g := g
  (m > -5/2 ∨ m < 3 * Real.log 3 - 13/2 → ∃ x : ℝ, g x = 0) ∧ 
  (m = -5/2 ∨ m = 3 * Real.log 3 - 13/2 → ∃ x y : ℝ, g x = 0 ∧ g y = 0 ∧ x ≠ y) ∧
  (3 * Real.log 3 - 13/2 < m ∧ m < -5/2 → ∃ x y z : ℝ, g x = 0 ∧ g y = 0 ∧ g z = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) :=
by
  sorry

end tangent_line_at_2_number_of_zeros_l751_751880


namespace initial_dolphins_l751_751670

variable (D : ℕ)

theorem initial_dolphins (h1 : 3 * D + D = 260) : D = 65 :=
by
  sorry

end initial_dolphins_l751_751670


namespace repeated_application_of_g_on_2_l751_751145

def g (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem repeated_application_of_g_on_2 :
  g(g(g(g(g(g(2)))))) = 2 :=
by
  sorry

end repeated_application_of_g_on_2_l751_751145


namespace simplify_f_evaluate_f_for_specific_alpha_l751_751851

def f (alpha : ℝ) : ℝ :=
  (sin (alpha - 3 * Real.pi) * cos (2 * Real.pi - alpha) * sin (-alpha + 3 * Real.pi / 2)) /
  (cos (-Real.pi - alpha) * sin (-Real.pi - alpha))

theorem simplify_f (alpha : ℝ) : f alpha = -cos alpha := by
  sorry

theorem evaluate_f_for_specific_alpha :
  f (-31 * Real.pi / 3) = -1 / 2 := by
  sorry

end simplify_f_evaluate_f_for_specific_alpha_l751_751851


namespace range_of_a_l751_751795

def operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, operation (x - a) (x + 1) < 1) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l751_751795


namespace daniella_lap_time_l751_751686

theorem daniella_lap_time
  (T_T : ℕ) (H_TT : T_T = 56)
  (meet_time : ℕ) (H_meet : meet_time = 24) :
  ∃ T_D : ℕ, T_D = 42 :=
by
  sorry

end daniella_lap_time_l751_751686


namespace solve_for_x_l751_751399

theorem solve_for_x {x : ℝ} (h : -3 * x - 10 = 4 * x + 5) : x = -15 / 7 :=
  sorry

end solve_for_x_l751_751399


namespace complement_A_correct_l751_751888

def A : Set ℝ := {x | 1 - (8 / (x - 2)) < 0}

def complement_A : Set ℝ := {x | x ≤ 2 ∨ x ≥ 10}

theorem complement_A_correct : (Aᶜ = complement_A) :=
by {
  -- Placeholder for the necessary proof
  sorry
}

end complement_A_correct_l751_751888


namespace triangle_interior_angle_contradiction_l751_751706

theorem triangle_interior_angle_contradiction :
  (∀ (A B C : ℝ), A + B + C = 180 ∧ A > 60 ∧ B > 60 ∧ C > 60 → false) :=
by
  sorry

end triangle_interior_angle_contradiction_l751_751706


namespace range_of_f_in_interval_interval_of_increase_of_log_f_l751_751875

noncomputable theory
open Real

def f (x : ℝ) : ℝ :=
  2 * sin² (x + 3 * π / 2) + sqrt 3 * sin (π - 2 * x)

theorem range_of_f_in_interval :
  ∀ x, 0 ≤ x ∧ x ≤ π / 2 → 0 ≤ f x ∧ f x ≤ 3 :=
sorry

theorem interval_of_increase_of_log_f :
  ∀ k : ℤ, 
  ∀ x, (π / 6 + k * π ≤ x ∧ x < π / 2 + k * π) ↔ (∀ x₁ x₂, (π / 6 + k * π ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < π / 2 + k * π) → (log (1 / 2) (f x₁) < log (1 / 2) (f x₂))) :=
sorry

end range_of_f_in_interval_interval_of_increase_of_log_f_l751_751875


namespace min_distance_midpoint_line_l751_751930

def C1 (t : Real) : Real × Real := (4 + Real.cos t, -3 + Real.sin t)
def C2 (θ : Real) : Real × Real := (6 * Real.cos θ, 2 * Real.sin θ)
def P := (4, -4) 
def Q (θ : Real) := (6 * Real.cos θ, 2 * Real.sin θ)
def M (θ : Real) := (2 + 3 * Real.cos θ, -2 + Real.sin θ)
def l := λ (x y : Real), x - Real.sqrt 3 * y - (8 + 2 * Real.sqrt 3) = 0

noncomputable def distance_to_line (x y : Real) : Real :=
  (Real.abs (x - Real.sqrt 3 * y - (8 + 2 * Real.sqrt 3))) / Real.sqrt (1 + (Real.sqrt 3)^2)

theorem min_distance_midpoint_line : ∃ θ, distance_to_line (2 + 3 * Real.cos θ) (-2 + Real.sin θ) = 3 - Real.sqrt 3 :=
by {
  sorry,
}

end min_distance_midpoint_line_l751_751930


namespace sum_a6_a7_a8_l751_751448

-- Sequence definition and sum of the first n terms
def S (n : ℕ) : ℕ := n^2 + 3 * n

theorem sum_a6_a7_a8 : S 8 - S 5 = 48 :=
by
  -- Definition and proof details are skipped
  sorry

end sum_a6_a7_a8_l751_751448


namespace eval_expr_eq_neg159_l751_751811

theorem eval_expr_eq_neg159 : 3 - (-3)^(3 - (-1)) * 2 = -159 := by
  sorry

end eval_expr_eq_neg159_l751_751811


namespace y_gt_two_if_log2_y_gt_one_l751_751503

-- Given variables:
variables (y : ℝ) (log2_y : ℝ)

-- Conditions:
-- 1. y and log2_y are real numbers
-- 2. log2_y > 1
-- 3. log2_y = log 2 of y (by the definition of logarithm base 2)

theorem y_gt_two_if_log2_y_gt_one (h1 : log2_y = real.log y / real.log 2)
                                  (h2 : log2_y > 1) :
  y > 2 := 
sorry

end y_gt_two_if_log2_y_gt_one_l751_751503


namespace find_y_value_l751_751852

-- Define the linear relationship
def linear_eq (k b x : ℝ) : ℝ := k * x + b

-- Given conditions
variables (k b : ℝ)
axiom h1 : linear_eq k b 0 = -1
axiom h2 : linear_eq k b (1/2) = 2

-- Prove that the value of y when x = -1/2 is -4
theorem find_y_value : linear_eq k b (-1/2) = -4 :=
by sorry

end find_y_value_l751_751852


namespace hawks_points_l751_751330

def touchdowns : ℕ := 3
def points_per_touchdown : ℕ := 7
def total_points (t : ℕ) (p : ℕ) : ℕ := t * p

theorem hawks_points : total_points touchdowns points_per_touchdown = 21 :=
by
  -- Proof will go here
  sorry

end hawks_points_l751_751330


namespace cyclist_wait_time_l751_751323

theorem cyclist_wait_time {d_hiker : ℝ} {d_cyclist : ℝ} (hiker_speed : ℝ) (cyclist_speed : ℝ) (wait_time : ℝ)
  (hiker_spd : hiker_speed = 4) 
  (cyclist_spd : cyclist_speed = 30) 
  (wait_time_min : wait_time = 5) :
  let time_to_catch_up := (d_cyclist / (cyclist_speed - hiker_speed)) * 60 in
  let total_wait_time := wait_time + time_to_catch_up in
  total_wait_time ≈ 10.77 :=
begin
  sorry
end

end cyclist_wait_time_l751_751323


namespace smallest_nonprime_in_range_l751_751148

def smallest_nonprime_with_no_prime_factors_less_than_20 (m : ℕ) : Prop :=
  ¬(Nat.Prime m) ∧ m > 10 ∧ ∀ p : ℕ, Nat.Prime p → p < 20 → ¬(p ∣ m)

theorem smallest_nonprime_in_range :
  smallest_nonprime_with_no_prime_factors_less_than_20 529 ∧ 520 < 529 ∧ 529 ≤ 540 := 
by 
  sorry

end smallest_nonprime_in_range_l751_751148


namespace movement_properties_l751_751621

noncomputable def distance (speed time : ℝ) : ℝ :=
  speed * time

noncomputable def pythagorean_distance (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

theorem movement_properties :
  let distance_roja := distance 5 4
  let distance_pooja := distance 3 4
  let distance_sooraj := distance 4 4 in
  distance_roja = 20 ∧
  distance_pooja = 12 ∧
  distance_sooraj = 16 ∧
  pythagorean_distance distance_roja distance_pooja = real.sqrt 544 ∧
  distance_sooraj = 16 ∧
  90 = 90 :=
by
  sorry

end movement_properties_l751_751621


namespace exists_point_with_distances_l751_751155

-- Definitions
variables (n : ℕ) (hn : n > 12)
variables (points : list (ℝ × ℝ)) (H_distinct : function.injective (prod.fst ∘ list.nth points))

-- Assumption: the list of points includes the n points P1, P2, ..., Pn, and a point Q
noncomputable def P (i : ℕ) (H : i < n) : ℝ × ℝ := list.nth_le points i H
noncomputable def Q : ℝ × ℝ := list.nth_le points n (by linarith [hn])

-- Distance function
def dist (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

-- The theorem to prove
theorem exists_point_with_distances (hn : n > 12) :
  ∃ i < n, (finset.filter (λ j, j ≠ i ∧ dist (P n i) (P n j) < dist (P n i) Q) (finset.range n)).card ≥ n / 6 - 1 :=
sorry

end exists_point_with_distances_l751_751155


namespace inequality_inequality_holds_l751_751318

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l751_751318


namespace yuna_has_biggest_number_l751_751274

-- Define the collections
def yoongi_collected : ℕ := 4
def jungkook_collected : ℕ := 6 - 3
def yuna_collected : ℕ := 5

-- State the theorem
theorem yuna_has_biggest_number :
  yuna_collected > yoongi_collected ∧ yuna_collected > jungkook_collected :=
by
  sorry

end yuna_has_biggest_number_l751_751274


namespace value_range_g_l751_751664

noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_range_g : Set.range (λ x, g x) ∩ Set.Icc 0 3 = Set.Icc (-1) 3 :=
by {
  sorry
}

end value_range_g_l751_751664


namespace function_positive_on_interval_l751_751213

theorem function_positive_on_interval (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → (2 - a^2) * x + a > 0) ↔ 0 < a ∧ a < 2 :=
by
  sorry

end function_positive_on_interval_l751_751213


namespace least_positive_integer_l751_751695

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 0) ∧ (a % 5 = 1) ∧ (a % 4 = 2) → a = 6 :=
by
  sorry

end least_positive_integer_l751_751695


namespace q_value_l751_751152

noncomputable def prove_q (a b m p q : Real) :=
  (a * b = 5) → 
  (b + 1/a) * (a + 1/b) = q →
  q = 36/5

theorem q_value (a b : ℝ) (h_roots : a * b = 5) : (b + 1/a) * (a + 1/b) = 36 / 5 :=
by 
  sorry

end q_value_l751_751152


namespace profit_margin_typeA_selling_price_typeB_units_purchased_units_during_new_year_l751_751740

-- Constants for cost and selling prices
def costA := 40
def sellA := 60
def costB := 50
def profit_marginB := 0.6

-- Profit margin for type A
def profit_marginA := (sellA - costA) / costA * 100

-- Selling price for type B
def sellB := costB + (costB * profit_marginB)

-- Total units and cost
def total_units := 60
def total_cost := 2600

-- Units equations
def unitsA (x : ℕ) := 40 * x + 50 * (total_units - x) = total_cost

-- Total amount spent and discount categories
def total_spent_on_B := 320
def total_spent_on_A := 432

def discount_rate (p : ℕ) : ℝ :=
  if p <= 380 then 1
  else if p <= 500 then 0.9
  else 0.8

def unitsA_purchased (dAunits : ℝ) : ℕ := 
  round ((total_spent_on_A / discount_rate ceil (total_spent_on_A / 60)) / 60)

-- Proof of problem parts
theorem profit_margin_typeA : profit_marginA = 50 := by
  sorry

theorem selling_price_typeB : sellB = 80 := by
  sorry

theorem units_purchased : ∃ x : ℕ, (unitsA x) :=
  exists.intro 40 (by
    simp [unitsA, total_units, total_cost, costA, costB]
    sorry)

theorem units_during_new_year : 
     ((1200 * total_spent_on_A / discount_rate ceil (total_spent_on_A / 60) % 60 = 0 
     ∨ total_spent_on_A / discount_rate ceil (total_spent_on_A / 60) % 60 = 0.0)
∧ (total_spent_on_B / sellB) = 4.0) := by
     sorry

end profit_margin_typeA_selling_price_typeB_units_purchased_units_during_new_year_l751_751740


namespace inequality_holds_l751_751306

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l751_751306


namespace Q_current_age_l751_751991

-- Definitions for the current ages of P and Q
variable (P Q : ℕ)

-- Conditions
-- 1. P + Q = 100
-- 2. P = 3 * (Q - (P - Q))  (from P is thrice as old as Q was when P was as old as Q is now)

axiom age_sum : P + Q = 100
axiom age_relation : P = 3 * (Q - (P - Q))

theorem Q_current_age : Q = 40 :=
by
  sorry

end Q_current_age_l751_751991


namespace find_transformed_polynomial_l751_751500

-- Given a polynomial equation and its roots
theorem find_transformed_polynomial:
  ∀ (a b c d : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) 
  ∧ a^4 - a^2 - 5 = 0 
  ∧ b^4 - b^2 - 5 = 0 
  ∧ c^4 - c^2 - 5 = 0 
  ∧ d^4 - d^2 - 5 = 0
  → Polynomial.of_fn ![5,1,-1].roots = ![(1/a)^2, (1/b)^2, (1/c)^2, (1/d)^2] := 
by {
  intros a b c d h,
  sorry
}

end find_transformed_polynomial_l751_751500


namespace range_of_a_l751_751889

variable (a : ℝ)

def set_A (a : ℝ) : Set ℝ := {x | abs (x - 2) ≤ a}
def set_B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

lemma disjoint_sets (A B : Set ℝ) : A ∩ B = ∅ :=
  sorry

theorem range_of_a (h : set_A a ∩ set_B = ∅) : a < 1 :=
  by
  sorry

end range_of_a_l751_751889


namespace triangle_area_ratio_l751_751089

-- Define the statement for the proof
theorem triangle_area_ratio (ABC P Q R : Triangle)
  (h_divide: perimeter ABC = 1)
  (h_equal_parts: PQ = QR)
  (h_pq_on_ab: P ∈ AB ∧ Q ∈ AB) :
  (area (triangle P Q R)) / (area ABC) > 2 / 9 := 
sorry

end triangle_area_ratio_l751_751089


namespace a_sufficient_not_necessary_l751_751442

theorem a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (¬(1 / a < 1 → a > 1)) :=
by
  sorry

end a_sufficient_not_necessary_l751_751442


namespace range_of_k_l751_751451

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 8 * x + 12 = 0

-- Define the equation of the line
def line_eq (k x y : ℝ) : Prop := y = k * x - 2

-- Define the distance from the center (-4, 0) to a line
def distance_to_line (k : ℝ) : ℝ := abs (-4 * k - 2) / sqrt (k^2 + 1)

-- The main statement to prove
theorem range_of_k (k : ℝ) : (∃ x y : ℝ, circle_eq x y ∧ line_eq k x y) ↔ -4/3 <= k ∧ k <= 0 :=
by {
  sorry,
}

end range_of_k_l751_751451


namespace glorias_turtle_time_l751_751896

theorem glorias_turtle_time :
  ∀ (t_G t_{Ge} t_{Gl} : ℕ), 
    t_G = 6 →
    t_{Ge} = t_G - 2 →
    t_{Gl} = 2 * t_{Ge} →
    t_{Gl} = 8 := by
  intros t_G t_{Ge} t_{Gl} hG hGe hGl
  rw [hG] at hGe
  rw [hGe, hG] at hGl
  rw [hGe]
  exact hGl
  sorry -- this is a placeholder indicating that there's no need to complete the proof steps

end glorias_turtle_time_l751_751896


namespace perp_OD_BC_l751_751109

open EuclideanGeometry

noncomputable def acute_triangle (A B C : Point) : Prop :=
-- A triangle is acute if all its internal angles are less than 90 degrees.
∠ A B C < 90 ∧ ∠ B A C < 90 ∧ ∠ A C B < 90

noncomputable def angle_bisector (A B C D : Point) : Prop :=
-- D is on the angle bisector of ∠BAC
collinear A D B ∧ collinear A D C

noncomputable def circumcenter (A B C O : Point) : Prop :=
-- O is the circumcenter of triangle ABC
dist O A = dist O B ∧ dist O B = dist O C 

theorem perp_OD_BC (A B C D E F M N O : Point) :
  acute_triangle A B C →
  angle_bisector A B C D →
  intersects_line D C B E AB →
  intersects_line D B C F AC →
  intersects_circle E F (circle A B C) M N →
  circumcenter D M N O →
  perpendicular O D B C :=
begin
  sorry
end

end perp_OD_BC_l751_751109


namespace least_positive_integer_l751_751694

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 0) ∧ (a % 5 = 1) ∧ (a % 4 = 2) → a = 6 :=
by
  sorry

end least_positive_integer_l751_751694


namespace no_valid_point_C_l751_751529

open Real EuclideanGeometry

noncomputable def distance (p q : Point ℝ) : ℝ := euclidean_distance p q

def perimeter (A B C : Point ℝ) : ℝ := distance A B + distance B C + distance C A

def area (A B C : Point ℝ) : ℝ := (1 / 2) * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)).abs

theorem no_valid_point_C : ∀ (A B : Point ℝ), distance A B = 10 → ¬ (∃ C : Point ℝ, perimeter A B C = 50 ∧ area A B C = 100) :=
by
  intros A B h_dist
  sorry

end no_valid_point_C_l751_751529


namespace number_of_possible_sums_l751_751268

def rolls_range : Set ℕ := {n : ℕ | 4 ≤ n ∧ n ≤ 24}

theorem number_of_possible_sums : rolls_range.card = 21 :=
by
  sorry

end number_of_possible_sums_l751_751268


namespace most_appropriate_chart_for_trend_l751_751522

theorem most_appropriate_chart_for_trend :
  (chart : String) (chart = "line chart") →
  (use_for : String) (use_for = "reflect the trend of changes in various data") →
  chart = "line chart" :=
by
  intros chart chart_eq use_for use_for_eq
  sorry

end most_appropriate_chart_for_trend_l751_751522


namespace a_difference_l751_751886

def a (n : ℕ) : ℚ :=
  1 + ∑ i in finset.range (3 * n - 1), (1 : ℚ) / (i + 1)

theorem a_difference (n : ℕ) : a (n + 1) - a n = (1 / (3 * n : ℚ)) + (1 / (3 * n + 1 : ℚ)) + (1 / (3 * n + 2 : ℚ)) :=
by
  sorry

end a_difference_l751_751886


namespace xyz_value_l751_751455

theorem xyz_value (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 30) 
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : x * y * z = 7 :=
by
  sorry

end xyz_value_l751_751455


namespace last_digit_of_product_l751_751364

theorem last_digit_of_product {n : ℕ} (hn : 1 ≤ n ∧ n ≤ 100) :
  (∀ x : ℕ, (1 ≤ x ∧ x ≤ n) → ¬ (x % 2 = 0 ∨ x % 5 = 0)) →
  (∃ m : ℕ, ∀ k : ℕ, k < 10 → last_digit_of ((∏ x in finset.range 100, if x % 2 ≠ 0 ∧ x % 5 ≠ 0 then x else 1) = m) :=
sorry

end last_digit_of_product_l751_751364


namespace total_cost_l751_751119

def daily_rental_cost : ℝ := 25
def cost_per_mile : ℝ := 0.20
def duration_days : ℕ := 4
def distance_miles : ℕ := 400

theorem total_cost 
: (daily_rental_cost * duration_days + cost_per_mile * distance_miles) = 180 := 
by
  sorry

end total_cost_l751_751119


namespace point_on_x_axis_l751_751539

theorem point_on_x_axis (a : ℝ) (P : ℝ × ℝ) (h : P = (a-3, 2*a+1)) (hx : P.2 = 0) : a = -1/2 :=
by
  cases h
  rw [prod.mk.eta] at hx
  sorry

end point_on_x_axis_l751_751539


namespace bookstore_floor_l751_751639

theorem bookstore_floor
  (academy_floor : ℤ)
  (reading_room_floor : ℤ)
  (bookstore_floor : ℤ)
  (h1 : academy_floor = 7)
  (h2 : reading_room_floor = academy_floor + 4)
  (h3 : bookstore_floor = reading_room_floor - 9) :
  bookstore_floor = 2 :=
by
  sorry

end bookstore_floor_l751_751639


namespace sum_first_n_nat_eq_276_l751_751267

theorem sum_first_n_nat_eq_276 : ∃ n : ℕ, (n * (n + 1) / 2 = 276) ∧ n = 23 :=
by {
  use 23,
  split,
  {
    norm_num,
  },
  {
    refl,
  }
}

end sum_first_n_nat_eq_276_l751_751267


namespace tangent_parabola_line_l751_751916

theorem tangent_parabola_line (a b : ℝ) : 
    (∀ x : ℝ, a * x^2 + b * x + 7 = 2 * x + 3 → 
        (b - 2)^2 = 16 * a) → 
    (b = 2 + 4 * real.sqrt a) ∨ (b = 2 - 4 * real.sqrt a) :=
begin
  sorry,
end

end tangent_parabola_line_l751_751916


namespace peanuts_in_box_l751_751084

variable (original_peanuts : Nat)
variable (additional_peanuts : Nat)

theorem peanuts_in_box (h1 : original_peanuts = 4) (h2 : additional_peanuts = 4) :
  original_peanuts + additional_peanuts = 8 := 
by
  sorry

end peanuts_in_box_l751_751084


namespace f_l751_751043

variable {x : ℝ}

-- Define the function f
def f (x : ℝ) (f'2 : ℝ) : ℝ := 3 * x^2 + 2 * x * f'2

-- Provide the condition where f' is the derivative of f
axiom f'_def : ∀ {x : ℝ} (f'2 : ℝ), deriv (λ x, f x f'2) x = 6 * x + 2 * f'2

-- Given condition for f'(2)
axiom f'_2_val : ∀ (f'2 : ℝ), 6 * 2 + 2 * f'2 = -12

-- The goal to be proven
theorem f'_5_eq_6 : ∀ (f'2 : ℝ), 6 * 5 + 2 * f'2 = 6 :=
by
  intro f'2
  have h : 2 * f'2 = -24 := sorry
  calc
    6 * 5 + 2 * f'2
      = 6 * 5 - 24 : by rw [h]
  ... = 6 : by norm_num

-- Skip proofs for automatic assumption resolution.
example : f'_5_eq_6 (-(12:ℝ)) :=
by
  rw [f'_5_eq_6, show (-24 : ℝ) = 2 * -(12 : ℝ) from by norm_num]
  exact rfl

end f_l751_751043


namespace sum_of_inverses_l751_751592

def f (x : ℝ) : ℝ :=
if x ≤ 2 then 3 - x else 3 * x - x^2

def f_inv (y : ℝ) : ℝ :=
if y = -4 then 4 else
if y = 1 then 2 else
if y = 4 then -1 else 0

theorem sum_of_inverses :
  f_inv (-4) + f_inv (1) + f_inv (4) = 5 :=
by 
  -- the proof itself is not required as per instruction
  sorry

end sum_of_inverses_l751_751592


namespace smallest_b_l751_751620

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b) 
(h3 : 2 + a ≤ b) (h4 : 1 / a + 1 / b ≤ 2) : b = 2 :=
sorry

end smallest_b_l751_751620


namespace placement_possible_l751_751575

def can_place_numbers : Prop :=
  ∃ (top bottom : Fin 4 → ℕ), 
    (∀ i, (top i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (∀ i, (bottom i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (List.product (List.ofFn top) = List.product (List.ofFn bottom))

theorem placement_possible : can_place_numbers :=
sorry

end placement_possible_l751_751575


namespace salvadore_earned_l751_751622

theorem salvadore_earned :
  ∃ S : ℝ, (S + S / 2 = 2934) ∧ (S = 1956) :=
by
  use 1956
  split
  sorry

end salvadore_earned_l751_751622


namespace range_of_a_range_of_m_l751_751031

-- Definition of proposition p: Equation has real roots
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a * x + a + 3 = 0

-- Definition of proposition q: m - 1 <= a <= m + 1
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 1

-- Part (I): Range of a when ¬p is true
theorem range_of_a (a : ℝ) (hp : ¬ p a) : -2 < a ∧ a < 6 :=
sorry

-- Part (II): Range of m when p is a necessary but not sufficient condition for q
theorem range_of_m (m : ℝ) (hnp : ∀ a, q m a → p a) (hns : ∃ a, q m a ∧ ¬p a) : m ≤ -3 ∨ m ≥ 7 :=
sorry

end range_of_a_range_of_m_l751_751031


namespace other_small_triangle_area_ratio_l751_751233

def ratio_of_areas (a b n : ℝ) (h_a_nonzero : a ≠ 0)
  (h_b_nonzero : b ≠ 0) (h_n_nonzero : n ≠ 0) : ℝ :=
  b / (4 * n * a)

theorem other_small_triangle_area_ratio
  (a b n : ℝ) (h_a_nonzero : a ≠ 0)
  (h_b_nonzero : b ≠ 0) (h_n_nonzero : n ≠ 0)
  (h : ∃ rect_area : ℝ, rect_area = a * b 
    ∧ (∃ tria_area1 : ℝ, tria_area1 = n * rect_area 
        ∧ ∃ tria_area2 : ℝ, tria_area2 = ratio_of_areas a b n h_a_nonzero h_b_nonzero h_n_nonzero * rect_area)) :
  ratio_of_areas a b n h_a_nonzero h_b_nonzero h_n_nonzero = b / (4 * n * a) :=
sorry

end other_small_triangle_area_ratio_l751_751233


namespace ten_digit_random_numbers_count_l751_751349

theorem ten_digit_random_numbers_count : 
  ∃ count : ℕ, count = 50 ∧ 
    ∀ (n : ℕ), (n > 0 ∧
                (∀ d : ℕ, d ∈ fin_num_digits n → (0 < d ∧ d < 10)) ∧
                (11 ∣ n) ∧ 
                (12 ∣ n) ∧ 
                (∀ perm : ℕ, permutes perm n → (12 ∣ perm))
    ) → n ∈ (generate_random_numbers 10) → count = 50
sorry

end ten_digit_random_numbers_count_l751_751349


namespace inequality_proof_l751_751278

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751278


namespace time_of_free_fall_l751_751655

theorem time_of_free_fall (h : ℝ) (t : ℝ) (height_fall_eq : h = 4.9 * t^2) (initial_height : h = 490) : t = 10 :=
by
  -- Proof is omitted
  sorry

end time_of_free_fall_l751_751655


namespace students_that_do_not_like_either_sport_l751_751735

def total_students : ℕ := 30
def students_like_basketball : ℕ := 15
def students_like_table_tennis : ℕ := 10
def students_like_both : ℕ := 3

theorem students_that_do_not_like_either_sport : (total_students - (students_like_basketball + students_like_table_tennis - students_like_both)) = 8 := 
by
  sorry

end students_that_do_not_like_either_sport_l751_751735


namespace min_sum_of_segments_is_305_l751_751221

noncomputable def min_sum_of_segments : ℕ := 
  let a : ℕ := 3
  let b : ℕ := 5
  100 * a + b

theorem min_sum_of_segments_is_305 : min_sum_of_segments = 305 := by
  sorry

end min_sum_of_segments_is_305_l751_751221


namespace goldfinch_percentage_l751_751610

def number_of_goldfinches := 6
def number_of_sparrows := 9
def number_of_grackles := 5
def total_birds := number_of_goldfinches + number_of_sparrows + number_of_grackles
def goldfinch_fraction := (number_of_goldfinches : ℚ) / total_birds

theorem goldfinch_percentage : goldfinch_fraction * 100 = 30 := 
by
  sorry

end goldfinch_percentage_l751_751610


namespace total_fish_in_tanks_l751_751122

theorem total_fish_in_tanks :
  ∃ (tank1_fish tank2_fish tank3_fish : ℕ), tank1_fish = 20 ∧ tank2_fish = 2 * tank1_fish ∧ tank3_fish = 2 * tank1_fish ∧ tank1_fish + tank2_fish + tank3_fish = 100 :=
by
  exists 20
  exists 40
  exists 40
  simp
  split
  rfl
  split
  rfl
  split
  rfl 
  simp
  sorry

end total_fish_in_tanks_l751_751122


namespace quad_root_l751_751843

theorem quad_root (m : ℝ) (β : ℝ) (root_condition : ∃ α : ℝ, α = -5 ∧ (α + β) * (α * β) = x^2 + m * x - 10) : β = 2 :=
by
  sorry

end quad_root_l751_751843


namespace distance_between_points_l751_751421

-- Define the points (0, 12) and (9, 0)
def point1 := (0, 12)
def point2 := (9, 0)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The statement to prove
theorem distance_between_points : distance point1 point2 = 15 :=
by
  sorry

end distance_between_points_l751_751421


namespace most_reasonable_plan_l751_751923

-- Defining the conditions as a type
inductive SurveyPlans
| A -- Surveying students in the second grade of School B
| C -- Randomly surveying 150 teachers
| B -- Surveying 600 students randomly selected from School C
| D -- Randomly surveying 150 students from each of the four schools

-- Define the main theorem asserting that the most reasonable plan is Option D
theorem most_reasonable_plan : SurveyPlans.D = SurveyPlans.D :=
by
  sorry

end most_reasonable_plan_l751_751923


namespace inequality_inequality_holds_l751_751317

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l751_751317


namespace general_term_formula_l751_751933

noncomputable def x : ℕ → ℝ
| 1 := 3
| 2 := 2
| n + 3 := x (n + 1) * x (n + 2) / (2 * x (n + 1) - x (n + 2))

theorem general_term_formula (n : ℕ) : x (n + 1) = 6 / (n + 2) := 
sorry

end general_term_formula_l751_751933


namespace g_neither_even_nor_odd_l751_751800

def g (x : ℝ) : ℝ := (3^x + 2) / (3^x + 3)

theorem g_neither_even_nor_odd : 
  ¬ (∀ x : ℝ, g (-x) = g x) ∧ ¬ (∀ x : ℝ, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l751_751800


namespace roof_shingles_area_l751_751214

-- Definitions based on given conditions
def base_main_roof : ℝ := 20.5
def height_main_roof : ℝ := 25
def upper_base_porch : ℝ := 2.5
def lower_base_porch : ℝ := 4.5
def height_porch : ℝ := 3
def num_gables_main_roof : ℕ := 2
def num_trapezoids_porch : ℕ := 4

-- Proof problem statement
theorem roof_shingles_area : 
  2 * (1 / 2 * base_main_roof * height_main_roof) +
  4 * (1 / 2 * (upper_base_porch + lower_base_porch) * height_porch) = 554.5 :=
by sorry

end roof_shingles_area_l751_751214


namespace inequality_ge_one_l751_751307

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l751_751307


namespace cube_face_product_l751_751571

open Finset

theorem cube_face_product (numbers : Finset ℕ) (hs : numbers = range (12 + 1)) :
  ∃ top_face bottom_face : Finset ℕ,
    top_face.card = 4 ∧
    bottom_face.card = 4 ∧
    (numbers \ (top_face ∪ bottom_face)).card = 4 ∧
    (∏ x in top_face, x) = (∏ x in bottom_face, x) :=
by
  use {2, 4, 9, 10}
  use {3, 5, 6, 8}
  repeat { split };
  -- Check cardinality conditions
  sorry

end cube_face_product_l751_751571


namespace three_term_inequality_l751_751295

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l751_751295


namespace possible_to_place_12_numbers_on_cube_edges_l751_751559

-- Define a list of numbers from 1 to 12
def nums := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the faces of the cube in terms of the indices of nums list
def top_face := [1, 2, 9, 10]
def bottom_face := [3, 5, 6, 8]

-- Define the product of the numbers on the faces of the cube
def product_face (face : List Nat) : Nat := face.foldr (*) 1

-- The lean statement proving the problem
theorem possible_to_place_12_numbers_on_cube_edges :
  product_face (top_face.map (λ i => nums.get! (i - 1))) =
  product_face (bottom_face.map (λ i => nums.get! (i - 1))) :=
by
  sorry

end possible_to_place_12_numbers_on_cube_edges_l751_751559


namespace find_x_l751_751714

   noncomputable def multiples_of_14_consecutive (a b : ℕ) (q : set ℕ) :=
     ∃ m n : ℕ, a = 14 * m ∧ b = 14 * n ∧ q = set.Icc a b ∧ ∃ c d : ℕ, q = set.Icc (14 * c) (14 * d) ∧ d - c = 8

   noncomputable def multiples_of_x (q : set ℕ) (x : ℕ) :=
     ∃ k l : ℕ, ∃ m n : ℕ, (k = m * x ∧ l = n * x) ∧ l - k = 16

   theorem find_x (a b x : ℕ) (q : set ℕ) (h1 : multiples_of_14_consecutive a b q)
     (h2 : multiples_of_x q x) : x = 7 :=
   sorry
   
end find_x_l751_751714


namespace trigonometric_arithmetic_sequence_l751_751413

theorem trigonometric_arithmetic_sequence (a : ℝ) (h1 : 0 < a) (h2 : a < 360) :
  (sin (a * π / 180) + sin (3 * a * π / 180) = 2 * sin (2 * a * π / 180)) ↔ a = 180 :=
by sorry

end trigonometric_arithmetic_sequence_l751_751413


namespace evaluate_expression_l751_751589

theorem evaluate_expression {x y : ℕ} (h₁ : 144 = 2^x * 3^y) (hx : x = 4) (hy : y = 2) : (1 / 7) ^ (y - x) = 49 := 
by
  sorry

end evaluate_expression_l751_751589


namespace sum_of_areas_of_two_parks_l751_751634

theorem sum_of_areas_of_two_parks :
  let side1 := 11
  let side2 := 5
  let area1 := side1 * side1
  let area2 := side2 * side2
  area1 + area2 = 146 := 
by 
  sorry

end sum_of_areas_of_two_parks_l751_751634


namespace rectangle_integer_sides_noncongruent_count_l751_751749

theorem rectangle_integer_sides_noncongruent_count (h w : ℕ) :
  (2 * (w + h) = 72 ∧ w ≠ h) ∨ ((w = h) ∧ 2 * (w + h) = 72) →
  (∃ (count : ℕ), count = 18) :=
by
  sorry

end rectangle_integer_sides_noncongruent_count_l751_751749


namespace no_square_pair_l751_751186

/-- 
Given integers a, b, and c, where c > 0, if a(a + 4) = c^2 and (a + 2 + c)(a + 2 - c) = 4, 
then the numbers a(a + 4) and b(b + 4) cannot both be squares.
-/
theorem no_square_pair (a b c : ℤ) (hc_pos : c > 0) (ha_eq : a * (a + 4) = c^2) 
  (hfac_eq : (a + 2 + c) * (a + 2 - c) = 4) : ¬(∃ d e : ℤ, d^2 = a * (a + 4) ∧ e^2 = b * (b + 4)) :=
by sorry

end no_square_pair_l751_751186


namespace noah_total_watts_used_l751_751982

theorem noah_total_watts_used :
  let bedroom_watts_per_hour := 6
  let office_watts_per_hour := 3 * bedroom_watts_per_hour
  let living_room_watts_per_hour := 4 * bedroom_watts_per_hour
  let hours_on := 2
  let bedroom_total := bedroom_watts_per_hour * hours_on
  let office_total := office_watts_per_hour * hours_on
  let living_room_total := living_room_watts_per_hour * hours_on
  bedroom_total + office_total + living_room_total = 96 :=
by
  -- Define the given conditions as variables
  let bedroom_watts_per_hour := 6
  let office_watts_per_hour := 3 * bedroom_watts_per_hour
  let living_room_watts_per_hour := 4 * bedroom_watts_per_hour
  let hours_on := 2
  
  -- Calculate watts used over two hours
  let bedroom_total := bedroom_watts_per_hour * hours_on
  let office_total := office_watts_per_hour * hours_on
  let living_room_total := living_room_watts_per_hour * hours_on
  
  -- Sum up the totals
  have h1 : bedroom_total = 12 := rfl
  have h2 : office_total = 36 := rfl
  have h3 : living_room_total = 48 := rfl
  have sum_totals : 12 + 36 + 48 = 96 := by norm_num

  -- Conclusion
  show bedroom_total + office_total + living_room_total = 96 from sum_totals

end noah_total_watts_used_l751_751982


namespace intersection_M_N_l751_751488

open Set

-- Definitions from conditions
def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {x | x < 1}

-- Proof statement
theorem intersection_M_N : M ∩ N = {-1} := 
by sorry

end intersection_M_N_l751_751488


namespace determine_Q_l751_751549

variables {A B C Q : Type}
variables [AddCommGroup A] [Module ℝ A]

-- Defining vectors representing points A, B, C in space.
variables (a b c : A) (x y z : ℝ)
variables {f g : A}

-- Conditions
-- 1. F lies on BC extended past C with ratio BF:FC = 4:1.
def F := (4 / (4 + 1)) • c - (1 / (4 + 1)) • b

-- 2. G lies on AC with ratio AG:GC = 6:2 simplified to 3:1.
def G := (1 / (3 + 1)) • a + (3 / (3 + 1)) • c

-- 3. Q is the intersection of lines BG and AF.
def Q := x • a + y • b + z • c

-- Ensure x + y + z = 1
def weight_sum_property := x + y + z = 1

-- The proof statement
theorem determine_Q : weight_sum_property → Q = a :=
by
  sorry

end determine_Q_l751_751549


namespace sum_f_k_l751_751868

def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom condition1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom condition2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_at_2 : g 2 = 4

theorem sum_f_k : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
by sorry

end sum_f_k_l751_751868


namespace parallel_vectors_l751_751062

variable (x : ℝ)

def a : ℝ × ℝ := (x, 3)
def b : ℝ × ℝ := (4, 6)

theorem parallel_vectors (h : ∃ k : ℝ, a = (λ k, (4 * k, 6 * k)) k) : x = 2 :=
by
  sorry

end parallel_vectors_l751_751062


namespace students_remaining_after_four_stops_l751_751077

theorem students_remaining_after_four_stops :
  let initial_students := 60 
  let fraction_remaining := (2 / 3 : ℚ)
  let stop1_students := initial_students * fraction_remaining
  let stop2_students := stop1_students * fraction_remaining
  let stop3_students := stop2_students * fraction_remaining
  let stop4_students := stop3_students * fraction_remaining
  stop4_students = (320 / 27 : ℚ) :=
by
  sorry

end students_remaining_after_four_stops_l751_751077


namespace exists_sign_selection_l751_751667

variable (points : Fin 8 → (ℝ × ℝ))

-- Function to compute the area of a triangle given three points
def area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((p1.1 * (p2.2 - p3.2)) + (p2.1 * (p3.2 - p1.2)) + (p3.1 * (p1.2 - p2.2)))

-- Variables representing the areas of triangles formed by any three points among the given 8 points
def triangle_areas : Fin 56 → ℝ
| ⟨n, h⟩ =>
  let ⟨i, j, k, ht⟩ := Fin.triplet_8 n h
  area (points i) (points j) (points k)

-- The proof problem statement
theorem exists_sign_selection : ∃ (s : Fin 56 → Bool), ∑ i, ((if s i then 1 else -1) * triangle_areas points i) = 0 :=
by sorry

end exists_sign_selection_l751_751667


namespace inequality_transformation_l751_751677

theorem inequality_transformation (x : ℝ) :
  x - 2 > 1 → x > 3 :=
by
  intro h
  linarith

end inequality_transformation_l751_751677


namespace ratio_of_angles_in_triangle_PUT_l751_751727

theorem ratio_of_angles_in_triangle_PUT
  (P Q R S T U : Point)
  (h1: regular_pentagon P Q R S T)
  (h2: lies_on U S T)
  (h3: ∠ Q P U = 90) :
  ratio_of_interior_angles_in_triangle P U T = (1, 3, 6) :=
by
  sorry

end ratio_of_angles_in_triangle_PUT_l751_751727


namespace votes_ratio_l751_751920

-- Declare the variables
variables 
  (x : ℕ) -- number of votes in the first and second polling stations
  (y : ℕ) -- number of votes in the third polling station

-- Define the conditions from the problem
def first_polling_station_votes (x : ℕ) : ℚ := (7 : ℚ) / 12 * x
def second_polling_station_votes (x : ℕ) : ℚ := (5 : ℚ) / 8 * x
def third_polling_station_votes (y : ℕ) : ℚ := (3 : ℚ) / 10 * y
def total_votes (x y : ℕ) : ℚ := 2 * x + y
def half_total_votes (x y : ℕ) : ℚ := total_votes x y / 2

-- Define the proof problem
theorem votes_ratio (x y : ℕ) (h : first_polling_station_votes x + second_polling_station_votes x + (total_votes x y - third_polling_station_votes y) / 2 = half_total_votes x y) :
  x * 25 = 24 * y :=
begin
  sorry
end

end votes_ratio_l751_751920


namespace binary_subtraction_correct_l751_751380

theorem binary_subtraction_correct :
  (11011_2 - 101_2 = 10110_2) := by
  -- Definitions of binary numbers:
  let b1 : ℕ := 27  -- 11011 in decimal
  let b2 : ℕ := 5   -- 101 in decimal
  
  -- Perform subtraction:
  let result : ℕ := b1 - b2
  
  -- Check binary representation of the result:
  have h : result = 22 := by
    show 27 - 5 = 22
    rfl

  have h2 : 22 = 22 := rfl
  
  -- Converting 22 to binary should be 10110
  sorry

end binary_subtraction_correct_l751_751380


namespace fraction_equality_l751_751728

theorem fraction_equality : 
  (9 : ℝ) / ((7 : ℝ) * (53 : ℝ)) = (0.9 : ℝ) / ((0.7 : ℝ) * (53 : ℝ)) :=
by 
sory

end fraction_equality_l751_751728


namespace inscribed_cylinder_radius_l751_751351

-- Definitions of the problem conditions
def cylinder_radius (cylinder_height: ℚ) : ℚ := cylinder_height / 2

def cone_base_radius (cone_diameter: ℚ) : ℚ := cone_diameter / 2

def height_above_base (cone_height: ℚ) (cylinder_height: ℚ) : ℚ := cone_height - cylinder_height

theorem inscribed_cylinder_radius :
  let r_cylinder := cylinder_radius (cylinder_height:=2 * (cylinder_radius _))
  let r_cone := cone_base_radius (cone_diameter:=14)
  let h_cone := 16
  let h_cylinder := 2 * (cylinder_radius _)
  let h_above := height_above_base h_cone h_cylinder
  ∀ r : ℚ, (16 - 2 * r) / r = 16 / 7 → r = 56 / 15 :=
begin
  intros r h,
  sorry
end

end inscribed_cylinder_radius_l751_751351


namespace geometric_seq_arith_condition_half_l751_751007

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions from the problem
def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def positive_terms (a : ℕ → ℝ) := ∀ n, a n > 0
def arithmetic_condition (a : ℕ → ℝ) (q : ℝ) := 
  a 1 = q * a 0 ∧ (1/2 : ℝ) * a 2 = a 1 + 2 * a 0

-- The statement to be proven
theorem geometric_seq_arith_condition_half (a : ℕ → ℝ) (q : ℝ) :
  geometric_seq a q →
  positive_terms a →
  arithmetic_condition a q →
  q = 2 →
  (a 2 + a 3) / (a 3 + a 4) = 1 / 2 :=
by
  intros h1 h2 h3 hq
  sorry

end geometric_seq_arith_condition_half_l751_751007


namespace a_5_value_l751_751796

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 1 => 1989 ^ 1989
| (n+1) => digit_sum (a n)

theorem a_5_value : a 5 = 9 := by
  sorry  -- To be filled with the actual proof steps

end a_5_value_l751_751796


namespace evaluate_integral_l751_751405

noncomputable def integral_value : ℝ := 
∫ x in 2..3, (x - 2) / ((x - 1) * (x - 4))

theorem evaluate_integral :
  integral_value = -1/3 * real.log 2 :=
by 
  sorry

end evaluate_integral_l751_751405


namespace vertical_asymptote_l751_751005

theorem vertical_asymptote (x : ℝ) : 4 * x - 9 = 0 → x = 9 / 4 := by
  sorry

end vertical_asymptote_l751_751005


namespace angle_AXP_45_degrees_l751_751583

noncomputable theory

-- Given conditions
variables {A B C I D P Q X : Type*}

-- Definitions based on given conditions
def triangle_has_incenter (A B C I D : Type*) : Prop := sorry -- Definition of having an incenter
def segment_intersects_incircle (AI D : Type*) : Prop := sorry -- Definition of segment intersection with incircle
def line_is_perpendicular (BD AC : Type*) : Prop := sorry -- Definition of lines being perpendicular
def angle_condition (P A I : Type*) : Prop := true /- ∃ (P : point), P ∈ angle_relations A I 90 -/ -- Definition of angle 90 degrees given the points
def point_lies_on_segment (Q BD : Type*) : Prop := sorry -- Definition of a point lying on a segment
def circumcircle_tangent (A B Q BI : Type*) : Prop := sorry -- Definition of circumcircle tangency
def point_lies_on_line (X PQ : Type*) : Prop := sorry -- Definition of a point lying on a line
def angle_equality (I A X C : Type*) : Prop := sorry -- Definition of angle equality \angle IAX = \angle XAC

-- The main theorem to prove
theorem angle_AXP_45_degrees {A B C I D P Q X : Type*}
  (h1 : triangle_has_incenter A B C I D)
  (h2 : segment_intersects_incircle AI D)
  (h3 : line_is_perpendicular BD AC)
  (h4 : angle_condition P A I)
  (h5 : point_lies_on_segment Q BD)
  (h6 : circumcircle_tangent A B Q BI)
  (h7 : point_lies_on_line X PQ)
  (h8 : angle_equality I A X C) :
  ∠AXP = 45 := 
sorry

end angle_AXP_45_degrees_l751_751583


namespace monotonicity_of_f_f_greater_than_2_ln_a_plus_3_div_2_l751_751048

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x y, x < y → f x a > f y a) ∧
  (a > 0 →
    (∀ x, x < Real.log (1 / a) → f x a > f (Real.log (1 / a)) a) ∧
    (∀ x, x > Real.log (1 / a) → f x a > f (Real.log (1 / a)) a)) :=
sorry

theorem f_greater_than_2_ln_a_plus_3_div_2 (a : ℝ) (h : a > 0) (x : ℝ) :
  f x a > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_of_f_f_greater_than_2_ln_a_plus_3_div_2_l751_751048


namespace solve_for_x_l751_751632

theorem solve_for_x (x : ℤ) : 3 * (5 - x) = 9 → x = 2 :=
by {
  sorry
}

end solve_for_x_l751_751632


namespace determinant_of_triangle_sines_eq_two_l751_751605

theorem determinant_of_triangle_sines_eq_two 
  {A B C : ℝ} 
  (hA: A + B + C = π) : 
  matrix.det ![
    [Real.sin A, 1, 1],
    [1, Real.sin B, 1],
    [1, 1, Real.sin C]
  ] = 2 := 
sorry

end determinant_of_triangle_sines_eq_two_l751_751605


namespace sin_comparison_l751_751545

-- Definitions of the conditions
def y_sin_increasing_on_interval : Prop :=
  ∀ x y, x ∈ set.Icc (0 : ℝ) (Real.pi / 2) → y ∈ set.Icc (0 : ℝ) (Real.pi / 2) → x < y → Real.sin x < Real.sin y

def three_pi_over_seven_in_interval : Prop :=
  (3 * Real.pi / 7) ∈ set.Icc (0 : ℝ) (Real.pi / 2)

def two_pi_over_five_in_interval : Prop :=
  (2 * Real.pi / 5) ∈ set.Icc (0 : ℝ) (Real.pi / 2)

def three_pi_over_seven_gt_two_pi_over_five : Prop :=
  (3 * Real.pi / 7) > (2 * Real.pi / 5)

-- The main theorem statement
theorem sin_comparison :
  y_sin_increasing_on_interval →
  three_pi_over_seven_gt_two_pi_over_five →
  three_pi_over_seven_in_interval →
  two_pi_over_five_in_interval →
  Real.sin (3 * Real.pi / 7) > Real.sin (2 * Real.pi / 5) :=
by
  intros h1 h2 h3 h4
  exact sorry

end sin_comparison_l751_751545


namespace points_connected_l751_751017

theorem points_connected (m l : ℕ) (h1 : l < m) (h2 : Even (l * m)) :
  ∃ points : Finset (ℕ × ℕ), ∀ p ∈ points, (∃ q, q ∈ points ∧ (p ≠ q → p.snd = q.snd → p.fst = q.fst)) :=
sorry

end points_connected_l751_751017


namespace inequality_holds_l751_751301

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l751_751301


namespace count_three_digit_values_satisfying_condition_l751_751140

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def is_three_digit (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem count_three_digit_values_satisfying_condition :
  (Finset.filter (λ x => sum_of_digits (sum_of_digits x) = 4 ∧ is_three_digit x) (Finset.range 1000)).card = 38 :=
by
  sorry

end count_three_digit_values_satisfying_condition_l751_751140


namespace triangle_sides_l751_751342

variables {r hypotenuse : ℝ}

-- Given Conditions
-- 1. The inradius
def inradius (r : ℝ) : ℝ := r

-- 2. The distance from the right-angle vertex to the center of the inscribed circle
def right_angle_to_incenter (dist : ℝ) : ℝ := dist

-- 3. Hypotenuse of length proportionality and circle inradius relationship
def hypotenuse_ratio (hypotenuse : ℝ) := hypotenuse = 5 * x

-- Main Statement to prove
theorem triangle_sides (x : ℝ) (r : ℝ) (hypotenuse : ℝ) 
  (h1 : inradius r = 2) 
  (h2 : right_angle_to_incenter (sqrt 8) = r * sqrt 2) 
  (h3 : hypotenuse_ratio hypotenuse) : 
  hypotenuse = 10 ∧ (2 * x + 2 = 6) ∧ (3 * x + 2 = 8) :=
begin
  unfold inradius at h1,
  unfold right_angle_to_incenter at h2,
  sorry
end

end triangle_sides_l751_751342


namespace complement_correct_l751_751138

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}
def complement_U (M: Set ℕ) (U: Set ℕ) := {x ∈ U | x ∉ M}

theorem complement_correct : complement_U M U = {3, 4, 6} :=
by 
  sorry

end complement_correct_l751_751138


namespace total_bill_l751_751638

/-
Ten friends dined at a restaurant and split the bill equally.
One friend, Chris, forgets his money.
Each of the remaining nine friends agreed to pay an extra $3 to cover Chris's share.
How much was the total bill?

Correct answer: 270
-/

theorem total_bill (t : ℕ) (h1 : ∀ x, t = 10 * x) (h2 : ∀ x, t = 9 * (x + 3)) : t = 270 := by
  sorry

end total_bill_l751_751638


namespace general_term_sum_terms_l751_751026

-- Definition of the sequence and the conditions
variable (a : ℕ+ → ℝ) (S : ℕ+ → ℝ)

-- Conditions
axiom condition_1 : ∀ n : ℕ+, a n + 1/2 = sqrt (2 * S n + 1/4)
axiom condition_2 : ∀ n : ℕ+, S n = (1/2) * (a n)^2 + (1/2) * (a n)
axiom positive_terms : ∀ n : ℕ+, a n > 0

-- Prove that the general term of the sequence is a_n = n
theorem general_term (n : ℕ+) : a n = n := 
by sorry

-- Define the sequence b_n
def b (n : ℕ+) := (1/2)^n * a n

-- Define the sum T_n of the sequence b_n
def T (n : ℕ+) := ∑ i in finset.range (n : ℕ), b ⟨i+1, nat.succ_pos _⟩

-- Prove the sum of the first n terms of b_n
theorem sum_terms (n : ℕ+) : T n = 2 - (2 + n) / 2^n :=
by sorry

end general_term_sum_terms_l751_751026


namespace students_book_difference_l751_751988

theorem students_book_difference (A B AB : ℕ) 
    (h1 : A + B + AB = 600)
    (h2 : AB = 0.20 * (A + AB))
    (h3 : AB = 0.25 * (B + AB)) :
    (A - B = 75) :=
sorry

end students_book_difference_l751_751988


namespace num_digits_8_22_5_19_l751_751812

def num_digits_in_decimal_form (n : ℕ) : ℕ :=
  have h1 : 8 = 2^3 := rfl
  have h2 : 8^22 = 2^66 := by sorry -- This follows from h1 and the properties of exponents
  have h3 : (8^22) * (5^19) = (2^66) * (5^19) := by rw [h2]
  have h4 : (2^66) * (5^19) = (2^(66 - 19)) * (10^19) := by sorry -- Simplification using 2*5=10
  have h5 : 2^(47) has 15 digits := by sorry
  have h6 : 10^19 has 20 digits := 20
  have h7 : total digits in (2^47) * (10^19) = 15 + 20 := 35 -- Combining the number of digits
  35

theorem num_digits_8_22_5_19 : num_digits_in_decimal_form (8^22 * 5^19) = 35 := 
by sorry -- Main proof.

end num_digits_8_22_5_19_l751_751812


namespace regular_hexagon_area_perimeter_l751_751752

noncomputable def inscribed_hexagon_area_perimeter
  (r : ℝ) (h_hexagon : ∀ (side_length : ℝ), side_length = r) :
  (area : ℝ) × (perimeter : ℝ) :=
  let side_length := r in
  let triangle_area := (side_length^2 * Real.sqrt 3) / 4 in
  let hexagon_area := 6 * triangle_area in
  let hexagon_perimeter := 6 * side_length in
  (hexagon_area, hexagon_perimeter)

theorem regular_hexagon_area_perimeter
  (hr : 3 > 0) :
  inscribed_hexagon_area_perimeter 3 (λ s, s = 3) =
  (27 * Real.sqrt 3 / 2, 18) :=
by
  have side_length := 3
  have triangle_area := (3^2 * Real.sqrt 3) / 4
  have hexagon_area := 6 * triangle_area
  have hexagon_perimeter := 6 * side_length
  show (hexagon_area, hexagon_perimeter) = (27 * Real.sqrt 3 / 2, 18)
  sorry

end regular_hexagon_area_perimeter_l751_751752


namespace probability_x_gt_3y_in_rectangle_l751_751615

noncomputable def probability_of_x_gt_3y :ℝ :=
  let base := 2010
  let height := 2011
  let triangle_height := 670
  (1/2 * base * triangle_height) / (base * height)

theorem probability_x_gt_3y_in_rectangle:
  probability_of_x_gt_3y = 335 / 2011 := 
by
  sorry

end probability_x_gt_3y_in_rectangle_l751_751615


namespace dot_product_a_b_norm_diff_a_b_l751_751890

-- Given unit vectors a and b
variables (a b : ℝ^3) (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)

-- Given the condition
axiom condition : (2 * a - 3 * b) • (2 * a + b) = 3

-- Prove the statements
theorem dot_product_a_b : a • b = -1 / 2 :=
by sorry

theorem norm_diff_a_b : ∥2 * a - b∥ = real.sqrt 7 :=
by sorry

end dot_product_a_b_norm_diff_a_b_l751_751890


namespace student_correct_answers_l751_751764

variable (C I : ℕ)

theorem student_correct_answers :
  C + I = 100 ∧ C - 2 * I = 76 → C = 92 :=
by
  intros h
  sorry

end student_correct_answers_l751_751764


namespace angle_405_eq_45_l751_751707

def same_terminal_side (angle1 angle2 : ℝ) : Prop :=
  ∃ k : ℤ, angle1 = angle2 + k * 360

theorem angle_405_eq_45 (k : ℤ) : same_terminal_side 405 45 := 
sorry

end angle_405_eq_45_l751_751707


namespace rectangle_x_value_l751_751690

theorem rectangle_x_value (x : ℝ) (h : (x - 3) * (3 * x + 4) = 12 * x - 7) (hx : x > 3) :
  x = (17 + Real.sqrt 349) / 6 := 
sorry

end rectangle_x_value_l751_751690


namespace exists_n_for_5_power_l751_751185

theorem exists_n_for_5_power (m : ℕ) (hm : 0 < m) : 
  ∃ n : ℕ, (0 < n) ∧ (∀ (d : ℕ), d < 10 ^ (Int.natAbs ⌈ Math.log 10 (5 ^ m) ⌉) →
  (5 ^ m) % (10 ^ d) = (5 ^ n) % (10 ^ d)) :=
sorry

end exists_n_for_5_power_l751_751185


namespace weight_of_square_piece_l751_751356

open Real

theorem weight_of_square_piece 
  (uniform_density : Prop)
  (side_length_triangle side_length_square : ℝ)
  (weight_triangle : ℝ)
  (ht : side_length_triangle = 6)
  (hs : side_length_square = 6)
  (wt : weight_triangle = 48) :
  ∃ weight_square : ℝ, weight_square = 27.7 :=
by
  sorry

end weight_of_square_piece_l751_751356


namespace max_n_for_Sn_neg_l751_751136

noncomputable def Sn (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem max_n_for_Sn_neg (a : ℕ → ℝ) (h1 : ∀ n : ℕ, (n + 1) * Sn n a < n * Sn (n + 1) a)
  (h2 : a 8 / a 7 < -1) :
  ∀ n : ℕ, S_13 < 0 ∧ S_14 > 0 →
  ∀ m : ℕ, m > 13 → Sn m a ≥ 0 :=
sorry

end max_n_for_Sn_neg_l751_751136


namespace sales_analysis_l751_751402

theorem sales_analysis :
  ∀ (s₁ s₂ : ℝ), s₁ = 25000 → s₂ = 4 * s₁ → s₂ = 100000 :=
by
  intros s₁ s₂ h₁ h₂
  rw [h₁, h₂]
  norm_num

end sales_analysis_l751_751402


namespace sets_partition_if_and_only_if_coprime_l751_751598

theorem sets_partition_if_and_only_if_coprime (r s n : ℕ) (hr : r > 0) (hs : s > 0) (hn : n > 0) (h_sum : r + s = n)
  (A : Set ℕ := {k | ∃ i, i ∈ Finset.range r ∧ k = ⌊(i * n) / r⌋})
  (B : Set ℕ := {k | ∃ j, j ∈ Finset.range s ∧ k = ⌊(j * n) / s⌋})
  (M : Set ℕ := {1, 2, .., n-2}) :
  (Disjoint A B ∧ (A ∪ B = M)) ↔ (Nat.coprime r n ∧ Nat.coprime s n) := 
sorry

end sets_partition_if_and_only_if_coprime_l751_751598


namespace combined_length_of_trains_l751_751687

open BigOperators

-- Condition definitions
def speed_train_a := 360 -- kmph
def time_cross_pole := 5 -- seconds
def speed_train_b := 420 -- kmph
def time_cross_each_other := 10 -- seconds

-- Conversion factors
def kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  speed_kmph * 1000 / 3600

-- Calculated values
def speed_train_a_mps := kmph_to_mps speed_train_a
def speed_train_b_mps := kmph_to_mps speed_train_b
def relative_speed_mps := speed_train_a_mps + speed_train_b_mps

-- Length calculations
def length_train_a := speed_train_a_mps * time_cross_pole

theorem combined_length_of_trains : (relative_speed_mps * time_cross_each_other) = 2166.7 := by
  sorry

end combined_length_of_trains_l751_751687


namespace inequality_proof_l751_751282

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751282


namespace find_x_l751_751835

-- Define the percentages and multipliers as constants
def percent_47 := 47.0 / 100.0
def percent_36 := 36.0 / 100.0

-- Define the given quantities
def quantity1 := 1442.0
def quantity2 := 1412.0

-- Calculate the percentages of the quantities
def part1 := percent_47 * quantity1
def part2 := percent_36 * quantity2

-- Calculate the expression
def expression := (part1 - part2) + 63.0

-- Define the value of x given
def x := 232.42

-- Theorem stating the proof problem
theorem find_x : expression = x := by
  -- proof goes here
  sorry

end find_x_l751_751835


namespace solve_inequality_l751_751830

theorem solve_inequality (x : ℝ) : (x - 3) * (x + 2) < 0 ↔ x ∈ Set.Ioo (-2) (3) :=
sorry

end solve_inequality_l751_751830


namespace student_A_did_not_pass_l751_751535

theorem student_A_did_not_pass (A B C D : Prop)
  (hA : A ↔ (A ∨ C ∨ D))
  (hB : B ↔ C)
  (hC : C ↔ (A ∨ B))
  (hD : D ↔ B)
  (exactly_two_true : (if A then 1 else 0) + (if B then 1 else 0) + (if C then 1 else 0) + (if D then 1 else 0) = 2)
  (false_implies_did_not_pass : ¬ C ∧ (A ∧ C)) :
  A := by
  have : A ∧ C, from false_implies_did_not_pass.2,
  sorry

end student_A_did_not_pass_l751_751535


namespace categorize_numbers_l751_751647

theorem categorize_numbers :
  let numbers := [(-4 : ℚ), (1 : ℚ), (-3/5 : ℚ), (0 : ℚ), -(-22/7 : ℚ), |(-3)|, -(+5 : ℚ), -(1/3 : ℚ)]
  let non_negative_integers := [1, 0, 3]
  let negative_fractions := [-3/5, -(1/3)]
  (∀ x ∈ non_negative_integers, x ∈ numbers ∧ x ≥ 0) ∧ 
  (∀ x ∈ negative_fractions, x ∈ numbers ∧ x < 0 ∧ is_fraction x) := 
  sorry

run_cmd
 -/
  #eval categorize_numbers
-/

end categorize_numbers_l751_751647


namespace div_n_by_8_eq_2_8089_l751_751602

theorem div_n_by_8_eq_2_8089
  (n : ℕ)
  (h : n = 16^2023) :
  n / 8 = 2^8089 := by
  sorry

end div_n_by_8_eq_2_8089_l751_751602


namespace tangent_equation_midpoint_coordinates_l751_751856

def circle (x y : ℝ) := (x - 3)^2 + (y - 4)^2 = 4

def line1 (x y : ℝ) (k : ℝ) := y = k * (x - 1)

def point_A : ℝ × ℝ := (1, 0)

def is_tangent (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), line x y ∧ ((x - 3)^2 + (y - 4)^2 = 4) ∧ (differentiable_iff_has_deriv_at ℝ).1
  (λ (x : ℝ), if line x 0 then x else sorry)

theorem tangent_equation :
  ∀ (x y : ℝ), is_tangent (λ x y, x = 1) ∨ is_tangent (λ (x y : ℝ), 3 * x - 4 * y - 3 = 0) :=
sorry

def intersects (line : ℝ → ℝ → Prop) : ℝ × ℝ := -- Function to find intersection points
  if line (4, 3) then (4, 3)
  else sorry

theorem midpoint_coordinates :
  ∀ (x1 y1 x2 y2 : ℝ), line1 x1 y1 1 ∧ line1 x2 y2 1 ∧ circle x1 y1 ∧ circle x2 y2 →
   (4, 3) = ((x1 + x2) / 2, (y1 + y2) / 2) :=
sorry

end tangent_equation_midpoint_coordinates_l751_751856


namespace juniper_initial_bones_l751_751128

theorem juniper_initial_bones (B : ℕ) (h : 2 * B - 2 = 6) : B = 4 := 
by
  sorry

end juniper_initial_bones_l751_751128


namespace grilled_cheese_sandwiches_l751_751946

-- Define the number of ham sandwiches Joan makes
def ham_sandwiches := 8

-- Define the cheese requirements for each type of sandwich
def cheddar_for_ham := 1
def swiss_for_ham := 1
def cheddar_for_grilled := 2
def gouda_for_grilled := 1

-- Total cheese used
def total_cheddar := 40
def total_swiss := 20
def total_gouda := 30

-- Prove the number of grilled cheese sandwiches Joan makes
theorem grilled_cheese_sandwiches (ham_sandwiches : ℕ) (cheddar_for_ham : ℕ) (swiss_for_ham : ℕ)
                                  (cheddar_for_grilled : ℕ) (gouda_for_grilled : ℕ)
                                  (total_cheddar : ℕ) (total_swiss : ℕ) (total_gouda : ℕ) :
    (total_cheddar - ham_sandwiches * cheddar_for_ham) / cheddar_for_grilled = 16 :=
by
  sorry

end grilled_cheese_sandwiches_l751_751946


namespace minimize_quadratic_function_l751_751435

noncomputable def quadratic_function (b : ℝ) : ℝ :=
  (1/2) * b^2 + 5 * b - 3

theorem minimize_quadratic_function :
  ∀ b : ℝ, quadratic_function (-5) ≤ quadratic_function (b) :=
begin
  intros b,
  have h : quadratic_function b = (1/2) * (b + 5)^2 - 31 / 2,
  { sorry },  -- Completing the square step
  rw h,
  have h_min : (1/2) * (b + 5)^2 ≥ 0,
  { exact mul_nonneg (by norm_num) (sq_nonneg (b + 5)) },
  linarith,
end

end minimize_quadratic_function_l751_751435


namespace crackers_initial_count_l751_751611

theorem crackers_initial_count (num_friends : ℕ) (crackers_per_friend : ℕ) (total_crackers : ℕ) 
  (h1 : num_friends = 18) (h2 : crackers_per_friend = 2) (h3 : total_crackers = num_friends * crackers_per_friend) 
  : total_crackers = 36 :=
by {
  rw [h1, h2] at h3,
  sorry
}

end crackers_initial_count_l751_751611


namespace max_min_values_l751_751215

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem max_min_values (a b : ℝ) (h₁ : a = 0) (h₂ : b = 3) :
  ∃ x₁ x₂, (x₁ ∈ set.Icc a b ∧ x₂ ∈ set.Icc a b) ∧ 
  (∀ x ∈ set.Icc a b, f x₁ ≤ f x ∧ f x ≤ f x₂) ∧ f x₁ = 2 ∧ f x₂ = 20 := 
by
  sorry

end max_min_values_l751_751215


namespace sum_of_digits_in_rectangle_l751_751992

theorem sum_of_digits_in_rectangle :
  ∃ A B C : ℕ,
    (4 + A + 1 + B = 12) ∧
    (4 + A + 1 + B = 6 + 6) ∧
    (C + 1 + 6 + C = 11) ∧
    (1 + B + 2 + C = 9) ∧
    (A + 8 + 8 = 8) ∧
    (A + 8 + B + 2 + C = 13) ∧
    (9 = 4 + A + 1 + B) ∧
    (B + 2 + C = 9) ∧    
    B = 5 ∧ A = 2 ∧ C = 6 :=
sorry

end sum_of_digits_in_rectangle_l751_751992


namespace concyclic_points_l751_751132

open EuclideanGeometry

variables {A B C D E F G : Point}

-- Let  \Delta ABC  be a scalene triangle
axiom is_scalene_triangle (hABC : Triangle A B C) : Scalene A B C

-- Points  D,E  lie on side  \overline{AC}  in the order,  A,E,D,C
axiom order_AEDC : same_order A E D C

-- Let the parallel through  E  to  BC  intersect  \odot (ABD)  at  F, 
-- such that,  E  and  F  lie on the same side of  AB
axiom parallel_EBC (hABC : Triangle A B C) (hE : lies_on E (Line A C)) (hF : lies_on F (Circle A B D)) : 
parallel (line_through_points E F) (line_through_points B C)

axiom same_side_EF (hABC : Triangle A B C) : same_side_line E F (line_through_points A B)

-- Let the parallel through  E  to  AB  intersect  \odot (BDC)  at  G , 
-- such that,  E  and  G  lie on the same side of  BC
axiom parallel_EAB (hABC : Triangle A B C) (hE : lies_on E (Line A C)) (hG : lies_on G (Circle B D C)) : 
parallel (line_through_points E G) (line_through_points A B)

axiom same_side_EG (hABC : Triangle A B C) : same_side_line E G (line_through_points B C)

-- Prove Points  D, F, E, G  are concyclic
theorem concyclic_points (hABC : Triangle A B C) (hscalene : is_scalene_triangle hABC) 
  (horder : order_AEDC) (hparallel1 : parallel_EBC hABC)
  (hside1 : same_side_EF hABC) (hparallel2 : parallel_EAB hABC)
  (hside2 : same_side_EG hABC) : concyclic {D, F, E, G} := 
sorry

end concyclic_points_l751_751132


namespace free_square_positions_l751_751235

theorem free_square_positions {board : Fin 8 × Fin 8} {rectangles : Fin 21} :
  (∃ (free_square : Fin 8 × Fin 8), 
    (∀ (rect : rectangles), covers_three_colors rect) ∧ 
    (count_covered_squares board rectangles = 63) ∧ 
    (count_uncovered_squares board rectangles = 1))
  →
  (free_square = (3, 3) ∨ free_square = (3, 6) ∨ free_square = (6, 3) ∨ free_square = (6, 6)) := sorry

end free_square_positions_l751_751235


namespace min_tokens_in_grid_l751_751697

theorem min_tokens_in_grid : 
  ∀ (G : Type) [fintype G] (grid : G → G → Prop),
  (∀ x y : G, set.Squares4x4 x y grid → ∃ t : ℕ, t ≥ 8) →
  ∃ m : ℕ, m = 4801 :=
by sorry

end min_tokens_in_grid_l751_751697


namespace transform_G_l751_751248

-- Define the transformations
def T1 (x : Char) : Char := 
  if x = 'R' then 'y'
  else if x = 'L' then '\urcorner'
  else if x = 'G' then '\cap'
  else x

def T2 (x : Char) : Char := 
  if x = 'y' then 'B'
  else if x = '\urcorner' then '\Gamma'
  else if x = '\cap' then '\cup'
  else x

-- Define the transformation properties for R and L
def transform_R : Prop := T2 (T1 'R') = 'B'
def transform_L : Prop := T2 (T1 'L') = '\Gamma'

-- Main theorem to prove
theorem transform_G : T2 (T1 'G') = '\cup' :=
by
  -- Use the given conditions about transformations
  have hR : transform_R := sorry
  have hL : transform_L := sorry
  -- Prove the goal based solely on the conditions.
  sorry

end transform_G_l751_751248


namespace tank_capacity_l751_751354

noncomputable def inlet_rate_1 : ℝ := 2.5 * 60 -- liters per hour
noncomputable def inlet_rate_2 : ℝ := 1.5 * 60 -- liters per hour
noncomputable def leak_rate_1 (C : ℝ) : ℝ := C / 6 -- liters per hour
noncomputable def leak_rate_2 (C : ℝ) : ℝ := C / 12 -- liters per hour
noncomputable def net_rate (C : ℝ) : ℝ := (inlet_rate_1 + inlet_rate_2) - (leak_rate_1 C + leak_rate_2 C) -- liters per hour

theorem tank_capacity : ∃ C : ℝ, (net_rate C * 8 = C) ∧ C = 640 :=
by
  have h1 : inlet_rate_1 = 150 := by norm_num
  have h2 : inlet_rate_2 = 90 := by norm_num
  have h3 : ∀ (C : ℝ), leak_rate_1 C = C / 6 := by norm_num
  have h4 : ∀ (C : ℝ), leak_rate_2 C = C / 12 := by norm_num
  -- Now applying the values and conditions and to state the equivalence
  use 640
  sorry -- skipping the proof steps specific to Lean

end tank_capacity_l751_751354


namespace possible_to_place_12_numbers_on_cube_edges_l751_751558

-- Define a list of numbers from 1 to 12
def nums := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the faces of the cube in terms of the indices of nums list
def top_face := [1, 2, 9, 10]
def bottom_face := [3, 5, 6, 8]

-- Define the product of the numbers on the faces of the cube
def product_face (face : List Nat) : Nat := face.foldr (*) 1

-- The lean statement proving the problem
theorem possible_to_place_12_numbers_on_cube_edges :
  product_face (top_face.map (λ i => nums.get! (i - 1))) =
  product_face (bottom_face.map (λ i => nums.get! (i - 1))) :=
by
  sorry

end possible_to_place_12_numbers_on_cube_edges_l751_751558


namespace find_a_l751_751861

noncomputable def f (a : ℝ) (x : ℝ) := 
if x < (1 : ℝ)/3 then 
  3 - Real.sin (a * x) 
else 
  a * x + Real.logBase 3 x

theorem find_a (a : ℝ) (h_pos : a > 0) : 
  (∃ x : ℝ, f a x = 1) ↔ a = 6 := 
sorry

end find_a_l751_751861


namespace verify_f_form_f_ge_neg_x_sq_plus_x_find_range_k_l751_751879

-- Given the function f(x) = e^x - x^2 + a and the tangent line at x = 0 is y = bx
def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - x^2 + a

-- 1. Verify that f(x) = e^x - x^2 - 1
theorem verify_f_form (x : ℝ) : ∃ a, (f x a = Real.exp x - x^2 - 1) :=
begin
  use -1,
  simp [f],
  sorry
end

-- 2. Prove that for the specific function f(x) with a = -1, f(x) ≥ -x^2 + x ∀ x ∈ ℝ
theorem f_ge_neg_x_sq_plus_x (x : ℝ) : f x -1 ≥ -x^2 + x :=
begin
  have h := show f x -1 ≥ -x^2 + x,
  sorry,
end

-- 3. Find the range of k such that f(x) > kx ∀ x ∈ (0, +∞)
theorem find_range_k (x : ℝ) : (∀ x > 0, f x -1 > k * x) → k < Real.exp 1 - 2 :=
begin
  have h := show k < Real.exp 1 - 2,
  sorry
end

end verify_f_form_f_ge_neg_x_sq_plus_x_find_range_k_l751_751879


namespace smallest_positive_n_l751_751260

theorem smallest_positive_n : ∃ (n : ℕ), n = 626 ∧ ∀ m : ℕ, m < 626 → ¬ (sqrt m - sqrt (m - 1) < 0.02) := by
  sorry

end smallest_positive_n_l751_751260


namespace integer_ratio_condition_l751_751968

variable (x y : ℝ)

theorem integer_ratio_condition 
  (h : 3 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 6)
  (h_int : ∃ t : ℤ, x = t * y) :
  ∃ t : ℤ, t = -2 :=
by
  sorry

end integer_ratio_condition_l751_751968


namespace regular_triangular_prism_properties_l751_751855

-- Regular triangular pyramid defined
structure RegularTriangularPyramid (height : ℝ) (base_side : ℝ)

-- Regular triangular prism defined
structure RegularTriangularPrism (height : ℝ) (base_side : ℝ) (lateral_area : ℝ)

-- Given data
def pyramid := RegularTriangularPyramid 15 12
def prism_lateral_area := 120

-- Statement of the problem
theorem regular_triangular_prism_properties (h_prism : ℝ) (ratio_lateral_area : ℚ) :
  (h_prism = 10 ∨ h_prism = 5) ∧ (ratio_lateral_area = 1/9 ∨ ratio_lateral_area = 4/9) :=
sorry

end regular_triangular_prism_properties_l751_751855


namespace inequality_solution_l751_751194

theorem inequality_solution :
  { x : ℝ // x < 2 ∨ (3 < x ∧ x < 6) ∨ (7 < x ∧ x < 8) } →
  ((x - 3) * (x - 5) * (x - 7)) / ((x - 2) * (x - 6) * (x - 8)) > 0 :=
by
  sorry

end inequality_solution_l751_751194


namespace number_of_points_C_l751_751527

theorem number_of_points_C (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (dist_AB : dist A B = 10) (perimeter_ABC : ∀ C, dist A B + dist A C + dist B C = 50) 
  (area_ABC : ∀ C, abs (C.y - A.y) = 20) : 
  ∃! C, dist A B + dist A C + dist B C = 50 ∧ (abs (C.y - A.y) = 20) :=
sorry

end number_of_points_C_l751_751527


namespace B_visible_from_A_l751_751041

noncomputable def visibility_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 → 4 * x - 2 > 2 * x^2

theorem B_visible_from_A (a : ℝ) : visibility_condition a ↔ a < 10 :=
by
  -- sorry statement is used to skip the proof part.
  sorry

end B_visible_from_A_l751_751041


namespace problem1_problem2_l751_751487

variable (t : ℝ)

-- Problem 1
theorem problem1 (h : (4:ℝ) - 8 * t + 16 < 0) : t > 5 / 2 :=
sorry

-- Problem 2
theorem problem2 (hp: 4 - t > t - 2) (hq : t - 2 > 0) (hdisjoint : (∃ (p : Prop) (q : Prop), (p ∨ q) ∧ ¬(p ∧ q))):
  (2 < t ∧ t ≤ 5 / 2) ∨ (t ≥ 3) :=
sorry


end problem1_problem2_l751_751487


namespace squirrels_in_tree_l751_751671

theorem squirrels_in_tree (N S : ℕ) (h₁ : N = 2) (h₂ : S - N = 2) : S = 4 :=
by
  sorry

end squirrels_in_tree_l751_751671


namespace standard_spherical_coordinates_l751_751799

theorem standard_spherical_coordinates (ρ θ φ : ℝ) (hρ : ρ > 0) (hθ1 : θ = 15 * Real.pi / 4) (hφ : φ = 3 * Real.pi / 4) : 
  ρ = 5 ∧ (0 ≤ (θ - 2 * Real.pi) < 2 * Real.pi) ∧ φ = 3 * Real.pi / 4 :=
by
  have hθ : θ - 2 * Real.pi = 7 * Real.pi / 4, sorry
  exact ⟨rfl, ⟨by norm_num, hθ⟩, rfl⟩

end standard_spherical_coordinates_l751_751799


namespace find_n_l751_751394

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 1 then 2
  else if even n then 2 + sequence (n / 2)
  else 1 / (2 * sequence (n - 1))

theorem find_n (h : sequence 33 = 19 / 96) : n = 33 :=
sorry

end find_n_l751_751394


namespace gcd_of_rope_lengths_l751_751165

theorem gcd_of_rope_lengths : 
  let l := [42, 56, 63, 77]
  gcd_list l = 7 := 
by
  -- This is where the proof will go
  sorry

end gcd_of_rope_lengths_l751_751165


namespace value_of_expression_l751_751591

theorem value_of_expression {a b c : ℝ} (h_eqn : a + b + c = 15)
  (h_ab_bc_ca : ab + bc + ca = 13) (h_abc : abc = 8)
  (h_roots : Polynomial.roots (Polynomial.X^3 - 15 * Polynomial.X^2 + 13 * Polynomial.X - 8) = {a, b, c}) :
  (a / (1/a + b*c)) + (b / (1/b + c*a)) + (c / (1/c + a*b)) = 199/9 :=
by sorry

end value_of_expression_l751_751591


namespace log_b_a_values_l751_751465

theorem log_b_a_values (a b y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ 1) (h4 : b ≠ 1)
    (h : 4 * (Real.log y / Real.log a)^2 + 5 * (Real.log y / Real.log b)^2 = 10 * (Real.log y)^2) :
    Real.log b a = sqrt (5 / 6) ∨ Real.log b a = -sqrt (5 / 6) :=
by
  intro
  sorry

end log_b_a_values_l751_751465


namespace theta_range_pos_f_l751_751234

noncomputable def f (x θ : ℝ) : ℝ := 1 / (x - 2) ^ 2 - 2 * x + Real.cos (2 * θ) - 3 * Real.sin θ + 2

def g (x : ℝ) : ℝ := 2 * x - 1 / (x - 2) ^ 2

theorem theta_range_pos_f
  (θ : ℝ)
  (h1 : 0 < θ)
  (h2 : θ < Real.pi)
  (H : ∀ x : ℝ, x < 2 → f x θ > 0) :
  (0 < θ ∧ θ < Real.pi / 6) ∨ (5 * Real.pi / 6 < θ ∧ θ < Real.pi) :=
by
  sorry

end theta_range_pos_f_l751_751234


namespace river_flow_and_boat_schedule_l751_751678

/-- The conditions: distances, speeds, departure, and arrival times. --/
variables (distance_bc : ℝ := 20) (train_speed : ℝ := 30) 
variables (departure_b_to_c_1 : ℕ := 8 * 60 + 20) (departure_b_to_c_2 : ℕ := 10 * 60 + 20)
variables (arrival_c_to_b_1 : ℕ := 10 * 60) (arrival_c_to_b_2 : ℕ := 11 * 60 + 50)
variables (departure_c_to_b : ℕ := 10 * 60 + 25) 

/-- The conditions: the boat engine operates the same way on both trips. --/
variable (boat_engine_same : Prop := true)

/-- The main theorem to prove: river direction, speed, and boat schedule. --/
theorem river_flow_and_boat_schedule :
  (let river_direction := "C to B",
       river_speed := 1,
       boat_b_to_c_departure := 8 * 60,
       boat_b_to_c_arrival := 10 * 60,
       boat_c_to_b_departure := 10 * 60 + 25,
       boat_c_to_b_arrival := 12 * 60 + 25
     in 
     river_direction = "C to B" ∧
     river_speed = 1 ∧
     boat_b_to_c_departure = 8 * 60 ∧
     boat_b_to_c_arrival = 10 * 60 ∧
     boat_c_to_b_departure = 10 * 60 + 25 ∧
     boat_c_to_b_arrival = 12 * 60 + 25) :=
sorry

end river_flow_and_boat_schedule_l751_751678


namespace factory_profit_l751_751741

theorem factory_profit (x : ℕ) (h : x > 1000) :
  let cost := 500 + 2 * x
  let revenue := 2.5 * x
  revenue > cost :=
by
  sorry

end factory_profit_l751_751741


namespace least_positive_integer_remainder_l751_751693

theorem least_positive_integer_remainder :
  ∃ n : ℕ, (n > 0) ∧ (n % 5 = 1) ∧ (n % 4 = 2) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 5 = 1) ∧ (m % 4 = 2) → n ≤ m) :=
sorry

end least_positive_integer_remainder_l751_751693


namespace graph_chromatic_number_l751_751952

variable {V : Type} [Fintype V] [DecidableEq V]

def max_degree (G : SimpleGraph V) : ℕ := G.degree' (classical.some G.nonempty)

def chromatic_number_le (G : SimpleGraph V) (D : ℕ) 
  (hD: max_degree G ≤ D) : Prop := 
  G.chromaticNumber ≤ D + 1

theorem graph_chromatic_number (G : SimpleGraph V) (D : ℕ) 
  (h: max_degree G ≤ D) : chromatic_number_le G D h :=
sorry

end graph_chromatic_number_l751_751952


namespace find_a_n_plus_b_n_l751_751887

noncomputable def a (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else if n = 2 then 3 
  else sorry -- Placeholder for proper recursive implementation

noncomputable def b (n : ℕ) : ℕ := 
  if n = 1 then 5
  else sorry -- Placeholder for proper recursive implementation

theorem find_a_n_plus_b_n (n : ℕ) (i j k l : ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3) (h3 : b 1 = 5) 
  (h4 : i + j = k + l) (h5 : a i + b j = a k + b l) : a n + b n = 4 * n + 2 := 
by
  sorry

end find_a_n_plus_b_n_l751_751887


namespace ratio_of_houses_second_to_first_day_l751_751125

theorem ratio_of_houses_second_to_first_day 
    (houses_day1 : ℕ)
    (houses_day2 : ℕ)
    (sales_per_house : ℕ)
    (sold_pct_day2 : ℝ) 
    (total_sales_day1 : ℕ)
    (total_sales_day2 : ℝ) :
    houses_day1 = 20 →
    sales_per_house = 2 →
    sold_pct_day2 = 0.8 →
    total_sales_day1 = houses_day1 * sales_per_house →
    total_sales_day2 = sold_pct_day2 * houses_day2 * sales_per_house →
    total_sales_day1 = total_sales_day2 →
    (houses_day2 : ℝ) / houses_day1 = 5 / 4 :=
by
    intro h1 h2 h3 h4 h5 h6
    sorry

end ratio_of_houses_second_to_first_day_l751_751125


namespace problem_1_problem_2_l751_751878

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.logBase 9 (9^x + 1) + k * x

theorem problem_1 (k : ℝ) : 
  (∀ x, f x k = f (-x) k) → k = -1/2 := 
by
  intro h
  -- Proof must be filled by the user
  sorry

theorem problem_2 (a x : ℝ) (h_a : a > 0) : 
  (f x (-1/2) - Real.logBase 9 (a + 1/a) > 0) ↔ 
  (a ≠ 1 → (a > 1 → (x > Real.logBase 3 a ∨ x < Real.logBase 3 (1/a))) ∧ 
           (a < 1 → (x > Real.logBase 3 (1/a) ∨ x < Real.logBase 3 a))) ∧ 
  (a = 1 → x ≠ 0) := 
by
  intro ha
  -- Proof must be filled by the user
  sorry

end problem_1_problem_2_l751_751878


namespace complex_pow_diff_zero_l751_751779

theorem complex_pow_diff_zero {i : ℂ} (h : i^2 = -1) : (2 + i)^(12) - (2 - i)^(12) = 0 := by
  sorry

end complex_pow_diff_zero_l751_751779


namespace log_value_power_function_l751_751480

-- Define the power function condition
def power_function (x : ℝ) (α : ℝ) : ℝ := x ^ α

-- The condition that the function passes through the point (2, 4)
def passes_through (α : ℝ) : Prop :=
  power_function 2 α = 4

-- Proof problem
theorem log_value_power_function (α : ℝ) (h : passes_through α) :
  Real.log (power_function (1/2) α) / Real.log 2 = -2 :=
by
  -- The proof will be completed here.
  sorry

end log_value_power_function_l751_751480


namespace find_x_eq_zero_l751_751815

theorem find_x_eq_zero (x : ℝ) : 3^4 * 3^x = 81 → x = 0 := by
  sorry

end find_x_eq_zero_l751_751815


namespace obtuse_angle_inequality_l751_751013

theorem obtuse_angle_inequality
    (γ : ℝ)
    (a b c T : ℝ)
    (h1 : T > 0)
    (h2 : a > 0)
    (h3 : b > 0)
    (h4 : c > 0)
    (h_inequality : T / Real.sqrt (a^2 * b^2 - 4 * T^2) 
                    + T / Real.sqrt (b^2 * c^2 - 4 * T^2)
                    + T / Real.sqrt (c^2 * a^2 - 4 * T^2) 
                    ≥ 3 * Real.sqrt 3 / 2) :
    105.248 * (Math.pi / 180) ≥ γ ∧ γ > (Math.pi / 2) :=
sorry

end obtuse_angle_inequality_l751_751013


namespace total_fish_in_tanks_l751_751121

theorem total_fish_in_tanks :
  ∃ (tank1_fish tank2_fish tank3_fish : ℕ), tank1_fish = 20 ∧ tank2_fish = 2 * tank1_fish ∧ tank3_fish = 2 * tank1_fish ∧ tank1_fish + tank2_fish + tank3_fish = 100 :=
by
  exists 20
  exists 40
  exists 40
  simp
  split
  rfl
  split
  rfl
  split
  rfl 
  simp
  sorry

end total_fish_in_tanks_l751_751121


namespace average_effective_increase_correct_l751_751674

noncomputable def effective_increase (initial_price: ℕ) (price_increase_percent: ℕ) (discount_percent: ℕ) : ℕ :=
let increased_price := initial_price + (initial_price * price_increase_percent / 100)
let final_price := increased_price - (increased_price * discount_percent / 100)
(final_price - initial_price) * 100 / initial_price

noncomputable def average_effective_increase : ℕ :=
let increase1 := effective_increase 300 10 5
let increase2 := effective_increase 450 15 7
let increase3 := effective_increase 600 20 10
(increase1 + increase2 + increase3) / 3

theorem average_effective_increase_correct :
  average_effective_increase = 6483 / 100 :=
by
  sorry

end average_effective_increase_correct_l751_751674


namespace range_of_k_sum_of_roots_when_max_k_l751_751218

-- Given definitions and conditions
def quadratic (k x : ℝ) := k * x^2 - 6 * x + 1
def has_two_real_roots (k : ℝ) := (hb : (-6)^2 - 4 * k * 1 ≥ 0) ∧ (k ≠ 0)

-- Proving the range of k
theorem range_of_k (k : ℝ) : has_two_real_roots k → 0 < k ∧ k ≤ 9 :=
by
  intros
  sorry

-- Finding the sum of the roots when k = 9
theorem sum_of_roots_when_max_k : (x1 x2 : ℝ) (h : k = 9) →
  (quad_eq : quadratic k x1 = 0 ∧ quadratic k x2 = 0) →
  (x1 + x2 = 2 / 3) :=
by
  intros
  sorry

end range_of_k_sum_of_roots_when_max_k_l751_751218


namespace problem_PB_value_l751_751955

theorem problem_PB_value
  (P O T A B : Type)
  (h1 : ¬P ∈ O)
  (tangent : ∀ (P : Type), is_tangent P O T)
  (secant : ∀ (P : Type), secant_line P O A B)
  (h2 : PA < PB)
  (h3 : PA = 4)
  (h4 : PT = 2 * (PA - AB)) :
  PB = 7 :=
by
  sorry

end problem_PB_value_l751_751955


namespace tank_capacity_is_1592_litres_l751_751743

theorem tank_capacity_is_1592_litres :
  ∃ C : ℝ, C = 1592 ∧ 
           (let leak_rate := C / 7 in 
            let inlet_rate := 6 * 60 in
            let net_rate := C / 12 in
            inlet_rate - leak_rate = net_rate) := 
sorry

end tank_capacity_is_1592_litres_l751_751743


namespace even_number_of_convenient_numbers_l751_751160

def is_convenient (n : ℕ) : Prop :=
  (n^2 + 1) % 1000001 = 0

theorem even_number_of_convenient_numbers :
  (1 ≤ 1000000) →
  let convenient_count := (Finset.range 1000001).filter is_convenient in
  even (Finset.card convenient_count) :=
sorry

end even_number_of_convenient_numbers_l751_751160


namespace sin_450_eq_1_l751_751387

theorem sin_450_eq_1 : Real.sin (450 * Real.pi / 180) = 1 := by
  -- Using the fact that angle measures can be reduced modulo 2π radians (360 degrees)
  let angle := 450 * Real.pi / 180
  let reduced_angle := angle % (2 * Real.pi)
  have h1 : reduced_angle = Real.pi / 2 := by
    -- 450 degrees is 450 * π / 180 radians, which simplifies to π / 2 radians mod 2π
    calc
      reduced_angle = (450 * Real.pi / 180) % (2 * Real.pi)  : rfl
      ... = (5 * Real.pi / 2) % (2 * Real.pi)                : by simp [mul_div_assoc, Real.pi_div_two]
      ... = Real.pi / 2                                       : by norm_num1
  -- The sine of π / 2 radians is 1
  rw [h1, Real.sin_pi_div_two]
  exact rfl

end sin_450_eq_1_l751_751387


namespace new_babysitter_charge_per_scream_l751_751169

theorem new_babysitter_charge_per_scream
    (current_hourly_rate : ℕ)
    (new_hourly_rate : ℕ)
    (hours : ℕ)
    (screams : ℕ)
    (cost_difference : ℕ)
    (current_total_cost : ℕ := current_hourly_rate * hours)
    (new_total_cost_without_scream : ℕ := new_hourly_rate * hours)
    (new_total_cost_with_scream : ℕ := current_total_cost - cost_difference) :
    (new_total_cost_with_scream - new_total_cost_without_scream) / screams = 3 := by
  sorry

#eval current_total_cost 16 ụdd6 -- This should evaluate to 96
#eval new_total_cost_without_scream 12  6 -- This should evaluate to 72
#eval new_total_cost_with_scream 96 18 -- This should evaluate to 78

end new_babysitter_charge_per_scream_l751_751169


namespace part1_part2_l751_751476

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a*x^2 - 2*a*x + 2 + a
noncomputable def g (x : ℝ) (m : ℝ) (a : ℝ) : ℝ := f x a - m*x

theorem part1 (a : ℝ) (h : a < 0) (h_max : ∀ x ∈ Icc 2 3, f x a ≤ 1) : a = -1 := 
sorry

theorem part2 (m : ℝ) (h_a : a = -1) (h_mono : ∀ x y ∈ Icc 2 4, (x ≤ y → g x m (-1) ≤ g y m (-1))) : 
m ≤ -6 ∨ m ≥ -2 :=
sorry

end part1_part2_l751_751476


namespace units_digit_calculation_l751_751424

theorem units_digit_calculation : 
  let units_digit (n : ℕ) := n % 10 in
  units_digit (9 * 19 * 1989 - 9^3) = 0 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_calculation_l751_751424


namespace problem_l751_751998

-- Define the values in the grid
def grid : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ := (4, 3, 1, 1, 6, 2, 3)

-- Define the variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def condition_1 := (A = 3) ∧ (B = 2) ∧ (C = 4)
def condition_2 := (4 + A + 1 + B + C + 3 = 9)
def condition_3 := (A + 1 + 6 = 9)
def condition_4 := (1 + A + 6 = 9)
def condition_5 := (B + 2 + C + 5 = 9)

-- Define that the sum of the red cells is equal to any row
def sum_of_red_cells := (A + B + C = 9)

-- The final goal to prove
theorem problem : condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5 ∧ sum_of_red_cells := 
by {
  refine ⟨_, _, _, _, _, _⟩;
  sorry   -- proofs for each condition
}

end problem_l751_751998


namespace math_problem_solution_l751_751646

def equation_has_roots (a b c d : ℕ) : Prop :=
  ∀ x : ℝ, 
  (1 / x + 1 / (x + 3) - 1 / (x + 5) - 1 / (x + 7) - 1 / (x + 9) - 1 / (x + 11) + 1 / (x + 13) + 1 / (x + 15) = 0) ↔
  (∃ t : ℝ, 
  x = -a + t ∨ x = -a - t ∨ 
  x = -a + √(b + c * √d) ∨ x = -a - √(b - c * √d))

noncomputable def find_constants : ℕ := 8 + 19 + 6 + 5

theorem math_problem_solution :
  equation_has_roots 8 19 6 5 → find_constants = 38 := by
  sorry

end math_problem_solution_l751_751646


namespace inequality_inequality_holds_l751_751321

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l751_751321


namespace problem_statement_l751_751028

noncomputable def P (X : set α) : ℝ := sorry

variables (A B : set α)
variable [MeasureSpace α]

def independent (A B : set α) : Prop := P (A ∩ B) = P (A) * P (B)

def complement (X : set α) : set α := {x | x ∉ X}

theorem problem_statement
  (hA : P A = 0.7)
  (hB : P B = 0.2)
  (h_indep : independent A B) :
  P (complement A ∩ complement B) = 0.24 ∧
  P (complement A ∩ B) = 0.06 :=
begin
  sorry
end

end problem_statement_l751_751028


namespace coefficient_x4_in_expansion_l751_751419

theorem coefficient_x4_in_expansion :
  let f (x : ℝ) := (1 + x) * (1 + 2 * x) ^ 5 in
  (∃ c : ℝ, c * x^4 = (∑ i in finset.range 0 6, (binom 5 i) * 2^i * ⟨polynomial.C 1 x⟩ * ⟨polynomial.monomial i (2^5-i⟩)) 
  ∧ c = 160 :=
begin
  sorry
end

end coefficient_x4_in_expansion_l751_751419


namespace parallel_line_plane_l751_751059

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry

-- Predicate for parallel lines
noncomputable def is_parallel_line (a b : line) : Prop := sorry

-- Predicate for parallel line and plane
noncomputable def is_parallel_plane (a : line) (α : plane) : Prop := sorry

-- Predicate for line contained within the plane
noncomputable def contained_in_plane (b : line) (α : plane) : Prop := sorry

theorem parallel_line_plane
  (a b : line) (α : plane)
  (h1 : is_parallel_line a b)
  (h2 : ¬ contained_in_plane a α)
  (h3 : contained_in_plane b α) :
  is_parallel_plane a α :=
sorry

end parallel_line_plane_l751_751059


namespace cube_edge_numbers_possible_l751_751566

theorem cube_edge_numbers_possible :
  ∃ (top bottom : Finset ℕ), top.card = 4 ∧ bottom.card = 4 ∧ 
  (top ∪ bottom).card = 8 ∧ 
  ∀ (n : ℕ), n ∈ top ∪ bottom → n ∈ (Finset.range 12).map (+1) ∧ 
  (∏ x in top, x) = (∏ y in bottom, y) :=
by {
  sorry,
}

end cube_edge_numbers_possible_l751_751566


namespace difference_between_mean_and_median_l751_751006

theorem difference_between_mean_and_median :
  let students := [0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 5, 5] in
  let n := length students in
  n = 15 →
  let median := List.nthLe (students.sort Nat.le) (n / 2) sorry in
  let mean := (List.sum students) / n in
  abs (mean - median) = 1 / 3 := sorry

end difference_between_mean_and_median_l751_751006


namespace total_length_circle_l751_751630

-- Definitions based on conditions
def num_strips : ℕ := 16
def length_each_strip : ℝ := 10.4
def overlap_each_strip : ℝ := 3.5

-- Theorem stating the total length of the circle-shaped colored tape
theorem total_length_circle : 
  (num_strips * length_each_strip) - (num_strips * overlap_each_strip) = 110.4 := 
by 
  sorry

end total_length_circle_l751_751630


namespace cos_sin_identity_l751_751784

theorem cos_sin_identity : 
  (Real.cos (14 * Real.pi / 180) * Real.cos (59 * Real.pi / 180) + 
   Real.sin (14 * Real.pi / 180) * Real.sin (121 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  sorry

end cos_sin_identity_l751_751784


namespace imaginary_part_z_l751_751873

variable (a : ℝ)
def z := (1 - a * Complex.i) / (1 + Complex.i)

theorem imaginary_part_z : Complex.im z = -2 :=
by
  have h_real : Complex.re z = -1 := sorry
  sorry

end imaginary_part_z_l751_751873


namespace train_speed_is_correct_l751_751765

-- Definitions of the given conditions.
def train_length : ℕ := 250
def bridge_length : ℕ := 150
def time_taken : ℕ := 20

-- Definition of the total distance covered by the train.
def total_distance : ℕ := train_length + bridge_length

-- The speed calculation.
def speed : ℕ := total_distance / time_taken

-- The theorem that we need to prove.
theorem train_speed_is_correct : speed = 20 := by
  -- proof steps go here
  sorry

end train_speed_is_correct_l751_751765


namespace arithmetic_sequence_value_l751_751450

theorem arithmetic_sequence_value :
  ∀ (a : ℕ → ℤ), 
  a 1 = 1 → 
  a 3 = -5 → 
  (a 1 - a 2 - a 3 - a 4 = 16) :=
by
  intros a h1 h3
  sorry

end arithmetic_sequence_value_l751_751450


namespace probability_club_then_queen_l751_751237

theorem probability_club_then_queen : 
  let total_cards := 52
  let total_clubs := 13
  let total_queens := 4
  let queen_of_clubs := 1
  let non_queen_clubs := total_clubs - queen_of_clubs
  
  let prob_queen_of_clubs_then_other_queen := (queen_of_clubs / total_cards) * ((total_queens - 1) / (total_cards - 1))
  let prob_non_queen_clubs_then_queen := (non_queen_clubs / total_cards) * (total_queens / (total_cards - 1))
  let total_probability := prob_queen_of_clubs_then_other_queen + prob_non_queen_clubs_then_queen
  
  total_probability = 1 / 52 := by
  let total_cards := 52
  let total_clubs := 13
  let total_queens := 4
  let queen_of_clubs := 1
  let non_queen_clubs := total_clubs - queen_of_clubs
  
  let prob_queen_of_clubs_then_other_queen := (queen_of_clubs / total_cards) * ((total_queens - 1) / (total_cards - 1))
  let prob_non_queen_clubs_then_queen := (non_queen_clubs / total_cards) * (total_queens / (total_cards - 1))
  let total_probability := prob_queen_of_clubs_then_other_queen + prob_non_queen_clubs_then_queen
  
  sorry

end probability_club_then_queen_l751_751237


namespace king_middle_school_teachers_l751_751581

/-- King Middle School has 1200 students, each student takes 5 classes a day, each teacher teaches
4 classes, each class has 30 students and 1 teacher. We need to show that the number of teachers is 50. -/
theorem king_middle_school_teachers:
  let students := 1200 in
  let classes_per_student := 5 in
  let classes_per_teacher := 4 in
  let students_per_class := 30 in
  let total_classes := students * classes_per_student in
  let unique_classes := total_classes / students_per_class in
  let teachers := unique_classes / classes_per_teacher in
  teachers = 50 :=
by
  simp [students, classes_per_student, classes_per_teacher, students_per_class, total_classes, unique_classes, teachers]
  sorry

end king_middle_school_teachers_l751_751581


namespace right_triangle_median_length_l751_751104

theorem right_triangle_median_length
  (D E F : Type*)
  [metric_space D] [metric_space E] [metric_space F]
  (DE DF EF : ℝ)
  (hDE : DE = 15)
  (hDF : DF = 9)
  (hEF : EF = 12)
  (h_right : is_right_triangle D E F) :
  distance_to_midpoint F DE = 7.5 :=
by sorry

end right_triangle_median_length_l751_751104


namespace remaining_food_after_days_l751_751755

theorem remaining_food_after_days (initial_food : ℕ) (used_after_one_day : ℚ) (used_after_two_days : ℚ) :
  initial_food = 400 →
  used_after_one_day = (2/5 : ℚ) →
  used_after_two_days = (3/5 : ℚ) →
  let remaining_after_one_day := initial_food - (initial_food * used_after_one_day).to_nat in
  let remaining_after_two_days := remaining_after_one_day - (remaining_after_one_day * used_after_two_days).to_nat in
  remaining_after_two_days = 96 :=
by
  intros h_initial_food h_used_after_one_day h_used_after_two_days
  simp [h_initial_food, h_used_after_one_day, h_used_after_two_days]
  sorry

end remaining_food_after_days_l751_751755


namespace remainder_ab_cd_l751_751595

theorem remainder_ab_cd (n : ℕ) (hn: n > 0) (a b c d : ℤ) 
  (hac : a * c ≡ 1 [ZMOD n]) (hbd : b * d ≡ 1 [ZMOD n]) : 
  (a * b + c * d) % n = 2 :=
by
  sorry

end remainder_ab_cd_l751_751595


namespace minimum_15_equal_differences_l751_751143

-- Definition of distinct integers a_i
def distinct_sequence (a : Fin 100 → ℕ) : Prop :=
  ∀ i j : Fin 100, i < j → a i < a j

-- Definition of the differences d_i
def differences (a : Fin 100 → ℕ) (d : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, d i = a ⟨i + 1, Nat.lt_of_lt_of_le (Nat.succ_lt_succ i.2) (by norm_num)⟩ - a i

-- Main theorem statement
theorem minimum_15_equal_differences (a : Fin 100 → ℕ) (d : Fin 99 → ℕ) :
  (∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 400) →
  distinct_sequence a →
  differences a d →
  ∃ t : Finset ℕ, t.card ≥ 15 ∧ ∀ x : ℕ, x ∈ t → (∃ i j : Fin 99, i ≠ j ∧ d i = x ∧ d j = x) :=
sorry

end minimum_15_equal_differences_l751_751143


namespace greatest_pow2_factorial_20_l751_751911

theorem greatest_pow2_factorial_20 : 
  let n := Nat.factorial 20 in
  ∃ k : ℕ, (2 ^ k ∣ n) ∧ (∀ m : ℕ, (m > k → ¬ (2 ^ m ∣ n))) ∧ k = 18 := 
sorry

end greatest_pow2_factorial_20_l751_751911


namespace ellipse_equation_area_range_l751_751456

noncomputable def ellipse : Type := 
  {a b : ℝ // a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, (x^2 / (a^2) + y^2 / (b^2)) = 1}

def point_on_ellipse (e : ellipse) : Prop :=
  let ⟨a, b, hab⟩ := e in ∃ (P : ℝ × ℝ), P = (-1, real.sqrt 2 / 2) 

theorem ellipse_equation
  (e : ellipse) (P : ℝ × ℝ) (F1 : ℝ × ℝ) (F2 : ℝ × ℝ)
  (hP : point_on_ellipse e) (hF1 : F1 = (-1, 0)) (hF2 : F2 = (1, 0))
  (hM : ∃ M : ℝ × ℝ, P - M = M - F2) : 
  e.1 * e.1 = 2 ∧ e.2 * e.2 = 1 :=
begin
  sorry
end

def segment_area (F1 C D : ℝ × ℝ) : ℝ :=
  0.5 * real.abs ((F1.1 - C.1) * (F1.2 - D.2) - (F1.2 - C.2) * (F1.1 - D.1))

def range_area (F1 F2 C D : ℝ × ℝ) (l : ℝ) : Prop :=
  ∃ S : set ℝ, S = Icc (4 * real.sqrt 3 / 5) (4 * real.sqrt 6 / 7) ∧
    segment_area F1 C D ∈ S

theorem area_range
  (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) (C D : ℝ × ℝ)
  (hF1 : F1 = (-1, 0)) (hF2 : F2 = (1, 0)) (hSegment : segment_area F1 C D)
  (hl : ∃ l : Prop, l ∧ ∃ (A B : ℝ × ℝ), (A B coincide with circle and line l))
  (hLambda : ∀ 𝜆 : ℝ, 𝜆 ∈ Icc (2 / 3) 1) :
  range_area F1 F2 C D :=
begin
  sorry
end

end ellipse_equation_area_range_l751_751456


namespace incircle_radius_l751_751680

-- Definitions based on conditions
variables {D E F : Type} [EuclideanGeometry] [Triangle DEF]
variables (right_angle_F : ∠D F E = 90°) (angle_D : ∠D = 45°) (side_DF : length D F = 8)

-- Theorem statement
theorem incircle_radius (h : right_angle_F) (h' : angle_D) (h'' : side_DF) : 
  incircle_radius DEF = 4 - 2 * Real.sqrt 2 := 
sorry

end incircle_radius_l751_751680


namespace min_max_abs_f_l751_751824

def f (x y : ℝ) : ℝ := x^3 - x * y

theorem min_max_abs_f :
  (∃ y : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → ∃ (m : ℝ), (∀ y', y' ∈ set.Ioo 0 y → |f x y'| ≤ m) ∧ m = 0) ∧ m = 0)
sorry

end min_max_abs_f_l751_751824


namespace quadratic_inequality_solution_l751_751436

-- Given a quadratic inequality, prove the solution set in interval notation.
theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x ^ 2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 :=
sorry

end quadratic_inequality_solution_l751_751436


namespace other_vehicle_is_quad_bike_l751_751676

/--
Timmy's parents have a garage containing various vehicles including:
- Timmy's bicycle with 2 wheels.
- Each of Timmy's parents' bicycles, with 2 wheels each (2 bicycles).
- Joey's tricycle with 3 wheels.
- Timmy's dad's unicycle with 1 wheel.
- Two cars, each typically having 4 wheels.
- The total wheel count in the garage is 22.

Prove that there is another vehicle in the garage (besides the listed ones)
with 4 wheels, such that total number of wheels sums to 22.
-/
theorem other_vehicle_is_quad_bike :
  let Timmy_bicycle := 2,
      Parents_bicycles := 2 * 2,
      Joey_tricycle := 3,
      Dad_unicycle := 1,
      Cars_total := 2 * 4,
      Known_wheels := Timmy_bicycle + Parents_bicycles + Joey_tricycle + Dad_unicycle + Cars_total,
      Total_wheels := 22
  in Total_wheels - Known_wheels = 4 :=
by
  -- Proof steps can be filled here 
  sorry

end other_vehicle_is_quad_bike_l751_751676


namespace eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five_l751_751901

theorem eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five:
  (0.85 * 40) - (4 / 5 * 25) = 14 :=
by
  sorry

end eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five_l751_751901


namespace total_fruits_consumed_l751_751989

def starting_cherries : ℝ := 16.5
def remaining_cherries : ℝ := 6.3

def starting_strawberries : ℝ := 10.7
def remaining_strawberries : ℝ := 8.4

def starting_blueberries : ℝ := 20.2
def remaining_blueberries : ℝ := 15.5

theorem total_fruits_consumed 
  (sc : ℝ := starting_cherries)
  (rc : ℝ := remaining_cherries)
  (ss : ℝ := starting_strawberries)
  (rs : ℝ := remaining_strawberries)
  (sb : ℝ := starting_blueberries)
  (rb : ℝ := remaining_blueberries) :
  (sc - rc) + (ss - rs) + (sb - rb) = 17.2 := by
  sorry

end total_fruits_consumed_l751_751989


namespace lawrence_walked_distance_l751_751130

variable (s : ℝ) (t : ℝ)
#align
theorem lawrence_walked_distance (h1 : s = 3) (h2 : t = 1 + 20 / 60) : s * t = 4 := by
  sorry

end lawrence_walked_distance_l751_751130


namespace part1_part2_period_part2_intervals_l751_751046

noncomputable def f (x : ℝ) : ℝ := cos x * (sin x + cos x) - 1/2

theorem part1 (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : sin α = sqrt 2 / 2) : f α = 1/2 :=
sorry

theorem part2_period : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
sorry

theorem part2_intervals : ∀ k : ℤ, is_increasing_on f (Icc (k * π - 3 * π / 8) (k * π + π / 8)) :=
sorry

end part1_part2_period_part2_intervals_l751_751046


namespace trapezoid_qr_length_l751_751766

theorem trapezoid_qr_length :
  ∀ (PQ RS altitude area QR : ℝ),
  area = 220 ∧ altitude = 10 ∧ PQ = 13 ∧ RS = 20 →
  QR = 9.43 :=
by {
  intros PQ RS altitude area QR h,
  have h0 : area = 220 := and.elim_left h,
  have h1 : altitude = 10 := and.elim_left (and.elim_right h),
  have h2 : PQ = 13 := and.elim_left (and.elim_right (and.elim_right h)),
  have h3 : RS = 20 := and.elim_right (and.elim_right (and.elim_right h)),
  sorry
}

end trapezoid_qr_length_l751_751766


namespace least_positive_integer_remainder_l751_751692

theorem least_positive_integer_remainder :
  ∃ n : ℕ, (n > 0) ∧ (n % 5 = 1) ∧ (n % 4 = 2) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 5 = 1) ∧ (m % 4 = 2) → n ≤ m) :=
sorry

end least_positive_integer_remainder_l751_751692


namespace empirical_regression_eq_mathematical_expectation_eq_l751_751709

variables {n : ℕ} (x y : Fin n → ℝ)

def x_samples : Fin 5 → ℝ := ![3, 4, 5, 6, 7]
def y_samples : Fin 5 → ℝ := ![20, 16, 15, 12, 6]

noncomputable def mean (z : Fin 5 → ℝ) : ℝ :=
  (Finset.univ.sum (λ i, z i)) / 5

noncomputable def slope (x y : Fin 5 → ℝ) : ℝ :=
  (Finset.univ.sum (λ i, (x i - mean x) * (y i - mean y))) /
  (Finset.univ.sum (λ i, (x i - mean x) ^ 2))

noncomputable def intercept (x y : Fin 5 → ℝ) : ℝ :=
  (mean y) - (slope x y) * (mean x)

noncomputable def regression (x : ℝ) : ℝ :=
  (intercept x_samples y_samples) + (slope x_samples y_samples) * x

theorem empirical_regression_eq : 
  ∀ x, regression x = -3.2 * x + 29.8 :=
sorry

noncomputable def residual (i : Fin 5) : ℝ :=
  y_samples i - regression (x_samples i)

noncomputable def sub_data_count (k : ℕ) : ℕ :=
  Finset.card (Finset.univ.filter (λ i, abs (residual i) > 1.2))

noncomputable def expectation (p : Fin 3 → ℝ) : ℝ :=
  Finset.univ.sum (λ i, p i * i)

theorem mathematical_expectation_eq :
  ∀ (samples : Finset (Fin 5)), samples.card = 3 →
  let p₀ := sub_data_count samples in
  let p := [p₀, p₀, p₀ + 1] in
  expectation p = 1.2 :=
sorry

end empirical_regression_eq_mathematical_expectation_eq_l751_751709


namespace right_triangle_median_length_l751_751103

theorem right_triangle_median_length
  (D E F : Type*)
  [metric_space D] [metric_space E] [metric_space F]
  (DE DF EF : ℝ)
  (hDE : DE = 15)
  (hDF : DF = 9)
  (hEF : EF = 12)
  (h_right : is_right_triangle D E F) :
  distance_to_midpoint F DE = 7.5 :=
by sorry

end right_triangle_median_length_l751_751103


namespace hyperbola_eccentricity_l751_751079

theorem hyperbola_eccentricity {a b : ℝ} (ha : a ≠ 0) (hb : b = 2 * a / 2) :
  (∀ x y : ℝ, ((x^2) / (a^2) - (y^2) / (b^2) = 1 ) ∧ (x^2 = 4 * y) → y = -1 ) →
  (1 / 2 * (b / a) * 2 = 2) →
  sqrt (a^2 + b^2) / a = sqrt 5 / 2 :=
  by
  sorry

end hyperbola_eccentricity_l751_751079


namespace sum_f_to_22_l751_751870

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom a1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom a2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom a3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom a4 : g(2) = 4

theorem sum_f_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_to_22_l751_751870


namespace sharon_prob_discard_card_l751_751758

theorem sharon_prob_discard_card {cards : Fin 49 → (Fin 7 × Fin 7)}
  (h_unique : ∀ i j, i ≠ j → cards i ≠ cards j)
  (h_colors : ∀ c : Fin 7, ∃ i, (cards i).1 = c)
  (h_numbers : ∀ n : Fin 7, ∃ i, (cards i).2 = n)
  (h_selection : ∀ (s : Finset (Fin 49)), s.card = 8 →
                 ∃ t : Finset (Fin 49), t ⊆ s ∧ t.card = 7 ∧
                 ∀ c : Fin 7, ∃ i ∈ t, (cards i).1 = c ∧
                 ∀ n : Fin 7, ∃ i ∈ t, (cards i).2 = n):
  let p := 4
  let q := 9
  Nat.gcd p q = 1 ∧ (p : ℚ) / q = 4 / 9 ∧ p + q = 13 :=
by
  sorry

end sharon_prob_discard_card_l751_751758


namespace chess_group_games_l751_751230

open Finset

theorem chess_group_games (n : ℕ) (h_n : n = 8) : (card (powersetLen 2 (range n))).val = 28 :=
by
  rw ← h_n
  simp only [card_powersetLen, range_val, finset.card]
  sorry

end chess_group_games_l751_751230


namespace remainder_of_x50_div_by_x_sub_1_cubed_l751_751829

theorem remainder_of_x50_div_by_x_sub_1_cubed :
  (x^50 % (x-1)^3) = (1225*x^2 - 2500*x + 1276) :=
sorry

end remainder_of_x50_div_by_x_sub_1_cubed_l751_751829


namespace fraction_value_l751_751425

variable (x y : ℚ)

theorem fraction_value (h₁ : x = 4 / 6) (h₂ : y = 8 / 12) : 
  (6 * x + 8 * y) / (48 * x * y) = 7 / 16 :=
by
  sorry

end fraction_value_l751_751425


namespace James_age_after_x_years_l751_751129

variable (x : ℕ)
variable (Justin Jessica James : ℕ)

-- Define the conditions
theorem James_age_after_x_years 
  (H1 : Justin = 26) 
  (H2 : Jessica = Justin + 6) 
  (H3 : James = Jessica + 7)
  (H4 : James + 5 = 44) : 
  James + x = 39 + x := 
by 
  -- proof steps go here 
  sorry

end James_age_after_x_years_l751_751129


namespace sum_of_numbers_l751_751506

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l751_751506


namespace attendance_rate_comparison_l751_751730

theorem attendance_rate_comparison (attendees_A total_A attendees_B total_B : ℕ) 
  (hA : (attendees_A / total_A: ℚ) > (attendees_B / total_B: ℚ)) : 
  (attendees_A > attendees_B) → false :=
by
  sorry

end attendance_rate_comparison_l751_751730


namespace cost_price_of_article_l751_751357

-- Define the conditions
variable (C : ℝ) -- Cost price of the article
variable (SP : ℝ) -- Selling price of the article

-- Conditions according to the problem
def condition1 : Prop := SP = 0.75 * C
def condition2 : Prop := SP + 500 = 1.15 * C

-- The theorem to prove the cost price
theorem cost_price_of_article (h₁ : condition1 C SP) (h₂ : condition2 C SP) : C = 1250 :=
by
  sorry

end cost_price_of_article_l751_751357


namespace free_fall_time_l751_751657

theorem free_fall_time (h : ℝ) (t : ℝ) (h_eq : h = 4.9 * t^2) (h_val : h = 490) : t = 10 :=
by
  sorry

end free_fall_time_l751_751657


namespace hyperbola_eccentricity_is_two_l751_751035

noncomputable def hyperbola_eccentricity (a b : ℝ) (c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity_is_two :
  ∃ (a b c : ℝ), one_of_foci (x y : ℝ) := (x^2 / a^2 - y^2 / b^2 = 1)  (2, 0)  ∧
  is_tangent (x y : ℝ) := (x-2)^2 + y^2 = 1 ∧
  a = 1 ∧ c = 2 ∧
  hyperbola_eccentricity a b c = 2 := 
sorry

end hyperbola_eccentricity_is_two_l751_751035


namespace noah_total_watts_used_l751_751983

theorem noah_total_watts_used :
  let bedroom_watts_per_hour := 6
  let office_watts_per_hour := 3 * bedroom_watts_per_hour
  let living_room_watts_per_hour := 4 * bedroom_watts_per_hour
  let hours_on := 2
  let bedroom_total := bedroom_watts_per_hour * hours_on
  let office_total := office_watts_per_hour * hours_on
  let living_room_total := living_room_watts_per_hour * hours_on
  bedroom_total + office_total + living_room_total = 96 :=
by
  -- Define the given conditions as variables
  let bedroom_watts_per_hour := 6
  let office_watts_per_hour := 3 * bedroom_watts_per_hour
  let living_room_watts_per_hour := 4 * bedroom_watts_per_hour
  let hours_on := 2
  
  -- Calculate watts used over two hours
  let bedroom_total := bedroom_watts_per_hour * hours_on
  let office_total := office_watts_per_hour * hours_on
  let living_room_total := living_room_watts_per_hour * hours_on
  
  -- Sum up the totals
  have h1 : bedroom_total = 12 := rfl
  have h2 : office_total = 36 := rfl
  have h3 : living_room_total = 48 := rfl
  have sum_totals : 12 + 36 + 48 = 96 := by norm_num

  -- Conclusion
  show bedroom_total + office_total + living_room_total = 96 from sum_totals

end noah_total_watts_used_l751_751983


namespace inequality_holds_l751_751304

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l751_751304


namespace sin_450_eq_1_l751_751389

theorem sin_450_eq_1 : Real.sin (450 * Real.pi / 180) = 1 := by
  -- Using the fact that angle measures can be reduced modulo 2π radians (360 degrees)
  let angle := 450 * Real.pi / 180
  let reduced_angle := angle % (2 * Real.pi)
  have h1 : reduced_angle = Real.pi / 2 := by
    -- 450 degrees is 450 * π / 180 radians, which simplifies to π / 2 radians mod 2π
    calc
      reduced_angle = (450 * Real.pi / 180) % (2 * Real.pi)  : rfl
      ... = (5 * Real.pi / 2) % (2 * Real.pi)                : by simp [mul_div_assoc, Real.pi_div_two]
      ... = Real.pi / 2                                       : by norm_num1
  -- The sine of π / 2 radians is 1
  rw [h1, Real.sin_pi_div_two]
  exact rfl

end sin_450_eq_1_l751_751389


namespace find_stamps_l751_751273

def stamps_problem (x y : ℕ) : Prop :=
  (x + y = 70) ∧ (y = 4 * x + 5)

theorem find_stamps (x y : ℕ) (h : stamps_problem x y) : 
  x = 13 ∧ y = 57 :=
sorry

end find_stamps_l751_751273


namespace find_speed_second_hour_l751_751660

-- Definitions of the conditions
variables (speed_first_hour speed_avg total_time : ℝ)
variable (speed_second_hour : ℝ)

-- Given conditions
def conditions := speed_first_hour = 60 ∧ speed_avg = 45 ∧ total_time = 2

-- The theorem to prove
theorem find_speed_second_hour (h : conditions) : speed_second_hour = 30 :=
by {
  have h1 : speed_first_hour = 60 := h.1,
  have h2 : speed_avg = 45 := h.2.1,
  have h3 : total_time = 2 := h.2.2,
  sorry
}

end find_speed_second_hour_l751_751660


namespace combined_jail_time_in_weeks_l751_751243

-- Definitions based on conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def daily_arrests_per_city : ℕ := 10
def days_in_jail_pre_trial : ℕ := 4
def sentence_weeks : ℕ := 2
def jail_fraction_of_sentence : ℕ := 1 / 2

-- Calculate the combined weeks of jail time
theorem combined_jail_time_in_weeks : 
  (days_of_protest * daily_arrests_per_city * number_of_cities) * 
  (days_in_jail_pre_trial + (sentence_weeks * 7 * jail_fraction_of_sentence)) / 
  7 = 9900 := 
by sorry

end combined_jail_time_in_weeks_l751_751243


namespace sequence_formula_l751_751546

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n / (1 + 2 * a n)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_formula_l751_751546


namespace factorization_l751_751874

open Polynomial

noncomputable def expr (a b c : ℤ) : ℤ := a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3)

noncomputable def p (a b c : ℤ) : ℤ := a * b^3 + a * c^3 + a * b * c^2 + b^2 * c^2

theorem factorization (a b c : ℤ) : 
  expr a b c = (a - b) * (b - c) * (c - a) * p a b c :=
by sorry

end factorization_l751_751874


namespace nine_sequence_sum_eq_target_l751_751183

-- Definitions based on conditions
def sequence := "9999999"
def target := 1989

-- The Lean statement to prove the mathematically equivalent problem
theorem nine_sequence_sum_eq_target : 
  ∃ a b c : ℕ, a + b - c = target ∧ a = 999 ∧ b = 999 ∧ c = 9 :=
by
  use 999
  use 999
  use 9
  have h1 : 999 + 999 = 1998 := rfl
  have h2 : 1998 - 9 = 1989 := rfl
  exact ⟨h2, rfl, rfl, rfl⟩

end nine_sequence_sum_eq_target_l751_751183


namespace inequality_solution_l751_751193

theorem inequality_solution :
  { x : ℝ // x < 2 ∨ (3 < x ∧ x < 6) ∨ (7 < x ∧ x < 8) } →
  ((x - 3) * (x - 5) * (x - 7)) / ((x - 2) * (x - 6) * (x - 8)) > 0 :=
by
  sorry

end inequality_solution_l751_751193


namespace quadratic_prime_roots_no_value_l751_751778

theorem quadratic_prime_roots_no_value (k : ℕ) :
  (∃ p q : ℕ, p.prime ∧ q.prime ∧ p + q = 29 ∧ p * q = k) → false :=
by
  sorry

end quadratic_prime_roots_no_value_l751_751778


namespace angel_vowels_written_l751_751773

theorem angel_vowels_written (num_vowels : ℕ) (times_written : ℕ) (h1 : num_vowels = 5) (h2 : times_written = 4) : num_vowels * times_written = 20 := by
  sorry

end angel_vowels_written_l751_751773


namespace problem_l751_751999

-- Define the values in the grid
def grid : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ := (4, 3, 1, 1, 6, 2, 3)

-- Define the variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def condition_1 := (A = 3) ∧ (B = 2) ∧ (C = 4)
def condition_2 := (4 + A + 1 + B + C + 3 = 9)
def condition_3 := (A + 1 + 6 = 9)
def condition_4 := (1 + A + 6 = 9)
def condition_5 := (B + 2 + C + 5 = 9)

-- Define that the sum of the red cells is equal to any row
def sum_of_red_cells := (A + B + C = 9)

-- The final goal to prove
theorem problem : condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5 ∧ sum_of_red_cells := 
by {
  refine ⟨_, _, _, _, _, _⟩;
  sorry   -- proofs for each condition
}

end problem_l751_751999


namespace vector_triple_product_result_l751_751157

def p : ℝ × ℝ × ℝ := (-1, 4, 2)
def q : ℝ × ℝ × ℝ := (3, 5, -1)
def r : ℝ × ℝ × ℝ := (0, 3, 9)

def vector_sub (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2, a.3 - b.3)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

theorem vector_triple_product_result :
  dot_product (vector_sub p q) (cross_product (vector_sub q r) (vector_sub r p)) = 112 :=
by
  sorry

end vector_triple_product_result_l751_751157


namespace new_total_lines_l751_751990

-- Definitions and conditions
variable (L : ℕ)
def increased_lines : ℕ := L + 60
def percentage_increase := (60 : ℚ) / L = 1 / 3

-- Theorem statement
theorem new_total_lines : percentage_increase L → increased_lines L = 240 :=
by
  sorry

end new_total_lines_l751_751990


namespace find_max_value_l751_751823

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan (x + (2 * Real.pi / 3)) - Real.tan (x + (Real.pi / 6)) + Real.cos (x + (Real.pi / 6))

theorem find_max_value :
  (max (f x) (-5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 3)) = 11 * Real.sqrt 3 / 6 := 
sorry

end find_max_value_l751_751823


namespace largest_integer_consecutive_sum_l751_751822

theorem largest_integer_consecutive_sum :
  ∃ A, (∀ (σ : Fin 100 → ℕ), (∃ n : Fin 91, 10.consecutive_sum n σ ≥ A)) ∧ A = 505 :=
sorry

end largest_integer_consecutive_sum_l751_751822


namespace possible_to_place_12_numbers_on_cube_edges_l751_751556

-- Define a list of numbers from 1 to 12
def nums := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the faces of the cube in terms of the indices of nums list
def top_face := [1, 2, 9, 10]
def bottom_face := [3, 5, 6, 8]

-- Define the product of the numbers on the faces of the cube
def product_face (face : List Nat) : Nat := face.foldr (*) 1

-- The lean statement proving the problem
theorem possible_to_place_12_numbers_on_cube_edges :
  product_face (top_face.map (λ i => nums.get! (i - 1))) =
  product_face (bottom_face.map (λ i => nums.get! (i - 1))) :=
by
  sorry

end possible_to_place_12_numbers_on_cube_edges_l751_751556


namespace lines_parallel_non_intersecting_l751_751609

theorem lines_parallel_non_intersecting (u v x : ℝ) : 
  (∀ u v : ℝ, 
    (∃ a b c d : ℝ, a + u * 6 = c + v * x ∧ b + u * (-2) = d + v * (-3) 
    → (1, 3) + u • (6, -2) ≠ (-4, 5) + v • (x, -3))) ↔ x = 9 :=
by sorry

end lines_parallel_non_intersecting_l751_751609


namespace cricket_team_new_win_percentage_l751_751523

theorem cricket_team_new_win_percentage :
  ∀ (initial_matches additional_matches initial_win_percent : ℕ) (initial_win_percent_prop : initial_win_percent = 30)
  (initial_matches_prop : initial_matches = 120)
  (additional_matches_prop : additional_matches = 55),
  let initial_wins := (initial_win_percent * initial_matches) / 100
  let total_wins := initial_wins + additional_matches
  let total_matches := initial_matches + additional_matches
  let new_win_percentage := (total_wins * 100) / total_matches
  new_win_percentage = 52 :=
by
  intros initial_matches additional_matches initial_win_percent initial_win_percent_prop initial_matches_prop additional_matches_prop
  suffices : initial_matches = 120 ∧ additional_matches = 55 ∧ initial_win_percent = 30 by {
    cases this with initial_matches_eq initial_matches_rest
    cases initial_matches_rest with additional_matches_eq initial_win_percent_eq
    rw [initial_matches_eq, additional_matches_eq, initial_win_percent_eq]
    -- Definitions based on provided conditions
    let initial_wins := (30 * 120) / 100
    let total_wins := initial_wins + 55
    let total_matches := 120 + 55
    let new_win_percentage := (total_wins * 100) / total_matches
    have h_initial_wins : initial_wins = 36 := by norm_num
    rw h_initial_wins at total_wins
    let new_win_percentage := (91 * 100) / 175
    have : new_win_percentage = 52 := by norm_num
    assumption
  },
  tauto

end cricket_team_new_win_percentage_l751_751523


namespace proof_math_problem_l751_751966

-- Given conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def periodic_property (f : ℝ → ℝ) := ∀ x, f (x + 2) = -f x
def specific_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = 2 * x - x^2

-- Theorem to prove all mentioned properties and results
theorem proof_math_problem (f : ℝ → ℝ)
  (h1 : odd_function f)
  (h2 : periodic_property f)
  (h3 : specific_interval f) :
  -- 1. Prove periodicity
  (∀ x, f (x + 4) = f x) ∧
  -- 2. Find the expression for x in [2, 4]
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f x = x^2 - 6 * x + 8) ∧
  -- 3. Calculate the sum f(0) + f(1) + ... + f(2014)
  (f 0 + f 1 + f 2 + ⋯ + f 2014 = 1) :=
by {
  sorry -- Proof is omitted
}

end proof_math_problem_l751_751966


namespace cube_edge_numbers_equal_top_bottom_l751_751564

theorem cube_edge_numbers_equal_top_bottom (
  numbers : List ℕ,
  h : numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) :
  ∃ (top bottom : List ℕ),
    (∀ x, x ∈ top → x ∈ numbers) ∧
    (∀ x, x ∈ bottom → x ∈ numbers) ∧
    (top ≠ bottom) ∧
    (top.length = 4) ∧ 
    (bottom.length = 4) ∧ 
    (top.product = bottom.product) :=
begin
  sorry
end

end cube_edge_numbers_equal_top_bottom_l751_751564


namespace hyperbola_asymptotes_l751_751211

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 4 - y^2 = 1) → (y = x / 2 ∨ y = -x / 2) :=
sorry

end hyperbola_asymptotes_l751_751211


namespace impossible_broken_line_through_centers_l751_751750

/-- 
A rectangle measuring 7 × 9 is divided into 1 × 1 squares. 
The central square is shaded. 
Prove that it is impossible to draw a broken line 
through the centers of all the unshaded squares such that 
each segment connects the centers of side-adjacent squares without passing through any square more than once.
-/
theorem impossible_broken_line_through_centers : 
  ∀ (rect : array (fin 7) (array (fin 9) bool)), 
    (rect 3 4 = true) → -- the central square is shaded
    ¬ ∃ (f : fin 63 → fin 63), -- mapping representing the "broken line" 
      (bijective f) ∧ -- to visit all squares
      (∀ i, abs ((f i).1 - f (succ i).1) + abs ((f i).2 - f (succ i).2) = 1) ∧ -- adjacency condition
      (∀ i, rect (f i).1 (f i).2 = false) -- unshaded condition 
      sorry

end impossible_broken_line_through_centers_l751_751750


namespace coralee_trip_cost_l751_751792

-- Given definitions based on conditions.
def distance_DF := 2800
def distance_EF := 2900
def cost_bus_per_km := 0.20
def cost_plane_per_km := 0.12
def cost_plane_booking := 120

-- Pythagorean Theorem for calculating distance DE.
def distance_DE := Real.sqrt (distance_DF ^ 2 + distance_EF ^ 2)

-- Calculate cost for each leg and identify the cheapest.
def cost_DE_plane := cost_plane_booking + cost_plane_per_km * distance_DE
def cost_DE_bus := cost_bus_per_km * distance_DE
def cost_DE := min cost_DE_plane cost_DE_bus

def cost_EF_plane := cost_plane_booking + cost_plane_per_km * distance_EF
def cost_EF_bus := cost_bus_per_km * distance_EF
def cost_EF := min cost_EF_plane cost_EF_bus

def cost_DF_plane := cost_plane_booking + cost_plane_per_km * distance_DF
def cost_DF_bus := cost_bus_per_km * distance_DF
def cost_DF := min cost_DF_plane cost_DF_bus

def total_cost := cost_DE + cost_EF + cost_DF

-- The theorem to prove that the total cost is $1075.
theorem coralee_trip_cost : total_cost = 1075 := by
  sorry

end coralee_trip_cost_l751_751792


namespace sum_of_roots_eq_three_l751_751704

theorem sum_of_roots_eq_three (x : ℝ) :
  (∃ x : ℝ, x^2 - 3 * x + 2 = 12) →
  ((roots (X^2 - 3 * X - 10)).sum = 3) :=
by
  intros h,
  sorry

end sum_of_roots_eq_three_l751_751704


namespace electricity_bill_written_as_decimal_l751_751712

-- Definitions as conditions
def number : ℝ := 71.08

-- Proof statement
theorem electricity_bill_written_as_decimal : number = 71.08 :=
by sorry

end electricity_bill_written_as_decimal_l751_751712


namespace harriets_siblings_product_l751_751064

theorem harriets_siblings_product (HarrySisters HarryBrothers : ℕ)
  (h₁ : HarrySisters = 4)
  (h₂ : HarryBrothers = 3)
  (S : ℕ := HarrySisters - 1)
  (B : ℕ := HarryBrothers) :
  S * B = 9 := 
by
  simp [h₁, h₂]
  sorry

end harriets_siblings_product_l751_751064


namespace optimal_colonel_blotto_distribution_l751_751112

theorem optimal_colonel_blotto_distribution
  (x : Fin 10 → ℕ)
  (h_sum : (∑ i, x i) = 100) :
  ∃ (y : Fin 10 → ℕ), 
  (∑ i, y i) = 100 ∧ 
  (y = fun i => if i % 2 = 0 then 17 else 3) ∨
  (y = fun i => if i < 7 then 12 else (if i = 7 then 9 else if i = 8 then 6 else 1))
  :=
sorry

end optimal_colonel_blotto_distribution_l751_751112


namespace angle_equality_l751_751725

variable (A B C D M N K L: Point)

-- Conditions
variable (squareABCD: Square A B C D)
variable (midpointM: M = midpoint B C)
variable (midpointN: N = midpoint A D)
variable (pointK: ∃ E, K = extension A C E)
variable (intersectL: ∃ F, KM ∩ AB = L)

-- Proof goal
theorem angle_equality :
  ∠ K N A = ∠ L N A :=
sorry

end angle_equality_l751_751725


namespace find_a_l751_751526

-- Define the conditions in Lean
variable (a : ℕ) -- a is a natural number representing the total number of balls
variable (red_balls : ℕ) -- number of red balls
variable (freq_red : ℝ) -- frequency of picking a red ball

-- Set the specific values for the problem
def red_balls := 5
def freq_red := 0.20

-- The Lean theorem statement to prove the problem's condition
theorem find_a (h : freq_red = red_balls / a) : a = 25 :=
by
  -- Skipping the proof as per the instructions
  sorry

end find_a_l751_751526


namespace inequality_ge_one_l751_751314

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l751_751314


namespace needle_can_pierce_cube_l751_751738

def isNeedlePathPossible (cube_side : ℕ) (brick_count : ℕ) (brick_dim : ℕ × ℕ × ℕ) : Prop :=
  ∃ path : (ℕ × ℕ) × (ℕ × ℕ), (path.1 ≠ path.2) ∧ 
    (∀ (p : ℕ × ℕ), (p = path.1 ∨ p = path.2) → ¬ (∃ b : ℕ, (b < brick_count ∧ -- Check if there is a block in the path
    ∀ i j k, (i, j, k) = brick_dim))

theorem needle_can_pierce_cube : isNeedlePathPossible 20 2000 (2, 2, 1) :=
  sorry

end needle_can_pierce_cube_l751_751738


namespace at_least_three_pos_and_neg_l751_751444

theorem at_least_three_pos_and_neg (a : ℕ → ℝ) (h : ∀ n : ℕ, 2 ≤ n ∧ n ≤ 11 → a (n-1) * (a n - a (n-1) + a (n + 1)) < 0) :
  (∃ i j k l m n : ℕ, (i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ n) ∧ (2 ≤ i ∧ i ≤ 4) ∧ (2 ≤ j ∧ j ≤ 4) ∧ (2 ≤ k ∧ k ≤ 4) ∧ (2 ≤ l ∧ l ≤ 4) ∧ (2 ≤ m ∧ m ≤ 4) ∧ (2 ≤ n ∧ n ≤ 4)) ∧
  (∃ p q r s t u : ℕ, (p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ t ∧ t ≠ u) ∧ (2 ≤ p ∧ p ≤ 4) ∧ (2 ≤ q ∧ q ≤ 4) ∧ (2 ≤ r ∧ r ≤ 4) ∧ (2 ≤ s ∧ s ≤ 4) ∧ (2 ≤ t ∧ t ≤ 4) ∧ (2 ≤ u ∧ u ≤ 4)) :=
begin
  sorry
end

end at_least_three_pos_and_neg_l751_751444


namespace count_even_integers_form_3k_plus_4_l751_751497

theorem count_even_integers_form_3k_plus_4 :
  { n : ℕ | 20 ≤ n ∧ n ≤ 250 ∧ ∃ k : ℕ, n = 3 * k + 4 ∧ (n % 2 = 0) }.finite.to_finset.card = 39 :=
by
  sorry

end count_even_integers_form_3k_plus_4_l751_751497


namespace remaining_food_after_trip_calculate_remaining_food_l751_751757

theorem remaining_food_after_trip (initial_food : ℕ)
    (first_day_rate : ℚ)
    (second_day_rate : ℚ)
    (first_day_usage : initial_food * first_day_rate.to_nat)
    (remaining_after_first_day : initial_food - first_day_usage)
    (second_day_usage : remaining_after_first_day * second_day_rate.to_nat)
    (final_remaining_food : remaining_after_first_day - second_day_usage)
    : final_remaining_food = 96 := by
  sorry

# Conditions specific to this problem
def initial_food := 400
def first_day_rate := 2 / 5
def second_day_rate := 3 / 5

-- Calculations
def first_day_usage := initial_food * 2 / 5
def remaining_after_first_day := 400 - first_day_usage
def second_day_usage := remaining_after_first_day * 3 / 5
def final_remaining_food := remaining_after_first_day - second_day_usage

theorem calculate_remaining_food : final_remaining_food = 96 := by
  sorry

(theorem : calculate_remaining_food : final_remaining_food = 96 := by
  calc
    final_remaining_food 
      = (240 - 144) : sorry
      ... = 96 : rfl)

end remaining_food_after_trip_calculate_remaining_food_l751_751757


namespace count_true_propositions_l751_751217

-- Definitions corresponding to the given conditions

def cond1_converse : Prop :=
  ∀ (x y : ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 = 0)

def cond2_negation : Prop :=
  ∀ (ΔΔ : Type) (t1 t2 : ΔΔ), (congruent t1 t2 → ¬similar t1 t2)

def cond3_contrapositive : Prop :=
  ∀ (m : ℝ), (¬∃ (x : ℝ), x^2 + x - m = 0) → m ≤ 0

def cond4_contrapositive : Prop :=
  ∀ (a b c : ℝ), (c^2 ≠ a^2 + b^2) → (¬ (∃ (C : ℝ), C = 90 ∧ C = ∠ ABC))

-- Main theorem stating the problem
theorem count_true_propositions : (∑ b in [cond1_converse, cond3_contrapositive, cond4_contrapositive], if b then 1 else 0) = 3 := by
  sorry

end count_true_propositions_l751_751217


namespace ratio_b4_b3_a2_a1_l751_751156

variables {x y d d' : ℝ}
variables {a1 a2 a3 b1 b2 b3 b4 : ℝ}
-- Conditions
variables (h1 : x ≠ y)
variables (h2 : a1 = x + d)
variables (h3 : a2 = x + 2 * d)
variables (h4 : a3 = x + 3 * d)
variables (h5 : y = x + 4 * d)
variables (h6 : b2 = x + d')
variables (h7 : b3 = x + 2 * d')
variables (h8 : y = x + 3 * d')
variables (h9 : b4 = x + 4 * d')

theorem ratio_b4_b3_a2_a1 :
  (b4 - b3) / (a2 - a1) = 8 / 3 :=
by sorry

end ratio_b4_b3_a2_a1_l751_751156


namespace least_possible_value_of_element_in_T_l751_751137

theorem least_possible_value_of_element_in_T :
  ∃ T : Finset ℕ, T.card = 7 ∧ (∀ a b ∈ T, a < b → ¬(b % a = 0)) ∧ 
    T.min' (by sorry { assume h, sorry }) = 4 :=
sorry

end least_possible_value_of_element_in_T_l751_751137


namespace gloria_turtle_time_l751_751894

theorem gloria_turtle_time (g_time : ℕ) (george_time : ℕ) (gloria_time : ℕ) 
  (h1 : g_time = 6) 
  (h2 : george_time = g_time - 2)
  (h3 : gloria_time = 2 * george_time) : 
  gloria_time = 8 :=
sorry

end gloria_turtle_time_l751_751894


namespace find_m_if_f_odd_l751_751037

def f (x : ℝ) (m : ℝ) : ℝ := Real.logBase 2 (2 * x / (x - 1) + m)

theorem find_m_if_f_odd : 
  (∀ x : ℝ, f (-x) m = -f x m) ↔ m = -1 := 
sorry

end find_m_if_f_odd_l751_751037


namespace find_unique_n_k_l751_751816

theorem find_unique_n_k (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) :
    (n+1)^n = 2 * n^k + 3 * n + 1 ↔ (n = 3 ∧ k = 3) := by
  sorry

end find_unique_n_k_l751_751816


namespace max_value_of_a_in_domain_l751_751434

def f (a : ℝ) (x : ℝ) := sqrt (sqrt 2 * a * (sin (Real.pi * x) + cos (Real.pi * x)))

-- The proof statement for the maximum value of a
theorem max_value_of_a_in_domain : ∃ a : ℝ, (∀ x : ℝ, (0 ≤ sqrt (sqrt 2 * a * (sin (Real.pi * x) + cos (Real.pi * x))))) ∧ (a = 9 / 32) :=
sorry

end max_value_of_a_in_domain_l751_751434


namespace inequality_ge_one_l751_751310

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l751_751310


namespace jail_time_weeks_l751_751241

theorem jail_time_weeks (days_protest : ℕ) (cities : ℕ) (arrests_per_day : ℕ)
  (days_pre_trial : ℕ) (half_week_sentence_days : ℕ) :
  days_protest = 30 →
  cities = 21 →
  arrests_per_day = 10 →
  days_pre_trial = 4 →
  half_week_sentence_days = 7 →
  (21 * 30 * 10 * (4 + 7)) / 7 = 9900 :=
by
  intros h_days_protest h_cities h_arrests_per_day h_days_pre_trial h_half_week_sentence_days
  rw [h_days_protest, h_cities, h_arrests_per_day, h_days_pre_trial, h_half_week_sentence_days]
  exact sorry

end jail_time_weeks_l751_751241


namespace incircle_radius_is_correct_l751_751684

-- Define the given conditions
structure Triangle :=
  (D E F : ℝ)
  (angleD : ℝ)
  (DF : ℝ)

def right_angle_at_F (T : Triangle) : Prop := T.angleD = 45 ∧ T.DF = 8

-- Define the radius of the incircle function based on conditions
noncomputable def radius_of_incircle (T : Triangle) : ℝ :=
  let DF := T.DF in
  let DE := DF in
  let EF := DF * Real.sqrt 2 in
  let area := (DF * DE) / 2 in
  let s := (DF + DE + EF) / 2 in
  let r := area / s in
  r

-- The proof goal
theorem incircle_radius_is_correct (T : Triangle) (h : right_angle_at_F T) :
  radius_of_incircle T = 4 - 2 * Real.sqrt 2 := by
  sorry

end incircle_radius_is_correct_l751_751684


namespace most_likely_units_digit_of_sum_is_zero_l751_751805

theorem most_likely_units_digit_of_sum_is_zero : 
  ∃ (J K : Fin 13), (J ∈ Finset.range 1 13) ∧ (K ∈ Finset.range 1 13) ∧
  (∃ n, n.units_digit = 0 ∧ 
       (∑ i in Finset.range 1 13, ∑ j in Finset.range 1 13, if (i + j) % 10 = n then 1 else 0) ≥ 
       (∑ i in Finset.range 1 13, ∑ j in Finset.range 1 13, if (i + j) % 10 = m then 1 else 0) ∀ m ≠ n)
:= 
sorry

end most_likely_units_digit_of_sum_is_zero_l751_751805


namespace incenter_circumcenter_k_collinear_l751_751232

/-- Given three congruent circles inside triangle ABC with a common point K,
    and each circle tangent to two sides of triangle ABC,
    the incenter I, circumcenter O of triangle ABC, and K are collinear. -/
theorem incenter_circumcenter_k_collinear
  (A B C K : Point)
  (circumcenter : Triangle → Point)
  (incenter : Triangle → Point)
  (circumcenter_ABC := circumcenter (Triangle.mk A B C))
  (incenter_ABC := incenter (Triangle.mk A B C))
  (congruent_circles : ∀ O1 O2 O3 : Point, 
                       (circle_tangent (circle.mk O1 K) A B) → 
                       (circle_tangent (circle.mk O2 K) B C) → 
                       (circle_tangent (circle.mk O3 K) C A))
  : collinear circumcenter_ABC incenter_ABC K :=
sorry

end incenter_circumcenter_k_collinear_l751_751232


namespace percentage_of_hundred_l751_751252

theorem percentage_of_hundred : (30 / 100) * 100 = 30 := 
by
  sorry

end percentage_of_hundred_l751_751252


namespace inequality_ge_one_l751_751311

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l751_751311


namespace mixed_number_division_l751_751374

theorem mixed_number_division :
  (5 + 1 / 2 - (2 + 2 / 3)) / (1 + 1 / 5 + 3 + 1 / 4) = 0 + 170 / 267 := 
by
  sorry

end mixed_number_division_l751_751374


namespace students_difference_l751_751533

theorem students_difference 
  (C : ℕ → ℕ) 
  (hC1 : C 1 = 24) 
  (hC2 : ∀ n, C n.succ = C n - d)
  (h_total : C 1 + C 2 + C 3 + C 4 + C 5 = 100) :
  d = 2 :=
by sorry

end students_difference_l751_751533


namespace recurring_decimal_addition_l751_751406

-- Conditions
def recurring_decimal (d : ℝ) (b : ℤ) : Prop := 
  ∃ x : ℝ, x = b / (10 ^ integer_length b - 1) ∧ x = d

-- Define the recurring decimal 0.7overline{23}
def recurring_023 := recurring_decimal 0.023 (23 : ℤ)

-- Define the problem statement in Lean
theorem recurring_decimal_addition :
  recurring_023 ∧ recurring_decimal 0.7 7 → 
  0.7 + 0.023 + (1 / 3) = 417 / 330 :=
by {
  sorry
}

end recurring_decimal_addition_l751_751406


namespace minimize_distance_l751_751029

-- Defining the problem with conditions
variables {A B C D X : ℝ}
variables (a b c d x : ℝ)
hypothesis h_order: a < b ∧ b < c ∧ c < d

-- Formulating the function
def sum_distances (a b c d x : ℝ) : ℝ :=
  abs (x - a) + abs (x - b) + abs (x - c) + abs (x - d)

-- Statement of the problem
theorem minimize_distance :
  ∃ x, (b ≤ x ∧ x ≤ c) ∧ ∀ y, sum_distances a b c d y ≥ sum_distances a b c d x :=
by
  sorry

end minimize_distance_l751_751029


namespace factorization_l751_751407

theorem factorization (x y : ℝ) : 6 * x^2 * y - 3 * x * y = 3 * x * y * (2 * x - 1) :=
by
  sorry

end factorization_l751_751407


namespace no_values_for_g_g_x_eq_9_l751_751794

def g (x : ℝ) : ℝ :=
  if x ≥ -2 then x^2 - 1 else x + 5

theorem no_values_for_g_g_x_eq_9 : set.card {x : ℝ | g (g x) = 9} = 0 :=
  by
  sorry

end no_values_for_g_g_x_eq_9_l751_751794


namespace students_not_playing_games_l751_751228

theorem students_not_playing_games 
  (total_students : ℕ)
  (basketball_players : ℕ)
  (volleyball_players : ℕ)
  (both_players : ℕ)
  (h1 : total_students = 20)
  (h2 : basketball_players = (1 / 2) * total_students)
  (h3 : volleyball_players = (2 / 5) * total_students)
  (h4 : both_players = (1 / 10) * total_students) :
  total_students - ((basketball_players + volleyball_players) - both_players) = 4 :=
by
  sorry

end students_not_playing_games_l751_751228


namespace menelaus_theorem_l751_751177

-- Define points in the Lean 4 coordinate space
variables (A B C A1 B1 C1 : Type)
variables [metric_space A1] [metric_space B1] [metric_space C1]

-- Define Menelaus' theorem condition as ratios for collinearity
theorem menelaus_theorem 
  (hB1_AC : B1 ∈ segment ℝ A C)
  (hC1_AB : C1 ∈ segment ℝ A B)
  (hA1_CB_ext : A1 ∉ segment ℝ B C ∧ A1 ∈ line_through B C) :
  (∃ k l m, 
    (AB1 / B1C) = k ∧ 
    (BC1 / C1A) = l ∧ 
    (CA1 / A1B) = m ∧ 
    k * l * m = 1 ↔ 
    collinear ℝ ({A1, B1, C1} : set Type)) :=
sorry

end menelaus_theorem_l751_751177


namespace part_1_part_2_l751_751927

-- Convert polar line equation to rectangular coordinate system
theorem part_1 (x y : ℝ) : 
  (l : x * sqrt 3 - y - sqrt 3 = 0) ↔ (ρ θ : ℝ) (h₀ : ρ = x / cos θ) (h₁ : θ = y / sin θ) (h₂ : ρ * sin (θ - π / 3) = -sqrt 3 / 2) :=
by sorry

-- Polar coordinate equation of the circle
theorem part_2 (ρ θ : ℝ) (P : ℝ × ℝ) (h₀ : P = (sqrt 2, π / 4))
  (h₁ : center_x = 1) (h₂ : center_y = 0) (h₃ : ∀ (P1 P2 : ℝ × ℝ), P1 = P → P2 = (center_x, center_y) → dist P1 P2 = 1)
  : (C : ρ = 2 * cos θ) :=
by sorry

end part_1_part_2_l751_751927


namespace cardinality_of_M_ge_l751_751131

open Finset

variables {A B : Type}

noncomputable def is_partition (p : ℕ) (U : Finset A) (P : Finset (Finset A)) : Prop :=
  P.card = p ∧ (∀ s ∈ P, ∀ t ∈ P, s ≠ t → Disjoint s t) ∧ (P.bUnion id) = U

theorem cardinality_of_M_ge (p : ℕ) 
  (M : Finset A) (𝓐 𝓑 : Finset (Finset A))
  (hA : is_partition p M 𝓐)
  (hB : is_partition p M 𝓑)
  (cond : ∀ A_i ∈ 𝓐, ∀ B_j ∈ 𝓑, (Disjoint A_i B_j) → (A_i.card + B_j.card > p)) :
  M.card ≥ (1 + p^2) / 2 :=
sorry

end cardinality_of_M_ge_l751_751131


namespace sum_distances_l751_751950

open Real Finset

variable (n : ℕ) (h : n > 2)
variable (P : Finₙ ℕ → ℝ × ℝ) (l : ℝ × ℝ → Prop) (Q : ℝ × ℝ)
variable (d : Finₙ ℕ → ℝ) (c : Finₙ ℕ → ℝ)

theorem sum_distances :
  (∀ i, ∃ j, d i = ∏ j in {0 .. n-1} \ {i}, dist (P i) (P j)) →
  (∀ i, c i = dist Q (P i)) →
  (∀ i, l (P i)) →
  (¬ l Q) →
  if n = 3 then ∑ i in {0 .. n-1}, (-1)^(n-i) * (c i)^2 / (d i) = 1
  else if n ≥ 4 then ∑ i in {0 .. n-1}, (-1)^(n-i) * (c i)^2 / (d i) = 0
  else False :=
by
  sorry

end sum_distances_l751_751950


namespace placement_possible_l751_751577

def can_place_numbers : Prop :=
  ∃ (top bottom : Fin 4 → ℕ), 
    (∀ i, (top i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (∀ i, (bottom i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (List.product (List.ofFn top) = List.product (List.ofFn bottom))

theorem placement_possible : can_place_numbers :=
sorry

end placement_possible_l751_751577


namespace noah_total_wattage_l751_751984

def bedroom_wattage := 6
def office_wattage := 3 * bedroom_wattage
def living_room_wattage := 4 * bedroom_wattage
def hours_on := 2

theorem noah_total_wattage : 
  bedroom_wattage * hours_on + 
  office_wattage * hours_on + 
  living_room_wattage * hours_on = 96 := by
  sorry

end noah_total_wattage_l751_751984


namespace solution_is_correct_l751_751723

noncomputable def solve_system_of_inequalities : Prop :=
  ∃ x y : ℝ, 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧ 
    (2 * x - 4 * y ≤ -3) ∧ 
    (x = -1/3) ∧ 
    (y = 2/3)

theorem solution_is_correct : solve_system_of_inequalities :=
sorry

end solution_is_correct_l751_751723


namespace total_books_l751_751612

def books_per_shelf_mystery : ℕ := 7
def books_per_shelf_picture : ℕ := 5
def books_per_shelf_sci_fi : ℕ := 8
def books_per_shelf_biography : ℕ := 6

def shelves_mystery : ℕ := 8
def shelves_picture : ℕ := 2
def shelves_sci_fi : ℕ := 3
def shelves_biography : ℕ := 4

theorem total_books :
  (books_per_shelf_mystery * shelves_mystery) + 
  (books_per_shelf_picture * shelves_picture) + 
  (books_per_shelf_sci_fi * shelves_sci_fi) + 
  (books_per_shelf_biography * shelves_biography) = 114 :=
by
  sorry

end total_books_l751_751612


namespace math_problem_l751_751227

noncomputable def a (n : ℕ) : ℚ :=
nat.rec_on n (1/2) (λ n a_n, 1 / (1 - a_n))

theorem math_problem :
  a 2 = 2 ∧ a 3 = -1 ∧ a 4 = 1/2 ∧ a 2010 = -1 ∧ a 2011 = 1/2 ∧ a 2012 = 2 :=
by
  sorry

end math_problem_l751_751227


namespace exists_m_min_m_l751_751343

-- Definition of the friendly condition for the competition
def friendly_competition (T : Type) [DecidableEq T] (R : T → T → Prop) :=
  ∀ x y : T, ∃ (u : Fin 100 → T) (k : ℕ), k ≥ 2 ∧ (x = u 0) ∧ (y = u (Fin.mk (k - 1) (by linarith))) ∧ 
    ∀ i : Fin (k-1), R (u i) (u (Fin.mk (i + 1) (by linarith [i.2])))

-- Given friendly competition result T, there exists a positive integer m
theorem exists_m (T : Type) [DecidableEq T] (R : T → T → Prop) (hT : friendly_competition T R) :
  ∃ m > 0, ∀ x y : T, ∃ (z : Fin m → T), (x = z 0) ∧ (y = z (Fin.mk (m - 1) (by linarith))) ∧ 
    ∀ i : Fin (m-1), R (z i) (z (Fin.mk (i + 1) (by linarith [i.2]))) :=
sorry

-- The minimum value of m(T) is 199
theorem min_m (T : Type) [DecidableEq T] (R : T → T → Prop) (hT : friendly_competition T R) :
  ∃ m : ℕ, m = 199 ∧ (∀ x y : T, ∃ (z : Fin m → T), (x = z 0) ∧ (y = z (Fin.mk (m - 1) (by linarith))) ∧ 
    ∀ i : Fin (m-1), R (z i) (z (Fin.mk (i + 1) (by linarith [i.2])))) :=
sorry

end exists_m_min_m_l751_751343


namespace inequality_true_l751_751905

theorem inequality_true (a b : ℝ) (h : a > b) : (2 * a - 1) > (2 * b - 1) :=
by {
  sorry
}

end inequality_true_l751_751905


namespace find_a_l751_751032

variable (a : ℝ) 

def p (x : ℝ) : Prop := 2 * x^2 - 5 * x + 3 < 0
def q (x : ℝ) : Prop := (x - (2 * a + 1)) * (x - 2 * a) ≤ 0

theorem find_a (h : ∀ x, p x → q x ∧ ∃ x, ¬ q x) : 1/4 ≤ a ∧ a ≤ 1/2 :=
by 
  sorry

end find_a_l751_751032


namespace max_value_of_f_range_sin_B_sin_C_l751_751892

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (2 * sin x, 2 * cos x)

noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (3 * sin x + 4 * cos x, -cos x)

noncomputable def f (x : ℝ) : ℝ := 
  let ⟨a1, a2⟩ := vector_a x
  let ⟨b1, b2⟩ := vector_b x
  a1 * b1 + a2 * b2

theorem max_value_of_f :
  ∃ x : ℝ, f x = 4 * Real.sqrt 2 + 2 :=
sorry

variables {A B C : ℝ} {a b c : ℝ}

theorem range_sin_B_sin_C (h₁ : 0 < B) (h₂ : B < π / 2) (h₃ : 0 < C) (h₄ : C < π / 2)
  (h₅ : A = π / 2 - B - C) (h₆ : f (B / 2 + π / 4) = 4 * c / a + 2) :
  ∃ I : set ℝ, I = set.Icc (Real.sqrt 2 / 2) ((2 + Real.sqrt 2) / 4) ∧ ∀ B C, sin B * sin C ∈ I :=
sorry

end max_value_of_f_range_sin_B_sin_C_l751_751892


namespace correct_proposition_about_empty_set_l751_751272

theorem correct_proposition_about_empty_set (A B C D : Prop) 
  (hA : "Very small integers" is uncertain -> ¬ A)
  (hB : ¬(setof y | y = 2*x^2 + 1 = setof (x, y) | y = 2*x^2 + 1) -> ¬ B)
  (hC : (|-1/2| = 0.5) ∧ (0.5 = 1/2) -> ¬ (set_size {1, 2, |-1/2|, 0.5, 1/2} = 5) -> ¬ C)
  (hD : ∀ (X : Set), ∅ ⊆ X) :
  B :=
by
  sorry

end correct_proposition_about_empty_set_l751_751272


namespace length_of_AB_l751_751931

variables (a b c : ℕ) (G I : Type)

-- Embed the conditions of the problem
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom perimeter : a + b + c = 35
axiom is_centroid : is_centroid(G, I, a, b, c)
axiom is_incenter : is_incenter(G, I, a, b, c)
axiom right_angle : ∠ G I C = 90

-- Define the proof target
theorem length_of_AB : c = 11 :=
sorry

end length_of_AB_l751_751931


namespace trigonometric_arithmetic_sequence_l751_751412

theorem trigonometric_arithmetic_sequence (a : ℝ) (h1 : 0 < a) (h2 : a < 360) :
  (sin (a * π / 180) + sin (3 * a * π / 180) = 2 * sin (2 * a * π / 180)) ↔ a = 180 :=
by sorry

end trigonometric_arithmetic_sequence_l751_751412


namespace value_of_diamond_l751_751074

def diamond (a b : ℕ) : ℕ := 4 * a + 2 * b

theorem value_of_diamond : diamond 6 3 = 30 :=
by {
  sorry
}

end value_of_diamond_l751_751074


namespace sin_alpha_value_l751_751860

open Real

theorem sin_alpha_value (α β : ℝ) 
  (h1 : cos (α - β) = 3 / 5) 
  (h2 : sin β = -5 / 13) 
  (h3 : 0 < α ∧ α < π / 2) 
  (h4 : -π / 2 < β ∧ β < 0) 
  : sin α = 33 / 65 :=
sorry

end sin_alpha_value_l751_751860


namespace height_at_15_inches_l751_751746

-- Define the conditions
def parabolic_eq (a x : ℝ) : ℝ := a * x^2 + 24
noncomputable def a : ℝ := -2 / 75
def x : ℝ := 15
def expected_y : ℝ := 18

-- Lean 4 statement
theorem height_at_15_inches :
  parabolic_eq a x = expected_y :=
by
  sorry

end height_at_15_inches_l751_751746


namespace more_freshmen_than_sophomores_l751_751532

variables (students : ℕ) (juniors_percent seniors_count not_sophomores_percent : ℝ)
variables (freshmen sophomores juniors seniors : ℕ)

-- Given conditions
def total_students : students = 800 := by sorry
def juniors_count : juniors = (22/100) * students := by sorry
def seniors_count : seniors = 160 := by sorry
def sophomores_percent : 1 - not_sophomores_percent = 0.26 := by sorry

theorem more_freshmen_than_sophomores :
  freshmen - sophomores = 48 := by
  sorry

end more_freshmen_than_sophomores_l751_751532


namespace impossible_to_partition_l751_751554

def regularHexagonArea (s : ℝ) : ℝ :=
  3 * (s * s * Real.sqrt 3 / 2)

def rightTriangleArea (a b : ℝ) : ℝ :=
  (a * b) / 2

theorem impossible_to_partition
  (s : ℝ) (a b : ℝ) (n : ℕ)
  (h_hexagon : s = 1)
  (h_triangle : a = 1 ∧ b = Real.sqrt 3)
  (h_area_eq : regularHexagonArea s + n * rightTriangleArea a b = regularHexagonArea 1 + n * rightTriangleArea 1 (Real.sqrt 3)) :
  False := sorry

end impossible_to_partition_l751_751554


namespace min_expression_value_l751_751825

theorem min_expression_value : 
  ∃ x : ℝ, ∀ x : ℝ, 
    \(\frac{\sin^6 x + \cos^6 x + 1}{\sin^4 x + \cos^4 x + 1}\)
    = \(\frac{5}{6}\) :=
by
  sorry

end min_expression_value_l751_751825


namespace nth_digit_100_sum_of_first_100_digits_l751_751753

def seq : List ℕ := [2, 9, 4, 7, 3, 6].cycle

-- Define a function to get the nth digit from the sequence
def nth_digit (n : ℕ) : ℕ :=
seq.get? (n - 1) |>.getD 0

-- Define a function to calculate the sum of the first n digits from the sequence
def sum_of_first_n_digits (n : ℕ) : ℕ :=
(seq.take n).sum

theorem nth_digit_100 : nth_digit 100 = 7 :=
by
  -- The proof for this statement would be here.
  sorry

theorem sum_of_first_100_digits : sum_of_first_n_digits 100 = 518 :=
by
  -- The proof for this statement would be here.
  sorry

end nth_digit_100_sum_of_first_100_digits_l751_751753


namespace solution_set_of_abs_inequality_l751_751659

theorem solution_set_of_abs_inequality : 
  {x : ℝ | abs (x - 1) - abs (x - 5) < 2} = {x : ℝ | x < 4} := 
by 
  sorry

end solution_set_of_abs_inequality_l751_751659


namespace jane_emily_total_accessories_l751_751944

def total_accessories : ℕ :=
  let jane_dresses := 4 * 10
  let emily_dresses := 3 * 8
  let jane_ribbons := 3 * jane_dresses
  let jane_buttons := 2 * jane_dresses
  let jane_lace_trims := 1 * jane_dresses
  let jane_beads := 4 * jane_dresses
  let emily_ribbons := 2 * emily_dresses
  let emily_buttons := 3 * emily_dresses
  let emily_lace_trims := 2 * emily_dresses
  let emily_beads := 5 * emily_dresses
  let emily_bows := 1 * emily_dresses
  jane_ribbons + jane_buttons + jane_lace_trims + jane_beads +
  emily_ribbons + emily_buttons + emily_lace_trims + emily_beads + emily_bows 

theorem jane_emily_total_accessories : total_accessories = 712 := 
by
  sorry

end jane_emily_total_accessories_l751_751944


namespace smallest_n_l751_751369

theorem smallest_n (n : ℕ) (y : Fin n → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ y i)
  (h_sum : (Finset.univ : Finset (Fin n)).sum y = 2)
  (h_sum_sq : (Finset.univ : Finset (Fin n)).sum (λ i, (y i) ^ 2) ≤ 2 / 50) :
  n ≥ 25 :=
sorry

end smallest_n_l751_751369


namespace rectangle_area_l751_751734

-- Definitions based on the conditions provided
variables {r : ℝ}
variables {AB BC CD : ℝ}

def is_tangent (circle_radius : ℝ) (rect_side1 rect_side2 rect_side3 : ℝ) : Prop :=
  rect_side1 = circle_radius ∧ rect_side2 = 2 * circle_radius ∧ rect_side3 = circle_radius

def center_on_diagonal (diag_length : ℝ) (point_pos : ℝ) : Prop :=
  point_pos = diag_length / 2

-- The main statement to prove that the area of the rectangle is 2r^2
theorem rectangle_area (r : ℝ) (AB BC CD : ℝ) (h1 : is_tangent r AB BC CD) (h2 : center_on_diagonal (sqrt (AB^2 + BC^2)) (r)) :
  AB * BC = 2 * r ^ 2 :=
by
  sorry

end rectangle_area_l751_751734


namespace white_marbles_in_C_equals_15_l751_751673

variables (A_red A_yellow B_green B_yellow C_yellow : ℕ) (w : ℕ)

-- Conditions from the problem
def conditions : Prop :=
  A_red = 4 ∧ A_yellow = 2 ∧
  B_green = 6 ∧ B_yellow = 1 ∧
  C_yellow = 9 ∧
  (A_red - A_yellow = 2) ∧
  (B_green - B_yellow = 5) ∧
  (w - C_yellow = 6)

-- Proving w = 15 given the conditions
theorem white_marbles_in_C_equals_15 (h : conditions A_red A_yellow B_green B_yellow C_yellow w) : w = 15 :=
  sorry

end white_marbles_in_C_equals_15_l751_751673


namespace distinct_values_for_T_l751_751928

-- Define the conditions given in the problem:
def distinct_digits (n : ℕ) : Prop :=
  n / 1000 ≠ (n / 100 % 10) ∧ n / 1000 ≠ (n / 10 % 10) ∧ n / 1000 ≠ (n % 10) ∧
  (n / 100 % 10) ≠ (n / 10 % 10) ∧ (n / 100 % 10) ≠ (n % 10) ∧
  (n / 10 % 10) ≠ (n % 10)

def Psum (P S T : ℕ) : Prop := P + S = T

-- Main theorem statement:
theorem distinct_values_for_T : ∀ (P S T : ℕ),
  distinct_digits P ∧ distinct_digits S ∧ distinct_digits T ∧
  Psum P S T → 
  (∃ (values : Finset ℕ), values.card = 7 ∧ ∀ val ∈ values, val = T) :=
by
  sorry

end distinct_values_for_T_l751_751928


namespace log_sum_reciprocals_of_logs_l751_751331

-- Problem (1)
theorem log_sum (log_two : Real.log 2 ≠ 0) :
    Real.log 4 / Real.log 10 + Real.log 50 / Real.log 10 - Real.log 2 / Real.log 10 = 2 := by
  sorry

-- Problem (2)
theorem reciprocals_of_logs (a b : Real) (h : 1 + Real.log a / Real.log 2 = 2 + Real.log b / Real.log 3 ∧ (1 + Real.log a / Real.log 2) = Real.log (a + b) / Real.log 6) : 
    1 / a + 1 / b = 6 := by
  sorry

end log_sum_reciprocals_of_logs_l751_751331


namespace jail_time_calculation_l751_751247

def total_arrests (arrests_per_day : ℕ) (cities : ℕ) (days : ℕ) : ℕ := 
  arrests_per_day * cities * days

def jail_time_before_trial (arrests : ℕ) (days_before_trial : ℕ) : ℕ := 
  days_before_trial * arrests

def jail_time_after_trial (arrests : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_after_trial * arrests

def combined_jail_time (weeks_before_trial : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_before_trial + weeks_after_trial

noncomputable def total_jail_time_in_weeks : ℕ := 
  let arrests := total_arrests 10 21 30
  let weeks_before_trial := jail_time_before_trial arrests 4 / 7
  let weeks_after_trial := jail_time_after_trial arrests 1
  combined_jail_time weeks_before_trial weeks_after_trial

theorem jail_time_calculation : 
  total_jail_time_in_weeks = 9900 :=
sorry

end jail_time_calculation_l751_751247


namespace hilt_books_difference_l751_751980

noncomputable def original_price : ℝ := 11
noncomputable def discount_rate : ℝ := 0.20
noncomputable def discount_price (price : ℝ) (rate : ℝ) : ℝ := price * (1 - rate)
noncomputable def quantity : ℕ := 15
noncomputable def sale_price : ℝ := 25
noncomputable def tax_rate : ℝ := 0.10
noncomputable def price_with_tax (price : ℝ) (rate : ℝ) : ℝ := price * (1 + rate)

noncomputable def total_cost : ℝ := discount_price original_price discount_rate * quantity
noncomputable def total_revenue : ℝ := price_with_tax sale_price tax_rate * quantity
noncomputable def profit : ℝ := total_revenue - total_cost

theorem hilt_books_difference : profit = 280.50 :=
by
  sorry

end hilt_books_difference_l751_751980


namespace total_pages_of_book_l751_751340

-- Definitions for the conditions
def firstChapterPages : Nat := 66
def secondChapterPages : Nat := 35
def thirdChapterPages : Nat := 24

-- Theorem stating the main question and answer
theorem total_pages_of_book : firstChapterPages + secondChapterPages + thirdChapterPages = 125 := by
  -- Proof will be provided here
  sorry

end total_pages_of_book_l751_751340


namespace nails_for_smaller_planks_l751_751011

def total_large_planks := 13
def nails_per_plank := 17
def total_nails := 229

def nails_for_large_planks : ℕ :=
  total_large_planks * nails_per_plank

theorem nails_for_smaller_planks :
  total_nails - nails_for_large_planks = 8 :=
by
  -- Proof goes here
  sorry

end nails_for_smaller_planks_l751_751011


namespace number_of_positive_area_triangles_l751_751813

theorem number_of_positive_area_triangles : 
  let A B C : Point,
      P1 P2 P3 : Point, -- Points on AB
      Q1 Q2 Q3 Q4 : Point, -- Points on BC
      R1 R2 R3 R4 R5 : Point -- Points on CA
  in
  (choose 15 3) - (choose 5 3) - (choose 6 3) - (choose 7 3) = 390 := 
by {
  sorry
}

end number_of_positive_area_triangles_l751_751813


namespace measure_angle_CAG_eq_48_l751_751404

-- Definitions:
def is_equilateral_triangle (A B C : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] :=
  ∀ (a b c : ℝ), a = b ∧ b = c ∧ c = a

def is_regular_pentagon (B C F G H : Type) [DecidableEq B] [DecidableEq C] [DecidableEq F] [DecidableEq G] [DecidableEq H] :=
  ∀ (bc cf fg gh hb : ℝ), bc = cf ∧ cf = fg ∧ fg = gh ∧ gh = hb ∧ hb = bc

-- Conditions:
variables {A B C F G H : Type} [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq F] [DecidableEq G] [DecidableEq H]
variable h_eq_triangle : is_equilateral_triangle A B C
variable h_reg_pentagon : is_regular_pentagon B C F G H
variable h_common_vertex : B = B
variable h_common_side : BC = BC

-- Proof problem:
theorem measure_angle_CAG_eq_48 : 
  ∠ C A G = 48 :=
sorry

end measure_angle_CAG_eq_48_l751_751404


namespace longest_segment_is_CD_l751_751113

-- Define points A, B, C, D
def A := (-3, 0)
def B := (0, 2)
def C := (3, 0)
def D := (0, -1)

-- Angles in triangle ABD
def angle_ABD := 35
def angle_BAD := 95
def angle_ADB := 50

-- Angles in triangle BCD
def angle_BCD := 55
def angle_BDC := 60
def angle_CBD := 65

-- Length comparison conclusion from triangle ABD
axiom compare_lengths_ABD : ∀ (AD AB BD : ℝ), AD < AB ∧ AB < BD

-- Length comparison conclusion from triangle BCD
axiom compare_lengths_BCD : ∀ (BC BD CD : ℝ), BC < BD ∧ BD < CD

-- Combine results
theorem longest_segment_is_CD : ∀ (AD AB BD BC CD : ℝ), AD < AB → AB < BD → BC < BD → BD < CD → CD ≥ AD ∧ CD ≥ AB ∧ CD ≥ BD ∧ CD ≥ BC :=
by
  intros AD AB BD BC CD h1 h2 h3 h4
  sorry

end longest_segment_is_CD_l751_751113


namespace find_angle_A_range_of_bc_l751_751936

variable {A B C a b c : ℝ}

-- Equivalent conditions 
def condition1 : Prop := 
  (a * c) / (b^2 - a^2 - c^2) = (sin A * cos A) / cos (A + C)

def given_triangle : Prop := a > 0 ∧ b > 0 ∧ c > 0 -- Basic triangle sides are all positive

theorem find_angle_A
  (h : given_triangle)
  (hc1 : condition1) :
  A = π / 4 :=
   sorry

theorem range_of_bc
  (h : given_triangle)
  (ha : a = sqrt 2)
  (hc1 : condition1) :
  A = π / 4 →
  0 < b * c ∧ b * c ≤ 2 + sqrt 2 :=
  sorry

end find_angle_A_range_of_bc_l751_751936


namespace value_y1_y2_l751_751176

variable {x1 x2 y1 y2 : ℝ}

-- Points on the inverse proportion function
def on_graph (x y : ℝ) : Prop := y = -3 / x

-- Given conditions
theorem value_y1_y2 (hx1 : on_graph x1 y1) (hx2 : on_graph x2 y2) (hxy : x1 * x2 = 2) : y1 * y2 = 9 / 2 :=
by
  sorry

end value_y1_y2_l751_751176


namespace probability_two_females_l751_751172

theorem probability_two_females (total_contestants female_contestants : ℕ) 
  (h_total : total_contestants = 5) (h_female : female_contestants = 3) : 
  (nat.choose female_contestants 2 / nat.choose total_contestants 2 : ℚ) = 3 / 10 :=
by
  sorry

end probability_two_females_l751_751172


namespace min_value_of_expression_l751_751001

theorem min_value_of_expression (x y z : ℝ) : ∃ a : ℝ, (∀ x y z : ℝ, x^2 + x * y + y^2 + y * z + z^2 ≥ a) ∧ (a = 0) :=
sorry

end min_value_of_expression_l751_751001


namespace correct_calculation_incorrect_option_A_incorrect_option_C_incorrect_option_D_l751_751708

theorem correct_calculation : (√12 / √3 = 2) :=
sorry

theorem incorrect_option_A : (√((-3)^2) ≠ -3) :=
sorry

theorem incorrect_option_C : (√(4 + 1/9) ≠ 2 + 1/3)%Q :=
sorry

theorem incorrect_option_D : ((-2 * √5)^2 ≠ 10) :=
sorry

end correct_calculation_incorrect_option_A_incorrect_option_C_incorrect_option_D_l751_751708


namespace min_f_l751_751973

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x then (x + 1) * Real.log x
else 2 * x + 3

noncomputable def f' (x : ℝ) : ℝ :=
if 0 < x then Real.log x + (x + 1) / x
else 2

theorem min_f'_for_x_pos : ∃ (c : ℝ), c = 2 ∧ ∀ x > 0, f' x ≥ c := 
  sorry

end min_f_l751_751973


namespace find_r_plus_s_l751_751540

-- Define the coordinates of points D and F
structure Point :=
  (x : ℝ)
  (y : ℝ)

def D := Point.mk 10 15
def F := Point.mk 19 18

-- Define the coordinates of point E
variables (r s : ℝ)
def E := Point.mk r s

-- Define the midpoint M of DF
def M := Point.mk ((D.x + F.x) / 2) ((D.y + F.y) / 2)

-- Define the slope condition for the median line to DF
def median_slope_condition : Prop :=
  ((s - M.y) / (r - M.x)) = -3

-- Define the area condition for triangle DEF
def area_condition : Prop :=
  65 = (1 / 2) * abs ((r * (D.y - F.y)) + (D.x * (F.y - s)) + (F.x * (s - D.y)))

-- Combine all conditions into a single statement
theorem find_r_plus_s (r s : ℝ) (h_slope : median_slope_condition r s) (h_area : area_condition r s) : 
  r + s = 96.9333 :=
sorry

end find_r_plus_s_l751_751540


namespace time_of_free_fall_l751_751654

theorem time_of_free_fall (h : ℝ) (t : ℝ) (height_fall_eq : h = 4.9 * t^2) (initial_height : h = 490) : t = 10 :=
by
  -- Proof is omitted
  sorry

end time_of_free_fall_l751_751654


namespace moles_CO2_required_l751_751827

theorem moles_CO2_required
  (moles_MgO : ℕ) 
  (moles_MgCO3 : ℕ) 
  (balanced_equation : ∀ (MgO CO2 MgCO3 : ℕ), MgO + CO2 = MgCO3) 
  (reaction_produces : moles_MgO = 3 ∧ moles_MgCO3 = 3) :
  3 = 3 :=
by
  sorry

end moles_CO2_required_l751_751827


namespace concentric_circle_problem_l751_751776

theorem concentric_circle_problem (n : ℕ) (h : n ≥ 3) :
  (∃ (perm : Fin n → Fin n), ∀ k, ∃ unique m, (perm m + k) % n = m + k % n) ↔ Odd n :=
by {
  sorry -- Proof is skipped as per the instruction.
}

end concentric_circle_problem_l751_751776


namespace part_I_part_II_l751_751881

-- Definition of the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Part (I) Statement
theorem part_I : ∀ x : ℝ, f(x + 2) ≥ 2 ↔ (x ≤ -3/2 ∨ x ≥ 1/2) :=
by
  sorry

-- Part (II) Statement
theorem part_II (a : ℝ) : (∀ x : ℝ, f(x) ≥ a) → (a ≤ 1) :=
by
  sorry

end part_I_part_II_l751_751881


namespace problem_l751_751679

theorem problem (PQ QR PR : ℝ) (PQRRight : (PQ^2 + QR^2 = PR^2)) (S_on_PR : S ∈ PR)
  (QS_bisects_P_Q : bisects QS ⟨P, Q, S⟩) (r1 r2 : ℝ) :
  PQ = 5 ∧ QR = 12 ∧ PR = 13 ∧ ∃ (r1 r2 : ℝ),
    (ratio_of_inradii := r1 / r2) ∈ { (1 / 28) * (10 - √2), 
                                      (3 / 56) * (10 - √2), 
                                      (1 / 14) * (10 - √2),
                                      (5 / 56) * (10 - √2), 
                                      (3 / 28) * (10 - √2) } :=
sorry

end problem_l751_751679


namespace minimum_cost_is_8600_l751_751626

-- Defining the conditions
def shanghai_units : ℕ := 12
def nanjing_units : ℕ := 6
def suzhou_needs : ℕ := 10
def changsha_needs : ℕ := 8
def cost_shanghai_suzhou : ℕ := 400
def cost_shanghai_changsha : ℕ := 800
def cost_nanjing_suzhou : ℕ := 300
def cost_nanjing_changsha : ℕ := 500

-- Defining the function for total shipping cost
def total_shipping_cost (x : ℕ) : ℕ :=
  cost_shanghai_suzhou * x +
  cost_shanghai_changsha * (shanghai_units - x) +
  cost_nanjing_suzhou * (suzhou_needs - x) +
  cost_nanjing_changsha * (x - (shanghai_units - suzhou_needs))

-- Define the minimum shipping cost function
def minimum_shipping_cost : ℕ :=
  total_shipping_cost 10

-- State the theorem to prove
theorem minimum_cost_is_8600 : minimum_shipping_cost = 8600 :=
sorry

end minimum_cost_is_8600_l751_751626


namespace problem_solution_l751_751938

noncomputable def triangle_ineq (AC BC AB : ℝ) : Prop :=
  AC + BC > AB ∧ AB + BC > AC ∧ AC + AB > BC

theorem problem_solution :
  ∃ (m n : ℝ), (∀ x : ℝ, triangle_ineq x (32 / x + 4) 8 → 4 < x ∧ x < 16) ∧ m + n = 20 :=
by {
  use [4, 16],
  intros x hx,
  sorry
}

end problem_solution_l751_751938


namespace coeff_of_term_l751_751254

theorem coeff_of_term (a b c : ℝ) :
  (coeff (a^3 * b^3 * c^2) (((a + b)^6) * (c + c⁻¹)^8)) = 1120 :=
by
  sorry

end coeff_of_term_l751_751254


namespace angle_AMD_eq_angle_CMD_l751_751584

noncomputable def quadInscribedInCircle (A B C D M : Point) (circle : Circle) :=
  IsInscribedQuadrilateral A B C D circle ∧ -- A, B, C, D inscribed in circle
  Midpoint M B D ∧ -- M is midpoint of segment BD
  AreConcurrent (TangentAt circle B) (TangentAt circle D) (Extension A C) -- Tangents at B and D concurrent with the extension of AC

theorem angle_AMD_eq_angle_CMD (A B C D M : Point) (circle : Circle) 
  (h : quadInscribedInCircle A B C D M circle) : 
  ∠ A M D = ∠ C M D := 
sorry -- Proof omitted

end angle_AMD_eq_angle_CMD_l751_751584


namespace circumcenter_ratios_l751_751063

/-- Given a triangle ABC with circumcenter O and circumradius R, 
     and points D, E, and F where AO, BO, and CO intersect the 
     respective opposite sides, prove that:
     1/AD + 1/BE + 1/CF = 2/R. -/
theorem circumcenter_ratios (A B C O D E F : Point) (R : Real)
  (hO : is_circumcenter A B C O)
  (hR : is_circumradius O R)
  (hD : is_intersection_of AO BC D)
  (hE : is_intersection_of BO CA E)
  (hF : is_intersection_of CO AB F) :
  1 / dist A D + 1 / dist B E + 1 / dist C F = 2 / R :=
by
  sorry

end circumcenter_ratios_l751_751063


namespace sin_A_value_triangle_area_l751_751551

-- Definition of the given conditions
variables {a b c : ℝ}
variables (A B C : ℝ)
variables (triangle_ABC : Prop) [triangle_ABC : IsTriangle a b c]

-- The conditions
axiom condition1 : 4 * a = Real.sqrt 5 * c
axiom condition2 : Real.cos C = 3 / 5
axiom condition3 : b = 11
axiom opposite_sides : SidesOppositeToAngles a b c A B C

-- Proving the required results
theorem sin_A_value : sin A = Real.sqrt 5 / 5 :=
by
  sorry

theorem triangle_area : area_triangle a b c = 22 :=
by
  sorry

end sin_A_value_triangle_area_l751_751551


namespace condition_p_implies_condition_q_l751_751452

-- Definitions of conditions p and q
def condition_p (f : ℝ → ℝ) (m : ℝ) :=
  ∀ x, x > (1 : ℝ) / 2 → derivative f x ≥ 0

def condition_q (m : ℝ) :=
  m ≥ - 4 / 3

-- Definition of quadratic function f
def quadratic (m : ℝ) : ℝ → ℝ := λ x, x^2 + m * x + 1

-- The mathematical statement
theorem condition_p_implies_condition_q (m : ℝ) :
  condition_p (quadratic m) m → condition_q m :=
by
  sorry

end condition_p_implies_condition_q_l751_751452


namespace minimum_ab_value_is_two_l751_751038

noncomputable def minimum_value_ab (a b : ℝ) (h1 : a^2 ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b = a^2 + 1) : ℝ :=
|a * b|

theorem minimum_ab_value_is_two (a b : ℝ) (h1 : a^2 ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b = a^2 + 1) : minimum_value_ab a b h1 h2 h3 = 2 := by
  sorry

end minimum_ab_value_is_two_l751_751038


namespace no_solutions_iff_l751_751418

theorem no_solutions_iff (a : ℝ) : (-∞ < a ∧ a < -Real.sqrt 5 / 2) ∨ (Real.sqrt 5 / 2 < a ∧ a < ∞) ↔
  ¬ (∃ x : ℝ, |x| - x^2 = a^2 - sin (π * x)^2) := 
sorry

end no_solutions_iff_l751_751418


namespace inequality_proof_l751_751290

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751290


namespace coordinate_relationship_l751_751368

theorem coordinate_relationship (x y : ℝ) (h : |x| - |y| = 0) : (|x| - |y| = 0) :=
by
    sorry

end coordinate_relationship_l751_751368


namespace jeff_can_store_songs_l751_751945

def gbToMb (gb : ℕ) : ℕ := gb * 1000

def newAppsStorage : ℕ :=
  5 * 450 + 5 * 300 + 5 * 150

def newPhotosStorage : ℕ :=
  300 * 4 + 50 * 8

def newVideosStorage : ℕ :=
  15 * 400 + 30 * 200

def newPDFsStorage : ℕ :=
  25 * 20

def totalNewStorage : ℕ :=
  newAppsStorage + newPhotosStorage + newVideosStorage + newPDFsStorage

def existingStorage : ℕ :=
  gbToMb 7

def totalUsedStorage : ℕ :=
  existingStorage + totalNewStorage

def totalStorage : ℕ :=
  gbToMb 32

def remainingStorage : ℕ :=
  totalStorage - totalUsedStorage

def numSongs (storage : ℕ) (avgSongSize : ℕ) : ℕ :=
  storage / avgSongSize

theorem jeff_can_store_songs : 
  numSongs remainingStorage 20 = 320 :=
by
  sorry

end jeff_can_store_songs_l751_751945


namespace number_of_valid_triples_l751_751002

theorem number_of_valid_triples : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|)
    → (∃ (s : Set (ℝ × ℝ × ℝ)), s.card = 6 ∧ ∀ (a, b, c) ∈ s, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) := 
  by
    sorry

end number_of_valid_triples_l751_751002


namespace noah_total_wattage_l751_751985

def bedroom_wattage := 6
def office_wattage := 3 * bedroom_wattage
def living_room_wattage := 4 * bedroom_wattage
def hours_on := 2

theorem noah_total_wattage : 
  bedroom_wattage * hours_on + 
  office_wattage * hours_on + 
  living_room_wattage * hours_on = 96 := by
  sorry

end noah_total_wattage_l751_751985


namespace part_a_sum_part_b_sum_l751_751719

noncomputable def f (x : ℝ) : ℝ := x^2
def f_periodic : ∀ x, f (x + 2 * Real.pi) = f x := 
  by intro x; rw [f, f]; have : (x + 2 * Real.pi)^2 = x^2; ring; exact this

def fourier_series (n : ℕ) : ℝ :=
  if n = 0 then (2 * Real.pi ^ 2) / 3
  else (-1) ^ n * (4 / n^2)

noncomputable def alternating_series (x : ℝ) (series_sum : ℝ → ℝ) : ℝ :=
  x - 4 * series_sum (λ n, (-1)^n * (1 / n^2))

noncomputable def series_sum (series_sum : ℝ → ℝ) : ℝ :=
  4 * series_sum (λ n, 1 / n^2)

theorem part_a_sum :
  alternating_series 0 (λ n, ∑ i in finset.range n, (-1)^i * (1 / (i + 1)^2)) = Real.pi^2 / 12 :=
  sorry

theorem part_b_sum :
  series_sum (λ n, ∑ i in finset.range n, 1 / (i + 1)^2) = Real.pi^2 / 6 :=
  sorry

end part_a_sum_part_b_sum_l751_751719


namespace final_number_at_least_one_over_n_l751_751853

theorem final_number_at_least_one_over_n (n : ℕ) (h : n > 0) : 
  ∀ (process : ((fin n) → ℝ) → ((fin n) → ℝ)), 
  (∀ x, (process (λ _, 1)).card = 1) →
  (∀ x y, process (λ _, 1) = λ i, if i = fin.mk 0 sorry then (x + y) / 4 else (λ _, x + y) / 4) → 
  (process (λ _, 1) (fin.mk 0 sorry) ≥ (1 / n : ℝ)) :=
sorry

end final_number_at_least_one_over_n_l751_751853


namespace Pam_read_more_than_Harrison_l751_751225

theorem Pam_read_more_than_Harrison :
  ∀ (assigned : ℕ) (Harrison : ℕ) (Pam : ℕ) (Sam : ℕ),
    assigned = 25 →
    Harrison = assigned + 10 →
    Sam = 2 * Pam →
    Sam = 100 →
    Pam - Harrison = 15 :=
by
  intros assigned Harrison Pam Sam h1 h2 h3 h4
  sorry

end Pam_read_more_than_Harrison_l751_751225


namespace placement_possible_l751_751578

def can_place_numbers : Prop :=
  ∃ (top bottom : Fin 4 → ℕ), 
    (∀ i, (top i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (∀ i, (bottom i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (List.product (List.ofFn top) = List.product (List.ofFn bottom))

theorem placement_possible : can_place_numbers :=
sorry

end placement_possible_l751_751578


namespace smaller_container_capacity_proof_l751_751346

-- Definitions for the conditions
def large_container_30_percent_full (C : ℝ) : ℝ := 0.30 * C
def large_container_after_addition (C : ℝ) : ℝ := large_container_30_percent_full C + 27
def large_container_75_percent_full (C : ℝ) : ℝ := 0.75 * C
def smaller_container_capacity (C : ℝ) : ℝ := (large_container_30_percent_full C) / 2

-- Proof statement
theorem smaller_container_capacity_proof : ∀ (C : ℝ), 
  large_container_after_addition C = large_container_75_percent_full C →
  smaller_container_capacity C = 9 := 
by
  intros C h
  sorry

end smaller_container_capacity_proof_l751_751346


namespace total_clips_avg_earning_per_clip_l751_751502

variable (x W : ℝ)

-- Conditions
def clips_in_april := x
def clips_in_may := x / 2
def clips_in_june := 0.625 * x
def total_earnings := W

-- Statement for total clips sold
theorem total_clips (x : ℝ) : clips_in_april x + clips_in_may x + clips_in_june x = 2.125 * x :=
by
  sorry

-- Statement for average earning per clip
theorem avg_earning_per_clip (x W : ℝ) :
  total_earnings W / (clips_in_april x + clips_in_may x + clips_in_june x) = W / (2.125 * x) :=
by
  sorry

end total_clips_avg_earning_per_clip_l751_751502


namespace smallest_n_ineq_l751_751259

theorem smallest_n_ineq : ∃ n : ℕ, 3 * Real.sqrt n - 2 * Real.sqrt (n - 1) < 0.03 ∧ 
  (∀ m : ℕ, (3 * Real.sqrt m - 2 * Real.sqrt (m - 1) < 0.03) → n ≤ m) ∧ n = 433715589 :=
by
  sorry

end smallest_n_ineq_l751_751259


namespace haley_tickets_l751_751494

-- Conditions
def cost_per_ticket : ℕ := 4
def extra_tickets : ℕ := 5
def total_spent : ℕ := 32
def cost_extra_tickets : ℕ := extra_tickets * cost_per_ticket

-- Main proof problem
theorem haley_tickets (T : ℕ) (h : 4 * T + cost_extra_tickets = total_spent) :
  T = 3 := sorry

end haley_tickets_l751_751494


namespace inequality_holds_l751_751305

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l751_751305


namespace expressions_same_type_l751_751772

def same_type_as (e1 e2 : ℕ × ℕ) : Prop :=
  e1 = e2

def exp_of_expr (a_exp b_exp : ℕ) : ℕ × ℕ :=
  (a_exp, b_exp)

def exp_3a2b := exp_of_expr 2 1
def exp_neg_ba2 := exp_of_expr 2 1

theorem expressions_same_type :
  same_type_as exp_neg_ba2 exp_3a2b :=
by
  sorry

end expressions_same_type_l751_751772


namespace sum_of_inverses_geq_one_l751_751154

-- Define what it means to be a k-set
def is_k_set (A : Set ℤ) (k : ℕ) : Prop :=
  ∃ (x : Fin k → ℤ), ∀ (i j : Fin k), i ≠ j → Disjoint ((fun xi => xi + A) (x i)) ((fun xi => xi + A) (x j))

-- State the main theorem
theorem sum_of_inverses_geq_one 
  (t : ℕ) 
  (A : Fin t → Set ℤ)
  (k : Fin t → ℕ)
  (H_tk : ∀ i, is_k_set (A i) (k i))
  (union_A_is_Z : (⋃ i, A i) = Set.univ) :
  (∑ i in Finset.univ, (1 : ℚ) / k i) ≥ 1 := sorry

end sum_of_inverses_geq_one_l751_751154


namespace coordinates_of_point_A_l751_751925

theorem coordinates_of_point_A (A' : ℝ × ℝ) (hx : A'.fst = 3) (hy : A'.snd = 2) : ∃ A : ℝ × ℝ, A.fst + 2 = A'.fst ∧ A.snd = A'.snd :=
begin
  have hA : (1 : ℝ, 2 : ℝ).fst + 2 = A'.fst ∧ (1 : ℝ, 2 : ℝ).snd = A'.snd,
  { split;
    { norm_num } },
  exact ⟨(1, 2), hA⟩,
end

end coordinates_of_point_A_l751_751925


namespace vietnamese_dishes_distribution_l751_751008

-- Define the main problem **statement**.
theorem vietnamese_dishes_distribution : 
    let dishes := ["Phở", "Nem", "Bún Chả", "Bánh cuốn", "Xôi gà"] in
    let days := [Monday, Tuesday, Wednesday] in
    (∏ (_ in dishes), 3) - 
    (∏ (_ in ["Only one day"]), 3 * 1) - 
    (∏ (_ in ["Two days"]), 3 * (2^5 - 2)) 
    = 150 :=
begin
    sorry
end

end vietnamese_dishes_distribution_l751_751008


namespace prob_car_z_l751_751094

theorem prob_car_z {P : Type} [ProbabilitySpace P] (P_X P_Y P_Z : P → ℝ) :
  P_X = (λ _, 1 / 8) ∧ P_Y = (λ _, 1 / 12) ∧ (∀ p, P_X p + P_Y p + P_Z p = 0.375) → 
  (∀ p, P_Z p = 1 / 6) :=
begin
  sorry
end

end prob_car_z_l751_751094


namespace quadrilateral_area_inequality_l751_751713

variables {a b c d S : ℝ}

theorem quadrilateral_area_inequality
  (h1 : S = area_of_quadrilateral a b c d (a + c))
  (h2 : a + b > 0)
  (h3 : c + d > 0) :
  S ≤ 1 / 4 * (a + b) * (c + d) :=
sorry

end quadrilateral_area_inequality_l751_751713


namespace max_min_values_f_l751_751849

def f (x : ℝ) : ℝ :=
  if x ∈ Icc 1 2 then 2 * x + 6
  else if x ∈ Icc (-1) 1 then x + 7
  else 0  -- this piece won't affect our proof since x will be within known intervals

theorem max_min_values_f :
  (∀ x, f x ≤ 10) ∧ (∃ x, f x = 10) ∧ (∀ x, 6 ≤ f x) ∧ (∃ x, f x = 6) :=
sorry

end max_min_values_f_l751_751849


namespace equilateral_triangle_side_length_l751_751635

theorem equilateral_triangle_side_length (s x : ℝ)
  (h₁ : ∀ (A B C P : ℝ×ℝ), ∃ s, equilateral_triangle A B C ∧ inside_triangle P A B C ∧ AP = x ∧ BP = x+1 ∧ CP = x+2)
  (h₂ : area_triangle ABC = (1/2) * area_hexagon_by_reflections P A B C) :
  s = sqrt(11) := sorry

# Definitions for equilateral_triangle, inside_triangle, AP, BP, CP, area_triangle, and area_hexagon_by_reflections would need to be established separately.

end equilateral_triangle_side_length_l751_751635


namespace multiply_meaning_example_problem_l751_751334

theorem multiply_meaning (x y : ℕ) : x * y = x + (y - 1) * x :=
by
  sorry

theorem example_problem : 28 * 5 = "enlarge" 28 by 5 times :=
by
  sorry

end multiply_meaning_example_problem_l751_751334


namespace sum_of_eight_numbers_l751_751514

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l751_751514


namespace painting_colors_area_l751_751361

theorem painting_colors_area
  (B G Y : ℕ)
  (h_total_blue : B + (1 / 3 : ℝ) * G = 38)
  (h_total_yellow : Y + (2 / 3 : ℝ) * G = 38)
  (h_grass_sky_relation : G = B + 6) :
  B = 27 ∧ G = 33 ∧ Y = 16 :=
by
  sorry

end painting_colors_area_l751_751361


namespace locus_is_hyperbola_l751_751469

theorem locus_is_hyperbola
  (x y a θ₁ θ₂ c : ℝ)
  (h1 : (x - a) * Real.cos θ₁ + y * Real.sin θ₁ = a)
  (h2 : (x - a) * Real.cos θ₂ + y * Real.sin θ₂ = a)
  (h3 : Real.tan (θ₁ / 2) - Real.tan (θ₂ / 2) = 2 * c)
  (hc : c > 1) 
  : ∃ k l m : ℝ, k * (x ^ 2) + l * x * y + m * (y ^ 2) = 1 := sorry

end locus_is_hyperbola_l751_751469


namespace integral_sqrt_plus_linear_l751_751808

theorem integral_sqrt_plus_linear :
  (∫ x in 0..1, sqrt (1 - x^2) + (1/2) * x) = (π + 1) / 4 :=
by
  sorry

end integral_sqrt_plus_linear_l751_751808


namespace probability_arithmetic_sequence_3_of_20_l751_751439

theorem probability_arithmetic_sequence_3_of_20 :
  let S := {1, 2, ..., 20}
  let count_triplets := (S.choose 3).count
  let valid_triplets := ((S.choose 3).filter (λ (t : Finset ℕ), let l := t.to_list in 2 * l[1] = l[0] + l[2])).count
  in (valid_triplets : ℚ) / count_triplets = 1 / 38 :=
by
  sorry

end probability_arithmetic_sequence_3_of_20_l751_751439


namespace power_of_integer_is_two_l751_751910

-- Definitions based on conditions
def is_power_of_integer (n : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ), n = m^k

-- Given conditions translated to Lean definitions
def g : ℕ := 14
def n : ℕ := 3150 * g

-- The proof problem statement in Lean
theorem power_of_integer_is_two (h : g = 14) : is_power_of_integer n :=
sorry

end power_of_integer_is_two_l751_751910


namespace sin_450_eq_1_l751_751383

theorem sin_450_eq_1 :
  sin (450 * Real.pi / 180) = 1 :=
by
  sorry

end sin_450_eq_1_l751_751383


namespace minimum_cost_of_candies_l751_751360

variable (Orange Apple Grape Strawberry : ℕ)

-- Conditions
def CandyRelation1 := Apple = 2 * Orange
def CandyRelation2 := Strawberry = 2 * Grape
def CandyRelation3 := Apple = 2 * Strawberry
def TotalCandies := Orange + Apple + Grape + Strawberry = 90
def CandyCost := 0.1

-- Question
theorem minimum_cost_of_candies :
  CandyRelation1 Orange Apple → 
  CandyRelation2 Grape Strawberry → 
  CandyRelation3 Apple Strawberry → 
  TotalCandies Orange Apple Grape Strawberry → 
  Orange ≥ 3 ∧ Apple ≥ 3 ∧ Grape ≥ 3 ∧ Strawberry ≥ 3 →
  (5 * CandyCost + 3 * CandyCost + 3 * CandyCost + 3 * CandyCost = 1.4) :=
sorry

end minimum_cost_of_candies_l751_751360


namespace collinear_points_m_equals_4_l751_751085

theorem collinear_points_m_equals_4 (m : ℝ)
  (h1 : (3 - 12) / (1 - -2) = (-6 - 12) / (m - -2)) : m = 4 :=
by
  sorry

end collinear_points_m_equals_4_l751_751085


namespace isosceles_triangle_area_l751_751082

theorem isosceles_triangle_area 
  (h : height AD = 18) 
  (m : median BM = 15) : 
  area ABC = 144 := 
sorry

end isosceles_triangle_area_l751_751082


namespace smallest_n_l751_751263

theorem smallest_n (n : ℕ) (h : ↑n > 0 ∧ (Real.sqrt (↑n) - Real.sqrt (↑n - 1)) < 0.02) : n = 626 := 
by
  sorry

end smallest_n_l751_751263


namespace cos_Z_and_sin_X_l751_751922

variable {XYZ : Type}
variable [right_triangle XYZ]
variable (X Y Z : XYZ)
variable (angle_X : ∠ X = 90)
variable (XY_length : XY = 8)
variable (YZ_length : YZ = 17)

theorem cos_Z_and_sin_X (h : right_triangle XYZ) (hx : ∠ X = 90) (hXY : XY = 8) (hYZ : YZ  = 17) : 
  cos Z = 15 / 17 ∧ sin X = 1 := 
sorry

end cos_Z_and_sin_X_l751_751922


namespace min_chord_length_l751_751484

noncomputable def circle_center : (ℝ × ℝ) := (2, 0)
noncomputable def circle_radius : ℝ := sqrt 3

def line_eq (k x : ℝ) : ℝ := k * (x - 1) + 1
def circle_eq (x y : ℝ) : ℝ := x^2 - 4 * x + y^2 + 1

def dist_point_line (k : ℝ) : ℝ := abs (k + 1) / sqrt (k^2 + 1)

theorem min_chord_length (k : ℝ) : 2 * sqrt(2 - (2 * k) / (k^2 + 1)) ≥ 2 := by
  sorry

end min_chord_length_l751_751484


namespace circle_area_above_line_l751_751253

theorem circle_area_above_line : 
  let circle_eq := ∀ x y, x^2 + y^2 + 6 * x - 8 * y = 0
  let line_eq := ∀ x y, y = -x
  ∃ a : ℝ, a = 12.5 * Real.pi ∧ 
    (∀ x y, (circle_eq x y → y ≥ -x) → sorry) :=
sorry

end circle_area_above_line_l751_751253


namespace cube_edge_numbers_equal_top_bottom_l751_751562

theorem cube_edge_numbers_equal_top_bottom (
  numbers : List ℕ,
  h : numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) :
  ∃ (top bottom : List ℕ),
    (∀ x, x ∈ top → x ∈ numbers) ∧
    (∀ x, x ∈ bottom → x ∈ numbers) ∧
    (top ≠ bottom) ∧
    (top.length = 4) ∧ 
    (bottom.length = 4) ∧ 
    (top.product = bottom.product) :=
begin
  sorry
end

end cube_edge_numbers_equal_top_bottom_l751_751562


namespace range_of_a_l751_751481

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (4 ≤ y ∧ y ≤ 5) → x * y ≤ a * x^2 + 2 * y^2) ↔ a ≥ -6 :=
by
  sorry

end range_of_a_l751_751481


namespace placement_possible_l751_751576

def can_place_numbers : Prop :=
  ∃ (top bottom : Fin 4 → ℕ), 
    (∀ i, (top i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (∀ i, (bottom i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (List.product (List.ofFn top) = List.product (List.ofFn bottom))

theorem placement_possible : can_place_numbers :=
sorry

end placement_possible_l751_751576


namespace num_books_l751_751336

-- Definitions of the conditions
variables (P : Type) [Fintype P] [DecidableEq P] 
           (B : Type) [Fintype B] [DecidableEq B] 
           (bought : P → Finset B)
           
-- Condition: There are 579 people
def num_people : ℕ := Fintype.card P
def four_books_each (p : P) : (bought p).card = 4
def common_books (p q : P) (h : p ≠ q) : (bought p ∩ bought q).card = 2

-- Theorem statement
theorem num_books (h_people : num_people = 579) 
  (h_each : ∀ p : P, four_books_each p) 
  (h_common : ∀ p q : P, p ≠ q → common_books p q) :
  Fintype.card B = 20 :=
sorry

end num_books_l751_751336


namespace julie_initial_savings_l751_751327

theorem julie_initial_savings (S r : ℝ) 
  (h1 : (S / 2) * r * 2 = 120) 
  (h2 : (S / 2) * ((1 + r)^2 - 1) = 124) : 
  S = 1800 := 
sorry

end julie_initial_savings_l751_751327


namespace binary_to_decimal_110011_l751_751396

theorem binary_to_decimal_110011 :
  1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 51 :=
by
  sorry

end binary_to_decimal_110011_l751_751396


namespace find_m_l751_751054

-- Define the set A
def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3 * m + 2}

-- Main theorem statement
theorem find_m (m : ℝ) (h : 2 ∈ A m) : m = 3 := by
  sorry

end find_m_l751_751054


namespace sum_f_to_22_l751_751869

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom a1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom a2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom a3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom a4 : g(2) = 4

theorem sum_f_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_to_22_l751_751869


namespace neg_cube_squared_l751_751726

theorem neg_cube_squared (x : ℝ) : (-x^3) ^ 2 = x ^ 6 :=
by
  sorry

end neg_cube_squared_l751_751726


namespace sum_of_eight_numbers_l751_751516

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l751_751516


namespace cotangent_expression_l751_751142

noncomputable theory

variables {a b c : ℝ} {α β γ : ℝ}
variables (h1 : a^2 + b^2 = 2010 * c^2)
variables (h2 : c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos γ)
variables (h3 : Real.sin α = a / c * Real.sin γ)
variables (h4 : Real.sin β = b / c * Real.sin γ)

theorem cotangent_expression (h1 : a^2 + b^2 = 2010 * c^2)
                             (h2 : c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos γ)
                             (h3 : Real.sin α = a / c * Real.sin γ)
                             (h4 : Real.sin β = b / c * Real.sin γ) :
    (Real.cot γ) / (Real.cot α + Real.cot β) = 1004.5 :=
sorry

end cotangent_expression_l751_751142


namespace inequality_CD_plus_EF_less_than_AC_l751_751019

-- Define the cyclic quadrilateral and other geometrical constructs
variables {A B C D E F : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]

-- Define the necessary angles and lengths for A, B, C, D, E and F
variables (α β γ δ ε ζ : ℝ) -- Representing angles in radians

-- Given conditions
axiom cyclic_quadrilateral (A B C D : Type) : Prop
axiom angle_adds_to_right : ∠ B A C + ∠ B A D = π / 2
axiom E_on_diagonal_BD (B D E : Type) : Prop
axiom perpendicular_from_E_to_AB (E F A B : Type) : Prop
axiom lengths_AD_BE_equal : ∀ {AD BE : ℝ}, AD = BE
axiom EF_perpendicular_to_AB {E A B : Type} : Prop

-- The final theorem we want to prove
theorem inequality_CD_plus_EF_less_than_AC 
  [h1 : cyclic_quadrilateral A B C D]
  [h2 : angle_adds_to_right α β]
  [h3 : E_on_diagonal_BD B D E]
  [h4 : perpendicular_from_E_to_AB E F A B]
  [h5 : lengths_AD_BE_equal]
  [h6 : EF_perpendicular_to_AB] :
  CD + EF < AC := sorry

end inequality_CD_plus_EF_less_than_AC_l751_751019


namespace inequality_inequality_holds_l751_751319

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l751_751319


namespace paperback_copies_sold_l751_751231

theorem paperback_copies_sold
  (H P : ℕ)
  (h1 : H = 36000)
  (h2 : H + P = 440000) :
  P = 404000 :=
by
  rw [h1] at h2
  sorry

end paperback_copies_sold_l751_751231


namespace angle_Z_90_l751_751090

-- Definitions and conditions from step a)
def Triangle (X Y Z : ℝ) : Prop :=
  X + Y + Z = 180

def in_triangle_XYZ (X Y Z : ℝ) : Prop :=
  Triangle X Y Z ∧ (X + Y = 90)

-- Proof problem from step c)
theorem angle_Z_90 (X Y Z : ℝ) (h : in_triangle_XYZ X Y Z) : Z = 90 :=
  by
  sorry

end angle_Z_90_l751_751090


namespace area_of_rectangle_given_conditions_l751_751759

-- Defining the conditions given in the problem
variables (s d r a : ℝ)

-- Given conditions for the problem
def is_square_inscribed_in_circle (s d : ℝ) := 
  d = s * Real.sqrt 2 ∧ 
  d = 4

def is_circle_inscribed_in_rectangle (r : ℝ) :=
  r = 2

def rectangle_dimensions (length width : ℝ) :=
  length = 2 * width ∧ 
  width = 2

-- The theorem we want to prove
theorem area_of_rectangle_given_conditions :
  ∀ (s d r length width : ℝ),
  is_square_inscribed_in_circle s d →
  is_circle_inscribed_in_rectangle r →
  rectangle_dimensions length width →
  a = length * width →
  a = 8 :=
by
  intros s d r length width h1 h2 h3 h4
  sorry

end area_of_rectangle_given_conditions_l751_751759


namespace fraction_simplification_l751_751391

theorem fraction_simplification :
  (2 ^ 1010) ^ 2 - (2 ^ 1008) ^ 2) / ((2 ^ 1009) ^ 2 - (2 ^ 1007) ^ 2) = 4 := 
by {
  sorry
}

end fraction_simplification_l751_751391


namespace probability_B_not_losing_l751_751091

-- Let P(draw) = 1/2
def P_draw := (1 : ℝ) / 2

-- Let P(win_B) = 1/3
def P_win_B := (1 : ℝ) / 3

-- Prove that P(not_lose_B) = 5/6
theorem probability_B_not_losing : P_draw + P_win_B = (5 : ℝ) / 6 := by
  sorry

end probability_B_not_losing_l751_751091


namespace smallest_positive_period_value_at_pi_over_8_l751_751521

def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 6)

theorem smallest_positive_period :
  ∃ p > 0, ∀ x, f (x + p) = f x ∧ p = Real.pi / 2 :=
sorry

theorem value_at_pi_over_8 : 
  f (Real.pi / 8) = 2 - Real.sqrt 3 :=
sorry

end smallest_positive_period_value_at_pi_over_8_l751_751521


namespace range_of_a_l751_751473

def quadratic_function (a x : ℝ) : ℝ := x^2 - 4 * x + a

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (0 : ℝ) 1, quadratic_function a x ≥ 0) ↔ 3 ≤ a :=
by sorry

end range_of_a_l751_751473


namespace find_constants_l751_751818

theorem find_constants (A B C : ℝ) (hA : A = 7) (hB : B = -9) (hC : C = 5) :
  (∀ (x : ℝ), x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → 
    ( -2 * x ^ 2 + 5 * x - 7) / (x ^ 3 - x) = A / x + (B * x + C) / (x ^ 2 - 1) ) :=
by
  intros x hx
  rw [hA, hB, hC]
  sorry

end find_constants_l751_751818


namespace conditional_probabilities_l751_751057

noncomputable theory

def p (e : Prop) (prob : ℝ) := true -- Placeholder for probability function

variables (e f g : Prop)
variables (p_e : ℝ) (p_f : ℝ) (p_g : ℝ) (p_e_and_f : ℝ) (p_e_and_g : ℝ) (p_f_and_g : ℝ)

-- Conditions
def p_e_def : p e 0.25 := p e 0.25
def p_f_def : p f 0.75 := p f 0.75
def p_g_def : p g 0.60 := p g 0.60
def p_e_and_f_def : p (e ∧ f) 0.20 := p (e ∧ f) 0.20
def p_e_and_g_def : p (e ∧ g) 0.15 := p (e ∧ g) 0.15
def p_f_and_g_def : p (f ∧ g) 0.50 := p (f ∧ g) 0.50

-- Theorem statement
theorem conditional_probabilities :
  (∀ p : Prop, p e → p f → p g → 
    ((0.20 / 0.75 = 0.2667) ∧ 
     (0.20 / 0.25 = 0.80) ∧ 
     (0.15 / 0.25 = 0.60) ∧ 
     (0.50 / 0.75 = 0.6667))) :=
by {
  intros,
  sorry
}

end conditional_probabilities_l751_751057


namespace cube_edge_numbers_possible_l751_751569

theorem cube_edge_numbers_possible :
  ∃ (top bottom : Finset ℕ), top.card = 4 ∧ bottom.card = 4 ∧ 
  (top ∪ bottom).card = 8 ∧ 
  ∀ (n : ℕ), n ∈ top ∪ bottom → n ∈ (Finset.range 12).map (+1) ∧ 
  (∏ x in top, x) = (∏ y in bottom, y) :=
by {
  sorry,
}

end cube_edge_numbers_possible_l751_751569


namespace arithmetic_sequence_properties_l751_751958

variables {a : ℕ → ℤ} {S T : ℕ → ℤ}

theorem arithmetic_sequence_properties 
  (h₁ : a 2 = 11)
  (h₂ : S 10 = 40)
  (h₃ : ∀ n, S n = n * a 1 + (n * (n - 1)) / 2 * (a 2 - a 1)) -- Sum of first n terms of arithmetic sequence
  (h₄ : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1)) -- General term formula of arithmetic sequence
  : (∀ n, a n = -2 * n + 15) ∧
    ( (∀ n, 1 ≤ n ∧ n ≤ 7 → T n = -n^2 + 14 * n) ∧ 
      (∀ n, n ≥ 8 → T n = n^2 - 14 * n + 98)) :=
by
sorry

end arithmetic_sequence_properties_l751_751958


namespace parallel_vector_m_perpendicular_vector_m_minimum_kt_l751_751061

-- (I) Prove if a is parallel to b, then m = -sqrt(3)/3
theorem parallel_vector_m (m : ℝ) : 
  let a := (m, -1 : ℝ × ℝ)
  let b := (1/2 : ℝ, real.sqrt 3 / 2 : ℝ) in
  m * (real.sqrt 3 / 2) - (-1) * (1/2) = 0 → 
  m = -real.sqrt 3 / 3 :=
sorry

-- (II) Prove if a is perpendicular to b, then m = sqrt(3)
theorem perpendicular_vector_m (m : ℝ) : 
  let a := (m, -1 : ℝ × ℝ)
  let b := (1/2 : ℝ, real.sqrt 3 / 2 : ℝ) in
  m * (1/2) + (-1) * (real.sqrt 3 / 2) = 0 → 
  m = real.sqrt 3 :=
sorry

-- (III) Prove the minimum value of (k + t^2) / t given additional conditions.
theorem minimum_kt (m : ℝ) :
  let a := (m, -1 : ℝ × ℝ)
  let b := (1/2 : ℝ, real.sqrt 3 / 2 : ℝ) in
  let k t : ℝ := (t^2 - 3) * t / 4, t in
  m * (1/2) + (-1) * (real.sqrt 3 / 2) = 0 →
  (λ t, (k + t^2) / t) (-2) = -7/4 :=
sorry

end parallel_vector_m_perpendicular_vector_m_minimum_kt_l751_751061


namespace train_time_correct_l751_751716

noncomputable def train_crossing_time
  (train_length : ℕ) (train_speed_kmph : ℕ) (bridge_length : ℕ) : ℝ :=
  let total_distance := (train_length + bridge_length : ℕ) in
  let train_speed_mps := (train_speed_kmph : ℝ) * (1000 / 3600) in
  (total_distance : ℝ) / train_speed_mps

theorem train_time_correct :
  train_crossing_time 110 90 132 = 242 / 25 := by
  sorry

end train_time_correct_l751_751716


namespace count_three_digit_multiples_of_24_l751_751900

-- We define that the range of three-digit integers is from 100 to 999
def range_start := 100
def range_end := 999

-- We define that the number we are looking for must be a multiple of both 6 and 8, which has an LCM of 24
def lcm_6_8 := Nat.lcm 6 8

-- We state the problem of counting the number of multiples of 24 within the range 100 to 999
theorem count_three_digit_multiples_of_24 : 
  let multiples := finset.Ico range_start range_end
  multiples.filter (λ n, n % lcm_6_8 = 0).card = 37 :=
by
  sorry

end count_three_digit_multiples_of_24_l751_751900


namespace right_triangle_median_length_l751_751102

theorem right_triangle_median_length
  (D E F : Type*)
  [metric_space D] [metric_space E] [metric_space F]
  (DE DF EF : ℝ)
  (hDE : DE = 15)
  (hDF : DF = 9)
  (hEF : EF = 12)
  (h_right : is_right_triangle D E F) :
  distance_to_midpoint F DE = 7.5 :=
by sorry

end right_triangle_median_length_l751_751102


namespace both_questions_correct_l751_751501

-- Define variables as constants
def nA : ℝ := 0.85  -- 85%
def nB : ℝ := 0.70  -- 70%
def nAB : ℝ := 0.60 -- 60%

theorem both_questions_correct:
  nAB = 0.60 := by
  sorry

end both_questions_correct_l751_751501


namespace no_other_common_term_l751_751395

-- Define the sequences x_n and y_n with their initial conditions and recurrence relations.
def x : ℕ → ℤ
| 0       := 1
| 1       := 1
| (n + 1) := x n + 2 * x (n - 1)

def y : ℕ → ℤ
| 0       := 1
| 1       := 7
| (n + 1) := 2 * y n + 3 * y (n - 1)

-- Define the theorem to prove that x_n and y_n have no common terms other than 1.
theorem no_other_common_term : ∀ n m : ℕ, (x n = y m) → (n = 0 ∧ m = 0) := by
  -- skipping the proof for this theorem as per the instruction
  sorry

end no_other_common_term_l751_751395


namespace proof_problem1_proof_problem2_l751_751785

noncomputable def problem1 := (real.sqrt 16 - real.cbrt 8 + real.sqrt (1 / 9)) = 7 / 3

noncomputable def problem2 := (real.sqrt 4 + real.cbrt (-27) - abs (2 - real.sqrt 3)) = real.sqrt 3 - 3

theorem proof_problem1 : problem1 := by
  sorry

theorem proof_problem2 : problem2 := by
  sorry

end proof_problem1_proof_problem2_l751_751785


namespace angle_bisector_intersects_side_l751_751640

variables {ABCD : Type*} [IsoscelesTrapezoid ABCD]
variables {BC AD : ℝ} {S_ABCD : ℝ}

-- Condition: shorter base
def shorter_base (t : ABCD) : Prop := BC = 4

-- Condition: longer base
def longer_base (t : ABCD) : Prop := AD = 8

-- Condition: area
def trapezoid_area (t : ABCD) : Prop := S_ABCD = 21

-- Statement: angle bisector intersection
theorem angle_bisector_intersects_side (t : ABCD) 
  (h1 : shorter_base t)
  (h2 : longer_base t)
  (h3 : trapezoid_area t) : 
  intersects (angle_bisector_of (_ : angle (vertex A)) (vertex C) (side (A, D))) (side (C, D)) := 
sorry

end angle_bisector_intersects_side_l751_751640


namespace three_term_inequality_l751_751293

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l751_751293


namespace smallest_positive_n_l751_751261

theorem smallest_positive_n : ∃ (n : ℕ), n = 626 ∧ ∀ m : ℕ, m < 626 → ¬ (sqrt m - sqrt (m - 1) < 0.02) := by
  sorry

end smallest_positive_n_l751_751261


namespace sum_three_least_divisors_eq_17_l751_751962

section
open Nat

def tau (n : ℕ) : ℕ := (1 to n).filter (λ d, n % d = 0).length

theorem sum_three_least_divisors_eq_17 :
  let ns := (1 to 100).filter (λ n, tau(n) + tau(n+1) = 8) in
  (ns.head + ns.tail.head + ns.tail.tail.head) = 17 := by
  sorry

end

end sum_three_least_divisors_eq_17_l751_751962


namespace rhombus_condition_to_area_equality_l751_751590

open Set
open Function

variable {V : Type*} [InnerProductSpace ℝ V]

/-- Define vectors representation for the points in parallelogram and inner rhombus -/
variable (A B C D A1 B1 C1 D1 : V)

/-- Define the conditions of the problem -/
def isRhombus (A B C D : V) : Prop := dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A
def isParallel (u v : V) : Prop := ∃ k : ℝ, u = k • v

/-- Area function for quadrilaterals, using a placeholder function. Actual implementation to be done -/
noncomputable def area (quad : V × V × V × V) : ℝ := sorry

/-- The main theorem to be proven -/
theorem rhombus_condition_to_area_equality :
  isRhombus A1 B1 C1 D1 → 
  isParallel (B - A) (B1 - A1) → 
  isParallel (C - B) (C1 - B1) → 
  (isRhombus A B C D ↔ area (A, A1, D1, D) + area (B, C, C1, B1) = area (A, B, B1, A1) + area (C, D, D1, C1)) :=
begin
  intros,
  sorry
end

end rhombus_condition_to_area_equality_l751_751590


namespace infinite_pairs_sum_equality_l751_751178

theorem infinite_pairs_sum_equality :
  ∃∞ (k N : ℕ), 1 ≤ k ∧ 1 ≤ N ∧ (1 + 2 + ... + k) = (k + 1) + (k + 2) + ... + N :=
sorry

end infinite_pairs_sum_equality_l751_751178


namespace sin_450_eq_1_l751_751381

theorem sin_450_eq_1 :
  sin (450 * Real.pi / 180) = 1 :=
by
  sorry

end sin_450_eq_1_l751_751381


namespace ladder_length_l751_751345

-- Given conditions translated to Lean 4 definitions
def bottom_initial (ladder_base_to_wall_initial : ℝ) : Prop := ladder_base_to_wall_initial = 7
def top_slipped (slip_down : ℝ) : Prop := slip_down = 3
def bottom_final (ladder_base_to_wall_final : ℝ) : Prop := ladder_base_to_wall_final = 12.85

-- Prove the length of the ladder
theorem ladder_length :
  ∃ (L : ℝ),
    bottom_initial 7 ∧ top_slipped 3 ∧ bottom_final 12.85 ∧ L ≈ 21.98 := 
by
  -- sorry is a placeholder for the proof
  sorry

end ladder_length_l751_751345


namespace convert_to_polar_coordinates_l751_751791

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt ((x * x) + (y * y))
  let θ :=
    if y ≥ 0 then Real.arctan (y / x)
    else if x < 0 then Real.arctan (y / x) + Real.pi
    else Real.arctan (y / x) + 2 * Real.pi
  in (r, θ)

theorem convert_to_polar_coordinates :
  rectangular_to_polar (-1) (-1) = (Real.sqrt 2, 5 * Real.pi / 4) :=
sorry

end convert_to_polar_coordinates_l751_751791


namespace find_sum_of_integers_l751_751198

theorem find_sum_of_integers (w x y z : ℤ)
  (h1 : w - x + y = 7)
  (h2 : x - y + z = 8)
  (h3 : y - z + w = 4)
  (h4 : z - w + x = 3) : w + x + y + z = 11 :=
by
  sorry

end find_sum_of_integers_l751_751198


namespace area_correct_l751_751095

noncomputable def area_of_rectangle_after_excluding : ℝ :=
let length := 1 - (-8) in
let width := 1 - (-7) in
let area_rectangle := length * width in
let radius := 3 in
let area_circle := Real.pi * (radius ^ 2) in
let base_triangle := 1 - (-6) in
let height_triangle := -4 - (-7) in
let area_triangle := (1 / 2) * base_triangle * height_triangle in
area_rectangle - (area_circle + area_triangle)

def remaining_area_approx_equals : Prop :=
abs (area_of_rectangle_after_excluding - 33.2257) < 0.001

theorem area_correct : remaining_area_approx_equals := by
sorry

end area_correct_l751_751095


namespace find_other_root_l751_751844

theorem find_other_root (m : ℝ) (α : ℝ) :
  (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C m * Polynomial.X - Polynomial.C 10 = 0) →
  (α = -5) →
  ∃ β : ℝ, (α + β = -m) ∧ (α * β = -10) :=
by 
  sorry

end find_other_root_l751_751844


namespace students_count_rental_cost_l751_751352

theorem students_count (k m : ℕ) (n : ℕ) 
  (h1 : n = 35 * k)
  (h2 : n = 55 * (m - 1) + 45) : 
  n = 175 := 
by {
  sorry
}

theorem rental_cost (x y : ℕ) 
  (total_buses : x + y = 4)
  (cost_limit : 35 * x + 55 * y ≤ 1500) : 
  320 * x + 400 * y = 1440 := 
by {
  sorry 
}

end students_count_rental_cost_l751_751352


namespace xyz_inequality_l751_751587

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z + x * y + y * z + z * x = 4) : x + y + z ≥ 3 := 
by
  sorry

end xyz_inequality_l751_751587


namespace student_failed_by_marks_l751_751353

theorem student_failed_by_marks (passing_percent : ℕ) (max_marks : ℕ) (obtained_marks : ℕ) (passing_marks : ℕ) (failed_by : ℕ) : 
  passing_percent = 35 → 
  max_marks = 400 → 
  obtained_marks = 100 → 
  passing_marks = (passing_percent * max_marks) / 100 → 
  failed_by = passing_marks - obtained_marks → 
  failed_by = 40 := 
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end student_failed_by_marks_l751_751353


namespace equal_distances_from_circumcenter_l751_751135

theorem equal_distances_from_circumcenter
  (A B C O P Q K L M : Type)
  [triangle ABC]
  (circumcenter O ABC : is_center_of_circumcircle O ABC)
  (P_on_CA : P ∈ segment CA)
  (Q_on_AB : Q ∈ segment AB)
  (mid_K : K = midpoint B P)
  (mid_L : L = midpoint C Q)
  (mid_M : M = midpoint P Q)
  (circle_Gamma : circle_through_points G K L M)
  (PQ_tangent_Gamma : tangent_at_line PQ G) :
  dist O P = dist O Q :=
sorry

end equal_distances_from_circumcenter_l751_751135


namespace sum_of_eight_numbers_l751_751520

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l751_751520


namespace part1_l751_751774

theorem part1 (f : ℝ → ℝ) (a b: ℝ) (h_cont : Continuous (f'')) (h_prime : f'(a) = 0 ∧ f'(b) = 0) :
  f(b) - f(a) = ∫ x in a..b, (a + b) / 2 - x * (f'' x) := sorry

end part1_l751_751774


namespace moles_of_hydrogen_l751_751826

-- Define the reaction and the stoichiometry from the balanced chemical equation
def balanced_reaction (Fe H₂SO₄ H₂ FeSO₄ : ℕ) : Prop :=
  Fe = H₂SO₄ ∧ H₂ = Fe ∧ FeSO₄ = H₂

-- Define the given quantities
def given_quantities (Fe H₂SO₄ : ℕ) : Prop :=
  Fe = 2 ∧ H₂SO₄ = 2

-- The main statement combining all conditions and proving our target goal
theorem moles_of_hydrogen (Fe H₂SO₄ FeSO₄ H₂ : ℕ) :
  balanced_reaction Fe H₂SO₄ H₂ FeSO₄ ∧ given_quantities Fe H₂SO₄ →
  H₂ = 2 :=
by
  intro h,
  cases h with h1 h2,
  cases h1 with h3 h4,
  exact h4.1.symm ▸ sorry -- Proof skipped

end moles_of_hydrogen_l751_751826


namespace sum_of_odd_numbers_l751_751783

theorem sum_of_odd_numbers (n : ℕ) : (Finset.range (n + 1)).sum (λ k, 2 * k + 1) = (n + 1)^2 :=
  sorry

end sum_of_odd_numbers_l751_751783


namespace simplify_complex_fraction_l751_751628

theorem simplify_complex_fraction : 
  (2 - 2*complex.I) / (3 + 4*complex.I) = -2/25 - (14/25)*complex.I :=
by
  have hI : complex.I^2 = -1, by exact complex.I_mul_I.symm
  sorry

end simplify_complex_fraction_l751_751628


namespace creases_form_ellipse_exterior_l751_751174

noncomputable def ellipse (R a : ℝ) : set (ℝ × ℝ) :=
{ P | let x := P.1, y := P.2 in
  (x - a / 2) ^ 2 / (R / 2) ^ 2 + y ^ 2 / (R / 2) ^ 2 - (a / 2) ^ 2 ≥ 1 }

theorem creases_form_ellipse_exterior (R a : ℝ) (hR : 0 < R) (ha : 0 < a) (haR : a < R) :
  ∀ P : ℝ × ℝ, (∃ α : ℝ, ∃ Q : ℝ × ℝ,
    Q.1 = R * Real.cos α ∧ Q.2 = R * Real.sin α ∧ 
    ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 = (P.1 - a) ^ 2 + P.2 ^ 2)) ->
  P ∈ ellipse R a :=
sorry

end creases_form_ellipse_exterior_l751_751174


namespace at_least_one_failure_probability_l751_751426

noncomputable def probability_at_least_one_failure : ℝ :=
  1 - (1 - 0.2)^5

theorem at_least_one_failure_probability :
  probability_at_least_one_failure ≈ 0.67232 :=
by sorry

end at_least_one_failure_probability_l751_751426


namespace symmetry_center_l751_751081

theorem symmetry_center (a : ℝ) (hx : 0 < a) (hp : (f : ℝ → ℝ) = λ x, sin (a * x) + cos (a * x))
    (hT : 2 * π / a = 1) : ∃ x, f x = 0 ∧ x = - (1 / 8) :=
by
  let f := λ x, sqrt 2 * sin (2 * π * x + π / 4)
  have h_period : (2 * π) / (2 * π) = 1,
  { rw ← hT, simp [hx] },
  use - (1 / 8),
  dsimp [f],
  rw [sin_add, sin_neg, sin_pi_div_four, cos_neg, cos_pi_div_four],
  field_simp [pi_ne_zero],
  sorry

end symmetry_center_l751_751081


namespace vector_triangle_rule_l751_751941

variables (a b : Vec3)

theorem vector_triangle_rule (BC CA : Vec3) (h1 : BC = a) (h2 : CA = b) : 
  ∃ AB : Vec3, AB = -a - b :=
by
  use -a - b
  sorry

end vector_triangle_rule_l751_751941


namespace student_club_assignment_l751_751629

theorem student_club_assignment :
  let students := ["A", "B", "C", "D", "E"]
  let clubs := ["Spring Sunshine Literature Club", "Fitness Club", "Basketball Home", "Go Garden"]
  (∀ s : students, ∃ c : clubs, ∀ s₁ s₂ : students, s₁ ≠ s₂ → s₁ ≠ "A" → c ≠ "Go Garden") →
  ∃ n : ℕ, n = 180 :=
sorry

end student_club_assignment_l751_751629


namespace cube_face_product_l751_751574

open Finset

theorem cube_face_product (numbers : Finset ℕ) (hs : numbers = range (12 + 1)) :
  ∃ top_face bottom_face : Finset ℕ,
    top_face.card = 4 ∧
    bottom_face.card = 4 ∧
    (numbers \ (top_face ∪ bottom_face)).card = 4 ∧
    (∏ x in top_face, x) = (∏ x in bottom_face, x) :=
by
  use {2, 4, 9, 10}
  use {3, 5, 6, 8}
  repeat { split };
  -- Check cardinality conditions
  sorry

end cube_face_product_l751_751574


namespace arithmetic_progression_probability_l751_751847

theorem arithmetic_progression_probability (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  total_outcomes = 6^4 ∧ favorable_outcomes = 3 →
  favorable_outcomes / total_outcomes = 1 / 432 :=
by
  sorry

end arithmetic_progression_probability_l751_751847


namespace min_tokens_in_grid_l751_751698

theorem min_tokens_in_grid : 
  ∀ (G : Type) [fintype G] (grid : G → G → Prop),
  (∀ x y : G, set.Squares4x4 x y grid → ∃ t : ℕ, t ≥ 8) →
  ∃ m : ℕ, m = 4801 :=
by sorry

end min_tokens_in_grid_l751_751698


namespace inequality_proof_l751_751286

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751286


namespace solve_for_c_l751_751189

theorem solve_for_c (c : ℝ) (h : sqrt(4 + sqrt(12 + 6 * c)) + sqrt(6 + sqrt(3 + c)) = 4 + 2 * sqrt(3)) : 
  c ≈ 280.32 := 
sorry

end solve_for_c_l751_751189


namespace find_ABC_sum_l751_751392

open Real

variables (A B C : ℤ)
def g (x : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

noncomputable def hasVerticalAsymptotes (A B C : ℤ) : Prop :=
  ∀ x : ℝ, (x = -1 ∨ x = 2) → (A * x^2 + B * x + C = 0)

noncomputable def conditionPosOnInterval (A B C : ℤ) : Prop :=
  ∀ x : ℝ, x > 4 → g A B C x > 0.5

theorem find_ABC_sum :
  hasVerticalAsymptotes A B C →
  conditionPosOnInterval A B C →
  A + B + C = -4 :=
by
  sorry

end find_ABC_sum_l751_751392


namespace sum_of_eight_numbers_l751_751512

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l751_751512


namespace intersection_points_max_distance_l751_751924

-- Definitions based on given parametric equations of the curve C
def curve (α : ℝ) : ℝ × ℝ := (sqrt 2 * cos α, sin α)

-- Cartesian form of the curve from parametric equations
def is_on_curve (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

-- Cartesian equation derived from the polar equation of the line l
def is_on_line (x y : ℝ) : Prop := y = x

-- Problem 1: Prove the correct intersection points
theorem intersection_points :
  ∃ x y : ℝ, is_on_line x y ∧ is_on_curve x y ∧
    ((x = sqrt 6 / 3 ∧ y = sqrt 6 / 3) ∨ (x = -sqrt 6 / 3 ∧ y = -sqrt 6 / 3)) :=
by sorry

-- Problem 2: Prove the maximum distance
theorem max_distance  :
  ∀ α : ℝ, ∃ (d : ℝ), d = abs (sqrt 2 * cos α - sin α) / sqrt 2 ∧ 
    d ≤ sqrt 6 / 2 :=
by sorry

end intersection_points_max_distance_l751_751924


namespace last_locker_is_2046_l751_751762

def last_opened_locker : ℕ :=
  let lockers := array.mk_array 2048 false
  let toggle_lockers (step : ℕ) (locked : array 2048 bool) : array 2048 bool :=
    array.mk (locked.map_with_index (λ i b, b || (i % step == 0)))
  let open_locker (locked : array 2048 bool) (idx : ℕ) : array 2048 bool :=
    locked.write idx true
  
  -- Initial pass: every third locker, starting from 0
  let lockers := toggle_lockers 3 lockers

  -- Continue the process of opening lockers
  -- This part would involve simulating the complete described process,
  -- which is complex and involves patterns based on modulo calculations.

  -- We would encapsulate the rest of the process in some iterative simulation function
  let simulate_lockers (locked : array 2048 bool) : array 2048 bool := sorry

  let final_lockers := simulate_lockers lockers

  let last_opened := (final_lockers.idx item (λ i b, b)).get_or_else 0

  last_opened

theorem last_locker_is_2046 :
  last_opened_locker = 2046 :=
sorry

end last_locker_is_2046_l751_751762


namespace part1_max_distance_part2_area_of_triangle_l751_751107

def parametric_line_equation (t : ℝ) : ℝ × ℝ :=
  (4 - (real.sqrt 2 / 2) * t, 4 + (real.sqrt 2 / 2) * t)

def polar_curve_C (theta : ℝ) : ℝ :=
  8 * real.sin theta

def line_l (x y : ℝ) : Prop :=
  x + y - 8 = 0

noncomputable def distance_point_to_line (point : ℝ × ℝ) : ℝ :=
  let (x, y) := point
  real.abs (x + y - 8) / real.sqrt 2

def max_distance_pt_to_line (theta : ℝ) : ℝ :=
  distance_point_to_line (0, 4) + 4 -- center (0, 4) of the circle and radius 4 plus the distance of center to the line

theorem part1_max_distance (theta : ℝ) (hθ: 0 ≤ θ ∧ θ ≤ real.pi) :
  ∃ (t : ℝ), parametric_line_equation t = (4, 4) → polar_curve_C theta = 4 → max_distance_pt_to_line theta = 4 + 2 * real.sqrt 2 :=
sorry

def intersection_point_B (x y : ℝ) : Prop :=
  line_l x y ∧ (x, y) ∈ (λ theta, (8 * real.sin theta, 4)) -- point B (4, 4)

def angle_AOB : ℝ :=
  (7 * real.pi) / 12

theorem part2_area_of_triangle (theta_A θ_B : ℝ) :
  ∃ (x y : ℝ), intersection_point_B x y ∧ polar_curve_C θ_A = 4 ∧ polar_curve_C θ_B = 4 * real.sqrt 2 ∧ real.sin angle_AOB = (real.sqrt 2 + real.sqrt 6) / 4 →
  1 / 2 * (4 * 4 * 2) * ((real.sqrt 2 + real.sqrt 6)/4) = 4 + 4 * real.sqrt 3 :=
sorry

end part1_max_distance_part2_area_of_triangle_l751_751107


namespace sum_of_draws_l751_751408

-- Define the conditions as stated in part (a)
variables (A B : ℕ)
variables (num_slips : finset ℕ := (finset.range 51).erase 0)
variables (draws : finset.product num_slips num_slips := num_slips.product num_slips)
variables (is_not_prime (n : ℕ) : Prop := ¬ nat.prime n)

-- Hypotheses based on conditions
variables (alice_cannot_determine_prime : ¬ nat.prime A ∧ ¬ is_not_prime A)
variable (bob_conditions : is_not_prime B ∧ ∃ k : ℕ, k * k = B)
variable (alice_condition : ∃ k : ℕ, k * k = 50 * B + A)

-- Theorem statement proving the correct answer
theorem sum_of_draws (A B : ℕ) :
  alice_cannot_determine_prime ∧ bob_conditions ∧ alice_condition → A + B = 43 :=
by
  sorry

end sum_of_draws_l751_751408


namespace value_of_x_that_makes_sqrt_undefined_l751_751460

theorem value_of_x_that_makes_sqrt_undefined (x : ℕ) (hpos : 0 < x) : (x = 1) ∨ (x = 2) ↔ (x - 3 < 0) := by
  sorry

end value_of_x_that_makes_sqrt_undefined_l751_751460


namespace polygon_diagonals_regions_l751_751116

theorem polygon_diagonals_regions (n : ℕ) (hn : n ≥ 3) :
  let D := n * (n - 3) / 2
  let P := n * (n - 1) * (n - 2) * (n - 3) / 24
  let R := D + P + 1
  R = n * (n - 1) * (n - 2) * (n - 3) / 24 + n * (n - 3) / 2 + 1 :=
by
  sorry

end polygon_diagonals_regions_l751_751116


namespace sum_of_reciprocals_of_roots_l751_751470

noncomputable def poly := (42 : ℝ) * X^3 - (35 : ℝ) * X^2 + (10 : ℝ) * X - 1

theorem sum_of_reciprocals_of_roots (a b c : ℝ)
  (h_roots : IsRoot poly a ∧ IsRoot poly b ∧ IsRoot poly c)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_range : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) :
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 2.875) :=
by 
  sorry

end sum_of_reciprocals_of_roots_l751_751470


namespace circle_shaded_region_perimeter_l751_751541

theorem circle_shaded_region_perimeter
  (O P Q : Type) [MetricSpace O]
  (r : ℝ) (OP OQ : ℝ) (arc_PQ : ℝ)
  (hOP : OP = 8)
  (hOQ : OQ = 8)
  (h_arc_PQ : arc_PQ = 8 * Real.pi) :
  (OP + OQ + arc_PQ = 16 + 8 * Real.pi) :=
by
  sorry

end circle_shaded_region_perimeter_l751_751541


namespace range_of_f_cos_of_angle_diff_l751_751474

-- Definitions for the first part of the problem
def f (x : ℝ) : ℝ := 2 * sin (x + π / 6) * cos x

theorem range_of_f : set.Icc 0 (3 / 2) = set.range f ∩ set.Icc 0 (3 / 2) :=
by sorry

-- Definitions for the second part of the problem
variables (A B C a b c : ℝ)

def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = π ∧ 
  ∀ x, x ∈ {A, B, C} → x > 0 ∧ x < π

def side_lengths (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def f_function (A : ℝ) : ℝ := sin (2 * A + π / 6) + 1 / 2

noncomputable def cos_diff (A B : ℝ) : ℝ := cos A * cos B + sin A * sin B

-- Theorem to prove the second part of the problem
theorem cos_of_angle_diff (A B C a b c : ℝ) 
  (h1 : triangle_ABC A B C a b c)
  (h2 : side_lengths a b c)
  (h3 : A < π / 2)
  (h4 : f_function A = 1)
  (h5 : b = 2)
  (h6 : c = 3) : 
  cos_diff A B = (5 * sqrt 7) / 14 :=
by sorry

end range_of_f_cos_of_angle_diff_l751_751474


namespace three_term_inequality_l751_751298

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l751_751298


namespace repeated_application_of_g_on_2_l751_751144

def g (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem repeated_application_of_g_on_2 :
  g(g(g(g(g(g(2)))))) = 2 :=
by
  sorry

end repeated_application_of_g_on_2_l751_751144


namespace sin_450_eq_1_l751_751388

theorem sin_450_eq_1 : Real.sin (450 * Real.pi / 180) = 1 := by
  -- Using the fact that angle measures can be reduced modulo 2π radians (360 degrees)
  let angle := 450 * Real.pi / 180
  let reduced_angle := angle % (2 * Real.pi)
  have h1 : reduced_angle = Real.pi / 2 := by
    -- 450 degrees is 450 * π / 180 radians, which simplifies to π / 2 radians mod 2π
    calc
      reduced_angle = (450 * Real.pi / 180) % (2 * Real.pi)  : rfl
      ... = (5 * Real.pi / 2) % (2 * Real.pi)                : by simp [mul_div_assoc, Real.pi_div_two]
      ... = Real.pi / 2                                       : by norm_num1
  -- The sine of π / 2 radians is 1
  rw [h1, Real.sin_pi_div_two]
  exact rfl

end sin_450_eq_1_l751_751388


namespace distance_from_F_to_midpoint_l751_751101

-- Definitions of the given problem
variable (DE DF EF : ℝ)
variable (F : {x : ℝ × ℝ // x.1 ^ 2 + x.2 ^ 2 = DF ^ 2})
def is_right_triangle (DE DF EF : ℝ) : Prop := (DE ^ 2 + DF ^ 2 = EF ^ 2)

theorem distance_from_F_to_midpoint (h : is_right_triangle DE DF EF) (hDE : DE = 15) (hDF : DF = 9) (hEF : EF = 12) :
  (DE / 2 - sqrt (((DF ^ 2) / 2)^2 - (DF / 2) ^ 2) = 6) :=
by
  sorry

end distance_from_F_to_midpoint_l751_751101


namespace roots_cubic_polynomial_l751_751149

theorem roots_cubic_polynomial (p q r : ℂ) 
  (h1 : p^3 + p^2 - 2p - 1 = 0)
  (h2 : q^3 + q^2 - 2q - 1 = 0)
  (h3 : r^3 + r^2 - 2r - 1 = 0) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -1 := 
sorry

end roots_cubic_polynomial_l751_751149


namespace restore_original_price_l751_751751

def price_after_increases (p : ℝ) : ℝ :=
  let p1 := p * 1.10
  let p2 := p1 * 1.10
  let p3 := p2 * 1.05
  p3

theorem restore_original_price (p : ℝ) (h : p = 1) : 
  ∃ x : ℝ, x = 22 ∧ (price_after_increases p) * (1 - x / 100) = 1 := 
by 
  sorry

end restore_original_price_l751_751751


namespace area_cosine_curve_l751_751472

theorem area_cosine_curve : 
  (∫ x in (0:ℝ)..(3/2) * real.pi, real.cos x) = 3 := 
by 
  sorry

end area_cosine_curve_l751_751472


namespace maximum_value_of_f_l751_751836

theorem maximum_value_of_f (x : ℝ) (h : x^4 + 36 ≤ 13 * x^2) : 
  ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), (x^4 + 36 ≤ 13 * x^2) → (x^3 - 3 * x ≤ m) :=
sorry

end maximum_value_of_f_l751_751836


namespace part1_part2_l751_751477

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem part1 (h : 1 - a = -1) : a = 2 ∧ 
                                  (∀ x : ℝ, x < Real.log 2 → (Real.exp x - 2) < 0) ∧ 
                                  (∀ x : ℝ, x > Real.log 2 → (Real.exp x - 2) > 0) :=
by
  sorry

theorem part2 (h1 : x1 < Real.log 2) (h2 : x2 > Real.log 2) (h3 : f 2 x1 = f 2 x2) : 
  x1 + x2 < 2 * Real.log 2 :=
by
  sorry

end part1_part2_l751_751477


namespace parabola_shifted_left_and_down_l751_751915

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ :=
  3 * (x - 4) ^ 2 + 3

-- Define the transformation (shift 4 units to the left and 4 units down)
def transformed_parabola (x : ℝ) : ℝ :=
  initial_parabola (x + 4) - 4

-- Prove that after transformation the given parabola becomes y = 3x^2 - 1
theorem parabola_shifted_left_and_down :
  ∀ x : ℝ, transformed_parabola x = 3 * x ^ 2 - 1 := 
by 
  sorry

end parabola_shifted_left_and_down_l751_751915


namespace AB_eq_14_BC_eq_20_BC_AB_diff_half_constant_PQ_when_10_l751_751058

noncomputable def pointA := -24
noncomputable def pointB := -10
noncomputable def pointC := 10

def AB := abs (pointA - pointB)  -- |a - b|
def BC := abs (pointB - pointC)  -- |b - c|
def BC_AB_diff_half (t: ℕ) := (BC - AB) / 2 -- (BC - AB) / 2

def P (t : ℕ) := if t ≤ 14 then pointA + t else pointA + (t - 14)
def Q (t : ℕ) := if t ≤ 14 then pointA else pointA + 3 * (t - 14)
def PQ (t : ℕ) := abs (P t - Q t)

theorem AB_eq_14 : AB = 14 := by sorry

theorem BC_eq_20 : BC = 20 := by sorry

theorem BC_AB_diff_half_constant : BC_AB_diff_half = λ t, 3 := by sorry

theorem PQ_when_10 (t : ℕ) : PQ t = 10 ↔ t = 10 ∨ t = 16 ∨ t = 26 := by sorry

end AB_eq_14_BC_eq_20_BC_AB_diff_half_constant_PQ_when_10_l751_751058


namespace remaining_food_after_trip_calculate_remaining_food_l751_751756

theorem remaining_food_after_trip (initial_food : ℕ)
    (first_day_rate : ℚ)
    (second_day_rate : ℚ)
    (first_day_usage : initial_food * first_day_rate.to_nat)
    (remaining_after_first_day : initial_food - first_day_usage)
    (second_day_usage : remaining_after_first_day * second_day_rate.to_nat)
    (final_remaining_food : remaining_after_first_day - second_day_usage)
    : final_remaining_food = 96 := by
  sorry

# Conditions specific to this problem
def initial_food := 400
def first_day_rate := 2 / 5
def second_day_rate := 3 / 5

-- Calculations
def first_day_usage := initial_food * 2 / 5
def remaining_after_first_day := 400 - first_day_usage
def second_day_usage := remaining_after_first_day * 3 / 5
def final_remaining_food := remaining_after_first_day - second_day_usage

theorem calculate_remaining_food : final_remaining_food = 96 := by
  sorry

(theorem : calculate_remaining_food : final_remaining_food = 96 := by
  calc
    final_remaining_food 
      = (240 - 144) : sorry
      ... = 96 : rfl)

end remaining_food_after_trip_calculate_remaining_food_l751_751756


namespace sqrt_x_minus_3_undefined_l751_751462

theorem sqrt_x_minus_3_undefined (x : ℕ) (h_pos : x > 0) : 
  (x = 1 ∨ x = 2) ↔ real.sqrt (x - 3) = 0 := sorry

end sqrt_x_minus_3_undefined_l751_751462


namespace sqrt_domain_l751_751914

theorem sqrt_domain (x : ℝ) : (∃ y, y = sqrt (x - 2023)) ↔ x ≥ 2023 := 
by sorry

end sqrt_domain_l751_751914


namespace max_property_l751_751153

noncomputable def f : ℚ → ℚ := sorry

axiom f_zero : f 0 = 0
axiom f_pos_of_nonzero : ∀ α : ℚ, α ≠ 0 → f α > 0
axiom f_mul : ∀ α β : ℚ, f (α * β) = f α * f β
axiom f_add : ∀ α β : ℚ, f (α + β) ≤ f α + f β
axiom f_bounded_by_1989 : ∀ m : ℤ, f m ≤ 1989

theorem max_property (α β : ℚ) (h : f α ≠ f β) : f (α + β) = max (f α) (f β) := sorry

end max_property_l751_751153


namespace diophantine_solution_exists_l751_751619

theorem diophantine_solution_exists (D : ℤ) : 
  ∃ (x y z : ℕ), x^2 - D * y^2 = z^2 ∧ ∃ m n : ℕ, m^2 > D * n^2 :=
sorry

end diophantine_solution_exists_l751_751619


namespace number_of_bags_l751_751505

theorem number_of_bags (cookies_per_bag total_cookies : ℕ) (h1 : cookies_per_bag = 41) (h2 : total_cookies = 2173) : total_cookies / cookies_per_bag = 53 :=
by
  rw [h1, h2]
  norm_num

end number_of_bags_l751_751505


namespace minimize_transportation_cost_l751_751718

noncomputable def transportation_minimization
  (m n : ℕ)
  (a : Fin m → ℝ)
  (b : Fin n → ℝ)
  (c : Matrix (Fin m) (Fin n) ℝ)
  (h_balanced : ∑ i, a i = ∑ j, b j) : ℝ :=
  let x : Matrix (Fin m) (Fin n) ℝ := sorry in
  if h_feasible : (∀ i, ∑ j, x i j ≤ a i) ∧ (∀ j, ∑ i, x i j ≥ b j) ∧ (∀ i j, 0 ≤ x i j)
  then ∑ i, ∑ j, c i j * x i j
  else 0    -- This represents the minimum cost when conditions are met, 0 otherwise (eligible for further optimization proofs).

theorem minimize_transportation_cost
  (m n : ℕ)
  (a : Fin m → ℝ)
  (b : Fin n → ℝ)
  (c : Matrix (Fin m) (Fin n) ℝ)
  (h_balanced : ∑ i, a i = ∑ j, b j) :
  ∃ x : Matrix (Fin m) (Fin n) ℝ, 
    (∀ i, ∑ j, x i j ≤ a i) ∧ 
    (∀ j, ∑ i, x i j ≥ b j) ∧ 
    (∀ i j, 0 ≤ x i j) ∧
    (∑ i, ∑ j, c i j * x i j = transportation_minimization m n a b c h_balanced) :=
sorry

end minimize_transportation_cost_l751_751718


namespace minimum_value_of_f_l751_751049

noncomputable def f (x : ℝ) : ℝ := x + (1 / (4 * x)) + 1

theorem minimum_value_of_f :
  (∀ x > 0, f x ≥ 2) ∧ (∃! x > 0, f x = 2) :=
by
  sorry

end minimum_value_of_f_l751_751049


namespace value_of_a_l751_751441

-- Define the sets A and B and the intersection condition
def A (a : ℝ) : Set ℝ := {a ^ 2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a ^ 2 + 1}

theorem value_of_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -1 :=
by {
  -- Insert proof here when ready, using h to show a = -1
  sorry
}

end value_of_a_l751_751441


namespace current_height_of_tree_l751_751359

-- Definitions of conditions
def growth_per_year : ℝ := 0.5
def years : ℕ := 240
def final_height : ℝ := 720

-- The goal is to prove that the current height of the tree is 600 inches
theorem current_height_of_tree :
  final_height - (growth_per_year * years) = 600 := 
sorry

end current_height_of_tree_l751_751359


namespace inequality_inequality_holds_l751_751320

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l751_751320


namespace arithmetic_sequence_properties_l751_751957

variables {a : ℕ → ℤ} {S T : ℕ → ℤ}

theorem arithmetic_sequence_properties 
  (h₁ : a 2 = 11)
  (h₂ : S 10 = 40)
  (h₃ : ∀ n, S n = n * a 1 + (n * (n - 1)) / 2 * (a 2 - a 1)) -- Sum of first n terms of arithmetic sequence
  (h₄ : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1)) -- General term formula of arithmetic sequence
  : (∀ n, a n = -2 * n + 15) ∧
    ( (∀ n, 1 ≤ n ∧ n ≤ 7 → T n = -n^2 + 14 * n) ∧ 
      (∀ n, n ≥ 8 → T n = n^2 - 14 * n + 98)) :=
by
sorry

end arithmetic_sequence_properties_l751_751957


namespace wheel_diameter_proof_l751_751767

def find_wheel_diameter (distance_revolutions : ℝ) (revolutions : ℝ) (π_val : ℝ) : ℝ := 
  distance_revolutions / (revolutions * π_val)

theorem wheel_diameter_proof :
  let distance : ℝ := 1408
  let revolutions : ℝ := 16.014558689717926
  let π_val : ℝ := 3.14159
  find_wheel_diameter distance revolutions π_val ≈ 27.986 :=
by
  let d := find_wheel_diameter 1408 16.014558689717926 3.14159
  sorry

end wheel_diameter_proof_l751_751767


namespace valid_accommodation_count_l751_751669

def Room := Fin 8

def opposite (r : Room) : Room := 
  match r with
  | ⟨i, h⟩ => ⟨(i + 4) % 8, sorry⟩ -- assuming modulo arithmetic to find the opposite room

def adjacent (r1 r2 : Room) : Prop :=
  abs (r1.1 - r2.1) = 1 

def is_valid (occupied : List Room) : Prop :=
  occupied.length = 4 ∧
  ∀ r1 r2, (r1 ∈ occupied ∧ r2 ∈ occupied → ¬adjacent r1 r2 ∧ r2 ≠ opposite r1)

noncomputable def available_rooms : List (List Room) := 
  ( List.permutations [⟨0, sorry⟩, ⟨1, sorry⟩, ⟨2, sorry⟩, ⟨3, sorry⟩, ⟨4, sorry⟩, ⟨5, sorry⟩, ⟨6, sorry⟩, ⟨7, sorry⟩] ).filter is_valid

theorem valid_accommodation_count : available_rooms.length = 120 := 
  sorry

end valid_accommodation_count_l751_751669


namespace beetle_walks_less_percentage_l751_751365

theorem beetle_walks_less_percentage
  (ant_distance_m : ℝ)
  (ant_time_min : ℝ)
  (beetle_speed_kmh : ℝ)
  (beetle_distance_m : ℝ)
  (ant_distance_m = 600)
  (ant_time_min = 12)
  (beetle_speed_kmh = 2.55)
  (beetle_distance_m = beetle_speed_kmh * (ant_time_min / 60) * 1000) :
  ((ant_distance_m - beetle_distance_m) / ant_distance_m) * 100 = 15 :=
by
  sorry

end beetle_walks_less_percentage_l751_751365


namespace inequality_proof_l751_751276

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751276


namespace diameter_of_circle_l751_751256

theorem diameter_of_circle (A : ℝ) (h : A = 100 * Real.pi) : ∃ d : ℝ, d = 20 :=
by
  sorry

end diameter_of_circle_l751_751256


namespace proof_problem_l751_751397

noncomputable def satisfy_conditions (n p : ℕ) : Bool :=
  p.prime ∧ n > 1 ∧ ((p - 1) ^ n + 1) % (n ^ (p - 1)) = 0

theorem proof_problem :
  {np : ℕ × ℕ // let n := np.1 in let p := np.2 in satisfy_conditions n p } =
  {(2, 2), (3, 3)} := 
by sorry

end proof_problem_l751_751397


namespace number_of_points_C_l751_751528

theorem number_of_points_C (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (dist_AB : dist A B = 10) (perimeter_ABC : ∀ C, dist A B + dist A C + dist B C = 50) 
  (area_ABC : ∀ C, abs (C.y - A.y) = 20) : 
  ∃! C, dist A B + dist A C + dist B C = 50 ∧ (abs (C.y - A.y) = 20) :=
sorry

end number_of_points_C_l751_751528


namespace inequality_proof_l751_751283

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751283


namespace projection_not_convex_pentagon_l751_751553

noncomputable def set_of_points : set (ℝ × ℝ × ℝ) :=
  { p | ∃ (x y z : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 
                       0 ≤ y ∧ y ≤ 1 ∧ 
                       0 ≤ z ∧ z ≤ 1 ∧ 
                       p = (x, y, z) }

def is_projection_convex_pentagon (proj : set (ℝ × ℝ)) : Prop := 
  ∃ (vertices : list (ℝ × ℝ)), vertices.length = 5 ∧ 
    (∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices ∧ v2 ∈ vertices ∧ v1 ≠ v2 → 
                        ∀ (λ : ℝ), 0 < λ ∧ λ < 1 → 
                        λ • v1 + (1 - λ) • v2 ∉ vertices) ∧
    (∀ (v : ℝ × ℝ), v ∈ proj → ∃ (λi : ℝ) (i : Fin 5), 0 < λi ∧ 
                                                    λi < 1 ∧ 
                                                    v = λi • (vertices.nth_le i (by sorry)) + 
                                                        (1 - λi) • (vertices.nth_le ((i + 1) % 5) (by sorry)))

theorem projection_not_convex_pentagon (proj : set (ℝ × ℝ)) :
  (∃ (P : set_of_points → set (ℝ × ℝ)), ∀ (p : ℝ × ℝ × ℝ) (_ : p ∈ set_of_points), P p = proj) →
  ¬ is_projection_convex_pentagon proj :=
by
  sorry

end projection_not_convex_pentagon_l751_751553


namespace line_parabola_intersections_l751_751052

theorem line_parabola_intersections (k : ℝ) :
  ((∃ x y, y = k * (x - 2) + 1 ∧ y^2 = 4 * x) ↔ k = 0) ∧
  (¬∃ x₁ x₂, x₁ ≠ x₂ ∧ (k * (x₁ - 2) + 1)^2 = 4 * x₁ ∧ (k * (x₂ - 2) + 1)^2 = 4 * x₂) ∧
  (¬∃ x y, y = k * (x - 2) + 1 ∧ y^2 = 4 * x) :=
by sorry

end line_parabola_intersections_l751_751052


namespace find_x_l751_751600

def star (a b : ℝ) : ℝ :=
  (Real.sqrt (a + b)) / (Real.sqrt (a - b))

theorem find_x :
  (∃ x : ℝ, star x 48 = 3) :=
by {
  use 60,
  unfold star,
  apply congr_arg,
  sorry -- This is where the proof steps would go
}

end find_x_l751_751600


namespace sqrt_x_minus_3_undefined_l751_751461

theorem sqrt_x_minus_3_undefined (x : ℕ) (h_pos : x > 0) : 
  (x = 1 ∨ x = 2) ↔ real.sqrt (x - 3) = 0 := sorry

end sqrt_x_minus_3_undefined_l751_751461


namespace cellular_plan_fee_l751_751747

theorem cellular_plan_fee :
  ∃ F : ℝ, 
  (∀ t : ℝ, t = 2500 → F + (t - 500) * 0.35 = 75 + (t - 1000) * 0.45)
   ↔ F = 50 :=
begin
  sorry
end

end cellular_plan_fee_l751_751747


namespace kaye_more_stamps_l751_751717

theorem kaye_more_stamps (x : ℕ) :
  let kaye_initial := 5 * x
  let alberto_initial := 3 * x
  let kaye_after_gift := kaye_initial - 12
  let alberto_after_gift := alberto_initial + 12
  (kaye_after_gift * 3 = alberto_after_gift * 4) →
  kaye_after_gift = alberto_after_gift + 32 :=
by
  intros x h
  let kaye_initial := 5 * x
  let alberto_initial := 3 * x
  let kaye_after_gift := kaye_initial - 12
  let alberto_after_gift := alberto_initial + 12
  have h1 : 3 * (kaye_initial - 12) = 4 * (alberto_initial + 12), from h
  -- sorry to skip the proof steps
  sorry

end kaye_more_stamps_l751_751717


namespace combined_jail_time_in_weeks_l751_751242

-- Definitions based on conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def daily_arrests_per_city : ℕ := 10
def days_in_jail_pre_trial : ℕ := 4
def sentence_weeks : ℕ := 2
def jail_fraction_of_sentence : ℕ := 1 / 2

-- Calculate the combined weeks of jail time
theorem combined_jail_time_in_weeks : 
  (days_of_protest * daily_arrests_per_city * number_of_cities) * 
  (days_in_jail_pre_trial + (sentence_weeks * 7 * jail_fraction_of_sentence)) / 
  7 = 9900 := 
by sorry

end combined_jail_time_in_weeks_l751_751242


namespace cos_75_degree_l751_751787

theorem cos_75_degree :
  real.cos (75 * real.pi / 180) = (real.sqrt 6 - real.sqrt 2) / 4 := by
  sorry

-- Defining the relevant trigonometric identities as hypotheses
lemma cos_45_degree : real.cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by
  sorry

lemma sin_45_degree : real.sin (45 * real.pi / 180) = real.sqrt 2 / 2 := by
  sorry

lemma cos_30_degree : real.cos (30 * real.pi / 180) = real.sqrt 3 / 2 := by
  sorry

lemma sin_30_degree : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end cos_75_degree_l751_751787


namespace total_fish_is_100_l751_751123

-- Definition of the number of fish in each tank based on conditions
def first_tank : ℕ := 20
def second_tank : ℕ := 2 * first_tank
def third_tank : ℕ := 2 * first_tank

-- Definition of the total number of fish in all three tanks
def total_fish : ℕ := first_tank + second_tank + third_tank

-- Theorem statement to prove that the total number of fish is 100
theorem total_fish_is_100 : total_fish = 100 := 
by
  simp [first_tank, second_ttank, third_tank, total_fish]
  sorry

end total_fish_is_100_l751_751123


namespace common_properties_trapezoid_rhombus_common_properties_triangle_parallelogram_common_properties_rectangle_circle_l751_751420

-- Definitions related to geometric figures
def is_convex_polygon (P : Type) [polygon P] : Prop := sorry
def sum_of_exterior_angles (P : Type) [polygon P] : Real := 360
def line_intersects_boundary_at_two_points (P : Type) (p : Point) (l : Line) [polygon P] : Prop := sorry
def has_central_symmetry (F : Type) [figure F] : Prop := sorry

-- Theorem statements for common properties
theorem common_properties_trapezoid_rhombus (Trapezoid Rhombus : Type) 
  [polygon Trapezoid] [polygon Rhombus] :
  is_convex_polygon Trapezoid ∧ is_convex_polygon Rhombus ∧
  sum_of_exterior_angles Trapezoid = 360 ∧ sum_of_exterior_angles Rhombus = 360 ∧
  ∀ p : Point, ∀ l : Line, line_intersects_boundary_at_two_points Trapezoid p l ∧ 
                          line_intersects_boundary_at_two_points Rhombus p l := sorry

theorem common_properties_triangle_parallelogram (Triangle Parallelogram : Type) 
  [polygon Triangle] [polygon Parallelogram] :
  is_convex_polygon Triangle ∧ is_convex_polygon Parallelogram ∧
  sum_of_exterior_angles Triangle = 360 ∧ sum_of_exterior_angles Parallelogram = 360 ∧
  ∀ p : Point, ∀ l : Line, line_intersects_boundary_at_two_points Triangle p l ∧ 
                          line_intersects_boundary_at_two_points Parallelogram p l := sorry

theorem common_properties_rectangle_circle (Rectangle Circle : Type) 
  [figure Rectangle] [figure Circle] :
  has_central_symmetry Rectangle ∧ has_central_symmetry Circle := sorry

end common_properties_trapezoid_rhombus_common_properties_triangle_parallelogram_common_properties_rectangle_circle_l751_751420


namespace value_of_fraction_l751_751912

variable (m n : ℚ)

theorem value_of_fraction (h₁ : 3 * m + 2 * n = 0) (h₂ : m ≠ 0 ∧ n ≠ 0) :
  (m / n - n / m) = 5 / 6 := 
sorry

end value_of_fraction_l751_751912


namespace num_correct_statements_l751_751547

-- Definition of the point P
def pointP : ℝ × ℝ × ℝ := (1, 2, 3)

-- Definition of the midpoint in 3D Cartesian coordinates
def midpoint (p₁ p₂ : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2, (p₁.3 + p₂.3) / 2)

-- Definition of reflection about the x-axis
def reflection_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, -p.3)

-- Definition of reflection about the origin
def reflection_origin (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, -p.2, -p.3)

-- Definition of reflection about the xOy plane
def reflection_xoy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, -p.3)

-- Lean proof statement
theorem num_correct_statements : 
  let m := midpoint (0, 0, 0) pointP in
  let r_x := reflection_x pointP in
  let r_o := reflection_origin pointP in
  let r_xoy := reflection_xoy pointP in
  (m = (1/2, 1, 3/2) ∧ r_x = (1, -2, -3) ∧ r_o = (-1, -2, -3) ∧ r_xoy = (1, 2, -3)) :=
  true :=
  2
:=
by sorry

end num_correct_statements_l751_751547


namespace inequality_proof_l751_751289

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751289


namespace number_of_children_at_matinee_l751_751837

-- Definitions of constants based on conditions
def children_ticket_price : ℝ := 4.50
def adult_ticket_price : ℝ := 6.75
def total_receipts : ℝ := 405
def additional_children : ℕ := 20

-- Variables for number of adults and children
variable (A C : ℕ)

-- Assertions based on conditions
axiom H1 : C = A + additional_children
axiom H2 : children_ticket_price * (C : ℝ) + adult_ticket_price * (A : ℝ) = total_receipts

-- Theorem statement: Prove that the number of children is 48
theorem number_of_children_at_matinee : C = 48 :=
by
  sorry

end number_of_children_at_matinee_l751_751837


namespace inequality_ge_one_l751_751309

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l751_751309


namespace combined_jail_time_in_weeks_l751_751244

-- Definitions based on conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def daily_arrests_per_city : ℕ := 10
def days_in_jail_pre_trial : ℕ := 4
def sentence_weeks : ℕ := 2
def jail_fraction_of_sentence : ℕ := 1 / 2

-- Calculate the combined weeks of jail time
theorem combined_jail_time_in_weeks : 
  (days_of_protest * daily_arrests_per_city * number_of_cities) * 
  (days_in_jail_pre_trial + (sentence_weeks * 7 * jail_fraction_of_sentence)) / 
  7 = 9900 := 
by sorry

end combined_jail_time_in_weeks_l751_751244


namespace general_term_formula_sum_first_n_terms_l751_751022

-- Define the positive arithmetic geometric sequence {a_n}
def a_seq (n : ℕ) : ℝ := if n = 1 then 1/2 else 2^(n - 2)

-- Define the sequence b_n based on a_n
def b_seq : ℕ → ℝ
| n := real.logb 2 (a_seq n ^ 2) + 4

-- Define the sequence {1/(b_n * b_{n+1})}
def inverse_b_seq (n : ℕ) : ℝ := 1 / (b_seq n * b_seq (n + 1))

-- Define sum of the first n terms of the sequence
def T_n : ℕ → ℝ
| 0 := 0
| (n + 1) := T_n n + inverse_b_seq n

-- Part 1: General term formula for the sequence {a_n}
theorem general_term_formula (n : ℕ) :
  a_seq n = 2^(n - 2) := sorry

-- Part 2: Sum of the first n terms T_n of the sequence {1/(b_n * b_{n+1})}
theorem sum_first_n_terms (n : ℕ) :
  T_n n = n / (4 * (n + 1)) := sorry

end general_term_formula_sum_first_n_terms_l751_751022


namespace arithmetic_expression_result_l751_751780

theorem arithmetic_expression_result :
  (24 / (8 + 2 - 5)) * 7 = 33.6 :=
by
  sorry

end arithmetic_expression_result_l751_751780


namespace combined_resistance_parallel_l751_751325

theorem combined_resistance_parallel (x y r : ℝ) (hx : x = 4) (hy : y = 5)
  (h_combined : 1 / r = 1 / x + 1 / y) : r = 20 / 9 := by
  sorry

end combined_resistance_parallel_l751_751325


namespace no_integer_points_on_circle_l751_751839

theorem no_integer_points_on_circle : 
  ∀ x : ℤ, ¬ ((x - 3)^2 + (x + 1 + 2)^2 ≤ 64) := by
  sorry

end no_integer_points_on_circle_l751_751839


namespace matt_homework_time_other_subjects_l751_751979

theorem matt_homework_time_other_subjects : 
  let total_time := 150
  let math_time := 0.20 * total_time
  let science_time := 0.25 * total_time
  let history_time := 0.10 * total_time
  let english_time := 0.15 * total_time
  let adjusted_history_time := max history_time 20
  let adjusted_total_time := math_time + science_time + adjusted_history_time + english_time
  total_time - adjusted_total_time = 40 :=
by
  let total_time := 150
  let math_time := 0.20 * total_time
  let science_time := 0.25 * total_time
  let history_time := 0.10 * total_time
  let english_time := 0.15 * total_time
  let adjusted_history_time := max history_time 20
  let adjusted_total_time := math_time + science_time + adjusted_history_time + english_time
  have h_math_no_penalty : math_time >= 20 := by sorry
  have h_english_no_penalty : english_time >= 20 := by sorry
  have h_correct_adjustment : adjusted_total_time = 30 + 37.5 + 20 + 22.5 := by sorry
  have h_correct_time : total_time - adjusted_total_time = 40 := by sorry
  exact h_correct_time

end matt_homework_time_other_subjects_l751_751979


namespace evaluate_expression_l751_751810

theorem evaluate_expression (b : ℝ) (hb : b ≠ 0) :
  (1/25) * (b^0) + (1/(25*b))^0 - (125^(-1/3)) - (-25)^(-2/3) = 0.72336 :=
by
  sorry

end evaluate_expression_l751_751810


namespace derangements_of_5_l751_751921

def derangement : Nat → Nat
| 0       := 1
| 1       := 0
| (n+2) := (n+1) * (derangement (n+1) + derangement n)

theorem derangements_of_5 : derangement 5 = 44 := by
  sorry

end derangements_of_5_l751_751921


namespace solve_inequalities_l751_751720

-- Conditions from the problem
def condition1 (x y : ℝ) : Prop := 13 * x ^ 2 - 4 * x * y + 4 * y ^ 2 ≤ 2
def condition2 (x y : ℝ) : Prop := 2 * x - 4 * y ≤ -3

-- Given answers from the solution
def solution_x : ℝ := -1/3
def solution_y : ℝ := 2/3

-- Translate the proof problem in Lean
theorem solve_inequalities : condition1 solution_x solution_y ∧ condition2 solution_x solution_y :=
by
  -- Here you will provide the proof.
  sorry

end solve_inequalities_l751_751720


namespace cube_edge_numbers_possible_l751_751565

theorem cube_edge_numbers_possible :
  ∃ (top bottom : Finset ℕ), top.card = 4 ∧ bottom.card = 4 ∧ 
  (top ∪ bottom).card = 8 ∧ 
  ∀ (n : ℕ), n ∈ top ∪ bottom → n ∈ (Finset.range 12).map (+1) ∧ 
  (∏ x in top, x) = (∏ y in bottom, y) :=
by {
  sorry,
}

end cube_edge_numbers_possible_l751_751565


namespace region_T_area_l751_751181

-- Definitions of rhombus sides, angle, and region
structure Rhombus (A B C D : Type) :=
(side_length : ℝ)
(angle_Q : ℝ)

-- Definition that a given point is closer to Q than to any other points P, R, S
def is_closer_to_Q 
  (P Q R S : Type) (pt : Type) [inhabited pt] : Prop :=
  ∀ (x : pt), distance x Q < distance x P ∧ distance x Q < distance x R ∧ distance x Q < distance x S

-- Region T within the rhombus
def region_T (PQRS : Rhombus Type) := 
  { x : Type // is_closer_to_Q PQRS.P PQRS.Q PQRS.R PQRS.S x }

-- Main theorem
theorem region_T_area (PQRS : Rhombus Type) [PQRS.side_length = 4] [PQRS.angle_Q = 150] : 
  area (region_T PQRS) = (8 * sqrt 3) / 9 := 
sorry

end region_T_area_l751_751181


namespace third_square_is_G_l751_751403

variables {Square : Type} [fintype Square] [decidable_eq Square]

structure PlacementInfo :=
  (size : ℕ)
  (total_squares : ℕ)
  (squares : finset Square) 
  (last_placed : Square)
  (all_but_last_partially_visible : squares ≠ ∅)
  
constant EightSquaresScenario : PlacementInfo

theorem third_square_is_G
  (h1 : EightSquaresScenario.size = 2)
  (h2 : EightSquaresScenario.total_squares = 8)
  (h3 : (EightSquaresScenario.last_placed : Square = E)) :
  ∃ G : Square, G = third_square_in_order EightSquaresScenario :=
sorry

end third_square_is_G_l751_751403


namespace fn_has_two_distinct_real_roots_l751_751949

def f (x : ℝ) : ℝ := x^2 + 2018 * x + 1

def f_iter (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
| 0 := id
| 1 := f
| (k+2) := f ∘ f_iter f (k+1)

theorem fn_has_two_distinct_real_roots (n : ℕ) (hn : 0 < n) : 
  ∃ (x y : ℝ), x ≠ y ∧ f_iter f n x = 0 ∧ f_iter f n y = 0 :=
sorry

end fn_has_two_distinct_real_roots_l751_751949


namespace total_cats_and_kittens_received_l751_751582

theorem total_cats_and_kittens_received 
  (adult_cats : ℕ) 
  (perc_female : ℕ) 
  (frac_litters : ℚ) 
  (kittens_per_litter : ℕ)
  (rescued_cats : ℕ) 
  (total_received : ℕ)
  (h1 : adult_cats = 120)
  (h2 : perc_female = 60)
  (h3 : frac_litters = 2/3)
  (h4 : kittens_per_litter = 3)
  (h5 : rescued_cats = 30)
  (h6 : total_received = 294) :
  adult_cats + rescued_cats + (frac_litters * (perc_female * adult_cats / 100) * kittens_per_litter) = total_received := 
sorry

end total_cats_and_kittens_received_l751_751582


namespace probability_product_divisible_by_3_l751_751440

theorem probability_product_divisible_by_3 :
  let outcomes := 8 * 8 in
  let favorable_outcomes := 2 * 6 + 6 * 2 + 2 * 2 in
  favorable_outcomes / outcomes = 7 / 16 :=
by
  let outcomes := 8 * 8
  let favorable_outcomes := 2 * 6 + 6 * 2 + 2 * 2
  have h : favorable_outcomes / outcomes = 28 / 64, by sorry
  show favorable_outcomes / outcomes = 7 / 16, by 
    rw [h]
    norm_num

end probability_product_divisible_by_3_l751_751440


namespace three_term_inequality_l751_751297

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l751_751297


namespace value_of_b_l751_751249

theorem value_of_b (a b : ℕ) (r : ℝ) (h₁ : a = 2020) (h₂ : r = a / b) (h₃ : r = 0.5) : b = 4040 := 
by
  -- Hint: The proof takes steps to transform the conditions using basic algebraic manipulations.
  sorry

end value_of_b_l751_751249


namespace parabola_focus_distance_l751_751872

noncomputable def parabolic_distance (x y : ℝ) : ℝ :=
  x + x / 2

theorem parabola_focus_distance : 
  (∃ y : ℝ, (1 : ℝ) = (1 / 2) * y^2) → 
  parabolic_distance 1 y = 3 / 2 :=
by 
  intros hy
  obtain ⟨y, hy⟩ := hy
  unfold parabolic_distance
  have hx : 1 = (1 / 2) * y^2 := hy
  sorry

end parabola_focus_distance_l751_751872


namespace exists_inf_pairs_iff_c_le_2_l751_751817

theorem exists_inf_pairs_iff_c_le_2 (c : ℝ) (hc : 0 < c) :
  (∃ᶠ (n m : ℕ) in at_top, n ≥ m + c * real.sqrt (m - 1) + 1 ∧
    ∀ k, n ≤ k ∧ k ≤ 2 * n - m → ¬∃ i, i^2 = k) ↔ c ≤ 2 :=
sorry

end exists_inf_pairs_iff_c_le_2_l751_751817


namespace inequality_inequality_holds_l751_751322

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l751_751322


namespace odd_number_diff_of_squares_l751_751627

theorem odd_number_diff_of_squares (k : ℕ) : ∃ n : ℕ, k = (n+1)^2 - n^2 ↔ ∃ m : ℕ, k = 2 * m + 1 := 
by 
  sorry

end odd_number_diff_of_squares_l751_751627


namespace cyclic_sum_inequality_l751_751975

theorem cyclic_sum_inequality (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) (h_sum : (∑ i, a i) = 1) :
  (∑ i, a i * a ((i + 1) % n)) * (∑ i, a i / (a ((i + 1) % n) ^ 2 + a ((i + 1) % n))) ≥ n / (n + 1) := by
  
  sorry

end cyclic_sum_inequality_l751_751975


namespace cube_face_product_l751_751570

open Finset

theorem cube_face_product (numbers : Finset ℕ) (hs : numbers = range (12 + 1)) :
  ∃ top_face bottom_face : Finset ℕ,
    top_face.card = 4 ∧
    bottom_face.card = 4 ∧
    (numbers \ (top_face ∪ bottom_face)).card = 4 ∧
    (∏ x in top_face, x) = (∏ x in bottom_face, x) :=
by
  use {2, 4, 9, 10}
  use {3, 5, 6, 8}
  repeat { split };
  -- Check cardinality conditions
  sorry

end cube_face_product_l751_751570


namespace min_value_problem_l751_751862

theorem min_value_problem 
  (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1) : 
  ∃ (min_val : ℝ), min_val = 2 * x + 3 * y^2 ∧ min_val = 8 / 9 :=
by
  sorry

end min_value_problem_l751_751862


namespace f_increasing_inequality_solution_l751_751965

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ (m n : ℝ), f(m + n) = f(m) + f(n) - 1
axiom h2 : ∀ (x : ℝ), x > 0 → f(x) > 1
axiom h3 : f 3 = 4

theorem f_increasing : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2 := sorry

theorem inequality_solution : {a : ℝ | f (a^2 + a - 5) < 2} = {a : ℝ | -3 < a ∧ a < 2} := sorry

end f_increasing_inequality_solution_l751_751965


namespace sum_of_eight_numbers_l751_751513

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l751_751513


namespace solution_set_inequality_l751_751207

noncomputable def f : ℝ → ℝ := sorry 

theorem solution_set_inequality (f : ℝ → ℝ) (h_deriv: ∀ x, deriv f x > 2) (h_value: f (-3) = 1) : 
  {x : ℝ | f x < 2 * x + 7} = set.Iio (-3) :=
by
  sorry

end solution_set_inequality_l751_751207


namespace tan_cot_identity_l751_751015

open Real

theorem tan_cot_identity (θ φ : ℝ) (h : (tan θ) ^ 4 / (tan φ) ^ 2 + (cot θ) ^ 4 / (cot φ) ^ 2 = 2) :
  (tan φ) ^ 4 / (tan θ) ^ 2 + (cot φ) ^ 4 / (cot θ) ^ 2 = 2 :=
sorry

end tan_cot_identity_l751_751015


namespace solution_set_inequality_l751_751003

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) : 
  (x - 1) / x > 1 → x < 0 := 
by 
  sorry

end solution_set_inequality_l751_751003


namespace red_balls_estimation_l751_751536

noncomputable def numberOfRedBalls (x : ℕ) : ℝ := x / (x + 3)

theorem red_balls_estimation {x : ℕ} (h : numberOfRedBalls x = 0.85) : x = 17 :=
by
  sorry

end red_balls_estimation_l751_751536


namespace base_number_is_five_l751_751086

variable (a x y : Real)

theorem base_number_is_five (h1 : xy = 1) (h2 : (a ^ (x + y) ^ 2) / (a ^ (x - y) ^ 2) = 625) : a = 5 := 
sorry

end base_number_is_five_l751_751086


namespace triangle_area_ABD_l751_751115

-- Define the conditions of triangle ABC
variables {A B C D : Type} [inhabited A]
variables (AB AC : ℝ)
variable (is_midpoint_D : ∃ D, (AC / 2 = D) ∧ D = midpoint A C)
noncomputable def area_of_triangle (AB AD : ℝ) : ℝ := (1 / 2) * AB * AD

-- The main theorem to be proved
theorem triangle_area_ABD
  (H1 : (AB : ℝ) = 90)
  (H2 : (AC : ℝ) = 150)
  (H3 : ∃ D, is_midpoint_D) :
  area_of_triangle 90 75 = 3375 :=
by sorry

end triangle_area_ABD_l751_751115


namespace jail_time_weeks_l751_751239

theorem jail_time_weeks (days_protest : ℕ) (cities : ℕ) (arrests_per_day : ℕ)
  (days_pre_trial : ℕ) (half_week_sentence_days : ℕ) :
  days_protest = 30 →
  cities = 21 →
  arrests_per_day = 10 →
  days_pre_trial = 4 →
  half_week_sentence_days = 7 →
  (21 * 30 * 10 * (4 + 7)) / 7 = 9900 :=
by
  intros h_days_protest h_cities h_arrests_per_day h_days_pre_trial h_half_week_sentence_days
  rw [h_days_protest, h_cities, h_arrests_per_day, h_days_pre_trial, h_half_week_sentence_days]
  exact sorry

end jail_time_weeks_l751_751239


namespace number_of_planes_l751_751490

-- Define the concept of three lines in space, where one line intersects the other two lines
theorem number_of_planes (l1 l2 l3 : Line) (hl12 : intersects l1 l2) (hl13 : intersects l1 l3) :
    (∃ l : list Plane, l.length = 1 ∨ l.length = 2 ∨ l.length = 3) := by
  sorry

end number_of_planes_l751_751490


namespace f_monotonic_intervals_min_value_f_range_a_l751_751848

noncomputable def f (x : ℝ) := x * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := ∫ t in 0..x, 3 * t^2 + 2 * a * t - 1

theorem f_monotonic_intervals :
  (∀ x ∈ set.Icc (0 : ℝ) (1 / Real.exp 1), f' x < 0) ∧
  (∀ x ∈ set.Icc (1 / Real.exp 1) (Real.exp 1), f' x > 0) := sorry

theorem min_value_f (t : ℝ) (h : t > 0) :
  ∃ c, ∀ x ∈ set.Icc t (t + 2), f x ≥ c ∧ (x = t → f x = c) := sorry

theorem range_a (a : ℝ) :
  (∀ x, 0 < x → 2 * f x ≤ g a x + 2) → a ≥ -2 := sorry

end f_monotonic_intervals_min_value_f_range_a_l751_751848


namespace find_x_eq_zero_l751_751814

theorem find_x_eq_zero (x : ℝ) : 3^4 * 3^x = 81 → x = 0 := by
  sorry

end find_x_eq_zero_l751_751814


namespace selection_and_arrangement_l751_751624

-- Defining the problem conditions
def volunteers : Nat := 5
def roles : Nat := 4
def A_excluded_role : String := "music_composer"
def total_methods : Nat := 96

theorem selection_and_arrangement (h1 : volunteers = 5) (h2 : roles = 4) (h3 : A_excluded_role = "music_composer") :
  total_methods = 96 :=
by
  sorry

end selection_and_arrangement_l751_751624


namespace triangle_inequality_l751_751151

variables {A B C P D E F : Type} -- Variables representing points in the plane.
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
variables (PD PE PF PA PB PC : ℝ) -- Distances corresponding to the points.

-- Condition stating P lies inside or on the boundary of triangle ABC
axiom P_in_triangle_ABC : ∀ (A B C P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P], 
  (PD > 0 ∧ PE > 0 ∧ PF > 0 ∧ PA > 0 ∧ PB > 0 ∧ PC > 0)

-- Objective statement to prove
theorem triangle_inequality (PD PE PF PA PB PC : ℝ) 
  (h1 : PA ≥ 0) 
  (h2 : PB ≥ 0) 
  (h3 : PC ≥ 0) 
  (h4 : PD ≥ 0) 
  (h5 : PE ≥ 0) 
  (h6 : PF ≥ 0) :
  PA + PB + PC ≥ 2 * (PD + PE + PF) := 
sorry -- Proof to be provided later.

end triangle_inequality_l751_751151


namespace close_functions_on_interval_l751_751012

noncomputable def m (x : ℝ) : ℝ := x^2 - 3 * x + 4
noncomputable def n (x : ℝ) : ℝ := 2 * x - 3

theorem close_functions_on_interval :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → |m x - n x| ≤ 1) :=
by
  intro x
  intro hx
  have h_diff : m x - n x = x^2 - 5 * x + 7 :=
    calc
      m x - n x = (x^2 - 3 * x + 4) - (2 * x - 3) : by simp [m, n]
      ... = x^2 - 5 * x + 7 : by ring
  rw h_diff
  sorry

end close_functions_on_interval_l751_751012


namespace average_speed_is_correct_l751_751637

def speed_swim := 1 -- mile per hour
def speed_cycle := 14 -- miles per hour
def speed_run := 9 -- miles per hour

def distance_swim := 0.9 -- miles
def distance_cycle := 25 -- miles
def distance_run := 6.2 -- miles

def total_distance := distance_swim + distance_cycle + distance_run
def total_time := (distance_swim / speed_swim) + (distance_cycle / speed_cycle) + (distance_run / speed_run)

noncomputable def average_speed := total_distance / total_time

theorem average_speed_is_correct : abs (average_speed - 9.51) < 0.01 :=
by
  sorry

end average_speed_is_correct_l751_751637


namespace total_fish_is_100_l751_751124

-- Definition of the number of fish in each tank based on conditions
def first_tank : ℕ := 20
def second_tank : ℕ := 2 * first_tank
def third_tank : ℕ := 2 * first_tank

-- Definition of the total number of fish in all three tanks
def total_fish : ℕ := first_tank + second_tank + third_tank

-- Theorem statement to prove that the total number of fish is 100
theorem total_fish_is_100 : total_fish = 100 := 
by
  simp [first_tank, second_ttank, third_tank, total_fish]
  sorry

end total_fish_is_100_l751_751124


namespace free_fall_time_l751_751656

theorem free_fall_time (h : ℝ) (t : ℝ) (h_eq : h = 4.9 * t^2) (h_val : h = 490) : t = 10 :=
by
  sorry

end free_fall_time_l751_751656


namespace sum_of_numbers_l751_751508

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l751_751508


namespace cube_edge_numbers_equal_top_bottom_l751_751560

theorem cube_edge_numbers_equal_top_bottom (
  numbers : List ℕ,
  h : numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) :
  ∃ (top bottom : List ℕ),
    (∀ x, x ∈ top → x ∈ numbers) ∧
    (∀ x, x ∈ bottom → x ∈ numbers) ∧
    (top ≠ bottom) ∧
    (top.length = 4) ∧ 
    (bottom.length = 4) ∧ 
    (top.product = bottom.product) :=
begin
  sorry
end

end cube_edge_numbers_equal_top_bottom_l751_751560


namespace part_I_part_II_l751_751884

open Real

-- Definition of the piecewise function f(x)
def f (x : ℝ) : ℝ :=
  if x ≥ 2 then 4 * x - 6
  else if x > 4 / 3 then 2 * x - 2
  else -4 * x + 6

-- Part I: Prove the solution set for the inequality f(x) > 5x
theorem part_I : ∀ x : ℝ, f x > 5 * x ↔ x < 2 / 3 := 
by
  intro x,
  sorry

-- Part II: Prove that a² + b² ≥ 4/13 given 2a + 3b = 2
theorem part_II (a b : ℝ) (m := 2 / 3):
  2 * a + 3 * b = 3 * m → a^2 + b^2 ≥ 4 / 13 :=
by
  intro h,
  sorry

end part_I_part_II_l751_751884


namespace solve_cubic_fraction_l751_751188

noncomputable def problem_statement (x : ℝ) :=
  (x = (-(3:ℝ) + Real.sqrt 13) / 4) ∨ (x = (-(3:ℝ) - Real.sqrt 13) / 4)

theorem solve_cubic_fraction (x : ℝ) (h : (x^3 + 2*x^2 + 3*x + 5) / (x + 2) = x + 4) : 
  problem_statement x :=
by
  sorry

end solve_cubic_fraction_l751_751188


namespace part1_part2_l751_751045

def f (x a : ℝ) : ℝ := |x + a - 1| + |x - 2 * a|

-- Define the first part of the problem
theorem part1 (a : ℝ) (h : f 1 a < 3) : -2/3 < a ∧ a < 4/3 :=
sorry

-- Define the second part of the problem
theorem part2 (a x : ℝ) (h1 : a ≥ 1) : f x a ≥ 2 :=
sorry

end part1_part2_l751_751045


namespace coeff_x4_in_expansion_l751_751044

theorem coeff_x4_in_expansion :
  (let expr := (1 - 1 / x) * (1 + x) ^ 7 in 
  coeff_of_term expr 4 = 14) :=
sorry

end coeff_x4_in_expansion_l751_751044


namespace rhombus_area_correct_l751_751014

-- Define the coordinates for the vertices of the rhombus
structure Point where
  x : Real
  y : Real

noncomputable def rhombus := {A := Point.mk 1 2, B := Point.mk 10 2,
                              C := Point.mk 6 2, D := Point.mk 3.5 6}

-- Function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point) : Real :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Lengths of diagonals
noncomputable def d1 := distance rhombus.A rhombus.B
noncomputable def d2 := distance rhombus.C rhombus.D

-- Area of the rhombus
noncomputable def area (d1 d2 : Real) := (d1 * d2) / 2

def rhombus_area : Real := 21.24

theorem rhombus_area_correct : area d1 d2 = rhombus_area := by
  -- Here we skip the actual proof calculation
  sorry

end rhombus_area_correct_l751_751014


namespace incircle_radius_l751_751681

-- Definitions based on conditions
variables {D E F : Type} [EuclideanGeometry] [Triangle DEF]
variables (right_angle_F : ∠D F E = 90°) (angle_D : ∠D = 45°) (side_DF : length D F = 8)

-- Theorem statement
theorem incircle_radius (h : right_angle_F) (h' : angle_D) (h'' : side_DF) : 
  incircle_radius DEF = 4 - 2 * Real.sqrt 2 := 
sorry

end incircle_radius_l751_751681


namespace hyperbola_with_foci_condition_l751_751908

theorem hyperbola_with_foci_condition (k : ℝ) :
  ( ∀ x y : ℝ, (x^2 / (k + 3)) + (y^2 / (k + 2)) = 1 → ∀ x y : ℝ, (x^2 / (k + 3)) + (y^2 / (k + 2)) = 1 ∧ (k + 3 > 0 ∧ k + 2 < 0) ) ↔ (-3 < k ∧ k < -2) :=
sorry

end hyperbola_with_foci_condition_l751_751908


namespace volume_of_cone_example_l751_751658

noncomputable def volume_of_cone_given_slant_height_and_height (l h : ℝ) : ℝ :=
  let r := Math.sqrt (l^2 - h^2)
  (1 / 3) * Math.pi * r^2 * h

theorem volume_of_cone_example : 
  volume_of_cone_given_slant_height_and_height 15 9 = 432 * Math.pi := 
by
  sorry

end volume_of_cone_example_l751_751658


namespace find_y_l751_751967

theorem find_y (a b y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : y > 0)
  (h4 : (2 * a)^(4 * b) = a^b * y^(3 * b)) : y = 2^(4 / 3) * a :=
by
  sorry

end find_y_l751_751967


namespace greatest_int_with_gcd_10_less_than_100_l751_751257

theorem greatest_int_with_gcd_10_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ gcd n 30 = 10 ∧ ∀ m : ℕ, m < 100 → gcd m 30 = 10 → m ≤ n := by
  sorry

end greatest_int_with_gcd_10_less_than_100_l751_751257


namespace cube_edge_numbers_possible_l751_751568

theorem cube_edge_numbers_possible :
  ∃ (top bottom : Finset ℕ), top.card = 4 ∧ bottom.card = 4 ∧ 
  (top ∪ bottom).card = 8 ∧ 
  ∀ (n : ℕ), n ∈ top ∪ bottom → n ∈ (Finset.range 12).map (+1) ∧ 
  (∏ x in top, x) = (∏ y in bottom, y) :=
by {
  sorry,
}

end cube_edge_numbers_possible_l751_751568


namespace product_sum_of_roots_l751_751603

theorem product_sum_of_roots (p q r : ℂ)
  (h_eq : ∀ x : ℂ, (2 : ℂ) * x^3 + (1 : ℂ) * x^2 + (-7 : ℂ) * x + (2 : ℂ) = 0 → (x = p ∨ x = q ∨ x = r)) 
  : p * q + q * r + r * p = -7 / 2 := 
sorry

end product_sum_of_roots_l751_751603


namespace inequality_inequality_holds_l751_751316

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l751_751316


namespace correct_electron_transfer_l751_751180

-- Define the reaction
def reaction := "Zn + 2HNO_3 + NH_4NO_3 → N_2 ↑ + 3H_2O + Zn(NO_3)_2"

-- Define valence changes in the reaction
def valence_changes :=
  [("Zn", 0, +2), ("N_HNO_3", +5, 0), ("N_NH_4NO_3", -3, 0)]

-- Define the electron transfer calculation for producing 1 mol of N2
def electron_transfers (valence_changes : List (String × Int × Int)) : ℕ :=
  valence_changes.foldl (fun acc (_, from, to) => acc + (to - from).nat_abs) 0

-- Calculate the total number of electrons transferred when producing 1 mol of N2
def total_electron_transfers := electron_transfers valence_changes

-- Define the correct number of electrons transferred per mole
def correct_electrons_per_mol_n2 := 5
def avogadro_number := 6.02214076e23 -- Placeholder for Avogadro's number

-- The main theorem to be proven in Lean
theorem correct_electron_transfer :
  total_electron_transfers = correct_electrons_per_mol_n2 :=
by
  -- Sorry to skip the proof
  sorry

end correct_electron_transfer_l751_751180


namespace ratio_of_thermometers_to_hotwater_bottles_l751_751689

theorem ratio_of_thermometers_to_hotwater_bottles (T H : ℕ) (thermometer_price hotwater_bottle_price total_sales : ℕ) 
  (h1 : thermometer_price = 2) (h2 : hotwater_bottle_price = 6) (h3 : total_sales = 1200) (h4 : H = 60) 
  (h5 : total_sales = thermometer_price * T + hotwater_bottle_price * H) : 
  T / H = 7 :=
by
  sorry

end ratio_of_thermometers_to_hotwater_bottles_l751_751689


namespace maximum_height_reached_by_ball_l751_751731

def height (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

theorem maximum_height_reached_by_ball : ∃ t : ℝ, height t = 45 :=
sorry

end maximum_height_reached_by_ball_l751_751731


namespace find_side_c_in_triangle_l751_751939

theorem find_side_c_in_triangle :
  ∀ (a b c : ℝ) (A : ℝ),
    a = Real.sqrt 2 →
    b = Real.sqrt 6 →
    A = (Real.pi / 6) →
    c = 2 * Real.sqrt 2 := 
begin
  -- Placeholder for the proof
  sorry
end

end find_side_c_in_triangle_l751_751939


namespace no_n_for_given_sum_of_remainders_l751_751970

theorem no_n_for_given_sum_of_remainders :
  (∀ i j : ℕ, (1 ≤ i ∧ i ≤ 10) → (1 ≤ j ∧ j ≤ 10) → i ≠ j → (a i ≠ a j)) → -- distinctness condition
  (∀ i : ℕ, (1 ≤ i ∧ i ≤ 10) → (a i ≥ 3)) →                          -- each element is at least 3
  (∑ i in (finset.range 10).map (embedding.subtype _), a i = 678) →   -- sum condition
  ¬ (∃ n : ℕ, (∑ i in (finset.range 10).map (embedding.subtype _), n % (a i) + 
                   ∑ j in (finset.range 10).map (embedding.subtype _), n % (2 * a j)) = 2012) := -- sum of remainders condition
sorry

end no_n_for_given_sum_of_remainders_l751_751970


namespace OK_perp_AK_l751_751918

noncomputable def point (α : Type _) : Type _ := α 
variables {α : Type _} [ordered_ring α]

structure triangle (α : Type _) :=
(A B C : point α)

structure midpoint (α : Type _) (A B : point α) :=
(P : point α)

structure perpendicular (α : Type _) (P l : point α) :=
(meet : point α)

variables (A B C D E F M N O K : point α)
variables (T : triangle α)
variables (D_mid : midpoint α (T.B T.C).P D)
variables (E_mid : midpoint α (T.C T.A).P E)
variables (F_mid : midpoint α (T.A T.B).P F)
variables (M_perp : perpendicular α E T.C)
variables (N_perp : perpendicular α F T.A)
variables (O_int : point α)
variables (K_int : point α)

theorem OK_perp_AK :
  (K = by classical.some ((λ x, x ∈ [EM ∩ FN])) ∧ 
   (O = by classical.some ((λ x, x ∈ [CM ∩ BN])))) → 
  ∃ O, ∃ K, perpendicular α O K T.A
:= sorry

end OK_perp_AK_l751_751918


namespace cube_edge_numbers_equal_top_bottom_l751_751561

theorem cube_edge_numbers_equal_top_bottom (
  numbers : List ℕ,
  h : numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) :
  ∃ (top bottom : List ℕ),
    (∀ x, x ∈ top → x ∈ numbers) ∧
    (∀ x, x ∈ bottom → x ∈ numbers) ∧
    (top ≠ bottom) ∧
    (top.length = 4) ∧ 
    (bottom.length = 4) ∧ 
    (top.product = bottom.product) :=
begin
  sorry
end

end cube_edge_numbers_equal_top_bottom_l751_751561


namespace parallel_vectors_product_l751_751141

theorem parallel_vectors_product {x z : ℝ} 
  (hx : ∃ λ : ℝ, (x, 4, 3) = (λ * 3, λ * 2, λ * z)) : 
  x * z = 9 :=
by {
  sorry
}

end parallel_vectors_product_l751_751141


namespace mary_shirts_left_l751_751166

theorem mary_shirts_left :
  ∀ (initial_blue : ℕ) (initial_brown : ℕ),
  initial_blue = 26 →
  initial_brown = 36 →
  let given_away_blue := initial_blue / 2
  let given_away_brown := initial_brown / 3
  let left_blue := initial_blue - given_away_blue
  let left_brown := initial_brown - given_away_brown
  left_blue + left_brown = 37 :=
by
  intros initial_blue initial_brown h_initial_blue h_initial_brown
  dsimp
  rw [h_initial_blue, h_initial_brown]
  sorry

end mary_shirts_left_l751_751166


namespace roots_equivalence_l751_751139

open Polynomial

variables {p q α β γ δ : ℝ}

-- Conditions:
-- (1) α and β are roots of x^2 + px - 2 = 0
-- (2) γ and δ are roots of x^2 + qx - 2 = 0
theorem roots_equivalence 
  (h1 : (X^2 + C p * X - C 2).is_roots α β) 
  (h2 : (X^2 + C q * X - C 2).is_roots γ δ) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -2 * (q^2 - p^2) :=
sorry

end roots_equivalence_l751_751139


namespace placement_possible_l751_751579

def can_place_numbers : Prop :=
  ∃ (top bottom : Fin 4 → ℕ), 
    (∀ i, (top i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (∀ i, (bottom i) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) ∧
    (List.product (List.ofFn top) = List.product (List.ofFn bottom))

theorem placement_possible : can_place_numbers :=
sorry

end placement_possible_l751_751579


namespace election_winning_percentage_l751_751531

noncomputable def requiredWinningPercentage : ℕ → ℕ → ℕ → ℝ :=
  λ totalVotes percentReceived moreVotesNeeded =>
    let votesReceived := (percentReceived * totalVotes) / 100
    let totalWinningVotes := votesReceived + moreVotesNeeded
    (totalWinningVotes.toNat : ℝ) / (totalVotes.toNat : ℝ) * 100

theorem election_winning_percentage (totalVotes : ℕ) (percentReceived : ℕ) (moreVotesNeeded : ℕ) :
  totalVotes = 6000 →
  percentReceived = 0.5 →
  moreVotesNeeded = 3000 →
  requiredWinningPercentage totalVotes percentReceived moreVotesNeeded = 50.5 := by
  intro h1 h2 h3
  sorry

end election_winning_percentage_l751_751531


namespace total_cups_of_ingredients_l751_751653

-- Given that the ratio of butter:flour:sugar is 1:6:4,
-- prove that using 8 cups of sugar results in a total of 22 cups of butter, flour, and sugar.

theorem total_cups_of_ingredients (b f s : ℕ) (h : b : f : s = 1 : 6 : 4) (h_sugar : s = 8) : 
  b + f + s = 22 :=
sorry

end total_cups_of_ingredients_l751_751653


namespace proposition_judgement_l751_751270

theorem proposition_judgement (p q : Prop) (a b c x : ℝ) :
  (¬ (p ∨ q) → (¬ p ∧ ¬ q)) ∧
  (¬ (a > b → a * c^2 > b * c^2)) ∧
  (¬ (∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)) ∧
  ((x^2 - 3*x + 2 = 0) → (x = 2)) =
  false := sorry

end proposition_judgement_l751_751270


namespace parabola_midpoint_length_l751_751078

variable {xA xB : ℝ}

/-- Given a line intersecting the parabola y^2 = 4x at points M and N,
     and the x-coordinate of the midpoint of the segment MN is 3,
     the length of the segment MN is 8. -/
theorem parabola_midpoint_length 
  (h1 : (3:ℝ) = (xA + xB) / 2)
  : sqrt ((xA + xB + 2) * 4) = 8 :=
sorry

end parabola_midpoint_length_l751_751078


namespace area_of_polar_curve_l751_751781

-- Define the polar equation as a function
def polar_eq (φ : ℝ) : ℝ := Real.cos φ + Real.sin φ

-- State the theorem to calculate the area under the curve defined by the polar equation
theorem area_of_polar_curve : 
  ((∫ φ in (-Real.pi / 4)..(3 * Real.pi / 4), 1 / 2 * (polar_eq φ)^2) = Real.pi / 2) :=
by
  sorry

end area_of_polar_curve_l751_751781


namespace student_answers_all_correctly_l751_751212

/-- 
The exam tickets have 2 theoretical questions and 1 problem each. There are 28 tickets. 
A student is prepared for 50 theoretical questions out of 56 and 22 problems out of 28.
The probability that by drawing a ticket at random, and the student answers all questions 
correctly is 0.625.
-/
theorem student_answers_all_correctly :
  let total_theoretical := 56
  let total_problems := 28
  let prepared_theoretical := 50
  let prepared_problems := 22
  let p_correct_theoretical := (prepared_theoretical * (prepared_theoretical - 1)) / (total_theoretical * (total_theoretical - 1))
  let p_correct_problem := prepared_problems / total_problems
  let combined_probability := p_correct_theoretical * p_correct_problem
  combined_probability = 0.625 :=
  sorry

end student_answers_all_correctly_l751_751212


namespace find_side_b_in_triangle_l751_751088

theorem find_side_b_in_triangle (a : ℝ) (B C : ℝ) (A : ℝ) (sin : ℝ → ℝ) (b : ℝ)
  (ha : a = 8)
  (hB : B = 60)
  (hC : C = 75)
  (hA : A = 180 - 60 - 75)
  (hsinA : sin 45 = √2 / 2)
  (hsinB : sin 60 = √3 / 2)
  (h_sine_theorem : a / sin A = b / sin B) :
  b = 4 * √6 := by
  sorry

end find_side_b_in_triangle_l751_751088


namespace proof_problem_l751_751053

def proposition_p : Prop := ∃ x : ℝ, 2^(x-3) ≤ 0
def proposition_A : Prop := ∀ foci, (3*foci.1^2 + 4*foci.2^2 = 2) → foci.2 = 0
def proposition_B : Prop := ∃ y, (x^2 + y^2 - 2*x - 4*y - 1 = 0) ∧ (y = 0)
def proposition_C : Prop := ∀ (A B : Set), A ∪ B = A → B ⊆ A
def point_A := (1 : ℝ, 2 : ℝ)
def point_B := (3 : ℝ, 0 : ℝ)
def proposition_D : Prop := ¬ ∀ t ∈ Icc (0 : ℝ) 1, (point_A.1 + t*(point_B.1-point_A.1) + 2*(point_A.2 + t*(point_B.2-point_A.2)) - 3 ≠ 0)

theorem proof_problem : ¬p ∧ (proposition_D → False) := sorry

end proof_problem_l751_751053


namespace part1_part2_l751_751841

open Complex

def equation (a z : ℂ) : Prop := z^2 - (a + I) * z - (I + 2) = 0

theorem part1 (m : ℝ) (a : ℝ) : equation a m → a = 1 := by
  sorry

theorem part2 (a : ℝ) : ¬ ∃ n : ℝ, equation a (n * I) := by
  sorry

end part1_part2_l751_751841


namespace sqrt_fraction_eq_l751_751004

def nineFactorial : ℕ := 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def divisor : ℕ := 210
def fraction : ℚ := nineFactorial / divisor

theorem sqrt_fraction_eq : sqrt (fraction) = 216 * sqrt 3 := sorry

end sqrt_fraction_eq_l751_751004


namespace rightmost_nonzero_digit_of_a_78117_odd_l751_751430

   def a_n (n : ℕ) : ℕ := ((n + 8).factorial) / ((n - 1).factorial)
   
   def last_nonzero_digit (n : ℕ) : ℕ :=
     let digits := n.digits 10
     match digits.reverse.find (λ d => d ≠ 0) with
     | some d => d
     | none => 0

   theorem rightmost_nonzero_digit_of_a_78117_odd :
     last_nonzero_digit (a_n 78117) = 7 :=
   by
     sorry
   
end rightmost_nonzero_digit_of_a_78117_odd_l751_751430


namespace three_digit_numbers_count_correct_l751_751496

def digits : List ℕ := [2, 3, 4, 5, 5, 5, 6, 6]

def three_digit_numbers_count (d : List ℕ) : ℕ := 
  -- To be defined: Full implementation for counting matching three-digit numbers
  sorry

theorem three_digit_numbers_count_correct :
  three_digit_numbers_count digits = 85 :=
sorry

end three_digit_numbers_count_correct_l751_751496


namespace gcd_possible_values_l751_751269

theorem gcd_possible_values (a b : ℕ) (hab : a * b = 288) : 
  ∃ S : Finset ℕ, (∀ g : ℕ, g ∈ S ↔ ∃ p q r s : ℕ, p + r = 5 ∧ q + s = 2 ∧ g = 2^min p r * 3^min q s) 
  ∧ S.card = 14 := 
sorry

end gcd_possible_values_l751_751269


namespace seating_arrangement_ways_l751_751098

-- Define the problem conditions in Lean 4
def number_of_ways_to_seat (total_chairs : ℕ) (total_people : ℕ) := 
  Nat.factorial total_chairs / Nat.factorial (total_chairs - total_people)

-- Define the specific theorem to be proved
theorem seating_arrangement_ways : number_of_ways_to_seat 8 5 = 6720 :=
by
  sorry

end seating_arrangement_ways_l751_751098


namespace solution_set_of_fx_minus_2_l751_751606

noncomputable def f (x : ℝ) : ℝ := 2^x - 4

theorem solution_set_of_fx_minus_2:
  (∀ x, f(-x) = f(x)) → {x : ℝ | f(x-2) > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 4} :=
by
  intro h_even
  sorry

end solution_set_of_fx_minus_2_l751_751606


namespace problem_statement_l751_751416

open Real

theorem problem_statement (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * π)
  (h₁ : 2 * cos x ≤ sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x))
  ∧ sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x)) ≤ sqrt 2) :
  π / 4 ≤ x ∧ x ≤ 7 * π / 4 := sorry

end problem_statement_l751_751416


namespace slope_of_line_intersecting_hyperbola_l751_751034

theorem slope_of_line_intersecting_hyperbola 
  (A B : ℝ × ℝ)
  (hA : A.1^2 - A.2^2 = 1)
  (hB : B.1^2 - B.2^2 = 1)
  (midpoint_condition : (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1) :
  (B.2 - A.2) / (B.1 - A.1) = 2 :=
by
  sorry

end slope_of_line_intersecting_hyperbola_l751_751034


namespace length_ae_l751_751324

noncomputable def consecutive_points_length (ab bc cd de: ℝ) : ℝ :=
  ab + bc + cd + de

theorem length_ae {a b c d e : ℝ}
  (bc_eq_two_cd : bc = 2 * cd)
  (de_eq_8 : de = 8)
  (ab_eq_5 : ab = 5)
  (ac_eq_11 : ab + bc = 11) :
  consecutive_points_length ab bc cd de = 22 :=
by
  have bc_eq : bc = 11 - ab, from calc
    bc = 11 - ab : by sorry,

  have cd_eq : cd = bc / 2, from calc
    cd = bc / 2 : by sorry,

  have cd_val : cd = 3, from calc
    cd = 3 : by sorry,

  have ae_length : ab + bc + cd + de = 22, from calc
    ab + bc + cd + de = 22 : by sorry,

  exact ae_length

end length_ae_l751_751324


namespace members_per_group_l751_751668

theorem members_per_group (boys girls groups : ℕ) (h_boys : boys = 9) (h_girls : girls = 12) (h_groups : groups = 7) :
  (boys + girls) / groups = 3 :=
by
  rw [h_boys, h_girls, h_groups]
  norm_num
  sorry

end members_per_group_l751_751668


namespace goose_eggs_at_pond_l751_751173

noncomputable def total_goose_eggs (E : ℝ) : Prop :=
  (5 / 12) * (5 / 16) * (5 / 9) * (3 / 7) * E = 84

theorem goose_eggs_at_pond : 
  ∃ E : ℝ, total_goose_eggs E ∧ E = 678 :=
by
  use 678
  dsimp [total_goose_eggs]
  sorry

end goose_eggs_at_pond_l751_751173


namespace exists_small_triangle_l751_751024

variable {P1 P2 P3 P4 : Type}
variable {A B C : Type}
variables [HasArea P1] [HasArea P2] [HasArea P3] [HasArea P4] [HasArea A] [HasArea B] [HasArea C]
variable (area : A → ℝ)

-- Given that a quadrilateral P1P2P3P4 has its vertices lying on sides of triangle ABC
def quadrilateral_on_triangle_sides (P1 P2 P3 P4 A B C : Type) [HasArea P1] [HasArea P2] [HasArea P3] 
  [HasArea P4] [HasArea A] [HasArea B] [HasArea C] : Prop :=
  -- Define the quadrilateral vertices lying on the sides of the triangle
  sorry

-- Prove that one of the triangles' areas is ≤ 1/4 of the area of triangle ABC
theorem exists_small_triangle (P1 P2 P3 P4 : Type) (A B C : Type) [HasArea P1] [HasArea P2] [HasArea P3] 
  [HasArea P4] [HasArea A] [HasArea B] [HasArea C] 
  (h : quadrilateral_on_triangle_sides P1 P2 P3 P4 A B C)
  (area_ABC : ℝ) (area : A → ℝ) :
  ∃ (p1 p2 p3 : Type) [HasArea p1] [HasArea p2] [HasArea p3], 
    (p1 = P1 ∧ p2 = P2 ∧ p3 = P3 
      ∨ p1 = P1 ∧ p2 = P2 ∧ p3 = P4 
      ∨ p1 = P1 ∧ p2 = P3 ∧ p3 = P4 
      ∨ p1 = P2 ∧ p2 = P3 ∧ p3 = P4) 
    ∧ area p1 + area p2 + area p3 ≤ (1/4) * area_ABC :=
by
  sorry

end exists_small_triangle_l751_751024


namespace relationship_between_abc_l751_751016

-- Definitions and conditions
def a : ℝ := 1 + Real.sqrt 7
def b : ℝ := Real.sqrt 3 + Real.sqrt 5
def c : ℝ := 4

-- Proof statement
theorem relationship_between_abc : c > b ∧ b > a := by
  sorry

end relationship_between_abc_l751_751016


namespace price_difference_l751_751739

theorem price_difference (P : ℝ) :
  let new_price := 1.20 * P
  let discounted_price := 0.96 * P
  let difference := new_price - discounted_price
  difference = 0.24 * P := by
  let new_price := 1.20 * P
  let discounted_price := 0.96 * P
  let difference := new_price - discounted_price
  sorry

end price_difference_l751_751739


namespace set_intersection_complement_l751_751056

open Set

universe u

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

/-- Given the universal set U={0,1,2,3,4,5}, sets A={0,2,4}, and B={0,5}, prove that
    the intersection of A and the complement of B in U is {2,4}. -/
theorem set_intersection_complement:
  U = {0, 1, 2, 3, 4, 5} →
  A = {0, 2, 4} →
  B = {0, 5} →
  A ∩ (U \ B) = {2, 4} := 
by
  intros hU hA hB
  sorry

end set_intersection_complement_l751_751056


namespace quadratic_residue_iff_l751_751597

open Nat

theorem quadratic_residue_iff (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) (n : ℤ) (hn : n % p ≠ 0) :
  (∃ a : ℤ, (a^2) % p = n % p) ↔ (n ^ ((p - 1) / 2)) % p = 1 :=
sorry

end quadratic_residue_iff_l751_751597


namespace range_m_range_a_l751_751588

def f (x : ℝ) : ℝ := (x^2 - 3 * x + 8) / 2

def g (a x : ℝ) : ℝ := a ^ x

theorem range_m (x0 : ℝ) (h1 : 2 ≤ x0) (m : ℝ) (h2 : f x0 = m) : 3 ≤ m :=
sorry

theorem range_a (a : ℝ) (h : ∀ x1 : ℝ, 2 ≤ x1 → ∃ x2 : ℝ, 2 < x2 ∧ f x1 = g a x2) : 1 < a ∧ a < real.sqrt 3 :=
sorry

end range_m_range_a_l751_751588


namespace angle_at_7_20_is_100_degrees_l751_751370

def angle_between_hands_at_7_20 : ℝ := 100

theorem angle_at_7_20_is_100_degrees
    (hour_hand_pos : ℝ := 210) -- 7 * 30 degrees
    (minute_hand_pos : ℝ := 120) -- 4 * 30 degrees
    (hour_hand_move_per_minute : ℝ := 0.5) -- 0.5 degrees per minute
    (time_past_7_clock : ℝ := 20) -- 20 minutes
    (adjacent_angle : ℝ := 30) -- angle between adjacent numbers
    : angle_between_hands_at_7_20 = 
      (hour_hand_pos - (minute_hand_pos - hour_hand_move_per_minute * time_past_7_clock)) :=
sorry

end angle_at_7_20_is_100_degrees_l751_751370


namespace incorrect_method_D_l751_751271

-- Conditions definitions
def conditionA (locus : Set α) (cond : α → Prop) :=
  ∀ p, (p ∈ locus ↔ cond p)

def conditionB (locus : Set α) (cond : α → Prop) :=
  ∀ p, (cond p ↔ p ∈ locus)

def conditionC (locus : Set α) (cond : α → Prop) :=
  ∀ p, (¬ (p ∈ locus) ↔ ¬ (cond p))

def conditionD (locus : Set α) (cond : α → Prop) :=
  ∀ p, (p ∈ locus → cond p) ∧ (∃ p, cond p ∧ ¬ (p ∈ locus))

def conditionE (locus : Set α) (cond : α → Prop) :=
  ∀ p, (cond p ↔ p ∈ locus)

-- Main theorem
theorem incorrect_method_D {α : Type} (locus : Set α) (cond : α → Prop) :
  conditionD locus cond →
  ¬ (conditionA locus cond) ∧
  ¬ (conditionB locus cond) ∧
  ¬ (conditionC locus cond) ∧
  ¬ (conditionE locus cond) :=
  sorry

end incorrect_method_D_l751_751271


namespace find_OP_length_l751_751954

-- Definitions for the given problem.
variable (O Q P: Point)
variable (A B C: Point)
variable (ABC: Triangle A B C)
variable (AP CQ: LineSegment)
variable (medAP : IsMedian AP ABC)
variable (medCQ : IsMedian CQ ABC)
variable (O_is_centroid : IsCentroid O ABC)
variable (OQ_length : Length O Q = 4)
variable (AP_length : Length A P = 15)

-- The theorem to be proved.
theorem find_OP_length : Length O P = 10 := 
by
  sorry

end find_OP_length_l751_751954


namespace find_a9_l751_751083

variable (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ)

/- The polynomial condition -/
def polynomial_condition (x : ℝ) : Prop :=
  x^2 + x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + 
                a5 * (x+1)^5 + a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + 
                a9 * (x+1)^9 + a10 * (x+1)^10

/- The target property we need to prove -/
theorem find_a9 (h : ∀ x : ℝ, polynomial_condition x) : a9 = -10 :=
by
  sorry

end find_a9_l751_751083


namespace sum_of_roots_l751_751703

theorem sum_of_roots (z1 z2 : ℂ) (h : z1^2 + 5*z1 - 14 = 0 ∧ z2^2 + 5*z2 - 14 = 0) :
  z1 + z2 = -5 :=
sorry

end sum_of_roots_l751_751703


namespace last_digit_two_power_2015_l751_751986

/-- The last digit of powers of 2 cycles through 2, 4, 8, 6. Therefore, the last digit of 2^2015 is the same as 2^3, which is 8. -/
theorem last_digit_two_power_2015 : (2^2015) % 10 = 8 :=
by sorry

end last_digit_two_power_2015_l751_751986


namespace sum_of_eight_numbers_l751_751518

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l751_751518


namespace sum_of_digits_in_rectangle_l751_751995

theorem sum_of_digits_in_rectangle :
  ∃ A B C : ℕ,
    (4 + A + 1 + B = 12) ∧
    (4 + A + 1 + B = 6 + 6) ∧
    (C + 1 + 6 + C = 11) ∧
    (1 + B + 2 + C = 9) ∧
    (A + 8 + 8 = 8) ∧
    (A + 8 + B + 2 + C = 13) ∧
    (9 = 4 + A + 1 + B) ∧
    (B + 2 + C = 9) ∧    
    B = 5 ∧ A = 2 ∧ C = 6 :=
sorry

end sum_of_digits_in_rectangle_l751_751995


namespace seq_product_divisible_by_1419_l751_751737

theorem seq_product_divisible_by_1419 : 
  ∃ s, s = [86, 87, 88] ∧ 1419 ∣ (list.prod s) := 
by {
  use [86, 87, 88],
  split,
  { refl },
  { rw list.prod,
    --  product calculation steps
    sorry },
}

end seq_product_divisible_by_1419_l751_751737


namespace identify_unused_piece_l751_751614

def num_squares := [4, 5, 6, 7, 8]

def total_squares (pieces : List ℕ) : ℕ := pieces.sum

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem identify_unused_piece :
  ∃ unused_piece, is_perfect_square (total_squares num_squares - unused_piece) ∧ unused_piece ∈ num_squares ∧ unused_piece = 5 :=
by {
  sorry
}

end identify_unused_piece_l751_751614


namespace value_of_x_that_makes_sqrt_undefined_l751_751459

theorem value_of_x_that_makes_sqrt_undefined (x : ℕ) (hpos : 0 < x) : (x = 1) ∨ (x = 2) ↔ (x - 3 < 0) := by
  sorry

end value_of_x_that_makes_sqrt_undefined_l751_751459


namespace current_speed_is_1_kmh_l751_751732

-- Define upstream and downstream speed in km/min
def speed_upstream_km_per_min : ℝ := 1 / 20
def speed_downstream_km_per_min : ℝ := 1 / 12

-- Convert speeds to km/h
def speed_upstream_km_per_h : ℝ := speed_upstream_km_per_min * 60
def speed_downstream_km_per_h : ℝ := speed_downstream_km_per_min * 60

-- Define the speed of the current
def speed_of_current : ℝ := (speed_downstream_km_per_h - speed_upstream_km_per_h) / 2

-- Prove that the speed of the current is 1 km/h
theorem current_speed_is_1_kmh : speed_of_current = 1 :=
by 
  -- The proof will be inserted here
  sorry

end current_speed_is_1_kmh_l751_751732


namespace distance_traveled_by_car_correct_l751_751548

def total_distance : ℝ := 30.000000000000007
def distance_by_foot : ℝ := (1 / 3) * total_distance
def distance_by_bus : ℝ := (3 / 5) * total_distance
def total_covered_before_car : ℝ := distance_by_foot + distance_by_bus
def distance_by_car : ℝ := total_distance - total_covered_before_car

theorem distance_traveled_by_car_correct :
  distance_by_car = 2.000000000000001 := 
sorry

end distance_traveled_by_car_correct_l751_751548


namespace jail_time_calculation_l751_751245

def total_arrests (arrests_per_day : ℕ) (cities : ℕ) (days : ℕ) : ℕ := 
  arrests_per_day * cities * days

def jail_time_before_trial (arrests : ℕ) (days_before_trial : ℕ) : ℕ := 
  days_before_trial * arrests

def jail_time_after_trial (arrests : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_after_trial * arrests

def combined_jail_time (weeks_before_trial : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_before_trial + weeks_after_trial

noncomputable def total_jail_time_in_weeks : ℕ := 
  let arrests := total_arrests 10 21 30
  let weeks_before_trial := jail_time_before_trial arrests 4 / 7
  let weeks_after_trial := jail_time_after_trial arrests 1
  combined_jail_time weeks_before_trial weeks_after_trial

theorem jail_time_calculation : 
  total_jail_time_in_weeks = 9900 :=
sorry

end jail_time_calculation_l751_751245


namespace prime_sum_l751_751199

theorem prime_sum (p q r : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime)
    (h1 : p * q + q * r + r * p = 191)
    (h2 : p + q = r - 1) : p + q + r = 25 :=
by
  sorry  -- proof to be filled in 

end prime_sum_l751_751199


namespace functional_expression_and_range_l751_751463

-- We define the main problem conditions and prove the required statements based on those conditions
theorem functional_expression_and_range (x y : ℝ) (h1 : ∃ k : ℝ, (y + 2) = k * (4 - x) ∧ k ≠ 0)
                                        (h2 : x = 3 → y = 1) :
                                        (y = -3 * x + 10) ∧ ( -2 < y ∧ y < 1 → 3 < x ∧ x < 4) :=
by
  sorry

end functional_expression_and_range_l751_751463


namespace inequality_system_integer_solutions_l751_751482

theorem inequality_system_integer_solutions (m : ℝ) :
  (∃ x : ℤ, 3 * (x : ℝ) - m > 0 ∧ x - 1 ≤ 5) →
  (∃ b : ℕ, 4 = b ∧ {x : ℤ | 3 * (x : ℝ) - m > 0 ∧ x - 1 ≤ 5}.to_finset.card = b) →
  6 ≤ m ∧ m < 9 := 
sorry

end inequality_system_integer_solutions_l751_751482


namespace chord_length_intersection_l751_751544

-- Definition of the line in polar coordinates
def line_theta_pi_six (theta rho : ℝ) : Prop :=
  theta = π / 6

-- Definition of the circle in polar coordinates
def circle_rho_2cos_shifted (theta rho : ℝ) : Prop :=
  rho = 2 * cos (theta - π / 6)

-- The mathematical proof problem: Prove that the length of the chord is 2
theorem chord_length_intersection (theta rho : ℝ) :
  (∃ rho₁ rho₂, line_theta_pi_six θ rho₁ ∧ circle_rho_2cos_shifted θ rho₂) →
  ∃ chord_length, chord_length = 2 :=
by
  sorry

end chord_length_intersection_l751_751544


namespace verify_expressions_l751_751498

variable (x y : ℝ)
variable (h : x / y = 5 / 3)

theorem verify_expressions :
  (2 * x + y) / y = 13 / 3 ∧
  y / (y - 2 * x) = 3 / -7 ∧
  (x + y) / x = 8 / 5 ∧
  x / (3 * y) = 5 / 9 ∧
  (x - 2 * y) / y = -1 / 3 := by
sorry

end verify_expressions_l751_751498


namespace total_percentage_increase_l751_751948

variable (initial_earnings_USD : ℕ)
variable (first_earnings_EUR : ℕ)
variable (first_exchange_rate_USD_per_EUR : ℝ)
variable (second_earnings_GBP : ℕ)
variable (second_exchange_rate_USD_per_GBP : ℝ)

def earnings_USD_after_first_raise : ℝ := (first_earnings_EUR : ℝ) * first_exchange_rate_USD_per_EUR
def earnings_USD_after_second_raise : ℝ := (second_earnings_GBP : ℝ) * second_exchange_rate_USD_per_GBP

def percentage_increase (initial : ℝ) (final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem total_percentage_increase 
  (h1 : initial_earnings_USD = 30)
  (h2 : first_earnings_EUR = 35)
  (h3 : first_exchange_rate_USD_per_EUR = 1 / 0.90)
  (h4 : second_earnings_GBP = 28)
  (h5 : second_exchange_rate_USD_per_GBP = 1.35) :
  percentage_increase initial_earnings_USD
                      earnings_USD_after_second_raise = 26 :=
by
  sorry

end total_percentage_increase_l751_751948


namespace largest_of_three_l751_751675

structure RealTriple (x y z : ℝ) where
  h1 : x + y + z = 3
  h2 : x * y + y * z + z * x = -8
  h3 : x * y * z = -18

theorem largest_of_three {x y z : ℝ} (h : RealTriple x y z) : max x (max y z) = Real.sqrt 5 :=
  sorry

end largest_of_three_l751_751675


namespace perpendicular_bisectors_intersect_at_one_point_l751_751367

open EuclideanGeometry

variable {A B C P Q X Y N : Point}
variable [Circumcircle A B C Γ]
variable [MidpointArcExcludingA M B C]
variable [Parallel AM l_b l_c]
variable [PassesThrough l_b B P]
variable [PassesThrough l_c C Q]
variable [Intersects PQ AB X]
variable [Intersects PQ AC Y]
variable [CircumcircleTriangle A X Y]
variable [Intersects AM N]

theorem perpendicular_bisectors_intersect_at_one_point
  (h1 : B ∉ P)
  (h2 : C ∉ Q)
  (h3 : N ≠ A) :
  ∃ O : Point, PerpendicularBisector BC O ∧ PerpendicularBisector XY O ∧ PerpendicularBisector MN O :=
  sorry

end perpendicular_bisectors_intersect_at_one_point_l751_751367


namespace repeating_decimal_to_fraction_l751_751710

theorem repeating_decimal_to_fraction :
  (0.512341234123412341234 : ℝ) = (51229 / 99990 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l751_751710


namespace xy_extrema_l751_751651

noncomputable def xy_product (a : ℝ) : ℝ := a^2 - 1

theorem xy_extrema (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^2 + y^2 = -a^2 + 2) : 
  -1 ≤ xy_product a ∧ xy_product a ≤ 1/3 :=
by
  sorry

end xy_extrema_l751_751651


namespace rectangle_area_l751_751650

theorem rectangle_area (l w : ℕ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 120) : l * w = 800 :=
by
  -- proof to be filled in
  sorry

end rectangle_area_l751_751650


namespace range_of_function_l751_751645

def func (x : ℕ) : ℤ := x^2 - 2 * x

theorem range_of_function :
  {y : ℤ | ∃ x : ℕ, x ∈ {0, 1, 2, 3} ∧ y = func x} = {-1, 0, 3} :=
by
  sorry

end range_of_function_l751_751645


namespace prime_extension_l751_751018

noncomputable def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m ∈ {2, 3, …, p-1}, ¬(m ∣ p)

theorem prime_extension (n : ℕ) (k : ℕ) 
  (h1 : n ≥ 2) 
  (h2 : ∀ k, 0 ≤ k ∧ k ≤ (sqrt (n / 3)) → is_prime (k^2 + k + n))
  : ∀ k, 0 ≤ k ∧ k ≤ n - 2 → is_prime (k^2 + k + n) :=
sorry

end prime_extension_l751_751018


namespace complex_magnitude_product_l751_751376

theorem complex_magnitude_product:
  let z1 := (4 - 3 * Complex.i)
  let z2 := (4 + 3 * Complex.i)
  |z1| * |z2| = 25 :=
by
  let z1 := (4 - 3 * Complex.i)
  let z2 := (4 + 3 * Complex.i)
  show |z1| * |z2| = 25
  sorry

end complex_magnitude_product_l751_751376


namespace chips_placement_possible_l751_751328

theorem chips_placement_possible :
  ∃ (f : ℕ → ℕ → Prop), 
  (∀ i j, i ∈ {0, 1, 2, ..., 11} → j ∈ {0, 1, 2, ..., 11} → (if f i j then 1 else 0) = [0, 1]) ∧ 
  (∃ i j, i ∈ {0, 1, 2, ..., 10} → j ∈ {0, 1, 2, ..., 10} → (if f i j then 1 else 0) + (if f (i+1) j then 1 else 0) + (if f i (j+1) then 1 else 0) + (if f (i+1) (j+1) then 1 else 0) ≡ 1 [ZMOD 2]) ∧ 
  (∀ (i j : ℕ), i ∈ {0, 1, ..., 10} ∧ j ∈ {0, 1, ..., 10} ∧ ¬((i = 0 ∧ j = 0) ∨ (i = 1 ∧ j = 0) ∨ (i = 2 ∧ j = 0)) → ((if f i j then 1 else 0) + (if f (i+1) j then 1 else 0) + (if f i (j+1) then 1 else 0) + (if f (i+1) (j+1) then 1 else 0) ≡ 0 [ZMOD 2])) :=
  sorry

end chips_placement_possible_l751_751328


namespace next_perfect_square_l751_751348

theorem next_perfect_square (n : ℕ) (x : ℕ) (h : x = n^2) : ∃ y : ℕ, y = (n + 1)^2 ∧ y = x + 2 * (nat.sqrt x) + 1 :=
by
  use (n + 1)^2
  rw [h, nat.sqrt_eq]
  exact ⟨rfl, rfl⟩

end next_perfect_square_l751_751348


namespace number_of_lines_l751_751021

theorem number_of_lines (slope : ℝ) (ellipse : ℝ → ℝ → Prop) 
  (intersect : ∃ A B, ellipse A B ∧ A ≠ B)
  (length_is_integer : ∀ A B, |(A - B)| ∈ ℤ)
  (equiv : (y: ℝ) = (x: ℝ) + b → 3 * x^2 + 4 * b * x + 2 * b^2 - 4 = 0)
  : ∃ lines, lines = 6 :=
begin
  -- proof is skipped
  sorry
end

end number_of_lines_l751_751021


namespace part1_l751_751940

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {triangle_ABC : triangle A B C}

-- In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively.
axiom h1: side triangle_ABC A = a
axiom h2: side triangle_ABC B = b
axiom h3: side triangle_ABC C = c
axiom h4: tan A = sin B

-- Prove that 2ac = b^2 + c^2 - a^2.
theorem part1 : 2 * a * c = b^2 + c^2 - a^2 :=
sorry

end part1_l751_751940


namespace inequality_proof_l751_751277

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l751_751277


namespace solution_is_correct_l751_751722

noncomputable def solve_system_of_inequalities : Prop :=
  ∃ x y : ℝ, 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧ 
    (2 * x - 4 * y ≤ -3) ∧ 
    (x = -1/3) ∧ 
    (y = 2/3)

theorem solution_is_correct : solve_system_of_inequalities :=
sorry

end solution_is_correct_l751_751722


namespace rafts_travel_time_l751_751761

theorem rafts_travel_time (steamboat_downstream_days : ℕ) (steamboat_upstream_days : ℕ) :
  (steamboat_downstream_days = 5) →
  (steamboat_upstream_days = 7) →
  let C := 1 / 35 in
  1 / C = 35 :=
by {
  intros h_downstream h_upstream,
  let S := (1 / 5 + 1 / 7) / 2,
  let C := 1 / 35,
  have h_C : C = 1 / 35 := by rfl,
  exact h_C,
  sorry
}

end rafts_travel_time_l751_751761


namespace namzhil_smartwatch_funds_l751_751171

theorem namzhil_smartwatch_funds : 
  let a := 500 in
  let namzhil_amount := ((a^2 + 4 * a + 3) * (a - 2)^2 - a^2 * 503 * 497) in
  namzhil_amount < 2019 :=
by
  sorry

end namzhil_smartwatch_funds_l751_751171


namespace sum_of_eight_numbers_l751_751519

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l751_751519


namespace difference_of_perimeters_l751_751226

/-- We have two figures: figure1 and figure2 with specific dimensions and additional structs.
    figure1:
    - Outer rectangle: 5 units width, 2 units height, perimeter contribution = 2(5+2).
    - Middle vertical rectangle: additional height = 3, perimeter contribution = 2*3.
    - Inner vertical rectangle: additional height = 4, perimeter contribution = 2*4.
    
    figure2:
    - Outer rectangle: 5 units width, 3 units height, perimeter contribution = 2(5+3).
    - Five vertical lines each 3 units contributing double per line, perimeter contribution = 5*2*3.

    Prove that the difference of their perimeters is 2 units. -/
theorem difference_of_perimeters :
  let figure1_perimeter := 2 * (5 + 2) + 2 * 3 + 2 * 4,
      figure2_perimeter := 2 * (5 + 3) + 5 * 2 * 3
  in figure2_perimeter - figure1_perimeter = 2 := by
  sorry

end difference_of_perimeters_l751_751226


namespace cube_face_product_l751_751573

open Finset

theorem cube_face_product (numbers : Finset ℕ) (hs : numbers = range (12 + 1)) :
  ∃ top_face bottom_face : Finset ℕ,
    top_face.card = 4 ∧
    bottom_face.card = 4 ∧
    (numbers \ (top_face ∪ bottom_face)).card = 4 ∧
    (∏ x in top_face, x) = (∏ x in bottom_face, x) :=
by
  use {2, 4, 9, 10}
  use {3, 5, 6, 8}
  repeat { split };
  -- Check cardinality conditions
  sorry

end cube_face_product_l751_751573
