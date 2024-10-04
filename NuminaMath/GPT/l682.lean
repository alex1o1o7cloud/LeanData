import Mathlib

namespace angle_CBA_l682_682251

open Real EuclideanGeometry

variables {A B C D : Point}
variables (h_trapezoid : Trapezoid (Line.mk A B)(Line.mk C D))
variables (h_AB : dist A B = 3)
variables (h_CD : dist C D = 3)
variables (h_DA : dist D A = 3)
variables (h_angle_ADC : ∠ A D C = 120)

theorem angle_CBA : ∠ C B A = 30 :=
by
  sorry

end angle_CBA_l682_682251


namespace simplify_expr1_simplify_expr2_l682_682283

theorem simplify_expr1 (a b : ℝ) : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 (t : ℝ) : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l682_682283


namespace A_salary_is_3000_l682_682999

theorem A_salary_is_3000 
    (x y : ℝ) 
    (h1 : x + y = 4000)
    (h2 : 0.05 * x = 0.15 * y) 
    : x = 3000 := by
  sorry

end A_salary_is_3000_l682_682999


namespace calculate_expression_l682_682017

noncomputable def solve_expression : ℝ :=
  let term1 := (real.pi - 1) ^ 0
  let term2 := 4 * real.sin (real.pi / 4) -- sin 45° = sin (π/4)
  let term3 := real.sqrt 8
  let term4 := real.abs (-3)
  term1 + term2 - term3 + term4

theorem calculate_expression : solve_expression = 4 := by
  sorry

end calculate_expression_l682_682017


namespace perpendicular_OB_FD_OC_ED_parallel_OH_MN_l682_682196

section Geometry

variables {P Q R O H M N : Type} [MetricSpace P Q] [MetricSpace P R] [MetricSpace O Q] [MetricSpace O R] [MetricSpace H Q]
          [MetricSpace H R] [MetricSpace M Q] [MetricSpace M R] [MetricSpace N Q] [MetricSpace N R]

-- Definitions for triangle ABC and points
variables (A B C D E F : P) (O : O) (H : H) (M : M) (N : N)
  (AD : Line P) (BE : Line P) (CF : Line P) (ED : Line P) (FD : Line P)

-- Conditions
axiom triangle_ABC : Triangle A B C
axiom circumcenter_O : IsCircumcenter O A B C
axiom altitudes_intersect_H : IntersectAt AD BE CF H
axiom line_ED_intersects_M : Intersects ED (Line.mk A B) M
axiom line_FD_intersects_N : Intersects FD (Line.mk A C) N

-- Goal 1: OB ⟂ FD and OC ⟂ ED
theorem perpendicular_OB_FD_OC_ED 
  (OB FD OC ED : Line P) : Perpendicular OB FD ∧ Perpendicular OC ED :=
sorry

-- Goal 2: OH ∥ MN
theorem parallel_OH_MN
  (OH MN : Line P) : Parallel OH MN :=
sorry

end Geometry

end perpendicular_OB_FD_OC_ED_parallel_OH_MN_l682_682196


namespace book_arrangement_count_l682_682596

-- Conditions
def num_math_books := 4
def num_history_books := 5

-- The number of arrangements is
def arrangements (n m : Nat) : Nat :=
  let choose_end_books := n * (n - 1)
  let choose_middle_book := (n - 2)
  let remaining_books := (n - 3) + m
  choose_end_books * choose_middle_book * Nat.factorial remaining_books

theorem book_arrangement_count (n m : Nat) (h1 : n = num_math_books) (h2 : m = num_history_books) :
  arrangements n m = 120960 :=
by
  rw [h1, h2, arrangements]
  norm_num
  sorry

end book_arrangement_count_l682_682596


namespace smallest_positive_multiple_of_45_is_45_l682_682941

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682941


namespace minimum_value_of_a_l682_682262

variable {f : ℝ → ℝ}
variable {a : ℝ}

-- Given 1: f is twice differentiable
variable [twice_differentiable ℝ f]

-- Given 2: ∀ x ∈ ℝ, f(x) + f(-x) = 2x²
axiom cond1 : ∀ x : ℝ, f x + f (-x) = 2 * x ^ 2

-- Given 3: ∀ x < 0, f''(x) + 1 < 2x
axiom cond2 : ∀ x : ℝ, x < 0 → (deriv (deriv f)) x + 1 < 2 * x

-- Given 4: f(a+1) ≤ f(-a) + 2a + 1
axiom cond3 : f (a + 1) ≤ f (-a) + 2 * a + 1

-- Prove: a = -1 / 2
theorem minimum_value_of_a : a = -1 / 2 :=
by
  sorry

end minimum_value_of_a_l682_682262


namespace smallest_positive_multiple_of_45_l682_682845

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682845


namespace clock_displays_unique_digits_minutes_l682_682310

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end clock_displays_unique_digits_minutes_l682_682310


namespace plane_equation_l682_682726

theorem plane_equation (A B C D : ℤ) (h1 : A > 0)
  (h2 : Int.gcd (Int.natAbs A) (Int.natAbs B) (Int.natAbs C) (Int.natAbs D) = 1)
  (h3 : ∀ x y z : ℝ, (x + y + z = 1) ∧ (x - 2 * y + 2 * z = 4) → A * x + B * y + C * z + D = 0)
  (h4 : ∀ x y z : ℝ, A * x + B * y + C * z + D = 0 → 
        (abs (A * 1 + B * 2 + C * 0 + D) / sqrt (A^2 + B^2 + C^2) = 3 / sqrt 14)) :
  A = 0 ∧ B = 3 ∧ C = -1 ∧ D = 3 := sorry

end plane_equation_l682_682726


namespace smallest_positive_multiple_of_45_l682_682830

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682830


namespace value_of_x_in_logarithm_equation_l682_682599

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem value_of_x_in_logarithm_equation (n : ℝ) (h1 : n = 343) : 
  ∃ (x : ℝ), log_base x n + log_base 7 n = log_base 1 n :=
by
  sorry

end value_of_x_in_logarithm_equation_l682_682599


namespace cartesian_equations_line_curve_distance_range_l682_682472

open Real

def line_parametric_equations (t : ℝ) :=
  (x : ℝ) × (y : ℝ) := (t - 1, t + 2)

def curve_polar_equation (θ : ℝ) : ℝ :=
  sqrt 3 / sqrt (1 + 2 * cos θ ^ 2)

theorem cartesian_equations_line_curve 
  (t θ : ℝ)
  (L : ∀ t, line_parametric_equations t)
  (C : ∀ θ, curve_polar_equation θ) :
  (∃ t, L t = (x, y) ∧ x - y + 3 = 0) ∧
  (∃ θ, C θ = ρ ∧ ρ ^ 2 + 2 * ρ ^ 2 * cos θ ^ 2 = 3 → x^2 + y^2 / 3 = 1) :=
sorry

theorem distance_range (α : ℝ) :
  (∀ α, cos α = d / ((2 * cos (α + π / 3) + 3) / sqrt 2)) →
  (d >= sqrt 2 / 2 ∧ d <= 5 * sqrt 2 / 2) := 
sorry

end cartesian_equations_line_curve_distance_range_l682_682472


namespace smallest_positive_multiple_of_45_is_45_l682_682795

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682795


namespace equidistant_to_KL_l682_682618

open EuclideanGeometry

variables {A B C A' B' C' K L : Point} (incircle : Circle)

-- Given conditions as assumptions
variables (ABC_triangle : Triangle A B C) 
          (incircle_tangent_to_BC : incircle.TangentTo BC A')
          (incircle_tangent_to_AC : incircle.TangentTo AC B')
          (incircle_tangent_to_AB : incircle.TangentTo AB C')
          (K_L_on_incircle : K ∈ incircle ∧ L ∈ incircle)
          (angle_condition : ∠ AKB' + ∠ BKA' = 180 ∧ ∠ ALB' + ∠ BLA' = 180)

-- Target statement to prove
theorem equidistant_to_KL : 
  IsEquidistantFromLine (LineThrough K L) [A', B', C'] :=
sorry

end equidistant_to_KL_l682_682618


namespace hyperbola_standard_form_l682_682149

noncomputable def hyperbola_equations (a b c : ℝ) : Prop :=
  c = 10 ∧ c ^ 2 = a ^ 2 + b ^ 2 ∧
  ((b / a = 4 / 3 ∧ a = 6 ∧ b = 8 ∧ (x^2 / 36 - y^2 / 64 = 1)) ∨
   (a / b = 4 / 3 ∧ a = 8 ∧ b = 6 ∧ (y^2 / 64 - x^2 / 36 = 1)))

theorem hyperbola_standard_form :
  ∃ a b c : ℝ, hyperbola_equations a b c :=
begin
  sorry

end hyperbola_standard_form_l682_682149


namespace num_divisors_36_l682_682559

theorem num_divisors_36 : ∃ n, n = 9 ∧ ∀ d : ℕ, d ∣ 36 → (d > 0 ∧ d ≤ 36) → ∃ k : ℕ, d = (2^k or 3^k or (2^p * 3^q) where p, q <= 2):
begin
  have factorization : 36 = 2^2 * 3^2 := by norm_num,
  have exponents : (2+1)*(2+1) = 9 := by norm_num,
  use 9,
  split,
  { exact exponents },
  {
    intros d hd hpos_range,
    cases hpos_range with hpos hrange,
    sorry -- Proof showing that there are exactly 9 positive divisors.
  },
end

end num_divisors_36_l682_682559


namespace relationship_among_a_b_c_l682_682515

open Real

variable (x : ℝ) (a b c : ℝ)

noncomputable def conditions (x : ℝ) := (exp (-1) < x) ∧ (x < 1) 

noncomputable def a_def (x : ℝ) := log x

noncomputable def b_def (x : ℝ) := (1 / 2) ^ log x

noncomputable def c_def (x : ℝ) := exp (log x)

theorem relationship_among_a_b_c (h : conditions x) : let a := a_def x in
                    let b := b_def x in
                    let c := c_def x in
                    b > c ∧ c > a :=
by
  let a := a_def x
  let b := b_def x
  let c := c_def x
  sorry

end relationship_among_a_b_c_l682_682515


namespace smaller_radius_conf1_l682_682729

-- Definitions
def Configuration1 (radius : ℝ) : Prop :=
  ∃ r, r < radius ∧ is_regular_pentagon (distance_vertices = 1) (distance_penta_pyramid = 1)

def Configuration2 (radius : ℝ) : Prop :=
  ∃ r, r ≥ 1 ∧ is_regular_hexagon (distance_vertices = 1)

-- Main Problem Statement
theorem smaller_radius_conf1 :
  (∀ radius, Configuration1 radius) ∧ (∀ radius, Configuration2 radius) → 
  (∃ radius1 radius2, radius1 < radius2) :=
begin
  sorry,
end

end smaller_radius_conf1_l682_682729


namespace smallest_positive_multiple_of_45_l682_682966

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682966


namespace percentage_short_l682_682706

def cost_of_goldfish : ℝ := 0.25
def sale_price_of_goldfish : ℝ := 0.75
def tank_price : ℝ := 100
def goldfish_sold : ℕ := 110

theorem percentage_short : ((tank_price - (sale_price_of_goldfish - cost_of_goldfish) * goldfish_sold) / tank_price) * 100 = 45 := 
by
  sorry

end percentage_short_l682_682706


namespace number_of_true_propositions_l682_682162

noncomputable def f (x : ℝ) := x^3
def C := {p : ℝ × ℝ | p.2 = f p.1}

def proposition1 : Prop :=
  ∀ M ∈ C, ∃! t, is_tangent_line t C M

def proposition2 : Prop :=
  ∀ (P : ℝ × ℝ), P ∈ C ∧ P.1 ≠ 0 →
  ∃ (Q : ℝ × ℝ), Q ∈ C ∧ (P.1 + Q.1) / 2 = 0

noncomputable def g (x : ℝ) := |f x - 2 * sin (2 * x)|

def proposition3 : Prop := ∀ x, 0 ≤ g x

def proposition4 : Prop :=
  ∀ x ∈ Icc 1 2, ∀ a, f (x + a) ≤ 8 * f x → a ≤ 1

theorem number_of_true_propositions : 
  (proposition1 → true) ∧ (proposition2 → true) ∧ (proposition3 → true) ∧ (proposition4 → false) → true :=
sorry

end number_of_true_propositions_l682_682162


namespace smallest_positive_multiple_of_45_l682_682970

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682970


namespace statement_A_statement_B_statement_C_statement_D_statement_E_l682_682464

def diamond (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem statement_A : ∀ (x y : ℝ), diamond x y = diamond y x := sorry

theorem statement_B : ∀ (x y : ℝ), 2 * (diamond x y) ≠ diamond (2 * x) (2 * y) := sorry

theorem statement_C : ∀ (x : ℝ), diamond x 0 = x^2 := sorry

theorem statement_D : ∀ (x : ℝ), diamond x x = 0 := sorry

theorem statement_E : ∀ (x y : ℝ), x = y → diamond x y = 0 := sorry

end statement_A_statement_B_statement_C_statement_D_statement_E_l682_682464


namespace lisa_breakfast_eggs_l682_682637

noncomputable def total_eggs_per_year (children : ℕ) (eggs_per_child : ℕ) (husband_eggs : ℕ) (self_eggs : ℕ) (days_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  let eggs_per_day := (children * eggs_per_child) + husband_eggs + self_eggs
  in eggs_per_day * days_per_week * weeks_per_year

theorem lisa_breakfast_eggs :
  total_eggs_per_year 4 2 3 2 5 52 = 3380 :=
by
  sorry

end lisa_breakfast_eggs_l682_682637


namespace smallest_positive_multiple_of_45_is_45_l682_682951

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682951


namespace smallest_positive_multiple_of_45_is_45_l682_682804

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682804


namespace trapezoid_is_planar_l682_682184

theorem trapezoid_is_planar (P Q R: Type) [Point P] [Point Q] [Point R] [Trapezoid T] : 
  (three_points_determine_plane P Q R = false) → 
  (quadrilateral_is_planar QD = false) → 
  (trapezoid_is_planar T = true) → 
  (planes_intersect_at_three_non_collinear_points alpha beta = false) := 
begin
  intros h1 h2 h3 h4,
  sorry
end

end trapezoid_is_planar_l682_682184


namespace triangle_area_l682_682369

theorem triangle_area (base height : ℝ) (h_base : base = 8.4) (h_height : height = 5.8) :
  0.5 * base * height = 24.36 := by
  sorry

end triangle_area_l682_682369


namespace sign_of_b_l682_682725

variable (a b : ℝ)

theorem sign_of_b (h1 : (a + b > 0 ∨ a - b > 0) ∧ (a + b < 0 ∨ a - b < 0)) 
                  (h2 : (ab > 0 ∨ a / b > 0) ∧ (ab < 0 ∨ a / b < 0))
                  (h3 : (ab > 0 → a > 0 ∧ b > 0) ∨ (ab < 0 → (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0))) :
  b < 0 :=
sorry

end sign_of_b_l682_682725


namespace geometric_progression_infinite_sum_l682_682415

variables (a1 q : ℝ)
variables (h_q : |q| < 1)

theorem geometric_progression_infinite_sum :
  (∑' n : ℕ, a1 * ∑ i in Finset.range (n+1), q^i) = a1 / (1 - q)^2 :=
by
  sorry

end geometric_progression_infinite_sum_l682_682415


namespace incorrect_direct_proof_reliance_on_intermediate_l682_682213

-- Definitions based on conditions:
def PostulatesUtilizedWithoutProof : Prop := 
  ∀ (p : Prop), Postulates.contains(p) → ¬ provable(p)

def DifferentApproachesToProof : Prop :=
  ∀ (thm : Prop), ∃ (approach₁ approach₂ : ProofApproach), 
    (approach₁ ≠ approach₂) → provable_by(thm, approach₁) ∧ provable_by(thm, approach₂)

def AllTermsClearlyDefined : Prop :=
  ∀ (term : Term), ∃ (defn : Definition), Definition.contains(defn, term)

def CorrectConclusionFromFalsePremise : Prop :=
  ∀ (p q : Prop), (¬p) → (p → q) → ¬ q

def DirectProofNoIntermediateLemma : Prop :=
  ∃ (thm : Prop), provable(thm) ∧ ¬ ∃ (lemma : Prop), provable(lemma) ∧ (lemma → thm)

-- Statement to prove:
theorem incorrect_direct_proof_reliance_on_intermediate : ¬ DirectProofNoIntermediateLemma :=
by sorry

end incorrect_direct_proof_reliance_on_intermediate_l682_682213


namespace equation_of_ellipse_max_area_of_triangle_pab_l682_682126

-- Definitions of the conditions
def ecc : ℝ := sqrt 6 / 3
def a_gt_b_gt_0 : Prop := ∃ a b : ℝ, a > b ∧ b > 0 ∧ (a = sqrt 3) ∧ (ecc = sqrt 6 / 3) 
def distance_minor_axis_to_right_focus : ℝ := sqrt 3

-- Main statements to prove
theorem equation_of_ellipse (a b c : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c = sqrt 2) 
  (h₄ : a = sqrt 3) (h₅ : ecc = sqrt 6 / 3) (h₆ : b = sqrt (a^2 - c^2)) : (0 < a) ∧ (0 < b) ∧ (a^2 - c^2 = b^2) → 
  ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1} ↔ x^2 / 3 + y^2 = 1 := by
    sorry

theorem max_area_of_triangle_pab {a b : ℝ} (P : ℝ → ℝ × ℝ) (h₁ : a = sqrt 3) (h₂ : b = 1) 
  (h₃ : ∀ θ, P θ = (sqrt 3 * cos θ, sin θ)) : 
  (∀ Q : ℝ × ℝ, ((Q.1^2 / a^2 + Q.2^2 / b^2 = 1) → Q.2 = Q.1 + 1) → 
  let A := (0 : ℝ, 1 : ℝ), B := (-3 / 2, -1 / 2) in
  let d := (λ θ, (abs (sqrt 3 * cos θ - sin θ + 1)) / sqrt 2) in
  let max_d := (λ θ, d (-π / 6)) 
  in 1 / 2 * (3 / 2 * sqrt 2) * max_d = 9 / 4) := by
    sorry

end equation_of_ellipse_max_area_of_triangle_pab_l682_682126


namespace range_of_a_for_symmetric_points_l682_682163

open Real

noncomputable def symmetric_point_condition (a : ℝ) : Prop :=
  ∃ m n : ℝ, 
    (1 / 2) * exp (2 * m) + a = n ∧ 
    log n = m

theorem range_of_a_for_symmetric_points : 
  ∀ (a : ℝ), symmetric_point_condition a ↔ a ∈ Iic (1 / 2) := 
by sorry

end range_of_a_for_symmetric_points_l682_682163


namespace num_terms_in_expansion_equals_7_l682_682038

noncomputable def num_terms_expansion (a b : ℕ) : ℕ :=
  let expr := (a^2 - 4 * b^2) in
  (expr ^ 6).coeffs.length

theorem num_terms_in_expansion_equals_7 (a b : ℕ) : num_terms_expansion a b = 7 :=
  by
    sorry

end num_terms_in_expansion_equals_7_l682_682038


namespace nina_not_taller_than_lena_l682_682088

noncomputable def friends_heights := ℝ 
variables (M N L O : friends_heights)

def nina_shorter_than_masha (N M : friends_heights) : Prop := N < M
def lena_taller_than_olya (L O : friends_heights) : Prop := L > O
def nina_taller_than_lena (N L : friends_heights) : Prop := N > L

theorem nina_not_taller_than_lena (N M L O : friends_heights) 
  (h₁ : nina_shorter_than_masha N M) 
  (h₂ : lena_taller_than_olya L O) : 
  (0 : ℝ) = 0 :=
sorry

end nina_not_taller_than_lena_l682_682088


namespace math_expression_equivalent_l682_682026

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end math_expression_equivalent_l682_682026


namespace range_of_a_l682_682507

variables {x a : ℝ}

def p : Prop := abs(x + 1) ≥ 1
def q : Prop := x ≤ a

theorem range_of_a (h: ∀ x, p → q ∧ ¬(q → p)) : a ≤ -2 :=
sorry

end range_of_a_l682_682507


namespace smallest_positive_multiple_of_45_l682_682961

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682961


namespace smallest_positive_multiple_of_45_l682_682822

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682822


namespace compute_α_l682_682244

open Complex

def α : ℂ := 6 - 3 * Complex.i
def β : ℂ := 4 + 3 * Complex.i

theorem compute_α (h1 : ∃ x : ℝ, (α + β) = x ∧ 0 < x)
                  (h2 : ∃ z : ℝ, (Complex.i * (α - 3 * β)) = z ∧ 0 < z) :
  α = 6 - 3 * Complex.i :=
by
  sorry

end compute_α_l682_682244


namespace prime_sum_divisors_l682_682075

theorem prime_sum_divisors (p : ℕ) (s : ℕ) : 
  (2 ≤ s ∧ s ≤ 10) → 
  (p = 2^s - 1) → 
  (p = 3 ∨ p = 7 ∨ p = 31 ∨ p = 127) :=
by
  intros h1 h2
  sorry

end prime_sum_divisors_l682_682075


namespace smallest_positive_multiple_45_l682_682809

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682809


namespace polynomial_identity_l682_682392

theorem polynomial_identity (p : Polynomial ℝ ℝ × ℝ) :
  (∀ t : ℝ, p (cos t, sin t) = 0) → ∃ q : Polynomial ℝ ℝ × ℝ, p = (X^2 + Y^2 - 1) * q :=
by sorry

end polynomial_identity_l682_682392


namespace prove_sphere_velocities_l682_682423

-- Define the setup where two spheres are moving towards the vertex of the right angle
def sphere1_radius := 2
def sphere2_radius := 3
def initial_distance_small := 6
def initial_distance_large := 16

-- Define constants for time intervals given in the problem
def time_1_sec := 1
def time_3_sec := 3

-- Constant distance between centers after 1 second
def distance_centers_1_sec := 13

-- Function to calculate the distance between centers after a given time
def distance (x y : ℝ) (t : ℝ) : ℝ :=
  real.sqrt ((initial_distance_small - x * t)^2 + (initial_distance_large - y * t)^2)

-- Define the target velocities to prove
def velocitySmall := 1
def velocityLarge := 4

-- Prove that given the conditions, the velocities must be velocitySmall and velocityLarge
theorem prove_sphere_velocities : 
  ∃ (x y : ℝ), 
  (∀ t, 
    if t = time_1_sec 
    then distance x y t = distance_centers_1_sec 
    else if t = time_3_sec 
    then distance x y t = sphere1_radius + sphere2_radius 
    else true) ∧
  x = velocitySmall ∧ y = velocityLarge :=
by
  sorry

end prove_sphere_velocities_l682_682423


namespace calculate_expression_l682_682007

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculate_expression_l682_682007


namespace smallest_positive_multiple_of_45_l682_682832

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682832


namespace derivative_y_l682_682481

-- Define the function y
def y (x : ℝ) : ℝ :=
  Real.cot (Real.cos 5) - (1 / 40) * (Real.cos (20 * x))^2 / (Real.sin (40 * x))

-- Theorem statement to prove the derivative of y
theorem derivative_y (x : ℝ) : (deriv y x) = 1 / (4 * (Real.sin (20 * x))^2) :=
by
  sorry

end derivative_y_l682_682481


namespace max_m_x_range_l682_682508

variables {a b x : ℝ}

theorem max_m (h1 : a * b > 0) (h2 : a^2 * b = 4) : 
  a + b ≥ 3 :=
sorry

theorem x_range (h : 2 * |x - 1| + |x| ≤ 3) : 
  -1/3 ≤ x ∧ x ≤ 5/3 :=
sorry

end max_m_x_range_l682_682508


namespace smallest_positive_multiple_of_45_is_45_l682_682799

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682799


namespace probability_of_green_ball_l682_682461

-- Define the number of balls in each container
def number_balls_I := (10, 5)  -- (red, green)
def number_balls_II := (3, 6)  -- (red, green)
def number_balls_III := (3, 6)  -- (red, green)

-- Define the probability of selecting each container
noncomputable def probability_container_selected := (1 / 3 : ℝ)

-- Define the probability of drawing a green ball from each container
noncomputable def probability_green_I := (number_balls_I.snd : ℝ) / ((number_balls_I.fst + number_balls_I.snd) : ℝ)
noncomputable def probability_green_II := (number_balls_II.snd : ℝ) / ((number_balls_II.fst + number_balls_II.snd) : ℝ)
noncomputable def probability_green_III := (number_balls_III.snd : ℝ) / ((number_balls_III.fst + number_balls_III.snd) : ℝ)

-- Define the combined probabilities for drawing a green ball and selecting each container
noncomputable def combined_probability_I := probability_container_selected * probability_green_I
noncomputable def combined_probability_II := probability_container_selected * probability_green_II
noncomputable def combined_probability_III := probability_container_selected * probability_green_III

-- Define the total probability of drawing a green ball
noncomputable def total_probability_green := combined_probability_I + combined_probability_II + combined_probability_III

-- The theorem to be proved
theorem probability_of_green_ball : total_probability_green = (5 / 9 : ℝ) :=
by
  sorry

end probability_of_green_ball_l682_682461


namespace exists_bn_sequence_l682_682395

theorem exists_bn_sequence (a : ℕ → ℝ) (epsilon : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_div : ¬(∑' n, (a n)^2).convergent)
  (h_ep : 0 < epsilon ∧ epsilon < 1/2) :
  ∃ b : ℕ → ℝ, (∀ n, 0 < b n) ∧ (∑' n, (b n)^2).convergent ∧ (∀ N, (∑ n in finset.range N, a n * b n) > (∑ n in finset.range N, (a n)^2)^(1/2 - epsilon)) :=
sorry

end exists_bn_sequence_l682_682395


namespace num_ints_satisfying_ineq_l682_682037

theorem num_ints_satisfying_ineq :
  (∃ S : Set ℤ, S = {n | (n + 5) * (n - 6) ≤ 0} ∧ S.card = 12) :=
by
  sorry

end num_ints_satisfying_ineq_l682_682037


namespace smallest_positive_multiple_of_45_l682_682772

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682772


namespace four_diff_digits_per_day_l682_682321

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end four_diff_digits_per_day_l682_682321


namespace minimal_distance_sum_l682_682029

-- Define the coordinates of the points on a line
variable (P : Fin 8 → ℝ)

-- Define Q as the point we are looking for
noncomputable def Q := (P 3 + P 4) / 2

-- Define the function to calculate the sum of distances from Q to all points
def sum_of_distances (Q : ℝ) : ℝ :=
  ∑ i in Finset.finRange 8, abs (Q - P i)

-- The theorem to prove
theorem minimal_distance_sum : 
  ∀ Q', sum_of_distances P Q' ≥ sum_of_distances P Q := by sorry

end minimal_distance_sum_l682_682029


namespace jorge_total_ticket_cost_is_161_16_l682_682231

noncomputable def total_cost (adult_tickets senior_tickets child_tickets : ℕ) (adult_price senior_price child_price : ℝ) : ℝ :=
  adult_tickets * adult_price + senior_tickets * senior_price + child_tickets * child_price

noncomputable def discount_tier (total : ℝ) (adult_cost senior_cost : ℝ) : ℝ × ℝ :=
  if total >= 300 then (adult_cost * 0.70, senior_cost * 0.85)
  else if total >= 200 then (adult_cost * 0.80, senior_cost * 0.90)
  else if total >= 100 then (adult_cost * 0.90, senior_cost * 0.95)
  else (adult_cost, senior_cost)

noncomputable def extra_discount (cost_after_tier_discounts : ℝ) : ℝ :=
  let discount_rate := min ((cost_after_tier_discounts / 50).to_nat * 5) 15 in
  cost_after_tier_discounts * discount_rate / 100

noncomputable def final_cost (adult_tickets senior_tickets child_tickets : ℕ) 
  (adult_price senior_price child_price : ℝ) (child_cost : ℝ) : ℝ :=
  let original_total := total_cost adult_tickets senior_tickets child_tickets adult_price senior_price child_price in
  let (new_adult_cost, new_senior_cost) := discount_tier original_total (adult_tickets * adult_price) (senior_tickets * senior_price) in
  let tier_discounted_total := new_adult_cost + new_senior_cost + child_cost in
  tier_discounted_total - extra_discount tier_discounted_total
  
theorem jorge_total_ticket_cost_is_161_16 : 
  final_cost 10 8 6 12 8 6 36 = 161.16 :=
by
  sorry

end jorge_total_ticket_cost_is_161_16_l682_682231


namespace arrangement_count_l682_682516

noncomputable def count_arrangements : ℕ := 2304

theorem arrangement_count (B G : ℕ) (hB : B = 4) (hG : G = 3) (hAdj : ∀ i : ℕ, i < B → adjacent i G = 1) : 
  count_arrangements = 2304 :=
sorry

end arrangement_count_l682_682516


namespace part1_part2_l682_682993

-- Part 1: Prove that (√2 - 1)x + 1 < √(x + 1) < √2 for 0 < x < 1
theorem part1 (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) : 
  (√2 - 1) * x + 1 < sqrt (x + 1) ∧ sqrt (x + 1) < √2 := 
by {
  sorry
}

-- Part 2: Find lim_{a → 1⁻} (∫ a to 1 x * √(1 - x²) dx) / ((1 - a)³/²) = 2√2 / 3
theorem part2 : 
  tendsto (λ (a : ℝ), (∫ (x : ℝ) in a..1, x * sqrt (1 - x^2)) / (1 - a)^(3/2)) (𝓝[<] 1) (𝓝 (2 * sqrt 2 / 3)) :=
by {
  sorry
}

end part1_part2_l682_682993


namespace smallest_positive_multiple_45_l682_682810

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682810


namespace max_n_sum_positive_l682_682189

variable {a_n : ℕ → ℝ}  -- Define the arithmetic sequence a_n

-- Define conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) := ∃ d, ∀ n, a_n (n + 1) = a_n n + d

-- Conditions given in the problem
def conditions (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  a_n 1 > 0 ∧
  a_n 5 + a_n 6 > 0 ∧
  a_n 5 * a_n 6 < 0 ∧
  (a_n (n + 1) = a_n n + d)

theorem max_n_sum_positive (a_n : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a_n →
  conditions a_n d →
  ∃ n, n = 10 ∧ ∀ m, m < n → (∑ i in range m, a_n i) > 0 :=
by
  intros h_arith h_cond
  sorry

end max_n_sum_positive_l682_682189


namespace problem_statement_l682_682004

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end problem_statement_l682_682004


namespace smallest_positive_multiple_45_l682_682898

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682898


namespace ratio_students_above_8_to_8_years_l682_682202

-- Definitions of the problem's known conditions
def total_students : ℕ := 125
def students_below_8_years : ℕ := 25
def students_of_8_years : ℕ := 60

-- Main proof inquiry
theorem ratio_students_above_8_to_8_years :
  ∃ (A : ℕ), students_below_8_years + students_of_8_years + A = total_students ∧
             A * 3 = students_of_8_years * 2 := 
sorry

end ratio_students_above_8_to_8_years_l682_682202


namespace polynomial_expansion_sum_l682_682185

theorem polynomial_expansion_sum :
  let f := (1 - 2 * x)^2012 in
  let a := (λ n : ℕ, polynomial.coeff f n) in
  (∑ i in finset.range 2012, (a i + a (i+1))) = 1 - 2^2012 :=
by
  sorry

end polynomial_expansion_sum_l682_682185


namespace domain_of_f_l682_682466

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x - 1) + real.sqrt (8 - x)

theorem domain_of_f : ∀ x, (1 ≤ x ∧ x ≤ 8) ↔ (∃ y, f y = x) :=
by
  intro x
  split
  { intro h,
    cases h with h1 h2,
    use x,
    exact h1,
    exact h2 },
  { intro h,
    cases h with y hy,
    use x,
    exact hy.left,
    exact hy.right }
  sorry

end domain_of_f_l682_682466


namespace clock_displays_unique_digits_minutes_l682_682308

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end clock_displays_unique_digits_minutes_l682_682308


namespace part1_part2_l682_682536

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x + (1 / 2) * x ^ 2 - a * x + a

theorem part1 (a : ℝ) :
  (∀ x > 0, (1 / x + x - a) ≥ 0) → a ≤ 2 := 
sorry

theorem part2 (a x1 x2 : ℝ) (e : ℝ) (h_e : e > 0) (h_x : x2 ≥ e * x1) (h_ext1 : f x1 a = 0) (h_ext2 : f x2 a = 0) :
  f x2 a - f x1 a = 1 - e / 2 + 1 / (2 * e) :=
sorry


end part1_part2_l682_682536


namespace polynomial_degree_l682_682061

theorem polynomial_degree :
  degree ((X^3 + 2)^5 * (X^2 + 3*X + 1)^4) = 23 :=
by sorry

end polynomial_degree_l682_682061


namespace sum_of_consecutive_integers_product_is_negative_336_l682_682708

theorem sum_of_consecutive_integers_product_is_negative_336 :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) = -336 ∧ (n - 1) + n + (n + 1) = -21 :=
by
  sorry

end sum_of_consecutive_integers_product_is_negative_336_l682_682708


namespace correct_expression_l682_682430

theorem correct_expression (n : ℕ) : n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 → (2 * n - 1) = if n = 1 then 1 else if n = 2 then 3 else if n = 3 then 5 else if n = 4 then 7 else 9 :=
by
  intros h
  cases h
  { -- case n = 1
    rw h
    simp
  }
  cases h
  { -- case n = 2
    rw h
    simp
  }
  cases h
  { -- case n = 3
    rw h
    simp
  }
  cases h
  { -- case n = 4
    rw h
    simp
  }
  { -- case n = 5
    rw h
    simp
  }
  sorry -- Handle any other cases

end correct_expression_l682_682430


namespace solve_system_l682_682581

theorem solve_system :
  ∃ (x y : ℕ), 
    (∃ d : ℕ, d ∣ 42 ∧ x^2 + y^2 = 468 ∧ d + (x * y) / d = 42) ∧ 
    (x = 12 ∧ y = 18) ∨ (x = 18 ∧ y = 12) :=
sorry

end solve_system_l682_682581


namespace intersection_A_B_l682_682260

def greatestInt (x: ℝ) : ℤ := ⌊x⌋

def A : Set ℝ := {x : ℝ | x^2 - (greatestInt x) = 2}
def B : Set ℝ := {x : ℝ | x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, Real.sqrt 3} :=
by
  sorry

end intersection_A_B_l682_682260


namespace margarita_vs_ricciana_l682_682655

-- Ricciana's distances
def ricciana_run : ℕ := 20
def ricciana_jump : ℕ := 4
def ricciana_total : ℕ := ricciana_run + ricciana_jump

-- Margarita's distances
def margarita_run : ℕ := 18
def margarita_jump : ℕ := (2 * ricciana_jump) - 1
def margarita_total : ℕ := margarita_run + margarita_jump

-- Statement to prove Margarita ran and jumped 1 more foot than Ricciana
theorem margarita_vs_ricciana : margarita_total = ricciana_total + 1 := by
  sorry

end margarita_vs_ricciana_l682_682655


namespace determine_y_value_l682_682216

theorem determine_y_value {k y : ℕ} (h1 : k > 0) (h2 : y > 0) (hk : k < 10) (hy : y < 10) :
  (8 * 100 + k * 10 + 8) + (k * 100 + 8 * 10 + 8) - (1 * 100 + 6 * 10 + y * 1) = 8 * 100 + k * 10 + 8 → 
  y = 9 :=
by
  sorry

end determine_y_value_l682_682216


namespace smallest_positive_multiple_45_l682_682816

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682816


namespace roots_of_polynomial_l682_682534

theorem roots_of_polynomial {n : ℕ} {a : Fin n → ℝ} {λ : ℂ}
  (h_poly : (Polynomial.monic (λ i, if h : i < n then a ⟨i, h⟩ else 0)) λ = 0)
  (h_coeffs : ∀ i, 0 < a ⟨i, sorry⟩ ∧ a ⟨i, sorry⟩ ≤ 1)
  (h_mod_lambda : abs λ ≥ 1) : λ^(n+1) = 1 :=
sorry

end roots_of_polynomial_l682_682534


namespace smallest_b_gt_4_perfect_square_l682_682738

theorem smallest_b_gt_4_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ k : ℕ, 4 * b + 5 = k^2 ∧ b = 5 :=
by
  sorry

end smallest_b_gt_4_perfect_square_l682_682738


namespace parallelogram_conditions_l682_682494

-- Definitions for conditions
variables {A B C D : Type} [LinearOrder A] [LinearOrder B]
(a : A) (b : B)
variable (AB_eq_CD : A = B)
variable (AD_eq_BC : A = B)
variable (AB_parallel_CD : Prop)
variable (AD_parallel_BC : Prop)

-- Statement to be proved
theorem parallelogram_conditions :
  (∃ cond1 cond2, 
    (cond1 = (AB_eq_CD) ∧ cond2 = (AD_eq_BC)) ∨ 
    (cond1 = (AB_parallel_CD) ∧ cond2 = (AD_parallel_BC)) ∨
    (cond1 = (AD_eq_BC) ∧ cond2 = (AB_parallel_CD)) ∨ 
    (cond1 = (AB_eq_CD) ∧ cond2 = (AD_parallel_BC))) → 
  ∃ n, n = 4 := 
by
  sorry

end parallelogram_conditions_l682_682494


namespace exists_three_digit_number_l682_682473

theorem exists_three_digit_number : ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ (100 * a + 10 * b + c = a^3 + b^3 + c^3) ∧ (100 * a + 10 * b + c ≥ 100 ∧ 100 * a + 10 * b + c < 1000) := 
sorry

end exists_three_digit_number_l682_682473


namespace fill_bathtub_time_l682_682583

theorem fill_bathtub_time (V : ℝ) (cold_rate hot_rate drain_rate net_rate : ℝ) 
  (hcold : cold_rate = V / 10) 
  (hhot : hot_rate = V / 15) 
  (hdrain : drain_rate = -V / 12) 
  (hnet : net_rate = cold_rate + hot_rate + drain_rate) 
  (V_eq : V = 1) : 
  1 / net_rate = 12 :=
by {
  -- placeholder for proof steps
  sorry
}

end fill_bathtub_time_l682_682583


namespace original_price_l682_682728

theorem original_price (price_paid original_price : ℝ) 
  (h₁ : price_paid = 5) 
  (h₂ : price_paid = original_price / 10) : 
  original_price = 50 := by
  sorry

end original_price_l682_682728


namespace smallest_positive_multiple_of_45_l682_682764

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682764


namespace simplify_expression_l682_682671

variable {m : ℝ} (hm : m ≠ 0)

theorem simplify_expression (m : ℝ) (h : m ≠ 0) : 
  ( (1 / (3 * m)) ^ (-3) * (2 * m) ^ 4 ) = 432 * m ^ 7 := by
  sorry

end simplify_expression_l682_682671


namespace focal_distance_equation_of_ellipse_l682_682623

noncomputable def ellipse_c (a b : ℝ) := { p : ℝ × ℝ // (p.1^2 / a^2) + (p.2^2 / b^2) = 1 }

def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := real.sqrt (a^2 - b^2) in ((-c, 0), (c, 0))

theorem focal_distance (a b : ℝ) (h1 : a > b) (h2 : b > 0) (d : ℝ) (h_dist : d = 2)
  (θ : ℝ) (hθ : θ = real.pi / 3) :
  let c := real.sqrt (a^2 - b^2) 
  in 2 * c = 4 :=
by
  sorry

theorem equation_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (hb : b = 2)
  (d : ℝ) (h_dist : d = 2)
  (θ : ℝ) (hθ : θ = real.pi / 3) :
  a^2 = 9 ∧ b^2 = 5 ∧ (∀ p : ℝ × ℝ, (p.1^2 / a^2) + (p.2^2 / b^2) = 1) :=
by
  sorry

end focal_distance_equation_of_ellipse_l682_682623


namespace zan_stops_in_less_than_b_minus_a_seconds_l682_682981

theorem zan_stops_in_less_than_b_minus_a_seconds 
  (a b : ℕ) (h : b > a) (hlt: Nat.gcd a b = 1)
  (transformation : ∀ x y : ℕ, (x, y) → (x + 1, y + 1)) :
  ∃ t < b - a, ∃ n, (transformation ^ t) (a, b) = (n, n+1) :=
sorry

end zan_stops_in_less_than_b_minus_a_seconds_l682_682981


namespace smallest_positive_multiple_of_45_is_45_l682_682805

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682805


namespace radius_moon_scientific_notation_l682_682339

def scientific_notation := 1738000 = 1.738 * 10^6

theorem radius_moon_scientific_notation : scientific_notation := 
sorry

end radius_moon_scientific_notation_l682_682339


namespace pet_store_dogs_l682_682343

theorem pet_store_dogs (cats dogs : ℕ) (h1 : 18 = cats) (h2 : 3 * dogs = 4 * cats) : dogs = 24 :=
by
  sorry

end pet_store_dogs_l682_682343


namespace not_true_option_c_given_x_lt_y_l682_682114

variable (x y : ℝ)

theorem not_true_option_c_given_x_lt_y (h : x < y) : 
  (x - 2 < y - 2) ∧ (3 * x + 1 < 3 * y + 1) ∧ (x / 3 < y / 3) ∧ ¬(-2 * x < -2 * y) :=
by
  apply And.intro
  . exact (iff.mpr (sub_lt_sub_iff_right 2) h)
  apply And.intro
  . exact (iff.mpr (add_lt_add_iff_right 1) (mul_lt_mul_of_pos_left h (by norm_num)))
  apply And.intro
  . exact (iff.mpr (div_lt_div_iff (by norm_num : (0:ℝ)<3) (by norm_num : 0<3)) h)
  . exact (not_lt_of_ge (iff.mp (neg_le_neg_iff) h))

end not_true_option_c_given_x_lt_y_l682_682114


namespace range_of_a_l682_682145

theorem range_of_a (f : ℝ → ℝ) (h_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2) (h_ineq : f (1 - a) < f (2 * a - 1)) : a < 2 / 3 :=
sorry

end range_of_a_l682_682145


namespace unique_solution_l682_682475

theorem unique_solution :
  ∀ (x y z n : ℕ), n ≥ 2 → z ≤ 5 * 2^(2 * n) → (x^ (2 * n + 1) - y^ (2 * n + 1) = x * y * z + 2^(2 * n + 1)) → (x, y, z, n) = (3, 1, 70, 2) :=
by
  intros x y z n hn hzn hxyz
  sorry

end unique_solution_l682_682475


namespace smallest_n_correct_l682_682249

noncomputable def smallest_n (n : ℕ) (x : fin n → ℝ) : Prop :=
(x i : ℕ → ℝ) (h₁ : ∀ i, 0 ≤ x i)
  (h₂ : ∑ i, x i = 1)
  (h₃ : ∑ i, (x i) ^ 2 ≤ 1 / 400) := n = 400

theorem smallest_n_correct : ∃ n (x: fin n → ℝ), smallest_n n x := sorry

end smallest_n_correct_l682_682249


namespace average_book_width_l682_682643

theorem average_book_width :
  let widths := [5, 0.75, 1.25, 3, 11, 0.5]
  let n := 6
  (List.sum widths / n) = 3.58 :=
by
  let widths := [5, 0.75, 1.25, 3, 11, 0.5]
  let n := 6
  have h_sum : List.sum widths = 21.5 := by sorry
  have h_avg : List.sum widths / n = 21.5 / 6 := by
    rw [h_sum]
  exact eq.trans h_avg (by norm_num)

end average_book_width_l682_682643


namespace smallest_positive_multiple_of_45_l682_682893

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682893


namespace abs_expression_simplified_l682_682446

theorem abs_expression_simplified (e : ℝ) (h : e < 5) : |e - |e - 5|| = 2 * e - 5 :=
by
  sorry

end abs_expression_simplified_l682_682446


namespace coefficient_of_x_90_in_my_polynomial_l682_682480

open Polynomial

-- Definition of the polynomial (x - 1)(x^2 - 2)(x^3 - 3) ... (x^13 - 13)
noncomputable def my_polynomial : Polynomial ℝ :=
  ∏ i in (finset.range 13).map (λ n, n + 1),
    (X^(n : ℕ) - (n : ℝ))

-- The goal is to find the coefficient of x^90 in this polynomial
theorem coefficient_of_x_90_in_my_polynomial :
  coeff my_polynomial 90 = -1 :=
by
  sorry

end coefficient_of_x_90_in_my_polynomial_l682_682480


namespace sum_final_numbers_l682_682356

theorem sum_final_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 :=
by
  sorry

end sum_final_numbers_l682_682356


namespace possible_sums_l682_682261

open Finset

def is_100_element_subset (A : Finset ℕ) : Prop :=
  A.card = 100 ∧ A ⊆ (finset.range 121 \ {0})

theorem possible_sums (A : Finset ℕ)
  (hA : is_100_element_subset A) :
  ∃ n, ∀ S, (S = A.sum id) ↔ n = 2001 :=
sorry

end possible_sums_l682_682261


namespace smallest_positive_multiple_of_45_is_45_l682_682796

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682796


namespace roberto_outfits_l682_682660

theorem roberto_outfits : 
  let trousers := 5
  let shirts := 5
  let jackets := 3
  (trousers * shirts * jackets = 75) :=
by sorry

end roberto_outfits_l682_682660


namespace max_n_for_factored_quadratic_l682_682067

theorem max_n_for_factored_quadratic :
  ∃ (A B : ℤ), A * B = 54 ∧ (3 * B + A) = 163 :=
by
  use 1, 54
  split
  -- Proof part skipped
  sorry -- A * B = 54
  sorry -- 3 * B + A = 163

end max_n_for_factored_quadratic_l682_682067


namespace calculate_expression_l682_682010

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculate_expression_l682_682010


namespace smallest_positive_multiple_45_l682_682910

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682910


namespace camden_dogs_fraction_l682_682443

def number_of_dogs (Justins_dogs : ℕ) (extra_dogs : ℕ) : ℕ := Justins_dogs + extra_dogs
def dogs_from_legs (total_legs : ℕ) (legs_per_dog : ℕ) : ℕ := total_legs / legs_per_dog
def fraction_of_dogs (dogs_camden : ℕ) (dogs_rico : ℕ) : ℚ := dogs_camden / dogs_rico

theorem camden_dogs_fraction (Justins_dogs : ℕ) (extra_dogs : ℕ) (total_legs_camden : ℕ) (legs_per_dog : ℕ) :
  Justins_dogs = 14 →
  extra_dogs = 10 →
  total_legs_camden = 72 →
  legs_per_dog = 4 →
  fraction_of_dogs (dogs_from_legs total_legs_camden legs_per_dog) (number_of_dogs Justins_dogs extra_dogs) = 3 / 4 :=
by
  sorry

end camden_dogs_fraction_l682_682443


namespace slope_angle_range_l682_682169

theorem slope_angle_range (k θ : ℝ) (h_intersect : ∃ x y, y = k * x - sqrt 3 ∧ x + y = 3 ∧ x > 0 ∧ y > 0) :
  θ ∈ Set.Ioo (real.arctan (sqrt 3 / 3)) (real.arctan (real.pi / 2)) → k = real.tan θ := sorry

end slope_angle_range_l682_682169


namespace nina_not_taller_than_lena_l682_682085

noncomputable def friends_heights := ℝ 
variables (M N L O : friends_heights)

def nina_shorter_than_masha (N M : friends_heights) : Prop := N < M
def lena_taller_than_olya (L O : friends_heights) : Prop := L > O
def nina_taller_than_lena (N L : friends_heights) : Prop := N > L

theorem nina_not_taller_than_lena (N M L O : friends_heights) 
  (h₁ : nina_shorter_than_masha N M) 
  (h₂ : lena_taller_than_olya L O) : 
  (0 : ℝ) = 0 :=
sorry

end nina_not_taller_than_lena_l682_682085


namespace difference_of_x_values_l682_682190

theorem difference_of_x_values : 
  ∀ x y : ℝ, ( (x + 3) ^ 2 / (3 * x + 29) = 2 ∧ (y + 3) ^ 2 / (3 * y + 29) = 2 ) → |x - y| = 14 := 
sorry

end difference_of_x_values_l682_682190


namespace math_problem_l682_682517

noncomputable def problem_statement (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hxyz : x * y * z = 1) : Prop :=
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + x) * (1 + z)) + z^3 / ((1 + x) * (1 + y))) ≥ 3 / 4

theorem math_problem (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hxyz : x * y * z = 1) :
  problem_statement x y z hx hy hz hxyz :=
sorry

end math_problem_l682_682517


namespace smallest_positive_multiple_45_l682_682819

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682819


namespace max_min_x31_l682_682144

theorem max_min_x31 :
  ∃ (x : ℕ -> ℕ) (x31_max x31_min : ℕ),
    (∀ n, 1 ≤ x n) ∧
    (∀ (n m : ℕ), n < m -> x n < x m) ∧
    (∑ n in finset.range 31, x n.succ) = 2009 ∧
    x 31 = x31_max ∧ x31_max = 1544 ∧
    x 31 = x31_min ∧ x31_min = 80 :=
by {
  sorry -- proof would go here
}

end max_min_x31_l682_682144


namespace number_of_divisors_of_36_l682_682565

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end number_of_divisors_of_36_l682_682565


namespace min_daily_tourism_revenue_l682_682402

def f (t : ℕ) : ℚ := 4 + 1 / t

def g (t : ℕ) : ℚ := 125 - |t - 25|

def W (t : ℕ) : ℚ :=
  if 1 ≤ t ∧ t ≤ 25 then 401 + 4 * t + 100 / t
  else if 25 < t ∧ t ≤ 30 then 599 + 150 / t - 4 * t
  else 0  -- This will never be used since t ∈ [1, 30]

theorem min_daily_tourism_revenue :
  ∀ t, 1 ≤ t ∧ t ≤ 30 → 441 ≤ W t :=
by
  sorry

end min_daily_tourism_revenue_l682_682402


namespace complex_square_l682_682579

theorem complex_square (z : ℂ) (hz : z = 2 + 3 * complex.I) : z^2 = -5 + 12 * complex.I := by
  sorry

end complex_square_l682_682579


namespace smallest_positive_multiple_of_45_l682_682751

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682751


namespace smallest_three_digit_number_l682_682401

theorem smallest_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧
  (x % 2 = 0) ∧
  ((x + 1) % 3 = 0) ∧
  ((x + 2) % 4 = 0) ∧
  ((x + 3) % 5 = 0) ∧
  ((x + 4) % 6 = 0) ∧
  x = 122 :=
by
  sorry

end smallest_three_digit_number_l682_682401


namespace residents_attended_banquet_l682_682997

theorem residents_attended_banquet :
  ∃ R N : ℕ,
  R + N = 586 ∧
  12.95 * R + 17.95 * N = 9423.70 ∧
  R = 220 :=
by
  sorry

end residents_attended_banquet_l682_682997


namespace QST_perimeter_eq_l682_682724

-- Given conditions
def PQ : ℝ := 15
def QR : ℝ := 20
def PR : ℝ := 17

-- S is the midpoint of PR
def S_midpoint : ℝ := PR / 2

-- T is a point where a line through the incenter I of triangle PQR and parallel to PQ intersects PR
-- The perimeter of triangle QST needs to be checked
noncomputable def TQ_len := PQ / 2  -- Parallel line implies TQ = PQ / 2
noncomputable def ST_len := TQ_len + S_midpoint - S_midpoint

-- Perimeter of triangle QST
noncomputable def QST_perimeter : ℝ := TQ_len + TQ_len + TQ_len

-- Theorem to be proven
theorem QST_perimeter_eq : QST_perimeter = 22.5 := by
  sorry

end QST_perimeter_eq_l682_682724


namespace range_of_a_l682_682535

noncomputable def f (x : ℤ) (a : ℝ) := (3 * x^2 + a * x + 26) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x : ℕ+, f x a ≤ 2) → a ≤ -15 :=
by
  sorry

end range_of_a_l682_682535


namespace remainder_777_777_mod_13_l682_682374

theorem remainder_777_777_mod_13 : (777 ^ 777) % 13 = 12 := 
by 
  -- Proof steps would go here
  sorry

end remainder_777_777_mod_13_l682_682374


namespace proof_problem_l682_682155

variable {n : ℕ}

-- Arithmetic Sequence Definition
def arithmetic_seq (a : ℕ → ℤ) := ∀ m n, a (n + 1) = a n + d

-- Conditions of the problem
def conditions (a : ℕ → ℤ) (d : ℤ) :=
  (a 2 = 3) ∧ (a 1 + d = 3) ∧ (a 1 * (2 * a 1 + 7 * d) = (2 * d) ^ 2) ∧ (d > 0)

-- Definition of b_n
def b (a : ℕ → ℤ) (n : ℕ) := (3 : ℚ) / (a n * a (n + 1))

-- Sum S_n of b_n
def S (b : ℕ → ℚ) (n : ℕ) := ∑ i in range n, b i

-- Main theorem statement
theorem proof_problem (a : ℕ → ℤ) (d : ℤ) (b : ℕ → ℚ) :
  (conditions a d) →
  (∀ n, a n = 2 * n - 1) →
  ∀ n, S b n = 3 * n / (2 * n + 1) :=
begin
  intros,
  sorry
end

end proof_problem_l682_682155


namespace clock_four_different_digits_l682_682306

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end clock_four_different_digits_l682_682306


namespace problem_statement_l682_682524

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) :=
  ∀ x y, x ≤ y → f x ≤ f y

noncomputable def isOddFunction (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

noncomputable def isArithmeticSeq (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem problem_statement (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ) (a3 : ℝ):
  isMonotonicIncreasing f →
  isOddFunction f →
  isArithmeticSeq a →
  a 3 = a3 →
  a3 > 0 →
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by
  -- proof will go here
  sorry

end problem_statement_l682_682524


namespace smallest_positive_multiple_of_45_is_45_l682_682945

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682945


namespace smallest_positive_multiple_of_45_is_45_l682_682802

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682802


namespace simplify_expression_l682_682666

variable (m : ℝ) (h : m ≠ 0)

theorem simplify_expression : ( (1/(3*m))^(-3) * (2*m)^(4) ) = 432 * m^(7) := by sorry

end simplify_expression_l682_682666


namespace probability_A_nth_roll_l682_682998

def p (n : ℕ) : ℝ :=
  if n = 1 then 1
  else if n = 2 then 1 / 6
  else 0.5 - 1 / 3 * ((-2 / 3) ^ (n - 2))

theorem probability_A_nth_roll (n : ℕ) : p n = if n = 1 then 1 else if n = 2 then 1 / 6 else 0.5 - 1 / 3 * ((-2 / 3) ^ (n - 2)) := sorry

end probability_A_nth_roll_l682_682998


namespace smallest_positive_multiple_of_45_l682_682928

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682928


namespace smallest_positive_multiple_of_45_l682_682852

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682852


namespace part1_part2_l682_682539

-- Definitions and assumptions based on the problem
def f (x a : ℝ) : ℝ := abs (x - a)

-- Condition (1) with given function and inequality solution set
theorem part1 (a : ℝ) :
  (∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  sorry

-- Condition (2) with the range of m under the previously found value of a
theorem part2 (m : ℝ) :
  (∃ x, f x 2 + f (x + 5) 2 < m) → m > 5 :=
by
  sorry

end part1_part2_l682_682539


namespace domain_of_composed_function_l682_682693

theorem domain_of_composed_function
  (f : ℝ → ℝ)
  (H : ∀ y, 1 < y ∧ y < 4 → ∃ x, y = f x) :
  ∀ x, 2 < x ∧ x < 16 → ∃ y, y = f (log x / log 2) :=
by
  sorry

end domain_of_composed_function_l682_682693


namespace smallest_positive_multiple_45_l682_682808

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682808


namespace Tenisha_remains_with_50_puppies_l682_682682

theorem Tenisha_remains_with_50_puppies
  (total_dogs : ℕ)
  (percentage_female : ℕ)
  (frac_females_giving_birth : ℚ)
  (puppies_per_female_that_give_birth : ℕ)
  (puppies_donated : ℕ) :
  total_dogs = 40 →
  percentage_female = 60 →
  frac_females_giving_birth = 3/4 →
  puppies_per_female_that_give_birth = 10 →
  puppies_donated = 130 →
  (let number_of_females := (percentage_female * total_dogs) / 100 in
   let females_giving_birth := (frac_females_giving_birth * number_of_females) in
   let total_puppies := (females_giving_birth * puppies_per_female_that_give_birth).toNat in
   total_puppies - puppies_donated) = 50 := by
  sorry

end Tenisha_remains_with_50_puppies_l682_682682


namespace problem_a_l682_682384

variable {V : Type} [InnerProductSpace ℝ V]

def radii_increasing_order {A B C D : V}
(r1 r2 r3 r4 : ℝ) (h1 : r1 ≤ r2) (h2 : r2 ≤ r3) (h3 : r3 ≤ r4) 
(convexity : convex_hull ℝ (Set.insert A (Set.insert B (Set.insert C {D}))) ⊆ {x | ∃ a b c d, a + b + c + d = 1 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ a • A + b • B + c • C + d • D = x}) : Prop :=
  ∀ ABC BCD CDA DAB : triangle ℝ V, 
  inscribed_circle_radius ABC = r1 →
  inscribed_circle_radius BCD = r2 →
  inscribed_circle_radius CDA = r3 →
  inscribed_circle_radius DAB = r4 → 
  r4 ≤ 2 * r3

theorem problem_a
  {A B C D : V}
  (r1 r2 r3 r4 : ℝ)
  (h1 : r1 ≤ r2)
  (h2 : r2 ≤ r3)
  (h3 : r3 ≤ r4)
  (convexity : convex_hull ℝ (Set.insert A (Set.insert B (Set.insert C {D}))) ⊆ {x | ∃ a b c d, a + b + c + d = 1 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ a • A + b • B + c • C + d • D = x})
  : radii_increasing_order r1 r2 r3 r4 h1 h2 h3 convexity :=
sorry

end problem_a_l682_682384


namespace conic_section_is_ellipse_l682_682041

theorem conic_section_is_ellipse (x y : ℝ) :
  (sqrt (x^2 + (y - 2)^2) + sqrt ((x - 6)^2 + (y - 4)^2) = 12) →
  ∃ c1 c2 : ℝ, (c1 < 12 ∧ c2 < 12 ∧ 
               sqrt (x^2 + (y - 2)^2) = c1 ∧ 
               sqrt ((x - 6)^2 + (y - 4)^2) = c2 ∧
               (c1 + c2 = 12)) :=
begin
  sorry
end

end conic_section_is_ellipse_l682_682041


namespace parallelogram_sides_l682_682705

-- Define the conditions for the problem
variables {AB BC CD DA : ℝ}
variable acute_angle : ℝ
variable ratio : ℝ

-- Assume the perimeter and angle conditions
def parallelogram_conditions : Prop :=
  (AB = CD) ∧ (BC = DA) ∧ (2 * (AB + BC) = 90) ∧ (acute_angle = 60) ∧ (ratio = 1 / 3)

-- Define the theorem to prove
theorem parallelogram_sides (h : parallelogram_conditions) :
  AB = 15 ∧ BC = 30 :=
sorry

end parallelogram_sides_l682_682705


namespace expression_evaluation_l682_682015

theorem expression_evaluation :
  (π - 1)^0 + 4 * real.sin (real.pi / 4) - real.sqrt 8 + abs (-3) = 4 := 
sorry

end expression_evaluation_l682_682015


namespace smallest_positive_multiple_of_45_l682_682879

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682879


namespace smallest_positive_multiple_of_45_l682_682837

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682837


namespace bob_cookie_price_same_as_jane_l682_682493

theorem bob_cookie_price_same_as_jane
  (r_jane : ℝ)
  (s_bob : ℝ)
  (dough_jane : ℝ)
  (num_jane_cookies : ℕ)
  (price_jane_cookie : ℝ)
  (total_earning_jane : ℝ)
  (num_cookies_bob : ℝ)
  (price_bob_cookie : ℝ) :
  r_jane = 4 ∧
  s_bob = 6 ∧
  dough_jane = 18 * (Real.pi * r_jane^2) ∧
  price_jane_cookie = 0.50 ∧
  total_earning_jane = 18 * 50 ∧
  num_cookies_bob = dough_jane / s_bob^2 ∧
  total_earning_jane = num_cookies_bob * price_bob_cookie →
  price_bob_cookie = 36 :=
by
  intros
  sorry

end bob_cookie_price_same_as_jane_l682_682493


namespace calculate_expression_l682_682008

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculate_expression_l682_682008


namespace smallest_positive_multiple_of_45_l682_682892

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682892


namespace prob_second_grade_deviation_l682_682428

noncomputable def probability_of_second_grade_deviation 
  (p : ℝ) (sample_size : ℕ) (threshold : ℝ) : ℝ :=
  let std_error := Real.sqrt ((p * (1 - p)) / sample_size) in
  let Z1 := (p - threshold - p) / std_error in
  let Z2 := (p + threshold - p) / std_error in
  Real.cdf Z2 - Real.cdf Z1

theorem prob_second_grade_deviation :
  probability_of_second_grade_deviation 0.15 1000 0.02 ≈ 0.9232 :=
by
  sorry

end prob_second_grade_deviation_l682_682428


namespace solve_problem_l682_682537

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  sin (2 * x + π / 3) + cos (2 * x + π / 6) + m * sin (2 * x)

theorem solve_problem :
  (∃ m ∈ ℝ, f (π / 12) m = 2 ∧
    (let m := 1 in
      let fB := f (π / 3 / 2) m in
      fB = sqrt 3 ∧
      (let A_area := sqrt 3 in
        let b := 2 in
        let c := 2 in
        let a := 2 in
        A_area * 2 = sqrt 3 ∧
        a * c = 4 ∧
        a^2 + c^2 = 8 ∧
        a + c = 4 ∧
        (a + b + c = 6)))) :=
sorry

end solve_problem_l682_682537


namespace area_to_paint_l682_682278

def wall_height : ℕ := 10
def wall_length : ℕ := 15
def bookshelf_height : ℕ := 3
def bookshelf_length : ℕ := 5

theorem area_to_paint : (wall_height * wall_length) - (bookshelf_height * bookshelf_length) = 135 :=
by 
  sorry

end area_to_paint_l682_682278


namespace smallest_positive_multiple_of_45_l682_682834

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682834


namespace length_MN_l682_682137

section proof_problem

-- Define a quadrilateral in space.
variables {A B C D M N : Type*}

-- Assume A, B, C, D are points in space (in ℝ^3, for example).
variables [euclidean_space ℝ] {a b c d m n : 𝕜}

-- Assume M and N are midpoints of AB and CD respectively.
def midpoint (p1 p2 : 𝕜) : 𝕜 := (p1 + p2) / 2

-- Define A, B, C, D, M, and N in the context of our problem.
variables (A B C D M N : 𝕜)

-- Conditions:
-- AC = 4, BD = 6
axiom AC_eq_4 : dist A C = 4
axiom BD_eq_6 : dist B D = 6
axiom M_is_midpoint_AB : M = midpoint A B
axiom N_is_midpoint_CD : N = midpoint C D

-- Theorem to prove: The length MN satisfies 1 < MN < 5
theorem length_MN : 1 < dist M N ∧ dist M N < 5 :=
sorry

end proof_problem

end length_MN_l682_682137


namespace total_cost_with_discount_l682_682663

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def bulk_discount_threshold : ℕ := 10
def bulk_discount_amount : ℕ := 5
def num_sandwiches : ℕ := 6
def num_sodas : ℕ := 6

theorem total_cost_with_discount :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost - 
  if num_sandwiches + num_sodas > bulk_discount_threshold then bulk_discount_amount else 0 = 37 := by
  sorry

end total_cost_with_discount_l682_682663


namespace range_of_a_l682_682160

noncomputable def f (x : ℝ) : ℝ := 4 * x + 3 * Real.sin x

theorem range_of_a (a : ℝ) (h : f (1 - a) + f (1 - a^2) < 0) : 1 < a ∧ a < Real.sqrt 2 := sorry

end range_of_a_l682_682160


namespace find_f_l682_682055

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_domain (x : ℝ) : 0 < x → 0 < f x

axiom f_monotone (x y : ℝ) (h : x ≥ y) : (f x ≥ f y) ∨ (f x ≤ f y)

theorem find_f (f : ℝ → ℝ) :
  (∀ x y : ℝ, 0 < x → 0 < y → f(x * y) * f(f y / x) = 1) →
  (∀ x : ℝ, 0 < x → f x = 1 ∨ f x = 1 / x) :=
sorry

end find_f_l682_682055


namespace number_of_dogs_l682_682348

-- Conditions
def ratio_cats_dogs : ℚ := 3 / 4
def number_cats : ℕ := 18

-- Define the theorem to prove
theorem number_of_dogs : ∃ (dogs : ℕ), dogs = 24 :=
by
  -- Proof steps will go here, but we can use sorry for now to skip actual proving.
  sorry

end number_of_dogs_l682_682348


namespace analogical_reasoning_correct_l682_682977

theorem analogical_reasoning_correct (a b c : ℝ) (hc : c ≠ 0) : (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c :=
by
  sorry

end analogical_reasoning_correct_l682_682977


namespace number_of_intersection_points_between_line_and_curve_l682_682334

noncomputable def numberOfIntersectionPoints (a b c d : ℝ) : ℕ :=
  let curve : ℝ → ℝ → ℝ := λ x y, (x - a) * (x - b) - (y - c) * (y - d)
  let line : ℝ → ℝ → ℝ := λ x y, (c - d) * (x - b) - (a - b) * (y - d)
  -- proof skipped
  2

theorem number_of_intersection_points_between_line_and_curve (a b c d : ℝ) :
    (∀ x y : ℝ, (curve a b x y = 0 ↔ line a b x y = 0)) →
    numberOfIntersectionPoints a b c d = 2 :=
by
  sorry

end number_of_intersection_points_between_line_and_curve_l682_682334


namespace simplify_expression_l682_682668

variable {m : ℝ} (hm : m ≠ 0)

theorem simplify_expression : ( (1 / (3 * m)) ^ (-3) * (2 * m) ^ 4 ) = 432 * m ^ 7 := 
by
  sorry

end simplify_expression_l682_682668


namespace books_of_jason_l682_682227

theorem books_of_jason (M J : ℕ) (hM : M = 42) (hTotal : M + J = 60) : J = 18 :=
by
  sorry

end books_of_jason_l682_682227


namespace find_total_votes_l682_682633

variables (p q D : ℝ) (h_pq : p ≠ q)

-- Define the candidates' percentage of votes
-- A_votes and B_votes represent the votes received by candidates A and B respectively
def A_votes := (p / 100) * V
def B_votes := (q / 100) * V

-- Define the difference in votes
def votes_difference := A_votes - B_votes

-- The goal is to find an expression for V in terms of p, q, and D
theorem find_total_votes (h : votes_difference = D) : V = (D * 100) / (p - q) :=
by {
  -- Use the given conditions and derive the solution
  -- Proof steps would go here
  sorry
}

end find_total_votes_l682_682633


namespace pet_store_dogs_l682_682344

theorem pet_store_dogs (cats dogs : ℕ) (h1 : 18 = cats) (h2 : 3 * dogs = 4 * cats) : dogs = 24 :=
by
  sorry

end pet_store_dogs_l682_682344


namespace not_all_roots_real_l682_682236

theorem not_all_roots_real (P : ℝ[X]) : ¬ ∀ x : ℝ, (x^3 * P.eval x + 1 = 0) := 
sorry

end not_all_roots_real_l682_682236


namespace find_a_for_perpendicular_tangent_line_l682_682156

theorem find_a_for_perpendicular_tangent_line :
  ∃ a : ℝ, (∀ x y : ℝ, x = 3 → y = (x+1)/(x-1) →
    ∂ (λ x, (x+1)/(x-1)) x = -1/2 →
    ∃ t (h : t = -1 / (-a)), t = 1) ∧ a = -2 :=
by
  sorry

end find_a_for_perpendicular_tangent_line_l682_682156


namespace average_after_removal_l682_682684

theorem average_after_removal (s : Fin 12 → ℝ) (h_avg : (s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 + s 7 + s 8 + s 9 + s 10 + s 11) / 12 = 90) (h1: ∃ i j, i ≠ j ∧ s i = 80 ∧ s j = 90) : 
  (∑ i in (Finset.filter (λ x, s x ≠ 80 ∧ s x ≠ 90) Finset.univ), s i) / 10 = 91 :=
sorry

end average_after_removal_l682_682684


namespace geometric_progression_fourth_term_l682_682692

theorem geometric_progression_fourth_term
  (a1 a2 a3 a4 : ℝ)
  (h1 : a1 = 5^(1/3))
  (h2 : a2 = 5^(1/5))
  (h3 : a3 = 5^(1/15))
  (h4 : ∀ n : ℕ, a(n + 1) / a(n) = a2 / a1) -- geometric progression
  : a4 = (5^(-1/15) : ℝ) := sorry

end geometric_progression_fourth_term_l682_682692


namespace correct_addition_by_changing_digit_l682_682217

theorem correct_addition_by_changing_digit :
  ∃ (x : ℕ), x = 789 ∧ (x % 100) + 436 + 527 - 10 = 1742 :=
begin
  sorry
end

end correct_addition_by_changing_digit_l682_682217


namespace area_triangle_ACM_l682_682710

-- Define the geometric setup of the problem
variables {A B C D E M : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace M]

-- Lengths of the legs of triangles ABC and ADE
def length_ABC_legs := (18 : ℝ, 10 : ℝ)
def length_ADE_legs := (14 : ℝ, 4 : ℝ)

-- Midpoint condition
def midpoint (A E M : ℝ) : Prop := M = (A + E) / 2

-- Define the area function (a simplified representation for the proof)
def triangle_area (a b : ℝ) : ℝ := 0.5 * a * b

-- Theorem statement to find the area of triangle ACM
theorem area_triangle_ACM : midpoint 18 14 16 → triangle_area 18 10 + triangle_area 14 4 = 118 ∧
  (∀ x, x = triangle_area (8/7) 4) ∧
  (∀ x, x = triangle_area (20/7) 10) →
  triangle_area 18 10 + triangle_area 14 4 + (16/7) - (100/7) = 106 →
  (1 / 2) * 106 = 53 :=
by
  sorry

end area_triangle_ACM_l682_682710


namespace converse_negation_contrapositive_l682_682979

variable {x : ℝ}

def P (x : ℝ) : Prop := x^2 - 3 * x + 2 ≠ 0
def Q (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ 2

theorem converse (h : Q x) : P x := by
  sorry

theorem negation (h : ¬ P x) : ¬ Q x := by
  sorry

theorem contrapositive (h : ¬ Q x) : ¬ P x := by
  sorry

end converse_negation_contrapositive_l682_682979


namespace number_of_terms_AP_is_10_l682_682702

noncomputable def find_num_of_terms (a n d : ℕ) (sum_odd sum_even last_diff: ℤ) : Prop :=
  (n % 2 = 0) ∧ -- number of terms is even
  (sum_odd = 56) ∧ -- sum of odd-numbered terms
  (sum_even = 80) ∧ -- sum of even-numbered terms
  (last_diff = 18) ∧ -- difference between last and first term
  ((n-1) * d = 18) ∧ -- (n-1)d = 18
  (n * (a + (n-2) * d) = 112) ∧ -- equation for the sum of odd-numbered terms
  (n * (a + (n-1) * d + d) = 160) -- equation for the sum of even-numbered terms
  
theorem number_of_terms_AP_is_10 : 
  ∀ (a n d : ℕ) (sum_odd sum_even last_diff: ℤ),
  find_num_of_terms a n d sum_odd sum_even last_diff -> n = 10 :=
by 
  intro a n d sum_odd sum_even last_diff h,
  sorry

end number_of_terms_AP_is_10_l682_682702


namespace bottles_left_l682_682420

-- Define initial conditions
def bottlesInRefrigerator : Nat := 4
def bottlesInPantry : Nat := 4
def bottlesBought : Nat := 5
def bottlesDrank : Nat := 3

-- Goal: Prove the total number of bottles left
theorem bottles_left : bottlesInRefrigerator + bottlesInPantry + bottlesBought - bottlesDrank = 10 :=
by
  sorry

end bottles_left_l682_682420


namespace bounded_variation_l682_682620

theorem bounded_variation {f : ℝ → ℝ}
  (h1 : ∀ x ≥ 1, f x = ∫ t in (x - 1)..x, f t)
  (h2 : Differentiable ℝ f)
  : ∫ x in set.Ici (1:ℝ), |deriv f x| < ⊤ :=
begin
  sorry
end

end bounded_variation_l682_682620


namespace expansion_no_x2_term_l682_682523

theorem expansion_no_x2_term (n : ℕ) (h1 : 5 ≤ n) (h2 : n ≤ 8) :
  ¬ ∃ (r : ℕ), 0 ≤ r ∧ r ≤ n ∧ n - 4 * r = 2 → n = 7 := by
  sorry

end expansion_no_x2_term_l682_682523


namespace graduate_degree_ratio_l682_682198

theorem graduate_degree_ratio (G C N : ℕ) (h1 : C = (2 / 3 : ℚ) * N)
  (h2 : (G : ℚ) / (G + C) = 0.15789473684210525) :
  (G : ℚ) / N = 1 / 8 :=
  sorry

end graduate_degree_ratio_l682_682198


namespace find_constants_l682_682058

theorem find_constants 
  (P Q R : ℚ) 
  (hP : P = 7 / 15) 
  (hQ : Q = -4 / 3) 
  (hR : R = 14 / 5) :
  ∀ x : ℚ, 
  (x^2 - 8) / ((x - 1) * (x - 4) * (x - 6)) = P / (x - 1) + Q / (x - 4) + R / (x - 6) := 
by {
  intros x,
  rw [hP, hQ, hR],
  sorry
}

end find_constants_l682_682058


namespace find_functional_solution_l682_682474

theorem find_functional_solution (c : ℝ) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) :
  ∀ x : ℝ, f x = x ^ 3 + c * x := by
  sorry

end find_functional_solution_l682_682474


namespace circle_tangent_l682_682178

theorem circle_tangent (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 1) →
  (∀ x y : ℝ, x^2 + y^2 - 6 * x - 8 * y + m = 0) →
  (∃ (x1 y1 x2 y2 : ℝ), (x1 = 0 ∧ y1 = 0 ∧ x2 = 3 ∧ y2 = 4) ∧
   (real.sqrt (3^2 + 4^2) = real.sqrt (25 - m) + 1)) →
  m = 9 :=
by
  intros hC1 hC2 hTangent
  sorry

end circle_tangent_l682_682178


namespace number_of_students_owning_both_pets_l682_682433

theorem number_of_students_owning_both_pets :
  ∀ (total : ℕ) (dogs : ℕ) (cats : ℕ) (students_owning_pets : ℕ),
    total = 50 →
    dogs = 28 →
    cats = 35 →
    students_owning_pets = total →
    ∃ (both : ℕ), dogs + cats - both = students_owning_pets ∧ both = 13 :=
by
  intros total dogs cats students_owning_pets h_total h_dogs h_cats h_students
  use 13
  split
  · rw [h_total, h_dogs, h_cats, h_students]
    linarith
  · refl


end number_of_students_owning_both_pets_l682_682433


namespace min_value_of_bS_l682_682105

variable (n : ℕ)

noncomputable def a_n : ℝ := ∫ x in 0..n, (2 * x + 1)

noncomputable def S_n (a : ℕ → ℝ) : ℝ := ∑ i in Finset.range n,  1 / a (i + 1)

noncomputable def b_n (n : ℕ) : ℤ := n - 8

noncomputable def bS (b : ℕ → ℤ) (S : ℕ → ℝ) (n : ℕ) : ℝ := b n * S n

theorem min_value_of_bS :
    ∃ n : ℕ, bS b_n (S_n a_n) n = -4 :=
sorry

end min_value_of_bS_l682_682105


namespace perpendicular_sufficient_but_not_necessary_l682_682522

theorem perpendicular_sufficient_but_not_necessary
  {A B C l : Type*}
  (h_perp_AB : l ⟂ AB)
  (h_perp_AC : l ⟂ AC)
  (h_collinear_A : collinear {A, B, C})
  (h_plane_ABC : is_plane ABC) :
  (∀ l ⟂ AB ∧ l ⟂ AC → l ⟂ BC) ∧ 
  (¬(∀ l ⟂ BC → l ⟂ AB ∧ l ⟂ AC)) :=
by
  -- Proof goes here
  sorry

end perpendicular_sufficient_but_not_necessary_l682_682522


namespace smallest_positive_multiple_of_45_l682_682842

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682842


namespace smallest_positive_multiple_of_45_l682_682838

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682838


namespace faye_rows_l682_682052

theorem faye_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h_total_pencils : total_pencils = 720)
  (h_pencils_per_row : pencils_per_row = 24) : 
  total_pencils / pencils_per_row = 30 := by 
  sorry

end faye_rows_l682_682052


namespace lattice_point_triangle_exists_l682_682469

theorem lattice_point_triangle_exists :
  ∃ (A B C : ℤ × ℤ),
    let T := triangle.mk A B C in
    is_lattice_point T.orthocenter ∧ 
    is_lattice_point T.circumcenter ∧ 
    is_lattice_point T.incenter ∧ 
    is_lattice_point T.centroid :=
begin
  sorry
end

structure triangle :=
  (A B C : ℤ × ℤ)

def is_lattice_point (point : ℤ × ℤ) : Prop :=
  ∃ x y : ℤ, point = (x, y)

noncomputable def triangle.orthocenter (T : triangle) : ℤ × ℤ := 
  T.C -- since it's a right triangle at C

noncomputable def triangle.circumcenter (T : triangle) : ℤ × ℤ :=
  let (xA, yA) := T.A in
  let (xB, yB) := T.B in
  ((xA + xB) / 2, (yA + yB) / 2)

noncomputable def triangle.incenter (T : triangle) : ℤ × ℤ :=
  let r := ((1/2 * abs (fst T.A * snd T.B)) / ((abs (fst T.A) + abs (snd T.B)))) in
  (r, r)

noncomputable def triangle.centroid (T : triangle) : ℤ × ℤ :=
  let (xA, yA) := T.A in
  let (xB, yB) := T.B in
  let (xC, yC) := T.C in
  ((xA + xB + xC) / 3, (yA + yB + yC) / 3)

end lattice_point_triangle_exists_l682_682469


namespace smallest_positive_multiple_of_45_l682_682770

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682770


namespace volume_of_region_l682_682441

-- Given conditions
def region (x y z : ℝ) : Prop :=
  |x + y + z| + |x + y - z| ≤ 12 ∧
  |x + z - y| + |x - y - z| ≤ 6 ∧
  x ≥ 0 ∧
  y ≥ 0 ∧
  z ≥ 0

-- The statement that needs to be proved
theorem volume_of_region : 
  ∀ (x y z : ℝ), region x y z →
  volume (region x y z) = 121.5 :=
sorry


end volume_of_region_l682_682441


namespace sum_sequence_2018_l682_682173

def sequence (n : ℕ) : ℝ :=
if n % 2 = 1 then 1 / (n^2 + 2 * n)
else Real.sin (n * Real.pi / 4)

def sum_sequence (n : ℕ) : ℝ :=
(∑ i in Finset.range (n + 1), sequence i)

theorem sum_sequence_2018 :
  sum_sequence 2018 = 3028 / 2019 := 
sorry

end sum_sequence_2018_l682_682173


namespace B_2_is_correct_sum_of_B_2n_l682_682501

noncomputable def f (i : ℕ) (A : List ℤ) : List ℤ :=
  if i % 2 = 1 then
    A.map (λ x, if x % 2 = 0 then x - 1 else x + i)
  else
    A.map (λ x, if x % 2 = 0 then x + 2 * i else x - 2)

def B : ℕ → List ℤ
| 0 => [2, 0, 2, 3, 5, 7]
| (n+1) => f (n+1) (B n)

theorem B_2_is_correct :
  B 2 = [-1, -3, -1, 8, 10, 12] :=
sorry

theorem sum_of_B_2n (n : ℕ) :
  n > 0 → (B (2 * n)).sum = 9 * n^2 + 4 * n + 19 :=
sorry

end B_2_is_correct_sum_of_B_2n_l682_682501


namespace tenisha_puppies_proof_l682_682680

def tenisha_remains_with_puppies (total_dogs : ℕ) (percent_female : ℚ) (fraction_giving_birth : ℚ) (puppies_per_dog : ℕ) (donated_puppies : ℕ) : ℕ :=
  let female_dogs := percent_female * total_dogs
  let female_giving_birth := fraction_giving_birth * female_dogs
  let total_puppies := female_giving_birth * puppies_per_dog
  total_puppies - donated_puppies

theorem tenisha_puppies_proof :
  tenisha_remains_with_puppies 40 0.60 0.75 10 130 = 50 :=
by
  sorry

end tenisha_puppies_proof_l682_682680


namespace collinear_points_l682_682635

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def non_collinear (a b : V) : Prop := ¬ (∃ λ : ℝ, a = λ • b)

theorem collinear_points 
  (a b : V) (p : ℝ)
  (h1 : non_collinear a b) 
  (h2 : ∀ (A B D : V), A = 2 • a + p • b → B = a + b → D = a - 2 • b → ∃ λ : ℝ, (A - B) = λ • (B - D)) :
  p = -1 :=
by
  sorry

end collinear_points_l682_682635


namespace positive_reals_inequality_l682_682117

variable {a b c : ℝ}

theorem positive_reals_inequality (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (a * b)^(1/4) + (b * c)^(1/4) + (c * a)^(1/4) < 1/4 := 
sorry

end positive_reals_inequality_l682_682117


namespace part_I_part_II_l682_682164

open Real

noncomputable def f (x : ℝ) : ℝ := sin x * (cos x - (sqrt 3 / 3) * sin x)

theorem part_I: 
  let I := set.Icc 0 (pi / 2) in
  set.range f I = set.Icc (-sqrt 3 / 3) (sqrt 3 / 6) :=
sorry

theorem part_II (α : ℝ) (hα : α ∈ set.Icc 0 pi) (hf : f (α / 2) = -sqrt 3 / 12) :
  cos (π / 6 - 2 * α) = -sqrt 15 / 8 :=
sorry

end part_I_part_II_l682_682164


namespace smallest_positive_multiple_of_45_l682_682850

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682850


namespace smallest_positive_multiple_of_45_l682_682846

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682846


namespace arithmetic_sequence_ninth_term_l682_682683

theorem arithmetic_sequence_ninth_term (a d : ℤ) 
    (h5 : a + 4 * d = 23) (h7 : a + 6 * d = 37) : 
    a + 8 * d = 51 := 
by 
  sorry

end arithmetic_sequence_ninth_term_l682_682683


namespace sum_of_segments_equals_side_length_l682_682221

variable {A B C P K L M : Type}
variable [Triangle A B C] [Equilateral A B C]
variable [Point P A B C]
variable [Point K A B] [Point L B C] [Point M C A]

-- Define a function to represent being parallel
def parallel (x y : Line) : Prop := sorry

-- Define the scenario
def setup : Prop :=
  ∃ (K : Point) (L : Point) (M : Point),
  (K ∈ segment A B) ∧ (L ∈ segment B C) ∧ (M ∈ segment C A) ∧
  parallel (line_through P K) (line_through B C) ∧
  parallel (line_through P L) (line_through A C) ∧
  parallel (line_through P M) (line_through A B)

-- Main theorem to prove
theorem sum_of_segments_equals_side_length :
  setup → distance P K + distance P L + distance P M = distance A B :=
by
  intro h,
  sorry

end sum_of_segments_equals_side_length_l682_682221


namespace percentage_increase_extra_day_l682_682273

-- Definitions of given conditions
def daily_rate_porter : ℝ := 8
def work_days_week : ℕ := 5
def total_earnings_with_overtime : ℝ := 208
def weeks_in_month : ℕ := 4

-- Definition of the problem statement to be proved
theorem percentage_increase_extra_day :
  let weekly_earnings := daily_rate_porter * work_days_week in
  let monthly_earnings_without_overtime := weekly_earnings * weeks_in_month in
  let earnings_from_overtime := total_earnings_with_overtime - monthly_earnings_without_overtime in
  let earnings_extra_day := earnings_from_overtime / weeks_in_month in
  let percentage_increase := ((earnings_extra_day - daily_rate_porter) / daily_rate_porter) * 100 in
  percentage_increase = 50 := by
  sorry

end percentage_increase_extra_day_l682_682273


namespace race_result_l682_682078

-- Define the contestants
inductive Contestants
| Alyosha
| Borya
| Vanya
| Grisha

open Contestants

-- Define their statements
def Alyosha_statement (place : Contestants → ℕ) : Prop :=
  place Alyosha ≠ 1 ∧ place Alyosha ≠ 4

def Borya_statement (place : Contestants → ℕ) : Prop :=
  place Borya ≠ 4

def Vanya_statement (place : Contestants → ℕ) : Prop :=
  place Vanya = 1

def Grisha_statement (place : Contestants → ℕ) : Prop :=
  place Grisha = 4

-- Define that exactly one statement is false and the rest are true
def three_true_one_false (place : Contestants → ℕ) : Prop :=
  (Alyosha_statement place ∧ ¬ Vanya_statement place ∧ Borya_statement place ∧ Grisha_statement place) ∨
  (¬ Alyosha_statement place ∧ Vanya_statement place ∧ Borya_statement place ∧ Grisha_statement place) ∨
  (Alyosha_statement place ∧ Vanya_statement place ∧ ¬ Borya_statement place ∧ Grisha_statement place) ∨
  (Alyosha_statement place ∧ Vanya_statement place ∧ Borya_statement place ∧ ¬ Grisha_statement place)

-- Define the conclusion: Vanya lied and Borya was first
theorem race_result (place : Contestants → ℕ) : 
  three_true_one_false place → 
  (¬ Vanya_statement place ∧ place Borya = 1) :=
sorry

end race_result_l682_682078


namespace martha_bottles_l682_682418

def total_bottles_left (a b c d : ℕ) : ℕ :=
  a + b + c - d

theorem martha_bottles : total_bottles_left 4 4 5 3 = 10 :=
by
  sorry

end martha_bottles_l682_682418


namespace clock_shows_four_different_digits_for_588_minutes_l682_682301

-- Definition of the problem
def isFourDifferentDigits (h1 h2 m1 m2 : Nat) : Bool :=
  (h1 ≠ h2) && (h1 ≠ m1) && (h1 ≠ m2) && (h2 ≠ m1) && (h2 ≠ m2) && (m1 ≠ m2)

noncomputable def countFourDifferentDigitsMinutes : Nat :=
  let validMinutes := List.filter (λ (t : Nat × Nat),
    let (h, m) := t
    let h1 := h / 10
    let h2 := h % 10
    let m1 := m / 10
    let m2 := m % 10
    isFourDifferentDigits h1 h2 m1 m2
  ) (List.product (List.range 24) (List.range 60))
  validMinutes.length

theorem clock_shows_four_different_digits_for_588_minutes :
  countFourDifferentDigitsMinutes = 588 := sorry

end clock_shows_four_different_digits_for_588_minutes_l682_682301


namespace value_of_MA_MB_l682_682543

-- Given conditions
def line_parametric (t : ℝ) : ℝ × ℝ :=
  (5 + (real.sqrt 3) / 2 * t, (real.sqrt 3) + 1 / 2 * t)

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ = 2 * real.cos θ

def rectangular_curve (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 - 2 * x = 0

def point_M : ℝ × ℝ :=
  (5, real.sqrt 3)

-- Objective
theorem value_of_MA_MB :
  | MA | * | MB | = 18 :=
sorry

end value_of_MA_MB_l682_682543


namespace number_of_divisors_of_36_l682_682548

theorem number_of_divisors_of_36 :  
  let n := 36
  number_of_divisors n = 9 := 
by 
  sorry

end number_of_divisors_of_36_l682_682548


namespace scientific_notation_36600_l682_682701

theorem scientific_notation_36600 : ∃ a n : ℝ, 36600 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.66 ∧ n = 4 := 
by 
  use 3.66, 4
  split
  . show 36600 = 3.66 * 10^4, by norm_num
  split
  . show 1 ≤ |3.66|, by norm_num 
  split
  . show |3.66| < 10, by norm_num 
  split
  . show 3.66 = 3.66, by norm_num 
  . show 4 = 4, by norm_num

end scientific_notation_36600_l682_682701


namespace range_a_12_15_l682_682511

variable (a : ℝ)
def M : Set ℝ := { x | x^2 - 9 > 0 }
def N : Set ℤ := { x | (x:ℝ)^2 - 8 * x + a < 0 }
def intersection_cardinality_is_4 : Prop :=
  (M ∩ ↑N).to_finset.card = 4

theorem range_a_12_15 : intersection_cardinality_is_4 a → 12 ≤ a ∧ a < 15 :=
sorry

end range_a_12_15_l682_682511


namespace max_exp_equals_min_quad_l682_682715

def quadratic_function (x: ℝ) : ℝ := x^2 - 2*x + 3
def exp_function (a x : ℝ) : ℝ := a^x

theorem max_exp_equals_min_quad (a : ℝ) (h1 : a > 0) :
  (∀ x ∈ set.Icc (-1 : ℝ) 2, exp_function a x ≤ 2) ↔ a = real.sqrt 2 ∨ a = 1 / 2 :=
by
  sorry

end max_exp_equals_min_quad_l682_682715


namespace house_painting_time_l682_682045

theorem house_painting_time (people_initial : ℕ) (hours_initial : ℕ) (people_additional : ℕ) (total_work : ℕ)
  (H1 : people_initial = 8) (H2 : hours_initial = 3) (H3 : people_additional = 5) (H4 : total_work = people_initial * hours_initial) :
  total_work / people_initial = 3 :=
by
  rw [H1, H2] at H4
  exact (Nat.mul_div_cancel_left 3 (by have : 8 > 0 := by decide; exact this)).symm

end house_painting_time_l682_682045


namespace line_intersects_segment_l682_682485

theorem line_intersects_segment (a : ℝ) : -∞ < a ∧ a ≤ -2 ∨ 1 ≤ a ∧ a < ∞ ↔ 
  ∃ (A B : ℝ × ℝ), 
  A = (2, 3) ∧ B = (-3, 2) ∧ 
  ∃ line : ℝ × ℝ → ℝ, 
  (∀ x y, line (x, y) = a * x + y + 1) ∧ 
  (∃ t ∈ (set.Icc 0 1), 
    let C := (1 - t) • A + t • B in line C = 0) :=
by
  -- Proof is skipped.
  sorry

end line_intersects_segment_l682_682485


namespace arithmetic_sequence_common_difference_l682_682503

noncomputable def common_difference (a b : ℝ) : ℝ := a - 1

theorem arithmetic_sequence_common_difference :
  ∀ (a b : ℝ), 
    (a - 1 = b - a) → 
    ((a + 2) ^ 2 = 3 * (b + 5)) → 
    common_difference a b = 3 := by
  intros a b h1 h2
  sorry

end arithmetic_sequence_common_difference_l682_682503


namespace kanul_initial_cash_l682_682614

noncomputable def raw_materials_without_tax : ℝ := 500 / 1.05
noncomputable def machinery_without_tax : ℝ := 400 / 1.05
noncomputable def total_cost_before_tax : ℝ := raw_materials_without_tax + machinery_without_tax

theorem kanul_initial_cash (C : ℝ) 
  (H1 : raw_materials_without_tax = 476.19) 
  (H2 : machinery_without_tax = 380.95)
  (H3 : total_cost_before_tax = 857.14)
  (H4 : C - (0.10 * C + total_cost_before_tax) = 900) : 
  C = 1952.38 :=
begin
  sorry
end

end kanul_initial_cash_l682_682614


namespace smallest_positive_multiple_45_l682_682811

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682811


namespace smallest_positive_multiple_of_45_is_45_l682_682948

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682948


namespace sum_of_sequence_l682_682048

def alternating_sum (n : ℕ) : ℕ :=
  ∑ i in range (n/2), (2000 - 10*i) - (1995 - 10*i) + 5

theorem sum_of_sequence : alternating_sum 399 = 1000 :=
by
  sorry

end sum_of_sequence_l682_682048


namespace a_less_than_b_l682_682133

theorem a_less_than_b 
  (a b : ℝ) 
  (h1 : 3^a + 13^b = 17^a) 
  (h2 : 5^a + 7^b = 11^b) : 
  a < b :=
sorry

end a_less_than_b_l682_682133


namespace only_prime_if_n_eq_1_l682_682477

theorem only_prime_if_n_eq_1 (n: ℕ) : (Nat.prime (3^(2*n) - 2^n)) ↔ n = 1 := 
by
  sorry

end only_prime_if_n_eq_1_l682_682477


namespace smallest_positive_multiple_of_45_is_45_l682_682949

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682949


namespace derivative_at_one_is_negative_one_l682_682150

theorem derivative_at_one_is_negative_one (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h₁ : ∀ x, deriv f x = f' x)
  (h₂ : ∀ x, f x = 2 * x * f' 1 + log x) :
  f' 1 = -1 := by
  sorry

end derivative_at_one_is_negative_one_l682_682150


namespace radius_moon_scientific_notation_l682_682338

def scientific_notation := 1738000 = 1.738 * 10^6

theorem radius_moon_scientific_notation : scientific_notation := 
sorry

end radius_moon_scientific_notation_l682_682338


namespace nina_not_taller_than_lena_l682_682086

noncomputable def friends_heights := ℝ 
variables (M N L O : friends_heights)

def nina_shorter_than_masha (N M : friends_heights) : Prop := N < M
def lena_taller_than_olya (L O : friends_heights) : Prop := L > O
def nina_taller_than_lena (N L : friends_heights) : Prop := N > L

theorem nina_not_taller_than_lena (N M L O : friends_heights) 
  (h₁ : nina_shorter_than_masha N M) 
  (h₂ : lena_taller_than_olya L O) : 
  (0 : ℝ) = 0 :=
sorry

end nina_not_taller_than_lena_l682_682086


namespace binomial_coefficient_divisibility_l682_682685

theorem binomial_coefficient_divisibility (n k : ℕ) (hkn : k ≤ n - 1) :
  ((n.prime) ∨ (¬ (n.prime) ∧ ∃ p, Nat.Prime p ∧ p ∣ n ∧ ¬ (n ∣ Nat.choose n p))) :=
by sorry

end binomial_coefficient_divisibility_l682_682685


namespace problem1_problem2_l682_682174

noncomputable def A := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
noncomputable def B (m : ℝ) := {x : ℝ | m - 3 ≤ x ∧ x ≤ m + 3}

-- Problem 1
theorem problem1 (m : ℝ) : (A ∩ B m = {x : ℝ | 2 ≤ x ∧ x ≤ 3}) → m = 5 :=
sorry

-- Problem 2
theorem problem2 (m : ℝ) : (A ⊆ {x : ℝ | x < m - 3 ∨ m + 3 < x}) → (m ∈ set.Ioo (-∞) (-4) ∪ set.Ioo 6 ∞) :=
sorry

end problem1_problem2_l682_682174


namespace simplify_expression_l682_682665

variable (m : ℝ) (h : m ≠ 0)

theorem simplify_expression : ( (1/(3*m))^(-3) * (2*m)^(4) ) = 432 * m^(7) := by sorry

end simplify_expression_l682_682665


namespace train_speed_excluding_stoppages_l682_682049

theorem train_speed_excluding_stoppages (S : ℝ) 
  (h_inc_speed : ∀ t : ℝ, t = 1 → (S * 42 / 60 = 21)) (stop_time : ℝ) (h_stop_time : stop_time = 18 / 60) : 
  S = 30 :=
by
  let t := 1
  have h := h_inc_speed t (by norm_num)
  have t' : t * 60 = 60 := by norm_num
  have h1 : 42 / 60 = 7 / 10 := by norm_num
  calc
    S = 21 / (7 / 10) : by rw [←h, h1]; field_simp
    _ = 30          : by norm_num

end train_speed_excluding_stoppages_l682_682049


namespace margarita_jumps_farther_l682_682657

-- Definitions for conditions
def RiccianaTotalDistance := 24
def RiccianaRunDistance := 20
def RiccianaJumpDistance := 4

def MargaritaRunDistance := 18
def MargaritaJumpDistance := 2 * RiccianaJumpDistance - 1

-- Theorem statement to prove the question
theorem margarita_jumps_farther :
  (MargaritaRunDistance + MargaritaJumpDistance) - RiccianaTotalDistance = 1 :=
by
  -- Proof will be written here
  sorry

end margarita_jumps_farther_l682_682657


namespace smallest_positive_multiple_of_45_l682_682880

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682880


namespace seventyFifthTermInSequence_l682_682582

/-- Given a sequence that starts at 2 and increases by 4 each term, 
prove that the 75th term in this sequence is 298. -/
theorem seventyFifthTermInSequence : 
  (∃ a : ℕ → ℤ, (∀ n : ℕ, a n = 2 + 4 * n) ∧ a 74 = 298) :=
by
  sorry

end seventyFifthTermInSequence_l682_682582


namespace true_particular_proposition_l682_682427

theorem true_particular_proposition (α : ℝ) : (∃ α, Real.tan (π / 2 - α) = 1) := by
  exists (π / 4)
  simp [Real.tan_pi_div_four, Real.tan_add_pi_div_two]
  sorry

end true_particular_proposition_l682_682427


namespace solve_x_eq_3_l682_682351

theorem solve_x_eq_3 : ∀ (x : ℝ), (x = 3) ↔ (x - 3 = 0) :=
by
  intro x
  split
  . intro h
    rw [h]
    ring
  . intro h
    linarith
  

end solve_x_eq_3_l682_682351


namespace pyramid_volume_PQST_l682_682992

-- Definitions for the conditions
def CubeVolume : ℝ := 8
def SideLength : ℝ := real.cbrt CubeVolume

-- Definitions for the calculation of area and volume
def TriangleBaseArea (a : ℝ) := 0.5 * a * a
def PyramidVolume (base_area height : ℝ) := (1 / 3) * base_area * height

-- Statement of the theorem
theorem pyramid_volume_PQST :
  let side := SideLength in
  let height := side in
  PyramidVolume (TriangleBaseArea side) height = 4 / 3 := 
by 
  sorry

end pyramid_volume_PQST_l682_682992


namespace triangle_can_be_formed_with_single_color_sticks_l682_682996

theorem triangle_can_be_formed_with_single_color_sticks
  (a b c d e f : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (alternating_colors : (a = c ∧ c = e) ∧ (b = d ∧ d = f))
  (triangle_inequality : ∀ x y z : ℝ, x + y > z → y + z > x → z + x > y) :
  (b + c > a ∧ d + e > a ∧ f + b > e) →
  ∃ x y z : ℝ, (x = a ∨ x = c ∨ x = e) ∧ (y = a ∨ y = c ∨ y = e) ∧ (z = a ∨ z = c ∨ z = e) ∧ 
              triangle_inequality x y z :=
sorry

end triangle_can_be_formed_with_single_color_sticks_l682_682996


namespace problem_statement_l682_682005

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end problem_statement_l682_682005


namespace calculate_weight_difference_l682_682975

noncomputable def joe_weight := 43 -- Joe's weight in kg
noncomputable def original_avg_weight := 30 -- Original average weight in kg
noncomputable def new_avg_weight := 31 -- New average weight in kg after Joe joins
noncomputable def final_avg_weight := 30 -- Final average weight after two students leave

theorem calculate_weight_difference :
  ∃ (n : ℕ) (x : ℝ), 
  (original_avg_weight * n + joe_weight) / (n + 1) = new_avg_weight ∧
  (new_avg_weight * (n + 1) - 2 * x) / (n - 1) = final_avg_weight →
  x - joe_weight = -6.5 :=
by
  sorry

end calculate_weight_difference_l682_682975


namespace number_of_incorrect_statements_is_4_l682_682426

/-- Define each of the given statements as a Boolean. -/
def stmt1 : Bool := (-(-2)^2) = 4
def stmt2 : Bool := -5 / (1/5) = -5
def stmt3 : Bool := (2^2) / 3 = 4 / 9
def stmt4 : Bool := (-3)^2 * (-1/3) = -3
def stmt5 : Bool := -3^3 = -9

/-- Define the number of incorrect statements. -/
def incorrect_statements := [stmt1, stmt2, stmt3, stmt5].count (λ b => ¬b)

/-- Theorem statement. -/
theorem number_of_incorrect_statements_is_4 : incorrect_statements = 4 := by
  intros
  sorry

end number_of_incorrect_statements_is_4_l682_682426


namespace smallest_positive_multiple_of_45_is_45_l682_682916

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682916


namespace triangle_area_l682_682259

open Real EuclideanGeometry

/--
Given points O, A, B, and C in a three-dimensional coordinate system with
O as the origin, A on the positive x-axis with OA = 4, B on the positive y-axis with OB = 3,
C on the positive z-axis with OC = 6, and the angle ∠BAC = 45°.
Prove that the area of triangle ABC is 5 * sqrt 104 / 4.
-/
theorem triangle_area {A B C : Point}
  (O : Point)
  (hO: O = ⟨0, 0, 0⟩)
  (hA: A = ⟨4, 0, 0⟩)
  (hB: B = ⟨0, 3, 0⟩)
  (hC: C = ⟨0, 0, 6⟩)
  (angle_BAC : ∠O A C = π/4) :
  area (triangle A B C) = 5 * (sqrt 104) / 4 := sorry

end triangle_area_l682_682259


namespace probability_nina_taller_than_lena_l682_682099

variables {M N L O : ℝ}

theorem probability_nina_taller_than_lena (h₁ : N < M) (h₂ : L > O) : 
  ∃ P : ℝ, P = 0 ∧ ∀ M N L O, M ≠ N ∧ M ≠ L ∧ M ≠ O ∧ N ≠ L ∧ N ≠ O ∧ L ≠ O → 
  (M > N → O < L → P = 0) :=
by sorry

end probability_nina_taller_than_lena_l682_682099


namespace largest_prime_factor_expr_l682_682380

def expr : ℤ := 18^4 + 2 * 18^2 + 1 - 16^4

theorem largest_prime_factor_expr : largest_prime_factor expr = 53 :=
by
  sorry

end largest_prime_factor_expr_l682_682380


namespace lisa_eggs_per_year_l682_682639

theorem lisa_eggs_per_year :
  let days_per_week := 5 in
  let weeks_per_year := 52 in
  let children := 4 in
  let eggs_per_child := 2 in
  let eggs_for_husband := 3 in
  let eggs_for_herself := 2 in
  let breakfasts_per_year := days_per_week * weeks_per_year in
  let eggs_per_breakfast := (children * eggs_per_child) + eggs_for_husband + eggs_for_herself in
  let total_eggs := eggs_per_breakfast * breakfasts_per_year in
  total_eggs = 3380
by
  sorry

end lisa_eggs_per_year_l682_682639


namespace exists_prime_q_not_dividing_np_minus_p_l682_682630

theorem exists_prime_q_not_dividing_np_minus_p (p : ℕ) [hp : Nat.Prime p] :
  ∃ (q : ℕ), Nat.Prime q ∧ ∀ (n : ℤ), ¬(q ∣ (n ^ p - p)) :=
by
  sorry

end exists_prime_q_not_dividing_np_minus_p_l682_682630


namespace counterexample_n_eq_10_l682_682030

-- Define the conditions given in the problem

def not_prime (n : ℕ) : Prop := ¬ (Nat.Prime n)
def is_counterexample (n : ℕ) : Prop := not_prime n ∧ not_prime (n + 2)

-- Define the main statement: proving 10 is a counterexample

theorem counterexample_n_eq_10 : is_counterexample 10 :=
begin
  sorry  -- The proof goes here
end

end counterexample_n_eq_10_l682_682030


namespace smallest_positive_multiple_of_45_is_45_l682_682922

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682922


namespace probability_N_taller_than_L_l682_682096

variable (M N L O : ℕ)

theorem probability_N_taller_than_L 
  (h1 : N < M) 
  (h2 : L > O) : 
  0 = 0 := 
sorry

end probability_N_taller_than_L_l682_682096


namespace factorial_division_l682_682447

theorem factorial_division (n m : ℕ) (h : n = 52) (g : m = 50) : (n! / m!) = 2652 :=
by
  sorry

end factorial_division_l682_682447


namespace factorial_division_l682_682453

theorem factorial_division :
  52! / 50! = 2652 := by
  sorry

end factorial_division_l682_682453


namespace comparison1_comparison2_comparison3_l682_682028

theorem comparison1 : -3.2 > -4.3 :=
by sorry

theorem comparison2 : (1 : ℚ) / 2 > -(1 / 3) :=
by sorry

theorem comparison3 : (1 : ℚ) / 4 > 0 :=
by sorry

end comparison1_comparison2_comparison3_l682_682028


namespace inscribed_square_area_l682_682733

theorem inscribed_square_area (A_circle : ℝ) (r : ℝ) (s : ℝ) :
  A_circle = 324 * real.pi →
  r = real.sqrt (A_circle / real.pi) →
  2 * r = s * real.sqrt 2 →
  (s * s) = 648 :=
by
  intro hA_circle hr hs
  sorry

end inscribed_square_area_l682_682733


namespace ratio_of_sub_triangle_l682_682711

noncomputable theory

variables {A B C M N P : Type} 
variables [ordered_semiring A] [ordered_semiring B] [ordered_semiring C] [ordered_semiring M] [ordered_semiring N] [ordered_semiring P]
variables (ABC : triangle A B C)
variables (M N P : point A B C)
variables (h1 : AM = 1) (h2 : AB = 5 * AM) (h3 : BN = 1) (h4 : BC = 5 * BN) (h5 : CP = 1) (h6 : CA = 5 * CP)

theorem ratio_of_sub_triangle (t : triangle A B C) (AM : point A B) (MB : point B C) (BN : point B C) (NC : point C A)
    (CP : point C A) (PA : point A B)
    (h1 : AM / MB = 1 / 4)
    (h2 : BN / NC = 1 / 4)
    (h3 : CP / PA = 1 / 4) :
  area (triangle (point_intersection AN BP) (point_intersection BP CM) (point_intersection CM AN)) / area ABC = 3 / 7 :=
sorry

end ratio_of_sub_triangle_l682_682711


namespace sum_of_coefficients_l682_682194

-- Sum of the coefficients in the expansion of (x + y)^a is equal to 2^a
theorem sum_of_coefficients (a : ℕ) : 
  (∑ k in Finset.range (a + 1), Nat.choose a k) = 2^a := by
  sorry

end sum_of_coefficients_l682_682194


namespace smallest_positive_multiple_of_45_is_45_l682_682913

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682913


namespace sum_last_two_digits_l682_682974

theorem sum_last_two_digits (a b m n : ℕ) (h7 : a = 7) (h13 : b = 13) (h100 : m = 100) (h30 : n = 30) : 
 ((a ^ n) + (b ^ n)) % m = 98 :=
by
  have h₁ : 7 ^ 30 % 100 = (49 : ℕ) := by sorry
  have h₂ : 13 ^ 30 % 100 = 49 := by sorry
  calc
    (7 ^ 30 + 13 ^ 30) % 100
      = (49 + 49) % 100 : by { rw [h₁, h₂] }
  ... = 98 % 100 : by rfl
  ... = 98 : by rfl

end sum_last_two_digits_l682_682974


namespace complex_point_coordinates_l682_682295

noncomputable def Z : ℂ := complex.I * (1 + complex.I)

theorem complex_point_coordinates :
  (Z.re, Z.im) = (-1, 1) := 
sorry

end complex_point_coordinates_l682_682295


namespace bike_lock_combinations_l682_682233

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime_lt_30 (n : ℕ) : Prop := nat.prime n ∧ n < 30
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

noncomputable def num_combinations : ℕ :=
  let odds := { n : ℕ | n ∈ set.Icc 1 40 ∧ is_odd n },
      primes := { n : ℕ | n ∈ set.Icc 1 40 ∧ is_prime_lt_30 n },
      multiples_of_4 := { n : ℕ | n ∈ set.Icc 1 40 ∧ is_multiple_of_4 n },
      perfect_squares := { n : ℕ | n ∈ set.Icc 1 40 ∧ is_perfect_square n } in
    set.card odds * set.card primes * set.card multiples_of_4 * set.card perfect_squares

theorem bike_lock_combinations : num_combinations = 12000 :=
by sorry

end bike_lock_combinations_l682_682233


namespace math_expression_equivalent_l682_682025

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end math_expression_equivalent_l682_682025


namespace inequality_proof_l682_682518

noncomputable def positivity_inequality {n : ℕ} (a : Fin n → ℝ) : Prop :=
  ∀ i, 0 < a i

theorem inequality_proof {n : ℕ} (a : Fin n → ℝ) (h_pos : positivity_inequality a) :
  (∑ i in Finset.range (n-1), a (i+1) / (∑ j in Finset.range (i+1), a (j+1))^2) < 1 / a 0 :=
sorry

end inequality_proof_l682_682518


namespace general_solution_diff_eq_l682_682482
noncomputable theory

-- Define the general solution of the differential equation
theorem general_solution_diff_eq (r : ℝ → ℝ) (C : ℝ) (ϕ : ℝ)
  (h0 : ∀ ϕ, deriv r ϕ - r ϕ * deriv id ϕ = 0) :
  ∃ C : ℝ, ∀ ϕ, r ϕ = C * Real.exp ϕ :=
by
  -- Placeholder for the proof
  sorry

end general_solution_diff_eq_l682_682482


namespace existence_not_implied_by_validity_l682_682044

-- Let us formalize the theorem and then show that its validity does not imply the existence of such a function.

-- Definitions for condition (A) and the theorem statement
axiom condition_A (f : ℝ → ℝ) : Prop
axiom theorem_239 : ∀ f, condition_A f → ∃ T, ∀ x, f (x + T) = f x

-- Translation of the problem statement into Lean
theorem existence_not_implied_by_validity :
  (∀ f, condition_A f → ∃ T, ∀ x, f (x + T) = f x) → 
  ¬ (∃ f, condition_A f) :=
sorry

end existence_not_implied_by_validity_l682_682044


namespace smallest_positive_multiple_of_45_l682_682886

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682886


namespace simplify_expr1_simplify_expr2_l682_682282

variable (a b t : ℝ)

theorem simplify_expr1 : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l682_682282


namespace find_number_of_homes_l682_682230

-- Define the conditions
def slab_length : ℝ := 100
def slab_width : ℝ := 100
def slab_height : ℝ := 0.5
def concrete_density : ℝ := 150 -- in pounds per cubic foot
def cost_per_pound : ℝ := 0.02 -- in dollars per pound
def total_cost : ℝ := 45000 -- in dollars

-- Define the question and expected answer
def expected_number_of_homes : ℕ := 3

-- Define the theorem statement
theorem find_number_of_homes (length width height density cost_per_pound total_cost : ℝ) 
  (volume := length * width * height) :
  let weight := volume * density,
      cost_per_home := weight * cost_per_pound in
  (total_cost / cost_per_home) = expected_number_of_homes :=
by
  sorry

end find_number_of_homes_l682_682230


namespace brothers_complete_task_in_3_days_l682_682224

theorem brothers_complete_task_in_3_days :
  (1 / 4 + 1 / 12) * 3 = 1 :=
by
  sorry

end brothers_complete_task_in_3_days_l682_682224


namespace smallest_positive_multiple_of_45_l682_682825

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682825


namespace smallest_positive_multiple_of_45_l682_682931

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682931


namespace triangle_incenter_angle_l682_682607

theorem triangle_incenter_angle
  (X Y Z I P Q R : Type)
  [triangle X Y Z]
  [angle_bisectors XP YQ ZR]
  [incenter I XP YQ ZR]
  (h : ∠XYZ = 48) : ∠YIZ = 66 :=
begin
  sorry
end

end triangle_incenter_angle_l682_682607


namespace smallest_positive_multiple_of_45_l682_682856

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682856


namespace magazine_cost_l682_682032

variable (M : ℝ)

-- Condition 1: Daniel buys a magazine (M) and a pencil costing $0.50
def total_cost_before_coupon (M : ℝ) : ℝ := M + 0.50

-- Condition 2: He has a coupon that gives him $0.35 off
def total_cost_after_coupon (M : ℝ) : ℝ := (total_cost_before_coupon M) - 0.35

-- Condition 3: He spends $1 after applying the coupon
theorem magazine_cost : total_cost_after_coupon M = 1 → M = 0.85 :=
by
  intro h,
  -- We don't provide the proof, just skip with sorry
  sorry

end magazine_cost_l682_682032


namespace find_sticker_price_l682_682181

variable (x : ℝ)

def price_at_store_A (x : ℝ) : ℝ := 0.80 * x - 120
def price_at_store_B (x : ℝ) : ℝ := 0.70 * x
def savings (x : ℝ) : ℝ := price_at_store_B x - price_at_store_A x

theorem find_sticker_price (h : savings x = 30) : x = 900 :=
by
  -- proof can be filled in here
  sorry

end find_sticker_price_l682_682181


namespace smallest_positive_multiple_l682_682781

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682781


namespace nina_not_taller_than_lena_l682_682087

noncomputable def friends_heights := ℝ 
variables (M N L O : friends_heights)

def nina_shorter_than_masha (N M : friends_heights) : Prop := N < M
def lena_taller_than_olya (L O : friends_heights) : Prop := L > O
def nina_taller_than_lena (N L : friends_heights) : Prop := N > L

theorem nina_not_taller_than_lena (N M L O : friends_heights) 
  (h₁ : nina_shorter_than_masha N M) 
  (h₂ : lena_taller_than_olya L O) : 
  (0 : ℝ) = 0 :=
sorry

end nina_not_taller_than_lena_l682_682087


namespace science_and_technology_group_total_count_l682_682349

theorem science_and_technology_group_total_count 
  (number_of_girls : ℕ)
  (number_of_boys : ℕ)
  (h1 : number_of_girls = 18)
  (h2 : number_of_girls = 2 * number_of_boys - 2)
  : number_of_girls + number_of_boys = 28 := 
by
  sorry

end science_and_technology_group_total_count_l682_682349


namespace find_solutions_l682_682476
noncomputable def system_equation (x y : ℝ) : Prop :=
  (x^2 + y^2)^2 - xy * (x + y)^2 = 19

noncomputable def absolute_difference (x y : ℝ) : Prop :=
  |x - y| = 1

theorem find_solutions (x y : ℝ) :
  system_equation x y ∧ absolute_difference x y ↔ 
  (x = 2.8 ∧ y = 1.8) ∨ (x = -1.4 ∧ y = -2.4) ∨ 
  (x = 1.8 ∧ y = 2.8) ∨ (x = -2.4 ∧ y = -1.4) := 
sorry

end find_solutions_l682_682476


namespace smallest_positive_multiple_45_l682_682905

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682905


namespace problem_statement_l682_682606

noncomputable def ratio_AD_AB (AB AD : ℝ) (angle_A angle_B angle_ADE : ℝ) : Prop :=
  angle_A = 60 ∧ angle_B = 45 ∧ angle_ADE = 45 ∧
  AD / AB = (Real.sqrt 6 + Real.sqrt 2) / (4 * Real.sqrt 2)

theorem problem_statement {AB AD : ℝ} (angle_A angle_B angle_ADE : ℝ) 
  (h1 : angle_A = 60)
  (h2 : angle_B = 45)
  (h3 : angle_ADE = 45) : 
  ratio_AD_AB AB AD angle_A angle_B angle_ADE := by {
    sorry
}

end problem_statement_l682_682606


namespace annika_total_distance_hiked_east_l682_682431

def annika_hike (rate time_remaining distance_hiked : ℝ) : ℝ :=
  let time_already_spent := distance_hiked * rate
  let remaining_time := time_remaining - time_already_spent
  let further_distance := (remaining_time / 2) / rate
  distance_hiked + further_distance

theorem annika_total_distance_hiked_east:
  ∀ (rate time_remaining initial_distance_hiked : ℝ),
    rate = 10 ∧
    initial_distance_hiked = 2.5 ∧
    time_remaining = 45 →
    annika_hike rate time_remaining initial_distance_hiked = 3.5 :=
by
  intros rate time_remaining initial_distance_hiked h
  cases h
  sorry

end annika_total_distance_hiked_east_l682_682431


namespace comparison_of_exponents_l682_682139

theorem comparison_of_exponents (m n : ℕ) (h : m > n) : 
  let a := 0.2 ^ m
  let b := 0.2 ^ n 
  in a < b :=
by 
  let a := 0.2 ^ m
  let b := 0.2 ^ n
  sorry

end comparison_of_exponents_l682_682139


namespace LovelyCakeSlices_l682_682640

/-- Lovely cuts her birthday cake into some equal pieces.
    One-fourth of the cake was eaten by her visitors.
    Nine slices of cake were kept, representing three-fourths of the total number of slices.
    Prove: Lovely cut her birthday cake into 12 equal pieces. -/
theorem LovelyCakeSlices (totalSlices : ℕ) 
  (h1 : (3 / 4 : ℚ) * totalSlices = 9) : totalSlices = 12 := by
  sorry

end LovelyCakeSlices_l682_682640


namespace fifth_largest_divisor_3640350000_is_227521875_l682_682735

def fifth_largest_divisor (n : ℕ) (k : ℕ) : Option ℕ :=
  let divisors := (List.range (n + 1)).filter (λ m, m > 0 ∧ n % m = 0)
  divisors.reverse.nth (k - 1)
  
theorem fifth_largest_divisor_3640350000_is_227521875 :
  fifth_largest_divisor 3640350000 5 = some 227521875 :=
by
  sorry

end fifth_largest_divisor_3640350000_is_227521875_l682_682735


namespace number_of_dogs_l682_682346

-- Conditions
def ratio_cats_dogs : ℚ := 3 / 4
def number_cats : ℕ := 18

-- Define the theorem to prove
theorem number_of_dogs : ∃ (dogs : ℕ), dogs = 24 :=
by
  -- Proof steps will go here, but we can use sorry for now to skip actual proving.
  sorry

end number_of_dogs_l682_682346


namespace correct_fraction_simplification_l682_682268

theorem correct_fraction_simplification {a b c : ℕ} (ha : 0 ≤ a ∧ a < 10) (hb : 0 ≤ b ∧ b < 10) (hc : 0 ≤ c ∧ c < 10) (h : (10 * a + b) * c = a * (10 * b + c)) :
  (10 * a + b, 10 * b + c) ∈ {
    (19, 95), (16, 64), (11, 11), (26, 65), (22, 22), (33, 33), (49, 98), 
    (44, 44), (55, 55), (66, 66), (77, 77), (88, 88), (99, 99) } :=
by
  sorry

end correct_fraction_simplification_l682_682268


namespace sum_second_fourth_numbers_l682_682703

-- Define the given numbers
def numbers : List Int := [-3, 5, 7, 10, 15]

-- Define the conditions as Lean propositions
def condition1 (arr : List Int) : Prop := 
  (arr.nth 4 ≠ some 15) ∧ (arr.nth 2 = some 15 ∨ arr.nth 3 = some 15 ∨ arr.nth 4 = some 15)

def condition2 (arr : List Int) : Prop := 
  (arr.nth 0 ≠ some (-3)) ∧ (arr.nth 0 = some (-3) ∨ arr.nth 1 = some (-3) ∨ arr.nth 2 = some (-3))

def condition3 (arr : List Int) : Prop := 
  (arr.nth 0 ≠ some 7) ∧ (arr.nth 4 ≠ some 7)

-- Define the proposition that needs to be proved
theorem sum_second_fourth_numbers : ∃ arr : List Int, 
  condition1 arr ∧ condition2 arr ∧ condition3 arr ∧ (arr.nth 1.get_or_else 0 + arr.nth 3.get_or_else 0 = 12) :=
by
  sorry

end sum_second_fourth_numbers_l682_682703


namespace DM_perp_BE_l682_682608

variable {A B C D E M : Type}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space M]
variable {triangle_ABC : triangle A B C}

-- Given conditions
variable (median_BM : line_segment B M)
variable (AC : line_segment A C)
variable (BM_eq_AC : length median_BM = length AC)
variable (D_on_extension_BA : point_on_line_extension D B A)
variable (E_on_extension_AC : point_on_line_extension E A C)
variable (AD_eq_AB : length (line_segment A D) = length (line_segment A B))
variable (CE_eq_CM : length (line_segment C E) = length (line_segment C M))
variable (DM : line_segment D M)
variable (BE : line_segment B E)

-- To Prove
theorem DM_perp_BE 
  (median_BM : is_median triangle_ABC B M)
  (BM_eq_AC : length median_BM = length (line_segment A C))
  (D_on_extension_BA : point_on_extension D A B)
  (E_on_extension_AC : point_on_extension E A C)
  (AD_eq_AB : length (line_segment A D) = length (line_segment A B))
  (CE_eq_CM : length (line_segment C E) = length (line_segment C M)) :
  is_perpendicular DM BE :=
sorry

end DM_perp_BE_l682_682608


namespace areas_of_isosceles_triangles_l682_682659

theorem areas_of_isosceles_triangles (A B C : ℝ) (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13)
  (hA : A = 1/2 * a * a) (hB : B = 1/2 * b * b) (hC : C = 1/2 * c * c) :
  A + B = C :=
by
  sorry

end areas_of_isosceles_triangles_l682_682659


namespace calculate_expression_l682_682020

noncomputable def solve_expression : ℝ :=
  let term1 := (real.pi - 1) ^ 0
  let term2 := 4 * real.sin (real.pi / 4) -- sin 45° = sin (π/4)
  let term3 := real.sqrt 8
  let term4 := real.abs (-3)
  term1 + term2 - term3 + term4

theorem calculate_expression : solve_expression = 4 := by
  sorry

end calculate_expression_l682_682020


namespace general_term_seq_l682_682219

universe u

-- Define the sequence
def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 2 * a n + 1)

-- State the theorem
theorem general_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n, a n = 2^n - 1 :=
by
  sorry

end general_term_seq_l682_682219


namespace smallest_positive_multiple_of_45_l682_682875

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682875


namespace number_of_divisors_of_36_l682_682560

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end number_of_divisors_of_36_l682_682560


namespace smallest_positive_multiple_l682_682785

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682785


namespace circumcenters_collinear_l682_682253

theorem circumcenters_collinear 
  (A₁ A₂ A₃ I : Type)
  [Incenter A₁ A₂ A₃ I] 
  (C₁ C₂ C₃ : Circle A₁ A₂ A₃ I) 
  [∀ i, Tangent_to (C₁[i]) (A₁[i+1]) (A₁[i+2])]
  (B₁ B₂ B₃ : Type) 
  [∀ i, Intersection (B_i+1) (C_i+2) (C_i+3)] :
  Collinear (Circumcenter(Δ A₁ B₁ I)) (Circumcenter(Δ A₂ B₂ I)) (Circumcenter(Δ A₃ B₃ I)) :=
sorry

end circumcenters_collinear_l682_682253


namespace statement_II_must_be_true_l682_682580

-- Define types for enthusiasts and properties
variable (MathematicsEnthusiast : Type) (BoardGameMaster : Type)

-- Define properties of enjoying logic puzzles
variable (EnjoysLogicPuzzles : MathematicsEnthusiast → Prop)
variable (IsMathematicsEnthusiast : BoardGameMaster → Prop)

-- Define the conditions (premises)
variable [H1 : ∀ (m : MathematicsEnthusiast), EnjoysLogicPuzzles m] -- All mathematics enthusiasts enjoy logic puzzles
variable [H2 : ∃ (b : BoardGameMaster), IsMathematicsEnthusiast b] -- Some board game masters are mathematics enthusiasts

-- Theorem to prove that some board game masters enjoy logic puzzles
theorem statement_II_must_be_true :
  ∃ (b : BoardGameMaster), IsMathematicsEnthusiast b ∧ ∀ (m : MathematicsEnthusiast), EnjoysLogicPuzzles m := 
begin
  sorry
end

end statement_II_must_be_true_l682_682580


namespace factorial_division_l682_682449

theorem factorial_division (n m : ℕ) (h : n = 52) (g : m = 50) : (n! / m!) = 2652 :=
by
  sorry

end factorial_division_l682_682449


namespace heated_water_behavior_l682_682976

def is_endothermic_reaction (reaction : Type) (temp : ℝ) : Prop := 
  true -- This is a placeholder. In reality, the specifics of the endothermic reaction would be detailed.

def promotes_ionization (water : Type) (temp : ℝ) : Prop :=
  true -- Placeholder for the logic indicating ionization increases with temperature.

def ion_product_constant_increases (water : Type) (temp : ℝ) : Prop :=
  promotes_ionization water temp -- This can be inferred from ionization promotion due to temperature.

def pH_decreases (water : Type) (temp : ℝ) : Prop :=
  ion_product_constant_increases water temp -- Directly connected to the increase of ion product constant.

def water_is_neutral (water : Type) (temp : ℝ) : Prop :=
  promotes_ionization water temp -- This ensures the $[\text{H}^+] = [\text{OH}^-]$ balance.

theorem heated_water_behavior 
  (water : Type)
  (temp : ℝ)
  (h₁ : is_endothermic_reaction water temp)
  (h₂ : promotes_ionization water temp) :
  ion_product_constant_increases water temp ∧ pH_decreases water temp ∧ water_is_neutral water temp := 
by
  split;
  sorry -- Proofs for each part would go here, but are omitted as per instructions.

end heated_water_behavior_l682_682976


namespace product_of_roots_eq_neg_125_over_4_l682_682484

theorem product_of_roots_eq_neg_125_over_4 :
  (∀ x y : ℝ, (24 * x^2 + 60 * x - 750 = 0 ∧ 24 * y^2 + 60 * y - 750 = 0 ∧ x ≠ y) → x * y = -125 / 4) :=
by
  intro x y h
  sorry

end product_of_roots_eq_neg_125_over_4_l682_682484


namespace max_volume_box_l682_682410

def volume (a x : ℝ) : ℝ :=
  (a - 2 * x) ^ 2 * x

theorem max_volume_box {a : ℝ} (ha : a = 60) :
  ∃ x : ℝ, x = 10 ∧ ∀ y : ℝ, volume a y ≤ volume a x :=
begin
  sorry
end

end max_volume_box_l682_682410


namespace solve_quadratic_l682_682157

theorem solve_quadratic (x : ℝ) (h : (9 / x^2) - (6 / x) + 1 = 0) : 2 / x = 2 / 3 :=
by
  sorry

end solve_quadratic_l682_682157


namespace lower_bound_expression_l682_682076

theorem lower_bound_expression (n : ℤ) (L : ℤ) :
  (∃ k : ℕ, k = 20 ∧
          ∀ n, (L < 4 * n + 7 ∧ 4 * n + 7 < 80)) →
  L = 3 :=
by
  sorry

end lower_bound_expression_l682_682076


namespace min_value_l682_682252

variables {A B C P: Type} 
variables {BC CA AB PD PE PF: ℝ} 

-- Definitions of a, b, c (sides of triangle) and x, y, z (distances from point P to sides)
def a := BC
def b := CA
def c := AB
def x := PD
def y := PE
def z := PF

-- Condition on the area of the triangle
variable (S : ℝ) -- Area of triangle ABC

-- Condition that P is the incenter of triangle ABC
def is_incenter (P: Type) (A B C: Type) := PD = PE ∧ PE = PF

theorem min_value (h₁ : a * x + b * y + c * z = 2 * S)
    (h₂ : is_incenter P A B C):
  \frac{a}{x} + \frac{b}{y} + \frac{c}{z} = (a + b + c)^2 / (2 * S) :=
begin
  sorry
end

end min_value_l682_682252


namespace books_sold_on_Wednesday_l682_682421

theorem books_sold_on_Wednesday (initial_stock : ℕ) (books_not_sold : ℕ)
  (books_sold_Monday : ℕ) (books_sold_Tuesday : ℕ) (books_sold_Thursday : ℕ)
  (books_sold_Friday : ℕ) : 
  initial_stock = 800 →
  books_not_sold = 600 →
  books_sold_Monday = 60 →
  books_sold_Tuesday = 10 →
  books_sold_Thursday = 44 →
  books_sold_Friday = 66 →
  ∃ W : ℕ, books_sold_Monday + books_sold_Tuesday + W + books_sold_Thursday + books_sold_Friday = initial_stock - books_not_sold ∧ W = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  existsi 20
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end books_sold_on_Wednesday_l682_682421


namespace smallest_positive_multiple_45_l682_682900

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682900


namespace smallest_positive_multiple_of_45_l682_682774

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682774


namespace find_lambda_l682_682148

variables {V : Type*} [inner_product_space ℝ V]

def ab : V := sorry
def ac : V := sorry
def ap (λ : ℝ) : V := λ • ab + ac
def bc : V := ac - ab

theorem find_lambda (h_angle : real.angle_between ab ac = real.pi / 3)
  (h_ab_mag : ∥ab∥ = 3)
  (h_ac_mag : ∥ac∥ = 2)
  (h_perp : inner_product 𝕍 (ap λ) bc = 0) :
  λ = 7 / 12 :=
sorry

end find_lambda_l682_682148


namespace find_b_l682_682744

-- Define what it means for b to be a solution
def is_solution (b : ℤ) : Prop :=
  b > 4 ∧ ∃ k : ℤ, 4 * b + 5 = k * k

-- State the problem
theorem find_b : ∃ b : ℤ, is_solution b ∧ ∀ b' : ℤ, is_solution b' → b' ≥ 5 := by
  sorry

end find_b_l682_682744


namespace find_b_l682_682745

-- Define what it means for b to be a solution
def is_solution (b : ℤ) : Prop :=
  b > 4 ∧ ∃ k : ℤ, 4 * b + 5 = k * k

-- State the problem
theorem find_b : ∃ b : ℤ, is_solution b ∧ ∀ b' : ℤ, is_solution b' → b' ≥ 5 := by
  sorry

end find_b_l682_682745


namespace balls_in_boxes_l682_682574

theorem balls_in_boxes : 
  ∃ (n : ℕ), n = 5 ∧ 
    (∀ (partitions : list (ℕ)), (partitions.sum = 5) → 
      (partitions.length ≤ 3) → 
      (∀ (elem : ℕ), elem ∈ partitions → elem.nonneg)) :=
begin
  sorry
end

end balls_in_boxes_l682_682574


namespace divisors_of_36_l682_682568

theorem divisors_of_36 : ∀ n : ℕ, n = 36 → (∃ k : ℕ, k = 9) :=
by
  intro n hn
  have h_prime_factors : (n = 2^2 * 3^2) := by rw hn; norm_num
  -- Using the formula for the number of divisors based on prime factorization
  have h_num_divisors : (2 + 1) * (2 + 1) = 9 := by norm_num
  use 9
  rw h_num_divisors
  sorry

end divisors_of_36_l682_682568


namespace max_area_sum_eq_100_l682_682365

-- Mathematical definitions as per the conditions
variables {α : Type} [LinearOrderedField α]
variables (A B C G O : α) (tanA tanB : α) (AB : α)
def is_triangle (A B C : α) : Prop := sorry  -- This placeholder will hold the formal definition of a triangle

-- Given conditions
axiom tan_condition : tanA * tanB = 3
axiom side_condition : AB = 5

-- Define the maximum possible area for triangle CGO and that it can be written in a specific form
def area_CGO (A B C G O : α) : α := sorry -- Placeholder for area calculation

-- Final proof statement
theorem max_area_sum_eq_100 :
  ∃ (a b c : α) (hgcd : a.gcd c = 1) (hb : ¬(∃ p : α, p^2 ∣ b)),
  area_CGO A B C G O = a * real.sqrt b / c ∧ a + b + c = 100 :=
sorry

end max_area_sum_eq_100_l682_682365


namespace Samuel_fraction_spent_l682_682394

variable (totalAmount receivedRatio remainingAmount : ℕ)
variable (h1 : totalAmount = 240)
variable (h2 : receivedRatio = 3 / 4)
variable (h3 : remainingAmount = 132)

theorem Samuel_fraction_spent (spend : ℚ) : 
  (spend = (1 / 5)) :=
by
  sorry

end Samuel_fraction_spent_l682_682394


namespace smallest_positive_multiple_of_45_l682_682833

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682833


namespace simplify_expression_l682_682670

variable {m : ℝ} (hm : m ≠ 0)

theorem simplify_expression : ( (1 / (3 * m)) ^ (-3) * (2 * m) ^ 4 ) = 432 * m ^ 7 := 
by
  sorry

end simplify_expression_l682_682670


namespace option_a_option_b_option_c_final_result_l682_682111

-- Define the original functions and conditions
def f (ω φ x : ℝ) := Real.sin (ω * x + π / 3 + φ)
def g (ω φ x : ℝ) := Real.sin (ω * x + φ)

-- Prove φ = π / 6 given f(x) is even
theorem option_a (ω : ℝ) (hω : ω > 0) (φ : ℝ) (hφ : abs φ < π / 2) (h_even : ∀ x : ℝ, f ω φ x = f ω φ (-x)) :
  φ = π / 6 :=
sorry

-- Prove ω = 2 / 3 given the smallest positive period of g(x) is 3π
theorem option_b (ω : ℝ) (hφ : abs φ < π / 2) (h_period : ∀ x : ℝ, g ω φ (x + 3 * π) = g ω φ x) :
  ω = 2 / 3 :=
sorry

-- Prove 7 / 3 < ω ≤ 10 / 3 given g(x) has exactly 3 extreme points in the interval (0, π)
theorem option_c (ω : ℝ) (hφ : abs φ < π / 2) (h_extreme_points : ∀ x : ℝ, g ω φ x = g ω φ (x + π / 3) →  ∃! (a b c : ℝ), (0 < a < b < c < π)) :
   7 / 3 < ω ∧ ω ≤ 10 / 3 :=
sorry

-- Define the final result that combines all valid options
theorem final_result (ω : ℝ) (hω : ω > 0) (φ : ℝ) (hφ : abs φ < π / 2) :
  (φ = π / 6 ∧ (∀ x, g ω φ (x + 3 * π) = g ω φ x → ω = 2 / 3) ∧ (∀ x, g ω φ x = g ω φ (x + π / 3) → 3 < ω ∧ ω ≤ 10 / 3)) :=
sorry

end option_a_option_b_option_c_final_result_l682_682111


namespace calculate_expression_l682_682021

noncomputable def solve_expression : ℝ :=
  let term1 := (real.pi - 1) ^ 0
  let term2 := 4 * real.sin (real.pi / 4) -- sin 45° = sin (π/4)
  let term3 := real.sqrt 8
  let term4 := real.abs (-3)
  term1 + term2 - term3 + term4

theorem calculate_expression : solve_expression = 4 := by
  sorry

end calculate_expression_l682_682021


namespace find_interval_n_l682_682074

theorem find_interval_n 
  (n : ℕ) 
  (h1 : n < 500)
  (h2 : (∃ abcde : ℕ, 0 < abcde ∧ abcde < 99999 ∧ n * abcde = 99999))
  (h3 : (∃ uvw : ℕ, 0 < uvw ∧ uvw < 999 ∧ (n + 3) * uvw = 999)) 
  : 201 ≤ n ∧ n ≤ 300 := 
sorry

end find_interval_n_l682_682074


namespace LeibnizTriangleElement_l682_682990

theorem LeibnizTriangleElement (n k : ℕ) :
  L n k = 1 / ((n + 1) * nat.choose n k) := 
sorry

end LeibnizTriangleElement_l682_682990


namespace angle_LAD_105_l682_682120

-- Define points and conditions
def parallelogram (K L M N : Point) : Prop :=
  is_parallelogram K L M N

def on_segment (p a b : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • a + t • b

-- Main statement
theorem angle_LAD_105
  (K L M N A B C D : Point)
  (h_parallelogram : parallelogram K L M N)
  (h_KL : dist K L = 8)
  (h_KN : dist K N = (3 * real.sqrt 2) + real.sqrt 6)
  (h_angle_LKN : angle K L N = π / 4) -- 45 degrees in radians
  (h_A_on_KL : on_segment A K L ∧ dist K A / dist A L = 3 / 1)
  (h_B_on_parallel_line : parallel (line_through A (A + (L - K))) (line_through L M))
  (h_C_on_KN : on_segment C K N ∧ dist K C = dist A B)
  (h_D_intersection : intersect_at (line_through L C) (line_through M B) D) :
  angle L A D = 105 * π / 180 := -- 105 degrees in radians
sorry

end angle_LAD_105_l682_682120


namespace simplify_expression_l682_682673

variable {m : ℝ} (hm : m ≠ 0)

theorem simplify_expression (m : ℝ) (h : m ≠ 0) : 
  ( (1 / (3 * m)) ^ (-3) * (2 * m) ^ 4 ) = 432 * m ^ 7 := by
  sorry

end simplify_expression_l682_682673


namespace smallest_positive_multiple_of_45_l682_682860

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682860


namespace smallest_positive_multiple_of_45_l682_682851

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682851


namespace problem_solution_l682_682527

noncomputable def length_segment_AB : ℝ :=
  let k : ℝ := 1 -- derived from 3k - 3 = 0
  let A : ℝ × ℝ := (0, k) -- point (0, k)
  let C : ℝ × ℝ := (3, -1) -- center of the circle
  let r : ℝ := 1 -- radius of the circle
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) -- distance formula
  Real.sqrt (AC^2 - r^2)

theorem problem_solution :
  length_segment_AB = 2 * Real.sqrt 3 :=
by
  sorry

end problem_solution_l682_682527


namespace factorial_division_l682_682454

theorem factorial_division :
  52! / 50! = 2652 := by
  sorry

end factorial_division_l682_682454


namespace point_P_coordinates_l682_682602

-- Definitions based on conditions
def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs P.2 = d

def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs P.1 = d

-- The theorem statement based on the proof problem
theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, 
    in_fourth_quadrant P ∧ 
    distance_to_x_axis P 2 ∧ 
    distance_to_y_axis P 3 ∧ 
    P = (3, -2) :=
by
  sorry

end point_P_coordinates_l682_682602


namespace degree_of_resulting_polynomial_l682_682734

def p (x : ℝ) : ℝ := (3 * x^5 + 2 * x^3 - x - 15) * (2 * x^12 - 8 * x^8 + 7 * x^5 + 25)
def q (x : ℝ) : ℝ := (x^3 + 3)^6

theorem degree_of_resulting_polynomial : polynomial.degree (p x - q x) = 18 :=
sorry

end degree_of_resulting_polynomial_l682_682734


namespace probability_N_lt_L_is_zero_l682_682080

variable (M N L O : ℝ)
variable (hNM : N < M)
variable (hLO : L > O)

theorem probability_N_lt_L_is_zero : 
  (∃ (permutations : List (ℝ → ℝ)), 
  (∀ perm : ℝ → ℝ, perm ∈ permutations → N < M ∧ L > O) ∧ 
  ∀ perm : ℝ → ℝ, N > L) → false :=
by {
  sorry
}

end probability_N_lt_L_is_zero_l682_682080


namespace equal_probability_after_adding_balls_l682_682208

theorem equal_probability_after_adding_balls :
  let initial_white := 2
  let initial_yellow := 3
  let added_white := 4
  let added_yellow := 3
  let total_white := initial_white + added_white
  let total_yellow := initial_yellow + added_yellow
  let total_balls := total_white + total_yellow
  (total_white / total_balls) = (total_yellow / total_balls) := by
  sorry

end equal_probability_after_adding_balls_l682_682208


namespace range_of_a_l682_682165

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * Real.cos (Real.pi / 2 - x)

theorem range_of_a (a : ℝ) (h_condition : f (2 * a ^ 2) + f (a - 3) + f 0 < 0) : -3/2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l682_682165


namespace measure_AFE_175_l682_682605

-- Definitions of points and angles
variable (A B C D E F : Type) 
variable [Square A B C D]

-- Extension condition
variable (extension : Line CD E)
variable (angleCDE : CDEAngle E = 100)

-- Isosceles triangle condition
variable (isosceles : Isosceles DE DF)

-- Proposition to determine the measure of angle AFE
def measure_angle_AFE : Prop :=
  ∃ (m : ℝ), m = 175

theorem measure_AFE_175 : measure_angle_AFE A B C D E F extension angleCDE isosceles :=
  by sorry

end measure_AFE_175_l682_682605


namespace probability_nina_taller_than_lena_is_zero_l682_682092

-- Definition of participants and conditions
variable (M N L O : ℝ)

-- Conditions
def condition1 := N < M
def condition2 := L > O

-- Statement: Given conditions, the probability that N > L is 0
theorem probability_nina_taller_than_lena_is_zero
  (h1 : condition1)
  (h2 : condition2) :
  (P : ℝ) = 0 :=
by
  sorry

end probability_nina_taller_than_lena_is_zero_l682_682092


namespace smallest_positive_multiple_of_45_l682_682957

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682957


namespace solution_x_l682_682382

open Real

def cos_squared (θ : ℝ) := (1 + cos(2 * θ)) / 2

theorem solution_x (k : ℤ) : 
(∃ x : ℝ, 5.12 * cos_squared (3 * x) + cos_squared (4 * x) + cos_squared (5 * x) = 1.5) ↔ 
(∃ x : ℝ, x = (π / 16) * (2 * k + 1) ∨ x = (π / 3) * (3 * k ± 1) ) :=
sorry

end solution_x_l682_682382


namespace AF_perp_BE_l682_682500

variables {V : Type*} [InnerProductSpace ℝ V]

-- Define the points and their vectors
variables (A B C D E F : V)
variables (a b c d : V)

-- Conditions from the problem
def AB_eq_AD (A B D : V) (ab ad : V) : Prop := ab = ad
def angle_ABC_is_90 (A B C : V) (ab bc : V) : Prop := inner ab bc = 0
def angle_ADC_is_90 (A D C : V) (ad dc : V) : Prop := inner ad dc = 0
def DF_perp_AE (D F A E : V) (df ae : V) : Prop := inner df ae = 0

-- Main statement: AF is perpendicular to BE
theorem AF_perp_BE (a b c d : V) 
  (h1 : AB_eq_AD A B D b a)
  (h2 : angle_ABC_is_90 A B C b (C - B))
  (h3 : angle_ADC_is_90 A D C a (C - D))
  (h4 : DF_perp_AE D F A E (b + d - a) (a + c)) :
  inner (b + d) (a + c - b) = 0 := sorry

end AF_perp_BE_l682_682500


namespace calculate_expression_l682_682018

noncomputable def solve_expression : ℝ :=
  let term1 := (real.pi - 1) ^ 0
  let term2 := 4 * real.sin (real.pi / 4) -- sin 45° = sin (π/4)
  let term3 := real.sqrt 8
  let term4 := real.abs (-3)
  term1 + term2 - term3 + term4

theorem calculate_expression : solve_expression = 4 := by
  sorry

end calculate_expression_l682_682018


namespace expected_segments_to_intersect_l682_682641

noncomputable def expected_segments (origin : ℝ × ℝ) : ℝ :=
  let p := 1 / 2 in -- Probability of hitting an already drawn segment is 0.5
  let series := ∑' n from 3, (n : ℝ) * p^((n-2) : ℝ) in
  series

theorem expected_segments_to_intersect : expected_segments (0, 0) = 5 := 
sorry

end expected_segments_to_intersect_l682_682641


namespace inverse_of_3_mod_257_l682_682053

theorem inverse_of_3_mod_257 :
  ∃ x : ℕ, x < 257 ∧ (3 * x) % 257 = 1 := by
  use 86
  split
  sorry

end inverse_of_3_mod_257_l682_682053


namespace f_is_odd_and_periodic_and_correct_for_2011_l682_682140

noncomputable def f : ℝ → ℝ :=
sorry

theorem f_is_odd_and_periodic_and_correct_for_2011 :
  (∀ x, f(-x) = -f(x)) ∧
  (∀ x, f(x + 4) = f(x)) ∧
  (∀ x, 0 < x ∧ x < 2 → f(x) = Real.logb 2 (3 * x + 1)) →
  f(2011) = -2 :=
sorry

end f_is_odd_and_periodic_and_correct_for_2011_l682_682140


namespace sum_minimum_at_24_l682_682172

noncomputable def a (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S (n : ℕ) : ℤ := (Finset.range n).sum (λ k, a k)

theorem sum_minimum_at_24 : (∀ m : ℕ, S m >= S 24) :=
sorry

end sum_minimum_at_24_l682_682172


namespace smallest_positive_multiple_45_l682_682907

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682907


namespace clock_displays_unique_digits_minutes_l682_682309

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end clock_displays_unique_digits_minutes_l682_682309


namespace branches_sum_of_digits_l682_682031

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem branches_sum_of_digits : ∀ (n : ℕ),
  (⌈n / 3⌉ - ⌈n / 7⌉ = 10) →
  (sum_of_digits (multiset.sum [n | ⌈n / 3⌉ - ⌈n / 7⌉ = 10]) = 11) :=
by
  sorry

end branches_sum_of_digits_l682_682031


namespace pats_password_length_l682_682647

/-- Pat’s computer password is made up of several kinds of alphanumeric and symbol characters for security.
  He uses:
  1. A string of eight random lowercase letters.
  2. A string half that length of alternating upper case letters and numbers.
  3. One symbol on each end of the password.

  Prove that the total number of characters in Pat's computer password is 14.
-/ 
theorem pats_password_length : 
  let lowercase_len := 8 in
  let alternating_len := lowercase_len / 2 in
  let symbols := 2 in
  lowercase_len + alternating_len + symbols = 14 := 
by 
  -- definitions
  let lowercase_len : Nat := 8
  let alternating_len : Nat := lowercase_len / 2
  let symbols : Nat := 2
  
  -- calculation
  have total_length := lowercase_len + alternating_len + symbols
  
  -- assertion
  show total_length = 14 from sorry

end pats_password_length_l682_682647


namespace five_letter_words_count_l682_682613

theorem five_letter_words_count : 
  let num_letters := 26
  in num_letters * num_letters^3 = 456976 :=
by
  let num_letters := 26
  show num_letters * num_letters^3 = 456976
  sorry

end five_letter_words_count_l682_682613


namespace math_expression_equivalent_l682_682024

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end math_expression_equivalent_l682_682024


namespace range_of_b_l682_682168

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (a b x : ℝ) := (1 / 2) * a * x^2 - b * x
noncomputable def h (a b x : ℝ) := f x - g a b x

theorem range_of_b (a b : ℝ) (x₁ x₂ : ℝ) :
  g a b 2 = 2 ∧ a < 0 ∧ a = 0 ∧ h a b x₁ = 0 ∧ h a b x₂ = 0 ∧ x₁ ≠ x₂ 
  → b ∈ Ioo (-1 / Real.exp 1) 0 := 
sorry

end range_of_b_l682_682168


namespace add_to_fraction_eq_l682_682377

theorem add_to_fraction_eq (n : ℕ) : (4 + n) / (7 + n) = 6 / 7 → n = 14 :=
by sorry

end add_to_fraction_eq_l682_682377


namespace translate_parabola_l682_682723

theorem translate_parabola (x : ℝ) : 
  let y_initial := 3 * x^2,
      y_translated_left := 3 * (x + 1)^2,
      y_final := y_translated_left + 2
  in y_final = 3 * (x + 1)^2 + 2 := 
by
  sorry

end translate_parabola_l682_682723


namespace smallest_positive_multiple_45_l682_682909

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682909


namespace smallest_positive_multiple_of_45_l682_682824

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682824


namespace increasing_interval_func_l682_682035

section

def func (x : ℝ) : ℝ := -real.log (x^2 - 3*x + 2)

def interval_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y

theorem increasing_interval_func : interval_increasing func (-∞) 1 :=
sorry

end

end increasing_interval_func_l682_682035


namespace number_of_divisors_of_36_l682_682563

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end number_of_divisors_of_36_l682_682563


namespace nine_pow_eq_eighty_one_l682_682187

theorem nine_pow_eq_eighty_one (x : ℝ) (h : 9 ^ x = 81) : x = 2 := by
  sorry

end nine_pow_eq_eighty_one_l682_682187


namespace polynomial_divisible_by_x_sub_a_squared_l682_682275

theorem polynomial_divisible_by_x_sub_a_squared (a x : ℕ) (n : ℕ) 
    (h : a ≠ 0) : ∃ q : ℕ → ℕ, x ^ n - n * a ^ (n - 1) * x + (n - 1) * a ^ n = (x - a) ^ 2 * q x := 
by 
  sorry

end polynomial_divisible_by_x_sub_a_squared_l682_682275


namespace log_sum_tangent_intersections_l682_682166

noncomputable def f (n : ℕ+) : ℝ → ℝ := λ x, x^(n + 1)

def tangent_intersection_x (n : ℕ+) : ℝ := (n : ℝ) / (n + 1)

theorem log_sum_tangent_intersections : 
  (∑ k in finset.range 2012, real.log 2013 (tangent_intersection_x (k + 1 : ℕ+))) = -1 :=
sorry

end log_sum_tangent_intersections_l682_682166


namespace fifth_term_arithmetic_sequence_l682_682325

variables (x y : ℝ)

def a1 := 2 * x + 3 * y 
def a2 := 2 * x - 3 * y
def a3 := 4 * x * y
def a4 := 2 * x / (3 * y)

def common_difference (a b : ℝ) := b - a

theorem fifth_term_arithmetic_sequence :
  let d := common_difference a1 a2,
      d1 := common_difference a3 a2,
      d2 := common_difference a4 a3 in
  d = d1 ∧ d1 = d2 → 
  (2 * x / (3 * y)) + d = (54 * y^2 - 15) / (2 * y - 1) :=
by
  sorry

end fifth_term_arithmetic_sequence_l682_682325


namespace smallest_positive_multiple_of_45_l682_682843

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682843


namespace smallest_positive_multiple_of_45_l682_682932

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682932


namespace simplify_expr1_simplify_expr2_l682_682281

variable (a b t : ℝ)

theorem simplify_expr1 : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l682_682281


namespace probability_nina_taller_than_lena_l682_682103

variables {M N L O : ℝ}

theorem probability_nina_taller_than_lena (h₁ : N < M) (h₂ : L > O) : 
  ∃ P : ℝ, P = 0 ∧ ∀ M N L O, M ≠ N ∧ M ≠ L ∧ M ≠ O ∧ N ≠ L ∧ N ≠ O ∧ L ≠ O → 
  (M > N → O < L → P = 0) :=
by sorry

end probability_nina_taller_than_lena_l682_682103


namespace smallest_positive_multiple_of_45_is_45_l682_682791

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682791


namespace find_b_l682_682362

open Real EuclideanSpace

def a : ℝ^3 := ![2, 2, 2]
def b : ℝ^3 := ![8, -2, -4]

theorem find_b : ∃ b : ℝ^3, ∃ t : ℝ, 
  (b = ![(8:ℝ), -2, -4] - t • ![(2:ℝ), 2, 2]) ∧ 
  (a ⬝ b = 0) ∧ 
  (b = ![(22/3 : ℝ), -(8/3), -(14/3)]) :=
begin
  sorry
end

end find_b_l682_682362


namespace smallest_positive_multiple_45_l682_682807

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682807


namespace ratio_brown_eyes_l682_682716

theorem ratio_brown_eyes (total_people : ℕ) (blue_eyes : ℕ) (black_eyes : ℕ) (green_eyes : ℕ) (brown_eyes : ℕ) 
    (h1 : total_people = 100) 
    (h2 : blue_eyes = 19) 
    (h3 : black_eyes = total_people / 4) 
    (h4 : green_eyes = 6) 
    (h5 : brown_eyes = total_people - (blue_eyes + black_eyes + green_eyes)) : 
    brown_eyes / total_people = 1 / 2 :=
by sorry

end ratio_brown_eyes_l682_682716


namespace probability_N_lt_L_is_zero_l682_682081

variable (M N L O : ℝ)
variable (hNM : N < M)
variable (hLO : L > O)

theorem probability_N_lt_L_is_zero : 
  (∃ (permutations : List (ℝ → ℝ)), 
  (∀ perm : ℝ → ℝ, perm ∈ permutations → N < M ∧ L > O) ∧ 
  ∀ perm : ℝ → ℝ, N > L) → false :=
by {
  sorry
}

end probability_N_lt_L_is_zero_l682_682081


namespace antiparallelogram_sym_trapezoid_l682_682730

theorem antiparallelogram_sym_trapezoid (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  (AB CD BC DA : ℝ) (h1 : AB = CD) (h2 : BC = DA) (h3 : A ≠ B) (h4 : B ≠ C) (h5 : C ≠ D) (h6 : D ≠ A) 
  (h7 : AB > 0) (h8 : BC > 0) 
  (O : Type) [metric_space O] (h9 : is_interior A B O) (h10 : is_interior C D O) : 
  ∃ h11 : is_parallel AD BC, h12 : AD = BC, is_symmetric_trapezoid AC BD AD BC := by
  sorry

end antiparallelogram_sym_trapezoid_l682_682730


namespace smallest_positive_multiple_of_45_l682_682853

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682853


namespace green_ball_probability_l682_682458

-- Defining the number of red and green balls in each container
def containerI_red : ℕ := 10
def containerI_green : ℕ := 5

def containerII_red : ℕ := 3
def containerII_green : ℕ := 6

def containerIII_red : ℕ := 3
def containerIII_green : ℕ := 6

-- Probability of selecting any container
def prob_container : ℚ := 1 / 3

-- Defining the probabilities of drawing a green ball from each container
def prob_green_I : ℚ := containerI_green / (containerI_red + containerI_green)
def prob_green_II : ℚ := containerII_green / (containerII_red + containerII_green)
def prob_green_III : ℚ := containerIII_green / (containerIII_red + containerIII_green)

-- Law of total probability
def prob_green_total : ℚ :=
  prob_container * prob_green_I +
  prob_container * prob_green_II +
  prob_container * prob_green_III

-- The mathematical statement to be proven
theorem green_ball_probability :
  prob_green_total = 5 / 9 := by
  sorry

end green_ball_probability_l682_682458


namespace smallest_positive_multiple_of_45_l682_682871

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682871


namespace incandescent_percentage_correct_l682_682984

noncomputable def percent_incandescent_bulbs_switched_on 
  (I F : ℕ) 
  (h1 : 0.30 * I + 0.80 * F = 0.70 * (I + F)) : ℝ :=
(0.30 * I / (0.70 * (I + F))) * 100

theorem incandescent_percentage_correct 
  (I F : ℕ) 
  (h1 : 0.30 * I + 0.80 * F = 0.70 * (I + F)) : 
  percent_incandescent_bulbs_switched_on I F h1 ≈ 8.57 :=
by {
  calc percent_incandescent_bulbs_switched_on I F h1
        = (0.30 * I / (0.70 * (I + F))) * 100 : rfl
    ... = (0.30 / 0.70 * I / (I + F)) * 100 : by rw [mul_div_assoc, mul_div_cancel, div_mul_eq_div]
    ... = (3 / 7 * I / (I + F)) * 100 : by norm_num
    ... = (3 * I) / (7 * (I + F)) * 100 : by rw mul_div_assoc
    ... = (3 * 100 * I) / (7 * (I + F)) : by rw mul_assoc
    ... = (300 * I) / ((7 * I) + (7 * F)) : by rw [div_mul_eq_div, div_div]
    ... = 8.57 : sorry
}

end incandescent_percentage_correct_l682_682984


namespace smallest_a_for_5880_to_be_cube_l682_682486

theorem smallest_a_for_5880_to_be_cube : ∃ (a : ℕ), a > 0 ∧ (∃ (k : ℕ), 5880 * a = k ^ 3) ∧
  (∀ (b : ℕ), b > 0 ∧ (∃ (k : ℕ), 5880 * b = k ^ 3) → a ≤ b) ∧ a = 1575 :=
sorry

end smallest_a_for_5880_to_be_cube_l682_682486


namespace distance_from_A_after_1990_moves_l682_682205

-- Define the conditions
def square_side_length : ℝ := 8
def triangle_side_length : ℝ := 5

-- Function to compute the distance after a given number of moves
def distance_after_moves (moves : ℕ) : ℝ :=
  let full_cycle := 48
  let remaining_moves := moves % full_cycle
  let side_length := 8 -- Side length of the square
  if remaining_moves < 16 then
    -- Case within the first side AB
    (remaining_moves * side_length / 16)
  else if remaining_moves < 32 then
    -- Case on side BC
    (remaining_moves - 16) * side_length / 16
  else
    -- Case on side CD, we do not need the exact formula as we have clear periodicity
    (remaining_moves - 32) * side_length / 16

theorem distance_from_A_after_1990_moves : distance_after_moves 1990 = 3 :=
sorry

end distance_from_A_after_1990_moves_l682_682205


namespace ellipse_equation_line_equation_with_max_slope_product_l682_682533

variables (a b : ℝ) (C : Set (ℝ × ℝ)) (M : ℝ × ℝ) (eccentricity : ℝ)
  (Q P : ℝ × ℝ) (l : Affine Line ℝ ℝ) (k1 k2 : ℝ)

noncomputable def ellipse := { p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x^2 / a^2 + y^2 / b^2 = 1 }

def point_M := (sqrt 2, 1)
def point_Q := (1, 0)
def point_P := (4, 3)
def e := sqrt 2 / 2

theorem ellipse_equation :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ e ^ 2 = (a^2 - b^2) / a^2 ∧ point_M ∈ ellipse a b) →
  C = { p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x^2 / 4 + y^2 / 2 = 1 } :=
by sorry

theorem line_equation_with_max_slope_product :
  (∃ (A B : ℝ × ℝ), A ∈ ellipse 2 (sqrt 2) ∧ B ∈ ellipse 2 (sqrt 2) ∧ 
  line_through Q A ∧ line_through Q B ∧ 
  line l ∈ { p : ℝ × ℝ | ∃ m b, p = (x, mx + b) ∧ (P.2 - y) / (P.1 - x) = k1 ∧ (P.2 - y) / (P.1 - x) = k2 ∧ 
  k1 * k2 = 1}) →
  (∃ m : ℝ, m = 1 ∧ l = {x, y | x - y - 1 = 0}) :=
by sorry

end ellipse_equation_line_equation_with_max_slope_product_l682_682533


namespace smallest_positive_multiple_of_45_l682_682969

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682969


namespace perpendicular_aa2_go_l682_682389

variables {A B C G O : Type} [real_inner_product_space A] [real_inner_product_space B] [real_inner_product_space C]
variables (sideBC sideAB sideAC : ℝ)
variables (A0 A1 A2 : real_inner_product_space.point A)

-- Given conditions from the problem
def geometric_mean_bc_ab_ac (B C : real_inner_product_space.point A) (A : real_inner_product_space.point A): Prop :=
  sideBC = real.sqrt (sideAB * sideAC)

def midpoint_bc (B C : real_inner_product_space.point A) (A0 : real_inner_product_space.point A): Prop :=
  A0 = (B + C) / 2

def angle_bisector_intersection (A B C : real_inner_product_space.point A) (A1 : real_inner_product_space.point A): Prop :=
  real_inner_product_space.angle A B C / 2 = real_inner_product_space.angle A B A1

def reflection_over_midpoint (A0 A1 A2 : real_inner_product_space.point A): Prop :=
  A2 = (2 * A0 - A1)

-- The theorem to be proven
theorem perpendicular_aa2_go (A B C G O A0 A1 A2 : real_inner_product_space.point A)
  (h1 : geometric_mean_bc_ab_ac B C A)
  (h2 : midpoint_bc B C A0)
  (h3 : angle_bisector_intersection A B C A1)
  (h4 : reflection_over_midpoint A0 A1 A2)
  : real_inner_product_space.orthogonal (real_inner_product_space.line_through A A2) (real_inner_product_space.line_through G O) :=
sorry

end perpendicular_aa2_go_l682_682389


namespace smallest_positive_multiple_of_45_is_45_l682_682946

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682946


namespace smallest_positive_multiple_of_45_l682_682762

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682762


namespace value_of_m_l682_682106

theorem value_of_m
  (m : ℝ)
  (a : ℝ × ℝ := (-1, 3))
  (b : ℝ × ℝ := (m, m - 2))
  (collinear : a.1 * b.2 = a.2 * b.1) :
  m = 1 / 2 :=
sorry

end value_of_m_l682_682106


namespace constant_term_expansion_l682_682439

theorem constant_term_expansion : 
  let term := (x - 1 / (4 * x)) ^ 6 in
  let r := 3 in
  let binom_6_3 := Nat.choose 6 3 in
  let general_term := (-1 / 4) ^ r * binom_6_3 * (x ^ (6 - 2 * r)) in
  (6 - 2 * r = 0) → (general_term = -5 / 16) := 
by
  intros term r binom_6_3 general_term h
  sorry

end constant_term_expansion_l682_682439


namespace factorial_representation_2021_l682_682050

theorem factorial_representation_2021 :
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧ 
    (∃ p q : ℕ, (a! * p! = b! * q! * 2021) ∧ a ≥ p ∧ b ≥ q) ∧ 
    |a - b| = 4 ∧ 
    (∀ x y : ℕ, x > 0 ∧ y > 0 → ((x + y) < (a + b) → (¬ ∃ p q : ℕ, (x! * p! = y! * q! * 2021 ∧ x ≥ p ∧ y ≥ q)))) :=
sorry

end factorial_representation_2021_l682_682050


namespace right_angle_XBY_l682_682625

noncomputable def angle_ABC : Prop := ∠ ABC = 90°
noncomputable def angle_DBE : Prop := ∠ DBE = 90°

def point_X : Type := {X | X ∈ (line AD) ∧ X ∈ (line CE)}
def point_Y : Type := {Y | Y ∈ (line AE) ∧ Y ∈ (line CD)}

theorem right_angle_XBY 
  (h1: angle_ABC) 
  (h2: angle_DBE) 
  (h3: ∃ X, X ∈ point_X) 
  (h4: ∃ Y, Y ∈ point_Y) : 
  ∠ XBY = 90° := sorry

end right_angle_XBY_l682_682625


namespace more_cost_effective_option_1_x_100_costs_options_x_gt_100_most_cost_effective_plan_x_300_l682_682405

open BigOperators

-- Definition of conditions
def desk_price : ℝ := 200
def chair_price : ℝ := 80
def quantity_desks : ℕ := 100

-- Option 1 cost calculation for any number of chairs
def cost_option_1 (x : ℕ) : ℝ :=
  if x ≤ 100 then
    (quantity_desks * desk_price)
  else
    (quantity_desks * desk_price) + (chair_price * (x - quantity_desks))

-- Option 2 cost calculation for any number of chairs
def cost_option_2 (x : ℕ) : ℝ :=
  ((quantity_desks * desk_price) + (x * chair_price)) * 0.8

-- Problem (1) statement
theorem more_cost_effective_option_1_x_100 : cost_option_1 100 < cost_option_2 100 := 
by 
  rw [cost_option_1, cost_option_2] 
  iterate 2 {norm_num}
  sorry

-- Problem (2) statement
theorem costs_options_x_gt_100 (x : ℕ) (h : x > 100) :
  cost_option_1 x = 80 * x + 12000 ∧
  cost_option_2 x = 64 * x + 16000  :=
by
  rw [cost_option_1, cost_option_2]
  split_ifs
  sorry

-- Problem (3) statement
theorem most_cost_effective_plan_x_300 : 
  (cost_option_1 100 + cost_option_2 200 < cost_option_2 300) ∧ 
  (cost_option_1 100 + cost_option_2 200 < cost_option_1 300) := 
by 
  rw [cost_option_1, cost_option_2]
  iterate 5 {norm_num}
  sorry

end more_cost_effective_option_1_x_100_costs_options_x_gt_100_most_cost_effective_plan_x_300_l682_682405


namespace smallest_positive_multiple_of_45_l682_682761

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682761


namespace proportion_equation_l682_682576

theorem proportion_equation (x y : ℝ) (h : 3 * x = 4 * y) (hy : y ≠ 0) : (x / 4 = y / 3) :=
by
  sorry

end proportion_equation_l682_682576


namespace dawn_hourly_income_l682_682034

theorem dawn_hourly_income 
  (n : ℕ) (t_s t_p t_f I_p I_s I_f : ℝ)
  (h_n : n = 12)
  (h_t_s : t_s = 1.5)
  (h_t_p : t_p = 2)
  (h_t_f : t_f = 0.5)
  (h_I_p : I_p = 3600)
  (h_I_s : I_s = 1200)
  (h_I_f : I_f = 300) :
  (I_p + I_s + I_f) / (n * (t_s + t_p + t_f)) = 106.25 := 
  by
  sorry

end dawn_hourly_income_l682_682034


namespace expression_evaluation_l682_682013

theorem expression_evaluation :
  (π - 1)^0 + 4 * real.sin (real.pi / 4) - real.sqrt 8 + abs (-3) = 4 := 
sorry

end expression_evaluation_l682_682013


namespace solve_system_of_equations_l682_682353

theorem solve_system_of_equations :
  ∀ (x y : ℝ), (x + y = 3 ∧ 2 * (x + y) - y = 5) → (x = 2 ∧ y = 1) :=
by
  intro x y
  intro h
  cases h
  unfold at h
  sorry
  -- Proof goes here

end solve_system_of_equations_l682_682353


namespace smallest_positive_multiple_of_45_l682_682863

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682863


namespace emma_liam_sum_difference_l682_682047

-- Definitions for Emma's and Liam's number lists
def emma_numbers : List ℕ := List.range (40) ++ [40]

def replace_digit_3_with_2 (n : ℕ) : ℕ :=
  let digits := n.digits
  digits.foldl (fun acc d => acc * 10 + if d = 3 then 2 else d) 0

def liam_numbers : List ℕ := emma_numbers.map replace_digit_3_with_2

-- Sum of Emma's numbers
def emma_sum : ℕ := emma_numbers.sum

-- Sum of Liam's numbers
def liam_sum : ℕ := liam_numbers.sum

-- Statement of the theorem based on the problem and the solution
theorem emma_liam_sum_difference :
  emma_sum - liam_sum = 104 := sorry

end emma_liam_sum_difference_l682_682047


namespace smallest_positive_multiple_of_45_l682_682826

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682826


namespace smallest_positive_multiple_of_45_l682_682882

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682882


namespace probability_nina_taller_than_lena_is_zero_l682_682091

-- Definition of participants and conditions
variable (M N L O : ℝ)

-- Conditions
def condition1 := N < M
def condition2 := L > O

-- Statement: Given conditions, the probability that N > L is 0
theorem probability_nina_taller_than_lena_is_zero
  (h1 : condition1)
  (h2 : condition2) :
  (P : ℝ) = 0 :=
by
  sorry

end probability_nina_taller_than_lena_is_zero_l682_682091


namespace right_triangle_count_l682_682547

theorem right_triangle_count :
  (∃ n : ℕ, n = 4 ∧ ∀ a b c : ℕ, 
     a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * a + b * b = c * c) ∧ 
     (a * b / 2 = 4 * (a + b + c)) ∧ 
     ((a - 16) * (b - 16) = 128) → 
       (a, b) ∈ 
         { (a, b) : ℕ × ℕ | 
           (a - 16, b - 16) ∈ { (x, y) : ℕ × ℕ // x * y = 128 } } ∧ 
           (a, b) ∉ { (16, 32), (32, 16) } 
  ) := sorry

end right_triangle_count_l682_682547


namespace smallest_positive_multiple_of_45_is_45_l682_682920

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682920


namespace identify_element_l682_682403

-- Definitions based on the conditions
def molecular_weight (compound : Type) (weight : ℝ) : Prop := weight = 100
def consists_of (compound : Type) (elements : List (Type × ℕ)) : Prop :=
  elements = [(Calcium, 1), (Carbon, 1), (Oxygen, 3)]
def is_metal (element : Type) : Prop := element = Calcium
def commonly_found_in_limestone (element : Type) : Prop := element = Calcium

-- The main proof statement
theorem identify_element :
  (∃ compound, molecular_weight compound 100 ∧
               consists_of compound [(Calcium, 1), (Carbon, 1), (Oxygen, 3)] ∧
               (∃ element, is_metal element ∧ commonly_found_in_limestone element)) →
  ∃ element, element = Calcium := by
  sorry

end identify_element_l682_682403


namespace factorial_quotient_52_50_l682_682451

theorem factorial_quotient_52_50 : (Nat.factorial 52) / (Nat.factorial 50) = 2652 := 
by 
  sorry

end factorial_quotient_52_50_l682_682451


namespace smallest_positive_multiple_of_45_is_45_l682_682801

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682801


namespace smallest_positive_multiple_of_45_l682_682958

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682958


namespace triangle_altitude_l682_682294

variable (Area : ℝ) (base : ℝ) (altitude : ℝ)

theorem triangle_altitude (hArea : Area = 1250) (hbase : base = 50) :
  2 * Area / base = altitude :=
by
  sorry

end triangle_altitude_l682_682294


namespace smallest_positive_multiple_of_45_l682_682936

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682936


namespace smallest_positive_multiple_of_45_is_45_l682_682800

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682800


namespace smallest_four_digit_with_sum_15_l682_682736

theorem smallest_four_digit_with_sum_15 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n.digits.sum = 15) ∧
                                                ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m.digits.sum = 15 → n ≤ m :=
by
  sorry

end smallest_four_digit_with_sum_15_l682_682736


namespace cos_2φ_is_3_over_5_l682_682152

-- Define the function f and its symmetry property.
def f (x φ : ℝ) : ℝ := sin (x + φ) - 2 * cos (x + φ)

-- Define the statement to be proved.
theorem cos_2φ_is_3_over_5 (φ : ℝ) (hφ : 0 < φ ∧ φ < π) 
  (h_sym : ∀ x : ℝ, f x φ = f (π - x) φ) : cos (2 * φ) = 3 / 5 :=
by
  -- Placeholder for the proof; the focus is on the statement structure.
  sorry

end cos_2φ_is_3_over_5_l682_682152


namespace find_eccentricity_l682_682506

-- Given conditions
variables (a b : Real) (h₀ : a > b) (h₁ : b > 0)
variables (P : Real × Real) (α β : Real)
variables (h₂ : cos α = sqrt 5 / 5)
variables (h₃ : sin (α + β) = 3 / 5)

-- The point P is on the ellipse
variable h₄ : P.fst ^ 2 / a ^ 2 + P.snd ^ 2 / b ^ 2 = 1

-- The definition of eccentricity
def eccentricity (a c : Real) : Real := c / a

-- The main theorem to prove
theorem find_eccentricity (e : Real) (c : Real) 
  (h₅ : c ^ 2 = a ^ 2 - b ^ 2) 
  (h₆ : e = eccentricity a c) : e = sqrt 5 / 7 := 
sorry

end find_eccentricity_l682_682506


namespace smallest_positive_multiple_of_45_is_45_l682_682947

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682947


namespace smallest_positive_multiple_of_45_l682_682934

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682934


namespace six_n_digit_remains_divisible_by_7_l682_682280

-- Given the conditions
def is_6n_digit_number (N : ℕ) (n : ℕ) : Prop :=
  N < 10^(6*n) ∧ N ≥ 10^(6*(n-1))

def is_divisible_by_7 (N : ℕ) : Prop :=
  N % 7 = 0

-- Define new number M formed by moving the unit digit to the beginning
def new_number (N : ℕ) (n : ℕ) : ℕ :=
  let a_0 := N % 10
  let rest := N / 10
  a_0 * 10^(6*n - 1) + rest

-- The theorem statement
theorem six_n_digit_remains_divisible_by_7 (N : ℕ) (n : ℕ)
  (hN : is_6n_digit_number N n)
  (hDiv7 : is_divisible_by_7 N) : is_divisible_by_7 (new_number N n) :=
sorry

end six_n_digit_remains_divisible_by_7_l682_682280


namespace smallest_positive_multiple_l682_682783

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682783


namespace problem_1_l682_682994

theorem problem_1 (α : ℝ) (k : ℤ) (n : ℕ) (hk : k > 0) (hα : α ≠ k * Real.pi) (hn : n > 0) :
  n = 1 → (0.5 + Real.cos α) = (0.5 + Real.cos α) :=
by
  sorry

end problem_1_l682_682994


namespace smallest_positive_multiple_of_45_is_45_l682_682950

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682950


namespace horner_rule_v3_l682_682367

theorem horner_rule_v3 :
  let f : ℕ → ℕ := λ x, (((3 * x - 1) * x) * x + 2) * x + 1
  let v : ℕ → ℕ := λ k, match k with
    | 0 => 3
    | 1 => 3 * 2 - 1
    | 2 => (3 * 2 - 1) * 2
    | 3 => ((3 * 2 - 1) * 2) * 2
    | _ => 0
  in v 3 = 20 :=
by 
  let f : ℕ → ℕ := λ x, (((3 * x - 1) * x) * x + 2) * x + 1
  let v : ℕ → ℕ := λ k, match k with
    | 0 => 3
    | 1 => 3 * 2 - 1
    | 2 => (3 * 2 - 1) * 2
    | 3 => ((3 * 2 - 1) * 2) * 2
    | _ => 0
  show v 3 = 20
sorry

end horner_rule_v3_l682_682367


namespace smallest_positive_multiple_of_45_l682_682876

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682876


namespace smallest_positive_multiple_of_45_l682_682768

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682768


namespace num_divisors_36_l682_682557

theorem num_divisors_36 : ∃ n, n = 9 ∧ ∀ d : ℕ, d ∣ 36 → (d > 0 ∧ d ≤ 36) → ∃ k : ℕ, d = (2^k or 3^k or (2^p * 3^q) where p, q <= 2):
begin
  have factorization : 36 = 2^2 * 3^2 := by norm_num,
  have exponents : (2+1)*(2+1) = 9 := by norm_num,
  use 9,
  split,
  { exact exponents },
  {
    intros d hd hpos_range,
    cases hpos_range with hpos hrange,
    sorry -- Proof showing that there are exactly 9 positive divisors.
  },
end

end num_divisors_36_l682_682557


namespace zero_in_interval_l682_682358

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval : 2 ≤ x ∧ x ≤ 3 ∧ f 2 < 0 ∧ f 3 > 0 → ∃ c ∈ (2 : ℝ)..3, f c = 0 :=
by
  intro h
  sorry

end zero_in_interval_l682_682358


namespace smallest_positive_multiple_of_45_l682_682894

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682894


namespace smallest_positive_multiple_of_45_l682_682869

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682869


namespace smallest_positive_multiple_of_45_is_45_l682_682923

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682923


namespace largest_n_value_l682_682066

-- Define the conditions
def quadratic_expr (n : ℤ) : polynomial ℤ :=
  3 * polynomial.X ^ 2 + polynomial.C n * polynomial.X + 54

def factorizable_as_linear_factors (p : polynomial ℤ) : Prop :=
  ∃ A B : ℤ, p = (3 * polynomial.X + polynomial.C A) * (polynomial.X + polynomial.C B)

-- Statement of the theorem
theorem largest_n_value : ∃ n : ℤ, 
  (quadratic_expr n).factorizable_as_linear_factors ∧ 
  (∀ m : ℤ, (quadratic_expr m).factorizable_as_linear_factors → m ≤ 163) :=
sorry

end largest_n_value_l682_682066


namespace neg_distance_represents_west_l682_682191

def represents_east (distance : Int) : Prop :=
  distance > 0

def represents_west (distance : Int) : Prop :=
  distance < 0

theorem neg_distance_represents_west (pos_neg : represents_east 30) :
  represents_west (-50) :=
by
  sorry

end neg_distance_represents_west_l682_682191


namespace smallest_positive_multiple_of_45_l682_682755

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682755


namespace smallest_positive_multiple_of_45_l682_682821

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682821


namespace lights_on_after_2011_toggles_l682_682678

-- Definitions for light states and index of lights
inductive Light : Type
| A | B | C | D | E | F | G
deriving DecidableEq

-- Initial light state: function from Light to Bool (true means the light is on)
def initialState : Light → Bool
| Light.A => true
| Light.B => false
| Light.C => true
| Light.D => false
| Light.E => true
| Light.F => false
| Light.G => true

-- Toggling function: toggles the state of a given light
def toggleState (state : Light → Bool) (light : Light) : Light → Bool :=
  fun l => if l = light then ¬ (state l) else state l

-- Toggling sequence: sequentially toggle lights in the given list
def toggleSequence (state : Light → Bool) (seq : List Light) : Light → Bool :=
  seq.foldl toggleState state

-- Toggles the sequence n times
def toggleNTimes (state : Light → Bool) (seq : List Light) (n : Nat) : Light → Bool :=
  let rec aux (state : Light → Bool) (n : Nat) : Light → Bool :=
    if n = 0 then state
    else aux (toggleSequence state seq) (n - 1)
  aux state n

-- Toggling sequence: A, B, C, D, E, F, G
def toggleSeq : List Light := [Light.A, Light.B, Light.C, Light.D, Light.E, Light.F, Light.G]

-- Determine the final state after 2011 toggles
def finalState : Light → Bool := toggleNTimes initialState toggleSeq 2011

-- Proof statement: the state of the lights after 2011 toggles is such that lights A, D, F are on
theorem lights_on_after_2011_toggles :
  finalState Light.A = true ∧
  finalState Light.D = true ∧
  finalState Light.F = true ∧
  finalState Light.B = false ∧
  finalState Light.C = false ∧
  finalState Light.E = false ∧
  finalState Light.G = false :=
by
  sorry

end lights_on_after_2011_toggles_l682_682678


namespace validTwoDigitXsCount_l682_682626

def sumDigits (n : ℕ) : ℕ := n.digits 10 |>.sum

def isValidTwoDigit (x : ℕ) : Prop := (10 ≤ x) ∧ (x ≤ 99)

def countValidXs : ℕ := (Finset.range 100).filter (λ x => isValidTwoDigit x ∧ sumDigits (sumDigits x) = 4) |>.card

theorem validTwoDigitXsCount : countValidXs = 10 := by
  sorry

end validTwoDigitXsCount_l682_682626


namespace sum_inradii_is_correct_l682_682195

noncomputable def sum_inradii_of_inscribed_circles
  (A B C D : Point)
  (h_AB : dist A B = 7)
  (h_AC : dist A C = 9)
  (h_BC : dist B C = 12)
  (h_BD : dist B D = 2)
  (h_DC : dist D C = 10) : ℝ :=
  let s_ADB := (dist A B + dist A D + dist B D) / 2 in
  let s_ADC := (dist A C + dist A D + dist D C) / 2 in
  let area_ADB := sqrt (s_ADB * (s_ADB - dist A B) * (s_ADB - dist A D) * (s_ADB - dist B D)) in
  let area_ADC := sqrt (s_ADC * (s_ADC - dist A C) * (s_ADC - dist A D) * (s_ADC - dist D C)) in
  let inradius_ADB := area_ADB / s_ADB in
  let inradius_ADC := area_ADC / s_ADC in
  inradius_ADB + inradius_ADC

theorem sum_inradii_is_correct
  (A B C D : Point)
  (h_AB : dist A B = 7)
  (h_AC : dist A C = 9)
  (h_BC : dist B C = 12)
  (h_BD : dist B D = 2)
  (h_DC : dist D C = 10) :
  sum_inradii_of_inscribed_circles A B C D h_AB h_AC h_BC h_BD h_DC = (10 + 6 * Real.sqrt 5) / 4 :=
by
  sorry

end sum_inradii_is_correct_l682_682195


namespace brookdale_avg_temp_l682_682412

def highs : List ℤ := [51, 64, 60, 59, 48, 55]
def lows : List ℤ := [42, 49, 47, 43, 41, 44]

def average_temperature : ℚ :=
  let total_sum := highs.sum + lows.sum
  let count := (highs.length + lows.length : ℚ)
  total_sum / count

theorem brookdale_avg_temp :
  average_temperature = 49.4 :=
by
  -- The proof goes here
  sorry

end brookdale_avg_temp_l682_682412


namespace correct_statements_l682_682378

-- Definitions corresponding to the problem conditions
def statement_A {f : ℝ → ℝ} (x₀ : ℝ) : Prop :=
  f' x₀ = (fun x => f x - f x₀ - f' x₀ * (x - x₀)) / (x - x₀)

def statement_B {f : ℝ → ℝ} (x₀ : ℝ) : Prop :=
  f' x₀ = deriv (fun _ => f x₀) x₀

def statement_C {s : ℝ → ℝ} (t₀ : ℝ) : Prop :=
  s' t₀ = (fun t => s t - s t₀) / (t - t₀)

def statement_D {v : ℝ → ℝ} (t₀ : ℝ) : Prop :=
  v' t₀ = (fun t => v t - v t₀) / (t - t₀)

-- The main theorem to prove that statements A, C, and D are correct
theorem correct_statements {f s v : ℝ → ℝ} (x₀ t₀ : ℝ) :
  statement_A x₀ ∧ statement_C t₀ ∧ statement_D t₀ :=
by
  split
  -- Proof for statement_A
  apply sorry
  split
  -- Proof for statement_C
  apply sorry
  -- Proof for statement_D
  apply sorry

end correct_statements_l682_682378


namespace monomials_eq_fibonacci_l682_682277

-- Define the sequence a_n
def a : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := a n + a (n+1)

-- Define the Fibonacci sequence F_n
def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := F n + F (n+1)

-- Prove that a_n equals F_{n+1}
theorem monomials_eq_fibonacci (n : ℕ) : a n = F (n + 1) := by
  sorry

end monomials_eq_fibonacci_l682_682277


namespace choose_best_set_l682_682416

noncomputable def expectation_set_a (a1_idea_correct : ℚ) (a1_noidea_correct : ℚ) (a1_have_idea : ℚ)
  (a1_no_idea: ℚ) : ℚ :=
  let p_x_0  := (a1_no_idea * (1 - a1_noidea_correct)) * (a1_have_idea * (1 - a1_idea_correct)) +
                (a1_have_idea * (1 - a1_idea_correct))^2 in
  let p_x_1  := (a1_no_idea * (1 - a1_noidea_correct)) * (a1_have_idea * a1_idea_correct * 2) + 
                a1_have_idea^2 * a1_idea_correct * (1 - a1_idea_correct) in
  let p_x_2  := (a1_no_idea * a1_noidea_correct) * (a1_have_idea * a1_idea_correct) +
                (a1_have_idea * a1_idea_correct)^2 in
  0 * p_x_0 + 1 * p_x_1 + 2 * p_x_2

noncomputable def expectation_set_b (b_correct : ℚ) : ℚ :=
  2 * b_correct

theorem choose_best_set (a1_idea_correct : ℚ) (a1_noidea_correct : ℚ)
  (a1_have_idea : ℚ) (a1_no_idea: ℚ) (b_correct : ℚ) :
  let E_X := expectation_set_a a1_idea_correct a1_noidea_correct a1_have_idea a1_no_idea in
  let E_Y := expectation_set_b b_correct in
  E_X = 9 / 8 ∧ E_Y = 1.2 ∧ E_Y > E_X :=
begin
  sorry
end

variables (a1_idea_correct : ℚ := 2 / 3) (a1_noidea_correct : ℚ := 1 / 4)
  (a1_have_idea : ℚ := 3 / 4) (a1_no_idea : ℚ := 1 / 4) (b_correct : ℚ := 0.6)

#eval show E_Y > E_X, from choose_best_set a1_idea_correct a1_noidea_correct a1_have_idea a1_no_idea b_correct

end choose_best_set_l682_682416


namespace sum_ge_ineq_l682_682115

theorem sum_ge_ineq (x : ℕ → ℝ) (p m n : ℕ)
  (hx_pos : ∀ i, 1 ≤ i ∧ i ≤ p → 0 < x i)
  (x_p1_eq_x1 : x (p+1) = x 1)
  (hp_pos : 1 ≤ p)
  (hm_pos : 1 ≤ m)
  (hn_pos : 1 ≤ n)
  (hnm : n > m) :
  (∑ i in Finset.range p, (x (i+1) + 1)^n / (x (i+1))^m) ≥ p * (n^n) / (m^m * (n-m)^(n-m)) :=
by
  sorry

end sum_ge_ineq_l682_682115


namespace remainder_777_777_mod_13_l682_682372

theorem remainder_777_777_mod_13 : (777 ^ 777) % 13 = 12 := 
by 
  -- Proof steps would go here
  sorry

end remainder_777_777_mod_13_l682_682372


namespace collinear_AFE_l682_682499

open_locale classical
noncomputable theory

variables (α : Type*) [euclidean_geometry α]
variables {O P A B C D E F : α}

-- Definition of the points and their relationships
variables (h1 : ¬ O ∈ line_through P O)
variables (h2 : tangent PA (⊙ O) A)
variables (h3 : tangent PB (⊙ O) B)
variables (h4 : secant PCD (⊙ O) C D)
variables (h5 : betw P C D)
variables (h6 : parallel (line_through B E) (line_through C D))
variables (h7 : midpoint F C D)

-- The theorem to be proven
theorem collinear_AFE : collinear {A, F, E} :=
sorry

end collinear_AFE_l682_682499


namespace smallest_positive_multiple_of_45_is_45_l682_682944

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682944


namespace smallest_positive_multiple_of_45_is_45_l682_682917

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682917


namespace interval_of_decrease_l682_682327

noncomputable def g (x : ℝ) : ℝ := (1/2) ^ x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

def f_of_composite (x : ℝ) : ℝ := f (2 * x - x ^ 2)

theorem interval_of_decrease :
  ∀ x : ℝ, (x ∈ Set.Ioo 0 1) ↔ StrictMonoDecrOn f_of_composite (Set.Icc 0 2) :=
by
  sorry

end interval_of_decrease_l682_682327


namespace smallest_positive_multiple_of_45_l682_682765

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682765


namespace Tenisha_remains_with_50_puppies_l682_682681

theorem Tenisha_remains_with_50_puppies
  (total_dogs : ℕ)
  (percentage_female : ℕ)
  (frac_females_giving_birth : ℚ)
  (puppies_per_female_that_give_birth : ℕ)
  (puppies_donated : ℕ) :
  total_dogs = 40 →
  percentage_female = 60 →
  frac_females_giving_birth = 3/4 →
  puppies_per_female_that_give_birth = 10 →
  puppies_donated = 130 →
  (let number_of_females := (percentage_female * total_dogs) / 100 in
   let females_giving_birth := (frac_females_giving_birth * number_of_females) in
   let total_puppies := (females_giving_birth * puppies_per_female_that_give_birth).toNat in
   total_puppies - puppies_donated) = 50 := by
  sorry

end Tenisha_remains_with_50_puppies_l682_682681


namespace martha_bottles_l682_682417

def total_bottles_left (a b c d : ℕ) : ℕ :=
  a + b + c - d

theorem martha_bottles : total_bottles_left 4 4 5 3 = 10 :=
by
  sorry

end martha_bottles_l682_682417


namespace joan_seashells_left_l682_682229

/--
Joan found 245 seashells on the beach.
She gave Mike 3/5 of the seashells.
She gave Lisa 2/5 of the remaining seashells after giving some to Mike.
Prove that Joan had 59 seashells left at the end.
-/
def seashells_left : ℕ :=
  let total := 245
  let to_mike := total * 3 / 5
  let remaining_after_mike := total - to_mike
  let to_lisa := remaining_after_mike * 2 / 5
  let to_lisa_rounded := to_lisa -- Handling rounding step separately in the proof as simply assigning 39 is incorrect.
  remaining_after_mike - 39 -- to_lisa is FLOOR[98*(2/5)] = 39
  
theorem joan_seashells_left (total : ℕ) (to_mike : ℕ) (remaining_after_mike : ℕ) (to_lisa : ℕ) (to_lisa_rounded : ℕ) :
  total = 245 →
  to_mike = total * 3 / 5 →
  remaining_after_mike = total - to_mike →
  to_lisa = remaining_after_mike * 2 / 5 →
  to_lisa_rounded = 39 →
  (remaining_after_mike - to_lisa_rounded) = 59 :=
by {
    intros h_total h_to_mike h_remaining_after_mike h_to_lisa h_to_lisa_rounded,
    sorry
 }
 
end joan_seashells_left_l682_682229


namespace find_x_values_l682_682057

theorem find_x_values (x : ℝ) (h : x + 60 / (x - 3) = -12) : x = -3 ∨ x = -6 :=
sorry

end find_x_values_l682_682057


namespace number_of_divisors_of_36_l682_682564

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end number_of_divisors_of_36_l682_682564


namespace clock_display_four_different_digits_l682_682317

theorem clock_display_four_different_digits :
  (∑ t in finset.range (24*60), if (((t / 60).div1000 ≠ (t / 60).mod1000) ∧ 
    ((t / 60).div1000 ≠ (t % 60).div1000) ∧ ((t / 60).div1000 ≠ (t % 60).mod1000) ∧ 
    ((t / 60).mod1000 ≠ (t % 60).div1000) ∧ ((t / 60).mod1000 ≠ (t % 60).mod1000) ∧ 
    ((t % 60).div1000 ≠ (t % 60).mod1000)) then 1 else 0) = 588 :=
by
  sorry

end clock_display_four_different_digits_l682_682317


namespace smallest_positive_multiple_of_45_l682_682771

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682771


namespace lisa_eggs_per_year_l682_682638

theorem lisa_eggs_per_year :
  let days_per_week := 5 in
  let weeks_per_year := 52 in
  let children := 4 in
  let eggs_per_child := 2 in
  let eggs_for_husband := 3 in
  let eggs_for_herself := 2 in
  let breakfasts_per_year := days_per_week * weeks_per_year in
  let eggs_per_breakfast := (children * eggs_per_child) + eggs_for_husband + eggs_for_herself in
  let total_eggs := eggs_per_breakfast * breakfasts_per_year in
  total_eggs = 3380
by
  sorry

end lisa_eggs_per_year_l682_682638


namespace smallest_b_gt_4_perfect_square_l682_682737

theorem smallest_b_gt_4_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ k : ℕ, 4 * b + 5 = k^2 ∧ b = 5 :=
by
  sorry

end smallest_b_gt_4_perfect_square_l682_682737


namespace range_of_x_l682_682492

def operation (a b : ℝ) : ℝ :=
  if a > b then a else b

theorem range_of_x (x : ℝ) : (operation (2*x + 1) (x + 3) = x + 3) → (x < 2) :=
by
  sorry

end range_of_x_l682_682492


namespace find_b_l682_682743

-- Define what it means for b to be a solution
def is_solution (b : ℤ) : Prop :=
  b > 4 ∧ ∃ k : ℤ, 4 * b + 5 = k * k

-- State the problem
theorem find_b : ∃ b : ℤ, is_solution b ∧ ∀ b' : ℤ, is_solution b' → b' ≥ 5 := by
  sorry

end find_b_l682_682743


namespace sum_of_coeffs_binomial_eq_32_l682_682355

noncomputable def sum_of_coeffs_binomial (x : ℝ) : ℝ :=
  (3 * x - 1 / Real.sqrt x)^5

theorem sum_of_coeffs_binomial_eq_32 :
  sum_of_coeffs_binomial 1 = 32 :=
by
  sorry

end sum_of_coeffs_binomial_eq_32_l682_682355


namespace smallest_positive_multiple_of_45_l682_682938

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682938


namespace intersection_distance_l682_682218

noncomputable def parametric_eq_C1 (α : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos α, 2 * Real.sin α)

noncomputable def polar_eq_C2 (θ : ℝ) : ℝ :=
  4 * Real.cos θ / (1 - Real.cos (2 * θ))

theorem intersection_distance (θ : ℝ) (h1 : -Real.pi / 2 ≤ θ) (h2 : θ ≤ Real.pi / 2) (h3 : θ ≠ 0) :
  let ρ := 4 * Real.cos θ
  in ρ = polar_eq_C2 θ → |ρ| = 2 * Real.sqrt 2 :=
by sorry

end intersection_distance_l682_682218


namespace union_of_sets_l682_682510

universe u

variables {α : Type u} [DecidableEq α]

def set_A : Set α := {1, 2}
def set_B (a : α) : Set α := {3, a}

lemma intersection_condition (a : α) : set_A ∩ set_B a = {1} ↔ a = 1 :=
begin
  split,
  { intro h,
    suffices : a = 1,
    { exact this },
    -- Proof omitted for brevity
    sorry },
  { intro ha,
    rw ha,
    -- Further proof omitted for brevity
    sorry }
end

theorem union_of_sets (a : α) (h : set_A ∩ set_B a = {1}) : set_A ∪ set_B a = {1, 2, 3} :=
begin
  have ha : a = 1,
  { rw intersection_condition at h,
    exact h.mp h },
  rw ha,
  -- Further proof omitted for brevity
  sorry

end union_of_sets_l682_682510


namespace four_points_not_all_acute_l682_682391

theorem four_points_not_all_acute (A B C D : Point) :
  ∃ (T : Triangle), T ∈ triangles_formed_by {A, B, C, D} ∧ ¬ acute_triangle T :=
sorry

end four_points_not_all_acute_l682_682391


namespace coeff_x_neg2_in_binom_expansion_l682_682530

noncomputable def f : ℝ → ℝ := λ x, x + 9 / x
def interval := (1 : ℝ, 4 : ℝ)
noncomputable def n := Real.Inf {y | ∃ x ∈ set.Icc 1 4, f x = y}

theorem coeff_x_neg2_in_binom_expansion :
  n = 6 → 
  (∃ c : ℝ, (x - 1/x)^n = c * x^-2) → 
  c = 15 :=
by
  sorry

end coeff_x_neg2_in_binom_expansion_l682_682530


namespace smallest_positive_multiple_of_45_l682_682927

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682927


namespace largest_n_value_l682_682065

-- Define the conditions
def quadratic_expr (n : ℤ) : polynomial ℤ :=
  3 * polynomial.X ^ 2 + polynomial.C n * polynomial.X + 54

def factorizable_as_linear_factors (p : polynomial ℤ) : Prop :=
  ∃ A B : ℤ, p = (3 * polynomial.X + polynomial.C A) * (polynomial.X + polynomial.C B)

-- Statement of the theorem
theorem largest_n_value : ∃ n : ℤ, 
  (quadratic_expr n).factorizable_as_linear_factors ∧ 
  (∀ m : ℤ, (quadratic_expr m).factorizable_as_linear_factors → m ≤ 163) :=
sorry

end largest_n_value_l682_682065


namespace exists_real_x_for_k_l682_682491

theorem exists_real_x_for_k (k : ℕ) (h : k > 1) :
  ∃ (x : ℝ), ∀ (n : ℕ), (0 < n) ∧ (n < 1398) →
  (fract (x^n) < fract (x^(n - 1)) ↔ k ∣ n) :=
by
  sorry

end exists_real_x_for_k_l682_682491


namespace quilt_cut_identical_pieces_l682_682980

theorem quilt_cut_identical_pieces 
  (quilt : Type)
  [piece : Type]
  (identical_shape_and_size : piece → Prop)
  (sewn_pieces : piece → piece → quilt)
  (p1 p2 : piece)
  (h_identical : identical_shape_and_size p1 ∧ identical_shape_and_size p2)
  (h_quilt : quilt = sewn_pieces p1 p2) :
  ∃ (cut_line : Type),
  (∀ (q1 q2 : quilt), (cut_line → q1 × q2) → (q1 = p1 ∧ q2 = p2)) :=
sorry

end quilt_cut_identical_pieces_l682_682980


namespace smallest_positive_multiple_of_45_l682_682827

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682827


namespace five_digit_odd_numbers_l682_682077

theorem five_digit_odd_numbers : ∃ (n : ℕ), 
  (∃ (s : Finset (Fin 10)), 
    (∀ x ∈ s, x.val ∈ {1, 2, 3, 4, 5}) ∧
    s.card = 5 ∧ 
    (∃ (units_place : ℕ), units_place ∈ {1, 3, 5}) ∧
    ∀ x, x ∈ s → ∃ d, 0 ≤ d ∧ d < 5 ∧ 
    ( ∀ i j, i ≠ j → (s.to_list.nth_le i sorry) ≠ (s.to_list.nth_le j sorry) )
  )
  ∧ n = 72 := 
sorry

end five_digit_odd_numbers_l682_682077


namespace initial_percentage_of_salt_l682_682429

theorem initial_percentage_of_salt (P : ℝ) :
  (P / 100) * 80 = 8 → P = 10 :=
by
  intro h
  sorry

end initial_percentage_of_salt_l682_682429


namespace A_n_is_integer_l682_682254

open Real

noncomputable def A_n (a b : ℕ) (θ : ℝ) (n : ℕ) : ℝ :=
  (a^2 + b^2)^n * sin (n * θ)

theorem A_n_is_integer (a b : ℕ) (h : a > b) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < pi/2) (h_sin : sin θ = 2 * a * b / (a^2 + b^2)) :
  ∀ n : ℕ, ∃ k : ℤ, A_n a b θ n = k :=
by
  sorry

end A_n_is_integer_l682_682254


namespace count_multiples_5_or_7_not_35_l682_682183

def is_multiple (a b : ℕ) : Prop := ∃ k, a = k * b

def count_multiples (n m : ℕ) : ℕ :=
  (n / m)

theorem count_multiples_5_or_7_not_35 :
  let N := 3015 in
  let multiples_of_five := count_multiples N 5 in
  let multiples_of_seven := count_multiples N 7 in
  let multiples_of_thirty_five := count_multiples N 35 in
  multiples_of_five + multiples_of_seven - multiples_of_thirty_five = 948 :=
by
  let N := 3015
  let multiples_of_five := count_multiples N 5
  let multiples_of_seven := count_multiples N 7
  let multiples_of_thirty_five := count_multiples N 35
  have multiples_five_correct : multiples_of_five = 603 := by sorry
  have multiples_seven_correct : multiples_of_seven = 431 := by sorry
  have multiples_thirty_five_correct : multiples_of_thirty_five = 86 := by sorry
  rw [multiples_five_correct, multiples_seven_correct, multiples_thirty_five_correct]
  exact rfl

end count_multiples_5_or_7_not_35_l682_682183


namespace find_possible_m_values_l682_682610

-- Definitions based on the problem context
def is_right_triangle (vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) := 
  let ((x1, y1), (x2, y2), (x3, y3)) := vertices in
  (x2 = x1 ∨ y2 = y1 ∨ x3 = x1 ∨ y3 = y1 ∨ x3 = x2 ∨ y3 = y2)

def median_to_leg_midpoint (vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (line_eq : ℝ → ℝ) :=
  let ((x1, y1), (x2, y2), (x3, y3)) := vertices in
  let mid_x_leg := (x1 + x2) / 2 in
  let mid_y_leg := (y1 + y2) / 2 in
  (line_eq mid_x_leg = mid_y_leg) ∨ (line_eq mid_x_leg = y3) ∨ (line_eq x3 = mid_y_leg)

theorem find_possible_m_values (m : ℝ) :
  (∃ (vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)),
    is_right_triangle vertices ∧
    (median_to_leg_midpoint vertices (fun x => 4 * x + 1) ∧
    median_to_leg_midpoint vertices (fun x => m * x + 3))) ↔
    (m = 16 ∨ m = 1) :=
sorry

end find_possible_m_values_l682_682610


namespace smallest_positive_multiple_of_45_l682_682835

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682835


namespace finite_set_of_primes_l682_682237

def nonempty_positive_integers (S : Set ℕ) : Prop :=
  S.nonempty ∧ (∀ a b ∈ S, a * b + 1 ∈ S)

def finite_nondividing_primes (S : Set ℕ) : Prop :=
  ∃ M : ℕ, ∀ p : ℕ, Prime p → (¬ ∃ x ∈ S, p ∣ x) → p ≤ M

theorem finite_set_of_primes (S : Set ℕ)
  (h : nonempty_positive_integers S) : finite_nondividing_primes S :=
sorry

end finite_set_of_primes_l682_682237


namespace g_even_l682_682222

def g (x : ℝ) : ℝ := 3 / (2 * x^8 - 5)

theorem g_even : ∀ x : ℝ, g (-x) = g x :=
by
  intro x
  unfold g
  sorry

end g_even_l682_682222


namespace number_of_dogs_l682_682347

-- Conditions
def ratio_cats_dogs : ℚ := 3 / 4
def number_cats : ℕ := 18

-- Define the theorem to prove
theorem number_of_dogs : ∃ (dogs : ℕ), dogs = 24 :=
by
  -- Proof steps will go here, but we can use sorry for now to skip actual proving.
  sorry

end number_of_dogs_l682_682347


namespace circle_area_of_parabola_focus_tangent_to_directrix_l682_682604

theorem circle_area_of_parabola_focus_tangent_to_directrix :
  let a := 3 / 2 -- Parameter a of the parabola y^2 = 4ax
      focus := (a, 0) -- Focus of the parabola
      directrix := -a -- Directrix of the parabola
      radius := |a + directrix| -- Distance from focus to directrix (radius of the circle)
      area := Real.pi * radius^2 -- Area of the circle given its radius
  in area = 9 * Real.pi :=
by
  -- Definitions (conditions) from the problem
  let a : ℝ := 3 / 2
  let focus := (a, 0 : ℝ)
  let directrix := -a
  let radius := |a + directrix|
  let area := Real.pi * radius^2
  -- Assert the proof (answer)
  show area = 9 * Real.pi
  sorry

end circle_area_of_parabola_focus_tangent_to_directrix_l682_682604


namespace smallest_positive_multiple_45_l682_682817

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682817


namespace probability_N_lt_L_is_zero_l682_682082

variable (M N L O : ℝ)
variable (hNM : N < M)
variable (hLO : L > O)

theorem probability_N_lt_L_is_zero : 
  (∃ (permutations : List (ℝ → ℝ)), 
  (∀ perm : ℝ → ℝ, perm ∈ permutations → N < M ∧ L > O) ∧ 
  ∀ perm : ℝ → ℝ, N > L) → false :=
by {
  sorry
}

end probability_N_lt_L_is_zero_l682_682082


namespace land_leveling_inequality_l682_682197

-- Define the given conditions
def total_land := 500 -- Total land to be leveled, in m^2
def max_time := 3 -- Maximum time allowed, in hours
def first_half_hour_land := 60 -- Land leveled in the first half-hour, in m^2
def remaining_time := max_time - 0.5 -- Remaining time after the first half-hour
def land_per_hour (x : ℝ) := x -- Land leveled per hour in the remaining time, in m^2/hour

-- Define the statement to be proved
theorem land_leveling_inequality (x : ℝ) : 
  60 + (3 - 0.5) * x ≥ 500 :=
sorry

end land_leveling_inequality_l682_682197


namespace expression_identity_l682_682001

theorem expression_identity (k : ℤ) : 2 ^ (-3 * k) - 2 ^ (-(3 * k - 2)) + 2 ^ (-(3 * k + 2)) = -(11 / 4) * 2 ^ (-3 * k) :=
sorry

end expression_identity_l682_682001


namespace system_of_equations_solution_l682_682056

theorem system_of_equations_solution (b : ℝ) :
  (∀ (a : ℝ), ∃ (x y : ℝ), (x - 1)^2 + y^2 = 1 ∧ a * x + y = a * b) ↔ 0 ≤ b ∧ b ≤ 2 :=
by
  sorry

end system_of_equations_solution_l682_682056


namespace probability_N_taller_than_L_l682_682097

variable (M N L O : ℕ)

theorem probability_N_taller_than_L 
  (h1 : N < M) 
  (h2 : L > O) : 
  0 = 0 := 
sorry

end probability_N_taller_than_L_l682_682097


namespace number_of_girls_in_class_l682_682361

-- Define Ms. Smith's class
variable (g : ℕ) -- number of girls
variable (b : ℕ := 10) -- number of boys
variable (total_books : ℕ := 375) -- total number of books
variable (girls_books : ℕ := 225) -- total number of books the girls got

-- Define equal distribution of books
variable (books_per_student : ℕ := total_books / (g + b))

-- Define the main question: how many girls are in the class?
theorem number_of_girls_in_class : g = 15 := by
-- Total number of books distributed among boys
let boys_books := total_books - girls_books
-- Number of books each boy received
have hb : books_per_student = boys_books / b, from sorry
-- Number of books each girl received
have hg : books_per_student = girls_books / g, from sorry
-- Equate books per student received by boys and girls
have eq1 : books_per_student = 15 := by 
  sorry
-- From the equation, solve for number of girls
have final_eq : g = girls_books / 15 := by 
  sorry
-- Result of the number of girls
show g = 15, from final_eq
    
end number_of_girls_in_class_l682_682361


namespace smallest_positive_multiple_of_45_l682_682940

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682940


namespace sector_area_l682_682624

theorem sector_area (r : ℝ) (θ : ℝ) (arc_area : ℝ) : 
  r = 24 ∧ θ = 110 ∧ arc_area = 176 * Real.pi → 
  arc_area = (θ / 360) * (Real.pi * r ^ 2) :=
by
  intros
  sorry

end sector_area_l682_682624


namespace sin_line_intersection_ratios_l682_682690

theorem sin_line_intersection_ratios :
  ∃ p q : ℕ, nat.coprime p q ∧ p < q ∧ (∀ n : ℤ, let x1 := (30 + 360 * n : ℝ) in
                                               let x2 := (150 + 360 * n : ℝ) in
                                               (x2 - x1) = 120 ∧
                                               ((x1 + 360) - x2) = 240) ∧
                                               (p, q) = (1, 2) :=
begin
  sorry
end

end sin_line_intersection_ratios_l682_682690


namespace smallest_positive_multiple_of_45_l682_682889

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682889


namespace find_m_value_l682_682170

theorem find_m_value (m : ℝ) : 
  let a := (m, m - 1); 
      b := (1, 2) 
  in a.1 * b.1 + a.2 * b.2 = 0 → m = 2 / 3 := 
by 
  intro h_perp
  set a := (m, m - 1) with ha
  set b := (1, 2) with hb
  have h_dot : a.1 * b.1 + a.2 * b.2 = m * 1 + (m - 1) * 2, by sorry
  rw [ha, hb] at h_perp
  rw h_dot at h_perp
  have : m + 2 * m - 2 = 0 := h_perp
  have : 3 * m = 2 := by linarith
  exact (eq_div_iff (by norm_num)).mp this

end find_m_value_l682_682170


namespace james_total_money_l682_682225

noncomputable def total_money (wallet : list ℝ) (pocket : list (ℝ ⊕ ℝ)) (coins : list ℝ) (exchange_rate : ℝ) : ℝ :=
  let us_pocket := pocket.foldr (λ x acc, acc +
    match x with
    | sum.inl usd => usd
    | sum.inr eur => eur * exchange_rate
    end) 0
  wallet.sum + us_pocket + coins.sum

theorem james_total_money :
  total_money [50, 20, 5] [sum.inl 20, sum.inl 10, sum.inr 5] [0.25 * 2, 0.10 * 3, 0.01 * 5] 1.20 = 111.85 :=
by sorry

end james_total_money_l682_682225


namespace pets_remaining_is_correct_l682_682437

-- Definitions for the initial conditions and actions taken
def initial_puppies : Nat := 7
def initial_kittens : Nat := 6
def puppies_sold : Nat := 2
def kittens_sold : Nat := 3

-- Definition that calculates the remaining number of pets
def remaining_pets : Nat := initial_puppies + initial_kittens - (puppies_sold + kittens_sold)

-- The theorem to prove
theorem pets_remaining_is_correct : remaining_pets = 8 := by sorry

end pets_remaining_is_correct_l682_682437


namespace four_diff_digits_per_day_l682_682322

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end four_diff_digits_per_day_l682_682322


namespace highest_power_of_3_l682_682291

-- Define the integer N formed by writing 2-digit integers from 18 to 93 consecutively
def N : ℕ := -- actually defining this would be tedious, so we use sorry for this part
 sorry

-- Define the function to sum the digits of a number
def sum_digits (n : ℕ) : ℕ := n.digits 10.sum

-- The problem can be stated as a theorem
theorem highest_power_of_3 : ∃ k : ℕ, ∀ m : ℕ, 3 ^ m ∣ N ↔ m ≤ 1 :=
by sorry

end highest_power_of_3_l682_682291


namespace smallest_positive_multiple_of_45_l682_682836

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682836


namespace bottles_left_l682_682419

-- Define initial conditions
def bottlesInRefrigerator : Nat := 4
def bottlesInPantry : Nat := 4
def bottlesBought : Nat := 5
def bottlesDrank : Nat := 3

-- Goal: Prove the total number of bottles left
theorem bottles_left : bottlesInRefrigerator + bottlesInPantry + bottlesBought - bottlesDrank = 10 :=
by
  sorry

end bottles_left_l682_682419


namespace smallest_positive_multiple_of_45_l682_682766

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682766


namespace shaded_fraction_eq_one_fourth_l682_682203

theorem shaded_fraction_eq_one_fourth (s : ℝ) :
  let triangle_area := (1 / 8) * s^2 in
  let total_shaded_area := 2 * triangle_area in
  total_shaded_area / (s^2) = 1 / 4 :=
by 
  sorry

end shaded_fraction_eq_one_fourth_l682_682203


namespace time_to_write_each_song_l682_682046

theorem time_to_write_each_song
  (total_studio_time : ℕ)
  (recording_time_per_song : ℕ)
  (total_editing_time : ℕ)
  (num_songs : ℕ) :
  total_studio_time = 300 →
  recording_time_per_song = 12 →
  total_editing_time = 30 →
  num_songs = 10 →
  ((total_studio_time - (recording_time_per_song * num_songs) - total_editing_time) / num_songs) = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end time_to_write_each_song_l682_682046


namespace smallest_positive_multiple_of_45_l682_682935

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682935


namespace probability_N_lt_L_is_zero_l682_682079

variable (M N L O : ℝ)
variable (hNM : N < M)
variable (hLO : L > O)

theorem probability_N_lt_L_is_zero : 
  (∃ (permutations : List (ℝ → ℝ)), 
  (∀ perm : ℝ → ℝ, perm ∈ permutations → N < M ∧ L > O) ∧ 
  ∀ perm : ℝ → ℝ, N > L) → false :=
by {
  sorry
}

end probability_N_lt_L_is_zero_l682_682079


namespace smallest_positive_multiple_of_45_l682_682752

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682752


namespace pet_store_dogs_l682_682345

theorem pet_store_dogs (cats dogs : ℕ) (h1 : 18 = cats) (h2 : 3 * dogs = 4 * cats) : dogs = 24 :=
by
  sorry

end pet_store_dogs_l682_682345


namespace power_function_quadrants_l682_682171

theorem power_function_quadrants (f : ℝ → ℝ) (h : f (1/3) = 9) : 
  (∀ x : ℝ, f x = x ^ (-2)) ∧ (∀ x : ℝ, x > 0 → f x > 0 ∧ f (-x) > 0) :=
sorry

end power_function_quadrants_l682_682171


namespace number_of_divisors_of_36_l682_682553

theorem number_of_divisors_of_36 :  
  let n := 36
  number_of_divisors n = 9 := 
by 
  sorry

end number_of_divisors_of_36_l682_682553


namespace total_eyes_in_extended_family_l682_682226

def mom_eyes := 1
def dad_eyes := 3
def kids_eyes := 3 * 4
def moms_previous_child_eyes := 5
def dads_previous_children_eyes := 6 + 2
def dads_ex_wife_eyes := 1
def dads_ex_wifes_new_partner_eyes := 7
def child_of_ex_wife_and_partner_eyes := 8

theorem total_eyes_in_extended_family :
  mom_eyes + dad_eyes + kids_eyes + moms_previous_child_eyes + dads_previous_children_eyes +
  dads_ex_wife_eyes + dads_ex_wifes_new_partner_eyes + child_of_ex_wife_and_partner_eyes = 45 :=
by
  -- add proof here
  sorry

end total_eyes_in_extended_family_l682_682226


namespace smallest_positive_multiple_of_45_l682_682844

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682844


namespace find_radius_of_first_circle_l682_682600

-- Define the geometric entities
variables (r1 r2 : ℝ)
variables (A B C D E : Point ℝ)
variables (circle1 : Circle A r1)
variables (circle2 : Circle D r2)

-- Define the conditions
variables (AP BQ : Line)
variables (AP_tangent : IsTangent AP circle1)
variables (BQ_tangent : IsTangent BQ circle1)
variables (PCQ : Line)
variables (PCQ_tangent_A : IsTangent PCQ A)
variables (PCQ_tangent_B : IsTangent PCQ B)
variables (lineABC : LiesOnSameLine [A, B, C])
variables (external_tangent_condition : IsExtTangentAt circle1 circle2 D)
variables (AP_external_tangent : IsExtTangentAt AP E circle2)
variables (AP_length : AP.length = 5)
variables (BQ_length : BQ.length = 12)
variables (angle_DEC : angle E D C = 30°)

-- The objective condition to prove
theorem find_radius_of_first_circle :
  r1 = 8.5 :=
sorry

end find_radius_of_first_circle_l682_682600


namespace triangle_area_possible_values_l682_682529

noncomputable def line (m : ℝ) : ℝ → ℝ → Prop := 
  λ x y, (√3 * m + 1) * x - (m - √3) * y - 4 = 0

def circle (x y : ℝ) : Prop := 
  x^2 + y^2 = 16

def is_intersect (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ x1 y1 x2 y2,
    A = (x1, y1) ∧ B = (x2, y2) ∧ 
    line m x1 y1 ∧ line m x2 y2 ∧ 
    circle x1 y1 ∧ circle x2 y2

def area_triangle_OAB (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  (1 / 2) * abs (x1 * y2 - x2 * y1)

theorem triangle_area_possible_values (A B : ℝ × ℝ) (m : ℝ) 
  (h_intersect : is_intersect A B m) :
  area_triangle_OAB A B = 4 ∨ 
  area_triangle_OAB A B = 2 * √3 ∨ 
  area_triangle_OAB A B = 4 * √3 :=
sorry

end triangle_area_possible_values_l682_682529


namespace quadratic_equation_single_solution_l682_682054

theorem quadratic_equation_single_solution (q : ℚ) (h : q ≠ 0) :
  (∃ q : ℚ, q = 100 / 9 ∧ discriminant q (-20) 9 = 0) :=
by
  sorry

noncomputable def discriminant (a b c : ℚ) : ℚ :=
  b * b - 4 * a * c

end quadratic_equation_single_solution_l682_682054


namespace smallest_positive_multiple_of_45_is_45_l682_682794

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682794


namespace find_k_l682_682514

open Real

variables (a b : ℝ × ℝ) (k : ℝ)

def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

theorem find_k (ha : a = (1, 2)) (hb : b = (-2, 4)) (perpendicular : dot_product (k • a + b) b = 0) :
  k = - (10 / 3) :=
by
  sorry

end find_k_l682_682514


namespace clock_shows_four_different_digits_for_588_minutes_l682_682300

-- Definition of the problem
def isFourDifferentDigits (h1 h2 m1 m2 : Nat) : Bool :=
  (h1 ≠ h2) && (h1 ≠ m1) && (h1 ≠ m2) && (h2 ≠ m1) && (h2 ≠ m2) && (m1 ≠ m2)

noncomputable def countFourDifferentDigitsMinutes : Nat :=
  let validMinutes := List.filter (λ (t : Nat × Nat),
    let (h, m) := t
    let h1 := h / 10
    let h2 := h % 10
    let m1 := m / 10
    let m2 := m % 10
    isFourDifferentDigits h1 h2 m1 m2
  ) (List.product (List.range 24) (List.range 60))
  validMinutes.length

theorem clock_shows_four_different_digits_for_588_minutes :
  countFourDifferentDigitsMinutes = 588 := sorry

end clock_shows_four_different_digits_for_588_minutes_l682_682300


namespace smallest_positive_multiple_45_l682_682818

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682818


namespace smallest_positive_multiple_45_l682_682904

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682904


namespace election_votes_proof_l682_682207

def percentage_of (total: ℕ) (percent: ℕ): ℕ := total * percent / 100

theorem election_votes_proof:
  ∀ (T: ℕ) (P_inv: ℕ) (P_A: ℕ) (P_B: ℕ) (P_C: ℕ),
    T = 850000 →
    P_inv = 20 →
    P_A = 45 →
    P_B = 35 →
    P_C = 100 - (P_A + P_B) →
    let valid_votes := percentage_of T (100 - P_inv) in
    let V_A := percentage_of valid_votes P_A in
    let V_B := percentage_of valid_votes P_B in
    let V_C := percentage_of valid_votes P_C in
    V_A = 306000 ∧ V_B = 238000 ∧ V_C = 136000 := by
    intros T P_inv P_A P_B P_C hT hP_inv hP_A hP_B hP_C
    let valid_votes := percentage_of T (100 - P_inv)
    let V_A := percentage_of valid_votes P_A
    let V_B := percentage_of valid_votes P_B
    let V_C := percentage_of valid_votes P_C
    split
    sorry
    split
    sorry
    sorry

end election_votes_proof_l682_682207


namespace cindy_correct_answer_l682_682444

theorem cindy_correct_answer (x : ℝ) (h₀ : (x - 12) / 4 = 32) : (x - 7) / 5 = 27 :=
by
  sorry

end cindy_correct_answer_l682_682444


namespace minimum_sum_of_labels_l682_682456

def chessboard_label (i j : ℕ) : ℚ := 1 / (i + j - 1)

def valid_selection (selection : List (ℕ × ℕ)) : Prop :=
  (selection.length = 10) ∧
  (selection.map Prod.fst).Nodup ∧
  (selection.map Prod.snd).Nodup

noncomputable def selection_sum (selection : List (ℕ × ℕ)) : ℚ :=
  selection.map (λ ⟨i, j⟩ => chessboard_label i j).sum

theorem minimum_sum_of_labels : ∃ (selection : List (ℕ × ℕ)), valid_selection selection ∧ selection_sum selection = 20 / 11 :=
sorry

end minimum_sum_of_labels_l682_682456


namespace sum_last_two_digits_l682_682973

theorem sum_last_two_digits (a b m n : ℕ) (h7 : a = 7) (h13 : b = 13) (h100 : m = 100) (h30 : n = 30) : 
 ((a ^ n) + (b ^ n)) % m = 98 :=
by
  have h₁ : 7 ^ 30 % 100 = (49 : ℕ) := by sorry
  have h₂ : 13 ^ 30 % 100 = 49 := by sorry
  calc
    (7 ^ 30 + 13 ^ 30) % 100
      = (49 + 49) % 100 : by { rw [h₁, h₂] }
  ... = 98 % 100 : by rfl
  ... = 98 : by rfl

end sum_last_two_digits_l682_682973


namespace bus_stop_time_l682_682985

theorem bus_stop_time 
  (bus_speed_without_stoppages : ℤ)
  (bus_speed_with_stoppages : ℤ)
  (h1 : bus_speed_without_stoppages = 54)
  (h2 : bus_speed_with_stoppages = 36) :
  ∃ t : ℕ, t = 20 :=
by
  sorry

end bus_stop_time_l682_682985


namespace smallest_positive_multiple_45_l682_682901

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682901


namespace fourth_piece_length_l682_682390

theorem fourth_piece_length (a b c d : ℕ) (distinct_pieces : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (cut_pieces : list ℕ) (has_length_3 : cut_pieces.length = 3) (pieces_values : 8 ∈ cut_pieces ∧ 9 ∈ cut_pieces ∧ 10 ∈ cut_pieces) :
  ∃ x : ℕ, x ≠ 8 ∧ x ≠ 9 ∧ x ≠ 10 ∧ (x = 7 ∨ x = 11) :=
by
  sorry

end fourth_piece_length_l682_682390


namespace points_A_B_D_collinear_l682_682513

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (a b : V) (A B D : V)
variables (h_noncollinear : ¬ collinear ({a, b} : set V)) (h_nonzero : a ≠ 0 ∧ b ≠ 0)
variables (AB BC CD : V)
variables (h_AB : AB = a + 5 • b) (h_BC : BC = -2 • a + 8 • b) (h_CD : CD = 3 • a - 3 • b)

theorem points_A_B_D_collinear : collinear ({A, B, D} : set V) :=
by {
  -- proof here
  sorry
}

end points_A_B_D_collinear_l682_682513


namespace smallest_positive_multiple_of_45_l682_682937

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682937


namespace vacuum_cleaner_cost_l682_682033

-- Define initial amount collected
def initial_amount : ℕ := 20

-- Define amount added each week
def weekly_addition : ℕ := 10

-- Define number of weeks
def number_of_weeks : ℕ := 10

-- Define the total amount after 10 weeks
def total_amount : ℕ := initial_amount + (weekly_addition * number_of_weeks)

-- Prove that the total amount is equal to the cost of the vacuum cleaner
theorem vacuum_cleaner_cost : total_amount = 120 := by
  sorry

end vacuum_cleaner_cost_l682_682033


namespace probability_nina_taller_than_lena_is_zero_l682_682093

-- Definition of participants and conditions
variable (M N L O : ℝ)

-- Conditions
def condition1 := N < M
def condition2 := L > O

-- Statement: Given conditions, the probability that N > L is 0
theorem probability_nina_taller_than_lena_is_zero
  (h1 : condition1)
  (h2 : condition2) :
  (P : ℝ) = 0 :=
by
  sorry

end probability_nina_taller_than_lena_is_zero_l682_682093


namespace find_f_neg2_l682_682498

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ a b : ℝ, f (a + b) = f a * f b
axiom cond2 : ∀ x : ℝ, f x > 0
axiom cond3 : f 1 = 1 / 3

theorem find_f_neg2 : f (-2) = 9 := sorry

end find_f_neg2_l682_682498


namespace smallest_positive_multiple_of_45_l682_682930

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682930


namespace wall_length_eq_800_l682_682398

theorem wall_length_eq_800 
  (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_width : ℝ) (wall_height : ℝ)
  (num_bricks : ℝ) 
  (brick_volume : ℝ) 
  (total_brick_volume : ℝ)
  (wall_volume : ℝ) :
  brick_length = 25 → 
  brick_width = 11.25 → 
  brick_height = 6 → 
  wall_width = 600 → 
  wall_height = 22.5 → 
  num_bricks = 6400 → 
  brick_volume = brick_length * brick_width * brick_height → 
  total_brick_volume = brick_volume * num_bricks → 
  total_brick_volume = wall_volume →
  wall_volume = (800 : ℝ) * wall_width * wall_height :=
by
  sorry

end wall_length_eq_800_l682_682398


namespace smallest_positive_multiple_of_45_is_45_l682_682925

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682925


namespace toll_formula_l682_682357

def axles (wheels_per_axle : ℕ → ℕ) (total_wheels : ℕ) : ℕ :=
  total_wheels - wheels_per_axle 1 + wheels_per_axle 2 + wheels_per_axle 3

theorem toll_formula (t v : ℕ) (h : t = 2.50 + 0.50 * (v - 2)) (t_18_wheel_truck : ℕ) (w : ℕ):
  t_18_wheel_truck = 4 → axles (λ x, if x = 1 then 2 else 4) 18 = v :=
by {
  sorry
}

end toll_formula_l682_682357


namespace smallest_positive_multiple_of_45_l682_682749

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682749


namespace area_of_region_l682_682603

def region (x y : ℝ) : Prop := abs x ≤ 1 ∧ abs y ≤ 1

def inequality (a b x y : ℝ) : Prop := a * x - 2 * b * y ≤ 2

theorem area_of_region :
  (∀ x y, region x y → inequality a b x y) →
  let points := { p : ℝ × ℝ | ∃ (a b : ℝ), ∀ x y, region x y → inequality a b x y } in
  let area := set.card points in
  area = 4 :=
sorry

end area_of_region_l682_682603


namespace divisors_of_36_l682_682567

theorem divisors_of_36 : ∀ n : ℕ, n = 36 → (∃ k : ℕ, k = 9) :=
by
  intro n hn
  have h_prime_factors : (n = 2^2 * 3^2) := by rw hn; norm_num
  -- Using the formula for the number of divisors based on prime factorization
  have h_num_divisors : (2 + 1) * (2 + 1) = 9 := by norm_num
  use 9
  rw h_num_divisors
  sorry

end divisors_of_36_l682_682567


namespace common_solutions_l682_682704

-- Given conditions
def eq1 (x y : ℝ) : Prop := y = (x + 1) ^ 2
def eq2 (x y : ℝ) : Prop := x * y + y = 1

-- Statement to be proved
theorem common_solutions :
  ∃ (x y : ℂ), eq1 x y ∧ eq2 x y ∧
  ((∃ (x : ℝ), ∃ y : ℝ, eq1 x y ∧ eq2 x y ∧ x = 0 ∧ y = 1) ∧  -- One real solution at (0, 1)
   (∃ x : ℝ, ∃ y : ℂ, eq1 x y ∧ eq2 x y ∧ x ≠ 0)) ∧            -- Two complex solutions
   (∀ x y, eq1 x y ∧ eq2 x y → x = 0 ∨ x = -3/2 + (sqrt 3*i)/2 ∨ x = -3/2 - (sqrt 3*i)/2) :=
begin
  sorry -- Proof goes here
end

end common_solutions_l682_682704


namespace solve_divisor_problem_l682_682644

def divisor_problem : Prop :=
  ∃ D : ℕ, 12401 = (D * 76) + 13 ∧ D = 163

theorem solve_divisor_problem : divisor_problem :=
sorry

end solve_divisor_problem_l682_682644


namespace probability_N_taller_than_L_l682_682098

variable (M N L O : ℕ)

theorem probability_N_taller_than_L 
  (h1 : N < M) 
  (h2 : L > O) : 
  0 = 0 := 
sorry

end probability_N_taller_than_L_l682_682098


namespace conjugate_of_complex_number_l682_682104

theorem conjugate_of_complex_number (x y : ℝ) (i : ℂ) (h : i^2 = -1) (h1 : (x : ℂ) / (1 + i) = 1 - y * i) : x + y * i = 2 + i → conj (x + y * i) = 2 - i := 
by 
  sorry

end conjugate_of_complex_number_l682_682104


namespace smallest_positive_multiple_of_45_l682_682750

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682750


namespace distinct_integers_division_l682_682651

theorem distinct_integers_division (n : ℤ) (h : n > 1) :
  ∃ (a b c : ℤ), a = n^2 + n + 1 ∧ b = n^2 + 2 ∧ c = n^2 + 1 ∧
  n^2 < a ∧ a < (n + 1)^2 ∧ 
  n^2 < b ∧ b < (n + 1)^2 ∧ 
  n^2 < c ∧ c < (n + 1)^2 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ c ∣ (a ^ 2 + b ^ 2) := 
by
  sorry

end distinct_integers_division_l682_682651


namespace probability_of_green_ball_l682_682460

-- Define the number of balls in each container
def number_balls_I := (10, 5)  -- (red, green)
def number_balls_II := (3, 6)  -- (red, green)
def number_balls_III := (3, 6)  -- (red, green)

-- Define the probability of selecting each container
noncomputable def probability_container_selected := (1 / 3 : ℝ)

-- Define the probability of drawing a green ball from each container
noncomputable def probability_green_I := (number_balls_I.snd : ℝ) / ((number_balls_I.fst + number_balls_I.snd) : ℝ)
noncomputable def probability_green_II := (number_balls_II.snd : ℝ) / ((number_balls_II.fst + number_balls_II.snd) : ℝ)
noncomputable def probability_green_III := (number_balls_III.snd : ℝ) / ((number_balls_III.fst + number_balls_III.snd) : ℝ)

-- Define the combined probabilities for drawing a green ball and selecting each container
noncomputable def combined_probability_I := probability_container_selected * probability_green_I
noncomputable def combined_probability_II := probability_container_selected * probability_green_II
noncomputable def combined_probability_III := probability_container_selected * probability_green_III

-- Define the total probability of drawing a green ball
noncomputable def total_probability_green := combined_probability_I + combined_probability_II + combined_probability_III

-- The theorem to be proved
theorem probability_of_green_ball : total_probability_green = (5 / 9 : ℝ) :=
by
  sorry

end probability_of_green_ball_l682_682460


namespace number_of_divisors_of_36_l682_682552

theorem number_of_divisors_of_36 :  
  let n := 36
  number_of_divisors n = 9 := 
by 
  sorry

end number_of_divisors_of_36_l682_682552


namespace smallest_positive_multiple_of_45_l682_682862

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682862


namespace find_a_for_function_equality_l682_682540

theorem find_a_for_function_equality (a : ℝ) (h_a_pos : 0 < a)
  (h_condition : ∀ x1 ∈ set.Icc (1 : ℝ) 2, ∃ x2 ∈ set.Icc (1 : ℝ) 2, 
    (x1 * x2) = ((a * x1^2 - x1) * (a * x2^2 - x2))) :
  a = 3 / 2 :=
begin
  sorry
end

end find_a_for_function_equality_l682_682540


namespace smallest_positive_multiple_of_45_l682_682859

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682859


namespace smallest_positive_multiple_of_45_is_45_l682_682915

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682915


namespace simplify_expression_l682_682669

variable {m : ℝ} (hm : m ≠ 0)

theorem simplify_expression : ( (1 / (3 * m)) ^ (-3) * (2 * m) ^ 4 ) = 432 * m ^ 7 := 
by
  sorry

end simplify_expression_l682_682669


namespace generating_functions_l682_682989

variables {X : Type} [DiscreteProb X]
variables (p : ℕ → ℝ) [MeasureTheory.ProbabilityMeasure p]
noncomputable def G (s : ℝ) : ℝ := ∑' n, (p n) * s^n

def q (n : ℕ) : ℝ := ∑' i in finset.range n, p i
def r (n : ℕ) : ℝ := 1 - q n

noncomputable def G_q (s : ℝ) : ℝ := ∑' n, q n * s^n
noncomputable def G_r (s : ℝ) : ℝ := ∑' n, r n * s^n

theorem generating_functions :
  (∀ s, G_q s = (1 - G s) / (1 - s)) ∧
  (∀ s, G_r s = (G s) / (1 - s)) ∧
  (∀ s, s → 1 → (1 - G s) / (1 - s) → ∑' n, q n) := sorry

end generating_functions_l682_682989


namespace clock_display_four_different_digits_l682_682314

theorem clock_display_four_different_digits :
  (∑ t in finset.range (24*60), if (((t / 60).div1000 ≠ (t / 60).mod1000) ∧ 
    ((t / 60).div1000 ≠ (t % 60).div1000) ∧ ((t / 60).div1000 ≠ (t % 60).mod1000) ∧ 
    ((t / 60).mod1000 ≠ (t % 60).div1000) ∧ ((t / 60).mod1000 ≠ (t % 60).mod1000) ∧ 
    ((t % 60).div1000 ≠ (t % 60).mod1000)) then 1 else 0) = 588 :=
by
  sorry

end clock_display_four_different_digits_l682_682314


namespace smallest_positive_multiple_of_45_is_45_l682_682793

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682793


namespace smallest_positive_multiple_of_45_l682_682773

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682773


namespace triangle_BC_length_l682_682587

theorem triangle_BC_length 
  (A B C D E F : Point)
  (AB AC BC : ℝ) 
  (AD AE AF : Line) 
  (h1 : AD.isAngleBisector ∠BAC)
  (h2 : AE.isMedian)
  (h3 : AF.isAltitude)
  (h4 : AB = 154)
  (h5 : AC = 128)
  (h6 : 9 * DE.length = EF.length) : 
  BC = 94 := 
sorry

end triangle_BC_length_l682_682587


namespace smallest_positive_multiple_of_45_is_45_l682_682918

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682918


namespace find_monic_quadratic_polynomial_l682_682483

noncomputable def f : ℝ → ℝ := λ x, x^2 + 5 * x + 6

theorem find_monic_quadratic_polynomial :
  (∀ x, f x = x^2 + 5 * x + 6) ∧ (f 0 = 6) ∧ (f 1 = 12) :=
by
  show (∀ x, f x = x^2 + 5 * x + 6) ∧ (f 0 = 6) ∧ (f 1 = 12)
  sorry

end find_monic_quadratic_polynomial_l682_682483


namespace number_of_possible_x_values_l682_682329

theorem number_of_possible_x_values :
  let x_values := {x : ℤ | 18 < x ∧ x < 42} in
  x_values.to_finset.card = 23 :=
by
  sorry

end number_of_possible_x_values_l682_682329


namespace sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l682_682971

theorem sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30 :
  (7^30 + 13^30) % 100 = 0 := 
sorry

end sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l682_682971


namespace smallest_positive_multiple_of_45_l682_682849

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682849


namespace sum_of_midsegments_lt_half_sum_of_edges_l682_682331

-- Define the vertices of the tetrahedron
variables {A B C S : Point}

-- Define midpoints of the edges
def midpoint (p1 p2 : Point) : Point := sorry

-- Midpoints of the specified edges
def M : Point := midpoint A S
def N : Point := midpoint S C
def P : Point := midpoint S B
def K : Point := midpoint A B
def L : Point := midpoint B C
def R : Point := midpoint A C

-- Distance function
def distance (p1 p2 : Point) : ℝ := sorry

-- Midsegment lengths
def ML : ℝ := distance M L
def KN : ℝ := distance K N
def PR : ℝ := distance P R

-- Edge lengths
def AS : ℝ := distance A S
def SC : ℝ := distance S C
def SB : ℝ := distance S B
def AB : ℝ := distance A B
def BC : ℝ := distance B C
def AC : ℝ := distance A C

theorem sum_of_midsegments_lt_half_sum_of_edges :
  ML + KN + PR < (AS + SC + SB + AB + BC + AC) / 2 := sorry

end sum_of_midsegments_lt_half_sum_of_edges_l682_682331


namespace divisors_of_36_l682_682570

theorem divisors_of_36 : ∀ n : ℕ, n = 36 → (∃ k : ℕ, k = 9) :=
by
  intro n hn
  have h_prime_factors : (n = 2^2 * 3^2) := by rw hn; norm_num
  -- Using the formula for the number of divisors based on prime factorization
  have h_num_divisors : (2 + 1) * (2 + 1) = 9 := by norm_num
  use 9
  rw h_num_divisors
  sorry

end divisors_of_36_l682_682570


namespace fraction_of_girls_on_trip_l682_682432

variables (b : ℕ) (g : ℕ)
variable (twice_as_many_girls : g = 2 * b)
variable (girls_on_trip_fraction : ℚ := 2 / 3)
variable (boys_on_trip_fraction : ℚ := 3 / 5)

theorem fraction_of_girls_on_trip (b g : ℕ) 
  (twice_as_many_girls : g = 2 * b) 
  (girls_on_trip_fraction : ℚ = 2 / 3) 
  (boys_on_trip_fraction : ℚ = 3 / 5) : 
  (girls_on_trip_fraction * g) / (girls_on_trip_fraction * g + boys_on_trip_fraction * b) = 20 / 29 :=
by
  sorry

end fraction_of_girls_on_trip_l682_682432


namespace circle_equation_line_tangent_to_circle_l682_682118

variable (m : ℝ)

noncomputable def conditions (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 0) ∨ (y = m * (x - 1)) ∨ (x ^ 2 + y ^ 2 = (x - 1) ^ 2 + y ^ 2)

theorem circle_equation (x y : ℝ) (h : conditions x y) :
  (x - 1) ^ 2 + y ^ 2 = 1 :=
sorry

theorem line_tangent_to_circle (x y : ℝ) :
  (x = 2 ∧ y = 3 → (x = 2 ∨ 4 * x - 3 * y + 1 = 0)) :=
sorry

end circle_equation_line_tangent_to_circle_l682_682118


namespace decimal_to_octal_365_l682_682462

theorem decimal_to_octal_365 : nat.toDigits 8 365 = [5, 5, 5] :=
by 
  -- proof omitted
  sorry

end decimal_to_octal_365_l682_682462


namespace determine_f_neg_l682_682694

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem determine_f_neg (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_pos : ∀ x, 0 < x → f x = x^2 + x) : 
  ∀ x, x < 0 → f x = -x^2 + x :=
by
  intros x h
  have h1 : f (-x) = -f x := h_odd x
  have h2 : f (-x) = (-x)^2 + (-x) := h_pos (-x) (by linarith)
  rw [neg_sq, neg_eq, neg_neg] at h2
  have h3 : -f x = x^2 - x := h2
  simp at h3
  exact eq_neg_of_eq_neg h3

end determine_f_neg_l682_682694


namespace smallest_b_for_perfect_square_l682_682740

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ (∃ (n : ℤ), 4 * b + 5 = n ^ 2) ∧ b = 5 := 
sorry

end smallest_b_for_perfect_square_l682_682740


namespace probability_real_roots_l682_682577

theorem probability_real_roots : 
  let S := {2, 4, 6, 8}
  let pairs := { (b, c) | b ∈ S ∧ c ∈ S ∧ b ≠ c ∧ b^2 - 4 * c ≥ 0 }
  let total_pairs := { (b, c) | b ∈ S ∧ c ∈ S ∧ b ≠ c }
  fintype.card(pairs) = 6 ∧ fintype.card(total_pairs) = 12 →
  (fintype.card(pairs).to_real / fintype.card(total_pairs).to_real = 1/2) :=
by
  let S := {2, 4, 6, 8}
  let pairs := { (b, c) | b ∈ S ∧ c ∈ S ∧ b ≠ c ∧ b^2 - 4 * c ≥ 0 }
  let total_pairs := { (b, c) | b ∈ S ∧ c ∈ S ∧ b ≠ c }
  have h1: fintype.card(pairs) = 6 := sorry
  have h2: fintype.card(total_pairs) = 12 := sorry
  exact sorry

end probability_real_roots_l682_682577


namespace problem_solution_l682_682528

noncomputable def length_segment_AB : ℝ :=
  let k : ℝ := 1 -- derived from 3k - 3 = 0
  let A : ℝ × ℝ := (0, k) -- point (0, k)
  let C : ℝ × ℝ := (3, -1) -- center of the circle
  let r : ℝ := 1 -- radius of the circle
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) -- distance formula
  Real.sqrt (AC^2 - r^2)

theorem problem_solution :
  length_segment_AB = 2 * Real.sqrt 3 :=
by
  sorry

end problem_solution_l682_682528


namespace choir_members_l682_682686

theorem choir_members : ∃ n : ℤ, 150 < n ∧ n < 300 ∧ n % 6 = 1 ∧ n % 8 = 3 ∧ n % 9 = 5 := 
by {
  use 193, sorry,
}

end choir_members_l682_682686


namespace smallest_positive_multiple_of_45_is_45_l682_682914

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682914


namespace pats_password_length_l682_682646

-- Definitions based on conditions
def num_lowercase_letters := 8
def num_uppercase_numbers := num_lowercase_letters / 2
def num_symbols := 2

-- Translate the math proof problem to Lean 4 statement
theorem pats_password_length : 
  num_lowercase_letters + num_uppercase_numbers + num_symbols = 14 := by
  sorry

end pats_password_length_l682_682646


namespace length_AB_is_2sqrt3_l682_682525

open Real

-- Definitions of circle C and line l, point A
def circle_C := {x : ℝ × ℝ | (x.1 - 3)^2 + (x.2 + 1)^2 = 1}
def line_l (k : ℝ) := {p : ℝ × ℝ | k * p.1 + p.2 - 2 = 0}
def point_A (k : ℝ) := (0, k)

-- Conditions: line l passes through the center of the circle and is the axis of symmetry
def is_axis_of_symmetry_l (k : ℝ) := ∀ p: ℝ × ℝ, p ∈ circle_C → line_l k p

-- Main theorem to be proved
theorem length_AB_is_2sqrt3 (k : ℝ) (h_sym: is_axis_of_symmetry_l k) : 
  let A := point_A 1 in 
  let C := (3, -1) in 
  let radius := 1 in 
  let AC := sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) in
  sqrt (AC^2 - radius^2) = 2 * sqrt 3 :=
sorry -- proof not required

end length_AB_is_2sqrt3_l682_682525


namespace solution_set_of_equation_l682_682072

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem solution_set_of_equation (x : ℝ) (h : x > 0): (x^(log_base 10 x) = x^3 / 100) ↔ (x = 10 ∨ x = 100) := 
by sorry

end solution_set_of_equation_l682_682072


namespace smallest_positive_multiple_of_45_is_45_l682_682803

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682803


namespace number_of_divisors_of_36_l682_682550

theorem number_of_divisors_of_36 :  
  let n := 36
  number_of_divisors n = 9 := 
by 
  sorry

end number_of_divisors_of_36_l682_682550


namespace bob_weight_l682_682354

theorem bob_weight (j b : ℝ) (h1 : j + b = 200) (h2 : b - j = b / 3) : b = 120 :=
sorry

end bob_weight_l682_682354


namespace lisa_breakfast_eggs_l682_682636

noncomputable def total_eggs_per_year (children : ℕ) (eggs_per_child : ℕ) (husband_eggs : ℕ) (self_eggs : ℕ) (days_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  let eggs_per_day := (children * eggs_per_child) + husband_eggs + self_eggs
  in eggs_per_day * days_per_week * weeks_per_year

theorem lisa_breakfast_eggs :
  total_eggs_per_year 4 2 3 2 5 52 = 3380 :=
by
  sorry

end lisa_breakfast_eggs_l682_682636


namespace sufficient_not_necessary_condition_l682_682995

open Complex

theorem sufficient_not_necessary_condition (a b : ℝ) (i := Complex.I) :
  (a = 1 ∧ b = 1) → ((a + b * i)^2 = 2 * i) ∧ ¬((a + b * i)^2 = 2 * i → a = 1 ∧ b = 1) :=
by
  sorry

end sufficient_not_necessary_condition_l682_682995


namespace rachel_stuffing_envelopes_l682_682654

theorem rachel_stuffing_envelopes :
  ∀ (total_envelopes : ℕ) (hours_available : ℕ) (envelopes_first_hour : ℕ) (envelopes_second_hour : ℕ) 
    (remaining_envelopes : ℕ) (remaining_hours : ℕ) (envelopes_per_hour_needed : ℕ),
  total_envelopes = 1500 →
  hours_available = 8 →
  envelopes_first_hour = 135 →
  envelopes_second_hour = 141 →
  remaining_envelopes = total_envelopes - envelopes_first_hour - envelopes_second_hour →
  remaining_hours = hours_available - 2 →
  envelopes_per_hour_needed = remaining_envelopes / remaining_hours →
  envelopes_per_hour_needed = 204 :=
by
  intros total_envelopes hours_available envelopes_first_hour envelopes_second_hour remaining_envelopes remaining_hours envelopes_per_hour_needed
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4] at h5
  rw [h1, h2, h5] at h6
  rw [h6, h5] at h7
  exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num) h7

end rachel_stuffing_envelopes_l682_682654


namespace dogs_count_l682_682342

namespace PetStore

-- Definitions derived from the conditions
def ratio_cats_dogs := 3 / 4
def num_cats := 18
def num_groups := num_cats / 3
def num_dogs := 4 * num_groups

-- The statement to prove
theorem dogs_count : num_dogs = 24 :=
by
  sorry

end PetStore

end dogs_count_l682_682342


namespace horner_example_l682_682000

def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldr (λ a acc => a + x * acc) 0

theorem horner_example : horner [12, 35, -8, 79, 6, 5, 3] (-4) = 220 := by
  sorry

end horner_example_l682_682000


namespace concyclic_iff_perpendicular_l682_682617

variables (A B C G A': Type) [MetricSpace G] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace A']
variables [IsCentroid G A B C]  -- G is the centroid of triangle ABC
variables [IsSymmetric A C A']  -- A' is the symmetric of A with respect to C

theorem concyclic_iff_perpendicular :
  Concyclic G B C A' ↔ Perpendicular (dist G A) (dist G C) :=
sorry

end concyclic_iff_perpendicular_l682_682617


namespace roger_needs_packs_l682_682223

def members : ℕ := 13
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people : ℕ := members + coaches + helpers

def trail_mix_pack_size : ℕ := 6
def granola_bars_pack_size : ℕ := 8
def fruit_cups_pack_size : ℕ := 4

def packs_needed (total : ℕ) (pack_size : ℕ) : ℕ :=
  if total % pack_size = 0 then total / pack_size else total / pack_size + 1

theorem roger_needs_packs :
  packs_needed total_people trail_mix_pack_size = 3 ∧
  packs_needed total_people granola_bars_pack_size = 3 ∧
  packs_needed total_people fruit_cups_pack_size = 5 :=
by
  unfold total_people
  unfold trail_mix_pack_size
  unfold granola_bars_pack_size
  unfold fruit_cups_pack_size
  unfold packs_needed
  simp
  sorry

end roger_needs_packs_l682_682223


namespace alice_cell_phone_cost_l682_682591

theorem alice_cell_phone_cost
  (base_cost : ℕ)
  (included_hours : ℕ)
  (text_cost_per_message : ℕ)
  (extra_minute_cost : ℕ)
  (messages_sent : ℕ)
  (hours_spent : ℕ) :
  base_cost = 25 →
  included_hours = 40 →
  text_cost_per_message = 4 →
  extra_minute_cost = 5 →
  messages_sent = 150 →
  hours_spent = 42 →
  (base_cost + (messages_sent * text_cost_per_message) / 100 + ((hours_spent - included_hours) * 60 * extra_minute_cost) / 100) = 37 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end alice_cell_phone_cost_l682_682591


namespace four_diff_digits_per_day_l682_682318

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end four_diff_digits_per_day_l682_682318


namespace smallest_positive_multiple_of_45_l682_682895

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682895


namespace max_length_PQ_l682_682272

theorem max_length_PQ (A B C M N : Point) (MN : Segment) (P Q : Point)
  (h_circ : OnCircle A B C M N)
  (h_diam : Diameter MN (CircleCircumference A B C M N))
  (h_side : SameSideMN A B C MN)
  (h_midpoint : MidpointArc A MN)
  (h_intersection_P : Intersection CA MN P)
  (h_intersection_Q : Intersection CB MN Q)
  (h_MN_length : length MN = 1)
  (h_MB_length : length MB = 12/13) : length (PQ) ≤ (17 - 4 * sqrt 15) / 7 :=
sorry

end max_length_PQ_l682_682272


namespace smallest_positive_multiple_of_45_l682_682867

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682867


namespace circle_point_outside_range_l682_682154

theorem circle_point_outside_range (m : ℝ) :
  ¬ (1 + 1 + 4 * m - 2 * 1 + 5 * m = 0) → 
  (m > 1 ∨ (0 < m ∧ m < 1 / 4)) := 
sorry

end circle_point_outside_range_l682_682154


namespace probability_nina_taller_than_lena_l682_682100

variables {M N L O : ℝ}

theorem probability_nina_taller_than_lena (h₁ : N < M) (h₂ : L > O) : 
  ∃ P : ℝ, P = 0 ∧ ∀ M N L O, M ≠ N ∧ M ≠ L ∧ M ≠ O ∧ N ≠ L ∧ N ≠ O ∧ L ≠ O → 
  (M > N → O < L → P = 0) :=
by sorry

end probability_nina_taller_than_lena_l682_682100


namespace measure_PQR_degrees_l682_682214

open Real

variable (R P Q S : Type)
variable {angle_RSP : R → P → Q → Prop}
variable {angle_QSP : ℝ}
variable {RS_SQ_neq : R ≠ S ∧ P ≠ S}
variable {angle_PSQ : ℝ}
variable (line_RSP : R → S → P)
variable [DecidableEq Type]

theorem measure_PQR_degrees : 
  angle_RSP R S P ∧ angle_QSP = 70 ∧ RS_SQ_neq ∧ angle_PSQ = 60 → angle_PQR = 60 := 
by 
  sorry

end measure_PQR_degrees_l682_682214


namespace max_value_f_at_a1_f_div_x_condition_l682_682159

noncomputable def f (a x : ℝ) : ℝ := (a - x) * Real.exp x - 1

theorem max_value_f_at_a1 :
  ∀ x : ℝ, (f 1 0) = 0 ∧ ( ∀ y : ℝ, y ≠ 0 → f 1 y < f 1 0) := 
sorry

theorem f_div_x_condition :
  ∀ x : ℝ, x ≠ 0 → (((f 1 x) / x) < 1) :=
sorry

end max_value_f_at_a1_f_div_x_condition_l682_682159


namespace derivative_of_y_l682_682988

variable {x : ℝ}

def y (x : ℝ) : ℝ := 
  x / (2 * Real.sqrt (1 - 4 * x^2)) * Real.arcsin (2 * x) + 
  1 / 8 * Real.log (1 - 4 * x^2)

theorem derivative_of_y (h : x ≠ 1 / 2, h' : x ≠ -1 / 2) : 
  deriv y x = Real.arcsin (2 * x) / (2 * (1 - 4 * x^2) * Real.sqrt (1 - 4 * x^2)) :=
by 
  sorry

end derivative_of_y_l682_682988


namespace smallest_positive_multiple_l682_682777

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682777


namespace general_formula_a_general_formula_b_sum_T_n_l682_682138

noncomputable def a_n (n : ℕ) : ℕ := 3 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 2 ^ n
noncomputable def S_n (n : ℕ) : ℕ := n * (a_n 1 + a_n n) / 2  -- Sum of arithmetic series
noncomputable def T_n (n : ℕ) : ℕ := (3*n - 4) * 2^(n+1) + 8

-- Given conditions
def cond1 : Prop := a_1 = 2
def cond2 : Prop := b_1 = 2
def cond3 : Prop := a_3 + b_3 = 16
def cond4 : Prop := S_4 + b_3 = 34

-- Theorem statements
theorem general_formula_a (n : ℕ) : cond1 ∧ cond2 ∧ cond3 ∧ cond4 → a_n n = 3 * n - 1 := by
  sorry

theorem general_formula_b (n : ℕ) : cond1 ∧ cond2 ∧ cond3 ∧ cond4 → b_n n = 2 ^ n := by
  sorry

theorem sum_T_n (n : ℕ) : general_formula_a n ∧ general_formula_b n ∧ cond1 ∧ cond2 ∧ cond3 ∧ cond4 → T_n n = (3*n - 4) * 2^(n + 1) + 8 := by
  sorry

end general_formula_a_general_formula_b_sum_T_n_l682_682138


namespace clock_four_different_digits_l682_682303

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end clock_four_different_digits_l682_682303


namespace constant_length_O1O2_l682_682601

-- Define trapezoid ABCD, where AD is parallel to BC, and E is a point on AB
variable {A B C D E : Point}
variable {O1 O2 : Point}

-- Conditions
axiom trapezoid (hAD_BC : AD ∥ BC) :
    IsTrapezoid A B C D := sorry

axiom point_on_side (hE_on_AB : E ∈ Seg A B) :
    PointOnSide E A B := sorry

axiom circumcenter_AED (hO1 : IsCircumcenterOfTriangle O1 A E D) :
    IsCircumcenterOfTriangle O1 A E D := sorry

axiom circumcenter_BEC (hO2 : IsCircumcenterOfTriangle O2 B E C) :
    IsCircumcenterOfTriangle O2 B E C := sorry

-- Define the statement to prove
theorem constant_length_O1O2 :
    ∀ (E : Point),
    (E ∈ Seg A B) →
    IsCircumcenterOfTriangle O1 A E D →
    IsCircumcenterOfTriangle O2 B E C →
    O1O2 = (DC / (2 * Real.sin A)) :=
by
    intros E hE_on_AB hO1 hO2
    sorry

end constant_length_O1O2_l682_682601


namespace perp_OI_BC_l682_682250

variables {A B C E F I P Q O : Type*}
variables [triangle ABC : Type*] [scalene ABC] [angle (A, B, C) = 60]
variables [incenter I ABC] [feet_of_angle_bisectors E F ABC]
variables [equilateral P E F] [equilateral Q E F]
variables [circumcenter O (A, P, Q)]

theorem perp_OI_BC : (O I) ⊥ (B C) :=
sorry

end perp_OI_BC_l682_682250


namespace smallest_b_for_perfect_square_l682_682741

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ (∃ (n : ℤ), 4 * b + 5 = n ^ 2) ∧ b = 5 := 
sorry

end smallest_b_for_perfect_square_l682_682741


namespace smallest_positive_multiple_l682_682778

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682778


namespace general_term_formula_for_sequence_l682_682179

theorem general_term_formula_for_sequence (a b : ℕ → ℝ) 
  (h1 : ∀ n, 2 * b n = a n + a (n + 1)) 
  (h2 : ∀ n, (a (n + 1))^2 = b n * b (n + 1)) 
  (h3 : a 1 = 1) 
  (h4 : a 2 = 3) :
  ∀ n, a n = (n^2 + n) / 2 :=
by
  sorry

end general_term_formula_for_sequence_l682_682179


namespace calculate_expression_l682_682011

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculate_expression_l682_682011


namespace tan_2alpha_cos_pi_six_minus_alpha_l682_682264

theorem tan_2alpha (α : ℝ) (h1 : tan α = 3 / 4) : tan (2 * α) = 24 / 7 := by
  sorry

theorem cos_pi_six_minus_alpha (α : ℝ) (h1 : tan α = 3 / 4) (h2 : sin α = 3 / 5) (h3 : cos α = 4 / 5) : 
  cos (π / 6 - α) = (3 + 4 * Real.sqrt 3) / 10 := by
  sorry

end tan_2alpha_cos_pi_six_minus_alpha_l682_682264


namespace option_a_option_b_option_c_final_result_l682_682110

-- Define the original functions and conditions
def f (ω φ x : ℝ) := Real.sin (ω * x + π / 3 + φ)
def g (ω φ x : ℝ) := Real.sin (ω * x + φ)

-- Prove φ = π / 6 given f(x) is even
theorem option_a (ω : ℝ) (hω : ω > 0) (φ : ℝ) (hφ : abs φ < π / 2) (h_even : ∀ x : ℝ, f ω φ x = f ω φ (-x)) :
  φ = π / 6 :=
sorry

-- Prove ω = 2 / 3 given the smallest positive period of g(x) is 3π
theorem option_b (ω : ℝ) (hφ : abs φ < π / 2) (h_period : ∀ x : ℝ, g ω φ (x + 3 * π) = g ω φ x) :
  ω = 2 / 3 :=
sorry

-- Prove 7 / 3 < ω ≤ 10 / 3 given g(x) has exactly 3 extreme points in the interval (0, π)
theorem option_c (ω : ℝ) (hφ : abs φ < π / 2) (h_extreme_points : ∀ x : ℝ, g ω φ x = g ω φ (x + π / 3) →  ∃! (a b c : ℝ), (0 < a < b < c < π)) :
   7 / 3 < ω ∧ ω ≤ 10 / 3 :=
sorry

-- Define the final result that combines all valid options
theorem final_result (ω : ℝ) (hω : ω > 0) (φ : ℝ) (hφ : abs φ < π / 2) :
  (φ = π / 6 ∧ (∀ x, g ω φ (x + 3 * π) = g ω φ x → ω = 2 / 3) ∧ (∀ x, g ω φ x = g ω φ (x + π / 3) → 3 < ω ∧ ω ≤ 10 / 3)) :=
sorry

end option_a_option_b_option_c_final_result_l682_682110


namespace probability_nina_taller_than_lena_l682_682101

variables {M N L O : ℝ}

theorem probability_nina_taller_than_lena (h₁ : N < M) (h₂ : L > O) : 
  ∃ P : ℝ, P = 0 ∧ ∀ M N L O, M ≠ N ∧ M ≠ L ∧ M ≠ O ∧ N ≠ L ∧ N ≠ O ∧ L ≠ O → 
  (M > N → O < L → P = 0) :=
by sorry

end probability_nina_taller_than_lena_l682_682101


namespace present_age_ratio_l682_682709

-- Define the variables and the conditions
variable (S M : ℕ)

-- Condition 1: Sandy's present age is 84 because she was 78 six years ago
def present_age_sandy := S = 84

-- Condition 2: Sixteen years from now, the ratio of their ages is 5:2
def age_ratio_16_years := (S + 16) * 2 = 5 * (M + 16)

-- The goal: The present age ratio of Sandy to Molly is 7:2
theorem present_age_ratio {S M : ℕ} (h1 : S = 84) (h2 : (S + 16) * 2 = 5 * (M + 16)) : S / M = 7 / 2 :=
by
  -- Integrating conditions
  have hS : S = 84 := h1
  have hR : (S + 16) * 2 = 5 * (M + 16) := h2
  -- We need a proof here, but we'll skip it for now
  sorry

end present_age_ratio_l682_682709


namespace jill_marathon_time_l682_682611

noncomputable def marathon_distance : ℝ := 41 -- km
noncomputable def jack_time : ℝ := 4.5 -- hours
noncomputable def speed_ratio : ℝ := 0.9111111111111111 -- ratio Jack : Jill

theorem jill_marathon_time :
  let jill_speed := (marathon_distance / jack_time) / speed_ratio in
  let jill_time := marathon_distance / jill_speed in
  jill_time = 4.1 :=
by
  sorry

end jill_marathon_time_l682_682611


namespace quadratic_has_distinct_roots_l682_682335

theorem quadratic_has_distinct_roots
  (a b c : ℝ)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_angle : ∃ β : ℝ, β > (real.pi / 3) ∧ b^2 = a^2 + c^2 - 2 * a * c * real.cos β) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * b * x1 + c = 0 ∧ a * x2^2 + 2 * b * x2 + c = 0 :=
by
  sorry

end quadratic_has_distinct_roots_l682_682335


namespace orchid_bushes_planted_l682_682717

theorem orchid_bushes_planted (b1 b2 : ℕ) (h1 : b1 = 22) (h2 : b2 = 35) : b2 - b1 = 13 :=
by 
  sorry

end orchid_bushes_planted_l682_682717


namespace smallest_positive_multiple_of_45_l682_682865

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682865


namespace smallest_positive_multiple_of_45_is_45_l682_682955

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682955


namespace smallest_positive_multiple_of_45_l682_682854

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682854


namespace compute_α_l682_682242

open Complex

def α : ℂ := 6 - 3 * Complex.i
def β : ℂ := 4 + 3 * Complex.i

theorem compute_α (h1 : ∃ x : ℝ, (α + β) = x ∧ 0 < x)
                  (h2 : ∃ z : ℝ, (Complex.i * (α - 3 * β)) = z ∧ 0 < z) :
  α = 6 - 3 * Complex.i :=
by
  sorry

end compute_α_l682_682242


namespace values_pqr_l682_682248

open Complex

theorem values_pqr (p q r : ℂ) (h1 : p + q + r = 2) (h2 : p * q * r = 2) (h3 : p * q + p * r + q * r = 0) :
  {p, q, r} = {2, Complex.sqrt 2, -Complex.sqrt 2} :=
by
  sorry

end values_pqr_l682_682248


namespace pow_two_gt_cube_l682_682650

theorem pow_two_gt_cube (n : ℕ) (h : 10 ≤ n) : 2^n > n^3 := sorry

end pow_two_gt_cube_l682_682650


namespace solve_cubic_sum_l682_682676

theorem solve_cubic_sum :
  ∀ (x : ℝ), (real.cbrt (x + 2) + real.cbrt (3 * x - 1) = real.cbrt (16 * x + 4)) →
  (x = -0.25 ∨ x = 1.5) →
  -0.25 + 1.5 = 1.25 :=
by {
  intros x h_eq h_sol,
  cases h_sol;
  rw h_sol;
  norm_num
}

end solve_cubic_sum_l682_682676


namespace smallest_positive_multiple_of_45_l682_682748

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682748


namespace smallest_positive_multiple_of_45_l682_682763

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682763


namespace sum_22_consecutive_integers_is_perfect_cube_l682_682713

theorem sum_22_consecutive_integers_is_perfect_cube :
  ∃ n k : ℕ, (sum (range (n + 22)) - sum (range n) = k^3 ∧ sum (range (n + 22)) - sum (range n) = 1331) :=
by
  sorry

end sum_22_consecutive_integers_is_perfect_cube_l682_682713


namespace evaluate_fraction_sum_l682_682512

-- Define the problem conditions and target equation
theorem evaluate_fraction_sum
    (p q r : ℝ)
    (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 8) :
    6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 11 / 5 := by
  sorry

end evaluate_fraction_sum_l682_682512


namespace smallest_positive_multiple_of_45_l682_682963

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682963


namespace smallest_positive_multiple_45_l682_682815

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682815


namespace solution_set_l682_682161

def f (x : ℝ) : ℝ :=
if x ≤ 0 then -x + 2 else x + 2

theorem solution_set (x : ℝ) : f x ≥ x^2 ↔ -2 ≤ x ∧ x ≤ 2 :=
by sorry

end solution_set_l682_682161


namespace greatest_angle_of_intersection_l682_682397

/--
A strip is the region between two parallel lines. Let A and B be two strips in a plane.
The intersection of strips A and B is a parallelogram P. Let A' be a rotation of A 
in the plane by 60 degrees. The intersection of strips A' and B is a 
parallelogram with the same area as P. Let x be the measure (in degrees) of one interior angle of P.
Prove that the greatest possible value of the number x is 150 degrees.
-/
theorem greatest_angle_of_intersection (A B A' : set (ℝ × ℝ))
  (hA' : A' = {p | ∃ (q ∈ A), p = rotate 60 q} )
  (h_int : ∃ (P : parallelogram), P = A ∩ B)
  (h_int_rot : ∃ (P' : parallelogram), P' = A' ∩ B)
  (h_area : area P' = area P):
  ∃ x : ℝ, x ∈ {angle | angle ∈ interior_angles P ∧ angle ≤ 180} ∧ x = 150 := 
sorry

end greatest_angle_of_intersection_l682_682397


namespace coeff_x2_in_pq_expansion_l682_682060

def p (x : ℝ) : ℝ := 4 * x^3 + 3 * x^2 + 2 * x + 1
def q (x : ℝ) : ℝ := 2 * x^3 + x^2 + 6 * x + 5

theorem coeff_x2_in_pq_expansion :
  (p * q).coeff 2 = 5 :=
sorry

end coeff_x2_in_pq_expansion_l682_682060


namespace find_first_term_l682_682714

noncomputable def a (r : ℝ) : ℝ := 6 * (1 - r)

theorem find_first_term 
    (h1 : ∀ a r, a / (1 - r) = 6)
    (h2 : ∀ a r, a + a * r = 8/3) :
  ∃ r : ℝ, a r = 6 + 2 * Real.sqrt 5 ∨ a r = 6 - 2 * Real.sqrt 5 :=
begin
  sorry
end

end find_first_term_l682_682714


namespace smallest_positive_multiple_of_45_l682_682939

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682939


namespace undefined_sum_slope_y_intercept_of_vertical_line_l682_682271

theorem undefined_sum_slope_y_intercept_of_vertical_line :
  ∀ (C D : ℝ × ℝ), C.1 = 8 → D.1 = 8 → C.2 ≠ D.2 →
  ∃ (m b : ℝ), false :=
by
  intros
  sorry

end undefined_sum_slope_y_intercept_of_vertical_line_l682_682271


namespace purchase_price_of_furniture_l682_682698

theorem purchase_price_of_furniture (marked_price discount_rate profit_rate : ℝ) 
(h_marked_price : marked_price = 132) 
(h_discount_rate : discount_rate = 0.1)
(h_profit_rate : profit_rate = 0.1)
: ∃ a : ℝ, (marked_price * (1 - discount_rate) - a = profit_rate * a) ∧ a = 108 := by
  sorry

end purchase_price_of_furniture_l682_682698


namespace parabola_range_intersection_intersection_expression_integral_evaluation_l682_682240

noncomputable def parabola_intersection_range (u : ℝ) : Prop :=
  let C1 := λ x : ℝ, -x^2 + 1
  let C2 := λ x : ℝ, (x - u)^2 + u
  ∃ x : ℝ, C1 x = C2 x

theorem parabola_range_intersection :
  ∀ u : ℝ, parabola_intersection_range u ↔ -Real.sqrt 3 - 1 ≤ u ∧ u ≤ Real.sqrt 3 - 1 :=
by
  sorry

theorem intersection_expression (u : ℝ) (hu : -Real.sqrt 3 - 1 ≤ u ∧ u ≤ Real.sqrt 3 - 1) :
  let x1 := u / 2 - Real.sqrt((u + 2) * (u - 1)) / 2
  let x2 := u / 2 + Real.sqrt((u + 2) * (u - 1)) / 2
  let y1 := 1 - x1^2
  let y2 := 1 - x2^2
  2 * |x1 * y2 - x2 * y1| = (u^2 + u + 1) * Real.sqrt(2 - u - u^2) :=
by
  sorry

theorem integral_evaluation :
  let f := λ u : ℝ, (u^2 + u + 1) * Real.sqrt(2 - u - u^2)
  ∫ u in -Real.sqrt(3) - 1..Real.sqrt(3) - 1, f u = (21 * Real.pi) / 8 :=
by
  sorry

end parabola_range_intersection_intersection_expression_integral_evaluation_l682_682240


namespace smallest_positive_multiple_l682_682779

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682779


namespace find_sum_of_squares_l682_682123

theorem find_sum_of_squares 
  (A B C H O : Point)
  (R : ℝ)
  (BC CA AB : ℝ)
  (d : ℝ)
  (orthocenter_cond : Orthocenter A B C H)
  (circumcenter_cond : Circumcenter A B C O)
  (circumradius_cond : R = 3)
  (distance_cond : d = 1)
  (sides_cond : BC = |B - C| ∧ CA = |C - A| ∧ AB = |A - B|) :
  BC^2 + CA^2 + AB^2 = 80 :=
sorry

end find_sum_of_squares_l682_682123


namespace monotonic_increasing_interval_l682_682332

noncomputable def f : ℝ → ℝ := λ x, log (1 / 3) (4 + 3 * x - x^2)

theorem monotonic_increasing_interval : ∀ x, x ∈ Icc (3 / 2) 4 → monotone (f : ℝ → ℝ) := by
  sorry

end monotonic_increasing_interval_l682_682332


namespace smallest_positive_multiple_of_45_l682_682872

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682872


namespace arithmetic_sequence_general_formula_lambda_range_l682_682505

-- Definitions of arithmetic and geometric sequences
def arith_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def geom_seq (b r : ℚ) (n : ℕ) : ℚ := b * r^n

-- Statement of the proof problem
theorem arithmetic_sequence_general_formula 
  (a1 : ℤ) (d : ℤ)
  (S7 : ℤ) 
  (h1 : 7 * a1 + (7 * 6 / 2) * d = S7)
  (h2 : (arith_seq a1 d 1)^2 = (arith_seq a1 d 0) * (arith_seq a1 d 9))
  (hnz : d ≠ 0)
  (H : S7 = 35) :
  ∀ n : ℕ, arith_seq a1 d n = n + 1 :=
by
  intros
  sorry

theorem lambda_range 
  (a1 : ℤ) (d : ℤ) 
  (lambda : ℝ) 
  (T_n : ℕ → ℝ)
  (a : ℕ → ℤ := arith_seq a1 d)
  (H : ∀ n : ℕ, T_n n = 1 / 2 - 1 / (n + 2)) 
  (h : ∃ n : ℕ, T_n n - λ * (a n) ≥ 0) :
  λ ≤ 1 / 16 :=
by
  sorry

end arithmetic_sequence_general_formula_lambda_range_l682_682505


namespace min_reciprocal_sum_l682_682632

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x) + (1/y) = 3 + 2 * Real.sqrt 2 :=
sorry

end min_reciprocal_sum_l682_682632


namespace probability_nina_taller_than_lena_l682_682102

variables {M N L O : ℝ}

theorem probability_nina_taller_than_lena (h₁ : N < M) (h₂ : L > O) : 
  ∃ P : ℝ, P = 0 ∧ ∀ M N L O, M ≠ N ∧ M ≠ L ∧ M ≠ O ∧ N ≠ L ∧ N ≠ O ∧ L ≠ O → 
  (M > N → O < L → P = 0) :=
by sorry

end probability_nina_taller_than_lena_l682_682102


namespace smallest_positive_multiple_of_45_l682_682964

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682964


namespace ratio_perimeter_pentagon_to_square_l682_682296

theorem ratio_perimeter_pentagon_to_square
  (a : ℝ) -- Let a be the length of each side of the square
  (T_perimeter S_perimeter : ℝ) 
  (h1 : T_perimeter = S_perimeter) -- Given the perimeter of the triangle equals the perimeter of the square
  (h2 : S_perimeter = 4 * a) -- Given the perimeter of the square is 4 times the length of its side
  (P_perimeter : ℝ)
  (h3 : P_perimeter = (T_perimeter + S_perimeter) - 2 * a) -- Perimeter of the pentagon considering shared edge
  :
  P_perimeter / S_perimeter = 3 / 2 := 
sorry

end ratio_perimeter_pentagon_to_square_l682_682296


namespace smallest_positive_multiple_of_45_l682_682877

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682877


namespace angle_equality_l682_682631

variables {A B C D X Y : Type*}
variables [ConvexQuadrilateral A B C D]
variables [PointInQuad X A B C D]
variables [IntersectionOfPerpendicularBisectors Y A B C D]

-- Conditions
def conditions (h1: ¬Parallel A B C D)
               (h2: ∠ ADX = ∠ BCX ∧ ∠ ADX < 90)
               (h3: ∠ DAX = ∠ CBX ∧ ∠ DAX < 90) : Prop := 
  ¬Parallel A B C D ∧ ∠ ADX = ∠ BCX ∧ ∠ ADX < 90 ∧ ∠ DAX = ∠ CBX ∧ ∠ DAX < 90

-- Theorem statement
theorem angle_equality (h1: ¬Parallel A B C D)
                       (h2: ∠ ADX = ∠ BCX ∧ ∠ ADX < 90)
                       (h3: ∠ DAX = ∠ CBX ∧ ∠ DAX < 90)
                       (h4: PerpendicularBisectorsIntersect Y A B C D) :
  ∠ AY B = 2 * ∠ ADX :=
by {
  apply conditions h1 h2 h3,
  -- Insert proof steps here
  sorry
}

end angle_equality_l682_682631


namespace smallest_bob_number_l682_682422

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : Set ℕ := { p | is_prime p ∧ p ∣ n }

def alice_number := 36
def bob_number (m : ℕ) : Prop := prime_factors alice_number ⊆ prime_factors m

-- Proof problem statement
theorem smallest_bob_number :
  ∃ m, bob_number m ∧ m = 6 :=
sorry

end smallest_bob_number_l682_682422


namespace smallest_positive_multiple_l682_682786

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682786


namespace two_colonies_reach_limit_in_same_time_l682_682983

theorem two_colonies_reach_limit_in_same_time (d : ℕ) (h : 16 = d): 
  d = 16 :=
by
  /- Asserting that if one colony takes 16 days, two starting together will also take 16 days -/
  sorry

end two_colonies_reach_limit_in_same_time_l682_682983


namespace simplify_expression_l682_682667

variable (m : ℝ) (h : m ≠ 0)

theorem simplify_expression : ( (1/(3*m))^(-3) * (2*m)^(4) ) = 432 * m^(7) := by sorry

end simplify_expression_l682_682667


namespace smallest_positive_multiple_45_l682_682908

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682908


namespace probability_two_point_distribution_l682_682531

theorem probability_two_point_distribution 
  (P : ℕ → ℚ)
  (two_point_dist : P 0 + P 1 = 1)
  (condition : P 1 = (3 / 2) * P 0) :
  P 1 = 3 / 5 :=
by
  sorry

end probability_two_point_distribution_l682_682531


namespace ratio_of_graduate_to_non_graduate_l682_682201

variable (G C N : ℕ)

theorem ratio_of_graduate_to_non_graduate (h1 : C = (2:ℤ)*N/(3:ℤ))
                                         (h2 : G.toRat / (G + C) = 0.15789473684210525) :
  G.toRat / N.toRat = 1 / 8 :=
sorry

end ratio_of_graduate_to_non_graduate_l682_682201


namespace hyperbola_focal_distance_distance_focus_to_asymptote_l682_682063

theorem hyperbola_focal_distance :
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  (2 * c = 4) :=
by sorry

theorem distance_focus_to_asymptote :
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let focus := (c, 0)
  let A := -Real.sqrt 3
  let B := 1
  let C := 0
  let distance := (|A * focus.fst + B * focus.snd + C|) / Real.sqrt (A ^ 2 + B ^ 2)
  (distance = Real.sqrt 3) :=
by sorry

end hyperbola_focal_distance_distance_focus_to_asymptote_l682_682063


namespace outfits_count_l682_682290

theorem outfits_count 
  (shirts : ℕ) 
  (ties : ℕ)
  (ties_optional : ties + 1) 
  (num_outfits : ℕ := shirts * (ties + 1))
  (h_shirts : shirts = 8)
  (h_ties : ties = 7) :
  num_outfits = 64 := 
by
  sorry

end outfits_count_l682_682290


namespace smallest_positive_multiple_of_45_l682_682753

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682753


namespace lambda_greater_than_one_l682_682532

variables (n : Nat) (c λ : ℝ)
noncomputable def S (n : ℕ) : ℝ := 2^(n+1) + c
noncomputable def a (n : ℕ) : ℝ := 2^n

theorem lambda_greater_than_one :
  (∀ n : ℕ, n > 0 → a n + 2 * (-1) ^ n < λ * (a (n + 1) + 2 * (-1) ^ (n + 1))) →
  λ > 1 :=
sorry

end lambda_greater_than_one_l682_682532


namespace total_profit_calculation_l682_682211

variable (x : ℝ) (total_profit : ℝ)
variable (A_investment B_investment C_investment : ℝ)

-- Conditions
def condition1 : Prop := A_investment / C_investment = 3 / 2
def condition2 : Prop := A_investment / B_investment = 3 / 1
def condition3 : Prop := C_investment / total_profit = 1 / 3 ∧ C_investment = 20000

theorem total_profit_calculation (h1 : condition1) (h2 : condition2) (h3 : condition3) : total_profit = 60000 := by
  sorry

end total_profit_calculation_l682_682211


namespace num_elements_intersection_l682_682135

-- Define the sets A and B
def A : set ℝ := { x | x < 0 ∨ x > 2 }
def B : set ℕ := set.univ

-- Define the complement of A in the real numbers
def complement_A : set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define the intersection of complement_A and B
def intersection_complement_A_B : set ℝ := complement_A ∩ (coe '' B)

-- Define the expected result set
def expected_set : set ℝ := {0, 1, 2}

-- State the theorem about the number of elements in the intersection set
theorem num_elements_intersection : set.finite intersection_complement_A_B ∧ set.to_finset intersection_complement_A_B.card = 3 := 
sorry

end num_elements_intersection_l682_682135


namespace smallest_positive_multiple_45_l682_682813

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682813


namespace dogs_count_l682_682340

namespace PetStore

-- Definitions derived from the conditions
def ratio_cats_dogs := 3 / 4
def num_cats := 18
def num_groups := num_cats / 3
def num_dogs := 4 * num_groups

-- The statement to prove
theorem dogs_count : num_dogs = 24 :=
by
  sorry

end PetStore

end dogs_count_l682_682340


namespace max_n_for_factored_quadratic_l682_682068

theorem max_n_for_factored_quadratic :
  ∃ (A B : ℤ), A * B = 54 ∧ (3 * B + A) = 163 :=
by
  use 1, 54
  split
  -- Proof part skipped
  sorry -- A * B = 54
  sorry -- 3 * B + A = 163

end max_n_for_factored_quadratic_l682_682068


namespace angle_range_of_scalene_triangle_l682_682209

theorem angle_range_of_scalene_triangle (a b c : ℝ) (A : ℝ) (h1 : a^2 < b^2 + c^2) (h2 : 0 < b) (h3 : 0 < c) (h4 : a = (b^2 + c^2 - 2 * b * c * Math.cos A)^0.5) : 0 < A ∧ A < 90 :=
by
  sorry

end angle_range_of_scalene_triangle_l682_682209


namespace max_La_value_l682_682167

noncomputable def f (a : ℝ) (x : ℝ) := a * x^2 + 8 * x + 3

theorem max_La_value :
  ∃ a < 0, (∀ x ∈ set.Icc (0 : ℝ) (L a), |f a x| ≤ 5) ∧ (L a = (real.sqrt 5 + 1) / 2) where
  L (a : ℝ) : ℝ := sorry := sorry

end max_La_value_l682_682167


namespace fill_bathtub_time_l682_682585

theorem fill_bathtub_time
  (r_cold : ℚ := 1/10)
  (r_hot : ℚ := 1/15)
  (r_empty : ℚ := -1/12)
  (net_rate : ℚ := r_cold + r_hot + r_empty) :
  net_rate = 1/12 → 
  t = 12 :=
by
  sorry

end fill_bathtub_time_l682_682585


namespace find_number_l682_682732

theorem find_number : ∃ x : ℝ, 0.0001 * x = 1.2356 ∧ x = 12356 :=
by
  use 12356
  split
  { rw [mul_comm]
    norm_num
  }
  { norm_num }

end find_number_l682_682732


namespace correlation_coefficient_properties_l682_682212

theorem correlation_coefficient_properties (r : ℝ) :
  (|r| ≤ 1) ∧ (|r| ≠ 1 → the closer |r| is to 0, the weaker the linear correlation between x and y) :=
by
  sorry

end correlation_coefficient_properties_l682_682212


namespace PQ_length_0_l682_682622

noncomputable def length_of_PQ (AC BD : ℝ) (hAC : AC = 12) (hBD : BD = 20) : ℝ :=
  let O : EuclideanGeometry.Point := EuclideanGeometry.midpoint (12 / 2) (20 / 2)
  let N : EuclideanGeometry.Point := EuclideanGeometry.midpoint_point_AB
  let P : EuclideanGeometry.Point := EuclideanGeometry.feet_perpendicular N (12 / 2)
  let Q : EuclideanGeometry.Point := EuclideanGeometry.feet_perpendicular N (20 / 2)
  EuclideanGeometry.distance P Q

theorem PQ_length_0 (AC BD : ℝ) (hAC : AC = 12) (hBD : BD = 20) : length_of_PQ AC BD hAC hBD = 0 :=
by
  sorry

end PQ_length_0_l682_682622


namespace a_500_is_1343_l682_682592

noncomputable def a : ℕ → ℤ
| 0       := 1011 
| 1       := 1013
| (n + 2) := 2 * n - a n - a (n + 1)

theorem a_500_is_1343 : a 499 = 1343 :=
by sorry

end a_500_is_1343_l682_682592


namespace dogs_count_l682_682341

namespace PetStore

-- Definitions derived from the conditions
def ratio_cats_dogs := 3 / 4
def num_cats := 18
def num_groups := num_cats / 3
def num_dogs := 4 * num_groups

-- The statement to prove
theorem dogs_count : num_dogs = 24 :=
by
  sorry

end PetStore

end dogs_count_l682_682341


namespace find_hourly_rate_l682_682266

-- Defining the conditions
def hours_worked : ℝ := 7.5
def overtime_factor : ℝ := 1.5
def total_hours_worked : ℝ := 10.5
def total_earnings : ℝ := 48

-- Proving the hourly rate
theorem find_hourly_rate (R : ℝ) (h : 7.5 * R + (10.5 - 7.5) * 1.5 * R = 48) : R = 4 := by
  sorry

end find_hourly_rate_l682_682266


namespace construct_focus_l682_682127

noncomputable def find_second_focus
(F : Point) (e1 e2 e3 : Line)
(S : Point) : Prop :=
  (∃ F1' F2' F3' : Point,
    (reflect_over_tangent F e1 = F1') ∧
    (reflect_over_tangent F e2 = F2') ∧
    (reflect_over_tangent F e3 = F3') ∧
    let circumcenter := circumcenter_of_triangle F1' F2' F3' in
    circumcenter = S)

theorem construct_focus
(F : Point) (e1 e2 e3 : Line) : ∃ S : Point, find_second_focus F e1 e2 e3 S :=
sorry

end construct_focus_l682_682127


namespace smallest_positive_multiple_45_l682_682814

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682814


namespace sqrt_three_is_an_infinite_non_repeating_decimal_l682_682036

theorem sqrt_three_is_an_infinite_non_repeating_decimal :
  ¬(∃ p q : ℤ, (q ≠ 0) ∧ (p / q : ℝ) = (sqrt 3)) ∧
  (∀ r : ℝ, r = sqrt 3 → ¬ (∃ m : ℕ, (∃ n : ℕ, r = m + n → r = (m / 10^n : ℝ)))) :=
sorry

end sqrt_three_is_an_infinite_non_repeating_decimal_l682_682036


namespace sufficient_but_not_necessary_l682_682609

variable {a : ℝ}

theorem sufficient_but_not_necessary (h : a > 1) : a^2 > a :=
by
  sorry

end sufficient_but_not_necessary_l682_682609


namespace probability_second_roll_odd_given_first_roll_odd_l682_682661

theorem probability_second_roll_odd_given_first_roll_odd :
  let A := (fun (ω : Fin 6) => ω % 2 = 1)
  let B := (fun (ω : Fin 6) => ω % 2 = 1)
  let P := (fun (A : Set (Fin 6)) => (Set.size A).toReal / (Set.size (Fin 6)).toReal)
  P(B | A) = 1 / 2 :=
by
  sorry

end probability_second_roll_odd_given_first_roll_odd_l682_682661


namespace max_team_members_l682_682399

theorem max_team_members (s : Finset ℕ) (h_distinct : ∀ (x ∈ s) (y ∈ s), x ≠ y ∨ x=y) 
  (h_bounds : ∀ (x ∈ s), x ≥ 1 ∧ x ≤ 100) 
  (h_sum : ∀ (x ∈ s) (y ∈ s) (z ∈ s), x ≠ y → x ≠ z → y ≠ z → x ≠ y + z) 
  (h_double : ∀ (x ∈ s) (y ∈ s), x ≠ 2 * y) : 
  s.card ≤ 50 := 
sorry

end max_team_members_l682_682399


namespace obtuse_angle_range_of_a_l682_682193

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem obtuse_angle_range_of_a (a : ℝ) :
  let K := (1 - a, 1 + a)
  let Q := (3, 2 * a)
  slope K Q < 0 ↔ a ∈ set.Ioo (-2 : ℝ) 1 :=
by
  let K := (1 - a, 1 + a)
  let Q := (3, 2 * a)
  calc slope K Q = (2 * a - (1 + a)) / (3 - (1 - a)) : by reflexivity
               ... = (a - 1) / (2 + a) : by linarith
  sorry

end obtuse_angle_range_of_a_l682_682193


namespace pets_remaining_l682_682434

-- Definitions based on conditions
def initial_puppies : ℕ := 7
def initial_kittens : ℕ := 6
def sold_puppies : ℕ := 2
def sold_kittens : ℕ := 3

-- Theorem statement
theorem pets_remaining : initial_puppies + initial_kittens - (sold_puppies + sold_kittens) = 8 :=
by
  sorry

end pets_remaining_l682_682434


namespace smallest_positive_multiple_of_45_is_45_l682_682942

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682942


namespace smallest_positive_multiple_of_45_l682_682767

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682767


namespace system_solution_l682_682352

theorem system_solution (x y : ℤ) (h1 : x + y = 1) (h2 : 2*x + y = 5) : x = 4 ∧ y = -3 :=
by {
  sorry
}

end system_solution_l682_682352


namespace clock_shows_four_different_digits_for_588_minutes_l682_682298

-- Definition of the problem
def isFourDifferentDigits (h1 h2 m1 m2 : Nat) : Bool :=
  (h1 ≠ h2) && (h1 ≠ m1) && (h1 ≠ m2) && (h2 ≠ m1) && (h2 ≠ m2) && (m1 ≠ m2)

noncomputable def countFourDifferentDigitsMinutes : Nat :=
  let validMinutes := List.filter (λ (t : Nat × Nat),
    let (h, m) := t
    let h1 := h / 10
    let h2 := h % 10
    let m1 := m / 10
    let m2 := m % 10
    isFourDifferentDigits h1 h2 m1 m2
  ) (List.product (List.range 24) (List.range 60))
  validMinutes.length

theorem clock_shows_four_different_digits_for_588_minutes :
  countFourDifferentDigitsMinutes = 588 := sorry

end clock_shows_four_different_digits_for_588_minutes_l682_682298


namespace triangle_solution_l682_682978

theorem triangle_solution (a b c : ℝ) (A B : ℝ) : 
  (a = 7 ∧ b = 14 ∧ A = 30) ∨
  (a = 6 ∧ b = 9 ∧ A = 45) ∨
  (a = 30 ∧ b = 25 ∧ A = 150) ∨
  (a = 9 ∧ b = 10 ∧ B = 60) →
  (∃ B, (a = 30 ∧ b = 25 ∧ A = 150) ∧ 
        (sin B = 5 / 12 ∧ B < 90)) :=
sorry

end triangle_solution_l682_682978


namespace megan_earnings_l682_682267

-- Define the given conditions
def bead_necklaces : ℕ := 7
def gem_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 9

-- Define the total number of necklaces
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

-- Define the total earnings
def total_earnings : ℕ := total_necklaces * cost_per_necklace

-- Prove that the total earnings are 90 dollars
theorem megan_earnings : total_earnings = 90 := by
  sorry

end megan_earnings_l682_682267


namespace calculate_expression_l682_682009

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculate_expression_l682_682009


namespace quadratic_condition_not_necessary_and_sufficient_l682_682108

theorem quadratic_condition_not_necessary_and_sufficient (a b c : ℝ) :
  ¬((∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (b^2 - 4 * a * c < 0)) :=
sorry

end quadratic_condition_not_necessary_and_sufficient_l682_682108


namespace sqrt23_minus1_mul_sqrt23_plus1_eq_22_l682_682440

theorem sqrt23_minus1_mul_sqrt23_plus1_eq_22 :
  (sqrt 23 - 1) * (sqrt 23 + 1) = 22 :=
by 
  sorry

end sqrt23_minus1_mul_sqrt23_plus1_eq_22_l682_682440


namespace smallest_positive_multiple_of_45_l682_682881

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682881


namespace total_length_remaining_segments_l682_682414

theorem total_length_remaining_segments (w h : ℕ) (part1 part2 : ℕ) (full_heights partial_widths : list ℕ)
  (rect_dim : w = 10 ∧ h = 5)
  (remaining_heights : full_heights = [h, h])
  (remaining_widths : partial_widths = [part1, part2] ∧ part1 = 3 ∧ part2 = 2) :
  full_heights.sum + partial_widths.sum = 15 := 
by
  sorry

end total_length_remaining_segments_l682_682414


namespace expression_divisible_by_41_l682_682276

theorem expression_divisible_by_41 (n : ℕ) : 41 ∣ (5 * 7^(2*(n+1)) + 2^(3*n)) :=
  sorry

end expression_divisible_by_41_l682_682276


namespace probability_same_color_l682_682210

theorem probability_same_color
  (A_white : ℕ) (A_red : ℕ) (B_white : ℕ) (B_red : ℕ)
  (hA : A_white = 8) (hAa : A_red = 4) (hB : B_white = 6) (hBb : B_red = 6) :
  let total_A := A_white + A_red,
      total_B := B_white + B_red in
  (A_white / total_A * B_white / total_B + A_red / total_A * B_red / total_B) = 1 / 2 := by
  sorry

end probability_same_color_l682_682210


namespace tau_bound_l682_682368

def distinct_points_in_plane (n : ℕ) : Prop := 
  ∃ (points : fin n → (ℝ × ℝ)), function.injective points

def tau (n : ℕ) : ℕ := 
  -- This is a placeholder. In practice, we would define tau(n) based on the distinct points and their segments.
  sorry

theorem tau_bound (n : ℕ) (h : distinct_points_in_plane n): tau(n) ≤ n^2 / 3 := 
  sorry

end tau_bound_l682_682368


namespace four_diff_digits_per_day_l682_682319

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end four_diff_digits_per_day_l682_682319


namespace two_monochromatic_triangles_exists_l682_682071

open Finset

/-- In any 2-coloring of the complete graph with 10 vertices, 
there exist two monochromatic triangles that do not share a common vertex. -/
theorem two_monochromatic_triangles_exists : 
  ∃ (n : ℕ), (n = 10) ∧ (∀ c : (Sym2 (Fin n)) → Bool, ∃ (t₁ t₂ : Finset (Fin n)), 
    t₁.card = 3 ∧ t₂.card = 3 ∧ (∀ (v ∈ t₁), v ∉ t₂) ∧ (∀ (e ∈ t₁.pairs ∪ t₂.pairs), c e = true ∨ c e = false) ∧
    ((∀ (e ∈ t₁.pairs), c e = true) ∨ (∀ (e ∈ t₁.pairs), c e = false)) ∧
    ((∀ (e ∈ t₂.pairs), c e = true) ∨ (∀ (e ∈ t₂.pairs), c e = false))) :=
sorry

end two_monochromatic_triangles_exists_l682_682071


namespace tangent_line_values_l682_682192

theorem tangent_line_values (m : ℝ) :
  (∃ s : ℝ, 3 * s^2 = 12 ∧ 12 * s + m = s^3 - 2) ↔ (m = -18 ∨ m = 14) :=
by
  sorry

end tangent_line_values_l682_682192


namespace smallest_positive_multiple_45_l682_682897

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682897


namespace olympiad_min_problems_l682_682595

theorem olympiad_min_problems (n : ℕ) (h : n = 55) : ∃ (a : ℕ), ((a + 1) * (a + 2) / 2 = n) ∧ a = 9 := by
  use 9
  split
  · calc
      ((9 + 1) * (9 + 2) / 2) = 10 * 11 / 2 := by norm_num
      _ = 55 := by norm_num
  · rfl

end olympiad_min_problems_l682_682595


namespace product_fraction_l682_682051

theorem product_fraction :
  (∏ k in (Finset.range 100).erase 3, (1 - (1 : ℚ) / (k + 2))) = (50 : ℚ) / 101 :=
sorry

end product_fraction_l682_682051


namespace value_of_X_l682_682188

noncomputable def M : ℕ := 2013 / 3
noncomputable def N : ℕ := (M / 3).natCeil
noncomputable def X : ℕ := M + N

theorem value_of_X : X = 895 :=
by
  -- Proof is omitted
  sorry

end value_of_X_l682_682188


namespace sqrt_81_eq_pm_9_abs_sqrt_15_minus_4_l682_682712

theorem sqrt_81_eq_pm_9 : ∃ x, (x = 9 ∨ x = -9) ∧ x^2 = 81 :=
by
  existsi 9
  constructor
  . left; refl
  . norm_num

theorem abs_sqrt_15_minus_4 : |real.sqrt 15 - 4| = 4 - real.sqrt 15 :=
by sorry

end sqrt_81_eq_pm_9_abs_sqrt_15_minus_4_l682_682712


namespace sum_of_digits_of_6_13_l682_682375

noncomputable def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem sum_of_digits_of_6_13 :
  let n := (4 + 2)^13
  in units_digit n + tens_digit n = 13 := 
by
  have h1 : (4 + 2)^13 = 6^13 := by norm_num
  have h2 : units_digit (6^13) = 6 := by { sorry }
  have h3 : tens_digit (6^13) = 7 := by { sorry }
  rw h1 at h2 h3
  have sum_eq : units_digit n + tens_digit n = 6 + 7 := by
    rw [h1, h2, h3]
  have : 6 + 7 = 13 := by norm_num
  exact this

end sum_of_digits_of_6_13_l682_682375


namespace area_of_FGHIJ_l682_682688

-- Definitions corresponding to the given conditions:
noncomputable def pentagon_FGHIJ : Type :=
{F G H I J : ℝ}

noncomputable def pentagon_angles (F G H I J : ℝ) : Prop :=
  (F = 120 ∧ G = 120)

noncomputable def pentagon_sides (JF FG GH HI IJ : ℝ) : Prop :=
  (JF = 3 ∧ FG = 3 ∧ GH = 3 ∧ HI = 5 ∧ IJ = 5)

-- The final proof statement:
theorem area_of_FGHIJ (F G H I J JF FG GH HI IJ : ℝ)
  (h_angles : pentagon_angles F G H I J)
  (h_sides : pentagon_sides JF FG GH HI IJ) :
  let area := (9 * Real.sqrt 3 / 4) + 3 * Real.sqrt 22.75 in
  ∃ area_FGHIJ, area_FGHIJ = area :=
sorry -- Proof to be completed

end area_of_FGHIJ_l682_682688


namespace smallest_positive_multiple_l682_682789

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682789


namespace smallest_positive_multiple_of_45_l682_682968

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682968


namespace find_length_BF_l682_682407

-- Define the conditions
structure Rectangle :=
  (short_side : ℝ)
  (long_side : ℝ)

def folded_paper (rect : Rectangle) : Prop :=
  rect.short_side = 12

def congruent_triangles (rect : Rectangle) : Prop :=
  rect.short_side = 12

-- Define the length of BF to prove
def length_BF (rect : Rectangle) : ℝ := 10

-- The theorem statement
theorem find_length_BF (rect : Rectangle) (h1 : folded_paper rect) (h2 : congruent_triangles rect) :
  length_BF rect = 10 := 
  sorry

end find_length_BF_l682_682407


namespace negation_of_exists_l682_682333

theorem negation_of_exists (x : ℝ) :
  ¬ (∃ x > 0, 2 * x + 3 ≤ 0) ↔ ∀ x > 0, 2 * x + 3 > 0 :=
by
  sorry

end negation_of_exists_l682_682333


namespace smallest_positive_multiple_of_45_l682_682960

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682960


namespace number_of_divisors_of_36_l682_682561

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end number_of_divisors_of_36_l682_682561


namespace assign_numbers_to_points_l682_682136

theorem assign_numbers_to_points 
(white_points black_points : Type)
(arrows : white_points → black_points → ℕ)
(h : ∀ (path : list (white_points ⊕ black_points)), 
    (∀ (i : ℕ) (hi : i < path.length - 1), 
      path.nth_le i hi ≠ path.nth_le (i + 1) (by simp [hi])) → 
    (∀ (i j : ℕ) (hi : i < path.length) (hj : j < path.length), 
      ((path.nth_le i hi).elim (λ w, ∀ (b : black_points), arrows w b = (arrows w b) * arrows w b) 
                           (λ b, ∀ (w : white_points), arrows w b = (arrows w b) * arrows w b)) = 
      ((path.nth_le j hj).elim (λ w, ∀ (b : black_points), arrows w b = (arrows w b) * arrows w b) 
                           (λ b, ∀ (w : white_points), arrows w b = (arrows w b) * arrows w b)) → 
    ∏ (i : ℕ) (hi : i < path.length - 1), 
      if h : path.nth_le i hi < path.nth_le (i + 1) (by simp [hi]) 
      then arrows (path.nth_le i hi).elim (λ w, w) sorry 
      else arrows (path.nth_le (i + 1) (by simp [hi])).elim (λ w, w) sorry = 1) →
∃ (numbers : white_points ⊕ black_points → ℕ), 
  ∀ w b, arrows w b = numbers (sum.inl w) * numbers (sum.inr b) :=
sorry

end assign_numbers_to_points_l682_682136


namespace smallest_positive_multiple_of_45_is_45_l682_682919

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682919


namespace smallest_positive_multiple_45_l682_682906

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682906


namespace axis_of_symmetry_vertex_on_x_axis_range_of_m_l682_682544

noncomputable def parabola : ℝ → ℝ → ℝ :=
λ a x, a * x^2 + 2 * a * x - 1

theorem axis_of_symmetry (a : ℝ) :
    (∃ x y, y = parabola a x) → (∃ h, ∀ x, h = -1) :=
sorry

theorem vertex_on_x_axis :
    (∃ a, ∀ x, x = -1 → a + 1 = 0 → parabola (-1) x = -(x^2) - 2 * x - 1) :=
sorry

theorem range_of_m (a : ℝ) (m : ℝ) (y₁ y₂ : ℝ)
    (h₁: y₁ > y₂) :
    (a = -1) →
    (¬((-4 < m) ∧ (m < 2))) →
    (parabola (-1) m = y₁) ∧ (parabola (-1) 2 = y₂) :=
sorry

end axis_of_symmetry_vertex_on_x_axis_range_of_m_l682_682544


namespace lattice_points_on_sphere_l682_682206

-- Definitions
def is_lattice_point_on_sphere (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 9

-- The main statement
theorem lattice_points_on_sphere : 
  { p : ℤ × ℤ × ℤ // is_lattice_point_on_sphere p.1 p.2.1 p.2.2 }.card = 30 :=
by sorry

end lattice_points_on_sphere_l682_682206


namespace max_n_no_constant_term_l682_682158

theorem max_n_no_constant_term (n : ℕ) (h : n < 10 ∧ n ≠ 3 ∧ n ≠ 6 ∧ n ≠ 9 ∧ n ≠ 2 ∧ n ≠ 5 ∧ n ≠ 8): n ≤ 7 :=
by {
  sorry
}

end max_n_no_constant_term_l682_682158


namespace homothety_angle_theorem_l682_682721

-- Define the circles, their intersection points, and the center of homothety
variables (ω1 ω2 : Circle) (A B O : Point)
variables (C D : Point) (E F : Point)
variables (secant : Line) (α : real)

-- Define conditions
def conditions := (ω1 ∩ ω2 = {A, B}) ∧ 
                  (secant ∩ ω1 = {C, D}) ∧
                  (secant ∩ ω2 = {E, F}) ∧ 
                  (O ∈ secant) ∧ 
                  (homothety_center ω1 ω2 O)

-- Define the theorem statement
theorem homothety_angle_theorem 
(ω1 ω2 : Circle) (A B O : Point) 
(C D E F : Point) (secant : Line) (α : real)
(h : conditions ω1 ω2 A B O C D E F secant α) : 
  angle_seen_from A (segment C E) = α ∨ 
  angle_seen_from A (segment C E) = 180 - α :=
sorry

end homothety_angle_theorem_l682_682721


namespace solve_equation_l682_682675

theorem solve_equation (t : ℝ) :
  (∃ t, (sqrt (3 * sqrt (3 * t - 6)) = real.rpow (8 - t) (1 / 4))) ↔
  (t = (-43 + real.sqrt 2321) / 2 ∨ t = (-43 - real.sqrt 2321) / 2) :=
by
  sorry

end solve_equation_l682_682675


namespace divisors_of_36_l682_682571

theorem divisors_of_36 : ∀ n : ℕ, n = 36 → (∃ k : ℕ, k = 9) :=
by
  intro n hn
  have h_prime_factors : (n = 2^2 * 3^2) := by rw hn; norm_num
  -- Using the formula for the number of divisors based on prime factorization
  have h_num_divisors : (2 + 1) * (2 + 1) = 9 := by norm_num
  use 9
  rw h_num_divisors
  sorry

end divisors_of_36_l682_682571


namespace intersection_points_on_same_circle_or_line_l682_682129

noncomputable def circle (P Q R : Point) : Prop :=
∃ (O : Point) (r : ℝ), ∀ (X : Point), dist O X = r ↔ X = P ∨ X = Q ∨ X = R

def lie_on_same_circle_or_line (A B C D : Point) : Prop :=
(∃ O r, circle A B C ∧ circle A B D ∧ circle A C D ∧ circle B C D) ∨
collinear A B C ∨ collinear A B D ∨ collinear A C D ∨ collinear B C D

variables (S₁ S₂ S₃ S₄ : circle) 
variables (P Q R S T U V W : Point)
variables (h₁ : intersects S₁ S₂ P Q)
variables (h₂ : intersects S₁ S₄ R S)
variables (h₃ : intersects S₃ S₂ T U)
variables (h₄ : intersects S₃ S₄ V W)
variables (h₅ : lie_on_same_circle_or_line P Q V W)

theorem intersection_points_on_same_circle_or_line :
  lie_on_same_circle_or_line R S T U :=
sorry

end intersection_points_on_same_circle_or_line_l682_682129


namespace math_expression_equivalent_l682_682022

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end math_expression_equivalent_l682_682022


namespace intersection_points_l682_682590

noncomputable def num_intersections : ℕ := 2

-- Define the line equation
def line (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the parametric equations of the circle
def circle_parametric (θ : ℝ) : ℝ × ℝ :=
  (-1 + 2 * Real.cos θ, 2 + 2 * Real.sin θ)

-- Define the circle in standard form based on the parametric equations
def circle_standard (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = 4

-- Prove that the number of intersection points is 2
theorem intersection_points : ∀ (l : ℝ → ℝ → Prop) (c : ℝ × ℝ → Prop),
  (l = line) → (c = λ p, circle_standard p.1 p.2) → 
  ∃ n : ℕ, n = 2 :=
by
  assume (l : ℝ → ℝ → Prop) (c : ℝ × ℝ → Prop) (hl : l = line) (hc : c = λ p, circle_standard p.1 p.2)
  exact ⟨num_intersections, rfl⟩

#eval intersection_points line (λ p, circle_standard p.1 p.2) rfl rfl -- Expected output: 2

end intersection_points_l682_682590


namespace solve_equation_l682_682350

theorem solve_equation : ∃ x : ℝ, 2 * x - 3 = 5 ∧ x = 4 := 
by
  -- Introducing x as a real number and stating the goal
  use 4
  -- Show that 2 * 4 - 3 = 5
  simp
  -- Adding the sorry to skip the proof step
  sorry

end solve_equation_l682_682350


namespace tim_prank_combinations_l682_682364

def number_of_combinations (monday_choices : ℕ) (tuesday_choices : ℕ) (wednesday_choices : ℕ) (thursday_choices : ℕ) (friday_choices : ℕ) : ℕ :=
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

theorem tim_prank_combinations : number_of_combinations 2 3 0 6 1 = 0 :=
by
  -- Calculation yields 2 * 3 * 0 * 6 * 1 = 0
  sorry

end tim_prank_combinations_l682_682364


namespace smallest_positive_multiple_of_45_l682_682885

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682885


namespace smallest_positive_multiple_of_45_l682_682848

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682848


namespace smallest_positive_multiple_of_45_l682_682847

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682847


namespace find_eccentricity_l682_682147

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : ℝ :=
  let c := real.sqrt (a^2 + b^2);
  if h1 : a > 0 ∧ b > 0 ∧ P = (a, b) ∧ (|P.1|, |P.2|) = (|F1.1|, |F1.2|) ∧ |P.1| = 2 * |F2.1| 
  then real.sqrt 5
  else 0

theorem find_eccentricity
  (a b : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (h4 : P.1^2 + P.2^2 = a^2 + b^2)
  (h5: |P.1 - F1.1| = 2 * |P.1 - F2.1|) :
  eccentricity_of_hyperbola a b P F1 F2 = real.sqrt 5 :=
sorry

end find_eccentricity_l682_682147


namespace smallest_x_value_l682_682285

theorem smallest_x_value (x : ℝ) (h : 3 * (8 * x^2 + 10 * x + 12) = x * (8 * x - 36)) : x = -3 :=
sorry

end smallest_x_value_l682_682285


namespace find_angle_x_l682_682371

theorem find_angle_x (A B C : Type) (x : ℝ) (angleA angleB angleC : ℝ) 
(h_triangle : Angle(angleA) + Angle(angleB) + Angle(angleC) = 180) 
(h_angleA : angleA = x) 
(h_angleB : angleB = 2 * x) 
(h_angleC : angleC = 40) : x = 140 / 3 := 
by
  sorry

end find_angle_x_l682_682371


namespace inverse_of_congruent_triangles_areas_l682_682381

theorem inverse_of_congruent_triangles_areas (A B : Triangle) :
  (congruent A B → area A = area B) ↔ (area A = area B → congruent A B) :=
by
  sorry

end inverse_of_congruent_triangles_areas_l682_682381


namespace solve_for_x_l682_682467

theorem solve_for_x (x : ℚ) (h : 5 * (x - 6) = 3 * (3 - 3 * x) + 9) : x = 24 / 7 :=
sorry

end solve_for_x_l682_682467


namespace smallest_positive_multiple_45_l682_682902

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682902


namespace pets_remaining_is_correct_l682_682436

-- Definitions for the initial conditions and actions taken
def initial_puppies : Nat := 7
def initial_kittens : Nat := 6
def puppies_sold : Nat := 2
def kittens_sold : Nat := 3

-- Definition that calculates the remaining number of pets
def remaining_pets : Nat := initial_puppies + initial_kittens - (puppies_sold + kittens_sold)

-- The theorem to prove
theorem pets_remaining_is_correct : remaining_pets = 8 := by sorry

end pets_remaining_is_correct_l682_682436


namespace total_cost_of_typing_l682_682387

noncomputable def calculate_total_cost (initial_rate: ℝ) (revision_rate: ℝ)
  (total_pages: ℕ) (pages_revised_once: ℕ) (pages_revised_twice: ℕ): ℝ :=
  let pages_no_revision := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost := total_pages * initial_rate
  let revision_cost := (pages_revised_once * revision_rate) +
                       (pages_revised_twice * revision_rate * 2)
  initial_typing_cost + revision_cost

theorem total_cost_of_typing (initial_rate: ℝ) (revision_rate: ℝ)
  (total_pages: ℕ) (pages_revised_once: ℕ) (pages_revised_twice: ℕ)
  (h1: initial_rate = 5) (h2: revision_rate = 4)
  (h3: total_pages = 100) (h4: pages_revised_once = 30)
  (h5: pages_revised_twice = 20):
  calculate_total_cost initial_rate revision_rate total_pages pages_revised_once pages_revised_twice = 780 := by
  have pages_no_revision := total_pages - pages_revised_once - pages_revised_twice
  have initial_typing_cost := total_pages * initial_rate
  have revision_cost := (pages_revised_once * revision_rate) + (pages_revised_twice * revision_rate * 2)
  have total_cost := initial_typing_cost + revision_cost
  rw [h1, h2, h3, h4, h5]
  have h_calc: total_cost = 100 * 5 + (30 * 4 + 20 * 4 * 2) := by norm_num
  rw [h_calc]
  norm_num
  sorry

end total_cost_of_typing_l682_682387


namespace find_parallel_line_l682_682697

theorem find_parallel_line (h_parallel : ∀ x y, 2 * x + 3 * y + c = 0) (h_through_origin : ∀ x y, 2 * 0 + 3 * 0 = 0) :
  c = 0 :=
sorry

end find_parallel_line_l682_682697


namespace probability_N_lt_L_is_zero_l682_682083

variable (M N L O : ℝ)
variable (hNM : N < M)
variable (hLO : L > O)

theorem probability_N_lt_L_is_zero : 
  (∃ (permutations : List (ℝ → ℝ)), 
  (∀ perm : ℝ → ℝ, perm ∈ permutations → N < M ∧ L > O) ∧ 
  ∀ perm : ℝ → ℝ, N > L) → false :=
by {
  sorry
}

end probability_N_lt_L_is_zero_l682_682083


namespace conclusion_A_conclusion_B_conclusion_C_conclusion_D_l682_682112

noncomputable theory

open Real

section
-- Define the functions and mathematical assumptions
variable (ω : ℝ) (ϕ : ℝ)
def f (x : ℝ) : ℝ := sin (ω * x + π / 3 + ϕ)
def g (x : ℝ) : ℝ := sin (ω * x + ϕ)

-- Assumptions
axiom h1 : ω > 0
axiom h2 : abs ϕ < π / 2
axiom h3 : ∀ x : ℝ, f x = f (-x)  -- f(x) is even

-- Proofs of the conclusions
theorem conclusion_A : ϕ = π / 6 :=
sorry

theorem conclusion_B (h_smallest_period : ∃ T > 0, ∀ x : ℝ, g (x + T) = g x ∧ T = 3 * π) : ω = 2 / 3 :=
sorry

theorem conclusion_C (h_extreme_points : ∀ t ∈ Ioo (0 : ℝ) π, (x | g x = sin t) → |g' x| = 1) : 7 / 3 < ω ∧ ω ≤ 10 / 3 :=
sorry

theorem conclusion_D (h_value_at_pi_4 : g (π / 4) = sqrt 3 / 2) : min ω (2 / 3) :=
sorry
end

end conclusion_A_conclusion_B_conclusion_C_conclusion_D_l682_682112


namespace smallest_positive_multiple_of_45_l682_682769

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682769


namespace pets_remaining_l682_682435

-- Definitions based on conditions
def initial_puppies : ℕ := 7
def initial_kittens : ℕ := 6
def sold_puppies : ℕ := 2
def sold_kittens : ℕ := 3

-- Theorem statement
theorem pets_remaining : initial_puppies + initial_kittens - (sold_puppies + sold_kittens) = 8 :=
by
  sorry

end pets_remaining_l682_682435


namespace smallest_positive_multiple_of_45_l682_682868

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682868


namespace complement_intersection_l682_682509

def A := {x : ℝ | x > -1}
def B := {-2, -1, 0, 1} : Set ℝ

theorem complement_intersection :
  (Set.uȼompl A) ∩ B = {-2, -1} :=
by
  rw [Set.compl_set_of]
  sorry

end complement_intersection_l682_682509


namespace smallest_positive_multiple_of_45_l682_682840

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682840


namespace triangle_determines_plane_l682_682687

-- Define the problem as a proposition
def determines_plane (object : Type) : Prop := sorry

-- Define the conditions as a data type
inductive Condition
| two_lines : Condition
| point_and_line : Condition
| triangle : Condition
| three_points : Condition

-- Proposition stating that a triangle determines a plane
theorem triangle_determines_plane : determines_plane Condition.triangle := 
begin
  sorry
end

end triangle_determines_plane_l682_682687


namespace money_distribution_l682_682385

theorem money_distribution (a : ℕ) (h1 : 5 * a = 1500) : 7 * a - 3 * a = 1200 := by
  sorry

end money_distribution_l682_682385


namespace thief_run_distance_l682_682411

noncomputable def speed_thief_km_per_hr : ℝ := 8
noncomputable def speed_policeman_km_per_hr : ℝ := 10
noncomputable def initial_distance_m : ℝ := 150

noncomputable def km_per_hr_to_m_per_s (v: ℝ) : ℝ := v * 1000 / 3600

noncomputable def speed_thief_m_per_s : ℝ := km_per_hr_to_m_per_s speed_thief_km_per_hr
noncomputable def speed_policeman_m_per_s : ℝ := km_per_hr_to_m_per_s speed_policeman_km_per_hr

noncomputable def relative_speed_m_per_s : ℝ := speed_policeman_m_per_s - speed_thief_m_per_s
noncomputable def time_seconds : ℝ := initial_distance_m / relative_speed_m_per_s

noncomputable def distance_thief_run : ℝ := speed_thief_m_per_s * time_seconds

theorem thief_run_distance (h : distance_thief_run ≈ 594.64) : distance_thief_run = 594.64 := 
sorry

end thief_run_distance_l682_682411


namespace remainder_777_777_mod_13_l682_682373

theorem remainder_777_777_mod_13 : (777 ^ 777) % 13 = 12 := 
by 
  -- Proof steps would go here
  sorry

end remainder_777_777_mod_13_l682_682373


namespace eval_x2_sub_y2_l682_682143

theorem eval_x2_sub_y2 (x y : ℝ) (h1 : x + y = 10) (h2 : 2 * x + y = 13) : x^2 - y^2 = -40 := by
  sorry

end eval_x2_sub_y2_l682_682143


namespace correct_method_eliminates_y_l682_682175

def eliminate_y_condition1 (x y : ℝ) : Prop :=
  5 * x + 2 * y = 20

def eliminate_y_condition2 (x y : ℝ) : Prop :=
  4 * x - y = 8

theorem correct_method_eliminates_y (x y : ℝ) :
  eliminate_y_condition1 x y ∧ eliminate_y_condition2 x y →
  5 * x + 2 * y + 2 * (4 * x - y) = 36 :=
by
  sorry

end correct_method_eliminates_y_l682_682175


namespace five_times_x_plus_four_l682_682186

theorem five_times_x_plus_four (x : ℝ) (h : 4 * x - 3 = 13 * x + 12) : 5 * (x + 4) = 35 / 3 := 
by
  sorry

end five_times_x_plus_four_l682_682186


namespace squares_cover_unit_square_l682_682404

theorem squares_cover_unit_square (N : ℕ) (a : fin N → ℝ) 
    (h_sum : ∑ i, (a i) ^ 2 = 4) : 
    ∃ S : set (fin N), 
      (sum (λ i, (a i) * (a i)) S  = 4) ∧
      (S ⊆ univ) ∧
      (⋃ i ∈ S, closed_ball (0 : ℝ × ℝ) (a i / 2) = closed_ball (0 : ℝ × ℝ) 0.5) :=
sorry

end squares_cover_unit_square_l682_682404


namespace smallest_positive_multiple_of_45_l682_682775

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l682_682775


namespace shifted_parabola_eq_l682_682336

theorem shifted_parabola_eq :
  ∀ x, (∃ y, y = 2 * (x - 3)^2 + 2) →
       (∃ y, y = 2 * (x + 0)^2 + 4) :=
by sorry

end shifted_parabola_eq_l682_682336


namespace smallest_positive_multiple_of_45_l682_682890

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682890


namespace smallest_b_for_perfect_square_l682_682742

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ (∃ (n : ℤ), 4 * b + 5 = n ^ 2) ∧ b = 5 := 
sorry

end smallest_b_for_perfect_square_l682_682742


namespace same_face_color_possible_l682_682360

-- Conditions given in the problem
structure Cube :=
(colors : fin 6 → color)

structure Rectangle :=
(cubes : matrix (fin m) (fin n) Cube)

-- Allowed operations: row and column rotations
def rotate_row (r : Rectangle) (i : fin m) : Rectangle := sorry
def rotate_column (r : Rectangle) (i : fin n) : Rectangle := sorry

-- Proof goal
theorem same_face_color_possible (r : Rectangle) : ∃ face_color : color,
  ∀ i j, (rotate_row (rotate_column r i) j).cubes[i][j].colors 0 = face_color :=
sorry

end same_face_color_possible_l682_682360


namespace smallest_positive_multiple_of_45_l682_682746

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682746


namespace problem_statements_correctness_l682_682042

theorem problem_statements_correctness :
  (3 ∣ 15) ∧ (11 ∣ 121 ∧ ¬(11 ∣ 60)) ∧ (12 ∣ 72 ∧ 12 ∣ 120) ∧ (7 ∣ 49 ∧ 7 ∣ 84) ∧ (7 ∣ 63) → 
  (3 ∣ 15) ∧ (11 ∣ 121 ∧ ¬(11 ∣ 60)) ∧ (7 ∣ 63) :=
by
  intro h
  sorry

end problem_statements_correctness_l682_682042


namespace smallest_positive_multiple_45_l682_682820

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682820


namespace count_true_statements_l682_682393

open Set

variable {M P : Set α}

theorem count_true_statements (h : ¬ ∀ x ∈ M, x ∈ P) (hne : Nonempty M) :
  (¬ ∃ x, x ∈ M ∧ x ∈ P ∨ ∀ x, x ∈ M → x ∈ P) ∧ (∃ x, x ∈ M ∧ x ∉ P) ∧ 
  ¬ (∃ x, x ∈ M ∧ x ∈ P) ∧ (¬ ∀ x, x ∈ M → x ∈ P) :=
sorry

end count_true_statements_l682_682393


namespace divisors_of_36_l682_682566

theorem divisors_of_36 : ∀ n : ℕ, n = 36 → (∃ k : ℕ, k = 9) :=
by
  intro n hn
  have h_prime_factors : (n = 2^2 * 3^2) := by rw hn; norm_num
  -- Using the formula for the number of divisors based on prime factorization
  have h_num_divisors : (2 + 1) * (2 + 1) = 9 := by norm_num
  use 9
  rw h_num_divisors
  sorry

end divisors_of_36_l682_682566


namespace balls_in_boxes_l682_682575

theorem balls_in_boxes : 
  ∃ (n : ℕ), n = 5 ∧ 
    (∀ (partitions : list (ℕ)), (partitions.sum = 5) → 
      (partitions.length ≤ 3) → 
      (∀ (elem : ℕ), elem ∈ partitions → elem.nonneg)) :=
begin
  sorry
end

end balls_in_boxes_l682_682575


namespace pats_password_length_l682_682645

-- Definitions based on conditions
def num_lowercase_letters := 8
def num_uppercase_numbers := num_lowercase_letters / 2
def num_symbols := 2

-- Translate the math proof problem to Lean 4 statement
theorem pats_password_length : 
  num_lowercase_letters + num_uppercase_numbers + num_symbols = 14 := by
  sorry

end pats_password_length_l682_682645


namespace part_I_part_II_l682_682121

open BigOperators

-- Define the sequence a_n with the given condition
def a (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / 4^n

-- Define the general formula condition
def general_formula_condition (n : ℕ) : Prop :=
∑ i in Finset.range n, (4 ^ i) * a (i + 1) = n / 4

-- Define the sequence b_n
def b (n : ℕ) : ℝ :=
2^n * Real.log (a n) / Real.log 4

-- Define the sum of the first n terms of sequence b_n
def T (n : ℕ) :=
∑ i in Finset.range n, b (i + 1)

theorem part_I (n : ℕ) (hn: n ≠ 0) : a n = 1 / 4^n :=
sorry

theorem part_II (n : ℕ) (hn: n ≠ 0) : 
  T n = (1 - n : ℝ) * 2^(n + 1) - 2 :=
sorry

end part_I_part_II_l682_682121


namespace exists_infinitely_many_m_consecutive_squares_sum_m3_l682_682652

theorem exists_infinitely_many_m_consecutive_squares_sum_m3 :
  ∃ᶠ m in at_top, ∃ a, (∑ k in finset.range m, (a + k + 1)^2 = m^3) :=
sorry

end exists_infinitely_many_m_consecutive_squares_sum_m3_l682_682652


namespace calculate_expression_l682_682019

noncomputable def solve_expression : ℝ :=
  let term1 := (real.pi - 1) ^ 0
  let term2 := 4 * real.sin (real.pi / 4) -- sin 45° = sin (π/4)
  let term3 := real.sqrt 8
  let term4 := real.abs (-3)
  term1 + term2 - term3 + term4

theorem calculate_expression : solve_expression = 4 := by
  sorry

end calculate_expression_l682_682019


namespace area_of_triangle_l682_682542

open Real

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

noncomputable def F1 : ℝ × ℝ := (-5, 0)
noncomputable def F2 : ℝ × ℝ := (5, 0)

theorem area_of_triangle (P : ℝ × ℝ)
  (hP : hyperbola P.1 P.2)
  (hAngle : ∠ (F1) P (F2) = π / 2) :
  (1 / 2) * dist P F1 * dist P F2 = 16 := sorry

end area_of_triangle_l682_682542


namespace proof_value_g_expression_l682_682289

noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom g_invertible : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x
axiom g_table : ∀ x, (x = 1 → g x = 4) ∧ (x = 2 → g x = 5) ∧ (x = 3 → g x = 7) ∧ (x = 4 → g x = 9) ∧ (x = 5 → g x = 10)

theorem proof_value_g_expression :
  g (g 2) + g (g_inv 9) + g_inv (g_inv 7) = 21 :=
by
  sorry

end proof_value_g_expression_l682_682289


namespace problem_statement_l682_682002

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end problem_statement_l682_682002


namespace total_number_of_prime_factors_l682_682487

-- Conditions
noncomputable def four_to_13 := (4 : ℕ)^13
noncomputable def seven_to_5 := (7 : ℕ)^5
noncomputable def eleven_to_2 := (11 : ℕ)^2

-- Expression under consideration
noncomputable def expression := four_to_13 * seven_to_5 * eleven_to_2

-- Target
def total_prime_factors : ℕ := 26 + 5 + 2

-- Theorem stating the total number of prime factors
theorem total_number_of_prime_factors : 
  (total_prime_factors = 33) := 
begin
  sorry -- Proof omitted
end

end total_number_of_prime_factors_l682_682487


namespace intersection_distance_eq_sqrt_neg2_plus_2_sqrt5_l682_682691

theorem intersection_distance_eq_sqrt_neg2_plus_2_sqrt5 :
  let y1 := sqrt ((sqrt 5 - 1) / 2),
      y2 := -sqrt ((sqrt 5 - 1) / 2),
      x := (3 - sqrt 5) / 2,
      pt1 := (x, y1),
      pt2 := (x, y2),
      distance := sqrt ((pt1.1 - pt2.1)^2 + (pt1.2 - pt2.2)^2)
  in distance = sqrt (-2 + 2 * sqrt 5) :=
by
  sorry

end intersection_distance_eq_sqrt_neg2_plus_2_sqrt5_l682_682691


namespace period_and_increasing_interval_cos_2x0_of_zero_point_l682_682538

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 
  Real.sin (x + Real.pi / 4) * Real.sin (x - Real.pi / 4)

def period_of_f := Real.pi

def increasing_interval (k : ℤ) := 
  (-Real.pi / 6 + k * Real.pi, Real.pi / 3 + k * Real.pi)

theorem period_and_increasing_interval : 
  ((∀ x, f (x + period_of_f) = f x) ∧ 
  (∀ k : ℤ, ∃ a b : ℝ, a = -Real.pi / 6 + k * Real.pi ∧ b = Real.pi / 3 + k * Real.pi ∧ 
  ∀ x, (a ≤ x ∧ x ≤ b → 
  (0 ≤ x ≤ b - a) → (f x ≤ f (x + x / (b - a)))))) :=
sorry

theorem cos_2x0_of_zero_point (x₀ : ℝ) (h₀ : 0 ≤ x₀ ∧ x₀ ≤ Real.pi / 2) (hx₀ : f x₀ = 0) :
  Real.cos (2 * x₀) = (3 * Real.sqrt 5 + 1) / 8 :=
sorry

end period_and_increasing_interval_cos_2x0_of_zero_point_l682_682538


namespace smallest_positive_multiple_of_45_l682_682829

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682829


namespace find_k_l682_682330

theorem find_k {k : ℚ} :
    (∃ x y : ℚ, y = 3 * x + 6 ∧ y = -4 * x - 20 ∧ y = 2 * x + k) →
    k = 16 / 7 := 
  sorry

end find_k_l682_682330


namespace tenisha_puppies_proof_l682_682679

def tenisha_remains_with_puppies (total_dogs : ℕ) (percent_female : ℚ) (fraction_giving_birth : ℚ) (puppies_per_dog : ℕ) (donated_puppies : ℕ) : ℕ :=
  let female_dogs := percent_female * total_dogs
  let female_giving_birth := fraction_giving_birth * female_dogs
  let total_puppies := female_giving_birth * puppies_per_dog
  total_puppies - donated_puppies

theorem tenisha_puppies_proof :
  tenisha_remains_with_puppies 40 0.60 0.75 10 130 = 50 :=
by
  sorry

end tenisha_puppies_proof_l682_682679


namespace A_n1_gt_B_n_l682_682497

noncomputable def A : ℕ → ℕ
| 0     := 3
| (n+1) := 3 ^ (A n)

noncomputable def B : ℕ → ℕ
| 0     := 8
| (n+1) := 8 ^ (B n)

theorem A_n1_gt_B_n (n : ℕ) : A (n + 1) > B n :=
sorry

end A_n1_gt_B_n_l682_682497


namespace cost_of_double_scoop_l682_682293

theorem cost_of_double_scoop
  (price_kiddie_scoop : ℕ := 3)
  (price_regular_scoop : ℕ := 4)
  (total_cost : ℕ := 32) :
  (∃ (D : ℕ), 2 * price_regular_scoop + 2 * price_kiddie_scoop + 3 * D = total_cost) →
  ∃ (D : ℕ), D = 6 :=
by
  intro h
  obtain ⟨D, h_eq⟩ := h
  have : 2 * 4 + 2 * 3 + 3 * D = 32 := h_eq
  simp at this
  linarith
  sorry

end cost_of_double_scoop_l682_682293


namespace sunny_ahead_second_race_l682_682204

variables {h d s w : ℝ}
-- Conditions
def distance := h^2 / 100
def lead := 2 * d
def start_gap := 2 * d
def sunny_speed := s
def windy_speed := w

-- Given Sunny's lead in the first race
axiom sunny_windy_speed_ratio : s / w = (h^2 / 100) / (h^2 / 100 - 2 * d)

-- Conclusion
theorem sunny_ahead_second_race 
  (distance_pos : distance > 0)
  (lead_nonneg : lead >= 0)
  (speed_nonneg : s > 0 ∧ w > 0)
  : Sunny finishes \(start\_gap + \frac{400 d^2}{h^2}\) meters ahead:
  (w * (h^2 / 100 + 2 * d) / s) = (sunny_start * (h^2/100 - 2d)^(h^2/100)):
  sorry

end sunny_ahead_second_race_l682_682204


namespace complex_arithmetic_l682_682141

noncomputable def imaginary_unit := Complex.i

theorem complex_arithmetic : 
  Complex.div (Complex.i) (Complex.ofReal 3 + Complex.i) = 1 / 10 + 3 / 10 * Complex.i := 
by 
  sorry

end complex_arithmetic_l682_682141


namespace arithmetic_sequence_proof_l682_682124

-- Define the basic conditions and the sequence
def sum_five_terms (a₁ d : ℝ) : Prop :=
  5 * a₁ + 10 * d = 55

def sixth_seventh_sum (a₁ d : ℝ) : Prop :=
  2 * a₁ + 11 * d = 36

-- Expressing the general term based on the conditions
def general_term_formula (a₁ d : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = 2 * n + 5

-- Define the b_n sequence
def b_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 1 / ((a n - 6) * (a n - 4))

-- State the sum of first n terms of b_n sequence
def sum_b_terms (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in Finset.range n, b (i + 1)

-- The target proof problem
theorem arithmetic_sequence_proof :
  ∃ a₁ d : ℝ, sum_five_terms a₁ d ∧ sixth_seventh_sum a₁ d ∧ 
    (∀ a : ℕ → ℝ, general_term_formula a₁ d a → 
      ∀ b : ℕ → ℝ, b_sequence a b → 
        ∀ S : ℕ → ℝ, sum_b_terms b S → 
          ∀ n, S n = n / (2 * n + 1)) :=
by
  -- Placeholder for the proof
  sorry

end arithmetic_sequence_proof_l682_682124


namespace sin_2alpha_eq_neg_one_l682_682107

theorem sin_2alpha_eq_neg_one (α : ℝ) (h : sin α - cos α = sqrt 2) : sin (2 * α) = -1 := by
  sorry

end sin_2alpha_eq_neg_one_l682_682107


namespace power_of_half_decreasing_l682_682130

variable (a b : ℝ) (h_a_ne_zero : a ≠ 0) (h_b_ne_zero : b ≠ 0) (h_ineq : a > b)

theorem power_of_half_decreasing : (1 / 2) ^ a < (1 / 2) ^ b := by
  sorry

end power_of_half_decreasing_l682_682130


namespace smallest_positive_multiple_of_45_is_45_l682_682943

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682943


namespace geometric_sequence_solve_a1_l682_682151

noncomputable def geometric_sequence_a1 (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < q)
    (h2 : a 2 = 1) (h3 : a 3 * a 9 = 2 * (a 5 ^ 2)) :=
  a 1 = (Real.sqrt 2) / 2

-- Define the main statement
theorem geometric_sequence_solve_a1 (a : ℕ → ℝ) (q : ℝ)
    (hq : 0 < q) (ha2 : a 2 = 1) (ha3_ha9 : a 3 * a 9 = 2 * (a 5 ^ 2)) :
    a 1 = (Real.sqrt 2) / 2 :=
sorry  -- The proof will be written here

end geometric_sequence_solve_a1_l682_682151


namespace smallest_positive_multiple_l682_682788

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682788


namespace smallest_positive_multiple_of_45_l682_682883

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682883


namespace smallest_positive_multiple_of_45_l682_682933

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682933


namespace total_pencils_l682_682718

-- Variables representing the number of pencils in different locations
variables (x y z w : ℝ)

-- Conditions given in the problem
def pencils_in_drawer := 43.5
def pencils_originally_on_desk := 19.25
def pencils_in_pencil_case := 8.75
def pencils_added_by_dan := 16

-- Equation representing the total number of pencils
theorem total_pencils : 
  x = pencils_in_drawer → 
  y = pencils_originally_on_desk → 
  z = pencils_in_pencil_case → 
  w = pencils_added_by_dan → 
  x + y + w + z = 87.5 :=
by
  intros hx hy hz hw
  simp [hx, hy, hz, hw, pencils_in_drawer, pencils_originally_on_desk, pencils_in_pencil_case, pencils_added_by_dan]
  sorry

end total_pencils_l682_682718


namespace similar_triangles_parallelogram_l682_682235

theorem similar_triangles_parallelogram {O F T N A : Type} [InnerProductSpace ℝ O] [InnerProductSpace ℝ F] [InnerProductSpace ℝ T] [InnerProductSpace ℝ N] [InnerProductSpace ℝ A]
  (h_similar : similar O F T N O T) 
  (h_parallelogram : Parallelogram F A N O) :
  ∥O - F∥ * ∥O - N∥ = ∥O - A∥ * ∥O - T∥ :=
by
  sorry

end similar_triangles_parallelogram_l682_682235


namespace domain_of_f_l682_682297

-- Define the function.
def f (x : ℝ) : ℝ := (1 / (1 - x)) + log (x^2 + 1)

-- Main theorem statement.
theorem domain_of_f :
  {x : ℝ | 1 - x ≠ 0 ∧ x^2 + 1 > 0} = {x : ℝ | x ∉ {1}} :=
by
  sorry

end domain_of_f_l682_682297


namespace a_3_eq_5_l682_682122

-- Define the sequence recursively as per the conditions.
def sequence : ℕ → ℤ
| 1     := 1
| (n+1) := sequence n + 2

-- State the theorem.
theorem a_3_eq_5 : sequence 3 = 5 := 
by
  -- proof will go here
  sorry

end a_3_eq_5_l682_682122


namespace fewest_erasers_l682_682232

theorem fewest_erasers :
  ∀ (JK JM SJ : ℕ), 
  (JK = 6) →
  (JM = JK + 4) →
  (SJ = JM - 3) →
  (JK ≤ JM ∧ JK ≤ SJ) :=
by
  intros JK JM SJ hJK hJM hSJ
  sorry

end fewest_erasers_l682_682232


namespace smallest_positive_multiple_of_45_is_45_l682_682911

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682911


namespace probability_nina_taller_than_lena_is_zero_l682_682089

-- Definition of participants and conditions
variable (M N L O : ℝ)

-- Conditions
def condition1 := N < M
def condition2 := L > O

-- Statement: Given conditions, the probability that N > L is 0
theorem probability_nina_taller_than_lena_is_zero
  (h1 : condition1)
  (h2 : condition2) :
  (P : ℝ) = 0 :=
by
  sorry

end probability_nina_taller_than_lena_is_zero_l682_682089


namespace smallest_positive_multiple_of_45_l682_682855

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682855


namespace number_of_divisors_of_36_l682_682562

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end number_of_divisors_of_36_l682_682562


namespace smallest_positive_multiple_of_45_l682_682758

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682758


namespace divisors_of_36_l682_682569

theorem divisors_of_36 : ∀ n : ℕ, n = 36 → (∃ k : ℕ, k = 9) :=
by
  intro n hn
  have h_prime_factors : (n = 2^2 * 3^2) := by rw hn; norm_num
  -- Using the formula for the number of divisors based on prime factorization
  have h_num_divisors : (2 + 1) * (2 + 1) = 9 := by norm_num
  use 9
  rw h_num_divisors
  sorry

end divisors_of_36_l682_682569


namespace clock_displays_unique_digits_minutes_l682_682311

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end clock_displays_unique_digits_minutes_l682_682311


namespace dist_P_to_b_l682_682982

variables (a b : Line) (A B P : Point)
variables (h1 : A ∈ a) (h2 : B ∈ b) (h3 : AB = 2) (h4 : angle a b = 30) (h5 : AP = 4)

theorem dist_P_to_b : distance P b = 2 * sqrt 2 :=
sorry

end dist_P_to_b_l682_682982


namespace tangent_plane_eq_normal_line_eq_l682_682062

variables (x y z : ℝ) (F : ℝ → ℝ → ℝ → ℝ)
def surface_eq : Prop := F x  y z = x * y - z

def point_M : Prop := (1, 1, 1)

def partial_derivatives : Prop := 
  ∂ F / ∂ x = y ∧
  ∂ F / ∂ y = x ∧
  ∂ F / ∂ z = -1

theorem tangent_plane_eq : 
  surface_eq x y z ∧ point_M ∧ partial_derivatives → (x + y - z - 1 = 0) :=
sorry

theorem normal_line_eq : 
  surface_eq x y z ∧ point_M ∧ partial_derivatives → (frac(x - 1, 1) = frac(y - 1, 1) ∧ frac(z - 1, -1)) :=
sorry

end tangent_plane_eq_normal_line_eq_l682_682062


namespace circle_intersections_l682_682465

theorem circle_intersections (C1 C2 : Circle)
  (hC1 : C1.center = (0, 3 / 2) ∧ C1.radius = 3 / 2)
  (hC2 : C2.center = (5 / 2, 0) ∧ C2.radius = 5 / 2) :
  ∃ P1 P2 P3 P4 : Point, P1 ≠ P2 ∧ P1 ≠ P3 ∧ P1 ≠ P4 ∧ P2 ≠ P3 ∧ P2 ≠ P4 ∧ P3 ≠ P4 ∧
    (P1 ∈ C1.points ∧ P1 ∈ C2.points) ∧
    (P2 ∈ C1.points ∧ P2 ∈ C2.points) ∧
    (P3 ∈ C1.points ∧ P3 ∈ C2.points) ∧
    (P4 ∈ C1.points ∧ P4 ∈ C2.points) :=
sorry

end circle_intersections_l682_682465


namespace hours_per_day_l682_682612

theorem hours_per_day
  (num_warehouse : ℕ := 4)
  (num_managers : ℕ := 2)
  (rate_warehouse : ℝ := 15)
  (rate_manager : ℝ := 20)
  (tax_rate : ℝ := 0.10)
  (days_worked : ℕ := 25)
  (total_cost : ℝ := 22000) :
  ∃ h : ℝ, 6 * h * days_worked * (rate_warehouse + rate_manager) * (1 + tax_rate) = total_cost ∧ h = 8 :=
by
  sorry

end hours_per_day_l682_682612


namespace clock_display_four_different_digits_l682_682316

theorem clock_display_four_different_digits :
  (∑ t in finset.range (24*60), if (((t / 60).div1000 ≠ (t / 60).mod1000) ∧ 
    ((t / 60).div1000 ≠ (t % 60).div1000) ∧ ((t / 60).div1000 ≠ (t % 60).mod1000) ∧ 
    ((t / 60).mod1000 ≠ (t % 60).div1000) ∧ ((t / 60).mod1000 ≠ (t % 60).mod1000) ∧ 
    ((t % 60).div1000 ≠ (t % 60).mod1000)) then 1 else 0) = 588 :=
by
  sorry

end clock_display_four_different_digits_l682_682316


namespace probability_of_x_in_interval_l682_682142

theorem probability_of_x_in_interval (t : ℝ) (ht : t > 0) :
  let interval1 := [-t, 4 * t]
  let interval2 := [-(1 / 2) * t, t]
  (length_interval interval2) / (length_interval interval1) = 3 / 10 :=
by
  sorry

end probability_of_x_in_interval_l682_682142


namespace problem_l682_682180

noncomputable def vector_a (ω φ x : ℝ) : ℝ × ℝ := (Real.sin (ω / 2 * x + φ), 1)
noncomputable def vector_b (ω φ x : ℝ) : ℝ × ℝ := (1, Real.cos (ω / 2 * x + φ))
noncomputable def f (ω φ x : ℝ) : ℝ := 
  let a := vector_a ω φ x
  let b := vector_b ω φ x
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2)

theorem problem 
  (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π / 4)
  (h_period : Function.Periodic (f ω φ) 4)
  (h_point1 : f ω φ 1 = 1 / 2) : 
  ω = π / 2 ∧ ∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ f (π / 2) (π / 12) x ∧ f (π / 2) (π / 12) x ≤ 1 / 2 := 
by
  sorry

end problem_l682_682180


namespace symmetric_difference_count_l682_682256

noncomputable def symmetricDifference {α : Type*} (A B : set α) : set α :=
  (A \ B) ∪ (B \ A)

theorem symmetric_difference_count (n : ℕ) (h_pos : n > 0) (sets : fin (2^n + 1) → set ℕ) (red blue : set (fin (2^n + 1))) 
  (h_partition : ∀ i, i ∈ red ∨ i ∈ blue) (h_disjoint : disjoint red blue) (h_nonempty_red : red.nonempty) (h_nonempty_blue : blue.nonempty) :
  finset.card ({A Δ B | A ∈ red ∧ B ∈ blue}.to_finset) ≥ 2^n := 
begin
  sorry,
end

end symmetric_difference_count_l682_682256


namespace pats_password_length_l682_682648

/-- Pat’s computer password is made up of several kinds of alphanumeric and symbol characters for security.
  He uses:
  1. A string of eight random lowercase letters.
  2. A string half that length of alternating upper case letters and numbers.
  3. One symbol on each end of the password.

  Prove that the total number of characters in Pat's computer password is 14.
-/ 
theorem pats_password_length : 
  let lowercase_len := 8 in
  let alternating_len := lowercase_len / 2 in
  let symbols := 2 in
  lowercase_len + alternating_len + symbols = 14 := 
by 
  -- definitions
  let lowercase_len : Nat := 8
  let alternating_len : Nat := lowercase_len / 2
  let symbols : Nat := 2
  
  -- calculation
  have total_length := lowercase_len + alternating_len + symbols
  
  -- assertion
  show total_length = 14 from sorry

end pats_password_length_l682_682648


namespace last_three_nonzero_digits_of_80_factorial_mod_7_l682_682040

noncomputable def factorial80_last_three_nonzero_mod_7 : Nat := 6

theorem last_three_nonzero_digits_of_80_factorial_mod_7 :
  (let last_three_digits := (80!.div (10^19)) % 1000 in last_three_digits % 7 = factorial80_last_three_nonzero_mod_7) :=
by
  sorry

end last_three_nonzero_digits_of_80_factorial_mod_7_l682_682040


namespace existential_integer_divisible_by_11_and_9_universal_x_plus_one_over_x_ge_two_existential_integer_log2_gt_two_l682_682468

-- (1) There exists an integer that is divisible by both 11 and 9.
theorem existential_integer_divisible_by_11_and_9 : 
  ∃ x : ℤ, 11 ∣ x ∧ 9 ∣ x :=
sorry

-- (2) For all x in the set {x | x > 0}, x + 1/x ≥ 2.
theorem universal_x_plus_one_over_x_ge_two : 
  ∀ x : ℝ, x > 0 → x + 1/x ≥ 2 :=
sorry

-- (3) There exists an integer such that log2(x) > 2.
theorem existential_integer_log2_gt_two : 
  ∃ x : ℤ, real.log 2 x > 2 :=
sorry

end existential_integer_divisible_by_11_and_9_universal_x_plus_one_over_x_ge_two_existential_integer_log2_gt_two_l682_682468


namespace number_of_smoothies_l682_682615

-- Definitions of the given conditions
def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def total_cost : ℕ := 17

-- Statement of the proof problem
theorem number_of_smoothies (S : ℕ) : burger_cost + sandwich_cost + S * smoothie_cost = total_cost → S = 2 :=
by
  intro h
  sorry

end number_of_smoothies_l682_682615


namespace distance_is_sqrt_17_l682_682519

open Real

def distance_from_point_to_line (P A l : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ) : ℝ := 
  let PA := (P.1 - A.1, P.2 - A.2, P.3 - A.3)
  let cross_product := (a.2 * PA.3 - a.3 * PA.2, a.3 * PA.1 - a.1 * PA.3, a.1 * PA.2 - a.2 * PA.1)
  real_sqrt ((cross_product.1 ^ 2 + cross_product.2 ^ 2 + cross_product.3 ^ 2) /
             (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2))

theorem distance_is_sqrt_17 : distance_from_point_to_line (2, -1, 2) (1, 2, -1) (1, 2, -1) (-1, 0, 1) = √17 :=
by
  sorry

end distance_is_sqrt_17_l682_682519


namespace find_A_l682_682642

theorem find_A (A : ℝ) (h : (12 + 3) * (12 - A) = 120) : A = 4 :=
by sorry

end find_A_l682_682642


namespace incenter_geometric_locus_l682_682064

-- Define the three parallel lines on a plane
variables {a b c : ℝ → ℝ}

-- Define the vertices A, B, C of the triangles being on these parallel lines
variables {A : ℝ × ℝ} {B : ℝ × ℝ} {C : ℝ × ℝ}

-- Conditions on points A, B, C to be on lines a, b, c respectively
def on_line_a (A : ℝ × ℝ) : Prop := ∃ x, A = (x, a x)
def on_line_b (B : ℝ × ℝ) : Prop := ∃ x, B = (x, b x)
def on_line_c (C : ℝ × ℝ) : Prop := ∃ x, C = (x, c x)

-- Define the incenter of triangle ABC
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry -- Placeholder for actual incenter computation

-- Define the geometric locus of the centers of incircles
def geometric_locus (A B C : ℝ × ℝ) : set (ℝ × ℝ) := sorry -- Placeholder for actual locus definition

-- The main theorem
theorem incenter_geometric_locus :
  (on_line_a A) →
  (on_line_b B) →
  (on_line_c C) →
  incenter A B C ∈ geometric_locus A B C :=
sorry

end incenter_geometric_locus_l682_682064


namespace intersection_A_complementB_l682_682621

noncomputable def setA : Set ℝ := {x : ℝ | ∃ y : ℝ, y = log 2 (1 - x) ∧ 1 - x > 0}
def setB : Set ℝ := {x : ℝ | x ≤ -1}
def complementB : Set ℝ := {x : ℝ | x > -1}
def resultSet : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem intersection_A_complementB :
  (setA ∩ complementB) = resultSet :=
by
  sorry

end intersection_A_complementB_l682_682621


namespace margarita_jumps_farther_l682_682658

-- Definitions for conditions
def RiccianaTotalDistance := 24
def RiccianaRunDistance := 20
def RiccianaJumpDistance := 4

def MargaritaRunDistance := 18
def MargaritaJumpDistance := 2 * RiccianaJumpDistance - 1

-- Theorem statement to prove the question
theorem margarita_jumps_farther :
  (MargaritaRunDistance + MargaritaJumpDistance) - RiccianaTotalDistance = 1 :=
by
  -- Proof will be written here
  sorry

end margarita_jumps_farther_l682_682658


namespace solve_eq_l682_682286

theorem solve_eq (x : ℝ) (n : ℤ) : 
  (∃ x, ((2:ℝ)^(floor (Real.sin x)) = (3:ℝ)^(1 - (Real.cos x)))) → 
  (x = 2 * Real.pi * n) :=
sorry

end solve_eq_l682_682286


namespace arithmetic_progression_15th_term_l682_682059

theorem arithmetic_progression_15th_term :
  let a := 2
  let d := 3
  let n := 15
  a + (n - 1) * d = 44 :=
by
  let a := 2
  let d := 3
  let n := 15
  sorry

end arithmetic_progression_15th_term_l682_682059


namespace maximum_value_l682_682239

noncomputable def p : ℝ := 1 + 1/2 + 1/2^2 + 1/2^3 + 1/2^4 + 1/2^5

theorem maximum_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h_constraint : (x - 1)^2 + (y - 1)^2 + (z - 1)^2 = 27) : 
  x^p + y^p + z^p ≤ 40.4 :=
sorry

end maximum_value_l682_682239


namespace find_BD_l682_682220

theorem find_BD 
  (A B C D : Type)
  (AC BC : ℝ) (h₁ : AC = 10) (h₂ : BC = 10)
  (AD CD : ℝ) (h₃ : AD = 12) (h₄ : CD = 5) :
  ∃ (BD : ℝ), BD = 152 / 24 := 
sorry

end find_BD_l682_682220


namespace modular_sum_of_inverses_of_2_l682_682731

theorem modular_sum_of_inverses_of_2 (x: ℤ) (h: x = 2): 
  (x⁻¹ + x⁻² + x⁻³ + x⁻⁴ + x⁻⁵ + x⁻⁶) % 13 = 2 := sorry

end modular_sum_of_inverses_of_2_l682_682731


namespace smallest_positive_multiple_of_45_l682_682873

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682873


namespace simplified_factorial_fraction_l682_682438

theorem simplified_factorial_fraction :
  (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 :=
by
  sorry

end simplified_factorial_fraction_l682_682438


namespace smallest_positive_multiple_of_45_l682_682929

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682929


namespace reflection_constant_product_l682_682119

-- Define the setup for the problem
structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the reflection function
def reflection (A O : Point) : Point :=
  ⟨2 * O.x - A.x, 2 * O.y - A.y⟩

-- Define the main theorem
theorem reflection_constant_product
  (O : Point)
  (A : Point)
  (A_not_on_circle : distance O A ≠ O.radius)
  (P : Point)
  (P_on_circle : distance O P = O.radius)
  (B := reflection A O)
  (Q : Point)
  (Q_on_circle : distance O Q = O.radius)
  (perpendicular : ∀ P Q, is_perpendicular (line_through A P) (line_through P Q)):
  distance A P * distance B Q = constant := 
  sorry

end reflection_constant_product_l682_682119


namespace smallest_sum_squares_edges_is_cube_l682_682425

theorem smallest_sum_squares_edges_is_cube (V : ℝ) (a b c : ℝ)
  (h_vol : a * b * c = V) :
  a^2 + b^2 + c^2 ≥ 3 * (V^(2/3)) := 
sorry

end smallest_sum_squares_edges_is_cube_l682_682425


namespace smallest_positive_multiple_l682_682780

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682780


namespace mental_math_competition_l682_682664

theorem mental_math_competition :
  -- The number of teams that participated is 4
  (∃ (teams : ℕ) (numbers : List ℕ),
     -- Each team received a number that can be written as 15M + 11m where M is the largest odd divisor
     -- and m is the smallest odd divisor greater than 1.
     teams = 4 ∧ 
     numbers = [528, 880, 1232, 1936] ∧
     ∀ n ∈ numbers,
       ∃ M m, M > 1 ∧ m > 1 ∧
       M % 2 = 1 ∧ m % 2 = 1 ∧
       (∀ d, d ∣ n → (d % 2 = 1 → M ≥ d)) ∧ 
       (∀ d, d ∣ n → (d % 2 = 1 ∧ d > 1 → m ≤ d)) ∧
       n = 15 * M + 11 * m) :=
sorry

end mental_math_competition_l682_682664


namespace smallest_positive_multiple_45_l682_682899

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682899


namespace factorial_quotient_52_50_l682_682450

theorem factorial_quotient_52_50 : (Nat.factorial 52) / (Nat.factorial 50) = 2652 := 
by 
  sorry

end factorial_quotient_52_50_l682_682450


namespace hyperbola_asymptotes_l682_682324

theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → (y = x / 2 ∨ y = -x / 2) :=
by
  sorry

end hyperbola_asymptotes_l682_682324


namespace smallest_positive_multiple_of_45_is_45_l682_682798

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682798


namespace compute_α_l682_682243

open Complex

def α : ℂ := 6 - 3 * Complex.i
def β : ℂ := 4 + 3 * Complex.i

theorem compute_α (h1 : ∃ x : ℝ, (α + β) = x ∧ 0 < x)
                  (h2 : ∃ z : ℝ, (Complex.i * (α - 3 * β)) = z ∧ 0 < z) :
  α = 6 - 3 * Complex.i :=
by
  sorry

end compute_α_l682_682243


namespace problem_statement_l682_682003

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end problem_statement_l682_682003


namespace clock_four_different_digits_l682_682307

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end clock_four_different_digits_l682_682307


namespace mechanism_completion_times_l682_682727

theorem mechanism_completion_times :
  ∃ (x y : ℝ), (1 / x + 1 / y = 1 / 30) ∧ (6 * (1 / x + 1 / y) + 40 * (1 / y) = 1) ∧ x = 75 ∧ y = 50 :=
by {
  sorry
}

end mechanism_completion_times_l682_682727


namespace probability_winning_ticket_l682_682649

open Finset

theorem probability_winning_ticket : 
  let s := ({1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40} : Finset ℕ) in
  (∃ (chosen : Finset ℕ), chosen.card = 6 ∧ chosen ⊆ s ∧ 
    ∃ (win : Finset ℕ), win.card = 6 ∧ win ⊆ s ∧ 
    (∑ x in chosen, Real.log x / Real.log 10).denom = 1 ∧
    (∑ x in win, Real.log x / Real.log 10).denom = 1 ∧
    chosen = win) →
  (1 / 4 : ℝ) :=
by
  sorry

end probability_winning_ticket_l682_682649


namespace range_of_a_l682_682991

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) → a < 1 :=
by
  sorry

end range_of_a_l682_682991


namespace math_expression_equivalent_l682_682023

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end math_expression_equivalent_l682_682023


namespace smallest_positive_multiple_of_45_l682_682866

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682866


namespace clock_four_different_digits_l682_682304

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end clock_four_different_digits_l682_682304


namespace egg_sales_l682_682366

/-- Two vendors together sell 110 eggs and both have equal revenues.
    Given the conditions about changing the number of eggs and corresponding revenues,
    the first vendor sells 60 eggs and the second vendor sells 50 eggs. -/
theorem egg_sales (x y : ℝ) (h1 : x + (110 - x) = 110) (h2 : 110 * (y / x) = 5) (h3 : 110 * (y / (110 - x)) = 7.2) :
  x = 60 ∧ (110 - x) = 50 :=
by sorry

end egg_sales_l682_682366


namespace valid_strings_count_l682_682502

def vowels := { 'a', 'e', 'i', 'o', 'u', 'y' }
def consonants := { 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z' }

noncomputable def count_valid_strings (length : ℕ) : ℕ :=
  if length = 5 then
    2335200
  else
    0

theorem valid_strings_count :
  count_valid_strings 5 = 2335200 := 
  by 
    unfold count_valid_strings
    simp
    sorry

end valid_strings_count_l682_682502


namespace smallest_positive_multiple_of_45_l682_682828

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682828


namespace molecular_weight_compound_l682_682370

theorem molecular_weight_compound :
  let weight_H := 1.008
  let weight_Cr := 51.996
  let weight_O := 15.999
  let n_H := 2
  let n_Cr := 1
  let n_O := 4
  (n_H * weight_H) + (n_Cr * weight_Cr) + (n_O * weight_O) = 118.008 :=
by
  sorry

end molecular_weight_compound_l682_682370


namespace smallest_positive_multiple_of_45_l682_682754

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682754


namespace smallest_positive_multiple_l682_682784

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682784


namespace neighborhood_B_num_homes_l682_682228

theorem neighborhood_B_num_homes :
  (∃ x : ℕ, 10 * 2 * 2 < 2 * 5 * x ∧ 2 * 5 * x = 50) → x = 5 :=
by
  intro hx
  rcases hx with ⟨x, h1, h2⟩
  linarith
  sorry

end neighborhood_B_num_homes_l682_682228


namespace fill_bathtub_time_l682_682586

theorem fill_bathtub_time
  (r_cold : ℚ := 1/10)
  (r_hot : ℚ := 1/15)
  (r_empty : ℚ := -1/12)
  (net_rate : ℚ := r_cold + r_hot + r_empty) :
  net_rate = 1/12 → 
  t = 12 :=
by
  sorry

end fill_bathtub_time_l682_682586


namespace determine_figures_l682_682337

theorem determine_figures (ρ θ : ℝ) (hρ : ρ ≥ 0)
  (h1 : (ρ - 1) * (θ - π) = 0)
  (x y : ℝ)
  (h2 : x = tan θ)
  (h3 : y = 2 / cos θ) :
  (ρ = 1 → x^2 + y^2 = 1) ∧ (θ = π → (ρ ≥ 0 → x = -1 ∧ y = -2)) ∧ (x = tan θ ∧ y = 2 / cos θ → y^2 - 4 * x^2 = 4) :=
by
  sorry

end determine_figures_l682_682337


namespace smallest_positive_multiple_of_45_l682_682967

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682967


namespace smallest_positive_multiple_of_45_is_45_l682_682921

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682921


namespace expression_evaluation_l682_682014

theorem expression_evaluation :
  (π - 1)^0 + 4 * real.sin (real.pi / 4) - real.sqrt 8 + abs (-3) = 4 := 
sorry

end expression_evaluation_l682_682014


namespace negation_example_l682_682699

theorem negation_example :
  (¬ ∀ x : ℝ, x^2 ∈ ℚ) ↔ ∃ x : ℝ, x^2 ∉ ℚ :=
by sorry

end negation_example_l682_682699


namespace smallest_positive_multiple_of_45_l682_682756

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682756


namespace value_of_expression_when_b_is_3_l682_682376

theorem value_of_expression_when_b_is_3 :
  ∀ (b : ℝ), b = 3 → (4 * b^(-2) + (b^(-2) / 3)) / b^2 = 13 / 243 :=
by
  intros b hb
  rw [hb]
  sorry

end value_of_expression_when_b_is_3_l682_682376


namespace smallest_positive_multiple_of_45_l682_682870

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682870


namespace problem_statement_l682_682478

theorem problem_statement (n m : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) : 
  (n * 5^n)^n = m * 5^9 ↔ n = 3 ∧ m = 27 :=
by {
  sorry
}

end problem_statement_l682_682478


namespace smallest_positive_multiple_l682_682776

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682776


namespace factorial_quotient_52_50_l682_682452

theorem factorial_quotient_52_50 : (Nat.factorial 52) / (Nat.factorial 50) = 2652 := 
by 
  sorry

end factorial_quotient_52_50_l682_682452


namespace num_divisors_36_l682_682555

theorem num_divisors_36 : ∃ n, n = 9 ∧ ∀ d : ℕ, d ∣ 36 → (d > 0 ∧ d ≤ 36) → ∃ k : ℕ, d = (2^k or 3^k or (2^p * 3^q) where p, q <= 2):
begin
  have factorization : 36 = 2^2 * 3^2 := by norm_num,
  have exponents : (2+1)*(2+1) = 9 := by norm_num,
  use 9,
  split,
  { exact exponents },
  {
    intros d hd hpos_range,
    cases hpos_range with hpos hrange,
    sorry -- Proof showing that there are exactly 9 positive divisors.
  },
end

end num_divisors_36_l682_682555


namespace expression_evaluation_l682_682016

theorem expression_evaluation :
  (π - 1)^0 + 4 * real.sin (real.pi / 4) - real.sqrt 8 + abs (-3) = 4 := 
sorry

end expression_evaluation_l682_682016


namespace part1_part2_l682_682258

noncomputable theory

open Real

variables (x y z : ℝ)

-- Condition
-- Let x, y, z be real numbers (not necessarily positive) such that x^4 + y^4 + z^4 + xyz = 4.
def condition := (x^4 + y^4 + z^4 + x * y * z = 4)

-- Prove that x ≤ 2
theorem part1 (h : condition x y z) : x ≤ 2 := 
sorry

-- Prove that sqrt (2 - x) ≥ (y + z) / 2.
theorem part2 (h : condition x y z) : sqrt (2 - x) ≥ ( (y + z) / 2) :=
sorry

end part1_part2_l682_682258


namespace number_of_divisors_of_36_l682_682549

theorem number_of_divisors_of_36 :  
  let n := 36
  number_of_divisors n = 9 := 
by 
  sorry

end number_of_divisors_of_36_l682_682549


namespace clock_shows_four_different_digits_for_588_minutes_l682_682302

-- Definition of the problem
def isFourDifferentDigits (h1 h2 m1 m2 : Nat) : Bool :=
  (h1 ≠ h2) && (h1 ≠ m1) && (h1 ≠ m2) && (h2 ≠ m1) && (h2 ≠ m2) && (m1 ≠ m2)

noncomputable def countFourDifferentDigitsMinutes : Nat :=
  let validMinutes := List.filter (λ (t : Nat × Nat),
    let (h, m) := t
    let h1 := h / 10
    let h2 := h % 10
    let m1 := m / 10
    let m2 := m % 10
    isFourDifferentDigits h1 h2 m1 m2
  ) (List.product (List.range 24) (List.range 60))
  validMinutes.length

theorem clock_shows_four_different_digits_for_588_minutes :
  countFourDifferentDigitsMinutes = 588 := sorry

end clock_shows_four_different_digits_for_588_minutes_l682_682302


namespace ball_distribution_l682_682573

theorem ball_distribution : 
  ∃ n : ℕ, n = 5 ∧
  let balls := 5 in
  let boxes := 3 in
  let distinguishable := false in
  count_distributions balls boxes distinguishable = n :=
begin
  sorry

end ball_distribution_l682_682573


namespace probability_N_taller_than_L_l682_682095

variable (M N L O : ℕ)

theorem probability_N_taller_than_L 
  (h1 : N < M) 
  (h2 : L > O) : 
  0 = 0 := 
sorry

end probability_N_taller_than_L_l682_682095


namespace incircle_excircle_tangents_l682_682328

theorem incircle_excircle_tangents {A B C : Point}
  (a b c : ℝ) (K L : Point)
  (hA : dist A B = a) (hB : dist B C = b) (hC : dist C A = c)
  (incircle : Circle) (excircle : Circle)
  (hK : incircle.is_tangent_at K B C)
  (hL : excircle.is_tangent_at L B C) :
  dist C K = dist B L ∧ dist C K = (a + b - c) / 2 :=
by
  sorry

end incircle_excircle_tangents_l682_682328


namespace smallest_positive_multiple_of_45_l682_682878

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682878


namespace sum_m_n_t_eq_13_quadratic_function_expression_l682_682146

-- Definitions and conditions
def point := (ℝ, ℝ)

variables (m n t : ℕ) (h_m_lt_n : m < n) (h_m_pos : 0 < m) (h_n_pos : 0 < n) (h_t_pos : 0 < t)
variables (A B C O : point)
def OA : ℝ := m
def OB : ℝ := n
def OC : ℝ := t
#check A
#check B
#check C
#check O

-- Given points and their coordinates
def A : point := (-m, 0)
def B : point := (n, 0)
def C : point := (0, t)
def O : point := (0, 0)

-- Angle condition
axiom angle_ACB : ∠A C B = π / 2

-- Given equation
axiom equation_given : OA ^ 2 + OB ^ 2 + OC ^ 2 = 13 * (OA + OB - OC)

-- Task 1: Prove that m + n + t = 13
theorem sum_m_n_t_eq_13 : m + n + t = 13 := sorry

-- Task 2: Prove the expression of the quadratic function passing through points A, B, C
theorem quadratic_function_expression : 
  ∃ a b c : ℝ, 
    (∀ x, a * x * x + b * x + c = 0 ↔ x = -m ∨ x = n) ∧ 
    (a ≠ 0) ∧ 
    (a * 0 * 0 + b * 0 + c = t) :=
sorry

end sum_m_n_t_eq_13_quadratic_function_expression_l682_682146


namespace radius_of_circle_l682_682674

theorem radius_of_circle (r : ℝ) : 
  (∀ x : ℝ, y = x^2 → six_parabolas_tangent_45_degrees y circle)
→ r = 1 / 4 := by
sorry

end radius_of_circle_l682_682674


namespace smallest_positive_multiple_of_45_is_45_l682_682924

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682924


namespace expression_evaluation_l682_682012

theorem expression_evaluation :
  (π - 1)^0 + 4 * real.sin (real.pi / 4) - real.sqrt 8 + abs (-3) = 4 := 
sorry

end expression_evaluation_l682_682012


namespace gcd_repeated_three_digit_integers_l682_682408

theorem gcd_repeated_three_digit_integers : 
  ∀ m ∈ {n | 100 ≤ n ∧ n < 1000}, 
  gcd (1001 * m) (1001 * (m + 1)) = 1001 :=
by
  sorry

end gcd_repeated_three_digit_integers_l682_682408


namespace log2_x_y_squared_l682_682274

theorem log2_x_y_squared (x y : ℝ) (hx1 : x ≠ 1) (hy1 : y ≠ 1) (h1 : log 2 x = log y 16) (h2 : x * y = 64) : (log 2 (x / y))^2 = 20 := 
sorry

end log2_x_y_squared_l682_682274


namespace smallest_positive_multiple_l682_682790

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682790


namespace distance_between_skew_lines_l682_682677

-- Define the conditions of the problem
structure Tetrahedron where
  A B C O : ℝ^3
  OA_length : ∥A - O∥ = 1
  OB_length : ∥B - O∥ = 1
  OC_length : ∥C - O∥ = 1
  AB_length : ∥B - A∥ = 1
  BC_length : ∥C - B∥ = 1
  CA_length : ∥C - A∥ = 1

-- Midpoints
def E (t : Tetrahedron) : ℝ^3 := 0.5 • (t.A + t.B)
def F (t : Tetrahedron) : ℝ^3 := 0.5 • (t.O + t.C)

-- Define the proof statement
theorem distance_between_skew_lines (t : Tetrahedron) : 
  distance_between_lines (mk_line t.O (E t)) (mk_line t.B (F t)) = sqrt(19) / 19 :=
by 
  sorry

end distance_between_skew_lines_l682_682677


namespace Sam_scored_points_l682_682662

theorem Sam_scored_points (total_points friend_points S: ℕ) (h1: friend_points = 12) (h2: total_points = 87) (h3: total_points = S + friend_points) : S = 75 :=
by
  sorry

end Sam_scored_points_l682_682662


namespace cos_A_in_triangle_abc_l682_682589

theorem cos_A_in_triangle_abc (a b c S : ℝ) (A B C : ℝ) 
  (h1 : S + a^2 = (b + c)^2)
  (h2 : S = 1/2 * b * c * sin A)
  (h3 : sin A = 4 * cos A + 4)
  (h4 : sin A^2 + cos A^2 = 1) : 
  cos A = -15/17 :=
sorry

end cos_A_in_triangle_abc_l682_682589


namespace three_digit_palindrome_average_proof_l682_682073

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := List.reverse (nat.digits 10 n);
  nat.of_digits 10 digits = n

noncomputable def reverse_digits (n : ℕ) : ℕ :=
  nat.of_digits 10 (List.reverse (nat.digits 10 n))

theorem three_digit_palindrome_average_proof :
  ∃ m n : ℕ, 100 ≤ m ∧ m < 1000 ∧
    is_palindrome m ∧
    n = reverse_digits m ∧
    (m + n) / 2 = n := by
  sorry

end three_digit_palindrome_average_proof_l682_682073


namespace smallest_positive_multiple_of_45_is_45_l682_682954

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682954


namespace smallest_positive_multiple_of_45_l682_682841

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682841


namespace divide_plane_into_regions_l682_682457

theorem divide_plane_into_regions :
  (∀ (x y : ℝ), y = 3 * x ∨ y = x / 3) →
  ∃ (regions : ℕ), regions = 4 :=
by
  sorry

end divide_plane_into_regions_l682_682457


namespace smallest_positive_multiple_of_45_l682_682888

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682888


namespace angle_between_slant_height_and_base_l682_682409

theorem angle_between_slant_height_and_base (R : ℝ) (diam_base_upper diam_base_lower : ℝ) 
(h1 : diam_base_upper + diam_base_lower = 5 * R)
: ∃ θ : ℝ, θ = Real.arcsin (4 / 5) := 
sorry

end angle_between_slant_height_and_base_l682_682409


namespace find_scalars_l682_682131

noncomputable def vec_a : ℝ × ℝ × ℝ := (2, 2, 2)
noncomputable def vec_b : ℝ × ℝ × ℝ := (3, -2, 0)
noncomputable def vec_c : ℝ × ℝ × ℝ := (0, 2, -3)
noncomputable def vec_rhs : ℝ × ℝ × ℝ := (5, -1, 7)

theorem find_scalars :
  ∃ (p q r : ℝ),
  vec_rhs = (p * vec_a.1 + q * vec_b.1 + r * vec_c.1, 
             p * vec_a.2 + q * vec_b.2 + r * vec_c.2, 
             p * vec_a.3 + q * vec_b.3 + r * vec_c.3)
    ∧ p = (11 / 12)
    ∧ q = (17 / 13)
    ∧ r = (-19 / 13) :=
sorry

end find_scalars_l682_682131


namespace largest_possible_d_l682_682288

theorem largest_possible_d : 
  (∀ x : ℝ, -3 ≤ h(x) ∧ h(x) ≤ 5) ∧ 
  (∀ x : ℝ, -1 ≤ k(x) ∧ k(x) ≤ 4) → 
  ∃ c d : ℝ, (∀ x : ℝ, c ≤ h(x) * k(x) ∧ h(x) * k(x) ≤ d) ∧ d = 20 := 
by 
  sorry

end largest_possible_d_l682_682288


namespace conclusion_A_conclusion_B_conclusion_C_conclusion_D_l682_682113

noncomputable theory

open Real

section
-- Define the functions and mathematical assumptions
variable (ω : ℝ) (ϕ : ℝ)
def f (x : ℝ) : ℝ := sin (ω * x + π / 3 + ϕ)
def g (x : ℝ) : ℝ := sin (ω * x + ϕ)

-- Assumptions
axiom h1 : ω > 0
axiom h2 : abs ϕ < π / 2
axiom h3 : ∀ x : ℝ, f x = f (-x)  -- f(x) is even

-- Proofs of the conclusions
theorem conclusion_A : ϕ = π / 6 :=
sorry

theorem conclusion_B (h_smallest_period : ∃ T > 0, ∀ x : ℝ, g (x + T) = g x ∧ T = 3 * π) : ω = 2 / 3 :=
sorry

theorem conclusion_C (h_extreme_points : ∀ t ∈ Ioo (0 : ℝ) π, (x | g x = sin t) → |g' x| = 1) : 7 / 3 < ω ∧ ω ≤ 10 / 3 :=
sorry

theorem conclusion_D (h_value_at_pi_4 : g (π / 4) = sqrt 3 / 2) : min ω (2 / 3) :=
sorry
end

end conclusion_A_conclusion_B_conclusion_C_conclusion_D_l682_682113


namespace cost_price_of_ball_l682_682270

variable (C : ℝ)

theorem cost_price_of_ball (h : 15 * C - 720 = 5 * C) : C = 72 :=
by
  sorry

end cost_price_of_ball_l682_682270


namespace ball_distribution_l682_682572

theorem ball_distribution : 
  ∃ n : ℕ, n = 5 ∧
  let balls := 5 in
  let boxes := 3 in
  let distinguishable := false in
  count_distributions balls boxes distinguishable = n :=
begin
  sorry

end ball_distribution_l682_682572


namespace graduate_degree_ratio_l682_682199

theorem graduate_degree_ratio (G C N : ℕ) (h1 : C = (2 / 3 : ℚ) * N)
  (h2 : (G : ℚ) / (G + C) = 0.15789473684210525) :
  (G : ℚ) / N = 1 / 8 :=
  sorry

end graduate_degree_ratio_l682_682199


namespace fill_bathtub_time_l682_682584

theorem fill_bathtub_time (V : ℝ) (cold_rate hot_rate drain_rate net_rate : ℝ) 
  (hcold : cold_rate = V / 10) 
  (hhot : hot_rate = V / 15) 
  (hdrain : drain_rate = -V / 12) 
  (hnet : net_rate = cold_rate + hot_rate + drain_rate) 
  (V_eq : V = 1) : 
  1 / net_rate = 12 :=
by {
  -- placeholder for proof steps
  sorry
}

end fill_bathtub_time_l682_682584


namespace smallest_positive_multiple_of_45_l682_682757

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682757


namespace cos_double_angle_l682_682495

-- Given conditions
variable (θ : ℝ)
variable h₁ : sin θ + cos θ = 1 / 5
variable h₂ : (π / 2) ≤ θ ∧ θ ≤ (3 * π / 4)

-- Proof problem: Show that cos 2θ is -7/25
theorem cos_double_angle : cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l682_682495


namespace smallest_positive_multiple_of_45_l682_682926

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l682_682926


namespace math_problem_solution_l682_682125

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  ∃ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def parabola_equation (p : ℝ) (h3 : p > 0) : Prop :=
  ∃ (x y : ℝ), y^2 = 2 * p * x

noncomputable def conditions (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (eccentricity : ℝ) (h4 : c / a = eccentricity) : Prop :=
  (2 * b = 2) ∧ (a^2 = b^2 + c^2)

noncomputable def line_equation (k : ℝ) (c : ℝ) : Prop :=
  ∃ (x y : ℝ), y = k * (x - c)

noncomputable def ab_length (a : ℝ) (k : ℝ) : ℝ :=
  2 * sqrt 5 * (k^2 + 1) / (1 + 5 * k^2)

noncomputable def cd_length (k : ℝ) : ℝ :=
  8 * (k^2 + 1) / k^2

noncomputable def λ_exists (k : ℝ) (λ : ℝ) (ab cd : ℝ) : Prop :=
  1 / ab + λ / cd = 4 / (8 * sqrt 5)

theorem math_problem_solution :
  ∃ (a b c p : ℝ) (λ : ℝ),
    (ellipse_equation a b (by linarith) (by linarith)) ∧
    (parabola_equation p (by linarith)) ∧
    (conditions a b c (by linarith) (by linarith) (2 * sqrt 5 / 5) (by linarith)) ∧
    (line_equation (by linarith) c) ∧
    (λ_exists (by linarith) λ (ab_length a (by linarith)) (cd_length (by linarith))) :=
sorry

end math_problem_solution_l682_682125


namespace green_ball_probability_l682_682459

-- Defining the number of red and green balls in each container
def containerI_red : ℕ := 10
def containerI_green : ℕ := 5

def containerII_red : ℕ := 3
def containerII_green : ℕ := 6

def containerIII_red : ℕ := 3
def containerIII_green : ℕ := 6

-- Probability of selecting any container
def prob_container : ℚ := 1 / 3

-- Defining the probabilities of drawing a green ball from each container
def prob_green_I : ℚ := containerI_green / (containerI_red + containerI_green)
def prob_green_II : ℚ := containerII_green / (containerII_red + containerII_green)
def prob_green_III : ℚ := containerIII_green / (containerIII_red + containerIII_green)

-- Law of total probability
def prob_green_total : ℚ :=
  prob_container * prob_green_I +
  prob_container * prob_green_II +
  prob_container * prob_green_III

-- The mathematical statement to be proven
theorem green_ball_probability :
  prob_green_total = 5 / 9 := by
  sorry

end green_ball_probability_l682_682459


namespace find_lambda_find_sum_l682_682504

namespace ArithmeticSequence

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {n : ℕ} {λ : ℝ}

-- Conditions
def arithmetic_sequence (A B : ℝ) : Prop :=
  ∀ n, a n = A * n + B

def sum_first_n_terms (A B : ℝ) : Prop :=
  ∀ n, S n = (A * n * (n + 1) / 2) + (B * n)

def given_condition (λ : ℝ) : Prop :=
  ∀ n, 2 * S n = (a n) ^ 2 + λ * n

-- Questions (Proof Problems)
theorem find_lambda (A B : ℝ) (h1 : arithmetic_sequence A B) (h2 : sum_first_n_terms A B) (h3 : given_condition λ)
  (non_zero : A ≠ 0) : λ = 1 := 
  sorry

theorem find_sum (arithmetic_seq : ∀ n, a n = n) : 
  T n = (n : ℚ) / (2 * n + 1) :=
  sorry

end ArithmeticSequence

end find_lambda_find_sum_l682_682504


namespace num_divisors_36_l682_682558

theorem num_divisors_36 : ∃ n, n = 9 ∧ ∀ d : ℕ, d ∣ 36 → (d > 0 ∧ d ≤ 36) → ∃ k : ℕ, d = (2^k or 3^k or (2^p * 3^q) where p, q <= 2):
begin
  have factorization : 36 = 2^2 * 3^2 := by norm_num,
  have exponents : (2+1)*(2+1) = 9 := by norm_num,
  use 9,
  split,
  { exact exponents },
  {
    intros d hd hpos_range,
    cases hpos_range with hpos hrange,
    sorry -- Proof showing that there are exactly 9 positive divisors.
  },
end

end num_divisors_36_l682_682558


namespace DL_eq_DM_l682_682616

noncomputable def midpoint (A B : Point) : Point := sorry

noncomputable def perpendicular_foot (P L : Point) (line : Line) : Prop := sorry

theorem DL_eq_DM {A B C P L M D : Point} :
  Triangle ABC ∧ ∠PAC = ∠PBC ∧ perpendicular_foot P L (line_of B C) ∧ 
  perpendicular_foot P M (line_of C A) ∧ D = midpoint A B →
  distance D L = distance D M :=
by
  intro h
  sorry

end DL_eq_DM_l682_682616


namespace smallest_positive_multiple_45_l682_682896

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682896


namespace smallest_even_digit_multiple_of_9_l682_682070

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0
def all_even_digits (n : ℕ) : Prop := ∀ d ∈ digits 10 n, d ∈ {0, 2, 4, 6, 8}

theorem smallest_even_digit_multiple_of_9 : 
  ∃ (n : ℕ), is_multiple_of_9 n ∧ all_even_digits n ∧ ∀ m, is_multiple_of_9 m ∧ all_even_digits m → n ≤ m :=
begin
  use 288,
  split,
  { unfold is_multiple_of_9,
    norm_num, },
  split,
  { unfold all_even_digits,
    intro d,
    simp only [digits, list.mem_cons_iff, list.mem_singleton],
    intros hd,
    iterate 3 {cases hd,
    -- Prove 2, 8 and 8 are all in {0, 2, 4, 6, 8}
    },
     sorry }, 
  { intro m,
    intro h,
    have hmul : is_multiple_of_9 m := h.1, 
    have heven : all_even_digits m := h.2,
    -- Prove minimum condition ensuring n=288 is smallest
    sorry
  }
end

end smallest_even_digit_multiple_of_9_l682_682070


namespace clock_displays_unique_digits_minutes_l682_682312

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end clock_displays_unique_digits_minutes_l682_682312


namespace seat_ways_at_least_two_girls_next_together_l682_682292

open Nat

theorem seat_ways_at_least_two_girls_next_together :
  let total_ways := fact 8,
      ways_no_girls_together := fact 3 * fact 5
  in total_ways - ways_no_girls_together = 39600 := 
by
  sorry

end seat_ways_at_least_two_girls_next_together_l682_682292


namespace increase_percent_exceeds_l682_682177

theorem increase_percent_exceeds (p q M : ℝ) (M_positive : 0 < M) (p_positive : 0 < p) (q_positive : 0 < q) (q_less_p : q < p) :
  (M * (1 + p / 100) * (1 + q / 100) > M) ↔ (0 < p ∧ 0 < q) :=
by
  sorry

end increase_percent_exceeds_l682_682177


namespace probability_nina_taller_than_lena_is_zero_l682_682090

-- Definition of participants and conditions
variable (M N L O : ℝ)

-- Conditions
def condition1 := N < M
def condition2 := L > O

-- Statement: Given conditions, the probability that N > L is 0
theorem probability_nina_taller_than_lena_is_zero
  (h1 : condition1)
  (h2 : condition2) :
  (P : ℝ) = 0 :=
by
  sorry

end probability_nina_taller_than_lena_is_zero_l682_682090


namespace factorial_division_l682_682455

theorem factorial_division :
  52! / 50! = 2652 := by
  sorry

end factorial_division_l682_682455


namespace relationship_between_abc_l682_682109

theorem relationship_between_abc:
    (a = 2 ^ (-2)) →
    (b = (Real.pi - 2) ^ 0) →
    (c = (-1) ^ 3) →
    c < a ∧ a < b :=
by
  intros h_a h_b h_c
  sorry

end relationship_between_abc_l682_682109


namespace problem_sum_problem_m_plus_n_l682_682238

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def S : set (ℕ × ℕ × ℕ) := { t | is_even (t.1 + t.2 + t.3) }

noncomputable def sum_S : ℚ :=
  ∑ t in S, 1 / (2^t.1 * 3^t.2 * 5^t.3)

theorem problem_sum :
  sum_S = 25 / 12 := sorry

theorem problem_m_plus_n :
  ∃ m n : ℕ, nat.coprime m n ∧ 25 / 12 = m / n ∧ m + n = 37 := sorry

end problem_sum_problem_m_plus_n_l682_682238


namespace trapezoid_median_properties_l682_682069

-- Define the variables
variables (a b x : ℝ)

-- State the conditions and the theorem
theorem trapezoid_median_properties (h1 : x = (2 * a) / 3) (h2 : x = b + 3) (h3 : x = (a + b) / 2) : x = 6 :=
by
  sorry

end trapezoid_median_properties_l682_682069


namespace smallest_positive_multiple_of_45_is_45_l682_682952

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682952


namespace stickers_needed_for_prizes_l682_682027

theorem stickers_needed_for_prizes
  (christine_stickers : ℕ)
  (robert_stickers : ℕ)
  (small_prize_stickers : ℕ)
  (medium_prize_stickers : ℕ)
  (large_prize_stickers : ℕ)
  (total_stickers : ℕ := christine_stickers + robert_stickers)
  (needed_small : ℕ := if total_stickers >= small_prize_stickers then 0 else small_prize_stickers - total_stickers)
  (needed_medium : ℕ := if total_stickers >= medium_prize_stickers then 0 else medium_prize_stickers - total_stickers)
  (needed_large : ℕ := if total_stickers >= large_prize_stickers then 0 else large_prize_stickers - total_stickers) :
  total_stickers = 4_250 →
  (christine_stickers = 2_500) →
  (robert_stickers = 1_750) →
  (small_prize_stickers = 4_000) →
  (medium_prize_stickers = 7_000) →
  (large_prize_stickers = 10_000) →
  needed_small = 0 ∧
  needed_medium = 2_750 ∧
  needed_large = 5_750 := by
  sorry

end stickers_needed_for_prizes_l682_682027


namespace smallest_positive_multiple_of_45_l682_682864

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682864


namespace tangent_line_slope_range_y_range_l682_682241

variable {x y : ℝ}

def curve (x : ℝ) : ℝ := x^2 - x + 1

def slope (x : ℝ) : ℝ := 2 * x - 1

theorem tangent_line_slope_range_y_range :
  (x ∈ set.Icc 0 2) → (curve x ∈ set.Icc (3/4) 3) :=
by
  sorry

end tangent_line_slope_range_y_range_l682_682241


namespace alpha_value_l682_682247

noncomputable def beta : ℂ := 4 + 3 * complex.I

theorem alpha_value (α : ℂ)
  (h1 : ∃ (x : ℝ), α + beta = x ∧ x > 0)
  (h2 : ∃ (y : ℝ), complex.I * (α - 3 * beta) = y ∧ y > 0) :
  α = 12 - 3 * complex.I :=
sorry

end alpha_value_l682_682247


namespace isosceles_triangle_angle_l682_682128

theorem isosceles_triangle_angle (AC AB : ℝ) (angle_A : ℝ) (h1 : AC = AB) (h2 : angle_A = 70) : 
  let angle_B := (180 - angle_A) / 2
  in angle_B = 55 := 
by
  -- Definitions based directly on conditions
  have h3 : angle_B = (180 - 70) / 2, from congr_arg (λ angle_A, (180 - angle_A) / 2) h2,
  show angle_B = 55, by sorry

end isosceles_triangle_angle_l682_682128


namespace coffee_price_increase_l682_682689

variable (C : ℝ) -- cost per pound of green tea and coffee in June
variable (P_green_tea_july : ℝ := 0.1) -- price of green tea per pound in July
variable (mixture_cost : ℝ := 3.15) -- cost of mixture of equal quantities of green tea and coffee for 3 lbs
variable (green_tea_cost_per_lb_july : ℝ := 0.1) -- cost per pound of green tea in July
variable (green_tea_weight : ℝ := 1.5) -- weight of green tea in the mixture in lbs
variable (coffee_weight : ℝ := 1.5) -- weight of coffee in the mixture in lbs
variable (coffee_cost_per_lb_july : ℝ := 2.0) -- cost per pound of coffee in July

theorem coffee_price_increase :
  C = 1 → mixture_cost = 3.15 →
  P_green_tea_july * C = green_tea_cost_per_lb_july →
  green_tea_weight * green_tea_cost_per_lb_july + coffee_weight * coffee_cost_per_lb_july = mixture_cost →
  (coffee_cost_per_lb_july - C) / C * 100 = 100 :=
by
  intros
  sorry

end coffee_price_increase_l682_682689


namespace w_pow_six_eq_neg_one_l682_682234

-- Given condition
def w : ℂ := (complex.sqrt 3 + complex.I) / 2

-- Proof statement
theorem w_pow_six_eq_neg_one : w^6 = -1 :=
by sorry

end w_pow_six_eq_neg_one_l682_682234


namespace smallest_positive_multiple_of_45_l682_682956

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682956


namespace find_angle_A_find_perimeter_l682_682588

noncomputable def cos_rule (b c a : ℝ) (h : b^2 + c^2 - a^2 = b * c) : ℝ :=
(b^2 + c^2 - a^2) / (2 * b * c)

theorem find_angle_A (A B C : ℝ) (a b c : ℝ)
  (h1 : b^2 + c^2 - a^2 = b * c) (hA : cos_rule b c a h1 = 1 / 2) :
  A = Real.arccos (1 / 2) :=
by sorry

theorem find_perimeter (a b c : ℝ)
  (h_a : a = Real.sqrt 2) (hA : Real.sin (Real.arccos (1 / 2))^2 = (Real.sqrt 3 / 2)^2)
  (hBC : Real.sin (Real.arccos (1 / 2))^2 = Real.sin (Real.arccos (1 / 2)) * Real.sin (Real.arccos (1 / 2)))
  (h_bc : b * c = 2)
  (h_bc_eq : b^2 + c^2 - a^2 = b * c) :
  a + b + c = 3 * Real.sqrt 2 :=
by sorry

end find_angle_A_find_perimeter_l682_682588


namespace smallest_positive_multiple_of_45_l682_682874

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682874


namespace bottles_produced_l682_682986

def rate_of_one_machine (r : ℕ) : Prop :=
  6 * r = 270

theorem bottles_produced (r : ℕ) [rate_of_one_machine r] :
  8 * r * 4 = 1440 :=
by
  sorry

end bottles_produced_l682_682986


namespace smallest_positive_multiple_of_45_l682_682962

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682962


namespace root_in_interval_2_3_l682_682696

noncomputable def f (x : ℝ) : ℝ := -|x - 5| + 2^(x - 1)

theorem root_in_interval_2_3 :
  (f 2) * (f 3) < 0 → ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 := by sorry

end root_in_interval_2_3_l682_682696


namespace smallest_positive_multiple_of_45_l682_682965

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682965


namespace value_of_f_neg_a_l682_682326

variable (f : ℝ → ℝ)
variable (a : ℝ)

def f_def : f = (λ x : ℝ, x^3 + Real.sin x + 1) := sorry
def f_at_a_two : f a = 2 := sorry

theorem value_of_f_neg_a : f (-a) = 0 :=
by
  rw [f_def, f_at_a_two]
  sorry

end value_of_f_neg_a_l682_682326


namespace percentile_60_of_dataset_l682_682413

def dataset : List ℕ := [12, 11, 10, 20, 23, 28, 36, 36, 31, 24, 23, 19]

def sorted_dataset : List ℕ := [10, 11, 12, 19, 20, 23, 23, 24, 28, 31, 36, 36]

theorem percentile_60_of_dataset : (sorted_dataset.nth 7) = some 24 := by
  sorry

end percentile_60_of_dataset_l682_682413


namespace digits_sum_2_5_l682_682700

def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

def digits (a b : ℕ) : ℕ := Nat.floor (1 + b * log10 a)

theorem digits_sum_2_5 :
  digits 2 1997 + digits 5 1997 = 1998 := by
  sorry

end digits_sum_2_5_l682_682700


namespace ratio_of_graduate_to_non_graduate_l682_682200

variable (G C N : ℕ)

theorem ratio_of_graduate_to_non_graduate (h1 : C = (2:ℤ)*N/(3:ℤ))
                                         (h2 : G.toRat / (G + C) = 0.15789473684210525) :
  G.toRat / N.toRat = 1 / 8 :=
sorry

end ratio_of_graduate_to_non_graduate_l682_682200


namespace problem_statement_l682_682006

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end problem_statement_l682_682006


namespace number_of_divisors_of_36_l682_682551

theorem number_of_divisors_of_36 :  
  let n := 36
  number_of_divisors n = 9 := 
by 
  sorry

end number_of_divisors_of_36_l682_682551


namespace foxes_invaded_l682_682594

theorem foxes_invaded (initial_weasels : ℕ) (initial_rabbits : ℕ)
  (foxes_weasels_per_week : ℕ) (foxes_rabbits_per_week : ℕ)
  (weeks : ℕ) (remaining_rodents : ℕ) :
  initial_weasels = 100 →
  initial_rabbits = 50 →
  foxes_weasels_per_week = 4 →
  foxes_rabbits_per_week = 2 →
  weeks = 3 →
  remaining_rodents = 96 →
  let initial_rodents := initial_weasels + initial_rabbits,
      total_rodents_caught := initial_rodents - remaining_rodents,
      rodents_per_fox_per_week := foxes_weasels_per_week + foxes_rabbits_per_week,
      rodents_per_fox := rodents_per_fox_per_week * weeks in
  F = total_rodents_caught / rodents_per_fox →
  F = 3 :=
begin
  -- proof here
  sorry
end

end foxes_invaded_l682_682594


namespace exists_set_S_l682_682257

theorem exists_set_S (n : ℕ) (hn : n ≥ 3) :
  ∃ (S : Finset ℕ), S.card = 2 * n ∧ 
  (∀ m : ℕ, 2 ≤ m ∧ m ≤ n → 
    ∃ (A : Finset ℕ) (B : Finset ℕ),
      A ∪ B = S ∧ A ∩ B = ∅ ∧ A.card = m ∧ A.sum = B.sum) :=
begin
  sorry
end

end exists_set_S_l682_682257


namespace four_diff_digits_per_day_l682_682320

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end four_diff_digits_per_day_l682_682320


namespace number_of_lineups_with_C_between_A_and_B_l682_682489

-- Definitions based on the conditions
def students : Type := {A, B, C, D, E}
def is_between (C A B : students) (perm : List students) : Prop :=
  ∃ l1 l2 l3, perm = l1 ++ [A] ++ l2 ++ [C] ++ l3 ++ [B] ∨ perm = l1 ++ [B] ++ l2 ++ [C] ++ l3 ++ [A]

-- Statement of the proof problem
theorem number_of_lineups_with_C_between_A_and_B :
  ∃ perm : List students, is_between C A B perm → 
  (List.permutations [A, B, C, D, E]).length = 40 := 
by
  sorry

end number_of_lineups_with_C_between_A_and_B_l682_682489


namespace tangent_sphere_radius_l682_682720

theorem tangent_sphere_radius (R r : ℝ) 
  (h0 : 0 < R) 
  (h1 : Three tangent spheres of radius \( R \) are tangent to one another and to a plane) 
  (h2 : A fourth sphere of radius \( r \) is tangent to the three spheres and to the same plane) : 
  r = R / 3 := 
sorry

end tangent_sphere_radius_l682_682720


namespace correct_statement_is_D_l682_682379

-- Definitions of the conditions from part a)
def cond_A := "In order to understand the time spent by students in the school on doing math homework, Xiaoming surveyed 3 friends online."
def cond_B := "In order to understand the condition of the components of the 'Fengyun-3 G satellite,' the inspectors used a sampling survey."
def cond_C := "In order to understand the sleep time of children and teenagers nationwide, statisticians conducted a comprehensive survey."
def cond_D := "Given a set of data consisting of integers, where the maximum data is 42 and the minimum data is 8, if the class interval is 5, then the data should be divided into 7 groups."

-- The proof problem
theorem correct_statement_is_D : cond_A → cond_B → cond_C → cond_D → "D is the correct answer to the question 'Which of the following statements is correct?' for given conditions." :=
by
  sorry

end correct_statement_is_D_l682_682379


namespace max_elements_in_a_set_l682_682255

def is_positive (z : ℤ) : Prop :=
z > 0

def a_seq (a1 : ℤ) (n : ℕ) : ℤ :=
nat.rec_on n a1 λ n an, if an ≤ 18 then 2 * an else 2 * an - 36

def a_set (a1 : ℤ) : set ℤ :=
{an | ∃ n : ℕ, an = a_seq a1 n}

theorem max_elements_in_a_set (a1 : ℤ) (h1 : is_positive a1) (h2 : a1 ≤ 18) :
  ∃ (M : set ℤ), (a_set a1 = M) ∧ (∀ (N : set ℤ), (a_set a1 = N) → N.card ≤ 8) :=
sorry

end max_elements_in_a_set_l682_682255


namespace smallest_positive_multiple_of_45_l682_682747

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682747


namespace smallest_b_gt_4_perfect_square_l682_682739

theorem smallest_b_gt_4_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ k : ℕ, 4 * b + 5 = k^2 ∧ b = 5 :=
by
  sorry

end smallest_b_gt_4_perfect_square_l682_682739


namespace area_equality_of_BECD_and_ABC_l682_682627

variables {A B C O D E : Type} [metric_space A B C O D E]

-- Defining the circumcenter and the circumcircle
variable {circumcenter : A}
variable {circumcircle : A → B}

-- Defining the conditions in the problem
def conditions : Prop :=
  let BO := line_through B circumcenter in
  let AO_perpendicular := is_altitude A B C in
  BO.meets_circumcircle_at B D ∧ extended AO_perpendicular.meets_circumcircle_at A E

-- Define the areas
def area_quad_BECD (B E C D : A) : ℝ := 
  sorry -- Placeholder for area calculation of quadrilateral BECD

def area_triangle_ABC (A B C : A) : ℝ := 
  sorry -- Placeholder for area calculation of triangle ABC

-- The theorem to be proved
theorem area_equality_of_BECD_and_ABC 
  (h : conditions) : 
  area_quad_BECD B E C D = area_triangle_ABC A B C :=
sorry

end area_equality_of_BECD_and_ABC_l682_682627


namespace intersection_A_B_l682_682546

def A := {x : ℤ | ∃ k : ℤ, x = 2 * k + 1}
def B := {x : ℤ | 0 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {1, 3} :=
by
  sorry

end intersection_A_B_l682_682546


namespace roses_cut_l682_682363

theorem roses_cut :
  ∀ (i f : ℕ), i = 2 → f = 23 → f - i = 21 :=
by
  intros i f h1 h2
  rw [h1, h2]
  rfl

end roses_cut_l682_682363


namespace set_inter_complement_eq_l682_682265

-- Given conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 < 1}
def B : Set ℝ := {x | x^2 - 2 * x > 0}

-- Question translated to proof problem statement
theorem set_inter_complement_eq :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end set_inter_complement_eq_l682_682265


namespace parallel_lines_distance_l682_682445

open Real

def a : ℝ × ℝ := (3, -4)
def b : ℝ × ℝ := (-1, 1)
def d : ℝ × ℝ := (2, -5)

noncomputable def distance_between_parallel_lines (a b d : ℝ × ℝ) : ℝ :=
  let v : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
  let dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
  let proj_d_v : ℝ × ℝ := (dot_product v d / dot_product d d) * d
  let c : ℝ × ℝ := (v.1 - proj_d_v.1, v.2 - proj_d_v.2)
  sqrt ((c.1 * c.1) + (c.2 * c.2))

theorem parallel_lines_distance :
  distance_between_parallel_lines a b d = 150 * sqrt 2 / 29 :=
by
  sorry

end parallel_lines_distance_l682_682445


namespace find_triples_l682_682386

theorem find_triples (x y z : ℝ) :
  (x + 1)^2 = x + y + 2 ∧
  (y + 1)^2 = y + z + 2 ∧
  (z + 1)^2 = z + x + 2 ↔ (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end find_triples_l682_682386


namespace num_divisors_36_l682_682554

theorem num_divisors_36 : ∃ n, n = 9 ∧ ∀ d : ℕ, d ∣ 36 → (d > 0 ∧ d ≤ 36) → ∃ k : ℕ, d = (2^k or 3^k or (2^p * 3^q) where p, q <= 2):
begin
  have factorization : 36 = 2^2 * 3^2 := by norm_num,
  have exponents : (2+1)*(2+1) = 9 := by norm_num,
  use 9,
  split,
  { exact exponents },
  {
    intros d hd hpos_range,
    cases hpos_range with hpos hrange,
    sorry -- Proof showing that there are exactly 9 positive divisors.
  },
end

end num_divisors_36_l682_682554


namespace hyperbolas_same_asymptotes_l682_682695

noncomputable def hyperbola_asymptote_x (a b : ℝ) : Set (ℝ × ℝ) :=
{ p | p.2 = (b / a) * p.1 ∨ p.2 = -(b / a) * p.1 }

noncomputable def hyperbola_asymptote_y (a b h : ℝ) : Set (ℝ × ℝ) :=
{ p | p.2 = (a / b) * (p.1 - h) ∨ p.2 = -(a / b) * (p.1 - h) }

def same_asymptotes (as1 as2 : Set (ℝ × ℝ)) : Prop :=
as1 = as2

theorem hyperbolas_same_asymptotes :
  ∃ M : ℝ, 
    let asymptotes1 := hyperbola_asymptote_x 3 4,
        asymptotes2 := hyperbola_asymptote_y 5 (Real.sqrt M) 4 in
    same_asymptotes asymptotes1 asymptotes2 ∧ M = 225 / 16 :=
begin
  use 225 / 16,
  have h1 : asymptotes1 = asymptotes2, sorry,
  exact ⟨h1, rfl⟩
end

end hyperbolas_same_asymptotes_l682_682695


namespace smallest_positive_multiple_of_45_l682_682884

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682884


namespace conjugate_quadrant_l682_682597

noncomputable def z : ℂ := (1 / (1 + complex.I)) - (complex.I ^ 3)

theorem conjugate_quadrant :
  ((complex.conj z).re > 0) ∧ ((complex.conj z).im < 0) :=
begin
  -- Proof here
  sorry,
end

end conjugate_quadrant_l682_682597


namespace train_crosses_bridge_in_12_5_seconds_l682_682182

def length_of_train : ℝ := 110
def speed_of_train_kmh : ℝ := 72
def length_of_bridge : ℝ := 140

def speed_of_train_ms : ℝ :=
  speed_of_train_kmh * (1000 / 3600)

def total_distance : ℝ :=
  length_of_train + length_of_bridge

def time_to_cross_bridge : ℝ :=
  total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_12_5_seconds :
  time_to_cross_bridge = 12.5 := by
  sorry

end train_crosses_bridge_in_12_5_seconds_l682_682182


namespace num_divisors_36_l682_682556

theorem num_divisors_36 : ∃ n, n = 9 ∧ ∀ d : ℕ, d ∣ 36 → (d > 0 ∧ d ≤ 36) → ∃ k : ℕ, d = (2^k or 3^k or (2^p * 3^q) where p, q <= 2):
begin
  have factorization : 36 = 2^2 * 3^2 := by norm_num,
  have exponents : (2+1)*(2+1) = 9 := by norm_num,
  use 9,
  split,
  { exact exponents },
  {
    intros d hd hpos_range,
    cases hpos_range with hpos hrange,
    sorry -- Proof showing that there are exactly 9 positive divisors.
  },
end

end num_divisors_36_l682_682556


namespace smallest_positive_multiple_of_45_l682_682760

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682760


namespace share_total_l682_682424

theorem share_total (A B C : ℕ) (A_ratio B_ratio C_ratio : ℕ) (Amanda_share : ℕ) 
  (h_ratio : A_ratio = 2) (h_ratio_B : B_ratio = 3) (h_ratio_C : C_ratio = 8) (h_A_share : Amanda_share = 30) 
  (h_A : A = Amanda_share / A_ratio) : A_ratio * A + B_ratio * A + C_ratio * (A) = 195 :=
by
  have hA : A = 15 := by sorry
  have hB : B = 3 * A := by sorry
  have hC : C = 8 * A := by sorry
  have h_sum : 2 * A + 3 * A + 8 * A = 13 * A := by sorry
  rw [hA],
  exact h_ratio_C

end share_total_l682_682424


namespace machine_value_after_two_years_l682_682388

def machine_value (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - rate)^years

theorem machine_value_after_two_years :
  machine_value 8000 0.1 2 = 6480 :=
by
  sorry

end machine_value_after_two_years_l682_682388


namespace relative_magnitude_of_reciprocal_l682_682578

theorem relative_magnitude_of_reciprocal 
  (a b : ℝ) (hab : a < 1 / b) :
  (a > 0 ∧ b > 0 ∧ 1 / a > b) ∨ (a < 0 ∧ b < 0 ∧ 1 / a > b)
   ∨ (a > 0 ∧ b < 0 ∧ 1 / a < b) ∨ (a < 0 ∧ b > 0 ∧ 1 / a < b) :=
by sorry

end relative_magnitude_of_reciprocal_l682_682578


namespace alpha_value_l682_682245

noncomputable def beta : ℂ := 4 + 3 * complex.I

theorem alpha_value (α : ℂ)
  (h1 : ∃ (x : ℝ), α + beta = x ∧ x > 0)
  (h2 : ∃ (y : ℝ), complex.I * (α - 3 * beta) = y ∧ y > 0) :
  α = 12 - 3 * complex.I :=
sorry

end alpha_value_l682_682245


namespace nina_not_taller_than_lena_l682_682084

noncomputable def friends_heights := ℝ 
variables (M N L O : friends_heights)

def nina_shorter_than_masha (N M : friends_heights) : Prop := N < M
def lena_taller_than_olya (L O : friends_heights) : Prop := L > O
def nina_taller_than_lena (N L : friends_heights) : Prop := N > L

theorem nina_not_taller_than_lena (N M L O : friends_heights) 
  (h₁ : nina_shorter_than_masha N M) 
  (h₂ : lena_taller_than_olya L O) : 
  (0 : ℝ) = 0 :=
sorry

end nina_not_taller_than_lena_l682_682084


namespace problem1_problem2_l682_682442

-- Proof problem 1
theorem problem1 (x : ℝ) : (x - 1)^2 + x * (3 - x) = x + 1 := sorry

-- Proof problem 2
theorem problem2 (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -2) : (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1)) = 1 / (a + 2) := sorry

end problem1_problem2_l682_682442


namespace probability_N_taller_than_L_l682_682094

variable (M N L O : ℕ)

theorem probability_N_taller_than_L 
  (h1 : N < M) 
  (h2 : L > O) : 
  0 = 0 := 
sorry

end probability_N_taller_than_L_l682_682094


namespace rectangle_operation_count_l682_682269

noncomputable def minimum_operations_to_determine_rectangle : ℕ := 9

-- Conditions
variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- We define the distances between points.
variables (dist_AB dist_BC dist_CD dist_DA dist_AC dist_BD : ℝ)

-- We assume the recognition device can measure these distances and compare numbers absolutely.

-- We now state the theorem that the minimum number of operations to determine if quadrilateral ABCD is a rectangle is 9.

theorem rectangle_operation_count (A B C D : Type)
    [metric_space A] [metric_space B] [metric_space C] [metric_space D]
    (dist_AB dist_BC dist_CD dist_DA dist_AC dist_BD : ℝ) :
    minimum_operations_to_determine_rectangle = 9 := sorry

end rectangle_operation_count_l682_682269


namespace senya_mistakes_in_OCTAHEDRON_l682_682279

noncomputable def mistakes_in_word (word : String) : Nat :=
  if word = "TETRAHEDRON" then 5
  else if word = "DODECAHEDRON" then 6
  else if word = "ICOSAHEDRON" then 7
  else if word = "OCTAHEDRON" then 5 
  else 0

theorem senya_mistakes_in_OCTAHEDRON : mistakes_in_word "OCTAHEDRON" = 5 := by
  sorry

end senya_mistakes_in_OCTAHEDRON_l682_682279


namespace g_at_3_l682_682287

def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) : ℝ := 3 - 4 / x

def g (x : ℝ) : ℝ := 2 / (f_inv x) + 7

theorem g_at_3 : g 3 = 8.2 :=
by
  sorry

end g_at_3_l682_682287


namespace smallest_positive_multiple_45_l682_682812

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682812


namespace smallest_positive_multiple_of_45_l682_682857

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682857


namespace smallest_positive_multiple_of_45_l682_682823

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682823


namespace clock_display_four_different_digits_l682_682313

theorem clock_display_four_different_digits :
  (∑ t in finset.range (24*60), if (((t / 60).div1000 ≠ (t / 60).mod1000) ∧ 
    ((t / 60).div1000 ≠ (t % 60).div1000) ∧ ((t / 60).div1000 ≠ (t % 60).mod1000) ∧ 
    ((t / 60).mod1000 ≠ (t % 60).div1000) ∧ ((t / 60).mod1000 ≠ (t % 60).mod1000) ∧ 
    ((t % 60).div1000 ≠ (t % 60).mod1000)) then 1 else 0) = 588 :=
by
  sorry

end clock_display_four_different_digits_l682_682313


namespace smallest_positive_multiple_l682_682782

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682782


namespace hexagon_coloring_l682_682470

def valid_coloring_hexagon : Prop :=
  ∃ (A B C D E F : Fin 8), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ A ≠ D ∧ B ≠ D ∧ C ≠ D ∧
    B ≠ E ∧ C ≠ E ∧ D ≠ E ∧ A ≠ F ∧ C ≠ F ∧ E ≠ F

theorem hexagon_coloring : ∃ (n : Nat), valid_coloring_hexagon ∧ n = 20160 := 
sorry

end hexagon_coloring_l682_682470


namespace p_necessary_not_sufficient_for_q_l682_682132

variable {α : Type*} (f : α → ℝ)

/-- The definition of an even function -/
def is_even (f : α → ℝ) : Prop :=
∀ x, f x = f (-x)

/-- The definition of a monotonic function -/
def is_monotonic (f : α → ℝ) : Prop :=
∀ x y, x ≤ y → f x ≤ f y

/-- The proposition representing \( p \) -/
def p (f : α → ℝ) : Prop := ¬ is_even f

/-- The proposition representing \( q \) -/
def q (f : α → ℝ) : Prop := is_monotonic f

/-- Proving the relationship between \( p \) and \( q \) -/
theorem p_necessary_not_sufficient_for_q (f : α → ℝ) : (p f → q f) ∧ ¬ (q f → p f) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l682_682132


namespace hyperbola_tangent_inequality_l682_682541

-- Defining the hyperbola and its constraints
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Main theorem statement to prove that a ≥ -1/2
theorem hyperbola_tangent_inequality {x0 y0 x1 y1 x2 y2 a : ℝ}
  (M : x0^2 - y0^2 = 1) (P : x1^2 - y1^2 = 1) (Q : x2^2 - y2^2 = 1)
  (in_first_quadrant : 0 < x0 ∧ 0 < y0 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2)
  (tangent_through_M : x0 * x1 - y0 * y1 = 1 ∧ x0 * x2 - y0 * y2 = 1)
  (R : Prop) (RP_RQ_dot_product : ∀ (R : ℝ → ℝ → Prop)(P : ℝ → ℝ → Prop)(Q : ℝ → ℝ → Prop), R → P → Q → ℝ)
  : ∀ (R : ℝ → ℝ → Prop), ∃ a, RP_RQ_dot_product R P Q ≥ -1/2 
  := by
  -- Proof to be done
  sorry

end hyperbola_tangent_inequality_l682_682541


namespace optimal_perimeter_proof_l682_682722

-- Definition of conditions
def fencing_length : Nat := 400
def min_width : Nat := 50
def area : Nat := 8000

-- Definition of the perimeter to be proven as optimal
def optimal_perimeter : Nat := 360

-- Theorem statement to be proven
theorem optimal_perimeter_proof (l w : Nat) (h1 : l * w = area) (h2 : 2 * l + 2 * w <= fencing_length) (h3 : w >= min_width) :
  2 * l + 2 * w = optimal_perimeter :=
sorry

end optimal_perimeter_proof_l682_682722


namespace smallest_positive_multiple_of_45_l682_682839

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l682_682839


namespace factorial_division_l682_682448

theorem factorial_division (n m : ℕ) (h : n = 52) (g : m = 50) : (n! / m!) = 2652 :=
by
  sorry

end factorial_division_l682_682448


namespace simplify_expr1_simplify_expr2_l682_682284

theorem simplify_expr1 (a b : ℝ) : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 (t : ℝ) : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l682_682284


namespace smallest_positive_multiple_of_45_is_45_l682_682953

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682953


namespace unique_average_arc_distance_l682_682653

theorem unique_average_arc_distance {α : ℝ} : 
  ∃! (α : ℝ), (α = π / 2) ∧ ∀ (n : ℕ) (B : list ℝ), B.length = n → ∀ (X : ℝ), X ∈ (list.Icc 0 (2 * π)) → ∃ (Y : ℝ), Y ∈ (list.Icc 0 (2 * π)) ∧ 
  (1 / n) * (list.sum (list.map (λ b_i, min (|X - b_i|) (2 * π - |X - b_i|)) B)) = α := 
sorry

end unique_average_arc_distance_l682_682653


namespace sqrt_expr_approx_l682_682488

theorem sqrt_expr_approx :
  (real.sqrt (11 * 13) * (1 / 3) + 2 * (real.sqrt 17 / 3) - 4 * (real.sqrt 7 / 5)) ≈ 4.618 :=
sorry

end sqrt_expr_approx_l682_682488


namespace smallest_positive_multiple_l682_682787

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l682_682787


namespace smallest_positive_multiple_of_45_l682_682887

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682887


namespace smallest_positive_multiple_of_45_l682_682858

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682858


namespace A_is_infinite_l682_682521

open Set

-- Define the conditions
variable {f : ℝ → ℝ}

-- Condition 1: Inequality condition for function f
axiom function_inequality (x : ℝ) : f^(2 : ℝ) x ≤ 2 * x^2 * f(x / 2)

-- Condition 2: Definition of set A
def A : Set ℝ := {a | f(a) > a^2}

-- Non-empty set A assumption
axiom A_non_empty : ∃ a : ℝ, a ∈ A

-- Statement to prove that A is infinite
theorem A_is_infinite : Infinite A := sorry

end A_is_infinite_l682_682521


namespace smallest_positive_multiple_45_l682_682806

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l682_682806


namespace a_minus_b_perfect_square_l682_682619

theorem a_minus_b_perfect_square (a b : ℕ) (h : 2 * a^2 + a = 3 * b^2 + b) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℕ, a - b = k^2 :=
by sorry

end a_minus_b_perfect_square_l682_682619


namespace no_solution_exists_l682_682043

open Real

theorem no_solution_exists :
  ¬ ∃ (x y z t : ℚ) (n : ℕ), 
    ((x + y * √2) ^ (2 * n : ℕ) + (z + t * √2) ^ (2 * n : ℕ)) = (5 + 4 * √2) :=
by
  sorry 

end no_solution_exists_l682_682043


namespace smallest_positive_multiple_of_45_l682_682759

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l682_682759


namespace angle_between_given_planes_l682_682479

noncomputable def angle_between_planes (n1 n2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.arccos ((n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3) /
    (Real.sqrt (n1.1^2 + n1.2^2 + n1.3^2) * Real.sqrt (n2.1^2 + n2.2^2 + n2.3^2)))

theorem angle_between_given_planes :
  angle_between_planes (3, -2, 3) (0, 1, 1) = Real.arccos (1 / (2 * Real.sqrt 11)) :=
begin
  sorry
end

end angle_between_given_planes_l682_682479


namespace midpoint_E_of_FG_l682_682598

-- Define the geometric setup and prove the assertion
theorem midpoint_E_of_FG
  (ABC : Triangle)
  (I : Point) -- I is the incenter of triangle ABC
  (incircle_I : Circle) -- The circle inscribed in triangle ABC with center I
  (D E F G : Point) -- Points D, E, F, and G
  (h_tangent_D : Tangent incircle_I BC D) -- D is the point where the incircle touches BC
  (h_parallel : Line trough I parallel to (Line.of AD) intersects BC at E)
  (h_tangent_E : Tangent incircle_I E F G) -- Tangent at E intersects AB and AC at F and G respectively
  :
  midpoint E F G :=
begin
  -- The proof will go here
  sorry
end

end midpoint_E_of_FG_l682_682598


namespace coefficient_of_a_neg_one_l682_682215

theorem coefficient_of_a_neg_one :
  let expr := (a - 2 / (Real.sqrt a)) ^ 10 in
  ∃ c : ℝ, (term == a ^ (-1) → c == 11520) :=
begin
  sorry
end

end coefficient_of_a_neg_one_l682_682215


namespace polar_to_cartesian_l682_682707

theorem polar_to_cartesian (ρ θ : ℝ) (x y : ℝ) 
  (hρ : ρ = 2 * cos θ) 
  (hx : x = ρ * cos θ) 
  (hy : y = ρ * sin θ) :
  (x - 1) ^ 2 + y ^ 2 = 1 ∧ x^2 - 2*x + y^2 = 0 ∧ (1 = 1 ∧ 0 = 0) :=
by
  -- To complete the statement moving from polar to Cartesian coordinates
  -- we need a theorem relating the given conditions to the desired outcomes.
  sorry

end polar_to_cartesian_l682_682707


namespace clock_shows_four_different_digits_for_588_minutes_l682_682299

-- Definition of the problem
def isFourDifferentDigits (h1 h2 m1 m2 : Nat) : Bool :=
  (h1 ≠ h2) && (h1 ≠ m1) && (h1 ≠ m2) && (h2 ≠ m1) && (h2 ≠ m2) && (m1 ≠ m2)

noncomputable def countFourDifferentDigitsMinutes : Nat :=
  let validMinutes := List.filter (λ (t : Nat × Nat),
    let (h, m) := t
    let h1 := h / 10
    let h2 := h % 10
    let m1 := m / 10
    let m2 := m % 10
    isFourDifferentDigits h1 h2 m1 m2
  ) (List.product (List.range 24) (List.range 60))
  validMinutes.length

theorem clock_shows_four_different_digits_for_588_minutes :
  countFourDifferentDigitsMinutes = 588 := sorry

end clock_shows_four_different_digits_for_588_minutes_l682_682299


namespace smallest_positive_multiple_of_45_l682_682861

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l682_682861


namespace smallest_positive_multiple_of_45_is_45_l682_682797

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682797


namespace alpha_value_l682_682246

noncomputable def beta : ℂ := 4 + 3 * complex.I

theorem alpha_value (α : ℂ)
  (h1 : ∃ (x : ℝ), α + beta = x ∧ x > 0)
  (h2 : ∃ (y : ℝ), complex.I * (α - 3 * beta) = y ∧ y > 0) :
  α = 12 - 3 * complex.I :=
sorry

end alpha_value_l682_682246


namespace angle_AST_perpendicular_l682_682629

theorem angle_AST_perpendicular (Δ : Triangle) (A B C D T S : Point) (R : ℝ) 
(h1 : Δ.acute) 
(h2 : Δ.AC < Δ.AB) 
(h3 : Δ.circumradius = R) 
(h4 : D.is_foot_of_altitude_from A) 
(h5 : T.is_on_line_AD ∧ A.distance_to T = 2 * R ∧ D.is_between A T)
(h6 : S.is_center_of_arc_BC_not_containing A) :
  ∠AST = 90 :=
sorry

end angle_AST_perpendicular_l682_682629


namespace clock_display_four_different_digits_l682_682315

theorem clock_display_four_different_digits :
  (∑ t in finset.range (24*60), if (((t / 60).div1000 ≠ (t / 60).mod1000) ∧ 
    ((t / 60).div1000 ≠ (t % 60).div1000) ∧ ((t / 60).div1000 ≠ (t % 60).mod1000) ∧ 
    ((t / 60).mod1000 ≠ (t % 60).div1000) ∧ ((t / 60).mod1000 ≠ (t % 60).mod1000) ∧ 
    ((t % 60).div1000 ≠ (t % 60).mod1000)) then 1 else 0) = 588 :=
by
  sorry

end clock_display_four_different_digits_l682_682315


namespace number_of_blue_pens_minus_red_pens_is_seven_l682_682359

-- Define the problem conditions in Lean
variable (R B K T : ℕ) -- where R is red pens, B is black pens, K is blue pens, T is total pens

-- Define the hypotheses from the problem conditions
def hypotheses :=
  (R = 8) ∧ 
  (B = R + 10) ∧ 
  (T = 41) ∧ 
  (T = R + B + K)

-- Define the theorem we need to prove based on the question and the correct answer
theorem number_of_blue_pens_minus_red_pens_is_seven : 
  hypotheses R B K T → K - R = 7 :=
by 
  intro h
  sorry

end number_of_blue_pens_minus_red_pens_is_seven_l682_682359


namespace toothpicks_in_10th_stage_l682_682719

theorem toothpicks_in_10th_stage (n : ℕ) (h_start : n = 1 → 4) (h_subsequent : ∀ k, n = k + 1 → 4 + 3 * k) : 4 + 3 * (10 - 1) = 31 := by
sorry

end toothpicks_in_10th_stage_l682_682719


namespace math_problem_l682_682634

noncomputable def base_change_num (n : Nat) (b : Nat) : Nat := 
  let rec aux n mult acc := 
    match n with 
    | 0 => acc 
    | _ => aux (n / 10) (mult * b) (acc + (n % 10) * mult)
  aux n 1 0

noncomputable def condition (c : Nat) : Prop := 
  base_change_num 13 c * base_change_num 18 c * base_change_num 17 c = base_change_num 4357 c

noncomputable def sum_condition (c : Nat) : Nat := 
  base_change_num 13 c + base_change_num 18 c + base_change_num 17 c 

theorem math_problem : condition 11 → sum_condition 11 = base_change_num 47 11 := 
by
  sorry

end math_problem_l682_682634


namespace simplify_expression_l682_682672

variable {m : ℝ} (hm : m ≠ 0)

theorem simplify_expression (m : ℝ) (h : m ≠ 0) : 
  ( (1 / (3 * m)) ^ (-3) * (2 * m) ^ 4 ) = 432 * m ^ 7 := by
  sorry

end simplify_expression_l682_682672


namespace find_divisor_l682_682987

theorem find_divisor :
  ∃ D : ℕ, (∃ q1 : ℕ, 242 = D * q1 + 6) ∧ 
           (∃ q2 : ℕ, 698 = D * q2 + 13) ∧ 
           (∃ q3 : ℕ, 940 = D * q3 + 5) ∧ 
           D = 14 :=
begin
  sorry
end

end find_divisor_l682_682987


namespace exists_intersecting_line_l682_682593

open Set

variable {P : Type*}

structure ConvexPolygon (P : Type*) extends Set P :=
(convex : convex_hull P)

variable (P1 P2: Finset (ConvexPolygon 𝕜 P))

-- Conditions of the problem
axiom common_point {p1 : ConvexPolygon 𝕜 P} {p2 : ConvexPolygon 𝕜 P} (h1 : p1 ∈ P1) (h2 : p2 ∈ P2) : 
  ∃ x, (x ∈ p1) ∧ (x ∈ p2)

axiom non_overlapping_in_P1 : ∃ p1 q1: ConvexPolygon 𝕜 P, (p1 ∈ P1) ∧ (q1 ∈ P1) ∧ (Disjoint p1.carrier q1.carrier)
axiom non_overlapping_in_P2 : ∃ p2 q2: ConvexPolygon 𝕜 P, (p2 ∈ P2) ∧ (q2 ∈ P2) ∧ (Disjoint p2.carrier q2.carrier)

-- Statement that needs to be proved
theorem exists_intersecting_line : ∃ L: Set P, ∀ p1 ∈ P1, ∀ p2 ∈ P2, 
  (L ∩ p1.carrier ≠ ∅) ∧ (L ∩ p2.carrier ≠ ∅) :=
sorry

end exists_intersecting_line_l682_682593


namespace clock_four_different_digits_l682_682305

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end clock_four_different_digits_l682_682305


namespace remainder_when_dividing_f_by_x_plus_1_l682_682496

noncomputable theory

def f (x : ℝ) : ℝ := x^8 + 3

theorem remainder_when_dividing_f_by_x_plus_1 : 
  (x : ℝ) (rem : ℝ) (h : rem = 4) : 
  rem = f (-1) :=
by
  sorry

end remainder_when_dividing_f_by_x_plus_1_l682_682496


namespace sum_of_coefficients_sum_of_fractions_weighted_sum_l682_682116

noncomputable def expansion := 
  λ (x : ℝ), (1 - 2 * x) ^ 2021

theorem sum_of_coefficients : 
  let coefficients := (λ (a : ℕ → ℝ), a 0 + a 1 + a 2 + ... + a 2021)
  coefficients (λ (n : ℕ), (expansion x).coeff n) = -1 :=
sorry

theorem sum_of_fractions : 
  let term := (λ (a : ℕ → ℝ), 
    ∑ n in (finset.range 2021), (a n * 1 / 2^n))
  term (λ (n : ℕ), (expansion x).coeff n) = -1 :=
sorry

theorem weighted_sum : 
  let weighted := (λ (a : ℕ → ℝ), 
    ∑ n in (finset.range 2021), ((n+1) * a n))
  weighted (λ (n : ℕ), (expansion x).coeff n) = -4042 :=
sorry

end sum_of_coefficients_sum_of_fractions_weighted_sum_l682_682116


namespace probability_of_at_least_one_defective_l682_682383

noncomputable def prob_at_least_one_defective : ℚ :=
  let total_bulbs := 20
  let defective_bulbs := 4
  let non_defective_bulbs := total_bulbs - defective_bulbs in
  let prob_first_non_defective := non_defective_bulbs / total_bulbs
  let prob_second_non_defective := (non_defective_bulbs - 1) / (total_bulbs - 1) in
  let prob_both_non_defective := prob_first_non_defective * prob_second_non_defective in
  1 - prob_both_non_defective

theorem probability_of_at_least_one_defective :
  prob_at_least_one_defective = 7 / 19 :=
sorry

end probability_of_at_least_one_defective_l682_682383


namespace arithmetic_sequence_problem_l682_682520

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h1 : a 2 + a 3 + a 4 = 15)
  (h2 : (a 1 + 2) * (a 6 + 16) = (a 3 + 4) ^ 2)
  (h_positive : ∀ n, 0 < a n) :
  a 10 = 19 :=
sorry

end arithmetic_sequence_problem_l682_682520


namespace min_perimeter_of_8_sided_polygon_with_zeros_of_Q_l682_682628

noncomputable def Q (z : Complex) : Complex := z^8 + (8 * Complex.sqrt 2 + 12) * z^4 - (8 * Complex.sqrt 2 + 10)

theorem min_perimeter_of_8_sided_polygon_with_zeros_of_Q :
  let zeros := {z : Complex | Q z = 0}
  let perimeter (vertices : Finset Complex) : ℝ :=
    vertices.sum (λ z, Complex.abs (z - Complex.conj z))
  ∃ (vertices : Finset Complex), vertices.card = 8 ∧ (∀ z ∈ vertices, Q z = 0) ∧
  (∀ v, (v.card = 8 ∧ (∀ z ∈ v, Q z = 0)) → perimeter vertices ≤ perimeter v) ∧
  perimeter vertices = 8 * Complex.sqrt 2 :=
sorry

end min_perimeter_of_8_sided_polygon_with_zeros_of_Q_l682_682628


namespace smallest_positive_multiple_of_45_is_45_l682_682792

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682792


namespace ortho_proj_magnitude_sqrt5_l682_682176

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (1, -2)

-- Function to compute the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to compute the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Function to compute the orthogonal projection of vector a onto vector b
def orthogonal_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_ab := dot_product a b
  let mag_b2 := (magnitude b) ^ 2
  let scalar := dot_ab / mag_b2
  (scalar * b.1, scalar * b.2)

-- Function to compute the magnitude of the orthogonal projection
def magnitude_of_projection (a b : ℝ × ℝ) : ℝ :=
  magnitude (orthogonal_projection a b)

-- The theorem that states the magnitude of the orthogonal projection is sqrt(5)
theorem ortho_proj_magnitude_sqrt5 :
  magnitude_of_projection (3, -1) (1, -2) = Real.sqrt 5 :=
  sorry

end ortho_proj_magnitude_sqrt5_l682_682176


namespace find_fifth_integer_l682_682490

theorem find_fifth_integer (x y : ℤ) (h_pos : x > 0)
  (h_mean_median : (x + 2 + x + 7 + x + y) / 5 = x + 7) :
  y = 22 :=
sorry

end find_fifth_integer_l682_682490


namespace limes_remaining_l682_682463

-- Definitions based on conditions
def initial_limes : ℕ := 9
def limes_given_to_Sara : ℕ := 4

-- Theorem to prove
theorem limes_remaining : initial_limes - limes_given_to_Sara = 5 :=
by
  -- Sorry keyword to skip the actual proof
  sorry

end limes_remaining_l682_682463


namespace range_f_g_positive_l682_682153

def f : ℝ → ℝ := sorry -- given f is an odd function defined on ℝ
def g : ℝ → ℝ := sorry -- given g is an even function defined on ℝ

theorem range_f_g_positive :
  (∀ x, f(-x) = -f(x)) → -- f is odd
  (∀ x < 0, ∀ y < 0, x < y → f(x) > f(y)) → -- f is monotonically decreasing on (-∞, 0)
  (∀ x, g(-x) = g(x)) → -- g is even
  (∀ x ≤ 0, ∀ y ≤ 0, x < y → g(x) < g(y)) → -- g is monotonically increasing on (-∞, 0]
  (f 1 = 0) → -- given f(1) = 0
  (g 1 = 0) → -- given g(1) = 0
  {x : ℝ | f(x) * g(x) > 0} = { x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x) } :=
by
  sorry

end range_f_g_positive_l682_682153


namespace dwarf_diamond_distribution_l682_682471

-- Definitions for conditions
def dwarves : Type := Fin 8
structure State :=
  (diamonds : dwarves → ℕ)

-- Initial condition: Each dwarf has 3 diamonds
def initial_state : State := 
  { diamonds := fun _ => 3 }

-- Transition function: Each dwarf divides diamonds into two piles and passes them to neighbors
noncomputable def transition (s : State) : State := sorry

-- Proof goal: At a certain point in time, 3 specific dwarves have 24 diamonds in total,
-- with one dwarf having 7 diamonds, then prove the other two dwarves have 12 and 5 diamonds.
theorem dwarf_diamond_distribution (s : State)
  (h1 : ∃ t, s = (transition^[t]) initial_state ∧ ∃ i j k : dwarves, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    s.diamonds i + s.diamonds j + s.diamonds k = 24 ∧
    s.diamonds i = 7)
  : ∃ a b : dwarves, a ≠ b ∧ s.diamonds a = 12 ∧ s.diamonds b = 5 := sorry

end dwarf_diamond_distribution_l682_682471


namespace find_a_l682_682263

-- Definitions based on conditions
def xi_distribution := NormalDist 3 2

-- Statement of the theorem
theorem find_a (a : ℝ) : 
  (μ (set_of (λ ω, xi_distribution.pdf ω < 2 * a - 3)) = μ (set_of (λ ω, xi_distribution.pdf ω > a + 2))) → 
  a = 7 / 3 :=
sorry

end find_a_l682_682263


namespace pencils_purchased_l682_682396

theorem pencils_purchased 
  (total_cost : ℝ)
  (num_pens : ℕ)
  (price_per_pen : ℝ)
  (price_per_pencil : ℝ)
  (total_cost_condition : total_cost = 510)
  (num_pens_condition : num_pens = 30)
  (price_per_pen_condition : price_per_pen = 12)
  (price_per_pencil_condition : price_per_pencil = 2) :
  num_pens * price_per_pen + sorry = total_cost →
  150 / price_per_pencil = 75 :=
by
  sorry

end pencils_purchased_l682_682396


namespace smallest_positive_multiple_of_45_is_45_l682_682912

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l682_682912


namespace smallest_positive_multiple_of_45_l682_682831

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l682_682831


namespace taxi_problem_l682_682400

-- Defining the list of distances recorded
def distances : List Int := [+9, -3, -5, +4, +8, +6, +3, -6, -4, +10]

-- Calculating the total displacement (sum of distances)
def total_displacement : Int := distances.foldl (· + ·) 0

-- Calculating the total revenue
def total_revenue (price_per_km : Float) : Float :=
  distances.foldl (λ acc x => acc + Float.ofInt (Int.natAbs x)) 0.0 * price_per_km

-- Defining the theorem that encapsulates the proof problem
theorem taxi_problem
  (h1 : total_displacement = 22)
  (h2 : total_revenue 2.4 = 139.2)
  : true := by
  sorry

end taxi_problem_l682_682400


namespace union_of_A_and_B_l682_682545

def set_A : Set Int := {0, 1}
def set_B : Set Int := {0, -1}

theorem union_of_A_and_B : set_A ∪ set_B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l682_682545


namespace smallest_positive_multiple_of_45_l682_682959

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l682_682959


namespace smallest_positive_multiple_45_l682_682903

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l682_682903


namespace correct_statements_count_l682_682134

theorem correct_statements_count (a b : ℝ)
  (h : (a - Real.sqrt (a^2 - 1)) * (b - Real.sqrt (b^2 - 1)) = 1) :
  2 = ([a = b, a + b = 0, a * b = 1, a * b = -1].count (λ s, s)) :=
sorry

end correct_statements_count_l682_682134


namespace length_AB_is_2sqrt3_l682_682526

open Real

-- Definitions of circle C and line l, point A
def circle_C := {x : ℝ × ℝ | (x.1 - 3)^2 + (x.2 + 1)^2 = 1}
def line_l (k : ℝ) := {p : ℝ × ℝ | k * p.1 + p.2 - 2 = 0}
def point_A (k : ℝ) := (0, k)

-- Conditions: line l passes through the center of the circle and is the axis of symmetry
def is_axis_of_symmetry_l (k : ℝ) := ∀ p: ℝ × ℝ, p ∈ circle_C → line_l k p

-- Main theorem to be proved
theorem length_AB_is_2sqrt3 (k : ℝ) (h_sym: is_axis_of_symmetry_l k) : 
  let A := point_A 1 in 
  let C := (3, -1) in 
  let radius := 1 in 
  let AC := sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) in
  sqrt (AC^2 - radius^2) = 2 * sqrt 3 :=
sorry -- proof not required

end length_AB_is_2sqrt3_l682_682526


namespace roots_reciprocal_sum_eq_25_l682_682323

theorem roots_reciprocal_sum_eq_25 (p q r : ℝ) (hpq : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) (hroot : ∀ x, x^3 - 9*x^2 + 8*x + 2 = 0 → (x = p ∨ x = q ∨ x = r)) :
  1/p^2 + 1/q^2 + 1/r^2 = 25 :=
by sorry

end roots_reciprocal_sum_eq_25_l682_682323


namespace smallest_positive_multiple_of_45_l682_682891

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l682_682891


namespace radius_of_circle_zero_l682_682039

theorem radius_of_circle_zero (x y : ℝ) :
    (x^2 + 4*x + y^2 - 2*y + 5 = 0) → 0 = 0 :=
by
  sorry

end radius_of_circle_zero_l682_682039


namespace return_trip_time_l682_682406

noncomputable theory

variable {d p w : ℝ}

theorem return_trip_time :
  (∀ d p w, (d = 120 * (p - w)) →
    ∃ t, t = ⟪time_for_return_trip⟫ ∧
    (t = 60 ∨ t = 40)) :=
by
  intro d p w h
  have h1 : d = 120 * (p - w) := h
  have h2 : (d / (p + w)) = (d / p) - 20 := sorry
  let e1 := 20 * p^2 - 140 * p * w + 120 * w^2 = 0 : sorry
  let e2 := (2 * p - 6 * w) * (5 * p - 10 * w) = 0 : sorry
  cases eq_or_ne (2 * p - 6 * w) 0 with h3 h3; rw h3 at *
  { existsi (120 * (3 * w - w)) / (3 * w + w)
    simp }
  {
  existsi (120 * (2 * w - w)) / (2 * w + w)
  simp }
  have e1 := (p = 3*w) ∨ (p = 2*w) : sorry
  cases e1;
  { existsi 60; simp }
  { existsi 40; simp }

end return_trip_time_l682_682406


namespace sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l682_682972

theorem sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30 :
  (7^30 + 13^30) % 100 = 0 := 
sorry

end sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l682_682972


namespace margarita_vs_ricciana_l682_682656

-- Ricciana's distances
def ricciana_run : ℕ := 20
def ricciana_jump : ℕ := 4
def ricciana_total : ℕ := ricciana_run + ricciana_jump

-- Margarita's distances
def margarita_run : ℕ := 18
def margarita_jump : ℕ := (2 * ricciana_jump) - 1
def margarita_total : ℕ := margarita_run + margarita_jump

-- Statement to prove Margarita ran and jumped 1 more foot than Ricciana
theorem margarita_vs_ricciana : margarita_total = ricciana_total + 1 := by
  sorry

end margarita_vs_ricciana_l682_682656
