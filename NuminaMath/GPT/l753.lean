import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Fraction
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Theory
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Algebra.GraphTheory
import Real.Basic
import algebra.quadratic_discriminant

namespace intersection_A_B_l753_753661

namespace proof_example

def A : Set ℝ := {x | x^2 - 3*x ≤ 4}
def B : Set ℝ := {x | 2^x > 2}
def C : Set ℝ := {x | 1 < x ∧ x ≤ 4}

theorem intersection_A_B : A ∩ B = C := by
  sorry

end proof_example

end intersection_A_B_l753_753661


namespace smallest_b_theorem_l753_753759

open Real

noncomputable def smallest_b (a b c: ℝ) (h1: b > 0) (h2: a = b / r) (h3: c = b * r) (h4: a * b * c = 125) : Prop :=
  b = 5

theorem smallest_b_theorem (a b c: ℝ) (r: ℝ) (h1: b > 0) (h2: a = b / r) (h3: c = b * r) (h4: a * b * c = 125) :
  smallest_b a b c h1 h2 h3 h4 :=
by {
  sorry
}

end smallest_b_theorem_l753_753759


namespace marked_price_is_42_50_l753_753915

-- Define the initial cost and discount conditions
def initial_cost : ℝ := 36
def discount_rate : ℝ := 0.15
def cost_after_discount (initial_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  initial_cost * (1 - discount_rate)

-- Define the profit conditions and desired selling price
def profit_rate : ℝ := 0.25
def desired_selling_price (cost_after_discount : ℝ) (profit_rate : ℝ) : ℝ :=
  cost_after_discount * (1 + profit_rate)

-- Define the discount on the marked price and the equation to find marked price
def discount_on_mp_rate : ℝ := 0.10
def marked_price (desired_selling_price : ℝ) (discount_on_mp_rate : ℝ) : ℝ :=
  desired_selling_price / (1 - discount_on_mp_rate)

-- Prove that the marked price is 42.50 given the initial conditions
theorem marked_price_is_42_50 :
  marked_price (desired_selling_price (cost_after_discount initial_cost discount_rate) profit_rate) discount_on_mp_rate = 42.50 :=
by
  sorry

end marked_price_is_42_50_l753_753915


namespace additional_flour_minus_salt_l753_753443

structure CakeRecipe where
  flour    : ℕ
  sugar    : ℕ
  salt     : ℕ

def MaryHasAdded (cups_flour : ℕ) (cups_sugar : ℕ) (cups_salt : ℕ) : Prop :=
  cups_flour = 2 ∧ cups_sugar = 0 ∧ cups_salt = 0

variable (r : CakeRecipe)

theorem additional_flour_minus_salt (H : MaryHasAdded 2 0 0) : 
  (r.flour - 2) - r.salt = 3 :=
sorry

end additional_flour_minus_salt_l753_753443


namespace units_digit_product_l753_753512

theorem units_digit_product :
  ∃ d : ℕ, d = ((5^2 + 1) * (5^3 + 1) * (5^{23} + 1)) % 10 ∧ d = 6 := 
by
  sorry

end units_digit_product_l753_753512


namespace sum_areas_of_unique_right_triangles_l753_753916

theorem sum_areas_of_unique_right_triangles :
  (∑ (a b : ℕ) (h₁ : a ≤ b) (h₂ : (a - 6) * (b - 6) = 36), (a * b) / 2) = 471 :=
by
  sorry

end sum_areas_of_unique_right_triangles_l753_753916


namespace measure_shared_angle_l753_753033

-- Definitions based on the problem's conditions
def is_regular_polygon (n : ℕ) := Π (vertices : fin n → ℝ × ℝ),
  ∀ i j : fin n, i ≠ j → dist (vertices i) (vertices j) = dist (vertices (i + 1) (vertices j))

def is_equilateral_triangle (triangle : fin 3 → ℝ × ℝ) :=
  is_regular_polygon 3 triangle

def is_regular_pentagon (pentagon : fin 5 → ℝ × ℝ) :=
  is_regular_polygon 5 pentagon

def shared_vertex (triangle : fin 3 → ℝ × ℝ) (pentagon : fin 5 → ℝ × ℝ) (v : ℝ × ℝ) :=
  (triangle 0 = v) ∧ (pentagon 0 = v)

-- Main statement to prove
theorem measure_shared_angle (triangle : fin 3 → ℝ × ℝ) (pentagon : fin 5 → ℝ × ℝ) (v : ℝ × ℝ)
  (h1 : is_equilateral_triangle triangle) (h2 : is_regular_pentagon pentagon) (h3 : shared_vertex triangle pentagon v) :
  angle (triangle 1) v (pentagon 1) = 6 := sorry

end measure_shared_angle_l753_753033


namespace two_true_props_l753_753677

def Prop1 : Prop := ∀ (α β : ℝ), α = β → (∀ (x y : Point), angle x y α = angle y x β)
def Prop2 : Prop := ∀ (p : Parallelogram), p.diagonals_bisect
def Prop3 : Prop := ∀ (p : Parallelogram), (p.diagonals_equal ↔ p.is_square)
def Prop4 : Prop := ∀ (a b c : ℝ), a + b > c ∧ a + c > b ∧ b + c > a

def true_count : ℕ := [Prop1, Prop2, Prop3, Prop4].count (λ p, p = true)

theorem two_true_props : true_count = 2 := by
  sorry

end two_true_props_l753_753677


namespace cricket_initial_overs_l753_753377

theorem cricket_initial_overs
  (x : ℕ)
  (hx1 : ∃ x : ℕ, 0 ≤ x)
  (initial_run_rate : ℝ)
  (remaining_run_rate : ℝ)
  (remaining_overs : ℕ)
  (target_runs : ℕ)
  (H1 : initial_run_rate = 3.2)
  (H2 : remaining_run_rate = 6.25)
  (H3 : remaining_overs = 40)
  (H4 : target_runs = 282) :
  3.2 * (x : ℝ) + 6.25 * 40 = 282 → x = 10 := 
by 
  simp only [H1, H2, H3, H4]
  sorry

end cricket_initial_overs_l753_753377


namespace counterexample_non_prime_implies_prime_l753_753946

open Nat

def is_composite (n : ℕ) : Prop := 
  ¬ (Prime n) ∧ n > 1

theorem counterexample_non_prime_implies_prime :
  ∃ (n : ℕ), (n = 14 ∨ n = 18 ∨ n = 22 ∨ n = 24 ∨ n = 28) ∧ is_composite n ∧ is_composite (n + 2) :=
by
  sorry

end counterexample_non_prime_implies_prime_l753_753946


namespace find_a_l753_753335

-- Define the domains of the functions f and g
def A : Set ℝ :=
  {x | x < -1 ∨ x ≥ 1}

def B (a : ℝ) : Set ℝ :=
  {x | 2 * a < x ∧ x < a + 1}

-- Restate the problem as a Lean proposition
theorem find_a (a : ℝ) (h : a < 1) (hb : B a ⊆ A) :
  a ∈ {x | x ≤ -2 ∨ (1 / 2 ≤ x ∧ x < 1)} :=
sorry

end find_a_l753_753335


namespace reaction_products_and_thermochemistry_l753_753449

theorem reaction_products_and_thermochemistry
  (mass_P : ℝ) (mass_O2 : ℝ) (heat_released : ℝ) (combustion_heat : ℝ)
  (mass_P2O5 : ℝ) (mass_P2O3 : ℝ) (reaction_heat : ℝ)
  (thermochemical_eqn : String) :
  mass_P = 3.1 →
  mass_O2 = 3.2 →
  heat_released = X →
  combustion_heat = Y →
  mass_P2O5 = 3.55 →
  mass_P2O3 = 3.75 →
  reaction_heat = - (20 * X - Y) →
  thermochemical_eqn = "P(s) + 3/4 O2(g) = 1/2 P2O3(s) ∆H = -(20X - Y) kJ/mol" :=
begin
  sorry
end

end reaction_products_and_thermochemistry_l753_753449


namespace base10_log_pascal_l753_753426

def g (n : ℕ) : ℝ :=
  Real.log10 (List.prod (List.map (λ k, (Nat.choose n k : ℝ)) (List.range (n+1))))

theorem base10_log_pascal (n : ℕ) : 
  g(n) / Real.log10 2 = Real.log2 (n + 1) + n * (n - 1) / 2 :=
by
  sorry

end base10_log_pascal_l753_753426


namespace ratio_EG_GF_l753_753040

/-- In triangle ABC, M is the midpoint of BC.
    Given that AB = 15, AC = 20, E is on segment AC, and F is on segment AB.
    G is the intersection of segments EF and AM, and AE = 3AF.
    Then, EG/GF = 7/3.
-/
theorem ratio_EG_GF (A B C M E F G : Type)
  [linear_ordered_ring A]
  (hM : midpoint M B C)
  (hAB : dist A B = 15)
  (hAC : dist A C = 20)
  (hE_on_AC : on_line_segment A C E)
  (hF_on_AB : on_line_segment A B F)
  (hG_intersection : is_intersection_point G E F A M)
  (hAE_eq_3AF : dist A E = 3 * dist A F)
  : dist E G / dist G F = (7 / 3) := sorry

end ratio_EG_GF_l753_753040


namespace cost_plane_l753_753444

def cost_boat : ℝ := 254.00
def savings_boat : ℝ := 346.00

theorem cost_plane : cost_boat + savings_boat = 600 := 
by 
  sorry

end cost_plane_l753_753444


namespace curve_tangent_line_at_point_l753_753110

-- Given conditions
def curve (x : ℝ) : ℝ := 3 * x - 2 * x^3
def point_x := -1
def tangent_line_at (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := 
  λ x, ((-6 * a^2 + 3) * (x - a) + f a)

-- Question to be proved
def equation_of_tangent_line_at (x : ℝ) (y : ℝ) : Prop :=
  3 * x + y + 4 = 0

-- The theorem statement/proof problem
theorem curve_tangent_line_at_point : 
  equation_of_tangent_line_at point_x (curve point_x) :=
sorry

end curve_tangent_line_at_point_l753_753110


namespace correct_options_l753_753547

noncomputable def correct_statements (A B C D : Prop) : Prop :=
  (A ∧ C) ∧ ¬B ∧ ¬D

-- Define the conditions and the corresponding interpretations
axiom linear_correlation (r : ℝ) :
  A ↔ (|r| = 1)

axiom expectation_linear_transformation (X : ℝ → ℝ) (a b : ℝ) :
  E(a * X + b) = a * E(X) + b

axiom variance_linear_transformation (X : ℝ → ℝ) (a b : ℝ) :
  D(a * X + b) = a^2 * D(X)

axiom sum_squared_residuals (model_fit: Model) :
  C ↔ model_fit.sum_squared_residuals < model_fit.threshold

axiom chi_square_test (X Y: Category) (k crit_value: ℝ) :
  D ↔ (k < crit_value)

-- Prove that statements A and C are correct
theorem correct_options (A_correct: r = 1) (D_incorrect: ¬ (k < crit_value)) :
  correct_statements A B C D :=
by
  sorry

end correct_options_l753_753547


namespace boxes_per_case_l753_753740

theorem boxes_per_case (total_boxes : ℕ) (cases : ℕ) (h1 : total_boxes = 24) (h2 : cases = 3) : (total_boxes / cases) = 8 :=
by
  rw [h1, h2]
  norm_num

end boxes_per_case_l753_753740


namespace concurrent_or_parallel_l753_753711

variables {A B C D E F A' P Q : Type*}

-- Hypotheses and conditions from the description:
axioms 
  (is_triangle : Prop)
  (D_on_BC : Prop)
  (circumcircle_ABD_intersects_AC_at_E : Prop)
  (circumcircle_ACD_intersects_AB_at_F : Prop)
  (A'_reflection_of_A_across_BC : Prop)
  (A'C_intersects_DE_at_P : Prop)
  (A'B_intersects_DF_at_Q : Prop)
  (AD BP CQ : Prop)

theorem concurrent_or_parallel
  (h1 : is_triangle)
  (h2 : D_on_BC)
  (h3 : circumcircle_ABD_intersects_AC_at_E)
  (h4 : circumcircle_ACD_intersects_AB_at_F)
  (h5 : A'_reflection_of_A_across_BC)
  (h6 : A'C_intersects_DE_at_P)
  (h7 : A'B_intersects_DF_at_Q) : 
  AD ∧ BP ∧ CQ := sorry

end concurrent_or_parallel_l753_753711


namespace question1_question2_l753_753437

-- Define the parabola and points
def parabola (x y : ℝ) := y^2 = 2 * x
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (-2, 0)

-- Define line conditions and prove results
theorem question1:
  (∃ y: ℝ, parabola 2 y ∧ (x - 2*y + 2 = 0 ∨ x + 2*y + 2 = 0)) :=
sorry

theorem question2:
  (∀ (l: ℝ → ℝ) (M N : ℝ × ℝ),
   (parabola M.1 M.2 ∧ parabola N.1 N.2 ∧ 
   l M.2 = M.1 ∧ l N.2 = N.1 → 
   (l = (λ y, (y - x).coefficient + const_term) → 
   angle A B M = angle A B N))) :=
sorry

end question1_question2_l753_753437


namespace cost_to_paint_cube_l753_753479

-- Define the conditions
def paintCostPerKg : ℝ := 40.00
def coveragePerKg : ℝ := 20.00
def sideLength : ℝ := 10.00

-- Theorem statement to prove the cost to paint the cube
theorem cost_to_paint_cube :
  let faceArea := sideLength * sideLength in
  let totalSurfaceArea := 6 * faceArea in
  let paintRequired := totalSurfaceArea / coveragePerKg in
  let totalCost := paintRequired * paintCostPerKg in
  totalCost = 1200.00 :=
by
  sorry

end cost_to_paint_cube_l753_753479


namespace pieces_fish_fillet_l753_753643

theorem pieces_fish_fillet (pieces_team1 pieces_team2 total_pieces : ℕ)
  (h_team1 : pieces_team1 = 189)
  (h_team2 : pieces_team2 = 131)
  (h_total : total_pieces = 500) :
  (total_pieces - (pieces_team1 + pieces_team2) = 180) :=
by
  rw [h_team1, h_team2, h_total]
  norm_num
  exact rfl

end pieces_fish_fillet_l753_753643


namespace interest_rate_first_part_eq_3_l753_753096

variable (T P1 P2 r2 I : ℝ)
variable (hT : T = 3400)
variable (hP1 : P1 = 1300)
variable (hP2 : P2 = 2100)
variable (hr2 : r2 = 5)
variable (hI : I = 144)

theorem interest_rate_first_part_eq_3 (r : ℝ) (h : (P1 * r) / 100 + (P2 * r2) / 100 = I) : r = 3 :=
by
  -- leaning in the proof
  sorry

end interest_rate_first_part_eq_3_l753_753096


namespace area_of_region_l753_753507

def abs (x : ℝ) : ℝ := if x ≥ 0 then x else -x

theorem area_of_region : 
  let region := {p : ℝ × ℝ | p.2 = abs (p.1 - 3) ∧ p.2 ≤ 5 - abs (p.1 + 2)} in 
  let verts := [(-2, 5), (3, 0)] in
  ∑ (i : ℕ) in finset.range 2, 
    (verts.nth i).get_or_else (0, 0)
    -- uses Shoelace formula to compute the area
    (⊕ i, (region.verts[i] * (region.verts[(i + 1) % 2])) - (region.verts[(i + 1) % 2] * region.verts[i])) / 2 = 12.5 :=
sorry

end area_of_region_l753_753507


namespace digit_at_1500_l753_753963

theorem digit_at_1500 {n : ℕ} (h : 0.\overline{318181} = 7 / 22) (h_period : (318181).length = 6) :
  (digit_of_rpt_expansion 1500 0.\overline{318181}) = 1 :=
sorry

end digit_at_1500_l753_753963


namespace imaginary_powers_sum_zero_l753_753104

noncomputable def sum_of_imaginary_powers : ℤ :=
  ∑ k in finset.range 2012, (complex.I ^ k)

theorem imaginary_powers_sum_zero : sum_of_imaginary_powers = 0 :=
  sorry

end imaginary_powers_sum_zero_l753_753104


namespace functions_increasing_on_negative_interval_l753_753546

def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂

theorem functions_increasing_on_negative_interval :
  is_increasing_on (λ x: ℝ, x) {x | x < 0} ∧
  is_increasing_on (λ x: ℝ, -x^2) {x | x < 0} ∧
  is_increasing_on (λ x: ℝ, -1/x) {x | x < 0} ∧
  ¬ is_increasing_on (λ x: ℝ, 1 - x) {x | x < 0} :=
by
  sorry

end functions_increasing_on_negative_interval_l753_753546


namespace squirrel_hazelnuts_l753_753141

noncomputable def Pizizubka_found (x : ℕ) : ℕ := x
noncomputable def Zrzečka_found (x : ℕ) : ℕ := 2 * x
noncomputable def Ouška_found (x : ℕ) : ℕ := 3 * x

noncomputable def Pizizubka_remaining (x : ℕ) : ℕ := x / 2
noncomputable def Zrzečka_remaining (x : ℕ) : ℕ := 2 * x * 2 / 3
noncomputable def Ouška_remaining (x : ℕ) : ℕ := 3 * x * 3 / 4

theorem squirrel_hazelnuts (x : ℕ) :
  Pizizubka_remaining x + Zrzečka_remaining x + Ouška_remaining x = 196 -> 
  x = 48 ∧ 2 * x = 96 ∧ 3 * x = 144 :=
by
  sorry

end squirrel_hazelnuts_l753_753141


namespace triangle_sides_inequality_l753_753021

theorem triangle_sides_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
    (a/(b + c - a) + b/(c + a - b) + c/(a + b - c)) ≥ ((b + c - a)/a + (c + a - b)/b + (a + b - c)/c) ∧
    ((b + c - a)/a + (c + a - b)/b + (a + b - c)/c) ≥ 3 :=
by
  sorry

end triangle_sides_inequality_l753_753021


namespace sum_of_powers_of_negative_one_l753_753271

theorem sum_of_powers_of_negative_one :
  ∑ k in (Finset.range 2006).filter (λ k, k % 2 = 1), (-1)^k = -1003 := 
by
  sorry

end sum_of_powers_of_negative_one_l753_753271


namespace smallest_boxes_l753_753499

theorem smallest_boxes (n : Nat) (h₁ : n % 5 = 0) (h₂ : n % 24 = 0) : n = 120 := 
  sorry

end smallest_boxes_l753_753499


namespace ratio_area_rhombus_to_square_l753_753027

namespace Geometry

open Real

/-- Given a rhombus with an angle of 30 degrees,
    an inscribed circle, and a square inscribed in the circle,
    the ratio of the area of the rhombus to the area of the square is 4. -/
theorem ratio_area_rhombus_to_square (a : ℝ) (h₀: 0 < a) :
  let S1 := (a^2 * (1/2))
  let r := (a / 4)
  let b := (r * sqrt 2)
  let S2 := b^2 
  S1 / S2 = 4 :=
by
  sorry

end Geometry

end ratio_area_rhombus_to_square_l753_753027


namespace find_weeks_period_l753_753182

def weekly_addition : ℕ := 3
def bikes_sold : ℕ := 18
def bikes_in_stock : ℕ := 45
def initial_stock : ℕ := 51

theorem find_weeks_period (x : ℕ) :
  initial_stock + weekly_addition * x - bikes_sold = bikes_in_stock ↔ x = 4 := 
by 
  sorry

end find_weeks_period_l753_753182


namespace cleaner_for_rabbit_stain_l753_753442

theorem cleaner_for_rabbit_stain :
  ∃ R : ℕ, 6 * 6 + 4 * 3 + R = 49 ∧ R = 1 :=
begin
  use 1,
  split,
  { norm_num },
  { refl }
end

end cleaner_for_rabbit_stain_l753_753442


namespace integral_f_K_l753_753304

noncomputable def f (x : ℝ) : ℝ := 1 / x

def f_K (K : ℝ) (x : ℝ) : ℝ :=
  if K = 1 then
    if x ≥ 1 then 1 else 1 / x
  else if f x ≤ K then K else f x

theorem integral_f_K :
  ∫ x in (1/4 : ℝ)..2, f_K 1 x = 1 + 2 * Real.log 2 := by
  sorry

end integral_f_K_l753_753304


namespace ordered_pair_correct_l753_753118

def find_ordered_pair (s m : ℚ) : Prop :=
  (∀ t : ℚ, (∃ x y : ℚ, x = -3 + t * m ∧ y = s + t * (-7) ∧ y = (3/4) * x + 5))
  ∧ s = 11/4 ∧ m = -28/3

theorem ordered_pair_correct :
  find_ordered_pair (11/4) (-28/3) :=
by
  sorry

end ordered_pair_correct_l753_753118


namespace problem_statement_l753_753543

noncomputable def A : ℝ := real.cbrt (16 * real.sqrt 2)
noncomputable def B : ℝ := real.sqrt (9 * real.cbrt 9)
noncomputable def C : ℝ := ((real.sqrt 2)^(2/5))^2

theorem problem_statement :
  A^2 + B^3 + C^5 = 105 :=
by
  unfold A B C
  sorry

end problem_statement_l753_753543


namespace smallest_MPR_but_not_MPRUUD_l753_753462

-- Define what it means for a number to be MPR (Math Prize Resolvable)
def isMPR (n : ℕ) : Prop :=
  ∃ M A T H P R I Z E : ℕ, 
    M ≠ 0 ∧ P ≠ 0 ∧ 
    M ≠ A ∧ M ≠ T ∧ M ≠ H ∧ M ≠ P ∧ M ≠ R ∧ M ≠ I ∧ M ≠ Z ∧ 
    A ≠ T ∧ A ≠ H ∧ A ≠ P ∧ A ≠ R ∧ A ≠ I ∧ A ≠ Z ∧ 
    T ≠ H ∧ T ≠ P ∧ T ≠ R ∧ T ≠ I ∧ T ≠ Z ∧ 
    H ≠ P ∧ H ≠ R ∧ H ≠ I ∧ H ≠ Z ∧ 
    P ≠ R ∧ P ≠ I ∧ P ≠ Z ∧ 
    R ≠ I ∧ R ≠ Z ∧ 
    I ≠ Z ∧
    1000 * M + 100 * A + 10 * T + H + 10000 * P + 1000 * R + 100 * I + 10 * Z + E = n

-- Define what it means for a number to be MPRUUD (Math Prize Resolvable with Unique Units Digits)
def isMPRUUD (n : ℕ) : Prop :=
  isMPR n ∧
  ∀ (M1 A1 T1 H1 P1 R1 I1 Z1 E1 M2 A2 T2 H2 P2 R2 I2 Z2 E2 : ℕ),
    1000 * M1 + 100 * A1 + 10 * T1 + H1 + 10000 * P1 + 1000 * R1 + 100 * I1 + 10 * Z1 + E1 = n →
    1000 * M2 + 100 * A2 + 10 * T2 + H2 + 10000 * P2 + 1000 * R2 + 100 * I2 + 10 * Z2 + E2 = n →
    H1 = H2 ∧ E1 = E2 → 
    M1 = M2 ∧ A1 = A2 ∧ T1 = T2 ∧ P1 = P2 ∧ R1 = R2 ∧ I1 = I2 ∧ Z1 = Z2 ∧ E1 = E2

-- The main theorem statement, proving that 12843 is the smallest number which is MPR but not MPRUUD
theorem smallest_MPR_but_not_MPRUUD : 
  ∃ n, isMPR n ∧ ¬isMPRUUD n ∧ (∀ m, isMPR m ∧ ¬isMPRUUD m → m ≥ n) := 
begin
  use 12843,
  sorry
end

end smallest_MPR_but_not_MPRUUD_l753_753462


namespace probability_of_selecting_A_car_l753_753570

noncomputable def probability_selecting_type_A (total_types : ℕ) (selected_types : ℕ) : ℚ := 
  (selected_types - 1 : ℚ) / finset.card (finset.powerset_len selected_types (finset.range total_types))

theorem probability_of_selecting_A_car :
  probability_selecting_type_A 5 2 = 2 / 5 :=
sorry

end probability_of_selecting_A_car_l753_753570


namespace gcf_60_90_150_l753_753536

theorem gcf_60_90_150 : Nat.gcf 60 90 150 = 30 := by
  have factorization_60 : 60 = 2^2 * 3 * 5 := rfl
  have factorization_90 : 90 = 2 * 3^2 * 5 := rfl
  have factorization_150 : 150 = 2 * 3 * 5^2 := rfl
  sorry

end gcf_60_90_150_l753_753536


namespace opposite_sides_range_l753_753314

theorem opposite_sides_range (a : ℝ) : (2 * 1 + 3 * a + 1) * (2 * a - 3 * 1 + 1) < 0 ↔ -1 < a ∧ a < 1 := sorry

end opposite_sides_range_l753_753314


namespace lambda_parallel_condition_l753_753346

theorem lambda_parallel_condition {λ : ℝ} (h_parallel : ∃ k: ℝ, (2, 5) = k • (λ, 4)) : λ = 8 / 5 :=
by
  sorry

end lambda_parallel_condition_l753_753346


namespace incorrect_expression_l753_753007

variable (x y : ℝ)

theorem incorrect_expression (h : x > y) (hnx : x < 0) (hny : y < 0) : x^2 - 3 ≤ y^2 - 3 := by
sorry

end incorrect_expression_l753_753007


namespace range_of_m_l753_753366

-- Define the condition of the inverse proportion function being in the second and fourth quadrants
def in_second_and_fourth_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x > 0 ∧ x > 0 → f x < 0

-- Define the inverse proportion function
def inv_proportion (m : ℝ) (x : ℝ) : ℝ := (m + 3) / x

-- Define the problem
theorem range_of_m (m : ℝ) :
  (in_second_and_fourth_quadrants (inv_proportion m)) → m < -3 :=
sorry

end range_of_m_l753_753366


namespace distance_between_house_and_school_l753_753895

noncomputable def travel_time_to_library(distance: ℝ) (speed: ℝ): ℝ :=
  (distance / 2) / speed

noncomputable def travel_time_to_school(distance: ℝ) (speed: ℝ): ℝ :=
  (distance / 2) / speed

noncomputable def travel_time_home(distance: ℝ) (speed: ℝ): ℝ :=
  distance / speed

def total_time (D: ℝ) : ℝ :=
  travel_time_to_library(D, 3) + 
  0.5 + 
  travel_time_to_school(D, 2.5) + 
  travel_time_home(D, 2)

theorem distance_between_house_and_school : 
  ∃ D : ℝ, total_time(D) = 5.5 ∧ D = (900 / 89) :=
by
  sorry

end distance_between_house_and_school_l753_753895


namespace smallest_period_range_f_l753_753318

-- Define the vectors OA and OB based on the given conditions
def OA (x : ℝ) : ℝ × ℝ := (2 * real.cos x, real.sqrt 3)
def OB (x : ℝ) : ℝ × ℝ := (real.sin x + real.sqrt 3 * real.cos x, -1)

-- Define the function f(x) as the dot product of OA and OB plus 2
def f (x : ℝ) : ℝ :=
  let dot_product := (OA x).1 * (OB x).1 + (OA x).2 * (OB x).2
  dot_product + 2

-- Statement of the problem (1): Proving the smallest positive period of f(x) is π
theorem smallest_period (x : ℝ) :
  (∀ x, f(x + π) = f(x)) ∧ ¬(∃ δ, 0 < δ ∧ δ < π ∧ ∀ x, f(x + δ) = f(x)) := sorry

-- Statement of the problem (2): Proving the range of f(x) when x ∈ (0, π/2)
theorem range_f (x : ℝ) (h : 0 < x ∧ x < real.pi / 2) :
  ∃ y, f(x) = y ∧ (-real.sqrt 3 + 2 < y ∧ y ≤ 4) := sorry

end smallest_period_range_f_l753_753318


namespace sum_squares_roots_eq_zero_l753_753241

open Polynomial

-- Given conditions
variables (s : Fin 12 → ℂ)

def roots_sum_zero (s : Fin 12 → ℂ) : Prop :=
  ∑ i, s i = 0

def product_sum_zero (s : Fin 12 → ℂ) : Prop :=
  ∑ i j in Finset.univ.filter (λ p : Fin 12 × Fin 12, p.fst ≠ p.snd), s i * s j = 0

def polynomial_roots_of_degree_12 (s : Fin 12 → ℂ) : Prop :=
  (∃ p : Polynomial ℂ, p.degree = 12 ∧ (p.coeff 9) = 10 ∧ (p.coeff 3) = 5 ∧ (p.coeff 0) = 50 ∧ (∀ x, is_root p x ↔ x ∈ (multiset.map (λ i, s i) (Finset.univ : Finset (Fin 12)).val)))

-- Proof problem
theorem sum_squares_roots_eq_zero (s : Fin 12 → ℂ) 
  (h1 : roots_sum_zero s) 
  (h2 : product_sum_zero s) 
  (h3 : polynomial_roots_of_degree_12 s) : 
  ∑ i, (s i)^2 = 0 :=
sorry

end sum_squares_roots_eq_zero_l753_753241


namespace factorial_bounds_l753_753099

theorem factorial_bounds (n : ℕ) (h : n ≥ 1) : 2^(n-1) ≤ n! ∧ n! ≤ n^n := by
  induction n with
  | zero => 
    have h_zero : ¬ (0 ≥ 1) := by simp
    contradiction
  | succ n ih => 
    cases n with
    | zero => 
      simp
      exact ⟨by norm_num, by norm_num⟩
    | succ n => 
      have h_induct : 2^n ≤ (n+1)! ∧ (n+1)! ≤ (n+1)^(n+1) := ih (nat.succ_le_succ n.zero_le)
      split
      case left =>
        calc 2^n 
          ≤ 2 * (2^(n-1)) : by norm_num
          ... ≤ 2 * (n!) : by exact mul_le_mul_of_nonneg_left h_induct.1 (nat.zero_le _)
          ... ≤ (n + 2) * n! : by exact mul_le_mul (by exact le_of_lt (by exact nat.one_lt_le_iff.2 (by exact zero_lt_succ n))) (by rfl) (nat.zero_le _) (nat.fact_pos _)
          ... = (n + 2)! : by rw nat.factorial_succ
      case right =>
        calc (n+2)!
          ≤ (n + 2) * (n+1)! : by exact nat.factorial_succ n
          ... ≤ (n + 2) * (n + 1) ^ (n + 1) : by exact mul_le_mul_left (nat.succ_pos _)
          ... = (n + 2) ^ (n + 2) : by exact _ sorry

end factorial_bounds_l753_753099


namespace dealer_cannot_prevent_l753_753913

theorem dealer_cannot_prevent (m n : ℕ) (h : m < 3 * n ∧ n < 3 * m) :
  ∃ (a b : ℕ), (a = 3 * b ∨ b = 3 * a) ∨ (a = 0 ∧ b = 0):=
sorry

end dealer_cannot_prevent_l753_753913


namespace range_of_a_plus_b_l753_753320

variable (a b : ℝ)
variable (pos_a : 0 < a)
variable (pos_b : 0 < b)
variable (h : a + b + 1/a + 1/b = 5)

theorem range_of_a_plus_b : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end range_of_a_plus_b_l753_753320


namespace closest_point_on_line_l753_753282

theorem closest_point_on_line 
  (t : ℚ)
  (x y z : ℚ)
  (x_eq : x = 3 + t)
  (y_eq : y = 2 - 3 * t)
  (z_eq : z = -1 + 2 * t)
  (x_ortho_eq : (1 + t) = 0)
  (y_ortho_eq : (3 - 3 * t) = 0)
  (z_ortho_eq : (-3 + 2 * t) = 0) :
  (45/14, 16/14, -1/7) = (x, y, z) := by
  sorry

end closest_point_on_line_l753_753282


namespace triangle_CF_length_l753_753143

theorem triangle_CF_length 
  {A B C D E F : Type}
  (AB BC CA : ℕ)
  (h_AB : AB = 11)
  (h_BC : BC = 24)
  (h_CA : CA = 20)
  (bisector_intersect_D : ∃ D : Type, ∀ A B C : Type, bisector (angle A B C) ∩ BC = D)
  (circumcircle_intersect_E : ∃ E : Type, ∀ A B C : Type, circumcircle (triangle A B C) ∩ (line A E) ≠ ∅)
  (circumcircle_TrBED_intersect_F : ∃ F : Type, ∀ D E B : Type, circumcircle (triangle B E D) ∩ (line B F) ≠ ∅) :
  CF = 30 :=
by
  sorry

end triangle_CF_length_l753_753143


namespace func_range_eq_l753_753513

open Set

theorem func_range_eq : 
  let f x := x^2 - 2 * x - 3 in 
  (range (λ x : ℝ, if x ∈ Icc (-1 : ℝ) 2 then f x else 0)) = Ico (-4 : ℝ) 0 :=
by sorry

end func_range_eq_l753_753513


namespace annual_increase_rate_l753_753272

theorem annual_increase_rate (r : ℝ) : 
  (6400 * (1 + r) * (1 + r) = 8100) → r = 0.125 :=
by sorry

end annual_increase_rate_l753_753272


namespace train_speed_71_94_km_hr_l753_753592

noncomputable def train_speed (train_length_km tunnel_length_km time_minutes : ℝ) : ℝ :=
  let total_distance := train_length_km + tunnel_length_km
  let time_hours := time_minutes / 60
  total_distance / time_hours

theorem train_speed_71_94_km_hr :
  train_speed 0.1 2.9 2.5 ≈ 71.94 :=
by
  sorry

end train_speed_71_94_km_hr_l753_753592


namespace solution_set_l753_753967

namespace Solution

def valid_inequality (x : ℝ) (y : ℝ) : Prop :=
  y - x < real.sqrt (4 * x^2)

theorem solution_set (x y : ℝ) :
  valid_inequality x y ↔ ((x ≥ 0 ∧ y < 3 * x) ∨ (x < 0 ∧ y < -x)) :=
by
  unfold valid_inequality
  sorry

end Solution

end solution_set_l753_753967


namespace max_income_zero_l753_753561

-- Define three piles of stones
def piles : Type := fin 3 → ℕ

-- Define the move operation
def move (p : piles) (i j : fin 3) : piles :=
λ k => if k = i then p k - 1 else if k = j then p k + 1 else p k

-- Income function for a move from pile i to pile j
def income (p : piles) (i j : fin 3) : ℤ :=
p j - (p i - 1)

-- Theorem stating the maximum total earnings of Sisyphus
theorem max_income_zero (initial_piles : piles) :
  ∃ (moves : (fin 3 × fin 3) → ℕ), 
    let final_piles := foldl (λ p (i_j : fin 3 × fin 3) => 
                             move p i_j.fst i_j.snd)
                             initial_piles
                             (map (λ x => (x.fst, x.snd)) (fin_enum (moves)))
  in final_piles = initial_piles ∧ 
     ∑ x in fin_enum (moves), (income (foldl (λ p (i_j : fin 3 × fin 3) => 
                                              move p i_j.fst i_j.snd)
                                              initial_piles
                                              (take x (fin_enum (moves))))
                                         x.fst x.snd) = 0 :=
by
  sorry

end max_income_zero_l753_753561


namespace QPA_OQB_ninety_degrees_l753_753750

open EuclideanGeometry

variables {A B C P Q O : Point}

-- Definitions for conditions
def angle_A_lt_angle_C (ABC : Triangle) : Prop := ABC.A < ABC.C
def circumcenter (ABC : Triangle) : Point := circumcenter ABC
def external_bisector_intersects_BC (ABC : Triangle) (A' Q' : Point) : Prop :=
  bisector (angle A B C) = bisector (angle A C B) ∧ Q' ∈ (line B C)

def similar_triangles (P' A' : Point) : Prop :=
  similar (triangle B P' A') (triangle P' A' C)

def acute_triangle (ABC : Triangle) : Prop := is_acute ABC

theorem QPA_OQB_ninety_degrees
  (ABC : Triangle) 
  (A' Q' O' P' : Point)
  (acute : acute_triangle ABC)
  (circum : circumcenter ABC = O')
  (external_bisector : external_bisector_intersects_BC ABC A' Q')
  (ineq : angle_A_lt_angle_C ABC)
  (similar : similar_triangles P' A') :
  angle Q' P' A' + angle O' Q' B = 90 := 
sorry

end QPA_OQB_ninety_degrees_l753_753750


namespace no_solution_for_inequalities_l753_753626

theorem no_solution_for_inequalities (x : ℝ) : ¬ ((6 * x - 2 < (x + 2) ^ 2) ∧ ((x + 2) ^ 2 < 9 * x - 5)) :=
by sorry

end no_solution_for_inequalities_l753_753626


namespace h_2023_eq_4052_l753_753907

theorem h_2023_eq_4052 (h : ℕ → ℕ) (h1 : h 1 = 2) (h2 : h 2 = 2) 
    (h3 : ∀ n ≥ 3, h n = h (n-1) - h (n-2) + 2 * n) : h 2023 = 4052 := 
by
  -- Use conditions as given
  sorry

end h_2023_eq_4052_l753_753907


namespace car_cost_l753_753138

-- Define the weekly allowance in the first year
def first_year_allowance_weekly : ℕ := 50

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Calculate the total first year savings
def first_year_savings : ℕ := first_year_allowance_weekly * weeks_in_year

-- Define the hourly wage and weekly hours worked in the second year
def hourly_wage : ℕ := 9
def weekly_hours_worked : ℕ := 30

-- Calculate the total second year earnings
def second_year_earnings : ℕ := hourly_wage * weekly_hours_worked * weeks_in_year

-- Define the weekly spending in the second year
def weekly_spending : ℕ := 35

-- Calculate the total second year spending
def second_year_spending : ℕ := weekly_spending * weeks_in_year

-- Calculate the total second year savings
def second_year_savings : ℕ := second_year_earnings - second_year_spending

-- Calculate the total savings after two years
def total_savings : ℕ := first_year_savings + second_year_savings

-- Define the additional amount needed
def additional_amount_needed : ℕ := 2000

-- Calculate the total cost of the car
def total_cost_of_car : ℕ := total_savings + additional_amount_needed

-- Theorem statement
theorem car_cost : total_cost_of_car = 16820 := by
  -- The proof is omitted; it is enough to state the theorem
  sorry

end car_cost_l753_753138


namespace rectangle_problem_perpendicular_EO_ZD_l753_753307

noncomputable theory

variables {a b : ℝ} 
variables (A B C D E Z O : Point)  -- Assumed Point type exists

-- Definitions based on the conditions in (a)
def Rectangle : Prop :=
  AB = a ∧ BC = b ∧ intersection_diagonals(ABCD) = O

def Extensions : Prop :=
  (AE = AO) ∧ (BZ = BO)

def Equilateral_triangle : Prop := 
  equilateral (triangle( E Z C ))

-- Defining the mathematical problem
theorem rectangle_problem (h1 : Rectangle) 
                         (h2 : Extensions) 
                         (h3 : Equilateral_triangle) : 
                         b = a * (⟦sqrt 3⟧ : ℝ) := 
begin
  sorry  -- Proof of b = a * sqrt(3)
end

theorem perpendicular_EO_ZD (h1 : Rectangle) 
                            (h2 : Extensions) 
                            (h3 : Equilateral_triangle) : 
                            is_perpendicular (line(E O)) (line(Z D)) := 
begin
  sorry  -- Proof that EO is perpendicular to ZD
end

end rectangle_problem_perpendicular_EO_ZD_l753_753307


namespace triangle_count_in_circle_intersections_l753_753474

theorem triangle_count_in_circle_intersections
    (P : Finset.Points Circle) (hP : P.card = 10) (hInter : ∀ (C1 C2 C3 : Chord P), 
    ¬ Collinear (Intersection C1 C2) (Intersection C2 C3) (Intersection C1 C3)) : 
    (number_of_triangles P = 120) :=
sorry

end triangle_count_in_circle_intersections_l753_753474


namespace even_function_passing_through_points_l753_753217

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def passes_through_points (f : ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
f p1.1 = p1.2 ∧ f p2.1 = p2.2

theorem even_function_passing_through_points :
  ∃ f : ℝ → ℝ, is_even_function f ∧ 
                passes_through_points f (0, 0) (1, 1) ∧ 
                (f = λ x, x^4) :=
begin
  use (λ x, x^4),
  split,
  { intros x,
    simp, },
  split,
  { split,
    { simp, },
    { norm_num, }, },
  refl,
end

end even_function_passing_through_points_l753_753217


namespace intersection_area_pyramid_plane_l753_753207

noncomputable def point := (ℝ × ℝ × ℝ)

structure Pyramid :=
(base_coords : point)
(base_edges_equal : ∀ (A B : point), dist A B = 5)
(top_vertex : point)
(edge_length : ∀ (A : point), dist A top_vertex = 5)

def midpoint (p1 p2 : point) : point :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)

def plane_through (p1 p2 p3 : point) : (ℝ × ℝ × ℝ) → ℝ × ℝ × ℝ × ℝ :=
  sorry -- function to compute plane equation

theorem intersection_area_pyramid_plane 
  (A B C D E : point)
  (p : Pyramid)
  (h_AB : dist A B = 5)
  (h_AD : dist A D = 5)
  (h_AE : dist A E = 5)
  (h_BE : dist B E = 5)
  (h_DE : dist D E = 5)
  (h_CD : dist C D = 5)
  (mid_AE : point := midpoint A E)
  (mid_BD : point := midpoint B D)
  (mid_CD : point := midpoint C D)
  (π : (ℝ × ℝ × ℝ × ℝ) := plane_through mid_AE mid_BD mid_CD) :
  ∃ area : ℝ, area = sorry :=
sorry

end intersection_area_pyramid_plane_l753_753207


namespace find_price_of_first_brand_jeans_l753_753737

def regular_price_of_first_brand_jeans (d_1 d_2 P_1 : ℝ) : Prop :=
  d_1 + d_2 = 0.22 ∧
  d_2 = 0.15 ∧
  3 * P_1 * d_1 + 2 * 18 * d_2 = 8.55

theorem find_price_of_first_brand_jeans :
  ∃ P_1, regular_price_of_first_brand_jeans 0.07 0.15 P_1 ∧ P_1 = 15 :=
begin
  use 15,
  split,
  { unfold regular_price_of_first_brand_jeans,
    split,
    { exact rfl },
    split,
    { exact rfl },
    { norm_num }},
  { refl }
end

end find_price_of_first_brand_jeans_l753_753737


namespace find_f10_l753_753114

noncomputable def f : ℝ → ℝ := sorry -- Function will be defined by properties later.

-- Conditions
def y_odd (x : ℝ) : Prop := f(-(x-1)) = -f(x-1)
def y_even (x : ℝ) : Prop := f(-(x+1)) = f(x+1)
def f_condition (x : ℝ) : Prop := 0 ≤ x ∧ x < 1 → f(x) = 2^x

-- Theorem to prove
theorem find_f10 (h1 : ∀ x, y_odd x) (h2 : ∀ x, y_even x) (h3 : ∀ x, f_condition x) : f 10 = 0 :=
sorry

end find_f10_l753_753114


namespace relationship_of_abc_l753_753298

noncomputable theory

def a : ℝ := 2^(1/4)
def b : ℝ := (1/5)^(0.2)
def c : ℝ := Real.log 6 / Real.log (1/3) -- log base change rule

theorem relationship_of_abc : c < b ∧ b < a := 
by
  sorry

end relationship_of_abc_l753_753298


namespace f1_odd_f2_neither_f3_even_l753_753261

-- Definition of odd and even functions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Problem 1: Prove f1 is odd
def f1 (x : ℝ) := x^3 - 1/x
theorem f1_odd : is_odd f1 :=
  sorry

-- Problem 2: Prove f2 is neither odd nor even
def f2 (x : ℝ) := |x|
theorem f2_neither (x : ℝ) (hx : -4 ≤ x ∧ x ≤ 5) : ¬ is_odd f2 ∧ ¬ is_even f2 :=
  sorry

-- Problem 3: Prove f3 is even
def f3 (x : ℝ) : ℝ := if x > 1 then x^2 - 3*x else if x < -1 then x^2 + 3*x else 0
theorem f3_even : is_even f3 :=
  sorry

end f1_odd_f2_neither_f3_even_l753_753261


namespace garden_solution_l753_753189

def garden_problem (P C Pe : ℕ) : Prop := 
  (C = P - 60) ∧ 
  (Pe = 2 * C) ∧ 
  (P + C + Pe = 768)

theorem garden_solution : ∃ P, garden_problem P (P - 60) (2 * (P - 60)) ∧ P = 237 :=
by
  let P := 237
  exists P
  refine ⟨_, rfl⟩
  sorry

end garden_solution_l753_753189


namespace find_complex_number_z_l753_753615

theorem find_complex_number_z (z : ℂ) : 
  (z * (1 + complex.I) = 4 + 2 * complex.I) → 
  z = 3 - complex.I :=
by
  sorry

end find_complex_number_z_l753_753615


namespace loss_record_l753_753707

-- Conditions: a profit of 25 yuan is recorded as +25 yuan.
def profit_record (profit : Int) : Int :=
  profit

-- Statement we need to prove: A loss of 30 yuan is recorded as -30 yuan.
theorem loss_record : profit_record (-30) = -30 :=
by
  sorry

end loss_record_l753_753707


namespace coin_probability_l753_753580

theorem coin_probability :
  ∃ p : ℝ, p < 1/2 ∧ (binomial 6 3) * p^3 * (1 - p)^3 = 1/10 ∧
  p = (1 - Real.sqrt (1 - 4 * (1/200)^(1/3))) / 2 :=
by
  sorry

end coin_probability_l753_753580


namespace smallest_int_digits_product_l753_753632

/--
Consider a natural number n such that:
1. The digits of n, in increasing order, are d1, d2, ..., dk.
2. The sum of the squares of the digits is 85.
3. Each digit is larger than the one on its left.

Prove that the product of the digits of n is 18.
-/
theorem smallest_int_digits_product :
  ∃ n : ℕ,
  (∃ d : ℕ → ℕ,
    (∃ k : ℕ,
      (∀ i < k - 1, d i < d (i + 1)) ∧
      (finset.sum (finset.range k) (λ i, d i ^ 2) = 85)) ∧
    (n = (list.mk (finset.range k) (λ i, d i)).foldl (λ acc x, 10 * acc + x) 0)) ∧
  (∃ p : ℕ,
    p = finset.prod (finset.range k) (λ i, (d i)) ∧
    p = 18) := by
  sorry

end smallest_int_digits_product_l753_753632


namespace gcd_sum_of_cubes_l753_753289

-- Define the problem conditions
variables (n : ℕ) (h_pos : n > 27)

-- Define the goal to prove
theorem gcd_sum_of_cubes (h : n > 27) : 
  gcd (n^3 + 27) (n + 3) = n + 3 :=
by sorry

end gcd_sum_of_cubes_l753_753289


namespace total_pigs_correct_l753_753134

def initial_pigs : Float := 64.0
def incoming_pigs : Float := 86.0
def total_pigs : Float := 150.0

theorem total_pigs_correct : initial_pigs + incoming_pigs = total_pigs := by 
  sorry

end total_pigs_correct_l753_753134


namespace Menelaus_theorem_l753_753422

variables (A B C A1 B1 C1 : Type) [linear_ordered_field R]
variables (BC CA AB : R) (AC1 C1B BA1 A1C CB1 B1A : R)

def points_on_sides (A B C A1 B1 C1 : Type) : Prop :=
A1 ∈ line_segment BC ∧ B1 ∈ line_segment CA ∧ C1 ∈ line_segment AB

def collinear (A1 B1 C1 : Type) : Prop := sorry -- Definition or theorem that points A1, B1, C1 are collinear

theorem Menelaus_theorem (h : points_on_sides A B C A1 B1 C1) :
  collinear A1 B1 C1 ↔ 
  (AC1 / C1B) * (BA1 / A1C) * (CB1 / B1A) = -1 := sorry

end Menelaus_theorem_l753_753422


namespace coin_toss_probability_weather_forecast_accuracy_l753_753176

theorem coin_toss_probability :
  (1/2) ^ 5 = 1 / 32 :=
by sorry

theorem weather_forecast_accuracy (p_a p_b : ℝ) (h_a : p_a = 0.8) (h_b : p_b = 0.7) :
  p_a * p_b = 0.56 :=
by {
  rw [h_a, h_b],
  norm_num,
  sorry
}

end coin_toss_probability_weather_forecast_accuracy_l753_753176


namespace Ginger_sold_10_lilacs_l753_753297

variable (R L G : ℕ)

def condition1 := R = 3 * L
def condition2 := G = L / 2
def condition3 := L + R + G = 45

theorem Ginger_sold_10_lilacs
    (h1 : condition1 R L)
    (h2 : condition2 G L)
    (h3 : condition3 L R G) :
  L = 10 := 
  sorry

end Ginger_sold_10_lilacs_l753_753297


namespace two_hundred_fifteenth_digit_l753_753530

theorem two_hundred_fifteenth_digit (n : ℕ) (h : 215 = 215) :
  let decimal_sequence := "045945"
  let repeating_block_length := 6
  let position := 215 % repeating_block_length
  let digit := decimal_sequence.get (position - 1)
  digit = '9' :=
by {
  -- Assuming the repeating block is accurate and pre-computed
  have h_div : 17 / 370 = 0.045945945945..., sorry, -- Would require actual computation/proof
  have h_repeating : "045945".length = 6, sorry, -- Established length of repeating block
  have h_pos : position = 5 := by norm_num,
  have h_digit : digit = '9' := by norm_num,
  exact h_digit,
}

end two_hundred_fifteenth_digit_l753_753530


namespace range_of_k_l753_753709

theorem range_of_k
    (P : ℝ × ℝ)
    (k : ℝ)
    (on_line_P : P.1 * k + P.2 + 4 = 0)
    (on_circle_Q : (∃ Q : ℝ × ℝ, (Q.1 ^ 2 + Q.2 ^ 2 - 2 * Q.2 = 0) ∧ (dist P Q = 2)))
    : k ∈ (set.Iic (-2) ∪ set.Ici 2) := 
sorry

end range_of_k_l753_753709


namespace smallest_n_l753_753612

theorem smallest_n (n : ℕ) : 
  (n > 0 ∧ ((n^2 + n + 1)^2 > 1999) ∧ ∀ m : ℕ, (m > 0 ∧ (m^2 + m + 1)^2 > 1999) → m ≥ n) → n = 7 :=
sorry

end smallest_n_l753_753612


namespace mike_total_games_l753_753445

theorem mike_total_games
  (non_working : ℕ)
  (price_per_game : ℕ)
  (total_earnings : ℕ)
  (h1 : non_working = 9)
  (h2 : price_per_game = 5)
  (h3 : total_earnings = 30) :
  non_working + (total_earnings / price_per_game) = 15 := 
by
  sorry

end mike_total_games_l753_753445


namespace brick_wall_decrease_l753_753025

/-- In a certain brick wall, there are 5 rows in total and a total of 100 bricks.
The bottom row contains 18 bricks. Prove that the number of bricks decreases
by 1 as we go up each row. -/
theorem brick_wall_decrease :
  ∃ d : ℤ, ∃ (B2 B3 B4 B5 : ℤ),
  let B1 := 18 in
  let B := 100 in
  B1 + B2 + B3 + B4 + B5 = B ∧
  B2 = B1 - d ∧
  B3 = B2 - d ∧
  B4 = B3 - d ∧
  B5 = B4 - d ∧
  d = 1 :=
by
  sorry

end brick_wall_decrease_l753_753025


namespace range_of_a_l753_753175

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0) ↔ a ≤ -2 ∨ a = 1 :=
by sorry

end range_of_a_l753_753175


namespace distinct_trees_with_7_vertices_l753_753348

theorem distinct_trees_with_7_vertices : 
  count_distinct_trees 7 = 11 :=
sorry

end distinct_trees_with_7_vertices_l753_753348


namespace f_periodic_3_evaluate_f_l753_753762

def f (x : ℝ) : ℝ :=
if -2 ≤ x ∧ x ≤ 0 then 4 * x^2 - 2
else if 0 < x ∧ x < 1 then x
else 0  -- default case, values outside the interval [-2, 1) should be handled with periodicity

theorem f_periodic_3 (x : ℝ) : f (x + 3) = f x :=
sorry

theorem evaluate_f : f (5 / 2) = -1 :=
by
  have periodic := f_periodic_3
  have fx_neg_half : f (-1 / 2) = 4 * ((-1 / 2) ^ 2) - 2 := rfl
  rw [periodic, fx_neg_half]
  norm_num  -- simplifies 4 * (1/4) - 2

end f_periodic_3_evaluate_f_l753_753762


namespace triangles_with_two_white_vertices_l753_753653

theorem triangles_with_two_white_vertices (p f z : ℕ) 
    (h1 : p * f + p * z + f * z = 213)
    (h2 : (p * (p - 1) / 2) + (f * (f - 1) / 2) + (z * (z - 1) / 2) = 112)
    (h3 : p * f * z = 540)
    (h4 : (p * (p - 1) / 2) * (f + z) = 612) :
    (f * (f - 1) / 2) * (p + z) = 210 ∨ (f * (f - 1) / 2) * (p + z) = 924 := 
  sorry

end triangles_with_two_white_vertices_l753_753653


namespace solve_blind_boxes_problem_l753_753898
open nat probability

noncomputable def blind_boxes_same_model_prob : ℝ := 2 / 9
noncomputable def second_draw_library_prob : ℝ := 4 / 9
noncomputable def science_museum_model_dist (X : ℕ) : ℝ :=
  if X = 1 then 1 / 10
  else if X <= 10 then (9 / 10)^(X - 1) * (1 / 10)
  else 0

def blind_boxes_problem_statement : Prop :=
  blind_boxes_same_model_prob = 2 / 9 ∧
  second_draw_library_prob = 4 / 9 ∧
  ∀ (X : ℕ), (X = 1 ∨ (2 ≤ X ∧ X ≤ 10)) → science_museum_model_dist X =
    if X = 1 then 1 / 10
    else (9 / 10)^(X - 1) * (1 / 10)

theorem solve_blind_boxes_problem : blind_boxes_problem_statement :=
by sorry

end solve_blind_boxes_problem_l753_753898


namespace perpendicular_line_through_P_l753_753484

open Real

-- Define the point (1, 0)
def P : ℝ × ℝ := (1, 0)

-- Define the initial line x - 2y - 2 = 0
def initial_line (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the desired line 2x + y - 2 = 0
def desired_line (x y : ℝ) : Prop := 2 * x + y = 2

-- State that the desired line passes through the point (1, 0) and is perpendicular to the initial line
theorem perpendicular_line_through_P :
  (∃ m b, b ∈ Set.univ ∧ (∀ x y, desired_line x y → y = m * x + b)) ∧ ∀ x y, 
  initial_line x y → x ≠ 0 → desired_line y (-x / 2) :=
sorry

end perpendicular_line_through_P_l753_753484


namespace finite_moves_no_coins_beyond_l753_753473

/-- Statement of Problem Part 1: 
Given an initial configuration of coins on an infinite single-row table,
prove that after a finite number of allowed movements, no more moves can be performed. 
-/
theorem finite_moves (coins : ℕ → ℕ) :
  ∃ m, ∀ k ≥ m, ¬ (can_perform_operation k coins) :=
sorry

/-- Statement of Problem Part 2: 
If initially, there is exactly one coin in each room from 1 to n, 
prove that no coins can be placed in room n + 2 or any room to the right. 
-/
theorem no_coins_beyond (n : ℕ) (coins : ℕ → ℕ) 
  (h_init : ∀ k, 1 ≤ k ∧ k ≤ n → coins k = 1 ∧ (k > n ∨ coins k = 0)) :
  ∀ k ≥ n + 2, coins k = 0 :=
sorry

end finite_moves_no_coins_beyond_l753_753473


namespace distinct_lengths_from_E_to_DF_l753_753797

noncomputable def distinct_integer_lengths (DE EF: ℕ) : ℕ :=
if h : DE = 15 ∧ EF = 36 then 24 else 0

theorem distinct_lengths_from_E_to_DF :
  distinct_integer_lengths 15 36 = 24 :=
by {
  sorry
}

end distinct_lengths_from_E_to_DF_l753_753797


namespace max_sqrt_sum_l753_753664

theorem max_sqrt_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 3) :
  sqrt (a + 1) + sqrt (b + 2) ≤ 2 * sqrt 3 := by
  sorry

end max_sqrt_sum_l753_753664


namespace sum_sqrt_inequality_l753_753343

open Real

theorem sum_sqrt_inequality (n : ℕ) (a b : Fin n → ℝ)
  (h_pos_a : ∀ k, 0 < a k) (h_pos_b : ∀ k, 0 < b k) :
  (∑ k, sqrt ((a k) ^ 2 + (b k) ^ 2))
  ≥ sqrt ((∑ k, a k) ^ 2 + (∑ k, b k) ^ 2) :=
sorry

end sum_sqrt_inequality_l753_753343


namespace trajectory_of_T_l753_753317

-- Define coordinates for points A, T, and M
variables {x x0 y y0 : ℝ}
def A (x0: ℝ) (y0: ℝ) := (x0, y0)
def T (x: ℝ) (y: ℝ) := (x, y)
def M : ℝ × ℝ := (-2, 0)

-- Conditions
def curve (x : ℝ) (y : ℝ) := 4 * x^2 - y + 1 = 0
def vector_condition (x x0 y y0 : ℝ) := (x - x0, y - y0) = 2 * (-2 - x, -y)

theorem trajectory_of_T (x y x0 y0 : ℝ) (hA : curve x0 y0) (hV : vector_condition x x0 y y0) :
  4 * (3 * x + 4)^2 - 3 * y + 1 = 0 :=
by
  sorry

end trajectory_of_T_l753_753317


namespace find_percentage_l753_753358

theorem find_percentage (x p : ℝ) (h₀ : x = 780) (h₁ : 0.25 * x = (p / 100) * 1500 - 30) : p = 15 :=
by
  sorry

end find_percentage_l753_753358


namespace simplify_expression_l753_753466

theorem simplify_expression : 20 * (9 / 14) * (1 / 18) = 5 / 7 :=
by sorry

end simplify_expression_l753_753466


namespace even_number_in_center_is_8_l753_753924

/-- Define the 3x3 grid as a list of lists of integers -/
def grid : List (List ℤ) := 
  [[1, 2, 4],
   [3, 8, 5],
   [6, 7, 9]]

/-- Define the condition that consecutive numbers share an edge -/
def consecutive_numbers_share_edge (grid : List (List ℤ)) : Prop :=
  -- Implementation of the consecutive condition, omitted for clarity
  sorry

/-- Define the condition that the corners of the grid add up to 20 -/
def corners_add_up_to_20 (grid : List (List ℤ)) : Prop :=
  grid.head.head + grid.head.getLast 0 + grid.getLast 0.head + grid.getLast 0.getLast 0 = 20

/-- Main theorem stating the even number in the center of the grid is 8 -/
theorem even_number_in_center_is_8 
  (h1 : consecutive_numbers_share_edge grid)
  (h2 : corners_add_up_to_20 grid) : 
  grid.get! 1 |>.get! 1 = 8 :=
sorry

end even_number_in_center_is_8_l753_753924


namespace math_problem_l753_753438

-- Define the sequence {a_n} satisfying the given condition.
def a (n : ℕ) : ℝ := 
  if n = 1 then 2 else 2 / (2 * n - 1)

-- Define the sum S_n of the first n terms of the sequence {a_n / (2n+1)}
def S (n : ℕ) : ℝ :=
  ∑ k in finset.range(n + 1), a k / (2 * k + 1)

theorem math_problem (n : ℕ) : 
  ∑ k in finset.range(n + 1), (2 * k - 1) * a k = 2 * n ∧ 
  a 1 = 2 ∧ 
  (∀ m, a m = 2 / (2 * m - 1)) ∧ 
  (∀ m, S m = m * a (m + 1)) := 
by 
  sorry

end math_problem_l753_753438


namespace tangent_angle_existence_l753_753485

noncomputable def angle_tangent_line (n : ℤ) : ℝ :=
  (3 * Real.pi) / 4 + n * Real.pi ± Real.arcsin (Real.sqrt 2 / 3)

theorem tangent_angle_existence (n : ℤ) :
  let k := Real.tan (angle_tangent_line n)
  let M : ℝ × ℝ := (7, 1)
  let O : ℝ × ℝ := (4, 4)
  ∃ a : ℝ,
  a = angle_tangent_line n ∧
  (∃ t1 t2 : ℝ,
    t1 > 1 ∧ t2 > 2 ∧
    ((O.1 - M.1)^2 + (O.2 - M.2)^2 = 18) ∧
    (Real.sin (a - ((3 * Real.pi) / 4)) = Real.sqrt 2 / 3)) :=
begin
  sorry
end

end tangent_angle_existence_l753_753485


namespace sum_binom_eq_l753_753459

theorem sum_binom_eq (n k : ℕ) (h : k ≤ n) :
  ∑ i in Finset.range (n - k + 1), (Nat.choose n i) * (Nat.choose n (i + k)) = Nat.choose (2 * n) (n + k) :=
sorry

end sum_binom_eq_l753_753459


namespace total_length_bound_l753_753208

-- Definition of the problem conditions
variable (a_1 : ℝ) (k : ℝ) (n : ℕ) (strip_width : ℝ := 1)

-- Assume k is in the range (0, 1)
variable (k_pos : 0 < k)
variable (k_lt_1 : k < 1)

-- Definition of the n-th length
def panel_length (a_1 : ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  a_1 * k ^ n

-- Summing lengths of the first n panels
def total_length (a_1 : ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  a_1 * (1 - k ^ n) / (1 - k)

-- The Lean theorem stating the conclusion
theorem total_length_bound (n : ℕ) (h_gapless : ∀ i, (0 < i → i < n → k ^ (i-1) ≠ 0)) :
  ∃ N, ∀ n, total_length a_1 k n ≤ N :=
by 
  use a_1 / (1 - k)
  intros n
  have h_nonneg : ∀ n, 0 ≤ k^n := sorry
  calc
    total_length a_1 k n
        = a_1 * (1 - k ^ n) / (1 - k) : by sorry
    ... ≤ a_1 * 1 / (1 - k)           : by sorry
    ... = a_1 / (1 - k)               : by sorry

end total_length_bound_l753_753208


namespace line_perpendicular_to_plane_l753_753322

-- Given definitions
variables {m n : Line} {α β : Plane}

-- Conditions
axiom parallel_lines (m n : Line) : Prop
axiom perpendicular_line_plane (n : Line) (β : Plane) : Prop
axiom parallel_implies_perpendicular {m n : Line} {β : Plane} : (m ∥ n) → (n ⊥ β) → (m ⊥ β)

-- Statement to be proved
theorem line_perpendicular_to_plane :
  (m ∥ n) → (n ⊥ β) → (m ⊥ β) :=
by
  exact parallel_implies_perpendicular

end line_perpendicular_to_plane_l753_753322


namespace num_solutions_l753_753779

def f (x : ℝ) : ℝ :=
if x ≤ 0 then
  -x + 2
else
  3*x - 6

theorem num_solutions (s : Finset ℝ) (h₀ : ∀ x, f (f x) = 3 ↔ x ∈ s) : s.card = 3 := by
sorry

end num_solutions_l753_753779


namespace calc_value_l753_753288

theorem calc_value (n : ℕ) (h : 1 ≤ n) : 
  (5^(n+1) + 6^(n+2))^2 - (5^(n+1) - 6^(n+2))^2 = 144 * 30^(n+1) := 
sorry

end calc_value_l753_753288


namespace max_right_angles_in_triangular_prism_l753_753030

theorem max_right_angles_in_triangular_prism 
  (n_triangles : ℕ) 
  (n_rectangles : ℕ) 
  (max_right_angles_triangle : ℕ) 
  (max_right_angles_rectangle : ℕ)
  (h1 : n_triangles = 2)
  (h2 : n_rectangles = 3)
  (h3 : max_right_angles_triangle = 1)
  (h4 : max_right_angles_rectangle = 4) : 
  (n_triangles * max_right_angles_triangle + n_rectangles * max_right_angles_rectangle = 14) :=
by
  sorry

end max_right_angles_in_triangular_prism_l753_753030


namespace subset_0_in_X_l753_753017

-- Define the set X
def X : Set ℤ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Define the theorem to prove
theorem subset_0_in_X : {0} ⊆ X :=
by
  sorry

end subset_0_in_X_l753_753017


namespace solve_for_x_l753_753774

def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 2*x

theorem solve_for_x (x : ℝ) : f(f(x)) = f(x) ↔ (x = 0 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) := 
by
  sorry

end solve_for_x_l753_753774


namespace probability_palindrome_l753_753395

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

constant m : ℕ -- the number of valid starting digits

def count_palindromes : ℕ :=
  m * 10^3

def total_7_digit_numbers : ℕ :=
  m * 10^6

theorem probability_palindrome :
  (count_palindromes.to_Rat / total_7_digit_numbers) = 0.001 :=
by
  sorry

end probability_palindrome_l753_753395


namespace single_elimination_games_l753_753962

theorem single_elimination_games (n : ℕ) (h : n = 23) : 
  ∃ games : ℕ, games = n - 1 :=
by
  use 22
  sorry

end single_elimination_games_l753_753962


namespace selling_price_l753_753163

theorem selling_price 
  (cost_price : ℝ) 
  (profit_percentage : ℝ) 
  (h_cost : cost_price = 192) 
  (h_profit : profit_percentage = 0.25) : 
  ∃ selling_price : ℝ, selling_price = cost_price * (1 + profit_percentage) := 
by {
  sorry
}

end selling_price_l753_753163


namespace compute_area_l753_753778

open Real EuclideanSpace

noncomputable def area_of_triangle (a b : ℝ × ℝ × ℝ) : ℝ :=
  0.5 * norm (cross_product a b)

theorem compute_area : 
  let a := (3, -1, 2) in
  let b := (4, 2, -3) in 
  area_of_triangle a b = (sqrt 390) / 2 := sorry

end compute_area_l753_753778


namespace perimeter_triangle_CPQ_l753_753900

-- Define the inscribed circle within a right angle with radius R
variables {R : ℝ}

-- Define points A and B where the circle touches the sides of the right angle
variables {A B : ℝ}

-- Assume points P and Q are where the tangent from a point on the smaller arc AB intersects the sides CA and CB respectively
variables {P Q : ℝ}

-- Given the property of tangent segments from an external point to a circle
axiom tangent_segments : ∀ {x y : ℝ}, tend_from x = tend_from y

-- Prove the perimeter of the triangle CPQ is 4R
theorem perimeter_triangle_CPQ : 
  CP + PQ + QC = 4 * R :=
sorry

end perimeter_triangle_CPQ_l753_753900


namespace sequence_count_l753_753655

noncomputable def number_of_sequences (k : ℕ) : ℕ :=
  if k = 0 then 0 else 2^(k-1)

theorem sequence_count {k : ℕ} (hk : k > 0) 
    (a : ℕ → ℤ) 
    (ha1 : a 1 = 2)
    (hcond : ∀ n, 1 ≤ n ∧ n < k → (a n)^2 + (a (n+1))^2 = n^2 + (n+1)^2) 
    : (∃ (s : Finset (ℕ → ℤ)), s.card = number_of_sequences k) :=
begin
  sorry
end

end sequence_count_l753_753655


namespace compare_three_numbers_l753_753833

theorem compare_three_numbers :
  log 6 / log 0.7 < 0.7^6 ∧ 0.7^6 < 6^0.7 := sorry

end compare_three_numbers_l753_753833


namespace tangent_line_at_e_l753_753821

noncomputable def f (x : ℝ) := x * real.log x

noncomputable def f' (x : ℝ) := real.log x + 1

theorem tangent_line_at_e : ∀ (x y : ℝ), (x = e) → (f x = y) → (f' e = 2) → (y - e = 2 * (x - e)) :=
by
  intros x y h1 h2 h3
  sorry

end tangent_line_at_e_l753_753821


namespace point_outside_circle_l753_753015

theorem point_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, (a * x + b * y = 1) ∧ (x^2 + y^2 = 1)) :
  a^2 + b^2 > 1 := by
  sorry

end point_outside_circle_l753_753015


namespace brothers_percentage_fewer_trees_l753_753052

theorem brothers_percentage_fewer_trees (total_trees initial_days brother_days : ℕ) (trees_per_day : ℕ) (total_brother_trees : ℕ) (percentage_fewer : ℕ):
  initial_days = 2 →
  brother_days = 3 →
  trees_per_day = 20 →
  total_trees = 196 →
  total_brother_trees = total_trees - (trees_per_day * initial_days) →
  percentage_fewer = ((total_brother_trees / brother_days - trees_per_day) * 100) / trees_per_day →
  percentage_fewer = 60 :=
by
  sorry

end brothers_percentage_fewer_trees_l753_753052


namespace sequence_y_finite_limit_l753_753749

noncomputable def sequence_x (n : ℕ) : ℝ :=
  if n = 1 then 1 else (2 * n / ((n - 1)^2)) * ∑ i in finset.range (n - 1), sequence_x i

def sequence_y (n : ℕ) : ℝ :=
  sequence_x (n + 1) - sequence_x n

theorem sequence_y_finite_limit :
  ∃ L : ℝ, tendsto (λ n : ℕ, sequence_y n) at_top (𝓝 L) :=
sorry

end sequence_y_finite_limit_l753_753749


namespace digit_30_of_sum_l753_753532

-- Defining the given fractions
def fraction1 : ℚ := 1 / 11
def fraction2 : ℚ := 1 / 13

-- Defining the repeating decimals
def decimal1 := "0.09".cycle
def decimal2 := "0.076923".cycle

-- Defining the least common multiple of the periods
def lcm_period := Nat.lcm 2 6

-- Defining the repeating block of the sum's decimal form
def repeating_sum_seq := "097032".cycle

-- The Lean 4 statement to be proved
theorem digit_30_of_sum : 
  (fraction1 + fraction2).decimalExpansion !30 == some 2 := 
by
  sorry

end digit_30_of_sum_l753_753532


namespace max_rate_of_increase_at_M0_direction_of_greatest_decrease_at_M1_l753_753150

noncomputable def u (x y z : ℝ) : ℝ := 10 / (x^2 + y^2 + z^2 + 1)

theorem max_rate_of_increase_at_M0 : 
  ∃ (M0 : ℝ × ℝ × ℝ), M0 = (-1, 2, -2) ∧
  let grad_u := λ (x y z : ℝ), 
    (-(20 * x) / ((x^2 + y^2 + z^2 + 1)^2),
     -(20 * y) / ((x^2 + y^2 + z^2 + 1)^2),
     -(20 * z) / ((x^2 + y^2 + z^2 + 1)^2)) in
  (sqrt ((grad_u (-1) 2 (-2)).1^2 + (grad_u (-1) 2 (-2)).2^2 + (grad_u (-1) 2 (-2)).3^2) = 3 / 5) := 
  sorry

theorem direction_of_greatest_decrease_at_M1 : 
  ∃ (M1 : ℝ × ℝ × ℝ), M1 = (2, 0, 1) ∧
  let grad_u := λ (x y z : ℝ), 
    (-(20 * x) / ((x^2 + y^2 + z^2 + 1)^2),
     -(20 * y) / ((x^2 + y^2 + z^2 + 1)^2),
     -(20 * z) / ((x^2 + y^2 + z^2 + 1)^2)) in
  ((grad_u 2 0 1).1 = -(10 / 9) ∧ (grad_u 2 0 1).2 = 0 ∧ (grad_u 2 0 1).3 = -(5 / 9)) := 
  sorry

end max_rate_of_increase_at_M0_direction_of_greatest_decrease_at_M1_l753_753150


namespace third_function_l753_753848

variable {X : Type} -- Define the type X representing the domain and codomain

-- Assuming that φ is a function on X
variable (φ : X → X)
-- Assuming φ has an inverse defined for it
variable (φ_inv : X → X) -- This is φ^(-1)

-- Add two conditions: 
-- 1. g(x) is the inverse function of f(x).
-- 2. h(x) is the function symmetric to g(x) with respect to the line x + y = 0.
def f (x : X) := φ x
def g (x : X) := φ_inv x

-- The third function h(x) should satisfy symmetry with g(x) respect to x + y = 0
def symmetric (p q : X × X) : Prop := p.1 + p.2 = 0 ∧ q.1 + q.2 = 0 ∧ p.1 = -q.2 ∧ p.2 = -q.1

-- Prove that h(x) = -φ_inv(-x)
theorem third_function :
  (∀ x y, (g x = y) → symmetric (x, y) (-y, -x)) →
  (∀ x : X, h x = -φ_inv (-x)) :=
by
  intro symm_proof
  intro x
  rw [← symm_proof x (-φ_inv (-x)) (by sorry)] -- symm_proof is used to imply the symmetry property
  sorry

end third_function_l753_753848


namespace a_4k_plus_2_divisible_by_3_l753_753146

noncomputable def a : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 3
| (n+3) := a n + a (n+1)

theorem a_4k_plus_2_divisible_by_3 (k : ℕ) : 3 ∣ a (4*k + 2) :=
by
  induction k with
  | zero =>
    simp [a]
    use 1
    refl
  | succ k ih =>
    have : a (4*(k+1) + 2) = a (4*k + 6), 
      by simp
    rw [this, a_eq]
    sorry

end a_4k_plus_2_divisible_by_3_l753_753146


namespace solve_for_x_l753_753293

theorem solve_for_x (x : ℝ) : (5 + x) / (8 + x) = (2 + x) / (3 + x) → x = -1 / 2 :=
by
  sorry

end solve_for_x_l753_753293


namespace tangent_expression_l753_753609

theorem tangent_expression (x : ℝ) : 
  tan (real.pi * (18/180 - x/180)) * tan (real.pi * (12/180 + x/180)) + 
  real.sqrt 3 * (tan (real.pi * (18/180 - x/180)) + tan (real.pi * (12/180 + x/180))) = 1 := 
by
  sorry

end tangent_expression_l753_753609


namespace horizontal_length_of_monitor_l753_753083

theorem horizontal_length_of_monitor 
  (aspect_ratio_width : ℕ) 
  (aspect_ratio_height : ℕ) 
  (diagonal_length : ℝ)
  (width_height_ratio : aspect_ratio_width = 16 ∧ aspect_ratio_height = 9)
  (diagonal_length_eq : diagonal_length = 30) 
  : 16 * 30 / Real.sqrt(16^2 + 9^2) ≈ 26.14 := 
by 
  sorry

end horizontal_length_of_monitor_l753_753083


namespace sequence_general_term_sum_of_cn_l753_753984

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) :
  (∀ n, S n = 2 * a n - 2) →
  (∀ n, a n = if n = 0 then 2 else 2 * a (n - 1)) →
  a n = 2^n :=
by
  intros hS ha
  sorry

theorem sum_of_cn (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℚ) (T : ℕ → ℚ) (n : ℕ) :
  (∀ n, b n = Real.logBase 2 (a n)) →
  (∀ n, c n = 1 / (b n * b (n + 1))) →
  (∀ n, a n = 2^n) →
  T n = ∑ i in Finset.range n, c i →
  T n = n / (n + 1) :=
by
  intros hb hc ha hT
  sorry

end sequence_general_term_sum_of_cn_l753_753984


namespace shaded_region_area_l753_753729

-- Define the constants
def diameter_small : ℝ := 5
def radius_small : ℝ := diameter_small / 2
def diameter_large : ℝ := 5 * 5
def radius_large : ℝ := diameter_large / 2

-- Area of small semicircle
def area_small : ℝ := (1 / 2) * Real.pi * radius_small^2

-- Area of large semicircle
def area_large : ℝ := (1 / 2) * Real.pi * radius_large^2

-- Shaded area
def area_shaded : ℝ := area_large + area_small

-- The conclusion regarding the area of the shaded region
theorem shaded_region_area :
  area_shaded = (325 * Real.pi) / 4 := by
  sorry

end shaded_region_area_l753_753729


namespace arithmetic_example_l753_753936

theorem arithmetic_example : 2546 + 240 / 60 - 346 = 2204 := by
  sorry

end arithmetic_example_l753_753936


namespace inverse_function_l753_753255

def f (x : ℝ) : ℝ := sqrt (x^2 + 1) + 1

theorem inverse_function :
  (∀ x, x < 0 → f (x) = y) → (∀ y, y > 2 → ∃ x, x > 2 ∧ y = - sqrt (x^2 - 2x)) :=
sorry

end inverse_function_l753_753255


namespace find_k_value_l753_753361

-- Define the condition that point A(3, -5) lies on the graph of the function y = k / x
def point_on_inverse_proportion (k : ℝ) : Prop :=
  (3 : ℝ) ≠ 0 ∧ (-5) = k / (3 : ℝ)

-- The theorem to prove that k = -15 given the point on the graph
theorem find_k_value (k : ℝ) (h : point_on_inverse_proportion k) : k = -15 :=
by
  sorry

end find_k_value_l753_753361


namespace cone_surface_area_equals_l753_753904

def side_length_from_area (area : ℝ) (h : area = Real.sqrt 3) : ℝ :=
  (4 * area) / Real.sqrt 3 

def radius_of_base (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 3) * a

def altitude (a r : ℝ) : ℝ :=
  Real.sqrt (a^2 - r^2)

def slant_height (h r : ℝ) : ℝ :=
  Real.sqrt (h^2 + r^2)

def surface_area_of_cone (r l : ℝ) : ℝ :=
  Real.pi * r^2 + Real.pi * r * l

theorem cone_surface_area_equals :
  let area_eq := Real.sqrt 3
  let a := side_length_from_area area_eq sorry 
  let r := radius_of_base a 
  let h := altitude a r
  let l := slant_height h r
  surface_area_of_cone r l = 3 * Real.pi := sorry

end cone_surface_area_equals_l753_753904


namespace count_Nitrogen_atoms_l753_753903

theorem count_Nitrogen_atoms
  (num_H : ℕ)
  (num_I : ℕ)
  (molecular_weight : ℕ)
  (atomic_weight_H : ℕ)
  (atomic_weight_N : ℕ)
  (atomic_weight_I : ℕ)
  (total_weight : num_H * atomic_weight_H + num_I * atomic_weight_I ≤ molecular_weight) :
  ((molecular_weight - (num_H * atomic_weight_H + num_I * atomic_weight_I)) / atomic_weight_N) = 1 :=
by
  have h_weight_H : num_H = 4 := by sorry
  have h_weight_I : num_I = 1 := by sorry
  have h_atomic_weight_H : atomic_weight_H = 1 := by sorry
  have h_atomic_weight_N : atomic_weight_N = 14 := by sorry
  have h_atomic_weight_I : atomic_weight_I = 127 := by sorry
  have h_molecular_weight : molecular_weight = 145 := by sorry
  rw [h_weight_H, h_weight_I, h_atomic_weight_H, h_atomic_weight_N, h_atomic_weight_I, h_molecular_weight] at *
  have step1 : num_H * atomic_weight_H = 4 := by sorry
  have step2 : num_I * atomic_weight_I = 127 := by sorry
  have step3 : (4 + 127) = 131 := by sorry
  have step4 : (145 - 131) = 14 := by sorry
  have step5 : (14 / 14) = 1 := by sorry
  exact step5

end count_Nitrogen_atoms_l753_753903


namespace average_value_of_integers_l753_753944

theorem average_value_of_integers (m : ℕ) (b : Fin m → ℤ) (h1 : ∑ i in Finset.range (m-1), b ⟨i, sorry⟩ = 42 * (m - 1))
  (h2 : ∑ i in Finset.range (2, m-1), b ⟨i, sorry⟩ = 48 * (m - 2))
  (h3 : ∑ i in Finset.range 1, b ⟨i, sorry⟩ = 55 * (m - 1))
  (h4 : b ⟨m-1, sorry⟩ = b 0 + 90) :
  (∑ i in Finset.range m, b ⟨i, sorry⟩) / (m : ℤ) = 96.875 :=
sorry

end average_value_of_integers_l753_753944


namespace sin_alpha_value_l753_753362

-- Definitions based on the conditions
structure Point where
  x : ℝ
  y : ℝ

def point_P : Point := {x := -3, y := 4}

def distance_from_origin (p : Point) : ℝ :=
  Real.sqrt (p.x ^ 2 + p.y ^ 2)

def angle_on_terminal_side (p : Point) : Prop :=
  distance_from_origin p = 5

-- The theorem to prove
theorem sin_alpha_value :
  angle_on_terminal_side point_P →
  (point_P.y / distance_from_origin point_P) = 4 / 5 := by
  sorry

end sin_alpha_value_l753_753362


namespace percent_decrease_1990_to_2010_correct_l753_753712

theorem percent_decrease_1990_to_2010_correct : 
  let cost_1990 := 55
  let cost_2000 := 23
  let cost_2010 := 10
  let total_decrease := cost_1990 - cost_2000 + cost_2000 - cost_2010
  (total_decrease : ℝ) / cost_1990 * 100 = 81.82 :=
by
  let cost_1990 := 55
  let cost_2000 := 23
  let cost_2010 := 10
  let total_decrease := cost_1990 - cost_2000 + cost_2000 - cost_2010
  have h : (total_decrease : ℝ) / cost_1990 * 100 = ((cost_1990 - cost_2010) : ℝ) / cost_1990 * 100,
  { -- Combine the decrease from two steps into one single step
    simp [total_decrease, cost_1990, cost_2010, cost_2000],
    sorry,
  }
  rw h,
  -- plug in the numbers and calculate percentage
  have h2 : ((cost_1990 - cost_2010) : ℝ) / cost_1990 * 100 = 81.82,
  { norm_num [(cost_1990 - cost_2010), cost_1990],
    sorry,
  },
  exact h2

end percent_decrease_1990_to_2010_correct_l753_753712


namespace original_price_of_pants_l753_753511

variable (B P : ℝ)

-- The initial price relations
def pants_price_relation : Prop := P = B - 2.93

-- Discounted prices with tax
def discounted_belt : ℝ := 0.85 * B
def discounted_pants : ℝ := 0.90 * P
def total_cost_with_tax : ℝ := (discounted_belt + discounted_pants) * 1.075

-- Given conditions
def conditions : Prop :=
  total_cost_with_tax = 70.93

-- Proof of the original price of pants
theorem original_price_of_pants
  (h1 : pants_price_relation)
  (h2 : conditions) :
  P = 36.28 :=
sorry

end original_price_of_pants_l753_753511


namespace number_of_albums_l753_753142

-- Definitions for the given conditions
def pictures_from_phone : ℕ := 7
def pictures_from_camera : ℕ := 13
def pictures_per_album : ℕ := 4

-- We compute the total number of pictures
def total_pictures : ℕ := pictures_from_phone + pictures_from_camera

-- Statement: Prove the number of albums is 5
theorem number_of_albums :
  total_pictures / pictures_per_album = 5 := by
  sorry

end number_of_albums_l753_753142


namespace arithmetic_sequence_sum_l753_753389

-- Define the arithmetic sequence
def arithmetic_sequence (a_3 a_15 : ℤ) (d : ℤ) (n : ℕ) : ℤ := 
  if n = 3 then a_3
  else if n = 15 then a_15
  else a_3 + (n - 3) * d

-- Define the problem statement
theorem arithmetic_sequence_sum (a_3 a_15 : ℤ) (d : ℤ) :
  (a_3 + a_15 = 6) → 
  (a_7 + a_8 + a_9 + a_{10} + a_{11}) = 15 := 
by
  sorry

end arithmetic_sequence_sum_l753_753389


namespace smallest_positive_x_cos_eq_cos3x_l753_753076

theorem smallest_positive_x_cos_eq_cos3x :
  ∃ x : ℝ, x > 0 ∧ cos x = cos (3 * x) ∧ x = Float.round 1 2 (Float.pi / 2) :=
sorry

end smallest_positive_x_cos_eq_cos3x_l753_753076


namespace remainder_when_sum_divided_by_5_l753_753156

theorem remainder_when_sum_divided_by_5 (f y : ℤ) (k m : ℤ) 
  (hf : f = 5 * k + 3) (hy : y = 5 * m + 4) : 
  (f + y) % 5 = 2 := 
by {
  sorry
}

end remainder_when_sum_divided_by_5_l753_753156


namespace area_one_fourth_l753_753768

theorem area_one_fourth {A B C M K L : Point}
  (on_AB : M ∈ segment A B) (on_BC : K ∈ segment B C) (on_CA : L ∈ segment C A) :
  area (triangle M A L) ≤ (1 / 4) * area (triangle A B C) ∨
  area (triangle K B M) ≤ (1 / 4) * area (triangle A B C) ∨
  area (triangle L C K) ≤ (1 / 4) * area (triangle A B C) := 
by 
  sorry

end area_one_fourth_l753_753768


namespace range_of_t_l753_753685

theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * x + t ≤ 0 ∧ x ≤ t) ↔ (0 ≤ t ∧ t ≤ 9 / 4) := 
sorry

end range_of_t_l753_753685


namespace car_a_speed_l753_753237

theorem car_a_speed (d_A d_B v_B t v_A : ℝ)
  (h1 : d_A = 10)
  (h2 : v_B = 50)
  (h3 : t = 2.25)
  (h4 : d_A + 8 - d_B = v_A * t)
  (h5 : d_B = v_B * t) :
  v_A = 58 :=
by
  -- Work on the proof here
  sorry

end car_a_speed_l753_753237


namespace initial_students_count_l753_753901

variable (initial_students : ℕ)
variable (number_of_new_boys : ℕ := 5)
variable (initial_percentage_girls : ℝ := 0.40)
variable (new_percentage_girls : ℝ := 0.32)

theorem initial_students_count (h : initial_percentage_girls * initial_students = new_percentage_girls * (initial_students + number_of_new_boys)) : 
  initial_students = 20 := 
by 
  sorry

end initial_students_count_l753_753901


namespace surface_area_is_correct_volume_is_approximately_correct_l753_753919

noncomputable def surface_area_of_CXYZ (height : ℝ) (side_length : ℝ) : ℝ :=
  let area_CZX_CZY := 48
  let area_CXY := 9 * Real.sqrt 3
  let area_XYZ := 9 * Real.sqrt 15
  2 * area_CZX_CZY + area_CXY + area_XYZ

theorem surface_area_is_correct (height : ℝ) (side_length : ℝ) (h : height = 24) (s : side_length = 18) :
  surface_area_of_CXYZ height side_length = 96 + 9 * Real.sqrt 3 + 9 * Real.sqrt 15 :=
by
  sorry

noncomputable def volume_of_CXYZ (height : ℝ ) (side_length : ℝ) : ℝ :=
  -- Placeholder for the volume calculation approximation method.
  486

theorem volume_is_approximately_correct
  (height : ℝ) (side_length : ℝ) (h : height = 24) (s : side_length = 18) :
  volume_of_CXYZ height side_length = 486 :=
by
  sorry

end surface_area_is_correct_volume_is_approximately_correct_l753_753919


namespace time_taken_by_Arun_to_cross_train_B_l753_753166

structure Train :=
  (length : ℕ)
  (speed_kmh : ℕ)

def to_m_per_s (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000) / 3600

def relative_speed (trainA trainB : Train) : ℕ :=
  to_m_per_s trainA.speed_kmh + to_m_per_s trainB.speed_kmh

def total_length (trainA trainB : Train) : ℕ :=
  trainA.length + trainB.length

def time_to_cross (trainA trainB : Train) : ℕ :=
  total_length trainA trainB / relative_speed trainA trainB

theorem time_taken_by_Arun_to_cross_train_B :
  time_to_cross (Train.mk 175 54) (Train.mk 150 36) = 13 :=
by
  sorry

end time_taken_by_Arun_to_cross_train_B_l753_753166


namespace find_x_that_satisfies_sum_l753_753498

theorem find_x_that_satisfies_sum :
  ∀ (x : ℕ), 
    let row1 := [11, 6, x, 7],
        row2 := [11, 6 + x, x + 7],
        row3 := [11 + (6 + x), (6 + x) + (x + 7)],
        final_value := (11 + (6 + x)) + ((6 + x) + (x + 7)) in
    final_value = 60 → x = 10 :=
by
  intros x row1 row2 row3 final_value
  unfold row1 row2 row3 final_value
  sorry

end find_x_that_satisfies_sum_l753_753498


namespace range_of_sum_of_distances_squares_l753_753688

-- Define parametric curve C1
def C1 (φ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos φ, 3 * Real.sin φ)

-- Define the vertices of square ABCD
def A := (0 : ℝ, 2 : ℝ)
def B := (-2 : ℝ, 0 : ℝ)
def C := (0 : ℝ, -2 : ℝ)
def D := (2 : ℝ, 0 : ℝ)

-- Define the Euclidean distance squared
def dist_sq (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Statement of the theorem
theorem range_of_sum_of_distances_squares (φ : ℝ) :
  let P := C1 φ in
  let S := dist_sq P A + dist_sq P B + dist_sq P C + dist_sq P D in
  32 ≤ S ∧ S ≤ 52 :=
sorry

end range_of_sum_of_distances_squares_l753_753688


namespace stratified_sampling_large_class_l753_753828

theorem stratified_sampling_large_class 
  (small_class : ℕ) (medium_class : ℕ) (large_class : ℕ) (sample_size : ℕ)
  (h_small : small_class = 90) (h_medium : medium_class = 90) (h_large : large_class = 120)
  (h_sample : sample_size = 50) :
  let total_students := small_class + medium_class + large_class
  let proportion_large := large_class.to_rat / total_students.to_rat
  let sample_large := (proportion_large * sample_size.to_rat).floor
  sample_large = 20 :=
by
  have h_total : total_students = 300 := by rw [h_small, h_medium, h_large]; norm_num
  have h_proportion : proportion_large = (2 / 5 : ℚ) := by
    rw [←h_large, h_total]; norm_num
  have h_sample_large : sample_large = 20 := by
    rw [h_sample, h_proportion]; norm_num
  exact h_sample_large
  sorry

end stratified_sampling_large_class_l753_753828


namespace Lara_likes_numbers_number_of_last_digits_Lara_likes_number_of_different_last_digits_Lara_likes_l753_753782

theorem Lara_likes_numbers {n : ℕ} : (n % 3 = 0 ∧ n % 5 = 0) → (n % 10 = 0) :=
by sorry

theorem number_of_last_digits_Lara_likes : ∀ (n : ℕ), (n % 3 = 0 ∧ n % 5 = 0) → (∃ k : ℕ, (n % 10 = k ∧ k = 0)) :=
by
  intro n h
  have h₁ : n % 15 = 0, from (by sorry),
  exact ⟨0, by sorry⟩

theorem number_of_different_last_digits_Lara_likes : ∃ (k : ℕ), k = 1 :=
by
  use 1
  exact by sorry

end Lara_likes_numbers_number_of_last_digits_Lara_likes_number_of_different_last_digits_Lara_likes_l753_753782


namespace eccentricity_of_ellipse_is_sqrt2_div2_l753_753671

noncomputable def eccentricity_ellipse : ℝ :=
  let a := classical.some (show ∃ a, a > 0, from sorry)
  let b := classical.some (show ∃ b, b > 0 ∧ b < a, from sorry)
  let e := b / a
  sqrt 2 / 2

theorem eccentricity_of_ellipse_is_sqrt2_div2
  (a b : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : let m := sqrt (a^2 - b^2), n := b in m = n) :
  eccentricity_ellipse = sqrt 2 / 2 :=
by {
  sorry
}

end eccentricity_of_ellipse_is_sqrt2_div2_l753_753671


namespace more_than_four_numbers_make_polynomial_prime_l753_753642

def polynomial (n : ℕ) : ℤ := n^3 - 10 * n^2 + 31 * n - 17

def is_prime (k : ℤ) : Prop :=
  k > 1 ∧ ∀ m : ℤ, m > 1 ∧ m < k → ¬ (m ∣ k)

theorem more_than_four_numbers_make_polynomial_prime :
  (∃ n1 n2 n3 n4 n5 : ℕ, 
    n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧ n4 > 0 ∧ n5 > 0 ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ n1 ≠ n5 ∧
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ n2 ≠ n5 ∧ 
    n3 ≠ n4 ∧ n3 ≠ n5 ∧ 
    n4 ≠ n5 ∧ 
    is_prime (polynomial n1) ∧
    is_prime (polynomial n2) ∧
    is_prime (polynomial n3) ∧
    is_prime (polynomial n4) ∧
    is_prime (polynomial n5)) :=
sorry

end more_than_four_numbers_make_polynomial_prime_l753_753642


namespace binary_to_base5_l753_753248

theorem binary_to_base5 (n : ℕ) (h : n = 45) : nat_to_base 5 n = "140" :=
by {
  -- The proof would involve steps to show the conversion,
  -- but since the proof is not required, we use sorry.
  sorry
}

end binary_to_base5_l753_753248


namespace smallest_x_l753_753353

theorem smallest_x (x y : ℕ) (h1 : 0.75 = y / (254 + x)) (h2 : x > 0) (h3 : y > 0) : x = 2 := 
by
  sorry

end smallest_x_l753_753353


namespace total_value_is_76_percent_of_dollar_l753_753441

def coin_values : List Nat := [1, 5, 20, 50]

def total_value (coins : List Nat) : Nat :=
  List.sum coins

def percentage_of_dollar (value : Nat) : Nat :=
  value * 100 / 100

theorem total_value_is_76_percent_of_dollar :
  percentage_of_dollar (total_value coin_values) = 76 := by
  sorry

end total_value_is_76_percent_of_dollar_l753_753441


namespace distinct_lengths_from_E_to_DF_l753_753796

noncomputable def distinct_integer_lengths (DE EF: ℕ) : ℕ :=
if h : DE = 15 ∧ EF = 36 then 24 else 0

theorem distinct_lengths_from_E_to_DF :
  distinct_integer_lengths 15 36 = 24 :=
by {
  sorry
}

end distinct_lengths_from_E_to_DF_l753_753796


namespace gilbert_herb_garden_total_plants_l753_753296

theorem gilbert_herb_garden_total_plants :
  let initial_basil := 3,
      initial_parsley := 1,
      initial_mint := 2,
      initial_rosemary := 1,
      initial_thyme := 1,
      final_basil := 26, -- from growth and additions as per conditions
      final_parsley := 5, -- from growth and reductions by caterpillars
      final_mint := 2, -- ruined halfway, no growth
      final_rosemary := 1, -- stayed the same
      final_thyme := 1 -- stayed the same
  in initial_basil + initial_parsley + initial_mint + initial_rosemary + initial_thyme 
     = final_basil + final_parsley + final_mint + final_rosemary + final_thyme
  → final_basil + final_parsley + final_mint + final_rosemary + final_thyme = 35 :=
by sorry

end gilbert_herb_garden_total_plants_l753_753296


namespace range_of_a_l753_753316

theorem range_of_a (a : ℝ) (p : a^2 - 5 * a - 3 ≥ 3) (q : ¬(∃ x : ℝ, x^2 + a * x + 2 < 0)) : -real.sqrt 8 ≤ a ∧ a ≤ -1 :=
begin
  sorry
end

end range_of_a_l753_753316


namespace time_to_fill_pool_l753_753808

theorem time_to_fill_pool (V : ℕ) (n : ℕ) (r : ℕ) (fill_rate_per_hour : ℕ) :
  V = 24000 → 
  n = 4 →
  r = 25 → -- 2.5 gallons per minute expressed as 25/10 gallons
  fill_rate_per_hour = (n * r * 6) → -- since 6 * 10 = 60 (to convert per minute rate to per hour, we divide so r is 25 instead of 2.5)
  V / fill_rate_per_hour = 40 :=
by
  sorry

end time_to_fill_pool_l753_753808


namespace neg_root_sufficient_not_necessary_l753_753563

theorem neg_root_sufficient_not_necessary (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ (a < 0) :=
sorry

end neg_root_sufficient_not_necessary_l753_753563


namespace fn_natural_l753_753089

noncomputable def f : ℕ → ℕ
| 0     := 3
| 1     := 21
| 2     := 371
| (n+3) := 21 * f (n+2) - 35 * f (n+1) + 7 * f n

theorem fn_natural (n : ℕ) : Nat := f n

end fn_natural_l753_753089


namespace minimize_Sn_l753_753310

-- Define the arithmetic sequence with given conditions
variables (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (d : ℤ)

-- Define the conditions given in the problem
axiom a_arithmetic_sequence : ∀ n m, a (m + 1) - a m = d
axiom sum_first_n : ∀ n, S n = n * (a 1 + a n) / 2
axiom cond1 : a 2 + a 4 = -22
axiom cond2 : a 1 + a 4 + a 7 = -21

theorem minimize_Sn : S 5 < S n ∀ n ≠ 5 := 
sorry

end minimize_Sn_l753_753310


namespace hypotenuse_length_is_4_l753_753718

-- Define the right triangle with specific conditions
structure RightTriangle (α β H O : ℝ) :=
  (is_right_triangle : β = 30)
  (hypotenuse : H = 2 * O)

-- Given conditions: right triangle, angle 30 degrees, opposite side length 2 cm
def given_conditions : RightTriangle :=
  { is_right_triangle := by rfl,
    hypotenuse := 4 = 2 * 2 }

-- Proof statement to show that the hypotenuse length is 4 cm
theorem hypotenuse_length_is_4 : ∃ (H : ℝ), H = 4 :=
  sorry

end hypotenuse_length_is_4_l753_753718


namespace tan_three_pi_over_four_l753_753959

theorem tan_three_pi_over_four :
  ∀ (θ : ℝ), (θ = π / 4) → (tan (π - θ) = -tan θ) → (tan (π / 4) = 1) → tan (3 * π / 4) = -1 :=
by
  intros θ h1 h2 h3
  rw [← h1, tan_pi_sub θ h3, h3]
  sorry

end tan_three_pi_over_four_l753_753959


namespace alice_pens_count_l753_753595

theorem alice_pens_count :
  ∃ A : ℕ, ∃ C : ℕ,
  let alice_age := 20 in
  let clara_current_age := 61 - 5 in
  let pens_diff := 36 in
  C = (2 / 5 : ℚ) * A ∧
  A - C = pens_diff ∧
  A = 60 := 
by
  sorry

end alice_pens_count_l753_753595


namespace tangency_angle_equality_l753_753985

-- Definitions based on the conditions
variable {A B C : Point} (ABC : Triangle A B C)
variable (Γ : Circle) -- Circumcircle of triangle ABC
variable {C1 : Circle} -- A-mixtilinear incircle
variable {C2 : Circle} -- A-excircle
variable (S T : Point)

-- Given conditions
axiom Tangency_C1_Γ : Tangent C1 Γ S
axiom Tangency_C2_BC : Tangent C2 (Segment B C) T

-- The proof statement
theorem tangency_angle_equality : ∠ BAS = ∠ TAC :=
  sorry

end tangency_angle_equality_l753_753985


namespace find_a_l753_753010

theorem find_a (a : ℝ) (h1 : 0 < a)
  (c1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (c2 : ∀ x y : ℝ, x^2 + y^2 + 2 * a * y - 6 = 0)
  (h_chord : (2 * Real.sqrt 3) = 2 * Real.sqrt 3) :
  a = 1 := 
sorry

end find_a_l753_753010


namespace articleWords_l753_753201

-- Define the number of words per page for larger and smaller types
def wordsLargerType : Nat := 1800
def wordsSmallerType : Nat := 2400

-- Define the total number of pages and the number of pages in smaller type
def totalPages : Nat := 21
def smallerTypePages : Nat := 17

-- The number of pages in larger type
def largerTypePages : Nat := totalPages - smallerTypePages

-- Calculate the total number of words in the article
def totalWords : Nat := (largerTypePages * wordsLargerType) + (smallerTypePages * wordsSmallerType)

-- Prove that the total number of words in the article is 48,000
theorem articleWords : totalWords = 48000 := 
by
  sorry

end articleWords_l753_753201


namespace ratio_EG_GF_l753_753043

variables {A B C M E F G : Type}
variables [Field A]

-- Definitions
noncomputable def midpoint (B C : A) : A := (B + C) / 2
noncomputable def ratio_segment (A B : A) (p : Prop) (r : ℝ) : Prop := p → (A = r * B)
noncomputable def line_intersection (A B C : A) (l1 l2 : Prop) : A := sorry

-- Given conditions
variables (B C : A) (hM : M = midpoint B C)
variables (hAB : (A - B).abs = 15)
variables (hAC : (A - C).abs = 20)
variables (hE : E = sorry) -- E lies on AC
variables (hF : F = sorry) -- F lies on AB
variables (hAE_AF : AE = 3 * AF)
variables (hG : G = line_intersection F E A M)

-- Proof statement
theorem ratio_EG_GF : ∀ (A B C M E F G : A), M = midpoint B C → (A - B).abs = 15 → (A - C).abs = 20 → (AE = 3 * AF) → G = line_intersection F E A M → 
    ratio_segment E G G (2 / 3) :=
sorry

end ratio_EG_GF_l753_753043


namespace quadrilateral_ADEC_is_rectangle_l753_753456

-- Define the conditions
variable (A B C D E : Type) 
variable (parallelogram_ABCD : Parallelogram A B C D)
variable (parallel_AD_BC : Parallel AD BC)
variable (parallel_DE_AC : Parallel DE AC)
variable (perpendicular_AC_BC : Perpendicular AC BC)

-- Define the theorem
theorem quadrilateral_ADEC_is_rectangle :
  IsRectangle A D E C :=
by
  -- proof goes here
  sorry

end quadrilateral_ADEC_is_rectangle_l753_753456


namespace find_dot_product_l753_753731

noncomputable def vec : Type := sorry
axiom inner_product : vec → vec → ℝ
axiom magnitude : vec → ℝ

variables (AB EF CD : ℝ) 
variables (AD BC AC BD : vec)

-- Given conditions
axiom h1 : AB = 1
axiom h2 : EF = real.sqrt 2
axiom h3 : CD = 3
axiom h4 : inner_product AD BC = 15

-- // Problem statement
theorem find_dot_product (α β γ δ : vec) :
    (\(AB = 1) \wedge (EF = sqrt(2)) \wedge (CD = 3) \wedge (inner_product AD BC = 15)) ->
    (inner_product AC BD = 16) :=
begin
    sorry,
end

end find_dot_product_l753_753731


namespace imaginary_unit_cubic_l753_753492

def imaginary_unit_property (i : ℂ) : Prop :=
  i^2 = -1

theorem imaginary_unit_cubic (i : ℂ) (h : imaginary_unit_property i) : 1 + i^3 = 1 - i :=
  sorry

end imaginary_unit_cubic_l753_753492


namespace find_phi_l753_753107

noncomputable def omega := 2 / 3
noncomputable def f (x : ℝ) := 2 * Real.sin (omega * x + (π / 12))

theorem find_phi
    (h0 : omega > 0)
    (h1 : abs (π / 12) < π / 2)
    (h2 : ∀ x, f x = f (5 * π / 4 - (x - 5 * π / 8)))
    (h3 : f (11 * π / 8) = 0)
    (h4 : 2 * π < 3 * π) :
    ∃ φ, φ = π / 12 :=
by
  -- Proof here
  sorry

end find_phi_l753_753107


namespace min_students_discussing_same_problem_l753_753028

theorem min_students_discussing_same_problem (n : ℕ)
  (discusses : Fin n → Fin n → ℕ)
  (h_discussion : ∀ i j : Fin n, i ≠ j → discusses i j ∈ {0, 1, 2}) :
  (∃ s : Finset (Fin n), s.card = 3 ∧ ∀ i j : Fin 3, 
     i ≠ j → discusses s[i] s[j] = discusses s[i] s[j]) ↔ n ≥ 17 :=
begin
  sorry
end

end min_students_discussing_same_problem_l753_753028


namespace perimeter_area_ratio_le_8_l753_753066

/-- Let \( S \) be a shape in the plane obtained as a union of finitely many unit squares.
    The perimeter of a single unit square is 4 and its area is 1.
    Prove that the ratio of the perimeter \( P \) and the area \( A \) of \( S \)
    is at most 8, i.e., \(\frac{P}{A} \leq 8\). -/
theorem perimeter_area_ratio_le_8
  (S : Set (ℝ × ℝ)) 
  (unit_square : ∀ (x y : ℝ), (x, y) ∈ S → (x + 1, y + 1) ∈ S ∧ (x + 1, y) ∈ S ∧ (x, y + 1) ∈ S ∧ (x, y) ∈ S)
  (P A : ℝ)
  (unit_square_perimeter : ∀ (x y : ℝ), (x, y) ∈ S → P = 4)
  (unit_square_area : ∀ (x y : ℝ), (x, y) ∈ S → A = 1) :
  P / A ≤ 8 :=
sorry

end perimeter_area_ratio_le_8_l753_753066


namespace black_squares_in_45th_row_l753_753244

-- Definitions based on the conditions
def number_of_squares_in_row (n : ℕ) : ℕ := 2 * n + 1

def number_of_black_squares (total_squares : ℕ) : ℕ := (total_squares - 1) / 2

-- The theorem statement
theorem black_squares_in_45th_row : number_of_black_squares (number_of_squares_in_row 45) = 45 :=
by sorry

end black_squares_in_45th_row_l753_753244


namespace geometric_sequence_sum_correct_l753_753983

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 2 then 2^(n + 1) - 2
else 64 * (1 - (1 / 2)^n)

theorem geometric_sequence_sum_correct (a1 q : ℝ) (n : ℕ) 
  (h1 : q > 0) 
  (h2 : a1 + a1 * q^4 = 34) 
  (h3 : a1^2 * q^4 = 64) :
  geometric_sequence_sum a1 q n = 
  if q = 2 then 2^(n + 1) - 2 else 64 * (1 - (1 / 2)^n) :=
sorry

end geometric_sequence_sum_correct_l753_753983


namespace green_balloons_count_l753_753935

-- Define the conditions
def total_balloons : Nat := 50
def red_balloons : Nat := 12
def blue_balloons : Nat := 7

-- Define the proof problem
theorem green_balloons_count : 
  let green_balloons := total_balloons - (red_balloons + blue_balloons)
  green_balloons = 31 :=
by
  sorry

end green_balloons_count_l753_753935


namespace find_2a_plus_b_l753_753495

variable (a b : ℝ)

-- Define the function and its derivative
def func (x : ℝ) := x^3 + a * x + b
def func_derivative (x : ℝ) := 3 * x^2 + a

-- Define the tangent line
def tangent_line (k : ℝ) (x : ℝ) := k * x + 1

theorem find_2a_plus_b (k : ℝ) (h_tangent : ∀ x, x = 1 → func x = tangent_line k x)
  (slope_eq : k = 3 + a) (point_on_curve : func 1 = 3) : 2 * a + b = 1 :=
by sorry

end find_2a_plus_b_l753_753495


namespace simplify_expression_l753_753939

theorem simplify_expression : 
  ( ( √2 / 2 ) * ( ( 2 * √12 ) / ( 4 * √(1 / 8) ) - ( 3 * √48 ) ) ) = ( 2 * √3 - 6 * √6 ) :=
by
  sorry

end simplify_expression_l753_753939


namespace correct_sequence_count_l753_753463

def sequence_count (set1 : Finset Char) (set2 : Finset ℕ) : ℕ :=
  let choices := set1.to_list.product (set1.to_list.erase set1.find_Choice_Char).map (λ _, set2.to_list.product (set2.to_list.erase set2.find_Choice_Int).map ((λ set1 set2 : List _,
    if ('O' ∈ set1 ∨ 'Q' ∈ set1 ∨ 0 ∈ set2.mixChar2Int) then  0 else 1))
  else sorry

theorem correct_sequence_count : sequence_count {'O', 'P', 'Q', 'R', 'S'} (Finset.range 10) = 8424 := by
  sorry

end correct_sequence_count_l753_753463


namespace greatest_number_of_cool_cells_l753_753421

noncomputable def greatest_cool_cells (n : ℕ) (grid : Matrix (Fin n) (Fin n) ℝ) : ℕ :=
n^2 - 2 * n + 1

theorem greatest_number_of_cool_cells (n : ℕ) (grid : Matrix (Fin n) (Fin n) ℝ) (h : 0 < n) :
  ∃ m, m = (n - 1)^2 ∧ m = greatest_cool_cells n grid :=
sorry

end greatest_number_of_cool_cells_l753_753421


namespace complex_ratio_problem_l753_753079

/-- Given non-zero complex numbers a, b, c such that a / b = b / c = c / a, 
prove that the value of (a + b - c) / (a - b + c) is one of 1, ω, ω^2, where
ω is a non-real cube root of unity (ω = (-1 + sqrt(3) * I) / 2). -/
theorem complex_ratio_problem
  (a b c : ℂ)
  (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0)
  (h : a / b = b / c ∧ b / c = c / a)
  (ω : ℂ) (hω : ω = (-1 + Complex.sqrt 3 * Complex.I) / 2) :
  (a + b - c) / (a - b + c) ∈ ({1, ω, ω^2} : Set ℂ) := 
sorry

end complex_ratio_problem_l753_753079


namespace right_triangle_integer_segments_count_l753_753798

theorem right_triangle_integer_segments_count :
  ∀ (DE EF : ℕ), DE = 15 → EF = 36 → 
  let DF := Real.sqrt (DE^2 + EF^2) in
  let area := (DE * EF) / 2 in
  ∃ (integer_segment_count : ℕ),
  (integer_segment_count = 24) := 
by
  intros DE EF hDE hEF
  let DF := Real.sqrt (DE^2 + EF^2)
  let area := (DE * EF) / 2
  use 24
  sorry

end right_triangle_integer_segments_count_l753_753798


namespace gold_bars_left_after_tax_and_divorce_l753_753049

def initial_gold_bars := 60
def tax_rate := 0.10
def tax_paid := initial_gold_bars * tax_rate
def remaining_after_tax := initial_gold_bars - tax_paid
def divorce_loss := remaining_after_tax / 2
def gold_bars_left := remaining_after_tax - divorce_loss

theorem gold_bars_left_after_tax_and_divorce :
  gold_bars_left = 27 := 
by 
  simp [initial_gold_bars, tax_rate, tax_paid, remaining_after_tax, divorce_loss, gold_bars_left]
  sorry

end gold_bars_left_after_tax_and_divorce_l753_753049


namespace line_tangent_to_circle_l753_753830

noncomputable def distance_point_to_line {α : Type*} [LinearOrderedField α] 
  (x₀ y₀ A B C : α) : α := 
  abs (A * x₀ + B * y₀ + C) / sqrt (A^2 + B^2)

theorem line_tangent_to_circle : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1) →
  (∀ x y : ℝ, x / 4 + y / 3 = 1) →
  ∃ d : ℝ, d = 1 :=
by 
  intros circle_equation line_equation
  let cx := 1
  let cy := 1
  let r := 1
  let A := 1 / 4
  let B := 1 / 3
  let C := -1
  let d := distance_point_to_line cx cy A B C
  have : d = r := sorry
  exact ⟨d, this⟩

end line_tangent_to_circle_l753_753830


namespace triangle_inequality_l753_753644

theorem triangle_inequality
  (ABC : Triangle)
  (I : Point) -- assuming I is the incenter of triangle ABC
  (R_ABC : circumcircle_radius ABC)
  (R_ABI : circumcircle_radius (Triangle.mk A B I))
  (R_BCI : circumcircle_radius (Triangle.mk B C I))
  (R_CAI : circumcircle_radius (Triangle.mk C A I))
  (dAI dBI dCI: ℝ) -- assuming these are the distances AI, BI, and CI
  (h1 : dAI = distance A I)
  (h2 : dBI = distance B I)
  (h3 : dCI = distance C I)
  (h4 : R_ABI = circumcircle_radius (Triangle.mk A B I))
  (h5 : R_BCI = circumcircle_radius (Triangle.mk B C I))
  (h6 : R_CAI = circumcircle_radius (Triangle.mk C A I))
: 
  (1 / R_ABI + 1 / R_BCI + 1 / R_CAI) ≤ (1 / dAI + 1 / dBI + 1 / dCI) :=
sorry

end triangle_inequality_l753_753644


namespace prob_palindrome_7_digit_l753_753392

def is_palindrome (n : ℕ) : Prop := 
  let digits := List.ofNats n
  digits = digits.reverse

def count_palindromes (total_digits : ℕ) : ℕ :=
  if total_digits = 7 then 
    let valid_first_digits := 9 -- digits 1 to 9
    let rest_digits := 10 -- digits 0 to 9
    valid_first_digits * rest_digits ^ 3
  else 0

def count_total_phones (total_digits : ℕ) : ℕ :=
  if total_digits = 7 then 
    let valid_first_digits := 9 -- digits 1 to 9
    let rest_digits := 10 -- digits 0 to 9
    valid_first_digits * rest_digits ^ 6
  else 0

theorem prob_palindrome_7_digit : 
  count_palindromes 7 / count_total_phones 7 = 0.001 := 
by 
  sorry

end prob_palindrome_7_digit_l753_753392


namespace find_ratio_of_EK_KF_l753_753600

-- Define the cube and vertices
variables (A B C D E F G H K: Type) 
-- A, B, C, D, E, F, G, H are vertices of the cube
-- K is a point on edge EF
variables (cube : Set (Type × Type)) [is_cube : has_edges AB BC AD AE BF CG CF DG EH]

-- Define the plane passing through points A, C, and K
variables (α : Type) [is_plane α A C K]

-- Define the volume ratio condition
variable (volume_ratio : ℝ)
axiom volume_condition : volume_ratio = 3 / 1

-- Define the ratio EK/KF
variables (EK KF : ℝ)
axiom length_ratio : EK / KF = sqrt 3

theorem find_ratio_of_EK_KF : EK / KF = sqrt 3 :=
by 
  sorry

end find_ratio_of_EK_KF_l753_753600


namespace triangle_BC_length_l753_753020

noncomputable def BC_length (AB : ℝ) (angle_ABC : ℝ) (angle_ACB : ℝ) : ℝ :=
  sqrt(4 / 3)

theorem triangle_BC_length :
  ∀ (AB : ℝ) (angle_ABC : ℝ) (angle_ACB : ℝ),
  AB = 3 → angle_ABC = 75 → angle_ACB = 60 → BC_length AB angle_ABC angle_ACB = sqrt(4 / 3) :=
by
  intros AB angle_ABC angle_ACB h1 h2 h3
  sorry

end triangle_BC_length_l753_753020


namespace parallel_condition_necessary_not_sufficient_l753_753312

-- Defining non-zero vectors a and b
variables {a b : Type} [InnerProductSpace ℝ a] [InnerProductSpace ℝ b]

-- Assume vectors a and b are nonzero
axiom nonzero_a : a ≠ 0
axiom nonzero_b : b ≠ 0

-- Definition of parallel vectors
def parallel (a b : a) : Prop := ∃ k : ℝ, k ≠ 0 ∧ b = k • a

-- The condition we need to prove
theorem parallel_condition_necessary_not_sufficient :
  (∀ a b, parallel a b → nonzero_a → nonzero_b → a ≠ 0) ∧ ¬(∀ a b, nonzero_a → nonzero_b → a ≠ 0 → b = k • a) :=
by
  sorry

end parallel_condition_necessary_not_sufficient_l753_753312


namespace cyclic_quadrilateral_DLMF_cyclic_quadrilateral_BDKME_l753_753416

-- Definitions of points and properties given in the problem
variables (A B C O D E F K L M : Point)
variable (R : ℝ)
variable (c : Circle O R)
variable (c1 : Circle B (dist B A))
variable (AB_lt_AC : A.dist B < A.dist C)
variable (non_isosceles_acute_triangle_ABC : Triangle A B C ∧ ¬is_isosceles A B C ∧ is_acute A B C)
variable (inscribed_in_c : ∀ P, Triangle A B C ∧ P.is_on_circle c)
variable (c1_crosses_AC_at_K : K.is_on_circle c1 ∧ P.on_line A C)
variable (c1_crosses_c_at_E : E.is_on_circle c1 ∧ E.is_on_circle c)
variable (KE_crosses_c_at_F : F.is_on_line_segment K E ∧ F.is_on_circle c)
variable (BO_crosses_KE_at_L : L.is_on_line_segment B O ∧ L.is_on_line_segment K E)
variable (BO_crosses_AC_at_M : M.is_on_line_segment B O ∧ M.is_on_line_segment A C)
variable (AE_crosses_BF_at_D : D.is_on_line_segment A E ∧ D.is_on_line_segment B F)

-- Part (i): D, L, M, F are concyclic
theorem cyclic_quadrilateral_DLMF :
  cyclic_quadrilateral D L M F := sorry

-- Part (ii): B, D, K, M, E are concyclic
theorem cyclic_quadrilateral_BDKME :
  cyclic_quadrilateral B D K M E := sorry

end cyclic_quadrilateral_DLMF_cyclic_quadrilateral_BDKME_l753_753416


namespace avery_donates_16_clothes_l753_753226

theorem avery_donates_16_clothes : 
  let Shirts := 4
  let Pants := 4 * 2
  let Shorts := (4 * 2) / 2
in Shirts + Pants + Shorts = 16 :=
by 
  let Shirts := 4
  let Pants := 4 * 2
  let Shorts := (4 * 2) / 2
  show Shirts + Pants + Shorts = 16
  sorry

end avery_donates_16_clothes_l753_753226


namespace arithmetic_sequence_n_l753_753927

theorem arithmetic_sequence_n
  (x : ℝ)
  (b : ℕ → ℝ)
  (h1 : b 1 = x^2)
  (h2 : b 2 = x^2 + x)
  (h3 : b 3 = x^2 + 2x)
  : ∃ n : ℕ, b n = 2 * x^2 + 7 * x ∧ n = 8 := 
by 
  sorry

end arithmetic_sequence_n_l753_753927


namespace number_of_correct_statements_l753_753218

def statement1_proof (height_estimation_eachother_analogical : Bool) : Bool :=
  height_estimation_eachother_analogical = false

def statement2_proof (timely_snow_inductive : Bool) : Bool :=
  timely_snow_inductive = true

def statement3_proof (circle_to_sphere_analogical : Bool) : Bool :=
  circle_to_sphere_analogical = true

def statement4_proof (integer_ending_5_deductive : Bool) : Bool :=
  integer_ending_5_deductive = true

theorem number_of_correct_statements
  (height_estimation_eachother_analogical : Bool)
  (timely_snow_inductive : Bool)
  (circle_to_sphere_analogical : Bool)
  (integer_ending_5_deductive : Bool) :
  (statement1_proof height_estimation_eachother_analogical +
   statement2_proof timely_snow_inductive +
   statement3_proof circle_to_sphere_analogical +
   statement4_proof integer_ending_5_deductive = 2) :=
begin
  sorry
end

end number_of_correct_statements_l753_753218


namespace find_f91_plus_fm91_l753_753761

def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 3

theorem find_f91_plus_fm91 (a b c : ℝ) (h : f 91 a b c = 1) : f 91 a b c + f (-91) a b c = 2 := by
  sorry

end find_f91_plus_fm91_l753_753761


namespace find_x_l753_753869

theorem find_x (x y : ℤ) (some_number : ℤ) (h1 : y = 2) (h2 : some_number = 14) (h3 : 2 * x - y = some_number) : x = 8 :=
by 
  sorry

end find_x_l753_753869


namespace man_rate_in_still_water_l753_753162

theorem man_rate_in_still_water (speed_with_stream speed_against_stream : ℝ)
  (h1 : speed_with_stream = 22) (h2 : speed_against_stream = 10) :
  (speed_with_stream + speed_against_stream) / 2 = 16 := by
  sorry

end man_rate_in_still_water_l753_753162


namespace books_bought_l753_753476

-- Definitions based on given conditions
def books_price (n : ℕ) : ℝ := 
  match n with
  | 3 => 18.72
  | _ => 0.0

def total_cost : ℝ := 37.44

-- The goal is to find the number of books for a given total cost
theorem books_bought (price_per_book : ℝ) (number_of_books : ℕ) : number_of_books = 6 :=
by
  have price_per_book := 18.72 / 3
  have books_needed := total_cost / price_per_book
  have h : books_needed = 6 := by sorry
  exact h

end books_bought_l753_753476


namespace part1_part2_l753_753990

noncomputable def vector_parallel {a c : ℝ × ℝ} (h : ∃ k : ℝ, c = (k * a.1, k * a.2)) : Prop :=
  ∃ k : ℝ, c = ((k * a.1), (k * a.2))

theorem part1
  (a c : ℝ × ℝ)
  (h_a : a = (1, -2))
  (h_mag : real.sqrt (c.1 ^ 2 + c.2 ^ 2) = 2 * real.sqrt 5)
  (h_parallel : vector_parallel h_c h_a) :
  c = (2, -4) ∨ c = (-2, 4) :=
sorry

theorem part2
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h_a : a = (1, -2))
  (h_b_mag : real.sqrt(b.1 ^ 2 + b.2 ^ 2) = 1)
  (h_perp : (a.1 + b.1) * (a.1 - 2 * b.1) + (a.2 + b.2) * (a.2 - 2 * b.2) = 0) :
  real.cos (vector_angle a b) = 3 * real.sqrt 5 / 5 :=
sorry

end part1_part2_l753_753990


namespace value_of_m_l753_753544

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := -(x - m)^2 + m^2 + 1

-- Define the range for x
def x_range : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

-- The main proof statement
theorem value_of_m (m : ℝ) : (∀ x ∈ x_range, quadratic_function x m ≤ 4) → (m = 2 ∨ m = -sqrt 3) :=
sorry

end value_of_m_l753_753544


namespace distinct_solutions_eq_four_l753_753954

theorem distinct_solutions_eq_four : ∃! (x : ℝ), abs (x - abs (3 * x + 2)) = 4 :=
by sorry

end distinct_solutions_eq_four_l753_753954


namespace positive_integer_solutions_of_inequality_l753_753505

theorem positive_integer_solutions_of_inequality : 
  {x : ℕ | 3 * x - 1 ≤ 2 * x + 3} = {1, 2, 3, 4} :=
by
  sorry

end positive_integer_solutions_of_inequality_l753_753505


namespace Bran_remaining_payment_l753_753233

theorem Bran_remaining_payment :
  let tuition_fee : ℝ := 90
  let job_income_per_month : ℝ := 15
  let scholarship_percentage : ℝ := 0.30
  let months : ℕ := 3
  let scholarship_amount : ℝ := tuition_fee * scholarship_percentage
  let remaining_after_scholarship : ℝ := tuition_fee - scholarship_amount
  let total_job_income : ℝ := job_income_per_month * months
  let amount_to_pay : ℝ := remaining_after_scholarship - total_job_income
  amount_to_pay = 18 := sorry

end Bran_remaining_payment_l753_753233


namespace mark_sprint_speed_l753_753082

-- Define conditions
def Distance : ℝ := 144
def Time : ℝ := 24

-- Define Speed as Distance / Time
def Speed := Distance / Time

-- The theorem to prove that Speed is 6
theorem mark_sprint_speed : Speed = 6 := by
  sorry

end mark_sprint_speed_l753_753082


namespace walk_two_dogs_for_7_minutes_l753_753698

variable (x : ℕ)

def charge_per_dog : ℕ := 20
def charge_per_minute_per_dog : ℕ := 1
def total_earnings : ℕ := 171

def charge_one_dog := charge_per_dog + charge_per_minute_per_dog * 10
def charge_three_dogs := charge_per_dog * 3 + charge_per_minute_per_dog * 9 * 3
def charge_two_dogs (x : ℕ) := charge_per_dog * 2 + charge_per_minute_per_dog * x * 2

theorem walk_two_dogs_for_7_minutes 
  (h1 : charge_one_dog = 30)
  (h2 : charge_three_dogs = 87)
  (h3 : charge_one_dog + charge_three_dogs + charge_two_dogs x = total_earnings) : 
  x = 7 :=
by
  unfold charge_one_dog charge_three_dogs charge_per_dog charge_per_minute_per_dog total_earnings at *
  sorry

end walk_two_dogs_for_7_minutes_l753_753698


namespace perpendicular_planes_of_perpendicular_lines_l753_753780

open_locale classical

variables {α β : Type*}
variables {m n l1 l2 : α}

-- Defining the conditions
variables [plane α] [plane β]
variable m_perp_l1 : m ⊥ l1
variable m_perp_l2 : m ⊥ l2
variable m_in_alpha : m ∈ α
variable l1_in_beta : l1 ∈ β
variable l2_in_beta : l2 ∈ β
variable l1_intersect_l2 : intersect l1 l2

-- Statement of the theorem
theorem perpendicular_planes_of_perpendicular_lines (m_perp_l1 : m ⊥ l1) (m_perp_l2 : m ⊥ l2) (m_in_alpha : m ∈ α) (l1_in_beta : l1 ∈ β) (l2_in_beta : l2 ∈ β) (l1_intersect_l2 : intersect l1 l2) : α ⊥ β :=
  sorry

end perpendicular_planes_of_perpendicular_lines_l753_753780


namespace cuts_needed_to_create_100_20_sided_polygons_l753_753385

theorem cuts_needed_to_create_100_20_sided_polygons 
  (initial_vertices : ℕ := 4)
  (desired_polygons : ℕ := 100)
  (vertices_per_polygon : ℕ := 20)
  (max_vertices_per_cut : ℕ := 4)
  (min_vertices_needed : ℕ := desired_polygons * vertices_per_polygon)
  (max_vertices_achievable : ℕ := λ n, max_vertices_per_cut * n + initial_vertices) :
  ∃ (cuts_needed : ℕ), cuts_needed = 1699 :=
by
  sorry

end cuts_needed_to_create_100_20_sided_polygons_l753_753385


namespace rational_expression_equals_3_l753_753464

theorem rational_expression_equals_3 (x : ℝ) (hx : x^3 + x - 1 = 0) :
  (x^4 - 2*x^3 + x^2 - 3*x + 5) / (x^5 - x^2 - x + 2) = 3 := 
by
  sorry

end rational_expression_equals_3_l753_753464


namespace exponentiation_rule_l753_753922

theorem exponentiation_rule (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end exponentiation_rule_l753_753922


namespace speed_of_second_train_l753_753859

-- Define the given conditions as parameters
def first_train_length : ℝ := 137 / 1000 -- converting meters to kilometers
def second_train_length : ℝ := 163 / 1000 -- converting meters to kilometers
def total_length : ℝ := first_train_length + second_train_length

def time_to_clear_each_other_hours : ℝ := 12 / 3600 -- converting seconds to hours
def speed_first_train_kmph : ℝ := 42

-- The statement to prove
theorem speed_of_second_train (v2 : ℝ) :
  (speed_first_train_kmph + v2) * time_to_clear_each_other_hours = total_length → v2 = 48 :=
by
  sorry

end speed_of_second_train_l753_753859


namespace equation_of_curve_c_find_lambda_l753_753981

noncomputable def F : (ℝ × ℝ) := (0, 1)

def curve_c (M : ℝ × ℝ) : Prop := 
  (dist M F = dist M (0, -2) - 1)

theorem equation_of_curve_c :
  ∀ M : ℝ × ℝ, curve_c M → M.1^2 = 4 * M.2 :=
sorry

def intersects_curve (A B P : ℝ × ℝ) : Prop :=
  (A.1 + A.2 = 2) ∧ (B.1 + B.2 = 2) ∧ (4 * abs ((1/2) * ((A.1 * A.2 - 0) * abs (norm (A, B))))) = 4 * sqrt 2

theorem find_lambda (A B P O : ℝ × ℝ) :
  intersects_curve A B P →
  ∀ λ : ℝ, λ = (P.1 + 2 * sqrt 2 - 4 * sqrt 2)/(4 * sqrt 2 - 2) ∨ λ = (P.1 + 2 * sqrt 2 - 4 * sqrt 2)/(2 - 4 * sqrt 2) :=
sorry

end equation_of_curve_c_find_lambda_l753_753981


namespace function_zeros_range_l753_753013

open Real

def f_pos (x : ℝ) : ℝ :=
  ln x - x + 1

def f_neg (ω x : ℝ) : ℝ :=
  sin (ω * x + π / 3)

theorem function_zeros_range (ω : ℝ) (h1 : 0 < ω) :
  (∃ x > 0, f_pos x = 0) ∧ (∃ x ∈ Icc (-π : ℝ) 0, f_neg ω x = 0) →
  (ω ∈ Icc (10 / 3) (13 / 3)) :=
by
  sorry

end function_zeros_range_l753_753013


namespace fraction_of_satisfactory_grades_is_24_over_35_l753_753475

def num_students_with_grade_A : ℕ := 6
def num_students_with_grade_B : ℕ := 5
def num_students_with_grade_C : ℕ := 4
def num_students_with_grade_D : ℕ := 4
def num_students_with_grade_E : ℕ := 3
def num_students_with_grade_G : ℕ := 2
def num_students_with_grade_F : ℕ := 8
def num_students_with_grade_H : ℕ := 3

def satisfactory_grades : ℕ := 
  num_students_with_grade_A + num_students_with_grade_B + 
  num_students_with_grade_C + num_students_with_grade_D + 
  num_students_with_grade_E + num_students_with_grade_G

def total_students : ℕ := 
  satisfactory_grades + num_students_with_grade_F + num_students_with_grade_H

def fraction_satisfactory_grades : Rat := satisfactory_grades / total_students

-- The theorem we need to prove
theorem fraction_of_satisfactory_grades_is_24_over_35 : fraction_satisfactory_grades = 24 / 35 := by
  sorry

end fraction_of_satisfactory_grades_is_24_over_35_l753_753475


namespace distance_center_to_plane_l753_753724

theorem distance_center_to_plane (r : ℝ) (a b : ℝ) (h : a ^ 2 + b ^ 2 = 10 ^ 2) (d : ℝ) : 
  r = 13 → a = 6 → b = 8 → d = 12 := 
by 
  sorry

end distance_center_to_plane_l753_753724


namespace quadratic_roots_transformation_l753_753975

theorem quadratic_roots_transformation :
  ∀ (x : ℝ), (2 * x^2 - 5 * x - 8 = 0) ↔ (y : ℝ), (y^2 - 5 * y - 16 = 0) :=
by
  sorry

end quadratic_roots_transformation_l753_753975


namespace find_largest_number_with_fib_rule_l753_753279

noncomputable def fib_digit_seq : ℕ → ℕ
| 0       := 1
| 1       := 0
| (n + 2) := fib_digit_seq n + fib_digit_seq (n + 1)

def largest_fib_digit_seq (N : ℕ) : ℕ :=
(fib_digit_seq 0) * 10^(N-1) + (fib_digit_seq 1) * 10^(N-2) + 
(fib_digit_seq 2) * 10^(N-3) + (fib_digit_seq 3) * 10^(N-4) + 
(fib_digit_seq 4) * 10^(N-5) + (fib_digit_seq 5) * 10^(N-6) + 
(fib_digit_seq 6) * 10^(N-7) + (fib_digit_seq 7) * 10^(N-8)

theorem find_largest_number_with_fib_rule : 
  (∃ N, largest_fib_digit_seq N = 10112358) :=
by {
  use 8,
  sorry
}

end find_largest_number_with_fib_rule_l753_753279


namespace part1_correct_part2_correct_l753_753897

noncomputable def part1 : ℝ :=
  let total_ways := Nat.choose 16 3
  let ways_A0 := Nat.choose 12 3
  let ways_A1 := Nat.choose 4 1 * Nat.choose 12 2
  (ways_A0 + ways_A1) / total_ways

theorem part1_correct : part1 = 121 / 140 :=
sorry

noncomputable def binomial_distribution (n : ℕ) (p : ℝ) : ℕ → ℝ
| k => Nat.choose n k * (p^k) * ((1-p)^(n-k))

theorem part2_correct :
  ∀ k ∈ ({0, 1, 2, 3} : Finset ℕ), binomial_distribution 3 (1/4) k =
  match k with
  | 0 => 27 / 64
  | 1 => 27 / 64
  | 2 => 9 / 64
  | 3 => 1 / 64
  | _ => 0 :=
sorry


end part1_correct_part2_correct_l753_753897


namespace parabola_equation_and_fixed_point_l753_753306

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the parabola equation and the distance condition
def is_parabola (P : ℝ × ℝ) : Prop :=
  distance P (0, 1) = (distance P (0, -1) - 1)^2 

-- Define the fixed point existence for tangents from a point on line to parabola
def tangents_fixed_point (a : ℝ) (E : ℝ × ℝ) : Prop :=
  E = (a, -2) ∧ 
  ∀ (x0 : ℝ), let y0 := x0^2 / 4 in 
  let A := (x0 + a, y0) in 
  let B := (x0 - a, y0) in 
  ∃ (m : ℝ), (m * E.1 + 2 = A.2) ∧ m * 0 + 2 = B.2
  
theorem parabola_equation_and_fixed_point :
  (∀ (P : ℝ × ℝ), is_parabola P ↔ P.1^2 = 4 * P.2) ∧ 
  ∀ (E : ℝ × ℝ) (a : ℝ), E = (a, -2) → tangents_fixed_point a E :=
sorry

end parabola_equation_and_fixed_point_l753_753306


namespace rotated_squares_overlap_area_l753_753139

noncomputable def total_overlap_area (side_length : ℝ) : ℝ :=
  let base_area := side_length ^ 2
  3 * base_area

theorem rotated_squares_overlap_area : total_overlap_area 8 = 192 := by
  sorry

end rotated_squares_overlap_area_l753_753139


namespace total_movies_shown_l753_753196

-- Define the conditions of the problem
def screens := 6
def open_hours := 8
def movie_duration := 2

-- Define the statement to prove
theorem total_movies_shown : screens * (open_hours / movie_duration) = 24 := 
by
  sorry

end total_movies_shown_l753_753196


namespace hyperbola_range_m_l753_753363

theorem hyperbola_range_m (m : ℝ) : (m - 2) * (m - 6) < 0 ↔ 2 < m ∧ m < 6 :=
by sorry

end hyperbola_range_m_l753_753363


namespace mary_jane_takes_more_time_l753_753858

theorem mary_jane_takes_more_time
  (d : ℕ) (mary_jane_time : ℚ) (elizabeth_ann_time : ℚ) (mary_jane_speed_out: ℕ) (mary_jane_speed_back: ℕ) (elizabeth_ann_speed: ℕ) :
  d = 200 → 
  mary_jane_time = 125/3 → 
  elizabeth_ann_time = 40 → 
  mary_jane_speed_out = 12 → 
  mary_jane_speed_back = 8 → 
  elizabeth_ann_speed = 10 →
  mary_jane_time > elizabeth_ann_time :=
begin
  sorry
end

end mary_jane_takes_more_time_l753_753858


namespace quotient_three_l753_753758

theorem quotient_three (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a * b ∣ a^2 + b^2 + 1) :
  (a^2 + b^2 + 1) / (a * b) = 3 :=
sorry

end quotient_three_l753_753758


namespace triangular_region_area_l753_753921

theorem triangular_region_area : 
  ∀ (x y : ℝ),  (3 * x + 4 * y = 12) →
  (0 ≤ x ∧ 0 ≤ y) →
  ∃ (A : ℝ), A = 6 := 
by 
  sorry

end triangular_region_area_l753_753921


namespace percentage_less_than_l753_753583

variable (x y z n : ℝ)
variable (hx : x = 8 * y)
variable (hy : y = 2 * |z - n|)
variable (hz : z = 1.1 * n)

theorem percentage_less_than (hx : x = 8 * y) (hy : y = 2 * |z - n|) (hz : z = 1.1 * n) :
  ((x - y) / x) * 100 = 87.5 := sorry

end percentage_less_than_l753_753583


namespace brick_width_l753_753572

theorem brick_width (brick_count : ℕ) (brick_length brick_height wall_length wall_height wall_thickness wall_volume : ℝ)
                    (H1 : brick_count = 3200)
                    (H2 : brick_length = 50)
                    (H3 : brick_height = 6)
                    (H4 : wall_length = 800)
                    (H5 : wall_height = 600)
                    (H6 : wall_thickness = 22.5)
                    (H7 : wall_volume = wall_length * wall_height * wall_thickness) :
                    ∃ width : ℝ, width = 11.25 ∧ 3200 * (50 * width * 6) = wall_volume :=
by 
  use 11.25
  split
  sorry

end brick_width_l753_753572


namespace systematic_sampling_fourth_group_student_l753_753204

theorem systematic_sampling_fourth_group_student :
  ∀ (n : ℕ) (a b : ℕ), n = 90 → a = 14 → b = 23 → (∀ k, k = 4 → a + (k - 1) * (b - a) = 32) := 
  by
    intros n a b hn ha hb k hk
    have h_interval : b - a = 9 := by
      rw [hb, ha]
      norm_num
    rw [hn, ha, h_interval] at *
    cases hk
    norm_num
    sorry

end systematic_sampling_fourth_group_student_l753_753204


namespace percentage_decrease_is_correct_l753_753123

variable (original_price new_price decrease_price : ℝ) (P : ℝ)

-- Conditions
def original_price := 1100
def new_price := 836

-- Calculate the decrease in price
def decrease_price := original_price - new_price

-- Define the percentage decrease
def percentage_decrease := (decrease_price / original_price) * 100

-- The theorem to be proven
theorem percentage_decrease_is_correct : percentage_decrease = 24 := by
  sorry

end percentage_decrease_is_correct_l753_753123


namespace sum_of_solutions_l753_753634

-- Define the function whose roots we are interested in
def equation (x : ℝ) : ℝ :=
  2 * cos (2 * x) * (cos (2 * x) - cos (2012 * π^2 / x)) - (cos (4 * x) - 1)

-- Define the condition that x must be a positive real number
def is_positive_real (x : ℝ) : Prop := (0 < x)

-- State the theorem for the sum of all positive real solutions
theorem sum_of_solutions :
  ∃ S : ℝ, S = 1007 * π ∧
  (∀ x : ℝ, is_positive_real x → equation x = 0 → x ∈ {x | 0 < x} → ∃ x : ℝ, x ∈ {x | 0 < x} ∧ sum = S) :=
sorry

end sum_of_solutions_l753_753634


namespace minimum_value_of_quadratic_expression_l753_753630

def quadratic_expr (x y : ℝ) : ℝ := x^2 - x * y + y^2

def constraint (x y : ℝ) : Prop := x + y = 5

theorem minimum_value_of_quadratic_expression :
  ∃ m, ∀ x y, constraint x y → quadratic_expr x y ≥ m ∧ (∃ x y, constraint x y ∧ quadratic_expr x y = m) :=
sorry

end minimum_value_of_quadratic_expression_l753_753630


namespace part1_part2_l753_753674

open Real

-- Definitions based on the given conditions
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := x^2 - 2 * m * x + m^2 - 4 * m - 1

-- Part 1: Prove that if the equation has real roots, then m ≥ -1/4
theorem part1 (m : ℝ) : (∃ x : ℝ, quadratic_eq m x = 0) → m ≥ -1 / 4 :=
begin
  sorry
end

-- Part 2: Prove that if one root is 1, then m = 0 or m = 6
theorem part2 (m : ℝ) : quadratic_eq m 1 = 0 → (m = 0 ∨ m = 6) :=
begin
  sorry
end

end part1_part2_l753_753674


namespace constant_term_in_expansion_l753_753730

variable (x : ℝ)

theorem constant_term_in_expansion : 
  ∃ T : ℝ, T = -8 ∧ (∀ x : ℝ, ∑ i in finset.range 5, (nat.choose 4 i) * ((x ^ (1/3)) ^ (4 - i)) * ((-2 / x) ^ i) = T) :=
sorry

end constant_term_in_expansion_l753_753730


namespace sum_E_explicit_l753_753427

def S (n : ℕ) : Set (List (Set (Fin 1000))) :=
  {a | a.length = n ∧ ∀ Xi, Xi ∈ a → Xi ⊆ (Fin 1000).toFinset}

def E (a : List (Set (Fin 1000))) : ℕ :=
  (a.foldr (· ∪ ·) ∅).card

noncomputable def sum_E (n : ℕ) : ℕ :=
  ∑ a in (S n).toFinset, E a

theorem sum_E_explicit (n : ℕ) : 
  sum_E n = 1000 * ((2^n - 1) * 2^(999 * n)) :=
by
  sorry

end sum_E_explicit_l753_753427


namespace probability_david_meets_paul_l753_753949

theorem probability_david_meets_paul :
  ∀ (paul_arrival david_arrival : ℝ),
    (0 ≤ paul_arrival ∧ paul_arrival ≤ 60) ∧
    (0 ≤ david_arrival ∧ david_arrival ≤ 60) →
    let meet_probability := (450.0 + 900.0) / 3600.0 in
    meet_probability = 3 / 8 :=
by
  sorry

end probability_david_meets_paul_l753_753949


namespace max_sum_of_squares_ratios_l753_753140

theorem max_sum_of_squares_ratios :
  let r1 := 23 / 3
  let r2 := 23 / 2
  let r3 := 23
  let sum1 := 1 + 69 + 3
  let sum2 := 1 + 46 + 2
  let sum3 := 1 + 529 + 1
  max sum1 (max sum2 sum3) = sum3 ∧ sum3 = 1 + 529 + 1 := 
by
  have sum1_eq : sum1 = 1 + 69 + 3 := rfl
  have sum2_eq : sum2 = 1 + 46 + 2 := rfl
  have sum3_eq : sum3 = 1 + 529 + 1 := rfl
  simp only [sum1_eq, sum2_eq, sum3_eq]
  exact ⟨max_eq_right (le_max_left _ _), rfl⟩

end max_sum_of_squares_ratios_l753_753140


namespace real_roots_eq_4_l753_753982

noncomputable def f (x : ℝ) : ℝ := if x ∈ (-1 : ℝ) .. 1 then |x|
                                 else f (x - 2)

theorem real_roots_eq_4 :
  ∃ n, n = 4 ∧ ∀ x, (log 3 (|x|) - f x = 0) → n = 4 :=
begin
  sorry
end

end real_roots_eq_4_l753_753982


namespace problem_a2_minus_b2_problem_a3_minus_b3_l753_753008

variable (a b : ℝ)
variable (h1 : a + b = 8)
variable (h2 : a - b = 4)

theorem problem_a2_minus_b2 :
  a^2 - b^2 = 32 := 
by
sorry

theorem problem_a3_minus_b3 :
  a^3 - b^3 = 208 := 
by
sorry

end problem_a2_minus_b2_problem_a3_minus_b3_l753_753008


namespace smallest_period_of_f_maximum_value_of_f_alpha_value_l753_753338

noncomputable def f (x : ℝ) : ℝ :=
  sin x * cos x + sin x^2 - 1/2

theorem smallest_period_of_f :
  ∃ p > 0, ∀ x, f (x + p) = f x ∧ p = π := by
  sorry

theorem maximum_value_of_f :
  ∃ x, f x = sqrt 2 / 2 := by
  sorry

theorem alpha_value :
  ∀ α : ℝ, α > 0 ∧ α < π / 2 ∧ f α = sqrt 2 / 2 → α = 3 * π / 8 := by
  sorry

end smallest_period_of_f_maximum_value_of_f_alpha_value_l753_753338


namespace probability_palindromic_phone_number_l753_753398

/-- In a city where phone numbers consist of 7 digits, the Scientist easily remembers a phone number
    if it is a palindrome (reads the same forwards and backwards). Prove that the probability that 
    a randomly chosen 7-digit phone number is a palindrome is 0.001. -/
theorem probability_palindromic_phone_number : 
  let total_phone_numbers := 10^7
  let palindromic_phone_numbers := 10^4
  (palindromic_phone_numbers : ℝ) / total_phone_numbers = 0.001 :=
by
  let total_phone_numbers := 10^7
  let palindromic_phone_numbers := 10^4
  show (palindromic_phone_numbers : ℝ) / total_phone_numbers = 0.001 from sorry

end probability_palindromic_phone_number_l753_753398


namespace find_oranges_to_put_back_l753_753223

theorem find_oranges_to_put_back (A O x : ℕ) (h₁ : A + O = 15) (h₂ : 40 * A + 60 * O = 720) (h₃ : (360 + 360 - 60 * x) / (15 - x) = 45) : x = 3 := by
  sorry

end find_oranges_to_put_back_l753_753223


namespace coin_draws_expected_value_l753_753864

theorem coin_draws_expected_value :
  ∃ f : ℕ → ℝ, (∀ (n : ℕ), n ≥ 4 → f n = (3 : ℝ)) := sorry

end coin_draws_expected_value_l753_753864


namespace expected_rolls_in_non_leap_year_l753_753216

-- Define the conditions and the expected value
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def stops_rolling (n : ℕ) : Prop := is_prime n ∨ is_multiple_of_4 n

def expected_rolls_one_day : ℚ := 6 / 7

def non_leap_year_days : ℕ := 365

def expected_rolls_one_year := expected_rolls_one_day * non_leap_year_days

theorem expected_rolls_in_non_leap_year : expected_rolls_one_year = 314 :=
by
  -- Verification of the mathematical model
  sorry

end expected_rolls_in_non_leap_year_l753_753216


namespace frac_sum_property_l753_753356

theorem frac_sum_property (a b : ℕ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end frac_sum_property_l753_753356


namespace alice_bob_game_meet_at_same_point_l753_753214

theorem alice_bob_game_meet_at_same_point :
  ∃ k : ℕ, k > 0 ∧ (7 * k ≡ 4 * k [MOD 15]) :=
by sorry

end alice_bob_game_meet_at_same_point_l753_753214


namespace ratio_of_areas_l753_753899

theorem ratio_of_areas (w : ℝ) (h_rect : h = 2 * w) : 
  let r := w / 2,
      s := w / Real.sqrt 2,
      r_final := s / 2 in
  let A_circle := π * (r_final ^ 2),
      A_rectangle := w * h in
  A_circle / A_rectangle = π / 16 := by
sorry

end ratio_of_areas_l753_753899


namespace inverse_proportion_l753_753106

theorem inverse_proportion (α β k : ℝ) (h1 : α * β = k) (h2 : α = 5) (h3 : β = 10) : (α = 25 / 2) → (β = 4) := by sorry

end inverse_proportion_l753_753106


namespace Jim_remaining_distance_l753_753058

theorem Jim_remaining_distance (t d r : ℕ) (h₁ : t = 1200) (h₂ : d = 923) (h₃ : r = t - d) : r = 277 := 
by 
  -- Proof steps would go here
  sorry

end Jim_remaining_distance_l753_753058


namespace right_triangle_integer_segments_count_l753_753799

theorem right_triangle_integer_segments_count :
  ∀ (DE EF : ℕ), DE = 15 → EF = 36 → 
  let DF := Real.sqrt (DE^2 + EF^2) in
  let area := (DE * EF) / 2 in
  ∃ (integer_segment_count : ℕ),
  (integer_segment_count = 24) := 
by
  intros DE EF hDE hEF
  let DF := Real.sqrt (DE^2 + EF^2)
  let area := (DE * EF) / 2
  use 24
  sorry

end right_triangle_integer_segments_count_l753_753799


namespace area_ratio_theorem_l753_753766

variables {A B C D P : Type}
variables (AB AC AD AE AP DP BC : vector_space)

-- Hypotheses as given
hypothesis h1 : ∀ (T : Type) [inner_product_space ℝ T] (A B C D P : T),
  AD = (1/5 : ℝ) •( AB + AC )
hypothesis h2 : ∀ (T : Type) [inner_product_space ℝ T] (A B C D P : T),
  AP = AD + (1/10 : ℝ) • BC

-- Statement to prove the area ratio
theorem area_ratio_theorem 
    (T : Type) [inner_product_space ℝ T] (A B C D P : T) :
    ∃ (AD_AE : ℝ), AD_AE = (2 / 5 : ℝ) →
    ∃ (DP_BC : ℝ), DP_BC = (1 / 10 : ℝ) →
    (S (triangle APD) / S (triangle ABC)) = (1 / 25 : ℝ) :=
sorry

end area_ratio_theorem_l753_753766


namespace largest_non_prime_is_28_l753_753970

noncomputable def prime_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def gap_between_consecutive_primes : List ℕ :=
  prime_numbers.tail.map2 (fun a b => b - a) prime_numbers

def find_non_prime_gap (n m : ℕ) (l : List ℕ) : List ℕ :=
  let sublist := (List.range (m - n + 1)).map (fun x => x + n)
  if sublist.all (fun x => ¬(l.mem x)) then sublist else []

def largest_of_five_non_prime_less_than_30 : ℕ :=
  match find_non_prime_gap 24 28 prime_numbers with
  | [24, 25, 26, 27, 28] => 28
  | _ => 0 -- This should not occur if the theory is correct

theorem largest_non_prime_is_28 :
  largest_of_five_non_prime_less_than_30 = 28 :=
by {
  unfold largest_of_five_non_prime_less_than_30,
  simp,
  /- Skip proof details -/
  sorry
}

end largest_non_prime_is_28_l753_753970


namespace ellipse_standard_equation_length_major_axis_coordinates_of_foci_equation_of_directrix_l753_753987

-- Given conditions
variables (a b c e : ℝ)
variables (h1 : a > b) (h2 : b > 0)
variables (h3 : 2 * a = 4 * real.sqrt 2)
variables (h4 : e = real.sqrt 3 / 2)
variables (h5 : c = a * e)

-- Calculating auxiliary values
def ellipse_equation :=
  b^2 = a^2 - c^2

-- Statement of the problem
theorem ellipse_standard_equation :
  ∃ (a b : ℝ), (2 * a = 4 * real.sqrt 2) ∧ (b^2 = a^2 - (a * (real.sqrt 3 / 2))^2) ∧
    (frac {x^2} {a^2} + frac {y^2} {b^2} = 1)

theorem length_major_axis :
  2 * a = 4 * real.sqrt 2

theorem coordinates_of_foci : 
  ((-real.sqrt 6, 0) ∈ set_of (λ (p : ℝ × ℝ), 
    (frac {fst p^2} {a^2} + frac {snd p^2} {b^2} = 1)) ∧ 
  ((real.sqrt 6, 0) ∈ set_of (λ (p : ℝ × ℝ), 
    (frac {fst p^2} {a^2} + frac {snd p^2} {b^2} = 1)))

theorem equation_of_directrix :
  ∃ (a c : ℝ), (2 * a = 4 * real.sqrt 2) ∧ (c = a * real.sqrt 3 / 2) → 
    (a^2 / c = abs ((4 * real.sqrt 2)^2 / real.sqrt 6))

end ellipse_standard_equation_length_major_axis_coordinates_of_foci_equation_of_directrix_l753_753987


namespace circles_intersect_find_line_eqn_l753_753887

-- Definition for problem 1
def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Theorem for problem 1
theorem circles_intersect : ∀ (x y : ℝ), 
  (circle1_eqn x y → circle2_eqn x y → false) :=
by sorry

-- Definitions for problem 2
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 + 4*y - 21 = 0
def point_M : ℝ × ℝ := (-3, -3)
def chord_length : ℝ := 4 * Real.sqrt 5

-- Theorem for problem 2
theorem find_line_eqn (l : ℝ → ℝ → Prop) : 
  (∀ x y : ℝ, l x y <-> (2 * x - y + 3 = 0 ∨ x + 2 * y + 9 = 0)) := 
by sorry

end circles_intersect_find_line_eqn_l753_753887


namespace regular_polygon_not_necess_isosceles_triangle_l753_753549

theorem regular_polygon_not_necess_isosceles_triangle :
  ¬ (∀ (P : Type) [polygon P] [regular_polygon P], isosceles_triangle P) := 
sorry

end regular_polygon_not_necess_isosceles_triangle_l753_753549


namespace total_voters_in_districts_l753_753844

theorem total_voters_in_districts :
  let D1 := 322
  let D2 := (D1 / 2) - 19
  let D3 := 2 * D1
  let D4 := D2 + 45
  let D5 := (3 * D3) - 150
  let D6 := (D1 + D4) + (1 / 5) * (D1 + D4)
  let D7 := D2 + (D5 - D2) / 2
  D1 + D2 + D3 + D4 + D5 + D6 + D7 = 4650 := 
by
  sorry

end total_voters_in_districts_l753_753844


namespace mean_sub_median_of_sequence_is_two_l753_753557

theorem mean_sub_median_of_sequence_is_two (x: ℕ) (h: 0 < x) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 17)) / 5 in
  let median := x + 4 in
  mean - median = 2 := 
by 
  sorry

end mean_sub_median_of_sequence_is_two_l753_753557


namespace algorithm_d_has_no_conditionals_l753_753925

def algorithmA : Type := ∀ (a b c : ℝ), real
def algorithmB : Type := ∀ (x₁ y₁ x₂ y₂ : ℝ), real
def algorithmC : Type := ∀ (x : ℝ), real
def algorithmD : Type := ∀ (base_area height : ℝ), real

theorem algorithm_d_has_no_conditionals : 
  (∀ cond_stmt : algorithmA × algorithmB × algorithmC, false) 
  → (∀ cond_stmt : algorithmD, true) := 
sorry

end algorithm_d_has_no_conditionals_l753_753925


namespace alice_travel_time_l753_753482

theorem alice_travel_time (distance_AB : ℝ) (bob_speed : ℝ) (alice_speed : ℝ) (max_time_diff_hr : ℝ) (time_conversion : ℝ) :
  distance_AB = 60 →
  bob_speed = 40 →
  alice_speed = 60 →
  max_time_diff_hr = 0.5 →
  time_conversion = 60 →
  max_time_diff_hr * time_conversion = 30 :=
by
  intros
  sorry

end alice_travel_time_l753_753482


namespace Jame_gold_bars_left_l753_753051

theorem Jame_gold_bars_left (initial_bars : ℕ) (tax_rate : ℚ) (loss_rate : ℚ) :
  initial_bars = 60 → tax_rate = 0.1 → loss_rate = 0.5 →
  let bars_after_tax := initial_bars - (initial_bars * tax_rate).toNat in
  let bars_after_divorce := (bars_after_tax * (1 - loss_rate).toRat).toNat in
  bars_after_divorce = 27 :=
by
  intros h_initial h_tax h_loss
  let bars_after_tax := initial_bars - (initial_bars * tax_rate).toNat
  let bars_after_divorce := (bars_after_tax * (1 - loss_rate).toRat).toNat
  have h1 : initial_bars = 60, from h_initial
  have h2 : tax_rate = 0.1, from h_tax
  have h3 : loss_rate = 0.5, from h_loss
  sorry

end Jame_gold_bars_left_l753_753051


namespace find_smaller_number_l753_753881

theorem find_smaller_number (a b : ℕ) (h_ratio : 11 * a = 7 * b) (h_diff : b = a + 16) : a = 28 :=
by
  sorry

end find_smaller_number_l753_753881


namespace part1_part2_l753_753996

theorem part1 (x : ℝ) (m : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0) ↔ m ≤ 6 :=
sorry

theorem part2 (a b : ℝ) (n : ℝ) :
  n = 6 → (a > 0 ∧ b > 0 ∧ (4 / (a + 5 * b)) + (1 / (3 * a + 2 * b)) = 1) → (4 * a + 7 * b) ≥ 9 :=
sorry

end part1_part2_l753_753996


namespace swimming_distance_l753_753193

theorem swimming_distance
  (t : ℝ) (d_up : ℝ) (d_down : ℝ) (v_man : ℝ) (v_stream : ℝ)
  (h1 : v_man = 5) (h2 : t = 5) (h3 : d_up = 20) 
  (h4 : d_up = (v_man - v_stream) * t) :
  d_down = (v_man + v_stream) * t :=
by
  sorry

end swimming_distance_l753_753193


namespace min_distance_complex_l753_753078

open Complex

theorem min_distance_complex (z w : ℂ)
  (hz : abs (z - (2 + 2*I)) = 2)
  (hw : abs (w - (5 + 6*I)) = 4) :
  is_minimum (abs (z - w)) 11 :=
begin
  sorry
end

end min_distance_complex_l753_753078


namespace total_cost_of_motorcycle_l753_753053

-- Definitions from conditions
def total_cost (x : ℝ) := 0.20 * x = 400

-- The theorem to prove
theorem total_cost_of_motorcycle (x : ℝ) (h : total_cost x) : x = 2000 := 
by
  sorry

end total_cost_of_motorcycle_l753_753053


namespace z_conj_in_second_quadrant_l753_753331

/-- Definitions for imaginary unit and complex number operations -/
def i : ℂ := complex.I
def z : ℂ := 5 / (2 * i - 1)
def z_conj : ℂ := conj z

/-- Definition for location of a complex number in quadrants -/
def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- The target theorem that we need to prove -/
theorem z_conj_in_second_quadrant : is_second_quadrant z_conj :=
by
  sorry

end z_conj_in_second_quadrant_l753_753331


namespace prob_dice_product_is_72_l753_753157

noncomputable def dice_probability := 
  let S := {1, 2, 3, 4, 5, 6}
  let events := 
    { (a, b, c) | a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a * b * c = 72 } 
  (events.to_finset.card / (S.to_finset.card ^ 3) : ℚ)

theorem prob_dice_product_is_72 : dice_probability = 1 / 36 := by
  sorry

end prob_dice_product_is_72_l753_753157


namespace find_numbers_l753_753819

theorem find_numbers (a b : ℝ) (h1 : a - b = 7.02) (h2 : a = 10 * b) : a = 7.8 ∧ b = 0.78 :=
by
  sorry

end find_numbers_l753_753819


namespace count_special_multiples_l753_753350

theorem count_special_multiples : 
  let N := 150
  let count_multiples (a b : ℕ) := (N / a + N / b - N / (Nat.lcm a b)) 
  let count_valid_multiples : ℕ :=
    let multiples_10 := N / 10
    let multiples_35 := N / (Nat.lcm 5 7)
    count_multiples 5 7 - multiples_10 + multiples_35
  in count_valid_multiples = 34 :=
by sorry

end count_special_multiples_l753_753350


namespace amc_inequality_l753_753431

theorem amc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := 
by 
  sorry

end amc_inequality_l753_753431


namespace ratio_EG_GF_l753_753041

/-- In triangle ABC, M is the midpoint of BC.
    Given that AB = 15, AC = 20, E is on segment AC, and F is on segment AB.
    G is the intersection of segments EF and AM, and AE = 3AF.
    Then, EG/GF = 7/3.
-/
theorem ratio_EG_GF (A B C M E F G : Type)
  [linear_ordered_ring A]
  (hM : midpoint M B C)
  (hAB : dist A B = 15)
  (hAC : dist A C = 20)
  (hE_on_AC : on_line_segment A C E)
  (hF_on_AB : on_line_segment A B F)
  (hG_intersection : is_intersection_point G E F A M)
  (hAE_eq_3AF : dist A E = 3 * dist A F)
  : dist E G / dist G F = (7 / 3) := sorry

end ratio_EG_GF_l753_753041


namespace inequality_solution_l753_753914

theorem inequality_solution (x : ℝ) : x > 0 ∧ (x^(1/3) < 3 - x) ↔ x < 3 :=
by 
  sorry

end inequality_solution_l753_753914


namespace prove_smallest_number_divisible_by_1_to_10_and_sqrt_prime_gt_10_l753_753152

noncomputable def smallest_number_divisible_by_1_to_10_and_sqrt_prime_gt_10 : ℕ :=
  let lcm_1_to_10 : ℕ := Nat.lcm (Finset.range 11)
  let smallest_prime_gt_10 : ℕ := 11
  let sqrt_prime_gt_10_int : ℕ := Nat.sqrt smallest_prime_gt_10
  if sqrt_prime_gt_10_int ∣ lcm_1_to_10 then lcm_1_to_10 else 0

theorem prove_smallest_number_divisible_by_1_to_10_and_sqrt_prime_gt_10 :
  smallest_number_divisible_by_1_to_10_and_sqrt_prime_gt_10 = 2520 := by
  sorry

end prove_smallest_number_divisible_by_1_to_10_and_sqrt_prime_gt_10_l753_753152


namespace thirtieth_digit_fraction_sum_l753_753534

theorem thirtieth_digit_fraction_sum :
  let a := 1 / 11
  let b := 1 / 13
  let s := a + b
  (decimal_fractions.thirtieth_digit s) = 2 :=
sorry

end thirtieth_digit_fraction_sum_l753_753534


namespace perpendicular_lines_iff_l753_753564

theorem perpendicular_lines_iff (a : ℝ) : 
  (∀ b₁ b₂ : ℝ, b₁ ≠ b₂ → ¬ (∀ x : ℝ, a * x + b₁ = (a - 2) * x + b₂) ∧ 
   (a * (a - 2) = -1)) ↔ a = 1 :=
by
  sorry

end perpendicular_lines_iff_l753_753564


namespace sum_of_first_10_terms_theorem_l753_753330

noncomputable def sum_of_first_10_terms (s q : ℝ) :=
  (1 + q^10) * s = 21 ∧ (1 + q^10 + q^20) * s = 49 → s = 7 ∨ s = 63

theorem sum_of_first_10_terms_theorem :
  ∀ (s q : ℝ), (1 + q^10) * s = 21 ∧ (1 + q^10 + q^20) * s = 49 → s = 7 ∨ s = 63 :=
by
  intros s q h
  exact sum_of_first_10_terms s q h

end sum_of_first_10_terms_theorem_l753_753330


namespace problem_l753_753075

theorem problem (m : ℕ) (h : m = 16^2023) : m / 8 = 2^8089 :=
by {
  sorry
}

end problem_l753_753075


namespace binomial_expansion_sum_l753_753888

theorem binomial_expansion_sum (n : ℕ) :
  1 + ∑ k in Finset.range (n + 1), (3 ^ k) * Nat.choose n k = 4 ^ n :=
by
  sorry

end binomial_expansion_sum_l753_753888


namespace students_received_B_l753_753380

theorem students_received_B (x : ℕ) 
  (h1 : (0.8 * x : ℝ) + x + (1.2 * x : ℝ) = 28) : 
  x = 9 := 
by
  sorry

end students_received_B_l753_753380


namespace quadratic_to_standard_form_l753_753701

def quadratic_form (a b c x : ℝ) : ℝ := (a * x + b) ^ 2 + c

theorem quadratic_to_standard_form (a b c : ℤ) :
  (quadratic_form a b c : ℝ) = 16 * (λ x, x) ^ 2 - 40 * (λ x, x) + 18 → 
  a * b = -20 :=
by
  sorry

end quadratic_to_standard_form_l753_753701


namespace determine_angle_l753_753378

-- Definitions
def ABC := Type 
def A : ABC := sorry   -- Vertex A
def B : ABC := sorry   -- Vertex B
def C : ABC := sorry   -- Vertex C
def D : ABC := sorry   -- Point on the extended line BC

/- Extend the line BC and include point D on this extension -/
noncomputable def BC : ABC → ABC → Type := sorry

-- Angle at vertices
def angle_ABC := 70 : ℝ   
def angle_ACB := 50 : ℝ   
def angle_BCD := 45 : ℝ   
def x : ℝ := 35

-- Problem Statement
theorem determine_angle (hABC : ∀ {A B C : ABC}, angle_ABC = 70)
  (hACB : ∀ {A C B : ABC}, angle_ACB = 50)
  (hBCD : ∀ {B C D : ABC}, angle_BCD = 45) :
  x = 35 :=
by sorry -- proof omitted

end determine_angle_l753_753378


namespace value_of_f_5_l753_753006

def f (x : ℝ) : ℝ := (3 * x + 2) / (x - 2)

theorem value_of_f_5 : f 5 = 17 / 3 :=
by
  sorry

end value_of_f_5_l753_753006


namespace part_I_part_II_l753_753733

-- Define curve C1
def curve_C1 (t α : ℝ) := (x, y) where
  x = t * cos α
  y = t * sin α + 1

-- Define curve C2
def curve_C2 (s : ℝ) := (x, y) where
  x = - (Real.sqrt 2 / 2) * s + 1
  y = (Real.sqrt 2 / 2) * s - 1

-- Define curve C3 in polar coordinates and convert to rectangular coordinates
def curve_C3 (ρ θ : ℝ) := (ρ * cos θ, ρ * sin θ, ρ * cos θ - ρ * sin θ = 2)

-- Define point P
def point_P := (1, -1)

-- Define distance formula
def distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)

-- Theorem statements
theorem part_I : 
  ∃ (x y : ℝ), (curve_C2 s = (x, y)) ∧ (curve_C3 ρ θ = (x, y)) → (x, y) = point_P :=
  sorry

theorem part_II :
  ∀ (t : ℝ), t = 3 * Real.sqrt 2 / 2 → 
  ∃ (x1 y1 x2 y2 : ℝ), curve_C1 t α = (x1, y1) ∧ curve_C1 t α = (x2, y2) →
  (distance 1 -1 x1 y1) ^ 2 + (distance 1 -1 x2 y2) ^ 2 = 27 :=
  sorry

end part_I_part_II_l753_753733


namespace cube_plane_split_ratio_2_1_cube_plane_split_ratio_3_1_l753_753406

-- Define the conditions of the problem
variable {V : ℝ} -- volume of the cube

-- Define points on a cube
structure Point (α : Type*) :=
(x y z : α)

-- Assuming cube vertices (A, B, C, D, A', B', C', D'), we define the primary diagonal and division ratios
def diagonal_ratio1 : ℝ × ℝ := (2, 1)
def diagonal_ratio2 : ℝ × ℝ := (3, 1)

-- Main theorem to prove the volume ratio split by the plane based on the given ratio
theorem cube_plane_split_ratio_2_1 (V : ℝ) :
  let ratio := 1 / (2 + 1) in
  ratio = 1 / 6 ∧ (5 / 6) * V = (5 / 6) * V :=
sorry

theorem cube_plane_split_ratio_3_1 (V : ℝ) :
  let ratio := 9 / (9 + 119) in
  (ratio = 9 / 128) ∧ (119 / 128) * V = (119 / 128) * V :=
sorry

end cube_plane_split_ratio_2_1_cube_plane_split_ratio_3_1_l753_753406


namespace vector_parallel_solution_l753_753345

-- Define the vectors and the condition
def a (m : ℝ) := (2 * m + 1, 3)
def b (m : ℝ) := (2, m)

-- The proof problem statement
theorem vector_parallel_solution (m : ℝ) :
  (2 * m + 1) * m = 3 * 2 ↔ m = 3 / 2 ∨ m = -2 :=
by
  sorry

end vector_parallel_solution_l753_753345


namespace time_ratio_proof_l753_753578

noncomputable theory

open Real

def boat_stream_ratio (B S : ℝ) : Prop := B = 3 * S

def time_against_stream (D B S : ℝ) : ℝ := D / (B - S)

def time_favor_stream (D B S : ℝ) : ℝ := D / (B + S)

def time_ratio (T_against T_favor : ℝ) : ℝ := T_against / T_favor

theorem time_ratio_proof (D B S : ℝ) (h : boat_stream_ratio B S) :
  time_ratio (time_against_stream D B S) (time_favor_stream D B S) = 2 :=
by
  sorry

end time_ratio_proof_l753_753578


namespace solution_set_of_log_inequality_l753_753757

noncomputable def log_a (a x : ℝ) : ℝ := sorry -- The precise definition of the log base 'a' is skipped for brevity.

theorem solution_set_of_log_inequality (a x : ℝ)
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (h_max : ∃ y, log_a a (y^2 - 2*y + 3) = y):
  log_a a (x - 1) > 0 ↔ (1 < x ∧ x < 2) :=
sorry

end solution_set_of_log_inequality_l753_753757


namespace sugar_needed_l753_753781

theorem sugar_needed (total_sugar_needed : ℕ) (sugar_already_added : ℕ) 
(h1 : total_sugar_needed = 13) (h2 : sugar_already_added = 2) : 
(total_sugar_needed - sugar_already_added = 11) := 
by { rw [h1, h2], norm_num }

end sugar_needed_l753_753781


namespace max_log2_x_2log2_y_l753_753665

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem max_log2_x_2log2_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y^2 = 2) :
  log2 x + 2 * log2 y ≤ 0 :=
sorry

end max_log2_x_2log2_y_l753_753665


namespace social_network_friendship_l753_753029

-- Define the initial conditions
variables {V : Type} [fintype V] [decidable_eq V] (G : simple_graph V)
variable [h_card : fintype.card V = 2019]
variables (initial_degree : V → ℕ ) 

-- Define symmetry of friendships
def symmetric_friendship := symmetric (λ v u, G.adj v u)

-- Define initial degree conditions
def initial_degrees : Prop := 
  card { v : V | initial_degree v = 1009 } = 1010 ∧ 
  card { v : V | initial_degree v = 1010 } = 1009

-- Define the operation
def friend_switch_operation (A B C : V) : Prop := 
  G.adj A B ∧ G.adj A C ∧ ¬ G.adj B C →
  (∀ v, v ≠ A → G.adj v B ↔ v = C) ∧ 
  (∀ v, v ≠ A → G.adj v C ↔ v = B)

-- Define the theorem
theorem social_network_friendship :
  symmetric_friendship G ∧ initial_degrees G initial_degree →
  ∃ (seq : list (V × V × V)), 
    (∀ (A B C : V), (A, B, C) ∈ seq → friend_switch_operation G A B C) ∧
    ∀ v, G.degree v ≤ 1 :=
sorry

end social_network_friendship_l753_753029


namespace coord_H_min_x_coord_H_area_H_l753_753414

noncomputable def H_coord (t : ℝ) : ℝ × ℝ :=
  (Real.cos (t / 2) * Real.cos t, Real.cos (t / 2) * Real.sin t)

theorem coord_H (t : ℝ) (ht : 0 ≤ t ∧ t ≤ π) :
  H_coord t = (Real.cos (t / 2) * Real.cos t, Real.cos (t / 2) * Real.sin t) :=
sorry

theorem min_x_coord_H (t : ℝ) (ht : 0 ≤ t ∧ t ≤ π) :
  ∃ t₀, 0 ≤ t₀ ∧ t₀ ≤ π ∧ (H_coord t₀).fst = -1 / (3 * Real.sqrt 6) :=
sorry

def area_S (t : ℝ) : ℝ :=
  1 / 4 * (t + Real.sin t)

theorem area_H (t : ℝ) (ht : 0 ≤ t ∧ t ≤ π / 2) :
  ∃ S, S = (π + 2) / 8 :=
sorry

end coord_H_min_x_coord_H_area_H_l753_753414


namespace shortest_path_is_20_miles_l753_753574

-- Define the initial position of the cowboy
def C : ℝ × ℝ := (0, -5)

-- Define the position of the cabin
def B : ℝ × ℝ := (-5, 5)

-- Define the shortest path function
def shortest_path (c : ℝ × ℝ) (b : ℝ × ℝ) : ℝ :=
  let creek_crossing := (c.1, 0) in -- cowboy crosses creek at (c.1, 0)
  let c' := (c.1, -c.2) in -- reflect C over the creek
  let distance1 := real.sqrt ((creek_crossing.1 - c.1)^2 + (creek_crossing.2 - c.2)^2) in
  let distance2 := real.sqrt ((b.1 - c'.1)^2 + (b.2 - c'.2)^2) in
  (distance1 + distance2 + distance1 + distance2)

-- Prove the shortest path from C to B under given conditions is 20 miles
theorem shortest_path_is_20_miles : shortest_path C B = 20 :=
by sorry

end shortest_path_is_20_miles_l753_753574


namespace total_rounds_played_l753_753552

/-- William and Harry played some rounds of tic-tac-toe.
    William won 5 more rounds than Harry.
    William won 10 rounds.
    Prove that the total number of rounds they played is 15. -/
theorem total_rounds_played (williams_wins : ℕ) (harrys_wins : ℕ)
  (h1 : williams_wins = 10)
  (h2 : williams_wins = harrys_wins + 5) :
  williams_wins + harrys_wins = 15 := 
by
  sorry

end total_rounds_played_l753_753552


namespace sam_initial_puppies_l753_753800

theorem sam_initial_puppies (gave_away : ℝ) (now_has : ℝ) (initially : ℝ) 
    (h1 : gave_away = 2.0) (h2 : now_has = 4.0) : initially = 6.0 :=
by
  sorry

end sam_initial_puppies_l753_753800


namespace units_digit_p_plus_2_l753_753997

theorem units_digit_p_plus_2 {p : ℕ} 
  (h1 : p % 2 = 0) 
  (h2 : p % 10 ≠ 0) 
  (h3 : (p^3 % 10) = (p^2 % 10)) : 
  (p + 2) % 10 = 8 :=
sorry

end units_digit_p_plus_2_l753_753997


namespace number_of_correct_statements_l753_753035

theorem number_of_correct_statements {x y z m n : ℝ} (h : x > y ∧ y > z ∧ z > m ∧ m > n) :
  (1) * (∃ a : ℕ, |x-y| = x - y ∧ |z-m| = z - m) +
  (2) * (∀ b : ℕ, @context  _ (not (|x-y| - |z-m| - |m-n| = 0))) +
  (3) * (#| {(|x-y|-z-m-n = x - y - z - m - n), (x- |y-z| - m - n = x - y + z - m - n), (x-y-|z-m| - n = x - y - z + m - n), (x-y-z - |m-n| = x - y - z - m + n), (|x-y| - |z-m| - n = x - y - z + m - n), (|x-y|-z-|m-n| = x - y - z - m + n), (x - |y-z| - |m-n| = x - y + z - m + n)}| = 5) = 2 := sorry

end number_of_correct_statements_l753_753035


namespace towels_per_wash_l753_753461

-- Definitions based on the problem
def total_towels : ℕ := 98
def days_needed : ℕ := 7
def washes_per_day : ℕ := 2
def total_washes : ℕ := days_needed * washes_per_day

-- Problem statement: proving the number of towels washed in one wash
theorem towels_per_wash : total_towels / total_washes = 7 := by
  have h1: total_washes = 14 := by
    simp [total_washes, days_needed, washes_per_day]
  rw h1
  have h2: total_towels / 14 = 7 := by
    norm_num
  exact h2

#eval towels_per_wash

end towels_per_wash_l753_753461


namespace surface_area_of_brick_l753_753556

namespace SurfaceAreaProof

def brick_length : ℝ := 8
def brick_width : ℝ := 6
def brick_height : ℝ := 2

theorem surface_area_of_brick :
  2 * (brick_length * brick_width + brick_length * brick_height + brick_width * brick_height) = 152 :=
by
  sorry

end SurfaceAreaProof

end surface_area_of_brick_l753_753556


namespace truck_travel_and_cost_l753_753212

theorem truck_travel_and_cost
  (distance_per_5_gallons : ℕ)
  (cost_per_gallon : ℕ) :
  (distance_per_5_gallons = 240) →
  (cost_per_gallon = 3) →
  ((distance_per_5_gallons / 5 * 7) = 336) ∧ ((cost_per_gallon * 7) = 21) :=
by
  intro h_distance h_cost
  split
  · sorry
  · sorry

end truck_travel_and_cost_l753_753212


namespace probability_last_passenger_own_seat_is_half_l753_753605

open Classical

-- Define the behavior and probability question:

noncomputable def probability_last_passenger_own_seat (n : ℕ) : ℚ :=
  if n = 0 then 0 else 1 / 2

-- The main theorem stating the probability for an arbitrary number of passengers n
-- The theorem that needs to be proved:
theorem probability_last_passenger_own_seat_is_half (n : ℕ) (h : n > 0) : 
  probability_last_passenger_own_seat n = 1 / 2 :=
by sorry

end probability_last_passenger_own_seat_is_half_l753_753605


namespace prove_inequality_l753_753068

theorem prove_inequality (a b c d e f : ℕ) (hab : 0 < b) (hcd : 0 < d) (hef : 0 < f)
  (hineq1 : (a : ℚ) / b < c / d)
  (hineq2 : (c : ℚ) / d < e / f)
  (heq : a * f - b * e = -1) :
  d ≥ b + f := 
sorry

end prove_inequality_l753_753068


namespace solve_equation_l753_753806

theorem solve_equation (x y : ℝ) (k l : ℤ) :
  2^(- (sin x)^2) + 2^(- (cos x)^2) = sin y + cos y ∧ (sin x)^2 + (cos x)^2 = 1 →
  (∃ k : ℤ, x = π/4 + k * (π/2)) ∧ (∃ l : ℤ, y = π/4 + l * (2 * π)) :=
begin
  intros h,
  sorry
end

end solve_equation_l753_753806


namespace judgments_correct_l753_753325

variable {R : Type*} 
variables (f : R → R) (y : R)
variables (a b x : R)

-- Conditions
axiom even_function : ∀ x, f(x) = f(-x)
axiom mono_increasing_on_neg1_0 : ∀ x1 x2 : R, x1 ∈ Icc (-1:ℝ) 0 → x2 ∈ Icc (-1:ℝ) 0 → x1 < x2 → f(x1) ≤ f(x2)
axiom functional_equation : ∀ x, f(1 - x) + f(1 + x) = 0

-- Proof goals
theorem judgments_correct :
  f(-3) = 0 ∧ 
  (∀ x ∈ Icc (1:ℝ) 2, monotone_increasing_on_segment f x)
  ∧ (∀ x, f(x) = f(2 - x))
  ∧ (f(2) = min_value f)
  ∧ (∀ y, y < sup (f '' (Icc (-1:ℝ) 1))) :=
sorry

end judgments_correct_l753_753325


namespace max_distinct_angles_l753_753503

theorem max_distinct_angles (n : ℕ) (h : n > 1) :
  ∀ lines : fin (2*n) → (ℝ × ℝ) × ℝ, -- considering lines are represented in the form ax+by=c
    (∀ i ≠ j, lines i ≠ lines j) →
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i →
             let (a₁, b₁, c₁) := lines i in
             let (a₂, b₂, c₂) := lines j in
             let (a₃, b₃, c₃) := lines k in
             (a₁ * b₂ - a₂ * b₁) * (a₂ * b₃ - a₃ * b₂) ≠ 0) →  -- no three lines are concurrent
    ∃ angles : set (ℝ × ℝ),
      ∀ a ∈ angles, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a = (x₁, x₂) ∧
      (angles.finite ∧ angles.card ≤ 2*n - 1) :=
  sorry

end max_distinct_angles_l753_753503


namespace johns_remaining_money_l753_753061

theorem johns_remaining_money (H1 : ∃ (n : ℕ), n = 5376) (H2 : 5376 = 5 * 8^3 + 3 * 8^2 + 7 * 8^1 + 6) :
  (2814 - 1350 = 1464) :=
by {
  sorry
}

end johns_remaining_money_l753_753061


namespace sum_possible_values_f_l753_753243

theorem sum_possible_values_f (d : ℕ) (Hd : d > 0) :
  let f_values := {1, 2, 3, 4, 5, 6} in
  f_values.sum = 21 :=
by
  let P := 180 * d
  let product_eq := 60 * P = P^3
  have h₁ : ∀ f g, f * g * 3 = P → f * g = 60 * d := sorry
  let f_possible_pairs := [{(1, 60), (60, 1)}, 
                           {(2, 30), (30, 2)}, 
                           {(3, 20), (20, 3)}, 
                           {(4, 15), (15, 4)}, 
                           {(5, 12), (12, 5)}, 
                           {(6, 10), (10, 6)}]
  let f_values := {1, 2, 3, 4, 5, 6}
  have unique_f_values : ∀ f ∈ f_values, f = 1 ∨ f = 2 ∨ f = 3 ∨ f = 4 ∨ f = 5 ∨ f = 6 := sorry
  have sum_f_values : ∑ f in f_values, f = 21 := sorry
  sum_f_values
sorry

end sum_possible_values_f_l753_753243


namespace least_number_subtracted_divisible_l753_753555

theorem least_number_subtracted_divisible (n : ℕ) (divisor : ℕ) (rem : ℕ) :
  n = 427398 → divisor = 15 → n % divisor = rem → rem = 3 → ∃ k : ℕ, n - k = 427395 :=
by
  intros
  use 3
  sorry

end least_number_subtracted_divisible_l753_753555


namespace repeating_decimal_sum_l753_753867

theorem repeating_decimal_sum : (0.\overline{234} - 0.\overline{567} + 0.\overline{891}) = (186 / 333) :=
by
  have h1 : 0.\overline{234} = (234 / 999) := sorry
  have h2 : 0.\overline{567} = (567 / 999) := sorry
  have h3 : 0.\overline{891} = (891 / 999) := sorry
  calc
    (0.\overline{234} - 0.\overline{567} + 0.\overline{891})
        = ( (234 / 999) - (567 / 999) + (891 / 999) ) : by rw [h1, h2, h3]
    ... = ( (234 - 567 + 891) / 999 )            : by simp
    ... = 558 / 999                              : by norm_num
    ... = 186 / 333                              : by norm_num

end repeating_decimal_sum_l753_753867


namespace rhombus_area_l753_753667

theorem rhombus_area (a b c : ℝ) (side length : ℝ = sqrt 109) (diagonal_diff : ℝ = 12) :
  let x := -3 + sqrt 182 in
  let shorter_diagonal := 2 * x, longer_diagonal := 2 * (x + 6) in
  let area := shorter_diagonal * longer_diagonal * (1/2) in
  side_length = sqrt 109 ∧ diagonal_diff = 12 ∧
  x^2 + (x + 6)^2 = side_length^2 →
  area = 364 :=
by sorry

end rhombus_area_l753_753667


namespace intersection_distance_l753_753339

-- Definitions
def parametric_eq_line (t : ℝ) : ℝ × ℝ := (2 + t, 1 + t)
def polar_eq_circle (θ : ℝ) : ℝ := 4 * (Real.sin θ + Real.cos θ)

-- Rectangular equation of line
def line_eq (x y : ℝ) : Prop := y = x - 1

-- Rectangular equation of circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y = 0

-- Distance function (Euclidean)
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def point_P : ℝ × ℝ := (2, 1)

-- Main theorem
theorem intersection_distance :
  (∀ t : ℝ, line_eq (parametric_eq_line t).1 (parametric_eq_line t).2) ∧
  (∀ θ : ℝ, circle_eq (4 * (Real.sin θ), 4 * (Real.cos θ))) ∧
  (let A := parametric_eq_line t_1 in
   let B := parametric_eq_line t_2 in
   let PA := distance point_P A in
   let PB := distance point_P B in
   t_1 + t_2 = Real.sqrt 2 ∧ t_1 * t_2 = -7 →
   ||PA - PB|| = Real.sqrt 2) :=
by
  sorry

end intersection_distance_l753_753339


namespace ellipse_hyperbola_foci_l753_753825

theorem ellipse_hyperbola_foci {a b : ℝ} (h1 : b^2 - a^2 = 25) (h2 : a^2 + b^2 = 49) :
  a = 2 * Real.sqrt 3 ∧ b = Real.sqrt 37 :=
by sorry

end ellipse_hyperbola_foci_l753_753825


namespace tetrahedron_midpoint_distance_l753_753723

variables {V : Type*} [InnerProductSpace ℝ V]

def dist_sq (p q : V) : ℝ := ∥p - q∥^2

theorem tetrahedron_midpoint_distance
  {b c d r : V}
  (h : ⟪b, d⟫ = 0) :
  dist_sq r (c / 2) + dist_sq r ((b + d) / 2) = dist_sq r (d / 2) + dist_sq r ((b + c) / 2) :=
by sorry

end tetrahedron_midpoint_distance_l753_753723


namespace range_for_a_l753_753690

-- Definitions from the conditions
def f (x : ℝ) := x ^ (-1/2)

-- Theorem statement based on the translation
theorem range_for_a (a : ℝ) (h1 : a + 1 > 0) (h2 : 10 - 2 * a > 0) (h3 : f (a + 1) < f (10 - 2 * a)) : 3 < a ∧ a < 5 :=
by
  -- Skip the detailed proof
  sorry

end range_for_a_l753_753690


namespace shiny_pennies_total_a_plus_b_l753_753183

noncomputable def shiny_pennies_prob : ℚ := 155 / 231

theorem shiny_pennies_total_a_plus_b : (shiny_pennies_prob.num + shiny_pennies_prob.denom) = 386 := 
by
  sorry

end shiny_pennies_total_a_plus_b_l753_753183


namespace sum_of_2503_terms_l753_753918

def sequence (b : ℕ → ℤ) : Prop :=
  ∀ n ≥ 3, b n = b (n - 1) - b (n - 2)

def sum_of_first_n_terms (b : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, b (i + 1)

theorem sum_of_2503_terms (b : ℕ → ℤ) 
  (h_sequence : sequence b) 
  (sum_2001 : sum_of_first_n_terms b 2001 = 1492) 
  (sum_1492 : sum_of_first_n_terms b 1492 = 2001) : 
  sum_of_first_n_terms b 2503 = 1001 := 
sorry

end sum_of_2503_terms_l753_753918


namespace walking_west_10_neg_l753_753371

-- Define the condition that walking east for 20 meters is +20 meters
def walking_east_20 := 20

-- Assert that walking west for 10 meters is -10 meters given the east direction definition
theorem walking_west_10_neg : walking_east_20 = 20 → (-10 = -10) :=
by
  intro h
  sorry

end walking_west_10_neg_l753_753371


namespace permits_increase_l753_753477

theorem permits_increase :
  let old_permits := 26^2 * 10^3
  let new_permits := 26^4 * 10^4
  new_permits = 67600 * old_permits :=
by
  let old_permits := 26^2 * 10^3
  let new_permits := 26^4 * 10^4
  exact sorry

end permits_increase_l753_753477


namespace complex_expression_evaluation_l753_753673

noncomputable def z : ℂ := -1 + complex.I

theorem complex_expression_evaluation : (z + 2) / (z^2 + z) = -1 := 
by
  -- This is just a statement and we intentionally skip the proof with sorry.
  sorry

end complex_expression_evaluation_l753_753673


namespace smoking_lung_cancer_correct_answer_is_B_l753_753037

-- Definitions based on conditions
def smoking_related_to_lung_cancer : Prop := 
  -- The proposition that smoking is related to lung cancer
  sorry

def confidence_level (p : Prop) (conf : ℝ) : Prop := 
  -- The proposition that there is conf% confidence in the proposition p
  sorry

def error_probability (p : Prop) (err : ℝ) : Prop := 
  -- The proposition that the probability of making a mistake in believing p does not exceed err
  sorry

-- Given conditions
axiom hypothesis1 : confidence_level smoking_related_to_lung_cancer 0.99
axiom hypothesis2 : error_probability smoking_related_to_lung_cancer 0.01

-- Main statement to prove that the correct answer is "B"
theorem smoking_lung_cancer_correct_answer_is_B : 
  ∀ (A: Prop) (B: Prop) (C: Prop) (D: Prop),
  (B = (error_probability smoking_related_to_lung_cancer 0.01)) →
  (C = (∀ (s : Type), smokers s → lung_cancer s)) →
  (D = approximate_probability_of_smokers_with_lung_cancer 0.99) →
  B :=
begin
  intros A B C D,
  sorry
end

end smoking_lung_cancer_correct_answer_is_B_l753_753037


namespace handshake_problem_l753_753849

theorem handshake_problem :
  ∃ (a b : ℕ), a + b = 20 ∧ (a * (a - 1)) / 2 + (b * (b - 1)) / 2 = 106 ∧ a * b = 84 :=
by
  sorry

end handshake_problem_l753_753849


namespace prob_product_less_than_36_is_15_over_16_l753_753789

noncomputable def prob_product_less_than_36 : ℚ := sorry

theorem prob_product_less_than_36_is_15_over_16 :
  prob_product_less_than_36 = 15 / 16 := 
sorry

end prob_product_less_than_36_is_15_over_16_l753_753789


namespace decagon_angle_measure_l753_753102

theorem decagon_angle_measure (A B C D E F G H I J P : Type)
  [regular_decagon : regular_decagon ABCDEFGHIJ]
  (h1 : sides_extended_to_meet P A H C D) 
  : angle_P_AH_CD P = 108 :=
by
  sorry

end decagon_angle_measure_l753_753102


namespace volume_of_original_cube_l753_753788

theorem volume_of_original_cube (a : ℝ) 
  (h1 : (a + 3) * (a - 2) * a = a^3 + a^2 - 6a + 6)
  (h2 : (a^3 + a^2 - 6a - a^3) = 6) :
  (3 + Real.sqrt 15) ^ 3 = (a^3) :=
by
  sorry

end volume_of_original_cube_l753_753788


namespace find_locus_l753_753657

def trihedral_locus (O : ℝ × ℝ × ℝ) (a : ℝ) : set (ℝ × ℝ × ℝ) :=
  {P | let (x, y, z) := P in (|x| + |y| + |z| >= a) ∧ (x^2 + y^2 + z^2 <= a^2) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)}

theorem find_locus (O : ℝ × ℝ × ℝ) (a : ℝ) :
  O = (0, 0, 0) →
  ∀ (P : ℝ × ℝ × ℝ), trihedral_locus O a P ↔ ((|P.1| + |P.2| + |P.3| >= a) ∧ (P.1^2 + P.2^2 + P.3^2 <= a^2) ∧ (P.1 ≠ 0 ∨ P.2 ≠ 0 ∨ P.3 ≠ 0)) :=
by
  intros hO P
  simp [trihedral_locus]
  apply Iff.rfl

end find_locus_l753_753657


namespace two_truth_tellers_are_B_and_C_l753_753811

-- Definitions of students and their statements
def A_statement_false (A_said : Prop) (A_truth_teller : Prop) := ¬A_said = A_truth_teller
def B_statement_true (B_said : Prop) (B_truth_teller : Prop) := B_said = B_truth_teller
def C_statement_true (C_said : Prop) (C_truth_teller : Prop) := C_said = C_truth_teller
def D_statement_false (D_said : Prop) (D_truth_teller : Prop) := ¬D_said = D_truth_teller

-- Given statements
def A_said := ¬ (False : Prop)
def B_said := True
def C_said := B_said ∨ D_statement_false True True
def D_said := False

-- Define who is telling the truth
def A_truth_teller := False
def B_truth_teller := True
def C_truth_teller := True
def D_truth_teller := False

-- Proof problem statement
theorem two_truth_tellers_are_B_and_C :
  (A_statement_false A_said A_truth_teller) ∧
  (B_statement_true B_said B_truth_teller) ∧
  (C_statement_true C_said C_truth_teller) ∧
  (D_statement_false D_said D_truth_teller) →
  ((A_truth_teller = False) ∧
  (B_truth_teller = True) ∧
  (C_truth_teller = True) ∧
  (D_truth_teller = False)) := 
by {
  sorry
}

end two_truth_tellers_are_B_and_C_l753_753811


namespace simplify_expression_l753_753467

-- Define the fractions involved
def frac1 : ℚ := 1 / 2
def frac2 : ℚ := 1 / 3
def frac3 : ℚ := 1 / 5
def frac4 : ℚ := 1 / 7

-- Define the expression to be simplified
def expr : ℚ := (frac1 - frac2 + frac3) / (frac2 - frac1 + frac4)

-- The goal is to show that the expression simplifies to -77 / 5
theorem simplify_expression : expr = -77 / 5 := by
  sorry

end simplify_expression_l753_753467


namespace partition_not_always_possible_l753_753448

-- Definitions based on conditions
variables (cake : set (ℝ × ℝ)) (chocolates : finset (set (ℝ × ℝ)))
def is_square (s : set (ℝ × ℝ)) : Prop := ∃ a : ℝ, s = {p | p.1 ≥ 0 ∧ p.1 ≤ a ∧ p.2 ≥ 0 ∧ p.2 ≤ a}
def triangle (s : set (ℝ × ℝ)) : Prop := ∃ (v1 v2 v3 : ℝ × ℝ), s = convex_hull ℝ ({v1, v2, v3} : set (ℝ × ℝ))

axiom non_touching_chocolates (chocolates : finset (set (ℝ × ℝ))) : ∀ (c1 c2 ∈ chocolates), c1 ≠ c2 → disjoint c1 c2

def partition (s : set (ℝ × ℝ)) (polys : finset (set (ℝ × ℝ))) : Prop :=
  (∀ p ∈ polys, convex ℝ p) ∧ (⋃ p ∈ polys, p = s) ∧ (∀ p1 p2 ∈ polys, p1 ≠ p2 → disjoint p1 p2)

noncomputable def contains_exactly_one (s : set (ℝ × ℝ)) (chocolates : finset (set (ℝ × ℝ))) : Prop :=
  ∃ c ∈ chocolates, s = c

theorem partition_not_always_possible :
  is_square cake →
  (∀ t ∈ chocolates, triangle t) →
  non_touching_chocolates chocolates →
  ¬(∀ polys, partition cake polys → (∀ p ∈ polys, contains_exactly_one p chocolates)) :=
sorry

end partition_not_always_possible_l753_753448


namespace arithmetic_proof_l753_753890

def arithmetic_expression := 3889 + 12.952 - 47.95000000000027
def expected_result := 3854.002

theorem arithmetic_proof : arithmetic_expression = expected_result := by
  -- The proof goes here
  sorry

end arithmetic_proof_l753_753890


namespace final_position_A_final_position_B_fuel_consumption_A_fuel_consumption_B_less_fuel_consumption_l753_753523

-- Definitions of the driving records for trainee A and B
def driving_record_A : List Int := [15, -2, 5, -1, 10, -3, -2, 12, 4, -5, 6]
def driving_record_B : List Int := [-17, 9, -2, 8, 6, 9, -5, -1, 4, -7, -8]

-- Fuel consumption rate per kilometer
variable (a : ℝ)

-- Proof statements in Lean
theorem final_position_A : driving_record_A.sum = 39 := by sorry
theorem final_position_B : driving_record_B.sum = -4 := by sorry
theorem fuel_consumption_A : (driving_record_A.map (abs)).sum * a = 65 * a := by sorry
theorem fuel_consumption_B : (driving_record_B.map (abs)).sum * a = 76 * a := by sorry
theorem less_fuel_consumption : (driving_record_A.map (abs)).sum * a < (driving_record_B.map (abs)).sum * a := by sorry

end final_position_A_final_position_B_fuel_consumption_A_fuel_consumption_B_less_fuel_consumption_l753_753523


namespace phone_answered_before_fifth_ring_l753_753188

theorem phone_answered_before_fifth_ring:
  (0.1 + 0.2 + 0.25 + 0.25 = 0.8) :=
by
  sorry

end phone_answered_before_fifth_ring_l753_753188


namespace average_mark_of_first_class_is_40_l753_753136

open Classical

noncomputable def average_mark_first_class (n1 n2 : ℕ) (m2 : ℕ) (a : ℚ) : ℚ :=
  let x := (a * (n1 + n2) - n2 * m2) / n1
  x

theorem average_mark_of_first_class_is_40 : average_mark_first_class 30 50 90 71.25 = 40 := by
  sorry

end average_mark_of_first_class_is_40_l753_753136


namespace bran_amount_to_pay_l753_753236

variable (tuition_fee scholarship_percentage monthly_income payment_duration : ℝ)

def amount_covered_by_scholarship : ℝ := scholarship_percentage * tuition_fee

def remaining_after_scholarship : ℝ := tuition_fee - amount_covered_by_scholarship

def total_earnings_part_time_job : ℝ := monthly_income * payment_duration

def amount_still_to_pay : ℝ := remaining_after_scholarship - total_earnings_part_time_job

theorem bran_amount_to_pay (h_tuition_fee : tuition_fee = 90)
                          (h_scholarship_percentage : scholarship_percentage = 0.30)
                          (h_monthly_income : monthly_income = 15)
                          (h_payment_duration : payment_duration = 3) :
  amount_still_to_pay tuition_fee scholarship_percentage monthly_income payment_duration = 18 := 
by
  sorry

end bran_amount_to_pay_l753_753236


namespace sixth_employee_salary_l753_753120

def salaries : List Real := [1000, 2500, 3100, 3650, 1500]

def mean_salary_of_six : Real := 2291.67

theorem sixth_employee_salary : 
  let total_five := salaries.sum 
  let total_six := mean_salary_of_six * 6
  (total_six - total_five) = 2000.02 :=
by
  sorry

end sixth_employee_salary_l753_753120


namespace max_plates_l753_753215

def cost_pan : ℕ := 3
def cost_pot : ℕ := 5
def cost_plate : ℕ := 11
def total_cost : ℕ := 100
def min_pans : ℕ := 2
def min_pots : ℕ := 2

theorem max_plates (p q r : ℕ) :
  p >= min_pans → q >= min_pots → (cost_pan * p + cost_pot * q + cost_plate * r = total_cost) → r = 7 :=
by
  intros h_p h_q h_cost
  sorry

end max_plates_l753_753215


namespace trapezoid_area_l753_753148

theorem trapezoid_area :
  let y1 := 10
  let y2 := 18
  let x1 := y1 / 2
  let x2 := y2 / 2
  let height := y2 - y1
  let base1 := x1
  let base2 := x2
  let area := (1 / 2 : ℝ) * (base1 + base2) * height
  area = 56.0 :=
by
  let y1 := 10
  let y2 := 18
  let x1 := y1 / 2
  let x2 := y2 / 2
  let height := y2 - y1
  let base1 := x1
  let base2 := x2
  let area := (1 / 2 : ℝ) * (base1 + base2) * height
  show area = 56.0
  sorry

end trapezoid_area_l753_753148


namespace who_finished_in_7th_place_l753_753713

theorem who_finished_in_7th_place:
  ∀ (Alex Ben Charlie David Ethan : ℕ),
  (Ethan + 4 = Alex) →
  (David + 1 = Ben) →
  (Charlie = Ben + 3) →
  (Alex = Ben + 2) →
  (Ethan + 2 = David) →
  (Ben = 5) →
  Alex = 7 :=
by
  intros Alex Ben Charlie David Ethan h1 h2 h3 h4 h5 h6
  sorry

end who_finished_in_7th_place_l753_753713


namespace log_identity_l753_753938

theorem log_identity : (Real.log 2)^3 + 3 * (Real.log 2) * (Real.log 5) + (Real.log 5)^3 = 1 :=
by
  sorry

end log_identity_l753_753938


namespace probability_same_fruits_l753_753672

theorem probability_same_fruits :
  let fruits : Finset ℕ := {1, 2, 3, 4},
      n := fruits.card,
      k := 2
  in (∑ _ in (fruits.subsets k), 1 : ℚ) / ((∑ _ in (fruits.subsets k), 1 : ℚ) * (∑ _ in (fruits.subsets k), 1 : ℚ)) = 1 / 6 :=
by
  sorry

end probability_same_fruits_l753_753672


namespace find_common_difference_l753_753328

variable {a : ℕ → ℝ} (h_arith : ∀ n, a (n + 1) = a n + d)
variable (a7_minus_2a4_eq_6 : a 7 - 2 * a 4 = 6)
variable (a3_eq_2 : a 3 = 2)

theorem find_common_difference (d : ℝ) : d = 4 :=
by
  -- Proof would go here
  sorry

end find_common_difference_l753_753328


namespace negation_of_cos_inequality_l753_753454

theorem negation_of_cos_inequality :
  (∀ x : ℝ, x > 0 → cos x ≥ -1) ↔ (¬ ∃ x : ℝ, x > 0 ∧ cos x < -1) := by
  sorry

end negation_of_cos_inequality_l753_753454


namespace two_positions_of_point_c_l753_753716

theorem two_positions_of_point_c (A B C : Point) (distance_AB : dist A B = 12)
  (isosceles : is_isosceles A B C)
  (perimeter_ABC : perimeter A B C = 36)
  (area_ABC : area A B C = 72):
  ∃! C₁ C₂ : Point, dist A C₁ = dist B C₁ ∧ dist A C₂ = dist B C₂ ∧
              perimeter A B C₁ = 36 ∧ area A B C₁ = 72 ∧
              perimeter A B C₂ = 36 ∧ area A B C₂ = 72 :=
sorry

end two_positions_of_point_c_l753_753716


namespace angle_measure_ADB_l753_753031

-- Representing angles as real numbers and defining the right triangle ABC with specified angles
noncomputable def triangle_ABC (A B C D : Point) :=
  ∃ (α β γ : ℝ), α = 45 ∧ β = 45 ∧ γ = 90 ∧
  (α + β + γ = 180)

-- Defining the angle bisectors
def angle_bisector_AD (A B C D : Point) : Prop :=
  is_angle_bisector A B

def angle_bisector_BD (A B C D : Point) : Prop :=
  is_angle_bisector B A

-- Proving the measure of angle ADB
theorem angle_measure_ADB (A B C D : Point)
  (h₁ : triangle_ABC A B C D)
  (h₂ : angle_bisector_AD A B C D)
  (h₃ : angle_bisector_BD A B C D) :
  measure_angle A D B = 135 :=
by sorry -- Proof to be filled in later

end angle_measure_ADB_l753_753031


namespace coloring_ways_l753_753410

-- The problem setup
structure Grid (n : Nat) :=
  (cells : Fin n × Fin n → Fin 2) -- a coloring of the grid, where Fin 2 represents two colors

-- Condition for adjacent cells not sharing the same color
def valid_coloring (g : Grid 3) : Prop :=
  ∀ i j, (g.cells (⟨i, sorry⟩) = g.cells (⟨i + 1, sorry⟩) → False) ∧
         (g.cells (⟨i, sorry⟩) = g.cells (⟨i - 1, sorry⟩) → False) ∧
         (g.cells (⟨i, sorry⟩) = g.cells (⟨i, sorry⟩.symm) → False) ∧
         (g.cells (⟨j, sorry⟩) = g.cells (⟨j, sorry⟩.symm) → False) 

-- Proof problem statement
theorem coloring_ways :
  ∃! (g : Grid 3), valid_coloring g :=
sorry

end coloring_ways_l753_753410


namespace g_19_57_l753_753254

def g (a b : ℕ) : ℕ

axiom g_self {a : ℕ} (ha : a > 0) : g a a = a
axiom g_comm {a b : ℕ} (ha : a > 0) (hb : b > 0) : g a b = g b a
axiom g_third_condition {a b : ℕ} (ha : a > 0) (hb : b > 0) : (a + b) * g a b = b * g a (a + b)

theorem g_19_57 : g 19 57 = 57 :=
by
  sorry

end g_19_57_l753_753254


namespace M_lt_N_l753_753765

theorem M_lt_N (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let M := (x + y) / (2 + x + y)
  let N := (x / (2 + x) + y / (2 + y))
in M < N :=
by
  sorry

end M_lt_N_l753_753765


namespace muffins_division_l753_753741

theorem muffins_division (total_muffins total_people muffins_per_person : ℕ) 
  (h1 : total_muffins = 20) (h2 : total_people = 5) (h3 : muffins_per_person = total_muffins / total_people) : 
  muffins_per_person = 4 := 
by
  sorry

end muffins_division_l753_753741


namespace fifth_term_arithmetic_sequence_l753_753111

variable {x y : ℝ}

-- Conditions
def first_term : ℝ := x^2 + 2y
def second_term : ℝ := x^2 - 2y
def third_term : ℝ := x + y
def fourth_term : ℝ := x - y

-- Common difference calculation from the first two terms
def common_difference : ℝ := second_term - first_term

theorem fifth_term_arithmetic_sequence : 
  common_difference = -4y →
  first_term = x^2 + 2y →
  second_term = x^2 - 2y →
  third_term = x + y →
  fourth_term = x - y →
  fourth_term + common_difference = x - 5y := by
  intros h1 h2 h3 h4 h5
  rw [h4, h1]
  finish

end fifth_term_arithmetic_sequence_l753_753111


namespace traveler_can_cross_the_desert_l753_753700

noncomputable def canCrossDesert (distance : ℕ) (dailyDistance : ℕ) (maxDaysCarry : ℕ) (totalDays : ℕ) : Prop :=
  distance = 80 ∧ dailyDistance = 20 ∧ maxDaysCarry = 3 ∧ totalDays = 6 → true

theorem traveler_can_cross_the_desert :
  canCrossDesert 80 20 3 6 :=
by
  intros h
  cases h
  exact trivial

end traveler_can_cross_the_desert_l753_753700


namespace tangency_of_bd_l753_753710

open EuclideanGeometry

variables {A B C D E X Y Z O : Point}
variables {l m : Line}

theorem tangency_of_bd (h1 : ∠ A B C = 90) (h2 : ∠ A < ∠ C) 
  (h3 : tangent l (circumcircle A B C) A) (h4 : l -- BC = ⇑ D)
  (h5 : reflection BC A = E) 
  (h6 : perpendicular AX BE) (h7 : midpoint Y AX) 
  (h8 : meet_at BY (circumcircle A B C) Z) :
  is_tangent (circumcircle A D Z) BD :=
sorry

end tangency_of_bd_l753_753710


namespace number_of_managers_l753_753902

theorem number_of_managers 
    (avg_salary_managers : ℕ)
    (num_associates : ℕ)
    (avg_salary_associates : ℕ)
    (avg_salary_company : ℕ)
    (total_num_employees : ℕ) :
    avg_salary_managers = 90000 →
    num_associates = 75 →
    avg_salary_associates = 30000 →
    avg_salary_company = 40000 →
    total_num_employees = 15 + 75 →
    15 = total_num_employees - 75 :=
by
    intro h1 h2 h3 h4 h5
    have h_eq_manager : 15 = total_num_employees - 75, from
    calc
        15 = total_num_employees - 75 : by
            rw [h5]
            norm_num
    show 15 = total_num_employees - 75 from h_eq_manager

end number_of_managers_l753_753902


namespace conclusion_A_conclusion_B_conclusion_C1_conclusion_C2_l753_753660

variable {r a b x1 y1 x2 y2 : ℝ} -- variables used in the problem

-- conditions
def circle1 : x1^2 + y1^2 = r^2 := sorry -- Circle C1 equation
def circle2 : (x1 + a)^2 + (y1 + b)^2 = r^2 := sorry -- Circle C2 equation
def r_positive : r > 0 := sorry -- r > 0
def not_both_zero : ¬ (a = 0 ∧ b = 0) := sorry -- a, b are not both zero
def distinct_points : x1 ≠ x2 ∧ y1 ≠ y2 := sorry -- A(x1, y1) and B(x2, y2) are distinct

-- Proofs to be provided for each of the conclusions
theorem conclusion_A : 2 * a * x1 + 2 * b * y1 + a^2 + b^2 = 0 := sorry
theorem conclusion_B : a * (x1 - x2) + b * (y1 - y2) = 0 := sorry
theorem conclusion_C1 : x1 + x2 = -a := sorry
theorem conclusion_C2 : y1 + y2 = -b := sorry

end conclusion_A_conclusion_B_conclusion_C1_conclusion_C2_l753_753660


namespace determine_p_l753_753274

def p : ℝ → ℝ := λ x, 4 * x ^ 2 - 8 * x - 12

-- Lean 4 statement equivalent to the mathematical proof problem
theorem determine_p (p : ℝ → ℝ) :
  (∀ x, p x = 4 * (x - 3) * (x + 1)) ∧
  (∃ q : ℝ → ℝ, ∀ x, p x = 4 * q x ∧ degree q < 3) ∧
  (p 4 = 20) →
  (p = λ x, 4 * x^2 - 8 * x - 12) :=
by
  intros h
  sorry

end determine_p_l753_753274


namespace inequality_xyz_l753_753436

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 1/x + 1/y + 1/z = 3) : 
  (x - 1) * (y - 1) * (z - 1) ≤ (1/4) * (x * y * z - 1) := 
by 
  sorry

end inequality_xyz_l753_753436


namespace book_distribution_l753_753847

theorem book_distribution (a b : ℕ) (h1 : a + b = 282) (h2 : (3 / 4) * a = (5 / 9) * b) : a = 120 ∧ b = 162 := by
  sorry

end book_distribution_l753_753847


namespace intersection_points_of_regular_polygons_l753_753460

open BigOperators

theorem intersection_points_of_regular_polygons :
  ∀ (n : ℕ), n ∈ {6, 7, 8, 9, 10} →
  let pairs := [(6, 7), (6, 8), (6, 9), (6, 10), (7, 8), (7, 9), (7, 10), (8, 9), (8, 10), (9, 10)] in
  (∑ (p : ℕ × ℕ) in pairs, 2 * p.1) = 140 :=
by
  intros
  let pairs := [(6, 7), (6, 8), (6, 9), (6, 10), (7, 8), (7, 9), (7, 10), (8, 9), (8, 10), (9, 10)]
  have h_sum : (∑ (p : ℕ × ℕ) in pairs, 2 * p.1) = 140,
    sorry,
  exact h_sum

end intersection_points_of_regular_polygons_l753_753460


namespace prob_one_approval_l753_753199

theorem prob_one_approval (P_Y : ℝ) (hY : P_Y = 0.7) (P_N : ℝ) (hN : P_N = 1 - P_Y) :
    P_Y * P_N * P_N + P_N * P_Y * P_N + P_N * P_N * P_Y = 0.189 := 
by
  rw [hY, show P_N = 0.3, by rw [hY]; linarith]
  norm_num
  sorry

end prob_one_approval_l753_753199


namespace paint_room_alone_l753_753875

theorem paint_room_alone (x : ℝ) (hx : (1 / x) + (1 / 4) = 1 / 1.714) : x = 3 :=
by sorry

end paint_room_alone_l753_753875


namespace select_students_for_competitions_l753_753846

theorem select_students_for_competitions : 
  let total_students := 9
  let only_chess := 2
  let only_go := 3
  let both := 4
  total_students = only_chess + only_go + both → 
  (only_chess * only_go) + (both * only_go) + (both * only_chess) + Nat.choose both 2 = 32 := by
  intros
  sorry

end select_students_for_competitions_l753_753846


namespace positive_difference_of_A_and_B_is_967_l753_753269

def sumA : ℕ := (List.range' 1 44).sum (λ n => (n * (n + 1)))
def sumB : ℕ := (List.range' 1 43).sum (λ n =>
  if n % 2 == 1 then n else (n * (n + 1) / 2))

theorem positive_difference_of_A_and_B_is_967 :
  |sumA - sumB| = 967 := sorry

end positive_difference_of_A_and_B_is_967_l753_753269


namespace lcm_of_three_numbers_l753_753326

theorem lcm_of_three_numbers 
  (A B C : ℕ)
  (h1 : A * B * C = 185771616)
  (h2 : Nat.gcd (Nat.gcd A B) C = 121)
  (h3 : Nat.gcd A B = 363) : 
  Nat.lcm (Nat.lcm A B) C = 61919307 := 
by
sory

end lcm_of_three_numbers_l753_753326


namespace find_range_a_l753_753687

-- Define the parabola equation y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line equation y = (√3/3) * (x - a)
def line (x y a : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - a)

-- Define the focus of the parabola
def focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Define the condition that F is outside the circle with diameter CD
def F_outside_circle_CD (x1 y1 x2 y2 a : ℝ) : Prop :=
  (x1 - 1) * (x2 - 1) + y1 * y2 > 0

-- Define the parabola-line intersection points and the related Vieta's formulas
def intersection_points (a : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = 2 * a + 12 ∧ x1 * x2 = a^2

-- Define the final condition for a
def range_a (a : ℝ) : Prop :=
  -3 < a ∧ a < -2 * Real.sqrt 5 + 3

-- Main theorem statement
theorem find_range_a (a : ℝ) (hneg : a < 0)
  (x1 x2 y1 y2 : ℝ)
  (hparabola1 : parabola x1 y1)
  (hparabola2 : parabola x2 y2)
  (hline1 : line x1 y1 a)
  (hline2 : line x2 y2 a)
  (hfocus : focus 1 0)
  (hF_out : F_outside_circle_CD x1 y1 x2 y2 a)
  (hintersect : intersection_points a x1 x2) :
  range_a a := 
sorry

end find_range_a_l753_753687


namespace square_of_distance_centroid_incenter_l753_753793

-- Define the basic properties and distances
variable (G I : Point)
variable (p r R : ℝ)

-- Define the distance function
noncomputable def distance (P Q : Point) : ℝ := 
  sorry 

-- Define the centroid G and incenter I of a triangle
axiom centroid (triangle : Triangle) : Point := 
  sorry 

axiom incenter (triangle : Triangle) : Point := 
  sorry 

-- Define the properties p, r, R for the triangle
axiom semiperimeter (triangle : Triangle) : ℝ := 
  sorry 

axiom inradius (triangle : Triangle) : ℝ := 
  sorry 

axiom circumradius (triangle : Triangle) : ℝ := 
  sorry 

-- The proof statement
theorem square_of_distance_centroid_incenter (triangle : Triangle) :
  distance (centroid triangle) (incenter triangle) ^ 2 =
  1 / 9 * (semiperimeter triangle ^ 2 + 5 * inradius triangle ^ 2 - 16 * circumradius triangle * inradius triangle) := 
  sorry

end square_of_distance_centroid_incenter_l753_753793


namespace find_x_l753_753172

theorem find_x :
  (let x := 53.97 in
   (sqrt 97 + sqrt 486) / sqrt x = 4.340259786868312) :=
by
  let x := 53.97
  sorry

end find_x_l753_753172


namespace equilateral_triangle_of_kite_PIRQ_l753_753417

theorem equilateral_triangle_of_kite_PIRQ
  (A B C I A' B' C' P Q R : Type*)
  [IsTriangle A B C]
  (h1 : IsCircumcircle A B C Γ I)
  (h2 : InternalAngleBisector A B C meetsCircumcircleAt A' B' C')
  (h3 : LineIntersections P B'C' AA' Q AC)
  (h4 : LineIntersections R BB' AC)
  (h5 : Quadrilateral_is_kite P I R Q IP_eq_IR QP_eq_QR) :
  IsEquilateralTriangle A B C :=
sorry

end equilateral_triangle_of_kite_PIRQ_l753_753417


namespace hajun_school_trip_time_l753_753347

-- Define the conditions
def total_time : ℕ := 96
def times_went : ℕ := 5
def time_per_trip : ℝ := total_time / times_went

-- The proof problem statement
theorem hajun_school_trip_time :
  time_per_trip = 19.2 :=
sorry

end hajun_school_trip_time_l753_753347


namespace number_of_sets_containing_6_sum_30_l753_753135

open Finset

noncomputable def count_sets_contain_six_sum_30 : Nat := 
  (powerset (finset.range 11).erase 0).filter (λ s, s.card = 4 ∧ 6 ∈ s ∧ s.sum id = 30).card

theorem number_of_sets_containing_6_sum_30 :
  count_sets_contain_six_sum_30 = 6 := 
sorry

end number_of_sets_containing_6_sum_30_l753_753135


namespace divide_condition_l753_753622

theorem divide_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ n : ℕ, 0 < n ∧ a ∣ (b^n - n) :=
by
  sorry

end divide_condition_l753_753622


namespace arithmetic_geometric_common_ratio_l753_753659

theorem arithmetic_geometric_common_ratio (a₁ r : ℝ) 
  (h₁ : a₁ + a₁ * r^2 = 10) 
  (h₂ : a₁ * (1 + r + r^2 + r^3) = 15) : 
  r = 1/2 ∨ r = -1/2 :=
by {
  sorry
}

end arithmetic_geometric_common_ratio_l753_753659


namespace avery_donation_l753_753228

theorem avery_donation (shirts pants shorts : ℕ)
  (h_shirts : shirts = 4)
  (h_pants : pants = 2 * shirts)
  (h_shorts : shorts = pants / 2) :
  shirts + pants + shorts = 16 := by
  sorry

end avery_donation_l753_753228


namespace triangle_shape_right_range_of_dot_product_l753_753045

variables {α : Type*} [inner_product_space ℝ α] {A B C : α}

theorem triangle_shape_right (hAB : ∥B - A∥ = 1) (hAC : ∥C - A∥ = sqrt 3)
  (hBC : ∥(B - A) + (C - A)∥ = ∥C - B∥) : inner (B - A) (C - A) = 0 :=
sorry

theorem range_of_dot_product (hAB : ∥B - A∥ = 1) (hAC : ∥C - A∥ = sqrt 3)
  (h_lambda : ∃ λ : ℝ, ∥λ • (B - A) - (C - A)∥ ≤ sqrt 2) :
  inner (B - A) (C - A) ∈ set.Icc (-sqrt 3) (-1) ∪ set.Icc 1 (sqrt 3) :=
sorry

end triangle_shape_right_range_of_dot_product_l753_753045


namespace negation_equivalence_l753_753692

variable (a : ℝ)

def original_proposition (a : ℝ) : Prop := a > 0 → exp a ≥ 1

def negation_of_proposition : Prop := ∃ a > 0, exp a < 1

theorem negation_equivalence :
  ¬(∀ a, a > 0 → exp a ≥ 1) ↔ (∃ a, a > 0 ∧ exp a < 1) := by
  sorry

end negation_equivalence_l753_753692


namespace students_with_neither_cool_parents_l753_753715

theorem students_with_neither_cool_parents 
  (total_students : ℕ) 
  (cool_dads : ℕ) 
  (cool_moms : ℕ) 
  (both_cool_parents : ℕ) 
  (absent_students : ℕ) 
  (total_students = 50)
  (cool_dads = 25) 
  (cool_moms = 30) 
  (both_cool_parents = 14) 
  (absent_students = 5) : 
  4 ≤ (total_students - absent_students) - (cool_dads + cool_moms - both_cool_parents) ∧ 
  (total_students - absent_students) - (cool_dads + cool_moms - both_cool_parents) ≤ 9 :=
by 
  sorry

end students_with_neither_cool_parents_l753_753715


namespace value_of_x_l753_753727

-- Definitions of the angles and line segment
variables (R P S Q : Point)
variables (RPQ PQR SRQ PRQ : ℝ)

-- Given conditions
axiom RPQ_def : RPQ = 50
axiom PQR_def : PQR = 90
axiom R_on_PS : R ∈ line_seg(P, S)

-- Problem statement: Prove that the angle SRQ is 140 degrees
theorem value_of_x :
  SRQ = 140 :=
by
  sorry

end value_of_x_l753_753727


namespace emily_sees_emerson_time_l753_753787

theorem emily_sees_emerson_time :
  ∀ (emily_speed emerson_speed : ℝ) (distance_forward distance_backward : ℝ),
  emily_speed = 15 ∧ emerson_speed = 9 ∧ distance_forward = 1 ∧ distance_backward = 1 →
  let relative_speed := emily_speed - emerson_speed in
  let time_forward := distance_forward / relative_speed in
  let time_backward := distance_backward / relative_speed in
  let total_time_hours := time_forward + time_backward in
  let total_time_minutes := total_time_hours * 60 in
  total_time_minutes = 20 :=
by
  intros emily_speed emerson_speed distance_forward distance_backward,
  intro conditions,
  rcases conditions with ⟨h_emily_speed, h_emerson_speed, h_distance_forward, h_distance_backward⟩,
  have relative_speed := emily_speed - emerson_speed,
  have time_forward := distance_forward / relative_speed,
  have time_backward := distance_backward / relative_speed,
  have total_time_hours := time_forward + time_backward,
  have total_time_minutes := total_time_hours * 60,
  rw [h_emily_speed, h_emerson_speed, h_distance_forward, h_distance_backward] at *,
  sorry

end emily_sees_emerson_time_l753_753787


namespace pears_needed_to_match_weight_of_oranges_l753_753745

theorem pears_needed_to_match_weight_of_oranges 
  (weight_equiv : 7 * w_orange = 5 * w_pear)
  (oranges : nat) (h : oranges = 49) :
  5 * oranges / 7 = 35 :=
by sorry

end pears_needed_to_match_weight_of_oranges_l753_753745


namespace probability_both_selected_l753_753855

theorem probability_both_selected (P_X P_Y : ℝ) (hX : P_X = 1/7) (hY : P_Y = 2/5) :
    P_X * P_Y = 2/35 :=
by
  rw [hX, hY]
  norm_num
  sorry

end probability_both_selected_l753_753855


namespace find_divisor_value_l753_753545

theorem find_divisor_value
  (D : ℕ) 
  (h1 : ∃ k : ℕ, 242 = k * D + 6)
  (h2 : ∃ l : ℕ, 698 = l * D + 13)
  (h3 : ∃ m : ℕ, 940 = m * D + 5) : 
  D = 14 :=
by
  sorry

end find_divisor_value_l753_753545


namespace solve_equation_l753_753469

noncomputable def f (x : ℝ) : ℝ := (4 * x + 5) / (3 * (1 - x))

theorem solve_equation (x : ℝ) 
  (h1 : x ≠ 2)
  (h2 : x ≠ -1) :
  f ((x + 1) / (x - 2)) + 2 * f ((x - 2) / (x + 1)) = x := 
by 
  let z := (x - 2) / (x + 1)
  have hz_ne_one : z ≠ 1 := by 
    rw [← sub_ne_zero, ← mul_ne_zero_iff] 
    exact h2
  have hz_ne_zero : z ≠ 0 := by 
    rw [← sub_ne_zero, ← mul_ne_zero_iff_neg] 
    exact h1
  sorry

end solve_equation_l753_753469


namespace simplify_expression_l753_753803

-- Definitions derived from the problem statement
variable (x : ℝ)

-- Theorem statement
theorem simplify_expression : 1 - (1 + (1 - (1 + (1 - x)))) = 1 - x :=
sorry

end simplify_expression_l753_753803


namespace geometric_sequence_general_term_and_limit_l753_753032

/-- Lean 4 statement for proving the general term and limit of the geometric sequence -/
theorem geometric_sequence_general_term_and_limit (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) (n : ℕ) 
    (h_geom_seq : ∀ n, a (n + 1) = a n * q)
    (h_a1_pos : a 1 > 0)
    (h_Sn_80 : ∑ i in range n, a (i + 1) = 80)
    (h_max_n : ∃ (i < n), a (i + 1) = 54)
    (h_S2n_6560 : ∑ i in range (2 * n), a (i + 1) = 6560) :
    (∀ n, a n = 2 * 3 ^ (n - 1)) ∧ (tendsto (λ n, a n / ∑ i in range n, a (i + 1)) at_top (nhds (2 / 3))) := 
begin
    sorry
end

end geometric_sequence_general_term_and_limit_l753_753032


namespace fraction_difference_l753_753147

theorem fraction_difference :
  (↑(1+4+7) / ↑(2+5+8)) - (↑(2+5+8) / ↑(1+4+7)) = - (9 / 20) :=
by
  sorry

end fraction_difference_l753_753147


namespace q_days_work_l753_753880

noncomputable def q_days_work_takes : ℝ :=
  let work_left := 0.5333333333333333
  let work_p := 1 / 15
  fun p_work_days => sorry

theorem q_days_work (d : ℝ) : 
  let p_rate := 1 / 15
  let q_rate := 1 / d
  let together_rate := p_rate + q_rate
  let days_together := 4
  let fraction_left := 0.5333333333333333
  d = 20 :=
by
  let completed_fraction := 1 - fraction_left
  have h1 : completed_fraction = 4 * together_rate
  have : d = 20
  sorry

end q_days_work_l753_753880


namespace increasing_on_real_iff_a_range_l753_753336

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a / x

theorem increasing_on_real_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ -3 ≤ a ∧ a ≤ -2 := 
by
  sorry

end increasing_on_real_iff_a_range_l753_753336


namespace bmw_cars_sold_l753_753186

def percentage_non_bmw (ford_pct nissan_pct chevrolet_pct : ℕ) : ℕ :=
  ford_pct + nissan_pct + chevrolet_pct

def percentage_bmw (total_pct non_bmw_pct : ℕ) : ℕ :=
  total_pct - non_bmw_pct

def number_of_bmws (total_cars bmw_pct : ℕ) : ℕ :=
  (total_cars * bmw_pct) / 100

theorem bmw_cars_sold (total_cars ford_pct nissan_pct chevrolet_pct : ℕ)
  (h_total_cars : total_cars = 300)
  (h_ford_pct : ford_pct = 20)
  (h_nissan_pct : nissan_pct = 25)
  (h_chevrolet_pct : chevrolet_pct = 10) :
  number_of_bmws total_cars (percentage_bmw 100 (percentage_non_bmw ford_pct nissan_pct chevrolet_pct)) = 135 := by
  sorry

end bmw_cars_sold_l753_753186


namespace team_C_has_most_uniform_height_l753_753500

theorem team_C_has_most_uniform_height
  (S_A S_B S_C S_D : ℝ)
  (h_A : S_A = 0.13)
  (h_B : S_B = 0.11)
  (h_C : S_C = 0.09)
  (h_D : S_D = 0.15)
  (h_same_num_members : ∀ (a b c d : ℕ), a = b ∧ b = c ∧ c = d) 
  : S_C = min S_A (min S_B (min S_C S_D)) :=
by
  sorry

end team_C_has_most_uniform_height_l753_753500


namespace constant_term_of_expression_is_60_l753_753818

-- Conditions: Define the given expression.
def given_expression : (ℚ → ℚ) := λ x, (2 / x - Real.sqrt x) ^ 6

-- The goal is to prove that the constant term in the expansion of the given expression is 60.
theorem constant_term_of_expression_is_60 :
  (∃ c : ℚ, ∀ x : ℚ, x ≠ 0 → x ≠ 1 → given_expression x = c) → c = 60 :=
by
  sorry

end constant_term_of_expression_is_60_l753_753818


namespace triangle_inequality_angle_side_l753_753455

theorem triangle_inequality_angle_side (A B C : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C]
  (d : A → B → C → ℝ)
  (h : ∀ X Y Z : A, d X Y + d Y Z > d X Z)
  (triangle_ABC : ∀ x y : A, ∃ z, y ≠ z)
  (angle : A → B → C → ℝ) :
  (angle A B C < angle B A C) ↔ (d A C < d B C) :=
sorry

end triangle_inequality_angle_side_l753_753455


namespace sum_of_valid_n_l753_753257

theorem sum_of_valid_n :
  (∑ n in ({n | nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15}.to_finset), n) = 28 :=
by
  sorry

end sum_of_valid_n_l753_753257


namespace inequality_41_42_equality_holds_237_l753_753453

-- Definitions based on the given conditions.
theorem inequality_41_42 (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / a + 1 / b + 1 / c < 1) : 1 / a + 1 / b + 1 / c ≤ 41 / 42 :=
sorry

-- Proving the case for specific values where equality holds.
theorem equality_holds_237 : 1 / 2 + 1 / 3 + 1 / 7 = 41 / 42 :=
begin
  norm_num,
end

end inequality_41_42_equality_holds_237_l753_753453


namespace prove_a_eq_1_l753_753771

noncomputable def proofStatement : Prop :=
  ∀ (a b c d : ℕ) (k m : ℕ),
    odd a → odd b → odd c → odd d →
    0 < a → a < b → b < c → c < d →
    a * d = b * c →
    a + d = 2^k →
    b + c = 2^m →
    a = 1

theorem prove_a_eq_1 : proofStatement :=
  sorry

end prove_a_eq_1_l753_753771


namespace case_a_black_wins_case_a_white_wins_case_b_white_wins_case_c_white_wins_case_d_example_l753_753490

-- Definitions for the game board and rules need to be in place

def board (n : ℕ) : Type := Fin n

structure GameState :=
(piece_count_W : ℕ)
(piece_count_B : ℕ)
(board_sz : ℕ)

-- Define the game for case (a)
noncomputable def player_one_piece (N : ℕ) (h : N > 2) : GameState := { 
  piece_count_W := 1, 
  piece_count_B := 1, 
  board_sz := N 
}

theorem case_a_black_wins : ∀ (N : ℕ), N = 3 → let state := player_one_piece N (by sorry) in ¬ (White_wins state) :=
by sorry

theorem case_a_white_wins : ∀ (N : ℕ), N > 3 → let state := player_one_piece N (by sorry) in White_wins state :=
by sorry

-- Define the game for case (b)
noncomputable def player_two_pieces (N : ℕ) (h : N > 4) : GameState := { 
  piece_count_W := 2, 
  piece_count_B := 2, 
  board_sz := N 
}

theorem case_b_white_wins : ∀ (N : ℕ), N > 4 → let state := player_two_pieces N (by sorry) in White_wins state :=
by sorry

-- Define the game for case (c)
noncomputable def player_three_pieces (N : ℕ) (h : N > 6) : GameState := { 
  piece_count_W := 3, 
  piece_count_B := 3, 
  board_sz := N 
}

theorem case_c_white_wins : ∀ (N : ℕ), N > 6 → let state := player_three_pieces N (by sorry) in White_wins state :=
by sorry

-- Define the game for case (d): One example if white has fewer pieces yet wins.
noncomputable def less_white_wins :
  GameState := {
    piece_count_W := 1, 
    piece_count_B := 2, 
    board_sz := 4 -- Example board size
  }

theorem case_d_example : White_wins less_white_wins :=
by sorry

end case_a_black_wins_case_a_white_wins_case_b_white_wins_case_c_white_wins_case_d_example_l753_753490


namespace problem1_problem2_l753_753341

-- Define set A
def setA (a x : ℝ) := x^2 - 2*x + 2*a - a^2 ≤ 0

-- Define set B
def setB (x : ℝ) := sin (π * x - π / 3) + sqrt 3 * cos (π * x - π / 3) = 0

-- Problem 1
theorem problem1 (a : ℝ) : 2 ∈ { x : ℝ | setA a x } → a ≤ 0 ∨ a ≥ 2 := sorry

-- Problem 2
theorem problem2 (a : ℝ) :
  (∃ s : set ℝ, s = { x : ℝ | setA a x } ∩ { x : ℝ | setB x } ∧ s.card = 3) →
  a ∈ { a : ℝ | -1 < a ∧ a ≤ -1/2 ∨ 5/2 ≤ a ∧ a < 2 ∨ a = 0 ∨ a = 2 } := sorry

end problem1_problem2_l753_753341


namespace cannot_fill_grid_to_form_arithmetic_sequences_l753_753023

theorem cannot_fill_grid_to_form_arithmetic_sequences :
  ¬ ∃ (grid : ℕ → ℕ → ℕ),
    (∀ i, grid i 3 = [1, 2, 3, 4, 5, 7].nth i.get_or_else 0) ∧
    (∀ i j, i < 6 ∧ j < 7 → ∃ d, (grid i j = grid i (j - 1) + d ∀ j > 0) ∧
    (∀ i j, j < 7 ∧ i < 6 → ∃ d, (grid i j = grid (i - 1) j + d ∀ i > 0)) :=
by
  sorry

end cannot_fill_grid_to_form_arithmetic_sequences_l753_753023


namespace num_arrangement_options_l753_753184

def competition_events := ["kicking shuttlecocks", "jumping rope", "tug-of-war", "pushing the train", "multi-person multi-foot"]

def is_valid_arrangement (arrangement : List String) : Prop :=
  arrangement.length = 5 ∧
  arrangement.getLast? = some "tug-of-war" ∧
  arrangement.get? 0 ≠ some "multi-person multi-foot"

noncomputable def count_valid_arrangements : ℕ :=
  let positions := ["kicking shuttlecocks", "jumping rope", "pushing the train"]
  3 * positions.permutations.length

theorem num_arrangement_options : count_valid_arrangements = 18 :=
by
  sorry

end num_arrangement_options_l753_753184


namespace pentagon_area_is_approx_15_26_l753_753590

noncomputable def pentagon_area (s : ℝ) : ℝ :=
  let p := (4 * s) / 5
  in (5 * p^2 * Real.tan (3 * Real.pi / 10)) / 4

theorem pentagon_area_is_approx_15_26 (s : ℝ) (hs : s^2 = 16) : 
  pentagon_area s ≈ 15.26 :=
by
  sorry

end pentagon_area_is_approx_15_26_l753_753590


namespace dot_product_range_property_l753_753663

def ellipse (a b : ℝ) := set_of (λ (p : ℝ × ℝ), (p.fst / a) ^ 2 + (p.snd / b) ^ 2 = 1)

noncomputable def foci_distance (a b : ℝ) := 2 * Real.sqrt (a ^ 2 - b ^ 2)

noncomputable def dot_product_range (a b : ℝ) : set ℝ :=
  let c := Real.sqrt (a ^ 2 - b ^ 2) in
  let max_el := a ^ 2 - c ^ 2 in
  let min_el := 2 * b ^ 2 - a ^ 2 in
  set.Icc min_el max_el

theorem dot_product_range_property :
  dot_product_range 9 4 = set.Icc (-49 : ℝ) (11 : ℝ) :=
by
  sorry

end dot_product_range_property_l753_753663


namespace largest_undefined_value_l753_753537

open Real

theorem largest_undefined_value :
  let expr_undefined (x : ℝ) := (4 * x^3 - 40 * x^2 + 36 * x - 8 = 0)
  ∃ x : ℝ, expr_undefined x ∧ ∀ y : ℝ, expr_undefined y → y ≤ x :=
begin
  use 4 + sqrt 15,
  split,
  { sorry }, -- Prove that 4 + sqrt 15 is a root.
  { intros y hy,
    sorry } -- Prove that 4 + sqrt 15 is the largest root.
end

end largest_undefined_value_l753_753537


namespace original_price_calculation_l753_753577

variable (P : ℝ)
variable (selling_price : ℝ := 1040)
variable (loss_percentage : ℝ := 20)

theorem original_price_calculation :
  P = 1300 :=
by
  have sell_percent := 100 - loss_percentage
  have SP_eq := selling_price = (sell_percent / 100) * P
  sorry

end original_price_calculation_l753_753577


namespace inradius_constant_l753_753662

-- Let Γ be a parabola
variables (Γ : Type) [parabola Γ]

-- Let A, B, C be points on Γ
variables (A B C : Γ)

-- Orthocenter H of triangle ABC coincides with the focus of Γ
variable (H : focus_of Γ)
variable (H_orthocenter : is_orthocenter H A B C)

-- H remains fixed as A, B, C move along the parabola Γ
variable (H_fixed : ∀ (A' B' C' : Γ), is_orthocenter H A' B' C' → A = A' → B = B' → C = C')

-- The inradius of the triangle ABC
noncomputable def inradius (A B C : Γ) : ℝ := sorry

-- Statement: Prove that the inradius of triangle ABC remains constant
theorem inradius_constant (A B C : Γ) (H : focus_of Γ) (H_orthocenter : is_orthocenter H A B C) 
  (H_fixed : ∀ (A' B' C' : Γ), is_orthocenter H A' B' C' → A = A' → B = B' → C = C') : 
  (inradius A B C = inradius A B C) :=
sorry

end inradius_constant_l753_753662


namespace B_work_time_alone_l753_753566

theorem B_work_time_alone
  (A_rate : ℝ := 1 / 8)
  (together_rate : ℝ := 3 / 16) :
  ∃ (B_days : ℝ), B_days = 16 :=
by
  sorry

end B_work_time_alone_l753_753566


namespace range_of_M_l753_753992

theorem range_of_M (x y : ℝ) (h : x^2 + x * y + y^2 - 2 = 0) :
  let M := x^2 - x * y + y^2 in
  2 / 3 ≤ M ∧ M ≤ 6 :=
sorry

end range_of_M_l753_753992


namespace find_point_Q_l753_753508

theorem find_point_Q :
  ∃ Q : ℝ × ℝ × ℝ, (∃ (x y z : ℝ), (x - 2)^2 + (y - 1)^2 + (z + 4)^2 = 
    (x - Q.1)^2 + (y - Q.2.1)^2 + (z - Q.2.2)^2) ∧ 
    (12 - 6y + 18z = 78) :=
  ∃ (a b c : ℝ), Q = (a, b, c) ∧
  a = 8 ∧ b = -2 ∧ c = 5 ∧
  (12x - 6y + 18z = 78)


sorry

end find_point_Q_l753_753508


namespace percent_decrease_l753_753879

theorem percent_decrease (original_price sale_price : ℝ) 
  (h_original: original_price = 100) 
  (h_sale: sale_price = 75) : 
  (original_price - sale_price) / original_price * 100 = 25 :=
by
  sorry

end percent_decrease_l753_753879


namespace average_even_between_11_and_27_l753_753164

theorem average_even_between_11_and_27 : 
  let nums := [12, 14, 16, 18, 20, 22, 24, 26] in
  let sum := nums.sum in
  let count := nums.length in
  sum / count = 19 :=
by
  let nums := [12, 14, 16, 18, 20, 22, 24, 26]
  have sum_def : nums.sum = 152 := 
    by decide
  have count_def : nums.length = 8 :=
    by decide
  rw [sum_def, count_def]
  have div_def : 152 / 8 = 19 := 
    by decide
  exact div_def

end average_even_between_11_and_27_l753_753164


namespace cement_mixture_total_weight_l753_753896

def cement_mixture_weight (sand water cement lime gravel additives total_weight : ℝ) : Prop :=
  sand = 0.25 * total_weight ∧
  water = 0.20 * total_weight ∧
  cement = 0.15 * total_weight ∧
  lime = 0.10 * total_weight ∧
  gravel = 12 ∧
  additives = 0.07 * total_weight

theorem cement_mixture_total_weight (W : ℝ) (h : cement_mixture_weight (0.25 * W) (0.20 * W) (0.15 * W) (0.10 * W) 12 (0.07 * W) W) :
  W = 66.67 :=
begin
  sorry
end

end cement_mixture_total_weight_l753_753896


namespace article_A_profit_percent_l753_753387

noncomputable def cost_price_A (x : ℝ) : ℝ := (5 / 8) * x
noncomputable def new_selling_price_A (x : ℝ) : ℝ := 0.9 * x
noncomputable def adjusted_selling_price_A (x : ℝ) : ℝ := 0.972 * x
def profit_percent (cost : ℝ) (selling : ℝ) : ℝ := ((selling - cost) / cost) * 100

theorem article_A_profit_percent
    (x y z : ℝ)
    (h1 : ¬(x = 0))
    (h2 : ¬(y = 0))
    (h3 : ¬(z = 0))
    (h4 : selling_price_A : ℝ := x)
    (h5 : selling_price_B : ℝ := y)
    (h6 : selling_price_C : ℝ := z)
    (h7 : cost_price_A x = (5 / 8) * x)
    (h8 : new_selling_price_A x = 0.9 * x)
    (h9 : adjusted_selling_price_A x = 0.972 * x) :
    profit_percent (cost_price_A x) (adjusted_selling_price_A x) = 55.52 :=
by
   sorry

end article_A_profit_percent_l753_753387


namespace inlet_rate_l753_753911

variable (Volume LeakTime InletTime : ℝ)

-- Assign given values for the problem
axiom h1 : Volume = 12960
axiom h2 : LeakTime = 9
axiom h3 : InletTime = 12

-- Define the leak rate L and the inlet rate R
def L (Volume LeakTime : ℝ) := Volume / LeakTime
def R (Volume LeakTime InletTime : ℝ) := L Volume LeakTime + Volume / InletTime

-- Prove that the inlet rate R satisfies the given answer
theorem inlet_rate (Volume LeakTime InletTime : ℝ) (h1: Volume = 12960) (h2: LeakTime = 9) (h3: InletTime = 12) :
  R Volume LeakTime InletTime = 2520 :=
by
  have l_value: L Volume LeakTime = 1440 := by
    rw [h1, h2, show 12960 / 9 = 1440, by norm_num]
  show R Volume LeakTime InletTime = 2520
  rw [R, l_value, h1, h3, show 12960 / 12 = 1080, by norm_num]
  exact l_value.symm ▸ (by norm_num : 1440 + 1080 = 2520 : ℝ)

end inlet_rate_l753_753911


namespace prime_gt_10_exists_m_n_l753_753435

theorem prime_gt_10_exists_m_n (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_10 : p > 10) :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m + n < p ∧ p ∣ (5^m * 7^n - 1) :=
by
  sorry

end prime_gt_10_exists_m_n_l753_753435


namespace Oo_remains_stationary_l753_753602

-- Definitions and conditions of the problem
structure Point :=
(x : ℝ)
(y : ℝ)

-- Fixed points A and B
def A : Point := {x := 0, y := 0}
def B : Point := {x := 1, y := 0}

-- Moving point O with coordinates (ox, oy)
variable (ox oy : ℝ)
def O : Point := {x := ox, y := oy}

-- Define points A' and B' with given conditions
def A' : Point := {x := ox, y := oy}
def B' : Point := {x := ox, y := oy}

-- Midpoint O' of A'B'
def O' : Point := {x := (A'.x + B'.x) / 2, y := (A'.y + B'.y) / 2}

-- Given conditions on angles and distances
axiom angle_OAA' : ∃ θ : ℝ, θ = 90
axiom angle_OBB' : ∃ θ : ℝ, θ = 90
axiom distance_AA' : ∀ (A O : Point), A'.x - A.x = O.x - A.x ∧ A'.y - A.y = O.y - A.y
axiom distance_BB' : ∀ (B O : Point), B'.x - B.x = O.x - B.x ∧ B'.y - B.y = O.y - B.y

-- Theorem to be proved
theorem Oo_remains_stationary :
  ∀ (ox oy : ℝ), O' = {x := (A'.x + B'.x) / 2, y := (A'.y + B'.y) / 2} :=
by
  sorry

end Oo_remains_stationary_l753_753602


namespace tank_full_volume_l753_753908

theorem tank_full_volume (x : ℝ) (h1 : 5 / 6 * x > 0) (h2 : 5 / 6 * x - 15 = 1 / 3 * x) : x = 30 :=
by
  -- The proof is omitted as per the requirement.
  sorry

end tank_full_volume_l753_753908


namespace number_of_possible_functions_k_eq_3_l753_753301

def f (k : ℕ) := {f : ℕ+ → ℕ+ // ∀ (n : ℕ+), n > k → f n = n - k ∧ n ≤ k → 1 ≤ f n ∧ f n ≤ 3}

theorem number_of_possible_functions_k_eq_3 (k : ℕ) (h_k : k = 3) :
  (∃ f : ℕ+ → ℕ+, (∀ n : ℕ+, n > 3 → f n = n - 3) ∧ (∀ n : ℕ+, n ≤ 3 → 1 ≤ f n ∧ f n ≤ 3)) →
  ∃ S, S.card = 27 :=
sorry

end number_of_possible_functions_k_eq_3_l753_753301


namespace average_of_set_l753_753877

variable (x : ℕ)

-- hypothesis x is a prime number
def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n
axiom h1 : is_prime x

-- hypothesis x - 1 is the median and thus the set in ascending order is {2x - 4, x - 1, 3x + 3}
axiom h2 : 2 * x - 4 ≤ x - 1 

theorem average_of_set (hx1 : is_prime x) (hx2 : 2 * x - 4 ≤ x - 1) (hx3 : x ≠ 1) :
  (x = 2) → (1 + 9 + 0) / 3 = 10 / 3 :=
by
  intro hx4
  have h : x = 2, from hx4
  have a : 1 + 9 + 0 = 10, from rfl
  have b : 10 / 3 = 10 / 3, from rfl
  exact b

end average_of_set_l753_753877


namespace pages_copied_l753_753481

theorem pages_copied (dollar_to_cents : ℕ) (cents_per_page : ℕ) (total_dollars : ℕ) :
  dollar_to_cents = 100 →
  cents_per_page = 3 →
  total_dollars = 15 →
  (total_dollars * dollar_to_cents) / cents_per_page = 500 :=
by 
  intros h1 h2 h3 
  rw[h1, h2, h3]
  norm_num
  sorry

end pages_copied_l753_753481


namespace only_solution_is_zero_l753_753616

theorem only_solution_is_zero (x : Real) :
  (real.exp (1/5 * real.log (x^3 + 2*x)) = 
   real.exp (1/3 * real.log (x^5 - 2*x))) → x = 0 := 
by 
  sorry

end only_solution_is_zero_l753_753616


namespace binomial_terms_decreasing_or_increasing_l753_753080

theorem binomial_terms_decreasing_or_increasing (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∀ k : ℕ, k < n → (a + b)^(n - k) * b^k > (a + b)^(n - (k + 1)) * b^(k + 1)) ↔ a > (n:ℝ) * b
  ∧ (∀ k : ℕ, k < n → (a + b)^(n - k) * b^k < (a + b)^(n - (k + 1)) * b^(k + 1)) ↔ a < b / (n:ℝ) :=
by
  sorry

end binomial_terms_decreasing_or_increasing_l753_753080


namespace field_trip_students_l753_753835

theorem field_trip_students (seats_per_bus : ℕ) (buses_needed : ℕ) (total_students : ℕ) 
  (h1 : seats_per_bus = 7) 
  (h2 : buses_needed = 4) 
  (h3 : total_students = seats_per_bus * buses_needed) : 
  total_students = 28 := 
by
  rw [h1, h2, h3]
  simpl
  rfl

end field_trip_students_l753_753835


namespace unit_digit_x2012_l753_753439

noncomputable def sequence : ℕ → ℕ
| 1       := 1
| (n + 1) := 4 * (sequence n) + ⌊real.sqrt 11 * (sequence n)⌋

theorem unit_digit_x2012 : (sequence 2012) % 10 = 3 :=
sorry

end unit_digit_x2012_l753_753439


namespace cos_2theta_value_l753_753978

open Real

theorem cos_2theta_value (θ : ℝ) 
  (h: sin (2 * θ) - 4 * sin (θ + π / 3) * sin (θ - π / 6) = sqrt 3 / 3) : 
  cos (2 * θ) = 1 / 3 :=
  sorry

end cos_2theta_value_l753_753978


namespace tangent_line_at_1_range_of_a_for_two_extreme_values_l753_753337

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + 2 / x

theorem tangent_line_at_1 :
  let a := 1 in
  let f := f 1 1 in
  let df := (1/x - 1 - 2/(x^2)) in
  2*x + y - 3 = 0 :=
sorry

theorem range_of_a_for_two_extreme_values
  (a : ℝ) :
  (∀ x > 0, x < ∞, ((ax^2 - x + 2) has_two_roots_in (0, +∞))) → (0 < a ∧ a < 1/8) :=
sorry

end tangent_line_at_1_range_of_a_for_two_extreme_values_l753_753337


namespace second_player_can_win_or_draw_l753_753145

/-- Given the rules of the game:
1. Two players take turns writing natural numbers on their halves of the board.
2. The total sum of numbers on the board does not exceed 10,000.
3. The game ends when the total sum of numbers equals 10,000.
4. The player with the smaller sum of digits on their half wins.
5. In case of a tie, the game is a draw.

Show that the second player can always secure at least a draw or win regardless of the opponent's moves. -/
theorem second_player_can_win_or_draw : 
  ∃ strategy : (ℕ → Prop), (∀ move : ℕ, strategy move) →
  (∀ moves : list ℕ, sum moves ≤ 10000 → ∃ result : (ℕ × ℕ), result.1 ≤ result.2) ∧ (result.1 < result.2 → false) :=
  sorry

end second_player_can_win_or_draw_l753_753145


namespace infinite_sequences_l753_753960

noncomputable def sequence1 (n : ℕ) : ℤ :=
  if n % 3 = 1 then 1 else -1

def no_three_consecutive_same (a : ℕ → ℤ) : Prop :=
  ∀ n, ¬(a n = a (n + 1) ∧ a (n + 1) = a (n + 2))

def satisfies_relation (a : ℕ → ℤ) : Prop :=
  ∀ m n, a (m * n) = a m * a n

theorem infinite_sequences
  (a : ℕ → ℤ)
  (h_range : ∀ n, a n = 1 ∨ a n = -1)
  (h_no_three_consecutive_same : no_three_consecutive_same a)
  (h_satisfies_relation : satisfies_relation a) :
  (∀ n, (a = sequence1 ∨ a = function_to_prove)) sorry
where function_to_prove : ℕ → ℤ := sorry

end infinite_sequences_l753_753960


namespace find_biology_marks_l753_753253

variable (english mathematics physics chemistry average_marks : ℕ)

theorem find_biology_marks
  (h_english : english = 86)
  (h_mathematics : mathematics = 85)
  (h_physics : physics = 92)
  (h_chemistry : chemistry = 87)
  (h_average_marks : average_marks = 89) : 
  (english + mathematics + physics + chemistry + (445 - (english + mathematics + physics + chemistry))) / 5 = average_marks :=
by
  sorry

end find_biology_marks_l753_753253


namespace binary_to_base5_l753_753247

theorem binary_to_base5 (n : ℕ) (h : n = 45) : nat_to_base 5 n = "140" :=
by {
  -- The proof would involve steps to show the conversion,
  -- but since the proof is not required, we use sorry.
  sorry
}

end binary_to_base5_l753_753247


namespace invest_problem_l753_753097

theorem invest_problem :
  let init := 200
  let O1 := init * 1.15
  let B1 := init * 0.7
  let Z1 := init
  let O2 := O1 * 0.9
  let B2 := B1 * 1.3
  let Z2 := Z1 in
  B2 < Z2 ∧ Z2 < O2 :=
by 
  let init := 200
  let O1 := init * 1.15
  let B1 := init * 0.7
  let Z1 := init
  let O2 := O1 * 0.9
  let B2 := B1 * 1.3
  let Z2 := Z1
  sorry 

end invest_problem_l753_753097


namespace simple_2n_gon_exists_l753_753170

theorem simple_2n_gon_exists
  (n : ℕ)
  (a b : Fin n → ℝ)
  (h1 : ∀ i, 0 < a i)
  (h2 : ∀ i, 0 < b i)
  (hne : ∃ i j, a i ≠ a j)
  (ha_eq_subsets : ∃ (s t : Finset (Fin n)), s ∩ t = ∅ ∧ s ∪ t = Finset.univ ∧ (∑ i in s, a i) = (∑ i in t, a i))
  (hb_eq_subsets : ∃ (s t : Finset (Fin n)), s ∩ t = ∅ ∧ s ∪ t = Finset.univ ∧ (∑ i in s, b i) = (∑ i in t, b i)) :
  ∃ (polygon : Fin (2 * n) → ℝ × ℝ),
    (∀ i : Fin (2 * n), (polygon i).1 ∈ Set.range a) ∧
    (∀ i : Fin (2 * n), (polygon i).2 ∈ Set.range b) ∧
    SimplePolygon polygon :=
sorry

end simple_2n_gon_exists_l753_753170


namespace jeff_total_run_is_290_l753_753738

variables (monday_to_wednesday_run : ℕ)
variables (thursday_run : ℕ)
variables (friday_run : ℕ)

def jeff_weekly_run_total : ℕ :=
  monday_to_wednesday_run + thursday_run + friday_run

theorem jeff_total_run_is_290 :
  (60 * 3) + (60 - 20) + (60 + 10) = 290 :=
by
  sorry

end jeff_total_run_is_290_l753_753738


namespace exists_parallel_segments_closed_broken_line_l753_753752

theorem exists_parallel_segments_closed_broken_line (n : ℕ) (P : Fin 2n → Fin 2n) (H : Function.Bijective P) :
  ∃ (i j k m : Fin 2n), i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ i ≠ m ∧ (i + m) % 2n = (j + k) % 2n :=
sorry

end exists_parallel_segments_closed_broken_line_l753_753752


namespace rectangle_triangle_area_ratio_l753_753154

theorem rectangle_triangle_area_ratio (L W θ : ℝ) (h : 0 < θ ∧ θ < π) :
  let A_rect := L * W in
  let h := W * Real.sin θ in
  let A_tri := (1 / 2) * L * h in
  (A_rect / A_tri) = 2 / Real.sin θ := by
  sorry

end rectangle_triangle_area_ratio_l753_753154


namespace f_decreasing_iff_f_lt_g_iff_l753_753681

-- Defining the first problem's function and condition
def f (a x : ℝ) : ℝ := ln x - a^2 * x^2 + a * x

-- The range of values for 'a' such that 'f(x)' is decreasing in [1, +∞)
theorem f_decreasing_iff (a : ℝ) (h : a ≠ 0) : 
  (∀ x ∈ (Set.Ici 1), f a x ≤ f a (x + 1)) ↔ (a ∈ Set.Iic (-1/2) ∪ Set.Ici 1) :=
sorry

-- Defining the second problem's functions and condition
def g (a x : ℝ) : ℝ := (3 * a + 1) * x - (a^2 + a) * x^2

-- The range of values for 'a' such that 'f(x) < g(x)' for all x > 1
theorem f_lt_g_iff (a : ℝ) (h : a ≠ 0) : 
  (∀ x > 1, f a x < g a x) ↔ (a ∈ Set.Ico (-1) 0) :=
sorry

end f_decreasing_iff_f_lt_g_iff_l753_753681


namespace no_valid_placement_l753_753046

def can_place_9_dominos (board : Fin 10 × Fin 10 → bool) : Prop :=
  ∃ dominos : Fin 9 → (Fin 10 × Fin 10) × (Fin 10 × Fin 10),
  (∀ i : Fin 9, board (dominos i).1 = tt ∧ board (dominos i).2 = tt) ∧
  (∀ r : Fin 10, odd (Finset.card {c | board (r, c)})) ∧
  (∀ c : Fin 10, odd (Finset.card {r | board (r, c)}))

theorem no_valid_placement : ¬ can_place_9_dominos (λ _, ff) :=
sorry

end no_valid_placement_l753_753046


namespace ice_cream_scoops_l753_753594

theorem ice_cream_scoops 
    (aaron_savings : ℕ) 
    (carson_savings : ℕ) 
    (bill_fraction : ℚ) 
    (remaining_change : ℕ) 
    (ice_cream_cost : ℚ)
    (total_savings := (aaron_savings + carson_savings : ℕ))
    (bill := bill_fraction * total_savings)
    (remaining_money := total_savings - bill)
    (total_change := 2 * remaining_change)
    (spent_on_ice_cream := remaining_money - total_change)
    (total_scoops := spent_on_ice_cream / ice_cream_cost)
    (scoops_each := total_scoops / 2) :
    aaron_savings = 40 →
    carson_savings = 40 →
    bill_fraction = (3 / 4 : ℚ) →
    remaining_change = 1 →
    ice_cream_cost = (3 / 2 : ℚ) →
    scoops_each = 6 :=
by {
  intro aaron_savings_eq,
  intro carson_savings_eq,
  intro bill_fraction_eq,
  intro remaining_change_eq,
  intro ice_cream_cost_eq,
  sorry
}

end ice_cream_scoops_l753_753594


namespace green_tea_cost_in_july_l753_753480

-- Defines the cost per pound of green tea and coffee in June
variable (C : ℝ)

-- Conditions for the cost in July
def coffee_price_july := 2 * C
def green_tea_price_july := 0.3 * C

-- Condition for the cost of the mixture in July
def mixture_cost := 3.45
def mixture_weight := 3

-- The cost calculation of the mixture in July
def total_cost := (1.5 * green_tea_price_july C) + (1.5 * coffee_price_july C)

-- The cost per pound of green tea in July
def green_tea_cost_july := 0.3 * C

theorem green_tea_cost_in_july :
  total_cost C = mixture_cost →
  green_tea_cost_july C = 0.3 :=
by
  sorry

end green_tea_cost_in_july_l753_753480


namespace simplify_expression_l753_753885

theorem simplify_expression (x y : ℝ) (hxy : x ≠ y) : 
  ((x - y) ^ 3 / (x - y) ^ 2) * (y - x) = -(x - y) ^ 2 := 
by
  sorry

end simplify_expression_l753_753885


namespace probability_valid_triplets_l753_753726

def is_within_distance (points : List (ℝ × ℝ)) (d : ℝ) : Prop :=
  ∀ p1 p2 ∈ points, dist p1 p2 ≤ d

def K := { p : ℝ × ℝ | ∃ x y, x ∈ {-1, 0, 1} ∧ y ∈ {-1, 0, 1} ∧ p = (x, y)}

noncomputable def count_valid_triplets : ℕ :=
  (finset.univ.powersetOfCard 3).filter (λ pts, is_within_distance (pts.toList : List (ℝ × ℝ)) 2)).card

noncomputable def total_triplets : ℕ :=
  (finset.univ.powersetOfCard 3).card

theorem probability_valid_triplets : 
  (count_valid_triplets : ℚ) / total_triplets = 5 / 14 := 
sorry

end probability_valid_triplets_l753_753726


namespace value_of_g_at_five_l753_753113

def g (x : ℕ) : ℕ := x^2 - 2 * x

theorem value_of_g_at_five : g 5 = 15 := by
  sorry

end value_of_g_at_five_l753_753113


namespace third_pipe_emptying_time_l753_753857

theorem third_pipe_emptying_time (h1 : (1 : ℝ) / 10) (h2 : (1 : ℝ) / 12)
  (combined_rate : (1 : ℝ) / (60 / 7)) : (1 : ℝ) / 15 = (11 / 60 - 7 / 60) := by
    -- detailed setup equations and solving would be here
    sorry

end third_pipe_emptying_time_l753_753857


namespace number_of_green_hats_l753_753863

theorem number_of_green_hats 
  (B G : ℕ) 
  (h1 : B + G = 85) 
  (h2 : 6 * B + 7 * G = 550) 
  : G = 40 :=
sorry

end number_of_green_hats_l753_753863


namespace number_of_committees_l753_753604

theorem number_of_committees (physics_men : Finset ℕ) (physics_women : Finset ℕ)
                             (chemistry_men : Finset ℕ) (chemistry_women : Finset ℕ)
                             (biology_men : Finset ℕ) (biology_women : Finset ℕ)
                             (h_pm : physics_men.card = 3) (h_pw : physics_women.card = 3)
                             (h_cm : chemistry_men.card = 3) (h_cw : chemistry_women.card = 3)
                             (h_bm : biology_men.card = 3) (h_bw : biology_women.card = 3) :
  (number_of_ways physics_men physics_women) * (number_of_ways chemistry_men chemistry_women) * (number_of_ways biology_men biology_women) +
  6 * (special_case_num physics_men physics_women chemistry_men chemistry_women biology_men biology_women) = 1215 :=
sorry

end number_of_committees_l753_753604


namespace part_I_part_II_l753_753680

noncomputable def f (x a : ℝ) : ℝ := abs (x + 1) + a * abs (2 * x - 1)

theorem part_I (m n : ℝ) (hmn : 0 < m ∧ 0 < n) : 
  (∀ x : ℝ, f x (1 / 2) ≥ 1 / m + 1 / n) → m + n ≥ 8 / 3 :=
sorry

theorem part_II :
  (∀ x ∈ Icc (-1 : ℝ) 2, f x a ≥ abs (x - 2)) → 1 ≤ a :=
sorry

end part_I_part_II_l753_753680


namespace percentage_of_hybrids_is_60_percentage_l753_753714

-- Definitions of the conditions and the problem in Lean 4 code
def total_cars : ℕ := 600
def hybrids_with_full_headlights : ℕ := 216

-- Given that 40% of hybrids have only one headlight
def hybrids_one_headlight_percentage : ℝ := 0.40

-- We need to prove that the percentage of hybrid cars is 60%
theorem percentage_of_hybrids_is_60_percentage (H : ℕ) (h1 : 0.60 * H = 216) : (H / total_cars : ℝ) * 100 = 60 :=
by sorry

end percentage_of_hybrids_is_60_percentage_l753_753714


namespace triangle_area_less_than_quarter_l753_753851

theorem triangle_area_less_than_quarter
  (polygon : Set.Points)
  (area_polygon : area polygon = 1)
  (line1 line2 line3 : Line)
  (bisect_line1 : bisects_area line1 polygon)
  (bisect_line2 : bisects_area line2 polygon)
  (bisect_line3 : bisects_area line3 polygon) :
  area (triangle line1 line2 line3) < 1 / 4 := 
sorry

end triangle_area_less_than_quarter_l753_753851


namespace quadratic_has_negative_root_condition_l753_753280

theorem quadratic_has_negative_root_condition (a : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, ax^2 + 2*x + 1 = 0 ∧ x < 0) ↔ (a < 0 ∨ (0 < a ∧ a ≤ 1)) :=
by
  sorry

end quadratic_has_negative_root_condition_l753_753280


namespace necessarily_positive_l753_753094

theorem necessarily_positive
  (a b c : ℝ)
  (h1 : 0 < a) (h2 : a < 2)
  (h3 : -2 < b) (h4 : b < 0)
  (h5 : 0 < c) (h6 : c < 3) :
  (b + b^2 > 0) ∧ (b + 3b^2 > 0) := 
  sorry

end necessarily_positive_l753_753094


namespace new_triangle_height_l753_753816

/-- The base of the original triangle is 15 cm. -/
def base_original : ℝ := 15

/-- The height of the original triangle is 12 cm. -/
def height_original : ℝ := 12

/-- The base of the new triangle is 20 cm. -/
def base_new : ℝ := 20

/-- The height of the new triangle is 18 cm, satisfying the condition of having double the area of the original triangle. -/
theorem new_triangle_height :
  let area_original := (base_original * height_original) / 2
  let area_new := 2 * area_original
  let height_new := 18
  (base_new * height_new) / 2 = area_new :=
by
  let area_original := (base_original * height_original) / 2
  let area_new := 2 * area_original
  let height_new := 18
  show (base_new * height_new) / 2 = area_new
  calc
    (base_new * height_new) / 2 = (20 * 18) / 2 : by rfl
    ... = 360 / 2              : by ring
    ... = 180                  : by norm_num
    ... = area_new             : by rfl

end new_triangle_height_l753_753816


namespace probability_same_color_probability_different_color_l753_753843

def count_combinations {α : Type*} (s : Finset α) (k : ℕ) : ℕ :=
  Nat.choose s.card k

noncomputable def count_ways_same_color : ℕ :=
  (count_combinations (Finset.range 3) 2) * 2

noncomputable def count_ways_diff_color : ℕ :=
  (Finset.range 3).card * (Finset.range 3).card

noncomputable def total_ways : ℕ :=
  count_combinations (Finset.range 6) 2

noncomputable def prob_same_color : ℚ :=
  count_ways_same_color / total_ways

noncomputable def prob_diff_color : ℚ :=
  count_ways_diff_color / total_ways

theorem probability_same_color :
  prob_same_color = 2 / 5 := by
  sorry

theorem probability_different_color :
  prob_diff_color = 3 / 5 := by
  sorry

end probability_same_color_probability_different_color_l753_753843


namespace barycentric_coordinates_of_point_X_l753_753451

theorem barycentric_coordinates_of_point_X (A B C X K L : Point) (hX_in_triangle : point_in_triangle X A B C)
  (line_through_X_parallel_AC : parallel (line_through_points X AC) AC)
  (line_through_X_parallel_BC : parallel (line_through_points X BC) BC)
  (X_K_intersects_AB : intersection (line_through_points X K) AB = some K)
  (X_L_intersects_AB : intersection (line_through_points X L) AB = some L) :
  barycentric_coordinates X A B C = (length BL / length AK : length AK / length LK : length LK / length BL) := sorry

end barycentric_coordinates_of_point_X_l753_753451


namespace spherical_to_rectangular_conversion_l753_753251

noncomputable def spherical_to_rectangular_coords (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * sin phi * cos theta, rho * sin phi * sin theta, rho * cos phi)

theorem spherical_to_rectangular_conversion {ρ θ φ : ℝ} (hρ : ρ = 3) (hθ : θ = 5 * Real.pi / 12) (hφ : φ = Real.pi / 6) :
  spherical_to_rectangular_coords ρ θ φ = (3 * (Real.sqrt 6 + Real.sqrt 2) / 8, 3 * (Real.sqrt 6 - Real.sqrt 2) / 8, 3 * Real.sqrt 3 / 2) :=
by
  rw [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_conversion_l753_753251


namespace y_in_terms_of_x_l753_753299

theorem y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 4) : y = 4 - 3 * x := 
by
  sorry

end y_in_terms_of_x_l753_753299


namespace sum_of_undefined_values_l753_753542

theorem sum_of_undefined_values : 
  let roots := [3, 5] in 
  list.sum roots = 8 := by
-- sorry is added to skip the proof as per instructions, in a real scenario the detailed proof would be provided.
sorry

end sum_of_undefined_values_l753_753542


namespace maria_travel_fraction_before_first_stop_l753_753957

theorem maria_travel_fraction_before_first_stop (D : ℕ) (x : ℚ) :
  D = 480 ∧ 
  (1 - 4 * x) * 480 / 4 + x * 480 = 300 →
  x = 1 / 2 :=
by
  intros hD hx
  sorry

end maria_travel_fraction_before_first_stop_l753_753957


namespace tiles_reduce_to_one_l753_753206

-- Define the set of initial tiles from 1 to 144
def initialTiles := Finset.range 145

-- Define the condition for removing perfect squares
def removePerfectSquares (tiles : Finset ℕ) : Finset ℕ :=
  tiles.filter (λ n => ∀ m : ℕ, m * m ≠ n)

-- Define the function that counts the operations needed
noncomputable def countOperations : ℕ :=
  Nat.recOn 144 (λ n opCount =>
    if n = 1 then
      opCount
    else
      opCount + 1) 0

-- Prove that the number of operations to reduce tiles to one is 11
theorem tiles_reduce_to_one : countOperations = 11 := by
  sorry

end tiles_reduce_to_one_l753_753206


namespace no_such_function_exists_l753_753621

theorem no_such_function_exists :
  ¬∃ (f : ℝ → ℝ), (bounded (f)) ∧ (f 1 > 0) ∧ (∀ x y : ℝ, f(x)^2 + f(y)^2 ≤ 2 * f(x * y) + f(x + y)^2) := 
sorry

end no_such_function_exists_l753_753621


namespace remaining_amount_to_be_paid_is_1080_l753_753360

noncomputable def deposit : ℕ := 120
noncomputable def total_price : ℕ := 10 * deposit
noncomputable def remaining_amount : ℕ := total_price - deposit

theorem remaining_amount_to_be_paid_is_1080 :
  remaining_amount = 1080 :=
by
  sorry

end remaining_amount_to_be_paid_is_1080_l753_753360


namespace prob_palindrome_7_digit_l753_753390

def is_palindrome (n : ℕ) : Prop := 
  let digits := List.ofNats n
  digits = digits.reverse

def count_palindromes (total_digits : ℕ) : ℕ :=
  if total_digits = 7 then 
    let valid_first_digits := 9 -- digits 1 to 9
    let rest_digits := 10 -- digits 0 to 9
    valid_first_digits * rest_digits ^ 3
  else 0

def count_total_phones (total_digits : ℕ) : ℕ :=
  if total_digits = 7 then 
    let valid_first_digits := 9 -- digits 1 to 9
    let rest_digits := 10 -- digits 0 to 9
    valid_first_digits * rest_digits ^ 6
  else 0

theorem prob_palindrome_7_digit : 
  count_palindromes 7 / count_total_phones 7 = 0.001 := 
by 
  sorry

end prob_palindrome_7_digit_l753_753390


namespace trajectory_eq_no_such_m_l753_753081

noncomputable def point_A : (ℝ × ℝ) := (0, 1)
variable (m x y : ℝ) (h1 : m > 2)

def p := (x + m, y)
def q := (x - m, y)
def p_norm := Real.sqrt ((x + m)^2 + y^2)
def q_norm := Real.sqrt ((x - m)^2 + y^2)

-- Given conditions
axiom h2 : p_norm - q_norm = 4

theorem trajectory_eq :
  (∀ x y, (p_norm - q_norm) = 4 → (m > 2) → (x ≥ 2) → (x^2 / 4) - (y^2 / (m^2 - 4)) = 1) :=
sorry

def line_L (x : ℝ) : ℝ := (1/2) * x - 3

theorem no_such_m :
  (∀ x1 y1 x2 y2, 
  line_L x1 = y1 ∧ line_L x2 = y2 ∧
  ((x1^2 / 4) - (y1^2 / (m^2 - 4)) = 1) ∧
  ((x2^2 / 4) - (y2^2 / (m^2 - 4)) = 1) ∧
  ((x1, y1) ≠ (x2, y2)) →
  ∀ (m : ℝ), \(\overrightarrow{AB} \cdot \overrightarrow{AC} = \frac{9}{2}\) →
  false) :=
sorry

end trajectory_eq_no_such_m_l753_753081


namespace evaluate_floor_79_l753_753268

theorem evaluate_floor_79 : (⌊7.9⌋ = 7) :=
by
  sorry

end evaluate_floor_79_l753_753268


namespace remainder_of_b_mod_8_l753_753775

theorem remainder_of_b_mod_8 (m : ℕ) (hm : 0 < m) (b : ℕ)
  (hb : b ≡ (2 ^ (3 * m) + 5)⁻¹ [MOD 8]) : b ≡ 5 [MOD 8] :=
sorry

end remainder_of_b_mod_8_l753_753775


namespace mountain_height_interval_l753_753263

theorem mountain_height_interval (h : ℝ) :
  h < 8000 → h > 7900 → h > 7500 → h ∈ Ioo 7900 8000 :=
by
  intro h_lt_8000 h_gt_7900 h_gt_7500
  sorry

end mountain_height_interval_l753_753263


namespace circle_B_radius_l753_753850

-- Define the conditions
def radius_of_circle_B (diameter_A : ℝ) (ratio : ℝ) : ℝ :=
  let radius_A := diameter_A / 2 in
  radius_A / ratio

-- The problem states:
-- 1. The diameter of circle A is 80.
-- 2. The radius of circle A is 4 times the radius of circle B.

theorem circle_B_radius (d_A : ℝ) (r : ℝ) (h1 : d_A = 80) (h2 : r = 4) : 
  radius_of_circle_B d_A r = 10 := by
  sorry

end circle_B_radius_l753_753850


namespace negation_example_l753_753158

theorem negation_example :
  (¬ ∀ x y : ℝ, |x + y| > 3) ↔ (∃ x y : ℝ, |x + y| ≤ 3) :=
by
  sorry

end negation_example_l753_753158


namespace water_added_16_l753_753569

theorem water_added_16 (W : ℝ) 
  (h1 : ∃ W, 24 * 0.90 = 0.54 * (24 + W)) : 
  W = 16 := 
by {
  sorry
}

end water_added_16_l753_753569


namespace muffins_division_l753_753742

theorem muffins_division (total_muffins total_people muffins_per_person : ℕ) 
  (h1 : total_muffins = 20) (h2 : total_people = 5) (h3 : muffins_per_person = total_muffins / total_people) : 
  muffins_per_person = 4 := 
by
  sorry

end muffins_division_l753_753742


namespace multiplication_solution_l753_753637

theorem multiplication_solution 
  (x : ℤ) 
  (h : 72517 * x = 724807415) : 
  x = 9999 := 
sorry

end multiplication_solution_l753_753637


namespace vec_OB_dot_vec_OC_eq_negative_five_l753_753652

theorem vec_OB_dot_vec_OC_eq_negative_five :
  ∀ (x1 x2 y1 y2 : ℝ), 
  let A := (x1, y1),
      B := (x2, y2),
      C := (-x1, y1),
      OB := (x2, y2),
      OC := (-x1, y1) in
  (x1 + x2 = 2 + 4 / k^2) →
  (x1 * x2 = 1) →
  (y1 * y2 = -4) →
  OB.1 * OC.1 + OB.2 * OC.2 = -5 :=
by
  sorry  -- Proof to be filled in

end vec_OB_dot_vec_OC_eq_negative_five_l753_753652


namespace sequences_convergent_same_limit_l753_753415

-- Definitions and conditions
variable {t : ℕ → ℝ}
variable {x y : ℕ → ℝ}
variable {x₀ y₀ : ℝ}

-- Assume the sequence (t_n) is convergent and tends to a limit in (0,1)
axiom h1 : ∀ n, t n ∈ Set.Ioo 0 1
axiom h2 : ConvergentSequence t
axiom t_lim : ∃ l, l ∈ Set.Ioo 0 1 ∧ tendsto t atTop (𝓝 l)

-- Define the recurrence relations
def x (n : ℕ) : ℝ := 
if n = 0 then x₀ else
t (n - 1) * x (n - 1) + (1 - t (n - 1)) * y (n - 1)
  
def y (n : ℕ) : ℝ := 
if n = 0 then y₀ else
(1 - t (n - 1)) * x (n - 1) + t (n - 1) * y (n - 1)

-- Theorem statement proving convergence and equality of limits
theorem sequences_convergent_same_limit : 
  (convergent_sequence x) ∧ (convergent_sequence y) ∧ (SeqLimit x = SeqLimit y) := 
sorry

end sequences_convergent_same_limit_l753_753415


namespace john_bought_more_than_ray_l753_753812

variable (R_c R_d M_c M_d J_c J_d : ℕ)

-- Define the conditions
def conditions : Prop :=
  (R_c = 10) ∧
  (R_d = 3) ∧
  (M_c = R_c + 6) ∧
  (M_d = R_d + 1) ∧
  (J_c = M_c + 5) ∧
  (J_d = M_d + 2)

-- Define the question
def john_more_chickens_and_ducks (J_c R_c J_d R_d : ℕ) : ℕ :=
  (J_c - R_c) + (J_d - R_d)

-- The proof problem statement
theorem john_bought_more_than_ray :
  conditions R_c R_d M_c M_d J_c J_d → john_more_chickens_and_ducks J_c R_c J_d R_d = 14 :=
by
  intro h
  sorry

end john_bought_more_than_ray_l753_753812


namespace sum_of_max_values_l753_753682

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - Real.cos x)

theorem sum_of_max_values : (f π + f (3 * π)) = (Real.exp π + Real.exp (3 * π)) := 
by sorry

end sum_of_max_values_l753_753682


namespace min_socks_to_guarantee_pairs_l753_753187

@[simp] -- indicates it's a simple theorem
theorem min_socks_to_guarantee_pairs (red_sock green_sock blue_sock purple_sock orange_sock : ℕ)
  (h_red : red_sock = 150) (h_green : green_sock = 120) (h_blue : blue_sock = 100)
  (h_purple : purple_sock = 80) (h_orange : orange_sock = 50) :
  ∃ n, n = 36 ∧ ∀ selected_socks, selected_socks.length = n →
  ∃ pairs, pairs.length ≥ 15 :=
by 
  sorry

end min_socks_to_guarantee_pairs_l753_753187


namespace convex_polygon_mean_side_lt_mean_diagonal_l753_753457

theorem convex_polygon_mean_side_lt_mean_diagonal 
  (n : ℕ) (n ≥ 3) 
  (is_convex : ∀ (A : Fin n → ℝ × ℝ), convex_hull (range A) = t (A ∈ (Finset.univ : Finset (Fin n)) → A))
  (P : ℝ) (D : ℝ) : 
  (P / n : ℝ) < (2 * D / (n * (n - 3)) : ℝ) :=
  by sorry

end convex_polygon_mean_side_lt_mean_diagonal_l753_753457


namespace lower_right_corner_is_3_l753_753303

theorem lower_right_corner_is_3 :
  ∃ (grid : Matrix ℕ 3 3), 
    (∀ i, i < 3 → ∀ j, j < 3 → 1 ≤ grid i j ∧ grid i j ≤ 3) ∧
    (∀ i, (finset.range 3).card = finset.card (finset.image (λ j, grid i j) (finset.range 3))) ∧
    (∀ j, (finset.range 3).card = finset.card (finset.image (λ i, grid i j) (finset.range 3))) ∧
    grid 0 0 = 1 ∧ grid 0 2 = 3 ∧ grid 1 0 = 3 ∧ grid 1 1 = 2 ∧
    grid 2 2 = 3 :=
by {
  -- we provide such a grid as follows
  use ![[1, 2, 3], [3, 2, 1], [2, 1, 3]];
  sorry
}

end lower_right_corner_is_3_l753_753303


namespace sector_angle_l753_753327

theorem sector_angle (R : ℝ) (S : ℝ) (α : ℝ) (hR : R = 2) (hS : S = 8) : 
  α = 4 := by
  sorry

end sector_angle_l753_753327


namespace find_slope_of_line_passing_through_focus_l753_753999

open Real

variables {a b e : ℝ} {x y : ℝ}

def ellipse (x y a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (a b e : ℝ) := e = (sqrt (a^2 - b^2)) / a
def right_focus {a b : ℝ} : ℝ × ℝ := (a * sqrt (1 - (b^2 / a^2)), 0)
def passes_through_focus (l : ℝ → ℝ → Prop) (F : ℝ × ℝ) := l F.1 F.2
def vector_relation (A F B : ℝ × ℝ) := (A.1 - F.1, A.2 - F.2) = (3 * (F.1 - B.1), 3 * (F.2 - B.2))
def is_slope (l : ℝ → ℝ → Prop) (m : ℝ) := ∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l x₂ y₂ → x₁ ≠ x₂ → m = (y₂ - y₁) / (x₂ - x₁)

theorem find_slope_of_line_passing_through_focus
  (hyp_ellipse : ellipse x y a b) 
  (hyp_eccentricity : eccentricity e)
  (hyp_passes_through_focus : passes_through_focus l (right_focus)) 
  (hyp_vector_relation : ∃ A B, vector_relation A (right_focus) B ∧ A.2 > 0 ∧ l A.1 A.2 ∧ l B.1 B.2) :
  ∃ m, is_slope l m ∧ m = -sqrt 2 :=
sorry

end find_slope_of_line_passing_through_focus_l753_753999


namespace area_of_triangle_KLM_l753_753494

-- Define the right triangle ABC with legs AC and BC.
def right_triangle_AC_BC (AC BC : ℝ) (h : 90 = 90) : Prop :=
AC = 4 ∧ BC = 3

-- Define the hypotenuse AB using the Pythagorean theorem.
def hypotenuse_AB (AC BC : ℝ) : ℝ :=
Real.sqrt (AC^2 + BC^2)

-- Define the semi-perimeter of the triangle ABC.
def semiperimeter (AC BC AB : ℝ) : ℝ :=
(AC + BC + AB) / 2

-- Define the points of tangency K, L, and M.
def tangency_points (s AC BC : ℝ) : ℝ × ℝ :=
(s - AC, s - BC)

-- Define the area of the triangle ABC.
def area_triangle_ABC (AC BC : ℝ) : ℝ :=
(1 / 2) * AC * BC

-- Define the radius of the inscribed circle.
def incircle_radius (A s : ℝ) :=
A / s

-- Define the area of the triangle KLM.
def area_triangle_KLM (CK CL : ℝ) : ℝ :=
(1 / 2) * CK * CL

-- Final theorem statement.
theorem area_of_triangle_KLM (AC BC AB s CK CL : ℝ) 
  (h₁ : right_triangle_AC_BC AC BC h)
  (AB_val : AB = hypotenuse_AB AC BC)
  (s_val : s = semiperimeter AC BC AB)
  (CK_val : CK = s - AC)
  (CL_val : CL = s - BC)
  (area_ABC : area_triangle_ABC AC BC = 6)
  (radius : incircle_radius 6 s = 1)
  : area_triangle_KLM CK CL = 6/5 :=
sorry

end area_of_triangle_KLM_l753_753494


namespace hyperbola_equation_l753_753305

variable (a b : ℝ)
variable (c : ℝ) (h1 : c = 4)
variable (h2 : b / a = Real.sqrt 3)
variable (h3 : a ^ 2 + b ^ 2 = c ^ 2)

theorem hyperbola_equation : (a ^ 2 = 4) ∧ (b ^ 2 = 12) ↔ (∀ x y : ℝ, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1 → (x ^ 2 / 4) - (y ^ 2 / 12) = 1) := by
  sorry

end hyperbola_equation_l753_753305


namespace num_valid_integers_l753_753351

def digits_gt_4 (n : ℕ) (d : ℕ) : Prop := 
  n ≥ 100 ∧ n < 1000 ∧ (n / 100) > 4 ∧ ((n % 100) / 10) > 4 ∧ (n % 10) > 4

def divisible_by_6 (n : ℕ) : Prop := 
  n % 6 = 0

theorem num_valid_integers : {n : ℕ | digits_gt_4 n ∧ divisible_by_6 n}.card = 16 := 
sorry

end num_valid_integers_l753_753351


namespace objects_in_sphere_radius_2_l753_753872

noncomputable def tetrahedron_edge_length : ℝ := 2 * real.sqrt 2

noncomputable def hexagonal_pyramid_base_edge_length : ℝ := 1
noncomputable def hexagonal_pyramid_height : ℝ := 3.8

noncomputable def cylinder_base_diameter : ℝ := 1.6
noncomputable def cylinder_height : ℝ := 3.6

noncomputable def frustum_upper_base_edge_length : ℝ := 1
noncomputable def frustum_lower_base_edge_length : ℝ := 2
noncomputable def frustum_height : ℝ := 3

theorem objects_in_sphere_radius_2 :
  (tetrahedron_edge_length = 2 * real.sqrt 2 → (option A)) ∧
  (hexagonal_pyramid_base_edge_length = 1 ∧ hexagonal_pyramid_height = 3.8 → ¬ (option B)) ∧
  (cylinder_base_diameter = 1.6 ∧ cylinder_height = 3.6 → (option C)) ∧
  (frustum_upper_base_edge_length = 1 ∧ frustum_lower_base_edge_length = 2 ∧ frustum_height = 3 → (option D)) :=
by 
  sorry -- Proof is omitted

end objects_in_sphere_radius_2_l753_753872


namespace john_travel_distance_l753_753412

variables (J : ℝ)

def Jill_distance : ℝ := J - 5
def Jim_distance : ℝ := (Jill_distance J) / 5

theorem john_travel_distance (h : Jim_distance J = 2) : J = 15 :=
by sorry

end john_travel_distance_l753_753412


namespace solveProblems_l753_753517

noncomputable def problem1 (a b : ℤ) : Prop :=
2 * a + b = 35 ∧ a + 3 * b = 30 → a = 15 ∧ b = 5

noncomputable def problem2 (x : ℤ) : Prop :=
(∃ n, n = 5 ∧ 955 ≤ 15 * x + 5 * (120 - x) ∧ 15 * x + 5 * (120 - x) ≤ 1000)

noncomputable def problem3 (x : ℤ) (W : ℤ) : Prop :=
(∃ m, m = 960 ∧ W = 10 * x + 600 ∧ W_min = m)

-- statement of the proof problem
theorem solveProblems :
    (problem1 a b) ∧
    (problem2 x) ∧
    (problem3 x W) :=
sorry

end solveProblems_l753_753517


namespace curve_transformation_min_value_l753_753340

theorem curve_transformation_min_value :
  (∀ M : ℝ × ℝ, let (x, y) := M in (x^2 / 9 + y^2 = 1) → (x + 2 * sqrt 3 * y >= -sqrt 21)) :=
begin
  sorry
end

end curve_transformation_min_value_l753_753340


namespace find_number_l753_753891

theorem find_number (x : ℝ) (h : 0.40 * x = 130 + 190) : x = 800 :=
by {
  -- The proof will go here
  sorry
}

end find_number_l753_753891


namespace ordered_pair_line_parametrization_l753_753191

theorem ordered_pair_line_parametrization :
  ∃ (s m : ℚ), (∀ t : ℚ, let x := s + 2 * t, y := -4 + m * t in y = 5 * x + 7) ∧ s = -11/5 ∧ m = 10 :=
by
  sorry

end ordered_pair_line_parametrization_l753_753191


namespace arithmetic_sequence_ratio_l753_753658

-- Definitions and conditions from the problem
variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
variable (h2 : ∀ n, T n = n * (b 1 + b n) / 2)
variable (h3 : ∀ n, S n / T n = (3 * n - 1) / (n + 3))

-- The theorem that will give us the required answer
theorem arithmetic_sequence_ratio : 
  (a 8) / (b 5 + b 11) = 11 / 9 := by 
  have h4 := h3 15
  sorry

end arithmetic_sequence_ratio_l753_753658


namespace condition_M_intersect_N_N_l753_753302

theorem condition_M_intersect_N_N (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + (y - a)^2 ≤ 1 → y ≥ x^2)) ↔ (a ≥ 5 / 4) :=
sorry

end condition_M_intersect_N_N_l753_753302


namespace find_b_l753_753374

-- Define the triangle sides and angle
variables {a b c : ℝ} {B : ℝ}

-- Define the constants and conditions given in the problem
def condition1 := 2 * b = a + c
def condition2 := B = Real.pi / 6
def condition3 := (1 / 2) * a * c * Real.sin B = 3 / 2

-- State the theorem we need to prove
theorem find_b 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3) : 
  b = 1 + Real.sqrt 3 :=
sorry

end find_b_l753_753374


namespace probability_to_form_computers_l753_753060

def letters_in_campus : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def letters_in_threads : Finset Char := {'T', 'H', 'R', 'E', 'A', 'D', 'S'}
def letters_in_glow : Finset Char := {'G', 'L', 'O', 'W'}
def letters_in_computers : Finset Char := {'C', 'O', 'M', 'P', 'U', 'T', 'E', 'R', 'S'}

noncomputable def probability_campus : ℚ := 1 / Nat.choose 6 3
noncomputable def probability_threads : ℚ := 1 / Nat.choose 7 5
noncomputable def probability_glow : ℚ := 1 / (Nat.choose 4 2 / Nat.choose 3 1)

noncomputable def overall_probability : ℚ :=
  probability_campus * probability_threads * probability_glow

theorem probability_to_form_computers :
  overall_probability = 1 / 840 := by
  sorry

end probability_to_form_computers_l753_753060


namespace gas_station_constant_l753_753108

structure GasStationData where
  amount : ℝ
  unit_price : ℝ
  price_per_yuan_per_liter : ℝ

theorem gas_station_constant (data : GasStationData) (h1 : data.amount = 116.64) (h2 : data.unit_price = 18) (h3 : data.price_per_yuan_per_liter = 6.48) : data.unit_price = 18 :=
sorry

end gas_station_constant_l753_753108


namespace circle_intersects_sin_graph_more_than_sixteen_times_l753_753219

-- Definitions of the circle and sine graph
def circle (h k r : ℝ) (x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2
def sin_graph (x y : ℝ) : Prop := y = Real.sin x

-- The theorem that we aim to prove
theorem circle_intersects_sin_graph_more_than_sixteen_times:
  ∃ h k r, (∀ y, -1 ≤ k ∧ k ≤ 1) → (∃ (x1 x2 ... x17 : ℝ), ∀ i < 17, (∃ y, circle h k r (xi) y ∧ sin_graph xi y)) :=
begin
  sorry
end

end circle_intersects_sin_graph_more_than_sixteen_times_l753_753219


namespace find_cost_find_num_plans_min_cost_l753_753519

-- Define variables and constants
variables (a b : ℕ) -- a: cost of type A, b: cost of type B
variables (x y : ℕ) -- x: amount of type A, y: amount of type B
variable (W : ℕ)   -- W: total cost

-- Given conditions as hypotheses
hypothesis h1 : 2 * a + b = 35
hypothesis h2 : a + 3 * b = 30
hypothesis h3 : x + y = 120
hypothesis h4 : 955 ≤ 15 * x + 5 * y
hypothesis h5 : 15 * x + 5 * y ≤ 1000

-- Part 1: Prove the cost of one item of type A and one item of type B
theorem find_cost : a = 15 ∧ b = 5 :=
by sorry

-- Part 2: Prove that there are 5 different purchasing plans
theorem find_num_plans : ∃ (x : ℕ), 36 ≤ x ∧ x ≤ 40 ∧
                         (∃ (y : ℕ), x + y = 120 ∧ 955 ≤ 15 * x + 5 * y ∧ 15 * x + 5 * y ≤ 1000) :=
by sorry

-- Part 3: Prove the minimum amount of money needed
theorem min_cost (x : ℕ) (hx : 36 ≤ x ∧ x ≤ 40) : W = 960 :=
by sorry

end find_cost_find_num_plans_min_cost_l753_753519


namespace find_c_l753_753491

noncomputable theory

-- Definitions based on conditions
def parabola (a b c y : ℝ) := a * y^2 + b * y + c
def vertex_x := 2
def vertex_y := -3
def pass_through_x := 0
def pass_through_y := -5

-- The theorem statement
theorem find_c (a b c : ℝ) (h1: ∃a b c, ∀y, parabola a b c y = a * (y + 3)^2 + 2)
  (h2: parabola a b c (-3) = vertex_x)
  (h3: parabola a b c (-5) = pass_through_x) : c = -5/2 :=
sorry

end find_c_l753_753491


namespace total_differential_1_total_differential_2_l753_753559

-- Problem 1: Total differential of z = 5x^2y^3
theorem total_differential_1 (x y dx dy : ℝ) :
  let z := 5 * x^2 * y^3
      dz := 10 * x * y^3 * dx + 15 * x^2 * y^2 * dy
  in dz = 5 * x * y^2 * (2 * y * dx + 3 * x * dy) :=
sorry

-- Problem 2: Total differential of z = arctg(x^2 + 3y)
theorem total_differential_2 (x y dx dy : ℝ) :
  let z := arctan (x^2 + 3 * y)
      dz := (2 * x * dx + 3 * dy) / (1 + (x^2 + 3 * y)^2)
  in dz = (2 * x * dx + 3 * dy) / (1 + (x^2 + 3 * y)^2) :=
sorry

end total_differential_1_total_differential_2_l753_753559


namespace find_n_value_l753_753732

theorem find_n_value (m n k : ℝ) (h1 : n = k / m) (h2 : m = k / 2) (h3 : k ≠ 0): n = 2 :=
sorry

end find_n_value_l753_753732


namespace xiao_ming_mother_gets_newspaper_l753_753553

/-- Xiao Ming's mother might return home from her morning exercise at some time between 7:00 and 8:00 AM,
while the mailman might deliver newspapers to Xiao Ming's house mailbox at some time between 7:30 and 8:30 AM.
Prove that the probability that Xiao Ming's mother will open the mailbox and get the newspaper when she returns home from her morning exercise is 1/4.
 -/
theorem xiao_ming_mother_gets_newspaper :
  (∃ (t1 t2 : ℝ), 7 ≤ t1 ∧ t1 ≤ 8 ∧ 7.5 ≤ t2 ∧ t2 ≤ 8.5) →
  (∀ (t1 t2 : ℝ), 7 ≤ t1 ∧ t1 ≤ 8 ∧ 7.5 ≤ t2 ∧ t2 ≤ 8.5 → (t1,t2).fst < 7.5 ∨ (t1,t2).fst ≥ 7.5 ∧ (t1,t2).fst ≤ 8 ∧ (t1,t2).snd < 8 ∨ (t1,t2).snd ≥ 8) →
  (t1, t2).fst ∈ set.Icc 7 8 ∧ (t1, t2).snd ∈ set.Icc 7.5 8.5 →
  (1 / 2 : ℝ) * (1 / 2 : ℝ) = (1 / 4 : ℝ) :=
by
  sorry

end xiao_ming_mother_gets_newspaper_l753_753553


namespace sum_of_divisors_of_3777_that_are_perfect_squares_and_prime_is_zero_l753_753635

-- Define a function to check if a number is both a perfect square and a prime
def is_perfect_square_prime (n : ℕ) : Prop :=
  nat.prime n ∧ ∃ k : ℕ, k * k = n

-- Define the Lean statement for the given math proof problem
theorem sum_of_divisors_of_3777_that_are_perfect_squares_and_prime_is_zero :
  (∑ d in divisors 3777, if is_perfect_square_prime d then d else 0) = 0 :=
by
  -- Proof is omitted
  sorry

end sum_of_divisors_of_3777_that_are_perfect_squares_and_prime_is_zero_l753_753635


namespace find_a_l753_753478

theorem find_a (a : ℝ) (h : 6 * a + 4 = 0) : a = -2 / 3 :=
by
  sorry

end find_a_l753_753478


namespace sum_of_roots_l753_753529

-- Define the quadratic equation
def quadratic_eqn (x : ℝ) : ℝ := 3*x^2 - 9*x + 6

-- Define a property to state the sum of the roots for the quadratic equation
theorem sum_of_roots : (roots : List ℝ) (h : quadratic_eqn.roots = roots) → roots.sum = 3 :=
by
  sorry

end sum_of_roots_l753_753529


namespace train_speed_km_per_hr_l753_753211

-- Definitions for the conditions
def length_of_train_meters : ℕ := 250
def time_to_cross_pole_seconds : ℕ := 10

-- Conversion factors
def meters_to_kilometers (m : ℕ) : ℚ := m / 1000
def seconds_to_hours (s : ℕ) : ℚ := s / 3600

-- Theorem stating that the speed of the train is 90 km/hr
theorem train_speed_km_per_hr : 
  meters_to_kilometers length_of_train_meters / seconds_to_hours time_to_cross_pole_seconds = 90 := 
by 
  -- We skip the actual proof with sorry
  sorry

end train_speed_km_per_hr_l753_753211


namespace max_three_m_plus_four_n_l753_753510

theorem max_three_m_plus_four_n (m n : ℕ) 
  (h : m * (m + 1) + n ^ 2 = 1987) : 3 * m + 4 * n ≤ 221 :=
sorry

end max_three_m_plus_four_n_l753_753510


namespace sum_of_single_digit_replacements_divisible_by_9_l753_753258

theorem sum_of_single_digit_replacements_divisible_by_9 : 
  (∀ z : ℕ, z < 10 ∧ (35 * 10000 + z * 1000 + 91) % 9 = 0 → z = 0 ∨ z = 9) → (∀ z : ℕ, z = 0 ∨ z = 9 → z).sum = 9 :=
by
  sorry

end sum_of_single_digit_replacements_divisible_by_9_l753_753258


namespace number_of_arrangements_l753_753645

-- Define the number of boys and girls
def boys : ℕ := 6
def girls : ℕ := 2

-- Define the total number of students to be selected
def total_selected : ℕ := 4

-- State the problem
theorem number_of_arrangements (B G T : ℕ) (H_boys : B = boys) (H_girls : G = girls) (H_total : T = total_selected) :
  ∃ P : ℕ, P = 240 :=
sorry

end number_of_arrangements_l753_753645


namespace certain_number_is_3_l753_753706

-- Given conditions
variables (z x : ℤ)
variable (k : ℤ)
variable (n : ℤ)

-- Conditions
-- Remainder when z is divided by 9 is 6
def is_remainder_6 (z : ℤ) := ∃ k : ℤ, z = 9 * k + 6
-- (z + x) / 9 is an integer
def is_integer_division (z x : ℤ) := ∃ m : ℤ, (z + x) / 9 = m

-- Proof to show that x must be 3
theorem certain_number_is_3 (z : ℤ) (h1 : is_remainder_6 z) (h2 : is_integer_division z x) : x = 3 :=
sorry

end certain_number_is_3_l753_753706


namespace find_central_angle_l753_753814

def area_of_sector (θ : ℝ) (r : ℝ) : ℝ := (θ / 360) * real.pi * r ^ 2

theorem find_central_angle (θ : ℝ) (hθ : θ = 42.048) (r : ℝ) (hr : r = 18) (A : ℝ) (hA : A = 118.8) :
  area_of_sector θ r = A :=
by
  sorry

end find_central_angle_l753_753814


namespace jill_llamas_count_l753_753057

theorem jill_llamas_count : 
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  herd_after_sell = 18 := 
by
  -- Definitions for the conditions
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  -- Proof will be filled in here.
  sorry

end jill_llamas_count_l753_753057


namespace poly_divisibility_implies_C_D_l753_753620

noncomputable def poly_condition : Prop :=
  ∃ (C D : ℤ), ∀ (α : ℂ), α^2 - α + 1 = 0 → α^103 + C * α^2 + D * α + 1 = 0

/- The translated proof problem -/
theorem poly_divisibility_implies_C_D (C D : ℤ) :
  (poly_condition) → (C = -1 ∧ D = 0) :=
by
  intro h
  sorry

end poly_divisibility_implies_C_D_l753_753620


namespace prob_palindrome_7_digit_l753_753391

def is_palindrome (n : ℕ) : Prop := 
  let digits := List.ofNats n
  digits = digits.reverse

def count_palindromes (total_digits : ℕ) : ℕ :=
  if total_digits = 7 then 
    let valid_first_digits := 9 -- digits 1 to 9
    let rest_digits := 10 -- digits 0 to 9
    valid_first_digits * rest_digits ^ 3
  else 0

def count_total_phones (total_digits : ℕ) : ℕ :=
  if total_digits = 7 then 
    let valid_first_digits := 9 -- digits 1 to 9
    let rest_digits := 10 -- digits 0 to 9
    valid_first_digits * rest_digits ^ 6
  else 0

theorem prob_palindrome_7_digit : 
  count_palindromes 7 / count_total_phones 7 = 0.001 := 
by 
  sorry

end prob_palindrome_7_digit_l753_753391


namespace compare_logarithms_l753_753760

noncomputable def a : ℝ := Real.log 3 / Real.log 4 -- log base 4 of 3
noncomputable def b : ℝ := Real.log 4 / Real.log 3 -- log base 3 of 4
noncomputable def c : ℝ := Real.log 3 / Real.log 5 -- log base 5 of 3

theorem compare_logarithms : b > a ∧ a > c := sorry

end compare_logarithms_l753_753760


namespace f_is_even_l753_753074

variables {α : Type*} [linear_ordered_field α]
def f (g : α → α) (x : α) : α :=
|g (x^3)|

def is_odd (g : α → α) : Prop :=
∀ x, g (-x) = -g (x)

theorem f_is_even {g : α → α} (h_odd : is_odd g) : ∀ x, f g (-x) = f g x :=
by
  intros
  dsimp [f]
  rw [neg_cube, h_odd]
  simp
  sorry

#check f_is_even

end f_is_even_l753_753074


namespace binomial_divisibility_by_prime_l753_753971

theorem binomial_divisibility_by_prime
  (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - ⌊ n / p ⌋) :=
by
  sorry

end binomial_divisibility_by_prime_l753_753971


namespace amy_lily_tie_probability_l753_753521

theorem amy_lily_tie_probability (P_Amy P_Lily : ℚ) (hAmy : P_Amy = 4/9) (hLily : P_Lily = 1/3) :
  1 - P_Amy - (↑P_Lily : ℚ) = 2 / 9 := by
  sorry

end amy_lily_tie_probability_l753_753521


namespace sin_A_and_area_of_triangle_l753_753019

theorem sin_A_and_area_of_triangle
  (A B C : ℝ)
  (AC : ℝ) 
  (sin_B : ℝ)
  (h1 : C - A = π / 2)
  (h2 : sin B = 1 / 3)
  (h3 : AC = sqrt 6)
  (sin_A : ℝ)
  (cos_A : ℝ)
  (h4 : sin A = sqrt 3 / 3)
  (h5 : cos A = sqrt 6 / 3):
  sin A = sqrt 3 / 3 ∧ 
  (1/2) * AC * (AC * (sqrt 2 * 3)) * cos A = 3 * sqrt 2 := 
by
  split
  · exact h4
  · sorry

end sin_A_and_area_of_triangle_l753_753019


namespace education_fund_growth_l753_753520

theorem education_fund_growth (x : ℝ) :
  2500 * (1 + x)^2 = 3600 :=
sorry

end education_fund_growth_l753_753520


namespace tan_theta_sub_pi_over_4_l753_753994

open Real

theorem tan_theta_sub_pi_over_4 (θ : ℝ) (h1 : -π / 2 < θ ∧ θ < 0) 
  (h2 : sin (θ + π / 4) = 3 / 5) : tan (θ - π / 4) = -4 / 3 :=
by
  sorry

end tan_theta_sub_pi_over_4_l753_753994


namespace information_spread_time_correct_l753_753514

noncomputable def information_spread_time (n : ℕ) : ℕ :=
  2 ^ (n + 1)

lemma spread_information_in_one_day (n : ℕ) (h : information_spread_time n ≥ 1000001) : n ≥ 19 :=
by {
  have h2 : 0.3 * (n + 1) ≥ 6 := sorry,
  linarith,
}

theorem information_spread_time_correct : ∃ n, information_spread_time n ≥ 1000001 ∧ n = 19 :=
by {
  use 19,
  split,
  { unfold information_spread_time, norm_num, },
  { refl, }
}

end information_spread_time_correct_l753_753514


namespace unfair_coin_probability_l753_753928

theorem unfair_coin_probability :
  let P := λ n : ℕ, (1 : ℝ) / 2 * (1 + (1 / 2) ^ n)
  let Q := λ n : ℕ, (1 : ℝ) / 2 * (1 - (1 / 2) ^ n)
  in (P 60) * (Q 60) = (1 / 4) * (1 - (1 / 4 ^ 60)) :=
by
  sorry

end unfair_coin_probability_l753_753928


namespace incorrect_reciprocal_quotient_l753_753874

-- Definitions based on problem conditions
def identity_property (x : ℚ) : x * 1 = x := by sorry
def division_property (a b : ℚ) (h : b ≠ 0) : a / b = 0 → a = 0 := by sorry
def additive_inverse_property (x : ℚ) : x * (-1) = -x := by sorry

-- Statement that needs to be proved
theorem incorrect_reciprocal_quotient (a b : ℚ) (h1 : a ≠ 0) (h2 : b = 1 / a) : a / b ≠ 1 :=
by sorry

end incorrect_reciprocal_quotient_l753_753874


namespace set_intersection_l753_753694

open Set

def M : Set ℝ := {x | ∃ y, y = real.logb 2 x}
def N : Set ℝ := {y | ∃ x, x > 1 ∧ y = (1 / 2) ^ x}

theorem set_intersection (hM : M = {x | x > 0})
  (hN : N = {y | y < 1 / 2}) :
  M ∩ N = {x | 0 < x ∧ x < 1 / 2} :=
by {
  sorry
}

end set_intersection_l753_753694


namespace net_effect_on_sale_value_l753_753368

theorem net_effect_on_sale_value
  (P N : ℝ)
  (h1 : 0 < P)
  (h2 : 0 < N) :
  let new_price1 := 0.90 * P,
      new_units1 := 1.85 * N,
      sale_value1 := new_price1 * new_units1,

      new_price2 := 0.90 * P * 0.85,
      new_units2 := 1.85 * N * 1.50,
      sale_value2 := new_price2 * new_units2,

      new_price3 := 0.90 * P * 0.85 * 0.80,
      new_units3 := 1.85 * N * 1.50 * 1.30,
      sale_value3 := new_price3 * new_units3,

      original_sale_value := P * N,
      final_sale_value := sale_value3,

      net_effect := final_sale_value - original_sale_value in

  net_effect / original_sale_value = 0.1987 :=
by
  sorry

end net_effect_on_sale_value_l753_753368


namespace braking_distance_at_40_braking_distance_relationship_no_rear_end_l753_753567

-- Establish the linear relationship between braking speed and braking distance.
def braking_distance (x : ℝ) : ℝ := 0.25 * x

-- Prove part (1): Braking distance when speed is 40 km/h
theorem braking_distance_at_40 :
  braking_distance 40 = 10 :=
by
  simp [braking_distance]
  norm_num
  sorry

-- Prove part (3): Relationship between y and x for all non-negative x
theorem braking_distance_relationship (x : ℝ) (h : x ≥ 0) :
  braking_distance x = 0.25 * x :=
by
  simp [braking_distance]
  exact rfl

-- Prove part (4): Given speed 120 km/h and stopping distance 33 m, show no rear-end collision
theorem no_rear_end (x : ℝ) (d : ℝ) (h_speed : x = 120) (h_distance : d = 33) :
  braking_distance x < d :=
by
  simp [braking_distance]
  norm_num
  linarith
  sorry

end braking_distance_at_40_braking_distance_relationship_no_rear_end_l753_753567


namespace Bran_remaining_payment_l753_753234

theorem Bran_remaining_payment :
  let tuition_fee : ℝ := 90
  let job_income_per_month : ℝ := 15
  let scholarship_percentage : ℝ := 0.30
  let months : ℕ := 3
  let scholarship_amount : ℝ := tuition_fee * scholarship_percentage
  let remaining_after_scholarship : ℝ := tuition_fee - scholarship_amount
  let total_job_income : ℝ := job_income_per_month * months
  let amount_to_pay : ℝ := remaining_after_scholarship - total_job_income
  amount_to_pay = 18 := sorry

end Bran_remaining_payment_l753_753234


namespace find_x_l753_753728

theorem find_x (x : ℚ) (h1 : 3 * x + (4 * x - 10) = 90) : x = 100 / 7 :=
by {
  sorry
}

end find_x_l753_753728


namespace mutually_exclusive_not_contradictory_l753_753262

-- Definitions of events
def person := {A, B, C}
def card := {red, black, white}

def event_A_gets_red (distribution : person → card) : Prop :=
  distribution A = red

def event_B_gets_red (distribution : person → card) : Prop :=
  distribution B = red

-- Theorem stating relationship between the events
theorem mutually_exclusive_not_contradictory :
  ∀ (distribution : person → card),
    (event_A_gets_red distribution ∧ event_B_gets_red distribution) = false ∧ 
    ¬ ((event_A_gets_red distribution = false) ∧ (event_B_gets_red distribution = false)) :=
sorry

end mutually_exclusive_not_contradictory_l753_753262


namespace days_of_week_with_equal_tuesdays_and_fridays_in_30_day_month_l753_753581

theorem days_of_week_with_equal_tuesdays_and_fridays_in_30_day_month : 
  ∃ (initial_days : Finset ℕ), initial_days.card = 4 ∧
  ∀ d ∈ initial_days, let days := [d, (d+1) % 7] in 
  (30 - 4 * (4 + (if days.index 2 < 2 then 1 else 0))) = 
  (30 - 4 * (4 + (if days.index 5 < 2 then 1 else 0))) :=
sorry

end days_of_week_with_equal_tuesdays_and_fridays_in_30_day_month_l753_753581


namespace stratified_sampling_count_probability_model_X_l753_753202

-- Define the initial conditions
def num_sophomores : ℕ := 9
def num_freshmen : ℕ := 6
def num_total_students : ℕ := num_sophomores + num_freshmen
def num_selected : ℕ := 5

-- Define the stratified sampling results
def num_selected_sophomores : ℕ := 3
def num_selected_freshmen : ℕ := 2

-- Define the actual problem statement
theorem stratified_sampling_count :
  num_selected_sophomores = num_selected * num_sophomores / num_total_students ∧
  num_selected_freshmen = num_selected * num_freshmen / num_total_students :=
sorry

-- Define the probability calculation for the second part of the problem
theorem probability_model_X :
  (1 - 1 / ₁₀.to_rat) = (9 / 10 : ℚ) :=
sorry

end stratified_sampling_count_probability_model_X_l753_753202


namespace triangle_area_range_of_AC_l753_753403

-- Given conditions of the triangle
variable (A B C D : Type)
variable [Triangle A B C]
variable [AngleBisector AD A B C]
variable (AD_len : AD.length = 2)
variable (BAC_angle : ∠BAC = 2 * Real.pi / 3)
variable (AB_len : AB.length = 3)
variable (BD_len : BD.length = 3)

-- Area calculation to be proved
theorem triangle_area {S : ℝ} : S = (9 * Real.sqrt 3) / 2 :=
by
  -- Proof is omitted with sorry.
  sorry

-- Range of AC to be proved
theorem range_of_AC {AC_len : ℝ} (h : 5 / 4 < AC_len) : true :=
by
  -- Proof is omitted with sorry.
  sorry

end triangle_area_range_of_AC_l753_753403


namespace cyclic_pentagon_is_regular_l753_753601

variable (A B C D E : Type) [CyclicPentagon A B C D E]

-- Conditions translated to Lean definitions
def condition_1 : Line AC ∥ Line DE := sorry
def condition_2 : Line BD ∥ Line AE := sorry
def condition_3 : Line CE ∥ Line AB := sorry
def condition_4 : Line DA ∥ Line BC := sorry
def condition_5 : Line EB ∥ Line CD := sorry

-- The theorem to prove, with conditions
theorem cyclic_pentagon_is_regular (ABCDE : CyclicPentagon A B C D E)
(h1 : Line AC ∥ Line DE)
(h2 : Line BD ∥ Line AE)
(h3 : Line CE ∥ Line AB)
(h4 : Line DA ∥ Line BC)
(h5 : Line EB ∥ Line CD) :
RegularPentagon A B C D E :=
by sorry

end cyclic_pentagon_is_regular_l753_753601


namespace decreasing_function_range_l753_753364

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x > 1 then a / x else (2 - a) * x + 3

theorem decreasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) → 2 < a ∧ a ≤ 5 / 2 :=
by
  sorry

end decreasing_function_range_l753_753364


namespace polynomial_x3_plus_x_is_good_l753_753065

open Polynomial

noncomputable def is_good_polynomial (P : Polynomial ℤ) : Prop :=
  ∃ᶠ q in Filter.atTop, Prime q ∧
    (∃ n_set : Finset ℕ, (n_set.card ≥ (q + 1) / 2) ∧
      ∀ n ∈ n_set, P.eval n % q ≠ 0)

theorem polynomial_x3_plus_x_is_good : is_good_polynomial (X ^ 3 + X) :=
sorry

end polynomial_x3_plus_x_is_good_l753_753065


namespace inverse_matrix_l753_753278

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, -1], ![2, 5]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5/22, 1/22], ![-1/11, 2/11]]

theorem inverse_matrix :
  A⁻¹ = A_inv :=
sorry

end inverse_matrix_l753_753278


namespace family_movie_outing_l753_753905

theorem family_movie_outing
  (regular_ticket_cost : ℕ)
  (adult_tickets : ℕ)
  (children_tickets : ℕ)
  (ticket_discount : ℕ)
  (change_received : ℕ)
  (num_adults : ℕ)
  (num_children : ℕ)
  (h_regular_ticket_cost : regular_ticket_cost = 9)
  (h_ticket_discount : ticket_discount = 2)
  (h_change_received : change_received = 1)
  (h_num_adults : num_adults = 2)
  (h_num_children : num_children = 3) :
  let cost_adult_tickets := num_adults * regular_ticket_cost,
      cost_children_tickets := num_children * (regular_ticket_cost - ticket_discount),
      total_cost := cost_adult_tickets + cost_children_tickets,
      amount_given := total_cost + change_received in
  amount_given = 40 :=
by
  sorry

end family_movie_outing_l753_753905


namespace correct_selection_l753_753579

theorem correct_selection : 
  ∀ (total_students : ℕ) (num_select : ℕ) (initial_num : ℕ) (range_start : ℕ) (range_end : ℕ),
    total_students = 800 →
    num_select = 50 →
    initial_num = 7 →
    range_start = 33 →
    range_end = 48 →
    ∃ (k : ℕ), range_start ≤ initial_num + k * (total_students / num_select) ∧ initial_num + k * (total_students / num_select) ≤ range_end ∧ initial_num + k * (total_students / num_select) = 39 :=
begin
  intros total_students num_select initial_num range_start range_end,
  intros h_total_students h_num_select h_initial_num h_range_start h_range_end,
  use 2,
  simp [h_total_students, h_num_select, h_initial_num, h_range_start, h_range_end],
  split,
  { exact nat.le_of_lt (by norm_num) },
  split,
  { exact nat.le_of_lt (by norm_num) },
  { exact rfl }
end

end correct_selection_l753_753579


namespace general_term_sum_first_n_terms_l753_753986

-- Definitions for conditions
def a1 (d a1 : ℝ) : Prop := a1 + (a1 + 2*d) + (a1 + 7*d) = 9
def a2 (d a1 : ℝ) : Prop := (a1 + d) + (a1 + 4*d) + (a1 + 10*d) = 21

-- General term of the arithmetic sequence
theorem general_term (d a1 : ℝ) (h1 : a1 d) (h2 : a2 d) : ∀ (n : ℕ), a_n = 1.5 * n - 2 := sorry

-- Definition for c_n
def c_n (a_n : ℝ -> ℝ) : ℕ -> ℝ := λ n, 2^(a_n n + 3)

-- Sum of the first n terms in the new sequence
theorem sum_first_n_terms (a_n : ℕ -> ℝ) (c_n : ℕ -> ℝ) (h_a_n : ∀ (n : ℕ), a_n = 1.5 * n - 2) 
  : ∀ (n : ℕ), S_n = (17 / 9) + ((6 * n - 17) / 9) * (4 ^ n) := sorry

end general_term_sum_first_n_terms_l753_753986


namespace spherical_coordinates_transformation_l753_753633

theorem spherical_coordinates_transformation 
  (x y z : ℝ)
  (hx : x = 3 * sin (π / 4) * cos (9 * π / 7)) 
  (hy : y = 3 * sin (π / 4) * sin (9 * π / 7))
  (hz : z = 3 * cos (π / 4)) : 
  ∃ ρ θ φ, 
    ρ = 3 ∧ θ = 5 * π / 7 ∧ φ = π / 4 ∧ 
    (-y, x, z) = (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ) :=
by 
  use [3, 5 * π / 7, π / 4]
  simp [hx, hy, hz]
  sorry -- proof omitted

end spherical_coordinates_transformation_l753_753633


namespace measure_angle_BDA_l753_753809

-- Define the conditions
variables {A B C D : Type}
variable [angle : A -> A -> ℝ]

-- Given conditions: AB = AC = AD, and ∠BAC = 30°
-- And we need to prove ∠BDA = 75°

-- Assumptions
axiom AB_eq_AC : AB = AC
axiom AC_eq_AD : AC = AD
axiom angle_BAC : angle B A C = 30

-- Goal: Prove that the measure of ∠BDA is 75°
theorem measure_angle_BDA : angle B D A = 75 := 
sorry

end measure_angle_BDA_l753_753809


namespace dot_product_eq_one_diff_dot_product_eq_neg_three_norm_diff_eq_sqrt_three_l753_753650

variables {a b : ℝ^3} -- Assuming a and b are vectors in 3-D space.

-- Given conditions
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 2
axiom angle_ab : real.angle a b = real.pi / 3 -- 60 degrees in radians.

-- To prove
theorem dot_product_eq_one : ∥a∥ = 1 ∧ ∥b∥ = 2 ∧ real.angle a b = real.pi / 3 → a ⬝ b = 1 :=
by sorry

theorem diff_dot_product_eq_neg_three : ∥a∥ = 1 ∧ ∥b∥ = 2 ∧ real.angle a b = real.pi / 3 → (a - b) ⬝ (a + b) = -3 :=
by sorry

theorem norm_diff_eq_sqrt_three : ∥a∥ = 1 ∧ ∥b∥ = 2 ∧ real.angle a b = real.pi / 3 → ∥a - b∥ = real.sqrt 3 :=
by sorry

end dot_product_eq_one_diff_dot_product_eq_neg_three_norm_diff_eq_sqrt_three_l753_753650


namespace simplify_expression_l753_753103

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem simplify_expression : ((a^(2/3) * b^(1/4))^2 * a^(-1/2) * b^(1/3)) / (a * b^5)^(1/6) = a^(2/3) :=
by sorry

end simplify_expression_l753_753103


namespace triangle_largest_angle_l753_753038

theorem triangle_largest_angle (x m n : ℝ) (h1 : 1.5 < x)
  (h2 : x < 4)
  (h3 : AB = x + 6)
  (h4 : AC = 4x)
  (h5 : BC = x + 12)
  (h6 : BC > AB)
  (h7 : BC > AC) :
  n - m = 2.5 :=
by
  sorry

end triangle_largest_angle_l753_753038


namespace range_of_f_max_AB_dot_AC_l753_753648

noncomputable def f (x : ℝ) : ℝ :=
  2 * sin (x / 2) * (sqrt 3 * cos (x / 2) - sin (x / 2)) + 1

theorem range_of_f : 
  ∀ x, x ∈ set.Icc (π / 6) (2 * π / 3) → f(x) ∈ set.Icc 1 2 :=
by
  sorry

variables {A AB AC : ℝ}

theorem max_AB_dot_AC (hA : f(A) = 2) (hBC : BC = 1) (h_triangle : 0 < A ∧ A < π) :
  ∃ m, ∀ (AB AC : ℝ) , m = AB * AC * cos A → m ≤ 1 / 2 :=
by
  sorry

end range_of_f_max_AB_dot_AC_l753_753648


namespace race_runners_l753_753198

theorem race_runners (k : ℕ) (h1 : 2*(k - 1) = k - 1) (h2 : 2*(2*(k + 9) - 12) = k + 9) : 3*k - 2 = 31 :=
by
  sorry

end race_runners_l753_753198


namespace polygon_sides_l753_753200

open Real

theorem polygon_sides (x : ℝ) (hx : 180 = x + (2 / 3) * x) : 
  let external_angle := (2 / 3) * x in 
  let n := 360 / external_angle in n = 5 := 
by
  -- Proof goes here
  sorry

end polygon_sides_l753_753200


namespace sales_tax_difference_l753_753607

theorem sales_tax_difference
  (price : ℝ)
  (rate1 rate2 : ℝ)
  (h_rate1 : rate1 = 0.075)
  (h_rate2 : rate2 = 0.07)
  (h_price : price = 30) :
  (price * rate1 - price * rate2 = 0.15) :=
by
  sorry

end sales_tax_difference_l753_753607


namespace sequence_formula_l753_753656

theorem sequence_formula (a : ℝ) (h : a > 0) : 
  ∀ (n : ℕ), n > 0 → (a_n n = (2^(n-1) * a) / (1 + (2^(n-1) - 1) * a)) :=
by 
-- Definition of the sequence
def a_n : ℕ → ℝ
| 1       := a
| (n + 1) := (2 * a_n n) / (1 + a_n n)

-- Base case
have base_case : a_n 1 = (2^0 * a) / (1 + (2^0 - 1) * a), from rfl,

-- Inductive step
have inductive_step : ∀ k, k ≥ 1 → 
  (a_n k = (2^(k-1) * a) / (1 + (2^(k-1) - 1) * a)) → 
  (a_n (k+1) = (2^k * a) / (1 + (2^k - 1) * a)), 
  by sorry,

-- Final proof by induction
exact nat.strong_induction_on n 
  (λ n ih, 
    match n with 
    | 0 := by linarith
    | 1 := base_case
    | k+1 := inductive_step (k+1) (by omega) (ih n (by omega))
    end)

end sequence_formula_l753_753656


namespace equation_one_solution_equation_two_solution_l753_753884

theorem equation_one_solution (x : ℝ) : 4 * (x - 1)^2 - 9 = 0 ↔ (x = 5 / 2) ∨ (x = - 1 / 2) := 
by sorry

theorem equation_two_solution (x : ℝ) : x^2 - 6 * x - 7 = 0 ↔ (x = 7) ∨ (x = - 1) :=
by sorry

end equation_one_solution_equation_two_solution_l753_753884


namespace students_belonging_to_other_communities_l753_753720

noncomputable def number_of_students : ℕ := 2500
noncomputable def percentage_muslims : ℝ := 28 / 100
noncomputable def percentage_hindus : ℝ := 26 / 100
noncomputable def percentage_sikhs : ℝ := 12 / 100
noncomputable def percentage_buddhists : ℝ := 10 / 100
noncomputable def percentage_christians : ℝ := 6 / 100
noncomputable def percentage_jews : ℝ := 4 / 100

theorem students_belonging_to_other_communities : 
  (14 / 100) * number_of_students = 350 :=
by
  have total_percentage_other_communities : (100 - (percentage_muslims + percentage_hindus + percentage_sikhs + percentage_buddhists + percentage_christians + percentage_jews)) / 100 = 14 / 100,
  { sorry },
  have total_students_other_communities_calc : (14 / 100) * number_of_students = 350,
  { sorry },
  exact total_students_other_communities_calc

end students_belonging_to_other_communities_l753_753720


namespace tan_passing_through_point_l753_753683

theorem tan_passing_through_point :
  (∃ ϕ : ℝ, (∀ x : ℝ, y = Real.tan (2 * x + ϕ)) ∧ (Real.tan (2 * (π / 12) + ϕ) = 0)) →
  ϕ = - (π / 6) :=
by
  sorry

end tan_passing_through_point_l753_753683


namespace solution_set_for_xf_positive_l753_753979

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_monotonically_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f (x) < f (y)

def condition_1 (f : ℝ → ℝ) : Prop :=
  is_odd_function f

def condition_2 (f : ℝ → ℝ) : Prop :=
  is_monotonically_increasing f (Set.Ioi 0)

def condition_3 (f : ℝ → ℝ) : Prop :=
  f 3 = 0

theorem solution_set_for_xf_positive (f : ℝ → ℝ) (x : ℝ)
  (h1 : condition_1 f)
  (h2 : condition_2 f)
  (h3 : condition_3 f) :
  (x f(x) > 0) ↔ (x ∈ (Set.Iio (-3)) ∪ (Set.Ioi 3)) := by
  sorry

end solution_set_for_xf_positive_l753_753979


namespace time_after_3577_minutes_l753_753153

-- Definitions
def startingTime : Nat := 6 * 60 -- 6:00 PM in minutes
def startDate : String := "2020-12-31"
def durationMinutes : Nat := 3577
def minutesInHour : Nat := 60
def hoursInDay : Nat := 24

-- Theorem to prove that 3577 minutes after 6:00 PM on December 31, 2020 is January 3 at 5:37 AM
theorem time_after_3577_minutes : 
  (durationMinutes + startingTime) % (hoursInDay * minutesInHour) = 5 * minutesInHour + 37 :=
  by
  sorry -- proof goes here

end time_after_3577_minutes_l753_753153


namespace number_of_roses_picked_later_l753_753576

-- Given definitions
def initial_roses : ℕ := 50
def sold_roses : ℕ := 15
def final_roses : ℕ := 56

-- Compute the number of roses left after selling.
def roses_left := initial_roses - sold_roses

-- Define the final goal: number of roses picked later.
def picked_roses_later := final_roses - roses_left

-- State the theorem
theorem number_of_roses_picked_later : picked_roses_later = 21 :=
by
  sorry

end number_of_roses_picked_later_l753_753576


namespace violet_gumdrop_count_l753_753910

noncomputable def total_gumdrops (white_count : ℕ) (white_percent : ℚ) : ℚ :=
  white_count / white_percent

noncomputable def initial_count (total : ℚ) (percent : ℚ) : ℚ :=
  total * percent

noncomputable def new_violet_count (initial_violet : ℚ) (third_purple : ℚ) : ℚ :=
  initial_violet + third_purple

theorem violet_gumdrop_count
    (white_count : ℕ) (white_percent : ℚ) (total : ℚ) (purple_percent : ℚ)
    (violet_percent : ℚ) (purple_divisor : ℚ)  : 
    total = 150 → 
    white_count = 45 →
    white_percent = 0.30 →
    purple_percent = 0.25 →
    violet_percent = 0.20 →
    initial_count 150 0.25 = 38 (some rounding explained) →
    initial_count 150 0.20 = 30 →
    purple_divisor = 3 →
    new_violet_count 30 (38/3) = 43 :=
by
  intros
  sorry

end violet_gumdrop_count_l753_753910


namespace average_temperature_week_l753_753098

theorem average_temperature_week 
  (T_sun : ℝ := 40)
  (T_mon : ℝ := 50)
  (T_tue : ℝ := 65)
  (T_wed : ℝ := 36)
  (T_thu : ℝ := 82)
  (T_fri : ℝ := 72)
  (T_sat : ℝ := 26) :
  (T_sun + T_mon + T_tue + T_wed + T_thu + T_fri + T_sat) / 7 = 53 :=
by
  sorry

end average_temperature_week_l753_753098


namespace find_CD_length_l753_753418

noncomputable def CD_length : ℝ :=
  let O₁_radius := 4
  let O₂_radius := 6
  let AB_length := 2
  let AM_length := AB_length / 2
  let OM_length := real.sqrt (O₁_radius ^ 2 - AM_length ^ 2)
  let CD := 2 * real.sqrt (O₂_radius ^ 2 - OM_length ^ 2)
  CD

theorem find_CD_length :
  let O₁_radius := 4
  let O₂_radius := 6
  let AB_length := 2
  let AM_length := AB_length / 2
  let OM_length := real.sqrt (O₁_radius ^ 2 - AM_length ^ 2)
  2 * real.sqrt (O₂_radius ^ 2 - OM_length ^ 2) = 2 * real.sqrt 21 :=
sorry

end find_CD_length_l753_753418


namespace maximum_value_is_2000_l753_753832

noncomputable def maximum_possible_value (x : Fin 2002 → ℝ) (h : ∑ k in Finset.range 2000, |x k - x (k+1)| = 2001) : ℝ :=
  let y (k : Fin 2001) := (1 / (k : ℝ)) * ∑ i in Finset.range k, x i
  ∑ k in Finset.range 2000, |y k - y (k+1)|

theorem maximum_value_is_2000 (x : Fin 2002 → ℝ) (h : ∑ k in Finset.range 2000, |x k - x (k+1)| = 2001) : 
  maximum_possible_value x h = 2000 :=
sorry

end maximum_value_is_2000_l753_753832


namespace length_of_segment_l753_753367

-- Define the curves and calculate the intersection points
def curve_parametric_x (t : ℝ) := t
def curve_parametric_y (t : ℝ) := t^2

-- Line l in Cartesian coordinates from its polar form
def line_l (x y : ℝ) := x - y = 2

-- Define the points of intersection
def point_A := (-1, 1)
def point_B := (2, 4)

-- Distance formula for the points A and B
def distance_AB (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  √((x2 - x1)^2 + (y2 - y1)^2)

-- Main theorem
theorem length_of_segment : distance_AB point_A point_B = 3 * √2 := by
  sorry

end length_of_segment_l753_753367


namespace has_four_solutions_l753_753699

lemma count_solutions_abs_eq (x : ℝ) : 
  (| | |x - 1 | - 1 | - 1 | = 1) → x = 4 ∨ x = -2 ∨ x = 2 ∨ x = 0 :=
by sorry

theorem has_four_solutions : 
  (∃ x : set ℝ, {x | | | |x - 1 | - 1 | - 1 | = 1}.card) = 4 :=
by sorry

end has_four_solutions_l753_753699


namespace movies_shown_eq_twenty_four_l753_753195

-- Define conditions
variables (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ)

-- Define the total number of movies calculation
noncomputable def total_movies_shown (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  screens * (open_hours / movie_duration)

-- Theorem to prove the total number of movies shown is 24
theorem movies_shown_eq_twenty_four : 
  total_movies_shown 6 8 2 = 24 :=
by
  sorry

end movies_shown_eq_twenty_four_l753_753195


namespace find_n_for_2013_in_expansion_l753_753447

/-- Define the pattern for the last term of the expansion of n^3 -/
def last_term (n : ℕ) : ℕ :=
  n^2 + n - 1

/-- The main problem statement -/
theorem find_n_for_2013_in_expansion :
  ∃ n : ℕ, last_term (n - 1) ≤ 2013 ∧ 2013 < last_term n ∧ n = 45 :=
by
  sorry

end find_n_for_2013_in_expansion_l753_753447


namespace avery_donation_l753_753229

theorem avery_donation (shirts pants shorts : ℕ)
  (h_shirts : shirts = 4)
  (h_pants : pants = 2 * shirts)
  (h_shorts : shorts = pants / 2) :
  shirts + pants + shorts = 16 := by
  sorry

end avery_donation_l753_753229


namespace problem_l753_753696

open Nat

theorem problem (m n : ℕ) (M : Finset ℕ) (N : Finset ℕ) 
  (hyp_M : M = {1, 2, 3, m}) 
  (hyp_N : N = {4, 7, n^4, n^2 + 3 * n})
  (hyp_f : ∀ x ∈ M, (3 * x + 1) ∈ N) : m - n = 3 := 
by 
  sorry

end problem_l753_753696


namespace measure_angle_ZXE_l753_753402

theorem measure_angle_ZXE (a b θ : ℝ) (a_b_pos : 0 < a ∧ 0 < b) (bisect : θ = b / 2)
  (angle_sum_triangle : a + b + (180° - (a + 2 * θ)) = 180°) (right_angle : ∠n = 90°) : 
  ∠ZXE = 90° - θ := 
by
  sorry

end measure_angle_ZXE_l753_753402


namespace sum_smallest_largest_l753_753815

theorem sum_smallest_largest (n : ℕ) (y : ℕ) (h : n % 2 = 1) : 
  let a := y - n + 1 in
  2 * a + 2 * (n - 1) = 2 * y :=
by
  sorry

end sum_smallest_largest_l753_753815


namespace minimum_value_of_f_l753_753678

noncomputable def f (x a : ℝ) : ℝ := x * exp (a * x - 1) - log x - a * x

theorem minimum_value_of_f (a : ℝ) (h : a ≤ -1 / (exp 2)) : 
  ∃ x : ℝ, ∀ y : ℝ, f x a ≤ f y a ∧ f x a = 0 :=
by
  sorry

end minimum_value_of_f_l753_753678


namespace cassie_dogs_l753_753238

theorem cassie_dogs (total_nails : ℕ) (num_parrots : ℕ) (extra_toe_claws : ℕ) (dog_nails_per_dog : ℕ) : 
  total_nails = 113 → num_parrots = 8 → extra_toe_claws = 7 → dog_nails_per_dog = 16 → 
  let parrot_claw_sum := 42 + 7 in -- 42 claws for 7 normal parrots, 7 claws for one special
  let dog_nail_sum := total_nails - parrot_claw_sum in -- nails to cut for dogs
  dog_nail_sum / dog_nails_per_dog = 4 :=
by 
  intros h1 h2 h3 h4
  let parrot_claw_sum := 42 + 7
  let dog_nail_sum := 113 - parrot_claw_sum
  show dog_nail_sum / 16 = 4
  rw [h1, h2, h3, h4]
  sorry

end cassie_dogs_l753_753238


namespace transaction_loss_per_year_l753_753584

-- Definitions based on the provided conditions
def principal : ℝ := 7000
def rate_lend : ℝ := 6 / 100
def rate_borrow : ℝ := 4 / 100
def time : ℕ := 2

-- Calculate interest
def simple_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ := (P * R * T).toReal

-- Interest earned from lending
def interest_earned : ℝ := simple_interest principal rate_lend time

-- Interest paid for borrowing
def interest_paid : ℝ := simple_interest principal rate_borrow time

-- Total gain in 2 years
def total_gain_2_years : ℝ := interest_earned - interest_paid

-- Gain per year
def gain_per_year : ℝ := total_gain_2_years / time.toReal

-- Proof statement
theorem transaction_loss_per_year :
  gain_per_year = -140 := by
  sorry

end transaction_loss_per_year_l753_753584


namespace trigonometric_series_differentiability_l753_753003

noncomputable def sequence (k : ℕ) : ℝ := (1 : ℝ) / (k ^ 4)

def differentiated_series_converges (s : ℕ) : Prop :=
  series_converges (λ k, (k ^ s) * sequence k)

-- The main statement we're going to prove
theorem trigonometric_series_differentiability :
  ∃ n : ℕ, n = 2 ∧ ∀ s < 3, differentiated_series_converges s := sorry

end trigonometric_series_differentiability_l753_753003


namespace binary_to_base5_l753_753250

theorem binary_to_base5 (n : ℕ) (h1 : n = 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0)
  (h2 : nat.div_mod n 5 = (9, 0)) (h3 : nat.div_mod 9 5 = (1, 4)) (h4 : nat.div_mod 1 5 = (0, 1)) :
  45 = 140 :=
sorry

end binary_to_base5_l753_753250


namespace fraction_arithmetic_seq_sum_l753_753753

variables {a_n : ℕ → ℝ} [arithmetic_sequence : ∀ n, a_n (n+1) - a_n n = d]

def S (n : ℕ) := (n/2) * (a_n 1 + a_n n)

theorem fraction_arithmetic_seq_sum (h : S 9 / S 5 = 1) :
  ∀ (a_n : ℕ → ℝ) (S : ℕ → ℝ), (a_n 5 / a_n 3 = 5 / 9) → (S 9 / S 5 = 1) := sorry

end fraction_arithmetic_seq_sum_l753_753753


namespace train_length_l753_753210

theorem train_length (v_train_kmph : ℝ) (v_man_kmph : ℝ) (time_sec : ℝ) 
  (h1 : v_train_kmph = 25) 
  (h2 : v_man_kmph = 2) 
  (h3 : time_sec = 20) : 
  (150 : ℝ) = (v_train_kmph + v_man_kmph) * (1000 / 3600) * time_sec := 
by {
  -- sorry for the steps here
  sorry
}

end train_length_l753_753210


namespace polynomials_commute_l753_753528

noncomputable def P (x : ℝ) : ℝ := x^2 - α

theorem polynomials_commute 
  (Q R : ℝ → ℝ)
  (comm_QP : ∀ x, P (Q x) = Q (P x)) 
  (comm_RP : ∀ x, P (R x) = R (P x)) : 
  ∀ x, Q (R x) = R (Q x) := 
by
  sorry

end polynomials_commute_l753_753528


namespace S_equals_l753_753413
noncomputable def S : Real :=
  1 / (5 - Real.sqrt 23) + 1 / (Real.sqrt 23 - Real.sqrt 20) - 1 / (Real.sqrt 20 - 4) -
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 12) - 1 / (Real.sqrt 12 - 3)

theorem S_equals : S = 2 * Real.sqrt 23 - 2 :=
by
  sorry

end S_equals_l753_753413


namespace domain_of_f_l753_753820

noncomputable def f (x : ℝ) : ℝ := 1 / real.sqrt x

theorem domain_of_f :
  {x : ℝ | f x = 1 / real.sqrt x} = {x : ℝ | 0 < x} :=
by
  -- Domain proof skipped
  sorry

end domain_of_f_l753_753820


namespace percentage_of_cookies_ingrid_baked_l753_753878

noncomputable def irin_ratio : ℝ := 9.18
noncomputable def ingrid_ratio : ℝ := 5.17
noncomputable def nell_ratio : ℝ := 2.05
noncomputable def total_cookies : ℝ := 148

theorem percentage_of_cookies_ingrid_baked :
  let total_ratio := irin_ratio + ingrid_ratio + nell_ratio
  let ingrid_share := ingrid_ratio / total_ratio
  let percentage := ingrid_share * 100
  abs (percentage - 31.52) < 0.01 := 
by 
  let total_ratio := irin_ratio + ingrid_ratio + nell_ratio
  let ingrid_share := ingrid_ratio / total_ratio
  let percentage := ingrid_share * 100
  show abs (percentage - 31.52) < 0.01 from sorry

end percentage_of_cookies_ingrid_baked_l753_753878


namespace solution_of_modified_system_l753_753018

theorem solution_of_modified_system
  (a b x y : ℝ)
  (h1 : 2*a*3 + 3*4 = 18)
  (h2 : -3 + 5*b*4 = 17)
  : (x + y = 7 ∧ x - y = -1) → (2*a*(x+y) + 3*(x-y) = 18 ∧ (x+y) - 5*b*(x-y) = -17) → (x = (7 / 2) ∧ y = (-1 / 2)) :=
by
sorry

end solution_of_modified_system_l753_753018


namespace train_stoppage_time_l753_753159

theorem train_stoppage_time
  (D : ℝ) -- Distance in kilometers
  (T_no_stop : ℝ := D / 300) -- Time without stoppages in hours
  (T_with_stop : ℝ := D / 200) -- Time with stoppages in hours
  (T_stop : ℝ := T_with_stop - T_no_stop) -- Time lost due to stoppages in hours
  (T_stop_minutes : ℝ := T_stop * 60) -- Time lost due to stoppages in minutes
  (stoppage_per_hour : ℝ := T_stop_minutes / (D / 300)) -- Time stopped per hour of travel
  : stoppage_per_hour = 30 := sorry

end train_stoppage_time_l753_753159


namespace line_passing_fixed_point_l753_753169

theorem line_passing_fixed_point
  (O : Point) (conic : Conic) (A B : Point) 
  (right_angle : ∠ A O B = 90) 
  (on_conic_A : on_conic A conic) 
  (on_conic_B : on_conic B conic) :
  ∃ P' : Point, (P' ∈ normal O conic) ∧ 
  (line_through A B P') := 
sorry

end line_passing_fixed_point_l753_753169


namespace cards_draw_l753_753133

theorem cards_draw (
  cards_total : ℕ := 16,
  cards_per_color : ℕ := 4,
  colors : ℕ := 4,
  draws : ℕ := 3
  )
  (h_total : cards_total = colors * cards_per_color)
  (h_draws : draws = 3)
  : (nat.choose cards_total draws - colors * (nat.choose cards_per_color draws) = 544) :=
sorry

end cards_draw_l753_753133


namespace xy_sufficient_not_necessary_l753_753126

theorem xy_sufficient_not_necessary (x y : ℝ) :
  (xy ≠ 6) → (x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → xy ≠ 6) := by
  sorry

end xy_sufficient_not_necessary_l753_753126


namespace mushrooms_on_log_l753_753571

theorem mushrooms_on_log :
  ∃ (G : ℕ), ∃ (S : ℕ), S = 9 * G ∧ G + S = 30 ∧ G = 3 :=
by
  sorry

end mushrooms_on_log_l753_753571


namespace sum_a_eq_b_l753_753424

noncomputable def a : ℕ → ℝ
| 1     := 0.2
| (n+1) := if (n+1) % 2 = 1 then (0.201)^a n else (0.202)^a n

def is_sorted (l : list ℝ) : Prop := ∀ ⦃a b⦄, a ∈ l → b ∈ l → a ≤ b → list.index_of a l ≤ list.index_of b l

def perm_sorted (l : list ℝ) : list ℝ := l.qsort (≤)

def a_seq : list ℝ := (list.range 1000).map (λ n, a (n+1))

def b_seq : list ℝ := perm_sorted a_seq

theorem sum_a_eq_b :
  ∑ k in finset.range 1000, ite (a k = b_seq.nth_le k sorry) k 0 = 750 :=
sorry

end sum_a_eq_b_l753_753424


namespace anya_age_l753_753930

theorem anya_age (n : ℕ) (h : 110 ≤ (n * (n + 1)) / 2 ∧ (n * (n + 1)) / 2 ≤ 130) : n = 15 :=
sorry

end anya_age_l753_753930


namespace deposit_to_wife_percentage_l753_753585

theorem deposit_to_wife_percentage (total_income : ℝ) (children_percentage : ℝ) (num_children : ℕ)
  (donation_percentage : ℝ) (final_amount : ℝ) (wife_percentage : ℝ) :
  let distributed_to_children := (children_percentage / 100) * total_income * num_children
  let remaining_after_children := total_income - distributed_to_children
  let donated_to_orphan_house := (donation_percentage / 100) * remaining_after_children
  let remaining_after_donation := remaining_after_children - donated_to_orphan_house
  let deposited_to_wife := remaining_after_donation - final_amount
  let computed_wife_percentage := (deposited_to_wife / total_income) * 100
  in
  total_income = 200000 → children_percentage = 15 → num_children = 3 →
  donation_percentage = 5 → final_amount = 40000 →
  computed_wife_percentage = 32.25 := sorry

end deposit_to_wife_percentage_l753_753585


namespace number_of_algebra_textbooks_l753_753117

theorem number_of_algebra_textbooks
  (x y n : ℕ)
  (h₁ : x * n + y = 2015)
  (h₂ : y * n + x = 1580) :
  y = 287 := 
sorry

end number_of_algebra_textbooks_l753_753117


namespace min_value_fraction_l753_753646

theorem min_value_fraction (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a + 2 * b = 2) :
  (a + b) / (a * b) ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_fraction_l753_753646


namespace composite_10201_base_n_composite_10101_base_n_l753_753801

-- 1. Prove that 10201_n is composite given n > 2
theorem composite_10201_base_n (n : ℕ) (h : n > 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n^4 + 2*n^2 + 1 := 
sorry

-- 2. Prove that 10101_n is composite given n > 2.
theorem composite_10101_base_n (n : ℕ) (h : n > 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n^4 + n^2 + 1 := 
sorry

end composite_10201_base_n_composite_10101_base_n_l753_753801


namespace final_result_after_n_cycles_l753_753205

-- Define the initial conditions and the resulting property to be proven
def cube_reciprocal_sequence (x : ℝ) (n : ℕ) : ℝ :=
  if x = 0 then 0 else
    let cube := (λ a : ℝ, a^3) in
    let recip := (λ a : ℝ, a⁻¹) in
    (recip ∘ cube)^[n] x

theorem final_result_after_n_cycles (x : ℝ) (n : ℕ) (h : x ≠ 0) :
  cube_reciprocal_sequence x n = x ^ ((-3)^n) := by
  sorry

end final_result_after_n_cycles_l753_753205


namespace yards_mowed_by_christian_l753_753239

-- Definitions based on the provided conditions
def initial_savings := 5 + 7
def sue_earnings := 6 * 2
def total_savings := initial_savings + sue_earnings
def additional_needed := 50 - total_savings
def short_amount := 6
def christian_earnings := additional_needed - short_amount
def charge_per_yard := 5

theorem yards_mowed_by_christian : 
  (christian_earnings / charge_per_yard) = 4 :=
by
  sorry

end yards_mowed_by_christian_l753_753239


namespace inequality_sum_squares_l753_753429

theorem inequality_sum_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 :=
sorry

end inequality_sum_squares_l753_753429


namespace f_range_l753_753618

def f (x : ℝ) : ℝ := sin x - cos (x - π / 6)

theorem f_range : set.range f = set.Icc (-1) 1 :=
by
  sorry

end f_range_l753_753618


namespace ratio_of_chords_l753_753452

theorem ratio_of_chords 
  (A B C D : Point) 
  (AB BC CD : ℝ) 
  (h1 : AB = BC) 
  (h2 : BC = CD) 
  (radius : ℝ := AB / 2) 
  (l : Line) 
  (tangent_l : Tangent l (Circle.mk C radius))
  : ratio_of_chords (Chord.mk l (Circle.mk A radius)) (Chord.mk l (Circle.mk B radius)) = sqrt 6 / 2 :=
by 
  sorry

end ratio_of_chords_l753_753452


namespace minimum_value_of_quadratic_l753_753538

def quadratic_polynomial (x : ℝ) : ℝ := 2 * x^2 - 16 * x + 22

theorem minimum_value_of_quadratic : ∃ x : ℝ, quadratic_polynomial x = -10 :=
by 
  use 4
  { sorry }

end minimum_value_of_quadratic_l753_753538


namespace inequality_proof_l753_753324

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hSum : a + b + c = 1)

theorem inequality_proof :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 1 / 2 :=
by
  sorry

end inequality_proof_l753_753324


namespace integral_result_l753_753702

theorem integral_result (a : ℝ) (h : ∫ x in 0..(real.pi / 2), (real.sin x - a * real.cos x) = 2) : a = -1 :=
by
  sorry

end integral_result_l753_753702


namespace thirtieth_digit_fraction_sum_l753_753535

theorem thirtieth_digit_fraction_sum :
  let a := 1 / 11
  let b := 1 / 13
  let s := a + b
  (decimal_fractions.thirtieth_digit s) = 2 :=
sorry

end thirtieth_digit_fraction_sum_l753_753535


namespace sum_f_eq_2020_point_25_l753_753641

def f (m n : ℕ) (hn : n > 1) : ℝ :=
  (Real.root n (3^m)) / (Real.root n (3^m) + 3)

theorem sum_f_eq_2020_point_25 :
  ∑ k in Finset.range 4041, f k 2020 (by decide) = 2020.25 := 
sorry

end sum_f_eq_2020_point_25_l753_753641


namespace surface_area_correct_l753_753866

def total_edge_length_of_cube : ℝ := 180

def number_of_edges_of_cube : ℝ := 12

def edge_length_of_cube := total_edge_length_of_cube / number_of_edges_of_cube

def surface_area_of_cube (edge_length : ℝ) : ℝ := 6 * edge_length ^ 2

theorem surface_area_correct :
  surface_area_of_cube edge_length_of_cube = 1350 :=
by
  have h_edge_length : edge_length_of_cube = 180 / 12 := rfl
  rw [h_edge_length]
  norm_num
  sorry

end surface_area_correct_l753_753866


namespace cards_circle_impossibility_l753_753527

theorem cards_circle_impossibility (n : ℕ) (h : n > 3) :
  ¬ (∃ (moves : list (list ℕ)), 
      (∀ (move : list ℕ), move.length = 3 → 
        let ⟨a, b, c⟩ := (move.nth_le 0 sorry, move.nth_le 1 sorry, move.nth_le 2 sorry) in
        -- Define the move transformation here if necessary
        true) →
     ∀ (k : ℕ) (hk : k < n), 
      -- Initial condition: all cards back side visible
      let initial := vector.replicate n false in
      -- Moves:
      let final := initial.infer_moves moves in
        -- Final condition check: all cards front side visible
        final = vector.replicate n true)) :=
sorry

end cards_circle_impossibility_l753_753527


namespace digit_30_of_sum_l753_753533

-- Defining the given fractions
def fraction1 : ℚ := 1 / 11
def fraction2 : ℚ := 1 / 13

-- Defining the repeating decimals
def decimal1 := "0.09".cycle
def decimal2 := "0.076923".cycle

-- Defining the least common multiple of the periods
def lcm_period := Nat.lcm 2 6

-- Defining the repeating block of the sum's decimal form
def repeating_sum_seq := "097032".cycle

-- The Lean 4 statement to be proved
theorem digit_30_of_sum : 
  (fraction1 + fraction2).decimalExpansion !30 == some 2 := 
by
  sorry

end digit_30_of_sum_l753_753533


namespace incorrect_statement_D_l753_753548

-- Definitions based on conditions
def statement_A : Prop := ∀ (x : ℕ), isRat x
def statement_B : Prop := ∀ (p q : ℤ), q ≠ 0 → isRat (p / q)
def statement_C : Prop := ∀ (x : ℤ), true
def statement_D : Prop := ∀ (x : ℝ), (x > 0 ∨ x < 0) → isRat x

-- The incorrect statement
theorem incorrect_statement_D : ¬ statement_D := by
  sorry

end incorrect_statement_D_l753_753548


namespace probability_of_qualification_l753_753383

-- Define the probability of hitting a target and the number of shots
def probability_hit : ℝ := 0.4
def number_of_shots : ℕ := 3

-- Define the probability of hitting a specific number of targets
noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- Define the event of qualifying by hitting at least 2 targets
noncomputable def probability_qualify (n : ℕ) (p : ℝ) : ℝ :=
  binomial n 2 p + binomial n 3 p

-- The theorem we want to prove
theorem probability_of_qualification : probability_qualify number_of_shots probability_hit = 0.352 :=
  by sorry

end probability_of_qualification_l753_753383


namespace baseball_card_decrease_l753_753180

noncomputable def compounded_decrease (initial_value : ℝ) (decrease1 decrease2 decrease3 : ℝ) : ℝ :=
  let value_after_first_year := initial_value * (1 - decrease1)
  let value_after_second_year := value_after_first_year * (1 - decrease2)
  let value_after_third_year := value_after_second_year * (1 - decrease3)
  1 - (value_after_third_year / initial_value)

theorem baseball_card_decrease :
  compounded_decrease 100 0.20 0.10 0.15 = 0.388 := 
by
  unfold compounded_decrease
  norm_num
  sorry

end baseball_card_decrease_l753_753180


namespace existence_of_sums_l753_753064

theorem existence_of_sums (a b c : ℕ) (hc : ℤ) (h : a * b ≥ hc * hc) :
  ∃ n : ℕ, ∃ (x y : Fin n → ℤ),
  (∑ i, x i ^ 2 = a) ∧
  (∑ i, y i ^ 2 = b) ∧
  (∑ i, x i * y i = hc) :=
sorry

end existence_of_sums_l753_753064


namespace verify_cube_seq_l753_753631

def cube_seq (n : ℕ) : ℕ := n^3

theorem verify_cube_seq :
  cube_seq 4 = 64 ∧ cube_seq 5 = 125 ∧ cube_seq 6 = 216 :=
by
  split,
  { -- Proof for cube_seq 4
    sorry },
  split,
  { -- Proof for cube_seq 5
    sorry },
  { -- Proof for cube_seq 6
    sorry }

end verify_cube_seq_l753_753631


namespace mean_correct_and_no_seven_l753_753937

-- Define the set of numbers.
def numbers : List ℕ := 
  [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

-- Define the arithmetic mean of the numbers in the set.
def arithmetic_mean (l : List ℕ) : ℕ := (l.sum / l.length)

-- Specify the mean value
def mean_value : ℕ := 109629012

-- State the theorem that the mean value is correct and does not contain the digit 7.
theorem mean_correct_and_no_seven : arithmetic_mean numbers = mean_value ∧ ¬ 7 ∈ (mean_value.digits 10) :=
  sorry

end mean_correct_and_no_seven_l753_753937


namespace number_of_valid_tuples_l753_753965

theorem number_of_valid_tuples : 
  ∃ (n : ℕ), n = 12376 ∧ ∀ (a : ℕ → ℤ) (S : ℤ), 
    (S = ∑ i in finset.range 13, a i) → 
    (∀ i, i < 13 → (a i)^2 = 2 * (S - a i))
  sorry

end number_of_valid_tuples_l753_753965


namespace probability_of_zero_score_l753_753376

namespace BallDrawingProblem

def ball := ℕ

noncomputable def totalBalls : list ball := [1, 2, 3, 3, 3, 3]

noncomputable def score : ball → ℤ :=
  | 1 := 1
  | 2 := 0
  | _ := -1

def valid_scores (draws : list ball) : ℤ :=
  (draws.map score).sum

def count_valid_cases : ℕ :=
  let draws := list.prod (list.replicate 3 totalBalls)
  draws.count (λ d, valid_scores d = 0)

def total_cases : ℕ :=
  (totalBalls.length ^ 3)

theorem probability_of_zero_score :
  (count_valid_cases : ℚ) / (total_cases : ℚ) = 11 / 54 := 
sorry

end BallDrawingProblem

end probability_of_zero_score_l753_753376


namespace muffins_per_person_l753_753743

-- Definitions based on conditions
def total_friends : ℕ := 4
def total_people : ℕ := 1 + total_friends
def total_muffins : ℕ := 20

-- Theorem statement for the proof
theorem muffins_per_person : total_muffins / total_people = 4 := by
  sorry

end muffins_per_person_l753_753743


namespace regular_ngon_with_particular_angle_l753_753382

theorem regular_ngon_with_particular_angle (n : ℕ) (h₁ : n > 6)
    (h₂ : ∠P A₂ A₄ = 120°) : n = 18 := by
  sorry

end regular_ngon_with_particular_angle_l753_753382


namespace main_theorem_l753_753332

-- Define the conditions
variables {m n : ℝ}
def ellipse (x y : ℝ) : Prop := (x^2 / m^2) + y^2 = 1
def hyperbola (x y : ℝ) : Prop := (x^2 / n^2) - y^2 = 1
def coincident_foci := (m^2 - 1 = n^2 + 1)
def eccentricity_ellipse (m : ℝ) : ℝ := real.sqrt (1 - 1 / m^2)
def eccentricity_hyperbola (n : ℝ) : ℝ := real.sqrt (1 + 1 / n^2)

-- State the main theorem for proof
theorem main_theorem 
  (h1 : m > 1) 
  (h2 : n > 0) 
  (h3 : coincident_foci)
  : m > n ∧ (eccentricity_ellipse m * eccentricity_hyperbola n) > 1 :=
sorry

end main_theorem_l753_753332


namespace incorrect_proposition3_l753_753675

open Real

-- Definitions from the problem
def prop1 (x : ℝ) := 2 * sin (2 * x - π / 3) = 2
def prop2 (x y : ℝ) := tan x + tan (π - x) = 0
def prop3 (x1 x2 : ℝ) (k : ℤ) := x1 - x2 = (k : ℝ) * π → k % 2 = 1
def prop4 (x : ℝ) := cos x ^ 2 + sin x >= -1

-- Incorrect proposition proof
theorem incorrect_proposition3 (x1 x2 : ℝ) (k : ℤ) :
  sin (2 * x1 - π / 4) = 0 →
  sin (2 * x2 - π / 4) = 0 →
  x1 - x2 ≠ (k : ℝ) * π := sorry

end incorrect_proposition3_l753_753675


namespace smaller_than_negative_one_l753_753926

theorem smaller_than_negative_one :
  ∃ x ∈ ({0, -1/2, 1, -2} : Set ℝ), x < -1 ∧ x = -2 :=
by
  -- the proof part is skipped
  sorry

end smaller_than_negative_one_l753_753926


namespace sqrt_S_arithmetic_sequence_sum_c_n_l753_753329

noncomputable section

-- Define the sequences and conditions
def S (n : ℕ) : ℝ
def a (n : ℕ) : ℝ := 
if n = 1 then 1 else 1 + 2 * Real.sqrt (S (n - 1))

-- First part: Prove the sequence {sqrt(S_n)} forms an arithmetic sequence
theorem sqrt_S_arithmetic_sequence (n : ℕ) : (Real.sqrt (S (n + 1)) = Real.sqrt (S n) + 1) :=
sorry

-- Second part: Sum of the first n terms of sequence {c_n} is n * 2^(n+1)
def c (n : ℕ) : ℝ := (Real.sqrt (S n) + 1) * 2^n

def T (n : ℕ) : ℝ := (Finset.range n).sum (λ i, c (i + 1))

theorem sum_c_n (n : ℕ) : T n = n * 2^(n + 1) :=
sorry

end sqrt_S_arithmetic_sequence_sum_c_n_l753_753329


namespace ratio_of_perimeters_of_squares_l753_753522

theorem ratio_of_perimeters_of_squares (d1 d4 : ℝ) (h : d4 = 4 * d1) :
  let s1 := d1 / Real.sqrt 2,
      s4 := d4 / Real.sqrt 2,
      P1 := 4 * s1,
      P4 := 4 * s4 in
  P4 / P1 = 4 := by
  sorry

end ratio_of_perimeters_of_squares_l753_753522


namespace order_of_numbers_l753_753501

theorem order_of_numbers (a b c : ℝ) (ha : a = 3 ^ 0.7) (hb : b = 0.7 ^ 3) (hc : c = Real.log 0.7 / Real.log 3) : c < b ∧ b < a := by
  sorry

end order_of_numbers_l753_753501


namespace parabola_focus_eq_l753_753629

variable (y : ℝ)

def parabola_eq (y : ℝ) : ℝ := -1 / 4 * y^2 + 2

theorem parabola_focus_eq : 
  (1, 0) = (λ y, parabola_eq y) := 
sorry

end parabola_focus_eq_l753_753629


namespace least_k_for_maximal_arrangement_l753_753751

noncomputable def minimal_maximal_pieces (n : ℕ) (h: 3 ≤ n) : ℕ :=
  if hx : n ≥ 3 then 2 * n + 1
  else 0

theorem least_k_for_maximal_arrangement {n : ℕ} (h : 3 ≤ n) :
  ∃ k, k = minimal_maximal_pieces n h ∧ ∀ k' < k, ¬ (maximal_arrangement n k') := 
begin
  use 2 * n + 1,
  split,
  { simp [minimal_maximal_pieces, h], 
    exact if_pos h },
  { intros k' hk',
    sorry }
end

end least_k_for_maximal_arrangement_l753_753751


namespace min_value_of_sum_l753_753989

theorem min_value_of_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x * y + 2 * x + y = 4) : x + y ≥ 2 * Real.sqrt 6 - 3 :=
sorry

end min_value_of_sum_l753_753989


namespace max_value_ineq_l753_753121

theorem max_value_ineq (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : a + b + c = 1) :
  (a + 3 * b + 5 * c) * (a + b / 3 + c / 5) ≤ 9 / 5 :=
sorry

end max_value_ineq_l753_753121


namespace nonneg_expr_in_interval_l753_753974

variable (x : ℝ)

def num := x - 12 * x^2 + 36 * x^3
def denom := 9 - x^3
def expr := num / denom

theorem nonneg_expr_in_interval :
  ∀ x, 0 ≤ x ∧ x < 3 → expr x ≥ 0 :=
by
  sorry

end nonneg_expr_in_interval_l753_753974


namespace find_B_value_l753_753285

-- Define the polynomial and conditions
def polynomial (A B : ℤ) (z : ℤ) : ℤ := z^4 - 12 * z^3 + A * z^2 + B * z + 36

-- Define roots and their properties according to the conditions
def roots_sum_to_twelve (r1 r2 r3 r4 : ℕ) : Prop := r1 + r2 + r3 + r4 = 12

-- The final statement to prove
theorem find_B_value (r1 r2 r3 r4 : ℕ) (A B : ℤ) (h_sum : roots_sum_to_twelve r1 r2 r3 r4)
    (h_pos : r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0) 
    (h_poly : polynomial A B = (z^4 - 12*z^3 + Az^2 + Bz + 36)) :
    B = -96 :=
    sorry

end find_B_value_l753_753285


namespace memo_competition_language_selection_l753_753420

theorem memo_competition_language_selection
    (n : ℕ)
    (hn : n ≥ 3)
    (participants : Finset (Fin 3n))
    (languages : Finset (Fin n))
    (language_spoken_by : Fin 3n → Finset (Fin n))
    (h_participant_count : participants.card = 3n)
    (h_language_count : languages.card = n)
    (h_languages_per_participant : ∀ p ∈ participants, (language_spoken_by p).card = 3) :
  ∃ chosen_languages : Finset (Fin n),
    chosen_languages.card ≥ Nat.ceil (2 * n / 9 : ℚ) ∧
    ∀ p ∈ participants, (language_spoken_by p ∩ chosen_languages).card ≤ 2 := 
sorry

end memo_competition_language_selection_l753_753420


namespace pebble_sequence_10_l753_753931

-- A definition for the sequence based on the given conditions and pattern.
def pebble_sequence : ℕ → ℕ
| 0 => 1
| 1 => 5
| 2 => 12
| 3 => 22
| (n + 4) => pebble_sequence (n + 3) + (3 * (n + 1) + 1)

-- Theorem that states the value at the 10th position in the sequence.
theorem pebble_sequence_10 : pebble_sequence 9 = 145 :=
sorry

end pebble_sequence_10_l753_753931


namespace students_neither_l753_753785

-- Define the conditions
def total_students : ℕ := 60
def students_math : ℕ := 40
def students_physics : ℕ := 35
def students_both : ℕ := 25

-- Define the problem statement
theorem students_neither : total_students - ((students_math - students_both) + (students_physics - students_both) + students_both) = 10 :=
by
  sorry

end students_neither_l753_753785


namespace sum_of_cubes_of_three_consecutive_integers_l753_753837

theorem sum_of_cubes_of_three_consecutive_integers (a : ℕ) (h : (a * a) + (a + 1) * (a + 1) + (a + 2) * (a + 2) = 2450) : a * a * a + (a + 1) * (a + 1) * (a + 1) + (a + 2) * (a + 2) * (a + 2) = 73341 :=
by
  sorry

end sum_of_cubes_of_three_consecutive_integers_l753_753837


namespace midpoints_HK_BC_l753_753026

open Real EuclideanGeometry

variable (θ : Real) (H K B C M : Point) (HK BC : Line) (BH CK : Segment)

-- Definitions based on the given conditions
def is_midpoint (M B C : Point) : Prop := dist B M = dist M C
def are_perpendicular (l1 l2 : Line) : Prop := ∃ θ, angle l1 l2 = θ ∧ θ = π/2
def intersect_angle (l1 l2 : Line) (θ : Real) : Prop := angle l1 l2 = θ ∧ θ ≠ π / 2

theorem midpoints_HK_BC (θ : Real) (H K B C M : Point) (HK BC : Line) (BH CK : Segment) :
  (intersect_angle HK BC θ) →
  (is_midpoint M B C) →
  (are_perpendicular BH HK) →
  (are_perpendicular CK HK) →
  ((∃ θ, θ ≠ π/2) → ∃ H K M, dist M H = dist M K ∨ dist M H ≠ dist M K) :=
by
  intros h_intersect h_midpoint h_BH_perpendicular h_CK_perpendicular
  sorry

end midpoints_HK_BC_l753_753026


namespace sunflower_seed_total_l753_753408

theorem sunflower_seed_total (c s : ℕ) (h1 : c = 9) (h2 : s = 6) : c * s = 54 :=
by
  rw [h1, h2]
  simp
  exact rfl

end sunflower_seed_total_l753_753408


namespace min_final_exam_score_l753_753923

theorem min_final_exam_score (q1 q2 q3 q4 final_exam : ℤ)
    (H1 : q1 = 90) (H2 : q2 = 85) (H3 : q3 = 77) (H4 : q4 = 96) :
    (1/2) * (q1 + q2 + q3 + q4) / 4 + (1/2) * final_exam ≥ 90 ↔ final_exam ≥ 93 :=
by
    sorry

end min_final_exam_score_l753_753923


namespace S_subset_T_l753_753077
noncomputable theory

def S : set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ (x^2 - y^2) % 2 = 1 }

def T : set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ sin(2 * real.pi * x^2) = sin(2 * real.pi * y^2) ∧ cos(2 * real.pi * x^2) = cos(2 * real.pi * y^2) }

theorem S_subset_T : S ⊆ T :=
sorry

end S_subset_T_l753_753077


namespace different_flavors_possible_l753_753286

theorem different_flavors_possible 
  (blue_candies : ℕ) (yellow_candies : ℕ)
  (h_blue : blue_candies = 5) (h_yellow : yellow_candies = 4) :
  ∃ n, n = 18 :=
by
  use 18
  sorry

end different_flavors_possible_l753_753286


namespace parabola_focus_l753_753292

theorem parabola_focus {x : ℝ} : 
    (∃ y : ℝ, y = 4 * x^2) → 
    opens_upwards_and_focus_is 
        (0, 1 / 16) := 
by
  sorry

end parabola_focus_l753_753292


namespace hyperbola_solution_exists_l753_753651

noncomputable theory

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 9 = 1

def shared_foci (c : ℝ) : Prop := c = 4

def sum_eccentricities (e_ellipse e_hyper : ℝ) : Prop := 
  e_ellipse + e_hyper = 14 / 5

def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 12 = 1

theorem hyperbola_solution_exists :
  (∀ x y : ℝ, ellipse_eq x y) →
  shared_foci 4 →
  sum_eccentricities (4/5) 2 →
  (∀ x y : ℝ, hyperbola_eq x y) :=
by
  intros
  -- proof steps are omitted
  sorry

end hyperbola_solution_exists_l753_753651


namespace find_y_intercept_range_l753_753496

-- Definitions as per conditions
def line (k : ℝ) : (ℝ × ℝ) → Prop := λ p, p.2 = k * p.1 + 1
def hyperbola : (ℝ × ℝ) → Prop := λ p, p.1^2 - p.2^2 = 1
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def line_through (A B : ℝ × ℝ) : ℝ × ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Define the main problem and goal
theorem find_y_intercept_range :
  ∀ (k : ℝ), 1 < k ∧ k < sqrt 2 →
  ∃ b : ℝ, (line (-2, 0)) ∧ (hyperbola (A : ℝ × ℝ)) ∧ (hyperbola (B : ℝ × ℝ)) 
  ∧ (mid := midpoint A B),
  l = line_through mid (-2, 0)
  ∧ (b = y-intercept l)
  ∧ (b ∈ ((-∞, -2 - sqrt 2) ∪ (2, + ∞))) :=
sorry

end find_y_intercept_range_l753_753496


namespace kate_money_left_l753_753747

theorem kate_money_left : 
  let march_savings := 27
  let april_savings := 13
  let may_savings := 28
  let june_savings := 35
  let july_savings := 2 * april_savings
  let total_savings := march_savings + april_savings + may_savings + june_savings + july_savings
  let keyboard_cost := 49
  let mouse_cost := 5
  let headset_cost := 15
  let videogame_cost := 25
  let total_spent := keyboard_cost + mouse_cost + headset_cost + videogame_cost
  in total_savings - total_spent = 35 :=
by
  let march_savings := 27
  let april_savings := 13
  let may_savings := 28
  let june_savings := 35
  let july_savings := 2 * april_savings
  let total_savings := march_savings + april_savings + may_savings + june_savings + july_savings
  let keyboard_cost := 49
  let mouse_cost := 5
  let headset_cost := 15
  let videogame_cost := 25
  let total_spent := keyboard_cost + mouse_cost + headset_cost + videogame_cost
  show total_savings - total_spent = 35 from
    sorry

end kate_money_left_l753_753747


namespace shortest_altitude_l753_753245

theorem shortest_altitude {a b c : ℝ} (h_a : a = 9) (h_b : b = 12) (h_c : c = 15)
  (h_right : a^2 + b^2 = c^2) : 
  ∃ h : ℝ, (1/2) * c * h = (1/2) * a * b ∧ h = 7.2 :=
by
  have h_area : (1/2) * a * b = 54 := by sorry
  use 7.2
  simp
  split
  sorry
  have h_altitude : 7.2 = 108 / 15 := by sorry
  exact h_altitude

end shortest_altitude_l753_753245


namespace f_prime_midpoint_pos_l753_753763

def f (x a: ℝ) : ℝ := x ^ 2 + 2 * x - a * (Real.log x + x)
def f_prime (x a: ℝ) : ℝ := (2 * x - a) * (x + 1) / x

variable {a c: ℝ}
variable {x1 x2: ℝ}
-- The conditions
axiom h₁ : f x1 a = c
axiom h₂ : f x2 a = c
axiom h₃ : x1 ≠ x2
axiom h₄ : 0 < x1
axiom h₅ : 0 < x2
axiom h₆ : a > 0

-- What we want to prove
theorem f_prime_midpoint_pos : f_prime ((x1 + x2) / 2) a > 0 :=
sorry

end f_prime_midpoint_pos_l753_753763


namespace number_of_second_graders_l753_753132

-- Define the number of kindergartners, first graders, and total students
def k : ℕ := 14
def f : ℕ := 24
def t : ℕ := 42

-- Define the number of second graders
def s : ℕ := t - (k + f)

-- The theorem to prove
theorem number_of_second_graders : s = 4 := by
  -- We can use sorry here since we are not required to provide the proof
  sorry

end number_of_second_graders_l753_753132


namespace friendship_groups_ways_l753_753266

noncomputable def count_friendship_configurations : Nat :=
  let n : Nat := 8
  let pairs := n.choose 2  -- Number of ways to choose 2 out of 8.
  pairs / 2  -- Dividing by 2 because each pair is counted twice.
  
theorem friendship_groups_ways :
  count_friendship_configurations = 210 := by
  sorry

end friendship_groups_ways_l753_753266


namespace subset_not_covered_by_lines_l753_753524

theorem subset_not_covered_by_lines (points : set (ℝ × ℝ)) (h_card : points.card = 666)
  (h_not_covered : ∀ lines : set (set (ℝ × ℝ)), lines.card = 10 → ∃ p ∈ points, ∀ l ∈ lines, p ∉ l) :
  ∃ subset : set (ℝ × ℝ), subset ⊆ points ∧ subset.card = 66 ∧ ∀ lines : set (set (ℝ × ℝ)), lines.card = 10 → ∃ p ∈ subset, ∀ l ∈ lines, p ∉ l := 
sorry

end subset_not_covered_by_lines_l753_753524


namespace minimum_value_of_f_l753_753827

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem minimum_value_of_f : ∃ y, (∀ x, f x ≥ y) ∧ y = 3 := 
by
  sorry

end minimum_value_of_f_l753_753827


namespace total_students_after_four_years_l753_753232

noncomputable def total_students (initial_students : ℕ) 
                                 (join_start : ℕ) (join_diff : ℕ) 
                                 (leave_start : ℕ) (leave_diff : ℕ)
                                 (years : ℕ) : ℕ :=
initial_students + ∑ i in finset.range(years), (join_start + i * join_diff) - ∑ i in finset.range(years), (leave_start + i * leave_diff)

theorem total_students_after_four_years : 
  total_students 150 30 5 15 3 4 = 222 := by 
  sorry

end total_students_after_four_years_l753_753232


namespace real_solutions_of_polynomial_l753_753277

theorem real_solutions_of_polynomial (b : ℝ) :
  b < -4 → ∃! x : ℝ, x^3 - b * x^2 - 4 * b * x + b^2 - 4 = 0 :=
by
  sorry

end real_solutions_of_polynomial_l753_753277


namespace solve_system_of_equations_l753_753606

theorem solve_system_of_equations :
  ∃ x y : ℤ, (2 * x + 7 * y = -6) ∧ (2 * x - 5 * y = 18) ∧ (x = 4) ∧ (y = -2) := 
by
  -- Proof will go here
  sorry

end solve_system_of_equations_l753_753606


namespace min_sum_of_diagonals_l753_753109

theorem min_sum_of_diagonals (x y : ℝ) (α : ℝ) (hx : 0 < x) (hy : 0 < y) (hα : 0 < α ∧ α < π) (h_area : x * y * Real.sin α = 2) : x + y ≥ 2 * Real.sqrt 2 :=
sorry

end min_sum_of_diagonals_l753_753109


namespace probability_closer_to_center_l753_753573

def radius : ℝ := 1
def distance : ℝ := 20

theorem probability_closer_to_center : 
    let A_dartboard := Real.pi * radius^2,
        A_smaller_circle := Real.pi * (radius / 2)^2,
        Probability := A_smaller_circle / A_dartboard
    in Probability = 0.25 :=
by
  sorry

end probability_closer_to_center_l753_753573


namespace distinct_solutions_abs_eq_l753_753955

theorem distinct_solutions_abs_eq (x : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |x1 - |3 * x1 + 2|| = 4 ∧ |x2 - |3 * x2 + 2|| = 4 ∧
    (∀ x3 : ℝ, |x3 - |3 * x3 + 2|| = 4 → (x3 = x1 ∨ x3 = x2))) :=
sorry

end distinct_solutions_abs_eq_l753_753955


namespace max_abs_sum_l753_753995

theorem max_abs_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  |x| + |y| + |z| ≤ √3 :=
sorry

end max_abs_sum_l753_753995


namespace find_f_l753_753504

def Q (x : ℝ) : ℝ := 3 * x^4 + d * x^3 + e * x^2 + f * x + g

theorem find_f
  (h1 : d = 27)
  (h2 : g = -27)
  (h3 : 3 + d + e + f + g = -9)
  (h4 : -d/3 = -9)
  (h5 : g / 3 = -9)
  : f = -12 :=
by
  sorry

end find_f_l753_753504


namespace total_movies_shown_l753_753197

-- Define the conditions of the problem
def screens := 6
def open_hours := 8
def movie_duration := 2

-- Define the statement to prove
theorem total_movies_shown : screens * (open_hours / movie_duration) = 24 := 
by
  sorry

end total_movies_shown_l753_753197


namespace probability_palindrome_l753_753393

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

constant m : ℕ -- the number of valid starting digits

def count_palindromes : ℕ :=
  m * 10^3

def total_7_digit_numbers : ℕ :=
  m * 10^6

theorem probability_palindrome :
  (count_palindromes.to_Rat / total_7_digit_numbers) = 0.001 :=
by
  sorry

end probability_palindrome_l753_753393


namespace probability_palindrome_l753_753394

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

constant m : ℕ -- the number of valid starting digits

def count_palindromes : ℕ :=
  m * 10^3

def total_7_digit_numbers : ℕ :=
  m * 10^6

theorem probability_palindrome :
  (count_palindromes.to_Rat / total_7_digit_numbers) = 0.001 :=
by
  sorry

end probability_palindrome_l753_753394


namespace limit_of_sequence_l753_753791

noncomputable def limit_problem := 
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, |((2 * n - 3) / (n + 2) : ℝ) - 2| < ε

theorem limit_of_sequence : limit_problem :=
sorry

end limit_of_sequence_l753_753791


namespace distance_to_other_focus_l753_753603

-- Define the conditions
def distance_F2_A : ℝ := 1.5
def length_latus_rectum : ℝ := 5.4

-- Define the circle
def semi_minor_axis : ℝ := length_latus_rectum / 2

-- Main theorem statement
theorem distance_to_other_focus
  (distance_F2_A : ℝ)
  (length_latus_rectum : ℝ)
  (semi_minor_axis := length_latus_rectum / 2)
  (distance_F2_A = 1.5)
  (length_latus_rectum = 5.4) :
  let c := (5.04 / 3)
  2 * c = 12 :=
by
  sorry

end distance_to_other_focus_l753_753603


namespace find_common_ratio_l753_753319

variable (a_n : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

theorem find_common_ratio (h1 : a_n 1 = 2) (h2 : a_n 4 = 16) (h_geom : ∀ n, a_n n = a_n (n - 1) * q)
  : q = 2 := by
  sorry

end find_common_ratio_l753_753319


namespace charles_finishes_in_11_days_l753_753941

theorem charles_finishes_in_11_days : 
  ∀ (total_pages : ℕ) (pages_mon : ℕ) (pages_tue : ℕ) (pages_wed : ℕ) (pages_thu : ℕ) 
    (does_not_read_on_weekend : Prop),
  total_pages = 96 →
  pages_mon = 7 →
  pages_tue = 12 →
  pages_wed = 10 →
  pages_thu = 6 →
  does_not_read_on_weekend →
  ∃ days_to_finish : ℕ, days_to_finish = 11 :=
by
  intros
  sorry

end charles_finishes_in_11_days_l753_753941


namespace power_function_increasing_inverse_power_inequality_l753_753691

noncomputable def is_increasing (f : ℝ → ℝ) (S : set ℝ) : Prop :=
∀ x y ∈ S, x < y → f x < f y

theorem power_function_increasing (k : ℚ) (hk : k > 0) :
  is_increasing (λ x : ℝ, x^(k : ℝ)) {x | 0 ≤ x} := 
sorry

theorem inverse_power_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) (hc : c > 0) :
  a^(-c) < b^(-c) := 
sorry

end power_function_increasing_inverse_power_inequality_l753_753691


namespace value_by_which_number_is_multiplied_l753_753155

theorem value_by_which_number_is_multiplied (x : ℝ) : (5 / 6) * x = 10 ↔ x = 12 := by
  sorry

end value_by_which_number_is_multiplied_l753_753155


namespace max_n_leq_S_l753_753220

noncomputable def arith_seq (n : ℕ) : ℕ := 5 * n
noncomputable def geom_seq (n : ℕ) : ℕ := 5 * 2^(n-1)

def sum_arith_seq (N : ℕ) : ℕ := (N * (5 + arith_seq N)) / 2

theorem max_n_leq_S : 
  let S := sum_arith_seq 20 in
  (∀ (n : ℕ), geom_seq n ≤ S → n ≤ 8) :=
by 
  let S := sum_arith_seq 20
  intros n h
  have h₁ : geom_seq n = 5 * 2^(n-1), from rfl
  have h₂ : 5 * 2^(n-1) ≤ S, from h
  have S_val : sum_arith_seq 20 = 1050, from rfl
  have h₃ : 5 * 2^(n-1) ≤ 1050, by rw S_val; exact h₂
  have h₄ : 2^(n-1) ≤ 210, from nat.le_of_mul_le_mul_left h₃ (by dec_trivial)
  have h₅ : n-1 ≤ 7, from nat.log_le (2 : ℝ) h₄
  have h₆ : n ≤ 8, from nat.succ_le_succ h₅
  exact h₆

end max_n_leq_S_l753_753220


namespace farmer_field_area_l753_753009

variable (x : ℕ) (A : ℕ)

def planned_days : Type := {x : ℕ // 120 * x = 85 * (x + 2) + 40}

theorem farmer_field_area (h : {x : ℕ // 120 * x = 85 * (x + 2) + 40}) : A = 720 :=
by
  sorry

end farmer_field_area_l753_753009


namespace percent_exceed_not_ticketed_l753_753786

-- Defining the given conditions
def total_motorists : ℕ := 100
def percent_exceed_limit : ℕ := 50
def percent_with_tickets : ℕ := 40

-- Calculate the number of motorists exceeding the limit and receiving tickets
def motorists_exceed_limit := total_motorists * percent_exceed_limit / 100
def motorists_with_tickets := total_motorists * percent_with_tickets / 100

-- Theorem: Percentage of motorists exceeding the limit but not receiving tickets
theorem percent_exceed_not_ticketed : 
  (motorists_exceed_limit - motorists_with_tickets) * 100 / motorists_exceed_limit = 20 := 
by
  sorry

end percent_exceed_not_ticketed_l753_753786


namespace inequality_proof_l753_753770

theorem inequality_proof (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (h_condition : a * b + b * c + c * d + d * a = 1) :
    (a ^ 3 / (b + c + d)) + (b ^ 3 / (c + d + a)) + (c ^ 3 / (a + b + d)) + (d ^ 3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_proof_l753_753770


namespace grandfathers_age_ratio_l753_753160

theorem grandfathers_age_ratio (xiao_hong_age this_year: ℕ) (grandfather_age this_year: ℕ)
    (h1 : xiao_hong_age this_year = 8) (h2 : grandfather_age this_year = 64) : 
    (grandfather_age this_year - 1) / (xiao_hong_age this_year - 1) = 9 :=
by
  sorry

end grandfathers_age_ratio_l753_753160


namespace original_students_participated_l753_753384

theorem original_students_participated (S : ℕ) (h1 : S = 200) : 
  let FirstRoundEliminated := 0.4 * S,
      RemainingAfterFirst := 0.6 * S,
      ContinuedToThirdRound := 1 / 4 * RemainingAfterFirst in
      ContinuedToThirdRound = 30 :=
by
  sorry

end original_students_participated_l753_753384


namespace bisect_segment_trapezoid_l753_753792

variable {A B C D E : Type}

-- Definitions for points and segments
variables [Point A] [Point B] [Point C] [Point D] [Point E]
variables (l : Line)

-- Conditions: Trapezoid with parallel sides and intersection of diagonals
def is_trapezoid (A B C D : Point) : Prop := is_parallel (A, B) (C, D)
def intersection_point (A B C D E : Point) : Prop := 
  ∃ (X : Point), X ∈ line_through A C ∧ X ∈ line_through B D ∧ X = E

-- Additional condition: line segment through E parallel to AB and CD
def parallel_segment (A B : Point) (l : Line) : Prop :=
  ∃ (P Q : Point), P ∈ l ∧ Q ∈ l ∧ P ≠ Q ∧ line_through P Q ∥ line_through A B

-- The final proof statement
theorem bisect_segment_trapezoid
  (A B C D E : Point)
  (hlt : is_trapezoid A B C D)
  (hit : intersection_point A B C D E)
  (hp : parallel_segment A B l)
  : bisects E (midpoint (P, Q)) :=
sorry

end bisect_segment_trapezoid_l753_753792


namespace find_valid_pairs_l753_753550

-- Decalred the main definition for the problem.
def valid_pairs (x y : ℕ) : Prop :=
  (10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99) ∧ ((x + y)^2 = 100 * x + y)

-- Stating the theorem without the proof.
theorem find_valid_pairs :
  valid_pairs 20 25 ∧ valid_pairs 30 25 :=
sorry

end find_valid_pairs_l753_753550


namespace find_numbers_l753_753627

variables (a1 a2 a3 a4 : ℝ)

def geometric_progression : Prop := a2^2 = a1 * a3
def arithmetic_progression : Prop := 2 * a3 = a2 + a4
def sum_first_last : Prop := a1 + a4 = 21
def sum_middle : Prop := a2 + a3 = 18

theorem find_numbers :
  geometric_progression a1 a2 a3 a4 ∧
  arithmetic_progression a1 a2 a3 a4 ∧
  sum_first_last a1 a2 a3 a4 ∧
  sum_middle a1 a2 a3 a4 →
  (a1 = 3 ∧ a2 = 6 ∧ a3 = 12 ∧ a4 = 18) ∨
  (a1 = 18.75 ∧ a2 = 11.25 ∧ a3 = 6.75 ∧ a4 = 2.25) :=
by sorry

end find_numbers_l753_753627


namespace only_polynomial_of_form_x_plus_a0_l753_753275

theorem only_polynomial_of_form_x_plus_a0 (P : ℚ[X]) (h1 : P.leading_coeff = 1)
  (h2 : ∃ᶠ a in filter.at_top, irrational a ∧ P.eval a ∈ ℤ ∧ P.eval a > 0) :
  ∃ a0 : ℤ, P = X + C a0 := sorry

end only_polynomial_of_form_x_plus_a0_l753_753275


namespace circumcircle_radius_min_cosA_l753_753695

noncomputable def circumcircle_radius (a b c : ℝ) (A B C : ℝ) :=
  a / (2 * (Real.sin A))

theorem circumcircle_radius_min_cosA
  (a b c A B C : ℝ)
  (h1 : a = 2)
  (h2 : Real.sin C + Real.sin B = 4 * Real.sin A)
  (h3 : a^2 + b^2 - 2 * a * b * (Real.cos A) = c^2)
  (h4 : a^2 + c^2 - 2 * a * c * (Real.cos B) = b^2)
  (h5 : b^2 + c^2 - 2 * b * c * (Real.cos C) = a^2) :
  circumcircle_radius a b c A B C = 8 * Real.sqrt 15 / 15 :=
sorry

end circumcircle_radius_min_cosA_l753_753695


namespace find_a_value_l753_753669

theorem find_a_value (a : ℝ) (m : ℝ) (f g : ℝ → ℝ)
  (f_def : ∀ x, f x = Real.log x / Real.log a)
  (g_def : ∀ x, g x = (2 + m) * Real.sqrt x)
  (a_pos : 0 < a) (a_neq_one : a ≠ 1)
  (max_f : ∀ x ∈ Set.Icc (1 / 2) 16, f x ≤ 4)
  (min_f : ∀ x ∈ Set.Icc (1 / 2) 16, m ≤ f x)
  (g_increasing : ∀ x y, 0 < x → x < y → g x < g y):
  a = 2 :=
sorry

end find_a_value_l753_753669


namespace gcd_lcm_coprime_l753_753419

theorem gcd_lcm_coprime (a : ℕ → ℕ) (b : ℕ → ℕ) (k : ℕ) 
  (h_coprime : ∀ i : ℕ, i < k → Nat.coprime (a i) (b i))
  (h_lcm_m : ∀ m, m = Nat.lcm (finset.range k) b) :
  Nat.gcd (list.range k).map (λ i, a i * (m / b i)) = Nat.gcd (list.range k).map a := 
by 
  sorry

end gcd_lcm_coprime_l753_753419


namespace hyperbola_focal_distance_correct_l753_753684

noncomputable def hyperbola_focal_distance (a b : ℝ) (ha : a > 0) (hb : b > 0) (eccentricity : ℝ) : ℝ
  := 2 * (a * (eccentricity))

theorem hyperbola_focal_distance_correct (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : a^2 + b^2 = c^2)
  (eccentricity : ℝ) (hecc : eccentricity = sqrt 5 / 2) 
  (area : ℝ) (harea : area = 8 / 3)
  (right_focus_line : ℝ) (hright_focus_line: right_focus_line = a) : 
  hyperbola_focal_distance a b ha hb (sqrt 5 / 2) = 2 * sqrt 5 := 
sorry

end hyperbola_focal_distance_correct_l753_753684


namespace value_of_expression_l753_753246

-- Define the sequence p(n) with its recurrence relation and initial conditions
def p : ℕ → ℕ 
| 0       := 1  -- not given, but we need to define it for n-4 when n <= 3
| 1       := 2
| 2       := 4
| 3       := 7
| 4       := 13
| (n + 4) := p n + p (n + 1) + p (n + 2) + p (n + 3)

theorem value_of_expression : 
  (p 2004 - p 2002 - p 1999) / (p 2001 + p 2000) = 2 :=
by
  sorry

end value_of_expression_l753_753246


namespace train_speed_in_km_per_h_l753_753920

noncomputable def length_of_train : ℝ := 110
noncomputable def time_to_cross_platform : ℝ := 7.499400047996161
noncomputable def length_of_platform : ℝ := 165

noncomputable def total_distance_covered : ℝ := length_of_train + length_of_platform
noncomputable def speed_in_m_per_s : ℝ := total_distance_covered / time_to_cross_platform
noncomputable def speed_in_km_per_h : ℝ := speed_in_m_per_s * 3.6

theorem train_speed_in_km_per_h : abs (speed_in_km_per_h - 132.01) < 0.01 :=
by
  unfold length_of_train time_to_cross_platform length_of_platform total_distance_covered speed_in_m_per_s speed_in_km_per_h
  sorry

end train_speed_in_km_per_h_l753_753920


namespace limit_proof_l753_753611

noncomputable def limit_example : Prop :=
  let f := λ x : ℝ, (2 + cos x * sin (2 / (2 * x - π))) / (3 + 2 * x * sin x)
  limit (λ x, f x) (𝓝 (π / 2)) = 𝓝 (2 / (3 + π))

theorem limit_proof : limit_example :=
begin
  let f := λ x : ℝ, (2 + cos x * sin (2 / (2 * x - π))) / (3 + 2 * x * sin x),
  suffices : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - π/2) < δ -> abs (f x - 2/(3 + π)) < ε,
    from real.tendsto_nhds_nhds this,
  sorry
end

end limit_proof_l753_753611


namespace ducks_remaining_after_three_nights_l753_753845

theorem ducks_remaining_after_three_nights: 
  let initial_ducks := 500 in
  let ducks_eaten := initial_ducks * (1/5) in
  let after_first_night := initial_ducks - ducks_eaten in
  let ducks_flew_away := Int.sqrt after_first_night in
  let after_second_night := after_first_night - ducks_flew_away in
  let ducks_stolen := after_second_night * 35 / 100 in
  let after_third_night_theft := after_second_night - ducks_stolen in
  let ducks_returned := after_third_night_theft * (1/5) in
  let final_ducks := after_third_night_theft + Int.floor ducks_returned in
  final_ducks = 296 :=
by
  sorry

end ducks_remaining_after_three_nights_l753_753845


namespace slope_of_line_l_l753_753725

-- Define the parametric equations of the curve C
def curveC (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, 4 * Real.sin θ)

-- Define the parametric equations of the line l
def lineL (t α : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos α, 2 + t * Real.sin α)

-- Define the Cartesian equation of the curve C
def cartesianCurveC (x y : ℝ) : Prop :=
  (y^2 / 16) + (x^2 / 4) = 1

-- Define the Cartesian equation of the line l
def cartesianLineL (x y α : ℝ) : Prop :=
  x * Real.sin α - y * Real.cos α + 2 * Real.cos α - Real.sin α = 0

-- The theorem statement
theorem slope_of_line_l (α : ℝ)
  (H1 : cartesianCurveC (2 * Real.cos α) (4 * Real.sin α))
  (H2 : cartesianLineL 1 2 α)
  (MidpointCondition: (1 : ℝ, 2 : ℝ) = (1, 2)) :
  Real.tan α = -2 :=
sorry

end slope_of_line_l_l753_753725


namespace equivalent_expression_l753_753822

variable (x y z : ℝ)

def P := x + y + z
def Q := x - y - z

theorem equivalent_expression : 
    (P + Q) / (P - Q) - (P - Q) / (P + Q) = (x^2 - y^2 - 2 * y * z - z^2) / (x * (y + z)) :=
by
  sorry

end equivalent_expression_l753_753822


namespace ellipse_triangle_properties_l753_753087

noncomputable def ellipse {R : Type*} [Ring R] (a b : R) (x y : R) := (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def foci_of_ellipse {R : Type*} [LinearOrderedField R] {a b : R} (h : a > b) := (sqrt (a^2 - b^2))

noncomputable def triangle_perimeter {R : Type*} [LinearOrderedField R] (x y z : R) := x + y + z

noncomputable def triangle_area {R : Type*} [LinearOrderedField R] (base height : R) := (1 / 2) * base * height

theorem ellipse_triangle_properties 
  {R : Type*} [LinearOrderedField R] 
  {x y : R}
  (h1 : ellipse 2 (sqrt 3) x y)
  (h2 : let c := (foci_of_ellipse (show (2 : R) > sqrt 3, from sorry)) in 
        triangle_perimeter x y (2*c) = 6)
  (h3 : let c := (foci_of_ellipse (show (2 : R) > sqrt 3, from sorry)) in 
        let area := triangle_area 4 (sqrt 3 / 2) in area = sqrt 3) 
: (h2 = 6) ∧ (h3 = sqrt 3) := 
sorry

end ellipse_triangle_properties_l753_753087


namespace max_A_leq_abs_r2_l753_753772

def f (r_2 r_3 x : ℝ) : ℝ := x^2 - r_2 * x + r_3

noncomputable def g_seq (r_2 r_3 : ℝ) : ℕ → ℝ 
| 0 := 0
| (n + 1) := f r_2 r_3 (g_seq r_2 r_3 n)

variables {r_2 r_3 : ℝ}

theorem max_A_leq_abs_r2 : 
  (∀ (r_2 r_3 : ℝ), 
    (∀ i : ℕ, i <= 2011 -> (g_seq r_2 r_3 (2 * i) < g_seq r_2 r_3 (2 * i + 1) ∧ g_seq r_2 r_3 (2 * i + 1) > g_seq r_2 r_3 (2 * i + 2))) ∧ 
    (∃ j : ℕ, ∀ i > j, g_seq r_2 r_3 (i + 1) > g_seq r_2 r_3 i) ∧
    (∀ bound : ℝ, ∃ n : ℕ, g_seq r_2 r_3 n > bound) 
  ) → ∀ A : ℝ, A ≤ abs r_2 → A ≤ 2 := 
sorry

end max_A_leq_abs_r2_l753_753772


namespace smallest_circle_area_l753_753128

theorem smallest_circle_area
(a b c : ℝ)
(h1 : a = 2)
(h2 : b = 2 * Real.sqrt 2)
(h3 : c = Real.sqrt 2 + Real.sqrt 6) :
  (2 + Real.sqrt 3) * Real.pi = 
  let R := c / 2
  let A := Real.pi * (R ^ 2) in A :=
sorry

end smallest_circle_area_l753_753128


namespace max_soap_boxes_in_carton_l753_753190

theorem max_soap_boxes_in_carton :
  ∀ (V_carton V_soap_box : ℕ), 
  V_carton = 25 * 42 * 60 → 
  V_soap_box = 7 * 6 * 6 → 
  V_carton / V_soap_box = 250 := 
by
  intros V_carton V_soap_box h_V_Carton h_V_Soap_box
  rw [h_V_Carton, h_V_Soap_box]
  norm_num
  sorry

end max_soap_boxes_in_carton_l753_753190


namespace fifth_term_of_geometric_sequence_l753_753838

theorem fifth_term_of_geometric_sequence
  (a r : ℝ)
  (h1 : a * r^2 = 16)
  (h2 : a * r^6 = 2) : a * r^4 = 8 :=
sorry

end fifth_term_of_geometric_sequence_l753_753838


namespace line_equation_l753_753628

theorem line_equation :
  ∃ (L : ℝ → ℝ → Prop),
    (L 2 3) ∧
    (∃ P : ℝ × ℝ, (P.1 - 4 * P.2 - 1 = 0) ∧ 
                    ((2 * P.1 - 5 * P.2 + 9) / real.sqrt (2^2 + 5^2) = 
                     (2 * P.1 - 5 * P.2 - 7) / real.sqrt (2^2 + 5^2))) ∧
    (∃ A B : ℝ × ℝ, 
      A ≠ B ∧ 
      ((A.1 + B.1)/2, (A.2 + B.2)/2) = P ∧ 
      L A.1 A.2 ∧ L B.1 B.2) ∧
  ∀ x y : ℝ, L x y ↔ (4 * x - 5 * y + 7 = 0) := sorry

end line_equation_l753_753628


namespace commutative_not_associative_l753_753973

variable (k : ℝ) (h_k : 0 < k)

noncomputable def star (x y : ℝ) : ℝ := (x * y + k) / (x + y + k)

theorem commutative (x y : ℝ) (h_x : 0 < x) (h_y : 0 < y) :
  star k x y = star k y x :=
by sorry

theorem not_associative (x y z : ℝ) (h_x : 0 < x) (h_y : 0 < y) (h_z : 0 < z) :
  ¬(star k (star k x y) z = star k x (star k y z)) :=
by sorry

end commutative_not_associative_l753_753973


namespace a2b2_div_ab1_is_square_l753_753988

theorem a2b2_div_ab1_is_square (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : (ab + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, (a^2 + b^2) / (ab + 1) = k^2 :=
sorry

end a2b2_div_ab1_is_square_l753_753988


namespace monika_spending_l753_753084

variable (x : ℝ) (y : ℝ) (z_cucumbers : ℕ) (z_tomatoes : ℕ) (z_pineapples : ℕ)

def total_spent (x y : ℝ) (z_cucumbers z_tomatoes z_pineapples : ℕ) : ℝ :=
  let clothes_original := 250
  let movies_original := 24 * 3
  let clothes_discount := clothes_original * (x / 100)
  let movies_discount := movies_original * (y / 100)
  let clothes_cost := clothes_original - clothes_discount
  let movies_cost := movies_original - movies_discount
  let beans_cost := 20 * 1.25
  let cucumbers_cost := z_cucumbers * 2.50
  let tomatoes_cost := z_tomatoes * 5.00
  let pineapples_cost := z_pineapples * 6.50
  let farmers_market_cost := beans_cost + cucumbers_cost + tomatoes_cost + pineapples_cost
  clothes_cost + movies_cost + farmers_market_cost

theorem monika_spending :
  total_spent 15 10 5 3 2 = 342.80 :=
by
  sorry

end monika_spending_l753_753084


namespace sum_series_fraction_equal_l753_753958

theorem sum_series_fraction_equal :
  (∑ n in Finset.range 15, 1 / ((n + 1) * (n + 2))) = 15 / 16 := 
by
  sorry

end sum_series_fraction_equal_l753_753958


namespace max_A_leq_abs_r2_l753_753773

def f (r_2 r_3 x : ℝ) : ℝ := x^2 - r_2 * x + r_3

noncomputable def g_seq (r_2 r_3 : ℝ) : ℕ → ℝ 
| 0 := 0
| (n + 1) := f r_2 r_3 (g_seq r_2 r_3 n)

variables {r_2 r_3 : ℝ}

theorem max_A_leq_abs_r2 : 
  (∀ (r_2 r_3 : ℝ), 
    (∀ i : ℕ, i <= 2011 -> (g_seq r_2 r_3 (2 * i) < g_seq r_2 r_3 (2 * i + 1) ∧ g_seq r_2 r_3 (2 * i + 1) > g_seq r_2 r_3 (2 * i + 2))) ∧ 
    (∃ j : ℕ, ∀ i > j, g_seq r_2 r_3 (i + 1) > g_seq r_2 r_3 i) ∧
    (∀ bound : ℝ, ∃ n : ℕ, g_seq r_2 r_3 n > bound) 
  ) → ∀ A : ℝ, A ≤ abs r_2 → A ≤ 2 := 
sorry

end max_A_leq_abs_r2_l753_753773


namespace irreducible_fraction_unique_l753_753623

theorem irreducible_fraction_unique :
  ∃ (a b : ℕ), a = 5 ∧ b = 2 ∧ gcd a b = 1 ∧ (∃ n : ℕ, 10^n = a * b) :=
by
  sorry

end irreducible_fraction_unique_l753_753623


namespace sum_of_three_numbers_l753_753497

theorem sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : b = 10) 
  (h2 : (a + b + c) / 3 = a + 20) 
  (h3 : (a + b + c) / 3 = c - 25) : 
  a + b + c = 45 := 
by 
  sorry

end sum_of_three_numbers_l753_753497


namespace solve_quadratic_eq_l753_753805

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 * x = 2 ↔ (x = 2 + Real.sqrt 6) ∨ (x = 2 - Real.sqrt 6) :=
by
  sorry

end solve_quadratic_eq_l753_753805


namespace jill_llamas_count_l753_753056

theorem jill_llamas_count : 
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  herd_after_sell = 18 := 
by
  -- Definitions for the conditions
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  -- Proof will be filled in here.
  sorry

end jill_llamas_count_l753_753056


namespace volleyball_team_lineup_l753_753593

theorem volleyball_team_lineup :
  ∃ (choices : ℕ), choices = 18 * (Nat.choose 17 7) ∧ choices = 350064 :=
by
  use 18 * (Nat.choose 17 7)
  split
  · rfl
  · sorry

end volleyball_team_lineup_l753_753593


namespace area_of_circle_l753_753790

-- Lean 4 statement

theorem area_of_circle (A B : ℝ × ℝ) (P : ℝ × ℝ)
  (hA : A = (8, 15)) (hB : B = (16, 9)) (hP : P = (3, 0)) 
  (h_tangent_A : tangent_line_to_circle A ω meets_at P) 
  (h_tangent_B : tangent_line_to_circle B ω meets_at P)
  : area ω = 250 * Real.pi := 
sorry

end area_of_circle_l753_753790


namespace surface_area_of_larger_prism_l753_753221

def volume_of_brick := 288
def number_of_bricks := 11
def target_surface_area := 1368

theorem surface_area_of_larger_prism
    (vol: ℕ := volume_of_brick)
    (num: ℕ := number_of_bricks)
    (target: ℕ := target_surface_area)
    (exists_a_b_h : ∃ (a b h : ℕ), a = 12 ∧ b = 8 ∧ h = 3)
    (large_prism_dimensions : ∃ (L W H : ℕ), L = 24 ∧ W = 12 ∧ H = 11):
    2 * (24 * 12 + 24 * 11 + 12 * 11) = target :=
by
  sorry

end surface_area_of_larger_prism_l753_753221


namespace geometric_sum_condition_l753_753639

variable {a₁ q : ℝ}
-- Let Sₙ be the sum of the first n terms of a geometric series
def S (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

-- The given condition in the problem
theorem geometric_sum_condition (h : 2 * S 4 = S 5 + S 6) : q = -2 :=
sorry

end geometric_sum_condition_l753_753639


namespace number_of_solutions_l753_753966

theorem number_of_solutions (x : ℕ) :
  (∀ x: ℕ, ⌊x / 10⌋ = ⌊x / 11⌋ + 1 -> x) -> 
  (∃ n: ℕ, n = 110) :=
sorry

end number_of_solutions_l753_753966


namespace energy_consumption_correct_l753_753203

def initial_wattages : List ℕ := [60, 80, 100, 120]

def increased_wattages : List ℕ := initial_wattages.map (λ x => x + (x * 25 / 100))

def combined_wattage (ws : List ℕ) : ℕ := ws.sum

def daily_energy_consumption (cw : ℕ) : ℕ := cw * 6 / 1000

def total_energy_consumption (dec : ℕ) : ℕ := dec * 30

-- Main theorem statement
theorem energy_consumption_correct :
  total_energy_consumption (daily_energy_consumption (combined_wattage increased_wattages)) = 81 := 
sorry

end energy_consumption_correct_l753_753203


namespace part1_part2_l753_753735

-- Definitions based on the given conditions
variable (A B C I D E F : Type)
variable [IsIncenter ABC I]
variable [Perpendicular I BC D]
variable [Perpendicular I CA E]
variable [Perpendicular I AB F]

-- Main statements to prove
theorem part1 : 
  let incircle := Incircle I in 
  inversion_of_sides_is_circles_with_diameters ABC I incircle ID IE IF :=
sorry

theorem part2 : 
  all_intersections_on_AI_BI_CI ABC I D E F :=
sorry

end part1_part2_l753_753735


namespace product_of_num_and_sum_l753_753425

-- Define the function f : ℝ → ℝ with the given property
def f (x : ℝ) : ℝ := sorry 

-- Define the condition given in the problem
def condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f((x - y)^3) = f(x)^3 - 3*x*f(y^2) + y^3

-- Define the number of possible values for f(1), sum of all possible values
def number_of_values (f : ℝ → ℝ) : ℕ := sorry
def sum_of_values (f : ℝ → ℝ) : ℝ := sorry

-- Stating the theorem with the given conditions
theorem product_of_num_and_sum (f : ℝ → ℝ)
  (h : condition f) :
  number_of_values f = 2 ∧ sum_of_values f = 3 ∧ number_of_values f * sum_of_values f = 6 :=
begin
  sorry
end

end product_of_num_and_sum_l753_753425


namespace ratio_of_squares_l753_753240

theorem ratio_of_squares :
  (1625^2 - 1612^2) / (1631^2 - 1606^2) = 13 / 25 := by
  have h_num : 1625^2 - 1612^2 = 13 * 3237 := by sorry
  have h_den : 1631^2 - 1606^2 = 25 * 3237 := by sorry
  rw [h_num, h_den]
  exact div_eq_div_of_mul_eq_mul_right (ne_of_gt (show 3237 > 0 by sorry)) (by sorry)

end ratio_of_squares_l753_753240


namespace total_boys_in_camp_l753_753024

-- Definitions based on the conditions in a)
def total_boys := ℕ
def from_school_A (T : total_boys) := 0.20 * T
def study_science (A : total_boys) := 0.30 * A
def not_study_science (A : total_boys) := 0.70 * A

-- The given number of boys from school A but not studying science
def boys_not_study_science := 21

-- The problem statement to be proved
theorem total_boys_in_camp (T : total_boys) : 
  (not_study_science (from_school_A T) = boys_not_study_science) →
  T = 150 :=
by
  intro h
  simp [not_study_science, from_school_A] at h
  sorry

end total_boys_in_camp_l753_753024


namespace find_k_perpendicular_lines_l753_753311

theorem find_k_perpendicular_lines (k : ℚ) :
  let A := (-2 : ℚ, 0 : ℚ),
      B := (0 : ℚ, -6 : ℚ),
      X := (0 : ℚ, 12 : ℚ),
      Y := (10 : ℚ, k : ℚ) in
  let slope (p1 p2 : ℚ × ℚ) := (p2.snd - p1.snd) / (p2.fst - p1.fst) in
  slope A B * slope X Y = -1 → k = 26 / 3 :=
by
  intros; sorry

end find_k_perpendicular_lines_l753_753311


namespace count_valid_four_digit_numbers_l753_753001

-- Definitions and conditions
def valid_digits : Set ℕ := {2, 5, 7}

def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (∀ d ∈ n.digits, d ∈ valid_digits)

-- Statement of the problem
theorem count_valid_four_digit_numbers : 
  {n : ℕ | is_valid_four_digit_number n}.to_finset.card = 81 :=
by 
  sorry

end count_valid_four_digit_numbers_l753_753001


namespace joann_lollipops_l753_753411

theorem joann_lollipops :
  ∃ (a : ℝ), 
  (a + (a + 5) + (a + 10) + (a + 15) + (a + 20) + (a + 25) = 150) → 
  (a + 15 = 27.5) := 
begin
  sorry
end

end joann_lollipops_l753_753411


namespace value_range_sin_neg_l753_753839

theorem value_range_sin_neg (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) : 
  Set.Icc (-1) (Real.sqrt 2 / 2) ( - (Real.sin x) ) :=
sorry

end value_range_sin_neg_l753_753839


namespace probability_normal_distribution_phi_l753_753369

noncomputable def standard_normal_distribution := sorry

theorem probability_normal_distribution_phi (ξ : ℝ) (x : ℝ)
  (hξ : ξ ~ N(0,1)) :
  P(ξ < x) = Φ(x) :=
sorry

end probability_normal_distribution_phi_l753_753369


namespace sale_in_second_month_l753_753909

theorem sale_in_second_month
  (sale1 sale3 sale4 sale5 sale6 : ℕ)
  (average_sale : ℕ)
  (total_months : ℕ)
  (h_sale1 : sale1 = 5420)
  (h_sale3 : sale3 = 6200)
  (h_sale4 : sale4 = 6350)
  (h_sale5 : sale5 = 6500)
  (h_sale6 : sale6 = 6470)
  (h_average_sale : average_sale = 6100)
  (h_total_months : total_months = 6) :
  ∃ sale2 : ℕ, sale2 = 5660 := 
by
  sorry

end sale_in_second_month_l753_753909


namespace transform_fraction_l753_753486

theorem transform_fraction (x : ℝ) (h : x ≠ 1) : - (1 / (1 - x)) = 1 / (x - 1) :=
by
  sorry

end transform_fraction_l753_753486


namespace plane_divides_area_in_ratio_l753_753717

variable (A B C D K L M N : Type)
variable [Tetrahedron A B C D]
variable {area : Type → ℝ}

variable (P : Plane)
variable (intersections : P.intersects_with_edges [A, B, D, C])
hypothesis (h1 : area (triangle A K N) = (1/5) * area (face A B D))
hypothesis (h2 : area (triangle K B L) = (1/2) * area (face A B C))
hypothesis (h3 : area (triangle N D M) = (1/5) * area (face D B C))

theorem plane_divides_area_in_ratio :
  ratio (area (triangle B C D), area (triangle P ∩ face B C D)) = 1 / 7 :=
sorry

end plane_divides_area_in_ratio_l753_753717


namespace find_t_l753_753313

noncomputable def vectors_are_nonzero (m n : ℝ^3) : Prop :=
  m ≠ 0 ∧ n ≠ 0

noncomputable def magnitude_relation (m n : ℝ^3) : Prop :=
  ∥m∥ = 2 * ∥n∥

noncomputable def cosine_relation (m n : ℝ^3) : Prop :=
  (m • n) = (∥m∥ * ∥n∥ * (1 / 3))

noncomputable def orthogonality_condition (m n : ℝ^3) (t : ℝ) : Prop :=
  m • (t • n + m) = 0

theorem find_t (m n : ℝ^3) (t : ℝ) 
  (h1 : vectors_are_nonzero m n)
  (h2 : magnitude_relation m n)
  (h3 : cosine_relation m n)
  (h4 : orthogonality_condition m n t) 
  : t = -6 := 
sorry

end find_t_l753_753313


namespace distance_A1C1_to_base_l753_753308

variables (A B C D A1 B1 C1 D1 : Point)

-- Define the right square prism and its properties
def is_right_square_prism (A B C D A1 B1 C1 D1 : Point) : Prop :=
  is_square A B C D ∧
  is_square A1 B1 C1 D1 ∧
  parallel (plane A B C) (plane A1 B1 C1 D1) ∧
  height_of_prism A B A1 = sqrt 3
  
-- Define the base edge length
def base_edge_length_is_one (A B C D : Point) : Prop :=
  distance A B = 1 ∧ distance B C = 1 ∧ distance C D = 1 ∧ distance D A = 1

-- Define the angle condition
def angle_condition (A1 B B1 : Point) : Prop :=
  angle A1 B B1 = 60

-- Proof Statement
theorem distance_A1C1_to_base :
  ∀ (A B C D A1 B1 C1 D1 : Point),
    is_right_square_prism A B C D A1 B1 C1 D1 ∧
    base_edge_length_is_one A B C D ∧
    angle_condition A B B1 →
    distance (line A1 C1) (plane A B C) = sqrt 3 :=
by
  intros
  sorry  -- proof not required

end distance_A1C1_to_base_l753_753308


namespace find_f_of_half_l753_753005

theorem find_f_of_half :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f (sin x) = 1 - 2 * (sin x)^2) →
  f (1 / 2) = 1 / 2 :=
by
  intro h
  sorry

end find_f_of_half_l753_753005


namespace license_plate_combinations_l753_753224

open Nat

theorem license_plate_combinations : 
  (∃ (choose_two_letters: ℕ) (place_first_letter: ℕ) (place_second_letter: ℕ) (choose_non_repeated: ℕ)
     (first_digit: ℕ) (second_digit: ℕ) (third_digit: ℕ),
    choose_two_letters = choose 26 2 ∧
    place_first_letter = choose 5 2 ∧
    place_second_letter = choose 3 2 ∧
    choose_non_repeated = 24 ∧
    first_digit = 10 ∧
    second_digit = 9 ∧
    third_digit = 8 ∧
    choose_two_letters * place_first_letter * place_second_letter * choose_non_repeated * first_digit * second_digit * third_digit = 56016000) :=
sorry

end license_plate_combinations_l753_753224


namespace surface_area_cylinder_l753_753565

-- Definitions based on conditions
def diameter := 4
def height := 4
def radius := diameter / 2
def surface_area := 2 * Real.pi * radius * (radius + height)

-- Theorem statement based on question and correct answer
theorem surface_area_cylinder : surface_area = 24 * Real.pi :=
by
  sorry

end surface_area_cylinder_l753_753565


namespace shaded_region_area_l753_753509

theorem shaded_region_area
  (n : ℕ) (d : ℝ) 
  (h₁ : n = 25) 
  (h₂ : d = 10) 
  (h₃ : n > 0) : 
  (d^2 / n = 2) ∧ (n * (d^2 / (2 * n)) = 50) :=
by 
  sorry

end shaded_region_area_l753_753509


namespace circumcircles_intersection_on_segment_AB_l753_753777

noncomputable def centroid (A B C : Point) : Point := 
  (A + B + C) / 3

variable {A B C P Q G : Point}

theorem circumcircles_intersection_on_segment_AB
  (h_right_angle : \angle A B C = 90°)
  (h_centroid : G = centroid A B C)
  (h_P_on_AG : P.on_ray (AG) ∧ \angle CPA = \angle BAC)
  (h_Q_on_BG : Q.on_ray (BG) ∧ \angle CQB = \angle CBA) :
  ∃ (K : Point), K ∈ segment (A, B) ∧ IsCocircular K A Q G ∧ IsCocircular K B P G := by
  sorry

end circumcircles_intersection_on_segment_AB_l753_753777


namespace mixture_weight_l753_753130

theorem mixture_weight :
  let weight_a_per_liter := 900 -- in gm
  let weight_b_per_liter := 750 -- in gm
  let ratio_a := 3
  let ratio_b := 2
  let total_volume := 4 -- in liters
  let volume_a := (ratio_a / (ratio_a + ratio_b)) * total_volume
  let volume_b := (ratio_b / (ratio_a + ratio_b)) * total_volume
  let weight_a := volume_a * weight_a_per_liter
  let weight_b := volume_b * weight_b_per_liter
  let total_weight_gm := weight_a + weight_b
  let total_weight_kg := total_weight_gm / 1000 
  total_weight_kg = 3.36 :=
by
  sorry

end mixture_weight_l753_753130


namespace parity_of_function_parity_neither_odd_nor_even_l753_753256

def f (x p : ℝ) : ℝ := x * |x| + p * x^2

theorem parity_of_function (p : ℝ) :
  (∀ x : ℝ, f x p = - f (-x) p) ↔ p = 0 :=
by
  sorry

theorem parity_neither_odd_nor_even (p : ℝ) :
  (∀ x : ℝ, f x p ≠ f (-x) p) ∧ (∀ x : ℝ, f x p ≠ - f (-x) p) ↔ p ≠ 0 :=
by
  sorry

end parity_of_function_parity_neither_odd_nor_even_l753_753256


namespace twelfth_term_geometric_sequence_l753_753531

theorem twelfth_term_geometric_sequence :
  let a1 := 5
  let r := (2 / 5 : ℝ)
  (a1 * r ^ 11) = (10240 / 48828125 : ℝ) :=
by
  sorry

end twelfth_term_geometric_sequence_l753_753531


namespace unique_zero_in_0_1_l753_753122

def f (x : ℝ) : ℝ := 2^x + x^3 - 2

def strictly_increasing_in_0_1 : Prop :=
  ∀ x y, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x < y → f x < f y

def f_0_f_1_negative_product : Prop := f 0 * f 1 < 0

theorem unique_zero_in_0_1 (h1 : strictly_increasing_in_0_1) (h2 : f_0_f_1_negative_product) : 
  ∃! c : ℝ, 0 < c ∧ c < 1 ∧ f c = 0 := 
sorry

end unique_zero_in_0_1_l753_753122


namespace range_of_b_plus_c_l753_753489

noncomputable def func (b c x : ℝ) : ℝ := x^2 + b*x + c * 3^x

theorem range_of_b_plus_c {b c : ℝ} (h1 : ∃ x, func b c x = 0)
  (h2 : ∀ x, (func b c x = 0 ↔ func b c (func b c x) = 0)) : 
  0 ≤ b + c ∧ b + c < 4 :=
by
  sorry

end range_of_b_plus_c_l753_753489


namespace smallest_divisible_by_15_11_12_l753_753283

theorem smallest_divisible_by_15_11_12 : ∃ n : ℕ, (n > 0) ∧ (15 ∣ n) ∧ (11 ∣ n) ∧ (12 ∣ n) ∧ (∀ m : ℕ, (m > 0) ∧ (15 ∣ m) ∧ (11 ∣ m) ∧ (12 ∣ m) → n ≤ m) ∧ n = 660 :=
by
  sorry

end smallest_divisible_by_15_11_12_l753_753283


namespace conic_section_is_parabola_l753_753259

theorem conic_section_is_parabola
    (x y : ℝ)
    (h : abs (x - 3) = sqrt ((y + 4) ^ 2 + x ^ 2)) :
    "P" =
    if ∃ a b c : ℝ, (y ^ 2 + a * y + b = c * x) then
      "P"
    else
      "N" :=
by
  sorry

end conic_section_is_parabola_l753_753259


namespace shaded_area_correct_l753_753610

def diameter := 3 -- inches
def pattern_length := 18 -- inches equivalent to 1.5 feet

def radius := diameter / 2 -- radius calculation

noncomputable def area_of_one_circle := Real.pi * (radius ^ 2)
def number_of_circles := pattern_length / diameter
noncomputable def total_shaded_area := number_of_circles * area_of_one_circle

theorem shaded_area_correct :
  total_shaded_area = 13.5 * Real.pi :=
  by
  sorry

end shaded_area_correct_l753_753610


namespace sequence_value_a6_a12_equals_34_l753_753668

theorem sequence_value_a6_a12_equals_34 :
  (∀ n > 1, a (n + 1) - a n = a n - a (n - 1)) ∧ a 1 = 1 ∧ S 3 + S 5 = 2 * a 9 →
  a 6 + a 12 = 34 :=
by
  sorry

end sequence_value_a6_a12_equals_34_l753_753668


namespace probability_two_students_next_to_each_other_l753_753379

theorem probability_two_students_next_to_each_other : (2 * Nat.factorial 9) / Nat.factorial 10 = 1 / 5 :=
by
  sorry

end probability_two_students_next_to_each_other_l753_753379


namespace opposite_of_neg3_l753_753829

theorem opposite_of_neg3 : ∃ y : ℝ, -3 + y = 0 ∧ y = 3 := 
by
  use 3
  split
  . ring
  . rfl

end opposite_of_neg3_l753_753829


namespace product_of_roots_l753_753704

theorem product_of_roots (x : ℝ) :
  (x+3) * (x-4) = 22 →
  let a := 1 in
  let b := -1 in
  let c := -34 in
  a * x^2 + b * x + c = 0 → 
  (c / a) = -34 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end product_of_roots_l753_753704


namespace tan_of_neg_23_over_3_pi_l753_753129

theorem tan_of_neg_23_over_3_pi : (Real.tan (- 23 / 3 * Real.pi) = Real.sqrt 3) :=
by
  sorry

end tan_of_neg_23_over_3_pi_l753_753129


namespace distance_after_3rd_turn_l753_753568

theorem distance_after_3rd_turn (d1 d2 d4 total_distance : ℕ) 
  (h1 : d1 = 5) 
  (h2 : d2 = 8) 
  (h4 : d4 = 0) 
  (h_total : total_distance = 23) : 
  total_distance - (d1 + d2 + d4) = 10 := 
  sorry

end distance_after_3rd_turn_l753_753568


namespace binary_to_quaternary_l753_753613

theorem binary_to_quaternary : ∀ (b : String), 
  b = "11100" → (∀ (d : Nat), nat_of_bin b = d → (∀ (q : String), nat_to_quat d = q → q = "130")) :=
by
  intros b b_eq_11100 d d_eq_28 q q_eq_130
  sorry

def nat_of_bin (b : String) : Nat := 
  b.foldl (λ acc bit, 2 * acc + if bit = '1' then 1 else 0) 0

def nat_to_quat (n : Nat) : String := 
  if n = 0 then "0"
  else
    let rec aux (n : Nat) (acc : String) : String :=
      if n = 0 then acc else aux (n / 4) (String.push acc (Char.ofNat (48 + n % 4)))
    (aux n "").reverse

#eval nat_of_bin "11100"  -- Expected: 28
#eval nat_to_quat 28      -- Expected: "130"

end binary_to_quaternary_l753_753613


namespace find_common_ratio_l753_753127

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
∀ n, S n = (finset.range n).sum a

theorem find_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : sum_of_first_n_terms a S)
  (h3 : a 2 + S 3 = 0) :
  q = -1 :=
by sorry

end find_common_ratio_l753_753127


namespace lim_ln_f_div_ln_m_l753_753315

noncomputable def f (k m : ℕ) : ℕ :=
  sorry -- Definition of f(m) in Lean; placeholder since full definition needs proof details

theorem lim_ln_f_div_ln_m (k : ℕ) (h_k : k ≥ 2) :
  filter.tendsto (λ m : ℕ, (Real.log (f k m)) / (Real.log m)) filter.at_top (nhds (k / (k + 1) : ℝ)) :=
begin
  sorry -- Placeholder for proof
end

end lim_ln_f_div_ln_m_l753_753315


namespace min_a_and_angle_non_obtuse_l753_753697

-- Definitions of vectors
def m (a b : ℝ) : ℝ × ℝ := (a, b^2 - b + 7/3)
def n (a b : ℝ) : ℝ × ℝ := (a + b + 2, 1)
def mu : ℝ × ℝ := (2, 1)

-- Condition: m is parallel to mu
def m_parallel_mu (a b : ℝ) : Prop := m a b = (2 * (b^2 - b + 7/3), b^2 - b + 7/3)

-- Proof problem statement in Lean

theorem min_a_and_angle_non_obtuse (a b : ℝ) (h : m_parallel_mu a b) :
  (a = 25 / 6) ∧ ((m a b).fst * (n a b).fst + (m a b).snd * (n a b).snd ≥ 0) := 
by
  sorry

end min_a_and_angle_non_obtuse_l753_753697


namespace number_of_second_graders_l753_753131

-- Define the number of kindergartners, first graders, and total students
def k : ℕ := 14
def f : ℕ := 24
def t : ℕ := 42

-- Define the number of second graders
def s : ℕ := t - (k + f)

-- The theorem to prove
theorem number_of_second_graders : s = 4 := by
  -- We can use sorry here since we are not required to provide the proof
  sorry

end number_of_second_graders_l753_753131


namespace f_negative_l753_753323

-- Given conditions
axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom f_positive : ∀ x : ℝ, 0 < x → f(x) = x * (1 - x)

-- Theorem to prove
theorem f_negative (x : ℝ) (h : x ≤ 0) : f(x) = x * (1 + x) :=
by
  sorry

end f_negative_l753_753323


namespace avery_donates_16_clothes_l753_753227

theorem avery_donates_16_clothes : 
  let Shirts := 4
  let Pants := 4 * 2
  let Shorts := (4 * 2) / 2
in Shirts + Pants + Shorts = 16 :=
by 
  let Shirts := 4
  let Pants := 4 * 2
  let Shorts := (4 * 2) / 2
  show Shirts + Pants + Shorts = 16
  sorry

end avery_donates_16_clothes_l753_753227


namespace tan_half_angle_lt_l753_753100

theorem tan_half_angle_lt (x : ℝ) (h : 0 < x ∧ x ≤ π / 2) : 
  Real.tan (x / 2) < x := 
by
  sorry

end tan_half_angle_lt_l753_753100


namespace area_of_triangle_XYZ_l753_753044

noncomputable def triangle_area (XM YN : ℝ) : ℝ :=
  if XM = 10 ∧ YN = 15 then 100 else 0

theorem area_of_triangle_XYZ (XM YN : ℝ) (h_perpendicular : true) (h_XM : XM = 10) (h_YN : YN = 15) :
  triangle_area XM YN = 100 :=
by {
  assume h_perpendicular,
  assume h_XM,
  assume h_YN,
  -- proof goes here
  sorry
}

end area_of_triangle_XYZ_l753_753044


namespace pushover_probability_l753_753776

/-- Given a game setup with 2^(n+1) players numbered from 1 to 2^(n+1) in a circle,
    where each player has a 50% chance of winning each Pushover match, 
    and the tournament unfolds in n+1 rounds with random pairings,
    prove that the probability that players 1 and 2^n face each other in the last round is
    (2^n - 1) / 8^n. -/
theorem pushover_probability (n : ℕ) (h : 0 < n) : 
  ∃ P : ℕ → ℚ, P n = (2^n - 1) / 8^n := 
begin
  let P := λ n : ℕ, (2^n - 1) / 8^n,
  use P,
  intros,
  sorry
end

end pushover_probability_l753_753776


namespace no_family_argument_proportion_l753_753168

noncomputable def probability_no_family_argument (p quarrel_h husband quarreling with mother-in-law) (p_quarrel_w wife quarreling with mother-in-law) : ℝ :=
  let p_h_side_own := 1 / 2
  let p_w_side_own := 1 / 2
  let p_family_argument_h :=
    p quarrel_h husband quarreling with mother-in-law * p_h_side_own
  let p_family_argument_w :=
    p_quarrel_w wife quarreling with mother-in-law * p_w_side_own
  let p_family_argument := p_family_argument_h + p_family_argument_w - p_family_argument_h * p_family_argument_w
  1 - p_family_argument

theorem no_family_argument_proportion :
  probability_no_family_argument (2 / 3) (2 / 3) = 4 / 9 :=
by
  sorry

end no_family_argument_proportion_l753_753168


namespace grapes_purchased_l753_753929

-- Define the given conditions
def price_per_kg_grapes : ℕ := 68
def kg_mangoes : ℕ := 9
def price_per_kg_mangoes : ℕ := 48
def total_paid : ℕ := 908

-- Define the proof problem
theorem grapes_purchased : ∃ (G : ℕ), (price_per_kg_grapes * G + price_per_kg_mangoes * kg_mangoes = total_paid) ∧ (G = 7) :=
by {
  use 7,
  sorry
}

end grapes_purchased_l753_753929


namespace triangle_angle_sum_l753_753375

/-- Proof of the magnitude of angle A and the sum of sides b and c in triangle ABC -/
theorem triangle_angle_sum (a b c S : ℝ) (h1 : a = √3) (h2 : S = √3 / 2) 
  (h3 : ∀ B A, a * sin B = √3 * b * cos A) :
  (∃ A : ℝ, A = π / 3) ∧ ( ∃ b c : ℝ, b + c = 3) :=
by
  sorry

end triangle_angle_sum_l753_753375


namespace avery_donation_l753_753230

theorem avery_donation (shirts pants shorts : ℕ)
  (h_shirts : shirts = 4)
  (h_pants : pants = 2 * shirts)
  (h_shorts : shorts = pants / 2) :
  shirts + pants + shorts = 16 := by
  sorry

end avery_donation_l753_753230


namespace parallel_DE_FG_l753_753423

noncomputable def circ (A B C : Point) : Circle := sorry /-Dummy definition/

variables {A B C D E F G : Point}
variable [Circ : Circle]
variables [TriangleABC : Triangle]
variable [Acute : AcuteTriangle]
variables (onCircle : Incircle TriangleABC Circ)

theorem parallel_DE_FG (Gamma : Circle) 
  (h1 : IsCircumcircle_of TriangleABC Gamma) 
  (h2 : AD = AE)
  (h3 : PerpendicularBisector (BD) intersectArc AB Gamma F)
  (h4 : PerpendicularBisector (CE) intersectArc AC Gamma G) :
  Parallel DE FG :=
sorry  -- Proof omitted.

end parallel_DE_FG_l753_753423


namespace find_cost_find_num_plans_min_cost_l753_753518

-- Define variables and constants
variables (a b : ℕ) -- a: cost of type A, b: cost of type B
variables (x y : ℕ) -- x: amount of type A, y: amount of type B
variable (W : ℕ)   -- W: total cost

-- Given conditions as hypotheses
hypothesis h1 : 2 * a + b = 35
hypothesis h2 : a + 3 * b = 30
hypothesis h3 : x + y = 120
hypothesis h4 : 955 ≤ 15 * x + 5 * y
hypothesis h5 : 15 * x + 5 * y ≤ 1000

-- Part 1: Prove the cost of one item of type A and one item of type B
theorem find_cost : a = 15 ∧ b = 5 :=
by sorry

-- Part 2: Prove that there are 5 different purchasing plans
theorem find_num_plans : ∃ (x : ℕ), 36 ≤ x ∧ x ≤ 40 ∧
                         (∃ (y : ℕ), x + y = 120 ∧ 955 ≤ 15 * x + 5 * y ∧ 15 * x + 5 * y ≤ 1000) :=
by sorry

-- Part 3: Prove the minimum amount of money needed
theorem min_cost (x : ℕ) (hx : 36 ≤ x ∧ x ≤ 40) : W = 960 :=
by sorry

end find_cost_find_num_plans_min_cost_l753_753518


namespace angle_B_45_l753_753405

def triangle.angle_B (α β γ : ℝ) (CH : ℝ) (BC : ℝ) (condition1 : β = 90) (condition2 : CH = α)
  (condition3 : ∀ x y : ℝ, BH = y ∧ HD = y ∧ MD = x) 
  (condition4 : ∀ AC BC : ℝ, AC/BC = (x+y)/y := sorry) 
  (condition5 : ∀ AC BC : ℝ, AC/BC = 2*(x+y)/x := sorry) 
  (condition6 : 4*θ = γ) (condition7 : 0<|γ=90) : ℝ := 
  (BH = HD) ∧ 
  (MD = x)

theorem angle_B_45 : triangle.angle_B α α β γ = 45 := 
  begin
    sorry,  -- Placeholder for proof
  end

end angle_B_45_l753_753405


namespace conic_sections_union_l753_753115

theorem conic_sections_union :
  ∀ (x y : ℝ), (y^4 - 4*x^4 = 2*y^2 - 1) ↔ 
               (y^2 - 2*x^2 = 1) ∨ (y^2 + 2*x^2 = 1) := 
by
  sorry

end conic_sections_union_l753_753115


namespace find_B_l753_753222

-- Define variables for points A, B, C, D, E, F representing numbers from 1 to 6
variables (A B C D E F : ℕ)

-- Condition: all distinct and sum is 21
def distinct_points (A B C D E F : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F ∧
  A + B + C + D + E + F = 21 ∧
  A ∈ {1, 2, 3, 4, 5, 6} ∧ B ∈ {1, 2, 3, 4, 5, 6} ∧
  C ∈ {1, 2, 3, 4, 5, 6} ∧ D ∈ {1, 2, 3, 4, 5, 6} ∧
  E ∈ {1, 2, 3, 4, 5, 6} ∧ F ∈ {1, 2, 3, 4, 5, 6}

-- Line equations
def line_sums (A B C D E F : ℕ) : Prop :=
  let S1 := A + B + C in
  let S2 := A + E + F in
  let S3 := C + D + E in
  let S4 := B + D in
  let S5 := B + F in
  S1 + S2 + S3 + S4 + S5 = 47

-- The proof statement
theorem find_B : distinct_points A B C D E F → line_sums A B C D E F → B = 5 :=
by
  -- The actual proof would go here
  sorry

end find_B_l753_753222


namespace joan_games_l753_753059

theorem joan_games (last_year_games this_year_games total_games : ℕ)
  (h1 : last_year_games = 9)
  (h2 : total_games = 13)
  : this_year_games = total_games - last_year_games → this_year_games = 4 := 
by
  intros h
  rw [h1, h2] at h
  exact h

end joan_games_l753_753059


namespace population_increase_per_year_nearest_hundred_l753_753034

-- Definitions as per conditions
def births_per_day : ℝ := 24 / 6
def deaths_per_day : ℝ := 24 / 30
def net_increase_per_day : ℝ := births_per_day - deaths_per_day
def annual_population_increase : ℝ := net_increase_per_day * 365

-- Lean 4 statement proving that the population increase per year is 1200 people, when rounded to the nearest hundred
theorem population_increase_per_year_nearest_hundred : (Int.round annual_population_increase / 100) * 100 = 1200 :=
  by
    -- The proof would go here
    sorry

end population_increase_per_year_nearest_hundred_l753_753034


namespace probability_is_correct_l753_753178

def probability_grid_complete_black : ℚ := 1 / 65536

def check_grid_black (grid : List (List Bool)) : Prop :=
  ∀ row in grid, ∀ cell in row, cell = tt

noncomputable def probability_completely_black_after_rotations (grid : List (List Bool)) : ℚ :=
  if check_grid_black (rotate_twice_and_repaint grid) 
  then 1
  else 0

axiom rotate_twice_and_repaint : List (List Bool) → List (List Bool)

theorem probability_is_correct (initial_grid : List (List Bool)) (h : True) :
  probabilitY_completely_black_after_rotations initial_grid = probability_grid_complete_black := 
by 
  sorry

end probability_is_correct_l753_753178


namespace sum_possible_values_b_l753_753842

-- Given: The function g(x) = x^2 - bx + 3b
-- The zeroes of g(x) are integers
-- We need to prove that the sum of the possible values of b is 53
theorem sum_possible_values_b (b: ℤ) :
  (∃ a₁ a₂ : ℤ, (a₁ + a₂ = b) ∧ (a₁ * a₂ = 3 * b)) →
  -- If the zeroes are integers, then the sum of the possible b's equals 53
  ∑ b in {b : ℤ | ∃ a₁ a₂ : ℤ, (a₁ + a₂ = b) ∧ (a₁ * a₂ = 3 * b)}, b = 53 := by
    sorry

end sum_possible_values_b_l753_753842


namespace train_cross_platform_time_l753_753892

/-- Given a 300 meter long train, which crosses a signal pole in 18 seconds,
and a platform length of 350 meters, prove that it takes 39 seconds for the
train to cross the platform. -/
theorem train_cross_platform_time :
  ∀ (train_length platform_length time_to_cross_pole : ℝ),
  train_length = 300 →
  platform_length = 350 →
  time_to_cross_pole = 18 →
  let speed := train_length / time_to_cross_pole in
  let total_distance := train_length + platform_length in
  let time_to_cross_platform := total_distance / speed in
  time_to_cross_platform ≈ 39 := 
by
  intros train_length platform_length time_to_cross_pole h_train_length h_platform_length h_time_to_cross_pole
  let speed := train_length / time_to_cross_pole
  let total_distance := train_length + platform_length
  let time_to_cross_platform := total_distance / speed
  sorry

end train_cross_platform_time_l753_753892


namespace even_n_translation_odd_n_reflection_l753_753458

structure Line where
  -- Define a structure for a line
  slope : ℝ
  intercept : ℝ
  
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope  -- Two lines are parallel if they have the same slope

def is_translation (n : ℕ) : Prop :=
  ∃ t : ℝ, ∀ i : ℕ, i ≤ n → (lines (i+1)).intercept = lines i.intercept + t

def is_reflection (n : ℕ) : Prop :=
  ∃ l : Line, ∀ i : ℕ, (i < n) → are_parallel (lines (i+1)) l

axiom lines : ℕ → Line  -- Define lines as a sequence indexed by natural numbers

-- Problem statements
theorem even_n_translation (n : ℕ) (h_even : n % 2 = 0) (h_parallel : ∀ i < n, are_parallel (lines i) (lines (i + 1))) :
  is_translation n :=
sorry

theorem odd_n_reflection (n : ℕ) (h_odd : n % 2 = 1) (h_parallel : ∀ i < n, are_parallel (lines i) (lines (i + 1))) :
  is_reflection n :=
sorry

end even_n_translation_odd_n_reflection_l753_753458


namespace final_selling_price_correct_l753_753192

noncomputable def purchase_price_inr : ℝ := 8000
noncomputable def depreciation_rate_annual : ℝ := 0.10
noncomputable def profit_rate : ℝ := 0.10
noncomputable def discount_rate : ℝ := 0.05
noncomputable def sales_tax_rate : ℝ := 0.12
noncomputable def exchange_rate_at_purchase : ℝ := 80
noncomputable def exchange_rate_at_selling : ℝ := 75

noncomputable def depreciated_value_after_2_years (initial_value : ℝ) : ℝ :=
  initial_value * (1 - depreciation_rate_annual) * (1 - depreciation_rate_annual)

noncomputable def marked_price (initial_value : ℝ) : ℝ :=
  initial_value * (1 + profit_rate)

noncomputable def selling_price_before_tax (marked_price : ℝ) : ℝ :=
  marked_price * (1 - discount_rate)

noncomputable def final_selling_price_inr (selling_price_before_tax : ℝ) : ℝ :=
  selling_price_before_tax * (1 + sales_tax_rate)

noncomputable def final_selling_price_usd (final_selling_price_inr : ℝ) : ℝ :=
  final_selling_price_inr / exchange_rate_at_selling

theorem final_selling_price_correct :
  final_selling_price_usd (final_selling_price_inr (selling_price_before_tax (marked_price purchase_price_inr))) = 124.84 := 
sorry

end final_selling_price_correct_l753_753192


namespace milk_price_increase_day_l753_753823

theorem milk_price_increase_day (total_cost : ℕ) (old_price : ℕ) (new_price : ℕ) (days : ℕ) (x : ℕ)
    (h1 : old_price = 1500)
    (h2 : new_price = 1600)
    (h3 : days = 30)
    (h4 : total_cost = 46200)
    (h5 : (x - 1) * old_price + (days + 1 - x) * new_price = total_cost) :
  x = 19 :=
by
  sorry

end milk_price_increase_day_l753_753823


namespace distinct_solutions_eq_four_l753_753953

theorem distinct_solutions_eq_four : ∃! (x : ℝ), abs (x - abs (3 * x + 2)) = 4 :=
by sorry

end distinct_solutions_eq_four_l753_753953


namespace probability_B_to_E_l753_753088

variable {A B C D E: Type}

noncomputable def length (x y : Type) : ℝ := sorry -- Abstract length function

theorem probability_B_to_E
  (AB AD BE BC : ℝ)
  (h1 : AB = 4 * AD)
  (h2 : AB = 8 * BE)
  (h3 : AB = 2 * BC) :
  (BE / AB) = 1 / 8 :=
by
  have h_AD : AD = AB / 4 := by rw [h1, ← mul_div_cancel' AD (ne_of_gt (by norm_num : (0:ℝ) < 4))]
  have h_BE : BE = AB / 8 := by rw [h2, ← mul_div_cancel' BE (ne_of_gt (by norm_num : (0:ℝ) < 8))]
  have h_BC : BC = AB / 2 := by rw [h3, ← mul_div_cancel' BC (ne_of_gt (by norm_num : (0:ℝ) < 2))]
  rw [h_BE]
  norm_num

end probability_B_to_E_l753_753088


namespace age_ratio_in_six_years_l753_753852

-- Definitions for Claire's and Pete's current ages
variables (c p : ℕ)

-- Conditions given in the problem
def condition1 : Prop := c - 3 = 2 * (p - 3)
def condition2 : Prop := p - 7 = (1 / 4) * (c - 7)

-- The proof problem statement
theorem age_ratio_in_six_years (c p : ℕ) (h1 : condition1 c p) (h2 : condition2 c p) : 
  (c + 6) = 3 * (p + 6) :=
sorry

end age_ratio_in_six_years_l753_753852


namespace area_of_triangle_ABC_l753_753526

-- Given conditions
variables (A B C K : Point)
variables (AC BC : ℝ) (BK : ℝ)
variable [hTriangle : triangle A B C]
variable [point_on_line : K ∈ line_segment B C]
variable [altitude : is_altitude A K (triangle A B C)]
variable [AC_val : AC = 12]
variable [BK_val : BK = 6]
variable [BC_val : BC = 15]

-- To prove
noncomputable def calculate_area (A B C K : Point) : ℝ :=
  let CK := BC - BK in
  let AK := sqrt (AC^2 - CK^2) in
  (1/2) * BC * AK

theorem area_of_triangle_ABC :
  calculate_area A B C K = 45 * sqrt 7 / 2 := sorry

end area_of_triangle_ABC_l753_753526


namespace correct_value_l753_753004

theorem correct_value : ∀ (x : ℕ),  (x / 6 = 12) → (x * 7 = 504) :=
  sorry

end correct_value_l753_753004


namespace minimum_value_of_expression_l753_753764

noncomputable def min_value (p q r s t u : ℝ) : ℝ :=
  (1 / p) + (9 / q) + (25 / r) + (49 / s) + (81 / t) + (121 / u)

theorem minimum_value_of_expression (p q r s t u : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hs : 0 < s) (ht : 0 < t) (hu : 0 < u) (h_sum : p + q + r + s + t + u = 11) :
  min_value p q r s t u ≥ 1296 / 11 :=
by sorry

end minimum_value_of_expression_l753_753764


namespace Laran_large_posters_daily_l753_753062

/-
Problem statement:
Laran has started a poster business. She is selling 5 posters per day at school. Some posters per day are her large posters that sell for $10. The large posters cost her $5 to make. The remaining posters are small posters that sell for $6. They cost $3 to produce. Laran makes a profit of $95 per 5-day school week. How many large posters does Laran sell per day?
-/

/-
Mathematically equivalent proof problem:
Prove that the number of large posters Laran sells per day is 2, given the following conditions:
1) L + S = 5
2) 5L + 3S = 19
-/

variables (L S : ℕ)

-- Given conditions
def condition1 := L + S = 5
def condition2 := 5 * L + 3 * S = 19

-- Prove the desired statement
theorem Laran_large_posters_daily 
    (h1 : condition1 L S) 
    (h2 : condition2 L S) : 
    L = 2 := 
sorry

end Laran_large_posters_daily_l753_753062


namespace bran_amount_to_pay_l753_753235

variable (tuition_fee scholarship_percentage monthly_income payment_duration : ℝ)

def amount_covered_by_scholarship : ℝ := scholarship_percentage * tuition_fee

def remaining_after_scholarship : ℝ := tuition_fee - amount_covered_by_scholarship

def total_earnings_part_time_job : ℝ := monthly_income * payment_duration

def amount_still_to_pay : ℝ := remaining_after_scholarship - total_earnings_part_time_job

theorem bran_amount_to_pay (h_tuition_fee : tuition_fee = 90)
                          (h_scholarship_percentage : scholarship_percentage = 0.30)
                          (h_monthly_income : monthly_income = 15)
                          (h_payment_duration : payment_duration = 3) :
  amount_still_to_pay tuition_fee scholarship_percentage monthly_income payment_duration = 18 := 
by
  sorry

end bran_amount_to_pay_l753_753235


namespace investment_ratio_l753_753506

theorem investment_ratio 
  (P Q : ℝ) 
  (profitP profitQ : ℝ)
  (h1 : profitP = 7 * (profitP + profitQ) / 17) 
  (h2 : profitQ = 10 * (profitP + profitQ) / 17)
  (tP : ℝ := 10)
  (tQ : ℝ := 20) 
  (h3 : profitP / profitQ = (P * tP) / (Q * tQ)) :
  P / Q = 7 / 5 := 
sorry

end investment_ratio_l753_753506


namespace probability_of_drawing_green_or_black_l753_753151

theorem probability_of_drawing_green_or_black :
  let total_marbles := 4 + 3 + 6 in
  let favorable_outcomes := 4 + 3 in
  (favorable_outcomes : ℚ) / total_marbles = 7 / 13 := 
by {
  let total_marbles := 4 + 3 + 6,
  let favorable_outcomes := 4 + 3,
  have h1 : total_marbles = 13 := rfl,
  have h2 : favorable_outcomes = 7 := rfl,
  rw [h1, h2],
  norm_num,
  sorry
}

end probability_of_drawing_green_or_black_l753_753151


namespace train_crossing_time_l753_753860

/-- Given conditions -/
variables (speed_faster : ℝ) (speed_slower : ℝ) (length_faster : ℝ)

/-- Given values -/
def speed_faster_value : ℝ := 72
def speed_slower_value : ℝ := 36
def length_faster_value : ℝ := 370

/-- Conversion factor from kmph to m/s -/
def kmph_to_mps (speed : ℝ) : ℝ := speed * (5 / 18)

/-- Calculate the relative speed in m/s -/
def relative_speed (speed_faster speed_slower : ℝ) : ℝ :=
  kmph_to_mps (speed_faster - speed_slower)

/-- Prove that the time for the faster train to cross the man in the slower train is 37 seconds -/
theorem train_crossing_time :
  relative_speed speed_faster_value speed_slower_value ≠ 0 →
  length_faster_value / (relative_speed speed_faster_value speed_slower_value) = 37 :=
by
  sorry

end train_crossing_time_l753_753860


namespace BCE_collinear_ABDE_concyclic_l753_753932

variables (A B C D E : Point)
variable [geometry : EuclideanGeometry]

open EuclideanGeometry

-- Conditions
axiom h1 : is_excenter D A B C
axiom h2 : E = reflect_over_line A D C

-- Part 1: Prove that B, C, and E are collinear
theorem BCE_collinear : collinear B C E :=
by sorry

-- Part 2: Prove that A, B, D, and E are concyclic
theorem ABDE_concyclic : concyclic A B D E :=
by sorry

end BCE_collinear_ABDE_concyclic_l753_753932


namespace probability_palindromic_phone_number_l753_753396

/-- In a city where phone numbers consist of 7 digits, the Scientist easily remembers a phone number
    if it is a palindrome (reads the same forwards and backwards). Prove that the probability that 
    a randomly chosen 7-digit phone number is a palindrome is 0.001. -/
theorem probability_palindromic_phone_number : 
  let total_phone_numbers := 10^7
  let palindromic_phone_numbers := 10^4
  (palindromic_phone_numbers : ℝ) / total_phone_numbers = 0.001 :=
by
  let total_phone_numbers := 10^7
  let palindromic_phone_numbers := 10^4
  show (palindromic_phone_numbers : ℝ) / total_phone_numbers = 0.001 from sorry

end probability_palindromic_phone_number_l753_753396


namespace prob_between_minus1_and_1_l753_753670

variables {σ : ℝ} {Z : ℝ → ℝ}
noncomputable def normal_distribution (μ σ2 x : ℝ) : ℝ :=
  exp ( - (x - μ)^2 / (2 * σ2)) / sqrt (2 * π * σ2)

axiom Z_normal : ∀ x, Z x = normal_distribution 0 (σ^2) x
axiom P_Z_gt_1 : ∫ (x : ℝ) in 1..∞, Z x = 0.023

theorem prob_between_minus1_and_1 :
  ∫ (x : ℝ) in -1..1, Z x = 0.954 :=
by
  sorry

end prob_between_minus1_and_1_l753_753670


namespace face_opposite_to_A_is_D_l753_753945

-- Definitions of faces
inductive Face : Type
| A | B | C | D | E | F

open Face

-- Given conditions
def C_is_on_top : Face := C
def B_is_to_the_right_of_C : Face := B
def forms_cube (f1 f2 : Face) : Prop := -- Some property indicating that the faces are part of a folded cube
sorry

-- The theorem statement to prove that the face opposite to face A is D
theorem face_opposite_to_A_is_D (h1 : C_is_on_top = C) (h2 : B_is_to_the_right_of_C = B) (h3 : forms_cube A D)
    : ∃ f : Face, f = D := sorry

end face_opposite_to_A_is_D_l753_753945


namespace exercise_l753_753951

def op (a b : ℝ) : ℝ := a^3 / b

theorem exercise : (op (op 2 4) 6 - op 2 (op 4 6)) = 7 / 12 :=
by
  sorry

end exercise_l753_753951


namespace probability_odd_sum_given_odd_product_l753_753638

-- Definitions related to conditions:
def die_values := {1, 2, 3, 4, 5, 6}

-- Axioms to represent conditions
axiom dice_rolled : list ℕ -- List representing the dice each of which takes value from die_values
axiom dice_length_five : dice_rolled.length = 5
axiom odd_product : ∀ x ∈ dice_rolled, x % 2 = 1

-- Theorem representing the equivalence of the problem
theorem probability_odd_sum_given_odd_product :
  ∀ (dice_rolled : list ℕ), 
    dice_length_five ∧ (∀ x ∈ dice_rolled, x % 2 = 1) →
    ((dice_rolled.sum % 2 = 1) = 1) :=
sorry

end probability_odd_sum_given_odd_product_l753_753638


namespace num_ways_divisible_165_l753_753686

theorem num_ways_divisible_165 (replace_two_digits : Nat → Nat → (Nat × Nat)) : (∃ m n : Nat,
  let p := ("5" ++ "0".repeat(80) ++ "5").toList;
  let p := List.modifyNth (replace_two_digits 0) (replace_two_digits 1) p;
  let x := p.foldl (λ acc c, 10 * acc + (c.toNat - '0'.toNat)) 0;
  List.foldl (λ acc d, acc + d) 0 (x.digits 10) % 3 = 0 ∧
  List.foldl (λ acc (hd idx : Nat), if idx % 2 == 0 then acc + hd else acc - hd) 0 (x.digits 10).enum % 11 = 0
) -> 17280 := 
begin
  sorry
end

end num_ways_divisible_165_l753_753686


namespace sin_properties_l753_753824

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - 5 * Real.pi / 6)

theorem sin_properties :
  (∀ x : ℝ, f(x) = 3 * Real.sin(2 * x - 5 * Real.pi / 6)) ∧
  (∃ x : ℝ, x = Real.pi / 6 ∧ f x = -3) ∧
  (∃ x : ℝ, x = 2 * Real.pi / 3 ∧ f x = 3) :=
by 
  sorry

end sin_properties_l753_753824


namespace solutions_are_integers_l753_753952

-- Definitions following the conditions of the problem
def equation (a : ℝ) : Prop := log 8 (2 * a^2 - 18 * a + 32) = 3

-- The final proof statement confirming the nature of the solutions based on the provided correct answer
theorem solutions_are_integers : ∀ a : ℝ, equation a → (∃ a1 a2 : ℤ, a = a1 ∨ a = a2) :=
by sorry

end solutions_are_integers_l753_753952


namespace sum_is_402_3_l753_753541

def sum_of_numbers := 3 + 33 + 333 + 33.3

theorem sum_is_402_3 : sum_of_numbers = 402.3 := by
  sorry

end sum_is_402_3_l753_753541


namespace max_proj_area_of_regular_tetrahedron_l753_753865

theorem max_proj_area_of_regular_tetrahedron (a : ℝ) (h_a : a > 0) : 
    ∃ max_area : ℝ, max_area = a^2 / 2 :=
by
  existsi (a^2 / 2)
  sorry

end max_proj_area_of_regular_tetrahedron_l753_753865


namespace construct_isosceles_trapezoid_theorem_l753_753948

noncomputable def construct_isosceles_trapezoid (longer_parallel_side non_parallel_side : ℝ) (acute_angle : ℝ) : Prop :=
  ∃ (A B C D : ℝ → ℝ → Prop), 
  is_trapezoid A B C D ∧ 
  is_isosceles_trapezoid A B C D ∧
  (distance A B = longer_parallel_side ) ∧
  (distance A D = non_parallel_side) ∧
  (obtuse_angle_of_diagonals A B C D) / 2 = acute_angle

theorem construct_isosceles_trapezoid_theorem (longer_parallel_side non_parallel_side : ℝ) (acute_angle : ℝ) :
  construct_isosceles_trapezoid longer_parallel_side non_parallel_side acute_angle :=
sorry

end construct_isosceles_trapezoid_theorem_l753_753948


namespace triangle_area_PQR_l753_753144

noncomputable def point := ℤ × ℤ

def P : point := (2, 5)

def line1 := 3
def line2 := -1

def area_of_triangle (A B C : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  abs (1 / 2 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)))

theorem triangle_area_PQR :
  let Q := (1/3 : ℝ, 0)
  let R := (7, 0)
  area_of_triangle P Q R = 50 / 3 :=
by
  -- statement only, so we use sorry to skip proof
  sorry

end triangle_area_PQR_l753_753144


namespace five_collinear_points_l753_753242

variables {α β : Type}

noncomputable def conditions (Φ : set (ℝ × ℝ)) :=
  ∀ (a b c d : ℝ) (x y : ℝ × ℝ),
    -- Φ doesn't contain the origin and not all points are collinear
    ((0,0) ∉ Φ ∧ ¬ ∀ p₁ p₂ p₃ ∈ Φ, collinear_points p₁ p₂ p₃) ∧ 

    -- If α ∈ Φ, then -α ∈ Φ and cα ∉ Φ for c ≠ 1 or -1
    ((a, b) ∈ Φ → (-a, -b) ∈ Φ ∧ ∀ c : ℝ, c ≠ 1 ∧ c ≠ -1 → (c * a, c * b) ∉ Φ) ∧ 

    -- Reflection property
    ((a, b) ∈ Φ ∧ (c, d) ∈ Φ → reflect_in_line p₁ p₂ ∈ Φ) ∧ 

    -- Condition on dot product
    ((a, b) ∈ Φ ∧ (c, d) ∈ Φ → ∃ k : ℤ, 2 * (a * c + b * d) = k * (c^2 + d^2))

-- Define reflection
def reflect_in_line ((a, b) (c, d) : ℝ × ℝ) : ℝ × ℝ :=
  -- Implementation depends on specific geometry form but left unspecified for now
  (sorry, sorry)

-- Define collinearity
def collinear_points ((x1, y1) (x2, y2) (x3, y3) : ℝ × ℝ) : Prop :=
  -- Implementation depends on specific geometry form but left unspecified for now
  sorry

-- Theorem to prove
theorem five_collinear_points (Φ : set (ℝ × ℝ)) (cond : conditions Φ) : 
  ¬ ∃ (p1 p2 p3 p4 p5 : ℝ × ℝ), p1 ∈ Φ ∧ p2 ∈ Φ ∧ p3 ∈ Φ ∧ p4 ∈ Φ ∧ p5 ∈ Φ ∧ collinear_points p1 p2 p3 ∧ collinear_points p2 p3 p4 ∧ collinear_points p3 p4 p5 := sorry

end five_collinear_points_l753_753242


namespace fg_neg3_eq_6_l753_753357

def f (x : ℝ) : ℝ := 6 - 2 * real.sqrt x
def g (x : ℝ) : ℝ := 3 * x + x ^ 2

theorem fg_neg3_eq_6 : f (g (-3)) = 6 := by
  -- Proof goes here
  sorry

end fg_neg3_eq_6_l753_753357


namespace odd_and_decreasing_l753_753871

-- Definitions for the conditions
def f1 (x : ℝ) : ℝ := -x^2
def f2 (x : ℝ) : ℝ := 1 / x^2
def f3 (x : ℝ) : ℝ := 1 / x
def f4 (x : ℝ) : ℝ := x^3

-- Statement to prove
theorem odd_and_decreasing : 
  (∀ x : ℝ, 0 < x → f3(-x) = -f3(x) ∧ f3(x) > f3(x + 0.001)) :=
by
  sorry

end odd_and_decreasing_l753_753871


namespace binary_to_base5_l753_753249

theorem binary_to_base5 (n : ℕ) (h1 : n = 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0)
  (h2 : nat.div_mod n 5 = (9, 0)) (h3 : nat.div_mod 9 5 = (1, 4)) (h4 : nat.div_mod 1 5 = (0, 1)) :
  45 = 140 :=
sorry

end binary_to_base5_l753_753249


namespace area_triangle_AEF_l753_753388

theorem area_triangle_AEF (BE EC CF FD : ℝ)
  (h1 : BE = 5)
  (h2 : EC = 4)
  (h3 : CF = 4)
  (h4 : FD = 1) :
  let AD := BE + EC + CF + FD in
  let rectangle_area := (BE + EC) * AD in
  let triangle_EFC_area := 0.5 * EC * CF in
  let triangle_ABE_area := 0.5 * (BE + EC) * BE in
  let triangle_ADF_area := 0.5 * AD * FD in
  let triangle_AEF_area := rectangle_area - triangle_EFC_area - triangle_ABE_area - triangle_ADF_area in
  triangle_AEF_area = 42.5 :=
begin
  sorry
end

end area_triangle_AEF_l753_753388


namespace sum_intervals_length_l753_753091

noncomputable def f (x : ℝ) : ℝ := ∑ k in finset.range 70, k / (x - (k + 1))

theorem sum_intervals_length :
  (∑ k in finset.range 70, (find_sup (λ x, x ∈ Ioo (k : ℝ) (k + 1 + 1) ∧ f x = 5 / 4) - (k : ℝ))) = 1988 :=
sorry

end sum_intervals_length_l753_753091


namespace number_of_arithmetic_sequences_l753_753769

theorem number_of_arithmetic_sequences (n : ℕ) :
  let S := finset.range (n + 1) in
  let f : ℕ → ℕ → finset ℕ := λ a d, finset.range' (a - (d - 1)) (n + 1) \ finset.range' (a - d) (n + 2) in
  ∑ a in S.filter (λ a, 2 * a ≤ n), (n - 2 * a + 1) =
  if even n then n^2 / 4 else (n^2 - 1) / 4 :=
begin
  sorry -- Proof is omitted
end

end number_of_arithmetic_sequences_l753_753769


namespace onions_on_shelf_after_operations_l753_753137

theorem onions_on_shelf_after_operations :
  (let initial_count: ℕ := 98, 
       sold: ℕ := 65,
       added: ℕ := 20,
       given_fraction: ℝ := 1/4 in
    let remaining_after_sale := initial_count - sold in
    let remaining_percent: ℝ := 0.25 in
    let total_before_sale := (remaining_after_sale : ℝ) / remaining_percent in
    let after_new_shipment := remaining_after_sale + added in
    let given_away := (given_fraction * (after_new_shipment : ℝ)).to_nat in
    after_new_shipment - given_away = 40) :=
sorry

end onions_on_shelf_after_operations_l753_753137


namespace podium_cubes_total_l753_753471

def step1 := 4 * 4 * 2
def step2 := 4 * 4 * 3
def step3 := 4 * 4 * 4

theorem podium_cubes_total : step1 + step2 + step3 = 144 := by
  done

end podium_cubes_total_l753_753471


namespace other_x_intercept_l753_753291

theorem other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → ax^2 + bx + c = 10) (h_intercept : ∀ x, x = -2 → ax^2 + bx + c = 0) : ∃ x : ℝ, x = 12 ∧ ax^2 + bx + c = 0 :=
by
  -- condition: vertex at (5, 10)
  have h_symmetry : ∀ x, x = 5 → ax^2 + bx + c = 10 := h_vertex
  -- condition: one x-intercept at (-2, 0)
  have h_known_intercept : ∀ x, x = -2 → ax^2 + bx + c = 0 := h_intercept
  -- conclude that the other x-intercept is at x = 12
  use 12
  split
  -- proof part that x = 12 is valid (skipped)
  · exact rfl
  -- proof part that ax^2 + bx + c = 0 for x = 12 (skipped)
  · sorry

end other_x_intercept_l753_753291


namespace value_of_and_15_and_l753_753972

def op_and (x : ℝ) : ℝ := 8 - x
def and_op (x : ℝ) : ℝ := x - 8

theorem value_of_and_15_and : and_op (op_and 15) = -15 :=
by
  simp [op_and, and_op]
  sorry

end value_of_and_15_and_l753_753972


namespace definite_integral_ln_squared_l753_753558

noncomputable def integralFun : ℝ → ℝ := λ x => x * (Real.log x) ^ 2

theorem definite_integral_ln_squared (f : ℝ → ℝ) (a b : ℝ):
  (f = integralFun) → 
  (a = 1) → 
  (b = 2) → 
  ∫ x in a..b, f x = 2 * (Real.log 2) ^ 2 - 2 * Real.log 2 + 3 / 4 :=
by
  intros hfa hao hbo
  rw [hfa, hao, hbo]
  sorry

end definite_integral_ln_squared_l753_753558


namespace min_rank_remaining_weight_l753_753434

theorem min_rank_remaining_weight {n : ℕ} (h_pos : n > 0) (weights : Fin (2^n) → ℝ)
  (h_distinct : ∀ i j : Fin (2^n), i ≠ j → weights i ≠ weights j) :
  ∃ w : Fin (2^n), 
    (∀ t : Fin n, ∀ (groups : 2 → Fin (2^n/2) → Fin (2^n)), 
      (∀ i j, i ≠ j → ∀ x, groups i x ≠ groups j x) ∧ 
      (weights_grouped_mass groups 0 > weights_grouped_mass groups 1) → 
      ∃ next_group : Fin n → Fin (2^(n-t.succ)), 
      (∀ i j, i ≠ j → ∀ x, next_group i x = groups 0 i x ∨ next_group i x = groups 1 i x) ∧ 
      weights_grouped_mass next_group 0 > weights_grouped_mass next_group 1) ∧ 
    rank_in_ordered_weights weights w = 2^n - n :=
sorry

noncomputable def weights_grouped_mass 
  {m : ℕ} (groups : 2 → Fin m → Fin (2^m)) (i : Fin 2) : ℝ :=
  ∑ j, weights (groups i j)

noncomputable def rank_in_ordered_weights {m : ℕ} (weights : Fin m → ℝ) (w : Fin m) : ℕ :=
  ∃ r : Fin m, (∀ k : Fin m, weights k < weights w) ∧ r.val = m - n

end min_rank_remaining_weight_l753_753434


namespace floor_sum_inequality_l753_753802

theorem floor_sum_inequality (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∑ k in Finset.range (n + 1), (⌊(k * m : ℚ) / k^2⌋ : ℕ)) ≤ n + m * (m / 2 - 1) := 
sorry

end floor_sum_inequality_l753_753802


namespace question1_question2_application_l753_753861

theorem question1: (-4)^2 - (-3) * (-5) = 1 := by
  sorry

theorem question2 (a : ℝ) (h : a = -4) : a^2 - (a + 1) * (a - 1) = 1 := by
  sorry

theorem application (a : ℝ) (h : a = 1.35) : a * (a - 1) * 2 * a - a^3 - a * (a - 1)^2 = -1.35 := by
  sorry

end question1_question2_application_l753_753861


namespace infinite_values_of_a_with_1980_solutions_l753_753560

theorem infinite_values_of_a_with_1980_solutions :
  ∃ (a: ℕ), ∀ (n: ℕ), 
    (n > 6 * 1980^2) → 
    (a = 9 * n^3 + 6 * n^2) ∧ 
    ∃ (x y: ℕ), 
      (1 ≤ x ∧ 1 ≤ y) → 
      ∃ (k: ℕ), 
      (1 ≤ k ∧ k ≤ 1980) → 
      (x = n^2 + 4 * k ∧ y = 4 * n^2 + 2 * (n - k)) ∧ 
      (⌊x ^ (3/2)⌋ + ⌊y ^ (3/2)⌋ = a) :=
sorry

end infinite_values_of_a_with_1980_solutions_l753_753560


namespace volume_of_pyramid_l753_753171

-- Define the length of the side of the cube
def side_length : ℝ := 2

-- Define the base area of the equilateral triangle ABC
def base_area : ℝ := (sqrt 3 / 4) * side_length^2

-- Define the height of the pyramid ABCG
def height_pyramid : ℝ := side_length

-- Define the volume of pyramid ABCG
def pyramid_volume : ℝ := (1 / 3) * base_area * height_pyramid

-- State the theorem
theorem volume_of_pyramid (s : ℝ) (base_area : ℝ) (height : ℝ) : 
  pyramid_volume = (2 * sqrt 3) / 3 := by
  sorry

end volume_of_pyramid_l753_753171


namespace tank_fraction_l753_753085

theorem tank_fraction (x : ℚ) : 
  let tank1_capacity := 7000
  let tank2_capacity := 5000
  let tank3_capacity := 3000
  let tank2_fraction := 4 / 5
  let tank3_fraction := 1 / 2
  let total_water := 10850
  tank1_capacity * x + tank2_capacity * tank2_fraction + tank3_capacity * tank3_fraction = total_water → 
  x = 107 / 140 := 
by {
  sorry
}

end tank_fraction_l753_753085


namespace correct_mean_after_correction_l753_753119

noncomputable def initial_mean : ℚ := 235
def n : ℕ := 100
def incorrect_values : List ℚ := [300, 400, 210]
def correct_values : List ℚ := [320, 410, 230]

theorem correct_mean_after_correction
  (incorrect_mean : ℚ := initial_mean)
  (num_values : ℕ := n)
  (incorrect_vals : List ℚ := incorrect_values)
  (correct_vals : List ℚ := correct_values) :
  let incorrect_total_sum : ℚ := incorrect_mean * num_values
  let corrections : List ℚ := List.zipWith (-) correct_vals incorrect_vals
  let total_correction : ℚ := List.sum corrections
  let correct_total_sum : ℚ := incorrect_total_sum + total_correction
  correct_total_sum / num_values = 235.50 :=
by
  sorry

end correct_mean_after_correction_l753_753119


namespace ordered_pairs_geometric_seq_log_l753_753125

theorem ordered_pairs_geometric_seq_log :
  ∃! (a r : ℕ), (a > 0 ∧ r > 0) ∧ (∑ i in Finset.range 8, Real.logb 4 (a * r^i) = 840) :=
  sorry

end ordered_pairs_geometric_seq_log_l753_753125


namespace three_digit_numbers_divisible_by_17_l753_753352

theorem three_digit_numbers_divisible_by_17 : ∃ (n : ℕ), n = 53 ∧ ∀ k, 100 <= 17 * k ∧ 17 * k <= 999 ↔ (6 <= k ∧ k <= 58) :=
by
  sorry

end three_digit_numbers_divisible_by_17_l753_753352


namespace inradius_lt_l753_753090

variables {A B C A' B' C' : Type*} 
variable [ordered_triangle A B C] 
variable [ordered_triangle A' B' C']
variable (r_ABC : inradius A B C)
variable (r_A'B'C' : inradius A' B' C')

theorem inradius_lt (h : triangle_inside A B C A' B' C') : r_ABC < r_A'B'C' :=
sorry

end inradius_lt_l753_753090


namespace max_intersections_l753_753854

theorem max_intersections (X Y : Type) [Fintype X] [Fintype Y]
  (hX : Fintype.card X = 20) (hY : Fintype.card Y = 10) : 
  ∃ (m : ℕ), m = 8550 := by
  sorry

end max_intersections_l753_753854


namespace integral_evaluation_l753_753260

noncomputable def first_integral : ℝ := ∫ x in 1..Real.exp 1, 1 / x
noncomputable def second_integral : ℝ := ∫ x in 0..Real.pi / 2, Real.sin x

theorem integral_evaluation : (first_integral - second_integral) = 0 := by
  sorry

end integral_evaluation_l753_753260


namespace no_seq_for_lambda_le_e_l753_753624

open Real

theorem no_seq_for_lambda_le_e (λ : ℝ) (hλ : λ ∈ Ioo 0 (exp 1)) :
    ∀ (a : ℕ → ℝ), (∀ n ≥ 2, 0 < a n) → ¬ (∀ n ≥ 2, a n + 1 ≤ Real.sqrt n λ * a (n-1)) := by
    sorry

end no_seq_for_lambda_le_e_l753_753624


namespace victory_saved_less_l753_753862

-- Definitions based on conditions
def total_savings : ℕ := 1900
def sam_savings : ℕ := 1000
def victory_savings : ℕ := total_savings - sam_savings

-- Prove that Victory saved $100 less than Sam
theorem victory_saved_less : sam_savings - victory_savings = 100 := by
  -- placeholder for the proof
  sorry

end victory_saved_less_l753_753862


namespace y_coord_intersection_with_y_axis_l753_753841

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + 11

-- Define the point P
def P : ℝ × ℝ := (1, curve 1)

-- Define the derivative of the curve
def derivative (x : ℝ) : ℝ := 3 * x^2

-- Define the tangent line at point P (1, 12)
def tangent_line (x : ℝ) : ℝ := 3 * (x - 1) + 12

-- Proof statement
theorem y_coord_intersection_with_y_axis : 
  tangent_line 0 = 9 :=
by
  -- proof goes here
  sorry

end y_coord_intersection_with_y_axis_l753_753841


namespace point_in_plane_region_l753_753873

theorem point_in_plane_region :
  (2 * 0 + 1 - 6 < 0) ∧ ¬(2 * 5 + 0 - 6 < 0) ∧ ¬(2 * 0 + 7 - 6 < 0) ∧ ¬(2 * 2 + 3 - 6 < 0) :=
by
  -- Proof detail goes here.
  sorry

end point_in_plane_region_l753_753873


namespace greatest_digit_sum_base_8_l753_753149

/-- The greatest possible sum of the digits in the base-eight representation of a positive integer less than 5000 is 28. -/
theorem greatest_digit_sum_base_8 (n : ℕ) (h : 0 ≤ n ∧ n < 5000) :
  ∃ n, (sum_of_digits_base_8 n = 28) := sorry

end greatest_digit_sum_base_8_l753_753149


namespace no_three_consecutive_identical_digits_binary_l753_753969

theorem no_three_consecutive_identical_digits_binary :
  ∃ (count : ℕ), count = 228 ∧ ∀ n : ℕ, (4 ≤ n ∧ n ≤ 1023) → 
  ¬(∃ x : ℕ, (bit x 0 + bit x 0 = n) ∨ (bit x 1 + bit x 1 = n) ∨ (bit x + bit x + 1 = n)) :=
sorry

end no_three_consecutive_identical_digits_binary_l753_753969


namespace tile_width_l753_753095

theorem tile_width (W : ℝ) (hf_w : 150) (hf_l : 390) (ht_l : 25) (max_tiles : 36) :
  max_tiles = 15 * (hf_w / W) → W = 62.5 :=
by 
  intro h1
  sorry

end tile_width_l753_753095


namespace h_at_3_l753_753748

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 5
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (f x) + 1
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_at_3 : h 3 = 74 + 28 * Real.sqrt 2 :=
by
  sorry

end h_at_3_l753_753748


namespace no_valid_polynomial_l753_753617

-- Define the polynomial form
def Q (a b c : ℝ) : ℝ[X] := X^4 + a * X^3 + b * X^2 + c * X + 2048

-- State the theorem to be proved
theorem no_valid_polynomial : ∀ (a b c : ℝ), 
  ¬ ∃ s : ℂ, (Q a b c).isRoot s ∧ (Q a b c).isRoot (s^2) ∧ (Q a b c).isRoot (1/s) :=
by
  intro a b c
  sorry

end no_valid_polynomial_l753_753617


namespace alpha_plus_two_beta_l753_753993

theorem alpha_plus_two_beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
    (h3 : tan α = 1 / 7) (h4 : sin β = sqrt 10 / 10) : α + 2 * β = π / 4 :=
  sorry

end alpha_plus_two_beta_l753_753993


namespace verify_quadratic_function_conditions_l753_753689

-- Define the quadratic function with given conditions
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given values
def y_at_neg1 := 0
def y_at_0 := -1.5
def y_at_1 := -2
def y_at_2 := -1.5

-- The conditions of values at specific points
def condition1 := quadratic_function a b c (-1) = y_at_neg1
def condition2 := quadratic_function a b c 0 = y_at_0
def condition3 := quadratic_function a b c 1 = y_at_1
def condition4 := quadratic_function a b c 2 = y_at_2

-- Rectify the statements according to those conditions
theorem verify_quadratic_function_conditions : 
  (∀ a b c : ℝ,
    condition1 →
    condition2 →
    condition3 →
    condition4 →
    ((quadratic_function a b c x) = a * (x - 1)^2 - 2) ∧
    (∀ x, quadratic_function a b c x + 1.5 = 0 → (x = 0 ∨ x = 2))) := sorry

end verify_quadratic_function_conditions_l753_753689


namespace expand_expression_l753_753273

theorem expand_expression (x y z : ℝ) :
  (2 * x + 15) * (3 * y + 20 * z + 25) = 
  6 * x * y + 40 * x * z + 50 * x + 45 * y + 300 * z + 375 :=
by
  sorry

end expand_expression_l753_753273


namespace integral_cos_pi_over_6_l753_753270

theorem integral_cos_pi_over_6 : ∫ x in 0..(Real.pi / 6), Real.cos x = 1 / 2 := 
by
  sorry

end integral_cos_pi_over_6_l753_753270


namespace fraction_of_specials_divisible_by_18_l753_753950

def is_special_integer (n : ℕ) : Prop :=
  n > 30 ∧ n < 150 ∧ n % 2 = 0 ∧ (n.digits.sum = 12)

def is_divisible_by_18 (n : ℕ) : Prop := n % 18 = 0

theorem fraction_of_specials_divisible_by_18 : 
  ratio {n : ℕ | is_special_integer n ∧ is_divisible_by_18 n} 
        {n : ℕ | is_special_integer n} = 1 / 3 :=
by
  sorry

end fraction_of_specials_divisible_by_18_l753_753950


namespace relationship_f_3x_ge_f_2x_l753_753488

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0, and
    satisfying the symmetry condition f(1-x) = f(1+x) for any x ∈ ℝ,
    the relationship f(3^x) ≥ f(2^x) holds. -/
theorem relationship_f_3x_ge_f_2x (a b c : ℝ) (h_a : a > 0) (symm_cond : ∀ x : ℝ, (a * (1 - x)^2 + b * (1 - x) + c) = (a * (1 + x)^2 + b * (1 + x) + c)) :
  ∀ x : ℝ, (a * (3^x)^2 + b * 3^x + c) ≥ (a * (2^x)^2 + b * 2^x + c) :=
sorry

end relationship_f_3x_ge_f_2x_l753_753488


namespace probability_of_odd_sum_greater_than_36_l753_753853

theorem probability_of_odd_sum_greater_than_36 :
  let tiles := finset.range 12 in
  let configurations := finset.powerset_len 3 tiles in
  let num_configurations := configurations.card.instances in
  let valid_configurations :=
    configurations.filter (λ c, (c.card = 3) ∧ (c.sum % 2 = 1) ∧ (c.sum > 6)) in
  let num_valid_configurations := valid_configurations.card.instances in
  (valid_configurations.card * valid_configurations.card * valid_configurations.card)
  / num_configurations.choose 3 = 1 / 205 :=
sorry

end probability_of_odd_sum_greater_than_36_l753_753853


namespace total_arrangements_960_l753_753719

theorem total_arrangements_960 (students teachers : Nat) (teachers_next_to_each_other : Bool) (teachers_not_at_ends : Bool) :
  students = 5 → teachers = 2 → teachers_next_to_each_other = true → teachers_not_at_ends = true → 
  (students * 4 * 2)! / (students * 4 * 2 - 2)! = 960 :=
by
  intros h1 h2 h3 h4
  sorry

end total_arrangements_960_l753_753719


namespace divisor_correct_l753_753889

/--
Given that \(10^{23} - 7\) divided by \(d\) leaves a remainder 3, 
prove that \(d\) is equal to \(10^{23} - 10\).
-/
theorem divisor_correct :
  ∃ d : ℤ, (10^23 - 7) % d = 3 ∧ d = 10^23 - 10 :=
by
  sorry

end divisor_correct_l753_753889


namespace general_term_of_sequence_l753_753440

def A := {n : ℕ | ∃ k : ℕ, k + 1 = n }
def B := {m : ℕ | ∃ k : ℕ, 3 * k - 1 = m }

theorem general_term_of_sequence (k : ℕ) : 
  ∃ a_k : ℕ, a_k ∈ A ∩ B ∧ a_k = 9 * k^2 - 9 * k + 2 :=
sorry

end general_term_of_sequence_l753_753440


namespace modulus_z_plus_1_l753_753011

noncomputable def z : ℂ := (1 - 3 * complex.i) / (1 + complex.i)

theorem modulus_z_plus_1 : abs (z + 1) = 2 := by
  sorry

end modulus_z_plus_1_l753_753011


namespace distinct_solutions_abs_eq_l753_753956

theorem distinct_solutions_abs_eq (x : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |x1 - |3 * x1 + 2|| = 4 ∧ |x2 - |3 * x2 + 2|| = 4 ∧
    (∀ x3 : ℝ, |x3 - |3 * x3 + 2|| = 4 → (x3 = x1 ∨ x3 = x2))) :=
sorry

end distinct_solutions_abs_eq_l753_753956


namespace simplify_trig_identity_l753_753465

theorem simplify_trig_identity (x : ℝ) : 
  (\sin x ^ 2 - \cos x ^ 2) / (2 * \sin x * \cos x) = - Real.cot (2 * x) := 
by 
  sorry

end simplify_trig_identity_l753_753465


namespace fourth_guard_distance_l753_753836

-- Definitions of the problem conditions
def length : ℕ := 200
def width : ℕ := 300
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)
def total_distance_three_guards : ℕ := 850

-- Statement of the problem in Lean
theorem fourth_guard_distance :
  ∀ (length width : ℕ), length = 200 → width = 300 →
  let p := perimeter length width in
  ∀ total_distance_three_guards = 850 →
  p - total_distance_three_guards = 150 :=
by
  intros length width h1 h2 total_d h3
  simp [perimeter, h1, h2, h3]
  sorry

end fourth_guard_distance_l753_753836


namespace comet_interval_is_correct_l753_753598

noncomputable def next_comet_interval (t1 t2 t3 : ℝ) (c : ℝ) (r : ℝ): ℝ :=
  if h : t1 = 20 ∧ t2 = 10 ∧ t3 = 5 ∧ t1 + t2 + t3 = c ∧ t1 * t2 + t2 * t3 + t3 * t1 = 350 ∧ t1 * t2 * t3 = 1000 ∧ r = 0.5 then
    5 * 0.5
  else
    0 -- returning 0 if conditions are not met

theorem comet_interval_is_correct:
  ∀ (t1 t2 t3 c r : ℝ),
    t1 = 20 ∧ t2 = 10 ∧ t3 = 5 ∧ 
    t1 + t2 + t3 = c ∧ 
    t1 * t2 + t2 * t3 + t3 * t1 = 350 ∧ 
    t1 * t2 * t3 = 1000 ∧ 
    r = 0.5 →
    next_comet_interval t1 t2 t3 c r = 2.5 := 
by
  intros t1 t2 t3 c r h,
  sorry

end comet_interval_is_correct_l753_753598


namespace value_of_g_at_five_l753_753112

def g (x : ℕ) : ℕ := x^2 - 2 * x

theorem value_of_g_at_five : g 5 = 15 := by
  sorry

end value_of_g_at_five_l753_753112


namespace quadrilateral_CDHcHd_parallelogram_l753_753093

variable {A B C D Hc Hd : ℝ}

-- Assume A, B, C, and D are points on a circle (cyclic quadrilateral)
def is_cyclic_quad (A B C D : ℝ) : Prop := sorry -- needs definition

-- Define the orthocenters for given triangles
def is_orthocenter (H : ℝ) (A B C : ℝ) : Prop := sorry -- needs definition

-- The given conditions
axiom condition1 : is_cyclic_quad A B C D
axiom condition2 : is_orthocenter Hc A B D
axiom condition3 : is_orthocenter Hd A B C

-- The proof problem
theorem quadrilateral_CDHcHd_parallelogram : 
  is_cyclic_quad A B C D → is_orthocenter Hc A B D → is_orthocenter Hd A B C → parallelogram C D Hc Hd := 
by
  intro h1 h2 h3
  -- proof to be provided
  sorry

end quadrilateral_CDHcHd_parallelogram_l753_753093


namespace sum_of_x_for_f_eq_3010_l753_753906

noncomputable def f (x : ℝ) : ℝ := sorry

theorem sum_of_x_for_f_eq_3010 (T : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → 2 * f x + f (1 / x) = 3 * x + 6) →
  (f x = 3010 → x ≠ 0 → x ∈ { x : ℝ | x ≠ 0 }) →
  T = 1504 :=
begin
  intros h1 h2,
  sorry
end

end sum_of_x_for_f_eq_3010_l753_753906


namespace searchlight_darkness_probability_l753_753213

noncomputable def searchlight_revolutions : ℕ := 30
def first_searchlight_rpm : ℕ := 2
def second_searchlight_rpm : ℕ := 3
def third_searchlight_rpm : ℕ := 4
def traversal_time : ℕ := 30

theorem searchlight_darkness_probability :
  traversal_time = searchlight_revolutions / first_searchlight_rpm ∨
  traversal_time = searchlight_revolutions / second_searchlight_rpm ∨
  traversal_time = searchlight_revolutions / third_searchlight_rpm → 
  traversal_time = 0 :=
by 
  -- Construct the theorem conditions
  have first_sh_solved: 30 / 2 = 15 := by sorry,
  have second_sh_solved: 30 / 3 = 10 := by sorry,
  have third_sh_solved: 30 / 4 = 7.5 := by sorry,
  
  -- We assume that traversal time is given in seconds
  have traverse_in_time : traversal_time = first_sh_solved ∨ traversal_time = second_sh_solved ∨ traversal_time = third_sh_solved :=
    by sorry,

  exact traverse_in_time

end searchlight_darkness_probability_l753_753213


namespace observe_exchange_l753_753252

variables (cells : Type) (BrdU : Type) (cycle_stage : cells → ℕ) (stained : cells → Prop)
variables (chromatid : cells → set cells) (exchange_happened : cells → Prop)
variables (semi_conservative_replication : Prop) (BrdU_incorporation: cells → cells)
variables (dark_staining : cells → Prop) (light_staining : cells → Prop)

-- Conditions
axioms
  (A1 : semi_conservative_replication)
  (A2 : ∀ cell, ∃ BrdU, BrdU_incorporation cell = cell)
  (A3 : ∀ cell, dark_staining cell ↔ (¬ BrdU_incorporation cell))
  (A4 : ∀ cell, light_staining cell ↔ (BrdU_incorporation cell))
  (A5 : ∀ cell, cycle_stage cell = 2 → exchange_happened cell)

-- Hypotheses on the stained cells
axioms
  (H1 : ∀ cell, cycle_stage cell = 2 → chromatid cell)
  (H2 : ∀ cell, cycle_stage cell = 2 → stained cell)

-- Proof Statement
theorem observe_exchange (cell : cells) :
  cycle_stage cell = 2 →
  exchange_happened cell → 
  ∃ c ∈ chromatid cell, ((dark_staining c ∧ light_staining (BrdU_incorporation c)) ∨ (light_staining c ∧ dark_staining (BrdU_incorporation c))) :=
sorry

end observe_exchange_l753_753252


namespace value_of_x_l753_753868

theorem value_of_x : 
  (x : ℚ) (h : x = (2023^2 - 2023 - 1) / 2023) 
  -> x = 2022 - 1 / 2023 :=
by
  intro x h
  sorry

end value_of_x_l753_753868


namespace isosceles_triangle_perimeter_l753_753515

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 10) (h₂ : b = 22) 
(h₃ : a + b > b) (h₄ : b + b > a) : a + (b + b) = 54 :=
by {
  rw [h₁, h₂],
  norm_num,
  exact h₃,
  exact h₄,
  sorry
}

end isosceles_triangle_perimeter_l753_753515


namespace center_of_circle_l753_753964

theorem center_of_circle (x y : ℝ) 
  (h : x^2 + 6 * x + y^2 - 8 * y - 48 = 0) : 
  (-3, 4) = 
  let h := x + 3
  let k := y - 4 in
  (-h, k) := sorry

end center_of_circle_l753_753964


namespace find_sequence_l753_753400

def sequence (a : ℕ → ℚ) : Prop :=
  a 0 = 1 ∧ a 1 = 2 ∧ ∀ n ≥ 1, n * (n + 1) * a (n + 1) = n * (n - 1) * a n - (n - 2) * a (n - 1)

theorem find_sequence (a : ℕ → ℚ) (h : sequence a) : 
∀ n ≥ 2, a n = -1 / nat.factorial n :=
by
  sorry

end find_sequence_l753_753400


namespace monkeys_peaches_simultaneously_l753_753401

theorem monkeys_peaches_simultaneously (m1 p1 t1 m2 p2 : ℕ) (h1 : m1 = 5) (h2 : p1 = 5) (h3 : t1 = 2) 
(h4 : m2 = 15) (h5 : p2 = 15) : 
  (m1 = m2 ∧ p1 = p2) → t1 = 2 := 
begin
  sorry
end

end monkeys_peaches_simultaneously_l753_753401


namespace seven_digit_in_products_l753_753070

theorem seven_digit_in_products (n : ℕ) (h_pos : n > 0) : 
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 35 ∧ (∃ d, toString (k * n) = d ++ "7" ++ toString (k * n / 10^nat_pred (String.length d))) :=
by
  sorry

end seven_digit_in_products_l753_753070


namespace incorrect_proposition_d_l753_753174

theorem incorrect_proposition_d (α β : Plane) (l : Line) : α ⊥ β → ∃ (x : Line), x ∈ α ∧ ¬ (x ⊥ β) := by
  sorry

end incorrect_proposition_d_l753_753174


namespace amc_inequality_l753_753432

theorem amc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := 
by 
  sorry

end amc_inequality_l753_753432


namespace square_area_equilateral_triangle_on_hyperbola_l753_753840

theorem square_area_equilateral_triangle_on_hyperbola :
  ∃ (A : ℝ), (∀ (x y : ℝ), x * y = 3 →
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), x₁ * y₁ = 3 ∧ x₂ * y₂ = 3 ∧ x₃ * y₃ = 3 ∧
  ((x₁ + x₂ + x₃) / 3 = 0) ∧ ((y₁ + y₂ + y₃) / 3 = 0) ∧
  (x₁, y₁), (x₂, y₂), (x₃, y₃) form an equilateral triangle →
  A = 182.25) := sorry

end square_area_equilateral_triangle_on_hyperbola_l753_753840


namespace count_bottom_right_arrows_l753_753265

/-!
# Problem Statement
Each blank cell on the edge is to be filled with an arrow. The number in each square indicates the number of arrows pointing to that number. The arrows can point in the following directions: up, down, left, right, top-left, top-right, bottom-left, and bottom-right. Each arrow must point to a number. Figure 3 is provided and based on this, determine the number of arrows pointing to the bottom-right direction.
-/

def bottom_right_arrows_count : Nat :=
  2

theorem count_bottom_right_arrows :
  bottom_right_arrows_count = 2 :=
by
  sorry

end count_bottom_right_arrows_l753_753265


namespace correct_time_fraction_l753_753912

def twelve_hour_clock : Type := ℕ
def minute_count : Type := ℕ

def is_correct_hour (h : twelve_hour_clock) : Prop :=
h ∉ {2}

def is_correct_minute (m : minute_count) : Prop := 
m < 20 ∨ m > 29 ∨ m % 10 ≠ 2

def correct_hours_fraction : ℚ := 11 / 12
def correct_minutes_fraction : ℚ := 5 / 6

theorem correct_time_fraction :
  correct_hours_fraction * correct_minutes_fraction = 55 / 72 :=
by sorry

end correct_time_fraction_l753_753912


namespace françoise_total_benefit_l753_753294

def price_per_pot : ℝ := 12
def markup : ℝ := 0.25
def pots_sold : ℕ := 150
def total_benefit : ℝ := 450

theorem françoise_total_benefit :
  (pots_sold * (price_per_pot * markup)) = total_benefit :=
by
  sorry

end françoise_total_benefit_l753_753294


namespace ratio_EG_GF_l753_753042

variables {A B C M E F G : Type}
variables [Field A]

-- Definitions
noncomputable def midpoint (B C : A) : A := (B + C) / 2
noncomputable def ratio_segment (A B : A) (p : Prop) (r : ℝ) : Prop := p → (A = r * B)
noncomputable def line_intersection (A B C : A) (l1 l2 : Prop) : A := sorry

-- Given conditions
variables (B C : A) (hM : M = midpoint B C)
variables (hAB : (A - B).abs = 15)
variables (hAC : (A - C).abs = 20)
variables (hE : E = sorry) -- E lies on AC
variables (hF : F = sorry) -- F lies on AB
variables (hAE_AF : AE = 3 * AF)
variables (hG : G = line_intersection F E A M)

-- Proof statement
theorem ratio_EG_GF : ∀ (A B C M E F G : A), M = midpoint B C → (A - B).abs = 15 → (A - C).abs = 20 → (AE = 3 * AF) → G = line_intersection F E A M → 
    ratio_segment E G G (2 / 3) :=
sorry

end ratio_EG_GF_l753_753042


namespace triangle_third_side_l753_753281

theorem triangle_third_side (a b P : ℕ) (h_a : a = 5) (h_b : b = 20) (h_P : P = 55) : ℕ :=
  ∃ c : ℕ, P = a + b + c ∧ c = 30

end triangle_third_side_l753_753281


namespace inv_domain_inv_range_l753_753483

noncomputable def domain (f : ℝ → ℝ) : set ℝ := { x | ∃ y, f x = y }

noncomputable def range (f : ℝ → ℝ) : set ℝ := { y | ∃ x, f x = y }

noncomputable def inv (x : ℝ) : ℝ := 1 / x

theorem inv_domain : domain inv = {x : ℝ | x ≠ 0} :=
sorry

theorem inv_range : range inv = {y : ℝ | y ≠ 0} :=
sorry

end inv_domain_inv_range_l753_753483


namespace max_properly_connected_triples_l753_753562

open Finset

/-- There are 5 airway companies, 36 cities, and each pair of cities is operated by exactly one company. 
    Prove the largest possible value of properly-connected triples is 3780. -/
theorem max_properly_connected_triples : 
  ∀ (flights : Finset (Fin 36 × Fin 36) → Fin 5),
  ∃ k : ℕ, k = 3780 ∧ 
  ∀ v : Fin 36, 
    let x := (λ i : Fin 5, (univ.filter (λ e : Fin 36 × Fin 36, (e.1 = v ∨ e.2 = v) ∧ flights {e} = i)).card) in 
    ∑ i : Fin 5, x i * (x i - 1) / 2 ≥ k :=
begin
  sorry
end

end max_properly_connected_triples_l753_753562


namespace trajectory_and_max_area_l753_753998

theorem trajectory_and_max_area (P : ℝ × ℝ) (Q : ℝ × ℝ)
  (h1 : P.1 ^ 2 / 4 + P.2 ^ 2 / 2 = 1)
  (h2 : Q = (1 / 3) • P) :
  ((Q.1 ^ 2) / (4 / 9) + (Q.2 ^ 2) / (2 / 9) = 1) ∧
  (∀ (l : ℝ → ℝ) (k n : ℝ) (h3 : ∀ x, l x = k * x + n)
     (h4 : ∃ x y, l x = y ∧ (x ^ 2 / (4 / 9) + y ^ 2 / (2 / 9) = 1)
          ∧ (x ^ 2 + y ^ 2 = 4 / 9)),
   ∃ (A B : ℝ × ℝ), (A.1 ^ 2 + A.2 ^ 2 = 4 / 9) ∧ (B.1 ^ 2 + B.2 ^ 2 = 4 / 9)
     ∧ A ≠ B
     ∧ ∃ (O : ℝ × ℝ) (origin : O = (0, 0)),
       (∀ O O',
         O + O' = (k * A.1 + n - k * B.1 - n, k * A.2 + n - k * B.2 - n) →
         O = origin →
         area_of_triangle O A B ≤ 2 / 9) :=
sorry

end trajectory_and_max_area_l753_753998


namespace car_speed_range_l753_753813

variables (a b x : ℝ) (d y : ℝ)

-- Given constants and conditions
def condition_1 : 0 < x ∧ x < 17 := sorry
def condition_2 : a = 1 ∧ b = 4 := sorry
def condition_3 : (x = 6 → d = 10) ∧ (x = 16 → d = 50) := sorry

-- The safe distance d based on x
def safe_distance (x : ℝ) (a b : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 6 then x + b
  else if 6 < x ∧ x < 17 then (a / 6) * x^2 + (x / 3) + 2
  else 0

-- The function y in terms of x
def y_function (x : ℝ) (a b : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 6 then (3714 + 12*x) / x
  else if 6 < x ∧ x < 17 then (2*x^2 + 4*x + 3690) / x
  else 0

-- The requirement for y to be less than or equal to 280
def y_requirement (x : ℝ) : ℝ :=
  y_function x 1 4 ≤ 280

-- Propositional statement that encodes the proof problem
theorem car_speed_range {x : ℝ} (hx : 0 < x ∧ x < 17) : 
  (y_requirement x → 15 ≤ x ∧ x < 17) :=
  sorry

end car_speed_range_l753_753813


namespace total_balls_bought_l753_753086

-- Definitions from conditions
def total_money : ℤ := 1
def cost_per_plastic_ball : ℚ := 1/60 
def cost_per_glass_ball : ℚ := 1/36 
def cost_per_wooden_ball : ℚ := 1/45 
def num_plastic_balls : ℤ := 10
def num_glass_balls : ℤ := 10

-- Define the theorem to prove the total number of balls bought
theorem total_balls_bought : 
  let money_spent_on_plastic_and_glass := (cost_per_plastic_ball + cost_per_glass_ball) * num_plastic_balls in
  let remaining_money := total_money - money_spent_on_plastic_and_glass in
  let num_wooden_balls := remaining_money / cost_per_wooden_ball in
  num_plastic_balls + num_glass_balls + num_wooden_balls = 45 :=
sorry

end total_balls_bought_l753_753086


namespace solve_quadratic_equation_l753_753934

theorem solve_quadratic_equation (x : ℝ) :
    2 * x * (x - 5) = 3 * (5 - x) ↔ (x = 5 ∨ x = -3/2) :=
by
  sorry

end solve_quadratic_equation_l753_753934


namespace part_I_part_II_l753_753071

-- Define the sequence and the sum of the first n terms
def S (n : ℕ) (k : ℝ) : ℝ := k * n^2 + n

-- Statement for part I
theorem part_I (k : ℝ) :
  (∀ (n : ℕ), n ≠ 0 → 
  let a_n : ℕ → ℝ := λ n, if n = 1 then S 1 k else S n k - S (n - 1) k
  in a_n 1 = k + 1 ∧ (∀ (n : ℕ), n > 1 → a_n n = 2 * k * n - k + 1)) := 
by 
  intro k
  intro n hn
  let a_n := λ n : ℕ, if n = 1 then S 1 k else S n k - S (n - 1) k
  split
  { unfold S a_n
    simp [hn]
    sorry -- proof omitted
  }
  { intro hn1
    unfold S a_n
    split_ifs
    { simp [hn1]
      sorry -- proof omitted
    }
    { sorry -- proof omitted
    } }

-- Define the conditions for part II
def geometric_seq (a b c : ℝ) : Prop := b^2 = a * c

-- Statement for part II
theorem part_II (k : ℝ) :
  (∀ (m : ℕ), m ≠ 0 →
  let a_n : ℕ → ℝ := λ n, if n = 1 then S 1 k else S n k - S (n - 1) k
  in geometric_seq (a_n m) (a_n (2 * m)) (a_n (4 * m)) → k = 0 ∨ k = 1) :=
by 
  intro k
  intro m hm
  let a_n := λ n, if n = 1 then S 1 k else S n k - S (n - 1) k
  unfold geometric_seq
  sorry -- proof omitted

end part_I_part_II_l753_753071


namespace monotonic_range_of_a_l753_753365

def f (a: ℝ) (x: ℝ) : ℝ := if x < 1 then a*x + 1 else x + 3

theorem monotonic_range_of_a :
  ∀ a: ℝ, (∀ x y: ℝ, f a x ≤ f a y) ↔ (0 < a ∧ a ≤ 3) :=
by
  intros
  sorry

end monotonic_range_of_a_l753_753365


namespace most_stable_city_l753_753264

def variance_STD : ℝ := 12.5
def variance_A : ℝ := 18.3
def variance_B : ℝ := 17.4
def variance_C : ℝ := 20.1

theorem most_stable_city : variance_STD < variance_A ∧ variance_STD < variance_B ∧ variance_STD < variance_C :=
by {
  -- Proof skipped
  sorry
}

end most_stable_city_l753_753264


namespace beautiful_39th_moment_l753_753047

def is_beautiful (h : ℕ) (mm : ℕ) : Prop :=
  (h + mm) % 12 = 0

def start_time := (7, 49)

noncomputable def find_39th_beautiful_moment : ℕ × ℕ :=
  (15, 45)

theorem beautiful_39th_moment :
  find_39th_beautiful_moment = (15, 45) :=
by
  sorry

end beautiful_39th_moment_l753_753047


namespace monomial_properties_l753_753817

def coefficient (m : String) : ℤ := 
  if m = "-2xy^3" then -2 
  else sorry

def degree (m : String) : ℕ := 
  if m = "-2xy^3" then 4 
  else sorry

theorem monomial_properties : coefficient "-2xy^3" = -2 ∧ degree "-2xy^3" = 4 := 
by 
  exact ⟨rfl, rfl⟩

end monomial_properties_l753_753817


namespace intersection_circumcircle_C_P_Q_l753_753767

-- Definitions from the problem
variables (A B C P Q R X : Type)
variables (h₁ : PointOnSide P B C) 
          (h₂ : PointOnSide Q C A) 
          (h₃ : PointOnSide R A B)
          
-- Intersection condition from the problem
variable (h₄ : SecondIntersection X (Circumcircle A Q R) (Circumcircle B R P))

-- Goal: 
theorem intersection_circumcircle_C_P_Q :
  OnCircumcircle X (Circumcircle C P Q) :=
sorry

end intersection_circumcircle_C_P_Q_l753_753767


namespace alcohol_concentration_correct_l753_753179

structure Vessel :=
  (capacity : ℕ)
  (alcohol_concentration : ℚ) -- Representing percentage as a rational number for precision

def vessels : List Vessel :=
  [ {capacity := 2, alcohol_concentration := 0.30},
    {capacity := 6, alcohol_concentration := 0.40},
    {capacity := 4, alcohol_concentration := 0.25},
    {capacity := 3, alcohol_concentration := 0.35},
    {capacity := 5, alcohol_concentration := 0.20},
    {capacity := 7, alcohol_concentration := 0.50} ]

def total_capacity : ℕ := 30

noncomputable def total_alcohol (vessels : List Vessel) : ℚ :=
  vessels.foldl (λ acc v, acc + v.capacity * v.alcohol_concentration) 0

noncomputable def new_concentration : ℚ :=
  total_alcohol vessels / total_capacity

theorem alcohol_concentration_correct :
  new_concentration = 0.3183 := sorry

end alcohol_concentration_correct_l753_753179


namespace farmer_sales_after_fee_l753_753575

noncomputable def total_sales_after_fee : ℝ :=
let sales_potatoes := 250 / 25 * 1.9 in 
let sales_carrots := 320 / 20 * 2 in 
let sales_fresh_tomatoes := 480 / 2 * 1 in 
let sales_canned_tomatoes := (480 / 2) / 10 * 15 in 
let total_sales := sales_potatoes + sales_carrots + sales_fresh_tomatoes + sales_canned_tomatoes in
let market_fee := 0.05 * total_sales in 
total_sales - market_fee

theorem farmer_sales_after_fee : total_sales_after_fee = 618.45 :=
by
  -- sorry is used here to skip the proof
  sorry

end farmer_sales_after_fee_l753_753575


namespace find_integers_for_perfect_square_l753_753276

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

theorem find_integers_for_perfect_square :
  {x : ℤ | is_perfect_square (x^4 + x^3 + x^2 + x + 1)} = {-1, 0, 3} :=
by
  sorry

end find_integers_for_perfect_square_l753_753276


namespace complement_intersection_l753_753342

open Set

-- Definitions based on conditions given
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

-- The mathematical proof problem
theorem complement_intersection :
  (U \ A) ∩ B = {1, 3, 7} :=
by
  sorry

end complement_intersection_l753_753342


namespace cheddar_cheese_sticks_count_l753_753409

-- Definitions for the given conditions
variable (P : ℕ) (M : ℕ := 30) (J : ℕ := 45)
-- Probability condition: half of the sticks are pepperjack
variable (prob_condition : J = P / 2)

-- The number of cheddar cheese sticks to be proved
def cheddar_cheese_sticks := P - (M + J)

theorem cheddar_cheese_sticks_count : cheddar_cheese_sticks P = 15 :=
by
  -- Given conditions translated into Lean
  have h1 : M = 30 := rfl
  have h2 : J = 45 := rfl
  have h3 : J = P / 2 := prob_condition
  -- We know P = 2 * J
  have hP : P = 2 * J := by linarith
  -- Therefore, P = 45 * 2
  have hP_val : P = 90 := by linarith
  -- Calculate the number of cheddar cheese sticks
  show cheddar_cheese_sticks P = 15
  sorry

end cheddar_cheese_sticks_count_l753_753409


namespace no_intersection_at_roots_l753_753834

theorem no_intersection_at_roots :
  let r1 := 1
  let r3 := 3
  let f := λ x => x^2 - 4 * x + 3
  let g1 := λ y => y = x^2 - 4 * x + 4
  let g2 := λ y => y = x
  f r1 = 0 ∧ f r3 = 0 →
  ¬ (∀ x, g1 x = g2 x) :=
by {
  let r1 := 1
  let r3 := 3
  let f := λ x => x^2 - 4 * x + 3
  let g1 := λ x => y = x^2 - 4 * x + 4
  let g2 := λ x => y = x
  have h1 : f r1 = 0 := by sorry
  have h3 : f r3 = 0 := by sorry
  have no_root4 : g1 4 ≠ g2 4 := by sorry
  sorry
}

end no_intersection_at_roots_l753_753834


namespace magnitude_c_l753_753433

theorem magnitude_c (c : ℂ) : 
  (∀ x : ℂ, (x^2 - 2*x + 2) = 0 ∨ (x^2 - c*x + 4) = 0 ∨ (x^2 - 4*x + 8) = 0 → 
    (x = 1 + complex.i ∨ x = 1 - complex.i ∨ x = 2 + 2*complex.i ∨ x = 2 - 2*complex.i)) →
  |c| = √10 := 
sorry

end magnitude_c_l753_753433


namespace extra_charge_per_wand_l753_753551

theorem extra_charge_per_wand
  (cost_per_wand : ℕ)
  (num_wands : ℕ)
  (total_collected : ℕ)
  (num_wands_sold : ℕ)
  (h_cost : cost_per_wand = 60)
  (h_num_wands : num_wands = 3)
  (h_total_collected : total_collected = 130)
  (h_num_wands_sold : num_wands_sold = 2) :
  ((total_collected / num_wands_sold) - cost_per_wand) = 5 :=
by
  -- Proof goes here
  sorry

end extra_charge_per_wand_l753_753551


namespace count_concave_numbers_l753_753039

-- Definition of a concave number
def isConcaveNumber (H T O : ℕ) : Prop :=
  H ≠ T ∧ T ≠ O ∧ H ≠ O ∧ H > T ∧ O > T

-- Statement of the problem as a Lean theorem
theorem count_concave_numbers : 
  ∃ n : ℕ, 
  n = 240 ∧
  n = (Finset.range 10).sum (λ T, 
    let possibleH := (Finset.range (T+1) 10 \ {T}) in -- H > T
    let possibleO := (Finset.range (T+1) 10 \ {T}) in -- O > T
    (possibleH.card * (possibleO.card - 1))) :=
sorry

end count_concave_numbers_l753_753039


namespace Elliot_average_speed_l753_753267

noncomputable def total_distance : ℝ := 120
noncomputable def total_time : ℝ := 4

def average_speed : ℝ := total_distance / total_time

theorem Elliot_average_speed : average_speed = 30 :=
by
  sorry

end Elliot_average_speed_l753_753267


namespace parallelogram_angle_E_l753_753092

/-- Given that quadrilateral EFGH is a parallelogram and ∠EGH = 70°,
    prove that the degree measure of angle E is 110° -/
theorem parallelogram_angle_E (EFGH : Type) [parallelogram EFGH] (angle_EGH : ℝ) (h : angle_EGH = 70) : ∃ angle_E : ℝ, angle_E = 110 :=
  sorry

end parallelogram_angle_E_l753_753092


namespace find_a_l753_753321

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x >= 0 then a^x else a^(-x)

theorem find_a (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a)
(h_ge : ∀ x : ℝ, x >= 0 → f x a = a ^ x)
(h_a_gt_1 : a > 1)
(h_sol : ∀ x : ℝ, f x a ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2) :
a = 2 :=
sorry

end find_a_l753_753321


namespace range_lambda_l753_753472

def difference_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) < a (n + 1) - a n

theorem range_lambda (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ) (h1 : difference_decreasing_sequence a)
  (h2 : ∀ n : ℕ, 2 * S n = 3 * a n + 2 * λ - 1) : λ > 1 / 2 :=
sorry

end range_lambda_l753_753472


namespace map_distance_correct_l753_753231

noncomputable def distance_on_map : ℝ :=
  let speed := 60  -- miles per hour
  let time := 6.5  -- hours
  let scale := 0.01282051282051282 -- inches per mile
  let actual_distance := speed * time -- in miles
  actual_distance * scale -- convert to inches

theorem map_distance_correct :
  distance_on_map = 5 :=
by 
  sorry

end map_distance_correct_l753_753231


namespace min_value_proof_l753_753794

open Real

def minValue (r s t : Real) : Real :=
  (r-1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2

theorem min_value_proof :
  (∃ r s t : Real, 1 ≤ r ∧ r ≤ s ∧ s ≤ t ∧ t ≤ 4 ∧ minValue r s t = 12 - 8 * sqrt 2) :=
  sorry

end min_value_proof_l753_753794


namespace snowdrift_depth_l753_753894

theorem snowdrift_depth :
  ∀ (d1 d2 d3 d4 : ℕ), 
    (d2 = d1 / 2) → 
    (d3 = d2 + 6) → 
    (d4 = d3 + 18) → 
    (d4 = 34) → 
    (d1 = 20) :=
by {
  intros d1 d2 d3 d4 h_d2 h_d3 h_d4 h_end,
  -- The proof goes here
  sorry
}

end snowdrift_depth_l753_753894


namespace triple_solution_unique_l753_753625

theorem triple_solution_unique (a b c n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n) :
  (a^2 + b^2 = n * Nat.lcm a b + n^2) ∧
  (b^2 + c^2 = n * Nat.lcm b c + n^2) ∧
  (c^2 + a^2 = n * Nat.lcm c a + n^2) →
  (a = n ∧ b = n ∧ c = n) :=
by
  sorry

end triple_solution_unique_l753_753625


namespace median_length_l753_753525

theorem median_length {A B C M : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space M] 
  (AB AC BC : ℝ) (AB_10 : AB = 10) (AC_10 : AC = 10) (BC_12 : BC = 12) (median_AM : A ≠ B) : 
  AM = 8 := 
by
  sorry

end median_length_l753_753525


namespace Jame_gold_bars_left_l753_753050

theorem Jame_gold_bars_left (initial_bars : ℕ) (tax_rate : ℚ) (loss_rate : ℚ) :
  initial_bars = 60 → tax_rate = 0.1 → loss_rate = 0.5 →
  let bars_after_tax := initial_bars - (initial_bars * tax_rate).toNat in
  let bars_after_divorce := (bars_after_tax * (1 - loss_rate).toRat).toNat in
  bars_after_divorce = 27 :=
by
  intros h_initial h_tax h_loss
  let bars_after_tax := initial_bars - (initial_bars * tax_rate).toNat
  let bars_after_divorce := (bars_after_tax * (1 - loss_rate).toRat).toNat
  have h1 : initial_bars = 60, from h_initial
  have h2 : tax_rate = 0.1, from h_tax
  have h3 : loss_rate = 0.5, from h_loss
  sorry

end Jame_gold_bars_left_l753_753050


namespace map_distance_8_cm_l753_753370

-- Define the conditions
def scale : ℕ := 5000000
def actual_distance_km : ℕ := 400
def actual_distance_cm : ℕ := 40000000
def map_distance_cm (x : ℕ) : Prop := x * scale = actual_distance_cm

-- The theorem to be proven
theorem map_distance_8_cm : ∃ x : ℕ, map_distance_cm x ∧ x = 8 :=
by
  use 8
  unfold map_distance_cm
  norm_num
  sorry

end map_distance_8_cm_l753_753370


namespace probability_top_two_same_suit_l753_753591

theorem probability_top_two_same_suit :
  let deck_size := 52
  let suits := 4
  let cards_per_suit := 13
  let first_card_prob := (13 / 52 : ℚ)
  let remaining_cards := 51
  let second_card_same_suit_prob := (12 / 51 : ℚ)
  first_card_prob * second_card_same_suit_prob = (1 / 17 : ℚ) :=
by
  sorry

end probability_top_two_same_suit_l753_753591


namespace cube_face_area_l753_753185

theorem cube_face_area (V : ℝ) (hV : V = 125) : ∃ A : ℝ, A = 25 ∧ V = ((∛(A*A))^3) :=
by
  sorry

end cube_face_area_l753_753185


namespace quadratic_trinomial_l753_753355

variable {R : Type} [CommRing R]

variables (x y : R)

def A := x^2 * y + 2
def B := 3 * x^2 * y + x
def C := 4 * x^2 * y - x * y

theorem quadratic_trinomial : (A + B - C) = (2 + x + x * y) := 
by sorry

end quadratic_trinomial_l753_753355


namespace triangle_inequality_equality_condition_l753_753428

variables {a b c u v w P Q : ℝ}

theorem triangle_inequality (T1: { a b c : ℝ }, T2: { u v w : ℝ }) (area_P : P = sqrt (s * (s - a) * (s - b) * (s - c)))
    (area_Q : Q = sqrt (t * (t - u) * (t - v) * (t - w))) (s t : ℝ)
    (P_area : s = (a + b + c) / 2) (Q_area : t = (u + v + w) / 2) :
  16 * P * Q ≤ a^2 * (-u^2 + v^2 + w^2) + b^2 * (u^2 - v^2 + w^2) + c^2 * (u^2 + v^2 - w^2) :=
sorry

theorem equality_condition (a b c u v w : ℝ) (γ φ : ℝ)
  (similar_triangles : a / b = u / v ∧ γ = φ) :
  16 * P * Q = a^2 * (-u^2 + v^2 + w^2) + b^2 * (u^2 - v^2 + w^2) + c^2 * (u^2 + v^2 - w^2) :=
sorry

end triangle_inequality_equality_condition_l753_753428


namespace population_equal_in_18_years_l753_753882

theorem population_equal_in_18_years :
  ∃ n : ℕ, n = 18 ∧
    ∀ (initial_population_x initial_population_y decrease_rate_x increase_rate_y : ℕ),
      initial_population_x = 78000 →
      decrease_rate_x = 1200 →
      initial_population_y = 42000 →
      increase_rate_y = 800 →
      initial_population_x - decrease_rate_x * n = initial_population_y + increase_rate_y * n :=
by
  use 18
  intro initial_population_x initial_population_y decrease_rate_x increase_rate_y
  intros hp_x hr_x hp_y hr_y
  rw [hp_x, hr_x, hp_y, hr_y]
  linarith

end population_equal_in_18_years_l753_753882


namespace find_integer_a_l753_753961

theorem find_integer_a (x d e a : ℤ) :
  ((x - a)*(x - 8) - 3 = (x + d)*(x + e)) → (a = 6) :=
by
  sorry

end find_integer_a_l753_753961


namespace number_of_correct_statements_l753_753012

variable (f : ℝ → ℝ)

-- Conditions as Lean definitions
def domain : Prop := ∀ x, x ∈ ℝ
def even_function : Prop := ∀ x, f(2 * x + 1) = f(1 - 2 * x)
def sum_equality : Prop := ∀ x, f(2 - x) + f(2 + x) = 6

-- Statements to verify
def statement_2 : Prop := f(22) = 3
def statement_3 : Prop := ∀ x, f(x+5) = f(5-x)
def statement_4 : Prop := f(1) + f(2) + f(3) + ... + f(19) = 57

-- Main theorem
theorem number_of_correct_statements (hdomain : domain f) (heven : even_function f) (hsum : sum_equality f) :
  statement_2 f ∧ statement_3 f ∧ statement_4 f :=
sorry

end number_of_correct_statements_l753_753012


namespace jason_investing_months_l753_753468

noncomputable def initial_investment (total_amount earned_amount_per_month : ℕ) := total_amount / 3
noncomputable def months_investing (initial_investment earned_amount_per_month : ℕ) := (2 * initial_investment) / earned_amount_per_month

theorem jason_investing_months (total_amount earned_amount_per_month : ℕ) 
  (h1 : total_amount = 90) 
  (h2 : earned_amount_per_month = 12) 
  : months_investing (initial_investment total_amount earned_amount_per_month) earned_amount_per_month = 5 := 
by
  sorry

end jason_investing_months_l753_753468


namespace ratio_larva_to_cocoon_l753_753407

theorem ratio_larva_to_cocoon (total_days : ℕ) (cocoon_days : ℕ)
  (h1 : total_days = 120) (h2 : cocoon_days = 30) :
  (total_days - cocoon_days) / cocoon_days = 3 := by
  sorry

end ratio_larva_to_cocoon_l753_753407


namespace marble_187_is_blue_l753_753589

noncomputable def marble_color (n : ℕ) : string :=
  let cycle_length := 15 in
  let position_in_cycle := (n - 1) % cycle_length + 1 in
  if position_in_cycle <= 6 then "red"
  else if position_in_cycle <= 11 then "blue"
  else "green"

theorem marble_187_is_blue : marble_color 187 = "blue" :=
by sorry

end marble_187_is_blue_l753_753589


namespace number_of_factors_l753_753002

theorem number_of_factors (M : ℕ) (h : M = 2^5 * 3^4 * 5^3 * 7^3) : (finset.range (6)).card * (finset.range (5)).card * (finset.range (4)).card * (finset.range (4)).card = 480 := 
by 
  rw finset.card_range 6,
  rw finset.card_range 5,
  rw finset.card_range 4,
  rw finset.card_range 4,
  simp,
  exact h,
  sorry

end number_of_factors_l753_753002


namespace correct_length_DF_l753_753399

open Real

variable (A B C D E F : Point)
variable (AD BC : ℝ)
variable {AB DC : ℝ}
variable {B_on_DE : B ∈ line E D}
variable {C_eq_mid_DF : C = midpoint D F}

noncomputable def length_DF (AD BC AB DC : ℝ) (B_on_DE : B ∈ line E D) (C_eq_mid_DF : C = midpoint D F) : ℝ :=
  if AD = 7 ∧ BC = 7 ∧ AB = 6 ∧ DC = 14 then 28 else 0

def is_correct_length_DF : Prop :=
  length_DF A B C D E F AD BC AB DC B_on_DE C_eq_mid_DF = 28

-- Main theorem statement
theorem correct_length_DF :
  is_correct_length_DF :=
begin
  sorry,  -- Proof will be inserted here
end

end correct_length_DF_l753_753399


namespace reporters_not_covering_politics_l753_753554

-- Assigning proportions for simplicity.
variables (total_reporters : ℝ) (local_politics_reporters : ℝ) (non_local_politics_ratio : ℝ)

-- Defining the conditions given in the problem.
def condition1 : Prop :=
  local_politics_reporters = 0.18 * total_reporters

def condition2 : Prop :=
  non_local_politics_ratio = 0.40

-- Defining the final proof to equate the given percentages leading to the final correct answer.
theorem reporters_not_covering_politics (h1 : condition1 total_reporters local_politics_reporters)
  (h2 : condition2 non_local_politics_ratio) :
  let P := local_politics_reporters / 0.60 in
  100 - P = 70 :=
by 
  sorry

end reporters_not_covering_politics_l753_753554


namespace Tamika_probability_l753_753810

def Tamika_set : set ℕ := {7, 11, 14}
def Carlos_set : set ℕ := {2, 5, 7}

def Tamika_sums : set ℕ := {x + y | x ∈ Tamika_set, y ∈ Tamika_set, x ≠ y}
def Carlos_products : set ℕ := {x * y | x ∈ Carlos_set, y ∈ Carlos_set, x ≠ y}

theorem Tamika_probability:
  (∑ t in Tamika_sums, ∑ c in Carlos_products, if t >= c then 1 else 0) / (Tamika_sums.card * Carlos_products.card) =
  2 / 3 :=
by sorry

end Tamika_probability_l753_753810


namespace next_palindrome_after_2052_l753_753105

def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s == s.reverse

def product_of_digits (n : ℕ) : ℕ :=
  (toString n).toList.map (λ c, c.toNat - '0'.toNat).prod

theorem next_palindrome_after_2052 : 
  ∃ (y : ℕ), y > 2052 ∧ is_palindrome y ∧ product_of_digits y = 4 :=
begin
  use 2112,
  split,
  { norm_num },
  split,
  { unfold is_palindrome, simp },
  { unfold product_of_digits, simp }
end

end next_palindrome_after_2052_l753_753105


namespace concurrency_or_parallelism_l753_753161

noncomputable theory
open_locale classical

variables {A B C F E P Q : Type*}
variables [plane A] [plane B] [plane C] [plane F] [plane E] [plane P] [plane Q]
variables {ω : circle}

-- Assume additional conditions
variables (triangle_ABC : triangle A B C) (circumcircle_ABC : circumcircle ω triangle_ABC)
          (point_F_on_AC : point_on_line F AC) (point_E_on_AB : point_on_line E AB)
          (point_P_on_ω : point_on_circle P ω) (point_Q_on_ω : point_on_circle Q ω)
          (angle_AFB_90 : angle AFB = 90) (angle_AEC_90 : angle AEC = 90)
          (angle_APE_90 : angle APE = 90) (angle_AQF_90 : angle AQF = 90)

theorem concurrency_or_parallelism
  (triangle_ABC : triangle A B C)
  (circumcircle_ABC : circumcircle ω triangle_ABC)
  (point_F_on_AC : point_on_line F AC)
  (point_E_on_AB : point_on_line E AB)
  (point_P_on_ω : point_on_circle P ω)
  (point_Q_on_ω : point_on_circle Q ω)
  (angle_AFB_90 : angle AFB = 90)
  (angle_AEC_90 : angle AEC = 90)
  (angle_APE_90 : angle APE = 90)
  (angle_AQF_90 : angle AQF = 90) :
  concurrent_or_parallel (line BC) (line EF) (line PQ) :=
sorry

end concurrency_or_parallelism_l753_753161


namespace seq_arithmetic_sum_log_seq_l753_753917

noncomputable def a : ℕ → ℝ
| 0     := 0      -- a_0 is not defined, but let's add a placeholder
| (n+1) := if n = 0 then 6 else 6 - 9 / a n

def log_a : ℕ → ℝ := λ n, Real.log (a n)

theorem seq_arithmetic (h : ∀ n, n ≥ 1 → a (n+1) = 6 - 9 / a n) :
  ∃ d, ∀ n ≥ 1, (1 / (a (n+1) - 3)) - (1 / (a n - 3)) = d :=
sorry

theorem sum_log_seq (h : ∀ n, n ≥ 1 → a (n+1) = 6 - 9 / a n) (a1 : a 1 = 6) :
  ∑ i in finset.range 999, log_a i = 3 + 999 * Real.log 3 :=
sorry

end seq_arithmetic_sum_log_seq_l753_753917


namespace circles_parallel_l753_753856

theorem circles_parallel {S₁ S₂ : Type} [metric_space S₁] [metric_space S₂]
  (O₁ O₂ A A₁ A₂ : S₁)
  (h₀ : intersect_circles S₁ S₂ O₁ O₂ A A₁ A₂)
  : is_parallel (line_segment O₁ A₁) (line_segment O₂ A₂) :=
sorry

end circles_parallel_l753_753856


namespace inequality_holds_l753_753754

noncomputable def largestC (α : ℝ) (x y z : ℝ) : ℝ := 16

theorem inequality_holds (α : ℝ) (x y z : ℝ) 
  (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y + y * z + z * x = α) : 
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ largestC α x y z * (x / z + z / x + 2) :=
begin
  sorry
end

end inequality_holds_l753_753754


namespace projection_of_a_onto_b_is_2_l753_753344

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : vector) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

def projection (a b : vector) : ℝ :=
  (dot_product a b) / (magnitude b)

theorem projection_of_a_onto_b_is_2 :
  let a : vector := (2,1)
  let b : vector := (3,4)
  projection a b = 2 :=
by
  sorry

end projection_of_a_onto_b_is_2_l753_753344


namespace lines_intersection_points_l753_753947

theorem lines_intersection_points :
  let line1 (x y : ℝ) := 2 * y - 3 * x = 4
  let line2 (x y : ℝ) := 3 * x + y = 5
  let line3 (x y : ℝ) := 6 * x - 4 * y = 8
  ∃ p1 p2 : (ℝ × ℝ),
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
    (p1 = (2, 5)) ∧ (p2 = (14/9, 1/3)) :=
by
  sorry

end lines_intersection_points_l753_753947


namespace ball_arrangement_possible_l753_753940

theorem ball_arrangement_possible:
  ∃ (color_sets : Fin 5 → Finset (Fin 8)), -- Fin 5 represents 5 boxes and Fin 8 represents 8 colors
    (∀ i, (color_sets i).card = 3) ∧       -- Each box must contain exactly 3 balls
    (∀ i j, i ≠ j → (color_sets i) ≠ (color_sets j)) ∧ -- No two boxes contain the same set of colors
    (∀ i, (color_sets i).disjoint (color_sets ((i + 1) % 5))) -- No two adjacent boxes should share a color
  :=
by
  sorry

end ball_arrangement_possible_l753_753940


namespace area_of_square_l753_753502

theorem area_of_square (A_circle : ℝ) (hA_circle : A_circle = 39424) (cm_to_inch : ℝ) (hcm_to_inch : cm_to_inch = 2.54) :
  ∃ (A_square : ℝ), A_square = 121.44 := 
by
  sorry

end area_of_square_l753_753502


namespace count_even_three_digit_numbers_with_sum_of_hundreds_and_tens_9_l753_753349

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_hundreds_and_tens_is_nine (n : ℕ) : Prop :=
  (n / 100) + ((n / 10) % 10) = 9

theorem count_even_three_digit_numbers_with_sum_of_hundreds_and_tens_9 :
  {n : ℕ | is_three_digit n ∧ is_even n ∧ sum_of_hundreds_and_tens_is_nine n}.to_finset.card = 45 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_of_hundreds_and_tens_9_l753_753349


namespace gathered_amount_correct_l753_753022

-- Define the number of students in each category
def full_paying_students := 20
def merit_students := 5
def financial_needs_students := 7
def special_discount_students := 3

-- Define the payment amounts
def full_price := 50.0
def merit_price := 37.5
def financial_needs_price := 25.0
def special_discount_price := 45.0

-- Define the administrative fee
def administrative_fee := 100.0

-- Calculate the total amount gathered by the students
def total_students_amount : Float :=
  (full_paying_students * full_price) +
  (merit_students * merit_price) +
  (financial_needs_students * financial_needs_price) +
  (special_discount_students * special_discount_price)

-- Calculate the total amount gathered considering the administrative fee
def total_amount_gathered : Float :=
  total_students_amount - administrative_fee

-- Theorem statement to prove the final gathered amount
theorem gathered_amount_correct :
  total_amount_gathered = 1397.5 :=
by
  -- Proof goes here
  sorry

end gathered_amount_correct_l753_753022


namespace viktor_win_minimum_dominoes_l753_753804

theorem viktor_win_minimum_dominoes (n : ℕ) :
  (∃ (cover : Π (i j : ℕ), (i < 2022) → (j < 2022) → Prop),
    (∀ (i j : ℕ), (i < 2022) → (j < 2022) → (cover i j → (i+1 < 2022 → cover (i+1) j)))) →
    (∃ (choice : Π (k : ℕ), (k < 2022^2) → Prop),
      (∀ (k : ℕ), (k < 2022^2) → (choice k → cover (k // 2022) (k % 2022)))) →
  n = 1011^2 :=
by sorry

end viktor_win_minimum_dominoes_l753_753804


namespace ellipse_area_l753_753721

theorem ellipse_area (a b : ℝ) (Cx Cy : ℝ) 
  (h1 : ∃ a b (Cx Cy : ℝ), (Cx = 5) ∧ (Cy = 3) ∧ (a = 10) ∧ (b = 4 * Real.sqrt 3))
  (h2 : ∀ x y, (x, y) = (10, 9) → ((x - Cx)^2 / a^2) + ((y - Cy)^2 / b^2) = 1) :
  π * a * b = 40 * π * Real.sqrt 3 :=
by
  sorry

end ellipse_area_l753_753721


namespace even_function_a_neg_sqrt3_l753_753014

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (x + π / 4) + √3 * Real.sin (x - π / 4)

theorem even_function_a_neg_sqrt3 {a : ℝ} :
  (∀ x : ℝ, f a (-x) = f a x) → a = -√3 :=
sorry

end even_function_a_neg_sqrt3_l753_753014


namespace roots_first_equation_roots_second_equation_roots_third_equation_l753_753470

theorem roots_first_equation : 
  ∀ x1 x2 x3 : ℝ,
  (x1 + x2 + x3 = 12 ∧ x2 + x3 = 9 ∧ x1 * x2 * x3 = 60) →
  ({x1, x2, x3} = {3, 4, 5}) := 
by
  sorry

theorem roots_second_equation :
  ∀ x1 x2 x3 : ℝ,
  (x1 + x2 + x3 = 19 ∧ x2 * x3 = 48 ∧ x1 * x2 * x3 = 240) →
  ({x1, x2, x3} = {5, 6, 8}) :=
by
  sorry

theorem roots_third_equation :
  ∀ x1 x2 x3 x4 : ℝ,
  (x1 + x2 + x3 + x4 = 18 ∧ x1 + x2 = 7 ∧ x3 * x4 = 30 ∧ x1 * x2 * x3 * x4 = 360) →
  ({x1, x2, x3, x4} = {3, 4, 5, 6}) :=
by
  sorry

end roots_first_equation_roots_second_equation_roots_third_equation_l753_753470


namespace men_who_wore_glasses_l753_753784

theorem men_who_wore_glasses (total_people : ℕ) (women_ratio men_with_glasses_ratio : ℚ)  
  (h_total : total_people = 1260) 
  (h_women_ratio : women_ratio = 7 / 18)
  (h_men_with_glasses_ratio : men_with_glasses_ratio = 6 / 11)
  : ∃ (men_with_glasses : ℕ), men_with_glasses = 420 := 
by
  sorry

end men_who_wore_glasses_l753_753784


namespace equally_spaced_markings_number_line_l753_753016

theorem equally_spaced_markings_number_line 
  (steps : ℕ) (distance : ℝ) (z_steps : ℕ) (z : ℝ)
  (h1 : steps = 4)
  (h2 : distance = 16)
  (h3 : z_steps = 2) :
  z = (distance / steps) * z_steps :=
by
  sorry

end equally_spaced_markings_number_line_l753_753016


namespace number_below_267_is_301_l753_753597

-- Define the row number function
def rowNumber (n : ℕ) : ℕ :=
  Nat.sqrt n + 1

-- Define the starting number of a row
def rowStart (k : ℕ) : ℕ :=
  (k - 1) * (k - 1) + 1

-- Define the number in the row below given a number and its position in the row
def numberBelow (n : ℕ) : ℕ :=
  let k := rowNumber n
  let startK := rowStart k
  let position := n - startK
  let startNext := rowStart (k + 1)
  startNext + position

-- Prove that the number below 267 is 301
theorem number_below_267_is_301 : numberBelow 267 = 301 :=
by
  -- skip proof details, just the statement is needed
  sorry

end number_below_267_is_301_l753_753597


namespace min_section_area_of_regular_tetrahedron_l753_753654

theorem min_section_area_of_regular_tetrahedron
  (base_edge : ℝ)
  (height : ℝ)
  (is_regular_tetrahedron : is_regular_tetrahedron P A B C)
  (section_passing_base : section_passing_base P B C)
  (base_edge = 1)
  (height = real.sqrt 2) :
    ∃ (section_area : ℝ),
      section_area = 3 * real.sqrt 14 / 28 :=
begin
  sorry
end

end min_section_area_of_regular_tetrahedron_l753_753654


namespace find_c_l753_753586

theorem find_c (w : ℝ) (c : ℝ) (h_w : w = 2) (h_point : (3, w^3) ∈ set_of (λ p : ℝ × ℝ, p.2 = p.1^2 - c)) : c = 1 :=
by
  sorry

end find_c_l753_753586


namespace range_of_omega_l753_753116

theorem range_of_omega :
  ∀ (ω : ℝ), (ω > 0) →
  (∀ x, f x = sin(ω * x + (π / 6))) →
  (∃! z, z ∈ Ioo 0 (π / 2) ∧ f z = 0) ↔ ω ∈ Ioc (5 / 3) (11 / 3) :=
by {
  sorry
}

end range_of_omega_l753_753116


namespace find_scalars_l753_753072

open Matrix

noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![-2, 0]]

def I : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 1]]

theorem find_scalars :
  ∃ (p q : ℝ), (M ⬝ M) = p • M + q • I :=
  by
    use 3, -8
    sorry

end find_scalars_l753_753072


namespace sequence_periodic_of_period_9_l753_753101

theorem sequence_periodic_of_period_9 (a : ℕ → ℤ) (h : ∀ n, a (n + 2) = |a (n + 1)| - a n) (h_nonzero : ∃ n, a n ≠ 0) :
  ∃ m, ∃ k, m > 0 ∧ k > 0 ∧ (∀ n, a (n + m + k) = a (n + m)) ∧ k = 9 :=
by
  sorry

end sequence_periodic_of_period_9_l753_753101


namespace base6_sum_l753_753968

-- Define each of the numbers in base 6
def base6_555 : ℕ := 5 * 6^2 + 5 * 6^1 + 5 * 6^0
def base6_55 : ℕ := 5 * 6^1 + 5 * 6^0
def base6_5 : ℕ := 5 * 6^0
def base6_1103 : ℕ := 1 * 6^3 + 1 * 6^2 + 0 * 6^1 + 3 * 6^0 

-- The problem statement is to prove the sum equals the expected result in base 6
theorem base6_sum : base6_555 + base6_55 + base6_5 = base6_1103 :=
by
  sorry

end base6_sum_l753_753968


namespace b_uniform_interval_l753_753073

noncomputable def interval_b := [-6, -3]

axiom b1_uniform : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → ∃ b1, b1 ∈ set.Icc (0 : ℝ) 1 

theorem b_uniform_interval  :
  ∀ b1, b1 ∈ set.Icc (0 : ℝ) 1 → ((b1 - 2) * 3) ∈ set.Icc interval_b[0] interval_b[1] :=
by
  sorry

end b_uniform_interval_l753_753073


namespace b_seq_is_arithmetic_a_sum_formula_l753_753309

noncomputable def a_seq : ℕ → ℕ
| 1       := 3
| (n + 2) := 2 * a_seq (n + 1) + 2^(n + 2) - 1

def b_seq (n : ℕ) : ℕ :=
(a_seq n - 1) / 2^n

theorem b_seq_is_arithmetic : ∀ n ≥ 1, ∃ d : ℕ, ∀ m ≥ 1, b_seq (m + 1) - b_seq m = d :=
sorry

def a_sum (n : ℕ) : ℕ :=
(∑ i in finset.range n, a_seq (i + 1))

theorem a_sum_formula (n : ℕ) : a_sum n = n * 2^(n + 1) - 2^(n + 1) + 2 + n :=
sorry

end b_seq_is_arithmetic_a_sum_formula_l753_753309


namespace color_numbers_not_divisible_sum_l753_753588

theorem color_numbers_not_divisible_sum (p : Nat) (hp : Nat.Prime p) (n : Nat) (hn : 0 < n) :
  ∃ c : Fin (p - 1) → Fin (2 * n), ∀ i : Fin (n - 1), ∀ x : Fin (p - 1) → Nat, 
  ( ∑ k in (Finset.range (i + 2)), x (Fin.cast_succ k)) % p ≠ 0 :=
sorry

end color_numbers_not_divisible_sum_l753_753588


namespace num_children_proof_l753_753054

-- Definitions and Main Problem
def legs_of_javier : ℕ := 2
def legs_of_wife : ℕ := 2
def legs_per_child : ℕ := 2
def legs_per_dog : ℕ := 4
def legs_of_cat : ℕ := 4
def num_dogs : ℕ := 2
def num_cats : ℕ := 1
def total_legs : ℕ := 22

-- Proof problem: Prove that the number of children (num_children) is equal to 3
theorem num_children_proof : ∃ num_children : ℕ, legs_of_javier + legs_of_wife + (num_children * legs_per_child) + (num_dogs * legs_per_dog) + (num_cats * legs_of_cat) = total_legs ∧ num_children = 3 :=
by
  -- Proof goes here
  sorry

end num_children_proof_l753_753054


namespace dislikeBothTVAndBooks_l753_753177

-- Definitions of the given conditions
def totalPeople : ℕ := 1650
def percentDislikeTV : ℚ := 0.35
def percentDislikeBothGivenTV : ℚ := 0.15

-- Definition of the number of people who dislike both TV and Books
def solution := 87

-- The proof problem
theorem dislikeBothTVAndBooks :
  let p1 := (percentDislikeTV * totalPeople).ceil,
      p2 := (percentDislikeBothGivenTV * p1).ceil
  in p2 = solution :=
by {
  -- Placeholder for the actual proof
  sorry
}

end dislikeBothTVAndBooks_l753_753177


namespace max_value_of_f_l753_753487

def f : ℕ → ℕ
| n := if n < 10 then n + 10 else f (n - 5)

theorem max_value_of_f : ∃ n : ℕ, f n = 19 :=
by {
  sorry
}

end max_value_of_f_l753_753487


namespace period_of_f_l753_753334

def f (x : ℝ) (ω : ℝ) : ℝ :=
  (cos (ω * x / 2))^2 + (sqrt 3) * (sin (ω * x / 2)) * (cos (ω * x / 2)) - 1 / 2

theorem period_of_f :
  ∃ ω > 0,
    (∀ x, f x ω = sin (ω * x + π / 6)) ∧
    (∀ T > 0, (∀ x, f (x + T) ω = f x ω) → T = π) ∧
    ω = 2 ∧
    ∀ x, -1 ≤ f x 2 ∧ f x 2 ≤ 1 ∧
    ∀ x, ∀ k : ℤ, k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 →
    f x 2 ≤ f (x + (π / 3)) 2 :=
sorry

end period_of_f_l753_753334


namespace bowling_tournament_order_count_l753_753381

theorem bowling_tournament_order_count :
  let num_bowlers := 6
  let num_games := 5
  let num_prizes := 6
  let match_outcomes := 2
  num_bowlers = 6 →
  num_games = 5 →
  num_prizes = 6 →
  match_outcomes = 2 →
  match_outcomes ^ num_games = 32 :=
by
  intros
  unfold num_games num_prizes match_outcomes
  exact rfl


end bowling_tournament_order_count_l753_753381


namespace selected_numbers_divisibility_l753_753295

theorem selected_numbers_divisibility (S : Finset ℕ) (hS: S ⊆ Finset.range 201) (h_card : S.card = 100) :
  ∃ a b ∈ S, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
by
  sorry

end selected_numbers_divisibility_l753_753295


namespace total_value_of_pokemon_cards_l753_753739

def number_of_rare_cards (total_cards : ℕ) (percent_rare : ℕ) : ℕ := 
  (percent_rare * total_cards) / 100

def number_of_nonrare_cards (total_cards : ℕ) (rare_cards : ℕ) : ℕ := 
  total_cards - rare_cards

def value_of_cards (rare_cards : ℕ) (nonrare_cards : ℕ) (value_rare : ℕ) (value_nonrare : ℕ) : ℕ :=
  (rare_cards * value_rare) + (nonrare_cards * value_nonrare)

theorem total_value_of_pokemon_cards :
  let jenny_cards := 6,
      orlando_cards := jenny_cards + 2,
      richard_cards := 3 * orlando_cards,
      value_rare := 10,
      value_nonrare := 3 in
  let jenny_rare := number_of_rare_cards jenny_cards 50,
      jenny_nonrare := number_of_nonrare_cards jenny_cards jenny_rare,
      orlando_rare := number_of_rare_cards orlando_cards 40,
      orlando_nonrare := number_of_nonrare_cards orlando_cards orlando_rare,
      richard_rare := number_of_rare_cards richard_cards 25,
      richard_nonrare := number_of_nonrare_cards richard_cards richard_rare in
  value_of_cards jenny_rare jenny_nonrare value_rare value_nonrare +
  value_of_cards orlando_rare orlando_nonrare value_rare value_nonrare +
  value_of_cards richard_rare richard_nonrare value_rare value_nonrare = 198 :=
by
  sorry

end total_value_of_pokemon_cards_l753_753739


namespace question_B_question_D_l753_753991

section
variables {Line Plane : Type} (m n : Line) (α β : Plane)

-- Conditions and Definitions
variable (m_diff_n : ∀ (l : Line), l = m ∨ l = n → l ≠ m)
variable (alpha_diff_beta : ∀ (p : Plane), p = α ∨ p = β → p ≠ α)
variable (m_perp_alpha : m ⊥ α)
variable (m_perp_beta : m ⊥ β)
variable (n_perp_beta : n ⊥ β)
variable (alpha_perp_beta : α ⊥ β)

-- Statements to Prove
theorem question_B : (∀ (m : Line), (m ⊥ α ∧ m ⊥ β) → α ∥ β) :=
by
  assume (m : Line) (h : m ⊥ α ∧ m ⊥ β),
  sorry

theorem question_D : (∀ (m n : Line), (m ⊥ α ∧ n ⊥ β ∧ α ⊥ β) → m ⊥ n) :=
by
  assume (m n : Line) (h : m ⊥ α ∧ n ⊥ β ∧ α ⊥ β),
  sorry
end

end question_B_question_D_l753_753991


namespace total_stamps_l753_753722

def num_foreign_stamps : ℕ := 90
def num_old_stamps : ℕ := 70
def num_both_foreign_old_stamps : ℕ := 20
def num_neither_stamps : ℕ := 60

theorem total_stamps :
  (num_foreign_stamps + num_old_stamps - num_both_foreign_old_stamps + num_neither_stamps) = 220 :=
  by
    sorry

end total_stamps_l753_753722


namespace movies_shown_eq_twenty_four_l753_753194

-- Define conditions
variables (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ)

-- Define the total number of movies calculation
noncomputable def total_movies_shown (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  screens * (open_hours / movie_duration)

-- Theorem to prove the total number of movies shown is 24
theorem movies_shown_eq_twenty_four : 
  total_movies_shown 6 8 2 = 24 :=
by
  sorry

end movies_shown_eq_twenty_four_l753_753194


namespace intersection_of_A_and_B_l753_753693

def A : Set ℝ := { x | x < 3 }
def B : Set ℝ := { x | Real.log (x - 1) / Real.log 3 > 0 }

theorem intersection_of_A_and_B :
  (A ∩ B) = { x | 2 < x ∧ x < 3 } :=
sorry

end intersection_of_A_and_B_l753_753693


namespace solve_interval_l753_753647

def f (x : ℝ) : ℝ := 10^x
def g (x : ℝ) : ℝ := Real.log x / Real.log 10 -- This is log base 10

theorem solve_interval (x : ℝ) :
  (g x + x = 4) → 3 < x ∧ x < 4 :=
by
  -- proof to be completed
  sorry

end solve_interval_l753_753647


namespace probability_palindromic_phone_number_l753_753397

/-- In a city where phone numbers consist of 7 digits, the Scientist easily remembers a phone number
    if it is a palindrome (reads the same forwards and backwards). Prove that the probability that 
    a randomly chosen 7-digit phone number is a palindrome is 0.001. -/
theorem probability_palindromic_phone_number : 
  let total_phone_numbers := 10^7
  let palindromic_phone_numbers := 10^4
  (palindromic_phone_numbers : ℝ) / total_phone_numbers = 0.001 :=
by
  let total_phone_numbers := 10^7
  let palindromic_phone_numbers := 10^4
  show (palindromic_phone_numbers : ℝ) / total_phone_numbers = 0.001 from sorry

end probability_palindromic_phone_number_l753_753397


namespace probability_odd_sum_l753_753446

theorem probability_odd_sum (m n : ℕ) (h_mn_rel_prime : Nat.coprime m n) (h_prob : (m : ℝ) / n = 3 / 14) :
  m + n = 17 :=
sorry

end probability_odd_sum_l753_753446


namespace cole_average_speed_l753_753943

noncomputable def cole_average_speed_to_work : ℝ :=
  let time_to_work := 1.2
  let return_trip_speed := 105
  let total_round_trip_time := 2
  let time_to_return := total_round_trip_time - time_to_work
  let distance_to_work := return_trip_speed * time_to_return
  distance_to_work / time_to_work

theorem cole_average_speed : cole_average_speed_to_work = 70 := by
  sorry

end cole_average_speed_l753_753943


namespace inverse_function_log_base_2_l753_753493

theorem inverse_function_log_base_2 (x : ℝ) (h : x > 0) : ∃ y : ℝ, y = (log x) / (log 2) ∧ x = 2^y :=
by
  sorry

end inverse_function_log_base_2_l753_753493


namespace ternary_even_iff_digit_sum_even_base_even_most_significant_digit_even_base_odd_digit_sum_even_l753_753976

-- Define parity for nat type
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Part (A): Ternary system
theorem ternary_even_iff_digit_sum_even (a : List ℕ) (h : ∀ x ∈ a, x < 3) : 
  even (a.foldl (λ acc x, acc + x) 0) ↔ even (a.foldl (λ acc x, acc * 3 + x) 0) := 
sorry

-- Part (B): Base-n system
-- Even base
theorem base_even_most_significant_digit_even (n : ℕ) (h : even n) (a : List ℕ) (h1 : ∀ x ∈ a, x < n) : 
  even (a.head! % 2) ↔ even (a.foldl (λ acc x, acc * n + x) 0) := 
sorry

-- Odd base
theorem base_odd_digit_sum_even (n : ℕ) (h : ¬even n) (a : List ℕ) (h1 : ∀ x ∈ a, x < n) : 
  even (a.foldl (λ acc x, acc + x) 0) ↔ even (a.foldl (λ acc x, acc * n + x) 0) := 
sorry

end ternary_even_iff_digit_sum_even_base_even_most_significant_digit_even_base_odd_digit_sum_even_l753_753976


namespace problem_l753_753354

theorem problem (α : ℝ) (h : 3 * real.sin α + real.cos α = 0) :
  1 / (real.cos (2 * α) + real.sin (2 * α)) = 5 :=
sorry

end problem_l753_753354


namespace count_valid_four_digit_numbers_l753_753000

-- Definitions and conditions
def valid_digits : Set ℕ := {2, 5, 7}

def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (∀ d ∈ n.digits, d ∈ valid_digits)

-- Statement of the problem
theorem count_valid_four_digit_numbers : 
  {n : ℕ | is_valid_four_digit_number n}.to_finset.card = 81 :=
by 
  sorry

end count_valid_four_digit_numbers_l753_753000


namespace olympic_volunteers_selection_l753_753977

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem olympic_volunteers_selection :
  (choose 4 3 * choose 3 1) + (choose 4 2 * choose 3 2) + (choose 4 1 * choose 3 3) = 34 := 
by
  sorry

end olympic_volunteers_selection_l753_753977


namespace distance_between_intersection_points_l753_753599

theorem distance_between_intersection_points :
  let ellipse_eq (x y : ℝ) := x^2 / 36 + y^2 / 16 = 1
  let parabola_eq (x y : ℝ) := x = y^2 / 5 - sqrt 5
  let foci_shared : ∃ x y : ℝ, x = -2 * sqrt 5 ∧ y = 0
  let directrix_eq : ∀ y : ℝ, y = y
  let p1 : ℝ × ℝ := 
    let y := sqrt 140 / 3 in
    (y^2 / 5 - sqrt 5, y)
  let p2 : ℝ × ℝ := 
    let y := -sqrt 140 / 3 in
    (y^2 / 5 - sqrt 5, y)
  ∃ d : ℝ, d = (2 * sqrt 140) / 3 :=
sorry

end distance_between_intersection_points_l753_753599


namespace find_m_value_l753_753290

noncomputable def g (x : ℝ) : ℝ := Real.cot (x / 3) - Real.cot x

theorem find_m_value :
  (∀ x : ℝ, x ≠ 0 → g(x) = (λ m : ℝ, Real.sin (m * x) / (Real.sin (x / 3) * Real.sin x)) (2 / 3)) :=
begin
  sorry
end

end find_m_value_l753_753290


namespace max_term_1992_l753_753826

noncomputable def max_term_sequence (n : ℕ) : ℕ :=
  let seq_term := λ k : ℕ, k * (Nat.choose n k)
  Finset.fold max 0 (Finset.image seq_term (Finset.range (n + 1)))

theorem max_term_1992 : max_term_sequence 1992 = 997 * (Nat.choose 1992 997) :=
  sorry

end max_term_1992_l753_753826


namespace quadratic_no_real_solutions_l753_753067

theorem quadratic_no_real_solutions (a : ℝ) (h₀ : 0 < a) (h₁ : a^3 = 6 * (a + 1)) : 
  ∀ x : ℝ, ¬ (x^2 + a * x + a^2 - 6 = 0) :=
by
  sorry

end quadratic_no_real_solutions_l753_753067


namespace jennifer_fruits_left_l753_753055

theorem jennifer_fruits_left:
  (apples = 2 * pears) →
  (cherries = oranges / 2) →
  (grapes = 3 * apples) →
  pears = 15 →
  oranges = 30 →
  pears_given = 3 →
  oranges_given = 5 →
  apples_given = 5 →
  cherries_given = 7 →
  grapes_given = 3 →
  (remaining_fruits =
    (pears - pears_given) +
    (oranges - oranges_given) +
    (apples - apples_given) +
    (cherries - cherries_given) +
    (grapes - grapes_given)) →
  remaining_fruits = 157 :=
by
  intros
  sorry

end jennifer_fruits_left_l753_753055


namespace sum_a_divisors_l753_753287

theorem sum_a_divisors 
  (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : ∀ k ∈ {2, 3, 4, 5, 6}, 
    (a_1 / (k^2 + 1) + a_2 / (k^2 + 2) + a_3 / (k^2 + 3) + a_4 / (k^2 + 4) + a_5 / (k^2 + 5) = 1 / k^2)) : 
  (a_1 / 2 + a_2 / 3 + a_3 / 4 + a_4 / 5 + a_5 / 6) = 57 / 64 :=
by
  sorry

end sum_a_divisors_l753_753287


namespace average_marks_correct_l753_753165

def marks := [76, 65, 82, 62, 85]
def num_subjects := 5
def total_marks := marks.sum
def avg_marks := total_marks / num_subjects

theorem average_marks_correct : avg_marks = 74 :=
by sorry

end average_marks_correct_l753_753165


namespace sum_x_y_650_l753_753372

theorem sum_x_y_650 (x y : ℤ) (h1 : x - y = 200) (h2 : y = 225) : x + y = 650 :=
by
  sorry

end sum_x_y_650_l753_753372


namespace m_divides_n_l753_753069

theorem m_divides_n 
  (m n : ℕ) 
  (hm_pos : 0 < m) 
  (hn_pos : 0 < n) 
  (h : 5 * m + n ∣ 5 * n + m) 
  : m ∣ n :=
sorry

end m_divides_n_l753_753069


namespace PR_length_l753_753886

theorem PR_length (TU QR SU : ℝ) (h1 : ∆PQR ∼ ∆STU) (h2 : TU = 24) (h3 : QR = 30) (h4 : SU = 18) : 
  PR = 22.5 := 
by
  sorry

end PR_length_l753_753886


namespace distances_cardinality_eq_l753_753640

/-- 
  Define \( \mathcal{T}_k \) to be the set of lattice points \{(x, y) \mid x, y = 0, 1, \ldots, k-1 \} 
  for \( k \ge 2 \). Define \( d_1(k) \), \( d_2(k) \), \(\ldots\) to be the distinct distances 
  between any two points in \(\mathcal{T}_k\) in decreasing order. Let \( S_i(k) \) be the number 
  of distances equal to \( d_i(k) \).
  We want to prove that for any three positive integers \( m > n > i \), we have \( S_i(m) = S_i(n) \).
-/
theorem distances_cardinality_eq 
  (k : ℕ) (hk : k ≥ 2)
  (T_k := {(x, y) | x ∈ fin k ∧ y ∈ fin k} : set (ℕ × ℕ))
  (d : ℕ → ℝ) (S : ℕ → ℕ → ℕ)
  (h1 : ∀ k, ∃ f : ℕ → ℝ, strict_mono_decr f ∧ ∀ i, d i = f i)
  (h2 : ∀ k, (S i k) = ∑ j in finset.range (k * k), if dist j = d i then 1 else 0)
  (m n i : ℕ) (hmn : m > n > i) :
  S i m = S i n :=
by sorry

end distances_cardinality_eq_l753_753640


namespace y_expression_value_l753_753831

theorem y_expression_value
  (y : ℝ)
  (h : y + 2 / y = 2) :
  y^6 + 3 * y^4 - 4 * y^2 + 2 = 2 := sorry

end y_expression_value_l753_753831


namespace collinear_centers_diagonals_intersection_l753_753883

variables {A B C D P I I_D : Type}
variables [InCircumscribedQuadrilateral A B C D] [IntersectionOfDiagonals A C B D P]
variables [IncircleCenter ABC I] [ExcircleCenter ADC I_D AC]

theorem collinear_centers_diagonals_intersection :
  Collinear P I I_D :=
sorry

end collinear_centers_diagonals_intersection_l753_753883


namespace gold_bars_left_after_tax_and_divorce_l753_753048

def initial_gold_bars := 60
def tax_rate := 0.10
def tax_paid := initial_gold_bars * tax_rate
def remaining_after_tax := initial_gold_bars - tax_paid
def divorce_loss := remaining_after_tax / 2
def gold_bars_left := remaining_after_tax - divorce_loss

theorem gold_bars_left_after_tax_and_divorce :
  gold_bars_left = 27 := 
by 
  simp [initial_gold_bars, tax_rate, tax_paid, remaining_after_tax, divorce_loss, gold_bars_left]
  sorry

end gold_bars_left_after_tax_and_divorce_l753_753048


namespace probability_abs_x_le_one_l753_753582

noncomputable def geometric_probability (a b c d : ℝ) : ℝ := (b - a) / (d - c)

theorem probability_abs_x_le_one : 
  ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 3 →  
  geometric_probability (-1) 1 (-1) 3 = 1 / 2 := 
by
  sorry

end probability_abs_x_le_one_l753_753582


namespace sqrt_multiplication_and_subtraction_l753_753608

theorem sqrt_multiplication_and_subtraction :
  (Real.sqrt 21 * Real.sqrt 7 - Real.sqrt 3) = 6 * Real.sqrt 3 := 
by
  sorry

end sqrt_multiplication_and_subtraction_l753_753608


namespace binomial_constant_term_is_minus_80_l753_753708

noncomputable def binomial_constant_term (n r : ℕ) : ℚ :=
  (-2 : ℚ)^r * Nat.choose n r * ((1 / x ^ (1/3 : ℚ))^r) * ((x^ (1/2 : ℚ))^ (n - r))

theorem binomial_constant_term_is_minus_80 (h1 : (-2 : ℚ)^3 * Nat.choose 5 3 = -80) :
  binomial_constant_term 5 3 = -80 :=
sorry

end binomial_constant_term_is_minus_80_l753_753708


namespace inequality_sum_squares_l753_753430

theorem inequality_sum_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 :=
sorry

end inequality_sum_squares_l753_753430


namespace Statement2_l753_753333

-- Define the sets and properties
variable (Freshmen Humans GraduateStudents Pondering : Type) 

-- Conditions
axiom cond1 : ∀ f : Freshmen, f ∈ Humans
axiom cond2 : ∀ g : GraduateStudents, g ∈ Humans
axiom cond3 : ∃ g : GraduateStudents, g ∈ Pondering

-- Statement to be proved
theorem Statement2 : ∃ h : Humans, h ∈ Pondering :=
by
  -- Proof by translating the conditions
  sorry

end Statement2_l753_753333


namespace avery_donates_16_clothes_l753_753225

theorem avery_donates_16_clothes : 
  let Shirts := 4
  let Pants := 4 * 2
  let Shorts := (4 * 2) / 2
in Shirts + Pants + Shorts = 16 :=
by 
  let Shirts := 4
  let Pants := 4 * 2
  let Shorts := (4 * 2) / 2
  show Shirts + Pants + Shorts = 16
  sorry

end avery_donates_16_clothes_l753_753225


namespace inequality_one_inequality_two_l753_753173

-- Problem (1)
theorem inequality_one {a b : ℝ} (h1 : a ≥ b) (h2 : b > 0) : 2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b :=
sorry

-- Problem (2)
theorem inequality_two {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : (a ^ 2 / b + b ^ 2 / c + c ^ 2 / a) ≥ 1 :=
sorry

end inequality_one_inequality_two_l753_753173


namespace find_side_c_find_angle_B_l753_753736

noncomputable def triangleABC (A B c b : ℝ) : Prop :=
  A = π / 4 ∧ b = sqrt 6 ∧ B = π / 3 ∧ c = 1 + sqrt 3 ∧
  2 * (3 + sqrt 3) = b * c * sin A / 2 -- Area of the triangle

theorem find_side_c (a b : ℝ) : triangleABC π / 4 ?B (1 + sqrt 3) b → 
  b = sqrt 6 → 
  2 * (3 + sqrt 3) = b * (1 + sqrt 3) * sin (π / 4) →
  1 + sqrt 3 = (1 + sqrt 3) := 
by 
  sorry

theorem find_angle_B (A c b : ℝ) (B : ℝ) : A = π / 4 → b = sqrt 6 →
  c = 1 + sqrt 3 → 
  cos B = 1 / 2 → 
  B = π / 3 :=
by 
  sorry

end find_side_c_find_angle_B_l753_753736


namespace roots_equal_iff_m_values_l753_753636

theorem roots_equal_iff_m_values (m : ℝ) :
  (∀ x : ℝ, (x(x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = (x + 1) / (m + 1) → x^2 - x * (m + 1) - (m^2 + 5 * m + 4) = 0) →
  m = -1 ∨ m = -5 :=
sorry

end roots_equal_iff_m_values_l753_753636


namespace true_propositions_are_1_and_3_l753_753676

-- Define the given conditions as Lean definitions
def proposition_one : Prop :=
  ∀ x, cos (x - π / 4) * cos (x + π / 4) = 1 / 2 * cos (2 * x)

def proposition_two : Prop :=
  ∀ x, (x + 3) / (x - 1) = 4 / (x - 1) + 1

def proposition_three (a : ℝ) : Prop :=
  (a ≠ 0) ∧ (4 * a^2 + 4 * a = 0) → a = -1

def proposition_four : Prop :=
  ∃ (ABC : Type) [triangle ABC], AC ABC = sqrt 3 ∧ angle B ABC = 60 ∧ AB ABC = 1 →
  ∃! ABC, triangle ABC

-- Define the problem statement
theorem true_propositions_are_1_and_3 :
  (proposition_one ∧ proposition_three (-1)) ∧ ¬proposition_two ∧ ¬proposition_four :=
by
  split; sorry

end true_propositions_are_1_and_3_l753_753676


namespace find_d_given_n_eq_cda_div_a_minus_d_l753_753359

theorem find_d_given_n_eq_cda_div_a_minus_d (a c d n : ℝ) (h : n = c * d * a / (a - d)) :
  d = n * a / (c * d + n) := 
by
  sorry

end find_d_given_n_eq_cda_div_a_minus_d_l753_753359


namespace vectors_coplanar_l753_753167

/-- Vectors defined as 3-dimensional Euclidean space vectors. --/
def vector3 := (ℝ × ℝ × ℝ)

/-- Definitions for vectors a, b, c as given in the problem conditions. --/
def a : vector3 := (3, 1, -1)
def b : vector3 := (1, 0, -1)
def c : vector3 := (8, 3, -2)

/-- The scalar triple product of vectors a, b, c is the determinant of the matrix formed. --/
noncomputable def scalarTripleProduct (u v w : vector3) : ℝ :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  u1 * (v2 * w3 - v3 * w2) - u2 * (v1 * w3 - v3 * w1) + u3 * (v1 * w2 - v2 * w1)

/-- Statement to prove that vectors a, b, c are coplanar (i.e., their scalar triple product is zero). --/
theorem vectors_coplanar : scalarTripleProduct a b c = 0 :=
  by sorry

end vectors_coplanar_l753_753167


namespace muffins_per_person_l753_753744

-- Definitions based on conditions
def total_friends : ℕ := 4
def total_people : ℕ := 1 + total_friends
def total_muffins : ℕ := 20

-- Theorem statement for the proof
theorem muffins_per_person : total_muffins / total_people = 4 := by
  sorry

end muffins_per_person_l753_753744


namespace radian_measure_of_60_deg_l753_753539

-- Definitions based on conditions
def degree_to_radian (degree : ℝ) : ℝ := degree * (Real.pi / 180)

-- Lean statement for the proof problem
theorem radian_measure_of_60_deg : degree_to_radian 60 = (Real.pi / 3) := 
by
  sorry

end radian_measure_of_60_deg_l753_753539


namespace sum_of_first_n_terms_l753_753756

namespace ArithmeticGeometricSequence

-- Define the arithmetic sequence
def a_n (n : ℕ) (d : ℝ) : ℝ :=
  2 + (n - 1) * d

-- Geometric sequence condition for a_1, a_3, a_6
def geometric_condition (d : ℝ) : Prop :=
  (a_n 3 d) ^ 2 = (a_n 1 d) * (a_n 6 d)

theorem sum_of_first_n_terms (d : ℝ) (H : d ≠ 0) (Hgeo : geometric_condition d) :
  ∀ n : ℕ, n > 0 → (∑ i in finset.range n, a_n (i + 1) d) = (n ^ 2 / 4) + (7 * n / 4) :=
by
  sorry

end ArithmeticGeometricSequence

end sum_of_first_n_terms_l753_753756


namespace proof_f_n_plus_4_eq_neg2_l753_753063

noncomputable def f (x : ℝ) : ℝ :=
if x > 6 then -real.log (x + 1) / real.log 3 else real.exp (real.log 3 * (x - 6)) - 1

theorem proof_f_n_plus_4_eq_neg2 (n : ℝ) (h : f n = -8 / 9) : f (n + 4) = -2 :=
sorry

end proof_f_n_plus_4_eq_neg2_l753_753063


namespace crystal_bead_cost_l753_753181

theorem crystal_bead_cost
  (C : ℝ)  -- Cost of one set of crystal beads
  (cost_of_metal_beads : ℝ = 10)  -- Cost of one set of metal beads is $10
  (total_metal_beads_bought : ℝ = 2)  -- Nancy buys two sets of metal beads
  (total_cost : ℝ = 29)  -- Total amount spent is $29
  (H : C + (total_metal_beads_bought * cost_of_metal_beads) = total_cost) :
  C = 9 :=
by
  sorry

end crystal_bead_cost_l753_753181


namespace f_odd_function_f_bounds_on_interval_l753_753679

noncomputable def f : ℝ → ℝ :=
sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0
axiom f_one_half : f 1 = 1 / 2

theorem f_odd_function : ∀ x : ℝ, f x = -f (-x) :=
sorry

theorem f_bounds_on_interval : 
  (∀ x ∈ set.Icc (-2 : ℝ) 6, 
    f x ≥ -1 ∧ f x ≤ 3) ∧ 
    f (-2) = -1 ∧ 
    f (6) = 3 :=
sorry

end f_odd_function_f_bounds_on_interval_l753_753679


namespace equation_of_line_AB_l753_753209

noncomputable def center_of_circle : (ℝ × ℝ) := (-4, -1)

noncomputable def point_P : (ℝ × ℝ) := (2, 3)

noncomputable def slope_OP : ℝ :=
  let (x₁, y₁) := center_of_circle
  let (x₂, y₂) := point_P
  (y₂ - y₁) / (x₂ - x₁)

noncomputable def slope_AB : ℝ :=
  -1 / slope_OP

theorem equation_of_line_AB : (6 * x + 4 * y + 19 = 0) :=
  sorry

end equation_of_line_AB_l753_753209


namespace proof_equation_and_distance_l753_753734

-- Define the polar to rectangular coordinate conversion
def polar_to_rect (p θ : ℝ) : ℝ × ℝ := (p * Real.cos θ, p * Real.sin θ)

-- Define C1: p = 2 * cos θ
def C1 (p θ : ℝ) : Prop := p = 2 * Real.cos θ

-- Define the locus condition for Q: |OP| * |OQ| = 6
def locus_condition (|OP| |OQ| : ℝ) : Prop := |OP| * |OQ| = 6

-- Define the rectangular coordinate equation for C2
def C2 : Prop := ∀ x y : ℝ, (x = 3 → (∃ θ : ℝ, (polar_to_rect 6 (1 / (2 * Real.cos θ)) θ = (x, y))))

-- Define the line l: θ = π / 3
def line_l (θ : ℝ) : Prop := θ = Real.pi / 3

-- Define points A and B and their intersection conditions
def intersection_point_A : ℝ × ℝ := polar_to_rect (2 * Real.cos (Real.pi / 3)) (Real.pi / 3)
def intersection_point_B : ℝ × ℝ := (3, 3 * Real.sqrt 3)

-- Define the distance between points A and B
def distance_AB : ℝ := Real.sqrt ((3 - 1 / 2)^2 + (3 * Real.sqrt 3 - Real.sqrt 3 / 2)^2)

-- The Lean theorem to be proven, using the conditions above
theorem proof_equation_and_distance :
  C2 ∧ line_l (Real.pi / 3) ∧ Real.sqrt ((3 - intersection_point_A.fst)^2 + (3 * Real.sqrt 3 - intersection_point_A.snd)^2) = 5 :=
by
  sorry

end proof_equation_and_distance_l753_753734


namespace find_AH_l753_753386

-- Define the necessary geometric constructs and conditions
variables {A B C H M : Type} [inhabited A] [inhabited B] [inhabited C] [inhabited H] [inhabited M]

-- Let ∆ABC be an acute-angled triangle
axiom acute_angled_triangle : ∀ (A B C : Type), Type
def triangle_ABC := acute_angled_triangle A B C

-- Altitude BH
axiom altitude_BH : Type

-- Median AM
axiom median_AM : Type

-- Angle MCA is twice the angle MAC
axiom angle_MCA_twice_angle_MAC : Prop

-- BC = 10
axiom BC_length : ℝ
axiom BC_length_value : BC_length = 10

-- Prove that AH = 5
theorem find_AH (h : triangle_ABC ∧ altitude_BH ∧ median_AM ∧ angle_MCA_twice_angle_MAC ∧ BC_length_value) : AH = 5 := sorry

end find_AH_l753_753386


namespace ball_bounce_9th_time_total_distance_l753_753893

theorem ball_bounce_9th_time_total_distance (a n : ℕ) (h : a = 128) (k : n = 9) :
  let total_distance := 3 * a - (2 ^ (2 - n)) * a in
  total_distance = 383 := 
by
  sorry

end ball_bounce_9th_time_total_distance_l753_753893


namespace eliminate_denominators_l753_753870

theorem eliminate_denominators (x : ℝ) :
  (4 * (2 * x - 1) - 3 * (3 * x - 4) = 12) ↔ ((2 * x - 1) / 3 - (3 * x - 4) / 4 = 1) := 
by
  sorry

end eliminate_denominators_l753_753870


namespace find_a_value_l753_753649

theorem find_a_value (x : ℝ) (a : ℝ) (h0 : 0 < x) 
  (h1 : ∀ n : ℕ, x + n ^ n / x ^ n ≥ n + 1) :
  a = 2016 ^ 2016 :=
by
  have h2016 : x + 2016 ^ 2016 / x ^ 2016 ≥ 2017 :=
    h1 2016
  sorry

end find_a_value_l753_753649


namespace A_inter_N_eq_01_l753_753300

noncomputable def A : set ℝ := {x | x - 2 < 0}

noncomputable def N : set ℕ := {0, 1, 2, 3, ...}

theorem A_inter_N_eq_01 : A ∩ (N : set ℝ) = {0, 1} :=
  by
    sorry

end A_inter_N_eq_01_l753_753300


namespace hyperbola_standard_form_l753_753284

theorem hyperbola_standard_form :
  ∃ λ : ℝ, λ = 3 ∧ (∀ (x y : ℝ), (x, y) = (2, 2) → x^2 - y^2 / 4 = λ) →
  (∀ (x y : ℝ), (x^2 / 3 - y^2 / 12 = 1)) :=
by
  sorry

end hyperbola_standard_form_l753_753284


namespace max_sin_sq_l753_753404

variables {A B C : ℝ}
variables {a b c : ℝ}
variable (h : ∀ (a b c : ℝ), (a ^ 2 + b ^ 2 - c ^ 2 = 2 * a * b * cos C) ∧ (a ^ 2 + c ^ 2 - b ^ 2 = 2 * a * c * cos B))
variable (h₁ : ∀ (A : ℝ), sin (π - A) = sin A)
variable (h₂ : ∀ (A B C : ℝ), 2 * sin A * cos B - sin C * cos B = sin B * cos C)
variable (h₃ : A + B + C = π)
variable (h₄ : b * cos C / (c * cos B) = sin B * cos C / (sin C * cos B))

theorem max_sin_sq (A B C a b c : ℝ)
  (h₅ : ∀ (A B C a b c : ℝ), (2 * sin A - sin C) / sin C = (a ^ 2 + b ^ 2 - c ^ 2) / (a ^ 2 + c ^ 2 - b ^ 2))
  (triangle_condition : a ^ 2 + b ^ 2 - c ^ 2 = 2 * a * b * cos C)
  (angle_sum : A + B + C = π)
  (cosine_thm : cos (π - A) = -cos A)
  (cos_B_eq_half : cos B = 1 / 2) :
  sin^2 A + sin^2 C ≤ 3 / 2 :=
sorry

end max_sin_sq_l753_753404


namespace minimum_sum_of_nine_consecutive_integers_l753_753783

-- We will define the consecutive sequence and the conditions as described.
structure ConsecutiveIntegers (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ) : Prop :=
(seq : a1 + 1 = a2 ∧ a2 + 1 = a3 ∧ a3 + 1 = a4 ∧ a4 + 1 = a5 ∧ a5 + 1 = a6 ∧ a6 + 1 = a7 ∧ a7 + 1 = a8 ∧ a8 + 1 = a9)
(sq_cond : ∃ k : ℕ, (a1 + a3 + a5 + a7 + a9) = k * k)
(cube_cond : ∃ l : ℕ, (a2 + a4 + a6 + a8) = l * l * l)

theorem minimum_sum_of_nine_consecutive_integers :
  ∃ a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ,
  ConsecutiveIntegers a1 a2 a3 a4 a5 a6 a7 a8 a9 ∧ (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 = 18000) :=
  sorry

end minimum_sum_of_nine_consecutive_integers_l753_753783


namespace triangle_midpoint_equivalence_l753_753450

theorem triangle_midpoint_equivalence (A B C M D E : Point) (h_eq_triangle : equilateral_triangle A B C)
  (h_midpoint_M : midpoint M A B) (h_D_on_AC : D ∈ segment A C) (h_E_on_BC : E ∈ segment B C)
  (h_angle_DME : ∠DME = 60) : 
  distance A D + distance B E = distance D E + 1/2 * distance A B :=
sorry

end triangle_midpoint_equivalence_l753_753450


namespace arithmetic_sequence_term_eq_l753_753036

theorem arithmetic_sequence_term_eq  {
  (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 2) : 
    a 50 = 99 :=
begin
  sorry
end

end arithmetic_sequence_term_eq_l753_753036


namespace remainder_of_large_numbers_division_mod_23_l753_753540

theorem remainder_of_large_numbers_division_mod_23 :
  let A := 6_598_574_241_545_098_875_458_255_622_898_854_689_448_911_257_658_451_215_825_362_549_889
  let B := 3_721_858_987_156_557_895_464_215_545_212_524_189_541_456_658_712_589_687_354_871_258
  let D := 23
  (A % D - B % D) % D = 8 :=
by
  sorry

end remainder_of_large_numbers_division_mod_23_l753_753540


namespace solveProblems_l753_753516

noncomputable def problem1 (a b : ℤ) : Prop :=
2 * a + b = 35 ∧ a + 3 * b = 30 → a = 15 ∧ b = 5

noncomputable def problem2 (x : ℤ) : Prop :=
(∃ n, n = 5 ∧ 955 ≤ 15 * x + 5 * (120 - x) ∧ 15 * x + 5 * (120 - x) ≤ 1000)

noncomputable def problem3 (x : ℤ) (W : ℤ) : Prop :=
(∃ m, m = 960 ∧ W = 10 * x + 600 ∧ W_min = m)

-- statement of the proof problem
theorem solveProblems :
    (problem1 a b) ∧
    (problem2 x) ∧
    (problem3 x W) :=
sorry

end solveProblems_l753_753516


namespace product_of_roots_l753_753703

theorem product_of_roots (x : ℝ) :
  (x+3) * (x-4) = 22 →
  let a := 1 in
  let b := -1 in
  let c := -34 in
  a * x^2 + b * x + c = 0 → 
  (c / a) = -34 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end product_of_roots_l753_753703


namespace max_alpha_proof_l753_753980

noncomputable def max_constant_alpha (a b : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2 * a) : ℚ := 
1 / (2 * (a^2) - 2 * a * b + b^2)

theorem max_alpha_proof (a b : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2 * a)
    (N : ℕ) (hN : 0 < N)
    (H : ∀ (r s : ℕ), r ≤ N → s ≤ N → 
        ∃ (i j : ℕ), i < r ∧ j < s ∧ (i,j) are coordinates of a red unit square 
        in any a x b or b x a rectangle)
    : ∀ (N : ℕ), ∃ (c : ℚ), c = 1 / (2 * (a^2) - 2 * a * b + b^2) ∧
                   (N * N standard rectangle contains at least (c * N^2) red unit squares) :=
begin
  sorry -- Proof goes here
end

end max_alpha_proof_l753_753980


namespace find_number_l753_753373

theorem find_number (x : ℤ) (h : 5 * (x - 12) = 40) : x = 20 := 
by
  sorry

end find_number_l753_753373


namespace income_of_A_l753_753124

theorem income_of_A (x y : ℝ) (hx₁ : 5 * x - 3 * y = 1600) (hx₂ : 4 * x - 2 * y = 1600) : 
  5 * x = 4000 :=
by
  sorry

end income_of_A_l753_753124


namespace original_price_of_boots_l753_753942

theorem original_price_of_boots (P : ℝ) (h : P * 0.80 = 72) : P = 90 :=
by 
  sorry

end original_price_of_boots_l753_753942


namespace triangle_problem_l753_753933

noncomputable theory

-- Definitions and assumptions from problem conditions
variables (A B C M P D E : Type) [field E]
variables [metric_space E] [normed_group E] [normed_space ℝ E]
variables (a b c : E) -- Points A, B, C in the Euclidean space E
variables (m p d e : E) -- Points M, P, D, E in the Euclidean space E

-- Midpoint definition
def is_midpoint (m b c : E) : Prop := dist m b = dist m c

-- Angle bisector condition
def bisects_angle (a p b c : E) : Prop := sorry

-- Point on circumcircle condition
def on_circumcircle (e : E) (a b c : E) : Prop  := sorry

-- Given problem conditions as definitions
def problem_conditions : Prop :=
  is_midpoint m b c ∧
  bisects_angle a p b c ∧
  on_circumcircle p a b c ∧
  on_circumcircle p a c b ∧
  dist d e = dist m p

-- The translated problem statement as a Lean theorem
theorem triangle_problem 
  (h1 : is_midpoint M B C) 
  (h2 : bisects_angle A P B C)
  (h3 : on_circumcircle P A B C) 
  (h4 : on_circumcircle P A C B)
  (h5 : dist D E = dist M P) 
  : 2 * dist B P = dist B C := 
begin
  sorry
end

end triangle_problem_l753_753933


namespace solve_system_l753_753807

open Real

def condition1 (x y : ℝ) : Prop := (2:ℝ)^(log10 x) + (3:ℝ)^(log10 y) = 5
def condition2 (x y : ℝ) : Prop := (2:ℝ)^(log10 x) * (3:ℝ)^(log10 y) = 4

theorem solve_system (x y : ℝ) :
  (condition1 x y) ∧ (condition2 x y) →
  ((x = 100 ∧ y = 1) ∨ (x = 1 ∧ y = 10^(log10 4 / log10 3))) :=
sorry

end solve_system_l753_753807


namespace john_growth_l753_753746

theorem john_growth 
  (InitialHeight : ℤ)
  (GrowthRate : ℤ)
  (FinalHeight : ℤ)
  (h1 : InitialHeight = 66)
  (h2 : GrowthRate = 2)
  (h3 : FinalHeight = 72) :
  (FinalHeight - InitialHeight) / GrowthRate = 3 :=
by
  sorry

end john_growth_l753_753746


namespace find_m_l753_753666

-- Definitions of the given problem conditions
def circles_intersect_at_two_points (C1 C2 : Circle) : Prop :=
  -- Definition that two circles intersect at exactly two points
  sorry

def intersection_point (C1 C2 : Circle) : Point :=
  -- Definition to give the intersection point of two circles
  ⟨9, 6⟩

def product_of_radii (C1 C2 : Circle) : ℝ :=
  C1.radius * C2.radius

def tangents_to_circles (C1 C2 : Circle) (m : ℝ) : Prop :=
  -- Definition that x-axis and y = mx are tangent to both circles C1 and C2
  sorry

-- The main theorem statement
theorem find_m (C1 C2 : Circle) (m : ℝ) 
  (h1 : circles_intersect_at_two_points C1 C2)
  (h2 : intersection_point C1 C2 = ⟨9, 6⟩)
  (h3 : product_of_radii C1 C2 = 68)
  (h4 : tangents_to_circles C1 C2 m) :
  m = 12 * Real.sqrt 221 / 49 :=
sorry

end find_m_l753_753666


namespace parallelogram_area_l753_753755

theorem parallelogram_area {r s : ℝ^3} (hr : ∥r∥ = 1) (hs : ∥s∥ = 1) (angle_rs : real.angle_between r s = real.pi / 4) : 
  area_of_parallelogram ((r + 3 • s),(3 • r + s)) = 9 * real.sqrt 2 / 4 :=
sorry

end parallelogram_area_l753_753755


namespace ratio_of_areas_l753_753795

theorem ratio_of_areas (perimeter_A perimeter_B perimeter_C : ℕ) 
  (hA : perimeter_A = 16) 
  (hB : perimeter_B = 32) 
  (hC : perimeter_C = 20) : 
  (let side_length_A := perimeter_A / 4,
       side_length_B := perimeter_B / 4,
       side_length_C := perimeter_C / 4,
       area_B := side_length_B * side_length_B,
       area_C := side_length_C * side_length_C in
   (area_B * 25 = area_C * 64)) :=
by {
  let side_length_A := perimeter_A / 4,
  let side_length_B := perimeter_B / 4,
  let side_length_C := perimeter_C / 4,
  let area_B := side_length_B * side_length_B,
  let area_C := side_length_C * side_length_C,
  show (area_B * 25 = area_C * 64),
  sorry
}

end ratio_of_areas_l753_753795


namespace correct_answer_l753_753596

variables {m n : Type*} {α β γ : Type*}
variable [Line m] [Line n]
variable [Plane α] [Plane β] [Plane γ]

def proposition1 : Prop := ∀ α β γ (m n : Line), (parallel n α ∧ perp m α) → perp m n
def proposition2 : Prop := ∀ α β γ, (perp α γ ∧ perp β γ) → parallel α β
def proposition3 : Prop := ∀ α (m n : Line), (parallel m α ∧ parallel n α) → parallel m n
def proposition4 : Prop := ∀ α β γ (m : Line), (parallel α β ∧ parallel β γ ∧ perp m α) → perp m γ

def correct_propositions : Prop := proposition1 ∧ proposition4

theorem correct_answer : correct_propositions :=
sorry

end correct_answer_l753_753596


namespace custom_operations_identity_l753_753614

-- Definitions of the custom operations
def custom_star (A B : ℝ) : ℝ := (A - B) / 3
def custom_square (A B : ℝ) : ℝ := (A + B) * 3

-- The main statement to be proved
theorem custom_operations_identity : custom_square (custom_star 39 12) 3 = 36 := 
by 
  sorry

end custom_operations_identity_l753_753614


namespace non_degenerate_ellipse_l753_753619

theorem non_degenerate_ellipse (x y k : ℝ) : (∃ k, (2 * x^2 + 9 * y^2 - 12 * x - 27 * y = k) → k > -135 / 4) := sorry

end non_degenerate_ellipse_l753_753619


namespace fill_pool_with_B_only_l753_753587

theorem fill_pool_with_B_only
    (time_AB : ℝ)
    (R_AB : time_AB = 30)
    (time_A_B_then_B : ℝ)
    (R_A_B_then_B : (10 / 30 + (time_A_B_then_B - 10) / time_A_B_then_B) = 1)
    (only_B_time : ℝ)
    (R_B : only_B_time = 60) :
    only_B_time = 60 :=
by
    sorry

end fill_pool_with_B_only_l753_753587


namespace andrew_paid_total_l753_753876

-- Define the quantities and rates
def quantity_grapes : ℕ := 14
def rate_grapes : ℕ := 54
def quantity_mangoes : ℕ := 10
def rate_mangoes : ℕ := 62

-- Define the cost calculations
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes
def total_cost : ℕ := cost_grapes + cost_mangoes

-- Prove the total amount paid is as expected
theorem andrew_paid_total : total_cost = 1376 := by
  sorry 

end andrew_paid_total_l753_753876


namespace cosine_alpha_value_l753_753705

theorem cosine_alpha_value 
  (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : sin (α - π / 3) = 1 / 3) : 
  cos α = (2 * real.sqrt 2 - real.sqrt 3) / 6 :=
by 
  sorry 

end cosine_alpha_value_l753_753705
