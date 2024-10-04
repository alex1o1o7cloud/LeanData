import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Quadratic.Discriminants
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecificLimits
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.GCD
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Pnat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace twelve_times_y_plus_three_half_quarter_l108_108645

theorem twelve_times_y_plus_three_half_quarter (y : ℝ) : 
  (1 / 2) * (1 / 4) * (12 * y + 3) = (3 * y) / 2 + 3 / 8 :=
by sorry

end twelve_times_y_plus_three_half_quarter_l108_108645


namespace find_angle_BIE_l108_108280

-- Definitions and conditions
variables (A B C I D E F : Type*)
variables (angle : A → A → A → ℝ)
variables (is_triangle : Triangle A B C)
variables (angle_bisectors : Incenter I (A B C) (D E F))
variables (angle_ACB : angle A C B = 50)
variables (angle_BAC : angle B A C = 45)

-- Statement to prove
theorem find_angle_BIE :
  angle B I E = 65 :=
sorry

end find_angle_BIE_l108_108280


namespace problem_1_problem_2_problem_3_l108_108956

noncomputable def f (x : ℝ) (k : ℝ) := log 4 (4^x + 1) + k * x 

-- Problem 1: Proving k = -1/2 if f is even
theorem problem_1 (k : ℝ) : 
  (∀ x : ℝ, f (-x) k = f x k) → k = -1/2 := 
sorry

noncomputable def h (x : ℝ) (m : ℝ) :=
  4^(f x (-1/2) + x / 2) + m * 2^x - 1

-- Problem 2: Proving m = -1 if h(x) has a minimum value of 0 on [0, log(2) 3]
theorem problem_2 (m : ℝ) : 
  (∃ x ∈ set.Icc 0 (log 2 3), h x m = 0) → m = -1 := 
sorry

-- Problem 3: Proving the range of a where graphs have no intersections
noncomputable def f_simple (x : ℝ) := log 4 (4^x + 1)

theorem problem_3 (a : ℝ) : 
  (∀ x : ℝ, f_simple x ≠ (1 / 2) * x + a) → a ≤ 0 := 
sorry

end problem_1_problem_2_problem_3_l108_108956


namespace total_days_to_complete_work_l108_108464

-- Definitions based on the conditions
def work_done_per_day (men : ℕ) (days : ℕ) := 1 / (men * days)
def initial_work_done (men : ℕ) (days : ℕ) := men * work_done_per_day men days
def remaining_work (initial_work_done : ℚ) := 1 - initial_work_done
def time_to_complete (men : ℕ) (remaining_work : ℚ) (per_day_work : ℚ) := remaining_work / (men * per_day_work)

-- Lean 4 statement for the problem
theorem total_days_to_complete_work :
  let initial_men := 3 in
  let total_days := 6 in
  let initial_days := 2 in
  let total_work_done := (initial_men : ℚ) * work_done_per_day initial_men total_days * (initial_days : ℚ) in
  let remaining := remaining_work total_work_done in
  let additional_men := 3 in
  let all_men := initial_men + additional_men in
  let days_after_join := time_to_complete all_men remaining (work_done_per_day 1 total_days) in
  initial_days + days_after_join = 4 :=
by
  sorry

end total_days_to_complete_work_l108_108464


namespace arithmetic_sequence_y_l108_108260

theorem arithmetic_sequence_y : 
  ∀ (x y z : ℝ), (23 : ℝ), x, y, z, (47 : ℝ) → 
  (y = (23 + 47) / 2) → y = 35 :=
by
  intro x y z h1
  intro h2
  simp at *
  sorry

end arithmetic_sequence_y_l108_108260


namespace equilateral_triangles_count_l108_108137

theorem equilateral_triangles_count (lattice : Set Point)
  (uniform_distance : ∀ p1 p2 ∈ lattice, ∃ d : ℝ, d = 1) :
  ∃ count : ℕ, count = 20 :=
by sorry

end equilateral_triangles_count_l108_108137


namespace fourth_boy_total_payment_l108_108571

-- Definitions
def boat_cost : ℕ := 80
def equipment_cost : ℕ := 20

variables {a b c d : ℕ}

-- Conditions
axiom condition1 : a = (b + c + d) / 2
axiom condition2 : b = (a + c + d) / 4
axiom condition3 : c = (a + b + d) / 3
axiom condition4 : a + b + c + d = boat_cost

-- Hypothesis
def total_paid_by_fourth_boy : ℕ := d + equipment_cost / 4

-- The theorem we want to prove
theorem fourth_boy_total_payment : total_paid_by_fourth_boy = 23 :=
by sorry

end fourth_boy_total_payment_l108_108571


namespace distance_sosnovka_petrovka_l108_108031

theorem distance_sosnovka_petrovka 
  (AP : ℕ) (BS : ℕ) (travelled : ℕ)
  (hAP : AP = 70) 
  (hTravelled : travelled = 20) 
  (hBS : BS = 130) : 
  (BS + (AP - travelled) = 180) :=
by 
  rw [hAP, hTravelled, hBS]
  rw [←nat.sub_sub_self]
  {a := 0}
  (le_refl 20)
  rw nat.add_sub_assoc
  (le_refl 130)
  rw [nat.sub_self, nat.add_zero]
  rfl

end distance_sosnovka_petrovka_l108_108031


namespace leftmost_square_side_length_l108_108758

open Real

/-- Given the side lengths of three squares, 
    where the middle square's side length is 17 cm longer than the leftmost square,
    the rightmost square's side length is 6 cm shorter than the middle square,
    and the sum of the side lengths of all three squares is 52 cm,
    prove that the side length of the leftmost square is 8 cm. -/
theorem leftmost_square_side_length
  (x : ℝ)
  (h1 : ∀ m : ℝ, m = x + 17)
  (h2 : ∀ r : ℝ, r = x + 11)
  (h3 : x + (x + 17) + (x + 11) = 52) :
  x = 8 := by
  sorry

end leftmost_square_side_length_l108_108758


namespace new_shape_perimeter_l108_108491

-- Definitions based on conditions
def square_side : ℕ := 64 / 4
def is_tri_isosceles (a b c : ℕ) : Prop := a = b

-- Definition of given problem setup and perimeter calculation
theorem new_shape_perimeter
  (side : ℕ)
  (tri_side1 tri_side2 base : ℕ)
  (h_square_side : side = 64 / 4)
  (h_tri1 : tri_side1 = side)
  (h_tri2 : tri_side2 = side)
  (h_base : base = side) :
  (side * 5) = 80 :=
by
  sorry

end new_shape_perimeter_l108_108491


namespace statement_A_statement_B_statement_C_statement_D_l108_108957

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x
def g (x : ℝ) : ℝ := f x * f' x

theorem statement_A (x1 x2 : ℝ) (h : |f x1 - f x2| = 2 * Real.sqrt 2) : |x1 - x2| = Real.pi := sorry

theorem statement_B : ¬ ( ∀ x : ℝ, f (x + Real.pi / 4) = f (-x - Real.pi / 4)) := sorry

theorem statement_C (ω : ℝ) (h : ∀ x ∈ Icc 0 Real.pi, f (ω * x) = 0 → (x = 0 ∨ x = Real.pi / 4 ∨ x = Real.pi / 2 ∨ x = 3 * Real.pi / 4)) : ω ∈ Ico (15 / 4 : ℝ) (19 / 4) := sorry

theorem statement_D : ∀ x ∈ Ioo 0 (Real.pi / 4), g x ∈ Ioo 0 1 := sorry

end statement_A_statement_B_statement_C_statement_D_l108_108957


namespace arithmetic_sequence_middle_term_l108_108266

theorem arithmetic_sequence_middle_term (x y z : ℝ) (h : list.nth_le [23, x, y, z, 47] 2 sorry = y) :
  y = 35 :=
sorry

end arithmetic_sequence_middle_term_l108_108266


namespace favor_increase_condition_holds_l108_108245

-- Define variables for the initial and later votes
variables (x y m x' y' : ℕ)

-- Hypotheses based on the problem's conditions
def conditions := 
  (500 : ℕ = x + y) ∧
  (y > x) ∧
  (y - x = m) ∧
  (x' - y' = 3 * m) ∧
  (x' + y' = 500) ∧
  (x' = 13 * y / 12)

-- The proof statement
theorem favor_increase_condition_holds : 
  conditions x y m x' y' →
  (x' - x = 40) :=
sorry

end favor_increase_condition_holds_l108_108245


namespace tangent_line_eq_fx_positive_max_value_l108_108962

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x

-- (I) When a = 2, find the equation of the tangent line to the curve f(x) at the point (0,f(0))
theorem tangent_line_eq (a x : ℝ) (h : a = 2) :
  let f_x := f x a
  let f' := λ x, Real.exp x - a
  let tangent_slope := f' 0
  let f_0 := f 0 a in
  tangent_slope * x + f_0 = -x + 1 :=
by
  sorry

-- (II) Under the condition of (I), prove that f(x) > 0
theorem fx_positive (x : ℝ) : f x 2 > 0 :=
by
  sorry

-- (III) When a > 1, find the maximum value of the function f(x) on [0,a]
theorem max_value (a x : ℝ) (h : 1 < a) :
  let f_a := f a a in
  ∀ x ∈ Set.Icc 0 a, f x a ≤ f_a :=
by
  sorry

end tangent_line_eq_fx_positive_max_value_l108_108962


namespace points_p_on_ray_l108_108925

open Set Real

noncomputable def midpoint (x y : ℝ) : ℝ := (x + y) / 2

-- Assuming basic topology and geometry constructs for circles, lines, and points.
def is_tangent {C : ℝ → Prop} (l : ℝ → ℝ) (F : ℝ) : Prop := 
  ∀ x, C x → ∃ y, l y = 0 ∧ (l' F C x = 0)

def is_incircle {C : ℝ → Prop} (P Q R : ℝ) : Prop := 
  ∀ x, C x → ∃ t₁ t₂ t₃, (tangent P x ∧ tangent Q x ∧ tangent R x) 

-- The theorem statement: every point P found by reflected M lies on a specific ray.
theorem points_p_on_ray
  (C : ℝ → Prop)
  (l : ℝ → ℝ)
  (F M : ℝ) 
  (h_tangent : is_tangent C l F)
  (h_point : l M = 0) :
  ∃ P : ℝ,
    ∀ Q R : ℝ, 
      (Q ≠ R ∧ 
      l Q = 0 ∧ 
      l R = 0 ∧ 
      midpoint Q R = M ∧ 
      is_incircle C P Q R) ->
  (* Conclusion: P must lie on ray extending from F beyond G *)
  ray F P :=
sorry

end points_p_on_ray_l108_108925


namespace symmetry_and_monotonicity_l108_108958

noncomputable def function_f (x : ℝ) : ℝ :=
if x >= 1 then log x else log (2 - x)

theorem symmetry_and_monotonicity :
  function_f (2 - 1 / 2) = function_f (1 / 2) ∧
  function_f (2 - 1 / 3) = function_f (1 / 3) ∧
  function_f (2 - 2) = function_f (2) ∧
  function_f (1 / 2) < function_f (1 / 3) ∧
  function_f (1 / 3) < function_f (2) := 
by
    sorry

end symmetry_and_monotonicity_l108_108958


namespace coordinates_of_E_l108_108628

-- Define the parabola C
def parabola (x : ℝ) : ℝ := x^2 / 2

-- Define the points mentioned in the problem
def F := (0, 1 / 2)
def M (x1 : ℝ) := (x1, parabola x1)
def N (x1 : ℝ) := (0, parabola x1)

-- Define the necessary conditions for part 1
axiom is_isosceles_triangle (x1 : ℝ) :
  let M := M x1,
  let N := N x1,
  let MF := dist M F in
  let NF := dist N F in
  MF = NF

-- Define A, B, and D for part 2
def A := (0, parabola 0)
def B := (2, parabola 2)
def D := (1, 1)

-- Define point E and its condition
def E := (-1, parabola (-1))

-- Define the tangent line condition
axiom tangent_condition : 
  ∃ (E : ℝ × ℝ), 
  E = (-1, parabola (-1)) ∧ 
  ∀ (x0 : ℝ), x0 ≠ 0 → x0 ≠ 2 → 
  let k_ME := (parabola x0 - E.2) / (x0 - E.1) in 
  k_ME * x0 = -1

-- Assert the existence and correctness of point E
theorem coordinates_of_E :  E = (-1, 1/2) :=
sorry

end coordinates_of_E_l108_108628


namespace tank_capacity_l108_108805

theorem tank_capacity:
  ∃ (C : ℝ),
  (let leak_rate := C / 6
     inlet_rate := 150
     net_empty_rate := C / 8 in
   inlet_rate - leak_rate = net_empty_rate) → C = 3600 / 7 := 
by
  sorry

end tank_capacity_l108_108805


namespace infinitely_many_composite_powers_l108_108341

   theorem infinitely_many_composite_powers (a b c d : ℕ) : ∃ᶠ n in at_top, (¬ nat.prime (a^n + b^n + c^n + d^n)) :=
   sorry
   
end infinitely_many_composite_powers_l108_108341


namespace box_weight_limit_l108_108633

/-- The weight limit of the box in pounds -/
theorem box_weight_limit (cookie_weight_oz : ℕ) (num_cookies : ℕ) (oz_to_lb : ℕ) 
  (h1 : cookie_weight_oz = 2) (h2 : num_cookies = 320) (h3 : oz_to_lb = 16) : 
  (cookie_weight_oz * num_cookies) / oz_to_lb = 40 := by 
sorrr

end box_weight_limit_l108_108633


namespace midpoint_plane_divides_tetrahedron_l108_108946

theorem midpoint_plane_divides_tetrahedron
  (P A B C : Point)
  (M : Point) (N : Point)
  (hM : M = (A + B) / 2)
  (hN : N = (P + C) / 2) : 
  ∃ plane : Plane, (plane.contains M ∧ plane.contains N) ∧ (∀ V : Vol, divides_into_two_equal_volumes P A B C plane) :=
sorry

end midpoint_plane_divides_tetrahedron_l108_108946


namespace equilateral_of_incircle_condition_l108_108669

theorem equilateral_of_incircle_condition 
  (ABC : Triangle)
  (h_inscribed : ABC ∈ UnitCircle)
  (K : Point)
  (h_incenter : incenter ABC K)
  (h_condition : KA * KB * KC = 1) :
  equilateral ABC := 
sorry

end equilateral_of_incircle_condition_l108_108669


namespace factorization_example_l108_108097

open Function

theorem factorization_example (a b : ℤ) :
  (a - 1) * (b - 1) = ab - a - b + 1 :=
by
  sorry

end factorization_example_l108_108097


namespace simplify_expr_eq_l108_108955

-- Define the given expression and the expected simplified form.
def given_expr (b c : ℝ) : ℝ := (b⁻¹ * c⁻¹) / (b⁻² - c⁻²)
def expected_expr (b c : ℝ) : ℝ := b * c / (c^2 - b^2)

-- The theorem to be proven
theorem simplify_expr_eq (b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) (hbc : b ≠ c) : 
  given_expr b c = expected_expr b c := 
by 
  -- Since this is only a statement, we skip the proof.
  sorry

end simplify_expr_eq_l108_108955


namespace compare_powers_and_logarithm_l108_108395

theorem compare_powers_and_logarithm : 6 ^ 0.7 > 0.7 ^ 6 ∧ 0.7 ^ 6 > log 0.7 6 := 
by 
  sorry

end compare_powers_and_logarithm_l108_108395


namespace semicircle_radius_l108_108488

theorem semicircle_radius (a b : ℕ) (h : a = 12) (h' : b = 16) :
  ∃ r : ℕ, ∃ c : ℕ, (c^2 = a^2 + b^2) ∧ (r = (a * b) / (a + b + c)) ∧ (r = 4) :=
by
  -- assumption on sides a and b
  have ha : a = 12 := h,
  have hb : b = 16 := h',
  -- hypotenuse calculation using Pythagorean theorem
  let c := Nat.sqrt ((12^2) + (16^2)),
  have hyp_eq : c^2 = (12^2) + (16^2) := by
    rw [c, Nat.sqrt_eq_r_iff].2,
    norm_num,
    refl,
  -- radius of the semicircle (inradius of a right triangle)
  let area := (12 * 16) / 2,
  let perimeter := 12 + 16 + c,
  let r := area / perimeter,
  -- proving r = 4
  have r_eq : r = 4 := by
    rw [area, perimeter],
    norm_num,
  exact ⟨4, c, hyp_eq, r_eq, rfl⟩

end semicircle_radius_l108_108488


namespace vasechkin_result_l108_108333

theorem vasechkin_result (x : ℕ) (h : (x / 2 * 7) - 1001 = 7) : (x / 8) ^ 2 - 1001 = 295 :=
by
  sorry

end vasechkin_result_l108_108333


namespace exists_rational_non_integer_a_not_exists_rational_non_integer_b_l108_108536

-- Define rational non-integer numbers
def is_rational_non_integer (x : ℚ) : Prop := ¬(∃ (z : ℤ), x = z)

-- (a) Proof for existance of rational non-integer numbers y and x such that 19x + 8y, 8x + 3y are integers
theorem exists_rational_non_integer_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ a b : ℤ, 19 * x + 8 * y = a ∧ 8 * x + 3 * y = b) :=
sorry

-- (b) Proof for non-existance of rational non-integer numbers y and x such that 19x² + 8y², 8x² + 3y² are integers
theorem not_exists_rational_non_integer_b :
  ¬ ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ m n : ℤ, 19 * x^2 + 8 * y^2 = m ∧ 8 * x^2 + 3 * y^2 = n) :=
sorry

end exists_rational_non_integer_a_not_exists_rational_non_integer_b_l108_108536


namespace geometric_series_sum_l108_108139

theorem geometric_series_sum :
  let a := 1 / 4
  let r := - (1 / 4)
  ∃ S : ℚ, S = (a * (1 - r^6)) / (1 - r) ∧ S = 4095 / 81920 :=
by
  let a := 1 / 4
  let r := - (1 / 4)
  exists (a * (1 - r^6)) / (1 - r)
  sorry

end geometric_series_sum_l108_108139


namespace max_value_of_determinant_l108_108143

noncomputable def determinant_of_matrix (θ : ℝ) : ℝ :=
  Matrix.det ![
    ![1, 1, 1],
    ![1, 1 + Real.sin (2 * θ), 1],
    ![1, 1, 1 + Real.cos (2 * θ)]
  ]

theorem max_value_of_determinant : 
  ∃ θ : ℝ, (∀ θ : ℝ, determinant_of_matrix θ ≤ (1 / 2)) ∧ determinant_of_matrix (θ_at_maximum) = (1 / 2) :=
sorry

end max_value_of_determinant_l108_108143


namespace directrix_of_parabola_l108_108557

-- Definition of the given parabola
def parabola (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / 16

-- The mathematical statement to prove
theorem directrix_of_parabola : ∀ x : ℝ, parabola x = y ↔ y = -5 / 4 -> sorry :=
by
  sorry

end directrix_of_parabola_l108_108557


namespace greatest_fleet_exists_l108_108879

def is_fleet (grid_size : ℕ × ℕ) (ships : list (list (ℕ × ℕ))) : Prop :=
  ∀ s1 ∈ ships, ∀ s2 ∈ ships, s1 ≠ s2 → ∀ (x1, y1) ∈ s1, ∀ (x2, y2) ∈ s2, (x1, y1) ≠ (x2, y2) ∧ (x1 ≠ x2 ∨ y1 ≠ y2)

def partition_of_n (n : ℕ) : list (list ℕ) :=
  -- A function that generates all partitions of n (not provided here)
  sorry

theorem greatest_fleet_exists :
  ∃ n : ℕ, n = 25 ∧ ∀ p ∈ partition_of_n 25, 
    ∃ fleet : list (list (ℕ × ℕ)), 
      length fleet = length p ∧
      is_fleet (10, 10) fleet ∧
      (∀ ship ∈ fleet, ship.length ∈ p) :=
  sorry

end greatest_fleet_exists_l108_108879


namespace diamond_computation_l108_108569

def diamond (x y : ℝ) : ℝ := (x^2 + y^2) / (x^2 - y^2)

theorem diamond_computation : diamond (diamond 2 3) 4 = -569 / 231 := by
  sorry

end diamond_computation_l108_108569


namespace set_families_inequality_l108_108309

open Set

section Combinatorics

variable {α : Type*} [Fintype α]

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem set_families_inequality
  (A B: Finset (Finset α)) 
  (m p q : ℕ)
  (hA : ∀ i ∈ A, i.card = p)
  (hB : ∀ i ∈ B, i.card = q)
  (hdisjoint : ∀ i j ∈ A ∪ B, i ∩ j = ∅ ↔ i = j) :
  A.card ≤ binomial p q :=
by { sorry }

end Combinatorics

end set_families_inequality_l108_108309


namespace find_point_C_l108_108668

theorem find_point_C :
  ∃ C : ℝ × ℝ,
    let A := (3, 2 : ℝ), B := (-1, 5 : ℝ) in
    C.2 = 3 * C.1 + 3 ∧ 
    abs ((3 * C.1 - 1) / 1) = 4 ∧ 
    ((let AB := real.sqrt ((3 - (-1)) ^ 2 + (2 - 5) ^ 2) in 1 / 2 * AB * abs ((3 * C.1 - 1) / 1) = 10)) ∧  
    (C = (-1, 0) ∨ C = (5/3, 14)) :=
by
  -- Proof would go here
  sorry

end find_point_C_l108_108668


namespace equal_cubes_l108_108849

theorem equal_cubes (a : ℤ) : -(a ^ 3) = (-a) ^ 3 :=
by
  sorry

end equal_cubes_l108_108849


namespace triangle_area_is_correct_l108_108843

noncomputable def area_of_isosceles_triangle_with_perimeter_11
  (a b : ℕ) 
  (h1 : 2 * a + b = 11)
  (h2 : a = 3)
  (h3 : b = 5) : ℝ :=
  let h := real.sqrt 2.75 in
  (5 * h) / 2

theorem triangle_area_is_correct
  (a b : ℕ)
  (h1 : 2 * a + b = 11)
  (h2 : a = 3)
  (h3 : b = 5) :
  area_of_isosceles_triangle_with_perimeter_11 a b h1 h2 h3 = 5 * real.sqrt(2.75) / 2 :=
by
  sorry

end triangle_area_is_correct_l108_108843


namespace trees_not_pine_l108_108410

theorem trees_not_pine 
  (total_trees : ℕ)
  (percentage_pine : ℚ) 
  (H_total : total_trees = 350)
  (H_percentage : percentage_pine = 70 / 100) :
  total_trees - (total_trees * percentage_pine).toNat = 105 := 
by
  -- place a placeholder sorry for the proof
  sorry

end trees_not_pine_l108_108410


namespace system_of_linear_equations_solution_l108_108723

theorem system_of_linear_equations_solution :
  ∃ (x_1 x_2 x_3 x_4: ℝ), (
    x_1 - 2 * x_2 + x_4 = -3 ∧
    3 * x_1 - x_2 - 2 * x_3 = 1 ∧
    2 * x_1 + x_2 - 2 * x_3 - x_4 = 4 ∧
    x_1 + 3 * x_2 - 2 * x_3 - 2 * x_4 = 7
  ) ∧ (
    x_1 = -3 + 2 * x_2 - x_4 ∧
    x_2 = 2 + (2 / 5) * x_3 + (3 / 5) * x_4
  ) :=
begin
  sorry
end

end system_of_linear_equations_solution_l108_108723


namespace necessary_but_not_sufficient_condition_l108_108370

theorem necessary_but_not_sufficient_condition (a : ℝ)
    (h : -2 ≤ a ∧ a ≤ 2)
    (hq : ∃ x y : ℂ, x ≠ y ∧ (x ^ 2 + (a : ℂ) * x + 1 = 0) ∧ (y ^ 2 + (a : ℂ) * y + 1 = 0)) :
    ∃ z : ℂ, z ^ 2 + (a : ℂ) * z + 1 = 0 ∧ (¬ ∀ b, -2 < b ∧ b < 2 → b = a) :=
sorry

end necessary_but_not_sufficient_condition_l108_108370


namespace quinary_1234_eq_194_l108_108520

def quinary_digit_place (n b : ℕ) : ℕ := n * (b ^ 0) + n * (b ^ 1) + n * (b ^ 2) + n * (b ^ 3)

def quinary_to_decimal (digits : List ℕ) : ℕ :=
  digits.foldr (λ (d : ℕ) (acc : ℕ) → d + acc * 5) 0

def digit_1234_quinary : List ℕ := [4, 3, 2, 1]

theorem quinary_1234_eq_194 : quinary_to_decimal digit_1234_quinary = 194 := by
  sorry

end quinary_1234_eq_194_l108_108520


namespace lives_per_player_l108_108415

theorem lives_per_player (total_friends : ℕ) (friends_quit : ℕ) (total_lives : ℕ) 
  (h1 : total_friends = 8) (h2 : friends_quit = 5) (h3 : total_lives = 15) :
  total_lives / (total_friends - friends_quit) = 5 :=
by
  rw [h1, h2, h3]
  norm_num

end lives_per_player_l108_108415


namespace second_number_division_l108_108560

theorem second_number_division (d x r : ℕ) (h1 : d = 16) (h2 : 25 % d = r) (h3 : 105 % d = r) (h4 : r = 9) : x % d = r → x = 41 :=
by 
  simp [h1, h2, h3, h4] 
  sorry

end second_number_division_l108_108560


namespace harry_james_payment_l108_108808

theorem harry_james_payment (x y H : ℝ) (h1 : H - 12 = 44 / y) (h2 : y > 1) (h3 : H != 12 + 44/3) : H = 23 ∧ y = 4 :=
by
  sorry

end harry_james_payment_l108_108808


namespace find_S5_l108_108929

-- Definitions from conditions
def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^(n-1)
def sum_first_n_terms (a1 q : ℝ) (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

variables (a1 q : ℝ)
variables (Hq1 : q > 1)
variables (Ha3a5 : geometric_sequence a1 q 3 + geometric_sequence a1 q 5 = 20)
variables (Ha2a6 : geometric_sequence a1 q 2 * geometric_sequence a1 q 6 = 64)

-- The theorem to be proven
theorem find_S5 : sum_first_n_terms a1 q 5 = 31 :=
sorry

end find_S5_l108_108929


namespace distinct_prime_factors_of_60_l108_108985

theorem distinct_prime_factors_of_60 : (Nat.toFinset (Nat.primeFactors 60)).card = 3 := 
by
  -- The proof goes here
  sorry

end distinct_prime_factors_of_60_l108_108985


namespace cost_price_each_watch_l108_108023

open Real

theorem cost_price_each_watch
  (C : ℝ)
  (h1 : let lossPerc := 0.075 in
        let sp_each := C * (1 - lossPerc) in
        let gainPerc := 0.053 in
        let sp_more := sp_each + 265 in
        let sp_total := 3 * sp_more in
        sp_total = 3 * C * (1 + gainPerc)) :
  C ≈ 2070.31 := by
  sorry

end cost_price_each_watch_l108_108023


namespace area_AFBC_eq_10_l108_108667

noncomputable def area_of_quadrilateral (A B C F Q : Point) : ℝ :=
  if Q.is_centroid A B C ∧ Q.segment_length B = 2 ∧ Q.segment_length C = 3 ∧ B.segment_length C = 4 then
    let AQB := triangle.area A Q B in
    let FQC := triangle.area F Q C in
    AQB + FQC
  else
    0

-- The Lean formalization proving that the area of the quadrilateral is 10
theorem area_AFBC_eq_10 (A B C F Q : Point) :
  Q.is_centroid A B C ∧ 
  Q.segment_length B = 2 ∧ 
  Q.segment_length C = 3 ∧ 
  B.segment_length C = 4 →
  area_of_quadrilateral A B C F Q = 10 := 
by
  sorry

end area_AFBC_eq_10_l108_108667


namespace coefficient_x2_l108_108736

-- Given (1-x) + (1-x)^2 + ... + (1-x)^{10}
def poly_sum (x : ℝ) : ℝ := (1 - x) + (1 - x)^2 + (1 - x)^3 + (1 - x)^4 + (1 - x)^5 + (1 - x)^6 + (1 - x)^7 + (1 - x)^8 + (1 - x)^9 + (1 - x)^10

-- Coefficient of x^2 in the expansion
theorem coefficient_x2 (x : ℝ) : (polynomial.coeff (polynomial.of_real (poly_sum x)) 2) = 165 :=
by
  sorry

end coefficient_x2_l108_108736


namespace trapezoid_rotation_180_l108_108045

theorem trapezoid_rotation_180 (l_base s_base : ℝ) (h1 : l_base > s_base) (h2 : l_base > 0) (h3 : s_base > 0) :
  rotate_180_by_symmetry (trapezoid l_base s_base) = frustum_of_cone :=
sorry

end trapezoid_rotation_180_l108_108045


namespace quadratic_solution_transformation_l108_108193

theorem quadratic_solution_transformation
  (m h k : ℝ)
  (h_nonzero : m ≠ 0)
  (x1 x2 : ℝ)
  (h_sol1 : m * (x1 - h)^2 - k = 0)
  (h_sol2 : m * (x2 - h)^2 - k = 0)
  (h_x1 : x1 = 2)
  (h_x2 : x2 = 5) :
  (∃ x1' x2', x1' = 1 ∧ x2' = 4 ∧ m * (x1' - h + 1)^2 = k ∧ m * (x2' - h + 1)^2 = k) :=
by 
  -- Proof here
  sorry

end quadratic_solution_transformation_l108_108193


namespace verify_euler_relation_for_transformed_cube_l108_108920

def euler_relation_for_transformed_cube : Prop :=
  let V := 12
  let A := 24
  let F := 14
  V + F = A + 2

theorem verify_euler_relation_for_transformed_cube :
  euler_relation_for_transformed_cube :=
by
  sorry

end verify_euler_relation_for_transformed_cube_l108_108920


namespace juice_cost_equal_fifty_l108_108083

theorem juice_cost_equal_fifty :
  ∃ J : ℕ, (J = 50) ∧
  ((3 * 25 + 2 * 75 + J = 275)) :=
by
  use 50
  constructor
  · rfl
  · simp
  sorry

end juice_cost_equal_fifty_l108_108083


namespace solve_eq_solve_ineq_l108_108065

-- Proof Problem 1 statement
theorem solve_eq (x : ℝ) : (2 / (x + 3) - (x - 3) / (2 * x + 6) = 1) → (x = 1 / 3) :=
by sorry

-- Proof Problem 2 statement
theorem solve_ineq (x : ℝ) : (2 * x - 1 > 3 * (x - 1)) ∧ ((5 - x) / 2 < x + 4) → (-1 < x ∧ x < 2) :=
by sorry

end solve_eq_solve_ineq_l108_108065


namespace symmetric_point_x_axis_l108_108373

theorem symmetric_point_x_axis (x y z : ℝ) :
    let reflected_point : ℝ × ℝ × ℝ := (x, -y, -z)
    -- Given point (2, 3, 4)
    x = 2 ∧ y = 3 ∧ z = 4 →
    -- Symmetric point with respect to x-axis should be (2, -3, -4)
    reflected_point = (2, -3, -4) :=
by
  intro h
  cases h with hx hy hz
  simp [hx, hy, hz]
  -- Prove the equality
  show (x, -y, -z) = (2, -3, -4)
  sorry

end symmetric_point_x_axis_l108_108373


namespace fraction_value_l108_108221

theorem fraction_value (x : ℝ) (h : 1 - 5 / x + 6 / x^3 = 0) : 3 / x = 3 / 2 :=
by
  sorry

end fraction_value_l108_108221


namespace value_of_q_div_p_l108_108548

open Nat

-- Definitions of the given problem conditions
def slips : List Fin8 := List.replicate 5 1 ++ List.replicate 5 2 ++ List.replicate 5 3 ++ List.replicate 5 4 ++ 
                          List.replicate 5 5 ++ List.replicate 5 6 ++ List.replicate 5 7 ++ List.replicate 5 8

def choose6 (s : List α) : Nat := Nat.choose s.length 6

-- Define p and q based on conditions
def p : ℚ := 0  -- As calculated p = 0
def q : ℚ := (28 * 10 * 10 / choose50_6

-- Define the theorem to be proved
theorem value_of_q_div_p :
  p = 0 → q / p = undefined := sorry

end value_of_q_div_p_l108_108548


namespace function_f_expression_g_decreasing_intervals_g_max_min_value_l108_108623

open Real

-- Step 1: Define the function and conditions
def f (x : ℝ) (phi : ℝ) : ℝ := 3 * sin (2 * x + phi)
def g (x : ℝ) : ℝ := 3 * sin (x + π / 6)
def decreasing_interval (k : ℤ) : set ℝ := set.Icc (2 * (k : ℝ) * π + π / 3) (2 * (k : ℝ) * π + 4 * π / 3)

theorem function_f_expression :
  ∃ φ, (0 < φ ∧ φ < π / 2) ∧ 
  (∀ x : ℝ, f x φ = 3 * sin(2 * x + π / 6)) := sorry

theorem g_decreasing_intervals :
  ∀ k : ℤ, ∃ I, I = decreasing_interval k := sorry

theorem g_max_min_value : 
  (∀ x, x ∈ set.Icc (-π / 3) (π / 2) -> g x ∈ set.Icc (-3/2) 3) := sorry

end function_f_expression_g_decreasing_intervals_g_max_min_value_l108_108623


namespace EF_parallel_BC_l108_108238

theorem EF_parallel_BC
  (A B C E F : Point)
  (hMidE : E = midpoint A B)
  (hMidF : F = midpoint A C)
  (hMedian : ∀ (Δ : Triangle) (M : Point), median Δ M → M ∥ third_side Δ) :
  EF ∥ BC := by
sorry

end EF_parallel_BC_l108_108238


namespace find_whole_number_M_l108_108091

-- Define the conditions
def condition (M : ℕ) : Prop :=
  21 < M ∧ M < 23

-- Define the main theorem to be proven
theorem find_whole_number_M (M : ℕ) (h : condition M) : M = 22 := by
  sorry

end find_whole_number_M_l108_108091


namespace betty_paid_total_l108_108101

def cost_slippers (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_lipsticks (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_hair_colors (count : ℕ) (price : ℝ) : ℝ := count * price

def total_cost := 
  cost_slippers 6 2.5 +
  cost_lipsticks 4 1.25 +
  cost_hair_colors 8 3

theorem betty_paid_total :
  total_cost = 44 := 
  sorry

end betty_paid_total_l108_108101


namespace number_of_handshakes_l108_108859

theorem number_of_handshakes (n : ℕ) (g1 : ℕ) (s1 : ℕ) (g2 : ℕ) (o : ℕ)
  (h1 : n = 36) 
  (h2 : g1 = 25) 
  (h3 : s1 = 15) 
  (h4 : g2 = 6) 
  (h5 : o = 5) 
  (h6 : g1 - s1 = 10) :
  let h := s1 * (g2 + o) + (g1 - s1) * o + g2 * o
  in h = 245 :=
by
  intros
  sorry

end number_of_handshakes_l108_108859


namespace area_of_new_rectangle_is_9_l108_108732

-- Definitions for the given dimensions and derived values
def x : ℕ := 3
def y : ℕ := 4
def diagonal : ℝ := Real.sqrt (x^2 + y^2)
def base : ℝ := diagonal - y
def altitude : ℝ := diagonal + y
def area_new : ℝ := base * altitude

-- The main statement that we want to prove
theorem area_of_new_rectangle_is_9 : area_new = 9 := 
by 
  sorry

end area_of_new_rectangle_is_9_l108_108732


namespace tangent_line_at_1_minimum_a_l108_108199

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - 3 * x^2 - 11 * x

-- 1. Equation of the tangent line to the curve y = f(x) at x = 1
theorem tangent_line_at_1 :
  ∃ m b, (m = -15 ∧ b = 29 ∧ ∀ x, f x = m * (x - 1) + b) :=
sorry

-- 2. Minimum value of integer a such that f(x) ≤ (a-3)x^2 + (2a-13)x + 1 always holds
theorem minimum_a :
  ∃ a : ℤ, a = 1 ∧ ∀ x : ℝ, f(x) ≤ (a-3)*x^2 + (2*a-13)*x + 1 :=
sorry

end tangent_line_at_1_minimum_a_l108_108199


namespace polar_coordinates_of_3_neg3_l108_108127

def convert_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if y ≥ 0 then Real.atan2 y x else if x < 0 then Real.atan2 y x + Real.pi else Real.atan2 y x + 2 * Real.pi
  (r, θ)

theorem polar_coordinates_of_3_neg3 :
  convert_to_polar 3 (-3) = (3 * Real.sqrt 2, 7 * Real.pi / 4) := by
  sorry

end polar_coordinates_of_3_neg3_l108_108127


namespace exists_integer_lt_sqrt5_l108_108385

theorem exists_integer_lt_sqrt5 (k : ℤ) (h : - (sqrt 5) < k ∧ k < sqrt 5) : k = 0 :=
sorry

end exists_integer_lt_sqrt5_l108_108385


namespace min_swaps_TEAM_to_MATE_l108_108503

def is_adjacent_swap (s₁ s₂ : String) : Prop :=
  ∃ i, i < s₁.length - 1 ∧
       s₁ = s₂.take i ++ s₂.get! (i + 1).toString
            ++ s₂.get! i.toString ++ s₂.drop (i + 2)

def transformation_sequence (start : String) (steps : ℕ) (end_seq : String) : Prop :=
  ∃ seq : List String,
    seq.length = steps + 1 ∧
    seq.head = start ∧
    seq.last = end_seq ∧
    ∀ j < steps, is_adjacent_swap (seq.nth_le j sorry) (seq.nth_le (j + 1) sorry)

theorem min_swaps_TEAM_to_MATE :
  ∃ steps, transformation_sequence "TEAM" steps "MATE" ∧ steps = 5 :=
by
  sorry

end min_swaps_TEAM_to_MATE_l108_108503


namespace integer_solutions_l108_108523

theorem integer_solutions (a : ℤ) : 
  (∃ f : ℤ → ℤ → ℤ × ℤ, 
     (∀ x y : ℤ, f x y ∈ {v : ℤ × ℤ | v.1^2 + a*v.1*v.2 + v.2^2 = 1}.toFinset) 
     ∧ 
     (∀ x₁ y₁ x₂ y₂ : ℤ, (f x₁ y₁ = f x₂ y₂) → (x₁ = x₂ ∧ y₁ = y₂))) 
  ↔ 
  (a = 2 ∨ a = -2) := sorry

end integer_solutions_l108_108523


namespace sum_of_interior_angles_of_pentagon_l108_108403

theorem sum_of_interior_angles_of_pentagon :
  let n := 5 in (n - 2) * 180 = 540 := 
by 
  let n := 5
  show (n - 2) * 180 = 540
  sorry

end sum_of_interior_angles_of_pentagon_l108_108403


namespace intersection_point_polar_coords_max_value_MBC_l108_108830

noncomputable def polarEquation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

noncomputable def lineInclination (α : ℝ) : Prop := α = (3 * Real.pi) / 4

theorem intersection_point_polar_coords :
  ∃ (ρ θ : ℝ), polarEquation ρ θ ∧ lineInclination θ ∧ θ = (3 * Real.pi) / 4 ∧ ρ = 2 * Real.sqrt 2 :=
sorry

theorem max_value_MBC :
  ∃ (MB MC d : ℝ), d = 2 * Real.sqrt 2 ∧ 
                     lineInclination (Real.pi / 4) ∧ 
                    (MB = Real.abs (2 * (Real.sin (3 * Real.pi / 4) + Real.cos (3 * Real.pi / 4)))) ∧
                    (MC = Real.abs (2 * (Real.sin (7 * Real.pi / 4) + Real.cos (7 * Real.pi / 4)))) :=
sorry

end intersection_point_polar_coords_max_value_MBC_l108_108830


namespace max_sum_two_five_digit_numbers_l108_108543

theorem max_sum_two_five_digit_numbers : ∃ n₁ n₂ : ℕ, 
  n₁ + n₂ = 97531 + 86420 ∧ 
  (n₁ = 87431 ∨ n₂ = 87431) ∧ 
  (∀ d ∈ [0,1,2,3,4,5,6,7,8,9], 
    (d ∈ to_digit_list n₁) ∨ (d ∈ to_digit_list n₂)) := 
sorry

end max_sum_two_five_digit_numbers_l108_108543


namespace complex_quadrant_l108_108922

def is_quadrant_III (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_quadrant 
  (z : ℂ)
  (h : (1 + 2 * complex.i) * z = 3 - 4 * complex.i) : 
  is_quadrant_III z :=
sorry

end complex_quadrant_l108_108922


namespace bcm_hens_count_l108_108473

-- Propositions representing the given conditions
def total_chickens : ℕ := 100
def bcm_ratio : ℝ := 0.20
def bcm_hens_ratio : ℝ := 0.80

-- Theorem statement: proving the number of BCM hens
theorem bcm_hens_count : (total_chickens * bcm_ratio * bcm_hens_ratio = 16) := by
  sorry

end bcm_hens_count_l108_108473


namespace veg_price_proof_min_spent_proof_l108_108345

def A_veg_base_price (market_price: ℚ) (base_price: ℚ): Prop :=
  market_price = (5/4) * base_price

def purchase_equation (base_price: ℚ): Prop :=
  (300 / base_price) = (300 / ((5/4) * base_price)) + 3

def B_veg_base_price: ℚ := 30

def total_bundles (total: ℕ) (A: ℕ) (B: ℕ): Prop :=
  total = A + B

def A_not_exceed_B (A: ℕ) (B: ℕ): Prop :=
  A ≤ B

def discounted_price (base_price_A base_price_B: ℚ) (A B: ℕ): ℚ :=
  0.9 * base_price_A * A + 0.9 * base_price_B * B

def min_spent (base_price_A base_price_B: ℚ) (total: ℕ): ℚ :=
  let min_A := total / 2 in
  0.9 * base_price_A * min_A + 0.9 * base_price_B * (total - min_A)

theorem veg_price_proof: 
  ∃ base_price: ℚ, 
  A_veg_base_price (5 / 4 * base_price) base_price ∧ 
  purchase_equation base_price ∧ 
  base_price = 20 := 
by 
  sorry

theorem min_spent_proof:
  ∃ w: ℚ, 
  total_bundles 100 A B ∧ 
  A_not_exceed_B A B ∧ 
  (base_price_A = 20) ∧ 
  (base_price_B = B_veg_base_price) ∧ 
  w = min_spent base_price_A B_veg_base_price 100 ∧ 
  w = 2250 := 
by 
  sorry

end veg_price_proof_min_spent_proof_l108_108345


namespace min_pressure_l108_108328

-- Constants and assumptions
variables (V0 T0 a b c R : ℝ)
hypothesis h1 : c^2 < a^2 + b^2

-- Functions
def cyclic_process (V T : ℝ) : Prop :=
  (V / V0 - a)^2 + (T / T0 - b)^2 = c^2

def ideal_gas_law (P V T : ℝ) : Prop :=
  T = P * V / R

-- Minimum pressure
theorem min_pressure (P_min : ℝ) : 
  (∃ V T, cyclic_process V T ∧ ideal_gas_law P_min V T) → 
  P_min = (R * T0 / V0) * (a * sqrt (a^2 + b^2 - c^2) - b * c) / 
                       (b * sqrt (a^2 + b^2 - c^2) + a * c) :=
sorry

end min_pressure_l108_108328


namespace maximal_n_is_k_minus_1_l108_108428

section
variable (k : ℕ) (n : ℕ)
variable (cards : Finset ℕ)
variable (red : List ℕ) (blue : List (List ℕ))

-- Conditions
axiom h_k_pos : k > 1
axiom h_card_count : cards = Finset.range (2 * n + 1)
axiom h_initial_red : red = (List.range' 1 (2 * n)).reverse
axiom h_initial_blue : blue.length = k

-- Question translated to a goal
theorem maximal_n_is_k_minus_1 (h : ∀ (n' : ℕ), n' ≤ (k - 1)) : n = k - 1 :=
sorry
end

end maximal_n_is_k_minus_1_l108_108428


namespace water_needed_l108_108090

-- Definitions as per conditions
def heavy_wash : ℕ := 20
def regular_wash : ℕ := 10
def light_wash : ℕ := 2
def extra_light_wash (bleach : ℕ) : ℕ := bleach * light_wash

def num_heavy_washes : ℕ := 2
def num_regular_washes : ℕ := 3
def num_light_washes : ℕ := 1
def num_bleached_loads : ℕ := 2

-- Function to calculate total water usage
def total_water_used : ℕ :=
  (num_heavy_washes * heavy_wash) +
  (num_regular_washes * regular_wash) +
  (num_light_washes * light_wash) + 
  (extra_light_wash num_bleached_loads)

-- Theorem to be proved
theorem water_needed : total_water_used = 76 := by
  sorry

end water_needed_l108_108090


namespace find_special_two_digit_numbers_l108_108459

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_special (A : ℕ) : Prop :=
  let sum_A := sum_digits A
  sum_A^2 = sum_digits (A^2)

theorem find_special_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A < 100 ∧ is_special A} = {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by 
  sorry

end find_special_two_digit_numbers_l108_108459


namespace age_30_years_from_now_l108_108890

variables (ElderSonAge : ℕ) (DeclanAgeDiff : ℕ) (YoungerSonAgeDiff : ℕ) (ThirdSiblingAgeDiff : ℕ)

-- Given conditions
def elder_son_age : ℕ := 40
def declan_age : ℕ := elder_son_age + 25
def younger_son_age : ℕ := elder_son_age - 10
def third_sibling_age : ℕ := younger_son_age - 5

-- To prove the ages 30 years from now
def younger_son_age_30_years_from_now : ℕ := younger_son_age + 30
def third_sibling_age_30_years_from_now : ℕ := third_sibling_age + 30

-- The proof statement
theorem age_30_years_from_now : 
  younger_son_age_30_years_from_now = 60 ∧ 
  third_sibling_age_30_years_from_now = 55 :=
by
  sorry

end age_30_years_from_now_l108_108890


namespace tan_pi_div_4_sub_theta_l108_108165

theorem tan_pi_div_4_sub_theta (theta : ℝ) (h : Real.tan theta = 1 / 2) : 
  Real.tan (π / 4 - theta) = 1 / 3 := 
sorry

end tan_pi_div_4_sub_theta_l108_108165


namespace pool_capacity_is_1000_l108_108451

def total_capacity_of_pool := Nat
variable (C : total_capacity_of_pool)

-- Conditions
def condition1 : Prop := C * 55 / 100 + 300 = C * 85 / 100
def condition2 : Prop := C * 30 / 100 = 300

-- Conclusion
theorem pool_capacity_is_1000 (C : total_capacity_of_pool) 
    (h1 : condition1 C)
    (h2 : condition2 C) : C = 1000 := 
by
  sorry

end pool_capacity_is_1000_l108_108451


namespace philosopher_chessboard_max_cells_l108_108820

/-- The maximum number of cells such that one cannot go from any cell to another with a finite number
of movements of the Donkey on an infinite hexagonal grid -/
theorem philosopher_chessboard_max_cells (m n : ℕ) : 
  ∃ (max_cells : ℕ), 
    (∀ (x₁ y₁ x₂ y₂ : ℤ), valid_cell m n x₁ y₁ → valid_cell m n x₂ y₂ →
      not_movable m n x₁ y₁ x₂ y₂ ↔ (dist x₁ y₁ x₂ y₂) ≤ max_cells) ∧
    max_cells = m ^ 2 + m * n + n ^ 2 := 
sorry

noncomputable def valid_cell (m n : ℕ) (x y : ℤ) : Prop := 
(x >= 0) ∧ (y >= 0)

noncomputable def not_movable (m n : ℕ) (x₁ y₁ x₂ y₂ : ℤ) : Prop := 
¬∃ (k : ℕ), (valid_move m n x₁ y₁ x₂ y₂ k)

noncomputable def valid_move (m n : ℕ) (x₁ y₁ x₂ y₂ : ℤ) (k : ℕ) : Prop :=
-- Define the movements based on the problem's conditions
...

noncomputable def dist (x₁ y₁ x₂ y₂ : ℤ) : ℕ :=
-- Define the suitable distance due to the hexagonal grid
...

end philosopher_chessboard_max_cells_l108_108820


namespace find_x_solution_l108_108154

theorem find_x_solution (x : ℝ) (h : sqrt (x - 5) = 7) : x = 54 :=
sorry

end find_x_solution_l108_108154


namespace perfect_squares_99_l108_108218

theorem perfect_squares_99 : 
  {x : ℕ // (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ x = a^2 ∧ x + 99 = b^2)}.card = 3 :=
by
  sorry

end perfect_squares_99_l108_108218


namespace oil_depth_upright_l108_108840

noncomputable def oil_tank_volume (height radius : ℝ) : ℝ :=
  π * radius^2 * height

theorem oil_depth_upright :
  let height := 15
  let diameter := 6
  let radius := (diameter / 2 : ℝ)
  let height_flat := 4
  let volume := oil_tank_volume height radius
  let height_upright := volume / (π * radius^2)
  height_upright = 15 :=
by 
  sorry

end oil_depth_upright_l108_108840


namespace number_of_segments_l108_108360

theorem number_of_segments (tangent_chords : ℕ) (angle_ABC : ℝ) (h : angle_ABC = 80) :
  tangent_chords = 18 :=
sorry

end number_of_segments_l108_108360


namespace yvette_final_bill_l108_108288

def cost_alicia : ℝ := 7.50
def cost_brant : ℝ := 10.00
def cost_josh : ℝ := 8.50
def cost_yvette : ℝ := 9.00
def tip_rate : ℝ := 0.20

def total_cost := cost_alicia + cost_brant + cost_josh + cost_yvette
def tip := tip_rate * total_cost
def final_bill := total_cost + tip

theorem yvette_final_bill :
  final_bill = 42.00 :=
  sorry

end yvette_final_bill_l108_108288


namespace a_seq_formula_l108_108971

/-- Defining the sequence a_n and conjecturing the explicit formula. -/
def a_seq : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * a_seq n + 1

theorem a_seq_formula (n : ℕ) : a_seq (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end a_seq_formula_l108_108971


namespace circle_area_l108_108907

noncomputable def pi_approx : ℝ := 3.14159

theorem circle_area (d : ℝ) (h : d = 8) : (π * (d / 2)^2) ≈ 50.26544 :=
by
  have r := d / 2
  have r_value : r = 4 := by rw [h]; norm_num
  have area := π * r^2
  have approx_area : area ≈ 50.26544 := by
    rw [r_value, show π ≈ 3.14159 by norm_num, show 4^2 = 16 by norm_num]
    norm_num
  exact approx_area

end circle_area_l108_108907


namespace functional_equation_solution_l108_108551

theorem functional_equation_solution (f : ℤ → ℤ)
  (h : ∀ m n : ℤ, f (f (m + n)) = f m + f n) :
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a) ∨ (∀ n : ℤ, f n = 0) := by
  sorry

end functional_equation_solution_l108_108551


namespace equilateral_triangle_DE_length_l108_108658

theorem equilateral_triangle_DE_length
  (A B C D E F G H J : Type)
  [triangle A B C]
  [equilateral_triangle A B C]
  (H1 : points_on_sides A B C D E F G H J)
  (H2 : BD = DE)
  (H3 : EF = FG)
  (H4 : GH = HJ = JC)
  (BD_len : BD = 4)
  (EF_len : EF = 5)
  (HJ_len : HJ = 2) :
  DE = 4 :=
sorry

end equilateral_triangle_DE_length_l108_108658


namespace sum_of_reciprocal_of_divisors_360_l108_108564

-- Definitions
def is_divisor (a d : ℕ) : Prop := d ∣ a

def sum_of_divisors (a : ℕ) : ℕ :=
  (set.to_finset {d | is_divisor a d}).sum id

def sum_of_reciprocal_of_divisors (a : ℕ) : ℚ :=
  (set.to_finset {d | is_divisor a d}).sum (λ d, 1 / d)

-- The problem
theorem sum_of_reciprocal_of_divisors_360 :
  sum_of_reciprocal_of_divisors 360 = 13 / 4 :=
sorry

end sum_of_reciprocal_of_divisors_360_l108_108564


namespace abs_pi_sub_abs_pi_sub_nine_l108_108118

theorem abs_pi_sub_abs_pi_sub_nine (h : Real.pi < 9) : 
  |Real.pi - |Real.pi - 9|| = 9 - 2 * Real.pi :=
sorry

end abs_pi_sub_abs_pi_sub_nine_l108_108118


namespace min_value_f_when_a_eq_1_no_extrema_implies_a_ge_four_thirds_l108_108319

section
variables {a x : ℝ}

/-- Define the function f(x) = ax^3 - 2x^2 + x + c where c = 1 -/
def f (a x : ℝ) : ℝ := a * x^3 - 2 * x^2 + x + 1

/-- Proposition 1: Minimum value of f when a = 1 and f passes through (0,1) is 1 -/
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f 1 x ≥ 1) := 
by {
  -- Sorry for the full proof
  sorry
}

/-- Proposition 2: If f has no extremum points, then a ≥ 4/3 -/
theorem no_extrema_implies_a_ge_four_thirds (h : ∀ x : ℝ, 3 * a * x^2 - 4 * x + 1 ≠ 0) : 
  a ≥ (4 / 3) :=
by {
  -- Sorry for the full proof
  sorry
}

end

end min_value_f_when_a_eq_1_no_extrema_implies_a_ge_four_thirds_l108_108319


namespace cuboid_surface_area_l108_108453

-- Define the dimensions of the cuboid
def length : ℝ := 12
def breadth : ℝ := 6
def height : ℝ := 10

-- Define the surface area of a cuboid formula
def surface_area (l b h : ℝ) : ℝ := 2 * (l * h + l * b + b * h)

-- Stating the problem
theorem cuboid_surface_area : surface_area length breadth height = 504 := by
  sorry

end cuboid_surface_area_l108_108453


namespace mod_exp_result_l108_108794

theorem mod_exp_result :
  (2 ^ 46655) % 9 = 1 :=
by
  sorry

end mod_exp_result_l108_108794


namespace count_ordered_pairs_polynomial_factors_l108_108144

theorem count_ordered_pairs_polynomial_factors :
  (∃ N : ℕ, N = ∑ (a : ℕ) in finset.range 51, ∑ (b : ℕ) in finset.range (a * a + 1), 
   ∃ r s : ℕ, r + s = a ∧ r * s = b) := 
sorry

end count_ordered_pairs_polynomial_factors_l108_108144


namespace average_sales_six_months_l108_108077

theorem average_sales_six_months :
  let sales1 := 4000
  let sales2 := 6524
  let sales3 := 5689
  let sales4 := 7230
  let sales5 := 6000
  let sales6 := 12557
  let total_sales_first_five := sales1 + sales2 + sales3 + sales4 + sales5
  let total_sales_six := total_sales_first_five + sales6
  let average_sales := total_sales_six / 6
  average_sales = 7000 :=
by
  let sales1 := 4000
  let sales2 := 6524
  let sales3 := 5689
  let sales4 := 7230
  let sales5 := 6000
  let sales6 := 12557
  let total_sales_first_five := sales1 + sales2 + sales3 + sales4 + sales5
  let total_sales_six := total_sales_first_five + sales6
  let average_sales := total_sales_six / 6
  have h : total_sales_first_five = 29443 := by sorry
  have h1 : total_sales_six = 42000 := by sorry
  have h2 : average_sales = 7000 := by sorry
  exact h2

end average_sales_six_months_l108_108077


namespace arithmetic_seq_proof_l108_108194

open Nat

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a d : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (n : ℕ) : ℤ :=
n * (a 1) + n * (n - 1) / 2 * d

theorem arithmetic_seq_proof (a : ℕ → ℤ) (d : ℤ)
  (h1 : arithmetic_seq a d)
  (h2 : a 2 = 0)
  (h3 : sum_of_arithmetic_seq a d 3 + sum_of_arithmetic_seq a d 4 = 6) :
  a 5 + a 6 = 21 :=
sorry

end arithmetic_seq_proof_l108_108194


namespace rockets_win_series_in_exactly_7_games_l108_108365

theorem rockets_win_series_in_exactly_7_games :
  let p_warriors_win := (3 / 4 : ℚ),
      p_rockets_win := (1 / 4 : ℚ),
      comb_6_3 := (Nat.choose 6 3 : ℚ),
      prob_3_3_tie := comb_6_3 * (p_rockets_win ^ 3) * (p_warriors_win ^ 3),
      prob_rockets_win_7th := p_rockets_win
  in (prob_3_3_tie * prob_rockets_win_7th) = (135 / 4096 : ℚ) := by
  sorry

end rockets_win_series_in_exactly_7_games_l108_108365


namespace unique_x_ffx_eq_4_l108_108211

-- Assumptions about the function f
variable (f : ℝ → ℝ)
variable (h : ∀ x, f(x) = 4 ↔ x = 4)

-- Statement of the problem
theorem unique_x_ffx_eq_4 : set.count ((λ x, f(f x) = 4) '' {x : ℝ | true}) 1 := sorry

end unique_x_ffx_eq_4_l108_108211


namespace positive_difference_solutions_of_abs_eq_l108_108041

theorem positive_difference_solutions_of_abs_eq (x1 x2 : ℝ) (h1 : 2 * x1 - 3 = 15) (h2 : 2 * x2 - 3 = -15) : |x1 - x2| = 15 := by
  sorry

end positive_difference_solutions_of_abs_eq_l108_108041


namespace percent_y_of_x_l108_108454

-- Definitions and assumptions based on the problem conditions
variables (x y : ℝ)
-- Given: 20% of (x - y) = 14% of (x + y)
axiom h : 0.20 * (x - y) = 0.14 * (x + y)

-- Prove that y is 0.1765 (or 17.65%) of x
theorem percent_y_of_x (x y : ℝ) (h : 0.20 * (x - y) = 0.14 * (x + y)) : 
  y = 0.1765 * x :=
sorry

end percent_y_of_x_l108_108454


namespace tangent_lines_to_circle_O_through_point_M_equation_of_circle_center_M_fixed_point_and_constant_ratio_l108_108618

-- Definitions of the given conditions
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def point_M_x : ℝ := 1
def point_M_y : ℝ := 4

-- Theorem statement for the tangent line equations
theorem tangent_lines_to_circle_O_through_point_M :
  (∃ k : ℝ, 15 * point_M_x - 8 * point_M_y + 17 = 0 ∨ point_M_x = 1) :=
begin
  sorry
end

-- Theorem statement for the equation of the circle with specified conditions
theorem equation_of_circle_center_M :
  ∃ x y : ℝ, (x - point_M_x)^2 + (y - point_M_y)^2 = 36 := 
begin
  sorry
end

-- Theorem statements for the fixed point R and the constant ratio PQ/PR
theorem fixed_point_and_constant_ratio :
  (∃ R_x R_y : ℝ, (R_x = -1 ∧ R_y = -4 ∧ (∃ P_x P_y : ℝ, (P_x - R_x)^2 + (P_y - R_y)^2 = (P_x^2 + P_y^2 - 1) / 2)) ∨
                    (R_x = -1/17 ∧ R_y = -4/17 ∧ (∃ P_x P_y : ℝ, (P_x - R_x)^2 + (P_y - R_y)^2 = (P_x^2 + P_y^2 - 1) * 18/17)) :=
begin
  sorry
end

end tangent_lines_to_circle_O_through_point_M_equation_of_circle_center_M_fixed_point_and_constant_ratio_l108_108618


namespace team_total_points_l108_108495

theorem team_total_points (T : ℕ) (h1 : ∃ x : ℕ, x = T / 6)
    (h2 : (T + (92 - 85)) / 6 = 84) : T = 497 := 
by sorry

end team_total_points_l108_108495


namespace sqrt_eq_seven_iff_x_eq_fifty_four_l108_108147

theorem sqrt_eq_seven_iff_x_eq_fifty_four (x : ℝ) : sqrt (x - 5) = 7 ↔ x = 54 := by
  sorry

end sqrt_eq_seven_iff_x_eq_fifty_four_l108_108147


namespace tan_alpha_eq_cos_two_alpha_plus_quarter_pi_sin_beta_eq_l108_108162

-- Definitions
variables {α β : ℝ}

-- Condition: 0 < α < π / 2
def valid_alpha (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Condition: sin α = 4 / 5
def sin_alpha (α : ℝ) : Prop := Real.sin α = 4 / 5

-- Condition: 0 < β < π / 2
def valid_beta (β : ℝ) : Prop := 0 < β ∧ β < Real.pi / 2

-- Condition: cos (α + β) = -1 / 2
def cos_alpha_add_beta (α β : ℝ) : Prop := Real.cos (α + β) = - 1 / 2

/-- Proofs begin -/
-- Proof for tan α = 4 / 3 given 0 < α < π / 2 and sin α = 4 / 5
theorem tan_alpha_eq (α : ℝ) (h_valid : valid_alpha α) (h_sin : sin_alpha α) : Real.tan α = 4 / 3 := 
  sorry

-- Proof for cos (2α + π / 4) = -31√2 / 50 given 0 < α < π / 2 and sin α = 4 / 5
theorem cos_two_alpha_plus_quarter_pi (α : ℝ) (h_valid : valid_alpha α) (h_sin : sin_alpha α) : 
  Real.cos (2 * α + Real.pi / 4) = -31 * Real.sqrt 2 / 50 := 
  sorry

-- Proof for sin β = 4 + 3√3 / 10 given 0 < α < π / 2, sin α = 4 / 5, 0 < β < π / 2 and cos (α + β) = -1 / 2
theorem sin_beta_eq (α β : ℝ) (h_validα : valid_alpha α) (h_sinα : sin_alpha α) 
  (h_validβ : valid_beta β) (h_cosαβ : cos_alpha_add_beta α β) : Real.sin β = 4 + 3 * Real.sqrt 3 / 10 := 
  sorry

end tan_alpha_eq_cos_two_alpha_plus_quarter_pi_sin_beta_eq_l108_108162


namespace sqrt_diff_eq_l108_108743

theorem sqrt_diff_eq :
  sqrt (4 / 3) - sqrt (3 / 4) = sqrt 3 / 6 := 
sorry

end sqrt_diff_eq_l108_108743


namespace binomial_division_value_l108_108157

def generalized_binomial_coefficient (a : ℝ) (k : ℕ) : ℝ :=
  (List.range k).map (λ i, a - i).prod / (List.range k).map (λ i, i + 1).prod

theorem binomial_division_value :
  (generalized_binomial_coefficient (-3/2) 100) / (generalized_binomial_coefficient (3/2) 100) = -67 :=
  sorry

end binomial_division_value_l108_108157


namespace isabel_camera_pics_l108_108284

-- Conditions
def phone_pics := 2
def albums := 3
def pics_per_album := 2

-- Define the total pictures and camera pictures
def total_pics := albums * pics_per_album
def camera_pics := total_pics - phone_pics

theorem isabel_camera_pics : camera_pics = 4 :=
by
  -- The goal is translated from the correct answer in step b)
  sorry

end isabel_camera_pics_l108_108284


namespace revenue_change_l108_108061

theorem revenue_change (x : ℝ) 
  (increase_in_1996 : ∀ R : ℝ, R * (1 + x/100) > R) 
  (decrease_in_1997 : ∀ R : ℝ, R * (1 + x/100) * (1 - x/100) < R * (1 + x/100)) 
  (decrease_from_1995_to_1997 : ∀ R : ℝ, R * (1 + x/100) * (1 - x/100) = R * 0.96): 
  x = 20 :=
by
  sorry

end revenue_change_l108_108061


namespace MK_eq_KN_l108_108480

open Real -- if necessary, otherwise you can specify the field you're working with
open Set
open EuclideanGeometry -- assuming necessary definitions are provided here

namespace ProofOfMKKN

noncomputable def point (α : Type _) := α

variables {α : Type _} [MetricSpace α]

variables (A B K M N : point α)

-- Conditions
def on_diameter_circle (K : point α) (A B : point α) := K ∈ Circle (midpoint A B) (dist A B / 2) ∧ K ≠ A

def second_circle (B M N : point α) (r : ℝ) := M ∈ Circle B r ∧ N ∈ Circle B r

def intersects_at (A K : point α) (M N : point α) := Collinear A K M ∧ Collinear A K N ∧ M ≠ N

-- Theorem to prove
theorem MK_eq_KN (h1 : on_diameter_circle K A B) (h2 : second_circle B M N (dist B M)) (h3 : intersects_at A K M N) :
  dist M K = dist K N :=
by
  sorry

end ProofOfMKKN

end MK_eq_KN_l108_108480


namespace right_triangle_condition_l108_108048

theorem right_triangle_condition (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : a^2 + b^2 = c^2 :=
by sorry

end right_triangle_condition_l108_108048


namespace dessert_eating_contest_l108_108020

theorem dessert_eating_contest (a b c : ℚ) 
  (h1 : a = 5/6) 
  (h2 : b = 7/8) 
  (h3 : c = 1/2) :
  b - a = 1/24 ∧ a - c = 1/3 := 
by 
  sorry

end dessert_eating_contest_l108_108020


namespace count_multiples_of_5_not_10_or_15_l108_108635

theorem count_multiples_of_5_not_10_or_15 : 
  ∃ n : ℕ, n = 33 ∧ (∀ x : ℕ, x < 500 ∧ (x % 5 = 0) ∧ (x % 10 ≠ 0) ∧ (x % 15 ≠ 0) → x < 500 ∧ (x % 5 = 0) ∧ (x % 10 ≠ 0) ∧ (x % 15 ≠ 0)) :=
by
  sorry

end count_multiples_of_5_not_10_or_15_l108_108635


namespace fourth_quadrant_negative_half_x_axis_upper_half_plane_l108_108161

theorem fourth_quadrant (m : ℝ) : ((-7 < m ∧ m < 3) ↔ ((m^2 - 8 * m + 15 > 0) ∧ (m^2 + 3 * m - 28 < 0))) :=
sorry

theorem negative_half_x_axis (m : ℝ) : (m = 4 ↔ ((m^2 - 8 * m + 15 < 0) ∧ (m^2 + 3 * m - 28 = 0))) :=
sorry

theorem upper_half_plane (m : ℝ) : ((m ≥ 4 ∨ m ≤ -7) ↔ (m^2 + 3 * m - 28 ≥ 0)) :=
sorry

end fourth_quadrant_negative_half_x_axis_upper_half_plane_l108_108161


namespace limit_of_n_b_n_l108_108130

noncomputable def L (x : ℝ) : ℝ := x - x^3 / 3

noncomputable def b_n (n : ℕ) : ℝ :=
  (nat.iterate L n) (25 / n)

theorem limit_of_n_b_n :
  tendsto (λ n, n * (b_n n)) at_top (𝓝 (75 / 4)) :=
begin
  sorry
end

end limit_of_n_b_n_l108_108130


namespace determine_a_l108_108166

theorem determine_a (a : ℕ)
  (h1 : 2 / (2 + 3 + a) = 1 / 3) : a = 1 :=
by
  sorry

end determine_a_l108_108166


namespace find_a_from_conditions_l108_108000

noncomputable def f (x b : ℤ) : ℤ := 4 * x + b

theorem find_a_from_conditions (b a : ℤ) (h1 : a = f (-4) b) (h2 : -4 = f a b) : a = -4 :=
by
  sorry

end find_a_from_conditions_l108_108000


namespace customer_new_shoe_size_l108_108326

variable (old_size_customer : ℕ)
variable (old_size_son new_size_son : ℕ)
variable (f : ℕ → ℕ → ℕ)

-- Conditions
def condition1 : old_size_customer = 43 := by sorry
def condition2 : f old_size_son new_size_son = 10 := by sorry
def condition3 : old_size_son = 40 := by sorry
def condition4 : new_size_son = 25 := by sorry

-- Question: Calculate the customer's new shoe size given the conditions
theorem customer_new_shoe_size :
  f old_size_customer 10 / 2 = 26.5 := by
  exact sorry

end customer_new_shoe_size_l108_108326


namespace coat_price_reduction_l108_108811

theorem coat_price_reduction (original_price reduction_amount : ℝ) (h : original_price = 500) (h_red : reduction_amount = 150) :
  ((reduction_amount / original_price) * 100) = 30 :=
by
  rw [h, h_red]
  norm_num

end coat_price_reduction_l108_108811


namespace probability_is_half_l108_108725

noncomputable def probability_at_least_35_cents : ℚ :=
  let total_outcomes := 32
  let successful_outcomes := 8 + 4 + 4 -- from solution steps (1, 2, 3)
  successful_outcomes / total_outcomes

theorem probability_is_half :
  probability_at_least_35_cents = 1 / 2 := by
  -- proof details are not required as per instructions
  sorry

end probability_is_half_l108_108725


namespace triangle_properties_l108_108283

theorem triangle_properties :
  (∀ (α β γ : ℝ), α + β + γ = 180 → 
    (α = β ∨ α = γ ∨ β = γ ∨ 
     (α = 60 ∧ β = 60 ∧ γ = 60) ∨
     ¬(α = 90 ∧ β = 90))) :=
by
  -- Placeholder for the actual proof, ensuring the theorem can build
  intros α β γ h₁
  sorry

end triangle_properties_l108_108283


namespace solve_for_x_l108_108149

theorem solve_for_x (x : ℝ) (h : sqrt (x - 5) = 7) : x = 54 :=
by
  sorry

end solve_for_x_l108_108149


namespace hexagon_perimeter_l108_108549

def ab : ℕ := 8
def bc : ℕ := 15
def ef : ℕ := 5
def cd : ℕ := ab + ef
def af := bc - (ef + ab)
def ed := bc - (ef + ab)
def perimeter : ℕ := ab + bc + cd + ed + ef + af 

theorem hexagon_perimeter :
  ab + bc + cd + ed + ef + af = 56 :=
by
  -- Define necessary variables
  let AB := ab
  let BC := bc 
  let EF := ef 
  let CD := cd 
  let ED := ed 
  let AF := af 
  -- Begin with the given conditions
  have h1 : AB = 8 := rfl
  have h2 : BC = 15 := rfl
  have h3 : EF = 5 := rfl
  have h4 : CD = AB + EF := rfl
  have h5 : ED = BC - CD := rfl
  have h6 : AF = BC - CD := rfl
  -- Use them to verify the perimeter
  calc
    AB + BC + CD + ED + EF + AF
      = 8 + 15 + (8 + 5) + (15 - (8 + 5)) + 5 + (15 - (8 + 5)) : by rw [h1, h2, h3, h4, h5, h6]
  ... = 8 + 15 + 13 + (7) + 5 + 7 : by simp
  ... = 56 : by simp

end hexagon_perimeter_l108_108549


namespace parabola_directrix_l108_108559

theorem parabola_directrix (x : ℝ) :
  (∃ y : ℝ, y = (x^2 - 8*x + 12) / 16) →
  ∃ directrix : ℝ, directrix = -17 / 4 :=
by
  sorry

end parabola_directrix_l108_108559


namespace evaluate_expression_l108_108584

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem evaluate_expression : 1 + x^4 + x^8 + x^{12} + x^{16} = 0 := sorry

end evaluate_expression_l108_108584


namespace point_in_second_quadrant_l108_108663

theorem point_in_second_quadrant (z : ℂ) (h : z = 2 * complex.I) : 
    (complex.re z = 0 ∧ complex.im z > 0) := 
by {
  rw h,
  rw complex.re, rw complex.im,
  apply and.intro,
  { rw complex.re_I, ring, },
  { simp only [zero_lt_bit0, zero_lt_one], }
}

end point_in_second_quadrant_l108_108663


namespace part1_part2_l108_108213

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 3 ∧ ∀ n : ℕ, 1 <= n → a (n + 2) = 3 * a (n + 1) - 2 * a n

def geometric (d : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, d (n + 1) = r * d n

def T (b : ℕ → ℕ → ℝ) (T_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T_n n = ∑ k in finset.range (n + 1), b k

theorem part1 (a : ℕ → ℕ) (h : seq a) : 
  geometric (λ n, a (n + 1) - a n) :=
sorry

theorem part2 (a : ℕ → ℕ) (h : seq a)
  (b : ℕ → ℝ := λ n, 2^(n-1) / (a n * a (n + 1)))
  (T_n : ℕ → ℝ := λ n, ∑ k in finset.range (n + 1), b k) : 
  ∀ n : ℕ, T_n n < 1 / 2 :=
sorry

end part1_part2_l108_108213


namespace max_balloons_l108_108704

theorem max_balloons (regular_price : ℕ) (total_money : ℕ) (full_price_balloons : ℕ) :
  regular_price = 4 → full_price_balloons = 35 → 
  total_money = regular_price * full_price_balloons →
  (∃ max_balloons : ℕ, max_balloons = 42) :=
by
  intros h1 h2 h3
  use 42
  sorry

end max_balloons_l108_108704


namespace b_minus_c_eq_log_1729_one_over_220_l108_108156

-- Conditions as definitions in Lean 4
def a (n : ℕ) (hn : n > 1) : ℝ := 1 / Real.log 1729 / Real.log n

def b : ℝ := a 2 (by decide) + a 3 (by decide) + a 5 (by decide) + a 7 (by decide)

def c : ℝ := a 11 (by decide) + a 13 (by decide) + a 17 (by decide) + a 19 (by decide)

-- Proof problem statement
theorem b_minus_c_eq_log_1729_one_over_220 : b - c = Real.log 1729 (1 / 220) := by
  sorry

end b_minus_c_eq_log_1729_one_over_220_l108_108156


namespace evaluate_complex_power_expression_l108_108138

theorem evaluate_complex_power_expression : (i : ℂ)^23 + ((i : ℂ)^105 * (i : ℂ)^17) = -i - 1 := by
  sorry

end evaluate_complex_power_expression_l108_108138


namespace number_of_polynomials_l108_108135

theorem number_of_polynomials : 
  (∃ n a_0 a_1 ... a_n, n + |a_0| + |a_1| + ... + |a_n| = 5) → 
  (number_of_such_polynomials == 66) :=
sorry

end number_of_polynomials_l108_108135


namespace right_triangle_set_C_l108_108051

theorem right_triangle_set_C :
  ∃ (a b c : ℕ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_set_C_l108_108051


namespace find_a_from_conditions_l108_108001

noncomputable def f (x b : ℤ) : ℤ := 4 * x + b

theorem find_a_from_conditions (b a : ℤ) (h1 : a = f (-4) b) (h2 : -4 = f a b) : a = -4 :=
by
  sorry

end find_a_from_conditions_l108_108001


namespace calculate_total_amount_l108_108293

theorem calculate_total_amount : ∃ S : ℕ, S = 984100 :=
by
  let a1 := 100
  let r := 3
  let n := 9 -- 8.93 rounded to the nearest integer
  let S := a1 * (1 - r^n) / (1 - r)
  -- Use Lean3 syntax for calc
  calc S 
    = a1 * (1 - r^n) / (1 - r) : rfl
    ... = 100 * (1 - 3^9) / (1 - 3) : by rw[a1, r, n]
    ... = 100 * (1 - 19683) / (1 - 3) : by norm_num
    ... = 100 * -19682 / -2 : by norm_num
    ... = 100 * 9841 : by norm_num
    ... = 984100 : by norm_num
  use S
  sorry

end calculate_total_amount_l108_108293


namespace pascal_row_10_sum_l108_108873

-- Define the function that represents the sum of Row n in Pascal's Triangle
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- State the theorem to be proven
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  -- Proof is omitted
  sorry

end pascal_row_10_sum_l108_108873


namespace betty_paid_44_l108_108104

def slippers := 6
def slippers_cost := 2.5
def lipstick := 4
def lipstick_cost := 1.25
def hair_color := 8
def hair_color_cost := 3

noncomputable def total_cost := (slippers * slippers_cost) + (lipstick * lipstick_cost) + (hair_color * hair_color_cost)

theorem betty_paid_44 : total_cost = 44 :=
by
  sorry

end betty_paid_44_l108_108104


namespace ellipse_equation_l108_108948

theorem ellipse_equation :
  (∀ (a b : ℝ), (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∧
    ∃ c : ℝ, c = sqrt 2 ∧ a^2 - b^2 = c^2) →
  (∃ (a b : ℝ), (a^2 = 4 ∧ b^2 = 2) ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 2 = 1)) :=
begin
  sorry
end

end ellipse_equation_l108_108948


namespace quadratic_has_two_distinct_real_roots_l108_108918

theorem quadratic_has_two_distinct_real_roots (k : ℝ) (h1 : 4 + 4 * k > 0) (h2 : k ≠ 0) :
  k > -1 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l108_108918


namespace girls_not_join_field_trip_l108_108776

theorem girls_not_join_field_trip (total_students : ℕ) (number_of_boys : ℕ) (number_on_trip : ℕ)
  (h_total : total_students = 18)
  (h_boys : number_of_boys = 8)
  (h_equal : number_on_trip = number_of_boys) :
  total_students - number_of_boys - number_on_trip = 2 := by
sorry

end girls_not_join_field_trip_l108_108776


namespace exists_rational_non_integer_a_not_exists_rational_non_integer_b_l108_108538

-- Define rational non-integer numbers
def is_rational_non_integer (x : ℚ) : Prop := ¬(∃ (z : ℤ), x = z)

-- (a) Proof for existance of rational non-integer numbers y and x such that 19x + 8y, 8x + 3y are integers
theorem exists_rational_non_integer_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ a b : ℤ, 19 * x + 8 * y = a ∧ 8 * x + 3 * y = b) :=
sorry

-- (b) Proof for non-existance of rational non-integer numbers y and x such that 19x² + 8y², 8x² + 3y² are integers
theorem not_exists_rational_non_integer_b :
  ¬ ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ m n : ℤ, 19 * x^2 + 8 * y^2 = m ∧ 8 * x^2 + 3 * y^2 = n) :=
sorry

end exists_rational_non_integer_a_not_exists_rational_non_integer_b_l108_108538


namespace lowest_possible_sale_price_percentage_l108_108452

def list_price : ℝ := 80
def initial_discount : ℝ := 0.5
def additional_discount : ℝ := 0.2

theorem lowest_possible_sale_price_percentage 
  (list_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) :
  ( (list_price - (list_price * initial_discount)) - (list_price * additional_discount) ) / list_price * 100 = 30 :=
by
  sorry

end lowest_possible_sale_price_percentage_l108_108452


namespace sum_of_distances_eq_l108_108426

theorem sum_of_distances_eq (a b : ℝ) : 
  (∑ x in {(x : ℝ × ℝ) | x ∈ [a, 0] ∧ x ∈ [0, a]}, 
     ∑ y in {(y : ℝ × ℝ) | y ∈ [b, 0] ∧ y ∈ [0, b]}, 
       dist x y ^ 2) 
  = 3 * a^2 + 4 * b^2 :=
by sorry

end sum_of_distances_eq_l108_108426


namespace stratified_sampling_allocation_allocation_is_stratified_sampling_l108_108662

structure VillageTax :=
  (north : ℕ)
  (west : ℕ)
  (south : ℕ)
  (total_conscripts : ℕ)

def total_tax (v : VillageTax) : ℕ :=
  v.north + v.west + v.south

def proportion (part whole : ℕ) : ℚ :=
  part / whole

def conscripts_allocation (v : VillageTax) : (ℚ × ℚ × ℚ) :=
  let total := total_tax v
  (proportion v.north total * v.total_conscripts,
   proportion v.west total * v.total_conscripts,
   proportion v.south total * v.total_conscripts)

theorem stratified_sampling_allocation (v : VillageTax) :
  conscripts_allocation v =
  (proportion v.north (total_tax v) * v.total_conscripts,
   proportion v.west (total_tax v) * v.total_conscripts,
   proportion v.south (total_tax v) * v.total_conscripts) :=
sorry

def is_stratified_sampling (v : VillageTax) : Prop :=
  stratified_sampling_allocation v

def given_village_taxes : VillageTax :=
{ north := 8758, west := 7236, south := 8356, total_conscripts := 378 }

theorem allocation_is_stratified_sampling :
  is_stratified_sampling given_village_taxes :=
sorry

end stratified_sampling_allocation_allocation_is_stratified_sampling_l108_108662


namespace question1_question2_l108_108622

section problem1

variable (a b : ℝ)

theorem question1 (h1 : a = 1) (h2 : b = 2) : 
  ∀ x : ℝ, abs (2 * x + 1) + abs (3 * x - 2) ≤ 5 ↔ 
  (-4 / 5 ≤ x ∧ x ≤ 6 / 5) :=
sorry

end problem1

section problem2

theorem question2 :
  (∀ x : ℝ, abs (x - 1) + abs (x + 2) ≥ m^2 - 3 * m + 5) → 
  ∃ (m : ℝ), m ≤ 2 :=
sorry

end problem2

end question1_question2_l108_108622


namespace range_of_a_l108_108699

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then 2^(|x - a|) else x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ f 1 a) ↔ (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l108_108699


namespace odd_function_f_expression_l108_108690

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 + x else -x^2 + x

theorem odd_function_f_expression (x : ℝ) (h : f(2) = 6) (h_odd : ∀ x, f(-x) = -f(x)) :
  (x < 0 → f(x) = x^2 + x) →
  (x ≥ 0 → f(x) = -x^2 + x) :=
begin
  intro hx,
  split,
  { -- Case x < 0
    intro hx_neg,
    simp [hx_neg] at hx,
    exact hx_neg },
  { -- Case x ≥ 0
    intro hx_nonneg,
    have h_need : f(-x) = -f(x), from h_odd x,
    simp only [hx_nonneg, h_need],
    sorry
    } 
end

end odd_function_f_expression_l108_108690


namespace more_children_than_adults_l108_108471

-- Definitions and conditions
def total_members : ℕ := 120
def percentage_adults : ℝ := 0.40

-- Definitions for calculations of adults and children
def number_of_adults : ℕ := (percentage_adults * total_members).to_nat
def number_of_children : ℕ := (total_members - number_of_adults)

-- Theorem: More children than adults
theorem more_children_than_adults : number_of_children - number_of_adults = 24 := by
  sorry

end more_children_than_adults_l108_108471


namespace circle_equation_l108_108587

-- Define the center of the circle
def center : ℝ × ℝ := (2, -2)

-- Define the minimum distance from a point on the circle to the y-axis
def min_distance_to_y_axis : ℝ := 1

-- Define the radius based on the given minimum distance to the y-axis
def radius := min_distance_to_y_axis

-- Prove that the standard equation of the circle is as given
theorem circle_equation : 
    (radius = 1) → 
    ((center = (2, -2)) → 
    (∀ x y : ℝ, (x - 2)^2 + (y + 2)^2 = radius^2)) :=
by
  intros hr hcenter
  rw [←hcenter]
  simp [radius, hr]
  sorry

end circle_equation_l108_108587


namespace find_p_series_l108_108912

theorem find_p_series (p : ℝ) (h : 5 + (5 + p) / 5 + (5 + 2 * p) / 5^2 + (5 + 3 * p) / 5^3 + ∑' (n : ℕ), (5 + (n + 1) * p) / 5^(n + 1) = 10) : p = 16 :=
sorry

end find_p_series_l108_108912


namespace exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l108_108532

structure RatNonIntPair (x y : ℚ) :=
  (x_rational : x.is_rational)
  (x_not_integer : x.num ≠ x.denom)
  (y_rational : y.is_rational)
  (y_not_integer : y.num ≠ y.denom)

theorem exists_rat_nonint_sol_a :
  ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * x + int 8 * y).denom = 1 ∧ (int 8 * x + int 3 * y).denom = 1 := sorry

theorem no_exists_rat_nonint_sol_b :
  ¬ ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * (x^2) + int 8 * (y^2)).denom = 1 ∧ (int 8 * (x^2) + int 3 * (y^2)).denom = 1 := sorry

end exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l108_108532


namespace perimeter_of_grid_l108_108485

theorem perimeter_of_grid (area: ℕ) (side_length: ℕ) (perimeter: ℕ) 
  (h1: area = 144) 
  (h2: 4 * side_length * side_length = area) 
  (h3: perimeter = 4 * 2 * side_length) : 
  perimeter = 48 :=
by
  sorry

end perimeter_of_grid_l108_108485


namespace investment_period_ratio_l108_108465

theorem investment_period_ratio (I_B : ℝ) (T_B T_A : ℝ) 
  (I_A_eq : I_A = 3 * I_B) 
  (T_A_eq : T_A = k * T_B)
  (profit_B : ℝ) : 
  profit_B = 4000 → 
  I_A = 3 * I_B → 
  T_A = k * T_B → 
  Profit_A + profit_B = 28000 → 
  (Profit_A / profit_B) = 6 → 
  (3 * k = 6) →
  (T_A / T_B = 2) :=
begin
  sorry
end

end investment_period_ratio_l108_108465


namespace ellipse_parabola_isosceles_right_triangle_l108_108954

theorem ellipse_parabola_isosceles_right_triangle (x y p : ℝ) (F1 F2 F E A B : ℝ × ℝ)
  (h_ellipse : ∀ x y, x^2 / 4 + y^2 / 3 = 1)
  (h_parabola : ∀ x y, x^2 = 2 * p * y)
  (h_foci : F1 = (-1, 0) ∧ F2 = (1, 0))
  (h_triangle : ∃ F, F = (0, 1) ∧ F1.1 = -1 ∧ F1.2 = 0 ∧ F2.1 = 1 ∧ F2.2 = 0 ∧ p = 2)
  (h_line : E = (-2, 0))
  (h_tangents : ∃ l₁ l₂ k, (l₁ ∈ tangent A) ∧ (l₂ ∈ tangent B) ∧ l₁.l_perp l₂ ∧ k = 1/2) :
  p = 2 ∧ ∃ k, k = 1/2 ∧ ∀ x y, y = k * (x + 2) → y = 1/2 * (x + 2) := by 
  sorry

end ellipse_parabola_isosceles_right_triangle_l108_108954


namespace max_ab_l108_108203

def f (a x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - 0.5 * x^2
def h (x : ℝ) : ℝ := x^2 - x^2 * Real.log x

theorem max_ab (a b : ℝ) (h₁ : 0 < a) (h₂ : ∀ x : ℝ, 0 < x → f a x ≥ -0.5 * x ^ 2 + a * x + b) : ab ≤ e / 2 :=
by
  sorry

end max_ab_l108_108203


namespace games_in_first_part_l108_108894

theorem games_in_first_part {G x : ℕ} (hG : G = 125)
  (h1 : 0.75 * x = (3 / 4) * x) 
  (h2 : 0.5 * (G - x) = (1 / 2) * (G - x))
  (h3 : (0.75 * x + 0.5 * (G - x)) / G = 0.7) :
  x = 100 := by
  sorry

end games_in_first_part_l108_108894


namespace jerry_average_increase_l108_108291

-- Definitions of conditions
def first_three_tests_average (avg : ℕ) : Prop := avg = 85
def fourth_test_score (score : ℕ) : Prop := score = 97
def desired_average_increase (increase : ℕ) : Prop := increase = 3

-- The theorem to prove
theorem jerry_average_increase
  (first_avg first_avg_value : ℕ)
  (fourth_score fourth_score_value : ℕ)
  (increase_points : ℕ)
  (h1 : first_three_tests_average first_avg)
  (h2 : fourth_test_score fourth_score)
  (h3 : desired_average_increase increase_points) :
  fourth_score = 97 → (first_avg + fourth_score) / 4 = 88 → increase_points = 3 :=
by
  intros _ _
  sorry

end jerry_average_increase_l108_108291


namespace tangent_line_at_origin_l108_108613

theorem tangent_line_at_origin (a : ℝ) (f : ℝ → ℝ)
  (h₀ : f x = x^3 + a * x^2 + (a - 4) * x)
  (h₁ : ∀ x : ℝ, f'(-x) = f'(x)) :
  ∀ x : ℝ, f(0) = f'(0) * x :=
by
  sorry

end tangent_line_at_origin_l108_108613


namespace probability_sum_is_prime_l108_108028

open_locale classical

-- Definitions for the sectors of Spinner 1 and Spinner 2
def spinner1_sectors := {2, 3, 4}
def spinner2_sectors := {1, 3, 5}

-- Function to determine if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The set of prime sums from the possible outcomes of the spinners
def prime_sums (s1 s2 : set ℕ) : set ℕ :=
  { x | x ∈ {a + b | a ∈ s1, b ∈ s2} ∧ is_prime x }

-- The probability of landing on a prime sum
def probability_of_prime_sum : ℚ :=
  ∑ x in prime_sums spinner1_sectors spinner2_sectors, 1 / (spinner1_sectors.card * spinner2_sectors.card)

theorem probability_sum_is_prime : probability_of_prime_sum = 5 / 9 :=
by sorry

end probability_sum_is_prime_l108_108028


namespace sum_of_arithmetic_sequence_l108_108176

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (a_6 : a 6 = 2) : 
  (11 * (a 1 + (a 1 + 10 * ((a 6 - a 1) / 5))) / 2) = 22 :=
by
  sorry

end sum_of_arithmetic_sequence_l108_108176


namespace polynomial_divisibility_l108_108131

theorem polynomial_divisibility : ∃ m n : ℝ, (m = 8 ∧ n = -24) ∧ (∃ Q : ℝ[X], polynomial.div X^4 - 3*X^3 + m*X + n (X^2 - 2*X + 4) = polynomial.C 0) :=
by
  sorry

end polynomial_divisibility_l108_108131


namespace total_points_zach_ben_l108_108802

theorem total_points_zach_ben (zach_points ben_points : ℝ) (h1 : zach_points = 42.0) (h2 : ben_points = 21.0) : zach_points + ben_points = 63.0 :=
by
  sorry

end total_points_zach_ben_l108_108802


namespace interval_of_monotonic_decrease_range_of_k_l108_108981
open Real

noncomputable def f (x : ℝ) : ℝ := 
  let m := (sqrt 3 * sin (x / 4), 1)
  let n := (cos (x / 4), cos (x / 2))
  m.1 * n.1 + m.2 * n.2 -- vector dot product

-- Prove the interval of monotonic decrease for f(x)
theorem interval_of_monotonic_decrease (k : ℤ) : 
  4 * k * π + 2 * π / 3 ≤ x ∧ x ≤ 4 * k * π + 8 * π / 3 → f x = sin (x / 2 + π / 6) + 1 / 2 :=
sorry

-- Prove the range of k such that the zero condition is satisfied for g(x) - k
theorem range_of_k (k : ℝ) :
  0 ≤ k ∧ k ≤ 3 / 2 → ∃ x ∈ [0, 7 * π / 3], (sin (x / 2 - π / 6) + 1 / 2) - k = 0 :=
sorry

end interval_of_monotonic_decrease_range_of_k_l108_108981


namespace find_chocolate_cakes_l108_108847

variable (C : ℕ)
variable (h1 : 12 * C + 6 * 22 = 168)

theorem find_chocolate_cakes : C = 3 :=
by
  -- this is the proof placeholder
  sorry

end find_chocolate_cakes_l108_108847


namespace triangle_area_eq_six_l108_108125

theorem triangle_area_eq_six
  (Q : ℝ)
  (line : 12 * x - 4 * y + (Q - 305) = 0) :
  let x_intercept := - (Q - 305) / 12,
      y_intercept := (Q - 305) / 4,
      R := (1 / 2) * x_intercept * y_intercept
  in R = 6 :=
by
  sorry

end triangle_area_eq_six_l108_108125


namespace left_square_side_length_l108_108756

theorem left_square_side_length (x : ℕ) (h1 : ∀ y : ℕ, y = x + 17)
                                (h2 : ∀ z : ℕ, z = x + 11)
                                (h3 : 3 * x + 28 = 52) : x = 8 :=
by
  sorry

end left_square_side_length_l108_108756


namespace length_of_vector_sum_l108_108255

variable {A B C D : Point}

theorem length_of_vector_sum (h1 : is_rectangle A B C D)
                            (h2 : |vector AB| = sqrt 3)
                            (h3 : |vector BC| = 1) :
                            |vector (AB + BC)| = 2 := by
  sorry

end length_of_vector_sum_l108_108255


namespace sum_of_interior_angles_pentagon_l108_108400

theorem sum_of_interior_angles_pentagon : (5 - 2) * 180 = 540 := by
  sorry

end sum_of_interior_angles_pentagon_l108_108400


namespace number_of_apples_l108_108113

theorem number_of_apples (A : ℝ) (h : 0.75 * A * 0.5 + 0.25 * A * 0.1 = 40) : A = 100 :=
by
  sorry

end number_of_apples_l108_108113


namespace problem_statement_l108_108225

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x : ℝ, f x = x^2 + x + 1) 
  (h_a : a > 0) (h_b : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 3 :=
sorry

end problem_statement_l108_108225


namespace volume_correct_l108_108567

-- Define the conditions
variable (x y z : ℝ)
def condition1 : Prop := x ≥ 0
def condition2 : Prop := y ≥ 0
def condition3 : Prop := z ≥ 0
def inequality : Prop := |x + y + z| + |x + y - 2 * z| ≤ 12

-- Define a volume function for the region of interest
def volume_of_region (x y z : ℝ) : ℝ :=
  if condition1 x && condition2 y && condition3 z && inequality x y z then
    1/2 * 6 * 6 * 3
  else
    0

-- The main theorem statement
theorem volume_correct :
  ∀ x y z, volume_of_region x y z = 54 := sorry

end volume_correct_l108_108567


namespace tic_tac_toe_second_player_draw_l108_108343

def tic_tac_toe_optimal_strategy (player : String) : Prop :=
  player = "crosses" ∨ player = "noughts"

theorem tic_tac_toe_second_player_draw :
  ∀ (first_player second_player : String), 
  first_player = "crosses" → 
  second_player = "noughts" → 
  tic_tac_toe_optimal_strategy first_player → 
  (∀ strategy, strategy first_player → (first_player_wins first_player strategy ∨ draw first_player second_player strategy)) → 
  ¬(∀ strategy, strategy second_player → second_player_wins second_player strategy) :=
by
  intros first_player second_player hc hnc optimal_strat opt_play second_win
  sorry

end tic_tac_toe_second_player_draw_l108_108343


namespace geom_seq_b_value_l108_108774

variable (r : ℝ) (b : ℝ)

-- b is the second term of the geometric sequence with first term 180 and third term 36/25
-- condition 1
def geom_sequence_cond1 := 180 * r = b
-- condition 2
def geom_sequence_cond2 := b * r = 36 / 25

-- Prove b = 16.1 given the conditions
theorem geom_seq_b_value (hb_pos : b > 0) (h1 : geom_sequence_cond1 r b) (h2 : geom_sequence_cond2 r b) : b = 16.1 :=
by sorry

end geom_seq_b_value_l108_108774


namespace find_f_a_plus_1_l108_108200

def f (x : ℝ) : ℝ := x^2 + 1

theorem find_f_a_plus_1 (a : ℝ) : f (a + 1) = a^2 + 2 * a + 2 := by
  sorry

end find_f_a_plus_1_l108_108200


namespace BE_eq_AD_l108_108709

-- Definitions and assumptions
variables {Point : Type} {Line : Type}
variable (M : Point) (E D A B C F : Point)
variable (lineCE : Line) (lineMF : Line) (lineDE : Line)

-- Given conditions
def line_parallel (l1 l2 : Line) : Prop := sorry -- definition of parallel lines
def parallelogram (P1 P2 P3 P4 : Point) : Prop := sorry -- definition of parallelogram
def equal_length (P1 P2 P3 P4 : Point) : Prop := sorry -- definition of equal length segments

-- Assumed conditions
axiom parallel_mf_de : line_parallel M F E D
axiom intersect_ce_mf_at_f : M ∈ lineCE → F ∈ lineMF
axiom is_parallelogram_mdef : parallelogram M D E F
axiom length_mf_de : equal_length M F D E

-- Goal
theorem BE_eq_AD : equal_length B E A D :=
sorry

end BE_eq_AD_l108_108709


namespace jacket_total_cost_l108_108490

theorem jacket_total_cost :
  let original_price := 120
  let first_discount_rate := 0.25
  let coupon_discount := 10
  let sales_tax_rate := 0.10
  let discounted_price := original_price * (1 - first_discount_rate)
  let after_coupon_price := discounted_price - coupon_discount
  let total_cost := after_coupon_price * (1 + sales_tax_rate)
  total_cost = 88 := by
  let original_price := 120
  let first_discount_rate := 0.25
  let coupon_discount := 10
  let sales_tax_rate := 0.10
  let discounted_price := original_price * (1 - first_discount_rate)
  let after_coupon_price := discounted_price - coupon_discount
  let total_cost := after_coupon_price * (1 + sales_tax_rate)
  show total_cost = 88 by
  sorry

end jacket_total_cost_l108_108490


namespace perimeter_square_Y_result_l108_108358

noncomputable def square_X_perimeter : ℝ := 32
noncomputable def square_Y_area_divisor : ℝ := 3

def side_length_square_X (perimeter : ℝ) : ℝ := perimeter / 4
def area_square_X (side_length : ℝ) : ℝ := side_length ^ 2
def area_square_Y (area_X : ℝ) (divisor : ℝ) : ℝ := area_X / divisor
def side_length_square_Y (area_Y : ℝ) : ℝ := (Real.sqrt area_Y)
def perimeter_square_Y (side_length_Y : ℝ) : ℝ := 4 * side_length_Y

theorem perimeter_square_Y_result :
  perimeter_square_Y (side_length_square_Y (area_square_Y (area_square_X (side_length_square_X square_X_perimeter)) square_Y_area_divisor)) =
  32 * Real.sqrt 3 / 3 :=
sorry

end perimeter_square_Y_result_l108_108358


namespace set_B_right_triangle_proof_l108_108445

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem set_B_right_triangle_proof (a b c : ℝ) (h₁ : {a, b, c} = {3, 4, 5}) : is_right_triangle a b c :=
  sorry

end set_B_right_triangle_proof_l108_108445


namespace arithmetic_sequence_middle_term_l108_108271

theorem arithmetic_sequence_middle_term 
  (x y z : ℕ) 
  (h1 : ∀ i, (∀ j, 23 + (i - 0) * d = x)
  (h2 : ∀ i, (∀ j, y = 23 + (j - 1) * (23 + d)) 
  (h3 : ∀ i, (∀ j, z = 23 + (j - 2 * d)) 
  (h4 : ∀ i, 47 = 23 + (5 - 1) * d )
   : y = (23 + 47) / 2 :=
by 
  sorry

end arithmetic_sequence_middle_term_l108_108271


namespace convergence_of_frac_parts_l108_108882

noncomputable def a_seq : ℕ → ℝ
| 0     := 2022
| (n+1) := a_seq n + Real.exp (-a_seq n)

def frac_part (x : ℝ) : ℝ := x - Real.floor x

theorem convergence_of_frac_parts : ∃ r : ℝ, r > 0 ∧ ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (frac_part (r * a_seq (10^n)) - L) < ε := sorry

end convergence_of_frac_parts_l108_108882


namespace cost_price_of_each_watch_l108_108022

-- Define the given conditions.
def sold_at_loss (C : ℝ) := 0.925 * C
def total_transaction_price (C : ℝ) := 3 * C * 1.053
def sold_for_more (C : ℝ) := 0.925 * C + 265

-- State the theorem to prove the cost price of each watch.
theorem cost_price_of_each_watch (C : ℝ) :
  3 * sold_for_more C = total_transaction_price C → C = 2070.31 :=
by
  intros h
  sorry

end cost_price_of_each_watch_l108_108022


namespace max_distance_intersection_l108_108190

theorem max_distance_intersection :
  let P := (1, 3)
  let L1 := (2:ℝ, 1:ℝ)
  let L2 := (-1:ℝ, 4:ℝ)
  let L3 := (m:ℝ) ~> (P, (3:ℝ, -2*(m+mx), 1: (ℝ))
  ∀ m: ℝ, max_distance P (x + 2)m - y + 1 = 0
(max_distance P (mx + 2m - y + 1)) = √13
 sorry

end max_distance_intersection_l108_108190


namespace sum_of_solutions_l108_108439

theorem sum_of_solutions (n : ℕ) (h1 : ∀ n, n >= 1 → math.lcm n 120 = Int.gcd n 120 + 300 → n = 180) : n = 180 :=
by
  sorry

end sum_of_solutions_l108_108439


namespace direction_vector_reflection_over_line_l108_108008

open Matrix

theorem direction_vector_reflection_over_line (x y : ℚ) (a b : ℤ) (hmatrix : 
  (λ u v : Fin 2, if (u, v) = (0, 0) then 3/5 else if (u, v) = (0, 1) then 4/5 else if (u, v) = (1, 0) then 4/5 else -3/5) 
  ⬝ (λ u v : Fin 1, if (u, v) = (0, 0) then x else y) = 
  (λ u v : Fin 1, if (u, v) = (0, 0) then x else y)) 
  (hdirection : a ≠ 0 ∧ Int.gcd a b = 1 ∧ a > 0)
  : a = 4 ∧ b = -3 :=
sorry

end direction_vector_reflection_over_line_l108_108008


namespace reporter_arrangement_l108_108248

-- Definitions as per the conditions
def numA : ℕ := 5
def numB : ℕ := 5
def selectTotal : ℕ := 4

-- Proposition of the proof problem
theorem reporter_arrangement : 
  ∀ {A B : Type} (numA numB selectTotal : ℕ) 
    (ha : set A) (hb : set B) 
    (hA_card : Fintype.card ha = numA) 
    (hB_card : Fintype.card hb = numB), 
    numA = 5 ∧ numB = 5 ∧ selectTotal = 4 ∧
    (∃ x : set A, x ⊆ ha ∧ Fintype.card x ≥ 1) ∧
    (∃ y : set B, y ⊆ hb ∧ Fintype.card y ≥ 1) →
    -- include that reporters from A cannot ask consecutively as a complex condition: sorry for the lack of details
    ∃ arrangement : list (A ⊕ B),
      Fintype.card arrangement = 2400 :=
sorry -- proof not required

end reporter_arrangement_l108_108248


namespace min_value_z_l108_108039

theorem min_value_z : ∃ (min_z : ℝ), min_z = 24.1 ∧ 
  ∀ (x y : ℝ), (3 * x ^ 2 + 4 * y ^ 2 + 8 * x - 6 * y + 30) ≥ min_z :=
sorry

end min_value_z_l108_108039


namespace quadratic_complete_square_l108_108760

theorem quadratic_complete_square (b c : ℝ) (h : ∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) : b + c = -106 :=
by
  sorry

end quadratic_complete_square_l108_108760


namespace L_shapes_same_orientation_after_2005_subdivs_l108_108853

-- We will define the problem in terms of sequences and their relationships.

def a (n : ℕ) : ℕ := match n with
  | 0 => 1
  | _ => 2 * a (n - 1) + b (n - 1) + d (n - 1)
and b (n :ℕ) : ℕ := match n with
  | 0 => 0
  | _ => a (n - 1) + 2 * b (n - 1) + c (n - 1)
and c (n : ℕ) : ℕ := match n with
  | 0 => 0
  | _ => b (n - 1) + 2 * c (n - 1) + d (n - 1)
and d (n : ℕ) : ℕ := match n with
  | 0 => 0
  | _ => c (n - 1) + 2 * d (n - 1) + a (n - 1)

-- The theorem to be proved.
theorem L_shapes_same_orientation_after_2005_subdivs :
  a 2005 = 4^2004 + 2^2004 := sorry

end L_shapes_same_orientation_after_2005_subdivs_l108_108853


namespace average_age_of_cricket_team_l108_108060

theorem average_age_of_cricket_team
  (A : ℝ)
  (captain_age : ℝ) (wicket_keeper_age : ℝ)
  (team_size : ℕ) (remaining_players : ℕ)
  (captain_age_eq : captain_age = 24)
  (wicket_keeper_age_eq : wicket_keeper_age = 27)
  (remaining_players_eq : remaining_players = team_size - 2)
  (average_age_condition : (team_size * A - (captain_age + wicket_keeper_age)) = remaining_players * (A - 1)) : 
  A = 21 := by
  sorry

end average_age_of_cricket_team_l108_108060


namespace Andriyka_total_distance_AC_l108_108767

-- Definitions of given conditions
variable (ABCD : Type) [Rectangle ABCD]
variable (A B C D : Point ABCD) -- Points A, B, C, and D on the rectangle ABCD
variable (AC BD : Diagonal ABCD) -- Diagonals AC and BD
variable (distance : Real)

-- Given conditions
variable (angle_AC_BD : ∠ AC BD = 60) -- Angle between the diagonals is 60 degrees
variable (AB_greater_BC : distance AB > distance BC) -- AB is greater than BC
variable (ACBD_route_dist : Real) -- Distance of one route A-C-B-D-A
variable (ADA_route_dist : Real) -- Distance of one route A-D-A
variable (total_distance : Real := 4.5) -- Total distance Andriyka traveled

-- Distances covered by Andriyka
variable (ACBD_times : Nat := 10) 
variable (ADA_times : Nat := 15) 

-- The proof statement
theorem Andriyka_total_distance_AC (AC_val : Real) (s : Real) :
  (ACBD_times * ACBD_route_dist + ADA_times * ADA_route_dist = total_distance) →
  (ACBD_route_dist = 4 * AC_val) →
  (ADA_route_dist = 2 * s) →
  (AC_val = 2 * s) →
  AC_val = 0.0818 :=
by sorry

end Andriyka_total_distance_AC_l108_108767


namespace first_motorcyclist_distance_to_B_l108_108005

variable (v a T : ℝ)
-- Conditions
axiom (track_length : 1)  -- The length of the circular race track is 1 km.
axiom (start_at_A : true) -- The motorcyclists start simultaneously from point A.

-- One motorcyclist travels at a constant speed v km/h.
axiom (first_motorcyclist : v > 0)

-- The other motorcyclist accelerates uniformly with acceleration a km/h^2.
axiom (second_motorcyclist : a = 2 * v^2)

-- They first meet at point B on the track, and then for the second time at point A.
axiom (meet_at_B : v * T + v^2 * T^2 = 1)

-- Prove that the distance the first motorcyclist travels to point B is (√5 - 1)/2 km.
theorem first_motorcyclist_distance_to_B : v * T = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end first_motorcyclist_distance_to_B_l108_108005


namespace construct_quadrilateral_l108_108886

variables (α a b c d : ℝ)

-- α represents the sum of angles B and D
-- a represents the length of AB
-- b represents the length of BC
-- c represents the length of CD
-- d represents the length of DA

theorem construct_quadrilateral (α a b c d : ℝ) : 
  ∃ A B C D : ℝ × ℝ, 
    dist A B = a ∧ 
    dist B C = b ∧ 
    dist C D = c ∧ 
    dist D A = d ∧ 
    ∃ β γ δ, β + δ = α := 
sorry

end construct_quadrilateral_l108_108886


namespace minimum_value_of_quadratic_function_l108_108626

def quadratic_function (x : ℝ) : ℝ := x^2 + 8 * x + 12

theorem minimum_value_of_quadratic_function : ∃ x : ℝ, is_min_on (quadratic_function) {x} (-4) :=
by
  use -4
  sorry

end minimum_value_of_quadratic_function_l108_108626


namespace incircle_center_in_square_l108_108492

-- Define an acute-angled triangle
structure acute_triangle (A B C : Type) :=
  (is_acute : ∀ (angle : A ∨ B ∨ C), angle < 90)

-- Define the square inscribed in an acute-angled triangle
structure inscribed_square (A B C : Type) :=
  (square : Type) -- Details of the square are abstracted
  (vertices_on_sides : ∃ (D E F G : square), D ∈ side_AB ∧ E ∈ side_BC ∧ F ∈ side_CA ∧ G ∈ side_CA)

-- Prove that for an acute-angled triangle with an inscribed square, the center of the incircle of the triangle lies within the square
theorem incircle_center_in_square (A B C : Type) [acute_triangle A B C] [inscribed_square A B C] (O : Type) :
  ∃ (O : center_incircle A B C), O ∈ inscribed_square.square :=
sorry

end incircle_center_in_square_l108_108492


namespace quadrilateral_concyclic_l108_108933

theorem quadrilateral_concyclic (O O1 O2 O3 O4 : Type)
  (rO r1 r2 r3 r4 : ℝ)
  (A B C D : Type)
  (alpha beta gamma delta : ℝ)
  (a b c d l12 l23 l34 l41 l13 l24 : ℝ)
  (h1 : a = rO - r1)
  (h2 : b = rO - r2)
  (h3 : c = rO - r3)
  (h4 : d = rO - r4)
  (h5 : l12 = 2 * real.sqrt(ab) * real.sin(alpha / 2))
  (h6 : l23 = 2 * real.sqrt(bc) * real.sin(beta / 2))
  (h7 : l34 = 2 * real.sqrt(cd) * real.sin(gamma / 2))
  (h8 : l41 = 2 * real.sqrt(da) * real.sin(delta / 2))
  (h9 : l13 = 2 * real.sqrt(ac) * real.sin((alpha + beta) / 2))
  (h10 : l24 = 2 * real.sqrt(bd) * real.sin((beta + gamma) / 2)) :
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A) →
  (l12 * l34 + l23 * l41 = l13 * l24) →
  is_concyclic [A, B, C, D] :=
sorry

end quadrilateral_concyclic_l108_108933


namespace solve_inequality_l108_108357

def solve_quadratic_inequality (a : ℝ) : set ℝ :=
  if a > 1 then
    set.univ
  else if a = 1 then
    {x : ℝ | x ≠ -1}
  else
    {x : ℝ | x > -1 + real.sqrt (1 - a) ∨ x < -1 - real.sqrt (1 - a)}

theorem solve_inequality (a : ℝ) :
  ∀ x : ℝ, (x^2 + 2*x + a > 0) ↔ x ∈ solve_quadratic_inequality a := 
by
  sorry

end solve_inequality_l108_108357


namespace move_point_right_3_units_l108_108737

theorem move_point_right_3_units (x y : ℤ) (hx : x = 2) (hy : y = -1) :
  (x + 3, y) = (5, -1) :=
by
  sorry

end move_point_right_3_units_l108_108737


namespace sum_of_first_200_terms_l108_108192

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) - a n = d

def sum_of_first (s : ℕ → ℝ) (n : ℕ) (sum_value : ℝ) : Prop :=
  ∑ i in finset.range n, s i = sum_value

-- The given sequence and its arithmetic property in terms of logarithms
def log_sequence (x : ℕ → ℝ) : ℕ → ℝ := λ n, real.log x n / real.log 2

-- Problem Statement
theorem sum_of_first_200_terms 
  (x : ℕ → ℝ)
  (h1 : is_arithmetic_sequence (log_sequence x) 1)
  (h2 : sum_of_first x 100 100) : 
  sum_of_first x 200 (100 * (1 + 2^100)) :=
sorry

end sum_of_first_200_terms_l108_108192


namespace find_AC_angle_and_area_l108_108819

variables (A B C D P Q K T : Type)
variables (h1 : ∃ P, ∃ Q, intersects (ray A B) (ray D C) P ∧ intersects (ray B C) (ray A D) Q)
variables (h2 : similar_triangle ADP QAB)
variables (h3 : cyclic_quadrilateral ABCD)
variables (h4 : radius_circle (inscribed_circle ABCD) = 4)
variables (h5 : tangent_point_circle (inscribed_circle ABC) AC K)
variables (h6 : tangent_point_circle (inscribed_circle ACD) AC T)
variables (h7 : ratio_segments CK KT TA = 3 1 4) 

theorem find_AC : AC = 8 := sorry

theorem angle_and_area : anglem DAC = 45 ∧ area_quadrilateral ABCD = 31 := sorry

end find_AC_angle_and_area_l108_108819


namespace interval_of_monotonic_decrease_l108_108748

theorem interval_of_monotonic_decrease (a : ℝ) (x : ℝ) :
  (∃ (a : ℝ), a = -1) ∧ (y = x^a) ∧ (y = 2) ∧ (y = 1/2) → (Ioo (-∞) 0 ∪ Ioo 0 +∞) :=
begin
  sorry,
end

end interval_of_monotonic_decrease_l108_108748


namespace unique_magic_hexagon_l108_108386

/-- 
A mathematical problem stating that for a given set of numbers from 1 to 19, 
there exists a unique arrangement in a hexagonal grid such that the sum of numbers 
along each of the 12 lines equals 23.
-/
theorem unique_magic_hexagon {arrangement : ℕ → ℕ} 
  (h_range : ∀ i, 1 ≤ arrangement i ∧ arrangement i ≤ 19)
  (h_unique : function.injective arrangement)
  (h_sum : ∀ lines, (∑ i in lines, arrangement i) = 23) :
  ∃! arrangement, (∀ lines, (∑ i in lines, arrangement i) = 23) :=
sorry

end unique_magic_hexagon_l108_108386


namespace polygon_sides_l108_108773

theorem polygon_sides (sum_of_interior_angles : ℝ) (x : ℝ) (h : sum_of_interior_angles = 1080) : x = 8 :=
by
  sorry

end polygon_sides_l108_108773


namespace age_relation_l108_108078

theorem age_relation (S M D Y : ℝ)
  (h1 : M = S + 37)
  (h2 : M + 2 = 2 * (S + 2))
  (h3 : D = S - 4)
  (h4 : M + Y = 3 * (D + Y))
  : Y = -10.5 :=
by
  sorry

end age_relation_l108_108078


namespace concyclic_tangency_points_l108_108735

theorem concyclic_tangency_points
  (R1 R2 : Circle)
  (n : ℕ) (n_pos : 1 < n)
  (S : Fin n → Circle)
  (A : Fin (n - 1) → Point)
  (tangent_to_R1 : ∀ (i : Fin n), Tangent (S i) R1)
  (tangent_to_R2 : ∀ (i : Fin n), Tangent (S i) R2)
  (tangent_to_each_other : ∀ (i : Fin (n - 1)), TangentAt (S i) (S (i + 1)) (A i)) :
  Concyclic (A) :=
by
  sorry

end concyclic_tangency_points_l108_108735


namespace transformed_function_l108_108964

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + (Real.pi / 6))

theorem transformed_function :
  ∀ x, g(x) = 2 * Real.sin (2 * x + (Real.pi / 6)) :=
by
  sorry

end transformed_function_l108_108964


namespace holiday_customers_l108_108824

-- Define the normal rate of customers entering the store (175 people/hour)
def normal_rate : ℕ := 175

-- Define the holiday rate of customers entering the store
def holiday_rate : ℕ := 2 * normal_rate

-- Define the duration for which we are calculating the total number of customers (8 hours)
def duration : ℕ := 8

-- Define the correct total number of customers (2800 people)
def correct_total_customers : ℕ := 2800

-- The theorem that asserts the total number of customers in 8 hours during the holiday season is 2800
theorem holiday_customers : holiday_rate * duration = correct_total_customers := by
  sorry

end holiday_customers_l108_108824


namespace angle_BDC_eq_88_l108_108239

-- Define the problem scenario
variable (A B C : ℝ)
variable (α : ℝ)
variable (B1 B2 B3 C1 C2 C3 : ℝ)

-- Conditions provided
axiom angle_A_eq_42 : α = 42
axiom trisectors_ABC : B = B1 + B2 + B3 ∧ C = C1 + C2 + C3
axiom trisectors_eq : B1 = B2 ∧ B2 = B3 ∧ C1 = C2 ∧ C2 = C3
axiom angle_sum_ABC : α + B + C = 180

-- Proving the measure of ∠BDC
theorem angle_BDC_eq_88 :
  α + (B/3) + (C/3) = 88 :=
by
  sorry

end angle_BDC_eq_88_l108_108239


namespace xiao_cong_ways_to_go_up_9_steps_l108_108860

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

-- State the theorem
theorem xiao_cong_ways_to_go_up_9_steps : fib 9 = 55 :=
by
  sorry

end xiao_cong_ways_to_go_up_9_steps_l108_108860


namespace bcm_hens_count_l108_108475

theorem bcm_hens_count (total_chickens : ℕ) (percent_bcm : ℝ) (percent_bcm_hens : ℝ) : ℕ :=
  let total_bcm := total_chickens * percent_bcm
  let bcm_hens := total_bcm * percent_bcm_hens
  bcm_hens

example : bcm_hens_count 100 0.20 0.80 = 16 := by
  sorry

end bcm_hens_count_l108_108475


namespace coin_exchange_l108_108082

theorem coin_exchange :
  ∃ (t1 t2 t5 t10 : ℕ), 
    t2 = (3 / 5) * t1 ∧ 
    t5 = (3 / 5) * t2 ∧ 
    t10 = (3 / 5) * t5 - 7 ∧ 
    (50 ≤ (1 * t1 + 2 * t2 + 5 * t5 + 10 * t10) / 100 ∧ (1 * t1 + 2 * t2 + 5 * t5 + 10 * t10) / 100 ≤ 100) ∧ 
    t1 = 1375 ∧ 
    t2 = 825 ∧ 
    t5 = 495 ∧ 
    t10 = 290 :=
by 
  existsi [1375, 825, 495, 290]
  split; try {norm_num}; intros; linarith

end coin_exchange_l108_108082


namespace average_of_remaining_numbers_l108_108729

theorem average_of_remaining_numbers (f : Fin 12 → ℕ) (h_avg12 : (∑ i, f i) / 12 = 90)
  (h_remove65 : 65 ∈ multiset.map f (fin.enum 12))
  (h_remove85 : 85 ∈ multiset.map f (fin.enum 12)) :
  (∑ i in multiset.erase (multiset.erase (multiset.map f (fin.enum 12)) 65) 85) / 10 = 93 :=
by
  sorry

end average_of_remaining_numbers_l108_108729


namespace infinite_planes_l108_108681

variable (a b : ℝ^3 → ℝ^3) -- representation of the skew lines

-- Hypothesis stating the conditions
variables (α β : set ℝ^3) [plane α] [plane β]
hypothesis (h_a_in_alpha : ∀ x ∈ a, x ∈ α)
hypothesis (h_b_in_beta : ∀ x ∈ b, x ∈ β)
hypothesis (h_skew : ∀ p ∈ a, ∀ q ∈ b, p ≠ q)
hypothesis (h_angle : angle_between a b = 30 * π / 180)
hypothesis (h_perpendicular : α ⊥ β)

-- The statement to prove
theorem infinite_planes (h_a_in_alpha : ∀ x ∈ a, x ∈ α) 
                        (h_b_in_beta : ∀ x ∈ b, x ∈ β) 
                        (h_skew : ∀ p ∈ a, ∀ q ∈ b, p ≠ q) 
                        (h_angle : angle_between a b = 30 * π / 180) 
                        (h_perpendicular : α ⊥ β) : 
                        ∃ (As : ℕ → set ℝ^3) (Bs : ℕ → set ℝ^3), 
                        (∀ n : ℕ, plane (As n) ∧ a ⊆ (As n) ∧ plane (Bs n) ∧ b ⊆ (Bs n) ∧ (As n ⊥ Bs n)) :=
by
  sorry

end infinite_planes_l108_108681


namespace calculated_expression_l108_108111

theorem calculated_expression : 
  let expr := ((3 + 3 / 8 : ℚ)^(2 / 3 : ℚ) - (5 + 4 / 9 : ℚ)^(1 / 2 : ℚ) + (0.008 : ℚ)^(2 / 3 : ℚ) / (0.02 : ℚ)^(1 / 2 : ℚ) * (0.32 : ℚ)^(1 / 2 : ℚ)) / (0.0625 : ℚ)^(1 / 4 : ℚ)
  in expr = (23 / 150 : ℚ) :=
by 
  sorry

end calculated_expression_l108_108111


namespace cos_sum_to_product_l108_108379

theorem cos_sum_to_product (a b c d : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h : ∀ x : ℝ, cos (2 * x) + cos (4 * x) + cos (8 * x) + cos (10 * x) = a * cos (b * x) * cos (c * x) * cos (d * x)) :
  a + b + c + d = 14 :=
sorry

end cos_sum_to_product_l108_108379


namespace inequality_proof_l108_108342

theorem inequality_proof (n : ℕ) (x y : Fin n → ℝ) 
  (h1 : ∀ i, 0 < x i) 
  (h2 : ∀ i, 0 < y i) : 
  ∑ i, 1 / (x i * y i) ≥ 4 * (n ^ 2) / ∑ i, (x i + y i) ^ 2 := 
sorry

end inequality_proof_l108_108342


namespace points_robot_can_visit_l108_108486

-- Definitions based on the conditions
def A : (ℤ × ℤ) := (-5, 3)
def B : (ℤ × ℤ) := (4, -3)
def max_path_length : ℤ := 18

-- Statement to prove
theorem points_robot_can_visit : 
  ∃ (points : finset (ℤ × ℤ)), ((∀ p ∈ points, (abs (p.1 + 5) + abs (p.2 - 3) + abs (p.1 - 4) + abs (p.2 + 3) ≤ 18)) 
    ∧ (finset.card points = 90)) := 
sorry

end points_robot_can_visit_l108_108486


namespace coefficient_x3_expansion_sum_l108_108369

theorem coefficient_x3_expansion_sum :
  (∑ k in (finset.range 15).map (λ i, i + 1), (nat.choose (k + 3) 3)) = 1820 :=
  sorry

end coefficient_x3_expansion_sum_l108_108369


namespace solve_quadratic_eq_l108_108356

theorem solve_quadratic_eq (x : ℝ) : x^2 - x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

end solve_quadratic_eq_l108_108356


namespace determinant_roots_cubic_l108_108698

theorem determinant_roots_cubic (x y z s t : ℝ)
  (h : ∀ {u}, u ∈ {x, y, z} → u^3 + s * u^2 + t * u = 0):
  det ![
  ![x, y, z],
  ![y, z, x],
  ![z, x, y]
  ] = 0 := sorry

end determinant_roots_cubic_l108_108698


namespace divisible_by_2_not_by_3_count_l108_108850

theorem divisible_by_2_not_by_3_count : 
  ∃ n : ℕ, n = (finset.filter (λ x, (x % 2 = 0) ∧ (x % 3 ≠ 0)) (finset.range 101)).card ∧ n = 34
:= sorry

end divisible_by_2_not_by_3_count_l108_108850


namespace coat_price_reduction_l108_108390

theorem coat_price_reduction (original_price : ℝ) (reduction_percent : ℝ)
  (price_is_500 : original_price = 500)
  (reduction_is_30 : reduction_percent = 0.30) :
  original_price * reduction_percent = 150 :=
by
  sorry

end coat_price_reduction_l108_108390


namespace noah_uses_36_cups_of_water_l108_108324

theorem noah_uses_36_cups_of_water
  (O : ℕ) (hO : O = 4)
  (S : ℕ) (hS : S = 3 * O)
  (W : ℕ) (hW : W = 3 * S) :
  W = 36 := 
  by sorry

end noah_uses_36_cups_of_water_l108_108324


namespace eccentricity_range_elliptic_curve_l108_108177

-- Assume the conditions
variables {a b x1 y1 x2 y2 : ℝ}
def ellipse_eq1 : Prop := (x1^2 / a^2) + (y1^2 / b^2) = 1
def ellipse_eq2 : Prop := (x2^2 / a^2) + (y2^2 / b^2) = 1
def ellipse_conditions : Prop := a > b ∧ b > 0 ∧ x1 ≠ x2
def points_eq : Prop := (x1 - (a / 4))^2 + y1^2 = (x2 - (a / 4))^2 + y2^2
def eccentricity_range : Prop := 1 / 2 < (Real.sqrt (1 - (b^2 / a^2))) ∧ (Real.sqrt (1 - (b^2 / a^2))) < 1

theorem eccentricity_range_elliptic_curve :
  ellipse_eq1 ∧ ellipse_eq2 ∧ ellipse_conditions ∧ points_eq → eccentricity_range :=
by sorry

end eccentricity_range_elliptic_curve_l108_108177


namespace find_alpha_angle_l108_108258

theorem find_alpha_angle :
  ∃ α : ℝ, (7 * α + 8 * α + 45) = 180 ∧ α = 9 :=
by 
  sorry

end find_alpha_angle_l108_108258


namespace hyperbola_constants_sum_l108_108742

noncomputable def hyperbola_asymptotes_equation (x y : ℝ) : Prop :=
  (y = 2 * x + 5) ∨ (y = -2 * x + 1)

noncomputable def hyperbola_passing_through (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 7)

theorem hyperbola_constants_sum
  (a b h k : ℝ) (ha : a > 0) (hb : b > 0)
  (H1 : ∀ x y : ℝ, hyperbola_asymptotes_equation x y)
  (H2 : hyperbola_passing_through 0 7)
  (H3 : h = -1)
  (H4 : k = 3)
  (H5 : a = 2 * b)
  (H6 : b = Real.sqrt 3) :
  a + h = 2 * Real.sqrt 3 - 1 :=
sorry

end hyperbola_constants_sum_l108_108742


namespace num_passed_candidates_l108_108730

theorem num_passed_candidates
  (total_candidates : ℕ)
  (avg_passed_marks : ℕ)
  (avg_failed_marks : ℕ)
  (overall_avg_marks : ℕ)
  (h1 : total_candidates = 120)
  (h2 : avg_passed_marks = 39)
  (h3 : avg_failed_marks = 15)
  (h4 : overall_avg_marks = 35) :
  ∃ (P : ℕ), P = 100 :=
by
  sorry

end num_passed_candidates_l108_108730


namespace equilateral_intersection_area_l108_108615

noncomputable def area_of_circle (a : ℝ) : ℝ := 
  let R := Real.sqrt (a^2 - 1)
  π * (R^2)

theorem equilateral_intersection_area :
  ∀ (a : ℝ),
  ∃ A B C : ℝ → ℝ,
  (A.1 ^ 2 + A.2 ^ 2 - 2 * a * A.1 - 2 * A.2 + 2 = 0) →
  (B.1 ^ 2 + B.2 ^ 2 - 2 * a * B.1 - 2 * B.2 + 2 = 0) →
  (C = fun x => a * x) →
  (Real.dist A B = Real.dist B C) ∧
  (Real.dist B C = Real.dist C A) ∧
  (Real.dist C A = Real.dist A B) →
  area_of_circle a = 6 * π :=
by
  sorry

end equilateral_intersection_area_l108_108615


namespace part1_part2_l108_108650

noncomputable def triangle_condition_1 (a b c : ℝ) (cosA cosB sinB : ℝ) : Prop :=
a * cosB = (3 * c - b) * cosA ∧ a * sinB = 2 * sqrt 2

noncomputable def triangle_condition_2 (a b c : ℝ) (cosA cosB sinB : ℝ) (area : ℝ) : Prop :=
a * cosB = (3 * c - b) * cosA ∧ a = 2 * sqrt 2 ∧ area = sqrt 2

theorem part1 (a b c cosA cosB sinB : ℝ) (h : triangle_condition_1 a b c cosA cosB sinB) : b = 3 :=
by
  sorry

theorem part2 (a b c cosA cosB sinB area : ℝ) (h : triangle_condition_2 a b c cosA cosB sinB area) : a + b + c = 4 + 2 * sqrt 2 :=
by
  sorry

end part1_part2_l108_108650


namespace arithmetic_sequence_y_l108_108262

theorem arithmetic_sequence_y : 
  ∀ (x y z : ℝ), (23 : ℝ), x, y, z, (47 : ℝ) → 
  (y = (23 + 47) / 2) → y = 35 :=
by
  intro x y z h1
  intro h2
  simp at *
  sorry

end arithmetic_sequence_y_l108_108262


namespace largest_multiple_of_11_less_than_neg_200_l108_108431

theorem largest_multiple_of_11_less_than_neg_200 :
  ∃ x, (x % 11 = 0) ∧ x < -200 ∧ ∀ y, (y % 11 = 0) ∧ y < -200 → y ≤ x :=
begin
  use -209,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros y hy,
    have h : y ≤ -209, by sorry,
    exact h, }
end

end largest_multiple_of_11_less_than_neg_200_l108_108431


namespace range_of_a_l108_108614

noncomputable def f (a : ℝ) (x : ℝ) := x * Real.log x - a * x^2

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 
0 < a ∧ a < 1/2 :=
by
  sorry

end range_of_a_l108_108614


namespace explicit_formula_angle_C_proof_l108_108612

-- Definition of the initial conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def function_f (ω ϕ : ℝ) (x : ℝ) : ℝ := (1/2) * Real.sin (ω * x + ϕ)

-- Condition for highest and lowest distance
axiom distance_PQ (ω : ℝ) : |(Real.pi/ω)| ^ 2 + 1 ^ 2 = 2

-- Definition of sides and angle in triangle ABC
def sides (a b : ℝ) (A : ℝ) : Prop :=
  a = 1 ∧ b = Real.sqrt 2 ∧ (1/2) * Real.cos A = (Real.sqrt 3) / 4

noncomputable def angle_C (a b : ℝ) (A B C : ℝ) : Prop := 
  a = 1 ∧ b = Real.sqrt 2 ∧
  (1/2) * Real.cos A = (Real.sqrt 3) / 4 ∧
  Real.sin B = (Real.sqrt 2) / 2 ∧
  (C = Real.pi - A - B)

-- Proof that the function can be rewritten
theorem explicit_formula (ω ϕ : ℝ) (hϕ : 0 < ϕ ∧ ϕ < Real.pi) :
  is_even (function_f ω ϕ) → function_f ω ϕ = (λ x, (1/2) * Real.cos (Real.pi * x)) := sorry

-- Proof that angle C
theorem angle_C_proof (a b A C : ℝ) :
  sides a b A →
  angle_C a b A (Real.pi / 6) C → (C = 7 * Real.pi / 12 ∨ C = Real.pi / 12) := sorry

end explicit_formula_angle_C_proof_l108_108612


namespace parabola_focus_l108_108141

-- Define the parabola
def parabolaEquation (x y : ℝ) : Prop := y^2 = -6 * x

-- Define the focus
def focus (x y : ℝ) : Prop := x = -3 / 2 ∧ y = 0

-- The proof problem: showing the focus of the given parabola
theorem parabola_focus : ∃ x y : ℝ, parabolaEquation x y ∧ focus x y :=
by
    sorry

end parabola_focus_l108_108141


namespace graph_shift_sin_l108_108780

theorem graph_shift_sin :
  ∀ x : ℝ, sin (2 * x + π / 3) = sin (2 * (x + π / 6)) :=
by
  sorry

end graph_shift_sin_l108_108780


namespace angle_acb_circle_l108_108787

theorem angle_acb_circle (A B C : Point) (circle : Circle) (diameterAB : diameter circle A B) :
  (interior_point circle C → ∃ (obtuse : Angle), isObtuse (angle A C B))
  ∧ (exterior_point circle C → ∃ (acute : Angle), isAcute (angle A C B)) := by
  sorry

end angle_acb_circle_l108_108787


namespace initial_honey_amount_l108_108835

variable (H : ℝ)

theorem initial_honey_amount :
  (0.70 * 0.60 * 0.50) * H = 315 → H = 1500 :=
by
  sorry

end initial_honey_amount_l108_108835


namespace flash_catches_ace_l108_108846

theorem flash_catches_ace (v : ℝ) (x : ℝ) (y : ℝ) (hx : x > 1) :
  let t := y / (v * (x - 1))
  let ace_distance := v * t
  let flash_distance := x * v * t
  flash_distance = (xy / (x - 1)) :=
by
  let t := y / (v * (x - 1))
  let ace_distance := v * t
  let flash_distance := x * v * t
  have h1 : x * v * t = xy / (x - 1) := sorry
  exact h1

end flash_catches_ace_l108_108846


namespace range_of_a_l108_108648

theorem range_of_a (a : ℝ) : (2 * a - 8) / 3 < 0 → a < 4 :=
by sorry

end range_of_a_l108_108648


namespace sum_of_first_six_terms_l108_108252

variable {a_n : ℕ → ℕ}
variable {d : ℕ}

def is_arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d 

theorem sum_of_first_six_terms (a_3 a_4 : ℕ) (h : a_3 + a_4 = 30) :
  ∃ a_n d, is_arithmetic_sequence a_n d ∧ 
  a_n 3 = a_3 ∧ a_n 4 = a_4 ∧ 
  (3 * (a_n 1 + (a_n 1 + 5 * d))) = 90 := 
sorry

end sum_of_first_six_terms_l108_108252


namespace proposition_true_l108_108209

variable (α : Type) [Real α]

theorem proposition_true :
  (¬ (∃ x0 : α, Real.exp x0 ≤ 0)) ∨ (∀ x : α, 2^x > x^2) :=
by
  sorry

end proposition_true_l108_108209


namespace range_of_m_iff_l108_108214

noncomputable def range_of_m (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (0 < x) → (0 < y) → ((2 / x) + (1 / y) = 1) → (x + 2 * y > m^2 + 2 * m)

theorem range_of_m_iff : (range_of_m m) ↔ (-4 < m ∧ m < 2) :=
  sorry

end range_of_m_iff_l108_108214


namespace beads_per_bracelet_l108_108702

-- Definitions for the conditions
def Nancy_metal_beads : ℕ := 40
def Nancy_pearl_beads : ℕ := Nancy_metal_beads + 20
def Rose_crystal_beads : ℕ := 20
def Rose_stone_beads : ℕ := Rose_crystal_beads * 2
def total_beads : ℕ := Nancy_metal_beads + Nancy_pearl_beads + Rose_crystal_beads + Rose_stone_beads
def bracelets : ℕ := 20

-- Statement to prove
theorem beads_per_bracelet :
  total_beads / bracelets = 8 :=
by
  -- skip the proof
  sorry

end beads_per_bracelet_l108_108702


namespace find_sum_of_xyz_l108_108362

theorem find_sum_of_xyz (x y z : ℕ) (h1 : 0 < x ∧ 0 < y ∧ 0 < z)
  (h2 : (x + y + z)^3 - x^3 - y^3 - z^3 = 300) : x + y + z = 7 :=
by
  sorry

end find_sum_of_xyz_l108_108362


namespace original_price_is_256_l108_108417

noncomputable def originalPrice (P : ℝ) : Prop :=
  (P: ℝ) - (P / 10) = 230

theorem original_price_is_256 : ∃ P : ℝ, originalPrice P ∧ P = 256 := 
by {
  use 256,
  split,
  { rw originalPrice,
    linarith },
  { refl }
}


end original_price_is_256_l108_108417


namespace max_a_for_minimum_value_l108_108686

def f (x a : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

theorem max_a_for_minimum_value : ∀ a : ℝ, (∃ m : ℝ, ∀ x : ℝ, f x a ≥ m) → a ≤ 1 :=
by
  sorry

end max_a_for_minimum_value_l108_108686


namespace simplify_expression_l108_108876

variable (a : ℝ) (ha : a ≠ 0)

theorem simplify_expression : (21 * a^3 - 7 * a) / (7 * a) = 3 * a^2 - 1 := by
  sorry

end simplify_expression_l108_108876


namespace three_digit_multiples_of_24_l108_108219

theorem three_digit_multiples_of_24 : 
  let a := 120
  let d := 24
  let l := 984
  l = a + (n - 1) * d → n = 37 := 
by
  intros a d l n
  let a := 120
  let d := 24
  let l := 984
  have eqn1 : l = a + (n - 1) * d := eq.symm sorry
  have eqn2 : n = 37 := sorry
  exact eqn2

end three_digit_multiples_of_24_l108_108219


namespace exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l108_108531

structure RatNonIntPair (x y : ℚ) :=
  (x_rational : x.is_rational)
  (x_not_integer : x.num ≠ x.denom)
  (y_rational : y.is_rational)
  (y_not_integer : y.num ≠ y.denom)

theorem exists_rat_nonint_sol_a :
  ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * x + int 8 * y).denom = 1 ∧ (int 8 * x + int 3 * y).denom = 1 := sorry

theorem no_exists_rat_nonint_sol_b :
  ¬ ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * (x^2) + int 8 * (y^2)).denom = 1 ∧ (int 8 * (x^2) + int 3 * (y^2)).denom = 1 := sorry

end exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l108_108531


namespace ff_two_eq_three_l108_108620

noncomputable def f (x : ℝ) : ℝ :=
  if x < 6 then x^3 else Real.log x / Real.log x

theorem ff_two_eq_three : f (f 2) = 3 := by
  sorry

end ff_two_eq_three_l108_108620


namespace betty_paid_44_l108_108105

def slippers := 6
def slippers_cost := 2.5
def lipstick := 4
def lipstick_cost := 1.25
def hair_color := 8
def hair_color_cost := 3

noncomputable def total_cost := (slippers * slippers_cost) + (lipstick * lipstick_cost) + (hair_color * hair_color_cost)

theorem betty_paid_44 : total_cost = 44 :=
by
  sorry

end betty_paid_44_l108_108105


namespace smallest_height_locus_l108_108172

variable {V : Type*} [inner_product_space ℝ V] -- V is a real inner product space

-- Definitions of points A, B, C, and H
variables (A B C H : V)

-- Function that defines the height from a point to a plane given by a triangle
noncomputable def height_from_point_to_plane (P A B C : V) : ℝ := sorry

-- Function to define the anticomplementary triangle
noncomputable def anticomplementary_triangle (A B C : V) : set V := sorry

-- Statement of the problem
theorem smallest_height_locus (A B C : V) :
  ∃ H : V, (∀ P, height_from_point_to_plane P A B C ≥ height_from_point_to_plane P H A B) ↔  (H ∈ (anticomplementary_triangle A B C) \ frontier (anticomplementary_triangle A B C)) :=
sorry

end smallest_height_locus_l108_108172


namespace expression_value_l108_108515

theorem expression_value :
  sqrt (4/5) - real.cbrt (5/4) = (2 * sqrt 5) / 5 - real.cbrt (5 / real.cbrt 4) := 
sorry

end expression_value_l108_108515


namespace system_of_equations_solution_fractional_equation_has_no_solution_l108_108722

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
  (2 * x - 7 * y = 5) ∧ 
  (3 * x - 8 * y = 10) ∧ 
  (x = 6) ∧ 
  (y = 1) :=
by {
  sorry,
}

theorem fractional_equation_has_no_solution :
  ¬ ∃ (x : ℚ), 
  x ≠ 1 ∧ 
  (3/(x-1) - (x + 2)/(x*(x-1)) = 0) :=
by {
  sorry,
}

end system_of_equations_solution_fractional_equation_has_no_solution_l108_108722


namespace state_a_selection_percentage_l108_108246

theorem state_a_selection_percentage :
  ∀ (num_candidates : ℕ) (x : ℕ),
    num_candidates = 8000 → 
    (7 * num_candidates / 100 + 80 = 8000 * x / 100) →
    x = 7 :=
begin
  intros num_candidates x hcandidates heq,
  rw hcandidates at heq,
  sorry
end

end state_a_selection_percentage_l108_108246


namespace find_s_l108_108026

structure Point :=
  (x : ℝ)
  (y : ℝ)

def D : Point := ⟨-2, 6⟩
def E : Point := ⟨2, -4⟩
def F : Point := ⟨6, -4⟩

def line_eq (p1 p2 : Point) : ℝ → ℝ := 
  let m := (p2.y - p1.y) / (p2.x - p1.x)
  let b := p1.y - m * p1.x
  fun x => m * x + b

noncomputable def area_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

noncomputable def x_intersection (l : ℝ → ℝ) (x : ℝ) : Point :=
  ⟨x, l x⟩

def line_DE := line_eq D E
def line_DF := line_eq D F

noncomputable def triangle_area_condition (s : ℝ) : Prop :=
  let V := x_intersection line_DE s
  let W := x_intersection line_DF s
  area_triangle D V W = 20

theorem find_s (s : ℝ) (h : triangle_area_condition s) : s = 4 :=
sorry

end find_s_l108_108026


namespace altitude_tangent_sum_l108_108240

theorem altitude_tangent_sum {A B C: Type*} [normed_add_comm_group A] [inner_product_space ℝ A] 
  (triangle: ∀ A B C : A, Prop) (a b c: ℝ) (altitude: ℝ) (circumcircle: ∀ A' : A, Prop)
  (DA: A → ℝ) (tan: ℝ → ℝ) :
  (∀ B C : ℝ, ∃ A : ℝ, triangle A B C) →
  (∀ A' B C : ℝ, altitude DA = a ∧ circumcircle A' ∧ DA = c) →
  (∀ B C : ℝ, tan B + tan C = ∑ A B C : ℝ, (a / DA)) :=
begin
  sorry,
end

end altitude_tangent_sum_l108_108240


namespace mean_age_euler_family_l108_108364

-- Definition of the list of ages of the children
def ages : List ℝ := [12, 12, 12, 12, 9, 9, 15, 17]

-- Total sum of ages
def total_sum_ages : ℝ := ages.sum

-- Number of children
def number_of_children : ℝ := ages.length

-- Mean (average) age calculation
def mean_age : ℝ := total_sum_ages / number_of_children

-- The theorem stating that the mean age is 12.25
theorem mean_age_euler_family : mean_age = 12.25 :=
by
  sorry

end mean_age_euler_family_l108_108364


namespace jellybean_probability_l108_108826

theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 6
  let blue_jellybeans := 3
  let white_jellybeans := 6
  let total_chosen := 4
  let total_combinations := Nat.choose total_jellybeans total_chosen
  let red_combinations := Nat.choose red_jellybeans 3
  let non_red_combinations := Nat.choose (blue_jellybeans + white_jellybeans) 1
  let successful_outcomes := red_combinations * non_red_combinations
  let probability := (successful_outcomes : ℚ) / total_combinations
  probability = 4 / 91 :=
by 
  sorry

end jellybean_probability_l108_108826


namespace paintable_sum_l108_108982

def is_paintable (h t u : ℕ) : Prop :=
  (h ≠ 1) ∧ (h ≠ 2) ∧ 
  ((h = 3 ∧ t = 3 ∧ u = 3) ∨ (h = 4 ∧ t = 2 ∧ u = 4))

def paintable_value (h t u : ℕ) : ℕ :=
  100 * h + 10 * t + u

theorem paintable_sum :
  (∑ n in {n | ∃ h t u, is_paintable h t u ∧ n = paintable_value h t u}, id) = 757 :=
by
  sorry

end paintable_sum_l108_108982


namespace total_trees_planted_l108_108568

def side1 : ℕ := 198
def side2 : ℕ := 180
def side3 : ℕ := 210
def distance_between_trees : ℕ := 6

theorem total_trees_planted :
  let perimeter := side1 + side2 + side3 in
  let number_of_trees := perimeter / distance_between_trees in
  number_of_trees = 98 :=
by
  let perimeter := side1 + side2 + side3
  let number_of_trees := perimeter / distance_between_trees
  show number_of_trees = 98
  sorry

end total_trees_planted_l108_108568


namespace inequality_proof_l108_108936

theorem inequality_proof (n : ℕ) (h : n ≥ 2) (a : Fin n → ℝ) (A : ℝ)
  (cond : A + ∑ i, a i ^ 2 < (1 / (n - 1 : ℝ)) * (∑ i, a i) ^ 2) :
  ∀ (i j : Fin n), i < j → A < 2 * a i * a j :=
by
  sorry

end inequality_proof_l108_108936


namespace possible_values_sin_plus_cos_l108_108603

variable (x : ℝ)

theorem possible_values_sin_plus_cos (h : 2 * Real.cos x - 3 * Real.sin x = 2) :
    ∃ (values : Set ℝ), values = {3, -31 / 13} ∧ (Real.sin x + 3 * Real.cos x) ∈ values := by
  sorry

end possible_values_sin_plus_cos_l108_108603


namespace not_necessarily_square_l108_108926

-- Definitions to translate conditions
variable {X : Type*} [metric_space X]

def is_convex (q : quadrilateral X) : Prop := -- Definition for a convex quadrilateral
sorry

def divides_into_isosceles_triangles (q : quadrilateral X) : Prop :=
  ∀ diagonal : diagonal X, 
  divides_quad_into_isosceles q diagonal ∧
  divides_quad_into_four_isosceles q

-- The theorem statement
theorem not_necessarily_square (q : quadrilateral X) 
  (convex_q : is_convex q)
  (isosceles_division_q : divides_into_isosceles_triangles q) :
  ¬(must_be_square q) :=
sorry

end not_necessarily_square_l108_108926


namespace chars_read_first_day_l108_108018

-- Define the problem conditions.
variable (total_chars : ℕ := 34685)
variable (chars_first_day : ℕ)

-- Define the equation based on the conditions.
def reads_each_day (x : ℕ) := x + 2 * x + 4 * x = total_chars

-- Statement to prove, showing that the number of characters read on the first day is 4955.
theorem chars_read_first_day : reads_each_day chars_first_day → chars_first_day = 4955 := sorry

end chars_read_first_day_l108_108018


namespace partition_set_contains_sum_l108_108883

theorem partition_set_contains_sum (A : Finset ℕ) (hA : A = Finset.range 50 \ Finset.singleton 0)
    (P : A → Fin 3) :
  ∃ (S : Finset ℕ), S ⊆ A ∧ ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a + b = c := 
sorry

end partition_set_contains_sum_l108_108883


namespace unique_ordered_triple_l108_108092

theorem unique_ordered_triple (a b c : ℕ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : a^3 + b^3 + c^3 + 648 = (a + b + c)^3) :
  (a, b, c) = (3, 3, 3) ∨ (a, b, c) = (3, 3, 3) ∨ (a, b, c) = (3, 3, 3) :=
sorry

end unique_ordered_triple_l108_108092


namespace product_prime_probability_is_10_over_77_l108_108705

open Nat

/-- Paco's spinner selects a number between 1 and 7 and Manu's spinner selects a number between 1 and 11. 
Given these, the probability that the product of Manu's number and Paco's number is prime is 10/77. -/
theorem product_prime_probability_is_10_over_77 : 
  let Paco := {i | 1 ≤ i ∧ i ≤ 7}
  let Manu := {j | 1 ≤ j ∧ j ≤ 11}
  let total_outcomes := (finset.product (finset.range 8) (finset.range 12)).filter (λ (x : ℕ × ℕ), 1 ≤ x.1 ∧ x.1 ≤ 7 ∧ 1 ≤ x.2 ∧ x.2 ≤ 11) 
  let prime_outcomes := total_outcomes.filter (λ (x : ℕ × ℕ), prime (x.1 * x.2))
  (prime_outcomes.card : ℚ) / (total_outcomes.card : ℚ) = 10 / 77 :=
begin
  sorry
end

end product_prime_probability_is_10_over_77_l108_108705


namespace intersection_A_B_l108_108973

def A := {-2, -1, 0, 1, 2} : Set ℤ
def B := {x : ℝ | 3 - x^2 ≥ 0}

theorem intersection_A_B : (A ∩ B) = {-1, 0, 1} :=
by
  sorry

end intersection_A_B_l108_108973


namespace divisor_of_p_l108_108312

open Nat

theorem divisor_of_p (p q r s : ℕ) (hpq : gcd p q = 21) (hqr : gcd q r = 45) (hrs : gcd r s = 75) (hsp : 120 < gcd s p ∧ gcd s p < 180) :
  9 ∣ p :=
sorry

end divisor_of_p_l108_108312


namespace expenditure_of_negative_amount_l108_108228

theorem expenditure_of_negative_amount (x : ℝ) (h : x < 0) : 
  ∃ y : ℝ, y > 0 ∧ x = -y :=
by
  sorry

end expenditure_of_negative_amount_l108_108228


namespace cost_of_schools_renovation_plans_and_min_funding_l108_108659

-- Define costs of Type A and Type B schools
def cost_A : ℝ := 60
def cost_B : ℝ := 85

-- Initial conditions given in the problem
axiom initial_condition_1 : cost_A + 2 * cost_B = 230
axiom initial_condition_2 : 2 * cost_A + cost_B = 205

-- Variables for number of Type A and Type B schools to renovate
variables (x : ℕ) (y : ℕ)
-- Total schools to renovate
axiom total_schools : x + y = 6

-- National and local finance constraints
axiom national_finance_max : 60 * x + 85 * y ≤ 380
axiom local_finance_min : 10 * x + 15 * y ≥ 70

-- Proving the cost of one Type A and one Type B school
theorem cost_of_schools : cost_A = 60 ∧ cost_B = 85 := 
by {
  sorry
}

-- Proving the number of renovation plans and the least funding plan
theorem renovation_plans_and_min_funding :
  ∃ x y, (x + y = 6) ∧ 
         (10 * x + 15 * y ≥ 70) ∧ 
         (60 * x + 85 * y ≤ 380) ∧ 
         (x = 2 ∧ y = 4 ∨ x = 3 ∧ y = 3 ∨ x = 4 ∧ y = 2) ∧ 
         (∀ (a b : ℕ), (a + b = 6) ∧ 
                       (10 * a + 15 * b ≥ 70) ∧ 
                       (60 * a + 85 * b ≤ 380) → 
                       60 * a + 85 * b ≥ 410) :=
by {
  sorry
}

end cost_of_schools_renovation_plans_and_min_funding_l108_108659


namespace rectangle_area_in_triangle_l108_108088

theorem rectangle_area_in_triangle (b h x y : ℝ) (hb : 0 < b) (hh : 0 < h)
  (hy : y = (b * x) / h) :
  area_of_rectangle : ℝ := (b * x^2) / h := sorry

end rectangle_area_in_triangle_l108_108088


namespace veg_price_proof_min_spent_proof_l108_108344

def A_veg_base_price (market_price: ℚ) (base_price: ℚ): Prop :=
  market_price = (5/4) * base_price

def purchase_equation (base_price: ℚ): Prop :=
  (300 / base_price) = (300 / ((5/4) * base_price)) + 3

def B_veg_base_price: ℚ := 30

def total_bundles (total: ℕ) (A: ℕ) (B: ℕ): Prop :=
  total = A + B

def A_not_exceed_B (A: ℕ) (B: ℕ): Prop :=
  A ≤ B

def discounted_price (base_price_A base_price_B: ℚ) (A B: ℕ): ℚ :=
  0.9 * base_price_A * A + 0.9 * base_price_B * B

def min_spent (base_price_A base_price_B: ℚ) (total: ℕ): ℚ :=
  let min_A := total / 2 in
  0.9 * base_price_A * min_A + 0.9 * base_price_B * (total - min_A)

theorem veg_price_proof: 
  ∃ base_price: ℚ, 
  A_veg_base_price (5 / 4 * base_price) base_price ∧ 
  purchase_equation base_price ∧ 
  base_price = 20 := 
by 
  sorry

theorem min_spent_proof:
  ∃ w: ℚ, 
  total_bundles 100 A B ∧ 
  A_not_exceed_B A B ∧ 
  (base_price_A = 20) ∧ 
  (base_price_B = B_veg_base_price) ∧ 
  w = min_spent base_price_A B_veg_base_price 100 ∧ 
  w = 2250 := 
by 
  sorry

end veg_price_proof_min_spent_proof_l108_108344


namespace multiple_choice_problem_l108_108799

theorem multiple_choice_problem :
  8^0 = 1 ∧ |-8| = 8 ∧ -(-8) = 8 ∧ sqrt 8 = 2 * sqrt 2 :=
by {
  -- proof will go here
  sorry
}

end multiple_choice_problem_l108_108799


namespace find_lambda_range_l108_108630

noncomputable section

open Real

variables (a b c : ℝ × ℝ)
variables (λ : ℝ)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def collinear (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

def is_acute (a c : ℝ × ℝ) : Prop := dot_product a c > 0

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (1, 1)
def c (λ : ℝ) : ℝ × ℝ := (a.1 + λ * b.1, a.2 + λ * b.2)

theorem find_lambda_range (λ : ℝ) :
  λ > -5 / 2 ∧ λ ≠ 0 → is_acute a (c λ) ∧ ¬ collinear a (c λ) :=
by
  intro h
  -- proof omitted
  sorry

end find_lambda_range_l108_108630


namespace neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one_l108_108011

theorem neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one :
  ¬(∃ x : ℝ, x^2 < 1) ↔ ∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 := 
by 
  sorry

end neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one_l108_108011


namespace exists_rational_non_integer_xy_no_rational_non_integer_xy_l108_108527

-- Part (a)
theorem exists_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  (∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
   ∃ z1 z2 : ℤ, 19 * x + 8 * y = ↑z1 ∧ 8 * x + 3 * y = ↑z2) :=
sorry

-- Part (b)
theorem no_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  ¬ ∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
  ∃ z1 z2 : ℤ, 19 * x^2 + 8 * y^2 = ↑z1 ∧ 8 * x^2 + 3 * y^2 = ↑z2 :=
sorry

end exists_rational_non_integer_xy_no_rational_non_integer_xy_l108_108527


namespace find_original_number_l108_108649

theorem find_original_number (x : ℝ) :
  (((x / 2.5) - 10.5) * 0.3 = 5.85) -> x = 75 :=
by
  sorry

end find_original_number_l108_108649


namespace exists_rational_non_integer_a_not_exists_rational_non_integer_b_l108_108537

-- Define rational non-integer numbers
def is_rational_non_integer (x : ℚ) : Prop := ¬(∃ (z : ℤ), x = z)

-- (a) Proof for existance of rational non-integer numbers y and x such that 19x + 8y, 8x + 3y are integers
theorem exists_rational_non_integer_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ a b : ℤ, 19 * x + 8 * y = a ∧ 8 * x + 3 * y = b) :=
sorry

-- (b) Proof for non-existance of rational non-integer numbers y and x such that 19x² + 8y², 8x² + 3y² are integers
theorem not_exists_rational_non_integer_b :
  ¬ ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ m n : ℤ, 19 * x^2 + 8 * y^2 = m ∧ 8 * x^2 + 3 * y^2 = n) :=
sorry

end exists_rational_non_integer_a_not_exists_rational_non_integer_b_l108_108537


namespace smallest_positive_integer_a_for_polynomial_l108_108514

theorem smallest_positive_integer_a_for_polynomial :
  ∃ a b c : ℤ, (0 < a) ∧ (1 ≤ b ∧ b ≤ 2 * a - 1) ∧ (c ≥ 1) ∧ (ax^2 - bx + c).has_two_distinct_zeros_in_open_interval_zero_one ∧ (∀ a' b' c' : ℤ, (0 < a' → a' < a) → ¬(ax^2 - bx + c').has_two_distinct_zeros_in_open_interval_zero_one) := sorry

end smallest_positive_integer_a_for_polynomial_l108_108514


namespace product_fraction_eq_47041_l108_108117

theorem product_fraction_eq_47041 :
  (∏ n in Finset.range 30, (n + 1 + 4) / (n + 1)) = 47041 := by
  sorry

end product_fraction_eq_47041_l108_108117


namespace sqrt_eq_seven_iff_x_eq_fifty_four_l108_108146

theorem sqrt_eq_seven_iff_x_eq_fifty_four (x : ℝ) : sqrt (x - 5) = 7 ↔ x = 54 := by
  sorry

end sqrt_eq_seven_iff_x_eq_fifty_four_l108_108146


namespace sum_of_satisfying_ns_l108_108437

-- Defining the given condition
def satisfies_condition (n : ℕ) : Prop :=
  Nat.lcm n 120 = Nat.gcd n 120 + 300

-- The proof statement
theorem sum_of_satisfying_ns : 
  (∑ n in Finset.filter satisfies_condition (Finset.range 1000), n) = 180 :=
sorry

end sum_of_satisfying_ns_l108_108437


namespace inequality_relationship_l108_108923

noncomputable def a : ℝ := 3 ^ 0.3
noncomputable def b : ℝ := Real.log 3 / Real.log π
noncomputable def c : ℝ := Real.log e / Real.log 0.3

theorem inequality_relationship : a > b ∧ b > c := by
  sorry

end inequality_relationship_l108_108923


namespace contrapositive_sin_inequality_l108_108371

-- Definitions based on the problem conditions
variable (A B : ℝ) -- A and B are real numbers representing angles
variable (sin : ℝ → ℝ) -- sine function

-- The condition in the problem: A, B are angles in a triangle
def is_angle_of_triangle (a b : ℝ) : Prop :=
  0 < a ∧ a < π ∧ 0 < b ∧ b < π

-- The given Lean statement of the problem
theorem contrapositive_sin_inequality 
  (A B : ℝ) 
  (is_triangle : is_angle_of_triangle A B)
  (h : sin A ≤ sin B) : A ≤ B := 
sorry

end contrapositive_sin_inequality_l108_108371


namespace intersection_point_at_neg4_l108_108002

def f (x : Int) (b : Int) : Int := 4 * x + b
def f_inv (y : Int) (b : Int) : Int := (y - b) / 4

theorem intersection_point_at_neg4 (a b : Int) (h1 : f (-4) b = a) (h2 : f_inv (-4) b = a) : a = -4 := 
by 
  sorry

end intersection_point_at_neg4_l108_108002


namespace existence_of_k_value_of_Ak_l108_108010

def f (A : ℕ) (n : ℕ) (digits : ℕ → ℕ) : ℕ :=
  List.sum (List.map (λ i, 2^(n-i) * digits i) (List.range $ n + 1))

theorem existence_of_k (A : ℕ) (n : ℕ) (digits : ℕ → ℕ) (hA : ∀ i, digits i < 10) :
  ∃ k, let Aₖ := A.iterate f k in f Aₖ n digits = Aₖ := sorry

theorem value_of_Ak (A : ℕ) (hA : A = 19^86) :
  ∃ k, let Aₖ := A.iterate (λ a, f a (Nat.digits 10 a).length (λ i, (Nat.digits 10 a).get i)) k in
  Aₖ = 19 := sorry

end existence_of_k_value_of_Ak_l108_108010


namespace unit_prices_max_profit_l108_108782

-- Define the constants and variables
constant x : ℕ -- unit price of fresh Morel mushrooms in RMB/kg
constant y : ℕ -- unit price of dried Morel mushrooms in RMB/kg
constant a : ℕ -- quantity of fresh Morel mushrooms for 3rd purchase in kg
constant b : ℕ -- quantity of dried Morel mushrooms for 3rd purchase in kg

-- Given conditions for the first and second purchases
axiom first_purchase : 1000 * x + 300 * y = 152000
axiom second_purchase : 800 * x + 500 * y = 184000

-- Conclusion from solving the system of equations
theorem unit_prices : x = 80 ∧ y = 240 := sorry

-- Conditions for the third purchase to maximize profit
axiom total_weight : a + b = 1500
axiom dried_mushrooms_limit : b ≤ (1 / 3 : ℚ) * a

-- Retail prices and profit formula
def retail_price_fresh := 100
def retail_price_dried := 280
def profit := (retail_price_fresh - x) * a + (retail_price_dried - y) * b

-- Maximize profit under given constraints
theorem max_profit : a = 1125 ∧ b = 375 ∧ profit = 37500 := sorry

end unit_prices_max_profit_l108_108782


namespace measure_of_angle_DIG_l108_108506

-- Definition of a regular nonagon in terms of points
def is_regular_nonagon (A B C D E F G H I : Point) : Prop :=
  -- Assuming equidistant points from a central point and equal angles between consecutive points for simplicity
  ∃ O : Point, 
    (distance O A = distance O B) ∧ (distance O B = distance O C) ∧ 
    (distance O C = distance O D) ∧ (distance O D = distance O E) ∧
    (distance O E = distance O F) ∧ (distance O F = distance O G) ∧
    (distance O G = distance O H) ∧ (distance O H = distance O I) ∧
    (angle O A B = angle O B C) ∧ (angle O B C = angle O C D) ∧
    (angle O C D = angle O D E) ∧ (angle O D E = angle O E F) ∧
    (angle O E F = angle O F G) ∧ (angle O F G = angle O G H) ∧
    (angle O G H = angle O H I) ∧ (angle O H I = angle O I A) ∧
    -- 40 degrees central angle for each sector
    (angle O A B = 2 * math.pi / 9)

-- Statement of the theorem
theorem measure_of_angle_DIG (A B C D E F G H I : Point) :
  is_regular_nonagon A B C D E F G H I → 
  (measure (angle D I G) = 60) :=
by
  sorry

end measure_of_angle_DIG_l108_108506


namespace obtain_nonzero_function_l108_108457

noncomputable def f1 (x : ℝ) := x + 1
noncomputable def f2 (x : ℝ) := x^2 + 1
noncomputable def f3 (x : ℝ) := x^3 + 1
noncomputable def f4 (x : ℝ) := x^4 + 1

theorem obtain_nonzero_function :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, g x = x * f4 x) ∧
    (∀ x ≥ 0, g x ≥ 0) ∧ (∀ x ≤ 0, g x ≤ 0) :=
by {
  let g := λ x : ℝ, x * f4 x,
  use g,
  split,
  {
    intro x,
    refl,
  },
  split,
  {
    intro x,
    intro hx,
    simp [f4],
    nlinarith,
  },
  {
    intro x,
    intro hx,
    simp [f4],
    nlinarith,
  }
}


end obtain_nonzero_function_l108_108457


namespace second_number_l108_108068

theorem second_number (A B : ℝ) (h1 : A = 200) (h2 : 0.30 * A = 0.60 * B + 30) : B = 50 :=
by
  -- proof goes here
  sorry

end second_number_l108_108068


namespace cosine_sum_to_product_l108_108381

theorem cosine_sum_to_product :
  ∃ (a b c d : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  (∀ x : ℝ, cos 2 * x + cos 4 * x + cos 8 * x + cos 10 * x = a * cos b * x * cos c * x * cos d * x) ∧
  a + b + c + d = 14 := by
  existsi 4
  existsi 6
  existsi 3
  existsi 1
  split; try {norm_num}
  intro x
  sorry

end cosine_sum_to_product_l108_108381


namespace not_perfect_square_l108_108710

-- Definitions and Conditions
def N (k : ℕ) : ℕ := (10^300 - 1) / 9 * 10^k

-- Proof Statement
theorem not_perfect_square (k : ℕ) : ¬∃ (m: ℕ), m * m = N k := 
sorry

end not_perfect_square_l108_108710


namespace cos_A_value_l108_108991

-- Define necessary structures and assumptions
  
theorem cos_A_value (A : ℝ) (h₁ : tan A + cot A + sec A = 3) : cos A = 1 / 2 :=
by
  sorry

end cos_A_value_l108_108991


namespace part_I_part_II_l108_108320

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

-- Define the first part of the proof problem
theorem part_I :
  { x : ℝ | f x > 2 } = { x | x > 1 ∨ x < -5 } :=
sorry

-- Define the second part of the proof problem
theorem part_II (t : ℝ) :
  (∀ x : ℝ, f x ≥ t^2 - 11/2 * t) → t ∈ set.Icc (1/2) 5 :=
sorry

end part_I_part_II_l108_108320


namespace triangle_medians_perpendicular_sin_inequality_l108_108279

theorem triangle_medians_perpendicular_sin_inequality (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C]
  (triangle : Triangle A B C) 
  (medians_perpendicular : ∀ (G : Point) (B' C' : Point),
    is_centroid G triangle ∧ 
    is_median B G B' triangle ∧ 
    is_median C G C' triangle ∧ 
    is_perpendicular B' C' triangle):
  ∃ (medians_perpendicular : ∀ (triangle : Triangle A B C),
     (∀ (G : Point) (B' C' : Point),
       is_centroid G triangle ∧
       is_median B G B' triangle ∧
       is_median C G C' triangle ∧
       is_perpendicular B' C' triangle)),
  (∀ (B' C' : Point),
    is_perpendicular B' C' triangle →
    ∃ (G : Point),
      is_centroid G triangle →
      is_median B G B' triangle →
      is_median C G C' triangle →
      ∀ (B_angle C_angle : ℝ), 
        ∃ (sin_B sin_C sin_BC : ℝ),
          (sin_B = sin_of_angle B_angle ∧
           sin_C = sin_of_angle C_angle ∧
           sin_BC = sin_of_angle (B_angle + C_angle)) →
          sin_B * sin_C ≠ 0))) →
  (∀ (B_angle C_angle : ℝ), (∃ (sin_B sin_C sin_BC : ℝ),
    sin_B = sin_of_angle B_angle ∧
    sin_C = sin_of_angle C_angle ∧
    sin_BC = sin_of_angle (B_angle + C_angle)) →
    (sin_BC / (sin_B * sin_C)) ≥ (2 / 3))

end triangle_medians_perpendicular_sin_inequality_l108_108279


namespace a_plus_b_l108_108089

theorem a_plus_b (a b : ℤ) (k : ℝ) (Hk : k = a + Real.sqrt (b)) 
  (Hdist : Real.abs (Real.log10 k - Real.log10 (k + 9)) = 0.2) : a + b = 21 := by
  sorry

end a_plus_b_l108_108089


namespace number_of_non_pine_trees_l108_108411

theorem number_of_non_pine_trees (total_trees : ℕ) (percentage_pine : ℝ) (h1 : total_trees = 350) (h2 : percentage_pine = 0.70) : 
  total_trees - (percentage_pine * (total_trees : ℝ)).to_nat = 105 :=
by
  sorry

end number_of_non_pine_trees_l108_108411


namespace area_inside_circle_outside_square_l108_108493

theorem area_inside_circle_outside_square (square_side : ℕ) (circle_radius : ℕ) (same_center : Bool) :
  square_side = 2 → circle_radius = 1 → same_center = true → (real.pi : real) = real.pi :=
by
  sorry

end area_inside_circle_outside_square_l108_108493


namespace each_child_gets_twelve_cupcakes_l108_108064

def total_cupcakes := 96
def children := 8
def cupcakes_per_child : ℕ := total_cupcakes / children

theorem each_child_gets_twelve_cupcakes :
  cupcakes_per_child = 12 :=
by
  sorry

end each_child_gets_twelve_cupcakes_l108_108064


namespace area_of_BEIH_l108_108055

structure Point where
  x : ℚ
  y : ℚ

def B : Point := ⟨0, 0⟩
def A : Point := ⟨0, 2⟩
def D : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩
def E : Point := ⟨0, 1⟩
def F : Point := ⟨1, 0⟩
def I : Point := ⟨2/5, 6/5⟩
def H : Point := ⟨2/3, 2/3⟩

def quadrilateral_area (p1 p2 p3 p4 : Point) : ℚ :=
  (1/2 : ℚ) * 
  ((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) - 
   (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x))

theorem area_of_BEIH : quadrilateral_area B E I H = 7 / 15 := sorry

end area_of_BEIH_l108_108055


namespace train_lengths_combined_l108_108424

noncomputable def speed_to_mps (kmph : ℤ) : ℚ := (kmph : ℚ) * 5 / 18

def length_of_train (speed : ℚ) (time : ℚ) : ℚ := speed * time

theorem train_lengths_combined :
  let speed1_kmph := 100
  let speed2_kmph := 120
  let time1_sec := 9
  let time2_sec := 8
  let speed1_mps := speed_to_mps speed1_kmph
  let speed2_mps := speed_to_mps speed2_kmph
  let length1 := length_of_train speed1_mps time1_sec
  let length2 := length_of_train speed2_mps time2_sec
  length1 + length2 = 516.66 :=
by
  sorry

end train_lengths_combined_l108_108424


namespace cosine_angle_between_a_b_is_5_over_12_l108_108600

-- Definitions of vectors a and b, and proof that cosine of the angle between them is 5/12
variables {a b : RealVectorSpace}  -- Assuming appropriate definitions of vectors and spaces are in Mathlib

axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom cond1 : 2 * ∥a∥ = 3 * ∥b∥
axiom cond2 : dot_product a (a - 2 • b) = ∥b∥^2

theorem cosine_angle_between_a_b_is_5_over_12 :
  cos_angle a b = 5 / 12 :=
sorry

end cosine_angle_between_a_b_is_5_over_12_l108_108600


namespace rate_is_correct_l108_108728

noncomputable def rate_of_drawing_barbed_wire_per_meter
  (area_square_field : ℝ)
  (total_cost : ℝ)
  (width_gate : ℝ)
  (num_gates : ℕ) : ℝ :=
let side_length := real.sqrt area_square_field in
let perimeter := 4 * side_length in
let length_barbed_wire := perimeter - num_gates * width_gate in
total_cost / length_barbed_wire

theorem rate_is_correct :
  rate_of_drawing_barbed_wire_per_meter 3136 799.20 1 2 = 3.60 := by
  sorry

end rate_is_correct_l108_108728


namespace minimum_distance_to_2_l108_108318

theorem minimum_distance_to_2 (z : ℂ) (hz : |z| = 1) : ∃ (w : ℂ), |w| = 1 ∧ |w - 2| = 1 :=
sorry

end minimum_distance_to_2_l108_108318


namespace max_finite_roots_l108_108037

-- Given 100 distinct numbers partitioned into two sets of 50,
-- and an equation involving absolute values of these numbers,
-- prove the maximum number of finite roots is 49.

theorem max_finite_roots (a b : Fin 50 → ℝ) (h_distinct : (Set.range a ∪ Set.range b).InjOn id) :
  ∃ c d : Fin 50 → ℝ,
    (∀ x, 
      (∑ i, |x - a i| - ∑ i, |x - b i| = 0 → 
      ∑ i, |x - c i| - ∑ i, |x - d i| ≠ 0))
  ∧ ∀ x, (∑ i, |x - a i| = ∑ i, |x - b i| → 
      (x ∈ Set.range a ∪ Set.range b) →
      (∑ i, |x - a i| - ∑ i, |x - b i|) = 49) :=
sorry

end max_finite_roots_l108_108037


namespace compute_expression_value_l108_108119

-- Define necessary constants for the proof
def tan_60 := sqrt 3
def sin_45 := sqrt 2 / 2
def zero_exp (a : ℚ) := 1
def inv_exp (a : ℚ) := 1 / a
def abs_val (a : ℚ) := if a ≥ 0 then a else -a

-- Define the main expression to be proved
theorem compute_expression_value :
  2 * tan_60 - inv_exp (1/3) + (-2)^2 * zero_exp (2017 - sin_45) - abs_val (-sqrt 12) = 1 :=
by
  -- Proof goes here
  sorry

end compute_expression_value_l108_108119


namespace pool_water_volume_after_evaporation_l108_108857

theorem pool_water_volume_after_evaporation :
  let initial_volume := 300
  let evaporation_first_15_days := 1 -- in gallons per day
  let evaporation_next_15_days := 2 -- in gallons per day
  initial_volume - (15 * evaporation_first_15_days + 15 * evaporation_next_15_days) = 255 :=
by
  sorry

end pool_water_volume_after_evaporation_l108_108857


namespace inequality_solution_set_inequality_range_of_a_l108_108966

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : a = -8) :
  (|x - 3| + |x + 2| ≤ |a + 1|) ↔ (-3 ≤ x ∧ x ≤ 4) :=
by sorry

theorem inequality_range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x + 2| ≤ |a + 1|) ↔ (a ≤ -6 ∨ a ≥ 4) :=
by sorry

end inequality_solution_set_inequality_range_of_a_l108_108966


namespace candy_distribution_l108_108899

/-! 
    We want to prove that there are exactly 2187 ways to distribute 8 distinct pieces of candy 
    into three bags (red, blue, and white) such that each bag contains at least one piece of candy.
-/

theorem candy_distribution : 
  ∑ r in finset.range (8).filter (λ r, 1 ≤ r ∧ r ≤ 6), 
  ∑ b in finset.range (8 - r).filter (λ b, 1 ≤ b ∧ b ≤ 7 - r),
  nat.choose 8 r * nat.choose (8 - r) b = 2187 :=
by sorry

end candy_distribution_l108_108899


namespace complex_expr_proof_l108_108188

open Complex

def z : ℂ := 1 + Complex.i
def conj_z : ℂ := Complex.conj z
def imag_unit : ℂ := Complex.i

theorem complex_expr_proof :
  (z / imag_unit + imag_unit * conj_z) = 2 :=
by
  -- Here, you will provide the proof steps, which are omitted.
  sorry

end complex_expr_proof_l108_108188


namespace angle_YTZ_eq_180_l108_108707

noncomputable def circumcenter (A B C : Point) : Point := sorry
def circumcircle (A B C : Point) : Circle := sorry
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

variables (A B C O X Y Z T : Point)
variables {XY_eq_XZ : dist X Y = dist X Z}
variables {circ_O : O = circumcenter A B C}
variables {circ_X : X ∈ circumcircle B O C}
variables {T_on_AC : T ∈ line_chart(A, C)}
variables {T_circle : T ∈ circumcircle A B Y}

theorem angle_YTZ_eq_180 :
  O = circumcenter A B C →
  X ∈ circumcircle B O C →
  XY_eq_XZ →
  T ∈ circumcircle A B Y →
  T ∈ line_chart(A, C) →
  ∠ YTZ = 180 :=
by
  intro hO hX hXY hTcirc hTline
  sorry

end angle_YTZ_eq_180_l108_108707


namespace area_of_two_sectors_l108_108423

-- Define the radius and central angle
def radius : ℝ := 15
def central_angle : ℝ := (90 / 360)

-- Define the area of a full circle
def full_circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the area of one sector based on the central angle
def sector_area (r : ℝ) (angle_fraction : ℝ) : ℝ := angle_fraction * full_circle_area r

-- Proof statement: total area of two sectors
theorem area_of_two_sectors : 
  2 * sector_area radius central_angle = 112.5 * Real.pi :=
by
  -- We state this but defer the proof for now
  sorry

end area_of_two_sectors_l108_108423


namespace angle_between_vectors_l108_108632

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (angle_ab : ℝ)

-- Conditions
def condition1 : ∥a∥ = 2 := sorry
def condition2 : ∥b∥ = 2 := sorry
def condition3 : inner a (b - a) = -6 := sorry

-- Desired angle between vectors a and b
def desired_angle : angle_ab = 2 * Real.pi / 3 := sorry

-- Main theorem
theorem angle_between_vectors : 
  (∥a∥ = 2) → (∥b∥ = 2) → (inner a (b - a) = -6) → angle a b = 2 * Real.pi / 3 :=
by 
  intros 
  exact desired_angle

end angle_between_vectors_l108_108632


namespace midpoint_of_segment_l108_108038

theorem midpoint_of_segment (A B : (ℤ × ℤ)) (hA : A = (12, 3)) (hB : B = (-8, -5)) :
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = -1 :=
by
  sorry

end midpoint_of_segment_l108_108038


namespace sum_first_11_terms_l108_108175

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [Module ℝ α] 
variable (a : ℕ → α) (d : α)

noncomputable def arithmetic_sequence (a₁ : α) : ℕ → α
| 0     => a₁
| (n+1) => a₁ + (n+1) • d

theorem sum_first_11_terms (a₁ : α) (a₄_plus_a₈ : a 4 + a 8 = 16) :
  ∑ n in Finset.range 11, a n = 88 := sorry

end sum_first_11_terms_l108_108175


namespace age_proof_l108_108079

theorem age_proof (M S Y : ℕ) (h1 : M = 36) (h2 : S = 12) (h3 : M = 3 * S) : 
  (M + Y = 2 * (S + Y)) ↔ (Y = 12) :=
by 
  sorry

end age_proof_l108_108079


namespace max_value_of_f_for_ge_zero_l108_108432

def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 10

theorem max_value_of_f_for_ge_zero : ∀ x : ℝ, x ≥ 0 → f x ≤ 12 :=
begin
    sorry
end

end max_value_of_f_for_ge_zero_l108_108432


namespace arithmetic_sequence_tenth_term_l108_108321

noncomputable def sum_of_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

def nth_term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_tenth_term
  (a1 d : ℝ)
  (h1 : a1 + (a1 + d) + (a1 + 2 * d) = (a1 + 3 * d) + (a1 + 4 * d))
  (h2 : sum_of_arithmetic_sequence a1 d 5 = 60) :
  nth_term a1 d 10 = 26 :=
sorry

end arithmetic_sequence_tenth_term_l108_108321


namespace correct_proposition_is_D_l108_108257

-- Define propositions as given conditions
def proposition_A : Prop :=
  ∀ (l1 l2 l3 : Line), 
  (perpendicular l1 l3 ∧ perpendicular l2 l3) → parallel l1 l2

def proposition_B : Prop :=
  ∀ (P1 P2 : Plane) (l1 l2 : Line), 
  (parallel P1 l1 ∧ parallel P2 l2 ∧ perpendicular l1 l2) → perpendicular P1 P2

def proposition_C : Prop :=
  ∀ (P1 P2 : Plane) (l : Line) (p : Point), 
  (perpendicular P1 P2 ∧ intersects l P1 ∧ intersects l P2 ∧ perpendicular l intersection_line) → perpendicular l P2

def proposition_D : Prop :=
  ∀ (l1 : Line) (P : Plane), 
  (parallel l1 P ∧ ∀ (l2 : Line), perpendicular l2 P) → perpendicular l1 l2

-- Theorem to prove
theorem correct_proposition_is_D : proposition_D :=
  by
    -- Proof goes here 
    sorry

end correct_proposition_is_D_l108_108257


namespace triangle_angle_ratio_l108_108498

theorem triangle_angle_ratio
  (O : Point) (A B C E : Point)
  (h1 : AcuteTriangle A B C)
  (h2 : InscribedInCircle O A B C)
  (h3 : ArcAngle A B = 120)
  (h4 : ArcAngle B C = 72)
  (h5 : EOnMinorArc A C O)
  (h6 : Perpendicular (Line O E) (Line A C)) :
  Ratio (Angle O B E) (Angle B A C) = 1 / 3 :=
sorry

end triangle_angle_ratio_l108_108498


namespace num_green_balls_is_7_l108_108718

-- Definitions based on conditions
def total_balls : ℕ := 40
def blue_balls : ℕ := 11
def red_balls : ℕ := 2 * blue_balls
def noncomputable green_balls : ℕ := total_balls - (blue_balls + red_balls)

-- Theorem stating the problem and the expected answer
theorem num_green_balls_is_7 : green_balls = 7 :=
by
  sorry

end num_green_balls_is_7_l108_108718


namespace count_two_digit_factors_of_3_pow_24_minus_1_l108_108636

theorem count_two_digit_factors_of_3_pow_24_minus_1 : 
  ∃ n : ℕ, n = 5 ∧ ∀ k : ℕ, (10 ≤ k ∧ k < 100 ∧ k ∣ (3^24 - 1) ≡ k ∣ (k ∈ {13, 26, 41, 82, 91})) :=
by
  sorry

end count_two_digit_factors_of_3_pow_24_minus_1_l108_108636


namespace problem_statement_l108_108305

open BigOperators

def max_value_and_permutations (s : List ℕ) : ℕ × ℕ :=
  let perms := s.permutations
  let values := perms.map (λ l, (l.zip_with (*) l.rotate ++ [l.head' * l.last']).sum)
  let M := values.maximum
  let N := values.filter (λ v, v = M).length
  (M, N)

theorem problem_statement :
  let M_N := max_value_and_permutations [1, 2, 3, 4, 5, 6]
  M_N.fst + M_N.snd = 88 := by
  sorry

end problem_statement_l108_108305


namespace arith_prog_sum_120_l108_108235

open Nat

def sum_arith (n : ℕ) (a d : ℚ) := (n : ℚ) / 2 * (2 * a + (n - 1) * d)

axiom S15_eq : (sum_arith 15 a d) = 150
axiom S115_eq : (sum_arith 115 a d) = 5

theorem arith_prog_sum_120 (a d : ℚ) :
  sum_arith 120 a d = -2620 / 77 :=
by
  sorry

end arith_prog_sum_120_l108_108235


namespace sum_of_interior_angles_pentagon_l108_108401

theorem sum_of_interior_angles_pentagon : (5 - 2) * 180 = 540 := by
  sorry

end sum_of_interior_angles_pentagon_l108_108401


namespace probability_arithmetic_sequence_l108_108466

theorem probability_arithmetic_sequence :
  ∃ (p : ℚ), p = 1 / 4 ∧ ∀ (s : finset ℕ), s = {2, 0, 1, 9} →
  (∃ (selection : finset (finset ℕ)), 
      selection = {s_sub | s_sub.card = 3 ∧ ∃ d, ∀ x ∈ s_sub, x = s_sub.min' _ + d * s.indexOf x} →
      (selection.card * p = 1))  :=
begin
  sorry
end

end probability_arithmetic_sequence_l108_108466


namespace determine_n_l108_108809

theorem determine_n (n : ℕ) (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26) : n = 25 :=
by
  sorry

end determine_n_l108_108809


namespace area_perimeter_inequality_l108_108421

-- Definitions of the areas and perimeters to be used in the inequality
variables (A : Type) [inst : Simplex A] -- triangle A as a simplex
variables (B : Type) [instB : Polygon B] -- convex polygon B

def area_A : ℝ := area A
def area_B : ℝ := area B
def perimeter_A : ℝ := perimeter A
def perimeter_B : ℝ := perimeter B

-- Condition: A is contained within B
axiom contains : A ⊆ B

-- The theorem to be proven
theorem area_perimeter_inequality 
  (A_contained_in_B : contains)
  (area_A_def : area_A = area A)
  (area_B_def : area_B = area B)
  (perimeter_A_def : perimeter_A = perimeter A)
  (perimeter_B_def : perimeter_B = perimeter B) :
  (area_A / area_B) < (perimeter_A / perimeter_B) := 
sorry

end area_perimeter_inequality_l108_108421


namespace checkerboard_sums_l108_108363

-- Define the dimensions and the arrangement of the checkerboard
def n : ℕ := 10
def board (i j : ℕ) : ℕ := i * n + j + 1

-- Define corner positions
def top_left_corner : ℕ := board 0 0
def top_right_corner : ℕ := board 0 (n - 1)
def bottom_left_corner : ℕ := board (n - 1) 0
def bottom_right_corner : ℕ := board (n - 1) (n - 1)

-- Sum of the corners
def corner_sum : ℕ := top_left_corner + top_right_corner + bottom_left_corner + bottom_right_corner

-- Define the positions of the main diagonals
def main_diagonal (i : ℕ) : ℕ := board i i
def anti_diagonal (i : ℕ) : ℕ := board i (n - 1 - i)

-- Sum of the main diagonals
def diagonal_sum : ℕ := (Finset.range n).sum main_diagonal + (Finset.range n).sum anti_diagonal - (main_diagonal 0 + main_diagonal (n - 1))

-- Statement to prove
theorem checkerboard_sums : corner_sum = 202 ∧ diagonal_sum = 101 :=
by
-- Proof is not required as per the instructions
sorry

end checkerboard_sums_l108_108363


namespace min_value_of_a_plus_b_minus_c_l108_108180

theorem min_value_of_a_plus_b_minus_c (a b c : ℝ)
  (h : ∀ x y : ℝ, 3 * x + 4 * y - 5 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ 3 * x + 4 * y + 5) :
  a = 3 ∧ b = 4 ∧ -5 ≤ c ∧ c ≤ 5 ∧ a + b - c = 2 :=
by {
  sorry
}

end min_value_of_a_plus_b_minus_c_l108_108180


namespace circle_line_tangent_l108_108389

noncomputable def circle := { x : ℝ, y : ℝ // x^2 + y^2 + 4*x - 2*y + 1 = 0 }
noncomputable def line : ℝ → ℝ := λ (x : ℝ) => (4/3) * x

theorem circle_line_tangent :
  let c := (-2 : ℝ, 1 : ℝ) in  -- center of the circle (derived from completing the square)
  let r := 2 in                -- radius of the circle (derived)
  ∀ (x y : ℝ), (x, y) ∈ circle → 
  abs (3 * (-2) - 4 * 1) / sqrt (3^2 + (-4)^2) = r :=
by
  sorry

end circle_line_tangent_l108_108389


namespace sum_of_interior_angles_of_pentagon_l108_108399

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  sorry

end sum_of_interior_angles_of_pentagon_l108_108399


namespace probability_task_force_same_gender_probability_task_force_includes_english_probability_task_force_english_diff_gender_l108_108361

noncomputable def members := ["A", "B", "C", "D", "a", "b"]
noncomputable def english_speakers := ["B", "C", "D", "b"]

noncomputable def pairs := List.pairs members members

noncomputable def probability_same_gender := 
  let same_gender_pairs := [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"), ("a", "b")]
  same_gender_pairs.length / pairs.length

noncomputable def probability_english_speakers := 
  let non_english_pairs := [("A", "a")]
  (pairs.length - non_english_pairs.length) / pairs.length

noncomputable def probability_english_diff_gender := 
  let diff_gender_pairs := [("A", "b"), ("B", "a"), ("B", "b"), ("C", "a"), ("C", "b"), ("D", "a"), ("D", "b")]
  diff_gender_pairs.length / pairs.length

theorem probability_task_force_same_gender :
  probability_same_gender = 7 / 15 := sorry

theorem probability_task_force_includes_english :
  probability_english_speakers = 14 / 15 := sorry

theorem probability_task_force_english_diff_gender :
  probability_english_diff_gender = 7 / 15 := sorry

end probability_task_force_same_gender_probability_task_force_includes_english_probability_task_force_english_diff_gender_l108_108361


namespace angle_DBA_is_87_l108_108121

-- Define points and angles
variable {A B C D : Type}
variable [collinear_points : Collinear A B C]
variable [not_on_line : ¬ Collinear A B D]
variable (angle_ABC := 122)
variable (angle_ABD := 49)
variable (angle_DBC := 35)

-- The proof statement
theorem angle_DBA_is_87 :
  ∃ (angle_DBA : ℕ), angle_DBA = 87 :=
sorry

end angle_DBA_is_87_l108_108121


namespace sum_of_powers_l108_108585

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem sum_of_powers (x = Complex.exp (2 * Real.pi * Complex.I / 5)) :
  1 + x^4 + x^8 + x^{12} + x^{16} = 0 :=
sorry

end sum_of_powers_l108_108585


namespace problem_part1_problem_part2_problem_part3_l108_108596

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + n

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then S n else S n - S (n - 1)

noncomputable def b_n (n : ℕ) : ℕ := 2^(n - 1)

noncomputable def T_n (n : ℕ) : ℕ :=
  (4 * n - 5) * 2^n + 5

theorem problem_part1 (n : ℕ) (h : n > 0) : n > 0 → a_n n = 4 * n - 1 := by
  sorry

theorem problem_part2 (n : ℕ) (h : n > 0) : n > 0 → b_n n = 2^(n - 1) := by
  sorry

theorem problem_part3 (n : ℕ) (h : n > 0) : n > 0 → T_n n = (4 * n - 5) * 2^n + 5 := by
  sorry

end problem_part1_problem_part2_problem_part3_l108_108596


namespace distinct_integers_are_squares_l108_108067

theorem distinct_integers_are_squares
  (n : ℕ) 
  (h_n : n = 2000) 
  (x : Fin n → ℕ) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → x i ≠ x j)
  (h_product_square : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ∃ (m : ℕ), x i * x j * x k = m^2) :
  ∀ i : Fin n, ∃ (m : ℕ), x i = m^2 := 
sorry

end distinct_integers_are_squares_l108_108067


namespace smallest_process_result_l108_108434

def set_of_numbers : set ℕ := {2, 4, 6, 8, 10, 12}

noncomputable def smallest_result (s : set ℕ) : ℕ :=
  if h : (s = {2, 4, 6}) then
    let l := s.to_list in
    let a := l.nth 0 in
    let b := l.nth 1 in
    let c := l.nth 2 in
    min ((a + b) * c) (min ((a + c) * b) ((b + c) * a))
  else
    sorry

theorem smallest_process_result : smallest_result {2, 4, 6} = 20 :=
by {
  have h : set_of_numbers = {2, 4, 6} := rfl,
  simp [smallest_result, h],
  sorry
}

end smallest_process_result_l108_108434


namespace height_in_tank_l108_108657

noncomputable def height_of_water (r h water_percentage : ℝ) : ℝ :=
  let tank_volume := (1 / 3) * Real.pi * r^2 * h
  let water_volume := (water_percentage / 100) * tank_volume
  let y := (water_volume / ((1 / 3) * Real.pi * (r^2) * h)) ^ (1 / 3)
  h * y

theorem height_in_tank (r h : ℝ) (water_percentage : ℝ) (c d : ℕ) :
  r = 10 ∧ h = 60 ∧ water_percentage = 40 → height_of_water r h water_percentage = 12 * (d ^ (1 / 3).to_nat) ∧ c + d = 15 :=
by
  intro conditions
  sorry

end height_in_tank_l108_108657


namespace range_of_s_l108_108791

theorem range_of_s (s : ℝ → ℝ) : (∀ y, s y = 1 / ((1 + y)^2 + 1)) → set.range s = set.Ioc 0 1 :=
by
  intro hs
  sorry

end range_of_s_l108_108791


namespace secretary_longest_time_l108_108366

def ratio_times (x : ℕ) : Prop := 
  let t1 := 2 * x
  let t2 := 3 * x
  let t3 := 5 * x
  (t1 + t2 + t3 = 110) ∧ (t3 = 55)

theorem secretary_longest_time :
  ∃ x : ℕ, ratio_times x :=
sorry

end secretary_longest_time_l108_108366


namespace trees_not_pine_l108_108409

theorem trees_not_pine 
  (total_trees : ℕ)
  (percentage_pine : ℚ) 
  (H_total : total_trees = 350)
  (H_percentage : percentage_pine = 70 / 100) :
  total_trees - (total_trees * percentage_pine).toNat = 105 := 
by
  -- place a placeholder sorry for the proof
  sorry

end trees_not_pine_l108_108409


namespace max_a_for_minimum_value_l108_108685

def f (x a : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

theorem max_a_for_minimum_value : ∀ a : ℝ, (∃ m : ℝ, ∀ x : ℝ, f x a ≥ m) → a ≤ 1 :=
by
  sorry

end max_a_for_minimum_value_l108_108685


namespace prove_2f_sqrt_e_gt_f_e_l108_108167

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom condition (x : ℝ) (hx : x > 0) : (x * log x) * f'(x) < f(x)

theorem prove_2f_sqrt_e_gt_f_e : 2 * f (Real.sqrt Real.e) > f Real.e := by
  sorry

end prove_2f_sqrt_e_gt_f_e_l108_108167


namespace total_sandwiches_Carl_can_order_correct_l108_108014

noncomputable def total_sandwich_combinations (breads meats cheeses : ℕ) : ℕ :=
  breads * meats * cheeses

noncomputable def restricted_combinations (breads meats cheeses : ℕ) : ℕ :=
  let total := total_sandwich_combinations breads meats cheeses
  let condition1 := breads -- chicken and Swiss cheese combination
  let condition2 := cheeses -- rye bread and pepper bacon combination
  let condition3 := cheeses -- chicken and rye bread combination
  let overcount := 1 -- overlap of conditions

  total - condition1 - condition2 - condition3 + overcount

def different_sandwiches_Carl_can_order : ℕ :=
  restricted_combinations 5 7 6

#eval different_sandwiches_Carl_can_order -- Should evaluate to 194

theorem total_sandwiches_Carl_can_order_correct : different_sandwiches_Carl_can_order = 194 := by
  sorry

end total_sandwiches_Carl_can_order_correct_l108_108014


namespace jill_future_value_l108_108058

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_future_value :
  let P := 10000
  let r := 0.0396
  let n := 2
  let t := 2
  future_value P r n t ≈ 10815.66 :=
by
  sorry

end jill_future_value_l108_108058


namespace radius_of_circle_l108_108763

noncomputable def circle_radius : ℝ :=
  let eq := λ (x y : ℝ), x ^ 2 + y ^ 2 + 4 * x - 4 * y - 1 = 0 in
  3

theorem radius_of_circle (x y : ℝ) (h : x ^ 2 + y ^ 2 + 4 * x - 4 * y - 1 = 0) : 
  circle_radius = 3 := by
  sorry

end radius_of_circle_l108_108763


namespace number_of_valid_four_digit_numbers_l108_108427

def is_valid_number (n : ℕ) : Prop :=
  n < 2340 ∧ ∀ (d : ℕ), d ∈ [0, 1, 2, 3, 4] →

def count_valid_numbers : ℕ :=
  let digits := [0, 1, 2, 3, 4]
  let four_digit_numbers := { n | n < 10000 ∧ ∀ (d ∈ digits), d }
    (nums.filter (λ n, is_valid_number n)).length

theorem number_of_valid_four_digit_numbers : count_valid_numbers = 40 := sorry

end number_of_valid_four_digit_numbers_l108_108427


namespace find_x3_y3_l108_108643

theorem find_x3_y3 (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 + y^2 = 18) : x^3 + y^3 = 54 := 
by 
  sorry

end find_x3_y3_l108_108643


namespace candy_distribution_count_l108_108898

theorem candy_distribution_count :
  ∑ r in Finset.range 8 \ {0}, choose 8 r * (2 ^ (8 - r) - 1) = 
  ∑ r in Finset.range 1 8, ∑ b in Finset.range 1 (8 - r), choose 8 r * choose (8 - r) b * 2^ (8 - r - b) :=
by sorry

end candy_distribution_count_l108_108898


namespace find_ordered_pair_l108_108393

theorem find_ordered_pair (x y : ℝ) 
  (h : (⟨3, x, -9⟩ : ℝ × ℝ × ℝ) × (⟨4, 6, y⟩) = 0) : 
  x = 9 / 2 ∧ y = -12 :=
sorry

end find_ordered_pair_l108_108393


namespace g_diff_1_4_l108_108745

variable (g : ℝ → ℝ)

/-- g is a linear function satisfying g(d+1) - g(d) = 5 for all real numbers d. -/
axiom linear_g : ∀ (d : ℝ), g(d + 1) - g(d) = 5

theorem g_diff_1_4 : g 1 - g 4 = -15 :=
sorry

end g_diff_1_4_l108_108745


namespace num_edge_pairs_determining_plane_correct_l108_108220

-- Define the cube and its properties
def cube : Type := sorry
def edges (c : cube) : Finset (Finset cube) := sorry
lemma edges_card (c : cube) : (edges c).card = 12 := sorry

-- Define when two edges determine a plane
def determine_plane (e1 e2 : Finset cube) : Prop :=
  (parallel e1 e2 ∨ intersects e1 e2) -- assuming these relations being defined elsewhere

-- The total number of unordered pairs of edges that determine a plane
def num_edge_pairs_determining_plane (c : cube) : ℕ :=
  (Finset.powersetLen 2 (edges c)).count (λ pair, determine_plane pair.1 pair.2)

theorem num_edge_pairs_determining_plane_correct (c : cube) :
  num_edge_pairs_determining_plane c = 42 := 
  sorry

end num_edge_pairs_determining_plane_correct_l108_108220


namespace range_of_m_l108_108582

theorem range_of_m {m : ℝ} (h₀ : 0 < m) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ∈ Icc 0 1 ∧ x₂ ∈ Icc 0 1 ∧
    (m * x₁ - 3 + real.sqrt 2) ^ 2 - real.sqrt (x₁ + m) = 0 ∧
    (m * x₂ - 3 + real.sqrt 2) ^ 2 - real.sqrt (x₂ + m) = 0) ↔
  (3 - real.sqrt 2 < m ∧ m < 193 - 132 * real.sqrt 2) :=
sorry

end range_of_m_l108_108582


namespace part1_monotonicity_part2_find_a_l108_108959

-- Definition of the function f
def f (a x : ℝ) : ℝ := (1 / 3)^(a * x^2 - 4 * x + 3)

-- Part (1): intervals of monotonicity when a = -1
theorem part1_monotonicity (x : ℝ) : 
  monotonicity_intervals (λ x => f (-1 : ℝ) x)
  (-∞, -2) (-2, +∞) := sorry

-- Part (2): find value of a when f(x) has a maximum value of 3
theorem part2_find_a (h_max : ∃ x, f a x = 3) : 
  a = 1 := sorry

end part1_monotonicity_part2_find_a_l108_108959


namespace arrangement_girls_together_arrangement_boys_apart_l108_108017

-- Define the parameters of the problem.
def n_teacher := 1
def n_boys := 4
def n_girls := 2

-- The number of arrangements where the two girls must stand next to each other is 1440.
theorem arrangement_girls_together : 
  let n_elements := n_teacher + n_boys + (n_girls - 1)
  permutations n_elements × permutations n_girls = 1440 := 
by 
  sorry

-- The number of arrangements where the 4 boys must not stand next to each other is 144.
theorem arrangement_boys_apart :
  let n_elements := n_teacher + n_girls
  let gaps := n_elements + 1
  permutations n_elements × permutations n_boys = 144 := 
by 
  sorry

end arrangement_girls_together_arrangement_boys_apart_l108_108017


namespace number_of_integers_satisfying_condition_l108_108561

noncomputable def count_solutions : ℕ :=
  40200

theorem number_of_integers_satisfying_condition :
  ∃ n, (1 + ⌊(200 * n) / 201⌋ = ⌈(199 * n) / 200⌉) → count_solutions = 40200 :=
sorry

end number_of_integers_satisfying_condition_l108_108561


namespace can_form_triangle_8_6_4_l108_108095

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle_8_6_4 : can_form_triangle 8 6 4 :=
by
  unfold can_form_triangle
  simp
  exact ⟨by linarith, by linarith, by linarith⟩

end can_form_triangle_8_6_4_l108_108095


namespace sec_750_correct_l108_108904

noncomputable def sec_750_equivalent_proof : Real :=
  let deg_to_rad (d : ℝ) := d * (Real.pi / 180)
  Real.sec (deg_to_rad 750)

theorem sec_750_correct :
  sec_750_equivalent_proof = (2 * Real.sqrt 3) / 3 :=
by
  unfold sec_750_equivalent_proof
  have h1 : 750 % 360 = 30 := sorry
  have h2 : Real.cos (deg_to_rad 30) = Real.sqrt 3 / 2 := sorry
  have h3 : Real.sec (deg_to_rad 30) = 1 / (Real.cos (deg_to_rad 30)) := sorry
  rw [h1, h3]
  rw h2
  field_simp
  norm_num
  rw Real.sqrt_eq_rpow
  congr
  ring
  rw Real.sqrt_mul Real.two_ne_zero Real.sqrt_pos
  field_simp
  sorry

end sec_750_correct_l108_108904


namespace cylinder_radius_inscribed_box_l108_108484

theorem cylinder_radius_inscribed_box :
  ∀ (x y z r : ℝ),
    4 * (x + y + z) = 160 →
    2 * (x * y + y * z + x * z) = 600 →
    z = 40 - x - y →
    r = (1/2) * Real.sqrt (x^2 + y^2) →
    r = (15 * Real.sqrt 2) / 2 :=
by
  sorry

end cylinder_radius_inscribed_box_l108_108484


namespace smallest_number_divisible_l108_108795

theorem smallest_number_divisible (n : ℕ) :
  (∀ d ∈ [4, 6, 8, 10, 12, 14, 16], (n - 16) % d = 0) ↔ n = 3376 :=
by {
  sorry
}

end smallest_number_divisible_l108_108795


namespace angle_KPM_right_angle_l108_108714


theorem angle_KPM_right_angle (A B C D K M P : Point)
  (H_cyclic: cyclic A B C D)
  (H_diam: diameter C A)
  (H_projection_K: orthogonal_projection A B D K)
  (H_projection_M: orthogonal_projection C B D M)
  (H_parallel: is_parallel K P B C)
  (H_intersect: collinear A P C) : 
  angle K P M = 90 :=
sorry

end angle_KPM_right_angle_l108_108714


namespace part_a_part_b_l108_108542

/-- Define rational non-integer numbers x and y -/
structure RationalNonInteger (x y : ℚ) :=
  (h1 : x.denom ≠ 1)
  (h2 : y.denom ≠ 1)

/-- Part (a): There exist rational non-integer numbers x and y 
    such that 19x + 8y and 8x + 3y are integers -/
theorem part_a : ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x + 8*y ∈ ℤ) ∧ (8*x + 3*y ∈ ℤ) :=
by
  sorry

/-- Part (b): There do not exist rational non-integer numbers x and y 
    such that 19x^2 + 8y^2 and 8x^2 + 3y^2 are integers -/
theorem part_b : ¬ ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x^2 + 8*y^2 ∈ ℤ) ∧ (8*x^2 + 3*y^2 ∈ ℤ) :=
by
  sorry

end part_a_part_b_l108_108542


namespace intersection_of_A_and_B_l108_108602

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x < 1}

-- State the theorem that A ∩ B = {x | 0 < x < 1}
theorem intersection_of_A_and_B :
  A ∩ B = {x | 0 < x ∧ x < 1} :=
sorry

end intersection_of_A_and_B_l108_108602


namespace tangent_lines_to_circle_l108_108952

theorem tangent_lines_to_circle (x y : ℝ) (hx : x = 2 ∨ (x + 2*y - 5 = 0))
(tangent : ∀ l, (l = (x=2) ∨ l = (x + 2*y - 5 = 0)) →
is_tangent l ((y-2)^2 = 4) → passes_through l (2,4)) : hx = tangent :=
sorry

end tangent_lines_to_circle_l108_108952


namespace value_of_f_2015_5_l108_108684

-- Defining the given even function
axiom f : ℝ → ℝ
axiom even_f : ∀ x, f(x) = f(-x)
axiom periodic_f : ∀ x, f(x + 1) = -f(x)
axiom initial_condition_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(x) = x + 1

-- Statement to prove
theorem value_of_f_2015_5 : f 2015.5 = 1.5 :=
sorry

end value_of_f_2015_5_l108_108684


namespace trigonometric_expression_value_l108_108062

noncomputable def tg := Real.tan
noncomputable def ctg := (λ x : ℝ, 1 / Real.tan x)

theorem trigonometric_expression_value :
  tg (Real.pi / 8) + ctg (Real.pi / 8) +
  tg (Real.pi / 16) - ctg (Real.pi / 16) +
  tg (Real.pi / 24) + ctg (Real.pi / 24) =
  2 * (Real.sqrt 6 + Real.sqrt 2 - 1) :=
by
  sorry

end trigonometric_expression_value_l108_108062


namespace arith_seq_min_pos_l108_108310

noncomputable def arith_seq (a₀ d : ℕ → ℤ) (n : ℕ) : ℤ := a₀ n + n * d n

def conditions (a₀ d : ℕ → ℤ) : Prop := 
  (arith_seq a₀ d 10 ≠ 0) ∧
  (arith_seq a₀ d 11 / arith_seq a₀ d 10 < -1)

theorem arith_seq_min_pos (a₀ d : ℕ → ℤ) 
  (h1 : conditions a₀ d) 
  (h2 : ∃ n, arith_seq a₀ d n = 0) : 
  ∃ n = 20, S_n > 0 ∧ ∀ m < 20, S_m ≤ 0 :=
sorry

end arith_seq_min_pos_l108_108310


namespace hexagon_area_sum_l108_108697

variables (A B C D E F M X Y Z : Point)
variables [h1 : RegularHexagon A B C D E F]
variables [h2 : Midpoint M D E]
variables [h3 : Intersection X (Line A C) (Line B M)]
variables [h4 : Intersection Y (Line B F) (Line A M)]
variables [h5 : Intersection Z (Line A C) (Line B F)]
variables [h6 : Area (Hexagon A B C D E F) = 1]

theorem hexagon_area_sum :
  Area (Triangle B X C) + Area (Triangle A Y F) + Area (Triangle A B Z) - Area (Quadrilateral M X Z Y) = 0 :=
sorry

end hexagon_area_sum_l108_108697


namespace sphere_radius_equals_three_l108_108233

noncomputable def radius_of_sphere : ℝ := 3

theorem sphere_radius_equals_three {R : ℝ} (h1 : 4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) : 
  R = radius_of_sphere :=
by
  sorry

end sphere_radius_equals_three_l108_108233


namespace quadratic_complete_square_l108_108236

/-- Given quadratic expression, complete the square to find the equivalent form
    and calculate the sum of the coefficients a, h, k. -/
theorem quadratic_complete_square (a h k : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + 2 = a * (x - h)^2 + k) → a + h + k = -2 :=
by
  intro h₁
  sorry

end quadratic_complete_square_l108_108236


namespace number_of_4_edge_trips_l108_108750

-- Definitions
def is_vertex (v : ℕ) : Prop := v < 8
def edges (u v : ℕ) : Prop := (
  (u = 0 ∧ v = 1) ∨ (u = 1 ∧ v = 0) ∨
  (u = 0 ∧ v = 2) ∨ (u = 2 ∧ v = 0) ∨
  (u = 0 ∧ v = 4) ∨ (u = 4 ∧ v = 0) ∨
  (u = 1 ∧ v = 3) ∨ (u = 3 ∧ v = 1) ∨
  (u = 1 ∧ v = 5) ∨ (u = 5 ∧ v = 1) ∨
  (u = 2 ∧ v = 3) ∨ (u = 3 ∧ v = 2) ∨
  (u = 2 ∧ v = 6) ∨ (u = 6 ∧ v = 2) ∨
  (u = 3 ∧ v = 7) ∨ (u = 7 ∧ v = 3) ∨
  (u = 4 ∧ v = 5) ∨ (u = 5 ∧ v = 4) ∨
  (u = 4 ∧ v = 6) ∨ (u = 6 ∧ v = 4) ∨
  (u = 5 ∧ v = 7) ∨ (u = 7 ∧ v = 5) ∨
  (u = 6 ∧ v = 7) ∨ (u = 7 ∧ v = 6)
)

-- Main theorem
theorem number_of_4_edge_trips : ∃ n : ℕ, n = 6 ∧ 
  (∀ (t : list ℕ),
    t.length = 5 ∧ -- 4 edges mean 5 vertices in path
    (t.head = some 0) ∧ -- Start at A (vertex 0)
    (t.last = some 7) ∧ -- End at B (vertex 7)
    ∀ i : fin 4, edges (t.nth_le i _) (t.nth_le i.succ _)
  → length t = n)
:= sorry

end number_of_4_edge_trips_l108_108750


namespace range_p_l108_108198

noncomputable def a (n : ℕ) (p : ℝ) : ℝ := -n + p
def b (n : ℕ) : ℝ := 2^(n - 5)
def c (n : ℕ) (p : ℝ) : ℝ := if a n p ≤ b n then a n p else b n

theorem range_p (p : ℝ) :
  (∀ n : ℕ, n ≠ 8 → c 8 p > c n p) ↔ (12 < p ∧ p < 17) :=
sorry

end range_p_l108_108198


namespace idempotent_elements_are_zero_l108_108676

-- Definitions based on conditions specified in the problem
variables {R : Type*} [Ring R] [CharZero R]
variable {e f g : R}

def idempotent (x : R) : Prop := x * x = x

-- The theorem to be proved
theorem idempotent_elements_are_zero (h_e : idempotent e) (h_f : idempotent f) (h_g : idempotent g) (h_sum : e + f + g = 0) : 
  e = 0 ∧ f = 0 ∧ g = 0 := 
sorry

end idempotent_elements_are_zero_l108_108676


namespace find_x_solution_l108_108152

theorem find_x_solution (x : ℝ) (h : sqrt (x - 5) = 7) : x = 54 :=
sorry

end find_x_solution_l108_108152


namespace inequality_holds_for_unit_interval_l108_108340

theorem inequality_holds_for_unit_interval (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
    5 * (x ^ 2 + y ^ 2) ^ 2 ≤ 4 + (x + y) ^ 4 :=
by
    sorry

end inequality_holds_for_unit_interval_l108_108340


namespace limit_sum_of_perimeters_l108_108855

-- Initial problem conditions
def initial_side_length (s : ℝ) : ℝ := s

-- Recursive definition of side lengths
def side_length (s : ℝ) (n : ℕ) : ℝ := (2/3 : ℝ) ^ (n - 1) * s

-- Definition of the perimeter of each triangle
def perimeter (s : ℝ) (n : ℕ) : ℝ := 3 * side_length s n

-- Sum of the perimeters as an infinite series
def sum_of_perimeters (s : ℝ) : ℝ := ∑' n, perimeter s (n + 1)

-- The main theorem stating the limit of the sum of the perimeters
theorem limit_sum_of_perimeters (s : ℝ) : sum_of_perimeters s = 9 * s := 
sorry

end limit_sum_of_perimeters_l108_108855


namespace price_of_vegetable_seedlings_base_minimum_amount_spent_l108_108347

variable (x : ℝ) (m : ℕ)

/-- The price of each bundle of type A vegetable seedlings at the vegetable seedling base is 20 yuan. -/
theorem price_of_vegetable_seedlings_base : 
  (300 / x = 300 / (5 / 4) * x + 3) → x = 20 :=
begin
  intros,
  sorry
end

/-- The minimum amount spent on purchasing 100 bundles of type A and B vegetable seedlings is 2250 yuan. -/
theorem minimum_amount_spent :
  (∀ (m : ℕ), m ≤ 50 → (w = 20 * 0.9 * m + 30 * 0.9 * (100 - m)) →
    (w = -9 * m + 2700) → m ≤ 50) → 
  (min_cost = -9 * 50 + 2700) → min_cost = 2250 :=
begin
  intros,
  sorry
end

end price_of_vegetable_seedlings_base_minimum_amount_spent_l108_108347


namespace minimum_bailing_rate_is_seven_l108_108352

noncomputable def minimum_bailing_rate (shore_distance : ℝ) (paddling_speed : ℝ) 
                                       (water_intake_rate : ℝ) (max_capacity : ℝ) : ℝ := 
  let time_to_shore := shore_distance / paddling_speed
  let intake_total := water_intake_rate * time_to_shore
  let required_rate := (intake_total - max_capacity) / time_to_shore
  required_rate

theorem minimum_bailing_rate_is_seven 
  (shore_distance : ℝ) (paddling_speed : ℝ) (water_intake_rate : ℝ) (max_capacity : ℝ) :
  shore_distance = 2 →
  paddling_speed = 3 →
  water_intake_rate = 8 →
  max_capacity = 40 →
  minimum_bailing_rate shore_distance paddling_speed water_intake_rate max_capacity = 7 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end minimum_bailing_rate_is_seven_l108_108352


namespace remove_edge_from_tree_l108_108713

-- Define what it means for a graph to be a tree
structure Tree (V E : Type) :=
(adj : V → V → Prop)
(is_connected : ∀ u v : V, adj u v ∨ ∃ w, adj u w ∧ adj w v)
(acyclic : ∀ u, ¬ (adj u u))
(n_vertices : ℕ)
(edges_card : |E| = n_vertices - 1)

-- A graph is disconnected if there do not exist paths between at least one pair of nodes
def is_disconnected {V : Type} (adj : V → V → Prop) : Prop :=
∃ u v, ¬ (adj u v ∨ ∃ w, adj u w ∧ adj w v)

theorem remove_edge_from_tree {V E : Type} (T : Tree V E) (e : E) :
  is_disconnected (λ u v, T.adj u v ∧ ¬ (set.mem e (u, v))) :=
sorry

end remove_edge_from_tree_l108_108713


namespace total_cost_of_fencing_l108_108555

def P : ℤ := 42 + 35 + 52 + 66 + 40
def cost_per_meter : ℤ := 3
def total_cost : ℤ := P * cost_per_meter

theorem total_cost_of_fencing : total_cost = 705 := by
  sorry

end total_cost_of_fencing_l108_108555


namespace Kate_relies_on_dumpster_diving_Upscale_stores_discard_items_Kate_frugal_habits_l108_108724

structure Person :=
  (name : String)
  (age : Nat)
  (location : String)
  (occupation : String)

def kate : Person := {name := "Kate Hashimoto", age := 30, location := "New York", occupation := "CPA"}

-- Conditions
def lives_on_15_dollars_a_month (p : Person) : Prop := p = kate → true
def dumpster_diving (p : Person) : Prop := p = kate → true
def upscale_stores_discard_good_items : Prop := true
def frugal_habits (p : Person) : Prop := p = kate → true

-- Proof
theorem Kate_relies_on_dumpster_diving : lives_on_15_dollars_a_month kate ∧ dumpster_diving kate → true := 
by sorry

theorem Upscale_stores_discard_items : upscale_stores_discard_good_items → true := 
by sorry

theorem Kate_frugal_habits : frugal_habits kate → true := 
by sorry

end Kate_relies_on_dumpster_diving_Upscale_stores_discard_items_Kate_frugal_habits_l108_108724


namespace exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l108_108533

structure RatNonIntPair (x y : ℚ) :=
  (x_rational : x.is_rational)
  (x_not_integer : x.num ≠ x.denom)
  (y_rational : y.is_rational)
  (y_not_integer : y.num ≠ y.denom)

theorem exists_rat_nonint_sol_a :
  ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * x + int 8 * y).denom = 1 ∧ (int 8 * x + int 3 * y).denom = 1 := sorry

theorem no_exists_rat_nonint_sol_b :
  ¬ ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * (x^2) + int 8 * (y^2)).denom = 1 ∧ (int 8 * (x^2) + int 3 * (y^2)).denom = 1 := sorry

end exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l108_108533


namespace find_discount_percentage_l108_108489

-- Define the cost price
def CP : ℝ := 100

-- Define the selling prices with and without discount
def SP_with_discount : ℝ := 142.5
def SP_without_discount : ℝ := 150

-- Define the profit percentages
def Profit_with_discount : ℝ := 42.5 / 100 * CP
def Profit_without_discount : ℝ := 50 / 100 * CP

-- Percentage of discount calculation
def percentage_discount := (SP_without_discount - SP_with_discount) / SP_without_discount * 100

-- The proof statement
theorem find_discount_percentage :
  percentage_discount = 5 := by
  sorry

end find_discount_percentage_l108_108489


namespace lesha_cottage_area_l108_108296

theorem lesha_cottage_area
  (h_nonagon: ∀ D E F G H I K L M: ℝ^2, ∃ A B C: ℝ^2,
    side_eq_and_parallel (polygon D E F G H I K L M) 3)
  (h_triangle_area: ∀ A B C: ℝ^2, midpoint_triangle_area (polygon D E F G H I K L M) A B C 12) : 
  ∃ area: ℝ, area = 48 := 
sorry

end lesha_cottage_area_l108_108296


namespace num_sides_of_length4_eq_4_l108_108128

-- Definitions of the variables and conditions
def total_sides : ℕ := 6
def total_perimeter : ℕ := 30
def side_length1 : ℕ := 7
def side_length2 : ℕ := 4

-- The conditions imposed by the problem
def is_hexagon (x y : ℕ) : Prop := x + y = total_sides
def perimeter_condition (x y : ℕ) : Prop := side_length1 * x + side_length2 * y = total_perimeter

-- The proof problem: Prove that the number of sides of length 4 is 4
theorem num_sides_of_length4_eq_4 (x y : ℕ) 
    (h1 : is_hexagon x y) 
    (h2 : perimeter_condition x y) : y = 4 :=
sorry

end num_sides_of_length4_eq_4_l108_108128


namespace min_positive_value_of_Ep_l108_108302

theorem min_positive_value_of_Ep (p : ℕ) (hp : nat.prime p) (hp_odd : p % 2 = 1) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ E_p(x, y) = sqrt((p - 1)/2) + sqrt((p + 1)/2)
where
  E_p (x y: ℕ) := real.sqrt (2 * p) - real.sqrt x - real.sqrt y :=
by
  sorry

end min_positive_value_of_Ep_l108_108302


namespace zero_ordering_l108_108205

def f (x : ℝ) : ℝ := x + 2^x
def g (x : ℝ) : ℝ := x + log x
def h (x : ℝ) : ℝ := x^3 + x - 2

def x1 := {x : ℝ // f x = 0}
def x2 := {x : ℝ // g x = 0}
def x3 := {x : ℝ // h x = 0}

theorem zero_ordering (x1 : ℝ) (hx1 : f x1 = 0) (x2 : ℝ) (hx2 : g x2 = 0) (x3 : ℝ) (hx3 : h x3 = 0) :
  x1 < x2 ∧ x2 < x3 := 
sorry

end zero_ordering_l108_108205


namespace real_solutions_count_l108_108513

-- Define the system of equations
def sys_eqs (x y z w : ℝ) :=
  (x = z + w + z * w * x) ∧
  (z = x + y + x * y * z) ∧
  (y = w + x + w * x * y) ∧
  (w = y + z + y * z * w)

-- The statement of the proof problem
theorem real_solutions_count : ∃ S : Finset (ℝ × ℝ × ℝ × ℝ), (∀ t : ℝ × ℝ × ℝ × ℝ, t ∈ S ↔ sys_eqs t.1 t.2.1 t.2.2.1 t.2.2.2) ∧ S.card = 5 :=
by {
  sorry
}

end real_solutions_count_l108_108513


namespace cole_avg_speed_back_home_l108_108878

noncomputable def avg_speed_back_home 
  (speed_to_work : ℚ) 
  (total_round_trip_time : ℚ) 
  (time_to_work : ℚ) 
  (time_in_minutes : ℚ) :=
  let time_to_work_hours := time_to_work / time_in_minutes
  let distance_to_work := speed_to_work * time_to_work_hours
  let time_back_home := total_round_trip_time - time_to_work_hours
  distance_to_work / time_back_home

theorem cole_avg_speed_back_home :
  avg_speed_back_home 75 1 (35/60) 60 = 105 := 
by 
  -- The proof is omitted
  sorry

end cole_avg_speed_back_home_l108_108878


namespace leftmost_square_side_length_l108_108757

open Real

/-- Given the side lengths of three squares, 
    where the middle square's side length is 17 cm longer than the leftmost square,
    the rightmost square's side length is 6 cm shorter than the middle square,
    and the sum of the side lengths of all three squares is 52 cm,
    prove that the side length of the leftmost square is 8 cm. -/
theorem leftmost_square_side_length
  (x : ℝ)
  (h1 : ∀ m : ℝ, m = x + 17)
  (h2 : ∀ r : ℝ, r = x + 11)
  (h3 : x + (x + 17) + (x + 11) = 52) :
  x = 8 := by
  sorry

end leftmost_square_side_length_l108_108757


namespace right_triangle_area_l108_108004

theorem right_triangle_area (h : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) :
  h = 12 ∧ a = 30 ∧ b = 60 ∧ c = 90 →
  (a = 30 → h = 2 * (6 : ℝ)) →
  (12 = 2 * 6) →
  (6 = 12 / 2) →
  (18 * sqrt 3 = 1 / 2 * 6 * (6 * sqrt 3)) →
  18 * sqrt 3 = 18 * sqrt 3 :=
by sorry

end right_triangle_area_l108_108004


namespace relationship_among_a_b_c_l108_108186

noncomputable def f : ℝ → ℝ := sorry
def a : ℝ := π * f π
def b : ℝ := (-2) * f (-2)
def c : ℝ := f 1

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_inequality : ∀ {x}, x < 0 → f x + x * (deriv f x) < 0

theorem relationship_among_a_b_c : a > b ∧ b > c := sorry

end relationship_among_a_b_c_l108_108186


namespace range_of_a_l108_108187

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2*x else -(x^2 + 2*x)

theorem range_of_a (a : ℝ) :
  (-2 < a) ∧ (a < 1) ↔ f(2 - a^2) > f(a) :=
sorry

end range_of_a_l108_108187


namespace opposite_of_neg_3_is_3_l108_108388

theorem opposite_of_neg_3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg_3_is_3_l108_108388


namespace value_of_a_l108_108660

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C D E : V)
variables (a : ℝ) (hA : A ≠ B) (hB : A ≠ D) (hD : B ≠ D)

-- Conditions
axiom h_length_AB : ∥A - B∥ = a
axiom h_a_pos : 0 < a
axiom h_length_AD : ∥A - D∥ = 1
axiom h_angle_BAD : inner (B - A) (D - A) = ∥B - A∥ * ∥D - A∥ * real.cos (real.pi / 3)
axiom h_midpoint_E : E = (C + D) / 2
axiom h_dot_product : inner (C - A) (E - B) = 1

-- Question to prove
theorem value_of_a (h : h_dot_product) : a = 1 / 2 :=
sorry

end value_of_a_l108_108660


namespace min_value_of_f_in_interval_l108_108387

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) ∧ f x = -Real.sqrt 2 / 2 :=
by
  sorry

end min_value_of_f_in_interval_l108_108387


namespace total_fish_count_l108_108408

theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ)
  (h1 : num_fishbowls = 261) (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := 
  by 
    sorry

end total_fish_count_l108_108408


namespace determinant_example_l108_108374

noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)
noncomputable def sin_deg (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

-- Define the determinant of a 2x2 matrix in terms of its entries
def determinant_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Proposed theorem statement in Lean 4
theorem determinant_example : 
  determinant_2x2 (cos_deg 45) (sin_deg 75) (sin_deg 135) (cos_deg 105) = - (Real.sqrt 3 / 2) := 
by sorry

end determinant_example_l108_108374


namespace parabola_directrix_l108_108558

theorem parabola_directrix (x : ℝ) :
  (∃ y : ℝ, y = (x^2 - 8*x + 12) / 16) →
  ∃ directrix : ℝ, directrix = -17 / 4 :=
by
  sorry

end parabola_directrix_l108_108558


namespace evaluate_expression_l108_108583

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem evaluate_expression : 1 + x^4 + x^8 + x^{12} + x^{16} = 0 := sorry

end evaluate_expression_l108_108583


namespace exists_rational_non_integer_xy_no_rational_non_integer_xy_l108_108529

-- Part (a)
theorem exists_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  (∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
   ∃ z1 z2 : ℤ, 19 * x + 8 * y = ↑z1 ∧ 8 * x + 3 * y = ↑z2) :=
sorry

-- Part (b)
theorem no_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  ¬ ∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
  ∃ z1 z2 : ℤ, 19 * x^2 + 8 * y^2 = ↑z1 ∧ 8 * x^2 + 3 * y^2 = ↑z2 :=
sorry

end exists_rational_non_integer_xy_no_rational_non_integer_xy_l108_108529


namespace find_unknown_rate_l108_108056

variable {x : ℝ}

theorem find_unknown_rate (h : (3 * 100 + 1 * 150 + 2 * x) / 6 = 150) : x = 225 :=
by 
  sorry

end find_unknown_rate_l108_108056


namespace truth_of_compound_proposition_l108_108935

def p := ∃ x : ℝ, x - 2 > Real.log x
def q := ∀ x : ℝ, x^2 > 0

theorem truth_of_compound_proposition : p ∧ ¬ q :=
by
  sorry

end truth_of_compound_proposition_l108_108935


namespace estimate_double_hit_probability_l108_108100

-- The condition of the probability of hitting the bullseye is represented as 0.4
def probability_hit : ℝ := 0.4

-- The mapping from random numbers to hits and misses
def num_to_hit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- List of pairs of random numbers generated from simulations
def generated_pairs : List (ℕ × ℕ) :=
[(9, 3), (2, 8), (1, 2), (4, 5), (8, 5), (6, 9), (6, 8), (3, 4), (3, 1), (2, 5),
 (7, 3), (9, 3), (0, 2), (7, 5), (5, 6), (4, 8), (8, 7), (3, 0), (1, 1), (3, 5)]

-- The problem statement to prove
theorem estimate_double_hit_probability :
  let successful_pairs := generated_pairs.filter (λ (p : ℕ × ℕ), num_to_hit p.1 ∧ num_to_hit p.2),
      double_hit_probability := (successful_pairs.length : ℝ) / (generated_pairs.length : ℝ)
  in double_hit_probability = 0.2 :=
by
  sorry

end estimate_double_hit_probability_l108_108100


namespace directrix_of_parabola_l108_108556

-- Definition of the given parabola
def parabola (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / 16

-- The mathematical statement to prove
theorem directrix_of_parabola : ∀ x : ℝ, parabola x = y ↔ y = -5 / 4 -> sorry :=
by
  sorry

end directrix_of_parabola_l108_108556


namespace range_of_f_on_1_to_2_l108_108391

noncomputable def f (x : ℝ) := 2^x + Real.logb 2 x

theorem range_of_f_on_1_to_2 : 
  Set.range (λ x, f x) (Set.Icc 1 2) = Set.Icc 2 5 := 
sorry

end range_of_f_on_1_to_2_l108_108391


namespace power_mod_lemma_l108_108793

theorem power_mod_lemma : (7^137 % 13) = 11 := by
  sorry

end power_mod_lemma_l108_108793


namespace smallest_result_from_process_l108_108435

theorem smallest_result_from_process : 
  ∃ (a b c : ℕ), a ∈ {7, 11, 13, 17, 19, 23} ∧ b ∈ {7, 11, 13, 17, 19, 23} ∧ c ∈ {7, 11, 13, 17, 19, 23} ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  min ((a + b) * c) ((min ((a + c) * b) ((b + c) * a))) = 168 :=
by
  sorry

end smallest_result_from_process_l108_108435


namespace arithmetic_sequence_middle_term_l108_108265

theorem arithmetic_sequence_middle_term (x y z : ℝ) (h : list.nth_le [23, x, y, z, 47] 2 sorry = y) :
  y = 35 :=
sorry

end arithmetic_sequence_middle_term_l108_108265


namespace num_psafe_less_than_20000_l108_108155

def p_safe (p n : ℕ) : Prop :=
  ∀ m : ℤ, ¬ (abs (n - m * p)).natAbs ≤ 2

def simultaneously_psafe (n : ℕ) : Prop :=
  p_safe 5 n ∧ p_safe 7 n ∧ p_safe 17 n

theorem num_psafe_less_than_20000 : (finset.Icc 1 20000).filter simultaneously_psafe).card = 1584 := by
  sorry

end num_psafe_less_than_20000_l108_108155


namespace find_a_value_l108_108976

/- Define the universal set U -/
def U : Set ℕ := {1, 2, 5, 7}

/- Define the set M with variable a -/
def M (a : ℕ) : Set ℕ := {1, a - 5}

/- Define the complement of M in U -/
def M_complement (a : ℕ) : Set ℕ := U \ (M a)

theorem find_a_value : ∃ a : ℕ, M_complement a = {2, 7} ∧ M a = {1, 10 - 5} := 
by 
    existsi 10
    split
    · sorry
    · sorry

end find_a_value_l108_108976


namespace arithmetic_mean_of_range_l108_108034

-- Define the sequence of integers from -6 to 8
def integer_range : List Int := List.range' (-6) (8 - (-6) + 1)

-- Define a function to calculate the arithmetic mean
def arithmetic_mean (l : List Int) : Float :=
  l.sum.toFloat / l.length.toFloat

-- The theorem statement
theorem arithmetic_mean_of_range : arithmetic_mean integer_range = 1.0 := by
  sorry

end arithmetic_mean_of_range_l108_108034


namespace perpendicular_lines_l108_108164

variable {α β : Type} [Plane α] [Plane β]
variable {m n : Line}

axiom planes_different : α ≠ β
axiom lines_noncoincident : m ≠ n

axiom m_perp_alpha : m ⊥ α
axiom n_perp_beta : n ⊥ β
axiom alpha_perp_beta : α ⊥ β

theorem perpendicular_lines (m ⊥ α) (n ⊥ β) (α ⊥ β): m ⊥ n :=
by
  sorry

end perpendicular_lines_l108_108164


namespace smallest_positive_period_and_intervals_of_increase_value_of_cos_2x0_l108_108624

-- Given definitions based on the conditions
def f (x : ℝ) := sin x ^ 2 + 2 * sqrt 3 * sin x * cos x - 1 / 2 * cos (2 * x)

-- First part of the problem: period and monotonic intervals
theorem smallest_positive_period_and_intervals_of_increase :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π) ∧
  (∀ k : ℤ, ∃ a b : ℝ, a = k * π - π / 6 ∧ b = k * π + π / 3 ∧
    ∀ x, a ≤ x ∧ x ≤ b → monotonic_increasing_on f x) :=
sorry

-- Second part of the problem: value of cos 2x_0
theorem value_of_cos_2x0 (x0 : ℝ) (hx0 : 0 ≤ x0 ∧ x0 ≤ π / 2) (hfx0 : f x0 = 0) :
  cos (2 * x0) = (3 * sqrt 5 + 1) / 8 :=
sorry

end smallest_positive_period_and_intervals_of_increase_value_of_cos_2x0_l108_108624


namespace probability_sum_is_odd_l108_108575

theorem probability_sum_is_odd (S : Finset ℕ) (h_S : S = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
    (h_card : S.card = 12) :
  let choices := S.powerset.filter (λ t, t.card = 4),
      odd_sum_choices := choices.filter (λ t, t.sum % 2 = 1) in
  (odd_sum_choices.card : ℚ) / choices.card = 1 / 3 :=
by  
  sorry

end probability_sum_is_odd_l108_108575


namespace wreaths_per_greek_l108_108896

variable (m : ℕ) (m_pos : m > 0)

theorem wreaths_per_greek : ∃ x, x = 4 * m := 
sorry

end wreaths_per_greek_l108_108896


namespace sum_floor_half_eq_1022121_l108_108168

-- Definition of the function f satisfying the condition
def f (x : ℝ) := (1 : ℝ) / 2

-- The condition that f satisfies for any real numbers x, y, z
axiom condition (x y z : ℝ) : f(x * y) + f(x * z) - 2 * f(x) * f(y * z) ≥ 1 / 2

-- Main theorem statement
theorem sum_floor_half_eq_1022121 : 
    ∑ i in (Finset.range 2022).map (Finset.Nat.cast), ⌊(i : ℝ / 2)⌋ = 1022121 := 
by sorry

end sum_floor_half_eq_1022121_l108_108168


namespace quadratic_completing_square_b_plus_c_l108_108761

theorem quadratic_completing_square_b_plus_c :
  ∃ b c : ℤ, (λ x : ℝ, x^2 - 24 * x + 50) = (λ x, (x + b)^2 + c) ∧ b + c = -106 :=
by
  sorry

end quadratic_completing_square_b_plus_c_l108_108761


namespace isosceles_trapezoid_area_l108_108509

/-- Define the vertices of the trapezoid as points in the Euclidean plane -/
structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 0 0
def B := Point.mk 8 0
def C := Point.mk 6 10
def D := Point.mk 2 10

/-- Distance between two points in the Euclidean plane -/
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

/-- Check if the given points form an isosceles trapezoid and compute the area -/
noncomputable def area_of_trapezoid (A B C D : Point) : ℝ :=
  let base1 := distance A B
  let base2 := distance D C
  let height := D.y - A.y
  (1 / 2) * (base1 + base2) * height

theorem isosceles_trapezoid_area :
  let A := Point.mk 0 0
  let B := Point.mk 8 0
  let C := Point.mk 6 10
  let D := Point.mk 2 10
  area_of_trapezoid A B C D = 60 :=
by
  sorry

end isosceles_trapezoid_area_l108_108509


namespace line_equation_l108_108616

theorem line_equation (k b : ℝ) (h_k : k = 2) (h_b : b = 1) : ∀ x : ℝ, (∃ y : ℝ, y = k * x + b ↔ y = 2 * x + 1) :=
by
  intros x
  use 2 * x + 1
  split
  { intro h
    rw [h_k, h_b] at h
    exact h },
  { intro h
    rw [h_k, h_b]
    exact h }
 
end line_equation_l108_108616


namespace empty_atm_l108_108300

theorem empty_atm (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : a 9 < b 9)
    (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ 8 → a k ≠ b k) 
    (n : ℕ) (h₀ : n = 1) : 
    ∃ (sequence : ℕ → ℕ), (∀ i, sequence i ≤ n) → (∀ k, ∃ i, k > i → sequence k = 0) :=
sorry

end empty_atm_l108_108300


namespace product_divisibility_l108_108301

theorem product_divisibility (k m n : ℕ) (p : ℕ) 
  (hp : p = m + k + 1) (prime_p : Nat.Prime p) (p_gt_n1 : p > n + 1)
  (c : ℕ → ℕ) (hc : ∀ s, c s = s * (s + 1)) :
  (∏ i in Finset.range n, (c (m + i + 1) - c k)) ∣ (∏ i in Finset.range n, c (i + 1)) :=
by
  sorry

end product_divisibility_l108_108301


namespace min_pressure_l108_108327

-- Constants and assumptions
variables (V0 T0 a b c R : ℝ)
hypothesis h1 : c^2 < a^2 + b^2

-- Functions
def cyclic_process (V T : ℝ) : Prop :=
  (V / V0 - a)^2 + (T / T0 - b)^2 = c^2

def ideal_gas_law (P V T : ℝ) : Prop :=
  T = P * V / R

-- Minimum pressure
theorem min_pressure (P_min : ℝ) : 
  (∃ V T, cyclic_process V T ∧ ideal_gas_law P_min V T) → 
  P_min = (R * T0 / V0) * (a * sqrt (a^2 + b^2 - c^2) - b * c) / 
                       (b * sqrt (a^2 + b^2 - c^2) + a * c) :=
sorry

end min_pressure_l108_108327


namespace betty_total_cost_l108_108108

theorem betty_total_cost :
    (6 * 2.5) + (4 * 1.25) + (8 * 3) = 44 :=
by
    sorry

end betty_total_cost_l108_108108


namespace Vasechkin_result_l108_108334

-- Define the operations
def P (x : ℕ) : ℕ := (x / 2 * 7) - 1001
def V (x : ℕ) : ℕ := (x / 8) ^ 2 - 1001

-- Define the proposition
theorem Vasechkin_result (x : ℕ) (h_prime : P x = 7) : V x = 295 := 
by {
  -- Proof is omitted
  sorry
}

end Vasechkin_result_l108_108334


namespace max_star_value_l108_108694

def star (a b : ℝ) : ℝ := sin a * cos b

theorem max_star_value (x y : ℝ) (h : star x y - star y x = 1) : 
  ∃ (M : ℝ), (M = 1) ∧ ∀ z, (star x y + star y x) ≤ M :=
sorry

end max_star_value_l108_108694


namespace tangents_from_external_point_l108_108116

theorem tangents_from_external_point
  (O A B C : Point)
  (ω' : Circle)
  (h_ω'_radius : ω'.radius = 4)
  (h_O_center : ω'.center = O)
  (h_OA : dist O A = 15)
  (h_AB_tangent : is_tangent_from A B ω')
  (h_AC_tangent : is_tangent_from A C ω')
  (h_BC_tangent_line : is_tangent_line B C ω')
  (h_BC : dist B C = 8) :
  dist A B + dist A C = 2 * (Real.sqrt 209) + 8 := 
sorry

end tangents_from_external_point_l108_108116


namespace domain_of_f_l108_108738

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - real.log x)

theorem domain_of_f :
  {x : ℝ | 0 < x ∧ 1 - real.log x ≥ 0} = {x : ℝ | 0 < x ∧ x ≤ real.exp 1} :=
by
  sorry

end domain_of_f_l108_108738


namespace sean_total_spending_l108_108546

noncomputable def cost_first_bakery_euros : ℝ :=
  let almond_croissants := 2 * 4.00
  let salami_cheese_croissants := 3 * 5.00
  let total_before_discount := almond_croissants + salami_cheese_croissants
  total_before_discount * 0.90 -- 10% discount

noncomputable def cost_second_bakery_pounds : ℝ :=
  let plain_croissants := 3 * 3.50 -- buy-3-get-1-free
  let focaccia := 5.00
  let total_before_tax := plain_croissants + focaccia
  total_before_tax * 1.05 -- 5% tax

noncomputable def cost_cafe_dollars : ℝ :=
  let lattes := 3 * 3.00
  lattes * 0.85 -- 15% student discount

noncomputable def first_bakery_usd : ℝ :=
  cost_first_bakery_euros * 1.15 -- converting euros to dollars

noncomputable def second_bakery_usd : ℝ :=
  cost_second_bakery_pounds * 1.35 -- converting pounds to dollars

noncomputable def total_cost_sean_spends : ℝ :=
  first_bakery_usd + second_bakery_usd + cost_cafe_dollars

theorem sean_total_spending : total_cost_sean_spends = 53.44 :=
  by
  -- The proof can be handled here
  sorry

end sean_total_spending_l108_108546


namespace tyler_meal_choices_l108_108030

theorem tyler_meal_choices : 
  let meats := 4
  let vegetables := 4
  let desserts := 4
  let drinks := 2
  let meat_combinations := Nat.choose meats 2
  let vegetable_combinations := Nat.choose vegetables 2

  (meat_combinations * vegetable_combinations * desserts * drinks) = 288 :=
by
  let meats := 4
  let vegetables := 4
  let desserts := 4
  let drinks := 2
  let meat_combinations := Nat.choose meats 2
  let vegetable_combinations := Nat.choose vegetables 2
  have meat_combinations_calc := Nat.choose_eq meats 2
  have vegetable_combinations_calc := Nat.choose_eq vegetables 2
  calc
    (meat_combinations * vegetable_combinations * desserts * drinks)
      = (Nat.choose 4 2 * Nat.choose 4 2 * 4 * 2) : by rw [meat_combinations_calc, vegetable_combinations_calc]
  ... = (6 * 6 * 4 * 2) : by norm_num
  ... = 288 : by norm_num

end tyler_meal_choices_l108_108030


namespace shaded_area_l108_108829

noncomputable def pi := Real.pi

def circleArea (r : ℝ) : ℝ := pi * r^2

def isTangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  let dist := ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2).sqrt
  dist = r1 + r2

def A := (0, 0)    -- Center of the smaller circle
def rA := 4        -- Radius of the smaller circle
def B := (5, 0)    -- Center of the larger circle
def rB := 9        -- Radius of the larger circle

theorem shaded_area : isTangent A B rA rB → 
  (circleArea rB - circleArea rA) = 65 * pi :=
by
  intro h
  sorry

end shaded_area_l108_108829


namespace coefficient_of_x5_in_expansion_l108_108035

theorem coefficient_of_x5_in_expansion :
  (∃ (c : ℝ), c = 1344 * real.sqrt 3 ∧ 
  (∀ x : ℝ, polynomial.eval x ((x + 2 * real.sqrt 3)^8).coeff 5 = c)) := 
sorry

end coefficient_of_x5_in_expansion_l108_108035


namespace complex_calculation_l108_108511

def complex_add (a b : ℂ) : ℂ := a + b
def complex_mul (a b : ℂ) : ℂ := a * b

theorem complex_calculation :
  let z1 := (⟨2, -3⟩ : ℂ)
  let z2 := (⟨4, 6⟩ : ℂ)
  let z3 := (⟨-1, 2⟩ : ℂ)
  complex_mul (complex_add z1 z2) z3 = (⟨-12, 9⟩ : ℂ) :=
by 
  sorry

end complex_calculation_l108_108511


namespace ellipse_equation_fixed_point_l108_108931

def ellipse (a b : ℝ) := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def focus (a b : ℝ) : ℝ := real.sqrt (1 - (b^2 / a^2))

def chord_intersect (a b : ℝ) : Prop :=
  k : ℝ, m : ℝ, f := (focus a b),
  (a > 0 ∧ b > 0 ∧ a > b ∧ f = 1 / 2 ∧ m = 3 / (2 * b^2 / a)) ⊢ 
  line : ℝ → ℝ → Prop := λ x y, (x - f) = (y * m + f),

theorem ellipse_equation :
  ∀ a b : ℝ, a = 2 ∧ b = sqrt 3 → ellipse a b = ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 :=
sorry

theorem fixed_point (k : ℝ) :
  ∀ a b: ℝ, a = 2 ∧ b = sqrt 3 ∧
  (k * x + m = k (f, 0) ∧ m ∈ { -2 * k, -2 * k / 7 }) → line x (k x + m) ∧ (0, 2 / 7) :=
sorry

end ellipse_equation_fixed_point_l108_108931


namespace problem_statement_negation_statement_l108_108208

variable {a b : ℝ}

theorem problem_statement (h : a * b ≤ 0) : a ≤ 0 ∨ b ≤ 0 :=
sorry

theorem negation_statement (h : a * b > 0) : a > 0 ∧ b > 0 :=
sorry

end problem_statement_negation_statement_l108_108208


namespace calculate_ratio_l108_108895

-- Definitions of the probabilities involved
def total_configurations : ℕ := Nat.choose 24 4

def p' : ℚ :=
  let choices_for_3_balls := 5
  let choices_for_6_balls := 4
  let remaining_configurations := rat_of_nat (11.factorial / (4.factorial * 4.factorial * 4.factorial * 3.factorial))
  rat_of_nat (choices_for_3_balls * choices_for_6_balls) * remaining_configurations

def q : ℚ :=
  let total_permutations := rat_of_nat (20.factorial / (4.factorial * 4.factorial * 4.factorial * 4.factorial * 4.factorial))
  total_permutations

def ratio : ℚ := p' / q

theorem calculate_ratio : p' / q = 8 / 57 := by
  sorry -- Proof to be done

end calculate_ratio_l108_108895


namespace largest_three_digit_congruent_to_twelve_mod_fifteen_l108_108036

theorem largest_three_digit_congruent_to_twelve_mod_fifteen :
  ∃ n : ℕ, 100 ≤ 15 * n + 12 ∧ 15 * n + 12 < 1000 ∧ (15 * n + 12 = 987) :=
sorry

end largest_three_digit_congruent_to_twelve_mod_fifteen_l108_108036


namespace slant_asymptote_sum_l108_108881

noncomputable def function := λ x : ℝ, (3 * x^2 - 2 * x - 5) / (x - 4)

theorem slant_asymptote_sum : 
  (let y := λ x : ℝ, (3 * x^2 - 2 * x - 5) / (x - 4) in
  (∀ x : ℝ, x ≠ 4 → 
  (y x - (3 * x + 10) → 0) as x → ∞)) ∧ 
  (∀ x : ℝ, x ≠ 4 → 
  (y x - (3 * x + 10) → 0) as x → -∞)) ∧ 
  3 + 10 = 13 :=
begin
  sorry
end

end slant_asymptote_sum_l108_108881


namespace rearrange_squares_into_one_square_l108_108215

theorem rearrange_squares_into_one_square 
  (a b : ℕ) (h_a : a = 3) (h_b : b = 1) 
  (parts : Finset (ℕ × ℕ)) 
  (h_parts1 : parts.card ≤ 3)
  (h_parts2 : ∀ p ∈ parts, p.1 * p.2 = a * a ∨ p.1 * p.2 = b * b)
  : ∃ c : ℕ, (c * c = (a * a) + (b * b)) :=
by
  sorry

end rearrange_squares_into_one_square_l108_108215


namespace dice_probability_l108_108033

theorem dice_probability (n : ℕ) : 
  ∃ S : ℕ, 
    (5 ∣ n → S = (6 ^ n + 4) / 5) ∧ 
    (¬5 ∣ n → S = (6 ^ n - 1) / 5) ∧ 
    (probability (total_eyes_rolled n) = S / 6 ^ n) :=
begin
  sorry,
end

end dice_probability_l108_108033


namespace parametric_second_derivative_l108_108814

noncomputable def y''_xx (t : ℝ) : ℝ :=
-1 / (1 - cos t)^2

theorem parametric_second_derivative (t : ℝ) :
  let x := t - sin t in
  let y := 2 - cos t in
  deriv (λ t, deriv (λ t, y) t / deriv (λ t, x) t) t / deriv (λ t, x) t =
  y''_xx t :=
by
  let x := t - sin t
  let y := 2 - cos t
  sorry

end parametric_second_derivative_l108_108814


namespace advertised_mileage_l108_108339

theorem advertised_mileage (tank_capacity : ℕ) (total_miles : ℕ) (mileage_difference : ℕ) 
  (h1 : tank_capacity = 12) (h2 : total_miles = 372) (h3 : mileage_difference = 4) : 
  let actual_mileage := total_miles / tank_capacity in
  let advertised_mileage := actual_mileage + mileage_difference in
  advertised_mileage = 35 := 
by
  sorry

end advertised_mileage_l108_108339


namespace sqrt_diff_inequality_l108_108163

theorem sqrt_diff_inequality (k : ℕ) (hk : k ≥ 2) :
  (real.sqrt k - real.sqrt (k - 1) > real.sqrt (k + 1) - real.sqrt k) :=
sorry

end sqrt_diff_inequality_l108_108163


namespace limit_position_of_R_l108_108592

open Set

variables {e : Set (ℝ × ℝ)} {A B R : ℝ × ℝ} {d b : ℝ}

def line_passes_through (p1 p2 : ℝ × ℝ) (line : Set (ℝ × ℝ)) : Prop :=
  ∃ l u : ℝ, (line = λ t, (l * t + u)) ∧ (p1 ∈ line) ∧ (p2 ∈ line)

theorem limit_position_of_R (e_line : line_passes_through (0, 0) (1, 0) e)
    (A_coords : A = (d, a)) (B_coords : B = (0, b)) :
  ∃ (R_lim : ℝ × ℝ), limit_position R P e R_lim :=
sorry

end limit_position_of_R_l108_108592


namespace digit_theta_l108_108429

noncomputable def theta : ℕ := 7

theorem digit_theta (Θ : ℕ) (h1 : 378 / Θ = 40 + Θ + Θ) : Θ = theta :=
by {
  sorry
}

end digit_theta_l108_108429


namespace train_length_approx_l108_108806

theorem train_length_approx (speed_kmph : ℝ) (time_sec : ℝ) (length_approx : ℝ) :
  speed_kmph = 60 ∧ time_sec = 4 ∧ length_approx = 66.68 →
  (speed_kmph * 1000 / 3600) * time_sec ≈ length_approx :=
by
  sorry

end train_length_approx_l108_108806


namespace parallel_planes_l108_108951

section ProofProblem

variables {α β γ : Plane} {a b : Line}

-- Conditions
variable (H1 : Parallel a α)
variable (H2 : Parallel a β)
variable (H3 : Parallel a γ)
variable (H4 : Parallel b α)
variable (H5 : Parallel b β)
variable (H6 : Parallel b γ)

-- Theorem statement
theorem parallel_planes (H1 : Parallel a α) (H2 : Parallel a β)
                        (H3 : Parallel a γ) (H4 : Parallel b α)
                        (H5 : Parallel b β) (H6 : Parallel b γ) :
  (Parallel α β) ↔ (Parallel α γ ∧ Parallel β γ) :=
sorry

end ProofProblem

end parallel_planes_l108_108951


namespace ana_winning_strategy_l108_108099

theorem ana_winning_strategy (n : ℕ) (h : n ≥ 1) : 
  n = 1 ∨ n = 3 ∨ (n > 3 ∧ n % 2 = 0) ↔
  ∃ strategy : (ℕ → ℕ) → ℕ, (∀ (state : ℕ → ℕ), state 0 = n → (state (strategy state) = 0 → player_wins Ana)) :=
sorry

end ana_winning_strategy_l108_108099


namespace integral_sin_cos_eq_two_l108_108874

theorem integral_sin_cos_eq_two : 
  ∫ x in -Real.pi / 2 .. Real.pi / 2, (Real.sin x + Real.cos x) = 2 := 
by 
  sorry 

end integral_sin_cos_eq_two_l108_108874


namespace max_value_proof_l108_108184

noncomputable def max_value_b_minus_a (a b : ℝ) : ℝ :=
  b - a

theorem max_value_proof (a b : ℝ) (h1 : a < 0) (h2 : ∀ x, (x^2 + 2017 * a) * (x + 2016 * b) ≥ 0) : max_value_b_minus_a a b ≤ 2017 :=
sorry

end max_value_proof_l108_108184


namespace infinitely_many_routes_to_n_iff_odd_l108_108487

theorem infinitely_many_routes_to_n_iff_odd (n : ℤ) : 
  (∃ (routes : ℕ → Prop), (∀ k, routes k → 
  ∃ (steps : ℕ → ℤ) (h : ∀ s, steps s = 2^(s-1) ∨ steps s = -(2^(s-1))), 
  (sum (λ s, steps s) k = n)) ) ↔ (n % 2 = 1 ∨ n % 2 = -1) := 
sorry

end infinitely_many_routes_to_n_iff_odd_l108_108487


namespace year_C_passed_away_l108_108242

theorem year_C_passed_away :
  ∃ y : ℕ, 
    y = 1986 ∧ 
    (∃ a b c A_birth B_birth C_birth : ℕ,
      A_birth + 40 = B_birth ∧
      b = 40 + c ∧
      A_birth + 50 = 1980 ∧
      A_birth + 10 = C_birth ∧
      ∀ (n : ℕ), A_birth + n = B_birth + n ∧ 
      B_birth + n = C_birth + n) and
    (y = b) and
    (1980 - 40).

end year_C_passed_away_l108_108242


namespace james_choices_count_l108_108290

-- Define the conditions as Lean definitions
def isAscending (a b c d e : ℕ) : Prop := a < b ∧ b < c ∧ c < d ∧ d < e

def inRange (a b c d e : ℕ) : Prop := a ≤ 8 ∧ b ≤ 8 ∧ c ≤ 8 ∧ d ≤ 8 ∧ e ≤ 8

def meanEqualsMedian (a b c d e : ℕ) : Prop :=
  (a + b + c + d + e) / 5 = c

-- Define the problem statement
theorem james_choices_count :
  ∃ (s : Finset (ℕ × ℕ × ℕ × ℕ × ℕ)), 
    (∀ (a b c d e : ℕ), (a, b, c, d, e) ∈ s ↔ isAscending a b c d e ∧ inRange a b c d e ∧ meanEqualsMedian a b c d e) ∧
    s.card = 10 :=
sorry

end james_choices_count_l108_108290


namespace last_two_digits_of_7_pow_5_pow_6_l108_108133

theorem last_two_digits_of_7_pow_5_pow_6 : (7 ^ (5 ^ 6)) % 100 = 7 := 
  sorry

end last_two_digits_of_7_pow_5_pow_6_l108_108133


namespace perimeter_triangle_l108_108880

def ellipse (x y : ℝ) := x^2 / 25 + y^2 / 16 = 1

def foci := (F1 F2 : ℝ × ℝ) (F1 ≠ F2)

def line_intersects_ellipse (F2 : ℝ × ℝ) (A B : ℝ × ℝ) 
  (F2_line : ∃ m b : ℝ, ∀ x : ℝ, F2.2 = m * F2.1 + b ∧ 
  A.2 = m * A.1 + b ∧ B.2 = m * B.1 + b) : Prop := 
    ellipse A.1 A.2 ∧ ellipse B.1 B.2

theorem perimeter_triangle (F1 F2 A B : ℝ × ℝ)
  (h1 : foci F1 F2)
  (h2 : line_intersects_ellipse F2 A B)
  : dist F1 A + dist F1 B + dist A B = 20 :=
sorry

end perimeter_triangle_l108_108880


namespace john_max_correct_answers_l108_108081

theorem john_max_correct_answers 
  (c w b : ℕ) -- define c, w, b as natural numbers
  (h1 : c + w + b = 30) -- condition 1: total questions
  (h2 : 4 * c - 3 * w = 36) -- condition 2: scoring equation
  : c ≤ 12 := -- statement to prove
sorry

end john_max_correct_answers_l108_108081


namespace isosceles_triangle_sin_cos_rational_l108_108611

theorem isosceles_triangle_sin_cos_rational
  (a h : ℤ) -- Given BC and AD as integers
  (c : ℚ)  -- AB = AC = c
  (ha : 4 * c^2 = 4 * h^2 + a^2) : -- From c^2 = h^2 + (a^2 / 4)
  ∃ (sinA cosA : ℚ), 
    sinA = (a * h) / (h^2 + (a^2 / 4)) ∧
    cosA = (2 * h^2) / (h^2 + (a^2 / 4)) - 1 :=
sorry

end isosceles_triangle_sin_cos_rational_l108_108611


namespace concurrency_of_AA2_BB2_CC2_l108_108828

open EuclideanGeometry

/- Define the sets of points and the conditions given in the problem. -/
variables {A B C C1 C2 B1 B2 A1 A2 P : Point}
variables (hC1 : C1 ∈ (circle A B C))
variables (hC2 : C2 ∈ (circle A B C))
variables (hB1 : B1 ∈ (circle A B C))
variables (hB2 : B2 ∈ (circle A B C))
variables (hA1 : A1 ∈ (circle A B C))
variables (hA2 : A2 ∈ (circle A B C))
variables (hC1_AB : C1 ∈ segment A B)
variables (hC2_AB : C2 ∈ segment A B)
variables (hB1_CA : B1 ∈ segment C A)
variables (hB2_CA : B2 ∈ segment C A)
variables (hA1_BC : A1 ∈ segment B C)
variables (hA2_BC : A2 ∈ segment B C)
variables (h_concurrent_A1 : concurrent (line A A1) (line B B1) (line C C1))

theorem concurrency_of_AA2_BB2_CC2 :
  concurrent (line A A2) (line B B2) (line C C2) :=
sorry

end concurrency_of_AA2_BB2_CC2_l108_108828


namespace number_of_packs_bought_l108_108670

-- Variables for the problem
variable (amount_paid : ℕ) (change_received : ℕ) (cost_per_pack : ℕ)

-- Conditions given in the problem
def conditions (amount_paid change_received cost_per_pack : ℕ) : Prop :=
  amount_paid = 20 ∧ change_received = 11 ∧ cost_per_pack = 3

-- Question to prove
theorem number_of_packs_bought (amount_paid change_received cost_per_pack : ℕ) :
  conditions amount_paid change_received cost_per_pack →
  (amount_paid - change_received) / cost_per_pack = 3 :=
by
  intros h
  have h1 : amount_paid - change_received = 9, by
    { cases h, simp, }
  rw h1
  have h2 : 9 / cost_per_pack = 3, by
    { cases h, simp, }
  exact h2

end number_of_packs_bought_l108_108670


namespace angle_bisector_theorem_l108_108313

-- Define the problem and its conditions
variable {A B C K L : Type}
variable [Segment A B] [Segment A C] [Segment B C]
variable [Segment A K] [Segment B K] [Segment K C]
variable [Segment A L] [Segment L K]

-- Define angle bisector condition
def angle_bisector (A B C K : Type) : Prop :=
  ∃ (P : Type), Segment B P ∧ Segment P C ∧ (∃ r : ℝ, r = dist A P)

-- Define incenter condition
def incenter (A B C L : Type) : Prop :=
  ∃ (Q : Type), Segment A Q ∧ Segment Q B ∧ Segment Q C ∧ (∃ s : ℝ, s = dist Q P)

-- Lean statement of the problem
theorem angle_bisector_theorem (ABC : Type)
  (h1 : angle_bisector A B C K)
  (h2 : incenter A B C L) :
  dist A L > dist L K :=
sorry

end angle_bisector_theorem_l108_108313


namespace quadratic_form_correct_l108_108747

noncomputable theory

def optionA : Prop := ∃ (b c : ℝ), (λ x : ℝ => x^2 + b*x + c) = 0
def optionB : Prop := ∃ (a b c : ℝ), (λ x : ℝ => a*x^2 + b*x + c) = 0
def optionC : Prop := ∃ (a b c : ℝ), (0 < a ∧ (λ x : ℝ => a*x^2 + b*x + c) = 0)
def optionD : Prop := ¬ optionA ∧ ¬ optionB ∧ ¬ optionC

theorem quadratic_form_correct : optionC :=
sorry

end quadratic_form_correct_l108_108747


namespace arithmetic_sequence_y_l108_108261

theorem arithmetic_sequence_y : 
  ∀ (x y z : ℝ), (23 : ℝ), x, y, z, (47 : ℝ) → 
  (y = (23 + 47) / 2) → y = 35 :=
by
  intro x y z h1
  intro h2
  simp at *
  sorry

end arithmetic_sequence_y_l108_108261


namespace sphere_radius_eq_three_l108_108230

theorem sphere_radius_eq_three (R : ℝ) :
  4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3 → R = 3 :=
by
  intro h
  have h_eq : R^2 = (1 / 3) * R^3 := by
    have h_canceled : R^2 = (1 / 3) * R^3 := by
      -- simplify the given condition to the core relation
      sorry
  have h_nonzero : R ≠ 0 := by
    -- argument ensuring R is nonzero
    sorry
  have h_final : R = 3 := by
    -- deduce the radius
    sorry
  exact h_final

end sphere_radius_eq_three_l108_108230


namespace mary_circus_change_l108_108701

theorem mary_circus_change :
  let mary_ticket := 2
  let child_ticket := 1
  let num_children := 3
  let total_cost := mary_ticket + num_children * child_ticket
  let amount_paid := 20
  let change := amount_paid - total_cost
  change = 15 :=
by
  let mary_ticket := 2
  let child_ticket := 1
  let num_children := 3
  let total_cost := mary_ticket + num_children * child_ticket
  let amount_paid := 20
  let change := amount_paid - total_cost
  sorry

end mary_circus_change_l108_108701


namespace times_to_reach_below_2_from_200_l108_108359

def floor_divide_by_3 (n : ℕ) : ℕ := n / 3

def iterate_floor_divide (n : ℕ) (iterations : ℕ) : ℕ :=
match iterations with
| 0     => n
| (k+1) => floor_divide_by_3 (iterate_floor_divide n k)
end

theorem times_to_reach_below_2_from_200 : ∃ k : ℕ, iterate_floor_divide 200 k < 2 ∧ k = 5 := by
  sorry

end times_to_reach_below_2_from_200_l108_108359


namespace complement_P_in_U_l108_108975

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4}
noncomputable def P : Set ℕ := {x | -1 < x ∧ x < 3}

theorem complement_P_in_U : (U \ P) = {3, 4} :=
by
  -- Assertions or construction for the proof goes here
  sorry

end complement_P_in_U_l108_108975


namespace initial_pairs_of_shoes_l108_108706

theorem initial_pairs_of_shoes (x : ℕ) (h1 : x = (.70 * x) + 6 + 62) : x = 80 :=
sorry

end initial_pairs_of_shoes_l108_108706


namespace sequence_value_l108_108927

noncomputable def f : ℝ → ℝ := sorry

theorem sequence_value :
  ∃ a : ℕ → ℝ, 
    (a 1 = f 1) ∧ 
    (∀ n : ℕ, f (a (n + 1)) = f (2 * a n + 1)) ∧ 
    (a 2017 = 2 ^ 2016 - 1) := sorry

end sequence_value_l108_108927


namespace triangle_area_l108_108783

-- Define the necessary conditions and hypothesis
variable {ABC : Type} [triangle : ABC] [right_triangle : is_right_triangle ABC]
variable {A B C O : Point}
variable {angle_BAC : angle_degrees A B C = 30}
variable {angle_ACB : angle_degrees A C B = 60}
variable {O_center : is_incenter O A B C}
variable {circle_area : carre (circle_radius O ABC) * π = 4 * π}

-- The main theorem statement
theorem triangle_area (ABC : triangle)
  (right_triangle : is_right_triangle ABC)
  (angle_BAC : angle_degrees A B C = 30)
  (angle_ACB : angle_degrees A C B = 60)
  (O_center : is_incenter O A B C)
  (circle_area : carre (circle_radius O ABC) * π = 4 * π) :
  triangle_area ABC = 2 * sqrt 3 :=
sorry

end triangle_area_l108_108783


namespace max_element_n_l108_108821

def X : Set ℕ := {x | x > 0}  -- A set of 9 positive integers
def E (X : Set ℕ) : Set (Set ℕ) := {e | e ⊆ X} -- Subsets of X
def S (E : Set ℕ) : ℕ := E.sum -- Sum of elements of a set E

theorem max_element_n (X : Set ℕ) (hX : X.card = 9) (hn : ∀ x ∈ X, x ≤ 60) 
  (h_same_sum_subsets : ∃ A B ∈ E X, A ≠ B ∧ S A = S B) : 
  ∃ n ∈ X, n = 60 :=
sorry

end max_element_n_l108_108821


namespace sec_minus_tan_l108_108905

theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 :=
sorry

end sec_minus_tan_l108_108905


namespace solution_set_l108_108375

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := deriv f x

-- Definitions for conditions
axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom f_at_0 : f 0 = 2
axiom f_ineq : ∀ x : ℝ, f x + f' x > 1

-- Theorem statement
theorem solution_set (x: ℝ) : (e^x * f x > e^x + 1) ↔ (0 < x) :=
by 
  sorry

end solution_set_l108_108375


namespace same_number_of_acquaintances_l108_108712

theorem same_number_of_acquaintances (n : ℕ) (h : n ≥ 2) (acquaintances : Fin n → Fin n) :
  ∃ i j : Fin n, i ≠ j ∧ acquaintances i = acquaintances j :=
by
  -- Insert proof here
  sorry

end same_number_of_acquaintances_l108_108712


namespace ellipse_equation_area_range_l108_108606

open Real

-- Conditions
def vertex_A : (ℝ × ℝ) := (-2, 0)

def ellipse (a b : ℝ) : set (ℝ × ℝ) := 
  {p | ∃ x y, p = (x, y) ∧ x^2 / a^2 + y^2 / b^2 = 1}

def vertical_line : set (ℝ × ℝ) := 
  {p | p.1 = 1}

def intersections : set (ℝ × ℝ) :=
  {p | p ∈ ellipse 2 (sqrt 3) ∧ p ∈ vertical_line}

def points_PQ : (ℝ × ℝ) × (ℝ × ℝ) := 
  ((1, 3/2), (1, -3/2))

def distance_PQ : ℝ :=
  3

-- Proving Parts
-- Part 1: Equation of the ellipse
theorem ellipse_equation : (x y : ℝ) -> x^2 / 4 + y^2 / 3 = 1 ↔ (x, y) ∈ ellipse 2 (sqrt 3) :=
by sorry

-- Part 2: Range of area
theorem area_range : (0 < area (triangle vertex_A points_PQ.1 points_PQ.2) ≤ 9/2) :=
by sorry

end ellipse_equation_area_range_l108_108606


namespace ratio_brown_to_green_toads_l108_108734

def num_brown_toads_per_acre (B : ℕ) : Prop := B = 200
def num_green_toads_per_acre (G : ℕ) : Prop := G = 8
def spotted_brown_toads_per_acre (S : ℕ) : Prop := S = 50
def one_quarter_spotted (B S : ℕ) : Prop := B = 4 * S

theorem ratio_brown_to_green_toads (B G S : ℕ)
  (H1 : spotted_brown_toads_per_acre S)
  (H2 : one_quarter_spotted B S)
  (H3 : num_green_toads_per_acre G) :
  B / G = 25 :=
by
  rw [num_brown_toads_per_acre] at H2
  rw [num_green_toads_per_acre] at H3
  sorry

end ratio_brown_to_green_toads_l108_108734


namespace plot_area_in_acres_l108_108839

theorem plot_area_in_acres :
  let scale_cm_to_miles : ℝ := 3
  let base1_cm : ℝ := 20
  let base2_cm : ℝ := 25
  let height_cm : ℝ := 15
  let miles_to_acres : ℝ := 640
  let area_trapezoid_cm2 := (1 / 2) * (base1_cm + base2_cm) * height_cm
  let area_trapezoid_miles2 := area_trapezoid_cm2 * (scale_cm_to_miles ^ 2)
  let area_trapezoid_acres := area_trapezoid_miles2 * miles_to_acres
  area_trapezoid_acres = 1944000 := by
    sorry

end plot_area_in_acres_l108_108839


namespace inclination_angle_range_l108_108999

theorem inclination_angle_range (θ : ℝ) : 
  ∃ (α : ℝ), α ∈ ([0, Real.pi/4] ∪ [3 * Real.pi/4, Real.pi)) ∧ ∀ x : ℝ, y = x * Real.cos θ - 1 → α = Real.arctan (Real.cos θ) :=
by
  sorry

end inclination_angle_range_l108_108999


namespace find_a_range_l108_108978

variable (a : ℝ)

def setA : Set ℝ := { x | x ^ 2 + 2 * x - 8 > 0 }
def setB : Set ℝ := { x | abs (x - a) < 5 }

theorem find_a_range (h : setA ∪ setB = Set.univ) : -3 ≤ a ∧ a ≤ 1 :=
by
  sorry

end find_a_range_l108_108978


namespace proof_problem_l108_108947

-- Definitions based on the given conditions
def f : ℝ → ℝ
def f' : ℝ → ℝ -- Derivative of f

axiom domain_f : ∀ x, 0 < x → f x ≠ 0
axiom condition1 : ∀ x, 0 < x → f x + f' x = x * log x
axiom condition2 : f (1 / Real.exp 1) = - (1 / Real.exp 1)

noncomputable def g (x : ℝ) := f x * Real.exp (x - 1)

-- Proof question in Lean statement
theorem proof_problem :
  (g (1 / Real.exp 1) > f 1) ∧
  (f e * Real.exp (e - 1) > f 1) ∧
  (∀ x, 0 < x → f' x ≥ 0) := 
sorry

end proof_problem_l108_108947


namespace cyclic_quadrilateral_AR_AP_plus_AR_AQ_AQ_plus_AS_l108_108845

open Real
open EuclideanGeometry

theorem cyclic_quadrilateral_AR_AP_plus_AR_AQ_AQ_plus_AS
    (A P Q R S : Point)
    (h_circle : Circle (A) (P) (Q) (R) (S))
    (h_angles_equal : angle P A Q = angle Q A R ∧ angle Q A R = angle R A S) :
    dist A R * (dist A P + dist A R) = dist A Q * (dist A Q + dist A S) :=
sorry

end cyclic_quadrilateral_AR_AP_plus_AR_AQ_AQ_plus_AS_l108_108845


namespace pascal_row_10_sum_l108_108872

-- Define the function that represents the sum of Row n in Pascal's Triangle
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- State the theorem to be proven
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  -- Proof is omitted
  sorry

end pascal_row_10_sum_l108_108872


namespace problem_l108_108992

variables (q r s t : ℚ) -- Define variables as rational numbers

-- Define the given conditions
def cond1 := q / r = 12
def cond2 := s / r = 8
def cond3 := s / t = 1 / 3

-- The theorem stating the problem to be proved
theorem problem (h1 : cond1) (h2 : cond2) (h3 : cond3) : t / q = 2 :=
sorry

end problem_l108_108992


namespace remainder_theorem_l108_108679

noncomputable def Q (x : ℝ) := polynomial ℝ

theorem remainder_theorem (Q : ℝ → ℝ) (h₁ : Q 23 = 47) (h₂ : Q 47 = 23) :
  ∃ (c d : ℝ), ∀ x, Q x = (x - 23) * (x - 47) * polynomial C (x - 23) + c * x + d :=
by {
  -- Definition of Remainder Theorem specifics
  have cdef : c = -1, sorry,
  have ddef : d = 70, sorry,
  use [cdef, ddef],
  intro x,
  sorry
}

end remainder_theorem_l108_108679


namespace range_of_a_l108_108440

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → x^2 - 2*x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end range_of_a_l108_108440


namespace partI_partII_l108_108963

-- Define the absolute value function
def f (x : ℝ) := |x - 1|

-- Part I: Solve the inequality f(x) - f(x+2) < 1
theorem partI (x : ℝ) (h : f x - f (x + 2) < 1) : x > -1 / 2 := 
sorry

-- Part II: Find the range of values for a such that x - f(x + 1 - a) ≤ 1 for all x in [1,2]
theorem partII (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x - f (x + 1 - a) ≤ 1) : a ≤ 1 ∨ a ≥ 3 := 
sorry

end partI_partII_l108_108963


namespace chang_total_apples_l108_108115

def sweet_apple_price : ℝ := 0.5
def sour_apple_price : ℝ := 0.1
def sweet_apple_percentage : ℝ := 0.75
def sour_apple_percentage : ℝ := 1 - sweet_apple_percentage
def total_earnings : ℝ := 40

theorem chang_total_apples : 
  (total_earnings / (sweet_apple_percentage * sweet_apple_price + sour_apple_percentage * sour_apple_price)) = 100 :=
by
  sorry

end chang_total_apples_l108_108115


namespace cartesian_equation_of_circle_sum_distances_PA_PB_l108_108259

noncomputable def polar_to_cartesian_circle (rho : ℝ → ℝ) : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩, rho (real.sqrt (x^2 + y^2)) = 2 * real.sqrt 5 * real.sin y

def parametric_equation_line (t : ℝ) : ℝ × ℝ :=
  (3 - (real.sqrt 2 / 2) * t, real.sqrt 5 - (real.sqrt 2 / 2) * t)

theorem cartesian_equation_of_circle : ∀ x y : ℝ, 
  (polar_to_cartesian_circle (λ θ, 2 * real.sqrt 5 * real.sin θ) (x, y)) → 
  x^2 + y^2 - 2 * real.sqrt 5 * y = 0 :=
by sorry

theorem sum_distances_PA_PB : 
  ∃ A B : ℝ × ℝ, ((parametric_equation_line t = A) ∧ (parametric_equation_line t = B)) → 
  (∃ P : ℝ × ℝ, P = (3, real.sqrt 5) → 
  ∑ (distances : ℝ), abs (distances) = 3 * real.sqrt 2) :=
by sorry

end cartesian_equation_of_circle_sum_distances_PA_PB_l108_108259


namespace max_value_of_a_l108_108687
noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

theorem max_value_of_a (a : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y) → a ≤ 1 := 
sorry

end max_value_of_a_l108_108687


namespace min_visible_sum_l108_108823

-- Define the setup for the 4x4x4 cube being composed of 64 normal dice
structure Die where
  opposite_faces_sum_to_seven : \(\Pi\) (a b : nat), a + b = 7 → true

structure Cube where
  faces : list (list (list Die))
  size : nat
  size_eq : size = 4

noncomputable def sum_of_visible_faces (c : Cube) : nat := sorry

theorem min_visible_sum (c : Cube) (h : c.size = 4) :
  sum_of_visible_faces c = 144 :=
by
  sorry

end min_visible_sum_l108_108823


namespace range_of_inclination_angle_l108_108996

theorem range_of_inclination_angle (theta : ℝ) :
  let k : ℝ := Real.cos theta,
      α : ℝ := Real.arctan k
  in  α ∈ set.Icc 0 (Real.pi / 4) ∪ set.Ico (3 * Real.pi / 4) Real.pi := sorry

end range_of_inclination_angle_l108_108996


namespace magic_triangle_largest_sum_l108_108247

theorem magic_triangle_largest_sum :
  ∃ (x : Fin₇ → ℤ),
  (∀ i, 17 ≤ x i ∧ x i ≤ 25) ∧
  (∀ i j k, i ≠ j → j ≠ k → k ≠ i → x i + x j + x k = 87) ∧
  ∀ y, (∀ i, 17 ≤ y i ∧ y i ≤ 25) ∧ (∀ i j k, i ≠ j → j ≠ k → k ≠ i → y i + y j + y k = y i + x j + y k) → 87 ≤ 87 :=
sorry

end magic_triangle_largest_sum_l108_108247


namespace polynomial_reciprocal_sum_l108_108126

noncomputable def reciprocal_sum_polynomial : ℚ :=
  let a := -4/7
  let b := 6/7
  in (a / b)

theorem polynomial_reciprocal_sum:
  (reciprocal_sum_polynomial 7 4 6) = -2/3 :=
sorry

end polynomial_reciprocal_sum_l108_108126


namespace solution_sum_correct_l108_108910

noncomputable def is_solution (x : ℝ) : Prop :=
  2 * cos(2 * x) * (cos(2 * x) - cos((4028 * π^2) / x)) = cos(4 * x) - 1

noncomputable def sum_of_solutions : ℝ :=
  ∑ x in {x : ℝ | is_solution x ∧ x > 0}.to_finset, x

theorem solution_sum_correct : sum_of_solutions = 4320 * π :=
sorry

end solution_sum_correct_l108_108910


namespace sixty_second_pair_is_7_5_l108_108972

-- Define the sequence generating function.
def sequence_pair : ℕ → (ℕ × ℕ)
| 1 := (1, 1)
| n :=
  let d := n + 1 in
  let k := ((d-1)*(d-2)) / 2 + 1 in
  if n ≤ ((d*(d-1)) / 2) then
    (n - k + 1, d - (n - k))
  else
    sequence_pair (n - (d - 1))

theorem sixty_second_pair_is_7_5 : sequence_pair 62 = (7, 5) :=
sorry

end sixty_second_pair_is_7_5_l108_108972


namespace min_sum_dragon_l108_108170

theorem min_sum_dragon (n : ℕ) (a : ℕ → ℕ) (h_pos : n > 0) :
  ( ∃ k m, 1 ≤ k ∧ k ≤ m ∧ m ≤ n ∧ (∑ i in Finset.Icc k m, a i) / (m - k + 1) > 1 ) →
  ∑ i in Finset.range n, a i ≥ (n + 1) / 2 :=
sorry

end min_sum_dragon_l108_108170


namespace tangent_line_exists_l108_108692

-- Definitions of the given entities
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Parabola (F : Point) (d : Line) (P : Point) : Prop :=
  let d_f : ℝ := (P.x - F.x) ^ 2 + (P.y - F.y) ^ 2
  let d_d : ℝ := (d.a * P.x + d.b * P.y + d.c ) ^ 2 / (d.a ^ 2 + d.b ^ 2)
  d_f = d_d

-- The statement to be proved
theorem tangent_line_exists (F : Point) (d : Line) (P : Point) (A : Point) (H : Parabola F d A) : 
  ∃ ell : Line, ∀ A : Point, (Parabola F d A) → ∃ C : Circle, (diameter C F A) ∧ (tangent C ell) :=
begin
sorry
end

end tangent_line_exists_l108_108692


namespace pascal_row_10_sum_l108_108871

-- Define the function that represents the sum of Row n in Pascal's Triangle
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- State the theorem to be proven
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  -- Proof is omitted
  sorry

end pascal_row_10_sum_l108_108871


namespace find_n_given_tn_l108_108518

def t : ℕ → ℚ
| 1 := 2
| (n + 1) := if (n + 1) % 2 = 0 then 2 + t ((n + 1) / 2) else 2 / t n

theorem find_n_given_tn (n : ℕ) : t n = 29 / 131 → n = 193 := by
  intros h
  sorry

end find_n_given_tn_l108_108518


namespace major_axis_length_of_ellipse_tangent_to_y_axis_l108_108854

noncomputable def ellipse_major_axis_length
  (F1 F2 : ℝ × ℝ) (hxF1 : F1 = (12, 10)) (hxF2 : F2 = (12, 50))
  (tangent_y_axis : ℝ × ℝ) (hxTangent : tangent_y_axis.1 = 0) : 
  ℝ :=
  let F1' := (-F1.1, F1.2) in
  let distance := Real.sqrt ((F2.1 - F1'.1)^2 + (F2.2 - F1'.2)^2) in
  2 * distance

theorem major_axis_length_of_ellipse_tangent_to_y_axis : 
    ellipse_major_axis_length (12, 10) (12, 50) (12, 50) (0, _ /* arbitrary yT here but doesn't affect computation */) = 16 * Real.sqrt 34 :=
  sorry

end major_axis_length_of_ellipse_tangent_to_y_axis_l108_108854


namespace probability_of_chosen_primes_l108_108422

def is_prime (n : ℕ) : Prop := sorry -- Assume we have a function to check primality

def total_ways : ℕ := Nat.choose 30 2
def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
def primes_not_divisible_by_5 : List ℕ := [2, 3, 7, 11, 13, 17, 19, 23, 29]

def chosen_primes (s : Finset ℕ) : Prop :=
  s.card = 2 ∧
  (∀ n ∈ s, n ∈ primes_not_divisible_by_5)  ∧
  (∀ n ∈ s, n ≠ 5) -- (5 is already excluded in the prime list, but for completeness)

def favorable_ways : ℕ := Nat.choose 9 2  -- 9 primes not divisible by 5

def probability := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_chosen_primes:
  probability = (12 / 145 : ℚ) :=
by
  sorry

end probability_of_chosen_primes_l108_108422


namespace triangle_right_angle_l108_108281

theorem triangle_right_angle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = B - C) : B = 90 :=
by sorry

end triangle_right_angle_l108_108281


namespace sum_of_interior_angles_of_pentagon_l108_108404

theorem sum_of_interior_angles_of_pentagon :
  let n := 5 in (n - 2) * 180 = 540 := 
by 
  let n := 5
  show (n - 2) * 180 = 540
  sorry

end sum_of_interior_angles_of_pentagon_l108_108404


namespace total_bears_l108_108577

-- Definitions based on given conditions
def brown_bears : ℕ := 15
def white_bears : ℕ := 24
def black_bears : ℕ := 27

-- Theorem to prove the total number of bears
theorem total_bears : brown_bears + white_bears + black_bears = 66 := by
  sorry

end total_bears_l108_108577


namespace michael_bought_crates_on_thursday_l108_108323

theorem michael_bought_crates_on_thursday :
  ∀ (eggs_per_crate crates_tuesday crates_given current_eggs bought_on_thursday : ℕ),
    crates_tuesday = 6 →
    crates_given = 2 →
    eggs_per_crate = 30 →
    current_eggs = 270 →
    bought_on_thursday = (current_eggs - (crates_tuesday * eggs_per_crate - crates_given * eggs_per_crate)) / eggs_per_crate →
    bought_on_thursday = 5 :=
by
  intros _ _ _ _ _
  sorry

end michael_bought_crates_on_thursday_l108_108323


namespace minimum_value_l108_108601

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 2) : 
  (1 / m + 2 / n) ≥ 4 :=
sorry

end minimum_value_l108_108601


namespace tourist_arrangement_l108_108414

/-- There are 5 tourists, A, B, and 3 others, assigned to visit 4 scenic spots (A, B, C, D) in a city. Each spot must have at least one tourist, and tourists A and B cannot visit the same spot. -/
theorem tourist_arrangement :
  let total_tourists := 5 in
  let total_spots := 4 in
  let tourists := ["A", "B", "T1", "T2", "T3"] in
  let spots := ["A", "B", "C", "D"] in
  -- The number of valid arrangements satisfying all conditions is 216.
  ∃ (arrangements : fin 216), True :=
sorry

end tourist_arrangement_l108_108414


namespace division_of_A_l108_108562

/-- Definition of a number that consists of 1001 repeated sevens --/
def A : ℤ := (list.repeat 7 1001).foldl (λ acc d, 10 * acc + d) 0

/-- The theorem states that when A is divided by 1001, 
the quotient and remainder are as specified. --/
theorem division_of_A :
  let quotient := (list.repeat 777000 166).foldl (λ acc d, 10 ^ 6 * acc + d) 0,
      remainder := 700 in
  ∃ q r, A = q * 1001 + r ∧ r < 1001 ∧ r = remainder ∧ q = quotient :=
begin
  let quotient := (list.repeat 777000 166).foldl (λ acc d, 10 ^ 6 * acc + d) 0,
  let remainder := 700,
  use [quotient, remainder],
  sorry
end

end division_of_A_l108_108562


namespace total_spectators_l108_108416

-- Definitions of conditions
def num_men : Nat := 7000
def num_children : Nat := 2500
def num_women := num_children / 5

-- Theorem stating the total number of spectators
theorem total_spectators : (num_men + num_children + num_women) = 10000 := by
  sorry

end total_spectators_l108_108416


namespace _l108_108914

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
∀ n, S n = (a 0) * (1 - (a 1 / a 0) ^ n) / (1 - a 1 / a 0)

def sum_of_2n_terms (S : ℕ → α) (a : ℕ → α) : Prop :=
∀ n, S (2 * n) = 4 * (∑ k in range n, a (2 * k))

variable {a : ℕ → α} {S : ℕ → α} {q a_1 a_2 a_3 a_4 : α}

noncomputable def example_theorem : geometric_sequence a q → sum_of_first_n_terms a S →
sum_of_2n_terms S a → a 0 * a 1 * a 2 = 8 → a_2 = 2 → a_1 = 2 / 3 → q = 3 → a 3 = 18 :=
by
  intros h_geo h_sum h_sum_2n h_prod h_a2 h_a1 h_q
  sorry

end _l108_108914


namespace polynomials_with_three_different_roots_count_l108_108122

theorem polynomials_with_three_different_roots_count :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6: ℕ), 
    a_0 = 0 ∧ 
    (a_6 = 0 ∨ a_6 = 1) ∧
    (a_5 = 0 ∨ a_5 = 1) ∧
    (a_4 = 0 ∨ a_4 = 1) ∧
    (a_3 = 0 ∨ a_3 = 1) ∧
    (a_2 = 0 ∨ a_2 = 1) ∧
    (a_1 = 0 ∨ a_1 = 1) ∧
    (1 + a_6 + a_5 + a_4 + a_3 + a_2 + a_1) % 2 = 0 ∧
    (1 - a_6 + a_5 - a_4 + a_3 - a_2 + a_1) % 2 = 0) -> 
  ∃ (n : ℕ), n = 8 :=
sorry

end polynomials_with_three_different_roots_count_l108_108122


namespace extremely_large_number_of_digits_l108_108993

noncomputable def x (a : ℝ) := 3 ^ (3 ^ (3 ^ a))

theorem extremely_large_number_of_digits
  (h : log 3 (log 3 (log 3 x)) = 3) :
  let d := 3 ^ 27 * 0.477 in
  d > 10 ^ 1000000 := sorry

end extremely_large_number_of_digits_l108_108993


namespace points_in_region_satisfy_condition_l108_108336

noncomputable def fractional_part (z : ℝ) : ℝ := z - z.floor

theorem points_in_region_satisfy_condition :
  {p : ℝ × ℝ // -2 ≤ p.1 ∧ p.1 ≤ 2 ∧ -3 ≤ p.2 ∧ p.2 ≤ 3 ∧ fractional_part p.1 ≤ fractional_part p.2} :=
by sorry

end points_in_region_satisfy_condition_l108_108336


namespace min_distance_on_line_l108_108928

theorem min_distance_on_line (a b : ℝ) (h : 3 * a + 4 * b = 20) : ∃ d : ℝ, d = 4 ∧ ∀ (x y : ℝ), 3 * x + 4 * y = 20 → sqrt (x^2 + y^2) ≥ d :=
by
  use 4
  constructor
  . exact rfl
  . sorry

end min_distance_on_line_l108_108928


namespace arithmetic_sequence_middle_term_l108_108268

theorem arithmetic_sequence_middle_term 
  (x y z : ℕ) 
  (h1 : ∀ i, (∀ j, 23 + (i - 0) * d = x)
  (h2 : ∀ i, (∀ j, y = 23 + (j - 1) * (23 + d)) 
  (h3 : ∀ i, (∀ j, z = 23 + (j - 2 * d)) 
  (h4 : ∀ i, 47 = 23 + (5 - 1) * d )
   : y = (23 + 47) / 2 :=
by 
  sorry

end arithmetic_sequence_middle_term_l108_108268


namespace ellipse_properties_l108_108598

variable {x y a b c : ℝ}

-- Conditions for the ellipse
def ellipse_eq : Prop := (a > b) ∧ (b > 0) ∧ (a ^ 2 = 6) ∧ (b ^ 2 = 2) ∧ (c ^ 2 = 4)

-- Condition for the point M
def M_point : Prop := M = (-a^2/c, 0)

-- Main proof statement
theorem ellipse_properties (h : ellipse_eq) (hM : M_point) :
  ( ∃ (eq_ellipse : Prop), eq_ellipse ∧ ( ∀ (C F B : Point), points_collinear C F B) ∧ ( ∃ S, S = sqrt(3)/2 ) ) :=
by
  sorry


end ellipse_properties_l108_108598


namespace member_change_l108_108807

theorem member_change:
  ∀ (X : ℝ), 
  let fall_members := X * 1.06 in
  let spring_members := fall_members * 0.81 in
  ((X - spring_members) / X) * 100 = 14.14 := 
by
  sorry

end member_change_l108_108807


namespace domain_of_f_l108_108739

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - real.log x)

theorem domain_of_f :
  {x : ℝ | 0 < x ∧ 1 - real.log x ≥ 0} = {x : ℝ | 0 < x ∧ x ≤ real.exp 1} :=
by
  sorry

end domain_of_f_l108_108739


namespace product_of_distances_squared_eq_2017_l108_108599

theorem product_of_distances_squared_eq_2017 
  (n : ℕ) (h_n : n = 2017) (h_unit_circle : ∀ v : ℂ, abs v = 1) :
  let S := {x : ℂ | ∃ i j, 0 ≤ i < j < n ∧ x = abs (exp(2 * real.pi * complex.I * (i / n)) - exp(2 * real.pi * complex.I * (j / n)))}
  Q := ∏ s in S, s
  in Q^2 = 2017 :=
by
  sorry

end product_of_distances_squared_eq_2017_l108_108599


namespace total_games_l108_108325

-- The conditions
def working_games : ℕ := 6
def bad_games : ℕ := 5

-- The theorem to prove
theorem total_games : working_games + bad_games = 11 :=
by
  sorry

end total_games_l108_108325


namespace sum_of_interior_angles_of_pentagon_l108_108397

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  sorry

end sum_of_interior_angles_of_pentagon_l108_108397


namespace train_speed_proof_l108_108842

noncomputable def speed_of_train (train_length : ℝ) (time_seconds : ℝ) (man_speed : ℝ) : ℝ :=
  let train_length_km := train_length / 1000
  let time_hours := time_seconds / 3600
  let relative_speed := train_length_km / time_hours
  relative_speed - man_speed

theorem train_speed_proof :
  speed_of_train 605 32.99736021118311 6 = 60.028 :=
by
  unfold speed_of_train
  -- Direct substitution and expected numerical simplification
  norm_num
  sorry

end train_speed_proof_l108_108842


namespace hydroflow_rate_30_minutes_l108_108726

def hydroflow_pumped (rate_per_hour: ℕ) (minutes: ℕ) : ℕ :=
  let hours := minutes / 60
  rate_per_hour * hours

theorem hydroflow_rate_30_minutes : 
  hydroflow_pumped 500 30 = 250 :=
by 
  -- place the proof here
  sorry

end hydroflow_rate_30_minutes_l108_108726


namespace geometric_sequence_n_value_l108_108276

noncomputable def a1 : ℝ := 9 / 8
noncomputable def q : ℝ := 2 / 3
noncomputable def a_n (n : ℕ) : ℝ := a1 * q^(n - 1)

theorem geometric_sequence_n_value :
  a_n 4 = 1 / 3 :=
by
  unfold a_n
  rw [a1, q]
  norm_num
  sorry

end geometric_sequence_n_value_l108_108276


namespace max_volume_of_cuboid_l108_108075

theorem max_volume_of_cuboid (x y z : ℝ) (h1 : 4 * (x + y + z) = 60) : 
  x * y * z ≤ 125 :=
by
  sorry

end max_volume_of_cuboid_l108_108075


namespace ellipse_foci_and_eccentricity_max_tangent_distance_l108_108953

-- Define the ellipse G and circle C
def ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1
def circle (x y : ℝ) := x^2 + y^2 = 1

-- The coordinates of the foci and the eccentricity of the ellipse
theorem ellipse_foci_and_eccentricity : 
  (∀ x y : ℝ, ellipse x y → (x = ± sqrt 3 ∧ y = 0) ∨ (y = ± sqrt 3 ∧ x = 0)) ∧
  (∃ e : ℝ, e = sqrt 3 / 2) :=
sorry

-- Function expressing the distance |AB| and its maximum value given m
theorem max_tangent_distance (m : ℝ) (h : |m| ≥ 1) :
  let |AB| := 4 * sqrt 3 * |m| / (m^2 + 3) 
  in |AB| ≤ 2 ∧ (∃ m : ℝ, |AB| = 2 ∧ m = ± sqrt 3) :=
sorry

end ellipse_foci_and_eccentricity_max_tangent_distance_l108_108953


namespace number_of_non_pine_trees_l108_108412

theorem number_of_non_pine_trees (total_trees : ℕ) (percentage_pine : ℝ) (h1 : total_trees = 350) (h2 : percentage_pine = 0.70) : 
  total_trees - (percentage_pine * (total_trees : ℝ)).to_nat = 105 :=
by
  sorry

end number_of_non_pine_trees_l108_108412


namespace sum_of_all_positive_odd_numbers_less_than_100_l108_108396

def isOdd (n : ℕ) : Prop := n % 2 = 1

def positiveOddNumbersLessThan100 : List ℕ :=
  List.filter (fun n => isOdd n ∧ n > 0) (List.range 100)

def sumOddNumbersLessThan100 : ℕ :=
  (positiveOddNumbersLessThan100).sum

theorem sum_of_all_positive_odd_numbers_less_than_100 :
  sumOddNumbersLessThan100 = 2500 :=
by
  sorry

end sum_of_all_positive_odd_numbers_less_than_100_l108_108396


namespace total_wheels_from_four_wheelers_l108_108249

theorem total_wheels_from_four_wheelers (four_wheelers : ℕ) (wheels_per_four_wheeler : ℕ) : 
  four_wheelers = 11 ∧ wheels_per_four_wheeler = 4 → four_wheelers * wheels_per_four_wheeler = 44 := 
by
  intros h
  cases h with h1 h2
  rw [h1, h2]
  exact rfl

end total_wheels_from_four_wheelers_l108_108249


namespace gary_paycheck_l108_108481

theorem gary_paycheck 
  (normal_wage : ℝ)
  (hours_worked : ℝ)
  (overtime_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_hours : ℝ) :
  normal_wage = 12 →
  hours_worked = 52 →
  overtime_rate = 1.5 →
  regular_hours = 40 →
  overtime_hours = hours_worked - regular_hours →
  (regular_hours * normal_wage + 
   overtime_hours * (normal_wage * overtime_rate)) = 696 := 
by
  intros h_normal_wage h_hours_worked h_overtime_rate h_regular_hours h_overtime_hours
  rw [h_normal_wage, h_hours_worked, h_overtime_rate, h_regular_hours, h_overtime_hours]
  sorry

end gary_paycheck_l108_108481


namespace apple_collection_l108_108499

theorem apple_collection (Alyona Borya Vera Polina : ℕ) 
  (h1 : Alyona > Borya) 
  (h2 : (Alyona + Vera) % 3 = 0)
  (h3 : Alyona ∈ {11, 17, 19, 24})
  (h4 : Borya ∈ {11, 17, 19, 24})
  (h5 : Vera ∈ {11, 17, 19, 24})
  (h6 : Polina ∈ {11, 17, 19, 24})
  (h_distinct : List.nodup [Alyona, Borya, Vera, Polina])
  : Alyona = 19 ∧ Borya = 17 ∧ Vera = 11 ∧ Polina = 24 :=
by
  sorry

end apple_collection_l108_108499


namespace dot_product_calculation_l108_108224

variables {E : Type*} [inner_product_space ℝ E] 
variables (e1 e2 : E) (a b : E)

theorem dot_product_calculation (h1 : ∥e1∥ = 1) (h2 : ∥e2∥ = 1) 
  (h3 : ⟪e1, e2⟫ = Real.cos (Real.pi / 3)) (ha : a = 2 • e1 + e2) (hb : b = -3 • e1 + 2 • e2) :
  ⟪a, b⟫ = -(7 / 2) :=
sorry

end dot_product_calculation_l108_108224


namespace min_value_A2_sub_B2_l108_108689

noncomputable def A (x y z : ℝ) : ℝ :=
  sqrt (x + 2) + sqrt (y + 5) + sqrt (z + 10)

noncomputable def B (x y z : ℝ) : ℝ :=
  sqrt (x + 1) + sqrt (y + 1) + sqrt (z + 1)

theorem min_value_A2_sub_B2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  A x y z ^ 2 - B x y z ^ 2 ≥ 36 :=
sorry

end min_value_A2_sub_B2_l108_108689


namespace domain_of_sqrt_one_minus_ln_l108_108740

def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ Real.exp 1

theorem domain_of_sqrt_one_minus_ln (x : ℝ) : (1 - Real.log x ≥ 0) ∧ (x > 0) ↔ domain x := by
sorry

end domain_of_sqrt_one_minus_ln_l108_108740


namespace rob_reads_9_pages_l108_108350

def planned_reading_time : ℝ := 3 -- in hours
def fraction_of_time_spent : ℝ := 3 / 4
def reading_speed : ℝ := 1 / 15 -- in pages per minute

theorem rob_reads_9_pages :
  let actual_reading_time := planned_reading_time * fraction_of_time_spent
  let total_reading_minutes := actual_reading_time * 60 
  let number_of_pages_read := total_reading_minutes * reading_speed
  number_of_pages_read = 9 :=
by
  -- calculation steps here
  sorry

end rob_reads_9_pages_l108_108350


namespace find_number_l108_108043

theorem find_number (x : ℝ) 
  (h : 0.4 * x + (0.3 * 0.2) = 0.26) : x = 0.5 := 
by
  sorry

end find_number_l108_108043


namespace how_many_ways_l108_108063

theorem how_many_ways (A B C D E F : ℕ) (h: ∀ {x y z}, x > y ∧ y > z → x > z) :
  set.univ ⊆ ({1, 2, 3, 4, 5, 6} : set ℕ) ∧
  (∀ I J, I ≠ J → I ∈ {A, B, C, D, E, F} → J ∈ {A, B, C, D, E, F} → I ≠ J) ∧
  (∀ (x y : ℕ), x ∈ {A, B, C, D, E, F} → y ∈ {A, B, C, D, E, F} → x = y → x = y) ∧
  (∀ A B, A > B → ¬ B > A ∧ A ∉ {B ⋆- 1, B ⋆- 2, B ⋆- 3, B ⋆- 4, B ⋆- 5}) - ∗
  (15 ways to arrange {1, 2, 3, 4, 5, 6} in squares where if two squares are connected then higher square has greater number) ∧
  sorry 

end how_many_ways_l108_108063


namespace cosine_sum_to_product_l108_108380

theorem cosine_sum_to_product :
  ∃ (a b c d : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  (∀ x : ℝ, cos 2 * x + cos 4 * x + cos 8 * x + cos 10 * x = a * cos b * x * cos c * x * cos d * x) ∧
  a + b + c + d = 14 := by
  existsi 4
  existsi 6
  existsi 3
  existsi 1
  split; try {norm_num}
  intro x
  sorry

end cosine_sum_to_product_l108_108380


namespace shaded_percentage_l108_108044

-- Definition for the six-by-six grid and total squares
def total_squares : ℕ := 36
def shaded_squares : ℕ := 16

-- Definition of the problem: to prove the percentage of shaded squares
theorem shaded_percentage : (shaded_squares : ℚ) / total_squares * 100 = 44.4 :=
by
  sorry

end shaded_percentage_l108_108044


namespace towels_end_up_with_l108_108800

theorem towels_end_up_with (g w m : ℕ) (h1 : g = 125) (h2 : w = 130) (h3 : m = 180) : 
  let t := g + w in
  let r := t - m in
  r = 75 :=
by
  sorry

end towels_end_up_with_l108_108800


namespace quadratic_complete_square_l108_108759

theorem quadratic_complete_square (b c : ℝ) (h : ∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) : b + c = -106 :=
by
  sorry

end quadratic_complete_square_l108_108759


namespace angle_BAC_is_angle_between_BC_and_tangent_at_B_l108_108675

noncomputable section

open Real

def point_on_hyperbola (x : ℝ) : ℝ × ℝ := (x, 1 / x)

def tangent_at_point (x : ℝ) (y : ℝ) : ℝ := -1 / (x * x) * (x - y) + 1 / x

theorem angle_BAC_is_angle_between_BC_and_tangent_at_B
  (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > x₁) (hxA : x₁ < 0) :
  ∃ (B C A : ℝ × ℝ), B = point_on_hyperbola x₁ ∧ C = point_on_hyperbola x₂ ∧
  (angle BAC = angle BC (tangent_at_point x₁ (fst B))) := sorry

end angle_BAC_is_angle_between_BC_and_tangent_at_B_l108_108675


namespace price_of_vegetable_seedlings_base_minimum_amount_spent_l108_108346

variable (x : ℝ) (m : ℕ)

/-- The price of each bundle of type A vegetable seedlings at the vegetable seedling base is 20 yuan. -/
theorem price_of_vegetable_seedlings_base : 
  (300 / x = 300 / (5 / 4) * x + 3) → x = 20 :=
begin
  intros,
  sorry
end

/-- The minimum amount spent on purchasing 100 bundles of type A and B vegetable seedlings is 2250 yuan. -/
theorem minimum_amount_spent :
  (∀ (m : ℕ), m ≤ 50 → (w = 20 * 0.9 * m + 30 * 0.9 * (100 - m)) →
    (w = -9 * m + 2700) → m ≤ 50) → 
  (min_cost = -9 * 50 + 2700) → min_cost = 2250 :=
begin
  intros,
  sorry
end

end price_of_vegetable_seedlings_base_minimum_amount_spent_l108_108346


namespace minimum_pressure_l108_108330

theorem minimum_pressure (V T V0 T0 a b c R : ℝ) 
  (h_eq : (V / V0 - a)^2 + (T / T0 - b)^2 = c^2)
  (h_cond : c^2 < a^2 + b^2) :
  ∃ Pmin : ℝ, Pmin = (R * T0 / V0) * (a * sqrt (a^2 + b^2 - c^2) - b * c) / (b * sqrt (a^2 + b^2 - c^2) + a * c) :=
by sorry

end minimum_pressure_l108_108330


namespace worksheets_already_graded_l108_108494

theorem worksheets_already_graded {total_worksheets problems_per_worksheet problems_left_to_grade : ℕ} :
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left_to_grade = 16 →
  (total_worksheets - (problems_left_to_grade / problems_per_worksheet)) = 5 :=
by
  intros h1 h2 h3
  sorry

end worksheets_already_graded_l108_108494


namespace exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l108_108534

structure RatNonIntPair (x y : ℚ) :=
  (x_rational : x.is_rational)
  (x_not_integer : x.num ≠ x.denom)
  (y_rational : y.is_rational)
  (y_not_integer : y.num ≠ y.denom)

theorem exists_rat_nonint_sol_a :
  ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * x + int 8 * y).denom = 1 ∧ (int 8 * x + int 3 * y).denom = 1 := sorry

theorem no_exists_rat_nonint_sol_b :
  ¬ ∃ (x y : ℚ), (RatNonIntPair x y) ∧ (int 19 * (x^2) + int 8 * (y^2)).denom = 1 ∧ (int 8 * (x^2) + int 3 * (y^2)).denom = 1 := sorry

end exists_rat_nonint_sol_a_no_exists_rat_nonint_sol_b_l108_108534


namespace magnitude_of_z_l108_108196

noncomputable def z : ℂ := (1 - complex.I) / complex.I

theorem magnitude_of_z : complex.abs z = real.sqrt 2 :=
by
  sorry -- Proof goes here

end magnitude_of_z_l108_108196


namespace balls_in_boxes_l108_108919

theorem balls_in_boxes (balls boxes : ℕ) (h1 : balls = 4) (h2 : boxes = 4) :
  ∃ n : ℕ, (choose boxes 2 * (2 ^ balls - 2)) = n ∧ n = 84 :=
by
  use 84
  rw [h1, h2]
  simp
  sorry

end balls_in_boxes_l108_108919


namespace polar_coordinates_of_point_l108_108012

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let ρ := real.sqrt (x*x + y*y)
  let θ := real.arccos (x / ρ)
  if y < 0 then (ρ, 2 * real.pi - θ) else (ρ, θ)

theorem polar_coordinates_of_point :
  (polar_coordinates (-2) (-2 * real.sqrt 3)) = (4, 4 * real.pi / 3) :=
by
  sorry

end polar_coordinates_of_point_l108_108012


namespace intersection_point_l108_108372

theorem intersection_point (x y : ℝ) (h1 : y = x + 1) (h2 : y = -x + 1) : (x = 0) ∧ (y = 1) := 
by
  sorry

end intersection_point_l108_108372


namespace smallest_rectangle_area_l108_108554

theorem smallest_rectangle_area (r : ℝ) (h : r = 6) : 
  ∃ (area : ℝ), area = 144 :=
by
  let d := 2 * r
  have h1 : d = 12 := by linarith [h]
  have h2 : area = d * d := by linarith
  have h3 : area = 144 := by linarith
  exact ⟨area, h3⟩

end smallest_rectangle_area_l108_108554


namespace union_eq_l108_108974

-- Define the sets M and N
def M : Finset ℕ := {0, 3}
def N : Finset ℕ := {1, 2, 3}

-- Define the proof statement
theorem union_eq : M ∪ N = {0, 1, 2, 3} := 
by
  sorry

end union_eq_l108_108974


namespace yvettes_final_bill_l108_108285

theorem yvettes_final_bill :
  let alicia : ℝ := 7.5
  let brant : ℝ := 10
  let josh : ℝ := 8.5
  let yvette : ℝ := 9
  let tip_percentage : ℝ := 0.2
  ∃ final_bill : ℝ, final_bill = (alicia + brant + josh + yvette) * (1 + tip_percentage) ∧ final_bill = 42 :=
by
  sorry

end yvettes_final_bill_l108_108285


namespace combined_profit_percentage_is_32_35_l108_108085

def selling_price_A : ℝ := 120
def cost_price_A : ℝ := 0.70 * selling_price_A

def selling_price_B : ℝ := 150
def cost_price_B : ℝ := 0.80 * selling_price_B

def profit_A : ℝ := selling_price_A - cost_price_A
def profit_B : ℝ := selling_price_B - cost_price_B

def combined_profit : ℝ := profit_A + profit_B
def combined_cost_price : ℝ := cost_price_A + cost_price_B

def combined_profit_percentage : ℝ := (combined_profit / combined_cost_price) * 100

theorem combined_profit_percentage_is_32_35 :
  combined_profit_percentage = 32.35 :=
by
  sorry

end combined_profit_percentage_is_32_35_l108_108085


namespace rob_reads_9_pages_l108_108351

def planned_reading_time : ℝ := 3 -- in hours
def fraction_of_time_spent : ℝ := 3 / 4
def reading_speed : ℝ := 1 / 15 -- in pages per minute

theorem rob_reads_9_pages :
  let actual_reading_time := planned_reading_time * fraction_of_time_spent
  let total_reading_minutes := actual_reading_time * 60 
  let number_of_pages_read := total_reading_minutes * reading_speed
  number_of_pages_read = 9 :=
by
  -- calculation steps here
  sorry

end rob_reads_9_pages_l108_108351


namespace power_of_prime_implies_n_prime_l108_108721

theorem power_of_prime_implies_n_prime (n : ℕ) (p : ℕ) (k : ℕ) (hp : Nat.Prime p) :
  3^n - 2^n = p^k → Nat.Prime n :=
by
  sorry

end power_of_prime_implies_n_prime_l108_108721


namespace solution_set_inequality_l108_108191

-- Definitions of the conditions
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2^x - 3 else - (2^(-x) - 3)

-- Statement to prove
theorem solution_set_inequality :
  is_odd_function f ∧ (∀ x > 0, f x = 2^x - 3)
  → {x : ℝ | f x ≤ -5} = {x : ℝ | x ≤ -3} := by
  sorry

end solution_set_inequality_l108_108191


namespace range_of_n_l108_108202

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.exp 1) * Real.exp x + (1 / 2) * x^2 - x

theorem range_of_n :
  (∃ m : ℝ, f m ≤ 2 * n^2 - n) ↔ (n ≤ -1/2 ∨ 1 ≤ n) :=
sorry

end range_of_n_l108_108202


namespace probability_of_odd_sum_l108_108574

open Nat

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_odd_sum :
  (binomial 11 3) / (binomial 12 4) = 1 / 3 := by
sorry

end probability_of_odd_sum_l108_108574


namespace limit_measurable_l108_108303

variables {Ω : Type*} {E : Type*} [MeasurableSpace Ω] [MetricSpace E]
          (𝓕 : MeasurableSpace Ω)
          (𝓔 : MeasurableSpace E := borel E)
          (X : Ω → E)
          (Xn : ℕ → Ω → E)

-- Define the pointwise limit assumption
def pointwise_limit (X : Ω → E) (Xn : ℕ → Ω → E) :=
  ∀ ω, tendsto (λ n, Xn n ω) at_top (𝓝 (X ω))

-- Assumption: Xn are measurable w.r.t. the given sigma-algebras
def measurable_sequence (Xn : ℕ → Ω → E) :=
  ∀ n, measurable (Xn n)

-- Theorem: X is measurable if it is the pointwise limit of a sequence of measurable functions
theorem limit_measurable
  (h_lim : pointwise_limit X Xn)
  (h_meas : measurable_sequence Xn) :
  measurable X :=
begin
  sorry -- Proof is omitted
end

end limit_measurable_l108_108303


namespace abc_is_cube_l108_108299

theorem abc_is_cube (a b c : ℤ) (h : (a:ℚ) / (b:ℚ) + (b:ℚ) / (c:ℚ) + (c:ℚ) / (a:ℚ) = 3) : ∃ x : ℤ, abc = x^3 :=
by
  sorry

end abc_is_cube_l108_108299


namespace find_n_l108_108517

-- Definition of the sequence s_n
def s : ℕ → ℚ 
| 1     := 2
| (n+1) := if (n + 1) % 4 = 0 then 2 + s ((n + 1) / 4)
           else if (n + 1) % 2 = 1 then 1 / s n
           else s n + 1

-- Prove that n = 129 when s_n = 5/36
theorem find_n (n : ℕ) (h : s n = 5 / 36) : n = 129 := by
  sorry

end find_n_l108_108517


namespace area_of_region_l108_108510

theorem area_of_region : ∀ (x y : ℝ), (x^2 + y^2 - 3 = 2*y - 8*x + 6) → (area_of_region_eq_26π : real.pi * 26 = 26 * real.pi) :=
by
  sorry

end area_of_region_l108_108510


namespace handshakes_count_l108_108353

theorem handshakes_count {n : ℕ} (h : n = 7)
  (H : ∀ p : Fin n, ∀ q : Fin n, p ≠ q → p ≠ spouse q → person p ≠ person q → 
    handshake_occur p q) : 
  (∃ x y : Fin n, x ≠ y ∧ mutual_avoidance x y) → handshakes = 77 := 
begin
  sorry
end

-- Definitions
def person (p : Fin 14) : Fin 14
def spouse (p : Fin 14) : Fin 14
def handshake_occur (p q : Fin 14) : Prop
def mutual_avoidance (p q : Fin 14) : Prop

end handshakes_count_l108_108353


namespace ratio_A1B1_C2B2_eq_l108_108006

-- Definitions of the lengths of the sides of the triangle and the constraints a < b < c
variables (a b c : ℝ) (h1 : a < b) (h2 : b < c)

-- Definitions of distances BB_1, AA_1, CC_2, and BB_2
def BB1 : ℝ := c
def AA1 : ℝ := c
def CC2 : ℝ := a
def BB2 : ℝ := a

-- Prove the ratio A1B1 : C2B2 is equal to c / a
theorem ratio_A1B1_C2B2_eq (h3 : BB1 = c) (h4 : AA1 = c) (h5 : CC2 = a) (h6 : BB2 = a) : 
  (∃ A1B1 C2B2 : ℝ, A1B1 / C2B2 = c / a) := 
by
  sorry

end ratio_A1B1_C2B2_eq_l108_108006


namespace farmer_sowed_buckets_l108_108476

-- Define the initial and final buckets of seeds
def initial_buckets : ℝ := 8.75
def final_buckets : ℝ := 6.00

-- The goal: prove the number of buckets sowed is 2.75
theorem farmer_sowed_buckets : initial_buckets - final_buckets = 2.75 := by
  sorry

end farmer_sowed_buckets_l108_108476


namespace cos_sum_to_product_l108_108378

theorem cos_sum_to_product (a b c d : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h : ∀ x : ℝ, cos (2 * x) + cos (4 * x) + cos (8 * x) + cos (10 * x) = a * cos (b * x) * cos (c * x) * cos (d * x)) :
  a + b + c + d = 14 :=
sorry

end cos_sum_to_product_l108_108378


namespace line_ellipse_common_point_l108_108967

theorem line_ellipse_common_point (k : ℝ) (m : ℝ) :
  (∀ (x y : ℝ), y = k * x + 1 →
    (y^2 / m + x^2 / 5 ≤ 1)) ↔ (m ≥ 1 ∧ m ≠ 5) :=
by sorry

end line_ellipse_common_point_l108_108967


namespace B_finishes_alone_in_18_days_l108_108479

-- Definitions based on the conditions in a)
def Workman := Type

-- A and B are two workmen
variable (A B : Workman)

-- Define the speed of B's work (amount of work done per day)
variable (work_rate_B : ℝ)
-- Define that A is half as good a workman as B
def work_rate_A := work_rate_B / 2

-- Condition that together they finish a job in 12 days
def together_work_rate := work_rate_A + work_rate_B
def together_days := 12
def one_job := 1

-- Theorem to prove
theorem B_finishes_alone_in_18_days (h : together_work_rate = one_job / together_days) : work_rate_B = 1 / 18 := by
  sorry

end B_finishes_alone_in_18_days_l108_108479


namespace ratio_of_ages_l108_108717

theorem ratio_of_ages (M : ℕ) (S : ℕ) (h1 : M = 24) (h2 : S + 6 = 38) : 
  (S / M : ℚ) = 4 / 3 :=
by
  sorry

end ratio_of_ages_l108_108717


namespace math_problem_l108_108884

theorem math_problem
  (a b c : ℝ)
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 31)
  (h3 : a + 2b - c = 5) :
  ab - c = 37 / 2 := by
  sorry

end math_problem_l108_108884


namespace hyperbola_focus_asymptote_distance_l108_108938

-- Define the hyperbola and the given condition
def hyperbola (a : ℝ) (h_a : a > 0) : Prop := 
  ∃ (F : ℝ × ℝ), (F = (2 * (real.sqrt a), 0) ∨ F = (-2 * (real.sqrt a), 0)) ∧
  (∀ x y, x^2 / (3 * a) - y^2 / a = 1)

-- Define the asymptote
def asymptote (a : ℝ) : (ℝ × ℝ) → Prop := 
  λ P, ∃ x, P = (x, (real.sqrt 3 / 3) * x) ∨ P = (x, - (real.sqrt 3 / 3) * x)

-- Define the distance formula from a point to a line (asymptote)
def distance_from_focus_to_asymptote (a : ℝ) (F : ℝ × ℝ) : ℝ :=
  abs ((real.sqrt 3 / 3) * F.1 - F.2) / real.sqrt (1 + (real.sqrt 3 / 3)^2)

-- The theorem statement
theorem hyperbola_focus_asymptote_distance (a : ℝ) (h_a : a > 0) :
  ∃ F, (F = (2 * (real.sqrt a), 0) ∨ F = (-2 * (real.sqrt a), 0)) ∧
  distance_from_focus_to_asymptote a F = real.sqrt a := 
sorry

end hyperbola_focus_asymptote_distance_l108_108938


namespace trigonometric_inequalities_l108_108994

theorem trigonometric_inequalities (θ : ℝ) (h₁ : π < θ) (h₂ : θ < 5 * π / 4) :
  cos θ < sin θ ∧ sin θ < tan θ := 
sorry

end trigonometric_inequalities_l108_108994


namespace pascal_triangle_row_10_sum_l108_108868

theorem pascal_triangle_row_10_sum : (∑ (k : Fin 11), nat.choose 10 k) = 1024 := by
  sorry

end pascal_triangle_row_10_sum_l108_108868


namespace max_area_angle_l108_108856

-- Definitions of the problem
structure IsoscelesTrapezoid (a : ℝ) :=
  (b c : ℝ)
  (leg1_eq_base : b = a)
  (leg2_eq_base : c = a)

def area_of_trapezoid (a α : ℝ) : ℝ :=
  a^2 * (1 + Real.cos α) * (Real.sin α)

-- Theorem: The angle that maximizes the area of the trapezoid
theorem max_area_angle (a : ℝ) (h : 0 < a) :
  ∃ α, α = Real.pi / 3 ∧ ∀ α', 0 < α' ∧ α' < Real.pi / 2 → 
    area_of_trapezoid a α' ≤ area_of_trapezoid a (Real.pi / 3) :=
by
  sorry

end max_area_angle_l108_108856


namespace find_S16_l108_108680

-- Definitions
def geom_seq (a : ℕ → ℝ) : Prop := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def sum_of_geom_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, S n = a 0 * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))

-- Problem conditions
variables {a : ℕ → ℝ} {S : ℕ → ℝ}

axiom geom_seq_a : geom_seq a
axiom S4_eq : S 4 = 4
axiom S8_eq : S 8 = 12

-- Theorem
theorem find_S16 : S 16 = 60 :=
  sorry

end find_S16_l108_108680


namespace Sn_squared_value_l108_108581

theorem Sn_squared_value (n : ℕ) (hn : 2 ≤ n) : 
  let a : ℕ → ℝ := λ i, Inf (Set.image (λ k, k + i / k) {k | k > 0})
  let S : ℕ → ℝ := λ n, (Finset.range (n * n)).sum (λ i, ⌊a (i + 1)⌋)
  S n = (8 * n^3 - 3 * n^2 + 13 * n - 6) / 6 :=
begin
  sorry
end

end Sn_squared_value_l108_108581


namespace domain_of_f_l108_108132

-- Define the logarithm base 2 function using the natural log
def log_base2 (x: ℝ) : ℝ := Real.log x / Real.log 2

-- Define the function f(x) = 1 / sqrt(log_base2 x - 1)
noncomputable def f (x: ℝ) : ℝ := 1 / Real.sqrt (log_base2 x - 1)

-- Theorem stating the domain of the function f
theorem domain_of_f : { x : ℝ | f x = f x } = Ioi 2 := 
by 
  sorry

end domain_of_f_l108_108132


namespace length_of_arc_RP_l108_108654

/-- Given a circle with center O and radius OR, where the measure of the inscribed angle RIP is 45 degrees, 
    prove that the length of the arc RP is 6π cm. -/
theorem length_of_arc_RP (O R P : Point) (r : ℝ) (hR : dist O R = 12)
  (h_angle : ∠RIP = 45) : arc_length RP = 6 * π := by
  sorry

end length_of_arc_RP_l108_108654


namespace divides_expression_l108_108719

theorem divides_expression (y : ℤ) (hy : y > 1) : y - 1 ∣ y^(y^2 - y + 2) - 4*y + y^2021 + 3*y^2 - 1 :=
sorry

end divides_expression_l108_108719


namespace part_a_part_b_l108_108297

-- Define the problem setup
variables {A B C P : Type} 

-- Define the triangle and point P such that the triangles have the same area and perimeter
def is_triangle (A B C : Type) := sorry -- Definition of a triangle to be specified
def same_area_perimeter (P A B C : Type) := sorry -- Definition to ensure the sub-triangles have same area and perimeter

-- Define predicates for being in interior or exterior
def is_interior (P A B C : Type) := sorry -- Predicate definition for interior point
def is_exterior (P A B C : Type) := sorry -- Predicate definition for exterior point

-- Theorems to be proven
theorem part_a (A B C P : Type) (h1 : is_triangle A B C) (h2 : same_area_perimeter P A B C) (h3 : is_interior P A B C) : 
  A = B ∧ B = C ∧ C = A := sorry

theorem part_b (A B C P : Type) (h1 : is_triangle A B C) (h2 : same_area_perimeter P A B C) (h3 : is_exterior P A B C) : 
  (∃ R : Type, is_right_angle_triangle R A B C) := sorry

end part_a_part_b_l108_108297


namespace find_n_for_primes_l108_108552

def A_n (n : ℕ) : ℕ := 1 + 7 * (10^n - 1) / 9
def B_n (n : ℕ) : ℕ := 3 + 7 * (10^n - 1) / 9

theorem find_n_for_primes (n : ℕ) :
  (∀ n, n > 0 → (Nat.Prime (A_n n) ∧ Nat.Prime (B_n n)) ↔ n = 1) :=
sorry

end find_n_for_primes_l108_108552


namespace log2_eq_of_conditions_l108_108640

variable (p m n c : ℝ)
variable (log2 : ℝ → ℝ)
variable [log2_cond: ∀ x, log2 (2 * x) = c - log2 (p * n)]
variable [p_pos: 0 < p]

theorem log2_eq_of_conditions :
  ∃ m, log2 (2 * m) = c - log2 (p * n) →
  m = 2^(c-1) / (p * n) := by
  sorry

end log2_eq_of_conditions_l108_108640


namespace min_pieces_for_infinite_operations_l108_108588

structure Graph :=
(vertices : Type)
(adj : vertices → vertices → Prop)
(simple : ∀ (v : vertices), ¬adj v v)
(conn : ∀ (v1 v2 : vertices), ∃ (path : List vertices), v1 = path.head ∧ v2 = path.reverse.head ∧ 
                             ∀ (u v : vertices), (u, v) ∈ path.zip path.tail → adj u v)

def graph_edges (G : Graph) : ℕ :=
  G.vertices.to_list.sum_by (λ v, G.vertices.to_list.count (λ w, G.adj v w)) / 2

def min_pieces (G : Graph) : ℕ :=
  graph_edges G

theorem min_pieces_for_infinite_operations (G : Graph) : min_pieces G = graph_edges G :=
by
  sorry

end min_pieces_for_infinite_operations_l108_108588


namespace least_number_of_colors_for_17_gon_l108_108775

theorem least_number_of_colors_for_17_gon :
  let P := fin 17,
      different_colors_condition := λ i j : P, (∃ k : ℕ, k ≤ 3 ∧ (j - i).val = 2^k + 1)
  in ∃ colors : P → fin 4, ∀ i j : P, different_colors_condition i j → colors i ≠ colors j :=
by
  let P := fin 17
  let different_colors_condition := λ i j : P, (∃ k : ℕ, k ≤ 3 ∧ (j - i).val = 2^k + 1)
  exact exists.elim sorry

end least_number_of_colors_for_17_gon_l108_108775


namespace perfect_square_product_l108_108833

noncomputable def num_faces : ℕ := 5
noncomputable def faces : Finset ℕ := {1, 2, 4, 5, 6}
noncomputable def rolls : ℕ := 5

theorem perfect_square_product :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ (faces ^ rolls).card = 3125 ∧ m + n = 3653 :=
by
  sorry

end perfect_square_product_l108_108833


namespace train_length_eq_l108_108087

theorem train_length_eq (L : ℝ) (time_tree time_platform length_platform : ℝ)
  (h_tree : time_tree = 60) (h_platform : time_platform = 105) (h_length_platform : length_platform = 450)
  (h_speed_eq : L / time_tree = (L + length_platform) / time_platform) :
  L = 600 :=
by
  sorry

end train_length_eq_l108_108087


namespace smallest_number_is_20_l108_108778

theorem smallest_number_is_20 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ≤ b) (h5 : b ≤ c)
  (mean_condition : (a + b + c) / 3 = 30)
  (median_condition : b = 31)
  (largest_condition : b = c - 8) :
  a = 20 :=
sorry

end smallest_number_is_20_l108_108778


namespace number_of_apples_l108_108112

theorem number_of_apples (A : ℝ) (h : 0.75 * A * 0.5 + 0.25 * A * 0.1 = 40) : A = 100 :=
by
  sorry

end number_of_apples_l108_108112


namespace problem_l108_108683

-- Define the problem conditions
variable (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, x > 0 → y > 0 → f(x * y) = f(x) / y)
variable (h2 : f 1000 = 2)

-- Define the theorem to be proved
theorem problem : f 750 = 8 / 3 :=
by
  sorry

end problem_l108_108683


namespace concurrency_of_lines_l108_108282

open_locale euclidean_geometry

noncomputable theory

variables {A B C C' B' H H' : Point}

/-- In triangle ABC, a circle passing through points B and C intersects sides AB and AC at points 
C' and B' respectively. Prove that lines BB', CC', and HH' are concurrent, where H and H' are 
the orthocenters of triangles ABC and AB'C' respectively. -/
theorem concurrency_of_lines (hABC : ¬ collinear ℝ A B C)
  (h_circle : (circle (A, B, C, C') ∧ circle (A, B, C, B')))
  (h_orthocenter_ABC : orthocenter ℝ A B C = H)
  (h_orthocenter_AB'C' : orthocenter ℝ A B' C' = H') :
  concurrent ℝ [line[ℝ B B'], line[ℝ C C'], line[ℝ H H']] :=
sorry

end concurrency_of_lines_l108_108282


namespace Vasechkin_result_l108_108335

-- Define the operations
def P (x : ℕ) : ℕ := (x / 2 * 7) - 1001
def V (x : ℕ) : ℕ := (x / 8) ^ 2 - 1001

-- Define the proposition
theorem Vasechkin_result (x : ℕ) (h_prime : P x = 7) : V x = 295 := 
by {
  -- Proof is omitted
  sorry
}

end Vasechkin_result_l108_108335


namespace sphere_radius_equals_three_l108_108232

noncomputable def radius_of_sphere : ℝ := 3

theorem sphere_radius_equals_three {R : ℝ} (h1 : 4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) : 
  R = radius_of_sphere :=
by
  sorry

end sphere_radius_equals_three_l108_108232


namespace perpendicular_lines_intersect_at_point_l108_108027

theorem perpendicular_lines_intersect_at_point :
  ∀ (d k : ℝ), 
  (∀ x y, 3 * x - 4 * y = d ↔ 8 * x + k * y = d) → 
  (∃ x y, x = 2 ∧ y = -3 ∧ 3 * x - 4 * y = d ∧ 8 * x + k * y = d) → 
  d = -2 :=
by sorry

end perpendicular_lines_intersect_at_point_l108_108027


namespace conic_section_focus_l108_108197

noncomputable def m : ℝ :=
  if h : m ≠ 0 ∧ m ≠ 5 ∧ focus_conic_section (ellipse_coefficients m 5) = (2, 0) then
    9
  else
    0 

-- Definitions for clarity (assuming these exist and are imported from Mathlib or defined here as necessary):
def ellipse_coefficients (a b : ℝ) : ConicSection :=
  ConicSection.ellipse a b

def focus_conic_section (Γ : ConicSection) : (ℝ × ℝ) :=
  match Γ with
  | ConicSection.ellipse a b => (sqrt(a - b), 0) -- Simplified, actual formula would be different based on full context
  | _ => (0, 0)

theorem conic_section_focus (m : ℝ) : ∀ (h : m ≠ 0 ∧ m ≠ 5 ∧ focus_conic_section (ellipse_coefficients m 5) = (2, 0)), m = 9 :=
sorry


end conic_section_focus_l108_108197


namespace determine_y_in_terms_of_x_l108_108522

theorem determine_y_in_terms_of_x (x y : ℝ)
  (h : y = sqrt (x^2 - 2 * x + 4) + sqrt (x^2 + 2 * x + 4)) :
  y = sqrt ((x - 1)^2 + 3) + sqrt ((x + 1)^2 + 3) :=
sorry

end determine_y_in_terms_of_x_l108_108522


namespace workers_distribution_l108_108834

theorem workers_distribution (x y : ℕ) (h1 : x + y = 32) (h2 : 2 * 5 * x = 6 * y) : 
  (∃ x y : ℕ, x + y = 32 ∧ 2 * 5 * x = 6 * y) :=
sorry

end workers_distribution_l108_108834


namespace equilateral_triangle_AD_eq_CE_l108_108932

theorem equilateral_triangle_AD_eq_CE
  (A B C D E : Type*)
  [IsEquilateralTriangle A B C]
  (hD : IsExtensionOf D A C)
  (hE : IsExtensionOf E B C)
  (h : BD = DE) :
  AD = CE := by
  sorry

end equilateral_triangle_AD_eq_CE_l108_108932


namespace ratio_square_correct_l108_108013

noncomputable def ratio_square (a b : ℝ) (h : a / b = b / Real.sqrt (a^2 + b^2)) : ℝ :=
  let k := a / b
  let x := k * k
  x

theorem ratio_square_correct (a b : ℝ) (h : a / b = b / Real.sqrt (a^2 + b^2)) :
  ratio_square a b h = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end ratio_square_correct_l108_108013


namespace graph_shift_equiv_l108_108781

theorem graph_shift_equiv :
  (∀ x, sin(3 * x) + cos(3 * x) = sqrt 2 * cos (3 * (x - π / 12))) :=
sorry

end graph_shift_equiv_l108_108781


namespace chang_total_apples_l108_108114

def sweet_apple_price : ℝ := 0.5
def sour_apple_price : ℝ := 0.1
def sweet_apple_percentage : ℝ := 0.75
def sour_apple_percentage : ℝ := 1 - sweet_apple_percentage
def total_earnings : ℝ := 40

theorem chang_total_apples : 
  (total_earnings / (sweet_apple_percentage * sweet_apple_price + sour_apple_percentage * sour_apple_price)) = 100 :=
by
  sorry

end chang_total_apples_l108_108114


namespace time_difference_l108_108652

-- Definitions for the problem conditions
def Zoe_speed : ℕ := 9 -- Zoe's speed in minutes per mile
def Henry_speed : ℕ := 7 -- Henry's speed in minutes per mile
def Race_length : ℕ := 12 -- Race length in miles

-- Theorem to prove the time difference
theorem time_difference : (Race_length * Zoe_speed) - (Race_length * Henry_speed) = 24 :=
by
  sorry

end time_difference_l108_108652


namespace equation_of_line_l_l108_108169

noncomputable def line_l (P : ℝ × ℝ) (L : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, L a = P.2

theorem equation_of_line_l (P : ℝ × ℝ) (L : ℝ → ℝ) (intersect_line : ∀ (x : ℝ), x - L x + 5 = 0)
    (angle_condition : ∀ (x : ℝ), L x = x - 5) :
  P = (5, -2) → L = (λ x, -2) ∧ L = (λ x, 5) :=
by
  intros hP
  have hL : L = λ x, -2 := sorry
  have hLx : L = λ x, 5 := sorry
  exact ⟨hL, hLx⟩

end equation_of_line_l_l108_108169


namespace ratio_lcm_gcf_eq_10_l108_108042

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b
def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem ratio_lcm_gcf_eq_10 (a b : ℕ) (h₁: a = 252) (h₂: b = 630) :
  (lcm a b) / (gcf a b) = 10 := by
  sorry

end ratio_lcm_gcf_eq_10_l108_108042


namespace plate_arrangement_count_l108_108832

/-- Prove the number of circular arrangements of plates with given conditions. -/
theorem plate_arrangement_count : 
  let total_ways := 277200 in
  let green_adjacent := 138600 in
  let yellow_adjacent := 138600 in
  let both_adjacent := 50400 in
  total_ways - green_adjacent - yellow_adjacent + both_adjacent = 50400 :=
by sorry

end plate_arrangement_count_l108_108832


namespace sqrt_eq_seven_iff_x_eq_fifty_four_l108_108148

theorem sqrt_eq_seven_iff_x_eq_fifty_four (x : ℝ) : sqrt (x - 5) = 7 ↔ x = 54 := by
  sorry

end sqrt_eq_seven_iff_x_eq_fifty_four_l108_108148


namespace circle_geometry_l108_108074

variables {O T M A B A' : Type*}
variables [circle O] [point T] [point M] [point A] [point B] [point A']

-- Define the geometric relationships
variables (m : line) (is_tangent : ∀ (O : circle) (T : point), tangent O m T)
variables (is_perpendicular : ∀ (TM : line) (m : line), perpendicular TM m)
variables (symmetry : ∀ (A A' M : point), symmetric A A' M)

-- Define the circle, points, and diameter
variables (circle_has_diameter : A'B = diameter O)
variables (right_angle : ∀ (T A' B : point), angle A' T B = 90)

-- The main theorem to prove
theorem circle_geometry {O T M A B A'} 
  (h1 : is_tangent O T) 
  (h2 : is_perpendicular (line T M) m) 
  (h3 : symmetry A A' M) :
  circle_has_diameter → right_angle :=
by sorry

end circle_geometry_l108_108074


namespace pentagon_angle_E_l108_108848

-- Definition of the problem
def convex_pentagon (A B C D E : Type) : Prop :=
  ∀ (P Q R: Type), (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) ∧
                   (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E) ∧ 
                   (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E) ∧
                   (P ≠ Q ∧ Q ≠ R ∧ P ≠ R) → 
                   true

def equal_side_lengths (A B C D E : Type) (length : ℝ) : Prop :=
  (A.distance B = length) ∧ 
  (B.distance C = length) ∧ 
  (C.distance D = length) ∧ 
  (D.distance E = length) ∧ 
  (E.distance A = length)

def angle (A B C : Type) : ℝ := sorry

theorem pentagon_angle_E {A B C D E : Type} 
  (h_convex : convex_pentagon A B C D E)
  (h_sides : equal_side_lengths A B C D E 1) -- Assuming unit length for simplicity
  (h_angleA : angle E A B = 100)
  (h_angleB : angle A B C = 100) :
  angle D E A = 140 := 
sorry

end pentagon_angle_E_l108_108848


namespace exists_rational_non_integer_xy_no_rational_non_integer_xy_l108_108530

-- Part (a)
theorem exists_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  (∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
   ∃ z1 z2 : ℤ, 19 * x + 8 * y = ↑z1 ∧ 8 * x + 3 * y = ↑z2) :=
sorry

-- Part (b)
theorem no_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  ¬ ∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
  ∃ z1 z2 : ℤ, 19 * x^2 + 8 * y^2 = ↑z1 ∧ 8 * x^2 + 3 * y^2 = ↑z2 :=
sorry

end exists_rational_non_integer_xy_no_rational_non_integer_xy_l108_108530


namespace solve_for_x_l108_108151

theorem solve_for_x (x : ℝ) (h : sqrt (x - 5) = 7) : x = 54 :=
by
  sorry

end solve_for_x_l108_108151


namespace sum_remainders_l108_108500

theorem sum_remainders (n : ℤ) (h : n % 20 = 14) : (n % 4) + (n % 5) = 6 :=
  by
  sorry

end sum_remainders_l108_108500


namespace difference_of_iterative_averages_l108_108502

def iterative_average (a b c: ℚ) : ℚ :=
  (a+b)/2

theorem difference_of_iterative_averages :
  let a := 6
  let b := 7
  let c := 8
  let avg1 := iterative_average
  max (iterative_average (iterative_average 8 7) 6) 
      (iterative_average (iterative_average 6 7) 8) -
  min (iterative_average (iterative_average 8 7) 6) 
      (iterative_average (iterative_average 6 7) 8) = 1 / 2 :=
by
  sorry

end difference_of_iterative_averages_l108_108502


namespace intersect_planes_parallel_to_line_l108_108945

noncomputable def line := sorry
noncomputable def plane := sorry
noncomputable def perp (a b : Type) := sorry
noncomputable def skew (a b : Type) := sorry
noncomputable def not_in (a b : Type) := sorry
noncomputable def parallel (a b : Type) := sorry
noncomputable def intersection_line (a b : Type) := sorry

theorem intersect_planes_parallel_to_line 
  (m n : line) (α β : plane) (l : line) 
  (h1 : skew m n)
  (h2 : perp m α)
  (h3 : perp n β)
  (h4 : perp l m)
  (h5 : perp l n)
  (h6 : not_in l α)
  (h7 : not_in l β) :
  parallel (intersection_line α β) l :=
sorry

end intersect_planes_parallel_to_line_l108_108945


namespace arithmetic_sequence_middle_term_l108_108264

theorem arithmetic_sequence_middle_term (x y z : ℝ) (h : list.nth_le [23, x, y, z, 47] 2 sorry = y) :
  y = 35 :=
sorry

end arithmetic_sequence_middle_term_l108_108264


namespace y_range_l108_108580

variable (a b : ℝ)
variable (h₀ : 0 < a) (h₁ : 0 < b)

theorem y_range (x : ℝ) (y : ℝ) (h₂ : y = (a * Real.sin x + b) / (a * Real.sin x - b)) : 
  y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b) :=
sorry

end y_range_l108_108580


namespace base8_base9_equivalence_l108_108731

def base8_digit (x : ℕ) := 0 ≤ x ∧ x < 8
def base9_digit (y : ℕ) := 0 ≤ y ∧ y < 9

theorem base8_base9_equivalence 
    (X Y : ℕ) 
    (hX : base8_digit X) 
    (hY : base9_digit Y) 
    (h_eq : 8 * X + Y = 9 * Y + X) :
    (8 * 7 + 6 = 62) :=
by
  sorry

end base8_base9_equivalence_l108_108731


namespace smallest_positive_multiple_l108_108433

theorem smallest_positive_multiple : 
  ∃ n ∈ Nat, n > 0 ∧ (∃ k1 ∈ Nat, n = k1 * 3) 
                  ∧ (∃ k2 ∈ Nat, n = k2 * 5) 
                  ∧ (∃ k3 ∈ Nat, n = k3 * 7) 
                  ∧ (∃ k4 ∈ Nat, n = k4 * 9) 
                  ∧ (∀ m ∈ Nat, (∀ p, m = p * 3 → ∀ q, m = q * 5 → ∀ r, m = r * 7 → ∀ s, m = s * 9 → m ≥ n)) :=
begin
  use 315,
  split,
  { -- Check if 315 is in Nat and greater than 0
    exact Nat.zero_lt_succ 314, 
  },
  { -- Check conditions 1 to 4 for multiples
    split,
    { use 105, exact Nat.mul_eq_mul_left 3 105 },
    split,
    { use 63, exact Nat.mul_eq_mul_left 5 63 },
    split,
    { use 45, exact Nat.mul_eq_mul_left 7 45 },
    { use 35, exact Nat.mul_eq_mul_left 9 35,
      -- Verify 315 is the smallest number
      sorry 
    }
  }
end

end smallest_positive_multiple_l108_108433


namespace OA_perpendicular_OB_area_triangle_OAB_at_k_sqrt2_l108_108593

noncomputable def parabola := { p : ℝ × ℝ | p.2^2 = 2 * p.1 }
noncomputable def intersection_points (k : ℝ) :=
  { p : ℝ × ℝ | p ∈ parabola ∧ p.2 = k * (p.1 - 2) }

theorem OA_perpendicular_OB (k : ℝ) (A B : ℝ × ℝ) (hA : A ∈ intersection_points k) (hB : B ∈ intersection_points k) :
    let O := (0, 0)
    in prod.fst A * prod.fst B + prod.snd A * prod.snd B = 0 := 
  sorry

theorem area_triangle_OAB_at_k_sqrt2:
    let k := real.sqrt 2
    let O := (0, 0)
    let A := (1, -real.sqrt 2)
    let B := (4, 2 * real.sqrt 2)
    in 0.5 * real.sqrt ((prod.fst A)^2 + (prod.snd A)^2) *  real.sqrt ((prod.fst B)^2 + (prod.snd B)^2) = 3 * real.sqrt 2 := 
  sorry

end OA_perpendicular_OB_area_triangle_OAB_at_k_sqrt2_l108_108593


namespace round_robin_tournament_l108_108463

theorem round_robin_tournament (n : ℕ)
  (total_points_1 : ℕ := 3086) (total_points_2 : ℕ := 2018) (total_points_3 : ℕ := 1238)
  (pair_avg_1 : ℕ := (3086 + 1238) / 2) (pair_avg_2 : ℕ := (3086 + 2018) / 2) (pair_avg_3 : ℕ := (1238 + 2018) / 2)
  (overall_avg : ℕ := (3086 + 2018 + 1238) / 3)
  (all_pairwise_diff : pair_avg_1 ≠ pair_avg_2 ∧ pair_avg_1 ≠ pair_avg_3 ∧ pair_avg_2 ≠ pair_avg_3) :
  n = 47 :=
by
  sorry

end round_robin_tournament_l108_108463


namespace total_students_in_nursery_l108_108858

-- Definitions based on the conditions given in the problem:
def fraction_five_or_older (T : ℕ) : ℝ := (1/8 : ℝ) * T
def fraction_four (T : ℕ) : ℝ := (1/4 : ℝ) * T
def fraction_three_to_four (T : ℕ) : ℝ := (1/3 : ℝ) * T
def students_under_two : ℤ := 40
def students_not_between_two_and_three : ℤ := 60
def students_under_two_or_over_three : ℤ := 100

-- The total number of children in the nursery school:
theorem total_students_in_nursery : 
  ∃ T : ℕ, (fraction_five_or_older T + fraction_four T + fraction_three_to_four T) = (17 / 24) * T ∧ T = 142 := by
  sorry

end total_students_in_nursery_l108_108858


namespace spontaneous_low_temperature_l108_108788

theorem spontaneous_low_temperature (ΔH ΔS T : ℝ) (spontaneous : ΔG = ΔH - T * ΔS) :
  (∀ T, T > 0 → ΔG < 0 → ΔH < 0 ∧ ΔS < 0) := 
by 
  sorry

end spontaneous_low_temperature_l108_108788


namespace probability_exactly_three_integer_points_l108_108308

def diagonal_length : ℝ := 
  Real.sqrt ((1 / 4 + 1 / 4)^2 + (3 / 4 + 3 / 4)^2)

def side_length : ℝ := 
  Real.sqrt (2.5 / 2)

axiom point_uniform_random (v : ℝ × ℝ) : (0 ≤ v.1 ∧ v.1 ≤ 100) ∧ (0 ≤ v.2 ∧ v.2 ≤ 100)

def T (v : ℝ × ℝ) : set (ℝ × ℝ) := 
  -- translated square S centered at v (details omitted here)
  sorry

def contains_integer_points_exactly_three (S : set (ℝ × ℝ)) : Prop := 
  -- S contains exactly three integer coordinate points
  sorry

theorem probability_exactly_three_integer_points :
  ∀ (v : ℝ × ℝ), point_uniform_random v →
    contains_integer_points_exactly_three (T v) →
      (∑ v in (range (100 + 1)).product (range (100 + 1)), if contains_integer_points_exactly_three (T v) then 1 else 0) / (101 * 101) = 3 / 100 :=
by sorry

end probability_exactly_three_integer_points_l108_108308


namespace max_area_triangle_l108_108930

theorem max_area_triangle (a b c S : ℝ) (h₁ : S = a^2 - (b - c)^2) (h₂ : b + c = 8) :
  S ≤ 64 / 17 :=
sorry

end max_area_triangle_l108_108930


namespace find_angle_A_find_triangle_area_l108_108251

noncomputable def sin_formula (A B : ℝ) : Prop := 
    sin A = sin B * sin B + sin (π / 4 + B) * sin (π / 4 - B)

noncomputable def dot_product_condition (AB AC : ℝ) : Prop := 
    AB * AC * cos (π / 6) = 12

noncomputable def triangle_area (AB AC : ℝ) : ℝ := 
    (1 / 2) * AB * AC * sin (π / 6)

theorem find_angle_A (A B : ℝ) (h : sin_formula A B): 
    A = π / 6 :=
by
  sorry

theorem find_triangle_area (AB AC : ℝ) (h1 : AB * AC = 8 * sqrt 3) :
    triangle_area AB AC = 2 * sqrt 3 :=
by
  sorry

end find_angle_A_find_triangle_area_l108_108251


namespace inclination_angle_range_l108_108998

theorem inclination_angle_range (θ : ℝ) : 
  ∃ (α : ℝ), α ∈ ([0, Real.pi/4] ∪ [3 * Real.pi/4, Real.pi)) ∧ ∀ x : ℝ, y = x * Real.cos θ - 1 → α = Real.arctan (Real.cos θ) :=
by
  sorry

end inclination_angle_range_l108_108998


namespace volume_inequality_l108_108496

-- Define the conditions
variables {OA_1 OA_2 OA_3 : ℝ} {A_1 A_2 A_3 B_1 B_2 B_3 C_1 C_2 C_3 : Type}
variables (V_1 V_2 V_3 V : ℝ)

-- Conditions: Distinctness and order of points
axiom h1 : OA_1 > OA_2
axiom h2 : OA_2 > OA_3
axiom h3 : OA_3 > 0

-- Expressions for the volumes
def volume_OA_i_B_i_C_i (a_i : ℝ) : ℝ := (1/6) * a_i^3
def volume_OA_1_B_2_C_3 (a1 a2 a3 : ℝ) : ℝ := (1/6) * a1 * a2 * a3

-- Statements that assigns the given values to volumes
axiom h4 : V_1 = volume_OA_i_B_i_C_i OA_1
axiom h5 : V_2 = volume_OA_i_B_i_C_i OA_2
axiom h6 : V_3 = volume_OA_i_B_i_C_i OA_3
axiom h7 : V = volume_OA_1_B_2_C_3 OA_1 OA_2 OA_3

-- Statement to be proven
theorem volume_inequality : V_1 + V_2 + V_3 ≥ 3 * V :=
begin
  sorry
end

end volume_inequality_l108_108496


namespace area_of_triangle_N1N2N3_l108_108803

variables (A B C D E F N1 N2 N3 : Type) [metric_space A] [metric_space B]
          [metric_space C] [metric_space D] [metric_space E] [metric_space F]
          [metric_space N1] [metric_space N2] [metric_space N3]
          (ΔABC ΔADC ΔN1DC ΔN2EA ΔN3FB ΔN2N1CE ΔN1N2N3 : ℝ)

-- Definitions of one-half conditions
def halfSideCD (ΔABC ΔADC : ℝ) : Prop := ΔADC = (1 / 2) * ΔABC
def halfSideAE (ΔABC ΔN2EA : ℝ) : Prop := ΔN2EA = (1 / 8) * ΔABC
def halfSideBF (ΔABC ΔN3FB : ℝ) : Prop := ΔN3FB = (1 / 8) * ΔABC

-- Definitions of areas
def areaDelta (ΔABC ΔADC ΔN1DC ΔN2EA ΔN3FB ΔN2N1CE ΔN1N2N3 : ℝ) : Prop :=
  ΔADC = (1 / 2) * ΔABC ∧
  ΔN1DC = (1 / 4) * ΔADC ∧
  ΔN2EA = (1 / 8) * ΔABC ∧
  ΔN3FB = (1 / 8) * ΔABC ∧
  ΔN2N1CE = ΔADC - ΔN1DC - ΔN2EA ∧
  ΔN1N2N3 = ΔABC - 3 * ΔN2EveryN2N1CE - 3 * ΔN2EA - 3 * ΔN1DC

-- The theorem statement
theorem area_of_triangle_N1N2N3
  (h1 : halfSideCD ΔABC ΔADC)
  (h2 : halfSideAE ΔABC ΔN2EA)
  (h3 : halfSideBF ΔABC ΔN3FB)
  (h4 : areaDelta ΔABC ΔADC ΔN1DC ΔN2EA ΔN3FB ΔN2N1CE ΔN1N2N3) :
  ΔN1N2N3 = (1 / 8) * ΔABC :=
sorry  -- Proof omitted

end area_of_triangle_N1N2N3_l108_108803


namespace number_of_equal_sum_partitions_l108_108456

open Finset

def is_equal_sum_partition (A B : Finset ℕ) (M : Finset ℕ) :=
  A ∪ B = M ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id

theorem number_of_equal_sum_partitions : 
  let M := (finset.range 13).erase 0 in 
  ∃ (A B : Finset ℕ) (n : ℕ), n = 6 ∧ M.card = 12 ∧ 
  (A.card = n ∧ B.card = n) ∧ is_equal_sum_partition A B M ∧ 
  finset.card (finset.filter (λ (x : Finset ℕ × Finset ℕ), is_equal_sum_partition x.1 x.2 M) 
  (finset.powerset M ×ˢ finset.powerset M)) = 29 := 
sorry

end number_of_equal_sum_partitions_l108_108456


namespace sum_of_interior_angles_pentagon_l108_108402

theorem sum_of_interior_angles_pentagon : (5 - 2) * 180 = 540 := by
  sorry

end sum_of_interior_angles_pentagon_l108_108402


namespace part_a_part_b_l108_108541

/-- Define rational non-integer numbers x and y -/
structure RationalNonInteger (x y : ℚ) :=
  (h1 : x.denom ≠ 1)
  (h2 : y.denom ≠ 1)

/-- Part (a): There exist rational non-integer numbers x and y 
    such that 19x + 8y and 8x + 3y are integers -/
theorem part_a : ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x + 8*y ∈ ℤ) ∧ (8*x + 3*y ∈ ℤ) :=
by
  sorry

/-- Part (b): There do not exist rational non-integer numbers x and y 
    such that 19x^2 + 8y^2 and 8x^2 + 3y^2 are integers -/
theorem part_b : ¬ ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x^2 + 8*y^2 ∈ ℤ) ∧ (8*x^2 + 3*y^2 ∈ ℤ) :=
by
  sorry

end part_a_part_b_l108_108541


namespace select_students_exactly_one_female_l108_108413

theorem select_students_exactly_one_female :
  let males_a := 5
  let females_a := 3
  let males_b := 6
  let females_b := 2
  let comb := λ n k, Nat.choose n k
  (comb females_a 1 * comb males_a 1 * comb males_b 2) + (comb males_a 2 * comb males_b 1 * comb females_b 1) = 345 :=
by
  intros
  sorry

end select_students_exactly_one_female_l108_108413


namespace probability_of_event_A_probability_of_event_B_l108_108076

open Probability Theory

/-
Define the sets representing outcomes of rolling two dice.
-/
def outcomes : finset (ℕ × ℕ) :=
  finset.product (finset.range 6) (finset.range 6)

/-
Define event A: the sum of the numbers rolled is 8.
-/
def event_A (x y : ℕ) : Prop :=
  (x + 1) + (y + 1) = 8

/-
Define event B: the sum of the squares of the numbers rolled is ≤ 12.
-/
def event_B (x y : ℕ) : Prop :=
  (x + 1)^2 + (y + 1)^2 ≤ 12

/-
Calculate the probability of event A.
-/
theorem probability_of_event_A :
  (finset.card (outcomes.filter (λ p, event_A p.1 p.2))).to_rat / (outcomes.card).to_rat = 5 / 36 :=
sorry

/-
Calculate the probability of event B.
-/
theorem probability_of_event_B :
  (finset.card (outcomes.filter (λ p, event_B p.1 p.2))).to_rat / (outcomes.card).to_rat = 6 / 36 :=
sorry

end probability_of_event_A_probability_of_event_B_l108_108076


namespace martha_trip_gallons_l108_108521

-- Define the fuel efficiency of Darlene's car
def darlene_mpg : ℝ := 20

-- Define the fuel efficiency of Martha's car as half of Darlene's
def martha_mpg : ℝ := darlene_mpg / 2

-- Define the distance of the trip
def trip_distance : ℝ := 300

-- Define the function to calculate the gallons required
def gallons_required (distance : ℝ) (mpg : ℝ) : ℝ := distance / mpg

-- Theorem stating the required gallons for Martha's car to make the trip
theorem martha_trip_gallons : 
  gallons_required trip_distance martha_mpg = 30 :=
by
  sorry

end martha_trip_gallons_l108_108521


namespace hyperbola_equation_l108_108949

-- Definition of the ellipse given in the problem
def ellipse (x y : ℝ) := y^2 / 5 + x^2 = 1

-- Definition of the conditions for the hyperbola:
-- 1. The hyperbola shares a common focus with the ellipse.
-- 2. Distance from the focus to the asymptote of the hyperbola is 1.
def hyperbola (x y : ℝ) (c : ℝ) :=
  ∃ a b : ℝ, c = 2 ∧ a^2 + b^2 = c^2 ∧
             (b = 1 ∧ y = if x = 0 then 0 else x * (a / b))

-- The statement we need to prove
theorem hyperbola_equation : 
  (∃ a b : ℝ, ellipse x y ∧ hyperbola x y 2 ∧ b = 1 ∧ a^2 = 3) → 
  (y^2 / 3 - x^2 = 1) :=
sorry

end hyperbola_equation_l108_108949


namespace perfect_squares_sum_of_three_odd_composites_eq_l108_108047

def is_odd_composite (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k + 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ m∣n

def is_sum_of_three_odd_composites (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), is_odd_composite a ∧ is_odd_composite b ∧ is_odd_composite c ∧ n = a + b + c

noncomputable def set_of_perfect_squares_sum_of_three_odd_composites : set ℕ :=
  { n | ∃ k : ℕ, n = (2 * k + 1) ^ 2 ∧ k ≥ 3 }

theorem perfect_squares_sum_of_three_odd_composites_eq : 
  { n | is_sum_of_three_odd_composites n ∧ ∃ m : ℕ, n = m ^ 2 } =
  set_of_perfect_squares_sum_of_three_odd_composites :=
sorry

end perfect_squares_sum_of_three_odd_composites_eq_l108_108047


namespace shift_sin_cos_eq_l108_108419

noncomputable def phase_shift (f g : ℝ → ℝ) (s : ℝ) : Prop := 
  ∀ x, f (x + s) = g x

theorem shift_sin_cos_eq {x : ℝ} : 
  phase_shift (λ x, sqrt 2 * cos (3 * x)) (λ x, sin (3 * x) + cos (3 * x)) (-π / 4) :=
by
  sorry

end shift_sin_cos_eq_l108_108419


namespace base6_to_decimal_l108_108229

theorem base6_to_decimal (m : ℕ) (h : 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 4934) : m = 4 :=
by
  sorry

end base6_to_decimal_l108_108229


namespace betty_paid_total_l108_108103

def cost_slippers (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_lipsticks (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_hair_colors (count : ℕ) (price : ℝ) : ℝ := count * price

def total_cost := 
  cost_slippers 6 2.5 +
  cost_lipsticks 4 1.25 +
  cost_hair_colors 8 3

theorem betty_paid_total :
  total_cost = 44 := 
  sorry

end betty_paid_total_l108_108103


namespace part1_proof_l108_108693

variable (a r : ℝ) (f : ℝ → ℝ)

axiom a_gt_1 : a > 1
axiom r_gt_1 : r > 1

axiom f_condition : ∀ x > 0, f x * f x ≤ a * x * f (x / a)
axiom f_bound : ∀ x, 0 < x ∧ x < 1 / 2^2005 → f x < 2^2005

theorem part1_proof : ∀ x > 0, f x ≤ a^(1 - r) * x := 
by 
  sorry

end part1_proof_l108_108693


namespace gcd_390_455_546_l108_108384

theorem gcd_390_455_546 :
  Nat.gcd (Nat.gcd 390 455) 546 = 13 := 
sorry

end gcd_390_455_546_l108_108384


namespace minimum_pressure_l108_108329

theorem minimum_pressure (V T V0 T0 a b c R : ℝ) 
  (h_eq : (V / V0 - a)^2 + (T / T0 - b)^2 = c^2)
  (h_cond : c^2 < a^2 + b^2) :
  ∃ Pmin : ℝ, Pmin = (R * T0 / V0) * (a * sqrt (a^2 + b^2 - c^2) - b * c) / (b * sqrt (a^2 + b^2 - c^2) + a * c) :=
by sorry

end minimum_pressure_l108_108329


namespace probability_sum_is_odd_l108_108576

theorem probability_sum_is_odd (S : Finset ℕ) (h_S : S = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
    (h_card : S.card = 12) :
  let choices := S.powerset.filter (λ t, t.card = 4),
      odd_sum_choices := choices.filter (λ t, t.sum % 2 = 1) in
  (odd_sum_choices.card : ℚ) / choices.card = 1 / 3 :=
by  
  sorry

end probability_sum_is_odd_l108_108576


namespace diff_evens_odds_sum_l108_108174

theorem diff_evens_odds_sum {a : ℕ → ℕ} (d : ℕ) (h_d : d = 2) (n : ℕ) (h_n : n = 20)
  (h_arith_seq : ∀ k : ℕ, a (k + 1) = a k + d) :
  let S_e := ∑ k in Finset.range (n / 2), a (2 * (k + 1))
  let S_o := ∑ k in Finset.range (n / 2), a (2 * k + 1)
  in S_e - S_o = 20 :=
by
  sorry

end diff_evens_odds_sum_l108_108174


namespace quotient_remainder_div_by_18_l108_108046

theorem quotient_remainder_div_by_18 (M q : ℕ) (h : M = 54 * q + 37) : 
  ∃ k r, M = 18 * k + r ∧ r < 18 ∧ k = 3 * q + 2 ∧ r = 1 :=
by sorry

end quotient_remainder_div_by_18_l108_108046


namespace find_x_l108_108550

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 182) : x = 13 :=
sorry

end find_x_l108_108550


namespace exists_function_passing_through_point_l108_108446

-- Define the function that satisfies f(2) = 0
theorem exists_function_passing_through_point : ∃ f : ℝ → ℝ, f 2 = 0 := 
sorry

end exists_function_passing_through_point_l108_108446


namespace rob_read_pages_l108_108349

-- Definitions for the given conditions
def planned_reading_time_hours : ℝ := 3
def fraction_of_planned_time : ℝ := 3 / 4
def minutes_per_hour : ℝ := 60
def reading_rate_pages_per_minute : ℝ := 1 / 15

-- Helper definition to convert hours to minutes
def planned_reading_time_minutes : ℝ := planned_reading_time_hours * minutes_per_hour

-- Total actual reading time in minutes
def actual_reading_time_minutes : ℝ := fraction_of_planned_time * planned_reading_time_minutes

-- Number of pages read
def number_of_pages_read : ℝ := actual_reading_time_minutes * reading_rate_pages_per_minute

-- The theorem to prove
theorem rob_read_pages : number_of_pages_read = 9 :=
by
  -- Insert structured proof steps here
  sorry

end rob_read_pages_l108_108349


namespace C_completes_work_in_4_days_l108_108072

theorem C_completes_work_in_4_days
  (A_days : ℕ)
  (B_efficiency : ℕ → ℕ)
  (C_efficiency : ℕ → ℕ)
  (hA : A_days = 12)
  (hB : ∀ {x}, B_efficiency x = x * 3 / 2)
  (hC : ∀ {x}, C_efficiency x = x * 2) :
  (1 / (1 / (C_efficiency (B_efficiency A_days)))) = 4 := by
  sorry

end C_completes_work_in_4_days_l108_108072


namespace find_smaller_between_C_and_D_l108_108813

theorem find_smaller_between_C_and_D :
  ∃ A B C D E F : ℕ,
    {A, B, C, D, E, F} = {1, 2, 3, 4, 5, 6} ∧
    (A + B) % 2 = 0 ∧
    (C + D) % 3 = 0 ∧
    (E + F) % 5 = 0 ∧
    C = 1 ∧ D = 2 := sorry

end find_smaller_between_C_and_D_l108_108813


namespace value_of_a_cube_l108_108610

-- We define the conditions given in the problem.
def A (a : ℤ) : Set ℤ := {5, a^2 + 2 * a + 4}
def a_satisfies (a : ℤ) : Prop := 7 ∈ A a

-- We state the theorem.
theorem value_of_a_cube (a : ℤ) (h1 : a_satisfies a) : a^3 = 1 ∨ a^3 = -27 := by
  sorry

end value_of_a_cube_l108_108610


namespace exist_circles_with_ratio_sqrt2_l108_108589

theorem exist_circles_with_ratio_sqrt2 
  (M : set ℝ^2) 
  (convex_M : convex M) 
  (rotation_invariant_M : ∃ O : ℝ^2, ∀ P ∈ M, rotate O (π / 2) P ∈ M) :
  ∃ R r : ℝ, 
    (∃ C1 C2 : set ℝ^2, (∀ P ∈ M, dist O P ≤ R) ∧ (∀ P ∈ C2, dist O P ≤ r) ∧ (∀ P ∈ C2, P ∈ M)) ∧ 
    R / r = real.sqrt 2 :=
sorry

end exist_circles_with_ratio_sqrt2_l108_108589


namespace nonneg_diff_roots_l108_108790

theorem nonneg_diff_roots : 
  let a : ℝ := 40
  let b : ℝ := 250
  let c : ℝ := -50
  ∀ x : ℝ, (x^2 + a * x + b = c) →
  let roots := [(-a - sqrt(a^2 - 4 * (b - c))) / 2, (-a + sqrt(a^2 - 4 * (b - c))) / 2]
  ∃ (r1 r2 : ℝ), 
    r1 * r2 = roots.nth 0 ∧ 
    r1 * r2 = roots.nth 1 ∧
    abs (r1 - r2) = 20 :=
  sorry

end nonneg_diff_roots_l108_108790


namespace arithmetic_sequence_solution_l108_108123

theorem arithmetic_sequence_solution (x : ℚ) :
  let a1 := 1 / 3;
      a2 := x - 2;
      a3 := 4 * x;
      d1 := a2 - a1;
      d2 := a3 - a2
  in d1 = d2 → x = -13 / 6 :=
by
  sorry

end arithmetic_sequence_solution_l108_108123


namespace yvette_final_bill_l108_108287

def cost_alicia : ℝ := 7.50
def cost_brant : ℝ := 10.00
def cost_josh : ℝ := 8.50
def cost_yvette : ℝ := 9.00
def tip_rate : ℝ := 0.20

def total_cost := cost_alicia + cost_brant + cost_josh + cost_yvette
def tip := tip_rate * total_cost
def final_bill := total_cost + tip

theorem yvette_final_bill :
  final_bill = 42.00 :=
  sorry

end yvette_final_bill_l108_108287


namespace intersection_point_at_neg4_l108_108003

def f (x : Int) (b : Int) : Int := 4 * x + b
def f_inv (y : Int) (b : Int) : Int := (y - b) / 4

theorem intersection_point_at_neg4 (a b : Int) (h1 : f (-4) b = a) (h2 : f_inv (-4) b = a) : a = -4 := 
by 
  sorry

end intersection_point_at_neg4_l108_108003


namespace sum_of_powers_l108_108586

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem sum_of_powers (x = Complex.exp (2 * Real.pi * Complex.I / 5)) :
  1 + x^4 + x^8 + x^{12} + x^{16} = 0 :=
sorry

end sum_of_powers_l108_108586


namespace box_height_l108_108844

theorem box_height
  (V_wooden : ℕ := 336000000) -- volume of the wooden box in cm^3
  (V_rectangular : ℕ := λ h : ℕ, 28 * h) -- volume of one rectangular box in cm^3
  (max_boxes : ℕ := 2000000) -- maximum number of boxes that can be carried
  (height : ℕ) -- height of the rectangular box in cm
  (eqn : max_boxes * (V_rectangular height) = V_wooden) :
  height = 6 :=
by
  sorry

end box_height_l108_108844


namespace find_alpha_l108_108182

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

def is_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y ∈ s, x < y → f y ≤ f x

theorem find_alpha :
  ∀ α ∈ ({-2, -1, -1/2, 1/3, 1/2, 1, 2, 3} : Set ℝ),
  is_even_function (power_function α) ∧ is_decreasing_on (power_function α) (Set.Ioi 0) →
  α = -2 := by
  sorry

end find_alpha_l108_108182


namespace interest_rate_determination_l108_108647

-- Problem statement
theorem interest_rate_determination (P r : ℝ) :
  (50 = P * r * 2) ∧ (51.25 = P * ((1 + r) ^ 2 - 1)) → r = 0.05 :=
by
  intros h
  sorry

end interest_rate_determination_l108_108647


namespace minimum_g7_l108_108501

def is_tenuous (g : ℕ → ℤ) : Prop :=
∀ x y : ℕ, 0 < x → 0 < y → g x + g y > x^2

noncomputable def min_possible_value_g7 (g : ℕ → ℤ) (h : is_tenuous g) 
  (h_sum : (g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 + g 9 + g 10) = 
             -29) : ℤ :=
g 7

theorem minimum_g7 (g : ℕ → ℤ) (h : is_tenuous g)
  (h_sum : (g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 + g 9 + g 10) = 
             -29) :
  min_possible_value_g7 g h h_sum = 49 :=
sorry

end minimum_g7_l108_108501


namespace number_of_pages_correct_number_of_ones_correct_l108_108292

noncomputable def number_of_pages (total_digits : ℕ) : ℕ :=
  let single_digit_odd_pages := 5
  let double_digit_odd_pages := 45
  let triple_digit_odd_pages := (total_digits - (single_digit_odd_pages + 2 * double_digit_odd_pages)) / 3
  single_digit_odd_pages + double_digit_odd_pages + triple_digit_odd_pages

theorem number_of_pages_correct : number_of_pages 125 = 60 :=
by sorry

noncomputable def number_of_ones (total_digits : ℕ) : ℕ :=
  let ones_in_units_place := 12
  let ones_in_tens_place := 18
  let ones_in_hundreds_place := 10
  ones_in_units_place + ones_in_tens_place + ones_in_hundreds_place

theorem number_of_ones_correct : number_of_ones 125 = 40 :=
by sorry

end number_of_pages_correct_number_of_ones_correct_l108_108292


namespace initial_shed_bales_zero_l108_108777

def bales_in_barn_initial : ℕ := 47
def bales_added_by_benny : ℕ := 35
def bales_in_barn_total : ℕ := 82

theorem initial_shed_bales_zero (b_shed : ℕ) :
  bales_in_barn_initial + bales_added_by_benny = bales_in_barn_total → b_shed = 0 :=
by
  intro h
  sorry

end initial_shed_bales_zero_l108_108777


namespace molecular_weight_conservation_l108_108867

theorem molecular_weight_conservation :
  let Ca := 40.08
  let Br := 79.904
  let H := 1.008
  let O := 15.999
  (1 * Ca + 2 * Br) + (2 * (2 * H + O)) = (1 * Ca + 2 * O + 2 * H) + (2 * (H + Br)) :=
by
  have CaBr2_weight : 1 * Ca + 2 * Br = 199.888 := sorry
  have H2O_weight : 2 * (2 * H + O) = 36.03 := sorry
  have CaOH2_weight : 1 * Ca + 2 * O + 2 * H = 74.094 := sorry
  have HBr_weight : 2 * (H + Br) = 161.824 := sorry
  rw [CaBr2_weight, H2O_weight, CaOH2_weight, HBr_weight]
  exact rfl

end molecular_weight_conservation_l108_108867


namespace A_beats_B_by_distance_l108_108470

theorem A_beats_B_by_distance :
  ∀ (t_A t_B : ℕ) (d : ℕ) (s_A s_B : ℝ),
    t_A = 28 → t_B = 32 → d = 128 →
    s_A = d / t_A → s_B = d / t_B →
    (s_A * t_B) - d = 18.24 :=
by
  intros t_A t_B d s_A s_B h_tA h_tB h_d h_sA h_sB
  have hA_speed : s_A = 4.57 := by
    simp [h_d, h_tA, div_eq 128 28 4.57]
  have hB_speed : s_B = 4 := by
    simp [h_d, h_tB, div_eq 128 32 4]
  rw [h_tA, h_tB, h_d, hA_speed, hB_speed]
  norm_num
  sorry

end A_beats_B_by_distance_l108_108470


namespace conic_section_eccentricity_l108_108937

noncomputable def eccentricity (m : ℝ) : ℝ :=
  if m = 3 then real.sqrt 2 / real.sqrt 3 else 2

theorem conic_section_eccentricity (m : ℝ) (h : m^2 = 9) :
  eccentricity m = real.sqrt 6 / 3 ∨ eccentricity m = 2 :=
by
  -- We assume m^2 = 9, therefore m = 3 or m = -3
  have h1 : m = 3 ∨ m = -3 := by sorry
  cases h1 with h2 h3
  case inl { rw [h2, eccentricity], simp }
  case inr { rw [h3, eccentricity], simp }
  sorry

end conic_section_eccentricity_l108_108937


namespace right_triangle_set_C_l108_108050

theorem right_triangle_set_C :
  ∃ (a b c : ℕ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_set_C_l108_108050


namespace simplify_cos2_minus_sin2_l108_108354

theorem simplify_cos2_minus_sin2 {α : ℝ} :
    (cos (π / 4 - α)) ^ 2 - (sin (π / 4 - α)) ^ 2 = sin (2 * α) :=
by
  sorry

end simplify_cos2_minus_sin2_l108_108354


namespace num_sequences_satisfying_conditions_l108_108638

/-- The number of sequences of 7 digits y_1, y_2, ..., y_7 where no two adjacent y_i have the same
parity and y_1 is even is 78125. Leading zeroes are allowed. -/
theorem num_sequences_satisfying_conditions : 
  let sequences := list (fin 10) in
  let is_even (n : nat) := n % 2 = 0 in
  let is_odd (n : nat) := n % 2 = 1 in
  let generates (y : list (fin 10)) := 
    y.head.1 % 2 = 0 ∧ 
    ∀ (i : nat), i < 6 → (is_even y[i + 1].val ↔ is_odd y[i].val) in
  (list.filter generates (sequences.pi (list.replicate 7 (fin 10)))) = 78125 :=
by
  sorry

end num_sequences_satisfying_conditions_l108_108638


namespace bead_probability_l108_108054

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem bead_probability : 
  let total_beads : ℕ := 4 + 3 + 2 in
  let total_arrangements : ℕ := factorial total_beads / (factorial 4 * factorial 3 * factorial 2) in
  let valid_arrangements : ℕ := binomial 6 2 in
  let probability : ℚ := valid_arrangements / total_arrangements in
  probability = 1 / 84 :=
by
  sorry

end bead_probability_l108_108054


namespace solution_set_of_inequality_l108_108766

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + 2 * x - 3 > 0 } = { x : ℝ | x < -3 ∨ x > 1 } :=
sorry

end solution_set_of_inequality_l108_108766


namespace find_a_sq_plus_b_sq_l108_108644

-- Variables and conditions
variables (a b : ℝ)
-- Conditions from the problem
axiom h1 : a - b = 3
axiom h2 : a * b = 9

-- The proof statement
theorem find_a_sq_plus_b_sq (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 9) : a^2 + b^2 = 27 :=
by {
  sorry
}

end find_a_sq_plus_b_sq_l108_108644


namespace odd_function_at_negative_two_l108_108944

theorem odd_function_at_negative_two 
  (f : ℝ → ℝ) 
  (h_odd : ∀ x, f(-x) = -f(x)) 
  (h_def : ∀ x, 0 ≤ x → f(x) = x * (1 + x)) : 
  f (-2) = -6 := 
by 
  sorry

end odd_function_at_negative_two_l108_108944


namespace first_digit_base9_is_4_l108_108142

-- Definition of the conditions
def base3_num : ℕ := 2 * 3^17 + 1 * 3^16 + 2 * 3^15 + 1 * 3^14 + 1 * 3^13 + 1 * 3^12 + 2 * 3^11 + 2 * 3^10 + 1 * 3^9 + 2 * 3^8 + 1 * 3^7 + 1 * 3^6 + 2 * 3^5 + 1 * 3^4 + 1 * 3^3 + 2 * 3^2 + 1 * 3^1 + 1 * 3^0

-- The proof statement
theorem first_digit_base9_is_4 :
  let N := base3_num in
  let base9_rep := N / 9 ^ (Nat.log N 9) in
  base9_rep = 4 :=
sorry

end first_digit_base9_is_4_l108_108142


namespace supplementary_angle_l108_108796

theorem supplementary_angle (θ : ℝ) (k : ℤ) : (θ = 10) → (∃ k, θ + 250 = k * 360 + 360) :=
by
  sorry

end supplementary_angle_l108_108796


namespace polygon_sides_l108_108772

theorem polygon_sides (sum_of_interior_angles : ℝ) (x : ℝ) (h : sum_of_interior_angles = 1080) : x = 8 :=
by
  sorry

end polygon_sides_l108_108772


namespace water_in_mixture_l108_108420

theorem water_in_mixture (w: ℚ) (s: ℚ) (total_liters: ℚ) :
  let total_parts := w + s in
  let each_part := total_liters / total_parts in
  let water_parts := w in
  water_parts * each_part = 15 / 7 :=
by
  -- Given conditions
  have hw: w = 5 := rfl
  have hs: s = 2 := rfl
  have h_total_liters: total_liters = 3 := rfl
  -- Define total_parts and each_part
  let total_parts := w + s
  let each_part := total_liters / total_parts
  -- Calculate water amount
  have h_total_parts: total_parts = 7 := by rw [hw, hs]; exact by norm_num
  have h_each_part: each_part = 3 / 7 := by rw [h_total_liters, h_total_parts]; exact by norm_num
  -- Prove the answer
  have h_answer : water_parts * each_part = 15 / 7 := by rw [hw, h_each_part]; exact by norm_num
  exact h_answer

end water_in_mixture_l108_108420


namespace bug_crawl_distance_l108_108071

theorem bug_crawl_distance : 
  let start : ℤ := 3
  let first_stop : ℤ := -4
  let second_stop : ℤ := 7
  let final_stop : ℤ := -1
  |first_stop - start| + |second_stop - first_stop| + |final_stop - second_stop| = 26 := 
by
  sorry

end bug_crawl_distance_l108_108071


namespace iran_2004_2005_p1_l108_108816

open EuclideanGeometry

theorem iran_2004_2005_p1
  (ABC : Triangle)
  (O : Point)
  (A' : Midpoint ABC.BC)
  (A'' : IntersectionPoint (Line (ABC.AA') (Circumcircle ABC)))
  (Q_a : on (Line A' A'Q_a) (PerpendicularLine O A'Q_a))
  (P_a : IntersectionPoint (TangentAt (Circumcircle ABC) A'') (Line A' Q_a))
  (Q_b : similar_construction_case_for_B)
  (Q_c : similar_construction_case_for_C) :
  Collinear P_a P_b P_c := 
sorry

end iran_2004_2005_p1_l108_108816


namespace quadratic_completing_square_b_plus_c_l108_108762

theorem quadratic_completing_square_b_plus_c :
  ∃ b c : ℤ, (λ x : ℝ, x^2 - 24 * x + 50) = (λ x, (x + b)^2 + c) ∧ b + c = -106 :=
by
  sorry

end quadratic_completing_square_b_plus_c_l108_108762


namespace initial_population_l108_108455

variable (P : ℕ)

theorem initial_population
  (birth_rate : ℕ := 52)
  (death_rate : ℕ := 16)
  (net_growth_rate : ℚ := 1.2) :
  (P = 3000) :=
by
  sorry

end initial_population_l108_108455


namespace root_in_interval_l108_108891

open Function

noncomputable def f : ℝ → ℝ :=
  λ x, Real.log x + x - 3

theorem root_in_interval : ∃ x ∈ Ioo (2:ℝ) (3: ℝ), f x = 0 :=
by
  sorry

end root_in_interval_l108_108891


namespace smallest_solution_l108_108563

theorem smallest_solution (x : ℝ) (h : x * |x| = 3 * x - 2) : 
  x = 1 ∨ x = 2 ∨ x = (-(3 + Real.sqrt 17)) / 2 :=
by
  sorry

end smallest_solution_l108_108563


namespace range_of_a3_l108_108970

theorem range_of_a3 (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) + a n = 4 * n + 3)
  (h2 : ∀ n : ℕ, n > 0 → a n + 2 * n^2 ≥ 0) 
  : 2 ≤ a 3 ∧ a 3 ≤ 19 := 
sorry

end range_of_a3_l108_108970


namespace area_perimeter_ratio_inequality_l108_108888

-- Define a point D inside the equilateral triangle ABC
variables {A B C D E : Point}
variable (is_equilateral : EquilateralTriangle A B C)
variable (D_inside_ABC : InsideTriangle D A B C)
variable (E_inside_DBC : InsideTriangle E D B C)

-- Main theorem stating the desired inequality
theorem area_perimeter_ratio_inequality :
  (area (Triangle D B C) / (perimeter (Triangle D B C))^2) >
  (area (Triangle E B C) / (perimeter (Triangle E B C))^2) :=
  sorry

end area_perimeter_ratio_inequality_l108_108888


namespace sequences_converge_to_same_limit_l108_108032

noncomputable def a_sequence : ℕ → ℝ
| 0       := 3
| n + 1   := (a_sequence n + b_sequence n) / 2

noncomputable def b_sequence : ℕ → ℝ
| 0       := 1
| n + 1   := 2 * a_sequence n * b_sequence n / (a_sequence n + b_sequence n)

theorem sequences_converge_to_same_limit
  (a_sequence : ℕ → ℝ) (b_sequence : ℕ → ℝ)
  (h_initial_a : a_sequence 0 = 3)
  (h_initial_b : b_sequence 0 = 1)
  (h_a : ∀ n, a_sequence (n + 1) = (a_sequence n + b_sequence n) / 2)
  (h_b : ∀ n, b_sequence (n + 1) = 2 * a_sequence n * b_sequence n / (a_sequence n + b_sequence n)) :
  ∃ L : ℝ, (∀ n, a_sequence n → L) ∧ (∀ n, b_sequence n → L) ∧ L = sqrt 3 :=
sorry

end sequences_converge_to_same_limit_l108_108032


namespace all_statements_true_l108_108201

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.sin x

theorem all_statements_true :
  (∀ x : ℝ, f (-x) = -f (x)) ∧
  (∀ x1 x2 : ℝ, -Real.pi / 2 ≤ x1 ∧ x1 ≤ Real.pi / 2 ∧ -Real.pi / 2 ≤ x2 ∧ x2 ≤ Real.pi / 2 → (x1 + x2) * (f x1 + f x2) ≥ 0) ∧
  (∀ x : ℝ, -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2 → Derivative.deriv f x ≥ 0) :=
  by
   sorry

end all_statements_true_l108_108201


namespace _l108_108591

def point_P := (2, 3 : ℝ)

def inclination_angle := 120 * (Real.pi / 180)

def perpendicular_line :=  ∀ x y: ℝ, x - 2 * y + 1 = 0

def intercept_sum_condition (l : ℝ × ℝ → ℝ) : Prop :=
  let (h, k) := point_P
  let x_intercept := -(k / (l (1, 0))) in
  let y_intercept := -(h / (l (0, 1))) in
  x_intercept + y_intercept = 0

def line_l (l : ℝ × ℝ → ℝ) : Prop :=
  l point_P = 0 ∧
  l (1, Real.tan inclination_angle) = 0 ∧
  (∃ l_perp, perpendicular_line l_perp → ⟦l (l_perp (1, 0), l_perp (0, 1))⟧ ∩ ⟦l (1, Real.tan (inclination_angle - Real.pi/2))⟧ ≠ ∅) ∧
  (l ∘ intercept_sum_condition l)

def main_theorem : ∃ l : ℝ × ℝ → ℝ, line_l l ∧ (l = λ x y, 3 * x - 2 * y) ∨ (l = λ x y, x - y + 1 ) :=
sorry

end _l108_108591


namespace average_speed_ratio_l108_108825

-- Definitions from conditions
def boat_speed_still_water := 20 -- in mph
def current_speed := 4 -- in mph
def downstream_speed := boat_speed_still_water + current_speed -- in mph
def upstream_speed := boat_speed_still_water - current_speed -- in mph
def distance := λ d : ℝ, d -- distance travelled in each direction

-- Theorem statement
theorem average_speed_ratio (d : ℝ) (h : d > 0) :
  let total_time := d / downstream_speed + d / upstream_speed
  let total_distance := 2 * d
  let avg_speed := total_distance / total_time
  (avg_speed / boat_speed_still_water) = (24 / 25) :=
by
  sorry

end average_speed_ratio_l108_108825


namespace game_ends_after_six_rounds_X_probability_distribution_and_expectation_l108_108786

-- Define the conditions and the problem statement for Part 1
theorem game_ends_after_six_rounds :
  let P_A_wins := 2 / 3
  let P_B_wins := 1 / 3
  let P1 := binomial 5 3 * (P_A_wins ^ 3) * (P_B_wins ^ 2) * P_A_wins
  let P2 := binomial 5 3 * (P_B_wins ^ 3) * (P_A_wins ^ 2) * P_B_wins
  P1 + P2 = 200 / 729 :=
by sorry

-- Define the conditions and the problem statement for Part 2
theorem X_probability_distribution_and_expectation :
  let P_A_wins := 2 / 3
  let P_B_wins := 1 / 3
  let P_X2 := P_A_wins ^ 2
  let P_X3 := 2 * P_A_wins * P_B_wins * P_A_wins + P_B_wins ^ 3
  let P_X4 := 3 * (P_B_wins ^ 2) * P_A_wins
  let E_X := 2 * P_X2 + 3 * P_X3 + 4 * P_X4
  (P_X2 = 4 / 9) ∧ (P_X3 = 1 / 3) ∧ (P_X4 = 2 / 9) ∧ (E_X = 25 / 9) :=
by sorry

end game_ends_after_six_rounds_X_probability_distribution_and_expectation_l108_108786


namespace find_x_solution_l108_108153

theorem find_x_solution (x : ℝ) (h : sqrt (x - 5) = 7) : x = 54 :=
sorry

end find_x_solution_l108_108153


namespace triangle_angle_bisector_parallel_l108_108173

theorem triangle_angle_bisector_parallel (A B C D E : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ E] :
  let triangle_ABC := (A, B, C) in
  let bisector_A := bisector angle(A, B, C) in
  let bisector_B := bisector angle(B, C, A) in
  let parallel_CD_bisector_B := line_through(C) ∥ bisector_B in 
  let parallel_CD_bisector_A := line_through(C) ∥ bisector_A in 
  (D = intersection(parallel_CD_bisector_B, bisector_A)) →
  (E = intersection(parallel_CD_bisector_A, bisector_B)) →
  (DE ∥ AB) →
  CA = CB :=
by sorry

end triangle_angle_bisector_parallel_l108_108173


namespace fraction_spent_on_food_l108_108838

theorem fraction_spent_on_food (r c f : ℝ) (l s : ℝ)
  (hr : r = 1/10)
  (hc : c = 3/5)
  (hl : l = 16000)
  (hs : s = 160000)
  (heq : f * s + r * s + c * s + l = s) :
  f = 1/5 :=
by
  sorry

end fraction_spent_on_food_l108_108838


namespace value_of_k_l108_108751

def tangent_line_curve (k a b l : ℝ) : Prop :=
  (3 = k * l + 1) ∧ (y = x^3 + a * x + b)

theorem value_of_k (k a b l : ℝ) (h1 : 3 = k * l + 1) : k = 2 :=
by
  sorry

end value_of_k_l108_108751


namespace solve_for_x_l108_108990

noncomputable def find_x (x : ℝ) : Prop :=
  2^12 = 64^x

theorem solve_for_x (x : ℝ) (h : find_x x) : x = 2 :=
by
  sorry

end solve_for_x_l108_108990


namespace inequality_proof_l108_108696

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  (∑ i in [x, y, z], sqrt (3 * i * (i + y) * (y + z))) ≤ sqrt (4 * (x + y + z) ^ 3) :=
by
  sorry

end inequality_proof_l108_108696


namespace bakery_roll_combinations_l108_108467

theorem bakery_roll_combinations : 
  let kinds := 4
  let total_rolls := 8
  ∃ f : Fin kinds → ℕ,
    (∑ i in Finset.range kinds, f i = total_rolls) ∧
    (∀ i, 1 ≤ f i) :=
  ∑ i in Finset.range (total_rolls - kinds + 1), 
    \binom {total_rolls - kinds + kinds - 1} {kinds - 1} = 
  35 :=
  1
sorry

end bakery_roll_combinations_l108_108467


namespace range_of_t_l108_108969

noncomputable def a_n (n : ℕ) (t : ℝ) : ℝ := -n + t
noncomputable def b_n (n : ℕ) : ℝ := 3^(n-3)
noncomputable def c_n (n : ℕ) (t : ℝ) : ℝ := 
  let a := a_n n t 
  let b := b_n n
  (a + b) / 2 + (|a - b|) / 2

theorem range_of_t (t : ℝ) (h : ∀ n : ℕ, n > 0 → c_n n t ≥ c_n 3 t) : 10/3 < t ∧ t < 5 :=
    sorry

end range_of_t_l108_108969


namespace part_a_part_b_part_c_l108_108294

-- Define the conditions
inductive Color
| blue
| red
| green
| yellow

-- Each square can be painted in one of the colors: blue, red, or green.
def square_colors : List Color := [Color.blue, Color.red, Color.green]

-- Each triangle can be painted in one of the colors: blue, red, or yellow.
def triangle_colors : List Color := [Color.blue, Color.red, Color.yellow]

-- Condition that polygons with a common side cannot share the same color
def different_color (c1 c2 : Color) : Prop := c1 ≠ c2

-- Part (a)
theorem part_a : ∃ n : Nat, n = 7 := sorry

-- Part (b)
theorem part_b : ∃ n : Nat, n = 43 := sorry

-- Part (c)
theorem part_c : ∃ n : Nat, n = 667 := sorry

end part_a_part_b_part_c_l108_108294


namespace F_is_rational_function_l108_108673

noncomputable def F (a : ℕ → ℕ) (x : ℝ) : ℝ := ∑' n, (a n) * x^n

theorem F_is_rational_function 
  (a : ℕ → ℕ) 
  (ha : ∀ n, a n = 0 ∨ a n = 1) 
  (F_half_rational : ∃ q : ℚ, F a (1 / 2) = q) :
  ∃ (P Q : polynomial ℤ), F a = λ x, (P.eval x) / (Q.eval x) := 
sorry

end F_is_rational_function_l108_108673


namespace num_distinct_x_intercepts_l108_108634

theorem num_distinct_x_intercepts : 
  ∀ (x : ℝ), 
  (x = 4 ∨ x^2 + 6*x + 8 = 0) ↔ (x = 4 ∨ x = -4 ∨ x = -2) → 
  (Finset.card (Finset.ofList [4, -4, -2]) = 3) := 
by 
  sorry

end num_distinct_x_intercepts_l108_108634


namespace arithmetic_sequence_middle_term_l108_108273

variable (x y z : ℝ)

theorem arithmetic_sequence_middle_term :
  (23, x, y, z, 47) → y = 35 :=
by
  intro h
  have h1 : y = (23 + 47) / 2 := sorry
  have h2 : y = 35 := sorry
  exact h2

end arithmetic_sequence_middle_term_l108_108273


namespace hillside_camp_boys_percentage_l108_108277

theorem hillside_camp_boys_percentage (B G : ℕ) 
  (h1 : B + G = 60) 
  (h2 : G = 6) : (B: ℕ) / 60 * 100 = 90 :=
by
  sorry

end hillside_camp_boys_percentage_l108_108277


namespace melanie_books_bought_l108_108322

-- Define the initial and final number of books
def initialBooks : ℕ := 41
def finalBooks : ℕ := 87

-- Define the number of books bought at the yard sale
def booksBought (initialBooks finalBooks : ℕ) : ℕ :=
  finalBooks - initialBooks

-- Theorem stating that Melanie bought 46 books at the yard sale
theorem melanie_books_bought : booksBought initialBooks finalBooks = 46 := 
  by
    simp [booksBought, initialBooks, finalBooks]
    simp [Nat.sub]

-- Proof is to be completed (placeholder)
#check melanie_books_bought -- Ensure statement is valid

end melanie_books_bought_l108_108322


namespace monotonicity_of_f_l108_108392

theorem monotonicity_of_f (a b : ℝ) (h : 2^a + 2^b = 1):
  monotone_on (λ x : ℝ, x^2 - 2 * (a + b) * x + 2) (Set.Icc (-2 : ℝ) 2) :=
by 
  sorry

end monotonicity_of_f_l108_108392


namespace constant_term_in_expansion_eq_21_l108_108430

theorem constant_term_in_expansion_eq_21 :
  let f := λ (x : ℝ), x^3 + x^2 + 3
  let g := λ (x : ℝ), 2 * x^4 + x^3 + 7
  (f 0) * (g 0) = 21 := by
  sorry

end constant_term_in_expansion_eq_21_l108_108430


namespace part_a_part_b_l108_108539

/-- Define rational non-integer numbers x and y -/
structure RationalNonInteger (x y : ℚ) :=
  (h1 : x.denom ≠ 1)
  (h2 : y.denom ≠ 1)

/-- Part (a): There exist rational non-integer numbers x and y 
    such that 19x + 8y and 8x + 3y are integers -/
theorem part_a : ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x + 8*y ∈ ℤ) ∧ (8*x + 3*y ∈ ℤ) :=
by
  sorry

/-- Part (b): There do not exist rational non-integer numbers x and y 
    such that 19x^2 + 8y^2 and 8x^2 + 3y^2 are integers -/
theorem part_b : ¬ ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x^2 + 8*y^2 ∈ ℤ) ∧ (8*x^2 + 3*y^2 ∈ ℤ) :=
by
  sorry

end part_a_part_b_l108_108539


namespace cuboid_distance_properties_l108_108655

theorem cuboid_distance_properties (cuboid : Type) :
  (∃ P : cuboid → ℝ, ∀ V1 V2 : cuboid, P V1 = P V2) ∧
  ¬ (∃ Q : cuboid → ℝ, ∀ E1 E2 : cuboid, Q E1 = Q E2) ∧
  ¬ (∃ R : cuboid → ℝ, ∀ F1 F2 : cuboid, R F1 = R F2) := 
sorry

end cuboid_distance_properties_l108_108655


namespace constant_area_sum_l108_108338

noncomputable def area_of_triangle {V : Type*} [inner_product_space ℝ V] (a b c : V) : ℝ :=
1 / 2 * real.sqrt (abs (inner_product (b - a) (c - a))^2)

variables {V : Type*} [inner_product_space ℝ V]

theorem constant_area_sum
  {n : ℕ} (n_pos : 0 < n)
  (A : fin (2 * n) → V) -- vertices of the 2n-gon
  (P1 P2 P3 : V) -- non-collinear points inside the 2n-gon
  (h_collinear : ¬ collinear ℝ ({P1, P2, P3} : set V))
  (c : ℝ)
  (hP1 : ∑ i in finset.range n, area_of_triangle (A (2*i)) (A (2*i+1)) P1 = c)
  (hP2 : ∑ i in finset.range n, area_of_triangle (A (2*i)) (A (2*i+1)) P2 = c)
  (hP3 : ∑ i in finset.range n, area_of_triangle (A (2*i)) (A (2*i+1)) P3 = c)
  (P : V) -- any internal point
  (hP : P ∈ convex_hull ℝ (set.range A)) :
  ∑ i in finset.range n, area_of_triangle (A (2*i)) (A (2*i+1)) P = c :=
begin
  sorry
end

end constant_area_sum_l108_108338


namespace wallpaper_job_completion_l108_108029

theorem wallpaper_job_completion (x : ℝ) (y : ℝ) 
  (h1 : ∀ a b : ℝ, (a = 1.5) → (7/x + (7-a)/(x-3) = 1)) 
  (h2 : y = x - 3) 
  (h3 : x - y = 3) : 
  (x = 14) ∧ (y = 11) :=
sorry

end wallpaper_job_completion_l108_108029


namespace find_x_l108_108812

variables (x : ℝ)

theorem find_x : (x / 4) * 12 = 9 → x = 3 :=
by
  sorry

end find_x_l108_108812


namespace line_equation_l108_108015

-- Define vectors and projections
def vector2d (x y : ℝ) := (x, y)

-- Define projection operation
def proj (a v : ℝ × ℝ) : ℝ × ℝ :=
  let (ax, ay) := a
  let (vx, vy) := v
  let dot_product := ax * vx + ay * vy
  let norm_squared := ax^2 + ay^2
  let scalar := dot_product / norm_squared
  (scalar * ax, scalar * ay)

-- Define given conditions
def v := vector2d 5 (-1)
def target_projection := vector2d (-5/2) (-1)

-- Define theorem to be proved
theorem line_equation : 
  (proj v (vector2d x y) = target_projection) → 
  (y = (-5/2 : ℝ) * x - 29/4) := 
by
  sorry

end line_equation_l108_108015


namespace Bennett_has_6_brothers_l108_108497

theorem Bennett_has_6_brothers (num_aaron_brothers : ℕ) (num_bennett_brothers : ℕ) 
  (h1 : num_aaron_brothers = 4) 
  (h2 : num_bennett_brothers = 2 * num_aaron_brothers - 2) : 
  num_bennett_brothers = 6 := by
  sorry

end Bennett_has_6_brothers_l108_108497


namespace max_value_of_a_l108_108688
noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

theorem max_value_of_a (a : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y) → a ≤ 1 := 
sorry

end max_value_of_a_l108_108688


namespace yvettes_final_bill_l108_108286

theorem yvettes_final_bill :
  let alicia : ℝ := 7.5
  let brant : ℝ := 10
  let josh : ℝ := 8.5
  let yvette : ℝ := 9
  let tip_percentage : ℝ := 0.2
  ∃ final_bill : ℝ, final_bill = (alicia + brant + josh + yvette) * (1 + tip_percentage) ∧ final_bill = 42 :=
by
  sorry

end yvettes_final_bill_l108_108286


namespace fraction_of_tips_in_august_is_five_eighths_l108_108053

-- Definitions
def average_tips (other_tips_total : ℤ) (n : ℤ) : ℤ := other_tips_total / n
def total_tips (other_tips : ℤ) (august_tips : ℤ) : ℤ := other_tips + august_tips
def fraction (numerator : ℤ) (denominator : ℤ) : ℚ := (numerator : ℚ) / (denominator : ℚ)

-- Given conditions
variables (A : ℤ) -- average monthly tips for the other 6 months (March to July and September)
variables (other_months : ℤ := 6)
variables (tips_total_other : ℤ := other_months * A) -- total tips for the 6 other months
variables (tips_august : ℤ := 10 * A) -- tips for August
variables (total_tips_all : ℤ := tips_total_other + tips_august) -- total tips for all months

-- Prove the statement
theorem fraction_of_tips_in_august_is_five_eighths :
  fraction tips_august total_tips_all = 5 / 8 := by sorry

end fraction_of_tips_in_august_is_five_eighths_l108_108053


namespace f_is_odd_f_is_monotonic_solve_inequality_l108_108204

def f (x : ℝ) : ℝ := x / (x^2 + 1)

-- Question 1: Prove that f(x) is an odd function
theorem f_is_odd (x : ℝ) (h : -1 < x ∧ x < 1) : f (-x) = -f x := 
  sorry

-- Question 2: Prove that f(x) is monotonically increasing on (-1, 1)
theorem f_is_monotonic (x1 x2 : ℝ) (h1 : -1 < x1 ∧ x1 < 1) (h2 : -1 < x2 ∧ x2 < 1) (h3 : x1 < x2) : f x1 < f x2 := 
  sorry

-- Question 3: Solving the inequality f(2x - 1) + f(x) < 0 for f(x) = x / (x^2 + 1)
theorem solve_inequality (x : ℝ) (h : 0 < x ∧ x < 1 / 3) : f (2*x - 1) + f x < 0 := 
  sorry

end f_is_odd_f_is_monotonic_solve_inequality_l108_108204


namespace correct_propositions_l108_108979

-- Define the propositions as boolean conditions
def proposition1 (α β : Plane) (m n : Line) : Prop :=
  (α ∩ β = m) → (n ⊆ α) → (m ∥ n) ∨ (∃ p, p ∈ m ∧ p ∈ n)

def proposition2 (α β : Plane) (m n : Line) : Prop :=
  (α ∥ β) → (m ⊆ α) → (n ⊆ β) → (m ∥ n)

def proposition3 (α : Plane) (m n : Line) : Prop :=
  (m ∥ α) → (m ∥ n) → (n ∥ α)

def proposition4 (α β : Plane) (m n : Line) : Prop :=
  (α ∩ β = m) → (m ∥ n) → (n ∥ α) ∨ (n ∥ β)

-- The main theorem statement
theorem correct_propositions (α β : Plane) (m n : Line) :
  (proposition1 α β m n) ∧ (proposition4 α β m n) :=
by
  sorry

end correct_propositions_l108_108979


namespace range_of_inclination_angle_l108_108997

theorem range_of_inclination_angle (theta : ℝ) :
  let k : ℝ := Real.cos theta,
      α : ℝ := Real.arctan k
  in  α ∈ set.Icc 0 (Real.pi / 4) ∪ set.Ico (3 * Real.pi / 4) Real.pi := sorry

end range_of_inclination_angle_l108_108997


namespace trigonometric_identity_l108_108995

variable (α : Real)

theorem trigonometric_identity (h : Real.tan α = Real.sqrt 2) :
  (1/3) * Real.sin α^2 + Real.cos α^2 = 5/9 :=
sorry

end trigonometric_identity_l108_108995


namespace terminating_decimal_expansion_of_17_div_625_l108_108526

theorem terminating_decimal_expansion_of_17_div_625 : 
  ∃ d : ℚ, d = 17 / 625 ∧ d = 0.0272 :=
by
  sorry

end terminating_decimal_expansion_of_17_div_625_l108_108526


namespace vasechkin_result_l108_108332

theorem vasechkin_result (x : ℕ) (h : (x / 2 * 7) - 1001 = 7) : (x / 8) ^ 2 - 1001 = 295 :=
by
  sorry

end vasechkin_result_l108_108332


namespace count_sums_of_fours_and_fives_l108_108639

theorem count_sums_of_fours_and_fives :
  ∃ n, (∀ x y : ℕ, 4 * x + 5 * y = 1800 ↔ (x = 0 ∨ x ≤ 1800) ∧ (y = 0 ∨ y ≤ 1800)) ∧ n = 201 :=
by
  -- definition and theorem statement is complete. The proof is omitted.
  sorry

end count_sums_of_fours_and_fives_l108_108639


namespace ABFCDE_perimeter_l108_108887

-- Define the problem conditions
def square_side_length (perimeter: ℝ) : ℝ := perimeter / 4

-- State the theorem to prove the perimeter of figure ABFCDE
theorem ABFCDE_perimeter (perimeter_ABCD: ℝ) (h_perimeter: perimeter_ABCD = 48) :
  let s := square_side_length perimeter_ABCD in 
  6 * s = 72 :=
by
  sorry

end ABFCDE_perimeter_l108_108887


namespace count_integers_with_conditions_l108_108217

theorem count_integers_with_conditions :
  let numbers := {x : ℕ | 2000 ≤ x ∧ x < 3000 ∧ (digit 1000 x = 2) ∧
                  ((digit 100 x = 4 → digit 10 x ≠ 4 ∧ digit 1 x ≠ 4) ∧
                   (digit 100 x ≠ 4 → (digit 10 x = 3 ∨ digit 1 x = 3)) ∧
                   (digit 100 x = 3 ∨ digit 10 x = 3 ∨ digit 1 x = 3)) } in
  Fintype.card numbers = 306 :=
by sorry

end count_integers_with_conditions_l108_108217


namespace half_angle_second_quadrant_l108_108222

theorem half_angle_second_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end half_angle_second_quadrant_l108_108222


namespace part_a_part_b_l108_108540

/-- Define rational non-integer numbers x and y -/
structure RationalNonInteger (x y : ℚ) :=
  (h1 : x.denom ≠ 1)
  (h2 : y.denom ≠ 1)

/-- Part (a): There exist rational non-integer numbers x and y 
    such that 19x + 8y and 8x + 3y are integers -/
theorem part_a : ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x + 8*y ∈ ℤ) ∧ (8*x + 3*y ∈ ℤ) :=
by
  sorry

/-- Part (b): There do not exist rational non-integer numbers x and y 
    such that 19x^2 + 8y^2 and 8x^2 + 3y^2 are integers -/
theorem part_b : ¬ ∃ (x y : ℚ), RationalNonInteger x y ∧ (19*x^2 + 8*y^2 ∈ ℤ) ∧ (8*x^2 + 3*y^2 ∈ ℤ) :=
by
  sorry

end part_a_part_b_l108_108540


namespace number_of_bookshelves_l108_108508

theorem number_of_bookshelves (books_in_each: ℕ) (total_books: ℕ) (h_books_in_each: books_in_each = 56) (h_total_books: total_books = 504) : total_books / books_in_each = 9 :=
by
  sorry

end number_of_bookshelves_l108_108508


namespace order_abc_l108_108924

noncomputable def a : ℝ := (3 * (2 - Real.log 3)) / Real.exp 2
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.sqrt (Real.exp 1)) / (2 * Real.exp 1)

theorem order_abc : c < a ∧ a < b := by
  sorry

end order_abc_l108_108924


namespace sequence_solution_l108_108968

theorem sequence_solution (a : ℕ → ℝ)
  (h₁ : a 1 = 0)
  (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 4 * (Real.sqrt (a n + 1)) + 4) :
  ∀ n ≥ 1, a n = 4 * n^2 - 4 * n :=
by
  sorry

end sequence_solution_l108_108968


namespace solve_for_x_l108_108150

theorem solve_for_x (x : ℝ) (h : sqrt (x - 5) = 7) : x = 54 :=
by
  sorry

end solve_for_x_l108_108150


namespace concyclic_points_iff_ratios_l108_108504

variables {O A B C D E : Type} [EuclideanPlane O]
variables [FCircle O A B C D E] -- We assume these points lie on a circle O

variables (F G H I J : Type)
variables (AD DB BE EC CA : LineSegment)
variables (FGHJI : Pentagon AD DB BE EC CA F G H I J)

theorem concyclic_points_iff_ratios :
  (concyclic F G H I J) ↔
  (ratio FG CE = ratio GH DA ∧
   ratio GH DA = ratio HI BE ∧
   ratio HI BE = ratio IJ AC ∧
   ratio IJ AC = ratio JF BD) :=
sorry

end concyclic_points_iff_ratios_l108_108504


namespace construct_quadrilateral_l108_108885

variables (α a b c d : ℝ)

-- α represents the sum of angles B and D
-- a represents the length of AB
-- b represents the length of BC
-- c represents the length of CD
-- d represents the length of DA

theorem construct_quadrilateral (α a b c d : ℝ) : 
  ∃ A B C D : ℝ × ℝ, 
    dist A B = a ∧ 
    dist B C = b ∧ 
    dist C D = c ∧ 
    dist D A = d ∧ 
    ∃ β γ δ, β + δ = α := 
sorry

end construct_quadrilateral_l108_108885


namespace initial_gain_percentage_l108_108837

theorem initial_gain_percentage 
  (S : ℝ) (C : ℝ) (N : ℝ := 29.99999625000047) 
  (H1 : S = 60 / 20) 
  (H2 : 0.8 * N * C = 60) 
  (C_value : C = 60 / (0.8 * N))
  : (S - C) / C * 100 = 20 :=
by
  let S := 60 / 20
  let N := 29.99999625000047
  have H1 : S = 3, by sorry
  have C_value : 0.8 * N * C = 60, by sorry
  let C := 60 / (0.8 * N)
  have H2 : C = 2.5, by sorry
  show (S - C) / C * 100 = 20, by sorry

end initial_gain_percentage_l108_108837


namespace seven_gon_multicolored_triangles_exists_l108_108544

def is_adjacency_proper_colored (vertices : ℕ → char) : Prop :=
  ∀ n, vertices n ≠ vertices (n + 1)

def is_multicolored_triangle (vertices : list (ℕ → char)) : Prop :=
  (vertices 0 ≠ vertices 1) ∧ (vertices 0 ≠ vertices 2) ∧ (vertices 1 ≠ vertices 2)

theorem seven_gon_multicolored_triangles_exists :
  ∀ (vertices : ℕ → char),
  (∀ i, vertices i = 'r' ∨ vertices i = 'g' ∨ vertices i = 'b') →
  is_adjacency_proper_colored vertices →
  ∃ V1 V2 V3 V4 V5 V6 : ℕ, 
    is_multicolored_triangle [vertices V1, vertices V2, vertices V3] ∧
    is_multicolored_triangle [vertices V4, vertices V5, vertices V6] ∧
    V1 ≠ V2 ∧ V1 ≠ V3 ∧ V2 ≠ V3 ∧ V4 ≠ V5 ∧ V4 ≠ V6 ∧ V5 ≠ V6 :=
sorry

end seven_gon_multicolored_triangles_exists_l108_108544


namespace intersection_volume_l108_108784

noncomputable def volume_of_intersection (k : ℝ) : ℝ :=
  ∫ x in -k..k, 4 * (k^2 - x^2)

theorem intersection_volume (k : ℝ) : volume_of_intersection k = 16 * k^3 / 3 :=
  by
  sorry

end intersection_volume_l108_108784


namespace smallest_number_of_three_l108_108810

theorem smallest_number_of_three (x : ℕ) (h1 : x = 18)
  (h2 : ∀ y z : ℕ, y = 4 * x ∧ z = 2 * y)
  (h3 : (x + 4 * x + 8 * x) / 3 = 78)
  : x = 18 := by
  sorry

end smallest_number_of_three_l108_108810


namespace multiplication_more_than_subtraction_l108_108797

def x : ℕ := 22

def multiplication_result : ℕ := 3 * x
def subtraction_result : ℕ := 62 - x
def difference : ℕ := multiplication_result - subtraction_result

theorem multiplication_more_than_subtraction : difference = 26 :=
by
  sorry

end multiplication_more_than_subtraction_l108_108797


namespace binomial_sum_l108_108512

theorem binomial_sum :
  1 - (nat.choose 10 1) * (3^1) + (nat.choose 10 2) * (3^2) - (nat.choose 10 3) * (3^3) +
  ( /- ... -/ (nat.choose 10 10) * (3^10)) = 1024 :=
by
  sorry

end binomial_sum_l108_108512


namespace arithmetic_sequence_middle_term_l108_108270

theorem arithmetic_sequence_middle_term 
  (x y z : ℕ) 
  (h1 : ∀ i, (∀ j, 23 + (i - 0) * d = x)
  (h2 : ∀ i, (∀ j, y = 23 + (j - 1) * (23 + d)) 
  (h3 : ∀ i, (∀ j, z = 23 + (j - 2 * d)) 
  (h4 : ∀ i, 47 = 23 + (5 - 1) * d )
   : y = (23 + 47) / 2 :=
by 
  sorry

end arithmetic_sequence_middle_term_l108_108270


namespace find_lambda_l108_108629

variable {A B C G D : Type}
variables [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup G] [AddGroup D]
variables (vga vbg vcg vag vgd : A → B)
variables (λ : ℝ)

-- Define the conditions
def midpoint_condition (D : Type) (B C : A) : Prop := (D: A) = (B + C) / 2
def vector_zero_condition (vga vbg vcg : A → B) (G : A) : Prop := vga G + vbg G + vcg G = 0

-- Define the main equation to prove
def λ_value (vag vgd : A → B) (λ : ℝ) : Prop := vag = λ • vgd

-- The main theorem
theorem find_lambda 
  (hp1 : vector_zero_condition A A A A A vga vbg vcg)
  (hp2 : λ_value vag vgd λ)
  (hp3 : midpoint_condition D B C) :
  -- Conclude λ = -2
  (λ = -2) :=
sorry

end find_lambda_l108_108629


namespace cost_price_each_watch_l108_108024

open Real

theorem cost_price_each_watch
  (C : ℝ)
  (h1 : let lossPerc := 0.075 in
        let sp_each := C * (1 - lossPerc) in
        let gainPerc := 0.053 in
        let sp_more := sp_each + 265 in
        let sp_total := 3 * sp_more in
        sp_total = 3 * C * (1 + gainPerc)) :
  C ≈ 2070.31 := by
  sorry

end cost_price_each_watch_l108_108024


namespace xyz_value_l108_108181

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (xy + xz + yz) = 40) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) 
  : x * y * z = 10 :=
sorry

end xyz_value_l108_108181


namespace perimeter_of_ABFCDE_l108_108129

open Real

/-- Given a square ABCD with a perimeter of 64 inches, and an equilateral triangle BFC cut out and translated downward to form the figure ABFCDE, the perimeter of the figure ABFCDE is 80 inches. -/
theorem perimeter_of_ABFCDE :
  ∀ (s : ℝ), (4 * s = 64) →
  let t := s in
  let p_ABFCDE := 4 * s + 2 * t - t in
  p_ABFCDE = 80 :=
begin
  intros s h,
  let t := s,
  let p_ABFCDE := 4 * s + 2 * t - t,
  show p_ABFCDE = 80,
  sorry
end

end perimeter_of_ABFCDE_l108_108129


namespace value_of_m_l108_108234

theorem value_of_m (m : ℝ) (h : m ≠ 0)
  (h_roots : ∀ x, m * x^2 + 8 * m * x + 60 = 0 ↔ x = -5 ∨ x = -3) :
  m = 4 :=
sorry

end value_of_m_l108_108234


namespace jake_bitcoins_proportion_l108_108289

-- Define the conditions as hypotheses and then state the proposition to prove.

theorem jake_bitcoins_proportion (p : ℝ) :
  let initial_bitcoins := 80
  let after_first_donation := initial_bitcoins - 20
  let to_brother := p * after_first_donation
  let remaining_after_brother := after_first_donation - to_brother
  let after_tripling := 3 * remaining_after_brother
  let after_second_donation := after_tripling - 10
  after_second_donation = initial_bitcoins := 
  p = 1 / 2 := 
sorry

end jake_bitcoins_proportion_l108_108289


namespace ratio_of_only_B_to_both_A_and_B_l108_108482

theorem ratio_of_only_B_to_both_A_and_B 
  (Total_households : ℕ)
  (Neither_brand : ℕ)
  (Only_A : ℕ)
  (Both_A_and_B : ℕ)
  (Total_households_eq : Total_households = 180)
  (Neither_brand_eq : Neither_brand = 80)
  (Only_A_eq : Only_A = 60)
  (Both_A_and_B_eq : Both_A_and_B = 10) :
  (Total_households = Neither_brand + Only_A + (Total_households - Neither_brand - Only_A - Both_A_and_B) + Both_A_and_B) →
  (Total_households - Neither_brand - Only_A - Both_A_and_B) / Both_A_and_B = 3 :=
by
  intro H
  sorry

end ratio_of_only_B_to_both_A_and_B_l108_108482


namespace solve_for_x_l108_108987

theorem solve_for_x (x : ℕ) (h : 2^12 = 64^x) : x = 2 :=
by {
  sorry
}

end solve_for_x_l108_108987


namespace talent_school_l108_108250

theorem talent_school :
  ∀ (total_students cannot_sing cannot_dance cannot_act : ℕ),
    total_students = 150 →
    cannot_sing = 90 →
    cannot_dance = 100 →
    cannot_act = 60 →
    ∃ (students_with_two_talents : ℕ), students_with_two_talents = 50 :=
by
  assume (total_students cannot_sing cannot_dance cannot_act : ℕ),
  assume h1 : total_students = 150,
  assume h2 : cannot_sing = 90,
  assume h3 : cannot_dance = 100,
  assume h4 : cannot_act = 60,
  let students_sing := total_students - cannot_sing,
  let students_dance := total_students - cannot_dance,
  let students_act := total_students - cannot_act,
  let sum_sing_dance_act := students_sing + students_dance + students_act,
  let excess_students := sum_sing_dance_act - total_students,
  let students_with_two_talents := excess_students,
  have h5 : students_with_two_talents = 50 := by
    calc
      students_with_two_talents = sum_sing_dance_act - total_students : rfl
      ... = (students_sing + students_dance + students_act) - total_students : rfl
      ... = (60 + 50 + 90) - 150 : by rw [h1, h2, h3, h4]
      ... = 200 - 150 : rfl
      ... = 50 : rfl,
  existsi students_with_two_talents,
  exact h5,
  sorry

end talent_school_l108_108250


namespace probability_same_number_selected_l108_108863

theorem probability_same_number_selected :
  let Billy_numbers := {n : ℕ | 1 ≤ n ∧ n < 300 ∧ n % 20 = 0},
      Bobbi_numbers := {n : ℕ | 1 ≤ n ∧ n < 300 ∧ n % 30 = 0},
      common_numbers := Billy_numbers ∩ Bobbi_numbers
  in (↑(common_numbers.card) : ℚ) / (↑(Billy_numbers.card * Bobbi_numbers.card) : ℚ) = 1 / 30 :=
by
  sorry

end probability_same_number_selected_l108_108863


namespace circle_radius_l108_108016

theorem circle_radius
  (n : ℕ) (h_n : n = 2003)
  (angle_MON : ℝ) (h_angle : angle_MON = 60)
  (r1 : ℝ) (h_r1 : r1 = 1)
  (ratio : ℝ) (h_ratio : ratio = 3) :
  ∃ r2003 : ℝ, r2003 = 3^(2002) :=
by
  use 3^(2002)
  exact eq.refl (3^(2002))

end circle_radius_l108_108016


namespace radians_to_degrees_l108_108519

theorem radians_to_degrees : (2 / 3) * real.pi * (180 / real.pi) = -120 := 
by
  sorry

end radians_to_degrees_l108_108519


namespace function_maximum_at_2_l108_108565

noncomputable def function_f (c x : ℝ) : ℝ := x * (x - c) ^ 2

theorem function_maximum_at_2 (c : ℝ) : (∃ f', ∀ x,  f' x = derivative (function_f c x)) 
    → (2 * (2 - c) ^ 2 = 0) 
    → f'(2) = 0
    → (∀ x, (derivative (function_f c x) > 0 → x < 2) ∧ (derivative (function_f c x) < 0 → x > 2)) 
    → c = 6 := by
  sorry

end function_maximum_at_2_l108_108565


namespace calculate_expression_value_l108_108110

theorem calculate_expression_value (x y : ℚ) (hx : x = 4 / 7) (hy : y = 5 / 8) :
  (7 * x + 5 * y) / (70 * x * y) = 57 / 400 := by
  sorry

end calculate_expression_value_l108_108110


namespace ellipse_equation_area_range_l108_108607

open Real

-- Conditions
def vertex_A : (ℝ × ℝ) := (-2, 0)

def ellipse (a b : ℝ) : set (ℝ × ℝ) := 
  {p | ∃ x y, p = (x, y) ∧ x^2 / a^2 + y^2 / b^2 = 1}

def vertical_line : set (ℝ × ℝ) := 
  {p | p.1 = 1}

def intersections : set (ℝ × ℝ) :=
  {p | p ∈ ellipse 2 (sqrt 3) ∧ p ∈ vertical_line}

def points_PQ : (ℝ × ℝ) × (ℝ × ℝ) := 
  ((1, 3/2), (1, -3/2))

def distance_PQ : ℝ :=
  3

-- Proving Parts
-- Part 1: Equation of the ellipse
theorem ellipse_equation : (x y : ℝ) -> x^2 / 4 + y^2 / 3 = 1 ↔ (x, y) ∈ ellipse 2 (sqrt 3) :=
by sorry

-- Part 2: Range of area
theorem area_range : (0 < area (triangle vertex_A points_PQ.1 points_PQ.2) ≤ 9/2) :=
by sorry

end ellipse_equation_area_range_l108_108607


namespace solution_set_m_zero_range_of_m_for_solution_set_ℝ_l108_108206

-- Part 1: Solution set for m = 0
theorem solution_set_m_zero (x : ℝ) : 
  (0 - 1) * x^2 + (0 - 1) * x + 2 > 0 ↔ x ∈ Ioo (-2 : ℝ) 1 :=
sorry

-- Part 2: Range of m for solution set to be ℝ
theorem range_of_m_for_solution_set_ℝ (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0) ↔ 1 < m ∧ m < 9 :=
sorry

end solution_set_m_zero_range_of_m_for_solution_set_ℝ_l108_108206


namespace eccentricity_hyperbola_l108_108749

variables (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0)
variables (H : c = Real.sqrt (a^2 + b^2))
variables (L1 : ∀ x y : ℝ, x = c → (x^2/a^2 - y^2/b^2 = 1))
variables (L2 : ∀ (B C : ℝ × ℝ), (B.1 = c ∧ C.1 = c) ∧ (B.2 = -C.2) ∧ (B.2 = b^2/a))

theorem eccentricity_hyperbola : ∃ e, e = 2 :=
sorry

end eccentricity_hyperbola_l108_108749


namespace perfect_squares_diff_count_perfect_squares_less_than_20000_l108_108986

open Nat

theorem perfect_squares_diff (n : ℕ) (h : n < 20000 ∧ ∃ b : ℕ, n = (b + 2)^2 - b^2) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 70 ∧ n = 4 * k^2 :=
begin
  sorry
end

theorem count_perfect_squares_less_than_20000 :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 70 :=
sorry

end perfect_squares_diff_count_perfect_squares_less_than_20000_l108_108986


namespace minimum_additional_squares_to_shade_for_symmetry_l108_108243

def is_symmetric (grid : array (fin 6) (array (fin 6) bool)) : bool :=
  let horizontal := ∀ i, grid[i] = grid[5 - i]
  let vertical := ∀ i j, grid[i][j] = grid[i][5 - j]
  let diagonal1 := ∀ i j, grid[i][j] = grid[j][i]
  let diagonal2 := ∀ i j, grid[i][j] = grid[5 - j][5 - i]
  horizontal && vertical && diagonal1 && diagonal2

def count_additional_squares (initial_squares : list (fin 6 × fin 6)) : ℕ :=
  let initial_shade: list (fin 6 × fin 6) := [(2, 5), (3, 3), (4, 2), (6, 1)]
  let needed_squares: list (fin 6 × fin 6) = initial_squares -- some calculation here
  needed_squares.length -- should calculate the difference consisting on total squares - initial_shaded(pe)

theorem minimum_additional_squares_to_shade_for_symmetry 
  (initial_squares : list (fin 6 × fin 6)) :
  count_additional_squares initial_squares = 9 :=
sorry

end minimum_additional_squares_to_shade_for_symmetry_l108_108243


namespace meters_run_by_A_l108_108817

variable (t_1 t_2 : ℝ) (v_α v_β v_γ d_α d_β d_γ : ℝ)

def track_circumference := 360 -- circumference of the track in meters
def meeting_distance_half := 180 -- half of the track circumference in meters

-- B's speed is 4 times A's speed
def speed_relation : Prop := v_β = 4 * v_α

-- Distances covered by A, B, and C
def distance_α : Prop := d_α = v_α * t_1
def distance_β : Prop := d_β = v_β * t_1
def distance_γ : Prop := d_γ = v_γ * (t_1 - t_2)

-- Meeting conditions
def first_meeting : Prop := (d_α + d_β) = track_circumference
def gamma_position_first_meeting : Prop := (d_α + meeting_distance_half) - d_γ = d_α - meeting_distance_half
def alpha_gamma_meeting : Prop := (d_α + d_γ) = meeting_distance_half

-- B and C start after A has run 90 meters
theorem meters_run_by_A : ∃ t_1 t_2 v_α v_β v_γ d_α d_β d_γ, 
  speed_relation ∧
  distance_α ∧
  distance_β ∧
  distance_γ ∧
  first_meeting ∧
  gamma_position_first_meeting ∧
  alpha_gamma_meeting ∧
  d_α = 90 :=
begin
  sorry
end

end meters_run_by_A_l108_108817


namespace prove_f_log4_9_eq_neg_one_third_l108_108943

-- Definitions of the conditions
def f (x : ℝ) : ℝ :=
if x < 0 then 2^x else - 1 / (2^x)

-- The theorem we need to prove:  f(log_4 9) = -1 / 3
theorem prove_f_log4_9_eq_neg_one_third : 
  f (log 9 / log 4) = -1 / 3 :=
by
  sorry

end prove_f_log4_9_eq_neg_one_third_l108_108943


namespace unique_triple_property_l108_108753

theorem unique_triple_property (a b c : ℕ) (h1 : a ∣ b * c + 1) (h2 : b ∣ a * c + 1) (h3 : c ∣ a * b + 1) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a = 2 ∧ b = 3 ∧ c = 7) :=
by
  sorry

end unique_triple_property_l108_108753


namespace sum_of_roots_tan_l108_108911

theorem sum_of_roots_tan (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * real.pi) :
  ∑ root in {x | tan x * tan x - 5 * tan x + 6 = 0}, x = (3 * real.pi) / 2 := 
sorry

end sum_of_roots_tan_l108_108911


namespace arithmetic_sequence_middle_term_l108_108269

theorem arithmetic_sequence_middle_term 
  (x y z : ℕ) 
  (h1 : ∀ i, (∀ j, 23 + (i - 0) * d = x)
  (h2 : ∀ i, (∀ j, y = 23 + (j - 1) * (23 + d)) 
  (h3 : ∀ i, (∀ j, z = 23 + (j - 2 * d)) 
  (h4 : ∀ i, 47 = 23 + (5 - 1) * d )
   : y = (23 + 47) / 2 :=
by 
  sorry

end arithmetic_sequence_middle_term_l108_108269


namespace D_is_regular_dodecahedron_l108_108458

structure PPolyhedron where
  vertices : Nat
  faces : Nat
  faceTypes : String

structure Cube where 
  vertices : Nat

structure DPolyhedron (P : PPolyhedron) (K : Cube) where
  faces : Nat
  faceShape : String
  verticesMeetings : Nat

variables (P : PPolyhedron) (K : Cube) (D : DPolyhedron P K)

axiom P_properties (P : PPolyhedron) : P.vertices = 6 ∧ P.faces = 5 ∧ P.faceTypes = "1 square, 4 isosceles triangles"

axiom K_properties (K : Cube) : K.vertices = 8

axiom D_properties (D : DPolyhedron P K) : 
  D.faces = 12 ∧ 
  D.faceShape = "regular pentagons" ∧ 
  D.verticesMeetings = 3

theorem D_is_regular_dodecahedron (P : PPolyhedron) (K : Cube) (D : DPolyhedron P K) 
  (hP : P_properties P) 
  (hK : K_properties K) 
  (hD : D_properties D) : 
  D.faceShape = "regular pentagons" ∧ D.faces = 12 ∧ D.verticesMeetings = 3  → ∃ D. D.faces = 12 ∧ D.faceShape = "regular pentagons" ∧ D.verticesMeetings = 3 :=
by
  intros hD'
  exact ⟨D, hD'.1, hD'.2⟩
  sorry

end D_is_regular_dodecahedron_l108_108458


namespace linear_correlation_l108_108617

variable (x y : ℝ)

theorem linear_correlation (h : y = 0.5 + 2 * x) : 
  ∃ k > 0, y = k * x + 0.5 := 
by {
  use 2,
  split,
  { linarith, },
  { exact h, },
}

end linear_correlation_l108_108617


namespace find_lambda_l108_108980

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3 / 2 * x), Real.sin (3 / 2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))
noncomputable def f (x λ : ℝ) : ℝ := (Real.cos x) - 2 * λ * (2 * Real.cos x)

theorem find_lambda (λ : ℝ) (h_lambda : ∀ x, 0 < x ∧ x < Real.pi / 2 → f x λ ≥ -3 / 2) : λ = 1 / 2 :=
by
  sorry

end find_lambda_l108_108980


namespace measure_angle_CED_is_45_l108_108298

noncomputable def measure_angle_CED : ℝ := 45

variables {A B C D E : Type} 
  [convex_quadrilateral ABCD]
  [equilateral (triangle A B D)]
  [isosceles (triangle B C D)]
  [angle_C_90 ABCD]
  [midpoint E A D]

theorem measure_angle_CED_is_45 
  (h1 : convex_quadrilateral ABCD)
  (h2 : equilateral (triangle A B D))
  (h3 : isosceles (triangle B C D))
  (h4 : angle_C_90 ABCD)
  (h5 : midpoint E A D) 
  : measure_angle_CED = 45 := 
sorry

end measure_angle_CED_is_45_l108_108298


namespace arithmetic_sequence_middle_term_l108_108272

variable (x y z : ℝ)

theorem arithmetic_sequence_middle_term :
  (23, x, y, z, 47) → y = 35 :=
by
  intro h
  have h1 : y = (23 + 47) / 2 := sorry
  have h2 : y = 35 := sorry
  exact h2

end arithmetic_sequence_middle_term_l108_108272


namespace find_n_l108_108769

theorem find_n (n : ℕ) (h_sum : ∑ i in finset.range (n//2 + 1), 2 * i = 81 * 82) (h_odd : n % 2 = 1) : n = 163 :=
sorry

end find_n_l108_108769


namespace smallest_b_value_minimizes_l108_108716

noncomputable def smallest_b_value (a b : ℝ) (c : ℝ := 2) : ℝ :=
  if (1 < a) ∧ (a < b) ∧ (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) ∧ (¬ (1/b + 1/a > c ∧ 1/a + c > 1/b ∧ c + 1/b > 1/a)) then b else 0

theorem smallest_b_value_minimizes (a b : ℝ) (c : ℝ := 2) :
  (1 < a) ∧ (a < b) ∧ (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) ∧ (¬ (1/b + 1/a > c ∧ 1/a + c > 1/b ∧ c + 1/b > 1/a)) →
  b = 2 :=
by sorry

end smallest_b_value_minimizes_l108_108716


namespace max_total_score_l108_108815

-- Definitions for the conditions

def instructor_scores : List (List ℕ) :=
  [[1, 1, 1, 0, 0, 0, 0, 0, 0],
   [1, 1, 1, 0, 0, 0, 0, 0, 0],
   [1, 1, 1, 0, 0, 0, 0, 0, 0]]

/--
Given that each of the three instructors graded exactly 3 problems with a score of 1,
and the rest with a score of 0, prove that the maximum possible total score for the submission is 4.
-/
theorem max_total_score : 
  (∑ i, (List.sum (List.map (λ n, (n / 3 : ℚ).round) (List.transpose instructor_scores).nth i)) = 4 :=
by
  sorry

end max_total_score_l108_108815


namespace eight_digit_strictly_increasing_remainder_l108_108306

def strictly_increasing_eight_digit_numbers_count_mod (low high : ℕ) (digits : ℕ) : ℕ :=
  let count := binomial (high - low + digits) (digits - 1)
  count % 1000

theorem eight_digit_strictly_increasing_remainder :
  strictly_increasing_eight_digit_numbers_count_mod 3 8 8 = 21 := 
sorry

end eight_digit_strictly_increasing_remainder_l108_108306


namespace ellipse_and_area_l108_108604

section EllipseProblem

-- Definitions according to given conditions
def is_vertex (p : ℝ × ℝ) := p.1 = -2 ∧ p.2 = 0
def ellipse_eq (a b : ℝ) (a_pos : a > b) (b_pos : b > 0) (p : ℝ × ℝ) :=
  (p.1)^2 / a^2 + (p.2)^2 / b^2 = 1

-- Correct answers as definitions in Lean
def ellipse_equation := ∀ (p : ℝ × ℝ), ellipse_eq 2 (√3) dec_trivial dec_trivial p
def area_triangle_range := ∀ A P Q : ℝ × ℝ, 
  is_vertex A ∧ 
  (A ≠ P ∧ A ≠ Q) ∧ 
  (P.1 = 1 ∧ Q.1 = 1 ∧ P ≠ Q) ∧ 
  |P.2 - Q.2| = 3 → 
  0 < sorry ∧ sorry ≤ 9 / 2

-- Theorem statement
theorem ellipse_and_area :
  ellipse_equation ∧ area_triangle_range :=
by { sorry }

end EllipseProblem

end ellipse_and_area_l108_108604


namespace arithmetic_sequence_middle_term_l108_108267

theorem arithmetic_sequence_middle_term (x y z : ℝ) (h : list.nth_le [23, x, y, z, 47] 2 sorry = y) :
  y = 35 :=
sorry

end arithmetic_sequence_middle_term_l108_108267


namespace average_of_new_set_l108_108367

theorem average_of_new_set (S : ℕ) (numbers : list ℕ) (h₁ : numbers.length = 12) (h₂ : S = numbers.sum) 
  (h₃ : S / 12 = 90) :
  let new_sum := S + 80 + 90
      new_count := 14 in 
      (new_sum / new_count : ℝ) ≈ 89.2857142857 :=
by 
  have h₄ : S = 1080,
  { rw ←h₃,
    have h₅ : S / 12 = 1080 / 12 := by linarith,
    exact eq_of_div_eq_div_of_eq_iff_eq h₅, },
  let new_sum := S + 80 + 90,
  let new_count := 14,
  have h_new_sum : new_sum = 1250, by linarith [h₄],
  have h_avg : (new_sum : ℝ) / new_count ≈ 1250 / 14,
  { rw h_new_sum,
    apply rfl, },
  refine h_avg

end average_of_new_set_l108_108367


namespace seq_formula_exists_lambda_arith_l108_108171

-- Define the sequence sum Sn
def S (n : ℕ) : ℕ := n^2 + 2 * n

-- Proof problem for the general formula for the sequence {a_n}
theorem seq_formula (n : ℕ) : 
  (∀ a : ℕ → ℕ, (∀ n, a n = (S n) - (S (n - 1))) ∧ a 1 = S 1) → (SeqFormula: ∀ n, a n = 2 * n + 1) :=
by 
  sorry

-- Define the sequence a_n using the proved formula
def a (n : ℕ) : ℕ := 2 * n + 1

-- Define the sequence c_n
def c : ℕ → ℕ 
| 1 := 3
| (n + 1) := a (c n) + 2^n

-- Proof problem for the existence of λ
theorem exists_lambda_arith (λ : ℝ) :
  (∀ n, c n = 3 → c (n + 1) = a (c n) + 2^n) → 
  (∃ λ : ℝ, ∀ n, (2 * (c n + λ) / 2^n) = ((c n + λ) / 2 ^ (n - 1) + (c (n + 1) + λ) / 2 ^ (n + 1))) →
  (λ = 1) :=
by 
  sorry

end seq_formula_exists_lambda_arith_l108_108171


namespace find_P_l108_108207

theorem find_P
  (m : ℕ) (k : ℕ) (P : ℕ)
  (h1 : m = 7)
  (h2 : k = 12)
  (equation : 7! * 14! = 18 * P * 11!) :
  P = 54080 :=
by
  subst h1
  subst h2
  sorry

end find_P_l108_108207


namespace range_of_a_l108_108178

theorem range_of_a (a m : ℝ) (hp : 3 * a < m ∧ m < 4 * a) 
  (hq : 1 < m ∧ m < 3 / 2) :
  1 / 3 ≤ a ∧ a ≤ 3 / 8 :=
by
  sorry

end range_of_a_l108_108178


namespace range_of_m_l108_108961

noncomputable def f (a : ℝ) (x : ℝ) := a * real.log x - a * x - 3

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := (a * (x - 1)) / x

noncomputable def g (a : ℝ) (m : ℝ) (x : ℝ) := x^3 + x^2 * (f' a x + m/2)

theorem range_of_m (a m : ℝ) (h : f' (-2) 2 = 1) :
  (∀ t, 1 ≤ t ∧ t ≤ 2 → (∀ x, t < x ∧ x < 3 → 3 * x^2 + (m + 4) * x - 2 < 0 ∨ 3 * x^2 + (m + 4) * x - 2 > 0)) →
  -37 / 3 < m ∧ m < -9 :=
sorry

end range_of_m_l108_108961


namespace correct_answer_is_B_l108_108094

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f(x) > f(y)

def option_A (x : ℝ) : ℝ := abs x
def option_B (x : ℝ) : ℝ := x ^ (-2)
def option_C (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)
def option_D (x : ℝ) : ℝ := -x + 1

theorem correct_answer_is_B :
  (is_even option_B ∧ is_monotonically_decreasing option_B {x : ℝ | 0 < x}) ∧
  ¬ (is_even option_A ∧ is_monotonically_decreasing option_A {x : ℝ | 0 < x}) ∧
  ¬ (is_even option_C ∧ is_monotonically_decreasing option_C {x : ℝ | 0 < x}) ∧
  ¬ (is_even option_D ∧ is_monotonically_decreasing option_D {x : ℝ | 0 < x}) :=
begin
  sorry
end

end correct_answer_is_B_l108_108094


namespace total_clowns_l108_108744

def num_clown_mobiles : Nat := 5
def clowns_per_mobile : Nat := 28

theorem total_clowns : num_clown_mobiles * clowns_per_mobile = 140 := by
  sorry

end total_clowns_l108_108744


namespace dangerous_animals_remaining_in_swamp_l108_108278

-- Define the initial counts of each dangerous animals
def crocodiles_initial := 42
def alligators_initial := 35
def vipers_initial := 10
def water_moccasins_initial := 28
def cottonmouth_snakes_initial := 15
def piranha_fish_initial := 120

-- Define the counts of migrating animals
def crocodiles_migrating := 9
def alligators_migrating := 7
def vipers_migrating := 3

-- Define the total initial dangerous animals
def total_initial : Nat :=
  crocodiles_initial + alligators_initial + vipers_initial + water_moccasins_initial + cottonmouth_snakes_initial + piranha_fish_initial

-- Define the total migrating dangerous animals
def total_migrating : Nat :=
  crocodiles_migrating + alligators_migrating + vipers_migrating

-- Define the total remaining dangerous animals
def total_remaining : Nat :=
  total_initial - total_migrating

theorem dangerous_animals_remaining_in_swamp :
  total_remaining = 231 :=
by
  -- simply using the calculation we know
  sorry

end dangerous_animals_remaining_in_swamp_l108_108278


namespace book_set_cost_l108_108295

theorem book_set_cost (charge_per_sqft : ℝ) (lawn_length lawn_width : ℝ) (num_lawns : ℝ) (additional_area : ℝ) (total_cost : ℝ) :
  charge_per_sqft = 0.10 ∧ lawn_length = 20 ∧ lawn_width = 15 ∧ num_lawns = 3 ∧ additional_area = 600 ∧ total_cost = 150 →
  (num_lawns * (lawn_length * lawn_width) * charge_per_sqft + additional_area * charge_per_sqft = total_cost) :=
by
  sorry

end book_set_cost_l108_108295


namespace division_addition_l108_108069

theorem division_addition (n : ℕ) (h : 32 - 16 = n * 4) : n / 4 + 16 = 17 :=
by 
  sorry

end division_addition_l108_108069


namespace section_construction_correct_l108_108595

-- Define the points
variables {P Q R A A1 B B1 C C1 D D1 P1 P2 : Point}

-- Define the edges
variables {AA1 BC B1C1 AD A1D1 : Line}

-- Assume the point conditions
variable (hP_edge : P ∈ AA1) 
variable (hQ_edge : Q ∈ BC) 
variable (hR_edge : R ∈ B1C1)
variable (hP1_edge : P1 ∈ AD)
variable (hP2_edge : P2 ∈ A1D1)

-- Define the construction of the section
def construct_section := intersect_plane_with_cuboid P Q R AA1 BC B1C1 AD A1D1

-- The statement to prove: the intersection of the plane PQR with the cuboid forms the correct section
theorem section_construction_correct :
  ∃ P1 P2 : Point, 
    (P1 ∈ AD ∧ P2 ∈ A1D1) ∧ 
    connect_points P1 Q ∧ 
    connect_points P2 R ∧ 
    connect_points P1 R ∧ 
    connect_points P2 Q := 
sorry

end section_construction_correct_l108_108595


namespace symmetry_axis_of_sine_function_l108_108892

theorem symmetry_axis_of_sine_function (x : ℝ) :
  (∃ k : ℤ, 2 * x + π / 4 = k * π + π / 2) ↔ x = π / 8 :=
by sorry

end symmetry_axis_of_sine_function_l108_108892


namespace echo_earnings_correct_l108_108572

/-- 
  Define the constants for the number of students and days worked by each school
  and the total payment made.
--/
def delta_students := 10
def delta_days := 4

def echo_students := 6
def echo_days := 6

def foxtrot_students := 8
def foxtrot_days := 3

def golf_students := 3
def golf_days := 10

def total_payment := 1233

/-- Calculate the total number of student-days worked --/
def total_student_days := delta_students * delta_days
                     + echo_students * echo_days
                     + foxtrot_students * foxtrot_days
                     + golf_students * golf_days

/-- Calculate the daily wage per student --/
def daily_wage := total_payment / total_student_days

/-- Calculate the earnings of students from Echo school --/
def echo_earnings := daily_wage * (echo_students * echo_days)

/-- Prove that the earnings of Echo school's students is $341.45 --/
theorem echo_earnings_correct : echo_earnings = 341.45 := by
  sorry

end echo_earnings_correct_l108_108572


namespace tunnel_excavation_principle_l108_108764

theorem tunnel_excavation_principle :
  ∀ (P Q : Type) [metric_space P] [metric_space Q] (f : P → Q),
  (∀ x y : P, shortest_path f x y) →
  (∀ (x y : P), f x ≠ f y → dist (f x) (f y) = dist x y) →
  (∀ (x y : P), f x = f y ↔ x = y) →
  (∃ t : ℝ, ∀ x y : P, shortest_path f x y ↔ shortest_path t x y) →
  (∃ t : ℝ, ∀ x y : P, dist (f x) (f y) = t * dist x y) :=
by
  intros P Q hP hQ f h1 h2 h3 h4;
  sorry

end tunnel_excavation_principle_l108_108764


namespace find_bus_speed_l108_108841

-- Definitions and conditions
def bus_speed (x : ℝ) := x
def car_speed (x : ℝ) := 1.5 * x
def distance_to_museum : ℝ := 20
def time_delay : ℝ := 1 / 6

-- Proof problem statement
theorem find_bus_speed (x : ℝ) : 
  (20 / bus_speed x) - (20 / car_speed x) = time_delay := sorry

end find_bus_speed_l108_108841


namespace max_cookies_eaten_by_Andy_l108_108779
open Nat

def cookies_problem : Prop :=
  ∃ (a : ℕ), a ≤ 12 ∧ (∀ b, b ≤ 12 → b ≥ a → b = a)

theorem max_cookies_eaten_by_Andy : cookies_problem :=
by
  existsi 12
  split
  · exact le_refl 12
  · intros b hb hba
    exact hba

end max_cookies_eaten_by_Andy_l108_108779


namespace problem1_problem2_l108_108661

-- Define the polar and parametric equations
def line_parametric (t α : ℝ) : ℝ × ℝ :=
  (t * Math.cos α, 1 + t * Math.sin α)

def curve_polar (ρ θ : ℝ) : Prop :=
  ρ * Math.cos θ * Math.cos θ = 4 * Math.sin θ

def curve_cartesian (x y : ℝ) : Prop :=
  x^2 = 4 * y

-- Define the problems
theorem problem1 (x y : ℝ) (h : curve_cartesian x y) : -1 ≤ x + y :=
sorry

theorem problem2 (t1 t2 α : ℝ) (h1 : curve_cartesian (t1 * Math.cos α) (1 + t1 * Math.sin α))
  (h2 : curve_cartesian (t2 * Math.cos α) (1 + t2 * Math.sin α)) :
  |t1 - t2| = 4 :=
sorry

end problem1_problem2_l108_108661


namespace pascal_triangle_row_10_sum_l108_108869

theorem pascal_triangle_row_10_sum : (∑ (k : Fin 11), nat.choose 10 k) = 1024 := by
  sorry

end pascal_triangle_row_10_sum_l108_108869


namespace jackson_points_l108_108244

theorem jackson_points (team_total_points : ℕ)
                       (num_other_players : ℕ)
                       (average_points_other_players : ℕ)
                       (points_other_players: ℕ)
                       (points_jackson: ℕ)
                       (h_team_total_points : team_total_points = 65)
                       (h_num_other_players : num_other_players = 5)
                       (h_average_points_other_players : average_points_other_players = 6)
                       (h_points_other_players : points_other_players = num_other_players * average_points_other_players)
                       (h_points_total: points_jackson + points_other_players = team_total_points) :
  points_jackson = 35 :=
by
  -- proof will be done here
  sorry

end jackson_points_l108_108244


namespace quadratic_function_properties_l108_108906

noncomputable def quadratic_function (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  (quadratic_function 1 = 0) ∧ 
  (quadratic_function 5 = 0) ∧ 
  (quadratic_function 3 = 10) :=
by
  split
  · -- Prove f(1) = 0
    sorry
  split
  · -- Prove f(5) = 0
    sorry
  · -- Prove f(3) = 10
    sorry

end quadratic_function_properties_l108_108906


namespace minimize_circle_areas_l108_108505

noncomputable
def F_coordinates := (1 / Real.sqrt (3 - Real.sqrt 3), 0 : ℝ × ℝ)

-- Define the conditions
variable (xF : ℝ)
variable (F_valid : xF > 0)
variable (parabola_constraint : ∀ (x y : ℝ), y ^ 2 = 2 * xF * x ↔ ∃ (P : ℝ × ℝ), P.2 ^ 2 = 2 * xF * P.1 ∧ P.1 > 0 ∧ P.2 > 0)
variable (tangent_constraint : ∀ (PQ : ℝ), PQ = 2 ↔ ∃ (xP yP : ℝ), xP > 0 ∧ yP > 0 ∧ (PQ = 2 ∧ ∃ (xQ : ℝ), xQ < 0 ∧ |PQ| = |(xP - xQ, yP - 0)|))
variable (circle_tangent_constraint : ∀ (C1_radius C2_radius : ℝ), 
  ∃ (P_radius : ℝ), P_radius > 0 ∧ (C1_radius + C2_radius) minimized ∧ tangent)

theorem minimize_circle_areas : F_coordinates = (1 / Real.sqrt (3 - Real.sqrt 3), 0) :=
by
  sorry

end minimize_circle_areas_l108_108505


namespace arithmetic_sequence_y_l108_108263

theorem arithmetic_sequence_y : 
  ∀ (x y z : ℝ), (23 : ℝ), x, y, z, (47 : ℝ) → 
  (y = (23 + 47) / 2) → y = 35 :=
by
  intro x y z h1
  intro h2
  simp at *
  sorry

end arithmetic_sequence_y_l108_108263


namespace fractional_part_sum_leq_l108_108227

noncomputable def fractional_part (z : ℝ) : ℝ :=
  z - (⌊z⌋ : ℝ)

theorem fractional_part_sum_leq (x y : ℝ) :
  fractional_part (x + y) ≤ fractional_part x + fractional_part y :=
by
  sorry

end fractional_part_sum_leq_l108_108227


namespace positive_integer_divisors_g2023_l108_108159

def g (n : ℕ) : ℕ := 2 ^ n

theorem positive_integer_divisors_g2023 : 
  ∃ (n : ℕ), g(2023) = 2 ^ 2023 ∧ (∀ k : ℕ, (0 ≤ k ∧ k ≤ 2023) ↔ k = 2 ^ k) ∧ n = 2024 := 
by sorry

end positive_integer_divisors_g2023_l108_108159


namespace BE_eq_CE_l108_108665

-- Define the geometric objects and their properties.
variables {A B C D M E : Type}
variables [trapezoid ABCD] -- ABCD is defined as a trapezoid
variables (hAB_eq_BD : AB = BD) -- AB equals diagonal BD
variables (hM_mid_AC : midpoint M AC) -- M is the midpoint of AC
variables (hBM_intersect_CD : intersect BM CD = E) -- Line BM intersects CD at E

-- The theorem to be proved: BE equals CE
theorem BE_eq_CE : BE = CE :=
sorry

end BE_eq_CE_l108_108665


namespace train_crossing_time_l108_108637

theorem train_crossing_time
  (L_train : ℕ) (L_bridge : ℕ) (v_kmph : ℕ) :
  L_train = 250 → L_bridge = 390 → v_kmph = 72 → 
  let v_mps := v_kmph * 1000 / 3600,
      total_distance := L_train + L_bridge,
      t := total_distance / v_mps in
  t = 32 :=
by
  intros h1 h2 h3
  let v_mps := v_kmph * 1000 / 3600
  let total_distance := L_train + L_bridge
  let t := total_distance / v_mps
  rw [h1, h2, h3]
  -- the proof will proceed from here, but we will use sorry to indicate the proof is incomplete
  sorry

end train_crossing_time_l108_108637


namespace betty_paid_44_l108_108106

def slippers := 6
def slippers_cost := 2.5
def lipstick := 4
def lipstick_cost := 1.25
def hair_color := 8
def hair_color_cost := 3

noncomputable def total_cost := (slippers * slippers_cost) + (lipstick * lipstick_cost) + (hair_color * hair_color_cost)

theorem betty_paid_44 : total_cost = 44 :=
by
  sorry

end betty_paid_44_l108_108106


namespace am_gm_inequality_l108_108315

theorem am_gm_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a^3 + b^3 + a + b ≥ 4 * a * b :=
by
  sorry

end am_gm_inequality_l108_108315


namespace a_22_value_a_ij_expression_l108_108307

-- Definition of disjoint subsets and factorial
def disjoint_sets (P : Finset ℕ) (Ps : Finset (Finset ℕ)) : Prop :=
∀ P1 P2 ∈ Ps, P1 ≠ P2 → P1 ∩ P2 = ∅

-- Proof statement 1: a_22 = 9
theorem a_22_value : 
∃ P1 P2 : Finset ℕ, (P1 ∪ P2 = {1, 2}) ∧ (P1 ∩ P2 = ∅) ∧ 9 = 9 :=
by {
  sorry
}

-- Proof statement 2: a_ij = (2^j - 1)^i
theorem a_ij_expression (i j : ℕ) : 
∃ (Ps : Finset (Finset ℕ)), 
(disjoint_sets (Finset.range (i+1)) Ps) ∧
((Finset.card Ps = j) → (Finset.card ((Finset.range (i+1))^P) = (2^j - 1)^i)) :=
by {
  sorry
}

end a_22_value_a_ij_expression_l108_108307


namespace tree_growth_rate_l108_108672

noncomputable def growth_rate_per_week (initial_height final_height : ℝ) (months weeks_per_month : ℕ) : ℝ :=
  (final_height - initial_height) / (months * weeks_per_month)

theorem tree_growth_rate :
  growth_rate_per_week 10 42 4 4 = 2 := 
by
  sorry

end tree_growth_rate_l108_108672


namespace polygon_sides_l108_108771

theorem polygon_sides (x : ℕ) (h : 180 * (x - 2) = 1080) : x = 8 :=
by sorry

end polygon_sides_l108_108771


namespace betty_total_cost_l108_108109

theorem betty_total_cost :
    (6 * 2.5) + (4 * 1.25) + (8 * 3) = 44 :=
by
    sorry

end betty_total_cost_l108_108109


namespace necessary_sample_measure_for_proportion_of_height_range_l108_108025

-- Definitions based on given conditions
def Mean := sorry
def Variance := sorry
def Mode := sorry
def FrequencyDistribution := sorry

-- Theorem in proof format
theorem necessary_sample_measure_for_proportion_of_height_range :
  ∀ {height_range : Type} (sample : height_range -> Prop),
  ∃ (measure : Type),
  measure = FrequencyDistribution ->
  ( ∀ (students : height_range -> Prop),
    (students = sample) -> True ) :=
λ height_range sample,
  ⟨FrequencyDistribution, by simp [FrequencyDistribution]⟩

end necessary_sample_measure_for_proportion_of_height_range_l108_108025


namespace parabola_y_intercepts_l108_108984

theorem parabola_y_intercepts : 
  (∀ y, (4 * y ^ 2 - 8 * y + 4 = 0) → y = 1) :=
begin
  assume y h,
  -- Simplify and solve the quadratic equation
  sorry
end

end parabola_y_intercepts_l108_108984


namespace probability_of_odd_sum_l108_108573

open Nat

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_odd_sum :
  (binomial 11 3) / (binomial 12 4) = 1 / 3 := by
sorry

end probability_of_odd_sum_l108_108573


namespace cost_price_of_each_watch_l108_108021

-- Define the given conditions.
def sold_at_loss (C : ℝ) := 0.925 * C
def total_transaction_price (C : ℝ) := 3 * C * 1.053
def sold_for_more (C : ℝ) := 0.925 * C + 265

-- State the theorem to prove the cost price of each watch.
theorem cost_price_of_each_watch (C : ℝ) :
  3 * sold_for_more C = total_transaction_price C → C = 2070.31 :=
by
  intros h
  sorry

end cost_price_of_each_watch_l108_108021


namespace method_1_more_cost_effective_l108_108084

open BigOperators

def racket_price : ℕ := 20
def shuttlecock_price : ℕ := 5
def rackets_bought : ℕ := 4
def shuttlecocks_bought : ℕ := 30
def discount_rate : ℚ := 0.92

def total_price (rackets shuttlecocks : ℕ) := racket_price * rackets + shuttlecock_price * shuttlecocks

def method_1_cost (rackets shuttlecocks : ℕ) := 
  total_price rackets shuttlecocks - shuttlecock_price * rackets

def method_2_cost (total : ℚ) :=
  total * discount_rate

theorem method_1_more_cost_effective :
  method_1_cost rackets_bought shuttlecocks_bought
  <
  method_2_cost (total_price rackets_bought shuttlecocks_bought) :=
by
  sorry

end method_1_more_cost_effective_l108_108084


namespace time_to_fill_bucket_completely_l108_108057

-- Define the constants and the condition from the problem
constant two_thirds_fill_time : ℕ := 90
constant one : ℚ := 1
constant two_thirds : ℚ := 2 / 3

-- Define the statement to prove
theorem time_to_fill_bucket_completely :
  ∃ t : ℕ, two_thirds * t = two_thirds_fill_time → t = 135 :=
by
  -- The proof is omitted for this task, hence adding a sorry
  sorry

end time_to_fill_bucket_completely_l108_108057


namespace solve_for_x_l108_108988

theorem solve_for_x (x : ℕ) (h : 2^12 = 64^x) : x = 2 :=
by {
  sorry
}

end solve_for_x_l108_108988


namespace floor_T_squared_l108_108316

noncomputable def T : ℝ := ∑ i in Finset.range 3007 + 1, Real.sqrt (1 + 1/((i:ℝ) * (i:ℝ)) + 1/((i + 2:ℝ) * (i + 2:ℝ)))

theorem floor_T_squared : ⌊T ^ 2⌋ = 9048059 := by
  sorry

end floor_T_squared_l108_108316


namespace tan_alpha_plus_pi_div_four_l108_108223

theorem tan_alpha_plus_pi_div_four (α : ℝ) (h : (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 3) : 
  Real.tan (α + Real.pi / 4) = -3 := 
by 
  sorry

end tan_alpha_plus_pi_div_four_l108_108223


namespace bcm_hens_count_l108_108474

theorem bcm_hens_count (total_chickens : ℕ) (percent_bcm : ℝ) (percent_bcm_hens : ℝ) : ℕ :=
  let total_bcm := total_chickens * percent_bcm
  let bcm_hens := total_bcm * percent_bcm_hens
  bcm_hens

example : bcm_hens_count 100 0.20 0.80 = 16 := by
  sorry

end bcm_hens_count_l108_108474


namespace final_pair_count_l108_108754

-- Define the initial state of ordered pairs
def initial_pairs : List (ℕ × ℕ) :=
  [(2011, 2), (2010, 3), (2009, 4), ..., (1008, 1005), (1007, 1006)]

-- Define the replacement operation function
def replace_pair (p1 p2 : (ℕ × ℕ)) :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  ((x1 * y1 * x2 / y2), (x1 * y1) * y2 / x2)

-- Main theorem statement
theorem final_pair_count (pairs : List (ℕ × ℕ)) :
  list.length pairs = 1 → ∃ x y, pairs = [(x, y)] ∧ x * y = 504510 :=
sorry

end final_pair_count_l108_108754


namespace range_of_a_l108_108179

noncomputable def p (a : ℝ) : Prop :=
∀ x ∈ set.Icc (0 : ℝ) 1, a ≥ Real.exp x

noncomputable def q (a : ℝ) : Prop :=
∃ x : ℝ, x^2 + 4*x + a = 0

theorem range_of_a (a : ℝ) : p a ∧ q a → a ∈ set.Icc (Real.exp 1) 4 :=
by sorry

end range_of_a_l108_108179


namespace fraction_equation_solution_l108_108226

theorem fraction_equation_solution (x y : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 5) (hy1 : y ≠ 0) (hy2 : y ≠ 7)
  (h : (3 / x) + (2 / y) = 1 / 3) : 
  x = (9 * y) / (y - 6) :=
sorry

end fraction_equation_solution_l108_108226


namespace exists_close_numbers_l108_108720

theorem exists_close_numbers (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 < 1) (h2 : 0 ≤ x2 ∧ x2 < 1) (h3 : 0 ≤ x3 ∧ x3 < 1) :
  ∃ a b ∈ ({x1, x2, x3} : set ℝ), a ≠ b ∧ |b - a| < 1 / 2 :=
sorry

end exists_close_numbers_l108_108720


namespace pascal_triangle_row_10_sum_l108_108870

theorem pascal_triangle_row_10_sum : (∑ (k : Fin 11), nat.choose 10 k) = 1024 := by
  sorry

end pascal_triangle_row_10_sum_l108_108870


namespace axel_total_cost_l108_108507

theorem axel_total_cost :
  let aquarium_original_price := 120
  let markdown_percentage := 0.50
  let coupon_percentage := 0.10
  let additional_items_cost := 75
  let aquarium_tax_percentage := 0.05
  let additional_items_tax_percentage := 0.08
  let aquarium_discount := aquarium_original_price * markdown_percentage
  let aquarium_marked_down_price := aquarium_original_price - aquarium_discount
  let coupon_discount := aquarium_marked_down_price * coupon_percentage
  let aquarium_final_price := aquarium_marked_down_price - coupon_discount
  let aquarium_total_cost := aquarium_final_price + (aquarium_final_price * aquarium_tax_percentage)
  let additional_items_total_cost := additional_items_cost + (additional_items_cost * additional_items_tax_percentage)
  in aquarium_total_cost + additional_items_total_cost = 137.70 :=
begin
  sorry
end

end axel_total_cost_l108_108507


namespace sphere_radius_eq_three_l108_108231

theorem sphere_radius_eq_three (R : ℝ) :
  4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3 → R = 3 :=
by
  intro h
  have h_eq : R^2 = (1 / 3) * R^3 := by
    have h_canceled : R^2 = (1 / 3) * R^3 := by
      -- simplify the given condition to the core relation
      sorry
  have h_nonzero : R ≠ 0 := by
    -- argument ensuring R is nonzero
    sorry
  have h_final : R = 3 := by
    -- deduce the radius
    sorry
  exact h_final

end sphere_radius_eq_three_l108_108231


namespace top_view_area_proof_l108_108007

def rectangular_prism (length : ℕ) (width : ℕ) (height : ℕ) : Type := sorry

noncomputable def top_view_area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem top_view_area_proof : top_view_area 4 2 = 8 :=
  by simp [top_view_area]; sorry

end top_view_area_proof_l108_108007


namespace closest_whole_number_8_exp_l108_108545

theorem closest_whole_number_8_exp :
  let expr := (8^1500 + 8^1502) / (2 * 8^1501)
  abs ((expr : ℝ) - 4) < 1 := 
by
  let num := 8^1500 + 8^1502
  let denom := 2 * 8^1501
  have h_expr : expr = num / denom,
  simp only [expr],
  sorry

end closest_whole_number_8_exp_l108_108545


namespace arithmetic_sequence_middle_term_l108_108274

variable (x y z : ℝ)

theorem arithmetic_sequence_middle_term :
  (23, x, y, z, 47) → y = 35 :=
by
  intro h
  have h1 : y = (23 + 47) / 2 := sorry
  have h2 : y = 35 := sorry
  exact h2

end arithmetic_sequence_middle_term_l108_108274


namespace three_cards_probability_l108_108418

noncomputable def probability_first_king_second_queen_third_heart : ℚ :=
  (4 / 52) * (4 / 51) * (12 / 50)

theorem three_cards_probability :
  probability_first_king_second_queen_third_heart = 8 / 5525 := by
  sorry

end three_cards_probability_l108_108418


namespace range_is_correct_l108_108746

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x

def domain : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

def range_of_function : Set ℝ := {y | ∃ x ∈ domain, quadratic_function x = y}

theorem range_is_correct : range_of_function = Set.Icc (-4) 21 :=
by {
  sorry
}

end range_is_correct_l108_108746


namespace slope_of_OP_l108_108377

-- Define the ellipse and the line equations
def ellipse_eq (x y : ℝ) : Prop := x^2 + 2 * y^2 = 2
def line_eq (x y k : ℝ) : Prop := x + y = k

-- Midpoint P(x0, y0) of M(x1, y1) and N(x2, y2)
def midpoint (x1 y1 x2 y2 x0 y0 : ℝ) : Prop :=
  x1 + x2 = 2 * x0 ∧ y1 + y2 = 2 * y0

-- Prove the slope of OP is 1/2
theorem slope_of_OP (x1 y1 x2 y2 x0 y0 k : ℝ)
  (h_ellipse_M : ellipse_eq x1 y1)
  (h_ellipse_N : ellipse_eq x2 y2)
  (h_line_M : line_eq x1 y1 k)
  (h_line_N : line_eq x2 y2 k)
  (h_midpoint : midpoint x1 y1 x2 y2 x0 y0) :
  (y0 / x0) = 1 / 2 :=
sorry

end slope_of_OP_l108_108377


namespace distinct_terms_in_expansion_l108_108134

theorem distinct_terms_in_expansion (x y : ℝ) : 
  let expr := ((x + 4 * y) ^ 2 * (x - 4 * y) ^ 2) ^ 3
  in distinct_terms_in_expansion expr = 7 :=
by
  sorry

end distinct_terms_in_expansion_l108_108134


namespace exists_rational_non_integer_xy_no_rational_non_integer_xy_l108_108528

-- Part (a)
theorem exists_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  (∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
   ∃ z1 z2 : ℤ, 19 * x + 8 * y = ↑z1 ∧ 8 * x + 3 * y = ↑z2) :=
sorry

-- Part (b)
theorem no_rational_non_integer_xy 
  (x y : ℚ) (h1 : ¬ ∃ z : ℤ, x = z ∧ y = z) : 
  ¬ ∃ x y : ℚ, ¬(∃ z : ℤ, x = z ∨ y = z) ∧ 
  ∃ z1 z2 : ℤ, 19 * x^2 + 8 * y^2 = ↑z1 ∧ 8 * x^2 + 3 * y^2 = ↑z2 :=
sorry

end exists_rational_non_integer_xy_no_rational_non_integer_xy_l108_108528


namespace find_angle_PBC_l108_108674

-- Define the triangle ABC and point P with the given conditions
variable (A B C P : Type)
variable [Geometry ABC]
variable [Interior P ABC]

-- Define the given angles
variable (angle_BPA : ℝ)
variable (angle_ABP : ℝ)
variable (angle_PCA : ℝ)
variable (angle_PAC : ℝ)

axiom angle_BAP_10 : angle_BPA = 10
axiom angle_ABP_20 : angle_ABP = 20
axiom angle_PCA_30 : angle_PCA = 30
axiom angle_PAC_40 : angle_PAC = 40

-- Prove that the angle PBC is 60 degrees
theorem find_angle_PBC : (angle P B C = 60) :=
by {
    -- Here is where the proof would go
    sorry
}

end find_angle_PBC_l108_108674


namespace common_ratio_geometric_progression_l108_108908

noncomputable def a (x y z w : ℝ) : ℝ := x * (y - z)
noncomputable def ar (x y z w : ℝ) (r : ℝ) : ℝ := y * (z - x)
noncomputable def ar2 (x y z w : ℝ) (r : ℝ) : ℝ := z * (x - y)
noncomputable def ar3 (x y z w : ℝ) (r : ℝ) : ℝ := w * (x - y)

theorem common_ratio_geometric_progression 
  (x y z w : ℝ) (r : ℝ)
  (hxz : x ≠ z) (hxy : x ≠ y) (hyz : y ≠ z) (hwy : w ≠ y)
  (hxy0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) (hw0 : w ≠ 0)
  (h1 : a x y z w ≠ 0)
  (ha : a x y z w = x * (y - z))
  (har : ar x y z w r = y * (z - x))
  (har2 : ar2 x y z w r = z * (x - y))
  (har3 : ar3 x y z w r = w * (x - y))
  (hr : ar x y z w r = r * a x y z w) 
  (hr2 : ar2 x y z w r = r * ar x y z w r)
  (hr3 : ar3 x y z w r = r * ar2 x y z w r) :
  r^3 + r^2 + r + 1 = 0 := by
  sorry

end common_ratio_geometric_progression_l108_108908


namespace solution_set_l108_108376

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := deriv f x

-- Definitions for conditions
axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom f_at_0 : f 0 = 2
axiom f_ineq : ∀ x : ℝ, f x + f' x > 1

-- Theorem statement
theorem solution_set (x: ℝ) : (e^x * f x > e^x + 1) ↔ (0 < x) :=
by 
  sorry

end solution_set_l108_108376


namespace repeating_decimal_division_l108_108547

def repeating_decimal_142857 : ℚ := 1 / 7
def repeating_decimal_2_857143 : ℚ := 20 / 7

theorem repeating_decimal_division :
  (repeating_decimal_142857 / repeating_decimal_2_857143) = 1 / 20 :=
by
  sorry

end repeating_decimal_division_l108_108547


namespace evaluate_expression_l108_108901

theorem evaluate_expression (y : ℕ) (h₁ : y = 2) (x : ℕ) (h₂ : x = y + 1) :
    5 * Nat.factorial y * x^y + 3 * Nat.factorial x * y^x = 234 := by
  rw [h₁] at h₂
  rw [←h₁]
  sorry

end evaluate_expression_l108_108901


namespace cubed_inequality_l108_108682

variable {a b : ℝ}

theorem cubed_inequality (h : a > b) : a^3 > b^3 :=
sorry

end cubed_inequality_l108_108682


namespace arithmetic_sequence_middle_term_l108_108275

variable (x y z : ℝ)

theorem arithmetic_sequence_middle_term :
  (23, x, y, z, 47) → y = 35 :=
by
  intro h
  have h1 : y = (23 + 47) / 2 := sorry
  have h2 : y = 35 := sorry
  exact h2

end arithmetic_sequence_middle_term_l108_108275


namespace summation_identity_specific_summation_identity_l108_108818

theorem summation_identity (n : ℕ) (a : ℝ) :
  (∑ i in finset.range n, (i + 1) * (a - (i + 1))) = n * (n + 1) * (3 * a - 2 * n - 1) / 6 := sorry

theorem specific_summation_identity (n : ℕ) :
  (∑ i in finset.range n, (i + 1) * (n - i)) = n * (n + 1) * (n + 2) / 6 := sorry

end summation_identity_specific_summation_identity_l108_108818


namespace tan_alpha_frac_l108_108939

theorem tan_alpha_frac (α : ℝ) (h : Real.tan α = 2) : (Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 1 / 11 := by
  sorry

end tan_alpha_frac_l108_108939


namespace p_square_values_l108_108915

def p (x : ℤ) : ℤ := 4 * x^4 - 12 * x^3 + 17 * x^2 - 6 * x - 14

theorem p_square_values : {x : ℤ | ∃ k : ℤ, p x = k^2}.to_finset.card = 2 := 
sorry

end p_square_values_l108_108915


namespace segment_perpendicular_to_line_l108_108804

noncomputable def triangle := Type*

variables {A B C A1 B1 A0 B0 C_prime H O : triangle}
variables {AA1 BB1 : ∀{A B C : triangle}, triangle}
variables (C' : triangle)

-- The following conditions are applied:
axiom altitudes (ABC : triangle) (AA1 : ∀ A B C : triangle, triangle) (BB1 : ∀ A B C : triangle, triangle) : Prop
axiom intersects_midline (A1 B1 : triangle) (C' : triangle) : Prop
axiom orthocenter (ABC H : triangle) : Prop
axiom circumcenter (ABC O : triangle) : Prop
axiom perpendicular (CC_prime : triangle) (HO : triangle) : Prop

-- The main statement to prove:
theorem segment_perpendicular_to_line (ABC : triangle)
  (AA1 : ∀{A B C : triangle}, triangle) (BB1 : ∀{A B C : triangle}, triangle)
  (C' : triangle) 
  (H O : triangle)
  (h1 : altitudes ABC AA1 BB1)
  (h2 : intersects_midline (AA1 A B C) (BB1 A B C) C')
  (h3 : orthocenter ABC H)
  (h4 : circumcenter ABC O) :
  perpendicular (AA1 A B C) (BB1 A B C) C' H O :=
sorry

end segment_perpendicular_to_line_l108_108804


namespace probability_same_number_selected_l108_108862

theorem probability_same_number_selected :
  let Billy_numbers := {n : ℕ | 1 ≤ n ∧ n < 300 ∧ n % 20 = 0},
      Bobbi_numbers := {n : ℕ | 1 ≤ n ∧ n < 300 ∧ n % 30 = 0},
      common_numbers := Billy_numbers ∩ Bobbi_numbers
  in (↑(common_numbers.card) : ℚ) / (↑(Billy_numbers.card * Bobbi_numbers.card) : ℚ) = 1 / 30 :=
by
  sorry

end probability_same_number_selected_l108_108862


namespace total_teeth_of_sharks_l108_108086

noncomputable def tiger_shark_teeth : ℕ := 180
noncomputable def hammerhead_shark_teeth : ℕ := (1/6 : ℚ) * tiger_shark_teeth
noncomputable def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)
noncomputable def mako_shark_teeth : ℕ := (5/3 : ℚ) * hammerhead_shark_teeth

theorem total_teeth_of_sharks :
  tiger_shark_teeth + hammerhead_shark_teeth.toNat + great_white_shark_teeth + mako_shark_teeth.toNat = 680 :=
by
  sorry

end total_teeth_of_sharks_l108_108086


namespace value_of_f_f_neg1_l108_108578

def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x + 3 else 2 * x^2 + 1

theorem value_of_f_f_neg1 :
  f (f (-1)) = 3 :=
by
  sorry

end value_of_f_f_neg1_l108_108578


namespace incorrect_statement_l108_108052

theorem incorrect_statement :
  let statementA := "The shortest distance between two points is a line segment."
  let statementB := "Vertical angles are congruent."
  let statementC := "Complementary angles of the same measure are congruent."
  let statementD := "There is only one line passing through a point outside a given line that is parallel to the given line."
  (statementA = "correct") ∧ 
  (statementB = "correct") ∧ 
  (statementC = "correct") ∧ 
  (statementD = "incorrect") :=
by
  let statementA := "The shortest distance between two points is a line segment."
  let statementB := "Vertical angles are congruent."
  let statementC := "Complementary angles of the same measure are congruent."
  let statementD := "There is only one line passing through a point outside a given line that is parallel to the given line."
  have hA : statementA = "correct" := sorry
  have hB : statementB = "correct" := sorry
  have hC : statementC = "correct" := sorry
  have hD : statementD = "incorrect" := sorry
  exact ⟨hA, hB, hC, hD⟩

end incorrect_statement_l108_108052


namespace decreasing_interval_of_composite_function_l108_108009

def quadratic (x : ℝ) : ℝ := x^2 + 2 * x - 3

def composite_function (x : ℝ) : ℝ := sqrt (quadratic x)

noncomputable def monotonic_decreasing_interval := { x : ℝ | composite_function x }

theorem decreasing_interval_of_composite_function :
  monotonic_decreasing_interval = set.Iic (-3) :=
sorry

end decreasing_interval_of_composite_function_l108_108009


namespace period_of_sin_3x_plus_pi_l108_108040

def sine_period (b : ℝ) : ℝ := 2 * Real.pi / abs b

theorem period_of_sin_3x_plus_pi :
  sine_period 3 = 2 * Real.pi / 3 :=
by
  sorry

end period_of_sin_3x_plus_pi_l108_108040


namespace length_of_goods_train_l108_108477

-- Definitions
def speed_kmh : ℝ := 72 -- Speed of the train in km/hr
def length_platform : ℝ := 290 -- Length of the platform in meters
def time_seconds : ℝ := 26 -- Time taken to cross the platform in seconds

-- Conversion factor
def kmh_to_mps (v : ℝ) : ℝ := v * (5 / 18)

-- Speed in m/s
def speed_mps : ℝ := kmh_to_mps speed_kmh

-- Total distance covered while crossing the platform
def total_distance : ℝ := speed_mps * time_seconds

-- Length of the train
def length_train : ℝ := total_distance - length_platform

theorem length_of_goods_train :
  length_train = 230 := by
  sorry

end length_of_goods_train_l108_108477


namespace sum_of_solutions_l108_108438

theorem sum_of_solutions (n : ℕ) (h1 : ∀ n, n >= 1 → math.lcm n 120 = Int.gcd n 120 + 300 → n = 180) : n = 180 :=
by
  sorry

end sum_of_solutions_l108_108438


namespace cyclic_ngon_possible_l108_108831

def cyclic_ngon (n : ℕ) : Prop :=
  ∃ (triangles : list (triangle ℝ)), 
    is_divided_by_non_intersecting_diagonals n triangles ∧
    (∀ t ∈ triangles, ∃ t' ∈ triangles, t ≈ t')

theorem cyclic_ngon_possible (n : ℕ) : 
  cyclic_ngon n ↔ n = 4 ∨ ∃ (k : ℕ), k ≥ 2 ∧ n = 2 * k :=
by
  sorry

end cyclic_ngon_possible_l108_108831


namespace geom_seq_solution_l108_108406

theorem geom_seq_solution (a b x y : ℝ) 
  (h1 : x * (1 + y + y^2) = a) 
  (h2 : x^2 * (1 + y^2 + y^4) = b) :
  x = 1 / (4 * a) * (a^2 + b - Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∨ 
  x = 1 / (4 * a) * (a^2 + b + Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∧
  y = 1 / (2 * (a^2 - b)) * (a^2 + b - Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∨
  y = 1 / (2 * (a^2 - b)) * (a^2 + b + Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) := 
  sorry

end geom_seq_solution_l108_108406


namespace fifth_term_sequence_l108_108212

theorem fifth_term_sequence 
  (a : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h2 : a 2 = 6)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 5 = -6 := 
by
  sorry

end fifth_term_sequence_l108_108212


namespace find_real_solutions_l108_108140

theorem find_real_solutions (x y z : ℝ) :
  (tan x) ^ 2 + 2 * (cot (2 * y)) ^ 2 = 1 ∧
  (tan y) ^ 2 + 2 * (cot (2 * z)) ^ 2 = 1 ∧
  (tan z) ^ 2 + 2 * (cot (2 * x)) ^ 2 = 1 →
  (x = π / 6 ∧ y = π / 6 ∧ z = π / 6) ∨
  (x = π / 4 ∧ y = π / 4 ∧ z = π / 4) :=
by
  sorry

end find_real_solutions_l108_108140


namespace max_good_subset_size_l108_108695

open scoped BigOperators

def Sn (n : ℕ) : Set (Fin 2^n → Bool) := { a | true }

def d (n : ℕ) (a b : Fin 2^n → Bool) : ℕ :=
  ∑ i, if a i = b i then 0 else 1

def is_good_subset (n : ℕ) (A : Set (Fin 2^n → Bool)) : Prop :=
  ∀ (a b : Fin 2^n → Bool), a ∈ A → b ∈ A → a ≠ b → d n a b ≥ 2^(n-1)

theorem max_good_subset_size (n : ℕ) :
  ∃ (A : Set (Fin 2^n → Bool)), is_good_subset n A ∧ A.finite ∧ A.card = 2^n :=
sorry

end max_good_subset_size_l108_108695


namespace number_is_2_point_5_l108_108902

theorem number_is_2_point_5 (x : ℝ) (h: x^2 + 50 = (x - 10)^2) : x = 2.5 := 
by
  sorry

end number_is_2_point_5_l108_108902


namespace function_passes_four_quadrants_l108_108383

-- Define the function f(x)
def f (a x : ℝ) : ℝ := (1/3) * a * x^3 + (1/2) * a * x^2 - 2 * a * x + 2 * a + 1

-- Define the derivative of f(x)
def f_prime (a x : ℝ) : ℝ := a * x^2 + a * x - 2 * a

-- Problem statement in Lean 4
theorem function_passes_four_quadrants (a : ℝ) :
  (-6/5 < a ∧ a < -3/16) → 
  (∃ x : ℝ, f a x < 0 ∧ ∃ x : ℝ, f a x > 0 ∧ ∃ x : ℝ, f a x = 0) :=
by
  sorry

end function_passes_four_quadrants_l108_108383


namespace smallest_x_value_l108_108893

theorem smallest_x_value (x : ℝ) (h₁ : x ≠ 6) (h₂ : x ≠ -4) : 
  (x = (sqrt 21 - 7) / 2) → 
  (x = (-sqrt 21 - 7) / 2) :=
sorry

end smallest_x_value_l108_108893


namespace fraction_product_equals_64_l108_108875

theorem fraction_product_equals_64 : 
  (1 / 4) * (8 / 1) * (1 / 32) * (64 / 1) * (1 / 128) * (256 / 1) * (1 / 512) * (1024 / 1) * (1 / 2048) * (4096 / 1) * (1 / 8192) * (16384 / 1) = 64 :=
by
  sorry

end fraction_product_equals_64_l108_108875


namespace infinite_product_value_l108_108136

theorem infinite_product_value :
  (∀ n : ℕ, n > 0 → (2^n)^(1/(3^n))) →
  (∏' n : ℕ, (2^n)^(1/(3^n))) = real.sqrt (real.sqrt 8) :=
by
  sorry

end infinite_product_value_l108_108136


namespace betty_paid_total_l108_108102

def cost_slippers (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_lipsticks (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_hair_colors (count : ℕ) (price : ℝ) : ℝ := count * price

def total_cost := 
  cost_slippers 6 2.5 +
  cost_lipsticks 4 1.25 +
  cost_hair_colors 8 3

theorem betty_paid_total :
  total_cost = 44 := 
  sorry

end betty_paid_total_l108_108102


namespace millions_place_correct_l108_108703

def number := 345000000
def hundred_millions_place := number / 100000000 % 10  -- 3
def ten_millions_place := number / 10000000 % 10  -- 4
def millions_place := number / 1000000 % 10  -- 5

theorem millions_place_correct : millions_place = 5 := 
by 
  -- Mathematical proof goes here
  sorry

end millions_place_correct_l108_108703


namespace quad_area_l108_108483

theorem quad_area (a b : Int) (h1 : a > b) (h2 : b > 0) (h3 : 2 * |a - b| * |a + b| = 50) : a + b = 15 :=
by
  sorry

end quad_area_l108_108483


namespace smallest_positive_period_intervals_of_monotonic_decrease_max_min_values_l108_108960

noncomputable def f (x : ℝ) : ℝ := cos (π / 2 - x) * cos x + sqrt 3 * sin x ^ 2

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

theorem intervals_of_monotonic_decrease :
  ∀ k : ℤ, ∀ x ∈ Icc (5 * π / 12 + k * π) (11 * π / 12 + k * π), 
  ∀ y ∈ Icc (5 * π / 12 + k * π) (11 * π / 12 + k * π), x < y → f x ≥ f y := sorry

theorem max_min_values :
  ∃ (max min : ℝ), max = sqrt 3 / 2 + 1 ∧ min = sqrt 3 / 2 ∧
  ∀ x ∈ Icc (π / 6) (π / 2), f x ≤ max ∧ f x ≥ min := sorry

end smallest_positive_period_intervals_of_monotonic_decrease_max_min_values_l108_108960


namespace geometric_sequence_sum_l108_108656

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
    (h1 : a 2 = 2) 
    (h2 : 1 / a 1 + 1 / (a 1 * q^2) = 5 / 4) :
    a 1 + a 1 * q^2 = 5 :=
begin
    sorry
end

end geometric_sequence_sum_l108_108656


namespace find_a_l108_108642

noncomputable def a : ℚ := ((68^3 - 65^3) * (32^3 + 18^3)) / ((32^2 - 32 * 18 + 18^2) * (68^2 + 68 * 65 + 65^2))

theorem find_a : a = 150 := 
  sorry

end find_a_l108_108642


namespace NumFriendsNextToCaraOnRight_l108_108877

open Nat

def total_people : ℕ := 8
def freds_next_to_Cara : ℕ := 7

theorem NumFriendsNextToCaraOnRight (h : total_people = 8) : freds_next_to_Cara = 7 :=
by
  sorry

end NumFriendsNextToCaraOnRight_l108_108877


namespace algebraic_expression_value_l108_108941

-- Definitions based on the conditions
variable {a : ℝ}
axiom root_equation : 2 * a^2 + 3 * a - 4 = 0

-- Definition of the problem: Proving that 2a^2 + 3a equals 4.
theorem algebraic_expression_value : 2 * a^2 + 3 * a = 4 :=
by 
  have h : 2 * a^2 + 3 * a - 4 = 0 := root_equation
  have h' : 2 * a^2 + 3 * a = 4 := by sorry
  exact h'

end algebraic_expression_value_l108_108941


namespace george_abe_combined_time_l108_108921

-- Define the rates for George and Abe
def georgeRate : ℝ := 1 / 70
def abeRate : ℝ := 1 / 30

-- Define the combined rate
def combinedRate : ℝ := 1 / 21

-- Theorem: From the given conditions, prove that the time to complete 1 job is 21 minutes
theorem george_abe_combined_time : (georgeRate + abeRate) = combinedRate :=
  sorry

end george_abe_combined_time_l108_108921


namespace outstanding_non_members_probability_l108_108447

theorem outstanding_non_members_probability :
  let p_total := 0.10
      p_members := 0.20
      p_outstanding_members := 0.40
  in ∃ x : ℝ, ((p_members * p_outstanding_members) + ((1 - p_members) * x) = p_total) ∧ (x = 0.025) := by
  use 0.025
  -- proof to be added
  sorry

end outstanding_non_members_probability_l108_108447


namespace Edward_earnings_l108_108897

theorem Edward_earnings (lawns_mowed : ℕ) (cars_washed : ℕ) (fences_painted : ℕ)
  (earnings_per_lawn earnings_per_car earnings_per_fence : ℕ)
  (total_lawns needed_lawns : ℕ)
  (total_cars needed_cars : ℕ)
  (total_fences needed_fences : ℕ)
  (hl : lawns_mowed = total_lawns - needed_lawns)
  (he_l : earnings_per_lawn = 4)
  (total_lawns : total_lawns = 17)
  (needed_lawns : needed_lawns = 9)
  (total_cars : total_cars = 25)
  (needed_cars : needed_cars = 8)
  (total_fences : total_fences = 5)
  (needed_fences : needed_fences = 3)
  (hc : cars_washed = needed_cars)
  (he_c : earnings_per_car = 6)
  (hf : fences_painted = needed_fences)
  (he_f : earnings_per_fence = 10) :
  (lawns_mowed * earnings_per_lawn + cars_washed * earnings_per_car + fences_painted * earnings_per_fence) = 110 := 
by
  sorry

end Edward_earnings_l108_108897


namespace work_completion_days_l108_108450

theorem work_completion_days (A B : ℕ) (hA : A = 20) (hB : B = 20) : A + B / (A + B) / 2 = 10 :=
by 
  rw [hA, hB]
  -- Proof omitted
  sorry

end work_completion_days_l108_108450


namespace eigenvalues_and_eigenvectors_conversion_and_relationship_inequality_solution_range_l108_108019

-- Problem (1)
@[simp]
def matrix_A := ![![2, 1], ![3, 0]]

theorem eigenvalues_and_eigenvectors :
  ∃ (λ1 λ2 : ℝ) (v1 v2 : Fin 2 → ℝ),
    (λ1 = -1) ∧ (λ2 = 3) ∧
    (matrix_A.mul_vec v1 = λ1 • v1) ∧
    (matrix_A.mul_vec v2 = λ2 • v2) ∧
    (v1 = ![1, -3]) ∧ (v2 = ![1, 1]) :=
sorry

-- Problem (2)
def parametric_line (t : ℝ) : ℝ × ℝ := (t, 1 + 2 * t)

def polar_circle (θ : ℝ) : {p : ℝ // 0 ≤ p} × ℝ := (2 * (Real.sin θ + Real.cos θ), θ)

theorem conversion_and_relationship :
  ∃ (l_eq : ℝ → ℝ) (C_eq : ℝ × ℝ → Prop),
    (∀ t, parametric_line t = (t, l_eq t)) ∧ 
    (∀ θ, polar_circle θ = (2 * (Real.sin θ + Real.cos θ), θ)) ∧ 
    l_eq = (2 : ℝ) * (·) + 1 ∧
    C_eq = λ (p : ℝ × ℝ), (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 = 2 ∧
    ∀ p, (∃ t, parametric_line t = p) ↔ C_eq p :=
sorry

-- Problem (3)
def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

theorem inequality_solution_range (a b : ℝ) (ha : a ≠ 0) :
  ∃ I : Set ℝ, 
    I = {x | (1 / 2) ≤ x ∧ x ≤ (5 / 2)} ∧
    ∀ x, (a + b).abs + (a - b).abs ≥ a.abs * f x → x ∈ I :=
sorry

end eigenvalues_and_eigenvectors_conversion_and_relationship_inequality_solution_range_l108_108019


namespace necessarily_all_knights_l108_108677

theorem necessarily_all_knights (k N : ℕ) (hk : k > 1) (hN : N > 2 * k + 1)
  (h_equally : ∀ (i : ℕ), i < N → (let l_i := (λ j, a ((i + j) % N)) in
                                   let r_i := (λ j, a ((N + i - j) % N)) in
                                   ∃ a : ℕ → ℕ, l_i k = r_i k))
  (gcd_N : Nat.gcd N (2 * k + 1) = 1) : ∀ i : ℕ, i < N → (let a_i := 1) := 
sorry

end necessarily_all_knights_l108_108677


namespace problem_sum_of_fractions_l108_108066

theorem problem_sum_of_fractions : 
  (∑ k in Finset.range 999, 1 / (k + 1) - 1 / (k + 2)) = 1 - 1 / 1000 := 
sorry

end problem_sum_of_fractions_l108_108066


namespace platform_length_l108_108822

-- Given conditions
def train_length : ℝ := 300
def time_cross_signal : ℝ := 18
def time_cross_platform : ℝ := 51

-- Given the speed of the train
def train_speed : ℝ := train_length / time_cross_signal

-- Statement to prove
theorem platform_length :
  train_speed * time_cross_platform = train_length + 550 := by
  sorry

end platform_length_l108_108822


namespace number_of_integer_solutions_l108_108524

theorem number_of_integer_solutions :
  let p (x : Int) := 3 * x^2 + 7 * x - 5 > 0
  in Finset.card (Finset.filter (λ x => ¬p x) (Finset.Icc (-10 : Int) 10)) = 2 :=
by
  -- the goal is proving the number of integer values that do not satisfy 3x^2 + 7x - 5 > 0 is 2
  sorry

end number_of_integer_solutions_l108_108524


namespace subway_distance_per_minute_l108_108449

theorem subway_distance_per_minute :
  let total_distance := 120 -- kilometers
  let total_time := 110 -- minutes (1 hour and 50 minutes)
  let bus_time := 70 -- minutes (1 hour and 10 minutes)
  let bus_distance := (14 * 40.8) / 6 -- kilometers
  let subway_distance := total_distance - bus_distance -- kilometers
  let subway_time := total_time - bus_time -- minutes
  let distance_per_minute := subway_distance / subway_time
  distance_per_minute = 0.62 := 
by
  sorry

end subway_distance_per_minute_l108_108449


namespace number_of_correct_statements_l108_108096

def is_vertical_angle (α β : Angle) : Prop := α = β ∨ α + β = 180

def perpendicular_distance (P : Point) (L : Line) : ℝ := 
  let foot := (P.foot_perpendicular L)
  dist P foot

@[simp]
def shortest_distance (A B : Point) : ℝ := 
  dist A B

axiom unique_perpendicular (P : Point) (L : Line) : unique (λ M, perpendicular P M ∧ P ∈ L)

theorem number_of_correct_statements :
  let stmnt_1 := ∀ (α β : Angle), (¬ (is_vertical_angle α β = (α.vertex = β.vertex ∧ α.measure = β.measure)))
  let stmnt_2 := ∀ (P : Point) (L : Line), 
    perpendicular_distance P L = dist P (P.foot_perpendicular L)
  let stmnt_3 := ∀ (A B : Point), 
    shortest_distance A B = dist A B
  let stmnt_4 := ∀ (P : Point) (L : Line), 
    unique_perpendicular P L
  ∃ n : ℕ, (n = 2) := 
by
  sorry

end number_of_correct_statements_l108_108096


namespace shaded_area_is_correct_l108_108903

-- Definitions based on the conditions
def is_square (s : ℝ) (area : ℝ) : Prop := s * s = area
def rect_area (l w : ℝ) : ℝ := l * w

variables (s : ℝ) (area_s : ℝ) (rect1_l rect1_w rect2_l rect2_w : ℝ)

-- Given conditions
def square := is_square s area_s
def rect1 := rect_area rect1_l rect1_w
def rect2 := rect_area rect2_l rect2_w

-- Problem statement: Prove the area of the shaded region
theorem shaded_area_is_correct
  (s: ℝ)
  (rect1_l rect1_w rect2_l rect2_w : ℝ)
  (h_square: is_square s 16)
  (h_rect1: rect_area rect1_l rect1_w = 6)
  (h_rect2: rect_area rect2_l rect2_w = 2) :
  (16 - (6 + 2) = 8) := 
  sorry

end shaded_area_is_correct_l108_108903


namespace range_of_a_l108_108965

theorem range_of_a {a : ℝ} :
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ x^3 - a * x = 0) ↔ 1 < a ∧ a < 4 :=
begin
  sorry
end

end range_of_a_l108_108965


namespace sum_slope_y_intercept_l108_108337

-- Definitions of points C and D
def C := (2 : ℝ, 3 : ℝ)
def D := (5 : ℝ, 9 : ℝ)

-- Definition of the function to calculate the slope
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Definition of the slope of the line passing through points C and D
def m := slope C D

-- Definition of the y-intercept using the point-slope form of the line equation
def y_intercept (p : ℝ × ℝ) (m : ℝ) : ℝ := p.2 - m * p.1

-- Definition of the y-intercept for the line passing through point C
def b := y_intercept C m

-- The statement to prove
theorem sum_slope_y_intercept : m + b = 1 :=
by
  sorry

end sum_slope_y_intercept_l108_108337


namespace solve_for_x_l108_108989

noncomputable def find_x (x : ℝ) : Prop :=
  2^12 = 64^x

theorem solve_for_x (x : ℝ) (h : find_x x) : x = 2 :=
by
  sorry

end solve_for_x_l108_108989


namespace pyramid_cross_section_imp_l108_108711

theorem pyramid_cross_section_imp (n : ℕ) (h : n ≥ 5) :
    ¬ (exists (P : Set Point) (S : Pyramid),
        (is_regular_ngon (base S) n) ∧ (is_regular_ngon (cross_section S P) (n + 1))) :=
by sorry

end pyramid_cross_section_imp_l108_108711


namespace exists_pos_ints_l108_108785

open Nat

noncomputable def f (a : ℕ) : ℕ :=
  a^2 + 3 * a + 2

noncomputable def g (b c : ℕ) : ℕ :=
  b^2 - b + 3 * c^2 + 3 * c

theorem exists_pos_ints (a : ℕ) (ha : 0 < a) :
  ∃ (b c : ℕ), 0 < b ∧ 0 < c ∧ f a = g b c :=
sorry

end exists_pos_ints_l108_108785


namespace rob_read_pages_l108_108348

-- Definitions for the given conditions
def planned_reading_time_hours : ℝ := 3
def fraction_of_planned_time : ℝ := 3 / 4
def minutes_per_hour : ℝ := 60
def reading_rate_pages_per_minute : ℝ := 1 / 15

-- Helper definition to convert hours to minutes
def planned_reading_time_minutes : ℝ := planned_reading_time_hours * minutes_per_hour

-- Total actual reading time in minutes
def actual_reading_time_minutes : ℝ := fraction_of_planned_time * planned_reading_time_minutes

-- Number of pages read
def number_of_pages_read : ℝ := actual_reading_time_minutes * reading_rate_pages_per_minute

-- The theorem to prove
theorem rob_read_pages : number_of_pages_read = 9 :=
by
  -- Insert structured proof steps here
  sorry

end rob_read_pages_l108_108348


namespace parallelogram_preservation_l108_108254

-- Define the affine points and corresponding ratios
def Parallelogram (P Q R S : Type*) := sorry
def PointOnSide (P Q : Type*) (r : ℝ) := sorry
def RatiosEqual (a b c d e f g h : ℝ) := a / b = c / d = e / f = g / h

theorem parallelogram_preservation {A B C D A₁ B₁ C₁ D₁ A₂ B₂ C₂ D₂ : Type*}
  (h1 : Parallelogram A B C D)
  (h2 : PointOnSide A B rA)
  (h3 : PointOnSide B C rB)
  (h4 : PointOnSide C D rC)
  (h5 : PointOnSide D A rD)
  (h6 : RatiosEqual rA rB rC rD (pA₁D₂ / pD₁D₂) (pD₁C₂ / pC₁C₂) (pC₁B₂ / pB₁B₂) (pB₁A₂ / pA₁A₂))
  : Parallelogram A₂ B₂ C₂ D₂ :=
sorry

end parallelogram_preservation_l108_108254


namespace mark_exceeded_sugar_intake_by_100_percent_l108_108700

-- Definitions of the conditions
def softDrinkCalories : ℕ := 2500
def sugarPercentage : ℝ := 0.05
def caloriesPerCandy : ℕ := 25
def numCandyBars : ℕ := 7
def recommendedSugarIntake : ℕ := 150

-- Calculating the amount of added sugar in the soft drink
def addedSugarSoftDrink : ℝ := sugarPercentage * softDrinkCalories

-- Calculating the total added sugar from the candy bars
def addedSugarCandyBars : ℕ := numCandyBars * caloriesPerCandy

-- Summing the added sugar from the soft drink and the candy bars
def totalAddedSugar : ℝ := addedSugarSoftDrink + (addedSugarCandyBars : ℝ)

-- Calculate the excess intake of added sugar over the recommended amount
def excessSugarIntake : ℝ := totalAddedSugar - (recommendedSugarIntake : ℝ)

-- Prove that the percentage by which Mark exceeded the recommended intake of added sugar is 100%
theorem mark_exceeded_sugar_intake_by_100_percent :
  (excessSugarIntake / (recommendedSugarIntake : ℝ)) * 100 = 100 :=
by
  sorry

end mark_exceeded_sugar_intake_by_100_percent_l108_108700


namespace correct_calculation_among_given_options_l108_108798

theorem correct_calculation_among_given_options
  (h1 : sqrt 4 ≠ ±2)
  (h2 : ± sqrt (1/9) ≠ 1/3)
  (h3 : (sqrt 5)^2 = 5)
  (h4 : cbrt 8 ≠ ±2) :
  (sqrt 5)^2 = 5 :=
by
  sorry

end correct_calculation_among_given_options_l108_108798


namespace find_p5_l108_108311

noncomputable def p (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) + x^2 + 2

theorem find_p5 :
  let p : ℝ → ℝ := (λ x, (x - 1) * (x - 2) * (x - 3) * (x - 4) + x^2 + 2) in
  p 5 = 51 :=
  by {
    assume p,
    show p 5 = 51,
    sorry
  }

end find_p5_l108_108311


namespace probability_same_number_selected_l108_108861

theorem probability_same_number_selected :
  let Billy_numbers := {n : ℕ | 1 ≤ n ∧ n < 300 ∧ n % 20 = 0},
      Bobbi_numbers := {n : ℕ | 1 ≤ n ∧ n < 300 ∧ n % 30 = 0},
      common_numbers := Billy_numbers ∩ Bobbi_numbers
  in (↑(common_numbers.card) : ℚ) / (↑(Billy_numbers.card * Bobbi_numbers.card) : ℚ) = 1 / 30 :=
by
  sorry

end probability_same_number_selected_l108_108861


namespace find_lambda_l108_108631

variable (λ : ℚ)
variable (a : ℚ × ℚ × ℚ := (2, -1, 2))
variable (b : ℚ × ℚ × ℚ := (-1, 3, -3))
variable (c : ℚ × ℚ × ℚ := (13, 6, λ))

def vectors_coplanar (u v w : ℚ × ℚ × ℚ) : Prop :=
  ∃ (m n : ℚ), ∀ (i j k : ℚ), w = (i * u + j * v + k * w)

theorem find_lambda
  (H : vectors_coplanar (2, -1, 2) (-1, 3, -3) (13, 6, λ)) :
  λ = -57 / 17 := by
  sorry

end find_lambda_l108_108631


namespace Rahul_savings_l108_108715

variable (total_savings ppf_savings nsc_savings x : ℝ)

theorem Rahul_savings
  (h1 : total_savings = 180000)
  (h2 : ppf_savings = 72000)
  (h3 : nsc_savings = total_savings - ppf_savings)
  (h4 : x * nsc_savings = 0.5 * ppf_savings) :
  x = 1 / 3 :=
by
  -- Proof goes here
  sorry

end Rahul_savings_l108_108715


namespace left_square_side_length_l108_108755

theorem left_square_side_length (x : ℕ) (h1 : ∀ y : ℕ, y = x + 17)
                                (h2 : ∀ z : ℕ, z = x + 11)
                                (h3 : 3 * x + 28 = 52) : x = 8 :=
by
  sorry

end left_square_side_length_l108_108755


namespace polygon_sides_l108_108770

theorem polygon_sides (x : ℕ) (h : 180 * (x - 2) = 1080) : x = 8 :=
by sorry

end polygon_sides_l108_108770


namespace probability_same_number_l108_108866

theorem probability_same_number
  (Billy_num Bobbi_num : ℕ)
  (h_billy_range : Billy_num < 300)
  (h_bobbi_range : Bobbi_num < 300)
  (h_billy_multiple : Billy_num % 20 = 0)
  (h_bobbi_multiple : Bobbi_num % 30 = 0) :
  let total_combinations := 15 * 10 in
  let common_combinations := 5 in
  common_combinations / total_combinations = 1 / 30 :=
by
  sorry

end probability_same_number_l108_108866


namespace domain_of_sqrt_one_minus_ln_l108_108741

def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ Real.exp 1

theorem domain_of_sqrt_one_minus_ln (x : ℝ) : (1 - Real.log x ≥ 0) ∧ (x > 0) ↔ domain x := by
sorry

end domain_of_sqrt_one_minus_ln_l108_108741


namespace gcd_a_b_1_or_p_l108_108183

theorem gcd_a_b_1_or_p (a b p: ℤ) (h_gcd : Int.gcd a b = 1) (h_sum_ne_zero : a + b ≠ 0) (h_prime : Nat.Prime (Int.natAbs p)) (h_p_odd : p % 2 = 1) : 
  Int.gcd (a + b) ((a^p + b^p) / (a + b)) = 1 ∨ Int.gcd (a + b) ((a^p + b^p) / (a + b)) = p :=
by 
  sorry

end gcd_a_b_1_or_p_l108_108183


namespace find_smallest_d_l108_108145

theorem find_smallest_d (d : ℕ) : (5 + 6 + 2 + 4 + 8 + d) % 9 = 0 → d = 2 :=
by
  sorry

end find_smallest_d_l108_108145


namespace parabola_standard_equation_l108_108619

theorem parabola_standard_equation (h : ∀ y, y = 1/2) : ∃ c : ℝ, c = -2 ∧ (∀ x y, x^2 = c * y) :=
by
  -- Considering 'h' provides the condition for the directrix
  sorry

end parabola_standard_equation_l108_108619


namespace condition_necessary_but_not_sufficient_l108_108934

variable (m : ℝ)

/-- The problem statement and proof condition -/
theorem condition_necessary_but_not_sufficient :
  (∀ x : ℝ, |x - 2| + |x + 2| > m) → (∀ x : ℝ, x^2 + m * x + 4 > 0) :=
by {
  sorry
}

end condition_necessary_but_not_sufficient_l108_108934


namespace calculate_AP_squared_l108_108241

noncomputable theory
open Real

-- Definitions based on the problem statement.
def Triangle_Angle_A : ℝ := 45
def Triangle_Angle_C : ℝ := 60
def Side_AB : ℝ := 12

-- Main theorem to prove.
theorem calculate_AP_squared : 
  ∃ (AP : ℝ), AP^2 = 72 :=
by
  -- Condition: ∠A = 45° and ∠C = 60°
  have hA : Triangle_Angle_A = 45 := by rfl
  have hC : Triangle_Angle_C = 60 := by rfl
  have hAB : Side_AB = 12 := by rfl

  -- Further geometric properties and definitions skipped for proof.

  -- Final assertion that AP^2 = 72
  use 6 * sqrt 2
  simp [pow_two, mul_assoc, mul_comm, Real.sqrt_mul_self' (by norm_num : 2 > 0)]
  sorry

end calculate_AP_squared_l108_108241


namespace min_b_l108_108195

-- Definitions
def S (n : ℕ) : ℤ := 2^n - 1
def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2^(n-1)
def b (n : ℕ) : ℤ := (a n)^2 - 7 * (a n) + 6

-- Theorem
theorem min_b : ∃ n : ℕ, (b n = -6) :=
sorry

end min_b_l108_108195


namespace Q_2_plus_Q_neg2_l108_108678

variable {k : ℝ}

noncomputable def Q (x : ℝ) : ℝ := 0 -- Placeholder definition, real polynomial will be defined in proof.

theorem Q_2_plus_Q_neg2 (hQ0 : Q 0 = 2 * k)
  (hQ1 : Q 1 = 3 * k)
  (hQ_minus1 : Q (-1) = 4 * k) :
  Q 2 + Q (-2) = 16 * k :=
sorry

end Q_2_plus_Q_neg2_l108_108678


namespace find_n_l108_108237

-- Definitions of the conditions
variables (x n : ℝ)
variable (h1 : (x / 4) * n + 10 - 12 = 48)
variable (h2 : x = 40)

-- Theorem statement
theorem find_n (x n : ℝ) (h1 : (x / 4) * n + 10 - 12 = 48) (h2 : x = 40) : n = 5 :=
by
  sorry

end find_n_l108_108237


namespace num_digits_3_pow_30_l108_108216

theorem num_digits_3_pow_30 (log10_3 : ℝ) (h : log10_3 = 0.47712) : 
  let n : ℝ := 3^30 in 
  (⌊ log10 n ⌋ + 1) = 15 :=
by 
  sorry

end num_digits_3_pow_30_l108_108216


namespace min_sum_of_areas_l108_108608

axiom parabola_focus : (F : ℝ × ℝ) -> (p : ℝ × ℝ -> Prop)
axiom point_on_parabola : ∀ (A B : ℝ × ℝ), p A ∧ p B
axiom sides_of_x_axis : (A B : ℝ × ℝ) -> (A.snd * B.snd < 0)
axiom dot_product_condition : ∀ (A B : ℝ × ℝ), (A.fst * B.fst + A.snd * B.snd = 15)

theorem min_sum_of_areas (F : ℝ × ℝ) (A B O : ℝ × ℝ) (hF: parabola_focus F) (hAB: point_on_parabola A B) 
  (hx_axis: sides_of_x_axis A B) (hOA_OB: dot_product_condition A B):
  ∃ m : ℝ, m = (Real.sqrt 65) / 2 :=
by
  sorry

end min_sum_of_areas_l108_108608


namespace find_distance_k_l108_108900

/--
Equilateral $\triangle DEF$ has side length $300$. Points $R$ and $S$ lie outside the plane of $\triangle DEF$ and are on opposite sides of the plane. Furthermore, $RA=RB=RC$, and $SA=SB=SC$, and the planes containing $\triangle RDE$ and $\triangle SDE$ form a $150^{\circ}$ dihedral angle. There is a point $M$ whose distance from each of $D$, $E$, $F$, $R$, and $S$ is $k$.

We want to prove that $k = 300$.
-/
theorem find_distance_k : 
  ∃ (k : ℝ), 
    let side := 300,
        inradius := side * (Real.sqrt 3) / 6,
        circumradius := side * (Real.sqrt 3) / 3,
        G := sorry, -- circumcenter of triangle DEF
        R := sorry, -- lies at distance x from G
        S := sorry, -- lies at distance y from G
        M := sorry in -- M is midpoint of RS
    M.dist D = k ∧ M.dist E = k ∧ M.dist F = k ∧ M.dist R = k ∧ M.dist S = k ∧ k = 300 :=
  sorry

end find_distance_k_l108_108900


namespace largest_intersection_point_l108_108516

noncomputable theory

def polynomial (a : ℝ) : ℝ → ℝ := λ x, x^6 - 13 * x^5 + 42 * x^4 - 30 * x^3 + a * x^2
def line (c : ℝ) : ℝ → ℝ := λ x, 3 * x + c

-- Assume that the polynomial and the line intersect at three points
def intersects (a c x : ℝ) : Prop :=
  polynomial a x = line c x

theorem largest_intersection_point (a c : ℝ) (H1 : ∃ x1 x2 x3 : ℝ, 
  intersects a c x1 ∧ intersects a c x2 ∧ intersects a c x3 ∧ 
  polynomial a x1 < line c x1 ∧ polynomial a x2 < line c x2 ∧ polynomial a x3 < line c x3) : 
  ∃ x : ℝ, x = 4 :=
sorry

end largest_intersection_point_l108_108516


namespace number_of_benches_l108_108368

-- Define the conditions
def base5_to_base10 (n : ℕ) : ℕ := 3 * (5 ^ 2) + 1 * (5 ^ 1) + 0 * (5 ^ 0)

def people (n : ℕ) : ℕ := base5_to_base10 n

def people_per_bench : ℕ := 3

-- Define the goal
theorem number_of_benches (n : ℕ) (hn : n = 310) :
  (people n) / people_per_bench + if (people n) % people_per_bench ≠ 0 then 1 else 0 = 27 :=
sorry

end number_of_benches_l108_108368


namespace three_digit_tickets_into_50_boxes_impossible_3_digit_into_less_than_40_boxes_impossible_3_digit_into_less_than_50_boxes_four_digit_tickets_into_34_boxes_minimum_boxes_for_k_digit_tickets_l108_108407

-- Problem (1)
def tickets_3_digit := {n // 0 ≤ n ∧ n < 1000}
def boxes_2_digit := {n // 0 ≤ n ∧ n < 100}

-- Prove that it's possible to place all 3-digit tickets into 50 boxes
theorem three_digit_tickets_into_50_boxes (t : tickets_3_digit) : 
  ∃ b : boxes_2_digit → Prop, (∑ b, 1) = 50 :=
sorry

-- Prove that it's impossible to place all 3-digit tickets into fewer than 40 boxes
theorem impossible_3_digit_into_less_than_40_boxes (t : tickets_3_digit) : 
  ∀ b : boxes_2_digit → Prop, (∑ b, 1) < 40 → false :=
sorry

-- Prove that it is impossible to place all 3-digit tickets into fewer than 50 boxes
theorem impossible_3_digit_into_less_than_50_boxes (t : tickets_3_digit) : 
  ∀ b : boxes_2_digit → Prop, (∑ b, 1) < 50 → false :=
sorry

-- Problem (4)
def tickets_4_digit := {n // 0 ≤ n ∧ n < 10000}
def boxes_2_digit_from_4_digit_ticket := {n // 0 ≤ n ∧ n < 100}

-- Prove that it's possible to place all 4-digit tickets into 34 boxes
theorem four_digit_tickets_into_34_boxes (t : tickets_4_digit) : 
  ∃ b : boxes_2_digit_from_4_digit_ticket → Prop, (∑ b, 1) = 34 :=
sorry

-- Problem (5)
-- General theorem for k-digit tickets
def tickets_k_digit (k : ℕ) := {n // 0 ≤ n ∧ n < 10^k}

theorem minimum_boxes_for_k_digit_tickets (k : ℕ) : 
  ∃ b : {n // 0 ≤ n ∧ n < 10^((k - 1) * k)}, true := -- pseudo code for the formula
sorry

end three_digit_tickets_into_50_boxes_impossible_3_digit_into_less_than_40_boxes_impossible_3_digit_into_less_than_50_boxes_four_digit_tickets_into_34_boxes_minimum_boxes_for_k_digit_tickets_l108_108407


namespace initial_number_is_12_l108_108789

theorem initial_number_is_12 {x : ℤ} (h : ∃ k : ℤ, x + 17 = 29 * k) : x = 12 :=
by
  sorry

end initial_number_is_12_l108_108789


namespace sum_of_coefficients_g_l108_108462

theorem sum_of_coefficients_g :
  let α β γ : ℂ in 
  let f := 2 * X^3 + 7 * X^2 - 3 * X + 5 in
  -- Roots of the polynomial f
  let α + β + γ = -7 / 2,
  α * β + β * γ + γ * α = -3 / 2,
  α * β * γ = -5 / 2 ->
  let α_sq := α^2, β_sq := β^2, γ_sq := γ^2 in
  let g := Polynomial.of_coefficients [1, -(α_sq + β_sq + γ_sq), 
    (α_sq * β_sq + β_sq * γ_sq + γ_sq * α_sq), -(α_sq * β_sq * γ_sq)] in
  ∑ i in g.coefficients, i = -1427 / 40 
:= by
  sorry

end sum_of_coefficients_g_l108_108462


namespace sum_of_interior_angles_l108_108852

theorem sum_of_interior_angles (n : ℕ) (h_multiple_of_3 : n % 3 = 0) (h_geq_6 : n ≥ 6) : 
  ∑ i in (finset.range n), interior_angle i = 180 * (n - 4) :=
sorry

end sum_of_interior_angles_l108_108852


namespace probability_same_number_l108_108865

theorem probability_same_number
  (Billy_num Bobbi_num : ℕ)
  (h_billy_range : Billy_num < 300)
  (h_bobbi_range : Bobbi_num < 300)
  (h_billy_multiple : Billy_num % 20 = 0)
  (h_bobbi_multiple : Bobbi_num % 30 = 0) :
  let total_combinations := 15 * 10 in
  let common_combinations := 5 in
  common_combinations / total_combinations = 1 / 30 :=
by
  sorry

end probability_same_number_l108_108865


namespace vectors_projection_l108_108443

noncomputable def p := (⟨-44 / 53, 154 / 53⟩ : ℝ × ℝ)

theorem vectors_projection :
  let u := (⟨-4, 2⟩ : ℝ × ℝ)
  let v := (⟨3, 4⟩ : ℝ × ℝ)
  let w := (⟨7, 2⟩ : ℝ × ℝ)
  (⟨(7 * (24 / 53)) - 4, (2 * (24 / 53)) + 2⟩ : ℝ × ℝ) = p :=
by {
  -- proof skipped
  sorry
}

end vectors_projection_l108_108443


namespace AB_value_l108_108666

-- Variables for the lengths of the sides of the triangle
variables (AB AC BC : ℝ)
-- Angles of the triangle
variables (A B C : ℝ)

-- Conditions given in the problem
axiom angle_A_eq_90 : A = 90
axiom side_BC_eq_12 : BC = 12
axiom tan_C_eq_4cos_B : tan C = 4 * cos B

-- Pythagorean theorem in a right triangle
noncomputable def AB_from_Pythagorean : ℝ := Real.sqrt (BC^2 - AC^2)

-- Definition of what we need to prove
theorem AB_value : AB = 3 * Real.sqrt 15 :=
by
  -- Use the given conditions to compute AB
  have angle_A_is_right : ∠A = 90 := angle_A_eq_90
  have side_BC_is_12 : BC = 12 := side_BC_eq_12
  have tan_C_is_4cos_B : tan C = 4 * cos B := tan_C_eq_4cos_B

  -- Apply the given conditions and transformations
  sorry -- Detailed proof steps would go here

end AB_value_l108_108666


namespace area_of_region_l108_108394

theorem area_of_region (radius angle : ℝ) (a b c : ℝ) :
  radius = 3 ∧ angle = 90 ∧ (3 * (radius^2 * (angle / 360) * π - (sqrt 3 / 4 * (2 * radius)^2))) = a * sqrt b + c * π →
  a + b + c = 3 / 4 :=
by
  sorry

end area_of_region_l108_108394


namespace integer_k_count_l108_108570

theorem integer_k_count : 
  ∃ (kset : Set ℤ), kset.Card = 15 ∧ 
  ∀ k ∈ kset, ∃ m : ℤ, m^2 ≤ 200 ∧ √(200 - √k) = m := 
sorry

end integer_k_count_l108_108570


namespace angle_DEX_eq_42_l108_108448

noncomputable def regular_pentagon (A B C D E : Point) : Prop :=
  distance A B = distance B C ∧
  distance B C = distance C D ∧
  distance C D = distance D E ∧
  distance D E = distance E A ∧
  angle A B C = 108 ∧
  angle B C D = 108 ∧
  angle C D E = 108 ∧
  angle D E A = 108 ∧
  angle E A B = 108

noncomputable def circle (center : Point) (radius : ℝ) : Set Point :=
  { P | distance center P = radius }

noncomputable def intersection_point (A B : Point) (P : Point) : Prop :=
  P ∈ circle A (distance A B) ∧
  P ∈ circle B (distance B A)

variable {A B C D E X : Point}

theorem angle_DEX_eq_42 (h : regular_pentagon A B C D E) (hX : intersection_point A B X) :
  angle D E X = 42 := sorry

end angle_DEX_eq_42_l108_108448


namespace problem1_problem2_problem3_l108_108185

section Problems

variables {x a : ℝ} 
variables {f : ℝ → ℝ} (hx : x > 0 ∨ x < -1 / 4)

/-- Problem 1: Prove the solution set for f(x) > 0 when a = 5 is {x | x > 0 or x < -1 / 4}. -/
theorem problem1 (ha : a = 5) : 
  (f = λ x, log (2 : ℝ) (1 / x + a)) → 
  (f x > 0) ↔ hx := 
sorry

/-- Problem 2: Prove the range of a for which the equation f(x) - log_2([(a - 4) * x + 2a - 5]) = 0 has exactly one solution is a in (1, 2] union {3, 4}. -/
theorem problem2 :
  (f = λ x, log (2 : ℝ) (1 / x + a)) → 
  (∃! x, f x - log (2 : ℝ) ((a - 4) * x + 2 * a - 5) = 0) ↔ (1 < a ∧ a ≤ 2 ∨ a = 3 ∨ a = 4) := 
sorry

/-- Problem 3: Prove the range of a where the difference between maximum and minimum of f(x) on [t, t + 1] does not exceed 1 for any t in [1/2, 1] is a >= 2/3. -/
theorem problem3 (ht : 1 / 2 ≤ t ∧ t ≤ 1 ∧ a > 0) :
  (f = λ x, log (2 : ℝ) (1 / x + a)) → 
  (∀ t : ℝ, 1 / 2 ≤ t ∧ t ≤ 1 → max f (t + 1) - min f t ≤ 1) ↔ 
  (a ≥ 2 / 3) := 
sorry

end Problems

end problem1_problem2_problem3_l108_108185


namespace even_rows_count_in_pascals_triangle_l108_108331

theorem even_rows_count_in_pascals_triangle :
  (finset.filter (λ n, ∃ k, n = 2^k) (finset.range 51)).card = 5 :=
by
  sorry

end even_rows_count_in_pascals_triangle_l108_108331


namespace geometric_sequence_extreme_points_l108_108940

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

def f (x : ℝ) : ℝ :=
  x^3 / 3 - 5 * x^2 / 2 + 4 * x + 1

def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  deriv f x = 0

theorem geometric_sequence_extreme_points (a : ℕ → ℝ) (r : ℝ) :
  is_geometric_sequence a r →
  is_extreme_point f (a 1) →
  is_extreme_point f (a 5) →
  a 3 = 2 :=
by sorry

end geometric_sequence_extreme_points_l108_108940


namespace find_growth_rate_l108_108727

noncomputable def donation_first_day : ℝ := 10000
noncomputable def donation_third_day : ℝ := 12100
noncomputable def growth_rate (x : ℝ) : Prop :=
  (donation_first_day * (1 + x) ^ 2 = donation_third_day)

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.1 :=
by
  sorry

end find_growth_rate_l108_108727


namespace gcd_polynomial_l108_108942

open Nat

theorem gcd_polynomial (b : ℤ) (hb : 1632 ∣ b) : gcd (b^2 + 11 * b + 30) (b + 6) = 6 := by
  sorry

end gcd_polynomial_l108_108942


namespace range_x0_of_perpendicular_bisector_intersects_x_axis_l108_108597

open Real

theorem range_x0_of_perpendicular_bisector_intersects_x_axis
  (A B : ℝ × ℝ) 
  (hA : (A.1^2 / 9) + (A.2^2 / 8) = 1)
  (hB : (B.1^2 / 9) + (B.2^2 / 8) = 1)
  (N : ℝ × ℝ) 
  (P : ℝ × ℝ) 
  (hN : N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hP : P.2 = 0) 
  (hl : P.1 = N.1 + (8 * N.1) / (9 * N.2) * N.2)
  : -1/3 < P.1 ∧ P.1 < 1/3 :=
sorry

end range_x0_of_perpendicular_bisector_intersects_x_axis_l108_108597


namespace integers_for_square_fraction_l108_108916

theorem integers_for_square_fraction :
  {n : ℤ | 0 ≤ n ∧ n < 25 ∧ ∃ k : ℤ, k^2 = n / (25 - n) }.finite.card = 2 := 
sorry

end integers_for_square_fraction_l108_108916


namespace y_range_l108_108579

variable (a b : ℝ)
variable (h₀ : 0 < a) (h₁ : 0 < b)

theorem y_range (x : ℝ) (y : ℝ) (h₂ : y = (a * Real.sin x + b) / (a * Real.sin x - b)) : 
  y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b) :=
sorry

end y_range_l108_108579


namespace loan_period_l108_108641

variable (T : ℝ) -- period of the loan in years
variable (P : ℝ) := 3500 -- principal amount
variable (rA : ℝ) := 0.1 -- interest rate from A to B (10%)
variable (rC : ℝ) := 0.115 -- interest rate from B to C (11.5%)
variable (gain : ℝ) := 157.5 -- gain for B in Rs

theorem loan_period :
  let interest_paid_by_B_to_A := P * rA * T in
  let interest_received_by_B_from_C := P * rC * T in
  let B_gain := interest_received_by_B_from_C - interest_paid_by_B_to_A in
  B_gain = gain → T = 3 :=
by
  intros interest_paid_by_B_to_A interest_received_by_B_from_C B_gain h
  sorry

end loan_period_l108_108641


namespace hyperbola_ratio_l108_108627

theorem hyperbola_ratio (m n : ℝ) (hm : 0 < m) (hn : 0 < n) :
  let a := sqrt (1 / m), b := sqrt (1 / n), c := sqrt (m / (m * n) + n / (m * n))
  in 2 = c / a → m / n = 3 := 
by
  intros a b c h,
  sorry

end hyperbola_ratio_l108_108627


namespace sum_of_coefficients_of_shifted_parabola_l108_108442

theorem sum_of_coefficients_of_shifted_parabola :
  let f : ℝ → ℝ := λ x, 3 * x^2 + 2 * x + 5
  let g : ℝ → ℝ := λ x, f (x - 7)
  let a := (3 : ℝ)
  let b := (-40 : ℝ)
  let c := (138 : ℝ)
  a + b + c = 101 :=
by {
  sorry
}

end sum_of_coefficients_of_shifted_parabola_l108_108442


namespace ellipse_and_area_l108_108605

section EllipseProblem

-- Definitions according to given conditions
def is_vertex (p : ℝ × ℝ) := p.1 = -2 ∧ p.2 = 0
def ellipse_eq (a b : ℝ) (a_pos : a > b) (b_pos : b > 0) (p : ℝ × ℝ) :=
  (p.1)^2 / a^2 + (p.2)^2 / b^2 = 1

-- Correct answers as definitions in Lean
def ellipse_equation := ∀ (p : ℝ × ℝ), ellipse_eq 2 (√3) dec_trivial dec_trivial p
def area_triangle_range := ∀ A P Q : ℝ × ℝ, 
  is_vertex A ∧ 
  (A ≠ P ∧ A ≠ Q) ∧ 
  (P.1 = 1 ∧ Q.1 = 1 ∧ P ≠ Q) ∧ 
  |P.2 - Q.2| = 3 → 
  0 < sorry ∧ sorry ≤ 9 / 2

-- Theorem statement
theorem ellipse_and_area :
  ellipse_equation ∧ area_triangle_range :=
by { sorry }

end EllipseProblem

end ellipse_and_area_l108_108605


namespace max_months_with_5_sundays_l108_108080

theorem max_months_with_5_sundays (months : ℕ) (days_in_year : ℕ) (extra_sundays : ℕ) :
  months = 12 ∧ (days_in_year = 365 ∨ days_in_year = 366) ∧ extra_sundays = days_in_year % 7
  → ∃ max_months_with_5_sundays, max_months_with_5_sundays = 5 := 
by
  sorry

end max_months_with_5_sundays_l108_108080


namespace coefficient_x2_in_expansion_eq_80_l108_108664

theorem coefficient_x2_in_expansion_eq_80 (x : ℝ) :
  (∑ r in finset.range 6, (nat.choose 5 r * (2 : ℝ)^(5-r) * x^(5-3*r))) = 80 * x^2 :=
by
  sorry

end coefficient_x2_in_expansion_eq_80_l108_108664


namespace sum_of_satisfying_ns_l108_108436

-- Defining the given condition
def satisfies_condition (n : ℕ) : Prop :=
  Nat.lcm n 120 = Nat.gcd n 120 + 300

-- The proof statement
theorem sum_of_satisfying_ns : 
  (∑ n in Finset.filter satisfies_condition (Finset.range 1000), n) = 180 :=
sorry

end sum_of_satisfying_ns_l108_108436


namespace range_of_a_l108_108646

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ (a ≤ 0 ∨ a ≥ 4) :=
by {
  sorry
}

end range_of_a_l108_108646


namespace relationship_M_N_l108_108317

-- Define the sets M and N based on the conditions
def M : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def N : Set ℕ := {x | ∃ n : ℕ, x = 3 * n}

-- The statement to be proved
theorem relationship_M_N : ¬ (M ⊆ N) ∧ ¬ (N ⊆ M) :=
by
  sorry

end relationship_M_N_l108_108317


namespace proof_problem_l108_108653

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (2 + (Real.sqrt 2) / 2 * t, (Real.sqrt 2) / 2 * t)

noncomputable def polar_ellipse (θ : ℝ) : ℝ :=
  Real.sqrt (12 / (3 * Real.cos θ ^ 2 + 4 * Real.sin θ ^ 2))

noncomputable def cartesian_line : ℝ → Prop :=
  λ y, ∃ x : ℝ, y = x - 2

noncomputable def cartesian_ellipse : ℝ → ℝ → Prop :=
  λ x y, x^2 / 4 + y^2 / 3 = 1

noncomputable def distance_to_line (x0 y0 A B C : ℝ) : ℝ :=
  Real.abs (A * x0 + B * y0 + C) / Real.sqrt (A^2 + B^2)

theorem proof_problem :
  (∀ t, let (x, y) := parametric_line t in y = x - 2) ∧
  (∀ (x y : ℝ), polar_ellipse (Real.arctan2 y x) = Real.sqrt (x^2 + y^2) → x^2 / 4 + y^2 / 3 = 1) ∧
  (distance_to_line (-1) 0 (-1) 1 (-2) + distance_to_line 1 0 (-1) 1 (-2) = 2 * Real.sqrt 2) :=
by
  sorry

end proof_problem_l108_108653


namespace find_VD_l108_108256

-- Define the basic geometric entities and their properties.
variables (EF GH FM UV MH VN R : ℝ)
variables (E F G H M U V N : Point)
variables (EMH R E_passes_N : Line)

-- Define the conditions provided in the problem.
def rectangle (E F G H : Point) : Prop := 
  -- Add the properties of the rectangle
  sorry

def angle_EMH_90 (E M H : Point) : Prop := 
  ∠ E M H = 90

def perpendicular_UV_FG (U V : Point) : Prop := 
  line_perpendicular (line_through U V) (line_through F G)

def segment_equal (a b : Point) (l m : ℝ) : Prop := 
  distance a b = l ∧ distance a b = m

def intersects (l1 l2 : Line) (P : Point) : Prop := 
  lies_on P l1 ∧ lies_on P l2

def passes_through (R E : Point) (N : Point) : Prop :=
  lies_on N (line_through R E)

def triangle_sides (M N E : Point) : Prop :=
  distance M E = 30 ∧ distance E N = 35 ∧ distance M N = 25

-- Using the definitions above, state the theorem to prove VD.
theorem find_VD : 
  ∀ (EF GH FM UV MH VN R : ℝ) (E F G H M U V N : Point),
    rectangle E F G H →
    angle_EMH_90 E M H →
    perpendicular_UV_FG U V →
    segment_equal F M U V →
    intersects (line_through M H) (line_through U V) N →
    passes_through R E N →
    triangle_sides M N E →
    VD = 168 / 35 :=
sorry

end find_VD_l108_108256


namespace calculate_gross_profit_l108_108059

theorem calculate_gross_profit (sales_price : ℝ) (cost : ℝ) (gross_profit : ℝ) 
    (h1 : sales_price = 81)
    (h2 : gross_profit = 1.70 * cost)
    (h3 : sales_price = cost + gross_profit) : gross_profit = 51 :=
by
  sorry

end calculate_gross_profit_l108_108059


namespace minimize_y_l108_108625

theorem minimize_y (a b : ℝ) : 
  ∃ x : ℝ, x = (3 * a + b) / 4 ∧ 
  ∀ y : ℝ, (3 * (y - a) ^ 2 + (y - b) ^ 2) ≥ (3 * ((3 * a + b) / 4 - a) ^ 2 + ((3 * a + b) / 4 - b) ^ 2) :=
sorry

end minimize_y_l108_108625


namespace cannot_be_parallel_l108_108609

-- Defining skew lines
def skew_lines (a b : ℝ^3 → ℝ^3 → Prop) : Prop :=
  ¬ ∃ (P : ℝ^3), (a P ∧ b P)

-- Given conditions:
variables (a b c : ℝ^3 → ℝ^3 → Prop)
variables (ha : skew_lines a b) (hc_parallel_a : ∀ (x y : ℝ^3), c x y ↔ a x y)

-- Theorem statement
theorem cannot_be_parallel (hb_parallel_c : ∀ (x y : ℝ^3), b x y ↔ c x y) : false :=
  sorry

end cannot_be_parallel_l108_108609


namespace find_multiplicand_l108_108913

theorem find_multiplicand (m : ℕ) 
( h : 32519 * m = 325027405 ) : 
m = 9995 := 
by {
  sorry
}

end find_multiplicand_l108_108913


namespace hexagon_area_l108_108304

theorem hexagon_area (b : ℝ) (A B C D E F : ℝ × ℝ) 
  (hA : A = (0, 0))
  (hB : B = (b, 3))
  (hHexagon : ∀ P ∈ {A, B, C, D, E, F}, P.2 ∈ {0, 3, 6, 9, 12, 15})
  (hAngleFAB : ∠ (F - A) (B - A) = 150)
  (hParallel1 : AB ∥ DE)
  (hParallel2 : BC ∥ EF)
  (hParallel3 : CD ∥ FA) :
  ∃ m n : ℕ, is_coprime n (n.sqrt).floor ∧ hexagon_area ABCDEF = m * (n.sqrt) ∧ (m + n = 141) := 
sorry

end hexagon_area_l108_108304


namespace probability_of_5_weed_seeds_l108_108851

open Real

-- Definitions of the conditions
def proportion_of_weed_seeds := 0.004
def number_of_seeds_selected := 5000
def number_of_weed_seeds_to_find := 5

-- Poisson parameter
def lambda := number_of_seeds_selected * proportion_of_weed_seeds

-- Poisson probability mass function
def P (k : ℕ) (λ : ℝ) : ℝ := (λ^k * exp (-λ)) / k.factorial

-- The expected probability of finding exactly 5 weed seeds
def expected_probability := 0.000055

-- The theorem to be proven
theorem probability_of_5_weed_seeds :
  abs (P number_of_weed_seeds_to_find lambda - expected_probability) < 0.000001 :=
by sorry

end probability_of_5_weed_seeds_l108_108851


namespace real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l108_108160

variables {m : ℝ}

def complex_expression (m : ℝ) : ℂ := (1 + complex.i) * complex.ofReal(m^2) - m * (5 + 3 * complex.i) + 6

-- Theorem 1
theorem real_number_condition : (complex_expression m).im = 0 ↔ m = 0 ∨ m = 3 := 
sorry

-- Theorem 2
theorem imaginary_number_condition : (complex_expression m).re = 0 ∧ (complex_expression m).im ≠ 0 ↔ m ≠ 0 ∧ m ≠ 3 := 
sorry

-- Theorem 3
theorem pure_imaginary_number_condition : (complex_expression m).re = 0 ∧ (complex_expression m).im ≠ 0 ∧ m^2 - 5 * m + 6 = 0 ↔ m = 2 := 
sorry

end real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l108_108160


namespace smallest_digit_to_correct_sum_l108_108382

theorem smallest_digit_to_correct_sum : 
  (∑ i in [356, 781, 492], i) = 1629 →  -- Condition 1: Sum of numbers is 1629
  ∃ d, d < 3 ∧ (356 - 100 = 256) →     -- The smallest digit d that can change 356 to 256
  (∑ i in [256, 781, 492], i = 1529) →  -- The corrected sum is 1529
  d = 3 :=
sorry

end smallest_digit_to_correct_sum_l108_108382


namespace betty_total_cost_l108_108107

theorem betty_total_cost :
    (6 * 2.5) + (4 * 1.25) + (8 * 3) = 44 :=
by
    sorry

end betty_total_cost_l108_108107


namespace smallest_x_l108_108441

theorem smallest_x (x : ℕ) : 
  (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 9 = 8) ↔ x = 314 := 
by
  sorry

end smallest_x_l108_108441


namespace six_points_within_circle_l108_108355

/-- If six points are placed inside or on a circle with radius 1, then 
there always exist at least two points such that the distance between 
them is at most 1. -/
theorem six_points_within_circle : ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (points i).1^2 + (points i).2^2 ≤ 1) → 
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 1 :=
by
  -- Condition: Circle of radius 1
  intro points h_points
  sorry

end six_points_within_circle_l108_108355


namespace find_number_of_3cm_books_l108_108070

-- Define the conditions
def total_books : ℕ := 46
def total_thickness : ℕ := 200
def thickness_3cm : ℕ := 3
def thickness_5cm : ℕ := 5

-- Let x be the number of 3 cm thick books, y be the number of 5 cm thick books
variable (x y : ℕ)

-- Define the system of equations based on the given conditions
axiom total_books_eq : x + y = total_books
axiom total_thickness_eq : thickness_3cm * x + thickness_5cm * y = total_thickness

-- The theorem to prove: x = 15
theorem find_number_of_3cm_books : x = 15 :=
by
  sorry

end find_number_of_3cm_books_l108_108070


namespace heptagon_obtuse_angles_lower_bound_l108_108478

theorem heptagon_obtuse_angles_lower_bound :
  ∀ (a : ℕ → ℝ),
    (∀ i, 1 ≤ i ∧ i ≤ 7 → 90 < a i ∧ a i < 180) →
    (∑ i in finset.range 7, a i = 900) →
    (7 ≤ finset.card (finset.filter (λ i, 90 < a i ∧ a i <= 180) finset.univ)) ∧
    ((finset.card (finset.filter (λ i, 90 < a i ∧ a i <= 180) finset.univ) > 4) ∨ 
    (finset.card (finset.filter (λ i, 90 < a i ∧ a i <= 180) finset.univ) = 4)) :=
by
  sorry

end heptagon_obtuse_angles_lower_bound_l108_108478


namespace dodecahedron_greater_volume_than_icosahedron_l108_108314

noncomputable def volume_of_dodecahedron_gt_icosahedron (R : ℝ) : Prop :=
  ∀ (D I : Type), 
  is_regular_dodecahedron D →
  is_regular_icosahedron I →
  is_inscribed_in_sphere D R →
  is_inscribed_in_sphere I R →
  volume D > volume I

/--
  A regular dodecahedron inscribed in a sphere of radius R has a greater volume than
  a regular icosahedron inscribed in a sphere of the same radius.
--/
theorem dodecahedron_greater_volume_than_icosahedron (R : ℝ) :
  volume_of_dodecahedron_gt_icosahedron R := 
sorry

end dodecahedron_greater_volume_than_icosahedron_l108_108314


namespace alternating_series_converges_to_five_l108_108525

noncomputable def alternating_series : ℝ := 3 + 5 / (4 + 5 / (3 + 5 / (4 + 5 / (3 + 5 / ...))))

theorem alternating_series_converges_to_five :
  alternating_series = 5 :=
sorry

end alternating_series_converges_to_five_l108_108525


namespace ultraSquarishCount_l108_108098

def isTwoDigitPerfectSquare (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ m, m * m = n

def isUltraSquarish (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000 ∧ 
  (∀ d in List.range 8, (n / 10^d % 10) ≠ 0) ∧  -- None of the digits are zero
  ∃ z, z * z = n ∧  -- n is a perfect square
    isTwoDigitPerfectSquare (n / 1000000 % 100) ∧
    isTwoDigitPerfectSquare (n / 10000 % 100) ∧
    isTwoDigitPerfectSquare (n / 100 % 100) ∧
    isTwoDigitPerfectSquare (n % 100)

theorem ultraSquarishCount : 
  ∃ count, count = (Finset.range' 100000000 10000000).filter isUltraSquarish |>.card ∧ count = 0 := 
by {
  sorry
}

end ultraSquarishCount_l108_108098


namespace probability_overlap_l108_108733

/-- The cafeteria is open from noon until 2 PM every Monday for lunch.
Two professors eat 15-minute lunches sometime between noon and 2 PM.
Prove that the probability that they are in the cafeteria simultaneously on any given Monday is 2/7. -/
theorem probability_overlap (t1 t2 : ℕ) (t1_range : t1 ∈ set.Ico 0 106) (t2_range : t2 ∈ set.Ico 0 106) :
  (∃ t1 t2, t1 ∈ set.Ico 0 106 ∧ t2 ∈ set.Ico 0 106 ∧ |t1 - t2| < 15) →
  (105 * 30) / (105 * 105) = (2 : ℚ) / 7 :=
sorry

end probability_overlap_l108_108733


namespace bcm_hens_count_l108_108472

-- Propositions representing the given conditions
def total_chickens : ℕ := 100
def bcm_ratio : ℝ := 0.20
def bcm_hens_ratio : ℝ := 0.80

-- Theorem statement: proving the number of BCM hens
theorem bcm_hens_count : (total_chickens * bcm_ratio * bcm_hens_ratio = 16) := by
  sorry

end bcm_hens_count_l108_108472


namespace total_points_l108_108889

noncomputable def Darius_points : ℕ := 10
noncomputable def Marius_points : ℕ := Darius_points + 3
noncomputable def Matt_points : ℕ := Darius_points + 5
noncomputable def Sofia_points : ℕ := 2 * Matt_points

theorem total_points : Darius_points + Marius_points + Matt_points + Sofia_points = 68 :=
by
  -- Definitions are directly from the problem statement, proof skipped 
  sorry

end total_points_l108_108889


namespace sum_of_interior_angles_of_pentagon_l108_108398

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  sorry

end sum_of_interior_angles_of_pentagon_l108_108398


namespace total_ways_to_distribute_stickers_l108_108983

-- Definitions based on conditions
def total_stickers : ℕ := 9
def S (x y z : ℕ) : ℕ := total_stickers - x - y - z
def valid_distribution (x y z : ℕ) : Prop := 
  x + y + z ≤ 9 ∧ S(x, y, z) ≥ 2 * min x (min y z)

-- The proof problem statement
theorem total_ways_to_distribute_stickers :
  ∃ n : ℕ, n = 13 ∧ 
  complete (λ (x y z : ℕ), valid_distribution x y z -> True) sorry

end total_ways_to_distribute_stickers_l108_108983


namespace real_solutions_count_l108_108158

theorem real_solutions_count : 
  ∃! (x : ℝ), real.sqrt (-4 * (x - 3) ^ 2) ∈ ℝ := sorry

end real_solutions_count_l108_108158


namespace geom_inequality_l108_108691

noncomputable def midpoint (B T : Point) (B1 : Point) : Prop := dist B B1 = dist B1 T

noncomputable def extension (A T H : Point) (B1 : Point) : Prop := dist T H = dist T B1

noncomputable def angle_properties (A T B1 H : Point) : Prop := 
  ∠ T H B1 = 60 ∧ ∠ T B1 H = 60 ∧ dist H B1 = dist B1 T ∧ dist B1 T = dist B1 B

noncomputable def inequality (A B C T : Point) : Prop :=
  2 * dist A B + 2 * dist B C + 2 * dist C A > 4 * dist A T + 3 * dist B T + 2 * dist C T

theorem geom_inequality 
  (A B C T B1 H: Point)
  (h_midpoint: midpoint B T B1)
  (h_extension: extension A T H B1)
  (h_angle_prop: angle_properties A T B1 H) : 
  inequality A B C T := 
by sorry

end geom_inequality_l108_108691


namespace probability_of_second_ball_white_is_correct_l108_108468

-- Definitions based on the conditions
def initial_white_balls : ℕ := 8
def initial_black_balls : ℕ := 7
def total_initial_balls : ℕ := initial_white_balls + initial_black_balls
def white_balls_after_first_draw : ℕ := initial_white_balls
def black_balls_after_first_draw : ℕ := initial_black_balls - 1
def total_balls_after_first_draw : ℕ := white_balls_after_first_draw + black_balls_after_first_draw
def probability_second_ball_white : ℚ := white_balls_after_first_draw / total_balls_after_first_draw

-- The proof problem
theorem probability_of_second_ball_white_is_correct :
  probability_second_ball_white = 4 / 7 :=
by
  sorry

end probability_of_second_ball_white_is_correct_l108_108468


namespace translation_teams_selection_l108_108827

theorem translation_teams_selection :
  let translators := 11
  let english := 5
  let japanese := 4
  let both := 2
  let total_selected := 8
  let english_team := 4
  let japanese_team := 4
  in ∃ (schemes : ℕ), schemes = 144 :=
by
  simp only [translators, english, japanese, both, total_selected, english_team, japanese_team]
  use 144
  sorry

end translation_teams_selection_l108_108827


namespace quadrilateral_property_l108_108590

open EuclideanGeometry

noncomputable def cyclic_quadrilateral (A B C D : Point) :=
  cyclic_order A B C D

theorem quadrilateral_property
  {A B C D E : Point}
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : ∠ A = 2 * ∠ B)
  (h3 : angle_bisector_intersect E C (Line A B)) :
  (distance A D + distance A E = distance B E) :=
  sorry

end quadrilateral_property_l108_108590


namespace regression_residual_l108_108210

theorem regression_residual :
  ∀ (x y : ℝ), x = 165 → y = 57 →
  let y_hat := 0.85 * x - 85.7 in
  y - y_hat = 2.45 :=
by
  intros x y hx hy
  simp [hx, hy]
  sorry

end regression_residual_l108_108210


namespace average_score_of_girls_is_84_l108_108651

noncomputable def average_girls_score (num_girls num_boys avg_score_class avg_score_boys avg_score_girls : ℝ) : Prop :=
  num_boys = 1.8 * num_girls ∧
  avg_score_class = 75 ∧
  avg_score_girls = 1.2 * avg_score_boys ∧
  num_girls * avg_score_girls + num_boys * avg_score_boys = (num_girls + num_boys) * avg_score_class

theorem average_score_of_girls_is_84 :
  ∀ (num_girls num_boys avg_score_class avg_score_boys avg_score_girls : ℝ),
  average_girls_score num_girls num_boys avg_score_class avg_score_boys avg_score_girls →
  avg_score_girls = 84 :=
by
  intros _ _ _ _ _
  intro h
  unfold average_girls_score at h
  cases h with h1 h12
  cases h12 with h2 h3
  cases h3 with h4 h_sum
  sorry

end average_score_of_girls_is_84_l108_108651


namespace trig_identity_l108_108460

theorem trig_identity :
  cos 15 * cos 30 - sin 15 * sin 150 = (sqrt 2) / 2 :=
by sorry

end trig_identity_l108_108460


namespace sufficient_not_necessary_a_eq_one_l108_108444

noncomputable def f (a x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + a^2) - x)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x + f (-x) = 0

theorem sufficient_not_necessary_a_eq_one 
  (a : ℝ) 
  (h₁ : a = 1) 
  : is_odd_function (f a) := sorry

end sufficient_not_necessary_a_eq_one_l108_108444


namespace corrected_mean_l108_108752

theorem corrected_mean (n : ℕ) (incorrect_mean old_obs new_obs : ℚ) 
  (hn : n = 50) (h_mean : incorrect_mean = 40) (hold : old_obs = 15) (hnew : new_obs = 45) :
  ((n * incorrect_mean + (new_obs - old_obs)) / n) = 40.6 :=
by
  sorry

end corrected_mean_l108_108752


namespace preimage_of_2_neg2_l108_108950

-- Define the mapping function
def f : ℝ × ℝ → ℝ × ℝ :=
  λ p, (p.1 + p.2, p.1^2 - p.2)

-- Define the proposition to be proved
theorem preimage_of_2_neg2 :
  ∃ x y : ℝ, f (x, y) = (2, -2) ∧ x ≥ 0 ∧ (x, y) = (0, 2) :=
by
  -- To be proved
  sorry

end preimage_of_2_neg2_l108_108950


namespace find_correct_four_digit_number_l108_108801

theorem find_correct_four_digit_number (N : ℕ) (misspelledN : ℕ) (misspelled_unit_digit_correction : ℕ) 
  (h1 : misspelledN = (N / 10) * 10 + 6)
  (h2 : N - misspelled_unit_digit_correction = (N / 10) * 10 - 7 + 9)
  (h3 : misspelledN - 57 = 1819) : N = 1879 :=
  sorry


end find_correct_four_digit_number_l108_108801


namespace find_k_for_collinear_l108_108566

-- Definition of the points on a Cartesian plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Function to calculate the slope between two points
def slope (p1 p2 : Point) : ℝ :=
  if p1.x = p2.x then 0 else (p2.y - p1.y) / (p2.x - p1.x)

-- Definition of colinearity based on consistent slopes
def collinear (p1 p2 p3 : Point) : Prop :=
  slope p1 p2 = slope p2 p3

theorem find_k_for_collinear (k : ℝ) :
  collinear ⟨1, k / 3⟩ ⟨3, 1⟩ ⟨6, k / 2⟩ ↔ k = 5 / 2 :=
sorry

end find_k_for_collinear_l108_108566


namespace minimum_value_of_sum_l108_108461

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1/a + 2/b + 3/c = 2) : a + 2*b + 3*c = 18 ↔ (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end minimum_value_of_sum_l108_108461


namespace probability_at_least_one_spade_or_ace_l108_108073

open ProbabilityTheory

-- Definitions of the problem conditions
def card_deck : Finset ℕ := Finset.range 54 -- 54 cards represented as numbers from 0 to 53
def spades_or_aces : Finset ℕ := (Finset.range 13) ∪ (Finset.range 13).erase 0 ∪ (Finset.range 13).erase 1 ∪ (Finset.range 13).erase 2

theorem probability_at_least_one_spade_or_ace : 
  (1 - ((54 - 16) / 54) ^ 2) = (368 / 729) := 
by
  sorry
  -- Proof not required

end probability_at_least_one_spade_or_ace_l108_108073


namespace NumberOfTruePropositionsIsOne_l108_108594

-- Define the plane α and lines m and n
variables (α : Type) [plane α] (m n : line α)

-- Define the conditions for each proposition
def prop1 (m n : line α) [parallel m n] : Prop :=
  ∀ θ₁ θ₂, θ₁ = angle_of_line_and_plane m α ∧ θ₂ = angle_of_line_and_plane n α → θ₁ = θ₂

def prop2 (m n : line α) [parallel m α] [parallel n α] : Prop :=
  parallel m n

def prop3 (m n : line α) [perpendicular m α] [perpendicular m n] : Prop :=
  parallel n α

def prop4 (m n : line α) [skew m n] [parallel m α] : Prop :=
  ∃ p, p ∈ n ∧ p ∈ α

-- Number of true propositions among prop1 to prop4 is 1
theorem NumberOfTruePropositionsIsOne : 
  (prop1 m n) ∧ ¬ (prop2 m n) ∧ ¬ (prop3 m n) ∧ ¬ (prop4 m n) → true_props = 1 :=
sorry

end NumberOfTruePropositionsIsOne_l108_108594


namespace solve_problem_l108_108977

open Real

variable (a b : ℕ → ℚ) -- defining the arithmetic sequences a_n and b_n
variable (S T : ℕ → ℚ) -- defining the sums of the first n terms of the sequences a_n and b_n, respectively

-- Condition: S_n = sum of the first n terms of a
axiom hS : ∀ n : ℕ, S n = ∑ i in Finset.range (n+1), a i
-- Condition: T_n = sum of the first n terms of b
axiom hT : ∀ n : ℕ, T n = ∑ i in Finset.range (n+1), b i

-- Condition: For any positive integer n, Sₙ / Tₙ = (3n + 5) / (2n + 3)
axiom hRatio : ∀ n : ℕ, 0 < n → S n / T n = (3 * n + 5) / (2 * n + 3)

-- Theorem to prove: a₇ / b₇ = 44 / 29
theorem solve_problem : (a 7 / b 7) = 44 / 29 := sorry

end solve_problem_l108_108977


namespace probability_same_number_l108_108864

theorem probability_same_number
  (Billy_num Bobbi_num : ℕ)
  (h_billy_range : Billy_num < 300)
  (h_bobbi_range : Bobbi_num < 300)
  (h_billy_multiple : Billy_num % 20 = 0)
  (h_bobbi_multiple : Bobbi_num % 30 = 0) :
  let total_combinations := 15 * 10 in
  let common_combinations := 5 in
  common_combinations / total_combinations = 1 / 30 :=
by
  sorry

end probability_same_number_l108_108864


namespace smallest_int_greater_than_sqrt_6_l108_108765

theorem smallest_int_greater_than_sqrt_6 : ∃ n : ℤ, (n > nat.floor (Real.sqrt 6)) ∧ (n = 3) := by
  sorry

end smallest_int_greater_than_sqrt_6_l108_108765


namespace ratio_PR_QS_l108_108708

/-- Given points P, Q, R, and S on a straight line in that order with
    distances PQ = 3 units, QR = 7 units, and PS = 20 units,
    the ratio of PR to QS is 1. -/
theorem ratio_PR_QS (P Q R S : ℝ) (PQ QR PS : ℝ) (hPQ : PQ = 3) (hQR : QR = 7) (hPS : PS = 20) :
  let PR := PQ + QR
  let QS := PS - PQ - QR
  PR / QS = 1 :=
by
  -- Definitions from conditions
  let PR := PQ + QR
  let QS := PS - PQ - QR
  -- Proof not required, hence sorry
  sorry

end ratio_PR_QS_l108_108708


namespace time_for_trains_to_pass_l108_108425

-- Definitions for given conditions
def trainA_length : ℝ := 120
def trainA_speed_kmh : ℝ := 40
def trainB_length : ℝ := 180
def trainB_speed_kmh : ℝ := 60
def bridge_length : ℝ := 160

-- Helper functions to convert speeds from km/h to m/s
def convert_kmh_to_ms (speed_kmh : ℝ) : ℝ := (speed_kmh * 1000) / 3600

-- Converted speeds of trains in m/s
def trainA_speed_ms : ℝ := convert_kmh_to_ms trainA_speed_kmh
def trainB_speed_ms : ℝ := convert_kmh_to_ms trainB_speed_kmh

-- Total distance to be covered (lengths of both trains and the bridge)
def total_distance : ℝ := trainA_length + trainB_length + bridge_length

-- Relative speed of trains approaching each other
def relative_speed : ℝ := trainA_speed_ms + trainB_speed_ms

-- Time calculation
def time_to_pass : ℝ := total_distance / relative_speed

-- Proof problem statement
theorem time_for_trains_to_pass : time_to_pass ≈ 16.55 := by
  sorry

end time_for_trains_to_pass_l108_108425


namespace bag_ball_color_distribution_l108_108253

theorem bag_ball_color_distribution :
  ∀ (white red yellow green : Type),
  (∀ (white_labeled red_labeled yellow_labeled green_labeled: white),
  (white_labeled ≠ white) →
  (red_labeled ≠ red) →
  (yellow_labeled ≠ yellow) →
  (green_labeled ≠ green) →
  ∃ (contains_white contains_red contains_yellow contains_green: white),
  contains_white = red ∧
  contains_red = white ∧
  contains_yellow = green ∧
  contains_green = yellow) :=
by { 
  intros,
  sorry
}

end bag_ball_color_distribution_l108_108253


namespace find_ABC_constants_l108_108553

theorem find_ABC_constants : ∃ (A B C : ℚ), 
  (∀ x : ℚ, x ≠ 4 → x ≠ 2 →
  (5 * x + 1) / ((x - 4) * (x - 2) ^ 3) = A / (x - 4) + B / (x - 2) + C / (x - 2) ^ 3) ∧
  A = 21 / 8 ∧ B = 19 / 4 ∧ C = -11 / 2 :=
begin
  sorry
end

end find_ABC_constants_l108_108553


namespace tetrahedron_face_area_squared_l108_108768

variables {S0 S1 S2 S3 α12 α13 α23 : ℝ}

-- State the theorem
theorem tetrahedron_face_area_squared :
  (S0)^2 = (S1)^2 + (S2)^2 + (S3)^2 - 2 * S1 * S2 * (Real.cos α12) - 2 * S1 * S3 * (Real.cos α13) - 2 * S2 * S3 * (Real.cos α23) :=
sorry

end tetrahedron_face_area_squared_l108_108768


namespace sum_of_interior_angles_of_pentagon_l108_108405

theorem sum_of_interior_angles_of_pentagon :
  let n := 5 in (n - 2) * 180 = 540 := 
by 
  let n := 5
  show (n - 2) * 180 = 540
  sorry

end sum_of_interior_angles_of_pentagon_l108_108405


namespace spiral_stripe_length_l108_108469

noncomputable def can : Type :=
{ 
  circumference : ℝ // circumference = 18
  height : ℝ // height = 8 
  wraps : ℕ // wraps = 3 
}

noncomputable def stripe_length (c : can) : ℝ :=
  real.sqrt ((3 * c.circumference) ^ 2 + c.height ^ 2)

theorem spiral_stripe_length (c : can) : stripe_length c = real.sqrt 2980 :=
by 
  sorry

end spiral_stripe_length_l108_108469


namespace game_points_product_l108_108093

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2, 6]
def betty_rolls : List ℕ := [6, 3, 3, 2, 1]

def calculate_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_points_product :
  calculate_points allie_rolls * calculate_points betty_rolls = 702 :=
by
  sorry

end game_points_product_l108_108093


namespace impossible_to_color_all_red_l108_108124

def grid_points : set (ℤ × ℤ) := {(x, y) | 0 <= x ∧ x <= 3 ∧ 0 <= y ∧ y <= 3}

def initial_coloring (p : ℤ × ℤ) : Prop :=
  if p = (0, 1) then true else false

def valid_move (pts : set (ℤ × ℤ)) : Prop :=
  ∃ (l : set (ℤ × ℤ)), (∀ p ∈ l, p ∈ pts) ∧ (
    (∃ x, l = {(x, y) | 0 <= y ∧ y <= 3}) ∨
    (∃ y, l = {(x, y) | 0 <= x ∧ x <= 3}) ∨
    (∃ d, l = {(x, y) | x - y = d}) ∨
    (∃ d, l = {(x, y) | x + y = d})
  )

theorem impossible_to_color_all_red :
  ¬ ∃ (coloring : ℤ × ℤ → Prop),
    (∀ p ∈ grid_points, ¬initial_coloring p → coloring p = false) ∧
    (∀ mv ∈ grid_points, valid_move mv → ¬ ∃ p ∈ mv, coloring p = true) ∧
    (∀ p ∈ grid_points, ¬ coloring p) :=
sorry

end impossible_to_color_all_red_l108_108124


namespace remainder_9_5_4_6_5_7_mod_7_l108_108792

theorem remainder_9_5_4_6_5_7_mod_7 :
  ((9^5 + 4^6 + 5^7) % 7) = 2 :=
by sorry

end remainder_9_5_4_6_5_7_mod_7_l108_108792


namespace find_lambda_l108_108917

-- Condition 1: |omega| = 3
def omega (ω : ℂ) : Prop := abs ω = 3

-- Condition 2: lambda > 1
def lambda (λ : ℝ) : Prop := λ > 1

-- Condition 3: ω, ω^2, λω form an equilateral triangle
def equilateral (ω : ℂ) (λ : ℝ) : Prop :=
  let v := [ω, ω^2, λ * ω] in
  (dist v[0] v[1] = dist v[1] v[2]) ∧ (dist v[1] v[2] = dist v[2] v[0])

-- The statement to prove: lambda = 1, given the conditions
theorem find_lambda (ω : ℂ) (λ : ℝ) (h1 : omega ω) (h2 : lambda λ) (h3 : equilateral ω λ) : λ = 1 := sorry

end find_lambda_l108_108917


namespace modular_inverse_3_mod_23_l108_108909

theorem modular_inverse_3_mod_23 : ∃ (a : ℤ), 0 ≤ a ∧ a ≤ 22 ∧ (3 * a) % 23 = 1 :=
by {
  use 8,
  split, exact nat.zero_le 8, split, exact nat.le_of_lt (by norm_num),
  show (3 * 8) % 23 = 1, exact int.mod_eq_of_lt int.zero_lt_three (by norm_num),
  norm_num,
  sorry
}

end modular_inverse_3_mod_23_l108_108909


namespace john_must_study_4_5_hours_l108_108671

-- Let "study_time" be the amount of time John needs to study for the second exam.

noncomputable def study_time_for_avg_score (hours1 score1 target_avg total_exams : ℝ) (direct_relation : Prop) :=
  2 * target_avg - score1 / (score1 / hours1)

theorem john_must_study_4_5_hours :
  study_time_for_avg_score 3 60 75 2 (60 / 3 = 90 / study_time_for_avg_score 3 60 75 2 (60 / 3 = 90 / study_time_for_avg_score 3 60 75 2 (sorry))) = 4.5 :=
sorry

end john_must_study_4_5_hours_l108_108671


namespace right_triangle_condition_l108_108049

theorem right_triangle_condition (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : a^2 + b^2 = c^2 :=
by sorry

end right_triangle_condition_l108_108049


namespace Total_marbles_equal_231_l108_108120

def Connie_marbles : Nat := 39
def Juan_marbles : Nat := Connie_marbles + 25
def Maria_marbles : Nat := 2 * Juan_marbles
def Total_marbles : Nat := Connie_marbles + Juan_marbles + Maria_marbles

theorem Total_marbles_equal_231 : Total_marbles = 231 := sorry

end Total_marbles_equal_231_l108_108120


namespace inequality_generalization_l108_108189

theorem inequality_generalization (x : ℝ) (n : ℕ) (hn : 0 < n) (hx : 0 < x) :
  x + n^n / x^n ≥ n + 1 :=
sorry

end inequality_generalization_l108_108189


namespace exists_rational_non_integer_a_not_exists_rational_non_integer_b_l108_108535

-- Define rational non-integer numbers
def is_rational_non_integer (x : ℚ) : Prop := ¬(∃ (z : ℤ), x = z)

-- (a) Proof for existance of rational non-integer numbers y and x such that 19x + 8y, 8x + 3y are integers
theorem exists_rational_non_integer_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ a b : ℤ, 19 * x + 8 * y = a ∧ 8 * x + 3 * y = b) :=
sorry

-- (b) Proof for non-existance of rational non-integer numbers y and x such that 19x² + 8y², 8x² + 3y² are integers
theorem not_exists_rational_non_integer_b :
  ¬ ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ m n : ℤ, 19 * x^2 + 8 * y^2 = m ∧ 8 * x^2 + 3 * y^2 = n) :=
sorry

end exists_rational_non_integer_a_not_exists_rational_non_integer_b_l108_108535


namespace square_distance_eq_65_l108_108836

theorem square_distance_eq_65
  (B : ℝ × ℝ)
  (r : ℝ) (AB BC : ℝ)
  (angle_ABC : B.1 + 3 = B.2 ∧ - B.2 + 8 = B.1)
  (circle_eq : ∀ (P : ℝ × ℝ), P = B ∨ P.1 = B.1 ∨ P.2 = B.2 ∧ (P.1^2 + P.2^2 = 75))
  (radius : r = sqrt 75)
  (len_AB : AB = 8)
  (len_BC : BC = 3) :
  (B.1)^2 + (B.2)^2 = 65 :=
sorry

end square_distance_eq_65_l108_108836


namespace zeroes_lt_three_range_l108_108621

theorem zeroes_lt_three_range (f : ℝ → ℝ) (a : ℝ)
  (h_f_def : ∀ x, f x = x^3 - a * x^2 + 4)
  (h_zeroes_lt_three : (∃ x₁ x₂ x₃, f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) → false) :
  a ∈ set.Iic 3 :=
by
  sorry

end zeroes_lt_three_range_l108_108621
