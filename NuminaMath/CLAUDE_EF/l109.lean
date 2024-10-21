import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_scalar_l109_10922

/-- Given two 2D vectors are parallel, prove that the scalar k satisfies a specific value -/
theorem parallel_vectors_scalar (a b : ℝ × ℝ) (k : ℝ) : 
  a = (1, 2) → 
  b = (-3, 2) → 
  (∃ (t : ℝ), t ≠ 0 ∧ (k * a.1 + b.1, k * a.2 + b.2) = t • (a.1 - 3 * b.1, a.2 - 3 * b.2)) → 
  k = -1/3 := by
  sorry

#check parallel_vectors_scalar

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_scalar_l109_10922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_vector_l109_10980

/-- The vector to be reflected -/
def v : Fin 3 → ℝ := ![2, 3, 1]

/-- The vector to reflect over -/
def u : Fin 3 → ℝ := ![4, -1, 2]

theorem reflection_over_vector :
  (2 • ((v • u) / (u • u)) • u - v) = ![2/3, -11/3, 1/3] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_vector_l109_10980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tile_probability_l109_10984

/-- The number of tiles in the container -/
def n : ℕ := 98

/-- A tile is red if its number is congruent to 3 mod 7 -/
def is_red (k : ℕ) : Bool := k % 7 = 3

/-- The count of red tiles in the container -/
def red_count : ℕ := (Finset.range n).filter (fun k => is_red k) |>.card

/-- The probability of selecting a red tile -/
noncomputable def red_probability : ℚ := red_count / n

/-- Theorem: The probability of selecting a red tile is 1/7 -/
theorem red_tile_probability : red_probability = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tile_probability_l109_10984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_and_rule_winner_wallenstein_wins_7_7_l109_10968

/-- Represents the state of the "Divide and rule!" game -/
structure GameState where
  M : ℕ  -- Number of companies
  N : ℕ  -- Total number of soldiers
  soldiers : List ℕ  -- Number of soldiers in each company
  deriving Repr

/-- Represents a player in the game -/
inductive Player
  | Wallenstein
  | Tilly
  deriving Repr

/-- Defines the winner of the game based on the parity of N - M -/
def winner (state : GameState) : Player :=
  if (state.N - state.M) % 2 = 0 then
    Player.Wallenstein
  else
    Player.Tilly

/-- The main theorem stating the winning condition -/
theorem divide_and_rule_winner (state : GameState) :
  (∀ n ∈ state.soldiers, n ≥ 1) →  -- Each company has at least one soldier
  state.N = state.soldiers.sum →   -- Total soldiers is sum of all companies
  state.M = state.soldiers.length →  -- Number of companies matches list length
  winner state = Player.Wallenstein ↔ (state.N - state.M) % 2 = 0 :=
sorry

/-- Specific case for 7 companies with 7 soldiers each -/
theorem wallenstein_wins_7_7 :
  let state : GameState := ⟨7, 49, List.replicate 7 7⟩
  winner state = Player.Wallenstein :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_and_rule_winner_wallenstein_wins_7_7_l109_10968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_squared_beta_l109_10999

noncomputable def f (x : ℝ) := Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem f_squared_beta (α β : ℝ) 
  (h1 : Real.cos (β - α) = 4/5)
  (h2 : Real.cos (β + α) = -4/5)
  (h3 : 0 < α) (h4 : α < β) (h5 : β ≤ Real.pi/2) : 
  (f β)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_squared_beta_l109_10999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_theorem_l109_10902

noncomputable def a (θ : Real) : Fin 2 → Real := ![Real.sqrt 3, Real.sin θ]
noncomputable def b (θ : Real) : Fin 2 → Real := ![1, Real.cos θ]

noncomputable def f (x θ : Real) : Real := Real.sin (2 * x + θ)

theorem parallel_vectors_theorem (θ : Real) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2)
  (h2 : ∃ (k : Real), a θ = k • b θ) :
  (Real.sin θ = Real.sqrt 3 / 2 ∧ Real.cos θ = 1 / 2) ∧ 
  (∀ x, f x θ = f (x + Real.pi / 2) θ) ∧
  (∀ (k : Int), 
    StrictMonoOn (f · θ) 
      (Set.Icc (- (5 * Real.pi / 12) + k * Real.pi) ((Real.pi / 12) + k * Real.pi))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_theorem_l109_10902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_16_formula_l109_10994

variable (b : ℝ)

noncomputable def v : ℕ → ℝ
  | 0 => b  -- Added case for 0
  | 1 => b
  | n + 2 => -2 / (v n + 2)

theorem v_16_formula (h : b > 0) : v b 16 = -2 * (b + 1) / (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_16_formula_l109_10994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l109_10983

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x + a else x^2

-- State the theorem
theorem min_a_value (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) → -- f has a minimum value
  (f a (f a (-2)) = 16) → -- f[f(-2)] = 16
  (∀ (b : ℝ), (∃ (m : ℝ), ∀ (x : ℝ), f b x ≥ m) → b ≥ 2) → -- for all b that make f have a minimum value, b ≥ 2
  a = 2 -- the minimum value of a is 2
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l109_10983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_real_range_is_neg_inf_to_neg_one_increasing_in_half_to_one_l109_10964

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*a*x + 3) / Real.log (1/2)

-- Theorem 1: Domain is ℝ
theorem domain_is_real (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) → a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry

-- Theorem 2: Range is (-∞, -1]
theorem range_is_neg_inf_to_neg_one (a : ℝ) :
  (Set.range (f a) = Set.Iic (-1)) → (a = 1 ∨ a = -1) := by
  sorry

-- Theorem 3: Increasing in (1/2, 1)
theorem increasing_in_half_to_one (a : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Ioo (1/2) 1 → y ∈ Set.Ioo (1/2) 1 → x < y → f a x < f a y) →
  a ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_real_range_is_neg_inf_to_neg_one_increasing_in_half_to_one_l109_10964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_property_l109_10960

def f (c : ℕ) : ℕ → ℕ
  | 0 => 1  -- Add this case for n = 0
  | 1 => 1
  | 2 => c
  | (n + 3) => 2 * f c (n + 2) - f c (n + 1) + 2

theorem sequence_product_property (c : ℕ) (hc : c > 0) :
  ∀ k : ℕ, (f c k) * (f c (k + 1)) = f c ((f c k) + k) :=
by sorry

#check sequence_product_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_property_l109_10960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_article_cost_correct_l109_10929

/-- The actual cost of a single article before discounts and taxes -/
def single_article_cost : ℝ := 982.89

/-- The number of articles purchased -/
def num_articles : ℕ := 5

/-- The discount rate for purchasing 4 or more articles -/
def discount_rate : ℝ := 0.30

/-- The additional promotional discount rate -/
def promo_discount_rate : ℝ := 0.05

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.08

/-- The total cost after discounts and taxes -/
def total_cost : ℝ := 3530.37

/-- Theorem stating that the given single article cost satisfies the problem conditions -/
theorem single_article_cost_correct :
  let discounted_price := single_article_cost * (1 - discount_rate)
  let total_discounted := (num_articles : ℝ) * discounted_price * (1 - promo_discount_rate)
  let final_price := total_discounted * (1 + sales_tax_rate)
  abs (final_price - total_cost) < 0.01 := by
  sorry

#check single_article_cost_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_article_cost_correct_l109_10929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_foldable_configurations_l109_10904

/-- Represents a position where an additional square can be attached -/
inductive Position
| OuterEdge
| CenterAdjacent

/-- Represents a configuration of squares -/
structure Configuration :=
  (base : Fin 5)  -- Represents the 5 squares in the base arrangement
  (additional : Position)  -- Position of the additional square

/-- Predicate to determine if a configuration can be folded into a cube with one face missing -/
def can_fold_to_cube (config : Configuration) : Bool :=
  match config.additional with
  | Position.OuterEdge => true
  | Position.CenterAdjacent => false

/-- The total number of possible configurations -/
def total_configurations : Nat := 10

/-- The number of configurations that can be folded into a cube with one face missing -/
def foldable_configurations : Nat := 6

/-- Theorem stating that exactly 6 out of 10 configurations can be folded into a cube -/
theorem six_foldable_configurations :
  ∃ (configs : Finset Configuration),
    configs.card = total_configurations ∧
    (configs.filter (fun c => can_fold_to_cube c)).card = foldable_configurations :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_foldable_configurations_l109_10904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l109_10957

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(m^2 - 2*m - 1)

theorem sufficient_but_not_necessary :
  (∀ x y : ℝ, 0 < x ∧ x < y → f 1 y < f 1 x) ∧
  (∃ m : ℝ, m ≠ 1 ∧ ∀ x y : ℝ, 0 < x ∧ x < y → f m y < f m x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l109_10957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sphere_surface_area_l109_10991

noncomputable section

structure Cube where
  edge : ℝ
  vertices : Set (Fin 3 → ℝ)
  surfaceArea : ℝ

structure Sphere where
  center : Fin 3 → ℝ
  radius : ℝ
  surface : Set (Fin 3 → ℝ)
  surfaceArea : ℝ

theorem cube_sphere_surface_area (cube : Cube) (sphere : Sphere) :
  (∀ v ∈ cube.vertices, v ∈ sphere.surface) →
  cube.surfaceArea = 18 →
  sphere.surfaceArea = 9 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sphere_surface_area_l109_10991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_of_triplets_with_unit_dot_product_l109_10933

theorem sequence_of_triplets_with_unit_dot_product 
  (a₀ b₀ c₀ a b c : ℤ) 
  (h1 : Int.gcd a₀ (Int.gcd b₀ c₀) = 1) 
  (h2 : Int.gcd a (Int.gcd b c) = 1) : 
  ∃ (n : ℕ) (as bs cs : List ℤ), 
    (as.length = n + 1) ∧ 
    (bs.length = n + 1) ∧ 
    (cs.length = n + 1) ∧
    (as.head? = some a₀) ∧ 
    (bs.head? = some b₀) ∧ 
    (cs.head? = some c₀) ∧
    (as.getLast? = some a) ∧ 
    (bs.getLast? = some b) ∧ 
    (cs.getLast? = some c) ∧
    (∀ i : Fin n, 
      let a1 := (as.get? i.val).getD 0
      let a2 := (as.get? (i.val + 1)).getD 0
      let b1 := (bs.get? i.val).getD 0
      let b2 := (bs.get? (i.val + 1)).getD 0
      let c1 := (cs.get? i.val).getD 0
      let c2 := (cs.get? (i.val + 1)).getD 0
      a1 * a2 + b1 * b2 + c1 * c2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_of_triplets_with_unit_dot_product_l109_10933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_convex_is_convex_sum_of_support_rectangles_is_support_rectangle_l109_10955

-- Define a convex set
def IsConvex (S : Set (ℝ × ℝ)) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • x + t • y ∈ S

-- Define a support rectangle
def IsSupportRectangle (R : Set (ℝ × ℝ)) (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c d : ℝ, R = {z : ℝ × ℝ | a ≤ z.1 ∧ z.1 ≤ b ∧ c ≤ z.2 ∧ z.2 ≤ d} ∧ S ⊆ R

-- Define the Minkowski sum of sets
def MinkowskiSum (A B : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {z | ∃ x ∈ A, ∃ y ∈ B, z = (x.1 + y.1, x.2 + y.2)}

-- Theorem 1: Intersection of convex sets is convex
theorem intersection_of_convex_is_convex (K1 K2 : Set (ℝ × ℝ)) 
    (h1 : IsConvex K1) (h2 : IsConvex K2) : 
    IsConvex (K1 ∩ K2) := by
  sorry

-- Theorem 2: Sum of support rectangles is a support rectangle
theorem sum_of_support_rectangles_is_support_rectangle 
    (A B C M N : Set (ℝ × ℝ)) 
    (h1 : C = MinkowskiSum A B)
    (h2 : IsSupportRectangle M A)
    (h3 : IsSupportRectangle N B)
    (h4 : ∃ θ : ℝ, ∀ x y u v : ℝ × ℝ, x ∈ M → y ∈ M → u ∈ N → v ∈ N → 
      (y.2 - x.2) * Real.cos θ = (v.2 - u.2) * Real.cos θ ∧ 
      (y.2 - x.2) * Real.sin θ = (v.2 - u.2) * Real.sin θ) :
    IsSupportRectangle (MinkowskiSum M N) C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_convex_is_convex_sum_of_support_rectangles_is_support_rectangle_l109_10955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l109_10940

noncomputable section

def Triangle (A B C : ℝ × ℝ) := True

def angle (A B C : ℝ × ℝ) : ℝ := sorry

def length (A B : ℝ × ℝ) : ℝ := sorry

def orthocenter (t : Triangle A B C) : ℝ × ℝ := sorry

def incenter (t : Triangle A B C) : ℝ × ℝ := sorry

def circumcenter (t : Triangle A B C) : ℝ × ℝ := sorry

def midpoint_of_line (A C : ℝ × ℝ) : ℝ × ℝ := sorry

def area_hexagon (A B C D E F : ℝ × ℝ) : ℝ := sorry

def is_maximum (f : ℝ → ℝ) (x : ℝ) : Prop := sorry

theorem triangle_angle_theorem 
  (A B C : ℝ × ℝ) 
  (t : Triangle A B C) 
  (H : ℝ × ℝ) 
  (I : ℝ × ℝ) 
  (O : ℝ × ℝ) 
  (D : ℝ × ℝ) :
  angle B A C = 45 →
  angle C B A ≤ 90 →
  length B C = 2 →
  length A C ≥ length A B →
  H = orthocenter t →
  I = incenter t →
  O = circumcenter t →
  D = midpoint_of_line A C →
  is_maximum (λ θ => area_hexagon B C O I D H) (angle C B A) →
  angle C B A = 75 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l109_10940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l109_10908

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) := Real.log (x^2 - 3*x + a)
def g (a : ℝ) (x : ℝ) := (2*a - 5) * x

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, f a x = y
def q (a : ℝ) : Prop := StrictMono (g a)

-- State the theorem
theorem range_of_a :
  ∃ S : Set ℝ, S = Set.Ioc (9/4) (5/2) ∧
  ∀ a : ℝ, (((p a ∨ q a) ∧ ¬(p a ∧ q a)) ↔ a ∈ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l109_10908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_greater_than_one_if_A15_greater_than_A10_l109_10947

/-- Given a finite set of positive real numbers, A_n is the sum of all products of n elements from the set. -/
def A_n (M : Finset ℝ) (n : ℕ) : ℝ :=
  (M.powerset.filter (fun s => s.card = n)).sum (fun s => s.prod id)

/-- Theorem statement -/
theorem sum_greater_than_one_if_A15_greater_than_A10 (M : Finset ℝ) :
  M.card = 30 →
  (∀ x, x ∈ M → x > 0) →
  (∀ x y, x ∈ M → y ∈ M → x ≠ y → x ≠ y) →
  A_n M 15 > A_n M 10 →
  A_n M 1 > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_greater_than_one_if_A15_greater_than_A10_l109_10947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l109_10925

/-- The angle of inclination of the line x - y - 1 = 0 is 45 degrees. -/
theorem line_inclination_angle (x y : ℝ) :
  x - y - 1 = 0 → ∃ θ : ℝ, θ = 45 * Real.pi / 180 ∧ Real.tan θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l109_10925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_correct_l109_10997

/-- Two overlapping sectors with central angle 90° and radius 1 -/
structure OverlappingSectors where
  /-- Central angle of each sector in radians -/
  angle : ℝ
  /-- Radius of each sector -/
  radius : ℝ
  /-- Assumption that the central angle is 90° (π/2 radians) -/
  angle_is_90 : angle = Real.pi / 2
  /-- Assumption that the radius is 1 -/
  radius_is_1 : radius = 1

/-- The difference in area between regions S₁ and S₂ -/
noncomputable def areaDifference (s : OverlappingSectors) : ℝ :=
  (3 * Real.sqrt 3) / 8 - Real.pi / 6

/-- Theorem stating that the area difference is correct -/
theorem area_difference_correct (s : OverlappingSectors) :
  areaDifference s = (3 * Real.sqrt 3) / 8 - Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_correct_l109_10997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l109_10998

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3^(x - 2)

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x, 2 ≤ x ∧ x ≤ 4 ∧ f x = y} = {y : ℝ | 1 ≤ y ∧ y ≤ 9} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l109_10998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_modulus_is_three_l109_10934

noncomputable def z (a : ℝ) : ℂ := (a + 3 * Complex.I) / (1 + 2 * Complex.I)

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem z_modulus_is_three (a : ℝ) (h : is_pure_imaginary (z a)) : Complex.abs (z a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_modulus_is_three_l109_10934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l109_10938

def is_valid_polynomial (p : ℤ → ℤ) : Prop :=
  ∃ (degree : ℕ) (coeffs : Fin (degree + 1) → ℤ),
    ∀ x, p x = (Finset.range (degree + 1)).sum (λ i ↦ coeffs i * x^i)

theorem polynomial_value_theorem (p : ℤ → ℤ) :
  is_valid_polynomial p →
  p 0 = 0 →
  0 ≤ p 1 ∧ p 1 ≤ 10^7 →
  (∃ (a b : ℕ+), p a = 1999 ∧ p b = 2001) →
  p 1 = 1 ∨ p 1 = 1999 ∨ p 1 = 3996001 ∨ p 1 = 7992001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l109_10938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l109_10954

/-- The focus of a parabola given by the equation y = ax² + bx + c. -/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  let p := 1 / (4 * a)
  (h, k + p)

/-- Theorem: The focus of the parabola y = 9x² + 6x - 5 is (-1/3, -215/36). -/
theorem focus_of_specific_parabola :
  parabola_focus 9 6 (-5) = (-1/3, -215/36) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l109_10954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l109_10924

open Real MeasureTheory

noncomputable def f (x : ℝ) := Real.exp (2 * x) + Real.exp (x + 2) - 2 * Real.exp 4

noncomputable def g (a : ℝ) (x : ℝ) := x^2 - 3 * a * Real.exp x

def A (f : ℝ → ℝ) := {x : ℝ | f x = 0}

def B (g : ℝ → ℝ → ℝ) (a : ℝ) := {x : ℝ | g a x = 0}

theorem range_of_a :
  ∀ a : ℝ,
  (∃ x₁ ∈ A f, ∃ x₂ ∈ B g a, |x₁ - x₂| < 1) ↔
  a ∈ Set.Ioo (1 / (3 * Real.exp 1)) (4 / (3 * Real.exp 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l109_10924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l109_10963

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / (e.a^2) + p.y^2 / (e.b^2) = 1

/-- The focus of the ellipse -/
noncomputable def focus (e : Ellipse) : Point :=
  ⟨Real.sqrt (e.a^2 - e.b^2), 0⟩

theorem ellipse_focus_distance 
  (e : Ellipse) 
  (p : Point) 
  (h1 : e.a = 2 ∧ e.b = 1) 
  (h2 : isOnEllipse e p) 
  (h3 : p.x = Real.sqrt 3) :
  distance p (focus e) = 7/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l109_10963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_week_profit_is_245_l109_10915

/-- Represents the daily data for passion fruit sales -/
structure DailyData where
  priceChange : Int
  soldQuantity : Nat

/-- Calculates the total profit for a week of passion fruit sales -/
def calculateWeekProfit (costPrice : Nat) (standardPrice : Nat) (weekData : List DailyData) : Int :=
  weekData.foldl
    (fun acc day =>
      let sellingPrice := (standardPrice : Int) + day.priceChange
      acc + (sellingPrice - costPrice) * day.soldQuantity)
    0

/-- Theorem stating that the calculated week profit is 245 yuan -/
theorem week_profit_is_245 (costPrice : Nat) (standardPrice : Nat) (weekData : List DailyData) :
  costPrice = 10 →
  standardPrice = 15 →
  weekData = [
    ⟨1, 20⟩, ⟨-3, 35⟩, ⟨2, 10⟩, ⟨-1, 30⟩, ⟨3, 15⟩, ⟨4, 5⟩, ⟨-9, 50⟩
  ] →
  calculateWeekProfit costPrice standardPrice weekData = 245 := by
  sorry

#eval calculateWeekProfit 10 15 [
  ⟨1, 20⟩, ⟨-3, 35⟩, ⟨2, 10⟩, ⟨-1, 30⟩, ⟨3, 15⟩, ⟨4, 5⟩, ⟨-9, 50⟩
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_week_profit_is_245_l109_10915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_approximation_l109_10935

-- Define the expressions
noncomputable def expr1 : ℝ := (2 * Real.sqrt 3) / (Real.sqrt 3 - Real.sqrt 2)
noncomputable def expr2 : ℝ := ((3 + Real.sqrt 3) * (1 + Real.sqrt 5)) / ((5 + Real.sqrt 5) * (1 + Real.sqrt 3))

-- State the theorem
theorem expressions_approximation :
  (|expr1 - 10.899| < 0.0005) ∧ (|expr2 - 0.775| < 0.0005) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_approximation_l109_10935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squiggle_coverage_l109_10912

/-- Represents a squiggle composed of six equilateral triangles with side length 1 -/
noncomputable def Squiggle : Type := Unit

/-- Area of a squiggle -/
noncomputable def Squiggle.area : ℝ := 3 * Real.sqrt 3 / 2

/-- Represents an equilateral triangle with side length n -/
structure EquilateralTriangle (n : ℕ) where
  side_length : ℕ := n

/-- Area of an equilateral triangle -/
noncomputable def EquilateralTriangle.area {n : ℕ} (t : EquilateralTriangle n) : ℝ :=
  Real.sqrt 3 / 4 * n^2

/-- Predicate to check if a triangle can be covered by squiggles -/
def canBeCoveredBySquiggles {n : ℕ} (t : EquilateralTriangle n) : Prop :=
  ∃ k : ℕ, t.area = k * Squiggle.area

/-- Main theorem: An equilateral triangle can be covered by squiggles iff its side length is divisible by 12 -/
theorem squiggle_coverage (n : ℕ) :
  (∃ t : EquilateralTriangle n, canBeCoveredBySquiggles t) ↔ 12 ∣ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squiggle_coverage_l109_10912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_in_circle_l109_10920

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
noncomputable def octagonArea : ℝ := 72 * Real.sqrt ((1 - Real.sqrt 2 / 2) / 2)

/-- Theorem stating that the area of a regular octagon inscribed in a circle with radius 3 units
    is equal to 72 * √((1 - √2/2)/2) square units -/
theorem regular_octagon_area_in_circle (r : ℝ) (h : r = 3) :
  octagonArea = 8 * (1/2 * r * (2 * r * Real.sin (π/8))) := by
  sorry

#check octagonArea
#check regular_octagon_area_in_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_in_circle_l109_10920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clubsuit_sum_l109_10903

noncomputable def clubsuit (x : ℝ) : ℝ := (x^2 + x^3) / 2

theorem clubsuit_sum : clubsuit 2 + clubsuit 3 + clubsuit 4 = 64 := by
  -- Unfold the definition of clubsuit
  unfold clubsuit
  -- Simplify the expressions
  simp [pow_two, pow_three]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clubsuit_sum_l109_10903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_after_moving_l109_10974

/-- Given two points A(a,b) and B(c,d) on a Cartesian plane with midpoint M(m,n),
    if A is moved 3 units right and 5 units up, and B is moved 5 units left and 3 units down,
    then the distance between the original midpoint M and the new midpoint M' is √2. -/
theorem midpoint_distance_after_moving (a b c d m n : ℝ) :
  m = (a + c) / 2 →
  n = (b + d) / 2 →
  Real.sqrt ((((a + 3 + c - 5) / 2) - m)^2 + (((b + 5 + d - 3) / 2) - n)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_after_moving_l109_10974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l109_10977

/-- The operation ◦ defined on real numbers -/
def circleOp (x y : ℝ) : ℝ := 2*x - 4*y + 3*x*y

/-- Theorem stating that there exists a unique real number y such that 5 ◦ y = 7 -/
theorem unique_solution :
  ∃! y : ℝ, circleOp 5 y = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l109_10977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_equilateral_triangle_on_hyperbola_l109_10909

noncomputable section

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define a point on the hyperbola
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

-- Define the left vertex of the hyperbola
def left_vertex : PointOnHyperbola :=
  { x := -1, y := 0, on_hyperbola := by simp [hyperbola] }

-- Define an equilateral triangle
def is_equilateral (A B C : PointOnHyperbola) : Prop :=
  (B.x - A.x)^2 + (B.y - A.y)^2 = (C.x - B.x)^2 + (C.y - B.y)^2 ∧
  (C.x - A.x)^2 + (C.y - A.y)^2 = (B.x - A.x)^2 + (B.y - A.y)^2

-- Define the area of a triangle
noncomputable def triangle_area (A B C : PointOnHyperbola) : ℝ :=
  (1/2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

-- The main theorem
theorem area_of_equilateral_triangle_on_hyperbola
  (B C : PointOnHyperbola)
  (h1 : B.x > 0)
  (h2 : C.x > 0)
  (h3 : is_equilateral left_vertex B C) :
  triangle_area left_vertex B C = 3 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_equilateral_triangle_on_hyperbola_l109_10909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_greater_than_one_l109_10982

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x) / Real.log a

-- State the theorem
theorem increasing_function_a_greater_than_one :
  ∀ a : ℝ, a > 0 ∧ a ≠ 1 →
  (∀ x₁ x₂ : ℝ, 2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 → f a x₁ < f a x₂) →
  a > 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_greater_than_one_l109_10982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_specific_l109_10944

/-- The number of days it takes for three workers to complete a task together,
    given their individual completion times. -/
noncomputable def combined_work_time (t1 t2 t3 : ℝ) : ℝ :=
  1 / (1 / t1 + 1 / t2 + 1 / t3)

/-- Theorem stating that three workers with individual completion times of 24, 40, and 60 days
    can complete the task together in 12 days. -/
theorem combined_work_time_specific : combined_work_time 24 40 60 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_specific_l109_10944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_partition_l109_10993

def T (n : ℕ) : Set ℕ := {x | 2 ≤ x ∧ x ≤ n}

def hasProductTriple (S : Set ℕ) : Prop :=
  ∃ x y z, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x * y = z

def validPartition (n : ℕ) : Prop :=
  ∀ A B : Set ℕ, A ∪ B = T n → A ∩ B = ∅ →
    hasProductTriple A ∨ hasProductTriple B

theorem smallest_valid_partition : 
  (∀ n < 256, ¬validPartition n) ∧ validPartition 256 := by
  sorry

#check smallest_valid_partition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_partition_l109_10993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_odds_l109_10953

/-- Represents the odds against an event occurring as a ratio of two natural numbers -/
structure Odds where
  against : ℕ
  in_favor : ℕ

/-- Calculates the probability of an event given the odds against it -/
def oddsToProb (o : Odds) : ℚ :=
  o.in_favor / (o.against + o.in_favor)

theorem race_odds (x y z : Odds) (hx : x = ⟨4, 1⟩) (hy : y = ⟨1, 2⟩)
    (hsum : oddsToProb x + oddsToProb y + oddsToProb z = 1) :
    z = ⟨13, 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_odds_l109_10953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solutions_l109_10976

noncomputable def is_solution (n k m : ℕ) : Prop :=
  n ≤ 5 ∧ k ≤ 5 ∧ m ≤ 5 ∧
  Real.sin (Real.pi * n / 12) * Real.sin (Real.pi * k / 12) * Real.sin (Real.pi * m / 12) = 1 / 8

theorem trigonometric_equation_solutions :
  ∀ n k m : ℕ, is_solution n k m ↔
    ((n, k, m) = (2, 2, 2) ∨
     (n, k, m) = (1, 2, 5) ∨ (n, k, m) = (1, 5, 2) ∨
     (n, k, m) = (2, 1, 5) ∨ (n, k, m) = (2, 5, 1) ∨
     (n, k, m) = (5, 1, 2) ∨ (n, k, m) = (5, 2, 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solutions_l109_10976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_l109_10907

/-- The radius of a sphere -/
def sphere_radius : ℝ := 2

/-- The number of spheres -/
def num_spheres : ℕ := 8

/-- The number of octants -/
def num_octants : ℕ := 8

/-- Function to check if a sphere is tangent to coordinate planes -/
def is_tangent_to_planes (center : ℝ × ℝ × ℝ) : Prop :=
  ∃ (i j k : ℝ), center = (i * sphere_radius, j * sphere_radius, k * sphere_radius) ∧ 
  (i = 1 ∨ i = -1) ∧ (j = 1 ∨ j = -1) ∧ (k = 1 ∨ k = -1)

/-- The centers of the spheres form a cube -/
def centers_form_cube (centers : Finset (ℝ × ℝ × ℝ)) : Prop :=
  centers.card = num_spheres ∧ 
  ∀ c ∈ centers, is_tangent_to_planes c

/-- The radius of the smallest enclosing sphere -/
noncomputable def enclosing_sphere_radius (centers : Finset (ℝ × ℝ × ℝ)) : ℝ :=
  2 * Real.sqrt 3 + sphere_radius

/-- Theorem stating the radius of the smallest enclosing sphere -/
theorem smallest_enclosing_sphere_radius 
  (centers : Finset (ℝ × ℝ × ℝ)) 
  (h : centers_form_cube centers) :
  enclosing_sphere_radius centers = 2 * Real.sqrt 3 + 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_l109_10907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l109_10945

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_passing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A 700-meter long train traveling at 120 km/hour passes a 350-meter long bridge in approximately 31.5 seconds -/
theorem train_bridge_passing_time :
  let result := train_passing_time 700 350 120
  (result > 31.4) ∧ (result < 31.6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l109_10945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_is_optimal_l109_10949

/-- The maximum value of a such that y = mx + 3 passes through no lattice point for 0 < x ≤ 150 and 1/3 < m < a -/
def max_a : ℚ := 51 / 151

/-- A lattice point is a point with integer coordinates -/
def is_lattice_point (x y : ℚ) : Prop := ∃ (ix iy : ℤ), (x : ℝ) = ix ∧ (y : ℝ) = iy

/-- The line y = mx + 3 passes through a point (x, y) -/
def line_passes_through (m : ℚ) (x y : ℚ) : Prop := y = m * x + 3

/-- The theorem stating that 51/151 is the maximum possible value of a -/
theorem max_a_is_optimal :
  ∀ a : ℚ, (∀ m : ℚ, 1/3 < m → m < a → 
    ∀ x y : ℚ, 0 < x → x ≤ 150 → is_lattice_point x y → ¬ line_passes_through m x y) →
  a ≤ max_a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_is_optimal_l109_10949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_midpoint_l109_10900

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the focal points -/
def focal_distance (a c : ℝ) : Prop := 2 * c = 2

/-- Sum of distances from any point on C to the foci -/
def sum_distances (a : ℝ) : Prop := 2 * a = 2 * Real.sqrt 2

/-- Relationship between a, b, and c -/
def ellipse_parameters (a b c : ℝ) : Prop := a^2 = b^2 + c^2 ∧ a > b ∧ b > 0

/-- Definition of a chord of length 2 on the ellipse -/
def chord_length (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4

/-- Midpoint of a chord -/
def chord_midpoint (x₁ y₁ x₂ y₂ xm ym : ℝ) : Prop := xm = (x₁ + x₂) / 2 ∧ ym = (y₁ + y₂) / 2

/-- Distance from origin to a point -/
noncomputable def distance_from_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- Main theorem: Maximum distance from origin to midpoint of a chord of length 2 -/
theorem max_distance_to_midpoint (a b c : ℝ) :
  ellipse_parameters a b c →
  focal_distance a c →
  sum_distances a →
  ∀ x₁ y₁ x₂ y₂ xm ym : ℝ,
    ellipse x₁ y₁ a b →
    ellipse x₂ y₂ a b →
    chord_length x₁ y₁ x₂ y₂ →
    chord_midpoint x₁ y₁ x₂ y₂ xm ym →
    distance_from_origin xm ym ≤ Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_midpoint_l109_10900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l109_10923

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (x + 2 * Real.pi / 3) + 2 * (Real.cos (x / 2))^2

-- Define the properties of triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- internal angles
  (a b c : ℝ)  -- opposite sides

-- Theorem statement
theorem problem_solution :
  (∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 2) ∧  -- Range of f is [0, 2]
  (∀ t : Triangle, 
    f t.B = 1 → 
    t.b = 1 → 
    t.c = Real.sqrt 3 → 
    (t.a = 1 ∨ t.a = 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l109_10923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_expression_equal_to_20a_minus_18b_l109_10981

-- Define the unknown operations
def op1 : ℝ → ℝ → ℝ := sorry
def op2 : ℝ → ℝ → ℝ := sorry

-- Axioms representing the conditions of the problem
axiom op1_is_add_or_sub : 
  (∀ x y, op1 x y = x + y) ∨ (∀ x y, op1 x y = x - y) ∨ (∀ x y, op1 x y = y - x)

axiom op2_is_add_or_sub : 
  (∀ x y, op2 x y = x + y) ∨ (∀ x y, op2 x y = x - y) ∨ (∀ x y, op2 x y = y - x)

axiom ops_are_different : op1 ≠ op2

-- Theorem statement
theorem exists_expression_equal_to_20a_minus_18b :
  ∃ f : ℝ → ℝ → ℝ, ∀ a b, f a b = 20 * a - 18 * b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_expression_equal_to_20a_minus_18b_l109_10981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l109_10973

/-- Definition of the hyperbola equation -/
noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y + 2)^2) - Real.sqrt ((x - 5)^2 + (y + 2)^2) = 3

/-- The positive slope of the asymptote of the hyperbola -/
noncomputable def asymptote_slope : ℝ := Real.sqrt 7 / 3

/-- Theorem stating that the positive slope of the asymptote of the given hyperbola is √7/3 -/
theorem hyperbola_asymptote_slope :
  ∀ x y : ℝ, hyperbola_equation x y → (∃ s : ℝ, s > 0 ∧ s = asymptote_slope) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l109_10973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_heads_in_three_tosses_is_three_eighths_l109_10937

/-- The probability of getting exactly two heads when tossing a fair coin three times -/
def probability_two_heads_in_three_tosses : ℚ :=
  -- Define the sample space (all possible outcomes)
  let sample_space := {("H", "H", "H"), ("H", "H", "T"), ("H", "T", "H"), ("T", "H", "H"),
                       ("H", "T", "T"), ("T", "H", "T"), ("T", "T", "H"), ("T", "T", "T")}
  -- Define the event (outcomes with exactly two heads)
  let event := {("H", "H", "T"), ("H", "T", "H"), ("T", "H", "H")}
  -- Calculate the probability
  (Finset.card event : ℚ) / (Finset.card sample_space : ℚ)

#eval probability_two_heads_in_three_tosses

theorem probability_two_heads_in_three_tosses_is_three_eighths :
  probability_two_heads_in_three_tosses = 3 / 8 := by
  -- Unfold the definition
  unfold probability_two_heads_in_three_tosses
  -- Simplify the expression
  simp
  -- Evaluate the cardinalities
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_heads_in_three_tosses_is_three_eighths_l109_10937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_equals_five_l109_10975

theorem square_sum_equals_five (x y : ℝ) (hy : y > 0) : 
  let A : Set ℝ := {x^2 + x + 1, -x, -x - 1}
  let B : Set ℝ := {-y, -y/2, y + 1}
  A = B → x^2 + y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_equals_five_l109_10975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l109_10942

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the hypothesis that e₁ and e₂ are non-collinear
variable (h_non_collinear : ∀ (r : ℝ), r • e₁ ≠ e₂)

-- Define the vectors AB, CB, and CD
def AB (e₁ e₂ : V) : V := 3 • e₁ + 2 • e₂
def CB (e₁ e₂ : V) : V := 2 • e₁ - 5 • e₂
def CD (lambda : ℝ) (e₁ e₂ : V) : V := lambda • e₁ - e₂

-- Define the collinearity of points A, B, and D
def collinear (AB BD : V) : Prop :=
  ∃ (k : ℝ), AB = k • BD

-- State the theorem
theorem vector_collinearity (e₁ e₂ : V) (h_non_collinear : ∀ (r : ℝ), r • e₁ ≠ e₂) :
  collinear (AB e₁ e₂) ((CD 8 e₁ e₂) - (CB e₁ e₂)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l109_10942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_closed_form_l109_10939

noncomputable def x : ℕ → ℝ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 2 => Real.sqrt (2 * (x (n + 1))^4 + 6 * (x (n + 1))^2 + 3)

theorem x_closed_form (n : ℕ) (h : n ≥ 1) : 
  x n = Real.sqrt (1/2 * (5^(2^(n-1)) - 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_closed_form_l109_10939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l109_10967

noncomputable def dataset : List ℝ := [80, 81, 82, 83]

/-- The variance of a list of real numbers -/
noncomputable def variance (l : List ℝ) : ℝ :=
  let mean := l.sum / l.length
  (l.map (fun x => (x - mean) ^ 2)).sum / l.length

theorem variance_of_dataset :
  variance dataset = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l109_10967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_bound_l109_10927

noncomputable def nested_sqrt (n : ℕ) : ℕ → ℝ
  | 0 => 0
  | k + 1 => Real.sqrt ((k + 1 : ℝ) * nested_sqrt n k)

theorem nested_sqrt_bound (n : ℕ) (h : n > 0) : nested_sqrt n 2 < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_bound_l109_10927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_for_inequality_l109_10995

open Real

theorem max_k_for_inequality (f g : ℝ → ℝ) (h₁ : ∀ x, f x = log x) (h₂ : ∀ x, g x = exp x) :
  (∃ k, ∀ ε > 0, ε < k → ∃ x₁ x₂, x₁ ∈ Set.Icc 1 2 ∧ x₂ ∈ Set.Icc 1 2 ∧
    |g x₁ - g x₂| > (k - ε) * |f x₁ - f x₂|) ∧
  (∀ k > 2 * exp 2, ∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 →
    |g x₁ - g x₂| ≤ k * |f x₁ - f x₂|) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_for_inequality_l109_10995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_l109_10971

-- Define the function f(x) as noncomputable
noncomputable def f (a b x : ℝ) : ℝ := (Real.log x) / (Real.log a) + x - b

-- State the theorem
theorem zero_of_f_in_interval (a b : ℝ) (ha : 0 < a) (ha' : a ≠ 1) 
  (hab : 2 < a ∧ a < 3 ∧ 3 < b ∧ b < 4) :
  ∃ x₀ : ℝ, f a b x₀ = 0 ∧ 2 < x₀ ∧ x₀ < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_l109_10971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l109_10910

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_focus_distance (P : Point) 
  (h1 : P ∈ Parabola) 
  (h2 : P.x = 4) : 
  distance P focus = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l109_10910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_simplification_l109_10926

-- Define complex number addition
def add (z w : ℂ) : ℂ := z + w

-- Define complex number multiplication
def mul (z w : ℂ) : ℂ := z * w

-- Define scalar multiplication for complex numbers
def scalarMul (r : ℝ) (z : ℂ) : ℂ := r • z

-- State the theorem
theorem complex_simplification :
  add (scalarMul 3 (add 4 (scalarMul (-2) Complex.I)))
      (mul (scalarMul 2 Complex.I) (add 3 (scalarMul (-1) Complex.I)))
  = (14 : ℂ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_simplification_l109_10926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_similar_triangle_l109_10905

-- Define the original triangle
def original_triangle : ℝ × ℝ × ℝ := (4, 6, 7)

-- Define the area of the similar triangle
def similar_triangle_area : ℝ := 132

-- Function to calculate the semi-perimeter
noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Function to calculate the area using Heron's formula
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Function to calculate the scale factor
noncomputable def scale_factor (original_area similar_area : ℝ) : ℝ :=
  Real.sqrt (similar_area / original_area)

-- Theorem stating the length of the longest side of the similar triangle
theorem longest_side_of_similar_triangle :
  let (a, b, c) := original_triangle
  let original_area := triangle_area a b c
  let k := scale_factor original_area similar_triangle_area
  k * c = 73.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_similar_triangle_l109_10905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leibniz_theorem_l109_10951

/-- Leibniz's theorem for triangles -/
theorem leibniz_theorem (A B C M : EuclideanSpace ℝ (Fin 2)) :
  let G := (1/3 : ℝ) • (A + B + C)
  3 * ‖M - G‖^2 = ‖M - A‖^2 + ‖M - B‖^2 + ‖M - C‖^2 - 
    (1/3 : ℝ) * (‖A - B‖^2 + ‖B - C‖^2 + ‖C - A‖^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leibniz_theorem_l109_10951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_at_three_l109_10916

-- Define a fourth-degree polynomial with real coefficients
def fourthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := λ x ↦ a*x^4 + b*x^3 + c*x^2 + d*x + e

-- State the theorem
theorem absolute_value_at_three (a b c d e : ℝ) :
  let g := fourthDegreePolynomial a b c d e
  (|g 0| = 10) ∧ (|g 1| = 10) ∧ (|g 2| = 10) ∧ (|g 4| = 10) ∧ (|g 5| = 10) →
  |g 3| = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_at_three_l109_10916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l109_10928

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.cos (π / 2 + α) = 3 / 5)
  (h2 : α ∈ Set.Ioo (π / 2) (3 * π / 2)) :
  Real.tan α = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l109_10928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_focus_theorem_l109_10990

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the chord of an ellipse passing through a focus -/
theorem ellipse_chord_focus_theorem (e : Ellipse) (A B F' : Point) :
  e.a = 6 →
  e.b = 4 →
  F'.x = -Real.sqrt 20 →
  F'.y = 0 →
  isOnEllipse e A →
  isOnEllipse e B →
  distance A F' = 2 →
  (B.x - A.x) * F'.y = (B.y - A.y) * (F'.x - A.x) →  -- Collinearity condition
  distance B F' = Real.sqrt ((B.x + Real.sqrt 20)^2 + B.y^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_focus_theorem_l109_10990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_vote_difference_l109_10932

theorem election_vote_difference (total_students : ℕ) (voter_percentage : ℚ) (winner_percentage : ℚ) :
  total_students = 2000 →
  voter_percentage = 1/4 →
  winner_percentage = 55/100 →
  let total_votes : ℚ := (total_students : ℚ) * voter_percentage
  let winner_votes : ℚ := total_votes * winner_percentage
  let loser_votes : ℚ := total_votes - winner_votes
  ⌊winner_votes⌋ - ⌊loser_votes⌋ = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_vote_difference_l109_10932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_count_is_three_l109_10948

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Counts the number of common tangent lines between two circles -/
def commonTangentCount (c1 c2 : Circle) : ℕ :=
  sorry

/-- The main theorem stating that the number of common tangent lines between the given circles is 3 -/
theorem common_tangent_count_is_three : 
  let c1 : Circle := { center := (4, 0), radius := 3 }
  let c2 : Circle := { center := (0, 3), radius := 2 }
  commonTangentCount c1 c2 = 3 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_count_is_three_l109_10948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_5_40_l109_10986

/-- The angle between clock hands at a given time -/
noncomputable def clock_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let minute_angle := (minutes : ℝ) * 6
  let hour_angle := ((hours % 12) : ℝ) * 30 + (minutes : ℝ) * 0.5
  abs (hour_angle - minute_angle)

/-- The acute angle between clock hands at a given time -/
noncomputable def acute_clock_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  min (clock_angle hours minutes) (360 - clock_angle hours minutes)

theorem clock_angle_at_5_40 :
  acute_clock_angle 5 40 = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_5_40_l109_10986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_arrangement_theorem_l109_10959

def total_products : ℕ := 10
def defective_products : ℕ := 4

def arrange_products_scenario1 : ℕ :=
  (Nat.choose 6 4) * (Nat.choose 5 3) * (Nat.factorial 5) * (Nat.factorial 4)

def arrange_products_scenario2 : ℕ :=
  (Nat.choose 6 4) * (Nat.factorial 5)

theorem product_arrangement_theorem :
  (arrange_products_scenario1 = 103680) ∧
  (arrange_products_scenario2 = 576) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_arrangement_theorem_l109_10959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_specific_temps_l109_10956

-- Define the relationship between temperature change and volume change
noncomputable def volume_change_per_temp (temp_change : ℝ) : ℝ :=
  (3 / 4) * temp_change

-- Define the initial conditions
def initial_temp : ℝ := 30
def initial_volume : ℝ := 40

-- Define the target temperatures
def temp_22 : ℝ := 22
def temp_14 : ℝ := 14

-- Define the function to calculate volume at a given temperature
noncomputable def volume_at_temp (temp : ℝ) : ℝ :=
  initial_volume - volume_change_per_temp (initial_temp - temp)

-- State the theorem
theorem gas_volume_at_specific_temps :
  volume_at_temp temp_22 = 34 ∧ volume_at_temp temp_14 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_specific_temps_l109_10956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invertible_function_property_l109_10917

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- State the theorem
theorem invertible_function_property
  (g_inv : ℝ → ℝ)
  (h_left : Function.LeftInverse g_inv g)
  (h_right : Function.RightInverse g_inv g)
  (a c : ℝ)
  (h_ga : g a = c)
  (h_gc : g c = 5) :
  a - c = -3 :=
by
  sorry

#check invertible_function_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invertible_function_property_l109_10917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l109_10918

/-- A function f : ℝ → ℝ is in C¹[0,1] if it is continuously differentiable on [0,1] -/
def ContinuouslyDifferentiableOn (f : ℝ → ℝ) : Prop :=
  ContinuousOn f (Set.Icc 0 1) ∧ DifferentiableOn ℝ f (Set.Ioo 0 1)

theorem unique_function_satisfying_conditions :
  ∀ f : ℝ → ℝ,
  ContinuouslyDifferentiableOn f →
  f 1 = -1/6 →
  (∫ (x : ℝ) in Set.Icc 0 1, (deriv f x)^2) ≤ 2 * (∫ (x : ℝ) in Set.Icc 0 1, f x) →
  ∀ x, x ∈ Set.Icc 0 1 → f x = 1/3 - x^2/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l109_10918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_evaluation_l109_10952

theorem ceiling_sum_evaluation : 
  ⌈Real.sqrt (16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)⁻¹⌉ + ⌈(16/9 : ℝ)^3⌉ = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_evaluation_l109_10952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l109_10989

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), c > 0 ∧ F₁ = (-c, 0) ∧ F₂ = (c, 0)

-- Define points A and B on the left branch
def left_branch_points (A B : ℝ × ℝ) : Prop :=
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ A.1 < 0 ∧ B.1 < 0

-- Define collinearity of F₁, A, and B
def collinear (F₁ A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), A = F₁ + t • (B - F₁) ∨ B = F₁ + t • (A - F₁)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main theorem
theorem hyperbola_triangle_perimeter
  (F₁ F₂ A B : ℝ × ℝ)
  (h_foci : foci F₁ F₂)
  (h_left_branch : left_branch_points A B)
  (h_collinear : collinear F₁ A B)
  (h_AB_distance : distance A B = 5) :
  distance A F₂ + distance B F₂ + distance A B = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l109_10989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l109_10985

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain of x
def domain : Set ℝ := Set.Icc (-2) 4

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x y, x ∈ domain → y ∈ domain → x ≤ y → x ≤ 1 → f x ≥ f y :=
by
  sorry

#check f_decreasing_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l109_10985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_pin_purchase_l109_10988

/-- Calculates the total cost of pins including discount and sales tax -/
def total_cost (n : ℕ) (p d t : ℚ) : ℚ :=
  n * p * (1 - d) * (1 + t)

/-- Theorem stating the total cost for John's pin purchase -/
theorem johns_pin_purchase :
  total_cost 10 20 (15 / 100) (8 / 100) = 183.6 := by
  -- Unfold the definition of total_cost
  unfold total_cost
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

#eval total_cost 10 20 (15 / 100) (8 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_pin_purchase_l109_10988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_is_circle_l109_10965

-- Define the system of equations
def system (a x y : ℝ) : Prop :=
  a * x + y = 2 * a + 3 ∧ x - a * y = a + 4

-- Define the circle
def circle_eq (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

-- Define the solution set
def solution_set (x y : ℝ) : Prop :=
  circle_eq x y ∧ (x, y) ≠ (2, -1)

-- Theorem statement
theorem system_solution_is_circle :
  ∀ x y : ℝ, (∃ a : ℝ, system a x y) ↔ solution_set x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_is_circle_l109_10965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_DE_EF_l109_10913

open EuclideanSpace

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points D and E
variable (D : EuclideanSpace ℝ (Fin 2))
variable (E : EuclideanSpace ℝ (Fin 2))

-- Define the ratios
variable (h1 : D = (2 • A + B) / 3)
variable (h2 : E = (2 • B + C) / 3)

-- Define point F as the intersection of DE and AC
variable (F : EuclideanSpace ℝ (Fin 2))
variable (h3 : ∃ t : ℝ, F = D + t • (E - D))
variable (h4 : ∃ s : ℝ, F = A + s • (C - A))

-- Theorem statement
theorem ratio_DE_EF : 
  ‖E - D‖ / ‖F - E‖ = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_DE_EF_l109_10913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_books_eq_56_l109_10978

/-- The number of ways to select 5 books from 12 books in a row with no adjacent books chosen -/
def select_books : ℕ :=
  Finset.card (Finset.filter
    (fun t : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ => t.1 + t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2.1 + t.2.2.2.2.2 = 3)
    (Finset.product (Finset.range 4) (Finset.product (Finset.range 4) (Finset.product (Finset.range 4) (Finset.product (Finset.range 4) (Finset.product (Finset.range 4) (Finset.range 4)))))))

/-- The theorem stating that the number of ways to select 5 books from 12 books
    in a row with no adjacent books chosen is 56 -/
theorem select_books_eq_56 : select_books = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_books_eq_56_l109_10978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circles_distance_range_l109_10962

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x+3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x-3)^2 + y^2 = 4

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem ellipse_circles_distance_range :
  ∀ (px py mx my nx ny : ℝ),
  ellipse px py →
  circle1 mx my →
  circle2 nx ny →
  7 ≤ distance px py mx my + distance px py nx ny ∧
  distance px py mx my + distance px py nx ny ≤ 13 := by
  sorry

#check ellipse_circles_distance_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circles_distance_range_l109_10962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_in_interval_l109_10914

noncomputable def f (x : ℝ) := Real.sqrt 2 * Real.sin (2 * x - Real.pi / 3)

theorem f_max_min_in_interval :
  let a := Real.pi / 6
  let b := Real.pi / 2
  (∀ x, x ∈ Set.Icc a b → f x ≤ Real.sqrt 2) ∧
  (∃ x, x ∈ Set.Icc a b ∧ f x = Real.sqrt 2) ∧
  (∀ x, x ∈ Set.Icc a b → f x ≥ 0) ∧
  (∃ x, x ∈ Set.Icc a b ∧ f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_in_interval_l109_10914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_lambda_l109_10987

/-- Given vectors a, b, and c in ℝ², if a + λb is parallel to c, then λ = 1/2 -/
theorem parallel_vector_lambda (a b c : ℝ × ℝ) (lambda : ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (1, 0)) 
  (hc : c = (3, 4)) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + lambda • b = k • c) : 
  lambda = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_lambda_l109_10987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_properties_l109_10950

theorem divisibility_properties (n : ℕ) : 
  (∃ k : ℤ, (7 : ℤ)^(2*n) - (4 : ℤ)^(2*n) = 33*k) ∧ 
  (∃ m : ℤ, (3 : ℤ)^(6*n) - (2 : ℤ)^(6*n) = 35*m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_properties_l109_10950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l109_10930

/-- Given a geometric sequence {a_n} with first term a_1 = 1 and common ratio q ≠ 1,
    if a_m = a_1 * a_2 * a_3 * a_4 * a_5, then m = 11 -/
theorem geometric_sequence_product (q : ℝ) (h_q : q ≠ 1) :
  ∃ m : ℕ, (fun n : ℕ => q^(n-1)) m = (q^0) * (q^1) * (q^2) * (q^3) * (q^4) ∧ m = 11 :=
by
  -- We'll use 'm = 11' as our witness for the existential quantifier
  use 11
  constructor
  · -- Prove the first part of the conjunction
    simp [pow_sub, pow_zero]
    ring
  · -- Prove the second part of the conjunction
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l109_10930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l109_10921

open Set
open Real

noncomputable def f (x : ℝ) := 2 * sin (2 * x + π / 3)

theorem range_of_f :
  let S := {x : ℝ | -π/6 < x ∧ x < π/6}
  let R := {y : ℝ | ∃ x ∈ S, f x = y}
  R = Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l109_10921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l109_10941

noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin (θ - Real.pi / 4)

noncomputable def point_Q : ℝ × ℝ := (1, Real.pi / 4)

def circle_center : ℝ × ℝ := (-1, 1)

noncomputable def circle_radius : ℝ := Real.sqrt 2

noncomputable def shortest_distance : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem curve_properties :
  (∀ θ, (curve_C θ * Real.cos θ - circle_center.1)^2 + 
        (curve_C θ * Real.sin θ - circle_center.2)^2 = circle_radius^2) ∧
  (∀ P : ℝ × ℝ, (∃ θ, P.1 = curve_C θ * Real.cos θ ∧ P.2 = curve_C θ * Real.sin θ) →
    Real.sqrt ((P.1 - point_Q.1 * Real.cos point_Q.2)^2 + 
               (P.2 - point_Q.1 * Real.sin point_Q.2)^2) ≥ shortest_distance) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l109_10941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l109_10901

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola of the form y = ax² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The focal length of a hyperbola -/
noncomputable def focalLength (h : Hyperbola) : ℝ := 2 * Real.sqrt (h.a^2 + h.b^2)

/-- Checks if a parabola is tangent to an asymptote of a hyperbola -/
def isTangentToAsymptote (h : Hyperbola) (p : Parabola) : Prop :=
  ∃ (x : ℝ), p.a * x^2 + p.b = h.a / h.b * x

/-- The main theorem -/
theorem hyperbola_equation (h : Hyperbola) (p : Parabola) :
  focalLength h = 2 * Real.sqrt 5 →
  isTangentToAsymptote h p →
  p.a = 1/16 ∧ p.b = 1 →
  h.a^2 = 4 ∧ h.b^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l109_10901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_transformation_valid_l109_10943

noncomputable def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ 0 then -x
  else if 0 < x ∧ x ≤ 4 then x - 2
  else 0  -- default value for x outside [-4, 4]

noncomputable def h (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ 0 then -x/3 - 2
  else if 0 < x ∧ x ≤ 4 then x/3 - 8/3
  else 0  -- default value for x outside [-4, 4]

-- Theorem statement
theorem g_transformation_valid :
  ∀ x : ℝ, -4 ≤ x ∧ x ≤ 4 → h x = (1/3) * g x - 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_transformation_valid_l109_10943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l109_10992

/-- Given a line with equation 3x - 6y = 9, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope :
  ∀ (m : ℝ), (∀ (x y : ℝ), 3 * x - 6 * y = 9 → y = m * x + b) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l109_10992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solutions_l109_10969

/-- The functional equation that f must satisfy for all real x and y -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * y^2 * f y

/-- The set of all functions that satisfy the equation -/
def solution_set : Set (ℝ → ℝ) :=
  {f | satisfies_equation f}

/-- The zero function -/
def zero_func : ℝ → ℝ := λ x ↦ 0

/-- The fourth power function -/
def fourth_power : ℝ → ℝ := λ x ↦ x^4

theorem unique_solutions :
  solution_set = {zero_func, fourth_power} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solutions_l109_10969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_equality_l109_10996

-- Define the functions M and P
noncomputable def M (x : ℝ) : ℝ := 3 * Real.sqrt x
def P (x : ℝ) : ℝ := x^3

-- State the theorem
theorem nested_function_equality : 
  M (P (M (P (M (P 4))))) = 3 * Real.sqrt (372984 * 24 * Real.sqrt 24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_equality_l109_10996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_digit_is_six_l109_10906

def transform (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  let remainingNumber := n / 10
  4 * remainingNumber + lastDigit

def isEven (n : ℕ) : Prop := n % 2 = 0

def isDivisibleBy3 (n : ℕ) : Prop := n % 3 = 0

theorem final_digit_is_six :
  ∃ k : ℕ, 
    let initialNumber := 222222222
    let finalNumber := Nat.iterate transform k initialNumber
    finalNumber < 10 ∧ 
    isEven finalNumber ∧ 
    isDivisibleBy3 finalNumber ∧
    finalNumber = 6 :=
by
  -- The proof goes here
  sorry

#check final_digit_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_digit_is_six_l109_10906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_sum_sequence_formula_l109_10946

noncomputable def f (x : ℝ) : ℝ := x^2 + x + 1 - Real.log x

def tangent_line (x : ℝ) : ℝ := 2*x + 1

def a : ℕ → ℝ
  | 0 => 1
  | n+1 => tangent_line (a n)

def sum_sequence (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem sequence_formula (n : ℕ) :
  a n = 2^n - 1 := by sorry

theorem sum_sequence_formula (n : ℕ) :
  sum_sequence n = 2^(n+1) - 2 - n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_sum_sequence_formula_l109_10946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l109_10970

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define vectors a and b
variable (a b : V)

-- Define the scalar lambda
variable (lambda : ℝ)

-- Theorem statement
theorem parallel_vectors_lambda (h1 : ¬ ∃ (k : ℝ), a = k • b) 
  (h2 : ∃ (μ : ℝ), lambda • a + b = μ • (a + 2 • b)) : 
  lambda = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l109_10970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l109_10958

theorem triangle_angle_measure :
  ∀ (A B C a b c : Real),
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Angles are positive
  A + B + C = π ∧ -- Sum of angles in a triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- Sides are positive
  b^2 + c^2 - a^2 = b * c ∧ -- Given condition
  Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin C ^ 2 -- Given condition
  → B = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l109_10958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandths_digit_of_five_thirty_seconds_thousandths_digit_is_six_l109_10979

theorem thousandths_digit_of_five_thirty_seconds : 
  (5 : ℚ) / 32 = 0.15625 := by sorry

def thousandths_digit (q : ℚ) (n : ℕ) : ℕ :=
  ((q * 10^n).floor % 10).toNat

theorem thousandths_digit_is_six : 
  thousandths_digit ((5 : ℚ) / 32) 3 = 6 := by sorry

#eval thousandths_digit ((5 : ℚ) / 32) 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandths_digit_of_five_thirty_seconds_thousandths_digit_is_six_l109_10979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l109_10966

theorem inequality_proof (m n p q : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0) :
  let t := (m + n + p + q) / 2
  (m / (t + n + p + q)) + (n / (t + m + p + q)) + (p / (t + m + n + q)) + (q / (t + m + n + p)) ≥ 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l109_10966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_every_fourth_is_four_l109_10961

/-- A circular arrangement of 40 numbers where the sum of any four consecutive numbers is constant -/
def CircularArrangement := Fin 40 → ℕ

/-- The property that the sum of any four consecutive numbers in the arrangement is constant -/
def has_constant_sum (arr : CircularArrangement) : Prop :=
  ∃ (s : ℕ), ∀ (i : Fin 40), 
    arr i + arr (i + 1) + arr (i + 2) + arr (i + 3) = s

/-- The property that 3 and 4 are adjacent in the arrangement -/
def has_adjacent_3_4 (arr : CircularArrangement) : Prop :=
  ∃ (i : Fin 40), arr i = 3 ∧ arr (i + 1) = 4

/-- The theorem stating that under the given conditions, every 4th number must be 4 -/
theorem every_fourth_is_four (arr : CircularArrangement) 
  (h1 : has_constant_sum arr) (h2 : has_adjacent_3_4 arr) :
  ∀ (i : Fin 40), arr i = 4 ∨ arr (i + 1) = 4 ∨ arr (i + 2) = 4 ∨ arr (i + 3) = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_every_fourth_is_four_l109_10961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l109_10919

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 6*y - 15 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (4, -3)

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := Real.sqrt 40

theorem circle_center_and_radius :
  ∀ x y : ℝ, circle_equation x y ↔ 
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l109_10919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_properties_l109_10972

/-- The slope angle of the line 3x - 4y + 5 = 0 -/
noncomputable def α : ℝ := Real.arctan (3/4)

/-- Theorem stating the values of tan(2α) and cos(π/6 - α) for the given line -/
theorem line_angle_properties : 
  (Real.tan (2 * α) = 24/7) ∧ 
  (Real.cos (π/6 - α) = (3 + 4 * Real.sqrt 3) / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_properties_l109_10972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_satisfying_condition_unique_solution_to_equations_l109_10931

-- Part (a)
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem unique_prime_satisfying_condition :
  ∃! p : ℕ, is_prime p ∧ is_prime (4*p^2 + 1) ∧ is_prime (6*p^2 + 1) ∧ p = 5 :=
sorry

-- Part (b)
def satisfies_equations (x y z u : ℝ) : Prop :=
  x*y*z + x*y + y*z + z*x + x + y + z = 7 ∧
  y*z*u + y*z + z*u + u*y + y + z + u = 10 ∧
  z*u*x + z*u + u*x + x*z + z + u + x = 10 ∧
  u*x*y + u*x + x*y + y*u + u + x + y = 10

theorem unique_solution_to_equations :
  ∃! (x y z u : ℝ), satisfies_equations x y z u ∧ x = 1 ∧ y = 1 ∧ z = 1 ∧ u = 7/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_satisfying_condition_unique_solution_to_equations_l109_10931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_base_ratio_not_unique_l109_10911

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the ratio of an altitude to its corresponding base
noncomputable def altitude_base_ratio (t : Triangle) : ℝ := t.h_a / t.a

-- Theorem stating that the ratio of an altitude to its base does not uniquely determine the triangle's shape
theorem altitude_base_ratio_not_unique (r : ℝ) :
  ∃ t1 t2 : Triangle, altitude_base_ratio t1 = r ∧ altitude_base_ratio t2 = r ∧ t1 ≠ t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_base_ratio_not_unique_l109_10911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_triangle_max_perimeter_l109_10936

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Part 1
theorem triangle_trig_identity (t : Triangle) 
  (h1 : t.a * t.c = t.b^2)  -- Geometric progression
  (h2 : Real.cos t.B = 12/13) :
  (Real.cos t.A / Real.sin t.A) + (Real.cos t.C / Real.sin t.C) = 13/5 := by
sorry

-- Part 2
noncomputable def perimeter (α : Real) : Real := 4 * Real.sin (α + Real.pi/6) + 2

theorem triangle_max_perimeter (t : Triangle) (α : Real)
  (h1 : t.B = Real.pi/3)  -- Arithmetic progression of angles
  (h2 : t.b = 2)
  (h3 : t.A = α) :
  ∃ (max_α : Real), max_α = Real.pi/3 ∧ 
    ∀ (x : Real), 0 < x ∧ x < 2*Real.pi/3 → perimeter x ≤ perimeter max_α := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_triangle_max_perimeter_l109_10936
