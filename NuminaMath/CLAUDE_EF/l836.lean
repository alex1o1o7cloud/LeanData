import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l836_83648

/-- The range of slope angles for a line passing through two points and intersecting a unit circle -/
theorem slope_angle_range (P A : ℝ × ℝ) (h_P : P = (-Real.sqrt 3, -1)) (h_A : A = (-2, 0)) :
  let line := {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • P + t • A}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  (∃ (p : ℝ × ℝ), p ∈ line ∩ circle) →
  (∀ θ : ℝ, (θ ∈ Set.Icc 0 (Real.pi / 3) ↔ ∃ p q : ℝ × ℝ, p ∈ line ∧ q ∈ line ∧ p ≠ q ∧ θ = Real.arctan ((q.2 - p.2) / (q.1 - p.1)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l836_83648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_problem_l836_83686

theorem triangle_abc_problem (A B C : Real) (a b c : Real) :
  Real.cos (2 * C) = -1/4 →
  0 < C →
  C < π/2 →
  a = 2 →
  2 * Real.sin A = Real.sin C →
  Real.cos C = Real.sqrt 6 / 4 ∧ b = 2 * Real.sqrt 6 ∧ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_problem_l836_83686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_machines_production_l836_83633

/-- The number of shirts produced by a given number of machines in a given time -/
def shirts_produced (machines : ℕ) (minutes : ℕ) : ℕ := sorry

/-- The constant production rate of each machine (shirts per minute) -/
noncomputable def production_rate : ℝ := sorry

theorem sixteen_machines_production :
  -- Given conditions
  (shirts_produced 8 10 = 160) →
  (∀ m t, shirts_produced m t = ⌊m * t * production_rate⌋) →
  -- Theorem to prove
  shirts_produced 16 1 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_machines_production_l836_83633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_angle_range_l836_83677

-- Define the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the common focus
def common_focus (a b p : ℝ) : Prop := p = 2 * Real.sqrt (a^2 - b^2)

-- Define the condition that the common chord passes through the focus
def chord_through_focus (a b p : ℝ) : Prop :=
  ∃ x y : ℝ, hyperbola a b x y ∧ parabola p x y ∧ x = Real.sqrt (a^2 - b^2)

-- Define the angle formed by the asymptote
noncomputable def asymptote_angle (a b : ℝ) : ℝ := Real.arctan (b / a)

-- State the theorem
theorem asymptote_angle_range (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  (∃ x y, hyperbola a b x y) →
  (∃ x y, parabola p x y) →
  common_focus a b p →
  chord_through_focus a b p →
  π / 3 < asymptote_angle a b ∧ asymptote_angle a b < π / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_angle_range_l836_83677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l836_83660

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.exp (-x)

theorem f_properties :
  (∃! x₀ : ℝ, x₀ ∈ Set.Ioo (Real.pi/6) (Real.pi/4) ∧ (deriv f) x₀ = 0) ∧
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi ∧
    StrictMonoOn f (Set.Ioo 0 x₁) ∧
    StrictAntiOn f (Set.Ioo x₁ x₂) ∧
    StrictMonoOn f (Set.Ioo x₂ (2*Real.pi)) :=
by sorry

-- Additional information
axiom e_bounds : 7 < Real.exp 2 ∧ Real.exp 2 < 8 ∧ Real.exp 3 > 16
axiom e_neg_bound : Real.exp (-(3*Real.pi)/4) < Real.sqrt 2 / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l836_83660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_composition_equals_negative_three_l836_83667

-- Define the function p
noncomputable def p (x y : ℝ) : ℝ :=
  if x > 0 ∧ y ≥ 0 then x + 2*y
  else if x ≤ 0 ∧ y > 0 then 2*x - 3*y
  else if x < 0 ∧ y ≤ 0 then x^2 + 2*y
  else 2*x + y

-- State the theorem
theorem p_composition_equals_negative_three :
  p (p 2 (-1)) (p (-3) 1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_composition_equals_negative_three_l836_83667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l836_83635

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_equals_two (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 2 := by
  intro h
  -- The proof goes here
  sorry

#check even_function_implies_a_equals_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l836_83635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l836_83626

theorem complex_magnitude_problem : Complex.abs ((1 - 3*Complex.I) / (1 - Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l836_83626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_selections_count_l836_83615

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of derangements of n items -/
def derangement (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | n + 2 => n * (derangement (n + 1) + derangement n)

/-- The size of the square grid -/
def gridSize : ℕ := 6

/-- The number of blocks to select -/
def selectCount : ℕ := 4

/-- The number of ways to select 4 blocks from a 6x6 grid with given constraints -/
def validSelections : ℕ := 
  choose gridSize selectCount * choose gridSize selectCount * derangement selectCount

theorem valid_selections_count : validSelections = 2025 := by sorry

#eval validSelections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_selections_count_l836_83615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l836_83687

noncomputable def triangle_PQR (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = 8^2 ∧
  (px - rx)^2 + (py - ry)^2 = 15^2 ∧
  (qx - rx) * (px - rx) + (qy - ry) * (py - ry) = 0

noncomputable def point_S (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def point_T (P T : ℝ × ℝ) : Prop :=
  (P.1 - T.1)^2 + (P.2 - T.2)^2 = 15^2

noncomputable def area_TRS (T R S : ℝ × ℝ) : ℝ :=
  let a := ((T.1 - R.1)^2 + (T.2 - R.2)^2).sqrt
  let b := ((R.1 - S.1)^2 + (R.2 - S.2)^2).sqrt
  let c := ((S.1 - T.1)^2 + (S.2 - T.2)^2).sqrt
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

theorem triangle_area_theorem (P Q R T : ℝ × ℝ) :
  triangle_PQR P Q R →
  point_T P T →
  let S := point_S P Q
  area_TRS T R S = 2 * Real.sqrt 209 →
  ∃ (x y z : ℕ), x + y + z = 212 ∧
                 x > 0 ∧ y > 0 ∧ z > 0 ∧
                 Nat.Coprime x z ∧
                 ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ y) ∧
                 area_TRS T R S = (x * Real.sqrt y : ℝ) / z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l836_83687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l836_83650

theorem solve_exponential_equation (n : ℝ) : (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n = (81 : ℝ)^5 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l836_83650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_expression_l836_83640

theorem negative_expression : 
  (|(-2023 : ℤ)| > 0) ∧ 
  ((1 : ℚ) / 2023 > 0) ∧ 
  (-(-2023 : ℤ) > 0) ∧ 
  (-|(-2023 : ℤ)| < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_expression_l836_83640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_element_in_special_set_l836_83693

theorem least_element_in_special_set :
  ∃ (T : Finset ℕ),
    (Finset.card T = 7) ∧
    (∀ x, x ∈ T → x ∈ Finset.range 16 \ {0}) ∧
    (∀ c d, c ∈ T → d ∈ T → c < d → ¬(d % c = 0)) ∧
    (Finset.sum T id < 50) ∧
    (∀ x, x ∈ T → 4 ≤ x) ∧
    (4 ∈ T) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_element_in_special_set_l836_83693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_circular_configuration_l836_83658

/-- The area of the shaded region in a specific circular configuration -/
theorem shaded_area_in_circular_configuration : 
  (let r_large : ℝ := 10
   let r_small : ℝ := 4
   let area_large := π * r_large^2
   let area_small := π * r_small^2
   area_large - 2 * area_small) = 68 * π := by
  -- Unfold the let-bindings
  simp_all
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_circular_configuration_l836_83658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_xn_plus_one_eq_yn_plus_one_l836_83629

theorem no_solutions_xn_plus_one_eq_yn_plus_one (n : ℕ) (x y : ℕ+) :
  n ≥ 2 → Nat.Coprime x.val (n + 1) → x^n + 1 ≠ y^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_xn_plus_one_eq_yn_plus_one_l836_83629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_of_g_l836_83655

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := cos x - 8 * (cos (x / 4))^4

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ 
  T = 4 * π := by sorry

-- Define the function g(x) = f(2x - π/6)
noncomputable def g (x : ℝ) : ℝ := f (2 * x - π / 6)

-- Theorem for the range of g(x) on the given interval
theorem range_of_g :
  ∃ (a b : ℝ), a = -5 ∧ b = -4 ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → a ≤ g x ∧ g x ≤ b) ∧
  (∃ x₁ x₂, -π / 6 ≤ x₁ ∧ x₁ ≤ π / 4 ∧
             -π / 6 ≤ x₂ ∧ x₂ ≤ π / 4 ∧
             g x₁ = a ∧ g x₂ = b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_of_g_l836_83655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_eq_l836_83601

-- Define the reciprocal of the sum of x and y
noncomputable def reciprocal_sum (x y : ℝ) : ℝ := 1 / (x + y)

-- Theorem stating that the reciprocal of the sum of x and y is equal to 1 / (x + y)
theorem reciprocal_sum_eq (x y : ℝ) : reciprocal_sum x y = 1 / (x + y) := by
  -- Unfold the definition of reciprocal_sum
  unfold reciprocal_sum
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_eq_l836_83601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_point_l836_83627

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 2*x + 4*y - 1

-- Define the center of the circle
def circle_center : ℝ × ℝ :=
  (1, 2)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem distance_to_point :
  distance circle_center (13, 7) = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_point_l836_83627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batting_cage_charge_l836_83630

/-- The number of dozens for which the batting cage charges $30 -/
def charge_dozens : ℕ := sorry

/-- The number of dozens Dan buys -/
def dan_dozens : ℕ := 5

/-- The number of dozens Gus buys -/
def gus_dozens : ℕ := 2

/-- The number of golf balls Chris buys -/
def chris_balls : ℕ := 48

/-- The total number of golf balls purchased -/
def total_balls : ℕ := 132

/-- The number of golf balls in a dozen -/
def balls_per_dozen : ℕ := 12

theorem batting_cage_charge :
  charge_dozens = dan_dozens + gus_dozens + chris_balls / balls_per_dozen ∧
  charge_dozens = total_balls / balls_per_dozen :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batting_cage_charge_l836_83630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l836_83694

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (2*t, 1 + 4*t)

-- Define the circle C in polar coordinates
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin θ

-- Define the intersection property
def intersects (l : ℝ → ℝ × ℝ) (c : ℝ → ℝ) : Prop :=
  ∃ t θ, l t = (c θ * Real.cos θ, c θ * Real.sin θ)

-- Theorem statement
theorem line_circle_intersection :
  intersects line_l circle_C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l836_83694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2a_over_sin_pi_4_plus_a_l836_83684

theorem cos_2a_over_sin_pi_4_plus_a (α : ℝ) 
  (h1 : Real.cos (π/4 - α) = 12/13) 
  (h2 : 0 < α ∧ α < π/4) : 
  Real.cos (2*α) / Real.sin (π/4 + α) = 10/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2a_over_sin_pi_4_plus_a_l836_83684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_replacement_impossible_l836_83692

-- Define the tile types
inductive TileType
| Square  : TileType  -- 2x2 tile
| Rectangle : TileType  -- 1x4 tile

-- Implement BEq for TileType
instance : BEq TileType where
  beq a b := match a, b with
    | TileType.Square, TileType.Square => true
    | TileType.Rectangle, TileType.Rectangle => true
    | _, _ => false

-- Define a bathroom
structure Bathroom :=
  (width : ℕ)
  (height : ℕ)
  (tiles : List TileType)

-- Define a valid tiling
def isValidTiling (b : Bathroom) : Prop :=
  b.width * b.height = 4 * b.tiles.length ∧
  b.tiles.all (λ t => t == TileType.Square ∨ t == TileType.Rectangle)

-- Theorem: It's impossible to replace a 2x2 tile with a 1x4 tile or vice versa
theorem tile_replacement_impossible (b : Bathroom) (h : isValidTiling b) :
  ¬∃ (b' : Bathroom), isValidTiling b' ∧
    ((b'.tiles.count TileType.Square = b.tiles.count TileType.Square - 1 ∧
     b'.tiles.count TileType.Rectangle = b.tiles.count TileType.Rectangle + 1) ∨
    (b'.tiles.count TileType.Rectangle = b.tiles.count TileType.Rectangle - 1 ∧
     b'.tiles.count TileType.Square = b.tiles.count TileType.Square + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_replacement_impossible_l836_83692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l836_83678

noncomputable section

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 25

-- Define the line l
def l (x y a : ℝ) : Prop := y = -4/3 * x - a/3

-- Define the distance function from a point to a line
def dist_point_line (x y a : ℝ) : ℝ :=
  abs (4*x + 3*y + a) / 5

-- Theorem statement
theorem circle_line_intersection (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
    C x₁ y₁ ∧ C x₂ y₂ ∧ C x₃ y₃ ∧ C x₄ y₄ ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    dist_point_line x₁ y₁ a = 2 ∧
    dist_point_line x₂ y₂ a = 2 ∧
    dist_point_line x₃ y₃ a = 2 ∧
    dist_point_line x₄ y₄ a = 2) →
  a > -13 ∧ a < 17 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l836_83678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residuals_assess_fit_l836_83699

/-- Represents a residual in a statistical model. -/
structure Residual where
  value : ℝ

/-- Represents a statistical model. -/
structure Model where

/-- Assesses the fit of a model using residuals. -/
def assess_fit (model : Model) (residuals : List Residual) : Bool :=
  sorry -- Implementation details omitted for simplicity

/-- Theorem stating that residuals can be used to assess the fit of a model. -/
theorem residuals_assess_fit :
  ∃ (model : Model) (residuals : List Residual),
    assess_fit model residuals = true := by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_residuals_assess_fit_l836_83699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l836_83679

open Real

theorem expression_equality : 1.1^(0 : ℝ) + (216^(1/3 : ℝ)) - 0.5^((-2) : ℝ) + log 25 / log 10 + 2 * log 2 / log 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l836_83679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_slope_constant_l836_83606

/-- Ellipse C with equation x²/a² + y²/b² = 1 -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1 / a)^2 + (p.2 / b)^2 = 1}

/-- Line l with equation y = mx + c -/
def Line (m c : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + c}

/-- Point on the x-axis -/
def PointOnXAxis (x : ℝ) : ℝ × ℝ := (x, 0)

/-- Intersection of a line and the y-axis -/
noncomputable def IntersectionWithYAxis (l : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Intersections of a line and an ellipse -/
noncomputable def Intersections (l : Set (ℝ × ℝ)) (e : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

theorem ellipse_line_intersection_slope_constant
  (a b : ℝ)
  (h_a : a = 2)
  (h_b : b = 1)
  (h_a_gt_b : a > b)
  (h_b_pos : b > 0)
  (C : Set (ℝ × ℝ))
  (h_C : C = Ellipse a b)
  (M : ℝ × ℝ)
  (h_M : M.2 = 0)
  (l : Set (ℝ × ℝ))
  (m c : ℝ)
  (h_l : l = Line m c)
  (h_m_pos : m > 0)
  (N : ℝ × ℝ)
  (h_N : N = IntersectionWithYAxis l)
  (P Q : ℝ × ℝ)
  (h_PQ : {P, Q} = Intersections l C)
  (lambda mu : ℝ)
  (h_lambda_mu : lambda * mu = 1)
  (h_NP : (P.1 - N.1, P.2 - N.2) = lambda • (Q.1 - N.1, Q.2 - N.2))
  (h_MP : (P.1 - M.1, P.2 - M.2) = mu • (Q.1 - M.1, Q.2 - M.2))
  (h_distinct : N ≠ P ∧ N ≠ Q ∧ P ≠ Q ∧ M ≠ N ∧ M ≠ P ∧ M ≠ Q) :
  m = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_slope_constant_l836_83606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_change_l836_83638

theorem pyramid_volume_change (V₀ : ℝ) (l w h : ℝ) 
  (h₁ : V₀ = (1/3) * l * w * h)
  (h₂ : V₀ = 60) :
  (1/3) * (3*l) * (w/2) * (1.25*h) = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_change_l836_83638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l836_83637

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a rectangle with two opposite corners -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- The problem statement -/
theorem rectangle_area (p q r : Point) (circleP circleQ circleR : Circle) (rect : Rectangle) :
  -- Three congruent circles
  circleP.radius = circleQ.radius ∧ circleQ.radius = circleR.radius ∧
  -- Circles are tangent to the sides of the rectangle
  (circleP.center.x = rect.topLeft.x ∨ circleP.center.x = rect.bottomRight.x ∨
   circleP.center.y = rect.topLeft.y ∨ circleP.center.y = rect.bottomRight.y) ∧
  (circleQ.center.x = rect.topLeft.x ∨ circleQ.center.x = rect.bottomRight.x ∨
   circleQ.center.y = rect.topLeft.y ∨ circleQ.center.y = rect.bottomRight.y) ∧
  (circleR.center.x = rect.topLeft.x ∨ circleR.center.x = rect.bottomRight.x ∨
   circleR.center.y = rect.topLeft.y ∨ circleR.center.y = rect.bottomRight.y) ∧
  -- Circle Q has diameter 6
  circleQ.radius = 3 ∧
  -- Circle Q passes through P and R
  (p.x - q.x)^2 + (p.y - q.y)^2 = circleQ.radius^2 ∧
  (r.x - q.x)^2 + (r.y - q.y)^2 = circleQ.radius^2 →
  -- The area of the rectangle is 108
  (rect.bottomRight.x - rect.topLeft.x) * (rect.topLeft.y - rect.bottomRight.y) = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l836_83637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l836_83612

theorem tan_double_angle_special_case (α : ℝ) 
  (h : Real.sin α + Real.sqrt 3 * Real.cos α = 0) : 
  Real.tan (2 * α) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l836_83612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l836_83646

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 6*y = -9

-- Define the area of the region
noncomputable def region_area : ℝ := 7 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l836_83646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_g_of_3_l836_83653

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x⁻¹ - x⁻¹ / (1 - x⁻¹)

-- State the theorem
theorem g_of_g_of_3 : g (g 3) = -36 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_g_of_3_l836_83653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OPA_l836_83696

/-- Given a line y = kx + 4 passing through P(1, m) and parallel to y = -2x + 1,
    intersecting the x-axis at A, prove that the area of triangle OPA is 2. -/
theorem area_of_triangle_OPA (k m : ℝ) : 
  (∃ (x : ℝ), k * x + 4 = 0) →  -- Line intersects x-axis
  (k * 1 + 4 = m) →             -- Line passes through P(1, m)
  (k = -2) →                    -- Line is parallel to y = -2x + 1
  (∃ (A : ℝ × ℝ), A.1 > 0 ∧ A.2 = 0 ∧ k * A.1 + 4 = A.2) →  -- A is on positive x-axis
  (∃ (area : ℝ), area = 2 ∧ 
    area = (1/2) * (4/k) * 2) :=  -- Area formula: (1/2) * base * height
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OPA_l836_83696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_cos4_l836_83645

theorem min_sin4_cos4 (α : Real) (h : 0 ≤ α ∧ α ≤ Real.pi / 2) :
  Real.sin α ^ 4 + Real.cos α ^ 4 ≥ 1 / 2 ∧
  (Real.sin α ^ 4 + Real.cos α ^ 4 = 1 / 2 ↔ α = Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_cos4_l836_83645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_problem_l836_83636

/-- The distance between two points A and B --/
def distance : ℝ := sorry

/-- The speed of the first person --/
def speed1 : ℝ := sorry

/-- The speed of the second person --/
def speed2 : ℝ := sorry

/-- The distance between the first and second meeting points --/
def meeting_distance : ℝ := sorry

theorem walking_problem (h1 : speed1 / speed2 = 5 / 3)
                        (h2 : meeting_distance = 50) :
  distance = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_problem_l836_83636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_N2O5_approx_l836_83613

/-- The atomic mass of nitrogen in g/mol -/
noncomputable def atomic_mass_N : ℝ := 14.01

/-- The atomic mass of oxygen in g/mol -/
noncomputable def atomic_mass_O : ℝ := 16.00

/-- The number of nitrogen atoms in N2O5 -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in N2O5 -/
def num_O : ℕ := 5

/-- The molar mass of N2O5 in g/mol -/
noncomputable def molar_mass_N2O5 : ℝ := num_N * atomic_mass_N + num_O * atomic_mass_O

/-- The mass of oxygen in one mole of N2O5 in g -/
noncomputable def mass_O_in_N2O5 : ℝ := num_O * atomic_mass_O

/-- The mass percentage of oxygen in N2O5 -/
noncomputable def mass_percentage_O_in_N2O5 : ℝ := (mass_O_in_N2O5 / molar_mass_N2O5) * 100

/-- Theorem stating that the mass percentage of oxygen in N2O5 is approximately 74.06% -/
theorem mass_percentage_O_in_N2O5_approx :
  ∃ ε > 0, |mass_percentage_O_in_N2O5 - 74.06| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_N2O5_approx_l836_83613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_hemisphere_radius_l836_83695

theorem sphere_hemisphere_radius (R r : ℝ) (h1 : (4/3) * Real.pi * R^3 = (1/3) * Real.pi * r^3) (h2 : r = 4 * (2^(1/3))) : R = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_hemisphere_radius_l836_83695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_first_term_l836_83697

/-- Given an arithmetic sequence with common difference d and first term a₁,
    S_n represents the sum of the first n terms -/
noncomputable def S_n (n : ℕ) (a₁ d : ℝ) : ℝ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem: In an arithmetic sequence with common difference 2 and sum of first 10 terms equal to 100,
    the first term is equal to 1 -/
theorem arithmetic_sequence_first_term :
  ∀ a₁ : ℝ, S_n 10 a₁ 2 = 100 → a₁ = 1 := by
  intro a₁ h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_first_term_l836_83697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_b_50_value_proof_l836_83628

-- Define the sequence b
noncomputable def b : ℕ → ℝ
  | 0 => 1
  | (n + 1) => (121 ^ (1/4 : ℝ)) * b n

-- Theorem statement
theorem b_50_value : b 50 = 121 ^ (49/4 : ℝ) := by
  sorry

-- Lemma to prove the general formula
lemma b_general_formula (n : ℕ) : b n = (121 ^ (1/4 : ℝ)) ^ (n - 1) := by
  sorry

-- Theorem to prove the correctness of b_50_value using the general formula
theorem b_50_value_proof : b 50 = 121 ^ (49/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_b_50_value_proof_l836_83628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_seven_to_sixth_l836_83664

theorem cube_root_seven_to_sixth : (7 : ℝ) ^ (1/3) ^ 6 = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_seven_to_sixth_l836_83664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_percentage_change_is_22_percent_l836_83618

/-- Represents an item with old and new prices -/
structure Item where
  oldPrice : ℚ
  newPrice : ℚ

/-- Calculates the percentage change for an item -/
def percentageChange (item : Item) : ℚ :=
  (item.newPrice - item.oldPrice) / item.oldPrice * 100

/-- Theorem: The average percentage change across five items is 22% -/
theorem average_percentage_change_is_22_percent 
  (book : Item) 
  (laptop : Item) 
  (videoGame : Item) 
  (desk : Item) 
  (chair : Item) 
  (h1 : book.oldPrice = 300 ∧ book.newPrice = 450)
  (h2 : laptop.oldPrice = 800 ∧ laptop.newPrice = 1200)
  (h3 : videoGame.oldPrice = 50 ∧ videoGame.newPrice = 75)
  (h4 : desk.oldPrice = 250 ∧ desk.newPrice = 200)
  (h5 : chair.oldPrice = 100 ∧ chair.newPrice = 80) :
  (percentageChange book + percentageChange laptop + percentageChange videoGame + 
   percentageChange desk + percentageChange chair) / 5 = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_percentage_change_is_22_percent_l836_83618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_exists_l836_83621

def S (m n : ℕ) : Set (ℤ × ℤ) :=
  {p | ∃ (i j : ℤ), p = (i, j) ∧ 0 ≤ i ∧ i ≤ m ∧ 0 ≤ j ∧ j ≤ n}

-- Define the area_of_triangle function
def area_of_triangle (a b c : ℤ × ℤ) : ℚ :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  let (x₃, y₃) := c
  abs ((x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)) : ℚ) / 2

theorem triangle_area_exists (m n k : ℕ) (h : k ≤ m * n) :
  ∃ (a b c : ℤ × ℤ), a ∈ S m n ∧ b ∈ S m n ∧ c ∈ S m n ∧
    area_of_triangle a b c = k / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_exists_l836_83621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_can_prevent_win_l836_83616

/-- Represents a cell on the infinite grid -/
structure Cell where
  x : ℤ
  y : ℤ
deriving BEq, DecidableEq

/-- Represents a player in the game -/
inductive Player
| A
| B

/-- Represents the state of a cell -/
inductive CellState
| Empty
| X
| O
deriving BEq, DecidableEq

/-- Represents the game board -/
def GameBoard := Cell → CellState

/-- Represents a winning line (row, column, or diagonal) -/
def WinningLine := List Cell

/-- A strategy for Player B -/
def Strategy := GameBoard → Cell

/-- Checks if a winning line is filled with X -/
def is_winning_line (board : GameBoard) (line : WinningLine) : Prop :=
  line.length = 11 ∧ line.all (λ cell => board cell == CellState.X)

/-- The main theorem stating that Player B can always prevent Player A from winning -/
theorem player_b_can_prevent_win :
  ∃ (strategy : Strategy),
    ∀ (game : List Cell),
      (game.length % 2 = 0) →  -- It's Player B's turn
      ¬∃ (winning_line : WinningLine),
        is_winning_line
          (game.foldl
            (λ board cell =>
              if game.indexOf cell % 2 = 0
              then Function.update board cell CellState.X
              else Function.update board cell CellState.O)
            (λ _ => CellState.Empty))
          winning_line :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_can_prevent_win_l836_83616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_completion_time_specific_joint_completion_time_l836_83674

/-- Given two workers with individual completion times, calculate their joint completion time -/
theorem joint_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / (1/a + 1/b)) = (a * b) / (a + b) := by
  sorry

/-- Specific case for workers completing in 10 and 9 days -/
theorem specific_joint_completion_time :
  (1 / (1/10 + 1/9)) = 90 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_completion_time_specific_joint_completion_time_l836_83674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_midpoint_l836_83642

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with radius r -/
structure Circle where
  r : ℝ
  h_pos : 0 < r

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The length of the minor axis of an ellipse -/
def minor_axis_length (e : Ellipse) : ℝ := 2 * e.b

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The main theorem -/
theorem max_distance_midpoint (e : Ellipse) (c : Circle) :
  eccentricity e = Real.sqrt 3 / 2 →
  minor_axis_length e = 2 →
  c.r = 1 →
  ∃ (max_dist : ℝ), max_dist = 5/4 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      (x₁^2 / e.a^2 + y₁^2 / e.b^2 = 1) →
      (x₂^2 / e.a^2 + y₂^2 / e.b^2 = 1) →
      (∃ (m k : ℝ), (x₁ = m * y₁ + k) ∧ (x₂ = m * y₂ + k) ∧ (k^2 = m^2 + 1)) →
      distance 0 0 ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) ≤ max_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_midpoint_l836_83642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circles_area_ratio_l836_83617

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a circle is tangent to a line segment -/
def isTangent (c : Circle) (p1 p2 : ℝ × ℝ) : Prop := sorry

/-- The area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ := Real.pi * c.radius ^ 2

/-- The theorem statement -/
theorem octagon_circles_area_ratio
  (octagon : RegularOctagon)
  (circle1 circle2 : Circle) :
  isTangent circle1 (octagon.vertices 0) (octagon.vertices 1) →
  isTangent circle1 (octagon.vertices 2) (octagon.vertices 3) →
  isTangent circle1 (octagon.vertices 1) (octagon.vertices 2) →
  isTangent circle2 (octagon.vertices 0) (octagon.vertices 1) →
  isTangent circle2 (octagon.vertices 2) (octagon.vertices 3) →
  isTangent circle2 (octagon.vertices 3) (octagon.vertices 4) →
  circleArea circle2 / circleArea circle1 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circles_area_ratio_l836_83617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intuitive_diagram_area_l836_83647

theorem intuitive_diagram_area (side_length : ℝ) (area_ratio : ℝ) : 
  side_length = 2 →
  area_ratio = 2 * Real.sqrt 2 →
  (Real.sqrt 3 / area_ratio) = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intuitive_diagram_area_l836_83647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_shopping_popularity_l836_83634

/-- Represents the benefits of group shopping -/
structure GroupShoppingBenefits where
  cost_savings : ℝ
  quality_assessment : ℝ
  community_trust : ℝ

/-- Represents the risks associated with group shopping -/
def group_shopping_risks : ℝ := 1

/-- Calculates the total benefit of group shopping -/
def total_benefit (b : GroupShoppingBenefits) : ℝ :=
  b.cost_savings + b.quality_assessment + b.community_trust

/-- Theorem stating that group shopping is popular because its benefits outweigh the risks -/
theorem group_shopping_popularity (b : GroupShoppingBenefits) :
  total_benefit b > group_shopping_risks →
  (popularity : Prop) :=
by
  intro h
  sorry

/-- Example demonstrating the popularity of group shopping -/
def group_shopping_example : GroupShoppingBenefits :=
  { cost_savings := 0.5
  , quality_assessment := 0.3
  , community_trust := 0.3 }

#eval total_benefit group_shopping_example
#eval group_shopping_risks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_shopping_popularity_l836_83634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_crumble_amount_l836_83649

/-- Represents the ingredients in grams --/
structure Ingredients where
  flour : ℚ
  butter : ℚ
  sugar : ℚ

/-- The original recipe --/
def original_recipe : Ingredients := ⟨100, 50, 50⟩

/-- The scaling factor for family meal --/
def family_scale_factor : ℚ := 5/2

/-- Calculates the total amount of ingredients in grams --/
def total_grams (i : Ingredients) : ℚ :=
  i.flour + i.butter + i.sugar

/-- Converts grams to kilograms --/
def grams_to_kg (g : ℚ) : ℚ :=
  g / 1000

/-- Theorem: The total amount of crumble topping for a family meal is 0.5kg --/
theorem family_crumble_amount :
  grams_to_kg (family_scale_factor * total_grams original_recipe) = 1/2 := by
  -- Expand definitions
  unfold grams_to_kg
  unfold family_scale_factor
  unfold total_grams
  unfold original_recipe
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

#eval grams_to_kg (family_scale_factor * total_grams original_recipe)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_crumble_amount_l836_83649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_perpendicular_distances_l836_83685

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- A line in a 2D plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The perpendicular distance from a point to a line -/
noncomputable def perp_distance (p : ℝ × ℝ) (l : Line) : ℝ :=
  abs (l.slope * p.1 - p.2 + l.intercept) / Real.sqrt (l.slope^2 + 1)

/-- The theorem statement -/
theorem centroid_perpendicular_distances (t : Triangle) (l : Line) :
  let S := centroid t
  let A₁ := perp_distance t.A l
  let B₁ := perp_distance t.B l
  let C₁ := perp_distance t.C l
  (S.1 * l.slope + S.2 = l.intercept) → C₁ = A₁ + B₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_perpendicular_distances_l836_83685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l836_83644

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt (4 + 1 / x^2)

def sequence_a : ℕ+ → ℝ := sorry

def sequence_b : ℕ+ → ℝ := sorry

def S : ℕ+ → ℝ := sorry

def T : ℕ+ → ℝ := sorry

theorem sequence_properties 
  (h1 : ∀ n : ℕ+, f (sequence_a n) = -(1 / sequence_a (n + 1)))
  (h2 : sequence_a 1 = 1)
  (h3 : ∀ n : ℕ+, sequence_a n > 0)
  (h4 : ∀ n : ℕ+, T (n + 1) / (sequence_a n)^2 = T n / (sequence_a (n + 1))^2 + 16 * n^2 - 8 * n - 3) :
  (∀ n : ℕ+, sequence_a n = 1 / Real.sqrt (4 * n - 3)) ∧ 
  (sequence_b 1 = 1 ∧ ∀ n : ℕ+, sequence_b n = 8 * n - 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l836_83644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_from_four_cards_l836_83609

theorem three_digit_numbers_from_four_cards (cards : Finset ℕ) : 
  cards.card = 4 → (∀ n ∈ cards, n < 10) → 
  (cards.toList.permutations.map (λ l => l.take 3)).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_from_four_cards_l836_83609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_marked_cells_possible_l836_83610

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 2010
  col : Fin 2010

/-- Represents an L-shaped piece in the grid -/
structure LPiece where
  cells : Fin 3 → Cell

/-- Represents a marking of cells in the grid -/
def Marking := Cell → Bool

/-- The size of the grid -/
def gridSize : Nat := 2010

theorem equal_marked_cells_possible :
  ∃ (marking : Marking) (grid : Fin gridSize → Fin gridSize → LPiece),
    (∀ piece : LPiece, ∃ i : Fin 3, marking (piece.cells i)) ∧
    (∀ row : Fin gridSize, (Finset.univ.sum fun col => if marking ⟨row, col⟩ then 1 else 0) = gridSize / 3) ∧
    (∀ col : Fin gridSize, (Finset.univ.sum fun row => if marking ⟨row, col⟩ then 1 else 0) = gridSize / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_marked_cells_possible_l836_83610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_l836_83681

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := Real.log x + 1

-- Theorem statement
theorem f_monotonically_decreasing :
  ∀ x y : ℝ, 0 < x → x < Real.exp (-1) → 
    0 < y → y < Real.exp (-1) → x < y → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_l836_83681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_solid_volume_l836_83625

/-- A planar convex figure. -/
structure ConvexFigure where
  perimeter : ℝ
  area : ℝ

/-- The solid formed by extending a planar convex figure by a distance in all directions. -/
noncomputable def ExtendedSolid (figure : ConvexFigure) (d : ℝ) : ℝ :=
  2 * d * figure.area + Real.pi * (figure.perimeter / 2) * d^2 + (4 / 3) * Real.pi * d^3

/-- The volume of the extended solid is 2dS + πpd² + (4/3)πd³ -/
theorem extended_solid_volume (figure : ConvexFigure) (d : ℝ) :
  ExtendedSolid figure d = 2 * d * figure.area + Real.pi * (figure.perimeter / 2) * d^2 + (4 / 3) * Real.pi * d^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_solid_volume_l836_83625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_parallel_to_polar_axis_l836_83608

-- Define the point in polar coordinates
noncomputable def polar_point : ℝ × ℝ := (-3, -Real.pi/2)

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 3

-- Theorem statement
theorem line_through_point_parallel_to_polar_axis :
  -- The polar equation represents a line
  -- that passes through the given point
  polar_equation polar_point.1 polar_point.2 ∧
  -- and is parallel to the polar axis
  (∀ θ : ℝ, ∃ ρ : ℝ, polar_equation ρ θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_parallel_to_polar_axis_l836_83608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l836_83632

/-- The parabola C with focus F and directrix -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point -/
noncomputable def O : Point := ⟨0, 0⟩

/-- The focus of the parabola -/
noncomputable def F (c : Parabola) : Point := ⟨c.p / 2, 0⟩

/-- A point on the parabola -/
noncomputable def P (c : Parabola) : Point := ⟨c.p / 2, c.p⟩

/-- A point on the x-axis -/
noncomputable def Q (c : Parabola) : Point := ⟨5 * c.p / 2, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem stating the directrix equation of the parabola -/
theorem parabola_directrix (c : Parabola) 
  (h1 : P c ∈ {p : Point | p.y^2 = 2 * c.p * p.x}) -- P is on the parabola
  (h2 : (P c).y = c.p) -- PF is perpendicular to x-axis
  (h3 : ((P c).y - O.y) / ((P c).x - O.x) * ((Q c).y - (P c).y) / ((Q c).x - (P c).x) = -1) -- PQ ⟂ OP
  (h4 : distance (F c) (Q c) = 6) -- |FQ| = 6
  : {x : ℝ | x = -3/2} = {x : ℝ | x = -c.p/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l836_83632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_range_l836_83698

noncomputable def g (x : ℝ) : ℝ := (5 * x + 3) / (x + 1)

noncomputable def h (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m + 1

theorem symmetry_and_range :
  -- Part 1: Symmetry of g
  (∀ x : ℝ, x ≠ -1 → g x + g (-2 - x) = 10) ∧
  -- Part 2: Range of m
  (∃ m_min m_max : ℝ,
    m_min = -1 ∧ m_max = 3 ∧
    (∀ m : ℝ,
      -- h is symmetric with respect to (1,2)
      (∀ x : ℝ, h m x + h m (2 - x) = 4) →
      -- For any x₁ in [0,2], there exists x₂ in [-2/3,1] such that h(x₁) = g(x₂)
      (∀ x₁ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 2 →
        ∃ x₂ : ℝ, -2/3 ≤ x₂ ∧ x₂ ≤ 1 ∧ h m x₁ = g x₂) →
      -- m is in the range [m_min, m_max]
      m_min ≤ m ∧ m ≤ m_max)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_range_l836_83698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_half_l836_83672

/-- The sum of the infinite series ∑(k=1 to ∞) 3^k / (9^k - 1) -/
noncomputable def series_sum : ℝ := ∑' k, (3 : ℝ)^k / (9^k - 1)

/-- Theorem: The sum of the infinite series ∑(k=1 to ∞) 3^k / (9^k - 1) is equal to 1/2 -/
theorem series_sum_equals_half : series_sum = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_half_l836_83672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l836_83682

/-- The area of a triangle with vertices A(3, 2), B(9, 2), and C(5, 10) is 24 square units -/
theorem triangle_area : ∃ (area : ℝ), area = 24 := by
  let A : ℝ × ℝ := (3, 2)
  let B : ℝ × ℝ := (9, 2)
  let C : ℝ × ℝ := (5, 10)
  let area := (1 / 2) * abs ((A.1 - C.1) * (B.2 - A.2) - (B.1 - A.1) * (C.2 - A.2))
  exists area
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l836_83682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_roll_trajectory_l836_83605

-- Define the edge length of the tetrahedron
def edge_length : ℝ := 6

-- Define the radius of the circle through which the vertex moves
noncomputable def circle_radius : ℝ := 3 * Real.sqrt 3

-- Define the central angle of each roll
noncomputable def central_angle : ℝ := Real.pi - Real.arccos (1/3)

-- Define the length of a single arc
noncomputable def single_arc_length : ℝ := circle_radius * central_angle

-- Define the number of distinct arcs (due to symmetry)
def num_arcs : ℕ := 4

-- Theorem statement
theorem tetrahedron_roll_trajectory :
  (num_arcs : ℝ) * single_arc_length = 12 * Real.sqrt 3 * (Real.pi - Real.arccos (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_roll_trajectory_l836_83605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_third_l836_83676

/-- Represents a frequency distribution interval -/
structure FrequencyInterval where
  lower : ℝ
  upper : ℝ
  frequency : ℕ

/-- The sample data -/
def sampleData : List FrequencyInterval := [
  ⟨11.5, 15.5, 2⟩,
  ⟨15.5, 19.5, 4⟩,
  ⟨19.5, 23.5, 9⟩,
  ⟨23.5, 27.5, 18⟩,
  ⟨27.5, 31.5, 11⟩,
  ⟨31.5, 35.5, 12⟩,
  ⟨35.5, 39.5, 7⟩,
  ⟨39.5, 43.5, 3⟩
]

/-- The total sample size -/
def sampleSize : ℕ := 66

/-- The probability of data falling in the interval [31.5, 43.5) -/
noncomputable def probabilityInInterval : ℚ :=
  let intervalSum := (sampleData.filter (fun i => i.lower ≥ 31.5 && i.upper < 43.5)).map (·.frequency) |>.sum
  intervalSum / sampleSize

/-- Theorem: The probability of data falling in the interval [31.5, 43.5) is 1/3 -/
theorem probability_is_one_third : probabilityInInterval = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_third_l836_83676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_engine_efficiency_l836_83688

/-- A heat engine cycle with specific properties -/
structure HeatEngineCycle where
  -- The working substance is an ideal monoatomic gas
  is_ideal_monoatomic : Bool
  -- The maximum temperature is twice the minimum temperature
  temp_ratio : ℝ
  -- The cycle consists of isobaric, isothermal, and pressure-volume proportional processes
  has_required_processes : Bool

/-- The efficiency of a heat engine cycle -/
noncomputable def efficiency (cycle : HeatEngineCycle) : ℝ :=
  (1 - Real.log 2) / 5

/-- Theorem stating the efficiency of the specific heat engine cycle -/
theorem heat_engine_efficiency (cycle : HeatEngineCycle) 
  (h1 : cycle.is_ideal_monoatomic = true)
  (h2 : cycle.temp_ratio = 2)
  (h3 : cycle.has_required_processes = true) :
  efficiency cycle = (1 - Real.log 2) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_engine_efficiency_l836_83688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_xy_eq_neg_two_l836_83624

/-- Two terms are like terms if they have the same variables with the same exponents -/
def like_terms (term1 term2 : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b, ∃ c₁ c₂, term1 a b = c₁ * (a ^ (term1 1 1)) * (b ^ (term1 1 2)) ∧
                  term2 a b = c₂ * (a ^ (term2 1 1)) * (b ^ (term2 1 2)) ∧
                  term1 1 1 = term2 1 1 ∧ term1 1 2 = term2 1 2

/-- If 2a^(3x)b^(y+5) and 5a^(2-4y)b^(2x) are like terms, then xy = -2 -/
theorem like_terms_imply_xy_eq_neg_two :
  ∀ x y : ℝ, like_terms (λ a b ↦ 2 * (a ^ (3 * x)) * (b ^ (y + 5)))
                        (λ a b ↦ 5 * (a ^ (2 - 4 * y)) * (b ^ (2 * x)))
  → x * y = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_xy_eq_neg_two_l836_83624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_product_bound_l836_83690

theorem consecutive_product_bound (p : Fin 90 → Fin 90) (h : Function.Bijective p) :
  ∃ i : Fin 90, (p i).val * (p (i.succ)).val ≥ 2014 ∨
                (p (Fin.last 89)).val * (p 0).val ≥ 2014 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_product_bound_l836_83690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l836_83680

/-- A point in the xy-plane with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- The set of valid points in the 5x5 grid -/
def valid_points : Set Point :=
  {p : Point | 1 ≤ p.x ∧ p.x ≤ 5 ∧ 1 ≤ p.y ∧ p.y ≤ 5}

/-- A triangle represented by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Check if a triangle has positive area -/
def positive_area (t : Triangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of triangles with positive area and vertices in the valid points -/
def valid_triangles : Set Triangle :=
  {t : Triangle | t.p1 ∈ valid_points ∧ t.p2 ∈ valid_points ∧ t.p3 ∈ valid_points ∧ positive_area t}

-- Assuming valid_triangles is finite
instance : Fintype valid_triangles := sorry

theorem count_valid_triangles : Fintype.card valid_triangles = 2148 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l836_83680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_range_of_t_l836_83652

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x

-- Define the monotonicity interval
def monotonicity_interval (k : ℤ) : Set ℝ := 
  Set.Icc (-Real.pi / 12 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi)

-- Theorem for monotonicity
theorem f_monotone_increasing (k : ℤ) : 
  StrictMonoOn f (monotonicity_interval k) := by sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) :
  (∀ x ∈ Set.Icc t (Real.pi / 3), |f x - Real.sqrt 3 / 2| ≤ Real.sqrt 3 / 2) ↔ 
  t ∈ Set.Ico 0 (Real.pi / 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_range_of_t_l836_83652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_double_is_13_169_l836_83619

/-- Represents a domino set with integers from 0 to 12 -/
def DominoSet := Finset (Fin 13 × Fin 13)

/-- The complete domino set where each integer pairs with every other integer exactly once -/
def completeDominoSet : DominoSet :=
  Finset.filter (fun p => p.1 ≤ p.2) (Finset.product (Finset.univ : Finset (Fin 13)) (Finset.univ : Finset (Fin 13)))

/-- A double is a domino where both numbers are the same -/
def isDouble (d : Fin 13 × Fin 13) : Prop := d.1 = d.2

/-- The set of all doubles in the complete domino set -/
def doubleSet : DominoSet := Finset.filter (fun d => d.1 = d.2) completeDominoSet

/-- The probability of selecting a double from the complete domino set -/
noncomputable def probabilityOfDouble : ℚ :=
  (doubleSet.card : ℚ) / (completeDominoSet.card : ℚ)

theorem probability_of_double_is_13_169 : probabilityOfDouble = 13 / 169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_double_is_13_169_l836_83619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_source_height_l836_83639

-- Define the cube edge length
def cube_edge : ℚ := 2

-- Define the shadow area (excluding area beneath the cube)
def shadow_area : ℚ := 200

-- Define the function to calculate x based on shadow area
noncomputable def calculate_x (cube_edge : ℚ) (shadow_area : ℚ) : ℝ :=
  (4 * cube_edge) / (Real.sqrt (shadow_area + cube_edge^2) - cube_edge)

-- Theorem statement
theorem light_source_height (x : ℝ) :
  x = calculate_x cube_edge shadow_area →
  Int.floor (100 * x) = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_source_height_l836_83639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_special_numbers_l836_83620

theorem order_of_special_numbers :
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.log 3 / Real.log π
  let c : ℝ := -Real.log 3 / Real.log 2
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_special_numbers_l836_83620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_range_l836_83663

-- Define the set of valid values for 'a'
def valid_a_set : Set ℝ := {a | a > -1 ∧ a ≠ 0}

-- State the theorem
theorem log_base_range :
  ∀ x y : ℝ, (∃ a : ℝ, y = Real.log x / Real.log (a + 1) ∧ (a + 1) > 0 ∧ (a + 1) ≠ 1) ↔ ∃ a ∈ valid_a_set, y = Real.log x / Real.log (a + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_range_l836_83663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_f_geq_one_m_range_l836_83654

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.sin x + Real.cos x * Real.sin x

-- Theorem for the axis of symmetry
theorem axis_of_symmetry :
  ∀ k : ℤ, ∃ x : ℝ, f x = f (x + π) ∧ x = k * π / 2 + 3 * π / 8 :=
by sorry

-- Theorem for the set of values where f(x) ≥ 1
theorem f_geq_one :
  ∀ x : ℝ, f x ≥ 1 ↔ ∃ k : ℤ, π / 4 + k * π ≤ x ∧ x ≤ π / 2 + k * π :=
by sorry

-- Theorem for the range of m
theorem m_range :
  ∀ m : ℝ, (∀ x : ℝ, π / 6 ≤ x ∧ x ≤ π / 3 → f x - m < 2) → 
  m > (Real.sqrt 3 - 5) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_f_geq_one_m_range_l836_83654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l836_83657

noncomputable def f (x : ℝ) := Real.sin (2 * (x - Real.pi / 10) + Real.pi / 5)

theorem f_monotone_increasing :
  MonotoneOn f (Set.Icc (3 * Real.pi / 4) (5 * Real.pi / 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l836_83657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_sum_product_l836_83600

theorem not_prime_sum_product (a b c d : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → ¬(Nat.Prime (a * b + b * c + c * d + d * a)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_sum_product_l836_83600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_value_exists_f_with_M_eight_l836_83604

/-- The set A = {1, 2, 3, ..., 17} -/
def A : Finset ℕ := Finset.range 17

/-- Definition of f^[k] -/
def iterateF (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, x => x
| 1, x => f x
| (k+1), x => f (iterateF f k x)

/-- The main theorem -/
theorem max_M_value (f : ℕ → ℕ) (hf : Function.Bijective f) (M : ℕ) :
  (∀ m i, m < M ∧ i ∈ A → 
    (iterateF f m (i+1) - iterateF f m i) % 17 ≠ 1 ∧
    (iterateF f m (i+1) - iterateF f m i) % 17 ≠ 16 ∧
    (iterateF f m 1 - iterateF f m 17) % 17 ≠ 1 ∧
    (iterateF f m 1 - iterateF f m 17) % 17 ≠ 16) →
  (∀ i, i ∈ A → 
    ((iterateF f M (i+1) - iterateF f M i) % 17 = 1 ∨
     (iterateF f M (i+1) - iterateF f M i) % 17 = 16) ∧
    ((iterateF f M 1 - iterateF f M 17) % 17 = 1 ∨
     (iterateF f M 1 - iterateF f M 17) % 17 = 16)) →
  M ≤ 8 := by
  sorry

/-- There exists a function f that achieves M = 8 -/
theorem exists_f_with_M_eight :
  ∃ f : ℕ → ℕ, Function.Bijective f ∧
  (∀ m i, m < 8 ∧ i ∈ A → 
    (iterateF f m (i+1) - iterateF f m i) % 17 ≠ 1 ∧
    (iterateF f m (i+1) - iterateF f m i) % 17 ≠ 16 ∧
    (iterateF f m 1 - iterateF f m 17) % 17 ≠ 1 ∧
    (iterateF f m 1 - iterateF f m 17) % 17 ≠ 16) ∧
  (∀ i, i ∈ A → 
    ((iterateF f 8 (i+1) - iterateF f 8 i) % 17 = 1 ∨
     (iterateF f 8 (i+1) - iterateF f 8 i) % 17 = 16) ∧
    ((iterateF f 8 1 - iterateF f 8 17) % 17 = 1 ∨
     (iterateF f 8 1 - iterateF f 8 17) % 17 = 16)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_value_exists_f_with_M_eight_l836_83604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_triangle_perimeter_range_l836_83659

-- Problem 1
theorem tangent_line_slope (f : ℝ → ℝ) (α : ℝ) :
  (f = λ x ↦ 2*x + 2*Real.sin x + Real.cos x) →
  (deriv f α = 2) →
  (Real.sin (π - α) + Real.cos (-α)) / (2*Real.cos (π/2 - α) + Real.cos (2*π - α)) = 3/5 :=
by sorry

-- Problem 2
theorem triangle_perimeter_range (A B C a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a = 1 →
  a * Real.cos C + (1/2) * c = b →
  let l := a + b + c
  2 < l ∧ l ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_triangle_perimeter_range_l836_83659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_girls_l836_83691

theorem summer_camp_girls : ∃ girls : ℕ,
  girls ≤ 50 ∧
  (girls : ℚ) / 6 + ((50 - girls) : ℚ) / 8 = (50 - 43 : ℚ) ∧
  girls = 18 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_girls_l836_83691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_game_theorem_l836_83623

/-- Represents the number of games with n moves ending at E -/
noncomputable def a (n : ℕ) : ℝ := sorry

/-- x = 2 + √2 -/
noncomputable def x : ℝ := 2 + Real.sqrt 2

/-- y = 2 - √2 -/
noncomputable def y : ℝ := 2 - Real.sqrt 2

/-- The main theorem about the number of games on a regular octagon -/
theorem octagon_game_theorem :
  (∀ k : ℕ, k ≥ 1 → a (2*k - 1) = 0) ∧
  (∀ k : ℕ, k ≥ 1 → a (2*k) = (x^(k-1) - y^(k-1)) / Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_game_theorem_l836_83623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oar_probability_l836_83611

-- Define the probability of an oar working
variable (oar_prob : ℝ)

-- Define the probability of being able to row the canoe
def row_prob : ℝ := 0.84

-- Theorem statement
theorem oar_probability :
  (1 - (1 - oar_prob)^2 = row_prob) →
  oar_prob = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oar_probability_l836_83611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_implies_a_value_l836_83661

/-- Given that the coefficient of x^3 in the expansion of (a/x - √(x/2))^9 is 9/4, prove that a = 4 -/
theorem coefficient_implies_a_value (a : ℝ) : 
  (∃ c : ℝ, c = 9/4 ∧ 
   c = ((-1 : ℝ)^8) * a * 2^(-4 : ℝ) * (Nat.choose 9 8)) → 
  a = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_implies_a_value_l836_83661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_problem_l836_83671

theorem cosine_problem (α β : ℝ) 
  (h1 : Real.cos α = 1/3)
  (h2 : Real.cos (α + β) = -1/3)
  (h3 : 0 < α ∧ α < π/2)
  (h4 : 0 < β ∧ β < π/2) :
  Real.cos β = 7/9 ∧ 2*α + β = π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_problem_l836_83671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carriage_problem_l836_83614

/-- Represents the number of carriages -/
def x : ℕ := sorry

/-- The number of people when each carriage seats 3 with two empty carriages -/
def people_case1 (x : ℕ) : ℕ := 3 * (x - 2)

/-- The number of people when each carriage seats 2 with 9 people left without a seat -/
def people_case2 (x : ℕ) : ℕ := 2 * x + 9

/-- Theorem stating that the two cases represent the same number of people -/
theorem carriage_problem : people_case1 x = people_case2 x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carriage_problem_l836_83614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_percent_decrease_l836_83662

noncomputable def percent_decrease (original_price sale_price : ℝ) : ℝ :=
  (original_price - sale_price) / original_price * 100

theorem equal_percent_decrease (original_price1 sale_price1 original_price2 sale_price2 : ℝ)
  (h1 : original_price1 = 100)
  (h2 : sale_price1 = 70)
  (h3 : original_price2 = 150)
  (h4 : sale_price2 = 105)
  : percent_decrease original_price1 sale_price1 = percent_decrease original_price2 sale_price2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_percent_decrease_l836_83662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_condition_l836_83656

theorem triangle_construction_condition (s α ma : ℝ) : 
  (∃ (a b c : ℝ), a + b + c = 2 * s ∧ 
   ∃ (β γ : ℝ), α + β + γ = Real.pi ∧
   ma = (a * Real.sin γ) / 2) →
  ma < (s * (1 - Real.sin (α / 2))) / Real.cos (α / 2) ∨ 
  ma < s * Real.tan (Real.pi / 4 - α / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_condition_l836_83656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_x_axis_l836_83631

/-- The parabola in question -/
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (3, 0)

/-- The theorem to prove -/
theorem distance_to_x_axis (M : ℝ × ℝ) 
  (h_on_parabola : parabola M.1 M.2)
  (h_dist_focus : Real.sqrt ((M.1 - focus.1)^2 + (M.2 - focus.2)^2) = 9) : 
  M.2 = 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_x_axis_l836_83631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_common_ratio_l836_83683

/-- Given a geometric progression with first term a₁ and common ratio q,
    this function returns the sum of the first n terms. -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem stating that if a₂ + S₃ = 0 in a geometric progression,
    then the common ratio q must be -1. -/
theorem geometric_progression_common_ratio
  (a₁ : ℝ) (q : ℝ) (h₁ : a₁ ≠ 0) :
  a₁ * q + geometric_sum a₁ q 3 = 0 → q = -1 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_common_ratio_l836_83683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_neg_sqrt_5_minus_1_l836_83668

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_neg_sqrt_5_minus_1 : floor (-Real.sqrt 5 - 1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_neg_sqrt_5_minus_1_l836_83668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_cubic_implies_bound_l836_83651

/-- A function f is monotonic on an interval [a, b] if for all x, y in [a, b],
    x ≤ y implies f(x) ≤ f(y) -/
def Monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

/-- The function f(x) = x^3 - ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

theorem monotonic_cubic_implies_bound (a : ℝ) :
  (∀ x y, x ≤ y ∧ y ≤ -1 → f a x ≤ f a y) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_cubic_implies_bound_l836_83651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_missed_problems_l836_83665

theorem max_missed_problems (total_problems : ℕ) (pass_percentage : ℚ) 
  (h1 : total_problems = 50) 
  (h2 : pass_percentage = 75 / 100) : 
  Int.floor ((1 - pass_percentage) * total_problems) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_missed_problems_l836_83665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_digit_puzzle_l836_83673

def Statement (digit : ℕ) (n : ℕ) : Prop :=
  match n with
  | 1 => (digit = 1)
  | 2 => (digit ≠ 2)
  | 3 => (digit = 3)
  | 4 => (digit ≠ 4)
  | _ => False

theorem envelope_digit_puzzle (digit : ℕ) 
  (h1 : digit ≤ 9) 
  (h2 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
       a ≤ 4 ∧ b ≤ 4 ∧ c ≤ 4 ∧ 
       Statement digit a ∧ Statement digit b ∧ Statement digit c) 
  (h3 : ∃ (d : ℕ), d ≤ 4 ∧ ¬Statement digit d) : 
  Statement digit 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_digit_puzzle_l836_83673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_and_initial_conditions_l836_83607

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y)) x + 2 * (deriv y) x - 8 * y x = (12 * x + 20) * Real.exp (2 * x)

-- Define the initial conditions
def initial_conditions (y : ℝ → ℝ) : Prop :=
  y 0 = 0 ∧ (deriv y) 0 = 1

-- Define the solution function
noncomputable def solution (x : ℝ) : ℝ :=
  (1/3) * Real.exp (-4 * x) - (1/3) * Real.exp (2 * x) + (x^2 + 3 * x) * Real.exp (2 * x)

-- State the theorem
theorem solution_satisfies_diff_eq_and_initial_conditions :
  diff_eq solution ∧ initial_conditions solution := by
  sorry

#check solution_satisfies_diff_eq_and_initial_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_and_initial_conditions_l836_83607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_eight_fifteenths_l836_83643

def decagonal_die := Finset.range 10
def six_sided_die := Finset.range 6

def is_multiple_of_three (n : ℕ) : Bool := n % 3 = 0

def probability_multiple_of_three : ℚ :=
  let decagonal_multiples := (decagonal_die.filter (fun x => is_multiple_of_three (x + 1))).card
  let six_sided_multiples := (six_sided_die.filter (fun x => is_multiple_of_three (x + 1))).card
  let decagonal_non_multiples := decagonal_die.card - decagonal_multiples
  let six_sided_non_multiples := six_sided_die.card - six_sided_multiples

  (decagonal_multiples * six_sided_multiples +
   decagonal_multiples * six_sided_non_multiples +
   decagonal_non_multiples * six_sided_multiples) / (decagonal_die.card * six_sided_die.card)

theorem probability_is_eight_fifteenths :
  probability_multiple_of_three = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_eight_fifteenths_l836_83643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_theorem_l836_83622

/-- Represents a point on the lattice -/
structure LatticePoint where
  x : Int
  y : Int

/-- Checks if a jump from point A to B is valid -/
def validJump (a b : LatticePoint) : Prop :=
  abs (a.x * b.y - a.y * b.x) = 1

/-- Checks if a point is reachable from the starting point (1,0) -/
def isReachable (p : LatticePoint) : Prop :=
  ∃ (path : List LatticePoint), 
    path.head? = some ⟨1, 0⟩ ∧ 
    path.getLast? = some p ∧
    ∀ i j, i + 1 = j → j < path.length → validJump (path.get ⟨i, by sorry⟩) (path.get ⟨j, by sorry⟩)

/-- The main theorem to be proved -/
theorem grasshopper_theorem (p : LatticePoint) : 
  (isReachable p ↔ Int.gcd p.x p.y = 1) ∧ 
  (isReachable p → ∃ (path : List LatticePoint), 
    path.head? = some ⟨1, 0⟩ ∧ 
    path.getLast? = some p ∧
    path.length ≤ abs p.y + 2 ∧
    ∀ i j, i + 1 = j → j < path.length → validJump (path.get ⟨i, by sorry⟩) (path.get ⟨j, by sorry⟩)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_theorem_l836_83622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perpendicular_line_l836_83670

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  z : ℂ

/-- Represents a triangle in the complex plane -/
structure Triangle where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint

/-- Represents a parallelogram in the complex plane -/
structure Parallelogram where
  A : ComplexPoint
  F : ComplexPoint
  E : ComplexPoint
  G : ComplexPoint

/-- Checks if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  Complex.abs (t.B.z - t.A.z) = Complex.abs (t.C.z - t.A.z)

/-- Checks if AD is the diameter of the circumcircle of triangle ABC -/
def isDiameter (t : Triangle) (D : ComplexPoint) : Prop :=
  ∃ (center : ℂ), (center - t.A.z) = (t.B.z - center) ∧ (center - t.A.z) = (t.C.z - center) ∧ D.z - t.A.z = 2 * (center - t.A.z)

/-- Checks if point E is on line BC -/
def isOnLine (E : ComplexPoint) (B : ComplexPoint) (C : ComplexPoint) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ E.z = (1 - t) * B.z + t * C.z

/-- Checks if AFEG forms a parallelogram -/
def isParallelogram (p : Parallelogram) : Prop :=
  p.E.z - p.F.z = p.G.z - p.A.z

/-- Checks if two lines are perpendicular in the complex plane -/
def arePerpendicular (A : ComplexPoint) (B : ComplexPoint) (C : ComplexPoint) (D : ComplexPoint) : Prop :=
  (B.z - A.z).im * (D.z - C.z).re = (B.z - A.z).re * (D.z - C.z).im

theorem isosceles_triangle_perpendicular_line
  (t : Triangle) (D E F G : ComplexPoint) (p : Parallelogram)
  (h1 : isIsosceles t)
  (h2 : isDiameter t D)
  (h3 : isOnLine E t.B t.C)
  (h4 : p.A = t.A ∧ p.F.z - t.A.z = t.C.z - t.A.z ∧ p.G.z - t.A.z = t.B.z - t.A.z)
  (h5 : isParallelogram p)
  : arePerpendicular D E F G := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perpendicular_line_l836_83670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_unit_root_l836_83669

noncomputable def P (n : ℕ) (X : ℂ) : ℂ := Real.sqrt 3 * X^(n + 1) - X^n - 1

theorem least_n_for_unit_root : 
  ∀ n : ℕ, (∃ X : ℂ, Complex.abs X = 1 ∧ P n X = 0) ↔ n ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_unit_root_l836_83669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_pointed_star_exists_l836_83675

/-- Represents a seven-pointed star configuration -/
structure SevenPointedStar where
  points : Fin 14 → ℕ
  sum_to_30 : ∀ (line : Fin 4 → Fin 14), (Finset.univ.sum (λ i ↦ points (line i))) = 30
  all_different : ∀ i j, i ≠ j → points i ≠ points j
  range_1_to_14 : ∀ i, 1 ≤ points i ∧ points i ≤ 14

/-- There exists a valid configuration for a seven-pointed star -/
theorem seven_pointed_star_exists : ∃ star : SevenPointedStar, True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_pointed_star_exists_l836_83675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l836_83689

/-- Represents the time (in seconds) it takes for a train to cross an electric pole. -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem stating that a train 55 meters long, traveling at 36 km/hr, takes 5.5 seconds to cross an electric pole. -/
theorem train_crossing_pole_time :
  train_crossing_time 55 36 = 5.5 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l836_83689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_f3_symmetric_l836_83603

-- Define the four functions
noncomputable def f1 (x : ℝ) : ℝ := 10^x
noncomputable def f2 (x : ℝ) : ℝ := Real.log x / Real.log 0.1
noncomputable def f3 (x : ℝ) : ℝ := Real.log (-x) / Real.log 10
noncomputable def f4 (x : ℝ) : ℝ := 0.1^x

-- Define symmetry about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem stating that only f2 and f3 are symmetric about the origin
theorem only_f2_f3_symmetric :
  symmetric_about_origin f2 ∧ 
  symmetric_about_origin f3 ∧ 
  ¬symmetric_about_origin f1 ∧ 
  ¬symmetric_about_origin f4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_f3_symmetric_l836_83603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l836_83641

def sequence_a (n : ℕ) : ℚ := 
  if n % 2 = 1 
  then 4 - 3 * (1/2)^(n - 1) 
  else -4 + 3 * (1/2)^(n - 1)

def S (n : ℕ) : ℚ := sorry

axiom S_1 : S 1 = 1

axiom S_2 : S 2 = -3/2

axiom S_diff (n : ℕ) (h : n ≥ 3) : S n - S (n - 2) = 3 * (-1/2)^(n - 1)

theorem sequence_formula (n : ℕ) (h : n > 0) : 
  sequence_a n = if n % 2 = 1 
    then 4 - 3 * (1/2)^(n - 1) 
    else -4 + 3 * (1/2)^(n - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l836_83641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equals_five_l836_83666

-- Define the common logarithm (lg)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_five :
  2 * lg 5 * 2 * lg 2 + Real.exp 1 * Real.log 3 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equals_five_l836_83666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_can_exit_l836_83602

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance of a point from the origin -/
noncomputable def distanceFromOrigin (p : Point) : ℝ :=
  Real.sqrt (p.x ^ 2 + p.y ^ 2)

/-- Represents Michael's position and movement strategy -/
structure MichaelStrategy where
  initialPosition : Point
  moveDirection : ℕ → Point → Point
  
/-- Represents Catherine's ability to reverse direction -/
def catherine (direction : Point → Point) : Point → Point :=
  fun p => direction p

/-- Checks if a point is outside the circle -/
def isOutside (p : Point) (radius : ℝ) : Prop :=
  distanceFromOrigin p > radius

/-- The main theorem to prove -/
theorem michael_can_exit (radius : ℝ) (h : radius = 100) :
  ∃ (strategy : MichaelStrategy),
    ∀ (n : ℕ),
      let position := (strategy.moveDirection n) (strategy.initialPosition)
      distanceFromOrigin position > distanceFromOrigin strategy.initialPosition ∧
      (∃ (m : ℕ), isOutside ((strategy.moveDirection m) strategy.initialPosition) radius) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_can_exit_l836_83602
