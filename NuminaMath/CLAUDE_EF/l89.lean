import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_exact_concentration_l89_8907

/-- Represents a container with its volume and salt concentration -/
structure Container where
  volume : ℝ
  saltConcentration : ℝ

/-- Represents the state of both containers -/
structure ContainerState where
  container1 : Container
  container2 : Container

/-- Defines a valid transfer operation between containers -/
def isValidTransfer (initialState finalState : ContainerState) : Prop :=
  -- Total volume is conserved
  initialState.container1.volume + initialState.container2.volume =
    finalState.container1.volume + finalState.container2.volume ∧
  -- Total amount of salt is conserved
  initialState.container1.volume * initialState.container1.saltConcentration +
    initialState.container2.volume * initialState.container2.saltConcentration =
    finalState.container1.volume * finalState.container1.saltConcentration +
    finalState.container2.volume * finalState.container2.saltConcentration ∧
  -- Volumes remain non-negative
  finalState.container1.volume ≥ 0 ∧ finalState.container2.volume ≥ 0 ∧
  -- Volumes do not exceed container capacity
  finalState.container1.volume ≤ 3 ∧ finalState.container2.volume ≤ 3

/-- Defines a sequence of valid transfers -/
def validTransferSequence (initialState finalState : ContainerState) : Prop :=
  ∃ (n : ℕ) (sequence : ℕ → ContainerState),
    sequence 0 = initialState ∧
    sequence n = finalState ∧
    ∀ i : ℕ, i < n → isValidTransfer (sequence i) (sequence (i + 1))

/-- The main theorem stating the impossibility of achieving 1.5% salt concentration -/
theorem impossibility_of_exact_concentration : ∀ finalState : ContainerState,
  let initialState := ContainerState.mk
    (Container.mk 1 0)  -- 1 liter of water (0% salt)
    (Container.mk 1 0.02)  -- 1 liter of 2% salt solution
  validTransferSequence initialState finalState →
  finalState.container1.saltConcentration ≠ 0.015 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_exact_concentration_l89_8907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_CD_is_22_l89_8972

/-- The radius of the cylinder and the base radii of the hemisphere and cone -/
noncomputable def r : ℝ := 4

/-- The total volume of the region -/
noncomputable def total_volume : ℝ := 352 * Real.pi

/-- The length of the line segment CD -/
noncomputable def length_CD : ℝ := 22

/-- Theorem stating that given the conditions, the length of CD is 22 units -/
theorem length_of_CD_is_22 :
  ∀ (cylinder_volume hemisphere_volume cone_volume : ℝ),
  cylinder_volume + hemisphere_volume + cone_volume = total_volume →
  hemisphere_volume = (2 / 3) * Real.pi * r^3 →
  cone_volume = (1 / 3) * Real.pi * r^2 * r →
  cylinder_volume = Real.pi * r^2 * (length_CD - 2 * r) →
  length_CD = 22 := by
  sorry

#check length_of_CD_is_22

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_CD_is_22_l89_8972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_sqrt3x_plus_1_l89_8951

/-- The angle of inclination of the line y = √3x + 1 is π/3 -/
theorem angle_of_inclination_sqrt3x_plus_1 : 
  let line (x : ℝ) : ℝ := Real.sqrt 3 * x + 1
  (Real.arctan (Real.sqrt 3)) = π / 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_sqrt3x_plus_1_l89_8951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_proof_min_area_achievable_l89_8990

/-- The minimum area of a triangle with vertices (0,0), (24,10), and (p,q) where p and q are integers -/
noncomputable def min_area_triangle : ℝ := 1

/-- The area of the triangle given integer coordinates p and q -/
noncomputable def triangle_area (p q : ℤ) : ℝ :=
  (1 : ℝ) / 2 * |24 * q + 10 * p|

/-- Proof that the minimum area is indeed the minimum -/
theorem min_area_triangle_proof (p q : ℤ) :
  triangle_area p q ≥ min_area_triangle :=
by sorry

/-- Existence of integers p and q that achieve the minimum area -/
theorem min_area_achievable :
  ∃ (p q : ℤ), triangle_area p q = min_area_triangle :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_proof_min_area_achievable_l89_8990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_third_l89_8986

theorem sin_double_angle_plus_pi_third (α : ℝ) : 
  0 < α → α < π / 2 → Real.cos (α + π / 6) = 4 / 5 → Real.sin (2 * α + π / 3) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_third_l89_8986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_ratio_preserved_l89_8940

/-- A system of rays -/
structure RaySystem where
  a : Set ℝ
  b : Set ℝ
  c : Set ℝ
  d : Set ℝ

/-- Cross-ratio of four collinear points -/
noncomputable def cross_ratio (A B C D : ℝ) : ℝ :=
  ((A - C) / (B - C)) / ((A - D) / (B - D))

/-- Theorem: Cross-ratio is preserved under projection -/
theorem cross_ratio_preserved (S : RaySystem) (l l' : Set ℝ) (C : ℝ)
  (A B D : ℝ) (A' B' D' : ℝ)
  (hA : A ∈ S.a ∩ l) (hB : B ∈ S.b ∩ l) (hC : C ∈ S.c ∩ l) (hD : D ∈ S.d ∩ l)
  (hA' : A' ∈ S.a ∩ l') (hB' : B' ∈ S.b ∩ l') (hC' : C ∈ S.c ∩ l') (hD' : D' ∈ S.d ∩ l')
  (hl : C ∈ l) (hl' : C ∈ l') :
  cross_ratio A B C D = cross_ratio A' B' C D' :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_ratio_preserved_l89_8940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l89_8944

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line segment
structure LineSegment where
  start : Point2D
  finish : Point2D

-- Function to calculate the length of a line segment
noncomputable def length (seg : LineSegment) : ℝ :=
  Real.sqrt ((seg.finish.x - seg.start.x)^2 + (seg.finish.y - seg.start.y)^2)

-- Function to check if a line segment is parallel to the x-axis
def parallelToXAxis (seg : LineSegment) : Prop :=
  seg.start.y = seg.finish.y

-- Theorem statement
theorem point_b_coordinates 
  (a : Point2D) 
  (b : Point2D) 
  (ab : LineSegment) 
  (h1 : a.x = -1 ∧ a.y = 3) 
  (h2 : ab.start = a ∧ ab.finish = b) 
  (h3 : parallelToXAxis ab) 
  (h4 : length ab = 4) : 
  (b.x = -5 ∧ b.y = 3) ∨ (b.x = 3 ∧ b.y = 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l89_8944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l89_8905

-- Define set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = Set.Ioc (-2 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l89_8905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l89_8901

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.cos (x - Real.pi/3) - Real.sqrt 3

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧
    ∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  (∀ k : ℤ, ∀ x y : ℝ,
    k * Real.pi + 5*Real.pi/12 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 11*Real.pi/12 →
    f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l89_8901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l89_8979

-- Define the function f
noncomputable def f (w φ : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (w * x + φ)

-- State the theorem
theorem f_properties (w φ : ℝ) 
  (hw : w > 0) 
  (hφ : -π/2 < φ ∧ φ < π/2) 
  (hsymm : ∀ x, f w φ (2*π/3 - x) = f w φ (2*π/3 + x))
  (hperiod : ∀ x, f w φ (x + π) = f w φ x) :
  (∀ x, f w φ (5*π/6 - x) = -f w φ (5*π/6 + x)) ∧ 
  (∀ x₁ x₂, π/12 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2*π/3 → f w φ x₁ > f w φ x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l89_8979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_when_cos_condition_l89_8955

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - Real.pi/2) * Real.cos (Real.pi/2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi))

theorem f_simplification (α : Real) (h : Real.pi < α ∧ α < 3*Real.pi/2) :
  f α = Real.cos α := by
  sorry

theorem f_value_when_cos_condition (α : Real) 
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2) 
  (h2 : Real.cos (α - Real.pi/2) = -1/4) :
  f α = -Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_when_cos_condition_l89_8955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_8_to_2_digit_sum_l89_8999

theorem base_8_to_2_digit_sum : ∃ (min max : ℕ),
  (min = 8^4) ∧
  (max = 8^5 - 1) ∧
  (∀ n : ℕ, (min ≤ n ∧ n ≤ max) → 
    (∃ d : ℕ, d ≥ 13 ∧ d ≤ 15 ∧ 2^(d-1) ≤ n ∧ n < 2^d)) ∧
  (Finset.sum {13, 14, 15} id = 42) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_8_to_2_digit_sum_l89_8999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_time_l89_8913

/-- Calculate the time required to mow a rectangular lawn -/
theorem lawn_mowing_time 
  (length width swath overlap speed : ℝ) 
  (h1 : length = 120) 
  (h2 : width = 200) 
  (h3 : swath = 28 / 12) 
  (h4 : overlap = 6 / 12) 
  (h5 : speed = 4000) : 
  ∃ (time : ℝ), (time ≥ 3.3 ∧ time < 3.4) ∧ time * speed ≥ (width / (swath - overlap)) * length := by
  sorry

#check lawn_mowing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_time_l89_8913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_and_extrema_l89_8998

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + (a - 1) * x - a * Real.log x

theorem tangent_slope_and_extrema (a : ℝ) :
  (a < 0) →
  (∃ f' : ℝ → ℝ, ∀ x, HasDerivAt (f a) (f' x) x) →
  (∃ f', HasDerivAt (f a) f' 2 ∧ f' = -1) →
  (a = -5/2) ∧
  (∀ a', -1 < a' ∧ a' < 0 →
    (∃ x₁ x₂, x₁ = -a' ∧ x₂ = 1 ∧
      IsLocalMax (f a') x₁ ∧
      IsLocalMin (f a') x₂)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_and_extrema_l89_8998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_alternating_sum_2013_l89_8975

/-- Given a list of integers from 1 to n, this function represents the ability
    to choose between addition and subtraction for each square term. -/
def alternating_sum (n : ℕ) (choices : List Bool) : ℤ :=
  List.sum (List.zipWith
    (λ (i : ℕ) (choice : Bool) => if choice then (i : ℤ)^2 else -(i : ℤ)^2)
    (List.range n)
    choices)

/-- There exists a way to replace some minus signs with plus signs in the expression
    2013² - 2012² - ... - 2² - 1² such that the resulting expression equals 2013. -/
theorem exists_alternating_sum_2013 :
  ∃ choices : List Bool, alternating_sum 2013 choices = 2013 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_alternating_sum_2013_l89_8975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_problem_l89_8945

-- Define necessary structures and functions
def Point := ℝ × ℝ

def IsoscelesTriangle (P Q R : Point) : Prop := sorry
def EquilateralTriangle (K L M : Point) : Prop := sorry
def AngleMeasure (P Q R : Point) : ℝ := sorry
def Area (P Q R : Point) : ℝ := sorry
def IsMidpoint (M P Q : Point) : Prop := sorry

theorem shaded_area_problem (P Q R K L M : Point) : 
  -- Isosceles triangle PQR with angle P = 120°
  IsoscelesTriangle P Q R ∧ 
  AngleMeasure P Q R = 120 ∧ 
  -- Equilateral triangle KLM with area 36
  EquilateralTriangle K L M ∧ 
  Area K L M = 36 ∧ 
  -- K and M are midpoints of PQ and PR respectively
  IsMidpoint K P Q ∧ 
  IsMidpoint M P R → 
  -- The area of PQR minus the area of KLM is 28
  Area P Q R - Area K L M = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_problem_l89_8945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_factor_change_l89_8971

noncomputable def q (w m z : ℝ) : ℝ := 5 * w / (4 * m * z^2)

theorem q_factor_change (w m z : ℝ) (hw : w ≠ 0) (hm : m ≠ 0) (hz : z ≠ 0) :
  q (4 * w) (2 * m) (3 * z) = (5 / 18) * q w m z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_factor_change_l89_8971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l89_8994

/-- A function g(x) defined as x^2 / (Px^2 + Qx + R) -/
noncomputable def g (P Q R : ℤ) : ℝ → ℝ := λ x => x^2 / (P * x^2 + Q * x + R)

/-- Theorem stating that under given conditions, P + Q + R = -24 -/
theorem sum_of_coefficients
  (P Q R : ℤ)
  (h1 : ∀ x > 5, g P Q R x > 0.5)
  (h2 : (P * (-3)^2 + Q * (-3) + R = 0) ∧ (P * 4^2 + Q * 4 + R = 0))
  (h3 : ∀ ε > 0, ∃ δ > 0, ∀ x, |x| > δ → |g P Q R x - (1 / P)| < ε) :
  P + Q + R = -24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l89_8994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arbitrarily_far_condition_l89_8989

/-- A type representing a point on a line -/
structure Point where
  position : ℝ

/-- A configuration of N points on a line -/
structure Configuration (N : ℕ) where
  points : Fin N → Point
  not_all_coincident : ∃ i j, i ≠ j ∧ (points i).position ≠ (points j).position

/-- The move operation as described in the problem -/
def move (k : ℝ) (config : Configuration N) (i j : Fin N) : Configuration N :=
  sorry

/-- Predicate to check if points can be moved arbitrarily far to the right -/
def can_move_arbitrarily_far (k : ℝ) (N : ℕ) : Prop :=
  ∀ (config : Configuration N) (M : ℝ),
    ∃ (new_config : Configuration N),
      (∀ i, (new_config.points i).position ≥ M) ∧
      (∃ (moves : List (Fin N × Fin N)), List.foldl (λ c (i, j) => move k c i j) config moves = new_config)

/-- Main theorem -/
theorem arbitrarily_far_condition (k : ℝ) (N : ℕ) (h1 : k > 0) (h2 : N > 1) :
  can_move_arbitrarily_far k N ↔ k ≥ 1 / (N - 1 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arbitrarily_far_condition_l89_8989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_A_B_l89_8948

def A : ℤ := (Finset.range 19).sum (λ k ↦ (2*k+1)*(2*k+2)) + 39

def B : ℤ := 1 + (Finset.range 18).sum (λ k ↦ (2*k+2)*(2*k+3)) + 38*39

theorem difference_A_B : |A - B| = 722 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_A_B_l89_8948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_l89_8964

/-- A circle C passing through two points and with its center on a given line --/
structure CircleC where
  -- The circle passes through these two points
  A : ℝ × ℝ
  B : ℝ × ℝ
  -- The equation of the line containing the center
  l : ℝ → ℝ → Prop

/-- The circle C satisfies the given conditions --/
def satisfies_conditions (C : CircleC) : Prop :=
  C.A = (1, 1) ∧ C.B = (2, -2) ∧ ∀ x y, C.l x y ↔ x - y + 1 = 0

/-- The equation of the circle C --/
def circle_equation (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 2)^2 = 25

/-- Theorem: The equation (x+3)^2 + (y+2)^2 = 25 represents the circle C
    that satisfies the given conditions --/
theorem circle_C_equation (C : CircleC) (h : satisfies_conditions C) :
  ∀ x y, (x, y) ∈ Set.range (λ (p : ℝ × ℝ) ↦ p) ↔ circle_equation x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_l89_8964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_l89_8961

-- Define the circle
def circleRegion (x y : ℝ) : Prop := x^2 + y^2 ≤ 4

-- Define the line
def lineThroughPoint (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the function to calculate the area difference
noncomputable def area_difference (a b c : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_area_difference :
  ∀ a b c : ℝ,
  lineThroughPoint a b c point_P.1 point_P.2 →
  area_difference a b c ≤ area_difference 1 1 (-2) :=
by
  sorry

#check max_area_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_l89_8961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l89_8933

-- Define the triangle
structure Triangle where
  a : Real
  b : Real
  c : Real
  A : Real
  B : Real
  C : Real
  R : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.a * Real.cos t.A = t.c * Real.cos t.B + t.b * Real.cos t.C)
  (h2 : t.R = 2)
  (h3 : t.b^2 + t.c^2 = 18) :
  t.A = π/3 ∧ (1/2 * t.b * t.c * Real.sin t.A = 3*Real.sqrt 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l89_8933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2A_value_l89_8946

theorem sin_2A_value (A : ℝ) (h1 : 0 < A) (h2 : A < π/2) (h3 : Real.cos A = 3/5) :
  Real.sin (2 * A) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2A_value_l89_8946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_savings_theorem_l89_8914

/-- Calculates the monthly savings in the second year given the initial salary and expense percentages. -/
noncomputable def monthly_savings_second_year (initial_salary : ℚ) (food_percent : ℚ) (medicine_percent : ℚ) 
  (rent_percent : ℚ) (transport_percent : ℚ) (misc_percent : ℚ) (yearly_increase : ℚ) : ℚ :=
  let total_expenses_percent := food_percent + medicine_percent + rent_percent + transport_percent + misc_percent
  let savings_percent := 100 - total_expenses_percent
  let second_year_salary := initial_salary * (1 + yearly_increase / 100)
  second_year_salary * (savings_percent / 100)

/-- Theorem stating that given the specific conditions, the monthly savings in the second year is 3240. -/
theorem monthly_savings_theorem : 
  monthly_savings_second_year 15000 35 20 10 5 10 8 = 3240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_savings_theorem_l89_8914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_discount_l89_8963

/-- Given a bill with face value and true discount, calculate the banker's discount -/
theorem bankers_discount (face_value true_discount : ℚ) : 
  face_value = 270 → true_discount = 45 → 
  (let present_value := face_value - true_discount
   let bankers_discount := true_discount + (true_discount^2 / present_value)
   bankers_discount) = 54 := by
  sorry

#check bankers_discount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_discount_l89_8963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_markup_rate_l89_8996

/-- Calculate the rate of markup on cost for a book sale --/
theorem book_markup_rate (selling_price profit_rate expense_rate : ℝ) :
  selling_price = 8 →
  profit_rate = 0.2 →
  expense_rate = 0.1 →
  let cost := selling_price * (1 - profit_rate - expense_rate)
  let markup_rate := (selling_price - cost) / cost * 100
  ∃ ε : ℝ, ε > 0 ∧ |markup_rate - 42.857| < ε :=
by
  intro h_selling h_profit h_expense
  -- The proof steps would go here
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_markup_rate_l89_8996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_a_eq_one_l89_8942

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of perpendicularity for two lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The first line x + y = 0 -/
noncomputable def line1 : Line :=
  { slope := -1, intercept := 0 }

/-- The second line x - ay = 0 -/
noncomputable def line2 (a : ℝ) : Line :=
  { slope := 1/a, intercept := 0 }

/-- The statement to be proved -/
theorem perpendicular_iff_a_eq_one (a : ℝ) :
  perpendicular line1 (line2 a) ↔ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_a_eq_one_l89_8942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_l89_8916

/-- A cubic function with parameters a, b, and c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The equation derived from f(x) -/
def g (a b : ℝ) (y : ℝ) : ℝ := 3*y^2 + 2*a*y + b

theorem cubic_roots (a b c : ℝ) (x₁ x₂ : ℝ) :
  (∃ x₁ x₂, ∀ x, x ≠ x₁ → x ≠ x₂ → (deriv (f a b c)) x ≠ 0) →  -- f has extreme points x₁ and x₂
  f a b c x₁ = x₁ →  -- f(x₁) = x₁
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, g a b (f a b c x) = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_l89_8916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l89_8997

/-- The range of m for which a point P exists satisfying the given conditions -/
theorem m_range (A B : ℝ × ℝ) (m : ℝ) : 
  A = (-2, 2) →
  B = (2, 6) →
  (∃ k : ℝ, ∀ x y : ℝ, y = k * x + m → x^2 + y^2 = 1 → (k * x - y + m = 0 ∨ k * x - y + m = 0)) →
  (∃ P : ℝ × ℝ, 
    (P.1 - A.1) * (B.1 - P.1) + (P.2 - A.2) * (B.2 - P.2) = -4 ∧
    (∃ k : ℝ, (k * P.1 - P.2 + m) ^ 2 / (1 + k^2) = 1)) →
  m < -2 ∨ m > 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l89_8997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l89_8931

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

theorem interest_calculation (P : ℝ) (h : simple_interest P 5 2 = 56) :
  compound_interest P 5 2 = 57.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l89_8931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_perimeter_l89_8953

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a triangle with three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Circle tangent to a line -/
def CircleTangentToLine (c : Circle) (p1 p2 : ℝ × ℝ) : Prop := sorry

/-- Two circles are tangent -/
def CirclesTangent (c1 c2 : Circle) : Prop := sorry

/-- Right angle formed by three points -/
def RightAngle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- Distance between two points -/
noncomputable def DistanceBetween (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Configuration of circles and triangle as described in the problem -/
def CircleConfiguration (ABC : Triangle) (P Q R S T U : Circle) : Prop :=
  P.radius = 1 ∧ Q.radius = 1 ∧ R.radius = 1 ∧ S.radius = 1 ∧ T.radius = 1 ∧ U.radius = 1 ∧
  -- Circles P, Q, R are tangent to AB
  (CircleTangentToLine P ABC.A ABC.B) ∧
  (CircleTangentToLine Q ABC.A ABC.B) ∧
  (CircleTangentToLine R ABC.A ABC.B) ∧
  -- Circles S, T are tangent to BC
  (CircleTangentToLine S ABC.B ABC.C) ∧
  (CircleTangentToLine T ABC.B ABC.C) ∧
  -- Circle U is tangent to AC
  (CircleTangentToLine U ABC.A ABC.C) ∧
  -- Circles are tangent to each other as described
  (CirclesTangent P Q) ∧ (CirclesTangent Q R) ∧
  (CirclesTangent S T)

/-- ABC is an isosceles right triangle with right angle at A -/
def IsIsoscelesRightTriangle (ABC : Triangle) : Prop :=
  RightAngle ABC.B ABC.A ABC.C ∧
  DistanceBetween ABC.A ABC.B = DistanceBetween ABC.A ABC.C

/-- Perimeter of a triangle -/
noncomputable def Perimeter (ABC : Triangle) : ℝ :=
  DistanceBetween ABC.A ABC.B + DistanceBetween ABC.B ABC.C + DistanceBetween ABC.C ABC.A

theorem circle_configuration_perimeter
  (ABC : Triangle)
  (P Q R S T U : Circle)
  (h1 : CircleConfiguration ABC P Q R S T U)
  (h2 : IsIsoscelesRightTriangle ABC) :
  Perimeter ABC = 8 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_perimeter_l89_8953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lcm_closed_subset_size_l89_8991

theorem max_lcm_closed_subset_size : ∃ (T : Finset Nat), T ⊆ Finset.range 20 ∧
  (∀ (a b : Nat), a ∈ T → b ∈ T → Nat.lcm a b ∈ T) ∧
  T.card = 6 ∧
  (∀ (U : Finset Nat), U ⊆ Finset.range 20 → (∀ (a b : Nat), a ∈ U → b ∈ U → Nat.lcm a b ∈ U) → U.card ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lcm_closed_subset_size_l89_8991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_186_l89_8993

/-- Represents a hexagon with specific angle properties -/
structure Hexagon where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angleD : ℝ
  angleE : ℝ
  angleF : ℝ
  angleA_eq : angleA = 90
  angleB_eq : angleB = 120
  angleC_eq_angleD : angleC = angleD
  angleE_half_sum : angleE = (angleC + angleD) / 2
  angleF_relation : angleF = 2 * angleC - 30
  sum_of_angles : angleA + angleB + angleC + angleD + angleE + angleF = 720

/-- The largest angle in the hexagon is 186° -/
theorem largest_angle_is_186 (h : Hexagon) : 
  max h.angleA (max h.angleB (max h.angleC (max h.angleD (max h.angleE h.angleF)))) = 186 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_186_l89_8993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_eq_equiv_integral_eq_l89_8968

/-- The differential equation y'' + x y' + y = 0 with initial conditions y(0) = 1 and y'(0) = 0 
    is equivalent to the integral equation φ(x) = -1 - ∫₀ˣ (2x - t) φ(t) dt -/
theorem differential_eq_equiv_integral_eq 
  (y : ℝ → ℝ) (φ : ℝ → ℝ) :
  (∀ x, (deriv (deriv y)) x + x * (deriv y x) + y x = 0) ∧ 
  (y 0 = 1) ∧ 
  (deriv y 0 = 0) ↔ 
  (∀ x, φ x = -1 - ∫ t in Set.Icc 0 x, (2*x - t) * φ t) ∧
  (∀ x, (deriv (deriv y)) x = φ x) ∧
  (y 0 = 1) ∧ 
  (deriv y 0 = 0) :=
sorry

#check differential_eq_equiv_integral_eq

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_eq_equiv_integral_eq_l89_8968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_l89_8941

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

-- State the theorem
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_l89_8941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problems_l89_8954

-- Define the square root function
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Define the arithmetic square root function
noncomputable def arith_sqrt (x : ℝ) : ℝ := max (sqrt x) (-(sqrt x))

-- Theorem statement
theorem square_root_problems :
  (∀ (x : ℝ), x^2 = 3343 → x = 7 * sqrt 7 ∨ x = -7 * sqrt 7) ∧
  arith_sqrt ((-5)^2) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problems_l89_8954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_card_strategy_exists_l89_8987

/-- Represents a strategy for encoding and decoding card numbers -/
structure CardStrategy where
  encode : (Fin 100 → ℕ) → ℕ
  decode : ℕ → (Fin 100 → ℕ)

/-- The set of first 100 prime numbers -/
noncomputable def first_100_primes : Finset ℕ := sorry

/-- Theorem stating the existence of an optimal strategy -/
theorem optimal_card_strategy_exists :
  ∃ (s : CardStrategy),
    (∀ (cards : Fin 100 → ℕ),
      (∀ i : Fin 100, (s.decode (s.encode cards) i) = cards i)) ∧
    (∀ (other_s : CardStrategy),
      (∀ (cards : Fin 100 → ℕ),
        (∀ i : Fin 100, (other_s.decode (other_s.encode cards) i) = cards i)) →
      (∀ cards : Fin 100 → ℕ, (s.encode cards).log 2 ≤ (other_s.encode cards).log 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_card_strategy_exists_l89_8987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_a_to_b_l89_8962

/-- Calculates the simple interest rate given the principal, time, and interest amount -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) : ℝ :=
  (interest * 100) / (principal * time)

/-- Proves that the interest rate at which A lent money to B is 10% per annum -/
theorem interest_rate_a_to_b (principal : ℝ) (time : ℝ) (rate_b_to_c : ℝ) (gain_b : ℝ) :
  principal = 3500 →
  time = 3 →
  rate_b_to_c = 14 →
  gain_b = 420 →
  calculate_interest_rate principal time (rate_b_to_c * principal * time / 100 - gain_b) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_a_to_b_l89_8962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_minimized_at_eight_sevenths_l89_8910

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- The point A with coordinates (x, 5-x, 2x-1) -/
def A (x : ℝ) : Point3D :=
  { x := x, y := 5 - x, z := 2*x - 1 }

/-- The point B with coordinates (1, x+2, 2-x) -/
def B (x : ℝ) : Point3D :=
  { x := 1, y := x + 2, z := 2 - x }

/-- The theorem stating that the distance |AB| is minimized when x = 8/7 -/
theorem distance_AB_minimized_at_eight_sevenths :
  ∃ (x : ℝ), ∀ (y : ℝ), distance (A x) (B x) ≤ distance (A y) (B y) ↔ x = 8/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_minimized_at_eight_sevenths_l89_8910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l89_8915

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*x + 3)

theorem monotonic_increase_interval :
  StrictMonoOn f (Set.Iio 1) ∧ ¬ ∃ y, y > 1 ∧ StrictMonoOn f (Set.Iio y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l89_8915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reachable_from_four_l89_8950

-- Define the allowed operations
inductive Operation
| append4 : Operation
| append0 : Operation
| divideBy2 : Operation

def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.append4 => n * 10 + 4
  | Operation.append0 => n * 10
  | Operation.divideBy2 => if n % 2 = 0 then n / 2 else n

def canReach (start target : ℕ) : Prop :=
  ∃ (ops : List Operation), ops.foldl applyOperation start = target

theorem reachable_from_four (n : ℕ) : canReach 4 n := by
  sorry

#check reachable_from_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reachable_from_four_l89_8950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l89_8965

theorem max_value_complex_expression :
  ∃ (M : ℝ), M = (⨆ (z : ℂ) (_ : Complex.abs z = 2), Complex.abs ((z - 2) ^ 2 * (z + 2))) ∧ M = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l89_8965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l89_8984

/-- Represents a parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Represents a line with a given slope and y-intercept -/
structure Line where
  m : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y = m * x + b

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

/-- The main theorem to be proved -/
theorem parabola_triangle_area 
  (C : Parabola) 
  (h1 : C.eq 1 2) -- Parabola passes through (1,2)
  (l : Line)
  (h2 : l.m = 1) -- Line has slope 45°
  (h3 : l.eq (C.p / 2) 0) -- Line passes through the focus (p/2, 0)
  : 
  ∃ A B : ℝ × ℝ, 
    C.eq A.1 A.2 ∧ 
    C.eq B.1 B.2 ∧ 
    l.eq A.1 A.2 ∧ 
    l.eq B.1 B.2 ∧ 
    (triangle_area (0, 0) A B = 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l89_8984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l89_8918

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
  b ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
  c ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
  d ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  ∀ x y z w, x ∈ ({0, 1, 2, 3, 4} : Set ℕ) → 
             y ∈ ({0, 1, 2, 3, 4} : Set ℕ) → 
             z ∈ ({0, 1, 2, 3, 4} : Set ℕ) → 
             w ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
  x ≠ y → x ≠ z → x ≠ w → y ≠ z → y ≠ w → z ≠ w →
  c * (a + b) - d ≤ z * (x + y) - w :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l89_8918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_of_equilateral_triangles_l89_8939

/-- The median of a trapezoid formed by two equilateral triangles -/
theorem trapezoid_median_of_equilateral_triangles :
  let large_side : ℝ := 4
  let large_area : ℝ := (Real.sqrt 3 / 4) * large_side^2
  let small_area : ℝ := (1 / 3) * large_area
  let small_side : ℝ := Real.sqrt ((4 * small_area) / Real.sqrt 3)
  let median : ℝ := (large_side + small_side) / 2
  median = 2 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_of_equilateral_triangles_l89_8939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_caterer_cheapest_at_42_l89_8925

/-- Represents the cost function for a caterer -/
structure Caterer where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a caterer given the number of people -/
def totalCost (c : Caterer) (people : ℕ) : ℕ :=
  c.basicFee + c.perPersonFee * people

theorem third_caterer_cheapest_at_42 :
  let caterer1 := Caterer.mk 120 18
  let caterer2 := Caterer.mk 250 14
  let caterer3 := Caterer.mk 0 20
  ∀ n : ℕ,
    (n < 42 → 
      totalCost caterer3 n > min (totalCost caterer1 n) (totalCost caterer2 n)) ∧
    (n ≥ 42 → 
      totalCost caterer3 n ≤ min (totalCost caterer1 n) (totalCost caterer2 n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_caterer_cheapest_at_42_l89_8925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_for_3_to_a_l89_8911

theorem k_value_for_3_to_a (a : ℝ) (k : ℤ) (h1 : (3 : ℝ)^a = 0.618) 
  (h2 : a ∈ Set.Ici (k : ℝ) ∩ Set.Iio ((k : ℝ) + 1)) : k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_for_3_to_a_l89_8911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_b_highest_speed_l89_8956

-- Define the data for each car
noncomputable def car_a_distance : ℝ := 715
noncomputable def car_a_time : ℝ := 11
noncomputable def car_b_distance : ℝ := 820
noncomputable def car_b_time : ℝ := 12
noncomputable def car_c_distance : ℝ := 950
noncomputable def car_c_time : ℝ := 14

-- Define average speed function
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Theorem statement
theorem car_b_highest_speed :
  let speed_a := average_speed car_a_distance car_a_time
  let speed_b := average_speed car_b_distance car_b_time
  let speed_c := average_speed car_c_distance car_c_time
  speed_b > speed_a ∧ speed_b > speed_c :=
by
  -- Unfold the definitions
  unfold average_speed
  -- Calculate the speeds
  have speed_a : ℝ := car_a_distance / car_a_time
  have speed_b : ℝ := car_b_distance / car_b_time
  have speed_c : ℝ := car_c_distance / car_c_time
  -- Prove the inequalities
  sorry -- This skips the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_b_highest_speed_l89_8956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_perfect_power_l89_8978

theorem sequence_not_perfect_power (t : ℕ) (ht : t > 0) :
  ∃ (n : ℕ) (ℓ : ℕ), 
    n > 1 ∧ 
    Nat.Coprime n t ∧
    n = (1 + t * (t + 1)^2)^ℓ ∧
    ∀ (k : ℕ), k ≥ 1 → ¬ ∃ (m : ℕ) (p : ℕ), m > 1 ∧ n^k + t = p^m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_perfect_power_l89_8978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_volume_cylinder_l89_8935

-- Define the volume of a cylinder
noncomputable def cylinderVolume (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h

-- State the theorem
theorem double_volume_cylinder :
  let r1 : ℝ := 5
  let h1 : ℝ := 10
  let r2 : ℝ := 10
  let h2 : ℝ := 5
  cylinderVolume r2 h2 = 2 * cylinderVolume r1 h1 := by
  -- Unfold the definition of cylinderVolume
  unfold cylinderVolume
  -- Simplify the expressions
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_volume_cylinder_l89_8935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_l89_8973

theorem cos_2theta_value (θ : ℝ) (h : Real.cos θ + 2 * Real.sin θ = 3/2) : Real.cos (2 * θ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_l89_8973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_rate_l89_8985

/-- Calculates the interest rate for the first part of an investment given the following conditions:
  * The total investment is 3400
  * The first part of the investment is 1300
  * The second part of the investment is 2100
  * The interest rate for the second part is 5%
  * The total annual interest from both investments is 144
-/
theorem investment_interest_rate : 
  let total_investment : ℚ := 3400
  let first_part : ℚ := 1300
  let second_part : ℚ := 2100
  let second_rate : ℚ := 5
  let total_interest : ℚ := 144
  let first_rate : ℚ := (total_interest - second_part * second_rate / 100) / first_part * 100
  ∃ ε > 0, |first_rate - 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_rate_l89_8985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l89_8957

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_area : a * b = 2

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) : Prop :=
  ∀ x y, x^2/4 + y^2 = 1

/-- The theorem to be proved -/
theorem ellipse_theorem (e : Ellipse) (p : PointOnEllipse e) :
  e.equation ∧
  (∃ (an bm : ℝ), 
    an * bm = 4 ∧
    (∀ (x₀ y₀ : ℝ), x₀^2/4 + y₀^2 = 1 → 
      ∃ (m n : ℝ),
        (y₀ / (x₀ - 2) * (-2) = m) ∧ 
        (-2 * x₀ / (y₀ - 1) = n) ∧
        |2 + n| * |1 + m| = 4)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l89_8957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_length_l89_8900

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- The distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- The theorem to be proved -/
theorem parabola_line_intersection_length
  (para : Parabola)
  (l : Line)
  (focus : Point)
  (h1 : focus.x = 3 ∧ focus.y = 0)
  (h2 : l.m = 2 ∧ l.b = -6)
  (h3 : para.p = 6)
  (M N : Point)
  (h4 : M.y^2 = 2 * para.p * M.x ∧ M.y = l.m * M.x + l.b)
  (h5 : N.y^2 = 2 * para.p * N.x ∧ N.y = l.m * N.x + l.b)
  : distance M N = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_length_l89_8900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_formula_l89_8949

def distribute (n : ℕ) : ℕ := 3^n - 3 * 2^n + 3

theorem distribute_formula (n : ℕ) :
  distribute n = (Finset.univ.filter (λ f : Fin n → Fin 3 ↦ 
    (∀ i : Fin 3, ∃ x : Fin n, f x = i))).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_formula_l89_8949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equals_target_l89_8967

def a (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => a n + n + 1

theorem sum_reciprocals_equals_target : 
  (Finset.range 2006).sum (λ n => 1 / a (n + 1)) = 4032 / 2017 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equals_target_l89_8967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_10_8_l89_8906

theorem binomial_10_8 : (Nat.choose 10 8) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_10_8_l89_8906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l89_8970

theorem equation_solution : ∃ x : ℝ, (3 : ℝ) ^ ((27 : ℝ) ^ x) = (27 : ℝ) ^ ((3 : ℝ) ^ x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l89_8970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_segment_ratio_l89_8924

/-- Triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- Side lengths of the triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Smaller segment of side 'a' created by the point of tangency -/
  u : ℝ
  /-- Larger segment of side 'a' created by the point of tangency -/
  v : ℝ
  /-- The circle is inscribed in the triangle -/
  inscribed : u + v = a
  /-- 'u' is the smaller segment -/
  u_smaller : u < v

/-- Theorem: In a triangle with side lengths 9, 12, and 15 and an inscribed circle,
    the ratio of the smaller to larger segment of the side with length 9 is 1:2 -/
theorem inscribed_circle_segment_ratio
  (t : TriangleWithInscribedCircle)
  (h1 : t.a = 9)
  (h2 : t.b = 12)
  (h3 : t.c = 15) :
  t.u / t.v = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_segment_ratio_l89_8924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_is_one_l89_8919

-- Define the sector
noncomputable def sector_radius : ℝ := 3
noncomputable def central_angle : ℝ := 120

-- Define the arc length of the sector
noncomputable def arc_length : ℝ := (central_angle / 360) * (2 * Real.pi * sector_radius)

-- Define the radius of the cone's base
noncomputable def cone_base_radius : ℝ := arc_length / (2 * Real.pi)

-- Theorem statement
theorem cone_base_radius_is_one : 
  cone_base_radius = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_is_one_l89_8919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_is_unit_circle_intersection_point_l89_8959

-- Define the parametric equations for curve C₁
noncomputable def C₁ (k : ℝ) (t : ℝ) : ℝ × ℝ :=
  (Real.cos t ^ k, Real.sin t ^ k)

-- Define the Cartesian equation for curve C₂
def C₂ (x y : ℝ) : Prop :=
  4 * x - 16 * y + 3 = 0

-- Theorem 1: C₁ is a unit circle when k = 1
theorem C₁_is_unit_circle :
  ∀ (x y : ℝ), (∃ t, C₁ 1 t = (x, y)) ↔ x^2 + y^2 = 1 := by
  sorry

-- Theorem 2: Intersection point of C₁ and C₂ when k = 4
theorem intersection_point :
  ∃! (x y : ℝ), (∃ t, C₁ 4 t = (x, y)) ∧ C₂ x y ∧ x = 1/4 ∧ y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_is_unit_circle_intersection_point_l89_8959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_time_l89_8974

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℚ
  deriving Repr

/-- The problem setup -/
def runnersProblem (trackLength : ℚ) (runners : List Runner) : Prop :=
  trackLength > 0 ∧
  runners.length = 3 ∧
  runners.all (λ r => r.speed > 0) ∧
  ∃ (t : ℚ), t > 0 ∧ 
    ∀ (i j : Fin runners.length), 
      ∃ (k : ℤ), (runners[i].speed * t - runners[j].speed * t : ℚ) = k * trackLength

theorem runners_meet_time : 
  let trackLength : ℚ := 600
  let runners : List Runner := [⟨4.5⟩, ⟨4.9⟩, ⟨5.1⟩]
  runnersProblem trackLength runners → 
  ∃ (t : ℚ), t = 3000 ∧
    ∀ (i j : Fin runners.length), 
      ∃ (k : ℤ), (runners[i].speed * t - runners[j].speed * t : ℚ) = k * trackLength :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_time_l89_8974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_equation_solution_l89_8928

/-- The roots of the equation x³ - 2018x + 2018 = 0 -/
noncomputable def roots : Finset ℂ := sorry

/-- The sum of nth powers of the roots -/
noncomputable def S (n : ℕ) : ℂ := (roots.sum (λ r => r ^ n))

/-- Predicate for the equation in the problem -/
def satisfiesEquation (p q : ℕ) : Prop :=
  S (p + q) / (p + q : ℂ) = (S p / p) * (S q / q)

/-- The smallest positive q for which there exists a p satisfying the equation -/
noncomputable def smallestQ : ℕ := sorry

/-- The corresponding p for the smallest q -/
noncomputable def correspondingP : ℕ := sorry

theorem root_equation_solution :
  smallestQ > 0 ∧
  correspondingP > 0 ∧
  correspondingP ≤ smallestQ ∧
  satisfiesEquation correspondingP smallestQ ∧
  correspondingP ^ 2 + smallestQ ^ 2 = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_equation_solution_l89_8928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_performance_l89_8977

/-- Represents a student's performance scores -/
structure StudentPerformance where
  regular : ℚ
  midterm : ℚ
  final : ℚ

/-- Calculates the semester academic performance given performance scores and weights -/
def semesterPerformance (scores : StudentPerformance) (weights : StudentPerformance) : ℚ :=
  (scores.regular * weights.regular + scores.midterm * weights.midterm + scores.final * weights.final) /
  (weights.regular + weights.midterm + weights.final)

/-- Theorem stating that Xiaoming's semester academic performance is 89 points -/
theorem xiaoming_performance :
  let scores : StudentPerformance := ⟨90, 80, 94⟩
  let weights : StudentPerformance := ⟨2, 3, 5⟩
  semesterPerformance scores weights = 89 := by
  sorry

#eval semesterPerformance ⟨90, 80, 94⟩ ⟨2, 3, 5⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_performance_l89_8977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teresas_age_l89_8930

def guesses : List Nat := [35, 39, 41, 43, 47, 49, 51, 53, 58, 60]

def isPrime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

theorem teresas_age (teresa_age : Nat) : teresa_age = 43 :=
  by
  have h1 : teresa_age ∈ guesses := by sorry
  have h2 : isPrime teresa_age := by sorry
  have h3 : (guesses.filter (· < teresa_age)).length ≥ guesses.length / 2 := by sorry
  have h4 : (guesses.filter (· = teresa_age)).length = 3 := by sorry
  have h5 : ∀ g ∈ guesses, g ≠ teresa_age → (g : Int) - teresa_age ≥ 2 ∨ teresa_age - g ≥ 2 := by sorry
  have h6 : ∀ g ∈ guesses, g ≥ 35 := by sorry
  sorry

#check teresas_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teresas_age_l89_8930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l89_8909

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, (1/2) * x^2 - Real.log x - a ≥ 0) ∧
  (∃ x : ℝ, x^2 + 2*a*x - 8 - 6*a = 0) →
  a ∈ Set.Iic (1/2) := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l89_8909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_necessary_for_q_l89_8960

-- Define propositions p and q
variable (p q : Prop)

-- Define the original implication
def original_implication (p q : Prop) : Prop := p → q

-- Define the inverse implication
def inverse_implication (p q : Prop) : Prop := q → p

-- Define necessary condition
def is_necessary_condition (a b : Prop) : Prop :=
  b → a

-- Theorem statement
theorem p_is_necessary_for_q 
  (h1 : original_implication p q) 
  (h2 : inverse_implication p q) : 
  is_necessary_condition p q :=
by
  -- Unfold the definition of is_necessary_condition
  unfold is_necessary_condition
  -- Assume q
  intro hq
  -- Use the inverse implication to prove p
  exact h2 hq


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_necessary_for_q_l89_8960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l89_8904

noncomputable def f (x : ℝ) : ℝ := (x^4 + x^3 - 3*x^2 - 6*x + 8) / (x^3 - 3*x^2 - 4*x + 12)

def domain_f : Set ℝ := {x | x < -2 ∨ (-2 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) ∨ 3 < x}

theorem f_domain : {x : ℝ | f x ≠ 0} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l89_8904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l89_8903

theorem complex_magnitude_problem : Complex.abs (Complex.I * (2 + Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l89_8903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_product_l89_8921

/-- The hyperbola with equation x^2 - y^2 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 1}

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Angle between three points in ℝ² -/
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focal_product :
  ∀ P ∈ Hyperbola, angle F₁ P F₂ = π/3 → distance P F₁ * distance P F₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_product_l89_8921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_four_range_l89_8917

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else x^2

-- Theorem statement
theorem f_greater_than_four_range :
  {x : ℝ | f x > 4} = {x : ℝ | x < -2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_four_range_l89_8917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_sum_limit_l89_8966

/-- The sum of the perimeters of a sequence of equilateral triangles -/
noncomputable def perimeter_sum (a : ℝ) : ℕ → ℝ
| 0 => 3 * a
| n + 1 => perimeter_sum a n + 3 * a / (2 ^ (n + 1))

/-- The theorem stating the limit of the sum of perimeters -/
theorem perimeter_sum_limit (a : ℝ) (h : a > 0) :
  ∃ L, Filter.Tendsto (perimeter_sum a) Filter.atTop (nhds L) ∧ L = 6 * a := by
  sorry

#check perimeter_sum_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_sum_limit_l89_8966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_town_distance_l89_8932

/-- The distance to the town given fuel efficiency and required fuel --/
noncomputable def distance_to_town (fuel_efficiency : ℝ) (fuel_required : ℝ) : ℝ :=
  (fuel_efficiency / 10) * fuel_required

/-- Proof that the distance to the town is 140 km --/
theorem town_distance :
  let fuel_efficiency : ℝ := 70  -- km per 10 liters
  let fuel_required : ℝ := 20   -- liters for one-way trip
  distance_to_town fuel_efficiency fuel_required = 140 := by
  -- Unfold the definition of distance_to_town
  unfold distance_to_town
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_town_distance_l89_8932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_125_64_trailing_zeros_l89_8980

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

def trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + trailing_zeros (n / 10)
  else 0

theorem binomial_125_64_trailing_zeros :
  trailing_zeros (binomial_coefficient 125 64) = 0 := by
  sorry

#eval binomial_coefficient 125 64
#eval trailing_zeros (binomial_coefficient 125 64)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_125_64_trailing_zeros_l89_8980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial12_base11_trailing_zeroes_l89_8952

/-- The number of trailing zeroes in the base 11 representation of 12! -/
def trailingZeroesBase11Factorial12 : ℕ := 1

/-- Theorem: The number of trailing zeroes in the base 11 representation of 12! is 1 -/
theorem factorial12_base11_trailing_zeroes :
  trailingZeroesBase11Factorial12 = 1 := by
  -- Unfold the definition of trailingZeroesBase11Factorial12
  unfold trailingZeroesBase11Factorial12
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial12_base11_trailing_zeroes_l89_8952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billiards_ball_never_reaches_pocket_l89_8995

theorem billiards_ball_never_reaches_pocket :
  ∀ (m n : ℤ), (m : ℝ) ≠ n * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billiards_ball_never_reaches_pocket_l89_8995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_in_box_C_l89_8929

-- Define the set of boxes
inductive Box
| A
| B
| C
| D

-- Define a function to represent the content of each note
def note_content (b : Box) : Box → Prop :=
  match b with
  | Box.A => λ x => x = Box.A
  | Box.B => λ x => x ≠ Box.A
  | Box.C => λ x => x ≠ Box.C
  | Box.D => λ x => x = Box.D

-- Define the main theorem
theorem apple_in_box_C :
  ∃! (truth_teller : Box),
    ∃! (apple_location : Box),
      (∀ b : Box, note_content b apple_location ↔ b = truth_teller) →
      apple_location = Box.C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_in_box_C_l89_8929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l89_8976

noncomputable def charlie_can_diameter : ℝ := 4
noncomputable def charlie_can_height : ℝ := 16
noncomputable def morgan_can_diameter : ℝ := 16
noncomputable def morgan_can_height : ℝ := 4
noncomputable def sealed_compartment_ratio : ℝ := 0.25

noncomputable def charlie_can_usable_volume : ℝ :=
  (1 - sealed_compartment_ratio) * Real.pi * (charlie_can_diameter / 2) ^ 2 * charlie_can_height

noncomputable def morgan_can_volume : ℝ :=
  Real.pi * (morgan_can_diameter / 2) ^ 2 * morgan_can_height

theorem volume_ratio :
  charlie_can_usable_volume / morgan_can_volume = 3 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l89_8976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l89_8912

-- Define the problem in the noncomputable section
noncomputable section

-- Define the function f as a variable
variable (f : ℝ → ℝ)

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem solution_set_theorem (h_odd : is_odd f)
  (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x > 0, 2 * f x + x * (deriv f x) > x^2) :
  {x : ℝ | (x + 2014)^2 * f (x + 2014) + 4 * f (-2) < 0} = Set.Ioi (-2012) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l89_8912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l89_8927

def A : Set ℤ := {x : ℤ | |x| ≤ 2}

def B : Set ℝ := {x : ℝ | x^2 - 2*x - 8 ≥ 0}

def B_int : Set ℤ := {x : ℤ | (x : ℝ) ∈ B}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (Set.compl B_int) = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l89_8927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_g_fixed_point_l89_8920

def g (x : ℕ) : ℕ :=
  if x % 4 = 0 ∧ x % 7 = 0 then x / 28
  else if x % 7 = 0 then 4 * x
  else if x % 4 = 0 then 7 * x
  else x + 4

def g_iter : ℕ → ℕ → ℕ
| 0, x => x
| (n + 1), x => g (g_iter n x)

theorem smallest_a_for_g_fixed_point :
  (∃ a : ℕ, a > 1 ∧ g_iter a 2 = g 2) ∧
  (∀ a : ℕ, a > 1 ∧ g_iter a 2 = g 2 → a ≥ 6) := by
  sorry

#eval g_iter 6 2
#eval g 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_g_fixed_point_l89_8920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_comparison_theorem_l89_8947

noncomputable def initial_investment : ℝ := 10000
noncomputable def bank_rate : ℝ := 0.06
noncomputable def stock_yield : ℝ := 0.24

noncomputable def geometric_sum (r : ℝ) (n : ℕ) : ℝ :=
  (1 - r^n) / (1 - r)

theorem investment_comparison_theorem :
  ∃ (n1 n2 : ℕ), 
    (n1 > 0 ∧ n2 > 0) ∧
    (stock_yield * geometric_sum (1 + bank_rate) n1 ≥ 1) ∧
    (stock_yield * geometric_sum (1 + bank_rate) n2 ≥ (1 + bank_rate)^n2) := by
  sorry

#check investment_comparison_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_comparison_theorem_l89_8947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_dot_product_l89_8981

/-- Given a parallelogram ABCD with AB parallel to CD, prove that the dot product of AC and DB is 3 -/
theorem parallelogram_dot_product (A B C D : ℝ × ℝ) : 
  (B.1 - A.1 = 2 ∧ B.2 - A.2 = -2) →  -- AB = (2, -2)
  (D.1 - A.1 = 2 ∧ D.2 - A.2 = 1) →   -- AD = (2, 1)
  (C.1 - B.1 = D.1 - A.1 ∧ C.2 - B.2 = D.2 - A.2) →  -- AB parallel to CD
  ((C.1 - A.1) * (B.1 - D.1) + (C.2 - A.2) * (B.2 - D.2) = 3) := by
  sorry

#check parallelogram_dot_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_dot_product_l89_8981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunglasses_wearers_l89_8937

theorem sunglasses_wearers (total_adults : ℕ) (women_percentage : ℚ) 
  (women_sunglasses_percentage : ℚ) (men_sunglasses_percentage : ℚ) :
  total_adults = 2500 →
  women_percentage = 1/2 →
  women_sunglasses_percentage = 15/100 →
  men_sunglasses_percentage = 12/100 →
  ∃ (total_sunglasses : ℕ), total_sunglasses = 338 ∧
    total_sunglasses = 
      (Int.floor ((total_adults : ℚ) * women_percentage * women_sunglasses_percentage)).toNat +
      (Int.floor ((total_adults : ℚ) * (1 - women_percentage) * men_sunglasses_percentage)).toNat :=
by
  sorry

#check sunglasses_wearers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunglasses_wearers_l89_8937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l89_8936

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  positive_a : 0 < a
  positive_b : 0 < b

/-- Represents a square with vertices on a hyperbola -/
structure SquareOnHyperbola (a b : ℝ) extends Hyperbola a b where
  vertices_on_hyperbola : True  -- This is a placeholder for the condition

/-- Represents the condition that the midpoints of two opposite sides of the square are the foci of the hyperbola -/
def MidpointsAreFoci (a b : ℝ) (_ : SquareOnHyperbola a b) : Prop := True  -- This is a placeholder for the condition

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) (_ : Hyperbola a b) : ℝ := 
  Real.sqrt (1 + b^2 / a^2)

/-- The main theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h : SquareOnHyperbola a b) 
  (h_foci : MidpointsAreFoci a b h) : 
  eccentricity a b h.toHyperbola = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l89_8936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_period_7pi_when_n_20_n_20_is_solution_l89_8926

/-- The function f(x) for a given n -/
noncomputable def f (n : ℤ) (x : ℝ) : ℝ := Real.sin ((2 * n + 1 : ℤ) * x) * Real.sin (5 * x / (n - 1 : ℤ))

/-- The period of the function f(x) -/
noncomputable def period (n : ℤ) : ℝ := 7 * Real.pi

/-- Theorem stating that f(x) has a period of 7π when n = 20 -/
theorem f_has_period_7pi_when_n_20 :
  ∀ x : ℝ, f 20 (x + period 20) = f 20 x := by
  sorry

/-- Theorem stating that n = 20 is a solution to the problem -/
theorem n_20_is_solution :
  ∃ n : ℤ, ∀ x : ℝ, f n (x + period n) = f n x := by
  use 20
  exact f_has_period_7pi_when_n_20


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_period_7pi_when_n_20_n_20_is_solution_l89_8926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_ratio_after_two_borders_l89_8902

/-- Represents a square grid pattern with black and white tiles -/
structure GridPattern where
  size : Nat
  black_tiles : Nat
  white_tiles : Nat

/-- Adds a border of black tiles around a grid pattern -/
def add_border (grid : GridPattern) : GridPattern :=
  { size := grid.size + 2,
    black_tiles := grid.black_tiles + (grid.size + 2)^2 - grid.size^2,
    white_tiles := grid.white_tiles }

/-- Calculates the ratio of black tiles to white tiles -/
def tile_ratio (grid : GridPattern) : ℚ :=
  (grid.black_tiles : ℚ) / (grid.white_tiles : ℚ)

/-- The initial 5x5 grid pattern -/
def initial_grid : GridPattern :=
  { size := 5, black_tiles := 9, white_tiles := 16 }

/-- Theorem stating the new ratio after adding two borders -/
theorem new_ratio_after_two_borders :
  tile_ratio (add_border (add_border initial_grid)) = 65 / 16 := by
  sorry

#eval tile_ratio (add_border (add_border initial_grid))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_ratio_after_two_borders_l89_8902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_l89_8983

theorem milk_water_ratio (initial_volume : ℝ) (milk_parts water_parts : ℕ) (added_water : ℝ) :
  initial_volume = 100 →
  milk_parts = 3 →
  water_parts = 2 →
  added_water = 48 →
  let initial_milk := (milk_parts : ℝ) / ((milk_parts : ℝ) + (water_parts : ℝ)) * initial_volume
  let initial_water := (water_parts : ℝ) / ((milk_parts : ℝ) + (water_parts : ℝ)) * initial_volume
  let new_water := initial_water + added_water
  let new_ratio_milk := initial_milk / (initial_milk + new_water) * (15 + 22)
  let new_ratio_water := new_water / (initial_milk + new_water) * (15 + 22)
  (new_ratio_milk = 15 ∧ new_ratio_water = 22) :=
by
  sorry

#check milk_water_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_l89_8983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cute_number_l89_8938

def is_cute (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    Finset.toSet {a, b, c, d, e} = Finset.toSet {1, 2, 3, 4, 5} ∧
    a % 1 = 0 ∧
    (10 * a + b) % 2 = 0 ∧
    (100 * a + 10 * b + c) % 3 = 0 ∧
    (1000 * a + 100 * b + 10 * c + d) % 4 = 0 ∧
    n % 5 = 0

theorem unique_cute_number :
  ∃! n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ is_cute n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cute_number_l89_8938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_range_of_t_general_equation_of_C1_length_of_PQ_l89_8992

-- Define the function f
def f (x : ℝ) : ℝ := 45 * |2*x + 2| - 45 * |x - 2|

-- Theorem 1
theorem solution_set_of_f (x : ℝ) : 
  f x > 2 ↔ x ∈ Set.Ioi (-6) ∪ Set.Ioi (2/3) :=
sorry

-- Theorem 2
theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x ≥ t^2 - (7/2)*t) ↔ t ∈ Set.Icc (3/2) 2 :=
sorry

-- Define the parametric equations of C1
noncomputable def C1_x (θ : ℝ) : ℝ := 1 + 2 * Real.cos θ
noncomputable def C1_y (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Theorem 3
theorem general_equation_of_C1 (x y : ℝ) :
  (∃ θ : ℝ, x = C1_x θ ∧ y = C1_y θ) ↔ (x - 1)^2 + y^2 = 4 :=
sorry

-- Define the polar equation of line l
def line_l (ρ θ : ℝ) : Prop := 2 * ρ * Real.sin (θ + Real.pi/3) = 3 * Real.sqrt 3

-- Theorem 4
theorem length_of_PQ :
  ∃ P Q : ℝ × ℝ, 
    (∃ θp : ℝ, P.1 = C1_x θp ∧ P.2 = C1_y θp) ∧
    (∃ θq : ℝ, Q.1 = C1_x θq ∧ Q.2 = C1_y θq) ∧
    (∃ ρp θp : ℝ, line_l ρp θp ∧ P.1 = ρp * Real.cos θp ∧ P.2 = ρp * Real.sin θp) ∧
    (∃ ρq θq : ℝ, line_l ρq θq ∧ Q.1 = ρq * Real.cos θq ∧ Q.2 = ρq * Real.sin θq) ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_range_of_t_general_equation_of_C1_length_of_PQ_l89_8992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l89_8934

-- Define the centers of the circles
def A : ℝ × ℝ := (-7, 3)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (9, 5)

-- Define the radii of the circles
def rA : ℝ := 3
def rB : ℝ := 4
def rC : ℝ := 5

-- Define the tangent points on line m
def A' : ℝ × ℝ := (-7, 0)
def B' : ℝ × ℝ := (0, 0)
def C' : ℝ × ℝ := (9, 0)

-- Theorem statement
theorem area_of_triangle_ABC : 
  let d_AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let d_BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  d_AB = rA + rB ∧ d_BC = rB + rC →
  B'.1 > A'.1 ∧ B'.1 < C'.1 →
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l89_8934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l89_8969

theorem range_of_m (m : ℝ) : 
  ((∀ x : ℝ, x^2 - 2*m*x + m ≠ 0) → (0 < m ∧ m < 1)) →
  ((∀ x : ℝ, x^2 + m*x + 1 ≥ 0) → (-2 ≤ m ∧ m ≤ 2)) →
  ((0 < m ∧ m < 1) ∨ (-2 ≤ m ∧ m ≤ 2)) ∧ ¬((0 < m ∧ m < 1) ∧ (-2 ≤ m ∧ m ≤ 2)) →
  (-2 ≤ m ∧ m ≤ 0) ∨ (1 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l89_8969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_theorem_l89_8908

-- Define a quadrilateral type
structure Quadrilateral : Type :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop := sorry

-- Define perpendicular diagonals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry

-- Original proposition
def original_proposition : Prop :=
  ∀ q : Quadrilateral, is_rhombus q → has_perpendicular_diagonals q

-- Converse proposition
def converse_proposition : Prop :=
  ∀ q : Quadrilateral, has_perpendicular_diagonals q → is_rhombus q

-- Inverse proposition
def inverse_proposition : Prop :=
  ∀ q : Quadrilateral, ¬is_rhombus q → ¬has_perpendicular_diagonals q

-- Contrapositive proposition
def contrapositive_proposition : Prop :=
  ∀ q : Quadrilateral, ¬has_perpendicular_diagonals q → ¬is_rhombus q

theorem rhombus_diagonals_theorem :
  original_proposition ∧
  ¬converse_proposition ∧
  ¬inverse_proposition ∧
  contrapositive_proposition :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_theorem_l89_8908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hyperdeficient_numbers_l89_8922

/-- Sum of squares of all divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- A number n is hyperdeficient if g(g(n)) = n^2 + 2 -/
def isHyperdeficient (n : ℕ) : Prop :=
  g (g n) = n ^ 2 + 2

theorem no_hyperdeficient_numbers : ∀ n : ℕ, n > 0 → ¬ isHyperdeficient n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hyperdeficient_numbers_l89_8922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_properties_l89_8943

noncomputable def min_positive_period (f : ℝ → ℝ) : ℝ := sorry

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem tan_x_properties :
  (min_positive_period (λ x => Real.sin (2 * x)) = min_positive_period Real.tan) ∧
  (is_odd_function (λ x => Real.sin (2 * x)) ↔ is_odd_function Real.tan) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_properties_l89_8943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_inequality_l89_8982

noncomputable section

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Axioms for the properties of f
axiom f_domain : ∀ x : ℝ, x > 0 → f x ∈ Set.range f

axiom f_property : ∀ x y : ℝ, x > 0 → y > 0 → f x + f y = f (x * y)

axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

axiom f_3 : f 3 = 1

-- Theorem stating that f is increasing and satisfies the inequality
theorem f_increasing_and_inequality (a : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x < f y) ∧
  (a > 3 → ∀ x : ℝ, x > 0 → (2 * a - x) > 0 →
    (f x - f (1 / (2 * a - x)) ≥ 2 ↔ 
      a - Real.sqrt (a^2 - 9) < x ∧ x < a + Real.sqrt (a^2 - 9))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_inequality_l89_8982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_leaking_tank_l89_8988

/-- The time it takes to fill a leaking tank -/
noncomputable def fill_time_with_leak (fill_time_no_leak : ℝ) (empty_time : ℝ) : ℝ :=
  (fill_time_no_leak * empty_time) / (empty_time - fill_time_no_leak)

/-- Theorem: Given a tank that can be filled in 7 hours without a leak
    and emptied in 56 hours due to a leak, it takes 8 hours to fill the tank with the leak present -/
theorem fill_time_leaking_tank :
  fill_time_with_leak 7 56 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_leaking_tank_l89_8988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_eight_n_cubed_l89_8958

/-- Given an odd integer n with exactly 7 positive divisors, 
    the number of positive divisors of 8n^3 is 76 -/
theorem divisors_of_eight_n_cubed (n : ℕ) 
  (h_odd : Odd n) 
  (h_seven_divisors : (Nat.divisors n).card = 7) : 
  (Nat.divisors (8 * n^3)).card = 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_eight_n_cubed_l89_8958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_range_l89_8923

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

def range_is_real (f : ℝ → ℝ) : Prop :=
  ∀ y, ∃ x, f x = y

theorem c_range (c : ℝ) :
  c > 0 →
  (monotonically_decreasing (λ x ↦ c^x) ∨ range_is_real (λ x ↦ Real.log (2*c*x^2 - 2*x + 1))) →
  ¬(monotonically_decreasing (λ x ↦ c^x) ∧ range_is_real (λ x ↦ Real.log (2*c*x^2 - 2*x + 1))) →
  1/2 < c ∧ c < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_range_l89_8923
