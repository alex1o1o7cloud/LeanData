import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_fraction_l760_76037

theorem undefined_fraction (a : ℝ) : 
  (∀ x, (a + 3) / (a^2 - 4) ≠ x) ↔ a = -2 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_fraction_l760_76037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l760_76007

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def vector_addition (a b : V) : V := a + b
def vector_subtraction (a b : V) : V := a - b
def scalar_multiplication (c : ℝ) (a : V) : V := c • a
noncomputable def vector_magnitude (a : V) : ℝ := ‖a‖
def dot_product (a b : V) : ℝ := Inner.inner a b

theorem vector_operations (a b : V) (c : ℝ) :
  (∃ v : V, v = vector_addition a b) ∧
  (∃ v : V, v = vector_subtraction a b) ∧
  (∃ v : V, v = scalar_multiplication c a) ∧
  (∃ r : ℝ, r = vector_magnitude (a + b)) ∧
  (∃ r : ℝ, r = dot_product a b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l760_76007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l760_76049

/-- Given function f(x) = (ax+b)/(1+x²) where a and b are constants -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

/-- Theorem stating the properties of the function f -/
theorem f_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1) 1 → f a b x = -f a b (-x)) →  -- f is odd on (-1, 1)
  f a b (1/2) = 4/5 →                                  -- f(1/2) = 4/5
  (∃ a' b', ∀ x, f a b x = f a' b' x) ∧                -- There exist a', b' such that f(x) = f'(x)
  (∃ a' b', ∀ x, f a' b' x = 2*x / (1 + x^2)) ∧        -- f'(x) = 2x/(1+x²)
  (∀ x₁ x₂, x₁ ∈ Set.Ioo (-1) 1 → x₂ ∈ Set.Ioo (-1) 1 → x₁ < x₂ → 
    ∃ a' b', f a' b' x₁ < f a' b' x₂)                  -- f is increasing on (-1, 1)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l760_76049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_partition_size_l760_76062

def is_valid_partition (n : ℕ) (group1 group2 : Set ℕ) : Prop :=
  (∀ x, x ∈ group1 → x ≥ 2 ∧ x ≤ n) ∧
  (∀ x, x ∈ group2 → x ≥ 2 ∧ x ≤ n) ∧
  (group1 ∪ group2 = Finset.range (n - 1) \ {0, 1}) ∧
  (group1 ∩ group2 = ∅) ∧
  (∀ x y, x ∈ group1 → y ∈ group1 → x * y ∉ group1) ∧
  (∀ x y, x ∈ group2 → y ∈ group2 → x * y ∉ group2) ∧
  (∀ x, x ∈ group1 → x * x ∉ group1) ∧
  (∀ x, x ∈ group2 → x * x ∉ group2)

theorem max_partition_size :
  (∃ group1 group2 : Set ℕ, is_valid_partition 31 group1 group2) ∧
  (∀ n > 31, ¬ ∃ group1 group2 : Set ℕ, is_valid_partition n group1 group2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_partition_size_l760_76062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_exam_mean_median_difference_l760_76074

/-- Represents the distribution of scores in a math exam -/
structure ScoreDistribution where
  score70 : ℝ
  score80 : ℝ
  score85 : ℝ
  score90 : ℝ
  score95 : ℝ
  sum_to_one : score70 + score80 + score85 + score90 + score95 = 1

/-- Calculates the mean score given a score distribution -/
def mean (d : ScoreDistribution) : ℝ :=
  70 * d.score70 + 80 * d.score80 + 85 * d.score85 + 90 * d.score90 + 95 * d.score95

/-- Calculates the median score given a score distribution -/
noncomputable def median (d : ScoreDistribution) : ℝ :=
  if d.score70 + d.score80 < 0.5 ∧ d.score70 + d.score80 + d.score85 ≥ 0.5
  then 85
  else 0  -- This else case should never occur for the given distribution

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score70 = 0.1)
  (h2 : d.score80 = 0.25)
  (h3 : d.score85 = 0.2)
  (h4 : d.score90 = 0.15) :
  mean d - median d = 1 := by
  sorry

/-- Alternative formulation using a specific instance -/
def exam_distribution : ScoreDistribution := {
  score70 := 0.1,
  score80 := 0.25,
  score85 := 0.2,
  score90 := 0.15,
  score95 := 0.3,
  sum_to_one := by norm_num
}

theorem exam_mean_median_difference :
  mean exam_distribution - median exam_distribution = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_exam_mean_median_difference_l760_76074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_solution_l760_76069

theorem no_triangle_solution (a b : ℝ) (A : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 6) (h3 : A = 60 * π / 180) :
  ¬ ∃ (B : ℝ), 0 < B ∧ B < π ∧ Real.sin B = (b * Real.sin A) / a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_solution_l760_76069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_3_l760_76090

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x - 1

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ), Function.RightInverse f_inv f ∧ f_inv 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_3_l760_76090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l760_76022

-- Define the triangle ABC and points G, H, Q
variable (A B C G H Q : ℝ × ℝ × ℝ)

-- Define the conditions
axiom on_AC : ∃ t : ℝ, G = (1 - t) • A + t • C ∧ 0 ≤ t ∧ t ≤ 1
axiom on_AB : ∃ s : ℝ, H = (1 - s) • A + s • B ∧ 0 ≤ s ∧ s ≤ 1
axiom AG_GC_ratio : ∃ t : ℝ, G = (1 - t) • A + t • C ∧ t = 2 / 5
axiom AH_HB_ratio : ∃ s : ℝ, H = (1 - s) • A + s • B ∧ s = 3 / 4
axiom Q_on_BG : ∃ u : ℝ, Q = (1 - u) • B + u • G ∧ 0 ≤ u ∧ u ≤ 1
axiom Q_on_CH : ∃ v : ℝ, Q = (1 - v) • C + v • H ∧ 0 ≤ v ∧ v ≤ 1

-- The theorem to prove
theorem intersection_point :
  Q = (1/7 : ℝ) • A + (3/7 : ℝ) • B + (3/14 : ℝ) • C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l760_76022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pepperoni_coverage_half_l760_76024

/-- Represents a circular pizza with pepperoni slices -/
structure PepperoniPizza where
  pizza_diameter : ℝ
  pepperoni_count : ℕ
  pepperoni_across_diameter : ℕ

/-- Calculates the fraction of the pizza covered by pepperoni -/
noncomputable def pepperoni_coverage (p : PepperoniPizza) : ℝ :=
  let pepperoni_diameter := p.pizza_diameter / p.pepperoni_across_diameter
  let pepperoni_area := Real.pi * (pepperoni_diameter / 2) ^ 2
  let pizza_area := Real.pi * (p.pizza_diameter / 2) ^ 2
  (p.pepperoni_count : ℝ) * pepperoni_area / pizza_area

/-- Theorem stating that for the given pizza configuration, 
    the pepperoni coverage is 1/2 -/
theorem pepperoni_coverage_half : 
  let p : PepperoniPizza := { 
    pizza_diameter := 16, 
    pepperoni_count := 32, 
    pepperoni_across_diameter := 8 
  }
  pepperoni_coverage p = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pepperoni_coverage_half_l760_76024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_from_home_pattern_l760_76029

/-- Represents the percentage of working adults in Milton Town who worked from home in a given year -/
def WorkFromHomePercentage : ℕ → ℚ
  | 1980 => 7
  | 1990 => 15
  | 2000 => 12
  | 2010 => 25
  | _ => 0  -- For years not specified, we return 0

/-- Describes the pattern of change in the work from home percentage -/
inductive ChangePattern
  | SteadilyIncreasing
  | SteadilyDecreasing
  | IncreaseDecreaseThenSharpIncrease
  | DecreaseIncreaseThenSharpIncrease
  | Constant

/-- Determines the change pattern based on the given data -/
def determineChangePattern (data : ℕ → ℚ) : ChangePattern :=
  if data 1990 > data 1980 ∧ data 2000 < data 1990 ∧ data 2010 > data 2000 ∧ (data 2010 - data 2000 > data 1990 - data 1980)
  then ChangePattern.IncreaseDecreaseThenSharpIncrease
  else ChangePattern.SteadilyIncreasing  -- Default case, should not occur with given data

theorem work_from_home_pattern :
  determineChangePattern WorkFromHomePercentage = ChangePattern.IncreaseDecreaseThenSharpIncrease :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_from_home_pattern_l760_76029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_distance_is_twelve_l760_76048

/-- The distance between two consecutive trees in a yard -/
noncomputable def tree_distance (total_trees : ℕ) (yard_length : ℝ) : ℝ :=
  yard_length / (total_trees - 1)

/-- Theorem: The distance between consecutive trees is 12 meters -/
theorem tree_distance_is_twelve :
  tree_distance 26 300 = 12 :=
by
  -- Unfold the definition of tree_distance
  unfold tree_distance
  -- Simplify the arithmetic
  simp [Nat.cast_sub, Nat.cast_one]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_distance_is_twelve_l760_76048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l760_76043

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point over the y-axis -/
def reflectOverYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Reflect a point over the line y = -x -/
def reflectOverNegativeDiagonal (p : Point) : Point :=
  { x := -p.y, y := -p.x }

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  let a := p2.x - p1.x
  let b := p2.y - p1.y
  let c := p3.x - p1.x
  let d := p3.y - p1.y
  (1/2) * abs (a * d - b * c)

theorem triangle_DEF_area :
  let D : Point := { x := 5, y := 3 }
  let E : Point := reflectOverYAxis D
  let F : Point := reflectOverNegativeDiagonal E
  triangleArea D E F = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l760_76043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_coordinates_l760_76055

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 - 2*y^2 = 1

/-- The coordinates of a focus of the hyperbola -/
noncomputable def focus_coordinate : ℝ × ℝ := (Real.sqrt 6 / 2, 0)

/-- Theorem: The coordinates of the foci of the hyperbola x^2 - 2y^2 = 1 are (±√6/2, 0) -/
theorem hyperbola_foci_coordinates :
  ∀ (x y : ℝ), hyperbola_equation x y →
  (x = focus_coordinate.1 ∨ x = -focus_coordinate.1) ∧ y = focus_coordinate.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_coordinates_l760_76055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l760_76077

/-- The distance between Alice and Bob in miles -/
def distance : ℝ := 12

/-- The angle of elevation from Alice's perspective in radians -/
noncomputable def alice_angle : ℝ := Real.pi / 4

/-- The angle of elevation from Bob's perspective in radians -/
noncomputable def bob_angle : ℝ := Real.pi / 6

/-- The altitude of the airplane in miles -/
noncomputable def altitude : ℝ := 3 * Real.sqrt 3

theorem airplane_altitude :
  ∃ (alice_x alice_y bob_x bob_y plane_x plane_y : ℝ),
    alice_x ^ 2 + alice_y ^ 2 = altitude ^ 2 ∧
    (plane_x - bob_x) ^ 2 + plane_y ^ 2 = altitude ^ 2 ∧
    (plane_x - alice_x) ^ 2 + (plane_y - alice_y) ^ 2 = distance ^ 2 ∧
    alice_y / alice_x = Real.tan alice_angle ∧
    plane_y / (plane_x - bob_x) = Real.tan bob_angle :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l760_76077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_l760_76058

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  S : ℕ+ → ℚ
  h1 : a 5 = 28
  h2 : S 10 = 310
  h3 : ∀ n : ℕ+, S n = 3 * n.val ^ 2 + n.val

/-- The area of the triangle formed by three consecutive points on the graph of S -/
noncomputable def triangleArea (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  let A := (n.val, seq.S n)
  let B := ((n + 1).val, seq.S (n + 1))
  let C := ((n + 2).val, seq.S (n + 2))
  let area := (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2
  abs area

/-- The main theorem stating that the triangle area is always 3 -/
theorem triangle_area_is_three (seq : ArithmeticSequence) (n : ℕ+) :
  triangleArea seq n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_l760_76058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_fraction_identification_l760_76019

-- Define the concept of an algebraic fraction
def is_algebraic_fraction (f : ℝ → ℝ) : Prop :=
  ∃ (g h : ℝ → ℝ), (∀ x, f x = g x / h x) ∧ (∃ x, h x ≠ 0)

-- Define the given expressions
noncomputable def expr1 : ℝ → ℝ := λ _ => 1 / 3
noncomputable def expr2 : ℝ → ℝ := λ x => x / Real.pi
noncomputable def expr3 : ℝ → ℝ := λ x => 2 / (x + 3)
noncomputable def expr4 : ℝ → ℝ := λ x => (x + 2) / 3

-- State the theorem
theorem algebraic_fraction_identification :
  is_algebraic_fraction expr3 ∧
  ¬is_algebraic_fraction expr1 ∧
  ¬is_algebraic_fraction expr2 ∧
  ¬is_algebraic_fraction expr4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_fraction_identification_l760_76019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l760_76075

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the line passing through vertex and focus
noncomputable def line_equation (x y : ℝ) : Prop :=
  Real.sqrt 6 * x + 2 * y - 2 * Real.sqrt 6 = 0

-- Define the vertex and focus
noncomputable def vertex : ℝ × ℝ := (0, Real.sqrt 6)
noncomputable def focus : ℝ × ℝ := (2, 0)

-- Define the point P
noncomputable def point_P : ℝ × ℝ := (Real.sqrt 5, Real.sqrt 3)

theorem ellipse_and_tangent_line (e : Ellipse) 
  (h_vertex : line_equation vertex.1 vertex.2)
  (h_focus : line_equation focus.1 focus.2) :
  -- 1. Standard equation of the ellipse
  (∀ x y : ℝ, x^2 / 10 + y^2 / 6 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  -- 2. Equation of the tangent line
  (Real.sqrt 5 / 10 * point_P.1 + Real.sqrt 3 / 6 * point_P.2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l760_76075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l760_76002

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = (Real.sqrt 5 / 2) * t.b

-- Part 1
theorem part1 (t : Triangle) (h : triangle_conditions t) (h2 : t.C = 2 * t.B) :
  Real.cos t.B = Real.sqrt 5 / 4 :=
sorry

-- Part 2
theorem part2 (t : Triangle) (h : triangle_conditions t) 
  (h2 : t.b * t.c * Real.cos t.A = t.b * t.a * Real.cos t.C) :
  Real.cos (t.B + π/4) = -Real.sqrt 2 / 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l760_76002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_solution_l760_76088

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The original height of the cylinder -/
def originalHeight : ℝ := 5

/-- The increase in radius -/
def radiusIncrease : ℝ := 5

/-- The increase in height -/
def heightIncrease : ℝ := 7

theorem cylinder_radius_solution (r : ℝ) :
  (cylinderVolume (r + radiusIncrease) originalHeight - cylinderVolume r originalHeight =
   cylinderVolume r (originalHeight + heightIncrease) - cylinderVolume r originalHeight) →
  r = (25 - 10 * Real.sqrt 15) / 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_solution_l760_76088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_at_most_one_right_angle_l760_76057

theorem triangle_at_most_one_right_angle :
  ∀ (a b c : ℝ), 
  a + b + c = 180 → -- Sum of angles in a triangle is 180 degrees
  (a = 90 ∧ b = 90) → false := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_at_most_one_right_angle_l760_76057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_multiples_and_divisors_l760_76054

theorem common_multiples_and_divisors (a b : ℕ) (ha : a = 180) (hb : b = 300) :
  (Nat.lcm a b = 900) ∧ 
  (Finset.card (Finset.filter (fun x => x ∣ a ∧ x ∣ b) (Finset.range (Nat.min a b + 1))) = 12) ∧
  (Finset.filter (fun x => x ∣ a ∧ x ∣ b) (Finset.range (Nat.min a b + 1)) = 
   {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_multiples_and_divisors_l760_76054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_elements_not_set_l760_76092

-- Define a type for elements that may or may not have definite characteristics
inductive Element
| definite : Element
| indefinite : Element

-- Define a property for having definite characteristics
def has_definite_characteristics (e : Element) : Prop :=
  match e with
  | Element.definite => True
  | Element.indefinite => False

-- Define our own set-like structure to avoid conflict with existing definitions
def MySetLike (α : Type) := α → Prop

-- Theorem: A collection of elements without definite characteristics cannot form a set
theorem indefinite_elements_not_set :
  ∀ (s : MySetLike Element),
  (∃ (e : Element), s e ∧ ¬(has_definite_characteristics e)) →
  ¬(∃ (t : Set Element), ∀ (e : Element), s e ↔ e ∈ t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_elements_not_set_l760_76092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l760_76084

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) - 1/2

-- Theorem statement
theorem g_neither_even_nor_odd :
  (∃ x : ℝ, g (-x) ≠ g x) ∧ (∃ x : ℝ, g (-x) ≠ -g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l760_76084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_condition_l760_76015

/-- A monomial is a product of variables raised to non-negative integer powers -/
def IsMonomial (expr : ℕ → ℕ → ℚ) : Prop :=
  ∃ (c : ℚ) (m n : ℕ), ∀ x y, expr x y = c * x^m * y^n

/-- The sum of two expressions -/
def ExprSum (expr1 expr2 : ℕ → ℕ → ℚ) : ℕ → ℕ → ℚ :=
  λ x y => expr1 x y + expr2 x y

theorem monomial_sum_condition (m n : ℕ) :
  IsMonomial (ExprSum (λ x y => 2 * x^m * y^n) (λ x y => x * y^3)) →
  m = 1 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_condition_l760_76015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l760_76041

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) * (Real.cos x + Real.sin x)

-- State the theorem
theorem f_derivative (x : ℝ) : 
  deriv f x = -2 * Real.exp (-x) * Real.sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l760_76041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l760_76071

-- Define the condition function
def condition (m : ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → (2 * x - y / Real.exp 1) * Real.log (y / x) ≤ x / (m * Real.exp 1)

-- Define the theorem
theorem range_of_m (m : ℝ) :
  condition m ↔ m ∈ Set.Ioc 0 (1 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l760_76071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_equation_solution_l760_76063

theorem arccos_equation_solution :
  ∃ (x : ℝ), 
    x ∈ Set.Icc (-1 : ℝ) 1 ∧ 
    Real.arccos (3 * x) - Real.arccos x = π / 6 ∧
    (x = 1 / Real.sqrt (40 - 12 * Real.sqrt 3) ∨ 
     x = -(1 / Real.sqrt (40 - 12 * Real.sqrt 3))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_equation_solution_l760_76063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samsa_pasta_words_sum_l760_76004

def word_permutations (total : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial total / (repetitions.map Nat.factorial).prod

theorem samsa_pasta_words_sum : 
  (word_permutations 5 [2, 2]) + (word_permutations 5 [2]) = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_samsa_pasta_words_sum_l760_76004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_graph_matches_sequence_unique_correct_graph_l760_76012

/-- Represents the different phases of Mike's trip --/
inductive TripPhase
  | CityDriving
  | SuburbanDriving
  | HighwayDriving
  | MallStay

/-- Represents the slope of a line segment in the graph --/
inductive Slope
  | Gradual
  | Moderate
  | Steep
  | Horizontal

/-- Represents a segment of Mike's trip --/
structure TripSegment where
  phase : TripPhase
  slope : Slope

/-- Defines the correct sequence of trip segments for Mike's journey --/
def correctTripSequence : List TripSegment := [
  ⟨TripPhase.CityDriving, Slope.Gradual⟩,
  ⟨TripPhase.SuburbanDriving, Slope.Moderate⟩,
  ⟨TripPhase.HighwayDriving, Slope.Steep⟩,
  ⟨TripPhase.MallStay, Slope.Horizontal⟩,
  ⟨TripPhase.SuburbanDriving, Slope.Moderate⟩,
  ⟨TripPhase.HighwayDriving, Slope.Steep⟩,
  ⟨TripPhase.CityDriving, Slope.Gradual⟩
]

/-- Represents a graph of Mike's trip --/
structure TripGraph where
  segments : List TripSegment

/-- Predicate to check if a graph represents Mike's trip correctly --/
def representsTrip (g : TripGraph) : Prop :=
  g.segments = correctTripSequence

/-- Theorem stating that the correct graph must match the correct trip sequence --/
theorem correct_graph_matches_sequence (g : TripGraph) :
  representsTrip g ↔ g.segments = correctTripSequence :=
by sorry

/-- Theorem stating that only one graph option correctly represents Mike's trip --/
theorem unique_correct_graph (graphs : List TripGraph) (correct : TripGraph) :
  correct ∈ graphs ∧
  representsTrip correct ∧
  ∀ g ∈ graphs, g ≠ correct → ¬representsTrip g :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_graph_matches_sequence_unique_correct_graph_l760_76012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l760_76020

/-- The area of a quadrilateral given its diagonal and two offsets -/
noncomputable def quadrilateralArea (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) : ℝ :=
  (1/2) * diagonal * (offset1 + offset2)

/-- Theorem: Given a quadrilateral with one diagonal of 26 cm, one offset of 9 cm,
    and an area of 195 cm², the length of the second offset is 6 cm. -/
theorem second_offset_length 
  (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) (offset2 : ℝ)
  (h1 : diagonal = 26)
  (h2 : offset1 = 9)
  (h3 : area = 195)
  (h4 : quadrilateralArea diagonal offset1 offset2 = area) :
  offset2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l760_76020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l760_76053

/-- Given a triangle ABC with angles A, B, C and opposite sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The circumradius of the triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := 1 / (Real.sin t.B + Real.sin t.C)

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := t.a^2 - (t.b - t.c)^2

theorem triangle_properties (t : Triangle) :
  Real.tan t.A = 8/15 ∧ 
  ∀ s : Triangle, area s ≤ 4/17 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l760_76053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_l760_76089

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x * Real.exp 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.cos x

-- Define the derivatives of f and g
noncomputable def f' (x : ℝ) : ℝ := -Real.exp x - 1
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := a - 2 * Real.sin x

-- State the theorem
theorem perpendicular_tangents (a : ℝ) : 
  (∀ x y : ℝ, f' x * g' a y = -1) ↔ -1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_l760_76089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_rectangle_covering_l760_76017

-- Define the rectangle type
structure Rectangle where
  i : ℕ+
  j : ℕ+

-- Define the covers relation
def covers (r1 r2 : Rectangle) : Prop :=
  r1.i ≥ r2.i ∧ r1.j ≥ r2.j

-- State the theorem
theorem infinite_rectangle_covering (S : Set Rectangle) :
  Set.Infinite S →
  ∃ r1 r2, r1 ∈ S ∧ r2 ∈ S ∧ covers r1 r2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_rectangle_covering_l760_76017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_gold_coins_possible_l760_76080

/-- Represents a coin, which can be either gold or not gold -/
inductive Coin where
  | Gold
  | NotGold
deriving BEq, Repr

/-- Represents the result of a measurement -/
inductive Parity where
  | Even
  | Odd
deriving BEq, Repr

/-- Represents a device that can measure the parity of gold coins -/
def Device := List Coin → Parity

/-- Represents a strategy for finding gold coins -/
def Strategy := Device → List (List Coin)

/-- The number of coins -/
def total_coins : Nat := 6

/-- The number of gold coins -/
def gold_coins : Nat := 2

/-- The maximum number of allowed measurements -/
def max_measurements : Nat := 4

/-- Theorem stating that it's possible to find all gold coins in 4 or fewer measurements -/
theorem find_gold_coins_possible :
  ∃ (s : Strategy), ∀ (d : Device) (coins : List Coin),
    coins.length = total_coins →
    coins.count Coin.Gold = gold_coins →
    (s d).length ≤ max_measurements ∧
    ∀ (i : Nat), i < (s d).length → 
      ((s d).get! i).length ≤ coins.length ∧
      (s d).get! i ⊆ coins := by
  sorry

#check find_gold_coins_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_gold_coins_possible_l760_76080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_two_values_l760_76038

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define T as a function of n
noncomputable def T (n : ℤ) : ℂ := (1 + i)^n + (1 + i)^(-n)

-- Theorem stating that T can only take two distinct values
theorem T_two_values : ∃ (a b : ℂ), ∀ (n : ℤ), T n = a ∨ T n = b := by
  -- We'll use 0 and 2 as our two values
  use 0, 2
  intro n
  -- The proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_two_values_l760_76038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_slope_theorem_l760_76097

-- Define the midpoint of a line segment
def our_midpoint (x1 y1 x2 y2 : ℚ) : ℚ × ℚ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the slope between two points
def our_slope (x1 y1 x2 y2 : ℚ) : ℚ :=
  (y2 - y1) / (x2 - x1)

-- Theorem statement
theorem midpoint_slope_theorem : 
  let m1 := our_midpoint 2 3 4 5
  let m2 := our_midpoint 7 3 8 7
  our_slope m1.1 m1.2 m2.1 m2.2 = 2 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_slope_theorem_l760_76097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_future_time_sum_l760_76018

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Converts total seconds to Time structure -/
def secondsToTime (totalSeconds : Nat) : Time :=
  let hours := totalSeconds / 3600
  let minutes := (totalSeconds % 3600) / 60
  let seconds := totalSeconds % 60
  { hours := hours, minutes := minutes, seconds := seconds }

/-- Adds two Time structures -/
def addTime (t1 t2 : Time) : Time :=
  let totalSeconds := (t1.hours * 3600 + t1.minutes * 60 + t1.seconds) +
                      (t2.hours * 3600 + t2.minutes * 60 + t2.seconds)
  secondsToTime totalSeconds

/-- Converts 24-hour time to 12-hour time -/
def to12Hour (t : Time) : Time :=
  { hours := if t.hours % 12 == 0 then 12 else t.hours % 12,
    minutes := t.minutes,
    seconds := t.seconds }

theorem future_time_sum (startTime endTime : Time) : 
  startTime.hours = 15 ∧ 
  startTime.minutes = 0 ∧ 
  startTime.seconds = 0 ∧ 
  endTime = to12Hour (addTime startTime { hours := 287, minutes := 18, seconds := 53 }) →
  endTime.hours + endTime.minutes + endTime.seconds = 73 := by
  sorry

-- Remove the #eval line as it's not necessary and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_future_time_sum_l760_76018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_range_of_a_l760_76096

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Define the solution sets P and Q
def P (a : ℝ) : Set ℝ := {x | p x a}
def Q : Set ℝ := {x | q x}

-- Theorem 1: Intersection of P and Q when a = 1
theorem intersection_P_Q : P 1 ∩ Q = Set.Ioo 2 3 := by sorry

-- Define the negations of p and q
def not_p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 ≥ 0
def not_q (x : ℝ) : Prop := x ≤ 2 ∨ x > 3

-- Theorem 2: Range of a for which not_p is sufficient but not necessary for not_q
theorem range_of_a : 
  {a : ℝ | a > 0 ∧ (∀ x, not_p x a → not_q x) ∧ (∃ x, not_q x ∧ ¬(not_p x a))} = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_range_of_a_l760_76096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_matrix_M_l760_76027

noncomputable section

open Matrix

def M : Matrix (Fin 2) (Fin 2) ℝ := !![4, 0; 0, 1]

def N : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, (1/2)]

theorem find_matrix_M : 
  (M * N)⁻¹ = !![1/4, 0; 0, 2] := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_matrix_M_l760_76027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l760_76098

open Real

noncomputable def g (A : ℝ) : ℝ :=
  (cos A * (4 * sin A ^ 2 + sin A ^ 4 + 4 * cos A ^ 2 + cos A ^ 2 * sin A ^ 2)) /
  ((cos A / sin A) * (1 / sin A - cos A * (cos A / sin A)))

theorem g_range :
  ∀ A : ℝ, (∀ n : ℤ, A ≠ n * π) →
  ∃ y : ℝ, g A = y ∧ -17/4 < y ∧ y < 17/4 := by
  sorry

#check g_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l760_76098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_m_value_l760_76028

/-- The function f(x) = 3sin(x) + 4cos(x) -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x

/-- The angle θ that maximizes f(x) -/
noncomputable def θ : ℝ := Real.arctan (3 / 4)

theorem max_point_m_value (h : ∀ x : ℝ, f x ≤ f θ) :
  let m : ℝ := 4 * Real.tan θ
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_m_value_l760_76028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_implies_relation_l760_76051

def u : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 2 * u (n + 1) - 3 * u n

def v (a b c : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => b
  | 2 => c
  | (n + 3) => v a b c (n + 2) - 3 * v a b c (n + 1) + 27 * v a b c n

theorem sequence_divisibility_implies_relation (a b c : ℤ) :
  (∃ N : ℕ, ∀ n ≥ N, (u n ∣ v a b c n)) →
  3 * a = 2 * b + c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_implies_relation_l760_76051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_negative_sum_l760_76076

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem largest_negative_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a10_neg : a 10 < 0)
  (h_a11_pos : a 11 > 0)
  (h_a11_gt_abs_a10 : a 11 > |a 10|) :
  (∀ n : ℕ, S a n < 0 → S a n ≤ S a 19) ∧ S a 19 < 0 ∧ S a 20 > 0 := by
  sorry

#check largest_negative_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_negative_sum_l760_76076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_velocity_comparison_l760_76072

-- Define the setup
structure ProjectionSetup where
  l : ℝ → ℝ × ℝ  -- Parametric equation of line l
  m : ℝ → ℝ × ℝ  -- Parametric equation of line m
  X : ℝ × ℝ      -- Point X
  Y : ℝ × ℝ      -- Point Y
  v_X : ℝ        -- Velocity of X
  v_Y : ℝ        -- Velocity of Y
  θ : ℝ          -- Angle between lines l and m

-- Define the theorem
theorem shadow_velocity_comparison :
  ∃ (setup1 setup2 : ProjectionSetup),
    setup1.v_Y > setup1.v_X ∧ setup2.v_Y < setup2.v_X :=
by
  -- We'll use sorry to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_velocity_comparison_l760_76072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l760_76078

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y + 2)^2 = 9

-- Define a line
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := hyperbola x y ∧ x > 0 ∧ y = 0

-- Define the longest chord property
def longest_chord (a b c : ℝ) : Prop :=
  ∀ x y, circle_eq x y → line a b c x y → 
  ∀ a' b' c', (∃ x' y', circle_eq x' y' ∧ line a' b' c' x' y') →
  (∃ x1 y1 x2 y2, circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ 
   line a b c x1 y1 ∧ line a b c x2 y2 ∧
   (x1 - x2)^2 + (y1 - y2)^2 ≥ 
   (x' - x'')^2 + (y' - y'')^2)
  where
    x' := by sorry
    y' := by sorry
    x'' := by sorry
    y'' := by sorry

theorem line_equation : 
  ∃ x y, right_focus x y ∧ longest_chord 1 (-1) (-2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l760_76078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_implies_a_range_l760_76036

theorem log_condition_implies_a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ t : ℝ, -2*t^2 + 7*t - 5 > 0 → t^2 - (a+3)*t + (a+2) < 0) ∧
  (∃ t : ℝ, t^2 - (a+3)*t + (a+2) < 0 ∧ -2*t^2 + 7*t - 5 ≤ 0) →
  a > 1/2 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_implies_a_range_l760_76036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_minus_y_l760_76082

theorem max_value_x_minus_y (x y : Real) (h1 : Real.tan x = 3 * Real.tan y) (h2 : 0 ≤ y) (h3 : y ≤ x) (h4 : x < π / 2) :
  (∀ a b : Real, Real.tan a = 3 * Real.tan b ∧ 0 ≤ b ∧ b ≤ a ∧ a < π / 2 → x - y ≥ a - b) ∧ x - y ≤ π / 6 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_minus_y_l760_76082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_sum_magnitude_l760_76010

/-- Definition of the ellipse -/
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 4) + (P.2^2 / 3) = 1

/-- Left vertex of the ellipse -/
def A : ℝ × ℝ := (-2, 0)

/-- Right focus of the ellipse -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Vector from P to A -/
def PA (P : ℝ × ℝ) : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)

/-- Vector from P to F₂ -/
def PF₂ (P : ℝ × ℝ) : ℝ × ℝ := (F₂.1 - P.1, F₂.2 - P.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Sum of two 2D vectors -/
def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

/-- The main theorem -/
theorem ellipse_vector_sum_magnitude :
  ∃ (P : ℝ × ℝ), is_on_ellipse P ∧
    (∀ (Q : ℝ × ℝ), is_on_ellipse Q →
      dot_product (PF₂ P) (PA P) ≤ dot_product (PF₂ Q) (PA Q)) ∧
    magnitude (vector_sum (PA P) (PF₂ P)) = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_sum_magnitude_l760_76010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l760_76060

-- Define the logarithm function for base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_properties :
  (log5 ((-4)^2) = 2 * log5 4) ∧
  (log5 ((-2) * (-3)) = log5 2 + log5 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l760_76060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_inequality_l760_76035

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the semiperimeter of a triangle
noncomputable def semiperimeter (t : Triangle) : ℝ := 
  (t.a + t.b + t.c) / 2

-- Define the area of a triangle using Heron's formula
noncomputable def area (t : Triangle) : ℝ := 
  let s := semiperimeter t
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

-- Define the circumradius of a triangle
noncomputable def circumradius (t : Triangle) : ℝ := 
  t.a * t.b * t.c / (4 * area t)

-- Define the inradius of a triangle
noncomputable def inradius (t : Triangle) : ℝ := 
  area t / semiperimeter t

-- Define an equilateral triangle
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- State the theorem
theorem euler_inequality (t : Triangle) : 
  circumradius t ≥ 2 * inradius t ∧ 
  (circumradius t = 2 * inradius t ↔ is_equilateral t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_inequality_l760_76035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_function_intersection_l760_76067

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

noncomputable def g (x : ℝ) : ℝ := (x^2 + x - 1) / (x + 1)

theorem function_properties (a : ℝ) (h1 : a > 1) :
  (∀ x ∈ Set.Icc 1 a, f a x ∈ Set.Icc 1 a) ∧
  (Set.range (f a) = Set.Icc 1 a) →
  a = 2 := by
  sorry

theorem function_intersection (a : ℝ) (h1 : a > 1) :
  (∀ x ∈ Set.Icc 0 1, ∃ x_0 ∈ Set.Icc 0 1, f a x_0 = g x) →
  a ≥ 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_function_intersection_l760_76067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_digit_divisibility_l760_76014

theorem repeated_digit_divisibility (μ : ℕ) (h : 100 ≤ μ ∧ μ < 1000) :
  ∀ (d : ℕ), d ∈ ({7, 11, 13} : Set ℕ) → (1001 * μ) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_digit_divisibility_l760_76014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_price_reduction_l760_76091

theorem computer_price_reduction : 
  ∀ (original_price : ℝ), 
  original_price > 0 →
  let first_discount := 0.2
  let second_discount := 0.3
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  (original_price - final_price) / original_price = 0.44 := by
  intro original_price h_positive
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_price_reduction_l760_76091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l760_76047

/-- An arithmetic sequence with first term 0 and non-zero common difference -/
structure ArithmeticSequence where
  d : ℚ
  h_d_neq_0 : d ≠ 0

/-- The n-th term of the arithmetic sequence -/
def ArithmeticSequence.a (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n - 1 : ℚ) * seq.d

/-- The sum of the first n terms of the arithmetic sequence -/
def ArithmeticSequence.sum (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * ((n - 1 : ℚ) * seq.d) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  ∃ m : ℕ, seq.a m = seq.sum 9 ∧ m = 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l760_76047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decrease_interval_g_smallest_positive_m_for_even_l760_76073

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * (x - m) + Real.pi / 3)

def isMonotonicDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_monotonic_decrease_interval (k : ℤ) :
  isMonotonicDecreasing f (Real.pi/12 + k*Real.pi) (7*Real.pi/12 + k*Real.pi) := by sorry

theorem g_smallest_positive_m_for_even :
  (∀ m : ℝ, m > 0 → m < 5*Real.pi/12 → ¬isEven (g m)) ∧ isEven (g (5*Real.pi/12)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decrease_interval_g_smallest_positive_m_for_even_l760_76073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_time_correct_l760_76083

/-- The time at which the distance between two bugs reaches its minimum -/
noncomputable def minimum_distance_time (l : ℝ) (v_A v_B : ℝ) : ℝ :=
  1000 / 250

theorem minimum_distance_time_correct (l v_A v_B : ℝ) 
  (h1 : l = 1)  -- length of triangle legs in meters
  (h2 : v_A = 5)  -- speed of bug A in cm/s
  (h3 : v_B = 2 * v_A)  -- speed of bug B in cm/s
  : minimum_distance_time l v_A v_B = 4 := by
  sorry

#check minimum_distance_time_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_time_correct_l760_76083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_five_l760_76068

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Calculates the area of a triangle given two points in polar coordinates -/
noncomputable def triangleArea (a b : PolarPoint) : ℝ :=
  (1/2) * a.r * b.r * Real.sin (b.θ - a.θ)

/-- Theorem: The area of triangle AOB is 5 -/
theorem triangle_area_is_five : 
  let a : PolarPoint := ⟨5, 5*π/6⟩
  let b : PolarPoint := ⟨2, π/3⟩
  triangleArea a b = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_five_l760_76068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_f_l760_76030

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  2^x / (1 + 2^x) - 1/2

-- State the theorem
theorem range_of_floor_f :
  ∀ y : ℤ, (∃ x : ℝ, floor (f x) = y) ↔ y = 0 ∨ y = -1 :=
by
  sorry

#check range_of_floor_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_f_l760_76030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_range_of_a_l760_76040

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 2 ≤ (4 : ℝ)^x ∧ (4 : ℝ)^x ≤ 8}
def C (a : ℝ) : Set ℝ := {x | a - 4 < x ∧ x ≤ 2*a - 7}

-- Theorem 1: Prove that (∁_U A) ∩ B = {x | 1 ≤ x ≤ 3/2}
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 3/2} := by
  sorry

-- Theorem 2: If A ∩ C = C, then a ∈ (-∞, 4)
theorem range_of_a (a : ℝ) (h : A ∩ C a = C a) :
  a < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_range_of_a_l760_76040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_specific_circles_l760_76005

/-- Circle in 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Distance between two points in 2D plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Maximum distance between points on two circles -/
noncomputable def max_distance_between_circles (c1 c2 : Circle) : ℝ :=
  distance c1.center c2.center + c1.radius + c2.radius

theorem max_distance_between_specific_circles :
  let c1 : Circle := ⟨(3, 4), 2⟩
  let c2 : Circle := ⟨(0, 0), 1⟩
  max_distance_between_circles c1 c2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_specific_circles_l760_76005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l760_76034

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of the intersecting line -/
def line (x y : ℝ) : Prop := y = x + 1/2

/-- Points A and B are on both the ellipse and the line -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ line A.1 A.2 ∧ ellipse B.1 B.2 ∧ line B.1 B.2

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Main theorem: The distance between intersection points A and B is 2√11 / 3 -/
theorem intersection_distance (A B : ℝ × ℝ) :
  intersection_points A B → distance A B = 2 * Real.sqrt 11 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l760_76034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excenter_locus_l760_76000

/-- The hyperbola equation --/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The locus equation --/
def locus (x y : ℝ) : Prop := x^2 / 25 - 9 * y^2 / 25 = 1

/-- The theorem statement --/
theorem excenter_locus :
  ∀ (x₀ y₀ x y : ℝ),
    hyperbola x₀ y₀ →
    x₀ > 4 →
    (∃ (k : ℝ), k = 5/4 ∧ 
      x = (4 * (5/3 * x₀) + 5 * x₀) / 9 ∧
      y = 5 * y₀ / 9) →
    locus x y ∧ x > 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excenter_locus_l760_76000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_propositions_true_l760_76056

-- Define the propositions
def proposition1 : Prop := ∀ (a b : Real), a = b → a > 0 → b > 0
def proposition2 : Prop := ∀ (x y : Real), x + y = y + x
def proposition3 : Prop := ∀ (n : Nat), n > 0 → n * n ≥ n
def proposition4 : Prop := ∀ (a b c : Real), a + b > c → b + c > a → c + a > b

-- Theorem statement
theorem exactly_two_propositions_true :
  (proposition1 = False ∧
   proposition2 = True ∧
   proposition3 = False ∧
   proposition4 = True) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_propositions_true_l760_76056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_count_l760_76001

/-- A point in the 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Checks if a point is on the line segment between two given points -/
def isOnSegment (p q r : Point) : Prop :=
  (min p.x q.x ≤ r.x ∧ r.x ≤ max p.x q.x) ∧
  (min p.y q.y ≤ r.y ∧ r.y ≤ max p.y q.y)

/-- The set of all lattice points on the line segment -/
def latticePointsOnSegment (p q : Point) : Set Point :=
  {r : Point | isOnSegment p q r}

/-- The statement to be proved -/
theorem lattice_points_count :
  let p : Point := ⟨3, 17⟩
  let q : Point := ⟨48, 281⟩
  ∃ (s : Finset Point), s.card = 4 ∧ ∀ r ∈ s, r ∈ latticePointsOnSegment p q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_count_l760_76001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l760_76099

noncomputable def f (x : ℝ) : ℝ := |Real.log (x + 1)|

theorem function_values (a b : ℝ) (h1 : a < b) 
  (h2 : f a = f (-((b + 1) / (b + 2))))
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) :
  a = -2/5 ∧ b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l760_76099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zhou_yu_age_equation_l760_76065

/-- Represents the tens digit of Zhou Yu's age at death -/
def x : ℕ := sorry

/-- Zhou Yu's age is a two-digit number -/
axiom two_digit : 10 ≤ 10 * x + (x + 3) ∧ 10 * x + (x + 3) < 100

/-- The tens digit is three less than the units digit -/
axiom digit_relation : x + 3 = (10 * x + (x + 3)) % 10

/-- The units digit squared equals the lifespan -/
axiom lifespan_equation : (x + 3)^2 = 10 * x + (x + 3)

/-- The equation correctly represents Zhou Yu's age -/
theorem zhou_yu_age_equation : 10 * x + (x + 3) = (x + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zhou_yu_age_equation_l760_76065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_guaranteed_relationship_l760_76066

theorem no_guaranteed_relationship 
  (a b c d : ℕ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ∃ (P Q X M N Y : ℕ),
    P = Nat.gcd a b ∧
    Q = Nat.gcd c d ∧
    X = Nat.lcm P Q ∧
    M = Nat.lcm 2 6 ∧
    N = Nat.lcm c d ∧
    Y = Nat.gcd M N ∧
    (¬(X ∣ Y) ∧ ¬(Y ∣ X) ∧ X ≠ Y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_guaranteed_relationship_l760_76066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zero_difference_l760_76064

/-- A parabola passing through two points with a given vertex -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_point1 : a * 3^2 + b * 3 + c = -9
  passes_through_point2 : a * 5^2 + b * 5 + c = 7
  vertex_x : ℝ
  vertex_y : ℝ
  is_vertex : vertex_x = 3 ∧ vertex_y = -9

/-- The difference between the larger and smaller zeros of the quadratic equation -/
noncomputable def zero_difference (p : Parabola) : ℝ :=
  let x1 := (-p.b + Real.sqrt (p.b^2 - 4*p.a*p.c)) / (2*p.a)
  let x2 := (-p.b - Real.sqrt (p.b^2 - 4*p.a*p.c)) / (2*p.a)
  abs (x1 - x2)

/-- The main theorem stating that the difference between zeros is 3 -/
theorem parabola_zero_difference (p : Parabola) : zero_difference p = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zero_difference_l760_76064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l760_76086

-- Define the planar region
def is_in_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + Real.sqrt 3 * y - Real.sqrt 3 ≤ 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - Real.sqrt 3 / 2)^2 + (y - 1/2)^2 = 1

-- Define a line with slope √3
def line_with_slope_sqrt3 (m : ℝ) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x + m

-- Main theorem
theorem circle_and_line_properties :
  (∀ x y, is_in_region x y → circle_C x y) ∧
  (∀ m, (∃ A B : ℝ × ℝ, circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
                line_with_slope_sqrt3 m A.1 A.2 ∧
                line_with_slope_sqrt3 m B.1 B.2 ∧
                (A.1 - B.1)^2 + (A.2 - B.2)^2 = 3) →
        (m = 0 ∨ m = -2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l760_76086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_floors_l760_76021

/-- Represents the number of apartments per floor -/
def K : ℕ := sorry

/-- Represents the number of floors -/
def E : ℕ := sorry

/-- Represents the number of entrances -/
def P : ℕ := sorry

/-- The total number of apartments in the building -/
def total_apartments : ℕ := 715

theorem apartment_floors :
  (1 < K) →
  (K < E) →
  (E < P) →
  (K * E * P = total_apartments) →
  (E = 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_floors_l760_76021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_bounds_l760_76006

theorem triangle_cosine_sum_bounds (α β γ R r : Real) :
  α + β + γ = Real.pi →
  R > 0 →
  r > 0 →
  Real.cos α + Real.cos β + Real.cos γ = (R + r) / R →
  r ≤ R / 2 →
  1 < Real.cos α + Real.cos β + Real.cos γ ∧ Real.cos α + Real.cos β + Real.cos γ ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_bounds_l760_76006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l760_76013

/-- Parabola defined by y = 1/2 x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/2) * x^2

/-- Point on the parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- Tangent line to the parabola at a given point -/
def tangent_line (p : ParabolaPoint) (x y : ℝ) : Prop :=
  y - p.y = p.x * (x - p.x)

/-- Line passing through (1,1) and intersecting the parabola at two points -/
structure IntersectingLine where
  k : ℝ
  eq : (x y : ℝ) → Prop
  passes_through_Q : eq 1 1
  intersects_parabola : ∃ (a b : ParabolaPoint), a.x ≠ b.x ∧ eq a.x a.y ∧ eq b.x b.y

/-- Point of intersection of tangent lines -/
noncomputable def intersection_point (a b : ParabolaPoint) : ℝ × ℝ :=
  ((a.x + b.x) / 2, (a.x * b.x) / 2)

/-- Trajectory of intersection point P -/
def trajectory (x y : ℝ) : Prop := y = x - 1

/-- Area of triangle PAB -/
noncomputable def triangle_area (p : ℝ × ℝ) (a b : ParabolaPoint) : ℝ :=
  let k := p.1
  Real.sqrt ((k^2 - 2*k + 2)^3)

theorem parabola_intersection_theorem :
  ∀ (l : IntersectingLine),
  ∃ (p : ℝ × ℝ),
    (∀ (x y : ℝ), trajectory x y ↔ x = p.1 ∧ y = p.2) ∧
    (∀ (a b : ParabolaPoint), l.eq a.x a.y → l.eq b.x b.y → a.x ≠ b.x →
      triangle_area p a b ≥ 1) ∧
    (∃ (a b : ParabolaPoint), l.eq a.x a.y ∧ l.eq b.x b.y ∧ a.x ≠ b.x ∧
      triangle_area p a b = 1 ∧ l.eq = (λ x y => y = x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l760_76013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l760_76042

-- Define the polynomial g(x)
def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem sum_of_coefficients (a b c d : ℝ) :
  g a b c d (2*Complex.I) = 0 ∧ g a b c d (1 + Complex.I) = 0 → a + b + c + d = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l760_76042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equals_eight_l760_76016

theorem power_product_equals_eight (x y : ℝ) (h : x + 2 * y - 3 = 0) :
  (2 : ℝ)^x * (4 : ℝ)^y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equals_eight_l760_76016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_profit_per_minute_l760_76095

-- Define the departure time interval
noncomputable def t : ℝ := sorry

-- Define the passenger capacity function
noncomputable def p (t : ℝ) : ℝ :=
  if 10 ≤ t ∧ t ≤ 20 then 1300
  else if 2 ≤ t ∧ t < 10 then 1300 - 10 * (10 - t)^2
  else 0

-- Define the net profit per minute function
noncomputable def Q (t : ℝ) : ℝ := (6 * p t - 3960) / t - 350

-- State the theorem
theorem max_net_profit_per_minute :
  (∀ t, 2 ≤ t ∧ t ≤ 20 → Q t ≤ 130) ∧
  (Q 6 = 130) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_profit_per_minute_l760_76095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shape_l760_76061

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the set of points satisfying the equation
def ConeSet (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = Real.pi/2 - c}

-- Define a type for 3D vectors
def Vector3D := ℝ × ℝ × ℝ

-- Define a predicate for cone surface
def IsConeSurface (s : Set Vector3D) (vertex : Vector3D) (axis : Vector3D) (angle : ℝ) : Prop :=
  sorry -- The actual definition would go here

-- Theorem stating that the set forms a cone
theorem cone_shape (c : ℝ) : 
  ∃ (vertex : Vector3D) (axis : Vector3D) (angle : ℝ), 
    IsConeSurface (fun p : Vector3D => sorry) vertex axis angle :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shape_l760_76061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l760_76087

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

noncomputable def g (x : ℝ) := 2 * Real.sin (2 * x)

def is_symmetry_center (h : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x, h (c.1 + x) = h (c.1 - x)

theorem symmetry_center_of_g :
  (∀ x, g x = f (x - π/6)) →
  is_symmetry_center g (π/2, 0) := by
  intro h
  sorry

#check symmetry_center_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l760_76087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l760_76044

theorem max_value_theorem (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b = 100) :
  (∀ x y : ℝ, x > 1 → y > 1 → x * y = 100 → 
    x^((Real.log y)^2) ≤ a^((Real.log b)^2)) →
  a^((Real.log b)^2) = 10^(32/27) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l760_76044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_tangent_fraction_l760_76052

open Real

theorem integral_tangent_fraction :
  ∫ x in (0 : ℝ)..(π / 4), (2 * Real.tan x ^ 2 - 11 * Real.tan x - 22) / (4 - Real.tan x) = 2 * log (2 / 3) - 5 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_tangent_fraction_l760_76052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l760_76009

noncomputable def hour_hand_speed : ℝ := 30
noncomputable def minute_hand_speed : ℝ := 6

noncomputable def hour_hand_position (hours : ℝ) (minutes : ℝ) : ℝ :=
  (hours * hour_hand_speed) + (minutes * hour_hand_speed / 60)

noncomputable def minute_hand_position (minutes : ℝ) : ℝ :=
  minutes * minute_hand_speed

noncomputable def angle_between_hands (hours : ℝ) (minutes : ℝ) : ℝ :=
  abs (minute_hand_position minutes - hour_hand_position hours minutes)

theorem clock_angle_at_3_30 :
  angle_between_hands 3 30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l760_76009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_half_l760_76093

-- Define g(n) for positive integers n
noncomputable def g (n : ℕ+) : ℝ := ∑' k : ℕ+, (1 : ℝ) / ((k + 2 : ℝ) ^ n.val)

-- State the theorem
theorem sum_of_g_equals_half : ∑' n : ℕ+, g n = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_half_l760_76093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_maximum_l760_76079

def f (x : ℝ) : ℝ := x * abs (x - 4)

theorem f_monotonicity_and_maximum :
  (∀ x y, x < y ∧ ((x ≤ 2 ∧ y ≤ 2) ∨ (4 ≤ x ∧ 4 ≤ y)) → f x < f y) ∧
  (∀ x y, 2 ≤ x ∧ x < y ∧ y ≤ 4 → f x > f y) ∧
  (∀ m, m > 0 →
    (0 < m ∧ m < 2 → ∀ x ∈ Set.Icc 0 m, f x ≤ m * (4 - m)) ∧
    (2 ≤ m ∧ m ≤ 2 + 2 * Real.sqrt 2 → ∀ x ∈ Set.Icc 0 m, f x ≤ 4) ∧
    (m > 2 + 2 * Real.sqrt 2 → ∀ x ∈ Set.Icc 0 m, f x ≤ m * (4 - m))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_maximum_l760_76079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_midpoint_coordinates_l760_76031

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

def line_equation (x y : ℝ) : Prop :=
  y = 4/5 * (x - 3)

theorem ellipse_midpoint_coordinates (a b : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b 0 4 ∧
  eccentricity a b = 3/5 →
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse a b x₁ y₁ ∧
    ellipse a b x₂ y₂ ∧
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    (x₁ + x₂) / 2 = 3/2 ∧
    (y₁ + y₂) / 2 = -6/5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_midpoint_coordinates_l760_76031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candidates_per_state_candidates_per_state_proof_l760_76059

theorem candidates_per_state : ℕ :=
  let total_candidates_A : ℕ := 8400
  let total_candidates_B : ℕ := 8400
  let selected_rate_A : ℚ := 6 / 100
  let selected_rate_B : ℚ := 7 / 100
  let selected_difference : ℕ := 84

  -- Assumptions
  have h1 : total_candidates_A = total_candidates_B := rfl
  have h2 : (total_candidates_A : ℚ) * selected_rate_A + selected_difference = (total_candidates_B : ℚ) * selected_rate_B := by
    -- The proof of this equality would go here
    sorry

  total_candidates_A

theorem candidates_per_state_proof : candidates_per_state = 8400 := by
  -- The proof of this equality would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candidates_per_state_candidates_per_state_proof_l760_76059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_half_time_is_22_hours_l760_76085

/-- Represents a runner's journey with changing speed --/
structure RunnerJourney where
  totalDistance : ℝ
  initialSpeed : ℝ
  speedReductionFactor : ℝ
  timeDifference : ℝ

/-- Calculates the time taken for the second half of the journey --/
noncomputable def secondHalfTime (journey : RunnerJourney) : ℝ :=
  (journey.totalDistance / 2) / (journey.initialSpeed * journey.speedReductionFactor)

/-- Theorem stating the time taken for the second half of the specific journey --/
theorem second_half_time_is_22_hours (journey : RunnerJourney) 
  (h1 : journey.totalDistance = 40)
  (h2 : journey.speedReductionFactor = 1/2)
  (h3 : secondHalfTime journey - (journey.totalDistance / 2) / journey.initialSpeed = journey.timeDifference)
  (h4 : journey.timeDifference = 11) : 
  secondHalfTime journey = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_half_time_is_22_hours_l760_76085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l760_76008

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/3)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∃ M, ∀ x, f x ≤ M ∧ f (Real.pi/2) < M) ∧
  (∀ x, f x = Real.sin (x + Real.pi/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l760_76008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_learning_hours_l760_76046

/-- Represents the daily hours spent on learning a language -/
def LearningHours : Type := ℕ

/-- Ryan's daily hours spent learning English -/
def english_hours : ℕ := 7

/-- The difference between English and Chinese learning hours -/
def hour_difference : ℕ := 2

/-- Theorem stating that Ryan spends 5 hours learning Chinese -/
theorem chinese_learning_hours :
  ∃ (chinese_hours : ℕ),
    chinese_hours = english_hours - hour_difference ∧
    chinese_hours = 5 := by
  use 5
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_learning_hours_l760_76046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_through_vertices_l760_76032

/-- Predicate to check if a set of points in ℝ² forms a square -/
def is_square (s : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Function to get the side length of a square -/
def side_length (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Predicate to check if the sides of one square pass through the vertices of another -/
def sides_pass_through_vertices (larger : Set (ℝ × ℝ)) (smaller : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Given two squares, one with side length a and another with side length b,
    where the sides of the larger square pass through the vertices of the smaller square,
    prove that a/2 * √2 < b ≤ a * √2 must hold. -/
theorem square_through_vertices (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_passes_through : ∃ (larger_square smaller_square : Set (ℝ × ℝ)),
    is_square larger_square ∧ 
    is_square smaller_square ∧
    side_length larger_square = b ∧
    side_length smaller_square = a ∧
    sides_pass_through_vertices larger_square smaller_square) :
  a / 2 * Real.sqrt 2 < b ∧ b ≤ a * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_through_vertices_l760_76032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_inequality_solution_l760_76045

noncomputable def f (x : ℝ) : ℝ := x + 1 / (2 * x) + 2

theorem f_increasing_and_inequality_solution :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  {x : ℝ | f (2 * x - 1/2) < f (x + 1007)} = Set.Icc (3/4 : ℝ) (2015/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_inequality_solution_l760_76045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l760_76070

-- Define the points as vectors
def A : Fin 3 → ℝ := ![2, 0, -2]
def B : Fin 3 → ℝ := ![1, -1, -2]
def C : Fin 3 → ℝ := ![3, 0, -4]

-- Define vectors
def a : Fin 3 → ℝ := B - A
def b : Fin 3 → ℝ := C - A

-- Define BC vector
def BC : Fin 3 → ℝ := C - B

-- Function to calculate dot product
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Function to calculate magnitude
noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt (dot_product v v)

-- Theorem for part 1
theorem part1 : 
  ∃ (c : Fin 3 → ℝ), 
    (magnitude c = 3) ∧ 
    (∃ (t : ℝ), c = fun i => t * (BC i)) ∧
    (c = ![2, 1, -2] ∨ c = ![-2, -1, 2]) :=
sorry

-- Theorem for part 2
theorem part2 : 
  magnitude a * magnitude b * Real.sqrt (1 - (dot_product a b / (magnitude a * magnitude b))^2) = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l760_76070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_shift_l760_76033

theorem cos_graph_shift (x : ℝ) :
  Real.cos (2 * x - π / 4) = Real.cos (2 * (x - π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_shift_l760_76033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_cosine_rule_l760_76003

/-- Represents a tetrahedron with face areas and dihedral angles -/
structure Tetrahedron where
  S₀ : ℝ
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  α₁₂ : ℝ
  α₁₃ : ℝ
  α₂₃ : ℝ

/-- The cosine rule for a tetrahedron -/
theorem tetrahedron_cosine_rule (t : Tetrahedron) :
  t.S₀^2 = t.S₁^2 + t.S₂^2 + t.S₃^2 - 2*t.S₁*t.S₂*Real.cos t.α₁₂ - 2*t.S₁*t.S₃*Real.cos t.α₁₃ - 2*t.S₂*t.S₃*Real.cos t.α₂₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_cosine_rule_l760_76003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sum_is_96_l760_76050

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- Dimensions in geometric progression
  a : ℝ
  r : ℝ
  -- Volume constraint
  volume_eq : a * a * a = 512
  -- Surface area constraint
  surface_area_eq : 2 * (a^2 / r + a^2 + a^2 * r) = 384
  -- Geometric progression constraint
  geo_prog : r > 0

/-- The sum of lengths of all edges of the rectangular solid -/
noncomputable def edge_sum (solid : RectangularSolid) : ℝ :=
  4 * (solid.a / solid.r + solid.a + solid.a * solid.r)

/-- Theorem stating the sum of edge lengths for the specific solid -/
theorem edge_sum_is_96 (solid : RectangularSolid) : edge_sum solid = 96 := by
  sorry

#check edge_sum_is_96

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sum_is_96_l760_76050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_a_n_l760_76025

noncomputable def a (n : ℕ) : ℝ := (3/4)^(n-1) * ((3/4)^(n-1) - 1)

theorem max_min_a_n :
  (∀ n : ℕ, n ≥ 1 → a 1 ≥ a n) ∧ 
  (∀ n : ℕ, n ≥ 1 → a 3 ≤ a n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_a_n_l760_76025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l760_76026

theorem tan_alpha_plus_pi_fourth (α : ℝ) : 
  Real.tan (α - π/4) = 1/4 → Real.tan (α + π/4) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l760_76026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l760_76023

theorem congruent_integers_count : 
  (Finset.filter (fun n : ℕ => n > 0 ∧ n < 500 ∧ n % 7 = 3) (Finset.range 500)).card = 71 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l760_76023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_savings_proof_l760_76094

/-- Calculates the amount of money Julie will have left after purchasing a mountain bike -/
def julie_savings (initial_savings : ℕ) (bike_cost : ℕ) 
  (lawns : ℕ) (lawn_pay : ℕ) 
  (newspapers : ℕ) (newspaper_pay : ℚ) 
  (dogs : ℕ) (dog_pay : ℕ) : ℕ :=
  let total_earnings := (lawns * lawn_pay : ℚ) + 
                        (newspapers : ℚ) * newspaper_pay + 
                        (dogs * dog_pay : ℚ)
  let total_money := (initial_savings : ℚ) + total_earnings
  (total_money - bike_cost).floor.toNat

/-- Proves that Julie will have $155 left after purchasing the mountain bike -/
theorem julie_savings_proof : 
  julie_savings 1500 2345 20 20 600 (40/100) 24 15 = 155 := by
  sorry

#eval julie_savings 1500 2345 20 20 600 (40/100) 24 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_savings_proof_l760_76094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_function_property_l760_76011

/-- An exponential function f(x) = a^x where a > 0 and a ≠ 1 -/
noncomputable def ExpFunction (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : ℝ → ℝ := fun x ↦ a^x

theorem exp_function_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := ExpFunction a h1 h2
  f 3 = 27 → f 2 = 9 := by
  intro h
  have : a = 3 := by
    -- Proof that a = 3
    sorry
  -- Proof that f 2 = 9 using a = 3
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_function_property_l760_76011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_equals_average_l760_76081

/-- A right triangle with perpendicular distances to a line -/
structure RightTriangleWithDistances where
  -- The perpendicular distances from each vertex to the line
  distA : ℝ
  distB : ℝ
  distC : ℝ

/-- The perpendicular distance from the centroid to the line -/
noncomputable def centroidDistance (t : RightTriangleWithDistances) : ℝ :=
  (t.distA + t.distB + t.distC) / 3

theorem centroid_distance_equals_average 
  (t : RightTriangleWithDistances) 
  (h₁ : t.distA = 15) 
  (h₂ : t.distB = 9) 
  (h₃ : t.distC = 27) : 
  centroidDistance t = 17 := by
  -- Unfold the definition of centroidDistance
  unfold centroidDistance
  -- Substitute the given values
  rw [h₁, h₂, h₃]
  -- Simplify the arithmetic
  norm_num
  -- QED
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_equals_average_l760_76081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_K_M_l760_76039

/-- The maximum distance between points K and M under given conditions -/
theorem max_distance_K_M : ℝ := by
  -- Define points C, B, and A
  let C : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let A : ℝ × ℝ := (1, Real.sqrt 3)

  -- Define the locus of K
  let K_locus (x y : ℝ) : Prop :=
    (x - 2) ^ 2 + y ^ 2 = 1/4 * (x ^ 2 + y ^ 2)

  -- Define the locus of M
  let M_locus (x y : ℝ) : Prop :=
    (x - 1) ^ 2 + (y - Real.sqrt 3) ^ 2 = 1/4 * (x ^ 2 + y ^ 2)

  -- The maximum distance between K and M is 16/3
  let max_distance : ℝ := 16/3

  -- The theorem statement
  have h : ∀ (K_x K_y M_x M_y : ℝ),
    K_locus K_x K_y →
    M_locus M_x M_y →
    Real.sqrt ((K_x - M_x) ^ 2 + (K_y - M_y) ^ 2) ≤ max_distance := by
      sorry

  exact max_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_K_M_l760_76039
