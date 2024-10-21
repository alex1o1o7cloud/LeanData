import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2BAD_eq_seven_ninths_l325_32566

-- Define the triangles and their properties
def Triangle (A B C : ℝ × ℝ) := True

def RightTriangle (A B C : ℝ × ℝ) := Triangle A B C ∧ ∃ θ : ℝ, Real.cos θ = 0

def IsoscelesRightTriangle (A B C : ℝ × ℝ) := 
  RightTriangle A B C ∧ 
  ∃ l : ℝ, dist A B = l ∧ dist B C = l

def Perimeter (A B C : ℝ × ℝ) : ℝ := dist A B + dist B C + dist C A

noncomputable def AngleBAD (A B C D : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem sin_2BAD_eq_seven_ninths 
  (A B C D : ℝ × ℝ) 
  (h1 : RightTriangle A C D)
  (h2 : IsoscelesRightTriangle A B C)
  (h3 : ∃ l : ℝ, dist A B = l ∧ dist B C = l ∧ l = 2)
  (h4 : Perimeter A B C = Perimeter A C D) :
  Real.sin (2 * AngleBAD A B C D) = 7/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2BAD_eq_seven_ninths_l325_32566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_x_intercept_of_circle_l325_32502

-- Define the circle Γ
noncomputable def Γ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - (3 * Real.sqrt 7) / 2)^2 + (p.2 - (7 * Real.sqrt 3) / 2)^2 = 210 / 4}

-- Define the endpoints of the diameter
def endpoint1 : ℝ × ℝ := (0, 0)
noncomputable def endpoint2 : ℝ × ℝ := (3 * Real.sqrt 7, 7 * Real.sqrt 3)

-- Theorem statement
theorem other_x_intercept_of_circle (p : ℝ × ℝ) :
  p ∈ Γ ∧ p.2 = 0 ∧ p ≠ endpoint1 → p = (3 * Real.sqrt 7, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_x_intercept_of_circle_l325_32502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_l325_32511

theorem sin_inequality (x : ℝ) : 
  |Real.sin x + Real.sin (Real.sqrt 2 * x)| < 2 - 1 / (100 * (x^2 + 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_l325_32511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_d_range_l325_32504

-- Define the arithmetic sequence and its properties
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Define the sum of the first n terms
noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * a₁ + n * (n - 1 : ℝ) / 2 * d

-- Theorem statement
theorem arithmetic_sequence_d_range :
  ∀ d : ℝ, 
  (∀ n : ℕ, n ≠ 6 → S_n (-6) d n > S_n (-6) d 6) ∧
  (∀ ε > 0, ∃ n : ℕ, n ≠ 6 ∧ S_n (-6) d n < S_n (-6) d 6 + ε) →
  1 < d ∧ d < 6/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_d_range_l325_32504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_range_l325_32577

/-- Definition of the ellipse E -/
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- The focal distance of the ellipse -/
def focal_distance : ℝ := 4

/-- The length of the chord passing through the focus perpendicular to the x-axis -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt 2

/-- The left focus of the ellipse -/
def left_focus : ℝ × ℝ := (-2, 0)

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (2, 0)

/-- A point on the ellipse -/
def ellipse_point (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector from left focus to a point -/
def vector_from_left_focus (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - left_focus.1, p.2 - left_focus.2)

theorem ellipse_dot_product_range :
  ∀ (M N : ℝ × ℝ), ellipse_point M → ellipse_point N →
  ∃ (t : ℝ), M = (right_focus.1 + t, right_focus.2 * t) ∧ 
             N = (right_focus.1 + t, right_focus.2 * (-t)) →
  -4 ≤ dot_product (vector_from_left_focus M) (vector_from_left_focus N) ∧ 
  dot_product (vector_from_left_focus M) (vector_from_left_focus N) ≤ 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_range_l325_32577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stack_height_for_10cm_pipes_l325_32575

/-- The height of a stack of five identical cylindrical pipes -/
noncomputable def stack_height (pipe_diameter : ℝ) : ℝ :=
  pipe_diameter + pipe_diameter / 2 * Real.sqrt 3 + pipe_diameter / 2

/-- Theorem: The height of the stack of five pipes with diameter 10 cm is 15 + 5√3 cm -/
theorem stack_height_for_10cm_pipes :
  stack_height 10 = 15 + 5 * Real.sqrt 3 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stack_height_for_10cm_pipes_l325_32575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_row_20_fifth_sixth_l325_32550

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Pascal's Triangle row function -/
def pascalRow (n : ℕ) : List ℕ := List.range (n + 1) |>.map (binomial n)

theorem pascal_row_20_fifth_sixth :
  let row20 := pascalRow 20
  (row20[4]? = some 4845) ∧ (row20[5]? = some 15504) := by
  sorry

#eval pascalRow 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_row_20_fifth_sixth_l325_32550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAF_l325_32509

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define point A as the intersection of line_l and parabola above x-axis
noncomputable def point_A : ℝ × ℝ := (1 + Real.sqrt 3, 2 * Real.sqrt 3)

-- Theorem statement
theorem area_of_triangle_OAF :
  parabola (point_A.1) (point_A.2) ∧
  line_l (point_A.1) (point_A.2) ∧
  point_A.2 > 0 →
  (1/2) * (point_A.1 - 0) * (point_A.2 - 0) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAF_l325_32509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_point_difference_l325_32547

/-- An angle whose terminal side coincides with y = 3x and has negative sine -/
structure Angle (α : Real) : Prop where
  terminal_side : ∀ (x y : Real), y = 3 * x → (∃ (t : Real), x = t * Real.cos α ∧ y = t * Real.sin α)
  negative_sine : Real.sin α < 0

/-- A point on the terminal side of the angle -/
structure TerminalPoint (α : Real) (m n : Real) : Prop where
  on_line : n = 3 * m
  on_terminal : ∃ (t : Real), m = t * Real.cos α ∧ n = t * Real.sin α
  distance : m^2 + n^2 = 10

theorem angle_point_difference (α m n : Real) 
  (h_angle : Angle α) (h_point : TerminalPoint α m n) : m - n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_point_difference_l325_32547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_midpoint_equation_l325_32515

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := ((1/2) * t, (Real.sqrt 3 / 2) * t)

-- Define the circle
noncomputable def circle_ρ (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Statement for the length of the chord
theorem chord_length : 
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
  (line_l t₁).1^2 + (line_l t₁).2^2 = (circle_ρ (Real.arctan ((line_l t₁).2 / (line_l t₁).1)))^2 ∧
  (line_l t₂).1^2 + (line_l t₂).2^2 = (circle_ρ (Real.arctan ((line_l t₂).2 / (line_l t₂).1)))^2 ∧
  Real.sqrt (((line_l t₁).1 - (line_l t₂).1)^2 + ((line_l t₁).2 - (line_l t₂).2)^2) = 2 * Real.sqrt 3 :=
sorry

-- Statement for the midpoint equation
theorem midpoint_equation (θ : ℝ) :
  let ρ := circle_ρ θ / 2
  ρ = 2 * Real.sin θ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_midpoint_equation_l325_32515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l325_32587

theorem cube_root_simplification : (20^3 + 30^3 + 120^3 : ℝ)^(1/3) = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l325_32587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l325_32576

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.cos α = 5/13) (h4 : Real.cos (α + β) = -4/5) : Real.cos β = 16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l325_32576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_circle_equation_proof_l325_32567

-- Define the circle equation theorem
theorem circle_equation (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  let (a, b) := center
  (∃ k : ℝ, b = 2 * a) ∧ -- center on y = 2x
  (radius = |b|) ∧ -- tangent to x-axis
  (∃ chord_length : ℝ, chord_length = Real.sqrt 14 ∧
    chord_length^2 = 4 * radius^2 - 2 * (a - b)^2) → -- intercepted by x - y = 0 with chord length √14
  (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = radius^2 ↔
   ((x - 1)^2 + (y - 2)^2 = 4 ∨ (x + 1)^2 + (y + 2)^2 = 4))

-- Prove the existence of a center and radius satisfying the circle equation
theorem circle_equation_proof : ∃ center : ℝ × ℝ, ∃ radius : ℝ, circle_equation center radius := by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_circle_equation_proof_l325_32567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_angle_bisector_l325_32531

/-- Triangle ABC with angle bisector BL -/
structure Triangle :=
  (A B C L : ℝ × ℝ)
  (is_angle_bisector : Bool)

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_with_angle_bisector (t : Triangle) 
  (h1 : t.is_angle_bisector = true)
  (h2 : Real.sqrt ((t.A.1 - t.L.1)^2 + (t.A.2 - t.L.2)^2) = 2)
  (h3 : Real.sqrt ((t.B.1 - t.L.1)^2 + (t.B.2 - t.L.2)^2) = Real.sqrt 30)
  (h4 : Real.sqrt ((t.C.1 - t.L.1)^2 + (t.C.2 - t.L.2)^2) = 5) :
  triangle_area 
    (Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2))
    (Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2))
    (Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)) = 
  (7 * Real.sqrt 39) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_angle_bisector_l325_32531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_value_range_l325_32524

-- State the theorem using the built-in round function
theorem exact_value_range (a : ℝ) (h : Int.floor (a + 0.5) = 170) :
  169.5 ≤ a ∧ a < 170.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_value_range_l325_32524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l325_32598

/-- Given vectors a, b, c in ℝ², prove the coordinates of c and the value of k. -/
theorem vector_problem (a b c : ℝ × ℝ) :
  a = (-2, 1) →
  b = (3, 2) →
  ∃ (l : ℝ), c = l • a →
  ‖c‖ = 25 →
  ((c = (-10 * Real.sqrt 5, 5 * Real.sqrt 5)) ∨ 
   (c = (10 * Real.sqrt 5, -5 * Real.sqrt 5))) ∧
  (∃ (k : ℝ), k = -22/3 ∧ 
   (k • a - b) • (a + 2 • b) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l325_32598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_bound_l325_32536

/-- Linear relationship between x and y -/
structure LinearRelation where
  slope : ℝ
  intercept : ℝ

/-- Given data points -/
def data_points : List (ℝ × ℝ) := [(16, 11), (14, 9), (12, 8), (8, 5)]

/-- Maximum forecast value of y -/
def max_y : ℝ := 10

/-- Theorem: The maximum value of x cannot exceed 15 -/
theorem max_x_bound (relation : LinearRelation) :
  (∀ p : ℝ × ℝ, p ∈ data_points → p.2 = relation.slope * p.1 + relation.intercept) →
  (∀ x : ℝ, relation.slope * x + relation.intercept ≤ max_y) →
  ∀ x : ℝ, relation.slope * x + relation.intercept = max_y → x ≤ 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_bound_l325_32536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_hiring_theorem_l325_32591

/-- Calculates the new total number of employees after hiring -/
def new_employee_count (original_count : ℕ) (percentage_increase : ℚ) : ℕ :=
  original_count + Int.toNat ((original_count : ℚ) * percentage_increase).floor

/-- Theorem: If a factory with 1200 employees hires 40% more workers, 
    the total number of employees after hiring will be 1680 -/
theorem factory_hiring_theorem :
  new_employee_count 1200 (40/100) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_hiring_theorem_l325_32591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_digits_312_base5_l325_32557

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  digits.filter (fun d => d % 2 = 0) |>.length

/-- The number of even digits in the base-5 representation of 312 is 4 -/
theorem even_digits_312_base5 :
  countEvenDigits (toBase5 312) = 4 := by
  sorry

#eval toBase5 312
#eval countEvenDigits (toBase5 312)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_digits_312_base5_l325_32557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mississippi_arrangements_l325_32530

theorem mississippi_arrangements : ∃ n : ℕ, n = 34650 ∧ n = (Nat.factorial 11 / (Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 1)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mississippi_arrangements_l325_32530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circles_of_friends_l325_32593

/-- A conference with delegates and handshakes -/
structure Conference where
  delegates : Nat
  handshakes : Nat
  max_common_handshakes : Nat

/-- A circle of friends in the conference -/
def circle_of_friends (c : Conference) : Nat := 0  -- Placeholder definition

/-- The main theorem about the minimum number of circles of friends -/
theorem min_circles_of_friends (c : Conference) 
  (h1 : c.delegates = 24)
  (h2 : c.handshakes = 216)
  (h3 : c.max_common_handshakes = 10)
  : circle_of_friends c ≥ 864 := by
  sorry

#check min_circles_of_friends

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circles_of_friends_l325_32593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_color_integer_diff_l325_32525

/-- A coloring of integers using four colors -/
def Coloring := ℤ → Fin 4

/-- Proposition: For any coloring of integers using four colors, and for any two odd integers x and y
    where |x| ≠ |y|, there exist two integers a and b of the same color such that
    their difference is one of x, y, x+y, or x-y -/
theorem four_color_integer_diff (c : Coloring) (x y : ℤ) 
    (hx : x % 2 = 1) (hy : y % 2 = 1) (hxy : |x| ≠ |y|) :
    ∃ a b : ℤ, c a = c b ∧ (b - a = x ∨ b - a = y ∨ b - a = x + y ∨ b - a = x - y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_color_integer_diff_l325_32525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_third_angle_l325_32558

theorem cosine_third_angle (A B C : ℝ) : 
  0 < A → A < π →
  0 < B → B < π →
  A + B + C = π →
  Real.cos A = 3/5 →
  Real.cos B = 5/13 →
  Real.cos C = 33/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_third_angle_l325_32558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_l325_32513

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define point P
def P : ℝ × ℝ := (2, 2)

-- Define a function to calculate the slopes of tangent lines
noncomputable def tangentSlopes (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Set ℝ := 
  {k : ℝ | ∃ q ∈ c, (q.2 - p.2) = k * (q.1 - p.1) ∧ 
    ∀ r ∈ c, (r.2 - p.2) ≠ k * (r.1 - p.1) ∨ r = q}

-- Theorem statement
theorem tangent_slopes_product :
  let slopes := tangentSlopes P C
  ∃ k₁ k₂, k₁ ∈ slopes ∧ k₂ ∈ slopes ∧ k₁ ≠ k₂ ∧ k₁ * k₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_l325_32513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l325_32559

/-- 
A trinomial is a perfect square if and only if it can be expressed as (ax + b)^2
where a and b are real numbers and a ≠ 0.
-/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), p ≠ 0 ∧ ∀ (x : ℝ), a * x^2 + b * x + c = (p * x + q)^2

/-- 
If x^2 + mx + 16 is a perfect square trinomial, then m = 8 or m = -8.
-/
theorem perfect_square_condition (m : ℝ) :
  is_perfect_square_trinomial 1 m 16 → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l325_32559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_block_difference_count_l325_32545

/-- Represents the material of a block -/
inductive Material
  | Plastic
  | Wood
  | Metal
  deriving Repr, DecidableEq

/-- Represents the size of a block -/
inductive Size
  | Small
  | Medium
  | Large
  deriving Repr, DecidableEq

/-- Represents the color of a block -/
inductive Color
  | Blue
  | Green
  | Red
  | Yellow
  deriving Repr, DecidableEq

/-- Represents the shape of a block -/
inductive Shape
  | Circle
  | Hexagon
  | Square
  | Triangle
  | Rectangle
  deriving Repr, DecidableEq

/-- Represents a block with its properties -/
structure Block where
  material : Material
  size : Size
  color : Color
  shape : Shape
  deriving Repr, DecidableEq

/-- The reference block: metal medium blue rectangle -/
def referenceBlock : Block :=
  { material := Material.Metal
    size := Size.Medium
    color := Color.Blue
    shape := Shape.Rectangle }

/-- Counts the number of different properties between two blocks -/
def countDifferences (b1 b2 : Block) : Nat :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0)

/-- The set of all possible blocks -/
def allBlocks : Finset Block := sorry

/-- Theorem stating that there are 76 blocks differing in exactly 3 ways from the reference block -/
theorem block_difference_count :
  (allBlocks.filter (fun b => countDifferences b referenceBlock = 3)).card = 76 := by
  sorry

#eval referenceBlock

end NUMINAMATH_CALUDE_ERRORFEEDBACK_block_difference_count_l325_32545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_big_bottles_count_l325_32539

theorem big_bottles_count : ℕ := by
  let small_initial : ℕ := 6000
  let small_remain_percent : ℚ := 89 / 100
  let big_remain_percent : ℚ := 88 / 100
  let total_remain : ℕ := 18540

  have h : ∃ (big_initial : ℕ), 
    (small_initial : ℚ) * small_remain_percent + 
    (big_initial : ℚ) * big_remain_percent = total_remain := by sorry

  exact Classical.choose h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_big_bottles_count_l325_32539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_isothermal_compression_l325_32596

-- Constants
noncomputable def atmosphericPressure : ℝ := 103300  -- 103.3 kPa in Pa
def initialHeight : ℝ := 2.0  -- H in meters
def pistonDisplacement : ℝ := 1.0  -- h in meters
def cylinderRadius : ℝ := 0.4  -- R in meters

-- Function to calculate work done
noncomputable def workDone (p₀ H h R : ℝ) : ℝ :=
  let S := Real.pi * R^2
  p₀ * H * S * Real.log (H / (H - h))

-- Theorem statement
theorem work_done_isothermal_compression :
  ∃ ε > 0, |workDone atmosphericPressure initialHeight pistonDisplacement cylinderRadius - 72000| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_isothermal_compression_l325_32596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_in_parallel_faces_relationship_l325_32581

-- Define a rectangular solid
structure RectangularSolid where
  -- We don't need to specify the properties of a rectangular solid for this problem

-- Define a line in a plane
structure LineInPlane where
  -- We don't need to specify the properties of a line in a plane for this problem

-- Define the positional relationship between two lines
inductive PositionalRelationship
  | Parallel
  | Intersecting
  | Skew

-- Define functions to check if a line is in the top or bottom face
def isInTopFace (line : LineInPlane) (solid : RectangularSolid) : Prop := sorry
def isInBottomFace (line : LineInPlane) (solid : RectangularSolid) : Prop := sorry

-- Define a function to determine the positional relationship between two lines
def determineRelationship (a b : LineInPlane) : PositionalRelationship := sorry

-- Define the theorem
theorem lines_in_parallel_faces_relationship 
  (solid : RectangularSolid) 
  (a b : LineInPlane) 
  (h1 : isInTopFace a solid) 
  (h2 : isInBottomFace b solid) : 
  determineRelationship a b = PositionalRelationship.Parallel ∨ 
  determineRelationship a b = PositionalRelationship.Skew :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_in_parallel_faces_relationship_l325_32581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_105_l325_32564

-- Define the problem setup
def S : ℕ → Prop := sorry
def has_8_factors : ℕ → Prop := sorry
def speed_ratio : ℕ := 4
def is_integer : ℚ → Prop := sorry

-- Define the conditions
axiom S_positive : ∃ s : ℕ, S s ∧ s > 0
axiom S_has_8_factors : ∀ s, S s → has_8_factors s
axiom AC_is_integer : ∀ s, S s → is_integer ((4 * s : ℚ) / 5)
axiom AD_is_integer : ∀ s, S s → is_integer ((2 * s : ℚ) / 3)
axiom AE_is_integer : ∀ s, S s → is_integer (s / 21)

-- State the theorem
theorem distance_is_105 : ∃ s : ℕ, S s ∧ s = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_105_l325_32564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l325_32533

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((20 / (x + 10)) + a) / Real.log 10

-- State the theorem
theorem odd_function_properties :
  ∃ (a : ℝ), 
    (∀ x, f a x = -f a (-x)) ∧  -- f is an odd function
    (a = -1) ∧  -- Part I: value of a
    (∀ x, f a x > 0 ↔ -10 < x ∧ x < 0)  -- Part II: solution set of f(x) > 0
    := by
  -- Proof sketch
  use -1
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l325_32533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_journey_exists_l325_32562

/-- Represents the state of a crater (erupting or silent) -/
inductive CraterState
| Erupting
| Silent
deriving DecidableEq

/-- Represents the safety state of a path -/
inductive PathSafety
| Safe
| Dangerous
deriving DecidableEq

/-- Represents a crater with its eruption cycle -/
structure Crater where
  eruptionDuration : ℕ
  silentDuration : ℕ

/-- Represents the volcano with its craters and paths -/
structure Volcano where
  crater1 : Crater
  crater2 : Crater
  roadDuration : ℕ
  trailDuration : ℕ

/-- Determines the state of a crater at a given time -/
def craterState (c : Crater) (t : ℕ) : CraterState :=
  if t % (c.eruptionDuration + c.silentDuration) < c.eruptionDuration
  then CraterState.Erupting
  else CraterState.Silent

/-- Determines the safety of the road at a given time -/
def roadSafety (v : Volcano) (t : ℕ) : PathSafety :=
  if craterState v.crater1 t = CraterState.Erupting
  then PathSafety.Dangerous
  else PathSafety.Safe

/-- Determines the safety of the trail at a given time -/
def trailSafety (v : Volcano) (t : ℕ) : PathSafety :=
  if craterState v.crater1 t = CraterState.Erupting ∨ craterState v.crater2 t = CraterState.Erupting
  then PathSafety.Dangerous
  else PathSafety.Safe

/-- Checks if the journey is safe starting at a given time -/
def isSafeJourney (v : Volcano) (startTime : ℕ) : Prop :=
  (∀ t, startTime ≤ t ∧ t < startTime + v.roadDuration →
    roadSafety v t = PathSafety.Safe) ∧
  (∀ t, startTime + v.roadDuration ≤ t ∧ t < startTime + v.roadDuration + v.trailDuration →
    trailSafety v t = PathSafety.Safe) ∧
  (∀ t, startTime + v.roadDuration + v.trailDuration ≤ t ∧ t < startTime + v.roadDuration + 2 * v.trailDuration →
    trailSafety v t = PathSafety.Safe) ∧
  (∀ t, startTime + v.roadDuration + 2 * v.trailDuration ≤ t ∧ t < startTime + 2 * v.roadDuration + 2 * v.trailDuration →
    roadSafety v t = PathSafety.Safe)

/-- The main theorem: There exists a safe starting time for the journey -/
theorem safe_journey_exists (v : Volcano) : ∃ startTime : ℕ, isSafeJourney v startTime := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_journey_exists_l325_32562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_abc_l325_32523

theorem ascending_order_abc (a b c : ℝ) 
  (ha : a = Real.log 0.9 / Real.log 0.6)
  (hb : b = Real.log 0.9)
  (hc : c = Real.rpow 2 0.9) :
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_abc_l325_32523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_condition_l325_32507

theorem sine_inequality_condition (α β : Real) :
  (α ≠ β → Real.sin α ≠ Real.sin β) ∧ ¬(Real.sin α ≠ Real.sin β → α ≠ β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_condition_l325_32507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l325_32597

theorem solution_to_equation : ∃ x : ℝ, (4 : ℝ)^x - 2^(x+1) = 0 ∧ x = 1 := by
  use 1
  constructor
  · simp
    norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l325_32597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birch_planting_l325_32555

theorem birch_planting (total_students : ℕ) (total_plants : ℕ) 
  (h1 : total_students = 24)
  (h2 : total_plants = 24) : ℕ :=
by
  let boys : ℕ := sorry
  let girls : ℕ := total_students - boys
  let roses : ℕ := 3 * girls
  let birches : ℕ := boys / 3
  have h3 : roses + birches = total_plants := by sorry
  have h4 : ∀ b : ℕ, b ≤ total_students → (total_students - b) * 3 + b / 3 = total_plants → 
               b / 3 = 6 := by sorry
  exact 6

#check birch_planting

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birch_planting_l325_32555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l325_32537

-- Define the primitive fifth root of unity
noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

-- State the theorem
theorem no_integer_solution :
  ¬ ∃ (a b c d k : ℤ), k > 1 ∧ (a + b * ω + c * ω^2 + d * ω^3)^k = (1 : ℂ) + ω := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l325_32537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_cost_price_l325_32508

/-- The cost price of a watch in USD -/
noncomputable def cost_price_usd : ℝ := 10000 / 7

/-- The selling price of a watch sold at a loss in USD -/
noncomputable def selling_price_loss : ℝ := 0.9 * cost_price_usd

/-- The selling price of a watch sold at a gain in USD -/
noncomputable def selling_price_gain : ℝ := 1.04 * cost_price_usd

/-- Theorem stating the relationship between selling prices and cost price -/
theorem watch_cost_price : 
  selling_price_loss + 200 = selling_price_gain := by
  -- Unfold definitions
  unfold selling_price_loss selling_price_gain cost_price_usd
  -- Simplify the equation
  simp [mul_add, mul_assoc]
  -- Check equality
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_cost_price_l325_32508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_depth_notation_l325_32546

/-- Represents the altitude in meters, where positive values are above sea level
    and negative values are below sea level -/
def Altitude : Type := Int

/-- Given that 9050 meters above sea level is denoted as +9050,
    prove that 10907 meters below sea level should be denoted as -10907 -/
theorem depth_notation (h : Int) :
  h = 9050 → ((-10907 : Int) = -10907) :=
by
  intro h_eq
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_depth_notation_l325_32546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_repeat_unless_all_one_l325_32561

/-- Given four positive real numbers, generate the next set by multiplying each number
    by the next one, with the last multiplied by the first. -/
def next_set (a b c d : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (a * b, b * c, c * d, d * a)

/-- Generate the nth set in the sequence -/
def nth_set : ℝ → ℝ → ℝ → ℝ → ℕ → ℝ × ℝ × ℝ × ℝ
  | a, b, c, d, 0 => (a, b, c, d)
  | a, b, c, d, n + 1 => 
    let prev := nth_set a b c d n
    next_set prev.1 prev.2.1 prev.2.2.1 prev.2.2.2

/-- The main theorem: the original set never reappears unless all numbers are 1 -/
theorem no_repeat_unless_all_one (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∃ n : ℕ, n > 0 ∧ nth_set a b c d n = (a, b, c, d)) → (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_repeat_unless_all_one_l325_32561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sin_2x_l325_32551

-- Define the derivative of sine
axiom sine_derivative (t : ℝ) : deriv Real.sin t = Real.cos t

-- State the theorem
theorem derivative_sin_2x (x : ℝ) : 
  deriv (λ x => Real.sin (2 * x)) x = 2 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sin_2x_l325_32551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l325_32588

def customSequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a (n + 1) = a (n + 2) - a n) ∧
  a 1 = 2 ∧
  a 2 = 5

theorem fifth_term_value (a : ℕ → ℤ) (h : customSequence a) : a 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l325_32588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_fraction_l325_32565

/-- Represents a square pyramid --/
structure SquarePyramid where
  baseEdge : ℝ
  altitude : ℝ

/-- Calculates the volume of a square pyramid --/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.baseEdge^2 * p.altitude

/-- Theorem: The volume of the frustum is 23/24 of the original pyramid's volume --/
theorem frustum_volume_fraction (original : SquarePyramid) 
    (h1 : original.baseEdge = 24)
    (h2 : original.altitude = 18) : 
  let smaller : SquarePyramid := { 
    baseEdge := original.baseEdge / 3,
    altitude := original.altitude / 3
  }
  (pyramidVolume original - pyramidVolume smaller) / pyramidVolume original = 23 / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_fraction_l325_32565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l325_32553

theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α - Real.cos α = Real.sqrt 2) : 
  Real.tan α = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l325_32553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cuts_for_100_20gons_l325_32579

/-- Represents the number of cuts made on the paper -/
def NumCuts (n : ℕ) : Prop := true

/-- Represents the number of pieces after n cuts -/
def NumPieces (n : ℕ) : ℕ := n + 1

/-- Represents the maximum number of vertices after n cuts -/
def MaxVertices (n : ℕ) : ℕ := 4 * n + 4

/-- Represents the minimum number of vertices for the desired configuration -/
def MinVertices (n : ℕ) : ℕ := 2000 + 3 * n + 3 - 300

/-- The theorem stating the minimum number of cuts needed -/
theorem min_cuts_for_100_20gons :
  ∃ (n : ℕ), NumCuts n ∧
  (∀ m : ℕ, m < n → ¬(NumCuts m)) ∧
  NumPieces n ≥ 100 ∧
  MaxVertices n ≥ MinVertices n ∧
  n = 1699 := by
  sorry

#check min_cuts_for_100_20gons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cuts_for_100_20gons_l325_32579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_2_l325_32592

def a : ℕ → ℚ
| 0 => 2
| n + 1 => 1 / (1 - a n)

theorem a_2017_equals_2 : a 2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_2_l325_32592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_binary_digits_l325_32583

theorem consecutive_binary_digits (x : ℤ) (h : x > 2) :
  ∃ (n : ℕ), (((x^2 - 1).natAbs.digits 2).drop n).take 3 = [0, 0, 0] ∨
             (((x^2 - 1).natAbs.digits 2).drop n).take 3 = [1, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_binary_digits_l325_32583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l325_32544

theorem min_value_of_exponential_sum (a b : ℝ) (h : a > 0) (h' : b > 0) 
  (eq : Real.log a / Real.log 4 + (Real.log b / Real.log 2) / 2 = 1) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 
    Real.log x / Real.log 4 + (Real.log y / Real.log 2) / 2 = 1 → 
    (4 : ℝ)^a + (4 : ℝ)^b ≤ (4 : ℝ)^x + (4 : ℝ)^y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    Real.log x / Real.log 4 + (Real.log y / Real.log 2) / 2 = 1 ∧ 
    (4 : ℝ)^x + (4 : ℝ)^y = 32) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l325_32544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_solid_edge_sum_l325_32503

/-- Represents a rectangular solid with dimensions in geometric progression -/
structure GeometricSolid where
  a : ℝ
  r : ℝ
  volume : ℝ
  surface_area : ℝ
  volume_eq : volume = a^3 * r^3
  surface_area_eq : surface_area = 2 * (a^2 * r^4 + a^2 * r^2 + a^2 * r)

/-- The sum of all edge lengths of a GeometricSolid -/
def edge_sum (s : GeometricSolid) : ℝ :=
  4 * (s.a * s.r^2 + s.a * s.r + s.a)

/-- Theorem stating that a GeometricSolid with volume 288 and surface area 288
    has an edge sum of approximately 92 -/
theorem geometric_solid_edge_sum :
  ∃ (s : GeometricSolid),
    s.volume = 288 ∧
    s.surface_area = 288 ∧
    ∃ (ε : ℝ), ε > 0 ∧ |edge_sum s - 92| < ε := by
  sorry

#check geometric_solid_edge_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_solid_edge_sum_l325_32503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l325_32556

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2*x + (3*x)/(x^2 + 3) + (2*x*(x + 5))/(x^2 + 5) + (3*(x + 3))/(x*(x^2 + 5))

/-- Theorem stating that the minimum value of f(x) for x > 0 is 7 -/
theorem f_min_value (x : ℝ) (hx : x > 0) : f x ≥ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l325_32556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l325_32505

/-- The function to be minimized -/
noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

/-- The domain for x -/
def X : Set ℝ := { x | 3/7 ≤ x ∧ x ≤ 2/3 }

/-- The domain for y -/
def Y : Set ℝ := { y | 1/4 ≤ y ∧ y ≤ 3/5 }

/-- The theorem stating the minimum value of the function -/
theorem min_value_of_f :
  ∃ (x y : ℝ), x ∈ X ∧ y ∈ Y ∧
  (∀ (x' y' : ℝ), x' ∈ X → y' ∈ Y → f x y ≤ f x' y') ∧
  f x y = 288/876 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l325_32505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_toss_probability_l325_32569

-- Define a homogeneous coin
def HomogeneousCoin : Type := Unit

-- Define the result of a coin toss
inductive CoinResult
| Heads
| Tails

-- Define the probability of getting heads on a single toss
noncomputable def prob_heads (coin : HomogeneousCoin) : ℝ := 1 / 2

-- Define the probability of getting heads on the second toss given heads on the first toss
noncomputable def prob_second_heads_given_first_heads (coin : HomogeneousCoin) : ℝ :=
  prob_heads coin

-- Theorem statement
theorem second_toss_probability (coin : HomogeneousCoin) :
  prob_second_heads_given_first_heads coin = 1 / 2 := by
  -- Unfold the definition of prob_second_heads_given_first_heads
  unfold prob_second_heads_given_first_heads
  -- Unfold the definition of prob_heads
  unfold prob_heads
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_toss_probability_l325_32569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canteen_equidistant_distance_l325_32580

noncomputable section

/-- The distance from the girls' camp to the road -/
def girls_camp_distance : ℝ := 400

/-- The angle between the line from the girls' camp to the road and the perpendicular to the road -/
noncomputable def girls_camp_angle : ℝ := 30 * Real.pi / 180

/-- The distance between the girls' camp and the boys' camp -/
def camps_distance : ℝ := 600

/-- The angle between the line connecting the two camps and the road -/
noncomputable def camps_road_angle : ℝ := Real.pi / 2

/-- The closest distance from the road where a canteen can be built that is equidistant from both camps -/
noncomputable def canteen_distance : ℝ := 125 * Real.sqrt 2

theorem canteen_equidistant_distance :
  let perpendicular_distance := girls_camp_distance * Real.cos girls_camp_angle
  let horizontal_distance := girls_camp_distance * Real.sin girls_camp_angle
  let boys_camp_height := Real.sqrt (camps_distance ^ 2 - horizontal_distance ^ 2)
  canteen_distance = (boys_camp_height * horizontal_distance) / (perpendicular_distance + boys_camp_height) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canteen_equidistant_distance_l325_32580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_in_interval_f_upper_bound_l325_32535

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 1) * Real.exp x

-- Define the set of valid integer values for a
def valid_a : Set ℤ := {-6, -5, -4}

-- Statement 1: f has a maximum value in (a, a+5) if and only if a is in valid_a
theorem max_in_interval (a : ℤ) : 
  (∃ (c : ℝ), c ∈ Set.Ioo (a : ℝ) ((a : ℝ) + 5) ∧ 
    ∀ (x : ℝ), x ∈ Set.Ioo (a : ℝ) ((a : ℝ) + 5) → f x ≤ f c) ↔ 
  a ∈ valid_a :=
sorry

-- Statement 2: For all x > 0, f(x) < -3ln x + x^3 + (2x^2 - 4x)e^x + 7
theorem f_upper_bound (x : ℝ) (hx : x > 0) : 
  f x < -3 * Real.log x + x^3 + (2*x^2 - 4*x) * Real.exp x + 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_in_interval_f_upper_bound_l325_32535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l325_32570

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c d e f : ℝ) : ℝ :=
  |c - f| / Real.sqrt (a^2 + b^2)

/-- Two lines are parallel if their slope coefficients are proportional -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

theorem distance_between_specific_lines (a : ℝ) :
  are_parallel 3 (-4) a (-8) →
  distance_between_parallel_lines 3 (-4) (-12) a (-8) 11 = 7/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l325_32570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_when_a_zero_h_is_minimum_of_g_no_m_n_exist_l325_32516

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3^x

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := (f x)^2 - 2*a*(f x) + 3

-- Define the minimum value function h
noncomputable def h (a : ℝ) : ℝ :=
  if a < 1/3 then 28/9 - 2*a/3
  else if a ≤ 3 then 3 - a^2
  else 12 - 6*a

-- Theorem 1: Range of g when a = 0
theorem range_of_g_when_a_zero :
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → g 0 x ∈ Set.Icc (28/9 : ℝ) 12 :=
by sorry

-- Theorem 2: h(a) is the minimum value of g(x)
theorem h_is_minimum_of_g :
  ∀ a x, x ∈ Set.Icc (-1 : ℝ) 1 → g a x ≥ h a :=
by sorry

-- Theorem 3: Non-existence of m and n
theorem no_m_n_exist :
  ¬∃ m n : ℝ, m > n ∧ n > 3 ∧
  (∀ a, a ∈ Set.Icc n m → h a ∈ Set.Icc (n^2) (m^2)) ∧
  (∃ a₁ a₂, a₁ ∈ Set.Icc n m ∧ a₂ ∈ Set.Icc n m ∧ h a₁ = n^2 ∧ h a₂ = m^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_when_a_zero_h_is_minimum_of_g_no_m_n_exist_l325_32516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fan_shaped_field_area_l325_32501

/-- Represents a sector of a circle -/
structure Sector where
  arcLength : ℝ
  diameter : ℝ

/-- Calculates the area of a sector -/
noncomputable def sectorArea (s : Sector) : ℝ :=
  (s.arcLength * s.diameter) / 4

theorem fan_shaped_field_area :
  let field : Sector := { arcLength := 30, diameter := 16 }
  sectorArea field = 120 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fan_shaped_field_area_l325_32501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l325_32594

/-- The standard form of a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → Prop

/-- Given hyperbola -/
def given_hyperbola : Hyperbola where
  a := 4
  b := 3
  eq := λ x y ↦ x^2 / 16 - y^2 / 9 = 1

/-- Point P -/
noncomputable def P : ℝ × ℝ := (-Real.sqrt 5 / 2, -Real.sqrt 6)

/-- Theorem: The standard equation of the hyperbola with the same foci as the given hyperbola
    and passing through point P is x^2 - y^2/24 = 1 -/
theorem hyperbola_equation : 
  ∃ (h : Hyperbola), 
    (∀ x y, h.eq x y ↔ x^2 - y^2/24 = 1) ∧ 
    (h.a^2 + h.b^2 = given_hyperbola.a^2 + given_hyperbola.b^2) ∧
    h.eq P.1 P.2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l325_32594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_asin_l325_32532

/-- The point of tangency (a, b) on the hyperbola x² - (y-1)² = 1 where the tangent line passes through (0,0) with positive slope --/
def tangent_point (a b : ℝ) : Prop :=
  a^2 - (b - 1)^2 = 1 ∧
  (∃ m : ℝ, m > 0 ∧ b = m * a) ∧
  (a / (b - 1) = b / a)

/-- Theorem stating that for the point of tangency (a, b), arcsin(a/b) = π/4 --/
theorem tangent_point_asin (a b : ℝ) (h : tangent_point a b) : Real.arcsin (a / b) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_asin_l325_32532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_l325_32563

noncomputable def f (x : ℝ) : ℝ := x⁻¹ + x⁻¹ / (1 + x⁻¹)

theorem f_composition_negative_two :
  f (f (-2)) = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_l325_32563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l325_32568

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := Real.exp x

-- Define the tangent line equation
def is_tangent_line (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ 
    y = f 0 + (f' 0) * (x - 0)

-- Theorem statement
theorem tangent_line_at_zero : 
  is_tangent_line 1 (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l325_32568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_emptying_time_l325_32538

/-- Represents the time it takes for a cistern to empty through a leak, given the normal fill time and the fill time with a leak. -/
noncomputable def time_to_empty (normal_fill_time leak_fill_time : ℝ) : ℝ :=
  (leak_fill_time * normal_fill_time) / (leak_fill_time - normal_fill_time)

/-- Theorem stating that for a cistern that normally fills in 10 hours but takes 12 hours with a leak, it will take 60 hours to empty through the leak. -/
theorem cistern_emptying_time :
  time_to_empty 10 12 = 60 := by
  unfold time_to_empty
  -- The proof steps would go here, but for now we'll use sorry
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_emptying_time_l325_32538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_sum_of_trig_functions_l325_32500

theorem perpendicular_vectors_sum_of_trig_functions 
  (θ : Real) 
  (h1 : 0 < θ) 
  (h2 : θ < π / 2) 
  (h3 : (Real.sin θ) * 1 + (-2) * (Real.cos θ) = 0) : 
  Real.sin θ + Real.cos θ = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_sum_of_trig_functions_l325_32500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_2_neither_sufficient_nor_necessary_l325_32548

/-- Two lines are parallel if their direction vectors are scalar multiples of each other -/
def are_parallel (a b c d : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a = k * c ∧ b = k * d

/-- Definition of line l₁ -/
def l₁ (a : ℝ) (x y : ℝ) : Prop :=
  a * x + y + a = 0

/-- Definition of line l₂ -/
def l₂ (a : ℝ) (x y : ℝ) : Prop :=
  (a - 6) * x + (a - 4) * y - 4 = 0

/-- Theorem stating that a = 2 is neither sufficient nor necessary for l₁ ∥ l₂ -/
theorem a_eq_2_neither_sufficient_nor_necessary :
  (∃ a : ℝ, a ≠ 2 ∧ are_parallel a 1 (a - 6) (a - 4)) ∧
  (∃ a : ℝ, a = 2 ∧ ¬are_parallel a 1 (a - 6) (a - 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_2_neither_sufficient_nor_necessary_l325_32548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_zero_l325_32578

/-- The area of the shaded region between a larger circle and four identical smaller circles -/
noncomputable def shaded_area_between_circles (r : ℝ) (h1 : r = 4) : ℝ :=
  let R := 2 * r  -- Radius of the larger circle
  let A_large := Real.pi * R^2  -- Area of the larger circle
  let A_small := 4 * (Real.pi * r^2)  -- Total area of four smaller circles
  A_large - A_small

/-- The shaded area is zero -/
theorem shaded_area_is_zero :
  shaded_area_between_circles 4 rfl = 0 := by
  unfold shaded_area_between_circles
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_zero_l325_32578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2y2_in_expansion_l325_32543

theorem coefficient_x2y2_in_expansion : ℕ := by
  -- Define the binomial coefficient function
  let binomial : ℕ → ℕ → ℕ := fun n k => Nat.choose n k

  -- Define the expansion of (1+x)^3
  let expansion_x : ℕ → ℕ := fun n => binomial 3 n

  -- Define the expansion of (1+y)^4
  let expansion_y : ℕ → ℕ := fun n => binomial 4 n

  -- The coefficient of x^2y^2 is the product of the coefficients of x^2 and y^2
  let coefficient : ℕ := expansion_x 2 * expansion_y 2

  -- The theorem states that this coefficient is equal to 18
  have h : coefficient = 18 := by sorry

  exact 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2y2_in_expansion_l325_32543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l325_32529

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop :=
  parabola p.1 p.2

-- Define the vector sum condition
def vector_sum_zero (a b c : ℝ × ℝ) : Prop :=
  (a.1 - focus.1, a.2 - focus.2) + (b.1 - focus.1, b.2 - focus.2) + (c.1 - focus.1, c.2 - focus.2) = (0, 0)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- The main theorem
theorem parabola_focus_property (a b c : ℝ × ℝ) :
  point_on_parabola a → point_on_parabola b → point_on_parabola c →
  vector_sum_zero a b c →
  distance focus a + distance focus b + distance focus c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l325_32529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_bound_l325_32520

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Check if a point is inside or on a rectangle -/
def Point.inRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem smallest_distance_bound (points : Finset Point) (r : Rectangle) :
  r.width = 2 ∧ r.height = 1 →
  points.card = 6 →
  (∀ p ∈ points, p.inRectangle r) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_bound_l325_32520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_area_l325_32510

/-- Brahmagupta's formula for the area of a cyclic quadrilateral -/
noncomputable def brahmaguptaFormula (a b c d : ℝ) : ℝ :=
  let s := (a + b + c + d) / 2
  Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))

/-- The area of a cyclic quadrilateral with sides 4, 5, 7, and 10 is 36 -/
theorem cyclic_quadrilateral_area : brahmaguptaFormula 4 5 7 10 = 36 := by
  -- Unfold the definition of brahmaguptaFormula
  unfold brahmaguptaFormula
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_area_l325_32510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_non_dividing_primes_l325_32541

/-- A set S of positive integers is closed under the operation (a, b) ↦ ab + 1 -/
def ClosedUnderABPlusOne (S : Set ℕ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → (a * b + 1) ∈ S

/-- The set of primes that do not divide any element of a given set -/
def NonDividingPrimes (S : Set ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∀ s, s ∈ S → ¬(p ∣ s)}

theorem finite_non_dividing_primes (S : Set ℕ) (hS : S.Nonempty) (hClosed : ClosedUnderABPlusOne S) :
  (NonDividingPrimes S).Finite := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_non_dividing_primes_l325_32541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l325_32540

def my_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2007 ∧ a 2 = 2008 ∧ ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

theorem sequence_1000th_term (a : ℕ → ℤ) (h : my_sequence a) : a 1000 = 2340 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l325_32540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_iff_rational_l325_32519

/-- The function defining the sequence -/
def f (x : ℝ) : ℝ := 1 - |1 - 2*x|

/-- The sequence defined recursively -/
def x (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f (x x₀ n)

/-- A sequence is periodic if there exist distinct a and b such that x_a = x_b -/
def isPeriodic (s : ℕ → ℝ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ s a = s b

theorem sequence_periodic_iff_rational (x₀ : ℝ) (h : x₀ ∈ Set.Icc 0 1) :
  isPeriodic (x x₀) ↔ ∃ (p q : ℤ), q ≠ 0 ∧ x₀ = (p : ℝ) / q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_iff_rational_l325_32519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_composition_l325_32590

-- Define a rotation in the plane
structure Rotation where
  center : ℝ × ℝ
  angle : ℝ

-- Define a translation in the plane
structure Translation where
  vector : ℝ × ℝ

-- Define the composition of two rotations
def compose_rotations (r1 r2 : Rotation) : Rotation ⊕ Translation :=
  sorry

-- Main theorem
theorem rotation_composition (A B : ℝ × ℝ) (α β : ℝ) (h : A ≠ B) :
  let r1 := Rotation.mk A α
  let r2 := Rotation.mk B β
  (¬(∃ k : ℤ, α + β = k * 2 * Real.pi) →
    ∃ (O : ℝ × ℝ), compose_rotations r1 r2 = Sum.inl (Rotation.mk O (α + β))) ∧
  ((∃ k : ℤ, α + β = k * 2 * Real.pi) →
    ∃ (v : ℝ × ℝ), compose_rotations r1 r2 = Sum.inr (Translation.mk v)) :=
by
  sorry

#check rotation_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_composition_l325_32590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_cardinality_S_l325_32572

/-- Sum of digits of a positive integer -/
def digitSum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- Set of positive integers with digit sum 15 and between 100 and 99999 -/
def S : Finset ℕ := Finset.filter (fun n => digitSum n = 15 ∧ 100 ≤ n ∧ n < 10^5) (Finset.range 100000)

theorem digit_sum_of_cardinality_S : 
  Finset.card S = 4783 ∧ digitSum 4783 = 22 := by
  sorry

#eval Finset.card S
#eval digitSum (Finset.card S)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_cardinality_S_l325_32572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l325_32573

/-- The curve C in the Cartesian plane -/
def curve_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line l in the Cartesian plane -/
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x + y - 4| / Real.sqrt 2

/-- The maximum distance from any point on curve C to line l is 3 -/
theorem max_distance_curve_to_line : 
  ∃ (x y : ℝ), curve_C x y ∧ 
  (∀ (x' y' : ℝ), curve_C x' y' → distance_to_line x' y' ≤ distance_to_line x y) ∧
  distance_to_line x y = 3 := by
  sorry

#check max_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l325_32573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parallel_line_l325_32571

/-- A point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point is on a line --/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

instance : Membership Point Line where
  mem := Point.onLine

/-- There exists a unique parallel line through a point not on the given line --/
theorem unique_parallel_line (L : Line) (P : Point) (h : P ∉ L) :
  ∃! M : Line, P ∈ M ∧ parallel M L := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parallel_line_l325_32571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l325_32518

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y^2 = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_intersection_length :
  ∀ A B : Point,
  A ∈ Parabola →
  B ∈ Parabola →
  (∃ t : ℝ, A = ⟨t * (A.x - focus.x) + focus.x, t * (A.y - focus.y) + focus.y⟩) →
  (∃ s : ℝ, B = ⟨s * (B.x - focus.x) + focus.x, s * (B.y - focus.y) + focus.y⟩) →
  A.x + B.x = 6 →
  distance A B = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l325_32518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_tournament_solution_l325_32512

/-- Represents the number of women in the tournament -/
def n : ℕ → ℕ := id

/-- Total number of players in the tournament -/
def total_players (n : ℕ) : ℕ := 3 * n

/-- Total number of matches played in the tournament -/
def total_matches (n : ℕ) : ℕ := (total_players n * (total_players n - 1)) / 2

/-- Number of matches won by women -/
def women_wins (n : ℕ) : ℚ := 7 * ((total_matches n : ℚ) / 12)

/-- Number of matches won by men -/
def men_wins (n : ℕ) : ℚ := 5 * ((total_matches n : ℚ) / 12)

/-- The theorem states that given the conditions of the tennis tournament,
    the only value of n that satisfies all conditions is 3 -/
theorem tennis_tournament_solution :
  (∀ m : ℕ, m ≠ 3 → (women_wins m + men_wins m ≠ total_matches m ∨
                      ¬(women_wins m).isInt ∨
                      ¬(men_wins m).isInt)) ∧
  (women_wins 3 + men_wins 3 = total_matches 3 ∧
   (women_wins 3).isInt ∧
   (men_wins 3).isInt) :=
by sorry

#eval total_matches 3
#eval women_wins 3
#eval men_wins 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_tournament_solution_l325_32512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_digit_placement_l325_32521

/-- Represents a table filled with digits -/
def DigitTable := Fin 5 → Fin 8 → Fin 10

/-- Checks if a digit appears exactly four times in the given list -/
def appearsFourTimes (digit : Fin 10) (list : List (Fin 10)) : Prop :=
  (list.filter (· = digit)).length = 4

/-- Checks if each digit appears exactly four times in each row and column -/
def validTable (table : DigitTable) : Prop :=
  ∀ d : Fin 10,
    (∀ row : Fin 5, appearsFourTimes d (List.ofFn (λ col => table row col))) ∧
    (∀ col : Fin 8, appearsFourTimes d (List.ofFn (λ row => table row col)))

/-- Theorem stating the impossibility of the task -/
theorem impossible_digit_placement : ¬ ∃ (table : DigitTable), validTable table := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_digit_placement_l325_32521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l325_32542

/-- Properties of a circle with diameter 10 meters -/
theorem circle_properties (π : ℝ) (h : π > 0) :
  let d : ℝ := 10  -- diameter in meters
  let r : ℝ := d / 2  -- radius in meters
  let area_m2 : ℝ := π * r^2  -- area in square meters
  let area_cm2 : ℝ := area_m2 * 10000  -- area in square centimeters
  let circumference : ℝ := 2 * π * r  -- circumference in meters
  area_cm2 = 250000 * π ∧ circumference = 10 * π := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l325_32542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_for_130_units_units_for_90_yuan_l325_32517

/-- Calculates the cost of electricity usage based on the given pricing structure -/
def electricityCost (units : ℕ) : ℚ :=
  if units ≤ 100 then
    (units : ℚ) * (1/2)
  else
    50 + (units - 100 : ℚ) * (4/5)

/-- Theorem for the cost of 130 units of electricity -/
theorem cost_for_130_units :
  electricityCost 130 = 74 := by sorry

/-- Finds the number of units consumed given the total cost -/
def unitsConsumed (cost : ℚ) : ℕ :=
  if cost ≤ 50 then
    (cost * 2).floor.toNat
  else
    ((cost - 50) / (4/5) + 100).floor.toNat

/-- Theorem for the units consumed when the cost is 90 yuan -/
theorem units_for_90_yuan :
  unitsConsumed 90 = 150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_for_130_units_units_for_90_yuan_l325_32517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_squared_inverse_relation_sum_squared_residuals_l325_32527

/-- Correlation coefficient -/
def R_squared : ℝ → ℝ := sorry

/-- Sum of squared residuals -/
def sum_squared_residuals : ℝ → ℝ := sorry

/-- Theorem: As R² increases, the sum of squared residuals decreases -/
theorem R_squared_inverse_relation_sum_squared_residuals :
  ∀ (x y : ℝ), x < y → R_squared x < R_squared y → sum_squared_residuals x > sum_squared_residuals y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_squared_inverse_relation_sum_squared_residuals_l325_32527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l325_32584

theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (-3, 0) →
  (2 * a.1 + b.1) * (a.2 - m * b.2) = (2 * a.2 + b.2) * (a.1 - m * b.1) →
  m = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l325_32584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_points_order_l325_32599

noncomputable section

-- Define the inverse proportion function
def f (x : ℝ) : ℝ := 6 / x

-- Define the points A, B, and C as functions
def A (x₁ : ℝ) : ℝ × ℝ := (x₁, 6)
def B (x₂ : ℝ) : ℝ × ℝ := (x₂, 12)
def C (x₃ : ℝ) : ℝ × ℝ := (x₃, -6)

-- State the theorem
theorem inverse_proportion_points_order 
  (x₁ x₂ x₃ : ℝ)
  (h1 : f x₁ = (A x₁).2)
  (h2 : f x₂ = (B x₂).2)
  (h3 : f x₃ = (C x₃).2)
  : x₃ < x₂ ∧ x₂ < x₁ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_points_order_l325_32599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propaganda_group_selection_l325_32585

def group1_size : ℕ := 3
def group2_size : ℕ := 3
def group3_size : ℕ := 4
def total_selected : ℕ := 4

theorem propaganda_group_selection :
  (Finset.sum (Finset.filter
    (fun (abc : ℕ × ℕ × ℕ) => 
      let (a, b, c) := abc
      a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ a + b + c = total_selected)
    (Finset.product (Finset.range (group1_size + 1))
      (Finset.product (Finset.range (group2_size + 1)) (Finset.range (group3_size + 1)))))
    (fun (abc : ℕ × ℕ × ℕ) => 
      let (a, b, c) := abc
      Nat.choose group1_size a * Nat.choose group2_size b * Nat.choose group3_size c)) = 126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propaganda_group_selection_l325_32585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_extended_point_is_circle_l325_32595

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  a : ℝ  -- Half of the major axis length

/-- A point P on the ellipse -/
def PointOnEllipse (e : Ellipse) (P : ℝ × ℝ) : Prop :=
  dist P e.F₁ + dist P e.F₂ = 2 * e.a

/-- Check if Q is on the ray from F₁ through P -/
def OnRay (F₁ P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t ≥ 0 ∧ Q = (F₁.1 + t * (P.1 - F₁.1), F₁.2 + t * (P.2 - F₁.2))

/-- The point Q extended from F₁P such that |PQ| = |PF₂| -/
def ExtendedPoint (e : Ellipse) (P Q : ℝ × ℝ) : Prop :=
  PointOnEllipse e P ∧ OnRay e.F₁ P Q ∧ dist P Q = dist P e.F₂

/-- The theorem stating that the locus of Q is a circle -/
theorem locus_of_extended_point_is_circle (e : Ellipse) :
  ∀ Q : ℝ × ℝ, (∃ P : ℝ × ℝ, ExtendedPoint e P Q) →
  dist Q e.F₁ = 2 * e.a :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_extended_point_is_circle_l325_32595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_alpha_l325_32514

-- Define the angle α
noncomputable def α : Real := Real.arctan (4/3)

-- Define the condition that the initial side of α coincides with the positive half of the x-axis
axiom initial_side : ∃ (t : Real), t > 0 ∧ (t * 1 = t ∧ t * 0 = 0)

-- Define the condition that the terminal side of α lies on the ray 3x - 4y = 0 (where x < 0)
axiom terminal_side : ∃ (x y : Real), x < 0 ∧ 3 * x - 4 * y = 0

-- State the theorem
theorem sin_minus_cos_alpha : Real.sin α - Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_alpha_l325_32514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_terminating_decimal_l325_32589

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n + 150 = 2^a * 5^b

theorem smallest_n_for_terminating_decimal :
  (∃ n : ℕ, is_terminating_decimal n ∧ ∀ m : ℕ, is_terminating_decimal m → n ≤ m) ∧
  (∀ n : ℕ, (∀ m : ℕ, is_terminating_decimal m → n ≤ m) → n = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_terminating_decimal_l325_32589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_power_modulo_implies_nth_power_l325_32574

theorem nth_power_modulo_implies_nth_power (a n : ℕ) (ha : a > 0) (hn : n > 0)
  (h : ∀ k : ℕ, k ≥ 1 → ∃ b : ℕ, a ≡ b^n [MOD k]) : 
  ∃ m : ℕ, a = m^n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_power_modulo_implies_nth_power_l325_32574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_trig_expression_l325_32534

theorem second_quadrant_trig_expression (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) -- α is in the second quadrant
  (h2 : Real.sin α = 3 / 5) : 
  (1 + Real.sin α + Real.cos α + 2 * Real.sin α * Real.cos α) / (1 + Real.sin α + Real.cos α) = -1 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_trig_expression_l325_32534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implications_l325_32586

theorem inequality_implications (x y : ℝ) (h : (3 : ℝ)^x - (3 : ℝ)^y < (4 : ℝ)^(-x) - (4 : ℝ)^(-y)) :
  x < y ∧ (2 : ℝ)^(-y) < (2 : ℝ)^(-x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implications_l325_32586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_faces_imply_third_cyclic_l325_32560

/-- A quadrilateral -/
structure Quadrilateral where
  -- We'll leave this abstract for now
  mk :: -- constructor

/-- A truncated triangular pyramid -/
structure TruncatedTriangularPyramid where
  /-- The three lateral faces of the pyramid -/
  lateral_faces : Fin 3 → Quadrilateral

/-- A cyclic quadrilateral is a quadrilateral that can be inscribed in a circle -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- Main theorem: If two lateral faces of a truncated triangular pyramid are cyclic,
    then the third lateral face is also cyclic -/
theorem cyclic_faces_imply_third_cyclic (pyramid : TruncatedTriangularPyramid)
  (h1 : is_cyclic (pyramid.lateral_faces 0))
  (h2 : is_cyclic (pyramid.lateral_faces 1)) :
  is_cyclic (pyramid.lateral_faces 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_faces_imply_third_cyclic_l325_32560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_parallel_lines_planes_l325_32554

-- Define a type for lines in 3D space
def Line : Type := ℝ → ℝ × ℝ × ℝ

-- Define a predicate for parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define a function to count the number of planes determined by three lines
def planes_determined (l1 l2 l3 : Line) : Nat := sorry

-- Theorem statement
theorem three_parallel_lines_planes :
  ∀ (l1 l2 l3 : Line),
  parallel l1 l2 ∧ parallel l2 l3 ∧ parallel l1 l3 →
  (planes_determined l1 l2 l3 = 1 ∨ planes_determined l1 l2 l3 = 3) :=
by
  sorry

#check three_parallel_lines_planes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_parallel_lines_planes_l325_32554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_can_pass_through_l325_32528

/-- The radius of the Earth (and our sphere) in meters -/
noncomputable def R : ℝ := 6371000

/-- The length by which the wire is extended in meters -/
def extension : ℝ := 1

/-- The height of a typical mouse in meters -/
def mouse_height : ℝ := 0.05

/-- The gap formed between the wire and the sphere's surface -/
noncomputable def gap (R : ℝ) (extension : ℝ) : ℝ := extension / (2 * Real.pi)

theorem mouse_can_pass_through (h : R > 0) :
  gap R extension > mouse_height := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_can_pass_through_l325_32528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_remaining_number_l325_32506

def A : ℕ := 12345678987654321

def digits_to_remove : List ℕ := [1, 2, 3, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

def sum_of_removed_digits : ℕ := digits_to_remove.sum

def remaining_digits : List ℕ := [4, 8, 9]

def B : ℕ := 489

theorem smallest_remaining_number :
  sum_of_removed_digits = 60 ∧
  remaining_digits = (Nat.digits 10 A).filter (λ d => d ∉ digits_to_remove) ∧
  B = remaining_digits.foldl (λ acc d => acc * 10 + d) 0 ∧
  ∀ n : ℕ, (Nat.digits 10 n).filter (λ d => d ∉ digits_to_remove) = remaining_digits → n ≥ B :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_remaining_number_l325_32506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_rectangles_l325_32526

/-- Area of overlap between two rectangles -/
noncomputable def area_of_overlap (m n : ℝ) : ℝ := m

/-- Length of segment AB -/
noncomputable def length_AB (m n : ℝ) : ℝ := Real.sqrt (2 * n^2 - 2 * m * n)

/-- Given two overlapping 5x1 rectangles with 2 common vertices, 
    prove the area of overlap and length of segment AB. -/
theorem overlapping_rectangles (m n : ℝ) : 
  m + n = 5 →                            -- Sum of segment lengths
  m^2 + n^2 = 25 →                       -- Pythagorean theorem for diagonal
  m = 2.4 →                              -- Given from solution
  (2.4 : ℝ) = area_of_overlap m n ∧      -- Area of overlap
  (Real.sqrt 26 / 5 : ℝ) = length_AB m n -- Length of segment AB
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_rectangles_l325_32526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cans_is_85_l325_32552

def bag_contents : Fin 10 → ℕ
| 0 => 5   -- First bag
| 1 => 7   -- Second bag
| 2 => 12  -- Third bag
| 3 => 4   -- Fourth bag
| 4 => 8   -- Fifth bag
| 5 => 10  -- Sixth bag
| 6 => 15  -- Seventh bag
| 7 => 6   -- Eighth bag
| 8 => 5   -- Ninth bag
| 9 => 13  -- Tenth bag

theorem total_cans_is_85 : (Finset.univ.sum (fun i => bag_contents i)) = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cans_is_85_l325_32552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_side_length_40cm_l325_32549

/-- The length of one side of a regular octagon created from a wire of given total length -/
noncomputable def octagon_side_length (total_length : ℝ) : ℝ :=
  total_length / 8

theorem octagon_side_length_40cm :
  octagon_side_length 40 = 5 := by
  -- Unfold the definition of octagon_side_length
  unfold octagon_side_length
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_side_length_40cm_l325_32549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closerToF_probability_is_half_l325_32582

/-- A right triangle with side lengths 6, 8, and 10 -/
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  is_right : DE^2 + EF^2 = DF^2
  side_lengths : DE = 6 ∧ EF = 8 ∧ DF = 10

/-- The probability that a random point in the triangle is closer to F than to D or E -/
noncomputable def closerToF_probability (t : RightTriangle) : ℝ :=
  (t.EF * t.DF) / (4 * (t.DE * t.EF))

/-- Theorem: The probability is 1/2 -/
theorem closerToF_probability_is_half (t : RightTriangle) :
  closerToF_probability t = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closerToF_probability_is_half_l325_32582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_natural_number_solution_l325_32522

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem unique_natural_number_solution : 
  ∃! (x : ℕ), floor (3.8 * (x : ℝ)) = (floor 3.8 : ℤ) * (x : ℤ) + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_natural_number_solution_l325_32522
