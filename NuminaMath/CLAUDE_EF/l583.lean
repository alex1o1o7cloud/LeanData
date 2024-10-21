import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_cdf_correct_l583_58360

/-- The cumulative distribution function of an exponentially distributed random variable -/
noncomputable def exponential_cdf (α : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 0 else 1 - Real.exp (-α * x)

/-- The probability density function of an exponentially distributed random variable -/
noncomputable def exponential_pdf (α : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 0 else α * Real.exp (-α * x)

theorem exponential_cdf_correct (α : ℝ) (h : α > 0) :
  ∀ x : ℝ, exponential_cdf α x = ∫ t in Set.Iic x, exponential_pdf α t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_cdf_correct_l583_58360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l583_58396

/-- Definition of the complex number z as a function of real m -/
def z (m : ℝ) : ℂ := (1 + Complex.I) * m^2 + (5 - 2*Complex.I) * m + (6 - 15*Complex.I)

/-- Theorem stating the conditions for z to be real, purely imaginary, or on the line x+y+7=0 -/
theorem z_properties :
  (∀ m : ℝ, (z m).im = 0 ↔ m = 5 ∨ m = -3) ∧
  (∀ m : ℝ, (z m).re = 0 ↔ m = -2) ∧
  (∀ m : ℝ, (z m).re + (z m).im = -7 ↔ m = 1/2 ∨ m = -2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l583_58396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l583_58383

/-- An arithmetic sequence with positive terms and common difference 2 -/
def ArithSeq (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) - a n = 2)

/-- Sum of terms with indices that are powers of 3 -/
def SumPowersOf3 (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range (n + 1)) (fun k => a (3^k))

theorem arithmetic_sequence_properties
    (a : ℕ → ℝ)
    (h_seq : ArithSeq a)
    (h_eq : a 2 * a 4 = 4 * a 3 + 1) :
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, SumPowersOf3 a n = 3^(n+1) - n - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l583_58383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_is_equation_l583_58316

/-- Definition of an equation -/
def is_equation (s : String) : Prop :=
  ∃ (lhs rhs : ℚ → ℚ) (x : ℚ), s = toString (lhs x) ++ " = " ++ toString (rhs x) ∧ (∀ x, lhs x = rhs x)

/-- The statement "9x-1=6" -/
def statement : String := "9x-1=6"

/-- Theorem: The statement "9x-1=6" is an equation -/
theorem statement_is_equation : is_equation statement := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_is_equation_l583_58316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_roots_condition_roots_signs_roots_product_l583_58314

-- Define the quadratic equation
def quadratic_equation (α : Real) (x : Real) : Prop :=
  (2 * Real.cos α - 1) * x^2 - 4 * x + 4 * Real.cos α + 2 = 0

-- Define the condition for α
def alpha_condition (α : Real) : Prop :=
  α < Real.pi / 2

-- Theorem for the real roots condition
theorem real_roots_condition (α : Real) :
  (∃ x : Real, quadratic_equation α x) ↔ 
  (Real.pi / 6 ≤ α ∧ α < Real.pi / 2) :=
sorry

-- Theorem for the signs of the roots
theorem roots_signs (α : Real) :
  (∃ x y : Real, x > 0 ∧ y > 0 ∧ quadratic_equation α x ∧ quadratic_equation α y) ↔
  (Real.pi / 6 ≤ α ∧ α < Real.pi / 3) :=
sorry

-- Theorem for the product of roots
theorem roots_product (α : Real) :
  ∃ x y : Real, quadratic_equation α x ∧ quadratic_equation α y ∧
  x * y = 2 * (Real.tan (3 * α / 2) * (1 / Real.tan (α / 2))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_roots_condition_roots_signs_roots_product_l583_58314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l583_58395

/-- Represents a rhombus with diagonals d1 and d2 -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ

/-- Calculates the area of a rhombus given its diagonals -/
noncomputable def Rhombus.area (r : Rhombus) : ℝ := (r.d1 * r.d2) / 2

/-- Theorem: If a rhombus has one diagonal of 20 and an area of 160, then the other diagonal is 16 -/
theorem rhombus_other_diagonal (r : Rhombus) (h1 : r.d1 = 20) (h2 : r.area = 160) : r.d2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l583_58395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l583_58398

/-- Calculate the length of a train given its speed and the time it takes to cross a platform of known length. -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  speed = 72 * (1000 / 3600) → 
  platform_length = 250 → 
  crossing_time = 26 → 
  speed * crossing_time - platform_length = 270 := by
  intros h_speed h_platform h_time
  -- Convert the goal to concrete numbers
  have : speed * crossing_time - platform_length = 20 * 26 - 250 := by
    rw [h_speed, h_platform, h_time]
    norm_num
  -- Simplify the right-hand side
  rw [this]
  norm_num

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l583_58398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_on_chessboard_l583_58300

/-- Represents a chessboard with rooks placed on it. -/
structure Chessboard :=
  (rooks : Finset (Fin 8 × Fin 8))

/-- Checks if a rook attacks another rook. -/
def attacks (p q : Fin 8 × Fin 8) : Bool :=
  p.1 = q.1 ∨ p.2 = q.2

/-- A valid rook placement is one where each rook attacks at most one other rook. -/
def valid_placement (b : Chessboard) : Prop :=
  ∀ p ∈ b.rooks, (b.rooks.filter (fun q => attacks p q)).card ≤ 2

/-- The theorem stating the maximum number of rooks on an 8x8 chessboard
    where each rook attacks at most one other rook. -/
theorem max_rooks_on_chessboard :
  (∃ b : Chessboard, valid_placement b ∧ b.rooks.card = 10) ∧
  (∀ b : Chessboard, valid_placement b → b.rooks.card ≤ 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_on_chessboard_l583_58300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_point_cosine_l583_58384

/-- Given an angle α and a point P(m,1) on its terminal side, where cos α = -1/3,
    prove that m = -√2/4 and tan α = -2√2 -/
theorem angle_point_cosine (α : ℝ) (m : ℝ) 
    (h1 : ∃ (P : ℝ × ℝ), P = (m, 1) ∧ P.1 = m ∧ P.2 = 1)
    (h2 : Real.cos α = -1/3) : 
    m = -Real.sqrt 2 / 4 ∧ Real.tan α = -2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_point_cosine_l583_58384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l583_58317

/-- A set of points where every point is the midpoint of a segment with endpoints in the set -/
def MidpointSet (S : Set ℝ) : Prop :=
  ∀ x, x ∈ S → ∃ a b, a ∈ S ∧ b ∈ S ∧ x = (a + b) / 2

/-- Theorem: If S is a MidpointSet, then S is infinite -/
theorem midpoint_set_infinite (S : Set ℝ) (h : MidpointSet S) : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l583_58317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_fraction_l583_58385

theorem irreducible_fraction (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_fraction_l583_58385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_distributions_count_l583_58373

/-- Represents a student --/
inductive Student : Type
  | s1 : Student
  | s2 : Student
  | s3 : Student
deriving BEq, Repr

/-- Represents a book --/
inductive Book : Type
  | chinese : Book
  | math : Book
deriving BEq, Repr

/-- A distribution of books to students --/
def Distribution := Student → List Book

/-- Checks if a distribution is valid --/
def isValidDistribution (d : Distribution) : Prop :=
  (∀ s : Student, d s ≠ []) ∧
  (∃! s : Student, Book.math ∈ d s) ∧
  (∀ s : Student, (d s).count Book.chinese ≤ 2) ∧
  ((d Student.s1).count Book.chinese + (d Student.s2).count Book.chinese + (d Student.s3).count Book.chinese = 3)

/-- The number of valid distributions --/
def numValidDistributions : ℕ := sorry

theorem valid_distributions_count :
  numValidDistributions = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_distributions_count_l583_58373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l583_58380

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^2 * Real.sin (2*x) + (a-2) * Real.cos (2*x)

theorem max_value_of_f (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-π/4 - x)) →
  (∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) →
  (∃ x : ℝ, f a x = Real.sqrt 2 ∨ f a x = 4 * Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l583_58380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distinct_tetration_modulus_l583_58349

-- Define the tetration operation
def tetration (a : ℕ+) : ℕ → ℕ
| 0 => 1  -- Define the case for 0
| 1 => a
| n + 1 => a ^ (tetration a n)

-- Define the property we're looking for
def has_distinct_tetration (n : ℕ+) : Prop :=
  ∃ a : ℕ+, ¬(tetration a 6 ≡ tetration a 7 [MOD n])

-- State the theorem
theorem smallest_distinct_tetration_modulus :
  (∀ k : ℕ+, k < 283 → ¬(has_distinct_tetration k)) ∧ has_distinct_tetration 283 := by
  sorry

#check smallest_distinct_tetration_modulus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distinct_tetration_modulus_l583_58349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_7_value_l583_58347

mutual
  def a : ℕ → ℚ
    | 0 => 3
    | (n + 1) => (a n)^2 / (b n)

  def b : ℕ → ℚ
    | 0 => 5
    | (n + 1) => (b n)^2 / (a n)
end

theorem b_7_value : b 7 = 5^50 / 3^41 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_7_value_l583_58347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_simple_ratio_real_on_circle_or_line_iff_cross_ratio_real_l583_58315

-- Define the simple ratio of three complex numbers
noncomputable def simple_ratio (a b c : ℂ) : ℂ := (a - b) / (a - c)

-- Define the cross ratio of four complex numbers
noncomputable def cross_ratio (a b c d : ℂ) : ℂ := ((a - c) / (a - d)) / ((b - c) / (b - d))

-- Theorem for part (a)
theorem collinear_iff_simple_ratio_real (a b c : ℂ) :
  (∃ (t : ℝ), b = a + t • (c - a)) ↔ (∃ (r : ℝ), simple_ratio a b c = r) := by
  sorry

-- Theorem for part (b)
theorem on_circle_or_line_iff_cross_ratio_real (a b c d : ℂ) :
  (∃ (z w r : ℂ), r ≠ 0 ∧ (a - z) * (b - z) * (c - z) * (d - z) = r * (a - w) * (b - w) * (c - w) * (d - w)) ↔
  (∃ (r : ℝ), cross_ratio a b c d = r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_simple_ratio_real_on_circle_or_line_iff_cross_ratio_real_l583_58315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_difference_after_relocation_l583_58352

/-- Represents the energy stored in a configuration of three point charges -/
noncomputable def energy_stored (d : ℝ) (k q : ℝ) (config : List ℝ) : ℝ :=
  config.sum

/-- Initial configuration of charges -/
noncomputable def initial_config (d : ℝ) (k q : ℝ) : List ℝ :=
  [k * q^2 / d, k * q^2 / d, k * q^2 / (2*d)]

/-- Final configuration of charges after relocation -/
noncomputable def final_config (d : ℝ) (k q : ℝ) : List ℝ :=
  [k * q^2 / d, k * q^2 / d, k * q^2 / d]

theorem energy_difference_after_relocation 
  (d k q : ℝ) 
  (h1 : d > 0) 
  (h2 : k > 0) 
  (h3 : q ≠ 0) 
  (h4 : energy_stored d k q (initial_config d k q) = 18) :
  energy_stored d k q (final_config d k q) - 
  energy_stored d k q (initial_config d k q) = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_difference_after_relocation_l583_58352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l583_58357

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line
noncomputable def line (x y : ℝ) : Prop := y = -Real.sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the directrix of the parabola
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define the intersection points M and N
def intersection_points (p : ℝ) (M N : ℝ × ℝ) : Prop :=
  parabola p M.1 M.2 ∧ parabola p N.1 N.2 ∧
  line M.1 M.2 ∧ line N.1 N.2

-- State the theorem
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ) :
  parabola p M.1 M.2 →
  parabola p N.1 N.2 →
  line (focus p).1 (focus p).2 →
  intersection_points p M N →
  (p = 2 ∧
   ∃ (center : ℝ × ℝ) (radius : ℝ),
     (center.1 - M.1)^2 + (center.2 - M.2)^2 = radius^2 ∧
     (center.1 - N.1)^2 + (center.2 - N.2)^2 = radius^2 ∧
     abs (center.1 - (-p/2)) = radius) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l583_58357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l583_58338

noncomputable def f (ω : ℝ) (x : ℝ) := Real.cos (ω * x + Real.pi / 4)

theorem omega_range (ω : ℝ) :
  ω > 0 →
  (∀ x ∈ Set.Icc 0 Real.pi, f ω x ∈ Set.Icc (-1) (Real.sqrt 2 / 2)) →
  ω ∈ Set.Icc (3/4) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l583_58338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_neg3_4_l583_58307

noncomputable def rect_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.pi + Real.arctan (y / x)
           else if x < 0 && y < 0 then -Real.pi + Real.arctan (y / x)
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, θ)

theorem rect_to_polar_neg3_4 :
  let (r, θ) := rect_to_polar (-3) 4
  r = 5 ∧ θ = Real.pi - Real.arctan (4 / 3) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_neg3_4_l583_58307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l583_58366

/-- Triangle ABC with side lengths a, b, c and corresponding angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law for triangles -/
axiom sine_law {t : Triangle} : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The cosine law for triangles -/
axiom cosine_law {t : Triangle} : t.c^2 = t.a^2 + t.b^2 - 2 * t.a * t.b * Real.cos t.C

/-- The area of a triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ := 1/2 * t.a * t.c * Real.sin t.B

theorem triangle_properties (t : Triangle) (h1 : t.b = 2) :
  (t.a * Real.sin (2 * t.B) = Real.sqrt 3 * t.b * Real.sin t.A → t.B = π/6) ∧
  (∃ r : ℝ, t.a = 2/r ∧ t.c = 2*r → 
    ∀ s : Triangle, s.b = 2 → triangle_area s ≤ Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l583_58366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bouncing_ball_distance_l583_58394

/-- Calculates the total distance traveled by a bouncing ball -/
noncomputable def total_distance (initial_height : ℝ) (bounce_count : ℕ) : ℝ :=
  let descending := initial_height * (1 - (1/2)^bounce_count) / (1 - 1/2)
  let ascending := initial_height * (1 - (1/2)^(bounce_count - 1)) / 2
  descending + ascending

/-- The problem statement as a theorem -/
theorem bouncing_ball_distance :
  total_distance 100 4 = 275 := by
  -- The proof goes here
  sorry

-- Remove the #eval statement as it's not computable
-- #eval total_distance 100 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bouncing_ball_distance_l583_58394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_of_oscillating_sine_l583_58311

noncomputable def oscillating_sine_function (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (b * x + c) + d

theorem amplitude_of_oscillating_sine (a b c d : ℝ) :
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (d > 0) →
  (∀ x, oscillating_sine_function a b c d x ≤ 5) →
  (∀ x, oscillating_sine_function a b c d x ≥ -3) →
  (∃ x1 x2, oscillating_sine_function a b c d x1 = 5 ∧ 
            oscillating_sine_function a b c d x2 = -3) →
  a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_of_oscillating_sine_l583_58311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_f_range_l583_58333

noncomputable def a (x : ℝ) : ℝ × ℝ := (1, Real.cos (2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), -Real.sqrt 3)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem cos_2theta_value (θ : ℝ) (h : f (θ/2 + 2*Real.pi/3) = 6/5) :
  Real.cos (2 * θ) = 7/25 := by sorry

theorem f_range :
  Set.range f = Set.Icc (-Real.sqrt 3) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_f_range_l583_58333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_root_product_l583_58379

theorem simplify_root_product : Real.sqrt 27 * Real.rpow 125 (1/3) = 15 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_root_product_l583_58379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_80_l583_58387

/-- A rectangle and an isosceles triangle with the following properties:
  - Both have height 10
  - Rectangle's width is 12
  - Triangle's base is 8
  - They share a vertex at (12, 0)
  - A line is drawn from the top left of the rectangle to the top of the triangle -/
structure GeometricSetup where
  rect_height : ℝ
  rect_width : ℝ
  tri_base : ℝ
  shared_x : ℝ
  line_start : ℝ × ℝ
  line_end : ℝ × ℝ
  h_rect_height : rect_height = 10
  h_rect_width : rect_width = 12
  h_tri_base : tri_base = 8
  h_shared_x : shared_x = 12
  h_line_start : line_start = (0, rect_height)
  h_line_end : line_end = (shared_x + tri_base / 2, rect_height)

/-- The area of the shaded region in the geometric setup -/
noncomputable def shaded_area (setup : GeometricSetup) : ℝ :=
  (setup.line_end.1 - setup.line_start.1) * setup.rect_height / 2

/-- Theorem stating that the shaded area is 80 square units -/
theorem shaded_area_is_80 (setup : GeometricSetup) : shaded_area setup = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_80_l583_58387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_330_degrees_to_radians_l583_58344

/-- Converts degrees to radians -/
noncomputable def degreesToRadians (degrees : ℝ) : ℝ := degrees * (Real.pi / 180)

/-- Theorem: Converting -330° to radians equals -11π/6 -/
theorem negative_330_degrees_to_radians : 
  degreesToRadians (-330) = -11 * Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_330_degrees_to_radians_l583_58344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersection_circle_theorem_l583_58306

/-- A quadratic function f(x) = x^2 + 2x + b with three axis intersections -/
def QuadraticFunction (b : ℝ) : ℝ → ℝ := λ x ↦ x^2 + 2*x + b

/-- The circle passing through the three intersection points of the quadratic function with the axes -/
def IntersectionCircle (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - (b + 1)*p.2 + b = 0}

/-- The theorem stating the range of b and the fixed points of the intersection circle -/
theorem quadratic_intersection_circle_theorem :
  (∃ b : ℝ, b < 1 ∧ b ≠ 0 ∧
    (∃ x₁ x₂ x₃ : ℝ, 
      QuadraticFunction b x₁ = 0 ∧
      QuadraticFunction b x₂ = 0 ∧
      QuadraticFunction b 0 = x₃ ∧
      x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0)) ∧
  (∀ b : ℝ, b < 1 → b ≠ 0 →
    (∃ x₁ x₂ x₃ : ℝ, 
      QuadraticFunction b x₁ = 0 ∧
      QuadraticFunction b x₂ = 0 ∧
      QuadraticFunction b 0 = x₃ ∧
      x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0) →
    (0, 1) ∈ IntersectionCircle b ∧
    (-2, 1) ∈ IntersectionCircle b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersection_circle_theorem_l583_58306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l583_58321

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the vectors m and n
noncomputable def m (B : ℝ) : ℝ × ℝ := (2 * Real.sin B, -Real.sqrt 3)
noncomputable def n (B : ℝ) : ℝ × ℝ := (Real.cos (2 * B), 2 * (Real.cos (B / 2))^2 - 1)

-- Define the parallel condition
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h_parallel : are_parallel (m t.B) (n t.B)) 
  (h_b : t.b = 2) : 
  t.B = π / 3 ∧ 
  (∀ s : ℝ, s = (1/2) * t.a * t.c * Real.sin t.B → s ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l583_58321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l583_58325

theorem simplify_expression (a : ℝ) (ha : a ≠ 0) :
  (a^(-2 : ℤ) / a^(5 : ℤ)) * (4 * a / ((1/2 * a)^(-3 : ℤ))) = 1 / (2 * a^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l583_58325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_repeating_decimal_36_l583_58362

/-- The reciprocal of the common fraction form of the repeating decimal 0.363636... is 11/4 -/
theorem reciprocal_of_repeating_decimal_36 : 
  (1 / (0.363636363636 : ℚ)) = 11/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_repeating_decimal_36_l583_58362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_8_digit_properties_l583_58343

def base_10_num : ℕ := 7927

def base_8_representation : List ℕ := [1, 7, 7, 5, 7]

theorem base_8_digit_properties :
  (base_10_num.digits 8 = base_8_representation) ∧
  (base_8_representation.prod = 1715) ∧
  (base_8_representation.sum = 27) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_8_digit_properties_l583_58343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_results_l583_58340

theorem election_results (vote_percentage_A vote_percentage_B : ℚ) 
  (margin_A_over_B : ℕ) : 
  vote_percentage_A = 42 / 100 →
  vote_percentage_B = 37 / 100 →
  vote_percentage_A - vote_percentage_B = 5 / 100 →
  margin_A_over_B = 650 →
  ∃ (total_votes : ℕ),
    total_votes = 13000 ∧
    ∃ (votes_B votes_C : ℕ),
      votes_B = (vote_percentage_B * total_votes).floor ∧
      votes_C = ((1 - vote_percentage_A - vote_percentage_B) * total_votes).floor ∧
      votes_B - votes_C = 2080 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_results_l583_58340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_x_range_l583_58351

-- Define the exponential function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions
axiom f_is_exponential : ∃ (a : ℝ), ∀ (x : ℝ), f x = a^x
axiom f_passes_through_2_4 : f 2 = 4

-- Theorem 1: Prove that f(x) = 2^x
theorem f_expression : ∀ (x : ℝ), f x = 2^x := by sorry

-- Theorem 2: Prove that for all x such that f(x - 1) < 1, x < 1
theorem x_range : ∀ (x : ℝ), f (x - 1) < 1 → x < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_x_range_l583_58351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_exists_l583_58336

def P (n : ℕ) : Finset ℕ :=
  Finset.image (λ k => 2^(n - k) * 3^k) (Finset.range (n + 1))

def S (X : Finset ℕ) : ℕ := X.sum id

theorem subset_sum_exists (n : ℕ) (y : ℝ) 
  (h : 0 ≤ y ∧ y ≤ 3^(n + 1) - 2^(n + 1)) : 
  ∃ Y : Finset ℕ, Y ⊆ P n ∧ 0 ≤ y - (S Y) ∧ y - (S Y) < 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_exists_l583_58336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_intersection_point_l583_58372

/-- Definition of function f -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 7*x + 12) / (2*x - 6)

/-- Definition of function g with parameters a, b, c -/
noncomputable def g (a b c : ℝ) (x : ℝ) : ℝ := (a*x^2 + b*x + c) / (x - 3)

/-- The vertical asymptote of f and g -/
def vertical_asymptote : ℝ := 3

/-- The y-intercept of the intersection of oblique asymptotes -/
def oblique_asymptote_intersection : ℝ := -2

/-- One intersection point of f and g -/
def known_intersection : ℝ := -3

/-- Theorem stating the other intersection point of f and g -/
theorem other_intersection_point :
  ∃ (a b c : ℝ),
    (∀ x : ℝ, x ≠ vertical_asymptote → f x = g a b c x → x = known_intersection ∨ x = -2) ∧
    f (-2) = g a b c (-2) ∧
    f (-2) = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_intersection_point_l583_58372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_range_l583_58328

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line
def line_equation (x y a : ℝ) : Prop := x + y = a

-- Define the distance from a point to the line
noncomputable def distance_to_line (x y a : ℝ) : ℝ := |x + y - a| / Real.sqrt 2

-- State the theorem
theorem circle_line_intersection_range (a : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ),
    circle_equation x1 y1 ∧ circle_equation x2 y2 ∧
    line_equation x1 y1 a ∧ line_equation x2 y2 a ∧
    distance_to_line x1 y1 a = 1 ∧
    distance_to_line x2 y2 a = 1 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)) →
  -3 * Real.sqrt 2 < a ∧ a < 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_range_l583_58328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_transformed_curve_l583_58378

-- Define the original curve C in polar coordinates
noncomputable def C (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- Define the transformation
def transform (p : Real × Real) : Real × Real :=
  (3 * p.1, p.2)

-- Define the transformed curve C'
noncomputable def C' (θ : Real) : Real × Real :=
  transform (C θ)

-- Define the function to be minimized
noncomputable def f (p : Real × Real) : Real :=
  p.1 + 2 * Real.sqrt 3 * p.2

theorem min_value_on_transformed_curve :
  ∃ (k : Real), ∀ (θ : Real), f (C' θ) ≥ k ∧ ∃ (θ₀ : Real), f (C' θ₀) = k ∧ k = -Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_transformed_curve_l583_58378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_trajectory_length_l583_58305

/-- Represents the radius of the largest disk -/
noncomputable def largest_radius : ℝ := 5

/-- Represents the ratio of each succeeding disk's radius to its previous disk -/
noncomputable def radius_ratio : ℝ := 2/3

/-- Represents the angular velocity of each disk in radians per second -/
noncomputable def angular_velocity : ℝ := Real.pi/6

/-- Represents the time elapsed in seconds -/
noncomputable def time_elapsed : ℝ := 12

/-- Represents the complex number corresponding to the rotation of each disk relative to its parent -/
noncomputable def z (t : ℝ) : ℂ := radius_ratio * Complex.exp (Complex.I * angular_velocity * t)

/-- Represents the position of Alice after time t -/
noncomputable def alice_position (t : ℝ) : ℂ := (15/2) * (z t / (1 - z t))

/-- Theorem stating that the length of Alice's trajectory after 12 seconds is 18π -/
theorem alice_trajectory_length : 
  Complex.abs (alice_position time_elapsed - alice_position 0) = 18 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_trajectory_length_l583_58305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l583_58386

/-- Represents a train with a given length and time to cross a fixed point -/
structure Train where
  length : ℝ
  crossTime : ℝ

/-- Calculates the time for two trains to cross each other when traveling in opposite directions -/
noncomputable def timeToCross (train1 train2 : Train) : ℝ :=
  (train1.length + train2.length) / (train1.length / train1.crossTime + train2.length / train2.crossTime)

theorem trains_crossing_time :
  let train1 : Train := { length := 120, crossTime := 10 }
  let train2 : Train := { length := 120, crossTime := 20 }
  ∃ ε > 0, |timeToCross train1 train2 - 13.33| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l583_58386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_absolute_value_l583_58353

theorem nested_absolute_value : 
  abs (abs (abs (-2 + 1) - 2) + 2) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_absolute_value_l583_58353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l583_58364

def x : ℕ → ℤ → ℤ
  | 0, _ => 0  -- Handle the case when n = 0
  | 1, c => c
  | (n+2), c => x (n+1) c + ⌊(2 * x (n+1) c - ((n+2) + 2)) / (n+2)⌋ + 1

theorem x_general_term (c : ℤ) (n : ℕ) :
  x n c = match c % 3 with
    | 0 => ((c - 1) / 6) * (n + 1) * (n + 2) + 1
    | 1 => ((c - 2) / 6) * (n + 1) * (n + 2) + n + 1
    | 2 => ((c - 3) / 6) * (n + 1) * (n + 2) + ⌊((n + 2)^2 : ℚ) / 4⌋ + 1
    | _ => 0  -- This case is not possible, but needed for exhaustiveness
  := by sorry

#check x_general_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l583_58364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_ratios_l583_58329

/-- Represents a point on the ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  on_ellipse : 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0

/-- The ratio y/x for a point on the ellipse -/
noncomputable def ratio (p : EllipsePoint) : ℝ := p.y / p.x

/-- The maximum value of the ratio y/x for points on the ellipse -/
noncomputable def max_ratio : ℝ := sorry

/-- The minimum value of the ratio y/x for points on the ellipse -/
noncomputable def min_ratio : ℝ := sorry

/-- Theorem stating that the sum of max and min ratios is 37/22 -/
theorem sum_of_max_min_ratios : max_ratio + min_ratio = 37 / 22 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_ratios_l583_58329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sine_angle_equality_l583_58390

theorem contrapositive_sine_angle_equality :
  (∀ A B : ℝ, Real.sin A = Real.sin B → A = B) ↔
  (∀ A B : ℝ, A ≠ B → Real.sin A ≠ Real.sin B) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sine_angle_equality_l583_58390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_condition_cos_A_condition_l583_58350

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 ∧ t.C = Real.pi / 3

-- Theorem 1
theorem area_condition (t : Triangle) :
  triangle_conditions t →
  (1 / 2 * t.a * t.b * Real.sin t.C = Real.sqrt 3) →
  t.a = 2 ∧ t.b = 2 := by sorry

-- Theorem 2
theorem cos_A_condition (t : Triangle) :
  triangle_conditions t →
  Real.cos t.A = Real.sqrt 3 / 3 →
  t.b = 2 * (Real.sqrt 3 + Real.sqrt 2) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_condition_cos_A_condition_l583_58350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l583_58376

theorem inequality_solution (x : ℝ) : 
  (x^2 - 5*x + 6) / ((x - 3)^2) > 0 ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l583_58376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l583_58319

noncomputable def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  Real.sin A = 4/5 ∧ Real.cos B = 5/13 ∧ c = 56

theorem triangle_properties (A B C : Real) (a b c : Real) 
  (h : triangle_ABC A B C a b c) : 
  Real.sin C = 56/65 ∧ 2 * Real.pi * (c / (2 * Real.sin C)) = 65 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l583_58319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_pi_3_l583_58393

theorem sin_2alpha_plus_pi_3 (α : ℝ) (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.cos (α + π/4) = Real.sqrt 5 / 5) : 
  Real.sin (2*α + π/3) = (4 * Real.sqrt 3 + 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_pi_3_l583_58393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_l583_58308

theorem polynomial_root_sum (p q r s : ℝ) : 
  let g (x : ℂ) := x^4 + p*x^3 + q*x^2 + r*x + s
  (g (3*I) = 0 ∧ g (1 + 2*I) = 0) → p + q + r + s = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_l583_58308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_trip_time_l583_58313

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- plane's speed in still air
  w : ℝ  -- wind speed

/-- The time taken for a flight given distance, speed, and wind -/
noncomputable def flightTime (distance : ℝ) (speed : ℝ) (wind : ℝ) : ℝ :=
  distance / (speed + wind)

/-- The conditions of the flight scenario -/
def flightConditions (fs : FlightScenario) : Prop :=
  flightTime fs.d fs.p (-fs.w) = 120 ∧  -- against wind takes 120 minutes
  flightTime fs.d fs.p fs.w = flightTime fs.d fs.p 0 - 20  -- with wind takes 20 minutes less than still air

/-- The theorem stating that the return trip takes either 80 or 11 minutes -/
theorem return_trip_time (fs : FlightScenario) :
  flightConditions fs →
  flightTime fs.d fs.p fs.w = 80 ∨ flightTime fs.d fs.p fs.w = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_trip_time_l583_58313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l583_58301

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)) →
  ((∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l583_58301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_speed_proof_l583_58381

/-- The speed of the current in miles per hour -/
noncomputable def current_speed : ℝ := 2.28571428571

/-- The time taken for the upstream trip in hours -/
noncomputable def upstream_time : ℝ := 20 / 60

/-- The time taken for the downstream trip in hours -/
noncomputable def downstream_time : ℝ := 15 / 60

/-- The constant speed of the motorboat relative to the water in miles per hour -/
noncomputable def boat_speed : ℝ := 16

theorem motorboat_speed_proof :
  let d := boat_speed * upstream_time - current_speed * upstream_time
  d = boat_speed * downstream_time + current_speed * downstream_time →
  boat_speed = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_speed_proof_l583_58381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l583_58312

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 1 ≠ 0) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → a^2 - 3*a - x + 1 ≤ 0) →
  a ∈ Set.Ico 1 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l583_58312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_bounds_l583_58337

/-- Represents Arun's weight in kilograms -/
def arun_weight : ℝ := sorry

/-- Arun's opinion: lower bound of his weight -/
def arun_lower_bound : ℝ := 64

/-- Arun's opinion: upper bound of his weight -/
def arun_upper_bound : ℝ := 72

/-- Arun's brother's opinion: lower bound of Arun's weight -/
def brother_lower_bound : ℝ := 60

/-- Arun's mother's opinion: upper bound of Arun's weight -/
def mother_upper_bound : ℝ := 67

/-- Average of different probable weights of Arun -/
def average_weight : ℝ := 66

/-- Arun's brother's opinion: upper bound of Arun's weight (to be proven) -/
def brother_upper_bound : ℝ := 67

theorem arun_weight_bounds :
  arun_lower_bound < arun_weight ∧
  arun_weight < arun_upper_bound ∧
  brother_lower_bound < arun_weight ∧
  arun_weight ≤ mother_upper_bound ∧
  average_weight = 66 →
  brother_upper_bound = 67 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_bounds_l583_58337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_solution_l583_58397

theorem trigonometric_inequality_solution (x : ℝ) : 
  x ∈ Set.Icc (-4 * Real.pi / 3) (2 * Real.pi / 3) →
  (Real.cos x) ^ 2018 + (Real.sin x) ^ (-2019 : ℤ) ≥ (Real.sin x) ^ 2018 + (Real.cos x) ^ (-2019 : ℤ) ↔
  x ∈ Set.Ioc (-4 * Real.pi / 3) (-Real.pi) ∪ 
      Set.Ioc (-3 * Real.pi / 4) (-Real.pi / 2) ∪ 
      Set.Ioo 0 (Real.pi / 4) ∪ 
      Set.Ioo (Real.pi / 2) (2 * Real.pi / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_solution_l583_58397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_condition_l583_58331

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * a * x^2 - 2*x

-- Define what it means for a function to be monotonically decreasing on an interval
def monotonically_decreasing_on (f : ℝ → ℝ) (S : Set ℝ) :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x > f y

-- State the theorem
theorem monotonically_decreasing_interval_condition (a : ℝ) :
  (∃ S : Set ℝ, S.Nonempty ∧ monotonically_decreasing_on (f a) S) ↔ a < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_condition_l583_58331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_midpoint_l583_58320

-- Define the circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Define a point on the circle
def point_on_unit_circle (M : ℝ × ℝ) : Prop := unit_circle M.1 M.2

-- Define the midpoint of a line segment
def is_midpoint (P M : ℝ × ℝ) : Prop :=
  P.1 = (M.1 + B.1) / 2 ∧ P.2 = (M.2 + B.2) / 2

-- State the theorem
theorem trajectory_of_midpoint :
  ∀ (P : ℝ × ℝ), (∃ M : ℝ × ℝ, point_on_unit_circle M ∧ is_midpoint P M) →
  (2 * P.1 - 3)^2 + 4 * P.2^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_midpoint_l583_58320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_perimeter_l583_58363

/-- Represents a rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculates the area of a rhombus -/
noncomputable def area (r : Rhombus) : ℝ :=
  r.diagonal1 * r.diagonal2 / 2

/-- Calculates the side length of a rhombus given its diagonals -/
noncomputable def sideLength (r : Rhombus) : ℝ :=
  Real.sqrt ((r.diagonal1 / 2) ^ 2 + (r.diagonal2 / 2) ^ 2)

/-- Calculates the perimeter of a rhombus -/
noncomputable def perimeter (r : Rhombus) : ℝ :=
  4 * sideLength r

theorem rhombus_area_and_perimeter (r : Rhombus) 
  (h1 : r.diagonal1 = 6) (h2 : r.diagonal2 = 8) : 
  area r = 24 ∧ perimeter r = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_perimeter_l583_58363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_specific_pyramid_l583_58326

/-- A right triangular pyramid with equilateral triangles as lateral faces -/
structure RightTriangularPyramid where
  lateral_edge_length : ℝ
  is_right : Bool
  has_equilateral_faces : Bool

/-- The height of a right triangular pyramid -/
def pyramid_height (p : RightTriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the height of the specific pyramid -/
theorem height_of_specific_pyramid :
  ∀ (p : RightTriangularPyramid),
    p.lateral_edge_length = 3 ∧
    p.is_right = true ∧
    p.has_equilateral_faces = true →
    pyramid_height p = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_specific_pyramid_l583_58326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l583_58355

variable (α₁ α₂ α₃ ν : ℝ)

noncomputable def left_sum := (1 / (2*ν*α₁ + α₂ + α₃)) + (1 / (2*ν*α₂ + α₃ + α₁)) + (1 / (2*ν*α₃ + α₁ + α₂))

noncomputable def right_sum := (1 / (ν*α₁ + ν*α₂ + α₃)) + (1 / (ν*α₂ + ν*α₃ + α₁)) + (1 / (ν*α₃ + ν*α₁ + α₂))

theorem inequality_proof (hα₁ : α₁ > 0) (hα₂ : α₂ > 0) (hα₃ : α₃ > 0) (hν : ν > 0) :
  left_sum α₁ α₂ α₃ ν > (2*ν / (2*ν + 1)) * right_sum α₁ α₂ α₃ ν := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l583_58355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l583_58388

/-- Calculates the number of units produced today given the average production over the past n days,
    the new average including today, and the number of past days. -/
def units_produced_today (n : ℕ) (past_avg : ℚ) (new_avg : ℚ) : ℕ :=
  let past_total := n * past_avg
  let today_production := (n + 1) * new_avg - past_total
  (today_production.num / today_production.den).natAbs

/-- Proves that given the conditions in the problem, the number of units produced today is 115. -/
theorem problem_solution :
  units_produced_today 12 50 55 = 115 := by
  sorry

#eval units_produced_today 12 50 55

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l583_58388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_children_with_red_flags_l583_58334

theorem percentage_of_children_with_red_flags
  (total_flags : ℕ)
  (h_even : Even total_flags)
  (num_flags : ℕ → ℕ)
  (h_two_flags : ∀ child, num_flags child = 2)
  (percentage_with_blue : ℝ)
  (percentage_with_red : ℝ)
  (percentage_with_both : ℝ)
  (h_blue_percentage : percentage_with_blue = 60)
  (h_both_percentage : percentage_with_both = 5)
  (h_sum_percentages : percentage_with_blue + percentage_with_red + percentage_with_both = 100) :
  percentage_with_red = 40 := by
  sorry

#check percentage_of_children_with_red_flags

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_children_with_red_flags_l583_58334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l583_58361

/-- An ellipse E with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of an ellipse E -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The eccentricity of an ellipse E -/
noncomputable def eccentricity (E : Ellipse) : ℝ :=
  Real.sqrt (1 - E.b^2 / E.a^2)

/-- A line with slope k -/
structure Line where
  k : ℝ
  h_nonzero : k ≠ 0

/-- The perpendicular bisector of a line segment forms a triangle with the coordinate axes -/
noncomputable def perpendicular_bisector_triangle_area (l : Line) (E : Ellipse) : ℝ :=
  sorry -- Definition of the area calculation

theorem ellipse_properties (E : Ellipse) 
  (h_point : ellipse_equation E 0 (Real.sqrt 3))
  (h_ecc : eccentricity E = 1/2) :
  ∃ (l : Line),
    ellipse_equation E = fun x y => x^2/4 + y^2/3 = 1 ∧
    (perpendicular_bisector_triangle_area l E = 1/16 →
      (l.k > -3/2 ∧ l.k < -1/2) ∨ (l.k > 1/2 ∧ l.k < 3/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l583_58361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l583_58358

open Real

noncomputable def f (x : ℝ) := x + x * log x

theorem monotonic_decreasing_interval_f :
  StrictAntiOn f (Set.Ioo (0 : ℝ) (Real.exp (-2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l583_58358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_course_length_is_100_race_theorem_l583_58389

/-- A race between two runners A and B, where A is faster but B gets a head start. -/
structure Race where
  /-- Speed of runner B in meters per second -/
  speed_B : ℝ
  /-- Length of the race course in meters -/
  course_length : ℝ

/-- The time it takes for runner A to complete the race -/
noncomputable def time_A (r : Race) : ℝ := r.course_length / (4 * r.speed_B)

/-- The time it takes for runner B to complete the race -/
noncomputable def time_B (r : Race) : ℝ := (r.course_length - 75) / r.speed_B

/-- The theorem stating that the race course length is 100 meters 
    when both runners finish at the same time -/
theorem race_course_length_is_100 (r : Race) :
  time_A r = time_B r → r.course_length = 100 := by
  sorry

/-- The main theorem proving that when A is 4 times faster than B 
    and gives a 75-meter head start, the race course length for both 
    to finish simultaneously is 100 meters -/
theorem race_theorem :
  ∃ (r : Race), time_A r = time_B r ∧ r.course_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_course_length_is_100_race_theorem_l583_58389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_points_conditions_l583_58356

/-- The time when two objects meet, one thrown upward from point A with initial velocity v₀,
    and one dropped from point B which is a meters above A, under constant gravitational acceleration g. -/
noncomputable def time_of_encounter (v₀ a g : ℝ) : ℝ := a / v₀

/-- The distance traveled by the object thrown upward from point A. -/
noncomputable def distance_from_A (v₀ a g t : ℝ) : ℝ := v₀ * t - (1/2) * g * t^2

/-- The distance traveled by the object dropped from point B. -/
noncomputable def distance_from_B (g t : ℝ) : ℝ := (1/2) * g * t^2

/-- Theorem stating the conditions for the meeting points of the two objects. -/
theorem meeting_points_conditions (v₀ a g : ℝ) (h₁ : v₀ > 0) (h₂ : g > 0) :
  (a < v₀^2 / g → distance_from_A v₀ a g (time_of_encounter v₀ a g) > 0) ∧
  (a = 2 * v₀^2 / g → distance_from_A v₀ a g (time_of_encounter v₀ a g) = 0) ∧
  (a > 2 * v₀^2 / g → distance_from_A v₀ a g (time_of_encounter v₀ a g) < 0) :=
by
  sorry

#check meeting_points_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_points_conditions_l583_58356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_is_closed_interval_one_to_twentyeight_l583_58303

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1)^2 * x^(m^2 - 4*m + 2)
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := 2^x - t

-- State the theorem
theorem t_range_is_closed_interval_one_to_twentyeight :
  ∀ m : ℝ,
  (∀ x > 0, Monotone (f m)) →
  (∃ t : ℝ, ∀ x₁ ∈ Set.Icc 1 6, ∃ x₂ ∈ Set.Icc 1 6, f m x₁ = g t x₂) →
  ∃ t : ℝ, t ∈ Set.Icc 1 28 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_is_closed_interval_one_to_twentyeight_l583_58303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_one_l583_58392

/-- Given a system of equations and two distinct points, prove that the distance
    from the origin to the line passing through these points is 1. -/
theorem distance_to_line_is_one (a b θ : ℝ) (ha : a^2 * Real.sin θ + a * Real.cos θ - 1 = 0)
    (hb : b^2 * Real.sin θ + b * Real.cos θ - 1 = 0) (hab : a ≠ b) :
  let line := {p : ℝ × ℝ | p.1 * Real.cos θ + p.2 * Real.sin θ = 1}
  let origin : ℝ × ℝ := (0, 0)
  let distance := fun (p : ℝ × ℝ) ↦ |p.1 * Real.cos θ + p.2 * Real.sin θ - 1| / Real.sqrt (Real.cos θ^2 + Real.sin θ^2)
  distance origin = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_one_l583_58392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_one_l583_58323

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3^x - 1

-- State the theorem
theorem inverse_f_at_one (h : Function.Bijective f) : 
  (Function.invFun f) 1 = Real.log 2 / Real.log 3 := by
  sorry

#check inverse_f_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_one_l583_58323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rosette_area_theorem_l583_58339

/-- The area of a rosette formed by inscribing semicircles on the sides of a rhombus -/
noncomputable def rosette_area (a b : ℝ) : ℝ :=
  (Real.pi * (a^2 + b^2) - 4 * a * b) / 8

/-- Theorem: The area of a rosette formed by inscribing semicircles on the sides of a rhombus
    with diagonals a and b is equal to (π(a² + b²) - 4ab) / 8 -/
theorem rosette_area_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  rosette_area a b = (Real.pi * (a^2 + b^2) - 4 * a * b) / 8 := by
  sorry

#check rosette_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rosette_area_theorem_l583_58339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circumscribed_circle_relation_l583_58377

/-- A triangle is inscribed in a circle with radius R -/
def inscribed_in_circle (triangle : Set (ℝ × ℝ)) (R : ℝ) : Prop :=
sorry

/-- A triangle is circumscribed about a circle with radius r -/
def circumscribed_about_circle (triangle : Set (ℝ × ℝ)) (r : ℝ) : Prop :=
sorry

/-- The distance between the centers of two circles with radii R and r -/
def distance_between_centers (R r : ℝ) : ℝ :=
sorry

/-- Given two circles with radii R and r, separated by a distance d, 
    if a triangle can be inscribed in the first circle and circumscribed about the second circle, 
    then d² = R² - 2Rr -/
theorem inscribed_circumscribed_circle_relation (R r d : ℝ) 
  (h : ∃ (triangle : Set (ℝ × ℝ)), 
    inscribed_in_circle triangle R ∧ 
    circumscribed_about_circle triangle r ∧ 
    distance_between_centers R r = d) : 
  d^2 = R^2 - 2*R*r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circumscribed_circle_relation_l583_58377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_range_implies_interval_domain_implies_solution_set_l583_58348

open Set
open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

-- Theorem 1
theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Ioo 0 a, f a x ≥ 4) ∧ (∃ x ∈ Ioo 0 a, f a x = 4) → a = 4 := by sorry

-- Theorem 2
theorem range_implies_interval (a : ℝ) (A : Set ℝ) :
  a = 4 ∧ (∀ x ∈ A, f a x ∈ Icc 4 5) ∧ (∀ y ∈ Icc 4 5, ∃ x ∈ A, f a x = y) → A = Icc 1 4 := by sorry

-- Theorem 3
theorem domain_implies_solution_set (a : ℝ) :
  (∀ x ≥ 2, f a x ≥ f a 2) →
  (f a (a^2 - a) ≥ f a (2*a + 4) ↔ (a ≥ 4 ∨ a = -1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_range_implies_interval_domain_implies_solution_set_l583_58348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l583_58345

/-- The intersection point of a line with the xz-plane --/
noncomputable def intersection_point (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let t : ℝ := (3 : ℝ) / 4
  (p.1 + t * (q.1 - p.1), 0, p.2.2 + t * (q.2.2 - p.2.2))

/-- Theorem stating that the calculated intersection point is correct --/
theorem intersection_point_correct (p q : ℝ × ℝ × ℝ) 
  (hp : p = (1, 3, 2)) (hq : q = (4, -1, 7)) :
  intersection_point p q = (13/4, 0, 23/4) ∧ 
  ∃ (t : ℝ), intersection_point p q = 
    (p.1 + t * (q.1 - p.1), 0, p.2.2 + t * (q.2.2 - p.2.2)) :=
by sorry

#check intersection_point_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l583_58345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_due_in_nine_months_l583_58346

/-- Calculates the time in months until a bill is due, given its face value, true discount, and annual interest rate. -/
noncomputable def time_until_due (face_value : ℝ) (true_discount : ℝ) (annual_interest_rate : ℝ) : ℝ :=
  let present_value := face_value - true_discount
  let r := annual_interest_rate / 100
  let time_in_years := (face_value / present_value - 1) / r
  time_in_years * 12

/-- Theorem stating that for a bill with face value 2240 Rs., true discount 240 Rs., and 16% annual interest rate, the time until due is 9 months. -/
theorem bill_due_in_nine_months :
  time_until_due 2240 240 16 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_due_in_nine_months_l583_58346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_conditions_imply_sum_l583_58304

noncomputable def equation1 (a b x : ℝ) : ℝ := (x + a) * (x + b) * (x + 10) / ((x + 4) ^ 2)
noncomputable def equation2 (a b x : ℝ) : ℝ := (x + 2*a) * (x + 4) * (x + 7) / ((x + b) * (x + 10))

theorem root_conditions_imply_sum (a b : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    equation1 a b x = 0 ∧ equation1 a b y = 0 ∧ equation1 a b z = 0 ∧
    ∀ w, equation1 a b w = 0 → w = x ∨ w = y ∨ w = z) →
  (∃! x, equation2 a b x = 0) →
  100 * a + b = 207 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_conditions_imply_sum_l583_58304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l583_58341

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)

-- State the theorem
theorem f_properties :
  -- f is defined on (-1, 1)
  (∀ x : ℝ, -1 < x ∧ x < 1 → f x ∈ Set.univ) ∧
  -- f is an odd function
  (∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x) ∧
  -- f is increasing on (0, 1)
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l583_58341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_m_eq_half_2009_l583_58322

def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2008}

noncomputable def median (A : Finset ℕ) : ℚ :=
  sorry

noncomputable def m (A : Finset ℕ) : ℚ :=
  if A.card % 2 = 1 then
    median A
  else
    (median A + median A) / 2

noncomputable def average_m : ℚ :=
  sorry

theorem average_m_eq_half_2009 : average_m = 2009 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_m_eq_half_2009_l583_58322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l583_58371

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) * Real.cos (2 * x)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l583_58371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_animals_on_ranch_l583_58342

/-- Represents the number of ponies on the ranch -/
def num_ponies : ℕ := sorry

/-- Represents the number of horses on the ranch -/
def num_horses : ℕ := sorry

/-- The number of ponies with horseshoes -/
def ponies_with_horseshoes : ℕ := (5 * num_ponies) / 6

/-- The number of Icelandic ponies with horseshoes -/
def icelandic_ponies_with_horseshoes : ℕ := (2 * ponies_with_horseshoes) / 3

/-- The total number of animals on the ranch -/
def total_animals : ℕ := num_ponies + num_horses

theorem minimum_animals_on_ranch :
  (ponies_with_horseshoes = (5 * num_ponies) / 6) →
  (icelandic_ponies_with_horseshoes = (2 * ponies_with_horseshoes) / 3) →
  (num_horses = num_ponies + 4) →
  (∃ (n : ℕ), num_ponies = 18 * n ∧ n > 0) →
  (total_animals ≥ 40 ∧ total_animals = 40 → 
    ∀ m, total_animals ≤ m → m = 40) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_animals_on_ranch_l583_58342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_with_real_roots_l583_58375

open Real

-- Define the equation
noncomputable def equation (a : ℕ) (x : ℝ) : ℝ :=
  (cos (π * (a - x)))^2 - 2 * cos (π * (a - x)) +
  cos (3 * π * x / (2 * a)) * cos (π * x / (2 * a) + π / 3) + 2

-- Define the property of having real roots
def has_real_roots (a : ℕ) : Prop :=
  ∃ x : ℝ, equation a x = 0

-- State the theorem
theorem smallest_a_with_real_roots :
  (∀ a : ℕ, 0 < a → a < 6 → ¬(has_real_roots a)) ∧ has_real_roots 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_with_real_roots_l583_58375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_two_digit_numbers_divisible_by_four_l583_58330

theorem count_two_digit_numbers_divisible_by_four : 
  (Finset.filter (fun n : Fin 100 => n % 4 = 0) Finset.univ).card = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_two_digit_numbers_divisible_by_four_l583_58330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_exponential_power_sum_l583_58374

theorem logarithm_exponential_power_sum : 
  Real.log 4 / Real.log (Real.sqrt 2) + Real.exp (Real.log 3) + (0.125 : ℝ)^(-(2/3 : ℝ)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_exponential_power_sum_l583_58374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l583_58391

variable {V : Type*} [NormedAddCommGroup V] [Module ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b ∨ b = k • a

theorem vector_relationships (a b : V) : 
  (¬ (parallel a b → a = b)) ∧ 
  (¬ (norm a = norm b → a = b)) ∧ 
  (¬ (norm a = norm b → parallel a b)) ∧ 
  (a = b → norm a = norm b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l583_58391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l583_58359

/-- Represents the distribution of scores in a statistics test -/
structure ScoreDistribution where
  score60 : ℝ  -- Percentage of students scoring 60
  score75 : ℝ  -- Percentage of students scoring 75
  score85 : ℝ  -- Percentage of students scoring 85
  score95 : ℝ  -- Percentage of students scoring 95

/-- Calculates the median score given a score distribution -/
def median (_ : ScoreDistribution) : ℝ := 85

/-- Calculates the mean score given a score distribution -/
def mean (d : ScoreDistribution) : ℝ := 
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 95 * d.score95

/-- Theorem stating the difference between median and mean scores -/
theorem median_mean_difference (d : ScoreDistribution) 
  (h1 : d.score60 = 0.2)
  (h2 : d.score75 = 0.2)
  (h3 : d.score85 = 0.4)
  (h4 : d.score95 = 0.2) :
  median d - mean d = 5 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l583_58359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_slope_range_l583_58318

-- Define the circle and points
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8
def C : ℝ × ℝ := (-1, 0)
def A : ℝ × ℝ := (1, 0)

-- Define the conditions
def conditions (P Q M : ℝ × ℝ) : Prop :=
  circle_eq P.1 P.2 ∧
  ∃ t : ℝ, Q = C + t • (P - C) ∧
  ∃ s : ℝ, M = A + s • (P - A) ∧
  (M - Q) • (A - P) = 0 ∧
  A - P = 2 • (A - M)

-- Define the trajectory of Q
def trajectory (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the tangent line and intersection conditions
def tangent_and_intersection (k b : ℝ) (F H : ℝ × ℝ) : Prop :=
  (k^2 + 1 = b^2) ∧
  (trajectory F.1 F.2 ∧ F.2 = k * F.1 + b) ∧
  (trajectory H.1 H.2 ∧ H.2 = k * H.1 + b) ∧
  F ≠ H ∧
  3/4 ≤ (F.1 * H.1 + F.2 * H.2) ∧ (F.1 * H.1 + F.2 * H.2) ≤ 4/5

-- Main theorem
theorem trajectory_and_slope_range :
  (∀ P Q M, conditions P Q M → trajectory Q.1 Q.2) ∧
  (∀ k, (∃ b F H, tangent_and_intersection k b F H) ↔ 
    (-Real.sqrt 2 / 2 ≤ k ∧ k ≤ -Real.sqrt 3 / 3) ∨ 
    (Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 2 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_slope_range_l583_58318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clever_points_exist_l583_58310

-- Define the concept of a "clever point"
def has_clever_point (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = (deriv f) x

-- Define the four functions
noncomputable def f₁ : ℝ → ℝ := λ x ↦ x^2
noncomputable def f₂ : ℝ → ℝ := λ x ↦ Real.exp (-x)
noncomputable def f₃ : ℝ → ℝ := λ x ↦ Real.log x
noncomputable def f₄ : ℝ → ℝ := λ x ↦ Real.tan x

-- State the theorem
theorem clever_points_exist :
  has_clever_point f₁ ∧ 
  ¬ has_clever_point f₂ ∧ 
  has_clever_point f₃ ∧ 
  ¬ has_clever_point f₄ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clever_points_exist_l583_58310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_properties_l583_58369

theorem alpha_beta_properties (α β : Real) 
  (h1 : 0 < α) (h2 : α < π / 2) (h3 : π / 2 < β) (h4 : β < π)
  (h5 : Real.tan (α / 2) = 1 / 2)
  (h6 : Real.cos (β - α) = Real.sqrt 2 / 10) :
  (Real.cos (2 * α) / (Real.sqrt 2 * Real.cos (π / 4 + α) * Real.sin α) = 7 / 4) ∧
  (β = 3 * π / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_properties_l583_58369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_in_unit_squares_l583_58354

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

/-- The statement to be proved -/
theorem area_of_triangle_in_unit_squares :
  ∃ (A B C : ℝ × ℝ),
    (∃ (x y : ℝ), A = (x, y) ∧ B = (x+1, y+1)) ∧  -- A and B form a diagonal of a unit square
    (∃ (x y : ℝ), C = (x, y) ∧ 
      ((x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) ∨ (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2))) ∧  -- C is a vertex of an adjacent unit square
    triangle_area A B C = 1/2 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_in_unit_squares_l583_58354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_eq_neg_cos_l583_58327

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => Real.cos x
  | (n + 1) => λ x => deriv (f n) x

-- State the theorem
theorem f_2010_eq_neg_cos : f 2010 = λ x => -Real.cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_eq_neg_cos_l583_58327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l583_58382

theorem expression_value : (3^2 + 3^1 + 3^0) / ((1/3) + (1/9) + (1/27)) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l583_58382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l583_58370

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_properties (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n = 10)
  (h2 : geometric_sequence a) :
  (∃ q : ℝ, q = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q) ∧
  (∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q) ∧
  (∃ S : ℕ → ℝ, (∀ n : ℕ, S n = (a 0) * (1 - (1 : ℝ)^n) / (1 - 1)) ∧
    geometric_sequence (λ n ↦ S n)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l583_58370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colins_first_mile_time_l583_58324

/-- Proves that Colin's first mile time was 6 minutes -/
theorem colins_first_mile_time :
  ∀ (first_mile : ℝ),
  (first_mile + 2 * 5 + 4) / 4 = 5 →
  first_mile = 6 := by
  intro first_mile hypothesis
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colins_first_mile_time_l583_58324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_hours_equality_l583_58365

/-- Proves that t = 5 given the conditions of the problem -/
theorem work_hours_equality (t : ℝ) : 
  (t - 4 > 0) →  -- My working hours are positive
  (t - 2 > 0) →  -- Sarah's working hours are positive
  (t - 4) * (3*t - 7) = (t - 2) * (t + 1) →  -- Both earned the same amount
  t = 5 := by
  intro h1 h2 h3
  -- The proof steps would go here
  sorry

#check work_hours_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_hours_equality_l583_58365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_x_l583_58367

open MeasureTheory Interval Real Set

theorem integral_sqrt_plus_x : ∫ x in (Icc 0 1), (Real.sqrt x + x) = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_x_l583_58367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_cat_max_distance_l583_58309

/-- The maximum distance from any point on a circle to the origin, 
    given the circle's center and radius -/
theorem max_distance_to_origin (center_x center_y radius : ℝ) :
  let center_distance := Real.sqrt (center_x^2 + center_y^2)
  let max_distance := radius + center_distance
  ∀ point : ℝ × ℝ, 
    (point.1 - center_x)^2 + (point.2 - center_y)^2 = radius^2 →
    (point.1^2 + point.2^2) ≤ max_distance^2 :=
by sorry

/-- The specific case for the cat problem -/
theorem cat_max_distance :
  let center : ℝ × ℝ := (5, -2)
  let radius : ℝ := 15
  let max_distance := radius + Real.sqrt (center.1^2 + center.2^2)
  max_distance = 15 + Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_cat_max_distance_l583_58309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_downstream_l583_58368

/-- Calculates the distance travelled downstream by a boat -/
noncomputable def distance_downstream (boat_speed : ℝ) (current_rate : ℝ) (time_minutes : ℝ) : ℝ :=
  (boat_speed + current_rate) * (time_minutes / 60)

/-- Theorem stating the distance travelled downstream by a boat -/
theorem boat_distance_downstream :
  distance_downstream 15 3 12 = 3.6 := by
  -- Unfold the definition of distance_downstream
  unfold distance_downstream
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_downstream_l583_58368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_and_optimal_investment_unique_optimal_investment_l583_58399

/-- The profit function for a company's technical reform investment. -/
noncomputable def profit_function (m : ℝ) : ℝ := 28 - m - 16 / (m + 1)

/-- The theorem stating the maximum profit and optimal investment. -/
theorem max_profit_and_optimal_investment :
  (∃ (m : ℝ), m ≥ 0 ∧ profit_function m = 21 ∧
    ∀ (n : ℝ), n ≥ 0 → profit_function n ≤ profit_function m) ∧
  (∃ (m : ℝ), m = 3 ∧ profit_function m = 21 ∧
    ∀ (n : ℝ), n ≥ 0 → profit_function n ≤ profit_function m) :=
by sorry

/-- The theorem proving the uniqueness of the optimal investment. -/
theorem unique_optimal_investment (m n : ℝ) :
  m ≥ 0 → n ≥ 0 →
  profit_function m = 21 →
  (∀ (x : ℝ), x ≥ 0 → profit_function x ≤ profit_function m) →
  profit_function n = 21 →
  (∀ (x : ℝ), x ≥ 0 → profit_function x ≤ profit_function n) →
  m = n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_and_optimal_investment_unique_optimal_investment_l583_58399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l583_58335

noncomputable def fill_time (time_a : ℝ) (speed_ratio : ℝ) : ℝ :=
  1 / (1 / time_a + speed_ratio / time_a)

theorem tank_fill_time (time_a : ℝ) (speed_ratio : ℝ) 
  (h1 : time_a = 56) 
  (h2 : speed_ratio = 7) : 
  fill_time time_a speed_ratio = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l583_58335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_sampling_l583_58302

/-- Represents a summer camp with a total number of students and camp divisions. -/
structure SummerCamp where
  total_students : Nat
  camp_i_end : Nat
  camp_ii_end : Nat
  camp_iii_end : Nat

/-- Represents a systematic sampling method. -/
structure SystematicSampling where
  sample_size : Nat
  start_number : Nat
  step : Nat

/-- Calculates the number of students selected from a specific range. -/
def students_selected (start : Nat) (end_ : Nat) (sampling : SystematicSampling) : Nat :=
  sorry

/-- Theorem stating the number of students selected from each camp. -/
theorem summer_camp_sampling (camp : SummerCamp) (sampling : SystematicSampling) : 
  camp.total_students = 600 →
  camp.camp_i_end = 200 →
  camp.camp_ii_end = 500 →
  camp.camp_iii_end = 600 →
  sampling.sample_size = 50 →
  sampling.start_number = 3 →
  sampling.step = 12 →
  (students_selected 1 camp.camp_i_end sampling = 17 ∧
   students_selected (camp.camp_i_end + 1) camp.camp_ii_end sampling = 25 ∧
   students_selected (camp.camp_ii_end + 1) camp.camp_iii_end sampling = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_sampling_l583_58302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l583_58332

/-- Given a triangle ABC with angles A, B, C opposite sides a, b, c respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Vectors m and n defined based on triangle angles -/
noncomputable def m (t : Triangle) : Fin 2 → ℝ
  | 0 => Real.sin t.A
  | 1 => Real.sin t.B
  | _ => 0  -- This case is added to satisfy the type checker

noncomputable def n (t : Triangle) : Fin 2 → ℝ
  | 0 => Real.cos t.B
  | 1 => Real.cos t.A
  | _ => 0  -- This case is added to satisfy the type checker

/-- Dot product of vectors -/
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

/-- Theorem stating the main results -/
theorem triangle_properties (t : Triangle) 
  (h1 : dot_product (m t) (n t) = Real.sin (2 * t.C))
  (h2 : t.a + t.b = 2)
  : t.C = π / 3 ∧ ∃ (CD : ℝ), CD ≥ Real.sqrt 3 / 2 ∧ 
    ∀ (CD' : ℝ), CD' ≥ Real.sqrt 3 / 2 → CD ≤ CD' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l583_58332
