import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_F₂AB_eq_four_thirds_l770_77056

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := (1, 0)

/-- The inclination angle of chord AB -/
noncomputable def θ : ℝ := Real.pi / 4

/-- Chord AB passes through F₁ -/
def chord_passes_F₁ (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = F₁ + t • (B - F₁)

/-- The area of triangle F₂AB -/
noncomputable def area_F₂AB (A B : ℝ × ℝ) : ℝ :=
  abs ((A.1 - F₂.1) * (B.2 - F₂.2) - (B.1 - F₂.1) * (A.2 - F₂.2)) / 2

/-- The main theorem -/
theorem area_F₂AB_eq_four_thirds :
  ∀ A B : ℝ × ℝ,
  ellipse A.1 A.2 → ellipse B.1 B.2 →
  chord_passes_F₁ A B →
  (B.2 - A.2) / (B.1 - A.1) = Real.tan θ →
  area_F₂AB A B = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_F₂AB_eq_four_thirds_l770_77056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_from_bisector_l770_77022

/-- Given a triangle with sides a and b enclosing an angle α, and angle bisector l,
    the angle α is equal to 2 arccos((l(a + b))/(2ab)). -/
theorem angle_from_bisector (a b l : ℝ) (ha : a > 0) (hb : b > 0) (hl : l > 0) :
  ∃ α : ℝ, α > 0 ∧ α < π ∧ l = (2 * a * b * Real.cos (α / 2)) / (a + b) →
    α = 2 * Real.arccos ((l * (a + b)) / (2 * a * b)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_from_bisector_l770_77022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_point_l770_77064

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

-- Define the point A
def A : ℝ × ℝ := (-2, 2)

-- Define the left focus F
def F : ℝ × ℝ := (-3, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the objective function to be minimized
noncomputable def objective (B : ℝ × ℝ) : ℝ :=
  distance A B + (5/3) * distance B F

-- State the theorem
theorem minimal_point :
  ∃ (B : ℝ × ℝ), is_on_ellipse B.1 B.2 ∧
    (∀ (C : ℝ × ℝ), is_on_ellipse C.1 C.2 → objective B ≤ objective C) ∧
    B = (-5 * Real.sqrt 3 / 2, 2) := by
  sorry

#check minimal_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_point_l770_77064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l770_77089

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

noncomputable def geometric_last_term (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r^(n - 1)

theorem geometric_series_sum :
  ∃ n : ℕ, 
    geometric_last_term 1 3 n = 19683 ∧
    geometric_sum 1 3 n = 29524 := by
  -- Proof goes here
  sorry

#eval Nat.log 3 19683 + 1  -- This should evaluate to 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l770_77089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l770_77003

-- Define the curves and points
noncomputable def C1 (θ : ℝ) : ℝ := 4 * Real.cos θ

def C2 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def C3 (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def ray_angle : ℝ := Real.pi / 3

noncomputable def point_A : ℝ := C1 ray_angle

def point_B : ℝ := 1  -- Since C3 in polar form is ρ = 1

-- State the theorem
theorem distance_AB : |point_A - point_B| = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l770_77003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_26_5_l770_77066

/-- Represents a rectangular plot with fencing cost -/
structure RectangularPlot where
  breadth : ℚ
  length : ℚ
  total_fencing_cost : ℚ
  length_breadth_relation : length = breadth + 50
  length_value : length = 75

/-- Calculates the cost of fencing per meter for a given rectangular plot -/
def fencing_cost_per_meter (plot : RectangularPlot) : ℚ :=
  plot.total_fencing_cost / (2 * (plot.length + plot.breadth))

/-- Theorem stating that the fencing cost per meter is 26.5 for the given conditions -/
theorem fencing_cost_is_26_5 (plot : RectangularPlot) 
  (h : plot.total_fencing_cost = 5300) : 
  fencing_cost_per_meter plot = 26.5 := by
  sorry

def example_plot : RectangularPlot := {
  breadth := 25,
  length := 75,
  total_fencing_cost := 5300,
  length_breadth_relation := by norm_num,
  length_value := by rfl
}

#eval fencing_cost_per_meter example_plot

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_26_5_l770_77066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b4e_hex_to_decimal_l770_77051

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3 | '4' => 4
  | '5' => 5 | '6' => 6 | '7' => 7 | '8' => 8 | '9' => 9
  | 'A' => 10 | 'B' => 11 | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_to_decimal (s : String) : ℕ :=
  s.toList.reverse.enum.foldl
    (fun acc (i, c) => acc + (hex_to_dec c) * (16 ^ i))
    0

/-- The theorem stating that B4E₁₆ is equal to 2894 in decimal -/
theorem b4e_hex_to_decimal :
  hex_to_decimal "B4E" = 2894 := by
  sorry

#eval hex_to_decimal "B4E"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b4e_hex_to_decimal_l770_77051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_angle_theorem_l770_77021

/-- A point with integer coordinates --/
def LatticePoint (p : ℚ × ℚ) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

/-- A triangle with vertices A, B, and C --/
structure Triangle :=
  (A B C : ℚ × ℚ)

/-- A point P is inside a triangle ABC --/
noncomputable def InsideTriangle (P : ℚ × ℚ) (T : Triangle) : Prop :=
  sorry  -- Definition of a point being inside a triangle

/-- The angle between three points --/
noncomputable def Angle (P Q R : ℚ × ℚ) : ℝ :=
  sorry  -- Definition of angle between three points

/-- A real number is a rational multiple of π --/
def IsRationalMultipleOfPi (x : ℝ) : Prop :=
  ∃ (q : ℚ), x = q * Real.pi

theorem lattice_point_angle_theorem (T : Triangle) (P : ℚ × ℚ) 
  (h1 : LatticePoint T.A ∧ LatticePoint T.B ∧ LatticePoint T.C)
  (h2 : LatticePoint P)
  (h3 : InsideTriangle P T) :
  ¬(IsRationalMultipleOfPi (Angle P T.A T.B) ∧ 
    IsRationalMultipleOfPi (Angle P T.B T.C) ∧ 
    IsRationalMultipleOfPi (Angle P T.C T.A)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_angle_theorem_l770_77021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_theorem_l770_77027

/-- The shortest distance from a point on the parabola y² = 2x to the line x - y + 2 = 0 -/
noncomputable def shortest_distance : ℝ := (3 * Real.sqrt 2) / 4

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := x - y + 2 = 0

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 2*x

/-- Theorem stating the shortest distance property -/
theorem shortest_distance_theorem :
  ∀ (x y : ℝ), parabola_equation x y →
  (∀ (x' y' : ℝ), parabola_equation x' y' →
    Real.sqrt ((x - x')^2 + (y - y')^2) ≥ shortest_distance) ∧
  (∃ (x₀ y₀ : ℝ), parabola_equation x₀ y₀ ∧
    (∃ (x' y' : ℝ), line_equation x' y' ∧
      Real.sqrt ((x₀ - x')^2 + (y₀ - y')^2) = shortest_distance)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_theorem_l770_77027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l770_77049

/-- Represents an investment venture with three participants -/
structure Venture where
  investmentA : ℚ
  investmentB : ℚ
  investmentC : ℚ
  interestRateA : ℚ
  interestRateB : ℚ
  interestRateC : ℚ
deriving Repr

/-- Calculates the total profit share for a given venture -/
def totalProfitShare (v : Venture) : ℚ :=
  v.investmentA * v.interestRateA +
  v.investmentB * v.interestRateB +
  v.investmentC * v.interestRateC

/-- Calculates C's profit share for a given venture -/
def profitShareC (v : Venture) : ℚ :=
  v.investmentC * v.interestRateC

theorem total_profit_calculation (v1 v2 : Venture) 
  (h : profitShareC v1 + profitShareC v2 = 5550) :
  totalProfitShare v1 + totalProfitShare v2 = 5940 :=
by
  sorry

def venture1 : Venture := {
  investmentA := 5000
  investmentB := 15000
  investmentC := 30000
  interestRateA := 5 / 100
  interestRateB := 10 / 100
  interestRateC := 15 / 100
}

def venture2 : Venture := {
  investmentA := 6000
  investmentB := 10000
  investmentC := 24000
  interestRateA := 12 / 100
  interestRateB := 8 / 100
  interestRateC := 6 / 100
}

#eval venture1
#eval venture2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_l770_77049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l770_77012

theorem triangle_angle_B (a b : ℝ) (A B : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  B = π / 3 ∨ B = 2 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l770_77012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_prove_single_trial_probability_l770_77030

/-- The probability of an event occurring in a single trial, given that:
  1) There are 4 independent trials
  2) The probability of the event occurring remains constant for each trial
  3) The probability of the event occurring at least once in 4 trials is 65/81 -/
noncomputable def probability_single_trial : ℝ := 1 / 3

/-- The number of independent trials -/
def num_trials : ℕ := 4

/-- The probability of the event occurring at least once in 4 trials -/
noncomputable def probability_at_least_once : ℝ := 65 / 81

theorem probability_calculation : 
  (1 - probability_single_trial) ^ num_trials = 1 - probability_at_least_once :=
by sorry

theorem prove_single_trial_probability : 
  probability_single_trial = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_prove_single_trial_probability_l770_77030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_k_l770_77028

-- Define the function k(x)
noncomputable def k (x : ℝ) : ℝ := 1 / (x + 6) + 1 / (x^2 + 2*x + 9) + 1 / (x^3 - 27)

-- Define the domain of k(x)
def domain_k : Set ℝ := {x | x < -6 ∨ (-6 < x ∧ x < 3) ∨ 3 < x}

-- Theorem stating that the domain of k is correct
theorem domain_of_k : 
  ∀ x : ℝ, k x ≠ 0 ↔ x ∈ domain_k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_k_l770_77028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l770_77014

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 2 * y = 4) :
  (∀ a b : ℝ, a + 2 * b = 4 → (2 : ℝ)^x + (4 : ℝ)^y ≤ (2 : ℝ)^a + (4 : ℝ)^b) ∧ 
  (∃ x y : ℝ, x + 2 * y = 4 ∧ (2 : ℝ)^x + (4 : ℝ)^y = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l770_77014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l770_77044

/-- The function g(x) = 4 / (3x^8 - 5x + 6) -/
noncomputable def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 5 * x + 6)

/-- g is neither even nor odd -/
theorem g_neither_even_nor_odd :
  ¬(∀ x, g (-x) = g x) ∧ ¬(∀ x, g (-x) = -g x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l770_77044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l770_77061

/-- Given circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + 45 = 0

/-- Given line equation -/
def given_line (x y : ℝ) : Prop := y = 3*x

/-- General form of parallel lines -/
def parallel_line (x y c : ℝ) : Prop := y = 3*x + c

/-- Distance from point (x₀, y₀) to line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  (|A*x₀ + B*y₀ + C|) / Real.sqrt (A^2 + B^2)

/-- Theorem: The only lines parallel to y = 3x and tangent to the given circle
    are y = 3x + 4√10 - 9 and y = 3x - 4√10 - 9 -/
theorem tangent_lines_to_circle :
  ∀ (c : ℝ), (∀ (x y : ℝ), circle_eq x y → ¬parallel_line x y c) ∨
             (c = 4 * Real.sqrt 10 - 9 ∨ c = -4 * Real.sqrt 10 - 9) := by
  sorry

#check tangent_lines_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l770_77061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_sumOfSquares_relation_l770_77078

/-- Represents a geometric sequence with n terms -/
structure GeometricSequence where
  n : ℕ
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- The product of n terms in a geometric sequence -/
noncomputable def product (seq : GeometricSequence) : ℝ :=
  seq.a^seq.n * seq.r^(seq.n * (seq.n - 1) / 2)

/-- The sum of n terms in a geometric sequence -/
noncomputable def sum (seq : GeometricSequence) : ℝ :=
  seq.a * (1 - seq.r^seq.n) / (1 - seq.r)

/-- The sum of squares of n terms in a geometric sequence -/
noncomputable def sumOfSquares (seq : GeometricSequence) : ℝ :=
  seq.a^2 * (1 - seq.r^(2 * seq.n)) / (1 - seq.r^2)

/-- Theorem stating the relationship between product, sum, and sum of squares -/
theorem product_sum_sumOfSquares_relation (seq : GeometricSequence) :
  product seq = (sumOfSquares seq / sum seq)^((seq.n - 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_sumOfSquares_relation_l770_77078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_numbers_l770_77035

def five_fives (n : ℕ) : Prop :=
  ∃ (a b c d e : ℚ),
    (a = 5 ∨ a = 5⁻¹) ∧
    (b = 5 ∨ b = 5⁻¹) ∧
    (c = 5 ∨ c = 5⁻¹) ∧
    (d = 5 ∨ d = 5⁻¹) ∧
    (e = 5 ∨ e = 5⁻¹) ∧
    (∃ (op₁ op₂ op₃ op₄ : ℚ → ℚ → ℚ),
      (∀ x y, op₁ x y = x + y ∨ op₁ x y = x - y ∨ op₁ x y = x * y ∨ op₁ x y = x / y) ∧
      (∀ x y, op₂ x y = x + y ∨ op₂ x y = x - y ∨ op₂ x y = x * y ∨ op₂ x y = x / y) ∧
      (∀ x y, op₃ x y = x + y ∨ op₃ x y = x - y ∨ op₃ x y = x * y ∨ op₃ x y = x / y) ∧
      (∀ x y, op₄ x y = x + y ∨ op₄ x y = x - y ∨ op₄ x y = x * y ∨ op₄ x y = x / y) ∧
      n = Nat.floor (op₁ (op₂ (op₃ a b) c) (op₄ d e)))

theorem construct_numbers : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 17 → five_fives n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_numbers_l770_77035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_100_triangles_l770_77080

/-- A polygon that can be divided into a specific number of rectangles -/
structure RectangularPolygon where
  /-- The number of rectangles the polygon can be divided into -/
  num_rectangles : ℕ
  /-- The polygon cannot be divided into one fewer rectangle -/
  not_fewer : num_rectangles > 0

/-- Represents a triangle -/
structure Triangle

/-- Represents a partition of a polygon into triangles -/
def PolygonPartition (P : RectangularPolygon) (triangles : Fin n → Triangle) : Prop :=
  sorry -- Definition of polygon partition

/-- Theorem: A polygon that can be divided into 100 rectangles but not 99 cannot be divided into 100 triangles -/
theorem no_100_triangles (P : RectangularPolygon) (h : P.num_rectangles = 100) :
  ¬ ∃ (t : ℕ), t = 100 ∧ (∃ (triangles : Fin t → Triangle), PolygonPartition P triangles) :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_100_triangles_l770_77080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_y_negative_l770_77025

-- Define the lines
def line1 (a x y : ℝ) : Prop := 2 * x - a * y + 2 = 0
def line2 (x y : ℝ) : Prop := x + y = 0

-- Define the intersection point
noncomputable def intersection (a : ℝ) : ℝ × ℝ :=
  (2 / (2 + a), -2 / (2 + a))

-- State the theorem
theorem intersection_y_negative (a : ℝ) :
  (intersection a).2 < 0 ↔ a < -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_y_negative_l770_77025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l770_77001

-- Define the triangle ABC and its properties
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ  -- Circumcenter

-- Define the vectors
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define the dot product
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a vector
noncomputable def mag (v : ℝ × ℝ) : ℝ := Real.sqrt (dot v v)

-- Theorem for part 1
theorem part1 (t : Triangle) (x y : ℝ) 
    (h1 : vec t.A t.O = (x * (t.B.1 - t.A.1) + y * (t.C.1 - t.A.1), 
                         x * (t.B.2 - t.A.2) + y * (t.C.2 - t.A.2)))
    (h2 : dot (vec t.A t.B) (vec t.A t.C) = -1)
    (h3 : mag (vec t.A t.B) = 1)
    (h4 : mag (vec t.A t.C) = 2) :
  x = 4/3 ∧ y = 5/6 := by sorry

-- Theorem for part 2
theorem part2 (t : Triangle) (x y : ℝ)
    (h1 : vec t.A t.O = (x * (t.B.1 - t.A.1) + y * (t.C.1 - t.A.1), 
                         x * (t.B.2 - t.A.2) + y * (t.C.2 - t.A.2)))
    (h2 : dot (vec t.A t.B) (vec t.A t.C) = 1/3 * mag (vec t.A t.B) * mag (vec t.A t.C)) :
  x + y ≤ 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l770_77001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l770_77010

noncomputable def greatest_even_le (y : ℝ) : ℤ :=
  2 * ⌊y / 2⌋

noncomputable def smallest_odd_gt (x : ℝ) : ℤ :=
  2 * ⌈x / 2⌉ + 1

def x : ℝ := 3.25
def y : ℝ := 12.5

theorem problem_solution :
  (6.32 - (greatest_even_le y : ℝ)) * ((smallest_odd_gt x : ℝ) - x) = -9.94 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l770_77010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_expression_is_five_fourths_l770_77033

open Real

/-- A quadratic function with positive coefficients that has real roots. -/
structure QuadraticWithRealRoots where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  has_real_roots : 0 ≤ b^2 - 4*a*c

/-- The minimum of the three expressions for a quadratic function. -/
noncomputable def min_expression (q : QuadraticWithRealRoots) : ℝ :=
  min (min ((q.b + q.c) / q.a) ((q.c + q.a) / q.b)) ((q.a + q.b) / q.c)

/-- The theorem stating that the maximum value of the minimum expression is 5/4. -/
theorem max_min_expression_is_five_fourths :
  ∀ q : QuadraticWithRealRoots, min_expression q ≤ 5/4 ∧ 
  ∃ q' : QuadraticWithRealRoots, min_expression q' = 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_expression_is_five_fourths_l770_77033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_with_even_and_pi_period_l770_77006

noncomputable def f₁ (x : ℝ) := Real.cos (abs (2 * x))
noncomputable def f₂ (x : ℝ) := abs (Real.sin (x + Real.pi))
noncomputable def f₃ (x : ℝ) := abs (Real.sin (2 * x + Real.pi / 2))
noncomputable def f₄ (x : ℝ) := Real.tan (abs x)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  has_period f p ∧ ∀ q, 0 < q ∧ q < p → ¬ has_period f q

theorem functions_with_even_and_pi_period :
  (is_even f₁ ∧ smallest_positive_period f₁ Real.pi) ∧
  (is_even f₂ ∧ smallest_positive_period f₂ Real.pi) ∧
  ¬(is_even f₃ ∧ smallest_positive_period f₃ Real.pi) ∧
  ¬(is_even f₄ ∧ smallest_positive_period f₄ Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_with_even_and_pi_period_l770_77006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_adjacent_to_seven_l770_77079

def divisors_245 : List Nat := [5, 7, 25, 35, 49, 175, 245]

def has_common_factor (a b : Nat) : Prop :=
  ∃ k : Nat, k > 1 ∧ k ∣ a ∧ k ∣ b

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i : Nat, i < arr.length → has_common_factor (arr.get! i) (arr.get! ((i + 1) % arr.length))

theorem sum_adjacent_to_seven (arr : List Nat) :
  arr ∈ divisors_245.permutations →
  is_valid_arrangement arr →
  7 ∈ arr →
  (∃ i : Nat, i < arr.length ∧ arr.get! i = 7 ∧
    arr.get! ((i - 1 + arr.length) % arr.length) + arr.get! ((i + 1) % arr.length) = 84) :=
by sorry

#check sum_adjacent_to_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_adjacent_to_seven_l770_77079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l770_77075

-- Define the line l
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the circle C
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

-- Define the intersection points A and B
def intersect (k : ℝ) : Prop := ∃ (x1 y1 x2 y2 : ℝ),
  circleEq x1 y1 ∧ circleEq x2 y2 ∧
  y1 = line k x1 ∧ y2 = line k x2 ∧
  (x1 ≠ x2 ∨ y1 ≠ y2)

-- Theorem statement
theorem min_chord_length (k : ℝ) (h : intersect k) :
  ∃ (x1 y1 x2 y2 : ℝ),
    circleEq x1 y1 ∧ circleEq x2 y2 ∧
    y1 = line k x1 ∧ y2 = line k x2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l770_77075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_points_l770_77095

/-- Represents the points earned in a single race. -/
inductive RacePoints
  | first  : RacePoints
  | second : RacePoints
  | third  : RacePoints

/-- Converts RacePoints to a natural number. -/
def pointValue (p : RacePoints) : ℕ :=
  match p with
  | .first  => 6
  | .second => 4
  | .third  => 2

/-- Represents the results of four races. -/
def FourRaces := Fin 4 → RacePoints

/-- Calculates the total points for a set of four races. -/
def totalPoints (races : FourRaces) : ℕ :=
  Finset.sum (Finset.range 4) fun i => pointValue (races i)

/-- States that 22 is the smallest number of points that guarantees winning. -/
theorem smallest_winning_points :
  (∃ (winner : FourRaces), totalPoints winner = 22) ∧
  (∀ (winner : FourRaces), totalPoints winner ≥ 22 →
    ∀ (other : FourRaces), other ≠ winner → totalPoints other < totalPoints winner) ∧
  (∀ (n : ℕ), n < 22 →
    ∃ (a b : FourRaces), a ≠ b ∧ totalPoints a = n ∧ totalPoints b = n) := by
  sorry

#check smallest_winning_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_points_l770_77095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisors_ratio_l770_77031

def first_10_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def n : Nat := first_10_primes.prod

noncomputable def S (n : Nat) : Nat :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum fun x =>
    (Finset.filter (fun y => x * y ∣ n) (Finset.range (n + 1))).sum fun y =>
      Nat.totient x * y

theorem sum_divisors_ratio (n : Nat) (hn : n = first_10_primes.prod) :
  S n / n = 1024 := by
  sorry

#eval first_10_primes.prod

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisors_ratio_l770_77031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_numbers_l770_77047

/-- The number of red balls -/
def num_red_balls : ℕ := 5

/-- The number of black balls -/
def num_black_balls : ℕ := 5

/-- The total number of balls -/
def total_balls : ℕ := num_red_balls + num_black_balls

/-- The number of balls drawn -/
def balls_drawn : ℕ := 4

/-- The number of different numbers on the balls -/
def num_different_numbers : ℕ := 5

/-- The probability of drawing 4 balls with different numbers -/
def prob_different_numbers : ℚ := 8 / 21

/-- Theorem stating the probability of drawing 4 balls with different numbers -/
theorem probability_different_numbers :
  (Nat.choose total_balls balls_drawn : ℚ)⁻¹ *
  (Nat.choose num_different_numbers balls_drawn * 2^balls_drawn : ℚ) =
  prob_different_numbers := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_numbers_l770_77047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_positions_bound_l770_77041

/-- Represents a plane in 3D space -/
structure Plane where

/-- Represents a sphere in 3D space -/
structure Sphere where

/-- The configuration of three planes and a sphere in 3D space -/
structure SpaceConfiguration where
  planes : Fin 3 → Plane
  sphere : Sphere

/-- A function that counts the number of valid positions for a second sphere -/
noncomputable def countValidPositions (config : SpaceConfiguration) : ℕ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the number of valid positions is between 0 and 16 -/
theorem valid_positions_bound (config : SpaceConfiguration) :
  countValidPositions config ≤ 16 ∧ 0 ≤ countValidPositions config := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_positions_bound_l770_77041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_shopping_total_l770_77002

-- Define the original prices
def jacket_price : ℚ := 45.28
def shorts_price : ℚ := 29.99
def shirt_price : ℚ := 26.50
def shoes_price : ℚ := 84.25

-- Define the discount rates
def jacket_discount : ℚ := 15 / 100
def shoes_discount : ℚ := 20 / 100
def shorts_discount : ℚ := 50 / 100

-- Define the tax rate
def tax_rate : ℚ := 825 / 10000

-- Function to calculate the discounted price
def apply_discount (price : ℚ) (discount : ℚ) : ℚ :=
  price * (1 - discount)

-- Function to calculate the total price before tax
def total_before_tax : ℚ :=
  apply_discount jacket_price jacket_discount +
  apply_discount shoes_price shoes_discount +
  shirt_price +
  apply_discount shorts_price shorts_discount

-- Function to calculate the final total price
def final_total : ℚ :=
  total_before_tax * (1 + tax_rate)

-- Function to round to nearest cent
def round_to_cent (x : ℚ) : ℚ :=
  (⌊x * 100 + 1/2⌋ : ℤ) / 100

-- Theorem statement
theorem jason_shopping_total :
  round_to_cent final_total = 159.54 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_shopping_total_l770_77002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l770_77083

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.log (2 - x)

-- State the theorem
theorem domain_of_f :
  {x : ℝ | x + 1 ≥ 0 ∧ 2 - x > 0} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l770_77083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l770_77050

noncomputable section

/-- Curve C in the Cartesian coordinate system -/
def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 2 * Real.sin θ)

/-- Line l in the Cartesian coordinate system -/
def line_l (t : ℝ) : ℝ × ℝ := (t + Real.sqrt 3, 2 * t - 2 * Real.sqrt 3)

/-- Left focus of an ellipse with semi-major axis a and semi-minor axis b -/
def left_focus (a b : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 - b^2), 0)

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem intersection_properties (A B : ℝ × ℝ) (h1 : ∃ θ₁, curve_C θ₁ = A) 
    (h2 : ∃ θ₂, curve_C θ₂ = B) (h3 : ∃ t₁, line_l t₁ = A) (h4 : ∃ t₂, line_l t₂ = B) :
  let F := left_focus 4 2
  distance A B = 40 / 17 ∧ 
  dot_product (A.1 - F.1, A.2 - F.2) (B.1 - F.1, B.2 - F.2) = 44 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l770_77050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_A_and_B_l770_77000

-- Define the possible identities
inductive Identity : Type
  | Knight : Identity
  | Liar : Identity

-- Define A and B as having identities
variable (A B : Identity)

-- Define the statement made by A
def A_statement (A B : Identity) : Prop := (B = Identity.Knight) → (A = Identity.Liar)

-- Define what it means for a statement to be true based on the speaker's identity
def statement_is_true (speaker : Identity) (statement : Prop) : Prop :=
  (speaker = Identity.Knight ∧ statement) ∨ (speaker = Identity.Liar ∧ ¬statement)

-- Theorem statement
theorem identify_A_and_B :
  statement_is_true A (A_statement A B) →
  A = Identity.Knight ∧ B = Identity.Liar :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_A_and_B_l770_77000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l770_77017

-- Define the function f(x) = 2 + ln x
noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l770_77017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_APB_l770_77019

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point P
noncomputable def P : ℝ × ℝ := (-1, Real.sqrt 3)

-- Define the angle APB
noncomputable def angle_APB (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_angle_APB :
  ∃ (max_angle : ℝ), 
    (∀ (A B : ℝ × ℝ), is_on_circle A.1 A.2 → is_on_circle B.1 B.2 → angle_APB A B ≤ max_angle) ∧
    (∃ (A B : ℝ × ℝ), is_on_circle A.1 A.2 ∧ is_on_circle B.1 B.2 ∧ angle_APB A B = max_angle) ∧
    max_angle = π / 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_APB_l770_77019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_is_circumcircle_lines_are_tangent_l770_77053

noncomputable section

-- Define the points
def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (6, 2 * Real.sqrt 3)
def B : ℝ × ℝ := (4, 4)
noncomputable def P : ℝ × ℝ := (0, 4 * Real.sqrt 3)

-- Define the circle equation
noncomputable def circle_eq (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 3 * Real.sqrt 3 / 2)^2 = 16

-- Define the line equations
def line_eq1 (x : ℝ) : Prop := x = 0
noncomputable def line_eq2 (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 12 = 0

-- Theorem statements
theorem circle_is_circumcircle :
  circle_eq O.1 O.2 ∧ circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 := by sorry

theorem lines_are_tangent :
  (∃ (x y : ℝ), line_eq1 x ∧ circle_eq x y) ∧
  (∃ (x y : ℝ), line_eq2 x y ∧ circle_eq x y) ∧
  line_eq1 P.1 ∧ line_eq2 P.1 P.2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_is_circumcircle_lines_are_tangent_l770_77053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_true_l770_77072

-- Define the propositions
def proposition1 : Prop := ¬(∀ x : ℝ, x > 2 → x > 3)

def proposition2 : Prop := ¬(∀ a : ℝ, a > 0 → (∀ x y : ℝ, x < y → a^x < a^y))

noncomputable def proposition3 : Prop := (∀ x : ℝ, Real.sin x = Real.sin (x + Real.pi)) ∨ 
                                         (∀ x : ℝ, Real.sin (2*x) = Real.sin (2*(x + 2*Real.pi)))

-- Theorem stating that all three propositions are true
theorem all_propositions_true : proposition1 ∧ proposition2 ∧ proposition3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_true_l770_77072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imported_car_price_l770_77091

/-- Represents the two-tiered tax system for imported cars -/
structure TaxSystem where
  firstTierRate : ℚ  -- Tax rate for the first tier
  secondTierRate : ℚ  -- Tax rate for the second tier
  firstTierThreshold : ℚ  -- Price threshold for the first tier

/-- Calculates the tax for a given car price under the specified tax system -/
noncomputable def calculateTax (system : TaxSystem) (price : ℚ) : ℚ :=
  let firstTierTax := min price system.firstTierThreshold * system.firstTierRate
  let secondTierTax := max 0 (price - system.firstTierThreshold) * system.secondTierRate
  firstTierTax + secondTierTax

/-- Theorem: Given the specified tax system and total tax paid, the car price is $30,000 -/
theorem imported_car_price
  (system : TaxSystem)
  (h_first_rate : system.firstTierRate = 1/4)
  (h_second_rate : system.secondTierRate = 3/20)
  (h_threshold : system.firstTierThreshold = 10000)
  (h_tax_paid : calculateTax system 30000 = 5500) :
  ∃ (price : ℚ), price = 30000 ∧ calculateTax system price = 5500 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_imported_car_price_l770_77091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016th_term_l770_77060

def sequenceProperty (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → S n / n = a (n + 1) / 2

theorem sequence_2016th_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : sequenceProperty a S) : a 2016 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016th_term_l770_77060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_beta_l770_77081

theorem cos_two_beta (α β : ℝ) 
  (h1 : Real.sin (α - β) = 3/5)
  (h2 : Real.cos (α + β) = -3/5)
  (h3 : α - β ∈ Set.Ioo (π/2) π)
  (h4 : α + β ∈ Set.Ioo (π/2) π) : 
  Real.cos (2 * β) = 24/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_beta_l770_77081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l770_77015

def sequence_a : ℕ → ℝ
  | 0 => 1  -- Adding this case to handle n = 0
  | 1 => 1
  | 2 => 3
  | (n + 3) => 3 * sequence_a (n + 2) - 2 * sequence_a (n + 1)

theorem sequence_a_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_a (n + 2) - sequence_a (n + 1) = 2 * (sequence_a (n + 1) - sequence_a n)) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l770_77015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_4_l770_77073

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 4)
noncomputable def f (x : ℝ) : ℝ := 4 - t x

-- Theorem statement
theorem t_of_f_4 : t (f 4) = Real.sqrt (20 - 8 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_4_l770_77073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_externally_tangent_l770_77086

/-- Define a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  distance c1.h c1.k c2.h c2.k = c1.r + c2.r

/-- The first circle: x^2 + y + 2x - 2y - 2 = 0 -/
def circle1 : Circle :=
  { h := -1, k := 1, r := 2 }

/-- The second circle: (x-2)^2 + (y+3)^2 = 9 -/
def circle2 : Circle :=
  { h := 2, k := -3, r := 3 }

/-- Theorem: The two given circles are externally tangent -/
theorem circles_are_externally_tangent :
  are_externally_tangent circle1 circle2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_externally_tangent_l770_77086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_10T_l770_77045

-- Define the function T
noncomputable def T (x y z : ℝ) : ℝ := (1/4) * x^2 - (1/5) * y^2 + (1/6) * z^2

-- State the theorem
theorem min_value_of_10T :
  ∀ x y z : ℝ,
  1 ≤ x ∧ x ≤ 4 →
  1 ≤ y ∧ y ≤ 4 →
  1 ≤ z ∧ z ≤ 4 →
  x - y + z = 4 →
  10 * T x y z ≥ 23 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_10T_l770_77045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neighborhood_cable_cost_correct_l770_77052

/-- Calculates the total cost of cable for a neighborhood with given street layout and cable requirements. -/
def neighborhood_cable_cost
  (num_ew_streets : ℕ)
  (num_ns_streets : ℕ)
  (ew_street_length : ℝ)
  (ns_street_length : ℝ)
  (cable_per_mile : ℝ)
  (cable_cost_per_mile : ℝ) : ℝ :=
  let total_ew_length := num_ew_streets * ew_street_length
  let total_ns_length := num_ns_streets * ns_street_length
  let total_street_length := total_ew_length + total_ns_length
  let total_cable_length := total_street_length * cable_per_mile
  total_cable_length * cable_cost_per_mile

theorem neighborhood_cable_cost_correct
  (num_ew_streets : ℕ)
  (num_ns_streets : ℕ)
  (ew_street_length : ℝ)
  (ns_street_length : ℝ)
  (cable_per_mile : ℝ)
  (cable_cost_per_mile : ℝ)
  (h1 : num_ew_streets = 18)
  (h2 : num_ns_streets = 10)
  (h3 : ew_street_length = 2)
  (h4 : ns_street_length = 4)
  (h5 : cable_per_mile = 5)
  (h6 : cable_cost_per_mile = 2000)
  : neighborhood_cable_cost num_ew_streets num_ns_streets ew_street_length ns_street_length cable_per_mile cable_cost_per_mile = 760000 :=
by
  -- The total cost of cable for the neighborhood is $760,000
  sorry

#eval neighborhood_cable_cost 18 10 2 4 5 2000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neighborhood_cable_cost_correct_l770_77052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l770_77059

theorem ordering_abc (a b c : ℝ) 
  (ha : a = Real.log 0.5 / Real.log 0.6) 
  (hb : b = Real.log 0.5) 
  (hc : c = 0.6^(1/2)) : 
  a > c ∧ c > b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l770_77059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l770_77098

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 + 3*x + 2) / (x + 1)

-- State the theorem
theorem f_range :
  ∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, x ≠ -1 ∧ f x = y :=
by
  -- Proof is omitted
  sorry

-- Additional lemma to show that f x = x + 2 when x ≠ -1
lemma f_simplification (x : ℝ) (h : x ≠ -1) :
  f x = x + 2 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l770_77098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l770_77054

/-- CoeffX n p returns the coefficient of x^n in the polynomial p -/
def CoeffX (n : ℕ) (p : ℝ → ℝ) : ℝ := sorry

/-- The coefficient of x^2 in the expansion of (1 - 1/x)(1+x)^4 is 2 -/
theorem coefficient_x_squared (x : ℝ) : 
  (CoeffX 2 (fun x => (1 - 1/x) * (1 + x)^4)) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l770_77054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l770_77071

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define point P
def P : ℝ × ℝ := (2, -1)

-- Define that P is the midpoint of AB
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define that AB is a chord of the circle
def is_chord (A B : ℝ × ℝ) : Prop :=
  my_circle A.1 A.2 ∧ my_circle B.1 B.2

-- Theorem statement
theorem chord_equation (A B : ℝ × ℝ) :
  is_midpoint P A B → is_chord A B →
  ∃ (k : ℝ), A.2 - B.2 = k * (A.1 - B.1) ∧ k = 1 ∧
  (λ (x y : ℝ) ↦ x - y - 3 = 0) A.1 A.2 ∧ (λ (x y : ℝ) ↦ x - y - 3 = 0) B.1 B.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l770_77071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l770_77029

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (5 * x + 2) / Real.sqrt (x - 7)

-- Define the domain of f
def domain_f : Set ℝ := {x | x > 7}

-- Theorem stating that domain_f is the correct domain for f
theorem domain_of_f : 
  ∀ x : ℝ, f x ∈ Set.univ ↔ x ∈ domain_f :=
by
  intro x
  constructor
  · intro _
    sorry -- Proof that if f x is defined, then x > 7
  · intro h
    sorry -- Proof that if x > 7, then f x is defined


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l770_77029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l770_77036

-- Define the square side length
def square_side : ℝ := 12

-- Define the quarter circle radius
def quarter_circle_radius : ℝ := 6

-- Define the area of the square
def square_area : ℝ := square_side ^ 2

-- Define the area of one full circle
noncomputable def full_circle_area : ℝ := Real.pi * quarter_circle_radius ^ 2

-- Theorem statement
theorem shaded_area_calculation :
  square_area - full_circle_area = 144 - 36 * Real.pi := by
  sorry

#eval square_area
#eval quarter_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l770_77036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l770_77048

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- The coordinates of the left vertex, top vertex, and right focus of an ellipse -/
def left_vertex (e : Ellipse) : ℝ × ℝ := (-e.a, 0)
def top_vertex (e : Ellipse) : ℝ × ℝ := (0, e.b)
noncomputable def right_focus (e : Ellipse) : ℝ × ℝ := (Real.sqrt (e.a^2 - e.b^2), 0)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector from point p to point q -/
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Theorem: If the vector from the top vertex to the left vertex is perpendicular
    to the vector from the top vertex to the right focus, then the eccentricity
    of the ellipse is (√5 - 1)/2 -/
theorem ellipse_eccentricity (e : Ellipse) :
  let M := left_vertex e
  let N := top_vertex e
  let F := right_focus e
  dot_product (vector N M) (vector N F) = 0 →
  eccentricity e = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l770_77048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l770_77042

/-- The time taken for two trains to cross each other -/
noncomputable def time_to_cross (train_length : ℝ) (time1 time2 : ℝ) : ℝ :=
  (2 * train_length) / (train_length / time1 + train_length / time2)

/-- Theorem stating the time taken for the trains to cross each other -/
theorem trains_crossing_time :
  let train_length : ℝ := 120
  let time1 : ℝ := 10
  let time2 : ℝ := 20
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |time_to_cross train_length time1 time2 - 13.33| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l770_77042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_per_pound_l770_77070

/-- Proves that the price per pound of apples is $4 given the specified conditions --/
theorem apple_price_per_pound :
  let apple_weight : ℚ := 1/4
  let daily_consumption : ℚ := 1/2
  let days : ℕ := 14
  let total_cost : ℚ := 7
  total_cost / (daily_consumption * ↑days * apple_weight) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_per_pound_l770_77070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hawks_lost_percentage_approx_30_l770_77087

/-- Represents the number of games in each category --/
structure GameStats where
  won : ℕ
  lost : ℕ
  draw : ℕ

/-- Calculates the percentage of games lost --/
def percentLost (stats : GameStats) : ℚ :=
  stats.lost / (stats.won + stats.lost + stats.draw) * 100

/-- The given conditions of the problem --/
def hawksStats (k : ℕ) : GameStats :=
  { won := 7 * k
  , lost := 3 * k
  , draw := 5 }

/-- Theorem stating that the percentage of games lost is approximately 30% --/
theorem hawks_lost_percentage_approx_30 : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ ∀ (k : ℕ), k > 0 → |percentLost (hawksStats k) - 30| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hawks_lost_percentage_approx_30_l770_77087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l770_77018

open Real

theorem trigonometric_identities (α : ℝ) : 
  (Real.sin (π/3 + α) = Real.sin (2*π/3 - α)) ∧ 
  (Real.sin (π/4 + α) = -Real.cos (5*π/4 - α)) ∧ 
  (Real.tan α ^ 2 * Real.sin α ^ 2 = Real.tan α ^ 2 - Real.sin α ^ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l770_77018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_with_divisor_l770_77046

-- Define the set of prime numbers excluding 1
def PrimeSet := {n : ℕ | Nat.Prime n ∧ n ≠ 1}

-- Define the theorem
theorem smallest_cube_with_divisor 
  (p q r s : ℕ) 
  (hp : p ∈ PrimeSet) 
  (hq : q ∈ PrimeSet) 
  (hr : r ∈ PrimeSet) 
  (hs : s ∈ PrimeSet) 
  (hdistinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
  (p^2 * q^3 * r^4 * s^5) ∣ (p * q * r * s^2)^3 ∧ 
  ∀ (n : ℕ), (p^2 * q^3 * r^4 * s^5) ∣ n^3 → (p * q * r * s^2)^3 ≤ n^3 :=
by
  sorry

#check smallest_cube_with_divisor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_with_divisor_l770_77046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sqrt_a_fractional_part_l770_77023

noncomputable def a : ℕ → ℝ
  | 0 => 3
  | 1 => 7
  | (n + 2) => (a (n + 1) * a n + 4) / (a (n + 1) - 2)

theorem limit_sqrt_a_fractional_part :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |Real.sqrt (a n) - Int.floor (Real.sqrt (a n)) - (1/2)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sqrt_a_fractional_part_l770_77023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_co2_decomposition_l770_77038

/-- Represents the enthalpy change of the reaction -/
def ΔH : ℝ := sorry

/-- Represents the entropy change of the reaction -/
def ΔS : ℝ := sorry

/-- Represents the Gibbs free energy change of the reaction -/
def ΔG : ℝ → ℝ := sorry

/-- Represents the temperature of the reaction -/
def T : ℝ := sorry

/-- Condition for low temperature -/
def is_low_temperature : ℝ → Prop := sorry

/-- The reaction: 2CO₂(g) = 2CO(g) + O₂(g) -/
def reaction : Prop := sorry

/-- Ruthenium complexes are used as catalysts -/
def ruthenium_catalyst : Prop := sorry

/-- Photocatalysis is involved in the reaction -/
def photocatalysis : Prop := sorry

theorem co2_decomposition (h_reaction : reaction) (h_catalyst : ruthenium_catalyst) 
  (h_photo : photocatalysis) :
  (ΔH > 0) ∧ 
  (ΔS > 0) ∧ 
  (∀ t, is_low_temperature t → ΔG t > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_co2_decomposition_l770_77038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l770_77037

open Set
open Function
open Real

-- Define the function f(x) = ln x - x
noncomputable def f : ℝ → ℝ := λ x ↦ log x - x

-- Define the domain of f
def domain : Set ℝ := {x | x > 0}

-- Define the monotonic increasing interval
def monotonic_increasing_interval : Set ℝ := Ioo 0 1

-- Theorem statement
theorem f_monotonic_increasing :
  StrictMonoOn f monotonic_increasing_interval ∧
  ∀ x ∈ domain, x ∉ monotonic_increasing_interval →
    ¬ StrictMonoOn f {y | y = x} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l770_77037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_equality_l770_77068

theorem nested_sqrt_equality : 
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * (125 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_equality_l770_77068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reciprocal_heights_similar_l770_77009

/-- Given a triangle ABC with sides a, b, c and corresponding heights m_a, m_b, m_c,
    prove that the triangle formed by the reciprocals of the heights is similar to ABC. -/
theorem triangle_reciprocal_heights_similar (a b c m_a m_b m_c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pos_ma : m_a > 0) (h_pos_mb : m_b > 0) (h_pos_mc : m_c > 0)
  (h_area_a : a * m_a = 2 * (1/2 * a * m_a))
  (h_area_b : b * m_b = 2 * (1/2 * b * m_b))
  (h_area_c : c * m_c = 2 * (1/2 * c * m_c)) :
  (1/m_a : ℝ) / (1/m_b : ℝ) = a / b ∧ 
  (1/m_b : ℝ) / (1/m_c : ℝ) = b / c ∧ 
  (1/m_a : ℝ) / (1/m_c : ℝ) = a / c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reciprocal_heights_similar_l770_77009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_side_for_12_by_9_table_l770_77034

/-- The minimum length of the shorter side of a rectangular room that can accommodate a rectangular table diagonally -/
noncomputable def minimum_room_side (table_length : ℝ) (table_width : ℝ) : ℝ :=
  Real.sqrt (table_length ^ 2 + table_width ^ 2)

/-- Theorem: For a 12' by 9' table, the minimum room side length is 15' -/
theorem min_room_side_for_12_by_9_table :
  minimum_room_side 12 9 = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_side_for_12_by_9_table_l770_77034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l770_77077

/-- Calculates the time taken for two trains to cross each other --/
noncomputable def train_crossing_time (length_A length_B speed_A speed_B : ℝ) : ℝ :=
  let total_length := length_A + length_B
  let relative_speed := (speed_A + speed_B) * (5/18)  -- Convert km/hr to m/s
  total_length / relative_speed

/-- Theorem stating that the time taken for the given trains to cross is 13 seconds --/
theorem train_crossing_theorem :
  train_crossing_time 175 150 54 36 = 13 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry to skip the detailed proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l770_77077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2n_is_perfect_square_l770_77058

/-- Represents the number of natural numbers with digit sum n, using only digits 1, 3, and 4 -/
def a : ℕ → ℕ := sorry

/-- The sequence a₂ₙ is always a perfect square -/
theorem a_2n_is_perfect_square : ∀ n : ℕ, ∃ k : ℕ, a (2 * n) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2n_is_perfect_square_l770_77058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_monday_exists_l770_77020

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the months of the year -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August
  | September
  | October
  | November
  | December

/-- Returns the number of days in a given month -/
def daysInMonth (m : Month) : ℕ :=
  match m with
  | Month.January => 31
  | Month.February => 28  -- Assuming non-leap year for simplicity
  | Month.March => 31
  | Month.April => 30
  | Month.May => 31
  | Month.June => 30
  | Month.July => 31
  | Month.August => 31
  | Month.September => 30
  | Month.October => 31
  | Month.November => 30
  | Month.December => 31

/-- Returns the day of the week for the 13th of a given month, 
    based on the day of the week for the 13th of the previous month -/
def nextMonthDay13 (prevDay : DayOfWeek) (prevMonth : Month) : DayOfWeek :=
  let daysBetween := (daysInMonth prevMonth) % 7
  match prevDay with
  | DayOfWeek.Monday => 
      if daysBetween = 0 then DayOfWeek.Monday
      else if daysBetween = 1 then DayOfWeek.Tuesday
      else if daysBetween = 2 then DayOfWeek.Wednesday
      else if daysBetween = 3 then DayOfWeek.Thursday
      else if daysBetween = 4 then DayOfWeek.Friday
      else if daysBetween = 5 then DayOfWeek.Saturday
      else DayOfWeek.Sunday
  -- Similar cases for other days of the week...
  | _ => DayOfWeek.Monday  -- Placeholder for other cases

/-- Returns the next month -/
def nextMonth (m : Month) : Month :=
  match m with
  | Month.January => Month.February
  | Month.February => Month.March
  | Month.March => Month.April
  | Month.April => Month.May
  | Month.May => Month.June
  | Month.June => Month.July
  | Month.July => Month.August
  | Month.August => Month.September
  | Month.September => Month.October
  | Month.October => Month.November
  | Month.November => Month.December
  | Month.December => Month.January

/-- Theorem: In any year, there exists at least one month where the 13th falls on a Monday -/
theorem thirteenth_monday_exists : 
  ∀ (startMonth : Month) (startDay : DayOfWeek),
  ∃ (m : Month), 
  (nextMonthDay13 startDay startMonth = DayOfWeek.Monday) ∨ 
  (∃ (n : ℕ), n < 12 ∧ 
    (nextMonthDay13 
      (nextMonthDay13 startDay startMonth) 
      (nextMonth startMonth) = DayOfWeek.Monday)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_monday_exists_l770_77020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l770_77004

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (x / 4) * Real.cos (x / 4) + 
                       Real.sqrt 6 * (Real.cos (x / 4))^2 - Real.sqrt 6 / 2 + m

-- State the theorem
theorem m_range_theorem :
  ∀ m : ℝ, (∀ x : ℝ, -5*π/6 ≤ x ∧ x ≤ π/6 → f x m ≤ 0) → m ≤ -Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l770_77004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l770_77093

-- Define the ellipse C
noncomputable def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the line passing through F₂ and perpendicular to x-axis
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 1}

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ellipse_equation (C : Set (ℝ × ℝ)) 
  (h1 : ∃ a b : ℝ, C = Ellipse a b)
  (h2 : F₁ ∉ C ∧ F₂ ∉ C)
  (h3 : ∃ y₁ y₂ : ℝ, (1, y₁) ∈ C ∧ (1, y₂) ∈ C ∧ y₁ ≠ y₂)
  (h4 : ∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ Line ∧ B ∈ Line ∧ distance A B = 3) :
  C = Ellipse 2 (Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l770_77093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_inequality_l770_77088

theorem sine_sum_inequality (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ π) 
  (hy : 0 ≤ y ∧ y ≤ π) 
  (hz : 0 ≤ z ∧ z ≤ π) : 
  Real.sin x + Real.sin y + Real.sin z ≥ Real.sin (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_inequality_l770_77088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olga_has_three_stripes_l770_77097

/-- The number of stripes on each of Olga's tennis shoes -/
def olga_stripes : ℕ := 3

/-- The number of stripes on each of Rick's tennis shoes -/
def rick_stripes : ℕ := olga_stripes - 1

/-- The number of stripes on each of Hortense's tennis shoes -/
def hortense_stripes : ℕ := 2 * olga_stripes

/-- The total number of stripes on all of their pairs of tennis shoes -/
def total_stripes : ℕ := 22

theorem olga_has_three_stripes :
  2 * olga_stripes + 2 * rick_stripes + 2 * hortense_stripes = total_stripes :=
by
  -- Expand the definitions
  unfold olga_stripes rick_stripes hortense_stripes total_stripes
  -- Simplify the arithmetic
  simp [Nat.mul_add, Nat.add_assoc]
  -- Check the equality
  rfl

#eval olga_stripes -- Should output 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olga_has_three_stripes_l770_77097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l770_77011

theorem diophantine_equation_solutions :
  let S := {(x, y) : ℤ × ℤ | x ≠ 0 ∧ y ≠ 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 19}
  S = {(38, 38), (380, 20), (-342, 18), (20, 380), (18, -342)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l770_77011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l770_77090

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (6 * x + 15) / (x - 5)

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≠ 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l770_77090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_decreasing_interval_of_shifted_cosine_squared_minus_sine_squared_l770_77067

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sin x ^ 2

-- Define the shifted function g
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 12)

-- Theorem statement
theorem max_decreasing_interval_of_shifted_cosine_squared_minus_sine_squared :
  ∃ (a : ℝ), a = 5 * Real.pi / 12 ∧
  (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a → g x₂ < g x₁) ∧
  (∀ a' > a, ∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a' ∧ g x₂ ≥ g x₁) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_decreasing_interval_of_shifted_cosine_squared_minus_sine_squared_l770_77067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_consumption_l770_77099

noncomputable def fuel_consumption (x k : ℝ) : ℝ := (1/5) * (x - k + 4500/x)

noncomputable def fuel_consumption_100km (x k : ℝ) : ℝ := (100/x) * fuel_consumption x k

theorem car_fuel_consumption 
  (x k : ℝ) 
  (h1 : 60 ≤ x ∧ x ≤ 120) 
  (h2 : 60 ≤ k ∧ k ≤ 100) :
  (fuel_consumption x 100 ≤ 9 → 60 ≤ x ∧ x ≤ 100) ∧
  ((75 ≤ k ∧ k < 100 → 
    ∃ (x_min : ℝ), fuel_consumption_100km x_min k = 20 - k^2/900 ∧
    ∀ (y : ℝ), 60 ≤ y ∧ y ≤ 120 → fuel_consumption_100km y k ≥ fuel_consumption_100km x_min k) ∧
   (60 ≤ k ∧ k < 75 → 
    fuel_consumption_100km 120 k = 105/4 - k/6 ∧
    ∀ (y : ℝ), 60 ≤ y ∧ y ≤ 120 → fuel_consumption_100km y k ≥ fuel_consumption_100km 120 k)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_consumption_l770_77099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l770_77063

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  first : a 1 = a 0 + d  -- Definition of arithmetic sequence
  nth : ∀ n, a (n + 1) = a n + d  -- General term formula

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Theorem: If a_6 = -3 and S_6 = 12 in an arithmetic sequence, then a_5 = -1 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 6 = -3)
  (h2 : sum_n seq 6 = 12) :
  seq.a 5 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l770_77063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_l770_77094

/-- The number of children in the circle -/
def num_children : ℕ := 73

/-- The number of candies distributed -/
def num_candies : ℕ := 2020

/-- The position of the child receiving the n-th candy -/
def candy_position (n : ℕ) : ℕ := (n * (n + 1) / 2) % num_children

/-- The set of children who received candies -/
def children_with_candy : Finset ℕ := Finset.range num_candies

theorem candy_distribution :
  (Finset.filter (fun i => ∃ n ∈ children_with_candy, candy_position n = i) (Finset.range num_children)).card = 37 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_l770_77094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l770_77016

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- The angle of inclination of a line --/
noncomputable def angle_of_inclination (l : Line) : ℝ := Real.arctan (- l.a / l.b)

/-- Check if a point lies on a line --/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has equal intercepts on both axes --/
def equal_intercepts (l : Line) : Prop :=
  l.a * (- l.c / l.b) = l.b * (- l.c / l.a)

/-- The reference line x-4y+3=0 --/
def reference_line : Line := { a := 1, b := -4, c := 3, nonzero := by simp }

theorem line_equation (l : Line) :
  point_on_line l 3 2 →
  angle_of_inclination l = 2 * angle_of_inclination reference_line →
  equal_intercepts l →
  (l.a = 8 ∧ l.b = 15 ∧ l.c = 6) ∨
  (l.a = 2 ∧ l.b = -3 ∧ l.c = 0) ∨
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l770_77016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equal_angles_l770_77039

/-- The ellipse with equation x²/4 + y² = 1 and one focus at (√3, 0) -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + p.2^2 = 1}

/-- The focus point F = (√3, 0) -/
noncomputable def F : ℝ × ℝ := (Real.sqrt 3, 0)

/-- A chord of the ellipse passing through the focus F -/
def Chord (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = m * (p.1 - F.1)}

/-- The point P = (p, 0) on the x-axis -/
def P (p : ℝ) : ℝ × ℝ := (p, 0)

/-- The property that angles APF and BPF are equal for any chord AB passing through F -/
def EqualAngles (p : ℝ) : Prop :=
  ∀ m : ℝ, ∀ A B : ℝ × ℝ,
    A ∈ Ellipse ∩ Chord m → B ∈ Ellipse ∩ Chord m →
    A ≠ B → A ≠ F → B ≠ F →
    (A.1 - p) * (B.1 - F.1) = (B.1 - p) * (A.1 - F.1)

theorem ellipse_equal_angles :
  ∃! p : ℝ, p > 0 ∧ EqualAngles p ∧ p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equal_angles_l770_77039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_lambdas_constant_l770_77024

/-- The ellipse C: x²/4 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line L: x = my + 1 -/
def line_L (m x y : ℝ) : Prop := x = m*y + 1

/-- The right focus of ellipse C -/
def right_focus : ℝ × ℝ := (1, 0)

/-- Point A is on ellipse C and line L -/
noncomputable def point_A (m : ℝ) : ℝ × ℝ := sorry

/-- Point B is on ellipse C and line L -/
noncomputable def point_B (m : ℝ) : ℝ × ℝ := sorry

/-- Point M is the intersection of line L and y-axis -/
noncomputable def point_M (m : ℝ) : ℝ × ℝ := (0, -1/m)

/-- Vector from point P to point Q -/
def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Scalar lambda_1 such that MA = lambda_1 * AF -/
noncomputable def lambda_1 (m : ℝ) : ℝ := sorry

/-- Scalar lambda_2 such that MB = lambda_2 * BF -/
noncomputable def lambda_2 (m : ℝ) : ℝ := sorry

theorem sum_of_lambdas_constant (m : ℝ) : lambda_1 m + lambda_2 m = -8/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_lambdas_constant_l770_77024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_imply_omega_range_l770_77040

noncomputable section

-- Define the function f
def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)

-- State the theorem
theorem function_zeros_imply_omega_range (ω : ℝ) :
  ω > 0 →
  (∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi / 3 ∧
    f ω x₁ = 0 ∧ f ω x₂ = 0 ∧
    ∀ x, 0 < x ∧ x < Real.pi / 3 ∧ f ω x = 0 → (x = x₁ ∨ x = x₂)) →
  21 / 4 < ω ∧ ω ≤ 33 / 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_imply_omega_range_l770_77040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_linear_equations_l770_77065

-- Define the equations and inequality
def eq1 (x y : ℝ) : Prop := 3 * x - y = 2
def eq2 (x : ℝ) : Prop := x + 1 / x + 2 = 0
def eq3 (x : ℝ) : Prop := x^2 - 2 * x - 3 = 0
def eq4 (x : ℝ) : Prop := x = 0
def ineq5 (x : ℝ) : Prop := 3 * x - 1 ≥ 5
def eq6 (x : ℝ) : Prop := (1/2) * x = 1/2
def eq7 (x : ℝ) : Prop := (2 * x + 1) / 3 = (1/6) * x

-- Define a predicate for linear equations
def isLinear (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y, f x y ↔ a * x + b * y = c

-- Theorem statement
theorem count_linear_equations :
  (isLinear eq1) ∧
  ¬(isLinear (λ x _ ↦ eq2 x)) ∧
  ¬(isLinear (λ x _ ↦ eq3 x)) ∧
  (isLinear (λ x _ ↦ eq4 x)) ∧
  ¬(isLinear (λ x _ ↦ ineq5 x)) ∧
  (isLinear (λ x _ ↦ eq6 x)) ∧
  (isLinear (λ x _ ↦ eq7 x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_linear_equations_l770_77065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intermediate_positions_l770_77007

/-- Represents a position in the 2D plane -/
structure Position where
  x : ℤ
  y : ℤ

/-- Represents a cardinal direction -/
inductive Direction
  | North
  | South
  | East
  | West

/-- The length of the i-th jump -/
def jumpLength (i : ℕ) : ℕ := 2^(i - 1)

/-- The sequence of n jumps -/
def JumpSequence (n : ℕ) := Fin n → Direction

/-- The position after the i-th jump given a jump sequence -/
def positionAfterJump (n : ℕ) (i : Fin n) (js : JumpSequence n) : Position :=
  sorry

/-- The final position after n jumps -/
def finalPosition (n : ℕ) (js : JumpSequence n) : Position :=
  positionAfterJump n ⟨n-1, sorry⟩ js

/-- Theorem: Given the final position, we can uniquely determine all intermediate positions -/
theorem unique_intermediate_positions (n : ℕ) (js1 js2 : JumpSequence n) :
  finalPosition n js1 = finalPosition n js2 →
  ∀ i : Fin n, positionAfterJump n i js1 = positionAfterJump n i js2 :=
  by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intermediate_positions_l770_77007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l770_77062

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_equation : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x, y) = M ∨ (x, y) = N
  h_foci_vector : F₂.1 - F₁.1 = 3 * (N.1 - M.1) ∧ F₂.2 - F₁.2 = 3 * (N.2 - M.2)
  h_orthogonal : (M.1 - F₁.1) * (N.1 - F₂.1) + (M.2 - F₁.2) * (N.2 - F₂.2) = 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.F₂.1 - h.F₁.1)^2 + (h.F₂.2 - h.F₁.2)^2) / (2 * h.a)

/-- Theorem stating that the eccentricity of the hyperbola with given properties is √5 + √2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : eccentricity h = Real.sqrt 5 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l770_77062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_females_in_coach_class_l770_77084

theorem females_in_coach_class 
  (total_passengers : ℕ) 
  (female_percentage : ℚ) 
  (first_class_percentage : ℚ) 
  (male_first_class_fraction : ℚ) 
  (h1 : total_passengers = 120)
  (h2 : female_percentage = 55 / 100)
  (h3 : first_class_percentage = 10 / 100)
  (h4 : male_first_class_fraction = 1 / 3) :
  (female_percentage * total_passengers - 
    (1 - male_first_class_fraction) * first_class_percentage * total_passengers).floor = 58 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_females_in_coach_class_l770_77084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_r_l770_77032

-- Define the function r(x)
noncomputable def r (x : ℝ) : ℝ := 1 / (1 - x)^2

-- State the theorem about the range of r(x)
theorem range_of_r :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, x ≠ 1 ∧ r x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_r_l770_77032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_implies_a_equals_two_l770_77013

/-- Curve C₁ in parametric form -/
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos θ, 2 * Real.sin θ)

/-- Point P defined in terms of point M on C₁ -/
def P (a : ℝ) (M : ℝ × ℝ) : ℝ × ℝ := (a * M.1, a * M.2)

/-- Curve C₂ as the locus of point P -/
def C₂ (a : ℝ) (x y : ℝ) : Prop := (x - 2*a)^2 + y^2 = 4*a^2

/-- Point A in Cartesian coordinates -/
noncomputable def A : ℝ × ℝ := (1, Real.sqrt 3)

/-- Area of triangle AOB -/
noncomputable def areaAOB (a : ℝ) (α : ℝ) : ℝ :=
  let B := (2*a + 2*a*Real.cos α, 2*a*Real.sin α)
  abs (B.1 * A.2 - B.2 * A.1) / 2

/-- The theorem to be proved -/
theorem max_area_implies_a_equals_two (a : ℝ) : 
  (a > 0) → (a ≠ 1) → (∀ α, areaAOB a α ≤ 4 + 2*Real.sqrt 3) → 
  (∃ α, areaAOB a α = 4 + 2*Real.sqrt 3) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_implies_a_equals_two_l770_77013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_lineup_count_l770_77082

/-- The number of ways to choose 5 starters from a team of 12 players,
    including a set of 4 quadruplets, with exactly 2 quadruplets in the starting lineup -/
theorem soccer_team_lineup_count : 
  (let total_players : ℕ := 12
   let quadruplets : ℕ := 4
   let starters : ℕ := 5
   let quadruplets_in_lineup : ℕ := 2
   let non_quadruplets : ℕ := total_players - quadruplets
   let remaining_spots : ℕ := starters - quadruplets_in_lineup
   (Nat.choose quadruplets quadruplets_in_lineup) * (Nat.choose non_quadruplets remaining_spots)) = 336 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_lineup_count_l770_77082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l770_77057

/-- Given a projection that takes (3, 3) to (45/10, 15/10), 
    prove that the projection of (1, -1) is (3/5, 1/5) -/
theorem projection_problem (P : ℝ × ℝ → ℝ × ℝ) 
    (h : P (3, 3) = (45/10, 15/10)) : 
  P (1, -1) = (3/5, 1/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l770_77057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_axis_l770_77076

theorem sine_symmetry_axis :
  ∀ x : ℝ, Real.sin (π/2 + x) = Real.sin (π/2 - x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_axis_l770_77076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_surface_area_l770_77026

/-- Given a cylinder with a plane parallel to its axis dividing the base circumference
    in ratio m:n and creating a cross-section of area S, the lateral surface area of
    the cylinder is πS / sin(πn / (m+n)). -/
theorem cylinder_lateral_surface_area
  (m n S : ℝ) (m_pos : 0 < m) (n_pos : 0 < n) (S_pos : 0 < S) :
  ∃ (R H : ℝ),
    R > 0 ∧ H > 0 ∧
    S = 2 * R * Real.sin (π * n / (m + n)) * H ∧
    2 * π * R * H = π * S / Real.sin (π * n / (m + n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_surface_area_l770_77026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l770_77085

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 1

-- Theorem statement
theorem function_properties :
  (∀ x : ℝ, f x ≤ f 0) ∧  -- Maximum value at x = 0
  (∀ x : ℝ, f x ≥ f 1) ∧  -- Minimum value at x = 1
  (f 0 = 1) ∧             -- Maximum value is 1
  (f 1 = 5/6) ∧           -- Minimum value is 5/6
  (∫ x in (0)..(3/2), (1 - f x) = 9/64) -- Area of closed figure
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l770_77085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_recipe_servings_l770_77005

/-- 
Given a recipe that uses a certain amount of pasta to serve people, 
this theorem proves how many people the original recipe serves 
based on a scaled-up version.
-/
theorem original_recipe_servings 
  (original_pasta : ℝ) 
  (scaled_pasta : ℝ) 
  (scaled_servings : ℝ) 
  (h1 : original_pasta = 2) 
  (h2 : scaled_pasta = 10) 
  (h3 : scaled_servings = 35) :
  (original_pasta * scaled_servings) / scaled_pasta = 7 := by
  sorry

#check original_recipe_servings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_recipe_servings_l770_77005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_plus_primes_l770_77069

def first_100_even_sum : ℕ := 100 * (2 + 200) / 2

def first_100_odd_sum : ℕ := 100 * (1 + 199) / 2

def primes_odd_not_div_3 (n : ℕ) : Bool :=
  Nat.Prime n ∧ n ≤ 100 ∧ n % 2 = 1 ∧ n % 3 ≠ 0

def sum_primes_odd_not_div_3 : ℕ := 
  (Finset.filter (fun n => primes_odd_not_div_3 n) (Finset.range 101)).sum id

theorem sum_difference_plus_primes :
  (first_100_even_sum - first_100_odd_sum) + sum_primes_odd_not_div_3 = 1063 := by
  sorry

#eval (first_100_even_sum - first_100_odd_sum) + sum_primes_odd_not_div_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_plus_primes_l770_77069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_points_exist_l770_77043

-- Define the plane as a type
def Plane := ℝ × ℝ

-- Define a color type
inductive Color
| Red
| Blue

-- Define a coloring function
def coloring : Plane → Color := sorry

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Plane) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem same_color_points_exist :
  ∃ (p1 p2 : Plane), coloring p1 = coloring p2 ∧ distance p1 p2 = 2004 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_points_exist_l770_77043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l770_77008

/-- Given a point P (8m, 3) on the terminal side of angle α where cos α = -4/5, prove that m = -1/2 -/
theorem point_on_terminal_side (m : ℝ) (α : ℝ) : 
  (∃ P : ℝ × ℝ, P = (8*m, 3) ∧ P.1 = 8*m * Real.cos α ∧ P.2 = 8*m * Real.sin α) →
  Real.cos α = -4/5 →
  m = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l770_77008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_opposite_point_l770_77092

/-- Represents a circular path with a given circumference -/
structure CircularPath where
  circumference : ℝ
  circumference_pos : circumference > 0

/-- Represents a walker on the circular path -/
structure Walker where
  speed : ℝ
  speed_pos : speed > 0

/-- Calculates the meeting point of two walkers on a circular path -/
noncomputable def meetingPoint (path : CircularPath) (walker1 walker2 : Walker) : ℝ :=
  (walker1.speed * path.circumference) / (walker1.speed + walker2.speed)

/-- Theorem stating that two walkers with a 2:1 speed ratio meet opposite their starting point -/
theorem meet_opposite_point (path : CircularPath) (slow_walker fast_walker : Walker)
    (h_speed_ratio : fast_walker.speed = 2 * slow_walker.speed) :
    meetingPoint path slow_walker fast_walker = path.circumference / 2 := by
  sorry

#check meet_opposite_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_opposite_point_l770_77092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_leq_one_l770_77055

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < 2^a}

-- State the theorem
theorem subset_implies_a_leq_one (a : ℝ) :
  B a ⊆ A → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_leq_one_l770_77055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_calculation_l770_77074

/-- The depth of the well in feet -/
def well_depth : ℝ := 1332

/-- The total time from dropping the stone to hearing it hit the bottom, in seconds -/
def total_time : ℝ := 9.5

/-- The coefficient in the stone's descent equation (d = 20t^2) -/
def descent_coeff : ℝ := 20

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1000

/-- Theorem stating that the calculated well depth is approximately 1332 feet -/
theorem well_depth_calculation :
  ∃ (t₁ t₂ : ℝ),
    t₁ + t₂ = total_time ∧
    well_depth = descent_coeff * t₁^2 ∧
    t₂ = well_depth / sound_speed ∧
    |well_depth - 1332| < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_calculation_l770_77074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l770_77096

theorem log_problem (x : ℝ) (h : Real.log (x - 3) / Real.log 16 = 1/4) :
  Real.log x / Real.log 216 = 1/3 * (Real.log 5 / Real.log 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l770_77096
