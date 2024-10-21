import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l23_2336

noncomputable def f (a c x : ℝ) : ℝ := a * x^2 + c

theorem x0_value (a c x0 : ℝ) (ha : a ≠ 0) (hx0 : 0 ≤ x0 ∧ x0 ≤ 1) :
  (∫ (x : ℝ) in Set.Icc 0 1, f a c x) = f a c x0 → x0 = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l23_2336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ratio_theorem_l23_2374

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_ratio (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => (double_factorial (2*i + 1)) / (double_factorial (2*i + 2)))

noncomputable def lowest_terms_denominator (q : ℚ) : ℕ × ℕ :=
  let d := q.den
  let c := (2^(d.log2)) / (d / (2^(d.log2)))
  (c, d / (2^c))

theorem sum_ratio_theorem :
  let S := sum_ratio 1005
  let (c, d) := lowest_terms_denominator S
  (c * d : ℚ) / 10 = 1999 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ratio_theorem_l23_2374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l23_2300

-- Define the points
variable (A : Fin 11 → ℝ × ℝ) (B : ℝ × ℝ)

-- Define the conditions
def equal_segments (A : Fin 11 → ℝ × ℝ) : Prop :=
  ∀ i j : Fin 10, dist (A i) (A (i+1)) = dist (A j) (A (j+1))

def regular_triangle (A : Fin 11 → ℝ × ℝ) (B : ℝ × ℝ) : Prop :=
  dist (A 8) (A 10) = dist (A 8) B ∧ dist (A 8) (A 10) = dist (A 10) B

-- Define the angle function
noncomputable def angle (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem sum_of_angles (A : Fin 11 → ℝ × ℝ) (B : ℝ × ℝ) :
  equal_segments A →
  regular_triangle A B →
  angle B (A 0) (A 10) + angle B (A 2) (A 10) + angle B (A 3) (A 10) + angle B (A 4) (A 10) = 60 := by
  sorry

#check sum_of_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l23_2300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l23_2358

/-- Represents a parabola y² = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def focus (para : Parabola) : Point :=
  { x := para.p / 2, y := 0 }

noncomputable def directrix (para : Parabola) : Line :=
  { a := 1, b := 0, c := -para.p / 2 }

def is_on_parabola (para : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * para.p * pt.x

def is_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

noncomputable def chord_length (circle_center : Point) (circle_radius : ℝ) (chord_end1 chord_end2 : Point) : ℝ :=
  sorry -- Definition of chord length calculation

theorem parabola_chord_theorem (para : Parabola) 
  (P : Point) (h_P_on_parabola : is_on_parabola para P)
  (h_PF_perp_x : is_perpendicular 
    { a := P.x - (para.p / 2), b := P.y, c := 0 } 
    { a := 1, b := 0, c := 0 })
  (h_chord_length : chord_length 
    { x := para.p / 4, y := 0 } (para.p / 4) 
    { x := -para.p / 2, y := 0 } P = 2) :
  para.p = 2 * Real.sqrt 2 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l23_2358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_four_pow_positive_l23_2357

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (16 - (4 : ℝ)^x)

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ∈ Set.Ici 0 ∩ Set.Iio 4 :=
by sorry

-- Define the property that 4^x > 0 for all real x
theorem four_pow_positive : ∀ x : ℝ, (4 : ℝ)^x > 0 :=
by
  intro x
  apply Real.rpow_pos_of_pos
  norm_num

-- Note: Set.Ici 0 represents [0, ∞) and Set.Iio 4 represents (-∞, 4)
-- Their intersection is [0, 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_four_pow_positive_l23_2357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_expenditure_ratio_l23_2359

def income : ℕ := 19000
def savings : ℕ := 11400

def expenditure : ℕ := income - savings

def income_to_expenditure_ratio : ℚ := income / expenditure

theorem income_expenditure_ratio :
  income_to_expenditure_ratio = 95 / 38 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_expenditure_ratio_l23_2359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l23_2361

/-- Definition of the function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then 2^x else x^2 - 6*x + 9

/-- The solution set of f(x) > f(1) -/
def solution_set : Set ℝ :=
  {x | x < 1 ∨ x > 2}

/-- Theorem stating that the solution set is correct -/
theorem f_inequality_solution_set :
  ∀ x : ℝ, f x > f 1 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l23_2361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_16_to_6th_power_l23_2386

theorem fourth_root_of_16_to_6th_power : (16 : ℝ) ^ (1/4) ^ 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_16_to_6th_power_l23_2386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l23_2306

-- Define the ellipse parameters
variable (a b : ℝ)
variable (h₁ : a > 0)
variable (h₂ : b > 0)
variable (h₃ : a > b)

-- Define the point P on the left directrix
def P : ℝ × ℝ := (-3, 1)

-- Define the line l passing through P
def l (x y : ℝ) : Prop := 5 * x + 2 * y = 13

-- Define the reflection line
def reflection_line (y : ℝ) : Prop := y = -2

-- Define the eccentricity of the ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b / a)^2)

-- State the theorem
theorem ellipse_eccentricity :
  P.1 = -(a^2 / Real.sqrt (a^2 - b^2)) →
  l P.1 P.2 →
  (∃ (x y : ℝ), l x y ∧ reflection_line y ∧ x = -Real.sqrt (a^2 - b^2)) →
  eccentricity a b = Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l23_2306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l23_2349

def series_sum (n : ℕ) : ℚ :=
  if n % 2 = 0
  then (n + 1) / (2^(n + 2))
  else (n + 1) / (3^(n + 2))

def infinite_series_sum : ℚ := 73 / 96

theorem problem_statement (c d : ℕ) (hcd : Nat.Coprime c d) 
  (h : (c : ℚ) / d = ∑' n, series_sum n) : c + d = 169 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l23_2349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_catches_hares_l23_2377

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the game setup -/
structure GameSetup where
  A : Point
  B : Point
  C : Point
  D : Point
  squareSideLength : ℝ
  foxSpeed : ℝ
  hareSpeed : ℝ

/-- Represents a game strategy (not implemented) -/
def GameStrategy : Type := Unit

/-- Determines if the fox wins given a strategy (not implemented) -/
def fox_wins (_strategy : GameStrategy) : Prop := sorry

/-- Theorem stating the condition for the fox to catch both hares -/
theorem fox_catches_hares (game : GameSetup) : 
  game.squareSideLength = 1 ∧ 
  game.A = ⟨0, 0⟩ ∧ 
  game.B = ⟨0, 1⟩ ∧ 
  game.C = ⟨1, 1⟩ ∧ 
  game.D = ⟨1, 0⟩ ∧ 
  game.hareSpeed ≤ 1 →
  (∃ (_strategy : GameStrategy), fox_wins _strategy) ↔ game.foxSpeed ≥ 1 + Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_catches_hares_l23_2377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l23_2312

theorem factorial_divisibility (p : ℕ) (h : Nat.Prime p) : 
  ∃ k : ℕ, (Nat.factorial (p^2)) = k * (Nat.factorial p)^(p+1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l23_2312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_l23_2383

/-- Given complex numbers p, q, r forming an equilateral triangle with side length 24 and |p+q+r| = 48, prove that |pq + pr + qr| = 768 -/
theorem equilateral_triangle_complex (p q r : ℂ) : 
  (Complex.abs (p - q) = 24 ∧ Complex.abs (q - r) = 24 ∧ Complex.abs (r - p) = 24) → 
  Complex.abs (p + q + r) = 48 → 
  Complex.abs (p * q + p * r + q * r) = 768 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_l23_2383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_exists_l23_2376

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 8*x + 20

-- Define the circle
def on_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 9

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem shortest_distance_exists :
  ∃ (a x y : ℝ), y = parabola a ∧ on_circle x y ∧
  ∀ (b u v : ℝ), v = parabola b → on_circle u v →
    distance a (parabola a) x y ≤ distance b (parabola b) u v := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_exists_l23_2376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l23_2364

noncomputable def p : ℝ := Real.sqrt 2 + Real.sqrt 5 + Real.sqrt 10
noncomputable def q : ℝ := -Real.sqrt 2 + Real.sqrt 5 + Real.sqrt 10
noncomputable def r : ℝ := Real.sqrt 2 - Real.sqrt 5 + Real.sqrt 10
noncomputable def s : ℝ := -Real.sqrt 2 - Real.sqrt 5 + Real.sqrt 10

theorem sum_of_reciprocals_squared :
  (1/p + 1/q + 1/r + 1/s)^2 = 128/45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l23_2364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l23_2354

-- Define the function f
noncomputable def f (a : ℕ) (x : ℝ) : ℝ := (x^2 - a*x + a + 1) * Real.exp x

-- Define the property of having only one extremum in the interval (1, 3)
def has_one_extremum (a : ℕ) : Prop :=
  ∃! x : ℝ, 1 < x ∧ x < 3 ∧ deriv (f a) x = 0

-- State the theorem
theorem tangent_line_equation (a : ℕ) (h : has_one_extremum a) :
  ∃ m b : ℝ, m = 1 ∧ b = 6 ∧
  ∀ x y : ℝ, y = m * x + b ↔ y - f a 0 = deriv (f a) 0 * (x - 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l23_2354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_tank_depth_l23_2384

/-- The volume of a right frustum in cubic centimeters -/
noncomputable def frustum_volume (r1 : ℝ) (r2 : ℝ) (h : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (r1^2 + r2^2 + r1*r2)

/-- Theorem: The depth of a right frustum oil tank with given properties -/
theorem oil_tank_depth (top_edge : ℝ) (bottom_edge : ℝ) (volume_liters : ℝ) :
  top_edge = 60 →
  bottom_edge = 40 →
  volume_liters = 190 →
  ∃ (depth : ℝ), depth = 75 ∧ 
    frustum_volume (top_edge/2) (bottom_edge/2) depth = volume_liters * 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_tank_depth_l23_2384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_sum_unit_vectors_l23_2380

/-- Given two unit vectors a and b with an angle of 120° between them, 
    prove that (a + b) · a = 1/2 -/
theorem dot_product_sum_unit_vectors (a b : ℝ → ℝ → ℝ → ℝ) : 
  (∀ x y z, ‖a x y z‖ = 1) → 
  (∀ x y z, ‖b x y z‖ = 1) → 
  (∀ x y z, (a x y z) • (b x y z) = -1/2) → 
  (∀ x y z, ((a x y z) + (b x y z)) • (a x y z) = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_sum_unit_vectors_l23_2380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_not_shorter_than_line_l23_2388

-- Define a line in Euclidean geometry
def Line : Type := Unit

-- Define a ray in Euclidean geometry
def Ray : Type := Unit

-- Define the concept of "shorter than" for geometric objects
def shorter_than (a b : Type) : Prop := False

-- Theorem stating that it's not true that a ray is shorter than a line
theorem ray_not_shorter_than_line :
  ¬ (∀ (r : Ray) (l : Line), shorter_than Ray Line) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_not_shorter_than_line_l23_2388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_mn_distance_l23_2307

noncomputable section

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the isosceles property
def isIsosceles (t : Triangle) : Prop :=
  let (xa, ya) := t.A
  let (xb, yb) := t.B
  let (xc, yc) := t.C
  (xa - xb)^2 + (ya - yb)^2 = (xa - xc)^2 + (ya - yc)^2

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define points P and R
noncomputable def P (t : Triangle) : ℝ × ℝ :=
  let (xb, yb) := t.B
  let (xc, yc) := t.C
  ((3 * xb + 9 * xc) / 12, (3 * yb + 9 * yc) / 12)

noncomputable def R (t : Triangle) : ℝ × ℝ :=
  let (xb, yb) := t.B
  let (xc, yc) := t.C
  ((9 * xb + 3 * xc) / 12, (9 * yb + 3 * yc) / 12)

-- Define midpoints S and T
noncomputable def S (t : Triangle) : ℝ × ℝ :=
  let (xa, ya) := t.A
  let (xb, yb) := t.B
  ((xa + xb) / 2, (ya + yb) / 2)

noncomputable def T (t : Triangle) : ℝ × ℝ :=
  let (xa, ya) := t.A
  let (xc, yc) := t.C
  ((xa + xc) / 2, (ya + yc) / 2)

-- Define the perpendicular foot M and N
noncomputable def M (t : Triangle) : ℝ × ℝ := sorry
noncomputable def N (t : Triangle) : ℝ × ℝ := sorry

theorem isosceles_triangle_mn_distance (t : Triangle) 
  (h1 : isIsosceles t)
  (h2 : distance t.A t.B = 10)
  (h3 : distance t.B t.C = 12) :
  distance (M t) (N t) = (10 * Real.sqrt 13) / 13 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_mn_distance_l23_2307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l23_2365

/-- A line in 2D space. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) is on a given line. -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Checks if a line intersects a given quadrant. -/
def Line.intersectsQuadrant (l : Line) (q : ℕ) : Prop :=
  match q with
  | 1 => ∃ x y, x > 0 ∧ y > 0 ∧ l.contains x y
  | 2 => ∃ x y, x < 0 ∧ y > 0 ∧ l.contains x y
  | 4 => ∃ x y, x > 0 ∧ y < 0 ∧ l.contains x y
  | _ => False

/-- The x-intercept of a line. -/
noncomputable def Line.xIntercept (l : Line) : ℝ := -l.c / l.a

/-- The y-intercept of a line. -/
noncomputable def Line.yIntercept (l : Line) : ℝ := -l.c / l.b

theorem line_equation (l : Line) :
  l.contains 1 2 ∧
  l.intersectsQuadrant 1 ∧
  l.intersectsQuadrant 2 ∧
  l.intersectsQuadrant 4 ∧
  l.xIntercept + l.yIntercept = 6 →
  (l.a = 2 ∧ l.b = 1 ∧ l.c = -4) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l23_2365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_l23_2353

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def min_sum_at_six_condition (d : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 →
    sum_of_arithmetic_sequence (-20) d 6 ≤ sum_of_arithmetic_sequence (-20) d n

theorem necessary_but_not_sufficient_condition :
  (∀ d : ℝ, min_sum_at_six_condition d → 3 < d ∧ d < 5) ∧
  ¬(∀ d : ℝ, 3 < d ∧ d < 5 → min_sum_at_six_condition d) :=
by
  sorry

#check necessary_but_not_sufficient_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_l23_2353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_edge_length_l23_2331

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- The radius of the hemisphere -/
  hemisphere_radius : ℝ
  /-- The height of the pyramid -/
  pyramid_height : ℝ
  /-- The hemisphere is tangent to one side face of the pyramid -/
  tangent_to_face : Unit

/-- The edge length of the base of the pyramid -/
noncomputable def base_edge_length (p : PyramidWithHemisphere) : ℝ :=
  2 * Real.sqrt 55

/-- Theorem stating the edge length of the base of the pyramid -/
theorem pyramid_base_edge_length (p : PyramidWithHemisphere) 
  (h1 : p.hemisphere_radius = 3)
  (h2 : p.pyramid_height = 8) :
  base_edge_length p = 2 * Real.sqrt 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_edge_length_l23_2331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l23_2330

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * (b-1) * x^2 + b^2 * x

noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := x^2 + (b-1) * x + b^2

theorem extreme_value_at_one (b : ℝ) :
  (∀ x, f' b x = 0 → x = 1) → b = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l23_2330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_solution_l23_2399

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the value of a 6-digit number -/
def SixDigitValue (a b c d e f : Digit) : ℕ :=
  100000 * a.val + 10000 * b.val + 1000 * c.val + 100 * d.val + 10 * e.val + f.val

theorem cryptarithm_solution :
  ∃! (B I D F O R : Digit),
    (B ≠ I) ∧ (B ≠ D) ∧ (B ≠ F) ∧ (B ≠ O) ∧ (B ≠ R) ∧
    (I ≠ D) ∧ (I ≠ F) ∧ (I ≠ O) ∧ (I ≠ R) ∧
    (D ≠ F) ∧ (D ≠ O) ∧ (D ≠ R) ∧
    (F ≠ O) ∧ (F ≠ R) ∧
    (O ≠ R) ∧
    3 * (SixDigitValue B I D F O R) = 4 * (SixDigitValue F O R B I D) ∧
    B = ⟨5, by norm_num⟩ ∧ 
    I = ⟨7, by norm_num⟩ ∧ 
    D = ⟨1, by norm_num⟩ ∧ 
    F = ⟨4, by norm_num⟩ ∧ 
    O = ⟨2, by norm_num⟩ ∧ 
    R = ⟨8, by norm_num⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_solution_l23_2399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_1_hyperbola_standard_equation_2_hyperbola_standard_equation_3_l23_2301

/-- The standard equation of a hyperbola given its parameters -/
def hyperbola_equation (a b : ℝ) (x_axis : Bool) : ℝ → ℝ → Prop :=
  if x_axis then
    λ x y ↦ x^2 / a^2 - y^2 / b^2 = 1
  else
    λ x y ↦ y^2 / a^2 - x^2 / b^2 = 1

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem: Standard equation of hyperbola with given parameters -/
theorem hyperbola_standard_equation_1 (a b : ℝ) (h1 : a = 3) (h2 : b = 4) :
  hyperbola_equation a b true = λ x y ↦ x^2 / 9 - y^2 / 16 = 1 :=
by sorry

/-- Theorem: Standard equation of hyperbola with given foci and distance difference -/
theorem hyperbola_standard_equation_2 (f1 f2 diff : ℝ) 
  (h1 : f1 = 10) (h2 : f2 = -10) (h3 : diff = 16) :
  ∃ a b : ℝ, hyperbola_equation a b false = λ x y ↦ x^2 / 64 - y^2 / 36 = 1 ∧
    ∀ x y : ℝ, (hyperbola_equation a b false x y) → 
      |distance x y 0 f1 - distance x y 0 f2| = diff :=
by sorry

/-- Theorem: Standard equation of hyperbola with given foci and passing point -/
theorem hyperbola_standard_equation_3 (f px py : ℝ) 
  (h1 : f = 5) (h2 : px = 4 * Real.sqrt 3 / 3) (h3 : py = 2 * Real.sqrt 3) :
  ∃ a b : ℝ, hyperbola_equation a b false = λ x y ↦ y^2 / 9 - x^2 / 16 = 1 ∧
    hyperbola_equation a b false px py ∧
    (∀ x y : ℝ, hyperbola_equation a b false x y → 
      |distance x y 0 f - distance x y 0 (-f)| = 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_1_hyperbola_standard_equation_2_hyperbola_standard_equation_3_l23_2301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_calculation_trip_time_is_45_l23_2343

theorem trip_time_calculation (highway_distance : ℝ) (coastal_distance : ℝ) 
  (coastal_time : ℝ) (highway_speed_multiplier : ℝ) : ℝ :=
  by
  -- Define the conditions
  have h1 : highway_distance = 50 := by sorry
  have h2 : coastal_distance = 10 := by sorry
  have h3 : coastal_time = 20 := by sorry
  have h4 : highway_speed_multiplier = 4 := by sorry

  -- Calculate the total time
  let coastal_speed : ℝ := coastal_distance / coastal_time
  let highway_speed : ℝ := coastal_speed * highway_speed_multiplier
  let highway_time : ℝ := highway_distance / highway_speed
  let total_time : ℝ := coastal_time + highway_time

  -- Return the total time
  exact total_time

-- Proof that the total time is 45 minutes
theorem trip_time_is_45 (highway_distance : ℝ) (coastal_distance : ℝ) 
  (coastal_time : ℝ) (highway_speed_multiplier : ℝ) :
  trip_time_calculation highway_distance coastal_distance coastal_time highway_speed_multiplier = 45 :=
  by
  -- Apply the conditions
  have h1 : highway_distance = 50 := by sorry
  have h2 : coastal_distance = 10 := by sorry
  have h3 : coastal_time = 20 := by sorry
  have h4 : highway_speed_multiplier = 4 := by sorry

  -- Unfold the definition of trip_time_calculation
  unfold trip_time_calculation

  -- Perform the calculation
  simp [h1, h2, h3, h4]
  
  -- The final proof step
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_calculation_trip_time_is_45_l23_2343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l23_2381

noncomputable def expression (x y : ℝ) : ℝ :=
  (1.2 * x^4 + 4 * y^3 / 3) * (0.86)^3 - 
  (Real.sqrt 0.1)^3 / (0.86 * x^2 * y^2) + 
  0.086 + (0.1)^2 * (2 * x^3 - 3 * y^4 / 2)

theorem expression_evaluation :
  let x : ℝ := 1.5
  let y : ℝ := -2
  abs (expression x y + 3.012286) < 0.000001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l23_2381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_angle_l23_2398

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-2, 0)

-- Define the right focus
def right_focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the right focus
def line_through_right_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 2)

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Main theorem
theorem ellipse_intersection_slope_angle :
  ∀ (k : ℝ), k ≠ 0 →
  (∃ (x1 y1 x2 y2 : ℝ),
    ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧
    line_through_right_focus k x1 y1 ∧
    line_through_right_focus k x2 y2 ∧
    distance x1 y1 x2 y2 = Real.sqrt 6) →
  k = 1 ∨ k = -1 := by
  sorry

#check ellipse_intersection_slope_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_angle_l23_2398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_sum_l23_2345

/-- The area of a regular hexagon with side length 3 can be expressed as √p + √q where p and q are positive integers, and p + q = 364 -/
theorem regular_hexagon_area_sum : 
  ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ (Real.sqrt (p : ℝ) + Real.sqrt (q : ℝ)) ^ 2 = 27 * Real.sqrt 3 ∧ p + q = 364 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_sum_l23_2345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l23_2335

/-- The equation of a hyperbola with given foci and eccentricity -/
theorem hyperbola_equation (c a b : ℝ) (h1 : c = 5) (h2 : a = 4) (h3 : b^2 = c^2 - a^2) :
  (fun x y => x^2 / a^2 - y^2 / b^2 = 1) =
  (fun x y => x^2 / 16 - y^2 / 9 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l23_2335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l23_2372

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 4)

-- Define the domain of f
def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ Real.pi / 8 + k * Real.pi / 2

-- Theorem statement
theorem f_properties :
  -- 1. Domain of f
  (∀ x : ℝ, f x ≠ 0 ↔ domain x) ∧
  -- 2. Smallest positive period of f
  (∀ x : ℝ, f (x + Real.pi / 2) = f x) ∧
  (∀ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) → p ≥ Real.pi / 2) ∧
  -- 3. Value of α
  (∀ α : ℝ, 0 < α ∧ α < Real.pi / 4 ∧ f (α / 2) = 2 * Real.cos (2 * α) → α = Real.pi / 12) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l23_2372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_l23_2385

theorem cos_alpha_minus_beta (α β γ : ℝ) 
  (h1 : Real.sin α + Real.sin β + Real.sin γ = 0)
  (h2 : Real.cos α + Real.cos β + Real.cos γ = 0) : 
  Real.cos (α - β) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_l23_2385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_in_xoz_plane_l23_2339

/-- Given two points A and B in 3D space, this theorem proves that their midpoint lies in the xOz plane. -/
theorem midpoint_in_xoz_plane (A B : ℝ × ℝ × ℝ) 
  (hA : A = (1, -1, 1)) (hB : B = (3, 1, 5)) : 
  let M := ((A.1 + B.1) / 2, (A.2.1 + B.2.1) / 2, (A.2.2 + B.2.2) / 2)
  M.2.1 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_in_xoz_plane_l23_2339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l23_2373

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 2}
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 1}
  let chord := Set.inter line circle
  ∃ a b : ℝ × ℝ, a ∈ chord ∧ b ∈ chord ∧ a ≠ b ∧ 
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l23_2373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_simplest_l23_2368

/-- A square root is considered simple if it cannot be simplified further. -/
noncomputable def IsSimpleSquareRoot (x : ℝ) : Prop :=
  ∀ y z : ℝ, y * y = x → z * z = x → y = z ∨ y = -z

/-- The given square roots to compare -/
noncomputable def GivenSquareRoots : List ℝ := [Real.sqrt 4, Real.sqrt 5, Real.sqrt 8, Real.sqrt (5/2)]

/-- Theorem stating that √5 is the simplest among the given square roots -/
theorem sqrt_5_simplest :
  ∃ (x : ℝ), x ∈ GivenSquareRoots ∧ IsSimpleSquareRoot x ∧
  ∀ y ∈ GivenSquareRoots, IsSimpleSquareRoot y → y = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_simplest_l23_2368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_on_directrix_l23_2363

/-- Parabola structure -/
structure Parabola where
  t : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ y^2 = 4 * t * x

/-- Tangent to a parabola -/
def Tangent (p : Parabola) : Type :=
  { l : ℝ → ℝ → Prop // ∃ u : ℝ, ∀ x y, l x y ↔ y = (p.t / u) * x + u }

/-- Triangle formed by three tangents to a parabola -/
structure TangentTriangle (p : Parabola) where
  t₁ : Tangent p
  t₂ : Tangent p
  t₃ : Tangent p

/-- Orthocenter of a triangle -/
noncomputable def Orthocenter (p : Parabola) (tri : TangentTriangle p) : ℝ × ℝ :=
  sorry

/-- The theorem to be proved -/
theorem orthocenter_on_directrix (p : Parabola) (tri : TangentTriangle p) :
  (Orthocenter p tri).1 = -p.t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_on_directrix_l23_2363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_problem_l23_2325

theorem hcf_problem (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A ≥ B) 
  (h4 : Nat.lcm A B = Nat.gcd A B * 21 * 23) (h5 : A = 460) : 
  Nat.gcd A B = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_problem_l23_2325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_solutions_l23_2315

open Real

-- Define the triangle properties
def isIsoscelesTriangle (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), a = Real.sin x ∧ b = Real.sin x ∧ c = Real.cos (8 * x) ∧
  (a = b) ∧ (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Define the vertex angle condition
def hasVertexAngle3x (x : ℝ) : Prop :=
  3 * x = 180 - 2 * x

-- Theorem statement
theorem isosceles_triangle_solutions :
  ∀ x : ℝ, 0 < x ∧ x < 90 →
  isIsoscelesTriangle x ∧ hasVertexAngle3x x →
  x = 15 ∨ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_solutions_l23_2315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l23_2318

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the line function
def line (x y : ℝ) : ℝ := x - y - 2

-- Theorem statement
theorem min_distance_curve_to_line :
  ∃ (P : ℝ × ℝ), P.1 > 0 ∧ P.2 = f P.1 ∧
  (∀ (Q : ℝ × ℝ), Q.1 > 0 → Q.2 = f Q.1 →
    Real.sqrt 2 ≤ |line Q.1 Q.2| / Real.sqrt (1^2 + (-1)^2)) ∧
  |line P.1 P.2| / Real.sqrt (1^2 + (-1)^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l23_2318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_y_axis_intersection_length_l23_2342

/-- A circle passing through three given points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of a point on a 2D plane -/
def Point := ℝ × ℝ

/-- The circle passes through the given points -/
def passes_through (c : Circle) (p : Point) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- A point lies on the y-axis -/
def on_y_axis (p : Point) : Prop :=
  p.1 = 0

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The main theorem -/
theorem circle_y_axis_intersection_length :
  ∃ (c : Circle) (m n : Point),
    passes_through c (1, 3) ∧
    passes_through c (4, 2) ∧
    passes_through c (1, -7) ∧
    on_y_axis m ∧
    on_y_axis n ∧
    passes_through c m ∧
    passes_through c n ∧
    distance m n = 4 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_y_axis_intersection_length_l23_2342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l23_2322

open Real

-- Define the statements
def statement1 : Prop := ∀ x ∈ Set.Ioo 0 (π/2), StrictMono (λ x => tan x)

def statement2 : Prop := ∀ θ ∈ Set.Ioo (π/2) π, tan (θ/2) > cos (θ/2)

def statement3 : Prop := ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  sin a = a ∧ sin b = b ∧ sin c = c

def statement4 : Prop := ∀ x : ℝ, cos x ^ 2 + sin x ≥ -1

def statement5 : Prop := ∀ x ∈ Set.Ioo 0 π, 
  ∀ ε > 0, sin (x - π/2) > sin (x + ε - π/2)

-- Theorem stating which statements are correct
theorem correct_statements : 
  statement2 ∧ statement4 ∧ ¬statement1 ∧ ¬statement3 ∧ ¬statement5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l23_2322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_height_in_meters_l23_2311

/-- Martha's height in feet -/
def martha_height_feet : ℝ := 5

/-- Conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Number of centimeters in a meter -/
def cm_per_meter : ℕ := 100

/-- Rounds a real number to three decimal places -/
noncomputable def round_to_three_decimals (x : ℝ) : ℝ :=
  ⌊x * 1000 + 0.5⌋ / 1000

/-- Theorem stating Martha's height in meters -/
theorem martha_height_in_meters :
  round_to_three_decimals (martha_height_feet * inches_per_foot * inch_to_cm / cm_per_meter) = 1.524 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_height_in_meters_l23_2311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirts_produced_machine_output_l23_2329

/-- Given a machine that produces shirts at a constant rate, 
    calculate the total number of shirts produced in a given time. -/
theorem shirts_produced (rate : ℕ) (time : ℕ) : rate * time = rate * time :=
by rfl

/-- Prove that a machine producing 2 shirts per minute for 6 minutes makes 12 shirts. -/
theorem machine_output : 2 * 6 = 12 :=
by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirts_produced_machine_output_l23_2329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l23_2360

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (π / 4) (π / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (π / 4) (π / 2) → f y ≥ f x) ∧
  f x = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l23_2360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_equals_120_l23_2313

theorem factorial_ratio_equals_120 (n : ℕ) (hn : n > 3) :
  Nat.factorial n / Nat.factorial (n - 3) = 120 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_equals_120_l23_2313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_derivative_property_l23_2308

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g
def g (x : ℝ) : ℝ := x * f x

-- State the theorem
theorem odd_function_derivative_property 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_deriv : ∀ x > 0, x * (deriv f x) < f (-x)) :
  {x : ℝ | g 1 > g (1 - 2 * x)} = Set.Ioi 1 ∪ Set.Iic 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_derivative_property_l23_2308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l23_2371

-- Define the basic geometric concepts
structure Line
structure Plane

-- Define geometric relationships
axiom perpendicular : Line → Plane → Prop
axiom parallel : Plane → Plane → Prop
axiom intersect : Line → Line → Prop
axiom in_plane : Line → Plane → Prop
axiom line_of_intersection : Plane → Plane → Line

-- Define the propositions
def proposition_1 (l1 l2 : Line) (p1 p2 : Plane) : Prop :=
  in_plane l1 p1 ∧ in_plane l2 p1 ∧ parallel p1 p2 → parallel p1 p2

def proposition_2 (l1 l2 l3 : Line) : Prop :=
  perpendicular l1 (Plane.mk) ∧ perpendicular l2 (Plane.mk) → parallel (Plane.mk) (Plane.mk)

def proposition_3 (l : Line) (p1 p2 : Plane) : Prop :=
  perpendicular l p1 ∧ in_plane l p2 → perpendicular l p1

def proposition_4 (l : Line) (p1 p2 : Plane) : Prop :=
  perpendicular l p1 ∧ in_plane l p1 ∧ ¬perpendicular l (Plane.mk) →
  ¬perpendicular l p2

theorem geometric_propositions :
  (∀ l1 l2 p1 p2, ¬proposition_1 l1 l2 p1 p2) ∧
  (∀ l1 l2 l3, ¬proposition_2 l1 l2 l3) ∧
  (∀ l p1 p2, proposition_3 l p1 p2) ∧
  (∀ l p1 p2, proposition_4 l p1 p2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l23_2371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l23_2326

noncomputable section

/-- Line l: ax + (1/a)y - 1 = 0 -/
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x + (1/a) * y - 1 = 0

/-- Circle O: x^2 + y^2 = 1 -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Point A: intersection of line l with x-axis -/
noncomputable def point_A (a : ℝ) : ℝ × ℝ := (1/a, 0)

/-- Point B: intersection of line l with y-axis -/
noncomputable def point_B (a : ℝ) : ℝ × ℝ := (0, a)

/-- Length of AB -/
noncomputable def length_AB (a : ℝ) : ℝ := Real.sqrt (a^2 + (1/a)^2)

/-- Length of CD -/
noncomputable def length_CD (a : ℝ) : ℝ := 2 * Real.sqrt (1 - 1 / (a^2 + (1/a)^2))

/-- Area of triangle AOB -/
noncomputable def area_AOB (a : ℝ) : ℝ := (1/2) * a * (1/a)

theorem line_circle_intersection (a : ℝ) (h : a > 0) :
  (∀ a > 0, area_AOB a = 1/2) ∧
  (∀ a > 0, length_AB a ≥ length_CD a) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l23_2326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_shuffle_order_perfect_shuffle_all_n_l23_2356

/-- Perfect shuffle function -/
def perfectShuffle (n : ℕ) (seq : Fin (2 * n) → ℕ) : Fin (2 * n) → ℕ :=
  λ i ↦ if i.val % 2 = 0 then seq ⟨i.val / 2, by sorry⟩ else seq ⟨n + (i.val - 1) / 2, by sorry⟩

/-- Theorem stating that the sequence returns to original order after φ(2n+1) shuffles -/
theorem perfect_shuffle_order (n : ℕ) :
  ∃ k : ℕ, k = Nat.totient (2 * n + 1) ∧
    (∀ seq : Fin (2 * n) → ℕ, (Nat.iterate (perfectShuffle n) k seq) = seq) :=
  sorry

/-- Corollary stating that the theorem holds for all natural numbers n -/
theorem perfect_shuffle_all_n :
  ∀ n : ℕ, ∃ k : ℕ, k = Nat.totient (2 * n + 1) ∧
    (∀ seq : Fin (2 * n) → ℕ, (Nat.iterate (perfectShuffle n) k seq) = seq) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_shuffle_order_perfect_shuffle_all_n_l23_2356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_perpendicular_l23_2304

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 + x - 6*y + m = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1*x2 + y1*y2 = 0

-- Theorem statement
theorem circle_intersection_perpendicular (m : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    circle_eq x1 y1 m ∧
    circle_eq x2 y2 m ∧
    line_eq x1 y1 ∧
    line_eq x2 y2 ∧
    perpendicular x1 y1 x2 y2) →
  m = 3 :=
by
  sorry

#check circle_intersection_perpendicular

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_perpendicular_l23_2304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l23_2389

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

-- State the theorem
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = Real.pi ∧ ∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (x : ℝ), |f x| ≤ 4) ∧
  (∃ (x₁ x₂ : ℝ), f x₁ = 4 ∧ f x₂ = -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l23_2389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l23_2355

-- Define the * operation
noncomputable def star (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := star 1 x

-- Theorem statement
theorem f_expression : ∀ x : ℝ, f x = if x ≥ 1 then 1 else x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l23_2355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_l23_2375

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  side_length : ℝ

/-- Calculates the position after diagonal movement -/
noncomputable def diagonal_position (s : Square) (distance : ℝ) : Point :=
  { x := distance / (s.side_length * Real.sqrt 2) * s.side_length,
    y := distance / (s.side_length * Real.sqrt 2) * s.side_length }

/-- Calculates the final position after the turn and second movement -/
noncomputable def final_position (s : Square) (diag_dist : ℝ) (turn_dist : ℝ) : Point :=
  let pos := diagonal_position s diag_dist
  { x := pos.x + turn_dist,
    y := pos.y }

/-- Calculates the average distance to all sides of the square -/
noncomputable def average_distance_to_sides (s : Square) (p : Point) : ℝ :=
  let left := p.x
  let bottom := p.y
  let right := s.side_length - p.x
  let top := s.side_length - p.y
  (left + bottom + right + top) / 4

/-- The main theorem to prove -/
theorem lemming_average_distance (s : Square) (diag_dist : ℝ) (turn_dist : ℝ) :
  s.side_length = 12 → diag_dist = 7 → turn_dist = 4 →
  average_distance_to_sides s (final_position s diag_dist turn_dist) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_l23_2375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_correct_l23_2347

def sequencePattern : ℕ → ℕ
| 1 => 10
| 2 => 210
| 3 => 3210
| 4 => 43210
| 5 => 54321
| n => sorry  -- We don't define the general term here

theorem tenth_term_is_correct : sequencePattern 10 = 109876543210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_correct_l23_2347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adrian_greater_probability_l23_2362

-- Define the intervals for Natasha and Adrian
def natasha_interval : Set ℝ := Set.Icc 0 1524
def adrian_interval : Set ℝ := Set.Icc 0 3048

-- Define the probability space
def Ω : Type := ℝ × ℝ

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event where Adrian's number is greater than Natasha's
def event : Set Ω := {ω : Ω | ω.2 > ω.1}

-- State the theorem
theorem adrian_greater_probability :
  P {ω : Ω | ω.1 ∈ natasha_interval ∧ ω.2 ∈ adrian_interval ∧ ω.2 > ω.1} / 
  P {ω : Ω | ω.1 ∈ natasha_interval ∧ ω.2 ∈ adrian_interval} = 3/4 := by
  sorry

#check adrian_greater_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adrian_greater_probability_l23_2362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l23_2396

/-- The range of m given the conditions in the problem -/
theorem range_of_m (x m : ℝ) : 
  (∀ x : ℝ, |1 - (x - 1) / 3| ≤ 2) →
  (∀ x : ℝ, (x - 1 + m) * (x - 1 - m) ≤ 0) →
  m > 0 →
  (∀ x : ℝ, (x - 1 + m) * (x - 1 - m) ≤ 0 → |1 - (x - 1) / 3| ≤ 2) →
  (∃ x : ℝ, |1 - (x - 1) / 3| ≤ 2 ∧ (x - 1 + m) * (x - 1 - m) > 0) →
  0 < m ∧ m ≤ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l23_2396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_neg3_4_l23_2324

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 ∧ y ≥ 0 then Real.pi + Real.arctan (y / x)
           else if x < 0 ∧ y < 0 then -Real.pi + Real.arctan (y / x)
           else if x = 0 ∧ y > 0 then Real.pi / 2
           else if x = 0 ∧ y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, θ)

theorem rectangular_to_polar_neg3_4 :
  let (r, θ) := rectangular_to_polar (-3) 4
  r = 5 ∧
  θ = Real.pi - Real.arctan (4 / 3) ∧
  r > 0 ∧
  0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_neg3_4_l23_2324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_m_l23_2321

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (abs (x + m)) / Real.log 2

-- State the theorem
theorem symmetric_function_m (m : ℝ) :
  (∀ x : ℝ, f m x = f m (2 - x)) → m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_m_l23_2321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_inequality_l23_2351

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1)

-- Part 1
theorem f_monotonicity :
  ∀ x > 0, x ≠ 1 → (deriv (f 1) x > 0 ↔ x < 1) ∧ (deriv (f 1) x < 0 ↔ x > 1) := by sorry

-- Part 2
theorem f_inequality (a : ℝ) :
  (∀ x ≥ 1, f a x ≤ Real.log x / (x + 1)) ↔ a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_inequality_l23_2351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l23_2393

theorem product_remainder (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l23_2393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l23_2350

def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x^2)}
def B : Set ℝ := {y | y > 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l23_2350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l23_2390

/-- An odd function f on [-a,a] where a > 0 -/
def OddFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x, x ∈ Set.Icc (-a) a → f (-x) = -f x

theorem odd_function_properties (f : ℝ → ℝ) (a : ℝ) (m : ℝ) 
    (h_odd : OddFunction f a)
    (h_def : ∀ x ∈ Set.Icc (-a) 0, f x = m * Real.exp (-2*x) + Real.exp (-x)) :
  (∀ x ∈ Set.Icc 0 a, f x = Real.exp (2*x) - Real.exp x) ∧
  (∃ M : ℝ, M = Real.exp a * (Real.exp a - 1) ∧
    ∀ x ∈ Set.Icc 0 a, f x ≤ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l23_2390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_46_multiples_l23_2302

theorem unique_number_with_46_multiples : ∃! n : ℕ, 
  (n > 0) ∧ 
  (∃ k : ℕ, k = 46 ∧ 
    (∀ m : ℕ, (10 ≤ m * n ∧ m * n ≤ 100) ↔ (1 ≤ m ∧ m ≤ k))) ∧
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_46_multiples_l23_2302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l23_2334

/-- The ellipse with equation x^2/25 + y^2/16 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 16 = 1}

/-- The foci of the ellipse -/
noncomputable def F₁ : ℝ × ℝ := sorry
noncomputable def F₂ : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ellipse_property :
  ∀ P ∈ Ellipse, distance P F₁ + distance P F₂ = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l23_2334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_equals_nine_l23_2352

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The value of s -/
noncomputable def s : ℝ := 1 / Real.sqrt 2

/-- The nth term of the series -/
noncomputable def term (n : ℕ) : ℝ :=
  10 * s^(2*n)

/-- The sum of the series -/
noncomputable def w : ℝ :=
  1 + ∑' n, floor (term n)

/-- The theorem to prove -/
theorem w_equals_nine : w = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_equals_nine_l23_2352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_journey_average_speed_l23_2310

/-- Represents the train's journey with distances and times -/
structure TrainJourney where
  distance1 : ℝ
  time1 : ℝ
  stop1 : ℝ
  distance2 : ℝ
  time2 : ℝ
  stop2 : ℝ
  distance3 : ℝ
  time3 : ℝ

/-- Calculates the overall average speed of the train journey -/
noncomputable def overallAverageSpeed (journey : TrainJourney) : ℝ :=
  let totalDistance := journey.distance1 + journey.distance2 + journey.distance3
  let totalTime := journey.time1 + journey.time2 + journey.time3 + (journey.stop1 + journey.stop2) / 60
  totalDistance / totalTime

/-- The specific train journey described in the problem -/
def specificJourney : TrainJourney := {
  distance1 := 250
  time1 := 2
  stop1 := 30
  distance2 := 200
  time2 := 2
  stop2 := 20
  distance3 := 150
  time3 := 1.5
}

/-- Theorem stating that the overall average speed of the specific journey is approximately 94.74 km/h -/
theorem specific_journey_average_speed :
  abs (overallAverageSpeed specificJourney - 94.74) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_journey_average_speed_l23_2310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sequence_sequence_property_l23_2378

def sequence_a : ℕ → ℤ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 7
  | (n + 3) => sequence_a (n + 2) -- This is still a placeholder definition

theorem odd_sequence (n : ℕ) (h : n > 1) : 
  ∃ k : ℤ, sequence_a n = 2 * k + 1 :=
by
  sorry

theorem sequence_property (n : ℕ) (h : n ≥ 2) :
  -1/2 < (sequence_a (n + 1) : ℚ) - (sequence_a n ^ 2 : ℚ) / (sequence_a (n - 1) : ℚ) ∧ 
  (sequence_a (n + 1) : ℚ) - (sequence_a n ^ 2 : ℚ) / (sequence_a (n - 1) : ℚ) ≤ 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sequence_sequence_property_l23_2378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equality_condition_l23_2397

/-- Definition of the piecewise function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

/-- Theorem stating the condition for f(1 - a) = f(1 + a) -/
theorem f_equality_condition (a : ℝ) (h : a ≠ 0) :
  f a (1 - a) = f a (1 + a) ↔ a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equality_condition_l23_2397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_equivalence_l23_2337

-- Define the points
variable (A A' B B' C C' : EuclideanPlane) 

-- Define the angle function
noncomputable def angle (p q r : EuclideanPlane) : ℝ := sorry

-- State the theorem
theorem angle_equality_equivalence (A A' B B' C C' : EuclideanPlane) :
  angle A' A C = angle A B B' ↔ angle A C' C = angle A A' B :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_equivalence_l23_2337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l23_2366

-- Define set A
def A : Set ℝ := {x | x ≤ 4}

-- Define set B
def B : Set ℝ := {x | x ≥ 1/2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (1/2) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l23_2366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_theorem_l23_2333

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a transformation in the xy-plane -/
structure Transformation where
  apply : Point → Point

/-- Clockwise rotation by 45° around (2, 3) -/
noncomputable def rotate45 : Transformation :=
  { apply := λ p =>
      let dx := p.x - 2
      let dy := p.y - 3
      { x := 2 + (dx * Real.sqrt 2 / 2 + dy * Real.sqrt 2 / 2)
        y := 3 + (-dx * Real.sqrt 2 / 2 + dy * Real.sqrt 2 / 2) } }

/-- Scaling by factor 2 with respect to origin -/
def scale2 : Transformation :=
  { apply := λ p => { x := 2 * p.x, y := 2 * p.y } }

/-- Reflection about y = x -/
def reflectYEqX : Transformation :=
  { apply := λ p => { x := p.y, y := p.x } }

/-- Composition of transformations -/
def compose (t1 t2 : Transformation) : Transformation :=
  { apply := λ p => t2.apply (t1.apply p) }

theorem point_transformation_theorem (Q : Point) :
  (compose (compose rotate45 scale2) reflectYEqX).apply Q = { x := 14, y := 2 } →
  Q.y - Q.x = 6.66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_theorem_l23_2333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_lines_l23_2382

/-- Curve defined by parametric equations -/
noncomputable def x (t : ℝ) : ℝ := (1 + t^3) / (t^2 - 1)
noncomputable def y (t : ℝ) : ℝ := t / (t^2 - 1)

/-- Parameter value -/
def t₀ : ℝ := 2

/-- Theorem stating the equations of tangent and normal lines -/
theorem tangent_and_normal_lines :
  (∀ ε > 0, ∃ δ > 0, ∀ t, |t - t₀| < δ → |x t - x t₀| < ε * |y t - y t₀|) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ t, |t - t₀| < δ → |y t - y t₀| < ε * |x t - 3|) := by
  sorry

#check tangent_and_normal_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_lines_l23_2382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l23_2379

/-- A perfect point is a point where the abscissa and ordinate are equal -/
def isPerfectPoint (x y : ℝ) : Prop := x = y

/-- The quadratic function -/
noncomputable def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x + c

/-- The adjusted quadratic function -/
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := f a c x - 5/4

theorem quadratic_function_range (a c m : ℝ) : 
  a ≠ 0 → 
  (∃! x, isPerfectPoint x (f a c x)) → 
  isPerfectPoint 2 (f a c 2) → 
  (∀ x, 0 ≤ x ∧ x ≤ m → g a c x ≥ -21/4) → 
  (∀ x, 0 ≤ x ∧ x ≤ m → g a c x ≤ 1) → 
  (∃ x, 0 ≤ x ∧ x ≤ m ∧ g a c x = -21/4) → 
  (∃ x, 0 ≤ x ∧ x ≤ m ∧ g a c x = 1) → 
  5/2 ≤ m ∧ m ≤ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l23_2379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_40_l23_2369

/-- The speed of a train in km/hr, given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: The speed of a train is approximately 40 km/hr -/
theorem train_speed_approx_40 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |train_speed 100 9 - 40| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_40_l23_2369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_congruence_count_l23_2348

theorem four_digit_congruence_count : 
  let S : Finset ℕ := Finset.filter (fun x => 1000 ≤ x ∧ x ≤ 9999 ∧ (4582 * x + 902) % 17 = 2345 % 17) (Finset.range 10000)
  Finset.card S = 530 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_congruence_count_l23_2348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_where_g_equals_2_l23_2316

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -1 then 2*x + 3
  else if x ≤ 1 then -x
  else 2*x - 3

theorem sum_of_x_coordinates_where_g_equals_2 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ = 2 ∧ g x₂ = 2 ∧ x₁ + x₂ = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_where_g_equals_2_l23_2316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_l23_2340

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = 2 + m / (a^x - 1) -/
noncomputable def f (a m : ℝ) (x : ℝ) : ℝ := 2 + m / (a^x - 1)

/-- If f(x) = 2 + m / (a^x - 1) is an odd function, then m = 4 -/
theorem odd_function_m_value (a : ℝ) (h : a ≠ 1) :
  ∃ m : ℝ, IsOdd (f a m) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_l23_2340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l23_2317

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define points
def O : ℝ × ℝ := (0, 0)
def F (a b : ℝ) : ℝ × ℝ := sorry
def A (a b : ℝ) : ℝ × ℝ := sorry
def B (a b : ℝ) : ℝ × ℝ := sorry

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  ∃ (x y : ℝ), hyperbola a b x y ∧
  (F a b).1 > 0 ∧  -- F is the right focus
  (A a b).2 = b/a * (A a b).1 ∧  -- A is on one asymptote
  (B a b).2 = -b/a * (B a b).1 ∧  -- B is on the other asymptote
  (A a b).1 = (F a b).1 ∧  -- AF is perpendicular to x-axis
  (B a b).2 - (F a b).2 = b/a * ((B a b).1 - (F a b).1) ∧  -- BF is parallel to OA
  ((A a b).1 - (B a b).1) * (B a b).1 + 
  ((A a b).2 - (B a b).2) * (B a b).2 = 0  -- AB · OB = 0

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := 
  Real.sqrt (1 + b^2 / a^2)

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) :
  conditions a b → eccentricity a b = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l23_2317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_parallel_transitive_perpendicular_l23_2309

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Theorem 1
theorem perpendicular_parallel_implies_perpendicular
  (m n : Line) (α : Plane)
  (h1 : perpendicular_line_plane m α)
  (h2 : parallel_line_plane n α) :
  perpendicular_line_line m n :=
sorry

-- Theorem 2
theorem parallel_transitive_perpendicular
  (m : Line) (α β γ : Plane)
  (h1 : parallel_plane α β)
  (h2 : parallel_plane β γ)
  (h3 : perpendicular_line_plane m α) :
  perpendicular_line_plane m γ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_parallel_transitive_perpendicular_l23_2309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_equality_l23_2370

theorem cos_sin_sum_equality : 
  Real.cos (15 * π / 180) * Real.cos (45 * π / 180) + 
  Real.sin (15 * π / 180) * Real.sin (45 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_equality_l23_2370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l23_2327

theorem tan_theta_plus_pi_fourth (θ : Real) :
  Real.tan (θ + Real.pi/4) = -3 → 2 * (Real.sin θ)^2 - (Real.cos θ)^2 = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l23_2327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_cost_when_combined_l23_2328

/-- Represents the cost calculation for photocopies with discounts -/
def photocopy_cost (bw_copies : ℚ) (color_copies : ℚ) : ℚ :=
  let total_copies := bw_copies + color_copies
  let bw_cost := 0.02 * bw_copies
  let color_cost := 0.05 * color_copies
  let total_cost := bw_cost + color_cost
  let discount_rate := if total_copies > 200 then 0.35
                       else if total_copies > 100 then 0.25
                       else 0
  total_cost * (1 - discount_rate)

/-- Theorem stating the additional cost when submitting orders together -/
theorem additional_cost_when_combined :
  photocopy_cost 80 80 - (photocopy_cost 80 0 + photocopy_cost 0 80) = 2.8 := by
  sorry

#eval photocopy_cost 80 80 - (photocopy_cost 80 0 + photocopy_cost 0 80)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_cost_when_combined_l23_2328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_satisfies_conditions_l23_2320

-- Define the arithmetic progression
def arithmetic_progression : List ℤ := [2, 1, 0, -1]

-- Theorem statement
theorem arithmetic_progression_satisfies_conditions :
  -- Condition 1: The list has 4 elements
  arithmetic_progression.length = 4 ∧
  -- Condition 2: All elements are integers (implicit in the type ℤ)
  -- Condition 3: The largest number is equal to the sum of squares of the other three
  List.maximum arithmetic_progression = some (List.sum (List.map (λ x => x^2) (List.tail arithmetic_progression))) := by
  sorry

#eval arithmetic_progression
#eval List.maximum arithmetic_progression
#eval List.sum (List.map (λ x => x^2) (List.tail arithmetic_progression))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_satisfies_conditions_l23_2320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_quadrants_range_l23_2387

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x + 2*a) * Real.log (x + 1)

-- Define the property of passing through all four quadrants
def passes_through_all_quadrants (a : ℝ) : Prop :=
  (∃ x₁ > 0, f a x₁ > 0) ∧
  (∃ x₂ > 0, f a x₂ < 0) ∧
  (∃ x₃ < 0, f a x₃ > 0) ∧
  (∃ x₄ < 0, f a x₄ < 0)

-- Theorem statement
theorem f_quadrants_range (a : ℝ) :
  passes_through_all_quadrants a ↔ -1/3 < a ∧ a < 0 := by
  sorry

#check f_quadrants_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_quadrants_range_l23_2387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_expressions_equal_24_l23_2319

def numbers : List ℚ := [3, 4, -6, 10]

inductive Expression : Type
| Num : ℚ → Expression
| Add : Expression → Expression → Expression
| Sub : Expression → Expression → Expression
| Mul : Expression → Expression → Expression
| Div : Expression → Expression → Expression

def eval : Expression → ℚ
| Expression.Num n => n
| Expression.Add e1 e2 => eval e1 + eval e2
| Expression.Sub e1 e2 => eval e1 - eval e2
| Expression.Mul e1 e2 => eval e1 * eval e2
| Expression.Div e1 e2 => eval e1 / eval e2

def containsNumber : Expression → ℚ → Prop
| Expression.Num n, m => n = m
| Expression.Add e1 e2, n => containsNumber e1 n ∨ containsNumber e2 n
| Expression.Sub e1 e2, n => containsNumber e1 n ∨ containsNumber e2 n
| Expression.Mul e1 e2, n => containsNumber e1 n ∨ containsNumber e2 n
| Expression.Div e1 e2, n => containsNumber e1 n ∨ containsNumber e2 n

def validExpression (e : Expression) : Prop :=
  ∀ n : ℚ, containsNumber e n → n ∈ numbers

theorem two_valid_expressions_equal_24 :
  ∃ e1 e2 : Expression, e1 ≠ e2 ∧ validExpression e1 ∧ validExpression e2 ∧ eval e1 = 24 ∧ eval e2 = 24 :=
sorry

#check two_valid_expressions_equal_24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_expressions_equal_24_l23_2319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_pi_over_four_l23_2346

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle (x y : V) : ℝ := Real.arccos ((inner x y) / (norm x * norm y))

theorem vector_angle_pi_over_four (a b : V) 
  (ha : ‖a‖ = Real.sqrt 2)
  (hb : ‖b‖ = 1)
  (hab : ‖a - 2 • b‖ = Real.sqrt 2) : 
  angle a b = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_pi_over_four_l23_2346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_shift_l23_2332

theorem sin_cos_shift (x : ℝ) : 
  (Real.sin (2 * x) + Real.cos (2 * x) = Real.sqrt 2 * Real.sin (2 * x + π / 4)) ∧
  (Real.sin (2 * x) - Real.cos (2 * x) = Real.sqrt 2 * Real.sin (2 * x - π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_shift_l23_2332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l23_2323

theorem equation_solutions : 
  {(a, b) : ℤ × ℤ | a^2 + a*b - b = 2018} = 
  {(2, 2014), (0, -2018), (2018, -2018), (-2016, 2014)} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l23_2323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l23_2367

noncomputable def f (x : ℝ) : ℝ := 4^x - 6 * 2^x + 8

theorem f_minimum :
  ∃ (x_min : ℝ), 
    (∀ x, f x ≥ f x_min) ∧ 
    f x_min = -1 ∧ 
    x_min = Real.log 3 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l23_2367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l23_2344

theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) (h1 : p > 0) : 
  let C := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let F := (p/2, 0)
  let directrix := {(x, y) : ℝ × ℝ | x = -p/2}
  A ∈ C ∧ B ∈ C ∧
  (A.1 - F.1, A.2 - F.2) = 3 • (B.1 - F.1, B.2 - F.2) ∧
  |((A.1 + B.1)/2 + p/2)| = 16/3 →
  p = 4 :=
by sorry

#check parabola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l23_2344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_division_theorem_l23_2392

/-- The number of fish each boy initially takes before the fraction -/
def r : ℕ := sorry

/-- The total number of boys -/
def x : ℕ := sorry

/-- The number of fish each boy receives after the division -/
def y : ℕ := sorry

/-- The total number of fish -/
def T : ℕ := sorry

/-- The proposition that the division method results in equal shares for all boys -/
def equal_division : Prop :=
  (x > 0) ∧ 
  (r > 0) ∧
  (y = x * r) ∧
  (∀ i : ℕ, i > 0 → i ≤ x → 
    (i * r + (T - (i * r)) / 7 = y))

theorem fish_division_theorem :
  equal_division → x = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_division_theorem_l23_2392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l23_2394

theorem trig_problem :
  ∀ α : Real,
  Real.tan α = -4/3 →
  α ∈ Set.Icc (3*Real.pi/2) (2*Real.pi) →
  (Real.sin α = -4/5 ∧ Real.cos α = 3/5) ∧
  Real.sin (25*Real.pi/6) + Real.cos (26*Real.pi/3) + Real.tan (-25*Real.pi/4) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l23_2394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_on_vertex_l23_2395

/-- Given a triangle ABC and positive real numbers a, b, c, where a ≥ b + c,
    the point A minimizes the expression a · MA + b · MB + c · MC
    for any point M in the plane of triangle ABC. -/
theorem min_point_on_vertex (A B C M : EuclideanSpace ℝ (Fin 2))
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hge : a ≥ b + c) :
  a * ‖M - A‖ + b * ‖M - B‖ + c * ‖M - C‖ ≥ b * ‖B - A‖ + c * ‖C - A‖ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_on_vertex_l23_2395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l23_2303

open Real

theorem integral_proof (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 4) (h₃ : x ≠ -5) :
  deriv (λ y => y^2 - log (abs y) + (1/9) * log (abs (y - 4)) - (1/9) * log (abs (y + 5))) x =
  (2*x^4 + 2*x^3 - 41*x^2 + 20) / (x * (x - 4) * (x + 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l23_2303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_third_mile_time_speed_decreases_l23_2341

/-- Represents the speed of a particle at a given mile -/
noncomputable def speed (n : ℕ) : ℝ :=
  1 / ((n - 2 : ℝ) ^ 2)

/-- Represents the time taken to traverse a given mile -/
noncomputable def time (n : ℕ) : ℝ :=
  1 / speed n

theorem particle_movement (n : ℕ) (h : n ≥ 3) :
  time n = (n - 2 : ℝ) ^ 2 := by
  sorry

theorem third_mile_time :
  time 3 = 4 := by
  sorry

theorem speed_decreases (n m : ℕ) (hn : n > 3) (hm : m > n) :
  speed m < speed n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_third_mile_time_speed_decreases_l23_2341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_nonnegativity_l23_2338

open Real Set

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - a * x

theorem f_monotonicity_and_nonnegativity (a : ℝ) :
  (∀ x ∈ Ioo 0 (exp (a - 1)), StrictMonoOn (fun x => -(f a x)) (Ioo 0 (exp (a - 1)))) ∧
  (∀ x ∈ Ioi (exp (a - 1)), StrictMonoOn (f a) (Ioi (exp (a - 1)))) ∧
  (∀ x > 0, f a x + a ≥ 0 ↔ a = 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_nonnegativity_l23_2338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_approx_74_l23_2391

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of sides and median
def side_AB : ℝ := 9
def side_AC : ℝ := 17
def median_AM : ℝ := 12

-- Define the area of the triangle
noncomputable def triangle_area : ℝ :=
  let s := (side_AB + side_AC + Real.sqrt 452) / 2
  Real.sqrt (s * (s - side_AB) * (s - side_AC) * (s - Real.sqrt 452))

-- Theorem statement
theorem triangle_ABC_area_approx_74 :
  abs (triangle_area - 74) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_approx_74_l23_2391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l23_2314

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((a^2 - 1) * x^2 + (a + 1) * x + 1)

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (1 ≤ a ∧ a ≤ 5/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l23_2314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_maintenance_l23_2305

/-- Represents the demand function for a product -/
noncomputable def demand_function (initial_price initial_demand price_increase : ℝ) (x : ℝ) : ℝ :=
  initial_demand - (x / price_increase) * (initial_demand - (initial_demand * initial_price) / (initial_price + price_increase))

/-- Theorem stating the necessary decrease in demand to maintain revenue -/
theorem revenue_maintenance
  (initial_price initial_demand price_increase : ℝ)
  (h1 : initial_price = 20)
  (h2 : initial_demand = 500)
  (h3 : price_increase = 5)
  (h4 : initial_price > 0)
  (h5 : initial_demand > 0)
  (h6 : price_increase > 0) :
  ∃ (decrease : ℝ),
    decrease = 100 ∧
    (initial_price + price_increase) * (initial_demand - decrease) ≥ initial_price * initial_demand ∧
    ∀ (d : ℝ), d < decrease →
      (initial_price + price_increase) * (initial_demand - d) < initial_price * initial_demand :=
by
  sorry

#check revenue_maintenance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_maintenance_l23_2305
