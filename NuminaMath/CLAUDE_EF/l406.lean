import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l406_40675

/-- Square ABCD with side length s -/
structure Square (s : ℝ) where
  side : s > 0

/-- Rectangle with width w and height h -/
structure Rectangle (w h : ℝ) where
  width : ℝ
  height : ℝ
  width_pos : w > 0
  height_pos : h > 0

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle w h) : ℝ := r.width * r.height

/-- Square ABCD divided into three rectangles of equal area -/
def divided_square (s : ℝ) (r1 r2 r3 : Rectangle w h) : Prop :=
  ∃ (sq : Square s),
  r1.area = r2.area ∧ r2.area = r3.area ∧
  r1.width + r2.width + r3.width = s ∧
  r1.height = s ∧ r2.height = s ∧ r3.height = s

theorem square_area (s : ℝ) (r1 r2 r3 : Rectangle w h) 
  (h1 : divided_square s r1 r2 r3) (h2 : r1.width = 4) : s^2 = 144 := by
  sorry

#check square_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l406_40675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_is_four_l406_40671

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Represents a curve in polar form -/
structure PolarCurve where
  ρ : ℝ → ℝ

/-- The curve C with polar equation ρ = 4sin θ -/
noncomputable def curveC : PolarCurve :=
  { ρ := fun θ => 4 * Real.sin θ }

/-- The line l with parametric equations x = 2t, y = √3 * t + 2 -/
noncomputable def lineL : ParametricLine :=
  { x := fun t => 2 * t
    y := fun t => Real.sqrt 3 * t + 2 }

/-- The length of the segment cut off by curve C on line l -/
def segmentLength : ℝ := 4

/-- Theorem stating that the length of the segment cut off by curve C on line l is 4 -/
theorem segment_length_is_four :
  segmentLength = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_is_four_l406_40671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_race_theorem_l406_40670

/-- The distance from Chang'an to Qi in miles -/
def distance_to_qi : ℚ := 1125

/-- The distance traveled by the good horse on day n -/
def good_horse_distance (n : ℕ) : ℚ := 103 + 13 * (n - 1)

/-- The distance traveled by the poor horse on day n -/
def poor_horse_distance (n : ℕ) : ℚ := 97 - (1/2) * (n - 1)

/-- The total distance traveled by the good horse after n days -/
def good_horse_total_distance (n : ℕ) : ℚ := n * (103 + good_horse_distance n) / 2

/-- The total distance traveled by the poor horse after n days -/
def poor_horse_total_distance (n : ℕ) : ℚ := n * (97 + poor_horse_distance n) / 2

theorem horse_race_theorem :
  (poor_horse_distance 7 = 94) ∧
  (∃ n : ℕ, good_horse_total_distance n + poor_horse_total_distance n = 2 * distance_to_qi ∧
            good_horse_total_distance n = 1395) := by
  sorry

#eval poor_horse_distance 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_race_theorem_l406_40670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l406_40625

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry (x : ℝ) :
  (∀ y, f (x + y) = f (x - y)) ↔ x = Real.pi / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l406_40625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_correctness_l406_40692

-- Proposition 1
noncomputable def f1 (a : ℝ) (x : ℝ) : ℝ := 2 * a^(2*x - 1) - 1

-- Proposition 2
noncomputable def f2 (x : ℝ) : ℝ := if x ≥ 0 then x * (x + 1) else -x * (x + 1)

-- Proposition 3
noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Proposition 4
noncomputable def f4 (x : ℝ) : ℝ := sorry

-- Proposition 5
noncomputable def f5 (x : ℝ) : ℝ := Real.log x

theorem propositions_correctness :
  (∃ a : ℝ, f1 a (1/2) ≠ -1) ∧
  (∃ a : ℝ, f2 a = -2 ∧ a ≠ -1 ∧ a ≠ 2) ∧
  (∀ a : ℝ, log_a (1/2) > 1 → 1/2 < a ∧ a < 1) ∧
  (∀ x : ℝ, f4 x = f4 (4 - x) → ∀ y : ℝ, f4 (2 + y) = f4 (2 - y)) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → f5 ((x₁ + x₂) / 2) ≥ (f5 x₁ + f5 x₂) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_correctness_l406_40692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_above_median_l406_40632

/-- Represents the frequency distribution of scores -/
structure FrequencyDistribution :=
  (group1 : ℕ)
  (group2 : ℕ)
  (group3 : ℕ)
  (group4 : ℕ)
  (group5 : ℕ)

/-- Represents the scores in the 15.6-18.6 group -/
def group3Scores : List ℝ := [15.7, 16.0, 16.0, 16.2, 16.6, 16.8, 17.2, 17.5, 17.8, 18.0, 18.2, 18.4]

/-- Calculates the median of a sorted list -/
noncomputable def median (l : List ℝ) : ℝ := sorry

/-- Theorem stating that Xiao Ming's score is above the median -/
theorem xiao_ming_above_median (fd : FrequencyDistribution) 
  (h1 : fd.group1 + fd.group2 + fd.group3 + fd.group4 + fd.group5 = 60)
  (h2 : fd.group1 = 8)
  (h3 : fd.group2 = 17)
  (h4 : fd.group3 = 12)
  (h5 : fd.group5 = 3) :
  let medianScore := median group3Scores
  17.2 > medianScore := by sorry

#check xiao_ming_above_median

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_above_median_l406_40632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_weight_difference_l406_40609

theorem box_weight_difference (a b c d : ℕ) : 
  a < b → b < c → c < d →
  {a + b, a + c, a + d, b + c, b + d, c + d} ⊆ ({22, 23, 27, 29, 30, 25} : Set ℕ) →
  d - a = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_weight_difference_l406_40609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l406_40667

/-- Parabola structure with focus and directrix -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  directrix : ℝ → ℝ × ℝ
  eq : (x y : ℝ) → Prop

/-- Point on a parabola -/
def PointOnParabola (para : Parabola) (x y : ℝ) : Prop :=
  para.eq x y

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: For a parabola y^2 = 2px (p > 0), if a point M(3, y) on the parabola
    satisfies |EF| = |MF|, where E is the foot of the perpendicular from M to the directrix
    and F is the focus, then p = 2 -/
theorem parabola_property (para : Parabola) (h1 : para.p > 0) :
  let M : ℝ × ℝ := (3, Real.sqrt (6 * para.p))
  let E : ℝ × ℝ := (-para.p/2, M.2)
  let F : ℝ × ℝ := para.focus
  PointOnParabola para M.1 M.2 →
  distance E F = distance M F →
  para.p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l406_40667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l406_40643

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | 1 => 2
  | n + 2 => sequence_a (n + 1) + Real.log (1 + 1 / (n + 1))

theorem sequence_a_formula (n : ℕ) (h : n ≥ 1) :
  sequence_a n = 2 + Real.log n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l406_40643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l406_40603

/-- The radius of the circumcircle of a triangle with two sides of length a and one side of length b -/
noncomputable def circumradius (a b : ℝ) : ℝ :=
  a^2 / Real.sqrt (4 * a^2 - b^2)

/-- Theorem: The circumradius of a triangle with two sides of length a and one side of length b
    is equal to a^2 / sqrt(4a^2 - b^2) -/
theorem triangle_circumradius (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : b < 2*a) :
  circumradius a b = a^2 / Real.sqrt (4 * a^2 - b^2) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l406_40603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l406_40612

theorem least_number_with_remainder (n : ℕ) : n = 1447 ↔
  (∀ d ∈ ({8, 12, 16, 24, 30, 36} : Set ℕ), n % d = 7) ∧
  (∀ m < n, ∃ d ∈ ({8, 12, 16, 24, 30, 36} : Set ℕ), m % d ≠ 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l406_40612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_r_l406_40660

theorem solve_for_r (k : ℝ) (r : ℝ) 
  (h1 : 5 = k * 2^r) 
  (h2 : 45 = k * 8^r) : 
  r = 1/2 * Real.log 9 / Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_r_l406_40660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_l406_40620

-- Define the equation
def equation (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + 3*m = 0

-- Define the triangle
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2
  is_valid : side1 + side2 > base ∧ side1 + base > side2 ∧ side2 + base > side1

-- Theorem statement
theorem perimeter_of_triangle (m : ℝ) (t : IsoscelesTriangle) : 
  (equation m 2) ∧ 
  (∃ x y : ℝ, equation m x ∧ equation m y ∧ x ≠ y ∧ ({x, y} : Set ℝ) = {t.side1, t.base}) →
  t.side1 + t.side2 + t.base = 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_l406_40620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_proportion_is_one_thirteenth_l406_40662

/-- Represents the proportion of time taken to travel the first quarter of a distance
    given that the speed for the first quarter is 4 times the speed for the remaining distance -/
noncomputable def travelTimeProportion (D : ℝ) (V : ℝ) : ℝ :=
  let time1 := D / (16 * V)
  let time2 := 3 * D / (4 * V)
  time1 / (time1 + time2)

/-- Theorem stating that the proportion of time taken to travel the first quarter
    of the distance is 1/13 of the total time -/
theorem travel_time_proportion_is_one_thirteenth (D : ℝ) (V : ℝ) 
  (h1 : D > 0) (h2 : V > 0) : 
  travelTimeProportion D V = 1 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_proportion_is_one_thirteenth_l406_40662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l406_40624

-- Define a circle with center O and radius r
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in the plane
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define what it means for a point to be inside a circle
def is_inside (p : Point) (c : Circle) : Prop :=
  distance p c.center < c.radius

-- Theorem statement
theorem point_inside_circle (O : Point) (P : Point) :
  let c : Circle := { center := O, radius := 3 }
  distance O P = 2 → is_inside P c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l406_40624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_and_sufficient_condition_l406_40640

theorem quadratic_roots_and_sufficient_condition 
  (x m a : ℝ) : 
  ((∃ x, x^2 - m*x + m^2 - 2*m + 1 = 0) ↔ (2/3 ≤ m ∧ m ≤ 2)) ∧
  (((1 - 2*a < m ∧ m < a + 1) → (∃ x, x^2 - m*x + m^2 - 2*m + 1 = 0)) ↔ (a ≤ 1/6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_and_sufficient_condition_l406_40640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l406_40638

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def line_eq (x y k m : ℝ) : Prop := y = k * x - m

-- Define point A
def point_A : ℝ × ℝ := (0, 1)

-- Define the intersection of the line and circle
def intersects (k m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq x y k m

-- Define the angle BAC
def angle_BAC (k m : ℝ) : ℝ := 60

-- Define the areas of triangles
noncomputable def area_ABC (k m : ℝ) : ℝ := sorry
noncomputable def area_OBC (k m : ℝ) : ℝ := sorry

-- Theorem statement
theorem line_circle_intersection
  (k m : ℝ)
  (h1 : intersects k m)
  (h2 : angle_BAC k m = 60)
  (h3 : area_ABC k m = 2 * area_OBC k m) :
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l406_40638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l406_40629

def digits : List Nat := [2, 0, 1, 3, 0, 5, 1, 8]

def is_valid_permutation (perm : List Nat) : Bool :=
  perm.length = 8 &&
  perm.head? ≠ some 0 &&
  (perm.getLast? = some 1 || perm.getLast? = some 3 || perm.getLast? = some 5) &&
  perm.toFinset = digits.toFinset

def count_valid_permutations : Nat :=
  (List.permutations digits).filter is_valid_permutation |>.length

theorem valid_permutations_count : count_valid_permutations = 3600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l406_40629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_40_degrees_l406_40680

theorem complex_power_40_degrees :
  let z : ℂ := 3 * (Complex.cos (40 * Real.pi / 180) + Complex.I * Complex.sin (40 * Real.pi / 180))
  z^4 = Complex.mk (-81 * (Real.sqrt 3 + 1) / (2 * Real.sqrt 2)) (81 * (Real.sqrt 3 - 1) / (2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_40_degrees_l406_40680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_proof_l406_40658

noncomputable def start_point : ℝ × ℝ × ℝ := (1, 2, 3)
noncomputable def end_point : ℝ × ℝ × ℝ := (0, -2, -2)
noncomputable def sphere_radius : ℝ := 2
noncomputable def sphere_center : ℝ × ℝ × ℝ := (0, 0, 0)

noncomputable def line_direction : ℝ × ℝ × ℝ :=
  (end_point.1 - start_point.1, end_point.2.1 - start_point.2.1, end_point.2.2 - start_point.2.2)

noncomputable def intersection_distance : ℝ := 12 * Real.sqrt 66 / 7

theorem intersection_distance_proof :
  let line := λ t : ℝ => (start_point.1 + t * line_direction.1,
                          start_point.2.1 + t * line_direction.2.1,
                          start_point.2.2 + t * line_direction.2.2)
  let sphere := λ (x y z : ℝ) => (x - sphere_center.1)^2 + (y - sphere_center.2.1)^2 + (z - sphere_center.2.2)^2 = sphere_radius^2
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
    sphere (line t₁).1 (line t₁).2.1 (line t₁).2.2 ∧
    sphere (line t₂).1 (line t₂).2.1 (line t₂).2.2 ∧
    Real.sqrt ((line t₁).1 - (line t₂).1)^2 + ((line t₁).2.1 - (line t₂).2.1)^2 + ((line t₁).2.2 - (line t₂).2.2)^2 = intersection_distance :=
by sorry

#check intersection_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_proof_l406_40658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l406_40689

open Real Set

-- Define the functional equation
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Ioo (-1) 1 → y ∈ Ioo (-1) 1 → x + y ∈ Ioo (-1) 1 →
    f (x + y) = (f x + f y) / (1 - f x * f y)

-- Main theorem
theorem functional_equation_solution 
    {f : ℝ → ℝ} (hf : Continuous f) 
    (heq : SatisfiesFunctionalEquation f) :
    ∃ a : ℝ, abs a ≤ π/2 ∧ ∀ x ∈ Ioo (-1) 1, f x = tan (a * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l406_40689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l406_40694

-- Define the sign function
noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x < 0 then -1
  else 0

-- Define the equation
def equation (x : ℝ) : Prop :=
  x^2 - x * sgn x - 6 = 0

-- Theorem statement
theorem equation_roots : 
  ∀ x : ℝ, equation x ↔ (x = -3 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l406_40694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convexity_condition_l406_40650

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/12) * x^4 - (m/6) * x^3 - (3/2) * x^2

-- Define the second derivative of f(x)
noncomputable def f_second_derivative (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x - 3

-- Define the property of being convex on an interval
def is_convex_on (m : ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → (f_second_derivative m x) < 0

-- State the theorem
theorem convexity_condition (m : ℝ) :
  (is_convex_on m 1 3) → m ≥ 2 := by
  sorry

#check convexity_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convexity_condition_l406_40650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_minus_6y_values_l406_40664

open Real

theorem cos_x_minus_6y_values (x y : ℝ) 
  (h1 : Real.sin (3 * x) / ((2 * Real.cos (2 * x) + 1) * Real.sin (2 * y)) = 1/5 + (Real.cos (x - 2 * y))^2)
  (h2 : Real.cos (3 * x) / ((1 - 2 * Real.cos (2 * x)) * Real.cos (2 * y)) = 4/5 + (Real.sin (x - 2 * y))^2) :
  Real.cos (x - 6 * y) = -3/5 ∨ Real.cos (x - 6 * y) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_minus_6y_values_l406_40664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l406_40678

/-- An isosceles triangle DEF with altitude DG -/
structure IsoscelesTriangle where
  /-- The length of the equal sides DE and DF -/
  side : ℝ
  /-- The ratio of EG to GF -/
  ratio : ℝ
  /-- The side DE equals the given side length -/
  h_de_eq : side = 5
  /-- The side DF equals the given side length -/
  h_df_eq : side = 5
  /-- EG is ratio times GF -/
  h_eg_gf : ratio = 2

/-- The length of the base EF in the isosceles triangle -/
noncomputable def baseLength (t : IsoscelesTriangle) : ℝ := 5 * Real.sqrt 3

/-- Theorem stating that the base length of the isosceles triangle is 5√3 -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle) : 
  baseLength t = 5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l406_40678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_l406_40645

/-- The equation from the original problem -/
noncomputable def f (a : ℝ) : ℝ := (10 * Real.sqrt ((2*a)^2 + 1) - 3*a^2 - 2) / (Real.sqrt (1 + 3*a^2) + 4)

/-- Theorem stating that -4 is the smallest real number satisfying the equation -/
theorem smallest_solution :
  ∃ (a : ℝ), f a = 3 ∧ ∀ (b : ℝ), f b = 3 → a ≤ b ∧ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_l406_40645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_height_solution_l406_40685

/-- The height function for the projectile -/
def height_function (t : ℝ) : ℝ := 60 + 8*t - 5*t^2

/-- The theorem stating the existence of a positive solution -/
theorem projectile_height_solution :
  ∃ t : ℝ, t > 0 ∧ height_function t = 45 ∧ abs (t - 2.708) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_height_solution_l406_40685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_eq_one_solution_l406_40602

theorem cos_minus_sin_eq_one_solution (n : ℕ+) :
  ∀ x : ℝ, (Real.cos (n * x) - Real.sin (n * x) = 1) ↔ 
  ∃ k : ℤ, x = (2 * k * Real.pi) / (n : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_eq_one_solution_l406_40602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l406_40681

theorem function_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let f := fun (x : ℝ) => (1/2 : ℝ)^x
  f ((a + b) / 2) ≤ f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) ≤ f (2 * a * b / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l406_40681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l406_40623

theorem binomial_expansion_constant_term (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 9 k) * a^k = 84 ∧ 9 - 3*k = 0) → a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l406_40623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l406_40686

open Real

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (log x) / x

-- State the theorem
theorem monotonic_decreasing_interval_f :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, y > x → f y < f x) ↔ x > Real.exp 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l406_40686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_domain_l406_40679

open Set
open Function

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem f_negative_domain
  (hf : DifferentiableOn ℝ f domain_f)
  (h_domain : ∀ x ∈ domain_f, f x ≠ 0 → x * (deriv f x) > f x)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x < 0} = Ioo 0 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_domain_l406_40679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_wages_calculation_l406_40619

noncomputable def monthly_budget : ℝ := 4500

noncomputable def food_expense : ℝ := monthly_budget / 3
noncomputable def supplies_expense : ℝ := monthly_budget / 4
noncomputable def rent_expense : ℝ := 800
noncomputable def utilities_expense : ℝ := 300
noncomputable def tax_expense : ℝ := monthly_budget * 0.1

noncomputable def total_expenses : ℝ := food_expense + supplies_expense + rent_expense + utilities_expense + tax_expense

noncomputable def employee_wages : ℝ := monthly_budget - total_expenses

theorem employee_wages_calculation : employee_wages = 325 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_wages_calculation_l406_40619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_progression_sine_ratio_bounds_l406_40693

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem triangle_geometric_progression_sine_ratio_bounds 
  (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * c = b^2 →
  (golden_ratio - 1) < (Real.sin B) / (Real.sin A) ∧ 
  (Real.sin B) / (Real.sin A) < golden_ratio :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_progression_sine_ratio_bounds_l406_40693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_arrangement_l406_40691

def number_of_arrangements (n m : ℕ) : ℕ := 
  -- number of ways to arrange n indistinguishable chairs and m indistinguishable stools
  -- such that no two stools are adjacent
  sorry

theorem committee_arrangement (n m : ℕ) (hn : n = 7) (hm : m = 3) : 
  number_of_arrangements n m = Nat.choose (n + 1) m := by
  sorry

#check committee_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_arrangement_l406_40691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_seniors_playing_instrument_count_l406_40610

def non_seniors_playing_instrument (total_students : ℕ) 
  (senior_play_percent : ℚ) (non_senior_not_play_percent : ℚ) 
  (total_not_play_percent : ℚ) : ℕ :=
  600

axiom total_students_count : (non_seniors_playing_instrument 800 (60 / 100) (25 / 100) (55 / 100)) = 600

theorem non_seniors_playing_instrument_count :
  non_seniors_playing_instrument 800 (60 / 100) (25 / 100) (55 / 100) = 600 := by
  exact total_students_count

#eval non_seniors_playing_instrument 800 (60 / 100) (25 / 100) (55 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_seniors_playing_instrument_count_l406_40610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l406_40622

def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h : isOdd (λ x => f x + x^2)) (h1 : f 1 = 1) : 
  f (-1) = -3 := by
  have h2 : (λ x => f x + x^2) (-1) = -((λ x => f x + x^2) 1) := h 1
  simp at h2
  rw [h1] at h2
  linarith

#check odd_function_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l406_40622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_range_l406_40639

/-- The curve equation y = 1 + √(1 - x²) -/
noncomputable def curve (x : ℝ) : ℝ := 1 + Real.sqrt (1 - x^2)

/-- The line equation y = k(x - 3) + 3 -/
def line (k : ℝ) (x : ℝ) : ℝ := k * (x - 3) + 3

/-- Two points are distinct if they are not equal -/
def distinct_points (p q : ℝ × ℝ) : Prop := p ≠ q

/-- The curve and line have two distinct intersection points -/
def has_two_distinct_intersections (k : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ curve x₁ = line k x₁ ∧ curve x₂ = line k x₂

/-- The main theorem -/
theorem intersection_slope_range :
  ∀ k : ℝ, has_two_distinct_intersections k ↔ (3 - Real.sqrt 3) / 4 < k ∧ k ≤ 1 / 2 := by
  sorry

#check intersection_slope_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_range_l406_40639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greater_nordland_population_change_l406_40635

/-- Represents the population change in Greater Nordland during a leap year -/
def populationChange (hoursPerBirth deathsPerDay daysInYear : ℕ) : ℕ :=
  let birthsPerDay := 24 / hoursPerBirth
  let netChangePerDay := birthsPerDay - deathsPerDay
  netChangePerDay * daysInYear

/-- Rounds a number to the nearest hundred -/
def roundToNearestHundred (n : ℕ) : ℕ :=
  ((n + 50) / 100) * 100

/-- Theorem stating the population change in Greater Nordland during a leap year -/
theorem greater_nordland_population_change :
  roundToNearestHundred (populationChange 6 2 366) = 700 := by
  sorry

#eval roundToNearestHundred (populationChange 6 2 366)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greater_nordland_population_change_l406_40635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l406_40627

def U : Set ℕ := {1, 3, 5, 7, 9}

def A (a : ℝ) : Set ℝ := {1, |a - 5|, 9}

def complement_A : Set ℕ := {5, 7}

theorem find_a : ∃ a : ℝ, (A a ∩ {1, 3, 9} = {1, 3, 9}) ∧ (a = 2 ∨ a = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l406_40627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_constant_for_gcd_property_l406_40601

theorem existence_of_constant_for_gcd_property :
  ∃ (c : ℝ), c > 0 ∧
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 →
  (∀ (i j : ℕ), i ≤ n → j ≤ n → Nat.gcd (a + i) (b + j) > 1) →
  min (a : ℝ) b > c^n * n^(n / 2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_constant_for_gcd_property_l406_40601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_pi_l406_40617

open Real

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := cos x * sin x
noncomputable def f₂ (x : ℝ) : ℝ := cos x + sin x
noncomputable def f₃ (x : ℝ) : ℝ := sin x / cos x
noncomputable def f₄ (x : ℝ) : ℝ := 2 * sin x ^ 2

-- Define the period of a function
def isPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Theorem statement
theorem smallest_period_pi :
  (isPeriodic f₁ π) ∧
  (isPeriodic f₃ π) ∧
  (isPeriodic f₄ π) ∧
  ¬(isPeriodic f₂ π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_pi_l406_40617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_impossibility_l406_40616

def repeated_operation : ℕ → (ℕ → ℕ → ℕ → ℕ) → ℕ → ℕ
  | 0, _, x => x
  | n + 1, f, x => repeated_operation n f (f x x x)

theorem board_game_impossibility (n : ℕ) (hn : n = 1989) :
  ¬ ∃ (steps : ℕ), ∀ (operation : ℕ → ℕ → ℕ → ℕ),
  (∀ a b, operation a b (a + b) = a - b) →
  (repeated_operation steps operation (List.range n).sum) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_impossibility_l406_40616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_sum_at_two_and_neg_two_l406_40697

/-- A cubic polynomial with specific properties -/
def Q (k : ℝ) : ℝ → ℝ := sorry

/-- Q is a cubic polynomial -/
axiom Q_cubic (k : ℝ) : ∃ (a b c : ℝ), ∀ x, Q k x = a * x^3 + b * x^2 + c * x + 2 * k

/-- Q(0) = 2k -/
axiom Q_at_zero (k : ℝ) : Q k 0 = 2 * k

/-- Q(1) = 3k -/
axiom Q_at_one (k : ℝ) : Q k 1 = 3 * k

/-- Q(-1) = 4k -/
axiom Q_at_neg_one (k : ℝ) : Q k (-1) = 4 * k

/-- The main theorem: Q(2) + Q(-2) = 16k -/
theorem Q_sum_at_two_and_neg_two (k : ℝ) : Q k 2 + Q k (-2) = 16 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_sum_at_two_and_neg_two_l406_40697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_intersection_l406_40684

-- Define the ellipse and circle
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 12

-- Define the locus of point Q
noncomputable def locus_Q (x y : ℝ) : Prop := x^2/36 + y^2/48 = 1

-- Define the area of triangle OPQ
noncomputable def area_OPQ (x y : ℝ) : ℝ := (1/2) * x * y

-- Theorem statement
theorem ellipse_tangent_intersection :
  -- For any point P(x₀, y₀) on the ellipse
  ∀ x₀ y₀ : ℝ, ellipse x₀ y₀ →
  -- There exists a point Q(x₁, y₁)
  ∃ x₁ y₁ : ℝ,
    -- Such that Q is on the locus
    locus_Q x₁ y₁ ∧
    -- And when P is in the first quadrant
    (x₀ > 0 ∧ y₀ > 0 →
      -- The maximum area of triangle OPQ is √3/2
      area_OPQ x₀ y₀ ≤ Real.sqrt 3 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_intersection_l406_40684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_not_both_red_l406_40626

/-- The number of red balls -/
def num_red : ℕ := 3

/-- The number of white balls -/
def num_white : ℕ := 2

/-- The total number of balls -/
def total_balls : ℕ := num_red + num_white

/-- The number of balls drawn -/
def num_drawn : ℕ := 2

/-- The probability of drawing two balls that are not both red -/
theorem prob_not_both_red : 
  (7 : ℚ) / 10 = 
    1 - (Nat.choose num_red num_drawn : ℚ) / (Nat.choose total_balls num_drawn : ℚ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_not_both_red_l406_40626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circumcircle_l406_40628

-- Define the parabola and points
def Parabola (a : ℝ) := {p : ℝ × ℝ | p.2 = a * p.1^2}
def PointA : ℝ × ℝ := (1, 1)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector from one point to another
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- State the theorem
theorem parabola_and_circumcircle 
  (a : ℝ) 
  (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ 1 ∧ x₂ ≠ 1 ∧ x₁ ≠ x₂) 
  (h_orthogonal : dot_product (vector PointA (x₁, x₁^2)) (vector (x₁, x₁^2) (x₂, x₂^2)) = 0) :
  -- 1. The equation of the parabola is y = x²
  (a = 1) ∧
  -- 2. The minimum area of the circumcircle of triangle ABC is π
  (∃ (r : ℝ), r = 1 ∧ π * r^2 = π) ∧
  -- 3. The equation of the circumcircle in the minimum area case is x² + (y-1)² = 1
  (∃ (center : ℝ × ℝ), center = (0, 1) ∧ 
    ∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circumcircle_l406_40628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mamma_permutations_l406_40669

def word : String := "MAMMA"

def letter_count (s : String) : Nat := s.length

def m_count (s : String) : Nat := s.foldr (fun c acc => if c = 'M' then acc + 1 else acc) 0

def a_count (s : String) : Nat := s.foldr (fun c acc => if c = 'A' then acc + 1 else acc) 0

theorem mamma_permutations :
  Nat.factorial (letter_count word) / Nat.factorial (m_count word) / Nat.factorial (a_count word) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mamma_permutations_l406_40669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_with_tan_l406_40614

theorem trig_identity_with_tan (α : Real) 
  (h1 : Real.tan α = 4/3) 
  (h2 : 0 < α ∧ α < Real.pi/2) : 
  Real.sin (Real.pi + α) + Real.cos (Real.pi - α) = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_with_tan_l406_40614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l406_40648

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.sin x

-- State the theorem
theorem tangent_line_parallel (a : ℝ) :
  (∃ k : ℝ, k = 3 ∧ (deriv (f a)) 0 = k) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l406_40648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l406_40677

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  cosine_rule : c^2 = a^2 + b^2 - a*b

theorem triangle_properties (t : Triangle) : 
  t.C = π/3 ∧ 
  (∃ (x : ℝ), ∀ (y : ℝ), Real.cos t.A + Real.cos t.B ≤ x ∧ Real.cos t.A + Real.cos t.B = x → x = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l406_40677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unspecified_racer_seventh_l406_40668

def Race := Fin 15

structure RaceResult where
  alice : Race
  bob : Race
  charlie : Race
  dana : Race
  emily : Race
  charlie_emily : charlie.val = emily.val - 5
  dana_bob : dana.val = bob.val + 2
  alice_emily : alice.val = emily.val + 3
  bob_charlie : bob.val = charlie.val + 3
  dana_place : dana = ⟨10, by norm_num⟩

theorem unspecified_racer_seventh (result : RaceResult) : 
  result.alice ≠ ⟨7, by norm_num⟩ ∧ 
  result.bob ≠ ⟨7, by norm_num⟩ ∧ 
  result.charlie ≠ ⟨7, by norm_num⟩ ∧ 
  result.dana ≠ ⟨7, by norm_num⟩ ∧ 
  result.emily ≠ ⟨7, by norm_num⟩ := by
  sorry

#check unspecified_racer_seventh

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unspecified_racer_seventh_l406_40668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_3_mod_100_l406_40654

/-- The polynomial q(x) = x^2020 + x^2019 + x^2018 + ... + x + 1 -/
def q (x : ℝ) : ℝ := (Finset.range 2021).sum (fun i => x^i)

/-- The polynomial x^3 + x^2 + 2x + 1 -/
def divisor (x : ℝ) : ℝ := x^3 + x^2 + 2*x + 1

/-- s(x) is the polynomial remainder when q(x) is divided by x^3 + x^2 + 2x + 1 -/
noncomputable def s (x : ℝ) : ℝ := q x % divisor x

theorem remainder_of_s_3_mod_100 : |s 3| % 100 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_3_mod_100_l406_40654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_tangent_line_l406_40652

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (5/2) * x^2 - x + 185/6

noncomputable def f' (x : ℝ) : ℝ := x^2 - 5*x - 1

theorem min_value_on_tangent_line :
  ∀ m n : ℝ, m > 0 → n > 0 →
  (∃ (y : ℝ), y - f 5 = f' 5 * (m - 5) ∧ y = n) →
  1/m + 4/n ≥ 9/10 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧
  (∃ (y : ℝ), y - f 5 = f' 5 * (m₀ - 5) ∧ y = n₀) ∧
  1/m₀ + 4/n₀ = 9/10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_tangent_line_l406_40652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_points_count_l406_40630

def M : Finset ℕ := {1, 2, 3}
def N : Finset ℕ := {1, 5}

def cartesian_product (A B : Finset ℕ) : Finset (ℕ × ℕ) :=
  A.product B

theorem distinct_points_count : 
  ((cartesian_product M N) ∪ (cartesian_product N M)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_points_count_l406_40630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carmichael_function_bound_l406_40657

theorem carmichael_function_bound (m n : ℕ) (hm : m ≥ 2) 
  (h : ∀ a : ℕ, a > 0 → Nat.Coprime a n → n ∣ a^m - 1) : 
  n ≤ 4 * m * (2^m - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carmichael_function_bound_l406_40657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_80_l406_40651

/-- The speed of a car that takes 5 seconds longer to travel 1 km compared to traveling at 90 km/hr -/
noncomputable def car_speed : ℝ :=
  let time_90 := 1 / 90 -- Time in hours to travel 1 km at 90 km/hr
  let time_v := time_90 + 5 / 3600 -- Time in hours to travel 1 km at unknown speed v
  1 / time_v -- Speed = distance / time

theorem car_speed_is_80 : car_speed = 80 := by
  sorry

-- Remove #eval as it's not computable
-- #eval car_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_80_l406_40651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_best_fit_l406_40674

-- Define the types of functions we're considering
inductive FunctionType
  | Linear
  | Quadratic
  | Exponential
  | Logarithmic

-- Define a property for rapid initial growth
noncomputable def has_rapid_initial_growth (f : ℝ → ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo 0 ε, deriv f x > deriv f ε

-- Define a property for gradually slowing growth rate
noncomputable def has_slowing_growth_rate (f : ℝ → ℝ) : Prop :=
  ∃ a > 0, ∀ x y, a < x ∧ x < y → deriv f x > deriv f y

-- Main theorem
theorem logarithmic_best_fit (f : ℝ → ℝ) (t : FunctionType) :
  (has_rapid_initial_growth f ∧ has_slowing_growth_rate f) ↔ t = FunctionType.Logarithmic :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_best_fit_l406_40674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_propositions_true_l406_40682

-- Define a structure for quadrilaterals
structure Quadrilateral where
  -- You can add more specific properties here if needed
  mk :: -- Empty constructor for now

-- Define the properties of quadrilaterals
def has_one_pair_parallel_sides_and_congruent_angles (q : Quadrilateral) : Prop := sorry
def has_perpendicular_congruent_diagonals (q : Quadrilateral) : Prop := sorry
def is_midpoint_quadrilateral_of_rectangle (q : Quadrilateral) : Prop := sorry
def diagonals_bisect_angles (q : Quadrilateral) : Prop := sorry

-- Define the types of quadrilaterals
def is_parallelogram (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry

-- Define the propositions
def proposition1 : Prop := ∀ q : Quadrilateral, has_one_pair_parallel_sides_and_congruent_angles q → is_parallelogram q
def proposition2 : Prop := ∀ q : Quadrilateral, has_perpendicular_congruent_diagonals q → is_square q
def proposition3 : Prop := ∀ q : Quadrilateral, is_midpoint_quadrilateral_of_rectangle q → is_rhombus q
def proposition4 : Prop := ∀ q : Quadrilateral, is_rhombus q → diagonals_bisect_angles q

-- Theorem statement
theorem exactly_three_propositions_true : 
  (proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ proposition4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_propositions_true_l406_40682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bear_cycle_theorem_l406_40611

/-- A bear cycle on a square grid -/
def BearCycle (n : ℕ) := {cycle : List (ℕ × ℕ) // 
  cycle.length = n * n + 1 ∧ 
  cycle.head? = cycle.getLast? ∧
  ∀ i j, (i, j) ∈ cycle.tail ↔ 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n}

/-- The maximum length of remaining paths after removing a row or column -/
noncomputable def MaxRemainingPathLength (n : ℕ) (cycle : BearCycle n) (k : ℕ) : ℕ :=
  match k with
  | 0 => 0  -- row removal
  | _ => 0  -- column removal

/-- The theorem statement -/
theorem bear_cycle_theorem :
  ∀ (cycle : BearCycle 100),
    (∃ k, MaxRemainingPathLength 100 cycle k ≤ 5000) ∧
    (∀ m, m < 5000 → ∃ cycle', ∀ k, MaxRemainingPathLength 100 cycle' k > m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bear_cycle_theorem_l406_40611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_nine_l406_40631

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_two_even_two_odd (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (λ d => d % 2 = 0)).length = 2 ∧
  (digits.filter (λ d => d % 2 ≠ 0)).length = 2

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ n.digits 10

theorem smallest_four_digit_divisible_by_nine :
  ∀ n : ℕ,
    is_four_digit n →
    n % 9 = 0 →
    has_two_even_two_odd n →
    contains_digit n 5 →
    n ≥ 1058 :=
by
  sorry

#eval (1058 : ℕ).digits 10
#eval (1058 : ℕ) % 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_nine_l406_40631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2x_plus_y_l406_40641

theorem min_value_2x_plus_y (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 1 / (x + 1) + 1 / (x + 2 * y) = 1) : 
  (2 * x + y ≥ 1 / 2 + Real.sqrt 3) ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 1) + 1 / (x₀ + 2 * y₀) = 1 ∧ 
    2 * x₀ + y₀ = 1 / 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2x_plus_y_l406_40641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_18_l406_40642

theorem x_plus_y_equals_18 (x y : ℝ) 
  (h1 : (3 : ℝ)^x = (27 : ℝ)^(y+2)) 
  (h2 : (16 : ℝ)^y = (4 : ℝ)^(x-6)) : 
  x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_18_l406_40642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l406_40646

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + 1
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi/3)

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  g (A/2) = 1 →
  a = 2 →
  b + c = 4 →
  0 < A ∧ A < Real.pi →
  (∃ S : ℝ, S = (1/2) * b * c * Real.sin A ∧ S = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l406_40646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_bounds_l406_40683

noncomputable def y (x : ℝ) : ℝ := (x - 2) * abs x

noncomputable def min_value (a : ℝ) : ℝ :=
  if 1 ≤ a ∧ a ≤ 2 then a^2 - 2*a
  else if 1 - Real.sqrt 2 ≤ a ∧ a < 1 then -1
  else -a^2 + 2*a

theorem y_bounds (a : ℝ) (h : a ≤ 2) :
  (∀ x ∈ Set.Icc a 2, y x ≤ 0) ∧
  (∀ x ∈ Set.Icc a 2, min_value a ≤ y x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_bounds_l406_40683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_to_exponential_equation_l406_40688

theorem two_solutions_to_exponential_equation :
  ∃! (s : Set ℝ), (∀ x ∈ s, (2 : ℝ)^(2*x^2 - 7*x + 5) = 1) ∧ s.ncard = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_to_exponential_equation_l406_40688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_sum_is_three_fifths_l406_40676

def S : Finset ℕ := {1, 2, 3, 4, 5}

def is_odd (n : ℕ) : Prop := n % 2 = 1

def distinct_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  (S.product S).filter (λ p => p.1 < p.2)

def odd_sum_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  (distinct_pairs S).filter (λ p => (p.1 + p.2) % 2 = 1)

theorem probability_odd_sum_is_three_fifths :
  (odd_sum_pairs S).card / (distinct_pairs S).card = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_sum_is_three_fifths_l406_40676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_example_l406_40644

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 4 (π/3) (π/6) = (1, Real.sqrt 3, 2 * Real.sqrt 3) := by
  unfold spherical_to_rectangular
  simp [Real.sin, Real.cos]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_example_l406_40644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_balanced_l406_40615

def balanced (S : Finset ℕ) (n : ℕ) : Prop :=
  ∀ a ∈ S, ∃ b ∈ S, b ≠ a ∧ (a + b) / 2 ∈ S

theorem subset_balanced (k : ℕ) (h_k : k > 1) (n : ℕ) (h_n : n = 2^k) (S : Finset ℕ) 
    (h_S : S ⊆ Finset.range n) (h_card : S.card > 3 * n / 4) :
  balanced S n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_balanced_l406_40615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l406_40618

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

-- Theorem statement
theorem f_properties :
  (f (f 3) = -2) ∧
  (∀ x y : ℝ, x ≠ 1 ∧ y ≠ 1 → f (1 + (1 - x)) = f (1 + (y - 1)) → x = y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l406_40618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_dist_properties_l406_40604

variable (ξ : ℝ → ℝ) (μ σ : ℝ)

-- ξ follows a normal distribution with mean μ and standard deviation σ
def normal_dist (ξ : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  ∀ x, ξ x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

-- Cumulative distribution function
noncomputable def cdf (ξ : ℝ → ℝ) (x : ℝ) : ℝ := ∫ y in Set.Iic x, ξ y

-- Probability of ξ being greater than x
noncomputable def prob_gt (ξ : ℝ → ℝ) (x : ℝ) : ℝ := 1 - cdf ξ x

-- Probability of ξ being less than x
noncomputable def prob_lt (ξ : ℝ → ℝ) (x : ℝ) : ℝ := cdf ξ x

theorem normal_dist_properties
  (h_normal : normal_dist ξ μ σ)
  (h_σ_pos : σ > 0) :
  (∀ a : ℝ, prob_gt ξ (a + 1) > prob_gt ξ (a + 2)) ∧
  (prob_lt ξ μ = 1/2) ∧
  (prob_gt ξ (μ + 1) = prob_lt ξ (μ - 1)) ∧
  (∀ a : ℝ, prob_lt ξ (a + 3) - prob_lt ξ (a - 1) > prob_lt ξ (a + 4) - prob_lt ξ a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_dist_properties_l406_40604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_twenty_l406_40672

theorem divisible_by_twenty (A : Finset ℕ) 
  (h1 : A.card = 7)
  (h2 : ∀ x ∈ A, x ≤ 20)
  (h3 : ∀ x ∈ A, x > 0) :
  ∃ a b c d, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  20 ∣ (a + b - c - d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_twenty_l406_40672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l406_40655

/-- Represents a geometric sequence -/
structure GeometricSequence where
  firstTerm : ℝ
  commonRatio : ℝ

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumOfTerms (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.firstTerm * (1 - g.commonRatio^n) / (1 - g.commonRatio)

theorem geometric_sequence_sum (g : GeometricSequence) :
  sumOfTerms g 3000 = 1000 →
  sumOfTerms g 6000 = 1900 →
  sumOfTerms g 9000 = 2710 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l406_40655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l406_40690

noncomputable section

-- Define the points M and N
def M : ℝ × ℝ := (1, 5/4)
def N : ℝ × ℝ := (-4, -5/4)

-- Define the midpoint of MN
def midpoint_MN : ℝ × ℝ := (-(3/2), 0)

-- Define the slope of the perpendicular bisector
def perp_slope : ℝ := -2

-- Define the curves
def curve1 (x y : ℝ) : Prop := 4*x + 2*y - 1 = 0
def curve2 (x y : ℝ) : Prop := x^2 + y^2 = 3
def curve3 (x y : ℝ) : Prop := x^2/2 + y^2 = 1
def curve4 (x y : ℝ) : Prop := x^2/2 - y^2 = 1

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := y = perp_slope * (x + 3/2)

-- Theorem statement
theorem curves_intersection :
  (∃ x y : ℝ, curve2 x y ∧ perp_bisector x y) ∧
  (∃ x y : ℝ, curve3 x y ∧ perp_bisector x y) ∧
  (∃ x y : ℝ, curve4 x y ∧ perp_bisector x y) ∧
  ¬(∃ x y : ℝ, curve1 x y ∧ perp_bisector x y) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l406_40690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l406_40633

theorem cube_surface_area (a : ℝ) (h : 4 * Real.pi * ((Real.sqrt 3 / 2 * a) ^ 2) = 6 * Real.pi) : 
  6 * a ^ 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l406_40633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_center_distance_l406_40647

/-- Two identical clocks with hour hands -/
structure ClockPair where
  /-- Minimum distance between the ends of hour hands -/
  m : ℝ
  /-- Maximum distance between the ends of hour hands -/
  M : ℝ
  /-- m and M are non-negative -/
  m_nonneg : 0 ≤ m
  M_nonneg : 0 ≤ M
  /-- M is greater than or equal to m -/
  M_ge_m : m ≤ M

/-- The distance between the centers of the clocks -/
noncomputable def centerDistance (cp : ClockPair) : ℝ := (cp.M + cp.m) / 2

/-- Theorem: The distance between the centers of the clocks is (M + m) / 2 -/
theorem clock_center_distance (cp : ClockPair) :
  centerDistance cp = (cp.M + cp.m) / 2 := by
  -- Unfold the definition of centerDistance
  unfold centerDistance
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_center_distance_l406_40647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_size_for_table_rotation_l406_40636

/-- The smallest integer dimension that allows a 10' × 12' table to be rotated in a rectangular room -/
def min_room_dimension : ℕ := 16

/-- The length of the table -/
def table_length : ℝ := 10

/-- The width of the table -/
def table_width : ℝ := 12

/-- The diagonal of the table -/
noncomputable def table_diagonal : ℝ := Real.sqrt (table_length ^ 2 + table_width ^ 2)

theorem min_room_size_for_table_rotation :
  (min_room_dimension : ℝ) > table_diagonal ∧
  ∀ n : ℕ, (n : ℝ) > table_diagonal → n ≥ min_room_dimension :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_size_for_table_rotation_l406_40636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l406_40661

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((1 - Complex.I) / (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l406_40661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_intersection_l406_40687

-- Define the circle C1
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the locus curve L
noncomputable def L (a b : ℕ) (x : ℝ) : ℝ := a * x^2 - b * x + (b^2 : ℝ) / (4 * a)

-- Define the line that intersects L
def intersectingLine (a b : ℕ) (x y : ℝ) : Prop :=
  4 * (Real.sqrt 7 - 1) * a * b * x - 4 * a * y + b^2 + a^2 - 6958 * a = 0

theorem locus_and_intersection (a b : ℕ) (h1 : a ≠ 0) (h2 : ∀ x, 2 * a * L a b x + b^2 = 0) 
  (C1 : Circle) (h3 : C1.center.2 = C1.radius) :
  (∀ x, x ≠ b / (2 * a) → C1.center.2 = L a b x) ∧ 
  (∃! p : ℝ × ℝ, p.2 = L a b p.1 ∧ intersectingLine a b p.1 p.2) →
  ((a = 6272 ∧ b = 784) ∨ (a = 686 ∧ b = 784)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_intersection_l406_40687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koschei_containment_l406_40699

/-- Represents the side of the corridor a guard is leaning on -/
inductive Side
| West
| East

/-- Represents the position of Koschei in the corridor -/
inductive Position
| Room1
| Room2
| Room3
| Room4

/-- Represents the state of the guards in the corridor -/
structure GuardState where
  guard1 : Side
  guard2 : Side
  guard3 : Side

/-- Represents a single move of Koschei between adjacent rooms -/
inductive Move
| North
| South

/-- Updates the guard state based on Koschei's move -/
def updateGuardState (state : GuardState) (move : Move) (pos : Position) : GuardState :=
  sorry

/-- Checks if all guards are on the same side -/
def allOnSameSide (state : GuardState) : Bool :=
  sorry

/-- Theorem stating that there exists an initial configuration that prevents Koschei's escape -/
theorem koschei_containment :
  ∃ (initialState : GuardState) (initialPos : Position),
    ∀ (moves : List Move),
      let finalState := moves.foldl (fun s m => updateGuardState s m (sorry : Position)) initialState
      ¬(allOnSameSide finalState) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_koschei_containment_l406_40699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2007_of_2007_eq_145_l406_40673

/-- Sum of squares of digits of a positive integer -/
def f (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d => d * d) |>.sum

/-- Recursive definition of fₖ -/
def f_k : ℕ → ℕ → ℕ
  | 0, n => n  -- Base case for k = 0
  | 1, n => f n
  | k + 1, n => f (f_k k n)

/-- Main theorem -/
theorem f_2007_of_2007_eq_145 : f_k 2007 2007 = 145 := by
  sorry

#eval f_k 2007 2007  -- For testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2007_of_2007_eq_145_l406_40673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_property_l406_40600

/-- Curve C₁ in parametric form -/
noncomputable def C₁ (a b : ℝ) (φ : ℝ) : ℝ × ℝ :=
  (a * Real.cos φ, b * Real.sin φ)

/-- Curve C₂ in polar form -/
noncomputable def C₂ (θ : ℝ) : ℝ :=
  2 * Real.cos θ

/-- Point on C₁ in polar coordinates -/
noncomputable def point_on_C₁ (a b : ℝ) (θ : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (a^2 * (Real.cos θ)^2 + b^2 * (Real.sin θ)^2)
  (ρ, θ)

theorem curve_property (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b)
    (h₄ : C₁ a b (π/3) = (2, Real.sqrt 3))
    (h₅ : point_on_C₁ a b (π/4) = (Real.sqrt 2, π/4)) :
    ∀ θ : ℝ,
    let (ρ₁, _) := point_on_C₁ a b θ
    let (ρ₂, _) := point_on_C₁ a b (θ + π/2)
    1 / ρ₁^2 + 1 / ρ₂^2 = 5/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_property_l406_40600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_c_intersection_theorem_l406_40637

/-- Curve C is defined by the property that the difference between the distance 
    from any point on C to F(0, 1) and its distance to the x-axis is always 1 -/
def CurveC (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y - 1)^2) - abs y = 1

/-- Line y = kx + m intersects curve C at points A and B -/
def Intersect (k m : ℝ) (A B : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  CurveC x₁ y₁ ∧ CurveC x₂ y₂ ∧ 
  y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m

/-- Vector product FA · FB -/
def DotProduct (A B : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  x₁ * x₂ + (y₁ - 1) * (y₂ - 1)

theorem curve_c_intersection_theorem (m : ℝ) :
  (m > 0) →
  (∀ k : ℝ, ∃ A B : ℝ × ℝ, Intersect k m A B ∧ DotProduct A B < 0) →
  3 - 2 * Real.sqrt 2 < m ∧ m < 3 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_c_intersection_theorem_l406_40637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_girls_l406_40666

def club_size : ℕ := 12
def num_girls : ℕ := 6
def num_boys : ℕ := 6
def chosen_members : ℕ := 3

theorem probability_all_girls :
  (Nat.choose num_girls chosen_members : ℚ) / (Nat.choose club_size chosen_members : ℚ) = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_girls_l406_40666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l406_40653

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

theorem polynomial_characterization (P : IntPolynomial) :
  (∀ n ≥ 2016, (P n : ℝ) > 0 ∧ S (P n).toNat = P (S n))
  →
  (∃ c ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ), ∀ n, P n = c)
  ∨
  (∀ n, P n = n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l406_40653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_interval_l406_40696

-- Define the function f(x) = x - ln x
noncomputable def f (x : ℝ) : ℝ := x - Real.log x

-- State the theorem
theorem f_monotone_decreasing_on_interval :
  StrictMonoOn f (Set.Ioo 0 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_interval_l406_40696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_books_from_different_genres_l406_40665

theorem two_books_from_different_genres 
  (mystery_books : Finset Book) 
  (fantasy_books : Finset Book) 
  (biography_books : Finset Book) 
  (h1 : mystery_books.card = 4)
  (h2 : fantasy_books.card = 4)
  (h3 : biography_books.card = 4)
  (h4 : Disjoint mystery_books fantasy_books)
  (h5 : Disjoint mystery_books biography_books)
  (h6 : Disjoint fantasy_books biography_books) :
  (mystery_books.card * fantasy_books.card + 
   mystery_books.card * biography_books.card + 
   fantasy_books.card * biography_books.card) = 48 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_books_from_different_genres_l406_40665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_condition_l406_40606

theorem tan_one_condition (x : ℝ) : 
  (∃ k : ℤ, x = k * Real.pi + Real.pi / 4) ↔ Real.tan x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_condition_l406_40606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l406_40607

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and focal distance c,
    where a > b > c > 0, and a circle with center at the right focus and radius b-c,
    if the minimum distance from any point on the ellipse to a tangent point on the circle
    is (√3/2)(a-c), then the eccentricity e of the ellipse satisfies 3/5 ≤ e < √2/2. -/
theorem ellipse_eccentricity_range (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hab : a > b) (hbc : b > c) 
    (h_min_dist : ∀ P : ℝ × ℝ, P.1^2 / a^2 + P.2^2 / b^2 = 1 →
      ∃ T : ℝ × ℝ, (T.1 - c)^2 + T.2^2 = (b - c)^2 ∧
      (P.1 - T.1)^2 + (P.2 - T.2)^2 ≥ (3/4) * (a - c)^2) :
    let e := c / a
    3/5 ≤ e ∧ e < Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l406_40607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l406_40605

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 2*a*x else (2*a - 1)*x - 3*a + 6

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc 1 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l406_40605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l406_40634

theorem min_value_theorem (x y : ℝ) (h : Real.log x + Real.log y = Real.log 10) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ (a b : ℝ), Real.log a + Real.log b = Real.log 10 → (2 / a + 5 / b ≥ min_val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l406_40634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_underwater_archaeology_oxygen_consumption_l406_40621

theorem underwater_archaeology_oxygen_consumption 
  (x : ℝ) 
  (h_x_range : x ∈ Set.Icc 6 12) :
  let y := x / 2 + 32 / x + 8
  16 ≤ y ∧ y ≤ 50 / 3 := by
  sorry

#check underwater_archaeology_oxygen_consumption

end NUMINAMATH_CALUDE_ERRORFEEDBACK_underwater_archaeology_oxygen_consumption_l406_40621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_sum_l406_40698

noncomputable def ellipse (a₁ b₁ : ℝ) (x y : ℝ) : Prop := x^2 / a₁^2 + y^2 / b₁^2 = 1

noncomputable def hyperbola (a₂ b₂ : ℝ) (x y : ℝ) : Prop := x^2 / a₂^2 - y^2 / b₂^2 = 1

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem min_eccentricity_sum 
  (a₁ b₁ a₂ b₂ c : ℝ) 
  (h₁ : a₁ > 0) (h₂ : b₁ > 0) (h₃ : a₂ > 0) (h₄ : b₂ > 0)
  (h₅ : ∃ x y, ellipse a₁ b₁ x y ∧ hyperbola a₂ b₂ x y)
  (h₆ : a₁^2 + a₂^2 = 2*c^2)
  (e₁ : ℝ) (h₇ : e₁ = eccentricity a₁ c)
  (e₂ : ℝ) (h₈ : e₂ = eccentricity a₂ c) :
  4 * e₁^2 + e₂^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_sum_l406_40698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_area_l406_40649

-- Define the tetrahedron P-ABC
structure Tetrahedron where
  PA : ℝ
  PB : ℝ
  PC : ℝ

-- Define the condition for maximized sum of side face areas
def is_max_side_faces_area (t : Tetrahedron) : Prop :=
  t.PA * t.PB + t.PA * t.PC + t.PB * t.PC = t.PA^2 + t.PB^2 + t.PC^2

-- Define the surface area of the sphere passing through P, A, B, C
noncomputable def sphere_surface_area (t : Tetrahedron) : ℝ :=
  4 * Real.pi * (t.PA^2 + t.PB^2 + t.PC^2) / 4

-- Theorem statement
theorem tetrahedron_sphere_area :
  ∀ t : Tetrahedron,
    t.PA = 2 ∧ t.PB = Real.sqrt 6 ∧ t.PC = Real.sqrt 6 ∧
    is_max_side_faces_area t →
    sphere_surface_area t = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_area_l406_40649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l406_40608

/-- Represents a hyperbola with equation x²/9 - y²/m = 1 -/
structure Hyperbola where
  m : ℝ

/-- Represents a circle with equation x² + y² - 4x - 5 = 0 -/
def circleEq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 5 = 0

/-- A focus of a hyperbola -/
structure Focus (h : Hyperbola) where
  x : ℝ
  y : ℝ

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = (4/3)*x ∨ y = -(4/3)*x}

/-- The theorem stating that if one focus of the hyperbola lies on the circle,
    then the asymptotes are y = ±(4/3)x -/
theorem hyperbola_asymptotes (h : Hyperbola) (f : Focus h) :
  circleEq f.x f.y → asymptotes h = {(x, y) | y = (4/3)*x ∨ y = -(4/3)*x} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l406_40608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_tax_percentage_l406_40656

/-- The percentage of cultivated land that is taxed -/
noncomputable def taxed_percentage (total_tax village_tax willam_tax willam_land_percentage : ℝ) : ℝ :=
  willam_tax / (willam_land_percentage * village_tax)

/-- Theorem stating that the percentage of cultivated land taxed is 50% -/
theorem farm_tax_percentage 
  (total_tax : ℝ) 
  (willam_tax : ℝ) 
  (willam_land_percentage : ℝ) 
  (h1 : total_tax = 3840)
  (h2 : willam_tax = 480)
  (h3 : willam_land_percentage = 0.25) :
  taxed_percentage total_tax total_tax willam_tax willam_land_percentage = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_tax_percentage_l406_40656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_equals_two_l406_40613

theorem det_B_equals_two (x y : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, 2; -3, y]
  (B + B⁻¹ = 0) → Matrix.det B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_equals_two_l406_40613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_theorem_l406_40659

-- Define the line ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the inclination angle
noncomputable def inclinationAngle (l : Line) : ℝ := Real.arctan (-l.a / l.b)

-- State the theorem
theorem line_inclination_theorem (l : Line) (θ : ℝ) 
  (h : inclinationAngle l = θ) 
  (h_condition : Real.sin θ + Real.cos θ = 0) : 
  l.a - l.b = 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_theorem_l406_40659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_difference_l406_40663

def a : ℕ → ℚ
  | 0 => 1/7
  | n+1 => 7/2 * a n * (1 - a n)

theorem a_difference : a 1412 - a 1313 = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_difference_l406_40663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l406_40695

-- Define P, Q, and R
noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

-- Theorem statement
theorem log_inequality : R < Q ∧ Q < P := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l406_40695
